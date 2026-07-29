//! Scanout paint: guest surface (mapping or EFI FB) → host BGRA8 row buffer.
//!
//! C owns the QEMU DisplaySurface; Rust fills it (apple-gfx's
//! encodeCurrentFrame / getBytes role). Page-table paint uses
//! [`crate::contract::iosurface_pages`] when the mapping has entries;
//! otherwise we fall back to the programmed EFI framebuffer or clear.
//!
//! Early-boot / present policy (archive apple-pv-gpu + live Monterey + PGDisplay):
//! - Front formats: archive prefers type-11 **RGBA16Float** (0x73); live Monterey
//!   boot logo/progress also stores full-screen type-11 **BGRA8** / **RGBA8**
//!   before the first DisplaySwap — paint those formats too pre-boundary.
//! - Geometry barrier (archive same_geom): first early paint establishes console
//!   size from the **guest surface** (mapper geom / job size); later pre-boundary
//!   paints only when the job matches that size. Never invent or clamp dimensions
//!   (Apple `modeChangeHandler` sizeInPixels = presented surface size).
//! - After DisplaySwap (`frame_flush_seen`): writebacks do **not** rename the
//!   presented surface (`present_mapping` stays the last CmdDisplaySwap mid).
//!   Paint is present-boundary only (PGDisplay newFrame / hostPresentCount).
//! - At CmdDisplaySwap the host **retains** the named mapping after wait_surface
//!   drains (`capture_present_frame` = PGDisplay presentFrame → +0x188), then
//!   HostAction paint blits that snapshot. Later scanout / gfx_update re-shows
//!   it (`hostPresentCount`). Freeze is at present (before stamp completion
//!   lets the guest recycle the mid), not deferred to BH after stamp.

use crate::contract::pixel_format::{
    self, convert_rgba8_to_row, convert_row_to_rgba8, MTL_FORMAT_BGRA8_UNORM,
    MTL_FORMAT_RGBA16_FLOAT, MTL_FORMAT_RGBA8_UNORM, RGBA8_BPP,
};
use crate::model::{DeviceState, EFI_BOOT_HEIGHT, EFI_BOOT_WIDTH, MAX_SCANOUT_DIM};
use crate::runtime::host::HostMemory;

/// Type-11 color formats that may be the compositor front before DisplaySwap.
///
/// Archive `front_buffer` is RGBA16Float only; live Monterey also draws the
/// early boot logo into BGRA8/RGBA8 type-11 full-frame targets. Not a size list.
#[inline]
fn is_front_buffer_format(fmt: u16) -> bool {
    matches!(
        fmt,
        MTL_FORMAT_RGBA16_FLOAT | MTL_FORMAT_BGRA8_UNORM | MTL_FORMAT_RGBA8_UNORM
    )
}

/// Result of a scanout copy attempt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScanoutCopyResult {
    /// Pixels written (or black clear).
    Painted,
    /// Content generation matches last paint — C should skip surface update.
    Unchanged,
    /// Hard failure (bad args).
    Failed,
}

#[cfg(feature = "backend-vulkan")]
/// Read mapping pages into `dst` without updating present/paint generation.
///
/// Used by draw bind materialization (sampled type-11 textures). Returns true
/// when geometry and page table produced a full image.
pub fn read_mapping_bgra8<M: HostMemory + crate::runtime::host::HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mapping_id: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
) -> bool {
    if width == 0
        || height == 0
        || width > MAX_SCANOUT_DIM
        || height > MAX_SCANOUT_DIM
        || dst_stride < width.saturating_mul(RGBA8_BPP)
    {
        return false;
    }
    let need = (height as u64).saturating_mul(dst_stride as u64) as usize;
    if dst.len() < need {
        return false;
    }
    let _ = crate::runtime::mapper::ensure_resolved_for_scanout(state, host, mapping_id);
    paint_mapping(state, host, mapping_id, dst, dst_stride, width, height)
}

/// Always-on census of the capture readback-elision ratio (never silent):
/// `full` = readback + proxy scan ran; `light` = the window is carrying the frame
/// from the engine resident, readback skipped. Deduped to one line per 1024 total
/// captures (a line every ~8 s at 120 Hz), so it confirms the elision engages
/// without flooding the fail view.
fn maybe_log_capture_sampling(state: &DeviceState) {
    let full = state.present.full_captures;
    let light = state.present.light_captures;
    let total = full.wrapping_add(light);
    if total != 0 && total.is_multiple_of(1024) {
        crate::observe::off(format!("capture_sampling full={full} light={light}"));
    }
}

/// Fill `buf` from the mapping's GPU resident, without any guest-page scatter.
///
/// Returns whether the resident supplied the whole frame. On `true` `buf` holds
/// tight BGRA8 and `last_paint_src` is [`crate::model::PaintSrc::Resident`]; on
/// `false` `buf` is untouched and the caller takes the guest-page path. A miss
/// is an expected steady-state condition (cold mid / no resident yet), so it is
/// counted in the `capture_source` census rather than logged per present.
#[cfg(feature = "backend-vulkan")]
fn try_capture_from_resident(
    state: &mut crate::model::DeviceState,
    buf: &mut Vec<u8>,
    mapping_id: u32,
    width: u32,
    height: u32,
) -> bool {
    let need = buf.len();
    let identity =
        crate::runtime::present_identity::surface_identity(state, mapping_id, width, height);
    let Some(bgra) = crate::backend::vulkan::engine::read_resident_bgra(&identity, need) else {
        return false;
    };
    debug_assert_eq!(bgra.len(), need);
    // Move (not copy) the readback in; the untouched scratch returns to the pool.
    state.present.capture_scratch = std::mem::replace(buf, bgra);
    state.present.last_paint_src = crate::model::PaintSrc::Resident;
    true
}

/// Non-Vulkan backends have no resident registry; capture stays on guest pages.
#[cfg(not(feature = "backend-vulkan"))]
fn try_capture_from_resident(
    _state: &mut crate::model::DeviceState,
    _buf: &mut [u8],
    _mapping_id: u32,
    _width: u32,
    _height: u32,
) -> bool {
    false
}

/// Snapshot the named mapping into the stable present frame.
///
/// PGDisplay retains the presented surface at present (`+0x188`); encode /
/// re-show use that retain. Product freezes the finished surface at
/// CmdDisplaySwap after wait_surface drains — before the packet stamp lets the
/// guest recycle the mid (BH-deferred freeze captured mid-recycle partials).
///
/// Guest pages ARE the surface content — the draw path's CPU writeback lands
/// Stores in them. There is exactly one capture source.
/// Takes no `HostOps`: with the guest-page capture path gone this reads the GPU
/// resident and the host surface cache only — it never touches guest memory.
pub fn capture_present_frame(
    state: &mut DeviceState,
    mapping_id: u32,
    width: u32,
    height: u32,
    generation: u32,
) -> bool {
    // Test isolation: exclude proxy-sequence assertions running in parallel.
    #[cfg(test)]
    let _proxy_shared = crate::runtime::census::present_proxy::test_shared();
    if mapping_id == 0
        || width == 0
        || height == 0
        || width > MAX_SCANOUT_DIM
        || height > MAX_SCANOUT_DIM
    {
        return false;
    }
    let stride = width.saturating_mul(RGBA8_BPP);
    let need = (height as u64).saturating_mul(stride as u64) as usize;
    if need == 0 {
        return false;
    }
    state.advance_present_epoch();
    // --- Capture readback elision ---
    // When the previous present's window publish handed the window an engine
    // resident (`display_from_resident` — the macOS engine-swapchain handoff), the
    // display reads that resident directly and does NOT consume this CPU capture.
    // The ~8-12 ms guest-page gather + full-frame proxy scan below is then pure
    // present-hot-path overhead that serializes the guest behind the drain lock
    // (the fullscreen-video slowdown class). Skip it on those presents; the cheap
    // protocol-structural a/b guard still runs on every light present.
    //
    // Never taken where the window owns its swapchain and uploads CPU pixels —
    // every non-macOS host — because `display_from_resident` only becomes true
    // after a resident publish succeeds.
    //
    // The full-frame readback has EXACTLY ONE reason to exist: the DISPLAY needs
    // CPU pixels because no resident is carrying the frame, and the window will
    // blit `frame_bgra`. Nothing else reads it.
    //
    // Consequence: with a resident carrying, `frame_bgra` holds no frame for this
    // present, and the branch below drops it so that stays literally true.
    let display_needs_cpu_frame = !state.present.display_from_resident;
    if !display_needs_cpu_frame {
        // Publish the new resident and leave `frame_bgra` empty: the window
        // ignores CPU pixels while the resident carries the display. A publish
        // miss costs one dropped frame (the window holds its last good frame and
        // publish logs the drop), then `display_from_resident=false` forces the
        // next capture to read back for fallback.
        //
        // Dropping it is what makes "no CPU pixels for this present" a fact
        // rather than an inference. Skipping the readback only leaves the buffer
        // empty if it was already empty, and it is not: the first present of a
        // boot runs the full path — before the guest has painted anything, so it
        // captures black — and every light present after it retained those bytes.
        // Everything downstream that asks "were there pixels" asks
        // `frame_bgra.is_empty()`, so all of them read that one stale frame as
        // the current one. `present_content_verdict` judged it Black on 481 of
        // 481 presents of a boot whose screen was correct throughout (0
        // `present_content`, 0 `present_content_unsampled`), sending
        // `present_black_retain` to the always-on failure sink 481 times with no
        // guest work lost — the wolf-cry the `Unsampled` verdict exists to
        // prevent, reintroduced through a buffer it could not see was stale. The
        // console blit is gated on the same emptiness and would have painted
        // those bytes as the live frame.
        state.present.frame_bgra.clear();
        state.present.frame_mapping = mapping_id;
        state.present.frame_width = width;
        state.present.frame_height = height;
        state.present.frame_generation = generation;
        state.present.frame_valid = true;
        // First host paint after a present blits +0x188 (mirror the full path).
        state.present.frame_encode_pending = true;
        // The present declares this mapping's pages the finished frame; the first
        // LOAD draw after must re-seed from guest pages (dual-mid strobe class).
        state.presented_needs_guest_seed.insert(mapping_id);
        state.present.light_captures = state.present.light_captures.wrapping_add(1);
        maybe_log_capture_sampling(state);
        return true;
    }
    state.present.full_captures = state.present.full_captures.wrapping_add(1);
    maybe_log_capture_sampling(state);
    // Attribute this capture's lock hold to the tranche `capture_us` bucket (it
    // runs on the present drain, not a render draw). Every real return below
    // notes the elapsed time so a capture-bound hitch stops hiding in `other_us`.
    // Recycle the warm double-buffered scratch instead of a fresh `vec![0u8;
    // need]` per present (which zeroes 8 MiB and faults fresh anon pages every
    // time, only to overwrite them). `resize` is a no-op at steady geometry;
    // every byte in `[0, need)` is fully written below (host_cache
    // `copy_from_slice`, `paint_mapping` row fill, or the reuse-store copy), so
    // no pre-zero is needed. On failure `buf` returns to `capture_scratch`
    // unchanged, leaving the prior `frame_bgra` retain intact (keep-prior).
    let mut buf = std::mem::take(&mut state.present.capture_scratch);
    buf.clear();
    buf.resize(need, 0);
    // Prefer host render-cache when encode/clear wrote it (Linux discrete GPU
    // path — kb tahoe-x86-host-reims_vgpu §8.5). Fall back to guest type-4 pages.
    let from_host_cache = if let Some(cached) =
        crate::runtime::surface_cache::get(state, mapping_id, width, height)
    {
        buf.copy_from_slice(cached);
        true
    } else {
        false
    };
    // Resident-direct capture — the ONLY GPU-content capture source.
    //
    // The proxies need the finished frame's BYTES; they do not need those bytes
    // to be in guest pages. This reads the resident and nothing else: no
    // `flush_intersecting`. Nothing is owed — a type-11 render Store lands its
    // own guest-page writeback (`mapping_write::write_rgba8_image_changed`), and
    // the deferred rails that remain (compute storage, linear, GVA) are keyed on
    // resources this capture does not touch and flush on a genuine guest read
    // (LOAD re-seed / SynchronizeResources / guest CPU read). The retained
    // `frame_bgra` filled here is unchanged, so the present-boundary seed (which
    // reads the retained front frame first, guest pages only as fallback) is
    // unaffected.
    //
    // There is deliberately NO guest-page capture fallback. A capture that
    // predated this read the same resident and then scattered it into the
    // fragmented guest pages purely to read it back out — a second, parallel
    // implementation of "get the present frame" that cost a full-frame writeback
    // per sampled present. Keeping it would mean maintaining two veins of the
    // same operation, so a missing resident now fails VISIBLY (keep_prior + the
    // `capture_fail` proxy) instead of silently diverging onto another path.
    // Live evidence for the delete: `capture_source resident=51 guest=0` across a
    // full boot (pre-convergence included), zero `present_capture FAIL`.
    //
    // Consequence for the non-Vulkan backends: they have no resident registry, so
    // capture fails there and the console holds its prior retain. That is the
    // known arm/Metal breakage this pathway already carries.
    if !from_host_cache && !try_capture_from_resident(state, &mut buf, mapping_id, width, height) {
        crate::observe::off(format!(
            "present_capture FAIL mid={mapping_id} {width}x{height} gen={generation} \
             reason=no_resident_content present_mapping={} frame_mapping={}",
            state.present.present_mapping, state.present.frame_mapping
        ));
        // Recycle the untouched scratch; the prior retain stays intact.
        state.present.capture_scratch = buf;
        return false;
    }
    let from_last_store = from_host_cache;
    // Accurate present-capture provenance. `from_host_cache` reports the type-4
    // surface_cache hit; when it misses, `paint_mapping` records which of its
    // sub-paths actually filled the frame so the `paint_us` cost is attributable
    // (a deferred-flush reuse is cheap; a cold fragmented read is the ~12 ms path).
    let src = if from_host_cache {
        "host_cache"
    } else {
        match state.present.last_paint_src {
            crate::model::PaintSrc::Resident => "resident",
            crate::model::PaintSrc::ReuseStore => "reuse_store",
            crate::model::PaintSrc::GuestPagesContig => "guest_pages_contig",
            crate::model::PaintSrc::GuestPagesFragmented => "guest_pages_frag",
            crate::model::PaintSrc::None => "guest_pages",
        }
    };
    // The occupancy scan and the three lines it fed are diagnostic: each is an
    // O(w*h) walk of the just-captured 8 MiB frame on the present drain, under
    // the device lock, and the `peers` field walks every same-geometry host
    // surface on top of that. The always-on alarm for a black console is
    // `present_black`, which does its own scan at the drain boundary where the
    // verdict is acted on.
    if crate::observe::draw_log_enabled() {
        let (nz, maxb, rgb_nz, max_rgb, px0) = crate::observe::bgra_present_stats(&buf);
        let mut peers = String::new();
        for (&mid, e) in state.host_surfaces.iter() {
            if mid == mapping_id || e.width != width || e.height != height || e.bgra.is_empty() {
                continue;
            }
            let (pnz, pmax, _) = crate::observe::bgra_rgb_stats(&e.bgra);
            if pmax > 0 && pnz > 10_000 {
                if !peers.is_empty() {
                    peers.push(',');
                }
                peers.push_str(&format!(
                    "mid{mid}:rgb_nz={pnz}:max_rgb={pmax}:hgen={}",
                    e.host_gen
                ));
            }
        }
        crate::observe::line(format!(
            "present_capture mid={mapping_id} {width}x{height} gen={generation} src={src} last_store={} host_cache={} rgb_nz={rgb_nz} max_rgb={max_rgb} byte_nz={nz} byte_max={maxb} px0=[{},{},{},{}] present_mapping={} frame_mapping={} frame_flush={} peers=[{peers}]",
            from_last_store as u8,
            from_host_cache as u8,
            px0[0],
            px0[1],
            px0[2],
            px0[3],
            state.present.present_mapping,
            state.present.frame_mapping,
            state.present.frame_flush_seen as u8,
        ));
    }
    // Publish the new frame and recycle the old retain buffer as the next
    // capture scratch (warm 8 MiB alloc, no per-present malloc/free/zero).
    let old_frame = std::mem::replace(&mut state.present.frame_bgra, buf);
    state.present.capture_scratch = old_frame;
    state.present.frame_mapping = mapping_id;
    state.present.frame_width = width;
    state.present.frame_height = height;
    state.present.frame_generation = generation;
    state.present.frame_valid = true;
    // The present declares this mapping's guest pages the finished frame; the
    // guest may CPU-write them next (inter-buffer damage forward-copy). The
    // first LOAD draw after this present must re-seed from guest pages
    // instead of chaining the resident (dual-mid strobe class).
    state.presented_needs_guest_seed.insert(mapping_id);
    // Force the next host paint to blit +0x188. Early pre-boundary paints may
    // have latched painted_mapping/generation (live type-11 paint_mapping or
    // paint_efi) to the same mid+gen; with encode_pending=false that made
    // copy_to_bgra8 return Unchanged and left the QEMU console on frozen EFI
    // while +0x188 held logo+pill (live serial-20260715-054015:
    // present_capture rgb_nz≈6k then present_paint Unchanged only).
    state.present.frame_encode_pending = true;
    true
}

/// Blit tight BGRA8 `src` into `dst` (tight or strided).
fn blit_bgra_buffer(src: &[u8], dst: &mut [u8], dst_stride: u32, width: u32, height: u32) -> bool {
    let src_stride = width.saturating_mul(RGBA8_BPP) as usize;
    if src.len() < src_stride.saturating_mul(height as usize) {
        return false;
    }
    for y in 0..height as usize {
        let so = y * src_stride;
        let doff = y * (dst_stride as usize);
        let n = src_stride.min(dst_stride as usize);
        dst[doff..doff + n].copy_from_slice(&src[so..so + n]);
    }
    true
}

/// Blit the stable present snapshot into `dst` (tight or strided BGRA8).
///
/// presentFrame freezes +0x188 at swap time; post-stamp guest writes must not
/// change the retain (archive encodeCurrentFrame / hostPresentCount re-show).
/// Mid-writeback is_front Stores must **not** recapture +0x188 (archive:
/// post-boundary front writebacks do not paint — tile-through thrash).
fn blit_present_snapshot(
    state: &DeviceState,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
) -> bool {
    blit_bgra_buffer(&state.present.frame_bgra, dst, dst_stride, width, height)
}

/// Fill `dst` (BGRA8, `dst_stride` bytes/row) for the named mapping.
///
/// `expected_generation` is from the HostAction (0 = always paint).
/// After DisplaySwap, the first copy **encodes** the stable snapshot from live
/// pages (host paint time); later copies re-show that snapshot
/// (`hostPresentCount`) without re-reading guest pages.
///
/// This is now the only console paint. `copy_to_host_ptr_gpu` used to get first
/// refusal on a QEMU-allocated, alignment-negotiated display buffer: the engine
/// imported it and recorded a resident→buffer GPU copy, so no framebuffer bytes
/// crossed the CPU. It went out with the host-pointer import that made it
/// possible — the mechanism is the same one that can address guest RAM, and it
/// is not requested any more, whichever allocation is on the other end.
#[allow(
    clippy::too_many_arguments,
    reason = "the scanout copy API mirrors its destination and present geometry"
)]
pub fn copy_to_bgra8<M: HostMemory + crate::runtime::host::HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mapping_id: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
    expected_generation: u32,
) -> ScanoutCopyResult {
    if width == 0
        || height == 0
        || width > MAX_SCANOUT_DIM
        || height > MAX_SCANOUT_DIM
        || dst_stride < width.saturating_mul(RGBA8_BPP)
    {
        return ScanoutCopyResult::Failed;
    }
    let need = (height as u64).saturating_mul(dst_stride as u64) as usize;
    if dst.len() < need {
        return ScanoutCopyResult::Failed;
    }
    // PGDisplay encodeCurrentFrame always re-shows +0x188 when the retain
    // matches paint geom — frozen at presentFrame (present boundary only).
    if state.present.frame_valid
        && state.present.frame_width == width
        && state.present.frame_height == height
        && !state.present.frame_bgra.is_empty()
    {
        if !state.present.frame_encode_pending
            && state.present.painted_mapping == state.present.frame_mapping
            && state.present.painted_generation == state.present.frame_generation
        {
            crate::observe::off(format!(
                "present_paint Unchanged mid={} gen={} (console already holds +0x188)",
                state.present.frame_mapping, state.present.frame_generation
            ));
            return ScanoutCopyResult::Unchanged;
        }
        if blit_present_snapshot(state, dst, dst_stride, width, height) {
            let shown_mid = state.present.frame_mapping;
            let shown_gen = state.present.frame_generation;
            // Reuse the fused scan `capture_present_frame` already ran over this
            // frozen frame instead of two more full 8 MiB passes under the lock.
            // `frame_bgra` has a single writer (capture), so a matching
            // mapping+generation means the stashed stats describe these exact
            // bytes; a mismatch (e.g. a test-injected frame) falls back to a scan.
            // Per-paint census on the QEMU display thread — ~30k lines/session
            // each under a continuously-animating app, plus an O(w·h)
            // `bgra_present_stats` full-frame scan built PURELY to
            // populate the log. Both the scan and the two lines are log-only
            // (nothing below consumes the stats), so gate the whole block behind
            // REIMS_VGPU_DRAW_LOG: a normal boot pays neither the scan nor the flood.
            // The always-on present rate/occupancy signal lives in the
            // present_proxy summary + `present_import`.
            if crate::observe::draw_log_enabled() {
                let (nz, maxb, rgb_nz, max_rgb, px0) =
                    crate::observe::bgra_present_stats(&state.present.frame_bgra);
                crate::observe::line(format!(
                    "scanout paint_snapshot mid={} (action mid={} gen={}) {}x{} retain_gen={} nz={} max={}",
                    shown_mid, mapping_id, expected_generation, width, height, shown_gen, nz, maxb
                ));
                crate::observe::line(format!(
                    "present_paint Painted mid={shown_mid} (action mid={mapping_id} gen={expected_generation}) {width}x{height} rgb_nz={rgb_nz} max_rgb={max_rgb} px0=[{},{},{},{}] (this is what QMP shows)",
                    px0[0], px0[1], px0[2], px0[3]
                ));
            }
            state.present.valid = true;
            state.present.mapping_id = shown_mid;
            state.present.width = width;
            state.present.height = height;
            state.present.generation = shown_gen;
            state.present.painted_mapping = shown_mid;
            state.present.painted_generation = shown_gen;
            // First successful +0x188 blit after capture clears encode pending.
            state.present.frame_encode_pending = false;
            return ScanoutCopyResult::Painted;
        }
    }

    // Present-path after first content boundary: never fall through to live
    // paint_mapping of a clear-only dual-mid (would freeze console black).
    let post_boundary = state.present.frame_flush_seen
        && width == state.present.width
        && height == state.present.height;

    if post_boundary {
        let is_current_present =
            mapping_id == state.present.host_mapping || mapping_id == state.present.present_mapping;

        // Capture failed at DisplaySwap for the still-current present — retry once.
        if is_current_present
            && (state.present.frame_encode_pending || !state.present.frame_valid)
            && (expected_generation == 0 || expected_generation == state.present.generation)
        {
            let gen = if expected_generation != 0 {
                expected_generation
            } else {
                state.present.generation
            };
            let _ = capture_present_frame(state, mapping_id, width, height, gen);
            if state.present.frame_valid
                && state.present.frame_width == width
                && state.present.frame_height == height
                && blit_present_snapshot(state, dst, dst_stride, width, height)
            {
                state.present.painted_mapping = state.present.frame_mapping;
                state.present.painted_generation = state.present.frame_generation;
                state.present.frame_encode_pending = false;
                return ScanoutCopyResult::Painted;
            }
        }

        if is_current_present {
            crate::observe::fail(format!(
                "scanout post_boundary no retain mid={mapping_id} {width}x{height} gen={expected_generation}"
            ));
            return ScanoutCopyResult::Failed;
        }
        return ScanoutCopyResult::Unchanged;
    }

    // Pre-boundary only: live mapping paint (early logo/pill before first RGB retain).
    let _ = crate::runtime::mapper::ensure_resolved_for_scanout(state, host, mapping_id);

    // Only latch painted_generation on a real pixel source. A clear-to-black
    // fallback must not stamp generation — that freezes the console on black
    // forever when the first paint races the mapper (Unchanged on next gen).
    if paint_mapping(state, host, mapping_id, dst, dst_stride, width, height) {
        let need = (height as usize)
            .saturating_mul(width as usize)
            .saturating_mul(4);
        let sample = &dst[..need.min(dst.len())];
        let (nz, maxb) = crate::observe::nonzero_stats(sample);
        crate::observe::line(format!(
            "scanout paint_mapping ok mid={} {}x{} gen={} nz={} max={}",
            mapping_id, width, height, expected_generation, nz, maxb
        ));
        state.present.valid = true;
        state.present.mapping_id = mapping_id;
        state.present.width = width;
        state.present.height = height;
        state.present.generation = expected_generation;
        state.present.painted_mapping = mapping_id;
        state.present.painted_generation = expected_generation;
        ScanoutCopyResult::Painted
    } else if paint_efi(state, host, dst, dst_stride, width, height) {
        // EFI/BAR1 fallback fills the console for early verbose boot only.
        // Do **not** latch painted_mapping/generation to the product mid —
        // that made post-capture Unchanged skip +0x188 (logo/pill retain)
        // while the console still held EFI text.
        crate::observe::line(format!("scanout paint_efi ok {}x{}", width, height));
        state.present.valid = true;
        state.present.mapping_id = mapping_id;
        state.present.width = width;
        state.present.height = height;
        state.present.generation = expected_generation;
        ScanoutCopyResult::Painted
    } else {
        // Always-on: a total paint failure means a black/stale console. Logging
        // this via the gated `line()` sink made the always-on fail log silently
        // lie about a black screen (scanout audit Rank-3).
        crate::observe::fail(format!(
            "scanout paint FAIL mid={} {}x{} gen={} (console black/stale)",
            mapping_id, width, height, expected_generation
        ));
        ScanoutCopyResult::Failed
    }
}

/// Paint from guest-programmed EFI framebuffer (MMIO 0x1210 + stride 0x1228).
///
/// Used by product scanout fallback and by the pre-boundary host console when
/// the guest relocates the kernel video console off BAR1 into system RAM
/// (live serial: `console relocated to 0xf1000000` while BAR1 freezes).
pub fn paint_efi_console<M: HostMemory>(
    state: &DeviceState,
    host: &M,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
) -> bool {
    let fb = state.gfx.efi_fb_start;
    if fb == 0 {
        return false;
    }
    // Use programmed EFI dims when they match the surface request, else skip.
    let efi_w = EFI_BOOT_WIDTH;
    let efi_h = EFI_BOOT_HEIGHT;
    if width != efi_w || height != efi_h {
        return false;
    }
    let stride = if state.gfx.efi_fb_stride != 0 {
        state.gfx.efi_fb_stride
    } else {
        efi_w.saturating_mul(RGBA8_BPP)
    };
    if stride < efi_w.saturating_mul(RGBA8_BPP) {
        return false;
    }
    let row_bytes = (efi_w as usize) * (RGBA8_BPP as usize);
    for y in 0..efi_h {
        let gpa = fb + (y as u64) * (stride as u64);
        let dst_off = (y as usize) * (dst_stride as usize);
        if dst_off + row_bytes > dst.len() {
            return false;
        }
        if host
            .read_gpa(gpa, &mut dst[dst_off..dst_off + row_bytes])
            .is_err()
        {
            return false;
        }
    }
    true
}

fn paint_efi<M: HostMemory>(
    state: &DeviceState,
    host: &M,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
) -> bool {
    paint_efi_console(state, host, dst, dst_stride, width, height)
}

/// Why a console capture paint produced no pixels.
///
/// Every one of these shows as a black or stale console, so the reason is the
/// whole diagnostic for the "why is it black" class.
///
/// # Why these are prefixed
///
/// Bare, three of them were claimed by another rail: `unmapped` and `short_view`
/// were also `import_present`'s words for different checks, and `no_mapping` was
/// also the type-5 loader's — so `grep reason=unmapped` returned a mix of the
/// capture rail and the import rail and could not be read. The `capture_` prefix
/// is the same fix the slate reasons and the MRT proxies took.
///
/// # Two names became six
///
/// `short_view` stood for one `if` with three `||`-ed conditions — a null host
/// pointer, a view shorter than the sample window, and a base offset past the
/// end of it — which are three different faults with three different fixes.
/// `read_multi_row_oob` and `read_multi_missing` each stood for two sites whose
/// bounds differ (the convert path slices `tight` bytes, the direct path
/// `min(dst_row, tight)`), so they can fire under different conditions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CaptureDecline {
    /// No mapping record for this id.
    NoMapping,
    /// The guest has paged the surface off.
    Unmapped,
    /// Mapped, but the page list is empty — a resolve gap.
    NoPages,
    /// Geometry has never been latched.
    NoGeom,
    /// The mapping's latched geometry is not the console's.
    GeomMismatch { have_w: u32, have_h: u32 },
    /// The pixel format has no known bytes-per-pixel.
    BppUnknown { format: u16 },
    /// The pixel format has no known tight row size.
    TightRowUnknown { format: u16 },
    /// No type-11 sample window could be derived.
    NoSampleWindow,
    /// The descriptor's row stride is narrower than a tight row.
    BprBelowTight { bpr: u64, tight: u32 },
    /// The contig host view resolved to a null pointer.
    ContigViewNull,
    /// The contig host view is shorter than the sample window.
    ContigViewShort { contig_len: u64, span_end: u64 },
    /// The sample window's base is at or past its end, so there is nothing to
    /// read. `contig` says which path found it, because the two reach it
    /// differently and the check is the same.
    BaseBeyondSpan {
        base_off: u64,
        span_end: u64,
        contig: bool,
    },
    /// The fragmented multi-import read of the sample window failed.
    MultiReadFailed { len: usize },
    /// The destination row would run past the end of the console buffer.
    DstOverflow { row: u32 },
    /// Converting path: the requested row lies outside the multi-import buffer.
    ConvertRowOob { row: u32 },
    /// Converting path: neither a contig base nor a multi-import buffer exists.
    ConvertRowMissing { row: u32 },
    /// Converting path: the guest format could not be converted to RGBA8.
    ConvertToRgba { format: u16 },
    /// Converting path: RGBA8 could not be converted back to the console's BGRA8.
    ConvertFromRgba,
    /// Direct-BGRA path: the requested row lies outside the multi-import buffer.
    DirectRowOob { row: u32 },
    /// Direct-BGRA path: neither a contig base nor a multi-import buffer exists.
    DirectRowMissing { row: u32 },
}

impl crate::observe::Decline for CaptureDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::NoMapping => "capture_no_mapping",
            Self::Unmapped => "capture_unmapped",
            Self::NoPages => "capture_no_pages",
            Self::NoGeom => "capture_no_geom",
            Self::GeomMismatch { .. } => "capture_geom_mismatch",
            Self::BppUnknown { .. } => "capture_bpp_unknown",
            Self::TightRowUnknown { .. } => "capture_tight_row_unknown",
            Self::NoSampleWindow => "capture_no_sample_window",
            Self::BprBelowTight { .. } => "capture_bpr_below_tight",
            Self::ContigViewNull => "capture_contig_view_null",
            Self::ContigViewShort { .. } => "capture_contig_view_short",
            Self::BaseBeyondSpan { .. } => "capture_base_beyond_span",
            Self::MultiReadFailed { .. } => "capture_multi_read_failed",
            Self::DstOverflow { .. } => "capture_dst_overflow",
            Self::ConvertRowOob { .. } => "capture_convert_row_oob",
            Self::ConvertRowMissing { .. } => "capture_convert_row_missing",
            Self::ConvertToRgba { .. } => "capture_convert_to_rgba",
            Self::ConvertFromRgba => "capture_convert_from_rgba",
            Self::DirectRowOob { .. } => "capture_direct_row_oob",
            Self::DirectRowMissing { .. } => "capture_direct_row_missing",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::GeomMismatch { have_w, have_h } => vec![("have", format!("{have_w}x{have_h}"))],
            Self::BppUnknown { format }
            | Self::TightRowUnknown { format }
            | Self::ConvertToRgba { format } => vec![("format", format.to_string())],
            Self::BprBelowTight { bpr, tight } => {
                vec![("bpr", bpr.to_string()), ("tight", tight.to_string())]
            }
            Self::ContigViewShort {
                contig_len,
                span_end,
            } => vec![
                ("contig_len", contig_len.to_string()),
                ("span_end", span_end.to_string()),
            ],
            Self::BaseBeyondSpan {
                base_off,
                span_end,
                contig,
            } => vec![
                ("base_off", base_off.to_string()),
                ("span_end", span_end.to_string()),
                ("contig", u8::from(*contig).to_string()),
            ],
            Self::MultiReadFailed { len } => vec![("len", len.to_string())],
            Self::DstOverflow { row }
            | Self::ConvertRowOob { row }
            | Self::ConvertRowMissing { row }
            | Self::DirectRowOob { row }
            | Self::DirectRowMissing { row } => vec![("row", row.to_string())],
            _ => Vec::new(),
        }
    }
}

fn paint_mapping<M: HostMemory + crate::runtime::host::HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mapping_id: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
) -> bool {
    use crate::runtime::mapping_write::type11_sample_window;

    // Every `false` return here shows as a black/stale console; log the specific
    // reason so the "why is it black" class is diagnosable (scanout audit Rank-3).
    // Each site exits the function, so this fires at most once per paint call.
    let fail = |d: CaptureDecline| -> bool {
        crate::observe::Emit::decline("scanout_paint_mapping", &d)
            .field("mid", mapping_id)
            .field("want", format!("{width}x{height}"))
            .fail();
        false
    };

    // Deferred-writeback flush-on-access: scanout reads guest pages through a
    // raw contig view below, which bypasses the hooked readers — land any
    // resident-authoritative window (compute or render Store) first.
    //
    let _ = crate::runtime::storage_flush::flush_intersecting(state, host, mapping_id, 0, u64::MAX);

    let Some(m) = state.mappings.get(&mapping_id) else {
        return fail(CaptureDecline::NoMapping);
    };
    // Split the two teardown-window causes so a scanout paint miss names which
    // one fired (AGENTS.md: each distinct check owns its slug) — `unmapped` is
    // the guest having paged the surface off, `no_pages` an empty page list from
    // a resolve gap; both are benign-transient but must stay distinguishable.
    if !m.mapped {
        return fail(CaptureDecline::Unmapped);
    }
    if m.page_entries.is_empty() {
        return fail(CaptureDecline::NoPages);
    }
    // Geometry must be latched (same rule as write_bgra8 / archive scanout_type11).
    if !m.has_geom || m.width == 0 || m.height == 0 {
        return fail(CaptureDecline::NoGeom);
    }
    let mw = m.width;
    let mh = m.height;
    let format = if m.format != 0 {
        m.format
    } else {
        MTL_FORMAT_BGRA8_UNORM
    };
    if mw != width || mh != height {
        return fail(CaptureDecline::GeomMismatch {
            have_w: mw,
            have_h: mh,
        });
    }
    let Some(bpp) = pixel_format::bytes_per_pixel(format) else {
        return fail(CaptureDecline::BppUnknown { format });
    };
    let _ = bpp;
    let Some(tight) = pixel_format::tight_row_bytes(mw, format) else {
        return fail(CaptureDecline::TightRowUnknown { format });
    };
    // Same sample window as writeback (device descriptor base/bpr when present).
    let Some((base_off, bpr_u32, span_end)) = type11_sample_window(m, mapping_id, mw, mh, format)
    else {
        return fail(CaptureDecline::NoSampleWindow);
    };
    let bpr = bpr_u32 as usize;
    if (bpr as u64) < tight as u64 {
        return fail(CaptureDecline::BprBelowTight {
            bpr: bpr as u64,
            tight,
        });
    }
    // Contig HostOps view when possible; multi-import read_mapping_bytes otherwise.
    // Never plan_span / read_gpa walk (freelist class).
    let contig = crate::runtime::mapper::ensure_contig_view(state, host, mapping_id);
    if let Some((ptr, contig_len)) = contig {
        // Three separate faults, three separate names: the view resolved to
        // nothing, the view is shorter than the window, or the window itself is
        // degenerate. They shared one `short_view` and one `||`.
        if ptr == 0 {
            return fail(CaptureDecline::ContigViewNull);
        }
        if (contig_len as u64) < span_end {
            return fail(CaptureDecline::ContigViewShort {
                contig_len: contig_len as u64,
                span_end,
            });
        }
        if base_off >= span_end {
            return fail(CaptureDecline::BaseBeyondSpan {
                base_off,
                span_end,
                contig: true,
            });
        }
    } else if base_off >= span_end {
        return fail(CaptureDecline::BaseBeyondSpan {
            base_off,
            span_end,
            contig: false,
        });
    }
    // SAFETY: when Some, contig_len covers span_end; base_off < span_end.
    let base = contig.map(|(ptr, _)| unsafe { (ptr as *const u8).add(base_off as usize) });
    // Fragmented fullscreen IOSurfaces have hundreds of packed GPA runs. Read
    // the sample window once, not once per row: read_mapping_bytes revalidates
    // and rebuilds the run plan, so a row loop made setup O(height × pages)
    // (live 1920×1080 cold draw: setup_us≈7.5s for 2040 pages).
    let multi = if base.is_none() {
        let len = span_end.saturating_sub(base_off) as usize;
        let mut bytes = vec![0u8; len];
        if !crate::runtime::mapper::read_mapping_bytes(
            state, host, mapping_id, base_off, &mut bytes,
        ) {
            return fail(CaptureDecline::MultiReadFailed { len });
        }
        Some(bytes)
    } else {
        None
    };

    // Measure-only provenance: a contiguous host-span read vs the cold fragmented
    // multi-import (the ~12 ms/present path). `base` is Some only for a packed view.
    state.present.last_paint_src = if base.is_some() {
        crate::model::PaintSrc::GuestPagesContig
    } else {
        crate::model::PaintSrc::GuestPagesFragmented
    };

    let mut src_row = vec![0u8; tight as usize];
    let mut rgba_row = if format == MTL_FORMAT_BGRA8_UNORM
        || format == pixel_format::MTL_FORMAT_BGRA8_UNORM_SRGB
    {
        None
    } else {
        Some(vec![0u8; (mw as usize) * (RGBA8_BPP as usize)])
    };

    for y in 0..mh {
        let dst_off = (y as usize) * (dst_stride as usize);
        let dst_row_len = (mw as usize) * (RGBA8_BPP as usize);
        if dst_off + dst_row_len > dst.len() {
            return fail(CaptureDecline::DstOverflow { row: y });
        }
        let src_off = (y as usize).saturating_mul(bpr);

        if let Some(ref mut rgba) = rgba_row {
            // Non-BGRA source: stage the tight guest row, then convert via RGBA8.
            if let Some(base) = base {
                let src = unsafe { base.add(src_off) };
                unsafe {
                    std::ptr::copy_nonoverlapping(src, src_row.as_mut_ptr(), tight as usize);
                }
            } else if let Some(bytes) = multi.as_ref() {
                let end = src_off.saturating_add(tight as usize);
                let Some(row) = bytes.get(src_off..end) else {
                    return fail(CaptureDecline::ConvertRowOob { row: y });
                };
                src_row.copy_from_slice(row);
            } else {
                return fail(CaptureDecline::ConvertRowMissing { row: y });
            }
            let dst_row = &mut dst[dst_off..dst_off + dst_row_len];
            if !convert_row_to_rgba8(format, &src_row[..tight as usize], mw, rgba) {
                return fail(CaptureDecline::ConvertToRgba { format });
            }
            if !convert_rgba8_to_row(MTL_FORMAT_BGRA8_UNORM, rgba, mw, dst_row) {
                return fail(CaptureDecline::ConvertFromRgba);
            }
        } else {
            // Already BGRA8 — copy the guest row straight into dst. Skipping the
            // src_row bounce halves the per-present capture memcpy traffic (the
            // dominant `paint_us` cost on the present drain lock).
            let copy_len = dst_row_len.min(tight as usize);
            let dst_row = &mut dst[dst_off..dst_off + dst_row_len];
            if let Some(base) = base {
                let src = unsafe { base.add(src_off) };
                unsafe {
                    std::ptr::copy_nonoverlapping(src, dst_row.as_mut_ptr(), copy_len);
                }
            } else if let Some(bytes) = multi.as_ref() {
                let end = src_off.saturating_add(copy_len);
                let Some(row) = bytes.get(src_off..end) else {
                    return fail(CaptureDecline::DirectRowOob { row: y });
                };
                dst_row[..copy_len].copy_from_slice(row);
            } else {
                return fail(CaptureDecline::DirectRowMissing { row: y });
            }
            if (tight as usize) < dst_row_len {
                dst_row[tight as usize..].fill(0);
            }
        }
    }
    true
}

/// Resolve host-visible width/height for a scanout action from guest mapping geom.
pub fn present_dims(state: &DeviceState, mapping_id: u32) -> (u32, u32) {
    if let Some(m) = state.mappings.get(&mapping_id) {
        if m.has_geom && m.width > 0 && m.height > 0 {
            return (m.width, m.height);
        }
    }
    if state.present.width > 0 && state.present.height > 0 {
        return (state.present.width, state.present.height);
    }
    (0, 0)
}

/// After a successful type-11 color writeback: maybe latch front mapping / paint.
///
/// Contract:
/// - **PGDisplay**: present names one surface; mode size = that surface's geom
///   (`modeChangeHandler` sizeInPixels). We never invent host mode sizes.
/// - **Archive same_geom**: paint pre-boundary only when console unset or job
///   W×H equals established console (strips/other RTs do not resize the window).
/// - **Live Monterey**: early logo also lands in BGRA8/RGBA8 type-11 (not only
///   0x73); accept those formats pre-boundary. Post-boundary paint is DisplaySwap
///   only — writebacks must not rename `present_mapping` after `frame_flush_seen`.
pub fn note_front_buffer_writeback<M: HostMemory + crate::runtime::host::HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mapping_id: u32,
    width: u32,
    height: u32,
    rt_format: u16,
) {
    use crate::runtime::host::HostAction;

    if mapping_id == 0 || width == 0 || height == 0 {
        return;
    }
    if width > MAX_SCANOUT_DIM || height > MAX_SCANOUT_DIM {
        return;
    }
    let (map_fmt, mapped_ok, has_geom, map_w, map_h, gen) = match state.mappings.get(&mapping_id) {
        Some(m) => (
            m.format,
            m.mapped && !m.page_entries.is_empty(),
            m.has_geom && m.width > 0 && m.height > 0,
            m.width,
            m.height,
            m.content_generation,
        ),
        None => return,
    };
    let fmt = if rt_format != 0 { rt_format } else { map_fmt };
    if !is_front_buffer_format(fmt) {
        return;
    }
    if !mapped_ok {
        return;
    }

    // After the first CmdDisplaySwap, writebacks must not rename
    // `present_mapping` (PGDisplay presents the surface named by DisplaySwap).
    // Still track Composite full-FB writebacks as dual-mid peer for ClearOnly
    // present capture (x86: present mid 2/3 ClearOnly, content mid 1/4/5).
    if state.present.frame_flush_seen {
        if matches!(
            state.surface_write_kind(mapping_id),
            crate::model::SurfaceWriteKind::Composite
        ) && state.present.width > 0
            && state.present.height > 0
            && width == state.present.width
            && height == state.present.height
        {
            // Track the latest Composite full-FB writeback. Pre-boundary this
            // feeds `early_scanout_target`; post-boundary it is the peer named
            // on the `front_wb` / `present_order_hold` lines. Always update
            // here so a later writeback into the same mid refreshes the gen.
            state.present.early_front_mapping = mapping_id;
            state.present.early_front_generation = gen;
        }
        return;
    }

    // Archive same_geom vs s->surface: first enqueue establishes provisional
    // console size; later early-boot paints only when job matches. No min/max
    // dimension clamps — different sizes are refused, not rewritten to EFI.
    let console_established =
        state.present.valid && state.present.width > 0 && state.present.height > 0;
    if console_established && (width != state.present.width || height != state.present.height) {
        // Still latch which front the compositor is writing for early_scanout,
        // but do not resize/paint (mode change waits for DisplaySwap).
        state.present.present_mapping = mapping_id;
        return;
    }

    // HostAction size = mapper registry geom when known (archive
    // scanout_type11_mapping / PG sizeInPixels from the named surface).
    let (paint_w, paint_h) = if has_geom {
        (map_w, map_h)
    } else {
        (width, height)
    };
    if paint_w == 0 || paint_h == 0 || paint_w > MAX_SCANOUT_DIM || paint_h > MAX_SCANOUT_DIM {
        return;
    }

    // Establish console size at enqueue so subsequent different-geom writebacks
    // hit the barrier even before C finishes copy (archive surface after paint
    // is sequential; our async HostAction queue needs the latch here).
    state.present.present_mapping = mapping_id;
    state.present.valid = true;
    state.present.mapping_id = mapping_id;
    state.present.width = paint_w;
    state.present.height = paint_h;
    state.present.generation = gen;
    // Sticky early front: only Composite Stores own the pre-boundary console.
    // ClearOnly buffer-setup presents may overwrite present_mapping but must not
    // clear this — otherwise gfx_update falls back to BAR1 (kdp log thrash).
    if matches!(
        state.surface_write_kind(mapping_id),
        crate::model::SurfaceWriteKind::Composite
    ) {
        state.present.early_front_mapping = mapping_id;
        state.present.early_front_generation = gen;
    }

    crate::observe::line(format!(
        "front_wb LATCH mid={mapping_id} {paint_w}x{paint_h} gen={gen} fmt={fmt:#x} early_front={} (pre-boundary early paint enqueue)",
        state.present.early_front_mapping
    ));
    host.enqueue(HostAction::scanout_gen(mapping_id, paint_w, paint_h, gen));
}

/// Target for pre-boundary `gfx_update` re-pull (archive fb_update path).
///
/// Guest mapping id + geometry matching the **established console** only.
/// `None` after DisplaySwap (Apple hostPresentCount re-show only).
///
/// **Mode-switch contract:** writebacks may latch
/// `present_mapping` to a new-resolution FB before the present boundary, but
/// must not resize the host window. Only [`note_front_buffer_writeback`]
/// same-geom paints and **CmdDisplaySwap** (HostAction) may change console
/// size — matching archive same_geom + PG `modeChangeHandler` at present.
pub fn early_scanout_target(state: &DeviceState) -> Option<(u32, u32, u32, u32)> {
    if state.present.frame_flush_seen {
        return None;
    }
    // Prefer sticky Composite writeback front over present_mapping (often a
    // ClearOnly flip buffer after present_defer_boundary).
    let candidates = [
        state.present.early_front_mapping,
        state.present.present_mapping,
    ];
    for mapping_id in candidates {
        if mapping_id == 0 {
            continue;
        }
        // ClearOnly init without retain: skip (would feed solid black).
        match state.surface_write_kind(mapping_id) {
            crate::model::SurfaceWriteKind::ClearOnly if !state.present.frame_valid => {
                continue;
            }
            _ => {}
        }
        let Some(m) = state.mappings.get(&mapping_id) else {
            continue;
        };
        if !m.mapped {
            continue;
        }
        if m.format != 0 && !is_front_buffer_format(m.format) {
            continue;
        }
        let (w, h) = present_dims(state, mapping_id);
        if w == 0 || h == 0 {
            continue;
        }
        if state.present.valid
            && state.present.width > 0
            && state.present.height > 0
            && (w != state.present.width || h != state.present.height)
        {
            continue;
        }
        let gen = if mapping_id == state.present.early_front_mapping
            && state.present.early_front_generation != 0
        {
            state
                .present
                .early_front_generation
                .max(m.content_generation)
        } else {
            m.content_generation
        };
        return Some((mapping_id, w, h, gen));
    }
    None
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::contract::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
    use crate::model::{DeviceId, PAGE_SHIFT_ARM64E};
    use crate::runtime::host::{FakeHost, HostActionKind};

    const ALL_CAPTURE: &[CaptureDecline] = &[
        CaptureDecline::NoMapping,
        CaptureDecline::Unmapped,
        CaptureDecline::NoPages,
        CaptureDecline::NoGeom,
        CaptureDecline::GeomMismatch {
            have_w: 0,
            have_h: 0,
        },
        CaptureDecline::BppUnknown { format: 0 },
        CaptureDecline::TightRowUnknown { format: 0 },
        CaptureDecline::NoSampleWindow,
        CaptureDecline::BprBelowTight { bpr: 0, tight: 0 },
        CaptureDecline::ContigViewNull,
        CaptureDecline::ContigViewShort {
            contig_len: 0,
            span_end: 0,
        },
        CaptureDecline::BaseBeyondSpan {
            base_off: 0,
            span_end: 0,
            contig: true,
        },
        CaptureDecline::MultiReadFailed { len: 0 },
        CaptureDecline::DstOverflow { row: 0 },
        CaptureDecline::ConvertRowOob { row: 0 },
        CaptureDecline::ConvertRowMissing { row: 0 },
        CaptureDecline::ConvertToRgba { format: 0 },
        CaptureDecline::ConvertFromRgba,
        CaptureDecline::DirectRowOob { row: 0 },
        CaptureDecline::DirectRowMissing { row: 0 },
    ];

    /// Every capture reason names the rail that wrote it.
    ///
    /// Bare, three of these belonged to other rails too — `unmapped` and
    /// `short_view` to the guest-page import path and `no_mapping` to the type-5
    /// loader — so a `grep reason=unmapped` over one boot returned a mix of
    /// subsystems. Crate-wide distinctness is `observe::gate`'s job; the prefix is
    /// this module's.
    #[test]
    fn every_capture_reason_names_its_rail_and_is_distinct() {
        use crate::observe::Decline as _;
        let mut slugs: Vec<&str> = Vec::new();
        for d in ALL_CAPTURE {
            assert!(
                d.slug().starts_with("capture_"),
                "{} is not namespaced to the capture rail",
                d.slug()
            );
            slugs.push(d.slug());
        }
        slugs.sort_unstable();
        let before = slugs.len();
        slugs.dedup();
        assert_eq!(before, slugs.len(), "duplicate CaptureDecline slug");
    }

    /// **`short_view` was one `if` with three `||`-ed conditions.** A null host
    /// pointer, a view shorter than the sample window, and a degenerate window
    /// are three different faults with three different fixes, and they reported
    /// one name. This is the "N checks behind one status" class, inside a single
    /// expression.
    #[test]
    fn the_three_faults_that_shared_short_view_have_three_names() {
        use crate::observe::Decline as _;
        let names = [
            CaptureDecline::ContigViewNull.slug(),
            CaptureDecline::ContigViewShort {
                contig_len: 4,
                span_end: 8,
            }
            .slug(),
            CaptureDecline::BaseBeyondSpan {
                base_off: 8,
                span_end: 8,
                contig: true,
            }
            .slug(),
        ];
        let mut sorted = names.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), 3, "{names:?}");
        assert!(names.iter().all(|n| *n != "capture_short_view"));
    }

    /// The two row-read sites the old names merged sit on paths whose bounds
    /// differ — the converting path slices `tight` bytes, the direct path
    /// `min(dst_row, tight)` — so they can fail under different conditions and
    /// must not answer to one name.
    #[test]
    fn the_convert_and_direct_row_paths_do_not_share_a_name() {
        use crate::observe::Decline as _;
        assert_ne!(
            CaptureDecline::ConvertRowOob { row: 1 }.slug(),
            CaptureDecline::DirectRowOob { row: 1 }.slug()
        );
        assert_ne!(
            CaptureDecline::ConvertRowMissing { row: 1 }.slug(),
            CaptureDecline::DirectRowMissing { row: 1 }.slug()
        );
    }

    /// A capture refusal must carry the numbers behind it — "the view was short"
    /// without the two lengths does not say by how much, and the console is black
    /// either way.
    #[test]
    fn a_capture_refusal_carries_its_numbers() {
        use crate::observe::Decline as _;
        assert_eq!(
            CaptureDecline::ContigViewShort {
                contig_len: 4096,
                span_end: 8_294_400,
            }
            .fields(),
            vec![
                ("contig_len", "4096".to_string()),
                ("span_end", "8294400".to_string()),
            ]
        );
        assert_eq!(
            CaptureDecline::GeomMismatch {
                have_w: 800,
                have_h: 600,
            }
            .fields(),
            vec![("have", "800x600".to_string())]
        );
        // Field values are grepped out of a space-separated line.
        for d in ALL_CAPTURE {
            for (k, v) in d.fields() {
                assert!(!k.contains(' '), "{k}");
                assert!(!v.contains(' '), "{}: {v}", d.slug());
            }
        }
    }

    #[test]
    fn missing_mapping_fails_without_latching() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let mut host = FakeHost::new();
        let w = 4u32;
        let h = 2u32;
        let stride = w * 4;
        let mut dst = vec![0xAAu8; (stride * h) as usize];
        assert_eq!(
            copy_to_bgra8(&mut state, &mut host, 1, &mut dst, stride, w, h, 0),
            ScanoutCopyResult::Failed
        );
        // Destination untouched; generation not latched.
        assert!(dst.iter().all(|&b| b == 0xAA));
        assert!(!state.present.valid);
    }

    #[test]
    fn early_boot_front_formats_and_geometry_barrier() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let mut host = FakeHost::new();
        let mapping_id = 7u32;
        assert!(state.map_surface(mapping_id));
        {
            let m = state.mappings.get_mut(&mapping_id).unwrap();
            m.mapped = true;
            m.has_geom = true;
            m.width = 1920;
            m.height = 1080;
            m.format = MTL_FORMAT_BGRA8_UNORM;
            m.page_entries = vec![(1u32 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
            m.content_generation = 3;
        }
        // Live Monterey: first full-frame BGRA8 writeback enqueues at guest geom.
        note_front_buffer_writeback(
            &mut state,
            &mut host,
            mapping_id,
            1920,
            1080,
            MTL_FORMAT_BGRA8_UNORM,
        );
        assert_eq!(host.actions.len(), 1);
        assert_eq!(host.actions[0].kind, HostActionKind::ScanoutUpdate);
        assert_eq!(host.actions[0].a1, 1920);
        assert_eq!(host.actions[0].a2, 1080);
        assert!(state.present.valid);
        host.actions.clear();
        // Same geom → paint again.
        note_front_buffer_writeback(
            &mut state,
            &mut host,
            mapping_id,
            1920,
            1080,
            MTL_FORMAT_BGRA8_UNORM,
        );
        assert_eq!(host.actions.len(), 1);
        host.actions.clear();
        // Different geom → latch only, no paint (archive same_geom; not resized).
        note_front_buffer_writeback(
            &mut state,
            &mut host,
            mapping_id,
            1920,
            24,
            MTL_FORMAT_BGRA8_UNORM,
        );
        assert!(host.actions.is_empty());
        assert_eq!(state.present.width, 1920);
        assert_eq!(state.present.height, 1080);
        // Non-front format → ignore.
        note_front_buffer_writeback(&mut state, &mut host, mapping_id, 1920, 1080, 0x9999);
        assert!(host.actions.is_empty());
    }

    #[test]
    fn early_scanout_target_refuses_resize_geom() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        // Established console 1920×1080 after first paint.
        state.present.valid = true;
        state.present.width = 1920;
        state.present.height = 1080;
        state.present.present_mapping = 5;
        assert!(state.map_surface(5));
        {
            let m = state.mappings.get_mut(&5).unwrap();
            m.mapped = true;
            m.has_geom = true;
            m.width = 1440;
            m.height = 1080;
            m.format = MTL_FORMAT_RGBA16_FLOAT;
            m.content_generation = 9;
        }
        // present_mapping points at new mode FB, but early gfx_update must not
        // resize — DisplaySwap owns modeChangeHandler sizeInPixels.
        assert!(early_scanout_target(&state).is_none());

        // Same geom re-pull still allowed.
        {
            let m = state.mappings.get_mut(&5).unwrap();
            m.width = 1920;
            m.height = 1080;
        }
        let t = early_scanout_target(&state).expect("same-geom target");
        assert_eq!(t, (5, 1920, 1080, 9));
    }

    /// ClearOnly init present_mapping must not early-paint (keep BAR1).
    #[test]
    fn early_scanout_target_refuses_clear_only_init() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        state.present.valid = true;
        state.present.width = 1920;
        state.present.height = 1080;
        state.present.present_mapping = 2;
        state.present.frame_flush_seen = false;
        state.present.frame_valid = false;
        assert!(state.map_surface(2));
        {
            let m = state.mappings.get_mut(&2).unwrap();
            m.mapped = true;
            m.has_geom = true;
            m.width = 1920;
            m.height = 1080;
            m.format = MTL_FORMAT_BGRA8_UNORM;
            m.content_generation = 1;
        }
        state.note_surface_clear(2);
        assert!(early_scanout_target(&state).is_none());

        // Composite latch may early-paint.
        state.note_surface_composite(2);
        let t = early_scanout_target(&state).expect("composite early target");
        assert_eq!(t.0, 2);
        assert_eq!((t.1, t.2), (1920, 1080));
    }

    /// Sticky early_front survives ClearOnly present_mapping thrash.
    #[test]
    fn early_scanout_prefers_sticky_composite_over_clear_present() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        state.present.valid = true;
        state.present.width = 1920;
        state.present.height = 1080;
        state.present.frame_flush_seen = false;
        state.present.frame_valid = false;
        // Logo mid (composite writeback).
        assert!(state.map_surface(1));
        {
            let m = state.mappings.get_mut(&1).unwrap();
            m.mapped = true;
            m.has_geom = true;
            m.width = 1920;
            m.height = 1080;
            m.format = MTL_FORMAT_BGRA8_UNORM;
            m.content_generation = 5;
            m.page_entries = vec![1];
        }
        state.note_surface_composite(1);
        state.present.early_front_mapping = 1;
        state.present.early_front_generation = 5;
        // Guest ClearOnly flip mid overwrites present_mapping (buffer setup).
        assert!(state.map_surface(2));
        {
            let m = state.mappings.get_mut(&2).unwrap();
            m.mapped = true;
            m.has_geom = true;
            m.width = 1920;
            m.height = 1080;
            m.format = MTL_FORMAT_BGRA8_UNORM;
            m.content_generation = 1;
            m.page_entries = vec![1];
        }
        state.note_surface_clear(2);
        state.present.present_mapping = 2;
        let t = early_scanout_target(&state).expect("sticky early front");
        assert_eq!(t.0, 1, "must keep logo mid, not ClearOnly flip");
        assert_eq!(t.3, 5);
    }

    /// After CmdDisplaySwap, writebacks into the back buffer must not rename
    /// `present_mapping` (PGDisplay presents the surface named by DisplaySwap only).
    #[test]
    fn post_display_swap_writeback_does_not_rename_present_mapping() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let mut host = FakeHost::new();
        // Front mid from DisplaySwap (ch4 op8).
        state.present.frame_flush_seen = true;
        state.present.present_mapping = 3;
        state.present.host_mapping = 3;
        state.present.mapping_id = 3;
        state.present.valid = true;
        state.present.width = 1440;
        state.present.height = 1080;

        // Back buffer mid=4 receives a full-frame composite writeback.
        assert!(state.map_surface(4));
        {
            let m = state.mappings.get_mut(&4).unwrap();
            m.mapped = true;
            m.has_geom = true;
            m.width = 1440;
            m.height = 1080;
            m.format = MTL_FORMAT_RGBA16_FLOAT;
            m.page_entries = vec![(1u32 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
            m.content_generation = 12;
        }
        note_front_buffer_writeback(
            &mut state,
            &mut host,
            4,
            1440,
            1080,
            MTL_FORMAT_RGBA16_FLOAT,
        );
        // No early paint, no rename of the presented mid.
        assert!(host.actions.is_empty());
        assert_eq!(state.present.present_mapping, 3);
        assert_eq!(state.present.host_mapping, 3);
        assert!(early_scanout_target(&state).is_none());
    }

    /// Fragmented IOSurface page list: paint_mapping multi-imports instead of
    /// failing not_contig (live boot class: fullscreen present surfaces).
    #[test]
    fn paint_mapping_fragmented_pages_multi_import() {
        use crate::contract::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
        use crate::model::PAGE_SHIFT_X86;
        use crate::runtime::mapping_write::write_bgra8;

        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        host.strict_linux_map = true;
        let page = 1u64 << PAGE_SHIFT_X86;
        let gpa0 = 0x5000_0000u64;
        let gpa1 = 0x6000_0000u64;
        host.map_range(gpa0, page as usize, 0);
        host.map_range(gpa1, page as usize, 0);
        let pfn0 = (gpa0 >> PAGE_SHIFT_X86) as u32;
        let pfn1 = (gpa1 >> PAGE_SHIFT_X86) as u32;
        let mid = 7u32;
        assert!(state.map_surface(mid));
        {
            let m = state.mappings.get_mut(&mid).unwrap();
            m.mapped = true;
            m.mapping_internal = 1;
            m.page_entries = vec![
                (pfn0 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID,
                (pfn1 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID,
            ];
        }
        assert!(state.set_mapping_geom(mid, 2, 2, MTL_FORMAT_BGRA8_UNORM));
        let frame = [
            0xCCu8, 0x00, 0x00, 0xFF, 0xCC, 0x00, 0x00, 0xFF, 0xCC, 0x00, 0x00, 0xFF, 0xCC, 0x00,
            0x00, 0xFF,
        ];
        assert!(write_bgra8(&mut state, &mut host, mid, &frame, 8, 2, 2));
        let mut dst = vec![0u8; 16];
        assert!(
            paint_mapping(&mut state, &mut host, mid, &mut dst, 8, 2, 2),
            "fragmented paint must multi-import, not not_contig"
        );
        assert_eq!(&dst[..], &frame[..]);
        // Provenance: a non-contiguous page list is the cold fragmented read path,
        // not a deferred-flush reuse — the `src=` on the capture line must say so.
        assert_eq!(
            state.present.last_paint_src,
            crate::model::PaintSrc::GuestPagesFragmented
        );
    }

    /// The capture buffer is a recycled warm double-buffer, not a fresh 8 MiB
    /// alloc per present. Lock that (a) a successful capture recycles the prior
    /// retain into `capture_scratch` so the next capture reuses that allocation,
    /// and (b) a failed capture returns the scratch untouched and leaves the
    /// prior `frame_bgra` retain intact (keep-prior contract).
    #[test]
    fn capture_recycles_scratch_and_keeps_prior_retain_on_failure() {
        use crate::runtime::mapping_write::write_bgra8;

        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let mut host = FakeHost::new();
        let mid = 6u32;
        let pfn = 0x90u32;
        let gpa = (pfn as u64) << PAGE_SHIFT_ARM64E;
        host.map_range(gpa, 0x4000, 0);
        let entry = (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
        assert!(state.map_surface(mid));
        {
            let m = state.mappings.get_mut(&mid).unwrap();
            m.mapped = true;
            m.mapping_internal = 1;
            m.page_entries = vec![entry];
        }
        assert!(state.set_mapping_geom(mid, 2, 2, MTL_FORMAT_BGRA8_UNORM));
        state.present.frame_flush_seen = true;
        state.present.width = 2;
        state.present.height = 2;

        let frame_a = [
            0x10u8, 0x20, 0x30, 0xFF, 0x10, 0x20, 0x30, 0xFF, 0x10, 0x20, 0x30, 0xFF, 0x10, 0x20,
            0x30, 0xFF,
        ];
        assert!(write_bgra8(&mut state, &mut host, mid, &frame_a, 8, 2, 2));
        let gen_a = state.mappings.get(&mid).unwrap().content_generation;
        assert!(capture_present_frame(&mut state, mid, 2, 2, gen_a));
        assert_eq!(&state.present.frame_bgra[..16], &frame_a[..]);

        // Second successful capture: the prior retain buffer is recycled into
        // capture_scratch (warm, exactly the frame size — no per-present alloc).
        let frame_b = [
            0x00u8, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF,
            0x00, 0xFF,
        ];
        assert!(write_bgra8(&mut state, &mut host, mid, &frame_b, 8, 2, 2));
        let gen_b = state.mappings.get(&mid).unwrap().content_generation;
        assert!(capture_present_frame(&mut state, mid, 2, 2, gen_b));
        assert_eq!(&state.present.frame_bgra[..16], &frame_b[..]);
        assert_eq!(
            state.present.capture_scratch.len(),
            16,
            "prior retain recycled as warm scratch of the frame size"
        );

        // Capture failure (unmapped surface) must return the scratch untouched
        // and leave the frame_b retain intact.
        let bad = 99u32;
        assert!(state.map_surface(bad));
        state.present.width = 2;
        state.present.height = 2;
        assert!(!capture_present_frame(&mut state, bad, 2, 2, gen_b + 1));
        assert_eq!(
            &state.present.frame_bgra[..16],
            &frame_b[..],
            "failed capture must not disturb the prior retain"
        );
        assert_eq!(
            state.present.capture_scratch.len(),
            16,
            "failed capture recycles its (untouched) scratch"
        );
    }

    /// After host encode, later guest writebacks must not change scanout until
    /// the next DisplaySwap (Apple encodeCurrentFrame + hostPresentCount re-show).
    #[test]
    fn display_swap_snapshot_stable_against_post_swap_writeback() {
        use crate::runtime::mapping_write::write_bgra8;

        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let mut host = FakeHost::new();
        let pfn = 0x20u32;
        let gpa = (pfn as u64) << PAGE_SHIFT_ARM64E;
        host.map_range(gpa, 0x4000, 0);
        let entry = (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
        let mid = 3u32;
        assert!(state.map_surface(mid));
        {
            let m = state.mappings.get_mut(&mid).unwrap();
            m.mapped = true;
            m.mapping_internal = 1;
            m.page_entries = vec![entry];
        }
        assert!(state.set_mapping_geom(mid, 2, 2, MTL_FORMAT_BGRA8_UNORM));
        // Frame A: solid blue-ish BGRA.
        let frame_a = [
            0xCCu8, 0x00, 0x00, 0xFF, 0xCC, 0x00, 0x00, 0xFF, 0xCC, 0x00, 0x00, 0xFF, 0xCC, 0x00,
            0x00, 0xFF,
        ];
        assert!(write_bgra8(&mut state, &mut host, mid, &frame_a, 8, 2, 2));
        let gen = state.mappings.get(&mid).unwrap().content_generation;
        // Present path state as after DisplaySwap (encode pending → first paint).
        state.present.frame_flush_seen = true;
        state.present.host_mapping = mid;
        state.present.present_mapping = mid;
        state.present.width = 2;
        state.present.height = 2;
        state.present.generation = gen;
        state.present.frame_encode_pending = true;
        state.present.frame_valid = false;

        // First host paint encodes A.
        let mut dst = vec![0u8; 16];
        assert_eq!(
            copy_to_bgra8(&mut state, &mut host, mid, &mut dst, 8, 2, 2, gen),
            ScanoutCopyResult::Painted
        );
        assert_eq!(&dst[..], &frame_a[..]);
        assert!(state.present.frame_valid);
        assert!(!state.present.frame_encode_pending);

        // Post-encode composite mutates guest pages (mid-pass / next damage).
        let frame_b = [
            0x00u8, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF,
            0x00, 0xFF,
        ];
        assert!(write_bgra8(&mut state, &mut host, mid, &frame_b, 8, 2, 2));

        // Re-show same present gen still paints frozen A.
        assert_eq!(
            copy_to_bgra8(&mut state, &mut host, mid, &mut dst, 8, 2, 2, gen),
            ScanoutCopyResult::Unchanged
        );
        // Force re-blit snapshot (generation match still A after paint).
        state.present.painted_generation = 0;
        assert_eq!(
            copy_to_bgra8(&mut state, &mut host, mid, &mut dst, 8, 2, 2, gen),
            ScanoutCopyResult::Painted
        );
        assert_eq!(&dst[..], &frame_a[..]);
    }

    /// Live class (serial-20260715-054015): early paint latched painted mid/gen
    /// (EFI or live black) then capture installed logo+pill into +0x188; with
    /// encode_pending cleared at capture, host paint returned Unchanged and
    /// QMP stayed on EFI. Capture must force one snapshot blit.
    #[test]
    fn capture_forces_paint_even_when_painted_mid_gen_already_match() {
        use crate::runtime::mapping_write::write_bgra8;

        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let mut host = FakeHost::new();
        let mid = 2u32;
        let pfn = 0x50u32;
        let gpa = (pfn as u64) << PAGE_SHIFT_ARM64E;
        host.map_range(gpa, 0x4000, 0);
        let entry = (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
        assert!(state.map_surface(mid));
        {
            let m = state.mappings.get_mut(&mid).unwrap();
            m.mapped = true;
            m.mapping_internal = 1;
            m.page_entries = vec![entry];
        }
        assert!(state.set_mapping_geom(mid, 2, 2, MTL_FORMAT_BGRA8_UNORM));
        // Guest composite: logo-class sparse (non-black BGRA).
        let logo = [
            0x11u8, 0x22, 0x33, 0xFF, 0x11, 0x22, 0x33, 0xFF, 0x11, 0x22, 0x33, 0xFF, 0x11, 0x22,
            0x33, 0xFF,
        ];
        assert!(write_bgra8(&mut state, &mut host, mid, &logo, 8, 2, 2));
        let gen = state.mappings.get(&mid).unwrap().content_generation;

        // Early paint already "painted" this mid@gen (EFI path or prior live
        // paint) — the false Unchanged class without encode_pending.
        state.present.painted_mapping = mid;
        state.present.painted_generation = gen;
        state.present.frame_flush_seen = true;
        state.present.width = 2;
        state.present.height = 2;

        assert!(capture_present_frame(&mut state, mid, 2, 2, gen));
        assert!(
            state.present.frame_encode_pending,
            "capture must force next paint of +0x188"
        );
        assert_eq!(&state.present.frame_bgra[..16], &logo[..]);

        let mut dst = vec![0u8; 16];
        assert_eq!(
            copy_to_bgra8(&mut state, &mut host, mid, &mut dst, 8, 2, 2, gen),
            ScanoutCopyResult::Painted,
            "must blit retain even when painted mid/gen already match"
        );
        assert_eq!(&dst[..], &logo[..]);
        assert!(!state.present.frame_encode_pending);
        // Second paint: true Unchanged (console holds +0x188).
        assert_eq!(
            copy_to_bgra8(&mut state, &mut host, mid, &mut dst, 8, 2, 2, gen),
            ScanoutCopyResult::Unchanged
        );
    }

    /// The full-frame readback exists for exactly one reason: the DISPLAY needs
    /// CPU pixels. Two halves:
    ///
    /// - dmabuf carrying → NO readback, ever, however long since the last one.
    ///   The proxies are fed by the GPU reduction instead, so there is no
    ///   sampling floor forcing a copy any more. `frame_bgra` is dropped rather
    ///   than left holding the previous readback, while the present metadata
    ///   advances so `publish_window_frame` exports the fresh resident.
    ///
    ///   The buffer still holding the earlier frame used to be this test's
    ///   evidence that no copy ran, and it was the wrong evidence: that same
    ///   buffer is what the content verdict and the console blit read as the
    ///   CURRENT frame. `full_captures` counts readbacks directly, so it answers
    ///   the question the assertion was actually asking.
    /// - dmabuf NOT carrying → the window blits `frame_bgra`, so the readback
    ///   runs. This is what keeps the display off any env gate.
    #[test]
    fn readback_runs_only_for_the_display_never_for_the_proxies() {
        use crate::runtime::mapping_write::write_bgra8;

        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let mut host = FakeHost::new();
        let mid = 2u32;
        let pfn = 0x20u32;
        let gpa = (pfn as u64) << PAGE_SHIFT_ARM64E;
        host.map_range(gpa, 0x4000, 0);
        let entry = (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
        assert!(state.map_surface(mid));
        {
            let m = state.mappings.get_mut(&mid).unwrap();
            m.mapped = true;
            m.mapping_internal = 1;
            m.page_entries = vec![entry];
        }
        assert!(state.set_mapping_geom(mid, 2, 2, MTL_FORMAT_BGRA8_UNORM));

        // dmabuf NOT carrying: the display needs pixels, so the readback runs.
        let frame_a = [0x11u8, 0x22, 0x33, 0xFF].repeat(4);
        assert!(write_bgra8(&mut state, &mut host, mid, &frame_a, 8, 2, 2));
        let gen_a = state.mappings.get(&mid).unwrap().content_generation;
        assert!(capture_present_frame(&mut state, mid, 2, 2, gen_a));
        assert_eq!(
            &state.present.frame_bgra[..16],
            &frame_a[..],
            "display fallback must read back when no dmabuf carries the frame"
        );
        assert_eq!(state.present.full_captures, 1);
        assert_eq!(state.present.light_captures, 0);

        // dmabuf carrying: there must be no readback.
        state.present.display_from_resident = true;
        let frame_b = [0x44u8, 0x55, 0x66, 0xFF].repeat(4);
        assert!(write_bgra8(&mut state, &mut host, mid, &frame_b, 8, 2, 2));
        let gen_b = state.mappings.get(&mid).unwrap().content_generation;
        assert_ne!(gen_a, gen_b);
        assert!(capture_present_frame(&mut state, mid, 2, 2, gen_b));
        assert_eq!(
            state.present.full_captures, 1,
            "no sampling floor may force a copy once the GPU oracle feeds proxies"
        );
        assert!(
            state.present.frame_bgra.is_empty(),
            "the readback did not run, so no frame belongs to this present"
        );
        assert_eq!(state.present.light_captures, 1);
        // Present metadata still advances so the fresh resident gets exported.
        assert_eq!(state.present.frame_generation, gen_b);
        assert_eq!(state.present.frame_mapping, mid);
        assert!(state.present.frame_valid);
        assert!(state.present.frame_encode_pending);
    }

    /// qemu-shim DisplaySwap: guest pages are the single capture source
    /// (unified memory). Unreadable pages fail the capture — no mirror exists
    /// to invent content from; the prior +0x188 retain covers the console.
    #[test]
    fn capture_present_reads_pages_and_fails_when_unreadable() {
        use crate::runtime::mapping_write::write_bgra8;

        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let mut host = FakeHost::new();
        let mid = 3u32;
        let pfn = 0x40u32;
        let gpa = (pfn as u64) << PAGE_SHIFT_ARM64E;
        host.map_range(gpa, 0x4000, 0);
        let entry = (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
        assert!(state.map_surface(mid));
        {
            let m = state.mappings.get_mut(&mid).unwrap();
            m.mapped = true;
            m.mapping_internal = 1;
            m.page_entries = vec![entry];
        }
        assert!(state.set_mapping_geom(mid, 2, 2, MTL_FORMAT_BGRA8_UNORM));

        // Finished composite in pages: solid white BGRA.
        let white = [0xFFu8; 16];
        assert!(write_bgra8(&mut state, &mut host, mid, &white, 8, 2, 2));
        let gen = state.mappings.get(&mid).unwrap().content_generation;
        assert!(capture_present_frame(&mut state, mid, 2, 2, gen));
        assert_eq!(&state.present.frame_bgra[0..4], &[255, 255, 255, 255]);

        // Page table unreadable + host-cache evicted → capture fails
        // (guest pages unreadable and no host encode retain).
        {
            let m = state.mappings.get_mut(&mid).unwrap();
            m.page_entries.clear();
        }
        crate::runtime::surface_cache::evict(&mut state, mid);
        assert!(!capture_present_frame(&mut state, mid, 2, 2, gen + 1));
    }

    /// Dual-mid HostAction race (PGDisplay +0x188 / encodeCurrentFrame):
    /// DisplaySwap mid3 freezes white, mid4 freezes black (overwrites +0x188).
    /// Late HostAction for mid3 still encodes **current** +0x188 (mid4 black) —
    /// not recycled live mid3 pages (logo) and not a per-mid white backlog.
    #[test]
    fn dual_mid_host_action_paints_latest_plus188_not_recycled_pages() {
        use crate::runtime::mapping_write::write_bgra8;

        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let mut host = FakeHost::new();
        for (mid, pfn) in [(3u32, 0x30u32), (4u32, 0x31u32)] {
            let gpa = (pfn as u64) << PAGE_SHIFT_ARM64E;
            host.map_range(gpa, 0x4000, 0);
            let entry = (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
            assert!(state.map_surface(mid));
            let m = state.mappings.get_mut(&mid).unwrap();
            m.mapped = true;
            m.mapping_internal = 1;
            m.page_entries = vec![entry];
            assert!(state.set_mapping_geom(mid, 2, 2, MTL_FORMAT_BGRA8_UNORM));
        }

        let white = [0xFFu8; 16];
        let black = [0x00u8; 16];
        let logo = [0xAAu8; 16];
        assert!(write_bgra8(&mut state, &mut host, 3, &white, 8, 2, 2));
        let gen3 = state.mappings.get(&3).unwrap().content_generation;
        // DisplaySwap mid3: +0x188 = white.
        assert!(capture_present_frame(&mut state, 3, 2, 2, gen3));
        state.present.frame_flush_seen = true;
        state.present.host_mapping = 3;
        state.present.present_mapping = 3;
        state.present.width = 2;
        state.present.height = 2;
        state.present.generation = gen3;

        // Guest recycles mid3 after stamp (logo damage / partial composite).
        assert!(write_bgra8(&mut state, &mut host, 3, &logo, 8, 2, 2));

        // DisplaySwap mid4: PGDisplay presentFrame installs named mid4 black.
        assert!(write_bgra8(&mut state, &mut host, 4, &black, 8, 2, 2));
        let gen4 = state.mappings.get(&4).unwrap().content_generation;
        assert!(capture_present_frame(&mut state, 4, 2, 2, gen4));
        state.present.host_mapping = 4;
        state.present.present_mapping = 4;
        state.present.generation = gen4;
        assert_eq!(
            &state.present.frame_bgra[..],
            &black[..],
            "presentFrame replaces +0x188 with named mid"
        );
        assert_eq!(state.present.frame_mapping, 4);

        // Late HostAction for mid3 — encodeCurrentFrame shows current +0x188 (mid4).
        let mut dst = vec![0u8; 16];
        assert_eq!(
            copy_to_bgra8(&mut state, &mut host, 3, &mut dst, 8, 2, 2, gen3),
            ScanoutCopyResult::Painted
        );
        assert_eq!(
            &dst[..],
            &black[..],
            "late mid3 HostAction must show current +0x188 (mid4)"
        );
        assert_ne!(&dst[..], &logo[..], "must not re-read recycled mid3 pages");
        assert_ne!(&dst[..], &white[..], "must not keep superseded mid3 retain");

        // mid4 HostAction Unchanged after paint.
        let mut dst4 = vec![0u8; 16];
        assert_eq!(
            copy_to_bgra8(&mut state, &mut host, 4, &mut dst4, 8, 2, 2, gen4),
            ScanoutCopyResult::Unchanged
        );
    }

    /// Capture fail (no pages, no host_cache) returns false; prior +0x188 stays.
    #[test]
    fn capture_fail_keeps_prior_frame() {
        use crate::runtime::mapping_write::write_bgra8;

        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let mut host = FakeHost::new();
        let pfn = 0x20u32;
        let gpa = (pfn as u64) << PAGE_SHIFT_ARM64E;
        host.map_range(gpa, 0x4000, 0);
        let entry = (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
        assert!(state.map_surface(1));
        {
            let m = state.mappings.get_mut(&1).unwrap();
            m.mapped = true;
            m.mapping_internal = 1;
            m.page_entries = vec![entry];
        }
        assert!(state.set_mapping_geom(1, 2, 2, MTL_FORMAT_BGRA8_UNORM));
        let mut content = [0u8; 16];
        content[0] = 80;
        content[1] = 80;
        content[2] = 80;
        content[3] = 255;
        assert!(write_bgra8(&mut state, &mut host, 1, &content, 8, 2, 2));
        let gen1 = state.mappings.get(&1).unwrap().content_generation;
        assert!(capture_present_frame(&mut state, 1, 2, 2, gen1));
        assert_eq!(&state.present.frame_bgra[..], &content[..]);

        // Mid2 never mapped — capture fails; prior retain intact.
        assert!(state.map_surface(2));
        assert!(state.set_mapping_geom(2, 2, 2, MTL_FORMAT_BGRA8_UNORM));
        assert!(!capture_present_frame(&mut state, 2, 2, 2, 1));
        assert_eq!(&state.present.frame_bgra[..], &content[..]);
        assert_eq!(state.present.frame_mapping, 1);
    }

    /// Dual-mid qemu-shim: Clear Store (seed=None) on lagging mid must wipe prior
    /// logo before DisplaySwap encode; alternating swap shows each mid's own
    /// finished content (not logo bleed under toolbar-only damage).
    #[test]
    fn dual_mid_clear_store_then_display_swap_both_composites() {
        use crate::runtime::mapping_write::{write_bgra8, write_rgba8_image_changed};

        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let mut host = FakeHost::new();
        // Two 2x2 mids, separate pages.
        for (mid, pfn) in [(3u32, 0x30u32), (4u32, 0x31u32)] {
            let gpa = (pfn as u64) << PAGE_SHIFT_ARM64E;
            host.map_range(gpa, 0x4000, 0);
            let entry = (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
            assert!(state.map_surface(mid));
            let m = state.mappings.get_mut(&mid).unwrap();
            m.mapped = true;
            m.mapping_internal = 1;
            m.page_entries = vec![entry];
            assert!(state.set_mapping_geom(mid, 2, 2, MTL_FORMAT_BGRA8_UNORM));
            // Boot logo seed on both.
            let logo = [0xAAu8; 16];
            assert!(write_bgra8(&mut state, &mut host, mid, &logo, 8, 2, 2));
        }
        // mid3: Clear Store full wipe to black (toolbar damage would leave logo
        // with image_changed vs clear seed — seed=None is the Clear contract).
        let clear = [0u8; 16];
        assert!(write_rgba8_image_changed(
            &mut state, &mut host, 3, &clear, None, 2, 2
        ));
        // mid4: full finished frame (white).
        let full = [0xFFu8; 16];
        assert!(write_bgra8(&mut state, &mut host, 4, &full, 8, 2, 2));

        for (mid, expect) in [(3u32, clear.as_slice()), (4u32, full.as_slice())] {
            let gen = state.mappings.get(&mid).unwrap().content_generation;
            state.present.frame_flush_seen = true;
            state.present.host_mapping = mid;
            state.present.present_mapping = mid;
            state.present.width = 2;
            state.present.height = 2;
            state.present.generation = gen;
            state.present.frame_encode_pending = true;
            state.present.frame_valid = false;
            let mut dst = vec![0u8; 16];
            assert_eq!(
                copy_to_bgra8(&mut state, &mut host, mid, &mut dst, 8, 2, 2, gen),
                ScanoutCopyResult::Painted
            );
            assert_eq!(
                &dst[..],
                expect,
                "DisplaySwap mid={mid} must encode finished content, not logo"
            );
        }
    }

    /// Pre-boundary: first same-geom writeback latches present_mapping + paints;
    /// a later different-geom front only latches mapping (mode waits DisplaySwap).
    #[test]
    fn pre_boundary_writeback_latches_present_mapping() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let mut host = FakeHost::new();
        assert!(!state.present.frame_flush_seen);
        assert!(state.map_surface(1));
        {
            let m = state.mappings.get_mut(&1).unwrap();
            m.mapped = true;
            m.has_geom = true;
            m.width = 1920;
            m.height = 1080;
            m.format = MTL_FORMAT_BGRA8_UNORM;
            m.page_entries = vec![(1u32 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
            m.content_generation = 1;
        }
        note_front_buffer_writeback(&mut state, &mut host, 1, 1920, 1080, MTL_FORMAT_BGRA8_UNORM);
        assert_eq!(state.present.present_mapping, 1);
        assert_eq!(host.actions.len(), 1);

        // Mode-switch size: latch new mid, no paint/resize.
        assert!(state.map_surface(3));
        {
            let m = state.mappings.get_mut(&3).unwrap();
            m.mapped = true;
            m.has_geom = true;
            m.width = 1440;
            m.height = 1080;
            m.format = MTL_FORMAT_RGBA16_FLOAT;
            m.page_entries = vec![(2u32 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
            m.content_generation = 2;
        }
        host.actions.clear();
        note_front_buffer_writeback(
            &mut state,
            &mut host,
            3,
            1440,
            1080,
            MTL_FORMAT_RGBA16_FLOAT,
        );
        assert!(host.actions.is_empty());
        assert_eq!(state.present.present_mapping, 3);
        assert_eq!(state.present.width, 1920);
        assert_eq!(state.present.height, 1080);
    }

    /// Regression guard for `present_dims`, the scanout sizing lookup. The
    /// blit copies `width*height` from these dims, so their precedence is
    /// load-bearing: a present that reads the wrong dimensions blits with the
    /// wrong stride/extent -> a torn or clipped scanout. Lock the 3-tier
    /// precedence: the mapping's own valid geometry wins; else the retained
    /// present dims; else (0, 0). A mapping with `has_geom == false` or a zero
    /// axis must NOT be trusted (it falls through to the present dims).
    #[test]
    fn present_dims_precedence_mapping_then_present_then_zero() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let mid = 5u32;

        // No mapping, no present -> (0, 0), never a partial/garbage size.
        assert_eq!(present_dims(&state, mid), (0, 0));

        // Present dims present but no valid mapping geometry -> present dims.
        state.present.width = 1440;
        state.present.height = 900;
        assert_eq!(present_dims(&state, mid), (1440, 900));

        assert!(state.map_surface(mid));
        {
            let m = state.mappings.get_mut(&mid).unwrap();
            m.width = 800;
            m.height = 600;
            m.has_geom = false; // geometry not yet valid
        }
        assert_eq!(
            present_dims(&state, mid),
            (1440, 900),
            "has_geom == false must fall through to present dims",
        );

        // A zero axis is not valid mapping geometry either.
        {
            let m = state.mappings.get_mut(&mid).unwrap();
            m.has_geom = true;
            m.height = 0;
        }
        assert_eq!(
            present_dims(&state, mid),
            (1440, 900),
            "a zero-height mapping must not override present dims",
        );

        // Fully valid mapping geometry wins over the retained present dims.
        {
            let m = state.mappings.get_mut(&mid).unwrap();
            m.height = 600;
        }
        assert_eq!(present_dims(&state, mid), (800, 600));
    }

    /// Regression guard for the per-row scanout copy (`blit_bgra_buffer`).
    ///
    /// This is the primitive every present ultimately writes pixels through
    /// (`blit_present_snapshot` → `copy_to_bgra8`). Its correctness maps
    /// directly onto the named framebuffer bugs:
    ///  - a per-row offset/shear (indexing dst by `src_stride` or vice-versa)
    ///    is exactly "a/b framebuffer corruption";
    ///  - copying into a strided dst must land each `width*BPP` row at
    ///    `y*dst_stride` and leave the row's padding tail untouched — writing
    ///    into the pad, or reading past the row into the next, corrupts the
    ///    scanout;
    ///  - a too-small `src` must be rejected whole (no partial garbage copy),
    ///    else stale/torn bytes reach the display (residue).
    #[test]
    fn blit_bgra_buffer_row_offsets_and_bounds_are_exact() {
        let bpp = RGBA8_BPP as usize;
        let (width, height) = (3u32, 2u32);
        let src_stride = width as usize * bpp; // 12
        assert_eq!(src_stride, 12);

        // Distinct per-byte source so any misrouted row/byte is detectable.
        let src: Vec<u8> = (0..(src_stride * height as usize) as u8).collect();

        // 1) Tight dst (dst_stride == src_stride): byte-exact full copy.
        {
            let mut dst = vec![0u8; src.len()];
            assert!(blit_bgra_buffer(
                &src,
                &mut dst,
                src_stride as u32,
                width,
                height
            ));
            assert_eq!(dst, src, "tight blit must be byte-identical");
        }

        // 2) Strided dst (dst_stride > src_stride): each row lands at
        //    y*dst_stride; the trailing pad of every row stays untouched.
        {
            let dst_stride = src_stride + bpp; // one extra pixel of pad/row
            let mut dst = vec![0xEEu8; dst_stride * height as usize];
            assert!(blit_bgra_buffer(
                &src,
                &mut dst,
                dst_stride as u32,
                width,
                height
            ));
            for y in 0..height as usize {
                let doff = y * dst_stride;
                let soff = y * src_stride;
                assert_eq!(
                    &dst[doff..doff + src_stride],
                    &src[soff..soff + src_stride],
                    "row {y} must land at y*dst_stride, not sheared",
                );
                // The pad past the copied width must be preserved (not clobbered,
                // not fed the next row's bytes).
                assert!(
                    dst[doff + src_stride..doff + dst_stride]
                        .iter()
                        .all(|&b| b == 0xEE),
                    "row {y} padding tail must be left untouched",
                );
            }
        }

        // 3) Undersized src: reject whole, leave dst pristine (no partial copy).
        {
            let short = &src[..src.len() - 1];
            let mut dst = vec![0x11u8; src.len()];
            assert!(
                !blit_bgra_buffer(short, &mut dst, src_stride as u32, width, height),
                "src shorter than width*height*BPP must be refused",
            );
            assert!(
                dst.iter().all(|&b| b == 0x11),
                "a refused blit must not have written any dst byte",
            );
        }

        // 4) Narrower dst than src (dst_stride < src_stride): copy min(strides)
        //    per row — never overread src past the row nor overrun dst.
        {
            let dst_stride = src_stride - bpp; // 8 bytes/row, drops last pixel
            let mut dst = vec![0u8; dst_stride * height as usize];
            assert!(blit_bgra_buffer(
                &src,
                &mut dst,
                dst_stride as u32,
                width,
                height
            ));
            for y in 0..height as usize {
                let doff = y * dst_stride;
                let soff = y * src_stride;
                assert_eq!(
                    &dst[doff..doff + dst_stride],
                    &src[soff..soff + dst_stride],
                    "row {y} must copy min(strides) bytes from the row start",
                );
            }
        }
    }

    /// Regression: the present-boundary seed flag `presented_needs_guest_seed`
    /// (inserted on every capture) MUST be pruned when a mapping is torn down, so
    /// a recycled mapping_id (a new, unrelated surface reusing the id after
    /// DeleteIOSurfaceBacking2) does not have its FIRST LOAD draw consume a stale
    /// flag and bleed the current retained front frame over its own resident —
    /// the "background does not clear cleanly" residue class. Both teardown
    /// hooks (`unmap_surface`, `condemn_surface_backing`) route through
    /// `forget_compositor_mapping`.
    #[test]
    fn present_boundary_seed_flag_is_pruned_on_teardown() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let (torn, kept) = (5u32, 6u32);
        // Both mids were presented (flag set), as a capture would.
        state.presented_needs_guest_seed.insert(torn);
        state.presented_needs_guest_seed.insert(kept);

        // Tearing down `torn` prunes only its flag.
        state.unmap_surface(torn);
        assert!(
            !state.presented_needs_guest_seed.contains(&torn),
            "teardown must prune the present-boundary seed flag (stale-recycle bleed)"
        );
        assert!(
            state.presented_needs_guest_seed.contains(&kept),
            "an unrelated mid's flag must be untouched"
        );

        // The condemn (DeleteIOSurfaceBacking2) path prunes it too.
        state.condemn_surface_backing(kept);
        assert!(
            !state.presented_needs_guest_seed.contains(&kept),
            "condemn_surface_backing must also prune the present-boundary seed flag"
        );
    }

    /// The display transaction names one surface, and that surface alone is the
    /// capture source. A second compositor member of identical geometry holding
    /// different content is not consulted, however fresh it is: the frame that
    /// comes out is byte-for-byte the named mid's own pixels, and it is the same
    /// frame the named mid produces when no such peer exists at all.
    #[test]
    fn capture_reads_only_the_named_surface_never_a_same_geometry_peer() {
        use crate::runtime::mapping_write::write_bgra8;

        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let mut host = FakeHost::new();
        for (mid, pfn) in [(1u32, 0x40u32), (5u32, 0x41u32)] {
            let gpa = (pfn as u64) << PAGE_SHIFT_ARM64E;
            host.map_range(gpa, 0x4000, 0);
            let entry = (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
            assert!(state.map_surface(mid));
            let m = state.mappings.get_mut(&mid).unwrap();
            m.mapped = true;
            m.mapping_internal = 1;
            m.page_entries = vec![entry];
            assert!(state.set_mapping_geom(mid, 2, 2, MTL_FORMAT_BGRA8_UNORM));
        }

        // The named mid holds grey; the same-geometry peer holds white and is
        // the fresher of the two by every ordering the model tracks.
        let grey = [0x55u8; 16];
        let white = [0xFFu8; 16];
        assert!(write_bgra8(&mut state, &mut host, 1, &grey, 8, 2, 2));
        let gen1 = state.mappings.get(&1).unwrap().content_generation;
        assert!(write_bgra8(&mut state, &mut host, 5, &white, 8, 2, 2));

        // Capture the named mid with no peer published: the baseline frame.
        assert!(capture_present_frame(&mut state, 1, 2, 2, gen1));
        let alone = state.present.frame_bgra.clone();
        assert_eq!(&alone[..], &grey[..]);

        // Publish a full frame for the peer as well, then re-capture.
        state.note_dense_frame_published(1, 2, 2);
        state.note_dense_frame_published(5, 2, 2);
        // The arrangement has to be one a peer-reading capture would act on, or
        // the assertion below passes for the wrong reason: two surfaces of equal
        // geometry, the peer holding different pixels, and the peer written more
        // recently — mid 5's `write_bgra8` above runs after mid 1's, so it is the
        // later write by program order. (The model tracks no cross-mapping write
        // stamp to assert that with: the one it had existed only to feed a
        // present-staleness census and went with it.)
        let peer = state.mappings.get(&5).unwrap();
        assert_eq!((peer.width, peer.height), (2, 2));
        assert_ne!(&grey[..], &white[..]);
        state.present.frame_bgra.clear();
        state.present.frame_valid = false;
        assert!(capture_present_frame(&mut state, 1, 2, 2, gen1));
        assert_eq!(
            &state.present.frame_bgra[..],
            &alone[..],
            "a same-geometry peer must not change the named surface's frame"
        );
        assert_eq!(state.present.frame_mapping, 1);
    }

    /// A light capture must leave no frame behind, because everything
    /// downstream reads "is there a frame for this present" off
    /// `frame_bgra.is_empty()`.
    ///
    /// Skipping the readback only leaves the buffer empty if it was empty going
    /// in, and on a real boot it is not: the first present runs the full path
    /// before the guest has painted anything, capturing black, and direct
    /// present then carries every frame after it. Boot 86 on the x86/Vulkan rail
    /// judged that one frame `Black` on 481 of 481 presents — 0 `present_content`
    /// and 0 `present_content_unsampled` — while the host window rendered
    /// correctly from settle through an 11-minute idle.
    #[test]
    fn a_light_capture_leaves_no_stale_frame_behind() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let (w, h) = (64u32, 64u32);
        // The frame a full capture leaves: opaque, RGB black — the first present
        // of a boot, before anything has been composited.
        let stale: Vec<u8> = (0..(w * h) as usize).flat_map(|_| [0, 0, 0, 255]).collect();
        state.present.frame_bgra = stale.clone();
        state.present.frame_width = w;
        state.present.frame_height = h;

        // Direct present is carrying the display, so this capture takes the
        // light path and reads back nothing.
        state.present.display_from_resident = true;
        let before = state.present.light_captures;
        assert!(capture_present_frame(&mut state, 1, w, h, 1));
        assert_eq!(
            state.present.light_captures,
            before.wrapping_add(1),
            "the light path is the one under test"
        );
        assert!(
            state.present.frame_bgra.is_empty(),
            "a light capture wrote no pixels, so it must not leave {} bytes of an \
             earlier present for the content verdict and the console blit to read \
             as this one",
            state.present.frame_bgra.len()
        );

        // And the verdict that reads it now says so, instead of reporting the
        // stale frame's colour as this present's.
        use crate::runtime::drain::present_content_verdict;
        assert_eq!(
            present_content_verdict(&state.present.frame_bgra, 0),
            crate::runtime::drain::PresentContentVerdict::Unsampled
        );
        assert_eq!(
            present_content_verdict(&stale, 0),
            crate::runtime::drain::PresentContentVerdict::Black,
            "the retained frame is what used to be judged"
        );
    }
}
