//! Type-11 Store into guest IOSurface pages (workstream E).
//!
//! Vulkan composite Stores to type-11 mappings:
//! 1. **Packed contig** — `ensure_contig_view` + `present_into_host_ptr_strided`
//!    (true zero-copy GPU DMA into guest pages on stable host mappings).
//! 2. **Fragmented multi-run zero-copy** — split page table into maximal packed
//!    GPA runs, `map_pages` each run, GPU DMA via [`engine::present_into_host_runs`]
//!    (no host staging, no CPU pixel copy, no `write_gpa`) when
//!    `HostOps::map_pages_stable()` is true.
//! 3. **Non-stable host mapping fallback** — arm64/macOS maps guest pages
//!    through transient `mach_vm_remap` views; those pointers must not be
//!    retained in the Vulkan host-import cache, so MoltenVK reads the resident
//!    target back once and writes through the existing mapping writer.
//!
//! Fail-closed on unmapped / resolve-fail / short view / map_generation drift.
//! There is **no** silent CPU invent path: every failed attempt logs a concrete
//! `import_present used=0 reason=<slug>` line.
//!
//! `backend-vulkan` only — Metal-direct already has unified guest views.

#![cfg(feature = "backend-vulkan")]

use crate::backend::vulkan::engine::{self, TargetIdentity};
use crate::contract::pixel_format::{MTL_FORMAT_BGRA8_UNORM, RGBA8_BPP};
use crate::model::DeviceState;
use crate::runtime::host::{HostMemory, HostOps};
use crate::runtime::mapper;
use crate::runtime::mapping_write;
use crate::runtime::surface_cache;
use std::time::Instant;

// RGBA8_BPP used only for tight row checks in try_import_present_store.

/// Outcome of an import attempt (always-on `import_present used=` log).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImportPresentResult {
    /// GPU DMA into guest pages succeeded.
    Ok,
    /// Not attempted (no mid / zero geom).
    Skip(&'static str),
    /// Attempted and failed — **no** CPU writeback; guest pages unchanged.
    Fail(&'static str),
}

impl ImportPresentResult {
    pub fn used(self) -> bool {
        matches!(self, Self::Ok)
    }

    pub fn reason(self) -> &'static str {
        match self {
            Self::Ok => "ok",
            Self::Skip(r) | Self::Fail(r) => r,
        }
    }
}

impl crate::observe::Refusal for ImportPresentResult {
    /// Only a `Fail` is a refusal.
    ///
    /// `Skip` is control flow — the import was never attempted, because there is
    /// no mid or the geometry is zero — and `Ok` obviously is not. Making that an
    /// exhaustive `match` is the point: `Emit::refusal` returns `None` for both,
    /// so neither can be logged as a failure by accident, and a new variant
    /// forces this decision open at compile time.
    fn refusal(&self) -> Option<&'static str> {
        match self {
            Self::Ok | Self::Skip(_) => None,
            Self::Fail(r) => Some(r),
        }
    }
}

/// A Store/import holding the ordered render worker beyond this boundary is
/// worth decomposing. Measurement only; it never changes Store behavior.
const IMPORT_STORE_TIMING_SLOW_US: u64 = 1_000;

struct ImportTiming {
    started: Instant,
    revalidate_us: u64,
    window_us: u64,
    resident_stats_us: u64,
}

impl ImportTiming {
    fn new() -> Self {
        Self {
            started: Instant::now(),
            revalidate_us: 0,
            window_us: 0,
            resident_stats_us: 0,
        }
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "the proxy line records each independent import outcome counter"
    )]
    fn log(
        &self,
        path: &'static str,
        mapping_id: u32,
        width: u32,
        height: u32,
        map_us: u64,
        dma_us: u64,
        post_us: u64,
        runs: usize,
        cache_hits: usize,
        cache_misses: usize,
        cached: bool,
    ) {
        let total_us = self.started.elapsed().as_micros() as u64;
        if import_store_timing_is_slow(total_us) {
            crate::observe::off(import_store_timing_line(
                path,
                mapping_id,
                width,
                height,
                total_us,
                self.revalidate_us,
                self.window_us,
                self.resident_stats_us,
                map_us,
                dma_us,
                post_us,
                runs,
                cache_hits,
                cache_misses,
                cached,
            ));
        }
    }
}

#[inline]
fn import_store_timing_is_slow(total_us: u64) -> bool {
    total_us >= IMPORT_STORE_TIMING_SLOW_US
}

#[allow(clippy::too_many_arguments)]
fn import_store_timing_line(
    path: &'static str,
    mapping_id: u32,
    width: u32,
    height: u32,
    total_us: u64,
    revalidate_us: u64,
    window_us: u64,
    resident_stats_us: u64,
    map_us: u64,
    dma_us: u64,
    post_us: u64,
    runs: usize,
    cache_hits: usize,
    cache_misses: usize,
    cached: bool,
) -> String {
    format!(
        "import_store_timing path={path} mid={mapping_id} {width}x{height} total_us={total_us} revalidate_us={revalidate_us} window_us={window_us} resident_stats_us={resident_stats_us} map_us={map_us} dma_us={dma_us} post_us={post_us} runs={runs} cache_hits={cache_hits} cache_misses={cache_misses} cached={} threshold_us={IMPORT_STORE_TIMING_SLOW_US}",
        cached as u8
    )
}

/// Build a protocol-stable resident identity for this mapping at its current
/// [`crate::model::MappingEntry::map_generation`].
///
/// Compositor-output members at a ≥2-member geometry resolve to the shared
/// [`TargetIdentity::OutputGroup`] — the guest's copy-swap contract makes the
/// members alternating storage for one logical framebuffer.
/// Fail-visible when `identity` — the resident a surface just resolved to — is
/// not keyed by that surface's own mapping id.
///
/// `ResourcePools::registry` is keyed by `TargetIdentity`, so any two mappings
/// with equal identities render into and capture from ONE `VkImage`. Distinct
/// guest surfaces have independent damage histories (WindowServer redraws a
/// buffer only where it differs from what that buffer last held), so a shared
/// resident makes every frame a fusion of damage from several buffers — the
/// mixed-generation frame class, which no other counter detects.
///
/// O(1) on the hot path: the member scan runs only when the line is emitted,
/// and emission is deduplicated by geometry and member count so a growing
/// group reports each new size once rather than per draw.
fn note_resident_identity_sharing(
    state: &DeviceState,
    identity: &TargetIdentity,
    mapping_id: u32,
    width: u32,
    height: u32,
) {
    if mapping_id == 0 || identity.surface_mapping_id() == Some(mapping_id) {
        return;
    }
    let mids: Vec<u32> = state
        .present
        .compositor_output_members
        .iter()
        .filter(|(_, m)| m.width == width && m.height == height)
        .map(|(&mid, _)| mid)
        .collect();
    use std::sync::Mutex;
    static SEEN: Mutex<Option<std::collections::BTreeSet<(u32, u32, usize)>>> = Mutex::new(None);
    let first = {
        let mut guard = SEEN.lock().unwrap_or_else(|p| p.into_inner());
        guard
            .get_or_insert_with(Default::default)
            .insert((width, height, mids.len()))
    };
    if first {
        crate::observe::fail(format!(
            "resident_identity_shared {width}x{height} mapping={mapping_id} \
             identity={identity:?} mids={mids:?} count={}",
            mids.len()
        ));
    }
}

pub fn surface_identity(
    state: &DeviceState,
    mapping_id: u32,
    width: u32,
    height: u32,
) -> TargetIdentity {
    if let Some(gid) = state.output_group_for(mapping_id, width, height) {
        {
            use std::sync::Mutex;
            static LOGGED: Mutex<Option<std::collections::BTreeSet<(u32, u32)>>> = Mutex::new(None);
            let mut guard = LOGGED.lock().unwrap_or_else(|p| p.into_inner());
            if guard
                .get_or_insert_with(Default::default)
                .insert((width, height))
            {
                let members: Vec<u32> = state
                    .present
                    .compositor_output_members
                    .iter()
                    .filter(|(_, m)| m.width == width && m.height == height)
                    .map(|(&mid, _)| mid)
                    .collect();
                crate::observe::fail(format!(
                    "compositor_group_unify {width}x{height} members={members:?} first_mid={mapping_id}"
                ));
            }
        }
        let identity = TargetIdentity::OutputGroup {
            id: gid,
            width,
            height,
            generation: 0,
        };
        note_resident_identity_sharing(state, &identity, mapping_id, width, height);
        return identity;
    }
    let gen = state
        .mappings
        .get(&mapping_id)
        .map(|m| m.map_generation as u64)
        .unwrap_or(0);
    let identity = TargetIdentity::Surface {
        id: mapping_id,
        width,
        height,
        generation: gen,
    };
    // Passes by construction today; kept on this path so the guard survives any
    // future change to what a surface resolves to.
    note_resident_identity_sharing(state, &identity, mapping_id, width, height);
    identity
}

/// Type-11 Store with non-zero geom — the only product Store class that imports.
pub fn eligible(mapping_id: u32, width: u32, height: u32) -> bool {
    mapping_id != 0 && width > 0 && height > 0
}

/// Whether two mapping ids at one geometry render into and capture from the
/// SAME resident — i.e. whether [`surface_identity`] answers equally for both.
///
/// Side-effect free, unlike calling [`surface_identity`] twice: that emits the
/// deduplicated `compositor_group_unify` / `resident_identity_shared` lines, and
/// a census asking a question must not consume the budget of the lines that
/// answer a different one.
///
/// Kept honest by `same_resident_agrees_with_surface_identity`, which asserts it
/// against real [`surface_identity`] results rather than restating the rule.
pub fn resolves_to_same_resident(
    state: &DeviceState,
    a: u32,
    b: u32,
    width: u32,
    height: u32,
) -> bool {
    if a == b {
        return true;
    }
    // Distinct mids can only share a resident through output-group unification:
    // the `Surface` variant carries the mapping id, so two of them never match.
    let ga = state.output_group_for(a, width, height);
    ga.is_some() && ga == state.output_group_for(b, width, height)
}

/// The per-member `Surface` identity, ignoring output-group unification.
pub fn member_surface_identity(
    state: &DeviceState,
    mapping_id: u32,
    width: u32,
    height: u32,
) -> TargetIdentity {
    let gen = state
        .mappings
        .get(&mapping_id)
        .map(|m| m.map_generation as u64)
        .unwrap_or(0);
    TargetIdentity::Surface {
        id: mapping_id,
        width,
        height,
        generation: gen,
    }
}

/// Rebuild exactly the resident identity a deferred render window pinned at
/// defer time (see [`try_defer_present_store`]): the unified group identity
/// for `grouped` windows, the per-mid Surface identity otherwise. Never
/// re-resolves against current membership — a membership change between
/// defer and flush must not unpin a different image than was pinned.
pub fn render_deferred_identity(
    mapping_id: u32,
    entry: &crate::model::RenderDeferredEntry,
) -> TargetIdentity {
    if entry.grouped {
        TargetIdentity::OutputGroup {
            id: OUTPUT_GROUP_ID,
            width: entry.width,
            height: entry.height,
            generation: 0,
        }
    } else {
        TargetIdentity::Surface {
            id: mapping_id,
            width: entry.width,
            height: entry.height,
            generation: entry.map_generation,
        }
    }
}

/// The single compositor-output group id (`DeviceState::output_group_for`
/// always returns this; the geometry inside the identity disambiguates).
pub const OUTPUT_GROUP_ID: u32 = 1;

/// Safety backstop on live `render_deferred_flush` windows. Each window pins a
/// resident, so an unbounded population would grow toward `MAX_MAPPINGS` (4096)
/// and pin the whole registry. This is deliberately set *above* the measured
/// YouTube page-load burst peak (~323 windows) and below `REGISTRY_CAP` (512):
/// normal use — even a heavy compositing burst — stays under it and is *absorbed*
/// by the registry (evicts=0), so the oldest-first force-flush here **does not
/// fire** and the drain worker never pays a synchronous readback storm (the
/// `cap_flush` census stays silent). It only engages for a pathological runaway
/// (a workload spraying >384 simultaneously-live deferred surfaces), where
/// landing the oldest windows early is the lesser evil vs. pinning ~all of VRAM.
pub const RENDER_DEFERRED_WINDOW_CAP: usize = 384;

/// Eagerly drop every deferred present-store window for `mapping_id`, unpinning
/// each window's resident target so the registry LRU can reclaim it. Returns
/// the number of windows dropped.
///
/// Called when a surface is unmapped or re-backed (`mapper::apply_capture`):
/// the surface's guest pages are gone / re-generated, so its deferred resident
/// content is stale and no longer needed. This is the *eager* form of the
/// `map_generation_drift` drop already in `render_flush_one` (unpin, no
/// writeback — guest pages keep their stale-but-coherent pre-Store bytes). That
/// guard only fires *lazily*, when some later access flushes the mapping; an
/// unmapped surface whose pages are never touched again never triggers it, so
/// its pin lingers for the guest lifetime. Enough such leaks pin the registry
/// past its slot cap (soft-exceed), and the non-pinned tail thrashes eviction —
/// the measured YouTube "cap blown → 120→5fps" cliff (`cap_pressure pinned=…
/// render_win=…` ballooning to hundreds).
pub fn drop_render_deferred_windows(state: &mut DeviceState, mapping_id: u32) -> u64 {
    let mut dropped = 0u64;
    for (key, entry) in state.take_render_deferred_windows(mapping_id, 0, u64::MAX) {
        engine::unpin_resident_target(&render_deferred_identity(key.mapping_id, &entry));
        dropped += 1;
    }
    dropped
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ResidentContentStats {
    rgb_nz: usize,
    rgb_max: u8,
    alpha_nz: usize,
    alpha_opaque: usize,
}

impl From<engine::Color8ContentStats> for ResidentContentStats {
    fn from(stats: engine::Color8ContentStats) -> Self {
        Self {
            rgb_nz: stats.rgb_nz,
            rgb_max: stats.rgb_max,
            alpha_nz: stats.alpha_nz,
            alpha_opaque: stats.alpha_opaque,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct ChannelTransitionStats {
    changed_px: usize,
    rb_swap_px: usize,
}

/// Measure an exact red/blue exchange between two BGRA rows.
///
/// This never selects Store behavior. It distinguishes the window-damage class
/// where changed target pixels preserve G/A but exchange the B/R bytes.
fn channel_transition_stats_bgra(before: &[u8], after: &[u8]) -> ChannelTransitionStats {
    let mut stats = ChannelTransitionStats::default();
    for (old, new) in before.chunks_exact(4).zip(after.chunks_exact(4)) {
        if old != new {
            stats.changed_px += 1;
        }
        if old[0] != old[2]
            && new[0] == old[2]
            && new[1] == old[1]
            && new[2] == old[0]
            && new[3] == old[3]
        {
            stats.rb_swap_px += 1;
        }
    }
    stats
}

/// CPU implementation of the resident content stats the GPU reduction computes
/// on the zero-copy path (`present_stats.comp` / `Color8ContentStats`). Used in
/// tests and on non-stable host mappings, where arm64/MoltenVK must take a CPU
/// readback before writing transient `mach_vm_remap` views.
fn resident_content_stats_from_rgba(px: &[u8]) -> ResidentContentStats {
    let mut stats = ResidentContentStats {
        rgb_nz: 0,
        rgb_max: 0,
        alpha_nz: 0,
        alpha_opaque: 0,
    };
    for pixel in px.chunks_exact(4) {
        let rgb_max = pixel[0].max(pixel[1]).max(pixel[2]);
        stats.rgb_nz += usize::from(rgb_max != 0);
        stats.rgb_max = stats.rgb_max.max(rgb_max);
        stats.alpha_nz += usize::from(pixel[3] != 0);
        stats.alpha_opaque += usize::from(pixel[3] == u8::MAX);
    }
    stats
}

fn should_measure_resident_content(width: u32, height: u32) -> bool {
    (width >= 1280 && height >= 720)
        || crate::runtime::census::present_proxy::is_menu_strip_geom(width, height)
}

/// Measure-only: read one tight BGRA guest mapping row (multi-import safe).
#[allow(
    clippy::too_many_arguments,
    reason = "the diagnostic reader names the exact mapping row geometry"
)]
fn guest_row_bgra_at_y<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    mapping_id: u32,
    width: u32,
    height: u32,
    base_off: u64,
    bpr: u32,
    y: u32,
) -> Option<Vec<u8>> {
    if width == 0 || height == 0 || bpr < width.saturating_mul(RGBA8_BPP) || y >= height {
        return None;
    }
    let tight = (width as usize).saturating_mul(RGBA8_BPP as usize);
    let mut row = vec![0u8; tight];
    let moff = base_off.saturating_add((y as u64).saturating_mul(bpr as u64));
    if !mapper::read_mapping_bytes(state, host, mapping_id, moff, &mut row) {
        return None;
    }
    Some(row)
}

/// Measure-only: sample one guest mapping row after DMA (multi-import safe).
#[allow(
    clippy::too_many_arguments,
    reason = "the diagnostic sampler names the exact mapping row geometry"
)]
fn guest_row_rgb_stats_at_y<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    mapping_id: u32,
    width: u32,
    height: u32,
    base_off: u64,
    bpr: u32,
    y: u32,
) -> Option<(usize, u8)> {
    let row = guest_row_bgra_at_y(state, host, mapping_id, width, height, base_off, bpr, y)?;
    // BGRA on wire — occupancy is channel-order invariant for max of first 3.
    let (nz, max, _) = crate::observe::rgba_rgb_stats(&row);
    Some((nz, max))
}

/// Measure-only: sample guest mapping after DMA (center row, multi-import safe).
fn guest_center_row_rgb_stats<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    mapping_id: u32,
    width: u32,
    height: u32,
    base_off: u64,
    bpr: u32,
) -> Option<(usize, u8)> {
    guest_row_rgb_stats_at_y(
        state,
        host,
        mapping_id,
        width,
        height,
        base_off,
        bpr,
        height / 2,
    )
}

/// Always-on content proxy: resident vs guest after import.
///
/// `res_rgb_nz=None` → skipped (small) or read_target failed.
/// Class map:
/// - res high, guest low → DMA / multi-run / sample-window bug
/// - res low, guest low → GPU drew empty (upstream sample/clear/load)
/// - both high → import content OK (visual issue is scanout ownership)
fn import_content_line(
    mapping_id: u32,
    width: u32,
    height: u32,
    outcome: &str,
    res: Option<ResidentContentStats>,
    guest: Option<(usize, u8)>,
    transition: Option<ChannelTransitionStats>,
) -> String {
    let res = res.unwrap_or(ResidentContentStats {
        rgb_nz: usize::MAX,
        rgb_max: 0,
        alpha_nz: usize::MAX,
        alpha_opaque: usize::MAX,
    });
    let (gnz, gmax) = guest.unwrap_or((usize::MAX, 0));
    let transition = transition.unwrap_or(ChannelTransitionStats {
        changed_px: usize::MAX,
        rb_swap_px: usize::MAX,
    });
    format!(
        "import_content mid={mapping_id} {width}x{height} outcome={outcome} res_rgb_nz={} res_max={} res_alpha_nz={} res_alpha_opaque={} guest_row_nz={} guest_row_max={} changed_row_px={} rb_swap_row_px={}",
        if res.rgb_nz == usize::MAX { -1i64 } else { res.rgb_nz as i64 },
        res.rgb_max,
        if res.alpha_nz == usize::MAX { -1i64 } else { res.alpha_nz as i64 },
        if res.alpha_opaque == usize::MAX { -1i64 } else { res.alpha_opaque as i64 },
        if gnz == usize::MAX { -1i64 } else { gnz as i64 },
        gmax,
        if transition.changed_px == usize::MAX { -1i64 } else { transition.changed_px as i64 },
        if transition.rb_swap_px == usize::MAX { -1i64 } else { transition.rb_swap_px as i64 }
    )
}

/// `success` selects the sink: an import-content census attached to a
/// successful used=1 store is offline-analysis (OFF, fires per import on the
/// healthy path); the same census attached to a used=0 import failure stays
/// fail-visible so the drop is not hidden.
#[allow(
    clippy::too_many_arguments,
    reason = "the proxy records all load-bearing import identity and occupancy fields"
)]
fn log_import_content(
    mapping_id: u32,
    width: u32,
    height: u32,
    outcome: &str,
    res: Option<ResidentContentStats>,
    guest: Option<(usize, u8)>,
    transition: Option<ChannelTransitionStats>,
    success: bool,
) {
    let line = import_content_line(mapping_id, width, height, outcome, res, guest, transition);
    if success {
        crate::observe::off(line);
    } else {
        crate::observe::fail(line);
    }
}

/// Measure-only: strip Store ABI census (menu-bar residual).
///
/// Logs mapping latch vs job dims, multiplanar, invent-vs-device sample window,
/// and y0 + center guest row occupancy. Does **not** change Store behavior.
#[allow(
    clippy::too_many_arguments,
    reason = "the strip proxy records the decoded job and mapping geometry"
)]
fn log_strip_import<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    mapping_id: u32,
    job_w: u32,
    job_h: u32,
    bpr: u32,
    base_off: u64,
    from_device: bool,
    reason: &str,
    res: Option<ResidentContentStats>,
) {
    if !crate::runtime::census::present_proxy::is_menu_strip_geom(job_w, job_h) {
        return;
    }
    let (map_w, map_h, map_fmt, pages, multi) = match state.mappings.get(&mapping_id) {
        Some(m) => (
            m.width,
            m.height,
            m.format,
            m.page_entries.len(),
            crate::runtime::objects::mapping_is_multiplanar(m) as u8,
        ),
        None => (0, 0, 0, 0, 0),
    };
    let y0 = guest_row_rgb_stats_at_y(state, host, mapping_id, job_w, job_h, base_off, bpr, 0);
    let ymid = guest_center_row_rgb_stats(state, host, mapping_id, job_w, job_h, base_off, bpr);
    let (rnz, rmax) = res
        .map(|stats| (stats.rgb_nz, stats.rgb_max))
        .unwrap_or((usize::MAX, 0));
    let (y0_nz, y0_max) = y0.unwrap_or((usize::MAX, 0));
    let (ym_nz, ym_max) = ymid.unwrap_or((usize::MAX, 0));
    // Menu-strip A/B diagnostic census on a SUCCESSFUL import (both call sites
    // pass `ok`/`ok_runs`) — route OFF so it does not pollute the curated
    // real-error view; still greppable as `OFF strip_import`.
    crate::observe::off(format!(
        "strip_import mid={mapping_id} job={job_w}x{job_h} map={map_w}x{map_h} map_fmt={map_fmt:#x} multi={multi} pages={pages} bpr={bpr} invent={} base_off={base_off:#x} reason={reason} res_rgb_nz={} res_max={} guest_y0_nz={} guest_y0_max={} guest_ymid_nz={} guest_ymid_max={}",
        (!from_device) as u8,
        if rnz == usize::MAX { -1i64 } else { rnz as i64 },
        rmax,
        if y0_nz == usize::MAX { -1i64 } else { y0_nz as i64 },
        y0_max,
        if ym_nz == usize::MAX { -1i64 } else { ym_nz as i64 },
        ym_max
    ));
    // Once-per-process: dump first guest row as PPM for A/B vs present top band.
    static DUMPED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !DUMPED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        maybe_dump_strip_guest_row(state, host, mapping_id, job_w, job_h, base_off, bpr);
    }
}

/// Measure-only: write first guest row of a strip Store to `/tmp/reims-vgpu-strip-import-*.ppm`.
fn maybe_dump_strip_guest_row<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    mapping_id: u32,
    width: u32,
    height: u32,
    base_off: u64,
    bpr: u32,
) {
    if width == 0 || bpr < width.saturating_mul(RGBA8_BPP) {
        return;
    }
    let dump_h = height.clamp(1, 4);
    let tight = (width as usize).saturating_mul(RGBA8_BPP as usize);
    let mut rows = vec![0u8; tight.saturating_mul(dump_h as usize)];
    for y in 0..dump_h {
        let moff = base_off.saturating_add((y as u64).saturating_mul(bpr as u64));
        let off = (y as usize).saturating_mul(tight);
        if !mapper::read_mapping_bytes(state, host, mapping_id, moff, &mut rows[off..off + tight]) {
            return;
        }
    }
    let path = format!("/tmp/reims-vgpu-strip-import-mid{mapping_id}-{width}x{dump_h}.ppm");
    if let Ok(mut f) = std::fs::File::create(&path) {
        use std::io::Write;
        let _ = writeln!(f, "P6\n{width} {dump_h}\n255");
        for px in rows.chunks_exact(4) {
            // Guest wire is BGRA.
            let _ = f.write_all(&[px[2], px[1], px[0]]);
        }
        crate::observe::fail(format!(
            "strip_import dump path={path} mid={mapping_id} {width}x{dump_h}"
        ));
    }
}

/// DMA resident BGRA target into guest mapping pages (stride-correct).
///
/// Preconditions (fail-closed):
/// 1. Mapping still mapped with same `map_generation` as `identity`
/// 2. [`mapper::revalidate_mapping_pages`] + sample window covers height×BPR
/// 3. Packed contig import **or** multi-run zero-copy GPU DMA into page runs
///
/// On success: bumps content generation, notes composite, **evicts** host_cache
/// so capture reads the freshly-written guest pages.
pub fn try_import_present_store<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    identity: &TargetIdentity,
    mapping_id: u32,
    width: u32,
    height: u32,
    full_quad_bounds: bool,
) -> ImportPresentResult {
    let mut timing = ImportTiming::new();
    if !eligible(mapping_id, width, height) {
        return ImportPresentResult::Skip("ineligible");
    }

    // Lifetime gate. Per-mid Surface identities embed the mapping lifetime —
    // the generation must still match.
    //
    // A group identity carries no member lifetime (its generation is constant),
    // so all this arm can check is that the pinned identity describes *this*
    // Store: same group, same geometry. It must not re-resolve membership.
    // Membership at defer time is what chose which resident to read, and that
    // choice is already inside the pinned identity; asking again at flush time
    // re-decides a settled question using `presented_geoms`, which our own
    // publish detector infers. The mapping's lifetime — the thing that would
    // make the write unsafe — is covered by a real generation compare:
    // `storage_flush::render_flush_one` runs one against the defer-time record
    // for every deferred window before calling here, and a synchronous Store
    // resolved this identity a few lines earlier in the same packet.
    //
    // Re-resolving cost real guest paint. `map_surface` and
    // `condemn_surface_backing` both prune `presented_geoms` while
    // deliberately KEEPING `map_generation` and the mapping's deferred windows,
    // so `mapper::resolve` can settle the lifetime later by fingerprint. On a
    // swapchain buffer recycle the flush trigger lands in that gap: pages just
    // reprieved, membership not yet re-granted. Measured on the x86/Vulkan rail
    // (`page-loads.sh`, boots 82 and 83): 4/4 refusals were exactly that state
    // (`miss=presented_pruned`, generation intact), with the compositor
    // re-granting membership to the same mid at the same geometry 30-95 ms
    // later. Nothing there is a different logical surface; the refusal was
    // destroying a paint the layer below had just certified.
    let gate_fail = if matches!(identity, TargetIdentity::OutputGroup { .. }) {
        // Structural, not stateful: compare against the identity a Store of
        // this geometry pins, so a future producer adding a field to
        // `OutputGroup` cannot silently drop out of the comparison.
        let for_this_store = TargetIdentity::OutputGroup {
            id: OUTPUT_GROUP_ID,
            width,
            height,
            generation: 0,
        };
        if *identity != for_this_store {
            Some(ImportDecline::GroupGeomDrift)
        } else {
            // Measure-only: the write now proceeds while the group no longer
            // admits this mapping. This is the case the re-resolve used to
            // refuse, so it must stay visible — if the deletion above is wrong,
            // this is the line that says where to look.
            if let Err(miss) = state.output_group_resolve(mapping_id, width, height) {
                crate::observe::fail(format!(
                    "import_group_membership_stale mid={mapping_id} geom={width}x{height} \
                     miss={} (writing the pinned resident anyway; the lifetime gate passed)",
                    match miss {
                        crate::model::OutputGroupMiss::NotPresentedHere { presented: None } =>
                            "presented_pruned".to_string(),
                        crate::model::OutputGroupMiss::NotPresentedHere {
                            presented: Some((w, h)),
                        } => format!("presented_elsewhere:{w}x{h}"),
                        crate::model::OutputGroupMiss::NoPeer => "no_peer".to_string(),
                        crate::model::OutputGroupMiss::PriorIncarnation {
                            entry_gen,
                            current_gen,
                            ..
                        } => format!("prior_incarnation:{entry_gen}->{current_gen}"),
                    }
                ));
            }
            None
        }
    } else {
        let gen_now = state
            .mappings
            .get(&mapping_id)
            .map(|m| m.map_generation as u64)
            .unwrap_or(0);
        (identity.generation() != gen_now).then_some(ImportDecline::MapGenDrift)
    };
    if let Some(decline) = gate_fail {
        return import_fail(mapping_id, width, height, decline);
    }
    if !state
        .mappings
        .get(&mapping_id)
        .map(|m| m.mapped)
        .unwrap_or(false)
    {
        return import_fail(mapping_id, width, height, ImportDecline::Unmapped);
    }

    let revalidate_started = Instant::now();
    let revalidate_reason = mapper::revalidate_mapping_reason(state, host, mapping_id);
    timing.revalidate_us = revalidate_started.elapsed().as_micros() as u64;
    if let Some(reason) = revalidate_reason {
        // Surface the precise revalidate miss (gone / unmapped / no_pages /
        // resolve_fail) instead of a bare `revalidate`, so a benign teardown
        // window is never confused with a genuine live-table content drop.
        log_result(mapping_id, width, height, ImportPresentResult::Fail(reason));
        return ImportPresentResult::Fail(reason);
    }

    // Sample window (offset + guest BPR) — same contract as former write_bgra8.
    let window_started = Instant::now();
    let fmt = state
        .mappings
        .get(&mapping_id)
        .map(|m| {
            if m.format != 0 {
                m.format
            } else {
                MTL_FORMAT_BGRA8_UNORM
            }
        })
        .unwrap_or(MTL_FORMAT_BGRA8_UNORM);
    let Some((base_off, bpr, span_end, from_device)) = state
        .mappings
        .get(&mapping_id)
        .and_then(|m| mapping_write::type11_sample_window_ex(m, width, height, fmt))
    else {
        return import_fail(mapping_id, width, height, ImportDecline::NoSampleWindow);
    };
    let tight = width.saturating_mul(RGBA8_BPP);
    if bpr < tight {
        return import_fail(
            mapping_id,
            width,
            height,
            ImportDecline::BprBelowTight { bpr, tight },
        );
    }

    let need = span_end.max(base_off.saturating_add((bpr as u64).saturating_mul(height as u64)));
    timing.window_us = window_started.elapsed().as_micros() as u64;

    // Pre-DMA guest center row (display-sized surfaces). Packed Stores perform
    // a separate resident diagnostic readback; fragmented Stores reuse the
    // mandatory scatter readback. Comparing them is measurement-only and
    // localizes exact R/B-swapped damage before the Store overwrites the row.
    let guest_before_center = if width >= 1280 && height >= 720 {
        guest_row_bgra_at_y(
            state,
            host,
            mapping_id,
            width,
            height,
            base_off,
            bpr,
            height / 2,
        )
    } else {
        None
    };
    if !host.map_pages_stable() {
        return try_import_present_cpu_unstable(
            state,
            host,
            identity,
            mapping_id,
            width,
            height,
            base_off,
            bpr,
            guest_before_center,
            from_device,
            timing,
        );
    }
    // --- Path 1: packed contig zero-copy ---
    let map_started = Instant::now();
    if let Some((ptr, contig_len)) = mapper::ensure_contig_view(state, host, mapping_id) {
        let map_us = map_started.elapsed().as_micros() as u64;
        if (contig_len as u64) < need {
            return import_fail(
                mapping_id,
                width,
                height,
                ImportDecline::ShortView {
                    contig_len: contig_len as u64,
                    need,
                },
            );
        }
        if (ptr as u64).checked_add(base_off).is_none() {
            return import_fail(mapping_id, width, height, ImportDecline::BaseOffOverflow);
        }
        let dst = (ptr as *mut u8).wrapping_add(base_off as usize) as *mut std::ffi::c_void;
        let dst_len = (contig_len as u64).saturating_sub(base_off);

        // SAFETY: contig view is live for this mapping; revalidate + map_gen gate
        // above; execution is sync-per-packet so no concurrent unmap mid-DMA.
        let dma_started = Instant::now();
        let measure_content = should_measure_resident_content(width, height);
        let res = unsafe {
            engine::present_into_host_ptr_strided(identity, dst, dst_len, bpr, measure_content)
        };
        let dma_us = dma_started.elapsed().as_micros() as u64;
        let post_started = Instant::now();
        // Content stats come from the GPU reduction the store armed (no
        // full-frame CPU readback); split the Result so finish_import still
        // sees the bare Ok/Err. Symmetric with the fragmented path.
        let (res_for_finish, res_stats) = match res {
            Ok(content) => (Ok(()), content.map(ResidentContentStats::from)),
            Err(e) => (Err(e), None),
        };
        // Center-row channel transition: read ONE guest row back out of the
        // pages the GPU just wrote (~7.7 KiB at 1920), not the 8 MiB frame the
        // old full readback handed over for free — same as the fragmented path.
        let transition_stats = if res_for_finish.is_ok() {
            guest_before_center.as_deref().and_then(|before| {
                let after = guest_row_bgra_at_y(
                    state,
                    host,
                    mapping_id,
                    width,
                    height,
                    base_off,
                    bpr,
                    height / 2,
                )?;
                (before.len() == after.len()).then(|| channel_transition_stats_bgra(before, &after))
            })
        } else {
            None
        };
        if let Some(stats) = res_stats {
            #[cfg(test)]
            let _proxy_shared = crate::runtime::census::present_proxy::test_shared();
            crate::runtime::census::present_proxy::note_selected_peer_divergence(
                mapping_id,
                state.present.frame_mapping,
                width,
                height,
                stats.rgb_nz,
            );
        }
        let r = finish_import(
            state,
            mapping_id,
            width,
            height,
            bpr,
            base_off,
            res_for_finish,
            "ok",
        );
        if r.used() {
            let guest =
                guest_center_row_rgb_stats(state, host, mapping_id, width, height, base_off, bpr);
            log_import_content(
                mapping_id,
                width,
                height,
                "ok",
                res_stats,
                guest,
                transition_stats,
                true,
            );
            log_strip_import(
                state,
                host,
                mapping_id,
                width,
                height,
                bpr,
                base_off,
                from_device,
                "ok",
                res_stats,
            );
        }
        let post_us = post_started.elapsed().as_micros() as u64;
        // Per-tranche attribution: present-capture store readback+writeback is
        // not a render draw; count its lock hold out of the opaque `other_us`.
        state
            .tranche
            .note_store(map_us.saturating_add(dma_us).saturating_add(post_us));
        timing.log(
            "contig", mapping_id, width, height, map_us, dma_us, post_us, 1, 0, 0, false,
        );
        // A packed-contig surface needs exactly one window, which looks immune to
        // a budget bound and is not: the budget is global, so one run is refused
        // once the other surfaces have spent it.
        return cpu_writeback_on_window_shortage(
            r,
            1,
            state,
            host,
            identity,
            mapping_id,
            width,
            height,
            base_off,
            bpr,
            guest_before_center,
            from_device,
            timing,
        );
    }

    // --- Path 2: fragmented multi-run zero-copy (no staging) ---
    try_import_present_multi_run(
        state,
        host,
        identity,
        mapping_id,
        width,
        height,
        base_off,
        bpr,
        need,
        guest_before_center,
        from_device,
        full_quad_bounds,
        timing,
    )
}

/// Defer a type-11 import-present Store instead of DMAing now (ack-fast rung).
///
/// The engine resident target is pinned and stays authoritative; the guest
/// window is recorded stale in `state.render_deferred_flush` and lands on the
/// first host-side access (`storage_flush::flush_intersecting`) by replaying
/// [`try_import_present_store`]. Returns `false` (caller performs the
/// synchronous Store) when any gate fails — geometry, generation, sample
/// window, or the engine pin. Lifetime boundaries drop, never write.
pub fn try_defer_present_store<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    identity: &TargetIdentity,
    mapping_id: u32,
    width: u32,
    height: u32,
    full_quad_bounds: bool,
) -> bool {
    // A portability-subset device may use synchronous external-host-memory
    // Store, but guest pages must be authoritative before packet completion.
    if !engine::deferred_gpu_only_content_allowed() {
        return false;
    }
    if !eligible(mapping_id, width, height) {
        return false;
    }
    let Some(m) = state.mappings.get(&mapping_id) else {
        return false;
    };
    // The flush reconstructs the identity via `render_deferred_identity` —
    // defer only when the draw's physical per-member identity is current.
    let expected = surface_identity(state, mapping_id, width, height);
    if !m.mapped || *identity != expected {
        return false;
    }
    let grouped = matches!(identity, TargetIdentity::OutputGroup { .. });
    let member_map_generation = m.map_generation as u64;
    let fmt = if m.format != 0 {
        m.format
    } else {
        MTL_FORMAT_BGRA8_UNORM
    };
    let Some((base_off, bpr, span_end, _)) =
        mapping_write::type11_sample_window_ex(m, width, height, fmt)
    else {
        return false;
    };
    if bpr < width.saturating_mul(RGBA8_BPP) {
        return false;
    }
    // Pin refuses absent / not-ready targets — the sync Store would fail on
    // those too, but it owns the precise fail line. On pin failure older
    // windows stay untouched (the sync fallback supersedes them on success).
    if !engine::pin_resident_target(identity) {
        return false;
    }
    // This deferral supersedes any older deferred window on the mapping (same
    // rule as finish_import). Pins are counted: the pin above holds this
    // window's own count, so every superseded window releases its count —
    // including one on the same identity.
    let mut superseded = 0u64;
    for (key, entry) in state.take_render_deferred_windows(mapping_id, 0, u64::MAX) {
        engine::unpin_resident_target(&render_deferred_identity(key.mapping_id, &entry));
        superseded += 1;
    }
    // Measure-only: these older windows are dropped before any consumer read
    // their guest pages — the fresher defer below replaces them.
    crate::runtime::census::writeback_census::note_superseded(superseded);
    let map_generation = member_map_generation;
    // Async readback prefetch: the composite draws for this resident are already
    // in flight, so submit the GPU→host copy now (dedicated fence, keyed by this
    // monotonic seq) — the flush consumes it once its fence signals, moving the
    // ~8 ms readback wait off the guest packet path. seq==0 means not armed
    // (pool saturated / not ready); the flush then reads synchronously. The
    // strict seq gate is the a/b-glitch guard (see [`engine::prefetch`]).
    //
    state.render_deferred_seq = state.render_deferred_seq.wrapping_add(1);
    let armed_seq = state.render_deferred_seq;
    state.render_deferred_flush.insert(
        crate::model::RenderDeferredKey {
            mapping_id,
            surface_offset: base_off,
            span_end: span_end.max(base_off.saturating_add((bpr as u64) * height as u64)),
        },
        crate::model::RenderDeferredEntry {
            width,
            height,
            map_generation,
            full_quad_bounds,
            grouped,
            armed_seq,
        },
    );
    state.index_deferred_alias_pages(mapping_id);
    // Same protocol bookkeeping as a landed Store (finish_import Ok): the
    // composite exists — its bytes are just resident-side until flushed.
    let _ = state.mark_mapping_written(mapping_id);
    state.note_surface_composite(mapping_id);
    state.note_compositor_member_published(mapping_id, width, height);
    surface_cache::evict(state, mapping_id);
    crate::observe::line(format!(
        "render_writeback_deferred mapping={mapping_id} {width}x{height} gen={map_generation} off={base_off} span_end={span_end}"
    ));
    // Measure-only consume census: this deferred window is now pending. Whether
    // it is later flushed (a consumer read the pages) or superseded (dropped
    // unread) is the signal for whether the writeback is elidable under dmabuf.
    crate::runtime::census::writeback_census::note_armed(state.present.dmabuf_active);
    // Bound the outstanding-window population: each window pins a resident, and
    // a compositing burst (YouTube page-load: many short-lived surfaces deferred
    // faster than consumers read them) otherwise balloons the pinned registry to
    // hundreds — past its slot cap, so the LRU cannot shrink and the non-pinned
    // tail thrashes eviction (measured `cap_pressure reg=332 pinned=323`). Flush
    // the least-recently-armed windows first (proper GPU->guest writeback + unpin
    // via `render_flush_one`, so no content is lost — just landed early) until we
    // are back under the cap. Mirrors the GVA path's `GVA_DEFERRED_WINDOW_CAP`.
    if state.render_deferred_flush.len() > RENDER_DEFERRED_WINDOW_CAP {
        let flush_started = Instant::now();
        let mut forced = 0u64;
        while state.render_deferred_flush.len() > RENDER_DEFERRED_WINDOW_CAP {
            let Some((old_key, old_entry)) = state.take_oldest_render_deferred_window() else {
                break;
            };
            let _ =
                crate::runtime::storage_flush::render_flush_one(state, host, &old_key, &old_entry);
            forced += 1;
        }
        crate::runtime::census::present_proxy::cap_flush::note(
            forced,
            flush_started.elapsed().as_micros() as u64,
        );
    }
    true
}

#[allow(clippy::too_many_arguments)]
fn try_import_present_cpu_unstable<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    identity: &TargetIdentity,
    mapping_id: u32,
    width: u32,
    height: u32,
    base_off: u64,
    bpr: u32,
    guest_before_center: Option<Vec<u8>>,
    from_device: bool,
    timing: ImportTiming,
) -> ImportPresentResult {
    let read_started = Instant::now();
    let pixels = match engine::read_target(identity) {
        Ok(pixels) => pixels,
        Err(e) => {
            let reason = import_fail_reason(&e);
            crate::observe::Emit::decline("import_present", &e)
                .field("used", 0)
                .field("mid", mapping_id)
                .field("geom", format!("{width}x{height}"))
                .field("bpr", bpr)
                .field("base_off", format!("{base_off:#x}"))
                .fail();
            log_result(mapping_id, width, height, ImportPresentResult::Fail(reason));
            log_import_content(mapping_id, width, height, reason, None, None, None, false);
            return ImportPresentResult::Fail(reason);
        }
    };
    let read_us = read_started.elapsed().as_micros() as u64;
    let res_stats = should_measure_resident_content(width, height)
        .then(|| resident_content_stats_from_rgba(&pixels));

    let write_started = Instant::now();
    let stride = width.saturating_mul(RGBA8_BPP);
    if !mapping_write::write_bgra8(state, host, mapping_id, &pixels, stride, width, height) {
        let reason = "cpu_write";
        crate::observe::fail(format!(
            "import_present used=0 reason={reason} mid={mapping_id} {width}x{height} bpr={bpr} base_off={base_off:#x}"
        ));
        log_result(mapping_id, width, height, ImportPresentResult::Fail(reason));
        log_import_content(
            mapping_id, width, height, reason, res_stats, None, None, false,
        );
        return ImportPresentResult::Fail(reason);
    }
    let write_us = write_started.elapsed().as_micros() as u64;

    let post_started = Instant::now();
    let transition_stats = guest_before_center.as_deref().and_then(|before| {
        let after = guest_row_bgra_at_y(
            state,
            host,
            mapping_id,
            width,
            height,
            base_off,
            bpr,
            height / 2,
        )?;
        (before.len() == after.len()).then(|| channel_transition_stats_bgra(before, &after))
    });
    let r = finish_import(
        state,
        mapping_id,
        width,
        height,
        bpr,
        base_off,
        Ok(()),
        "cpu_unstable",
    );
    if r.used() {
        let guest =
            guest_center_row_rgb_stats(state, host, mapping_id, width, height, base_off, bpr);
        log_import_content(
            mapping_id,
            width,
            height,
            "cpu_unstable",
            res_stats,
            guest,
            transition_stats,
            true,
        );
        log_strip_import(
            state,
            host,
            mapping_id,
            width,
            height,
            bpr,
            base_off,
            from_device,
            "cpu_unstable",
            res_stats,
        );
    }
    let post_us = post_started.elapsed().as_micros() as u64;
    state
        .tranche
        .note_store(read_us.saturating_add(write_us).saturating_add(post_us));
    timing.log(
        "cpu_unstable",
        mapping_id,
        width,
        height,
        0,
        read_us,
        write_us.saturating_add(post_us),
        0,
        0,
        0,
        false,
    );
    r
}

/// Fragmented surfaces: map each packed page run and GPU-DMA into it directly.
#[allow(
    clippy::too_many_arguments,
    reason = "the import operation mirrors explicit mapping, surface, and row geometry"
)]
fn try_import_present_multi_run<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    identity: &TargetIdentity,
    mapping_id: u32,
    width: u32,
    height: u32,
    base_off: u64,
    bpr: u32,
    need: u64,
    guest_before_center: Option<Vec<u8>>,
    from_device: bool,
    full_quad_bounds: bool,
    mut timing: ImportTiming,
) -> ImportPresentResult {
    let map_started = Instant::now();
    let Some(gpas) = mapper::mapping_page_gpas(state, host, mapping_id) else {
        return import_fail(mapping_id, width, height, ImportDecline::Revalidate);
    };
    let page_size = state.page_size();
    let page_sz = page_size as usize;
    let span = (gpas.len() as u64).saturating_mul(page_size);
    if span < need {
        return import_fail(
            mapping_id,
            width,
            height,
            ImportDecline::ShortTable { span, need },
        );
    }

    mapper::flush_retired_views(state, host);
    let runs = crate::runtime::gva_view::contig_page_runs(&gpas, page_size);
    if runs.is_empty() {
        return import_fail(mapping_id, width, height, ImportDecline::NoRuns);
    }

    // Map only runs that intersect the sample window; keep (ptr,len,mlo) for GPU.
    struct MappedRun {
        ptr: usize,
        len: usize,
        mlo: u64,
    }
    let mut mapped: Vec<MappedRun> = Vec::new();
    let unmap_all = |host: &mut H, mapped: &[MappedRun]| {
        for m in mapped {
            host.unmap_pages(m.ptr, m.len);
        }
    };

    for run in &runs {
        let run_gpas = &gpas[run.clone()];
        let run_mlo = (run.start as u64).saturating_mul(page_size);
        let run_mhi = (run.end as u64).saturating_mul(page_size);
        if run_mhi <= base_off || run_mlo >= need {
            continue;
        }
        let Some(ptr) = host.map_pages(run_gpas, page_sz) else {
            unmap_all(host, &mapped);
            return import_fail(
                mapping_id,
                width,
                height,
                ImportDecline::MapRunFailed {
                    mlo: run_mlo,
                    pages: run_gpas.len(),
                },
            );
        };
        let len = run_gpas.len().saturating_mul(page_sz);
        mapped.push(MappedRun {
            ptr,
            len,
            mlo: run_mlo,
        });
    }
    if mapped.is_empty() {
        return import_fail(mapping_id, width, height, ImportDecline::NoIntersect);
    }

    let host_runs: Vec<engine::HostMappedRun> = mapped
        .iter()
        .map(|m| engine::HostMappedRun {
            host_ptr: m.ptr as *mut std::ffi::c_void,
            ptr_len: m.len as u64,
            linear_base: m.mlo,
            linear_len: m.len as u64,
        })
        .collect();
    let map_us = map_started.elapsed().as_micros() as u64;

    // SAFETY: mapped views remain live and disjoint until the synchronous
    // readback/scatter returns; product drain cannot unmap them concurrently.
    let dma_started = Instant::now();
    let measure_content = should_measure_resident_content(width, height);
    let res = unsafe {
        engine::present_into_host_runs(
            identity,
            base_off,
            bpr,
            &host_runs,
            measure_content,
            host.map_pages_stable(),
        )
    };
    let dma_us = dma_started.elapsed().as_micros() as u64;
    let post_started = Instant::now();
    unmap_all(host, &mapped);

    let result = match res {
        Ok(present) => {
            // Phase split of the runs_readback DMA: readback (sync GPU
            // render/readback fence-wait + copy-out) vs CPU scatter into the
            // fragmented guest runs. Names which half of the ~11 ms/store lag
            // Per-store scatter timing — one line per import store, entirely
            // wall-clock (SCHED_IDLE-contaminated under the agent harness).
            // Diagnostic-only; gated behind REIMS_VGPU_DRAW_LOG. The always-on
            // store-rate signal is `present_import` `used_hz`.
            crate::observe::line(format!(
                "import_store_split mid={mapping_id} {width}x{height} dma_us={dma_us} scatter_us={} runs={}",
                present.scatter_us,
                mapped.len()
            ));
            // Always-on store-path census: the GPU-direct scatter is now the
            // only writeback, so a rise in decline counters is a hard failure
            // (the store returns Err), not a silent degrade.
            let (gpu_stores, fb_unres, fb_submit) = engine::host_scatter_snapshot();
            crate::runtime::census::present_proxy::store_scatter::note(
                true, gpu_stores, fb_unres, fb_submit,
            );
            let resident_stats_started = Instant::now();
            let res_stats = present.content.map(ResidentContentStats::from);
            // Before/after channel transition of the centre row. The frame no
            // longer exists on the CPU, so read the "after" back out of the
            // guest pages the GPU just wrote — ONE row (~7.7 KiB at 1920), not
            // the 8 MiB frame the old readback handed over for free.
            let transition_stats = guest_before_center.as_deref().and_then(|before| {
                let after = guest_row_bgra_at_y(
                    state,
                    host,
                    mapping_id,
                    width,
                    height,
                    base_off,
                    bpr,
                    height / 2,
                )?;
                (before.len() == after.len()).then(|| channel_transition_stats_bgra(before, &after))
            });
            timing.resident_stats_us = resident_stats_started.elapsed().as_micros() as u64;
            if let Some(stats) = res_stats {
                #[cfg(test)]
                let _proxy_shared = crate::runtime::census::present_proxy::test_shared();
                crate::runtime::census::present_proxy::note_selected_peer_divergence(
                    mapping_id,
                    state.present.frame_mapping,
                    width,
                    height,
                    stats.rgb_nz,
                );
                // Incomplete-swap-base class marker: a draw
                // whose decoded geometry spans the full target should move the
                // resident content stats; byte-identical rgb_nz after a
                // full-quad pass names the "fade deposits nothing" disease.
                // Measure-only — legitimate zero-opacity transition frames may
                // fire a bounded handful of these during a fade.
                let prev = state.import_rgb_nz.insert(mapping_id, stats.rgb_nz);
                if full_quad_bounds && prev == Some(stats.rgb_nz) {
                    crate::observe::fail(format!(
                        "fullquad_store_noop mid={mapping_id} {width}x{height} rgb_nz={} (full-quad draw left resident content stats unchanged)",
                        stats.rgb_nz
                    ));
                }
            }
            let _ = state.mark_mapping_written(mapping_id);
            state.note_surface_composite(mapping_id);
            // Full-frame publish: the entire resident target was scattered into
            // the guest page runs covering the whole w×h window, so guest pages
            // now hold a complete guest-rendered frame for this mapping. That
            // is the protocol-level completeness proof compositor-output
            // membership needs — full-coverage *draw* edges are one-shot
            // transition events that miss a double-buffer half on snapshot
            // boots (2026-07-16 census: mid 5 stored 41 full frames, zero
            // edges, present pin frozen on mid 1).
            state.note_compositor_member_published(mapping_id, width, height);
            surface_cache::evict(state, mapping_id);
            // Success census: the import was used, fires once per fragmented
            // present (~one per present under compositing, ~77k/session under a
            // continuously-animating app). Count into the windowed
            // `present_import` summary and gate the per-present line behind
            // REIMS_VGPU_DRAW_LOG — the `used=0 reason=<slug>` fallbacks stay fail-visible.
            crate::runtime::census::present_proxy::present_import::note(true, false);
            if crate::observe::draw_log_enabled() {
                crate::observe::line(format!(
                    "import_present used=1 reason=ok_runs mid={mapping_id} {width}x{height} runs={}",
                    host_runs.len()
                ));
            }
            let guest =
                guest_center_row_rgb_stats(state, host, mapping_id, width, height, base_off, bpr);
            log_import_content(
                mapping_id,
                width,
                height,
                "ok_runs",
                res_stats,
                guest,
                transition_stats,
                true,
            );
            log_strip_import(
                state,
                host,
                mapping_id,
                width,
                height,
                bpr,
                base_off,
                from_device,
                "ok_runs",
                res_stats,
            );
            ImportPresentResult::Ok
        }
        Err(e) => {
            let reason = import_fail_reason(&e);
            crate::observe::Emit::decline("import_present", &e)
                .field("used", 0)
                .field("mid", mapping_id)
                .field("geom", format!("{width}x{height}"))
                .field("bpr", bpr)
                .field("base_off", format!("{base_off:#x}"))
                .fail();
            // The shared readback itself failed, so resident diagnostics are
            // unavailable; the specific transfer reason remains fail-visible.
            log_import_content(mapping_id, width, height, reason, None, None, None, false);
            ImportPresentResult::Fail(reason)
        }
    };
    let post_us =
        (post_started.elapsed().as_micros() as u64).saturating_sub(timing.resident_stats_us);
    // Per-tranche attribution: fragmented present-capture store readback+writeback
    // is not a render draw; count its lock hold out of the opaque `other_us`.
    state
        .tranche
        .note_store(map_us.saturating_add(dma_us).saturating_add(post_us));
    timing.log(
        "runs_readback",
        mapping_id,
        width,
        height,
        map_us,
        dma_us,
        post_us,
        host_runs.len(),
        0,
        0,
        false,
    );
    cpu_writeback_on_window_shortage(
        result,
        host_runs.len(),
        state,
        host,
        identity,
        mapping_id,
        width,
        height,
        base_off,
        bpr,
        guest_before_center,
        from_device,
        timing,
    )
}

/// Re-route a host-import **window shortage** to the CPU writeback, or return
/// `result` untouched.
///
/// The host-import windows are a **cache**, and a cache running out is not a
/// reason to lose the guest's work.
///
/// Whether a writeback can go through host-pointer DMA is asked once, far above
/// the import, as `host.map_pages_stable()` — a property of the *host*. That is
/// the right question for a capability and the wrong one here: the scatter
/// resolves every run of the surface before its DMA and is all-or-nothing, so
/// what actually decides is how many import windows *this* surface needs against
/// how many are free *now*. A live x86 boot measured a working set of ten 1 GiB
/// buckets against a budget of eight, thrashing `creates=1771 evictions=1763`
/// with victims at `age_ms=0`, and dropped seven full renders as
/// `deferred_flush_lost reason=host_import_total_byte_cap` while
/// [`try_import_present_cpu_unstable`] sat correct and unreachable behind the
/// capability gate.
///
/// A capability or structural refusal must NOT re-route: the CPU path would fail
/// those too, and the loss has to stay visible rather than being retried into a
/// second failure. [`is_import_window_shortage`] draws that line.
///
/// This demotes the import budget from a correctness bound to a performance one,
/// which is what a cache budget is supposed to be. Raising the budget instead
/// cannot work in principle — a surface straddling more buckets than the budget
/// fails at every setting.
///
/// Both import paths end here rather than each testing the result themselves.
/// The contig path was added second and stayed a drop for one commit, which is
/// what a duplicated decision costs: `runs` is the only thing the two sites
/// disagree about, so it is a parameter and the decision is not.
///
/// Confirmed live on an x86/Vulkan boot driven with six heavy pages in separate
/// windows, scrolls and rubber-band drags, once the thrash signature this exists
/// for had actually been reached (12 distinct 1 GiB buckets against the budget of
/// eight, `creates=845 evictions=846`): **14 shortages, 14 absorptions, 1:1, and
/// zero `deferred_flush_lost reason=host_import_total_byte_cap`**. The absorbed
/// surfaces were mostly fragmented — three full 1920x1080 scanouts at `runs=505`
/// among them — and the two losses that remained were `map_generation_drift`,
/// which is the guest rewiring the pages under a deferred window and is correct
/// to drop. The comparable pre-fix boot lost seven renders and absorbed none.
///
/// Reaching that state is the hard part and it is not a knob on the drive script:
/// the budget is spent by page *spread*, so it needs several large surfaces live
/// at once. Sustained frame rate alone does not do it, and a guest sitting at the
/// login window produces a perfectly healthy-looking log with zero of everything.
#[allow(
    clippy::too_many_arguments,
    reason = "forwards the import operation's mapping, surface, and row geometry unchanged"
)]
fn cpu_writeback_on_window_shortage<H: HostMemory + HostOps>(
    result: ImportPresentResult,
    runs: usize,
    state: &mut DeviceState,
    host: &mut H,
    identity: &TargetIdentity,
    mapping_id: u32,
    width: u32,
    height: u32,
    base_off: u64,
    bpr: u32,
    guest_before_center: Option<Vec<u8>>,
    from_device: bool,
    timing: ImportTiming,
) -> ImportPresentResult {
    let ImportPresentResult::Fail(reason) = result else {
        return result;
    };
    if !is_import_window_shortage(reason) {
        return result;
    }
    crate::observe::fail(format!(
        "import_window_cpu_writeback mid={mapping_id} {width}x{height} \
         runs={runs} reason={reason} (host-import budget exhausted; \
         writing the render back on the CPU instead of dropping it)"
    ));
    try_import_present_cpu_unstable(
        state,
        host,
        identity,
        mapping_id,
        width,
        height,
        base_off,
        bpr,
        guest_before_center,
        from_device,
        timing,
    )
}

/// True when an import refusal was a shortage of **our own** host-import
/// windows, rather than something the CPU writeback would hit as well.
///
/// Compared against the typed slugs rather than string literals so that renaming
/// a cause moves both sides together. The two admitted here are the budget
/// bounds; every other [`HostImportDecline`] is a capability, alignment, or
/// range fact that a retry cannot change.
fn is_import_window_shortage(reason: &str) -> bool {
    use crate::backend::vulkan::engine::HostImportDecline;
    use crate::observe::Decline;
    reason == HostImportDecline::TotalBytes.slug() || reason == HostImportDecline::RegionCount.slug()
}

/// Why the runtime refused a guest-page import before the engine was asked.
///
/// # Why these are prefixed
///
/// Bare, two of them belonged to another rail as well: `unmapped` and
/// `short_view` were also the console-capture rail's words for *different*
/// checks, so `grep reason=unmapped` returned a mix of "the guest paged the
/// surface off" (capture) and "this mapping is not mapped" (import) and could not
/// be read. The `import_` prefix is the same fix the slate reasons and the MRT
/// proxies took; crate-wide distinctness is `observe::gate`'s job.
///
/// A `Skip` is deliberately **not** in here: "no mid, zero geometry" is
/// control flow, not a refusal, and typing it would invite logging it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ImportDecline {
    /// A per-mid `Surface` identity whose embedded generation no longer matches
    /// the mapping's, so the frame we would DMA belongs to a superseded
    /// incarnation. Per-mid only — a group identity carries no member lifetime
    /// and its generation is constant, so it can never fail this way; that case
    /// is [`Self::GroupGeomDrift`].
    MapGenDrift,
    /// An `OutputGroup` identity this mapping no longer resolves to — its group
    /// membership or geometry moved. Despite sharing a gate with
    /// [`Self::MapGenDrift`], no generation is compared here (a group identity's
    /// is constant), so the two must not share a slug: one says the mapping was
    /// re-wired under us, the other says the pinned resident is not the one this
    /// Store's geometry names.
    ///
    /// This arm used to re-resolve group *membership* as a stand-in for the
    /// lifetime a group identity cannot carry, and refused whenever the mapping
    /// no longer resolved to the group. Measurement killed that: every observed
    /// refusal was a swapchain buffer whose membership had been pruned by a site
    /// that deliberately keeps the generation and the deferred window, and which
    /// the compositor re-granted tens of milliseconds later. The lifetime is
    /// checked by a real generation compare before this call, so what is left
    /// here is only the geometry.
    GroupGeomDrift,
    /// The mapping is not mapped.
    Unmapped,
    /// No type-11 sample window could be derived for this geometry.
    NoSampleWindow,
    /// The descriptor's row stride is narrower than a tight RGBA row.
    BprBelowTight { bpr: u32, tight: u32 },
    /// The contig host view is shorter than the frame the import needs.
    ShortView { contig_len: u64, need: u64 },
    /// `host_ptr + base_off` overflowed, so the destination is unrepresentable.
    BaseOffOverflow,
    /// The mapping's page table could not be revalidated.
    Revalidate,
    /// The mapping's pages span less than the frame needs.
    ShortTable { span: u64, need: u64 },
    /// The page list yielded no contiguous runs.
    NoRuns,
    /// A run's pages could not be mapped into a host view.
    MapRunFailed { mlo: u64, pages: usize },
    /// No run intersects the sample window.
    NoIntersect,
}

impl crate::observe::Decline for ImportDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::MapGenDrift => "import_map_gen_drift",
            Self::GroupGeomDrift => "import_group_geom_drift",
            Self::Unmapped => "import_unmapped",
            Self::NoSampleWindow => "import_no_sample_window",
            Self::BprBelowTight { .. } => "import_bpr_below_tight",
            Self::ShortView { .. } => "import_short_view",
            Self::BaseOffOverflow => "import_base_off_overflow",
            Self::Revalidate => "import_revalidate",
            Self::ShortTable { .. } => "import_short_table",
            Self::NoRuns => "import_no_runs",
            Self::MapRunFailed { .. } => "import_map_run_failed",
            Self::NoIntersect => "import_no_intersect",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::BprBelowTight { bpr, tight } => {
                vec![("bpr", bpr.to_string()), ("tight", tight.to_string())]
            }
            Self::ShortView { contig_len, need } => vec![
                ("contig_len", contig_len.to_string()),
                ("need", need.to_string()),
            ],
            Self::ShortTable { span, need } => {
                vec![("span", span.to_string()), ("need", need.to_string())]
            }
            Self::MapRunFailed { mlo, pages } => {
                vec![("mlo", format!("{mlo:#x}")), ("pages", pages.to_string())]
            }
            _ => Vec::new(),
        }
    }
}

/// Log one import refusal and return it.
///
/// Every site used to write its reason **twice** — once into `log_result` and
/// once into the returned `Fail` — which is a shape that lets the logged reason
/// and the returned one drift apart silently. One argument now feeds both.
fn import_fail(mid: u32, w: u32, h: u32, d: ImportDecline) -> ImportPresentResult {
    use crate::observe::Decline as _;
    crate::runtime::census::present_proxy::present_import::note(false, true);
    crate::observe::Emit::decline("import_present", &d)
        .field("used", 0)
        .field("mid", mid)
        .field("geom", format!("{w}x{h}"))
        .fail();
    ImportPresentResult::Fail(d.slug())
}

/// Name the check that refused a host present, **from the type**.
///
/// # What this replaced
///
/// Three separate `e.to_string().contains(…)` ladders — one per import path —
/// mapped the engine's *prose* onto four coarse buckets (`no_resident`,
/// `run_gap`, `run_oob`, `vk_err`, `no_ext`, `short_import`, `cpu_read`). That
/// made a `DrawError` payload's wording load-bearing behaviour with no test and
/// no gate over it, and one branch was already dead: nothing in the crate
/// produces a payload containing `external_memory_host`, because that check had
/// been typed into `DrawReason::PresentHostPtrImportUnavailable`, whose slug
/// does not contain the substring. Every host-pointer present failure on a host
/// without `VK_EXT_external_memory_host` was reported as a driver error — a
/// misdiagnosis that lands on the two non-DMA rows of the support matrix, which
/// is why nobody booting this project's two hosts ever saw it.
///
/// The classification is now `DrawError`'s own slug, so a new check is named by
/// construction rather than by remembering to extend a ladder. The reasons it
/// yields are *more* specific than the buckets, not fewer: `run_gap` covered
/// three distinct layout faults that now name themselves.
fn import_fail_reason(e: &crate::backend::vulkan::engine::DrawError) -> &'static str {
    use crate::observe::Decline as _;
    e.slug()
}

#[allow(
    clippy::too_many_arguments,
    reason = "the completion helper accounts for each import result and timing component"
)]
fn finish_import(
    state: &mut DeviceState,
    mapping_id: u32,
    width: u32,
    height: u32,
    bpr: u32,
    base_off: u64,
    res: Result<(), crate::backend::vulkan::engine::DrawError>,
    ok_reason: &'static str,
) -> ImportPresentResult {
    match res {
        Ok(()) => {
            // A landed Store supersedes any older deferred render window on
            // this mapping — the DMA'd content is strictly newer, so a later
            // flush replaying the old identity would corrupt it. (No-op on
            // the flush path itself: the window was taken before replay.)
            let mut superseded = 0u64;
            for (key, entry) in state.take_render_deferred_windows(mapping_id, 0, u64::MAX) {
                engine::unpin_resident_target(&render_deferred_identity(key.mapping_id, &entry));
                superseded += 1;
                crate::observe::off(format!(
                    "render_deferred_superseded mapping={} {}x{} gen={} grouped={}",
                    key.mapping_id,
                    entry.width,
                    entry.height,
                    entry.map_generation,
                    entry.grouped as u8
                ));
            }
            crate::runtime::census::writeback_census::note_superseded(superseded);
            let _ = state.mark_mapping_written(mapping_id);
            state.note_surface_composite(mapping_id);
            // Full-frame publish — same membership grant as the multi-run path
            // (contiguous swap buffers must not re-open the pin-freeze class).
            state.note_compositor_member_published(mapping_id, width, height);
            // Capture prefers host_cache over guest pages — must evict so the
            // next paint_mapping reads the DMA'd guest BGRA.
            surface_cache::evict(state, mapping_id);
            let _ = ok_reason;
            log_result(mapping_id, width, height, ImportPresentResult::Ok);
            ImportPresentResult::Ok
        }
        Err(e) => {
            let reason = import_fail_reason(&e);
            crate::observe::Emit::decline("import_present", &e)
                .field("used", 0)
                .field("mid", mapping_id)
                .field("geom", format!("{width}x{height}"))
                .field("bpr", bpr)
                .field("base_off", format!("{base_off:#x}"))
                .fail();
            ImportPresentResult::Fail(reason)
        }
    }
}

fn log_result(mid: u32, w: u32, h: u32, r: ImportPresentResult) {
    // Always count into the windowed `present_import` summary (drain worker,
    // off-main-core). used=1 is ~1/present (~77k/session under animation) — a
    // raw flood on both schedulers, redundant with the summary count — so gate
    // its per-present census behind REIMS_VGPU_DRAW_LOG. Skip/Fail (`used=0`) stay
    // fail-visible with their reason.
    let is_fail = matches!(r, ImportPresentResult::Fail(_));
    crate::runtime::census::present_proxy::present_import::note(r.used(), is_fail);
    if r.used() {
        if crate::observe::draw_log_enabled() {
            crate::observe::line(format!(
                "import_present used=1 reason={} mid={mid} {w}x{h}",
                r.reason()
            ));
        }
    } else {
        crate::observe::fail(format!(
            "import_present used=0 reason={} mid={mid} {w}x{h}",
            r.reason()
        ));
    }
}

#[cfg(test)]
mod tests {
    /// The retry decision this module makes must partition the eight host-import
    /// causes by **whether a retry can possibly help**, not by how they read.
    ///
    /// Getting it wrong is silent both ways. Admit a capability or alignment
    /// cause and every such refusal is retried into a second, slower failure
    /// while the original loss stops being named. Omit a budget cause and the
    /// render is dropped for a shortage of our own cache — the defect this
    /// routing exists to end, which cost seven full renders on one measured x86
    /// boot.
    #[test]
    fn only_our_own_window_shortages_are_worth_a_cpu_retry() {
        use crate::backend::vulkan::engine::HostImportDecline;
        use crate::observe::Decline;

        // A retry frees windows and can then succeed: these are *our* limits.
        for shortage in [HostImportDecline::TotalBytes, HostImportDecline::RegionCount] {
            assert!(
                is_import_window_shortage(shortage.slug()),
                "{} is a budget bound and must re-route to the CPU writeback",
                shortage.slug()
            );
        }

        // Nothing a retry can change: the extension is absent for the device's
        // lifetime, and alignment/range are facts about the span itself.
        for hard in [
            HostImportDecline::ExtensionAbsent,
            HostImportDecline::ZeroLength,
            HostImportDecline::PointerMisaligned { host_ptr: 0x1001, alignment: 4096 },
            HostImportDecline::SizeMisaligned { size: 0x2001, alignment: 4096 },
            HostImportDecline::RangeOverflow { host_ptr: usize::MAX, len: 0x1000 },
            HostImportDecline::NoValidWindow { host_ptr: 0x1000, len: 0x1000, alignment: 4096 },
        ] {
            assert!(
                !is_import_window_shortage(hard.slug()),
                "{} cannot be fixed by retrying and must stay a visible loss",
                hard.slug()
            );
        }

        // An unrelated reason from elsewhere in the chain must not be admitted
        // just because it mentions imports.
        assert!(!is_import_window_shortage("host_import_resolve"));
        assert!(!is_import_window_shortage("map_generation_drift"));
    }

    use super::*;
    use crate::backend::vulkan::engine::reason::{DrawReason, HostPresentDecline};
    use crate::backend::vulkan::engine::DrawError;
    use crate::model::{DeviceId, DeviceState, PAGE_SHIFT_X86};

    /// **The defect this classification replaced.** A host without
    /// `VK_EXT_external_memory_host` refuses the packed-contig import at the
    /// extension check, and the old ladder tested
    /// `e.to_string().contains("external_memory_host")` for it — a substring no
    /// payload in the crate produces, because that check had already been typed
    /// into a `DrawReason` whose slug does not contain it. So the branch was dead
    /// and every such failure was reported as a driver error, which is a
    /// misdiagnosis on exactly the two non-DMA rows of the support matrix.
    ///
    /// This asserts the classification names the *extension*, and it fails
    /// against the prose ladder.
    #[test]
    fn a_missing_host_import_extension_is_not_reported_as_a_driver_error() {
        /// The ladder this replaced, kept *only* as the proof that it was
        /// broken. Do not reintroduce it.
        fn classify_on_prose(e: &DrawError) -> &'static str {
            if e.to_string().contains("external_memory_host") {
                "no_ext"
            } else if e.to_string().contains("ptr_len") {
                "short_import"
            } else if e.to_string().contains("unknown identity")
                || e.to_string().contains("no ready content")
            {
                "no_resident"
            } else {
                "vk_err"
            }
        }
        let e = DrawError::Unsupported(DrawReason::PresentHostPtrImportUnavailable);
        // The bug, pinned: the prose ladder could not reach its own `no_ext`.
        assert_eq!(classify_on_prose(&e), "vk_err");
        // The fix: the check names itself.
        assert_eq!(
            import_fail_reason(&e),
            "present_host_ptr_import_unavailable"
        );
    }

    /// The three layout faults the old ladder collapsed into one `run_gap` are
    /// three different bugs — a hole in the row walk, a mis-ordered run table,
    /// and a run that claims more than it maps — so they must not share a name.
    #[test]
    fn the_layout_faults_that_shared_run_gap_now_name_themselves() {
        let of = |d| import_fail_reason(&DrawError::Present(d));
        let names = [
            of(HostPresentDecline::RunsUncoveredRow { row: 3 }),
            of(HostPresentDecline::RunsOutOfOrder { index: 1 }),
            of(HostPresentDecline::RunsLenExceedsPtr {
                linear_len: 8,
                ptr_len: 4,
            }),
        ];
        let mut sorted = names;
        sorted.sort_unstable();
        let before = sorted.len();
        let mut dedup = sorted.to_vec();
        dedup.dedup();
        assert_eq!(
            before,
            dedup.len(),
            "{names:?} collapsed to one reason again"
        );
        assert!(names.iter().all(|n| *n != "run_gap"));
    }

    /// Every runtime-side import reason names its rail, and none of them
    /// collides with the console-capture rail any more.
    ///
    /// `unmapped` and `short_view` each had **two** claimants — this rail and the
    /// capture rail, for different checks — so `grep reason=unmapped` over one
    /// boot returned a mix and could not be read. That is the same defect
    /// crate-wide uniqueness caught between the two MRT proxies.
    #[test]
    fn every_import_reason_names_its_rail_and_is_distinct() {
        use crate::observe::Decline as _;
        const ALL: &[ImportDecline] = &[
            ImportDecline::MapGenDrift,
            ImportDecline::GroupGeomDrift,
            ImportDecline::Unmapped,
            ImportDecline::NoSampleWindow,
            ImportDecline::BprBelowTight { bpr: 0, tight: 0 },
            ImportDecline::ShortView {
                contig_len: 0,
                need: 0,
            },
            ImportDecline::BaseOffOverflow,
            ImportDecline::Revalidate,
            ImportDecline::ShortTable { span: 0, need: 0 },
            ImportDecline::NoRuns,
            ImportDecline::MapRunFailed { mlo: 0, pages: 0 },
            ImportDecline::NoIntersect,
        ];
        let mut slugs: Vec<&str> = Vec::new();
        for d in ALL {
            assert!(
                d.slug().starts_with("import_"),
                "{} is not namespaced to the import rail",
                d.slug()
            );
            for (k, v) in d.fields() {
                assert!(!k.contains(' ') && !v.contains(' '), "{k}={v}");
            }
            slugs.push(d.slug());
        }
        slugs.sort_unstable();
        let before = slugs.len();
        slugs.dedup();
        assert_eq!(before, slugs.len(), "duplicate ImportDecline slug");
    }

    /// **A `Skip` is not a refusal.** "No mid, zero geometry" is control flow —
    /// the import was never attempted — so it must stay outside the decline
    /// vocabulary, or the fail log gains a line per ineligible present.
    #[test]
    fn a_skip_is_not_a_decline() {
        use crate::observe::Refusal as _;
        assert_eq!(ImportPresentResult::Skip("ineligible").refusal(), None);
        assert_eq!(ImportPresentResult::Ok.refusal(), None);
        assert_eq!(
            ImportPresentResult::Fail("import_no_runs").refusal(),
            Some("import_no_runs")
        );
    }

    /// `DrawError` delegates a typed present refusal without replacing its
    /// reason or structured row-byte facts.
    #[test]
    fn draw_error_preserves_the_typed_present_decline() {
        use crate::backend::vulkan::engine::reason::HostPresentDecline;
        use crate::observe::Decline as _;
        let e = DrawError::Present(HostPresentDecline::HostPtrBadRowBytes {
            row_bytes: 7,
            tight: 16,
        });
        assert_eq!(e.slug(), "host_ptr_bad_row_bytes");
        assert_eq!(
            e.fields(),
            vec![("row_bytes", "7".to_string()), ("tight", "16".to_string())]
        );
    }

    #[test]
    fn drop_render_deferred_windows_releases_a_mappings_windows_only() {
        use crate::model::{RenderDeferredEntry, RenderDeferredKey};
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let entry = RenderDeferredEntry {
            width: 8,
            height: 8,
            map_generation: 1,
            full_quad_bounds: false,
            grouped: false,
            armed_seq: 0,
        };
        let rkey = |mapping_id: u32, lo: u64, hi: u64| RenderDeferredKey {
            mapping_id,
            surface_offset: lo,
            span_end: hi,
        };
        // Two windows on the unmapped surface (7) plus one on a survivor (8).
        state.render_deferred_flush.insert(rkey(7, 0, 256), entry);
        state.render_deferred_flush.insert(rkey(7, 256, 512), entry);
        state.render_deferred_flush.insert(rkey(8, 0, 256), entry);

        // Unmap-time eager drop clears exactly mapping 7's windows (unpin is a
        // no-op here — no live engine registry — but the leak the fix targets is
        // precisely these lingering *windows* pinning residents past cap).
        let dropped = drop_render_deferred_windows(&mut state, 7);
        assert_eq!(dropped, 2, "both mapping-7 windows released");
        assert_eq!(state.render_deferred_flush.len(), 1);
        assert!(state.render_deferred_flush.contains_key(&rkey(8, 0, 256)));

        // Idempotent: a second drop on the same mapping releases nothing.
        assert_eq!(drop_render_deferred_windows(&mut state, 7), 0);
    }

    #[test]
    fn take_oldest_render_deferred_window_is_least_recently_armed() {
        use crate::model::{RenderDeferredEntry, RenderDeferredKey};
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let entry = |mapping_id: u32, armed_seq: u64| {
            (
                RenderDeferredKey {
                    mapping_id,
                    surface_offset: 0,
                    span_end: 256,
                },
                RenderDeferredEntry {
                    width: 8,
                    height: 8,
                    map_generation: 1,
                    full_quad_bounds: false,
                    grouped: false,
                    armed_seq,
                },
            )
        };
        // Insert out of key order to prove ordering is by armed_seq, not the
        // BTreeMap key: mapping 5 armed first (seq 1), 9 last (seq 3).
        for (k, e) in [entry(9, 3), entry(5, 1), entry(7, 2)] {
            state.render_deferred_flush.insert(k, e);
        }
        let seqs: Vec<u64> = (0..3)
            .map(|_| {
                state
                    .take_oldest_render_deferred_window()
                    .unwrap()
                    .1
                    .armed_seq
            })
            .collect();
        assert_eq!(seqs, vec![1, 2, 3], "oldest-first by armed_seq");
        assert!(state.render_deferred_flush.is_empty());
        assert!(state.take_oldest_render_deferred_window().is_none());
    }

    #[test]
    fn import_content_proxy_reports_resident_alpha_occupancy() {
        let px = [
            1, 0, 0, 0, // RGB content with zero destination alpha.
            0, 0, 0, 1, // Non-zero, non-opaque alpha.
            0, 2, 0, 255, // Opaque alpha.
        ];
        let stats = resident_content_stats_from_rgba(&px);
        assert_eq!(
            stats,
            ResidentContentStats {
                rgb_nz: 2,
                rgb_max: 2,
                alpha_nz: 2,
                alpha_opaque: 1,
            }
        );
        let line = import_content_line(
            5,
            1920,
            1080,
            "ok_runs",
            Some(stats),
            Some((1920, 255)),
            Some(ChannelTransitionStats {
                changed_px: 7,
                rb_swap_px: 6,
            }),
        );
        for field in [
            "res_rgb_nz=2",
            "res_max=2",
            "res_alpha_nz=2",
            "res_alpha_opaque=1",
            "guest_row_nz=1920",
            "guest_row_max=255",
            "changed_row_px=7",
            "rb_swap_row_px=6",
        ] {
            assert!(line.contains(field), "missing {field}: {line}");
        }
    }

    #[test]
    fn resident_content_measurement_is_bounded_to_display_and_menu_strip() {
        assert!(should_measure_resident_content(1920, 1080));
        assert!(should_measure_resident_content(1920, 24));
        assert!(!should_measure_resident_content(715, 625));
        assert!(!should_measure_resident_content(31, 24));
    }

    #[test]
    fn channel_transition_proxy_distinguishes_exact_rb_swap_damage() {
        let before = [
            10, 20, 30, 255, // exact B/R exchange below
            40, 50, 60, 128, // arbitrary change below
            70, 80, 70, 255, // equal B/R must not count as a swap
            1, 2, 3, 4, // unchanged
        ];
        let after = [
            30, 20, 10, 255, 41, 52, 63, 128, 70, 80, 70, 255, 1, 2, 3, 4,
        ];
        assert_eq!(
            channel_transition_stats_bgra(&before, &after),
            ChannelTransitionStats {
                changed_px: 2,
                rb_swap_px: 1,
            }
        );
    }
    use crate::runtime::host::FakeHost;

    #[test]
    fn eligible_requires_mid_and_geom() {
        assert!(!eligible(0, 1920, 1080));
        assert!(!eligible(1, 0, 1080));
        assert!(!eligible(1, 320, 0));
        assert!(eligible(1, 1, 1));
        assert!(eligible(1, 1920, 1080));
    }

    #[test]
    fn strip_geom_uses_present_proxy_classifier() {
        // Keep import enrich predicate aligned with present_proxy census.
        assert!(crate::runtime::census::present_proxy::is_menu_strip_geom(
            1920, 24
        ));
        assert!(!crate::runtime::census::present_proxy::is_menu_strip_geom(
            1920, 1080
        ));
    }

    #[test]
    fn import_store_timing_proxy_names_every_phase_and_threshold() {
        assert!(!import_store_timing_is_slow(
            IMPORT_STORE_TIMING_SLOW_US - 1
        ));
        assert!(import_store_timing_is_slow(IMPORT_STORE_TIMING_SLOW_US));
        let line = import_store_timing_line(
            "runs_readback",
            3,
            1920,
            1080,
            11,
            1,
            2,
            3,
            4,
            5,
            6,
            507,
            0,
            0,
            false,
        );
        for field in [
            "path=runs_readback",
            "mid=3",
            "1920x1080",
            "total_us=11",
            "revalidate_us=1",
            "window_us=2",
            "resident_stats_us=3",
            "map_us=4",
            "dma_us=5",
            "post_us=6",
            "runs=507",
            "cache_hits=0",
            "cache_misses=0",
            "cached=0",
            "threshold_us=1000",
        ] {
            assert!(line.contains(field), "missing {field}: {line}");
        }
    }

    #[test]
    fn surface_identity_tracks_map_generation() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        state.map_surface(3);
        let id1 = surface_identity(&state, 3, 1920, 1080);
        let _ = state.invalidate_mapping_pages(3);
        let id2 = surface_identity(&state, 3, 1920, 1080);
        assert_ne!(id1.generation(), id2.generation());
    }

    /// The lifetime gate has two arms and they fail for different reasons: a
    /// per-mid `Surface` identity fails when the mapping's generation moved
    /// under it, and an `OutputGroup` identity fails when the pinned resident is
    /// not the size this Store names, with no generation compared at all (a
    /// group identity's is constant). They shared `import_map_gen_drift`, which
    /// named only the first and made the largest surviving render-loss reason
    /// unattributable between the two.
    #[test]
    fn the_lifetime_gate_names_which_of_its_two_arms_refused() {
        use crate::runtime::host::FakeHost;
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        let member = |w, h| crate::model::CompositorOutputMember {
            width: w,
            height: h,
            source: 13,
        };
        for mid in [1u32, 5u32] {
            state.map_surface(mid);
            state.present.compositor_output_members.insert(mid, member(1920, 1080));
            state.note_presented_geom(mid, 1920, 1080);
        }
        let group = surface_identity(&state, 1, 1920, 1080);
        assert!(matches!(group, TargetIdentity::OutputGroup { .. }));
        // The grouped arm refuses on geometry: the pinned resident is 1920x1080
        // and this Store names 640x480, so it is not the image to write.
        assert_eq!(
            try_import_present_store(&mut state, &mut host, &group, 1, 640, 480, false).reason(),
            "import_group_geom_drift",
            "a pinned resident of a different size is not a generation drift"
        );
        // ...and it does NOT refuse on membership. Mapping 1 is presented at a
        // new geometry, so it no longer resolves to the group — the state the
        // gate used to reject. It now falls through to the checks that read the
        // mapping itself, which is what the deletion claims.
        state.note_presented_geom(1, 640, 480);
        assert_ne!(surface_identity(&state, 1, 1920, 1080), group);
        let reason =
            try_import_present_store(&mut state, &mut host, &group, 1, 1920, 1080, false).reason();
        assert_ne!(
            reason, "import_group_geom_drift",
            "membership is not the grouped arm's business any more"
        );
        assert!(
            reason.starts_with("revalidate_"),
            "the flush must reach the checks that read the mapping, got {reason}"
        );
        // The other arm, unchanged: a per-mid identity whose generation moved.
        let per_mid = surface_identity(&state, 1, 1920, 1080);
        assert!(matches!(per_mid, TargetIdentity::Surface { .. }));
        {
            let m = state.mappings.get_mut(&1).unwrap();
            DeviceState::bump_map_generation(1, m);
        }
        assert_eq!(
            try_import_present_store(&mut state, &mut host, &per_mid, 1, 1920, 1080, false).reason(),
            "import_map_gen_drift"
        );
    }

    /// Group admission fails in three distinguishable states, and the miss
    /// carries which one from the check that refused.
    ///
    /// This is what retired the membership re-resolve from the import gate.
    /// `condemn_surface_backing` and a fresh `map_surface` both prune
    /// `presented_geoms` while deliberately *keeping* `map_generation` and the
    /// mapping's deferred windows, so a deferred flush's recycled-pages guard
    /// passes and the gate was the only thing refusing — against a window the
    /// layer below had just certified as the same incarnation. `unmap_surface`
    /// prunes the same way but bumps the generation, so the lifetime layer
    /// refuses first and the gate never decides. One slug covered all of it.
    ///
    /// The reading still feeds `import_group_membership_stale`, which is now
    /// measure-only: the write proceeds, and the line is what would locate a
    /// regression from letting it.
    #[test]
    fn the_grouped_arm_reports_which_membership_reading_refused() {
        use crate::model::OutputGroupMiss;
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let member = |w, h| crate::model::CompositorOutputMember {
            width: w,
            height: h,
            source: 13,
        };
        for mid in [1u32, 5u32] {
            state.map_surface(mid);
            state
                .present
                .compositor_output_members
                .insert(mid, member(1920, 1080));
            state.note_presented_geom(mid, 1920, 1080);
        }
        assert_eq!(state.output_group_resolve(1, 1920, 1080), Ok(1));

        // A geometry only ever presented from one surface: no peer to unify.
        state.note_presented_geom(7, 800, 600);
        assert_eq!(
            state.output_group_resolve(7, 800, 600),
            Err(OutputGroupMiss::NoPeer)
        );

        // A resize: still presented, elsewhere. The reading names where.
        state.note_presented_geom(5, 640, 480);
        assert_eq!(
            state.output_group_resolve(5, 1920, 1080),
            Err(OutputGroupMiss::NotPresentedHere {
                presented: Some((640, 480))
            })
        );

        // A fresh MAP does NOT move the lifetime — it stashes the page plan as
        // the incarnation fingerprint and lets the next resolve decide. Assert
        // that precondition first, because it is what makes keeping the
        // qualification correct rather than merely permissive.
        let gen_before = state.mappings.get(&1).unwrap().map_generation;
        state.map_surface(1);
        assert_eq!(
            state.mappings.get(&1).unwrap().map_generation,
            gen_before,
            "a fresh MAP defers the lifetime decision to the fingerprint compare"
        );
        assert_eq!(
            state.output_group_resolve(1, 1920, 1080),
            Ok(1),
            "an undecided recycle must not demote a proven swapchain member"
        );

        // `condemn_surface_backing` is the other undecided route, same answer.
        let mut condemned = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        for mid in [2u32, 3u32] {
            condemned.map_surface(mid);
            condemned
                .present
                .compositor_output_members
                .insert(mid, member(1920, 1080));
            condemned.note_presented_geom(mid, 1920, 1080);
            condemned.mappings.get_mut(&mid).unwrap().page_entries = vec![0x301];
        }
        let gen_before = condemned.mappings.get(&2).unwrap().map_generation;
        assert!(condemned.condemn_surface_backing(2));
        assert_eq!(
            condemned.mappings.get(&2).unwrap().map_generation,
            gen_before,
            "a condemned backing keeps content state for the fingerprint compare"
        );
        assert_eq!(condemned.output_group_resolve(2, 1920, 1080), Ok(1));

        // The contrast that makes those two preconditions mean something:
        // `unmap_surface` DOES bump the generation, so the evidence expires on
        // its own and the recycled id must re-earn its qualification. Same
        // observable state as the two above under the old unconditional prune —
        // which is exactly why they had to become distinguishable.
        let gen_before = condemned.mappings.get(&3).unwrap().map_generation;
        condemned.unmap_surface(3);
        let gen_now = condemned.mappings.get(&3).unwrap().map_generation;
        assert_ne!(
            gen_now, gen_before,
            "an unmap is a lifetime boundary, not a deferred decision"
        );
        assert_eq!(
            condemned.output_group_resolve(3, 1920, 1080),
            Err(OutputGroupMiss::PriorIncarnation {
                presented: (1920, 1080),
                entry_gen: gen_before,
                current_gen: gen_now,
            })
        );

        // `output_group_for` is this resolve's `.ok()`, so the admission
        // decision cannot drift from the reading that explains it.
        for (mid, w, h) in [(1u32, 1920u32, 1080u32), (5, 1920, 1080), (7, 800, 600)] {
            assert_eq!(
                state.output_group_for(mid, w, h),
                state.output_group_resolve(mid, w, h).ok()
            );
        }
    }

    /// A swapchain buffer recycled onto an IDENTICAL page plan must keep
    /// resolving to the output group — this is the black-desktop class.
    ///
    /// `objects.rs`'s type-4 attach calls `map_surface` and only *afterwards*
    /// compares the fresh page plan against the stashed fingerprint. On an
    /// identical plan that compare declines to bump `map_generation`, on the
    /// stated grounds that this is the SAME incarnation and the resident and
    /// deferred windows survive. But `map_surface` had already dropped the
    /// mapping's present evidence, so `output_group_for` demoted it and every
    /// draw until its next present landed on a private per-mid resident that
    /// had never held the accumulated full frame. Measured on the rail: the
    /// resident's non-zero pixel count collapsed from 1.84 M to 0.62-0.78 M of
    /// 2.07 M and never recovered, and the screen kept only the damaged rects.
    #[test]
    fn a_recycle_onto_the_same_plan_does_not_demote_a_swapchain_member() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        for mid in [1u32, 5u32] {
            state.map_surface(mid);
            state.mappings.get_mut(&mid).unwrap().page_entries = vec![0x301];
            state.note_presented_geom(mid, 1920, 1080);
        }
        let group = surface_identity(&state, 1, 1920, 1080);
        assert!(
            matches!(group, TargetIdentity::OutputGroup { .. }),
            "two presented members at one geometry are a proven swapchain"
        );

        // The recycle. `map_surface` stashes the page plan as the incarnation
        // fingerprint and does NOT bump — assert that precondition, because it
        // is the whole reason the demotion was wrong rather than correct.
        let gen_before = state.mappings.get(&1).unwrap().map_generation;
        state.map_surface(1);
        assert_eq!(
            state.mappings.get(&1).unwrap().map_generation,
            gen_before,
            "an identical-plan recycle is the same incarnation"
        );
        assert_eq!(
            surface_identity(&state, 1, 1920, 1080),
            group,
            "a member the fingerprint compare calls the SAME incarnation must \
             not be demoted to a private resident"
        );

        // The converse, so this is not just \"never demote\": a recycle onto a
        // DIFFERENT plan bumps the generation, and the id must re-earn its
        // qualification rather than inherit the predecessor's.
        {
            let m = state.mappings.get_mut(&1).unwrap();
            DeviceState::bump_map_generation(1, m);
        }
        assert!(
            matches!(
                surface_identity(&state, 1, 1920, 1080),
                TargetIdentity::Surface { id: 1, .. }
            ),
            "a genuinely new surface must not inherit present evidence"
        );
        assert!(matches!(
            state.output_group_resolve(1, 1920, 1080),
            Err(crate::model::OutputGroupMiss::PriorIncarnation { .. })
        ));

        // And a surface that was NEVER presented stays out however much it
        // publishes — the Safari-scroll black-band guard, unchanged.
        state.map_surface(7);
        state
            .present
            .compositor_output_members
            .insert(7, crate::model::CompositorOutputMember {
                width: 1920,
                height: 1080,
                source: 13,
            });
        assert!(matches!(
            surface_identity(&state, 7, 1920, 1080),
            TargetIdentity::Surface { id: 7, .. }
        ));
    }

    /// Two proven same-geometry compositor members resolve to one shared
    /// OutputGroup identity.
    #[test]
    fn surface_identity_unifies_compositor_members() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        for mid in [1u32, 5u32, 7u32] {
            state.map_surface(mid);
        }
        let member = |w, h| crate::model::CompositorOutputMember {
            width: w,
            height: h,
            source: 13,
        };
        // One member only: per-mid.
        state
            .present
            .compositor_output_members
            .insert(1, member(1920, 1080));
        state.note_presented_geom(1, 1920, 1080);
        assert!(matches!(
            surface_identity(&state, 1, 1920, 1080),
            TargetIdentity::Surface { id: 1, .. }
        ));
        // Second member at the same geometry but publish-only (never
        // presented): both stay per-mid — the black-band regression guard.
        // WebKit content tiles satisfy membership by full-frame publishing;
        // unifying them chains DISTINCT surfaces onto one resident.
        state
            .present
            .compositor_output_members
            .insert(5, member(1920, 1080));
        assert!(matches!(
            surface_identity(&state, 1, 1920, 1080),
            TargetIdentity::Surface { id: 1, .. }
        ));
        assert!(matches!(
            surface_identity(&state, 5, 1920, 1080),
            TargetIdentity::Surface { id: 5, .. }
        ));
        // Present evidence for the second member unifies the logical output.
        state.note_presented_geom(5, 1920, 1080);
        let a = surface_identity(&state, 1, 1920, 1080);
        let b = surface_identity(&state, 5, 1920, 1080);
        assert!(matches!(a, TargetIdentity::OutputGroup { .. }));
        assert_eq!(a, b, "members share one logical framebuffer identity");
        // Non-member at the same geometry: per-mid.
        assert!(matches!(
            surface_identity(&state, 7, 1920, 1080),
            TargetIdentity::Surface { id: 7, .. }
        ));
        // Member at a different geometry than proven: per-mid.
        assert!(matches!(
            surface_identity(&state, 1, 1280, 720),
            TargetIdentity::Surface { id: 1, .. }
        ));
    }

    /// A proven swapchain geometry keeps unifying a lone presented member across
    /// the buffer recycles that momentarily leave a single concurrently
    /// presented member. This is the black-background / desktop-residue class:
    /// WindowServer continuously recycles swapchain buffers (fresh mid ids, old
    /// ones unmapped), so the *concurrent* peer count drops to one; without the
    /// sticky `output_group_geoms` latch the fresh buffer resolved to a per-mid
    /// resident that never held the accumulated full frame, and the guest's
    /// damage-only draw left everything outside the damaged rect black.
    #[test]
    fn proven_swapchain_geometry_unifies_a_lone_recycled_member() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        for mid in [1u32, 5u32, 9u32] {
            state.map_surface(mid);
        }
        let member = |w, h| crate::model::CompositorOutputMember {
            width: w,
            height: h,
            source: 13,
        };
        // Two members presented together: the geometry is a proven swapchain.
        state
            .present
            .compositor_output_members
            .insert(1, member(1920, 1080));
        state
            .present
            .compositor_output_members
            .insert(5, member(1920, 1080));
        state.note_presented_geom(1, 1920, 1080);
        state.note_presented_geom(5, 1920, 1080);
        assert!(matches!(
            surface_identity(&state, 1, 1920, 1080),
            TargetIdentity::OutputGroup { .. }
        ));

        // WindowServer recycles the swapchain: the old buffers are gone and a
        // fresh buffer (mid 9) is the only member presented at the geometry.
        for mid in [1u32, 5u32] {
            state.present.compositor_output_members.remove(&mid);
            state.present.presented_geoms.remove(&mid);
        }
        state
            .present
            .compositor_output_members
            .insert(9, member(1920, 1080));
        state.note_presented_geom(9, 1920, 1080);

        // The lone fresh buffer still resolves to the shared OutputGroup
        // resident (per-mid resolution here is the black-background bug).
        assert!(
            matches!(
                surface_identity(&state, 9, 1920, 1080),
                TargetIdentity::OutputGroup { .. }
            ),
            "a proven swapchain geometry stays unified across buffer recycles"
        );

        // A geometry that was never double-buffered stays per-mid: a lone
        // member at a fresh resolution must not spuriously unify.
        state
            .present
            .compositor_output_members
            .insert(9, member(1280, 720));
        state.note_presented_geom(9, 1280, 720);
        assert!(matches!(
            surface_identity(&state, 9, 1280, 720),
            TargetIdentity::Surface { id: 9, .. }
        ));
    }

    /// Detector for the mixed-generation frame class.
    ///
    /// `ResourcePools::registry` is keyed by `TargetIdentity`, so a resident
    /// that is NOT keyed by the mapping id that resolved to it is reachable
    /// from another surface too — and distinct guest surfaces have independent
    /// damage histories, so sharing one resident fuses their damage into a
    /// single frame. `surface_mapping_id()` is the O(1) predicate the always-on
    /// `resident_identity_shared` line is built on. Two unified members trip it
    /// today; per-surface residents cannot.
    #[test]
    fn shared_resident_is_detectable_from_the_identity_alone() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        for mid in [1u32, 5u32] {
            state.map_surface(mid);
        }
        let member = crate::model::CompositorOutputMember {
            width: 1920,
            height: 1080,
            source: 13,
        };
        // A lone member stays per-surface: keyed by its own mapping id.
        state.present.compositor_output_members.insert(1, member);
        state.note_presented_geom(1, 1920, 1080);
        assert_eq!(
            surface_identity(&state, 1, 1920, 1080).surface_mapping_id(),
            Some(1),
            "a per-surface resident belongs to the surface that asked for it"
        );

        // A second presented member at the same geometry unifies both onto one
        // resident, and neither identity names the surface that resolved to it.
        state.present.compositor_output_members.insert(5, member);
        state.note_presented_geom(5, 1920, 1080);
        let a = surface_identity(&state, 1, 1920, 1080);
        let b = surface_identity(&state, 5, 1920, 1080);
        assert_eq!(a, b, "two distinct guest surfaces reached one resident");
        assert_eq!(a.surface_mapping_id(), None);
        assert_eq!(b.surface_mapping_id(), None);
    }

    /// `resolves_to_same_resident` must answer exactly what comparing two
    /// `surface_identity` results answers, in every state the ClearOnly peer
    /// census can observe. Asserted against real `surface_identity` calls rather
    /// than against a restatement of the unification rule, so the two cannot
    /// drift: the census's whole value is that "the peer is a different name for
    /// the same image" means what the render path means by it.
    #[test]
    fn same_resident_agrees_with_surface_identity() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        for mid in [1u32, 5u32, 9u32] {
            state.map_surface(mid);
        }
        let member = crate::model::CompositorOutputMember {
            width: 1920,
            height: 1080,
            source: 13,
        };
        for mid in [1u32, 5u32, 9u32] {
            state.present.compositor_output_members.insert(mid, member);
        }
        let agrees = |state: &DeviceState, a: u32, b: u32| {
            let want = surface_identity(state, a, 1920, 1080) == surface_identity(state, b, 1920, 1080);
            assert_eq!(
                resolves_to_same_resident(state, a, b, 1920, 1080),
                want,
                "mids {a},{b} disagree with surface_identity"
            );
            want
        };

        // Nothing presented yet: every mid is its own resident, and a mid is
        // always the same resident as itself.
        assert!(!agrees(&state, 1, 5));
        assert!(agrees(&state, 1, 1));

        // One presented surface is still a lone member — no group to share.
        state.note_presented_geom(1, 1920, 1080);
        assert!(!agrees(&state, 1, 5));

        // Two presented surfaces latch the geometry, and both resolve to the
        // shared group: a peer hop between them cannot change the image.
        state.note_presented_geom(5, 1920, 1080);
        assert!(agrees(&state, 1, 5));

        // A mid that never qualified stays private even at the latched
        // geometry, so a hop onto it DOES change the image.
        assert!(!agrees(&state, 1, 9));
        assert!(!agrees(&state, 9, 5));

        // …and joins once the guest names it in a display transaction.
        state.note_presented_geom(9, 1920, 1080);
        assert!(agrees(&state, 1, 9));
    }

    /// The direct-present fallback identity is always per-member.
    #[test]
    fn member_surface_identity_is_always_per_mid_even_when_unified() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        for mid in [1u32, 5u32] {
            state.map_surface(mid);
        }
        let member = crate::model::CompositorOutputMember {
            width: 1920,
            height: 1080,
            source: 13,
        };
        state.present.compositor_output_members.insert(1, member);
        state.present.compositor_output_members.insert(5, member);
        state.note_presented_geom(1, 1920, 1080);
        state.note_presented_geom(5, 1920, 1080);
        assert!(matches!(
            surface_identity(&state, 1, 1920, 1080),
            TargetIdentity::OutputGroup { .. }
        ));
        let g1 = state.mappings.get(&1).unwrap().map_generation as u64;
        assert_eq!(
            member_surface_identity(&state, 1, 1920, 1080),
            TargetIdentity::Surface {
                id: 1,
                width: 1920,
                height: 1080,
                generation: g1,
            }
        );
        // A distinct member gets its own distinct key (never collapsed).
        assert_ne!(
            member_surface_identity(&state, 1, 1920, 1080),
            member_surface_identity(&state, 5, 1920, 1080)
        );
    }

    /// A grouped deferred window rebuilds the OutputGroup identity it pinned
    /// (never re-resolved against current membership); per-mid windows keep
    /// the Surface identity with the member's map_generation.
    #[test]
    fn render_deferred_identity_rebuilds_pinned_identity() {
        let entry = |grouped| crate::model::RenderDeferredEntry {
            width: 1920,
            height: 1080,
            map_generation: 7,
            full_quad_bounds: false,
            grouped,
            armed_seq: 0,
        };
        assert_eq!(
            render_deferred_identity(4, &entry(false)),
            TargetIdentity::Surface {
                id: 4,
                width: 1920,
                height: 1080,
                generation: 7
            }
        );
        assert_eq!(
            render_deferred_identity(4, &entry(true)),
            TargetIdentity::OutputGroup {
                id: OUTPUT_GROUP_ID,
                width: 1920,
                height: 1080,
                generation: 0
            }
        );
    }

    /// The predicate test above says which reasons *ought* to re-route. This
    /// says the routing actually happens, which is a separate claim: the contig
    /// path shipped with the predicate already correct and still dropped its
    /// renders, because it never asked.
    ///
    /// Observable without an engine. A shortage leaves the import path for
    /// [`try_import_present_cpu_unstable`], which cannot succeed in a unit test —
    /// but it fails at `read_target` with a *different* reason, so "the returned
    /// reason is no longer the byte cap" is exactly the assertion that the
    /// re-route was taken. A hard refusal must come back byte-identical.
    #[test]
    fn a_window_shortage_leaves_the_import_path_and_a_hard_refusal_does_not() {
        use crate::backend::vulkan::engine::HostImportDecline;
        use crate::observe::Decline as _;

        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::default();
        // Distinct from every other mid in this module: the fail log is a single
        // shared file and the assertion below greps it.
        let mid = 23063u32;
        let id = TargetIdentity::Surface {
            id: mid,
            width: 2,
            height: 2,
            generation: 0,
        };
        // The fail log appends across boots *and* across test runs, so a line
        // this test wrote on an earlier run would satisfy the assertion below
        // with the routing deleted. Read only what this run appends.
        let log_path = crate::observe::fail_log_path();
        let mark = std::fs::metadata(log_path).map(|m| m.len()).unwrap_or(0) as usize;

        let mut route = |result| {
            cpu_writeback_on_window_shortage(
                result,
                1,
                &mut state,
                &mut host,
                &id,
                mid,
                2,
                2,
                0,
                8,
                None,
                false,
                ImportTiming::new(),
            )
        };

        // Not a failure at all: passes through untouched.
        assert_eq!(route(ImportPresentResult::Ok), ImportPresentResult::Ok);

        // A capability refusal the CPU path would hit too: the loss stays named.
        let hard = HostImportDecline::ExtensionAbsent.slug();
        assert_eq!(
            route(ImportPresentResult::Fail(hard)),
            ImportPresentResult::Fail(hard),
            "a refusal a retry cannot fix must be returned unchanged, not retried"
        );

        // Our own budget: must leave this path rather than be returned as a drop.
        let cap = HostImportDecline::TotalBytes.slug();
        assert_ne!(
            route(ImportPresentResult::Fail(cap)),
            ImportPresentResult::Fail(cap),
            "a shortage of our own windows must re-route to the CPU writeback"
        );

        let whole = std::fs::read_to_string(log_path).expect("fail log");
        let appended = &whole[mark.min(whole.len())..];
        let line = |reason: &str| format!("import_window_cpu_writeback mid={mid} 2x2 runs=1 reason={reason}");
        assert!(
            appended.contains(&line(cap)),
            "the re-route must name itself and the check that refused"
        );
        assert!(
            !appended.contains(&line(hard)),
            "a hard refusal must not be logged as a re-route"
        );
    }

    #[test]
    fn import_fails_closed_when_unmapped() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::default();
        let id = TargetIdentity::Surface {
            id: 1,
            width: 1920,
            height: 1080,
            generation: 0,
        };
        let r = try_import_present_store(&mut state, &mut host, &id, 1, 1920, 1080, false);
        assert!(matches!(r, ImportPresentResult::Fail(_)));
    }
}
