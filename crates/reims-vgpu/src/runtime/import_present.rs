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
}

impl ImportTiming {
    fn new() -> Self {
        Self {
            started: Instant::now(),
            revalidate_us: 0,
            window_us: 0,
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
    map_us: u64,
    dma_us: u64,
    post_us: u64,
    runs: usize,
    cache_hits: usize,
    cache_misses: usize,
    cached: bool,
) -> String {
    format!(
        "import_store_timing path={path} mid={mapping_id} {width}x{height} total_us={total_us} revalidate_us={revalidate_us} window_us={window_us} map_us={map_us} dma_us={dma_us} post_us={post_us} runs={runs} cache_hits={cache_hits} cache_misses={cache_misses} cached={} threshold_us={IMPORT_STORE_TIMING_SLOW_US}",
        cached as u8
    )
}

/// Build a protocol-stable resident identity for this mapping at its current
/// [`crate::model::MappingEntry::map_generation`].
///
/// One identity per mapping, always. `ResourcePools::registry` is keyed by
/// `TargetIdentity`, so two mappings with equal identities would render into and
/// capture from ONE `VkImage` — and distinct guest surfaces have independent
/// damage histories, because WindowServer redraws a buffer only where it differs
/// from what THAT buffer last held. Sharing a resident between them makes every
/// frame a fusion of damage from several buffers, which is the rubber-band
/// residue class.
pub fn surface_identity(
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

/// Type-11 Store with non-zero geom — the only product Store class that imports.
pub fn eligible(mapping_id: u32, width: u32, height: u32) -> bool {
    mapping_id != 0 && width > 0 && height > 0
}

/// Rebuild exactly the resident identity a deferred render window pinned at
/// defer time (see [`try_defer_present_store`]). Never re-resolves against
/// current state: whatever moved between defer and flush must not unpin a
/// different image than was pinned.
pub fn render_deferred_identity(
    mapping_id: u32,
    entry: &crate::model::RenderDeferredEntry,
) -> TargetIdentity {
    TargetIdentity::Surface {
        id: mapping_id,
        width: entry.width,
        height: entry.height,
        generation: entry.map_generation,
    }
}

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
    let gen_now = state
        .mappings
        .get(&mapping_id)
        .map(|m| m.map_generation as u64)
        .unwrap_or(0);
    let gate_fail = (identity.generation() != gen_now).then_some(ImportDecline::MapGenDrift);
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
    let Some((base_off, bpr, span_end)) = state
        .mappings
        .get(&mapping_id)
        .and_then(|m| mapping_write::type11_sample_window(m, width, height, fmt))
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
        let res = unsafe { engine::present_into_host_ptr_strided(identity, dst, dst_len, bpr) };
        let dma_us = dma_started.elapsed().as_micros() as u64;
        let post_started = Instant::now();
        let res_for_finish = res;
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
    let member_map_generation = m.map_generation as u64;
    let fmt = if m.format != 0 {
        m.format
    } else {
        MTL_FORMAT_BGRA8_UNORM
    };
    let Some((base_off, bpr, span_end)) = mapping_write::type11_sample_window(m, width, height, fmt)
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
            armed_seq,
        },
    );
    state.index_deferred_alias_pages(mapping_id);
    // Same protocol bookkeeping as a landed Store (finish_import Ok): the
    // composite exists — its bytes are just resident-side until flushed.
    let _ = state.mark_mapping_written(mapping_id);
    state.note_surface_composite(mapping_id);
    state.note_dense_frame_published(mapping_id, width, height);
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
            return ImportPresentResult::Fail(reason);
        }
    };
    let read_us = read_started.elapsed().as_micros() as u64;
    let write_started = Instant::now();
    let stride = width.saturating_mul(RGBA8_BPP);
    if !mapping_write::write_bgra8(state, host, mapping_id, &pixels, stride, width, height) {
        let reason = "cpu_write";
        crate::observe::fail(format!(
            "import_present used=0 reason={reason} mid={mapping_id} {width}x{height} bpr={bpr} base_off={base_off:#x}"
        ));
        log_result(mapping_id, width, height, ImportPresentResult::Fail(reason));
        return ImportPresentResult::Fail(reason);
    }
    let write_us = write_started.elapsed().as_micros() as u64;

    let post_started = Instant::now();
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
    timing: ImportTiming,
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
    let res = unsafe {
        engine::present_into_host_runs(identity, base_off, bpr, &host_runs, host.map_pages_stable())
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
            state.note_dense_frame_published(mapping_id, width, height);
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
    };
    let post_us = post_started.elapsed().as_micros() as u64;
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
    /// The identity's embedded generation no longer matches the mapping's, so
    /// the frame we would DMA belongs to a superseded incarnation.
    MapGenDrift,
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
                    "render_deferred_superseded mapping={} {}x{} gen={}",
                    key.mapping_id,
                    entry.width,
                    entry.height,
                    entry.map_generation,
                ));
            }
            crate::runtime::census::writeback_census::note_superseded(superseded);
            let _ = state.mark_mapping_written(mapping_id);
            state.note_surface_composite(mapping_id);
            // Full-frame publish — same membership grant as the multi-run path
            // (contiguous swap buffers must not re-open the pin-freeze class).
            state.note_dense_frame_published(mapping_id, width, height);
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
    use crate::runtime::host::FakeHost;
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
    fn eligible_requires_mid_and_geom() {
        assert!(!eligible(0, 1920, 1080));
        assert!(!eligible(1, 0, 1080));
        assert!(!eligible(1, 320, 0));
        assert!(eligible(1, 1, 1));
        assert!(eligible(1, 1920, 1080));
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

    /// **Two guest surfaces never reach one resident.** This is the rubber-band
    /// residue fix, stated as an invariant rather than as the absence of the
    /// mechanism that broke it.
    ///
    /// `ResourcePools::registry` is keyed by `TargetIdentity`, so two mappings
    /// with equal identities render into and capture from ONE `VkImage`. Guest
    /// surfaces have independent damage histories — WindowServer redraws a
    /// buffer only where it differs from what THAT buffer last held — so a
    /// shared resident makes a damage-only draw composite over the wrong base
    /// and strands whatever the other buffer left there.
    ///
    /// Four scanout buffers at one geometry used to collapse onto a single
    /// `TargetIdentity::OutputGroup`, and a held drag that reverses direction
    /// left a selection-rectangle fragment on the desktop in about half of
    /// rounds. Interleaved on/off A/B over four boots: 5 of 12 rounds with the
    /// collapse, 1 of 12 without, and the dominant sub-class (a 15x15 fragment
    /// at the press point) went 4 to 0.
    ///
    /// `surface_mapping_id()` is the O(1) predicate: every identity a mapping
    /// resolves to must name that mapping.
    #[test]
    fn every_presented_surface_keeps_a_resident_of_its_own() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mids = [1u32, 3, 5, 7];
        for mid in mids {
            state.map_surface(mid);
        }
        // Present all four at one geometry — a compositor swapchain, which is
        // exactly the shape that used to unify.
        for mid in mids {
            state.note_presented_geom(mid, 1920, 1080);
            assert!(state.presented_at(mid, 1920, 1080));
        }
        let ids: Vec<_> = mids
            .iter()
            .map(|&mid| surface_identity(&state, mid, 1920, 1080))
            .collect();
        for (mid, id) in mids.iter().zip(&ids) {
            assert_eq!(
                id.surface_mapping_id(),
                Some(*mid),
                "a resident must belong to the surface that asked for it"
            );
        }
        for (i, a) in ids.iter().enumerate() {
            for b in &ids[i + 1..] {
                assert_ne!(a, b, "two scanout buffers must not share one resident");
            }
        }
    }

    /// A deferred window rebuilds the identity it pinned, carrying the member's
    /// `map_generation` — never re-resolved against current state, so nothing
    /// that moves between defer and flush can unpin a different image.
    #[test]
    fn render_deferred_identity_rebuilds_pinned_identity() {
        let entry = crate::model::RenderDeferredEntry {
            width: 1920,
            height: 1080,
            map_generation: 7,
            armed_seq: 0,
        };
        assert_eq!(
            render_deferred_identity(4, &entry),
            TargetIdentity::Surface {
                id: 4,
                width: 1920,
                height: 1080,
                generation: 7
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
        let r = try_import_present_store(&mut state, &mut host, &id, 1, 1920, 1080);
        assert!(matches!(r, ImportPresentResult::Fail(_)));
    }
}
