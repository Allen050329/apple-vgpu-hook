//! Persistent Vulkan draw + compute engine for the Linux metal2vulkan product path.
//!
//! Facade: [`execute_draw`] / [`execute_draw_request`] / [`execute_compute`] /
//! [`read_target`]. Caches L2–L7 + Lc + memory pools so a warm
//! identical static key performs zero `vkCreate*` and zero `vkAllocateMemory` on
//! the product path.

#![allow(unsafe_op_in_unsafe_fn)]

mod caches;
mod compute_execution;
mod compute_validation;
mod content_stats;
mod context;
mod counters;
mod desc_arena;
mod device_lost;
mod digest;
mod draw_execution;
mod draw_preparation;
mod draw_validation;
// `pub(crate)` so the host-window present thread ([[host-window]] direct-present
// route B) can import the engine's exported scanout dmabuf on its own device.
pub(crate) mod dmabuf_export;
mod exec;
mod exec_compute;
mod facade_decline;
pub mod fd_dup;
mod host_import_decline;
pub(crate) mod host_scatter;
pub mod init_decline;
mod pools;
pub(crate) mod present_stats_spv;
pub mod reason;
mod slab;
pub(crate) mod stats_reduce;
pub mod types;
pub mod vk_call;
#[cfg(all(feature = "host-window", target_os = "macos"))]
mod window_present;

pub use compute_execution::ComputeExecutionDecline;
pub use compute_validation::ComputeValidationDecline;
pub use content_stats::Color8ContentStats;
pub use context::{FENCE_TIMEOUT_NS, MAX_DEVICE_RECREATES};
pub use counters::{CounterSnapshot, EngineCounters};
pub use device_lost::{DeviceLostDecline, DeviceLostOp};
pub use draw_execution::DrawExecutionDecline;
pub use draw_preparation::DrawPreparationDecline;
pub use draw_validation::DrawValidationDecline;
pub use facade_decline::EngineFacadeDecline;
use fd_dup::{FdDupDecline, FdDupRail};
pub use host_import_decline::HostImportDecline;
pub use init_decline::InitDecline;
pub use reason::DrawReason;
pub use types::{
    BlendFactor, BlendOp, BlendStateResource, BufferContent, ComputeBufferOutput,
    ComputeBufferResource, ComputeHostWriteback, ComputeOutput, ComputeRequest,
    ComputeResidentSampleBind, ComputeSampledImageResource, ComputeStorageImageResource,
    ComputeStorageResidency, CullMode, DepthState, DrawError, DrawOutput, DrawRequest, DrawTicket,
    GuestRun, GuestRunSource, IndexType, IndexedDrawResource, LoadOp, PrimitiveTopology,
    SampledContentIdentity, SampledImageResource, SampledSource, SamplerAddressMode,
    SamplerBorderColor, SamplerCompareFunction, SamplerFilter, SamplerMipFilter, SamplerResource,
    ScissorResource, SecondaryColorTarget, StencilFaceOps, StencilOp, StencilState,
    StorageBufferResource, StorageImageFormat, TargetIdentity, VertexAttributeFormat,
    VertexAttributeResource, VertexStepFunction, ViewportResource, WindowPresentSource,
    COLOR_INPUT_BINDING,
};
pub use vk_call::{VkCall, VkOp};
#[cfg(all(feature = "host-window", target_os = "macos"))]
pub use window_present::WindowPresentOutcome;

use caches::ObjectCaches;
use content_stats::copy_color8_with_stats_to_vec;
use context::ContextOwner;
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use pools::ResourcePools;
use std::os::fd::{FromRawFd, IntoRawFd};
use std::sync::atomic::Ordering;
use types::ComputeError;

pub type OptionalExportedPresent = (Option<i32>, u64, u32, u32, usize);
type ResidentReadback = (Vec<u8>, Option<Color8ContentStats>);

struct EngineState {
    owner: ContextOwner,
    caches: ObjectCaches,
    pools: ResourcePools,
    counters: EngineCounters,
    /// One cached exportable dmabuf scanout image (reused across presents; only
    /// re-created on a geometry change). Empty until the GL/dmabuf scanout path
    /// runs. Tied to the current device context — dropped on device recreate /
    /// guest reset alongside the other device-derived state.
    scanout_export: dmabuf_export::ScanoutExportCache,
    /// Ring of exportable dmabuf images for the direct-present path (host window
    /// route B): a fresh slot each present so the engine never overwrites the
    /// slot the window is mid-blit reading. Empty until `export_present_from_
    /// resident` runs; torn down with the context like `scanout_export`.
    present_export_ring: dmabuf_export::ScanoutExportRing,
    #[cfg(all(feature = "host-window", target_os = "macos"))]
    window_presenter: Option<window_present::WindowPresenter>,
}

impl EngineState {
    fn new() -> Self {
        Self {
            owner: ContextOwner::new(),
            caches: ObjectCaches::new(),
            pools: ResourcePools::new(),
            counters: EngineCounters::default(),
            scanout_export: dmabuf_export::ScanoutExportCache::new(),
            present_export_ring: dmabuf_export::ScanoutExportRing::new(),
            #[cfg(all(feature = "host-window", target_os = "macos"))]
            window_presenter: None,
        }
    }

    fn flush_device_derived(&mut self) {
        if let Some(ctx) = self.owner.ctx.as_ref() {
            unsafe {
                #[cfg(all(feature = "host-window", target_os = "macos"))]
                if let Some(mut presenter) = self.window_presenter.take() {
                    presenter.destroy(ctx, Some(&mut self.pools));
                }
                if let Some(fd) = self.scanout_export.destroy(&ctx.device) {
                    // Close via OwnedFd drop (crate idiom; libc is Apple-only).
                    drop(std::os::fd::OwnedFd::from_raw_fd(fd));
                }
                for fd in self.present_export_ring.destroy(&ctx.device) {
                    drop(std::os::fd::OwnedFd::from_raw_fd(fd));
                }
                self.caches.destroy_all(&ctx.device);
                self.pools.destroy_all(&ctx.device);
            }
        } else {
            self.caches.clear_logical();
        }
        self.pools = ResourcePools::new();
        self.caches = ObjectCaches::new();
    }
}

static ENGINE: Lazy<Mutex<EngineState>> = Lazy::new(|| Mutex::new(EngineState::new()));

// Engine-lock contention census. The single global `ENGINE` mutex serializes all
// 34 engine entry points across the drain worker AND the QEMU main/present path.
// Under a heavy-composite burst (dock magnification + a constant-animation tab)
// the per-bind sample resolver's `resident_content_ready` acquisition can block
// behind a present-path holder — a wall-clock cost that looks like "resolve work"
// but is pure lock wait. These accumulators (all-atomic, no extra lock) let the
// off-main-core census distinguish the two: if `wait_ns` per acquisition spikes
// during a freeze, the drain worker is lock-bound, not compute-bound.
static ENGINE_LOCK_WAIT_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static ENGINE_LOCK_ACQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static ENGINE_LOCK_MAX_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Acquire the global engine lock, accumulating acquisition-wait time for the
/// contention census. One `Instant::now()` pair per lock (~20 ns via the vDSO
/// clock) — negligible vs the work under the lock, and the whole point is to
/// catch the case where the *wait* dwarfs the work.
#[inline]
fn lock_engine() -> parking_lot::MutexGuard<'static, EngineState> {
    let t = std::time::Instant::now();
    let guard = ENGINE.lock();
    let ns = t.elapsed().as_nanos() as u64;
    ENGINE_LOCK_WAIT_NS.fetch_add(ns, Ordering::Relaxed);
    ENGINE_LOCK_ACQ.fetch_add(1, Ordering::Relaxed);
    let mut cur = ENGINE_LOCK_MAX_NS.load(Ordering::Relaxed);
    while ns > cur {
        match ENGINE_LOCK_MAX_NS.compare_exchange_weak(
            cur,
            ns,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => break,
            Err(v) => cur = v,
        }
    }
    guard
}

/// Cumulative `(total_wait_ns, acquisitions, max_wait_ns)`; the max is reset each
/// read so the census reports a per-window peak, not a boot-lifetime peak.
pub fn engine_lock_wait_snapshot() -> (u64, u64, u64) {
    (
        ENGINE_LOCK_WAIT_NS.load(Ordering::Relaxed),
        ENGINE_LOCK_ACQ.load(Ordering::Relaxed),
        ENGINE_LOCK_MAX_NS.swap(0, Ordering::Relaxed),
    )
}

/// Device-reset proxy: guest-derived Vulkan objects evicted at the lifetime boundary.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GuestResetStats {
    pub resident_targets: usize,
    pub pooled_targets: usize,
    pub sampled_images: usize,
    pub storage_images: usize,
    pub had_context: bool,
}

/// Drop guest-identity/resource state while preserving the Vulkan context and
/// immutable content-keyed shader/pipeline caches.
pub fn reset_guest_state() -> GuestResetStats {
    let mut guard = lock_engine();
    let (resident_targets, pooled_targets, sampled_images, storage_images) =
        guard.pools.guest_reset_counts();
    let stats = GuestResetStats {
        resident_targets,
        pooled_targets,
        sampled_images,
        storage_images,
        had_context: guard.owner.ctx.is_some(),
    };
    let EngineState {
        ref owner,
        ref mut pools,
        ref mut scanout_export,
        ref mut present_export_ring,
        #[cfg(all(feature = "host-window", target_os = "macos"))]
        ref mut window_presenter,
        ..
    } = &mut *guard;
    if let Some(ctx) = owner.ctx.as_ref() {
        if let Err(error) = unsafe { ctx.device.device_wait_idle() } {
            let decline = VkCall::new(VkOp::GuestResetDeviceWaitIdle, error);
            crate::observe::Emit::decline("vulkan_guest_reset", &decline).fail_once(0);
        }
        unsafe {
            #[cfg(all(feature = "host-window", target_os = "macos"))]
            if let Some(presenter) = window_presenter.as_mut() {
                presenter.release_pins_after_idle(pools);
            }
            if let Some(fd) = scanout_export.destroy(&ctx.device) {
                drop(std::os::fd::OwnedFd::from_raw_fd(fd));
            }
            for fd in present_export_ring.destroy(&ctx.device) {
                drop(std::os::fd::OwnedFd::from_raw_fd(fd));
            }
            pools.destroy_all(&ctx.device);
        }
    }
    *pools = ResourcePools::new();
    crate::observe::off(format!(
        "vulkan_guest_reset resident={} pooled_targets={} sampled={} storage={} context={}",
        stats.resident_targets,
        stats.pooled_targets,
        stats.sampled_images,
        stats.storage_images,
        u8::from(stats.had_context)
    ));
    stats
}

/// Ensure the macOS host-window surface and swapchain exist on the engine's
/// Vulkan instance/device.
#[cfg(all(feature = "host-window", target_os = "macos"))]
pub fn window_present_attach(
    display: raw_window_handle::RawDisplayHandle,
    window: raw_window_handle::RawWindowHandle,
    width: u32,
    height: u32,
) -> Result<(), DrawError> {
    let mut guard = lock_engine();
    let EngineState {
        ref mut owner,
        ref counters,
        ref mut window_presenter,
        ..
    } = &mut *guard;
    if window_presenter.is_some() {
        return Ok(());
    }
    let ctx = owner.ensure(counters)?;
    *window_presenter = Some(unsafe {
        window_present::WindowPresenter::create(ctx, display, window, width, height)?
    });
    Ok(())
}

#[cfg(all(feature = "host-window", target_os = "macos"))]
pub fn window_present_resize(width: u32, height: u32) {
    let mut guard = lock_engine();
    if let Some(presenter) = guard.window_presenter.as_mut() {
        presenter.resize(width, height);
    }
}

/// Present the current compositor resident through the engine-owned MoltenVK
/// swapchain. Acquire is nonblocking, so a vblank wait never holds `ENGINE`.
#[cfg(all(feature = "host-window", target_os = "macos"))]
pub fn window_present_frame(
    source: Option<&WindowPresentSource>,
) -> Result<WindowPresentOutcome, DrawError> {
    let mut guard = lock_engine();
    let EngineState {
        ref mut owner,
        ref mut pools,
        ref counters,
        ref mut window_presenter,
        ..
    } = &mut *guard;
    let ctx = owner.ensure(counters)?;
    let presenter = window_presenter.as_mut().ok_or(DrawError::Facade(
        EngineFacadeDecline::WindowPresenterNotAttached,
    ))?;
    unsafe { presenter.present(ctx, pools, counters, source) }
}

/// Destroy the engine-owned surface while the native AppKit window still
/// exists. Called from winit's `exiting` callback.
#[cfg(all(feature = "host-window", target_os = "macos"))]
pub fn window_present_detach() {
    let mut guard = lock_engine();
    let Some(mut presenter) = guard.window_presenter.take() else {
        return;
    };
    let EngineState {
        ref owner,
        ref mut pools,
        ..
    } = &mut *guard;
    if let Some(ctx) = owner.ctx.as_ref() {
        unsafe { presenter.destroy(ctx, Some(pools)) };
    }
}

/// Execute one offscreen draw; returns tight RGBA8 pixels (w*h*4).
pub fn execute_draw(req: DrawRequest) -> Result<DrawOutput, DrawError> {
    execute_draw_request(&req)
}

/// Borrow form of [`execute_draw`].
pub fn execute_draw_request(req: &DrawRequest) -> Result<DrawOutput, DrawError> {
    let mut guard = lock_engine();
    let EngineState {
        ref mut owner,
        ref mut caches,
        ref mut pools,
        ref counters,
        ..
    } = &mut *guard;
    let result = unsafe { exec::execute_draw_inner(owner, caches, pools, counters, req) };
    match result {
        Ok(out) => Ok(out),
        Err(DrawError::DeviceLost(decline)) => {
            guard.counters.device_lost.fetch_add(1, Ordering::Relaxed);
            guard.owner.mark_device_lost();
            guard.flush_device_derived();
            if let Err(error) = {
                let EngineState {
                    ref mut owner,
                    ref counters,
                    ..
                } = &mut *guard;
                owner.ensure(counters)
            } {
                crate::observe::Emit::decline("vk_device_recreate", &error).fail_once(1);
            }
            Err(DrawError::DeviceLost(decline))
        }
        Err(e) => Err(e),
    }
}

/// Submit any open deferred draw batch (draw batching increment 1). Called at
/// the end of every drain tranche so batched work never idles unsubmitted
/// while the worker sleeps; every in-engine consumer path (reads, compute,
/// prefetch, next non-joinable draw) already flushes via begin_entry, so this
/// only bounds the idle-tail latency. No-op without a context or open batch.
pub fn flush_batched_draws() {
    let mut guard = lock_engine();
    let EngineState {
        ref mut owner,
        ref mut pools,
        ref counters,
        ..
    } = &mut *guard;
    let Some(ctx) = owner.ctx.as_ref() else {
        return;
    };
    if let Err(e) = unsafe { pools.batch_flush(ctx, counters) } {
        // A lost device surfaces again on the next draw, which runs the full
        // recreate path; here just make the flush failure visible.
        crate::observe::Emit::decline("vk_batch_flush", &e).fail_once(0);
    }
}

/// Execute one compute dispatch; returns storage buffer + storage image readbacks.
pub fn execute_compute(req: ComputeRequest) -> Result<ComputeOutput, ComputeError> {
    execute_compute_request(&req)
}

/// Borrow form of [`execute_compute`].
pub fn execute_compute_request(req: &ComputeRequest) -> Result<ComputeOutput, ComputeError> {
    let lock_started = std::time::Instant::now();
    let mut guard = lock_engine();
    guard
        .counters
        .lock_wait_us
        .fetch_add(lock_started.elapsed().as_micros() as u64, Ordering::Relaxed);
    let EngineState {
        ref mut owner,
        ref mut caches,
        ref mut pools,
        ref counters,
        ..
    } = &mut *guard;
    let result =
        unsafe { exec_compute::execute_compute_inner(owner, caches, pools, counters, req) };
    match result {
        Ok(out) => Ok(out),
        Err(DrawError::DeviceLost(decline)) => {
            guard.counters.device_lost.fetch_add(1, Ordering::Relaxed);
            guard.owner.mark_device_lost();
            guard.flush_device_derived();
            if let Err(error) = {
                let EngineState {
                    ref mut owner,
                    ref counters,
                    ..
                } = &mut *guard;
                owner.ensure(counters)
            } {
                crate::observe::Emit::decline("vk_device_recreate", &error).fail_once(2);
            }
            Err(DrawError::DeviceLost(decline))
        }
        Err(e) => Err(e),
    }
}

/// Measure-only: does the target registry hold **content_ready** for this identity?
///
/// Used by type-11 sample dig (`sample_src=… resident_ready=`) to detect the
/// resident-vs-guest split without a full readback. Does not create devices or
/// allocate; returns false if the engine is uninit or the key is absent.
pub fn resident_content_ready(identity: &TargetIdentity) -> bool {
    let guard = lock_engine();
    guard
        .pools
        .registry_get(identity)
        .is_some_and(|s| s.content_ready)
}

/// Whether this backend may leave guest-visible content only in GPU-resident
/// engine state.
///
/// Held back by the `guest_pages_stay_authoritative` driver quirk, because a
/// device recreate drops that registry before guest pages are updated. See
/// [`crate::backend::vulkan::caps::DriverQuirk`] for what the quirk covers and
/// how to retire it.
pub fn deferred_gpu_only_content_allowed() -> bool {
    lock_engine()
        .owner
        .ctx
        .as_ref()
        .is_some_and(|ctx| !ctx.caps.quirks.guest_pages_stay_authoritative)
}

/// Whether the selected device enabled `VK_EXT_external_memory_host`.
///
/// This is distinct from `deferred_gpu_only_content_allowed`: a portability
/// device may synchronously DMA a completed Store into stable guest pages even
/// though it must not leave those pages stale across packet boundaries.
pub fn external_memory_host_available() -> bool {
    lock_engine()
        .owner
        .ctx
        .as_ref()
        .is_some_and(|ctx| ctx.ext_external_memory_host.is_some())
}

/// Measure-only: newest-generation registry Surface entry for `id` at this
/// geometry, ignoring generation. Returns `(generation, content_ready)`.
/// Diagnoses generation-orphaned residents at type-11 sample fallbacks.
pub fn resident_probe_surface_any_gen(id: u32, width: u32, height: u32) -> Option<(u64, bool)> {
    let guard = lock_engine();
    guard
        .pools
        .registry_probe_surface_any_gen(id, width, height)
}

/// Pin a content-ready resident render target against LRU eviction (deferred
/// render Store — the GPU image is the only copy until flush-on-access lands
/// it in guest pages). Returns false when the identity is absent or not
/// ready; the caller must then perform the synchronous Store instead.
pub fn pin_resident_target(identity: &TargetIdentity) -> bool {
    let mut guard = lock_engine();
    guard.pools.pin_resident_target(identity, true)
}

/// Drop the deferred render-Store pin (flushed, or the window was dropped at
/// a lifetime boundary). The target stays registered — only LRU protection
/// ends. No-op for an absent identity.
pub fn unpin_resident_target(identity: &TargetIdentity) {
    let mut guard = lock_engine();
    let _ = guard.pools.pin_resident_target(identity, false);
}

/// Refresh a resident target's idle-drain timestamp without doing GPU work.
/// Direct-present uses this before export so the displayed resident is not
/// reclaimed underneath the window on a present that does no draw.
pub fn touch_resident_target(identity: Option<&TargetIdentity>, now_ms: u64) {
    let Some(identity) = identity else {
        return;
    };
    let mut guard = lock_engine();
    guard.pools.registry_touch_at(identity, now_ms);
}

/// Arm a GPU-side present-proxy stats reduction for `identity` (see
/// [`stats_reduce`]). Submits the reduction dispatch and returns immediately —
/// the present path never waits.
///
/// This is what lets the always-on proxies observe a frame with **no**
/// full-frame GPU→CPU copy: the kernel runs on the GPU that already holds the
/// resident and writes a 32-byte block. `seq` must be nonzero. Returns whether
/// a reduction was armed; `false` simply means no stats exist for this present
/// (there is deliberately no read-the-frame-back fallback).
pub fn arm_present_stats(identity: &TargetIdentity, seq: u64) -> bool {
    if seq == 0 {
        let decline = stats_reduce::StatsReduceDecline::ZeroSequence;
        crate::observe::Emit::decline("stats_reduce", &decline).fail_once(0);
        return false;
    }
    let mut guard = lock_engine();
    let EngineState {
        ref mut owner,
        ref mut pools,
        ref mut caches,
        ref counters,
        ..
    } = &mut *guard;
    let ctx = match owner.ensure(counters) {
        Ok(ctx) => ctx,
        Err(error) => {
            engine_probe_decline(EngineProbe::PresentStatsContext, &error)
                .fail_once(EngineProbe::PresentStatsContext.discriminant());
            return false;
        }
    };
    if let Err(error) = unsafe { pools.ensure_init(ctx, counters) } {
        engine_probe_decline(EngineProbe::PresentStatsPools, &error)
            .fail_once(EngineProbe::PresentStatsPools.discriminant());
        return false;
    }
    unsafe { pools.arm_present_stats(ctx, counters, caches, identity, seq) }
}

/// Take a completed stats reduction for `(identity, generation, seq)`.
///
/// Non-blocking: `None` means either nothing was armed under that key or the
/// dispatch has not finished, so the caller should just try again on a later
/// present. Because the slot is matched on the exact `(identity, generation,
/// seq)` triple, a block returned here always belongs to the frame the caller
/// asked about — stats may lag, but they are never misattributed.
pub fn take_present_stats(
    identity: &TargetIdentity,
    generation: u64,
    seq: u64,
) -> Option<stats_reduce::PresentStats> {
    let mut guard = lock_engine();
    let EngineState {
        ref mut owner,
        ref mut pools,
        ref counters,
        ..
    } = &mut *guard;
    let ctx = match owner.ensure(counters) {
        Ok(ctx) => ctx,
        Err(error) => {
            engine_probe_decline(EngineProbe::TakeStatsContext, &error)
                .fail_once(EngineProbe::TakeStatsContext.discriminant());
            return None;
        }
    };
    unsafe { pools.take_present_stats(ctx, identity, generation, seq) }
}

/// Cancel an armed stats reduction whose present is gone.
pub fn cancel_present_stats(seq: u64) {
    lock_engine().pools.cancel_present_stats(seq);
}

/// `(arms, hits, misses, not_ready, saturated)` stats-reduction diagnostics for
/// the always-on census.
pub fn present_stats_snapshot() -> (u64, u64, u64, u64, u64) {
    lock_engine().pools.present_stats_counters()
}

#[derive(Clone, Copy, Debug)]
enum EngineProbe {
    PresentStatsContext,
    PresentStatsPools,
    TakeStatsContext,
    HostImportContext,
    HostImportPools,
    ComputeWritebackAlignment,
    StorageWriteWithoutFormat,
    ComputeCapable,
    SampledR32fLinearFilter,
}

impl EngineProbe {
    fn name(self) -> &'static str {
        match self {
            Self::PresentStatsContext => "present_stats_context",
            Self::PresentStatsPools => "present_stats_pools",
            Self::TakeStatsContext => "take_stats_context",
            Self::HostImportContext => "host_import_context",
            Self::HostImportPools => "host_import_pools",
            Self::ComputeWritebackAlignment => "compute_writeback_alignment",
            Self::StorageWriteWithoutFormat => "storage_write_without_format",
            Self::ComputeCapable => "compute_capable",
            Self::SampledR32fLinearFilter => "sampled_r32f_linear_filter",
        }
    }

    fn discriminant(self) -> u64 {
        match self {
            Self::PresentStatsContext => 1,
            Self::PresentStatsPools => 2,
            Self::TakeStatsContext => 3,
            Self::HostImportContext => 4,
            Self::HostImportPools => 5,
            Self::ComputeWritebackAlignment => 6,
            Self::StorageWriteWithoutFormat => 7,
            Self::ComputeCapable => 8,
            Self::SampledR32fLinearFilter => 9,
        }
    }
}

fn engine_probe_decline(probe: EngineProbe, error: &DrawError) -> crate::observe::Emit {
    crate::observe::Emit::decline("vk_engine_probe", error).field("probe", probe.name())
}

/// Ensure every run's guest-RAM host span is covered by a cached
/// VK_EXT_external_memory_host import (creating one on first sight — a 1 GiB-capped
/// window of the containing VMA, aligned span fallback; see AGENTS.md). The
/// zero-copy gates call this with the whole coalesced run list BEFORE choosing
/// [`SampledSource::GuestRuns`]; a `false` means the caller must stay on the CPU
/// byte path.
///
/// Takes the run list rather than one span because the engine prologue — global
/// lock, device-context ensure, pool-init check — is per *entry*, not per span,
/// while the question it guards is per *window*. A fragmented full-screen
/// IOSurface coalesces to ~77 runs that all resolve against the same handful of
/// 1 GiB windows, so the per-span form paid that prologue 77 times per bind:
/// measured at 14.8 µs a call and 18 324 calls over 237 binds, it was 271 ms of
/// the 453 ms those binds spent resolving.
pub fn ensure_host_imports(runs: &[GuestRun]) -> bool {
    if runs.is_empty() {
        return true;
    }
    let mut guard = lock_engine();
    let EngineState {
        ref mut owner,
        ref mut pools,
        ref counters,
        ..
    } = &mut *guard;
    let ctx = match owner.ensure(counters) {
        Ok(ctx) => ctx,
        Err(error) => {
            engine_probe_decline(EngineProbe::HostImportContext, &error)
                .fail_once(EngineProbe::HostImportContext.discriminant());
            return false;
        }
    };
    if let Err(error) = unsafe { pools.ensure_init(ctx, counters) } {
        engine_probe_decline(EngineProbe::HostImportPools, &error)
            .fail_once(EngineProbe::HostImportPools.discriminant());
        return false;
    }
    runs.iter()
        .all(|r| unsafe { pools.host_import_resolve(ctx, r.host_ptr, r.len) }.is_ok())
}

/// Generation of a resident compute storage image, if the engine holds one.
///
/// Measure/skip aid for the runtime's stage-time guest-read skip: a skip is
/// taken only when this equals the mapping's current content generation. Does
/// not create devices or allocate; returns `None` when the engine is uninit
/// or the key is absent.
pub fn compute_resident_storage_generation(
    identity: &crate::model::ComputeStorageResidencyKey,
) -> Option<u32> {
    let guard = lock_engine();
    guard.pools.compute_resident_generation(identity)
}

/// Generation + engine format of a resident compute storage image, if the
/// engine holds one.
///
/// Skip aid for the runtime's copy-on-sample gate: a sampled guest read is
/// skipped only when the generation matches the runtime's residency mirror
/// AND the resident's vk format equals what the sampled view will bind (the
/// engine's resident-bind path guards format equality and would fail the
/// whole request on mismatch). Does not create devices or allocate; returns
/// `None` when the engine is uninit or the key is absent.
pub fn compute_resident_sample_source(
    identity: &crate::model::ComputeStorageResidencyKey,
) -> Option<(u32, StorageImageFormat)> {
    let guard = lock_engine();
    guard.pools.compute_resident_sample_source(identity)
}

/// Drop the deferred-writeback pin of a resident whose guest window can no
/// longer be flushed (ReplacePhysical / unmap drop paths). The resident stays
/// registered — only LRU protection ends. No-op for an absent identity.
pub fn unpin_resident_storage(identity: &crate::model::ComputeStorageResidencyKey) {
    let mut guard = lock_engine();
    guard.pools.pin_resident_storage(identity, false);
}

/// Minimum imported-host-pointer alignment for GPU-direct compute writeback,
/// or `None` when `VK_EXT_external_memory_host` is unavailable. Ensures the
/// device (first caller pays context creation, like any dispatch would).
pub fn compute_host_writeback_alignment() -> Option<u64> {
    let mut guard = lock_engine();
    let EngineState {
        ref mut owner,
        ref counters,
        ..
    } = &mut *guard;
    let ctx = match owner.ensure(counters) {
        Ok(ctx) => ctx,
        Err(error) => {
            engine_probe_decline(EngineProbe::ComputeWritebackAlignment, &error)
                .fail_once(EngineProbe::ComputeWritebackAlignment.discriminant());
            return None;
        }
    };
    ctx.ext_external_memory_host.as_ref()?;
    Some(ctx.min_imported_host_pointer_alignment.max(1))
}

/// True when the device supports format-less storage-image writes
/// (`shaderStorageImageWriteWithoutFormat`). The compute path needs this to
/// composite a guest `BGRA8Unorm` storage surface into a `B8G8R8A8_UNORM` view
/// without an R/B channel swap; when absent it degrades to a `R8G8B8A8_UNORM`
/// view (swapped) and logs the degraded class. Returns `false` if the engine
/// cannot initialize.
pub fn supports_storage_image_write_without_format() -> bool {
    let mut guard = lock_engine();
    let EngineState {
        ref mut owner,
        ref counters,
        ..
    } = &mut *guard;
    match owner.ensure(counters) {
        Ok(ctx) => ctx.storage_image_write_without_format,
        Err(error) => {
            engine_probe_decline(EngineProbe::StorageWriteWithoutFormat, &error)
                .fail_once(EngineProbe::StorageWriteWithoutFormat.discriminant());
            false
        }
    }
}

/// Whether the bound device can sample an `R32_SFLOAT` image with **linear**
/// filtering. Gates the native single-channel float32 sampled rail (color
/// LUTs): `R16_SFLOAT` linear filtering is spec-mandatory and needs no gate,
/// but `R32_SFLOAT`'s is optional and absent on Apple/MoltenVK. Returns `false`
/// (declining the rail, leaving the sample fail-visible) if the engine cannot
/// initialize.
pub fn supports_sampled_r32f_linear_filter() -> bool {
    let mut guard = lock_engine();
    let EngineState {
        ref mut owner,
        ref counters,
        ..
    } = &mut *guard;
    match owner.ensure(counters) {
        Ok(ctx) => ctx.sampled_r32f_linear_filter,
        Err(error) => {
            engine_probe_decline(EngineProbe::SampledR32fLinearFilter, &error)
                .fail_once(EngineProbe::SampledR32fLinearFilter.discriminant());
            false
        }
    }
}

/// Read a content-ready **BGRA** resident target as tight BGRA8 for the present
/// capture (the proxy-oracle frame source).
///
/// This is the resident-direct capture source: it performs only the GPU→host
/// readback, with **no** guest-page scatter. `capture_present_frame`'s other
/// source (`flush_intersecting` → `present_into_host_runs`) reads the same
/// resident but additionally scatters it into the fragmented guest pages — work
/// the oracle does not need and which the deferred-writeback rail already
/// performs on a genuine guest read. Errors (rather than swapping channels) on a
/// non-BGRA resident: the caller's frame buffer is BGRA8, and an RGBA resident
/// would hand the proxies channel-swapped pixels.
///
/// Returns `None` for every *expected* absence — unknown identity, no ready
/// content, non-BGRA resident, or a short/oversized readback — so the caller can
/// fall back silently. These are speculative conditions on a normal boot (a cold
/// mid has no resident yet), not failures worth a fail-log line.
pub fn read_resident_bgra(identity: &TargetIdentity, need: usize) -> Option<Vec<u8>> {
    {
        let guard = lock_engine();
        let slot = guard.pools.registry_get(identity)?;
        if !slot.content_ready || !slot.bgra {
            return None;
        }
    }
    let mut px = match read_target_inner(identity, false) {
        Ok((pixels, _)) => pixels,
        Err(e) => {
            let mut emit = crate::observe::Emit::decline("present_capture", &e);
            for (key, value) in draw_execution::identity_fields(identity) {
                emit = emit.field(key, value);
            }
            emit.off();
            return None;
        }
    };
    if px.len() < need {
        return None;
    }
    px.truncate(need);
    Some(px)
}

fn read_target_inner(
    identity: &TargetIdentity,
    measure_content: bool,
) -> Result<ResidentReadback, DrawError> {
    let mut guard = lock_engine();
    let EngineState {
        ref mut owner,
        ref mut pools,
        ref counters,
        ..
    } = &mut *guard;
    let ctx = owner.ensure(counters)?;
    unsafe { pools.ensure_init(ctx, counters)? };
    let slot = pools.registry_get(identity).ok_or(DrawError::Present(
        reason::HostPresentDecline::ReadTargetUnknownIdentity,
    ))?;
    if !slot.content_ready {
        return Err(DrawError::Present(
            reason::HostPresentDecline::ReadTargetNoReadyContent,
        ));
    }
    let width = slot.width;
    let height = slot.height;
    let image = slot.image;
    let old_layout = slot.layout;
    let rb_size = (width as u64) * (height as u64) * 4;
    let readback = unsafe { pools.acquire_readback(ctx, rb_size, counters)? };

    unsafe {
        // Async ring advance (retires only the one slot it reuses), NOT a
        // whole-ring quiesce: this is a pure content_ready readback, not an
        // UNDEFINED-layout seed, so the `ALL_COMMANDS → TRANSFER` barrier below
        // + single-queue submission order fully order the copy after every
        // prior-submitted draw (the same argument the async prefetch path
        // relies on). `begin_entry_sync` would block this guest-drain readback
        // behind an unrelated in-flight heavy draw — the `finish_us` tail. We
        // wait only our own `fence` after submit.
        let (cb, fence) = pools.begin_entry(ctx, counters)?;
        ctx.device
            .reset_command_buffer(cb, ash::vk::CommandBufferResetFlags::empty())
            .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::ReadbackResetCb, e)))?;
        ctx.device
            .begin_command_buffer(
                cb,
                &ash::vk::CommandBufferBeginInfo::default()
                    .flags(ash::vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::ReadbackBeginCb, e)))?;
        if old_layout != ash::vk::ImageLayout::TRANSFER_SRC_OPTIMAL {
            let barrier = [ash::vk::ImageMemoryBarrier::default()
                .src_access_mask(
                    ash::vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                        | ash::vk::AccessFlags::TRANSFER_WRITE
                        | ash::vk::AccessFlags::SHADER_WRITE,
                )
                .dst_access_mask(ash::vk::AccessFlags::TRANSFER_READ)
                .old_layout(old_layout)
                .new_layout(ash::vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .image(image)
                .subresource_range(ash::vk::ImageSubresourceRange {
                    aspect_mask: ash::vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })];
            ctx.device.cmd_pipeline_barrier(
                cb,
                ash::vk::PipelineStageFlags::ALL_COMMANDS,
                ash::vk::PipelineStageFlags::TRANSFER,
                ash::vk::DependencyFlags::empty(),
                &[],
                &[],
                &barrier,
            );
        }
        let region = [ash::vk::BufferImageCopy::default()
            .image_subresource(ash::vk::ImageSubresourceLayers {
                aspect_mask: ash::vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .image_extent(ash::vk::Extent3D {
                width,
                height,
                depth: 1,
            })];
        ctx.device.cmd_copy_image_to_buffer(
            cb,
            image,
            ash::vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            readback.buffer,
            &region,
        );
        ctx.device
            .end_command_buffer(cb)
            .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::ReadbackEndCb, e)))?;
        let queue = ctx.queue();
        let cbs = [cb];
        let si = ash::vk::SubmitInfo::default().command_buffers(&cbs);
        ctx.device
            .queue_submit(queue, &[si], fence)
            .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::ReadbackSubmit, e)))?;
        let cleanup = pools.seal_entry(Vec::new(), Vec::new());
        pools.finish_entry_async(cleanup);
        // Wait ONLY our own readback fence; the slot stays pending and the ring
        // retires it later (no-wait drain, fence already signaled).
        pools.wait_entry_fence(ctx, counters, fence)?;
        let ptr = ctx
            .device
            .map_memory(
                readback.memory,
                0,
                rb_size,
                ash::vk::MemoryMapFlags::empty(),
            )
            .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::ReadbackMap, e)))?
            as *const u8;
        // No pre-zero: every one of `rb_size` bytes is written below (the
        // stats copy or the plain memcpy from the mapped readback), so
        // `vec![0u8; rb_size]`'s zeroing of a full 8 MiB frame per present is
        // pure waste on the guest-blocking present drain. Allocate uninit and
        // fill in one pass.
        let (out, content) = if measure_content {
            let src = std::slice::from_raw_parts(ptr, rb_size as usize);
            let (out, stats) = copy_color8_with_stats_to_vec(src);
            (out, Some(stats))
        } else {
            (
                exec_compute::copy_mapped_output(ptr, rb_size as usize),
                None,
            )
        };
        ctx.device.unmap_memory(readback.memory);
        pools.registry_set_layout(identity, ash::vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
        counters.note_readback(rb_size);
        Ok((out, content))
    }
}

/// Full-frame readback of a resident target (present / Synchronize / Map / Store boundary).
pub fn read_target(identity: &TargetIdentity) -> Result<Vec<u8>, DrawError> {
    read_target_inner(identity, false).map(|(pixels, _)| pixels)
}

/// Flush read of a **pinned deferred-writeback resident storage image**: copy
/// the GPU content to the host as tight `width*height*texel` bytes and unpin.
///
/// The caller (runtime deferred-flush) writes these bytes into the guest
/// window and re-establishes its residency mirror. `expected_generation`
/// guards against flushing content from a different chain step than the one
/// the caller deferred — a mismatch (or an absent/evicted resident) is the
/// named error the caller reports as `deferred_flush_lost`. Returns
/// `(bytes, texel_size)`.
pub fn read_resident_storage(
    identity: &crate::model::ComputeStorageResidencyKey,
    expected_generation: u32,
) -> Result<(Vec<u8>, u32), DrawError> {
    let mut guard = lock_engine();
    let EngineState {
        ref mut owner,
        ref mut pools,
        ref counters,
        ..
    } = &mut *guard;
    let ctx = owner.ensure(counters)?;
    unsafe { pools.ensure_init(ctx, counters)? };
    let (image, key, generation, old_layout) =
        pools.compute_resident_snapshot(identity).ok_or_else(|| {
            DrawError::Facade(EngineFacadeDecline::StorageReadResidentAbsent {
                identity: *identity,
            })
        })?;
    if generation != expected_generation {
        return Err(DrawError::Facade(
            EngineFacadeDecline::StorageReadGenerationMismatch {
                identity: *identity,
                actual_generation: generation,
                expected_generation,
            },
        ));
    }
    let texel = key.format.bytes_per_texel() as u32;
    let rb_size = (key.width as u64) * (key.height as u64) * texel as u64;
    let readback = unsafe { pools.acquire_readback(ctx, rb_size, counters)? };

    unsafe {
        // Async ring advance (retires only the one slot it reuses), NOT a
        // whole-ring quiesce: this is a pure content_ready readback, not an
        // UNDEFINED-layout seed, so the `ALL_COMMANDS → TRANSFER` barrier below
        // + single-queue submission order fully order the copy after every
        // prior-submitted draw (the same argument the async prefetch path
        // relies on). `begin_entry_sync` would block this guest-drain readback
        // behind an unrelated in-flight heavy draw — the `finish_us` tail. We
        // wait only our own `fence` after submit.
        let (cb, fence) = pools.begin_entry(ctx, counters)?;
        ctx.device
            .reset_command_buffer(cb, ash::vk::CommandBufferResetFlags::empty())
            .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::StorageReadResetCb, e)))?;
        ctx.device
            .begin_command_buffer(
                cb,
                &ash::vk::CommandBufferBeginInfo::default()
                    .flags(ash::vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::StorageReadBeginCb, e)))?;
        if old_layout != ash::vk::ImageLayout::TRANSFER_SRC_OPTIMAL {
            let barrier = [ash::vk::ImageMemoryBarrier::default()
                .src_access_mask(
                    ash::vk::AccessFlags::TRANSFER_WRITE | ash::vk::AccessFlags::SHADER_WRITE,
                )
                .dst_access_mask(ash::vk::AccessFlags::TRANSFER_READ)
                .old_layout(old_layout)
                .new_layout(ash::vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .image(image)
                .subresource_range(ash::vk::ImageSubresourceRange {
                    aspect_mask: ash::vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })];
            ctx.device.cmd_pipeline_barrier(
                cb,
                ash::vk::PipelineStageFlags::ALL_COMMANDS,
                ash::vk::PipelineStageFlags::TRANSFER,
                ash::vk::DependencyFlags::empty(),
                &[],
                &[],
                &barrier,
            );
        }
        let region = [ash::vk::BufferImageCopy::default()
            .image_subresource(ash::vk::ImageSubresourceLayers {
                aspect_mask: ash::vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .image_extent(ash::vk::Extent3D {
                width: key.width,
                height: key.height,
                depth: 1,
            })];
        ctx.device.cmd_copy_image_to_buffer(
            cb,
            image,
            ash::vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            readback.buffer,
            &region,
        );
        ctx.device
            .end_command_buffer(cb)
            .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::StorageReadEndCb, e)))?;
        let queue = ctx.queue();
        let cbs = [cb];
        let si = ash::vk::SubmitInfo::default().command_buffers(&cbs);
        ctx.device
            .queue_submit(queue, &[si], fence)
            .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::StorageReadSubmit, e)))?;
        let cleanup = pools.seal_entry(Vec::new(), Vec::new());
        pools.finish_entry_async(cleanup);
        // Wait ONLY our own readback fence; the slot stays pending and the ring
        // retires it later (no-wait drain, fence already signaled).
        pools.wait_entry_fence(ctx, counters, fence)?;
        let ptr = ctx
            .device
            .map_memory(
                readback.memory,
                0,
                rb_size,
                ash::vk::MemoryMapFlags::empty(),
            )
            .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::StorageReadMap, e)))?
            as *const u8;
        let mut out = vec![0u8; rb_size as usize];
        std::ptr::copy_nonoverlapping(ptr, out.as_mut_ptr(), rb_size as usize);
        ctx.device.unmap_memory(readback.memory);
        pools.set_resident_storage_layout(identity, ash::vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
        pools.pin_resident_storage(identity, false);
        counters.note_compute_deferred_flush(rb_size);
        Ok((out, texel))
    }
}

/// Present a resident target **directly into caller-supplied host memory** via
/// `VK_EXT_external_memory_host` — the GPU copies the frame straight into the
/// pages behind `host_ptr` (guest RAM in production), with **no CPU readback
/// copy**. This is the Vulkan analog of Apple's `commitIntoGPUPageTable`
/// (workstream E). The caller owns `host_ptr`; on success those bytes hold
/// tight BGRA/RGBA8 when `buffer_row_bytes == 0`, or **padded guest row layout**
/// when `buffer_row_bytes` is the IOSurface bytes-per-row (≥ `width*4`). Uses
/// `VkBufferImageCopy::buffer_row_length` so padded IOSurface rows (e.g. 7736 vs
/// 7680) land correctly — a flat `w*h*4` DMA was the known correctness gap on
/// live mids 1/5.
///
/// When `measure_content` is set, content stats over the resident are computed
/// **on the GPU** (a 32-byte compute reduction armed before the blit) and
/// returned — the packed-contig store's measurement path, symmetric with
/// [`present_into_host_runs`], so it never needs a full-frame CPU readback for
/// stats. Returns `None` stats when `measure_content` is off, the target is not
/// BGRA, or the reduction is declined.
///
/// Errors (caller falls back to [`read_target`] + CPU copy): capability absent,
/// unknown/not-ready target, `ptr_len` too small, bad stride, or `host_ptr`/size
/// not meeting `min_imported_host_pointer_alignment`.
///
/// # Safety
/// `host_ptr` must point to at least `ptr_len` bytes of host memory that stays
/// valid and unaliased for the duration of the call.
pub unsafe fn present_into_host_ptr_strided(
    identity: &TargetIdentity,
    host_ptr: *mut std::ffi::c_void,
    ptr_len: u64,
    buffer_row_bytes: u32,
    measure_content: bool,
) -> Result<Option<Color8ContentStats>, DrawError> {
    let mut guard = lock_engine();
    let EngineState {
        ref mut owner,
        ref mut caches,
        ref mut pools,
        ref counters,
        ..
    } = &mut *guard;
    let ctx = owner.ensure(counters)?;
    pools.ensure_init(ctx, counters)?;
    if ctx.ext_external_memory_host.is_none() {
        return Err(DrawError::Unsupported(
            reason::DrawReason::PresentHostPtrImportUnavailable,
        ));
    }
    let (width, height, image, generation) = {
        let slot = pools.registry_get(identity).ok_or_else(|| {
            DrawError::Present(reason::HostPresentDecline::HostPtrUnknownIdentity)
        })?;
        if !slot.content_ready {
            return Err(DrawError::Present(
                reason::HostPresentDecline::HostPtrNoReadyContent,
            ));
        }
        (slot.width, slot.height, slot.image, slot.generation)
    };
    let tight = width.saturating_mul(4);
    let row_bytes = if buffer_row_bytes == 0 {
        tight
    } else {
        buffer_row_bytes
    };
    if row_bytes < tight || (row_bytes % 4) != 0 {
        return Err(DrawError::Present(
            reason::HostPresentDecline::HostPtrBadRowBytes {
                row_bytes: buffer_row_bytes,
                tight,
            },
        ));
    }
    let frame_size = (row_bytes as u64).saturating_mul(height as u64);
    // Imported allocation size must be a multiple of the min alignment (Vulkan
    // VU); round the frame up and require the caller's buffer to cover it.
    let align = ctx.min_imported_host_pointer_alignment.max(1);
    let import_size = frame_size.div_ceil(align) * align;
    if ptr_len < import_size {
        return Err(DrawError::Present(
            reason::HostPresentDecline::HostPtrShort {
                ptr_len,
                import_size,
            },
        ));
    }

    // GPU-direct content stats for the packed-contig store's measurement path.
    // Arm the reduction (reads the resident as SHADER_READ_ONLY_OPTIMAL) BEFORE
    // the blit transitions it to TRANSFER_SRC_OPTIMAL — the same ordering rule
    // as present_scatter_gpu. The 32-byte block is consumed after retire below,
    // so the packed store never needs a full-frame CPU readback for stats. arm
    // leaves the resident in SHADER_READ_ONLY_OPTIMAL and records that layout in
    // the registry, so `old_layout` is read AFTER arming.
    let stats_seq =
        STORE_STATS_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed) | STORE_STATS_SEQ_TAG;
    let armed =
        measure_content && pools.arm_present_stats(ctx, counters, caches, identity, stats_seq);
    let old_layout = pools
        .registry_get(identity)
        .map(|s| s.layout)
        .unwrap_or(ash::vk::ImageLayout::TRANSFER_SRC_OPTIMAL);

    // Resolve through the same capped cache as sampled gathers and fragmented
    // Store. Never create a direct uncapped host-pointer import here.
    let (buffer, buffer_offset) =
        unsafe { pools.host_import_resolve(ctx, host_ptr as usize, import_size) }?;

    let (cb, fence) = pools.begin_entry_sync(ctx, counters)?;
    // buffer_row_length is in **texels** (not bytes) for VkBufferImageCopy.
    let buffer_row_length_texels = row_bytes / 4;
    let record = || -> Result<(), DrawError> {
        ctx.device
            .reset_command_buffer(cb, ash::vk::CommandBufferResetFlags::empty())
            .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::HostPresentResetCb, e)))?;
        ctx.device
            .begin_command_buffer(
                cb,
                &ash::vk::CommandBufferBeginInfo::default()
                    .flags(ash::vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::HostPresentBeginCb, e)))?;
        if old_layout != ash::vk::ImageLayout::TRANSFER_SRC_OPTIMAL {
            let barrier = [ash::vk::ImageMemoryBarrier::default()
                .src_access_mask(
                    ash::vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                        | ash::vk::AccessFlags::TRANSFER_WRITE
                        | ash::vk::AccessFlags::SHADER_WRITE,
                )
                .dst_access_mask(ash::vk::AccessFlags::TRANSFER_READ)
                .old_layout(old_layout)
                .new_layout(ash::vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .image(image)
                .subresource_range(ash::vk::ImageSubresourceRange {
                    aspect_mask: ash::vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })];
            ctx.device.cmd_pipeline_barrier(
                cb,
                ash::vk::PipelineStageFlags::ALL_COMMANDS,
                ash::vk::PipelineStageFlags::TRANSFER,
                ash::vk::DependencyFlags::empty(),
                &[],
                &[],
                &barrier,
            );
        }
        let region = [ash::vk::BufferImageCopy::default()
            .buffer_offset(buffer_offset)
            .buffer_row_length(buffer_row_length_texels)
            .buffer_image_height(height)
            .image_subresource(ash::vk::ImageSubresourceLayers {
                aspect_mask: ash::vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .image_extent(ash::vk::Extent3D {
                width,
                height,
                depth: 1,
            })];
        ctx.device.cmd_copy_image_to_buffer(
            cb,
            image,
            ash::vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            buffer,
            &region,
        );
        ctx.device
            .end_command_buffer(cb)
            .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::HostPresentEndCb, e)))?;
        Ok(())
    };
    record()?;
    let queue = ctx.queue();
    let cbs = [cb];
    let si = ash::vk::SubmitInfo::default().command_buffers(&cbs);
    if let Err(e) = ctx.device.queue_submit(queue, &[si], fence) {
        return Err(DrawError::VkCall(VkCall::new(VkOp::HostPresentSubmit, e)));
    }
    let sealed = pools.seal_entry(Vec::new(), Vec::new());
    pools.finish_entry_async(sealed);
    pools.retire_all(ctx, counters)?;
    // HOST_COHERENT: the GPU DMA is visible to the guest CPU with no map/flush.
    pools.registry_set_layout(identity, ash::vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
    counters.note_import_present();
    // Collect the stats reduction armed before the blit (32-byte block, no
    // full-frame copy). `None` when not armed (measure_content off, non-bgra
    // target, or arm declined).
    let content = if armed {
        pools.consume_store_stats_blocking(ctx, identity, generation, stats_seq)
    } else {
        None
    };
    Ok(content)
}

/// Upload a finished tight-BGRA frame into the cached exportable scanout image
/// and return `(dup_fd, row_pitch)` for GL/dmabuf scanout (the C shim's
/// `qemu_console_gl_scanout_dmabuf`). `bgra` is `width*4*height` bytes, top-left
/// origin. The returned fd is a **dup** the importer (QEMU) owns and closes each
/// present; the cache keeps its own fd + image across presents (no per-frame
/// image/fd churn), re-creating only on a geometry change.
///
/// This is the display-path source that stays robust against present
/// identity/grouping: the pixels come from the CPU capture (`frame_bgra`) via a
/// staging copy, so it never has to reconstruct a `TargetIdentity`. A direct
/// resident-slot → export blit (no `frame_bgra` at all) is the follow-up perf
/// step; this establishes the verified display path first.
///
/// # Safety
/// Must be called with no other engine borrow held; allocates + submits on the
/// engine device.
pub unsafe fn export_scanout_from_bgra(
    width: u32,
    height: u32,
    bgra: &[u8],
) -> Result<(i32, u64), DrawError> {
    if width == 0 || height == 0 {
        return Err(DrawError::Facade(
            EngineFacadeDecline::ExportScanoutZeroGeometry { width, height },
        ));
    }
    let expected = (width as usize)
        .saturating_mul(4)
        .saturating_mul(height as usize);
    if bgra.len() != expected {
        return Err(DrawError::Facade(
            EngineFacadeDecline::ExportScanoutLengthMismatch {
                width,
                height,
                actual_len: bgra.len(),
                expected_len: expected,
            },
        ));
    }
    let mut guard = lock_engine();
    let EngineState {
        ref mut owner,
        ref mut pools,
        ref mut scanout_export,
        ref counters,
        ..
    } = &mut *guard;
    let ctx = owner.ensure(counters)?;
    pools.ensure_init(ctx, counters)?;
    if ctx.ext_external_memory_fd.is_none() {
        return Err(DrawError::Unsupported(
            reason::DrawReason::ScanoutExportUnavailable,
        ));
    }

    // Reuse the cached export image; on a geometry change retire the old one.
    // The previous present's blit into the retired image already retired at the
    // end of its own call (each call ends with retire_all), so destroying it
    // here cannot alias in-flight work.
    if let Some(old) = scanout_export.acquire(ctx, width, height)? {
        let fd = old.fd;
        old.destroy(&ctx.device);
        drop(std::os::fd::OwnedFd::from_raw_fd(fd));
    }
    let export = scanout_export
        .current()
        .expect("acquire established the current export image");
    let export_image = export.image;
    let row_pitch = export.row_pitch;
    let cached_fd = export.fd;

    // Host-visible staging: memcpy the tight BGRA frame, then copy_buffer_to_image
    // into the LINEAR export image (the copy honors the image's own row pitch, so
    // the staging buffer stays tight).
    let need = expected as u64;
    let staging =
        pools.acquire_staging(ctx, need, ash::vk::BufferUsageFlags::TRANSFER_SRC, counters)?;
    {
        let ptr = ctx
            .device
            .map_memory(staging.memory, 0, need, ash::vk::MemoryMapFlags::empty())
            .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::ExportScanoutMapStaging, e)))?;
        std::ptr::copy_nonoverlapping(bgra.as_ptr(), ptr as *mut u8, expected);
        ctx.device.unmap_memory(staging.memory);
    }

    let (cb, fence) = pools.begin_entry_sync(ctx, counters)?;
    let color_range = ash::vk::ImageSubresourceRange::default()
        .aspect_mask(ash::vk::ImageAspectFlags::COLOR)
        .base_mip_level(0)
        .level_count(1)
        .base_array_layer(0)
        .layer_count(1);
    let record = || -> Result<(), DrawError> {
        ctx.device
            .reset_command_buffer(cb, ash::vk::CommandBufferResetFlags::empty())
            .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::ExportScanoutResetCb, e)))?;
        ctx.device
            .begin_command_buffer(
                cb,
                &ash::vk::CommandBufferBeginInfo::default()
                    .flags(ash::vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::ExportScanoutBeginCb, e)))?;
        // Export image UNDEFINED -> TRANSFER_DST (we overwrite every texel).
        let pre = [ash::vk::ImageMemoryBarrier::default()
            .src_access_mask(ash::vk::AccessFlags::empty())
            .dst_access_mask(ash::vk::AccessFlags::TRANSFER_WRITE)
            .old_layout(ash::vk::ImageLayout::UNDEFINED)
            .new_layout(ash::vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .image(export_image)
            .subresource_range(color_range)];
        ctx.device.cmd_pipeline_barrier(
            cb,
            ash::vk::PipelineStageFlags::ALL_COMMANDS,
            ash::vk::PipelineStageFlags::TRANSFER,
            ash::vk::DependencyFlags::empty(),
            &[],
            &[],
            &pre,
        );
        // buffer_row_length is in texels; tight BGRA = width texels/row.
        let region = [ash::vk::BufferImageCopy::default()
            .buffer_row_length(width)
            .buffer_image_height(height)
            .image_subresource(ash::vk::ImageSubresourceLayers {
                aspect_mask: ash::vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .image_extent(ash::vk::Extent3D {
                width,
                height,
                depth: 1,
            })];
        ctx.device.cmd_copy_buffer_to_image(
            cb,
            staging.buffer,
            export_image,
            ash::vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &region,
        );
        // -> GENERAL, transfer write made available to the external dmabuf reader.
        let post = [ash::vk::ImageMemoryBarrier::default()
            .src_access_mask(ash::vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(ash::vk::AccessFlags::MEMORY_READ)
            .old_layout(ash::vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(ash::vk::ImageLayout::GENERAL)
            .image(export_image)
            .subresource_range(color_range)];
        ctx.device.cmd_pipeline_barrier(
            cb,
            ash::vk::PipelineStageFlags::TRANSFER,
            ash::vk::PipelineStageFlags::ALL_COMMANDS,
            ash::vk::DependencyFlags::empty(),
            &[],
            &[],
            &post,
        );
        ctx.device
            .end_command_buffer(cb)
            .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::ExportScanoutEndCb, e)))?;
        Ok(())
    };
    record()?;
    let queue = ctx.queue();
    let cbs = [cb];
    let si = ash::vk::SubmitInfo::default().command_buffers(&cbs);
    ctx.device
        .queue_submit(queue, &[si], fence)
        .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::ExportScanoutSubmit, e)))?;
    let sealed = pools.seal_entry(Vec::new(), Vec::new());
    pools.finish_entry_async(sealed);
    pools.retire_all(ctx, counters)?;

    // dup for the importer to own+close this present; the cache keeps `cached_fd`.
    // std OwnedFd dup (libc is Apple-only); into_raw_fd hands ownership to QEMU.
    let dup_fd = std::os::fd::BorrowedFd::borrow_raw(cached_fd)
        .try_clone_to_owned()
        .map_err(|e| DrawError::FdDup(FdDupDecline::new(FdDupRail::ExportScanout, &e)))?
        .into_raw_fd();
    Ok((dup_fd, row_pitch))
}

/// True zero-copy scanout export: blit a content-ready **resident** present
/// target straight into the next exportable LINEAR ring image (GPU→GPU, no CPU
/// readback) and return `(dup_fd, row_pitch, width, height, ring_idx)` for the
/// direct-present path (host window route B). Unlike [`export_scanout_from_bgra`],
/// this reads no `frame_bgra`: the source is the resident `VkImage` the engine
/// already rendered, resolved by `identity`. `ring_idx` names which ring slot the
/// fd backs so the window imports each slot's dmabuf once and re-blits the cached
/// import (the engine writes fresh content into that slot's shared memory); the
/// ring hands a different slot each present so no slot is written while the
/// window still reads it (tear-safety without a cross-device semaphore).
///
/// Fails visibly (never silently) when the resident cannot back this export:
/// `reason=vk_engine_export_present_unknown_identity` (no such identity),
/// `reason=vk_engine_export_present_not_ready` (resident has no content yet),
/// `reason=present_export_resident_not_bgra` (an RGBA resident would export
/// swapped channels), or `reason=present_export_unavailable` (device lacks the
/// dmabuf export extensions).
/// The caller falls back to the CPU `export_scanout_from_bgra` / staging path on
/// any error, so a miss degrades to the copy path, never to a wrong frame.
///
/// The blit leaves the resident in `TRANSFER_SRC_OPTIMAL`; the registry layout is
/// updated to match (mirrors [`read_target_inner`]). The export image ends in
/// `GENERAL`, ready for the external dmabuf consumer.
///
/// # Safety
/// Must be called with no other engine borrow held; allocates + submits on the
/// engine device.
/// Short protocol-kind label for a [`TargetIdentity`] (diagnostics only).
fn identity_kind(identity: &TargetIdentity) -> &'static str {
    match identity {
        TargetIdentity::Surface { .. } => "surface",
        TargetIdentity::Texture { .. } => "texture",
        TargetIdentity::Gva { .. } => "gva",
        TargetIdentity::Anonymous { .. } => "anon",
    }
}

/// Classify an `export_present_from_resident_fd_policy` registry miss
/// into an actionable outcome, ONE always-on line per (kind, geometry) per
/// process (deduped so a steady miss never floods). A resident that exists under
/// a DIFFERENT key at this geometry
/// (group present while present asked for a surface, or a stale-generation
/// surface) is an identity/generation ORPHAN — fixable by aligning the export
/// lookup. No resident at all at this geometry means the composited frame was
/// never rendered into a GPU target (it lives only in guest pages / the CPU
/// surface cache) — direct present then needs the composite render kept
/// resident, not just an identity tweak. This census writes `outcome=`, while
/// the typed decline reaches the caller's `export_present reason=…` boundary.
/// Measure-only; no product branch.
fn classify_export_present_miss(pools: &pools::ResourcePools, identity: &TargetIdentity) {
    use std::collections::BTreeSet;
    use std::sync::Mutex;
    type ExportMissKey = (&'static str, u32, u32);
    static LOGGED: Mutex<Option<BTreeSet<ExportMissKey>>> = Mutex::new(None);
    let (w, h) = (identity.width(), identity.height());
    let kind = identity_kind(identity);
    {
        let mut guard = LOGGED.lock().unwrap_or_else(|p| p.into_inner());
        if !guard.get_or_insert_with(BTreeSet::new).insert((kind, w, h)) {
            return;
        }
    }
    let c = pools.registry_geom_census(w, h);
    // Summarize surfaces compactly: id:gen:ready, capped so the line stays bounded.
    let mut surf = String::new();
    for (i, (id, gen, ready)) in c.surfaces.iter().take(8).enumerate() {
        if i > 0 {
            surf.push(',');
        }
        surf.push_str(&format!("{id}:{gen}:{}", *ready as u8));
    }
    if c.surfaces.len() > 8 {
        surf.push_str(",…");
    }
    // The orphan verdict: content exists at this geometry under another key.
    let orphan = !c.surfaces.is_empty() || c.gva > 0;
    crate::observe::off(format!(
        "export_present_miss outcome={} want={kind} geom={w}x{h} want_gen={} \
         surfaces=[{surf}] gva={} reg_len={}",
        if orphan { "orphan" } else { "absent" },
        identity.generation(),
        c.gva,
        c.total,
    ));
}

/// Blit `identity`'s resident into the next present export ring slot and hand
/// back that slot's dmabuf.
///
/// The exported pixels are exactly the named resident's; nothing else is mixed
/// in. `fd_needed` lets the caller decide whether the selected export ring slot
/// needs another fd duplicate. The export image is still updated every call;
/// suppressing the fd only avoids redundant fd dup/close churn when the
/// consumer has acknowledged an existing import.
///
/// # Safety
///
/// Engine teardown must not race this call. When `fd_needed` requests an fd,
/// the caller owns it and must close it after importing or abandoning it.
pub unsafe fn export_present_from_resident_fd_policy(
    identity: &TargetIdentity,
    fd_needed: impl FnOnce(usize, u32, u32) -> bool,
) -> Result<OptionalExportedPresent, DrawError> {
    let mut guard = lock_engine();
    let EngineState {
        ref mut owner,
        ref mut pools,
        ref mut present_export_ring,
        ref counters,
        ..
    } = &mut *guard;
    let ctx = owner.ensure(counters)?;
    pools.ensure_init(ctx, counters)?;
    if ctx.ext_external_memory_fd.is_none() {
        return Err(DrawError::Unsupported(
            reason::DrawReason::PresentExportUnavailable,
        ));
    }

    // Resolve the resident and snapshot the Copy fields, dropping the pools
    // borrow before we mutate pools (begin_entry) / scanout_export (acquire).
    let (src_image, src_layout, width, height) = {
        let slot = match pools.registry_get(identity) {
            Some(s) => s,
            None => {
                classify_export_present_miss(pools, identity);
                return Err(DrawError::Facade(
                    EngineFacadeDecline::ExportPresentUnknownIdentity {
                        identity: identity.clone(),
                    },
                ));
            }
        };
        if !slot.content_ready {
            return Err(DrawError::Facade(
                EngineFacadeDecline::ExportPresentNotReady {
                    identity: identity.clone(),
                },
            ));
        }
        // The export image is B8G8R8A8_UNORM; an RGBA resident would export with
        // R/B swapped. Reject rather than emit wrong colors (a caller-visible
        // fallback to the CPU path handles it correctly).
        if !slot.bgra {
            return Err(DrawError::Unsupported(
                reason::DrawReason::PresentExportResidentNotBgra,
            ));
        }
        (slot.image, slot.layout, slot.width, slot.height)
    };
    // Keep the presented target alive against the idle drain: a re-presented but
    // not re-drawn resident is resolved via registry_get (no draw-path stamp), so
    // without this it could age out from under the display.
    pools.registry_touch(identity);

    // Advance to the next ring slot (a DIFFERENT dmabuf than the last few
    // presents), so the engine never overwrites the image the window is still
    // reading — the tear-safety guarantee. On a geometry change every prior slot
    // is retired here; destroy + close each (the prior blit already retired at
    // the end of its own call, and the window drops its imports of them on the
    // same geometry change via dmabuf refcounting).
    let (ring_idx, retired) = present_export_ring.acquire_next(ctx, width, height)?;
    for old in retired {
        let fd = old.fd;
        old.destroy(&ctx.device);
        drop(std::os::fd::OwnedFd::from_raw_fd(fd));
    }
    let export = present_export_ring.slot(ring_idx);
    let row_pitch = export.row_pitch;
    let cached_fd = export.fd;

    // Phase split (measure-only): `begin_entry_sync` drains the guest's in-flight
    // compositing draws into the resident (inherent), the later `retire_all` waits
    // for our own export blit. Timing them apart tells whether the ~50ms/present at
    // 4K is ours to optimize before the hard async-export work — see
    // present_proxy::export_present::note_phases.
    let drain_started = std::time::Instant::now();
    let (cb, fence) = pools.begin_entry_sync(ctx, counters)?;
    let drain_us = drain_started.elapsed().as_micros() as u64;
    ctx.device
        .reset_command_buffer(cb, ash::vk::CommandBufferResetFlags::empty())
        .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::ExportPresentResetCb, e)))?;
    ctx.device
        .begin_command_buffer(
            cb,
            &ash::vk::CommandBufferBeginInfo::default()
                .flags(ash::vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )
        .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::ExportPresentBeginCb, e)))?;
    // GPU→GPU: resident (OPTIMAL, current layout) → export (LINEAR). Same helper
    // the byte-identical blit test validates.
    dmabuf_export::record_blit_present_into_export(&ctx.device, cb, src_image, src_layout, export);
    ctx.device
        .end_command_buffer(cb)
        .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::ExportPresentEndCb, e)))?;
    let queue = ctx.queue();
    let cbs = [cb];
    let si = ash::vk::SubmitInfo::default().command_buffers(&cbs);
    ctx.device
        .queue_submit(queue, &[si], fence)
        .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::ExportPresentSubmit, e)))?;
    let sealed = pools.seal_entry(Vec::new(), Vec::new());
    pools.finish_entry_async(sealed);
    let blit_started = std::time::Instant::now();
    pools.retire_all(ctx, counters)?;
    let blit_us = blit_started.elapsed().as_micros() as u64;
    crate::runtime::census::present_proxy::export_present::note_phases(drain_us, blit_us);
    // record_blit leaves the resident in TRANSFER_SRC_OPTIMAL — keep the tracked
    // layout in sync so the next draw/readback barriers from the right layout.
    pools.registry_set_layout(identity, ash::vk::ImageLayout::TRANSFER_SRC_OPTIMAL);

    let dup_fd = if fd_needed(ring_idx, width, height) {
        Some(
            std::os::fd::BorrowedFd::borrow_raw(cached_fd)
                .try_clone_to_owned()
                .map_err(|e| DrawError::FdDup(FdDupDecline::new(FdDupRail::ExportPresent, &e)))?
                .into_raw_fd(),
        )
    } else {
        None
    };
    Ok((dup_fd, row_pitch, width, height, ring_idx))
}

/// One packed host-mapped run of guest IOSurface pages.
///
/// Mapping-linear range covered is `[linear_base, linear_base + linear_len)`
/// with `linear_len <= ptr_len`. A tight resident readback is scattered to
/// host offset `(sample_base + y*bpr + x*4) - linear_base`.
#[derive(Clone, Copy, Debug)]
pub struct HostMappedRun {
    pub host_ptr: *mut std::ffi::c_void,
    /// Host allocation capacity (at least `linear_len`).
    pub ptr_len: u64,
    pub linear_base: u64,
    /// Mapping-linear bytes this run owns (page-table span; planning bound).
    pub linear_len: u64,
}

#[derive(Debug)]
pub struct HostRunPresent {
    pub content: Option<Color8ContentStats>,
    /// Wall-clock of the GPU-direct scatter (plan -> submit -> fence wait).
    pub scatter_us: u64,
}

/// Present a resident target into fragmented guest pages.
///
/// Preferred path is a GPU-direct scatter ([`try_gpu_scatter`]): the planned
/// copies are issued as `vkCmdCopyImageToBuffer` regions into guest RAM
/// imported through `VK_EXT_external_memory_host`, so no frame bytes touch the
/// CPU at all. An earlier attempt at this backed out because it retained
/// hundreds of imports per guest surface and the resulting process-wide
/// pressure serialized later target allocation on the live NVIDIA host; the
/// windowed resolver (`HOST_IMPORT_WINDOW_CAP`) removed that failure mode by
/// bucketing spans into a handful of shared 1 GiB windows, which is what makes
/// the GPU path viable now.
///
/// Fallback (no extension, non-Linux, alignment or region cap) is the portable
/// path: one pooled tight readback plus a bounded CPU scatter, returning that
/// readback for caller-side diagnostics. When `measure_content` is set, the GPU
/// path takes RGB/alpha occupancy from the reduction kernel and the CPU path
/// fuses it into the mapped-readback copy.
///
/// Layout contract (same as packed [`present_into_host_ptr_strided`]):
/// pixel `(x,y)` lives at mapping linear `sample_base_off + y*row_bytes + x*4`.
/// Each run covers a contiguous host span for a maximal packed GPA run; rows
/// may be split across runs (mid-row page breaks). Only the tight `width×4`
/// content is written; guest row padding beyond `width*4` is left untouched.
///
/// # Safety
/// Each `runs[i].host_ptr` must remain valid for `ptr_len` bytes for the whole
/// call. Runs must be ordered by `linear_base` and must not overlap.
pub unsafe fn present_into_host_runs(
    identity: &TargetIdentity,
    sample_base_off: u64,
    buffer_row_bytes: u32,
    runs: &[HostMappedRun],
    measure_content: bool,
    runs_stable: bool,
) -> Result<HostRunPresent, DrawError> {
    if runs.is_empty() {
        return Err(DrawError::Present(reason::HostPresentDecline::RunsEmpty));
    }
    let width = identity.width();
    let height = identity.height();
    let tight = width.checked_mul(4).ok_or(DrawError::Present(
        reason::HostPresentDecline::RunsTightRowOverflow,
    ))?;
    let row_bytes = if buffer_row_bytes == 0 {
        tight
    } else {
        buffer_row_bytes
    };
    if row_bytes < tight || (row_bytes % 4) != 0 {
        return Err(DrawError::Present(
            reason::HostPresentDecline::RunsBadRowBytes {
                row_bytes: buffer_row_bytes,
                tight,
            },
        ));
    }
    let mut prior_end = None;
    for (index, run) in runs.iter().enumerate() {
        if run.host_ptr.is_null() || run.ptr_len == 0 || run.linear_len == 0 {
            return Err(DrawError::Present(
                reason::HostPresentDecline::RunsNullOrEmpty { index },
            ));
        }
        if run.linear_len > run.ptr_len {
            return Err(DrawError::Present(
                reason::HostPresentDecline::RunsLenExceedsPtr {
                    linear_len: run.linear_len,
                    ptr_len: run.ptr_len,
                },
            ));
        }
        let run_end = run
            .linear_base
            .checked_add(run.linear_len)
            .ok_or(DrawError::Present(
                reason::HostPresentDecline::RunsEndOverflow,
            ))?;
        if prior_end.is_some_and(|end| run.linear_base < end) {
            return Err(DrawError::Present(
                reason::HostPresentDecline::RunsOutOfOrder { index },
            ));
        }
        prior_end = Some(run_end);
    }

    let row_bytes = row_bytes as u64;
    let tight = tight as u64;
    /// One contiguous horizontal texel run of one guest row.
    struct CopySpan {
        run_index: usize,
        /// Byte offset within that run.
        dst_offset: u64,
        /// Source texel coords in the resident image.
        x: u32,
        y: u32,
        texels: u32,
    }
    let mut copies = Vec::new();
    for y in 0..height as u64 {
        let row_start = sample_base_off
            .checked_add(y.checked_mul(row_bytes).ok_or(DrawError::Present(
                reason::HostPresentDecline::RunsRowOffsetOverflow,
            ))?)
            .ok_or(DrawError::Present(
                reason::HostPresentDecline::RunsSampleOffsetOverflow,
            ))?;
        let row_end = row_start.checked_add(tight).ok_or(DrawError::Present(
            reason::HostPresentDecline::RunsRowEndOverflow,
        ))?;
        let mut cursor = row_start;
        for (run_index, run) in runs.iter().enumerate() {
            let run_end = run.linear_base + run.linear_len;
            if run_end <= cursor {
                continue;
            }
            if run.linear_base > cursor {
                break;
            }
            let copy_end = row_end.min(run_end);
            if copy_end <= cursor {
                continue;
            }
            let dst_offset = cursor - run.linear_base;
            let copy_len = copy_end - cursor;
            if dst_offset
                .checked_add(copy_len)
                .is_none_or(|end| end > run.ptr_len)
            {
                return Err(DrawError::Present(
                    reason::HostPresentDecline::RunsScatterOob {
                        dst_offset,
                        len: copy_len,
                        cap: run.ptr_len,
                    },
                ));
            }
            copies.push(CopySpan {
                run_index,
                dst_offset,
                x: ((cursor - row_start) / 4) as u32,
                y: y as u32,
                texels: (copy_len / 4) as u32,
            });
            cursor = copy_end;
            if cursor == row_end {
                break;
            }
        }
        if cursor != row_end {
            return Err(DrawError::Present(
                reason::HostPresentDecline::RunsUncoveredRow { row: y as u32 },
            ));
        }
    }

    // GPU-direct scatter: copy the resident straight into the guest's imported
    // pages. No frame bytes touch the CPU on this path — there is no readback
    // and no scatter memcpy.
    let scatter_started = std::time::Instant::now();
    let spans: Vec<host_scatter::ScatterSpan> = copies
        .iter()
        .map(|c| host_scatter::ScatterSpan {
            run_index: c.run_index,
            dst_offset: c.dst_offset,
            x: c.x,
            y: c.y,
            texels: c.texels,
        })
        .collect();
    // The imports this path caches outlive the caller's `map_pages` view, so a
    // transient mapping would leave the GPU writing into a torn-down address
    // range. Fail closed rather than write somewhere unowned.
    if !runs_stable {
        return Err(DrawError::Unsupported(
            reason::DrawReason::PresentRunsUnstable,
        ));
    }
    let content = try_gpu_scatter(identity, runs, &spans, measure_content)?;
    Ok(HostRunPresent {
        content,
        scatter_us: scatter_started.elapsed().as_micros() as u64,
    })
}

/// Distinct seq space for the store path's stats reductions, so a store's
/// block can never be consumed by the present-proxy oracle's `(identity,
/// generation, seq)` lookup or vice versa.
static STORE_STATS_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
const STORE_STATS_SEQ_TAG: u64 = 1 << 63;

/// Attempt the GPU-direct scatter, preserving the exact initialization,
/// resource-state, import, or Vulkan-call decline through `import_present`.
fn try_gpu_scatter(
    identity: &TargetIdentity,
    runs: &[HostMappedRun],
    spans: &[host_scatter::ScatterSpan],
    measure_content: bool,
) -> Result<Option<Color8ContentStats>, DrawError> {
    let scatter_runs: Vec<host_scatter::ScatterRun> = runs
        .iter()
        .map(|r| host_scatter::ScatterRun {
            host_ptr: r.host_ptr as usize,
            ptr_len: r.ptr_len,
        })
        .collect();
    let seq =
        STORE_STATS_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed) | STORE_STATS_SEQ_TAG;
    let mut guard = lock_engine();
    let EngineState {
        ref mut owner,
        ref mut caches,
        ref mut pools,
        ref counters,
        ..
    } = &mut *guard;
    let ctx = owner.ensure(counters)?;
    unsafe { pools.ensure_init(ctx, counters) }?;
    unsafe {
        pools.present_scatter_gpu(
            ctx,
            counters,
            caches,
            identity,
            &scatter_runs,
            spans,
            measure_content,
            seq,
        )
    }
}

/// `(gpu_stores, fallback_unresolved, fallback_submit)` for the store proxy.
pub fn host_scatter_snapshot() -> (u64, u64, u64) {
    lock_engine().pools.host_scatter_counters()
}

/// One reading for the always-on cap-pressure census: current cap occupancy +
/// the cumulative eviction / reupload counters. The census (
/// `present_proxy::cap_pressure`) turns these into per-window deltas so a
/// cap-blow (eviction/reupload storm) becomes a visible line instead of an
/// unexplained frame-rate cliff.
pub fn cap_pressure_snapshot() -> crate::runtime::census::present_proxy::cap_pressure::Sample {
    use std::sync::atomic::Ordering::Relaxed;
    let guard = lock_engine();
    let occ = guard.pools.cap_pressure_occupancy();
    crate::runtime::census::present_proxy::cap_pressure::Sample {
        registry_len: occ.registry_len,
        registry_cap: occ.registry_cap,
        registry_pinned: occ.registry_pinned,
        sampled_len: occ.sampled_len,
        sampled_cap: occ.sampled_cap,
        sampled_bytes: occ.sampled_bytes,
        sampled_byte_cap: occ.sampled_byte_cap,
        graveyard_len: occ.graveyard_len,
        render_windows: 0,
        target_evicts: guard.counters.target_evicts.load(Relaxed),
        gen_mismatch: guard.counters.gen_mismatch.load(Relaxed),
        sampled_reuploads: guard.counters.sampled_reuploads.load(Relaxed),
        sampled_reupload_bytes: guard.counters.sampled_reupload_bytes.load(Relaxed),
    }
}

/// Snapshot the slab pool's physical VkDeviceMemory footprint + the recycle-pool
/// image counts for the always-on `vram` census (`present_proxy::vram`). Unlike
/// `cap_pressure_snapshot` (gated on eviction/reupload pressure, so silent at
/// idle) this drives a line even on a static page, so post-burst VRAM return is
/// visible and a fragmentation stall (`resident` flat with `blocks` full of
/// `free` bytes) is diagnosable. Pure read under the engine lock.
pub fn vram_occupancy() -> crate::runtime::census::present_proxy::vram::Sample {
    let guard = lock_engine();
    let occ = guard.pools.cap_pressure_occupancy();
    crate::runtime::census::present_proxy::vram::Sample {
        slab_blocks: occ.slab.blocks,
        slab_empty_blocks: occ.slab.empty_blocks,
        slab_resident_bytes: occ.slab.resident_bytes,
        slab_free_bytes: occ.slab.free_bytes,
        slab_live_subs: occ.slab.live_subs,
        block_allocs: occ.slab.block_allocs,
        block_frees: occ.slab.block_frees,
        max_block_free_bytes: occ.slab.max_block_free_bytes,
        size_buckets: occ.slab.size_buckets,
        registry_len: occ.registry_len,
        registry_pinned: occ.registry_pinned,
        desc_blocks: occ.desc_blocks,
        sampled_len: occ.sampled_len,
        target_free_imgs: occ.target_free_imgs,
        sampled_free_imgs: occ.sampled_free_imgs,
        storage_free_imgs: occ.storage_free_imgs,
        storage_recycle_admits: occ.storage_recycle_admits,
        storage_recycle_cap_drops: occ.storage_recycle_cap_drops,
        storage_resident: occ.storage_resident,
        storage_resident_pinned: occ.storage_resident_pinned,
        staging_free_bytes: occ.staging_free_bytes,
        readback_free_bytes: occ.readback_free_bytes,
    }
}

/// Advance the wall-clock resident-target idle-drain clock to `now_ms`, keep the
/// currently-presented target (`display`) alive, and reclaim aged non-pinned
/// residents. Called from the poll heartbeat (so the clock keeps ticking when the
/// guest stops publishing) and each present publish. Returns the count reclaimed
/// this call for the always-on `idle_target_drain` census. No-op before the
/// device context exists.
pub fn maintain_idle_residents(display: Option<&TargetIdentity>, now_ms: u64) -> usize {
    let mut guard = lock_engine();
    let EngineState {
        ref mut owner,
        ref mut pools,
        ..
    } = &mut *guard;
    let Some(ctx) = owner.ctx.as_ref() else {
        return 0;
    };
    unsafe { pools.advance_registry_touch_and_drain(ctx, now_ms, display) }
}

/// Snapshot of create/alloc/hit-miss counters (for tests and thrash proxies).
pub fn counter_snapshot() -> CounterSnapshot {
    let eng = lock_engine();
    let mut snap = eng.counters.snapshot();
    // Sampled-cache recycle diagnostics live on ResourcePools (single-threaded
    // under this lock), not the atomic counters; merge them in here.
    let (free_hits, free_allocs, recycle_admits, recycle_cap_drops) = eng.pools.recycle_stats();
    snap.sampled_free_hits = free_hits;
    snap.sampled_free_allocs = free_allocs;
    snap.sampled_recycle_admits = recycle_admits;
    snap.sampled_recycle_cap_drops = recycle_cap_drops;
    let (t_hits, t_allocs, t_admits, t_cap_drops) = eng.pools.target_recycle_stats();
    snap.target_free_hits = t_hits;
    snap.target_free_allocs = t_allocs;
    snap.target_recycle_admits = t_admits;
    snap.target_recycle_cap_drops = t_cap_drops;
    snap
}

/// Reset create/alloc/hit-miss counters (not device_lost/recreates). For reuse-gate tests.
pub fn reset_draw_counters() {
    lock_engine().counters.reset();
}

/// Full counter reset including device-loss stats (tests only).
pub fn reset_all_counters() {
    lock_engine().counters.reset_all();
}

/// Test-only: destroy device, clear recreate budget, rebuild on next draw.
pub fn test_reset_engine() {
    let mut g = lock_engine();
    if let Some(mut ctx) = g.owner.ctx.take() {
        unsafe {
            // The dmabuf export rings hold VkImages bound to THIS device and are
            // NOT owned by caches/pools. Tear them down here — exactly as the
            // production device-recreate path (`flush_device_derived`) does —
            // before destroying the device. Skipping them orphans stale image
            // handles that a later `reset_guest_state`/flush destroys against the
            // recreated device, which faults inside the driver (a real SIGSEGV
            // the serial parity suite hit at `guest_reset_evicts_*`).
            if let Some(fd) = g.scanout_export.destroy(&ctx.device) {
                drop(std::os::fd::OwnedFd::from_raw_fd(fd));
            }
            for fd in g.present_export_ring.destroy(&ctx.device) {
                drop(std::os::fd::OwnedFd::from_raw_fd(fd));
            }
            g.caches.destroy_all(&ctx.device);
            g.pools.destroy_all(&ctx.device);
            ctx.destroy();
        }
    }
    g.caches = ObjectCaches::new();
    g.pools = ResourcePools::new();
    g.owner = ContextOwner::new();
    g.counters.reset_all();
}

/// Test hook: next execute reports device lost (named path).
pub fn test_force_device_lost_once() {
    lock_engine().owner.force_device_lost = true;
}

/// Test hook: flush any open batch and retire every in-flight ring slot so
/// pending pool cleanup recycles deterministically. Warm-path allocation-free
/// assertions depend on the ring phase without this.
pub fn test_quiesce_ring() {
    let mut guard = lock_engine();
    let EngineState {
        ref mut owner,
        ref mut pools,
        ref counters,
        ..
    } = &mut *guard;
    let Some(ctx) = owner.ctx.as_ref() else {
        return;
    };
    let _ = unsafe { pools.retire_all(ctx, counters) };
}

/// Recreate budget remaining / count (for tests).
pub fn device_recreate_count() -> u32 {
    lock_engine().owner.recreate_count
}

/// Mark context poisoned and flush as if device lost (tests that assert recreate cap).
pub fn test_poison_and_flush() {
    let mut g = lock_engine();
    g.counters.device_lost.fetch_add(1, Ordering::Relaxed);
    g.owner.mark_device_lost();
    g.flush_device_derived();
}

/// Named create counter line for logs (`vk_engine_creates=N` style).
pub fn creates_this_draw_prefix(delta_creates: u64) -> String {
    format!("vk_engine_creates={delta_creates}")
}

/// Negative-cache init so repeated draws after failed init fail fast without re-probing.
pub fn init_error_cached() -> Option<String> {
    lock_engine()
        .owner
        .init_error
        .as_ref()
        .map(ToString::to_string)
}

/// For tests that need the raw counters atomic snapshot after a cold miss.
pub fn note_create_for_test() {
    ENGINE
        .lock()
        .counters
        .creates
        .fetch_add(1, Ordering::Relaxed);
}

/// Whether the live device has a combined GRAPHICS|COMPUTE queue family.
pub fn compute_capable() -> bool {
    let mut g = lock_engine();
    let EngineState {
        ref mut owner,
        ref counters,
        ..
    } = &mut *g;
    match owner.ensure(counters) {
        Ok(ctx) => ctx.compute_capable,
        Err(error) => {
            engine_probe_decline(EngineProbe::ComputeCapable, &error)
                .fail_once(EngineProbe::ComputeCapable.discriminant());
            false
        }
    }
}

#[cfg(test)]
mod probe_visibility_tests {
    use super::*;

    #[test]
    fn each_engine_probe_preserves_the_typed_initialization_reason() {
        let error = vk_call::exec_submit_device_lost_fixture();
        for probe in [
            EngineProbe::PresentStatsContext,
            EngineProbe::PresentStatsPools,
            EngineProbe::TakeStatsContext,
            EngineProbe::HostImportContext,
            EngineProbe::HostImportPools,
            EngineProbe::ComputeWritebackAlignment,
            EngineProbe::StorageWriteWithoutFormat,
            EngineProbe::ComputeCapable,
        ] {
            let line = engine_probe_decline(probe, &error).render();
            assert!(line.starts_with("vk_engine_probe reason=vk_exec_submit "));
            assert!(line.ends_with(&format!(" probe={}", probe.name())));
        }
    }

    /// One engine entry per run list, whatever its length.
    ///
    /// The prologue this guards — global lock, device-context ensure, pool-init
    /// check — is per entry, while the coverage question is per import window. A
    /// fragmented full-screen IOSurface coalesces to ~77 runs that all resolve
    /// against the same handful of windows, so a per-span form pays the prologue
    /// 77 times to answer one question. Counting acquisitions is the only
    /// assertion that fails if this regresses to a loop of single-span calls;
    /// the returned bool is identical either way.
    ///
    /// The runs must be spans that actually RESOLVE. `all` short-circuits, so a
    /// list of unbacked pointers refuses on the first run and takes exactly one
    /// engine entry in the per-span shape too. A first draft used unbacked spans,
    /// passed against a deliberately reintroduced per-span loop, and was no gate
    /// at all — hence real, page-aligned, mapped host memory here.
    #[test]
    fn ensure_host_imports_enters_the_engine_once_for_the_whole_run_list() {
        crate::observe::redirect_logs_for_tests();
        const PAGE: usize = 4096;
        let buf = vec![0u8; PAGE * 24];
        let base = (buf.as_ptr() as usize).next_multiple_of(PAGE);
        let runs: Vec<GuestRun> = (0..8)
            .map(|i| GuestRun {
                host_ptr: base + i * PAGE,
                len: PAGE as u64,
            })
            .collect();
        let (_, before, _) = engine_lock_wait_snapshot();
        let ok = ensure_host_imports(&runs);
        let (_, after, _) = engine_lock_wait_snapshot();
        if !ok {
            eprintln!("SKIP engine-entry count: no device or no VK_EXT_external_memory_host here");
            return;
        }
        assert_eq!(
            after - before,
            1,
            "{} resolvable runs took {} engine entries; the prologue is per entry, not per span",
            runs.len(),
            after - before
        );

        // An empty list asks nothing and must not enter at all.
        let (_, before, _) = engine_lock_wait_snapshot();
        assert!(ensure_host_imports(&[]));
        let (_, after, _) = engine_lock_wait_snapshot();
        assert_eq!(after, before);
    }
}
