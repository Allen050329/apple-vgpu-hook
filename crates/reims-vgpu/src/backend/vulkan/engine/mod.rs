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
mod context;
mod counters;
mod desc_arena;
mod device_lost;
mod digest;
mod draw_execution;
mod draw_phase;
mod draw_preparation;
mod draw_validation;
mod exec;
mod exec_compute;
mod facade_decline;
pub mod init_decline;
mod pools;
pub mod reason;
mod slab;
pub mod types;
pub mod vk_call;
#[cfg(feature = "host-window")]
mod window_present;

pub use compute_execution::ComputeExecutionDecline;
pub use compute_validation::ComputeValidationDecline;
pub use context::{FENCE_TIMEOUT_NS, MAX_DEVICE_RECREATES};
pub use counters::{CounterSnapshot, EngineCounters};
pub use device_lost::{DeviceLostDecline, DeviceLostOp};
pub use draw_execution::DrawExecutionDecline;
pub use draw_phase::{take_window as draw_phase_window, DrawPhaseWindow};
pub use draw_preparation::DrawPreparationDecline;
pub use draw_validation::DrawValidationDecline;
pub use facade_decline::EngineFacadeDecline;
pub use init_decline::InitDecline;
pub use reason::DrawReason;
pub use types::{
    BlendFactor, BlendOp, BlendStateResource, BufferContent, ColorWriteMask, ComputeBufferOutput,
    ComputeBufferResource, ComputeOutput, ComputeRequest, ComputeResidentSampleBind,
    ComputeSampledImageResource, ComputeStorageImageResource, ComputeStorageResidency, CullMode,
    DepthState, DrawError, DrawOutput, DrawRequest, DrawTicket, GuestRun, GuestRunSource,
    IndexType, IndexedDrawResource, LoadOp, PrimitiveTopology, SampledContentIdentity,
    SampledImageResource, SampledSource, SamplerAddressMode, SamplerBorderColor,
    SamplerCompareFunction, SamplerFilter, SamplerMipFilter, SamplerResource, ScissorResource,
    SecondaryColorTarget, SeedOrder, StencilFaceOps, StencilOp, StencilState,
    StorageBufferResource, StorageImageFormat, TargetIdentity, VertexAttributeFormat,
    VertexAttributeResource, VertexStepFunction, ViewportResource, WindowPresentSource,
    COLOR_INPUT_BINDING,
};
pub use vk_call::{VkCall, VkOp};
#[cfg(feature = "host-window")]
pub use window_present::{WindowCpuFrame, WindowPresentOutcome};

use caches::ObjectCaches;
use context::ContextOwner;
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use pools::ResourcePools;
use std::sync::atomic::Ordering;
use types::ComputeError;

struct EngineState {
    owner: ContextOwner,
    caches: ObjectCaches,
    pools: ResourcePools,
    counters: EngineCounters,
    #[cfg(feature = "host-window")]
    window_presenter: Option<window_present::WindowPresenter>,
}

impl EngineState {
    fn new() -> Self {
        Self {
            owner: ContextOwner::new(),
            caches: ObjectCaches::new(),
            pools: ResourcePools::new(),
            counters: EngineCounters::default(),
            #[cfg(feature = "host-window")]
            window_presenter: None,
        }
    }

    fn flush_device_derived(&mut self) {
        if let Some(ctx) = self.owner.ctx.as_ref() {
            unsafe {
                #[cfg(feature = "host-window")]
                if let Some(mut presenter) = self.window_presenter.take() {
                    presenter.destroy(ctx, Some(&mut self.pools));
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

/// Acquire the global engine lock. The single `ENGINE` mutex serializes all 34
/// engine entry points across the drain worker and the QEMU main/present path,
/// so this is on every one of them.
#[inline]
fn lock_engine() -> parking_lot::MutexGuard<'static, EngineState> {
    ENGINE.lock()
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
        #[cfg(feature = "host-window")]
        ref mut window_presenter,
        ..
    } = &mut *guard;
    if let Some(ctx) = owner.ctx.as_ref() {
        if let Err(error) = unsafe { ctx.device.device_wait_idle() } {
            let decline = VkCall::new(VkOp::GuestResetDeviceWaitIdle, error);
            crate::observe::Emit::decline("vulkan_guest_reset", &decline).fail_once(0);
        }
        unsafe {
            #[cfg(feature = "host-window")]
            if let Some(presenter) = window_presenter.as_mut() {
                presenter.release_pins_after_idle(pools);
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
#[cfg(feature = "host-window")]
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

/// Whether the host window is presenting from the engine's own device.
///
/// Read by the present-capture path on the drain worker, which must decide
/// whether to read the finished frame back into host memory *before* it does so
/// — deciding at publish time leaves the readback already paid for. A relaxed
/// atomic rather than [`lock_engine`] because that call site runs once per
/// present on the only thread that executes guest work, and taking the engine
/// lock there to read one bit would serialize it against the window thread's
/// own present.
#[cfg(feature = "host-window")]
static WINDOW_PRESENT_ATTACHED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Publish the window's rail choice. Called by the window thread from exactly
/// the two places that create and destroy the presenter.
#[cfg(feature = "host-window")]
pub fn note_window_present_attached(attached: bool) {
    WINDOW_PRESENT_ATTACHED.store(attached, Ordering::Release);
}

#[cfg(feature = "host-window")]
pub fn window_present_attached() -> bool {
    WINDOW_PRESENT_ATTACHED.load(Ordering::Acquire)
}

#[cfg(feature = "host-window")]
pub fn window_present_resize(width: u32, height: u32) {
    let mut guard = lock_engine();
    if let Some(presenter) = guard.window_presenter.as_mut() {
        presenter.resize(width, height);
    }
}

/// Present the current compositor resident through the engine-owned swapchain,
/// falling back to `cpu` for presents no resident carries. Acquire is
/// nonblocking, so a vblank wait never holds `ENGINE`.
#[cfg(feature = "host-window")]
pub fn window_present_frame(
    source: Option<&WindowPresentSource>,
    cpu: Option<WindowCpuFrame<'_>>,
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
    unsafe { presenter.present(ctx, pools, counters, source, cpu) }
}

/// Destroy the engine-owned surface while the native AppKit window still
/// exists. Called from winit's `exiting` callback.
#[cfg(feature = "host-window")]
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

/// Borrow form of [`execute_compute`].
pub fn execute_compute_request(req: &ComputeRequest) -> Result<ComputeOutput, ComputeError> {
    let mut guard = lock_engine();
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
/// Whether the window presenter would take this resident for a present at
/// `width`x`height`. Shares [`pools::slot_presentable`] with the presenter's own
/// selection so the two cannot answer differently.
///
/// Not gated on `host-window`, because the question is about the target registry
/// rather than about a window: `runtime::drain`'s `present_unbacked` gate asks it
/// to tell "the guest sent no full frame for this mid AND nothing can carry the
/// present" (a black frame) from "no full frame, but a resident carries it
/// anyway" (a census). That distinction has to be available on every Vulkan
/// build, not only the ones that opened a window.
pub fn resident_presentable(identity: &TargetIdentity, width: u32, height: u32) -> bool {
    let guard = lock_engine();
    guard
        .pools
        .registry_get(identity)
        .is_some_and(|slot| pools::slot_presentable(slot, width, height))
}

pub fn resident_content_ready(identity: &TargetIdentity) -> bool {
    let guard = lock_engine();
    guard
        .pools
        .registry_get(identity)
        .is_some_and(|s| s.content_ready)
}

/// The mapping content epoch this resident's pixels were stamped with, or
/// `None` when the identity is absent, evicted, or has not been vouched for
/// since its last draw.
///
/// Compared by the type-11 LOAD against
/// [`crate::model::MappingEntry::surface_content_epoch`]: equal means the
/// resident already holds exactly the bytes a CPU seed would upload, so the
/// pass may take [`LoadOp::LoadFromTarget`] and skip the upload. Every way the
/// answer can be unknown — no slot, recycled image, a draw since the stamp —
/// resolves to `None` and therefore to the seed.
pub fn resident_content_epoch(identity: &TargetIdentity) -> Option<u32> {
    let guard = lock_engine();
    guard.pools.registry_get(identity)?.content_epoch
}

/// Record that this resident holds the mapping's content as of `epoch`. Returns
/// false when the identity is absent or not content_ready, which the caller
/// must treat as "the elision is off for this surface" rather than ignore.
pub fn stamp_resident_content_epoch(identity: &TargetIdentity, epoch: u32) -> bool {
    let mut guard = lock_engine();
    guard.pools.registry_stamp_content_epoch(identity, epoch)
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
/// The present publish uses this so the displayed resident is not
/// reclaimed underneath the window on a present that does no draw.
pub fn touch_resident_target(identity: Option<&TargetIdentity>, now_ms: u64) {
    let Some(identity) = identity else {
        return;
    };
    let mut guard = lock_engine();
    guard.pools.registry_touch_at(identity, now_ms);
}

/// Which engine entry point's initialization prologue refused, for the
/// `vk_engine_probe` decline's `probe=` field.
///
/// [`EngineProbe::discriminant`] is the `fail_once` dedup key, so it is a
/// stable numbering and not an index: 1 through 6 are retired holes. 1, 2 and 3
/// named the present-proxy GPU stats oracle's context / pool / take prologues
/// (`present_stats_context`, `present_stats_pools`, `take_stats_context`); 4 and
/// 5 named the host-pointer import prologues (`host_import_context`,
/// `host_import_pools`), which went out with the import subsystem; 6 was
/// `compute_writeback_alignment`, which went out with the GPU-direct compute
/// writeback. Do not reuse them — a fail-log line already carrying one of those
/// keys must not be conflated with a new probe's.
#[derive(Clone, Copy, Debug)]
enum EngineProbe {
    StorageWriteWithoutFormat,
    ComputeCapable,
    SampledR32fLinearFilter,
}

impl EngineProbe {
    fn name(self) -> &'static str {
        match self {
            Self::StorageWriteWithoutFormat => "storage_write_without_format",
            Self::ComputeCapable => "compute_capable",
            Self::SampledR32fLinearFilter => "sampled_r32f_linear_filter",
        }
    }

    /// 1 through 6 are retired (see the type's docs); the rest keep the numbers
    /// they were first logged under.
    fn discriminant(self) -> u64 {
        match self {
            Self::StorageWriteWithoutFormat => 7,
            Self::ComputeCapable => 8,
            Self::SampledR32fLinearFilter => 9,
        }
    }
}

fn engine_probe_decline(probe: EngineProbe, error: &DrawError) -> crate::observe::Emit {
    crate::observe::Emit::decline("vk_engine_probe", error).field("probe", probe.name())
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
    let mut px = match read_target_inner(identity) {
        // The `slot.bgra` gate above already established the order, so the
        // reported one cannot disagree and the bytes pass through untouched.
        Ok(rb) => rb.pixels,
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

/// The six fallible Vulkan calls a whole-image readback makes, named per rail.
///
/// The rails differ in nothing else, but they must not share slugs: a
/// `reason=vk_readback_submit` that could have come from either the present
/// drain or a deferred compute flush names neither, which is the collapse the
/// typed [`VkOp`] vocabulary exists to prevent.
struct ReadbackOps {
    reset_cb: VkOp,
    begin_cb: VkOp,
    end_cb: VkOp,
    submit: VkOp,
    map: VkOp,
    /// `vkInvalidateMappedMemoryRanges`, which the readback owes whenever
    /// `MemoryClass::Readback` landed on a host-cached non-coherent type.
    invalidate: VkOp,
}

/// Copy level 0 of a resident color image to host bytes, tightly packed.
///
/// Shared by the target readback (present / Synchronize / Map / Store boundary)
/// and the pinned-storage deferred flush. `src_access` is the only Vulkan
/// difference: a render target may have a `COLOR_ATTACHMENT_WRITE` to drain, a
/// storage image cannot.
///
/// Async ring advance (retires only the one slot it reuses), NOT a whole-ring
/// quiesce: this reads content that is already ready, not an UNDEFINED-layout
/// seed, so the `ALL_COMMANDS → TRANSFER` barrier plus single-queue submission
/// order fully order the copy after every prior-submitted draw. `begin_entry_sync`
/// would block this guest-drain readback behind an unrelated in-flight heavy
/// draw — the `finish_us` tail. We wait only our own `fence` after submit, and
/// the slot stays pending for the ring to retire later.
#[allow(clippy::too_many_arguments)]
unsafe fn copy_image_level0_to_host(
    ctx: &context::DeviceContext,
    pools: &mut pools::ResourcePools,
    counters: &EngineCounters,
    image: ash::vk::Image,
    old_layout: ash::vk::ImageLayout,
    src_access: ash::vk::AccessFlags,
    width: u32,
    height: u32,
    rb_size: u64,
    ops: ReadbackOps,
) -> Result<Vec<u8>, DrawError> {
    let readback = pools.acquire_readback(ctx, rb_size, counters)?;
    let (cb, fence) = pools.begin_entry(ctx, counters)?;
    ctx.device
        .reset_command_buffer(cb, ash::vk::CommandBufferResetFlags::empty())
        .map_err(|e| DrawError::VkCall(VkCall::new(ops.reset_cb, e)))?;
    ctx.device
        .begin_command_buffer(
            cb,
            &ash::vk::CommandBufferBeginInfo::default()
                .flags(ash::vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )
        .map_err(|e| DrawError::VkCall(VkCall::new(ops.begin_cb, e)))?;
    if old_layout != ash::vk::ImageLayout::TRANSFER_SRC_OPTIMAL {
        let barrier = [ash::vk::ImageMemoryBarrier::default()
            .src_access_mask(src_access)
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
        .map_err(|e| DrawError::VkCall(VkCall::new(ops.end_cb, e)))?;
    let queue = ctx.queue();
    let cbs = [cb];
    let si = ash::vk::SubmitInfo::default().command_buffers(&cbs);
    ctx.device
        .queue_submit(queue, &[si], fence)
        .map_err(|e| DrawError::VkCall(VkCall::new(ops.submit, e)))?;
    let cleanup = pools.seal_entry(Vec::new(), Vec::new());
    pools.finish_entry_async(cleanup);
    pools.wait_entry_fence(ctx, counters, fence)?;
    pools::read_back_slot(ctx, &readback, rb_size, ops.map, ops.invalidate)
}

/// A resident target's pixels plus the physical channel order they came out in.
///
/// Reported rather than derivable, and read from the registry slot under the
/// same lock as the copy, so it is the order of the image the bytes were
/// actually copied out of. A caller that re-derived it from the identity would
/// be restating a rule the engine owns; when the two disagree the symptom is an
/// R/B exchange on a whole frame, which is a colour defect no assertion in this
/// crate was watching for.
pub struct TargetReadback {
    pub pixels: Vec<u8>,
    /// BGRA8 when true, semantic RGBA8 otherwise.
    pub bgra: bool,
}

impl TargetReadback {
    /// The frame in semantic RGBA8, exchanging R and B only when it is not
    /// already in that order.
    pub fn into_rgba8(mut self) -> Vec<u8> {
        if self.bgra {
            for px in self.pixels.chunks_exact_mut(4) {
                px.swap(0, 2);
            }
        }
        self.pixels
    }

    /// The frame in guest scanout order (BGRA8), exchanging only when needed.
    ///
    /// The mirror of `into_rgba8`, for the guest-page writers that are declared in
    /// scanout order (`mapping_write::write_bgra8`). Both exist so that neither
    /// caller has to know which namespace it is reading: a `Surface` resident is
    /// already BGRA and this is a no-op, and a resident that is not stays correct
    /// instead of landing R and B exchanged in guest memory.
    pub fn into_bgra8(mut self) -> Vec<u8> {
        if !self.bgra {
            for px in self.pixels.chunks_exact_mut(4) {
                px.swap(0, 2);
            }
        }
        self.pixels
    }
}

fn read_target_inner(identity: &TargetIdentity) -> Result<TargetReadback, DrawError> {
    let mut guard = lock_engine();
    let EngineState {
        ref mut owner,
        ref mut pools,
        ref counters,
        ..
    } = &mut *guard;
    let ctx = owner.ensure(counters)?;
    unsafe { pools.ensure_init(ctx, counters)? };
    let slot = pools.registry_get(identity).ok_or(DrawError::TargetRead(
        reason::TargetReadDecline::UnknownIdentity,
    ))?;
    if !slot.content_ready {
        return Err(DrawError::TargetRead(
            reason::TargetReadDecline::NoReadyContent,
        ));
    }
    let width = slot.width;
    let height = slot.height;
    let image = slot.image;
    let old_layout = slot.layout;
    let bgra = slot.bgra;
    let rb_size = (width as u64) * (height as u64) * 4;
    unsafe {
        let out = copy_image_level0_to_host(
            ctx,
            pools,
            counters,
            image,
            old_layout,
            ash::vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                | ash::vk::AccessFlags::TRANSFER_WRITE
                | ash::vk::AccessFlags::SHADER_WRITE,
            width,
            height,
            rb_size,
            ReadbackOps {
                reset_cb: VkOp::ReadbackResetCb,
                begin_cb: VkOp::ReadbackBeginCb,
                end_cb: VkOp::ReadbackEndCb,
                submit: VkOp::ReadbackSubmit,
                map: VkOp::ReadbackMap,
                invalidate: VkOp::ReadbackInvalidate,
            },
        )?;
        pools.registry_set_layout(identity, ash::vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
        counters.note_target_read(rb_size);
        Ok(TargetReadback { pixels: out, bgra })
    }
}

/// Full-frame readback of a resident target (present / Synchronize / Map / Store boundary).
pub fn read_target(identity: &TargetIdentity) -> Result<TargetReadback, DrawError> {
    read_target_inner(identity)
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
        pools.compute_resident_snapshot(identity).ok_or({
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
    unsafe {
        let out = copy_image_level0_to_host(
            ctx,
            pools,
            counters,
            image,
            old_layout,
            // A storage image is never a color attachment, so there is no
            // `COLOR_ATTACHMENT_WRITE` to drain here.
            ash::vk::AccessFlags::TRANSFER_WRITE | ash::vk::AccessFlags::SHADER_WRITE,
            key.width,
            key.height,
            rb_size,
            ReadbackOps {
                reset_cb: VkOp::StorageReadResetCb,
                begin_cb: VkOp::StorageReadBeginCb,
                end_cb: VkOp::StorageReadEndCb,
                submit: VkOp::StorageReadSubmit,
                map: VkOp::StorageReadMap,
                invalidate: VkOp::StorageReadInvalidate,
            },
        )?;
        pools.set_resident_storage_layout(identity, ash::vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
        pools.pin_resident_storage(identity, false);
        counters.note_compute_deferred_flush(rb_size);
        Ok((out, texel))
    }
}

/// The non-pinned resident-target slot cap. Exposed so a test that must blow
/// past the LRU sweep derives its filler count from the live value instead of
/// hard-coding one — `vk_engine_parity` previously fixed 70 fillers against a
/// cap later retuned to 320, so no eviction fired and its assert could not hold.
pub fn registry_cap() -> usize {
    pools::REGISTRY_CAP
}

/// Advance the wall-clock resident-target idle-drain clock to `now_ms`, keep the
/// currently-presented target (`display`) alive, and reclaim aged non-pinned
/// residents. Called from the poll heartbeat (so the clock keeps ticking when the
/// guest stops publishing) and each present publish. No-op before the device
/// context exists.
pub fn maintain_idle_residents(display: Option<&TargetIdentity>, now_ms: u64) {
    let mut guard = lock_engine();
    let EngineState {
        ref mut owner,
        ref mut pools,
        ..
    } = &mut *guard;
    let Some(ctx) = owner.ctx.as_ref() else {
        return;
    };
    unsafe {
        pools.advance_registry_touch_and_drain(ctx, now_ms, display);
    }
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

/// Test-only: destroy device, clear recreate budget, rebuild on next draw.
pub fn test_reset_engine() {
    let mut g = lock_engine();
    if let Some(mut ctx) = g.owner.ctx.take() {
        unsafe {
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
            EngineProbe::StorageWriteWithoutFormat,
            EngineProbe::ComputeCapable,
            EngineProbe::SampledR32fLinearFilter,
        ] {
            let line = engine_probe_decline(probe, &error).render();
            assert!(line.starts_with("vk_engine_probe reason=vk_exec_submit "));
            assert!(line.ends_with(&format!(" probe={}", probe.name())));
        }
    }
}
