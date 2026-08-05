//! Reims vGPU host path — single crate.
//!
//! | Module | Role |
//! | --- | --- |
//! | [`contract`] | Stable facts: formats, layouts, pure arithmetic |
//! | [`model`] | Live guest-visible state (regs, rings, objects, present) |
//! | [`runtime`] | Drain / parse / resolve / plan / HostActions |
//! | [`backend`] | Trait + self-contained [`backend::metal`] / [`backend::vulkan`] |
//! | [`qemu`] | QEMU C ABI surface only |
//!
//! Features: exactly one of `backend-metal` (default) or `backend-vulkan`.
//! Vulkan product path is self-contained `ash` ([`backend::vulkan::engine`]).
//!
//! # The three supported arms
//!
//! A build is exactly one of these, and the guards below reject anything else:
//!
//! | Arm | `cfg` | Host GPU API |
//! | --- | --- | --- |
//! | Metal | `all(feature = "backend-metal", target_os = "macos")` | native Metal |
//! | Vulkan / MoltenVK | `all(feature = "backend-vulkan", target_os = "macos")` | MoltenVK |
//! | Vulkan / native | `all(feature = "backend-vulkan", target_os = "linux")` | native ICD |
//!
//! **Gate the host on `target_os` and nothing else.** `macos` and `linux` are
//! the only two values this crate names, so the three arms differ in one term
//! each and a reader greps one key to find every host gate.
//!
//! There is **no** host-stub Metal arm. `backend-metal` off macOS has no Metal
//! to call, so it is a compile error rather than a binary that links and cannot
//! draw.
//!
//! The consequence the rest of the crate relies on: **the Metal arm and the
//! Vulkan arms partition every buildable configuration.** So the engine path is
//! spelled positively as `feature = "backend-vulkan"` and the Metal path as
//! `all(feature = "backend-metal", target_os = "macos")`, with no negation
//! of one standing in for the other. Do not reintroduce
//! `not(all(feature = "backend-metal", target_os = "macos"))` as a spelling
//! of "the engine path" — it says what the build is *not*, which stops being
//! equivalent the moment a fourth arm exists.

#![deny(unsafe_op_in_unsafe_fn)]
#![deny(rust_2018_idioms)]

#[cfg(all(feature = "backend-metal", feature = "backend-vulkan"))]
compile_error!("select exactly one of backend-metal or backend-vulkan");

#[cfg(not(any(feature = "backend-metal", feature = "backend-vulkan")))]
compile_error!("select exactly one of backend-metal or backend-vulkan");

#[cfg(all(feature = "backend-metal", not(target_os = "macos")))]
compile_error!(
    "backend-metal requires target_os = \"macos\": there is no host-stub Metal \
     arm. Use --no-default-features --features backend-vulkan,host-window on \
     any other host."
);

// Vulkan reaches the GPU through MoltenVK on macOS and a native ICD on Linux.
// Any other host is untested rather than known-broken — name it here so a new
// port is a deliberate edit to this list, not an accident.
#[cfg(all(
    feature = "backend-vulkan",
    not(any(target_os = "macos", target_os = "linux"))
))]
compile_error!(
    "backend-vulkan is supported on target_os = \"macos\" (MoltenVK) and \
     target_os = \"linux\" (native ICD) only"
);

pub mod contract;
/// Every environment variable this device reads, and the rule that an override
/// may only narrow what it does — see the module doc.
pub mod env;
pub mod model;
/// Crate-wide observability: the always-on fail sink and the decline
/// vocabulary. Above `runtime/` because every subsystem owes the reader a
/// reason, and `translate/` + `caps/` must be able to name one without
/// depending on `runtime/`.
pub mod observe;
pub mod runtime;

pub mod backend;
pub mod qemu;

/// Host-owned presentation window (winit + VkSurfaceKHR) — see
/// [[host-window]]. The `host-window` feature implies `backend-vulkan`, and is
/// enabled for every verification command the x86 pathway is checked with.
#[cfg(feature = "host-window")]
pub mod host_window;

/// The device side of that window: its link, its four QEMU entry points, and
/// the two publish paths the drain and the poll call. Unconditional, because
/// the entry points keep a stub arm without the feature and the QEMU ABI
/// surface must be the same shape either way.
mod window_publish;
pub use window_publish::{
    device_window_run_main, device_window_set_early_fb, device_window_start, device_window_stop,
};

pub use backend::Backend;
pub use contract::pixel_format;
// Convenience re-exports used by qemu ABI and tests
pub use model::{Device, DeviceId};
pub use runtime::{HostAction, HostOps};

// --- Device lifecycle registry (used by qemu::abi) ---------------------------

use once_cell::sync::Lazy;
use parking_lot::Mutex;
use std::collections::{HashMap, VecDeque};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;

use crate::qemu::host_ops::{NullHost, QemuHost, ReimsVgpuHostOps};

#[cfg(feature = "backend-metal")]
type SelectedBackend = backend::metal::MetalBackend;

#[cfg(feature = "backend-vulkan")]
type SelectedBackend = backend::vulkan::VulkanBackend;

/// Mutable protocol/backend state. The drain worker may hold this lock across
/// shader translation and a GPU wait, so MMIO producers must never wait for it.
struct DeviceInner {
    device: Device<SelectedBackend>,
    /// Actions for the QEMU BH to apply after drain.
    actions: VecDeque<HostAction>,
}

#[derive(Clone, Copy, Debug)]
struct QueuedGfxWrite {
    offset: u64,
    data: u64,
    size: u32,
    /// When the vCPU published this write, or `None` when it was applied
    /// straight through without ever entering the queue.
    ///
    /// The guest's store retires the moment this is pushed, so the guest cannot
    /// see the delay; the age measured against this stamp is the only place the
    /// deferral becomes visible. See [`crate::runtime::drain::DoorbellCensus`].
    queued_at: Option<std::time::Instant>,
}

/// One live device. Registry lookup and MMIO ingress remain short even while
/// `inner` is owned by the ordered render worker.
struct BoundDevice {
    inner: Mutex<DeviceInner>,
    gfx_ingress: Mutex<VecDeque<QueuedGfxWrite>>,
    gfx_read_cache: Mutex<HashMap<(u64, u32), u64>>,
    gfx_read_busy_logged: AtomicBool,
    /// Prompt HostActions (IRQ pulses, cursor moves): poppable without `inner`
    /// so the BH delivers them while the drain worker still owns the device
    /// lock. Scanout/glyph actions stay in `DeviceInner::actions`.
    prompt_actions: Mutex<VecDeque<HostAction>>,
    /// Lock-free clones of the read-to-clear interrupt-status registers
    /// (`state.gfx.interrupt_status_disp` / `_gpu`): the guest ISR read at
    /// 0x1014/0x1018 must observe live bits mid-drain, never a stale cache.
    intr_disp: Arc<AtomicU32>,
    intr_gpu: Arc<AtomicU32>,
    /// Child channels the guest has rung, OR'd from the vCPU thread with no
    /// device lock; see [`model::GfxRegs::child_doorbell_rung`].
    child_doorbell_rung: Arc<AtomicU32>,
    /// Lock-free clone of the fault status (0x102c) — the ISR's third read.
    intr_fault: Arc<AtomicU32>,
    /// Lock-free clone of the main-FIFO consumer counter (0x100c): the guest
    /// `writeFifo` producer spin must observe drain progress live, not a
    /// cached snapshot from before the tranche.
    fifo_read_live: Arc<AtomicU32>,
    /// An accepted present is waiting for QEMU to consume its scanout action.
    /// Kept outside `inner` so new worker wakeups can yield without racing the
    /// main-loop copy for the same device lock.
    present_action_pending: AtomicBool,
    /// Monotonic per-boot publication of `frame_flush_seen`. QEMU refresh must
    /// never reinterpret a contended device lock as a return to BAR1/EFI.
    present_boundary_seen: AtomicBool,
    /// Monotonic reset sequence for cross-boot lifecycle diagnosis.
    reset_count: AtomicU64,
    /// Lock-free snapshot of the display VBL state, republished on every
    /// lock-acquired `device_poll`. Lets a *contended* poll (the drain worker
    /// owns `inner`) still pulse VBL so the guest keeps its display time base
    /// under load — without it, `device_poll` early-returns on the `try_lock`
    /// miss and drops the VBL entirely (kb present-thrash-proxies: VBL collapses
    /// to ~7 Hz under interaction). `vbl_shared_gpa == 0` ⇒ not online yet.
    vbl_shared_gpa: AtomicU64,
    vbl_display_index: AtomicU32,
    vbl_online: AtomicBool,
    /// Wall-clock ms of the last VBL claimed by either the locked or contended
    /// poll path. One shared limiter keeps guest pacing independent of which
    /// path happens to win the device lock.
    vbl_last_us: AtomicU64,
    /// QEMU HostOps (GPA / clock / schedule worker). None in pure unit tests.
    ops: Option<ReimsVgpuHostOps>,
    /// Host-owned presentation window ([[host-window]]), once
    /// `device_window_start` has spawned it. `None` on a normal QEMU-display
    /// boot (the window is opt-in behind `REIMS_VGPU_WINDOW`).
    #[cfg(feature = "host-window")]
    window: Mutex<Option<window_publish::WindowLink>>,
    /// Early-boot framebuffer (BAR1 GOP) registered by the C shim, shown in the
    /// window until the product present path latches.
    #[cfg(feature = "host-window")]
    early_fb: Mutex<Option<window_publish::EarlyFb>>,
    /// Monotonic ns of the last early frame pushed to the window (poll-path
    /// throttle so the pre-boundary pump does not memcpy the FB every 4 ms).
    #[cfg(feature = "host-window")]
    early_last_ns: AtomicU64,
}

type DeviceMap = HashMap<u64, Arc<BoundDevice>>;

static DEVICES: Lazy<Mutex<DeviceMap>> = Lazy::new(|| Mutex::new(HashMap::new()));
static NEXT_ID: Lazy<Mutex<u64>> = Lazy::new(|| Mutex::new(1));

fn device_slot(id: u64) -> Option<Arc<BoundDevice>> {
    DEVICES.lock().get(&id).cloned()
}

fn schedule_device(slot: &BoundDevice) {
    let Some(ops) = slot.ops else {
        return;
    };
    if let Some(schedule) = ops.schedule_bh {
        // SAFETY: QEMU owns ctx for the device lifetime; schedule_bh is the
        // thread-safe wake callback supplied by the shim.
        unsafe { schedule(ops.ctx) }
    }
}

#[inline]
fn publish_present_boundary(slot: &BoundDevice, frame_flush_seen: bool) {
    if frame_flush_seen {
        slot.present_boundary_seen.store(true, Ordering::Release);
    }
}

fn apply_gfx_write(inner: &mut DeviceInner, slot: &BoundDevice, write: QueuedGfxWrite) {
    match write.queued_at {
        Some(at) => {
            runtime::drain::note_doorbell_queued(write.offset, at.elapsed().as_micros() as u64)
        }
        None => runtime::drain::note_doorbell_direct(),
    }
    if let Some(ops) = slot.ops {
        let mut host = QemuHost::new(&ops, &mut inner.actions, &slot.prompt_actions);
        inner
            .device
            .gfx_write(&mut host, write.offset, write.data, write.size);
    } else {
        let mut host = NullHost;
        inner
            .device
            .gfx_write(&mut host, write.offset, write.data, write.size);
    }
}

/// Apply queued MMIO writes in publication order. Lock order is ingress then
/// inner everywhere; producers use `try_lock` for inner and therefore never
/// wait behind shader translation/GPU work.
fn lock_for_drain(slot: &BoundDevice) -> parking_lot::MutexGuard<'_, DeviceInner> {
    let mut ingress = slot.gfx_ingress.lock();
    let mut inner = slot.inner.lock();
    while let Some(write) = ingress.pop_front() {
        apply_gfx_write(&mut inner, slot, write);
    }
    drop(ingress);
    // Here rather than only inside `drain_pending`, because this is the one
    // point every entry to the drain passes through. `device_drain` returns
    // before `drain_pending` when the device has no host ops, and
    // `publish_stranded_fifos` re-publishes from `active_child_mask` — a ring
    // left unfolded would be invisible to both.
    runtime::drain::fold_rung_child_doorbells(&mut inner.device.state);
    inner
}

fn make_backend() -> SelectedBackend {
    #[cfg(feature = "backend-metal")]
    {
        backend::metal::MetalBackend::new()
    }
    #[cfg(feature = "backend-vulkan")]
    {
        backend::vulkan::VulkanBackend::new()
    }
}

/// Create a device. `ops` is the QEMU host-service table (nullable for tests).
///
/// `page_shift` must be [`model::PAGE_SHIFT_X86`] (12) or [`model::PAGE_SHIFT_ARM64E`] (14).
/// There is no default (including no `0` → arm); unsupported values return `None`.
pub fn device_create(ops: Option<ReimsVgpuHostOps>, page_shift: u32) -> Option<u64> {
    use model::{PAGE_SHIFT_ARM64E, PAGE_SHIFT_X86};
    if page_shift != PAGE_SHIFT_ARM64E && page_shift != PAGE_SHIFT_X86 {
        return None;
    }
    let mut id_guard = NEXT_ID.lock();
    let id = *id_guard;
    *id_guard = id.saturating_add(1);
    drop(id_guard);
    let backend = make_backend();
    let dev = Device::new(DeviceId(id), backend, page_shift);
    let intr_disp = Arc::clone(&dev.state.gfx.interrupt_status_disp);
    let intr_gpu = Arc::clone(&dev.state.gfx.interrupt_status_gpu);
    let child_doorbell_rung = Arc::clone(&dev.state.gfx.child_doorbell_rung);
    let intr_fault = Arc::clone(&dev.state.gfx.interrupt_fault);
    let fifo_read_live = Arc::clone(&dev.state.gfx.fifo_read);
    DEVICES.lock().insert(
        id,
        Arc::new(BoundDevice {
            inner: Mutex::new(DeviceInner {
                device: dev,
                actions: VecDeque::new(),
            }),
            gfx_ingress: Mutex::new(VecDeque::new()),
            gfx_read_cache: Mutex::new(HashMap::new()),
            gfx_read_busy_logged: AtomicBool::new(false),
            prompt_actions: Mutex::new(VecDeque::new()),
            intr_disp,
            intr_gpu,
            child_doorbell_rung,
            intr_fault,
            fifo_read_live,
            present_action_pending: AtomicBool::new(false),
            present_boundary_seen: AtomicBool::new(false),
            reset_count: AtomicU64::new(0),
            vbl_shared_gpa: AtomicU64::new(0),
            vbl_display_index: AtomicU32::new(0),
            vbl_online: AtomicBool::new(false),
            vbl_last_us: AtomicU64::new(0),
            ops,
            #[cfg(feature = "host-window")]
            window: Mutex::new(None),
            #[cfg(feature = "host-window")]
            early_fb: Mutex::new(None),
            #[cfg(feature = "host-window")]
            early_last_ns: AtomicU64::new(0),
        }),
    );
    Some(id)
}

pub fn device_reset(id: u64) -> bool {
    if let Some(slot) = device_slot(id) {
        let mut d = lock_for_drain(&slot);
        let seq = slot.reset_count.fetch_add(1, Ordering::Relaxed) + 1;
        let state = &d.device.state;
        let mappings = state.mappings.len();
        let tasks = state.tasks.iter().filter(|task| task.active).count();
        let host_surfaces = state.host_surfaces.len();
        let host_textures = state.host_texture_surfaces.len();
        let host_gvas = state.host_gva_surfaces.len();
        let host_linear = state.host_linear_textures.len();
        let frame_valid = state.present.frame_valid;
        let frame_mapping = state.present.frame_mapping;
        let boundary = state.present.frame_flush_seen;
        let views = if let Some(ops) = slot.ops {
            let DeviceInner { device, actions } = &mut *d;
            let mut host = QemuHost::new(&ops, actions, &slot.prompt_actions);
            device.reset_with_host(&mut host)
        } else {
            d.device.reset();
            0
        };
        observe::off(format!(
            "device_reset id={id} seq={seq} mappings={mappings} tasks={tasks} host_surface={host_surfaces} host_texture={host_textures} host_gva={host_gvas} host_linear={host_linear} frame_valid={} frame_mapping={frame_mapping} boundary={} unmapped_views={views}",
            u8::from(frame_valid),
            u8::from(boundary)
        ));
        d.actions.clear();
        slot.prompt_actions.lock().clear();
        slot.gfx_read_cache.lock().clear();
        slot.present_action_pending.store(false, Ordering::Release);
        slot.present_boundary_seen.store(false, Ordering::Release);
        runtime::census::present_proxy::reset_for_device();
        true
    } else {
        false
    }
}

pub fn device_destroy(id: u64) -> bool {
    DEVICES.lock().remove(&id).is_some()
}

pub fn device_gfx_read(id: u64, offset: u64, size: u32) -> Option<u64> {
    use model::{
        GFX_REG_FIFO_READ, GFX_REG_INTR_FAULT, GFX_REG_INTR_STATUS_DISP, GFX_REG_INTR_STATUS_GPU,
    };
    let slot = device_slot(id)?;
    // Guest spin/ISR registers: served lock-free from the shared atomics so a
    // drain-tranche-held device lock never turns a fresh stamp signal into a
    // stale cached mask (0x1014/0x1018 r2c) nor hides drain progress from the
    // writeFifo producer spin (0x100c) or the ISR fault read (0x102c).
    if size == 4 {
        if offset == GFX_REG_INTR_STATUS_DISP {
            return Some(slot.intr_disp.swap(0, Ordering::AcqRel) as u64);
        }
        if offset == GFX_REG_INTR_STATUS_GPU {
            return Some(slot.intr_gpu.swap(0, Ordering::AcqRel) as u64);
        }
        if offset == GFX_REG_FIFO_READ {
            return Some(slot.fifo_read_live.load(Ordering::Acquire) as u64);
        }
        if offset == GFX_REG_INTR_FAULT {
            return Some(slot.intr_fault.load(Ordering::Acquire) as u64);
        }
    }
    if let Some(mut d) = slot.inner.try_lock() {
        let value = d.device.gfx_read(offset, size);
        slot.gfx_read_cache.lock().insert((offset, size), value);
        slot.gfx_read_busy_logged.store(false, Ordering::Relaxed);
        return Some(value);
    }
    let value = slot
        .gfx_read_cache
        .lock()
        .get(&(offset, size))
        .copied()
        .unwrap_or(0);
    if !slot.gfx_read_busy_logged.swap(true, Ordering::Relaxed) {
        observe::fail(format!(
            "device_lock_busy reason=gfx_read_deferred offset={offset:#x} size={size} cached={value:#x}"
        ));
    }
    Some(value)
}

pub fn device_gfx_write(id: u64, offset: u64, data: u64, size: u32) -> bool {
    use model::{GFX_REG_INTR_STATUS_DISP, GFX_REG_INTR_STATUS_GPU};
    let Some(slot) = device_slot(id) else {
        return false;
    };
    // Interrupt-status mask clears are order-independent of FIFO doorbells;
    // apply them lock-free instead of queueing behind a busy drain tranche.
    if size == 4 {
        if offset == GFX_REG_INTR_STATUS_DISP {
            slot.intr_disp.fetch_and(!(data as u32), Ordering::AcqRel);
            return true;
        }
        if offset == GFX_REG_INTR_STATUS_GPU {
            slot.intr_gpu.fetch_and(!(data as u32), Ordering::AcqRel);
            return true;
        }
        // The child doorbell, which measurement says is the *entire* queueing
        // stall on this pathway: `gfx_doorbell_delay` reads `offsets=1` on
        // every window that queued anything, ~100 rings a second applied up to
        // 45 ms late, and that delay is the drain tranche the write could not
        // take the lock through.
        //
        // It is the one register that can be served this way, because it
        // carries no state the decode depends on — its effect is to say a
        // channel has work. `fold_rung_child_doorbells` turns the bit into
        // `active_child_mask` / `pending.child_mask`, which is exactly what the
        // locked handler in `runtime::mmio` does for the same register.
        //
        // The channel-number check mirrors that handler rather than trusting
        // the guest: a value outside the channel range names no channel, and
        // shifting by it would be undefined. An out-of-range ring is dropped
        // here as it is there, and deliberately still schedules nothing.
        if offset == model::GFX_REG_CHILD_DOORBELL || offset == model::GFX_REG_CHILD_REPLAY_DOORBELL
        {
            let channel = data as u32;
            if model::is_child_channel(channel) {
                slot.child_doorbell_rung
                    .fetch_or(1u32 << channel, Ordering::AcqRel);
                runtime::drain::note_doorbell_lock_free();
                schedule_device(&slot);
            }
            return true;
        }
    }
    let mut write = QueuedGfxWrite {
        offset,
        data,
        size,
        queued_at: None,
    };
    let mut ingress = slot.gfx_ingress.lock();
    if ingress.is_empty() {
        if let Some(mut inner) = slot.inner.try_lock() {
            apply_gfx_write(&mut inner, &slot, write);
            return true;
        }
    }
    // Stamped only on the path that actually defers, so the direct path pays no
    // clock read at all.
    write.queued_at = Some(std::time::Instant::now());
    ingress.push_back(write);
    drop(ingress);
    schedule_device(&slot);
    true
}

/// Take the device lock from the vCPU thread, measuring the wait.
///
/// The guest's MMIO access is stopped for exactly as long as this blocks, and
/// the drain worker holds this same lock across a full-surface readback. Every
/// other figure about that stall is taken from the holder's side, which makes
/// the step to "the guest missed a frame" an inference; this measures it where
/// it is actually paid.
///
/// The uncontended path takes `try_lock` and never reads the clock, so a fast
/// access pays nothing for the instrument.
fn lock_device_for_vcpu(slot: &BoundDevice) -> impl std::ops::DerefMut<Target = DeviceInner> + '_ {
    if let Some(guard) = slot.inner.try_lock() {
        runtime::drain::note_vcpu_lock_free();
        return guard;
    }
    let waited = std::time::Instant::now();
    let guard = slot.inner.lock();
    runtime::drain::note_vcpu_lock_wait(waited.elapsed().as_micros() as u64);
    guard
}

pub fn device_iosfc_read(id: u64, offset: u64, size: u32) -> Option<u64> {
    let slot = device_slot(id)?;
    let d = lock_device_for_vcpu(&slot);
    Some(d.device.iosfc_read(offset, size))
}

pub fn device_iosfc_write(id: u64, offset: u64, data: u64, size: u32) -> bool {
    let Some(slot) = device_slot(id) else {
        return false;
    };
    let mut d = lock_device_for_vcpu(&slot);
    if let Some(ops) = slot.ops {
        let DeviceInner { device, actions } = &mut *d;
        let mut host = QemuHost::new(&ops, actions, &slot.prompt_actions);
        device.iosfc_write(&mut host, offset, data, size);
    } else {
        let mut host = NullHost;
        d.device.iosfc_write(&mut host, offset, data, size);
    }
    true
}

/// Worker body: drain pending FIFOs using QEMU GPA callbacks; enqueue HostActions.
pub fn device_drain(id: u64) -> bool {
    let Some(slot) = device_slot(id) else {
        return false;
    };
    // The action BH needs the same device state to copy +0x188. A doorbell may
    // wake this worker before that BH runs; do not reacquire the lock and hide
    // the queued scanout behind another synchronous render/compute tranche.
    if slot.present_action_pending.load(Ordering::Acquire) {
        runtime::drain::note_drain_skipped();
        return true;
    }
    let mut d = lock_for_drain(&slot);
    let Some(ops) = slot.ops else {
        // No host services — nothing to resolve from guest RAM.
        return true;
    };
    let DeviceInner { device, actions } = &mut *d;
    let mut host = QemuHost::new(&ops, actions, &slot.prompt_actions);
    // Presentation-path selector for this tranche: with a live host window the
    // drain publishes frames + self-acks; without one every present must
    // enqueue the CPU `ScanoutUpdate` and the ack belongs to the console paint
    // (see `enqueue_present_scanout` / the drain tail below).
    #[cfg(feature = "host-window")]
    {
        device.state.present.window_active = slot.window.lock().is_some();
    }
    #[cfg(not(feature = "host-window"))]
    {
        device.state.present.window_active = false;
    }
    // Split the tranche's two phases: guest work, then our host-window export.
    // Both hold the device lock, and which one owns the worker's wall clock is
    // the question `drain_duty` exists to answer.
    let tranche_started = std::time::Instant::now();
    device.drain(&mut host);
    // Submit any deferred draw batch before the worker sleeps: consumers
    // inside the tranche flush on their own (engine begin_entry), this bounds
    // only the idle-tail latency of the last same-target run.
    #[cfg(feature = "backend-vulkan")]
    backend::vulkan::engine::flush_batched_draws();
    publish_present_boundary(&slot, device.state.present.frame_flush_seen);
    let drain_us = tranche_started.elapsed().as_micros() as u64;
    let publish_started = std::time::Instant::now();
    // Push the finished present frame to the host-owned window (if running).
    // Off the QEMU main loop; a small dedicated mutex, never the render lock.
    #[cfg(feature = "host-window")]
    window_publish::publish_window_frame(&slot, &mut device.state);
    runtime::drain::note_drain_tranche(drain_us, publish_started.elapsed().as_micros() as u64);
    // Same one-second cadence, so the cache trend lines up row-for-row with
    // `store_routes` and `drain_duty`. Measure-only; see `note_cache_levels`.
    runtime::surface_cache::note_cache_levels(&device.state, &host);
    // The present-completion ack, re-homed off the QEMU paint — ONLY while the
    // host window is the display. With the window live no per-present
    // `ScanoutUpdate` is enqueued, so `device_scanout_copy` — the only other
    // caller of `note_present_paint_consumed` — will not run for this present.
    // Acking here clears `unpainted_presents` (releasing the DisplaySwap
    // backpressure gate at `MAX_UNPAINTED_PRESENTS`) and `host_action_yield`, so
    // the check below leaves `present_action_pending` clear and the worker keeps
    // draining. Without this the display channel wedges on the second present.
    //
    // On the window path the ack is deliberately NOT keyed on "a frame was
    // published": the publish legitimately early-returns on a duplicate
    // (mapping, generation), on a frame not yet valid, and on a short buffer. An
    // ack that fired only when the window took a fresh frame would wedge on the
    // first repeated present.
    //
    // Without a window the QEMU console owns the paint: `enqueue_present_scanout`
    // enqueued the `ScanoutUpdate`, `host_action_yield` stays set, the flag below
    // arms `present_action_pending`, and `device_scanout_copy` both paints and
    // acks. Pre-acking here would let `device_scanout_copy`'s nonblocking
    // `try_lock` path swallow the paint as `Unchanged` under worker contention —
    // the frozen-console class this split fixes.
    if device.state.present.window_active {
        runtime::drain::note_present_paint_consumed(&mut device.state);
    }
    if device.state.pending.host_action_yield {
        slot.present_action_pending.store(true, Ordering::Release);
    }
    true
}

/// Periodic tick (gfx_update / poll): archive `poll_tick` subset.
///
/// - Dekker rescue: publish main/child/iosfc work to the asynchronous drain
///   owner when producer state may have advanced without a doorbell.
/// - Re-drive display ONLINE after guest enable() publishes the mask.
///
/// Enqueues HostActions (gfx IRQ / scanout); QEMU must deliver actions after
/// this call.
pub fn device_poll(id: u64) -> bool {
    let Some(slot) = device_slot(id) else {
        return false;
    };
    let Some(mut d) = slot.inner.try_lock() else {
        // Contended: the drain worker owns `inner` doing present/GPU-encode.
        // The full poll below would early-return and drop the VBL — under load
        // that starves the guest's only display time base (present-complete is
        // inert; kb present-thrash-proxies). Pulse VBL lock-free from the state
        // the last successful poll published, so pacing survives the contention.
        vbl_contended_pulse(&slot);
        return true;
    };
    let Some(ops) = slot.ops else {
        return true;
    };
    let DeviceInner { device, actions } = &mut *d;
    let mut host = QemuHost::new(&ops, actions, &slot.prompt_actions);
    // Before the rescue reads `active_child_mask`, which is the mask a
    // lock-free ring lands in only once folded. Without this the Dekker rescue
    // could not see the very channels the doorbell rail is responsible for.
    runtime::drain::fold_rung_child_doorbells(&mut device.state);
    runtime::drain::publish_stranded_fifos(&mut device.state, &mut host);
    runtime::drain::try_display_online(&mut device.state, &mut host);
    // After ONLINE, pulse VBL so the guest compositor has a display time base
    //. Missing VBL → clear-only dual-mid present thrash.
    runtime::drain::signal_display_vbl(&mut device.state, &mut host, &slot.vbl_last_us);
    // Republish the lock-free VBL snapshot for the contended fast path above.
    // These change only at online-ack/reinit, but publishing every poll keeps
    // the snapshot fresh with no extra synchronization on the rare-change path.
    slot.vbl_shared_gpa
        .store(device.state.display.shared_gpa, Ordering::Release);
    slot.vbl_display_index
        .store(device.state.display.display_index, Ordering::Release);
    slot.vbl_online
        .store(device.state.display.online_acked, Ordering::Release);
    // Census both source polls and the independently time-gated VBL rate.
    // Drive the resident idle-drain off the poll heartbeat, which ticks even when
    // the guest stops compositing (a static page means no publishes at all).
    // A publish-clocked drain froze there, pinning a burst's ~260
    // stale residents (~516 MiB) for the guest lifetime; the wall clock keeps
    // advancing and returns VRAM to baseline. The presented target is kept alive
    // by identity so it is never reclaimed from under the display. The engine
    // throttles the actual reclaim to IDLE_DRAIN_INTERVAL_MS internally.
    #[cfg(feature = "backend-vulkan")]
    {
        let present = &device.state.present;
        let display_id = present.frame_valid.then(|| {
            runtime::present_identity::surface_identity(
                &device.state,
                present.frame_mapping,
                present.frame_width,
                present.frame_height,
            )
        });
        crate::backend::vulkan::engine::maintain_idle_residents(
            display_id.as_ref(),
            observe::elapsed_ms() as u64,
        );
    }
    // Pre-boundary early-console → host window (headless-safe: the heartbeat
    // drives poll even under -display none). No-op post-boundary or with no
    // window attached.
    #[cfg(feature = "host-window")]
    {
        let now_ns = host.mono_ns();
        window_publish::publish_window_early_frame(&slot, &device.state, &host, now_ns);
    }
    true
}

/// Lock-free VBL pulse for a `device_poll` that could not take `inner`.
///
/// Raises the display VBL — OR the VBL bit into the shared-page pending word,
/// set the read-to-clear display interrupt bit, enqueue the gfx IRQ pulse — all
/// through paths that never touch the device `inner` lock (guest-memory RMW via
/// HostOps, the `Arc<AtomicU32>` interrupt clone, the lock-free `prompt_actions`
/// queue). Uses the VBL state the last lock-acquired poll published. No-op until
/// ONLINE is acked. It shares the same time limiter as the locked path, so a
/// change in lock ownership cannot change the guest's pacing rate.
///
/// The pending-word RMW can race the worker's own present-complete write; the
/// loser drops one bit for one heartbeat (re-raised ~16 ms later). Both writers
/// clear the acked ONLINE bit, so a torn write cannot resurrect it — far better
/// than dropping ~90% of VBLs, which is the pre-fix behaviour under load.
fn vbl_contended_pulse(slot: &BoundDevice) {
    use crate::runtime::host::HostMemory;
    let gpa = slot.vbl_shared_gpa.load(Ordering::Acquire);
    let now = crate::observe::elapsed_ms() as u64;
    if gpa == 0 || !slot.vbl_online.load(Ordering::Acquire) {
        runtime::drain::note_vbl(runtime::drain::VBL_NOT_ONLINE, now);
        return;
    }
    let Some(ops) = slot.ops else {
        return;
    };
    // Both poll paths share one limiter, so both have to report into one census
    // or the delivered rate reads low by whatever share of polls found the
    // device lock contended.
    if !runtime::drain::claim_display_vbl(&slot.vbl_last_us, crate::observe::elapsed_us()) {
        runtime::drain::note_vbl(runtime::drain::VBL_NOT_CLAIMED, now);
        return;
    }
    runtime::drain::note_vbl(runtime::drain::VBL_DELIVERED, now);
    let mut scratch = VecDeque::new();
    let mut host = QemuHost::new(&ops, &mut scratch, &slot.prompt_actions);
    let mut buf = [0u8; 4];
    if host
        .read_gpa(gpa + model::DISPLAY_SHARED_PENDING, &mut buf)
        .is_err()
    {
        return;
    }
    // ONLINE is acked here (vbl_online), so a lingering bit2 is stale — drop it
    // (mirrors signal_display_vbl's post-ack masking) and OR in the VBL bit.
    let pending = u32::from_le_bytes(buf);
    let next = (pending & !model::DISPLAY_ONLINE_EVENT_MASK) | model::DISPLAY_VBL_EVENT_MASK;
    if host
        .write_gpa(gpa + model::DISPLAY_SHARED_PENDING, &next.to_le_bytes())
        .is_err()
    {
        return;
    }
    let idx = slot.vbl_display_index.load(Ordering::Acquire);
    slot.intr_disp
        .fetch_or(1u32 << (idx & 0x1f), Ordering::AcqRel);
    host.enqueue(HostAction::irq_gfx());
}

/// Pop one HostAction for the QEMU BH. Returns false if the queue is empty.
///
/// Prompt actions (IRQ pulses, cursor moves) pop without the device lock so
/// they deliver mid-drain; lock-owning actions (scanout, cursor glyph) keep
/// their after-drain semantics behind `try_lock`.
pub fn device_pop_action(id: u64) -> Option<HostAction> {
    let slot = device_slot(id)?;
    if let Some(a) = slot.prompt_actions.lock().pop_front() {
        return Some(a);
    }
    let mut d = slot.inner.try_lock()?;
    d.actions.pop_front()
}

/// Which source owns the host console right now.
///
/// The three arms are exhaustive and ordered: see [`device_console_feed`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsoleFeed {
    /// BAR1 UEFI GOP, or the guest-programmed `efi_fb_start` when it has one.
    Firmware,
    /// A latched same-geometry early front writeback — the Apple logo and
    /// progress pill, before the compositor's first present.
    Early {
        mapping_id: u32,
        width: u32,
        height: u32,
        generation: u32,
    },
    /// The compositor has presented; product present owns the console.
    Product,
}

impl ConsoleFeed {
    /// The `REIMS_VGPU_CONSOLE_FEED_*` discriminant the QEMU shims switch on.
    /// Must match `include/reims_vgpu_qemu_abi.h`; see
    /// `the_abi_header_agrees_on_the_console_feed_kinds`.
    pub fn kind(&self) -> u32 {
        match self {
            Self::Firmware => 0,
            Self::Early { .. } => 1,
            Self::Product => 2,
        }
    }
}

/// The whole console-ownership decision, from protocol state only.
///
/// **Not** content, sparsity, boot stage, or any screenshot heuristic.
///
/// Both QEMU shims used to reach this answer themselves, by calling
/// `present_boundary_seen` and `early_scanout_target` and branching on the pair.
/// That put the rule in C twice over, beside a third copy here
/// ([`host_console_uses_bar1`], which the host-window pump uses) whose doc
/// admitted as much — it said "shipped C mirrors this". Three copies of one
/// display-ownership rule is three chances for a pathway to disagree about who
/// owns the screen. This is the one copy; the shims paint what it returns.
///
/// The boundary check is deliberately the **monotonic** latch and not
/// `state.present.frame_flush_seen`. The latter is not monotonic — a flush-less
/// (ClearOnly) present clears it — and on the arm64 MMIO console that re-armed
/// the early paint mid-session, flickering stale pre-boundary GOP content (the
/// Apple logo at the old geometry) against live product presents.
///
/// Returns `None` only when `id` names no device.
pub fn device_console_feed(id: u64) -> Option<ConsoleFeed> {
    let slot = device_slot(id)?;
    if slot.present_boundary_seen.load(Ordering::Acquire) {
        return Some(ConsoleFeed::Product);
    }
    // A contended device is reported as `Firmware` rather than as an error: the
    // caller is a display tick, the early console is the pre-boundary source by
    // definition, and the alternative — failing the call — makes the shim invent
    // a policy for "no answer", which is the thing being removed.
    let Some(d) = slot.inner.try_lock() else {
        return Some(ConsoleFeed::Firmware);
    };
    Some(
        match runtime::scanout::early_scanout_target(&d.device.state) {
            // `Early` guarantees a paintable target, so the shims do not re-test
            // the geometry — both used to, which is a decode rule living in C.
            // A zero here is not a target, it is the absence of one.
            Some((mapping_id, width, height, generation))
                if mapping_id != 0 && width != 0 && height != 0 =>
            {
                ConsoleFeed::Early {
                    mapping_id,
                    width,
                    height,
                    generation,
                }
            }
            _ => ConsoleFeed::Firmware,
        },
    )
}

/// Whether a present naming `mapping_id` may paint the host console.
///
/// This is the second half of the console-ownership rule, and it belonged in the
/// shims for exactly as long as it took one of them to disagree. The x86 PCI
/// shim held it — `kind == FIRMWARE || (kind == EARLY && latched != mapping_id)`,
/// assembled from [`device_console_feed`]'s kind and mapping out-params — and the
/// arm64 MMIO shim held nothing at all and painted every present it was handed.
/// One rule, two pathways, one of them missing it: the same shape as the GPA
/// attrs drift, on the question of who owns the screen.
///
/// The three arms, from protocol state only:
///
/// - [`ConsoleFeed::Product`]: the compositor owns the console, so every present
///   paints.
/// - [`ConsoleFeed::Early`]: only the latched front may paint. A clear-only
///   present naming some other mapping must not steal the surface from the
///   firmware console underneath it.
/// - [`ConsoleFeed::Firmware`]: the guest is still on BAR1 / `efi_fb`; nothing
///   presented here paints.
///
/// Returns `None` only when `id` names no device.
pub fn device_scanout_may_paint(id: u64, mapping_id: u32) -> Option<bool> {
    Some(match device_console_feed(id)? {
        ConsoleFeed::Product => true,
        ConsoleFeed::Early {
            mapping_id: latched,
            ..
        } => latched == mapping_id,
        ConsoleFeed::Firmware => false,
    })
}

/// [`ConsoleFeed::Firmware`], for a caller that already holds device state.
///
/// From **protocol state only** — not content, sparsity, boot stage, or any
/// screenshot heuristic:
///
/// - `frame_flush_seen` (DisplaySwap / x86 present): product owns the console.
/// - Else if `early_front_latched` (same-geom type-11 front writeback): the
///   product early paint, logo + pill, before the swap.
/// - Else: BAR1 / guest-programmed `efi_fb_start` (UEFI + PE log console).
///
/// This doc used to say "shipped C mirrors this", and it did — that mirroring is
/// what [`device_console_feed`] removed, so the QEMU shims now ask rather than
/// reconstruct. What is left here is not a second copy of the rule for them; it
/// is the form the host-window early pump needs, which runs inside the drain
/// with `&DeviceState` in hand and so cannot take `device_console_feed`'s
/// by-id, lock-free path.
///
/// The two are deliberately **not** identical, and the difference is the
/// `frame_flush_seen` argument: `device_console_feed` reads the monotonic
/// boundary latch instead, because a flush-less (ClearOnly) present clears
/// `frame_flush_seen` and a console that re-armed on that flickered stale
/// pre-boundary content. This predicate takes whatever its caller passes, and
/// its caller passes the non-monotonic flag.
#[inline]
pub fn host_console_uses_bar1(frame_flush_seen: bool, early_front_latched: bool) -> bool {
    !frame_flush_seen && !early_front_latched
}

/// Copy guest EFI console FB (programmed at 0x1210) into a host BGRA8 surface.
///
/// `false` if no efi_fb_start or GPA read fails — C uses BAR1 then.
pub fn device_efi_console_copy(
    id: u64,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
) -> bool {
    let Some(slot) = device_slot(id) else {
        return false;
    };
    let Some(mut d) = slot.inner.try_lock() else {
        return false;
    };
    if d.device.state.gfx.efi_fb_start == 0 {
        return false;
    }
    let Some(ops) = slot.ops else {
        return false;
    };
    let DeviceInner { device, actions } = &mut *d;
    let host = QemuHost::new(&ops, actions, &slot.prompt_actions);
    runtime::scanout::paint_efi_console(&device.state, &host, dst, dst_stride, width, height)
}

/// Fill a host BGRA8 framebuffer from the named guest mapping (or EFI FB).
///
/// `dst` is a row-major buffer with `dst_stride` bytes per row. QEMU passes
/// `surface_data` / `surface_stride` from the DisplaySurface.
/// `generation` is the HostAction stamp (0 = always paint).
pub fn device_scanout_copy(
    id: u64,
    mapping_id: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
    generation: u32,
) -> runtime::scanout::ScanoutCopyResult {
    use runtime::scanout::ScanoutCopyResult;
    let Some(slot) = device_slot(id) else {
        return ScanoutCopyResult::Failed;
    };
    let present_action = slot.present_action_pending.load(Ordering::Acquire);
    let mut d = if present_action {
        // The worker observes `present_action_pending` before taking this lock
        // and returns immediately. Blocking here is therefore bounded to that
        // short handoff; pre-boundary periodic scanout remains nonblocking.
        slot.inner.lock()
    } else {
        let Some(d) = slot.inner.try_lock() else {
            return ScanoutCopyResult::Unchanged;
        };
        d
    };
    if let Some(ops) = slot.ops {
        let DeviceInner { device, actions } = &mut *d;
        let mut host = QemuHost::new(&ops, actions, &slot.prompt_actions);
        // `frame_encode_pending` also means a valid +0x188 snapshot was just
        // installed and must be blitted once. `copy_to_bgra8` distinguishes
        // that case from capture-fail retry without draining guest commands in
        // this QEMU display context; do not discard the only scanout action.
        let rc = runtime::scanout::copy_to_bgra8(
            &mut device.state,
            &mut host,
            mapping_id,
            dst,
            dst_stride,
            width,
            height,
            generation,
        );
        note_scanout_copy_consumed(&mut device.state, &mut host, &slot, rc, present_action);
        rc
    } else {
        // No QEMU ops table is bound (unit tests / headless create), so every
        // guest read fails and the copy falls back to a black clear.
        let mut host = NullHost;
        let rc = runtime::scanout::copy_to_bgra8(
            &mut d.device.state,
            &mut host,
            mapping_id,
            dst,
            dst_stride,
            width,
            height,
            generation,
        );
        note_scanout_copy_consumed(&mut d.device.state, &mut host, &slot, rc, present_action);
        rc
    }
}

/// Entry-side `waitForPendingFrames`: what one `copy_to_bgra8` owes the device
/// once it returns.
///
/// Two conditions, one rule. A copy that painted — or that found nothing to
/// redo — frees the DisplaySwap packet held at channel head. A copy that
/// *failed* owes exactly the same whenever `present_action` was set, because
/// the host consumed the action either way and C cannot replay it; leaving it
/// outstanding strands every later channel behind it.
///
/// The stamp for accepted presents fires at retain, not here.
///
/// This lives apart from [`device_scanout_copy`] because that function picks
/// between two hosts and the rule is the same under both. It was written out
/// once per arm before, and the two copies had already drifted: the headless
/// arm cleared the pending flag after a failed copy without recording the
/// consumption, which is the stranding the paragraph above forbids.
fn note_scanout_copy_consumed<H: HostOps>(
    state: &mut crate::model::DeviceState,
    host: &mut H,
    slot: &BoundDevice,
    rc: runtime::scanout::ScanoutCopyResult,
    present_action: bool,
) {
    use runtime::scanout::ScanoutCopyResult;
    let painted = matches!(
        rc,
        ScanoutCopyResult::Painted | ScanoutCopyResult::Unchanged
    );
    if !painted && !present_action {
        return;
    }
    runtime::drain::note_present_paint_consumed(state);
    slot.present_action_pending.store(false, Ordering::Release);
    host.schedule_bh();
}

/// Cursor glyph metadata for the QEMU console.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct CursorGlyphInfo {
    pub width: u32,
    pub height: u32,
    pub hot_x: u32,
    pub hot_y: u32,
    pub pixel_count: u32,
}

pub fn device_cursor_glyph_info(id: u64) -> Option<CursorGlyphInfo> {
    let slot = device_slot(id)?;
    let d = slot.inner.try_lock()?;
    let c = &d.device.state.cursor;
    if !c.glyph_ready || c.pixels.is_empty() {
        return None;
    }
    Some(CursorGlyphInfo {
        width: c.width as u32,
        height: c.height as u32,
        hot_x: c.hot_x as u32,
        hot_y: c.hot_y as u32,
        pixel_count: c.pixels.len() as u32,
    })
}

/// Copy QEMUCursor ARGB pixels. Returns number of pixels written.
pub fn device_cursor_glyph_copy(id: u64, out: &mut [u32]) -> Option<usize> {
    let slot = device_slot(id)?;
    let d = slot.inner.try_lock()?;
    let c = &d.device.state.cursor;
    if !c.glyph_ready || c.pixels.is_empty() {
        return None;
    }
    let n = c.pixels.len().min(out.len());
    out[..n].copy_from_slice(&c.pixels[..n]);
    Some(n)
}

pub fn backend_name() -> &'static str {
    #[cfg(feature = "backend-metal")]
    {
        "metal"
    }
    #[cfg(feature = "backend-vulkan")]
    {
        "vulkan"
    }
}

pub fn unwind_safe<T, F>(f: F, on_panic: T) -> T
where
    F: FnOnce() -> T + std::panic::UnwindSafe,
{
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(v) => v,
        Err(_) => on_panic,
    }
}

#[cfg(test)]
mod tests;
