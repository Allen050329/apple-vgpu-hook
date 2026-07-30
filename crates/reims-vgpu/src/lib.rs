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

use crate::qemu::host_ops::{NullHost, QemuHost, ReimsVgpuHostAction, ReimsVgpuHostOps};

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
}

/// Link to a running host-owned presentation window ([[host-window]]).
///
/// Held on the device so the drain can publish finished frames into `frames`
/// (latest-wins) and skip re-publishing an unchanged present via `last`. The
/// input back-channel is the window thread's `InputSink`, not stored here — it
/// pushes onto `prompt_actions` directly.
#[cfg(feature = "host-window")]
type WindowFrameKey = (u32, u32, u64);

#[cfg(feature = "host-window")]
fn window_frame_key(present: &crate::model::PresentState) -> WindowFrameKey {
    #[cfg(target_os = "macos")]
    let present_epoch = present.present_epoch;
    #[cfg(not(target_os = "macos"))]
    let present_epoch = 0;
    (
        present.frame_mapping,
        present.frame_generation,
        present_epoch,
    )
}

#[cfg(feature = "host-window")]
struct WindowLink {
    /// Shared latest-frame slot the window thread reads each redraw.
    frames: host_window::present::FrameSlot,
    /// `(mapping_id, generation, present_epoch)` of the last frame published.
    ///
    /// The resource generation alone is insufficient: the guest can update a
    /// resident in place and present it again without changing that identity.
    /// On macOS, `present_epoch` advances once per accepted capture, so those
    /// frames publish while drain passes with no new DisplaySwap remain
    /// deduplicated. It remains zero on Linux, preserving the verified
    /// `(mapping_id, generation)` publication contract there.
    last: WindowFrameKey,
    /// Monotonic frame sequence stamped onto each published [`Frame`] so the
    /// window uploads only new frames (skips the per-vblank re-upload of
    /// unchanged content). Bumped on every write.
    seq: u64,
    /// Dedup latch for the `frame_bgra_short` drop log: the `(w,h)` last logged
    /// as short, so a persistent mismatch logs once per geometry instead of
    /// every present. Cleared when a well-formed frame publishes.
    ///
    /// Both platforms: the CPU-fallback publish arm is shared since the two
    /// publish paths were unified, and a present with no resident behind it
    /// (firmware framebuffer, a cleared-but-never-rendered mapping, the frames
    /// after a device reset) is normal on macOS too.
    bgra_short_geom: Option<(u32, u32)>,
    /// Set to ask the window thread to exit (VM teardown); the thread polls it.
    stop: host_window::present::StopFlag,
    /// Window thread handle. `device_window_stop` sets `stop` and joins it, so
    /// the window's Vulkan objects tear down before QEMU teardown proceeds
    /// (avoids the driver-unload-during-exit crash class).
    thread: Option<std::thread::JoinHandle<Result<(), host_window::present::WindowError>>>,
    /// Published after the process-main AppKit loop has destroyed the native
    /// window and its Vulkan objects.
    #[cfg(target_os = "macos")]
    exited: host_window::present::ExitedFlag,
}

/// Registered early-boot framebuffer (BAR1 GOP host RAM) the C shim hands the
/// device so the window can show UEFI/OpenCore/boot.efi output before the
/// product present path latches. `ptr` is a stable RAMBlock host pointer valid
/// for the device lifetime; the guest writes it live (a torn read only flickers
/// one early frame).
#[cfg(feature = "host-window")]
#[derive(Clone, Copy)]
struct EarlyFb {
    ptr: usize,
    stride: u32,
    width: u32,
    height: u32,
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
    vbl_last_ms: AtomicU64,
    /// QEMU HostOps (GPA / clock / schedule worker). None in pure unit tests.
    ops: Option<ReimsVgpuHostOps>,
    /// Host-owned presentation window ([[host-window]]), once
    /// `device_window_start` has spawned it. `None` on a normal QEMU-display
    /// boot (the window is opt-in behind `REIMS_VGPU_WINDOW`).
    #[cfg(feature = "host-window")]
    window: Mutex<Option<WindowLink>>,
    /// Early-boot framebuffer (BAR1 GOP) registered by the C shim, shown in the
    /// window until the product present path latches.
    #[cfg(feature = "host-window")]
    early_fb: Mutex<Option<EarlyFb>>,
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
    if let Some(ops) = slot.ops {
        let mut host = QemuHost::with_prompt(&ops, &mut inner.actions, &slot.prompt_actions);
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
            intr_fault,
            fifo_read_live,
            present_action_pending: AtomicBool::new(false),
            present_boundary_seen: AtomicBool::new(false),
            reset_count: AtomicU64::new(0),
            vbl_shared_gpa: AtomicU64::new(0),
            vbl_display_index: AtomicU32::new(0),
            vbl_online: AtomicBool::new(false),
            vbl_last_ms: AtomicU64::new(0),
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
            let mut host = QemuHost::with_prompt(&ops, actions, &slot.prompt_actions);
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

/// Start the host-owned presentation window for `id` ([[host-window]]).
///
/// Wires the window's input `InputSink` to the device prompt-action rail (push
/// and `notify_actions`, both thread-safe) via a `Weak` device ref so the window
/// never keeps a destroyed device alive. Frames reach the window through
/// [`publish_window_frame`], called by the drain. Idempotent; `true` on success.
#[cfg(feature = "host-window")]
pub fn device_window_start(id: u64, width: u32, height: u32) -> bool {
    use host_window::present::{FrameSlot, InputSink, WindowConfig};
    let Some(slot) = device_slot(id) else {
        return false;
    };
    let mut link = slot.window.lock();
    if link.is_some() {
        return true; // already running (idempotent)
    }
    // FrameSlot is a std::sync::Mutex (owned by the window module); lib.rs's
    // bare `Mutex` is parking_lot, so qualify it here.
    let frames: FrameSlot = Arc::new(std::sync::Mutex::new(None));
    // Weak so a live window does not pin a destroyed device; post-destroy input
    // upgrades to None and is dropped (the guest is gone anyway).
    let weak = Arc::downgrade(&slot);
    let on_input: InputSink = Arc::new(move |action: HostAction| {
        let Some(dev) = weak.upgrade() else {
            return;
        };
        dev.prompt_actions.lock().push_back(action);
        // Wake the HostAction-delivery BH so the guest sees the input without
        // waiting for the next drain tranche (same rail as IRQ/cursor).
        if let Some(ops) = dev.ops {
            if let Some(notify) = ops.notify_actions {
                // SAFETY: QEMU owns ctx for the device lifetime; notify_actions
                // is the thread-safe BH-schedule callback.
                unsafe { notify(ops.ctx) }
            }
        }
    });
    let cfg = WindowConfig {
        title: "Reims vGPU".to_string(),
        width: if width == 0 {
            model::EFI_BOOT_WIDTH
        } else {
            width
        },
        height: if height == 0 {
            model::EFI_BOOT_HEIGHT
        } else {
            height
        },
    };
    let stop: host_window::present::StopFlag = Arc::new(std::sync::atomic::AtomicBool::new(false));
    #[cfg(target_os = "macos")]
    let (thread, exited) = {
        let exited: host_window::present::ExitedFlag =
            Arc::new(std::sync::atomic::AtomicBool::new(false));
        if let Err(error) = host_window::present::start_main_thread(
            id,
            cfg,
            on_input,
            Arc::clone(&frames),
            Arc::clone(&stop),
            Arc::clone(&exited),
        ) {
            observe::Emit::decline("host_window_start", &error)
                .field("id", id)
                .fail();
            return false;
        }
        (None, exited)
    };
    #[cfg(not(target_os = "macos"))]
    let thread = Some(host_window::present::spawn(
        cfg,
        on_input,
        Arc::clone(&frames),
        Arc::clone(&stop),
    ));
    *link = Some(WindowLink {
        frames,
        last: (u32::MAX, u32::MAX, u64::MAX),
        seq: 0,
        bgra_short_geom: None,
        stop,
        thread,
        #[cfg(target_os = "macos")]
        exited,
    });
    observe::off(format!(
        "host_window_start id={id} {}x{}",
        if width == 0 {
            model::EFI_BOOT_WIDTH
        } else {
            width
        },
        if height == 0 {
            model::EFI_BOOT_HEIGHT
        } else {
            height
        }
    ));
    true
}

/// No-op stub when the `host-window` feature is off: the FFI symbol still links
/// (so the C shim binds regardless) but there is no window to start.
#[cfg(not(feature = "host-window"))]
pub fn device_window_start(_id: u64, _width: u32, _height: u32) -> bool {
    false
}

/// Run the main-thread-owned macOS window. QEMU calls this from its process-main
/// UI entry after device realize; it blocks until the window exits.
#[cfg(all(feature = "host-window", target_os = "macos"))]
pub fn device_window_run_main(id: u64) -> bool {
    match host_window::present::run_main_thread(id) {
        Ok(()) => true,
        Err(error) => {
            observe::Emit::decline("host_window_main", &error)
                .field("id", id)
                .fail();
            false
        }
    }
}

#[cfg(not(all(feature = "host-window", target_os = "macos")))]
pub fn device_window_run_main(_id: u64) -> bool {
    false
}

/// Publish the current finished present frame into the window's frame slot, if a
/// window is running and this present has not been published yet. Runs on the
/// drain worker under no device lock of its own (its own small mutex), so it
/// never contends the render tranche. Latest-wins.
#[cfg(feature = "host-window")]
fn publish_window_frame(slot: &BoundDevice, state: &mut crate::model::DeviceState) {
    let mut guard = slot.window.lock();
    let Some(link) = guard.as_mut() else {
        // No window consumes the capture: revert the next capture to the full
        // readback path (a torn-down window must not leave `frame_bgra` stale
        // behind an unreset `display_from_resident`).
        state.present.display_from_resident = false;
        return;
    };
    let p = &state.present;
    if !p.frame_valid || p.frame_width == 0 || p.frame_height == 0 {
        return;
    }
    let key = window_frame_key(p);
    if key == link.last {
        return;
    }
    // Copied out rather than held behind `p`: the branches below assign
    // `state.present.display_from_resident`, and the frame bytes are the only
    // thing that still has to be read through the borrow.
    let (mapping, width, height, generation) = (
        p.frame_mapping,
        p.frame_width,
        p.frame_height,
        p.frame_generation,
    );
    let need = (width as usize)
        .saturating_mul(height as usize)
        .saturating_mul(4);
    if need == 0 {
        return;
    }
    let present_identity =
        crate::runtime::present_identity::surface_identity(state, mapping, width, height);
    // Keep the resident this present names alive across the idle sweep below,
    // then reclaim targets idle past the wall-clock age threshold so VRAM returns
    // to the working-set baseline after a compositing burst instead of sitting at
    // the high REGISTRY_CAP for the guest lifetime.
    let now_ms = crate::observe::elapsed_ms() as u64;
    crate::backend::vulkan::engine::touch_resident_target(Some(&present_identity), now_ms);
    crate::backend::vulkan::engine::maintain_idle_residents(Some(&present_identity), now_ms);
    // The window presenting from the engine's own device can take the resident
    // as it stands, so the frame never crosses host memory. `display_from_resident`
    // is what tells the NEXT capture not to read it back, and it is only set
    // when a resident actually carried this one.
    if crate::backend::vulkan::engine::window_present_attached()
        && crate::backend::vulkan::engine::resident_presentable(&present_identity, width, height)
    {
        let resident_source = crate::backend::vulkan::engine::WindowPresentSource {
            width,
            height,
            candidates: vec![present_identity],
        };
        let published = window_write_frame(link, width, height, Vec::new(), Some(resident_source));
        crate::runtime::census::present_proxy::window_publish::note(published);
        if published {
            link.last = key;
            state.present.display_from_resident = true;
        }
        return;
    }
    // No resident carries this present (firmware framebuffer, a mapping the
    // compositor cleared but never rendered into, the frames after a device
    // reset), or the window is driving its own device because the engine's
    // cannot present to this surface. Either way the window needs CPU pixels,
    // and the next capture must read them back.
    state.present.display_from_resident = false;
    if state.present.frame_bgra.len() < need {
        // No usable CPU frame: nothing to publish. Reachable via keep-prior
        // when a capture FAILS at a new/larger geometry (dims advanced, the
        // buffer kept the smaller prior), and on the present right after a
        // resident-carried one, whose capture deliberately left the buffer
        // empty. Skipping is correct (never publish a short/torn frame; the
        // window holds its last good frame), but silence would hide "the window
        // froze because captures keep failing at this geometry". Fail-visible +
        // deduped per geometry so a persistent mismatch logs once, not every
        // present (no flood).
        if link.bgra_short_geom != Some((width, height)) {
            link.bgra_short_geom = Some((width, height));
            observe::off(format!(
                "publish_window_frame DROP reason=frame_bgra_short mid={} {}x{} \
                 have={} need={need} gen={}",
                mapping,
                width,
                height,
                state.present.frame_bgra.len(),
                generation
            ));
        }
        crate::runtime::census::present_proxy::window_publish::note(false);
        return;
    }
    // A well-formed frame cleared the short-buffer condition; re-arm the latch
    // so a later mismatch at the same geometry logs again.
    link.bgra_short_geom = None;
    let bgra = state.present.frame_bgra[..need].to_vec();
    let published = window_write_frame(link, width, height, bgra, None);
    crate::runtime::census::present_proxy::window_publish::note(published);
    if published {
        link.last = key;
    }
}

/// Write a frame into the window's slot, stamping the next monotonic `seq` so
/// the window prepares only new content. Returns false if the slot lock is
/// poisoned (a panicked window thread — the window is gone, drop the publish).
/// Bound so the inner guard drops before the caller's outer `window` guard.
#[cfg(feature = "host-window")]
fn window_write_frame(
    link: &mut WindowLink,
    width: u32,
    height: u32,
    bgra: Vec<u8>,
    resident: Option<crate::backend::vulkan::engine::WindowPresentSource>,
) -> bool {
    link.seq = link.seq.wrapping_add(1);
    let frame = std::sync::Arc::new(host_window::present::Frame {
        seq: link.seq,
        width,
        height,
        bgra,
        resident,
    });
    match link.frames.lock() {
        Ok(mut slot_frame) => {
            *slot_frame = Some(frame);
            true
        }
        Err(_) => false,
    }
}

/// Stop the host-owned window during VM teardown. Sets the stop flag so the
/// event loop exits, then waits for its Vulkan objects to tear down before QEMU
/// proceeds to process/driver teardown. Linux joins the dedicated window thread;
/// macOS waits for the process-main loop's exit publication. Idempotent; no-op
/// without a window.
#[cfg(feature = "host-window")]
pub fn device_window_stop(id: u64) -> bool {
    let Some(slot) = device_slot(id) else {
        return false;
    };
    let link = slot.window.lock().take();
    let Some(mut link) = link else {
        return true; // no window (or already stopped)
    };
    link.stop.store(true, std::sync::atomic::Ordering::Relaxed);
    #[cfg(target_os = "macos")]
    {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while !link.exited.load(Ordering::Acquire) && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        if !link.exited.load(Ordering::Acquire) {
            observe::fail(format!(
                "host_window_stop FAIL reason=main_thread_teardown_timeout id={id}"
            ));
            return false;
        }
    }
    if let Some(thread) = link.thread.take() {
        // The window thread's `WindowError` return was discarded here, so a
        // `build_event_loop`/`run_app` failure on the Linux spawn path vanished
        // with no line. Emit the typed decline instead. (macOS never takes this
        // branch — its window runs on the process main thread, so `thread` is
        // None; the join runs only on the Linux `spawn` path.)
        match thread.join() {
            Ok(Ok(())) => {}
            Ok(Err(error)) => {
                observe::Emit::decline("host_window_run", &error)
                    .field("id", id)
                    .fail();
            }
            // A panic in the window thread; the default panic hook already wrote
            // its message to stderr, and there is no guest command to decline.
            Err(_) => {}
        }
    }
    true
}

#[cfg(not(feature = "host-window"))]
pub fn device_window_stop(_id: u64) -> bool {
    false
}

/// Register the early-boot framebuffer (BAR1 GOP host RAM) so the window can
/// show UEFI/OpenCore/boot.efi output before the product present path latches.
/// `ptr` is a stable RAMBlock host pointer valid for the device lifetime.
///
/// SAFETY: the caller guarantees `ptr` addresses at least `stride * height`
/// readable bytes for the device lifetime (the QEMU BAR1 RAMBlock).
#[cfg(feature = "host-window")]
pub fn device_window_set_early_fb(
    id: u64,
    ptr: usize,
    stride: u32,
    width: u32,
    height: u32,
) -> bool {
    let Some(slot) = device_slot(id) else {
        return false;
    };
    if ptr == 0 || stride == 0 || width == 0 || height == 0 {
        return false;
    }
    *slot.early_fb.lock() = Some(EarlyFb {
        ptr,
        stride,
        width,
        height,
    });
    true
}

#[cfg(not(feature = "host-window"))]
pub fn device_window_set_early_fb(
    _id: u64,
    _ptr: usize,
    _stride: u32,
    _width: u32,
    _height: u32,
) -> bool {
    false
}

/// Pre-boundary early-console pump: while the guest is still on the BAR1/EFI
/// console (no product present latched), push that framebuffer to the window so
/// early boot is visible. Runs on the poll (heartbeat) path so it works headless
/// (`-display none` never ticks `gfx_update`). Gated by `host_console_uses_bar1`
/// — the same protocol-state ownership rule the C `gfx_update` uses, so the
/// window never fights the product present for the frame — and throttled to
/// ~30 fps so it does not memcpy the FB every 4 ms.
#[cfg(feature = "host-window")]
fn publish_window_early_frame<M: crate::runtime::host::HostMemory>(
    slot: &BoundDevice,
    state: &crate::model::DeviceState,
    host: &M,
    now_ns: u64,
) {
    let mut guard = slot.window.lock();
    let Some(link) = guard.as_mut() else {
        return;
    };
    // Console-ownership gate (mirror of host_console_uses_bar1): only feed the
    // window while it is on the early console, never after the product present
    // owns it or a same-geom early front is latched (the drain publishes those).
    let early_latched = runtime::scanout::early_scanout_target(state).is_some();
    if !host_console_uses_bar1(state.present.frame_flush_seen, early_latched) {
        return;
    }
    // ~30 fps throttle (33 ms) on the 4 ms poll.
    let last = slot.early_last_ns.load(Ordering::Relaxed);
    if now_ns.saturating_sub(last) < 33_000_000 {
        return;
    }
    let w = model::EFI_BOOT_WIDTH;
    let h = model::EFI_BOOT_HEIGHT;
    let stride = w.saturating_mul(4);
    let mut buf = vec![0u8; (stride as usize).saturating_mul(h as usize)];
    // Prefer the guest-programmed EFI FB (kernel-relocated console), else the
    // BAR1 GOP framebuffer the option ROM drives — the same order as C's
    // reims_vgpu_pci_copy_early_console.
    let painted = if state.gfx.efi_fb_start != 0 {
        runtime::scanout::paint_efi_console(state, host, &mut buf, stride, w, h)
    } else {
        false
    };
    let painted = painted || copy_early_bar1(slot, &mut buf, stride, w, h);
    if !painted {
        return;
    }
    slot.early_last_ns.store(now_ns, Ordering::Relaxed);
    // Early boot frames come from the BAR1 GOP framebuffer, not a resident
    // target, so there is no resident source to hand over.
    window_write_frame(link, w, h, buf, None);
}

/// Copy the registered BAR1 early framebuffer into `dst` (tight BGRA8). Returns
/// false when no early FB is registered or its geometry cannot cover the request.
#[cfg(feature = "host-window")]
fn copy_early_bar1(slot: &BoundDevice, dst: &mut [u8], dst_stride: u32, w: u32, h: u32) -> bool {
    let efb = *slot.early_fb.lock();
    let Some(efb) = efb else {
        return false;
    };
    if efb.ptr == 0 || efb.width < w || efb.height < h {
        return false;
    }
    let src_len = (efb.stride as usize).saturating_mul(efb.height as usize);
    // SAFETY: efb.ptr is the BAR1 RAMBlock host pointer registered by the C shim
    // at realize, valid for the device lifetime and at least stride*height bytes
    // (device_window_set_early_fb contract). The guest may write concurrently; a
    // torn read only flickers one early-boot frame.
    let src = unsafe { std::slice::from_raw_parts(efb.ptr as *const u8, src_len) };
    let row = (w as usize).saturating_mul(4);
    for y in 0..h as usize {
        let so = y.saturating_mul(efb.stride as usize);
        let doff = y.saturating_mul(dst_stride as usize);
        if so + row > src.len() || doff + row > dst.len() {
            return false;
        }
        dst[doff..doff + row].copy_from_slice(&src[so..so + row]);
    }
    true
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
    }
    let write = QueuedGfxWrite { offset, data, size };
    let mut ingress = slot.gfx_ingress.lock();
    if ingress.is_empty() {
        if let Some(mut inner) = slot.inner.try_lock() {
            apply_gfx_write(&mut inner, &slot, write);
            return true;
        }
    }
    ingress.push_back(write);
    drop(ingress);
    schedule_device(&slot);
    true
}

pub fn device_iosfc_read(id: u64, offset: u64, size: u32) -> Option<u64> {
    let slot = device_slot(id)?;
    let d = slot.inner.lock();
    Some(d.device.iosfc_read(offset, size))
}

pub fn device_iosfc_write(id: u64, offset: u64, data: u64, size: u32) -> bool {
    let Some(slot) = device_slot(id) else {
        return false;
    };
    let mut d = slot.inner.lock();
    if let Some(ops) = slot.ops {
        let DeviceInner { device, actions } = &mut *d;
        let mut host = QemuHost::with_prompt(&ops, actions, &slot.prompt_actions);
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
    let mut host = QemuHost::with_prompt(&ops, actions, &slot.prompt_actions);
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
    publish_window_frame(&slot, &mut device.state);
    runtime::drain::note_drain_tranche(drain_us, publish_started.elapsed().as_micros() as u64);
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
    let mut host = QemuHost::with_prompt(&ops, actions, &slot.prompt_actions);
    runtime::drain::publish_stranded_fifos(&mut device.state, &mut host);
    runtime::drain::try_display_online(&mut device.state, &mut host);
    // After ONLINE, pulse VBL so the guest compositor has a display time base
    //. Missing VBL → clear-only dual-mid present thrash.
    runtime::drain::signal_display_vbl(&mut device.state, &mut host, &slot.vbl_last_ms);
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
    // the guest stops compositing (a static page → `present_import used_hz=0` →
    // no publishes). A publish-clocked drain froze there, pinning a burst's ~260
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
        publish_window_early_frame(&slot, &device.state, &host, now_ns);
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
    if !runtime::drain::claim_display_vbl(&slot.vbl_last_ms, now) {
        runtime::drain::note_vbl(runtime::drain::VBL_NOT_CLAIMED, now);
        return;
    }
    runtime::drain::note_vbl(runtime::drain::VBL_DELIVERED, now);
    let mut scratch = VecDeque::new();
    let mut host = QemuHost::with_prompt(&ops, &mut scratch, &slot.prompt_actions);
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
pub fn device_pop_action(id: u64) -> Option<ReimsVgpuHostAction> {
    let slot = device_slot(id)?;
    if let Some(a) = slot.prompt_actions.lock().pop_front() {
        return Some(ReimsVgpuHostAction::from(a));
    }
    let mut d = slot.inner.try_lock()?;
    d.actions.pop_front().map(ReimsVgpuHostAction::from)
}

/// Early-boot console re-pull target: `(mapping_id, width, height, generation)`.
///
/// `None` after the first present boundary (`frame_flush_seen`) or when no
/// compositor front mapping has been written yet.
pub fn device_early_scanout_target(id: u64) -> Option<(u32, u32, u32, u32)> {
    let slot = device_slot(id)?;
    // Monotonic boundary gate (same rule the x86 PCI shim applies to the BAR1
    // GOP overlay): once the compositor has presented, the early console feed
    // never returns. `early_scanout_target`'s own `frame_flush_seen` check is
    // NOT monotonic — a flush-less (ClearOnly) present clears it, and on the
    // arm64 MMIO console that re-armed this early paint mid-session, flickering
    // stale pre-boundary GOP content (Apple logo at the old geometry) against
    // live product presents.
    if slot.present_boundary_seen.load(Ordering::Acquire) {
        return None;
    }
    let d = slot.inner.try_lock()?;
    runtime::scanout::early_scanout_target(&d.device.state)
}

/// True after the first product present boundary (`frame_flush_seen` / DisplaySwap).
///
/// QEMU uses this to stop overlaying BAR1 UEFI GOP onto the host console — only
/// after the compositor owns scanout, not on the first early logo writeback.
/// The published value is monotonic and lock-free so worker contention cannot
/// masquerade as a return to the early-console feed.
pub fn device_present_boundary_seen(id: u64) -> Option<bool> {
    let slot = device_slot(id)?;
    Some(slot.present_boundary_seen.load(Ordering::Acquire))
}

/// Host-console feed decision (shipped C mirrors this).
///
/// Host-console feed selection from **protocol state only** (not content).
///
/// - `frame_flush_seen` (DisplaySwap / x86 present): product owns console.
/// - Else if `early_front_latched` (same-geom type-11 front writeback latched
///   for `early_scanout_target`): product early paint (logo+pill pre-swap).
/// - Else: BAR1 / guest-programmed `efi_fb_start` (UEFI + PE log console).
///
/// **Not** content, sparsity, boot-stage, or screenshot heuristics.
/// See early-boot + `reims-vgpu-pci` gfx_update.
#[inline]
pub fn host_console_uses_bar1(frame_flush_seen: bool, early_front_latched: bool) -> bool {
    !frame_flush_seen && !early_front_latched
}

/// Copy guest EFI console FB (programmed at 0x1210) into a host BGRA8 surface.
///
/// Returns `None` if no efi_fb_start or GPA read fails — C uses BAR1 then.
pub fn device_efi_console_copy(
    id: u64,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
) -> Option<(u64, u32)> {
    let slot = device_slot(id)?;
    let mut d = slot.inner.try_lock()?;
    let gpa = d.device.state.gfx.efi_fb_start;
    if gpa == 0 {
        return None;
    }
    if let Some(ops) = slot.ops {
        let DeviceInner { device, actions } = &mut *d;
        let host = QemuHost::with_prompt(&ops, actions, &slot.prompt_actions);
        if runtime::scanout::paint_efi_console(&device.state, &host, dst, dst_stride, width, height)
        {
            let stride = if device.state.gfx.efi_fb_stride != 0 {
                device.state.gfx.efi_fb_stride
            } else {
                crate::model::EFI_BOOT_WIDTH.saturating_mul(4)
            };
            return Some((gpa, stride));
        }
    }
    None
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
        let mut host = QemuHost::with_prompt(&ops, actions, &slot.prompt_actions);
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
        // Entry-side waitForPendingFrames: paint frees held DisplaySwap packets
        // at channel head (unpainted_presents cleared). Stamp of accepted
        // presents already fired at retain — do not complete stamps here.
        if matches!(
            rc,
            ScanoutCopyResult::Painted | ScanoutCopyResult::Unchanged
        ) {
            runtime::drain::note_present_paint_consumed(&mut device.state);
            slot.present_action_pending.store(false, Ordering::Release);
            host.schedule_bh();
        } else if present_action {
            // The host consumed this action even though the copy failed. Do
            // not strand all later channels behind an action C cannot replay.
            runtime::drain::note_present_paint_consumed(&mut device.state);
            slot.present_action_pending.store(false, Ordering::Release);
            host.schedule_bh();
        }
        rc
    } else {
        // Unit tests / no host: GPA+KVA fail → black clear.
        struct EmptyMem;
        impl runtime::host::HostMemory for EmptyMem {
            fn read_gpa(&self, _gpa: u64, _buf: &mut [u8]) -> Result<(), runtime::host::MemError> {
                Err(runtime::host::MemError::Unmapped)
            }
            fn write_gpa(&mut self, _gpa: u64, _buf: &[u8]) -> Result<(), runtime::host::MemError> {
                Err(runtime::host::MemError::Unmapped)
            }
        }
        impl HostOps for EmptyMem {
            fn mono_ns(&self) -> u64 {
                0
            }
            fn enqueue(&mut self, _action: HostAction) {}
            fn schedule_bh(&mut self) {}
        }
        let mut empty = EmptyMem;
        let rc = runtime::scanout::copy_to_bgra8(
            &mut d.device.state,
            &mut empty,
            mapping_id,
            dst,
            dst_stride,
            width,
            height,
            generation,
        );
        if matches!(
            rc,
            ScanoutCopyResult::Painted | ScanoutCopyResult::Unchanged
        ) {
            runtime::drain::note_present_paint_consumed(&mut d.device.state);
        }
        if present_action {
            slot.present_action_pending.store(false, Ordering::Release);
        }
        rc
    }
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
mod tests {

    use super::*;
    use crate::model::PAGE_SHIFT_ARM64E;

    fn null_host_ops() -> ReimsVgpuHostOps {
        ReimsVgpuHostOps {
            abi_version: crate::qemu::abi::REIMS_VGPU_QEMU_ABI_VERSION,
            struct_size: std::mem::size_of::<ReimsVgpuHostOps>() as u32,
            ctx: std::ptr::null_mut(),
            read_gpa: None,
            write_gpa: None,
            mono_ns: None,
            schedule_bh: None,
            read_kva: None,
            read_xreg: None,
            map_pages: None,
            unmap_pages: None,
            map_pages_stable: 0,
            is_ram_gpa: None,
            notify_actions: None,
        }
    }

    #[test]
    fn lifecycle() {
        let id = device_create(None, PAGE_SHIFT_ARM64E).expect("create");
        assert_ne!(id, 0);
        assert!(device_reset(id));
        assert!(device_destroy(id));
        assert!(!device_destroy(id));
    }

    #[test]
    fn exactly_one_backend_name() {
        let n = backend_name();
        assert!(n == "metal" || n == "vulkan");
    }

    #[test]
    fn panic_does_not_escape() {
        let v = unwind_safe(|| panic!("boom"), 42i32);
        assert_eq!(v, 42);
    }

    #[test]
    fn mmio_hooks() {
        let id = device_create(None, PAGE_SHIFT_ARM64E).expect("create");
        assert!(device_gfx_write(id, 0x1034, 0x3e, 4));
        assert_eq!(device_gfx_read(id, 0x1034, 4), Some(0x3e));
        assert!(device_iosfc_write(id, 0x1008, 0x400, 4));
        assert_eq!(device_iosfc_read(id, 0x1008, 4), Some(0x400));
        assert!(device_destroy(id));
    }

    #[test]
    fn drain_without_ops_is_ok() {
        let id = device_create(None, PAGE_SHIFT_ARM64E).expect("create");
        assert!(device_drain(id));
        assert!(device_pop_action(id).is_none());
        assert!(device_destroy(id));
    }

    #[cfg(all(feature = "host-window", target_os = "macos"))]
    #[test]
    fn window_publish_key_advances_for_in_place_present() {
        let mut state =
            crate::model::DeviceState::new(crate::model::DeviceId(1), PAGE_SHIFT_ARM64E);
        state.present.frame_mapping = 7;
        state.present.frame_generation = 11;
        let first = window_frame_key(&state.present);

        state.advance_present_epoch();
        assert_ne!(
            window_frame_key(&state.present),
            first,
            "a repeated resource generation still represents a new DisplaySwap"
        );
    }

    /// The guest ISR read of the read-to-clear interrupt-status registers
    /// must observe live bits (and clear them) even while the drain worker
    /// owns the device lock — a stale cached mask loses stamp signals.
    #[test]
    fn intr_status_reads_are_live_while_device_lock_held() {
        let id = device_create(None, PAGE_SHIFT_ARM64E).expect("create");
        let slot = device_slot(id).expect("slot");
        let _drain_guard = slot.inner.lock();
        // Drain-side signal lands while the lock is held.
        slot.intr_gpu.fetch_or(0x21, Ordering::AcqRel);
        slot.intr_disp.fetch_or(0x1, Ordering::AcqRel);
        // ISR sees live bits; second read is clear (read-to-clear).
        assert_eq!(device_gfx_read(id, 0x1018, 4), Some(0x21));
        assert_eq!(device_gfx_read(id, 0x1018, 4), Some(0));
        assert_eq!(device_gfx_read(id, 0x1014, 4), Some(0x1));
        assert_eq!(device_gfx_read(id, 0x1014, 4), Some(0));
        drop(_drain_guard);
        assert!(device_destroy(id));
    }

    /// The main-FIFO consumer counter (0x100c) must show drain progress live
    /// while the device lock is held — the guest writeFifo producer spins on
    /// it and a cached pre-tranche snapshot stalls the producer for the whole
    /// tranche.
    #[test]
    fn fifo_read_counter_is_live_while_device_lock_held() {
        let id = device_create(None, PAGE_SHIFT_ARM64E).expect("create");
        let slot = device_slot(id).expect("slot");
        let _drain_guard = slot.inner.lock();
        slot.fifo_read_live.store(0x1234, Ordering::Release);
        assert_eq!(device_gfx_read(id, 0x100c, 4), Some(0x1234));
        slot.fifo_read_live.store(0x1300, Ordering::Release);
        assert_eq!(device_gfx_read(id, 0x100c, 4), Some(0x1300));
        drop(_drain_guard);
        assert!(device_destroy(id));
    }

    /// Interrupt-status mask-clear writes apply lock-free too.
    #[test]
    fn intr_status_write_clears_mask_while_device_lock_held() {
        let id = device_create(None, PAGE_SHIFT_ARM64E).expect("create");
        let slot = device_slot(id).expect("slot");
        let _drain_guard = slot.inner.lock();
        slot.intr_gpu.fetch_or(0x7, Ordering::AcqRel);
        assert!(device_gfx_write(id, 0x1018, 0x2, 4));
        assert_eq!(device_gfx_read(id, 0x1018, 4), Some(0x5));
        drop(_drain_guard);
        assert!(device_destroy(id));
    }

    /// Prompt actions (IRQ pulses) pop without the device lock so the BH can
    /// deliver MSIs mid-drain; lock-owning actions still wait for the lock.
    #[test]
    fn prompt_actions_pop_while_device_lock_held() {
        let id = device_create(None, PAGE_SHIFT_ARM64E).expect("create");
        let slot = device_slot(id).expect("slot");
        slot.prompt_actions.lock().push_back(HostAction::irq_gfx());
        let _drain_guard = slot.inner.lock();
        let a = device_pop_action(id).expect("prompt action pops mid-drain");
        assert_eq!(a.kind, runtime::HostActionKind::IrqGfxPulse as u32);
        assert!(device_pop_action(id).is_none());
        drop(_drain_guard);
        assert!(device_destroy(id));
    }

    /// The interrupt-status atomics stay wired to the same slot across reset
    /// (GfxRegs::reset must preserve the shared Arcs, only zeroing values).
    #[test]
    fn intr_status_atomics_survive_reset() {
        let id = device_create(None, PAGE_SHIFT_ARM64E).expect("create");
        let slot = device_slot(id).expect("slot");
        slot.intr_gpu.fetch_or(0xff, Ordering::AcqRel);
        assert!(device_reset(id));
        // Reset cleared pending bits.
        assert_eq!(device_gfx_read(id, 0x1018, 4), Some(0));
        // Post-reset signals still reach the lock-free read rail.
        {
            let d = slot.inner.lock();
            d.device
                .state
                .gfx
                .interrupt_status_gpu
                .fetch_or(0x9, Ordering::AcqRel);
        }
        assert_eq!(device_gfx_read(id, 0x1018, 4), Some(0x9));
        assert!(device_destroy(id));
    }

    /// Pre-boundary without early front: BAR1/efi. Boundary or early latch: leave.
    #[test]
    fn host_console_bar1_until_present_boundary() {
        // (frame_flush, early_latched)
        assert!(host_console_uses_bar1(false, false));
        assert!(!host_console_uses_bar1(true, false));
        assert!(!host_console_uses_bar1(false, true));
        assert!(!host_console_uses_bar1(true, true));

        let id = device_create(None, PAGE_SHIFT_ARM64E).expect("create");
        assert_eq!(device_present_boundary_seen(id), Some(false));
        assert!(host_console_uses_bar1(
            device_present_boundary_seen(id).unwrap(),
            false
        ));

        // Present bookkeeping without boundary must not leave BAR1 by itself —
        // only frame_flush_seen or early_front_latched.
        {
            let slot = device_slot(id).expect("device");
            let mut d = slot.inner.lock();
            d.device.state.present.valid = true;
            d.device.state.present.width = 1920;
            d.device.state.present.height = 1080;
            d.device.state.present.present_mapping = 3;
        }
        assert_eq!(device_present_boundary_seen(id), Some(false));
        assert!(host_console_uses_bar1(false, false));
        // Early latch (protocol) leaves BAR1 before DisplaySwap.
        assert!(!host_console_uses_bar1(false, true));

        {
            let slot = device_slot(id).expect("device");
            let mut d = slot.inner.lock();
            d.device.state.present.frame_flush_seen = true;
            publish_present_boundary(&slot, d.device.state.present.frame_flush_seen);
        }
        assert_eq!(device_present_boundary_seen(id), Some(true));
        assert!(!host_console_uses_bar1(
            device_present_boundary_seen(id).unwrap(),
            false
        ));
        assert!(device_destroy(id));
    }

    #[test]
    fn present_boundary_query_is_monotonic_and_lock_free() {
        let id = device_create(None, PAGE_SHIFT_ARM64E).expect("create");
        let slot = device_slot(id).expect("device");
        let inner = slot.inner.lock();

        assert_eq!(device_present_boundary_seen(id), Some(false));
        publish_present_boundary(&slot, true);
        assert_eq!(
            device_present_boundary_seen(id),
            Some(true),
            "QEMU refresh must read the boundary while the worker owns device state"
        );
        publish_present_boundary(&slot, false);
        assert_eq!(
            device_present_boundary_seen(id),
            Some(true),
            "the per-boot product-console boundary must not regress to BAR1"
        );

        drop(inner);
        assert!(device_reset(id));
        assert_eq!(device_present_boundary_seen(id), Some(false));
        assert!(device_destroy(id));
    }

    /// Regression proxy for the IPI-timeout class: a doorbell arriving while
    /// the render worker owns device state queues without waiting for that
    /// state lock, then is applied by the next ordered drain.
    #[test]
    fn gfx_mmio_queues_while_render_worker_owns_device() {
        let id = device_create(None, PAGE_SHIFT_ARM64E).expect("create");
        let slot = device_slot(id).expect("device");
        let inner = slot.inner.lock();

        assert!(device_gfx_write(
            id,
            crate::model::GFX_REG_CHILD_DOORBELL,
            4,
            crate::model::MMIO_U32,
        ));
        assert_eq!(slot.gfx_ingress.lock().len(), 1);
        drop(inner);

        assert!(device_drain(id));
        let inner = slot.inner.lock();
        assert_eq!(slot.gfx_ingress.lock().len(), 0);
        assert_ne!(inner.device.state.pending.child_mask & (1 << 4), 0);
        drop(inner);
        assert!(device_destroy(id));
    }

    #[test]
    fn present_action_owns_worker_boundary_until_scanout_copy() {
        // Still valid as a test of `device_scanout_copy`'s own contract, which is
        // reachable for pre-boundary console paints and QMP screendump. Note the
        // production present path no longer depends on it: `device_drain` acks
        // each present itself after publishing to the host window, since no
        // per-present `ScanoutUpdate` is enqueued for QEMU to apply.
        let id = device_create(Some(null_host_ops()), PAGE_SHIFT_ARM64E).expect("create");
        let slot = device_slot(id).expect("device");
        {
            let mut inner = slot.inner.lock();
            let present = &mut inner.device.state.present;
            present.frame_valid = true;
            present.frame_mapping = 4;
            present.frame_width = 2;
            present.frame_height = 2;
            present.frame_generation = 7;
            present.frame_bgra = vec![0x55; 16];
            present.unpainted_presents = 1;
            inner.device.state.pending.host_action_yield = true;
        }
        slot.present_action_pending.store(true, Ordering::Release);
        slot.gfx_ingress.lock().push_back(QueuedGfxWrite {
            offset: crate::model::GFX_REG_CHILD_DOORBELL,
            data: 4,
            size: crate::model::MMIO_U32,
        });

        assert!(device_drain(id));
        assert_eq!(
            slot.gfx_ingress.lock().len(),
            1,
            "a newly woken worker must not overtake the queued scanout action"
        );

        let mut dst = vec![0u8; 16];
        assert_eq!(
            device_scanout_copy(id, 4, &mut dst, 8, 2, 2, 7),
            runtime::scanout::ScanoutCopyResult::Painted
        );
        assert!(!slot.present_action_pending.load(Ordering::Acquire));
        assert!(!slot.inner.lock().device.state.pending.host_action_yield);

        assert!(device_drain(id));
        assert_eq!(slot.gfx_ingress.lock().len(), 0);
        assert!(device_destroy(id));
    }
}
