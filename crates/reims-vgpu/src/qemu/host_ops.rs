//! HostOps / HostActions: services Rust cannot provide (guest memory, IRQ, display).
//!
//! Pattern mirrors apple-gfx ↔ ParavirtualizedGraphics.framework:
//! QEMU C owns only the host-service callbacks; Rust owns protocol + drain and
//! enqueues [`ReimsVgpuHostAction`]s for a QEMU BH to apply on the main loop.

use crate::runtime::host::{HostAction, HostActionKind, HostMemory, HostOps, MemError};
use std::collections::VecDeque;
use std::os::raw::{c_int, c_void};

/// Versioned host callback table offered by QEMU C to Rust.
///
/// Layout must match `ReimsVgpuHostOps` in `include/reims_vgpu_qemu_abi.h`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ReimsVgpuHostOps {
    pub abi_version: u32,
    pub struct_size: u32,
    pub ctx: *mut c_void,
    /// Read guest physical memory into `buf` (`len` bytes) from `gpa`.
    /// Returns 0 on success.
    pub read_gpa:
        Option<unsafe extern "C" fn(ctx: *mut c_void, gpa: u64, buf: *mut u8, len: usize) -> i32>,
    /// Write guest physical memory from `buf`.
    pub write_gpa:
        Option<unsafe extern "C" fn(ctx: *mut c_void, gpa: u64, buf: *const u8, len: usize) -> i32>,
    /// Monotonic nanoseconds.
    pub mono_ns: Option<unsafe extern "C" fn(ctx: *mut c_void) -> u64>,
    /// Wake QEMU main-loop BH to drain pending work / HostActions.
    /// Safe from any thread (schedules oneshot BH).
    pub schedule_bh: Option<unsafe extern "C" fn(ctx: *mut c_void)>,
    /// Read guest kernel VA (cpu_memory_rw_debug). Returns 0 on success.
    pub read_kva:
        Option<unsafe extern "C" fn(ctx: *mut c_void, kva: u64, buf: *mut u8, len: usize) -> i32>,
    /// Read guest CPU X-register `index` into `*out`. Returns 0 on success.
    pub read_xreg: Option<unsafe extern "C" fn(ctx: *mut c_void, index: u32, out: *mut u64) -> i32>,
    /// Contiguous host-VA view of guest pages (mach_vm_remap of guest RAM).
    pub map_pages: Option<
        unsafe extern "C" fn(
            ctx: *mut c_void,
            gpas: *const u64,
            count: usize,
            out_ptr: *mut *mut c_void,
        ) -> i32,
    >,
    pub unmap_pages: Option<unsafe extern "C" fn(ctx: *mut c_void, ptr: *mut c_void, len: usize)>,
    /// 1 = guest RAM, 0 = not RAM. Optional (None → treat as RAM for unit fixtures).
    pub is_ram_gpa: Option<unsafe extern "C" fn(ctx: *mut c_void, gpa: u64) -> i32>,
    /// Schedule the HostAction-delivery BH (pop_action consumer). Safe from any
    /// thread. Distinct from `schedule_bh` (drain-worker wake): prompt actions
    /// (IRQ pulses, cursor moves) must be deliverable mid-drain.
    pub notify_actions: Option<unsafe extern "C" fn(ctx: *mut c_void)>,
    /// 1 when `map_pages` returns a stable guest-RAM alias whose address is
    /// never unmapped or recycled during the device lifetime (x86 PCI: a
    /// direct RAMBlock pointer; arm MMIO: a direct HVA or retained packed
    /// `mach_vm_remap` view). `unmap_pages` is a no-op. Only a stable alias may
    /// be retained in a cached host-pointer import — see `map_pages_stable`.
    pub map_pages_stable: c_int,
}

// SAFETY: QEMU keeps the table valid for the device lifetime; callbacks only
// touch QEMU state under the BQL / from the AIO BH. We store the table as raw
// pointers and never move the C context.
unsafe impl Send for ReimsVgpuHostOps {}
unsafe impl Sync for ReimsVgpuHostOps {}

/// Failures in the QEMU service adapter that cannot ride a fallible HostOps
/// return value.
///
/// Guest-memory reads and writes return [`MemError`] directly. The clock,
/// wake, and page-map methods predate I2 and still expose `u64`, `()`, or
/// `Option<usize>`, so an omitted callback or failed map would otherwise be
/// indistinguishable from an ordinary higher-level miss.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum QemuHostDecline {
    MonoNsCallbackMissing,
    ScheduleBhCallbackMissing,
    MapPagesCallbackMissing {
        first_gpa: u64,
        page_count: usize,
        page_size: usize,
    },
    MapPagesCallbackFailed {
        rc: i32,
        first_gpa: u64,
        page_count: usize,
        page_size: usize,
    },
    MapPagesNullPointer {
        first_gpa: u64,
        page_count: usize,
        page_size: usize,
    },
    UnmapPagesCallbackMissing {
        ptr: usize,
        len: usize,
    },
}

impl crate::observe::Decline for QemuHostDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::MonoNsCallbackMissing => "qemu_mono_ns_callback_missing",
            Self::ScheduleBhCallbackMissing => "qemu_schedule_bh_callback_missing",
            Self::MapPagesCallbackMissing { .. } => "qemu_map_pages_callback_missing",
            Self::MapPagesCallbackFailed { .. } => "qemu_map_pages_callback_failed",
            Self::MapPagesNullPointer { .. } => "qemu_map_pages_null_pointer",
            Self::UnmapPagesCallbackMissing { .. } => "qemu_unmap_pages_callback_missing",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::MonoNsCallbackMissing | Self::ScheduleBhCallbackMissing => Vec::new(),
            Self::MapPagesCallbackMissing {
                first_gpa,
                page_count,
                page_size,
            }
            | Self::MapPagesNullPointer {
                first_gpa,
                page_count,
                page_size,
            } => vec![
                ("first_gpa", format!("{first_gpa:#x}")),
                ("page_count", page_count.to_string()),
                ("page_size", page_size.to_string()),
            ],
            Self::MapPagesCallbackFailed {
                rc,
                first_gpa,
                page_count,
                page_size,
            } => vec![
                ("rc", rc.to_string()),
                ("first_gpa", format!("{first_gpa:#x}")),
                ("page_count", page_count.to_string()),
                ("page_size", page_size.to_string()),
            ],
            Self::UnmapPagesCallbackMissing { ptr, len } => {
                vec![("ptr", format!("{ptr:#x}")), ("len", len.to_string())]
            }
        }
    }
}

impl QemuHostDecline {
    fn emit(self, discriminant: u64) {
        crate::observe::Emit::decline("qemu_host_adapter", &self).fail_once(discriminant);
    }
}

/// Typed actions Rust enqueues for the QEMU BH to perform on the main loop.
///
/// Discriminants match [`HostActionKind`] and `REIMS_VGPU_HOST_ACTION_*` in the header.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReimsVgpuHostActionKind {
    None = 0,
    IrqGfxPulse = 1,
    IrqIosfcPulse = 2,
    ScanoutUpdate = 3,
    CursorUpdate = 4,
    Trace = 5,
    CursorGlyph = 6,
    // 7 is retired (the removed GL/dmabuf scanout action); the values below are
    // spelled out so its removal did not renumber the wire.
    InputKey = 8,
    InputPointerMove = 9,
    InputPointerButton = 10,
    WindowClosed = 11,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ReimsVgpuHostAction {
    pub kind: u32,
    pub a0: u64,
    pub a1: u64,
    pub a2: u64,
    pub a3: u64,
}

impl Default for ReimsVgpuHostAction {
    fn default() -> Self {
        Self {
            kind: ReimsVgpuHostActionKind::None as u32,
            a0: 0,
            a1: 0,
            a2: 0,
            a3: 0,
        }
    }
}

impl From<HostAction> for ReimsVgpuHostAction {
    fn from(a: HostAction) -> Self {
        Self {
            kind: a.kind as u32,
            a0: a.a0,
            a1: a.a1,
            a2: a.a2,
            a3: a.a3,
        }
    }
}

/// Production host bridge: GPA/KVA via C callbacks, actions queued for the BH.
///
/// Two action rails:
/// - `actions` (inside the device lock): scanout / cursor-glyph / trace —
///   delivered by the BH after the drain tranche releases the lock (the
///   scanout apply re-enters the device for the +0x188 copy).
/// - `prompt` (outside the device lock): IRQ pulses + cursor moves — pushed to
///   the slot-level queue and `notify_actions`-scheduled immediately, so a
///   guest ISR sees its stamp-completion MSI while the drain worker is still
///   rendering later packets (ack fast / render async).
pub struct QemuHost<'a> {
    ops: &'a ReimsVgpuHostOps,
    actions: &'a mut VecDeque<HostAction>,
    prompt: Option<&'a parking_lot::Mutex<VecDeque<HostAction>>>,
}

impl<'a> QemuHost<'a> {
    pub fn new(ops: &'a ReimsVgpuHostOps, actions: &'a mut VecDeque<HostAction>) -> Self {
        Self {
            ops,
            actions,
            prompt: None,
        }
    }

    pub fn with_prompt(
        ops: &'a ReimsVgpuHostOps,
        actions: &'a mut VecDeque<HostAction>,
        prompt: &'a parking_lot::Mutex<VecDeque<HostAction>>,
    ) -> Self {
        Self {
            ops,
            actions,
            prompt: Some(prompt),
        }
    }

    fn notify_actions(&self) {
        if let Some(f) = self.ops.notify_actions {
            // SAFETY: QEMU owns ctx; thread-safe oneshot BH schedule.
            unsafe { f(self.ops.ctx) }
        }
    }

    fn callback_decline(error: MemError, address: u64, len: usize, discriminant: u64) -> MemError {
        crate::observe::Emit::decline("qemu_host_callback", &error)
            .field("address", format!("{address:#x}"))
            .field("len", len)
            .fail_once(discriminant);
        error
    }
}

impl HostMemory for QemuHost<'_> {
    fn read_gpa(&self, gpa: u64, buf: &mut [u8]) -> Result<(), MemError> {
        if buf.is_empty() {
            return Ok(());
        }
        let Some(f) = self.ops.read_gpa else {
            return Err(Self::callback_decline(
                MemError::QemuReadGpaCallbackMissing,
                gpa,
                buf.len(),
                0,
            ));
        };
        // SAFETY: QEMU owns ctx; buf is valid for len.
        let rc = unsafe { f(self.ops.ctx, gpa, buf.as_mut_ptr(), buf.len()) };
        match rc {
            0 => Ok(()),
            -2 => Err(MemError::NoCpu),
            _ => Err(Self::callback_decline(
                MemError::QemuReadGpaCallbackFailed(rc),
                gpa,
                buf.len(),
                gpa,
            )),
        }
    }

    fn write_gpa(&mut self, gpa: u64, buf: &[u8]) -> Result<(), MemError> {
        if buf.is_empty() {
            return Ok(());
        }
        let Some(f) = self.ops.write_gpa else {
            return Err(Self::callback_decline(
                MemError::QemuWriteGpaCallbackMissing,
                gpa,
                buf.len(),
                0,
            ));
        };
        // SAFETY: QEMU owns ctx; buf is valid for len.
        let rc = unsafe { f(self.ops.ctx, gpa, buf.as_ptr(), buf.len()) };
        if rc == 0 {
            Ok(())
        } else {
            Err(Self::callback_decline(
                MemError::QemuWriteGpaCallbackFailed(rc),
                gpa,
                buf.len(),
                gpa,
            ))
        }
    }
}

impl HostOps for QemuHost<'_> {
    fn mono_ns(&self) -> u64 {
        match self.ops.mono_ns {
            // SAFETY: QEMU owns ctx.
            Some(f) => unsafe { f(self.ops.ctx) },
            None => {
                QemuHostDecline::MonoNsCallbackMissing.emit(0);
                0
            }
        }
    }

    fn enqueue(&mut self, action: HostAction) {
        // Prompt rail: IRQ pulses and cursor moves carry no device state and
        // must not wait for the drain tranche to finish. Push to the slot
        // queue (poppable without the device lock) and wake the delivery BH.
        if let Some(prompt) = self.prompt {
            match action.kind {
                HostActionKind::IrqGfxPulse | HostActionKind::IrqIosfcPulse => {
                    let mut q = prompt.lock();
                    // Coalesce: an undelivered pulse of the same kind already
                    // covers this one (status bits accumulate in the r2c regs).
                    if !q.iter().any(|a| a.kind == action.kind) {
                        q.push_back(action);
                    }
                    drop(q);
                    self.notify_actions();
                    return;
                }
                HostActionKind::CursorUpdate => {
                    let mut q = prompt.lock();
                    q.retain(|a| a.kind != HostActionKind::CursorUpdate);
                    q.push_back(action);
                    drop(q);
                    self.notify_actions();
                    return;
                }
                HostActionKind::InputKey
                | HostActionKind::InputPointerMove
                | HostActionKind::InputPointerButton
                | HostActionKind::WindowClosed => {
                    // Host-window input + the window-closed signal: ordered and
                    // lossless. Unlike cursor moves and IRQ pulses these must NOT
                    // coalesce — a dropped key-up sticks a modifier, a reordered
                    // move+click lands the click at the wrong spot, and a dropped
                    // WindowClosed would leave the VM running headless. Push in
                    // arrival order and wake the delivery BH so the guest (or the
                    // shutdown path) sees it without waiting for a drain tranche.
                    prompt.lock().push_back(action);
                    self.notify_actions();
                    return;
                }
                _ => {}
            }
        }
        // apple-gfx new_frame_handler_bh: drop frames when guest gets too far
        // ahead of encode (pending_frames >= 2). Product: coalesce pending
        // ScanoutUpdates so BH paint encodes current +0x188 once (latest
        // presentFrame), not a backlog of dual-mid halves.
        if action.kind == HostActionKind::ScanoutUpdate {
            self.actions
                .retain(|a| a.kind != HostActionKind::ScanoutUpdate);
        }
        self.actions.push_back(action);
    }

    fn schedule_bh(&mut self) {
        if let Some(f) = self.ops.schedule_bh {
            // SAFETY: QEMU owns ctx; schedules oneshot BH (apple-gfx pattern).
            unsafe { f(self.ops.ctx) }
        } else {
            QemuHostDecline::ScheduleBhCallbackMissing.emit(0);
        }
    }

    fn read_kva(&self, kva: u64, buf: &mut [u8]) -> Result<(), MemError> {
        if buf.is_empty() {
            return Ok(());
        }
        let Some(f) = self.ops.read_kva else {
            return Err(Self::callback_decline(
                MemError::QemuReadKvaCallbackMissing,
                kva,
                buf.len(),
                0,
            ));
        };
        // SAFETY: QEMU owns ctx; buf valid for len.
        let rc = unsafe { f(self.ops.ctx, kva, buf.as_mut_ptr(), buf.len()) };
        match rc {
            0 => Ok(()),
            -2 => Err(MemError::NoCpu),
            _ => Err(Self::callback_decline(
                MemError::QemuReadKvaCallbackFailed(rc),
                kva,
                buf.len(),
                kva,
            )),
        }
    }

    fn read_xreg(&self, index: u32) -> Result<u64, MemError> {
        let Some(f) = self.ops.read_xreg else {
            return Err(Self::callback_decline(
                MemError::QemuReadXregCallbackMissing,
                u64::from(index),
                std::mem::size_of::<u64>(),
                0,
            ));
        };
        let mut out = 0u64;
        // SAFETY: QEMU owns ctx; out is stack local.
        let rc = unsafe { f(self.ops.ctx, index, &mut out) };
        if rc == 0 {
            Ok(out)
        } else {
            Err(Self::callback_decline(
                MemError::QemuReadXregCallbackFailed(rc),
                u64::from(index),
                std::mem::size_of::<u64>(),
                u64::from(index),
            ))
        }
    }

    fn map_pages(&mut self, gpas: &[u64], page_size: usize) -> Option<usize> {
        if gpas.is_empty() {
            return None;
        }
        // QEMU C side uses the device guest page shift (x86 4 KiB / arm 16 KiB).
        let first_gpa = gpas[0];
        let Some(f) = self.ops.map_pages else {
            QemuHostDecline::MapPagesCallbackMissing {
                first_gpa,
                page_count: gpas.len(),
                page_size,
            }
            .emit(first_gpa);
            return None;
        };
        let mut out: *mut c_void = std::ptr::null_mut();
        // SAFETY: QEMU owns ctx; gpas valid for count; out is stack local.
        let rc = unsafe { f(self.ops.ctx, gpas.as_ptr(), gpas.len(), &mut out) };
        if rc != 0 {
            QemuHostDecline::MapPagesCallbackFailed {
                rc,
                first_gpa,
                page_count: gpas.len(),
                page_size,
            }
            .emit(first_gpa);
            return None;
        }
        if out.is_null() {
            QemuHostDecline::MapPagesNullPointer {
                first_gpa,
                page_count: gpas.len(),
                page_size,
            }
            .emit(first_gpa);
            return None;
        }
        Some(out as usize)
    }

    fn map_pages_stable(&self) -> bool {
        self.ops.map_pages_stable != 0
    }

    fn unmap_pages(&mut self, ptr: usize, len: usize) {
        if ptr == 0 || len == 0 {
            return;
        }
        if let Some(f) = self.ops.unmap_pages {
            // SAFETY: ptr/len came from a successful map_pages.
            unsafe { f(self.ops.ctx, ptr as *mut c_void, len) }
        } else if !self.map_pages_stable() {
            // Stable aliases explicitly require no release. A missing callback
            // is a leak only for a transient view.
            QemuHostDecline::UnmapPagesCallbackMissing { ptr, len }.emit(ptr as u64);
        }
    }

    fn is_ram_gpa(&self, gpa: u64) -> bool {
        match self.ops.is_ram_gpa {
            // SAFETY: QEMU owns ctx; pure address-space query.
            Some(f) => unsafe { f(self.ops.ctx, gpa) != 0 },
            // Older/missing table: do not invent a reject (map_pages still RAM-checks).
            None => true,
        }
    }
}

/// Host used when no QEMU ops table is bound (unit tests / headless create).
pub struct NullHost;

impl HostMemory for NullHost {
    fn read_gpa(&self, _gpa: u64, _buf: &mut [u8]) -> Result<(), MemError> {
        Err(MemError::Unmapped)
    }
    fn write_gpa(&mut self, _gpa: u64, _buf: &[u8]) -> Result<(), MemError> {
        Err(MemError::Unmapped)
    }
}

impl HostOps for NullHost {
    fn mono_ns(&self) -> u64 {
        0
    }
    fn enqueue(&mut self, _action: HostAction) {}
    fn schedule_bh(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::host::HostAction as HA;

    unsafe extern "C" fn fail_read_gpa(
        _ctx: *mut c_void,
        _gpa: u64,
        _buf: *mut u8,
        _len: usize,
    ) -> i32 {
        -7
    }

    unsafe extern "C" fn fail_write_gpa(
        _ctx: *mut c_void,
        _gpa: u64,
        _buf: *const u8,
        _len: usize,
    ) -> i32 {
        -8
    }

    unsafe extern "C" fn no_cpu_read_kva(
        _ctx: *mut c_void,
        _kva: u64,
        _buf: *mut u8,
        _len: usize,
    ) -> i32 {
        -2
    }

    unsafe extern "C" fn fail_read_xreg(_ctx: *mut c_void, _index: u32, _out: *mut u64) -> i32 {
        -9
    }

    unsafe extern "C" fn fail_map_pages(
        _ctx: *mut c_void,
        _gpas: *const u64,
        _count: usize,
        _out: *mut *mut c_void,
    ) -> i32 {
        -11
    }

    unsafe extern "C" fn null_map_pages(
        _ctx: *mut c_void,
        _gpas: *const u64,
        _count: usize,
        out: *mut *mut c_void,
    ) -> i32 {
        // SAFETY: the HostOps callback contract supplies a writable out slot.
        unsafe {
            *out = std::ptr::null_mut();
        }
        0
    }

    fn null_ops() -> ReimsVgpuHostOps {
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
    fn enqueue_routes_prompt_kinds_to_prompt_queue() {
        let ops = null_ops();
        let mut actions = VecDeque::new();
        let prompt = parking_lot::Mutex::new(VecDeque::new());
        let mut host = QemuHost::with_prompt(&ops, &mut actions, &prompt);

        host.enqueue(HA::irq_gfx());
        host.enqueue(HA::irq_gfx()); // coalesced: one undelivered pulse covers both
        host.enqueue(HA::irq_iosfc());
        host.enqueue(HA::cursor(10, 20, true));
        host.enqueue(HA::cursor(30, 40, true)); // latest cursor wins
        host.enqueue(HA::scanout_gen(1, 1920, 1080, 7));

        let q = prompt.lock();
        assert_eq!(q.len(), 3, "gfx pulse + iosfc pulse + one cursor");
        assert_eq!(q[0].kind, HostActionKind::IrqGfxPulse);
        assert_eq!(q[1].kind, HostActionKind::IrqIosfcPulse);
        assert_eq!(q[2].kind, HostActionKind::CursorUpdate);
        assert_eq!(q[2].a0, 30);
        drop(q);
        assert_eq!(actions.len(), 1, "scanout stays on the lock-owning rail");
        assert_eq!(actions[0].kind, HostActionKind::ScanoutUpdate);
    }

    #[test]
    fn enqueue_without_prompt_keeps_legacy_single_queue() {
        let ops = null_ops();
        let mut actions = VecDeque::new();
        let mut host = QemuHost::new(&ops, &mut actions);
        host.enqueue(HA::irq_gfx());
        host.enqueue(HA::scanout_gen(1, 640, 480, 1));
        assert_eq!(actions.len(), 2);
    }

    #[test]
    fn missing_qemu_memory_callbacks_are_exact() {
        let ops = null_ops();
        let mut actions = VecDeque::new();
        let mut host = QemuHost::new(&ops, &mut actions);
        assert_eq!(
            host.read_gpa(0x1000, &mut [0; 1]),
            Err(MemError::QemuReadGpaCallbackMissing)
        );
        assert_eq!(
            host.write_gpa(0x1000, &[1]),
            Err(MemError::QemuWriteGpaCallbackMissing)
        );
        assert_eq!(
            host.read_kva(0xffff_fe00_1000, &mut [0; 1]),
            Err(MemError::QemuReadKvaCallbackMissing)
        );
        assert_eq!(
            host.read_xreg(19),
            Err(MemError::QemuReadXregCallbackMissing)
        );
    }

    #[test]
    fn qemu_callback_return_codes_keep_their_operation_and_value() {
        let mut ops = null_ops();
        ops.read_gpa = Some(fail_read_gpa);
        ops.write_gpa = Some(fail_write_gpa);
        ops.read_kva = Some(no_cpu_read_kva);
        ops.read_xreg = Some(fail_read_xreg);
        let mut actions = VecDeque::new();
        let mut host = QemuHost::new(&ops, &mut actions);
        assert_eq!(
            host.read_gpa(0x1000, &mut [0; 1]),
            Err(MemError::QemuReadGpaCallbackFailed(-7))
        );
        assert_eq!(
            host.write_gpa(0x1000, &[1]),
            Err(MemError::QemuWriteGpaCallbackFailed(-8))
        );
        assert_eq!(
            host.read_kva(0xffff_fe00_1000, &mut [0; 1]),
            Err(MemError::NoCpu),
            "-2 is the no-current-vCPU state, not an unmapped KVA"
        );
        assert_eq!(
            host.read_xreg(22),
            Err(MemError::QemuReadXregCallbackFailed(-9))
        );
        assert_eq!(
            crate::observe::Emit::decline(
                "qemu_host_callback",
                &MemError::QemuReadGpaCallbackFailed(-7),
            )
            .field("address", "0x1000")
            .field("len", 4)
            .render(),
            "qemu_host_callback reason=mem_qemu_read_gpa_callback_failed \
             rc=-7 address=0x1000 len=4"
        );
    }

    #[test]
    fn qemu_host_adapter_declines_are_exact_registered_and_log_safe() {
        use crate::observe::Decline;

        let declines = [
            QemuHostDecline::MonoNsCallbackMissing,
            QemuHostDecline::ScheduleBhCallbackMissing,
            QemuHostDecline::MapPagesCallbackMissing {
                first_gpa: 0x4000,
                page_count: 2,
                page_size: 0x4000,
            },
            QemuHostDecline::MapPagesCallbackFailed {
                rc: -11,
                first_gpa: 0x4000,
                page_count: 2,
                page_size: 0x4000,
            },
            QemuHostDecline::MapPagesNullPointer {
                first_gpa: 0x4000,
                page_count: 2,
                page_size: 0x4000,
            },
            QemuHostDecline::UnmapPagesCallbackMissing {
                ptr: 0x10000,
                len: 0x8000,
            },
        ];
        let expected = [
            "qemu_mono_ns_callback_missing",
            "qemu_schedule_bh_callback_missing",
            "qemu_map_pages_callback_missing",
            "qemu_map_pages_callback_failed",
            "qemu_map_pages_null_pointer",
            "qemu_unmap_pages_callback_missing",
        ];
        let row = crate::observe::REGISTRY
            .iter()
            .find(|class| class.type_name == "QemuHostDecline")
            .expect("QemuHostDecline registry row");
        assert_eq!(row.slugs, expected);
        for (decline, expected_slug) in declines.iter().zip(expected) {
            assert_eq!(decline.slug(), expected_slug);
            assert!(expected_slug
                .bytes()
                .all(|byte| byte.is_ascii_lowercase() || byte == b'_' || byte.is_ascii_digit()));
        }
        assert_eq!(
            crate::observe::Emit::decline("qemu_host_adapter", &declines[3]).render(),
            "qemu_host_adapter reason=qemu_map_pages_callback_failed \
             rc=-11 first_gpa=0x4000 page_count=2 page_size=16384"
        );
    }

    #[test]
    fn map_pages_distinguishes_missing_failed_and_null_callbacks() {
        let mut ops = null_ops();
        let mut actions = VecDeque::new();
        let mut host = QemuHost::new(&ops, &mut actions);
        assert_eq!(host.map_pages(&[0x4000], 0x4000), None);

        ops.map_pages = Some(fail_map_pages);
        let mut host = QemuHost::new(&ops, &mut actions);
        assert_eq!(host.map_pages(&[0x8000], 0x4000), None);

        ops.map_pages = Some(null_map_pages);
        let mut host = QemuHost::new(&ops, &mut actions);
        assert_eq!(host.map_pages(&[0xc000], 0x4000), None);
    }

    /// The two parallel action-kind enums (runtime `HostActionKind`, consumed by
    /// `From<HostAction>` via `as u32`, and the FFI `ReimsVgpuHostActionKind` that
    /// mirrors the C `REIMS_VGPU_HOST_ACTION_*` constants) must share discriminants, or
    /// a wire-value drift would mislabel actions to the C shim. Locks every kind.
    #[test]
    fn action_kind_discriminants_match_ffi_and_wire() {
        use crate::runtime::host::HostActionKind as K;
        let pairs = [
            (K::None, ReimsVgpuHostActionKind::None, 0u32),
            (K::IrqGfxPulse, ReimsVgpuHostActionKind::IrqGfxPulse, 1),
            (K::IrqIosfcPulse, ReimsVgpuHostActionKind::IrqIosfcPulse, 2),
            (K::ScanoutUpdate, ReimsVgpuHostActionKind::ScanoutUpdate, 3),
            (K::CursorUpdate, ReimsVgpuHostActionKind::CursorUpdate, 4),
            (K::Trace, ReimsVgpuHostActionKind::Trace, 5),
            (K::CursorGlyph, ReimsVgpuHostActionKind::CursorGlyph, 6),
            // 7 is a retired wire value (the removed GL/dmabuf scanout action).
            // The jump from 6 to 8 is deliberate: the input kinds keep the
            // values the C shim already dispatches on.
            (K::InputKey, ReimsVgpuHostActionKind::InputKey, 8),
            (
                K::InputPointerMove,
                ReimsVgpuHostActionKind::InputPointerMove,
                9,
            ),
            (
                K::InputPointerButton,
                ReimsVgpuHostActionKind::InputPointerButton,
                10,
            ),
            (K::WindowClosed, ReimsVgpuHostActionKind::WindowClosed, 11),
        ];
        for (k, ak, wire) in pairs {
            assert_eq!(k as u32, wire);
            assert_eq!(ak as u32, wire);
        }
    }
}
