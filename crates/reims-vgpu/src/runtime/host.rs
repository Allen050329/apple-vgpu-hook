//! Host memory + action sink abstractions for the device model.
//!
//! Production: QEMU C offers HostOps callbacks; Rust enqueues HostActions for a
//! QEMU BH. Tests: [`FakeHost`] owns an in-memory GPA space and action log.

use std::collections::BTreeMap;

/// Guest-physical memory access error.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemError {
    Unmapped,
    NoCpu,
    Overflow,
    BadArgs,
    /// The QEMU host table omitted a mandatory guest-physical read callback.
    QemuReadGpaCallbackMissing,
    /// QEMU's guest-physical read callback returned a transaction failure.
    QemuReadGpaCallbackFailed(i32),
    /// The QEMU host table omitted a mandatory guest-physical write callback.
    QemuWriteGpaCallbackMissing,
    /// QEMU's guest-physical write callback returned a transaction failure.
    QemuWriteGpaCallbackFailed(i32),
    /// The QEMU host table omitted the guest-kernel-VA debug-read callback.
    QemuReadKvaCallbackMissing,
    /// QEMU's guest-kernel-VA debug-read callback failed for a reason other
    /// than the explicitly represented [`Self::NoCpu`] state.
    QemuReadKvaCallbackFailed(i32),
    /// The host cannot expose a CPU register on this pathway.
    XregUnavailable,
    /// The QEMU host table omitted the guest-register callback.
    QemuReadXregCallbackMissing,
    /// QEMU's guest-register callback rejected the register read.
    QemuReadXregCallbackFailed(i32),
    /// The guest page-table walk refused, carrying **which** of its fifteen
    /// checks did.
    ///
    /// Every GVA path used to answer `Unmapped` here, so a malformed PTE, a
    /// zero root PFN and an out-of-range address were one value — on top of the
    /// genuinely-unmapped GPA cases that also answer `Unmapped`. That is the
    /// "one status for N checks" shape the ground rules name by example, and it
    /// sat on the guest-memory hot path.
    Unresolved(crate::contract::gva_resolve::ResolveStatus),
    /// The task is not active, or its directory PFN is zero, so there is no page
    /// table to walk. Distinct from [`Self::Unresolved`]: the walk never began.
    NoTaskDirectory,
    /// No page-table geometry for the guest's page shift. A create-time
    /// configuration error, not a guest one.
    UnsupportedPageShift,
    /// The task root (directory → root PFN + depth) could not be read.
    TaskRootRead,
    /// Neither the wire task id nor its `>> 1` define-task form names an active
    /// task, so there is no address space to resolve against.
    NoSuchTask,
    /// A page of the span resolves to a GPA that is not guest RAM, so no host
    /// mapping can cover it (mapper / wild-PFN class).
    NotRam,
    /// [`HostOps::map_pages`] refused a **packed** page run the walk had already
    /// resolved — a RAMBlock or MemoryRegion edge, not a gap in the GPA list.
    /// Fragmentation alone never reaches here: the multi-import path splits a
    /// gapped span into packed runs and maps them one at a time.
    MapPagesRefused,
    /// A packed run's copy window fell outside the bytes `map_pages` returned or
    /// outside the caller's buffer. Run arithmetic, not a guest condition.
    RunOutOfRange,
}

impl MemError {
    /// The guest's own page table says nothing is mapped at this address.
    ///
    /// A zero PFN in a task PTE is a *decoded guest fact*: the guest owns that
    /// entry and wrote the zero. So a deferred writeback whose target answers
    /// this has no target — the guest tore the range down — and that is a
    /// different outcome from a write that failed while the target still
    /// existed. Callers landing deferred content use it to pick between
    /// "discharge the obligation" and "report lost guest work".
    ///
    /// Deliberately **only** the zero-PFN status. The other fourteen walk
    /// refusals describe a table that is malformed, out of range or unreadable,
    /// none of which is the guest saying "I unmapped this", and widening the set
    /// to make a log quieter would turn this into the exception list the ground
    /// rules forbid.
    pub fn is_guest_teardown(&self) -> bool {
        matches!(
            self,
            Self::Unresolved(crate::contract::gva_resolve::ResolveStatus::ErrZeroPfn)
        )
    }
}

impl crate::observe::Decline for MemError {
    fn slug(&self) -> &'static str {
        match self {
            Self::Unmapped => "mem_unmapped",
            Self::NoCpu => "mem_no_cpu",
            Self::Overflow => "mem_overflow",
            Self::BadArgs => "mem_bad_args",
            Self::QemuReadGpaCallbackMissing => "mem_qemu_read_gpa_callback_missing",
            Self::QemuReadGpaCallbackFailed(_) => "mem_qemu_read_gpa_callback_failed",
            Self::QemuWriteGpaCallbackMissing => "mem_qemu_write_gpa_callback_missing",
            Self::QemuWriteGpaCallbackFailed(_) => "mem_qemu_write_gpa_callback_failed",
            Self::QemuReadKvaCallbackMissing => "mem_qemu_read_kva_callback_missing",
            Self::QemuReadKvaCallbackFailed(_) => "mem_qemu_read_kva_callback_failed",
            Self::XregUnavailable => "mem_xreg_unavailable",
            Self::QemuReadXregCallbackMissing => "mem_qemu_read_xreg_callback_missing",
            Self::QemuReadXregCallbackFailed(_) => "mem_qemu_read_xreg_callback_failed",
            // Delegates, so the walk's own fifteen slugs stay the reason rather
            // than being flattened into one and reconstructed from a field.
            Self::Unresolved(status) => match crate::observe::Refusal::refusal(status) {
                Some(slug) => slug,
                // `Unresolved(Ok)` is a construction bug, not a walk failure.
                // Naming it beats reporting a plausible walk reason for
                // something the walk never said.
                None => "mem_unresolved_ok",
            },
            Self::NoTaskDirectory => "mem_no_task_directory",
            Self::UnsupportedPageShift => "mem_unsupported_page_shift",
            Self::TaskRootRead => "mem_task_root_read",
            Self::NoSuchTask => "mem_no_such_task",
            Self::NotRam => "mem_not_ram",
            Self::MapPagesRefused => "mem_map_pages_refused",
            Self::RunOutOfRange => "mem_run_out_of_range",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::QemuReadGpaCallbackFailed(rc)
            | Self::QemuWriteGpaCallbackFailed(rc)
            | Self::QemuReadKvaCallbackFailed(rc)
            | Self::QemuReadXregCallbackFailed(rc) => vec![("rc", rc.to_string())],
            _ => Vec::new(),
        }
    }
}

/// Checked guest physical memory (no scanning; directed GPA only).
pub trait HostMemory {
    fn read_gpa(&self, gpa: u64, buf: &mut [u8]) -> Result<(), MemError>;
    fn write_gpa(&mut self, gpa: u64, buf: &[u8]) -> Result<(), MemError>;
}

/// Typed actions for the QEMU main loop (or FakeHost log).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HostActionKind {
    None = 0,
    IrqGfxPulse = 1,
    IrqIosfcPulse = 2,
    ScanoutUpdate = 3,
    CursorUpdate = 4,
    Trace = 5,
    /// New software cursor glyph ready in device state (C pulls via ABI).
    CursorGlyph = 6,
    // 7 is a retired wire value: it named a pre-host-window QEMU GL/dmabuf
    // scanout action that no longer exists on either side. Every discriminant
    // below is written out so removing it did not renumber the wire; do not
    // reuse 7 for a new action.
    /// A guest keyboard key from the host-owned window (see
    /// [`crate::runtime::input`]). `a0` = Linux evdev keycode (`KEY_*`),
    /// `a1` = 1 down / 0 up. The window thread maps the platform key into the
    /// stable evdev space; the C shim forwards it verbatim via
    /// `qemu_input_event_send_key_linux` (QEMU owns the evdev→qcode table), so
    /// no QEMU keycode constants leak into Rust. See [`HostAction::input_key`].
    InputKey = 8,
    /// Absolute pointer move from the host-owned window. `a0` = x pixel,
    /// `a1` = y pixel, `a2` = surface width (px), `a3` = surface height (px).
    /// Absolute (not relative) because the guest binds an absolute pointer
    /// (usb-tablet); the C shim scales into the abs axis range with
    /// `qemu_input_queue_abs` (min_in = 0, max_in = dim). See
    /// [`HostAction::input_pointer_move`].
    InputPointerMove = 9,
    /// Pointer button (including wheel) from the host-owned window. `a0` = the
    /// neutral [`crate::runtime::input::ReimsVgpuButton`] code, `a1` = 1 down / 0 up.
    /// The C shim maps the neutral code to QEMU's `InputButton`; a wheel click
    /// is a down+up pair the window thread emits, so the C side stays uniform
    /// (one `qemu_input_queue_btn` + sync per action). See
    /// [`HostAction::input_pointer_button`].
    InputPointerButton = 10,
    /// The host-owned window was closed through its UI (title-bar close /
    /// compositor close). Carries no payload; the C shim turns it into
    /// `qemu_system_shutdown_request` so closing the window shuts the VM down —
    /// the window is the VM's display, so closing it is closing the machine.
    /// See [`HostAction::window_closed`].
    WindowClosed = 11,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostAction {
    pub kind: HostActionKind,
    pub a0: u64,
    pub a1: u64,
    pub a2: u64,
    pub a3: u64,
}

impl HostAction {
    pub fn irq_gfx() -> Self {
        Self {
            kind: HostActionKind::IrqGfxPulse,
            a0: 0,
            a1: 0,
            a2: 0,
            a3: 0,
        }
    }

    pub fn irq_iosfc() -> Self {
        Self {
            kind: HostActionKind::IrqIosfcPulse,
            a0: 0,
            a1: 0,
            a2: 0,
            a3: 0,
        }
    }

    pub fn scanout_gen(mapping_id: u32, width: u32, height: u32, generation: u32) -> Self {
        Self {
            kind: HostActionKind::ScanoutUpdate,
            a0: mapping_id as u64,
            a1: width as u64,
            a2: height as u64,
            a3: generation as u64,
        }
    }

    pub fn cursor(x: u16, y: u16, show: bool) -> Self {
        Self {
            kind: HostActionKind::CursorUpdate,
            a0: x as u64,
            a1: y as u64,
            a2: u64::from(show),
            a3: 0,
        }
    }

    pub fn cursor_glyph() -> Self {
        Self {
            kind: HostActionKind::CursorGlyph,
            a0: 0,
            a1: 0,
            a2: 0,
            a3: 0,
        }
    }

    /// Guest keyboard key from the host-owned window. `evdev_keycode` is a Linux
    /// `KEY_*` code (the stable neutral space the window thread maps into); the
    /// C shim hands it straight to `qemu_input_event_send_key_linux`.
    pub fn input_key(evdev_keycode: u32, down: bool) -> Self {
        Self {
            kind: HostActionKind::InputKey,
            a0: u64::from(evdev_keycode),
            a1: u64::from(down),
            a2: 0,
            a3: 0,
        }
    }

    /// Absolute pointer move from the host-owned window. `x`/`y` are pixel
    /// coordinates within a `width`x`height` surface; the C shim scales them
    /// into the abs axis range. `width`/`height` must be non-zero (the window
    /// surface is always sized before a move is emitted).
    pub fn input_pointer_move(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self {
            kind: HostActionKind::InputPointerMove,
            a0: u64::from(x),
            a1: u64::from(y),
            a2: u64::from(width),
            a3: u64::from(height),
        }
    }

    /// Pointer button (or wheel) from the host-owned window. Wheel clicks are
    /// emitted as a `down` then `up` pair by the window thread.
    pub fn input_pointer_button(
        button: crate::runtime::input::ReimsVgpuButton,
        down: bool,
    ) -> Self {
        Self {
            kind: HostActionKind::InputPointerButton,
            a0: u64::from(button as u32),
            a1: u64::from(down),
            a2: 0,
            a3: 0,
        }
    }

    /// The host-owned window was closed through its UI. No payload; the C shim
    /// requests a VM shutdown when it applies this.
    pub fn window_closed() -> Self {
        Self {
            kind: HostActionKind::WindowClosed,
            a0: 0,
            a1: 0,
            a2: 0,
            a3: 0,
        }
    }
}

/// Services the device cannot provide itself (time, wake, action enqueue,
/// guest CPU / KVA access for the IOSurface mapper path).
pub trait HostOps {
    fn mono_ns(&self) -> u64;
    fn enqueue(&mut self, action: HostAction);
    fn schedule_bh(&mut self);

    /// Read guest kernel virtual address (cpu_memory_rw_debug). Default: fail.
    fn read_kva(&self, _kva: u64, _buf: &mut [u8]) -> Result<(), MemError> {
        Err(MemError::Unmapped)
    }

    /// Read guest CPU X-register `index` (0..30). Default: none.
    /// Used only on the MMIO path that publishes an iosfc mapper request so
    /// x19/x21/x22 still hold the directed MappingInternal handoff.
    fn read_xreg(&self, _index: u32) -> Result<u64, MemError> {
        Err(MemError::XregUnavailable)
    }

    /// Build one contiguous host-VA view over guest pages (page-aligned GPAs).
    ///
    /// `page_size` is the guest page size (4 KiB x86 / 16 KiB arm64e). Each
    /// `gpas[i]` is one guest page base. View length is `gpas.len() * page_size`.
    /// ParavirtualizedGraphics mapMemory model: the view aliases guest RAM so
    /// CPU/GPU access *is* guest memory. Default: unavailable.
    fn map_pages(&mut self, _gpas: &[u64], _page_size: usize) -> Option<usize> {
        None
    }

    /// Release a view obtained from [`HostOps::map_pages`].
    fn unmap_pages(&mut self, _ptr: usize, _len: usize) {}

    /// True when [`HostOps::map_pages`] returns a **stable** alias of guest
    /// RAM: the pointer stays valid for the device lifetime,
    /// [`HostOps::unmap_pages`] is a no-op, and the address is never recycled
    /// for unrelated memory.
    ///
    /// Only a stable alias may be retained in a cached
    /// `VK_EXT_external_memory_host` import window, which is what GPU-direct
    /// present writeback needs: the GPU writes through that import after the
    /// caller has already unmapped its view. Caching an import of a transient
    /// view leaves the GPU DMAing into an address range the host has since
    /// torn down or reused.
    ///
    /// Default `false` — the conservative answer, so a host that has not
    /// declared stability keeps the portable CPU writeback.
    fn map_pages_stable(&self) -> bool {
        false
    }

    /// True if `gpa` is guest RAM (not MMIO / ROM / unmapped). Product QEMU
    /// implements via `address_space_translate` + `memory_region_is_ram`.
    /// Default: true (fixtures / NullHost without a RAM map).
    fn is_ram_gpa(&self, _gpa: u64) -> bool {
        true
    }
}

/// Arm64e guest page size (16 KiB). FakeHost / map_pages test fixture only —
/// product paths use `state.page_size()` / `page_size_of(page_shift)`.
/// Named without a portable alias: do not use this as x86 page size.
pub const GUEST_PAGE_SIZE_ARM64E: usize = 16384;
/// Historical alias for arm fixtures; prefer [`GUEST_PAGE_SIZE_ARM64E`].
pub const GUEST_PAGE_SIZE: usize = GUEST_PAGE_SIZE_ARM64E;

/// mach VM aliasing for FakeHost views — the same mechanism the QEMU shim
/// uses in production (`mach_vm_remap` of guest RAM), exercised for real in
/// unit tests so view coherence is tested, not simulated.
#[cfg(target_os = "macos")]
mod mach_vm {
    #[allow(non_upper_case_globals)]
    extern "C" {
        pub static mach_task_self_: u32;
        pub fn mach_vm_allocate(task: u32, addr: *mut u64, size: u64, flags: i32) -> i32;
        pub fn mach_vm_deallocate(task: u32, addr: u64, size: u64) -> i32;
        #[allow(clippy::too_many_arguments)]
        pub fn mach_vm_remap(
            target: u32,
            addr: *mut u64,
            size: u64,
            mask: u64,
            flags: i32,
            src_task: u32,
            src_addr: u64,
            copy: i32,
            cur_protection: *mut i32,
            max_protection: *mut i32,
            inheritance: u32,
        ) -> i32;
    }
    pub const VM_FLAGS_ANYWHERE: i32 = 1;
    pub const VM_FLAGS_FIXED_OVERWRITE: i32 = 0x4000;
    pub const VM_INHERIT_NONE: u32 = 2;
}

/// A real, 16KiB-aligned memory block backing a GPA range in [`FakeHost`].
#[derive(Debug)]
struct RealRange {
    gpa: u64,
    len: usize,
    ptr: usize,
    alloc_len: usize,
}

/// Combined host for unit tests: GPA store + action log + BH flag.
///
/// GPA ranges are backed by real page-aligned host memory so
/// [`HostOps::map_pages`] views work exactly like production (mach_vm_remap
/// aliasing): a GPU/CPU write through a view is immediately visible via
/// `read_gpa` and vice versa. Bytes outside mapped ranges live in a sparse
/// map (synthetic KVA fixtures); unmapped reads stay permissive zeros.
#[derive(Debug, Default)]
pub struct FakeHost {
    ranges: Vec<RealRange>,
    /// Sparse byte store for addresses outside real ranges.
    pages: BTreeMap<u64, u8>,
    /// Live map_pages views (ptr, len) for cleanup (mach remap or bounce).
    views: Vec<(usize, usize)>,
    /// Linux bounce buffers: host ptr must be written back to GPA on unmap.
    bounce: Vec<BounceView>,
    /// Synthetic guest X-regs for mapper capture tests.
    pub xregs: BTreeMap<u32, u64>,
    pub actions: Vec<HostAction>,
    pub mono_ns: u64,
    pub bh_scheduled: bool,
    /// When true (any host platform): `map_pages` matches the product Linux
    /// PCI shim (`reims_vgpu_pci_map_pages`) — packed sequential host alias inside an
    /// existing range only; no provisioning, no bounce/remap for scattered
    /// GPAs. Used to unit-test multi-import of fragmented GVA spans.
    pub strict_linux_map: bool,
    /// Test-controlled answer for [`HostOps::map_pages_stable`]. Keep separate
    /// from `strict_linux_map`: packed shape and pointer lifetime are distinct
    /// host contracts.
    pub stable_map_pages: bool,
    /// Number of HostOps page-import attempts (test proxy for import amplification).
    pub map_pages_calls: u64,
}

/// Contiguous bounce for [`FakeHost::map_pages`] on non-macOS (sparse GPA store).
#[derive(Debug)]
struct BounceView {
    ptr: usize,
    #[cfg(not(target_os = "macos"))]
    len: usize,
    gpas: Vec<u64>,
    page_sz: usize,
}

impl Drop for FakeHost {
    fn drop(&mut self) {
        #[cfg(target_os = "macos")]
        unsafe {
            for (ptr, len) in self.views.drain(..) {
                mach_vm::mach_vm_deallocate(mach_vm::mach_task_self_, ptr as u64, len as u64);
            }
            for r in self.ranges.drain(..) {
                mach_vm::mach_vm_deallocate(
                    mach_vm::mach_task_self_,
                    r.ptr as u64,
                    r.alloc_len as u64,
                );
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            for b in self.bounce.drain(..) {
                // SAFETY: ptr from Box::into_raw in map_pages.
                unsafe {
                    let _ =
                        Box::from_raw(std::ptr::slice_from_raw_parts_mut(b.ptr as *mut u8, b.len));
                }
            }
            for r in self.ranges.drain(..) {
                let layout = std::alloc::Layout::from_size_align(r.alloc_len, GUEST_PAGE_SIZE)
                    .unwrap_or(std::alloc::Layout::from_size_align(r.alloc_len, 1).unwrap());
                // SAFETY: ptr/alloc_len from alloc_block.
                unsafe { std::alloc::dealloc(r.ptr as *mut u8, layout) };
            }
        }
    }
}

impl FakeHost {
    pub fn new() -> Self {
        Self::default()
    }

    fn alloc_block(len: usize) -> Option<(usize, usize)> {
        #[cfg(target_os = "macos")]
        unsafe {
            let alloc_len = len.max(1).next_multiple_of(GUEST_PAGE_SIZE);
            let mut addr = 0u64;
            let kr = mach_vm::mach_vm_allocate(
                mach_vm::mach_task_self_,
                &mut addr,
                alloc_len as u64,
                mach_vm::VM_FLAGS_ANYWHERE,
            );
            if kr != 0 {
                return None;
            }
            Some((addr as usize, alloc_len))
        }
        #[cfg(not(target_os = "macos"))]
        {
            // Real host pages so map_pages can return an aliasing pointer (product
            // contig write path). Align to 16 KiB so arm fixtures work.
            let alloc_len = len.max(1).next_multiple_of(GUEST_PAGE_SIZE);
            let layout = std::alloc::Layout::from_size_align(alloc_len, GUEST_PAGE_SIZE).ok()?;
            // SAFETY: non-zero layout.
            let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
            if ptr.is_null() {
                return None;
            }
            Some((ptr as usize, alloc_len))
        }
    }

    fn range_containing(&self, gpa: u64) -> Option<usize> {
        self.ranges
            .iter()
            .position(|r| gpa >= r.gpa && gpa < r.gpa + r.len as u64)
    }

    /// Register a real range at `gpa`, seeding from any sparse bytes there.
    fn provision_range(&mut self, gpa: u64, len: usize) -> Option<usize> {
        let (ptr, alloc_len) = Self::alloc_block(len)?;
        // Seed from sparse bytes previously written at these addresses.
        for off in 0..len as u64 {
            if let Some(b) = self.pages.remove(&gpa.wrapping_add(off)) {
                unsafe { *((ptr + off as usize) as *mut u8) = b };
            }
        }
        self.ranges.push(RealRange {
            gpa,
            len,
            ptr,
            alloc_len,
        });
        Some(self.ranges.len() - 1)
    }

    /// Map a contiguous GPA range filled with `fill` (or zeros).
    pub fn map_range(&mut self, gpa: u64, len: usize, fill: u8) {
        if len == 0 {
            return;
        }
        // Fully inside an existing range: fill in place.
        if let Some(i) = self.range_containing(gpa) {
            let r = &self.ranges[i];
            if gpa + len as u64 <= r.gpa + r.len as u64 {
                let off = (gpa - r.gpa) as usize;
                unsafe { std::ptr::write_bytes((r.ptr + off) as *mut u8, fill, len) };
                return;
            }
        }
        debug_assert!(
            !self
                .ranges
                .iter()
                .any(|r| gpa < r.gpa + r.len as u64 && r.gpa < gpa + len as u64),
            "FakeHost::map_range partial overlap at {gpa:#x}+{len:#x}"
        );
        if let Some(i) = self.provision_range(gpa, len) {
            let r = &self.ranges[i];
            unsafe { std::ptr::write_bytes(r.ptr as *mut u8, fill, len) };
        } else {
            // Non-macOS fallback: sparse bytes.
            for i in 0..len {
                self.pages.insert(gpa.wrapping_add(i as u64), fill);
            }
        }
    }

    /// Write a LE u32 at GPA.
    pub fn put_u32(&mut self, gpa: u64, v: u32) {
        let b = v.to_le_bytes();
        let _ = self.write_gpa(gpa, &b);
    }

    /// Read a LE u32 at GPA (zero if unmapped).
    pub fn get_u32(&self, gpa: u64) -> u32 {
        let mut b = [0u8; 4];
        let _ = self.read_gpa(gpa, &mut b);
        u32::from_le_bytes(b)
    }

    /// Count actions of a given kind.
    pub fn action_count(&self, kind: HostActionKind) -> usize {
        self.actions.iter().filter(|a| a.kind == kind).count()
    }

    /// Set a synthetic X-register value (mapper capture tests).
    pub fn set_xreg(&mut self, index: u32, value: u64) {
        self.xregs.insert(index, value);
    }
}

impl FakeHost {
    /// If `addr` is in a live bounce view, return `(bounce_base, offset, max_contig)`.
    fn bounce_slot(&self, addr: u64) -> Option<(usize, usize, usize)> {
        for b in &self.bounce {
            for (i, &pg) in b.gpas.iter().enumerate() {
                if addr >= pg && addr < pg + b.page_sz as u64 {
                    let within = (addr - pg) as usize;
                    let off = i * b.page_sz + within;
                    return Some((b.ptr, off, b.page_sz - within));
                }
            }
        }
        None
    }
}

impl HostMemory for FakeHost {
    fn read_gpa(&self, gpa: u64, buf: &mut [u8]) -> Result<(), MemError> {
        if buf.is_empty() {
            return Ok(());
        }
        let mut done = 0usize;
        while done < buf.len() {
            let addr = gpa.checked_add(done as u64).ok_or(MemError::Overflow)?;
            // Bounce views alias guest pages until unmap.
            if let Some((bptr, off, max)) = self.bounce_slot(addr) {
                let n = (buf.len() - done).min(max);
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        (bptr + off) as *const u8,
                        buf[done..].as_mut_ptr(),
                        n,
                    );
                }
                done += n;
                continue;
            }
            if let Some(i) = self.range_containing(addr) {
                let r = &self.ranges[i];
                let off = (addr - r.gpa) as usize;
                let n = (buf.len() - done).min(r.len - off);
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        (r.ptr + off) as *const u8,
                        buf[done..].as_mut_ptr(),
                        n,
                    );
                }
                done += n;
            } else {
                buf[done] = self.pages.get(&addr).copied().unwrap_or(0);
                done += 1;
            }
        }
        Ok(())
    }

    fn write_gpa(&mut self, gpa: u64, buf: &[u8]) -> Result<(), MemError> {
        if buf.is_empty() {
            return Ok(());
        }
        let mut done = 0usize;
        while done < buf.len() {
            let addr = gpa.checked_add(done as u64).ok_or(MemError::Overflow)?;
            if let Some((bptr, off, max)) = self.bounce_slot(addr) {
                let n = (buf.len() - done).min(max);
                unsafe {
                    std::ptr::copy_nonoverlapping(buf[done..].as_ptr(), (bptr + off) as *mut u8, n);
                }
                done += n;
                continue;
            }
            if let Some(i) = self.range_containing(addr) {
                let r = &self.ranges[i];
                let off = (addr - r.gpa) as usize;
                let n = (buf.len() - done).min(r.len - off);
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        buf[done..].as_ptr(),
                        (r.ptr + off) as *mut u8,
                        n,
                    );
                }
                done += n;
            } else {
                self.pages.insert(addr, buf[done]);
                done += 1;
            }
        }
        Ok(())
    }
}

impl HostOps for FakeHost {
    fn mono_ns(&self) -> u64 {
        self.mono_ns
    }

    fn enqueue(&mut self, action: HostAction) {
        // apple-gfx new_frame_handler_bh: drop/coalesce when guest ahead of
        // encode (pending_frames). Keep only the latest ScanoutUpdate pending.
        if action.kind == HostActionKind::ScanoutUpdate {
            self.actions
                .retain(|a| a.kind != HostActionKind::ScanoutUpdate);
        }
        self.actions.push(action);
    }

    fn schedule_bh(&mut self) {
        self.bh_scheduled = true;
    }

    fn read_kva(&self, kva: u64, buf: &mut [u8]) -> Result<(), MemError> {
        // Tests map "KVA" into the same sparse store as GPA.
        self.read_gpa(kva, buf)
    }

    fn read_xreg(&self, index: u32) -> Result<u64, MemError> {
        self.xregs
            .get(&index)
            .copied()
            .ok_or(MemError::XregUnavailable)
    }

    fn map_pages_stable(&self) -> bool {
        self.stable_map_pages
    }

    /// Contiguous host view; `page_size` is the guest page size from the device.
    fn map_pages(&mut self, gpas: &[u64], page_size: usize) -> Option<usize> {
        self.map_pages_calls = self.map_pages_calls.saturating_add(1);
        if gpas.is_empty() || page_size == 0 || !page_size.is_power_of_two() {
            return None;
        }
        if gpas.iter().any(|g| *g % page_size as u64 != 0) {
            return None;
        }
        if self.strict_linux_map {
            // Match reims_vgpu_pci_map_pages on EVERY host platform: a packed
            // sequential alias inside one already-provisioned RAM range only.
            // No range provisioning, no remap/bounce packing of fragmented
            // lists, and the alias is never tracked as a view (unmap of a
            // product Linux alias is a no-op).
            if gpas.iter().any(|&gpa| self.range_containing(gpa).is_none()) {
                return None;
            }
            let i = self.range_containing(gpas[0])?;
            let r = &self.ranges[i];
            let base_off = (gpas[0] - r.gpa) as usize;
            let need = gpas.len() * page_size;
            let packed = gpas
                .iter()
                .enumerate()
                .all(|(n, &gpa)| gpa == gpas[0] + (n * page_size) as u64);
            if base_off + need > r.len || !packed {
                return None;
            }
            return Some(r.ptr + base_off);
        }
        #[cfg(target_os = "macos")]
        {
            if self.stable_map_pages {
                for &gpa in gpas {
                    if self.range_containing(gpa).is_none() {
                        let _ = self.provision_range(gpa, page_size)?;
                    }
                }
                if let Some(i) = self.range_containing(gpas[0]) {
                    let r = &self.ranges[i];
                    let base_off = (gpas[0] - r.gpa) as usize;
                    let need = gpas.len() * page_size;
                    if base_off + need <= r.len {
                        let ok = gpas
                            .iter()
                            .enumerate()
                            .all(|(n, &gpa)| gpa == gpas[0] + (n * page_size) as u64);
                        if ok {
                            return Some(r.ptr + base_off);
                        }
                    }
                }
                return None;
            }
            let mut srcs = Vec::with_capacity(gpas.len());
            for &gpa in gpas {
                let idx = match self.range_containing(gpa) {
                    Some(i) => i,
                    None => self.provision_range(gpa, page_size)?,
                };
                let r = &self.ranges[idx];
                let off = (gpa - r.gpa) as usize;
                if off + page_size > r.alloc_len || !(r.ptr + off).is_multiple_of(page_size) {
                    return None;
                }
                srcs.push(r.ptr + off);
            }
            let len = gpas.len() * page_size;
            unsafe {
                let mut view = 0u64;
                if mach_vm::mach_vm_allocate(
                    mach_vm::mach_task_self_,
                    &mut view,
                    len as u64,
                    mach_vm::VM_FLAGS_ANYWHERE,
                ) != 0
                {
                    return None;
                }
                for (i, &src) in srcs.iter().enumerate() {
                    let mut dst = view + (i * page_size) as u64;
                    let (mut cur, mut max) = (0i32, 0i32);
                    if mach_vm::mach_vm_remap(
                        mach_vm::mach_task_self_,
                        &mut dst,
                        page_size as u64,
                        0,
                        mach_vm::VM_FLAGS_FIXED_OVERWRITE,
                        mach_vm::mach_task_self_,
                        src as u64,
                        0,
                        &mut cur,
                        &mut max,
                        mach_vm::VM_INHERIT_NONE,
                    ) != 0
                    {
                        mach_vm::mach_vm_deallocate(mach_vm::mach_task_self_, view, len as u64);
                        return None;
                    }
                }
                self.views.push((view as usize, len));
                Some(view as usize)
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            for &gpa in gpas {
                if self.range_containing(gpa).is_none() {
                    let _ = self.provision_range(gpa, page_size)?;
                }
            }
            // Fast path: single contiguous span in one RealRange → alias ptr.
            // Product Linux map_pages: page i at base + i*page only.
            if let Some(i) = self.range_containing(gpas[0]) {
                let r = &self.ranges[i];
                let base_off = (gpas[0] - r.gpa) as usize;
                let need = gpas.len() * page_size;
                if base_off + need <= r.len {
                    let mut ok = true;
                    for (n, &gpa) in gpas.iter().enumerate() {
                        if gpa != gpas[0] + (n * page_size) as u64 {
                            ok = false;
                            break;
                        }
                    }
                    if ok {
                        let ptr = r.ptr + base_off;
                        self.views.push((ptr, need));
                        return Some(ptr);
                    }
                }
            }
            // Scattered pages: bounce + write-back on unmap (test convenience).
            let total = gpas.len().checked_mul(page_size)?;
            let mut buf = vec![0u8; total].into_boxed_slice();
            for (i, &gpa) in gpas.iter().enumerate() {
                let off = i * page_size;
                let _ = self.read_gpa(gpa, &mut buf[off..off + page_size]);
            }
            let ptr = Box::into_raw(buf) as *mut u8 as usize;
            self.bounce.push(BounceView {
                ptr,
                len: total,
                gpas: gpas.to_vec(),
                page_sz: page_size,
            });
            self.views.push((ptr, total));
            Some(ptr)
        }
    }

    fn unmap_pages(&mut self, ptr: usize, len: usize) {
        #[cfg(target_os = "macos")]
        {
            if let Some(pos) = self.views.iter().position(|&(p, l)| p == ptr && l == len) {
                self.views.remove(pos);
                unsafe {
                    mach_vm::mach_vm_deallocate(mach_vm::mach_task_self_, ptr as u64, len as u64);
                }
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            if let Some(pos) = self
                .bounce
                .iter()
                .position(|b| b.ptr == ptr && b.len == len)
            {
                let b = self.bounce.remove(pos);
                self.views.retain(|&(p, l)| !(p == ptr && l == len));
                // Write bounce back into guest GPA store.
                // SAFETY: bounce exclusive for this view lifetime.
                let slice = unsafe { std::slice::from_raw_parts(ptr as *const u8, len) };
                for (i, &gpa) in b.gpas.iter().enumerate() {
                    let off = i * b.page_sz;
                    let _ = self.write_gpa(gpa, &slice[off..off + b.page_sz]);
                }
                unsafe {
                    let _ = Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr as *mut u8, len));
                }
            } else {
                // Aliasing view into a RealRange — drop tracking only.
                self.views.retain(|&(p, l)| !(p == ptr && l == len));
            }
        }
    }
}

/// Helpers usable with any HostMemory.
pub fn read_u32<M: HostMemory>(mem: &M, gpa: u64) -> Result<u32, MemError> {
    let mut b = [0u8; 4];
    mem.read_gpa(gpa, &mut b)?;
    Ok(u32::from_le_bytes(b))
}

/// Direct `write_gpa` u32 helper — **tests / FakeHost only**.
/// Product control-plane uses [`crate::runtime::gpa_map::write_u32`].
pub fn write_u32<M: HostMemory>(mem: &mut M, gpa: u64, v: u32) -> Result<(), MemError> {
    mem.write_gpa(gpa, &v.to_le_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Unified-memory contract: a map_pages view aliases guest RAM — writes
    /// via write_gpa are visible through the view pointer and vice versa,
    /// including scattered (non-adjacent GPA) pages.
    ///
    /// On Linux FakeHost, map_pages is optional (no remappable guest RAM
    /// allocator) — skip when unsupported.
    #[test]
    fn map_pages_view_aliases_guest_ram() {
        let mut h = FakeHost::new();
        let p = GUEST_PAGE_SIZE as u64;
        h.map_range(0x10 * p, GUEST_PAGE_SIZE, 0);
        h.map_range(0x99 * p, GUEST_PAGE_SIZE, 0);
        let Some(view) = h.map_pages(&[0x10 * p, 0x99 * p], GUEST_PAGE_SIZE) else {
            // FakeHost without contig remap: not the product QEMU path.
            return;
        };
        // write_gpa → view
        h.put_u32(0x99 * p + 8, 0xdead_beef);
        let via_view = unsafe { *((view + GUEST_PAGE_SIZE + 8) as *const u32) };
        assert_eq!(via_view, 0xdead_beef);
        // view → read_gpa
        unsafe { *((view + 4) as *mut u32) = 0x1122_3344 };
        assert_eq!(h.get_u32(0x10 * p + 4), 0x1122_3344);
        h.unmap_pages(view, 2 * GUEST_PAGE_SIZE);
    }

    #[test]
    fn fake_host_roundtrip() {
        let mut h = FakeHost::new();
        h.map_range(0x1000, 16, 0);
        h.put_u32(0x1000, 0x1122_3344);
        assert_eq!(h.get_u32(0x1000), 0x1122_3344);
        h.enqueue(HostAction::irq_gfx());
        assert_eq!(h.action_count(HostActionKind::IrqGfxPulse), 1);
    }

    /// apple-gfx pending_frames coalesce: multiple presentFrame signals before
    /// encode keep only the latest ScanoutUpdate (encode current +0x188 once).
    #[test]
    fn scanout_update_enqueue_coalesces_to_latest() {
        let mut h = FakeHost::new();
        h.enqueue(HostAction::irq_gfx());
        h.enqueue(HostAction::scanout_gen(3, 1440, 1080, 10));
        h.enqueue(HostAction::cursor(1, 2, true));
        h.enqueue(HostAction::scanout_gen(4, 1440, 1080, 11));
        assert_eq!(h.action_count(HostActionKind::IrqGfxPulse), 1);
        assert_eq!(h.action_count(HostActionKind::CursorUpdate), 1);
        assert_eq!(h.action_count(HostActionKind::ScanoutUpdate), 1);
        let scan = h
            .actions
            .iter()
            .find(|a| a.kind == HostActionKind::ScanoutUpdate)
            .expect("one ScanoutUpdate");
        assert_eq!(scan.a0, 4, "latest present mid wins");
        assert_eq!(scan.a3, 11);
    }

    /// Exactly one refusal means "the guest unmapped this".
    ///
    /// `is_guest_teardown` decides whether a deferred writeback that could not
    /// land is discharged quietly or reported as lost guest work, so widening it
    /// silences real losses and is the exception-list anti-pattern in miniature.
    /// Asserted exhaustively over every walk status and every `MemError` rather
    /// than by naming the one, so a variant added later has to be classified
    /// here on purpose instead of falling into whichever side `matches!`
    /// happens to put it.
    #[test]
    fn only_a_zero_pfn_means_the_guest_tore_the_range_down() {
        use crate::contract::gva_resolve::ResolveStatus as R;
        const WALK: &[R] = &[
            R::Ok,
            R::ErrArgs,
            R::ErrInactiveTask,
            R::ErrNoDirectory,
            R::ErrDirectoryRead,
            R::ErrZeroRootPfn,
            R::ErrZeroDepth,
            R::ErrDepthTooDeep,
            R::ErrAddressOutOfRange,
            R::ErrPageTableRead,
            R::ErrZeroPfn,
            R::ErrMalformedPte,
            R::ErrUnsupportedGeometry,
        ];
        let teardown: Vec<R> = WALK
            .iter()
            .copied()
            .filter(|r| MemError::Unresolved(*r).is_guest_teardown())
            .collect();
        assert_eq!(teardown, vec![R::ErrZeroPfn]);

        for e in [
            MemError::Unmapped,
            MemError::NoCpu,
            MemError::Overflow,
            MemError::BadArgs,
            MemError::QemuReadGpaCallbackMissing,
            MemError::QemuReadGpaCallbackFailed(-1),
            MemError::QemuWriteGpaCallbackMissing,
            MemError::QemuWriteGpaCallbackFailed(-1),
            MemError::QemuReadKvaCallbackMissing,
            MemError::QemuReadKvaCallbackFailed(-1),
            MemError::XregUnavailable,
            MemError::QemuReadXregCallbackMissing,
            MemError::QemuReadXregCallbackFailed(-1),
            MemError::NoTaskDirectory,
            MemError::UnsupportedPageShift,
            MemError::TaskRootRead,
            MemError::NoSuchTask,
            MemError::NotRam,
            MemError::MapPagesRefused,
            MemError::RunOutOfRange,
        ] {
            assert!(
                !e.is_guest_teardown(),
                "{} is not the guest saying it unmapped the range",
                crate::observe::Decline::slug(&e)
            );
        }
    }
}
