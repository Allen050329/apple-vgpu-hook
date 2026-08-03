//! Versioned C ABI for the QEMU thin shim.
//!
//! - opaque handles
//! - repr(C) fixed-width structures
//! - ABI version + struct-size fields
//! - explicit status codes
//! - catch_unwind on every entry
//!
//! # Safety
//! Every exported unsafe entry point is called through the matching C header.
//! Opaque device pointers must come from `reims_vgpu_device_create`; input buffers must
//! remain readable for their declared lengths; output buffers and callback
//! tables must remain writable for the duration of the call.
#![allow(
    clippy::missing_safety_doc,
    reason = "the shared QEMU C ABI safety contract is documented at module scope"
)]

use crate::qemu::host_ops::{ReimsVgpuHostAction, ReimsVgpuHostOps};
use crate::{
    backend_name, device_create, device_cursor_glyph_copy, device_cursor_glyph_info,
    device_destroy, device_drain, device_early_scanout_target, device_efi_console_copy,
    device_gfx_read, device_gfx_write, device_iosfc_read, device_iosfc_write, device_poll,
    device_pop_action, device_present_boundary_seen, device_reset, device_scanout_copy,
    device_window_run_main, device_window_set_early_fb, device_window_start, device_window_stop,
    unwind_safe, CursorGlyphInfo,
};
use std::os::raw::{c_char, c_int};
use std::slice;

/// Bump when breaking the C shim contract.
///
/// v13 adds `guest_written_pages` on [`ReimsVgpuHostOps`]: the per-page form of
/// v12's generation. A whole-set generation is enough to decide whether to reuse
/// a host-side copy, and not enough to decide what to write back — a deferred
/// writeback that discards its frame because one page moved loses the Store, and
/// one that writes the whole frame anyway loses the guest's own store.
/// v12 adds the guest-write tracking triple on [`ReimsVgpuHostOps`]:
/// `track_guest_writes`, `untrack_guest_writes`, `guest_write_gen`. A surface's
/// pages are plain guest RAM and the guest CPU stores into them with no device
/// operation, so no counter this crate keeps can witness such a store and every
/// host-side copy of those pages is stale from that instant with nothing to say
/// so. The hypervisor's dirty bitmap is the only witness; these are the door.
/// v11 adds `reims_vgpu_qemu_window_run_main`, which lets the Darwin MMIO shim make the
/// AppKit-owned winit loop QEMU's process-main UI entry.
/// v9 adds the host-window lifecycle + early framebuffer: `reims_vgpu_qemu_window_stop`
/// (close + join on VM teardown), `reims_vgpu_qemu_window_set_early_fb` (register BAR1
/// GOP so the window shows early boot), and the `WindowClosed` HostAction kind
/// (11) the window emits on a UI close so the shim requests a VM shutdown.
/// v8 adds `reims_vgpu_qemu_window_start` (host-owned presentation window; see
/// [[host-window]]). The symbol is always present; when the staticlib was built
/// without the `host-window` feature it returns `REIMS_VGPU_QEMU_ERR_STATE` so the C
/// shim falls back to QEMU's own display.
pub const REIMS_VGPU_QEMU_ABI_VERSION: u32 = 13;

#[repr(C)]
pub struct ReimsVgpuQemuCreateInfo {
    pub abi_version: u32,
    pub struct_size: u32,
    /// QEMU host-service table (GPA / clock / schedule_bh). Null for tests.
    pub host_ops: *const ReimsVgpuHostOps,
    /// Guest page shift: 12 (x86 Tahoe) or 14 (arm64e). 0 is invalid (no default).
    pub guest_page_shift: u32,
}

#[repr(C)]
pub struct ReimsVgpuQemuDevice {
    pub abi_version: u32,
    pub struct_size: u32,
    pub handle: u64,
}

pub const REIMS_VGPU_QEMU_OK: c_int = 0;
pub const REIMS_VGPU_QEMU_ERR_ARGS: c_int = 1;
pub const REIMS_VGPU_QEMU_ERR_STATE: c_int = 2;
pub const REIMS_VGPU_QEMU_ERR_PANIC: c_int = 3;
/// pop_action: queue empty (not a hard failure).
pub const REIMS_VGPU_QEMU_EMPTY: c_int = 4;

fn copy_host_ops(ops: *const ReimsVgpuHostOps) -> Option<ReimsVgpuHostOps> {
    if ops.is_null() {
        return None;
    }
    // SAFETY: QEMU passes a live ReimsVgpuHostOps for the device lifetime.
    let ops = unsafe { &*ops };
    if ops.abi_version != REIMS_VGPU_QEMU_ABI_VERSION {
        return None;
    }
    if (ops.struct_size as usize) < std::mem::size_of::<ReimsVgpuHostOps>() {
        return None;
    }
    Some(*ops)
}

/// SAFETY: `out` must be valid for write when non-null; `info` may be null (defaults).
#[no_mangle]
pub unsafe extern "C" fn reims_vgpu_qemu_device_create(
    info: *const ReimsVgpuQemuCreateInfo,
    out: *mut ReimsVgpuQemuDevice,
) -> c_int {
    unwind_safe(
        || {
            if out.is_null() {
                return REIMS_VGPU_QEMU_ERR_ARGS;
            }
            let mut ops = None;
            let mut page_shift = 0u32;
            if !info.is_null() {
                // SAFETY: caller-provided create info.
                let info = unsafe { &*info };
                if info.abi_version != REIMS_VGPU_QEMU_ABI_VERSION {
                    return REIMS_VGPU_QEMU_ERR_ARGS;
                }
                if (info.struct_size as usize) < std::mem::size_of::<ReimsVgpuQemuCreateInfo>() {
                    return REIMS_VGPU_QEMU_ERR_ARGS;
                }
                page_shift = info.guest_page_shift;
                if !info.host_ops.is_null() {
                    match copy_host_ops(info.host_ops) {
                        Some(o) => ops = Some(o),
                        None => return REIMS_VGPU_QEMU_ERR_ARGS,
                    }
                }
            }
            let handle = match device_create(ops, page_shift) {
                Some(h) => h,
                None => return REIMS_VGPU_QEMU_ERR_ARGS,
            };
            if handle == 0 {
                return REIMS_VGPU_QEMU_ERR_STATE;
            }
            // SAFETY: out is non-null.
            unsafe {
                *out = ReimsVgpuQemuDevice {
                    abi_version: REIMS_VGPU_QEMU_ABI_VERSION,
                    struct_size: std::mem::size_of::<ReimsVgpuQemuDevice>() as u32,
                    handle,
                };
            }
            REIMS_VGPU_QEMU_OK
        },
        REIMS_VGPU_QEMU_ERR_PANIC,
    )
}

/// SAFETY: handle from create; no-op if unknown.
#[no_mangle]
pub unsafe extern "C" fn reims_vgpu_qemu_device_reset(handle: u64) -> c_int {
    unwind_safe(
        || {
            if handle == 0 {
                return REIMS_VGPU_QEMU_ERR_ARGS;
            }
            if device_reset(handle) {
                REIMS_VGPU_QEMU_OK
            } else {
                REIMS_VGPU_QEMU_ERR_STATE
            }
        },
        REIMS_VGPU_QEMU_ERR_PANIC,
    )
}

/// SAFETY: handle from create; destroy is idempotent for unknown ids (ERR_STATE).
#[no_mangle]
pub unsafe extern "C" fn reims_vgpu_qemu_device_destroy(handle: u64) -> c_int {
    unwind_safe(
        || {
            if handle == 0 {
                return REIMS_VGPU_QEMU_ERR_ARGS;
            }
            if device_destroy(handle) {
                REIMS_VGPU_QEMU_OK
            } else {
                REIMS_VGPU_QEMU_ERR_STATE
            }
        },
        REIMS_VGPU_QEMU_ERR_PANIC,
    )
}

/// Start the host-owned presentation window (winit + VkSurfaceKHR) for this
/// device ([[host-window]]). Spawns the window on a dedicated thread; the drain
/// publishes each finished present frame to it, and window input (keys, pointer,
/// wheel) is injected via the neutral `Input*` prompt-action rail.
///
/// `width`/`height` seed the initial window size (0 → the boot EFI geometry).
/// Idempotent: a second call while the window is up is a no-op success.
///
/// Returns `REIMS_VGPU_QEMU_ERR_STATE` when the staticlib was built without the
/// `host-window` feature (C then leaves QEMU's own display in charge) or when
/// the handle is unknown.
///
/// SAFETY: `handle` from create.
#[no_mangle]
pub unsafe extern "C" fn reims_vgpu_qemu_window_start(
    handle: u64,
    width: u32,
    height: u32,
) -> c_int {
    unwind_safe(
        || {
            if handle == 0 {
                return REIMS_VGPU_QEMU_ERR_ARGS;
            }
            if device_window_start(handle, width, height) {
                REIMS_VGPU_QEMU_OK
            } else {
                REIMS_VGPU_QEMU_ERR_STATE
            }
        },
        REIMS_VGPU_QEMU_ERR_PANIC,
    )
}

/// Run a main-thread-owned host window until it exits.
///
/// Returns after UI close/backend stop. `REIMS_VGPU_QEMU_ERR_STATE` means this build or
/// platform has no process-main host window for `handle`.
///
/// SAFETY: `handle` from create; call on the same main thread as window start.
#[no_mangle]
pub unsafe extern "C" fn reims_vgpu_qemu_window_run_main(handle: u64) -> c_int {
    unwind_safe(
        || {
            if handle == 0 {
                return REIMS_VGPU_QEMU_ERR_ARGS;
            }
            if device_window_run_main(handle) {
                REIMS_VGPU_QEMU_OK
            } else {
                REIMS_VGPU_QEMU_ERR_STATE
            }
        },
        REIMS_VGPU_QEMU_ERR_PANIC,
    )
}

/// Stop the host-owned window and join its thread (VM teardown). Sets the stop
/// flag, the event loop exits, and the window's Vulkan objects tear down before
/// this returns — so call it before `reims_vgpu_qemu_device_destroy` and before the
/// process/driver teardown. Idempotent; `REIMS_VGPU_QEMU_OK` even with no window.
/// SAFETY: `handle` from create.
#[no_mangle]
pub unsafe extern "C" fn reims_vgpu_qemu_window_stop(handle: u64) -> c_int {
    unwind_safe(
        || {
            if handle == 0 {
                return REIMS_VGPU_QEMU_ERR_ARGS;
            }
            if device_window_stop(handle) {
                REIMS_VGPU_QEMU_OK
            } else {
                REIMS_VGPU_QEMU_ERR_STATE
            }
        },
        REIMS_VGPU_QEMU_ERR_PANIC,
    )
}

/// Register the early-boot framebuffer (BAR1 GOP host RAM) so the window shows
/// UEFI/OpenCore/boot.efi output before the product present path latches. `ptr`
/// must stay valid (and hold at least `stride * height` bytes) for the device
/// lifetime — pass the BAR1 RAMBlock host pointer. Tight BGRA8 assumed.
///
/// SAFETY: `handle` from create; `ptr` valid for `stride * height` bytes for the
/// device lifetime.
#[no_mangle]
pub unsafe extern "C" fn reims_vgpu_qemu_window_set_early_fb(
    handle: u64,
    ptr: *const u8,
    stride: u32,
    width: u32,
    height: u32,
) -> c_int {
    unwind_safe(
        || {
            if handle == 0 || ptr.is_null() {
                return REIMS_VGPU_QEMU_ERR_ARGS;
            }
            if device_window_set_early_fb(handle, ptr as usize, stride, width, height) {
                REIMS_VGPU_QEMU_OK
            } else {
                REIMS_VGPU_QEMU_ERR_STATE
            }
        },
        REIMS_VGPU_QEMU_ERR_PANIC,
    )
}

/// Write backend name into caller buffer (NUL-terminated).
/// SAFETY: buf must have buf_len bytes when non-null.
#[no_mangle]
pub unsafe extern "C" fn reims_vgpu_qemu_backend_name(buf: *mut c_char, buf_len: usize) -> c_int {
    unwind_safe(
        || {
            if buf.is_null() || buf_len == 0 {
                return REIMS_VGPU_QEMU_ERR_ARGS;
            }
            let name = backend_name().as_bytes();
            let n = name.len().min(buf_len - 1);
            // SAFETY: buf valid for buf_len.
            unsafe {
                std::ptr::copy_nonoverlapping(name.as_ptr(), buf as *mut u8, n);
                *buf.add(n) = 0;
            }
            REIMS_VGPU_QEMU_OK
        },
        REIMS_VGPU_QEMU_ERR_PANIC,
    )
}

/// ABI version getter (no allocation).
#[no_mangle]
pub unsafe extern "C" fn reims_vgpu_qemu_abi_version() -> u32 {
    unwind_safe(|| REIMS_VGPU_QEMU_ABI_VERSION, 0)
}

/// Gfx MMIO read. SAFETY: out_val non-null.
#[no_mangle]
pub unsafe extern "C" fn reims_vgpu_qemu_gfx_read(
    handle: u64,
    offset: u64,
    size: u32,
    out_val: *mut u64,
) -> c_int {
    unwind_safe(
        || {
            if handle == 0 || out_val.is_null() {
                return REIMS_VGPU_QEMU_ERR_ARGS;
            }
            match device_gfx_read(handle, offset, size) {
                Some(v) => {
                    // SAFETY: out_val non-null.
                    unsafe {
                        *out_val = v;
                    }
                    REIMS_VGPU_QEMU_OK
                }
                None => REIMS_VGPU_QEMU_ERR_STATE,
            }
        },
        REIMS_VGPU_QEMU_ERR_PANIC,
    )
}

/// Gfx MMIO write (may schedule QEMU BH via HostOps).
#[no_mangle]
pub unsafe extern "C" fn reims_vgpu_qemu_gfx_write(
    handle: u64,
    offset: u64,
    data: u64,
    size: u32,
) -> c_int {
    unwind_safe(
        || {
            if handle == 0 {
                return REIMS_VGPU_QEMU_ERR_ARGS;
            }
            if device_gfx_write(handle, offset, data, size) {
                REIMS_VGPU_QEMU_OK
            } else {
                REIMS_VGPU_QEMU_ERR_STATE
            }
        },
        REIMS_VGPU_QEMU_ERR_PANIC,
    )
}

/// Iosfc MMIO read. SAFETY: out_val non-null.
#[no_mangle]
pub unsafe extern "C" fn reims_vgpu_qemu_iosfc_read(
    handle: u64,
    offset: u64,
    size: u32,
    out_val: *mut u64,
) -> c_int {
    unwind_safe(
        || {
            if handle == 0 || out_val.is_null() {
                return REIMS_VGPU_QEMU_ERR_ARGS;
            }
            match device_iosfc_read(handle, offset, size) {
                Some(v) => {
                    unsafe {
                        *out_val = v;
                    }
                    REIMS_VGPU_QEMU_OK
                }
                None => REIMS_VGPU_QEMU_ERR_STATE,
            }
        },
        REIMS_VGPU_QEMU_ERR_PANIC,
    )
}

/// Iosfc MMIO write (may schedule QEMU BH via HostOps).
#[no_mangle]
pub unsafe extern "C" fn reims_vgpu_qemu_iosfc_write(
    handle: u64,
    offset: u64,
    data: u64,
    size: u32,
) -> c_int {
    unwind_safe(
        || {
            if handle == 0 {
                return REIMS_VGPU_QEMU_ERR_ARGS;
            }
            if device_iosfc_write(handle, offset, data, size) {
                REIMS_VGPU_QEMU_OK
            } else {
                REIMS_VGPU_QEMU_ERR_STATE
            }
        },
        REIMS_VGPU_QEMU_ERR_PANIC,
    )
}

/// BH body: drain pending FIFOs (GPA via HostOps). Then pop actions with
/// [`reims_vgpu_qemu_device_pop_action`].
#[no_mangle]
pub unsafe extern "C" fn reims_vgpu_qemu_device_drain(handle: u64) -> c_int {
    unwind_safe(
        || {
            if handle == 0 {
                return REIMS_VGPU_QEMU_ERR_ARGS;
            }
            if device_drain(handle) {
                REIMS_VGPU_QEMU_OK
            } else {
                REIMS_VGPU_QEMU_ERR_STATE
            }
        },
        REIMS_VGPU_QEMU_ERR_PANIC,
    )
}

/// Periodic poll (gfx_update): display ONLINE re-drive after guest enable().
/// Deliver HostActions with [`reims_vgpu_qemu_device_pop_action`] after this call.
#[no_mangle]
pub unsafe extern "C" fn reims_vgpu_qemu_device_poll(handle: u64) -> c_int {
    unwind_safe(
        || {
            if handle == 0 {
                return REIMS_VGPU_QEMU_ERR_ARGS;
            }
            if device_poll(handle) {
                REIMS_VGPU_QEMU_OK
            } else {
                REIMS_VGPU_QEMU_ERR_STATE
            }
        },
        REIMS_VGPU_QEMU_ERR_PANIC,
    )
}

/// Pop one HostAction for the QEMU BH. Returns REIMS_VGPU_QEMU_OK with *out filled,
/// REIMS_VGPU_QEMU_EMPTY when the queue is empty.
/// SAFETY: out non-null when a value is expected.
#[no_mangle]
pub unsafe extern "C" fn reims_vgpu_qemu_device_pop_action(
    handle: u64,
    out: *mut ReimsVgpuHostAction,
) -> c_int {
    unwind_safe(
        || {
            if handle == 0 || out.is_null() {
                return REIMS_VGPU_QEMU_ERR_ARGS;
            }
            match device_pop_action(handle) {
                Some(a) => {
                    // SAFETY: out non-null.
                    unsafe {
                        *out = a;
                    }
                    REIMS_VGPU_QEMU_OK
                }
                None => REIMS_VGPU_QEMU_EMPTY,
            }
        },
        REIMS_VGPU_QEMU_ERR_PANIC,
    )
}

/// Whether the guest has crossed the first product present boundary
/// (`frame_flush_seen`). REIMS_VGPU_QEMU_OK + `*out_seen` 0/1; ERR_STATE if no device.
/// The per-boot value is monotonic and does not contend on mutable device state.
///
/// C uses this so BAR1 UEFI GOP stays on the host console until DisplaySwap —
/// not until the first early front writeback (which was killing GOP too early).
///
/// SAFETY: `out_seen` non-null.
#[no_mangle]
pub unsafe extern "C" fn reims_vgpu_qemu_present_boundary_seen(
    handle: u64,
    out_seen: *mut u32,
) -> c_int {
    unwind_safe(
        || {
            if handle == 0 || out_seen.is_null() {
                return REIMS_VGPU_QEMU_ERR_ARGS;
            }
            match device_present_boundary_seen(handle) {
                Some(seen) => {
                    unsafe {
                        *out_seen = if seen { 1 } else { 0 };
                    }
                    REIMS_VGPU_QEMU_OK
                }
                None => REIMS_VGPU_QEMU_ERR_STATE,
            }
        },
        REIMS_VGPU_QEMU_ERR_PANIC,
    )
}

/// Pre-boundary early console: copy guest EFI FB (MMIO 0x1210) into `dst`.
///
/// REIMS_VGPU_QEMU_OK when efi_fb_start is programmed and GPA read succeeds.
/// REIMS_VGPU_QEMU_EMPTY when efi_fb_start == 0 (C falls back to BAR1 GOP RAM).
///
/// SAFETY: `dst` valid for dst_stride*height.
#[no_mangle]
pub unsafe extern "C" fn reims_vgpu_qemu_efi_console_copy(
    handle: u64,
    dst: *mut u8,
    dst_stride: u32,
    width: u32,
    height: u32,
) -> c_int {
    unwind_safe(
        || {
            if handle == 0 || dst.is_null() || width == 0 || height == 0 {
                return REIMS_VGPU_QEMU_ERR_ARGS;
            }
            let len = (dst_stride as usize).saturating_mul(height as usize);
            if len == 0 {
                return REIMS_VGPU_QEMU_ERR_ARGS;
            }
            let buf = unsafe { slice::from_raw_parts_mut(dst, len) };
            if device_efi_console_copy(handle, buf, dst_stride, width, height) {
                REIMS_VGPU_QEMU_OK
            } else {
                REIMS_VGPU_QEMU_EMPTY
            }
        },
        REIMS_VGPU_QEMU_ERR_PANIC,
    )
}

/// Pre-boundary scanout target for `gfx_update` re-pull (logo + pill).
///
/// REIMS_VGPU_QEMU_OK fills all outs; REIMS_VGPU_QEMU_EMPTY when post-boundary or no front
/// mapping yet (C should only re-show the last surface).
///
/// SAFETY: out pointers non-null when OK is expected.
#[no_mangle]
pub unsafe extern "C" fn reims_vgpu_qemu_early_scanout_target(
    handle: u64,
    out_mapping_id: *mut u32,
    out_width: *mut u32,
    out_height: *mut u32,
    out_generation: *mut u32,
) -> c_int {
    unwind_safe(
        || {
            if handle == 0
                || out_mapping_id.is_null()
                || out_width.is_null()
                || out_height.is_null()
                || out_generation.is_null()
            {
                return REIMS_VGPU_QEMU_ERR_ARGS;
            }
            match device_early_scanout_target(handle) {
                Some((mid, w, h, gen)) => {
                    unsafe {
                        *out_mapping_id = mid;
                        *out_width = w;
                        *out_height = h;
                        *out_generation = gen;
                    }
                    REIMS_VGPU_QEMU_OK
                }
                None => REIMS_VGPU_QEMU_EMPTY,
            }
        },
        REIMS_VGPU_QEMU_ERR_PANIC,
    )
}

/// Fill a host BGRA8 framebuffer (QEMU DisplaySurface) from a guest mapping.
///
/// `generation` is HostAction.a3 (0 = always paint). Returns REIMS_VGPU_QEMU_EMPTY when
/// content is unchanged (C should skip console update).
///
/// SAFETY: `dst` must be valid for `dst_stride * height` bytes.
#[no_mangle]
pub unsafe extern "C" fn reims_vgpu_qemu_scanout_copy(
    handle: u64,
    mapping_id: u32,
    dst: *mut u8,
    dst_stride: u32,
    width: u32,
    height: u32,
    generation: u32,
) -> c_int {
    use crate::runtime::scanout::ScanoutCopyResult;
    unwind_safe(
        || {
            if handle == 0 || dst.is_null() || width == 0 || height == 0 || dst_stride == 0 {
                return REIMS_VGPU_QEMU_ERR_ARGS;
            }
            let nbytes = (height as usize).saturating_mul(dst_stride as usize);
            if nbytes == 0 {
                return REIMS_VGPU_QEMU_ERR_ARGS;
            }
            // SAFETY: caller owns DisplaySurface buffer for nbytes.
            let buf = unsafe { slice::from_raw_parts_mut(dst, nbytes) };
            match device_scanout_copy(
                handle, mapping_id, buf, dst_stride, width, height, generation,
            ) {
                ScanoutCopyResult::Painted => REIMS_VGPU_QEMU_OK,
                ScanoutCopyResult::Unchanged => REIMS_VGPU_QEMU_EMPTY,
                ScanoutCopyResult::Failed => REIMS_VGPU_QEMU_ERR_STATE,
            }
        },
        REIMS_VGPU_QEMU_ERR_PANIC,
    )
}

/// Cursor glyph geometry. Returns REIMS_VGPU_QEMU_EMPTY when no glyph is ready.
#[no_mangle]
pub unsafe extern "C" fn reims_vgpu_qemu_cursor_glyph_info(
    handle: u64,
    out: *mut CursorGlyphInfo,
) -> c_int {
    unwind_safe(
        || {
            if handle == 0 || out.is_null() {
                return REIMS_VGPU_QEMU_ERR_ARGS;
            }
            match device_cursor_glyph_info(handle) {
                Some(info) => {
                    unsafe {
                        *out = info;
                    }
                    REIMS_VGPU_QEMU_OK
                }
                None => REIMS_VGPU_QEMU_EMPTY,
            }
        },
        REIMS_VGPU_QEMU_ERR_PANIC,
    )
}

/// Copy glyph pixels as QEMUCursor ARGB (`0xAARRGGBB`). `count` is capacity in
/// u32 pixels; on success writes min(count, pixel_count) and returns OK.
#[no_mangle]
pub unsafe extern "C" fn reims_vgpu_qemu_cursor_glyph_copy(
    handle: u64,
    out_argb: *mut u32,
    count: usize,
) -> c_int {
    unwind_safe(
        || {
            if handle == 0 || out_argb.is_null() || count == 0 {
                return REIMS_VGPU_QEMU_ERR_ARGS;
            }
            // SAFETY: caller buffer for count u32s.
            let buf = unsafe { slice::from_raw_parts_mut(out_argb, count) };
            match device_cursor_glyph_copy(handle, buf) {
                Some(n) if n > 0 => REIMS_VGPU_QEMU_OK,
                _ => REIMS_VGPU_QEMU_EMPTY,
            }
        },
        REIMS_VGPU_QEMU_ERR_PANIC,
    )
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::model::{PAGE_SHIFT_ARM64E, PAGE_SHIFT_X86};

    #[test]
    fn create_reset_destroy() {
        let mut dev = ReimsVgpuQemuDevice {
            abi_version: 0,
            struct_size: 0,
            handle: 0,
        };
        let info = ReimsVgpuQemuCreateInfo {
            abi_version: REIMS_VGPU_QEMU_ABI_VERSION,
            struct_size: std::mem::size_of::<ReimsVgpuQemuCreateInfo>() as u32,
            host_ops: std::ptr::null(),
            guest_page_shift: PAGE_SHIFT_ARM64E, // arm64e — must choose 12 or 14 explicitly
        };
        let rc = unsafe { reims_vgpu_qemu_device_create(&info, &mut dev) };
        assert_eq!(rc, REIMS_VGPU_QEMU_OK);
        assert_ne!(dev.handle, 0);
        assert_eq!(
            unsafe { reims_vgpu_qemu_device_reset(dev.handle) },
            REIMS_VGPU_QEMU_OK
        );
        assert_eq!(
            unsafe { reims_vgpu_qemu_device_destroy(dev.handle) },
            REIMS_VGPU_QEMU_OK
        );
    }

    #[test]
    fn create_rejects_zero_page_shift() {
        let mut dev = ReimsVgpuQemuDevice {
            abi_version: 0,
            struct_size: 0,
            handle: 0,
        };
        let info = ReimsVgpuQemuCreateInfo {
            abi_version: REIMS_VGPU_QEMU_ABI_VERSION,
            struct_size: std::mem::size_of::<ReimsVgpuQemuCreateInfo>() as u32,
            host_ops: std::ptr::null(),
            guest_page_shift: 0,
        };
        let rc = unsafe { reims_vgpu_qemu_device_create(&info, &mut dev) };
        assert_eq!(rc, REIMS_VGPU_QEMU_ERR_ARGS);
    }

    #[test]
    fn backend_name_metal_default() {
        let mut buf = [0i8; 32];
        assert_eq!(
            unsafe { reims_vgpu_qemu_backend_name(buf.as_mut_ptr(), buf.len()) },
            REIMS_VGPU_QEMU_OK
        );
        let s = unsafe { std::ffi::CStr::from_ptr(buf.as_ptr()) }
            .to_str()
            .unwrap();
        assert!(s == "metal" || s == "vulkan", "got {s}");
    }

    #[test]
    fn mmio_version_roundtrip() {
        let mut dev = ReimsVgpuQemuDevice {
            abi_version: 0,
            struct_size: 0,
            handle: 0,
        };
        let info = ReimsVgpuQemuCreateInfo {
            abi_version: REIMS_VGPU_QEMU_ABI_VERSION,
            struct_size: std::mem::size_of::<ReimsVgpuQemuCreateInfo>() as u32,
            host_ops: std::ptr::null(),
            guest_page_shift: PAGE_SHIFT_X86,
        };
        assert_eq!(
            unsafe { reims_vgpu_qemu_device_create(&info, &mut dev) },
            REIMS_VGPU_QEMU_OK
        );
        // 0x1034 version handshake
        assert_eq!(
            unsafe { reims_vgpu_qemu_gfx_write(dev.handle, 0x1034, 0x3e, 4) },
            REIMS_VGPU_QEMU_OK
        );
        let mut val = 0u64;
        assert_eq!(
            unsafe { reims_vgpu_qemu_gfx_read(dev.handle, 0x1034, 4, &mut val) },
            REIMS_VGPU_QEMU_OK
        );
        assert_eq!(val, 0x3e);
        assert_eq!(
            unsafe { reims_vgpu_qemu_device_destroy(dev.handle) },
            REIMS_VGPU_QEMU_OK
        );
    }

    #[test]
    fn drain_pop_empty() {
        let mut dev = ReimsVgpuQemuDevice {
            abi_version: 0,
            struct_size: 0,
            handle: 0,
        };
        let info = ReimsVgpuQemuCreateInfo {
            abi_version: REIMS_VGPU_QEMU_ABI_VERSION,
            struct_size: std::mem::size_of::<ReimsVgpuQemuCreateInfo>() as u32,
            host_ops: std::ptr::null(),
            guest_page_shift: PAGE_SHIFT_X86, // x86 Tahoe
        };
        assert_eq!(
            unsafe { reims_vgpu_qemu_device_create(&info, &mut dev) },
            REIMS_VGPU_QEMU_OK
        );
        assert_eq!(
            unsafe { reims_vgpu_qemu_device_drain(dev.handle) },
            REIMS_VGPU_QEMU_OK
        );
        let mut action = ReimsVgpuHostAction::default();
        assert_eq!(
            unsafe { reims_vgpu_qemu_device_pop_action(dev.handle, &mut action) },
            REIMS_VGPU_QEMU_EMPTY
        );
        assert_eq!(
            unsafe { reims_vgpu_qemu_device_destroy(dev.handle) },
            REIMS_VGPU_QEMU_OK
        );
    }
}
