//! Non-Apple host stub for the Metal backend.
//!
//! Same public types as the real path (`MetalBackend`, ABI constants, cache
//! invalidate hooks) so protocol/runtime/QEMU can link `backend-metal` on
//! Linux/x86 while encode stays fail-closed. Replace with real MTL encode when
//! the host has Metal (or a future non-Apple Metal-compatible rail).

use crate::backend::Backend;

/// Runtime probe handle — no MTLDevice on non-Apple hosts.
pub struct MetalRuntime;

impl MetalRuntime {
    pub fn device() -> Option<()> {
        None
    }
}

/// Device lifecycle handle; encode paths are stubs until Metal is available.
#[derive(Debug, Default)]
pub struct MetalBackend {
    ready: bool,
}

impl MetalBackend {
    pub fn new() -> Self {
        Self {
            // Ready for protocol object bookkeeping; there is no GPU encode here.
            ready: true,
        }
    }

    pub fn ready(&self) -> bool {
        self.ready
    }

    pub fn name(&self) -> &'static str {
        "metal"
    }

    pub fn reset(&mut self) {}
}

impl Backend for MetalBackend {
    fn reset(&mut self) {
        MetalBackend::reset(self);
        crate::runtime::icb::clear_icb_cache();
        runtime::type11_guest_texture_invalidate_all();
    }
}

pub fn system_device_name() -> Option<String> {
    None
}

/// Runtime hooks used by model/mapper on type-11 lifecycle.
pub mod runtime {
    /// Drop any host-side type-11 texture cache entry (no-op on stub host).
    pub fn type11_guest_texture_invalidate(_mapping_id: u32) {}

    /// Drop all host-side type-11 texture cache entries (no-op on stub host).
    pub fn type11_guest_texture_invalidate_all() {}
}

/// C ABI surface expected by tests; all entry points fail closed.
pub mod c_abi {
    use super::super::abi::*;
    use std::os::raw::c_char;

    #[no_mangle]
    pub extern "C" fn reims_vgpu_backend_begin_native_color_format(_pixel_format: u32) {}

    #[no_mangle]
    pub extern "C" fn reims_vgpu_backend_end_native_color_format() {}

    #[no_mangle]
    pub extern "C" fn reims_vgpu_backend_metal_cache_stats(out: *mut ReimsVgpuMetalCacheStats) {
        if !out.is_null() {
            unsafe {
                *out = std::mem::zeroed();
            }
        }
    }

    #[no_mangle]
    pub extern "C" fn reims_vgpu_backend_metal_cache_stats_reset() {}

    #[no_mangle]
    pub extern "C" fn reims_vgpu_backend_dispatch_compute_mtlb(
        _mtlb: *const u8,
        _mtlb_len: usize,
        _buffers: *mut ReimsVgpuBuffer,
        _buffer_count: usize,
        _grid_x: u32,
        err: *mut c_char,
        err_cap: usize,
    ) -> i32 {
        if !err.is_null() && err_cap > 0 {
            let msg = b"host metal stub\0";
            let n = msg.len().min(err_cap);
            unsafe {
                std::ptr::copy_nonoverlapping(msg.as_ptr() as *const c_char, err, n);
            }
        }
        REIMS_VGPU_ERR_EXECUTE
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stub_backend_name() {
        assert_eq!(MetalBackend::new().name(), "metal");
        assert!(MetalBackend::new().ready());
        assert!(MetalRuntime::device().is_none());
    }
}
