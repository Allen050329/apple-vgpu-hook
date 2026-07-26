//! Non-Apple host stub for the Metal backend.
//!
//! Same public types as the real path (`MetalBackend`, ABI constants, cache
//! invalidate hooks) so protocol/runtime/QEMU can link `backend-metal` on
//! Linux/x86 while encode stays fail-closed. Replace with real MTL encode when
//! the host has Metal (or a future non-Apple Metal-compatible rail).

use crate::backend::{Backend, BackendError, BackendKind, BackendOp, TextureDesc};
use crate::runtime::plan::blit::PlannedBlit;
use crate::runtime::plan::compute::PlannedCompute;
use crate::runtime::plan::render::PlannedRender;

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
    pub last_mapping_id: u32,
    pub texture_mappings: std::collections::BTreeMap<u32, u32>,
}

impl MetalBackend {
    pub fn new() -> Self {
        Self {
            // Ready for protocol object bookkeeping; GPU encode remains Unsupported.
            ready: true,
            last_mapping_id: 0,
            texture_mappings: std::collections::BTreeMap::new(),
        }
    }

    pub fn ready(&self) -> bool {
        self.ready
    }

    pub fn name(&self) -> &'static str {
        "metal"
    }

    pub fn reset_caches(&mut self) {}

    pub fn reset(&mut self) {
        self.texture_mappings.clear();
        self.last_mapping_id = 0;
    }

    pub fn encode_simple_draw(
        &self,
        _vert_mtlb: &[u8],
        _frag_mtlb: &[u8],
        _width: u32,
        _height: u32,
        _vertex_count: usize,
        _first_vertex: usize,
        _instance_count: usize,
        _primitive_type: u32,
        _color_pixel_format: u32,
        _target_seed_rgba: Option<&[u8]>,
        _out_rgba: &mut [u8],
    ) -> Result<(), BackendError> {
        Err(BackendError::Unsupported(
            BackendOp::EncodeSimpleDraw,
            BackendKind::MetalHostStub,
        ))
    }
}

impl Backend for MetalBackend {
    fn reset(&mut self) {
        MetalBackend::reset(self);
        crate::runtime::icb::clear_icb_cache();
        runtime::type11_guest_texture_invalidate_all();
    }

    fn create_buffer(
        &mut self,
        _ref_: u32,
        _length: u64,
        _bytes: Option<&[u8]>,
    ) -> Result<(), BackendError> {
        Ok(())
    }

    fn create_texture(&mut self, _ref_: u32, _desc: &TextureDesc) -> Result<(), BackendError> {
        Ok(())
    }

    fn write_texture(
        &mut self,
        ref_: u32,
        _level: u32,
        _slice: u32,
        bytes: &[u8],
        bytes_per_row: u32,
    ) -> Result<(), BackendError> {
        if let Some(m) = self.mapping_for_texture(ref_) {
            self.last_mapping_id = m;
        }
        if bytes.is_empty() || bytes_per_row == 0 {
            return Err(BackendError::InvalidArgument);
        }
        Err(BackendError::Unsupported(
            BackendOp::WriteTexture,
            BackendKind::MetalHostStub,
        ))
    }

    fn read_texture(
        &mut self,
        _ref_: u32,
        _level: u32,
        _slice: u32,
        _out: &mut [u8],
        _bytes_per_row: u32,
    ) -> Result<(), BackendError> {
        Err(BackendError::Unsupported(
            BackendOp::ReadTexture,
            BackendKind::MetalHostStub,
        ))
    }

    fn set_pipeline_library(
        &mut self,
        _pipeline_ref: u32,
        _mtlb: &[u8],
        _function_name: &str,
    ) -> Result<(), BackendError> {
        Err(BackendError::Unsupported(
            BackendOp::SetPipelineLibrary,
            BackendKind::MetalHostStub,
        ))
    }

    fn execute_blit(&mut self, _plan: &PlannedBlit) -> Result<(), BackendError> {
        Err(BackendError::Unsupported(
            BackendOp::ExecuteBlit,
            BackendKind::MetalHostStub,
        ))
    }

    fn execute_compute(&mut self, _plan: &PlannedCompute) -> Result<(), BackendError> {
        Err(BackendError::Unsupported(
            BackendOp::ExecuteCompute,
            BackendKind::MetalHostStub,
        ))
    }

    fn execute_render(&mut self, plan: &PlannedRender) -> Result<(), BackendError> {
        match plan {
            PlannedRender::Draw { .. } => Err(BackendError::Unsupported(
                BackendOp::RenderDraw,
                BackendKind::MetalHostStub,
            )),
            PlannedRender::SetPipeline { .. }
            | PlannedRender::SetBuffer { .. }
            | PlannedRender::SetTexture { .. }
            | PlannedRender::SetViewport { .. }
            | PlannedRender::SetScissor { .. }
            | PlannedRender::Fence { .. }
            | PlannedRender::Other { .. } => Ok(()),
        }
    }

    fn present(&mut self, _texture_ref: u32) -> Result<(), BackendError> {
        Ok(())
    }

    fn bind_texture_mapping(&mut self, ref_: u32, mapping_id: u32) {
        if ref_ != 0 && mapping_id != 0 {
            self.texture_mappings.insert(ref_, mapping_id);
            self.last_mapping_id = mapping_id;
        }
    }

    fn mapping_for_texture(&self, ref_: u32) -> Option<u32> {
        self.texture_mappings.get(&ref_).copied()
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
