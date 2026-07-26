//! MTLDevice probe helpers + Backend trait for device lifecycle.

use crate::backend::metal::runtime::{system_device, system_device_name as runtime_device_name};
use crate::backend::{Backend, BackendError, BackendKind, BackendOp, TextureDesc};
use crate::runtime::plan::blit::PlannedBlit;
use crate::runtime::plan::compute::PlannedCompute;
use crate::runtime::plan::render::PlannedRender;
use metal::Device;

/// Runtime handle for pure-Rust Metal probes.
pub struct MetalRuntime;

impl MetalRuntime {
    pub fn device() -> Option<&'static Device> {
        system_device()
    }
}

/// Device lifecycle handle; product encode is the C ABI in `ffi`.
///
/// Texture ref → IOSurface mapping_id associations are filled by the runtime
/// object-list path so writeback can call [`crate::runtime::mapping_write`].
#[derive(Debug, Default)]
pub struct MetalBackend {
    ready: bool,
    /// Last type-11 mapping touched by a texture create (for writeback hooks).
    pub last_mapping_id: u32,
    /// Texture object refs known to be type-11 (ref → mapping_id).
    pub texture_mappings: std::collections::BTreeMap<u32, u32>,
}

impl MetalBackend {
    pub fn new() -> Self {
        Self {
            ready: system_device().is_some(),
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

    pub fn reset_caches(&mut self) {
        crate::backend::metal::cache::cache_stats_reset();
    }

    pub fn reset(&mut self) {
        self.reset_caches();
        self.texture_mappings.clear();
        self.last_mapping_id = 0;
    }

    /// Encode a simple draw with vert/frag MTLB into `out_rgba` (RGBA8).
    ///
    /// Used by the device exec path when pipeline function MTLBs resolve.
    pub fn encode_simple_draw(
        &self,
        vert_mtlb: &[u8],
        frag_mtlb: &[u8],
        width: u32,
        height: u32,
        vertex_count: usize,
        first_vertex: usize,
        instance_count: usize,
        primitive_type: u32,
        color_pixel_format: u32,
        target_seed_rgba: Option<&[u8]>,
        out_rgba: &mut [u8],
    ) -> Result<(), BackendError> {
        if !self.ready {
            return Err(BackendError::DeviceLost);
        }
        let mut err_buf = [0i8; 256];
        let err = (err_buf.as_mut_ptr(), err_buf.len());
        let st = crate::backend::metal::render::render_core(
            vert_mtlb,
            frag_mtlb,
            width,
            height,
            vertex_count,
            first_vertex,
            instance_count.max(1),
            0,
            primitive_type,
            None,
            None,
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            color_pixel_format,
            target_seed_rgba,
            Some(out_rgba),
            err,
        );
        if st.is_ok() {
            Ok(())
        } else {
            Err(BackendError::ShaderError)
        }
    }
}

impl Backend for MetalBackend {
    fn reset(&mut self) {
        MetalBackend::reset(self);
        crate::runtime::icb::clear_icb_cache();
        crate::backend::metal::runtime::type11_guest_texture_invalidate_all();
    }

    fn create_buffer(
        &mut self,
        _ref_: u32,
        _length: u64,
        _bytes: Option<&[u8]>,
    ) -> Result<(), BackendError> {
        if !self.ready {
            return Err(BackendError::DeviceLost);
        }
        Ok(())
    }

    fn create_texture(&mut self, _ref_: u32, _desc: &TextureDesc) -> Result<(), BackendError> {
        if !self.ready {
            return Err(BackendError::DeviceLost);
        }
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
        if !self.ready {
            return Err(BackendError::DeviceLost);
        }
        // Encode path not fully wired: remember association for runtime writeback.
        // When `ref_` is a known type-11, callers use mapping_write + mark_written.
        if let Some(m) = self.mapping_for_texture(ref_) {
            self.last_mapping_id = m;
        }
        if bytes.is_empty() || bytes_per_row == 0 {
            return Err(BackendError::InvalidArgument);
        }
        // Real MTL texture upload still in ffi encode path; surface writeback
        // is owned by runtime::mapping_write after decode/exec.
        Err(BackendError::Unsupported(
            BackendOp::WriteTexture,
            BackendKind::Metal,
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
            BackendKind::Metal,
        ))
    }

    fn set_pipeline_library(
        &mut self,
        _pipeline_ref: u32,
        mtlb: &[u8],
        _function_name: &str,
    ) -> Result<(), BackendError> {
        let device = system_device().ok_or(BackendError::DeviceLost)?;
        let library = device
            .new_library_with_data(mtlb)
            .map_err(|_| BackendError::ShaderError)?;
        if library.function_names().len() != 1 {
            return Err(BackendError::ShaderError);
        }
        Ok(())
    }

    fn execute_blit(&mut self, _plan: &PlannedBlit) -> Result<(), BackendError> {
        Err(BackendError::Unsupported(
            BackendOp::ExecuteBlit,
            BackendKind::Metal,
        ))
    }

    fn execute_compute(&mut self, _plan: &PlannedCompute) -> Result<(), BackendError> {
        Err(BackendError::Unsupported(
            BackendOp::ExecuteCompute,
            BackendKind::Metal,
        ))
    }

    fn execute_render(&mut self, plan: &PlannedRender) -> Result<(), BackendError> {
        if !self.ready {
            return Err(BackendError::DeviceLost);
        }
        // Full MTL encode is still ffi/async work. Clear-only passes are applied
        // in runtime::exec via mapping_write (no GPU). Draws remain Unsupported
        // until the MTLB pipeline path is wired through this trait.
        match plan {
            PlannedRender::Draw { .. } => Err(BackendError::Unsupported(
                BackendOp::RenderDraw,
                BackendKind::Metal,
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
        Err(BackendError::Unsupported(
            BackendOp::Present,
            BackendKind::Metal,
        ))
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
    runtime_device_name()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn system_device() {
        assert!(MetalRuntime::device().is_some());
        assert!(system_device_name().is_some());
        assert!(MetalBackend::new().ready());
    }
}
