//! Self-contained Vulkan execution backend (build-time alternate to Metal).
//!
//! Ownership mirrors [`crate::backend::metal`]: all host GPU work for this rail
//! lives under `backend/vulkan/`, driven by `ash`. Product draw encode uses the
//! internal [`engine`] (persistent ash context + content-keyed caches). This
//! crate has no external graphics-executor dependency; AIR translation comes
//! from the pinned public `metal2vulkan` crate.
//!
//! The [`Backend`] trait stays inert for draws; the live seam is
//! `runtime/metal_draw::try_metal2vulkan_draw` → [`engine::execute_draw`].
//!
//! [`caps`] classifies the bound host GPU into the four-cell support matrix
//! (unified/discrete memory × has/has-no DMA) that every path here must keep
//! working. Capability decisions belong there, not at call sites.
//!
//! [`translate`] is the matching seam for *state*: decoded Metal formats and
//! pipeline enums become Vulkan ones there and nowhere else, so the same
//! decision cannot be made twice with two different answers.

pub mod caps;
pub mod engine;
pub mod translate;

use crate::backend::{Backend, BackendError, BackendKind, BackendOp, TextureDesc};
use crate::runtime::plan::blit::PlannedBlit;
use crate::runtime::plan::compute::PlannedCompute;
use crate::runtime::plan::render::PlannedRender;

/// Vulkan-rail backend handle. Holds the ash-facing context when encode lands.
#[derive(Debug, Default)]
pub struct VulkanBackend {
    ready: bool,
}

impl VulkanBackend {
    pub fn new() -> Self {
        // Device/instance spin-up stays lazy until the first real encode path
        // needs it, so off-VM protocol tests can construct the shell without a
        // Vulkan ICD.
        Self { ready: true }
    }

    pub fn name(&self) -> &'static str {
        "vulkan"
    }

    pub fn ready(&self) -> bool {
        self.ready
    }
}

impl Backend for VulkanBackend {
    fn reset(&mut self) {
        engine::reset_guest_state();
    }

    fn create_buffer(
        &mut self,
        _ref_: u32,
        _length: u64,
        _bytes: Option<&[u8]>,
    ) -> Result<(), BackendError> {
        // Resource tables will live in this module; accept creates for now so
        // the device model can exercise object ids without a GPU.
        Ok(())
    }

    fn create_texture(&mut self, _ref_: u32, _desc: &TextureDesc) -> Result<(), BackendError> {
        Ok(())
    }

    fn write_texture(
        &mut self,
        _ref_: u32,
        _level: u32,
        _slice: u32,
        _bytes: &[u8],
        _bytes_per_row: u32,
    ) -> Result<(), BackendError> {
        Err(BackendError::Unsupported(
            BackendOp::WriteTexture,
            BackendKind::Vulkan,
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
            BackendKind::Vulkan,
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
            BackendKind::Vulkan,
        ))
    }

    fn execute_blit(&mut self, _plan: &PlannedBlit) -> Result<(), BackendError> {
        Err(BackendError::Unsupported(
            BackendOp::ExecuteBlit,
            BackendKind::Vulkan,
        ))
    }

    fn execute_compute(&mut self, _plan: &PlannedCompute) -> Result<(), BackendError> {
        Err(BackendError::Unsupported(
            BackendOp::ExecuteCompute,
            BackendKind::Vulkan,
        ))
    }

    fn execute_render(&mut self, _plan: &PlannedRender) -> Result<(), BackendError> {
        Err(BackendError::Unsupported(
            BackendOp::ExecuteRender,
            BackendKind::Vulkan,
        ))
    }

    fn present(&mut self, _texture_ref: u32) -> Result<(), BackendError> {
        Ok(())
    }
}

// Touch ash so the optional dependency is part of the feature graph and unused-
// crate linting does not strip it before encode modules land.
#[allow(dead_code)]
fn _ash_linkage_anchor() {
    let _ = std::mem::size_of::<ash::vk::ApplicationInfo<'static>>();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn name_is_vulkan() {
        assert_eq!(VulkanBackend::new().name(), "vulkan");
    }

    #[test]
    fn ash_is_linked() {
        _ash_linkage_anchor();
    }
}
