//! Backend interface and implementations.
//!
//! - This module root = backend *protocol* (Metal-semantic ops, no GPU deps).
//! - [`metal`] / [`vulkan`] = concrete backends (feature-selected), each
//!   **self-contained** in this crate (Metal via `metal`; Vulkan via `ash` +
//!   [`vulkan::engine`]).
//! - Draws + compute use [`vulkan::engine`] (self-contained ash). AIR translation
//!   comes from the pinned public `metal2vulkan` crate.
//!
//! Metal indices/semantics are canonical (guest wire is serialized Metal).
//! Vulkan-only binding rewrites live only in [`vulkan`].

use crate::runtime::plan::blit::PlannedBlit;
use crate::runtime::plan::compute::PlannedCompute;
use crate::runtime::plan::render::PlannedRender;

#[cfg(feature = "backend-metal")]
pub mod metal;
#[cfg(feature = "backend-vulkan")]
pub mod vulkan;

/// Which backend refused. Rides along on [`BackendError::Unsupported`] so a
/// single operation slug still tells you *whose* implementation declined.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendKind {
    /// The Vulkan backend's protocol-level surface (`backend/vulkan/mod.rs`).
    Vulkan,
    /// The Metal device backend (`backend/metal/device.rs`).
    Metal,
    /// The no-Metal host stub used off Apple (`backend/metal/host_stub.rs`).
    MetalHostStub,
}

impl BackendKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Vulkan => "vulkan",
            Self::Metal => "metal",
            Self::MetalHostStub => "metal_host_stub",
        }
    }
}

/// The operation a backend declined to perform.
///
/// `BackendError::Unsupported` used to be payload-free and was constructed at
/// **19 sites**, which is precisely the defect `AGENTS.md` cites as history —
/// "the 16 MiB cap returned bare `Unsupported` among six other sites, invisible
/// for a day". `DrawError` was given `Unsupported(DrawReason)` and the fix
/// stopped there; this carries it across.
///
/// The 19 sites are (operation × backend), so the slug names the **operation**
/// and [`BackendKind`] rides alongside as a field. A grep for
/// `reason=unsupported_execute_render` finds the class; `backend=` on the same
/// line says whose. Splitting into 19 literal slugs would name the same thing
/// twice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendOp {
    WriteTexture,
    ReadTexture,
    SetPipelineLibrary,
    ExecuteBlit,
    ExecuteCompute,
    ExecuteRender,
    /// A `PlannedRender::Draw` specifically, as opposed to the whole method —
    /// a backend may encode other render plans and still refuse a draw.
    RenderDraw,
    Present,
    EncodeSimpleDraw,
}

impl BackendOp {
    pub fn slug(self) -> &'static str {
        match self {
            Self::WriteTexture => "unsupported_write_texture",
            Self::ReadTexture => "unsupported_read_texture",
            Self::SetPipelineLibrary => "unsupported_set_pipeline_library",
            Self::ExecuteBlit => "unsupported_execute_blit",
            Self::ExecuteCompute => "unsupported_execute_compute",
            Self::ExecuteRender => "unsupported_execute_render",
            Self::RenderDraw => "unsupported_render_draw",
            Self::Present => "unsupported_present",
            Self::EncodeSimpleDraw => "unsupported_encode_simple_draw",
        }
    }
}

/// Backend-neutral error for semantic execution.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BackendError {
    /// This backend does not implement the named operation. Carries which
    /// operation and which backend, so the log can tell 19 sites apart.
    Unsupported(BackendOp, BackendKind),
    InvalidArgument,
    ResourceMissing,
    ShaderError,
    DeviceLost,
    Other(&'static str),
}

impl crate::observe::Decline for BackendError {
    fn slug(&self) -> &'static str {
        match self {
            Self::Unsupported(op, _) => op.slug(),
            Self::InvalidArgument => "backend_invalid_argument",
            Self::ResourceMissing => "backend_resource_missing",
            Self::ShaderError => "backend_shader_error",
            Self::DeviceLost => "backend_device_lost",
            Self::Other(_) => "backend_other",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::Unsupported(_, kind) => vec![("backend", kind.as_str().to_string())],
            Self::Other(what) => vec![("what", (*what).to_string())],
            _ => Vec::new(),
        }
    }
}

/// Minimal texture description for backend resource creation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TextureDesc {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub pixel_format: u16,
    pub mipmap_levels: u32,
    pub array_length: u32,
    pub usage: u32,
}

/// Compute dispatch parameters after planning.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ComputeDispatch {
    pub pipeline_ref: u32,
    pub grid_x: u64,
    pub grid_y: u64,
    pub grid_z: u64,
    pub threads_per_threadgroup_x: u64,
    pub threads_per_threadgroup_y: u64,
    pub threads_per_threadgroup_z: u64,
    pub threads: bool,
}

/// Render draw parameters after planning.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RenderDraw {
    pub pipeline_ref: u32,
    pub indexed: bool,
    pub count: u32,
    pub instance_count: u32,
    pub primitive_type: u32,
}

/// Semantic backend operations. Implemented by `backend::metal` / `backend::vulkan`.
pub trait Backend {
    /// Drop all state derived from the current guest lifetime.
    ///
    /// Immutable, content-keyed shader/pipeline caches may survive. Guest object
    /// identities, resident images, and aliases of guest memory must not.
    fn reset(&mut self) {}

    fn create_buffer(
        &mut self,
        ref_: u32,
        length: u64,
        bytes: Option<&[u8]>,
    ) -> Result<(), BackendError>;

    fn create_texture(&mut self, ref_: u32, desc: &TextureDesc) -> Result<(), BackendError>;

    fn write_texture(
        &mut self,
        ref_: u32,
        level: u32,
        slice: u32,
        bytes: &[u8],
        bytes_per_row: u32,
    ) -> Result<(), BackendError>;

    fn read_texture(
        &mut self,
        ref_: u32,
        level: u32,
        slice: u32,
        out: &mut [u8],
        bytes_per_row: u32,
    ) -> Result<(), BackendError>;

    fn set_pipeline_library(
        &mut self,
        pipeline_ref: u32,
        mtlb: &[u8],
        function_name: &str,
    ) -> Result<(), BackendError>;

    fn execute_blit(&mut self, plan: &PlannedBlit) -> Result<(), BackendError>;

    fn execute_compute(&mut self, plan: &PlannedCompute) -> Result<(), BackendError>;

    fn execute_render(&mut self, plan: &PlannedRender) -> Result<(), BackendError>;

    fn present(&mut self, texture_ref: u32) -> Result<(), BackendError>;

    /// Associate a texture object ref with an IOSurface mapping_id (type-11).
    fn bind_texture_mapping(&mut self, _ref_: u32, _mapping_id: u32) {}

    /// Lookup mapping_id for a texture object ref, if known.
    fn mapping_for_texture(&self, _ref_: u32) -> Option<u32> {
        None
    }
}

/// Null backend for protocol/device tests without a GPU.
#[derive(Default)]
pub struct NullBackend;

impl Backend for NullBackend {
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
        _ref_: u32,
        _level: u32,
        _slice: u32,
        _bytes: &[u8],
        _bytes_per_row: u32,
    ) -> Result<(), BackendError> {
        Ok(())
    }
    fn read_texture(
        &mut self,
        _ref_: u32,
        _level: u32,
        _slice: u32,
        _out: &mut [u8],
        _bytes_per_row: u32,
    ) -> Result<(), BackendError> {
        Ok(())
    }
    fn set_pipeline_library(
        &mut self,
        _pipeline_ref: u32,
        _mtlb: &[u8],
        _function_name: &str,
    ) -> Result<(), BackendError> {
        Ok(())
    }
    fn execute_blit(&mut self, _plan: &PlannedBlit) -> Result<(), BackendError> {
        Ok(())
    }
    fn execute_compute(&mut self, _plan: &PlannedCompute) -> Result<(), BackendError> {
        Ok(())
    }
    fn execute_render(&mut self, _plan: &PlannedRender) -> Result<(), BackendError> {
        Ok(())
    }
    fn present(&mut self, _texture_ref: u32) -> Result<(), BackendError> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn null_backend_smoke() {
        let mut b = NullBackend;
        b.create_buffer(1, 64, None).unwrap();
        b.create_texture(
            2,
            &TextureDesc {
                width: 4,
                height: 4,
                depth: 1,
                pixel_format: 0x50,
                mipmap_levels: 1,
                array_length: 1,
                usage: 0,
            },
        )
        .unwrap();
        b.present(2).unwrap();
    }
}
