//! Caps matching `reims_vgpu_backend_metal.m`.

pub const REIMS_VGPU_METAL_MAX_ATTRS: usize = 31;
pub const REIMS_VGPU_METAL_MAX_BUFFERS: usize = 31;
pub const REIMS_VGPU_METAL_MAX_TEXTURES: usize = 31;
pub const REIMS_VGPU_METAL_MAX_SAMPLERS: usize = 16;
/// Metal max color attachments per render pass / PSO.
pub const REIMS_VGPU_METAL_MAX_COLOR_RTS: usize = 8;

pub const REIMS_VGPU_FN_CACHE_CAP: usize = 96;
pub const REIMS_VGPU_RENDER_PSO_CACHE_CAP: usize = 64;
pub const REIMS_VGPU_COMPUTE_PSO_CACHE_CAP: usize = 64;
pub const REIMS_VGPU_SAMPLER_CACHE_CAP: usize = 32;
pub const REIMS_VGPU_DEPTH_STENCIL_CACHE_CAP: usize = 16;
pub const REIMS_VGPU_COMPUTE_REFLECT_CACHE_CAP: usize = 64;

/// Metal `MTLBufferLayoutStrideDynamic` == `NSUIntegerMax`.
pub const MTL_BUFFER_LAYOUT_STRIDE_DYNAMIC: u64 = u64::MAX;

/// Upper bound used when validating attribute formats (`MTLAttributeFormatFloatRGB9E5`).
pub const MTL_ATTRIBUTE_FORMAT_FLOAT_RGB9E5: u32 = 54;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::metal::abi::{
        REIMS_VGPU_BINDING_SAMPLER_BASE, REIMS_VGPU_BINDING_TEXTURE_BASE,
    };

    #[test]
    fn resource_binding_bands_do_not_overlap() {
        assert!(REIMS_VGPU_METAL_MAX_BUFFERS as u32 <= REIMS_VGPU_BINDING_TEXTURE_BASE);
        assert!(
            REIMS_VGPU_BINDING_TEXTURE_BASE + REIMS_VGPU_METAL_MAX_TEXTURES as u32
                <= REIMS_VGPU_BINDING_SAMPLER_BASE
        );
    }

    #[test]
    fn fixed_cache_caps_are_nonzero_and_cover_each_cache_family() {
        for cap in [
            REIMS_VGPU_FN_CACHE_CAP,
            REIMS_VGPU_RENDER_PSO_CACHE_CAP,
            REIMS_VGPU_COMPUTE_PSO_CACHE_CAP,
            REIMS_VGPU_SAMPLER_CACHE_CAP,
            REIMS_VGPU_DEPTH_STENCIL_CACHE_CAP,
            REIMS_VGPU_COMPUTE_REFLECT_CACHE_CAP,
        ] {
            assert!(cap > 0);
        }
        assert_eq!(MTL_BUFFER_LAYOUT_STRIDE_DYNAMIC, u64::MAX);
    }
}
