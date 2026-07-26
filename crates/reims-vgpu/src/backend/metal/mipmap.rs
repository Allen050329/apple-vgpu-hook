//! Metal-filtered `generateMipmapsForTexture:` for multi-level 2D textures.
//!
//! Builds a temporary Shared storage texture, uploads level 0 in the guest
//! native pixel format, runs the Metal blit encoder filter, and reads back
//! every level as tightly packed native rows.

use crate::backend::metal::format::pixel_format_from_u32;
use crate::backend::metal::runtime::{system_device, thread_queue};
use crate::contract::pixel_format::{self, bytes_per_pixel};
use metal::{
    MTLCommandBufferStatus, MTLOrigin, MTLPixelFormat, MTLRegion, MTLSize, MTLStorageMode,
    MTLTextureType, MTLTextureUsage, TextureDescriptor,
};

/// One mip level after Metal filter generation (tight native packing).
#[derive(Clone, Debug)]
pub struct MetalMipLevel {
    pub width: u32,
    pub height: u32,
    pub tight_bytes: Vec<u8>,
}

/// Exact failed checks for the Metal mipmap path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MetalMipmapError {
    NoDevice,
    UnsupportedFormat {
        format: u16,
    },
    WidthZero,
    HeightZero,
    LevelCountTooSmall {
        levels: u32,
    },
    BaseSpanOverflow {
        width: u32,
        height: u32,
        bpp: u32,
    },
    Level0TooShort {
        len: usize,
        expected: u64,
    },
    LevelCountRejected {
        requested: u32,
        actual: u64,
    },
    CommandBufferFailed,
    LevelSpanOverflow {
        level: u32,
        row_bytes: u64,
        height: u32,
    },
}

impl crate::observe::Decline for MetalMipmapError {
    fn slug(&self) -> &'static str {
        match self {
            Self::NoDevice => "metal_mipmap_device_unavailable",
            Self::UnsupportedFormat { .. } => "metal_mipmap_format_unsupported",
            Self::WidthZero => "metal_mipmap_width_zero",
            Self::HeightZero => "metal_mipmap_height_zero",
            Self::LevelCountTooSmall { .. } => "metal_mipmap_level_count_too_small",
            Self::BaseSpanOverflow { .. } => "metal_mipmap_base_span_overflow",
            Self::Level0TooShort { .. } => "metal_mipmap_level0_too_short",
            Self::LevelCountRejected { .. } => "metal_mipmap_level_count_rejected",
            Self::CommandBufferFailed => "metal_mipmap_command_buffer_failed",
            Self::LevelSpanOverflow { .. } => "metal_mipmap_level_span_overflow",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::NoDevice | Self::WidthZero | Self::HeightZero | Self::CommandBufferFailed => {
                Vec::new()
            }
            Self::UnsupportedFormat { format } => vec![("format", format.to_string())],
            Self::LevelCountTooSmall { levels } => vec![("levels", levels.to_string())],
            Self::BaseSpanOverflow { width, height, bpp } => vec![
                ("width", width.to_string()),
                ("height", height.to_string()),
                ("bpp", bpp.to_string()),
            ],
            Self::Level0TooShort { len, expected } => {
                vec![("len", len.to_string()), ("expected", expected.to_string())]
            }
            Self::LevelCountRejected { requested, actual } => vec![
                ("requested", requested.to_string()),
                ("actual", actual.to_string()),
            ],
            Self::LevelSpanOverflow {
                level,
                row_bytes,
                height,
            } => vec![
                ("level", level.to_string()),
                ("row_bytes", row_bytes.to_string()),
                ("height", height.to_string()),
            ],
        }
    }
}

/// Guest MTL formats that Metal treats as filterable for `generateMipmapsForTexture:`.
///
/// Integer formats are not filterable and must fail visibly.
pub fn filterable_format(format: u16) -> Option<(MTLPixelFormat, u32)> {
    let bpp = bytes_per_pixel(format)?;
    match format {
        pixel_format::MTL_FORMAT_A8_UNORM
        | pixel_format::MTL_FORMAT_R8_UNORM
        | pixel_format::MTL_FORMAT_RG8_UNORM
        | pixel_format::MTL_FORMAT_R16_FLOAT
        | pixel_format::MTL_FORMAT_RG16_FLOAT
        | pixel_format::MTL_FORMAT_RGBA8_UNORM
        | pixel_format::MTL_FORMAT_RGBA8_UNORM_SRGB
        | pixel_format::MTL_FORMAT_BGRA8_UNORM
        | pixel_format::MTL_FORMAT_BGRA8_UNORM_SRGB
        | pixel_format::MTL_FORMAT_RGBA16_FLOAT
        | pixel_format::MTL_FORMAT_RGBA32_FLOAT => {
            Some((pixel_format_from_u32(format as u32), bpp))
        }
        _ => None,
    }
}

/// Metal's standard 2D mip dimension for `base` at `level` (floor divide by 2 each step).
pub fn metal_mip_extent(base: u32, level: u32) -> u32 {
    if base == 0 {
        return 0;
    }
    (base >> level).max(1)
}

/// Upload L0, run Metal-filtered mip generation, return levels `[0..levels)`.
///
/// `level0` must be tightly packed native rows (`width * bpp` per row). `levels`
/// must be `> 1`. Level 0 in the result is a copy of the input; levels 1.. are
/// Metal-filtered.
pub fn generate_mipmaps_filtered(
    format: u16,
    width: u32,
    height: u32,
    levels: u32,
    level0: &[u8],
) -> Result<Vec<MetalMipLevel>, MetalMipmapError> {
    if width == 0 {
        return Err(MetalMipmapError::WidthZero);
    }
    if height == 0 {
        return Err(MetalMipmapError::HeightZero);
    }
    if levels <= 1 {
        return Err(MetalMipmapError::LevelCountTooSmall { levels });
    }
    let (mtl_fmt, bpp) =
        filterable_format(format).ok_or(MetalMipmapError::UnsupportedFormat { format })?;
    let tight0 = (width as u64)
        .checked_mul(height as u64)
        .and_then(|v| v.checked_mul(bpp as u64))
        .ok_or(MetalMipmapError::BaseSpanOverflow { width, height, bpp })?;
    if level0.len() < tight0 as usize {
        return Err(MetalMipmapError::Level0TooShort {
            len: level0.len(),
            expected: tight0,
        });
    }
    // Both factors are u32, so their product always fits in u64.
    let bytes_per_row0 = (width as u64) * (bpp as u64);

    let device = system_device().ok_or(MetalMipmapError::NoDevice)?;
    let queue = thread_queue(device);

    let descriptor = TextureDescriptor::new();
    descriptor.set_texture_type(MTLTextureType::D2);
    descriptor.set_pixel_format(mtl_fmt);
    descriptor.set_width(width as u64);
    descriptor.set_height(height as u64);
    descriptor.set_mipmap_level_count(levels as u64);
    descriptor.set_storage_mode(MTLStorageMode::Shared);
    // ShaderRead is the documented usage for filterable sampled textures;
    // generateMipmapsForTexture operates on filterable color textures.
    descriptor.set_usage(MTLTextureUsage::ShaderRead);
    let texture = device.new_texture(&descriptor);
    if texture.mipmap_level_count() < levels as u64 {
        return Err(MetalMipmapError::LevelCountRejected {
            requested: levels,
            actual: texture.mipmap_level_count(),
        });
    }

    let region0 = MTLRegion {
        origin: MTLOrigin { x: 0, y: 0, z: 0 },
        size: MTLSize {
            width: width as u64,
            height: height as u64,
            depth: 1,
        },
    };
    texture.replace_region(region0, 0, level0.as_ptr() as *const _, bytes_per_row0);

    let command_buffer = queue.new_command_buffer().to_owned();
    let blit = command_buffer.new_blit_command_encoder();
    blit.generate_mipmaps(&texture);
    blit.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
    if command_buffer.status() == MTLCommandBufferStatus::Error {
        return Err(MetalMipmapError::CommandBufferFailed);
    }

    let mut out = Vec::with_capacity(levels as usize);
    for level in 0..levels {
        let w = metal_mip_extent(width, level);
        let h = metal_mip_extent(height, level);
        // Both factors are u32, so their product always fits in u64.
        let bpr = (w as u64) * (bpp as u64);
        let need = bpr
            .checked_mul(h as u64)
            .ok_or(MetalMipmapError::LevelSpanOverflow {
                level,
                row_bytes: bpr,
                height: h,
            })?;
        let mut tight = vec![0u8; need as usize];
        let region = MTLRegion {
            origin: MTLOrigin { x: 0, y: 0, z: 0 },
            size: MTLSize {
                width: w as u64,
                height: h as u64,
                depth: 1,
            },
        };
        texture.get_bytes(tight.as_mut_ptr() as *mut _, bpr, region, level as u64);
        out.push(MetalMipLevel {
            width: w,
            height: h,
            tight_bytes: tight,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::pixel_format::MTL_FORMAT_RGBA8_UNORM;
    use crate::observe::Emit;

    #[test]
    fn filterable_accepts_unorm_rejects_uint() {
        assert!(filterable_format(MTL_FORMAT_RGBA8_UNORM).is_some());
        assert!(filterable_format(pixel_format::MTL_FORMAT_BGRA8_UNORM).is_some());
        assert!(filterable_format(pixel_format::MTL_FORMAT_RGBA8_UINT).is_none());
        assert!(filterable_format(pixel_format::MTL_FORMAT_RGBA16_UINT).is_none());
    }

    #[test]
    fn metal_mip_extent_chain() {
        assert_eq!(metal_mip_extent(8, 0), 8);
        assert_eq!(metal_mip_extent(8, 1), 4);
        assert_eq!(metal_mip_extent(8, 3), 1);
        assert_eq!(metal_mip_extent(5, 1), 2);
        assert_eq!(metal_mip_extent(5, 2), 1);
        assert_eq!(metal_mip_extent(3, 1), 1);
    }

    #[test]
    fn metal_generate_constant_rgba8_preserves_color() {
        // 4×4 solid (200, 10, 20, 255) → filtered mips stay that color.
        let w = 4u32;
        let h = 4u32;
        let levels = 3u32;
        let mut l0 = vec![0u8; (w * h * 4) as usize];
        for px in l0.chunks_exact_mut(4) {
            px[0] = 200;
            px[1] = 10;
            px[2] = 20;
            px[3] = 255;
        }
        let chain = generate_mipmaps_filtered(MTL_FORMAT_RGBA8_UNORM, w, h, levels, &l0)
            .expect("metal generate");
        assert_eq!(chain.len(), 3);
        assert_eq!((chain[0].width, chain[0].height), (4, 4));
        assert_eq!((chain[1].width, chain[1].height), (2, 2));
        assert_eq!((chain[2].width, chain[2].height), (1, 1));
        for level in &chain {
            for px in level.tight_bytes.chunks_exact(4) {
                assert_eq!(px, &[200, 10, 20, 255], "mip {} color", level.width);
            }
        }
    }

    #[test]
    fn metal_generate_rejects_single_level() {
        let l0 = [1u8, 2, 3, 4];
        let error = generate_mipmaps_filtered(MTL_FORMAT_RGBA8_UNORM, 1, 1, 1, &l0).unwrap_err();
        assert_eq!(error, MetalMipmapError::LevelCountTooSmall { levels: 1 });
        assert_eq!(
            Emit::decline("metal_mipmap_test", &error).render(),
            "metal_mipmap_test reason=metal_mipmap_level_count_too_small levels=1"
        );
    }

    #[test]
    fn metal_generate_rejects_uint() {
        let l0 = [1u8, 2, 3, 4];
        let error = generate_mipmaps_filtered(pixel_format::MTL_FORMAT_RGBA8_UINT, 1, 1, 2, &l0)
            .unwrap_err();
        assert_eq!(
            error,
            MetalMipmapError::UnsupportedFormat {
                format: pixel_format::MTL_FORMAT_RGBA8_UINT
            }
        );
        assert_eq!(
            Emit::decline("metal_mipmap_test", &error).render(),
            format!(
                "metal_mipmap_test reason=metal_mipmap_format_unsupported format={}",
                pixel_format::MTL_FORMAT_RGBA8_UINT
            )
        );
    }

    #[test]
    fn metal_generate_reports_the_level_zero_byte_requirement() {
        let error =
            generate_mipmaps_filtered(MTL_FORMAT_RGBA8_UNORM, 2, 2, 2, &[0; 15]).unwrap_err();
        assert_eq!(
            error,
            MetalMipmapError::Level0TooShort {
                len: 15,
                expected: 16
            }
        );
        assert_eq!(
            Emit::decline("metal_mipmap_test", &error).render(),
            "metal_mipmap_test reason=metal_mipmap_level0_too_short len=15 expected=16"
        );
    }

    #[test]
    fn metal_generate_names_each_zero_axis_separately() {
        let level0 = [0u8; 4];
        let width =
            generate_mipmaps_filtered(MTL_FORMAT_RGBA8_UNORM, 0, 1, 2, &level0).unwrap_err();
        let height =
            generate_mipmaps_filtered(MTL_FORMAT_RGBA8_UNORM, 1, 0, 2, &level0).unwrap_err();

        assert_eq!(width, MetalMipmapError::WidthZero);
        assert_eq!(height, MetalMipmapError::HeightZero);
        assert_eq!(
            Emit::decline("metal_mipmap_test", &width).render(),
            "metal_mipmap_test reason=metal_mipmap_width_zero"
        );
        assert_eq!(
            Emit::decline("metal_mipmap_test", &height).render(),
            "metal_mipmap_test reason=metal_mipmap_height_zero"
        );
    }
}
