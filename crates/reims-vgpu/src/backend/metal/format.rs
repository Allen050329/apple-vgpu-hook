//! Pixel-format helpers matching ObjC `reims_vgpu_storage_image_format` / `reims_vgpu_mtl_pixel_format_bpp`.

use crate::backend::metal::abi::*;
use metal::MTLPixelFormat;

pub fn storage_image_format(format: u32) -> Option<(MTLPixelFormat, usize)> {
    match format {
        REIMS_VGPU_SIMG_RGBA8_UINT => Some((MTLPixelFormat::RGBA8Uint, 4)),
        REIMS_VGPU_SIMG_RGBA8_SINT => Some((MTLPixelFormat::RGBA8Sint, 4)),
        REIMS_VGPU_SIMG_RGBA16_UINT => Some((MTLPixelFormat::RGBA16Uint, 8)),
        REIMS_VGPU_SIMG_RGBA16_FLOAT => Some((MTLPixelFormat::RGBA16Float, 8)),
        REIMS_VGPU_SIMG_RGBA32_FLOAT => Some((MTLPixelFormat::RGBA32Float, 16)),
        REIMS_VGPU_SIMG_RGBA8_UNORM => Some((MTLPixelFormat::RGBA8Unorm, 4)),
        REIMS_VGPU_SIMG_BGRA8_UNORM => Some((MTLPixelFormat::BGRA8Unorm, 4)),
        REIMS_VGPU_SIMG_R16_FLOAT => Some((MTLPixelFormat::R16Float, 2)),
        REIMS_VGPU_SIMG_RG16_FLOAT => Some((MTLPixelFormat::RG16Float, 4)),
        REIMS_VGPU_SIMG_R8_UNORM => Some((MTLPixelFormat::R8Unorm, 1)),
        REIMS_VGPU_SIMG_RG8_UNORM => Some((MTLPixelFormat::RG8Unorm, 2)),
        REIMS_VGPU_SIMG_RGBA32_UINT => Some((MTLPixelFormat::RGBA32Uint, 16)),
        _ => None,
    }
}

pub fn mtl_pixel_format_bpp(pixel_format: u32) -> Option<usize> {
    // Compare raw MTLPixelFormat values (same as ObjC switch).
    match pixel_format {
        x if x == MTLPixelFormat::A8Unorm as u32 || x == MTLPixelFormat::R8Unorm as u32 => Some(1),
        x if x == MTLPixelFormat::R16Float as u32 || x == MTLPixelFormat::RG8Unorm as u32 => {
            Some(2)
        }
        x if x == MTLPixelFormat::RGBA8Unorm as u32
            || x == MTLPixelFormat::RGBA8Unorm_sRGB as u32
            || x == MTLPixelFormat::BGRA8Unorm as u32
            || x == MTLPixelFormat::BGRA8Unorm_sRGB as u32
            || x == MTLPixelFormat::RG16Float as u32 =>
        {
            Some(4)
        }
        x if x == MTLPixelFormat::RGBA16Float as u32 || x == MTLPixelFormat::RGBA16Uint as u32 => {
            Some(8)
        }
        x if x == MTLPixelFormat::RGBA32Float as u32 || x == MTLPixelFormat::RGBA32Uint as u32 => {
            Some(16)
        }
        _ => None,
    }
}

pub fn pixel_format_from_u32(v: u32) -> MTLPixelFormat {
    // SAFETY: callers only pass raw Metal enum values from the ABI contract.
    unsafe { std::mem::transmute(v as u64) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn storage_image_formats_report_their_metal_format_and_texel_size() {
        let cases = [
            (REIMS_VGPU_SIMG_RGBA8_UINT, MTLPixelFormat::RGBA8Uint, 4),
            (REIMS_VGPU_SIMG_RGBA8_SINT, MTLPixelFormat::RGBA8Sint, 4),
            (REIMS_VGPU_SIMG_RGBA16_UINT, MTLPixelFormat::RGBA16Uint, 8),
            (REIMS_VGPU_SIMG_RGBA16_FLOAT, MTLPixelFormat::RGBA16Float, 8),
            (
                REIMS_VGPU_SIMG_RGBA32_FLOAT,
                MTLPixelFormat::RGBA32Float,
                16,
            ),
            (REIMS_VGPU_SIMG_RGBA8_UNORM, MTLPixelFormat::RGBA8Unorm, 4),
            (REIMS_VGPU_SIMG_BGRA8_UNORM, MTLPixelFormat::BGRA8Unorm, 4),
            (REIMS_VGPU_SIMG_R16_FLOAT, MTLPixelFormat::R16Float, 2),
            (REIMS_VGPU_SIMG_RG16_FLOAT, MTLPixelFormat::RG16Float, 4),
            (REIMS_VGPU_SIMG_R8_UNORM, MTLPixelFormat::R8Unorm, 1),
            (REIMS_VGPU_SIMG_RG8_UNORM, MTLPixelFormat::RG8Unorm, 2),
            (REIMS_VGPU_SIMG_RGBA32_UINT, MTLPixelFormat::RGBA32Uint, 16),
        ];
        for (wire, metal, bytes) in cases {
            let (actual, actual_bytes) = storage_image_format(wire).expect("mapped format");
            assert_eq!(actual as u64, metal as u64);
            assert_eq!(actual_bytes, bytes);
        }
        assert_eq!(storage_image_format(u32::MAX), None);
        assert_eq!(mtl_pixel_format_bpp(u32::MAX), None);
    }

    #[test]
    fn render_pixel_formats_report_their_byte_widths() {
        let cases = [
            (MTLPixelFormat::A8Unorm, 1),
            (MTLPixelFormat::R8Unorm, 1),
            (MTLPixelFormat::R16Float, 2),
            (MTLPixelFormat::RG8Unorm, 2),
            (MTLPixelFormat::RGBA8Unorm, 4),
            (MTLPixelFormat::RGBA8Unorm_sRGB, 4),
            (MTLPixelFormat::BGRA8Unorm, 4),
            (MTLPixelFormat::BGRA8Unorm_sRGB, 4),
            (MTLPixelFormat::RG16Float, 4),
            (MTLPixelFormat::RGBA16Float, 8),
            (MTLPixelFormat::RGBA16Uint, 8),
            (MTLPixelFormat::RGBA32Float, 16),
            (MTLPixelFormat::RGBA32Uint, 16),
        ];
        for (format, bytes) in cases {
            assert_eq!(mtl_pixel_format_bpp(format as u32), Some(bytes));
        }
    }

    #[test]
    fn raw_pixel_format_conversion_preserves_the_abi_value() {
        let raw = MTLPixelFormat::BGRA8Unorm_sRGB as u32;
        assert_eq!(pixel_format_from_u32(raw) as u64, raw as u64);
    }
}
