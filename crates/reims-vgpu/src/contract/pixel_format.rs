//! Metal pixel-format helpers (port of `host/utils/reims-vgpu-pixel-format`).

use crate::contract::endian::{ld16, st16};
use crate::contract::checked_mul_u64;

pub const COMPONENT_COUNT: usize = 4;
pub const COMPONENT_R: usize = 0;
pub const COMPONENT_G: usize = 1;
pub const COMPONENT_B: usize = 2;
pub const COMPONENT_A: usize = 3;

pub const R8_BPP: u32 = 1;
pub const RG8_BPP: u32 = 2;
pub const RGBA8_BPP: u32 = 4;
pub const BGRA8_BPP: u32 = RGBA8_BPP;
pub const R16F_BPP: u32 = 2;
pub const R32F_BPP: u32 = 4;
pub const RG16F_BPP: u32 = 4;
pub const RGBA16_BPP: u32 = 8;
pub const RGBA16F_BPP: u32 = RGBA16_BPP;
pub const RGBA32_BPP: u32 = 16;
pub const RGBA32F_BPP: u32 = RGBA32_BPP;
pub const R32_BPP: u32 = 4;

pub const IOSURFACE_ROW_ALIGNMENT: u32 = 128;

// MTLPixelFormat values (Metal.framework Headers/MTLPixelFormat.h).
pub const MTL_FORMAT_A8_UNORM: u16 = 0x01;
pub const MTL_FORMAT_R8_UNORM: u16 = 0x0a;
pub const MTL_FORMAT_R16_FLOAT: u16 = 0x19;
pub const MTL_FORMAT_RG8_UNORM: u16 = 0x1e;
pub const MTL_FORMAT_R32_UINT: u16 = 0x35;
pub const MTL_FORMAT_R32_SINT: u16 = 0x36;
pub const MTL_FORMAT_R32_FLOAT: u16 = 0x37;
pub const MTL_FORMAT_RG16_FLOAT: u16 = 0x41;
pub const MTL_FORMAT_RGBA8_UNORM: u16 = 0x46;
pub const MTL_FORMAT_RGBA8_UNORM_SRGB: u16 = 0x47;
pub const MTL_FORMAT_RGBA8_UINT: u16 = 0x49;
pub const MTL_FORMAT_RGBA8_SINT: u16 = 0x4a;
pub const MTL_FORMAT_BGRA8_UNORM: u16 = 0x50;
pub const MTL_FORMAT_BGRA8_UNORM_SRGB: u16 = 0x51;
/// Packed RGB9E5 shared-exponent float. 32-bit texels.
pub const MTL_FORMAT_RGB9E5_FLOAT: u16 = 0x5d;
pub const MTL_FORMAT_RGBA16_UINT: u16 = 0x71;
pub const MTL_FORMAT_RGBA16_FLOAT: u16 = 0x73;
pub const MTL_FORMAT_RGBA32_UINT: u16 = 0x7b;
pub const MTL_FORMAT_RGBA32_FLOAT: u16 = 0x7d;
// Depth / stencil (Metal.framework Headers/MTLPixelFormat.h).
pub const MTL_FORMAT_DEPTH16_UNORM: u16 = 250;
pub const MTL_FORMAT_DEPTH32_FLOAT: u16 = 252;
pub const MTL_FORMAT_STENCIL8: u16 = 253;
pub const MTL_FORMAT_DEPTH24_UNORM_STENCIL8: u16 = 255;
pub const MTL_FORMAT_DEPTH32_FLOAT_STENCIL8: u16 = 260;
pub const MTL_FORMAT_X32_STENCIL8: u16 = 261;
pub const MTL_FORMAT_X24_STENCIL8: u16 = 262;

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StorageImageSelector {
    Rgba8Uint = 0,
    Rgba8Sint = 1,
    Rgba16Uint = 2,
    Rgba16Float = 3,
    Rgba32Float = 4,
    Rgba8Unorm = 5,
    Bgra8Unorm = 6,
    R16Float = 7,
    Rg16Float = 8,
    R8Unorm = 9,
    Rg8Unorm = 10,
    Rgba32Uint = 11,
    R32Uint = 12,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SampledClass {
    Unsupported = 0,
    A8Unorm,
    R8Unorm,
    Rg8Unorm,
    Rgba8Unorm,
    Bgra8Unorm,
    Rgba16Float,
}

/// The byte layout of one guest texel on the sampled rails, independent of any
/// host graphics API.
///
/// This is the vocabulary `runtime/` speaks about sampled texels. It is
/// deliberately *narrow*: these are exactly the layouts a CPU-origin upload or
/// an in-place guest gather can hand a sampled image without a conversion pass.
/// A rail that carries the full format set names the host format instead — the
/// engine stores `VkFormat` and can therefore express an sRGB sampled view,
/// which a layout enum by construction cannot.
///
/// It lives in the contract rather than in either the runtime or the backend
/// because both used to hold their own copy of it, with a hand-written mapping
/// between the two that nothing checked.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TexelLayout {
    /// 4 bytes/texel, guest channel order R,G,B,A — the default CPU-origin
    /// layout, and what every convert-to-RGBA8 loader produces.
    Rgba8,
    /// 4 bytes/texel, guest channel order B,G,R,A. Uploaded as-is so the
    /// sampler swaps channels in hardware and the CPU never runs a per-pixel
    /// swizzle.
    Bgra8,
    /// 1 byte/texel — a biplanar video luma plane, sampled at its native
    /// footprint rather than expanded to RGBA8.
    R8,
    /// 2 bytes/texel — a biplanar video chroma plane, likewise native.
    Rg8,
    /// 2 bytes/texel — a single-channel `float16` texture, sampled natively as
    /// `R16_SFLOAT` (the shader reads `.x`, the other lanes expand to `0,0,1`).
    /// Color-management 1D LUTs (macOS WindowServer's `UberCompositeFragment`
    /// display-profile pass) are stored this way; converting them to unorm8
    /// would quantize the transfer curve, and the CPU `texel_to_rgba8` loader
    /// has no float arm, so this native rail is the only correct path. Not a
    /// four-byte color layout, so it never rides the RGBA8-shaped loaders.
    R16Float,
    /// 4 bytes/texel — a single-channel `float32` texture, sampled natively as
    /// `R32_SFLOAT`. Same color-LUT role as [`Self::R16Float`], but its
    /// linear-filter feature is optional (absent on Apple/MoltenVK), so the
    /// rail that emits this layout must first confirm the host supports it.
    /// Four bytes wide but **not** a colour order, so it stays out of the
    /// RGBA8-shaped loaders and `is_four_byte_color`.
    R32Float,
}

impl TexelLayout {
    /// Bytes occupied by one texel in guest linear storage.
    pub fn bytes_per_texel(self) -> u32 {
        match self {
            Self::Rgba8 | Self::Bgra8 => RGBA8_BPP,
            Self::R8 => R8_BPP,
            Self::Rg8 => RG8_BPP,
            Self::R16Float => R16F_BPP,
            Self::R32Float => R32F_BPP,
        }
    }

    /// Whether this layout is one of the two four-byte colour orders.
    ///
    /// Several rails admit only these: the RGBA8-shaped diagnostics, the
    /// tight-row loaders and the zero-copy gathers all assume a four-byte
    /// texel. Named once so a rail states which set it takes instead of
    /// re-listing the variants.
    pub fn is_four_byte_color(self) -> bool {
        matches!(self, Self::Rgba8 | Self::Bgra8)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum RenderTargetClass {
    Unsupported = 0,
    /// Stable C ABI ordinals: 1=BGRA8, 2=RGBA16F (bring-up compositor set).
    Bgra8Unorm = 1,
    Rgba16Float = 2,
    /// App/intermediate Metal color RTs (Metal color-renderable; not heuristics).
    Rgba8Unorm = 3,
    Rgba8UnormSrgb = 4,
    Bgra8UnormSrgb = 5,
    /// Two-channel float16 color RT (Metal color-renderable). Used as the
    /// secondary MRT mask/coverage slot of vibrancy UI tiles (Control Center,
    /// widgets). The GPU pass is 8-bit UNORM like every other target; the R/G
    /// channels round-trip through the f16 LUT at guest writeback/seed.
    Rg16Float = 6,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SwizzleSource {
    Zero = 0,
    One = 1,
    R = 2,
    G = 3,
    B = 4,
    A = 5,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SwizzlePlan {
    pub source: [SwizzleSource; COMPONENT_COUNT],
}

impl Default for SwizzlePlan {
    fn default() -> Self {
        swizzle_identity()
    }
}

const UNORM8_MIN: u8 = 0x00;
const UNORM8_MAX: u8 = 0xff;

const F16_SIGN_MASK: u16 = 0x8000;
const F16_EXP_SHIFT: u32 = 10;
const F16_EXP_MASK: u32 = 0x1f;
const F16_MANT_MASK: u32 = 0x03ff;
const F16_HIDDEN_BIT: u32 = 0x0400;
const F16_EXP_BIAS: i32 = 15;
const F16_EXP_INF_NAN: u32 = F16_EXP_MASK;
const F16_INF_BITS: u16 = 0x7c00;
const F16_SUBNORMAL_EXP_MIN: i32 = -10;
const F16_SUBNORMAL_SHIFT_BASE: i32 = 14;
const F16_F32_SIGN_SHIFT: u32 = 16;
const F32_EXP_SHIFT: u32 = 23;
const F32_EXP_MASK: u32 = 0xff;
const F32_MANT_MASK: u32 = 0x007f_ffff;
const F32_HIDDEN_BIT: u32 = 0x0080_0000;
const F32_EXP_BIAS: i32 = 127;
const F32_INF_BITS: u32 = 0x7f80_0000;
const F16_F32_MANT_SHIFT: u32 = 13;
const F32_TO_F16_ROUND_BIT: u32 = 0x1000;

pub fn bytes_per_pixel(format: u16) -> Option<u32> {
    Some(match format {
        MTL_FORMAT_A8_UNORM | MTL_FORMAT_R8_UNORM | MTL_FORMAT_STENCIL8 => R8_BPP,
        MTL_FORMAT_R16_FLOAT | MTL_FORMAT_RG8_UNORM | MTL_FORMAT_DEPTH16_UNORM => RG8_BPP,
        MTL_FORMAT_RG16_FLOAT => RG16F_BPP,
        MTL_FORMAT_RGBA8_UNORM
        | MTL_FORMAT_RGBA8_UNORM_SRGB
        | MTL_FORMAT_RGBA8_UINT
        | MTL_FORMAT_RGBA8_SINT
        | MTL_FORMAT_BGRA8_UNORM
        | MTL_FORMAT_BGRA8_UNORM_SRGB
        | MTL_FORMAT_R32_UINT
        | MTL_FORMAT_R32_SINT
        | MTL_FORMAT_R32_FLOAT
        | MTL_FORMAT_RGB9E5_FLOAT
        | MTL_FORMAT_DEPTH32_FLOAT
        | MTL_FORMAT_DEPTH24_UNORM_STENCIL8
        | MTL_FORMAT_X24_STENCIL8 => RGBA8_BPP,
        // Depth32Float_Stencil8 / X32_Stencil8: 64-bit cells on Apple Silicon
        // (40-bit logical DS + pad; Metal allocates 8 B/texel for this family).
        MTL_FORMAT_DEPTH32_FLOAT_STENCIL8 | MTL_FORMAT_X32_STENCIL8 => 8,
        MTL_FORMAT_RGBA16_UINT | MTL_FORMAT_RGBA16_FLOAT => RGBA16_BPP,
        MTL_FORMAT_RGBA32_UINT | MTL_FORMAT_RGBA32_FLOAT => RGBA32_BPP,
        _ => return None,
    })
}

/// Whether `format` has a depth plane (for `MTLBlitOptionDepthFromDepthStencil`).
pub fn format_has_depth_aspect(format: u16) -> bool {
    matches!(
        format,
        MTL_FORMAT_DEPTH16_UNORM
            | MTL_FORMAT_DEPTH32_FLOAT
            | MTL_FORMAT_DEPTH24_UNORM_STENCIL8
            | MTL_FORMAT_DEPTH32_FLOAT_STENCIL8
    )
}

/// Whether `format` has a stencil plane (for `MTLBlitOptionStencilFromDepthStencil`).
pub fn format_has_stencil_aspect(format: u16) -> bool {
    matches!(
        format,
        MTL_FORMAT_STENCIL8
            | MTL_FORMAT_DEPTH24_UNORM_STENCIL8
            | MTL_FORMAT_DEPTH32_FLOAT_STENCIL8
            | MTL_FORMAT_X32_STENCIL8
            | MTL_FORMAT_X24_STENCIL8
    )
}

/// Linear packing of a combined depth-stencil texel (full cell in guest storage).
///
/// Plane sizes when extracted to a buffer match Metal blit options:
/// depth plane → 4 B (Depth32Float / Depth24 expanded unorm32), stencil → 1 B.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DepthStencilPacking {
    /// Full texel size in guest linear storage.
    pub full_bpp: u32,
    /// Byte offset of the depth field within the texel (if present).
    pub depth_offset: u32,
    /// Raw depth field size in the packed texel (before buffer expansion).
    pub depth_raw_size: u32,
    /// Buffer-side depth plane size after Metal extraction.
    pub depth_plane_bpp: u32,
    /// Byte offset of the stencil field within the texel (if present).
    pub stencil_offset: u32,
    pub stencil_plane_bpp: u32,
    /// How depth is stored in the packed texel.
    pub depth_layout: DepthFieldLayout,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DepthFieldLayout {
    /// Not a depth-bearing format (stencil-only combined / X\*\_Stencil8).
    None,
    /// IEEE-754 binary32 LE at `depth_offset`.
    Float32,
    /// 24-bit unorm in bits \[8:31\] of a LE u32 (stencil in low 8 bits).
    Unorm24High,
}

/// Combined depth-stencil packing for formats that interleave both planes.
///
/// Pure Depth32Float / Stencil8 / Depth16 return `None` (no repack; aspect is identity).
pub fn depth_stencil_packing(format: u16) -> Option<DepthStencilPacking> {
    match format {
        // Apple docs: 40-bit logical (32f + 8); Apple Silicon cells are 8 B.
        // Layout: depth f32 @0, stencil u8 @4, pad @5..7.
        MTL_FORMAT_DEPTH32_FLOAT_STENCIL8 => Some(DepthStencilPacking {
            full_bpp: 8,
            depth_offset: 0,
            depth_raw_size: 4,
            depth_plane_bpp: 4,
            stencil_offset: 4,
            stencil_plane_bpp: 1,
            depth_layout: DepthFieldLayout::Float32,
        }),
        // 32-bit cell: stencil in low 8, depth unorm24 in high 24 (Metal/macOS common packing).
        MTL_FORMAT_DEPTH24_UNORM_STENCIL8 => Some(DepthStencilPacking {
            full_bpp: 4,
            depth_offset: 0,
            depth_raw_size: 4,
            depth_plane_bpp: 4,
            stencil_offset: 0,
            stencil_plane_bpp: 1,
            depth_layout: DepthFieldLayout::Unorm24High,
        }),
        // X32_Stencil8: same 8 B cell as Depth32Float_Stencil8 without meaningful depth.
        MTL_FORMAT_X32_STENCIL8 => Some(DepthStencilPacking {
            full_bpp: 8,
            depth_offset: 0,
            depth_raw_size: 0,
            depth_plane_bpp: 0,
            stencil_offset: 4,
            stencil_plane_bpp: 1,
            depth_layout: DepthFieldLayout::None,
        }),
        // X24_Stencil8: 4 B cell, stencil in low 8.
        MTL_FORMAT_X24_STENCIL8 => Some(DepthStencilPacking {
            full_bpp: 4,
            depth_offset: 0,
            depth_raw_size: 0,
            depth_plane_bpp: 0,
            stencil_offset: 0,
            stencil_plane_bpp: 1,
            depth_layout: DepthFieldLayout::None,
        }),
        _ => None,
    }
}

/// Bytes per texel for a blit aspect selection.
///
/// Pure depth/stencil formats: aspect matches full bpp (option is identity).
/// Combined formats: Full uses packed `full_bpp`; depth plane is 4 B; stencil is 1 B.
pub fn blit_aspect_bytes_per_pixel(
    format: u16,
    depth_aspect: bool,
    stencil_aspect: bool,
) -> Option<u32> {
    if depth_aspect && stencil_aspect {
        return None;
    }
    if depth_aspect {
        if !format_has_depth_aspect(format) {
            return None;
        }
        if let Some(p) = depth_stencil_packing(format) {
            return if p.depth_plane_bpp != 0 {
                Some(p.depth_plane_bpp)
            } else {
                None
            };
        }
        return Some(match format {
            MTL_FORMAT_DEPTH16_UNORM => 2,
            MTL_FORMAT_DEPTH32_FLOAT => 4,
            _ => return None,
        });
    }
    if stencil_aspect {
        if !format_has_stencil_aspect(format) {
            return None;
        }
        if let Some(p) = depth_stencil_packing(format) {
            return Some(p.stencil_plane_bpp);
        }
        return Some(1);
    }
    // Full texel.
    bytes_per_pixel(format)
}

/// Whether a plane extract/insert pass is required (combined DS + aspect option).
pub fn blit_aspect_needs_repack(format: u16, depth_aspect: bool, stencil_aspect: bool) -> bool {
    if !depth_aspect && !stencil_aspect {
        return false;
    }
    depth_stencil_packing(format).is_some()
}

/// Extract one plane from a packed depth-stencil texel into `dst` (plane-native size).
pub fn extract_depth_stencil_plane(
    format: u16,
    depth_aspect: bool,
    stencil_aspect: bool,
    texel: &[u8],
    dst: &mut [u8],
) -> bool {
    if depth_aspect == stencil_aspect {
        return false;
    }
    let Some(p) = depth_stencil_packing(format) else {
        return false;
    };
    if texel.len() < p.full_bpp as usize {
        return false;
    }
    if depth_aspect {
        if p.depth_plane_bpp == 0 || dst.len() < p.depth_plane_bpp as usize {
            return false;
        }
        match p.depth_layout {
            DepthFieldLayout::Float32 => {
                let o = p.depth_offset as usize;
                dst[..4].copy_from_slice(&texel[o..o + 4]);
            }
            DepthFieldLayout::Unorm24High => {
                // Packed LE u32: stencil @bits0-7, depth unorm24 @bits8-31.
                // Metal depth buffer plane is 32-bit unorm (depth in low 24).
                let packed = u32::from_le_bytes([texel[0], texel[1], texel[2], texel[3]]);
                let depth24 = packed >> 8;
                dst[..4].copy_from_slice(&depth24.to_le_bytes());
            }
            DepthFieldLayout::None => return false,
        }
        return true;
    }
    // Stencil plane.
    if dst.len() < p.stencil_plane_bpp as usize {
        return false;
    }
    dst[0] = texel[p.stencil_offset as usize];
    true
}

/// Insert one plane into a packed depth-stencil texel (read-modify-write).
///
/// `texel` holds the current full cell (updated in place). `src` is plane-native.
pub fn insert_depth_stencil_plane(
    format: u16,
    depth_aspect: bool,
    stencil_aspect: bool,
    src: &[u8],
    texel: &mut [u8],
) -> bool {
    if depth_aspect == stencil_aspect {
        return false;
    }
    let Some(p) = depth_stencil_packing(format) else {
        return false;
    };
    if texel.len() < p.full_bpp as usize {
        return false;
    }
    if depth_aspect {
        if p.depth_plane_bpp == 0 || src.len() < p.depth_plane_bpp as usize {
            return false;
        }
        match p.depth_layout {
            DepthFieldLayout::Float32 => {
                let o = p.depth_offset as usize;
                texel[o..o + 4].copy_from_slice(&src[..4]);
            }
            DepthFieldLayout::Unorm24High => {
                let depth24 = u32::from_le_bytes([src[0], src[1], src[2], src[3]]) & 0x00ff_ffff;
                let packed = u32::from_le_bytes([texel[0], texel[1], texel[2], texel[3]]);
                let stencil = packed & 0xff;
                let out = stencil | (depth24 << 8);
                texel[..4].copy_from_slice(&out.to_le_bytes());
            }
            DepthFieldLayout::None => return false,
        }
        return true;
    }
    if src.is_empty() {
        return false;
    }
    texel[p.stencil_offset as usize] = src[0];
    true
}

/// Extract a tight plane row from a strided packed texture row.
pub fn extract_plane_row(
    format: u16,
    depth_aspect: bool,
    stencil_aspect: bool,
    src_row: &[u8],
    width: u32,
    dst_plane: &mut [u8],
) -> bool {
    let Some(p) = depth_stencil_packing(format) else {
        return false;
    };
    let plane_bpp = if depth_aspect {
        p.depth_plane_bpp
    } else if stencil_aspect {
        p.stencil_plane_bpp
    } else {
        return false;
    } as usize;
    let full = p.full_bpp as usize;
    let w = width as usize;
    let Some(need_src) = full.checked_mul(w) else {
        return false;
    };
    let Some(need_dst) = plane_bpp.checked_mul(w) else {
        return false;
    };
    if src_row.len() < need_src || dst_plane.len() < need_dst {
        return false;
    }
    for x in 0..w {
        let t = &src_row[x * full..x * full + full];
        let d = &mut dst_plane[x * plane_bpp..x * plane_bpp + plane_bpp];
        if !extract_depth_stencil_plane(format, depth_aspect, stencil_aspect, t, d) {
            return false;
        }
    }
    true
}

/// Insert a tight plane row into a strided packed texture row (RMW per texel).
pub fn insert_plane_row(
    format: u16,
    depth_aspect: bool,
    stencil_aspect: bool,
    src_plane: &[u8],
    width: u32,
    dst_row: &mut [u8],
) -> bool {
    let Some(p) = depth_stencil_packing(format) else {
        return false;
    };
    let plane_bpp = if depth_aspect {
        p.depth_plane_bpp
    } else if stencil_aspect {
        p.stencil_plane_bpp
    } else {
        return false;
    } as usize;
    let full = p.full_bpp as usize;
    let w = width as usize;
    let Some(need_dst) = full.checked_mul(w) else {
        return false;
    };
    let Some(need_src) = plane_bpp.checked_mul(w) else {
        return false;
    };
    if dst_row.len() < need_dst || src_plane.len() < need_src {
        return false;
    }
    for x in 0..w {
        let s = &src_plane[x * plane_bpp..x * plane_bpp + plane_bpp];
        let t = &mut dst_row[x * full..x * full + full];
        if !insert_depth_stencil_plane(format, depth_aspect, stencil_aspect, s, t) {
            return false;
        }
    }
    true
}

/// Whether `format` stores sRGB-encoded values, so Metal decodes on sample and
/// encodes on write.
///
/// The class lookups below deliberately fold each `_SRGB` format onto its
/// linear sibling — the classes name a *byte layout*, and the two share one.
/// That fold is only safe because this predicate exists beside it: a caller that
/// takes the class has lost the qualifier and can ask here whether it just did.
/// Without it the loss is indistinguishable from the format never having been
/// sRGB at all.
pub fn is_srgb(format: u16) -> bool {
    matches!(
        format,
        MTL_FORMAT_RGBA8_UNORM_SRGB | MTL_FORMAT_BGRA8_UNORM_SRGB
    )
}

pub fn sampled_class(format: u16) -> Option<SampledClass> {
    Some(match format {
        MTL_FORMAT_A8_UNORM => SampledClass::A8Unorm,
        MTL_FORMAT_R8_UNORM => SampledClass::R8Unorm,
        MTL_FORMAT_RG8_UNORM => SampledClass::Rg8Unorm,
        MTL_FORMAT_RGBA8_UNORM | MTL_FORMAT_RGBA8_UNORM_SRGB => SampledClass::Rgba8Unorm,
        MTL_FORMAT_BGRA8_UNORM | MTL_FORMAT_BGRA8_UNORM_SRGB => SampledClass::Bgra8Unorm,
        MTL_FORMAT_RGBA16_FLOAT => SampledClass::Rgba16Float,
        _ => return None,
    })
}

pub fn storage_selector(format: u16) -> Option<(StorageImageSelector, u32)> {
    Some(match format {
        MTL_FORMAT_R8_UNORM => (StorageImageSelector::R8Unorm, R8_BPP),
        MTL_FORMAT_R32_UINT => (StorageImageSelector::R32Uint, R32_BPP),
        MTL_FORMAT_RG8_UNORM => (StorageImageSelector::Rg8Unorm, RG8_BPP),
        MTL_FORMAT_R16_FLOAT => (StorageImageSelector::R16Float, R16F_BPP),
        MTL_FORMAT_RG16_FLOAT => (StorageImageSelector::Rg16Float, RG16F_BPP),
        MTL_FORMAT_RGBA8_UNORM => (StorageImageSelector::Rgba8Unorm, RGBA8_BPP),
        MTL_FORMAT_BGRA8_UNORM => (StorageImageSelector::Bgra8Unorm, BGRA8_BPP),
        MTL_FORMAT_RGBA8_UINT => (StorageImageSelector::Rgba8Uint, RGBA8_BPP),
        MTL_FORMAT_RGBA8_SINT => (StorageImageSelector::Rgba8Sint, RGBA8_BPP),
        MTL_FORMAT_RGBA16_UINT => (StorageImageSelector::Rgba16Uint, RGBA16_BPP),
        MTL_FORMAT_RGBA16_FLOAT => (StorageImageSelector::Rgba16Float, RGBA16F_BPP),
        MTL_FORMAT_RGBA32_UINT => (StorageImageSelector::Rgba32Uint, RGBA32_BPP),
        MTL_FORMAT_RGBA32_FLOAT => (StorageImageSelector::Rgba32Float, RGBA32F_BPP),
        _ => return None,
    })
}

pub fn render_target_class(format: u16) -> Option<(RenderTargetClass, u32)> {
    Some(match format {
        // Metal color-renderable 8-bit + float16 family. sRGB variants share
        // storage bpp with their unorm counterparts (Metal texture view rules).
        MTL_FORMAT_RGBA8_UNORM => (RenderTargetClass::Rgba8Unorm, RGBA8_BPP),
        MTL_FORMAT_RGBA8_UNORM_SRGB => (RenderTargetClass::Rgba8UnormSrgb, RGBA8_BPP),
        MTL_FORMAT_BGRA8_UNORM => (RenderTargetClass::Bgra8Unorm, BGRA8_BPP),
        MTL_FORMAT_BGRA8_UNORM_SRGB => (RenderTargetClass::Bgra8UnormSrgb, BGRA8_BPP),
        MTL_FORMAT_RGBA16_FLOAT => (RenderTargetClass::Rgba16Float, RGBA16F_BPP),
        MTL_FORMAT_RG16_FLOAT => (RenderTargetClass::Rg16Float, RG16F_BPP),
        _ => return None,
    })
}

pub fn render_target_bpp(format: u16) -> Option<u32> {
    render_target_class(format).map(|(_, bpp)| bpp)
}

pub fn tight_row_bytes(width: u32, format: u16) -> Option<u32> {
    if width == 0 {
        return None;
    }
    let bpp = bytes_per_pixel(format)?;
    width.checked_mul(bpp)
}

pub fn row_bytes_aligned(width: u32, format: u16, alignment: u32) -> Option<u32> {
    if width == 0 || alignment == 0 {
        return None;
    }
    let bpp = bytes_per_pixel(format)?;
    let row = checked_mul_u64(width as u64, bpp as u64)?;
    let rem = row % alignment as u64;
    let row = if rem != 0 {
        row.checked_add(alignment as u64 - rem)?
    } else {
        row
    };
    if row > u32::MAX as u64 {
        None
    } else {
        Some(row as u32)
    }
}

pub fn iosurface_row_bytes(width: u32, format: u16) -> Option<u32> {
    if width == 0 {
        return None;
    }
    let (_, bpp) = render_target_class(format)?;
    let row = checked_mul_u64(width as u64, bpp as u64)?;
    let rem = row % IOSURFACE_ROW_ALIGNMENT as u64;
    let row = if rem != 0 {
        row.checked_add(IOSURFACE_ROW_ALIGNMENT as u64 - rem)?
    } else {
        row
    };
    if row > u32::MAX as u64 {
        None
    } else {
        Some(row as u32)
    }
}

pub fn tight_image_size(width: u32, height: u32, format: u16) -> Option<usize> {
    if width == 0 || height == 0 {
        return None;
    }
    let bpp = bytes_per_pixel(format)?;
    let pixels = checked_mul_u64(width as u64, height as u64)?;
    let bytes = checked_mul_u64(pixels, bpp as u64)?;
    usize::try_from(bytes).ok()
}

pub fn swizzle_identity() -> SwizzlePlan {
    SwizzlePlan {
        source: [
            SwizzleSource::R,
            SwizzleSource::G,
            SwizzleSource::B,
            SwizzleSource::A,
        ],
    }
}

fn swizzle_selector_source(selector: u8) -> Option<SwizzleSource> {
    Some(match selector {
        0 => SwizzleSource::Zero,
        1 => SwizzleSource::One,
        2 => SwizzleSource::R,
        3 => SwizzleSource::G,
        4 => SwizzleSource::B,
        5 => SwizzleSource::A,
        _ => return None,
    })
}

pub fn swizzle_plan(raw: &[u8; COMPONENT_COUNT]) -> Option<SwizzlePlan> {
    let mut source = [SwizzleSource::Zero; COMPONENT_COUNT];
    for i in 0..COMPONENT_COUNT {
        source[i] = swizzle_selector_source(raw[i])?;
    }
    Some(SwizzlePlan { source })
}

pub fn swizzle_is_identity(plan: &SwizzlePlan) -> bool {
    plan.source
        == [
            SwizzleSource::R,
            SwizzleSource::G,
            SwizzleSource::B,
            SwizzleSource::A,
        ]
}

pub fn swizzle_word(raw: &[u8; COMPONENT_COUNT]) -> u32 {
    u32::from(raw[0])
        | (u32::from(raw[1]) << 8)
        | (u32::from(raw[2]) << 16)
        | (u32::from(raw[3]) << 24)
}

pub fn apply_swizzle_rgba8(plan: &SwizzlePlan, in_rgba: [u8; 4]) -> [u8; 4] {
    let mut out = [0u8; 4];
    for (component, source) in out.iter_mut().zip(plan.source) {
        *component = match source {
            SwizzleSource::Zero => UNORM8_MIN,
            SwizzleSource::One => UNORM8_MAX,
            SwizzleSource::R => in_rgba[COMPONENT_R],
            SwizzleSource::G => in_rgba[COMPONENT_G],
            SwizzleSource::B => in_rgba[COMPONENT_B],
            SwizzleSource::A => in_rgba[COMPONENT_A],
        };
    }
    out
}

pub fn f64_to_unorm8(value: f64) -> u8 {
    if !matches!(value.partial_cmp(&0.0), Some(std::cmp::Ordering::Greater)) {
        UNORM8_MIN
    } else if value >= 1.0 {
        UNORM8_MAX
    } else {
        (value * f64::from(UNORM8_MAX) + 0.5) as u8
    }
}

pub fn f16_to_f32(half_bits: u16) -> f32 {
    let sign = (u32::from(half_bits & F16_SIGN_MASK)) << F16_F32_SIGN_SHIFT;
    let exp = (u32::from(half_bits) >> F16_EXP_SHIFT) & F16_EXP_MASK;
    let mut mant = u32::from(half_bits) & F16_MANT_MASK;
    let bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            let mut normal_exp: i32 = 1;
            while (mant & F16_HIDDEN_BIT) == 0 {
                mant <<= 1;
                normal_exp -= 1;
            }
            mant &= F16_MANT_MASK;
            sign | (((normal_exp - F16_EXP_BIAS + F32_EXP_BIAS) as u32) << F32_EXP_SHIFT)
                | (mant << F16_F32_MANT_SHIFT)
        }
    } else if exp == F16_EXP_INF_NAN {
        sign | F32_INF_BITS | (mant << F16_F32_MANT_SHIFT)
    } else {
        let f32_exp = (exp as i32 - F16_EXP_BIAS + F32_EXP_BIAS) as u32;
        sign | (f32_exp << F32_EXP_SHIFT) | (mant << F16_F32_MANT_SHIFT)
    };
    f32::from_bits(bits)
}

fn build_f16_to_unorm8_lut() -> Box<[u8; 65536]> {
    let mut lut = Box::new([0u8; 65536]);
    for i in 0..=u16::MAX {
        let f = f16_to_f32(i);
        lut[i as usize] = if !matches!(f.partial_cmp(&0.0), Some(std::cmp::Ordering::Greater)) {
            UNORM8_MIN
        } else if f >= 1.0 {
            UNORM8_MAX
        } else {
            (f * f32::from(UNORM8_MAX) + 0.5) as u8
        };
    }
    lut
}

fn f16_to_unorm8_lut() -> &'static [u8; 65536] {
    use std::sync::OnceLock;
    static LUT: OnceLock<Box<[u8; 65536]>> = OnceLock::new();
    LUT.get_or_init(build_f16_to_unorm8_lut)
}

pub fn f16_to_unorm8(half_bits: u16) -> u8 {
    f16_to_unorm8_lut()[half_bits as usize]
}

fn unorm8_to_f16_slow(value: u8) -> u16 {
    let f = f32::from(value) / f32::from(UNORM8_MAX);
    let x = f.to_bits();
    let sign = ((x >> F16_F32_SIGN_SHIFT) as u16) & F16_SIGN_MASK;
    let e = ((x >> F32_EXP_SHIFT) & F32_EXP_MASK) as i32 - F32_EXP_BIAS + F16_EXP_BIAS;
    let mut m = x & F32_MANT_MASK;

    if f <= 0.0 {
        return sign;
    }
    if e >= F16_EXP_INF_NAN as i32 {
        return sign | F16_INF_BITS;
    }
    if e <= 0 {
        if e < F16_SUBNORMAL_EXP_MIN {
            return sign;
        }
        m |= F32_HIDDEN_BIT;
        let shift = (F16_SUBNORMAL_SHIFT_BASE - e) as u32;
        let mut hm = m >> shift;
        if ((m >> (shift - 1)) & 1) != 0 {
            hm += 1;
        }
        return sign | (hm as u16);
    }

    let mut h = sign | (((e as u32) << F16_EXP_SHIFT) as u16) | ((m >> F16_F32_MANT_SHIFT) as u16);
    if (m & F32_TO_F16_ROUND_BIT) != 0 {
        h = h.wrapping_add(1);
    }
    h
}

fn unorm8_to_f16_lut() -> &'static [u16; 256] {
    use std::sync::OnceLock;
    static LUT: OnceLock<[u16; 256]> = OnceLock::new();
    LUT.get_or_init(|| {
        let mut lut = [0u16; 256];
        for i in 0..=UNORM8_MAX {
            lut[i as usize] = unorm8_to_f16_slow(i);
        }
        lut
    })
}

pub fn unorm8_to_f16(value: u8) -> u16 {
    unorm8_to_f16_lut()[value as usize]
}

pub fn texel_to_rgba8(format: u16, src: &[u8]) -> Option<[u8; 4]> {
    let bpp = bytes_per_pixel(format)? as usize;
    if src.len() < bpp {
        return None;
    }
    let mut rgba = [0u8; 4];
    match format {
        MTL_FORMAT_A8_UNORM => {
            rgba[COMPONENT_A] = src[0];
        }
        MTL_FORMAT_R8_UNORM => {
            rgba[COMPONENT_R] = src[0];
            rgba[COMPONENT_A] = UNORM8_MAX;
        }
        MTL_FORMAT_RG8_UNORM => {
            rgba[COMPONENT_R] = src[0];
            rgba[COMPONENT_G] = src[1];
            rgba[COMPONENT_A] = UNORM8_MAX;
        }
        MTL_FORMAT_RGBA8_UNORM | MTL_FORMAT_RGBA8_UNORM_SRGB => {
            rgba.copy_from_slice(&src[..4]);
        }
        MTL_FORMAT_BGRA8_UNORM | MTL_FORMAT_BGRA8_UNORM_SRGB => {
            rgba[COMPONENT_R] = src[2];
            rgba[COMPONENT_G] = src[1];
            rgba[COMPONENT_B] = src[0];
            rgba[COMPONENT_A] = src[3];
        }
        MTL_FORMAT_RGBA16_FLOAT => {
            let lut = f16_to_unorm8_lut();
            rgba[COMPONENT_R] = lut[ld16(&src[0..2]) as usize];
            rgba[COMPONENT_G] = lut[ld16(&src[2..4]) as usize];
            rgba[COMPONENT_B] = lut[ld16(&src[4..6]) as usize];
            rgba[COMPONENT_A] = lut[ld16(&src[6..8]) as usize];
        }
        MTL_FORMAT_RG16_FLOAT => {
            // Two float16 channels → R,G; B has no source (0), A opaque. Mirrors
            // the RGBA16Float LUT path (values clamp to [0,1] through the u8 LUT).
            let lut = f16_to_unorm8_lut();
            rgba[COMPONENT_R] = lut[ld16(&src[0..2]) as usize];
            rgba[COMPONENT_G] = lut[ld16(&src[2..4]) as usize];
            rgba[COMPONENT_A] = UNORM8_MAX;
        }
        _ => return None,
    }
    Some(rgba)
}

pub fn rgba8_to_texel(format: u16, rgba: [u8; 4], dst: &mut [u8]) -> bool {
    let Some(bpp) = bytes_per_pixel(format) else {
        return false;
    };
    if dst.len() < bpp as usize {
        return false;
    }
    match format {
        MTL_FORMAT_RGBA8_UNORM | MTL_FORMAT_RGBA8_UNORM_SRGB => {
            dst[..4].copy_from_slice(&rgba);
        }
        MTL_FORMAT_BGRA8_UNORM | MTL_FORMAT_BGRA8_UNORM_SRGB => {
            dst[0] = rgba[COMPONENT_B];
            dst[1] = rgba[COMPONENT_G];
            dst[2] = rgba[COMPONENT_R];
            dst[3] = rgba[COMPONENT_A];
        }
        MTL_FORMAT_RGBA16_FLOAT => {
            let lut = unorm8_to_f16_lut();
            st16(&mut dst[0..2], lut[rgba[COMPONENT_R] as usize]);
            st16(&mut dst[2..4], lut[rgba[COMPONENT_G] as usize]);
            st16(&mut dst[4..6], lut[rgba[COMPONENT_B] as usize]);
            st16(&mut dst[6..8], lut[rgba[COMPONENT_A] as usize]);
        }
        MTL_FORMAT_RG16_FLOAT => {
            // R,G → two float16 channels; B,A have no destination (RG16 is
            // 4 bytes). Inverse of the texel_to_rgba8 RG16Float path.
            let lut = unorm8_to_f16_lut();
            st16(&mut dst[0..2], lut[rgba[COMPONENT_R] as usize]);
            st16(&mut dst[2..4], lut[rgba[COMPONENT_G] as usize]);
        }
        _ => return false,
    }
    true
}

fn row_walk_backward(
    src_len: usize,
    src_stride: usize,
    dst_len: usize,
    dst_stride: usize,
    same_base: bool,
) -> Option<bool> {
    // Non-overlapping or zero lengths: forward.
    if src_len == 0 || dst_len == 0 {
        return Some(false);
    }
    // We cannot detect true pointer overlap without raw pointers; for Rust
    // slice APIs we only allow in-place when same_base is true (caller asserts
    // src and dst alias the same allocation).
    if !same_base {
        return Some(false);
    }
    Some(dst_stride > src_stride)
}

pub fn convert_row_to_rgba8(format: u16, src: &[u8], pixels: u32, dst_rgba: &mut [u8]) -> bool {
    convert_row_to_rgba8_ex(format, src, pixels, dst_rgba, false)
}

pub fn convert_row_to_rgba8_inplace(format: u16, buf: &mut [u8], pixels: u32, bpp: u32) -> bool {
    // In-place expand: process backward if bpp < 4.
    let src_len = (pixels as usize).checked_mul(bpp as usize).unwrap_or(0);
    let dst_len = (pixels as usize)
        .checked_mul(RGBA8_BPP as usize)
        .unwrap_or(0);
    if buf.len() < dst_len.max(src_len) {
        return false;
    }
    // Copy src first into temporary when expanding in place.
    let src_owned = buf[..src_len].to_vec();
    convert_row_to_rgba8(format, &src_owned, pixels, &mut buf[..dst_len])
}

fn convert_row_to_rgba8_ex(
    format: u16,
    src: &[u8],
    pixels: u32,
    dst_rgba: &mut [u8],
    same_base: bool,
) -> bool {
    if pixels == 0 {
        return true;
    }
    let Some(bpp) = bytes_per_pixel(format) else {
        return false;
    };
    let src_len = match (pixels as u64).checked_mul(bpp as u64) {
        Some(v) => v as usize,
        None => return false,
    };
    let dst_len = match (pixels as u64).checked_mul(RGBA8_BPP as u64) {
        Some(v) => v as usize,
        None => return false,
    };
    if src.len() < src_len || dst_rgba.len() < dst_len {
        return false;
    }
    let Some(backward) = row_walk_backward(
        src_len,
        bpp as usize,
        dst_len,
        RGBA8_BPP as usize,
        same_base,
    ) else {
        return false;
    };

    if format == MTL_FORMAT_RGBA16_FLOAT {
        let lut = f16_to_unorm8_lut();
        let iter: Box<dyn Iterator<Item = u32>> = if backward {
            Box::new((0..pixels).rev())
        } else {
            Box::new(0..pixels)
        };
        for i in iter {
            let sp = (i as usize) * RGBA16F_BPP as usize;
            let dp = (i as usize) * RGBA8_BPP as usize;
            dst_rgba[dp + COMPONENT_R] = lut[ld16(&src[sp..sp + 2]) as usize];
            dst_rgba[dp + COMPONENT_G] = lut[ld16(&src[sp + 2..sp + 4]) as usize];
            dst_rgba[dp + COMPONENT_B] = lut[ld16(&src[sp + 4..sp + 6]) as usize];
            dst_rgba[dp + COMPONENT_A] = lut[ld16(&src[sp + 6..sp + 8]) as usize];
        }
        return true;
    }

    let iter: Box<dyn Iterator<Item = u32>> = if backward {
        Box::new((0..pixels).rev())
    } else {
        Box::new(0..pixels)
    };
    for i in iter {
        let sp = (i as usize) * bpp as usize;
        let dp = (i as usize) * RGBA8_BPP as usize;
        let Some(rgba) = texel_to_rgba8(format, &src[sp..sp + bpp as usize]) else {
            return false;
        };
        dst_rgba[dp..dp + 4].copy_from_slice(&rgba);
    }
    true
}

pub fn convert_rgba8_to_row(format: u16, src_rgba: &[u8], pixels: u32, dst: &mut [u8]) -> bool {
    if pixels == 0 {
        return true;
    }
    let Some(bpp) = bytes_per_pixel(format) else {
        return false;
    };
    let src_len = match (pixels as u64).checked_mul(RGBA8_BPP as u64) {
        Some(v) => v as usize,
        None => return false,
    };
    let dst_len = match (pixels as u64).checked_mul(bpp as u64) {
        Some(v) => v as usize,
        None => return false,
    };
    if src_rgba.len() < src_len || dst.len() < dst_len {
        return false;
    }

    if format == MTL_FORMAT_RGBA16_FLOAT {
        let lut = unorm8_to_f16_lut();
        for i in 0..pixels {
            let sp = (i as usize) * RGBA8_BPP as usize;
            let dp = (i as usize) * RGBA16F_BPP as usize;
            st16(
                &mut dst[dp..dp + 2],
                lut[src_rgba[sp + COMPONENT_R] as usize],
            );
            st16(
                &mut dst[dp + 2..dp + 4],
                lut[src_rgba[sp + COMPONENT_G] as usize],
            );
            st16(
                &mut dst[dp + 4..dp + 6],
                lut[src_rgba[sp + COMPONENT_B] as usize],
            );
            st16(
                &mut dst[dp + 6..dp + 8],
                lut[src_rgba[sp + COMPONENT_A] as usize],
            );
        }
        return true;
    }

    for i in 0..pixels {
        let sp = (i as usize) * RGBA8_BPP as usize;
        let dp = (i as usize) * bpp as usize;
        let mut rgba = [0u8; 4];
        rgba.copy_from_slice(&src_rgba[sp..sp + 4]);
        if !rgba8_to_texel(format, rgba, &mut dst[dp..dp + bpp as usize]) {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bytes_per_pixel_matrix() {
        let cases = [
            (MTL_FORMAT_A8_UNORM, 1),
            (MTL_FORMAT_R8_UNORM, 1),
            (MTL_FORMAT_R16_FLOAT, 2),
            (MTL_FORMAT_RG8_UNORM, 2),
            (MTL_FORMAT_R32_UINT, 4),
            (MTL_FORMAT_R32_SINT, 4),
            (MTL_FORMAT_R32_FLOAT, 4),
            (MTL_FORMAT_RG16_FLOAT, 4),
            (MTL_FORMAT_RGBA8_UNORM, 4),
            (MTL_FORMAT_RGBA8_UNORM_SRGB, 4),
            (MTL_FORMAT_RGBA8_UINT, 4),
            (MTL_FORMAT_RGBA8_SINT, 4),
            (MTL_FORMAT_BGRA8_UNORM, 4),
            (MTL_FORMAT_BGRA8_UNORM_SRGB, 4),
            (MTL_FORMAT_RGB9E5_FLOAT, 4),
            (MTL_FORMAT_RGBA16_UINT, 8),
            (MTL_FORMAT_RGBA16_FLOAT, 8),
            (MTL_FORMAT_RGBA32_UINT, 16),
            (MTL_FORMAT_RGBA32_FLOAT, 16),
            (MTL_FORMAT_DEPTH16_UNORM, 2),
            (MTL_FORMAT_DEPTH32_FLOAT, 4),
            (MTL_FORMAT_STENCIL8, 1),
            (MTL_FORMAT_DEPTH24_UNORM_STENCIL8, 4),
            (MTL_FORMAT_DEPTH32_FLOAT_STENCIL8, 8),
            (MTL_FORMAT_X32_STENCIL8, 8),
            (MTL_FORMAT_X24_STENCIL8, 4),
        ];
        for (fmt, bpp) in cases {
            assert_eq!(bytes_per_pixel(fmt), Some(bpp));
        }
        assert_eq!(bytes_per_pixel(0xffff), None);
    }

    #[test]
    fn blit_aspect_bpp_depth_stencil() {
        // Pure depth + depth option.
        assert_eq!(
            blit_aspect_bytes_per_pixel(MTL_FORMAT_DEPTH32_FLOAT, true, false),
            Some(4)
        );
        assert_eq!(
            blit_aspect_bytes_per_pixel(MTL_FORMAT_DEPTH32_FLOAT, false, false),
            Some(4)
        );
        // Pure depth cannot take stencil option.
        assert_eq!(
            blit_aspect_bytes_per_pixel(MTL_FORMAT_DEPTH32_FLOAT, false, true),
            None
        );
        // Pure stencil.
        assert_eq!(
            blit_aspect_bytes_per_pixel(MTL_FORMAT_STENCIL8, false, true),
            Some(1)
        );
        // Combined: depth plane 4 B, stencil 1 B, full = packing full_bpp.
        assert_eq!(
            blit_aspect_bytes_per_pixel(MTL_FORMAT_DEPTH32_FLOAT_STENCIL8, true, false),
            Some(4)
        );
        assert_eq!(
            blit_aspect_bytes_per_pixel(MTL_FORMAT_DEPTH32_FLOAT_STENCIL8, false, true),
            Some(1)
        );
        assert_eq!(
            blit_aspect_bytes_per_pixel(MTL_FORMAT_DEPTH32_FLOAT_STENCIL8, false, false),
            Some(8)
        );
        assert!(blit_aspect_needs_repack(
            MTL_FORMAT_DEPTH32_FLOAT_STENCIL8,
            true,
            false
        ));
        assert!(!blit_aspect_needs_repack(
            MTL_FORMAT_DEPTH32_FLOAT,
            true,
            false
        ));
        // Color cannot take DS options.
        assert_eq!(
            blit_aspect_bytes_per_pixel(MTL_FORMAT_BGRA8_UNORM, true, false),
            None
        );
    }

    /// Exactly the two `_SRGB` wire values carry the transfer function, and the
    /// class lookups fold each onto the linear sibling's byte layout. Both
    /// halves matter: the fold is what makes a class usable, `is_srgb` is what
    /// keeps the fold from being a silent loss.
    #[test]
    fn srgb_is_named_beside_the_class_that_folds_it() {
        assert!(is_srgb(MTL_FORMAT_RGBA8_UNORM_SRGB));
        assert!(is_srgb(MTL_FORMAT_BGRA8_UNORM_SRGB));
        for fmt in [
            MTL_FORMAT_RGBA8_UNORM,
            MTL_FORMAT_BGRA8_UNORM,
            MTL_FORMAT_A8_UNORM,
            MTL_FORMAT_RGBA16_FLOAT,
            MTL_FORMAT_DEPTH32_FLOAT,
            0xffff,
        ] {
            assert!(!is_srgb(fmt), "{fmt:#x}");
        }
        assert_eq!(
            sampled_class(MTL_FORMAT_RGBA8_UNORM_SRGB),
            sampled_class(MTL_FORMAT_RGBA8_UNORM)
        );
        assert_eq!(
            sampled_class(MTL_FORMAT_BGRA8_UNORM_SRGB),
            sampled_class(MTL_FORMAT_BGRA8_UNORM)
        );
        // The render-target classes do NOT fold — they keep the qualifier.
        assert_ne!(
            render_target_class(MTL_FORMAT_RGBA8_UNORM_SRGB),
            render_target_class(MTL_FORMAT_RGBA8_UNORM)
        );
    }

    #[test]
    fn sampled_and_storage() {
        assert_eq!(
            sampled_class(MTL_FORMAT_A8_UNORM),
            Some(SampledClass::A8Unorm)
        );
        assert_eq!(sampled_class(MTL_FORMAT_R16_FLOAT), None);
        assert_eq!(
            storage_selector(MTL_FORMAT_R8_UNORM),
            Some((StorageImageSelector::R8Unorm, 1))
        );
        assert_eq!(storage_selector(MTL_FORMAT_A8_UNORM), None);
        // R32Uint is storage-capable (specialized to the R32ui storage path);
        // its single-channel sint/float siblings are not.
        assert_eq!(
            storage_selector(MTL_FORMAT_R32_UINT),
            Some((StorageImageSelector::R32Uint, R32_BPP))
        );
        assert_eq!(storage_selector(MTL_FORMAT_R32_SINT), None);
        assert_eq!(storage_selector(MTL_FORMAT_R32_FLOAT), None);
        assert_eq!(
            render_target_class(MTL_FORMAT_BGRA8_UNORM),
            Some((RenderTargetClass::Bgra8Unorm, 4))
        );
        assert_eq!(
            render_target_class(MTL_FORMAT_RGBA8_UNORM),
            Some((RenderTargetClass::Rgba8Unorm, 4))
        );
        assert_eq!(
            render_target_class(MTL_FORMAT_RGBA8_UNORM_SRGB),
            Some((RenderTargetClass::Rgba8UnormSrgb, 4))
        );
        assert_eq!(
            render_target_class(MTL_FORMAT_BGRA8_UNORM_SRGB),
            Some((RenderTargetClass::Bgra8UnormSrgb, 4))
        );
        assert_eq!(
            render_target_class(MTL_FORMAT_RGBA16_FLOAT),
            Some((RenderTargetClass::Rgba16Float, 8))
        );
        assert_eq!(
            render_target_class(MTL_FORMAT_RG16_FLOAT),
            Some((RenderTargetClass::Rg16Float, 4))
        );
        // Integer / non-color formats stay fail-closed.
        assert_eq!(render_target_class(MTL_FORMAT_RGBA8_UINT), None);
        assert_eq!(render_target_class(MTL_FORMAT_R8_UNORM), None);
    }

    /// RG16Float MRT slots (vibrancy UI tile masks) must admit as color RTs so
    /// `mrt_draw_request` no longer drops the whole pass. Two channels survive
    /// the RGBA8-intermediate round trip; B has no source and A is opaque.
    #[test]
    fn rg16float_render_target_roundtrips_two_channels() {
        assert_eq!(render_target_bpp(MTL_FORMAT_RG16_FLOAT), Some(4));
        let w = 16u32;
        let mut rgba = vec![0u8; (w as usize) * 4];
        for i in 0..(w as usize) {
            rgba[i * 4] = 40; // R
            rgba[i * 4 + 1] = 90; // G
            rgba[i * 4 + 2] = 200; // B (dropped by RG16)
            rgba[i * 4 + 3] = 128; // A (dropped by RG16)
        }
        let tight = tight_row_bytes(w, MTL_FORMAT_RG16_FLOAT).unwrap();
        assert_eq!(tight, w * 4);
        let mut native = vec![0u8; tight as usize];
        assert!(convert_rgba8_to_row(
            MTL_FORMAT_RG16_FLOAT,
            &rgba,
            w,
            &mut native
        ));
        let mut back = vec![0u8; (w as usize) * 4];
        assert!(convert_row_to_rgba8(
            MTL_FORMAT_RG16_FLOAT,
            &native,
            w,
            &mut back
        ));
        // R,G round-trip through the u8→f16→u8 LUT; B has no source (0); A opaque.
        assert_eq!(back[0], 40);
        assert_eq!(back[1], 90);
        assert_eq!(back[2], 0);
        assert_eq!(back[3], 255);
    }

    /// Metal color-renderable 8-bit + f16 set used as Reims VGPU pass attachments.
    /// Bring-up only admitted BGRA8/RGBA16F (compositor FBs); apps use RGBA8.
    #[test]
    fn color_renderable_formats_admit_app_rts() {
        for (fmt, bpp) in [
            (MTL_FORMAT_RGBA8_UNORM, 4u32),
            (MTL_FORMAT_RGBA8_UNORM_SRGB, 4),
            (MTL_FORMAT_BGRA8_UNORM, 4),
            (MTL_FORMAT_BGRA8_UNORM_SRGB, 4),
            (MTL_FORMAT_RGBA16_FLOAT, 8),
        ] {
            assert_eq!(render_target_bpp(fmt), Some(bpp), "fmt={fmt:#x}");
            // Round-trip tight row for write_gva / mapping store.
            let w = 16u32;
            let mut rgba = vec![0u8; (w as usize) * 4];
            for i in 0..(w as usize) {
                rgba[i * 4] = 10;
                rgba[i * 4 + 1] = 20;
                rgba[i * 4 + 2] = 30;
                rgba[i * 4 + 3] = 255;
            }
            let tight = tight_row_bytes(w, fmt).unwrap();
            let mut native = vec![0u8; tight as usize];
            assert!(
                convert_rgba8_to_row(fmt, &rgba, w, &mut native),
                "convert host RGBA8 → guest fmt={fmt:#x}"
            );
            let mut back = vec![0u8; (w as usize) * 4];
            assert!(
                convert_row_to_rgba8(fmt, &native, w, &mut back),
                "convert guest fmt={fmt:#x} → host RGBA8"
            );
            // 8-bit unorm/sRGB and float16 (via unorm8 LUT) keep the solid color.
            assert_eq!(back[0], 10);
            assert_eq!(back[1], 20);
            assert_eq!(back[2], 30);
            assert_eq!(back[3], 255);
        }
    }

    #[test]
    fn rows_and_image_size() {
        assert_eq!(iosurface_row_bytes(200, MTL_FORMAT_BGRA8_UNORM), Some(896));
        assert_eq!(iosurface_row_bytes(64, MTL_FORMAT_BGRA8_UNORM), Some(256));
        assert_eq!(iosurface_row_bytes(250, MTL_FORMAT_BGRA8_UNORM), Some(1024));
        assert_eq!(
            iosurface_row_bytes(200, MTL_FORMAT_RGBA16_FLOAT),
            Some(1664)
        );
        assert_eq!(iosurface_row_bytes(0, MTL_FORMAT_BGRA8_UNORM), None);
        // Same 4 Bpp packing as BGRA8 → same 128 B aligned row for w=200.
        assert_eq!(iosurface_row_bytes(200, MTL_FORMAT_RGBA8_UNORM), Some(896));
        assert_eq!(tight_row_bytes(200, MTL_FORMAT_BGRA8_UNORM), Some(800));
        assert_eq!(row_bytes_aligned(3, MTL_FORMAT_RGBA32_FLOAT, 64), Some(64));
        assert_eq!(tight_image_size(4, 3, MTL_FORMAT_RGBA8_UNORM), Some(48));
        assert_eq!(
            tight_image_size(u32::MAX, u32::MAX, MTL_FORMAT_RGBA32_FLOAT),
            None
        );
    }

    #[test]
    fn swizzle_and_texels() {
        let plan = swizzle_plan(&[2, 3, 4, 5]).unwrap();
        assert!(swizzle_is_identity(&plan));
        assert_eq!(swizzle_word(&[2, 3, 4, 5]), 0x05040302);
        let bgra = [10u8, 20, 30, 40];
        let rgba = texel_to_rgba8(MTL_FORMAT_BGRA8_UNORM, &bgra).unwrap();
        assert_eq!(rgba, [30, 20, 10, 40]);
        let mut out = [0u8; 4];
        assert!(rgba8_to_texel(MTL_FORMAT_BGRA8_UNORM, rgba, &mut out));
        assert_eq!(out, bgra);

        // f16 round-trip identity for unorm8 extremes
        assert_eq!(f16_to_unorm8(unorm8_to_f16(0)), 0);
        assert_eq!(f16_to_unorm8(unorm8_to_f16(255)), 255);
        assert_eq!(f16_to_unorm8(unorm8_to_f16(128)), 128);

        let row = [1u8, 2, 3, 4, 5, 6, 7, 8];
        let mut rgba_row = [0u8; 8];
        assert!(convert_row_to_rgba8(
            MTL_FORMAT_RGBA8_UNORM,
            &row,
            2,
            &mut rgba_row
        ));
        assert_eq!(rgba_row, row);
        let mut back = [0u8; 8];
        assert!(convert_rgba8_to_row(
            MTL_FORMAT_RGBA8_UNORM,
            &rgba_row,
            2,
            &mut back
        ));
        assert_eq!(back, row);

        // property: random unorm8 -> f16 -> unorm8 stable for all bytes
        for v in 0u8..=255 {
            assert_eq!(f16_to_unorm8(unorm8_to_f16(v)), v);
        }
        let _ = convert_row_to_rgba8_inplace;
        let _ = f64_to_unorm8(0.5);
        let _ = f16_to_f32(0x3c00); // 1.0
    }

    #[test]
    fn unsupported_fail_closed() {
        // Unknown formats fail closed. Depth/stencil families have bpp for blit
        // packing but remain unsampled / non-storage / non-RT.
        for fmt in [0xffffu16, 130, 204] {
            assert!(bytes_per_pixel(fmt).is_none());
            assert!(sampled_class(fmt).is_none());
            assert!(storage_selector(fmt).is_none());
            assert!(render_target_class(fmt).is_none());
            assert!(texel_to_rgba8(fmt, &[0; 16]).is_none());
        }
        for fmt in [
            MTL_FORMAT_DEPTH32_FLOAT,
            MTL_FORMAT_STENCIL8,
            MTL_FORMAT_DEPTH32_FLOAT_STENCIL8,
            MTL_FORMAT_DEPTH24_UNORM_STENCIL8,
        ] {
            assert!(bytes_per_pixel(fmt).is_some());
            assert!(sampled_class(fmt).is_none());
            assert!(storage_selector(fmt).is_none());
            assert!(render_target_class(fmt).is_none());
            assert!(texel_to_rgba8(fmt, &[0; 16]).is_none());
        }
    }

    #[test]
    fn depth32_stencil8_plane_roundtrip() {
        let fmt = MTL_FORMAT_DEPTH32_FLOAT_STENCIL8;
        let p = depth_stencil_packing(fmt).unwrap();
        assert_eq!(p.full_bpp, 8);
        // Depth = 1.0f32, stencil = 0xAB
        let mut texel = [0u8; 8];
        texel[0..4].copy_from_slice(&1.0f32.to_bits().to_le_bytes());
        texel[4] = 0xab;
        let mut depth = [0u8; 4];
        assert!(extract_depth_stencil_plane(
            fmt, true, false, &texel, &mut depth
        ));
        assert_eq!(depth, 1.0f32.to_bits().to_le_bytes());
        let mut st = [0u8; 1];
        assert!(extract_depth_stencil_plane(
            fmt, false, true, &texel, &mut st
        ));
        assert_eq!(st[0], 0xab);
        // Insert new depth, keep stencil.
        let mut t2 = texel;
        let new_d = 0.5f32.to_bits().to_le_bytes();
        assert!(insert_depth_stencil_plane(
            fmt, true, false, &new_d, &mut t2
        ));
        assert_eq!(t2[4], 0xab);
        let mut d2 = [0u8; 4];
        assert!(extract_depth_stencil_plane(fmt, true, false, &t2, &mut d2));
        assert_eq!(d2, new_d);
        // Row extract 2 pixels.
        let mut row = [0u8; 16];
        row[..8].copy_from_slice(&texel);
        row[8..16].copy_from_slice(&t2);
        let mut planes = [0u8; 8];
        assert!(extract_plane_row(fmt, true, false, &row, 2, &mut planes));
        assert_eq!(&planes[0..4], &1.0f32.to_bits().to_le_bytes());
        assert_eq!(&planes[4..8], &new_d);
    }

    #[test]
    fn depth24_stencil8_plane_pack() {
        let fmt = MTL_FORMAT_DEPTH24_UNORM_STENCIL8;
        // stencil=0x11, depth24=0xAABBCC → packed LE
        let depth24 = 0x00aabbccu32;
        let packed = 0x11u32 | (depth24 << 8);
        let texel = packed.to_le_bytes();
        let mut depth = [0u8; 4];
        assert!(extract_depth_stencil_plane(
            fmt, true, false, &texel, &mut depth
        ));
        assert_eq!(u32::from_le_bytes(depth), depth24);
        let mut st = [0u8; 1];
        assert!(extract_depth_stencil_plane(
            fmt, false, true, &texel, &mut st
        ));
        assert_eq!(st[0], 0x11);
        let mut t2 = [0u8; 4];
        assert!(insert_depth_stencil_plane(
            fmt,
            false,
            true,
            &[0x22],
            &mut t2
        ));
        assert!(insert_depth_stencil_plane(
            fmt,
            true,
            false,
            &depth24.to_le_bytes(),
            &mut t2
        ));
        assert_eq!(u32::from_le_bytes(t2), 0x22 | (depth24 << 8));
    }

    #[test]
    fn property_fuzz_row_roundtrip_rgba8() {
        // Corpus-driven property: random-ish patterns through BGRA/RGBA convert.
        let patterns: &[&[u8]] = &[
            &[0, 0, 0, 0],
            &[255, 255, 255, 255],
            &[1, 2, 3, 4],
            &[10, 20, 30, 40, 50, 60, 70, 80],
            &[0xde, 0xad, 0xbe, 0xef, 0xca, 0xfe, 0xba, 0xbe],
        ];
        for p in patterns {
            let pixels = (p.len() / 4) as u32;
            if pixels == 0 {
                continue;
            }
            let mut rgba = vec![0u8; p.len()];
            assert!(convert_row_to_rgba8(
                MTL_FORMAT_RGBA8_UNORM,
                p,
                pixels,
                &mut rgba
            ));
            let mut back = vec![0u8; p.len()];
            assert!(convert_rgba8_to_row(
                MTL_FORMAT_RGBA8_UNORM,
                &rgba,
                pixels,
                &mut back
            ));
            assert_eq!(&back[..], *p);
        }
    }
}
