//! `MTLBlendFactor` / `MTLBlendOperation` → the engine's blend state →
//! `VkBlendFactor` / `VkBlendOp`.
//!
//! Both halves of the crossing live here: the wire value → engine enum decode,
//! and the engine enum → Vulkan spelling. Keeping them apart is how a blend
//! factor ends up decoded on one path and spelled differently on another.

use ash::vk;

use super::reason::TranslateReason;
use crate::backend::vulkan::engine::{BlendFactor, BlendOp, BlendStateResource};
use crate::runtime::decode::resource::{
    ColorWriteMask, MTL_COLOR_WRITE_MASK_ALPHA, MTL_COLOR_WRITE_MASK_BLUE,
    MTL_COLOR_WRITE_MASK_GREEN, MTL_COLOR_WRITE_MASK_RED,
};

/// `MTLBlendFactor` (SDK numeric values, Metal header order).
pub fn factor(mtl: u32) -> Result<BlendFactor, TranslateReason> {
    Ok(match mtl {
        0 => BlendFactor::Zero,
        1 => BlendFactor::One,
        2 => BlendFactor::SrcColor,
        3 => BlendFactor::OneMinusSrcColor,
        4 => BlendFactor::SrcAlpha,
        5 => BlendFactor::OneMinusSrcAlpha,
        6 => BlendFactor::DstColor,
        7 => BlendFactor::OneMinusDstColor,
        8 => BlendFactor::DstAlpha,
        9 => BlendFactor::OneMinusDstAlpha,
        10 => BlendFactor::SrcAlphaSaturated,
        11 => BlendFactor::ConstantColor,
        12 => BlendFactor::OneMinusConstantColor,
        13 => BlendFactor::ConstantAlpha,
        14 => BlendFactor::OneMinusConstantAlpha,
        other => return Err(TranslateReason::UnknownBlendFactor(other)),
    })
}

/// `MTLBlendOperation` (SDK numeric values).
pub fn operation(mtl: u32) -> Result<BlendOp, TranslateReason> {
    Ok(match mtl {
        0 => BlendOp::Add,
        1 => BlendOp::Subtract,
        2 => BlendOp::ReverseSubtract,
        3 => BlendOp::Min,
        4 => BlendOp::Max,
        other => return Err(TranslateReason::UnknownBlendOperation(other)),
    })
}

/// A whole decoded type-7 colour-attachment blend descriptor.
///
/// Fails on the first unrepresentable component rather than substituting a
/// default for it — a blend that silently becomes `ONE, ZERO` is a rendering
/// bug with no log line.
#[allow(clippy::too_many_arguments)]
pub fn state(
    src_rgb: u32,
    dst_rgb: u32,
    op_rgb: u32,
    src_alpha: u32,
    dst_alpha: u32,
    op_alpha: u32,
    constants: [f32; 4],
) -> Result<BlendStateResource, TranslateReason> {
    Ok(BlendStateResource {
        src_color: factor(src_rgb)?,
        dst_color: factor(dst_rgb)?,
        color_op: operation(op_rgb)?,
        src_alpha: factor(src_alpha)?,
        dst_alpha: factor(dst_alpha)?,
        alpha_op: operation(op_alpha)?,
        constants,
    })
}

/// `MTLColorWriteMask` → `VkColorComponentFlags`.
///
/// Metal's bits run alpha-first from the low end (`alpha = 1 << 0` …
/// `red = 1 << 3`); Vulkan's run red-first (`R = 1 << 0` … `A = 1 << 3`). The
/// two are bit-reversed over four bits, not equal, so a straight cast would
/// swap red and alpha and leave green and blue exchanged — a mask asking for
/// alpha-only would write red-only.
///
/// Total over the mask's range by construction: the input is `ColorWriteMask`,
/// whose only producer is the decoder, and the decoder refuses anything above
/// `MTLColorWriteMaskAll` by name. Bits above the fourth are ignored here
/// rather than declined a second time.
pub fn vk_color_write_mask(mask: ColorWriteMask) -> vk::ColorComponentFlags {
    let bits = mask.bits();
    let mut out = vk::ColorComponentFlags::empty();
    if bits & MTL_COLOR_WRITE_MASK_RED != 0 {
        out |= vk::ColorComponentFlags::R;
    }
    if bits & MTL_COLOR_WRITE_MASK_GREEN != 0 {
        out |= vk::ColorComponentFlags::G;
    }
    if bits & MTL_COLOR_WRITE_MASK_BLUE != 0 {
        out |= vk::ColorComponentFlags::B;
    }
    if bits & MTL_COLOR_WRITE_MASK_ALPHA != 0 {
        out |= vk::ColorComponentFlags::A;
    }
    out
}

pub fn vk_factor(factor: BlendFactor) -> vk::BlendFactor {
    match factor {
        BlendFactor::Zero => vk::BlendFactor::ZERO,
        BlendFactor::One => vk::BlendFactor::ONE,
        BlendFactor::SrcColor => vk::BlendFactor::SRC_COLOR,
        BlendFactor::OneMinusSrcColor => vk::BlendFactor::ONE_MINUS_SRC_COLOR,
        BlendFactor::SrcAlpha => vk::BlendFactor::SRC_ALPHA,
        BlendFactor::OneMinusSrcAlpha => vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
        BlendFactor::DstColor => vk::BlendFactor::DST_COLOR,
        BlendFactor::OneMinusDstColor => vk::BlendFactor::ONE_MINUS_DST_COLOR,
        BlendFactor::DstAlpha => vk::BlendFactor::DST_ALPHA,
        BlendFactor::OneMinusDstAlpha => vk::BlendFactor::ONE_MINUS_DST_ALPHA,
        BlendFactor::SrcAlphaSaturated => vk::BlendFactor::SRC_ALPHA_SATURATE,
        BlendFactor::ConstantColor => vk::BlendFactor::CONSTANT_COLOR,
        BlendFactor::OneMinusConstantColor => vk::BlendFactor::ONE_MINUS_CONSTANT_COLOR,
        BlendFactor::ConstantAlpha => vk::BlendFactor::CONSTANT_ALPHA,
        BlendFactor::OneMinusConstantAlpha => vk::BlendFactor::ONE_MINUS_CONSTANT_ALPHA,
    }
}

pub fn vk_operation(op: BlendOp) -> vk::BlendOp {
    match op {
        BlendOp::Add => vk::BlendOp::ADD,
        BlendOp::Subtract => vk::BlendOp::SUBTRACT,
        BlendOp::ReverseSubtract => vk::BlendOp::REVERSE_SUBTRACT,
        BlendOp::Min => vk::BlendOp::MIN,
        BlendOp::Max => vk::BlendOp::MAX,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Metal's blend enums are dense from 0, so the whole range maps and the
    /// first value past the end declines by name.
    #[test]
    fn the_blend_enums_are_total_over_their_sdk_range() {
        for mtl in 0..=14u32 {
            assert!(factor(mtl).is_ok(), "MTLBlendFactor {mtl}");
        }
        assert_eq!(
            factor(15).unwrap_err(),
            TranslateReason::UnknownBlendFactor(15)
        );
        for mtl in 0..=4u32 {
            assert!(operation(mtl).is_ok(), "MTLBlendOperation {mtl}");
        }
        assert_eq!(
            operation(5).unwrap_err(),
            TranslateReason::UnknownBlendOperation(5)
        );
    }

    /// Every engine factor reaches a distinct Vulkan factor — two collapsing
    /// onto one would silently change how a surface composites.
    #[test]
    fn every_blend_factor_has_a_distinct_vulkan_spelling() {
        let all: Vec<BlendFactor> = (0..=14).map(|m| factor(m).unwrap()).collect();
        let mut vks: Vec<i32> = all.iter().map(|f| vk_factor(*f).as_raw()).collect();
        vks.sort_unstable();
        let before = vks.len();
        vks.dedup();
        assert_eq!(before, vks.len());

        let mut ops: Vec<i32> = (0..=4)
            .map(|m| vk_operation(operation(m).unwrap()).as_raw())
            .collect();
        ops.sort_unstable();
        let before = ops.len();
        ops.dedup();
        assert_eq!(before, ops.len());
    }

    /// The two enums share an ordering with Metal's headers for the first
    /// several values; spot-check the ones a transcription slip would swap.
    #[test]
    fn the_load_bearing_arms_match_the_metal_header() {
        assert_eq!(vk_factor(factor(4).unwrap()), vk::BlendFactor::SRC_ALPHA);
        assert_eq!(
            vk_factor(factor(5).unwrap()),
            vk::BlendFactor::ONE_MINUS_SRC_ALPHA
        );
        assert_eq!(
            vk_factor(factor(10).unwrap()),
            vk::BlendFactor::SRC_ALPHA_SATURATE
        );
        assert_eq!(
            vk_operation(operation(2).unwrap()),
            vk::BlendOp::REVERSE_SUBTRACT
        );
    }

    /// Metal's mask bits are alpha-first and Vulkan's are red-first, so the
    /// two are bit-reversed over four bits. A cast would swap red with alpha
    /// and green with blue, and the mask that motivated decoding this field at
    /// all — alpha-only — would come out as red-only, which writes colour and
    /// drops the coverage the guest was punching in.
    #[test]
    fn the_metal_and_vulkan_write_mask_bit_orders_are_reversed_not_equal() {
        use crate::runtime::decode::resource::{
            MTL_COLOR_WRITE_MASK_ALL, MTL_COLOR_WRITE_MASK_BLUE, MTL_COLOR_WRITE_MASK_GREEN,
            MTL_COLOR_WRITE_MASK_NONE,
        };
        let of = |bits: u32| vk_color_write_mask(ColorWriteMask::new(bits).unwrap());

        assert_eq!(of(MTL_COLOR_WRITE_MASK_ALPHA), vk::ColorComponentFlags::A);
        assert_eq!(of(MTL_COLOR_WRITE_MASK_RED), vk::ColorComponentFlags::R);
        assert_eq!(of(MTL_COLOR_WRITE_MASK_GREEN), vk::ColorComponentFlags::G);
        assert_eq!(of(MTL_COLOR_WRITE_MASK_BLUE), vk::ColorComponentFlags::B);
        assert_eq!(of(MTL_COLOR_WRITE_MASK_ALL), vk::ColorComponentFlags::RGBA);
        assert_eq!(
            of(MTL_COLOR_WRITE_MASK_NONE),
            vk::ColorComponentFlags::empty()
        );
        // The default is `all`, which is what an entry with no tag means.
        assert_eq!(
            vk_color_write_mask(ColorWriteMask::default()),
            vk::ColorComponentFlags::RGBA
        );
        // A straight cast would agree on `all` and `none` and disagree on
        // every single-channel mask; assert the disagreement so a later
        // "simplification" to `from_raw(bits)` fails here.
        assert_ne!(
            of(MTL_COLOR_WRITE_MASK_ALPHA),
            vk::ColorComponentFlags::from_raw(MTL_COLOR_WRITE_MASK_ALPHA)
        );
        // Every mask in range maps injectively — two collapsing onto one would
        // silently merge distinct pipelines.
        let mut seen: Vec<u32> = (0..=MTL_COLOR_WRITE_MASK_ALL)
            .map(|m| of(m).as_raw())
            .collect();
        let before = seen.len();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(before, seen.len());
    }

    /// A whole descriptor fails on its first bad component instead of
    /// substituting a default — the difference between a visible decline and a
    /// surface that composites wrong for the rest of the boot.
    #[test]
    fn a_bad_component_fails_the_whole_descriptor() {
        let ok = state(1, 5, 0, 1, 5, 0, [0.0; 4]).unwrap();
        assert_eq!(ok.src_color, BlendFactor::One);
        assert_eq!(ok.dst_color, BlendFactor::OneMinusSrcAlpha);
        assert_eq!(ok.color_op, BlendOp::Add);
        assert_eq!(
            state(1, 99, 0, 1, 5, 0, [0.0; 4]).unwrap_err(),
            TranslateReason::UnknownBlendFactor(99)
        );
        assert_eq!(
            state(1, 5, 0, 1, 5, 77, [0.0; 4]).unwrap_err(),
            TranslateReason::UnknownBlendOperation(77)
        );
    }
}
