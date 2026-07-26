//! `MTLPrimitiveType`, cull mode, winding, compare function, stencil operation
//! and index type → their Vulkan spellings.
//!
//! Metal and Vulkan agree on the *ordering* of several of these enums, which
//! makes a numeric cast tempting and wrong: the agreement is a coincidence of
//! two independent specs, not a contract, and a cast turns a future divergence
//! into silently wrong rasterization. Every arm is spelled out so the compiler
//! catches a drift instead of the screen.

use ash::vk;

use super::reason::TranslateReason;
use crate::backend::vulkan::engine::{
    CullMode, IndexType, PrimitiveTopology, SamplerCompareFunction, StencilOp,
};

/// `MTLPrimitiveType` (SDK numeric values).
pub fn primitive_topology(mtl: u32) -> Result<PrimitiveTopology, TranslateReason> {
    Ok(match mtl {
        0 => PrimitiveTopology::Point,
        1 => PrimitiveTopology::Line,
        2 => PrimitiveTopology::LineStrip,
        3 => PrimitiveTopology::Triangle,
        4 => PrimitiveTopology::TriangleStrip,
        other => return Err(TranslateReason::UnknownPrimitiveType(other)),
    })
}

/// `MTLCullMode` (SDK numeric values).
pub fn cull_mode(mtl: u32) -> Result<CullMode, TranslateReason> {
    Ok(match mtl {
        0 => CullMode::None,
        1 => CullMode::Front,
        2 => CullMode::Back,
        other => return Err(TranslateReason::UnknownCullMode(other)),
    })
}

/// `MTLWinding` → whether the front face is counter-clockwise.
pub fn front_face_ccw(mtl: u32) -> Result<bool, TranslateReason> {
    match mtl {
        0 => Ok(false), // MTLWindingClockwise, Metal's default
        1 => Ok(true),  // MTLWindingCounterClockwise
        other => Err(TranslateReason::UnknownWinding(other)),
    }
}

/// `MTLCompareFunction` (SDK numeric values). Depth test, stencil test and
/// sampler compare all carry this same Metal enum.
pub fn compare_function(mtl: u32) -> Result<SamplerCompareFunction, TranslateReason> {
    Ok(match mtl {
        0 => SamplerCompareFunction::Never,
        1 => SamplerCompareFunction::Less,
        2 => SamplerCompareFunction::Equal,
        3 => SamplerCompareFunction::LessEqual,
        4 => SamplerCompareFunction::Greater,
        5 => SamplerCompareFunction::NotEqual,
        6 => SamplerCompareFunction::GreaterEqual,
        7 => SamplerCompareFunction::Always,
        other => return Err(TranslateReason::UnknownCompareFunction(other)),
    })
}

/// `MTLStencilOperation` (SDK numeric values).
pub fn stencil_operation(mtl: u32) -> Result<StencilOp, TranslateReason> {
    Ok(match mtl {
        0 => StencilOp::Keep,
        1 => StencilOp::Zero,
        2 => StencilOp::Replace,
        3 => StencilOp::IncrementClamp,
        4 => StencilOp::DecrementClamp,
        5 => StencilOp::Invert,
        6 => StencilOp::IncrementWrap,
        7 => StencilOp::DecrementWrap,
        other => return Err(TranslateReason::UnknownStencilOperation(other)),
    })
}

/// `MTLIndexType` (SDK numeric values).
///
/// The shared runtime loader owns the typed refusal because both Metal and
/// Vulkan consume it; `None` therefore remains a classification here, and the
/// caller turns it into `IndexLoadReason::TypeUnsupported`.
pub fn index_type(mtl: u32) -> Option<IndexType> {
    match mtl {
        0 => Some(IndexType::U16),
        1 => Some(IndexType::U32),
        _ => None,
    }
}

pub fn vk_topology(topology: PrimitiveTopology) -> vk::PrimitiveTopology {
    match topology {
        PrimitiveTopology::Point => vk::PrimitiveTopology::POINT_LIST,
        PrimitiveTopology::Line => vk::PrimitiveTopology::LINE_LIST,
        PrimitiveTopology::LineStrip => vk::PrimitiveTopology::LINE_STRIP,
        PrimitiveTopology::Triangle => vk::PrimitiveTopology::TRIANGLE_LIST,
        PrimitiveTopology::TriangleStrip => vk::PrimitiveTopology::TRIANGLE_STRIP,
    }
}

pub fn vk_compare_op(compare: SamplerCompareFunction) -> vk::CompareOp {
    match compare {
        SamplerCompareFunction::Never => vk::CompareOp::NEVER,
        SamplerCompareFunction::Less => vk::CompareOp::LESS,
        SamplerCompareFunction::Equal => vk::CompareOp::EQUAL,
        SamplerCompareFunction::LessEqual => vk::CompareOp::LESS_OR_EQUAL,
        SamplerCompareFunction::Greater => vk::CompareOp::GREATER,
        SamplerCompareFunction::NotEqual => vk::CompareOp::NOT_EQUAL,
        SamplerCompareFunction::GreaterEqual => vk::CompareOp::GREATER_OR_EQUAL,
        SamplerCompareFunction::Always => vk::CompareOp::ALWAYS,
    }
}

pub fn vk_stencil_op(op: StencilOp) -> vk::StencilOp {
    match op {
        StencilOp::Keep => vk::StencilOp::KEEP,
        StencilOp::Zero => vk::StencilOp::ZERO,
        StencilOp::Replace => vk::StencilOp::REPLACE,
        StencilOp::IncrementClamp => vk::StencilOp::INCREMENT_AND_CLAMP,
        StencilOp::DecrementClamp => vk::StencilOp::DECREMENT_AND_CLAMP,
        StencilOp::Invert => vk::StencilOp::INVERT,
        StencilOp::IncrementWrap => vk::StencilOp::INCREMENT_AND_WRAP,
        StencilOp::DecrementWrap => vk::StencilOp::DECREMENT_AND_WRAP,
    }
}

pub fn vk_index_type(index: IndexType) -> vk::IndexType {
    match index {
        IndexType::U16 => vk::IndexType::UINT16,
        IndexType::U32 => vk::IndexType::UINT32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every SDK value in range maps, and the first one past the end declines
    /// by its own slug rather than a shared one.
    #[test]
    fn each_raster_enum_is_total_over_its_sdk_range() {
        for mtl in 0..=4u32 {
            assert!(primitive_topology(mtl).is_ok(), "primitive {mtl}");
        }
        assert_eq!(
            primitive_topology(5).unwrap_err(),
            TranslateReason::UnknownPrimitiveType(5)
        );
        for mtl in 0..=2u32 {
            assert!(cull_mode(mtl).is_ok(), "cull {mtl}");
        }
        assert_eq!(
            cull_mode(3).unwrap_err(),
            TranslateReason::UnknownCullMode(3)
        );
        assert!(!front_face_ccw(0).unwrap());
        assert!(front_face_ccw(1).unwrap());
        assert_eq!(
            front_face_ccw(2).unwrap_err(),
            TranslateReason::UnknownWinding(2)
        );
        for mtl in 0..=7u32 {
            assert!(compare_function(mtl).is_ok(), "compare {mtl}");
            assert!(stencil_operation(mtl).is_ok(), "stencil {mtl}");
        }
        assert_eq!(
            compare_function(8).unwrap_err(),
            TranslateReason::UnknownCompareFunction(8)
        );
        assert_eq!(
            stencil_operation(8).unwrap_err(),
            TranslateReason::UnknownStencilOperation(8)
        );
    }

    /// The exact wire order of `MTLCompareFunction`, value by value.
    ///
    /// Injectivity below proves no two values collide; it does not prove the
    /// table is not *rotated*. A rotation still round-trips and still renders —
    /// it just inverts occlusion for every 3D draw — so the mapping is pinned
    /// arm by arm. (Moved here from `runtime/metal_draw/mod.rs`, which held a second
    /// copy of this table; the assertion outlived the duplicate.)
    #[test]
    fn compare_function_matches_the_metal_abi_order() {
        use SamplerCompareFunction as C;
        assert_eq!(compare_function(0), Ok(C::Never));
        assert_eq!(compare_function(1), Ok(C::Less));
        assert_eq!(compare_function(2), Ok(C::Equal));
        assert_eq!(compare_function(3), Ok(C::LessEqual));
        assert_eq!(compare_function(4), Ok(C::Greater));
        assert_eq!(compare_function(5), Ok(C::NotEqual));
        assert_eq!(compare_function(6), Ok(C::GreaterEqual));
        assert_eq!(compare_function(7), Ok(C::Always));
        assert_eq!(
            compare_function(99).unwrap_err(),
            TranslateReason::UnknownCompareFunction(99)
        );
    }

    /// Same, for `MTLStencilOperation`. The increment/decrement pairs are the
    /// transcription hazard: clamp and wrap differ only in overflow behaviour.
    #[test]
    fn stencil_operation_matches_the_metal_abi_order() {
        use StencilOp as O;
        assert_eq!(stencil_operation(0), Ok(O::Keep));
        assert_eq!(stencil_operation(1), Ok(O::Zero));
        assert_eq!(stencil_operation(2), Ok(O::Replace));
        assert_eq!(stencil_operation(3), Ok(O::IncrementClamp));
        assert_eq!(stencil_operation(4), Ok(O::DecrementClamp));
        assert_eq!(stencil_operation(5), Ok(O::Invert));
        assert_eq!(stencil_operation(6), Ok(O::IncrementWrap));
        assert_eq!(stencil_operation(7), Ok(O::DecrementWrap));
        assert_eq!(
            stencil_operation(99).unwrap_err(),
            TranslateReason::UnknownStencilOperation(99)
        );
    }

    /// The compare and stencil spellings must be injective. These are the
    /// enums whose Metal and Vulkan orderings *nearly* agree, which is exactly
    /// where a transcription slip hides: swapping LESS_OR_EQUAL and GREATER
    /// still renders, just wrongly.
    #[test]
    fn compare_and_stencil_spellings_are_injective() {
        let mut ops: Vec<i32> = (0..=7)
            .map(|m| vk_compare_op(compare_function(m).unwrap()).as_raw())
            .collect();
        ops.sort_unstable();
        let before = ops.len();
        ops.dedup();
        assert_eq!(before, ops.len());

        let mut stencil: Vec<i32> = (0..=7)
            .map(|m| vk_stencil_op(stencil_operation(m).unwrap()).as_raw())
            .collect();
        stencil.sort_unstable();
        let before = stencil.len();
        stencil.dedup();
        assert_eq!(before, stencil.len());
    }

    /// Spot-check the arms whose Metal and Vulkan names differ, so a
    /// "the orderings match, just cast it" refactor cannot pass.
    #[test]
    fn the_arms_whose_names_differ_are_spelled_out() {
        assert_eq!(
            vk_compare_op(compare_function(3).unwrap()),
            vk::CompareOp::LESS_OR_EQUAL
        );
        assert_eq!(
            vk_compare_op(compare_function(6).unwrap()),
            vk::CompareOp::GREATER_OR_EQUAL
        );
        assert_eq!(
            vk_stencil_op(stencil_operation(3).unwrap()),
            vk::StencilOp::INCREMENT_AND_CLAMP
        );
        assert_eq!(
            vk_stencil_op(stencil_operation(7).unwrap()),
            vk::StencilOp::DECREMENT_AND_WRAP
        );
        // Metal's strip primitives are 2 and 4, not 4 and 5.
        assert_eq!(
            vk_topology(primitive_topology(2).unwrap()),
            vk::PrimitiveTopology::LINE_STRIP
        );
        assert_eq!(
            vk_topology(primitive_topology(4).unwrap()),
            vk::PrimitiveTopology::TRIANGLE_STRIP
        );
    }

    #[test]
    fn index_types_map_by_width() {
        assert_eq!(index_type(0), Some(IndexType::U16));
        assert_eq!(index_type(1), Some(IndexType::U32));
        assert_eq!(index_type(2), None);
        assert_eq!(vk_index_type(IndexType::U16), vk::IndexType::UINT16);
        assert_eq!(vk_index_type(IndexType::U32), vk::IndexType::UINT32);
    }
}
