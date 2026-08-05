//! Guest ordinals to Metal enums, checked.
//!
//! Every value that reaches this module was decoded out of the guest's command
//! stream, so it is an arbitrary `u32`. The `MTL*` types are fieldless
//! `#[repr(u64)]` enums, and producing one whose discriminant is not a declared
//! variant is **undefined behaviour**, not a decode error — the same rule
//! `reims-vgpu-wire`'s invariant 4 states for wire structs, and the reason that
//! crate stores raw scalars and exposes fallible accessors. This module is the
//! device-side half of it: name every variant, return `None` for anything else,
//! and let the caller turn that into a typed refusal.
//!
//! It replaced 28 `transmute::<u64, MTL*>` calls. Two things that audit found
//! are worth keeping, because both defeat the obvious repair:
//!
//! - **A `<= max` range check is not sufficient, because two of these enums have
//!   holes.** `MTLVertexFormat` and `MTLAttributeFormat` run 0–42 and then
//!   resume at 45: Apple left 43 and 44 unassigned between
//!   `UChar4Normalized_BGRA` and `UChar`. A bound of "not above the last
//!   variant" admits both, and `stage_input`'s attribute-format guard did.
//! - **Five of the sites had no check at all.** The two vertex-descriptor
//!   builders (`backend::metal::render::make_vertex_descriptor` and
//!   `runtime::icb::metal_vertex_descriptor_from_attrs_for_draw`) transmuted a
//!   type-7 pipeline descriptor's `format` and `step_function` words straight
//!   through, and those are guest bytes with no producer-side clamp anywhere.
//!
//! ## Where the variant lists come from
//!
//! The `metal` crate's own enum declarations, which are what define the Rust
//! type — a value it does not declare has no legal representation here however
//! well Apple documents it. Those declarations were checked against the Metal
//! SDK headers on this host (macOS 26 SDK, `MTLVertexDescriptor.h`,
//! `MTLStageInputOutputDescriptor.h`, `MTLRenderPipeline.h`,
//! `MTLRenderCommandEncoder.h`, `MTLRenderPass.h`, `MTLDepthStencil.h`,
//! `MTLArgument.h`), and they agree everywhere except at the top of four:
//!
//! | enum | SDK tail | `metal` 0.33 tail |
//! |---|---|---|
//! | `MTLVertexFormat` | `FloatRG11B10` 54, `FloatRGB9E5` 55 | stops at `Half` 53 |
//! | `MTLAttributeFormat` | `FloatRG11B10` 54, `FloatRGB9E5` 55 | stops at `Half` 53 |
//! | `MTLBlendFactor` | `Unspecialized` 19 | stops at `OneMinusSource1Alpha` 18 |
//! | `MTLBlendOperation` | `Unspecialized` 5 | stops at `Max` 4 |
//!
//! So a guest asking for one of those six values is declined here rather than
//! executed. That is a narrowing against what Metal itself would accept, and it
//! is deliberate: the alternative is undefined behaviour, and there is no third
//! option inside `metal`'s type. The refusal is counted at each call site, so a
//! non-zero reading is the measured argument for reaching those values the way
//! [`crate::backend::metal::raw_metal`] reaches other unwrapped selectors — by
//! sending the setter a raw `u64` and never materializing an enum at all. Until
//! one fires, that is four wrappers for traffic nobody has seen.

use metal::{
    MTLAttributeFormat, MTLBlendFactor, MTLBlendOperation, MTLCompareFunction, MTLCullMode,
    MTLIndexType, MTLLoadAction, MTLPrimitiveType, MTLSamplerAddressMode, MTLSamplerBorderColor,
    MTLSamplerMinMagFilter, MTLSamplerMipFilter, MTLStencilOperation, MTLStepFunction,
    MTLStoreAction, MTLVertexFormat, MTLVertexStepFunction, MTLWinding,
};

/// Define a checked ordinal to enum conversion plus its declared-variant table.
///
/// The table exists so the generated test can assert both directions without a
/// second hand-written list: every declared variant converts back to itself, and
/// nothing else in the surrounding range converts at all. A variant left out of
/// the macro invocation therefore fails its own test rather than silently
/// becoming a refusal.
macro_rules! checked_ordinal {
    (
        $(#[$outer:meta])*
        fn $fn_name:ident -> $ty:ty;
        const $variants:ident;
        test $test_name:ident;
        [ $($variant:ident),+ $(,)? ]
    ) => {
        /// Every variant the `metal` crate declares for this enum.
        ///
        /// Test-only: the conversion below does not read it, and its purpose is
        /// to give the generated test the accepted set without a second
        /// hand-written list that could disagree with the one above it.
        #[cfg(test)]
        const $variants: &[$ty] = &[$(<$ty>::$variant),+];

        $(#[$outer])*
        pub(crate) fn $fn_name(ordinal: u32) -> Option<$ty> {
            $(
                if ordinal == <$ty>::$variant as u32 {
                    return Some(<$ty>::$variant);
                }
            )+
            None
        }

        // Compared as ordinals rather than as values: `metal` derives
        // `PartialEq` on most of these enums and not on all of them
        // (`MTLStoreAction` and `MTLLoadAction` have no derive at all), and the
        // ordinal is what the assertion is actually about.
        #[cfg(test)]
        #[test]
        fn $test_name() {
            let declared: Vec<u32> = $variants.iter().map(|&v| v as u32).collect();
            for &v in $variants {
                assert_eq!(
                    $fn_name(v as u32).map(|got| got as u32),
                    Some(v as u32),
                    "{} rejected its own variant {}",
                    stringify!($fn_name),
                    v as u32
                );
            }
            // Everything up to four past the last declared variant, so the run's
            // interior holes and its upper edge are both covered.
            let ceiling = declared.iter().copied().max().unwrap_or(0).saturating_add(4);
            for ordinal in 0..=ceiling {
                if declared.contains(&ordinal) {
                    continue;
                }
                assert!(
                    $fn_name(ordinal).is_none(),
                    "{} accepted undeclared ordinal {}",
                    stringify!($fn_name),
                    ordinal
                );
            }
            assert!($fn_name(u32::MAX).is_none());
        }
    };
}

checked_ordinal! {
    /// `MTLVertexFormat` for a vertex-descriptor attribute.
    ///
    /// Declines 43 and 44 — the gap Apple left between `UChar4Normalized_BGRA`
    /// and `UChar` — and 54/55, which the SDK declares and `metal` does not.
    fn vertex_format -> MTLVertexFormat;
    const VERTEX_FORMATS;
    test every_declared_vertex_format_converts_and_nothing_else_does;
    [
        Invalid, UChar2, UChar3, UChar4, Char2, Char3, Char4,
        UChar2Normalized, UChar3Normalized, UChar4Normalized,
        Char2Normalized, Char3Normalized, Char4Normalized,
        UShort2, UShort3, UShort4, Short2, Short3, Short4,
        UShort2Normalized, UShort3Normalized, UShort4Normalized,
        Short2Normalized, Short3Normalized, Short4Normalized,
        Half2, Half3, Half4, Float, Float2, Float3, Float4,
        Int, Int2, Int3, Int4, UInt, UInt2, UInt3, UInt4,
        Int1010102Normalized, UInt1010102Normalized, UChar4Normalized_BGRA,
        UChar, Char, UCharNormalized, CharNormalized,
        UShort, Short, UShortNormalized, ShortNormalized, Half,
    ]
}

checked_ordinal! {
    /// `MTLAttributeFormat` for a compute stage-input attribute.
    ///
    /// The same numbering as [`vertex_format`], including the 43/44 hole; Metal
    /// declares the two enums separately and this device must not assume they
    /// stay in step.
    fn attribute_format -> MTLAttributeFormat;
    const ATTRIBUTE_FORMATS;
    test every_declared_attribute_format_converts_and_nothing_else_does;
    [
        Invalid, UChar2, UChar3, UChar4, Char2, Char3, Char4,
        UChar2Normalized, UChar3Normalized, UChar4Normalized,
        Char2Normalized, Char3Normalized, Char4Normalized,
        UShort2, UShort3, UShort4, Short2, Short3, Short4,
        UShort2Normalized, UShort3Normalized, UShort4Normalized,
        Short2Normalized, Short3Normalized, Short4Normalized,
        Half2, Half3, Half4, Float, Float2, Float3, Float4,
        Int, Int2, Int3, Int4, UInt, UInt2, UInt3, UInt4,
        Int1010102Normalized, UInt1010102Normalized, UChar4Normalized_BGRA,
        UChar, Char, UCharNormalized, CharNormalized,
        UShort, Short, UShortNormalized, ShortNormalized, Half,
    ]
}

checked_ordinal! {
    /// `MTLVertexStepFunction` for a vertex-descriptor buffer layout.
    fn vertex_step_function -> MTLVertexStepFunction;
    const VERTEX_STEP_FUNCTIONS;
    test every_declared_vertex_step_function_converts_and_nothing_else_does;
    [Constant, PerVertex, PerInstance, PerPatch, PerPatchControlPoint]
}

/// `MTLStepFunction` by ordinal, indexed by Apple's numbering.
///
/// This is the one conversion here that cannot name its variants, because
/// **`metal` 0.33 assigns six of this enum's nine names to the wrong numbers.**
/// Against `MTLStageInputOutputDescriptor.h` on the macOS 26 SDK:
///
/// | Apple | crate |
/// |---|---|
/// | `PerVertex` 1, `PerInstance` 2, `PerPatch` 3, `PerPatchControlPoint` 4 | 4, 1, 2, 3 |
/// | `ThreadPositionInGridY` 6, `ThreadPositionInGridXIndexed` 7 | 7, 6 |
///
/// Only `Constant` 0, `ThreadPositionInGridX` 5 and
/// `ThreadPositionInGridYIndexed` 8 agree. `stage_input.rs` already knew about
/// the second row — its own `MTL_STEP_*` constants exist for exactly that
/// reason — but its comment names only that swap, and the four vertex-side
/// values are wrong too.
///
/// So a `MTLStepFunction::PerVertex` written here would reach Apple as 4, which
/// Apple reads as `PerPatchControlPoint`: naming a variant silently rewrites
/// the guest's step function. The table is therefore indexed by Apple's
/// ordinal and holds whichever crate variant carries that discriminant, which
/// is why the entries read misaligned — they are.
///
/// The check stays exhaustive rather than a range: the crate declares 0 through
/// 8 with no gaps, so an ordinal in that run has a declared representation
/// whatever the crate calls it, and one outside it does not.
const STEP_FUNCTION_BY_ORDINAL: [MTLStepFunction; 9] = [
    MTLStepFunction::Constant,                     // 0
    MTLStepFunction::PerInstance,                  // 1  Apple: PerVertex
    MTLStepFunction::PerPatch,                     // 2  Apple: PerInstance
    MTLStepFunction::PerPatchControlPoint,         // 3  Apple: PerPatch
    MTLStepFunction::PerVertex,                    // 4  Apple: PerPatchControlPoint
    MTLStepFunction::ThreadPositionInGridX,        // 5
    MTLStepFunction::ThreadPositionInGridXIndexed, // 6  Apple: ThreadPositionInGridY
    MTLStepFunction::ThreadPositionInGridY,        // 7  Apple: ThreadPositionInGridXIndexed
    MTLStepFunction::ThreadPositionInGridYIndexed, // 8
];

/// `MTLStepFunction` for a compute stage-input buffer layout.
pub(crate) fn step_function(ordinal: u32) -> Option<MTLStepFunction> {
    STEP_FUNCTION_BY_ORDINAL.get(ordinal as usize).copied()
}

checked_ordinal! {
    /// `MTLBlendFactor` for one color attachment's blend state.
    fn blend_factor -> MTLBlendFactor;
    const BLEND_FACTORS;
    test every_declared_blend_factor_converts_and_nothing_else_does;
    [
        Zero, One, SourceColor, OneMinusSourceColor, SourceAlpha,
        OneMinusSourceAlpha, DestinationColor, OneMinusDestinationColor,
        DestinationAlpha, OneMinusDestinationAlpha, SourceAlphaSaturated,
        BlendColor, OneMinusBlendColor, BlendAlpha, OneMinusBlendAlpha,
        Source1Color, OneMinusSource1Color, Source1Alpha, OneMinusSource1Alpha,
    ]
}

checked_ordinal! {
    /// `MTLBlendOperation` for one color attachment's blend state.
    fn blend_operation -> MTLBlendOperation;
    const BLEND_OPERATIONS;
    test every_declared_blend_operation_converts_and_nothing_else_does;
    [Add, Subtract, ReverseSubtract, Min, Max]
}

checked_ordinal! {
    /// `MTLCullMode` for the render encoder's raster state.
    fn cull_mode -> MTLCullMode;
    const CULL_MODES;
    test every_declared_cull_mode_converts_and_nothing_else_does;
    [None, Front, Back]
}

checked_ordinal! {
    /// `MTLWinding` for the render encoder's raster state.
    fn winding -> MTLWinding;
    const WINDINGS;
    test every_declared_winding_converts_and_nothing_else_does;
    [Clockwise, CounterClockwise]
}

checked_ordinal! {
    /// `MTLCompareFunction` for a depth or stencil test.
    fn compare_function -> MTLCompareFunction;
    const COMPARE_FUNCTIONS;
    test every_declared_compare_function_converts_and_nothing_else_does;
    [Never, Less, Equal, LessEqual, Greater, NotEqual, GreaterEqual, Always]
}

checked_ordinal! {
    /// `MTLStencilOperation` for one stencil face.
    fn stencil_operation -> MTLStencilOperation;
    const STENCIL_OPERATIONS;
    test every_declared_stencil_operation_converts_and_nothing_else_does;
    [
        Keep, Zero, Replace, IncrementClamp, DecrementClamp, Invert,
        IncrementWrap, DecrementWrap,
    ]
}

checked_ordinal! {
    /// `MTLLoadAction` for a render pass attachment.
    fn load_action -> MTLLoadAction;
    const LOAD_ACTIONS;
    test every_declared_load_action_converts_and_nothing_else_does;
    [DontCare, Load, Clear]
}

checked_ordinal! {
    /// `MTLStoreAction` for a render pass attachment.
    fn store_action -> MTLStoreAction;
    const STORE_ACTIONS;
    test every_declared_store_action_converts_and_nothing_else_does;
    [
        DontCare, Store, MultisampleResolve, StoreAndMultisampleResolve,
        Unknown, CustomSampleDepthStore,
    ]
}

checked_ordinal! {
    /// `MTLIndexType` for an indexed draw or a stage-input index buffer.
    fn index_type -> MTLIndexType;
    const INDEX_TYPES;
    test every_declared_index_type_converts_and_nothing_else_does;
    [UInt16, UInt32]
}

checked_ordinal! {
    /// `MTLPrimitiveType` for a draw.
    fn primitive_type -> MTLPrimitiveType;
    const PRIMITIVE_TYPES;
    test every_declared_primitive_type_converts_and_nothing_else_does;
    [Point, Line, LineStrip, Triangle, TriangleStrip]
}

checked_ordinal! {
    /// `MTLSamplerMinMagFilter` for a sampler's minification or magnification.
    fn sampler_min_mag_filter -> MTLSamplerMinMagFilter;
    const SAMPLER_MIN_MAG_FILTERS;
    test every_declared_sampler_min_mag_filter_converts_and_nothing_else_does;
    [Nearest, Linear]
}

checked_ordinal! {
    /// `MTLSamplerMipFilter` for a sampler's mip selection.
    fn sampler_mip_filter -> MTLSamplerMipFilter;
    const SAMPLER_MIP_FILTERS;
    test every_declared_sampler_mip_filter_converts_and_nothing_else_does;
    [NotMipmapped, Nearest, Linear]
}

checked_ordinal! {
    /// `MTLSamplerAddressMode` for one sampler axis.
    fn sampler_address_mode -> MTLSamplerAddressMode;
    const SAMPLER_ADDRESS_MODES;
    test every_declared_sampler_address_mode_converts_and_nothing_else_does;
    [
        ClampToEdge, MirrorClampToEdge, Repeat, MirrorRepeat, ClampToZero,
        ClampToBorderColor,
    ]
}

checked_ordinal! {
    /// `MTLSamplerBorderColor` for a sampler clamping to a border.
    fn sampler_border_color -> MTLSamplerBorderColor;
    const SAMPLER_BORDER_COLORS;
    test every_declared_sampler_border_color_converts_and_nothing_else_does;
    [TransparentBlack, OpaqueBlack, OpaqueWhite]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two format enums have a hole at 43 and 44, and that is the whole
    /// reason this module exists rather than a `<= max` bound at each call site.
    ///
    /// Apple's `MTLVertexDescriptor.h` and `MTLStageInputOutputDescriptor.h` on
    /// the macOS 26 SDK run `UChar4Normalized_BGRA = 42` straight to
    /// `UChar = 45`, so a check that only rejects values above the last variant
    /// lets two undefined discriminants through.
    #[test]
    fn the_format_enums_leave_forty_three_and_forty_four_undeclared() {
        for ordinal in [43u32, 44] {
            assert!(vertex_format(ordinal).is_none());
            assert!(attribute_format(ordinal).is_none());
        }
        assert_eq!(
            vertex_format(42),
            Some(MTLVertexFormat::UChar4Normalized_BGRA)
        );
        assert_eq!(vertex_format(45), Some(MTLVertexFormat::UChar));
        assert_eq!(
            attribute_format(42),
            Some(MTLAttributeFormat::UChar4Normalized_BGRA)
        );
        assert_eq!(attribute_format(45), Some(MTLAttributeFormat::UChar));
    }

    /// The four values the SDK declares and `metal` does not, pinned so the
    /// narrowing this module's doc describes stays a measured fact.
    ///
    /// If a `metal` bump adds them these assertions flip, which is the signal to
    /// add the variants above and delete this test rather than to relax it.
    #[test]
    fn the_values_metal_does_not_declare_are_declined_rather_than_transmuted() {
        // MTLVertexFormatFloatRG11B10 / MTLVertexFormatFloatRGB9E5.
        assert!(vertex_format(54).is_none());
        assert!(vertex_format(55).is_none());
        assert!(attribute_format(54).is_none());
        assert!(attribute_format(55).is_none());
        // MTLBlendFactorUnspecialized / MTLBlendOperationUnspecialized.
        assert!(blend_factor(19).is_none());
        assert!(blend_operation(5).is_none());
    }

    /// The step-function table carries Apple's numbering, and every entry's
    /// own discriminant is its index.
    ///
    /// This is what pins `metal` 0.33's misnumbering rather than working around
    /// it silently: if a crate bump renumbers the variants to match Apple, the
    /// table above still produces the right ordinals and this test still
    /// passes, but the misaligned-looking comments become wrong. If a bump
    /// renumbers them some *other* way, this fails.
    #[test]
    fn the_step_function_table_maps_each_ordinal_to_the_variant_carrying_it() {
        for (ordinal, &value) in STEP_FUNCTION_BY_ORDINAL.iter().enumerate() {
            assert_eq!(value as u32, ordinal as u32);
            assert_eq!(
                step_function(ordinal as u32).map(|v| v as u32),
                Some(ordinal as u32)
            );
        }
        assert!(step_function(STEP_FUNCTION_BY_ORDINAL.len() as u32).is_none());
        assert!(step_function(u32::MAX).is_none());
    }

    /// The names `metal` 0.33 gives the step function are not Apple's, and the
    /// three that happen to agree are the only ones that may be used by name.
    ///
    /// Apple: `PerVertex` 1, `PerInstance` 2, `PerPatch` 3,
    /// `PerPatchControlPoint` 4, `ThreadPositionInGridY` 6,
    /// `ThreadPositionInGridXIndexed` 7.
    #[test]
    fn the_step_function_names_metal_declares_disagree_with_apples_numbering() {
        assert_eq!(MTLStepFunction::Constant as u32, 0);
        assert_eq!(MTLStepFunction::ThreadPositionInGridX as u32, 5);
        assert_eq!(MTLStepFunction::ThreadPositionInGridYIndexed as u32, 8);
        // Everything else is off, so naming it would rewrite the guest's value.
        assert_ne!(MTLStepFunction::PerVertex as u32, 1);
        assert_ne!(MTLStepFunction::PerInstance as u32, 2);
        assert_ne!(MTLStepFunction::PerPatch as u32, 3);
        assert_ne!(MTLStepFunction::PerPatchControlPoint as u32, 4);
        assert_ne!(MTLStepFunction::ThreadPositionInGridY as u32, 6);
        assert_ne!(MTLStepFunction::ThreadPositionInGridXIndexed as u32, 7);
    }

    /// Every conversion refuses the ordinal one past its own last variant.
    ///
    /// The per-enum tests the macro generates already sweep their own range;
    /// this one exists so a table added later without its generated test still
    /// gets an upper bound checked.
    #[test]
    fn no_conversion_accepts_the_ordinal_past_its_last_variant() {
        macro_rules! past_end {
            ($fn_name:ident, $variants:ident) => {{
                let max = $variants.iter().map(|&v| v as u32).max().unwrap();
                assert!(
                    $fn_name(max + 1).is_none(),
                    "{} accepted {}",
                    stringify!($fn_name),
                    max + 1
                );
            }};
        }
        past_end!(vertex_format, VERTEX_FORMATS);
        past_end!(attribute_format, ATTRIBUTE_FORMATS);
        past_end!(vertex_step_function, VERTEX_STEP_FUNCTIONS);
        past_end!(blend_factor, BLEND_FACTORS);
        past_end!(blend_operation, BLEND_OPERATIONS);
        past_end!(cull_mode, CULL_MODES);
        past_end!(winding, WINDINGS);
        past_end!(compare_function, COMPARE_FUNCTIONS);
        past_end!(stencil_operation, STENCIL_OPERATIONS);
        past_end!(load_action, LOAD_ACTIONS);
        past_end!(store_action, STORE_ACTIONS);
        past_end!(index_type, INDEX_TYPES);
        past_end!(primitive_type, PRIMITIVE_TYPES);
        past_end!(sampler_min_mag_filter, SAMPLER_MIN_MAG_FILTERS);
        past_end!(sampler_mip_filter, SAMPLER_MIP_FILTERS);
        past_end!(sampler_address_mode, SAMPLER_ADDRESS_MODES);
        past_end!(sampler_border_color, SAMPLER_BORDER_COLORS);
    }
}
