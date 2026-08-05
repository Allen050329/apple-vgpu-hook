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

/// Define a checked ordinal to enum conversion, and prove it at compile time.
///
/// The generated `const` block asserts both directions against the variant list
/// given here: every listed variant converts back to itself, and no ordinal that
/// is *not* listed converts at all, sweeping to four past the highest listed
/// discriminant so interior holes and the upper edge are both covered.
///
/// # What this cannot catch
///
/// **A variant the `metal` crate declares and this invocation omits.** The list
/// is the only statement of the accepted set, so an omitted variant is absent
/// from the conversion and from the sweep alike: the ordinal reads as undeclared,
/// `$fn_name` returns `None` for it, and the assertion that undeclared ordinals
/// return `None` passes. The guest's value silently becomes a refusal.
///
/// This was verified by deleting `Half3` from `vertex_format`'s list and
/// watching the build stay green. It is not fixable here — Rust cannot enumerate
/// a foreign enum's variants — so the guard is the module doc above: the lists
/// were read off `metal` 0.33's own declarations, and a crate bump means
/// re-reading them. An earlier version of this doc claimed omission "fails its
/// own test"; it never did.
macro_rules! checked_ordinal {
    (
        $(#[$outer:meta])*
        fn $fn_name:ident -> $ty:ty;
        [ $($variant:ident),+ $(,)? ]
    ) => {
        $(#[$outer])*
        pub(crate) const fn $fn_name(ordinal: u32) -> Option<$ty> {
            $(
                if ordinal == <$ty>::$variant as u32 {
                    return Some(<$ty>::$variant);
                }
            )+
            None
        }

        // The same sweep the generated `#[test]` used to run, as a `const`
        // block, and `$fn_name` is a `const fn` so that it can be. The reason is
        // the one `super::constants` spells out: this module is
        // `backend-metal`-gated, so a `#[cfg(test)]` test in it is compiled out
        // of the Vulkan arm and its `--lib` suite runs on Apple hosts only.
        // Seventeen tables' worth of "no undeclared ordinal converts" was
        // therefore checked on no machine anybody edits this code from — and
        // what it stands between is a guest `u32` and a `transmute` into a
        // `#[repr(u64)]` enum, which is undefined behaviour rather than a decode
        // error. `rustc` evaluates this on every arm that compiles the file,
        // including the cross-compiled Metal clippy run `AGENTS.md` requires
        // from Linux.
        //
        // Ordinals are compared rather than values: `metal` derives `PartialEq`
        // on most of these enums and not all (`MTLStoreAction` and
        // `MTLLoadAction` have no derive at all), and the ordinal is what the
        // assertion is about.
        const _: () = {
            // Every declared variant converts back to itself.
            $(
                assert!(
                    match $fn_name(<$ty>::$variant as u32) {
                        Some(got) => got as u32 == <$ty>::$variant as u32,
                        None => false,
                    },
                    concat!(stringify!($fn_name), " rejected one of its own variants"),
                );
            )+

            // Everything up to four past the last declared variant, so the
            // run's interior holes and its upper edge are both covered.
            let mut ceiling = 0u32;
            $(
                if (<$ty>::$variant as u32) > ceiling {
                    ceiling = <$ty>::$variant as u32;
                }
            )+
            ceiling = ceiling.saturating_add(4);

            let mut ordinal = 0u32;
            while ordinal <= ceiling {
                let mut declared = false;
                $(
                    if ordinal == <$ty>::$variant as u32 {
                        declared = true;
                    }
                )+
                assert!(
                    declared || $fn_name(ordinal).is_none(),
                    concat!(stringify!($fn_name), " accepted an undeclared ordinal"),
                );
                ordinal += 1;
            }

            assert!($fn_name(u32::MAX).is_none());
        };
    };
}

checked_ordinal! {
    /// `MTLVertexFormat` for a vertex-descriptor attribute.
    ///
    /// Declines 43 and 44 — the gap Apple left between `UChar4Normalized_BGRA`
    /// and `UChar` — and 54/55, which the SDK declares and `metal` does not.
    fn vertex_format -> MTLVertexFormat;
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
///
/// `const` so the pins below it are `const` assertions, checked on every arm
/// that compiles this file rather than by a suite this pathway never runs.
/// Spelled as an explicit bound rather than `get().copied()` because slice
/// indexing is not available in a `const fn`.
pub(crate) const fn step_function(ordinal: u32) -> Option<MTLStepFunction> {
    if (ordinal as usize) < STEP_FUNCTION_BY_ORDINAL.len() {
        Some(STEP_FUNCTION_BY_ORDINAL[ordinal as usize])
    } else {
        None
    }
}

checked_ordinal! {
    /// `MTLBlendFactor` for one color attachment's blend state.
    fn blend_factor -> MTLBlendFactor;
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
    [Add, Subtract, ReverseSubtract, Min, Max]
}

checked_ordinal! {
    /// `MTLCullMode` for the render encoder's raster state.
    fn cull_mode -> MTLCullMode;
    [None, Front, Back]
}

checked_ordinal! {
    /// `MTLWinding` for the render encoder's raster state.
    fn winding -> MTLWinding;
    [Clockwise, CounterClockwise]
}

checked_ordinal! {
    /// `MTLCompareFunction` for a depth or stencil test.
    fn compare_function -> MTLCompareFunction;
    [Never, Less, Equal, LessEqual, Greater, NotEqual, GreaterEqual, Always]
}

checked_ordinal! {
    /// `MTLStencilOperation` for one stencil face.
    fn stencil_operation -> MTLStencilOperation;
    [
        Keep, Zero, Replace, IncrementClamp, DecrementClamp, Invert,
        IncrementWrap, DecrementWrap,
    ]
}

checked_ordinal! {
    /// `MTLLoadAction` for a render pass attachment.
    fn load_action -> MTLLoadAction;
    [DontCare, Load, Clear]
}

checked_ordinal! {
    /// `MTLStoreAction` for a render pass attachment.
    fn store_action -> MTLStoreAction;
    [
        DontCare, Store, MultisampleResolve, StoreAndMultisampleResolve,
        Unknown, CustomSampleDepthStore,
    ]
}

checked_ordinal! {
    /// `MTLIndexType` for an indexed draw or a stage-input index buffer.
    fn index_type -> MTLIndexType;
    [UInt16, UInt32]
}

checked_ordinal! {
    /// `MTLPrimitiveType` for a draw.
    fn primitive_type -> MTLPrimitiveType;
    [Point, Line, LineStrip, Triangle, TriangleStrip]
}

checked_ordinal! {
    /// `MTLSamplerMinMagFilter` for a sampler's minification or magnification.
    fn sampler_min_mag_filter -> MTLSamplerMinMagFilter;
    [Nearest, Linear]
}

checked_ordinal! {
    /// `MTLSamplerMipFilter` for a sampler's mip selection.
    fn sampler_mip_filter -> MTLSamplerMipFilter;
    [NotMipmapped, Nearest, Linear]
}

checked_ordinal! {
    /// `MTLSamplerAddressMode` for one sampler axis.
    fn sampler_address_mode -> MTLSamplerAddressMode;
    [
        ClampToEdge, MirrorClampToEdge, Repeat, MirrorRepeat, ClampToZero,
        ClampToBorderColor,
    ]
}

checked_ordinal! {
    /// `MTLSamplerBorderColor` for a sampler clamping to a border.
    fn sampler_border_color -> MTLSamplerBorderColor;
    [TransparentBlack, OpaqueBlack, OpaqueWhite]
}

/// Assert a conversion answers `ordinal` with exactly `variant`, at compile time.
///
/// `Option<MTL*>` cannot be compared with `==` in a `const` block — `metal`
/// derives `PartialEq` on only some of these enums, and none of the derives are
/// `const` — so the comparison is on the ordinal, through a `match`.
macro_rules! const_converts {
    ($fn_name:ident($ordinal:expr) == $ty:ty : $variant:ident) => {
        const _: () = assert!(match $fn_name($ordinal) {
            Some(got) => got as u32 == <$ty>::$variant as u32,
            None => false,
        });
    };
}

// The two format enums have a hole at 43 and 44, and that is the whole reason
// this module exists rather than a `<= max` bound at each call site. Apple's
// `MTLVertexDescriptor.h` and `MTLStageInputOutputDescriptor.h` on the macOS 26
// SDK run `UChar4Normalized_BGRA = 42` straight to `UChar = 45`, so a check that
// only rejects values above the last variant lets two undefined discriminants
// through.
const _: () = assert!(vertex_format(43).is_none());
const _: () = assert!(vertex_format(44).is_none());
const _: () = assert!(attribute_format(43).is_none());
const _: () = assert!(attribute_format(44).is_none());
const_converts!(vertex_format(42) == MTLVertexFormat: UChar4Normalized_BGRA);
const_converts!(vertex_format(45) == MTLVertexFormat: UChar);
const_converts!(attribute_format(42) == MTLAttributeFormat: UChar4Normalized_BGRA);
const_converts!(attribute_format(45) == MTLAttributeFormat: UChar);

// The four values the SDK declares and `metal` does not, pinned so the narrowing
// this module's doc describes stays a measured fact. If a `metal` bump adds them
// these assertions flip, which is the signal to add the variants above and
// delete these lines rather than to relax them.
//
// MTLVertexFormatFloatRG11B10 / MTLVertexFormatFloatRGB9E5.
const _: () = assert!(vertex_format(54).is_none());
const _: () = assert!(vertex_format(55).is_none());
const _: () = assert!(attribute_format(54).is_none());
const _: () = assert!(attribute_format(55).is_none());
// MTLBlendFactorUnspecialized / MTLBlendOperationUnspecialized.
const _: () = assert!(blend_factor(19).is_none());
const _: () = assert!(blend_operation(5).is_none());

// The step-function table carries Apple's numbering, and every entry's own
// discriminant is its index.
//
// This is what pins `metal` 0.33's misnumbering rather than working around it
// silently: if a crate bump renumbers the variants to match Apple, the table
// above still produces the right ordinals and this still holds, but the
// misaligned-looking comments become wrong. If a bump renumbers them some
// *other* way, the build fails here.
const _: () = {
    let mut ordinal = 0usize;
    while ordinal < STEP_FUNCTION_BY_ORDINAL.len() {
        assert!(STEP_FUNCTION_BY_ORDINAL[ordinal] as u32 == ordinal as u32);
        assert!(match step_function(ordinal as u32) {
            Some(got) => got as u32 == ordinal as u32,
            None => false,
        });
        ordinal += 1;
    }
};
const _: () = assert!(step_function(STEP_FUNCTION_BY_ORDINAL.len() as u32).is_none());
const _: () = assert!(step_function(u32::MAX).is_none());

// The names `metal` 0.33 gives the step function are not Apple's, and the three
// that happen to agree are the only ones that may be used by name.
//
// Apple: `PerVertex` 1, `PerInstance` 2, `PerPatch` 3, `PerPatchControlPoint` 4,
// `ThreadPositionInGridY` 6, `ThreadPositionInGridXIndexed` 7.
const _: () = assert!(MTLStepFunction::Constant as u32 == 0);
const _: () = assert!(MTLStepFunction::ThreadPositionInGridX as u32 == 5);
const _: () = assert!(MTLStepFunction::ThreadPositionInGridYIndexed as u32 == 8);
// Everything else is off, so naming it would rewrite the guest's value.
const _: () = assert!(MTLStepFunction::PerVertex as u32 != 1);
const _: () = assert!(MTLStepFunction::PerInstance as u32 != 2);
const _: () = assert!(MTLStepFunction::PerPatch as u32 != 3);
const _: () = assert!(MTLStepFunction::PerPatchControlPoint as u32 != 4);
const _: () = assert!(MTLStepFunction::ThreadPositionInGridY as u32 != 6);
const _: () = assert!(MTLStepFunction::ThreadPositionInGridXIndexed as u32 != 7);
