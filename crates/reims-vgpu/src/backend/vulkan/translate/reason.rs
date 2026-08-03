//! Typed decline reasons for the Metal → Vulkan translation boundary.
//!
//! Every translation entry point is total: it returns either a Vulkan value or
//! one of these, never a silent default. The variants exist so a decline can be
//! logged, tested and grepped by **name** — `AGENTS.md` requires each distinct
//! check to carry its own `reason=<slug>` rather than collapsing several causes
//! into one status. A free-text payload cannot satisfy that mechanically.
//!
//! Shape is a plain enum implementing [`Decline`] plus
//! [`TranslateReason::slug`]. The offending numeric value
//! rides along so the fail-visible line carries the load-bearing field, and
//! [`std::fmt::Display`] renders both.

use crate::observe::Decline;

/// Why a decoded Metal value has no Vulkan equivalent this backend will emit.
///
/// The payload is always the raw wire value the decoder produced, so a log line
/// names both the class of failure and the exact number that caused it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TranslateReason {
    /// `MTLPixelFormat` value outside the set the decode contract defines.
    /// "Unknown wire format stays unknown" — no fallback texel layout is
    /// invented for it.
    UnknownPixelFormat(u16),
    /// A pipeline honoured the format's channel layout but **not** its sRGB
    /// transfer function, so the hardware will not encode on write. Recorded by
    /// the site that takes [`super::pixel::PixelFormat::linear_vk`] instead of
    /// `vk`; it is a downgrade, not a failure, and the draw still runs.
    SrgbDowngraded(u16),
    /// The format is defined by the contract but the sampled rail carries no
    /// byte layout for it, so its texels cannot be uploaded without a CPU
    /// convert pass. Distinct from [`Self::UnknownPixelFormat`]: the format is
    /// understood, this rail just does not carry it.
    NoSampledLayout(u16),
    /// The format is defined by the contract but is not one the engine can use
    /// as a colour attachment. Distinct from [`Self::UnknownPixelFormat`] for
    /// the same reason.
    NoColorAttachmentFormat(u16),
    /// The format is defined by the contract but the compute rail carries no
    /// storage-image layout for it. Same shape as [`Self::NoSampledLayout`]:
    /// the format is understood, this rail just does not carry it.
    NoStorageImageFormat(u16),
    /// A `StorageImageSelector` ordinal outside the contract enum. Distinct
    /// from [`Self::NoStorageImageFormat`] — that one starts from a Metal
    /// format, this one from an already-narrowed selector, so a mismatch here
    /// means the two vocabularies have drifted apart rather than that a format
    /// is unsupported.
    UnknownStorageSelector(u32),
    /// `MTLVertexFormat` value outside the SDK enum.
    UnknownVertexFormat(u32),
    /// `MTLVertexStepFunction` value outside the SDK enum.
    UnknownVertexStepFunction(u32),
    /// `MTLPrimitiveType` value outside the SDK enum.
    UnknownPrimitiveType(u32),
    /// `MTLBlendFactor` value outside the SDK enum.
    UnknownBlendFactor(u32),
    /// `MTLBlendOperation` value outside the SDK enum.
    UnknownBlendOperation(u32),
    /// `MTLCompareFunction` value outside the SDK enum (depth, stencil and
    /// sampler compare share the Metal enum, hence one reason).
    UnknownCompareFunction(u32),
    /// `MTLStencilOperation` value outside the SDK enum.
    UnknownStencilOperation(u32),
    /// `MTLCullMode` value outside the SDK enum.
    UnknownCullMode(u32),
    /// `MTLWinding` value outside the SDK enum.
    UnknownWinding(u32),
    /// `MTLSamplerMinMagFilter` value outside the SDK enum.
    UnknownSamplerFilter(u32),
    /// `MTLSamplerMipFilter` value outside the SDK enum.
    UnknownSamplerMipFilter(u32),
    /// `MTLSamplerAddressMode` value outside the SDK enum.
    UnknownSamplerAddressMode(u32),
    /// `MTLSamplerBorderColor` value outside the SDK enum.
    UnknownSamplerBorderColor(u32),
    /// A type-8 view swizzle selector outside the decoded contract's range.
    UnknownSwizzleSelector(u8),
    /// The device does not advertise the requested `VkFormat` as a vertex
    /// buffer format, and no portable substitute exists for it.
    /// Payload is the `VkFormat` raw value.
    FormatNotVertexBuffer(i32),
}

impl crate::observe::Decline for TranslateReason {
    /// Stable snake_case slug for `reason=` in the always-on fail log.
    ///
    /// One slug per distinct check, never shared: the point is that a grep of
    /// `/tmp/reims-vgpu-fail.log` tells you which translation refused, not merely that
    /// one did.
    ///
    /// This was an inherent method with a per-enum uniqueness test, which is how
    /// `unknown_pixel_format` came to be claimed by `runtime/heap_query`'s
    /// `QueryError` as well: both enums were internally consistent and nothing
    /// compared them. Implementing the crate trait gives every slug here one
    /// vocabulary to be distinct within.
    fn slug(&self) -> &'static str {
        match self {
            Self::UnknownPixelFormat(_) => "unknown_pixel_format",
            Self::NoStorageImageFormat(_) => "no_storage_image_format",
            Self::UnknownStorageSelector(_) => "unknown_storage_selector",
            Self::SrgbDowngraded(_) => "srgb_downgraded",
            Self::NoSampledLayout(_) => "no_sampled_layout",
            Self::NoColorAttachmentFormat(_) => "no_color_attachment_format",
            Self::UnknownVertexFormat(_) => "unknown_vertex_format",
            Self::UnknownVertexStepFunction(_) => "unknown_vertex_step_function",
            Self::UnknownPrimitiveType(_) => "unknown_primitive_type",
            Self::UnknownBlendFactor(_) => "unknown_blend_factor",
            Self::UnknownBlendOperation(_) => "unknown_blend_operation",
            Self::UnknownCompareFunction(_) => "unknown_compare_function",
            Self::UnknownStencilOperation(_) => "unknown_stencil_operation",
            Self::UnknownCullMode(_) => "unknown_cull_mode",
            Self::UnknownWinding(_) => "unknown_winding",
            Self::UnknownSamplerFilter(_) => "unknown_sampler_filter",
            Self::UnknownSamplerMipFilter(_) => "unknown_sampler_mip_filter",
            Self::UnknownSamplerAddressMode(_) => "unknown_sampler_address_mode",
            Self::UnknownSamplerBorderColor(_) => "unknown_sampler_border_color",
            Self::UnknownSwizzleSelector(_) => "unknown_swizzle_selector",
            Self::FormatNotVertexBuffer(_) => "format_not_vertex_buffer",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        vec![("value", self.value().to_string())]
    }
}

impl TranslateReason {
    /// The raw decoded value that could not be translated, widened to `u32` for
    /// uniform logging. `VkFormat` values are `i32` on the wire and are
    /// reinterpreted, not truncated.
    pub fn value(self) -> u32 {
        match self {
            Self::UnknownPixelFormat(v)
            | Self::SrgbDowngraded(v)
            | Self::NoSampledLayout(v)
            | Self::NoColorAttachmentFormat(v)
            | Self::NoStorageImageFormat(v) => u32::from(v),
            Self::UnknownVertexFormat(v)
            | Self::UnknownVertexStepFunction(v)
            | Self::UnknownStorageSelector(v)
            | Self::UnknownPrimitiveType(v)
            | Self::UnknownBlendFactor(v)
            | Self::UnknownBlendOperation(v)
            | Self::UnknownCompareFunction(v)
            | Self::UnknownStencilOperation(v)
            | Self::UnknownCullMode(v)
            | Self::UnknownWinding(v)
            | Self::UnknownSamplerFilter(v)
            | Self::UnknownSamplerMipFilter(v)
            | Self::UnknownSamplerAddressMode(v)
            | Self::UnknownSamplerBorderColor(v) => v,
            Self::UnknownSwizzleSelector(v) => u32::from(v),
            Self::FormatNotVertexBuffer(v) => v as u32,
        }
    }
}

impl std::fmt::Display for TranslateReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "reason={} value={}", self.slug(), self.value())
    }
}

impl std::error::Error for TranslateReason {}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every reason this module can produce, so the exhaustiveness tests below
    /// fail to compile-or-assert when a variant is added without a slug.
    const ALL: &[TranslateReason] = &[
        TranslateReason::UnknownPixelFormat(0),
        TranslateReason::SrgbDowngraded(0),
        TranslateReason::NoSampledLayout(0),
        TranslateReason::NoColorAttachmentFormat(0),
        TranslateReason::NoStorageImageFormat(0),
        TranslateReason::UnknownStorageSelector(0),
        TranslateReason::UnknownVertexFormat(0),
        TranslateReason::UnknownVertexStepFunction(0),
        TranslateReason::UnknownPrimitiveType(0),
        TranslateReason::UnknownBlendFactor(0),
        TranslateReason::UnknownBlendOperation(0),
        TranslateReason::UnknownCompareFunction(0),
        TranslateReason::UnknownStencilOperation(0),
        TranslateReason::UnknownCullMode(0),
        TranslateReason::UnknownWinding(0),
        TranslateReason::UnknownSamplerFilter(0),
        TranslateReason::UnknownSamplerMipFilter(0),
        TranslateReason::UnknownSamplerAddressMode(0),
        TranslateReason::UnknownSamplerBorderColor(0),
        TranslateReason::UnknownSwizzleSelector(0),
        TranslateReason::FormatNotVertexBuffer(0),
    ];

    /// Two checks sharing a slug is the exact failure `AGENTS.md` names: you
    /// grep the fail log, see the slug fire, and still cannot tell which of the
    /// two refused.
    #[test]
    fn every_reason_has_its_own_slug() {
        let mut slugs: Vec<&str> = ALL.iter().map(|r| r.slug()).collect();
        slugs.sort_unstable();
        let before = slugs.len();
        slugs.dedup();
        assert_eq!(before, slugs.len(), "duplicate translate reason slug");
    }

    /// Slugs are grepped out of a space-separated log line, so they may not
    /// carry whitespace or an `=`, and they stay kebab/snake for consistency
    /// with the existing `caps` and `present_proxy` slugs.
    #[test]
    fn slugs_are_log_safe() {
        for r in ALL {
            let s = r.slug();
            assert!(!s.is_empty());
            assert!(
                s.bytes()
                    .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_'),
                "slug {s:?} must be lowercase snake_case"
            );
        }
    }

    /// The payload survives into the rendered line — a decline that names only
    /// its class leaves the reader without the value that caused it.
    #[test]
    fn display_carries_the_offending_value() {
        let r = TranslateReason::UnknownPixelFormat(0x1234);
        assert_eq!(r.to_string(), "reason=unknown_pixel_format value=4660");
        // VkFormat is a signed handle; reinterpretation must not truncate.
        let f = TranslateReason::FormatNotVertexBuffer(-9);
        assert_eq!(f.value(), u32::MAX - 8);
    }
}
