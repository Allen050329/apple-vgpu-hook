//! What the Vulkan pathway does with every decoded guest GPU-state field.
//!
//! # Why a manifest and not a comment
//!
//! `translate/` made value mapping total: every `MTLPixelFormat` this contract
//! defines reaches a `VkFormat` or declines by name. Nothing made *state
//! coverage* total. A Metal field that is never decoded cannot decline, because
//! no code knows it exists — so the pipeline builder hardcoded `RGBA` for the
//! colour write mask and `TYPE_1` for the sample count, and neither the
//! decoder, the gate, nor the fixed-state census could report a thing. The
//! failure is by *omission*, and omission is exactly what a hand-maintained
//! list of known gaps cannot catch: the fields nobody remembered are the ones
//! it omits.
//!
//! So every field gets a disposition, and there is no state for "nobody
//! looked":
//!
//! * [`Disposition::Honored`] — the pathway consumes it, and the manifest
//!   names the semantic site that does.
//! * [`Disposition::Declined`] — decoded, deliberately not bound, and the site
//!   says so with a named `reason=` slug in the always-on log.
//! * [`Disposition::DroppedSilently`] — decoded, not bound, and **nothing says
//!   so**. A defect with a name rather than a resting state; the count is
//!   pinned by a test so it can be driven down and cannot grow unnoticed.
//! * [`Disposition::NotOnTheWire`] — absent from the decoded contract
//!   entirely, with a note recording what is known about where it would live.
//!   This is the honest form of "we never looked", written down.
//!
//! # What the tests enforce
//!
//! 1. **Every decoded field is listed or explicitly transport.** Every public
//!    struct under `runtime/decode/` is scanned at test time. Its `pub` fields
//!    must appear in this manifest unless the whole struct or field has a
//!    reasoned transport exclusion. Adding a field without deciding its fate
//!    is a failing test. Public enums are pinned in a separate 21-type census.
//! 2. **`Honored` names a site that exists.** A disposition claiming the
//!    builder binds something must point at source that is still there.
//! 3. **`Honored` and a hardcoded constant are mutually exclusive.** The
//!    builder lines that pin a value regardless of guest state are enumerated,
//!    and the fields they pin may not claim to be honoured. This is the check
//!    that would have caught `color_write_mask(RGBA)`.
//!
//! # What this module deliberately does not do
//!
//! It changes no behaviour. It is an inventory, and its value is that the
//! inventory is *checked* — the gaps it records are the input to closing them,
//! not a substitute for it.
//!
//! Because it is only ever read by its own tests, it is declared
//! `#[cfg(test)]` alongside `translate::gate` and `caps::gate`, the other two
//! modules here that assert about the source rather than run in it. Nothing
//! outside names `coverage::`; if a product path ever needs a disposition at
//! runtime, that is the signal to un-gate it deliberately rather than by
//! accident.

/// What happens to one decoded Metal command/descriptor field on the Vulkan
/// pathway.
///
/// Three states, with the declining one typed as a `TranslateReason`, is the
/// obvious shape and it is wrong twice over. The fail-visible declines on this path are `observe` slugs, not
/// translation failures — the field translated fine, the *rail* cannot carry
/// it — and, more importantly, there turned out to be a fourth state the sketch
/// had no room for: decoded, dropped, and **silent**. Forcing those into
/// `Declined` would have made the manifest assert a log line that does not
/// exist, which is the failure mode this whole module is against.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Disposition {
    /// The pathway consumes it. `site` names the source that does, as
    /// `path/to/file.rs:symbol`, so the claim can be checked rather than
    /// trusted.
    Honored { site: &'static str },
    /// Decoded, deliberately not bound, and **fail-visible**: `slug` is the
    /// `reason=` that reaches `/tmp/reims-vgpu-fail.log` so a guest relying on the
    /// field learns why it did not take effect.
    Declined {
        slug: &'static str,
        site: &'static str,
    },
    /// Decoded, not bound, and **nothing says so**.
    ///
    /// This is a defect with a name, not a resting state. It is the exact shape
    /// `AGENTS.md` forbids — a decoded guest value dropped with no log line —
    /// and it is spelled out here so the count can be driven to zero and cannot
    /// grow unnoticed.
    DroppedSilently { note: &'static str },
    /// Not present in the decoded contract at all. `note` records what is known
    /// about where it would live on the wire and what binding it would need, so
    /// the gap is a written RE task rather than an absence.
    NotOnTheWire { note: &'static str },
}

impl Disposition {
    pub fn is_honored(&self) -> bool {
        matches!(self, Self::Honored { .. })
    }
}

/// One decoded GPU-state field or one SDK field still absent from the wire.
#[derive(Clone, Copy, Debug)]
pub struct FieldCoverage {
    /// The Metal SDK name for descriptor state, or the canonical
    /// `module::Struct.field` spelling for command/resource fields.
    pub field: &'static str,
    /// The decoded wire struct field this corresponds to, preferably as
    /// `module::Struct.field` (the older unique `Struct.field` spelling is
    /// retained for the render baseline), or `None` when the field is
    /// [`Disposition::NotOnTheWire`].
    pub decoded_as: Option<&'static str>,
    pub disposition: Disposition,
}

/// One Metal descriptor/command family and every field of it.
#[derive(Clone, Copy, Debug)]
pub struct DescriptorFamily {
    /// The Metal SDK class or command/resource family name.
    pub descriptor: &'static str,
    /// The decode structs that carry this family's fields, as source-visible
    /// `struct` names.
    pub decode_structs: &'static [&'static str],
    pub fields: &'static [FieldCoverage],
}

const fn honored(
    field: &'static str,
    decoded_as: &'static str,
    site: &'static str,
) -> FieldCoverage {
    FieldCoverage {
        field,
        decoded_as: Some(decoded_as),
        disposition: Disposition::Honored { site },
    }
}

const fn declined(
    field: &'static str,
    decoded_as: &'static str,
    slug: &'static str,
    site: &'static str,
) -> FieldCoverage {
    FieldCoverage {
        field,
        decoded_as: Some(decoded_as),
        disposition: Disposition::Declined { slug, site },
    }
}

const fn dropped(
    field: &'static str,
    decoded_as: &'static str,
    note: &'static str,
) -> FieldCoverage {
    FieldCoverage {
        field,
        decoded_as: Some(decoded_as),
        disposition: Disposition::DroppedSilently { note },
    }
}

const fn absent(field: &'static str, note: &'static str) -> FieldCoverage {
    FieldCoverage {
        field,
        decoded_as: None,
        disposition: Disposition::NotOnTheWire { note },
    }
}

const fn decoded_honored(decoded_as: &'static str, site: &'static str) -> FieldCoverage {
    honored(decoded_as, decoded_as, site)
}

const fn decoded_declined(
    decoded_as: &'static str,
    slug: &'static str,
    site: &'static str,
) -> FieldCoverage {
    declined(decoded_as, decoded_as, slug, site)
}

const fn decoded_dropped(decoded_as: &'static str, note: &'static str) -> FieldCoverage {
    dropped(decoded_as, decoded_as, note)
}

/// Compact spelling for the Phase-6 command/resource inventories. Repeated
/// fields under one `honored` group really do converge at the named semantic
/// consumer; `declined` and `dropped` groups keep their shared reason visible.
macro_rules! decoded_fields {
    (
        $(honored $site:literal => [$($honored:literal),* $(,)?];)*
        $(declined $slug:literal at $declined_site:literal => [$($declined:literal),* $(,)?];)*
        $(dropped $note:literal => [$($dropped:literal),* $(,)?];)*
    ) => {
        &[
            $($(decoded_honored($honored, $site),)*)*
            $($(decoded_declined($declined, $slug, $declined_site),)*)*
            $($(decoded_dropped($dropped, $note),)*)*
        ]
    };
}

/// `MTLRenderPipelineDescriptor`.
const RENDER_PIPELINE: &[FieldCoverage] = &[
    honored(
        "vertexFunction",
        "RenderPipelineDescriptor.vertex_func_ref",
        "runtime/metal_draw/mod.rs:load_mtlb",
    ),
    honored(
        "fragmentFunction",
        "RenderPipelineDescriptor.fragment_func_ref",
        "runtime/metal_draw/mod.rs:load_mtlb",
    ),
    honored(
        "objectFunction",
        "RenderPipelineDescriptor.object_func_ref",
        "runtime/metal_draw/mod.rs:load_mtlb",
    ),
    honored(
        "meshFunction",
        "RenderPipelineDescriptor.mesh_func_ref",
        "runtime/metal_draw/mod.rs:load_mtlb",
    ),
    honored(
        "vertexDescriptor",
        "RenderPipelineDescriptor.vertex_attributes",
        "backend/vulkan/engine/caches.rs:vertex_attribute_descs",
    ),
    honored(
        "colorAttachments[n].pixelFormat",
        "PipelineColorAttachment.pixel_format",
        "backend/vulkan/translate/pixel.rs:color_attachment",
    ),
    honored(
        "colorAttachments[n].blendingEnabled",
        "PipelineColorAttachment.blending_enabled",
        "backend/vulkan/engine/caches.rs:blend_att",
    ),
    honored(
        "colorAttachments[n].sourceRGBBlendFactor",
        "PipelineColorAttachment.src_rgb",
        "backend/vulkan/engine/caches.rs:blend_att",
    ),
    honored(
        "colorAttachments[n].destinationRGBBlendFactor",
        "PipelineColorAttachment.dst_rgb",
        "backend/vulkan/engine/caches.rs:blend_att",
    ),
    honored(
        "colorAttachments[n].rgbBlendOperation",
        "PipelineColorAttachment.op_rgb",
        "backend/vulkan/engine/caches.rs:blend_att",
    ),
    honored(
        "colorAttachments[n].sourceAlphaBlendFactor",
        "PipelineColorAttachment.src_alpha",
        "backend/vulkan/engine/caches.rs:blend_att",
    ),
    honored(
        "colorAttachments[n].destinationAlphaBlendFactor",
        "PipelineColorAttachment.dst_alpha",
        "backend/vulkan/engine/caches.rs:blend_att",
    ),
    honored(
        "colorAttachments[n].alphaBlendOperation",
        "PipelineColorAttachment.op_alpha",
        "backend/vulkan/engine/caches.rs:blend_att",
    ),
    // The lapse this manifest exists for, now closed. The tag was read off a
    // live guest by `note_color_entry_fields` rather than guessed: it is
    // `0x09`, the ninth property in `MTLRenderPipeline.h`, following the eight
    // above it in header order.
    honored(
        "colorAttachments[n].writeMask",
        "PipelineColorAttachment.write_mask",
        "backend/vulkan/engine/caches.rs:blend_att",
    ),
    absent(
        "rasterSampleCount",
        "The pipeline sample count is not decoded; the builder pins \
         SampleCountFlags::TYPE_1, so an MSAA pipeline rasterizes 1x. Every \
         render target this backend allocates is single-sampled, so binding a \
         decoded value would also need the attachment path to carry it.",
    ),
    absent(
        "alphaToCoverageEnabled",
        "Not decoded. Meaningful only alongside a sample count above 1, which \
         is itself NotOnTheWire, so this is blocked behind that.",
    ),
    absent(
        "alphaToOneEnabled",
        "Not decoded. Same dependency on multisampling as alphaToCoverage.",
    ),
    absent(
        "rasterizationEnabled",
        "Not decoded. A pipeline with rasterization disabled runs the vertex \
         stage for its side effects only; this backend would render it.",
    ),
    absent(
        "inputPrimitiveTopology",
        "Not decoded as a pipeline field. The topology bound is the *draw's* \
         MTLPrimitiveType (translate::raster::primitive_topology), which is a \
         different Metal concept: this field constrains the pipeline's topology \
         class and matters for tessellation and mesh pipelines.",
    ),
    absent(
        "maxVertexAmplificationCount",
        "Not decoded. Vertex amplification needs multiview; unused by the \
         compositor workloads observed.",
    ),
    absent(
        "supportIndirectCommandBuffers",
        "Not decoded as a pipeline flag. ICB support is handled as a separate \
         command path (runtime/icb/mod.rs), not as pipeline state.",
    ),
    absent(
        "tessellation*",
        "The whole tessellation field group (partitionMode, factorScaleEnabled, \
         factorFormat, controlPointIndexType, factorStepFunction, \
         outputWindingOrder, maxTessellationFactor) is not decoded. No observed \
         guest workload binds a tessellated pipeline.",
    ),
    absent(
        "depthAttachmentPixelFormat",
        "Not decoded as a pipeline field. The depth format the pipeline is \
         built against comes from the transient depth buffer the pass \
         allocates (translate::pixel::TRANSIENT_DEPTH_FORMAT), so a pipeline \
         declaring a different depth format would bind mismatched.",
    ),
    absent(
        "stencilAttachmentPixelFormat",
        "Not decoded, same shape as depthAttachmentPixelFormat: the combined \
         depth-stencil format is negotiated against the device rather than read \
         from the pipeline.",
    ),
];

/// `MTLRenderPassDescriptor`, including its attachment descriptors.
const RENDER_PASS: &[FieldCoverage] = &[
    honored(
        "colorAttachments[n].texture",
        "ColorAttachment.texture_ref",
        "runtime/metal_draw/mod.rs:color_target_request",
    ),
    honored(
        "colorAttachments[n].level",
        "ColorAttachment.level",
        "runtime/metal_draw/mod.rs:color_target_request",
    ),
    honored(
        "colorAttachments[n].loadAction",
        "ColorAttachment.load_action",
        "backend/vulkan/engine/caches.rs:load_op",
    ),
    honored(
        "colorAttachments[n].storeAction",
        "ColorAttachment.store_action",
        "runtime/metal_draw/mod.rs:map_store_action",
    ),
    honored(
        "colorAttachments[n].clearColor",
        "ColorAttachment.clear_color",
        "backend/vulkan/engine/exec.rs:LoadOp",
    ),
    honored(
        "colorAttachments[n].resolveTexture",
        "ColorAttachment.resolve_texture_ref",
        "runtime/metal_draw/mod.rs:color_target_request",
    ),
    honored(
        "depthAttachment.texture",
        "DepthAttachment.texture_ref",
        "runtime/metal_draw/mod.rs:encode_draw_and_writeback",
    ),
    honored(
        "depthAttachment.level",
        "DepthAttachment.level",
        "runtime/metal_draw/mod.rs:encode_draw_and_writeback",
    ),
    honored(
        "depthAttachment.clearDepth",
        "DepthAttachment.clear_depth",
        "runtime/metal_draw/mod.rs:encode_draw_and_writeback",
    ),
    honored(
        "depthAttachment.resolveTexture",
        "DepthAttachment.resolve_texture_ref",
        "runtime/metal_draw/mod.rs:encode_draw_and_writeback",
    ),
    honored(
        "depthAttachment.storeAction",
        "DepthAttachment.store_action",
        "runtime/metal_draw/mod.rs:map_store_action",
    ),
    // The transient depth buffer supports CLEAR only; a guest depth LOAD needs
    // a persistent depth resident, which the resident slot does not yet cache.
    declined(
        "depthAttachment.loadAction",
        "DepthAttachment.load_action",
        "depth_load_unsupported_transient",
        "runtime/metal_draw/vulkan.rs:depth_load_unsupported_transient",
    ),
    honored(
        "stencilAttachment.texture",
        "StencilAttachment.texture_ref",
        "runtime/metal_draw/mod.rs:encode_draw_and_writeback",
    ),
    honored(
        "stencilAttachment.level",
        "StencilAttachment.level",
        "runtime/metal_draw/mod.rs:encode_draw_and_writeback",
    ),
    honored(
        "stencilAttachment.clearStencil",
        "StencilAttachment.clear_stencil",
        "runtime/metal_draw/mod.rs:encode_draw_and_writeback",
    ),
    honored(
        "stencilAttachment.loadAction",
        "StencilAttachment.load_action",
        "runtime/metal_draw/mod.rs:map_load_action",
    ),
    honored(
        "stencilAttachment.storeAction",
        "StencilAttachment.store_action",
        "runtime/metal_draw/mod.rs:map_store_action",
    ),
    honored(
        "stencilAttachment.resolveTexture",
        "StencilAttachment.resolve_texture_ref",
        "runtime/metal_draw/mod.rs:encode_draw_and_writeback",
    ),
    honored(
        "colorAttachments[n].present",
        "ColorAttachment.present",
        "runtime/metal_draw/mod.rs:color_target_request",
    ),
    honored(
        "depthAttachment.present",
        "DepthAttachment.present",
        "runtime/metal_draw/mod.rs:encode_draw_and_writeback",
    ),
    honored(
        "stencilAttachment.present",
        "StencilAttachment.present",
        "runtime/metal_draw/mod.rs:encode_draw_and_writeback",
    ),
    absent(
        "colorAttachments[n].slice / depthPlane",
        "Not decoded. Array and 3D render targets would select a subresource \
         here; every observed target is a single-slice 2D image.",
    ),
    absent(
        "*.resolveLevel / resolveSlice / resolveDepthPlane",
        "Not decoded. The resolve subresource selectors accompany MSAA resolve, \
         which is blocked behind rasterSampleCount being NotOnTheWire.",
    ),
    absent(
        "*.storeActionOptions",
        "Not decoded. Carries the custom-sample-position resolve option, which \
         only applies to an MSAA resolve this backend never performs.",
    ),
    absent(
        "depthAttachment.depthResolveFilter",
        "Not decoded. Selects min/max depth resolve, which is MSAA-only and \
         blocked behind rasterSampleCount being NotOnTheWire.",
    ),
    absent(
        "stencilAttachment.stencilResolveFilter",
        "Not decoded. Selects min/max/depth-sampled stencil resolve, which is \
         MSAA-only and blocked behind rasterSampleCount being NotOnTheWire.",
    ),
    absent(
        "visibilityResultBuffer",
        "Not decoded as a pass field. Occlusion query results have no binding \
         in this backend.",
    ),
    absent(
        "renderTargetWidth / renderTargetHeight",
        "Not decoded. The render area comes from the resolved attachment \
         geometry, so an explicit smaller render target would be ignored.",
    ),
    absent(
        "renderTargetArrayLength",
        "Not decoded. Layered rendering would need array render targets and a \
         layer output from the vertex stage; no observed workload uses it.",
    ),
    absent(
        "defaultRasterSampleCount",
        "Not decoded. Supplies the pass sample count when no attachment does; \
         blocked behind multisampling being unsupported end to end.",
    ),
    absent(
        "imageblockSampleLength / threadgroupMemoryLength / tileWidth / tileHeight",
        "The tile-shading field group is not decoded. Tile pipelines are an \
         Apple-GPU feature with no Vulkan 1.2 baseline equivalent.",
    ),
    absent(
        "samplePositions / rasterizationRateMap / sampleBufferAttachments",
        "Not decoded. Programmable sample positions and variable rate shading \
         are unused; sample buffers are a profiling attachment.",
    ),
];

/// `MTLDepthStencilDescriptor` and its two `MTLStencilDescriptor` faces.
const DEPTH_STENCIL: &[FieldCoverage] = &[
    honored(
        "depthCompareFunction",
        "DepthStencilDescriptor.depth_compare_function",
        "backend/vulkan/translate/raster.rs:compare_function",
    ),
    honored(
        "depthWriteEnabled",
        "DepthStencilDescriptor.depth_write_enabled",
        "backend/vulkan/engine/caches.rs:depth_stencil",
    ),
    honored(
        "frontFaceStencil",
        "DepthStencilDescriptor.front_face",
        "runtime/metal_draw/mod.rs:engine_stencil_face",
    ),
    honored(
        "backFaceStencil",
        "DepthStencilDescriptor.back_face",
        "runtime/metal_draw/mod.rs:engine_stencil_face",
    ),
    honored(
        "frontFaceStencil.stencilCompareFunction",
        "DepthStencilFace.compare_function",
        "backend/vulkan/translate/raster.rs:compare_function",
    ),
    honored(
        "frontFaceStencil.stencilFailureOperation",
        "DepthStencilFace.stencil_failure_operation",
        "backend/vulkan/translate/raster.rs:stencil_operation",
    ),
    honored(
        "frontFaceStencil.depthFailureOperation",
        "DepthStencilFace.depth_failure_operation",
        "backend/vulkan/translate/raster.rs:stencil_operation",
    ),
    honored(
        "frontFaceStencil.depthStencilPassOperation",
        "DepthStencilFace.depth_stencil_pass_operation",
        "backend/vulkan/translate/raster.rs:stencil_operation",
    ),
    honored(
        "frontFaceStencil.readMask",
        "DepthStencilFace.read_mask",
        "backend/vulkan/engine/caches.rs:stencil_face",
    ),
    honored(
        "frontFaceStencil.writeMask",
        "DepthStencilFace.write_mask",
        "backend/vulkan/engine/caches.rs:stencil_face",
    ),
    honored(
        "(state object identity)",
        "DepthStencilDescriptor.depth_stencil_id",
        "runtime/metal_draw/mod.rs:load_depth_stencil_descriptor",
    ),
    honored(
        "(front stencil enable bit)",
        "DepthStencilDescriptor.front_stencil_enabled",
        "runtime/metal_draw/mod.rs:encode_draw_and_writeback",
    ),
    honored(
        "(back stencil enable bit)",
        "DepthStencilDescriptor.back_stencil_enabled",
        "runtime/metal_draw/mod.rs:encode_draw_and_writeback",
    ),
];

/// `MTLSamplerDescriptor`.
const SAMPLER: &[FieldCoverage] = &[
    honored(
        "minFilter",
        "SamplerDescriptor.min_filter",
        "backend/vulkan/translate/sampler.rs:filter",
    ),
    honored(
        "magFilter",
        "SamplerDescriptor.mag_filter",
        "backend/vulkan/translate/sampler.rs:filter",
    ),
    honored(
        "mipFilter",
        "SamplerDescriptor.mip_filter",
        "backend/vulkan/translate/sampler.rs:mip_filter",
    ),
    honored(
        "sAddressMode",
        "SamplerDescriptor.s_address",
        "backend/vulkan/translate/sampler.rs:address_mode",
    ),
    honored(
        "tAddressMode",
        "SamplerDescriptor.t_address",
        "backend/vulkan/translate/sampler.rs:address_mode",
    ),
    honored(
        "rAddressMode",
        "SamplerDescriptor.r_address",
        "backend/vulkan/translate/sampler.rs:address_mode",
    ),
    honored(
        "borderColor",
        "SamplerDescriptor.border_color",
        "backend/vulkan/translate/sampler.rs:border_color",
    ),
    honored(
        "compareFunction",
        "SamplerDescriptor.compare_function",
        "backend/vulkan/translate/raster.rs:compare_function",
    ),
    honored(
        "maxAnisotropy",
        "SamplerDescriptor.max_anisotropy",
        "runtime/metal_draw/mod.rs:vulkan_sampler_resource",
    ),
    honored(
        "lodMinClamp",
        "SamplerDescriptor.lod_min_clamp",
        "runtime/metal_draw/mod.rs:vulkan_sampler_resource",
    ),
    honored(
        "lodMaxClamp",
        "SamplerDescriptor.lod_max_clamp",
        "runtime/metal_draw/mod.rs:vulkan_sampler_resource",
    ),
    honored(
        "normalizedCoordinates",
        "SamplerDescriptor.normalized_coordinates",
        "runtime/metal_draw/mod.rs:vulkan_sampler_resource",
    ),
    // Both of these are bound on the **Metal** arm
    // (`backend/metal/samplers.rs` calls `set_lod_average` and
    // `set_support_argument_buffers`) and dropped on the Vulkan arm, where
    // `SamplerResource` has no field for either and nothing logs the loss. The
    // two arms therefore disagree about the same guest sampler, which is
    // exactly the class this manifest exists to surface.
    dropped(
        "lodAverage",
        "SamplerDescriptor.lod_average",
        "Metal averages the LOD across the footprint instead of computing it \
         per-sample. Vulkan's sampler has no equivalent knob, so this is \
         probably a genuine decline rather than a binding gap — but it must \
         say so by name instead of vanishing.",
    ),
    dropped(
        "supportArgumentBuffers",
        "SamplerDescriptor.support_argument_buffers",
        "A Metal residency/encoding concern with no Vulkan sampler counterpart. \
         Almost certainly correct to ignore; recorded because 'almost certainly \
         correct to ignore' is a judgement that should be written down once \
         rather than re-derived from an absence.",
    ),
];

/// `MTLRenderCommandEncoder` fixed-function state.
///
/// Not a descriptor, but the same question: the encoder's setters are where a
/// guest changes raster state per draw, and they are where the audit's list of
/// "hardcoded in the pipeline builder" fields actually live. Leaving them out
/// would put the manifest's most load-bearing gaps outside it.
const ENCODER_STATE: &[FieldCoverage] = &[
    honored(
        "setRenderPipelineState:",
        "DrawEncodeRequest.pipeline_ref",
        "runtime/metal_draw/mod.rs:load_render_pipeline",
    ),
    honored(
        "setDepthStencilState:",
        "DrawEncodeRequest.depth_stencil_ref",
        "runtime/metal_draw/mod.rs:load_depth_stencil_descriptor",
    ),
    honored(
        "setCullMode:",
        "DrawEncodeRequest.cull_mode",
        "backend/vulkan/translate/raster.rs:cull_mode",
    ),
    honored(
        "setFrontFacingWinding:",
        "DrawEncodeRequest.front_facing",
        "backend/vulkan/translate/raster.rs:front_face_ccw",
    ),
    honored(
        "setViewport:",
        "DrawEncodeRequest.viewport",
        "backend/vulkan/engine/caches.rs:vp_state",
    ),
    honored(
        "setScissorRect:",
        "DrawEncodeRequest.scissor",
        "backend/vulkan/engine/caches.rs:dynamic_states",
    ),
    honored(
        "setBlendColorRed:green:blue:alpha:",
        "DrawEncodeRequest.blend_color",
        "backend/vulkan/engine/caches.rs:blend_constants",
    ),
    honored(
        "setStencilReferenceValue:",
        "DrawEncodeRequest.stencil_ref",
        "backend/vulkan/engine/caches.rs:STENCIL_REFERENCE",
    ),
    // Decoded, and censused rather than bound. Metal's constant depth bias is
    // in depth-buffer units whose scale this backend cannot derive without
    // Apple ground truth, so binding a converted value would be a guess.
    declined(
        "setDepthBias:slopeScale:clamp:",
        "DrawEncodeRequest.depth_bias",
        "fixed_gap",
        "runtime/metal_draw/vulkan.rs:vulkan_fixed_state_gap",
    ),
    absent(
        "setDepthClipMode:",
        "Not decoded. `ReimsVgpuRasterState` has a `has_depth_clip_mode` flag and \
         `backend/metal/render.rs` calls `set_depth_clip_mode` when it is set — \
         but the only construction site hard-zeros the flag, so that encoder \
         code is unreachable on BOTH arms. Vulkan would need depthClampEnable \
         gated on the `depthClamp` device feature.",
    ),
    absent(
        "setTriangleFillMode:",
        "Not decoded, and unreachable on the Metal arm for the same reason as \
         setDepthClipMode: the `has_triangle_fill_mode` flag is never set. A \
         guest asking for MTLTriangleFillModeLines renders filled on every arm. \
         Vulkan would need polygonMode LINE gated on the fillModeNonSolid \
         device feature.",
    ),
    absent(
        "setVisibilityResultMode:offset:",
        "Not decoded. Occlusion query results have no binding in this backend \
         and no visibility result buffer is allocated.",
    ),
];

/// `MTLBlitCommandEncoder` records and their nested geometry.
const BLIT_COMMANDS: &[FieldCoverage] = decoded_fields! {
    honored "runtime/blit_exec.rs:execute_blit" => [
        "blit::Point.x",
        "blit::Point.y",
        "blit::Point.z",
        "blit::Size.width",
        "blit::Size.height",
        "blit::Size.depth",
        "blit::Command.opcode",
        "blit::Command.kind",
        "blit::Command.copy_kind",
        "blit::Command.source",
        "blit::Command.destination",
        "blit::Command.source_offset",
        "blit::Command.source_bytes_per_row",
        "blit::Command.source_bytes_per_image",
        "blit::Command.source_origin",
        "blit::Command.source_size",
        "blit::Command.destination_offset",
        "blit::Command.destination_bytes_per_row",
        "blit::Command.destination_bytes_per_image",
        "blit::Command.destination_origin",
        "blit::Command.size",
        "blit::Command.source_slice",
        "blit::Command.source_level",
        "blit::Command.destination_slice",
        "blit::Command.destination_level",
        "blit::Command.slice_count",
        "blit::Command.level_count",
        "blit::Command.has_options",
        "blit::Command.options",
        "blit::Command.resource",
        "blit::Command.buffer",
        "blit::Command.range_location",
        "blit::Command.range_length",
        "blit::Command.fill_value",
        "blit::Command.texture",
        "blit::Command.slice",
        "blit::Command.level",
        "blit::Command.fence",
    ];
};

/// Compute encoder state and dispatch records.
const COMPUTE_COMMANDS: &[FieldCoverage] = decoded_fields! {
    honored "runtime/compute_exec/mod.rs:apply_record" => [
        "compute::Size3.x",
        "compute::Size3.y",
        "compute::Size3.z",
        "compute::BufferBinding.ref_",
        "compute::BufferBinding.offset",
        "compute::BufferBinding.attribute_stride",
        "compute::BufferBinding.has_attribute_stride",
        "compute::RefBinding.ref_",
        "compute::SamplerBinding.ref_",
        "compute::SamplerBinding.lod_min_bits",
        "compute::SamplerBinding.lod_max_bits",
        "compute::SamplerBinding.has_lod_clamp",
        "compute::Region3.origin",
        "compute::Region3.size",
        "compute::Command.opcode",
        "compute::Command.kind",
        "compute::Command.pipeline_ref",
        "compute::Command.first",
        "compute::Command.count",
        "compute::Command.buffers",
        "compute::Command.textures",
        "compute::Command.samplers",
        "compute::Command.grid",
        "compute::Command.threads_per_threadgroup",
        "compute::Command.indirect_buffer_ref",
        "compute::Command.indirect_buffer_offset",
        "compute::Command.buffer_offset",
        "compute::Command.attribute_stride",
        "compute::Command.fence_ref",
    ];
    declined "linux_stage_in_imageblock" at "runtime/compute_exec/mod.rs:linux_stage_in_imageblock" => [
        "compute::Command.imageblock_width",
        "compute::Command.imageblock_height",
        "compute::Command.stage_in_region",
    ];
    declined "compute_session_no_vulkan_path" at "runtime/compute_session.rs:compute_session_no_vulkan_path" => [
        "compute::Command.condition_buffer_ref",
        "compute::Command.condition_buffer_offset",
        "compute::Command.condition_comparison",
        "compute::Command.condition_reference_value",
        "compute::Command.indirect_command_buffer_ref",
        "compute::Command.indirect_command_range_location",
        "compute::Command.indirect_command_range_length",
        "compute::Command.indirect_command_arguments_buffer_ref",
        "compute::Command.indirect_command_arguments_buffer_offset",
    ];
    dropped "Vulkan executes UseResources/UseHeaps and resource barriers as ordered no-ops, so these decoded residency and hazard operands never reach an API call or a refusal." => [
        "compute::Command.resources",
        "compute::Command.heaps",
        "compute::Command.resource_usage",
        "compute::Command.barrier_scope",
        "compute::Command.barrier_scope_reserved",
    ];
    dropped "The Vulkan direct-dispatch rail does not bind Metal dispatch-type, indirect stage-in, or dynamic threadgroup-memory state, and currently emits no field-specific refusal." => [
        "compute::Command.dispatch_type",
        "compute::Command.stage_in_indirect_buffer_ref",
        "compute::Command.stage_in_indirect_buffer_offset",
        "compute::Command.threadgroup_memory_length",
        "compute::Command.threadgroup_memory_index",
    ];
};

/// `MTLEvent` / `MTLSharedEvent` signal and wait records.
const EVENT_COMMANDS: &[FieldCoverage] = decoded_fields! {
    honored "runtime/fence_exec.rs:execute_event" => [
        "event::Command.opcode",
        "event::Command.kind",
        "event::Command.event_ref",
        "event::Command.value",
        "event::Command.has_timeout",
        "event::Command.timeout",
    ];
};

/// Guest resource-validity commands consumed by the FIFO drain.
const FIFO_RESOURCE_COMMANDS: &[FieldCoverage] = decoded_fields! {
    honored "runtime/drain/mod.rs:clear_host_valid" => [
        "fifo::InvalidateValidityOps.clear_host_valid",
        "fifo::InvalidateValidityOps.set_host_valid",
        "fifo::InvalidateValidityOps.clear_guest_valid",
        "fifo::InvalidateValidityOps.set_guest_valid",
        "fifo::InvalidateResourceRecord.object_id",
        "fifo::InvalidateResourceRecord.flags",
        "fifo::InvalidateResourceRecord.ops",
        "fifo::InvalidateResourcesCommand.task_id",
        "fifo::InvalidateResourcesCommand.count",
        "fifo::InvalidateResourcesCommand.records",
        "fifo::SynchronizeResourcesCommand.task_id",
        "fifo::SynchronizeResourcesCommand.count",
        "fifo::SynchronizeResourcesCommand.object_ids",
    ];
    honored "runtime/exec.rs:consume_resource_table" => [
        "fifo::ExecResourceDesc.object_id",
        "fifo::ExecResourceDesc.ops",
    ];
    declined "exec_res_table" at "runtime/exec.rs:consume_resource_table" => [
        // Zero across 84 868 records on the Ventura 13.7.8 x86 build, so their
        // unrecovered meaning costs nothing there. Read rather than dropped so a
        // build that populates them raises `exec_res_tail_populated` instead of
        // passing unread.
        "fifo::ExecResourceDesc.tail",
    ];
};

/// Full render command stream state, beyond the fixed-function request fields
/// already inventoried in `ENCODER_STATE`.
const RENDER_COMMANDS: &[FieldCoverage] = decoded_fields! {
    honored "runtime/exec.rs:handle_render_record" => [
        "render::Command.opcode",
        "render::Command.kind",
        "render::Command.stage",
        "render::Command.pipeline_ref",
        "render::Command.first",
        "render::Command.count",
        "render::Command.buffer_ref",
        "render::Command.buffer_offset",
        "render::Command.buffer_binds",
        "render::Command.texture_ref",
        "render::Command.ref_binds",
        "render::Command.sampler_ref",
        "render::Command.primitive_type",
        "render::Command.vertex_start",
        "render::Command.vertex_count",
        "render::Command.instance_count",
        "render::Command.index_count",
        "render::Command.index_type",
        "render::Command.index_buffer_ref",
        "render::Command.index_buffer_offset",
        "render::Command.viewport",
        "render::Command.scissor_x",
        "render::Command.scissor_y",
        "render::Command.scissor_w",
        "render::Command.scissor_h",
        "render::Command.fence_ref",
        "render::Command.blend_color",
        "render::Command.has_blend_color",
        "render::Command.cull_mode",
        "render::Command.has_cull_mode",
        "render::Command.front_facing",
        "render::Command.has_front_facing",
        "render::Command.depth_bias",
        "render::Command.has_depth_bias",
        "render::Command.depth_stencil_ref",
        "render::Command.stencil_ref_front",
        "render::Command.stencil_ref_back",
        "render::Command.has_stencil_ref",
        "render::Command.indirect_command_buffer_ref",
        "render::Command.icb_range_location",
        "render::Command.icb_range_length",
        "render::Command.icb_args_buffer_ref",
        "render::Command.icb_args_buffer_offset",
        "render::Command.icb_is_range",
    ];
    dropped "UseResource and UseHeap records are decoded but fall through the render executor's catch-all arm; their resource reference affects no Vulkan residency operation and emits no refusal." => [
        "render::Command.resource_ref",
    ];
};

/// Linear `MTLBuffer` allocation descriptor.
const BUFFER_DESCRIPTOR: &[FieldCoverage] = decoded_fields! {
    honored "runtime/compute_exec/mod.rs:buffer_gva_size" => [
        "resource::BufferDescriptor.allocation_size",
        "resource::BufferDescriptor.handle",
    ];
    dropped "The decoder truncates the 64-bit backing handle into the consumed 32-bit alias; non-zero high bits are discarded without a decline on every backend consumer." => [
        "resource::BufferDescriptor.handle64",
    ];
};

/// Linear/mipmapped texture storage and per-level geometry.
const TEXTURE_DESCRIPTOR: &[FieldCoverage] = decoded_fields! {
    honored "runtime/mipmap.rs:resolve_multi_mip_texture" => [
        "resource::TextureLevelLayout.offset",
        "resource::TextureLevelLayout.size",
        "resource::TextureLevelLayout.row_stride",
        "resource::TextureLevelLayout.width",
        "resource::TextureLevelLayout.height",
        "resource::TextureLevelLayout.depth",
        "resource::TextureDescriptor.allocation_size",
        "resource::TextureDescriptor.handle",
        "resource::TextureDescriptor.mipmap_level_count",
        "resource::TextureDescriptor.data_offset",
        "resource::TextureDescriptor.used_size",
        "resource::TextureDescriptor.row_stride",
        "resource::TextureDescriptor.width",
        "resource::TextureDescriptor.height",
        "resource::TextureDescriptor.depth",
        "resource::TextureDescriptor.pixel_format",
        "resource::TextureDescriptor.has_row_stride",
        "resource::TextureDescriptor.has_width",
        "resource::TextureDescriptor.has_height",
        "resource::TextureDescriptor.has_pixel_format",
        "resource::TextureDescriptor.levels",
    ];
    dropped "The linear texture decoder retains bytes-per-element, but Vulkan layout, upload, blit, mipmap, and sampling consumers derive size from pixel format and never read this wire field." => [
        "resource::TextureDescriptor.bytes_per_element",
    ];
};

/// `MTLVertexDescriptor` attribute/layout state embedded in the render
/// pipeline body.
const VERTEX_INPUT: &[FieldCoverage] = decoded_fields! {
    honored "runtime/metal_draw/vulkan.rs:try_metal2vulkan_draw" => [
        "resource::VertexAttribute.location",
        "resource::VertexAttribute.format",
        "resource::VertexAttribute.offset",
        "resource::VertexAttribute.buffer_index",
        "resource::VertexAttribute.stride",
        "resource::VertexAttribute.has_step_function",
        "resource::VertexAttribute.step_function",
        "resource::VertexAttribute.has_step_rate",
        "resource::VertexAttribute.step_rate",
    ];
};

/// Compute pipeline function and Metal stage-input descriptor.
const COMPUTE_PIPELINE: &[FieldCoverage] = decoded_fields! {
    honored "runtime/compute_exec/mod.rs:load_compute_pipeline" => [
        "resource::ComputePipelineDescriptor.kernel_func_ref",
    ];
    declined "linux_stage_in_imageblock" at "runtime/compute_exec/mod.rs:linux_stage_in_imageblock" => [
        "resource::ComputeStageInputAttribute.raw_bits",
        "resource::ComputeStageInputAttribute.location",
        "resource::ComputeStageInputAttribute.format",
        "resource::ComputeStageInputAttribute.offset",
        "resource::ComputeStageInputAttribute.buffer_index",
        "resource::ComputeStageInputLayout.raw_bits",
        "resource::ComputeStageInputLayout.buffer_index",
        "resource::ComputeStageInputLayout.step_function",
        "resource::ComputeStageInputLayout.step_rate",
        "resource::ComputeStageInputLayout.stride",
        "resource::ComputeStageInputDescriptor.word0",
        "resource::ComputeStageInputDescriptor.header0",
        "resource::ComputeStageInputDescriptor.header1",
        "resource::ComputeStageInputDescriptor.index_type",
        "resource::ComputeStageInputDescriptor.index_buffer_index",
        "resource::ComputeStageInputDescriptor.attributes",
        "resource::ComputeStageInputDescriptor.layouts",
        "resource::ComputePipelineDescriptor.stage_input",
    ];
    dropped "A stage-input table that exceeds decoder caps is converted to None before the Vulkan unsupported-stage-input refusal, so the two truncation counts silently erase the entire descriptor." => [
        "resource::ComputeStageInputDescriptor.dropped_attributes",
        "resource::ComputeStageInputDescriptor.dropped_layouts",
    ];
};

/// `MTLTexture` views, including ranged and swizzled views.
const TEXTURE_VIEW: &[FieldCoverage] = decoded_fields! {
    honored "runtime/metal_draw/mod.rs:resolve_texture_view_reasoned" => [
        "resource::TextureViewDescriptor.base_texture_ref",
        "resource::TextureViewDescriptor.pixel_format",
        "resource::TextureViewDescriptor.has_pixel_format",
        "resource::TextureViewDescriptor.texture_type",
        "resource::TextureViewDescriptor.has_texture_type",
        "resource::TextureViewDescriptor.level_base",
        "resource::TextureViewDescriptor.level_count",
        "resource::TextureViewDescriptor.has_levels",
        "resource::TextureViewDescriptor.slice_base",
        "resource::TextureViewDescriptor.slice_count",
        "resource::TextureViewDescriptor.has_slices",
        "resource::TextureViewDescriptor.swizzle",
        "resource::TextureViewDescriptor.has_swizzle",
    ];
};

/// AIR/MTLB function blob descriptor.
const FUNCTION_DESCRIPTOR: &[FieldCoverage] = decoded_fields! {
    honored "runtime/compute_exec/mod.rs:load_mtlb" => [
        "resource::FunctionDescriptor.blob_gva",
        "resource::FunctionDescriptor.blob_size",
    ];
    dropped "The function identifier is decoded and retained but both render and compute load the blob solely by the outer object reference; no Vulkan lookup or diagnostic consumes function_id." => [
        "resource::FunctionDescriptor.function_id",
    ];
};

/// `MTLIndirectCommandBufferDescriptor` and its serialized command-slot
/// layout. The Vulkan pathway refuses execution as a whole rather than
/// pretending any of these Metal-only mappings exist.
const ICB_DESCRIPTOR: &[FieldCoverage] = decoded_fields! {
    declined "icb_exec_no_metal_build" at "runtime/metal_draw/metal_icb.rs:icb_exec_no_metal_build" => [
        "resource::IcbCommandLayout.command_type_offset",
        "resource::IcbCommandLayout.barrier_offset",
        "resource::IcbCommandLayout.kernel_dispatch_arguments_offset",
        "resource::IcbCommandLayout.tessellation_factor_offset",
        "resource::IcbCommandLayout.pipeline_state_offset",
        "resource::IcbCommandLayout.vertex_buffer_bind_offset",
        "resource::IcbCommandLayout.fragment_buffer_bind_offset",
        "resource::IcbCommandLayout.object_buffer_bind_offset",
        "resource::IcbCommandLayout.mesh_buffer_bind_offset",
        "resource::IcbCommandLayout.kernel_buffer_bind_offset",
        "resource::IcbCommandLayout.attribute_stride_offset",
        "resource::IcbCommandLayout.object_threadgroup_memory_length_offset",
        "resource::IcbCommandLayout.threadgroup_memory_length_offset",
        "resource::IcbCommandLayout.command_arguments_offset",
        "resource::IcbCommandLayout.command_size",
        "resource::IndirectCommandBufferDescriptor.command_types",
        "resource::IndirectCommandBufferDescriptor.max_vertex_buffer_bind_count",
        "resource::IndirectCommandBufferDescriptor.max_fragment_buffer_bind_count",
        "resource::IndirectCommandBufferDescriptor.max_kernel_buffer_bind_count",
        "resource::IndirectCommandBufferDescriptor.max_object_buffer_bind_count",
        "resource::IndirectCommandBufferDescriptor.max_mesh_buffer_bind_count",
        "resource::IndirectCommandBufferDescriptor.max_kernel_threadgroup_memory_bind_count",
        "resource::IndirectCommandBufferDescriptor.max_object_threadgroup_memory_bind_count",
        "resource::IndirectCommandBufferDescriptor.inherit_buffers",
        "resource::IndirectCommandBufferDescriptor.inherit_pipeline_state",
        "resource::IndirectCommandBufferDescriptor.max_command_count",
        "resource::IndirectCommandBufferDescriptor.options",
        "resource::IndirectCommandBufferDescriptor.layout",
    ];
};

/// `newTextureWithDescriptor:offset:bytesPerRow:` over an `MTLBuffer`.
const BUFFER_TEXTURE: &[FieldCoverage] = decoded_fields! {
    honored "runtime/metal_draw/mod.rs:load_buffer_texture_rgba" => [
        "resource::BufferTextureDescriptor.buffer_ref",
        "resource::BufferTextureDescriptor.offset",
        "resource::BufferTextureDescriptor.bytes_per_row",
        "resource::BufferTextureDescriptor.pixel_format",
        "resource::BufferTextureDescriptor.width",
        "resource::BufferTextureDescriptor.height",
    ];
    dropped "The Vulkan buffer-texture loader always materializes one tight 2D level and ignores the decoded texture kind, depth, mip count, sample count, and array length without declining." => [
        "resource::BufferTextureDescriptor.texture_type",
        "resource::BufferTextureDescriptor.depth",
        "resource::BufferTextureDescriptor.mipmap_level_count",
        "resource::BufferTextureDescriptor.sample_count",
        "resource::BufferTextureDescriptor.array_length",
    ];
};

/// Every descriptor family this backend binds.
pub const MANIFEST: &[DescriptorFamily] = &[
    DescriptorFamily {
        descriptor: "MTLRenderPipelineDescriptor",
        decode_structs: &[
            "RenderPipelineDescriptor",
            "PipelineColorAttachment",
            "ColorWriteMask",
        ],
        fields: RENDER_PIPELINE,
    },
    DescriptorFamily {
        descriptor: "MTLRenderPassDescriptor",
        decode_structs: &["ColorAttachment", "DepthAttachment", "StencilAttachment"],
        fields: RENDER_PASS,
    },
    DescriptorFamily {
        descriptor: "MTLDepthStencilDescriptor",
        decode_structs: &["DepthStencilDescriptor", "DepthStencilFace"],
        fields: DEPTH_STENCIL,
    },
    DescriptorFamily {
        descriptor: "MTLSamplerDescriptor",
        decode_structs: &["SamplerDescriptor"],
        fields: SAMPLER,
    },
    DescriptorFamily {
        descriptor: "MTLRenderCommandEncoder (fixed-function state)",
        decode_structs: &["DrawEncodeRequest"],
        fields: ENCODER_STATE,
    },
    DescriptorFamily {
        descriptor: "MTLBlitCommandEncoder",
        decode_structs: &["blit::Point", "blit::Size", "blit::Command"],
        fields: BLIT_COMMANDS,
    },
    DescriptorFamily {
        descriptor: "MTLComputeCommandEncoder",
        decode_structs: &[
            "compute::Size3",
            "compute::BufferBinding",
            "compute::RefBinding",
            "compute::SamplerBinding",
            "compute::Region3",
            "compute::Command",
        ],
        fields: COMPUTE_COMMANDS,
    },
    DescriptorFamily {
        descriptor: "MTLEvent / MTLSharedEvent",
        decode_structs: &["event::Command"],
        fields: EVENT_COMMANDS,
    },
    DescriptorFamily {
        descriptor: "ApplePVGPU resource validity commands",
        decode_structs: &[
            "fifo::InvalidateValidityOps",
            "fifo::InvalidateResourceRecord",
            "fifo::InvalidateResourcesCommand",
            "fifo::SynchronizeResourcesCommand",
            "fifo::ExecResourceDesc",
        ],
        fields: FIFO_RESOURCE_COMMANDS,
    },
    DescriptorFamily {
        descriptor: "MTLRenderCommandEncoder (full command stream)",
        decode_structs: &["render::Command"],
        fields: RENDER_COMMANDS,
    },
    DescriptorFamily {
        descriptor: "MTLBuffer allocation",
        decode_structs: &["resource::BufferDescriptor"],
        fields: BUFFER_DESCRIPTOR,
    },
    DescriptorFamily {
        descriptor: "MTLTexture allocation and mip layouts",
        decode_structs: &[
            "resource::TextureLevelLayout",
            "resource::TextureDescriptor",
        ],
        fields: TEXTURE_DESCRIPTOR,
    },
    DescriptorFamily {
        descriptor: "MTLVertexDescriptor",
        decode_structs: &["resource::VertexAttribute"],
        fields: VERTEX_INPUT,
    },
    DescriptorFamily {
        descriptor: "MTLComputePipelineDescriptor / MTLStageInputOutputDescriptor",
        decode_structs: &[
            "resource::ComputeStageInputAttribute",
            "resource::ComputeStageInputLayout",
            "resource::ComputeStageInputDescriptor",
            "resource::ComputePipelineDescriptor",
        ],
        fields: COMPUTE_PIPELINE,
    },
    DescriptorFamily {
        descriptor: "MTLTexture view",
        decode_structs: &["resource::TextureViewDescriptor"],
        fields: TEXTURE_VIEW,
    },
    DescriptorFamily {
        descriptor: "Metal function blob",
        decode_structs: &["resource::FunctionDescriptor"],
        fields: FUNCTION_DESCRIPTOR,
    },
    DescriptorFamily {
        descriptor: "MTLIndirectCommandBufferDescriptor",
        decode_structs: &[
            "resource::IcbCommandLayout",
            "resource::IndirectCommandBufferDescriptor",
        ],
        fields: ICB_DESCRIPTOR,
    },
    DescriptorFamily {
        descriptor: "MTLBuffer-backed texture",
        decode_structs: &["resource::BufferTextureDescriptor"],
        fields: BUFFER_TEXTURE,
    },
];

/// Whole decode structs that are transport envelopes rather than guest GPU
/// state.
///
/// `runtime/decode/` holds 43 public structs and 21 public enums. Every struct
/// not named here is field-exhaustive in [`MANIFEST`].
pub const DECODE_STRUCT_EXCLUSIONS: &[(&str, &str)] = &[
    (
        "stream::Segment",
        "command-stream segment framing: byte ranges, chain bits, and offsets select records but do not encode Metal or resource state",
    ),
    (
        "stream::Record",
        "a normalized command-stream record envelope: segment identity and byte ranges route the payload to a rail decoder",
    ),
    (
        "resource::CompactTlv",
        "compact TLV framing retained by the resource decoder; its tag and offsets locate descriptor fields rather than becoming GPU state",
    ),
    (
        "resource::ListObjectEntry",
        "the task object-list routing envelope: object type and descriptor address locate the independently audited descriptor body",
    ),
    (
        "fifo::DisplayTimingEntry",
        "a host-to-guest display-timing response assembled by the FIFO rail, not a decoded guest descriptor or command",
    ),
];

/// Public decode enums are exhaustively inventoried separately because they
/// carry variants, not fields. The field-disposition gate covers structs; this
/// list makes the other half of the public decode surface impossible to omit.
pub const DECODE_ENUMS: &[&str] = &[
    "blit::BlitAspect",
    "blit::BlitOptionError",
    "blit::DecodeStatus",
    "blit::Kind",
    "blit::CopyKind",
    "blit::RefKind",
    "compute::DecodeStatus",
    "compute::OpcodeConfidence",
    "compute::Kind",
    "event::DecodeStatus",
    "event::Kind",
    "render::DecodeStatus",
    "render::Kind",
    "render::Stage",
    "resource::DecodeStatus",
    "resource::Descriptor",
    "stream::DecodeStatus",
    "stream::SegmentDisposition",
];

/// Decode-struct fields that carry no guest GPU state and so have no place in
/// the manifest: wire framing, object identity, and section offsets.
///
/// Each entry states why it is not descriptor state. This list is the one place
/// a decoded field may be excluded, so it stays short and reasoned rather than
/// becoming the escape hatch that makes the exhaustiveness test vacuous.
pub const DECODE_FIELD_EXCLUSIONS: &[(&str, &str)] = &[
    (
        "RenderPipelineDescriptor.object_id",
        "the type-7 object's own id — identity, not pipeline state",
    ),
    (
        "RenderPipelineDescriptor.word3",
        "an undecoded header word retained for tracing",
    ),
    (
        "RenderPipelineDescriptor.color_attachment_offset",
        "byte offset to the colour-attachment section — wire framing",
    ),
    (
        "RenderPipelineDescriptor.has_color_attachment_offset",
        "presence flag for the section offset above — wire framing",
    ),
    (
        "RenderPipelineDescriptor.color0",
        "compat alias for colour_attachments[0]; the slot's fields are listed once",
    ),
    (
        "RenderPipelineDescriptor.color_attachments",
        "the slot vector itself; its per-slot fields are listed individually",
    ),
    (
        "PipelineColorAttachment.slot",
        "the attachment's index, not a descriptor field",
    ),
    (
        "PipelineColorAttachment.has_pixel_format",
        "presence flag for pixel_format — wire framing",
    ),
    (
        "blit::Command.command_length",
        "record byte length validated by the blit decoder before semantic fields are produced",
    ),
    (
        "blit::Command.source_kind",
        "decoder-side reference classification duplicated by copy_kind and never exposed as Metal blit state",
    ),
    (
        "blit::Command.destination_kind",
        "decoder-side reference classification duplicated by copy_kind and never exposed as Metal blit state",
    ),
    (
        "blit::Command.resource_kind",
        "decoder-side reference classification for resource opcodes, not an independent encoder setting",
    ),
    (
        "compute::Command.command_length",
        "record byte length validated by the compute decoder before semantic fields are produced",
    ),
    (
        "compute::Command.confidence",
        "decoder confidence metadata used to classify observed opcodes, not guest compute encoder state",
    ),
    (
        "event::Command.command_length",
        "record byte length validated by the event decoder before semantic fields are produced",
    ),
    (
        "event::Command.raw_payload_offset",
        "diagnostic byte offset into the original event record, retained for tracing rather than execution",
    ),
    (
        "event::Command.raw_payload_length",
        "diagnostic byte length of the original event payload, retained for tracing rather than execution",
    ),
    (
        "render::Command.command_length",
        "record byte length validated by the render decoder before semantic fields are produced",
    ),
    (
        "render::Command.raw_payload_len",
        "diagnostic byte length of the original render payload, retained for tracing rather than execution",
    ),
    (
        "render::Command.color0",
        "container alias for the separately inventoried ColorAttachment fields; the executor also re-decodes all pass slots",
    ),
    (
        "render::Command.depth",
        "container alias for the separately inventoried DepthAttachment fields; the executor re-decodes the pass payload",
    ),
    (
        "render::Command.stencil",
        "container alias for the separately inventoried StencilAttachment fields; the executor re-decodes the pass payload",
    ),
    (
        "resource::TextureViewDescriptor.view_opcode",
        "descriptor-layout discriminator consumed while decoding the ranged/swizzled view body, not Metal view state itself",
    ),
    (
        "resource::TextureViewDescriptor.view_texture_ref",
        "the created view object's identity duplicated by the outer object-list entry, not view configuration",
    ),
    (
        "resource::BufferTextureDescriptor.new_texture_ref",
        "the created texture object's identity duplicated by the outer object-list entry, not buffer-texture configuration",
    ),
];

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    #[derive(Debug)]
    struct DecodeStruct {
        canonical: String,
        short: String,
        fields: Vec<String>,
    }

    fn crate_src() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src")
    }

    fn read(rel: &str) -> String {
        fs::read_to_string(crate_src().join(rel)).unwrap_or_else(|e| panic!("read {rel}: {e}"))
    }

    /// The `pub` field names of one struct in one source file.
    fn struct_fields(src: &str, name: &str) -> Vec<String> {
        let head = format!("pub struct {name} {{");
        let Some(start) = src.find(&head) else {
            panic!("struct {name} not found — the manifest names a decode struct that moved");
        };
        let body = &src[start + head.len()..];
        let end = body.find("\n}").expect("struct end");
        body[..end]
            .lines()
            .map(str::trim)
            .filter(|l| l.starts_with("pub "))
            .filter_map(|l| l.strip_prefix("pub "))
            .filter_map(|l| l.split(':').next())
            .map(|f| f.trim().to_string())
            .collect()
    }

    fn public_items(src: &str, kind: &str) -> Vec<String> {
        let prefix = format!("pub {kind} ");
        src.lines()
            .map(str::trim)
            .filter_map(|line| line.strip_prefix(&prefix))
            .filter_map(|tail| {
                tail.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                    .next()
            })
            .filter(|name| !name.is_empty())
            .map(str::to_string)
            .collect()
    }

    /// All public decoded structs, module-qualified so the four `Command`
    /// structs cannot alias one another.
    fn decode_structs() -> Vec<DecodeStruct> {
        let dir = crate_src().join("runtime/decode");
        let mut paths: Vec<PathBuf> = fs::read_dir(&dir)
            .expect("read runtime/decode")
            .map(|entry| entry.expect("decode dir entry").path())
            .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("rs"))
            .filter(|path| path.file_stem().and_then(|stem| stem.to_str()) != Some("mod"))
            .collect();
        paths.sort();

        let mut out = Vec::new();
        for path in paths {
            let module = path.file_stem().unwrap().to_str().unwrap();
            let src = fs::read_to_string(&path).expect("read decode source");
            for name in public_items(&src, "struct") {
                out.push(DecodeStruct {
                    canonical: format!("{module}::{name}"),
                    short: name.clone(),
                    fields: struct_fields(&src, &name),
                });
            }
        }
        out
    }

    fn decode_enums() -> Vec<String> {
        let dir = crate_src().join("runtime/decode");
        let mut paths: Vec<PathBuf> = fs::read_dir(&dir)
            .expect("read runtime/decode")
            .map(|entry| entry.expect("decode dir entry").path())
            .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("rs"))
            .filter(|path| path.file_stem().and_then(|stem| stem.to_str()) != Some("mod"))
            .collect();
        paths.sort();
        let mut out = Vec::new();
        for path in paths {
            let module = path.file_stem().unwrap().to_str().unwrap();
            let src = fs::read_to_string(&path).expect("read decode source");
            out.extend(
                public_items(&src, "enum")
                    .into_iter()
                    .map(|name| format!("{module}::{name}")),
            );
        }
        out
    }

    /// Resolve either a canonical `module::Struct.field` spelling or the old
    /// unique `Struct.field` spelling retained by the render manifest.
    fn resolve_decoded_field(raw: &str, structs: &[DecodeStruct]) -> Vec<String> {
        if raw.starts_with("DrawEncodeRequest.") {
            let src = read("runtime/metal_draw/mod.rs");
            let field = raw.split_once('.').unwrap().1;
            return struct_fields(&src, "DrawEncodeRequest")
                .into_iter()
                .filter(|candidate| candidate == field)
                .map(|_| raw.to_string())
                .collect();
        }
        let Some((owner, field)) = raw.rsplit_once('.') else {
            return Vec::new();
        };
        structs
            .iter()
            .filter(|st| st.canonical == owner || st.short == owner)
            .filter(|st| st.fields.iter().any(|candidate| candidate == field))
            .map(|st| format!("{}.{}", st.canonical, field))
            .collect()
    }

    fn resolve_struct(raw: &str, structs: &[DecodeStruct]) -> Vec<String> {
        if raw == "DrawEncodeRequest" {
            return vec![raw.to_string()];
        }
        structs
            .iter()
            .filter(|st| st.canonical == raw || st.short == raw)
            .map(|st| st.canonical.clone())
            .collect()
    }

    /// Every field the decoder produces has a disposition.
    ///
    /// This is the property that makes the manifest more than documentation: a
    /// new field on a decode struct fails this test until someone decides
    /// whether the pipeline binds it, declines it, or does neither.
    #[test]
    fn every_decoded_field_has_a_disposition() {
        let structs = decode_structs();
        let listed: Vec<String> = MANIFEST
            .iter()
            .flat_map(|f| f.fields.iter())
            .filter_map(|f| f.decoded_as)
            .flat_map(|field| resolve_decoded_field(field, &structs))
            .collect();
        let excused_fields: Vec<String> = DECODE_FIELD_EXCLUSIONS
            .iter()
            .flat_map(|(field, _)| resolve_decoded_field(field, &structs))
            .collect();
        let excused_structs: Vec<&str> =
            DECODE_STRUCT_EXCLUSIONS.iter().map(|(st, _)| *st).collect();

        let mut missing = Vec::new();
        for st in &structs {
            if excused_structs.contains(&st.canonical.as_str()) {
                continue;
            }
            for field in &st.fields {
                let qualified = format!("{}.{}", st.canonical, field);
                if !listed.contains(&qualified) && !excused_fields.contains(&qualified) {
                    missing.push(qualified);
                }
            }
        }
        assert!(
            missing.is_empty(),
            "these decoded fields have no disposition — say whether the \
             pipeline honours them, declines them by name, or neither:\n  {}",
            missing.join("\n  ")
        );
    }

    /// The manifest may not name a decode field that does not exist. Without
    /// this the test above passes by listing fields nobody has.
    #[test]
    fn the_manifest_names_only_real_decode_fields() {
        let structs = decode_structs();
        let mut bogus = Vec::new();
        for family in MANIFEST {
            for f in family.fields {
                if let Some(d) = f.decoded_as {
                    let resolved = resolve_decoded_field(d, &structs);
                    if resolved.len() != 1 {
                        bogus.push(format!("{} -> {d}", f.field));
                    }
                }
            }
        }
        for (excused, _) in DECODE_FIELD_EXCLUSIONS {
            if resolve_decoded_field(excused, &structs).len() != 1 {
                bogus.push(format!("(excused) {excused}"));
            }
        }
        for (excused, _) in DECODE_STRUCT_EXCLUSIONS {
            if resolve_struct(excused, &structs).len() != 1 {
                bogus.push(format!("(excused struct) {excused}"));
            }
        }
        for family in MANIFEST {
            for named in family.decode_structs {
                if resolve_struct(named, &structs).len() != 1 {
                    bogus.push(format!("{} -> struct {named}", family.descriptor));
                }
            }
        }
        assert!(
            bogus.is_empty(),
            "the manifest points at decode fields that do not exist:\n  {}",
            bogus.join("\n  ")
        );
    }

    #[test]
    fn the_public_decode_type_census_is_pinned_and_complete() {
        let structs = decode_structs();
        let mut actual_enums = decode_enums();
        let mut recorded_enums: Vec<String> = DECODE_ENUMS
            .iter()
            .map(|name| (*name).to_string())
            .collect();
        actual_enums.sort();
        recorded_enums.sort();
        assert_eq!(
            actual_enums, recorded_enums,
            "public decode enums changed; inventory the new type in DECODE_ENUMS"
        );
        assert_eq!(
            (structs.len(), actual_enums.len()),
            (43, 18),
            "the public decode type census moved; keep the 43-struct field \
             manifest and 18-enum inventory exhaustive, then update this pin"
        );
    }

    #[test]
    fn exclusions_are_small_reasoned_and_disjoint_from_the_manifest() {
        assert!(
            DECODE_STRUCT_EXCLUSIONS.len() <= 7,
            "whole-struct exclusions grew; audit the new struct field by field"
        );
        for (name, note) in DECODE_STRUCT_EXCLUSIONS {
            assert!(note.len() > 80, "{name} has no useful exclusion reason");
        }
        for (name, note) in DECODE_FIELD_EXCLUSIONS {
            assert!(note.len() > 40, "{name} has no useful exclusion reason");
        }
        let structs = decode_structs();
        let listed: Vec<String> = MANIFEST
            .iter()
            .flat_map(|family| family.fields)
            .filter_map(|field| field.decoded_as)
            .flat_map(|field| resolve_decoded_field(field, &structs))
            .collect();
        for (name, _) in DECODE_FIELD_EXCLUSIONS {
            let resolved = resolve_decoded_field(name, &structs);
            assert!(
                resolved.iter().all(|field| !listed.contains(field)),
                "{name} is both manifested and excluded"
            );
        }
    }

    #[test]
    fn every_public_decode_struct_is_manifested_or_excluded() {
        let structs = decode_structs();
        let named: Vec<String> = MANIFEST
            .iter()
            .flat_map(|family| family.decode_structs)
            .flat_map(|name| resolve_struct(name, &structs))
            .collect();
        let excluded: Vec<&str> = DECODE_STRUCT_EXCLUSIONS
            .iter()
            .map(|(name, _)| *name)
            .collect();
        let missing: Vec<&str> = structs
            .iter()
            .filter(|st| {
                !named.contains(&st.canonical) && !excluded.contains(&st.canonical.as_str())
            })
            .map(|st| st.canonical.as_str())
            .collect();
        assert!(
            missing.is_empty(),
            "public decode structs are neither manifested nor explicitly \
             excluded:\n  {}",
            missing.join("\n  ")
        );
    }

    #[test]
    fn decoded_fields_are_listed_once_globally() {
        let structs = decode_structs();
        let mut seen = Vec::new();
        let mut duplicates = Vec::new();
        for field in MANIFEST
            .iter()
            .flat_map(|family| family.fields)
            .filter_map(|field| field.decoded_as)
            .flat_map(|field| resolve_decoded_field(field, &structs))
        {
            if seen.contains(&field) {
                duplicates.push(field);
            } else {
                seen.push(field);
            }
        }
        assert!(
            duplicates.is_empty(),
            "decoded fields have more than one disposition:\n  {}",
            duplicates.join("\n  ")
        );
    }

    /// An `Honored` disposition must name source that is still there.
    #[test]
    fn honored_sites_exist() {
        let mut dangling = Vec::new();
        for family in MANIFEST {
            for f in family.fields {
                let Disposition::Honored { site } = f.disposition else {
                    continue;
                };
                let (file, symbol) = site.split_once(':').unwrap_or_else(|| {
                    panic!("{} site `{site}` must be `path.rs:symbol`", f.field)
                });
                let path = crate_src().join(file);
                if !path.exists() {
                    dangling.push(format!("{}: file {file} is gone", f.field));
                    continue;
                }
                let src = fs::read_to_string(&path).expect("read honored site");
                if !src.contains(symbol) {
                    dangling.push(format!("{}: {file} no longer contains `{symbol}`", f.field));
                }
            }
        }
        assert!(
            dangling.is_empty(),
            "these fields claim to be bound at a site that no longer exists:\n  {}",
            dangling.join("\n  ")
        );
    }

    /// The check that would have caught `color_write_mask(RGBA)`.
    ///
    /// The pipeline builder pins some Vulkan state to a constant regardless of
    /// guest state. A field whose value is pinned is by definition not
    /// honoured, so claiming both is a contradiction the manifest must not be
    /// able to express. Each entry pairs a builder call with the manifest field
    /// it overrides.
    #[test]
    fn a_pinned_builder_value_is_never_claimed_honored() {
        // `.color_write_mask(vk::ColorComponentFlags::RGBA)` used to head this
        // list and is gone: the builder now derives the mask from the guest's
        // decoded `writeMask`, so the field is `Honored` and there is nothing
        // left to pin it against.
        const PINNED: &[(&str, &str)] = &[(
            ".rasterization_samples(vk::SampleCountFlags::TYPE_1)",
            "rasterSampleCount",
        )];
        let builder = read("backend/vulkan/engine/caches.rs");
        for (call, field) in PINNED {
            assert!(
                builder.contains(call),
                "`{call}` is recorded as pinned but no longer appears in the \
                 pipeline builder — if it now reads guest state, promote \
                 `{field}` to Honored and drop this entry"
            );
            let entry = MANIFEST
                .iter()
                .flat_map(|f| f.fields.iter())
                .find(|f| f.field == *field)
                .unwrap_or_else(|| panic!("manifest has no entry for {field}"));
            assert!(
                !entry.disposition.is_honored(),
                "{field} claims to be honoured, but the builder pins it with \
                 `{call}` — one of the two is lying, and it is not the builder"
            );
        }
    }

    /// No field is listed twice within a family: a duplicate makes the
    /// exhaustiveness count meaningless and hides whichever disposition loses.
    #[test]
    fn fields_are_listed_once_per_family() {
        for family in MANIFEST {
            let mut seen: Vec<&str> = Vec::new();
            for f in family.fields {
                assert!(
                    !seen.contains(&f.field),
                    "{} lists {} twice",
                    family.descriptor,
                    f.field
                );
                seen.push(f.field);
            }
        }
    }

    /// A gap must actually say something. An empty or terse note turns the
    /// honest state back into the absence it was supposed to replace.
    #[test]
    fn gaps_carry_a_real_note() {
        for family in MANIFEST {
            for f in family.fields {
                match f.disposition {
                    Disposition::NotOnTheWire { note } => {
                        assert!(
                            note.len() > 40,
                            "{}::{} records no useful note about the gap",
                            family.descriptor,
                            f.field
                        );
                        assert!(
                            f.decoded_as.is_none(),
                            "{}::{} is NotOnTheWire but names a decoded field",
                            family.descriptor,
                            f.field
                        );
                    }
                    Disposition::DroppedSilently { note } => {
                        assert!(
                            note.len() > 40,
                            "{}::{} is dropped silently with no explanation",
                            family.descriptor,
                            f.field
                        );
                        assert!(
                            f.decoded_as.is_some(),
                            "{}::{} is DroppedSilently but names no decoded \
                             field — if it is not decoded it is NotOnTheWire",
                            family.descriptor,
                            f.field
                        );
                    }
                    _ => {}
                }
            }
        }
    }

    /// A `Declined` field claims a `reason=` slug reaches the always-on log.
    /// That claim is checkable: the slug must appear in the source that emits
    /// it. Otherwise the manifest promises a diagnostic nobody will find.
    #[test]
    fn declined_slugs_are_actually_emitted() {
        let mut missing = Vec::new();
        for family in MANIFEST {
            for f in family.fields {
                let Disposition::Declined { slug, site } = f.disposition else {
                    continue;
                };
                let file = site.split(':').next().unwrap_or(site);
                let src = fs::read_to_string(crate_src().join(file))
                    .unwrap_or_else(|e| panic!("{}: read {file}: {e}", f.field));
                if !src.contains(slug) {
                    missing.push(format!("{}: {file} never emits `{slug}`", f.field));
                }
            }
        }
        assert!(
            missing.is_empty(),
            "these fields claim a fail-visible decline that is not emitted:\n  {}",
            missing.join("\n  ")
        );
    }

    /// What the manifest currently says, as a number. Not a target — a
    /// baseline, so a change that closes gaps can show it moved and a change
    /// that silently loses coverage cannot.
    ///
    /// `dropped` is the one to watch: it counts decoded guest state that
    /// vanishes with no log line, which the ground rules forbid outright. It
    /// should only ever go down.
    #[test]
    fn the_coverage_census_is_what_the_last_audit_recorded() {
        // This ceiling is shrink-only: never raise it when updating the exact
        // census below. A repaired field lowers both numbers in the same commit.
        const DROPPED_SILENTLY_CEILING: usize = 23;
        let (mut honored, mut declined, mut dropped, mut absent) = (0, 0, 0, 0);
        for family in MANIFEST {
            for f in family.fields {
                match f.disposition {
                    Disposition::Honored { .. } => honored += 1,
                    Disposition::Declined { .. } => declined += 1,
                    Disposition::DroppedSilently { .. } => dropped += 1,
                    Disposition::NotOnTheWire { .. } => absent += 1,
                }
            }
        }
        assert_eq!(
            (honored, declined, dropped, absent),
            // Moved 2026-08-01, second: `ExecResourceDesc.flags` left the
            // manifest with the field. It was the raw dword `ops` decodes from,
            // kept only for the `exec_res_table` census histogram, so deleting
            // that census left nothing reading it. `declined` fell by one.
            //
            // Moved 2026-08-01: the four `ExecResourceDesc` fields entered the
            // manifest — the EXEC_INDIRECT2 resource table, which the device
            // used to step over unread. `object_id` and `ops` are Honored by
            // `consume_resource_table`; `tail` is Declined there too.
            //
            // Moved 2026-07-30: `colorAttachments[n].writeMask` went
            // NotOnTheWire -> Honored, so `absent` fell by one and `honored`
            // rose by one. It is on the wire after all, as tag 0x09.
            (253, 61, 23, 24),
            "the coverage census moved; update this baseline in the same commit \
             that moves it, and describe which way it moved"
        );
        assert!(
            dropped <= DROPPED_SILENTLY_CEILING,
            "DroppedSilently grew from the exhaustive baseline; classify the \
             new field honestly, but do not raise the shrink-only ceiling"
        );
    }
}
