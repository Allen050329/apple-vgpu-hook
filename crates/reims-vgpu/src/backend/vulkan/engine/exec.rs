//! Record / submit (bounded fence) / readback for one draw.

#![allow(unsafe_op_in_unsafe_fn)]

use ash::vk;
use std::collections::BTreeSet;

use super::caches::{
    AttrKey, BindingSig, LayoutKey, ObjectCaches, PassKey, PipelineKey, SecondaryAttachKey,
    MAX_SECONDARY_ATTACH,
};
use super::context::ContextOwner;
use super::counters::EngineCounters;
use super::device_lost::{DeviceLostDecline, DeviceLostOp};
use super::draw_execution::DrawExecutionDecline;
use super::draw_validation::DrawValidationDecline;
use super::pools::{BufferSlot, ResourcePools, SampledSlot, TargetKey};
use super::types::{
    BufferContent, DrawError, DrawOutput, DrawRequest, LoadOp, SampledSource, ScissorResource,
    VertexStepFunction, ViewportResource,
};
use super::vk_call::{VkCall, VkOp};

/// Stage one draw-time buffer content into a pooled slot, deduplicating
/// within the draw: several binds sharing one content (an `Arc`'d byte
/// allocation, or the same gathered guest span) get ONE slot and ONE gather.
/// Returns a handle copy of the slot.
///
/// `snapshot_volatile` — set for a deferred-submit (batched) draw — no longer
/// selects a mechanism. Every `GuestRuns` bind is now copied on the CPU at
/// preparation time, which is what the snapshot always did and what the
/// deferred CB needs: a gather recorded into a batched CB would have read guest
/// RAM at flush time, after ack-fast let the guest repaint the pages. The flag
/// survives because the counter it feeds distinguishes a batched bind from an
/// immediate one, which is still a real difference in when the bytes were read.
#[allow(
    clippy::too_many_arguments,
    reason = "buffer staging carries the Vulkan context, pools, binding, and lifetime sets"
)]
unsafe fn stage_buffer_content(
    ctx: &super::context::DeviceContext,
    pools: &mut ResourcePools,
    counters: &EngineCounters,
    content: &BufferContent,
    usage: vk::BufferUsageFlags,
    snapshot_volatile: bool,
    slots_by_content: &mut std::collections::HashMap<(usize, u64), BufferSlot>,
) -> Result<BufferSlot, DrawError> {
    let key = match content {
        BufferContent::Bytes(b) => (std::sync::Arc::as_ptr(b) as usize, b.len() as u64),
        BufferContent::GuestRuns(src) => (
            std::sync::Arc::as_ptr(&src.runs) as *const () as usize,
            src.total_len,
        ),
    };
    if let Some(slot) = slots_by_content.get(&key) {
        return Ok(*slot);
    }
    let slot = match content {
        BufferContent::Bytes(b) => {
            let slot = pools.acquire_staging(ctx, b.len() as u64, usage, counters)?;
            pools.write_staging(ctx, &slot, b)?;
            slot
        }
        // Guest runs are gathered by the CPU into the mapped staging span, with
        // no intermediate `cpu_bytes()` heap Vec (this is the deferred-submit
        // hot path, ~4.8 binds/draw under compositing).
        //
        // There used to be a second arm here that skipped this copy entirely:
        // it resolved each run through a `VK_EXT_external_memory_host` import
        // and had the command buffer read the guest pages directly. That is
        // gone — an imported host pointer is one the GPU can *write*, and the
        // runs point into guest RAM. `snapshot_volatile` therefore no longer
        // selects between two mechanisms; it only records that the runtime
        // asked for a stable snapshot, which is what this arm has always given
        // it.
        BufferContent::GuestRuns(src) => {
            let slot = pools.acquire_staging(ctx, src.total_len, usage, counters)?;
            pools.write_staging_from_runs(ctx, &slot, &src.runs, src.total_len)?;
            if snapshot_volatile {
                counters
                    .buffer_snapshot_binds
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            } else {
                crate::observe::ZeroCopyLost::BufferGuestRuns.note();
            }
            slot
        }
    };
    slots_by_content.insert(key, slot);
    Ok(slot)
}

enum PreparedSampled {
    Upload {
        binding: u32,
        image: SampledSlot,
        staging: BufferSlot,
        volume: bool,
        layers: u32,
    },
    /// Guest gather: the CPU packs the texel bytes out of the guest runs into a
    /// pooled scratch, then one buffer→image copy uploads it. No owned CPU byte
    /// buffer exists, so the content cache is bypassed and the slot is never
    /// retained — that is what still separates this from [`Self::Upload`].
    ///
    /// The gather used to happen on the device, out of per-run host-pointer
    /// imports of the guest pages, which is why the scratch carried
    /// `TRANSFER_DST` and a per-source copy list. Both are gone with the import.
    GuestGather {
        binding: u32,
        image: SampledSlot,
        scratch: BufferSlot,
        /// `bufferRowLength` for the buffer→image copy (0 = tight rows).
        row_length_texels: u32,
    },
    Cached {
        binding: u32,
        image: SampledSlot,
    },
    Resident {
        binding: u32,
        identity: super::types::TargetIdentity,
        image: vk::Image,
        view: vk::ImageView,
        old_layout: vk::ImageLayout,
    },
    Snapshot {
        binding: u32,
        identity: super::types::TargetIdentity,
        source_image: vk::Image,
        source_old_layout: vk::ImageLayout,
        image: SampledSlot,
    },
}

impl PreparedSampled {
    fn binding(&self) -> u32 {
        match self {
            Self::Upload { binding, .. }
            | Self::Cached { binding, .. }
            | Self::Resident { binding, .. }
            | Self::Snapshot { binding, .. }
            | Self::GuestGather { binding, .. } => *binding,
        }
    }

    fn view(&self) -> vk::ImageView {
        match self {
            Self::Upload { image, .. } => image.view,
            Self::Cached { image, .. } => image.view,
            Self::Resident { view, .. } => *view,
            Self::Snapshot { image, .. } => image.view,
            Self::GuestGather { image, .. } => image.view,
        }
    }
}

/// Shared validation for a draw-time buffer's content source. A `GuestRuns`
/// span must be internally consistent: the run lengths sum to `total_len`,
/// the span is non-empty, and `row_length_texels` is 0 (row strides are a
/// texture concept — buffers gather a flat byte span).
#[derive(Clone, Copy)]
enum BufferValidationRole {
    Vertex,
    Storage,
}

fn validate_buffer_content(
    content: &BufferContent,
    role: BufferValidationRole,
    resource_index: u32,
) -> Result<(), DrawError> {
    let BufferContent::GuestRuns(src) = content else {
        return Ok(());
    };
    if src.row_length_texels != 0 {
        let decline = match role {
            BufferValidationRole::Vertex => DrawValidationDecline::VertexGuestRunsRowStride {
                location: resource_index,
                row_length_texels: src.row_length_texels,
            },
            BufferValidationRole::Storage => DrawValidationDecline::StorageGuestRunsRowStride {
                binding: resource_index,
                row_length_texels: src.row_length_texels,
            },
        };
        return Err(DrawError::DrawValidation(decline));
    }
    let sum: u64 = src.runs.iter().map(|r| r.len).sum();
    if sum != src.total_len || src.total_len == 0 {
        let decline = match role {
            BufferValidationRole::Vertex => DrawValidationDecline::VertexGuestRunsCoverage {
                location: resource_index,
                covered: sum,
                declared: src.total_len,
            },
            BufferValidationRole::Storage => DrawValidationDecline::StorageGuestRunsCoverage {
                binding: resource_index,
                covered: sum,
                declared: src.total_len,
            },
        };
        return Err(DrawError::DrawValidation(decline));
    }
    Ok(())
}

pub(crate) fn validate_v1(req: &DrawRequest) -> Result<(), DrawError> {
    if req.width == 0 || req.height == 0 {
        return Err(DrawError::DrawValidation(
            DrawValidationDecline::ZeroTargetGeometry {
                width: req.width,
                height: req.height,
            },
        ));
    }
    if req.vert_spirv.is_empty() {
        return Err(DrawError::DrawValidation(
            DrawValidationDecline::EmptyVertexSpirv,
        ));
    }
    if req.frag_spirv.is_empty() {
        return Err(DrawError::DrawValidation(
            DrawValidationDecline::EmptyFragmentSpirv,
        ));
    }
    if req.viewports.len() > 1 || req.scissors.len() > 1 {
        return Err(DrawError::Unsupported(
            super::reason::DrawReason::MultiViewportArray {
                count: req.viewports.len().max(req.scissors.len()),
            },
        ));
    }
    for vp in &req.viewports {
        if !vp.x.is_finite()
            || !vp.y.is_finite()
            || !vp.width.is_finite()
            || !vp.height.is_finite()
            || !vp.min_depth.is_finite()
            || !vp.max_depth.is_finite()
        {
            return Err(DrawError::DrawValidation(
                DrawValidationDecline::NonFiniteViewport,
            ));
        }
        if vp.width <= 0.0 || vp.height <= 0.0 {
            return Err(DrawError::DrawValidation(
                DrawValidationDecline::NonPositiveViewport {
                    width_bits: vp.width.to_bits(),
                    height_bits: vp.height.to_bits(),
                },
            ));
        }
    }
    if let Some(blend) = req.blend {
        if blend.constants.iter().any(|c| !c.is_finite()) {
            return Err(DrawError::DrawValidation(
                DrawValidationDecline::NonFiniteBlendConstants,
            ));
        }
    }
    if let Some(target) = &req.target_rgba8 {
        let expected = req.width as usize * req.height as usize * 4;
        if target.len() != expected {
            return Err(DrawError::DrawValidation(
                DrawValidationDecline::TargetSeedLength {
                    actual: target.len(),
                    expected,
                },
            ));
        }
    }
    if let Some(LoadOp::LoadSeed(seed)) = &req.load_op {
        let expected = req.width as usize * req.height as usize * 4;
        if seed.len() != expected {
            return Err(DrawError::DrawValidation(
                DrawValidationDecline::LoadSeedLength {
                    actual: seed.len(),
                    expected,
                },
            ));
        }
    }
    if let Some(seed_identity) = &req.seed_from_target {
        if req.target_identity.is_none() {
            return Err(DrawError::DrawValidation(
                DrawValidationDecline::SeedMissingTargetIdentity,
            ));
        }
        if req.target_rgba8.is_some() || matches!(req.load_op, Some(LoadOp::LoadSeed(_))) {
            return Err(DrawError::DrawValidation(
                DrawValidationDecline::SeedConflictsCpuSeed,
            ));
        }
        if matches!(req.load_op, Some(LoadOp::LoadFromTarget)) {
            return Err(DrawError::DrawValidation(
                DrawValidationDecline::SeedConflictsLoadFromTarget,
            ));
        }
        if req.target_identity.as_ref() == Some(seed_identity) {
            return Err(DrawError::DrawValidation(
                DrawValidationDecline::SeedEqualsTarget,
            ));
        }
        if req.sampled_images.iter().any(|img| match &img.source {
            SampledSource::Target(identity) => identity == seed_identity,
            SampledSource::Bytes(_) | SampledSource::GuestRuns(_) => false,
        }) {
            return Err(DrawError::DrawValidation(
                DrawValidationDecline::SeedAlsoSampled,
            ));
        }
    }
    let no_vertex_fetch = draw_has_no_invocations(req);
    let last_record: u32 = match &req.indexed {
        Some(indexed) => {
            let need = indexed.index_count as usize * indexed.index_type.byte_size();
            if indexed.indices.len() < need {
                return Err(DrawError::DrawValidation(
                    DrawValidationDecline::IndexBytesShort {
                        actual: indexed.indices.len(),
                        expected: need,
                    },
                ));
            }
            if no_vertex_fetch {
                0
            } else {
                let (min_index, max_index) = indexed.index_range();
                let first = i64::from(min_index) + i64::from(indexed.vertex_offset);
                let last = i64::from(max_index) + i64::from(indexed.vertex_offset);
                if first < 0 || last < 0 || last > u32::MAX as i64 {
                    return Err(DrawError::DrawValidation(
                        DrawValidationDecline::IndexedVertexRange {
                            min_index,
                            max_index,
                            vertex_offset: indexed.vertex_offset,
                        },
                    ));
                }
                last as u32
            }
        }
        None => req.vertex_count.saturating_sub(1),
    };
    let mut bindings = BTreeSet::new();
    let mut vertex_locations = BTreeSet::new();
    let mut vertex_bindings = BTreeSet::new();
    for attribute in &req.vertex_attributes {
        if !vertex_locations.insert(attribute.location) {
            return Err(DrawError::DrawValidation(
                DrawValidationDecline::DuplicateVertexLocation {
                    location: attribute.location,
                },
            ));
        }
        if !vertex_bindings.insert(attribute.binding) {
            return Err(DrawError::DrawValidation(
                DrawValidationDecline::DuplicateVertexBinding {
                    binding: attribute.binding,
                },
            ));
        }
        if attribute.step_rate == 0 {
            return Err(DrawError::DrawValidation(
                DrawValidationDecline::ZeroVertexStepRate {
                    location: attribute.location,
                },
            ));
        }
        let format_size = attribute.format.byte_size();
        if attribute.stride < format_size {
            return Err(DrawError::DrawValidation(
                DrawValidationDecline::VertexStrideTooSmall {
                    location: attribute.location,
                    stride: attribute.stride,
                    format_size,
                },
            ));
        }
        let element_end = attribute.offset.checked_add(format_size).ok_or({
            DrawError::DrawValidation(DrawValidationDecline::VertexOffsetOverflow {
                location: attribute.location,
            })
        })?;
        if element_end > attribute.stride {
            return Err(DrawError::DrawValidation(
                DrawValidationDecline::VertexElementExceedsStride {
                    location: attribute.location,
                },
            ));
        }
        let last_element = if no_vertex_fetch {
            0
        } else {
            match attribute.step_function {
                VertexStepFunction::Constant => 0,
                VertexStepFunction::PerVertex => {
                    let first_record = if req.indexed.is_some() {
                        0
                    } else {
                        req.first_vertex as usize
                    };
                    first_record.checked_add(last_record as usize).ok_or({
                        DrawError::DrawValidation(DrawValidationDecline::VertexRangeOverflow {
                            location: attribute.location,
                        })
                    })?
                }
                VertexStepFunction::PerInstance => {
                    let instance_count = req.instance_count.unwrap_or(1);
                    let relative_element = if instance_count == 0 {
                        0
                    } else {
                        (instance_count - 1) / attribute.step_rate
                    };
                    req.base_instance.checked_add(relative_element).ok_or({
                        DrawError::DrawValidation(DrawValidationDecline::InstanceRangeOverflow {
                            location: attribute.location,
                        })
                    })? as usize
                }
            }
        };
        let required = (attribute.stride as usize)
            .checked_mul(last_element)
            .and_then(|span| (attribute.offset as usize).checked_add(span))
            .and_then(|end| end.checked_add(format_size as usize))
            .ok_or({
                DrawError::DrawValidation(DrawValidationDecline::VertexByteRangeOverflow {
                    location: attribute.location,
                })
            })?;
        if !no_vertex_fetch && attribute.content.len() < required {
            return Err(DrawError::DrawValidation(
                DrawValidationDecline::VertexDataShort {
                    location: attribute.location,
                    actual: attribute.content.len(),
                    expected: required,
                },
            ));
        }
        validate_buffer_content(
            &attribute.content,
            BufferValidationRole::Vertex,
            attribute.location,
        )?;
        // The Constant-step base-instance shift prepends a CPU prefix to the
        // bytes at prepare time; a gathered guest span has no CPU bytes. The
        // runtime keeps Constant-step streams on the CPU path — reaching here
        // with a gather is a gate bug, rejected before any GPU work.
        if !no_vertex_fetch
            && attribute.step_function == VertexStepFunction::Constant
            && req.base_instance != 0
            && matches!(attribute.content, BufferContent::GuestRuns(_))
        {
            return Err(DrawError::DrawValidation(
                DrawValidationDecline::ConstantStepGuestRuns {
                    location: attribute.location,
                },
            ));
        }
    }
    for buffer in &req.storage_buffers {
        if !bindings.insert(buffer.binding) {
            return Err(DrawError::DrawValidation(
                DrawValidationDecline::DuplicateStorageDescriptorBinding {
                    binding: buffer.binding,
                },
            ));
        }
        validate_buffer_content(
            &buffer.content,
            BufferValidationRole::Storage,
            buffer.binding,
        )?;
    }
    for image in &req.sampled_images {
        if image.width == 0 || image.height == 0 || image.layers == 0 {
            return Err(DrawError::DrawValidation(
                DrawValidationDecline::SampledZeroGeometry {
                    binding: image.binding,
                    width: image.width,
                    height: image.height,
                    layers: image.layers,
                },
            ));
        }
        if (image.arrayed as u8 + image.volume as u8 + image.cube as u8) > 1 {
            return Err(DrawError::DrawValidation(
                DrawValidationDecline::SampledShapeConflict {
                    binding: image.binding,
                    arrayed: image.arrayed,
                    volume: image.volume,
                    cube: image.cube,
                },
            ));
        }
        // A 1D image (`texture1d` / `texture1d_array`) is a single row: it may
        // combine only with `arrayed` (the 1D-array case) and always has
        // height 1. `volume`/`cube` are 2D/3D shapes and cannot co-occur.
        if image.one_dim && (image.volume || image.cube || image.height != 1) {
            return Err(DrawError::DrawValidation(
                DrawValidationDecline::SampledShapeConflict {
                    binding: image.binding,
                    arrayed: image.arrayed,
                    volume: image.volume,
                    cube: image.cube,
                },
            ));
        }
        if image.cube && (image.layers != 6 || image.width != image.height) {
            return Err(DrawError::DrawValidation(
                DrawValidationDecline::SampledCubeGeometry {
                    binding: image.binding,
                    width: image.width,
                    height: image.height,
                    layers: image.layers,
                },
            ));
        }
        if !image.arrayed && !image.volume && !image.cube && image.layers != 1 {
            return Err(DrawError::DrawValidation(
                DrawValidationDecline::SampledNonArrayLayers {
                    binding: image.binding,
                    layers: image.layers,
                },
            ));
        }
        // Footprint of one texel of the image's own format. `None` means a
        // format whose bytes are not one number per texel (block-compressed,
        // multi-planar) reached a rail that sizes a linear buffer — decline by
        // name rather than compute a wrong length.
        let Some(texel) = super::super::translate::pixel::bytes_per_texel(image.format) else {
            return Err(DrawError::DrawValidation(
                DrawValidationDecline::SampledNoLinearTexelFootprint {
                    binding: image.binding,
                    format: image.format,
                },
            ));
        };
        let texel = texel as usize;
        let expected = image.width as usize * image.height as usize * image.layers as usize * texel;
        match &image.source {
            SampledSource::Bytes(bytes) if bytes.len() != expected => {
                return Err(DrawError::DrawValidation(
                    DrawValidationDecline::SampledBytesLength {
                        binding: image.binding,
                        actual: bytes.len(),
                        expected,
                    },
                ));
            }
            SampledSource::Target(identity) => {
                if image.arrayed || image.volume || image.cube || image.layers != 1 {
                    return Err(DrawError::Unsupported(
                        super::reason::DrawReason::ResidentSampledNot2d {
                            binding: image.binding,
                        },
                    ));
                }
                if identity.width() != image.width || identity.height() != image.height {
                    return Err(DrawError::DrawValidation(
                        DrawValidationDecline::ResidentSampleGeometry {
                            binding: image.binding,
                            resident_width: identity.width(),
                            resident_height: identity.height(),
                            resource_width: image.width,
                            resource_height: image.height,
                        },
                    ));
                }
            }
            SampledSource::Bytes(_) => {}
            SampledSource::GuestRuns(src) => {
                // The zero-copy gather uploads a single array layer into a
                // single-depth image (`layer_count: 1`, `depth: 1` below), so
                // it serves any shape that is one layer deep: plain 2D, a
                // single-layer 2D array, and the 1D / single-layer 1D-array
                // color-transfer LUTs. Volume and multi-layer shapes still
                // decline by name — the gather would upload only their first
                // slice.
                if image.volume || image.cube || image.layers != 1 {
                    return Err(DrawError::Unsupported(
                        super::reason::DrawReason::GuestRunSampledNot2d {
                            binding: image.binding,
                        },
                    ));
                }
                // Padded layouts (`row_length_texels != 0`) span
                // `(height-1) * stride + tight_row` — the final row carries
                // only its texels (see `GuestRunSource`); tight layouts match
                // the full `width * height` window.
                let run_expected = if src.row_length_texels == 0 {
                    expected
                } else {
                    let stride = src.row_length_texels as usize * texel;
                    let tight_row = image.width as usize * texel;
                    if stride < tight_row {
                        return Err(DrawError::DrawValidation(
                            DrawValidationDecline::GuestSampleRowStride {
                                binding: image.binding,
                                stride,
                                tight_row,
                            },
                        ));
                    }
                    (image.height as usize - 1) * stride + tight_row
                };
                if src.total_len as usize != run_expected {
                    return Err(DrawError::DrawValidation(
                        DrawValidationDecline::GuestSampleLength {
                            binding: image.binding,
                            actual: src.total_len,
                            expected: run_expected,
                        },
                    ));
                }
                let sum: u64 = src.runs.iter().map(|r| r.len).sum();
                if sum != src.total_len || src.runs.is_empty() {
                    return Err(DrawError::DrawValidation(
                        DrawValidationDecline::GuestSampleCoverage {
                            binding: image.binding,
                            covered: sum,
                            declared: src.total_len,
                            runs: src.runs.len(),
                        },
                    ));
                }
            }
        }
        if !bindings.insert(image.binding) {
            return Err(DrawError::DrawValidation(
                DrawValidationDecline::DuplicateSampledDescriptorBinding {
                    binding: image.binding,
                },
            ));
        }
    }
    for sampler in &req.samplers {
        let lod_min = sampler.lod_min_f32();
        let lod_max = sampler.lod_max_f32();
        if !lod_min.is_finite() || !lod_max.is_finite() || lod_min > lod_max {
            return Err(DrawError::DrawValidation(
                DrawValidationDecline::InvalidSamplerLod {
                    binding: sampler.binding,
                    lod_min_bits: sampler.lod_min,
                    lod_max_bits: sampler.lod_max,
                },
            ));
        }
        if !bindings.insert(sampler.binding) {
            return Err(DrawError::DrawValidation(
                DrawValidationDecline::DuplicateSamplerDescriptorBinding {
                    binding: sampler.binding,
                },
            ));
        }
    }
    Ok(())
}

fn draw_has_no_invocations(req: &DrawRequest) -> bool {
    let element_count = req
        .indexed
        .as_ref()
        .map(|i| i.index_count)
        .unwrap_or(req.vertex_count);
    element_count == 0 || req.instance_count == Some(0)
}

pub(crate) unsafe fn execute_draw_inner(
    owner: &mut ContextOwner,
    caches: &mut ObjectCaches,
    pools: &mut ResourcePools,
    counters: &EngineCounters,
    req: &DrawRequest,
) -> Result<DrawOutput, DrawError> {
    validate_v1(req)?;
    let force_loss = owner.force_device_lost;
    if force_loss {
        owner.force_device_lost = false;
    }
    let ctx = owner.ensure(counters)?;
    pools.ensure_init(ctx, counters)?;

    // Draw batching (deferred submit): a draw that hands the CPU nothing
    // (skip_readback + resident target, no MRT) leaves its CB in recording
    // state for same-target successors; a successor whose work folds into the
    // open CB (LoadFromTarget — no CPU/GPU seed, not sampling its own target,
    // same identity/geometry/format) appends to it, skipping slot claim and
    // submit entirely. Every other draw claims a slot via begin_entry, which
    // flushes any open batch first (queue order = record order).
    let is_mrt = !req.secondary_targets.is_empty();
    let output_bgra = req.output_bgra && req.target_identity.is_some();
    // A sampled zero-copy source reads guest RAM when the CB *executes*, and
    // ack-fast means the guest may repaint that buffer as soon as the command
    // is consumed — deferred submit stretches record→execute from ~0 to a
    // whole batch, so the GPU samples half-repainted a/b window buffers
    // (large black bands under window drags, 2026-07-19 live A/B). Such draws take the immediate-submit path; buffer GuestRuns stay
    // batchable because `stage_buffer_content` snapshots them at record time.
    let has_zc_sampled = req
        .sampled_images
        .iter()
        .any(|s| matches!(s.source, SampledSource::GuestRuns(_)));
    let batch_eligible = !force_loss
        && !ctx.caps.quirks.no_deferred_draw_batching
        && !is_mrt
        && !has_zc_sampled
        && req.depth.is_none()
        && req.skip_readback
        && req.target_identity.is_some();
    let samples_own_target = req.sampled_images.iter().any(|s| {
        matches!(
            (&s.source, req.target_identity.as_ref()),
            (SampledSource::Target(t), Some(own)) if t == own
        )
    });
    let joins = batch_eligible
        && matches!(req.load_op, Some(LoadOp::LoadFromTarget))
        && req.target_rgba8.is_none()
        && req.seed_from_target.is_none()
        && !samples_own_target
        && req
            .target_identity
            .as_ref()
            .and_then(|id| pools.batch_slot(id, req.width, req.height, output_bgra))
            .is_some();
    // Claim the next ring slot — BEFORE any pool acquire, so a recycled slot
    // can never alias a still-in-flight CB. Blocks (retire) only when every
    // slot is still in flight; the wait lands in retire_wait_us. A batch
    // joiner reuses the open batch's slot instead (its CB is still recording).
    let (cb, fence) = if joins {
        let id = req
            .target_identity
            .as_ref()
            .expect("joins requires identity");
        pools
            .batch_slot(id, req.width, req.height, output_bgra)
            .expect("joins checked batch_slot")
    } else {
        pools.begin_entry(ctx, counters)?
    };

    // Build layout key from storage / sampled / sampler bindings.
    let mut layout_bindings = Vec::new();
    for b in &req.storage_buffers {
        layout_bindings.push(BindingSig {
            binding: b.binding,
            ty: vk::DescriptorType::STORAGE_BUFFER.as_raw() as u32,
            stages: (vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT).as_raw(),
        });
    }
    for b in &req.sampled_images {
        layout_bindings.push(BindingSig {
            binding: b.binding,
            ty: vk::DescriptorType::SAMPLED_IMAGE.as_raw() as u32,
            stages: (vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT).as_raw(),
        });
    }
    for b in &req.samplers {
        layout_bindings.push(BindingSig {
            binding: b.binding,
            ty: vk::DescriptorType::SAMPLER.as_raw() as u32,
            stages: (vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT).as_raw(),
        });
    }
    if req.color_input {
        layout_bindings.push(BindingSig {
            binding: super::types::COLOR_INPUT_BINDING,
            ty: vk::DescriptorType::INPUT_ATTACHMENT.as_raw() as u32,
            stages: vk::ShaderStageFlags::FRAGMENT.as_raw(),
        });
    }
    layout_bindings.sort_by_key(|b| b.binding);
    let layout_key = LayoutKey {
        bindings: layout_bindings,
    };
    // Resolve load action: explicit load_op > target_rgba8 > Clear.
    let resolved_load: LoadOp = match &req.load_op {
        Some(op) => match op {
            LoadOp::Clear(c) => LoadOp::Clear(*c),
            LoadOp::LoadFromTarget => LoadOp::LoadFromTarget,
            LoadOp::LoadSeed(b) => LoadOp::LoadSeed(b.clone()),
        },
        None => match &req.target_rgba8 {
            Some(b) => LoadOp::LoadSeed(b.clone()),
            None => LoadOp::Clear([0.0, 0.0, 0.0, 0.0]),
        },
    };
    let load_uses_gpu_content = matches!(resolved_load, LoadOp::LoadFromTarget);
    // output_bgra (computed with the batch decision above): BGRA output only
    // on the resident path (pooled targets stay RGBA); the whole
    // pass/pipeline/image chain then agrees on B8G8R8A8 so a raw image→buffer
    // copy lands guest scanout order with no CPU swizzle.
    let seed_bytes: Option<Vec<u8>> = match &resolved_load {
        LoadOp::LoadSeed(bytes) => {
            // LoadSeed is semantic RGBA8. Vulkan buffer→image copies do not
            // perform format conversion, so a BGRA resident attachment needs
            // physical B/G/R/A bytes. Otherwise partial draws preserve an
            // exact R/B-exchanged seed outside their damaged geometry.
            let mut native = bytes.clone();
            if output_bgra {
                for pixel in native.chunks_exact_mut(4) {
                    pixel.swap(0, 2);
                }
            }
            Some(native)
        }
        _ => None,
    };
    let mut pass_key = PassKey::single(
        load_uses_gpu_content || seed_bytes.is_some() || req.seed_from_target.is_some(),
        output_bgra,
    );
    for (i, sec) in req.secondary_targets.iter().enumerate() {
        if i >= MAX_SECONDARY_ATTACH {
            return Err(DrawError::Unsupported(
                super::reason::DrawReason::SecondaryAttachmentCap {
                    requested: req.secondary_targets.len(),
                    cap: MAX_SECONDARY_ATTACH,
                },
            ));
        }
        pass_key.secondary[i] = SecondaryAttachKey {
            format: sec.format,
            load: sec.load,
        };
    }
    pass_key.secondary_count = req.secondary_targets.len() as u8;
    pass_key.color_input = req.color_input;
    // Depth is opt-in per draw (only a non-trivial MTLDepthStencilState reaches
    // here). Combining it with MRT is not yet supported — the ad-hoc MRT
    // framebuffer would need the depth view appended and the pass rebuilt; no
    // known workload does both, so reject rather than silently drop depth.
    if req.depth.is_some() && is_mrt {
        return Err(DrawError::Unsupported(
            super::reason::DrawReason::DepthWithSecondaryAttachments,
        ));
    }
    if let Some(d) = &req.depth {
        pass_key.depth = Some(super::caches::DepthAttachKey {
            load: d.load,
            stencil: d.stencil.is_some(),
        });
    }
    let attr_keys: Vec<AttrKey> = req
        .vertex_attributes
        .iter()
        .map(|a| AttrKey {
            location: a.location,
            binding: a.binding,
            format: a.format,
            offset: a.offset,
            stride: a.stride,
            step_function: a.step_function,
            step_rate: a.step_rate,
        })
        .collect();

    let (vert_digest, vert_module) =
        caches.get_or_create_shader(ctx, &req.vert_spirv, counters, pools)?;
    let (frag_digest, frag_module) =
        caches.get_or_create_shader(ctx, &req.frag_spirv, counters, pools)?;
    let (dsl, pipeline_layout) = caches.get_or_create_layout(ctx, &layout_key, counters, pools)?;
    let render_pass = caches.get_or_create_pass(ctx, pass_key, counters, pools)?;
    let pipeline_key = PipelineKey {
        vert: vert_digest,
        frag: frag_digest,
        attrs: attr_keys,
        topology: req.primitive_topology,
        blend: req.blend.map(|b| b.key()),
        secondary_blend: {
            let mut per_slot = [None; MAX_SECONDARY_ATTACH];
            for (slot, target) in req
                .secondary_targets
                .iter()
                .take(MAX_SECONDARY_ATTACH)
                .enumerate()
            {
                per_slot[slot] = target.blend.map(|b| b.key());
            }
            per_slot
        },
        pass: pass_key,
        flip_y: req.flip_viewport_y,
        cull_mode: req.cull_mode,
        front_face_ccw: req.front_face_ccw,
        depth_test: req.depth.map(|d| d.test_enable).unwrap_or(false),
        depth_write: req.depth.map(|d| d.write_enable).unwrap_or(false),
        depth_compare: req
            .depth
            .map(|d| d.compare)
            .unwrap_or(super::types::SamplerCompareFunction::Always),
        stencil: req
            .depth
            .and_then(|d| d.stencil)
            .map(|s| super::caches::StencilKey {
                front: s.front,
                back: s.back,
            }),
        layout: layout_key.clone(),
    };
    let pipeline = if let Some(pipeline) = caches.get_prepared(&pipeline_key) {
        counters
            .pipeline_hits
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        pipeline
    } else {
        let pipe = caches.get_or_create_pipeline(
            ctx,
            &pipeline_key,
            vert_module,
            frag_module,
            pipeline_layout,
            render_pass,
            counters,
            pools,
        )?;
        caches.cache_prepared(pipeline_key.clone(), pipe);
        pipe
    };

    // Samplers
    let mut sampler_handles = Vec::new();
    for s in &req.samplers {
        let h = caches.get_or_create_sampler(ctx, &s.state_key(), counters, pools)?;
        sampler_handles.push((s.binding, h));
    }

    // Vertex buffers (with Constant step shift), deduplicated by content:
    // several attributes on one interleaved stream share one staging slot.
    let no_vertex_fetch = draw_has_no_invocations(req);
    let mut slots_by_content: std::collections::HashMap<(usize, u64), BufferSlot> =
        std::collections::HashMap::new();
    let mut vertex_bufs = Vec::new();
    for resource in &req.vertex_attributes {
        let needs_shift = !no_vertex_fetch
            && resource.step_function == VertexStepFunction::Constant
            && req.base_instance != 0;
        let slot = if needs_shift {
            // The shifted prefix makes the content unique to this bind; the
            // runtime keeps Constant-step binds on the CPU path.
            let BufferContent::Bytes(bytes) = &resource.content else {
                return Err(DrawError::DrawExecution(
                    DrawExecutionDecline::ConstantVertexRequiresCpuBytes {
                        location: resource.location,
                    },
                ));
            };
            let prefix = (req.base_instance as usize)
                .checked_mul(resource.stride as usize)
                .ok_or({
                    DrawError::DrawExecution(
                        DrawExecutionDecline::ConstantVertexBaseInstanceOverflow {
                            base_instance: req.base_instance,
                            stride: resource.stride,
                        },
                    )
                })?;
            let len = prefix.checked_add(bytes.len()).ok_or_else(|| {
                DrawError::DrawExecution(DrawExecutionDecline::ConstantVertexAllocationOverflow {
                    prefix,
                    bytes_len: bytes.len(),
                })
            })?;
            let mut shifted = vec![0u8; len];
            shifted[prefix..].copy_from_slice(bytes);
            let slot = pools.acquire_staging(
                ctx,
                shifted.len() as u64,
                vk::BufferUsageFlags::VERTEX_BUFFER,
                counters,
            )?;
            pools.write_staging(ctx, &slot, &shifted)?;
            slot
        } else {
            stage_buffer_content(
                ctx,
                pools,
                counters,
                &resource.content,
                vk::BufferUsageFlags::VERTEX_BUFFER,
                batch_eligible,
                &mut slots_by_content,
            )?
        };
        vertex_bufs.push((resource.binding, slot));
    }

    // Index buffer
    let mut index_slot = None;
    if let Some(indexed) = &req.indexed {
        let slot = pools.acquire_staging(
            ctx,
            indexed.indices.len() as u64,
            vk::BufferUsageFlags::INDEX_BUFFER,
            counters,
        )?;
        pools.write_staging(ctx, &slot, &indexed.indices)?;
        index_slot = Some(slot);
    }

    // Storage buffers (deduplicated by content with the vertex streams: a
    // stage-in buffer doubling as a storage bind reuses the same slot —
    // staging slots always carry the full usage superset).
    let mut storage_slots = Vec::new();
    for resource in &req.storage_buffers {
        let slot = stage_buffer_content(
            ctx,
            pools,
            counters,
            &resource.content,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            batch_eligible,
            &mut slots_by_content,
        )?;
        storage_slots.push((resource.binding, slot));
    }

    // Target seed staging (CPU import only — not LoadFromTarget).
    let seed_slot = if let Some(rgba8) = &seed_bytes {
        let slot = pools.acquire_staging(
            ctx,
            rgba8.len() as u64,
            vk::BufferUsageFlags::TRANSFER_SRC,
            counters,
        )?;
        pools.write_staging(ctx, &slot, rgba8)?;
        counters.note_seed_upload(rgba8.len() as u64);
        Some(slot)
    } else {
        None
    };

    // Prefer identity-keyed resident target when provided; else geometry pool.
    let use_registry = req.target_identity.is_some();
    // A secondary MRT attachment is bound + rendered as attachment N of an
    // ad-hoc framebuffer built here. The primary slot 0 keeps its own single-RT
    // framebuffer (consistent with single-RT draws to the same target), so the
    // primary is ensured under a single-attachment pass even in an MRT draw;
    // the MRT render pass is used only for the ad-hoc framebuffer + pipeline.
    // The resident/pooled slot keeps a color-only framebuffer + pass; the
    // depth-carrying `render_pass` is used only for the ad-hoc framebuffer +
    // pipeline (same split MRT already uses for its secondary framebuffer).
    // A framebuffer-fetch draw also splits: the slot is ensured under the
    // color-only pass (its cached framebuffer stays input-ref-free — passes
    // with and without an input reference are NOT framebuffer-compatible),
    // and the fetch-carrying `render_pass` is used only for the ad-hoc
    // framebuffer + pipeline, exactly like MRT/depth.
    let primary_pass = if is_mrt || req.depth.is_some() || req.color_input {
        caches.get_or_create_pass(
            ctx,
            PassKey::single(pass_key.load_seed, pass_key.bgra),
            counters,
            pools,
        )?
    } else {
        render_pass
    };
    // (identity, image, tracked-layout-before-this-draw) per secondary — used
    // to barrier prior sampled reads and to mark ready afterward.
    let mut mrt_secondaries: Vec<(super::types::TargetIdentity, vk::Image, vk::ImageLayout)> =
        Vec::new();
    // Transient depth attachment (image, memory, view, ad-hoc framebuffer) —
    // owned for exactly this draw, disposed deferred after submit. `None` on the
    // 2D path so nothing changes there.
    let mut transient_depth: Option<(vk::Image, vk::DeviceMemory, vk::ImageView, vk::Framebuffer)> =
        None;
    let (target_image, target_fb, target_old_layout, target_view) =
        if let Some(identity) = &req.target_identity {
            let gen = identity.generation();
            let t = pools.registry_ensure(
                ctx,
                identity.clone(),
                req.width,
                req.height,
                primary_pass,
                gen,
                output_bgra,
                req.seed_from_target.as_ref(),
                counters,
            )?;
            if load_uses_gpu_content && !t.content_ready {
                return Err(DrawError::DrawExecution(
                    DrawExecutionDecline::LoadTargetContentNotReady {
                        identity: identity.clone(),
                    },
                ));
            }
            let primary_image = t.image;
            let primary_view = t.view;
            let primary_layout = t.layout;
            let primary_slot_fb = t.framebuffer;
            if is_mrt {
                // Ensure each secondary resident and collect its view for the MRT
                // framebuffer. Recently-ensured residents sit at the back of the
                // LRU order, so a later secondary's capacity sweep (front-first)
                // cannot evict the primary or an earlier secondary in this draw.
                let mut views = vec![primary_view];
                for sec in &req.secondary_targets {
                    let old_layout = pools
                        .registry_get(&sec.identity)
                        .map(|s| s.layout)
                        .unwrap_or(vk::ImageLayout::UNDEFINED);
                    let (img, view) = pools.registry_ensure_color(
                        ctx,
                        sec.identity.clone(),
                        sec.width,
                        sec.height,
                        sec.identity.generation(),
                        sec.format,
                        counters,
                    )?;
                    views.push(view);
                    mrt_secondaries.push((sec.identity.clone(), img, old_layout));
                }
                let fb = pools.create_mrt_framebuffer(
                    ctx,
                    render_pass,
                    &views,
                    req.width,
                    req.height,
                    counters,
                )?;
                (primary_image, fb, primary_layout, primary_view)
            } else if req.depth.is_some() {
                let (dimg, dmem, dview) = pools.create_transient_depth(
                    ctx,
                    req.width,
                    req.height,
                    req.depth.and_then(|d| d.stencil).is_some(),
                    counters,
                )?;
                let fb = pools.create_mrt_framebuffer(
                    ctx,
                    render_pass,
                    &[primary_view, dview],
                    req.width,
                    req.height,
                    counters,
                )?;
                transient_depth = Some((dimg, dmem, dview, fb));
                (primary_image, fb, primary_layout, primary_view)
            } else if req.color_input {
                // Fetch pass carries an input reference → the slot's cached
                // color-only framebuffer is incompatible; build an ad-hoc one
                // against `render_pass` (disposed deferred after submit).
                let fb = pools.create_mrt_framebuffer(
                    ctx,
                    render_pass,
                    &[primary_view],
                    req.width,
                    req.height,
                    counters,
                )?;
                (primary_image, fb, primary_layout, primary_view)
            } else {
                (primary_image, primary_slot_fb, primary_layout, primary_view)
            }
        } else {
            let target_key = TargetKey {
                width: req.width,
                height: req.height,
                with_transfer_dst: seed_bytes.is_some(),
            };
            // Acquire the pooled slot under the color-only `primary_pass` (same as
            // its cached framebuffer). For a depth draw, build a fresh ad-hoc
            // framebuffer [color, depth] under the depth `render_pass`.
            let t = pools.acquire_target(ctx, target_key, primary_pass, counters)?;
            let (pool_image, pool_view, pool_fb) = (t.image, t.view, t.framebuffer);
            if req.depth.is_some() {
                let (dimg, dmem, dview) = pools.create_transient_depth(
                    ctx,
                    req.width,
                    req.height,
                    req.depth.and_then(|d| d.stencil).is_some(),
                    counters,
                )?;
                let fb = pools.create_mrt_framebuffer(
                    ctx,
                    render_pass,
                    &[pool_view, dview],
                    req.width,
                    req.height,
                    counters,
                )?;
                transient_depth = Some((dimg, dmem, dview, fb));
                (pool_image, fb, vk::ImageLayout::UNDEFINED, pool_view)
            } else if req.color_input {
                let fb = pools.create_mrt_framebuffer(
                    ctx,
                    render_pass,
                    &[pool_view],
                    req.width,
                    req.height,
                    counters,
                )?;
                (pool_image, fb, vk::ImageLayout::UNDEFINED, pool_view)
            } else {
                (pool_image, pool_fb, vk::ImageLayout::UNDEFINED, pool_view)
            }
        };
    // GPU seed source: resolved after registry_ensure (which protects it from
    // the capacity sweep) so the handle cannot be destroyed under this draw.
    // Every rejection is a distinct named error — the runtime pre-checks
    // readiness, so these only fire on a runtime/protocol bug.
    let seed_from_resolved: Option<(vk::Image, vk::ImageLayout)> =
        if let Some(seed_identity) = &req.seed_from_target {
            let slot = pools.registry_get(seed_identity).ok_or_else(|| {
                DrawError::DrawExecution(DrawExecutionDecline::SeedResidentMissing {
                    identity: seed_identity.clone(),
                })
            })?;
            if !slot.content_ready {
                return Err(DrawError::DrawExecution(
                    DrawExecutionDecline::SeedResidentNotReady {
                        identity: seed_identity.clone(),
                    },
                ));
            }
            if slot.width != req.width || slot.height != req.height {
                return Err(DrawError::DrawExecution(
                    DrawExecutionDecline::SeedGeometryMismatch {
                        identity: seed_identity.clone(),
                        resident_width: slot.width,
                        resident_height: slot.height,
                        draw_width: req.width,
                        draw_height: req.height,
                    },
                ));
            }
            if slot.bgra != output_bgra {
                return Err(DrawError::DrawExecution(
                    DrawExecutionDecline::SeedFormatMismatch {
                        identity: seed_identity.clone(),
                        resident_bgra: slot.bgra,
                        draw_bgra: output_bgra,
                    },
                ));
            }
            Some((slot.image, slot.layout))
        } else {
            None
        };

    // Resolve sampled images only after ensuring the render target so registry
    // capacity eviction cannot destroy an image already selected for this draw.
    let mut sampled = Vec::new();
    for resource in &req.sampled_images {
        match &resource.source {
            SampledSource::Bytes(bytes) => {
                if let Some(image) = pools.find_cached_sampled(
                    resource.width,
                    resource.height,
                    resource.layers,
                    resource.volume,
                    resource.cube,
                    resource.arrayed,
                    resource.one_dim,
                    resource.format,
                    resource.swizzle,
                    bytes,
                    resource.identity,
                    counters,
                ) {
                    sampled.push(PreparedSampled::Cached {
                        binding: resource.binding,
                        image,
                    });
                    continue;
                }
                let img = pools.acquire_sampled(
                    ctx,
                    resource.width,
                    resource.height,
                    resource.layers,
                    resource.volume,
                    resource.cube,
                    resource.arrayed,
                    resource.one_dim,
                    resource.format,
                    resource.swizzle,
                    counters,
                )?;
                let st = pools.acquire_staging(
                    ctx,
                    bytes.len() as u64,
                    vk::BufferUsageFlags::TRANSFER_SRC,
                    counters,
                )?;
                pools.write_staging(ctx, &st, bytes)?;
                counters.note_sampled_reupload(bytes.len() as u64);
                sampled.push(PreparedSampled::Upload {
                    binding: resource.binding,
                    image: img,
                    staging: st,
                    volume: resource.volume,
                    layers: resource.layers,
                });
            }
            SampledSource::Target(identity) => {
                let (source_image, source_view, source_layout, source_bgra, source_ready, sw, sh) =
                    pools
                        .registry_get(identity)
                        .map(|slot| {
                            (
                                slot.image,
                                slot.view,
                                slot.layout,
                                slot.bgra,
                                slot.content_ready,
                                slot.width,
                                slot.height,
                            )
                        })
                        .ok_or_else(|| {
                            DrawError::DrawExecution(DrawExecutionDecline::SampledResidentMissing {
                                binding: resource.binding,
                                identity: identity.clone(),
                            })
                        })?;
                if !source_ready {
                    return Err(DrawError::DrawExecution(
                        DrawExecutionDecline::SampledResidentNotReady {
                            binding: resource.binding,
                            identity: identity.clone(),
                        },
                    ));
                }
                if sw != resource.width || sh != resource.height {
                    return Err(DrawError::DrawExecution(
                        DrawExecutionDecline::SampledResidentGeometryMismatch {
                            binding: resource.binding,
                            identity: identity.clone(),
                            resident_width: sw,
                            resident_height: sh,
                            resource_width: resource.width,
                            resource_height: resource.height,
                        },
                    ));
                }
                if req.target_identity.as_ref() == Some(identity) {
                    let image = pools.acquire_sampled(
                        ctx,
                        resource.width,
                        resource.height,
                        resource.layers,
                        resource.volume,
                        resource.cube,
                        resource.arrayed,
                        resource.one_dim,
                        super::super::translate::pixel::resident_color(source_bgra),
                        resource.swizzle,
                        counters,
                    )?;
                    sampled.push(PreparedSampled::Snapshot {
                        binding: resource.binding,
                        identity: identity.clone(),
                        source_image,
                        source_old_layout: source_layout,
                        image,
                    });
                } else {
                    sampled.push(PreparedSampled::Resident {
                        binding: resource.binding,
                        identity: identity.clone(),
                        image: source_image,
                        view: source_view,
                        old_layout: source_layout,
                    });
                }
                counters
                    .sampled_gpu_binds
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            SampledSource::GuestRuns(src) => {
                let img = pools.acquire_sampled(
                    ctx,
                    resource.width,
                    resource.height,
                    resource.layers,
                    resource.volume,
                    resource.cube,
                    resource.arrayed,
                    resource.one_dim,
                    resource.format,
                    resource.swizzle,
                    counters,
                )?;
                let scratch = pools.acquire_staging(
                    ctx,
                    src.total_len,
                    vk::BufferUsageFlags::TRANSFER_SRC,
                    counters,
                )?;
                // The CPU gathers the texel bytes out of the guest runs into
                // the mapped scratch; the command buffer then does one
                // buffer→image copy over it.
                //
                // The command buffer used to do the gather itself, copying from
                // per-run `VK_EXT_external_memory_host` imports of the guest
                // pages. That is the mechanism this removal is about: an
                // imported host pointer is one the GPU can write, and these
                // runs are guest RAM. `TRANSFER_DST` came off the scratch usage
                // with it — nothing on the device writes this buffer any more.
                pools.write_staging_from_runs(ctx, &scratch, &src.runs, src.total_len)?;
                crate::observe::ZeroCopyLost::SampledGuestRuns.note();
                sampled.push(PreparedSampled::GuestGather {
                    binding: resource.binding,
                    image: img,
                    scratch,
                    row_length_texels: src.row_length_texels,
                });
            }
        }
    }

    let rb_size = (req.width as u64) * (req.height as u64) * 4;
    let do_readback = !req.skip_readback;
    let readback = if do_readback {
        Some(pools.acquire_readback(ctx, rb_size, counters)?)
    } else {
        None
    };
    let _ = use_registry;

    // Descriptor set
    // Owning pool block travels alongside the set so the flush-time free routes
    // back to the block it was allocated from (arena may grow past block 0).
    let mut dset_pool: Option<vk::DescriptorPool> = None;
    let dset = if dsl != vk::DescriptorSetLayout::null() {
        let (dset, pool) = pools.alloc_descriptor_set(&ctx.device, dsl, counters)?;
        dset_pool = Some(pool);
        let buffer_infos: Vec<_> = storage_slots
            .iter()
            .map(|(_, s)| {
                vk::DescriptorBufferInfo::default()
                    .buffer(s.buffer)
                    .offset(0)
                    .range(vk::WHOLE_SIZE)
            })
            .collect();
        let sampled_infos: Vec<_> = sampled
            .iter()
            .map(|image| {
                vk::DescriptorImageInfo::default()
                    .image_view(image.view())
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            })
            .collect();
        let sampler_infos: Vec<_> = sampler_handles
            .iter()
            .map(|(_, s)| vk::DescriptorImageInfo::default().sampler(*s))
            .collect();
        // Framebuffer fetch: the input attachment IS the color target's view;
        // GENERAL matches the subpass references (see `get_or_create_pass`).
        let color_input_info = vk::DescriptorImageInfo::default()
            .image_view(target_view)
            .image_layout(vk::ImageLayout::GENERAL);
        let mut writes = Vec::new();
        for (i, (binding, _)) in storage_slots.iter().enumerate() {
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(dset)
                    .dst_binding(*binding)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(&buffer_infos[i])),
            );
        }
        for (i, image) in sampled.iter().enumerate() {
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(dset)
                    .dst_binding(image.binding())
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .image_info(std::slice::from_ref(&sampled_infos[i])),
            );
        }
        for (i, (binding, _)) in sampler_handles.iter().enumerate() {
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(dset)
                    .dst_binding(*binding)
                    .descriptor_type(vk::DescriptorType::SAMPLER)
                    .image_info(std::slice::from_ref(&sampler_infos[i])),
            );
        }
        if req.color_input {
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(dset)
                    .dst_binding(super::types::COLOR_INPUT_BINDING)
                    .descriptor_type(vk::DescriptorType::INPUT_ATTACHMENT)
                    .image_info(std::slice::from_ref(&color_input_info)),
            );
        }
        ctx.device.update_descriptor_sets(&writes, &[]);
        Some(dset)
    } else {
        None
    };

    // The ring slot's CB retired at begin_entry and its fence is unsignaled —
    // no pre-record wait remains (pre_record_wait_us stays 0 on this path).
    // A batch joiner's CB is already recording (opened by the batch opener);
    // its commands append after the previous draw's end_render_pass.
    if !joins {
        ctx.device
            .reset_command_buffer(cb, vk::CommandBufferResetFlags::empty())
            .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::ExecResetCb, e)))?;
        ctx.device
            .begin_command_buffer(
                cb,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::ExecBeginCb, e)))?;
    }

    // Metal permits a pass to sample the same texture it renders into. Vulkan
    // does not permit that attachment feedback loop on this path, so capture
    // the prior resident content into a same-format GPU image before changing
    // the attachment. This preserves the old CPU snapshot semantics without a
    // readback or host upload.
    let mut snapshotted_targets = std::collections::HashSet::new();
    let mut target_snapshotted = false;
    for sampled_image in &sampled {
        let PreparedSampled::Snapshot {
            identity,
            source_image,
            source_old_layout,
            image,
            ..
        } = sampled_image
        else {
            continue;
        };
        target_snapshotted = true;
        if snapshotted_targets.insert(identity.clone())
            && *source_old_layout != vk::ImageLayout::TRANSFER_SRC_OPTIMAL
        {
            let (src_stage, src_access) = layout_source_scope(*source_old_layout)?;
            let barrier = [vk::ImageMemoryBarrier::default()
                .src_access_mask(src_access)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .old_layout(*source_old_layout)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .image(*source_image)
                .subresource_range(color_range())];
            ctx.device.cmd_pipeline_barrier(
                cb,
                src_stage,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &barrier,
            );
        }
        let barrier = [vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .image(image.image)
            .subresource_range(color_range())];
        ctx.device.cmd_pipeline_barrier(
            cb,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &barrier,
        );
        let copy = [vk::ImageCopy::default()
            .src_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .dst_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .extent(vk::Extent3D {
                width: image.width,
                height: image.height,
                depth: 1,
            })];
        ctx.device.cmd_copy_image(
            cb,
            *source_image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            image.image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &copy,
        );
        let barrier = [vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image(image.image)
            .subresource_range(color_range())];
        ctx.device.cmd_pipeline_barrier(
            cb,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::VERTEX_SHADER | vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &barrier,
        );
    }

    // Seed upload (CPU import).
    if let Some(seed) = &seed_slot {
        let (src_stage, src_access, old_layout) = if target_snapshotted {
            (
                vk::PipelineStageFlags::TRANSFER,
                vk::AccessFlags::TRANSFER_READ,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            )
        } else {
            (
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::AccessFlags::empty(),
                vk::ImageLayout::UNDEFINED,
            )
        };
        let barrier = [vk::ImageMemoryBarrier::default()
            .src_access_mask(src_access)
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .old_layout(old_layout)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .image(target_image)
            .subresource_range(color_range())];
        ctx.device.cmd_pipeline_barrier(
            cb,
            src_stage,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &barrier,
        );
        let copy = [vk::BufferImageCopy::default()
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .image_extent(vk::Extent3D {
                width: req.width,
                height: req.height,
                depth: 1,
            })];
        ctx.device.cmd_copy_buffer_to_image(
            cb,
            seed.buffer,
            target_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &copy,
        );
        let barrier = [vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(
                vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            )
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .image(target_image)
            .subresource_range(color_range())];
        ctx.device.cmd_pipeline_barrier(
            cb,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &barrier,
        );
    } else if let Some((seed_image, seed_layout)) = seed_from_resolved {
        // GPU present-boundary seed: resident front frame → draw target copy,
        // then the pass runs with LOAD. Same target-side transitions as the
        // CPU seed branch; the source transitions from its tracked layout.
        if seed_layout != vk::ImageLayout::TRANSFER_SRC_OPTIMAL {
            let (src_stage, src_access) = layout_source_scope(seed_layout)?;
            let barrier = [vk::ImageMemoryBarrier::default()
                .src_access_mask(src_access)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .old_layout(seed_layout)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .image(seed_image)
                .subresource_range(color_range())];
            ctx.device.cmd_pipeline_barrier(
                cb,
                src_stage,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &barrier,
            );
        }
        let (dst_stage, dst_access, dst_old_layout) = if target_snapshotted {
            (
                vk::PipelineStageFlags::TRANSFER,
                vk::AccessFlags::TRANSFER_READ,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            )
        } else {
            (
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::AccessFlags::empty(),
                vk::ImageLayout::UNDEFINED,
            )
        };
        let barrier = [vk::ImageMemoryBarrier::default()
            .src_access_mask(dst_access)
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .old_layout(dst_old_layout)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .image(target_image)
            .subresource_range(color_range())];
        ctx.device.cmd_pipeline_barrier(
            cb,
            dst_stage,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &barrier,
        );
        let region = [vk::ImageCopy::default()
            .src_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .dst_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .extent(vk::Extent3D {
                width: req.width,
                height: req.height,
                depth: 1,
            })];
        ctx.device.cmd_copy_image(
            cb,
            seed_image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            target_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &region,
        );
        let barrier = [vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(
                vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            )
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .image(target_image)
            .subresource_range(color_range())];
        ctx.device.cmd_pipeline_barrier(
            cb,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &barrier,
        );
        counters
            .seed_gpu_copies
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        counters.seed_gpu_copy_bytes.fetch_add(
            (req.width as u64) * (req.height as u64) * 4,
            std::sync::atomic::Ordering::Relaxed,
        );
    } else if load_uses_gpu_content {
        // A prior direct sample may have left this target shader-readable;
        // transition from the registry's tracked layout back to attachment use.
        let old_layout = if target_snapshotted {
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL
        } else {
            target_old_layout
        };
        let (src_stage, src_access) = layout_source_scope(old_layout)?;
        let barrier = [vk::ImageMemoryBarrier::default()
            .src_access_mask(src_access)
            .dst_access_mask(
                vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            )
            .old_layout(old_layout)
            .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .image(target_image)
            .subresource_range(color_range())];
        ctx.device.cmd_pipeline_barrier(
            cb,
            src_stage,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &barrier,
        );
    } else if target_snapshotted {
        // The Clear render pass discards prior content via initialLayout
        // UNDEFINED, but its write must still wait for the snapshot read.
        let barrier = [vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_READ)
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)];
        ctx.device.cmd_pipeline_barrier(
            cb,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::DependencyFlags::empty(),
            &barrier,
            &[],
            &[],
        );
    }

    // Resident samples: transition the persistent target in place. Duplicate
    // bindings of one target share the same image and therefore one barrier.
    let mut transitioned_resident = std::collections::HashSet::new();
    for image in &sampled {
        let PreparedSampled::Resident {
            identity,
            image,
            old_layout,
            ..
        } = image
        else {
            continue;
        };
        if !transitioned_resident.insert(identity.clone())
            || *old_layout == vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        {
            continue;
        }
        let (src_stage, src_access) = layout_source_scope(*old_layout)?;
        let barrier = [vk::ImageMemoryBarrier::default()
            .src_access_mask(src_access)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .old_layout(*old_layout)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image(*image)
            .subresource_range(color_range())];
        ctx.device.cmd_pipeline_barrier(
            cb,
            src_stage,
            vk::PipelineStageFlags::VERTEX_SHADER | vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &barrier,
        );
    }

    // CPU-origin sampled uploads.
    for image in &sampled {
        let PreparedSampled::Upload {
            image: img,
            staging: st,
            volume,
            layers,
            ..
        } = image
        else {
            continue;
        };
        let array_layers = if *volume { 1 } else { *layers };
        let extent_depth = if *volume { *layers } else { 1 };
        let barrier = [vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .image(img.image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: array_layers,
            })];
        ctx.device.cmd_pipeline_barrier(
            cb,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &barrier,
        );
        let copy = [vk::BufferImageCopy::default()
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: array_layers,
            })
            .image_extent(vk::Extent3D {
                width: img.width,
                height: img.height,
                depth: extent_depth,
            })];
        ctx.device.cmd_copy_buffer_to_image(
            cb,
            st.buffer,
            img.image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &copy,
        );
        let barrier = [vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image(img.image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: array_layers,
            })];
        ctx.device.cmd_pipeline_barrier(
            cb,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::VERTEX_SHADER | vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &barrier,
        );
    }

    // Guest gathers: the scratch was packed by `write_staging_from_runs` during
    // preparation, so this is the same host-staged upload the CPU-origin loop
    // above performs, differing only in `row_length_texels` striding over guest
    // row padding (0 = tight rows).
    //
    // No HOST→TRANSFER barrier, matching that loop: writes the host made before
    // `vkQueueSubmit` are automatically visible to the device, and this scratch
    // is written before the submit like every other staging slot. The two
    // barriers that used to open this block ordered the *device-side* gather —
    // per-run copies out of imported guest pages — against the image copy, and
    // there is no device-side write to order any more.
    for image in &sampled {
        let PreparedSampled::GuestGather {
            image: img,
            scratch,
            row_length_texels,
            ..
        } = image
        else {
            continue;
        };
        let image_barrier = [vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .image(img.image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })];
        ctx.device.cmd_pipeline_barrier(
            cb,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &image_barrier,
        );
        let copy = [vk::BufferImageCopy::default()
            .buffer_row_length(*row_length_texels)
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .image_extent(vk::Extent3D {
                width: img.width,
                height: img.height,
                depth: 1,
            })];
        ctx.device.cmd_copy_buffer_to_image(
            cb,
            scratch.buffer,
            img.image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &copy,
        );
        let to_shader = [vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image(img.image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })];
        ctx.device.cmd_pipeline_barrier(
            cb,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::VERTEX_SHADER | vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &to_shader,
        );
    }

    // MRT secondary attachments that were left shader-readable (sampled by a
    // prior draw) must transition back to color-attachment use, and the write
    // must wait for that prior read (WAR). A freshly-created secondary tracks
    // UNDEFINED and needs no barrier — the render pass discards on CLEAR.
    for (_id, image, old_layout) in &mrt_secondaries {
        if *old_layout == vk::ImageLayout::UNDEFINED {
            continue;
        }
        let (src_stage, src_access) = layout_source_scope(*old_layout)?;
        let barrier = [vk::ImageMemoryBarrier::default()
            .src_access_mask(src_access)
            .dst_access_mask(
                vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            )
            .old_layout(*old_layout)
            .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .image(*image)
            .subresource_range(color_range())];
        ctx.device.cmd_pipeline_barrier(
            cb,
            src_stage,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &barrier,
        );
    }

    // One clear value per attachment (slot 0 + secondaries), in FB order. Only
    // CLEAR attachments consult these, but the count must cover all attachments.
    let mut clear = vec![vk::ClearValue {
        color: vk::ClearColorValue {
            float32: [0.0, 0.0, 0.0, 0.0],
        },
    }];
    for sec in &req.secondary_targets {
        clear.push(vk::ClearValue {
            color: vk::ClearColorValue { float32: sec.clear },
        });
    }
    // Depth attachment is last (after color + secondaries), matching the pass
    // attachment order. Only consulted when its load_op is CLEAR.
    if let Some(d) = &req.depth {
        clear.push(vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue {
                depth: d.clear_value,
                stencil: d.stencil.map(|s| s.clear_value).unwrap_or(0),
            },
        });
    }
    let rp_begin = vk::RenderPassBeginInfo::default()
        .render_pass(render_pass)
        .framebuffer(target_fb)
        .render_area(vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D {
                width: req.width,
                height: req.height,
            },
        })
        .clear_values(&clear);
    ctx.device
        .cmd_begin_render_pass(cb, &rp_begin, vk::SubpassContents::INLINE);
    ctx.device
        .cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, pipeline);

    // Dynamic viewport/scissor (Metal Y flip via negative height).
    let default_vp = ViewportResource {
        x: 0.0,
        y: 0.0,
        width: req.width as f32,
        height: req.height as f32,
        min_depth: 0.0,
        max_depth: 1.0,
    };
    let vp_src = req.viewports.first().copied().unwrap_or(default_vp);
    let viewport = if req.flip_viewport_y {
        vk::Viewport {
            x: vp_src.x,
            y: vp_src.y + vp_src.height,
            width: vp_src.width,
            height: -vp_src.height,
            min_depth: vp_src.min_depth,
            max_depth: vp_src.max_depth,
        }
    } else {
        vk::Viewport {
            x: vp_src.x,
            y: vp_src.y,
            width: vp_src.width,
            height: vp_src.height,
            min_depth: vp_src.min_depth,
            max_depth: vp_src.max_depth,
        }
    };
    ctx.device.cmd_set_viewport(cb, 0, &[viewport]);
    let default_sc = ScissorResource {
        x: 0,
        y: 0,
        width: req.width,
        height: req.height,
    };
    let sc_src = req.scissors.first().copied().unwrap_or(default_sc);
    let x = sc_src.x.min(req.width);
    let y = sc_src.y.min(req.height);
    let scissor = vk::Rect2D {
        offset: vk::Offset2D {
            x: x as i32,
            y: y as i32,
        },
        extent: vk::Extent2D {
            width: sc_src.width.min(req.width - x),
            height: sc_src.height.min(req.height - y),
        },
    };
    ctx.device.cmd_set_scissor(cb, 0, &[scissor]);
    // Dynamic stencil reference (Metal `setStencilFrontReferenceValue:back…`)
    // — only bound for stencil pipelines, which list STENCIL_REFERENCE as a
    // dynamic state; front/back set separately to honor Metal's split refs.
    if let Some(s) = req.depth.and_then(|d| d.stencil) {
        ctx.device
            .cmd_set_stencil_reference(cb, vk::StencilFaceFlags::FRONT, s.reference_front);
        ctx.device
            .cmd_set_stencil_reference(cb, vk::StencilFaceFlags::BACK, s.reference_back);
    }

    if let Some(dset) = dset {
        ctx.device.cmd_bind_descriptor_sets(
            cb,
            vk::PipelineBindPoint::GRAPHICS,
            pipeline_layout,
            0,
            &[dset],
            &[],
        );
    }
    for (binding, slot) in &vertex_bufs {
        ctx.device
            .cmd_bind_vertex_buffers(cb, *binding, &[slot.buffer], &[0]);
    }
    match (&req.indexed, &index_slot) {
        (Some(indexed), Some(ibuf)) => {
            ctx.device
                .cmd_bind_index_buffer(cb, ibuf.buffer, 0, indexed.index_type.vk());
            ctx.device.cmd_draw_indexed(
                cb,
                indexed.index_count,
                req.instance_count.unwrap_or(1),
                0,
                indexed.vertex_offset,
                req.base_instance,
            );
        }
        _ => {
            ctx.device.cmd_draw(
                cb,
                req.vertex_count,
                req.instance_count.unwrap_or(1),
                req.first_vertex,
                req.base_instance,
            );
        }
    }
    ctx.device.cmd_end_render_pass(cb);

    // Color already TRANSFER_SRC_OPTIMAL from pass final layout.
    if let Some(ref rb) = readback {
        let region = [vk::BufferImageCopy::default()
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .image_extent(vk::Extent3D {
                width: req.width,
                height: req.height,
                depth: 1,
            })];
        ctx.device.cmd_copy_image_to_buffer(
            cb,
            target_image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            rb.buffer,
            &region,
        );
    }
    // A batch-eligible draw defers end_command_buffer + submit: its CB stays
    // in recording state for same-target successors and is submitted by
    // pools.batch_flush (next begin_entry / retire / explicit flush).
    let defer_submit = batch_eligible;
    if !defer_submit {
        ctx.device
            .end_command_buffer(cb)
            .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::ExecEndCb, e)))?;
    }

    if force_loss {
        // Recycle transient resources before reporting loss.
        if let (Some(ds), Some(pool)) = (dset, dset_pool) {
            pools.free_descriptor_sets(&ctx.device, &[(ds, pool)]);
        }
        pools.recycle_staging();
        pools.recycle_readback();
        pools.recycle_sampled();
        return Err(DrawError::DeviceLost(DeviceLostDecline::ForcedDraw));
    }

    if !defer_submit {
        let queue = ctx.queue();
        let cbs = [cb];
        let si = vk::SubmitInfo::default().command_buffers(&cbs);
        match ctx.device.queue_submit(queue, &[si], fence) {
            Ok(()) => {}
            Err(e) if e == vk::Result::ERROR_DEVICE_LOST => {
                return Err(DrawError::DeviceLost(DeviceLostDecline::Driver {
                    op: DeviceLostOp::DrawSubmit,
                    result: e,
                }));
            }
            Err(e) => return Err(DrawError::VkCall(VkCall::new(VkOp::ExecSubmit, e))),
        }
    }
    // CPU-side bookkeeping: the retained target's content is queue-ordered
    // (mark ready), resident sampled layouts advance to the recorded
    // post-draw layout, and upload-path sampled bytes queue for cache
    // admission at retire time.
    if let Some(identity) = &req.target_identity {
        pools.registry_mark_ready(identity);
    }
    // MRT secondary attachments settle at COLOR_ATTACHMENT_OPTIMAL (the pass
    // final layout) and become sampleable residents; the consumer's
    // resident-sample barrier then transitions COLOR_ATTACHMENT→SHADER_READ,
    // carrying the color-write→shader-read dependency. (The ad-hoc MRT
    // framebuffer is disposed below, after `finish_entry_async` — see there.)
    if is_mrt {
        for (identity, _image, _old) in &mrt_secondaries {
            pools.registry_mark_ready_at(identity, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        }
    }
    let mut sampled_retains: Vec<(
        vk::Image,
        std::sync::Arc<Vec<u8>>,
        Option<super::types::SampledContentIdentity>,
    )> = Vec::new();
    for image in &sampled {
        let PreparedSampled::Upload { binding, image, .. } = image else {
            continue;
        };
        if let Some((SampledSource::Bytes(bytes), identity)) = req
            .sampled_images
            .iter()
            .find(|resource| resource.binding == *binding)
            .map(|resource| (&resource.source, resource.identity))
        {
            sampled_retains.push((image.image, bytes.clone(), identity));
        }
    }
    for image in &sampled {
        if let PreparedSampled::Resident { identity, .. } = image {
            pools.registry_set_layout(identity, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        }
    }
    if seed_from_resolved.is_some() {
        if let Some(seed_identity) = &req.seed_from_target {
            pools.registry_set_layout(seed_identity, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
        }
    }
    // Deferred-submit draw: park the per-draw descriptor set and sampled
    // admissions on the open batch (opening it if this is the first) and
    // return. The CPU-side bookkeeping above already ran — content_ready and
    // tracked layouts describe what the recorded CB produces, and every
    // consumer path flushes the batch before touching the GPU.
    if defer_submit {
        let identity = req
            .target_identity
            .clone()
            .expect("batch_eligible requires target identity");
        pools.batch_append(
            cb,
            fence,
            identity,
            req.width,
            req.height,
            output_bgra,
            dset.zip(dset_pool),
            sampled_retains,
            counters,
        );
        counters
            .render_post_wait_skips
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        return Ok(DrawOutput { pixels: Vec::new() });
    }

    // Park the owed cleanup (descriptor set, transient pool slots, cache
    // admissions) on this ring slot in every mode; whichever entry retires
    // the slot drains it. A failed wait below leaves the slot pending, so no
    // path ever reuses an unretired fence.
    let cleanup = pools.seal_entry(dset.zip(dset_pool).into_iter().collect(), sampled_retains);
    pools.finish_entry_async(cleanup);

    // Dispose the ad-hoc per-draw framebuffers (MRT and/or depth) now that
    // `finish_entry_async` has marked the slot in-flight (`in_flight > 0`): the
    // handles route to the graveyard and are freed only once every in-flight
    // fence — including this draw's — has retired. Disposing BEFORE this point
    // would immediate-free them (gpu_work_open() == false, since `in_flight` is
    // only bumped here, after submit) while the just-submitted CB still
    // references them → GPU fault.
    if is_mrt || (req.color_input && transient_depth.is_none()) {
        pools.dispose(
            &ctx.device,
            super::pools::DeferredHandle::Framebuffer(target_fb),
        );
    }
    if let Some((dimg, dmem, dview, dfb)) = transient_depth {
        pools.dispose(&ctx.device, super::pools::DeferredHandle::Framebuffer(dfb));
        pools.dispose(
            &ctx.device,
            super::pools::DeferredHandle::Image {
                image: dimg,
                view: dview,
                memory: dmem,
            },
        );
    }

    // A draw with no pixel readback (resident target, skip_readback) hands
    // the CPU nothing — skip the post-submit fence wait and return while the
    // GPU still runs on this ring slot.
    let Some(ref rb) = readback else {
        counters
            .render_post_wait_skips
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        return Ok(DrawOutput { pixels: Vec::new() });
    };

    // Wait ONLY this draw's fence, not the whole ring. The readback copy is the
    // tail of this CB, and single-queue submission order already guarantees it
    // observes every prior-submitted draw's writes (the same argument
    // `read_target_inner` relies on) — so `retire_all` here would just serialize
    // the guest-blocking readback behind an unrelated in-flight heavy draw (the
    // `finish_us` tail). The cleanup is already parked with `finish_entry_async`
    // above, so the slot stays pending and the ring retires it later with no
    // extra wait (its fence is already signaled).
    pools.wait_entry_fence(ctx, counters, fence)?;

    let out = {
        let ptr = ctx
            .device
            .map_memory(rb.memory, 0, rb_size, vk::MemoryMapFlags::empty())
            .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::ExecMapReadback, e)))?
            as *const u8;
        let mut pixels = vec![0u8; rb_size as usize];
        std::ptr::copy_nonoverlapping(ptr, pixels.as_mut_ptr(), rb_size as usize);
        ctx.device.unmap_memory(rb.memory);
        counters.note_readback(rb_size);
        pixels
    };

    Ok(DrawOutput { pixels: out })
}

fn color_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    }
}

fn layout_source_scope(
    layout: vk::ImageLayout,
) -> Result<(vk::PipelineStageFlags, vk::AccessFlags), DrawError> {
    match layout {
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL => Ok((
            vk::PipelineStageFlags::TRANSFER,
            vk::AccessFlags::TRANSFER_READ,
        )),
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL => Ok((
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
        )),
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL => Ok((
            vk::PipelineStageFlags::VERTEX_SHADER | vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::AccessFlags::SHADER_READ,
        )),
        other => Err(DrawError::DrawExecution(
            DrawExecutionDecline::UnsupportedTrackedLayout { layout: other },
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::vulkan::engine::types::{
        GuestRun, GuestRunSource, SampledImageResource, SampledSource,
    };
    use crate::observe::Decline;

    fn validation_slug(req: &DrawRequest) -> &'static str {
        match validate_v1(req) {
            Err(DrawError::DrawValidation(decline)) => decline.slug(),
            Err(other) => panic!("expected typed draw validation, got {other}"),
            Ok(()) => panic!("expected draw validation failure"),
        }
    }

    fn guest_run_req(w: u32, h: u32, total_len: u64, row_length_texels: u32) -> DrawRequest {
        DrawRequest {
            width: w,
            height: h,
            vert_spirv: std::sync::Arc::new(vec![0]),
            frag_spirv: std::sync::Arc::new(vec![0]),
            sampled_images: vec![SampledImageResource {
                binding: 32,
                width: w,
                height: h,
                layers: 1,
                arrayed: false,
                volume: false,
                cube: false,
                one_dim: false,
                source: SampledSource::GuestRuns(GuestRunSource {
                    runs: std::sync::Arc::new(vec![GuestRun {
                        host_ptr: 0x1000,
                        len: total_len,
                    }]),
                    total_len,
                    row_length_texels,
                }),
                format: crate::backend::vulkan::translate::pixel::vk_texel_layout(
                    crate::contract::pixel_format::TexelLayout::Bgra8,
                ),
                identity: None,
                swizzle: Default::default(),
            }],
            ..DrawRequest::default()
        }
    }

    #[test]
    fn unsupported_source_layout_returns_the_typed_execution_reason() {
        let decline = match layout_source_scope(vk::ImageLayout::UNDEFINED) {
            Err(DrawError::DrawExecution(decline)) => decline,
            Err(other) => panic!("expected typed draw execution decline, got {other}"),
            Ok(_) => panic!("expected unsupported tracked layout"),
        };
        assert_eq!(decline.slug(), "vk_draw_exec_unsupported_tracked_layout");
        assert_eq!(decline.fields(), vec![("layout", "UNDEFINED".into())]);
    }

    #[test]
    fn guest_runs_tight_total_validates() {
        let req = guest_run_req(1240, 622, 1240 * 622 * 4, 0);
        assert!(validate_v1(&req).is_ok());
    }

    /// The Safari content-layer case: width 1240, guest stride 1280 texels.
    /// The window spans (h-1)*stride + tight last row — NOT w*h*bpp; the
    /// tight comparison rejected every padded-stride zero-copy bind and the
    /// dropped draw left the app window content permanently blank.
    #[test]
    fn guest_runs_padded_stride_validates() {
        let padded = 621 * 1280 * 4 + 1240 * 4; // 3_184_480
        let req = guest_run_req(1240, 622, padded as u64, 1280);
        assert!(validate_v1(&req).is_ok());
    }

    #[test]
    fn guest_runs_padded_stride_rejects_tight_total() {
        let req = guest_run_req(1240, 622, 1240 * 622 * 4, 1280);
        assert_eq!(
            validation_slug(&req),
            "vk_draw_validate_guest_sample_length"
        );
    }

    #[test]
    fn guest_runs_rejects_stride_under_width() {
        let total = 621 * 1024 * 4 + 1240 * 4;
        let req = guest_run_req(1240, 622, total as u64, 1024);
        assert_eq!(
            validation_slug(&req),
            "vk_draw_validate_guest_sample_row_stride"
        );
    }

    fn buffer_guest_runs(
        run_lens: &[u64],
        total_len: u64,
        row_length_texels: u32,
    ) -> BufferContent {
        BufferContent::GuestRuns(super::super::types::GuestRunSource {
            runs: std::sync::Arc::new(
                run_lens
                    .iter()
                    .map(|&len| GuestRun {
                        host_ptr: 0x1000,
                        len,
                    })
                    .collect(),
            ),
            total_len,
            row_length_texels,
        })
    }

    fn storage_buffer_req(content: BufferContent) -> DrawRequest {
        DrawRequest {
            width: 8,
            height: 8,
            vert_spirv: std::sync::Arc::new(vec![0]),
            frag_spirv: std::sync::Arc::new(vec![0]),
            storage_buffers: vec![super::super::types::StorageBufferResource {
                binding: 0,
                content,
            }],
            ..DrawRequest::default()
        }
    }

    #[test]
    fn buffer_guest_runs_consistent_span_validates() {
        let req = storage_buffer_req(buffer_guest_runs(&[0x3000, 0x1000], 0x4000, 0));
        assert!(validate_v1(&req).is_ok());
    }

    #[test]
    fn buffer_guest_runs_rejects_span_mismatch() {
        let req = storage_buffer_req(buffer_guest_runs(&[0x3000], 0x4000, 0));
        assert_eq!(
            validation_slug(&req),
            "vk_draw_validate_storage_guest_runs_coverage"
        );
    }

    #[test]
    fn buffer_guest_runs_rejects_row_stride() {
        let req = storage_buffer_req(buffer_guest_runs(&[0x4000], 0x4000, 64));
        assert_eq!(
            validation_slug(&req),
            "vk_draw_validate_storage_guest_runs_row_stride"
        );
    }

    #[test]
    fn buffer_guest_runs_rejects_empty_span() {
        let req = storage_buffer_req(buffer_guest_runs(&[], 0, 0));
        assert_eq!(
            validation_slug(&req),
            "vk_draw_validate_storage_guest_runs_coverage"
        );
    }

    /// A Constant-step attribute with a nonzero base instance needs the CPU
    /// prefix shift; a gathered guest span must be rejected at validate time
    /// (the runtime gate keeps those streams on the CPU path).
    #[test]
    fn buffer_guest_runs_rejects_constant_step_shift() {
        let content = buffer_guest_runs(&[48 * 4], 48 * 4, 0);
        let mut req = DrawRequest {
            width: 8,
            height: 8,
            vert_spirv: std::sync::Arc::new(vec![0]),
            frag_spirv: std::sync::Arc::new(vec![0]),
            vertex_count: 3,
            base_instance: 2,
            ..DrawRequest::default()
        };
        req.vertex_attributes
            .push(super::super::types::VertexAttributeResource {
                location: 0,
                binding: 0,
                format: super::super::types::VertexAttributeFormat::Float4,
                offset: 0,
                stride: 48,
                step_function: VertexStepFunction::Constant,
                step_rate: 1,
                content: content.clone(),
            });
        assert_eq!(
            validation_slug(&req),
            "vk_draw_validate_constant_step_guest_runs"
        );
        // Same request with CPU bytes passes.
        req.vertex_attributes[0].content = vec![0u8; 48 * 4].into();
        assert!(validate_v1(&req).is_ok());
    }

    #[test]
    fn empty_vertex_and_fragment_spirv_have_distinct_reasons() {
        let mut req = DrawRequest {
            width: 8,
            height: 8,
            vert_spirv: std::sync::Arc::new(Vec::new()),
            frag_spirv: std::sync::Arc::new(vec![0]),
            ..DrawRequest::default()
        };
        assert_eq!(validation_slug(&req), "vk_draw_validate_empty_vertex_spirv");

        req.vert_spirv = std::sync::Arc::new(vec![0]);
        req.frag_spirv = std::sync::Arc::new(Vec::new());
        assert_eq!(
            validation_slug(&req),
            "vk_draw_validate_empty_fragment_spirv"
        );
    }

    /// `cpu_bytes` materializes a fragmented gather exactly (diagnostic /
    /// coverage-proof view of a zero-copy bind).
    #[test]
    fn buffer_content_cpu_bytes_materializes_runs() {
        let backing: Vec<u8> = (0u8..=255).collect();
        let runs = vec![
            GuestRun {
                host_ptr: backing.as_ptr() as usize,
                len: 100,
            },
            GuestRun {
                host_ptr: backing.as_ptr() as usize + 200,
                len: 56,
            },
        ];
        let content = BufferContent::GuestRuns(super::super::types::GuestRunSource {
            runs: std::sync::Arc::new(runs),
            total_len: 156,
            row_length_texels: 0,
        });
        assert_eq!(content.len(), 156);
        let bytes = content.cpu_bytes();
        assert_eq!(&bytes[..100], &backing[..100]);
        assert_eq!(&bytes[100..156], &backing[200..256]);
    }
}
