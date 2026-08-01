//! Record / submit (bounded fence) / readback for one compute dispatch.

#![allow(unsafe_op_in_unsafe_fn)]

use ash::vk;
use std::collections::BTreeSet;

use super::caches::{BindingSig, ComputePipelineKey, LayoutKey, ObjectCaches};
use super::compute_execution::ComputeExecutionDecline;
use super::compute_validation::ComputeValidationDecline;
use super::context::ContextOwner;
use super::counters::EngineCounters;
use super::device_lost::{DeviceLostDecline, DeviceLostOp};
use super::pools::{BufferSlot, ResourcePools, StorageImageKey, StorageImageSlot};
use super::types::{
    ComputeBufferOutput, ComputeOutput, ComputeRequest, ComputeResidentSampleBind,
    ComputeSampledImageResource, ComputeStorageResidency, DrawError,
};
use super::vk_call::{VkCall, VkOp};

struct PreparedStorageImage {
    binding: u32,
    slot: StorageImageSlot,
    seed: Option<BufferSlot>,
    dst: ComputeImageDst,
    len: usize,
    width: u32,
    height: u32,
    initial_layout: vk::ImageLayout,
    residency: Option<ComputeStorageResidency>,
}

/// One prepared sampled input: a transient sampled-only image seeded either
/// from a host staging upload or from a device-local resident copy.
struct PreparedSampledImage {
    binding: u32,
    img: StorageImageSlot,
    upload: Option<BufferSlot>,
    /// Copy-on-sample source `(resident image, its current layout)`.
    resident_src: Option<(vk::Image, vk::ImageLayout)>,
    /// Byte-reinterpret hop for a resident whose vk format differs from the
    /// sampled view but whose rows are byte-identical (equal row bytes and
    /// height): the seed goes image→buffer→image through this pooled buffer,
    /// since `vkCmdCopyImage` cannot cross texel-block sizes.
    reinterpret: Option<ReinterpretHop>,
    width: u32,
    height: u32,
}

struct ReinterpretHop {
    buffer: vk::Buffer,
    src_width: u32,
    src_height: u32,
}

/// Post-dispatch copy destination for one storage image.
enum ComputeImageDst {
    /// Pooled host-visible buffer; the CPU reads it back and the runtime
    /// writes guest pages itself.
    ///
    /// This is now the only non-deferred destination. A third variant,
    /// `Direct`, bound a transfer-dst buffer over an imported view of the
    /// caller's guest window so the dispatch's own copy landed there and no
    /// bytes crossed device→host. It is gone with the import: a buffer the GPU
    /// can write, backed by guest pages, is the exposure this removal is about.
    Readback(BufferSlot),
    /// No post-dispatch copy at all: the pinned resident storage image stays
    /// the authoritative content until the caller flushes it to guest pages
    /// (`read_resident_storage`). Requires a residency identity.
    Deferred,
}

pub(crate) fn validate_compute(req: &ComputeRequest) -> Result<(), DrawError> {
    if req.spirv.is_empty() {
        return Err(DrawError::ComputeValidation(
            ComputeValidationDecline::EmptySpirv,
        ));
    }
    if req.entry.is_empty() {
        return Err(DrawError::ComputeValidation(
            ComputeValidationDecline::EmptyEntry,
        ));
    }
    if req.entry.as_bytes().contains(&0) {
        return Err(DrawError::ComputeValidation(
            ComputeValidationDecline::EntryInteriorNul,
        ));
    }
    if req.grid.contains(&0) {
        return Err(DrawError::ComputeValidation(
            ComputeValidationDecline::ZeroGrid { grid: req.grid },
        ));
    }
    let mut bindings = BTreeSet::new();
    for b in &req.storage_buffers {
        if !bindings.insert(b.binding) {
            return Err(DrawError::ComputeValidation(
                ComputeValidationDecline::DuplicateStorageBufferBinding { binding: b.binding },
            ));
        }
        if b.bytes.is_empty() {
            return Err(DrawError::ComputeValidation(
                ComputeValidationDecline::EmptyStorageBuffer { binding: b.binding },
            ));
        }
    }
    for img in &req.sampled_images {
        if !bindings.insert(img.binding) {
            return Err(DrawError::ComputeValidation(
                ComputeValidationDecline::DuplicateSampledImageBinding {
                    binding: img.binding,
                },
            ));
        }
        if img.width == 0 || img.height == 0 {
            return Err(DrawError::ComputeValidation(
                ComputeValidationDecline::SampledZeroGeometry {
                    binding: img.binding,
                    width: img.width,
                    height: img.height,
                },
            ));
        }
        let expected = (img.width as usize)
            .saturating_mul(img.height as usize)
            .saturating_mul(img.format.bytes_per_texel());
        if img.bytes.len() != expected {
            return Err(DrawError::ComputeValidation(
                ComputeValidationDecline::SampledBytesLength {
                    binding: img.binding,
                    actual: img.bytes.len(),
                    expected,
                },
            ));
        }
    }
    for sampler in &req.samplers {
        let lod_min = sampler.lod_min_f32();
        let lod_max = sampler.lod_max_f32();
        if !lod_min.is_finite() || !lod_max.is_finite() || lod_min > lod_max {
            return Err(DrawError::ComputeValidation(
                ComputeValidationDecline::InvalidSamplerLod {
                    binding: sampler.binding,
                    lod_min_bits: sampler.lod_min,
                    lod_max_bits: sampler.lod_max,
                },
            ));
        }
        if !bindings.insert(sampler.binding) {
            return Err(DrawError::ComputeValidation(
                ComputeValidationDecline::DuplicateSamplerBinding {
                    binding: sampler.binding,
                },
            ));
        }
    }
    for img in &req.storage_images {
        if !bindings.insert(img.binding) {
            return Err(DrawError::ComputeValidation(
                ComputeValidationDecline::DuplicateStorageImageBinding {
                    binding: img.binding,
                },
            ));
        }
        if img.width == 0 || img.height == 0 {
            return Err(DrawError::ComputeValidation(
                ComputeValidationDecline::StorageZeroGeometry {
                    binding: img.binding,
                    width: img.width,
                    height: img.height,
                },
            ));
        }
        let expected = (img.width as usize)
            .saturating_mul(img.height as usize)
            .saturating_mul(img.format.bytes_per_texel());
        if img.bytes.len() != expected {
            return Err(DrawError::ComputeValidation(
                ComputeValidationDecline::StorageBytesLength {
                    binding: img.binding,
                    actual: img.bytes.len(),
                    expected,
                },
            ));
        }
    }
    Ok(())
}

/// Validate the resident sampled-image copy contract and return whether the
/// source needs a byte-preserving buffer reinterpret hop.
fn resident_sample_reinterpret(
    resource: &ComputeSampledImageResource,
    bind: ComputeResidentSampleBind,
    src_key: StorageImageKey,
) -> Result<bool, DrawError> {
    let exact = src_key.format.vk_format() == resource.format.vk_format()
        && src_key.width == resource.width
        && src_key.height == resource.height;
    let source_row_bytes = src_key.width as u64 * src_key.format.bytes_per_texel() as u64;
    let resource_row_bytes = resource.width as u64 * resource.format.bytes_per_texel() as u64;
    let byte_compatible =
        source_row_bytes == resource_row_bytes && src_key.height == resource.height;
    if !(exact || byte_compatible) {
        return Err(DrawError::ComputeExecution(
            ComputeExecutionDecline::ResidentSampleByteShapeMismatch {
                binding: resource.binding,
                identity: bind.identity,
                source_width: src_key.width,
                source_height: src_key.height,
                source_format: src_key.format,
                source_row_bytes,
                resource_width: resource.width,
                resource_height: resource.height,
                resource_format: resource.format,
                resource_row_bytes,
            },
        ));
    }
    Ok(!exact)
}

pub(crate) unsafe fn execute_compute_inner(
    owner: &mut ContextOwner,
    caches: &mut ObjectCaches,
    pools: &mut ResourcePools,
    counters: &EngineCounters,
    req: &ComputeRequest,
) -> Result<ComputeOutput, DrawError> {
    validate_compute(req)?;
    let force_loss = owner.force_device_lost;
    if force_loss {
        owner.force_device_lost = false;
    }
    let ctx = owner.ensure(counters)?;
    if !ctx.compute_capable {
        return Err(DrawError::Unsupported(
            super::reason::DrawReason::NoCombinedGraphicsComputeQueue,
        ));
    }
    pools.ensure_init(ctx, counters)?;

    // Claim the next ring slot — BEFORE any pool acquire, so a recycled slot
    // can never alias a still-in-flight CB. Blocks (retire) only when every
    // slot is still in flight; the wait lands in retire_wait_us.
    let (cb, fence) = pools.begin_entry(ctx, counters)?;

    let mut layout_bindings = Vec::new();
    for b in &req.storage_buffers {
        layout_bindings.push(BindingSig {
            binding: b.binding,
            ty: vk::DescriptorType::STORAGE_BUFFER.as_raw() as u32,
            stages: vk::ShaderStageFlags::COMPUTE.as_raw(),
        });
    }
    for img in &req.sampled_images {
        layout_bindings.push(BindingSig {
            binding: img.binding,
            ty: vk::DescriptorType::SAMPLED_IMAGE.as_raw() as u32,
            stages: vk::ShaderStageFlags::COMPUTE.as_raw(),
        });
    }
    for sampler in &req.samplers {
        layout_bindings.push(BindingSig {
            binding: sampler.binding,
            ty: vk::DescriptorType::SAMPLER.as_raw() as u32,
            stages: vk::ShaderStageFlags::COMPUTE.as_raw(),
        });
    }
    for img in &req.storage_images {
        layout_bindings.push(BindingSig {
            binding: img.binding,
            ty: vk::DescriptorType::STORAGE_IMAGE.as_raw() as u32,
            stages: vk::ShaderStageFlags::COMPUTE.as_raw(),
        });
    }
    layout_bindings.sort_by_key(|b| b.binding);
    let layout_key = LayoutKey {
        bindings: layout_bindings,
    };

    let (spirv_digest, module) = caches.get_or_create_shader(ctx, &req.spirv, counters, pools)?;
    let (dsl, pipeline_layout) = caches.get_or_create_layout(ctx, &layout_key, counters, pools)?;
    let cpipe_key = ComputePipelineKey {
        spirv: spirv_digest,
        entry: req.entry.clone(),
        layout: layout_key.clone(),
    };
    // One cache, consulted once; `get_or_create_compute_pipeline` counts the hit.
    let pipeline = caches.get_or_create_compute_pipeline(
        ctx,
        &cpipe_key,
        module,
        pipeline_layout,
        counters,
        pools,
    )?;

    // Storage buffers: host-visible staging used as SSBOs (same as draw path).
    let mut storage_slots = Vec::new();
    for resource in &req.storage_buffers {
        let slot = pools.acquire_staging(
            ctx,
            resource.bytes.len() as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            counters,
        )?;
        pools.write_staging(ctx, &slot, &resource.bytes)?;
        storage_slots.push((
            resource.binding,
            slot,
            resource.bytes.len(),
            resource.writable,
        ));
    }

    // Sampled images: device-local + staging seed upload — or a device-local
    // copy from a resident storage image (copy-on-sample: the transient never
    // aliases the live resident, so the same dispatch may storage-write it).
    let mut sampled_slots = Vec::new();
    for resource in &req.sampled_images {
        let key = StorageImageKey {
            width: resource.width,
            height: resource.height,
            format: resource.format,
            sampled_only: true,
        };
        let img = pools.acquire_storage_image(ctx, key, counters)?;
        let (upload, resident_src, reinterpret) = if let Some(bind) = resource.resident_bind {
            // The caller skipped the guest read; the placeholder bytes must
            // never reach the GPU. Every mismatch names the check that
            // refused.
            let Some((src_image, src_key, generation, src_layout)) =
                pools.compute_resident_snapshot(&bind.identity)
            else {
                return Err(DrawError::ComputeExecution(
                    ComputeExecutionDecline::ResidentSampleAbsent {
                        binding: resource.binding,
                        identity: bind.identity,
                        width: resource.width,
                        height: resource.height,
                    },
                ));
            };
            if generation != bind.generation {
                return Err(DrawError::ComputeExecution(
                    ComputeExecutionDecline::ResidentSampleGenerationMismatch {
                        binding: resource.binding,
                        identity: bind.identity,
                        actual_generation: generation,
                        expected_generation: bind.generation,
                    },
                ));
            }
            // An exact source copies image→image; a byte-reinterpret source
            // (rows byte-identical: equal row bytes and height, formats/widths
            // differ) hops through a buffer. Anything else is a shape loss.
            let src_row_bytes = src_key.width as u64 * src_key.format.bytes_per_texel() as u64;
            let reinterpret = resident_sample_reinterpret(resource, bind, src_key)?;
            let hop = if reinterpret {
                let st = pools.acquire_staging(
                    ctx,
                    src_row_bytes * src_key.height as u64,
                    vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
                    counters,
                )?;
                counters.note_compute_sampled_reinterpret_copy(resource.bytes.len() as u64);
                Some(ReinterpretHop {
                    buffer: st.buffer,
                    src_width: src_key.width,
                    src_height: src_key.height,
                })
            } else {
                None
            };
            counters.note_compute_sampled_resident_copy(resource.bytes.len() as u64);
            (None, Some((src_image, src_layout)), hop)
        } else {
            let st = pools.acquire_staging(
                ctx,
                resource.bytes.len() as u64,
                vk::BufferUsageFlags::TRANSFER_SRC,
                counters,
            )?;
            pools.write_staging(ctx, &st, &resource.bytes)?;
            counters.note_compute_sampled_upload(resource.bytes.len() as u64);
            (Some(st), None, None)
        };
        sampled_slots.push(PreparedSampledImage {
            binding: resource.binding,
            img,
            upload,
            resident_src,
            reinterpret,
            width: resource.width,
            height: resource.height,
        });
    }

    let mut sampler_handles = Vec::new();
    for sampler in &req.samplers {
        let handle = caches.get_or_create_sampler(ctx, &sampler.state_key(), counters, pools)?;
        sampler_handles.push((sampler.binding, handle));
    }

    // Storage images: device-local + staging seed upload + readback buffer.
    let mut simg_slots = Vec::new();
    for resource in &req.storage_images {
        let key = StorageImageKey {
            width: resource.width,
            height: resource.height,
            format: resource.format,
            sampled_only: false,
        };
        let (img, initial_layout, generation_match) = if let Some(residency) = resource.residency {
            let resident = pools.acquire_resident_storage_image(
                ctx,
                residency.identity,
                key,
                residency.seed_generation,
                counters,
            )?;
            (resident.slot, resident.layout, resident.generation_match)
        } else {
            (
                pools.acquire_storage_image(ctx, key, counters)?,
                vk::ImageLayout::UNDEFINED,
                false,
            )
        };
        if resource.seed_skipped {
            let Some(residency) = resource.residency else {
                return Err(DrawError::ComputeExecution(
                    ComputeExecutionDecline::SeedSkippedWithoutResidency {
                        binding: resource.binding,
                        width: resource.width,
                        height: resource.height,
                    },
                ));
            };
            if !generation_match {
                // The caller verified the resident generation at stage time
                // and skipped the guest read; seeding the zero placeholder now
                // would silently corrupt the chain. Named failure instead.
                return Err(DrawError::ComputeExecution(
                    ComputeExecutionDecline::ResidentSeedGenerationLost {
                        binding: resource.binding,
                        identity: residency.identity,
                        expected_generation: residency.seed_generation,
                    },
                ));
            }
        }
        let st = if generation_match {
            None
        } else {
            let staging = pools.acquire_staging(
                ctx,
                resource.bytes.len() as u64,
                vk::BufferUsageFlags::TRANSFER_SRC,
                counters,
            )?;
            pools.write_staging(ctx, &staging, &resource.bytes)?;
            counters.note_compute_storage_seed_upload(resource.bytes.len() as u64);
            Some(staging)
        };
        let dst = if resource.defer_readback && resource.residency.is_some() {
            ComputeImageDst::Deferred
        } else {
            // The dispatch's output crosses device→host and the runtime writes
            // the guest pages, so it lands in a readback buffer here.
            ComputeImageDst::Readback(pools.acquire_readback_extra(
                ctx,
                resource.bytes.len() as u64,
                counters,
            )?)
        };
        simg_slots.push(PreparedStorageImage {
            binding: resource.binding,
            slot: img,
            seed: st,
            dst,
            len: resource.bytes.len(),
            width: resource.width,
            height: resource.height,
            initial_layout,
            residency: resource.residency,
        });
    }

    // Descriptor set
    // Owning pool block travels with the set for a correctly-routed free.
    let mut dset_pool: Option<vk::DescriptorPool> = None;
    let dset = if dsl != vk::DescriptorSetLayout::null() {
        let (dset, pool) = pools.alloc_descriptor_set(&ctx.device, dsl, counters)?;
        dset_pool = Some(pool);
        let buffer_infos: Vec<_> = storage_slots
            .iter()
            .map(|(_, s, _, _)| {
                vk::DescriptorBufferInfo::default()
                    .buffer(s.buffer)
                    .offset(0)
                    .range(vk::WHOLE_SIZE)
            })
            .collect();
        let sampled_infos: Vec<_> = sampled_slots
            .iter()
            .map(|prepared| {
                vk::DescriptorImageInfo::default()
                    .image_view(prepared.img.view)
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            })
            .collect();
        let sampler_infos: Vec<_> = sampler_handles
            .iter()
            .map(|(_, sampler)| vk::DescriptorImageInfo::default().sampler(*sampler))
            .collect();
        let image_infos: Vec<_> = simg_slots
            .iter()
            .map(|prepared| {
                vk::DescriptorImageInfo::default()
                    .image_view(prepared.slot.view)
                    .image_layout(vk::ImageLayout::GENERAL)
            })
            .collect();
        let mut writes = Vec::new();
        for (i, (binding, _, _, _)) in storage_slots.iter().enumerate() {
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(dset)
                    .dst_binding(*binding)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(&buffer_infos[i])),
            );
        }
        for (i, prepared) in sampled_slots.iter().enumerate() {
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(dset)
                    .dst_binding(prepared.binding)
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
        for (i, prepared) in simg_slots.iter().enumerate() {
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(dset)
                    .dst_binding(prepared.binding)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(std::slice::from_ref(&image_infos[i])),
            );
        }
        ctx.device.update_descriptor_sets(&writes, &[]);
        Some(dset)
    } else {
        None
    };

    // The ring slot's CB retired at begin_entry and its fence is unsignaled —
    // no pre-record wait remains (pre_record_wait_us stays 0 on this path).
    ctx.device
        .reset_command_buffer(cb, vk::CommandBufferResetFlags::empty())
        .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::ComputeExecResetCb, e)))?;
    ctx.device
        .begin_command_buffer(
            cb,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )
        .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::ComputeExecBeginCb, e)))?;

    // Seed sampled images (staging upload or resident device copy)
    // → SHADER_READ_ONLY_OPTIMAL.
    for prepared in &sampled_slots {
        let img = &prepared.img;
        let range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };
        let barrier = [vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .image(img.image)
            .subresource_range(range)];
        ctx.device.cmd_pipeline_barrier(
            cb,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &barrier,
        );
        if let Some(st) = &prepared.upload {
            let copy = [vk::BufferImageCopy::default()
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_extent(vk::Extent3D {
                    width: prepared.width,
                    height: prepared.height,
                    depth: 1,
                })];
            ctx.device.cmd_copy_buffer_to_image(
                cb,
                st.buffer,
                img.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &copy,
            );
        } else if let Some((src_image, src_layout)) = prepared.resident_src {
            // Copy-on-sample. The resident stays in its registry layout on
            // exit so the storage-acquire's captured initial_layout (and the
            // storage pre-dispatch barrier, which syncs on TRANSFER when that
            // layout is TRANSFER_SRC_OPTIMAL) remains truthful.
            // Unconditional, and the scope comes from `resident_read_source_scope`
            // rather than from `src_layout`. A resident a draw just produced
            // already sits in TRANSFER_SRC_OPTIMAL — that is the layout a render
            // pass resolves its primary to — so gating on a transition being
            // needed skipped the dependency on exactly the content worth
            // copying. The old source mask compounded it: it named
            // SHADER_WRITE | TRANSFER_WRITE but not COLOR_ATTACHMENT_WRITE, so
            // even when it did fire it did not drain the draw that wrote the
            // pixels this copy is about to read.
            let (src_stage, src_access) = super::exec::resident_read_source_scope();
            let to_src = [vk::ImageMemoryBarrier::default()
                .src_access_mask(src_access)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .old_layout(src_layout)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .image(src_image)
                .subresource_range(range)];
            ctx.device.cmd_pipeline_barrier(
                cb,
                src_stage,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &to_src,
            );
            if let Some(hop) = &prepared.reinterpret {
                // Byte-reinterpret: image→buffer→image. The tight buffer holds
                // the identical byte stream under both formats (equal row
                // bytes, equal height — guarded at prepare).
                let layers = vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                };
                let to_buf = [vk::BufferImageCopy::default()
                    .image_subresource(layers)
                    .image_extent(vk::Extent3D {
                        width: hop.src_width,
                        height: hop.src_height,
                        depth: 1,
                    })];
                ctx.device.cmd_copy_image_to_buffer(
                    cb,
                    src_image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    hop.buffer,
                    &to_buf,
                );
                let buf_barrier = [vk::BufferMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                    .buffer(hop.buffer)
                    .offset(0)
                    .size(vk::WHOLE_SIZE)];
                ctx.device.cmd_pipeline_barrier(
                    cb,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &buf_barrier,
                    &[],
                );
                let to_img = [vk::BufferImageCopy::default()
                    .image_subresource(layers)
                    .image_extent(vk::Extent3D {
                        width: prepared.width,
                        height: prepared.height,
                        depth: 1,
                    })];
                ctx.device.cmd_copy_buffer_to_image(
                    cb,
                    hop.buffer,
                    img.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &to_img,
                );
            } else {
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
                        width: prepared.width,
                        height: prepared.height,
                        depth: 1,
                    })];
                ctx.device.cmd_copy_image(
                    cb,
                    src_image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    img.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &copy,
                );
            }
            if src_layout != vk::ImageLayout::TRANSFER_SRC_OPTIMAL {
                let restore = [vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_READ)
                    .dst_access_mask(
                        vk::AccessFlags::SHADER_READ
                            | vk::AccessFlags::SHADER_WRITE
                            | vk::AccessFlags::TRANSFER_READ
                            | vk::AccessFlags::TRANSFER_WRITE,
                    )
                    .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .new_layout(src_layout)
                    .image(src_image)
                    .subresource_range(range)];
                ctx.device.cmd_pipeline_barrier(
                    cb,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &restore,
                );
            }
        }
        let barrier = [vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image(img.image)
            .subresource_range(range)];
        ctx.device.cmd_pipeline_barrier(
            cb,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &barrier,
        );
    }

    // Upload storage-image misses, or transition a generation-matched resident
    // image directly from the prior readback layout into GENERAL.
    for prepared in &simg_slots {
        let img = &prepared.slot;
        let range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };
        let (src_stage, src_access) = match prepared.initial_layout {
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL => (
                vk::PipelineStageFlags::TRANSFER,
                vk::AccessFlags::TRANSFER_READ,
            ),
            vk::ImageLayout::GENERAL => (
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
            ),
            _ => (
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::AccessFlags::empty(),
            ),
        };
        if let Some(st) = &prepared.seed {
            let barrier = [vk::ImageMemoryBarrier::default()
                .src_access_mask(src_access)
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .old_layout(prepared.initial_layout)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .image(img.image)
                .subresource_range(range)];
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
                    width: prepared.width,
                    height: prepared.height,
                    depth: 1,
                })];
            ctx.device.cmd_copy_buffer_to_image(
                cb,
                st.buffer,
                img.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &copy,
            );
        }
        let old_layout = if prepared.seed.is_some() {
            vk::ImageLayout::TRANSFER_DST_OPTIMAL
        } else {
            prepared.initial_layout
        };
        let old_access = if prepared.seed.is_some() {
            vk::AccessFlags::TRANSFER_WRITE
        } else {
            src_access
        };
        let old_stage = if prepared.seed.is_some() {
            vk::PipelineStageFlags::TRANSFER
        } else {
            src_stage
        };
        let barrier = [vk::ImageMemoryBarrier::default()
            .src_access_mask(old_access)
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
            .old_layout(old_layout)
            .new_layout(vk::ImageLayout::GENERAL)
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
            old_stage,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &barrier,
        );
    }

    // Host-written SSBOs visible to compute.
    if !storage_slots.is_empty() {
        let buf_barriers: Vec<_> = storage_slots
            .iter()
            .map(|(_, s, _, writable)| {
                vk::BufferMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::HOST_WRITE)
                    .dst_access_mask(if *writable {
                        vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE
                    } else {
                        vk::AccessFlags::SHADER_READ
                    })
                    .buffer(s.buffer)
                    .offset(0)
                    .size(vk::WHOLE_SIZE)
            })
            .collect();
        ctx.device.cmd_pipeline_barrier(
            cb,
            vk::PipelineStageFlags::HOST,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &buf_barriers,
            &[],
        );
    }

    ctx.device
        .cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pipeline);
    if let Some(dset) = dset {
        ctx.device.cmd_bind_descriptor_sets(
            cb,
            vk::PipelineBindPoint::COMPUTE,
            pipeline_layout,
            0,
            &[dset],
            &[],
        );
    }
    ctx.device
        .cmd_dispatch(cb, req.grid[0], req.grid[1], req.grid[2]);

    // SSBO → host
    if storage_slots.iter().any(|(_, _, _, writable)| *writable) {
        let buf_barriers: Vec<_> = storage_slots
            .iter()
            .filter(|(_, _, _, writable)| *writable)
            .map(|(_, s, _, _)| {
                vk::BufferMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::HOST_READ)
                    .buffer(s.buffer)
                    .offset(0)
                    .size(vk::WHOLE_SIZE)
            })
            .collect();
        ctx.device.cmd_pipeline_barrier(
            cb,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(),
            &[],
            &buf_barriers,
            &[],
        );
    }

    // Storage images → readback buffers
    for prepared in &simg_slots {
        let img = &prepared.slot;
        let barrier = [vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
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
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &barrier,
        );
        let dst_buffer = match &prepared.dst {
            ComputeImageDst::Readback(slot) => slot.buffer,
            // Deferred: no copy at all — the barrier above still lands the
            // image in TRANSFER_SRC_OPTIMAL, matching the layout the registry
            // records post-fence (mark_resident_storage_image).
            ComputeImageDst::Deferred => continue,
        };
        // The pooled readback is always tightly packed from texel zero. The
        // offset and row length were the imported window's, and it had a
        // `buffer_offset` into the guest surface and a guest row stride.
        let (buffer_offset, row_length_texels) = (0u64, 0u32);
        let copy = [vk::BufferImageCopy::default()
            .buffer_offset(buffer_offset)
            .buffer_row_length(row_length_texels)
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .image_extent(vk::Extent3D {
                width: prepared.width,
                height: prepared.height,
                depth: 1,
            })];
        ctx.device.cmd_copy_image_to_buffer(
            cb,
            img.image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            dst_buffer,
            &copy,
        );
    }
    if !simg_slots.is_empty() {
        let barrier = [vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::HOST_READ)];
        ctx.device.cmd_pipeline_barrier(
            cb,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(),
            &barrier,
            &[],
            &[],
        );
    }

    ctx.device
        .end_command_buffer(cb)
        .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::ComputeExecEndCb, e)))?;

    if force_loss {
        if let (Some(ds), Some(pool)) = (dset, dset_pool) {
            pools.free_descriptor_sets(&ctx.device, &[(ds, pool)]);
        }
        pools.recycle_staging();
        pools.recycle_readback();
        pools.recycle_storage_images();
        return Err(DrawError::DeviceLost(DeviceLostDecline::ForcedCompute));
    }

    let queue = ctx.queue();
    let cbs = [cb];
    let si = vk::SubmitInfo::default().command_buffers(&cbs);
    match ctx.device.queue_submit(queue, &[si], fence) {
        Ok(()) => {}
        Err(e) if e == vk::Result::ERROR_DEVICE_LOST => {
            return Err(DrawError::DeviceLost(DeviceLostDecline::Driver {
                op: DeviceLostOp::ComputeSubmit,
                result: e,
            }));
        }
        Err(e) => return Err(DrawError::VkCall(VkCall::new(VkOp::ComputeExecSubmit, e))),
    }

    // A dispatch whose every output stays on the GPU (deferred storage-image
    // writebacks, no writable SSBO readbacks, no direct guest-window DMA) has
    // nothing to hand the CPU — skip the post-submit fence wait and return
    // while the GPU still runs. Ordering stays intact everywhere: every user
    // of the shared fence/CB waits it before reuse, the deferred flush
    // (read_resident_storage) waits it before copying, and the owed
    // descriptor-set/pool cleanup is stashed until a later wait proves the CB
    // retired (drain_pending_compute_cleanup).
    let all_writeback_deferred = storage_slots.iter().all(|(_, _, _, writable)| !writable)
        && simg_slots
            .iter()
            .all(|p| matches!(p.dst, ComputeImageDst::Deferred));
    // Park the owed cleanup (descriptor set + transient pool slots) on this
    // ring slot in every mode; whichever entry retires the slot drains it. A
    // failed wait below leaves the slot pending, so no path ever reuses an
    // unretired fence. The readback maps below stay valid: the BufferSlot
    // handles are held by value and nothing else runs under the engine lock.
    let cleanup = pools.seal_entry(dset.zip(dset_pool).into_iter().collect(), Vec::new());
    pools.finish_entry_async(cleanup);

    if all_writeback_deferred {
        counters
            .compute_post_wait_skips
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    } else {
        pools.retire_all(ctx, counters)?;
    }

    for prepared in &simg_slots {
        if let Some(residency) = prepared.residency {
            pools.mark_resident_storage_image(
                &residency.identity,
                residency.output_generation,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            );
        }
    }

    let mut buffers = Vec::with_capacity(
        storage_slots
            .iter()
            .filter(|(_, _, _, writable)| *writable)
            .count(),
    );
    for (binding, slot, len, writable) in &storage_slots {
        if !writable {
            continue;
        }
        let out = crate::backend::vulkan::engine::pools::read_back_slot(
            ctx,
            slot,
            *len as u64,
            VkOp::ComputeExecMapStorageReadback,
            VkOp::ComputeExecInvalidateStorageReadback,
        )?;
        counters.note_readback(*len as u64);
        buffers.push(ComputeBufferOutput {
            binding: *binding,
            bytes: out,
        });
    }
    let mut images = Vec::with_capacity(simg_slots.len());
    let mut images_deferred = Vec::with_capacity(simg_slots.len());
    for prepared in &simg_slots {
        let readback = match &prepared.dst {
            ComputeImageDst::Deferred => {
                // The pinned resident stays authoritative; the caller flushes
                // it to guest pages on access (read_resident_storage).
                if let Some(residency) = prepared.residency {
                    pools.pin_resident_storage(&residency.identity, true);
                }
                counters.note_compute_deferred_writeback(prepared.len as u64);
                images.push(Vec::new());
                images_deferred.push(true);
                continue;
            }
            ComputeImageDst::Readback(slot) => slot,
        };
        let out = crate::backend::vulkan::engine::pools::read_back_slot(
            ctx,
            readback,
            prepared.len as u64,
            VkOp::ComputeExecMapImageReadback,
            VkOp::ComputeExecInvalidateImageReadback,
        )?;
        counters.note_readback(prepared.len as u64);
        images.push(out);
        images_deferred.push(false);
    }

    // Cleanup was parked on the ring slot right after submit; nothing left
    // to free here (cleanup_us stays 0 on this path).

    counters
        .dispatches
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    Ok(ComputeOutput {
        buffers,
        images,
        images_deferred,
    })
}

/// Copy a completely initialized mapped Vulkan output without first touching
/// every destination page with a redundant zero fill. The caller supplies an
/// exact readable `len`-byte mapping and the copy initializes the entire Vec
/// capacity before its length becomes visible.
///
/// # Safety
///
/// `ptr` must reference a readable `len`-byte mapping for the duration of this
/// call.
pub(super) unsafe fn copy_mapped_output(ptr: *const u8, len: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(len);
    std::ptr::copy_nonoverlapping(ptr, out.as_mut_ptr(), len);
    out.set_len(len);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::vulkan::engine::{
        ComputeResidentSampleBind, ComputeSampledImageResource, ComputeStorageImageResource,
        SamplerResource, StorageImageFormat,
    };
    use crate::model::ComputeStorageResidencyKey;
    use crate::observe::Decline;

    fn residency_identity() -> ComputeStorageResidencyKey {
        ComputeStorageResidencyKey {
            mapping_id: 7,
            map_generation: 8,
            surface_offset: 0,
            surface_bpr: 4,
            span_end: 4,
            width: 1,
            height: 1,
            pixel_format: 80,
            texture_ref: 0,
        }
    }

    fn resident_sample_resource() -> ComputeSampledImageResource {
        ComputeSampledImageResource {
            binding: 32,
            format: StorageImageFormat::Rgba8Unorm,
            width: 1,
            height: 1,
            bytes: vec![0; 4],
            resident_bind: Some(ComputeResidentSampleBind {
                identity: residency_identity(),
                generation: 9,
            }),
        }
    }
    fn resident_sample_key() -> StorageImageKey {
        StorageImageKey {
            width: 1,
            height: 1,
            format: StorageImageFormat::Rgba8Unorm,
            sampled_only: false,
        }
    }

    fn resident_sample_shape_slug(
        resource: &ComputeSampledImageResource,
        source: StorageImageKey,
    ) -> &'static str {
        let bind = resource.resident_bind.unwrap();
        match resident_sample_reinterpret(resource, bind, source) {
            Err(DrawError::ComputeExecution(decline)) => decline.slug(),
            Err(other) => panic!("expected typed compute execution decline, got {other}"),
            Ok(_) => panic!("expected resident-sample shape refusal"),
        }
    }

    #[test]
    fn mapped_output_copy_initializes_exact_bytes_without_seed_buffer() {
        let source = [0x31, 0x00, 0x7f, 0xff, 0x42];
        let out = unsafe { copy_mapped_output(source.as_ptr(), source.len()) };
        assert_eq!(out, source);
    }

    #[test]
    fn compute_entry_with_interior_nul_is_rejected_before_cache_creation() {
        let req = ComputeRequest {
            spirv: vec![0x0723_0203],
            entry: "ma\0in".into(),
            grid: [1, 1, 1],
            ..Default::default()
        };
        let decline = match validate_compute(&req) {
            Err(DrawError::ComputeValidation(decline)) => decline,
            Err(other) => panic!("expected typed compute validation, got {other}"),
            Ok(()) => panic!("expected interior-NUL rejection"),
        };
        assert_eq!(decline.slug(), "vk_compute_validate_entry_interior_nul");
    }

    #[test]
    fn resident_sample_shape_causes_are_not_collapsed() {
        let exact = resident_sample_resource();
        assert_eq!(
            resident_sample_reinterpret(
                &exact,
                exact.resident_bind.unwrap(),
                resident_sample_key()
            ),
            Ok(false)
        );

        let mut reinterpret = resident_sample_resource();
        reinterpret.width = 2;
        reinterpret.format = StorageImageFormat::Rg8Unorm;
        reinterpret.bytes.resize(4, 0);
        assert_eq!(
            resident_sample_reinterpret(
                &reinterpret,
                reinterpret.resident_bind.unwrap(),
                resident_sample_key()
            ),
            Ok(true)
        );

        let mut byte_mismatch = resident_sample_resource();
        byte_mismatch.width = 2;
        byte_mismatch.bytes.resize(8, 0);
        assert_eq!(
            resident_sample_shape_slug(&byte_mismatch, resident_sample_key()),
            "vk_compute_exec_resident_sample_byte_shape_mismatch"
        );
    }

    #[test]
    fn sampled_and_storage_images_keep_distinct_descriptor_access() {
        let mut req = ComputeRequest {
            spirv: vec![0x0723_0203],
            entry: "main".into(),
            grid: [1, 1, 1],
            sampled_images: vec![ComputeSampledImageResource {
                binding: 32,
                format: StorageImageFormat::Rgba8Unorm,
                width: 1,
                height: 1,
                bytes: vec![0; 4],
                resident_bind: None,
            }],
            samplers: vec![SamplerResource::normalized_default(64)],
            storage_images: vec![ComputeStorageImageResource {
                binding: 34,
                format: StorageImageFormat::Rgba8Uint,
                width: 1,
                height: 1,
                bytes: vec![0; 4],
                residency: None,
                seed_skipped: false,
                defer_readback: false,
            }],
            ..Default::default()
        };
        assert!(validate_compute(&req).is_ok());

        req.storage_images[0].binding = 32;
        let decline = match validate_compute(&req) {
            Err(DrawError::ComputeValidation(decline)) => decline,
            Err(other) => panic!("expected typed compute validation, got {other}"),
            Ok(()) => panic!("expected descriptor collision"),
        };
        assert_eq!(
            decline.slug(),
            "vk_compute_validate_duplicate_storage_image_binding"
        );
    }
}
