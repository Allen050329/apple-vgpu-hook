//! Compute encode path: PSO cache, binds, dispatch core, reflection.

use crate::backend::metal::abi::*;
use crate::backend::metal::cache::{
    compute_pso_insert, compute_pso_lookup, reflect_insert, reflect_lookup,
};
use crate::backend::metal::constants::*;
use crate::backend::metal::format::storage_image_format;
use crate::backend::metal::function::load_only_function;
use crate::backend::metal::hash::hash_bytes;
use crate::backend::metal::raw_metal::{
    command_buffer_error_description, mtl_size, new_compute_pso_with_function_reflection,
    new_texture_view_swizzled, reflection_bindings, set_buffer_with_attribute_stride,
    set_imageblock_width_height, set_stage_in_region, set_stage_in_region_indirect,
    texture_swizzle_channels, BINDING_ACCESS_READ_ONLY, BINDING_ACCESS_READ_WRITE,
    BINDING_ACCESS_WRITE_ONLY, BINDING_TYPE_TEXTURE,
};
use crate::backend::metal::runtime::{new_buffer_from_host, system_device, thread_queue};
use crate::backend::metal::samplers::make_explicit_sampler;
use crate::backend::metal::stage_input::{
    has_indexed_layout, layout_for_buffer, make_compute_stage_input_descriptor,
};
use crate::backend::metal::util::{
    bytes_of, clear_err, f32_from_bits, image_len, sampler_index, set_err, texture_index,
    valid_buffer_binding, ErrOut, Status,
};
use metal::*;
use std::ptr;

pub fn hash_compute_stage_input(stage_input: Option<&ReimsVgpuComputeStageInputDescriptor>) -> u64 {
    match stage_input {
        None => 0,
        Some(s) => hash_bytes(bytes_of(s)),
    }
}

fn new_compute_pipeline_state_uncached(
    device: &Device,
    function: &Function,
    stage_input: Option<&ReimsVgpuComputeStageInputDescriptor>,
    err: ErrOut<'_>,
) -> Result<ComputePipelineState, Status> {
    match stage_input {
        None => device
            .new_compute_pipeline_state_with_function(function)
            .map_err(|e| {
                set_err(err, format!("compute PSO failed: {e}"));
                Status::execute("metal_compute_pso_create_failed")
            }),
        Some(si) => {
            let stage_descriptor = make_compute_stage_input_descriptor(si, err)?;
            let descriptor = ComputePipelineDescriptor::new();
            descriptor.set_compute_function(Some(function));
            descriptor.set_stage_input_descriptor(Some(&stage_descriptor));
            device.new_compute_pipeline_state(&descriptor).map_err(|e| {
                set_err(err, format!("compute PSO failed: {e}"));
                Status::execute("metal_compute_stage_input_pso_create_failed")
            })
        }
    }
}

pub fn new_compute_pipeline_state(
    device: &Device,
    function: &Function,
    mtlb: &[u8],
    stage_input: Option<&ReimsVgpuComputeStageInputDescriptor>,
    err: ErrOut<'_>,
) -> Result<ComputePipelineState, Status> {
    let mtlb_hash = hash_bytes(mtlb);
    let stage_hash = hash_compute_stage_input(stage_input);
    let has_stage = if stage_input.is_some() { 1u8 } else { 0u8 };
    if let Some(hit) = compute_pso_lookup(mtlb_hash, mtlb.len(), stage_hash, has_stage) {
        return Ok(hit);
    }
    let pso = new_compute_pipeline_state_uncached(device, function, stage_input, err)?;
    Ok(compute_pso_insert(
        mtlb_hash,
        mtlb.len(),
        stage_hash,
        has_stage,
        pso,
    ))
}

fn compute_buffer_backing(buffer: &ReimsVgpuBuffer) -> Result<(*mut u8, usize, usize), Status> {
    if !buffer.backing_data.is_null() {
        if buffer.backing_len == 0 {
            return Err(
                Status::args("metal_compute_backing_length_zero").field("binding", buffer.binding)
            );
        }
        if buffer.backing_offset > buffer.backing_len {
            return Err(Status::args("metal_compute_backing_offset_out_of_range")
                .field("binding", buffer.binding)
                .field("offset", buffer.backing_offset)
                .field("backing_len", buffer.backing_len));
        }
        if buffer.len > buffer.backing_len - buffer.backing_offset {
            return Err(Status::args("metal_compute_backing_span_out_of_range")
                .field("binding", buffer.binding)
                .field("len", buffer.len)
                .field("offset", buffer.backing_offset)
                .field("backing_len", buffer.backing_len));
        }
        Ok((
            buffer.backing_data,
            buffer.backing_len,
            buffer.backing_offset,
        ))
    } else {
        Ok((buffer.data, buffer.len, 0))
    }
}

fn compute_buffer_backing_matches(a: &ReimsVgpuBuffer, b: &ReimsVgpuBuffer) -> bool {
    match (compute_buffer_backing(a), compute_buffer_backing(b)) {
        (Ok((ad, al, _)), Ok((bd, bl, _))) => ad == bd && al == bl,
        _ => false,
    }
}

fn bind_compute_buffers(
    device: &Device,
    encoder: &ComputeCommandEncoderRef,
    buffers: &mut [ReimsVgpuBuffer],
    stage_input: Option<&ReimsVgpuComputeStageInputDescriptor>,
    mtl_buffers: &mut Vec<Buffer>,
    err: ErrOut<'_>,
) -> Status {
    let needs_index = has_indexed_layout(stage_input);
    if buffers.is_empty() {
        if needs_index {
            let idx = stage_input.map(|s| s.index_buffer_index).unwrap_or(0);
            set_err(
                err,
                format!("missing compute stageInputDescriptor index buffer {idx}"),
            );
            return Status::args("metal_compute_index_buffer_missing").field("buffer", idx);
        }
        return Status::OK;
    }

    let mut seen = [false; REIMS_VGPU_METAL_MAX_BUFFERS];
    for i in 0..buffers.len() {
        let buffer = &buffers[i];
        if !valid_buffer_binding(buffer.binding) {
            set_err(
                err,
                format!("invalid compute buffer binding {}", buffer.binding),
            );
            return Status::args("metal_compute_buffer_binding_out_of_range")
                .field("binding", buffer.binding)
                .field("limit", REIMS_VGPU_METAL_MAX_BUFFERS);
        }
        if buffer.data.is_null() {
            set_err(
                err,
                format!("invalid compute buffer binding {}", buffer.binding),
            );
            return Status::args("metal_compute_buffer_data_missing")
                .field("binding", buffer.binding);
        }
        if buffer.len == 0 {
            set_err(
                err,
                format!("invalid compute buffer binding {}", buffer.binding),
            );
            return Status::args("metal_compute_buffer_length_zero")
                .field("binding", buffer.binding);
        }
        if seen[buffer.binding as usize] {
            set_err(
                err,
                format!("duplicate compute buffer binding {}", buffer.binding),
            );
            return Status::args("metal_compute_buffer_binding_duplicate")
                .field("binding", buffer.binding);
        }
        seen[buffer.binding as usize] = true;

        let (stage_layout, stage_has_attr) = layout_for_buffer(stage_input, buffer.binding);
        if buffer.has_attribute_stride != 0 {
            let ok = stage_input.is_some()
                && stage_has_attr
                && stage_layout
                    .map(|l| l.stride == REIMS_VGPU_COMPUTE_STAGE_INPUT_STRIDE_DYNAMIC)
                    .unwrap_or(false);
            if !ok {
                set_err(
                    err,
                    format!(
                        "compute buffer {} has attributeStride but no matching dynamic \
                         compute stageInputDescriptor layout",
                        buffer.binding
                    ),
                );
                return Status::args("metal_compute_attribute_stride_without_dynamic_layout")
                    .field("binding", buffer.binding)
                    .field("stride", buffer.attribute_stride);
            }
        } else if stage_has_attr
            && stage_layout
                .map(|l| l.stride == REIMS_VGPU_COMPUTE_STAGE_INPUT_STRIDE_DYNAMIC)
                .unwrap_or(false)
        {
            set_err(
                err,
                format!(
                    "compute buffer {} uses a dynamic compute stageInputDescriptor layout \
                     without attributeStride",
                    buffer.binding
                ),
            );
            return Status::args("metal_compute_dynamic_layout_without_attribute_stride")
                .field("binding", buffer.binding);
        }

        let (backing_data, backing_len, backing_offset) = match compute_buffer_backing(buffer) {
            Ok(v) => v,
            Err(status) => {
                set_err(
                    err,
                    format!("invalid compute buffer backing {}", buffer.binding),
                );
                return status;
            }
        };

        let mut mtl_buffer: Option<Buffer> = None;
        for j in 0..i {
            if compute_buffer_backing_matches(buffer, &buffers[j]) {
                mtl_buffer = Some(mtl_buffers[j].clone());
                break;
            }
        }
        let mtl_buffer = match mtl_buffer {
            Some(b) => b,
            None => match new_buffer_from_host(device, backing_data, backing_len) {
                Some(b) => b,
                None => {
                    set_err(
                        err,
                        format!("failed to create compute buffer {}", buffer.binding),
                    );
                    return Status::execute("metal_compute_buffer_create_failed")
                        .field("binding", buffer.binding)
                        .field("backing_len", backing_len);
                }
            },
        };
        mtl_buffers.push(mtl_buffer.clone());
        if buffer.has_attribute_stride != 0 {
            set_buffer_with_attribute_stride(
                encoder,
                &mtl_buffer,
                backing_offset as u64,
                buffer.attribute_stride,
                buffer.binding as u64,
            );
        } else {
            encoder.set_buffer(
                buffer.binding as u64,
                Some(&mtl_buffer),
                backing_offset as u64,
            );
        }
    }

    if let Some(indexed_stage_input) = stage_input.filter(|_| needs_index) {
        let idx = indexed_stage_input.index_buffer_index as usize;
        if idx >= REIMS_VGPU_METAL_MAX_BUFFERS || !seen[idx] {
            set_err(
                err,
                format!("missing compute stageInputDescriptor index buffer {idx}"),
            );
            return Status::args("metal_compute_stage_input_index_buffer_missing")
                .field("buffer", idx);
        }
    }
    Status::OK
}

pub(crate) fn bind_storage_images(
    device: &Device,
    encoder: &ComputeCommandEncoderRef,
    images: &mut [ReimsVgpuStorageImage],
    mtl_images: &mut Vec<Texture>,
    err: ErrOut<'_>,
) -> Status {
    if images.is_empty() {
        return Status::OK;
    }
    let mut seen = [false; REIMS_VGPU_METAL_MAX_TEXTURES];
    for image in images.iter() {
        let Some(texture_index) = texture_index(image.binding) else {
            set_err(
                err,
                format!("invalid storage image binding {}", image.binding),
            );
            return Status::args("metal_compute_storage_binding_invalid")
                .field("binding", image.binding);
        };
        let Some((pixel_format, bpp)) = storage_image_format(image.format) else {
            set_err(
                err,
                format!("invalid storage image binding {}", image.binding),
            );
            return Status::args("metal_compute_storage_format_unsupported")
                .field("binding", image.binding)
                .field("format", image.format);
        };
        let Some(expected_len) = image_len(image.width, image.height, bpp) else {
            set_err(
                err,
                format!("invalid storage image binding {}", image.binding),
            );
            return Status::args("metal_compute_storage_geometry_invalid")
                .field("binding", image.binding)
                .field("width", image.width)
                .field("height", image.height)
                .field("bpp", bpp);
        };
        if image.data.is_null() {
            set_err(
                err,
                format!("invalid storage image binding {}", image.binding),
            );
            return Status::args("metal_compute_storage_data_missing")
                .field("binding", image.binding);
        }
        if image.len < expected_len {
            set_err(
                err,
                format!("invalid storage image binding {}", image.binding),
            );
            return Status::args("metal_compute_storage_data_too_short")
                .field("binding", image.binding)
                .field("len", image.len)
                .field("expected", expected_len);
        }
        if seen[texture_index] {
            set_err(
                err,
                format!("duplicate storage image binding {}", image.binding),
            );
            return Status::args("metal_compute_storage_binding_duplicate")
                .field("binding", image.binding);
        }
        seen[texture_index] = true;

        let descriptor = TextureDescriptor::new();
        descriptor.set_texture_type(MTLTextureType::D2);
        descriptor.set_pixel_format(pixel_format);
        descriptor.set_width(image.width as u64);
        descriptor.set_height(image.height as u64);
        descriptor.set_storage_mode(MTLStorageMode::Shared);
        descriptor.set_usage(MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite);
        let texture = device.new_texture(&descriptor);
        let region = MTLRegion::new_2d(0, 0, image.width as u64, image.height as u64);
        texture.replace_region(
            region,
            0,
            image.data as *const _,
            (image.width as u64) * (bpp as u64),
        );
        encoder.set_texture(texture_index as u64, Some(&texture));
        mtl_images.push(texture);
    }
    Status::OK
}

pub(crate) fn bind_compute_sampled_images(
    device: &Device,
    encoder: &ComputeCommandEncoderRef,
    sampled: &[ReimsVgpuComputeSampledImage],
    mtl_sampled: &mut Vec<Texture>,
    err: ErrOut<'_>,
) -> Status {
    if sampled.is_empty() {
        return Status::OK;
    }
    let mut seen = [false; REIMS_VGPU_METAL_MAX_TEXTURES];
    for image in sampled {
        let Some(texture_index) = texture_index(image.binding) else {
            set_err(
                err,
                format!("invalid sampled compute image binding {}", image.binding),
            );
            return Status::args("metal_compute_sampled_binding_invalid")
                .field("binding", image.binding);
        };
        let Some((pixel_format, bpp)) = storage_image_format(image.format) else {
            set_err(
                err,
                format!("invalid sampled compute image binding {}", image.binding),
            );
            return Status::args("metal_compute_sampled_format_unsupported")
                .field("binding", image.binding)
                .field("format", image.format);
        };
        let Some(expected_len) = image_len(image.width, image.height, bpp) else {
            set_err(
                err,
                format!("invalid sampled compute image binding {}", image.binding),
            );
            return Status::args("metal_compute_sampled_geometry_invalid")
                .field("binding", image.binding)
                .field("width", image.width)
                .field("height", image.height)
                .field("bpp", bpp);
        };
        if image.data.is_null() {
            set_err(
                err,
                format!("invalid sampled compute image binding {}", image.binding),
            );
            return Status::args("metal_compute_sampled_data_missing")
                .field("binding", image.binding);
        }
        if image.len < expected_len {
            set_err(
                err,
                format!("invalid sampled compute image binding {}", image.binding),
            );
            return Status::args("metal_compute_sampled_data_too_short")
                .field("binding", image.binding)
                .field("len", image.len)
                .field("expected", expected_len);
        }
        let swizzle = if image.has_swizzle != 0 {
            match texture_swizzle_channels(image.swizzle) {
                Some(s) => Some(s),
                None => {
                    set_err(
                        err,
                        format!("invalid sampled compute image swizzle {}", image.binding),
                    );
                    return Status::args("metal_compute_sampled_swizzle_invalid")
                        .field("binding", image.binding)
                        .field("swizzle", u32::from_le_bytes(image.swizzle));
                }
            }
        } else {
            None
        };
        if seen[texture_index] {
            set_err(
                err,
                format!("duplicate sampled compute image binding {}", image.binding),
            );
            return Status::args("metal_compute_sampled_binding_duplicate")
                .field("binding", image.binding);
        }
        seen[texture_index] = true;

        let descriptor = TextureDescriptor::new();
        descriptor.set_texture_type(MTLTextureType::D2);
        descriptor.set_pixel_format(pixel_format);
        descriptor.set_width(image.width as u64);
        descriptor.set_height(image.height as u64);
        descriptor.set_storage_mode(MTLStorageMode::Shared);
        let mut usage = MTLTextureUsage::ShaderRead;
        if swizzle.is_some() {
            usage |= MTLTextureUsage::PixelFormatView;
        }
        descriptor.set_usage(usage);
        let texture = device.new_texture(&descriptor);
        let region = MTLRegion::new_2d(0, 0, image.width as u64, image.height as u64);
        texture.replace_region(
            region,
            0,
            image.data as *const _,
            (image.width as u64) * (bpp as u64),
        );
        mtl_sampled.push(texture.clone());
        let bound = if let Some(sw) = swizzle {
            match new_texture_view_swizzled(&texture, pixel_format, sw) {
                Some(v) => {
                    mtl_sampled.push(v.clone());
                    v
                }
                None => {
                    set_err(
                        err,
                        format!(
                            "failed to create sampled compute swizzle view {}",
                            image.binding
                        ),
                    );
                    return Status::execute("metal_compute_sampled_swizzle_view_create_failed")
                        .field("binding", image.binding)
                        .field("format", image.format);
                }
            }
        } else {
            texture
        };
        encoder.set_texture(texture_index as u64, Some(&bound));
    }
    Status::OK
}

pub(crate) fn bind_compute_samplers(
    device: &Device,
    encoder: &ComputeCommandEncoderRef,
    samplers: &[ReimsVgpuSampler],
    err: ErrOut<'_>,
) -> Status {
    if samplers.is_empty() {
        return Status::OK;
    }
    if samplers.len() > REIMS_VGPU_METAL_MAX_SAMPLERS {
        set_err(err, "too many compute samplers");
        return Status::args("metal_compute_sampler_count_exceeded")
            .field("count", samplers.len())
            .field("limit", REIMS_VGPU_METAL_MAX_SAMPLERS);
    }
    let mut seen = [false; REIMS_VGPU_METAL_MAX_SAMPLERS];
    for s in samplers {
        let Some(index) = sampler_index(s.binding) else {
            set_err(
                err,
                format!("invalid compute sampler binding {}", s.binding),
            );
            return Status::args("metal_compute_sampler_binding_invalid")
                .field("binding", s.binding);
        };
        if seen[index] {
            set_err(
                err,
                format!("duplicate compute sampler binding {}", s.binding),
            );
            return Status::args("metal_compute_sampler_binding_duplicate")
                .field("binding", s.binding);
        }
        let sampler = match make_explicit_sampler(device, s, err) {
            Ok(s) => s,
            Err(st) => return st,
        };
        seen[index] = true;
        if s.has_lod_clamp != 0 {
            encoder.set_sampler_state_with_lod(
                index as u64,
                Some(&sampler),
                f32_from_bits(s.clamp_lod_min_bits)..f32_from_bits(s.clamp_lod_max_bits),
            );
        } else {
            encoder.set_sampler_state(index as u64, Some(&sampler));
        }
    }
    Status::OK
}

fn bind_threadgroup_memory(encoder: &ComputeCommandEncoderRef, tg: &[ReimsVgpuThreadgroupMemory]) {
    for entry in tg {
        encoder.set_threadgroup_memory_length(entry.index as u64, entry.length);
    }
}

fn bind_stage_in_region(
    encoder: &ComputeCommandEncoderRef,
    region: Option<&ReimsVgpuComputeStageInRegion>,
) {
    let Some(region) = region else {
        return;
    };
    let metal_region = MTLRegion {
        origin: MTLOrigin {
            x: region.origin_x,
            y: region.origin_y,
            z: region.origin_z,
        },
        size: MTLSize {
            width: region.size_x,
            height: region.size_y,
            depth: region.size_z,
        },
    };
    set_stage_in_region(encoder, metal_region);
}

fn bind_stage_in_region_indirect(
    device: &Device,
    encoder: &ComputeCommandEncoderRef,
    retained: &mut Vec<Buffer>,
    arguments: Option<&ReimsVgpuComputeStageInRegionIndirectArguments>,
) {
    let Some(arguments) = arguments else {
        return;
    };
    let bytes = bytes_of(arguments);
    let indirect = device.new_buffer_with_data(
        bytes.as_ptr() as *const _,
        bytes.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    retained.push(indirect.clone());
    set_stage_in_region_indirect(encoder, &indirect, 0);
}

fn mtl_dispatch_type(raw: u32) -> Option<MTLDispatchType> {
    match raw {
        REIMS_VGPU_MTL_DISPATCH_TYPE_SERIAL => Some(MTLDispatchType::Serial),
        REIMS_VGPU_MTL_DISPATCH_TYPE_CONCURRENT => Some(MTLDispatchType::Concurrent),
        _ => None,
    }
}

/// Metal resources retained after encode for deferred writeback (session path).
pub struct ComputeEncodeRetain {
    pub buffers: Vec<Buffer>,
    pub images: Vec<Texture>,
    pub sampled: Vec<Texture>,
    pub indirect: Vec<Buffer>,
}

/// Encode one compute dispatch onto an existing encoder (no end/commit/wait).
///
/// Used by multi-record control-flow sessions so nested dispatches sit inside
/// `encodeStartIf`/`While` SPI regions. Caller must keep `retain` alive until
/// after GPU completion, then call [`compute_writeback_from_mtl`].
// Twenty arguments because they are the compute dispatch's whole input set —
// the eight `ReimsVgpu*` bind arrays plus the grid — and this is the encoder-
// borrowing twin of `compute_core` below. Grouping them into a struct would
// have to be done to both or it splits one contract in two.
#[allow(clippy::too_many_arguments)]
pub fn compute_encode_on_encoder(
    device: &Device,
    encoder: &ComputeCommandEncoderRef,
    mtlb: &[u8],
    buffers: &mut [ReimsVgpuBuffer],
    images: &mut [ReimsVgpuStorageImage],
    sampled: &[ReimsVgpuComputeSampledImage],
    samplers: &[ReimsVgpuSampler],
    threadgroup_memory: &[ReimsVgpuThreadgroupMemory],
    stage_in_region: Option<&ReimsVgpuComputeStageInRegion>,
    stage_in_region_indirect: Option<&ReimsVgpuComputeStageInRegionIndirectArguments>,
    imageblock_dimensions: Option<&ReimsVgpuComputeImageblockDimensions>,
    stage_input: Option<&ReimsVgpuComputeStageInputDescriptor>,
    dispatch_kind: u32,
    grid_x: u32,
    grid_y: u32,
    grid_z: u32,
    tg_x: u32,
    tg_y: u32,
    tg_z: u32,
    err: ErrOut<'_>,
) -> Result<ComputeEncodeRetain, Status> {
    if grid_x == 0 {
        set_err(
            err,
            "compute grid and threadgroup dimensions must be non-zero",
        );
        return Err(Status::args("metal_compute_grid_x_zero"));
    }
    if grid_y == 0 {
        set_err(
            err,
            "compute grid and threadgroup dimensions must be non-zero",
        );
        return Err(Status::args("metal_compute_grid_y_zero"));
    }
    if grid_z == 0 {
        set_err(
            err,
            "compute grid and threadgroup dimensions must be non-zero",
        );
        return Err(Status::args("metal_compute_grid_z_zero"));
    }
    if tg_x == 0 {
        set_err(
            err,
            "compute grid and threadgroup dimensions must be non-zero",
        );
        return Err(Status::args("metal_compute_threadgroup_x_zero"));
    }
    if tg_y == 0 {
        set_err(
            err,
            "compute grid and threadgroup dimensions must be non-zero",
        );
        return Err(Status::args("metal_compute_threadgroup_y_zero"));
    }
    if tg_z == 0 {
        set_err(
            err,
            "compute grid and threadgroup dimensions must be non-zero",
        );
        return Err(Status::args("metal_compute_threadgroup_z_zero"));
    }
    let dispatch_threads = match dispatch_kind {
        REIMS_VGPU_COMPUTE_DISPATCH_KIND_THREADS => true,
        REIMS_VGPU_COMPUTE_DISPATCH_KIND_THREADGROUPS => false,
        other => {
            set_err(err, format!("invalid compute dispatch kind {other}"));
            return Err(Status::args("metal_compute_dispatch_kind_invalid").field("kind", other));
        }
    };

    let function = load_only_function(device, mtlb, "compute", err)?;
    let pso = new_compute_pipeline_state(device, &function, mtlb, stage_input, err)?;

    let threadgroup_total = (tg_x as u64) * (tg_y as u64) * (tg_z as u64);
    let max_tg = pso.max_total_threads_per_threadgroup();
    if threadgroup_total > max_tg {
        set_err(
            err,
            format!(
                "compute PSO max threads per threadgroup is {max_tg}, need {threadgroup_total}"
            ),
        );
        return Err(Status::execute("metal_compute_threadgroup_limit_exceeded")
            .field("requested", threadgroup_total)
            .field("limit", max_tg));
    }

    encoder.set_compute_pipeline_state(&pso);

    let mut mtl_buffers = Vec::with_capacity(buffers.len());
    let rc = bind_compute_buffers(device, encoder, buffers, stage_input, &mut mtl_buffers, err);
    if !rc.is_ok() {
        return Err(rc);
    }
    let mut mtl_images = Vec::with_capacity(images.len());
    let rc = bind_storage_images(device, encoder, images, &mut mtl_images, err);
    if !rc.is_ok() {
        return Err(rc);
    }
    let mut mtl_sampled = Vec::with_capacity(sampled.len());
    let rc = bind_compute_sampled_images(device, encoder, sampled, &mut mtl_sampled, err);
    if !rc.is_ok() {
        return Err(rc);
    }
    let rc = bind_compute_samplers(device, encoder, samplers, err);
    if !rc.is_ok() {
        return Err(rc);
    }
    bind_threadgroup_memory(encoder, threadgroup_memory);
    bind_stage_in_region(encoder, stage_in_region);
    let mut retained_indirect = Vec::new();
    bind_stage_in_region_indirect(
        device,
        encoder,
        &mut retained_indirect,
        stage_in_region_indirect,
    );
    if let Some(dims) = imageblock_dimensions {
        set_imageblock_width_height(encoder, dims.width as u64, dims.height as u64);
    }

    let grid = mtl_size(grid_x as u64, grid_y as u64, grid_z as u64);
    let tptg = mtl_size(tg_x as u64, tg_y as u64, tg_z as u64);
    if dispatch_threads {
        encoder.dispatch_threads(grid, tptg);
    } else {
        encoder.dispatch_thread_groups(grid, tptg);
    }
    clear_err(err);
    Ok(ComputeEncodeRetain {
        buffers: mtl_buffers,
        images: mtl_images,
        sampled: mtl_sampled,
        indirect: retained_indirect,
    })
}

/// Copy GPU buffer/image contents back into host `ReimsVgpuBuffer` / `ReimsVgpuStorageImage` pointers.
pub fn compute_writeback_from_mtl(
    buffers: &mut [ReimsVgpuBuffer],
    mtl_buffers: &[Buffer],
    images: &mut [ReimsVgpuStorageImage],
    mtl_images: &[Texture],
    err: ErrOut<'_>,
) -> Status {
    if mtl_buffers.len() != buffers.len() {
        set_err(err, "compute writeback buffer count mismatch");
        return Status::args("metal_compute_writeback_buffer_count_mismatch")
            .field("buffers", buffers.len())
            .field("metal_buffers", mtl_buffers.len());
    }
    if mtl_images.len() != images.len() {
        set_err(err, "compute writeback image count mismatch");
        return Status::args("metal_compute_writeback_image_count_mismatch")
            .field("images", images.len())
            .field("metal_images", mtl_images.len());
    }
    for i in 0..buffers.len() {
        let mut already = false;
        for j in 0..i {
            if compute_buffer_backing_matches(&buffers[i], &buffers[j]) {
                already = true;
                break;
            }
        }
        if already {
            continue;
        }
        let (backing_data, backing_len, _) = match compute_buffer_backing(&buffers[i]) {
            Ok(v) => v,
            Err(status) => {
                set_err(
                    err,
                    format!("invalid compute buffer backing {}", buffers[i].binding),
                );
                return status;
            }
        };
        let mtl_buffer = &mtl_buffers[i];
        unsafe {
            ptr::copy_nonoverlapping(
                mtl_buffer.contents() as *const u8,
                backing_data,
                backing_len,
            );
        }
    }
    for (i, image) in images.iter().enumerate() {
        let Some((_, bpp)) = storage_image_format(image.format) else {
            set_err(
                err,
                format!("invalid storage image format {}", image.format),
            );
            return Status::args("metal_compute_writeback_storage_format_unsupported")
                .field("binding", image.binding)
                .field("format", image.format);
        };
        let texture = &mtl_images[i];
        let region = MTLRegion::new_2d(0, 0, image.width as u64, image.height as u64);
        texture.get_bytes(
            image.data as *mut _,
            (image.width as u64) * (bpp as u64),
            region,
            0,
        );
    }
    clear_err(err);
    Status::OK
}

// The same input set as `compute_encode_on_encoder`, one encoder shorter.
#[allow(clippy::too_many_arguments)]
pub fn compute_core(
    mtlb: &[u8],
    buffers: &mut [ReimsVgpuBuffer],
    images: &mut [ReimsVgpuStorageImage],
    sampled: &[ReimsVgpuComputeSampledImage],
    samplers: &[ReimsVgpuSampler],
    threadgroup_memory: &[ReimsVgpuThreadgroupMemory],
    stage_in_region: Option<&ReimsVgpuComputeStageInRegion>,
    stage_in_region_indirect: Option<&ReimsVgpuComputeStageInRegionIndirectArguments>,
    imageblock_dimensions: Option<&ReimsVgpuComputeImageblockDimensions>,
    stage_input: Option<&ReimsVgpuComputeStageInputDescriptor>,
    dispatch_kind: u32,
    dispatch_type: u32,
    grid_x: u32,
    grid_y: u32,
    grid_z: u32,
    tg_x: u32,
    tg_y: u32,
    tg_z: u32,
    err: ErrOut<'_>,
) -> Status {
    let Some(metal_dispatch_type) = mtl_dispatch_type(dispatch_type) else {
        set_err(
            err,
            format!("invalid compute dispatch type {dispatch_type}"),
        );
        return Status::args("metal_compute_dispatch_type_invalid")
            .field("dispatch_type", dispatch_type);
    };

    let Some(device) = system_device() else {
        set_err(err, "MTLCreateSystemDefaultDevice returned nil");
        return Status::execute("metal_compute_device_unavailable");
    };

    let queue = thread_queue(device);
    let command_buffer = queue.new_command_buffer().to_owned();
    let encoder = command_buffer.compute_command_encoder_with_dispatch_type(metal_dispatch_type);

    let retain = match compute_encode_on_encoder(
        device,
        encoder,
        mtlb,
        buffers,
        images,
        sampled,
        samplers,
        threadgroup_memory,
        stage_in_region,
        stage_in_region_indirect,
        imageblock_dimensions,
        stage_input,
        dispatch_kind,
        grid_x,
        grid_y,
        grid_z,
        tg_x,
        tg_y,
        tg_z,
        err,
    ) {
        Ok(r) => r,
        Err(st) => {
            encoder.end_encoding();
            return st;
        }
    };

    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
    if command_buffer.status() == MTLCommandBufferStatus::Error {
        let detail = command_buffer_error_description(&command_buffer);
        set_err(err, format!("Metal command buffer failed: {detail}"));
        return Status::execute("metal_compute_command_buffer_failed");
    }

    let rc = compute_writeback_from_mtl(buffers, &retain.buffers, images, &retain.images, err);
    let _ = retain.sampled;
    let _ = retain.indirect;
    rc
}

pub fn reflect_compute_textures_mtlb(
    mtlb: &[u8],
    usages: *mut ReimsVgpuComputeTextureUsage,
    usage_cap: usize,
    out_usage_count: *mut usize,
    err: ErrOut<'_>,
) -> Status {
    if out_usage_count.is_null() {
        set_err(err, "invalid compute texture reflection output");
        return Status::args("metal_compute_reflection_count_output_missing");
    }
    if usages.is_null() && usage_cap != 0 {
        set_err(err, "invalid compute texture reflection output");
        return Status::args("metal_compute_reflection_usage_output_missing")
            .field("usage_cap", usage_cap);
    }
    unsafe {
        *out_usage_count = 0;
    }
    if mtlb.is_empty() {
        set_err(err, "compute MTLB is empty");
        return Status::args("metal_compute_reflection_mtlb_empty");
    }

    let mtlb_hash = hash_bytes(mtlb);
    if let Some(cached) = reflect_lookup(mtlb_hash, mtlb.len()) {
        if cached.len() > usage_cap {
            set_err(err, "too many compute texture bindings");
            return Status::args("metal_compute_reflection_cached_capacity_exceeded")
                .field("count", cached.len())
                .field("capacity", usage_cap);
        }
        if !usages.is_null() && !cached.is_empty() {
            unsafe {
                ptr::copy_nonoverlapping(cached.as_ptr(), usages, cached.len());
            }
        }
        unsafe {
            *out_usage_count = cached.len();
        }
        clear_err(err);
        return Status::OK;
    }

    let Some(device) = system_device() else {
        set_err(err, "MTLCreateSystemDefaultDevice returned nil");
        return Status::execute("metal_compute_reflection_device_unavailable");
    };
    let function = match load_only_function(device, mtlb, "compute", err) {
        Ok(f) => f,
        Err(st) => return st,
    };

    // MTLPipelineOptionArgumentInfo == BindingInfo == 1
    let (pso, reflection) = match new_compute_pso_with_function_reflection(device, &function, 1) {
        Ok(v) => v,
        Err(e) => {
            crate::observe::Emit::decline("metal_compute_reflection_pso", &e)
                .field("mtlb_hash", format!("{mtlb_hash:#x}"))
                .fail_once(mtlb_hash);
            set_err(err, format!("compute reflection PSO failed: {e}"));
            return Status::execute("metal_compute_reflection_pso_create_failed")
                .field("mtlb_hash", mtlb_hash);
        }
    };
    if reflection.is_null() {
        set_err(err, "compute pipeline reflection unavailable");
        return Status::execute("metal_compute_reflection_unavailable")
            .field("mtlb_hash", mtlb_hash);
    }

    let bindings = reflection_bindings(reflection);
    // Drop reflection retain.
    unsafe {
        let _: () = msg_send_release(reflection);
    }

    let mut seen = [false; REIMS_VGPU_METAL_MAX_TEXTURES];
    let mut local: Vec<ReimsVgpuComputeTextureUsage> = Vec::new();
    for b in bindings {
        if !b.used || b.type_ != BINDING_TYPE_TEXTURE {
            continue;
        }
        let access = match b.access {
            BINDING_ACCESS_READ_ONLY => REIMS_VGPU_COMPUTE_TEXTURE_ACCESS_READ,
            BINDING_ACCESS_READ_WRITE => REIMS_VGPU_COMPUTE_TEXTURE_ACCESS_READ_WRITE,
            BINDING_ACCESS_WRITE_ONLY => REIMS_VGPU_COMPUTE_TEXTURE_ACCESS_WRITE,
            other => {
                set_err(err, format!("unsupported compute texture access {other}"));
                return Status::args("metal_compute_reflection_texture_access_unsupported")
                    .field("access", other);
            }
        };
        for elem in 0..b.array_length {
            let texture_index = b.index + elem;
            if texture_index < b.index || texture_index as usize >= REIMS_VGPU_METAL_MAX_TEXTURES {
                set_err(
                    err,
                    format!("compute texture index {texture_index} exceeds backend cap"),
                );
                return Status::args("metal_compute_reflection_texture_index_exceeded")
                    .field("index", texture_index)
                    .field("base", b.index)
                    .field("limit", REIMS_VGPU_METAL_MAX_TEXTURES);
            }
            if seen[texture_index as usize] {
                set_err(
                    err,
                    format!("duplicate compute texture binding {texture_index}"),
                );
                return Status::args("metal_compute_reflection_texture_binding_duplicate")
                    .field("index", texture_index);
            }
            if local.len() >= REIMS_VGPU_METAL_MAX_TEXTURES || local.len() >= usage_cap {
                set_err(err, "too many compute texture bindings");
                return Status::args("metal_compute_reflection_texture_capacity_exceeded")
                    .field("count", local.len())
                    .field("backend_limit", REIMS_VGPU_METAL_MAX_TEXTURES)
                    .field("caller_capacity", usage_cap);
            }
            seen[texture_index as usize] = true;
            local.push(ReimsVgpuComputeTextureUsage {
                binding: REIMS_VGPU_BINDING_TEXTURE_BASE + texture_index as u32,
                access,
            });
        }
    }

    reflect_insert(mtlb_hash, mtlb.len(), local.clone());
    if !usages.is_null() && !local.is_empty() {
        unsafe {
            ptr::copy_nonoverlapping(local.as_ptr(), usages, local.len());
        }
    }
    unsafe {
        *out_usage_count = local.len();
    }
    clear_err(err);
    let _ = pso;
    Status::OK
}

unsafe fn msg_send_release(obj: *mut objc::runtime::Object) {
    use objc::{msg_send, sel, sel_impl};
    let _: () = msg_send![obj, release];
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observe::Emit;

    fn buffer(backing: &mut [u8]) -> ReimsVgpuBuffer {
        ReimsVgpuBuffer {
            binding: 7,
            data: std::ptr::null_mut(),
            len: 0,
            attribute_stride: 0,
            has_attribute_stride: 0,
            reserved0: 0,
            backing_data: backing.as_mut_ptr(),
            backing_len: backing.len(),
            backing_offset: 0,
        }
    }

    fn backing_refusal(buffer: &ReimsVgpuBuffer) -> String {
        let status = match compute_buffer_backing(buffer) {
            Ok(_) => panic!("invalid compute backing unexpectedly succeeded"),
            Err(status) => status,
        };
        Emit::refusal("metal_compute_test", &status)
            .expect("invalid compute backing must carry a refusal")
            .render()
    }

    #[test]
    fn compute_backing_rejections_preserve_the_failed_bound() {
        let mut empty = Vec::<u8>::with_capacity(1);
        let zero_len = buffer(empty.as_mut_slice());
        assert_eq!(
            backing_refusal(&zero_len),
            "metal_compute_test reason=metal_compute_backing_length_zero class=args binding=7"
        );

        let mut storage = vec![0u8; 8];
        let mut offset = buffer(&mut storage);
        offset.backing_offset = 9;
        assert_eq!(
            backing_refusal(&offset),
            "metal_compute_test reason=metal_compute_backing_offset_out_of_range class=args binding=7 offset=9 backing_len=8"
        );

        let mut span = buffer(&mut storage);
        span.backing_offset = 4;
        span.len = 5;
        assert_eq!(
            backing_refusal(&span),
            "metal_compute_test reason=metal_compute_backing_span_out_of_range class=args binding=7 len=5 offset=4 backing_len=8"
        );
    }
}
