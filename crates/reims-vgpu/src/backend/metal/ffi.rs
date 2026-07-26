//! C ABI exports (`reims_vgpu_backend_*`) with catch_unwind.

use crate::backend::metal::abi::*;
use crate::backend::metal::cache::{cache_stats, cache_stats_reset};
use crate::backend::metal::compute::{compute_core, reflect_compute_textures_mtlb};
use crate::backend::metal::error::write_err;
use crate::backend::metal::render::render_core;
use crate::backend::metal::runtime::{begin_native_color_format, end_native_color_format};
use crate::backend::metal::util::{ErrOut, Status};
use crate::observe::Emit;
use std::os::raw::c_char;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::slice;

fn err_pair(err: *mut c_char, err_cap: usize) -> ErrOut<'static> {
    // Lifetime is caller-bound; treat as static for internal helpers.
    (err, err_cap)
}

fn run(
    entry: &'static str,
    err: *mut c_char,
    err_cap: usize,
    f: impl FnOnce(ErrOut<'_>) -> Status + std::panic::UnwindSafe,
) -> i32 {
    match catch_unwind(AssertUnwindSafe(|| f(err_pair(err, err_cap)))) {
        Ok(st) => {
            if let Some(emit) = Emit::refusal("metal_ffi", &st) {
                emit.field("entry", entry)
                    .fail_once(entry_discriminant(entry));
            }
            st.code()
        }
        Err(_) => {
            write_err(err, err_cap, "reims-vgpu-metal: panic in C ABI entry");
            let status = Status::execute("metal_ffi_status_entry_panicked");
            Emit::refusal("metal_ffi", &status)
                .expect("panic status is a refusal")
                .field("entry", entry)
                .fail_once(entry_discriminant(entry));
            REIMS_VGPU_ERR_EXECUTE
        }
    }
}

fn run_void(entry: &'static str, f: impl FnOnce() + std::panic::UnwindSafe) {
    if catch_unwind(AssertUnwindSafe(f)).is_err() {
        let status = Status::execute("metal_ffi_void_entry_panicked");
        Emit::refusal("metal_ffi", &status)
            .expect("panic status is a refusal")
            .field("entry", entry)
            .fail_once(entry_discriminant(entry));
    }
}

fn entry_discriminant(entry: &str) -> u64 {
    entry.bytes().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ byte as u64).wrapping_mul(0x100_0000_01b3)
    })
}

fn slice_opt<'a, T>(
    ptr: *const T,
    count: usize,
    argument: &'static str,
) -> Result<&'a [T], Status> {
    if count == 0 {
        return Ok(&[]);
    }
    if ptr.is_null() {
        return Err(Status::args("metal_ffi_slice_pointer_null")
            .field("argument", argument)
            .field("count", count));
    }
    Ok(unsafe { slice::from_raw_parts(ptr, count) })
}

fn slice_opt_mut<'a, T>(
    ptr: *mut T,
    count: usize,
    argument: &'static str,
) -> Result<&'a mut [T], Status> {
    if count == 0 {
        return Ok(&mut []);
    }
    if ptr.is_null() {
        return Err(Status::args("metal_ffi_slice_pointer_null")
            .field("argument", argument)
            .field("count", count));
    }
    Ok(unsafe { slice::from_raw_parts_mut(ptr, count) })
}

macro_rules! checked_slice {
    ($ptr:expr, $count:expr, $argument:literal) => {
        match slice_opt($ptr, $count, $argument) {
            Ok(slice) => slice,
            Err(status) => return status,
        }
    };
}

macro_rules! checked_slice_mut {
    ($ptr:expr, $count:expr, $argument:literal) => {
        match slice_opt_mut($ptr, $count, $argument) {
            Ok(slice) => slice,
            Err(status) => return status,
        }
    };
}

// --- cache / color format ---

#[no_mangle]
pub extern "C" fn reims_vgpu_backend_begin_native_color_format(pixel_format: u32) {
    run_void("begin_native_color_format", || {
        begin_native_color_format(pixel_format)
    });
}

#[no_mangle]
pub extern "C" fn reims_vgpu_backend_end_native_color_format() {
    run_void("end_native_color_format", end_native_color_format);
}

#[no_mangle]
pub extern "C" fn reims_vgpu_backend_metal_cache_stats(out: *mut ReimsVgpuMetalCacheStats) {
    if out.is_null() {
        let status = Status::args("metal_ffi_cache_stats_output_null");
        Emit::refusal("metal_ffi", &status)
            .expect("null output is a refusal")
            .field("entry", "metal_cache_stats")
            .fail_once(entry_discriminant("metal_cache_stats"));
        return;
    }
    run_void("metal_cache_stats", || unsafe {
        *out = cache_stats();
    });
}

#[no_mangle]
pub extern "C" fn reims_vgpu_backend_metal_cache_stats_reset() {
    run_void("metal_cache_stats_reset", cache_stats_reset);
}

// --- compute MTLB ---

#[no_mangle]
pub extern "C" fn reims_vgpu_backend_dispatch_compute_mtlb(
    mtlb: *const u8,
    mtlb_len: usize,
    buffers: *mut ReimsVgpuBuffer,
    buffer_count: usize,
    grid_x: u32,
    err: *mut c_char,
    err_cap: usize,
) -> i32 {
    run("dispatch_compute_mtlb", err, err_cap, |e| {
        let mtlb = checked_slice!(mtlb, mtlb_len, "mtlb");
        let buffers = checked_slice_mut!(buffers, buffer_count, "buffers");
        compute_core(
            mtlb,
            buffers,
            &mut [],
            &[],
            &[],
            &[],
            None,
            None,
            None,
            None,
            REIMS_VGPU_COMPUTE_DISPATCH_KIND_THREADGROUPS,
            REIMS_VGPU_MTL_DISPATCH_TYPE_SERIAL,
            grid_x,
            1,
            1,
            64,
            1,
            1,
            e,
        )
    })
}

#[no_mangle]
pub extern "C" fn reims_vgpu_backend_dispatch_compute_texture_mtlb(
    mtlb: *const u8,
    mtlb_len: usize,
    buffers: *mut ReimsVgpuBuffer,
    buffer_count: usize,
    images: *mut ReimsVgpuStorageImage,
    image_count: usize,
    grid_x: u32,
    grid_y: u32,
    grid_z: u32,
    tg_x: u32,
    tg_y: u32,
    tg_z: u32,
    err: *mut c_char,
    err_cap: usize,
) -> i32 {
    run("dispatch_compute_texture_mtlb", err, err_cap, |e| {
        let mtlb = checked_slice!(mtlb, mtlb_len, "mtlb");
        let buffers = checked_slice_mut!(buffers, buffer_count, "buffers");
        let images = checked_slice_mut!(images, image_count, "images");
        compute_core(
            mtlb,
            buffers,
            images,
            &[],
            &[],
            &[],
            None,
            None,
            None,
            None,
            REIMS_VGPU_COMPUTE_DISPATCH_KIND_THREADGROUPS,
            REIMS_VGPU_MTL_DISPATCH_TYPE_SERIAL,
            grid_x,
            grid_y,
            grid_z,
            tg_x,
            tg_y,
            tg_z,
            e,
        )
    })
}

#[no_mangle]
pub extern "C" fn reims_vgpu_backend_dispatch_compute_texture_sampled_mtlb(
    mtlb: *const u8,
    mtlb_len: usize,
    buffers: *mut ReimsVgpuBuffer,
    buffer_count: usize,
    images: *mut ReimsVgpuStorageImage,
    image_count: usize,
    sampled: *const ReimsVgpuComputeSampledImage,
    sampled_count: usize,
    samplers: *const ReimsVgpuSampler,
    sampler_count: usize,
    grid_x: u32,
    grid_y: u32,
    grid_z: u32,
    tg_x: u32,
    tg_y: u32,
    tg_z: u32,
    err: *mut c_char,
    err_cap: usize,
) -> i32 {
    run("dispatch_compute_sampled_mtlb", err, err_cap, |e| {
        let mtlb = checked_slice!(mtlb, mtlb_len, "mtlb");
        let buffers = checked_slice_mut!(buffers, buffer_count, "buffers");
        let images = checked_slice_mut!(images, image_count, "images");
        let sampled = checked_slice!(sampled, sampled_count, "sampled");
        let samplers = checked_slice!(samplers, sampler_count, "samplers");
        compute_core(
            mtlb,
            buffers,
            images,
            sampled,
            samplers,
            &[],
            None,
            None,
            None,
            None,
            REIMS_VGPU_COMPUTE_DISPATCH_KIND_THREADGROUPS,
            REIMS_VGPU_MTL_DISPATCH_TYPE_SERIAL,
            grid_x,
            grid_y,
            grid_z,
            tg_x,
            tg_y,
            tg_z,
            e,
        )
    })
}

#[no_mangle]
pub extern "C" fn reims_vgpu_backend_dispatch_compute_texture_sampled_threadgroup_mtlb(
    mtlb: *const u8,
    mtlb_len: usize,
    buffers: *mut ReimsVgpuBuffer,
    buffer_count: usize,
    images: *mut ReimsVgpuStorageImage,
    image_count: usize,
    sampled: *const ReimsVgpuComputeSampledImage,
    sampled_count: usize,
    samplers: *const ReimsVgpuSampler,
    sampler_count: usize,
    threadgroup_memory: *const ReimsVgpuThreadgroupMemory,
    threadgroup_memory_count: usize,
    stage_in_region: *const ReimsVgpuComputeStageInRegion,
    stage_in_region_indirect: *const ReimsVgpuComputeStageInRegionIndirectArguments,
    imageblock_dimensions: *const ReimsVgpuComputeImageblockDimensions,
    dispatch_kind: u32,
    dispatch_type: u32,
    grid_x: u32,
    grid_y: u32,
    grid_z: u32,
    tg_x: u32,
    tg_y: u32,
    tg_z: u32,
    stage_input: *const ReimsVgpuComputeStageInputDescriptor,
    err: *mut c_char,
    err_cap: usize,
) -> i32 {
    run(
        "dispatch_compute_sampled_threadgroup_mtlb",
        err,
        err_cap,
        |e| {
            let mtlb = checked_slice!(mtlb, mtlb_len, "mtlb");
            let buffers = checked_slice_mut!(buffers, buffer_count, "buffers");
            let images = checked_slice_mut!(images, image_count, "images");
            let sampled = checked_slice!(sampled, sampled_count, "sampled");
            let samplers = checked_slice!(samplers, sampler_count, "samplers");
            let tg = checked_slice!(
                threadgroup_memory,
                threadgroup_memory_count,
                "threadgroup_memory"
            );
            let stage_in = if stage_in_region.is_null() {
                None
            } else {
                Some(unsafe { &*stage_in_region })
            };
            let stage_in_ind = if stage_in_region_indirect.is_null() {
                None
            } else {
                Some(unsafe { &*stage_in_region_indirect })
            };
            let imageblock = if imageblock_dimensions.is_null() {
                None
            } else {
                Some(unsafe { &*imageblock_dimensions })
            };
            let stage_input = if stage_input.is_null() {
                None
            } else {
                Some(unsafe { &*stage_input })
            };
            compute_core(
                mtlb,
                buffers,
                images,
                sampled,
                samplers,
                tg,
                stage_in,
                stage_in_ind,
                imageblock,
                stage_input,
                dispatch_kind,
                dispatch_type,
                grid_x,
                grid_y,
                grid_z,
                tg_x,
                tg_y,
                tg_z,
                e,
            )
        },
    )
}

#[no_mangle]
pub extern "C" fn reims_vgpu_backend_reflect_compute_textures_mtlb(
    mtlb: *const u8,
    mtlb_len: usize,
    usages: *mut ReimsVgpuComputeTextureUsage,
    usage_cap: usize,
    out_usage_count: *mut usize,
    err: *mut c_char,
    err_cap: usize,
) -> i32 {
    run("reflect_compute_textures_mtlb", err, err_cap, |e| {
        let mtlb = checked_slice!(mtlb, mtlb_len, "mtlb");
        reflect_compute_textures_mtlb(mtlb, usages, usage_cap, out_usage_count, e)
    })
}

// --- render wrappers ---

macro_rules! render_with_state_body {
    ($e:expr, $vert:expr, $frag:expr, $w:expr, $h:expr, $vc:expr, $fv:expr, $ic:expr, $bi:expr,
     $pt:expr, $pi:expr, $idx:expr, $attrs:expr, $attr_count:expr,
     $bufs:expr, $bc:expr, $fbufs:expr, $fbc:expr,
     $vimg:expr, $vic:expr, $vsmp:expr, $vsc:expr,
     $img:expr, $imc:expr, $smp:expr, $sc:expr,
     $vp:expr, $vpc:expr, $scis:expr, $scc:expr,
     $raster:expr, $db:expr, $ds:expr, $sr:expr, $da:expr, $sa:expr,
     $blend:expr, $tr:expr, $trl:expr, $out:expr, $outc:expr) => {{
        let vert = checked_slice!($vert.0, $vert.1, "vertex_mtlb");
        let frag = checked_slice!($frag.0, $frag.1, "fragment_mtlb");
        let attrs = checked_slice!($attrs, $attr_count, "attributes");
        let buffers = checked_slice!($bufs, $bc, "vertex_buffers");
        let frag_buffers = checked_slice!($fbufs, $fbc, "fragment_buffers");
        let vertex_images = checked_slice!($vimg, $vic, "vertex_images");
        let vertex_samplers = checked_slice!($vsmp, $vsc, "vertex_samplers");
        let images = checked_slice!($img, $imc, "fragment_images");
        let samplers = checked_slice!($smp, $sc, "fragment_samplers");
        let viewports = checked_slice!($vp, $vpc, "viewports");
        let scissors = checked_slice!($scis, $scc, "scissors");
        let raster = if $raster.is_null() {
            None
        } else {
            Some(unsafe { &*$raster })
        };
        let depth_bias = if $db.is_null() {
            None
        } else {
            Some(unsafe { &*$db })
        };
        let depth_stencil = if $ds.is_null() {
            None
        } else {
            Some(unsafe { &*$ds })
        };
        let stencil_reference = if $sr.is_null() {
            None
        } else {
            Some(unsafe { &*$sr })
        };
        let mut depth_attachment = if $da.is_null() {
            None
        } else {
            Some(unsafe { &mut *$da })
        };
        let mut stencil_attachment = if $sa.is_null() {
            None
        } else {
            Some(unsafe { &mut *$sa })
        };
        let blend = if $blend.is_null() {
            None
        } else {
            Some(unsafe { &*$blend })
        };
        let target = if $trl == 0 {
            None
        } else {
            Some(checked_slice!($tr, $trl, "target_rgba8"))
        };
        let out_rgba = if $outc == 0 {
            None
        } else {
            Some(checked_slice_mut!($out, $outc, "out_rgba8"))
        };
        // Cast through usize so null-literal call sites are not statically
        // proven non-null-only at the dereference (deny(deref_nullptr)).
        let pi_ptr = $pi as usize as *const ReimsVgpuPrimitiveIndirectDraw;
        let idx_ptr = $idx as usize as *const ReimsVgpuIndexedDraw;
        let primitive_indirect = if pi_ptr.is_null() {
            None
        } else {
            // SAFETY: non-null caller pointer for the duration of this call.
            Some(unsafe { &*pi_ptr })
        };
        let indexed = if idx_ptr.is_null() {
            None
        } else {
            // SAFETY: non-null caller pointer for the duration of this call.
            Some(unsafe { &*idx_ptr })
        };
        render_core(
            vert,
            frag,
            $w,
            $h,
            $vc,
            $fv,
            $ic,
            $bi,
            $pt,
            primitive_indirect,
            indexed,
            attrs,
            buffers,
            frag_buffers,
            vertex_images,
            vertex_samplers,
            images,
            samplers,
            viewports,
            scissors,
            raster,
            depth_bias,
            depth_stencil,
            stencil_reference,
            depth_attachment.as_deref_mut(),
            stencil_attachment.as_deref_mut(),
            blend,
            0,
            target,
            out_rgba,
            $e,
        )
    }};
}

#[no_mangle]
pub extern "C" fn reims_vgpu_backend_render_with_state(
    vert_mtlb: *const u8,
    vert_len: usize,
    frag_mtlb: *const u8,
    frag_len: usize,
    width: u32,
    height: u32,
    vertex_count: usize,
    first_vertex: usize,
    instance_count: usize,
    base_instance: usize,
    primitive_type: u32,
    primitive_indirect: *const ReimsVgpuPrimitiveIndirectDraw,
    buffers: *const ReimsVgpuBuffer,
    buffer_count: usize,
    frag_buffers: *const ReimsVgpuBuffer,
    frag_buffer_count: usize,
    vertex_images: *const ReimsVgpuSampledImage,
    vertex_image_count: usize,
    vertex_samplers: *const ReimsVgpuSampler,
    vertex_sampler_count: usize,
    viewports: *const ReimsVgpuViewport,
    viewport_count: usize,
    scissors: *const ReimsVgpuScissor,
    scissor_count: usize,
    raster: *const ReimsVgpuRasterState,
    depth_bias: *const ReimsVgpuDepthBiasState,
    depth_stencil: *const ReimsVgpuDepthStencilState,
    stencil_reference: *const ReimsVgpuStencilReferenceState,
    depth_attachment: *mut ReimsVgpuDepthAttachment,
    stencil_attachment: *mut ReimsVgpuStencilAttachment,
    blend: *const ReimsVgpuBlendState,
    target_rgba8: *const u8,
    target_rgba8_len: usize,
    out_rgba: *mut u8,
    out_cap: usize,
    err: *mut c_char,
    err_cap: usize,
) -> i32 {
    run("render_with_state", err, err_cap, |e| {
        render_with_state_body!(
            e,
            (vert_mtlb, vert_len),
            (frag_mtlb, frag_len),
            width,
            height,
            vertex_count,
            first_vertex,
            instance_count,
            base_instance,
            primitive_type,
            primitive_indirect,
            std::ptr::null::<ReimsVgpuIndexedDraw>(),
            std::ptr::null::<ReimsVgpuVertexAttr>(),
            0usize,
            buffers,
            buffer_count,
            frag_buffers,
            frag_buffer_count,
            vertex_images,
            vertex_image_count,
            vertex_samplers,
            vertex_sampler_count,
            std::ptr::null::<ReimsVgpuSampledImage>(),
            0usize,
            std::ptr::null::<ReimsVgpuSampler>(),
            0usize,
            viewports,
            viewport_count,
            scissors,
            scissor_count,
            raster,
            depth_bias,
            depth_stencil,
            stencil_reference,
            depth_attachment,
            stencil_attachment,
            blend,
            target_rgba8,
            target_rgba8_len,
            out_rgba,
            out_cap
        )
    })
}

#[no_mangle]
pub extern "C" fn reims_vgpu_backend_render(
    vert_mtlb: *const u8,
    vert_len: usize,
    frag_mtlb: *const u8,
    frag_len: usize,
    width: u32,
    height: u32,
    vertex_count: u32,
    first_vertex: u32,
    buffers: *const ReimsVgpuBuffer,
    buffer_count: usize,
    frag_buffers: *const ReimsVgpuBuffer,
    frag_buffer_count: usize,
    blend: *const ReimsVgpuBlendState,
    target_rgba8: *const u8,
    target_rgba8_len: usize,
    out_rgba: *mut u8,
    out_cap: usize,
    err: *mut c_char,
    err_cap: usize,
) -> i32 {
    reims_vgpu_backend_render_with_state(
        vert_mtlb,
        vert_len,
        frag_mtlb,
        frag_len,
        width,
        height,
        vertex_count as usize,
        first_vertex as usize,
        1,
        0,
        REIMS_VGPU_MTL_PRIMITIVE_TYPE_TRIANGLE,
        std::ptr::null(),
        buffers,
        buffer_count,
        frag_buffers,
        frag_buffer_count,
        std::ptr::null(),
        0,
        std::ptr::null(),
        0,
        std::ptr::null(),
        0,
        std::ptr::null(),
        0,
        std::ptr::null(),
        std::ptr::null(),
        std::ptr::null(),
        std::ptr::null(),
        std::ptr::null_mut(),
        std::ptr::null_mut(),
        blend,
        target_rgba8,
        target_rgba8_len,
        out_rgba,
        out_cap,
        err,
        err_cap,
    )
}

#[no_mangle]
pub extern "C" fn reims_vgpu_backend_render_stagein_with_state(
    vert_mtlb: *const u8,
    vert_len: usize,
    frag_mtlb: *const u8,
    frag_len: usize,
    width: u32,
    height: u32,
    vertex_count: usize,
    first_vertex: usize,
    instance_count: usize,
    base_instance: usize,
    primitive_type: u32,
    primitive_indirect: *const ReimsVgpuPrimitiveIndirectDraw,
    indexed: *const ReimsVgpuIndexedDraw,
    attrs: *const ReimsVgpuVertexAttr,
    attr_count: usize,
    buffers: *const ReimsVgpuBuffer,
    buffer_count: usize,
    frag_buffers: *const ReimsVgpuBuffer,
    frag_buffer_count: usize,
    vertex_images: *const ReimsVgpuSampledImage,
    vertex_image_count: usize,
    vertex_samplers: *const ReimsVgpuSampler,
    vertex_sampler_count: usize,
    viewports: *const ReimsVgpuViewport,
    viewport_count: usize,
    scissors: *const ReimsVgpuScissor,
    scissor_count: usize,
    raster: *const ReimsVgpuRasterState,
    depth_bias: *const ReimsVgpuDepthBiasState,
    depth_stencil: *const ReimsVgpuDepthStencilState,
    stencil_reference: *const ReimsVgpuStencilReferenceState,
    depth_attachment: *mut ReimsVgpuDepthAttachment,
    stencil_attachment: *mut ReimsVgpuStencilAttachment,
    blend: *const ReimsVgpuBlendState,
    target_rgba8: *const u8,
    target_rgba8_len: usize,
    out_rgba: *mut u8,
    out_cap: usize,
    err: *mut c_char,
    err_cap: usize,
) -> i32 {
    run("render_stagein_with_state", err, err_cap, |e| {
        render_with_state_body!(
            e,
            (vert_mtlb, vert_len),
            (frag_mtlb, frag_len),
            width,
            height,
            vertex_count,
            first_vertex,
            instance_count,
            base_instance,
            primitive_type,
            primitive_indirect,
            indexed,
            attrs,
            attr_count,
            buffers,
            buffer_count,
            frag_buffers,
            frag_buffer_count,
            vertex_images,
            vertex_image_count,
            vertex_samplers,
            vertex_sampler_count,
            std::ptr::null::<ReimsVgpuSampledImage>(),
            0usize,
            std::ptr::null::<ReimsVgpuSampler>(),
            0usize,
            viewports,
            viewport_count,
            scissors,
            scissor_count,
            raster,
            depth_bias,
            depth_stencil,
            stencil_reference,
            depth_attachment,
            stencil_attachment,
            blend,
            target_rgba8,
            target_rgba8_len,
            out_rgba,
            out_cap
        )
    })
}

#[no_mangle]
pub extern "C" fn reims_vgpu_backend_render_stagein(
    vert_mtlb: *const u8,
    vert_len: usize,
    frag_mtlb: *const u8,
    frag_len: usize,
    width: u32,
    height: u32,
    vertex_count: u32,
    first_vertex: u32,
    indexed: *const ReimsVgpuIndexedDraw,
    attrs: *const ReimsVgpuVertexAttr,
    attr_count: usize,
    buffers: *const ReimsVgpuBuffer,
    buffer_count: usize,
    frag_buffers: *const ReimsVgpuBuffer,
    frag_buffer_count: usize,
    blend: *const ReimsVgpuBlendState,
    target_rgba8: *const u8,
    target_rgba8_len: usize,
    out_rgba: *mut u8,
    out_cap: usize,
    err: *mut c_char,
    err_cap: usize,
) -> i32 {
    reims_vgpu_backend_render_stagein_with_state(
        vert_mtlb,
        vert_len,
        frag_mtlb,
        frag_len,
        width,
        height,
        vertex_count as usize,
        first_vertex as usize,
        1,
        0,
        REIMS_VGPU_MTL_PRIMITIVE_TYPE_TRIANGLE,
        std::ptr::null(),
        indexed,
        attrs,
        attr_count,
        buffers,
        buffer_count,
        frag_buffers,
        frag_buffer_count,
        std::ptr::null(),
        0,
        std::ptr::null(),
        0,
        std::ptr::null(),
        0,
        std::ptr::null(),
        0,
        std::ptr::null(),
        std::ptr::null(),
        std::ptr::null(),
        std::ptr::null(),
        std::ptr::null_mut(),
        std::ptr::null_mut(),
        blend,
        target_rgba8,
        target_rgba8_len,
        out_rgba,
        out_cap,
        err,
        err_cap,
    )
}

#[no_mangle]
pub extern "C" fn reims_vgpu_backend_render_textured_with_state(
    vert_mtlb: *const u8,
    vert_len: usize,
    frag_mtlb: *const u8,
    frag_len: usize,
    width: u32,
    height: u32,
    vertex_count: usize,
    first_vertex: usize,
    instance_count: usize,
    base_instance: usize,
    primitive_type: u32,
    primitive_indirect: *const ReimsVgpuPrimitiveIndirectDraw,
    buffers: *const ReimsVgpuBuffer,
    buffer_count: usize,
    frag_buffers: *const ReimsVgpuBuffer,
    frag_buffer_count: usize,
    vertex_images: *const ReimsVgpuSampledImage,
    vertex_image_count: usize,
    vertex_samplers: *const ReimsVgpuSampler,
    vertex_sampler_count: usize,
    images: *const ReimsVgpuSampledImage,
    image_count: usize,
    samplers: *const ReimsVgpuSampler,
    sampler_count: usize,
    viewports: *const ReimsVgpuViewport,
    viewport_count: usize,
    scissors: *const ReimsVgpuScissor,
    scissor_count: usize,
    raster: *const ReimsVgpuRasterState,
    depth_bias: *const ReimsVgpuDepthBiasState,
    depth_stencil: *const ReimsVgpuDepthStencilState,
    stencil_reference: *const ReimsVgpuStencilReferenceState,
    depth_attachment: *mut ReimsVgpuDepthAttachment,
    stencil_attachment: *mut ReimsVgpuStencilAttachment,
    blend: *const ReimsVgpuBlendState,
    target_rgba8: *const u8,
    target_rgba8_len: usize,
    out_rgba: *mut u8,
    out_cap: usize,
    err: *mut c_char,
    err_cap: usize,
) -> i32 {
    run("render_textured_with_state", err, err_cap, |e| {
        render_with_state_body!(
            e,
            (vert_mtlb, vert_len),
            (frag_mtlb, frag_len),
            width,
            height,
            vertex_count,
            first_vertex,
            instance_count,
            base_instance,
            primitive_type,
            primitive_indirect,
            std::ptr::null::<ReimsVgpuIndexedDraw>(),
            std::ptr::null::<ReimsVgpuVertexAttr>(),
            0usize,
            buffers,
            buffer_count,
            frag_buffers,
            frag_buffer_count,
            vertex_images,
            vertex_image_count,
            vertex_samplers,
            vertex_sampler_count,
            images,
            image_count,
            samplers,
            sampler_count,
            viewports,
            viewport_count,
            scissors,
            scissor_count,
            raster,
            depth_bias,
            depth_stencil,
            stencil_reference,
            depth_attachment,
            stencil_attachment,
            blend,
            target_rgba8,
            target_rgba8_len,
            out_rgba,
            out_cap
        )
    })
}

#[no_mangle]
pub extern "C" fn reims_vgpu_backend_render_textured(
    vert_mtlb: *const u8,
    vert_len: usize,
    frag_mtlb: *const u8,
    frag_len: usize,
    width: u32,
    height: u32,
    vertex_count: u32,
    first_vertex: u32,
    buffers: *const ReimsVgpuBuffer,
    buffer_count: usize,
    frag_buffers: *const ReimsVgpuBuffer,
    frag_buffer_count: usize,
    images: *const ReimsVgpuSampledImage,
    image_count: usize,
    samplers: *const ReimsVgpuSampler,
    sampler_count: usize,
    blend: *const ReimsVgpuBlendState,
    target_rgba8: *const u8,
    target_rgba8_len: usize,
    out_rgba: *mut u8,
    out_cap: usize,
    err: *mut c_char,
    err_cap: usize,
) -> i32 {
    reims_vgpu_backend_render_textured_with_state(
        vert_mtlb,
        vert_len,
        frag_mtlb,
        frag_len,
        width,
        height,
        vertex_count as usize,
        first_vertex as usize,
        1,
        0,
        REIMS_VGPU_MTL_PRIMITIVE_TYPE_TRIANGLE,
        std::ptr::null(),
        buffers,
        buffer_count,
        frag_buffers,
        frag_buffer_count,
        std::ptr::null(),
        0,
        std::ptr::null(),
        0,
        images,
        image_count,
        samplers,
        sampler_count,
        std::ptr::null(),
        0,
        std::ptr::null(),
        0,
        std::ptr::null(),
        std::ptr::null(),
        std::ptr::null(),
        std::ptr::null(),
        std::ptr::null_mut(),
        std::ptr::null_mut(),
        blend,
        target_rgba8,
        target_rgba8_len,
        out_rgba,
        out_cap,
        err,
        err_cap,
    )
}

#[no_mangle]
pub extern "C" fn reims_vgpu_backend_render_stagein_textured_with_state(
    vert_mtlb: *const u8,
    vert_len: usize,
    frag_mtlb: *const u8,
    frag_len: usize,
    width: u32,
    height: u32,
    vertex_count: usize,
    first_vertex: usize,
    instance_count: usize,
    base_instance: usize,
    primitive_type: u32,
    primitive_indirect: *const ReimsVgpuPrimitiveIndirectDraw,
    indexed: *const ReimsVgpuIndexedDraw,
    attrs: *const ReimsVgpuVertexAttr,
    attr_count: usize,
    buffers: *const ReimsVgpuBuffer,
    buffer_count: usize,
    frag_buffers: *const ReimsVgpuBuffer,
    frag_buffer_count: usize,
    vertex_images: *const ReimsVgpuSampledImage,
    vertex_image_count: usize,
    vertex_samplers: *const ReimsVgpuSampler,
    vertex_sampler_count: usize,
    images: *const ReimsVgpuSampledImage,
    image_count: usize,
    samplers: *const ReimsVgpuSampler,
    sampler_count: usize,
    viewports: *const ReimsVgpuViewport,
    viewport_count: usize,
    scissors: *const ReimsVgpuScissor,
    scissor_count: usize,
    raster: *const ReimsVgpuRasterState,
    depth_bias: *const ReimsVgpuDepthBiasState,
    depth_stencil: *const ReimsVgpuDepthStencilState,
    stencil_reference: *const ReimsVgpuStencilReferenceState,
    depth_attachment: *mut ReimsVgpuDepthAttachment,
    stencil_attachment: *mut ReimsVgpuStencilAttachment,
    blend: *const ReimsVgpuBlendState,
    target_rgba8: *const u8,
    target_rgba8_len: usize,
    out_rgba: *mut u8,
    out_cap: usize,
    err: *mut c_char,
    err_cap: usize,
) -> i32 {
    run("render_stagein_textured_with_state", err, err_cap, |e| {
        render_with_state_body!(
            e,
            (vert_mtlb, vert_len),
            (frag_mtlb, frag_len),
            width,
            height,
            vertex_count,
            first_vertex,
            instance_count,
            base_instance,
            primitive_type,
            primitive_indirect,
            indexed,
            attrs,
            attr_count,
            buffers,
            buffer_count,
            frag_buffers,
            frag_buffer_count,
            vertex_images,
            vertex_image_count,
            vertex_samplers,
            vertex_sampler_count,
            images,
            image_count,
            samplers,
            sampler_count,
            viewports,
            viewport_count,
            scissors,
            scissor_count,
            raster,
            depth_bias,
            depth_stencil,
            stencil_reference,
            depth_attachment,
            stencil_attachment,
            blend,
            target_rgba8,
            target_rgba8_len,
            out_rgba,
            out_cap
        )
    })
}

#[no_mangle]
pub extern "C" fn reims_vgpu_backend_render_stagein_textured(
    vert_mtlb: *const u8,
    vert_len: usize,
    frag_mtlb: *const u8,
    frag_len: usize,
    width: u32,
    height: u32,
    vertex_count: u32,
    first_vertex: u32,
    indexed: *const ReimsVgpuIndexedDraw,
    attrs: *const ReimsVgpuVertexAttr,
    attr_count: usize,
    buffers: *const ReimsVgpuBuffer,
    buffer_count: usize,
    frag_buffers: *const ReimsVgpuBuffer,
    frag_buffer_count: usize,
    images: *const ReimsVgpuSampledImage,
    image_count: usize,
    samplers: *const ReimsVgpuSampler,
    sampler_count: usize,
    blend: *const ReimsVgpuBlendState,
    target_rgba8: *const u8,
    target_rgba8_len: usize,
    out_rgba: *mut u8,
    out_cap: usize,
    err: *mut c_char,
    err_cap: usize,
) -> i32 {
    reims_vgpu_backend_render_stagein_textured_with_state(
        vert_mtlb,
        vert_len,
        frag_mtlb,
        frag_len,
        width,
        height,
        vertex_count as usize,
        first_vertex as usize,
        1,
        0,
        REIMS_VGPU_MTL_PRIMITIVE_TYPE_TRIANGLE,
        std::ptr::null(),
        indexed,
        attrs,
        attr_count,
        buffers,
        buffer_count,
        frag_buffers,
        frag_buffer_count,
        std::ptr::null(),
        0,
        std::ptr::null(),
        0,
        images,
        image_count,
        samplers,
        sampler_count,
        std::ptr::null(),
        0,
        std::ptr::null(),
        0,
        std::ptr::null(),
        std::ptr::null(),
        std::ptr::null(),
        std::ptr::null(),
        std::ptr::null_mut(),
        std::ptr::null_mut(),
        blend,
        target_rgba8,
        target_rgba8_len,
        out_rgba,
        out_cap,
        err,
        err_cap,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nonempty_ffi_slice_requires_a_pointer_and_names_the_argument() {
        let status = slice_opt::<u8>(std::ptr::null(), 3, "vertex_mtlb")
            .expect_err("a nonempty null slice must be rejected");
        assert_eq!(
            Emit::refusal("metal_ffi_test", &status)
                .expect("invalid FFI slice must carry a refusal")
                .render(),
            "metal_ffi_test reason=metal_ffi_slice_pointer_null class=args argument=vertex_mtlb count=3"
        );
        assert!(slice_opt::<u8>(std::ptr::null(), 0, "vertex_mtlb")
            .expect("empty null slice is valid")
            .is_empty());
    }
}
