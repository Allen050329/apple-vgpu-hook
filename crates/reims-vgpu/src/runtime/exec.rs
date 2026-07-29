//! CmdExecIndirect2: load streams, multi-attachment clears, Metal draw attempt.
//!
//! Clear-only passes write guest mapping pages (archive render_clear).
//! Draws try Metal encode when pipeline MTLBs resolve; otherwise color targets
//! are still marked dirty for DisplaySwap.

use crate::contract::endian::{ld32, ld64};
use crate::contract::pixel_format::{f64_to_unorm8, MTL_FORMAT_BGRA8_UNORM, RGBA8_BPP};
use crate::model::DeviceState;
use crate::runtime::blit_exec::{self, BlitStatus};
use crate::runtime::compute_exec::{self, ComputeStatus};
use crate::runtime::decode::blit::{self, Kind as BlitKind, OP_GENERATE_MIPMAPS};
use crate::runtime::decode::compute::{self, Kind as ComputeKind};
use crate::runtime::decode::event as event_decode;
use crate::runtime::decode::fifo::{
    CHILD_EXEC_INDIRECT_CMDBUF_COUNT, CHILD_EXEC_INDIRECT_CMDBUF_DESC_LEN,
    CHILD_EXEC_INDIRECT_CMDBUF_GVA, CHILD_EXEC_INDIRECT_CMDBUF_LENGTH,
    CHILD_EXEC_INDIRECT_HEADER_LEN, CHILD_EXEC_INDIRECT_RESOURCE_COUNT,
    CHILD_EXEC_INDIRECT_RESOURCE_DESC_LEN, CHILD_EXEC_INDIRECT_TASK_ID,
};
use crate::runtime::decode::render::{
    self, decode_color_attachment, decode_depth_attachment, decode_stencil_attachment,
    ColorAttachment, DepthAttachment, Kind as RenderKind, Stage, StencilAttachment,
    OP_UPDATE_FENCE as RENDER_OP_UPDATE_FENCE, OP_WAIT_FENCE as RENDER_OP_WAIT_FENCE,
    PASS_LOAD_ACTION_CLEAR, PASS_LOAD_ACTION_LOAD, PASS_MAX_COLOR_ATTACHMENTS,
    PASS_STORE_ACTION_STORE,
};
use crate::runtime::decode::stream::{
    self, decode_first_record, decode_next_record, SEGMENT_TYPE_BLIT, SEGMENT_TYPE_COMPUTE,
    SEGMENT_TYPE_EVENT, SEGMENT_TYPE_INFO, SEGMENT_TYPE_RENDER,
};
use crate::runtime::fence_exec::{self, FenceStatus};
use crate::runtime::gva_mem;
use crate::runtime::host::{HostMemory, HostOps};
use crate::runtime::mapping_write;
use crate::runtime::metal_draw::{
    self, BufferBind, EncodeStatus, IndexedDrawInfo, SamplerBind, TextureBind, MAX_BIND_SLOTS,
};
use crate::runtime::mipmap::{self, MipmapStatus};
use crate::runtime::objects;
use crate::runtime::plan::event_sync::{Domain as FenceDomain, FenceAction};
use crate::runtime::task_slot::{resolve_task_word, TaskWordSite};

/// Max descriptors per ExecIndirect2 (wire table size), not a byte budget.
const MAX_CMDBUFS: usize = 16;

/// Pending render-pass ICB execute (range form or indirect range buffer).
#[derive(Clone, Debug, Default)]
struct RenderIcbExecute {
    icb_ref: u32,
    is_range: bool,
    range_location: u64,
    range_length: u64,
    args_buffer_ref: u32,
    args_buffer_offset: u64,
}

/// One draw recorded with the bind state at that point (archive DrawRec / multi-draw job).
///
/// Archive `apple_pv_gpu_render_worker_run` executes **every** draw in order,
/// seeding draw N from draw N-1's writeback. Product previously kept only
/// `last_draw`, which dropped the logo when the pill was the final draw in the
/// same stream (journal: logo RG8 168×206 + pill → one type-11 FB).
#[derive(Clone, Debug, Default)]
struct PendingDraw {
    pipeline_ref: u32,
    /// (count, instance, prim, first_vertex)
    draw: (u32, u32, u32, u32),
    indexed: Option<IndexedDrawInfo>,
    vertex_buffers: Vec<BufferBind>,
    fragment_buffers: Vec<BufferBind>,
    vertex_textures: Vec<TextureBind>,
    fragment_textures: Vec<TextureBind>,
    vertex_samplers: Vec<SamplerBind>,
    fragment_samplers: Vec<SamplerBind>,
    viewport: Option<[f64; 6]>,
    scissor: Option<(u32, u32, u32, u32)>,
    blend_color: Option<[f32; 4]>,
    cull_mode: Option<u32>,
    front_facing: Option<u32>,
    depth_bias: Option<[f32; 3]>,
    depth_stencil_ref: u32,
    stencil_ref: Option<(u32, u32)>,
    depth_attach: Option<DepthAttachment>,
    stencil_attach: Option<StencilAttachment>,
}

#[derive(Clone, Debug, Default)]
struct StreamAccum {
    pipeline_ref: u32,
    /// Pending clears for color attachments (load=clear).
    clears: Vec<ColorAttachment>,
    /// Color targets as (pass slot index, attachment). Slot maps to Metal color(i).
    color_slots: Vec<(u32, ColorAttachment)>,
    color_targets: Vec<u32>,
    /// All draws in stream order (archive multi-draw job).
    draws: Vec<PendingDraw>,
    saw_draw: bool,
    /// Last render ICB execute (`0x14`/`0x15`) in this stream.
    execute_icb: Option<RenderIcbExecute>,
    vertex_buffers: Vec<BufferBind>,
    fragment_buffers: Vec<BufferBind>,
    vertex_textures: Vec<TextureBind>,
    fragment_textures: Vec<TextureBind>,
    vertex_samplers: Vec<SamplerBind>,
    fragment_samplers: Vec<SamplerBind>,
    viewport: Option<[f64; 6]>,
    scissor: Option<(u32, u32, u32, u32)>,
    indexed: Option<IndexedDrawInfo>,
    blend_color: Option<[f32; 4]>,
    cull_mode: Option<u32>,
    front_facing: Option<u32>,
    depth_bias: Option<[f32; 3]>,
    depth_stencil_ref: u32,
    stencil_ref: Option<(u32, u32)>,
    depth_attach: Option<DepthAttachment>,
    stencil_attach: Option<StencilAttachment>,
}

/// Cap multi-draw records per stream (archive REIMS_VGPU_MAX_DRAWS_PER_JOB order).
const MAX_DRAWS_PER_STREAM: usize = 64;

#[derive(Clone, Debug, Default)]
pub struct ExecResult {
    pub task_id: u32,
    pub streams_loaded: u32,
    /// Immutable shader translation is still running off the FIFO scheduler.
    /// The caller must keep this packet at the channel head and retry it.
    pub deferred: bool,
    /// Info-segment `0x1d1` ICB backing associations applied.
    pub icb_backing_ok: u32,
    pub icb_backing_fail: u32,
    pub texture_refs: Vec<u32>,
    pub type11_mappings: Vec<u32>,
    pub color_targets: Vec<u32>,
    pub saw_draw: bool,
    pub clears_applied: u32,
    pub metal_draws_ok: u32,
    pub metal_draws_fail: u32,
    /// Render-pass attachment sets resolved from guest objects. One Metal
    /// render stream has one fixed attachment set regardless of draw count.
    pub render_attachment_resolves: u32,
    /// Guest-visible color attachment Stores issued at render-pass completion.
    /// Multi-draw records stay resident; one pass must not full-frame import
    /// the same attachment after every draw.
    pub render_guest_stores: u32,
    pub buffer_binds: u32,
    pub texture_binds: u32,
    /// Explicit nil entries in render bind ranges. These must remove prior
    /// slot state rather than silently retaining a stale resource.
    pub buffer_unbinds: u32,
    pub texture_unbinds: u32,
    pub sampler_unbinds: u32,
    pub mipmaps_ok: u32,
    pub mipmaps_fail: u32,
    pub blit_fills_ok: u32,
    pub blit_copies_ok: u32,
    pub blit_fail: u32,
    pub blit_fences_ok: u32,
    pub blit_fences_pending: u32,
    pub blit_fences_fail: u32,
    pub compute_fences_ok: u32,
    pub compute_fences_pending: u32,
    pub compute_fences_fail: u32,
    pub compute_dispatches_ok: u32,
    pub compute_dispatches_fail: u32,
    pub compute_buffer_binds: u32,
    pub compute_texture_binds: u32,
    pub compute_sampler_binds: u32,
    /// Control-flow SPI encode ok / fail (`0xdc`–`0xe2`).
    pub compute_control_ok: u32,
    pub compute_control_fail: u32,
    /// ICB materialize+execute ok / fail (`0xe4`/`0xe5`).
    pub compute_icb_ok: u32,
    pub compute_icb_fail: u32,
    /// Render ICB execute ok / fail (`0x14`/`0x15`).
    pub render_icb_ok: u32,
    pub render_icb_fail: u32,
    pub render_fences_ok: u32,
    pub render_fences_pending: u32,
    pub render_fences_fail: u32,
    pub event_ops_ok: u32,
    pub event_ops_pending: u32,
    pub event_ops_fail: u32,
    /// Wall-time census for the synchronous packet body. Render records only
    /// accumulate state; their backend work is charged to `finish_us`.
    pub load_us: u64,
    pub render_us: u64,
    pub blit_us: u64,
    pub compute_us: u64,
    pub event_us: u64,
    pub info_us: u64,
    pub finish_us: u64,
    pub total_us: u64,
}

fn tally_fence(st: FenceStatus, ok: &mut u32, pending: &mut u32, fail: &mut u32) {
    match st {
        FenceStatus::Ok => *ok += 1,
        FenceStatus::Pending => *pending += 1,
        FenceStatus::Missing | FenceStatus::Unsupported(_) => *fail += 1,
    }
}

pub fn process_exec_indirect2<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    payload: &[u8],
) -> ExecResult {
    let exec_started = std::time::Instant::now();
    let mut out = ExecResult::default();
    // Batch-ceiling census: draw runs never span packets.
    state.last_draw_batch_key = None;
    if payload.len() < CHILD_EXEC_INDIRECT_HEADER_LEN as usize {
        return out;
    }
    let raw_task = ld32(&payload[CHILD_EXEC_INDIRECT_TASK_ID as usize..]);
    // The resolver guarantees a live slot or nothing, so there is no second
    // liveness check here. The refusal is always-on: an exec packet the crate
    // drops is a whole command stream of guest work lost, and it used to leave
    // no line at all.
    let Some(task_id) = resolve_task_word(&state.tasks, TaskWordSite::ExecIndirect2, raw_task)
    else {
        out.task_id = raw_task;
        crate::observe::fail(format!(
            "exec_indirect2 no_such_task task={raw_task} tasks={} plen={}",
            state.tasks.len(),
            payload.len()
        ));
        return out;
    };
    out.task_id = task_id;

    let resource_count = ld32(&payload[CHILD_EXEC_INDIRECT_RESOURCE_COUNT as usize..]);
    let cmdbuf_count = ld32(&payload[CHILD_EXEC_INDIRECT_CMDBUF_COUNT as usize..]);
    let resources_len = resource_count as u64 * CHILD_EXEC_INDIRECT_RESOURCE_DESC_LEN as u64;
    let cbufs_off = CHILD_EXEC_INDIRECT_HEADER_LEN as u64 + resources_len;
    let need = cbufs_off + cmdbuf_count as u64 * CHILD_EXEC_INDIRECT_CMDBUF_DESC_LEN as u64;
    if need > payload.len() as u64 {
        crate::observe::fail(format!(
            "exec_indirect2 short_payload task={task_id} res={resource_count} cbufs={cmdbuf_count} need={need} plen={}",
            payload.len()
        ));
        return out;
    }
    if cmdbuf_count == 0 {
        crate::observe::fail(format!(
            "exec_indirect2 zero_cbufs task={task_id} res={resource_count} plen={}",
            payload.len()
        ));
        return out;
    }

    let n_cb = (cmdbuf_count as usize).min(MAX_CMDBUFS);
    let page_shift = state.page_shift;
    let mut streams = Vec::with_capacity(n_cb);
    let load_started = std::time::Instant::now();
    for i in 0..n_cb {
        let off = (cbufs_off + i as u64 * CHILD_EXEC_INDIRECT_CMDBUF_DESC_LEN as u64) as usize;
        if off + CHILD_EXEC_INDIRECT_CMDBUF_DESC_LEN as usize > payload.len() {
            break;
        }
        let gva = ld64(&payload[off + CHILD_EXEC_INDIRECT_CMDBUF_GVA as usize..]);
        let length = ld64(&payload[off + CHILD_EXEC_INDIRECT_CMDBUF_LENGTH as usize..]);
        if length == 0 {
            crate::observe::fail(format!(
                "exec_cmdbuf skip task={task_id} i={i} gva={gva:#x} len=0"
            ));
            continue;
        }
        // Guest length is authoritative — no product MiB budget. Fail only if
        // the host process cannot address the allocation.
        let Some(stream_len) = crate::runtime::metal_draw::host_alloc_len(length) else {
            crate::observe::fail(format!(
                "exec_cmdbuf skip task={task_id} i={i} gva={gva:#x} len={length} (host_len)"
            ));
            continue;
        };
        let mut stream = vec![0u8; stream_len];
        // Product x86 uses page_shift=12; the unshifted helper defaults to arm14
        // and silently fails every stream load on Ventura/Tahoe x86.
        if gva_mem::read_task_gva_by_id(
            host,
            &state.tasks,
            task_id,
            gva,
            &mut stream,
            page_shift,
        )
        .is_err()
        {
            crate::observe::fail(format!(
                "exec_cmdbuf gva_fail task={task_id} i={i} gva={gva:#x} len={length} shift={page_shift}"
            ));
            continue;
        }
        out.streams_loaded += 1;
        streams.push(stream);
    }
    out.load_us = elapsed_us(load_started);

    // Plan before execute: cold AIR translation is immutable CPU work and can
    // run without protocol ownership. Keep the packet unconsumed until every
    // referenced render stage is ready, so replay cannot duplicate clears,
    // fences, compute dispatches, or guest writeback.
    #[cfg(feature = "backend-vulkan")]
    let translation_pending = streams.iter().fold(false, |pending, stream| {
        let render_pending = preflight_render_translations(state, host, task_id, stream);
        let compute_pending = preflight_compute_translations(state, host, task_id, stream);
        render_pending || compute_pending || pending
    });
    #[cfg(all(feature = "backend-metal", target_os = "macos"))]
    let translation_pending = false;
    if translation_pending {
        out.deferred = true;
        return out;
    }

    for stream in streams {
        let mut acc = StreamAccum::default();
        walk_stream(state, host, task_id, &stream, &mut out, &mut acc);
        let finish_started = std::time::Instant::now();
        finish_stream(state, host, task_id, &mut out, &acc);
        out.finish_us = out.finish_us.saturating_add(elapsed_us(finish_started));
    }
    out.total_us = elapsed_us(exec_started);
    out
}

fn elapsed_us(started: std::time::Instant) -> u64 {
    u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX)
}

#[cfg(feature = "backend-vulkan")]
fn preflight_render_translations<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    stream: &[u8],
) -> bool {
    let pipelines = render_pipeline_refs(stream);
    let mut pending = false;
    for pipeline_ref in pipelines {
        let Ok((v_air, f_air)) =
            metal_draw::load_render_air_pair(state, host, task_id, pipeline_ref)
        else {
            // Normal execution emits the precise pipeline/MTLB failure. A
            // missing plan input is deterministic, not asynchronous work.
            continue;
        };
        if !crate::runtime::m2v_cache::ensure_cached_async(
            &v_air,
            metal2vulkan::passes::Stage::Vertex,
            pipeline_ref,
        ) {
            pending = true;
        }
        if !crate::runtime::m2v_cache::ensure_cached_async(
            &f_air,
            metal2vulkan::passes::Stage::Fragment,
            pipeline_ref,
        ) {
            pending = true;
        }
    }
    pending
}

#[cfg(feature = "backend-vulkan")]
fn render_pipeline_refs(stream: &[u8]) -> Vec<u32> {
    // Deliberately silent on a framing refusal: this is a speculative pre-scan of
    // the very stream `walk_stream` is about to frame and report on. Logging here
    // would double every `stream_frame_fail` line for no added information.
    let Ok(segs) = stream::iter_segments(stream) else {
        return Vec::new();
    };
    let mut pipelines = Vec::new();
    for seg in segs {
        if seg.type_ != SEGMENT_TYPE_RENDER {
            continue;
        }
        let mut cursor = 0usize;
        let mut next = decode_first_record(stream, &seg, &mut cursor);
        while let Ok(rec) = next {
            let start = rec.bytes_offset as usize;
            let end = start.saturating_add(rec.length as usize);
            if let Some(bytes) = stream.get(start..end) {
                if let Ok(cmd) = render::decode(bytes) {
                    if cmd.kind == RenderKind::SetPipeline
                        && cmd.pipeline_ref != 0
                        && !pipelines.contains(&cmd.pipeline_ref)
                    {
                        pipelines.push(cmd.pipeline_ref);
                    }
                }
            }
            next = decode_next_record(stream, &seg, &mut cursor);
        }
    }

    pipelines
}

#[cfg(feature = "backend-vulkan")]
fn preflight_compute_translations<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    stream: &[u8],
) -> bool {
    let mut pending = false;
    for (pipeline_ref, local_size) in compute_translation_inputs(stream) {
        let Some(pipeline) =
            compute_exec::load_compute_pipeline(state, host, task_id, pipeline_ref)
        else {
            continue;
        };
        let Some(mtlb) = compute_exec::load_mtlb(state, host, task_id, pipeline.kernel_func_ref)
        else {
            continue;
        };
        let Ok(air) = crate::runtime::mtlb::extract_air(&mtlb) else {
            continue;
        };
        if !crate::runtime::m2v_cache::ensure_cached_kernel_async(air, local_size, pipeline_ref) {
            pending = true;
        }
    }
    pending
}

/// Structurally collect compute pipeline + LocalSize pairs in command order.
/// Threads-indirect carries LocalSize in guest argument memory rather than the
/// stream record, so it deliberately remains on the synchronous fallback.
#[cfg(feature = "backend-vulkan")]
fn compute_translation_inputs(stream: &[u8]) -> Vec<(u32, [u32; 3])> {
    // Silent for the same reason as `render_pipeline_refs`: a pre-scan whose
    // framing refusal `walk_stream` will report once, with the task attached.
    let Ok(segs) = stream::iter_segments(stream) else {
        return Vec::new();
    };
    let mut inputs = Vec::new();
    for seg in segs {
        if seg.type_ != SEGMENT_TYPE_COMPUTE {
            continue;
        }
        let mut pipeline_ref = 0u32;
        let mut cursor = 0usize;
        let mut next = decode_first_record(stream, &seg, &mut cursor);
        while let Ok(rec) = next {
            let start = rec.bytes_offset as usize;
            let end = start.saturating_add(rec.length as usize);
            if let Some(bytes) = stream.get(start..end) {
                if let Ok(cmd) = compute::decode(bytes) {
                    match cmd.kind {
                        ComputeKind::Pipeline => pipeline_ref = cmd.pipeline_ref,
                        ComputeKind::DispatchThreadgroups
                        | ComputeKind::DispatchThreadgroupsIndirect
                        | ComputeKind::DispatchThreads => {
                            let dims = cmd.threads_per_threadgroup;
                            let local_size = [
                                u32::try_from(dims.x).ok(),
                                u32::try_from(dims.y).ok(),
                                u32::try_from(dims.z).ok(),
                            ];
                            if pipeline_ref != 0 {
                                if let [Some(x), Some(y), Some(z)] = local_size {
                                    let item = (pipeline_ref, [x, y, z]);
                                    if x != 0 && y != 0 && z != 0 && !inputs.contains(&item) {
                                        inputs.push(item);
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
            next = decode_next_record(stream, &seg, &mut cursor);
        }
    }
    inputs
}

/// Walk every record in one segment, handing each to `handle`.
///
/// Lifting this out of `walk_stream`'s five near-identical arms gives the framing
/// decoder exactly one emission site. Each arm previously swallowed its refusals
/// twice over: `if let Ok(r) = decode_first_record(..)` dropped a malformed first
/// record with no line at all, and `Err(_) => break` made a truncated or
/// self-inconsistent segment indistinguishable from `Done` — so every remaining
/// record in that segment went unexecuted and unreported.
fn walk_segment_records(
    stream: &[u8],
    seg: &stream::Segment,
    mut handle: impl FnMut(&stream::Record),
) {
    let mut cursor = 0usize;
    let mut next = decode_first_record(stream, seg, &mut cursor);
    loop {
        match next {
            Ok(rec) => {
                handle(&rec);
                next = decode_next_record(stream, seg, &mut cursor);
            }
            // `Done` is end-of-segment and yields `None` here, so the normal exit
            // path stays silent; anything else names the check that refused.
            Err(status) => {
                if let Some(e) = crate::observe::Emit::refusal("stream_record_fail", &status) {
                    // Latch per segment family: a guest re-submitting a malformed
                    // stream sends it on every frame and the second line carries
                    // nothing the first did not. Keying on the family still tells
                    // a broken blit segment from a broken render one, which
                    // keying on the reason alone would hide.
                    e.field("seg", stream::segment_type_name(u32::from(seg.type_)))
                        .field("seg_off", seg.offset)
                        .field("seg_len", seg.length)
                        .field("cursor", cursor)
                        .fail_once(u64::from(seg.type_));
                }
                return;
            }
        }
    }
}

fn walk_stream<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    stream: &[u8],
    out: &mut ExecResult,
    acc: &mut StreamAccum,
) {
    let segs = match stream::iter_segments(stream) {
        Ok(s) => s,
        Err(status) => {
            // The outermost frame in the crate. A stream that will not frame
            // executes *nothing* — and until now that was indistinguishable from
            // an idle guest: no records, no work, no line.
            if let Some(e) = crate::observe::Emit::refusal("stream_frame_fail", &status) {
                e.field("task", task_id)
                    .field("bytes", stream.len())
                    .fail_once(u64::from(task_id));
            }
            return;
        }
    };
    for seg in segs {
        let segment_started = std::time::Instant::now();
        if let Some(e) =
            crate::observe::Emit::refusal("stream_segment", &stream::segment_disposition(seg.type_))
        {
            e.field("seg_type", seg.type_)
                .field("seg_off", seg.offset)
                .field("seg_len", seg.length)
                .fail_once(u64::from(seg.type_));
            continue;
        }
        match seg.type_ {
            SEGMENT_TYPE_RENDER => {
                walk_segment_records(stream, &seg, |r| {
                    handle_render_record(state, host, task_id, stream, r, out, acc)
                });
            }
            SEGMENT_TYPE_BLIT => {
                walk_segment_records(stream, &seg, |r| {
                    handle_blit_record(state, host, task_id, stream, r, out)
                });
            }
            SEGMENT_TYPE_COMPUTE => {
                let mut compute = crate::runtime::compute_session::ComputeSegment::default();
                walk_segment_records(stream, &seg, |r| {
                    handle_compute_record(state, host, task_id, stream, r, out, &mut compute)
                });
                if let Some(st) = crate::runtime::compute_session::finish_session(
                    &mut compute.session,
                    state,
                    host,
                    task_id,
                ) {
                    if !matches!(st, ComputeStatus::Ok) {
                        out.compute_control_fail += 1;
                        // Segment-end commit: the whole multi-record session's
                        // work is gone, and this counter was its only trace.
                        if let Some(e) =
                            crate::observe::Emit::refusal("compute_session_finish", &st)
                        {
                            e.field("task", task_id).fail_once(u64::from(task_id));
                        }
                    }
                }
            }
            SEGMENT_TYPE_EVENT => {
                walk_segment_records(stream, &seg, |r| {
                    handle_event_record(state, task_id, stream, r, out)
                });
            }
            SEGMENT_TYPE_INFO => {
                walk_segment_records(stream, &seg, |r| {
                    handle_info_record(state, host, task_id, stream, r, out)
                });
            }
            // Unreachable: `segment_disposition` already answered `Walk` for
            // exactly the five families above, and `continue`d on the rest.
            _ => {}
        }
        let us = elapsed_us(segment_started);
        match seg.type_ {
            SEGMENT_TYPE_RENDER => out.render_us = out.render_us.saturating_add(us),
            SEGMENT_TYPE_BLIT => out.blit_us = out.blit_us.saturating_add(us),
            SEGMENT_TYPE_COMPUTE => out.compute_us = out.compute_us.saturating_add(us),
            SEGMENT_TYPE_EVENT => out.event_us = out.event_us.saturating_add(us),
            SEGMENT_TYPE_INFO => out.info_us = out.info_us.saturating_add(us),
            _ => {}
        }
    }
}

fn handle_info_record<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    stream: &[u8],
    rec: &stream::Record,
    out: &mut ExecResult,
) {
    use crate::runtime::icb::{
        apply_icb_host_resource_info, decode_icb_host_resource_info, INFO_OP_ICB_HOST_RESOURCE,
    };
    let end = (rec.bytes_offset as usize).saturating_add(rec.length as usize);
    if end > stream.len() {
        return;
    }
    let bytes = &stream[rec.bytes_offset as usize..end];
    if rec.opcode == INFO_OP_ICB_HOST_RESOURCE {
        // `icb_backing_fail` was a counter with no reason beside it: an ICB
        // whose command memory never bound looked identical whether the payload
        // was malformed, the type-1 buffer was short, or the pathway has no ICB
        // execution at all. Latched per ICB ref — the guest re-sends `0x1d1`
        // for the same buffer, so an unlatched line would be one per frame.
        match decode_icb_host_resource_info(bytes) {
            Ok(info) => match apply_icb_host_resource_info(state, host, task_id, &info) {
                Ok(_) => out.icb_backing_ok = out.icb_backing_ok.saturating_add(1),
                Err(e) => {
                    crate::observe::Emit::decline("icb_backing", &e)
                        .field("task", task_id)
                        .field("icb", info.icb_ref)
                        .field("buffer", info.buffer_ref)
                        .fail_once(info.icb_ref as u64);
                    out.icb_backing_fail = out.icb_backing_fail.saturating_add(1);
                }
            },
            Err(e) => {
                crate::observe::Emit::decline("icb_backing", &e)
                    .field("task", task_id)
                    .field("len", bytes.len())
                    .fail_once(rec.length as u64);
                out.icb_backing_fail = out.icb_backing_fail.saturating_add(1);
            }
        }
    }
}

fn handle_event_record(
    state: &mut DeviceState,
    task_id: u32,
    stream: &[u8],
    rec: &stream::Record,
    out: &mut ExecResult,
) {
    let end = (rec.bytes_offset as usize).saturating_add(rec.length as usize);
    if end > stream.len() {
        return;
    }
    let cmd_bytes = &stream[rec.bytes_offset as usize..end];
    let cmd = match event_decode::decode(cmd_bytes) {
        Ok(c) => c,
        Err(_) => {
            out.event_ops_fail += 1;
            return;
        }
    };
    let st = fence_exec::execute_event(state, task_id, &cmd);
    tally_fence(
        st,
        &mut out.event_ops_ok,
        &mut out.event_ops_pending,
        &mut out.event_ops_fail,
    );
}

/// Name a compute refusal at the rail boundary.
///
/// Until this existed the three dispatch/control/ICB arms below only
/// *counted*: `compute_dispatches_fail` went up and nothing said which of the
/// rail's ~150 checks refused, because nine of `ComputeStatus`'s variants were
/// payload-free. The slug now rides in the status, so one line names the check,
/// the pipeline and the record kind.
///
/// Latched per `(reason, pipeline)`: the guest re-submits the same dispatch
/// every frame, so a persistent refusal would otherwise be a per-frame flood —
/// while a *different* pipeline failing the same check is a distinct event and
/// still gets its line.
fn note_compute_refusal(status: ComputeStatus, task_id: u32, pipeline_ref: u32, kind: ComputeKind) {
    // One event token for the whole rail, with `kind=` separating dispatch
    // from control-flow from ICB: the emission gate reads the *literal* first
    // argument, so a per-arm event passed in as a parameter would leave the
    // registry naming a line the gate cannot find.
    if let Some(e) = crate::observe::Emit::refusal("compute_record", &status) {
        e.field("task", task_id)
            .field("pipe", pipeline_ref)
            .field("kind", format!("{kind:?}"))
            .fail_once(u64::from(pipeline_ref));
    }
}

fn handle_compute_record<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    stream: &[u8],
    rec: &stream::Record,
    out: &mut ExecResult,
    seg: &mut crate::runtime::compute_session::ComputeSegment,
) {
    let end = (rec.bytes_offset as usize).saturating_add(rec.length as usize);
    if end > stream.len() {
        return;
    }
    let cmd_bytes = &stream[rec.bytes_offset as usize..end];
    let cmd = match compute::decode(cmd_bytes) {
        Ok(c) => c,
        // Same silent drop as the render path above.
        Err(status) => {
            if let Some(e) = crate::observe::Emit::refusal("compute_decode", &status) {
                // Latched per (reason, opcode): the guest re-encodes the same
                // stream every frame, so an unclassified opcode would arrive
                // once per draw. Magnitude is the encoder's fail counter's job.
                e.field("opcode", format!("{:#x}", rec.opcode))
                    .field("len", cmd_bytes.len())
                    .fail_once(rec.opcode as u64);
            }
            return;
        }
    };
    match cmd.kind {
        ComputeKind::UpdateFence | ComputeKind::WaitFence => {
            let action = if cmd.kind == ComputeKind::UpdateFence {
                FenceAction::Update
            } else {
                FenceAction::Wait
            };
            let st = fence_exec::execute_fence(
                state,
                task_id,
                FenceDomain::ComputeFence,
                cmd.fence_ref,
                action,
                0,
            );
            tally_fence(
                st,
                &mut out.compute_fences_ok,
                &mut out.compute_fences_pending,
                &mut out.compute_fences_fail,
            );
        }
        ComputeKind::BufferBind | ComputeKind::BufferBindAttributeStride => {
            let before = seg.acc.buffers.len();
            let _ = compute_exec::apply_record(state, host, task_id, &cmd, seg);
            if seg.acc.buffers.len() > before {
                out.compute_buffer_binds += (seg.acc.buffers.len() - before) as u32;
            } else {
                out.compute_buffer_binds =
                    out.compute_buffer_binds.max(seg.acc.buffers.len() as u32);
            }
        }
        ComputeKind::TextureBind => {
            let before = seg.acc.textures.len();
            let _ = compute_exec::apply_record(state, host, task_id, &cmd, seg);
            if seg.acc.textures.len() > before {
                out.compute_texture_binds += (seg.acc.textures.len() - before) as u32;
            } else {
                out.compute_texture_binds = out
                    .compute_texture_binds
                    .max(seg.acc.textures.len() as u32);
            }
        }
        ComputeKind::SamplerBind | ComputeKind::SamplerLod => {
            let before = seg.acc.samplers.len();
            let _ = compute_exec::apply_record(state, host, task_id, &cmd, seg);
            if seg.acc.samplers.len() > before {
                out.compute_sampler_binds += (seg.acc.samplers.len() - before) as u32;
            } else {
                out.compute_sampler_binds = out
                    .compute_sampler_binds
                    .max(seg.acc.samplers.len() as u32);
            }
        }
        ComputeKind::Pipeline
        | ComputeKind::BufferOffset
        | ComputeKind::BufferOffsetAttributeStride
        | ComputeKind::DispatchType
        | ComputeKind::StageInRegion
        | ComputeKind::StageInRegionIndirect
        | ComputeKind::ThreadgroupMemory
        | ComputeKind::ImageblockDimensions
        | ComputeKind::BarrierResources
        | ComputeKind::BarrierScope
        | ComputeKind::UseHeaps
        | ComputeKind::UseResources
        | ComputeKind::CompressedTextureFlush => {
            let _ = compute_exec::apply_record(state, host, task_id, &cmd, seg);
        }
        ComputeKind::DispatchThreadgroups
        | ComputeKind::DispatchThreads
        | ComputeKind::DispatchThreadgroupsIndirect
        | ComputeKind::DispatchThreadsIndirect => {
            let pipeline_ref = seg.acc.pipeline_ref;
            match compute_exec::apply_record(state, host, task_id, &cmd, seg) {
                Some(ComputeStatus::Ok) => out.compute_dispatches_ok += 1,
                Some(st) => {
                    out.compute_dispatches_fail += 1;
                    note_compute_refusal(st, task_id, pipeline_ref, cmd.kind);
                }
                None => out.compute_dispatches_fail += 1,
            }
        }
        ComputeKind::ControlStartDoWhile
        | ComputeKind::ControlEndDoWhile
        | ComputeKind::ControlStartWhile
        | ComputeKind::ControlEndWhile
        | ComputeKind::ControlStartIf
        | ComputeKind::ControlStartElse
        | ComputeKind::ControlEndIf => {
            let pipeline_ref = seg.acc.pipeline_ref;
            match compute_exec::apply_record(state, host, task_id, &cmd, seg) {
                Some(ComputeStatus::Ok) => out.compute_control_ok += 1,
                Some(st) => {
                    out.compute_control_fail += 1;
                    note_compute_refusal(st, task_id, pipeline_ref, cmd.kind);
                }
                None => out.compute_control_fail += 1,
            }
        }
        ComputeKind::ExecuteCommandsInBuffer | ComputeKind::ExecuteCommandsInBufferIndirect => {
            let pipeline_ref = seg.acc.pipeline_ref;
            match compute_exec::apply_record(state, host, task_id, &cmd, seg) {
                Some(ComputeStatus::Ok) => out.compute_icb_ok += 1,
                Some(st) => {
                    out.compute_icb_fail += 1;
                    note_compute_refusal(st, task_id, pipeline_ref, cmd.kind);
                }
                None => out.compute_icb_fail += 1,
            }
        }
        _ => {}
    }
}

fn handle_blit_record<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    stream: &[u8],
    rec: &stream::Record,
    out: &mut ExecResult,
) {
    let end = (rec.bytes_offset as usize).saturating_add(rec.length as usize);
    if end > stream.len() {
        return;
    }
    let cmd_bytes = &stream[rec.bytes_offset as usize..end];
    let cmd = match blit::decode(cmd_bytes) {
        Ok(c) => c,
        // Was `Err(_) => return`: a decoded blit record dropped with no line at
        // all, which on a live boot is indistinguishable from a segment that
        // carried no blit work. The status names which of the four checks
        // refused.
        Err(status) => {
            if let Some(e) = crate::observe::Emit::refusal("blit_decode", &status) {
                e.field("opcode", format!("{:#x}", rec.opcode))
                    .field("len", cmd_bytes.len())
                    .fail();
            }
            return;
        }
    };
    match cmd.kind {
        BlitKind::Resource if cmd.opcode == OP_GENERATE_MIPMAPS => {
            match mipmap::generate_mipmaps_linear(state, host, task_id, cmd.resource) {
                MipmapStatus::Ok => out.mipmaps_ok += 1,
                st => {
                    out.mipmaps_fail += 1;
                    // Was `st={st:?}` with no `reason=` at all, so none of the
                    // eight outcomes was greppable and the Debug spelling was
                    // the only handle on which check refused.
                    if let Some(e) = crate::observe::Emit::refusal("blit_generate_mipmaps", &st) {
                        e.field("resource", cmd.resource).fail();
                    }
                }
            }
        }
        // optimize*/synchronize* are protocol no-ops on the unified-memory path.
        BlitKind::Resource | BlitKind::Image => {}
        BlitKind::Fence => {
            // Log from the *blit* status, before the remap. The remap folds two
            // meanings into `FenceStatus::Missing` — an absent object and a zero
            // fence ref — and only the blit rail's own reason can tell them
            // apart; `Refusal for BlitStatus` reproduces this site's previous
            // log condition exactly.
            let blit_st = blit_exec::execute_blit_fence(state, task_id, &cmd);
            if let Some(e) = crate::observe::Emit::refusal("blit_fence_fail", &blit_st) {
                e.field("opcode", format!("{:#x}", cmd.opcode)).fail();
            }
            let st = match blit_st {
                BlitStatus::Ok => FenceStatus::Ok,
                BlitStatus::FencePending => FenceStatus::Pending,
                BlitStatus::MissingResource => FenceStatus::Missing,
                // Carry the blit rail's reason into the status instead of
                // dropping it: this arm covers six blit checks and the tally
                // below cannot tell them apart otherwise. The slug is owned by
                // the blit reason channel, not registered here.
                _ => FenceStatus::Unsupported(blit_exec::blit_fail_reason()),
            };
            tally_fence(
                st,
                &mut out.blit_fences_ok,
                &mut out.blit_fences_pending,
                &mut out.blit_fences_fail,
            );
        }
        BlitKind::FillBuffer | BlitKind::Copy => {
            match blit_exec::execute_blit(state, host, task_id, &cmd) {
                BlitStatus::Ok | BlitStatus::ZeroExtent => {
                    if cmd.kind == BlitKind::FillBuffer {
                        out.blit_fills_ok += 1;
                    } else {
                        out.blit_copies_ok += 1;
                    }
                }
                st => {
                    out.blit_fail += 1;
                    // Icon/upload path often uses blit copies; fail-visible for RE.
                    // The reason names the specific failing site inside blit_exec
                    // that produced the coarse `st` — 177 checks collapse into
                    // eight statuses, so the status alone says almost nothing.
                    // `Refusal` supplies it, and an uninstrumented site now reads
                    // `blit_unattributed` rather than rendering a bare `reason=`.
                    let src_ty = objects::lookup_list_entry(state, host, task_id, cmd.source)
                        .map(|e| e.object_type)
                        .unwrap_or(0);
                    let dst_ty = objects::lookup_list_entry(state, host, task_id, cmd.destination)
                        .map(|e| e.object_type)
                        .unwrap_or(0);
                    if let Some(e) = crate::observe::Emit::refusal("blit_fail", &st) {
                        e.field("st", format!("{st:?}"))
                            .field("kind", format!("{:?}", cmd.kind))
                            .field("opcode", format!("{:#x}", cmd.opcode))
                            .field("src", cmd.source)
                            .field("src_ty", src_ty)
                            .field("dst", cmd.destination)
                            .field("dst_ty", dst_ty)
                            .field("off", cmd.source_offset)
                            .field(
                                "lvl",
                                format!("{}/{}", cmd.destination_level, cmd.source_level),
                            )
                            .field(
                                "size",
                                format!(
                                    "{}x{}x{}",
                                    cmd.source_size.width,
                                    cmd.source_size.height,
                                    cmd.source_size.depth
                                ),
                            )
                            .fail();
                    }
                }
            }
        }
        BlitKind::Unknown => {
            crate::observe::fail(format!(
                "blit unknown opcode={:#x} len={}",
                cmd.opcode, rec.length
            ));
        }
    }
}

fn handle_render_record<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &M,
    task_id: u32,
    stream: &[u8],
    rec: &stream::Record,
    out: &mut ExecResult,
    acc: &mut StreamAccum,
) {
    let end = (rec.bytes_offset as usize).saturating_add(rec.length as usize);
    if end > stream.len() {
        return;
    }
    let cmd_bytes = &stream[rec.bytes_offset as usize..end];
    let cmd = match render::decode(cmd_bytes) {
        Ok(c) => c,
        // Was `Err(_) => return`: a malformed render command dropped with no
        // line, on the hottest path in the crate. Indistinguishable from a
        // segment that simply carried no render work.
        Err(status) => {
            if let Some(e) = crate::observe::Emit::refusal("render_decode", &status) {
                // Latched per (reason, opcode): the guest re-encodes the same
                // stream every frame, so an unclassified opcode would arrive
                // once per draw. Magnitude is the encoder's fail counter's job.
                e.field("opcode", format!("{:#x}", rec.opcode))
                    .field("len", cmd_bytes.len())
                    .fail_once(rec.opcode as u64);
            }
            return;
        }
    };
    match cmd.kind {
        RenderKind::SetPipeline if cmd.pipeline_ref != 0 => {
            acc.pipeline_ref = cmd.pipeline_ref;
        }
        RenderKind::SetBuffer => {
            // Multi-entry archive layout: slots first..first+n from buffer_binds.
            let binds = if !cmd.buffer_binds.is_empty() {
                cmd.buffer_binds.as_slice()
            } else if cmd.buffer_ref != 0 {
                // Legacy single-entry fallback.
                &[(cmd.buffer_ref, cmd.buffer_offset)][..]
            } else {
                &[][..]
            };
            for (i, &(buffer_ref, offset)) in binds.iter().enumerate() {
                let index = cmd.first.saturating_add(i as u32);
                if index >= MAX_BIND_SLOTS {
                    break;
                }
                if buffer_ref == 0 {
                    match cmd.stage {
                        Stage::Vertex => acc.vertex_buffers.retain(|b| b.index != index),
                        Stage::Fragment => acc.fragment_buffers.retain(|b| b.index != index),
                        _ => {}
                    }
                    out.buffer_unbinds = out.buffer_unbinds.saturating_add(1);
                    continue;
                }
                let bind = BufferBind {
                    stage: cmd.stage,
                    index,
                    buffer_ref,
                    offset,
                };
                match cmd.stage {
                    Stage::Vertex => upsert_buffer(&mut acc.vertex_buffers, bind),
                    Stage::Fragment => upsert_buffer(&mut acc.fragment_buffers, bind),
                    _ => {}
                }
            }
            out.buffer_binds = (acc.vertex_buffers.len() + acc.fragment_buffers.len()) as u32;
        }
        RenderKind::SetBufferOffset => {
            // Archive apply_buffer_offset: update offset on an already-bound slot.
            if cmd.first < MAX_BIND_SLOTS {
                let list = match cmd.stage {
                    Stage::Vertex => &mut acc.vertex_buffers,
                    Stage::Fragment => &mut acc.fragment_buffers,
                    _ => {
                        return;
                    }
                };
                if let Some(b) = list.iter_mut().find(|b| b.index == cmd.first) {
                    b.offset = cmd.buffer_offset;
                }
            }
        }
        RenderKind::SetTexture => {
            let refs: Vec<u32> = if !cmd.ref_binds.is_empty() {
                cmd.ref_binds.clone()
            } else if cmd.texture_ref != 0 {
                vec![cmd.texture_ref]
            } else {
                Vec::new()
            };
            for (i, &texture_ref) in refs.iter().enumerate() {
                let index = cmd.first.saturating_add(i as u32);
                if index >= MAX_BIND_SLOTS {
                    break;
                }
                if texture_ref == 0 {
                    match cmd.stage {
                        Stage::Vertex => acc.vertex_textures.retain(|b| b.index != index),
                        Stage::Fragment => acc.fragment_textures.retain(|b| b.index != index),
                        _ => {}
                    }
                    out.texture_unbinds = out.texture_unbinds.saturating_add(1);
                    continue;
                }
                if !out.texture_refs.contains(&texture_ref) {
                    out.texture_refs.push(texture_ref);
                }
                let bind = TextureBind {
                    stage: cmd.stage,
                    index,
                    texture_ref,
                };
                match cmd.stage {
                    Stage::Vertex => upsert_texture(&mut acc.vertex_textures, bind),
                    Stage::Fragment => upsert_texture(&mut acc.fragment_textures, bind),
                    _ => {}
                }
                if let Some(m) = objects::resolve_type11_ref(state, host, task_id, texture_ref) {
                    if !out.type11_mappings.contains(&m) {
                        out.type11_mappings.push(m);
                    }
                } else if objects::resolve_type4_surface(state, host, texture_ref) {
                    // x86 type-4: object ref is surface_id / mapping_id.
                    if !out.type11_mappings.contains(&texture_ref) {
                        out.type11_mappings.push(texture_ref);
                    }
                }
            }
            out.texture_binds = (acc.vertex_textures.len() + acc.fragment_textures.len()) as u32;
        }
        RenderKind::SetSampler => {
            let refs: Vec<u32> = if !cmd.ref_binds.is_empty() {
                cmd.ref_binds.clone()
            } else if cmd.sampler_ref != 0 {
                vec![cmd.sampler_ref]
            } else {
                Vec::new()
            };
            for (i, &sampler_ref) in refs.iter().enumerate() {
                let index = cmd.first.saturating_add(i as u32);
                if index >= MAX_BIND_SLOTS {
                    break;
                }
                if sampler_ref == 0 {
                    match cmd.stage {
                        Stage::Vertex => acc.vertex_samplers.retain(|b| b.index != index),
                        Stage::Fragment => acc.fragment_samplers.retain(|b| b.index != index),
                        _ => {}
                    }
                    out.sampler_unbinds = out.sampler_unbinds.saturating_add(1);
                    continue;
                }
                let bind = SamplerBind {
                    stage: cmd.stage,
                    index,
                    sampler_ref,
                };
                match cmd.stage {
                    Stage::Vertex => upsert_sampler(&mut acc.vertex_samplers, bind),
                    Stage::Fragment => upsert_sampler(&mut acc.fragment_samplers, bind),
                    _ => {}
                }
            }
        }
        RenderKind::SetViewport => {
            acc.viewport = Some(cmd.viewport);
        }
        RenderKind::SetScissor if cmd.scissor_w > 0 && cmd.scissor_h > 0 => {
            acc.scissor = Some((cmd.scissor_x, cmd.scissor_y, cmd.scissor_w, cmd.scissor_h));
        }
        RenderKind::SetBlendColor if cmd.has_blend_color => {
            acc.blend_color = Some(cmd.blend_color);
        }
        RenderKind::SetCullMode if cmd.has_cull_mode => {
            acc.cull_mode = Some(cmd.cull_mode);
        }
        RenderKind::SetFrontFacing if cmd.has_front_facing => {
            acc.front_facing = Some(cmd.front_facing);
        }
        RenderKind::SetDepthBias if cmd.has_depth_bias => {
            acc.depth_bias = Some(cmd.depth_bias);
        }
        RenderKind::SetDepthStencil => {
            acc.depth_stencil_ref = cmd.depth_stencil_ref;
        }
        RenderKind::SetStencilReference if cmd.has_stencil_ref => {
            acc.stencil_ref = Some((cmd.stencil_ref_front, cmd.stencil_ref_back));
        }
        RenderKind::RenderPass => {
            // Full multi-attachment: re-decode all color slots from payload.
            if cmd_bytes.len() >= 8 {
                let payload = &cmd_bytes[8..];
                let depth = decode_depth_attachment(payload);
                if depth.present && depth.level == 0 && depth.resolve_texture_ref == 0 {
                    acc.depth_attach = Some(depth);
                }
                let stencil = decode_stencil_attachment(payload);
                if stencil.present && stencil.level == 0 && stencil.resolve_texture_ref == 0 {
                    acc.stencil_attach = Some(stencil);
                }
                for i in 0..PASS_MAX_COLOR_ATTACHMENTS {
                    let att = decode_color_attachment(payload, i);
                    if !att.present || att.texture_ref == 0 {
                        continue;
                    }
                    let slot = i as u32;
                    if !acc
                        .color_slots
                        .iter()
                        .any(|(s, a)| *s == slot || a.texture_ref == att.texture_ref)
                    {
                        acc.color_slots.push((slot, att));
                    } else if let Some(entry) = acc.color_slots.iter_mut().find(|(s, _)| *s == slot)
                    {
                        entry.1 = att;
                    }
                    if !out.color_targets.contains(&att.texture_ref) {
                        out.color_targets.push(att.texture_ref);
                    }
                    if !acc.color_targets.contains(&att.texture_ref) {
                        acc.color_targets.push(att.texture_ref);
                    }
                    if !out.texture_refs.contains(&att.texture_ref) {
                        out.texture_refs.push(att.texture_ref);
                    }
                    if let Some(m) =
                        objects::resolve_type11_ref(state, host, task_id, att.texture_ref)
                    {
                        if !out.type11_mappings.contains(&m) {
                            out.type11_mappings.push(m);
                        }
                    } else if objects::resolve_type4_surface(state, host, att.texture_ref)
                        && !out.type11_mappings.contains(&att.texture_ref)
                    {
                        out.type11_mappings.push(att.texture_ref);
                    }
                    if att.load_action == PASS_LOAD_ACTION_CLEAR {
                        if att.store_action == PASS_STORE_ACTION_STORE {
                            acc.clears.push(att);
                        } else {
                            // Metal Clear + non-Store (e.g. DontCare): the clear
                            // seed is dropped from `acc.clears`, so a drawn pass
                            // loads stale content (residue) and a clear-only
                            // stream never reaches guest pages. We do NOT invent
                            // DontCare semantics (unknown wire stays unknown) —
                            // just make the drop visible so a boot reveals whether
                            // any guest emits it. Deduped per target.
                            note_clear_dropped(
                                "nonstore_store_action",
                                att.texture_ref,
                                &format!("store_action={} load_action=clear", att.store_action),
                            );
                        }
                    }
                }
            }
            // Also keep color0 from command for convenience.
            if cmd.color0.present
                && cmd.color0.load_action == PASS_LOAD_ACTION_CLEAR
                && cmd.color0.store_action == PASS_STORE_ACTION_STORE
                && !acc
                    .clears
                    .iter()
                    .any(|a| a.texture_ref == cmd.color0.texture_ref)
            {
                acc.clears.push(cmd.color0);
            }
        }
        RenderKind::Draw => {
            if cmd.opcode == render::OP_DRAW_INDEXED_WIDE {
                crate::observe::line(format!(
                    "render_wide_indexed task={task_id} target_refs={:?} pipeline={} prim={} index_type={} index_ref={} count={} offset={:#x}",
                    acc.color_targets,
                    acc.pipeline_ref,
                    cmd.primitive_type,
                    cmd.index_type,
                    cmd.index_buffer_ref,
                    cmd.index_count,
                    cmd.index_buffer_offset
                ));
            }
            acc.saw_draw = true;
            out.saw_draw = true;
            let count = if cmd.index_count != 0 {
                cmd.index_count
            } else {
                cmd.vertex_count
            };
            if cmd.index_count != 0 && cmd.index_buffer_ref != 0 {
                acc.indexed = Some(IndexedDrawInfo {
                    index_type: cmd.index_type,
                    index_count: cmd.index_count,
                    index_buffer_ref: cmd.index_buffer_ref,
                    index_buffer_offset: cmd.index_buffer_offset,
                });
            } else {
                acc.indexed = None;
            }
            // Snapshot bind state for this draw (archive multi-draw job).
            if acc.draws.len() < MAX_DRAWS_PER_STREAM && acc.pipeline_ref != 0 && count > 0 {
                acc.draws.push(PendingDraw {
                    pipeline_ref: acc.pipeline_ref,
                    draw: (
                        count,
                        cmd.instance_count.max(1),
                        cmd.primitive_type,
                        cmd.vertex_start,
                    ),
                    indexed: acc.indexed.clone(),
                    vertex_buffers: acc.vertex_buffers.clone(),
                    fragment_buffers: acc.fragment_buffers.clone(),
                    vertex_textures: acc.vertex_textures.clone(),
                    fragment_textures: acc.fragment_textures.clone(),
                    vertex_samplers: acc.vertex_samplers.clone(),
                    fragment_samplers: acc.fragment_samplers.clone(),
                    viewport: acc.viewport,
                    scissor: acc.scissor,
                    blend_color: acc.blend_color,
                    cull_mode: acc.cull_mode,
                    front_facing: acc.front_facing,
                    depth_bias: acc.depth_bias,
                    depth_stencil_ref: acc.depth_stencil_ref,
                    stencil_ref: acc.stencil_ref,
                    depth_attach: acc.depth_attach,
                    stencil_attach: acc.stencil_attach,
                });
            }
        }
        RenderKind::ExecuteCommands if cmd.indirect_command_buffer_ref != 0 => {
            acc.execute_icb = Some(RenderIcbExecute {
                icb_ref: cmd.indirect_command_buffer_ref,
                is_range: cmd.icb_is_range,
                range_location: cmd.icb_range_location,
                range_length: cmd.icb_range_length,
                args_buffer_ref: cmd.icb_args_buffer_ref,
                args_buffer_offset: cmd.icb_args_buffer_offset,
            });
        }
        RenderKind::Fence => {
            let action = match cmd.opcode {
                RENDER_OP_UPDATE_FENCE => FenceAction::Update,
                RENDER_OP_WAIT_FENCE => FenceAction::Wait,
                _ => {
                    out.render_fences_fail += 1;
                    return;
                }
            };
            let st = fence_exec::execute_fence(
                state,
                task_id,
                FenceDomain::RenderFence,
                cmd.fence_ref,
                action,
                0,
            );
            tally_fence(
                st,
                &mut out.render_fences_ok,
                &mut out.render_fences_pending,
                &mut out.render_fences_fail,
            );
        }
        RenderKind::OtherAccepted => {
            // An undecoded render opcode: the decoder accepts it (catch-all)
            // but no executor exists, so the guest command is effectively
            // dropped. That MUST stay fail-visible — but a per-draw op such as
            // 0x7c fires thousands of times per app render, so emitting per
            // record floods /tmp/reims-vgpu-fail.log (measured ~2620 lines from six
            // app launches). Dedup to ONE line per distinct opcode (the set is
            // tiny and boot-stable) and capture the raw wire on first sighting
            // so the layout can be decoded offline. Unknown wire stays unknown;
            // we never invent semantics for it.
            note_unimplemented_render_opcode(cmd.opcode, rec.length, cmd_bytes, task_id, acc);
        }
        _ => {}
    }
}

/// Fail-visible, deduped record of a render opcode the decoder accepts but has
/// no executor for (`RenderKind::OtherAccepted`). Fires exactly ONE line per
/// distinct opcode — the undecoded-opcode set is tiny and boot-stable, so this
/// keeps the "guest render command dropped" signal visible on the always-on
/// sink without the per-draw flood a bare emit would produce (a per-draw op
/// like 0x7c fired ~2620 times across six app launches). The line carries the
/// length, bound targets/pipeline, bind counts, and the first-sighting raw wire
/// (hex) so the exact layout can be decoded offline later. Runs on the drain
/// worker (off the QEMU main/vCPU threads). Diagnostic only — it never gates
/// behavior and never invents semantics for the unknown wire.
// Render opcodes are < 256 by contract (observed max 0x98); a dense lock-free
// table gives a zero-alloc, wait-free fast path after warmup. Module-scope so a
// test can reset it deterministically.
const UNIMPL_OPCODE_TABLE: usize = 256;
static UNIMPL_OPCODE_SEEN: [std::sync::atomic::AtomicBool; UNIMPL_OPCODE_TABLE] =
    [const { std::sync::atomic::AtomicBool::new(false) }; UNIMPL_OPCODE_TABLE];
static UNIMPL_OPCODE_OVERFLOW: std::sync::OnceLock<
    std::sync::Mutex<std::collections::HashSet<u32>>,
> = std::sync::OnceLock::new();

/// Returns `true` if this call emitted the line (first sighting of `opcode`),
/// `false` if it was deduped. The caller ignores it; tests use it to assert the
/// anti-flood behavior without depending on the shared always-on log file.
fn note_unimplemented_render_opcode(
    opcode: u32,
    length: u32,
    cmd_bytes: &[u8],
    task_id: u32,
    acc: &StreamAccum,
) -> bool {
    use std::sync::atomic::Ordering;
    if (opcode as usize) < UNIMPL_OPCODE_TABLE {
        // First sighting only: swap false->true; racers that lose stay quiet.
        if UNIMPL_OPCODE_SEEN[opcode as usize].swap(true, Ordering::Relaxed) {
            return false;
        }
    } else {
        // Out-of-range opcode (decode desync / garbage) — dedup through a
        // small overflow set so a runaway value cannot flood either.
        let set = UNIMPL_OPCODE_OVERFLOW.get_or_init(|| std::sync::Mutex::new(Default::default()));
        if let Ok(mut g) = set.lock() {
            if !g.insert(opcode) {
                return false;
            }
        }
    }
    let hex: String = cmd_bytes
        .iter()
        .take(48)
        .map(|b| format!("{:02x}", b))
        .collect::<Vec<_>>()
        .join("");
    crate::observe::fail(format!(
        "render_unimplemented reason=accepted_without_executor task={task_id} opcode={:#x} len={} target_refs={:?} pipeline={} vbufs={} fbufs={} ftex={} hex={}",
        opcode,
        length,
        acc.color_targets,
        acc.pipeline_ref,
        acc.vertex_buffers.len(),
        acc.fragment_buffers.len(),
        acc.fragment_textures.len(),
        hex
    ));
    true
}

/// Serializes the two tests that share the process-global unimplemented-opcode
/// dedup latch, so one test's reset cannot race the other's emissions.
#[cfg(test)]
static UNIMPL_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Clear the unimplemented-opcode dedup latch so a test can deterministically
/// observe the first-sighting line regardless of prior in-process emissions.
#[cfg(test)]
fn reset_unimplemented_opcode_dedup_for_test() {
    for slot in UNIMPL_OPCODE_SEEN.iter() {
        slot.store(false, std::sync::atomic::Ordering::Relaxed);
    }
    if let Some(set) = UNIMPL_OPCODE_OVERFLOW.get() {
        if let Ok(mut g) = set.lock() {
            g.clear();
        }
    }
}

fn upsert_buffer(list: &mut Vec<BufferBind>, bind: BufferBind) {
    if let Some(slot) = list.iter_mut().find(|b| b.index == bind.index) {
        *slot = bind;
    } else {
        list.push(bind);
    }
}

fn upsert_texture(list: &mut Vec<TextureBind>, bind: TextureBind) {
    if let Some(slot) = list.iter_mut().find(|b| b.index == bind.index) {
        *slot = bind;
    } else {
        list.push(bind);
    }
}

fn upsert_sampler(list: &mut Vec<SamplerBind>, bind: SamplerBind) {
    if let Some(slot) = list.iter_mut().find(|b| b.index == bind.index) {
        *slot = bind;
    } else {
        list.push(bind);
    }
}

fn finish_stream<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    out: &mut ExecResult,
    acc: &StreamAccum,
) {
    // Archive ApplePVGPUDrawJob: clear/load seed is private initial_rgba for the
    // async job; guest pages are written once at completion. Apply clear-to-guest
    // only for clear-only streams (no draws). When draws run, CLEAR is the Metal
    // pass seed inside encode (mrt_draw_request solid seed) — not a pre-draw
    // guest store that would expose intermediate pixels to DisplaySwap.
    let will_draw = acc.saw_draw && !acc.color_slots.is_empty() && !acc.draws.is_empty();
    if !will_draw {
        for att in &acc.clears {
            if apply_clear(state, host, task_id, att) {
                out.clears_applied += 1;
            }
        }
    }

    // Render ICB execute (`0x14`/`0x15`) — open pass over color slots and run ICB.
    if let Some(exec) = &acc.execute_icb {
        if !acc.color_slots.is_empty() {
            // Pipeline optional for empty ICB; use first pass color geometry via mrt helper.
            let pipeline = if acc.pipeline_ref != 0 {
                acc.pipeline_ref
            } else {
                // Still need a non-zero ref for mrt_draw_request gate — use 1 as placeholder
                // when only ICB execute (PSO lives inside filled slots). mrt_draw_request
                // only uses pipeline for encode_draw; for ICB we rebuild colors manually.
                0
            };
            let req = if pipeline != 0 {
                if let Some(pd) = acc.draws.last() {
                    let (count, inst, prim, first) = pd.draw;
                    metal_draw::mrt_draw_request(
                        state,
                        host,
                        task_id,
                        pipeline,
                        &acc.color_slots,
                        &acc.clears,
                        count.max(1),
                        inst.max(1),
                        prim,
                        first,
                    )
                } else {
                    metal_draw::mrt_draw_request(
                        state,
                        host,
                        task_id,
                        pipeline,
                        &acc.color_slots,
                        &acc.clears,
                        1,
                        1,
                        3,
                        0,
                    )
                }
            } else {
                // Build color RT list without pipeline (ICB-only execute).
                metal_draw::mrt_draw_request(
                    state,
                    host,
                    task_id,
                    1, // unused when we only need colors
                    &acc.color_slots,
                    &acc.clears,
                    1,
                    1,
                    3,
                    0,
                )
            };
            if let Some(mut req) = req {
                // ICB execute inherits stream bind state at end of stream.
                if let Some(pd) = acc.draws.last() {
                    fill_draw_binds_from_pending(&mut req, pd);
                } else {
                    fill_draw_binds_from_pending(
                        &mut req,
                        &PendingDraw {
                            vertex_buffers: acc.vertex_buffers.clone(),
                            fragment_buffers: acc.fragment_buffers.clone(),
                            vertex_textures: acc.vertex_textures.clone(),
                            fragment_textures: acc.fragment_textures.clone(),
                            vertex_samplers: acc.vertex_samplers.clone(),
                            fragment_samplers: acc.fragment_samplers.clone(),
                            viewport: acc.viewport,
                            scissor: acc.scissor,
                            indexed: acc.indexed.clone(),
                            blend_color: acc.blend_color,
                            cull_mode: acc.cull_mode,
                            front_facing: acc.front_facing,
                            depth_bias: acc.depth_bias,
                            depth_stencil_ref: acc.depth_stencil_ref,
                            stencil_ref: acc.stencil_ref,
                            depth_attach: acc.depth_attach,
                            stencil_attach: acc.stencil_attach,
                            ..Default::default()
                        },
                    );
                }
                let (loc, len) = if exec.is_range {
                    (exec.range_location, exec.range_length)
                } else {
                    // Indirect: stage 8-byte range from guest buffer.
                    match read_icb_exec_range(
                        state,
                        host,
                        task_id,
                        exec.args_buffer_ref,
                        exec.args_buffer_offset,
                    ) {
                        Some(v) => v,
                        None => {
                            // Sibling ICB arms all log; this one only bumped the
                            // counter (ICB audit) — name the reason.
                            crate::observe::fail(format!(
                                "render_icb fail reason=exec_range_read args_ref={} args_off={}",
                                exec.args_buffer_ref, exec.args_buffer_offset
                            ));
                            out.render_icb_fail += 1;
                            dirty_color_targets(state, host, task_id, &acc.color_targets);
                            return;
                        }
                    }
                };
                match metal_draw::encode_icb_execute_and_writeback(
                    state,
                    host,
                    &req,
                    exec.icb_ref,
                    loc,
                    len,
                ) {
                    EncodeStatus::Ok => out.render_icb_ok += 1,
                    st => {
                        out.render_icb_fail += 1;
                        // Was `st={st:?}` — the variant, Debug-rendered, with no
                        // `reason=` at all, so ten distinct checks in
                        // `encode_icb_execute_and_writeback` (plus every ICB
                        // refusal forwarded into it) shared four names and none
                        // of them was greppable. Latched per ICB: the guest
                        // re-executes the same one every frame.
                        if let Some(e) = crate::observe::Emit::refusal("render_icb", &st) {
                            e.field("icb_ref", exec.icb_ref)
                                .field("loc", loc)
                                .field("len", len)
                                .field("colors", acc.color_slots.len())
                                .fail_once(exec.icb_ref as u64);
                        }
                        dirty_color_targets(state, host, task_id, &acc.color_targets);
                    }
                }
            } else {
                out.render_icb_fail += 1;
                crate::observe::fail(format!(
                    "render_icb fail reason=mrt_request icb_ref={} colors={}",
                    exec.icb_ref,
                    acc.color_slots.len()
                ));
            }
        } else {
            out.render_icb_fail += 1;
            crate::observe::fail("render_icb fail reason=no_color_slots");
        }
        // ICB execute is the primary work; still allow a co-recorded draw below if present.
    }

    if acc.saw_draw && !acc.color_slots.is_empty() && !acc.draws.is_empty() {
        // Archive multi-draw (apple-pv-gpu-exec DrawJob): every honorable draw of
        // one exec packet targets one surface in decode order; the worker threads
        // each record's RGBA output as the next record's initial content; guest
        // writeback + completion stamp happen once for the final image.
        //
        // Chain in-process color0 RGBA8 between encodes (no float16 guest round-
        // trip between draws). Only the last successful encode stores to guest.
        let draw_list: Vec<&PendingDraw> = acc
            .draws
            .iter()
            .filter(|pd| pd.pipeline_ref != 0 && pd.draw.0 > 0)
            .collect();
        let mut chain_rgba: Option<Vec<u8>> = None;
        // Resident render-pass chain: intermediate records keep their content
        // on the engine target (no CPU chain buffer); records 2+ LoadFromTarget.
        let mut resident_chain = false;
        let mut saw_nometal = false;
        let first_draw = draw_list.first().copied();
        let mut first_req = first_draw.and_then(|pd| {
            let (count, inst, prim, first) = pd.draw;
            out.render_attachment_resolves = out.render_attachment_resolves.saturating_add(1);
            metal_draw::mrt_draw_request(
                state,
                host,
                task_id,
                pd.pipeline_ref,
                &acc.color_slots,
                &acc.clears,
                count,
                inst,
                prim,
                first,
            )
        });
        // A serialized Metal render stream is one render pass: its attachment
        // descriptors are fixed while pipeline, binds, and draw arguments may
        // change per record. Keep a seedless template so records 2+ do not
        // re-walk the same guest object list/page tables (or clone a full-frame
        // GVA LOAD seed). The resident target itself preserves record order.
        let attachment_template = first_req.as_ref().map(render_pass_attachment_template);
        if first_draw.is_some() && first_req.is_none() {
            let refs: Vec<u32> = acc.color_slots.iter().map(|(_, a)| a.texture_ref).collect();
            crate::observe::fail(format!(
                "metal_draw mrt_request fail task={task_id} pipe={} slots={refs:?} di=0/{}",
                first_draw.map(|pd| pd.pipeline_ref).unwrap_or(0),
                draw_list.len()
            ));
            out.metal_draws_fail = out.metal_draws_fail.saturating_add(1);
            dirty_color_targets(state, host, task_id, &acc.color_targets);
        }
        for (di, pd) in draw_list.iter().enumerate() {
            let mut req = if di == 0 {
                let Some(req) = first_req.take() else {
                    break;
                };
                req
            } else {
                let Some(template) = attachment_template.as_ref() else {
                    break;
                };
                retarget_render_pass_draw(template, pd)
            };
            {
                fill_draw_binds_from_pending(&mut req, pd);
                // A resident type-11 target carries attachment contents between
                // records without a CPU chain buffer. Like a native Metal render
                // pass, only the final record performs the guest-visible Store;
                // importing a full frame after every draw held DeviceInner for
                // seconds and starved the guest completion/status registers.
                let unified = req
                    .colors
                    .first()
                    .map(|c| c.mapping_id != 0)
                    .unwrap_or(false);
                // Records 2+ of a chain composite over the prior record: force
                // loadAction=Load on every color. Leaving the pass action on a
                // guest-backed target let a CLEAR re-run before each record,
                // wiping the full composite drawn by record 1 (live poison=1:
                // mid peak 10.9M native → 2.5M after later records).
                if di > 0 {
                    for c in &mut req.colors {
                        c.load_action = PASS_LOAD_ACTION_LOAD;
                    }
                    // Chain from the engine resident when available; otherwise
                    // seed from the prior encode output (archive "thread each
                    // record's output as next initial content"). MoltenVK's
                    // portability path returns CPU pixels for type-11 mappings,
                    // so `unified` does not imply that a resident exists.
                    // Moved, not cloned (multi-MiB).
                    match multi_draw_chain_source(resident_chain, chain_rgba.is_some()) {
                        MultiDrawChainSource::Resident => {
                            req.chain_from_resident = true;
                        }
                        MultiDrawChainSource::Cpu => {
                            if let Some(c0) = req.colors.first_mut() {
                                c0.target_seed_rgba = chain_rgba.take();
                            }
                        }
                        MultiDrawChainSource::Missing => {
                            crate::observe::fail(format!(
                                "multi_draw_chain_break reason=prior_output_missing \
                                 task={task_id} pipe={} di={di}/{} unified={}",
                                pd.pipeline_ref,
                                draw_list.len(),
                                unified as u8
                            ));
                        }
                    }
                }
                let (do_writeback, force_full_store) = multi_draw_store_plan(draw_list.len(), di);
                if do_writeback {
                    out.render_guest_stores = out.render_guest_stores.saturating_add(1);
                }
                let encode = metal_draw::encode_draw_chain(
                    state,
                    host,
                    &mut req,
                    do_writeback,
                    force_full_store,
                );
                match encode {
                    (EncodeStatus::Ok, Some(rgba)) => {
                        out.metal_draws_ok += 1;
                        if !resident_chain {
                            chain_rgba = Some(rgba);
                        }
                    }
                    (EncodeStatus::Ok, None) if req.chain_resident_established => {
                        // Resident render-pass chain intermediate: content stays
                        // on the engine target; the next record loads it there.
                        out.metal_draws_ok += 1;
                        resident_chain = true;
                    }
                    (EncodeStatus::Ok, None) => {
                        // Intermediate must return color0 for chaining; treat as
                        // break so we do not composite later draws on a missing seed.
                        out.metal_draws_ok += 1;
                        if !do_writeback && !unified {
                            // Land any earlier chain image before abandoning —
                            // same as the hard-fail path below. Dropping the
                            // chain left dual-mid pages black while gen advanced.
                            #[cfg(feature = "backend-vulkan")]
                            if resident_chain && chain_rgba.is_none() {
                                chain_rgba = metal_draw::read_resident_chain(state, &req);
                            }
                            if let Some(rgba) = chain_rgba.take() {
                                let _ = metal_draw::writeback_chain_rgba(
                                    state,
                                    host,
                                    task_id,
                                    &acc.color_slots,
                                    &rgba,
                                );
                            }
                            dirty_color_targets(state, host, task_id, &acc.color_targets);
                            break;
                        }
                    }
                    (st @ EncodeStatus::NoMetal(_), _) => {
                        saw_nometal = true;
                        out.metal_draws_fail += 1;
                        note_draw_encode_fail(task_id, pd.pipeline_ref, st, di, draw_list.len());
                        #[cfg(feature = "backend-vulkan")]
                        if resident_chain && chain_rgba.is_none() {
                            chain_rgba = metal_draw::read_resident_chain(state, &req);
                        }
                        if let Some(rgba) = chain_rgba.take() {
                            let _ = metal_draw::writeback_chain_rgba(
                                state,
                                host,
                                task_id,
                                &acc.color_slots,
                                &rgba,
                            );
                        }
                        dirty_color_targets(state, host, task_id, &acc.color_targets);
                        break;
                    }
                    // `Ok` and the distinct clear-fallback `NoMetal` recovery
                    // are exhausted above. Every remaining status is a typed
                    // terminal refusal, including the Metal-only carrier when
                    // that feature exists.
                    (st, _) => {
                        out.metal_draws_fail += 1;
                        note_draw_encode_fail(task_id, pd.pipeline_ref, st, di, draw_list.len());
                        // If earlier GVA draws produced a chain image, land it
                        // before abandoning the packet. Unified targets already
                        // landed each record in guest memory — never write the
                        // (zero) chain buffer over them.
                        #[cfg(feature = "backend-vulkan")]
                        if resident_chain && chain_rgba.is_none() {
                            chain_rgba = metal_draw::read_resident_chain(state, &req);
                        }
                        if let Some(rgba) = chain_rgba.take() {
                            let _ = metal_draw::writeback_chain_rgba(
                                state,
                                host,
                                task_id,
                                &acc.color_slots,
                                &rgba,
                            );
                        }
                        dirty_color_targets(state, host, task_id, &acc.color_targets);
                        break;
                    }
                }
            }
        }
        // Encode never landed Stores (NoMetal stubs, missing MTLB/pipeline, or
        // mrt resolve fail). Honor CLEAR load+store into guest/host pages so
        // dual-buffer display mids at least hold the pass clear color (archive
        // CLEAR seed — not a content heuristic). Applies for any draw-fail
        // class, not only NoMetal: mrt_request fail used to skip this and left
        // mid pages empty → nz_swing thrash on x86 Linux product.
        if out.metal_draws_ok == 0 && !acc.clears.is_empty() {
            for att in &acc.clears {
                if apply_clear(state, host, task_id, att) {
                    out.clears_applied = out.clears_applied.saturating_add(1);
                }
            }
            if out.clears_applied > 0 || saw_nometal || out.metal_draws_fail > 0 {
                crate::observe::fail(format!(
                    "draw_fail_clear_fallback task={task_id} clears={} draws_fail={} nometal={}",
                    out.clears_applied, out.metal_draws_fail, saw_nometal as u8
                ));
            }
        }
    }
}

/// One-shot (per `pipeline_ref` x reason) always-on line for a failed draw
/// encode. `exec_indirect2 draws_fail=N` collapses every cause into one
/// counter with no reason; a persistently failing draw (e.g. an app window
/// layer that never paints) was invisible on a normal boot. The latch keys
/// on the pipeline so a new failing workload logs its own line while a
/// steady repeat (same pipeline failing every packet) stays at one line.
///
/// The `reason=` was the *variant* name until `EncodeStatus` carried its check:
/// six names for the rail's 27 refusals, so `reason=bad_args` could be a
/// zero-size target, a vertexless draw or an unresolvable MRT slot. Now the
/// variant prints as `class=` beside the check that produced it.
fn note_draw_encode_fail(
    task_id: u32,
    pipeline_ref: u32,
    status: EncodeStatus,
    di: usize,
    n: usize,
) {
    if let Some(e) = crate::observe::Emit::refusal("draw_encode_fail", &status) {
        e.field("pipe", pipeline_ref)
            .field("task", task_id)
            .field("di", format!("{di}/{n}"))
            .fail_once(pipeline_ref as u64);
    }
}

/// Seedless fixed-attachment template for records after the first draw in one
/// serialized Metal render pass. Construct fields explicitly so a multi-MiB
/// CPU LOAD seed is not cloned merely to reuse attachment identity/geometry.
fn render_pass_attachment_template(
    first: &metal_draw::DrawEncodeRequest,
) -> metal_draw::DrawEncodeRequest {
    let colors = first
        .colors
        .iter()
        .map(|c| metal_draw::ColorRtRequest {
            slot: c.slot,
            texture_ref: c.texture_ref,
            mapping_id: c.mapping_id,
            target_gva: c.target_gva,
            row_stride: c.row_stride,
            width: c.width,
            height: c.height,
            format: c.format,
            load_action: PASS_LOAD_ACTION_LOAD,
            store_action: c.store_action,
            clear_color: c.clear_color,
            target_seed_rgba: None,
        })
        .collect();
    metal_draw::DrawEncodeRequest {
        task_id: first.task_id,
        color_texture_ref: first.color_texture_ref,
        mapping_id: first.mapping_id,
        width: first.width,
        height: first.height,
        format: first.format,
        colors,
        ..Default::default()
    }
}

fn retarget_render_pass_draw(
    template: &metal_draw::DrawEncodeRequest,
    draw: &PendingDraw,
) -> metal_draw::DrawEncodeRequest {
    let (count, inst, prim, first) = draw.draw;
    let mut req = template.clone();
    req.pipeline_ref = draw.pipeline_ref;
    req.vertex_count = count;
    req.instance_count = inst;
    req.primitive_type = prim;
    req.first_vertex = first;
    req
}

fn read_icb_exec_range<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    buffer_ref: u32,
    offset: u64,
) -> Option<(u64, u64)> {
    use crate::runtime::compute_exec::read_buffer_window;
    let raw = read_buffer_window(state, host, task_id, buffer_ref, offset, 8).ok()?;
    let loc = u32::from_le_bytes(raw[0..4].try_into().ok()?) as u64;
    let len = u32::from_le_bytes(raw[4..8].try_into().ok()?) as u64;
    Some((loc, len))
}

/// Guest store plan for multi-draw record `di` of `draw_count` (0-based).
///
/// Archive DrawJob: one writeback of the final image. Multi-draw builds that
/// image in host memory; the last record must full-frame store even if its
/// scissor is partial (else wallpaper chained earlier never reaches guest).
pub(crate) fn multi_draw_store_plan(draw_count: usize, di: usize) -> (bool, bool) {
    if draw_count == 0 {
        return (false, false);
    }
    let last_i = draw_count - 1;
    let do_writeback = di == last_i;
    let force_full_store = do_writeback && draw_count > 1;
    (do_writeback, force_full_store)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MultiDrawChainSource {
    Resident,
    Cpu,
    Missing,
}

fn multi_draw_chain_source(resident_chain: bool, cpu_chain_ready: bool) -> MultiDrawChainSource {
    if resident_chain {
        MultiDrawChainSource::Resident
    } else if cpu_chain_ready {
        MultiDrawChainSource::Cpu
    } else {
        MultiDrawChainSource::Missing
    }
}

fn fill_draw_binds_from_pending(req: &mut metal_draw::DrawEncodeRequest, pd: &PendingDraw) {
    req.vertex_buffers = pd.vertex_buffers.clone();
    req.fragment_buffers = pd.fragment_buffers.clone();
    req.vertex_textures = pd.vertex_textures.clone();
    req.fragment_textures = pd.fragment_textures.clone();
    req.vertex_samplers = pd.vertex_samplers.clone();
    req.fragment_samplers = pd.fragment_samplers.clone();
    req.viewport = pd.viewport;
    req.scissor = pd.scissor;
    req.indexed = pd.indexed.clone();
    req.blend_color = pd.blend_color;
    req.cull_mode = pd.cull_mode;
    req.front_facing = pd.front_facing;
    req.depth_bias = pd.depth_bias;
    req.depth_stencil_ref = pd.depth_stencil_ref;
    req.stencil_ref = pd.stencil_ref;
    req.depth_attach = pd.depth_attach;
    req.stencil_attach = pd.stencil_attach;
}

// solid_rgba remains used by metal_draw via clears; keep helper for tests if needed.
#[allow(dead_code)]
fn _solid_rgba_keep(w: u32, h: u32, clear: &[f64; 4]) -> Vec<u8> {
    solid_rgba(w, h, clear)
}

fn dirty_color_targets<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &M,
    task_id: u32,
    refs: &[u32],
) {
    for &tex_ref in refs {
        if let Some(mid) = objects::resolve_type11_ref(state, host, task_id, tex_ref) {
            // Unified memory: the mapping's texture aliases guest pages, so
            // there is no mirror to drop — only bump gen for scanout skips.
            let _ = state.mark_mapping_written(mid);
        } else if objects::resolve_type4_surface(state, host, tex_ref) {
            let _ = state.mark_mapping_written(tex_ref);
        }
    }
}

fn solid_rgba(w: u32, h: u32, clear: &[f64; 4]) -> Vec<u8> {
    let r = f64_to_unorm8(clear[0]);
    let g = f64_to_unorm8(clear[1]);
    let b = f64_to_unorm8(clear[2]);
    let a = f64_to_unorm8(clear[3]);
    let px = [r, g, b, a];
    let n = (w as usize).saturating_mul(h as usize).saturating_mul(4);
    let mut img = vec![0u8; n];
    for i in 0..(w * h) as usize {
        img[i * 4..i * 4 + 4].copy_from_slice(&px);
    }
    img
}

/// Deduped, fail-visible record of a guest clear directive we did not honor.
/// Keyed by `(reason, texture_ref)` so a persistent condition logs exactly once
/// instead of per stream — no flood. Runs on the drain worker (off the QEMU
/// main core) via the always-on `observe::fail` sink. Returns `true` the first
/// time a given `(reason, tex_ref)` is seen (the call that emitted the line).
fn note_clear_dropped(reason: &'static str, tex_ref: u32, detail: &str) -> bool {
    use std::collections::HashSet;
    use std::sync::Mutex;
    static SEEN: Mutex<Option<HashSet<(&'static str, u32)>>> = Mutex::new(None);
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    let first = seen
        .get_or_insert_with(HashSet::new)
        .insert((reason, tex_ref));
    if first {
        crate::observe::fail(format!(
            "clear_dropped reason={reason} tex_ref={tex_ref} {detail}"
        ));
    }
    first
}

fn apply_clear<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    att: &ColorAttachment,
) -> bool {
    if att.texture_ref == 0 || att.store_action != PASS_STORE_ACTION_STORE {
        return false;
    }
    // Prefer full draw-path resolve (type-11 or type-2/3 GVA wallpaper targets).
    let Some(req) =
        metal_draw::color_target_request(state, host, task_id, att.texture_ref, 0, 0, 1, 0, 0)
    else {
        // A clear whose color target cannot resolve (mapping unresolved, geometry
        // missing) is dropped here with no other trace — the "background didn't
        // clear cleanly" class. Make it visible, deduped per target.
        note_clear_dropped(
            "target_unresolved",
            att.texture_ref,
            "color_target_request=none",
        );
        return false;
    };
    let c0 = req.colors.first().unwrap_or_else(|| unreachable!());
    let w = c0.width;
    let h = c0.height;
    let rgba = solid_rgba(w, h, &att.clear_color);
    if c0.target_gva != 0 {
        metal_draw::supersede_gva_window(state, host, c0.target_gva, w, h, "clear_store");
        return metal_draw::write_gva_rgba8(
            state,
            host,
            task_id,
            c0.target_gva,
            w,
            h,
            c0.row_stride,
            c0.format,
            &rgba,
        )
        .is_ok();
    }
    if c0.mapping_id == 0 {
        return false;
    }
    let r = f64_to_unorm8(att.clear_color[0]);
    let g = f64_to_unorm8(att.clear_color[1]);
    let b = f64_to_unorm8(att.clear_color[2]);
    let a = f64_to_unorm8(att.clear_color[3]);
    let px = [b, g, r, a];
    let stride = w.saturating_mul(RGBA8_BPP);
    let mut img = vec![0u8; (stride as usize).saturating_mul(h as usize)];
    for y in 0..h as usize {
        for x in 0..w as usize {
            let o = y * stride as usize + x * 4;
            img[o..o + 4].copy_from_slice(&px);
        }
    }
    let _ = MTL_FORMAT_BGRA8_UNORM;
    let ok = mapping_write::write_bgra8(state, host, c0.mapping_id, &img, stride, w, h);
    // host_cache also updated inside write_bgra8 (surface_cache::store).
    state.note_surface_clear(c0.mapping_id);
    ok
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::contract::endian::{st16, st32, st64};
    use crate::model::{DeviceId, PAGE_SHIFT_ARM64E, PAGE_SHIFT_X86};
    use crate::runtime::decode::render::{
        HEADER_LEN, OP_RENDER_PASS, PASS_ATTACH_CLEAR_COLOR, PASS_ATTACH_LOAD_ACTION,
        PASS_ATTACH_STORE_ACTION, PASS_ATTACH_TEXREF, PASS_COLOR_ATTACH_OFF,
        PASS_COLOR_ATTACH_STRIDE, PASS_LOAD_ACTION_CLEAR, PASS_STORE_ACTION_STORE,
    };
    use crate::runtime::host::FakeHost;

    #[test]
    fn short_payload_noop() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let mut host = FakeHost::new();
        let r = process_exec_indirect2(&mut state, &mut host, &[0u8; 4]);
        assert_eq!(r.streams_loaded, 0);
    }

    /// An exec packet naming a slot that is not live must be refused under the
    /// word the guest sent, not silently re-aimed at slot `word >> 1`.
    ///
    /// Slot 3 is live and slot 6 is not, so word `6` names a dead slot whose
    /// halved form is live — the exact ambiguity the two boots that justified
    /// this deletion measured on every single exec decode. The old fallback
    /// answered `3` here, and `3` is a different task: everything the packet
    /// goes on to do, including its guest writes, would run against page tables
    /// the guest never named for this work.
    ///
    /// `task_id` is the separator because it is what the crate acts as and what
    /// `exec_summary` reports. Asserting only "no streams loaded" would pass
    /// either way — with no page tables mapped nothing loads regardless, which
    /// is a probe that cannot distinguish the cases.
    #[test]
    fn an_exec_packet_naming_a_dead_slot_is_refused_not_aimed_at_its_neighbour() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        assert!(state.define_task(3, 0x1_0000, 2), "slot 3 must be live");
        assert!(state.tasks[3].active);
        assert!(!state.tasks[6].active, "slot 6 must be dead for this to bite");

        let mut payload = vec![0u8; CHILD_EXEC_INDIRECT_HEADER_LEN as usize];
        st32(&mut payload[CHILD_EXEC_INDIRECT_TASK_ID as usize..], 6);
        st32(&mut payload[CHILD_EXEC_INDIRECT_CMDBUF_COUNT as usize..], 1);

        let r = process_exec_indirect2(&mut state, &mut host, &payload);
        assert_eq!(
            r.task_id, 6,
            "the refusal must name the word the guest sent, not the slot we \
             would have substituted"
        );
        assert_eq!(r.streams_loaded, 0);
        assert!(!r.saw_draw);
    }

    /// One segment header whose declared length runs `overshoot` bytes past the
    /// buffer, followed by `tail` bytes of would-be records.
    fn truncated_segment(type_: u8, overshoot: usize, tail: usize) -> Vec<u8> {
        use crate::runtime::decode::stream::SEGMENT_HEADER_LEN;
        let mut stream = vec![0u8; SEGMENT_HEADER_LEN + tail];
        st32(
            &mut stream[0..4],
            (SEGMENT_HEADER_LEN + tail + overshoot) as u32,
        );
        stream[4] = type_;
        stream
    }

    fn sink_body() -> String {
        std::fs::read_to_string(crate::observe::fail_log_path()).unwrap_or_default()
    }

    #[test]
    fn a_stream_that_will_not_frame_says_so_instead_of_executing_nothing() {
        use crate::runtime::decode::stream::SEGMENT_TYPE_RENDER;
        // The defect this pins: `walk_stream` opened with `Err(_) => return`, so a
        // stream the framing decoder rejected executed zero records and produced
        // zero log lines — byte-for-byte indistinguishable at the sink from an
        // idle guest that submitted nothing.
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let mut host = FakeHost::new();
        let mut out = ExecResult::default();
        let mut acc = StreamAccum::default();
        let before = sink_body().len();
        // Task id doubles as the flood-latch discriminant, so it must be one no
        // other test in this process has already burned.
        let task_id = 0x5731_0001;
        walk_stream(
            &mut state,
            &mut host,
            task_id,
            &truncated_segment(SEGMENT_TYPE_RENDER, 64, 0),
            &mut out,
            &mut acc,
        );
        let added = sink_body()[before..].to_string();
        assert!(
            added.contains("stream_frame_fail"),
            "a stream that will not frame must reach the always-on sink, got:\n{added}"
        );
        assert!(
            added.contains("reason=stream_seg_len_past_buffer_end"),
            "the line must name which framing check refused, not just that one \
             did — 17 checks shared `ErrBadLength`. got:\n{added}"
        );
        assert!(
            added.contains(&format!("task={task_id}")),
            "the line must carry the task whose work was dropped, got:\n{added}"
        );
    }

    #[test]
    fn a_truncated_segment_names_the_check_rather_than_looking_like_end_of_records() {
        use crate::runtime::decode::stream::{
            segment_type_name, Segment, SEGMENT_HEADER_LEN, SEGMENT_TYPE_INFO,
        };
        // `Err(_) => break` treated a self-inconsistent segment exactly like
        // `Done`: the remaining records went unexecuted with nothing logged.
        let stream = vec![0u8; SEGMENT_HEADER_LEN + 4];
        // A segment claiming a longer body than the buffer holds, handed straight
        // to the record walker — the shape `iter_segments` would have rejected but
        // that an already-parsed `Segment` can still carry.
        let seg = Segment {
            offset: 0,
            length: (SEGMENT_HEADER_LEN + 64) as u32,
            type_: SEGMENT_TYPE_INFO,
            command_offset: SEGMENT_HEADER_LEN as u32,
            command_length: 64,
            ..Segment::default()
        };
        let before = sink_body().len();
        let mut handled = 0usize;
        walk_segment_records(&stream, &seg, |_| handled += 1);
        let added = sink_body()[before..].to_string();
        assert_eq!(handled, 0, "the malformed segment yields no records");
        assert!(
            added.contains("stream_record_fail"),
            "dropping a segment's records must reach the sink, got:\n{added}"
        );
        assert!(
            added.contains("reason=stream_reval_span_oob"),
            "the line must name the failing re-validation check, got:\n{added}"
        );
        assert!(
            added.contains(&format!(
                "seg={}",
                segment_type_name(u32::from(SEGMENT_TYPE_INFO))
            )),
            "the line must say which segment family lost its records, got:\n{added}"
        );
    }

    #[test]
    fn walking_a_well_formed_segment_to_its_end_logs_nothing() {
        use crate::runtime::decode::stream::{
            iter_segments, SEGMENT_HEADER_LEN, SEGMENT_TYPE_EVENT,
        };
        // The other half of the obligation: `Done` is how every segment ends, so
        // if it produced a line the sink would carry one per segment per frame.
        let mut records = [0u8; 8];
        st32(&mut records[0..4], 0x190);
        st32(&mut records[4..8], 8);
        let mut stream = vec![0u8; SEGMENT_HEADER_LEN];
        st32(
            &mut stream[0..4],
            (SEGMENT_HEADER_LEN + records.len()) as u32,
        );
        stream[4] = SEGMENT_TYPE_EVENT;
        stream.extend_from_slice(&records);

        let segs = iter_segments(&stream).expect("a well-formed stream frames");
        let before = sink_body().len();
        let mut handled = 0usize;
        walk_segment_records(&stream, &segs[0], |_| handled += 1);
        let added = sink_body()[before..].to_string();
        assert_eq!(handled, 1, "the one record is handed over");
        assert!(
            !added.contains("stream_record_fail"),
            "end-of-segment is control flow and must stay out of the log, got:\n{added}"
        );
    }

    #[test]
    fn an_unknown_segment_family_is_refused_and_the_type_5_envelope_is_not() {
        use crate::observe::Refusal;
        use crate::runtime::decode::stream::{
            segment_disposition, SegmentDisposition, SEGMENT_TYPE_BLIT,
            SEGMENT_TYPE_PROTECTION_OPTIONS,
        };
        // `walk_stream` ended in `_ => {}`, which gave one silence to two very
        // different things. Type 5 is a contract-correct skip; type 6 is wire
        // format the host has never seen.
        assert_eq!(
            segment_disposition(SEGMENT_TYPE_PROTECTION_OPTIONS),
            SegmentDisposition::Envelope
        );
        assert_eq!(
            segment_disposition(SEGMENT_TYPE_PROTECTION_OPTIONS).refusal(),
            None,
            "the envelope arrives on healthy frames; a line here is a flood"
        );
        assert_eq!(
            segment_disposition(SEGMENT_TYPE_BLIT),
            SegmentDisposition::Walk
        );
        assert_eq!(
            segment_disposition(6).refusal(),
            Some("stream_segment_type_unknown")
        );
        assert_eq!(
            segment_disposition(0xff).refusal(),
            Some("stream_segment_type_unknown")
        );
    }

    #[cfg(feature = "backend-vulkan")]
    #[test]
    fn render_preflight_collects_content_pipelines_without_duplicates() {
        use crate::runtime::decode::render::OP_SET_PIPELINE;
        use crate::runtime::decode::stream::{SEGMENT_HEADER_LEN, SEGMENT_TYPE_RENDER};

        let mut records = Vec::new();
        for pipeline in [41u32, 77, 41] {
            let mut cmd = [0u8; 12];
            st32(&mut cmd[0..4], OP_SET_PIPELINE);
            st32(&mut cmd[4..8], 12);
            st32(&mut cmd[8..12], pipeline);
            records.extend_from_slice(&cmd);
        }
        let mut stream = vec![0u8; SEGMENT_HEADER_LEN];
        let stream_len = stream.len() + records.len();
        st32(&mut stream[0..4], stream_len as u32);
        stream[4] = SEGMENT_TYPE_RENDER;
        stream.extend_from_slice(&records);

        assert_eq!(render_pipeline_refs(&stream), vec![41, 77]);
    }

    #[cfg(feature = "backend-vulkan")]
    #[test]
    fn compute_preflight_collects_pipeline_and_local_size_without_duplicates() {
        use crate::runtime::decode::compute::{
            OP_DISPATCH_THREADGROUPS, OP_DISPATCH_THREADS, OP_SET_PIPELINE,
        };
        use crate::runtime::decode::stream::{SEGMENT_HEADER_LEN, SEGMENT_TYPE_COMPUTE};

        let mut records = Vec::new();
        let mut pipeline = [0u8; 12];
        st32(&mut pipeline[0..4], OP_SET_PIPELINE);
        st32(&mut pipeline[4..8], 12);
        st32(&mut pipeline[8..12], 20);
        records.extend_from_slice(&pipeline);
        for opcode in [
            OP_DISPATCH_THREADGROUPS,
            OP_DISPATCH_THREADGROUPS,
            OP_DISPATCH_THREADS,
        ] {
            let mut dispatch = [0u8; 56];
            st32(&mut dispatch[0..4], opcode);
            st32(&mut dispatch[4..8], 56);
            st64(&mut dispatch[8..16], 6);
            st64(&mut dispatch[16..24], 11);
            st64(&mut dispatch[24..32], 1);
            st64(&mut dispatch[32..40], 16);
            st64(&mut dispatch[40..48], 16);
            st64(&mut dispatch[48..56], 1);
            records.extend_from_slice(&dispatch);
        }
        let mut stream = vec![0u8; SEGMENT_HEADER_LEN];
        let stream_len = stream.len() + records.len();
        st32(&mut stream[0..4], stream_len as u32);
        stream[4] = SEGMENT_TYPE_COMPUTE;
        stream.extend_from_slice(&records);

        assert_eq!(compute_translation_inputs(&stream), vec![(20, [16, 16, 1])]);
    }

    #[test]
    fn event_segment_signal_wait_in_stream() {
        use crate::model::FENCE_DOMAIN_EVENT;
        use crate::runtime::decode::event::{
            OP_SIGNAL_EVENT, OP_WAIT_EVENT, SIGNAL_WAIT_PAYLOAD_LEN,
        };
        use crate::runtime::decode::stream::{SEGMENT_HEADER_LEN, SEGMENT_TYPE_EVENT};

        fn push_segment(buf: &mut Vec<u8>, type_: u8, payload: &[u8]) {
            let len = (SEGMENT_HEADER_LEN + payload.len()) as u32;
            let mut hdr = [0u8; 8];
            st32(&mut hdr[0..4], len);
            hdr[4] = type_;
            buf.extend_from_slice(&hdr);
            buf.extend_from_slice(payload);
        }
        fn push_event_record(buf: &mut Vec<u8>, opcode: u32, event_ref: u32, value: u64) {
            let mut payload = [0u8; SIGNAL_WAIT_PAYLOAD_LEN];
            st32(&mut payload[0..4], event_ref);
            st64(&mut payload[4..12], value);
            let len = (HEADER_LEN + SIGNAL_WAIT_PAYLOAD_LEN) as u32;
            let mut hdr = [0u8; 8];
            st32(&mut hdr[0..4], opcode);
            st32(&mut hdr[4..8], len);
            buf.extend_from_slice(&hdr);
            buf.extend_from_slice(&payload);
        }

        let mut records = Vec::new();
        push_event_record(&mut records, OP_SIGNAL_EVENT, 11, 7);
        push_event_record(&mut records, OP_WAIT_EVENT, 11, 7);
        push_event_record(&mut records, OP_WAIT_EVENT, 11, 8); // pending
        let mut stream = Vec::new();
        push_segment(&mut stream, SEGMENT_TYPE_EVENT, &records);

        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let mut host = FakeHost::new();
        let mut out = ExecResult::default();
        let mut acc = StreamAccum::default();
        walk_stream(&mut state, &mut host, 1, &stream, &mut out, &mut acc);

        assert_eq!(out.event_ops_ok, 2);
        assert_eq!(out.event_ops_pending, 1);
        assert_eq!(out.event_ops_fail, 0);
        assert_eq!(state.fence_generation(1, FENCE_DOMAIN_EVENT, 11), Some(7));
    }

    #[test]
    fn multi_attachment_decode_in_pass() {
        let mut payload = vec![0u8; PASS_COLOR_ATTACH_OFF + PASS_COLOR_ATTACH_STRIDE * 2];
        for (i, tex) in [(0u32, 41u32), (1u32, 42u32)] {
            let slot = PASS_COLOR_ATTACH_OFF + i as usize * PASS_COLOR_ATTACH_STRIDE;
            st32(&mut payload[slot + PASS_ATTACH_TEXREF..], tex);
            st16(
                &mut payload[slot + PASS_ATTACH_LOAD_ACTION..],
                PASS_LOAD_ACTION_CLEAR,
            );
            st16(
                &mut payload[slot + PASS_ATTACH_STORE_ACTION..],
                PASS_STORE_ACTION_STORE,
            );
            st64(
                &mut payload[slot + PASS_ATTACH_CLEAR_COLOR..],
                1.0f64.to_bits(),
            );
            st64(
                &mut payload[slot + PASS_ATTACH_CLEAR_COLOR + 8..],
                0.0f64.to_bits(),
            );
            st64(
                &mut payload[slot + PASS_ATTACH_CLEAR_COLOR + 16..],
                0.0f64.to_bits(),
            );
            st64(
                &mut payload[slot + PASS_ATTACH_CLEAR_COLOR + 24..],
                1.0f64.to_bits(),
            );
        }
        let a0 = decode_color_attachment(&payload, 0);
        let a1 = decode_color_attachment(&payload, 1);
        assert_eq!(a0.texture_ref, 41);
        assert_eq!(a1.texture_ref, 42);
        let mut cmd = vec![0u8; HEADER_LEN + payload.len()];
        st32(&mut cmd[0..], OP_RENDER_PASS);
        st32(&mut cmd[4..], (HEADER_LEN + payload.len()) as u32);
        cmd[HEADER_LEN..].copy_from_slice(&payload);
        let c = render::decode(&cmd).unwrap();
        assert_eq!(c.kind, RenderKind::RenderPass);
        assert_eq!(c.color0.texture_ref, 41);
    }

    #[test]
    fn stream_accum_upserts_buffer_and_viewport() {
        use crate::runtime::decode::render::{
            OP_SET_FRAGMENT_BUFFER, OP_SET_VERTEX_BUFFER, OP_SET_VIEWPORT,
        };
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let host = FakeHost::new();
        let mut out = ExecResult::default();
        let mut acc = StreamAccum::default();

        let rec = |len: usize, opcode: u32| stream::Record {
            segment_index: 0,
            segment_type: 0,
            offset: 0,
            length: len as u32,
            opcode,
            bytes_offset: 0,
        };

        // setVertexBuffer multi-entry: first=2 count=1 ref=9 offset=16
        // payload = first:u32 + count:u32 + {ref:u32, offset:u64}
        let mut vb = vec![0u8; HEADER_LEN + 8 + 12];
        let vb_len = vb.len() as u32;
        st32(&mut vb[0..], OP_SET_VERTEX_BUFFER);
        st32(&mut vb[4..], vb_len);
        st32(&mut vb[8..], 2); // first
        st32(&mut vb[12..], 1); // count
        st32(&mut vb[16..], 9); // ref
        st64(&mut vb[20..], 16); // offset
        handle_render_record(
            &mut state,
            &host,
            0,
            &vb,
            &rec(vb.len(), OP_SET_VERTEX_BUFFER),
            &mut out,
            &mut acc,
        );
        assert_eq!(acc.vertex_buffers.len(), 1);
        assert_eq!(acc.vertex_buffers[0].index, 2);
        assert_eq!(acc.vertex_buffers[0].buffer_ref, 9);
        assert_eq!(acc.vertex_buffers[0].offset, 16);

        // overwrite same slot
        st32(&mut vb[16..], 10);
        handle_render_record(
            &mut state,
            &host,
            0,
            &vb,
            &rec(vb.len(), OP_SET_VERTEX_BUFFER),
            &mut out,
            &mut acc,
        );
        assert_eq!(acc.vertex_buffers.len(), 1);
        assert_eq!(acc.vertex_buffers[0].buffer_ref, 10);

        // fragment buffer multi-entry: first=0 count=1 ref=7 offset=0
        let mut fb = vec![0u8; HEADER_LEN + 8 + 12];
        let fb_len = fb.len() as u32;
        st32(&mut fb[0..], OP_SET_FRAGMENT_BUFFER);
        st32(&mut fb[4..], fb_len);
        st32(&mut fb[8..], 0); // first
        st32(&mut fb[12..], 1); // count
        st32(&mut fb[16..], 7); // ref
        st64(&mut fb[20..], 0); // offset
        handle_render_record(
            &mut state,
            &host,
            0,
            &fb,
            &rec(fb.len(), OP_SET_FRAGMENT_BUFFER),
            &mut out,
            &mut acc,
        );
        assert_eq!(acc.fragment_buffers.len(), 1);
        assert_eq!(out.buffer_binds, 2);

        // viewport
        let mut vp = vec![0u8; HEADER_LEN + 48];
        st32(&mut vp[0..], OP_SET_VIEWPORT);
        st32(&mut vp[4..], (HEADER_LEN + 48) as u32);
        for i in 0..6 {
            let bits = (i as f64 + 1.0).to_bits();
            st64(&mut vp[HEADER_LEN + i * 8..], bits);
        }
        handle_render_record(
            &mut state,
            &host,
            0,
            &vp,
            &rec(vp.len(), OP_SET_VIEWPORT),
            &mut out,
            &mut acc,
        );
        let v = acc.viewport.expect("viewport");
        assert!((v[0] - 1.0).abs() < 1e-9);
        assert!((v[5] - 6.0).abs() < 1e-9);
    }

    #[test]
    fn wide_indexed_draw_reaches_pending_draw() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let host = FakeHost::new();
        let mut out = ExecResult::default();
        let mut acc = StreamAccum {
            pipeline_ref: 61,
            ..Default::default()
        };
        let mut command = vec![0u8; 0x20];
        st32(&mut command[0..], render::OP_DRAW_INDEXED_WIDE);
        st32(&mut command[4..], 0x20);
        st16(&mut command[8..], 3);
        st16(&mut command[10..], 0);
        st32(&mut command[12..], 0x3e);
        st32(&mut command[16..], 6);
        st32(&mut command[24..], 0x10100);
        let rec = stream::Record {
            segment_index: 0,
            segment_type: 0,
            offset: 0,
            length: command.len() as u32,
            opcode: render::OP_DRAW_INDEXED_WIDE,
            bytes_offset: 0,
        };

        handle_render_record(&mut state, &host, 1, &command, &rec, &mut out, &mut acc);

        assert!(acc.saw_draw);
        assert!(out.saw_draw);
        assert_eq!(acc.draws.len(), 1);
        let indexed = acc.draws[0].indexed.as_ref().expect("indexed draw");
        assert_eq!(indexed.index_type, 0);
        assert_eq!(indexed.index_buffer_ref, 0x3e);
        assert_eq!(indexed.index_count, 6);
        assert_eq!(indexed.index_buffer_offset, 0x10100);
        assert_eq!(acc.draws[0].draw, (6, 1, 3, 0));
    }

    #[test]
    fn accepted_render_without_executor_is_fail_visible() {
        // The emit is deduped per opcode process-wide; hold the shared latch
        // lock and clear it so this test always observes its first-sighting line.
        let _guard = UNIMPL_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset_unimplemented_opcode_dedup_for_test();
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let host = FakeHost::new();
        let mut out = ExecResult::default();
        let mut acc = StreamAccum {
            pipeline_ref: 0xface,
            ..Default::default()
        };
        let task_id = 0xfeed;
        let mut command = vec![0u8; HEADER_LEN];
        st32(&mut command[0..], render::OP_ACCEPTED_LAST);
        st32(&mut command[4..], HEADER_LEN as u32);
        let rec = stream::Record {
            segment_index: 0,
            segment_type: 0,
            offset: 0,
            length: command.len() as u32,
            opcode: render::OP_ACCEPTED_LAST,
            bytes_offset: 0,
        };

        handle_render_record(
            &mut state, &host, task_id, &command, &rec, &mut out, &mut acc,
        );

        let body = std::fs::read_to_string(crate::observe::fail_log_path())
            .expect("reims-vgpu-fail.log readable");
        assert!(body.lines().any(|line| {
            line.contains(
                "render_unimplemented reason=accepted_without_executor task=65261 opcode=0x98 len=8",
            ) && line.contains("pipeline=64206")
        }));
    }

    /// Regression guard: the accepted-without-executor line is deduped to ONE
    /// emission per distinct opcode (a per-draw undecoded op must not flood the
    /// always-on sink), while distinct opcodes still each report once and the
    /// raw wire is captured. This locks the anti-flood behavior that replaced
    /// the ~2620-line-per-workload per-draw emit.
    #[test]
    fn unimplemented_render_opcode_dedups_per_opcode_with_wire() {
        let _guard = UNIMPL_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset_unimplemented_opcode_dedup_for_test();
        let task = 0x5151u32;
        let acc = StreamAccum {
            pipeline_ref: 0x1234,
            ..Default::default()
        };
        let wire: Vec<u8> = vec![0xde, 0xad, 0xbe, 0xef, 0x10, 0x00, 0x00, 0x00];

        // First sighting of an opcode emits; every repeat is deduped (no flood).
        assert!(
            note_unimplemented_render_opcode(0x7c, 16, &wire, task, &acc),
            "first sighting must emit",
        );
        for _ in 0..24 {
            assert!(
                !note_unimplemented_render_opcode(0x7c, 16, &wire, task, &acc),
                "a repeated opcode must be deduped",
            );
        }
        // A distinct opcode reports once independently of the first.
        assert!(note_unimplemented_render_opcode(
            0x9a, 24, &wire, task, &acc
        ));
        assert!(!note_unimplemented_render_opcode(
            0x9a, 24, &wire, task, &acc
        ));
        // Out-of-range opcodes (decode desync) are also deduped, not flooded.
        assert!(note_unimplemented_render_opcode(
            0x1_0001, 8, &wire, task, &acc
        ));
        assert!(!note_unimplemented_render_opcode(
            0x1_0001, 8, &wire, task, &acc
        ));

        // The first-sighting line captured the raw wire for offline decode.
        let body = std::fs::read_to_string(crate::observe::fail_log_path())
            .expect("reims-vgpu-fail.log readable");
        assert!(
            body.lines().any(|l| l.contains(&format!("task={task}"))
                && l.contains("opcode=0x7c")
                && l.contains("hex=deadbeef10000000")),
            "the raw wire must be captured on first sighting",
        );
    }

    /// The render rail's boundary counter must name the *check* that dropped the
    /// draw, not the class it was flattened into.
    ///
    /// Before `EncodeStatus` carried its reason this line read
    /// `draw_encode_fail reason=bad_args`, and `bad_args` alone spoke for eight
    /// distinct refusals in `encode_draw_chain_inner` — a zero-size target, a
    /// vertexless draw, an MRT slot with no backing. A window that never painted
    /// gave you the class and never the cause.
    #[test]
    fn a_dropped_draw_names_which_check_refused_not_just_its_class() {
        let task = 81u32;
        // Distinct from every other pipeline in the suite: `fail_once` latches per
        // (reason, pipeline) for the whole process.
        let pipe = 249_001u32;
        note_draw_encode_fail(
            task,
            pipe,
            EncodeStatus::BadArgs("draw_mtl_zero_geom"),
            1,
            3,
        );
        let body = sink_body();
        assert!(
            body.lines().any(|l| l
                .contains("draw_encode_fail reason=draw_mtl_zero_geom class=bad_args")
                && l.contains(&format!("pipe={pipe}"))
                && l.contains(&format!("task={task}"))
                && l.contains("di=1/3")),
            "the boundary line must carry the specific check and the class:\n{body}"
        );

        // Latched per (reason, pipeline): the guest re-submits the same failing
        // draw every frame, so a repeat adds nothing the first line did not…
        note_draw_encode_fail(
            task,
            pipe,
            EncodeStatus::BadArgs("draw_mtl_zero_geom"),
            2,
            3,
        );
        // …but a *different* check on the same pipeline is a different event and
        // must still be visible. Latching on the class would have hidden it, which
        // is exactly the failure this migration removes.
        note_draw_encode_fail(
            task,
            pipe,
            EncodeStatus::MetalFailed("draw_mtl_core_failed"),
            2,
            3,
        );
        let body = sink_body();
        assert_eq!(
            body.matches("reason=draw_mtl_zero_geom").count(),
            1,
            "a re-attempted refusal must log once:\n{body}"
        );
        assert!(
            body.contains("reason=draw_mtl_core_failed"),
            "a second check on the same pipeline must not be latched away:\n{body}"
        );

        // Success never reaches the sink — `Emit::refusal` has no line to send for
        // `Ok`, so the carve-out is enforced by the type rather than by a `return`
        // a future arm could forget.
        let before = sink_body().matches("draw_encode_fail").count();
        note_draw_encode_fail(task, pipe, EncodeStatus::Ok, 0, 1);
        assert_eq!(
            sink_body().matches("draw_encode_fail").count(),
            before,
            "an Ok encode logged a failure line"
        );
    }

    #[test]
    fn zero_ref_render_bind_unbinds_existing_slots() {
        use crate::runtime::decode::render::{
            OP_SET_FRAGMENT_SAMPLER, OP_SET_FRAGMENT_TEXTURE, OP_SET_VERTEX_BUFFER,
        };

        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let host = FakeHost::new();
        let mut out = ExecResult::default();
        let mut acc = StreamAccum::default();
        let rec = |len: usize, opcode: u32| stream::Record {
            segment_index: 0,
            segment_type: 0,
            offset: 0,
            length: len as u32,
            opcode,
            bytes_offset: 0,
        };

        let mut buffer = vec![0u8; HEADER_LEN + 8 + 12];
        st32(&mut buffer[0..], OP_SET_VERTEX_BUFFER);
        st32(&mut buffer[4..], (HEADER_LEN + 8 + 12) as u32);
        st32(&mut buffer[8..], 0);
        st32(&mut buffer[12..], 1);
        st32(&mut buffer[16..], 41);
        handle_render_record(
            &mut state,
            &host,
            0,
            &buffer,
            &rec(buffer.len(), OP_SET_VERTEX_BUFFER),
            &mut out,
            &mut acc,
        );
        st32(&mut buffer[16..], 0);
        handle_render_record(
            &mut state,
            &host,
            0,
            &buffer,
            &rec(buffer.len(), OP_SET_VERTEX_BUFFER),
            &mut out,
            &mut acc,
        );
        assert!(acc.vertex_buffers.is_empty());

        for (opcode, bound) in [
            (OP_SET_FRAGMENT_TEXTURE, 42u32),
            (OP_SET_FRAGMENT_SAMPLER, 43u32),
        ] {
            let mut command = vec![0u8; HEADER_LEN + 8 + 4];
            st32(&mut command[0..], opcode);
            st32(&mut command[4..], (HEADER_LEN + 8 + 4) as u32);
            st32(&mut command[8..], 3);
            st32(&mut command[12..], 1);
            st32(&mut command[16..], bound);
            handle_render_record(
                &mut state,
                &host,
                0,
                &command,
                &rec(command.len(), opcode),
                &mut out,
                &mut acc,
            );
            st32(&mut command[16..], 0);
            handle_render_record(
                &mut state,
                &host,
                0,
                &command,
                &rec(command.len(), opcode),
                &mut out,
                &mut acc,
            );
        }
        assert!(acc.fragment_textures.is_empty());
        assert!(acc.fragment_samplers.is_empty());
        assert_eq!(out.buffer_unbinds, 1);
        assert_eq!(out.texture_unbinds, 1);
        assert_eq!(out.sampler_unbinds, 1);
    }

    /// x86 type-4 display mid: clear-only stream must Store solid BGRA into pages.
    #[test]
    fn clear_only_type4_surface_writes_guest_pages() {
        use crate::contract::endian::{st32, st64};
        use crate::contract::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
        use crate::contract::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
        use crate::runtime::decode::render::ColorAttachment;
        use crate::runtime::objects::{self, OBJECT_TYPE_SURFACE};

        let mut host = FakeHost::new();
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        state.page_shift = PAGE_SHIFT_X86;
        // Identity-backed surface pages at pfn 0x40 (one 4K page enough for 16×16).
        let page = 0x40u64 << PAGE_SHIFT_X86;
        host.map_range(page, 0x2000, 0);
        // Task directory so object-list GVA reads work.
        let dir_gpa = 2u64 << PAGE_SHIFT_X86;
        let root_gpa = 3u64 << PAGE_SHIFT_X86;
        let data_gpa = 4u64 << PAGE_SHIFT_X86;
        host.map_range(dir_gpa, 0x20, 0);
        host.map_range(root_gpa, 0x1000, 0);
        host.map_range(data_gpa, 0x200, 0);
        let mut d = [0u8; 8];
        st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 3);
        st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
        let _ = host.write_gpa(dir_gpa, &d);
        st32(&mut d[..4], 4);
        let _ = host.write_gpa(root_gpa, &d[..4]);
        assert!(state.define_task(1, 0x1000, 2));
        assert!(state.set_object_list(1, 0, 8));
        // Type-4 at surface_id=5.
        let mut entry = [0u8; 12];
        st32(
            &mut entry[0..],
            (OBJECT_TYPE_SURFACE as u32) | (0x30u32 << 8),
        );
        entry[4..12].copy_from_slice(&0x80u64.to_le_bytes());
        let _ = host.write_gpa(data_gpa + 5 * 12, &entry);
        let mut desc = vec![0u8; 0x30];
        st64(&mut desc[0..], 0x1000);
        st32(&mut desc[8..], 0x40); // identity pfn
        st32(&mut desc[0xc..], 0x4247_5241); // 'BGRA'
        desc[0x10] = 1;
        st32(&mut desc[0x18..], 16);
        st32(&mut desc[0x1c..], 16);
        st32(&mut desc[0x20..], 64);
        let _ = host.write_gpa(data_gpa + 0x80, &desc);

        assert!(objects::resolve_type4_surface(&mut state, &host, 5));
        let mut out = ExecResult::default();
        let mut acc = StreamAccum::default();
        acc.clears.push(ColorAttachment {
            present: true,
            texture_ref: 5,
            resolve_texture_ref: 0,
            level: 0,
            load_action: PASS_LOAD_ACTION_CLEAR,
            store_action: PASS_STORE_ACTION_STORE,
            clear_color: [1.0, 0.0, 0.0, 1.0], // red → BGRA (0,0,255,255)
        });
        finish_stream(&mut state, &mut host, 1, &mut out, &acc);
        assert!(
            out.clears_applied >= 1,
            "type-4 clear must apply, got {}",
            out.clears_applied
        );
        // Read first pixel from guest page (BGRA).
        let mut px = [0u8; 4];
        assert!(host.read_gpa(page, &mut px).is_ok());
        assert_eq!(px, [0, 0, 255, 255], "expected opaque red BGRA, got {px:?}");
        let m = state.mappings.get(&5).expect("mapping");
        assert!(m.content_generation > 0 || m.mapped);
        let _ = PAGE_ENTRY_VALID;
        let _ = PAGE_ENTRY_PFN_SHIFT;
    }

    /// Archive DrawJob: clear-only packets store immediately; multi-draw packets
    /// keep CLEAR as private Metal seed (no pre-draw guest clear).
    #[test]
    fn finish_stream_clear_only_branch_without_draws() {
        use crate::runtime::decode::render::ColorAttachment;
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let mut host = FakeHost::new();
        let mut out = ExecResult::default();
        let mut acc = StreamAccum::default();
        acc.clears.push(ColorAttachment {
            present: true,
            texture_ref: 99,
            resolve_texture_ref: 0,
            level: 0,
            load_action: PASS_LOAD_ACTION_CLEAR,
            store_action: PASS_STORE_ACTION_STORE,
            clear_color: [0.0, 0.0, 0.0, 1.0],
        });
        // No draws → clear-only branch (attempts apply_clear; unresolvable ref).
        finish_stream(&mut state, &mut host, 1, &mut out, &acc);
        assert_eq!(out.metal_draws_ok, 0);
        assert_eq!(out.metal_draws_fail, 0);
    }

    #[test]
    fn finish_stream_with_draws_skips_guest_clear_prelude() {
        use crate::runtime::decode::render::ColorAttachment;
        use crate::runtime::metal_draw::BufferBind;
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let mut host = FakeHost::new();
        let mut out = ExecResult::default();
        let mut acc = StreamAccum::default();
        let att = ColorAttachment {
            present: true,
            texture_ref: 99,
            resolve_texture_ref: 0,
            level: 0,
            load_action: PASS_LOAD_ACTION_CLEAR,
            store_action: PASS_STORE_ACTION_STORE,
            clear_color: [1.0, 0.0, 0.0, 1.0],
        };
        acc.clears.push(att);
        acc.saw_draw = true;
        acc.color_slots.push((0, att));
        acc.draws.push(PendingDraw {
            pipeline_ref: 1,
            draw: (3, 1, 3, 0),
            indexed: None,
            vertex_buffers: vec![BufferBind {
                stage: Stage::Vertex,
                index: 0,
                buffer_ref: 1,
                offset: 0,
            }],
            fragment_buffers: Vec::new(),
            vertex_textures: Vec::new(),
            fragment_textures: Vec::new(),
            vertex_samplers: Vec::new(),
            fragment_samplers: Vec::new(),
            viewport: None,
            scissor: None,
            blend_color: None,
            cull_mode: None,
            front_facing: None,
            depth_bias: None,
            depth_stencil_ref: 0,
            stencil_ref: None,
            depth_attach: None,
            stencil_attach: None,
        });
        finish_stream(&mut state, &mut host, 1, &mut out, &acc);
        // Unresolvable RT → mrt_request fail before encode (not NoMetal); no clear.
        assert_eq!(
            out.clears_applied, 0,
            "unresolvable multi-draw must not guest-clear"
        );
    }

    /// Linux NoMetal: draws fail but CLEAR seed still Stores into type-4 pages.
    #[test]
    fn nometal_draw_falls_back_to_type4_clear() {
        use crate::contract::endian::{st32, st64};
        use crate::contract::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
        use crate::runtime::decode::render::ColorAttachment;
        use crate::runtime::metal_draw::BufferBind;
        use crate::runtime::objects::{self, OBJECT_TYPE_SURFACE};

        let mut host = FakeHost::new();
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        state.page_shift = PAGE_SHIFT_X86;
        let page = 0x50u64 << PAGE_SHIFT_X86;
        host.map_range(page, 0x2000, 0);
        let dir_gpa = 2u64 << PAGE_SHIFT_X86;
        let root_gpa = 3u64 << PAGE_SHIFT_X86;
        let data_gpa = 4u64 << PAGE_SHIFT_X86;
        host.map_range(dir_gpa, 0x20, 0);
        host.map_range(root_gpa, 0x1000, 0);
        host.map_range(data_gpa, 0x200, 0);
        let mut d = [0u8; 8];
        st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 3);
        st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
        let _ = host.write_gpa(dir_gpa, &d);
        st32(&mut d[..4], 4);
        let _ = host.write_gpa(root_gpa, &d[..4]);
        assert!(state.define_task(1, 0x1000, 2));
        assert!(state.set_object_list(1, 0, 8));
        let mut entry = [0u8; 12];
        st32(
            &mut entry[0..],
            (OBJECT_TYPE_SURFACE as u32) | (0x30u32 << 8),
        );
        entry[4..12].copy_from_slice(&0x80u64.to_le_bytes());
        let _ = host.write_gpa(data_gpa + 5 * 12, &entry);
        let mut desc = vec![0u8; 0x30];
        st64(&mut desc[0..], 0x1000);
        st32(&mut desc[8..], 0x50);
        st32(&mut desc[0xc..], 0x4247_5241);
        desc[0x10] = 1;
        st32(&mut desc[0x18..], 16);
        st32(&mut desc[0x1c..], 16);
        st32(&mut desc[0x20..], 64);
        let _ = host.write_gpa(data_gpa + 0x80, &desc);
        assert!(objects::resolve_type4_surface(&mut state, &host, 5));

        let mut out = ExecResult::default();
        let mut acc = StreamAccum::default();
        let att = ColorAttachment {
            present: true,
            texture_ref: 5,
            resolve_texture_ref: 0,
            level: 0,
            load_action: PASS_LOAD_ACTION_CLEAR,
            store_action: PASS_STORE_ACTION_STORE,
            clear_color: [0.0, 1.0, 0.0, 1.0], // green
        };
        acc.clears.push(att);
        acc.saw_draw = true;
        acc.color_slots.push((0, att));
        acc.draws.push(PendingDraw {
            pipeline_ref: 7,
            draw: (3, 1, 3, 0),
            indexed: None,
            vertex_buffers: vec![BufferBind {
                stage: Stage::Vertex,
                index: 0,
                buffer_ref: 1,
                offset: 0,
            }],
            fragment_buffers: Vec::new(),
            vertex_textures: Vec::new(),
            fragment_textures: Vec::new(),
            vertex_samplers: Vec::new(),
            fragment_samplers: Vec::new(),
            viewport: None,
            scissor: None,
            blend_color: None,
            cull_mode: None,
            front_facing: None,
            depth_bias: None,
            depth_stencil_ref: 0,
            stencil_ref: None,
            depth_attach: None,
            stencil_attach: None,
        });
        let mut second = acc.draws[0].clone();
        second.pipeline_ref = 8;
        acc.draws.push(second);
        finish_stream(&mut state, &mut host, 1, &mut out, &acc);
        assert_eq!(
            out.render_attachment_resolves, 1,
            "one render stream resolves its fixed attachment set once"
        );
        // Non-Apple: Linux encode Stores CLEAR load into type-4 (Ok) or
        // NoMetal clear fallback — either path must land green BGRA.
        #[cfg(feature = "backend-vulkan")]
        {
            assert!(
                out.metal_draws_ok >= 1 || out.clears_applied >= 1 || out.metal_draws_fail >= 1,
                "expected clear store path: ok={} clear={} fail={}",
                out.metal_draws_ok,
                out.clears_applied,
                out.metal_draws_fail
            );
            let mut px = [0u8; 4];
            assert!(host.read_gpa(page, &mut px).is_ok());
            // BGRA green = [0, 255, 0, 255]
            assert_eq!(px, [0, 255, 0, 255], "got {px:?}");
        }
    }

    /// Multi-draw packets force full-frame store on the final record even when
    /// that draw carries a partial scissor (dock damage over chained wallpaper).
    #[test]
    fn multi_draw_force_full_store_flag_for_chained_packet() {
        assert_eq!(multi_draw_store_plan(0, 0), (false, false));
        assert_eq!(multi_draw_store_plan(1, 0), (true, false));
        assert_eq!(multi_draw_store_plan(3, 0), (false, false));
        assert_eq!(multi_draw_store_plan(3, 1), (false, false));
        assert_eq!(multi_draw_store_plan(3, 2), (true, true));
    }

    /// qemu-shim style: multi-draw plan is one guest writeback on the last record
    /// only, with force_full so a partial scissor cannot leave wallpaper only in
    /// host chain memory (archive DrawJob single completion writeback).
    #[test]
    fn multi_draw_store_plan_matches_archive_drawjob_writeback() {
        // N independent single-draw packets: each stores, none force_full.
        for n in 1..8 {
            let (wb, full) = multi_draw_store_plan(1, 0);
            assert!(wb, "single-draw packet always stores");
            assert!(
                !full,
                "single-draw never force_full (scissor-local allowed)"
            );
            let _ = n;
        }
        // One multi-draw packet of 5: only di==4 stores, and force_full.
        let n = 5usize;
        for di in 0..n {
            let (wb, full) = multi_draw_store_plan(n, di);
            if di + 1 == n {
                assert!(wb && full, "final multi-draw record: writeback+force_full");
            } else {
                assert!(!wb && !full, "intermediate multi-draw: host-chain only");
            }
        }
    }

    #[test]
    fn multi_draw_chain_source_preserves_portable_unified_output() {
        assert_eq!(
            multi_draw_chain_source(true, false),
            MultiDrawChainSource::Resident
        );
        assert_eq!(
            multi_draw_chain_source(false, true),
            MultiDrawChainSource::Cpu
        );
        assert_eq!(
            multi_draw_chain_source(false, false),
            MultiDrawChainSource::Missing
        );
    }

    #[test]
    fn render_pass_template_reuses_attachment_without_load_seed() {
        let first = metal_draw::DrawEncodeRequest {
            task_id: 1,
            pipeline_ref: 7,
            color_texture_ref: 11,
            mapping_id: 3,
            width: 1920,
            height: 1080,
            format: 0x50,
            vertex_count: 3,
            instance_count: 1,
            primitive_type: 3,
            target_seed_rgba: Some(vec![0xaa; 16]),
            colors: vec![metal_draw::ColorRtRequest {
                slot: 0,
                texture_ref: 11,
                mapping_id: 3,
                target_gva: 0,
                row_stride: 0,
                width: 1920,
                height: 1080,
                format: 0x50,
                load_action: PASS_LOAD_ACTION_CLEAR,
                store_action: PASS_STORE_ACTION_STORE,
                clear_color: [0.1, 0.2, 0.3, 1.0],
                target_seed_rgba: Some(vec![0xbb; 16]),
            }],
            ..Default::default()
        };
        let template = render_pass_attachment_template(&first);
        assert!(template.target_seed_rgba.is_none());
        assert!(template.colors[0].target_seed_rgba.is_none());
        assert_eq!(template.colors[0].load_action, PASS_LOAD_ACTION_LOAD);
        assert_eq!(template.colors[0].mapping_id, 3);
        assert_eq!((template.width, template.height), (1920, 1080));

        let draw = PendingDraw {
            pipeline_ref: 42,
            draw: (6, 2, 4, 9),
            ..Default::default()
        };
        let req = retarget_render_pass_draw(&template, &draw);
        assert_eq!(req.pipeline_ref, 42);
        assert_eq!(
            (
                req.vertex_count,
                req.instance_count,
                req.primitive_type,
                req.first_vertex
            ),
            (6, 2, 4, 9)
        );
        assert_eq!(req.colors.len(), 1);
        assert_eq!(req.colors[0].mapping_id, 3);
        assert_eq!(first.target_seed_rgba.as_ref().map(Vec::len), Some(16));
        assert_eq!(
            first.colors[0].target_seed_rgba.as_ref().map(Vec::len),
            Some(16)
        );
    }

    #[test]
    fn dropped_clear_logs_once_per_reason_target() {
        // Unique keys per case so no shared-static reset is needed (the dedup set
        // is process-global). First sighting of a (reason, tex_ref) emits (true);
        // an immediate repeat is suppressed (false); a distinct target logs again.
        assert!(note_clear_dropped(
            "nonstore_store_action",
            0x9001,
            "store_action=0 load_action=clear"
        ));
        assert!(!note_clear_dropped(
            "nonstore_store_action",
            0x9001,
            "store_action=0 load_action=clear"
        ));
        assert!(note_clear_dropped(
            "nonstore_store_action",
            0x9002,
            "store_action=0 load_action=clear"
        ));
        // A different reason on the same target is a distinct blind spot and logs.
        assert!(note_clear_dropped(
            "target_unresolved",
            0x9001,
            "color_target_request=none"
        ));
        assert!(!note_clear_dropped(
            "target_unresolved",
            0x9001,
            "color_target_request=none"
        ));
    }
}
