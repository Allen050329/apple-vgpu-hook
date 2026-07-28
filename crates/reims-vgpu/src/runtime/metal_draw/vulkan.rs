/// Vulkan image shape for a reflected Metal sampled-image dimensionality.
///
/// The engine caps array layers at 1 (a single-layer array is still a distinct
/// descriptor type from a plain 2D image), so array shapes report `layers = 1`
/// and a genuinely multi-layer source declines on its byte length rather than
/// binding a truncated array.
#[cfg(feature = "backend-vulkan")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SampledImageShape {
    arrayed: bool,
    volume: bool,
    cube: bool,
    one_dim: bool,
    layers: u32,
}

/// Map a translated SPIR-V sampled-image dimensionality onto the Vulkan image
/// shape the sampled-draw path builds. `None` is a shape the path cannot yet
/// express (`Cube` / `CubeArray`); the caller declines it by name so the gap
/// stays visible instead of binding the wrong view type.
#[cfg(feature = "backend-vulkan")]
fn sampled_image_shape(
    kind: crate::runtime::spirv_bind::SampledImageKind,
) -> Option<SampledImageShape> {
    use crate::runtime::spirv_bind::SampledImageKind;
    let (arrayed, volume, cube, one_dim) = match kind {
        SampledImageKind::D1 => (false, false, false, true),
        SampledImageKind::D1Array => (true, false, false, true),
        SampledImageKind::D2 => (false, false, false, false),
        SampledImageKind::D2Array => (true, false, false, false),
        SampledImageKind::D3 => (false, true, false, false),
        SampledImageKind::Cube | SampledImageKind::CubeArray => return None,
    };
    Some(SampledImageShape {
        arrayed,
        volume,
        cube,
        one_dim,
        layers: 1,
    })
}

#[cfg(feature = "backend-vulkan")]
pub fn encode_draw_and_writeback<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    req: &mut DrawEncodeRequest,
) -> EncodeStatus {
    encode_draw_chain(state, host, req, true, true).0
}

/// Linux / non-Apple product rail: metal2vulkan + Vulkan offscreen, then Store.
///
/// `writeback_guest` is the archive multi-draw store plan (only the last record
/// of a serialized render-pass chain writes guest memory). Intermediate records **must still
/// encode** and return color0 for chaining — returning `NoMetal` when
/// `!writeback_guest` aborted every multi-draw stream after the first
/// record (live `draw_fail_clear_fallback nometal=1` on clear+draw packets).
#[cfg(feature = "backend-vulkan")]
pub fn encode_draw_chain<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    req: &mut DrawEncodeRequest,
    writeback_guest: bool,
    _force_full_store: bool,
) -> (EncodeStatus, Option<Vec<u8>>) {
    let colors: Vec<ColorRtRequest> = if req.colors.is_empty() {
        if req.color_texture_ref == 0 && req.mapping_id == 0 {
            return (EncodeStatus::BadArgs("draw_vk_no_color_target"), None);
        }
        vec![ColorRtRequest {
            slot: 0,
            texture_ref: req.color_texture_ref,
            mapping_id: req.mapping_id,
            target_gva: 0,
            row_stride: 0,
            width: req.width,
            height: req.height,
            format: if req.format != 0 {
                req.format
            } else {
                MTL_FORMAT_BGRA8_UNORM
            },
            load_action: PASS_LOAD_ACTION_CLEAR,
            store_action: PASS_STORE_ACTION_STORE,
            clear_color: [0.0, 0.0, 0.0, 1.0],
            target_seed_rgba: None,
        }]
    } else {
        req.colors.clone()
    };

    let mut any_store = false;
    let mut color0_rgba: Option<Vec<u8>> = None;
    // Solid CLEAR seed Stores only when this record owns guest writeback
    // (last of a serialized chain, or unified always-writeback).
    if writeback_guest {
        for (i, c) in colors.iter().enumerate() {
            if c.store_action != PASS_STORE_ACTION_STORE {
                continue;
            }
            if c.load_action != PASS_LOAD_ACTION_CLEAR
                && c.load_action != PASS_LOAD_ACTION_DONT_CARE
            {
                // Load/composite needs real encode (metal2vulkan) — skip Store.
                continue;
            }
            if c.width == 0 || c.height == 0 {
                continue;
            }
            let rgba = solid_rgba_local(c.width, c.height, &c.clear_color);
            let ok = if c.target_gva != 0 {
                supersede_gva_window(state, host, c.target_gva, c.width, c.height, "clear_store");
                write_gva_rgba8(
                    state,
                    host,
                    req.task_id,
                    c.target_gva,
                    c.width,
                    c.height,
                    c.row_stride,
                    c.format,
                    &rgba,
                )
                .is_ok()
            } else if c.mapping_id != 0 {
                // Type-11 CLEAR: contig HostOps path (write_bgra8 is contig-only).
                let bgra = swap_rb_channels(&rgba);
                let stride = c.width.saturating_mul(RGBA8_BPP);
                mapping_write::write_bgra8(
                    state,
                    host,
                    c.mapping_id,
                    &bgra,
                    stride,
                    c.width,
                    c.height,
                )
            } else {
                false
            };
            if ok {
                any_store = true;
                if i == 0 {
                    color0_rgba = Some(rgba);
                }
                crate::observe::line(format!(
                    "linux_clear_store mid={} gva={:#x} {}x{} pipe={} load={}",
                    c.mapping_id, c.target_gva, c.width, c.height, req.pipeline_ref, c.load_action
                ));
            }
        }
    }

    // metal2vulkan path: load MTLB → AIR → SPIR-V → internal Vulkan engine offscreen.
    let mut draw_rgba: Option<Vec<u8>> = None;
    #[cfg(feature = "backend-vulkan")]
    let mut draw_resident: Option<DrawResident> = None;
    // GVA Store landed as a deferred-writeback window (resident authoritative).
    #[cfg(feature = "backend-vulkan")]
    let mut gva_store_armed = false;
    if req.pipeline_ref != 0 && (req.vertex_count > 0 || req.indexed.is_some()) {
        req.chain_resident_established = false;
        match try_metal2vulkan_draw(state, host, req, writeback_guest) {
            Ok(M2vDrawSpan::Rgba(rgba)) => {
                draw_rgba = Some(rgba);
                crate::observe::line(format!(
                    "linux_m2v_draw ok pipe={} {}x{} vtx={}",
                    req.pipeline_ref, req.width, req.height, req.vertex_count
                ));
            }
            #[cfg(feature = "backend-vulkan")]
            Ok(M2vDrawSpan::ResidentBgra {
                identity,
                mapping_id,
                width,
                height,
                display_sample_mids,
                full_geometry_linear_sample,
                full_quad_bounds,
            }) => {
                draw_resident = Some((
                    identity,
                    mapping_id,
                    width,
                    height,
                    display_sample_mids,
                    full_geometry_linear_sample,
                    full_quad_bounds,
                ));
                crate::observe::line(format!(
                    "linux_m2v_draw ok resident_bgra mid={mapping_id} {width}x{height} pipe={}",
                    req.pipeline_ref
                ));
            }
            #[cfg(feature = "backend-vulkan")]
            Ok(M2vDrawSpan::ResidentChain) => {
                req.chain_resident_established = true;
                crate::observe::line(format!(
                    "linux_m2v_draw ok resident_chain pipe={} {}x{} mid={} gva={:#x}",
                    req.pipeline_ref,
                    req.width,
                    req.height,
                    req.colors.first().map(|c| c.mapping_id).unwrap_or(0),
                    req.colors.first().map(|c| c.target_gva).unwrap_or(0)
                ));
            }
            #[cfg(feature = "backend-vulkan")]
            Ok(M2vDrawSpan::ResidentGvaStore) => {
                if arm_gva_deferred_store(state, host, req) {
                    gva_store_armed = true;
                    crate::observe::line(format!(
                        "linux_m2v_draw ok resident_gva_store pipe={} {}x{} gva={:#x}",
                        req.pipeline_ref,
                        req.width,
                        req.height,
                        req.colors.first().map(|c| c.target_gva).unwrap_or(0)
                    ));
                } else {
                    // Arm gate failed (unwalkable span / pin refusal): land
                    // synchronously from the resident the draw just produced.
                    // read_resident_chain fail-logs a lost resident.
                    draw_rgba = read_resident_chain(state, req);
                    crate::observe::line(format!(
                        "linux_m2v_draw ok resident_gva_store_sync_fallback pipe={} {}x{} gva={:#x} rgba={}",
                        req.pipeline_ref,
                        req.width,
                        req.height,
                        req.colors.first().map(|c| c.target_gva).unwrap_or(0),
                        draw_rgba.is_some() as u8
                    ));
                }
            }
            Ok(M2vDrawSpan::None) => {
                crate::observe::line(format!(
                    "linux_m2v_draw skip pipe={} (no color0 geom)",
                    req.pipeline_ref
                ));
            }
            Err(e) => {
                // Always-on + latched: a rejected engine draw falls to the
                // clear-store fallback and surfaces as a bare `no_metal`
                // (the Safari padded-stride reject was invisible on a normal
                // boot — the content layer stayed blank with zero fail lines).
                // The decline names the specific check as the primary `reason=`
                // (an engine `vk_*` VkCall slug, a `DrawReason` refusal, or a
                // runtime `DrawPreparationDecline`).
                // The guest re-submits every frame, so latch on
                // (reason, pipeline_ref): a persistent reject cannot flood, but
                // a new reason on the same pipeline still surfaces.
                linux_m2v_draw_failure(&e, req).fail_once(req.pipeline_ref as u64);
            }
        }
    }

    // An intermediate draw into a resident type-11 attachment has completed
    // successfully even though this record does not publish a guest Store.
    // The Vulkan target remains resident and the next record loads it. Treating
    // this as NoMetal made callers abandon the render pass unless every record
    // redundantly imported the full attachment into guest pages.
    #[cfg(feature = "backend-vulkan")]
    if !writeback_guest && draw_resident.is_some() {
        req.chain_resident_established = true;
        return (EncodeStatus::Ok, None);
    }

    // Same for a resident render-pass chain intermediate: the exec loop reads
    // `chain_resident_established` and arms the next record's LoadFromTarget.
    #[cfg(feature = "backend-vulkan")]
    if req.chain_resident_established {
        return (EncodeStatus::Ok, None);
    }

    // Deferred GVA Store: the window is armed and the resident holds the
    // authoritative pixels — the contract Store lands on first access.
    #[cfg(feature = "backend-vulkan")]
    if gva_store_armed {
        return (EncodeStatus::Ok, None);
    }

    // Zero-copy only: resident BGRA + revalidated contig + strided import DMA.
    // No CPU write_bgra8 fallback for type-11 composite Stores.
    #[cfg(feature = "backend-vulkan")]
    if writeback_guest {
        if let Some((identity, mid, w, h, display_sample_mids, linear_sample, full_quad_bounds)) =
            draw_resident.take()
        {
            use crate::runtime::import_present::{
                try_defer_present_store, try_import_present_store, ImportPresentResult,
            };
            // Ack-fast rung: keep the composite resident-side and flush on
            // access instead of a ~7 ms synchronous DMA on the stamp path.
            let deferred =
                try_defer_present_store(state, host, &identity, mid, w, h, full_quad_bounds);
            let imp = if deferred {
                ImportPresentResult::Ok
            } else {
                // Direct synchronous Store (no defer happened) — no prefetch armed.
                // Measure-only: the synchronous scatter-DMA ran on the stamp path.
                crate::runtime::census::writeback_census::note_sync(state.present.dmabuf_active);
                try_import_present_store(state, host, &identity, mid, w, h, full_quad_bounds)
            };
            let pages_ok = matches!(imp, ImportPresentResult::Ok);
            // Mapping lifetime on every Store line — mids are recycled, so a
            // Store and a later sampled view correlate only via (mid, map_gen)
            // (the media-garble forensics class).
            let map_gen = state
                .mappings
                .get(&mid)
                .map(|m| m.map_generation)
                .unwrap_or(0);
            if !pages_ok {
                crate::observe::line(format!(
                    "linux_m2v_store mid={mid} map_gen={map_gen} {w}x{h} pipe={} import=0 reason={} (zero-copy only; no CPU fallback)",
                    req.pipeline_ref,
                    imp.reason()
                ));
            } else {
                crate::observe::line(format!(
                    "linux_m2v_store mid={mid} map_gen={map_gen} {w}x{h} pipe={} pages=1 import=1 reason={}",
                    req.pipeline_ref,
                    if deferred { "deferred" } else { "ok" }
                ));
            }
            // Measure-only: which Stores land on the present/retain mid.
            // Do **not** capture +0x188 or enqueue ScanoutUpdate here.
            // Archive apple-pv-gpu-render: post-boundary front writebacks only
            // update present_mapping tracking — they do NOT paint. Multi-pass
            // intermediates publish only at CmdDisplaySwap / present boundary
            // (tile-through thrash class: 14 consecutive same-mid writebacks
            // otherwise thrash incomplete halves onto the console).
            // G1 + present thrash. early_front still latched via note_front below.
            let is_front = state.present.frame_flush_seen
                && (state.present.present_mapping == mid
                    || state.present.host_mapping == mid
                    || state.present.frame_mapping == mid);
            crate::observe::line(format!(
                "m2v_store mid={mid} map_gen={map_gen} {w}x{h} pipe={} import={} pages={} is_front={} frame_flush={} present_mapping={} frame_mapping={}",
                req.pipeline_ref,
                imp.used() as u8,
                pages_ok as u8,
                is_front as u8,
                state.present.frame_flush_seen as u8,
                state.present.present_mapping,
                state.present.frame_mapping
            ));
            if pages_ok {
                if linear_sample {
                    let _ = crate::runtime::scanout::note_linear_compositor_edge(
                        state,
                        mid,
                        w,
                        h,
                        req.pipeline_ref,
                    );
                }
                for source_mid in display_sample_mids {
                    let _ = crate::runtime::scanout::note_compositor_edge(
                        state,
                        source_mid,
                        mid,
                        w,
                        h,
                        req.pipeline_ref,
                    );
                }
                crate::runtime::scanout::note_front_buffer_writeback(state, host, mid, w, h, 0);
            }
            let _ = any_store;
            return (EncodeStatus::Ok, None);
        }
    }

    if let Some(ref rgba) = draw_rgba {
        // Intermediate multi-draw GVA records: return color0 for chaining without
        // guest Store (archive store plan). Resident type-11 intermediates
        // returned above without materializing CPU pixels.
        if !writeback_guest {
            return (EncodeStatus::Ok, Some(rgba.clone()));
        }
        // Store draw result into primary color RT.
        if let Some(c0) = colors.first() {
            let mut rgb_nz = 0usize;
            let mut max_rgb = 0u8;
            for px in rgba.chunks_exact(4) {
                let m = px[0].max(px[1]).max(px[2]);
                if m != 0 {
                    rgb_nz += 1;
                }
                if m > max_rgb {
                    max_rgb = m;
                }
            }
            let ok = if c0.mapping_id != 0 {
                #[cfg(feature = "backend-vulkan")]
                let import_allowed = type11_import_allowed(
                    true,
                    host.map_pages_stable()
                        && crate::backend::vulkan::engine::external_memory_host_available(),
                    c0.mapping_id,
                    c0.width,
                    c0.height,
                );
                let cpu_fallback_allowed = type11_cpu_store_fallback_allowed(import_allowed);
                if cpu_fallback_allowed {
                    let ok = mapping_write::write_rgba8_image_changed(
                        state,
                        host,
                        c0.mapping_id,
                        rgba,
                        None,
                        c0.width,
                        c0.height,
                    );
                    if ok {
                        // Full-frame publish: same completeness proof as the
                        // import-present scatter paths — the write verified
                        // geometry (mw==w, mh==h) and landed the complete
                        // frame into the mapping's guest pages. Without it the
                        // `present_unbacked` gate is structurally dead on the
                        // CPU-portability Store path: no mapping's
                        // `dense_frame_seq` would ever advance.
                        publish_cpu_portability_store(
                            state,
                            host,
                            c0.mapping_id,
                            c0.width,
                            c0.height,
                            c0.format,
                        );
                        crate::observe::line(format!(
                            "linux_m2v_store mid={} {}x{} pipe={} import=0 reason=cpu_portability pages=1 rgb_nz={} max={}",
                            c0.mapping_id,
                            c0.width,
                            c0.height,
                            req.pipeline_ref,
                            rgb_nz,
                            max_rgb
                        ));
                    } else {
                        crate::observe::fail(format!(
                            "linux_m2v_store mid={} {}x{} pipe={} reason=cpu_portability_write_fail rgb_nz={} max={} fmt={:#x}",
                            c0.mapping_id,
                            c0.width,
                            c0.height,
                            req.pipeline_ref,
                            rgb_nz,
                            max_rgb,
                            c0.format
                        ));
                    }
                    ok
                } else {
                    // Type-11 composite Stores on native Vulkan must use
                    // import-present (ResidentBgra). Landing here means the draw
                    // returned CPU pixels — fail closed to preserve the zero-copy
                    // invariant on that pathway.
                    crate::observe::line(format!(
                        "linux_m2v_store mid={} {}x{} pipe={} reason=rgba_not_import (zero-copy only; no write_bgra8) rgb_nz={} max={}",
                        c0.mapping_id,
                        c0.width,
                        c0.height,
                        req.pipeline_ref,
                        rgb_nz,
                        max_rgb
                    ));
                    false
                }
            } else if c0.target_gva != 0 {
                supersede_gva_window(
                    state,
                    host,
                    c0.target_gva,
                    c0.width,
                    c0.height,
                    "sync_store",
                );
                let gva_ok = write_gva_rgba8(
                    state,
                    host,
                    req.task_id,
                    c0.target_gva,
                    c0.width,
                    c0.height,
                    c0.row_stride,
                    c0.format,
                    rgba,
                )
                .is_ok();
                // Discrete-GPU rail: type-2/3 encode into **texture_ref** + **GVA**
                // host caches (not surface_id mid map — list ids collide with
                // present mids;). Sample prefers GVA key then
                // texture_ref with live descriptor geom.
                if gva_ok {
                    let producer_object_type =
                        objects::lookup_list_entry(state, host, req.task_id, c0.texture_ref)
                            .map(|entry| entry.object_type)
                            .unwrap_or(0);
                    host_cache_store_gva_layer(
                        state,
                        req.task_id,
                        c0.texture_ref,
                        producer_object_type,
                        c0.target_gva,
                        c0.width,
                        c0.height,
                        rgba,
                    );
                }
                crate::observe::fail(format!(
                    "linux_m2v_store gva={:#x} {}x{} pipe={} ok={} rgb_nz={} max={}",
                    c0.target_gva,
                    c0.width,
                    c0.height,
                    req.pipeline_ref,
                    gva_ok as u8,
                    rgb_nz,
                    max_rgb
                ));
                crate::observe::off(format!(
                    "m2v_store_gva gva={:#x} {}x{} pipe={} tex_ref={} load={} ok={} rgb_nz={} max_rgb={} bpr={}",
                    c0.target_gva,
                    c0.width,
                    c0.height,
                    req.pipeline_ref,
                    c0.texture_ref,
                    c0.load_action,
                    gva_ok as u8,
                    rgb_nz,
                    max_rgb,
                    c0.row_stride
                ));
                gva_ok
            } else {
                crate::observe::fail(format!(
                    "linux_m2v_store no_target pipe={} rgb_nz={} max={}",
                    req.pipeline_ref, rgb_nz, max_rgb
                ));
                crate::observe::off(format!(
                    "m2v_store_no_target pipe={} tex_ref={} rgb_nz={} max_rgb={}",
                    req.pipeline_ref, c0.texture_ref, rgb_nz, max_rgb
                ));
                false
            };
            if ok {
                return (EncodeStatus::Ok, Some(rgba.clone()));
            }
        }
    }

    if any_store {
        if req.vertex_count > 0 || req.indexed.is_some() {
            crate::observe::fail(format!(
                "linux_clear_store draws_skipped pipe={} vtx={} (m2v pending)",
                req.pipeline_ref, req.vertex_count
            ));
        }
        (EncodeStatus::Ok, color0_rgba)
    } else {
        (EncodeStatus::NoMetal("draw_vk_nothing_stored"), None)
    }
}

/// Sampled texture source + geometry for an engine draw.
enum SampledSourceRequest {
    /// Shared texel bytes + optional producer identity (see
    /// [`LinearSampleIdentity`]) + the byte layout of those texels; the Arc lets
    /// memoized repeat binds skip the per-draw copy and the engine skip
    /// re-hashing.
    Bytes(
        std::sync::Arc<Vec<u8>>,
        Option<LinearSampleIdentity>,
        TexelLayout,
    ),
    #[cfg(feature = "backend-vulkan")]
    Target(crate::backend::vulkan::engine::TargetIdentity),
    /// Zero-copy guest gather: the engine copies the texel bytes from
    /// imported guest RAM inside the draw CB — no CPU read, no memo, no
    /// hash. Carries the native texel layout the image is created with.
    #[cfg(feature = "backend-vulkan")]
    GuestRuns(crate::backend::vulkan::engine::GuestRunSource, TexelLayout),
}

/// Producer identity + generation for CPU-sourced sampled bytes. `key` is the
/// texture's authoritative GVA (`host_gva_surfaces` keyspace) and `generation`
/// that cache entry's generation, so equal identity implies equal bytes under
/// the same coherence model the cache itself already relies on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LinearSampleIdentity {
    key: u64,
    generation: u64,
}

type LoadedType5View = (
    u32,
    u32,
    std::sync::Arc<Vec<u8>>,
    LinearSampleIdentity,
    TexelLayout,
);
type LoadedLinearSample = (
    u32,
    u32,
    std::sync::Arc<Vec<u8>>,
    Option<LinearSampleIdentity>,
    TexelLayout,
);
type DrawHandoffStage<'a> = (&'a str, &'a [u8], &'a [u8], &'a [u8]);
#[cfg(feature = "backend-vulkan")]
type DrawResident = (
    crate::backend::vulkan::engine::TargetIdentity,
    u32,
    u32,
    u32,
    Vec<u32>,
    bool,
    bool,
);

/// Authoritative contents when a fragment texture aliases a GVA color target.
///
/// A serialized Metal render stream may read `color(0)` through texture slot 0
/// while several draws remain in one render pass. For GVA targets, records 2+
/// carry the prior draw in `target_seed_rgba`; reloading guest pages here would
/// expose the pre-pass image to the shader even though attachment Load sees the
/// chained image.
#[cfg(feature = "backend-vulkan")]
#[derive(Clone, Copy, Debug, PartialEq)]
enum AttachmentAliasSample<'a> {
    Clear([f64; 4]),
    Seed(&'a [u8]),
    /// Records 2+ of a resident GVA chain: the prior record's content lives
    /// on the engine-resident target, not in a CPU seed. Bound as a resident
    /// sampled source (the engine snapshots on self-alias).
    #[cfg(feature = "backend-vulkan")]
    ResidentChain,
}

#[cfg(feature = "backend-vulkan")]
fn fragment_attachment_alias_sample<'a>(
    req: &'a DrawEncodeRequest,
    texture_index: u32,
    texture_ref: u32,
) -> Option<(u32, u32, AttachmentAliasSample<'a>)> {
    let color = req.colors.iter().find(|color| {
        color.slot == texture_index
            && color.texture_ref == texture_ref
            && color.mapping_id == 0
            && color.target_gva != 0
    })?;
    let need = (color.width as usize)
        .checked_mul(color.height as usize)?
        .checked_mul(RGBA8_BPP as usize)?;
    match color.load_action {
        PASS_LOAD_ACTION_CLEAR => Some((
            color.width,
            color.height,
            AttachmentAliasSample::Clear(color.clear_color),
        )),
        PASS_LOAD_ACTION_LOAD => {
            if let Some(seed) = color
                .target_seed_rgba
                .as_deref()
                .filter(|seed| seed.len() == need)
            {
                return Some((color.width, color.height, AttachmentAliasSample::Seed(seed)));
            }
            #[cfg(feature = "backend-vulkan")]
            if req.chain_from_resident {
                return Some((
                    color.width,
                    color.height,
                    AttachmentAliasSample::ResidentChain,
                ));
            }
            None
        }
        _ => None,
    }
}

#[cfg(feature = "backend-vulkan")]
#[allow(clippy::too_many_arguments)]
fn log_attachment_alias_chain(
    task_id: u32,
    texture_index: u32,
    texture_ref: u32,
    target_gva: u64,
    width: u32,
    height: u32,
    rgba: &[u8],
) {
    let (rgb_nz, max_rgb, _) = crate::observe::rgba_rgb_stats(rgba);
    let alpha_nz = rgba.chunks_exact(4).filter(|pixel| pixel[3] != 0).count();
    use std::collections::HashSet;
    use std::sync::Mutex;
    type AttachmentAliasKey = (u32, u32, u32, u64, u32, u32, usize, usize);
    static SEEN: Mutex<Option<HashSet<AttachmentAliasKey>>> = Mutex::new(None);
    let first = SEEN
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .get_or_insert_with(HashSet::new)
        .insert((
            task_id,
            texture_index,
            texture_ref,
            target_gva,
            width,
            height,
            rgb_nz,
            alpha_nz,
        ));
    if first {
        crate::observe::off(format!(
            "attachment_alias_chain reason=in_process_seed action=sample_seed task={task_id} i={texture_index} ref={texture_ref} gva={target_gva:#x} {width}x{height} rgb_nz={rgb_nz} max_rgb={max_rgb} alpha_nz={alpha_nz}"
        ));
    }
}

/// Sampled texture as CPU RGBA8 or a direct resident target.
///
/// Order (surface_id vs texture_ref namespaces are separate):
/// 1. type-5 RefTexture → surface_id; type-11 mapping → mid → `host_surfaces`
/// 2. type-2/3 linear: GVA encode cache → texture_ref cache (geom match) → guest GVA
/// 3. fallback texture_ref encode cache (tests / missing list entry)
///
/// Always-on dig: where type-11/surface sample bytes came from.
///
/// `src` = `cache` | `guest` | `resident`. Product prefers non-empty guest/cache,
/// then **resident GPU** when guest/cache RGB is empty (C_sample fix).
fn log_sample_src(
    texture_ref: u32,
    mid: u32,
    w: u32,
    h: u32,
    src: &str,
    rgba: &[u8],
    resident_ready: bool,
) {
    // Display-sized samples dominate the empty-boot class; skip tiny glyphs.
    if w < 1280 || h < 720 {
        return;
    }
    let (nz, max_rgb, px0) = crate::observe::rgba_rgb_stats(rgba);
    crate::observe::fail(format!(
        "sample_src={src} ref={texture_ref} mid={mid} {w}x{h} rgb_nz={nz} max_rgb={max_rgb} resident_ready={} px0=[{},{},{},{}]",
        resident_ready as u8,
        px0[0],
        px0[1],
        px0[2],
        px0[3]
    ));
}

/// Resolve a content-ready type-11 target without a CPU readback/reupload.
///
/// `id` is the resolved identity for the sampled mapping. The engine snapshots
/// it when the draw also targets the same identity.
#[cfg(feature = "backend-vulkan")]
/// Bind a resident surface directly as the sampled source. The caller MUST have
/// already confirmed `resident_content_ready(&id)` — this function does not
/// re-acquire the global engine lock to re-check it (that second acquisition per
/// resident bind was pure overhead on the hot sample-resolve path, and the check
/// cannot close the eviction race anyway: the readiness the caller observed and
/// the GPU bind are already separate lock acquisitions, so the engine must
/// tolerate a since-evicted target at record time regardless).
fn try_sample_resident_surface(
    texture_ref: u32,
    mid: u32,
    w: u32,
    h: u32,
    id: crate::backend::vulkan::engine::TargetIdentity,
) -> Option<(u32, u32, u32, SampledSourceRequest)> {
    if w >= 1280 && h >= 720 {
        crate::observe::line(format!(
            "sample_src=resident_direct ref={texture_ref} mid={mid} {w}x{h} resident_ready=1"
        ));
    }
    Some((w, h, mid, SampledSourceRequest::Target(id)))
}

/// A deferred GVA window may serve a sampled bind directly from its resident
/// target only when the sampled view is the exact window content: descriptor
/// geometry equals the window geometry, and the same storage-family gate that
/// would let the post-flush cache layer serve this object type accepts it. Any
/// mismatch must land the window (flush path) instead.
#[cfg(feature = "backend-vulkan")]
fn deferred_gva_sample_eligible(
    win: &crate::model::GvaDeferredEntry,
    desc_width: u32,
    desc_height: u32,
    sampler_object_type: u8,
) -> bool {
    win.width == desc_width
        && win.height == desc_height
        && gva_cache_owner_allows_object_type(win.producer_object_type, sampler_object_type)
}

/// Bind a still-deferred GVA render Store's resident target for a type-2/3
/// sampled bind instead of flushing it to guest memory and re-uploading.
///
/// Value-equal to the flush path: a flush lands the resident RGBA into the
/// `host_gva_surfaces` layer, and the sample would serve that layer back
/// (BGRA→RGBA swap) — so binding the resident directly yields the same texels
/// whenever the descriptor geometry matches the window and the cache owner
/// gate would accept the layer. Mismatches fall through to the flush path.
/// The window stays armed: the contract Store still lands on first guest
/// access, and the resident stays authoritative for further samples.
#[cfg(feature = "backend-vulkan")]
fn try_sample_deferred_gva<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    texture_ref: u32,
) -> Option<(u32, u32, u32, SampledSourceRequest)> {
    use crate::backend::vulkan::engine::{self, TargetIdentity};
    if state.gva_deferred_flush.is_empty() {
        return None;
    }
    let entry = objects::lookup_list_entry(state, host, task_id, texture_ref)?;
    let desc_bytes = objects::read_descriptor(state, host, task_id, &entry)?;
    let tex = decode_texture_descriptor(&desc_bytes).ok()?;
    let (gva, layout) = tex.level_gva(0, state.page_shift)?;
    let win = state.gva_deferred_flush.get(&gva)?;
    if !deferred_gva_sample_eligible(win, layout.width, layout.height, entry.object_type) {
        return None;
    }
    let id = TargetIdentity::Gva {
        gva,
        width: win.width,
        height: win.height,
        generation: 0,
    };
    if !engine::resident_content_ready(&id) {
        return None;
    }
    if win.width >= 1280 && win.height >= 720 {
        crate::observe::fail(format!(
            "sample_src=gva_resident_direct ref={texture_ref} gva={gva:#x} {}x{} resident_ready=1",
            win.width, win.height
        ));
    }
    sampled_census::note(sampled_census::Branch::GvaResident, 0);
    Some((win.width, win.height, 0, SampledSourceRequest::Target(id)))
}

/// Sibling of [`try_sample_deferred_gva`] for MRT secondary attachments (the
/// vibrancy coverage mask): the mask has no deferred-flush window (it is a
/// DontCare-store secondary, never a guest-visible Store), but the MRT producer
/// rendered it into an engine resident and recorded its GVA in
/// [`DeviceState::mrt_secondary_gvas`]. When a type-2/3 texture sampling this
/// GVA has a matching-geometry, content-ready resident, bind it directly so the
/// material alpha reads the real mask instead of zero. Coherent by
/// construction: only GVAs we actively rendered as secondaries are eligible.
#[cfg(feature = "backend-vulkan")]
fn try_sample_mrt_secondary<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    texture_ref: u32,
) -> Option<(u32, u32, u32, SampledSourceRequest)> {
    use crate::backend::vulkan::engine::{self, TargetIdentity};
    if state.mrt_secondary_gvas.is_empty() {
        return None;
    }
    let entry = objects::lookup_list_entry(state, host, task_id, texture_ref)?;
    let desc_bytes = objects::read_descriptor(state, host, task_id, &entry)?;
    let tex = decode_texture_descriptor(&desc_bytes).ok()?;
    let (gva, layout) = tex.level_gva(0, state.page_shift)?;
    let (w, h) = *state.mrt_secondary_gvas.get(&gva)?;
    // The sampler's descriptor geometry must match the rendered mask exactly.
    if layout.width != w || layout.height != h {
        // The sampled GVA IS a rendered mask, but the sampler's geometry differs
        // from what we rendered — the material cannot bind it and falls through to
        // fallback bytes (see note_mrt_mask_bind_miss).
        crate::runtime::census::present_proxy::note_mrt_mask_bind_miss(
            crate::runtime::census::present_proxy::MaskBindMiss::GeometryMismatch,
            w,
            h,
        );
        return None;
    }
    let id = TargetIdentity::Gva {
        gva,
        width: w,
        height: h,
        generation: 0,
    };
    if !engine::resident_content_ready(&id) {
        crate::runtime::census::present_proxy::note_mrt_mask_bind_miss(
            crate::runtime::census::present_proxy::MaskBindMiss::ResidentNotReady,
            w,
            h,
        );
        return None;
    }
    sampled_census::note(sampled_census::Branch::GvaResident, 0);
    Some((w, h, 0, SampledSourceRequest::Target(id)))
}

/// Resolve a sampled texture ref to `(width, height, mapping_id, source)`.
///
/// Backend-neutral: the returned [`SampledSourceRequest`] is either an engine
/// target to bind directly (zero-copy) or CPU bytes to upload, so this is the
/// resolver the engine draw path uses. Distinct from [`load_sampled_rgba`],
/// which is the Metal-path resolver and always materializes RGBA8 bytes.
fn resolve_sampled_source<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    texture_ref: u32,
    entry: Option<ListObjectEntry>,
) -> Option<(u32, u32, u32, SampledSourceRequest)> {
    if texture_ref == 0 {
        return None;
    }

    // Opcode-9 buffer-backed texture (type-8): the sampled bytes are an MTLBuffer's
    // guest storage, not a view over another texture. Resolve it directly before
    // the view/surface paths (which would mis-decode the opcode-9 descriptor).
    // `entry` (when supplied by the caller) reuses the object-list read this call
    // and its buffer-texture / view classification below would otherwise each
    // repeat — the guest object list is immutable for the draw.
    if let Some(bt) = buffer_texture_descriptor(state, host, task_id, texture_ref, entry) {
        let (w, h, rgba) = load_buffer_texture_rgba(state, host, task_id, texture_ref, &bt)?;
        log_sample_src(texture_ref, 0, w, h, "buftex_guest", &rgba, false);
        return Some((
            w,
            h,
            0,
            SampledSourceRequest::Bytes(std::sync::Arc::new(rgba), None, TexelLayout::Rgba8),
        ));
    }

    // Resolve type-5 RefTextureHandle → surface_id (type-4 object id).
    let mut surface_candidates: Vec<u32> = Vec::new();
    let mut is_linear_tex = false;
    let mut is_type5 = false;
    let mut type5_surface_id = 0u32;
    let mut type5_view: Option<objects::Type5TextureView> = None;
    let resolved_entry =
        entry.or_else(|| objects::lookup_list_entry(state, host, task_id, texture_ref));
    if let Some(entry) = resolved_entry {
        if entry.object_type == objects::OBJECT_TYPE_REF_TEXTURE {
            is_type5 = true;
            if let Some(desc) = objects::read_descriptor(state, host, task_id, &entry) {
                if desc.len() >= objects::TYPE5_MIN_LEN {
                    let sid = crate::contract::endian::ld32(&desc[objects::TYPE5_SURFACE_ID..]);
                    if sid != 0 {
                        type5_surface_id = sid;
                        type5_view = objects::decode_type5_texture_view(&desc);
                        surface_candidates.push(sid);
                    }
                }
            }
        }
        if entry.object_type == objects::OBJECT_TYPE_SURFACE {
            surface_candidates.push(texture_ref);
        }
        if entry.object_type == OBJECT_TYPE_TEXTURE
            || entry.object_type == OBJECT_TYPE_TEXTURE_VARIANT
        {
            is_linear_tex = true;
        }
    }
    if !is_type5 {
        let t_ref = std::time::Instant::now();
        let resolved = objects::resolve_type11_ref(state, host, task_id, texture_ref);
        crate::runtime::census::setup_tex_census::note_ref(t_ref.elapsed().as_micros() as u64);
        if let Some(mid) = resolved {
            surface_candidates.push(mid);
        }
    }
    surface_candidates.sort_unstable();
    surface_candidates.dedup();

    for mid in surface_candidates {
        // Ensure type-4 pages exist for this surface id.
        let t_ensure = std::time::Instant::now();
        let _ = objects::ensure_surface_for_present(state, host, mid);
        crate::runtime::census::setup_tex_census::note_ensure(t_ensure.elapsed().as_micros() as u64);
        // A type-5 serialized record is the exact Metal texture view over the
        // IOSurface bytes. Materialize it only when it differs from (or cannot
        // be inferred from) the base mapping. Exact base views keep the fast
        // resident/cache path below; an unknown 2-B/texel base FourCC exposed
        // as RG8 must instead use the serialized view's native interpretation.
        if mid == type5_surface_id {
            if let Some(view) = type5_view {
                let needs_materialization = state
                    .mappings
                    .get(&mid)
                    .map(|m| {
                        type5_view_requires_materialization(
                            m.has_geom, m.width, m.height, m.format, view,
                        )
                    })
                    .unwrap_or(true);
                if needs_materialization {
                    // Zero-copy the decoded plane straight from guest pages when
                    // it samples byte-identically (video NV12 R8/RG8, BGRA8/
                    // RGBA8). This bypasses the ~1.5 MB/plane/frame CPU read +
                    // upload the CPU loader below would pay every decoded frame.
                    #[cfg(feature = "backend-vulkan")]
                    if let Some(src) = try_type5_sample_zero_copy(state, host, mid, view) {
                        // Success path: a healthy video decodes ~2 planes/frame,
                        // so this fires per-bind (~99k lines/boot). The aggregate
                        // lives in `sampled_branch_census` (`t5_zc=count:bytes`),
                        // which is the always-on signal; keep the per-bind detail
                        // for deep debugging behind REIMS_VGPU_DRAW_LOG (observe::line)
                        // rather than flooding the always-on fail sink.
                        crate::observe::line(format!(
                            "type5_view_zc ref={texture_ref} sid={mid} view={}x{} fmt={:#x} plane={}",
                            view.width, view.height, view.pixel_format, view.plane_index
                        ));
                        return Some((view.width, view.height, mid, src));
                    }
                    let (w, h, rgba, identity, byte_format) =
                        load_type5_view_rgba(state, host, task_id, texture_ref, mid, view)?;
                    log_sample_src(texture_ref, mid, w, h, "type5_view_guest", &rgba, false);
                    return Some((
                        w,
                        h,
                        mid,
                        SampledSourceRequest::Bytes(rgba, Some(identity), byte_format),
                    ));
                }
            }
        }
        if let Some(m) = state.mappings.get(&mid) {
            if m.has_geom && m.width > 0 && m.height > 0 {
                let (w, h) = (m.width, m.height);
                // Attribute the resident-readiness/bind sub-slice of the resolve so
                // the census can separate engine-lock cost (this block) from the
                // object-list decode prelude. This block acquires the global engine
                // lock (`resident_content_ready`), the suspected dock-hover-freeze
                // contention site.
                let t_bind = std::time::Instant::now();
                // Resident-surface identity: computed once and reused for both the
                // readiness check and the direct bind. `surface_identity` locks a
                // global dedup mutex and does an output-group lookup; this bind
                // resolves the same (mid, w, h), so recomputing it per resident
                // sample (the census shows ~29k/session) is pure waste.
                #[cfg(feature = "backend-vulkan")]
                let resident_id =
                    crate::runtime::import_present::surface_identity(state, mid, w, h);
                let resident_ready = {
                    #[cfg(feature = "backend-vulkan")]
                    {
                        crate::backend::vulkan::engine::resident_content_ready(&resident_id)
                    }
                    #[cfg(not(feature = "backend-vulkan"))]
                    {
                        false
                    }
                };

                // A ready resident target is authoritative after a product
                // Store, exactly as it is for a subsequent attachment Load.
                // Bind it directly before touching full-frame CPU mirrors.
                #[cfg(feature = "backend-vulkan")]
                if resident_ready {
                    if let Some(v) =
                        try_sample_resident_surface(texture_ref, mid, w, h, resident_id)
                    {
                        crate::runtime::census::setup_tex_census::note_bind(
                            t_bind.elapsed().as_micros() as u64,
                        );
                        sampled_census::note(sampled_census::Branch::Resident, 0);
                        return Some(v);
                    }
                }
                crate::runtime::census::setup_tex_census::note_bind(
                    t_bind.elapsed().as_micros() as u64
                );

                // 1) Host cache.
                if let Some(bgra) = crate::runtime::surface_cache::get(state, mid, w, h) {
                    let rgba = swap_rb_channels(bgra);
                    let (nz, _, _) = crate::observe::rgba_rgb_stats(&rgba);
                    if nz > 0 || !resident_ready {
                        log_sample_src(texture_ref, mid, w, h, "cache", &rgba, resident_ready);
                        sampled_census::note(sampled_census::Branch::T11Cache, rgba.len());
                        return Some((
                            w,
                            h,
                            mid,
                            SampledSourceRequest::Bytes(
                                std::sync::Arc::new(rgba),
                                None,
                                TexelLayout::Rgba8,
                            ),
                        ));
                    }
                }

                // 2) Guest pages. When no resident is authoritative the guest
                // bytes are taken unconditionally (no nz promotion check), so
                // the zero-copy gather can bind them without CPU bytes; with a
                // ready resident the CPU load below keeps the empty→resident
                // promotion (needs the nz stat).
                // Why the zero-copy gather declined this bind — recorded on the
                // CPU t11_guest load below so a boot names the dominant lever
                // (below_floor / resident_gated / stride / …). ResidentGated
                // covers the case a ready resident pre-empts the gather (the
                // gather is only attempted when `!resident_ready`).
                #[allow(unused_mut)]
                let mut t11_zc_decline = t11_decline::Reason::ResidentGated;
                #[cfg(feature = "backend-vulkan")]
                if !resident_ready {
                    // Always-on discriminator (rare post group/incarnation
                    // fixes): a large mapping sampled from guest pages while
                    // the registry holds a same-surface resident under ANOTHER
                    // generation is the generation-orphaning signature
                    // (black-band class). Ring + counter
                    // surface in `rect_void_ctx` / THRASH `t11_fb=`.
                    if (w as u64) * (h as u64) >= 250_000 {
                        let map_gen = state
                            .mappings
                            .get(&mid)
                            .map(|m| m.map_generation)
                            .unwrap_or(0);
                        let probe = crate::backend::vulkan::engine::resident_probe_surface_any_gen(
                            mid, w, h,
                        );
                        crate::runtime::census::present_proxy::note_t11_large_fallback(
                            mid, map_gen, probe,
                        );
                        if crate::observe::enabled() {
                            crate::observe::line(format!(
                                "t11_zc_fallback mid={mid} {w}x{h} map_gen={map_gen} probe={probe:?}"
                            ));
                        }
                    }
                    match try_type11_sample_zero_copy(state, host, mid, w, h) {
                        Ok(src) => return Some((w, h, mid, src)),
                        Err(reason) => t11_zc_decline = reason,
                    }
                }
                // This path is reached only with `resident_ready == false` (a
                // ready resident returns above via `try_sample_resident_surface`),
                // so the guest bytes are taken unconditionally — the historical
                // `nz > 0 || !resident_ready` promotion gate is always true here.
                // The memo skips the convert/alloc on unchanged content and
                // returns a content identity so the engine skips re-hash+upload;
                // its census (T11Memo hit / T11Guest fill) is emitted internally.
                if let Some((rgba, identity)) = load_type11_rgba_memoized(state, host, mid) {
                    log_sample_src(texture_ref, mid, w, h, "guest_memo", &rgba, resident_ready);
                    t11_decline::note(t11_zc_decline, rgba.len());
                    return Some((
                        w,
                        h,
                        mid,
                        SampledSourceRequest::Bytes(rgba, Some(identity), TexelLayout::Rgba8),
                    ));
                }

                if w >= 1280 && h >= 720 {
                    crate::observe::fail(format!(
                        "sample_src=miss ref={texture_ref} mid={mid} {w}x{h} resident_ready={} (no guest/cache/resident bytes)",
                        resident_ready as u8
                    ));
                } else if (w as u64) * (h as u64) >= 250_000 {
                    // App-window-sized surfaces (e.g. a 1240x702 browser content
                    // layer) sit under the full-screen gate above; a persistent
                    // miss there paints the window blank with zero log. Latch
                    // per (mid, geom) so a steady repeat stays at one line.
                    use std::collections::HashSet;
                    use std::sync::Mutex;
                    static SEEN: Mutex<Option<HashSet<(u32, u32, u32)>>> = Mutex::new(None);
                    let mut guard = SEEN.lock().unwrap();
                    if guard.get_or_insert_with(HashSet::new).insert((mid, w, h)) {
                        crate::observe::fail(format!(
                            "sample_src=miss ref={texture_ref} mid={mid} {w}x{h} resident_ready={} latched=1 (no guest/cache/resident bytes)",
                            resident_ready as u8
                        ));
                    }
                }
            }
        }
    }

    // Type-2/3: GVA-keyed encode, then texture_ref with **descriptor** geom match.
    if is_linear_tex {
        // A still-deferred GVA render Store is GPU-resident and authoritative;
        // bind the resident target directly instead of flushing + re-uploading
        // (the gvadefer A/B showed 99% of windows were consumed by exactly this
        // sample path — readback relocation, not elimination).
        #[cfg(feature = "backend-vulkan")]
        if let Some(v) = try_sample_deferred_gva(state, host, task_id, texture_ref) {
            return Some(v);
        }
        // MRT secondary (e.g. the vibrancy RG16Float coverage mask): rendered
        // this frame as an engine secondary resident, not a deferred-flush
        // window. Bind it directly so the material's alpha modulation reads the
        // real mask instead of zero (frosted-background pass-through).
        #[cfg(feature = "backend-vulkan")]
        if let Some(v) = try_sample_mrt_secondary(state, host, task_id, texture_ref) {
            return Some(v);
        }
        // Zero-copy gather for large Vulkan-native linear textures: replaces
        // the CPU host-cache/memo byte paths below for eligible formats (the
        // lin_memo full-window re-read + memcmp per bind was the dominant
        // per-draw cost under compositor load).
        // Resolve + decode the texture descriptor ONCE for both linear loaders
        // below. The zero-copy attempt (which returns None on the ~35k/session
        // cache-fallback majority) and the host-cache fallback each read the same
        // descriptor blob and run the identical `decode_texture_descriptor`; the
        // object list is immutable for the draw, so one read+decode serves both.
        // `try_linear_sample_zero_copy` uses only the decoded descriptor;
        // `load_linear_from_host_caches` also needs `entry.object_type` for the
        // gva-cache owner check.
        if let (Some(le), Some(tex)) = (
            resolved_entry,
            resolved_entry.and_then(|e| {
                objects::read_descriptor(state, host, task_id, &e)
                    .and_then(|d| decode_texture_descriptor(&d).ok())
            }),
        ) {
            #[cfg(feature = "backend-vulkan")]
            if let Some((w, h, src)) =
                try_linear_sample_zero_copy(state, host, task_id, texture_ref, &tex)
            {
                return Some((w, h, 0, src));
            }
            if let Some((w, h, rgba, identity, byte_format)) =
                load_linear_from_host_caches(state, host, task_id, texture_ref, &le, &tex)
            {
                return Some((
                    w,
                    h,
                    0,
                    SampledSourceRequest::Bytes(rgba, identity, byte_format),
                ));
            }
        }
    }

    // Fallback: texture_ref encode cache any size (unit tests without object list).
    if let Some((w, h, bgra)) = crate::runtime::surface_cache::get_texture_any(state, texture_ref) {
        let need = (w as usize).saturating_mul(h as usize).saturating_mul(4);
        if bgra.len() >= need {
            let rgba = swap_rb_channels(&bgra[..need]);
            sampled_census::note(sampled_census::Branch::TexrefAny, rgba.len());
            return Some((
                w,
                h,
                0,
                SampledSourceRequest::Bytes(std::sync::Arc::new(rgba), None, TexelLayout::Rgba8),
            ));
        }
    }

    // Linear / view path returns only RGBA; recover geom from texture descriptor.
    let mut rgba = load_sampled_rgba_static(state, host, task_id, texture_ref)?;
    let entry = objects::lookup_list_entry(state, host, task_id, texture_ref)?;
    let desc = objects::read_descriptor(state, host, task_id, &entry)?;
    if let Ok(td) = decode_texture_descriptor(&desc) {
        let w = td.width.max(1);
        let h = td.height.max(1);
        let need = (w as usize).saturating_mul(h as usize).saturating_mul(4);
        if rgba.len() >= need {
            rgba.truncate(need);
            sampled_census::note(sampled_census::Branch::StaticTail, rgba.len());
            return Some((
                w,
                h,
                0,
                SampledSourceRequest::Bytes(std::sync::Arc::new(rgba), None, TexelLayout::Rgba8),
            ));
        }
    }
    // Fall back: assume square-ish from byte length.
    let px = rgba.len() / 4;
    if px == 0 {
        return None;
    }
    let side = (px as f64).sqrt() as u32;
    if side > 0 && (side as usize) * (side as usize) * 4 == rgba.len() {
        sampled_census::note(sampled_census::Branch::StaticTail, rgba.len());
        return Some((
            side,
            side,
            0,
            SampledSourceRequest::Bytes(std::sync::Arc::new(rgba), None, TexelLayout::Rgba8),
        ));
    }
    None
}

#[inline]
fn type5_view_requires_materialization(
    base_has_geom: bool,
    base_width: u32,
    base_height: u32,
    base_format: u16,
    view: objects::Type5TextureView,
) -> bool {
    !base_has_geom
        || view.depth != 1
        || base_format == 0
        || base_width != view.width
        || base_height != view.height
        || base_format != view.pixel_format
}

/// The decoded device-surface fields a failed sample-window derivation dumps
/// for diagnosis: `(width, height, pixel_format, bytes_per_row, alloc_size)`.
type SampleWindowDesc = (u32, u32, u32, u32, u32);

/// Why the type-5 serialized-view loader refused to materialize a plane.
///
/// # Why these slugs are prefixed `type5_view_`
///
/// The blit rail's `BlitStatus` already owns a `t5_*` vocabulary for the type-5
/// *copy* path (`t5_no_mapping`, `t5_sample_window`, `t5_fmt_bpp`,
/// `t5_unmapped`), and four of this loader's checks are conceptually the same
/// words. A bare `no_mapping` was in fact one of three claimants — console
/// capture, guest-page import and this loader — that the last present-rail
/// migration recorded as still sharing the word. The `type5_view_` prefix keeps
/// `grep reason=type5_view_…` answerable against the copy path that shares the
/// surface; crate-wide distinctness is `observe::gate`'s job.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Type5ViewDecline {
    /// The serialized view is volumetric; only `depth == 1` planes materialize.
    UnsupportedDepth { depth: u32 },
    /// The mapping's page table is not resident for scanout.
    Unresolved,
    /// The view's MTLPixelFormat has no known bytes-per-pixel.
    FormatBpp,
    /// The mapping id has no live entry.
    NoMapping,
    /// No sample window could be derived from the device descriptor for this
    /// plane geometry. Carries the base geometry and the decoded descriptor (or
    /// its absence) that disagreed.
    SampleWindow {
        base_w: u32,
        base_h: u32,
        base_fmt: u16,
        desc: Option<SampleWindowDesc>,
    },
    /// The mapping's resident pages span fewer bytes than the sample window
    /// ends at.
    Span {
        pages: usize,
        page_bytes: u64,
        span_end: u64,
        bpr: u32,
    },
    /// `width * bpp` overflowed a u32, so a tight row is unrepresentable.
    TightOverflow { bpp: u32 },
    /// The native plane byte length overflowed the host allocation cap.
    NativeLen { tight: u32 },
    /// The native plane window could not be read from guest memory.
    Read {
        base_w: u32,
        base_h: u32,
        base_fmt: u16,
        off: u64,
        bpr: u32,
        span_end: u64,
        pages: usize,
    },
    /// `width * 4` overflowed a u32, so the RGBA row is unrepresentable.
    RgbaStride,
    /// The RGBA buffer length overflowed the host allocation cap.
    RgbaLen { stride: u32 },
    /// A row failed to convert from the native format into RGBA8.
    Convert { row: usize, bpp: u32 },
}

impl crate::observe::Decline for Type5ViewDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::UnsupportedDepth { .. } => "type5_view_unsupported_depth",
            Self::Unresolved => "type5_view_unresolved",
            Self::FormatBpp => "type5_view_format_bpp",
            Self::NoMapping => "type5_view_no_mapping",
            Self::SampleWindow { .. } => "type5_view_sample_window",
            Self::Span { .. } => "type5_view_span",
            Self::TightOverflow { .. } => "type5_view_tight_overflow",
            Self::NativeLen { .. } => "type5_view_native_len",
            Self::Read { .. } => "type5_view_read",
            Self::RgbaStride => "type5_view_rgba_stride",
            Self::RgbaLen { .. } => "type5_view_rgba_len",
            Self::Convert { .. } => "type5_view_convert",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::UnsupportedDepth { depth } => vec![("depth", depth.to_string())],
            Self::SampleWindow {
                base_w,
                base_h,
                base_fmt,
                desc,
            } => {
                let mut v = vec![
                    ("base", format!("{base_w}x{base_h}")),
                    ("base_fmt", format!("{base_fmt:#x}")),
                ];
                match desc {
                    Some((w, h, fmt, bpr, alloc)) => {
                        v.push(("desc", format!("{w}x{h}")));
                        v.push(("desc_fmt", format!("{fmt:#x}")));
                        v.push(("bpr", bpr.to_string()));
                        v.push(("alloc", alloc.to_string()));
                    }
                    None => v.push(("desc", "missing".to_string())),
                }
                v
            }
            Self::Span {
                pages,
                page_bytes,
                span_end,
                bpr,
            } => vec![
                ("pages", pages.to_string()),
                ("page_bytes", page_bytes.to_string()),
                ("span_end", span_end.to_string()),
                ("bpr", bpr.to_string()),
            ],
            Self::TightOverflow { bpp } => vec![("bpp", bpp.to_string())],
            Self::NativeLen { tight } => vec![("tight", tight.to_string())],
            Self::Read {
                base_w,
                base_h,
                base_fmt,
                off,
                bpr,
                span_end,
                pages,
            } => vec![
                ("base", format!("{base_w}x{base_h}")),
                ("base_fmt", format!("{base_fmt:#x}")),
                ("off", off.to_string()),
                ("bpr", bpr.to_string()),
                ("span_end", span_end.to_string()),
                ("pages", pages.to_string()),
            ],
            Self::RgbaLen { stride } => vec![("stride", stride.to_string())],
            Self::Convert { row, bpp } => {
                vec![("row", row.to_string()), ("bpp", bpp.to_string())]
            }
            Self::Unresolved | Self::FormatBpp | Self::NoMapping | Self::RgbaStride => Vec::new(),
        }
    }
}

/// Materialize the exact serialized Metal view carried by a type-5 object.
///
/// The underlying type-4 FourCC is allocation metadata, not necessarily the
/// sampled Metal format. The view's format/geometry define the native row
/// interpretation; the type-4 device descriptor supplies its base/BPR/span.
/// Materialize a type-5 serialized texture view through the byte-exact
/// revalidated memo (same contract as [`load_linear_guest_memoized`]): every
/// bind re-reads the native plane window so a guest write is always observed;
/// conversion, allocation, and — via the returned content identity — the
/// engine upload are skipped when the bytes are unchanged.
fn load_type5_view_rgba<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    texture_ref: u32,
    mapping_id: u32,
    view: objects::Type5TextureView,
) -> Option<LoadedType5View> {
    let fail = |d: Type5ViewDecline| -> Option<LoadedType5View> {
        crate::observe::Emit::decline("type5_draw_view", &d)
            .field("task", task_id)
            .field("ref", texture_ref)
            .field("sid", mapping_id)
            .field("view", format!("{}x{}", view.width, view.height))
            .field("fmt", format!("{:#x}", view.pixel_format))
            .fail();
        None
    };

    if view.depth != 1 {
        return fail(Type5ViewDecline::UnsupportedDepth { depth: view.depth });
    }
    if !mapper::ensure_resolved_for_scanout(state, host, mapping_id) {
        return fail(Type5ViewDecline::Unresolved);
    }
    let Some(bpp) = pixel_format::bytes_per_pixel(view.pixel_format) else {
        return fail(Type5ViewDecline::FormatBpp);
    };
    let (base_off, surface_bpr, span_end, pages_n, base_w, base_h, base_fmt, map_gen, from_device) = {
        let Some(m) = state.mappings.get(&mapping_id) else {
            return fail(Type5ViewDecline::NoMapping);
        };
        let Some((base_off, surface_bpr, span_end, from_device)) =
            mapping_write::type5_sample_window(
                m,
                view.plane_index,
                view.width,
                view.height,
                view.pixel_format,
            )
        else {
            let desc =
                crate::contract::iosurface_pages::decode_device_surface(&m.device_desc).map(|d| {
                    (
                        d.width,
                        d.height,
                        d.pixel_format,
                        d.bytes_per_row,
                        d.alloc_size,
                    )
                });
            return fail(Type5ViewDecline::SampleWindow {
                base_w: m.width,
                base_h: m.height,
                base_fmt: m.format,
                desc,
            });
        };
        (
            base_off,
            surface_bpr,
            span_end,
            m.page_entries.len(),
            m.width,
            m.height,
            m.format,
            m.map_generation,
            from_device,
        )
    };
    let page_bytes = (pages_n as u64).saturating_mul(1u64 << state.page_shift);
    if page_bytes < span_end {
        return fail(Type5ViewDecline::Span {
            pages: pages_n,
            page_bytes,
            span_end,
            bpr: surface_bpr,
        });
    }
    let Some(tight) = view.width.checked_mul(bpp) else {
        return fail(Type5ViewDecline::TightOverflow { bpp });
    };
    let Some(native_len) = (tight as u64)
        .checked_mul(view.height as u64)
        .and_then(host_alloc_len)
    else {
        return fail(Type5ViewDecline::NativeLen { tight });
    };
    let mut native = vec![0u8; native_len];
    if !mapping_write::read_rect_raw_at(
        state,
        host,
        mapping_id,
        base_off,
        surface_bpr,
        span_end,
        0,
        0,
        view.width,
        view.height,
        bpp,
        &mut native,
        tight,
    ) {
        return fail(Type5ViewDecline::Read {
            base_w,
            base_h,
            base_fmt,
            off: base_off,
            bpr: surface_bpr,
            span_end,
            pages: pages_n,
        });
    }
    // Identity key namespace: bit 63 marks type-5 view content (guest linear
    // identities use the raw sampled GVA as key). Generations come from the
    // shared `guest_linear_gen` counter, so a (key, generation) pair can never
    // alias content across producers even on a key collision.
    let identity_key = (1u64 << 63) | ((view.plane_index as u64) << 32) | mapping_id as u64;
    let memo_key = (
        mapping_id,
        view.plane_index,
        view.width,
        view.height,
        view.pixel_format,
    );
    // A single/dual-channel plane (biplanar video Y = R8, CbCr = RG8) uploads at
    // its native footprint: `texel_to_rgba8` places R8→(r,0,0,255) and
    // RG8→(r,g,0,255), which is exactly what an R8_UNORM / R8G8_UNORM Vulkan
    // image samples to (`.r` / `.rg`, zero-filled tail). Skipping the CPU expand
    // and uploading native cuts 4×/2× the staging bytes with byte-exact texels.
    let byte_format = match view.pixel_format {
        pixel_format::MTL_FORMAT_R8_UNORM => TexelLayout::R8,
        pixel_format::MTL_FORMAT_RG8_UNORM => TexelLayout::Rg8,
        _ => TexelLayout::Rgba8,
    };
    let ok_line = |generation_source: &str, rgba: &[u8]| {
        // Per-draw success echo — fires on EVERY type-5 plane bind (thousands/sec
        // under video → ~36k lines/boot, 61% of the fail log), burying real
        // failures. The always-on health signal is the `sampled_branch_census`
        // aggregate (Type5View / T5Memo, noted on both paths below), so this
        // per-bind detail — and its O(w*h) `rgba_rgb_stats` scan — is diagnostic
        // only: gate both behind REIMS_VGPU_DRAW_LOG so a normal boot stays uncluttered.
        if !crate::observe::draw_log_enabled() {
            return;
        }
        let (nz, max, _) = crate::observe::rgba_rgb_stats(rgba);
        crate::observe::line(format!(
            "type5_draw_view ok task={task_id} ref={texture_ref} sid={mapping_id} map_gen={map_gen} view={}x{} fmt={:#x} bpp={bpp} base={base_w}x{base_h} base_fmt={base_fmt:#x} off={base_off} bpr={surface_bpr} span_end={span_end} invent={} src={generation_source} rgb_nz={nz} max_rgb={max}",
            view.width, view.height, view.pixel_format, (!from_device) as u8
        ));
    };
    if let Some(m) = state.type5_view_memo.get_touch(&memo_key) {
        // Vec equality is length + byte memcmp with early exit on change.
        if m.native == native {
            let rgba = m.rgba.clone();
            let generation = m.generation;
            ok_line("memo", &rgba);
            sampled_census::note(sampled_census::Branch::T5Memo, 0);
            return Some((
                view.width,
                view.height,
                rgba,
                LinearSampleIdentity {
                    key: identity_key,
                    generation,
                },
                byte_format,
            ));
        }
    }
    // RGBA8 formats expand per-pixel into a fresh RGBA8 buffer; native R8/RG8
    // upload the plane bytes verbatim (the memo stores those bytes as both the
    // memcmp key and the upload payload).
    let rgba: std::sync::Arc<Vec<u8>> = if byte_format == TexelLayout::Rgba8 {
        let Some(rgba_stride) = view.width.checked_mul(RGBA8_BPP) else {
            return fail(Type5ViewDecline::RgbaStride);
        };
        let Some(rgba_len) = (rgba_stride as u64)
            .checked_mul(view.height as u64)
            .and_then(host_alloc_len)
        else {
            return fail(Type5ViewDecline::RgbaLen {
                stride: rgba_stride,
            });
        };
        let mut rgba = vec![0u8; rgba_len];
        for y in 0..view.height as usize {
            let src_off = y.saturating_mul(tight as usize);
            let dst_off = y.saturating_mul(rgba_stride as usize);
            if !pixel_format::convert_row_to_rgba8(
                view.pixel_format,
                &native[src_off..src_off + tight as usize],
                view.width,
                &mut rgba[dst_off..dst_off + rgba_stride as usize],
            ) {
                return fail(Type5ViewDecline::Convert { row: y, bpp });
            }
        }
        std::sync::Arc::new(rgba)
    } else {
        std::sync::Arc::new(native.clone())
    };
    state.guest_linear_gen += 1;
    let generation = GUEST_LINEAR_GEN_BASE + state.guest_linear_gen;
    ok_line("fill", &rgba);
    let entry_bytes = native.len() + rgba.len();
    state.type5_view_memo.insert(
        memo_key,
        crate::model::GuestLinearMemo {
            native,
            rgba: rgba.clone(),
            // The type-5 view path carries its own native format (R8/Rg8/…);
            // this reused struct's `bgra8` flag is only read by the guest-linear
            // memo, so it is not load-bearing here.
            bgra8: false,
            generation,
        },
        entry_bytes,
    );
    sampled_census::note(sampled_census::Branch::Type5View, rgba.len());
    Some((
        view.width,
        view.height,
        rgba,
        LinearSampleIdentity {
            key: identity_key,
            generation,
        },
        byte_format,
    ))
}

/// Type-2/3 sample: GVA encode cache → texture_ref cache (descriptor geom) → guest GVA.
#[cfg(feature = "backend-vulkan")]
/// Serve the memoized swizzled RGBA only when it matches the authoritative
/// `host_gva_surfaces` entry on every axis (gva, generation, geometry) - a
/// stale memo entry is skipped, never served.
/// Zero-copy floor: below this the CPU byte path (one small read + memo) is
/// cheaper than a cached-window import plus a recorded GPU gather. Performance
/// threshold only — never a correctness gate.
///
/// Set to 64 KiB from a video-playback census (`t11_zc_decline`): after the
/// type-5 plane rail landed, the whole remaining CPU copy under video was
/// `t11_guest` (~226 MB/session), and 100% of those declines were `below_floor`
/// — per-frame-changing composite surfaces clustered at ~236 KiB, just under
/// the old 256 KiB. No memo can help (content changes every frame), so the CPU
/// path re-read + swizzled + double-SipHashed + re-uploaded ~236 KiB per frame
/// for nothing the GPU gather couldn't do from an already-imported (cached)
/// window. 64 KiB sits ~2× above the largest small-texture band that still
/// legitimately prefers the CPU byte path (small-UI / gva_copy binds measured at
/// ~21–34 KiB, and scroll glyphs at ~3.6 KiB served by the memo) and ~3.7×
/// below the video surfaces, so the band it opens to zero-copy is exactly those
/// per-frame video composites. Import windows are 1 GiB-bucketed and cached, so
/// lowering the floor does not multiply host imports.
#[cfg(feature = "backend-vulkan")]
const ZERO_COPY_SAMPLED_MIN_BYTES: u64 = 64 * 1024;

/// Zero-copy floor for draw-time vertex/storage buffer binds: below this the
/// CPU staging read is cheaper than a page walk plus a recorded GPU gather.
/// Performance threshold only — never a correctness gate.
#[cfg(feature = "backend-vulkan")]
const ZERO_COPY_BUFFER_MIN_BYTES: u64 = 16 * 1024;

/// Walk `span` bytes of `task_id`'s GVA space from `gva` and return the
/// packed guest-RAM runs covering it (GPA-contiguous stretches coalesced and
/// mapped to stable host pointers). `None` when any page is unmapped or the
/// mapping is incomplete. Shared by the sampled and buffer zero-copy rails;
/// callers must land intersecting deferred stores first and verify import
/// coverage per run.
#[cfg(feature = "backend-vulkan")]
fn task_gva_guest_runs<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &mut M,
    task_id: u32,
    gva: u64,
    span: u64,
) -> Option<Vec<crate::backend::vulkan::engine::GuestRun>> {
    use crate::backend::vulkan::engine;
    if !host.map_pages_stable() {
        return None;
    }
    let page = state.page_size();
    let mut gpas: Vec<u64> = Vec::new();
    gva_mem::visit_task_gva_page_gpas(
        host,
        &state.tasks,
        task_id,
        gva,
        span,
        state.page_shift,
        1,
        &mut |gpa| {
            gpas.push(gpa);
            true
        },
    );
    let expect = ((gva % page) + span).div_ceil(page);
    if gpas.len() as u64 != expect {
        return None;
    }
    let head_off = gva % page;
    let mut runs: Vec<engine::GuestRun> = Vec::new();
    let mut consumed = 0u64;
    let mut i = 0usize;
    while i < gpas.len() && consumed < span {
        let mut j = i + 1;
        while j < gpas.len() && gpas[j] == gpas[i] + ((j - i) as u64) * page {
            j += 1;
        }
        let base = host.map_pages(&gpas[i..j], page as usize)? as u64;
        let start_in_run = if i == 0 { head_off } else { 0 };
        let avail = ((j - i) as u64) * page - start_in_run;
        let len = avail.min(span - consumed);
        runs.push(engine::GuestRun {
            host_ptr: (base + start_in_run) as usize,
            len,
        });
        consumed += len;
        i = j;
    }
    if consumed != span {
        return None;
    }
    Some(runs)
}

/// Cap on [`DeviceState::guest_run_memo`] entries (FIFO evict). Entries are a
/// few dozen bytes each; the cap only bounds pathological churn.
const GUEST_RUN_MEMO_CAP: usize = 512;

/// Memoized [`task_gva_guest_runs`]: resolve `[gva, gva+span)` under the task
/// PT to host-VA runs, caching the result in `state.guest_run_memo`. The walk
/// was the dominant per-draw setup cost (~60 PT-leaf translations per 260 KB
/// bind); a hit skips it entirely. Invalidation contract = `gva_host_views`
/// (retired on Unmap/Map overlap, task redefine/delete — see gva_view.rs).
#[cfg(feature = "backend-vulkan")]
fn guest_runs_memoized<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    gva: u64,
    span: u64,
) -> Option<std::sync::Arc<Vec<crate::model::GuestRunSpan>>> {
    if !host.map_pages_stable() {
        return None;
    }
    if let Some(pos) = state.guest_run_memo.iter().position(|e| {
        crate::runtime::gva_view::task_matches(e.task_id, task_id)
            && e.gva == gva
            && e.length == span
    }) {
        state.tranche.run_memo_hit = state.tranche.run_memo_hit.saturating_add(1);
        // Sampled staleness verify (1-in-64 hits): the invalidation contract
        // (Unmap/Map2 overlap, task redefine/delete) should retire every
        // rewired range, but a PT change our notifies miss would serve stale
        // host runs silently — wrong-page GPU reads, the black-tile class.
        // A mismatch is fail-visible, self-heals the entry, and bounds any
        // contract hole to at most 64 stale draws per entry.
        if state.tranche.run_memo_hit.is_multiple_of(64) {
            match task_gva_guest_runs(state, host, task_id, gva, span) {
                Some(fresh) => {
                    let memo = state.guest_run_memo[pos].runs.clone();
                    let same = fresh.len() == memo.len()
                        && fresh
                            .iter()
                            .zip(memo.iter())
                            .all(|(f, m)| f.host_ptr == m.host_ptr && f.len == m.len);
                    if !same {
                        state.tranche.run_memo_stale =
                            state.tranche.run_memo_stale.saturating_add(1);
                        crate::observe::fail(format!(
                            "rmemo_stale task={task_id} gva={gva:#x} span={span:#x} memo_runs={} fresh_runs={}",
                            memo.len(),
                            fresh.len()
                        ));
                        let spans: std::sync::Arc<Vec<crate::model::GuestRunSpan>> =
                            std::sync::Arc::new(
                                fresh
                                    .iter()
                                    .map(|r| crate::model::GuestRunSpan {
                                        host_ptr: r.host_ptr,
                                        len: r.len,
                                    })
                                    .collect(),
                            );
                        state.guest_run_memo[pos].runs = spans.clone();
                        return Some(spans);
                    }
                }
                None => {
                    // The span no longer walks — the entry outlived its
                    // mapping. Retire it and make the caller fall back.
                    state.tranche.run_memo_stale = state.tranche.run_memo_stale.saturating_add(1);
                    crate::observe::fail(format!(
                        "rmemo_stale task={task_id} gva={gva:#x} span={span:#x} reason=walk_gone"
                    ));
                    state.guest_run_memo.remove(pos);
                    return None;
                }
            }
        }
        return Some(state.guest_run_memo[pos].runs.clone());
    }
    let runs = task_gva_guest_runs(state, host, task_id, gva, span)?;
    let spans: std::sync::Arc<Vec<crate::model::GuestRunSpan>> = std::sync::Arc::new(
        runs.iter()
            .map(|r| crate::model::GuestRunSpan {
                host_ptr: r.host_ptr,
                len: r.len,
            })
            .collect(),
    );
    if state.guest_run_memo.len() >= GUEST_RUN_MEMO_CAP {
        state.guest_run_memo.pop_front();
    }
    state
        .guest_run_memo
        .push_back(crate::model::GuestRunMemoEntry {
            task_id,
            gva,
            length: span,
            runs: spans.clone(),
        });
    state.tranche.run_memo_miss = state.tranche.run_memo_miss.saturating_add(1);
    Some(spans)
}

/// Slice memoized whole-span runs to the engine runs for `[offset, offset+span)`.
/// `None` when the slice exceeds the memoized extent (caller falls back).
#[cfg(feature = "backend-vulkan")]
fn slice_runs_to_engine(
    spans: &[crate::model::GuestRunSpan],
    offset: u64,
    span: u64,
) -> Option<Vec<crate::backend::vulkan::engine::GuestRun>> {
    use crate::backend::vulkan::engine;
    if span == 0 {
        return None;
    }
    let mut out: Vec<engine::GuestRun> = Vec::new();
    let mut skip = offset;
    let mut need = span;
    for r in spans {
        if need == 0 {
            break;
        }
        if skip >= r.len {
            skip -= r.len;
            continue;
        }
        let take = (r.len - skip).min(need);
        out.push(engine::GuestRun {
            host_ptr: r.host_ptr.checked_add(usize::try_from(skip).ok()?)?,
            len: take,
        });
        need -= take;
        skip = 0;
    }
    if need != 0 {
        return None;
    }
    Some(out)
}

/// Zero-copy draw-time buffer bind: resolve a type-1 buffer object's backing
/// span (from `offset`) to guest-RAM runs and hand the engine a
/// [`engine::BufferContent::GuestRuns`] — the GPU gathers the bytes from
/// imported guest RAM inside the draw's own CB. Replaces the per-draw CPU
/// re-read + double memcpy of the same ~50–260 KB vertex/SSBO buffers.
/// Guest CPU writes are still observed: the gather re-executes every draw
/// and reads at execute time (at least as fresh as the CPU path).
///
/// Gates (any miss → `None`, caller stays on the CPU staging read): span ≥
/// the buffer zero-copy floor, every page walkable into mappable runs, and
/// every run covered by a host import. Deferred stores intersecting the
/// span are landed first, exactly like the CPU path.
#[cfg(feature = "backend-vulkan")]
fn try_buffer_zero_copy_resolved<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    backing: &BufferBacking,
    offset: u64,
) -> Option<crate::backend::vulkan::engine::BufferContent> {
    use crate::backend::vulkan::engine;
    let (gva, size) = (backing.gva, backing.size);
    if offset >= size {
        return None;
    }
    let span = host_alloc_len(size - offset).filter(|&n| n > 0)? as u64;
    if span < ZERO_COPY_BUFFER_MIN_BYTES {
        return None;
    }
    if !host.map_pages_stable() {
        return None;
    }
    // Same coherence rule as the CPU read: land any resident-authoritative
    // writeback aliasing the span before the GPU reads the pages (the CPU
    // flush completes before this draw's submit).
    let t_flush = std::time::Instant::now();
    crate::runtime::storage_flush::flush_intersecting_task_gva(
        state,
        host,
        task_id,
        gva + offset,
        span,
    );
    let flush_ns = t_flush.elapsed().as_nanos() as u64;
    state.tranche.zc_flush_ns = state.tranche.zc_flush_ns.saturating_add(flush_ns);
    // Count the calls whose wall time zc_flush_ns sums, so the residual can be
    // divided to a true per-call cost (the isect/walk/sig sub-timers span all
    // flush call sites, not just this one). zc_flush_slow / zc_flush_max_ns
    // separate a uniform per-call memory stall (reducible) from rare preemption
    // spikes (scheduler noise): a > 100 µs call is off-CPU, not steady work.
    state.tranche.zc_flush_calls = state.tranche.zc_flush_calls.saturating_add(1);
    if flush_ns > 100_000 {
        state.tranche.zc_flush_slow = state.tranche.zc_flush_slow.saturating_add(1);
    }
    state.tranche.zc_flush_max_ns = state.tranche.zc_flush_max_ns.max(flush_ns);
    // Resolve via the whole-backing memo (bind offsets slide within the same
    // buffer allocation, so one memo entry serves every offset). Fall back to
    // the direct span walk when the whole backing does not resolve (e.g. an
    // unmapped tail page beyond the bound span).
    let runs = match host_alloc_len(size)
        .map(|n| n as u64)
        .and_then(|whole| guest_runs_memoized(state, host, task_id, gva, whole))
        .and_then(|spans| slice_runs_to_engine(&spans, offset, span))
    {
        Some(runs) => runs,
        None => task_gva_guest_runs(state, host, task_id, gva + offset, span)?,
    };
    let t_import = std::time::Instant::now();
    for r in &runs {
        if !engine::ensure_host_import(r.host_ptr, r.len) {
            state.tranche.zc_import_ns = state
                .tranche
                .zc_import_ns
                .saturating_add(t_import.elapsed().as_nanos() as u64);
            state.tranche.zc_fail_import = state.tranche.zc_fail_import.saturating_add(1);
            return None;
        }
    }
    state.tranche.zc_import_ns = state
        .tranche
        .zc_import_ns
        .saturating_add(t_import.elapsed().as_nanos() as u64);
    Some(engine::BufferContent::GuestRuns(engine::GuestRunSource {
        runs: std::sync::Arc::new(runs),
        total_len: span,
        row_length_texels: 0,
    }))
}

/// Load one draw-time buffer bind: the zero-copy rail when allowed and
/// eligible, else the CPU staging read. `allow_zero_copy` is false for
/// buffers feeding Constant-step attributes (the engine prepends a CPU
/// base-instance prefix to those).
#[cfg(feature = "backend-vulkan")]
fn load_buffer_content<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    buffer_ref: u32,
    offset: u64,
    allow_zero_copy: bool,
) -> Option<crate::backend::vulkan::engine::BufferContent> {
    // Resolve the backing (object-list entry + descriptor) ONCE and share it
    // between the zero-copy attempt and the CPU fallback. Sub-floor binds used
    // to walk the task PT twice — once in the failed ZC attempt, once in the
    // CPU read.
    let t_resolve = std::time::Instant::now();
    let backing = resolve_buffer_backing(state, host, task_id, buffer_ref)?;
    state.tranche.buf_resolve_ns = state
        .tranche
        .buf_resolve_ns
        .saturating_add(t_resolve.elapsed().as_nanos() as u64);
    if allow_zero_copy {
        if let Some(content) = try_buffer_zero_copy_resolved(state, host, task_id, &backing, offset)
        {
            return Some(content);
        }
    }
    let t_read = std::time::Instant::now();
    let bytes = read_buffer_bytes_resolved(state, host, task_id, &backing, offset)?;
    state.tranche.buf_read_ns = state
        .tranche
        .buf_read_ns
        .saturating_add(t_read.elapsed().as_nanos() as u64);
    Some(crate::backend::vulkan::engine::BufferContent::from(bytes))
}

/// Zero-copy linear sampled bind: resolve the texture's tight level-0 GVA
/// window to packed-contiguous guest-RAM runs and hand the engine a
/// [`SampledSource::GuestRuns`] — the GPU gathers the texels from imported
/// guest RAM inside the draw's own CB. Replaces the lin_memo class's
/// full-window CPU re-read + memcmp per bind (guest CPU writes are still
/// observed: the GPU copy re-executes every draw and reads at execute time).
///
/// Gates (any miss → `None`, caller stays on the CPU byte paths): native
/// texel layout Vulkan samples identically (BGRA8/RGBA8 UNORM), tight rows,
/// window inside the allocation, span ≥ the zero-copy floor, every page
/// walkable, packed-contiguous runs mappable, and every run covered by a
/// host import.
#[cfg(feature = "backend-vulkan")]
fn try_linear_sample_zero_copy<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    _texture_ref: u32,
    tex: &TextureDescriptor,
) -> Option<(u32, u32, SampledSourceRequest)> {
    use crate::backend::vulkan::engine;
    // The object-list entry + descriptor are resolved+decoded once by the caller
    // (`resolve_sampled_source`'s linear branch) and threaded in as `tex`; the
    // cache fallback shares the same decode.
    if !tex.has_pixel_format {
        return None;
    }
    // sRGB variants ride the same rail as their linear siblings: the layout is
    // identical and the CPU loaders never decoded either. The qualifier is
    // still lost, so the census records it rather than letting the fold be
    // silent.
    // Four-byte colour (BGRA8/RGBA8) or a single-channel float LUT: all sample
    // byte-identically through the matching native Vulkan image. Other layouts
    // (R8/Rg8 video planes) keep their existing CPU/type-5 rails. `R32_SFLOAT`
    // additionally needs the optional linear-filter feature — LUTs are sampled
    // with interpolation — so it is gated on the host capability and otherwise
    // declines here, leaving the sample fail-visible (no CPU float loader arm).
    let native = match translate::pixel::sampled_pixels(tex.pixel_format) {
        Ok((layout, decline))
            if layout.is_four_byte_color()
                || layout == TexelLayout::R16Float
                || (layout == TexelLayout::R32Float
                    && engine::supports_sampled_r32f_linear_filter()) =>
        {
            if decline.is_some() {
                srgb_census::note_downgrade(
                    srgb_census::site::LINEAR_SAMPLE_ZERO_COPY,
                    tex.pixel_format,
                );
            }
            layout
        }
        _ => return None,
    };
    let bpp = native.bytes_per_texel();
    let (gva, layout) = tex.level_gva(0, state.page_shift)?;
    let (w, h) = (layout.width, layout.height);
    if w == 0 || h == 0 {
        return None;
    }
    let bpr = layout.row_stride;
    let tight = (w as u64).checked_mul(bpp as u64)?;
    // Padded strides ride the same rail: the buffer→image copy strides over
    // the padding via `bufferRowLength` (texel units, so bpr must be a texel
    // multiple). The window ends after the last row's texels — trailing
    // padding may not be mapped.
    if bpr < tight || bpr % bpp as u64 != 0 {
        return None;
    }
    let span = bpr
        .checked_mul(h.checked_sub(1)? as u64)?
        .checked_add(tight)?;
    // The min-byte floor keeps small four-byte textures on the cheaper CPU
    // memo/cache path. Single-channel float LUTs have no CPU loader arm
    // (`texel_to_rgba8` returns `None`), so this native gather is their only
    // correct rail — exempt them from the floor or a small display-profile LUT
    // would fall through to a failed resolve.
    if native.is_four_byte_color() && span < ZERO_COPY_SAMPLED_MIN_BYTES {
        return None;
    }
    if !host.map_pages_stable() {
        return None;
    }
    let row_length_texels = if bpr == tight {
        0
    } else {
        u32::try_from(bpr / bpp as u64).ok()?
    };
    if tex.allocation_size != 0 && layout.offset.saturating_add(span) > tex.allocation_size {
        return None;
    }
    // Same coherence rule as the CPU loaders: land any resident-authoritative
    // writeback aliasing the span before the GPU reads the pages (the CPU
    // flush completes before this draw's submit).
    crate::runtime::storage_flush::flush_intersecting_task_gva(state, host, task_id, gva, span);
    // Fixed per-texture window — memoized directly (no offset slicing).
    let runs = match guest_runs_memoized(state, host, task_id, gva, span)
        .and_then(|spans| slice_runs_to_engine(&spans, 0, span))
    {
        Some(runs) => runs,
        None => task_gva_guest_runs(state, host, task_id, gva, span)?,
    };
    for r in &runs {
        if !engine::ensure_host_import(r.host_ptr, r.len) {
            return None;
        }
    }
    sampled_census::note(sampled_census::Branch::LinZeroCopy, 0);
    Some((
        w,
        h,
        SampledSourceRequest::GuestRuns(
            engine::GuestRunSource {
                runs: std::sync::Arc::new(runs),
                total_len: span,
                row_length_texels,
            },
            native,
        ),
    ))
}

/// Zero-copy rail for type-11 mapping-backed sampled binds. Eligible when
/// the mapping's raw bytes sample byte-identically through a native UNORM
/// image (BGRA8/RGBA8 families — the CPU loader's `texel_to_rgba8` is a
/// byte pass-through/swizzle for exactly these) and the caller established
/// the resident is not authoritative. Mirrors `paint_mapping`'s window math
/// (`type11_sample_window`) and its flush-on-access rule; any gate miss
/// falls back to the CPU byte path.
#[cfg(feature = "backend-vulkan")]
fn try_type11_sample_zero_copy<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mid: u32,
    w: u32,
    h: u32,
) -> Result<SampledSourceRequest, t11_decline::Reason> {
    use crate::backend::vulkan::engine;
    use crate::runtime::mapping_write::type11_sample_window;
    use t11_decline::Reason;
    if w == 0 || h == 0 {
        return Err(Reason::Unmapped);
    }
    let (native, base_off, bpr) = {
        let m = state.mappings.get(&mid).ok_or(Reason::Unmapped)?;
        if !m.mapped || m.page_entries.is_empty() {
            return Err(Reason::Unmapped);
        }
        let format = if m.format != 0 {
            m.format
        } else {
            pixel_format::MTL_FORMAT_BGRA8_UNORM
        };
        let native = match translate::pixel::sampled_pixels(format) {
            Ok((layout, decline)) if layout.is_four_byte_color() => {
                if decline.is_some() {
                    srgb_census::note_downgrade(srgb_census::site::TYPE11_SAMPLE_ZERO_COPY, format);
                }
                layout
            }
            _ => return Err(Reason::BadFormat),
        };
        let (base_off, bpr_u32, _span_end) =
            type11_sample_window(m, w, h, format).ok_or(Reason::NoWindow)?;
        (native, base_off, bpr_u32 as u64)
    };
    let tight = (w as u64)
        .checked_mul(RGBA8_BPP as u64)
        .ok_or(Reason::Stride)?;
    if bpr < tight || bpr % RGBA8_BPP as u64 != 0 {
        return Err(Reason::Stride);
    }
    let span = bpr
        .checked_mul((h - 1) as u64)
        .and_then(|v| v.checked_add(tight))
        .ok_or(Reason::Coverage)?;
    if span < ZERO_COPY_SAMPLED_MIN_BYTES {
        return Err(Reason::BelowFloor);
    }
    if !host.map_pages_stable() {
        return Err(Reason::UnstableMap);
    }
    // Land any resident-authoritative deferred window before the GPU reads
    // the pages (same coherence rule as paint_mapping / the linear rail).
    sampled_census::timed(sampled_census::Step::T11ZcFlush, || {
        let _ = crate::runtime::storage_flush::flush_intersecting(state, host, mid, 0, u64::MAX);
    });
    let gpas = sampled_census::timed(sampled_census::Step::T11ZcGpas, || {
        mapper::mapping_page_gpas(state, host, mid)
    })
    .ok_or(Reason::Coverage)?;
    let page = state.page_size();
    let window_end = base_off.checked_add(span).ok_or(Reason::Coverage)?;
    if (gpas.len() as u64).saturating_mul(page) < window_end {
        return Err(Reason::Coverage);
    }
    // Coalesce the pages covering [base_off, base_off+span) into packed host
    // runs (direct RAMBlock aliases from map_pages; unmap is a no-op).
    let first_page = (base_off / page) as usize;
    let head_off = base_off % page;
    let need_pages = (head_off + span).div_ceil(page) as usize;
    let window = gpas
        .get(first_page..first_page + need_pages)
        .ok_or(Reason::Coverage)?;
    let mut runs: Vec<engine::GuestRun> = Vec::new();
    let mut consumed = 0u64;
    let mut i = 0usize;
    while i < window.len() && consumed < span {
        let mut j = i + 1;
        while j < window.len() && window[j] == window[i] + ((j - i) as u64) * page {
            j += 1;
        }
        let base = sampled_census::timed(sampled_census::Step::T11ZcMap, || {
            host.map_pages(&window[i..j], page as usize)
        })
        .ok_or(Reason::ImportFail)? as u64;
        let start_in_run = if i == 0 { head_off } else { 0 };
        let avail = ((j - i) as u64) * page - start_in_run;
        let len = avail.min(span - consumed);
        runs.push(engine::GuestRun {
            host_ptr: (base + start_in_run) as usize,
            len,
        });
        consumed += len;
        i = j;
    }
    if consumed != span {
        return Err(Reason::ImportFail);
    }
    for r in &runs {
        if !sampled_census::timed(sampled_census::Step::T11ZcImport, || {
            engine::ensure_host_import(r.host_ptr, r.len)
        }) {
            return Err(Reason::ImportFail);
        }
    }
    let row_length_texels = if bpr == tight {
        0
    } else {
        u32::try_from(bpr / RGBA8_BPP as u64)
            .ok()
            .ok_or(Reason::Stride)?
    };
    sampled_census::note(sampled_census::Branch::T11ZeroCopy, 0);
    Ok(SampledSourceRequest::GuestRuns(
        engine::GuestRunSource {
            runs: std::sync::Arc::new(runs),
            total_len: span,
            row_length_texels,
        },
        native,
    ))
}

/// Zero-copy rail for a type-5 serialized IOSurface plane view — the video
/// hot path. VideoToolbox decodes to NV12 (Y = R8, CbCr = RG8; also
/// BGRA8/RGBA8 surfaces), sampled through the type-5 view path whose CPU
/// loader (`load_type5_view_rgba`) read + uploaded ~1.5 MB per plane per
/// decoded frame (census `t5_view`). This gathers the plane's guest pages
/// directly in the draw CB so the decoded frame never materializes CPU bytes.
/// Mirrors `try_type11_sample_zero_copy`'s page coalescing over the plane
/// window from `type5_sample_window` (which carries the wire plane index +
/// biplanar offset); any gate miss falls back to the CPU byte path.
#[cfg(feature = "backend-vulkan")]
fn try_type5_sample_zero_copy<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mid: u32,
    view: objects::Type5TextureView,
) -> Option<SampledSourceRequest> {
    use crate::backend::vulkan::engine;
    use crate::runtime::mapping_write::type5_sample_window;
    let (w, h) = (view.width, view.height);
    if w == 0 || h == 0 || view.depth != 1 {
        return None;
    }
    // Match the CPU path's resolution before reading the plane pages.
    if !mapper::ensure_resolved_for_scanout(state, host, mid) {
        return None;
    }
    let (native, bpp, base_off, bpr) = {
        let m = state.mappings.get(&mid)?;
        if !m.mapped || m.page_entries.is_empty() {
            return None;
        }
        // Native formats whose guest bytes sample byte-identically through the
        // matching Vulkan image (the CPU loader's `texel_to_rgba8` is a
        // pass-through/swizzle for exactly these); everything else stays CPU.
        // The texel size comes from the layout the translation chose, so it can
        // never disagree with the image the engine creates.
        let (native, bpp) = match translate::pixel::sampled_pixels(view.pixel_format) {
            Ok((layout, decline)) => {
                if decline.is_some() {
                    srgb_census::note_downgrade(
                        srgb_census::site::TYPE5_PLANE_ZERO_COPY,
                        view.pixel_format,
                    );
                }
                (layout, layout.bytes_per_texel())
            }
            Err(_) => return None,
        };
        // Only a real device-descriptor plane window rides zero copy; the
        // invented packed fallback over a stale multiplanar mapping (menu-strip
        // residual class) stays on the CPU path.
        let (base_off, bpr_u32, _span_end, from_device) =
            type5_sample_window(m, view.plane_index, w, h, view.pixel_format)?;
        if !from_device {
            return None;
        }
        (native, bpp, base_off, bpr_u32 as u64)
    };
    let tight = (w as u64).checked_mul(bpp as u64)?;
    if bpr < tight || bpr % bpp as u64 != 0 {
        return None;
    }
    let span = bpr.checked_mul((h - 1) as u64)?.checked_add(tight)?;
    if span < ZERO_COPY_SAMPLED_MIN_BYTES {
        return None;
    }
    if !host.map_pages_stable() {
        return None;
    }
    // Land any resident-authoritative deferred window before the GPU reads the
    // pages (same coherence rule as the CPU loader / the type-11 rail).
    let _ = crate::runtime::storage_flush::flush_intersecting(state, host, mid, 0, u64::MAX);
    let gpas = mapper::mapping_page_gpas(state, host, mid)?;
    let page = state.page_size();
    let window_end = base_off.checked_add(span)?;
    if (gpas.len() as u64).saturating_mul(page) < window_end {
        return None;
    }
    // Coalesce the pages covering [base_off, base_off+span) into packed host
    // runs (direct RAMBlock aliases from map_pages; unmap is a no-op).
    let first_page = (base_off / page) as usize;
    let head_off = base_off % page;
    let need_pages = (head_off + span).div_ceil(page) as usize;
    let window = gpas.get(first_page..first_page + need_pages)?;
    let mut runs: Vec<engine::GuestRun> = Vec::new();
    let mut consumed = 0u64;
    let mut i = 0usize;
    while i < window.len() && consumed < span {
        let mut j = i + 1;
        while j < window.len() && window[j] == window[i] + ((j - i) as u64) * page {
            j += 1;
        }
        let base = host.map_pages(&window[i..j], page as usize)? as u64;
        let start_in_run = if i == 0 { head_off } else { 0 };
        let avail = ((j - i) as u64) * page - start_in_run;
        let len = avail.min(span - consumed);
        runs.push(engine::GuestRun {
            host_ptr: (base + start_in_run) as usize,
            len,
        });
        consumed += len;
        i = j;
    }
    if consumed != span {
        return None;
    }
    for r in &runs {
        if !engine::ensure_host_import(r.host_ptr, r.len) {
            return None;
        }
    }
    let row_length_texels = if bpr == tight {
        0
    } else {
        u32::try_from(bpr / bpp as u64).ok()?
    };
    sampled_census::note(sampled_census::Branch::T5ZeroCopy, 0);
    Some(SampledSourceRequest::GuestRuns(
        engine::GuestRunSource {
            runs: std::sync::Arc::new(runs),
            total_len: span,
            row_length_texels,
        },
        native,
    ))
}

fn linear_sampled_memo_reuse(
    state: &DeviceState,
    task_id: u32,
    texture_ref: u32,
    gva: u64,
    host_gen: u32,
    width: u32,
    height: u32,
) -> Option<std::sync::Arc<Vec<u8>>> {
    // Peek (no recency bump): the caller already holds an immutable borrow of
    // `state` (the authoritative `bgra` view) across this call, so a `&mut`
    // touch is not possible here. Recency for this memo is instead driven by
    // inserts — each content change re-inserts and warms the entry — which is
    // sufficient for this authoritative-cache-backed reuse fast path.
    let m = state.linear_sampled_memo.peek(&(task_id, texture_ref))?;
    (m.gva == gva && m.host_gen == host_gen && m.width == width && m.height == height)
        .then(|| m.rgba.clone())
}

/// Generation namespace for guest-linear memo identities. Host-cache
/// generations (`host_gen`) are nonzero u32; guest-memo generations live
/// strictly above `u32::MAX` so the same `gva` identity key can never alias
/// content across the two producers.
const GUEST_LINEAR_GEN_BASE: u64 = 1 << 32;

/// Serve a guest-CPU-produced linear texture (tight OR padded row stride)
/// through the byte-exact revalidated memo. Every call re-reads the native
/// guest rows (a guest write is always observed); only the swizzle/gather +
/// allocation — and, via the returned generation identity, the engine's
/// content hash + upload — are skipped when the bytes are unchanged. Returns
/// the upload byte format (native BGRA8 when eligible, else RGBA8). Measured
/// on Safari fast-scroll: the padded-stride glyph/tile atlases re-present only
/// ~59 distinct gva keys with ~99% recurrence (`fallback_gva_churn`), so this
/// memo now serves that former `lin_guest_fb` hot path instead of a per-bind
/// re-read+re-upload. Returns `None` (no logging: a fast-path miss, not a
/// failure) only for sub-tight strides or formats `convert_row_to_rgba8`
/// cannot decode, which fall through to the general loader.
/// Convert the raw native rows read for a guest-linear texture (row stride
/// `bpr`, `tight` = the packed row byte count) into the tight upload buffer.
/// A 4-byte straight upload — RGBA8, or BGRA8 kept native — gathers each row
/// with a plain copy (padding skipped, no swizzle) and reports its native
/// format; every other format converts to RGBA8 per row. Shared by the
/// guest-linear memo's miss-fill so its padded and tight branches agree
/// byte-for-byte with the direct loader.
fn native_scratch_to_upload(
    scratch: &[u8],
    w: u32,
    h: u32,
    bpr: u64,
    sample_fmt: u16,
    tight: u64,
) -> Option<(Vec<u8>, TexelLayout)> {
    let out_row = (w as usize).checked_mul(RGBA8_BPP as usize)?;
    let out_len = out_row.checked_mul(h as usize)?;
    let bpr = bpr as usize;
    if let Some(fmt) = linear_native_upload_format(sample_fmt, true)
        .filter(|_| tight == (w as u64).saturating_mul(RGBA8_BPP as u64))
    {
        let row_bytes = tight as usize;
        let mut out = vec![0u8; out_len];
        for y in 0..h as usize {
            let src = y.checked_mul(bpr)?;
            let dst = y * row_bytes;
            out.get_mut(dst..dst + row_bytes)?
                .copy_from_slice(scratch.get(src..src + row_bytes)?);
        }
        return Some((out, fmt));
    }
    let trow = tight as usize;
    let mut out = vec![0u8; out_len];
    for y in 0..h as usize {
        let src = y.checked_mul(bpr)?;
        if !pixel_format::convert_row_to_rgba8(
            sample_fmt,
            scratch.get(src..src + trow)?,
            w,
            &mut out[y * out_row..],
        ) {
            return None;
        }
    }
    Some((out, TexelLayout::Rgba8))
}

#[allow(clippy::too_many_arguments)]
fn load_linear_guest_memoized<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    texture_ref: u32,
    object_type: u8,
    tex: &TextureDescriptor,
    gva: u64,
    w: u32,
    h: u32,
) -> Option<(
    std::sync::Arc<Vec<u8>>,
    Option<LinearSampleIdentity>,
    TexelLayout,
)> {
    if !tex.has_pixel_format {
        return None;
    }
    let sample_fmt = effective_view_sample_format(tex.pixel_format, None)?;
    let (_, layout) = tex.level_gva(0, state.page_shift)?;
    let bpr = layout.row_stride;
    let tight = pixel_format::tight_row_bytes(w, tex.pixel_format)? as u64;
    // Padded strides ride the same memo now — the native read below covers the
    // full `bpr*h` span (padding included, so a write anywhere is observed) and
    // `native_scratch_to_upload` gathers the tight rows. Only a sub-tight stride
    // (impossible geometry) or a zero dimension declines to the fallback.
    if bpr < tight || w == 0 || h == 0 {
        return None;
    }
    let span = bpr.checked_mul(h as u64)?;
    let native_len = host_alloc_len(span)?;
    if tex.allocation_size != 0 && layout.offset.saturating_add(span) > tex.allocation_size {
        return None;
    }
    // Same coherence rule as the general loader: land any resident-
    // authoritative writeback aliasing the sampled span before reading it.
    let t_flush = std::time::Instant::now();
    crate::runtime::storage_flush::flush_intersecting_task_gva(state, host, task_id, gva, span);
    let flush_us = t_flush.elapsed().as_micros() as u64;
    let mut scratch = std::mem::take(&mut state.guest_linear_scratch);
    scratch.resize(native_len, 0);
    let t_read = std::time::Instant::now();
    let read = gva_mem::read_task_gva_by_id(
        host,
        &state.tasks,
        task_id,
        gva,
        &mut scratch,
        state.page_shift,
    );
    let read_us = t_read.elapsed().as_micros() as u64;
    if read.is_err() {
        crate::runtime::census::setup_tex_census::note_lin_memo(flush_us, read_us, 0);
        state.guest_linear_scratch = scratch;
        return None;
    }
    let key = (task_id, gva, w, h, sample_fmt);
    let t_cmp = std::time::Instant::now();
    let hit = state
        .guest_linear_memo
        .get_touch(&key)
        // Vec equality is length + byte memcmp with early exit on change.
        .filter(|m| m.native == scratch)
        .map(|m| (m.rgba.clone(), m.generation, m.bgra8));
    crate::runtime::census::setup_tex_census::note_lin_memo(
        flush_us,
        read_us,
        t_cmp.elapsed().as_micros() as u64,
    );
    if let Some((rgba, generation, bgra8)) = hit {
        let fmt = if bgra8 {
            TexelLayout::Bgra8
        } else {
            TexelLayout::Rgba8
        };
        state.guest_linear_scratch = scratch;
        sampled_census::note(sampled_census::Branch::LinMemo, 0);
        return Some((
            rgba,
            Some(LinearSampleIdentity {
                key: gva,
                generation,
            }),
            fmt,
        ));
    }
    // First sight or native bytes changed: convert fresh, new generation.
    let Some((rgba, fmt)) = native_scratch_to_upload(&scratch, w, h, bpr, sample_fmt, tight) else {
        state.guest_linear_scratch = scratch;
        return None;
    };
    let rgba_len = rgba.len();
    state.guest_linear_gen += 1;
    let generation = GUEST_LINEAR_GEN_BASE + state.guest_linear_gen;
    let rgba = std::sync::Arc::new(rgba);
    log_linear_sample_src(
        task_id,
        texture_ref,
        object_type,
        gva,
        sample_fmt,
        bpr,
        w,
        h,
        "guest_memo_fill",
        0,
        &rgba,
    );
    let entry_bytes = scratch.len() + rgba.len();
    state.guest_linear_memo.insert(
        key,
        crate::model::GuestLinearMemo {
            native: scratch,
            rgba: rgba.clone(),
            bgra8: fmt == TexelLayout::Bgra8,
            generation,
        },
        entry_bytes,
    );
    sampled_census::note(sampled_census::Branch::LinGuest, rgba_len);
    Some((
        rgba,
        Some(LinearSampleIdentity {
            key: gva,
            generation,
        }),
        fmt,
    ))
}

fn load_linear_from_host_caches<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    texture_ref: u32,
    entry: &ListObjectEntry,
    tex: &TextureDescriptor,
) -> Option<LoadedLinearSample> {
    // The object-list entry + descriptor are resolved+decoded once by the caller
    // (`resolve_sampled_source`'s linear branch) and threaded in; the zero-copy
    // attempt above shares the same decode.
    let (gva, layout) = tex.level_gva(0, state.page_shift)?;
    let w = layout.width;
    let h = layout.height;
    if w == 0 || h == 0 {
        return None;
    }
    // A deferred GVA render Store at this base is the authoritative content —
    // land it first so the cache lookup below serves fresh bytes instead of a
    // stale (pre-Store) encode.
    if state.gva_deferred_flush.contains_key(&gva) {
        crate::runtime::storage_flush::flush_gva_exact(state, host, gva, true, "gva_sample");
    }
    if let Some((bgra, host_gen, producer_task, producer_ref, producer_type)) =
        crate::runtime::surface_cache::get_gva_with_owner(state, gva, w, h)
    {
        if gva_cache_owner_allows_object_type(producer_type, entry.object_type) {
            let identity = Some(LinearSampleIdentity {
                key: gva,
                generation: host_gen as u64,
            });
            // Memo fast path: same authoritative entry (gva/gen/geom) means the
            // swizzled copy is already made - reuse the Arc, skip copy+swizzle.
            if let Some(rgba) =
                linear_sampled_memo_reuse(state, task_id, texture_ref, gva, host_gen, w, h)
            {
                sampled_census::note(sampled_census::Branch::GvaMemo, 0);
                return Some((w, h, rgba, identity, TexelLayout::Rgba8));
            }
            let rgba = swap_rb_channels(bgra);
            log_linear_sample_src(
                task_id,
                texture_ref,
                entry.object_type,
                gva,
                tex.pixel_format,
                layout.row_stride,
                w,
                h,
                "gva_cache",
                host_gen,
                &rgba,
            );
            probe_linear_cache_guest(
                state,
                host,
                task_id,
                texture_ref,
                entry.object_type,
                gva,
                tex.pixel_format,
                layout.row_stride,
                w,
                h,
                host_gen,
                &rgba,
            );
            sampled_census::note(sampled_census::Branch::GvaCopy, rgba.len());
            let rgba = std::sync::Arc::new(rgba);
            let entry_bytes = rgba.len();
            state.linear_sampled_memo.insert(
                (task_id, texture_ref),
                crate::model::LinearSampledMemo {
                    gva,
                    host_gen,
                    width: w,
                    height: h,
                    rgba: rgba.clone(),
                },
                entry_bytes,
            );
            return Some((w, h, rgba, identity, TexelLayout::Rgba8));
        }
        log_linear_cache_authority_transition(
            task_id,
            texture_ref,
            entry.object_type,
            gva,
            w,
            h,
            host_gen,
            producer_task,
            producer_ref,
            producer_type,
        );
    }
    if let Some((bgra, host_gen)) =
        crate::runtime::surface_cache::get_texture_with_gen(state, texture_ref, w, h)
    {
        let rgba = swap_rb_channels(bgra);
        log_linear_sample_src(
            task_id,
            texture_ref,
            entry.object_type,
            gva,
            tex.pixel_format,
            layout.row_stride,
            w,
            h,
            "ref_cache",
            host_gen,
            &rgba,
        );
        sampled_census::note(sampled_census::Branch::RefCache, rgba.len());
        return Some((w, h, std::sync::Arc::new(rgba), None, TexelLayout::Rgba8));
    }
    // Guest-CPU-produced linear textures (wallpaper, glyph atlases) have no
    // host producer generation. Re-read the native rows and byte-compare
    // against the memo: unchanged content reuses the retained swizzled Arc
    // and carries a generation identity so the engine skips hash+memcmp too.
    if let Some((rgba, identity, byte_format)) = load_linear_guest_memoized(
        state,
        host,
        task_id,
        texture_ref,
        entry.object_type,
        tex,
        gva,
        w,
        h,
    ) {
        return Some((w, h, rgba, identity, byte_format));
    }
    let Some((rgba, byte_format)) =
        load_linear_texture_native_host(state, host, task_id, texture_ref, 0, None)
    else {
        if w >= 1280 && h >= 720 {
            crate::observe::fail(format!(
                "linear_sample_miss reason=guest_load task={task_id} ref={texture_ref} type={} gva={gva:#x} fmt={:#x} {w}x{h} bpr={}",
                entry.object_type, tex.pixel_format, layout.row_stride
            ));
        }
        return None;
    };
    let need = (w as usize).saturating_mul(h as usize).saturating_mul(4);
    if rgba.len() >= need {
        log_linear_sample_src(
            task_id,
            texture_ref,
            entry.object_type,
            gva,
            tex.pixel_format,
            layout.row_stride,
            w,
            h,
            "guest",
            0,
            &rgba[..need],
        );
        sampled_census::note(sampled_census::Branch::LinGuestFallback, need);
        // Measure-only: does this padded-fallback texture's authoritative gva
        // recur across binds? That is the ceiling on any gva-keyed padded memo's
        // hit rate — high repeat_pct means a memo (skip alloc + engine hash) is
        // worth building; a fresh-gva-dominated log means Safari rotates glyph
        // backing and a memo can never hit. Cheap (no content hash), bounded LRU.
        crate::runtime::census::sampled_gva_churn::note(task_id, gva, w, h);
        // `load_linear_texture_native_host` already returns a tight `need`-byte
        // buffer (RGBA8, or native BGRA8 when `byte_format == Bgra8`), so
        // `rgba.len() == need` in the common case — move it straight into the
        // Arc instead of a redundant `rgba[..need].to_vec()` copy. This is the
        // CONFIRMED Safari-scroll hot path (census `lin_guest_fb`), and BGRA8
        // sources now upload native (no CPU channel swap) — the engine binds a
        // BGRA8 image and the sampler swizzles in hardware. The slice+copy is
        // kept only for the defensive `len > need` case (padding overshoot).
        let arc = if rgba.len() == need {
            std::sync::Arc::new(rgba)
        } else {
            std::sync::Arc::new(rgba[..need].to_vec())
        };
        return Some((w, h, arc, None, byte_format));
    }
    if w >= 1280 && h >= 720 {
        crate::observe::fail(format!(
            "linear_sample_miss reason=short_rgba task={task_id} ref={texture_ref} type={} gva={gva:#x} fmt={:#x} {w}x{h} bpr={} got={} need={need}",
            entry.object_type,
            tex.pixel_format,
            layout.row_stride,
            rgba.len()
        ));
    }
    None
}

#[inline]
fn gva_cache_linear_texture_type(object_type: u8) -> bool {
    matches!(
        object_type,
        OBJECT_TYPE_TEXTURE | OBJECT_TYPE_TEXTURE_VARIANT
    )
}

/// A GVA cache is keyed by decoded linear texture storage. Type-2 and type-3
/// wrappers may alias the same GVA allocation, so a matching GVA+geometry cache
/// entry can serve either tag. Other nonzero object-type transitions remain
/// separate resource classes and fall through to current ref/guest backing.
#[inline]
fn gva_cache_owner_allows_object_type(producer_type: u8, current_type: u8) -> bool {
    producer_type == 0
        || current_type == 0
        || producer_type == current_type
        || (gva_cache_linear_texture_type(producer_type)
            && gva_cache_linear_texture_type(current_type))
}

#[allow(clippy::too_many_arguments)]
fn log_linear_cache_authority_transition(
    task_id: u32,
    texture_ref: u32,
    object_type: u8,
    gva: u64,
    width: u32,
    height: u32,
    host_gen: u32,
    producer_task: u32,
    producer_ref: u32,
    producer_type: u8,
) {
    crate::observe::off(format!(
        "linear_cache_authority reason=object_type_transition action=skip_gva_cache task={task_id} ref={texture_ref} type={object_type} gva={gva:#x} {width}x{height} host_gen={host_gen} producer_task={producer_task} producer_ref={producer_ref} producer_type={producer_type}"
    ));
}

/// Measure-only provenance for display-sized type-2/3 compositor inputs.
///
/// Content counters diagnose producer loss; they never select or gate product
/// behavior. Object type + descriptor identity avoid conclusions from recycled
/// texture refs alone.
#[allow(clippy::too_many_arguments)]
fn log_linear_sample_src(
    task_id: u32,
    texture_ref: u32,
    object_type: u8,
    gva: u64,
    pixel_format: u16,
    row_stride: u64,
    width: u32,
    height: u32,
    source: &str,
    host_gen: u32,
    rgba: &[u8],
) {
    let zero_rgb_alpha = rgba
        .chunks_exact(4)
        .filter(|p| p[0] | p[1] | p[2] == 0 && p[3] != 0)
        .count();
    if zero_rgb_alpha != 0 {
        let opaque_black = rgba
            .chunks_exact(4)
            .filter(|p| p[0] | p[1] | p[2] == 0 && p[3] == u8::MAX)
            .count();
        // One line per distinct sampled content identity. Small A8 glyph/mask
        // textures are expected to be sampled many times; repeating their
        // census on every draw would hide the useful signal in log volume.
        use std::collections::HashSet;
        use std::sync::Mutex;
        type ZeroRgbAlphaKey = (u32, u32, u64, u16, u32, u32, u32, usize, usize);
        static SEEN: Mutex<Option<HashSet<ZeroRgbAlphaKey>>> = Mutex::new(None);
        let first = {
            let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
            seen.get_or_insert_with(HashSet::new).insert((
                task_id,
                texture_ref,
                gva,
                pixel_format,
                width,
                height,
                host_gen,
                zero_rgb_alpha,
                opaque_black,
            ))
        };
        if first {
            crate::observe::off(format!(
                "sample_alpha_mask reason=zero_rgb_alpha_preserved src={source} task={task_id} ref={texture_ref} type={object_type} gva={gva:#x} fmt={pixel_format:#x} {width}x{height} bpr={row_stride} host_gen={host_gen} zero_rgb_alpha={zero_rgb_alpha} opaque_black={opaque_black}"
            ));
        }
    }
    if width < 1280 || height < 720 {
        return;
    }
    let (rgb_nz, max_rgb, px0) = crate::observe::rgba_rgb_stats(rgba);
    crate::observe::off(format!(
        "linear_sample src={source} task={task_id} ref={texture_ref} type={object_type} gva={gva:#x} fmt={pixel_format:#x} {width}x{height} bpr={row_stride} host_gen={host_gen} rgb_nz={rgb_nz} max_rgb={max_rgb} px0=[{},{},{},{}]",
        px0[0], px0[1], px0[2], px0[3]
    ));
}

/// Measurement-only comparison between a retained GVA encode and the bytes
/// currently visible through the guest page table for the same texture.
///
/// This must never select the sampled source. It names the stale-vs-retained
/// ambiguity without turning pixel content into device policy.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct LinearCacheGuestComparison {
    differing_texels: usize,
    cache_rgb_nz: usize,
    guest_rgb_nz: usize,
    cache_alpha0: usize,
    guest_alpha0: usize,
}

fn compare_linear_cache_guest(
    cache_rgba: &[u8],
    guest_rgba: &[u8],
) -> Option<LinearCacheGuestComparison> {
    if cache_rgba.len() != guest_rgba.len() || !cache_rgba.len().is_multiple_of(4) {
        return None;
    }
    let mut out = LinearCacheGuestComparison::default();
    for (cache, guest) in cache_rgba.chunks_exact(4).zip(guest_rgba.chunks_exact(4)) {
        out.differing_texels += usize::from(cache != guest);
        out.cache_rgb_nz += usize::from(cache[0] | cache[1] | cache[2] != 0);
        out.guest_rgb_nz += usize::from(guest[0] | guest[1] | guest[2] != 0);
        out.cache_alpha0 += usize::from(cache[3] == 0);
        out.guest_alpha0 += usize::from(guest[3] == 0);
    }
    Some(out)
}

#[allow(clippy::too_many_arguments)]
fn log_linear_cache_guest_comparison(
    task_id: u32,
    texture_ref: u32,
    object_type: u8,
    gva: u64,
    pixel_format: u16,
    row_stride: u64,
    width: u32,
    height: u32,
    host_gen: u32,
    stats: LinearCacheGuestComparison,
) {
    crate::observe::off(format!(
        "linear_cache_guest_probe status=ok task={task_id} ref={texture_ref} type={object_type} gva={gva:#x} fmt={pixel_format:#x} {width}x{height} bpr={row_stride} host_gen={host_gen} same={} diff_px={} cache_rgb={} guest_rgb={} cache_a0={} guest_a0={}",
        (stats.differing_texels == 0) as u8,
        stats.differing_texels,
        stats.cache_rgb_nz,
        stats.guest_rgb_nz,
        stats.cache_alpha0,
        stats.guest_alpha0
    ));
}

/// Probe each display-sized retained cache identity once. A full GVA read is
/// deduplicated before I/O so repeated compositor samples do not become
/// repeated multi-megabyte page-table walks.
#[allow(clippy::too_many_arguments)]
fn probe_linear_cache_guest<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    texture_ref: u32,
    object_type: u8,
    gva: u64,
    pixel_format: u16,
    row_stride: u64,
    width: u32,
    height: u32,
    host_gen: u32,
    cache_rgba: &[u8],
) {
    if width < 1280 || height < 720 {
        return;
    }
    use std::collections::HashSet;
    use std::sync::Mutex;
    type CacheGuestDeltaKey = (u32, u32, u64, u32, u32, u32);
    static SEEN: Mutex<Option<HashSet<CacheGuestDeltaKey>>> = Mutex::new(None);
    let first = {
        let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
        seen.get_or_insert_with(HashSet::new).insert((
            task_id,
            texture_ref,
            gva,
            width,
            height,
            host_gen,
        ))
    };
    if !first {
        return;
    }

    let Some(guest_rgba) =
        load_linear_texture_rgba_host(state, host, task_id, texture_ref, 0, None)
    else {
        crate::observe::off(format!(
            "linear_cache_guest_probe status=unavailable task={task_id} ref={texture_ref} type={object_type} gva={gva:#x} fmt={pixel_format:#x} {width}x{height} bpr={row_stride} host_gen={host_gen}"
        ));
        return;
    };
    let need = (width as usize)
        .saturating_mul(height as usize)
        .saturating_mul(4);
    let Some(stats) = guest_rgba
        .get(..need)
        .and_then(|guest| compare_linear_cache_guest(cache_rgba, guest))
    else {
        crate::observe::off(format!(
            "linear_cache_guest_probe status=short task={task_id} ref={texture_ref} type={object_type} gva={gva:#x} fmt={pixel_format:#x} {width}x{height} bpr={row_stride} host_gen={host_gen} cache_len={} guest_len={} need={need}",
            cache_rgba.len(), guest_rgba.len()
        ));
        return;
    };
    log_linear_cache_guest_comparison(
        task_id,
        texture_ref,
        object_type,
        gva,
        pixel_format,
        row_stride,
        width,
        height,
        host_gen,
        stats,
    );
}

/// Store type-2/3 encode into texture_ref + GVA host caches (BGRA).
#[allow(
    clippy::too_many_arguments,
    reason = "the cache identity mirrors the task, object, GVA, and texture geometry"
)]
pub(crate) fn host_cache_store_gva_layer(
    state: &mut DeviceState,
    task_id: u32,
    texture_ref: u32,
    object_type: u8,
    gva: u64,
    width: u32,
    height: u32,
    rgba: &[u8],
) {
    if width == 0 || height == 0 {
        return;
    }
    let need = (width as usize)
        .saturating_mul(height as usize)
        .saturating_mul(4);
    if rgba.len() < need {
        return;
    }
    let bgra = swap_rb_channels(&rgba[..need]);
    if texture_ref != 0 {
        crate::runtime::surface_cache::store_texture(
            state,
            texture_ref,
            width,
            height,
            bgra.clone(),
        );
    }
    if gva != 0 {
        crate::runtime::surface_cache::store_gva_owned(
            state,
            gva,
            width,
            height,
            bgra,
            task_id,
            texture_ref,
            object_type,
        );
    }
}

/// Store encode RGBA8 into **texture_ref** host cache as BGRA (not surface_id).
#[cfg(test)]
fn host_cache_store_rgba8(
    state: &mut DeviceState,
    texture_ref: u32,
    width: u32,
    height: u32,
    rgba: &[u8],
) {
    if texture_ref == 0 || width == 0 || height == 0 {
        return;
    }
    let need = (width as usize)
        .saturating_mul(height as usize)
        .saturating_mul(4);
    if rgba.len() < need {
        return;
    }
    let bgra = swap_rb_channels(&rgba[..need]);
    crate::runtime::surface_cache::store_texture(state, texture_ref, width, height, bgra);
}

/// Wall-clock µs since `t0` (saturate on overflow; research host clocks only).
#[inline]
fn elapsed_us(t0: std::time::Instant) -> u64 {
    t0.elapsed().as_micros().min(u64::MAX as u128) as u64
}

/// Always-on stage timing for the Linux metal2vulkan → engine draw path.
/// Grep `/tmp/reims-vgpu-fail.log` for `linux_m2v_timing`.
///
/// | Field | Meaning |
/// | --- | --- |
/// | `total_us` | wall clock for this draw attempt |
/// | `load_us` | pipeline + MTLB load + AIR extract |
/// | `m2v_us` | metal2vulkan translate (both stages; cache hit ≈ µs) |
/// | `setup_us` | binds, seed, SPIR-V reloc, build engine request |
/// | `setup_spv_us` | SPIR-V byte→word materialization (both stages) |
/// | `setup_bufs_us` | stream buffer loads + stage-in attrs + fragment reloc |
/// | `setup_tex_us` | sampled texture loads/binds + sampler descriptors |
/// | `setup_seed_us` | color0 Load/Clear seed byte resolution |
/// | `setup_asm_us` | DrawRequest assembly incl. present-boundary seed + coverage eval |
/// | `engine_us` | complete internal `vk_engine` execute |
/// | `composite_us` | post-engine CPU path (D3: residual; no content composites) |
/// | `context_us` / `pool_init_us` | Vulkan context and persistent-pool initialization |
/// | `cache_us` | shader/layout/pass/pipeline cache resolution |
/// | `*_create_us` | cold Vulkan object creation nested within cache/resource time |
/// | `resource_us` | samplers, staging/target resources, and descriptors |
/// | `*_prepare_us` | non-overlapping resource-preparation subphases |
/// | `memory_alloc_us` | Vulkan allocation calls nested in preparation subphases |
/// | `resource_unattributed_us` | residual after non-overlapping preparation subphases |
/// | `pre_record_wait_us` | prior fence wait + fence reset before recording |
/// | `record_us` / `submit_us` / `wait_us` / `readback_us` | engine stage splits (D1) |
/// | `engine_unattributed_us` | residual after the non-overlapping engine phases above |
/// Non-overlapping sub-spans of `setup_us` (see the table above).
#[cfg(feature = "backend-vulkan")]
#[derive(Clone, Copy, Debug, Default)]
struct SetupSplitUs {
    spv: u64,
    bufs: u64,
    tex: u64,
    seed: u64,
    assemble: u64,
    /// Assemble sub-stages: request prep (index/viewport/attr move), Store
    /// decision + present-boundary Load seed, CPU coverage eval, and the
    /// per-draw diag/census construction (content scans + resources line).
    asm_prep: u64,
    asm_load: u64,
    asm_cov: u64,
    asm_diag: u64,
}

#[cfg(feature = "backend-vulkan")]
#[derive(Clone, Copy, Debug, Default)]
struct M2vTiming {
    pipe: u32,
    w: u32,
    h: u32,
    total_us: u64,
    load_us: u64,
    m2v_us: u64,
    setup_us: u64,
    setup_split: SetupSplitUs,
    engine_us: u64,
    composite_us: u64,
}

#[cfg(feature = "backend-vulkan")]
fn log_linux_m2v_timing(
    timing: M2vTiming,
    counters: crate::backend::vulkan::engine::CounterSnapshot,
) -> crate::model::TrancheStats {
    let M2vTiming {
        pipe,
        w,
        h,
        total_us,
        load_us,
        m2v_us,
        setup_us,
        setup_split,
        engine_us,
        composite_us,
    } = timing;
    let crate::backend::vulkan::engine::CounterSnapshot {
        creates,
        allocs,
        target_free_hits: target_reuse,
        pipeline_hits: pipe_hit,
        pipeline_misses: pipe_miss,
        context_us,
        pool_init_us,
        cache_us,
        shader_create_us,
        layout_create_us,
        pass_create_us,
        pipeline_create_us,
        sampler_create_us,
        resource_us,
        sampler_prepare_us,
        vertex_prepare_us,
        index_prepare_us,
        storage_prepare_us,
        seed_prepare_us,
        target_prepare_us,
        sampled_prepare_us,
        readback_prepare_us,
        descriptor_prepare_us,
        memory_alloc_us,
        pre_record_wait_us,
        record_us,
        submit_us,
        wait_us,
        retire_wait_us,
        render_post_wait_skips: post_wait_skips,
        ring_retire_blocks,
        readback_us,
        readbacks,
        seed_uploads,
        sampled_reuploads,
        sampled_reupload_bytes,
        sampled_cache_hits,
        sampled_cache_hit_bytes,
        sampled_cache_misses,
        sampled_gpu_binds,
        ..
    } = counters;
    let resource_unattributed_us = engine_unattributed_us(
        resource_us,
        &[
            sampler_prepare_us,
            vertex_prepare_us,
            index_prepare_us,
            storage_prepare_us,
            seed_prepare_us,
            target_prepare_us,
            sampled_prepare_us,
            readback_prepare_us,
            descriptor_prepare_us,
        ],
    );
    let engine_unattributed_us = engine_unattributed_us(
        engine_us,
        &[
            context_us,
            pool_init_us,
            cache_us,
            resource_us,
            pre_record_wait_us,
            record_us,
            submit_us,
            wait_us,
            retire_wait_us,
            readback_us,
        ],
    );
    let SetupSplitUs {
        spv: setup_spv_us,
        bufs: setup_bufs_us,
        tex: setup_tex_us,
        seed: setup_seed_us,
        assemble: setup_asm_us,
        asm_prep: asm_prep_us,
        asm_load: asm_load_us,
        asm_cov: asm_cov_us,
        asm_diag: asm_diag_us,
    } = setup_split;
    // Per-draw perf telemetry: verbose-gated so it does not flood the always-on
    // fail log (~0.5M lines / boot) or build this ~1 KB string on a normal boot.
    // Set `REIMS_VGPU_DRAW_LOG=1` to collect it in /tmp/reims-vgpu-draw.log for a timing census.
    if crate::observe::draw_log_enabled() {
        crate::observe::line(format!(
            "linux_m2v_timing pipe={pipe} {w}x{h} total_us={total_us} load_us={load_us} m2v_us={m2v_us} setup_us={setup_us} setup_spv_us={setup_spv_us} setup_bufs_us={setup_bufs_us} setup_tex_us={setup_tex_us} setup_seed_us={setup_seed_us} setup_asm_us={setup_asm_us} asm_prep_us={asm_prep_us} asm_load_us={asm_load_us} asm_cov_us={asm_cov_us} asm_diag_us={asm_diag_us} engine_us={engine_us} engine_unattributed_us={engine_unattributed_us} composite_us={composite_us} vk_engine_creates={creates} vk_engine_allocs={allocs} pipe_hit={pipe_hit} pipe_miss={pipe_miss} context_us={context_us} pool_init_us={pool_init_us} cache_us={cache_us} shader_create_us={shader_create_us} layout_create_us={layout_create_us} pass_create_us={pass_create_us} pipeline_create_us={pipeline_create_us} sampler_create_us={sampler_create_us} resource_us={resource_us} resource_unattributed_us={resource_unattributed_us} sampler_prepare_us={sampler_prepare_us} vertex_prepare_us={vertex_prepare_us} index_prepare_us={index_prepare_us} storage_prepare_us={storage_prepare_us} seed_prepare_us={seed_prepare_us} target_prepare_us={target_prepare_us} sampled_prepare_us={sampled_prepare_us} readback_prepare_us={readback_prepare_us} descriptor_prepare_us={descriptor_prepare_us} memory_alloc_us={memory_alloc_us} pre_record_wait_us={pre_record_wait_us} record_us={record_us} submit_us={submit_us} wait_us={wait_us} retire_wait_us={retire_wait_us} post_wait_skips={post_wait_skips} ring_retire_blocks={ring_retire_blocks} readback_us={readback_us} vk_engine_readbacks={readbacks} seed_uploads={seed_uploads} sampled_reuploads={sampled_reuploads} sampled_reupload_bytes={sampled_reupload_bytes} sampled_cache_hits={sampled_cache_hits} sampled_cache_hit_bytes={sampled_cache_hit_bytes} sampled_cache_misses={sampled_cache_misses} sampled_gpu_binds={sampled_gpu_binds}"
        ));
    }
    // Per-tranche attribution: this draw's contribution to the drain lock hold.
    crate::model::TrancheStats {
        draws: 1,
        draw_total_us: total_us,
        load_us,
        m2v_us,
        setup_us,
        setup_bufs_us,
        setup_tex_us,
        setup_seed_us,
        setup_asm_us,
        engine_us,
        engine_resource_us: resource_us,
        engine_descriptor_us: descriptor_prepare_us,
        engine_record_us: record_us,
        engine_submit_us: submit_us,
        engine_target_us: target_prepare_us,
        engine_sampled_us: sampled_prepare_us,
        engine_bufprep_us: sampler_prepare_us
            .saturating_add(vertex_prepare_us)
            .saturating_add(index_prepare_us)
            .saturating_add(storage_prepare_us)
            .saturating_add(seed_prepare_us),
        engine_creates: creates,
        engine_allocs: allocs,
        target_reuse,
        engine_memory_alloc_us: memory_alloc_us,
        wait_us,
        retire_wait_us,
        readback_us,
        readbacks,
        reuploads: sampled_reuploads,
        reupload_bytes: sampled_reupload_bytes,
        // Non-draw classes (compute, store/flush) are noted at their own sites.
        ..Default::default()
    }
}

fn engine_unattributed_us(engine_us: u64, phases: &[u64]) -> u64 {
    phases.iter().fold(engine_us, |remaining, phase| {
        remaining.saturating_sub(*phase)
    })
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct FullTargetCoverage {
    bounds_span: bool,
    triangles_cover: bool,
}

impl FullTargetCoverage {
    fn full(self) -> bool {
        self.bounds_span && self.triangles_cover
    }
}

/// A fail-closed shader-pulled coverage proof.
///
/// Evaluator failures delegate their exact registered reason; the remaining
/// variants name the command/geometry checks around evaluation.
#[derive(Clone, Debug, PartialEq, Eq)]
enum ShaderPulledCoverageDecline {
    ZeroTarget,
    PartialViewportOrScissor,
    IndexStreamInvalid,
    TooFewIndices { count: usize },
    VertexEval(crate::runtime::spirv_vertex_eval::VertexEvalDecline),
    PositionWDegenerate { vertex_index: u32 },
    PositionNotFinite { vertex_index: u32 },
    TriangleGap,
    PartialBounds,
}

impl Decline for ShaderPulledCoverageDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::ZeroTarget => "shader_pulled_coverage_zero_target",
            Self::PartialViewportOrScissor => "shader_pulled_coverage_partial_viewport_or_scissor",
            Self::IndexStreamInvalid => "shader_pulled_coverage_index_stream_invalid",
            Self::TooFewIndices { .. } => "shader_pulled_coverage_too_few_indices",
            Self::VertexEval(reason) => reason.slug(),
            Self::PositionWDegenerate { .. } => "shader_pulled_coverage_position_w_degenerate",
            Self::PositionNotFinite { .. } => "shader_pulled_coverage_position_not_finite",
            Self::TriangleGap => "shader_pulled_coverage_triangle_gap",
            Self::PartialBounds => "shader_pulled_coverage_partial_bounds",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::TooFewIndices { count } => vec![("count", count.to_string())],
            Self::VertexEval(reason) => reason.fields(),
            Self::PositionWDegenerate { vertex_index }
            | Self::PositionNotFinite { vertex_index } => {
                vec![("vertex_index", vertex_index.to_string())]
            }
            _ => Vec::new(),
        }
    }
}

impl From<crate::runtime::spirv_vertex_eval::VertexEvalDecline> for ShaderPulledCoverageDecline {
    fn from(reason: crate::runtime::spirv_vertex_eval::VertexEvalDecline) -> Self {
        Self::VertexEval(reason)
    }
}

/// Whether decoded triangle geometry covers the complete color target.
///
/// This is resource/command geometry only: viewport, scissor, topology, index
/// stream, and location-0 Float position data. It never inspects rendered
/// pixels. A global vertex bounding box is not sufficient: two disjoint quads
/// can touch opposite target edges while leaving a large uncovered rectangle.
#[cfg(feature = "backend-vulkan")]
fn draw_full_target_coverage(
    resources: &crate::backend::vulkan::engine::DrawRequest,
    width: u32,
    height: u32,
    vertex_count: u32,
) -> FullTargetCoverage {
    let mut coverage = FullTargetCoverage::default();
    if width == 0 || height == 0 {
        return coverage;
    }
    if !full_viewport_scissor(resources, width, height) {
        return coverage;
    }

    let Some(position) = resources.vertex_attributes.iter().find(|a| {
        a.location == 0
            && a.step_function == crate::backend::vulkan::engine::VertexStepFunction::PerVertex
            && matches!(
                a.format,
                crate::backend::vulkan::engine::VertexAttributeFormat::Float2
                    | crate::backend::vulkan::engine::VertexAttributeFormat::Float3
                    | crate::backend::vulkan::engine::VertexAttributeFormat::Float4
            )
    }) else {
        return coverage;
    };
    if position.stride == 0 {
        return coverage;
    }

    let Some(indices) = decode_coverage_indices(resources, vertex_count) else {
        return coverage;
    };
    if indices.len() < 3 {
        return coverage;
    }

    let mut positions = std::collections::BTreeMap::<u32, [f64; 2]>::new();
    let mut min_x = f32::INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    let position_bytes = position.content.cpu_bytes();
    for &index in &indices {
        let Some(off) = (index as usize)
            .checked_mul(position.stride as usize)
            .and_then(|v| v.checked_add(position.offset as usize))
        else {
            return coverage;
        };
        let Some(raw) = position_bytes.get(off..off.saturating_add(8)) else {
            return coverage;
        };
        let x = f32::from_le_bytes(raw[0..4].try_into().unwrap());
        let y = f32::from_le_bytes(raw[4..8].try_into().unwrap());
        if !x.is_finite() || !y.is_finite() {
            return coverage;
        }
        positions.insert(index, [f64::from(x), f64::from(y)]);
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
    }

    let pixel_space =
        min_x <= 0.0 && min_y <= 0.0 && max_x >= width as f32 && max_y >= height as f32;
    let ndc_space = min_x <= -1.0 && min_y <= -1.0 && max_x >= 1.0 && max_y >= 1.0;
    coverage.bounds_span = pixel_space || ndc_space;
    if !coverage.bounds_span {
        return coverage;
    }

    if ndc_space && !pixel_space {
        for position in positions.values_mut() {
            position[0] = (position[0] + 1.0) * f64::from(width) * 0.5;
            position[1] = (position[1] + 1.0) * f64::from(height) * 0.5;
        }
    }

    coverage.triangles_cover = triangle_union_covers(
        &positions,
        &indices,
        resources.primitive_topology,
        width,
        height,
    );
    coverage
}

/// Decoded location-0 position AABB over the drawn index set, plus index
/// count. Raw vertex-space values (pixel or NDC — per-pipe contract); no
/// coverage judgment, no pixel inspection. `None` when there is no per-vertex
/// Float2/3/4 location-0 attribute or the index/position decode fails.
#[cfg(feature = "backend-vulkan")]
fn draw_position_bounds(
    resources: &crate::backend::vulkan::engine::DrawRequest,
    vertex_count: u32,
) -> Option<([f32; 4], usize)> {
    let position = resources.vertex_attributes.iter().find(|a| {
        a.location == 0
            && a.step_function == crate::backend::vulkan::engine::VertexStepFunction::PerVertex
            && matches!(
                a.format,
                crate::backend::vulkan::engine::VertexAttributeFormat::Float2
                    | crate::backend::vulkan::engine::VertexAttributeFormat::Float3
                    | crate::backend::vulkan::engine::VertexAttributeFormat::Float4
            )
    })?;
    if position.stride == 0 {
        return None;
    }
    let indices = decode_coverage_indices(resources, vertex_count)?;
    if indices.is_empty() {
        return None;
    }
    let mut min_x = f32::INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    let position_bytes = position.content.cpu_bytes();
    for &index in &indices {
        let off = (index as usize)
            .checked_mul(position.stride as usize)?
            .checked_add(position.offset as usize)?;
        let raw = position_bytes.get(off..off.saturating_add(8))?;
        let x = f32::from_le_bytes(raw[0..4].try_into().unwrap());
        let y = f32::from_le_bytes(raw[4..8].try_into().unwrap());
        if !x.is_finite() || !y.is_finite() {
            return None;
        }
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
    }
    Some(([min_x, min_y, max_x, max_y], indices.len()))
}

/// Whether every viewport/scissor covers the complete target.
#[cfg(feature = "backend-vulkan")]
fn full_viewport_scissor(
    resources: &crate::backend::vulkan::engine::DrawRequest,
    width: u32,
    height: u32,
) -> bool {
    if resources.viewports.iter().any(|vp| {
        !vp.x.is_finite()
            || !vp.y.is_finite()
            || !vp.width.is_finite()
            || !vp.height.is_finite()
            || vp.x > 0.0
            || vp.y > 0.0
            || vp.x + vp.width < width as f32
            || vp.y + vp.height < height as f32
    }) {
        return false;
    }
    !resources
        .scissors
        .iter()
        .any(|sc| sc.x != 0 || sc.y != 0 || sc.width < width || sc.height < height)
}

/// Decode the validated vertex-index stream for a coverage proof.
#[cfg(feature = "backend-vulkan")]
fn decode_coverage_indices(
    resources: &crate::backend::vulkan::engine::DrawRequest,
    vertex_count: u32,
) -> Option<Vec<u32>> {
    if let Some(indexed) = resources.indexed.as_ref() {
        let index_size = indexed.index_type.byte_size();
        let need = (indexed.index_count as usize).checked_mul(index_size)?;
        if indexed.indices.len() < need {
            return None;
        }
        let mut indices = Vec::with_capacity(indexed.index_count as usize);
        for i in 0..indexed.index_count as usize {
            let off = i * index_size;
            let raw = match indexed.index_type {
                crate::backend::vulkan::engine::IndexType::U16 => {
                    u16::from_le_bytes([indexed.indices[off], indexed.indices[off + 1]]) as u32
                }
                crate::backend::vulkan::engine::IndexType::U32 => u32::from_le_bytes([
                    indexed.indices[off],
                    indexed.indices[off + 1],
                    indexed.indices[off + 2],
                    indexed.indices[off + 3],
                ]),
            };
            let adjusted = i64::from(raw) + i64::from(indexed.vertex_offset);
            let adjusted = u32::try_from(adjusted).ok()?;
            indices.push(adjusted);
        }
        Some(indices)
    } else {
        Some(
            (resources.first_vertex..resources.first_vertex.saturating_add(vertex_count)).collect(),
        )
    }
}

/// Prove continuous scanline coverage across every target pixel-center row.
/// Linear in target height and triangle count; callers pre-filter with the
/// cheaper bounds test so ordinary partial draws never pay this walk.
#[cfg(feature = "backend-vulkan")]
fn triangle_union_covers(
    positions: &std::collections::BTreeMap<u32, [f64; 2]>,
    indices: &[u32],
    topology: crate::backend::vulkan::engine::PrimitiveTopology,
    width: u32,
    height: u32,
) -> bool {
    use crate::backend::vulkan::engine::PrimitiveTopology;
    let triangles: Vec<[u32; 3]> = match topology {
        PrimitiveTopology::Triangle => indices
            .chunks_exact(3)
            .map(|triangle| [triangle[0], triangle[1], triangle[2]])
            .collect(),
        PrimitiveTopology::TriangleStrip => indices
            .windows(3)
            .enumerate()
            .map(|(i, triangle)| {
                if i & 1 == 0 {
                    [triangle[0], triangle[1], triangle[2]]
                } else {
                    [triangle[1], triangle[0], triangle[2]]
                }
            })
            .collect(),
        PrimitiveTopology::Point | PrimitiveTopology::Line | PrimitiveTopology::LineStrip => {
            return false;
        }
    };
    if triangles.is_empty() {
        return false;
    }

    let target_left = 0.5f64;
    let target_right = f64::from(width) - 0.5;
    const EDGE_EPSILON: f64 = 1.0e-6;
    for row in 0..height {
        let sample_y = f64::from(row) + 0.5;
        let mut intervals = Vec::<[f64; 2]>::new();
        for triangle in &triangles {
            let Some(vertices) = triangle
                .iter()
                .map(|index| positions.get(index).copied())
                .collect::<Option<Vec<_>>>()
            else {
                return false;
            };
            let mut intersections = Vec::with_capacity(6);
            for edge in 0..3 {
                let a = vertices[edge];
                let b = vertices[(edge + 1) % 3];
                let min_y = a[1].min(b[1]);
                let max_y = a[1].max(b[1]);
                if sample_y + EDGE_EPSILON < min_y || sample_y - EDGE_EPSILON > max_y {
                    continue;
                }
                let dy = b[1] - a[1];
                if dy.abs() <= EDGE_EPSILON {
                    intersections.push(a[0]);
                    intersections.push(b[0]);
                } else {
                    let t = (sample_y - a[1]) / dy;
                    if (-EDGE_EPSILON..=1.0 + EDGE_EPSILON).contains(&t) {
                        intersections.push(a[0] + t * (b[0] - a[0]));
                    }
                }
            }
            if intersections.len() >= 2 {
                let left = intersections.iter().copied().fold(f64::INFINITY, f64::min);
                let right = intersections
                    .iter()
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max);
                if left.is_finite() && right.is_finite() && right >= left {
                    intervals.push([left, right]);
                }
            }
        }
        intervals.sort_by(|a, b| a[0].total_cmp(&b[0]));
        let mut covered_right = target_left;
        let mut started = false;
        for [left, right] in intervals {
            if right + EDGE_EPSILON < target_left || left - EDGE_EPSILON > target_right {
                continue;
            }
            if !started {
                if left > target_left + EDGE_EPSILON {
                    return false;
                }
                covered_right = right;
                started = true;
            } else if left > covered_right + EDGE_EPSILON {
                return false;
            } else {
                covered_right = covered_right.max(right);
            }
            if covered_right >= target_right - EDGE_EPSILON {
                break;
            }
        }
        if !started || covered_right < target_right - EDGE_EPSILON {
            return false;
        }
    }
    true
}

/// Coverage proof for shader-pulled positions: evaluate the translated vertex
/// SPIR-V against the decoded bound buffer bytes for every drawn index, then
/// run the same bounds + triangle-union proof in pixel space. Every failure is
/// a named slug for the `linear_coverage_gap` line; callers must treat it as
/// "coverage unproven", never as a position default.
#[cfg(feature = "backend-vulkan")]
fn draw_full_target_coverage_shader_pulled(
    resources: &crate::backend::vulkan::engine::DrawRequest,
    width: u32,
    height: u32,
    vertex_count: u32,
    v_words: &[u32],
) -> Result<FullTargetCoverage, ShaderPulledCoverageDecline> {
    let mut coverage = FullTargetCoverage::default();
    if width == 0 || height == 0 {
        return Err(ShaderPulledCoverageDecline::ZeroTarget);
    }
    if !full_viewport_scissor(resources, width, height) {
        return Err(ShaderPulledCoverageDecline::PartialViewportOrScissor);
    }
    let indices = decode_coverage_indices(resources, vertex_count)
        .ok_or(ShaderPulledCoverageDecline::IndexStreamInvalid)?;
    if indices.len() < 3 {
        return Err(ShaderPulledCoverageDecline::TooFewIndices {
            count: indices.len(),
        });
    }
    let mut unique = indices.clone();
    unique.sort_unstable();
    unique.dedup();
    let buffer_views: Vec<(u32, std::borrow::Cow<'_, [u8]>)> = resources
        .storage_buffers
        .iter()
        .map(|b| (b.binding, b.content.cpu_bytes()))
        .collect();
    let buffers: Vec<(u32, &[u8])> = buffer_views
        .iter()
        .map(|(binding, view)| (*binding, view.as_ref()))
        .collect();
    let clip = crate::runtime::spirv_vertex_eval::evaluate_vertex_clip_positions(
        v_words,
        &buffers,
        &unique,
        resources.base_instance,
    )?;
    let mut positions = std::collections::BTreeMap::<u32, [f64; 2]>::new();
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for (index, clip) in unique.iter().zip(&clip) {
        let clip_w = f64::from(clip[3]);
        if !clip_w.is_finite() || clip_w.abs() < 1.0e-9 {
            return Err(ShaderPulledCoverageDecline::PositionWDegenerate {
                vertex_index: *index,
            });
        }
        let ndc_x = f64::from(clip[0]) / clip_w;
        let ndc_y = f64::from(clip[1]) / clip_w;
        if !ndc_x.is_finite() || !ndc_y.is_finite() {
            return Err(ShaderPulledCoverageDecline::PositionNotFinite {
                vertex_index: *index,
            });
        }
        let px = (ndc_x + 1.0) * f64::from(width) * 0.5;
        let py = (ndc_y + 1.0) * f64::from(height) * 0.5;
        positions.insert(*index, [px, py]);
        min_x = min_x.min(px);
        min_y = min_y.min(py);
        max_x = max_x.max(px);
        max_y = max_y.max(py);
    }
    // Sub-half-pixel slack absorbs f32 transform rounding; the scanline proof
    // below (against exact pixel centers) remains the authority.
    const BOUNDS_EPSILON_PX: f64 = 1.0e-3;
    coverage.bounds_span = min_x <= BOUNDS_EPSILON_PX
        && min_y <= BOUNDS_EPSILON_PX
        && max_x >= f64::from(width) - BOUNDS_EPSILON_PX
        && max_y >= f64::from(height) - BOUNDS_EPSILON_PX;
    if !coverage.bounds_span {
        return Ok(coverage);
    }
    coverage.triangles_cover = triangle_union_covers(
        &positions,
        &indices,
        resources.primitive_topology,
        width,
        height,
    );
    Ok(coverage)
}

#[allow(clippy::too_many_arguments)]
fn log_compositor_linear_coverage(
    output_mapping: u32,
    width: u32,
    height: u32,
    pipeline_ref: u32,
    full: bool,
    bounds_span: bool,
    vertex_count: u32,
    indexed: bool,
    viewport: Option<[f64; 4]>,
    scissor: Option<(u32, u32, u32, u32)>,
    src: &str,
) {
    // This census is emitted for every same-geometry linear-sample draw. Under an
    // active compositor (e.g. Notification Center widgets, which draw tens of
    // thousands of vertex-pulled tile quads) that is tens of thousands of always-on
    // writes per interaction — a render-worker stall AND a flood that drowns
    // genuine failure lines. Dedup to one line per distinct
    // (mid, pipe, full, bounds, src) verdict, so every coverage-state transition is
    // still fail-visible while steady-state repeats are suppressed.
    use std::collections::HashSet;
    use std::sync::Mutex;
    type FullTargetVerdictKey = (u32, u32, bool, bool, String);
    static SEEN: Mutex<Option<HashSet<FullTargetVerdictKey>>> = Mutex::new(None);
    let first = {
        let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
        seen.get_or_insert_with(HashSet::new).insert((
            output_mapping,
            pipeline_ref,
            full,
            bounds_span,
            src.to_string(),
        ))
    };
    if !first {
        return;
    }
    crate::observe::off(format!(
        "compositor_linear_coverage output_mid={output_mapping} {width}x{height} pipe={pipeline_ref} full={} bounds={} src={src} vtx={vertex_count} indexed={} viewport={viewport:?} scissor={scissor:?}",
        full as u8,
        bounds_span as u8,
        indexed as u8,
    ));
    if bounds_span && !full {
        crate::observe::off(format!(
            "linear_coverage_gap reason=bounds_only_triangle_gap action=reject_edge output_mid={output_mapping} {width}x{height} pipe={pipeline_ref} vtx={vertex_count} indexed={}",
            indexed as u8
        ));
    }
}

#[allow(clippy::too_many_arguments)]
fn log_shader_pulled_coverage_gap(
    output_mapping: u32,
    width: u32,
    height: u32,
    pipeline_ref: u32,
    vertex_count: u32,
    indexed: bool,
    reflection: &metal2vulkan::reflect::ShaderReflection,
    decline: &ShaderPulledCoverageDecline,
) {
    if !crate::runtime::spirv_bind::vertex_position_pull_gate(reflection) {
        return;
    }
    // Vertex-pulled compositor tiles (Notification Center widgets) issue this
    // rejected-edge shape on every partial-coverage draw — 25k+ per interaction on
    // a single 1024x1024 tile. Dedup to one line per distinct (mid, pipe, eval) so
    // each unresolved-position cause stays fail-visible without flooding the
    // always-on log or stalling the render worker.
    use std::collections::HashSet;
    use std::sync::Mutex;
    static SEEN: Mutex<Option<HashSet<(u32, u32, String)>>> = Mutex::new(None);
    let rendered = crate::observe::Emit::decline("linear_coverage_gap", decline).render();
    let first = {
        let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
        seen.get_or_insert_with(HashSet::new)
            .insert((output_mapping, pipeline_ref, rendered))
    };
    if !first {
        return;
    }
    let bindings = crate::runtime::spirv_bind::vertex_pull_buffer_bindings(reflection)
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(",");
    crate::observe::Emit::decline("linear_coverage_gap", decline)
        .field("class", "shader_pulled_position_unresolved")
        .field("action", "reject_edge")
        .field("output_mid", output_mapping)
        .field("width", width)
        .field("height", height)
        .field("pipe", pipeline_ref)
        .field("vtx", vertex_count)
        .field("indexed", indexed as u8)
        .field("position_out", 1)
        .field("vertex_index", 1)
        .field("storage_bindings", format!("[{bindings}]"))
        .off();
}

/// Result of a Linux metal2vulkan draw.
#[cfg(feature = "backend-vulkan")]
enum M2vDrawSpan {
    /// No drawable color0 geom.
    None,
    /// CPU-side RGBA8 pixels (readback path).
    Rgba(Vec<u8>),
    /// Resident BGRA target ready for safe import-present (no CPU pixels).
    #[cfg(feature = "backend-vulkan")]
    ResidentBgra {
        identity: crate::backend::vulkan::engine::TargetIdentity,
        mapping_id: u32,
        width: u32,
        height: u32,
        /// Non-self full-geometry type-11 inputs resolved by this draw.
        display_sample_mids: Vec<u32>,
        /// A decoded same-geometry type-2/3 input covered the full output.
        full_geometry_linear_sample: bool,
        /// Decoded location-0 bounds span the full target (pixel or NDC) —
        /// diagnostic provenance for the `fullquad_store_noop` proxy.
        full_quad_bounds: bool,
    },
    /// Intermediate record of a resident render-pass chain: content stays on
    /// the protocol-keyed engine target (no CPU pixels, no fence wait, no guest
    /// Store this record). The final record reads back and performs the
    /// contract Store on portability devices.
    #[cfg(feature = "backend-vulkan")]
    ResidentChain,
    /// Final/single record of a GVA render Store executed into the registry
    /// resident with `skip_readback`: the caller arms a deferred-writeback
    /// window (`DeviceState::gva_deferred_flush`) instead of the sync
    /// readback + guest write on the stamp path; guest bytes + encode caches
    /// land on first access (`storage_flush::flush_gva_one`).
    #[cfg(feature = "backend-vulkan")]
    ResidentGvaStore,
}

#[cfg(feature = "backend-vulkan")]
#[inline]
fn type11_import_allowed(
    store_is_store: bool,
    stable_host_import: bool,
    mapping_id: u32,
    width: u32,
    height: u32,
) -> bool {
    store_is_store
        && stable_host_import
        && crate::runtime::import_present::eligible(mapping_id, width, height)
}

#[cfg(feature = "backend-vulkan")]
#[inline]
fn type11_cpu_store_fallback_allowed(import_allowed: bool) -> bool {
    !import_allowed
}

fn publish_cpu_portability_store<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mapping_id: u32,
    width: u32,
    height: u32,
    format: u16,
) {
    state.note_surface_composite(mapping_id);
    state.note_dense_frame_published(mapping_id, width, height);
    crate::runtime::scanout::note_front_buffer_writeback(
        state, host, mapping_id, width, height, format,
    );
}

/// Build the engine's secondary MRT attachments (slot 1..) from a draw's color
/// list. Empty result ⇒ the classic single-RT path (no regression). The vibrancy
/// tile is the driving case: slot 0 the visible BGRA8 target, slot 1 an
/// RG16Float coverage mask that a later draw samples — dropping slot 1 (the
/// pre-MRT engine) left that sample reading zero (frosted background
/// pass-through). Each secondary persists as a registry resident keyed by its
/// protocol identity so the later sample binds it directly.
///
/// Conservative by construction — any ambiguity yields an empty vector rather
/// than a guessed attachment: requires a resident primary, contiguous slots
/// (0,1,2,… matching the shader's `location`s), matching framebuffer geometry,
/// a known color-renderable format, and a resolvable identity.
#[cfg(feature = "backend-vulkan")]
fn build_secondary_targets(
    state: &DeviceState,
    colors: &[ColorRtRequest],
    pipeline: &crate::runtime::decode::resource::RenderPipelineDescriptor,
    primary: &crate::backend::vulkan::engine::TargetIdentity,
    fb_w: u32,
    fb_h: u32,
    blend_constants: [f32; 4],
) -> Vec<crate::backend::vulkan::engine::SecondaryColorTarget> {
    use crate::backend::vulkan::engine::{SecondaryColorTarget, TargetIdentity};
    if colors.len() <= 1 {
        return Vec::new();
    }
    let mut out = Vec::new();
    for (i, c) in colors.iter().enumerate().skip(1) {
        // Contiguous slots only — the render pass maps location N → attachment N,
        // so a gap would misalign the shader's outputs.
        if c.slot as usize != i || c.texture_ref == 0 {
            crate::runtime::census::present_proxy::note_secondary_mrt_drop(
                crate::runtime::census::present_proxy::MrtDrop::NonContiguousSlot,
                c.width,
                c.height,
            );
            return Vec::new();
        }
        // MRT requires every attachment to share the framebuffer geometry.
        if c.width != fb_w || c.height != fb_h {
            crate::runtime::census::present_proxy::note_secondary_mrt_drop(
                crate::runtime::census::present_proxy::MrtDrop::GeometryMismatch,
                c.width,
                c.height,
            );
            return Vec::new();
        }
        // Unknown wire format stays unknown — never guess a secondary layout —
        // and a known format whose sRGB qualifier this attachment cannot carry
        // says so instead of folding silently.
        let format = match translate::pixel::color_attachment(c.format) {
            Ok((format, decline)) => {
                if decline.is_some() {
                    srgb_census::note_downgrade(
                        srgb_census::site::SECONDARY_COLOR_TARGET,
                        c.format,
                    );
                }
                format
            }
            Err(_) => {
                crate::runtime::census::present_proxy::note_secondary_mrt_drop(
                    crate::runtime::census::present_proxy::MrtDrop::UnknownFormat,
                    c.width,
                    c.height,
                );
                return Vec::new();
            }
        };
        // Identity mirrors the primary namespaces: type-2/3 linear GVA, else
        // type-11 surface. generation 0 for GVA matches the later sample path
        // (`try_sample_deferred_gva`) so the resident key is bind-compatible.
        let identity = if c.target_gva != 0 {
            TargetIdentity::Gva {
                gva: c.target_gva,
                width: c.width,
                height: c.height,
                generation: 0,
            }
        } else if c.mapping_id != 0 {
            crate::runtime::import_present::surface_identity(state, c.mapping_id, c.width, c.height)
        } else {
            crate::runtime::census::present_proxy::note_secondary_mrt_drop(
                crate::runtime::census::present_proxy::MrtDrop::NoIdentity,
                c.width,
                c.height,
            );
            return Vec::new();
        };
        // A secondary aliasing the primary target is a degenerate feedback loop
        // the engine rejects — bail to the safe single-RT path.
        if identity == *primary {
            crate::runtime::census::present_proxy::note_secondary_mrt_drop(
                crate::runtime::census::present_proxy::MrtDrop::AliasesPrimary,
                c.width,
                c.height,
            );
            return Vec::new();
        }
        let load = c.load_action == PASS_LOAD_ACTION_LOAD;
        let clear = [
            c.clear_color[0] as f32,
            c.clear_color[1] as f32,
            c.clear_color[2] as f32,
            c.clear_color[3] as f32,
        ];
        // This slot's own blend, resolved exactly as the Metal arm resolves it:
        // find the pipeline's attachment entry for this Metal slot. No
        // `or_else(first())` fallback here — the Metal path has one for its
        // compat `color0` alias, but a secondary slot with no entry of its own
        // has no blend state, and borrowing slot 0's would be inventing one.
        let blend = pipeline
            .color_attachments
            .iter()
            .find(|a| a.slot == c.slot)
            .filter(|a| a.blending_enabled)
            .and_then(|a| {
                match translate::blend::state(
                    a.src_rgb,
                    a.dst_rgb,
                    a.op_rgb,
                    a.src_alpha,
                    a.dst_alpha,
                    a.op_alpha,
                    blend_constants,
                ) {
                    Ok(state) => {
                        crate::runtime::census::present_proxy::note_secondary_mrt_blend(
                            c.slot, c.width, c.height,
                        );
                        Some(state)
                    }
                    // An out-of-contract blend factor or op on a secondary
                    // slot: the attachment still renders, unblended, and the
                    // decline says which value refused rather than the slot
                    // quietly becoming a raw store the way every slot used to.
                    Err(reason) => {
                        crate::observe::fail(format!(
                            "secondary_blend_unmapped {reason} slot={} {}x{}",
                            c.slot, c.width, c.height
                        ));
                        None
                    }
                }
            });
        out.push(SecondaryColorTarget {
            identity,
            width: c.width,
            height: c.height,
            format,
            clear,
            load,
            blend,
        });
    }
    out
}

/// Draw-pipeline analog of the compute `dump_kernel_handoff`.
/// `REIMS_VGPU_M2V_DUMP_DRAW_PIPES` selects pipes through the shared
/// [`HandoffPipeSelection`] — a comma-separated list of pipeline refs, or
/// `all`, same grammar as `REIMS_VGPU_M2V_DUMP_COMPUTE_PIPES`. A listed pipe's
/// vertex and fragment stages land under
/// [`crate::runtime::compute_exec::m2v_handoff_dir`] once per boot as
/// `pipe<N>.draw.{vertex,fragment}.{mtlb,air,spv}` plus a `.txt` with the draw
/// shape. Probe tooling only — never alters device behavior; unset env means
/// zero work per draw beyond one cached parse.
#[cfg(feature = "backend-vulkan")]
fn dump_draw_handoff(
    req: &DrawEncodeRequest,
    vertex_func_ref: u32,
    fragment_func_ref: u32,
    stages: [DrawHandoffStage<'_>; 2],
) {
    use crate::runtime::compute_exec::HandoffPipeSelection;
    use std::sync::OnceLock;
    static WANTED: OnceLock<HandoffPipeSelection> = OnceLock::new();
    let pipe = req.pipeline_ref;
    if !WANTED
        .get_or_init(|| HandoffPipeSelection::from_env("REIMS_VGPU_M2V_DUMP_DRAW_PIPES"))
        .wants(pipe)
    {
        return;
    }
    use std::collections::HashSet;
    use std::sync::Mutex;
    static SEEN: Mutex<Option<HashSet<u32>>> = Mutex::new(None);
    {
        let mut g = SEEN.lock().unwrap_or_else(|e| e.into_inner());
        if !g.get_or_insert_with(HashSet::new).insert(pipe) {
            return;
        }
    }
    let dir = crate::runtime::compute_exec::m2v_handoff_dir();
    if std::fs::create_dir_all(&dir).is_err() {
        return;
    }
    for (stage, mtlb, air, spv) in stages {
        let stem = format!("pipe{pipe}.draw.{stage}");
        let _ = std::fs::write(dir.join(format!("{stem}.mtlb")), mtlb);
        let _ = std::fs::write(dir.join(format!("{stem}.air")), air);
        let _ = std::fs::write(dir.join(format!("{stem}.spv")), spv);
    }
    let ftex: Vec<u32> = req
        .fragment_textures
        .iter()
        .map(|t| t.texture_ref)
        .collect();
    let fbuf: Vec<u32> = req.fragment_buffers.iter().map(|b| b.index).collect();
    let meta = format!(
        "pipe={pipe}\nv_func_ref={vertex_func_ref}\nf_func_ref={fragment_func_ref}\n\
         geom={}x{} fmt={}\nvtx_count={} inst={} prim={} indexed={}\n\
         ftex_refs={ftex:?}\nfbuf_indices={fbuf:?}\ncolors={} depth={} stencil={}\n",
        req.width,
        req.height,
        req.format,
        req.vertex_count,
        req.instance_count,
        req.primitive_type,
        req.indexed.is_some(),
        req.colors.len(),
        req.depth_attach.is_some(),
        req.stencil_attach.is_some(),
    );
    let _ = std::fs::write(dir.join(format!("pipe{pipe}.draw.txt")), meta);
    crate::observe::fail(format!(
        "linux_m2v draw_dump pipe={pipe} dir={}",
        dir.display()
    ));
}

/// Translate guest MTLB stages via metal2vulkan and raster with the internal Vulkan engine.
///
/// Builds engine [`DrawRequest`] resources from stream binds (stage-in attrs, SSBOs,
/// sampled images) — bare `render_offscreen` without binds yields black alpha-only
/// frames that wipe CLEAR stores. Archive `render_draw_core` is the contract model.
///
/// Type-11 Stores return [`M2vDrawSpan::ResidentBgra`] for zero-copy import
/// (revalidate + strided host ptr) on backends that can keep guest-visible
/// content resident. Portability-subset devices take the synchronous CPU
/// writeback path so guest pages remain authoritative across device recreates.
#[cfg(feature = "backend-vulkan")]
fn prepare_vertex_attribute_format(
    attribute: &crate::runtime::decode::resource::VertexAttribute,
) -> Result<crate::backend::vulkan::engine::VertexAttributeFormat, DrawPreparationDecline> {
    translate::vertex::attribute_format(attribute.format).map_err(|reason| {
        DrawPreparationDecline::VertexAttributeFormat {
            location: attribute.location,
            buffer_index: attribute.buffer_index,
            raw_format: attribute.format,
            reason,
        }
    })
}

#[cfg(feature = "backend-vulkan")]
fn prepare_vertex_step_function(
    attribute: &crate::runtime::decode::resource::VertexAttribute,
) -> Result<crate::backend::vulkan::engine::VertexStepFunction, DrawPreparationDecline> {
    translate::vertex::step_function(attribute.has_step_function, attribute.step_function).map_err(
        |reason| DrawPreparationDecline::VertexStepFunctionUnsupported {
            location: attribute.location,
            buffer_index: attribute.buffer_index,
            reason,
        },
    )
}

#[cfg(feature = "backend-vulkan")]
fn try_metal2vulkan_draw<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    req: &DrawEncodeRequest,
    writeback_guest: bool,
) -> Result<M2vDrawSpan, DrawError> {
    // Only the final record of a portability render-pass chain reads back CPU
    // pixels; used by the resident-chain rail below (harmless on other paths).
    let _ = &writeback_guest;
    let t_total = std::time::Instant::now();
    let t_load = std::time::Instant::now();
    let pd = load_render_pipeline(state, host, req.task_id, req.pipeline_ref).ok_or_else(|| {
        DrawError::DrawPreparation(
            crate::backend::vulkan::engine::DrawPreparationDecline::PipelineMissing {
                task_id: req.task_id,
                pipeline_ref: req.pipeline_ref,
            },
        )
    })?;
    let v_mtlb = load_mtlb(state, host, req.task_id, pd.vertex_func_ref).ok_or_else(|| {
        DrawError::DrawPreparation(
            crate::backend::vulkan::engine::DrawPreparationDecline::VertexMtlbMissing {
                task_id: req.task_id,
                function_ref: pd.vertex_func_ref,
            },
        )
    })?;
    let f_mtlb = load_mtlb(state, host, req.task_id, pd.fragment_func_ref).ok_or_else(|| {
        DrawError::DrawPreparation(
            crate::backend::vulkan::engine::DrawPreparationDecline::FragmentMtlbMissing {
                task_id: req.task_id,
                function_ref: pd.fragment_func_ref,
            },
        )
    })?;
    let v_air = crate::runtime::mtlb::extract_air(&v_mtlb)
        .map_err(|reason| {
            DrawError::DrawPreparation(
                crate::backend::vulkan::engine::DrawPreparationDecline::VertexAirExtract {
                    function_ref: pd.vertex_func_ref,
                    reason,
                },
            )
        })?
        .to_vec();
    let f_air = crate::runtime::mtlb::extract_air(&f_mtlb)
        .map_err(|reason| {
            DrawError::DrawPreparation(
                crate::backend::vulkan::engine::DrawPreparationDecline::FragmentAirExtract {
                    function_ref: pd.fragment_func_ref,
                    reason,
                },
            )
        })?
        .to_vec();
    let load_us = elapsed_us(t_load);

    // AIR→SPIR-V is content-cached: live boots re-translated the same pipelines
    // dozens of times on the doorbell vCPU and tripped IPI timeout panics.
    let t_m2v = std::time::Instant::now();
    // Reflected translate: the cached shader carries the metal2vulkan reflection
    // facade so per-draw texture provisioning reads dimensionality straight from
    // the AIR-derived metadata (single source of truth) rather than re-walking the
    // emitted SPIR-V. `_shader.reflection` is used at the sampled-image binding
    // loop below; the SPIR-V walk stays as a cold fallback.
    let v_shader = crate::runtime::m2v_cache::translate_cached_reflected(
        &v_air,
        metal2vulkan::passes::Stage::Vertex,
        req.pipeline_ref,
    )
    .map_err(|reason| {
        DrawError::DrawPreparation(
            crate::backend::vulkan::engine::DrawPreparationDecline::VertexTranslate {
                pipeline_ref: req.pipeline_ref,
                reason,
            },
        )
    })?;
    let f_shader = crate::runtime::m2v_cache::translate_cached_reflected(
        &f_air,
        metal2vulkan::passes::Stage::Fragment,
        req.pipeline_ref,
    )
    .map_err(|reason| {
        DrawError::DrawPreparation(
            crate::backend::vulkan::engine::DrawPreparationDecline::FragmentTranslate {
                pipeline_ref: req.pipeline_ref,
                reason,
            },
        )
    })?;
    let m2v_us = elapsed_us(t_m2v);

    dump_draw_handoff(
        req,
        pd.vertex_func_ref,
        pd.fragment_func_ref,
        [
            ("vertex", &v_mtlb, &v_air, &v_shader.spirv),
            ("fragment", &f_mtlb, &f_air, &f_shader.spirv),
        ],
    );

    crate::observe::line(format!(
        "linux_m2v_pair pipe={} v_spv={} f_spv={}",
        req.pipeline_ref,
        v_shader.spirv.len(),
        f_shader.spirv.len()
    ));
    let (w, h) = if req.width > 0 && req.height > 0 {
        (req.width, req.height)
    } else if let Some(c0) = req.colors.first() {
        (c0.width, c0.height)
    } else {
        state.tranche.add(log_linux_m2v_timing(
            M2vTiming {
                pipe: req.pipeline_ref,
                total_us: elapsed_us(t_total),
                load_us,
                m2v_us,
                ..M2vTiming::default()
            },
            crate::backend::vulkan::engine::CounterSnapshot::default(),
        ));
        return Ok(M2vDrawSpan::None);
    };
    if w == 0 || h == 0 || w > 4096 || h > 4096 {
        return Err(DrawError::DrawPreparation(
            DrawPreparationDecline::GeometryUnsupported {
                width: w,
                height: h,
            },
        ));
    }

    // setup_us: SPIR-V words, reloc, guest binds, seed, engine DrawRequest assembly.
    let t_setup = std::time::Instant::now();

    // SPIR-V words for the engine, shared from the translation cache (Arc — no
    // per-draw materialization; fragment reloc variants are cached per shader).
    let v_words = v_shader.words.clone();
    #[allow(unused_mut)]
    let mut f_words = f_shader.words.clone();

    #[cfg(feature = "backend-vulkan")]
    {
        use crate::runtime::spirv_bind::{
            FRAG_BUFFER_BINDING_OFFSET, FRAG_SAMPLED_RESOURCE_BINDING_OFFSET, SAMPLER_BINDING_BASE,
            TEXTURE_BINDING_BASE,
        };
        let t_spv_done = std::time::Instant::now();

        // Materialize stream buffer binds (vertex + fragment). Large spans
        // ride the zero-copy rail (the GPU gathers them from imported guest
        // RAM at execute time); the rest stay on the CPU staging read.
        // Constant-step attribute streams stay CPU: the engine prepends a
        // base-instance prefix to those bytes at prepare time.
        let constant_step_bufs: std::collections::BTreeSet<u32> = pd
            .vertex_attributes
            .iter()
            .filter(|a| {
                a.format != 0 && a.stride != 0 && a.has_step_function && a.step_function == 0
            })
            .map(|a| a.buffer_index)
            .collect();
        let mut vtx_storage: Vec<(u32, crate::backend::vulkan::engine::BufferContent)> = Vec::new();
        for b in &req.vertex_buffers {
            if b.index >= MAX_BIND_SLOTS || b.buffer_ref == 0 {
                continue;
            }
            let allow_zc = !constant_step_bufs.contains(&b.index);
            let Some(content) =
                load_buffer_content(state, host, req.task_id, b.buffer_ref, b.offset, allow_zc)
            else {
                return Err(DrawError::DrawPreparation(
                    DrawPreparationDecline::VertexBufferMissing {
                        index: b.index,
                        buffer_ref: b.buffer_ref,
                        offset: b.offset,
                    },
                ));
            };
            vtx_storage.push((b.index, content));
        }
        let mut frag_storage: Vec<(u32, crate::backend::vulkan::engine::BufferContent)> =
            Vec::new();
        for b in &req.fragment_buffers {
            if b.index >= MAX_BIND_SLOTS || b.buffer_ref == 0 {
                continue;
            }
            let Some(content) =
                load_buffer_content(state, host, req.task_id, b.buffer_ref, b.offset, true)
            else {
                return Err(DrawError::DrawPreparation(
                    DrawPreparationDecline::FragmentBufferMissing {
                        index: b.index,
                        buffer_ref: b.buffer_ref,
                        offset: b.offset,
                    },
                ));
            };
            frag_storage.push((b.index, content));
        }
        state.tranche.bufs_load_ns = state
            .tranche
            .bufs_load_ns
            .saturating_add(t_spv_done.elapsed().as_nanos() as u64);

        // Stage-in attributes from pipeline vertex block + bound buffer bytes.
        let mut attrs: Vec<crate::backend::vulkan::engine::VertexAttributeResource> = Vec::new();
        let mut stage_in_bufs: std::collections::BTreeSet<u32> = Default::default();
        for a in &pd.vertex_attributes {
            if a.format == 0 || a.stride == 0 {
                continue;
            }
            let format = prepare_vertex_attribute_format(a).map_err(DrawError::DrawPreparation)?;
            let content = vtx_storage
                .iter()
                .find(|(idx, _)| *idx == a.buffer_index)
                .map(|(_, d)| d.clone())
                .unwrap_or_else(|| crate::backend::vulkan::engine::BufferContent::from(Vec::new()));
            if !content.is_empty() {
                stage_in_bufs.insert(a.buffer_index);
            } else if a.format != 0 {
                // Pipeline declares stage-in but stream did not bind bytes — fail
                // visibly rather than raster black garbage that wipes CLEAR.
                return Err(DrawError::DrawPreparation(
                    DrawPreparationDecline::StageInBytesMissing {
                        location: a.location,
                        buffer_index: a.buffer_index,
                        raw_format: a.format,
                        stride: a.stride,
                    },
                ));
            }
            let step = prepare_vertex_step_function(a).map_err(DrawError::DrawPreparation)?;
            let step_rate = if a.has_step_rate {
                a.step_rate.max(1)
            } else {
                1
            };
            attrs.push(crate::backend::vulkan::engine::VertexAttributeResource {
                location: a.location,
                // One Vulkan binding per location (archive render_draw_core).
                binding: a.location,
                format,
                offset: a.offset,
                stride: a.stride,
                step_function: step,
                step_rate,
                content,
            });
        }

        // Fragment/vertex buffer index collision → relocate fragment SPIR-V buffers.
        let vtx_idx: std::collections::BTreeSet<u32> =
            vtx_storage.iter().map(|(i, _)| *i).collect();
        let buf_collide = frag_storage.iter().any(|(i, _)| vtx_idx.contains(i));
        let has_vtx_tex = req
            .vertex_textures
            .iter()
            .any(|t| t.index < MAX_BIND_SLOTS && t.texture_ref != 0);
        let has_frag_tex = req
            .fragment_textures
            .iter()
            .any(|t| t.index < MAX_BIND_SLOTS && t.texture_ref != 0);
        let reflected_sampled_collision =
            reflected_sampled_binding_collision(&v_shader.reflection, &f_shader.reflection);
        let separate_sampled =
            (has_vtx_tex && has_frag_tex) || buf_collide || reflected_sampled_collision;
        // Sampled relocation first (archive order), then buffer band. The
        // buffer band lands at [104,136), clear of the [96,104) ColorInput /
        // framebuffer-fetch band, which neither relocation touches. The
        // sampled-with-buffer coupling is kept so the engine's image/sampler
        // binding base mirrors one flag pair, not a third variant.
        if separate_sampled || buf_collide {
            f_words = f_shader.fragment_words(separate_sampled, buf_collide);
        }

        // Non-stage-in vertex buffers + fragment buffers as storage buffers.
        //
        // A vertex buffer can be BOTH a stage-in source (the pipeline vertex
        // descriptor declares attributes on it) AND read directly as a
        // StorageBuffer by the vertex function (`[[buffer(N)]]` -> descriptor
        // binding N). WebKit's glyph vertex shader is exactly this: the pipeline
        // declares a stride-48 stage-in on buffer 1, but the translated SPIR-V
        // never reads a stage-in input — it indexes buffer 1 as a per-glyph
        // record array (StorageBuffer binding 1) by `gl_InstanceIndex`. Skipping
        // every stage-in buffer left that binding unbound, so each glyph read a
        // zero position/size and collapsed to a degenerate (zero-area) quad —
        // the "blank Safari body text" class. Bind a stage-in buffer as storage
        // too whenever the vertex SPIR-V structurally declares a StorageBuffer at
        // that binding (decoration-driven, never name-keyed).
        let mut storage: Vec<crate::backend::vulkan::engine::StorageBufferResource> = Vec::new();
        for (idx, content) in &vtx_storage {
            if !vertex_buffer_needs_storage_binding(&v_words, *idx, stage_in_bufs.contains(idx)) {
                continue;
            }
            storage.push(crate::backend::vulkan::engine::StorageBufferResource {
                binding: *idx,
                content: content.clone(),
            });
        }
        for (idx, content) in &frag_storage {
            let binding = if buf_collide {
                *idx + FRAG_BUFFER_BINDING_OFFSET
            } else {
                *idx
            };
            storage.push(crate::backend::vulkan::engine::StorageBufferResource {
                binding,
                content: content.clone(),
            });
        }

        // GUARD (always-on fail-visible, drain worker / off-main-core): does the
        // FRAGMENT shader DECLARE a `[[buffer(n)]]`/`[[texture(n)]]`/`[[sampler(n)]]`
        // the draw never bound? Such a resource reads an undefined descriptor and
        // paints garbage with no other fail-log — the fragment-stage analog of the
        // fixed vertex stage-in "blank body text" class (comment above), which the
        // fragment stage otherwise has no cross-check for. This closes that silent
        // mis-execution hole: the Vulkan engine builds its descriptor layout purely
        // from provided resources (`engine/exec.rs`), so a shader referencing an
        // unbound descriptor executes with no error. Fragment-only: a vertex
        // `[[buffer(n)]]` may legitimately be bound as a stage-in attribute (not
        // storage), so a vertex check would false-fire. Standard directly-bound
        // kinds only; ColorInput / ThreadgroupBuffer / StorageImage reach the shader
        // by other paths and carry their own census (`census_reflection_wellformed`).
        // Verified non-flooding: 0 fires across a full x86 boot (desktop convergence
        // + Safari + CSS gradients + a 23-binding compositor shader), so any fire is
        // a genuine bind gap, not expected control flow.
        {
            // Membership predicates over the (tiny) provided-resource slices — the
            // scan allocates nothing on the all-bound hot path (both result Vecs
            // stay empty). The `frag_embedded_*` reason names a DIFFERENT silent
            // hole: a fragment shader declaring an `EmbeddedArgBufferTexture` (m2v
            // flattened out of an `air.indirect_buffer` arg) that only the compute
            // path can source, so the render path leaves it structurally unbound.
            let (unbound, embedded) = frag_unbound_scan(
                &f_shader.reflection.bindings,
                |i| frag_storage.iter().any(|(x, _)| *x == i),
                |i| {
                    req.fragment_textures
                        .iter()
                        .any(|t| t.index == i && t.index < MAX_BIND_SLOTS && t.texture_ref != 0)
                },
                |i| {
                    req.fragment_samplers
                        .iter()
                        .any(|s| s.index == i && s.index < MAX_BIND_SLOTS && s.sampler_ref != 0)
                },
            );
            if !unbound.is_empty() {
                // Cold path only: build the provided-index sets for the log detail.
                let bufs: std::collections::BTreeSet<u32> =
                    frag_storage.iter().map(|(i, _)| *i).collect();
                let texs: std::collections::BTreeSet<u32> = req
                    .fragment_textures
                    .iter()
                    .filter(|t| t.index < MAX_BIND_SLOTS && t.texture_ref != 0)
                    .map(|t| t.index)
                    .collect();
                let smps: std::collections::BTreeSet<u32> = req
                    .fragment_samplers
                    .iter()
                    .filter(|s| s.index < MAX_BIND_SLOTS && s.sampler_ref != 0)
                    .map(|s| s.index)
                    .collect();
                crate::observe::fail(format!(
                    "shader_resource_declared_unbound reason=frag_declared_descriptor_unbound \
                     pipe={} unbound=[{}] provided_buf={bufs:?} provided_tex={texs:?} \
                     provided_smp={smps:?} {}x{}",
                    req.pipeline_ref,
                    unbound.join(","),
                    w,
                    h
                ));
            }
            if !embedded.is_empty() {
                crate::observe::fail(format!(
                    "shader_resource_declared_unbound reason=frag_embedded_argbuffer_unsupported \
                     pipe={} embedded_tex={embedded:?} {}x{} \
                     (render path cannot source air.indirect_buffer textures)",
                    req.pipeline_ref, w, h
                ));
            }
        }

        // Framebuffer fetch (`air.render_target` INPUT param `dest_N` →
        // reflection `ColorInput` at binding 96+N): the engine supports the
        // attachment-0 fetch as a Vulkan subpass input. `dest_N>0` (fetching a
        // secondary MRT attachment) has no engine path yet — fail visibly, never
        // execute a shader whose destination read would be unbound.
        let frag_color_input = {
            use metal2vulkan::reflect::ResourceKind;
            let mut fetch0 = false;
            for rb in &f_shader.reflection.bindings {
                if rb.kind == ResourceKind::ColorInput {
                    if rb.metal_index == 0 {
                        fetch0 = true;
                    } else {
                        return Err(DrawError::DrawPreparation(
                            DrawPreparationDecline::ColorInputMrtUnsupported {
                                destination_index: rb.metal_index,
                            },
                        ));
                    }
                }
            }
            fetch0
        };

        let t_bufs_done = std::time::Instant::now();
        // Sampled textures + samplers (metal2vulkan bands: textures 32+N, samplers 64+M).
        // Texture and sampler **indices are independent** (live logo SPIR-V: image
        // binding 35 = texture(3), sampler binding 64 = sampler(0)). Pairing
        // sampler to texture index left sampler 67 empty → black samples.
        // Fragment sampled resources use +FRAG_SAMPLED when either both stages
        // sample or fragment buffers moved into the sampled/static-sampler band.
        let mut images: Vec<crate::backend::vulkan::engine::SampledImageResource> = Vec::new();
        let mut samplers: Vec<crate::backend::vulkan::engine::SamplerResource> = Vec::new();
        let mut sampler_binds: std::collections::BTreeSet<u32> = Default::default();
        let mut display_sample_mids: std::collections::BTreeSet<u32> = Default::default();
        let mut full_geometry_linear_sample = false;
        let mut linear_sample_rows: Vec<sample_seed_relation::SampleRow> = Vec::new();
        // Measure multi-bind compositor: which refs load and with what occupancy.
        let mut tex_bind_diag: Vec<String> = Vec::new();
        // Per-bind rgb stats are computed once in the bind loop; the diag
        // censuses below reuse these accumulators instead of re-scanning the
        // (large, Arc-shared) sampled bytes a second and third time per draw.
        let mut bound_tex_rgb: usize = 0;
        // Only the vulkan cfg has resident Target binds; stays false elsewhere.
        #[allow(unused_mut)]
        let mut bound_any_resident_sample = false;
        let mut bound_all_bytes_rgb_empty = true;
        // Type-2/3 linear sample resolved to zero RGB (guest-zero wallpaper).
        // Drives Load keep-seed when multi-bind also binds non-empty layers
        // (serial-224146 pipe=60 sky wipe).
        let mut had_empty_type3_linear = false;
        {
            let mut push_tex = |index: u32,
                                texture_ref: u32,
                                frag_stage: bool|
             -> Result<(), DrawError> {
                if index >= MAX_BIND_SLOTS || texture_ref == 0 {
                    return Ok(());
                }
                // Measure-only setup_tex sub-split (off-main-core): time the full
                // per-bind resolution (guest object-list + descriptor reads +
                // resolve + surface ensure) vs the post-resolve stats scan, so a
                // boot log names which half of the ~800us/draw to cut.
                let t_resolve = std::time::Instant::now();
                let texture_entry =
                    objects::lookup_list_entry(state, host, req.task_id, texture_ref);
                // A type-8 view's channel remap. Resolved here rather than in
                // the loaders because it describes how the bind READS the
                // texture, not what the texture contains: the engine hands it
                // to the image view as a component mapping and the hardware
                // applies it at sample time, so the texels stay untouched and
                // the bind keeps whatever content rail it was already on.
                let view_swizzle = resolve_texture_view(state, host, req.task_id, texture_ref)
                    .and_then(|view| view.swizzle)
                    .filter(|plan| !pixel_format::swizzle_is_identity(plan));
                let is_linear_texture = texture_entry
                    .as_ref()
                    .map(|entry| {
                        entry.object_type == OBJECT_TYPE_TEXTURE
                            || entry.object_type == OBJECT_TYPE_TEXTURE_VARIANT
                    })
                    .unwrap_or(false);
                let linear_identity = texture_entry.as_ref().and_then(|entry| {
                    if !is_linear_texture {
                        return None;
                    }
                    let desc = objects::read_descriptor(state, host, req.task_id, entry)?;
                    let texture = decode_texture_descriptor(&desc).ok()?;
                    Some((entry.object_type, texture.pixel_format))
                });
                let attachment_alias = frag_stage
                    .then(|| fragment_attachment_alias_sample(req, index, texture_ref))
                    .flatten();
                let (tw, th, sampled_mid, loaded) = if let Some((aw, ah, alias)) = attachment_alias
                {
                    match alias {
                        AttachmentAliasSample::Clear(clear) => (
                            aw,
                            ah,
                            0,
                            SampledSourceRequest::Bytes(
                                std::sync::Arc::new(solid_rgba_local(aw, ah, &clear)),
                                None,
                                TexelLayout::Rgba8,
                            ),
                        ),
                        AttachmentAliasSample::Seed(seed) => {
                            let target_gva = req
                                .colors
                                .iter()
                                .find(|color| {
                                    color.slot == index && color.texture_ref == texture_ref
                                })
                                .map(|color| color.target_gva)
                                .unwrap_or(0);
                            log_attachment_alias_chain(
                                req.task_id,
                                index,
                                texture_ref,
                                target_gva,
                                aw,
                                ah,
                                seed,
                            );
                            (
                                aw,
                                ah,
                                0,
                                SampledSourceRequest::Bytes(
                                    std::sync::Arc::new(seed.to_vec()),
                                    None,
                                    TexelLayout::Rgba8,
                                ),
                            )
                        }
                        #[cfg(feature = "backend-vulkan")]
                        AttachmentAliasSample::ResidentChain => {
                            let identity = render_chain_identity(state, req).ok_or_else(|| {
                                DrawError::DrawPreparation(
                                    DrawPreparationDecline::AttachmentAliasIdentityMissing {
                                        index,
                                        texture_ref,
                                    },
                                )
                            })?;
                            if !crate::backend::vulkan::engine::resident_content_ready(&identity) {
                                return Err(DrawError::DrawPreparation(
                                    DrawPreparationDecline::AttachmentAliasResidentNotReady {
                                        index,
                                        texture_ref,
                                        width: identity.width(),
                                        height: identity.height(),
                                    },
                                ));
                            }
                            (
                                identity.width(),
                                identity.height(),
                                0,
                                SampledSourceRequest::Target(identity),
                            )
                        }
                    }
                } else {
                    let Some(loaded) = resolve_sampled_source(
                        state,
                        host,
                        req.task_id,
                        texture_ref,
                        texture_entry,
                    ) else {
                        let detail = sample_miss_detail(state, host, req.task_id, texture_ref);
                        return Err(DrawError::DrawPreparation(
                            DrawPreparationDecline::TextureResolveMissing {
                                stage: if frag_stage { "fragment" } else { "vertex" },
                                index,
                                texture_ref,
                                detail,
                            },
                        ));
                    };
                    loaded
                };
                let resolve_us = t_resolve.elapsed().as_micros() as u64;
                crate::runtime::census::setup_tex_census::note_resolve(resolve_us);
                sampled_census::note_resolve_us(resolve_us);
                let t_stats = std::time::Instant::now();
                if sampled_mid != 0 && tw == w && th == h {
                    display_sample_mids.insert(sampled_mid);
                }
                if is_linear_texture && sampled_mid == 0 && tw == w && th == h {
                    full_geometry_linear_sample = true;
                }
                let mut bytes_identity = None;
                // Byte layout of a CPU-origin bind. Default RGBA8; a native
                // single/dual-channel plane keeps its footprint. The RGBA8-shaped
                // diagnostics below (nz/alpha/center-row, empty-layer proxy) only make
                // sense for 4-byte texels, so they run only on the RGBA8 layout — a
                // native luma/chroma plane skips them. The host spelling is applied
                // once, where the engine resource is built.
                let mut sampled_format = TexelLayout::Rgba8;
                let source = match loaded {
                    SampledSourceRequest::Bytes(rgba, identity, byte_format) => {
                        bytes_identity = identity;
                        sampled_format = byte_format;
                        // The nz-RGB / alpha-zero empty-layer diagnostics only test
                        // presence/absence of colour, which is channel-order-
                        // independent, so they stay valid for a native BGRA8 upload
                        // (B,G,R is the same nonzero set as R,G,B; alpha is last in
                        // both). Run them for either 4-byte layout — losing the
                        // wallpaper/empty-clear proxy on BGRA8 binds would blind the
                        // "background doesn't clear" class. Only the seed-relation
                        // center-row capture below is order-sensitive (RGBA8 only).
                        if matches!(byte_format, TexelLayout::Rgba8 | TexelLayout::Bgra8) {
                            // Fused single pass: `tnz` is load-bearing (bound_tex_rgb,
                            // wallpaper/empty-type3 classification below) and always
                            // computed; the alpha-zero count (opaque-black empty a0=0 vs
                            // transparent empty a0≈w*h, for Load+premult wipe diagnosis)
                            // folds into the same scan instead of a second O(w*h) pass.
                            let (tnz, tmax, a0) = crate::observe::rgba_rgb_a0_stats(&rgba);
                            bound_tex_rgb += tnz;
                            if tnz != 0 {
                                bound_all_bytes_rgb_empty = false;
                            }
                            tex_bind_diag.push(format!(
                                "i{index}:r{texture_ref}:{tw}x{th}:nz={tnz}:max={tmax}:a0={a0}"
                            ));
                            // Wallpaper class: display-sized CPU sample resolved but zero RGB.
                            if tw >= 1280 && th >= 720 && tnz == 0 {
                                let detail =
                                    sample_miss_detail(state, host, req.task_id, texture_ref);
                                let gva_probe =
                                    empty_layer_gva_probe(state, host, req.task_id, texture_ref);
                                crate::observe::off(format!(
                                "m2v_empty_layer pipe={} i={index} ref={texture_ref} {tw}x{th} {detail} {gva_probe}",
                                req.pipeline_ref
                            ));
                                #[cfg(test)]
                                let _proxy_shared =
                                    crate::runtime::census::present_proxy::test_shared();
                                crate::runtime::census::present_proxy::note_empty_sample_if(
                                    texture_ref,
                                    tw,
                                    th,
                                    &rgba,
                                    if frag_stage { "frag_bind" } else { "vert_bind" },
                                );
                            }
                            if tnz == 0 {
                                if let Some(entry) = texture_entry.as_ref() {
                                    if entry.object_type == OBJECT_TYPE_TEXTURE
                                        || entry.object_type == OBJECT_TYPE_TEXTURE_VARIANT
                                    {
                                        had_empty_type3_linear = true;
                                    }
                                }
                            }
                            if byte_format == TexelLayout::Rgba8 && tw == w && th == h {
                                if let (Some((object_type, pixel_format)), Some(row)) = (
                                    linear_identity,
                                    sample_seed_relation::center_row(&rgba, tw, th),
                                ) {
                                    linear_sample_rows.push(sample_seed_relation::SampleRow {
                                        index,
                                        texture_ref,
                                        object_type,
                                        pixel_format,
                                        frag_stage,
                                        rgba: row.to_vec(),
                                    });
                                }
                            }
                        } else {
                            tex_bind_diag.push(format!(
                                "i{index}:r{texture_ref}:{tw}x{th}:native={byte_format:?}"
                            ));
                        }
                        crate::backend::vulkan::engine::SampledSource::Bytes(rgba)
                    }
                    #[cfg(feature = "backend-vulkan")]
                    SampledSourceRequest::Target(identity) => {
                        // A resident bound directly reuses the registry's own
                        // image view, which the engine creates once per target
                        // and cannot re-decorate per bind. Refuse rather than
                        // bind it unswizzled: reading the wrong channels is a
                        // rendering bug that looks like content, whereas a
                        // named decline is one grep away.
                        if view_swizzle.is_some() {
                            crate::runtime::census::view_swizzle_census::note_declined(
                                crate::runtime::census::view_swizzle_census::SwizzleDecline::ResidentDirectBind,
                                texture_ref,
                            );
                            return Ok(());
                        }
                        bound_any_resident_sample = true;
                        tex_bind_diag.push(format!(
                            "i{index}:r{texture_ref}:mid{sampled_mid}:{tw}x{th}:resident_direct"
                        ));
                        crate::backend::vulkan::engine::SampledSource::Target(identity)
                    }
                    #[cfg(feature = "backend-vulkan")]
                    SampledSourceRequest::GuestRuns(src, native) => {
                        sampled_format = native;
                        tex_bind_diag.push(format!(
                            "i{index}:r{texture_ref}:mid{sampled_mid}:{tw}x{th}:zero_copy"
                        ));
                        crate::backend::vulkan::engine::SampledSource::GuestRuns(src)
                    }
                };
                let base_off = if frag_stage && separate_sampled {
                    FRAG_SAMPLED_RESOURCE_BINDING_OFFSET
                } else {
                    0
                };
                let img_bind = TEXTURE_BINDING_BASE + index + base_off;
                // Texture dimensionality comes solely from the translator's reflection,
                // keyed on the UN-relocated descriptor binding. The always-on
                // `census_reflection_wellformed` guard (m2v_cache) proves the reflection
                // is internally consistent per translate. `Absent` is an unused/unbound
                // sampler slot (Metal permits it) — default 2D silently (expected control
                // flow). `Unsupported` is a texture shape reflection carries but the
                // sampled path can't express — log fail-visibly, then keep the 2D default
                // so the draw still paints rather than dropping content.
                use crate::runtime::spirv_bind::{ReflectedSampledKind, SampledImageKind};
                let reflection = if frag_stage {
                    &f_shader.reflection
                } else {
                    &v_shader.reflection
                };
                let image_kind = match crate::runtime::spirv_bind::reflected_sampled_kind(
                    reflection,
                    TEXTURE_BINDING_BASE + index,
                ) {
                    ReflectedSampledKind::Kind(k) => k,
                    ReflectedSampledKind::Absent => SampledImageKind::D2,
                    ReflectedSampledKind::Unsupported => {
                        crate::observe::fail(format!(
                        "reflection_sampled_shape_unsupported stage={} idx={index} ref={texture_ref} binding={img_bind}",
                        if frag_stage { "frag" } else { "vert" }
                    ));
                        SampledImageKind::D2
                    }
                };
                let Some(shape) = sampled_image_shape(image_kind) else {
                    return Err(DrawError::DrawPreparation(
                        DrawPreparationDecline::TextureDimensionUnsupported {
                            stage: if frag_stage { "fragment" } else { "vertex" },
                            index,
                            texture_ref,
                            binding: img_bind,
                            kind: format!("{image_kind:?}"),
                        },
                    ));
                };
                let SampledImageShape {
                    arrayed,
                    volume,
                    cube,
                    one_dim,
                    layers,
                } = shape;
                // A Vulkan 1D image is defined to have height 1; the descriptor
                // may report the LUT's texel count in either axis, so collapse
                // to a single row and fold the other axis into the width the
                // sampled bytes are validated against.
                let (tw, th) = if one_dim {
                    (tw.saturating_mul(th).max(1), 1)
                } else {
                    (tw, th)
                };
                images.push(crate::backend::vulkan::engine::SampledImageResource {
                    binding: img_bind,
                    width: tw,
                    height: th,
                    layers,
                    arrayed,
                    volume,
                    cube,
                    one_dim,
                    source,
                    format: translate::pixel::vk_texel_layout(sampled_format),
                    identity: bytes_identity.map(|i| {
                        crate::backend::vulkan::engine::SampledContentIdentity {
                            key: i.key,
                            generation: i.generation,
                        }
                    }),
                    swizzle: view_swizzle.unwrap_or_default(),
                });
                if view_swizzle.is_some() {
                    crate::runtime::census::view_swizzle_census::note_gpu_mapping();
                }
                crate::runtime::census::setup_tex_census::note_stats(
                    t_stats.elapsed().as_micros() as u64
                );
                Ok(())
            };
            for t in &req.vertex_textures {
                push_tex(t.index, t.texture_ref, false)?;
            }
            for t in &req.fragment_textures {
                push_tex(t.index, t.texture_ref, true)?;
            }
        }
        // Log all multi-bind and display-sized single-tex binds (pipe-26 chrome
        // wipe class is img=1 — previously only n>=4 logged).
        if !tex_bind_diag.is_empty()
            && (images.len() >= 4 || req.fragment_textures.len() >= 4 || w >= 1280)
        {
            crate::observe::line(format!(
                "m2v_tex_binds pipe={} n_img={} n_req={} [{}]",
                req.pipeline_ref,
                images.len(),
                req.fragment_textures.len(),
                tex_bind_diag.join(",")
            ));
        }
        {
            let mut push_smp =
                |index: u32, sampler_ref: u32, frag_stage: bool| -> Result<(), DrawError> {
                    if index >= MAX_BIND_SLOTS {
                        return Ok(());
                    }
                    let base_off = if frag_stage && separate_sampled {
                        FRAG_SAMPLED_RESOURCE_BINDING_OFFSET
                    } else {
                        0
                    };
                    let smp_bind = SAMPLER_BINDING_BASE + index + base_off;
                    if sampler_binds.insert(smp_bind) {
                        let sampler = if sampler_ref != 0 {
                            load_vulkan_sampler(state, host, req.task_id, sampler_ref, smp_bind)
                                .map_err(DrawError::DrawPreparation)?
                        } else {
                            crate::backend::vulkan::engine::SamplerResource::normalized_default(
                                smp_bind,
                            )
                        };
                        samplers.push(sampler);
                    }
                    Ok(())
                };
            // Stream sampler slots (often index 0 while texture is 3 for logo).
            for s in &req.vertex_samplers {
                if s.sampler_ref != 0 {
                    push_smp(s.index, s.sampler_ref, false)?;
                }
            }
            for s in &req.fragment_samplers {
                if s.sampler_ref != 0 {
                    push_smp(s.index, s.sampler_ref, true)?;
                }
            }
        }
        // AIR constexpr samplers carry their immutable state in reflection. Bind
        // those exact values before the residual SPIR-V scan provisions defaults
        // for translator-generated sampler-less read helpers.
        for (reflection, frag_stage) in
            [(&v_shader.reflection, false), (&f_shader.reflection, true)]
        {
            for reflected in &reflection.bindings {
                if reflected.kind != metal2vulkan::reflect::ResourceKind::StaticSampler {
                    continue;
                }
                let Some(descriptor) = reflected.descriptor else {
                    return Err(DrawError::DrawPreparation(
                        DrawPreparationDecline::StaticSamplerReflectionDescriptorMissing {
                            stage: if frag_stage { "fragment" } else { "vertex" },
                        },
                    ));
                };
                let Some(state) = reflected.static_sampler else {
                    return Err(DrawError::DrawPreparation(
                        DrawPreparationDecline::StaticSamplerReflectionStateMissing {
                            stage: if frag_stage { "fragment" } else { "vertex" },
                            binding: descriptor.binding,
                        },
                    ));
                };
                let binding = descriptor.binding
                    + if frag_stage && separate_sampled {
                        FRAG_SAMPLED_RESOURCE_BINDING_OFFSET
                    } else {
                        0
                    };
                if sampler_binds.insert(binding) {
                    let sampler = reflected_static_sampler_resource(
                        if frag_stage { "fragment" } else { "vertex" },
                        binding,
                        state,
                    )
                    .map_err(DrawError::DrawPreparation)?;
                    samplers.push(sampler);
                }
            }
        }
        // Reflect the residual shader interface and provision defaults only
        // where explicit guest or constexpr state did not already win.
        for binding in crate::runtime::spirv_bind::sampler_bindings(&v_words)
            .into_iter()
            .chain(crate::runtime::spirv_bind::sampler_bindings(&f_words))
        {
            if sampler_binds.insert(binding) {
                samplers.push(
                    crate::backend::vulkan::engine::SamplerResource::normalized_default(binding),
                );
            }
        }

        let t_tex_done = std::time::Instant::now();
        // Color load seed: CLEAR → solid; LOAD → guest/host seed when present.
        let mut target_rgba8: Option<Vec<u8>> = None;
        let mut seed_src = "none";
        #[cfg(feature = "backend-vulkan")]
        let gpu_only_content_allowed =
            crate::backend::vulkan::engine::deferred_gpu_only_content_allowed();
        // Records 2+ of a resident render-pass chain load the prior record's
        // content directly from the engine target (no CPU seed, no re-upload).
        #[cfg(feature = "backend-vulkan")]
        let mut chain_load_from_target = false;
        #[cfg(feature = "backend-vulkan")]
        if req.chain_from_resident {
            if let Some(identity) = render_chain_identity(state, req) {
                if crate::backend::vulkan::engine::resident_content_ready(&identity) {
                    chain_load_from_target = true;
                    seed_src = "chain_resident";
                } else {
                    // The armed chain lost its resident (engine reset /
                    // registry eviction). Seeding from stale guest/cache
                    // bytes here would silently wipe the chained records —
                    // fail visibly and let the exec loop abandon the chain.
                    return Err(DrawError::DrawPreparation(
                        DrawPreparationDecline::ChainResidentNotReady {
                            target_gva: req.colors.first().map(|c| c.target_gva).unwrap_or(0),
                            width: w,
                            height: h,
                        },
                    ));
                }
            }
        }
        // Cross-pass resident Load: a deferred GVA Store window at this exact
        // target means the engine resident — not guest/cache bytes — is the
        // authoritative prior content. When this record itself renders into
        // that registry identity, load directly from it (no CPU seed, no
        // flush); any mismatch lands the window first so the seeds below read
        // fresh bytes.
        #[cfg(feature = "backend-vulkan")]
        if !chain_load_from_target && gpu_only_content_allowed {
            if let Some(c0) = req.colors.first() {
                if c0.load_action == PASS_LOAD_ACTION_LOAD
                    && c0.mapping_id == 0
                    && c0.target_gva != 0
                    && c0.target_seed_rgba.is_none()
                    && req.target_seed_rgba.is_none()
                    && state.gva_deferred_flush.contains_key(&c0.target_gva)
                {
                    let entry_geom = state
                        .gva_deferred_flush
                        .get(&c0.target_gva)
                        .map(|e| (e.width, e.height));
                    let will_target_registry = c0.store_action == PASS_STORE_ACTION_STORE
                        && (!writeback_guest || gva_store_defer_eligible(req));
                    let identity = crate::backend::vulkan::engine::TargetIdentity::Gva {
                        gva: c0.target_gva,
                        width: w,
                        height: h,
                        generation: 0,
                    };
                    if will_target_registry
                        && entry_geom == Some((w, h))
                        && crate::backend::vulkan::engine::resident_content_ready(&identity)
                    {
                        chain_load_from_target = true;
                        seed_src = "gva_deferred_resident";
                    } else {
                        crate::runtime::storage_flush::flush_gva_exact(
                            state,
                            host,
                            c0.target_gva,
                            true,
                            "load_seed",
                        );
                    }
                }
            }
        }
        if let Some(c0) = req.colors.first() {
            match c0.load_action {
                #[cfg(feature = "backend-vulkan")]
                x if x == PASS_LOAD_ACTION_LOAD && chain_load_from_target => {
                    // Resident target carries the chain; no CPU seed bytes.
                }
                x if x == PASS_LOAD_ACTION_CLEAR => {
                    target_rgba8 = Some(solid_rgba_local(w, h, &c0.clear_color));
                    seed_src = "clear";
                }
                x if x == PASS_LOAD_ACTION_LOAD => {
                    if let Some(seed) = c0.target_seed_rgba.as_ref() {
                        if seed.len() == (w as usize) * (h as usize) * 4 {
                            // seed_color_load selected this by RT provenance.
                            // Black/transparent bytes are valid attachment data.
                            target_rgba8 = Some(seed.clone());
                            seed_src = "color_seed";
                        }
                    } else if let Some(seed) = req.target_seed_rgba.as_ref() {
                        if seed.len() == (w as usize) * (h as usize) * 4 {
                            target_rgba8 = Some(seed.clone());
                            seed_src = "req_seed";
                        }
                    } else if c0.mapping_id != 0 {
                        if let Some(bgra) =
                            crate::runtime::surface_cache::get(state, c0.mapping_id, w, h)
                        {
                            target_rgba8 = Some(swap_rb_channels(bgra));
                            seed_src = "host_cache";
                        } else {
                            seed_src = "host_cache_miss";
                        }
                    } else if c0.texture_ref != 0 {
                        // GVA type-2/3 Load: texture_ref encode cache (separate
                        // from surface_id mid map). Prefer GVA key when present.
                        let gva_hit = if c0.target_gva != 0 {
                            crate::runtime::surface_cache::get_gva(state, c0.target_gva, w, h)
                                .map(|b| b.to_vec())
                        } else {
                            None
                        };
                        if let Some(bgra) = gva_hit {
                            let mut rgba = bgra;
                            for px in rgba.chunks_exact_mut(4) {
                                px.swap(0, 2);
                            }
                            target_rgba8 = Some(rgba);
                            seed_src = "host_cache_gva";
                        } else if let Some(bgra) =
                            crate::runtime::surface_cache::get_texture(state, c0.texture_ref, w, h)
                        {
                            target_rgba8 = Some(swap_rb_channels(bgra));
                            seed_src = "host_cache_tex";
                        } else {
                            seed_src = "host_cache_tex_miss";
                        }
                    }
                    if w >= 1280 && h >= 720 && crate::observe::draw_log_enabled() {
                        let (snz, smax, spx) = target_rgba8
                            .as_ref()
                            .map(|s| crate::observe::rgba_rgb_stats(s))
                            .unwrap_or((0, 0, [0, 0, 0, 0]));
                        crate::observe::line(format!(
                            "m2v_load_seed mid={} {}x{} pipe={} tex_ref={} src={seed_src} rgb_nz={snz} max_rgb={smax} px0=[{},{},{},{}]",
                            c0.mapping_id,
                            w,
                            h,
                            req.pipeline_ref,
                            c0.texture_ref,
                            spx[0],
                            spx[1],
                            spx[2],
                            spx[3]
                        ));
                    }
                }
                _ => {}
            }
        } else if let Some(seed) = req.target_seed_rgba.as_ref() {
            if seed.len() == (w as usize) * (h as usize) * 4 {
                target_rgba8 = Some(seed.clone());
            }
        }

        let t_seed_done = std::time::Instant::now();
        let mut resources = crate::backend::vulkan::engine::DrawRequest {
            // Metal NDC is Y-up; Vulkan is Y-down.
            flip_viewport_y: true,
            // Honor the guest's face-culling state, its winding, and its
            // primitive type. All three come from `translate::raster`, and all
            // three fall back to a Metal default when the guest bound nothing —
            // but an out-of-contract *value* is a different thing from an unbound
            // one, and it says its own name before falling back. Silently
            // coercing here is how a guest that asked for lines got triangles
            // with nothing in the log to say so.
            cull_mode: raster_or_default(
                req.cull_mode,
                translate::raster::cull_mode,
                crate::backend::vulkan::engine::CullMode::None,
                req.pipeline_ref,
                "cull_mode_unmapped",
            ),
            // MTLWinding: CounterClockwise == 1; Metal defaults to Clockwise.
            front_face_ccw: raster_or_default(
                req.front_facing,
                translate::raster::front_face_ccw,
                false,
                req.pipeline_ref,
                "winding_unmapped",
            ),
            first_vertex: req.first_vertex,
            instance_count: Some(req.instance_count.max(1)),
            primitive_topology: raster_or_default(
                Some(req.primitive_type),
                translate::raster::primitive_topology,
                crate::backend::vulkan::engine::PrimitiveTopology::Triangle,
                req.pipeline_ref,
                "primitive_type_unmapped",
            ),
            ..crate::backend::vulkan::engine::DrawRequest::default()
        };
        if let Some(vp) = req.viewport {
            resources
                .viewports
                .push(crate::backend::vulkan::engine::ViewportResource {
                    x: vp[0] as f32,
                    y: vp[1] as f32,
                    width: vp[2] as f32,
                    height: vp[3] as f32,
                    min_depth: vp[4] as f32,
                    max_depth: vp[5] as f32,
                });
        }
        if let Some((x, y, sw, sh)) = req.scissor {
            resources
                .scissors
                .push(crate::backend::vulkan::engine::ScissorResource {
                    x,
                    y,
                    width: sw,
                    height: sh,
                });
        }
        if let Some(idx) = req.indexed.as_ref() {
            let index_type = translate::raster::index_type(idx.index_type).ok_or_else(|| {
                DrawError::DrawPreparation(DrawPreparationDecline::IndexLoad {
                    reason: IndexLoadReason::TypeUnsupported,
                })
            })?;
            let indices =
                load_index_bytes_reason(state, host, req.task_id, idx).map_err(|reason| {
                    DrawError::DrawPreparation(DrawPreparationDecline::IndexLoad { reason })
                })?;
            resources.indexed = Some(crate::backend::vulkan::engine::IndexedDrawResource {
                index_type,
                index_count: idx.index_count,
                // IndexedDrawInfo does not carry baseVertex yet (Metal path uses 0).
                vertex_offset: 0,
                indices,
            });
        }
        resources.vertex_attributes = attrs;
        resources.storage_buffers = storage;
        resources.sampled_images = images;
        resources.color_input = frag_color_input;
        resources.samplers = samplers;
        let t_asm_prep_done = std::time::Instant::now();
        // Load seed always goes to the GPU (workstream D3). Premult One/OMSA is
        // hardware blend over the Load-seeded target — identical math to the
        // retired software `src + seed*(1-src.a)` path. Sampled alpha is
        // protocol data and must not be rewritten from an RGB content census;
        // content-gated keep-seed / alpha0-holes composites are retired.
        let load_action = req.colors.first().map(|c| c.load_action);
        let premult_load = load_action == Some(PASS_LOAD_ACTION_LOAD)
            && target_rgba8.is_some()
            && color0_is_premult_one_omsa(&pd);
        let store_is_store = req
            .colors
            .first()
            .map(|c| c.store_action == PASS_STORE_ACTION_STORE)
            .unwrap_or(true);
        // Measure-only: how often Store targets guest-visible backings (D4 proxy).
        if store_is_store {
            crate::observe::line(format!(
                "linux_m2v_store_freq pipe={} store=1 (writeback-before-stamp class)",
                req.pipeline_ref
            ));
        }
        resources.target_rgba8 = target_rgba8;
        // Non-Store: skip_readback. Type-11 Store uses resident BGRA plus
        // import-present when stable host aliases and external host memory are
        // available. Portability devices take the synchronous import path;
        // only cross-packet deferred guest authority remains disabled there.
        let import_mid = req.colors.first().map(|c| c.mapping_id).unwrap_or(0);
        let stable_host_import = host.map_pages_stable()
            && crate::backend::vulkan::engine::external_memory_host_available();
        let try_import =
            type11_import_allowed(store_is_store, stable_host_import, import_mid, w, h);
        if try_import {
            let identity =
                crate::runtime::import_present::surface_identity(state, import_mid, w, h);
            resources.target_identity = Some(identity);
            resources.output_bgra = true;
            resources.skip_readback = true;
        } else {
            resources.skip_readback = !store_is_store;
        }
        // Ephemeral resident render-pass rail: intermediate Store records render
        // into a protocol-keyed RGBA target on every Vulkan backend. This does
        // not leave guest-visible content GPU-only: portability devices read the
        // final record back and perform the normal synchronous guest Store.
        // Cross-pass deferred ownership remains gated below.
        #[cfg(feature = "backend-vulkan")]
        let mut resident_render_chain = false;
        // Deferred GVA Store rail: the final/single record also stays on the
        // registry resident (skip_readback) — the caller arms a flush-on-
        // access window instead of the sync readback + guest write on the
        // stamp path (`arm_gva_deferred_store`).
        #[cfg(feature = "backend-vulkan")]
        let mut gva_resident_store = false;
        #[cfg(feature = "backend-vulkan")]
        if !try_import && (req.chain_from_resident || (store_is_store && !writeback_guest)) {
            if let Some(identity) = render_chain_identity(state, req) {
                resources.target_identity = Some(identity);
                if store_is_store && !writeback_guest {
                    resources.skip_readback = true;
                    resident_render_chain = true;
                }
            }
        }
        #[cfg(feature = "backend-vulkan")]
        if !try_import && gpu_only_content_allowed && store_is_store && writeback_guest {
            if let Some(identity) = gva_chain_identity(req) {
                if store_is_store && writeback_guest && gva_store_defer_eligible(req) {
                    resources.target_identity = Some(identity);
                    resources.skip_readback = true;
                    gva_resident_store = true;
                }
            }
        }
        #[cfg(feature = "backend-vulkan")]
        if chain_load_from_target {
            if resources.target_identity.is_none() {
                // chain_from_resident implies a protocol target identity; a
                // miss here is a rail wiring bug, not a content condition.
                return Err(DrawError::DrawPreparation(
                    DrawPreparationDecline::ChainResidentIdentityMissing {
                        target_gva: req.colors.first().map(|c| c.target_gva).unwrap_or(0),
                        width: w,
                        height: h,
                    },
                ));
            }
            resources.load_op = Some(crate::backend::vulkan::engine::LoadOp::LoadFromTarget);
            resources.target_rgba8 = None;
        }
        // Type-11 Load: discrete GPU must LOAD the resident image when ready.
        // Metal unified Load reads guest-backed attachment; after zero-copy we
        // evict host_cache and skip_readback so CPU seed is empty — without
        // LoadFromTarget the engine Clears black and wipes multi-pass layers
        // (class A: progressive res_rgb collapse on sequential Stores).
        if try_import {
            if let Some(ref identity) = resources.target_identity {
                let ready = crate::backend::vulkan::engine::resident_content_ready(identity);
                // Present boundary: the guest's compositor damage-draws each
                // swapchain buffer against the display's CURRENT front frame —
                // the present model itself is the only inter-buffer transfer
                // (boot-27 forensics: no CPU forward-copy, no blit, exactly one
                // full-display pass per transition; a buffer that misses it
                // never converges — dual-mid strobe class). First LOAD draw
                // after the mapping's own CmdDisplaySwap therefore seeds from
                // the retained front frame (+0x188); guest pages are the
                // fallback (they only hold our own flushed resident), then the
                // resident chain.
                // Same-mid present boundary: this mid was itself just presented
                // and the guest is damage-drawing it again (single-buffer case).
                let same_mid_present = load_action == Some(PASS_LOAD_ACTION_LOAD)
                    && state.presented_needs_guest_seed.remove(&import_mid);
                // Inter-buffer retention is STRUCTURAL, not copied. Two
                // surfaces the guest names at one geometry resolve to the same
                // `TargetIdentity::OutputGroup` and therefore share one
                // resident, so the frame one holds is already the frame the
                // other's LoadFromTarget reads. The a/b peer seed that used to
                // sit here copied a full frame between them; every such copy was
                // a copy onto itself and the unification filter dropped it. What
                // survived the filter only ever had a target the guest had NEVER
                // named as plane 0 (`peer_seed_unification outcome=survives
                // src_kind=group target_kind=surface`, the sole outcome recorded
                // across 11 process lifetimes) — a WebKit content tile receiving
                // the desktop's frame at ~50/s, which is unrelated content, not
                // retention.
                let presented_since_last_draw = same_mid_present;
                // If the resident image is unavailable, guest pages are the
                // final protocol-backed source. A successful read is a valid
                // seed even when every RGB component is zero.
                let mut gpu_boundary_seed = false;
                // Self-front elision: the retained front resolves to the same
                // ready resident as this target, including a unified group.
                let self_front_ready = ready
                    && presented_since_last_draw
                    && state.present.frame_valid
                    && state.present.frame_width == w
                    && state.present.frame_height == h
                    && (state.present.frame_mapping == import_mid
                        || crate::runtime::import_present::surface_identity(
                            state,
                            state.present.frame_mapping,
                            w,
                            h,
                        ) == *identity);
                if load_action == Some(PASS_LOAD_ACTION_LOAD)
                    && (!ready || presented_since_last_draw)
                    && !self_front_ready
                    && resources.target_rgba8.is_none()
                    && resources.seed_from_target.is_none()
                    && import_mid != 0
                {
                    // GPU rail first: the presented front frame's own engine
                    // resident, copied resident→target on the GPU — no CPU
                    // front-frame read, no full-frame seed upload. Falls back
                    // to the CPU retain when the front resident is absent,
                    // geometry differs, the front mapping IS this draw's
                    // target (resolver then picks LoadFromTarget/CPU as
                    // before), or the front is bound as a sampled image in
                    // this draw (mid-CB layout conflict).
                    if presented_since_last_draw
                        && state.present.frame_valid
                        && state.present.frame_width == w
                        && state.present.frame_height == h
                        && state.present.frame_mapping != 0
                        && state.present.frame_mapping != import_mid
                    {
                        let front_identity = crate::runtime::import_present::surface_identity(
                            state,
                            state.present.frame_mapping,
                            w,
                            h,
                        );
                        let front_sampled = resources.sampled_images.iter().any(|img| {
                            matches!(
                                &img.source,
                                crate::backend::vulkan::engine::SampledSource::Target(t)
                                    if *t == front_identity
                            )
                        });
                        if !front_sampled
                            && crate::backend::vulkan::engine::resident_content_ready(
                                &front_identity,
                            )
                        {
                            resources.seed_from_target = Some(front_identity);
                            gpu_boundary_seed = true;
                            seed_src = "front_frame_resident_gpu";
                            if w >= 1280 && h >= 720 {
                                crate::observe::fail(format!(
                                    "m2v_load_seed mid={import_mid} {w}x{h} pipe={} tex_ref={} src={seed_src} front_mid={} front_gen={}",
                                    req.pipeline_ref,
                                    req.colors.first().map(|c| c.texture_ref).unwrap_or(0),
                                    state.present.frame_mapping,
                                    state.present.frame_generation
                                ));
                            }
                        }
                    }
                    let front = if presented_since_last_draw && !gpu_boundary_seed {
                        present_front_frame_rgba(&state.present, w, h)
                    } else {
                        None
                    };
                    if gpu_boundary_seed {
                        // GPU rail armed: the whole CPU seed chain (front
                        // retain AND the guest-pages fallback) is elided —
                        // arming a CPU seed alongside is the engine-rejected
                        // double-seed class (190 dropped boundary draws /
                        // black desktop on boot 20260717-031554).
                    } else if let Some(rgba) = front {
                        // The full-frame stats scan exists only for the
                        // verbose line — never pay it on a normal boot.
                        let verbose = crate::observe::draw_log_enabled();
                        let (snz, smax, _) = if verbose {
                            crate::observe::rgba_rgb_stats(&rgba)
                        } else {
                            (0, 0, [0, 0, 0, 0])
                        };
                        resources.target_rgba8 = Some(rgba);
                        seed_src = "front_frame_present_boundary";
                        if verbose && w >= 1280 && h >= 720 {
                            crate::observe::line(format!(
                                "m2v_load_seed mid={import_mid} {w}x{h} pipe={} tex_ref={} src={seed_src} rgb_nz={snz} max_rgb={smax} front_mid={} front_gen={}",
                                req.pipeline_ref,
                                req.colors.first().map(|c| c.texture_ref).unwrap_or(0),
                                state.present.frame_mapping,
                                state.present.frame_generation
                            ));
                        }
                    } else if let Some(rgba) =
                        load_type11_rgba_static(state, host, import_mid, None)
                    {
                        let verbose = crate::observe::draw_log_enabled();
                        let (snz, smax, _) = if verbose {
                            crate::observe::rgba_rgb_stats(&rgba)
                        } else {
                            (0, 0, [0, 0, 0, 0])
                        };
                        resources.target_rgba8 = Some(rgba);
                        seed_src = if presented_since_last_draw {
                            "guest_pages_present_boundary"
                        } else {
                            "guest_pages"
                        };
                        if verbose && w >= 1280 && h >= 720 {
                            crate::observe::line(format!(
                                "m2v_load_seed mid={import_mid} {w}x{h} pipe={} tex_ref={} src={seed_src} rgb_nz={snz} max_rgb={smax}",
                                req.pipeline_ref,
                                req.colors.first().map(|c| c.texture_ref).unwrap_or(0)
                            ));
                        }
                    } else if presented_since_last_draw {
                        crate::observe::fail(format!(
                            "m2v_load_seed mid={import_mid} {w}x{h} pipe={} reason=present_boundary_pages_unreadable fallback=resident ready={}",
                            req.pipeline_ref, ready as u8
                        ));
                    }
                }
                let decision = crate::runtime::metal_draw::resolve_type11_load_decision(
                    load_action.unwrap_or(PASS_LOAD_ACTION_DONT_CARE),
                    ready,
                    resources.target_rgba8.as_deref(),
                    gpu_boundary_seed,
                    presented_since_last_draw,
                );
                // Six checks decide this and three answers come out, so the
                // existing `m2v_load_seed` line — verbose-gated, and only for
                // >=1280x720 — cannot say which one applied. Latched per
                // (check, pipeline, load_action, ready) so the reading is the
                // *set* of pipelines each arm serves; the present-boundary arm
                // is the one derived from forensics rather than a decoded field,
                // and nothing can weigh deleting it while its decisions are
                // indistinguishable from the two legitimate seed paths.
                //
                // `ready` is in the key, not merely in the line. The
                // present-boundary check runs *before* the resident check, so
                // its only distinctive power is overruling a ready resident —
                // with `ready = 0` the fall-through reaches `SeedWhileNotReady`
                // and the same outcome, and the arm decided nothing. Keyed
                // without `ready`, a first sighting at `ready = 0` would hide
                // every later firing at `ready = 1` for the whole process, and
                // "it never overruled a resident" would be unfalsifiable.
                {
                    use crate::observe::Decline;
                    let key = (u64::from(req.pipeline_ref) << 9)
                        | (u64::from(load_action.unwrap_or(0)) << 1)
                        | u64::from(ready);
                    if crate::observe::first_sight(decision.slug(), key) {
                        crate::observe::Emit::decline("t11_load", &decision)
                            .field("pipe", req.pipeline_ref)
                            .field("mid", import_mid)
                            .field("dims", format!("{w}x{h}"))
                            .field("ready", u8::from(ready))
                            .field("presented", u8::from(presented_since_last_draw))
                            .fail();
                    }
                }
                let choice = decision.choice();
                match choice {
                    Type11LoadChoice::LoadFromTarget => {
                        resources.load_op =
                            Some(crate::backend::vulkan::engine::LoadOp::LoadFromTarget);
                        // Avoid double-upload: LoadFromTarget ignores seed, but
                        // dropping it keeps counters/seed_uploads honest.
                        resources.target_rgba8 = None;
                        resources.seed_from_target = None;
                        if w >= 1280 && h >= 720 {
                            crate::observe::line(format!(
                                "m2v_load_seed mid={import_mid} {w}x{h} pipe={} tex_ref={} src=resident_load ready=1 self_front={}",
                                req.pipeline_ref,
                                req.colors.first().map(|c| c.texture_ref).unwrap_or(0),
                                self_front_ready as u8
                            ));
                        }
                    }
                    Type11LoadChoice::UseCpuSeed => {
                        // target_rgba8 already set from host cache, clear, or
                        // guest pages. Presence, not content, is authoritative.
                    }
                    Type11LoadChoice::ClearBlack => {
                        // No resident image and no readable protocol-backed seed.
                        resources.target_rgba8 = None;
                        resources.seed_from_target = None;
                    }
                }
            }
        }
        if load_action == Some(PASS_LOAD_ACTION_LOAD) && import_mid != 0 {
            if let Some(seed) = resources.target_rgba8.as_deref() {
                if let Some(seed_row) = sample_seed_relation::center_row(seed, w, h) {
                    sample_seed_relation::log_exact_relations(
                        req.pipeline_ref,
                        import_mid,
                        req.colors.first().map(|c| c.format).unwrap_or(0),
                        w,
                        h,
                        seed_src,
                        &linear_sample_rows,
                        seed_row,
                    );
                }
                log_black_load_seed_preserved(
                    req.task_id,
                    req.colors.first().map(|c| c.texture_ref).unwrap_or(0),
                    import_mid,
                    req.colors.first().map(|c| c.target_gva).unwrap_or(0),
                    w,
                    h,
                    seed_src,
                    seed,
                );
            } else if matches!(
                resources.load_op,
                Some(crate::backend::vulkan::engine::LoadOp::LoadFromTarget)
            ) && !linear_sample_rows.is_empty()
            {
                let target_format = req.colors.first().map(|c| c.format).unwrap_or(0);
                if let Some(seed_row) = sample_seed_relation::load_type11_center_row_rgba(
                    state,
                    host,
                    import_mid,
                    w,
                    h,
                    target_format,
                ) {
                    sample_seed_relation::log_exact_relations(
                        req.pipeline_ref,
                        import_mid,
                        target_format,
                        w,
                        h,
                        "resident_guest_mirror",
                        &linear_sample_rows,
                        &seed_row,
                    );
                }
            }
        }
        let _premult_load = premult_load; // retained for census logs below
                                          // Metal path always passes color0 blend into the encoder. Linux/engine
                                          // previously left `resources.blend = None` → opaque replace for every
                                          // draw, so Load seeds (gray/wallpaper/logo bases) were wiped by sparse
                                          // dock/chrome layers that Metal would alpha-blend over the attachment.
                                          // Contract: type-7 color attachment blend tags (decode/resource.rs).
        if pd.color0.blending_enabled {
            let constants = req.blend_color.unwrap_or([0.0; 4]);
            match translate::blend::state(
                pd.color0.src_rgb,
                pd.color0.dst_rgb,
                pd.color0.op_rgb,
                pd.color0.src_alpha,
                pd.color0.dst_alpha,
                pd.color0.op_alpha,
                constants,
            ) {
                Ok(b) => {
                    resources.blend = Some(b);
                    if w >= 1280 && h >= 720 {
                        crate::observe::line(format!(
                            "m2v_blend pipe={} src_rgb={} dst_rgb={} op_rgb={} src_a={} dst_a={} op_a={}",
                            req.pipeline_ref,
                            pd.color0.src_rgb,
                            pd.color0.dst_rgb,
                            pd.color0.op_rgb,
                            pd.color0.src_alpha,
                            pd.color0.dst_alpha,
                            pd.color0.op_alpha
                        ));
                    }
                }
                Err(e) => {
                    crate::observe::fail(format!(
                        "m2v_blend_map_fail pipe={} {e}",
                        req.pipeline_ref
                    ));
                }
            }
        }

        let vertex_count = if resources.indexed.is_some() {
            // Ignored by engine when indexed; still pass for validation.
            req.vertex_count.max(1)
        } else {
            req.vertex_count.max(1)
        };
        let output_mapping = req.colors.first().map(|c| c.mapping_id).unwrap_or(0);
        let same_geometry_linear_sample = full_geometry_linear_sample && output_mapping != 0;
        let t_asm_load_done = std::time::Instant::now();
        let mut linear_coverage = if same_geometry_linear_sample {
            draw_full_target_coverage(&resources, w, h, vertex_count)
        } else {
            Default::default()
        };
        let mut linear_coverage_src = "stage_in";
        if same_geometry_linear_sample && !linear_coverage.full() {
            let has_stage_in_position = resources.vertex_attributes.iter().any(|attribute| {
                attribute.location == 0
                    && attribute.step_function
                        == crate::backend::vulkan::engine::VertexStepFunction::PerVertex
                    && matches!(
                        attribute.format,
                        crate::backend::vulkan::engine::VertexAttributeFormat::Float2
                            | crate::backend::vulkan::engine::VertexAttributeFormat::Float3
                            | crate::backend::vulkan::engine::VertexAttributeFormat::Float4
                    )
            });
            if !has_stage_in_position {
                // Gate the shader-pulled coverage proof from reflection (writes
                // Position, reads VertexIndex, binds a buffer) — no SPIR-V walk.
                if crate::runtime::spirv_bind::vertex_position_pull_gate(&v_shader.reflection) {
                    {
                        // No stage-in position: evaluate the translated vertex
                        // SPIR-V against the decoded bound buffers to prove or
                        // reject coverage generically (fail closed).
                        let eval = draw_full_target_coverage_shader_pulled(
                            &resources,
                            w,
                            h,
                            vertex_count,
                            &v_words,
                        );
                        let gap_slug = match eval {
                            Ok(eval_coverage) if eval_coverage.full() => {
                                linear_coverage = eval_coverage;
                                linear_coverage_src = "shader_eval";
                                None
                            }
                            Ok(eval_coverage) if eval_coverage.bounds_span => {
                                Some(ShaderPulledCoverageDecline::TriangleGap)
                            }
                            Ok(_) => Some(ShaderPulledCoverageDecline::PartialBounds),
                            Err(decline) => Some(decline),
                        };
                        if let Some(decline) = gap_slug {
                            log_shader_pulled_coverage_gap(
                                output_mapping,
                                w,
                                h,
                                req.pipeline_ref,
                                vertex_count,
                                resources.indexed.is_some(),
                                &v_shader.reflection,
                                &decline,
                            );
                        }
                    }
                }
            }
        }
        let full_geometry_linear_sample = linear_coverage.full();
        if same_geometry_linear_sample {
            log_compositor_linear_coverage(
                output_mapping,
                w,
                h,
                req.pipeline_ref,
                full_geometry_linear_sample,
                linear_coverage.bounds_span,
                vertex_count,
                resources.indexed.is_some(),
                req.viewport.map(|v| [v[0], v[1], v[2], v[3]]),
                req.scissor,
                linear_coverage_src,
            );
        }
        // Measure-only per-draw geometry census for display-sized type-11
        // targets: decoded location-0 position AABB over the drawn index set,
        // raw (no NDC/pixel disambiguation — the census consumer knows the
        // pipe's vertex space). Answers "which draws ever touch region R of
        // buffer B" for the rect_void burn-in class without pixel inspection.
        let mut full_quad_bounds = false;
        if output_mapping != 0 && w >= 1280 && h >= 720 {
            if let Some((b, n_idx)) = draw_position_bounds(&resources, vertex_count) {
                let pixel_span = b[0] <= 0.0 && b[1] <= 0.0 && b[2] >= w as f32 && b[3] >= h as f32;
                let ndc_span = b[0] <= -1.0 && b[1] <= -1.0 && b[2] >= 1.0 && b[3] >= 1.0;
                full_quad_bounds = pixel_span || ndc_span;
                crate::observe::line(format!(
                    "m2v_draw_bounds mid={output_mapping} {w}x{h} pipe={} idx={n_idx} b=[{:.1},{:.1},{:.1},{:.1}]",
                    req.pipeline_ref, b[0], b[1], b[2], b[3]
                ));
            }
        }
        let t_asm_cov_done = std::time::Instant::now();

        // Decide FIRST whether a census line will be emitted at all; the
        // resource metas below (per-attr/ssbo format!, hex prefixes, 16-float
        // matrix dump) cost real per-draw CPU and were previously computed
        // unconditionally on every draw only to be dropped.
        let census_verbose = crate::observe::draw_log_enabled();
        let fixed_state_gap = vulkan_fixed_state_gap(req);
        let fixed_gap_first = !fixed_state_gap.is_empty() && {
            use std::collections::HashSet;
            use std::sync::Mutex;
            type FixedStateGapKey = (u32, u32, u32, String);
            static SEEN: Mutex<Option<HashSet<FixedStateGapKey>>> = Mutex::new(None);
            let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
            seen.get_or_insert_with(HashSet::new).insert((
                req.pipeline_ref,
                w,
                h,
                fixed_state_gap.clone(),
            ))
        };
        // Honor a bound NON-TRIVIAL depth-stencil state: attach a transient depth
        // buffer + enable the depth test. Decoded once per depth draw; the whole
        // 2D UI binds no depth-stencil (`depth_stencil_ref == 0`, 0 decodes), so
        // this is inert there. A trivial state (compare Always, no write, no
        // stencil) stays `None` — no depth attachment, byte-identical 2D path.
        // Still-unrepresented sub-cases (guest depth LOAD, stencil test,
        // out-of-contract compare) are dropped fail-visibly, deduped per
        // (pipe,slug) so 3D content cannot flood the log.
        if req.depth_stencil_ref != 0 {
            let ds = match load_depth_stencil_descriptor(
                state,
                host,
                req.task_id,
                req.depth_stencil_ref,
            ) {
                Ok(ds) => Some(ds),
                Err(reason) => {
                    // The guest bound a depth-stencil state (`ds_ref != 0`) that we
                    // could not resolve/decode: the draw silently renders with the
                    // depth test DISABLED (wrong occlusion for 3D content). Every
                    // other sub-case below is fail-visible, so name this one too —
                    // deduped per (pipe,reason) so 3D content cannot flood, and inert
                    // on the 2D UI path (which binds no depth-stencil).
                    if degrade_log_first(req.pipeline_ref, reason) {
                        crate::observe::fail(format!(
                            "shader_state_degraded reason={reason} \
                             pipe={} ds_ref={} {}x{} \
                             (bound depth-stencil unresolved; depth test disabled)",
                            req.pipeline_ref, req.depth_stencil_ref, w, h
                        ));
                    }
                    None
                }
            };
            if let Some(ds) = ds {
                if !depth_stencil_descriptor_is_trivial(&ds) {
                    match translate::raster::compare_function(ds.depth_compare_function).ok() {
                        Some(compare) => {
                            let (clear_value, load_action) = req
                                .depth_attach
                                .as_ref()
                                .map(|d| (d.clear_depth as f32, d.load_action))
                                .unwrap_or((1.0, PASS_LOAD_ACTION_CLEAR));
                            // The transient depth buffer supports CLEAR only; a
                            // guest depth LOAD needs a persistent depth resident
                            // (deferred). Degrade to CLEAR, fail-visible.
                            if load_action == PASS_LOAD_ACTION_LOAD
                                && degrade_log_first(
                                    req.pipeline_ref,
                                    "depth_load_unsupported_transient",
                                )
                            {
                                crate::observe::fail(format!(
                                    "shader_state_degraded reason=depth_load_unsupported_transient \
                                     pipe={} ds_ref={} {}x{} \
                                     (transient depth clears; multi-pass depth LOAD not yet resident)",
                                    req.pipeline_ref, req.depth_stencil_ref, w, h
                                ));
                            }
                            // Stencil test: engaged when either face is enabled.
                            // A face that is *not* enabled maps to Metal's
                            // documented `MTLStencilDescriptor` default (compare
                            // Always, all ops Keep, full masks) — a no-op face —
                            // NOT its raw decoded bytes, which for a disabled
                            // face need not be initialized. An out-of-contract
                            // compare/op on an enabled face drops stencil
                            // fail-visibly (unknown wire stays unknown); depth is
                            // still honored.
                            let stencil = if ds.front_stencil_enabled || ds.back_stencil_enabled {
                                use crate::backend::vulkan::engine::{
                                    SamplerCompareFunction, StencilFaceOps, StencilOp, StencilState,
                                };
                                const PASS_THROUGH: StencilFaceOps = StencilFaceOps {
                                    compare: SamplerCompareFunction::Always,
                                    fail_op: StencilOp::Keep,
                                    depth_fail_op: StencilOp::Keep,
                                    pass_op: StencilOp::Keep,
                                    read_mask: 0xFFFF_FFFF,
                                    write_mask: 0xFFFF_FFFF,
                                };
                                let front = if ds.front_stencil_enabled {
                                    engine_stencil_face(&ds.front_face)
                                } else {
                                    Ok(PASS_THROUGH)
                                };
                                let back = if ds.back_stencil_enabled {
                                    engine_stencil_face(&ds.back_face)
                                } else {
                                    Ok(PASS_THROUGH)
                                };
                                // Name the field that failed, not just "a
                                // stencil op somewhere did". `TranslateReason`
                                // carries which enum and which value, so a
                                // guest binding an unknown compare on the back
                                // face reads differently from one binding an
                                // unknown pass op on the front.
                                //
                                // The reason is kept **typed** all the way to
                                // the emitter. It used to be rendered into a
                                // nested `field=reason=… value=…` while the
                                // line's own `reason=` carried the coarse
                                // `stencil_op_unmapped` — so a grep for the
                                // specific check found nothing and a grep for
                                // the coarse one could not say which of the
                                // four stencil fields refused.
                                let stencil_reason: Option<translate::TranslateReason> =
                                    front.as_ref().err().or(back.as_ref().err()).copied();
                                let which_face = if front.is_err() { "front" } else { "back" };
                                match (front, back) {
                                    (Ok(front), Ok(back)) => {
                                        let (reference_front, reference_back) =
                                            req.stencil_ref.unwrap_or((0, 0));
                                        let clear_value = req
                                            .stencil_attach
                                            .as_ref()
                                            .map(|s| s.clear_stencil)
                                            .unwrap_or(0);
                                        Some(StencilState {
                                            front,
                                            back,
                                            reference_front,
                                            reference_back,
                                            clear_value,
                                        })
                                    }
                                    _ => {
                                        // Dedup on the *specific* slug, so an
                                        // unknown compare and an unknown pass op
                                        // on the same pipeline both get a line
                                        // rather than the second being silenced
                                        // as a repeat of the first.
                                        if let Some(reason) = stencil_reason {
                                            if degrade_log_first(req.pipeline_ref, reason.slug()) {
                                                crate::observe::Emit::decline(
                                                    "shader_state_degraded",
                                                    &reason,
                                                )
                                                .field("class", "stencil_op_unmapped")
                                                .field("face", which_face)
                                                .field("pipe", req.pipeline_ref)
                                                .field("ds_ref", req.depth_stencil_ref)
                                                .field("stencil_f", ds.front_stencil_enabled as u8)
                                                .field("stencil_b", ds.back_stencil_enabled as u8)
                                                .field("dims", format!("{w}x{h}"))
                                                .fail();
                                            }
                                        }
                                        None
                                    }
                                }
                            } else {
                                None
                            };
                            resources.depth = Some(crate::backend::vulkan::engine::DepthState {
                                test_enable: true,
                                write_enable: ds.depth_write_enabled,
                                compare,
                                clear_value,
                                // Transient buffer: always CLEAR (see above).
                                load: false,
                                stencil,
                            });
                        }
                        None => {
                            // Unknown wire stays unknown: no depth rather than a
                            // guessed compare direction.
                            if degrade_log_first(req.pipeline_ref, "depth_compare_unmapped") {
                                crate::observe::fail(format!(
                                    "shader_state_ignored reason=depth_compare_unmapped \
                                     pipe={} ds_ref={} compare={} {}x{}",
                                    req.pipeline_ref,
                                    req.depth_stencil_ref,
                                    ds.depth_compare_function,
                                    w,
                                    h
                                ));
                            }
                        }
                    }
                }
            }
        }
        // seed_rgb stays unconditional: the retired-proxy fire counters after
        // the draw (keep_seed_empty_sample / keep_seed_uncovered) consume it,
        // and it only scans when a CPU seed is actually attached.
        let seed_rgb = resources
            .target_rgba8
            .as_ref()
            .map(|s| {
                s.chunks_exact(4)
                    .filter(|p| p[0] | p[1] | p[2] != 0)
                    .count()
            })
            .unwrap_or(0);
        // Sum of per-bind rgb_nz over Bytes sources, accumulated in the bind
        // loop (resident Target binds contribute no CPU bytes, as before).
        if census_verbose || fixed_gap_first {
            let tex_rgb: usize = bound_tex_rgb;
            let idx_count = resources
                .indexed
                .as_ref()
                .map(|i| i.index_count)
                .unwrap_or(0);
            let attr0_len = resources
                .vertex_attributes
                .first()
                .map(|a| a.content.len())
                .unwrap_or(0);
            // Bring-up: bounded resource prefixes for black-frame RE. Fragment
            // buffer(0) is commonly a shader configuration record; keep its bytes
            // explicit instead of hiding the decisive fields after the 16-byte
            // generic SSBO prefix.
            let attr0_hex = resources
                .vertex_attributes
                .first()
                .map(|a| hex_prefix(&a.content.cpu_bytes(), 16))
                .unwrap_or_default();
            let attr_meta: String = resources
                .vertex_attributes
                .iter()
                .map(|a| {
                    format!(
                        "L{}:fmt={:?}:off={}:str={}:sf={:?}:sr={}:n={}",
                        a.location,
                        a.format,
                        a.offset,
                        a.stride,
                        a.step_function,
                        a.step_rate,
                        a.content.len()
                    )
                })
                .collect::<Vec<_>>()
                .join(",");
            let ssbo_meta: String = resources
                .storage_buffers
                .iter()
                .map(|b| {
                    format!(
                        "b{}:n={}:h={}",
                        b.binding,
                        b.content.len(),
                        hex_prefix(&b.content.cpu_bytes(), 16)
                    )
                })
                .collect::<Vec<_>>()
                .join(";");
            let frag0_hex = binding_hex_prefix(&frag_storage, 0, 52);
            // Generated compositor shaders source their six vertices from vertex
            // buffer(1) (48 bytes each) and keep image/blend uniforms in fragment
            // buffer(1). Capture those complete declared prefixes only for the
            // structurally complex multi-image/no-stage-in class.
            let complex_bind_diag =
                resources.vertex_attributes.is_empty() && resources.sampled_images.len() >= 4;
            let vtx1_hex = if complex_bind_diag {
                binding_hex_prefix(&vtx_storage, 1, 6 * 48)
            } else {
                Default::default()
            };
            let frag1_hex = if complex_bind_diag {
                binding_hex_prefix(&frag_storage, 1, 112)
            } else {
                Default::default()
            };
            // AIR reflection names fragment buffer(4) `colorP`; the generated
            // compositor fragment consumes it while applying color opcode 1.
            // Capture the complete 4x4-float-sized prefix so the matrix/vector ABI
            // can be compared against the shader without guessing from 16 bytes.
            let frag4_hex = if complex_bind_diag {
                binding_hex_prefix(&frag_storage, 4, 64)
            } else {
                Default::default()
            };
            let sampler_meta: String = resources
                .samplers
                .iter()
                .map(|s| {
                    format!(
                        "b{}:un={}:min={:?}:mag={:?}:mip={:?}:uvw={:?}/{:?}/{:?}",
                        s.binding,
                        s.unnormalized_coordinates as u8,
                        s.min_filter,
                        s.mag_filter,
                        s.mip_filter,
                        s.address_mode_u,
                        s.address_mode_v,
                        s.address_mode_w
                    )
                })
                .collect::<Vec<_>>()
                .join(";");
            let color_target_meta = color_target_diag(&req.colors);
            // Matrix (binding 2) first 4 cols as floats + first vertex color + indices.
            let mat_f: String = resources
                .storage_buffers
                .iter()
                .find(|b| b.binding == 2)
                .map(|b| {
                    b.content
                        .cpu_bytes()
                        .chunks_exact(4)
                        .take(16)
                        .map(|c| format!("{:.6}", f32::from_le_bytes([c[0], c[1], c[2], c[3]])))
                        .collect::<Vec<_>>()
                        .join(",")
                })
                .unwrap_or_default();
            let v0_color: String = resources
                .vertex_attributes
                .iter()
                .find(|a| a.location == 3)
                .and_then(|a| {
                    let bytes = a.content.cpu_bytes();
                    bytes
                        .get(a.offset as usize..a.offset as usize + 4)
                        .map(|c| format!("{:02x}{:02x}{:02x}{:02x}", c[0], c[1], c[2], c[3]))
                })
                .unwrap_or_default();
            let idx_hex: String = resources
                .indexed
                .as_ref()
                .map(|i| hex_prefix(&i.indices, 16))
                .unwrap_or_default();
            // The full ~1KB per-draw resource census is verbose-gated (REIMS_VGPU_DRAW_LOG →
            // /tmp/reims-vgpu-draw.log). The always-on log keeps ONLY the `fixed_gap` anomaly
            // (decoded fixed-function state the Vulkan request cannot represent), deduped
            // per (pipe, w, h, gap) so recurring depth/stencil shadow draws don't flood —
            // an active compositor otherwise emitted 80k+ of these ~1KB lines per
            // interaction.
            {
                let record = format!(
            "linux_m2v_resources pipe={} {}x{} vtx={} attrs={} ssbo={} img={} smp={} rt_n={} rt=[{}] fixed_gap=[{}] seed={} seed_rgb={} tex_rgb={} idx={} idx_n={} attr0={} a0hex={} v0col={} idxhex={} frag0hex={} vtx1hex={} frag1hex={} frag4hex={} mat2=[{mat_f}] meta=[{}] ssbo=[{}] sampler=[{}]",
            req.pipeline_ref,
            w,
            h,
            vertex_count,
            resources.vertex_attributes.len(),
            resources.storage_buffers.len(),
            resources.sampled_images.len(),
            resources.samplers.len(),
            req.colors.len(),
            color_target_meta,
            fixed_state_gap,
            resources.target_rgba8.is_some() as u8,
            seed_rgb,
            tex_rgb,
            resources.indexed.is_some() as u8,
            idx_count,
            attr0_len,
            attr0_hex,
            v0_color,
            idx_hex,
            frag0_hex,
            vtx1_hex,
            frag1_hex,
            frag4_hex,
            attr_meta,
            ssbo_meta,
            sampler_meta
            );
                if census_verbose {
                    crate::observe::line(record);
                } else {
                    crate::observe::off(record);
                }
            }
        }

        resources.vert_spirv = v_words;
        resources.frag_spirv = f_words;
        resources.width = w;
        resources.height = h;
        resources.vertex_count = vertex_count;
        // True MRT: render every color attachment (slot 1.. as engine secondary
        // residents) instead of dropping the shader's secondary outputs. Gated
        // on a resident primary + resolvable secondaries (empty ⇒ single-RT,
        // byte-identical). Record each GVA-backed mask so the later sampling
        // draw binds the resident directly (see `try_sample_mrt_secondary`).
        #[cfg(feature = "backend-vulkan")]
        if let Some(primary_id) = resources.target_identity.clone() {
            let secs = build_secondary_targets(
                state,
                &req.colors,
                &pd,
                &primary_id,
                w,
                h,
                req.blend_color.unwrap_or([0.0; 4]),
            );
            for sec in &secs {
                if let crate::backend::vulkan::engine::TargetIdentity::Gva {
                    gva,
                    width,
                    height,
                    ..
                } = &sec.identity
                {
                    state.mrt_secondary_gvas.insert(*gva, (*width, *height));
                }
            }
            resources.secondary_targets = secs;
        }
        let setup_us = elapsed_us(t_setup);
        let t_setup_done = std::time::Instant::now();
        let setup_split = SetupSplitUs {
            spv: t_spv_done.duration_since(t_setup).as_micros() as u64,
            bufs: t_bufs_done.duration_since(t_spv_done).as_micros() as u64,
            tex: t_tex_done.duration_since(t_bufs_done).as_micros() as u64,
            seed: t_seed_done.duration_since(t_tex_done).as_micros() as u64,
            assemble: t_setup_done.duration_since(t_seed_done).as_micros() as u64,
            asm_prep: t_asm_prep_done.duration_since(t_seed_done).as_micros() as u64,
            asm_load: t_asm_load_done.duration_since(t_asm_prep_done).as_micros() as u64,
            asm_cov: t_asm_cov_done.duration_since(t_asm_load_done).as_micros() as u64,
            asm_diag: t_setup_done.duration_since(t_asm_cov_done).as_micros() as u64,
        };

        // Draw-batching ceiling census (measure-only, never gates behavior):
        // would this draw have joined the previous draw's command buffer under
        // a narrow same-pass batching rule? Same (identity, geometry, bgra) as
        // the previous draw of this packet; joinable additionally requires the
        // load to fold into the open pass (LoadFromTarget, no CPU/GPU seed),
        // no readback, no MRT secondaries, and not sampling its own target.
        let (batch_same_target, batch_joinable) = {
            use std::hash::{Hash, Hasher};
            let key = resources.target_identity.as_ref().map(|id| {
                let mut hh = std::collections::hash_map::DefaultHasher::new();
                id.hash(&mut hh);
                (hh.finish(), w, h, resources.output_bgra)
            });
            let same = match (&key, &state.last_draw_batch_key) {
                (Some(k), Some(prev)) => k == prev,
                _ => false,
            };
            let joinable =
                same && matches!(
                    resources.load_op,
                    Some(crate::backend::vulkan::engine::LoadOp::LoadFromTarget)
                ) && resources.target_rgba8.is_none()
                    && resources.seed_from_target.is_none()
                    && resources.skip_readback
                    && resources.secondary_targets.is_empty()
                    && !resources.sampled_images.iter().any(|s| {
                        matches!(
                            (&s.source, resources.target_identity.as_ref()),
                            (
                                crate::backend::vulkan::engine::SampledSource::Target(t),
                                Some(own)
                            ) if t == own
                        )
                    });
            state.last_draw_batch_key = key;
            (same as u64, joinable as u64)
        };
        let before = crate::backend::vulkan::engine::counter_snapshot();
        let t_engine = std::time::Instant::now();
        // The engine's own typed `DrawError` (a `vk_*` VkCall slug, a
        // `DrawReason` refusal, an interim `_untyped`) propagates unchanged so
        // the boundary below names the engine's specific check as the primary
        // `reason=` rather than flattening it into a `vk_engine: {e}` blob.
        let out = crate::backend::vulkan::engine::execute_draw_request(&resources)?;
        let engine_us = elapsed_us(t_engine);
        let after = crate::backend::vulkan::engine::counter_snapshot();
        let d = after.delta_since(&before);
        crate::observe::line(format!(
            "linux_m2v_draw vk_engine_creates={} vk_engine_allocs={} pipe_hit={} pipe_miss={} sampled_reuploads={} sampled_reupload_bytes={} sampled_cache_hits={} sampled_cache_hit_bytes={} sampled_identity_hits={} sampled_cache_misses={} sampled_gpu_binds={} sampled_free_hits={} sampled_free_allocs={} sampled_recycle_admits={} sampled_recycle_cap_drops={} engine_us={}",
            d.creates,
            d.allocs,
            d.pipeline_hits,
            d.pipeline_misses,
            d.sampled_reuploads,
            d.sampled_reupload_bytes,
            d.sampled_cache_hits,
            d.sampled_cache_hit_bytes,
            d.sampled_identity_hits,
            d.sampled_cache_misses,
            d.sampled_gpu_binds,
            d.sampled_free_hits,
            d.sampled_free_allocs,
            d.sampled_recycle_admits,
            d.sampled_recycle_cap_drops,
            engine_us
        ));
        // Measure-only: RGB nonzero (ignore alpha) so black+alpha is not mistaken for content.
        // Resident/import path uses skip_readback → empty `out.pixels` is **expected**
        // and must not be read as "GPU drew black" (use import_content res_rgb_nz).
        let mut rgb_nz = 0usize;
        let mut max_rgb = 0u8;
        if out.pixels.is_empty() {
            crate::observe::line(format!(
                "linux_m2v_pixels pipe={} {}x{} skip_readback=1 (no CPU pixels; see import_content)",
                req.pipeline_ref, w, h
            ));
        } else {
            for px in out.pixels.chunks_exact(4) {
                let m = px[0].max(px[1]).max(px[2]);
                if m != 0 {
                    rgb_nz += 1;
                }
                if m > max_rgb {
                    max_rgb = m;
                }
            }
            crate::observe::line(format!(
                "linux_m2v_pixels pipe={} {}x{} rgb_nz={} max_rgb={} px0=[{},{},{},{}]",
                req.pipeline_ref,
                w,
                h,
                rgb_nz,
                max_rgb,
                out.pixels.first().copied().unwrap_or(0),
                out.pixels.get(1).copied().unwrap_or(0),
                out.pixels.get(2).copied().unwrap_or(0),
                out.pixels.get(3).copied().unwrap_or(0),
            ));
        }
        let t_composite = std::time::Instant::now();
        // D3: no content-gated CPU composites. Premult One/OMSA is hardware
        // Load+blend; keep-seed / alpha0-holes retired (real Metal has neither).
        // Fire-counter proxies still log when the OLD content gates *would* have
        // fired so live boots can prove they stay quiet.
        // `a0` (alpha-zero pixel count) is a full-frame scan of the readback
        // pixels but is consumed only by the two composite-fire proxies below —
        // gated on `_premult_load` or the LOAD/seed/max_rgb shape. Compute it
        // only when a consumer will actually fire; otherwise the O(pixels) scan
        // is discarded work on the drain worker (both consumer conditions are
        // a0-independent, so skipping is behavior-identical).
        let need_a0 = !out.pixels.is_empty()
            && (_premult_load
                || (load_action == Some(PASS_LOAD_ACTION_LOAD) && seed_rgb > 0 && max_rgb == 0));
        let a0 = if need_a0 {
            out.pixels.chunks_exact(4).filter(|p| p[3] == 0).count()
        } else {
            0
        };
        // Bind-loop accumulators: a resident Target bind counts as non-empty
        // (its bytes live on the GPU), exactly as the retired per-image
        // re-scan treated `SampledSource::Target`.
        let samples_all_rgb_empty = !resources.sampled_images.is_empty()
            && !bound_any_resident_sample
            && bound_all_bytes_rgb_empty;
        let empty_sample_keep = should_keep_seed_on_empty_draw(
            max_rgb,
            seed_rgb,
            resources.sampled_images.len(),
            samples_all_rgb_empty,
            had_empty_type3_linear,
        );
        if empty_sample_keep {
            crate::observe::fail(format!(
                "linux_m2v_composite_fire keep_seed_empty_sample=1 pipe={} seed_rgb={} n_img={} type3_empty={} (retired; measure-only)",
                req.pipeline_ref,
                seed_rgb,
                resources.sampled_images.len(),
                had_empty_type3_linear as u8
            ));
        }
        if _premult_load {
            crate::observe::fail(format!(
                "linux_m2v_composite_fire premult_hw=1 pipe={} seed_rgb={} draw_rgb={} a0={} (GPU Load+blend; no software premult)",
                req.pipeline_ref, seed_rgb, rgb_nz, a0
            ));
        }
        if load_action == Some(PASS_LOAD_ACTION_LOAD) && seed_rgb > 0 && max_rgb == 0 && a0 > 0 {
            crate::observe::fail(format!(
                "linux_m2v_composite_fire keep_seed_uncovered=1 pipe={} seed_rgb={} a0={} (retired; measure-only)",
                req.pipeline_ref, seed_rgb, a0
            ));
        }
        // Engine pixels are authoritative (empty when skip_readback; Store path
        // materializes bytes for surface_cache / guest writeback unless import).
        let pixels = out.pixels;
        let composite_us = elapsed_us(t_composite);
        let mut tranche_delta = log_linux_m2v_timing(
            M2vTiming {
                pipe: req.pipeline_ref,
                w,
                h,
                total_us: elapsed_us(t_total),
                load_us,
                m2v_us,
                setup_us,
                setup_split,
                engine_us,
                composite_us,
            },
            d,
        );
        tranche_delta.buf_zc = d.buffer_zerocopy_binds;
        tranche_delta.buf_snap = d.buffer_snapshot_binds;
        tranche_delta.batch_same_target = batch_same_target;
        tranche_delta.batch_joinable = batch_joinable;
        tranche_delta.batch_opened = d.batch_opens;
        tranche_delta.batch_joined = d.batch_joins;
        state.tranche.add(tranche_delta);
        if try_import {
            if let Some(identity) = resources.target_identity.clone() {
                return Ok(M2vDrawSpan::ResidentBgra {
                    identity,
                    mapping_id: import_mid,
                    width: w,
                    height: h,
                    display_sample_mids: display_sample_mids.into_iter().collect(),
                    full_geometry_linear_sample,
                    full_quad_bounds,
                });
            }
        }
        #[cfg(feature = "backend-vulkan")]
        if resident_render_chain {
            return Ok(M2vDrawSpan::ResidentChain);
        }
        #[cfg(feature = "backend-vulkan")]
        if gva_resident_store {
            return Ok(M2vDrawSpan::ResidentGvaStore);
        }
        Ok(M2vDrawSpan::Rgba(pixels))
    }
    #[cfg(not(feature = "backend-vulkan"))]
    {
        let _ = (
            v_words, f_words, w, h, pd, state, host, req, t_setup, load_us, m2v_us, t_total,
        );
        // Metal feature build on Linux: translation succeeds; raster needs --backend vulkan.
        Err("vk_engine not linked (rebuild with --backend vulkan)".into())
    }
}

/// Land a multi-draw chain image into guest color targets (full-frame store).
/// Used when a later draw in the packet fails after earlier encodes succeeded.
/// Engine-resident identity for a color0 render-pass chain.
///
/// This identity lives only from the first serialized record through its final
/// Store. Type-11 targets use their current protocol mapping identity; linear
/// type-2/3 targets use the GVA identity below. Unlike deferred writeback, this
/// lifetime is safe on portability-subset devices because the final record
/// materializes guest bytes before the packet completes.
#[cfg(feature = "backend-vulkan")]
fn render_chain_identity(
    state: &DeviceState,
    req: &DrawEncodeRequest,
) -> Option<crate::backend::vulkan::engine::TargetIdentity> {
    let c0 = req.colors.first()?;
    let (width, height) = if req.width > 0 && req.height > 0 {
        (req.width, req.height)
    } else {
        (c0.width, c0.height)
    };
    if width == 0 || height == 0 {
        return None;
    }
    if c0.mapping_id != 0 {
        return Some(crate::runtime::import_present::surface_identity(
            state,
            c0.mapping_id,
            width,
            height,
        ));
    }
    gva_chain_identity(req)
}

/// Engine-resident identity for a GVA (type-2/3) color0 render target.
///
/// Single source of truth for the resident GVA chain rail: the draw path,
/// the alias-sample bind, and the abandon-path landing must all agree on the
/// exact identity or the registry lookups miss. Generation is a constant 0 —
/// chain records never consult the resident across passes (record 1 always
/// re-seeds from protocol state), so no cross-pass freshness key is needed.
#[cfg(feature = "backend-vulkan")]
pub(crate) fn gva_chain_identity(
    req: &DrawEncodeRequest,
) -> Option<crate::backend::vulkan::engine::TargetIdentity> {
    let c0 = req.colors.first()?;
    if c0.mapping_id != 0 || c0.target_gva == 0 {
        return None;
    }
    let (w, h) = if req.width > 0 && req.height > 0 {
        (req.width, req.height)
    } else {
        (c0.width, c0.height)
    };
    if w == 0 || h == 0 {
        return None;
    }
    Some(crate::backend::vulkan::engine::TargetIdentity::Gva {
        gva: c0.target_gva,
        width: w,
        height: h,
        generation: 0,
    })
}

/// Read an abandoned resident render-pass chain so the exec loop can land the
/// last good record's pixels (`writeback_chain_rgba`). Every failure is
/// fail-visible; the guest keeps its pre-pass bytes on loss.
#[cfg(feature = "backend-vulkan")]
pub(crate) fn read_resident_chain(state: &DeviceState, req: &DrawEncodeRequest) -> Option<Vec<u8>> {
    let identity = render_chain_identity(state, req)?;
    match crate::backend::vulkan::engine::read_target(&identity) {
        Ok(rgba) => Some(rgba),
        Err(e) => {
            crate::observe::fail(format!(
                "chain_resident_land_fail reason=read_target target={identity:?} \
                 mid={} gva={:#x} {}x{} err={e}",
                req.colors.first().map(|c| c.mapping_id).unwrap_or(0),
                req.colors.first().map(|c| c.target_gva).unwrap_or(0),
                identity.width(),
                identity.height()
            ));
            None
        }
    }
}

/// Deferred GVA windows keep engine registry slots pinned (the LRU sweep
/// skips pinned slots and soft-exceeds `REGISTRY_CAP=64`); arming past this
/// count lands the oldest window first so pinned pressure stays bounded.
#[cfg(feature = "backend-vulkan")]
const GVA_DEFERRED_WINDOW_CAP: usize = 16;

/// Defer gate for the final/single record of a GVA render Store: the record
/// may keep its pixels on the engine registry resident and land guest bytes
/// on access (`storage_flush::flush_gva_one`) instead of a sync readback +
/// fence wait on the stamp path. All gates are protocol-shape checks (never
/// content): the flush must be able to replay the sync `write_gva_rgba8`
/// exactly — identity geometry == c0 geometry, convertible format, sane BPR.
#[cfg(feature = "backend-vulkan")]
fn gva_store_defer_eligible(req: &DrawEncodeRequest) -> bool {
    let Some(c0) = req.colors.first() else {
        return false;
    };
    if c0.mapping_id != 0 || c0.target_gva == 0 || c0.row_stride == 0 {
        return false;
    }
    let Some(identity) = gva_chain_identity(req) else {
        return false;
    };
    if identity.width() != c0.width || identity.height() != c0.height {
        return false;
    }
    pixel_format::tight_row_bytes(c0.width, c0.format).is_some_and(|t| c0.row_stride >= t)
}

/// Any host-side writer of the guest window at `gva` supersedes the deferred
/// Store window there: a later flush of the old window would clobber the
/// strictly-newer bytes. Same geometry drops the obligation (the new write
/// fully covers the window; its bytes were never observable without a flush,
/// which would have taken it); different geometry lands the old identity
/// first, preserving the sync serialization (old bytes, then new bytes).
#[cfg(feature = "backend-vulkan")]
pub(crate) fn supersede_gva_window<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    gva: u64,
    width: u32,
    height: u32,
    by: &str,
) {
    let Some(old) = state.gva_deferred_flush.get(&gva) else {
        return;
    };
    if old.width == width && old.height == height {
        let old_identity = crate::backend::vulkan::engine::TargetIdentity::Gva {
            gva,
            width,
            height,
            generation: 0,
        };
        let _ = state.take_gva_deferred_window(gva);
        crate::backend::vulkan::engine::unpin_resident_target(&old_identity);
        crate::observe::line(format!(
            "gva_deferred_superseded gva={gva:#x} {width}x{height} by={by}"
        ));
    } else {
        crate::runtime::storage_flush::flush_gva_exact(state, host, gva, true, by);
    }
}

/// Metal-direct builds never arm GVA windows — nothing to supersede.
#[cfg(not(feature = "backend-vulkan"))]
pub(crate) fn supersede_gva_window<M: HostMemory + HostOps>(
    _state: &mut DeviceState,
    _host: &mut M,
    _gva: u64,
    _width: u32,
    _height: u32,
    _by: &str,
) {
}

/// Arm the deferred-writeback window for a GVA render Store that just
/// executed into the registry resident (`M2vDrawSpan::ResidentGvaStore`).
///
/// Returns `false` when a gate fails (unwalkable span, pin refusal) — the
/// caller then lands the Store synchronously from a resident readback, and
/// the sync site's supersede handling covers any older window at this GVA.
#[cfg(feature = "backend-vulkan")]
fn arm_gva_deferred_store<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    req: &DrawEncodeRequest,
) -> bool {
    let Some(identity) = gva_chain_identity(req) else {
        return false;
    };
    let Some(c0) = req.colors.first() else {
        return false;
    };
    if !crate::backend::vulkan::engine::deferred_gpu_only_content_allowed() {
        return false;
    }
    let gva = c0.target_gva;
    let span = (c0.row_stride as u64).saturating_mul(c0.height as u64);
    // Defer-time physical page index: raw task-GVA reads aliasing these pages
    // flush first (`storage_flush::flush_intersecting_task_gva`). A span that
    // does not fully walk cannot be guarded — Store synchronously.
    let mut pages: std::collections::HashSet<u64> = std::collections::HashSet::new();
    crate::runtime::gva_mem::visit_task_gva_page_gpas(
        host,
        &state.tasks,
        req.task_id,
        gva,
        span,
        state.page_shift,
        1,
        &mut |gpa_page| {
            pages.insert(gpa_page);
            true
        },
    );
    let expect_pages = ((gva % state.page_size()) + span).div_ceil(state.page_size());
    if (pages.len() as u64) < expect_pages {
        return false;
    }
    // Supersede any previous window at this GVA before pinning: same
    // geometry means the same identity this draw just re-rendered (drop —
    // the helper's unpin is undone by the pin below); different geometry is
    // a distinct identity whose resident is intact (land it first).
    supersede_gva_window(state, host, gva, c0.width, c0.height, "rearm");
    if !crate::backend::vulkan::engine::pin_resident_target(&identity) {
        return false;
    }
    while state.gva_deferred_flush.len() >= GVA_DEFERRED_WINDOW_CAP {
        let Some((old_gva, old_entry)) = state.take_oldest_gva_deferred_window() else {
            break;
        };
        let _ = crate::runtime::storage_flush::flush_gva_one(
            state,
            host,
            old_gva,
            &old_entry,
            true,
            "window_cap",
        );
    }
    let producer_object_type = objects::lookup_list_entry(state, host, req.task_id, c0.texture_ref)
        .map(|entry| entry.object_type)
        .unwrap_or(0);
    // Stale encodes must not serve while the resident is authoritative —
    // host-path consumers flush first; anything else misses (fail-safe).
    crate::runtime::surface_cache::evict_gva(state, gva);
    if c0.texture_ref != 0 {
        crate::runtime::surface_cache::evict_texture(state, c0.texture_ref);
    }
    state.gva_deferred_seq = state.gva_deferred_seq.wrapping_add(1);
    let armed_seq = state.gva_deferred_seq;
    state.arm_gva_deferred_window(
        gva,
        crate::model::GvaDeferredEntry {
            task_id: req.task_id,
            texture_ref: c0.texture_ref,
            producer_object_type,
            width: c0.width,
            height: c0.height,
            row_stride: c0.row_stride,
            format: c0.format,
            armed_seq,
            pages,
        },
    );
    crate::observe::line(format!(
        "gva_writeback_deferred gva={gva:#x} {}x{} fmt={:#x} pipe={} windows={}",
        c0.width,
        c0.height,
        c0.format,
        req.pipeline_ref,
        state.gva_deferred_flush.len()
    ));
    true
}

pub fn writeback_chain_rgba<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    color_slots: &[(u32, crate::runtime::decode::render::ColorAttachment)],
    rgba: &[u8],
) -> bool {
    if color_slots.is_empty() || rgba.is_empty() {
        return false;
    }
    let Some((_, att)) = color_slots.first() else {
        return false;
    };
    if att.texture_ref == 0 {
        return false;
    }
    let Some((mapping_id, gva, w, h, bpr, fmt)) =
        lookup_render_target(state, host, task_id, att.texture_ref)
    else {
        return false;
    };
    let need = (w as usize).saturating_mul(h as usize).saturating_mul(4);
    if rgba.len() < need {
        return false;
    }
    if gva != 0 {
        supersede_gva_window(state, host, gva, w, h, "chain_land");
        return write_gva_rgba8(state, host, task_id, gva, w, h, bpr, fmt, rgba).is_ok();
    }
    if mapping_id == 0 {
        return false;
    }
    // An abandoned portability chain must still preserve the last successful
    // record. This is an error recovery rail, not normal product behavior: land
    // the resident readback into the type-11 mapping, publish the Composite
    // Store, and keep the degradation fail-visible.
    crate::observe::fail(format!(
        "writeback_chain_rgba reason=resident_chain_abandoned_cpu_recovery \
         mid={mapping_id} {w}x{h} fmt={fmt:#x}"
    ));
    let wrote = mapping_write::write_rgba8_image_changed(state, host, mapping_id, rgba, None, w, h);
    if wrote {
        publish_cpu_portability_store(state, host, mapping_id, w, h, fmt);
    }
    wrote
}

#[cfg(all(test, feature = "backend-vulkan"))]
mod vulkan_split_tests {
    use super::*;
    use crate::model::{DeviceId, PAGE_SHIFT_X86};
    use crate::runtime::host::FakeHost;

    #[test]
    fn m2v_draw_runtime_failure_returns_a_typed_decline() {
        use crate::observe::Decline as _;

        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        let req = DrawEncodeRequest {
            pipeline_ref: 41,
            ..DrawEncodeRequest::default()
        };

        let err = match try_metal2vulkan_draw(&mut state, &mut host, &req, true) {
            Err(err) => err,
            Ok(_) => panic!("an empty state cannot resolve pipeline 41"),
        };
        assert_eq!(err.slug(), "draw_prepare_pipeline_missing");
        assert_eq!(
            linux_m2v_draw_failure(&err, &req).render(),
            "linux_m2v_draw reason=draw_prepare_pipeline_missing \
             task_id=0 pipeline_ref=41 pipe=41 task=0 geom=0x0 vtx=0 inst=0 \
             prim=0 first=0 idx=0 colors=[] vbuf=[] fbuf=[] vtex=[] ftex=[] \
             viewport=None scissor=None"
        );
    }

    #[test]
    fn sampled_image_shape_maps_one_dimensional_luts() {
        use crate::runtime::spirv_bind::SampledImageKind;

        // A color-transfer LUT reflects as `texture1d` / `texture1d_array`.
        // Before this mapping the sampled path declined the whole draw with
        // `draw_prepare_texture_dimension_unsupported`, so the color-managed
        // desktop composite stored nothing and presented unbacked.
        let d1 = sampled_image_shape(SampledImageKind::D1).expect("D1 is expressible");
        assert!(d1.one_dim && !d1.arrayed && !d1.volume && !d1.cube);
        assert_eq!(d1.layers, 1);

        let d1_array =
            sampled_image_shape(SampledImageKind::D1Array).expect("D1Array is expressible");
        assert!(d1_array.one_dim && d1_array.arrayed && !d1_array.volume && !d1_array.cube);
        assert_eq!(d1_array.layers, 1);
    }

    #[test]
    fn sampled_image_shape_keeps_two_dimensional_shapes_flat() {
        use crate::runtime::spirv_bind::SampledImageKind;

        for kind in [
            SampledImageKind::D2,
            SampledImageKind::D2Array,
            SampledImageKind::D3,
        ] {
            let shape = sampled_image_shape(kind).expect("2D/3D shapes stay expressible");
            assert!(!shape.one_dim, "{kind:?} must not be a 1D image");
        }
    }

    #[test]
    fn sampled_image_shape_declines_cube_shapes_by_name() {
        use crate::runtime::spirv_bind::SampledImageKind;

        // Cube sampling is not expressed on the sampled-draw path yet; the
        // shape stays `None` so the caller declines it visibly rather than
        // binding a 2D view under a cube sampler.
        assert!(sampled_image_shape(SampledImageKind::Cube).is_none());
        assert!(sampled_image_shape(SampledImageKind::CubeArray).is_none());
    }
}
