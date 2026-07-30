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
    // Inert on this arm, and by construction rather than by omission: the Metal
    // arm consults it in `store_seed_policy` to suppress a scissor-local store,
    // and this rail has no scissor-local store to suppress — `req.scissor` only
    // ever reaches the pipeline scissor rect, never the Store extent.
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
                    note_type11_store_route("gva_deferred");
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
                    note_type11_store_route("gva_deferred_sync");
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

    // A resident render-pass chain intermediate: the exec loop reads
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

    // A type-11 composite Store reaches the guest only through the CPU writeback
    // below. The DMA rail that used to short-circuit it here — a resident BGRA
    // target landed straight in the mapping's guest pages through an imported
    // host pointer — is gone, because a pointer the GPU can read is one it can
    // write and those pages are guest RAM.
    //
    // Taken, not borrowed. Every exit from this block returns the frame, and
    // borrowing forced each of them to `rgba.clone()` a whole framebuffer — 8 MB
    // at 1080p, at the 28-111 Stores/s `store_routes` measures, on the drain
    // worker `drain_duty` shows at duty 0.93-0.99. The deferred type-11 arm is
    // the hot one and it cloned purely to hand back the buffer it already owned.
    if let Some(rgba) = draw_rgba.take() {
        // Intermediate multi-draw GVA records: return color0 for chaining without
        // guest Store (archive store plan). Resident type-11 intermediates
        // returned above without materializing CPU pixels.
        if !writeback_guest {
            return (EncodeStatus::Ok, Some(rgba));
        }
        // Store draw result into primary color RT.
        if let Some(c0) = colors.first() {
            // `rgb_nz`/`max_rgb` are diagnostic fields of the Store lines below,
            // and producing them is an O(w*h) pass over a whole framebuffer
            // readback — 2 073 600 pixels per Store at 1080p, at the 28-111
            // Stores/s `store_routes` measures under load. Computing it here
            // paid that on every route, including the type-11 one whose only
            // consumer is a `observe::line` a normal boot discards. Each arm
            // now scans only when it is about to write a line.
            let rgb_stats = || {
                let (nz, max, _) = crate::observe::rgba_rgb_stats(&rgba);
                (nz, max)
            };
            let ok = if c0.mapping_id != 0 {
                // Unconditional. This used to be `if
                // type11_cpu_store_fallback_allowed(import_allowed)`, where
                // `import_allowed` asked whether the device could import a host
                // pointer over the mapping's guest pages; when it could, the
                // draw took the import rail and landing here was a fail-closed
                // error (`rgba_not_import`) that preserved the zero-copy
                // invariant. There is no invariant left to preserve, and the
                // else arm was a refusal for a rail that cannot be chosen.
                {
                    // Deferred writeback: publish the frame to `surface_cache`
                    // — the source every other consumer already reads, so the
                    // Load seed and the present capture see exactly what they
                    // would have — and arm a window instead of scattering the
                    // frame into the mapping's guest pages now.
                    //
                    // That scatter is the cost. `write_rgba8_image_changed`
                    // converts every row RGBA→native and then copies it out,
                    // per row when the mapping's pages are fragmented: ~8 MB of
                    // CPU conversion and copy per Store, at the 28-111 Stores/s
                    // `store_routes` measures, on the drain worker `drain_duty`
                    // shows at duty 0.93-0.99. Nothing on the host-window
                    // present path reads those pages, so most of that work is
                    // owed to a guest reader that may never come.
                    let deferred = deferred_gpu_only_content_allowed_for_surface()
                        && arm_surface_deferred_store_with(
                            state,
                            host,
                            req,
                            c0.mapping_id,
                            c0.width,
                            c0.height,
                            &rgba,
                        );
                    if deferred {
                        note_type11_store_route("surface_deferred");
                        publish_surface_store(
                            state,
                            host,
                            c0.mapping_id,
                            c0.width,
                            c0.height,
                            c0.format,
                        );
                        return (EncodeStatus::Ok, Some(rgba));
                    }
                    note_type11_store_route("cpu_portability");
                    let ok = mapping_write::write_rgba8_image_changed(
                        state,
                        host,
                        c0.mapping_id,
                        &rgba,
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
                        publish_surface_store(
                            state,
                            host,
                            c0.mapping_id,
                            c0.width,
                            c0.height,
                            c0.format,
                        );
                        if crate::observe::draw_log_enabled() {
                            let (rgb_nz, max_rgb) = rgb_stats();
                            crate::observe::line(format!(
                                "linux_m2v_store mid={} {}x{} pipe={} import=0 reason=cpu_portability pages=1 rgb_nz={} max={}",
                                c0.mapping_id,
                                c0.width,
                                c0.height,
                                req.pipeline_ref,
                                rgb_nz,
                                max_rgb
                            ));
                        }
                    } else {
                        let (rgb_nz, max_rgb) = rgb_stats();
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
                    &rgba,
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
                        c0.texture_ref,
                        producer_object_type,
                        c0.target_gva,
                        c0.width,
                        c0.height,
                        &rgba,
                    );
                }
                let (rgb_nz, max_rgb) = rgb_stats();
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
                let (rgb_nz, max_rgb) = rgb_stats();
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
                return (EncodeStatus::Ok, Some(rgba));
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
        let resolved = objects::resolve_type11_ref(state, host, task_id, texture_ref);
        if let Some(mid) = resolved {
            surface_candidates.push(mid);
        }
    }
    surface_candidates.sort_unstable();
    surface_candidates.dedup();

    for mid in surface_candidates {
        // Ensure type-4 pages exist for this surface id.
        let _ = objects::ensure_surface_for_present(state, host, mid);
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
                // Resident-surface identity: computed once and reused for both the
                // readiness check and the direct bind. `surface_identity` locks a
                // global dedup mutex and does an output-group lookup; this bind
                // resolves the same (mid, w, h), so recomputing it per resident
                // sample (the census shows ~29k/session) is pure waste.
                #[cfg(feature = "backend-vulkan")]
                let resident_id =
                    crate::runtime::present_identity::surface_identity(state, mid, w, h);
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
                    {
                        return Some((w, h, mid, SampledSourceRequest::Target(resident_id)));
                    }
                }

                // 1) Host cache. Taken unconditionally: a ready resident already
                // returned above, so this line is reached only with
                // `resident_ready == false` and there is nothing for the cache
                // bytes to lose to.
                //
                // What stood here was `let (nz,_,_) = rgba_rgb_stats(&rgba); if
                // nz > 0 || !resident_ready`, i.e. an O(w*h) count of non-black
                // pixels — 2 073 600 per bind at 1080p, on a path the census
                // measures at ~29k resident samples a session — feeding a
                // decision about *which image gets bound*. Two things were wrong
                // with it and they compound: `runtime/census/README.md` forbids
                // exactly this ("a proxy that changes behaviour has stopped
                // being a proxy and become a content heuristic"), and an
                // all-black frame is a legal frame, so the test mistook a
                // correct black surface for an empty one.
                //
                // The disjunct also could not change the outcome. `!resident_ready`
                // is true here on both backends — under `backend-vulkan` because
                // the `if resident_ready` above returns, and under `backend-metal`
                // because `resident_ready` is bound to `false` outright — so the
                // condition was already unconditionally true and the scan's
                // result was discarded. The identical gate on the guest-pages
                // branch below had already been worked out and removed for this
                // reason; its comment says so. This one kept paying for the scan.
                if let Some(bgra) = crate::runtime::surface_cache::get(state, mid, w, h) {
                    return Some((
                        w,
                        h,
                        mid,
                        SampledSourceRequest::Bytes(
                            std::sync::Arc::new(swap_rb_channels(bgra)),
                            None,
                            TexelLayout::Rgba8,
                        ),
                    ));
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
                    match try_type11_sample_zero_copy(state, host, mid, w, h) {
                        Ok(src) => return Some((w, h, mid, src)),
                        Err(reason) => t11_zc_decline = reason,
                    }
                }
                // This path is reached only with `resident_ready == false` (a
                // ready resident binds its target and returns above),
                // so the guest bytes are taken unconditionally — the historical
                // `nz > 0 || !resident_ready` promotion gate is always true here.
                // The memo skips the convert/alloc on unchanged content and
                // returns a content identity so the engine skips re-hash+upload;
                // its census (T11Memo hit / T11Guest fill) is emitted internally.
                if let Some((rgba, identity)) = load_type11_rgba_memoized(state, host, mid) {
                    t11_decline::note(t11_zc_decline, rgba.len());
                    return Some((
                        w,
                        h,
                        mid,
                        SampledSourceRequest::Bytes(rgba, Some(identity), TexelLayout::Rgba8),
                    ));
                }

                {
                    // A sample that resolved to no bytes anywhere is a lost
                    // guest command at any geometry: an app-window layer paints
                    // blank exactly as a full-screen one does. Latched per
                    // (mid, geometry) so a steady repeat stays at one line.
                    use std::collections::HashSet;
                    use std::sync::Mutex;
                    static SEEN: Mutex<Option<HashSet<(u32, u32, u32)>>> = Mutex::new(None);
                    let mut guard = SEEN.lock().unwrap_or_else(|e| e.into_inner());
                    if guard.get_or_insert_with(HashSet::new).insert((mid, w, h)) {
                        crate::observe::fail(format!(
                            "sample_src=miss ref={texture_ref} mid={mid} {w}x{h} resident_ready={} (no guest/cache/resident bytes)",
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
            return Some((
                w,
                h,
                0,
                SampledSourceRequest::Bytes(std::sync::Arc::new(rgba), None, TexelLayout::Rgba8),
            ));
        }
    }

    // Linear / view path returns only RGBA; the geometry comes from the decoded
    // texture descriptor and from nowhere else. A payload shorter than the
    // descriptor's own `width * height * 4` is not a geometry this call may
    // invent one for: the caller turns `None` into a typed
    // `DrawPreparationDecline::TextureResolveMissing`, which names the ref and
    // the stage.
    let mut rgba = load_sampled_rgba_static(state, host, task_id, texture_ref)?;
    let entry = objects::lookup_list_entry(state, host, task_id, texture_ref)?;
    let desc = objects::read_descriptor(state, host, task_id, &entry)?;
    let td = decode_texture_descriptor(&desc).ok()?;
    let w = td.width.max(1);
    let h = td.height.max(1);
    let need = (w as usize).saturating_mul(h as usize).saturating_mul(4);
    if rgba.len() < need {
        return None;
    }
    rgba.truncate(need);
    Some((
        w,
        h,
        0,
        SampledSourceRequest::Bytes(std::sync::Arc::new(rgba), None, TexelLayout::Rgba8),
    ))
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

/// Why a type-11 attachment `LOAD` could not be seeded with the surface's own
/// prior contents.
///
/// This is not a degradation the caller absorbs. `exec.rs` resolves the pass load
/// action as "explicit `load_op` > `target_rgba8` > **Clear**", so a seed of
/// `None` makes `PassKey::single(load = false)` and the render pass begins with
/// `LoadOp::CLEAR` against the hardcoded `[0,0,0,0]` primary clear value. The
/// guest asked for its surface to be preserved and got a transparent-black wipe,
/// and the matching Store then reads that wipe back and publishes it. On a
/// compositor doing a damage-rect redraw that is one whole layer rendering solid
/// black — the reported black-rectangle class, whose screenshots show sharp
/// axis-aligned rectangles at layer boundaries.
///
/// It had no report of any kind. `surface_cache::get_shared` returns `Option` and
/// the arm simply left `target_rgba8` unset, so the loss was invisible on the
/// always-on channel. Measured on one x86/Vulkan boot before the guest-pages rung
/// existed: **121 distinct (mapping, geometry) wipes** in ~170 s, four of them at
/// the full 1920x1080 composite extent, against 0 in the idle phase.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Type11SeedDecline {
    /// The cache holds no entry for this mapping id and the mapping's own pages
    /// could not be read at the requested extent either.
    ///
    /// This is the whole population the pre-fix boot measured: every one of the
    /// 121 lines carried `hostgen=0`, and every one had `want == mapgeom`, which
    /// is what said the guest pages were readable and made the fallback rung the
    /// fix rather than a guess.
    CacheAbsent,
    /// An entry exists but at a different geometry, so the exact-geometry hit
    /// rule refuses it. `host_surfaces` keeps exactly one entry per mapping and
    /// every Store replaces it, so a Store at another geometry orphans every
    /// window still living at this one.
    ///
    /// Fired **0** times on that boot. Kept because it is a different check with
    /// a different fix (the entry is stale, not missing), and folding it into
    /// `CacheAbsent` would hide which one a future boot hit.
    CacheGeom { have_w: u32, have_h: u32 },
}

impl crate::observe::Decline for Type11SeedDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::CacheAbsent => "type11_seed_cache_absent",
            Self::CacheGeom { .. } => "type11_seed_cache_geom",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::CacheAbsent => Vec::new(),
            Self::CacheGeom { have_w, have_h } => vec![("have", format!("{have_w}x{have_h}"))],
        }
    }
}

/// Which rung of the type-11 `LOAD` seed ladder produced the attachment's prior
/// contents.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Type11SeedRung {
    /// The host render cache held this mapping at exactly this geometry.
    Cache,
    /// The cache missed and the surface's own guest IOSurface pages were read.
    GuestPages,
}

impl Type11SeedRung {
    fn name(self) -> &'static str {
        match self {
            Self::Cache => "cache_hit",
            Self::GuestPages => "guest_pages",
        }
    }
}

/// Report which way the type-11 `LOAD` seed branch went, once per
/// `(mapping, requested geometry, outcome)`.
///
/// Every outcome reports, because a zero on the miss arm has to be readable. A
/// probe that only fires on failure cannot separate "the cache always hit" from
/// "this branch never ran", and the branch is reached only for `mapping_id != 0`
/// under `PASS_LOAD_ACTION_LOAD` with no caller-supplied seed. With the served
/// arms beside it, an absent miss line next to present hit lines is evidence
/// rather than silence.
///
/// Naming the *rung* rather than just hit/miss is what prices the fallback: a
/// `guest_pages` line is a cache miss that was recovered, and its rate is the
/// only thing that says whether the recovery is cheap. Fusing it into `cache_hit`
/// would make the fix unmeasurable the moment it worked.
///
/// The mapping's own latched geometry and generation ride along on every arm:
/// `want == mapgeom` is the condition under which the guest-pages rung can serve
/// at all, so the pair says whether a miss was recoverable.
fn note_type11_load_seed(
    state: &DeviceState,
    mapping_id: u32,
    w: u32,
    h: u32,
    served: Option<Type11SeedRung>,
) {
    let (map_w, map_h, map_gen) = state
        .mappings
        .get(&mapping_id)
        .map(|m| (m.width, m.height, m.map_generation))
        .unwrap_or((0, 0, 0));
    let cached = state.host_surfaces.get(&mapping_id);
    let have = cached.map(|e| (e.width, e.height));
    let host_gen = cached.map(|e| e.host_gen).unwrap_or(0);
    // Latch before building the line: `Emit::field` renders eagerly, and this
    // sits on a branch the census measures at 28-111 entries a second.
    let outcome_bits = match served {
        None => 0u64,
        Some(Type11SeedRung::Cache) => 1,
        Some(Type11SeedRung::GuestPages) => 2,
    };
    let disc =
        (u64::from(mapping_id) << 40) | (u64::from(w) << 20) | u64::from(h) | (outcome_bits << 62);
    if let Some(rung) = served {
        if !crate::observe::first_sight("type11_load_seed_served", disc) {
            return;
        }
        crate::observe::off(format!(
            "type11_load_seed outcome={} mid={mapping_id} want={w}x{h} \
             mapgeom={map_w}x{map_h} mapgen={map_gen} hostgen={host_gen}",
            rung.name()
        ));
        return;
    }
    let d = match have {
        Some((have_w, have_h)) => Type11SeedDecline::CacheGeom { have_w, have_h },
        None => Type11SeedDecline::CacheAbsent,
    };
    if !crate::observe::first_sight(crate::observe::Decline::slug(&d), disc) {
        return;
    }
    crate::observe::Emit::decline("type11_load_seed", &d)
        .field("mid", mapping_id)
        .field("want", format!("{w}x{h}"))
        .field("mapgeom", format!("{map_w}x{map_h}"))
        .field("mapgen", map_gen)
        .field("hostgen", host_gen)
        .fail();
}

/// The prior contents of a type-11 attachment under `PASS_LOAD_ACTION_LOAD`,
/// with the byte order they are in.
///
/// Two rungs, in freshness order:
///
/// 1. **The host render cache.** The hot one: `store_routes` measures 28-111 of
///    these a second under a browser workload. It holds guest scanout order and
///    the pooled target is RGBA, so the buffer is handed over behind an `Arc` and
///    the R/B exchange rides the engine's single copy into mapped staging rather
///    than materializing a converted frame here.
/// 2. **The surface's own guest IOSurface pages.** The cache is an accelerator,
///    not the surface. What a type-11 attachment *contains* is its pages, so a
///    cache miss is a reason to read them — not a reason to drop the guest's
///    LOAD. Without this rung the pass began with `LoadOp::CLEAR` against the
///    hardcoded `[0,0,0,0]` primary clear and the matching Store published that
///    wipe, which is a whole compositing layer going solid black.
///
/// `load_type11_rgba_static` reads at the mapping's own latched geometry and
/// converts to RGBA8, so the length check is what confirms the pass wanted that
/// extent — the engine rejects a seed of any other length, and the decline this
/// falls through to carries both geometries so a mismatch is diagnosable rather
/// than silent. `paint_mapping` underneath it lands every intersecting deferred
/// window first, so the read observes our own not-yet-written-back Stores rather
/// than pre-Store bytes.
///
/// The sibling Metal path already had rung 2: type-11 `seed_color_load` falls
/// through to the same reader via `load_sampled_rgba_static`. Only the Vulkan arm
/// stopped at the cache.
///
/// `None` means the guest's LOAD could not be honoured at all, and
/// [`note_type11_load_seed`] has already said which check refused.
fn resolve_type11_load_seed<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mapping_id: u32,
    w: u32,
    h: u32,
) -> Option<(
    std::sync::Arc<Vec<u8>>,
    crate::backend::vulkan::engine::SeedOrder,
)> {
    use crate::backend::vulkan::engine::SeedOrder;
    let served =
        if let Some(bgra) = crate::runtime::surface_cache::get_shared(state, mapping_id, w, h) {
            Some((bgra, SeedOrder::Bgra8, Type11SeedRung::Cache))
        } else {
            load_type11_rgba_static(state, host, mapping_id, None)
                .filter(|rgba| rgba.len() == (w as usize) * (h as usize) * 4)
                .map(|rgba| {
                    (
                        std::sync::Arc::new(rgba),
                        SeedOrder::Rgba8,
                        Type11SeedRung::GuestPages,
                    )
                })
        };
    note_type11_load_seed(state, mapping_id, w, h, served.as_ref().map(|s| s.2));
    served.map(|(bytes, order, _)| (bytes, order))
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
/// per-frame video composites.
#[cfg(feature = "backend-vulkan")]
const ZERO_COPY_SAMPLED_MIN_BYTES: u64 = 64 * 1024;

/// Zero-copy floor for draw-time vertex/storage buffer binds: below this the
/// CPU staging read is cheaper than a page walk plus a recorded GPU gather.
/// Performance threshold only — never a correctness gate.
#[cfg(feature = "backend-vulkan")]
const ZERO_COPY_BUFFER_MIN_BYTES: u64 = 16 * 1024;

/// Does this host promise a guest-page alias that stays valid indefinitely?
///
/// Every guest-run producer below needs that promise, and needs it for a reason
/// that survived the removal of the host-pointer import: the runs are memoized
/// in `DeviceState::guest_run_memo` and reused by *later* draws, so a pointer
/// with a bounded lifetime would be read after its view was released.
///
/// A `false` is expected control flow — the caller falls through to the CPU
/// byte loader and the guest gets correct pixels — so it is not a decline. But
/// it is answered by the host once and then forever, and the whole rail
/// disappearing is not something a reader should have to infer from an absence,
/// so the first refusal of the process says so by name.
///
/// This is where the arm64 pathway now diverges: its MMIO shim can return a
/// `mach_vm_remap` view for a fragmented page list, and since that view is
/// released on `unmap_pages` rather than retained until teardown, the shim
/// answers 0. The x86 PCI shim never allocates — it refuses anything that is
/// not a packed host-contiguous run — so it still answers 1.
#[cfg(feature = "backend-vulkan")]
fn guest_run_alias_available<M: HostOps>(host: &M) -> bool {
    if host.map_pages_stable() {
        return true;
    }
    static NOTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !NOTED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        crate::observe::fail(String::from(
            "guest_run_rail off reason=host_page_alias_not_stable \
             (draw binds take the CPU byte loader)",
        ));
    }
    false
}

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
    if !guest_run_alias_available(host) {
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
    if !guest_run_alias_available(host) {
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
/// the buffer zero-copy floor and every page walkable into mappable runs.
/// Deferred stores intersecting the span are landed first, exactly like the
/// CPU path.
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
    if !guest_run_alias_available(host) {
        return None;
    }
    // Same coherence rule as the CPU read: land any resident-authoritative
    // writeback aliasing the span before the GPU reads the pages (the CPU
    // flush completes before this draw's submit).
    crate::runtime::storage_flush::flush_intersecting_task_gva(
        state,
        host,
        task_id,
        gva + offset,
        span,
    );
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
    let backing = resolve_buffer_backing(state, host, task_id, buffer_ref)?;
    if allow_zero_copy {
        if let Some(content) = try_buffer_zero_copy_resolved(state, host, task_id, &backing, offset)
        {
            return Some(content);
        }
    }
    let bytes = read_buffer_bytes_resolved(state, host, task_id, &backing, offset)?;
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
/// walkable, and packed-contiguous runs mappable.
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
    if !guest_run_alias_available(host) {
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
            type11_sample_window(m, mid, w, h, format).ok_or(Reason::NoWindow)?;
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
    if !guest_run_alias_available(host) {
        return Err(Reason::UnstableMap);
    }
    // Land any resident-authoritative deferred window before the GPU reads
    // the pages (same coherence rule as paint_mapping / the linear rail).
    {
        let _ = crate::runtime::storage_flush::flush_intersecting(state, host, mid, 0, u64::MAX);
    };
    let gpas = { mapper::mapping_page_gpas(state, host, mid) }.ok_or(Reason::Coverage)?;
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
        let base =
            { host.map_pages(&window[i..j], page as usize) }.ok_or(Reason::ImportFail)? as u64;
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
    let row_length_texels = if bpr == tight {
        0
    } else {
        u32::try_from(bpr / RGBA8_BPP as u64)
            .ok()
            .ok_or(Reason::Stride)?
    };
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
    if !guest_run_alias_available(host) {
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
    let row_length_texels = if bpr == tight {
        0
    } else {
        u32::try_from(bpr / bpp as u64).ok()?
    };
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
    crate::runtime::storage_flush::flush_intersecting_task_gva(state, host, task_id, gva, span);
    let mut scratch = std::mem::take(&mut state.guest_linear_scratch);
    scratch.resize(native_len, 0);
    let read = gva_mem::read_task_gva_by_id(
        host,
        &state.tasks,
        task_id,
        gva,
        &mut scratch,
        state.page_shift,
    );
    if read.is_err() {
        state.guest_linear_scratch = scratch;
        return None;
    }
    let key = (task_id, gva, w, h, sample_fmt);
    let hit = state
        .guest_linear_memo
        .get_touch(&key)
        // Vec equality is length + byte memcmp with early exit on change.
        .filter(|m| m.native == scratch)
        .map(|m| (m.rgba.clone(), m.generation, m.bgra8));
    if let Some((rgba, generation, bgra8)) = hit {
        let fmt = if bgra8 {
            TexelLayout::Bgra8
        } else {
            TexelLayout::Rgba8
        };
        state.guest_linear_scratch = scratch;
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
    state.guest_linear_gen += 1;
    let generation = GUEST_LINEAR_GEN_BASE + state.guest_linear_gen;
    let rgba = std::sync::Arc::new(rgba);
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
    if let Some((bgra, host_gen, producer_type)) =
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
                return Some((w, h, rgba, identity, TexelLayout::Rgba8));
            }
            let rgba = swap_rb_channels(bgra);
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
    }
    if let Some(bgra) = crate::runtime::surface_cache::get_texture(state, texture_ref, w, h) {
        let rgba = swap_rb_channels(bgra);
        return Some((w, h, std::sync::Arc::new(rgba), None, TexelLayout::Rgba8));
    }
    // Guest-CPU-produced linear textures (wallpaper, glyph atlases) have no
    // host producer generation. Re-read the native rows and byte-compare
    // against the memo: unchanged content reuses the retained swizzled Arc
    // and carries a generation identity so the engine skips hash+memcmp too.
    if let Some((rgba, identity, byte_format)) =
        load_linear_guest_memoized(state, host, task_id, tex, gva, w, h)
    {
        return Some((w, h, rgba, identity, byte_format));
    }
    let Some((rgba, byte_format)) =
        load_linear_texture_native_host(state, host, task_id, texture_ref, 0, None)
    else {
        crate::observe::fail(format!(
            "linear_sample_miss reason=guest_load task={task_id} ref={texture_ref} type={} gva={gva:#x} fmt={:#x} {w}x{h} bpr={}",
            entry.object_type, tex.pixel_format, layout.row_stride
        ));
        return None;
    };
    let need = (w as usize).saturating_mul(h as usize).saturating_mul(4);
    if rgba.len() >= need {
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
    crate::observe::fail(format!(
        "linear_sample_miss reason=short_rgba task={task_id} ref={texture_ref} type={} gva={gva:#x} fmt={:#x} {w}x{h} bpr={} got={} need={need}",
        entry.object_type,
        tex.pixel_format,
        layout.row_stride,
        rgba.len()
    ));
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

/// Store type-2/3 encode into texture_ref + GVA host caches (BGRA).
#[allow(
    clippy::too_many_arguments,
    reason = "the cache identity mirrors the object, GVA, and texture geometry"
)]
pub(crate) fn host_cache_store_gva_layer(
    state: &mut DeviceState,
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

/// Result of a Linux metal2vulkan draw.
#[cfg(feature = "backend-vulkan")]
enum M2vDrawSpan {
    /// No drawable color0 geom.
    None,
    /// CPU-side RGBA8 pixels (readback path).
    Rgba(Vec<u8>),
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

/// Name the guest-Store route this record actually took, once per distinct
/// route per process.
///
/// Every other record of these branches is `observe::line`, which writes
/// nothing unless `REIMS_VGPU_DRAW_LOG=1` — the only always-on `linux_m2v_store`
/// arm is the CPU write *failure*. So an always-on log could not tell "the CPU
/// Store ran" from "no Store happened at all", which is the branch-vs-arm hole:
/// a probe placed inside one of these arms cannot separate "the condition was
/// false" from "the outcome never occurred". This line names the branch itself,
/// at the point the branch is taken, so a zero for one route is readable
/// against the other routes' presence.
///
/// The dedup key is the route, so this is bounded at one line per outcome per
/// process — after the first record of each kind it costs a `BTreeSet` lookup
/// and a return, which is what makes it safe to leave on permanently.
///
/// Reachability is not uniform and must be read that way. `import` requires the
/// engine to have enabled a host-pointer import, which
/// [`crate::backend::vulkan::engine::external_memory_host_available`] now always
/// refuses, and `rgba_not_import` is its complement's complement — with
/// `import_allowed` always false, `type11_cpu_store_fallback_allowed` is always
/// true and that arm cannot be entered. Both are kept as call sites so their
/// absence is a *denominator* against the routes that do fire, not an
/// acquittal; if either ever appears, the extension came back.
/// The first-appearance line answers "is this route reachable" and cannot answer
/// "how often". Both questions are live: reachability is what the denominator
/// argument above needs, and the rate is what prices the route — `engine_delta`
/// shows ~20 full-frame readbacks a second and the routes are what attribute
/// them. So the dedup'd line stays and the rate is counted alongside it, into
/// the same one-second window as `drain_duty`.
#[cfg(feature = "backend-vulkan")]
fn note_type11_store_route(route: &'static str) {
    use std::sync::Mutex;
    static SEEN: Mutex<Option<std::collections::BTreeSet<&'static str>>> = Mutex::new(None);
    crate::runtime::drain::note_store_route(route);
    {
        let mut guard = SEEN.lock().unwrap_or_else(|p| p.into_inner());
        if !guard.get_or_insert_with(Default::default).insert(route) {
            return;
        }
    }
    crate::observe::fail(format!("type11_store_route route={route}"));
}

/// Advance the guest-visible publish milestones for a type-11 Store whose
/// pixels have landed in the mapping's guest pages.
///
/// Route-independent: the synchronous `cpu_portability` Store calls it inline,
/// and the deferred render rail calls it from the flush that finally performs
/// the same write (`storage_flush::flush_render_one`). Both have just proved
/// the same thing — `write_rgba8_image_changed` verified geometry and landed a
/// complete frame — and without it the `present_unbacked` gate is structurally
/// dead on whichever route skips it, because no mapping's `dense_frame_seq`
/// would advance.
pub(crate) fn publish_surface_store<M: HostMemory + HostOps>(
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
            crate::runtime::present_identity::surface_identity(
                state,
                c.mapping_id,
                c.width,
                c.height,
            )
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
        // The mask is read from the same entry but *not* through the
        // `blending_enabled` filter below: `MTLColorWriteMask` applies whether
        // or not the slot blends, and an entry with no mask means `all`.
        let color_write_mask = pipeline
            .color_attachments
            .iter()
            .find(|a| a.slot == c.slot)
            .map(|a| a.write_mask)
            .unwrap_or_default();
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
                    Ok(state) => Some(state),
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
            color_write_mask,
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
    let pd = load_render_pipeline(state, host, req.task_id, req.pipeline_ref).ok_or({
        DrawError::DrawPreparation(
            crate::backend::vulkan::engine::DrawPreparationDecline::PipelineMissing {
                task_id: req.task_id,
                pipeline_ref: req.pipeline_ref,
            },
        )
    })?;
    let v_mtlb = load_mtlb(state, host, req.task_id, pd.vertex_func_ref).ok_or({
        DrawError::DrawPreparation(
            crate::backend::vulkan::engine::DrawPreparationDecline::VertexMtlbMissing {
                task_id: req.task_id,
                function_ref: pd.vertex_func_ref,
            },
        )
    })?;
    let f_mtlb = load_mtlb(state, host, req.task_id, pd.fragment_func_ref).ok_or({
        DrawError::DrawPreparation(
            crate::backend::vulkan::engine::DrawPreparationDecline::FragmentMtlbMissing {
                task_id: req.task_id,
                function_ref: pd.fragment_func_ref,
            },
        )
    })?;
    // Borrowed from the `*_mtlb` locals above, which outlive every use below.
    // These were `.to_vec()`, which allocated and copied both AIR blobs on
    // every chain — `drain_duty` measures ~1142 chains/s, so that is ~2300
    // allocations a second on the drain worker for bytes that are only ever
    // read (`translate_cached_reflected` takes `&[u8]`, and its cache is keyed
    // by hashing them).
    let v_air = crate::runtime::mtlb::extract_air(&v_mtlb).map_err(|reason| {
        DrawError::DrawPreparation(
            crate::backend::vulkan::engine::DrawPreparationDecline::VertexAirExtract {
                function_ref: pd.vertex_func_ref,
                reason,
            },
        )
    })?;
    let f_air = crate::runtime::mtlb::extract_air(&f_mtlb).map_err(|reason| {
        DrawError::DrawPreparation(
            crate::backend::vulkan::engine::DrawPreparationDecline::FragmentAirExtract {
                function_ref: pd.fragment_func_ref,
                reason,
            },
        )
    })?;

    // AIR→SPIR-V is content-cached: live boots re-translated the same pipelines
    // dozens of times on the doorbell vCPU and tripped IPI timeout panics.
    // Reflected translate: the cached shader carries the metal2vulkan reflection
    // facade so per-draw texture provisioning reads dimensionality straight from
    // the AIR-derived metadata (single source of truth) rather than re-walking the
    // emitted SPIR-V. `_shader.reflection` is used at the sampled-image binding
    // loop below; the SPIR-V walk stays as a cold fallback.
    let v_shader = crate::runtime::m2v_cache::translate_cached_reflected(
        v_air,
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
        f_air,
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

    dump_draw_handoff(
        req,
        pd.vertex_func_ref,
        pd.fragment_func_ref,
        [
            ("vertex", &v_mtlb, v_air, &v_shader.spirv),
            ("fragment", &f_mtlb, f_air, &f_shader.spirv),
        ],
    );

    let (w, h) = if req.width > 0 && req.height > 0 {
        (req.width, req.height)
    } else if let Some(c0) = req.colors.first() {
        (c0.width, c0.height)
    } else {
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
                        AttachmentAliasSample::Seed(seed) => (
                            aw,
                            ah,
                            0,
                            SampledSourceRequest::Bytes(
                                std::sync::Arc::new(seed.to_vec()),
                                None,
                                TexelLayout::Rgba8,
                            ),
                        ),
                        #[cfg(feature = "backend-vulkan")]
                        AttachmentAliasSample::ResidentChain => {
                            let identity = render_chain_identity(state, req).ok_or({
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
                if sampled_mid != 0 && tw == w && th == h {
                    display_sample_mids.insert(sampled_mid);
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
                        crate::backend::vulkan::engine::SampledSource::Target(identity)
                    }
                    #[cfg(feature = "backend-vulkan")]
                    SampledSourceRequest::GuestRuns(src, native) => {
                        sampled_format = native;
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
                Ok(())
            };
            for t in &req.vertex_textures {
                push_tex(t.index, t.texture_ref, false)?;
            }
            for t in &req.fragment_textures {
                push_tex(t.index, t.texture_ref, true)?;
            }
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
        // Color load seed: CLEAR → solid; LOAD → guest/host seed when present.
        // `seed_order` names what is in those bytes; the engine folds any needed
        // R/B exchange into its copy into the mapped staging span rather than
        // making this side materialize a converted frame.
        let mut target_rgba8: Option<std::sync::Arc<Vec<u8>>> = None;
        let mut seed_order = crate::backend::vulkan::engine::SeedOrder::Rgba8;
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
                    target_rgba8 =
                        Some(std::sync::Arc::new(solid_rgba_local(w, h, &c0.clear_color)));
                }
                x if x == PASS_LOAD_ACTION_LOAD => {
                    if let Some(seed) = c0.target_seed_rgba.as_ref() {
                        if seed.len() == (w as usize) * (h as usize) * 4 {
                            // seed_color_load selected this by RT provenance.
                            // Black/transparent bytes are valid attachment data.
                            target_rgba8 = Some(std::sync::Arc::new(seed.clone()));
                        }
                    } else if let Some(seed) = req.target_seed_rgba.as_ref() {
                        if seed.len() == (w as usize) * (h as usize) * 4 {
                            target_rgba8 = Some(std::sync::Arc::new(seed.clone()));
                        }
                    } else if c0.mapping_id != 0 {
                        if let Some((bytes, order)) =
                            resolve_type11_load_seed(state, host, c0.mapping_id, w, h)
                        {
                            target_rgba8 = Some(bytes);
                            seed_order = order;
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
                            target_rgba8 = Some(std::sync::Arc::new(bgra));
                            seed_order = crate::backend::vulkan::engine::SeedOrder::Bgra8;
                        } else if let Some(bgra) =
                            crate::runtime::surface_cache::get_texture(state, c0.texture_ref, w, h)
                        {
                            target_rgba8 = Some(std::sync::Arc::new(bgra.to_vec()));
                            seed_order = crate::backend::vulkan::engine::SeedOrder::Bgra8;
                        }
                    }
                }
                _ => {}
            }
        } else if let Some(seed) = req.target_seed_rgba.as_ref() {
            if seed.len() == (w as usize) * (h as usize) * 4 {
                target_rgba8 = Some(std::sync::Arc::new(seed.clone()));
            }
        }
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
            let index_type = translate::raster::index_type(idx.index_type).ok_or({
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
        // Load seed always goes to the GPU (workstream D3). Premult One/OMSA is
        // hardware blend over the Load-seeded target — identical math to the
        // retired software `src + seed*(1-src.a)` path. Sampled alpha is
        // protocol data and must not be rewritten from an RGB content census;
        // content-gated keep-seed / alpha0-holes composites are retired.
        let store_is_store = req
            .colors
            .first()
            .map(|c| c.store_action == PASS_STORE_ACTION_STORE)
            .unwrap_or(true);
        resources.target_rgba8 = target_rgba8;
        resources.target_seed_order = seed_order;
        // A Store reads back; anything else skips it.
        //
        // A Store used to have a second option: when the host's page aliases
        // were stable *and* the device could import a host pointer over them,
        // it rendered into a BGRA resident with `skip_readback` and the
        // import-present rail DMA'd that resident into the guest's pages. The
        // import is gone, so the only way a Store's pixels reach the guest is
        // the CPU writeback, and that needs them read back.
        resources.skip_readback = !store_is_store;
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
        if req.chain_from_resident || (store_is_store && !writeback_guest) {
            if let Some(identity) = render_chain_identity(state, req) {
                resources.target_identity = Some(identity);
                if store_is_store && !writeback_guest {
                    resources.skip_readback = true;
                    resident_render_chain = true;
                }
            }
        }
        #[cfg(feature = "backend-vulkan")]
        if gpu_only_content_allowed && store_is_store && writeback_guest {
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
        // Type-11 Load used to have a GPU rail here. When the Store was going
        // to land by import, the attachment was a BGRA resident with no CPU
        // seed, so a `LoadFromTarget` had to resolve which resident image held
        // the frame the guest's compositor computes its damage against — the
        // presented front's own resident, this target's, or the guest pages —
        // and copy it resident-to-target on the GPU. Without that resolve the
        // engine would Clear black and wipe the multi-pass layers.
        //
        // A Store now always reads back and always seeds from guest pages, so
        // there is no resident-only attachment to reseed and nothing to
        // resolve: the ~170 lines of front-frame retention policy that stood
        // here were reachable only under `try_import`.
        // Metal path always passes color0 blend into the encoder. Linux/engine
        // previously left `resources.blend = None` → opaque replace for every
        // draw, so Load seeds (gray/wallpaper/logo bases) were wiped by sparse
        // dock/chrome layers that Metal would alpha-blend over the attachment.
        // Contract: type-7 color attachment blend tags (decode/resource.rs).
        // Outside the `blending_enabled` guard below, and deliberately: an
        // unblended attachment with a mask still leaves its unwritten channels
        // alone, so gating the mask on blending would drop it exactly where the
        // guest is replacing rather than compositing.
        resources.color_write_mask = pd.color0.write_mask;
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
        // Sum of per-bind rgb_nz over Bytes sources, accumulated in the bind
        // loop (resident Target binds contribute no CPU bytes, as before).
        if census_verbose || fixed_gap_first {
            // O(seed pixels), and the line below is its only reader.
            let seed_rgb = resources
                .target_rgba8
                .as_ref()
                .map(|s| {
                    s.chunks_exact(4)
                        .filter(|p| p[0] | p[1] | p[2] != 0)
                        .count()
                })
                .unwrap_or(0);
            let tex_rgb: usize = 0;
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
        // The engine's own typed `DrawError` (a `vk_*` VkCall slug, a
        // `DrawReason` refusal, an interim `_untyped`) propagates unchanged so
        // the boundary below names the engine's specific check as the primary
        // `reason=` rather than flattening it into a `vk_engine: {e}` blob.
        let out = crate::backend::vulkan::engine::execute_draw_request(&resources)?;
        // RGB nonzero (ignore alpha) so black+alpha is not mistaken for content.
        // Resident/import path uses skip_readback → empty `out.pixels` is **expected**
        // and must not be read as "GPU drew black" (use import_content res_rgb_nz).
        // The scan is O(pixels) on the drain worker and the line it feeds is the
        // only consumer, so it runs only when that sink is open.
        if census_verbose {
            if out.pixels.is_empty() {
                crate::observe::line(format!(
                    "linux_m2v_pixels pipe={} {}x{} skip_readback=1 (no CPU pixels; see import_content)",
                    req.pipeline_ref, w, h
                ));
            } else {
                let mut rgb_nz = 0usize;
                let mut max_rgb = 0u8;
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
        }
        // No content-gated CPU composites: premultiplied One/OneMinusSourceAlpha
        // is hardware Load+blend, and keep-seed / alpha0-hole compositing is not
        // something real Metal does. The blend state below is what makes that
        // true; a draw that lands wrong shows up as a typed decline on this
        // boundary, not as a pixel census.
        // Engine pixels are authoritative (empty when skip_readback; the Store
        // path materializes bytes for surface_cache and the guest writeback).
        //
        // A Store used to be able to return `M2vDrawSpan::ResidentBgra`
        // instead — no pixels at all, the resident staying authoritative until
        // the import-present rail DMA'd it into the mapping's guest pages.
        // That span is unreachable without the import and its variant is gone.
        let pixels = out.pixels;
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
        return Some(crate::runtime::present_identity::surface_identity(
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

/// Bound on live type-11 render windows. Each one pins a display-sized target
/// resident, so an unbounded population is the "~260 stale residents
/// (~516 MiB) pinned for the guest lifetime" shape. Sized like the GVA cap: a
/// composite touches a handful of layers, so this is headroom, not a working
/// set the guest routinely exceeds.
#[cfg(feature = "backend-vulkan")]
const SURFACE_DEFERRED_WINDOW_CAP: usize = 16;

/// Defer gate for a type-11 (surface) render Store.
///
/// All gates are protocol-shape checks, never content: the later flush has to
/// be able to replay the synchronous Store *exactly*, so anything the sync
/// route would have needed must be resolvable now. In particular the mapping's
/// plane window must resolve, because the deferred window's guest byte range —
/// which is what every reader intersects against to decide whether to flush —
/// comes from it. A window with no range would be armed and then never found
/// by a reader, which is the silent-stale-read failure this rail exists to
/// avoid.
#[cfg(feature = "backend-vulkan")]
fn surface_store_defer_eligible(
    state: &DeviceState,
    req: &DrawEncodeRequest,
) -> Option<crate::model::ComputeStorageResidencyKey> {
    let c0 = req.colors.first()?;
    if c0.mapping_id == 0 {
        return None;
    }
    if !crate::backend::vulkan::engine::deferred_gpu_only_content_allowed() {
        return None;
    }
    let (w, h) = if req.width > 0 && req.height > 0 {
        (req.width, req.height)
    } else {
        (c0.width, c0.height)
    };
    if w == 0 || h == 0 || w != c0.width || h != c0.height {
        return None;
    }
    let m = state.mappings.get(&c0.mapping_id)?;
    // The sync route calls `write_rgba8_image_changed`, which refuses unless
    // the mapping's latched geometry equals the draw's. Deferring a Store that
    // is going to be refused just moves the refusal somewhere it reads as a
    // lost flush, so gate on the same thing up front.
    let (surface_offset, surface_bpr, span_end) =
        crate::runtime::mapping_write::type11_sample_window(m, c0.mapping_id, w, h, c0.format)?;
    Some(crate::model::ComputeStorageResidencyKey {
        mapping_id: c0.mapping_id,
        map_generation: m.map_generation,
        surface_offset,
        surface_bpr,
        span_end,
        width: w,
        height: h,
        pixel_format: c0.format,
        texture_ref: 0,
    })
}

/// Arm the deferred window for a type-11 render Store, so the CPU writeback
/// into the mapping's guest pages happens on demand instead of every Store.
///
/// The caller has already read the target back and refreshed
/// `surface_cache` with this frame, so the pixels the flush will write are the
/// ones every other consumer already sees. **Only the guest-page copy is
/// deferred** — the readback, the cache, the Load seed and the present capture
/// are untouched, which is what keeps this rail out of the front-buffer
/// resolve problem that a resident-authoritative type-11 Load would reopen (see
/// the note at the type-11 Load arm).
///
/// The index is the mapping-keyed one the compute rail already uses, so every
/// guest-page reader drains this window through the `flush_intersecting` choke
/// point it already calls — no new trigger sites, and no way to cover one rail
/// and miss the other.
#[cfg(feature = "backend-vulkan")]
#[allow(
    clippy::too_many_arguments,
    reason = "the arm names the frame it is deferring and the geometry it was drawn at"
)]
fn arm_surface_deferred_store_with<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    req: &DrawEncodeRequest,
    mapping_id: u32,
    width: u32,
    height: u32,
    rgba: &[u8],
) -> bool {
    let Some(key) = surface_store_defer_eligible(state, req) else {
        return false;
    };
    if key.mapping_id != mapping_id || key.width != width || key.height != height {
        return false;
    }
    // Supersede — do not flush — the window this Store fully covers.
    //
    // This is `supersede_gva_window`'s rule, and it is what makes the rail a
    // deferral instead of a rescheduling. A compositor painting the same surface
    // every frame re-Stores the identical guest range, so the previous window
    // always intersects: flushing it here would perform exactly the guest write
    // this rail exists to skip, once per Store, and `surface_flush` would track
    // `surface_deferred` at a ratio of 1.
    //
    // Dropping is sound for the reason it is sound on the GVA rail: those bytes
    // were never observable without a flush — any reader would have taken the
    // window first — and this Store's pixels cover every byte of the range.
    let covered: Vec<crate::model::ComputeStorageResidencyKey> = state
        .compute_deferred_flush
        .iter()
        .filter(|(k, o)| {
            k.mapping_id == key.mapping_id
                && k.surface_offset == key.surface_offset
                && k.span_end == key.span_end
                && matches!(o, crate::model::DeferredOwner::Render { .. })
        })
        .map(|(k, _)| *k)
        .collect();
    for old in covered {
        if state.take_deferred_flush_window_exact(&old).is_some() {
            crate::observe::line(format!(
                "surface_deferred_superseded mapping={} {}x{} fmt={:#x}",
                old.mapping_id, old.width, old.height, old.pixel_format
            ));
        }
    }
    // Whatever still intersects covers guest bytes this Store does *not* write —
    // a different plane window on the same mapping — so it has to land.
    //
    // Each of those windows carries its own pixels, so unlike the first cut of
    // this rail the order against the cache refresh below no longer matters.
    if !crate::runtime::storage_flush::flush_intersecting(
        state,
        host,
        key.mapping_id,
        key.surface_offset,
        key.span_end,
    ) {
        // A window that would not land is a window whose guest bytes are now
        // unknown; arming over it would attribute its loss to this Store.
        return false;
    }
    // Convert once, into guest scanout order, and reference it twice: the Load
    // seed and the present capture read it through `surface_cache`, and the
    // window below owns it so the writeback it defers can always be performed.
    // This is the one copy this rail still pays.
    let frame = std::sync::Arc::new({
        let mut bgra = rgba.to_vec();
        for px in bgra.chunks_exact_mut(4) {
            px.swap(0, 2);
        }
        bgra
    });
    crate::runtime::surface_cache::store_shared(
        state,
        mapping_id,
        width,
        height,
        std::sync::Arc::clone(&frame),
    );
    evict_render_windows_to_cap(state, host);
    state.surface_deferred_seq = state.surface_deferred_seq.wrapping_add(1);
    let armed_seq = state.surface_deferred_seq;
    state.compute_deferred_flush.insert(
        key,
        crate::model::DeferredOwner::Render {
            armed_seq,
            bgra: frame,
        },
    );
    // Raw task-GVA reads that alias these physical pages flush through
    // `flush_intersecting_task_gva`, which finds the mapping via this index.
    state.index_deferred_alias_pages(key.mapping_id);
    crate::observe::line(format!(
        "surface_writeback_deferred mapping={} {}x{} fmt={:#x} pipe={} windows={}",
        key.mapping_id,
        key.width,
        key.height,
        key.pixel_format,
        req.pipeline_ref,
        state.compute_deferred_flush.len()
    ));
    true
}

/// Whether the type-11 writeback may be deferred at all. Same engine-level
/// gate the GVA rail asks, so one switch turns every deferred-writeback rail
/// off together.
#[cfg(feature = "backend-vulkan")]
fn deferred_gpu_only_content_allowed_for_surface() -> bool {
    crate::backend::vulkan::engine::deferred_gpu_only_content_allowed()
}

/// Live [`crate::model::DeferredOwner::Render`] windows, for the population cap.
#[cfg(feature = "backend-vulkan")]
fn render_window_count(state: &DeviceState) -> usize {
    state
        .compute_deferred_flush
        .values()
        .filter(|o| matches!(o, crate::model::DeferredOwner::Render { .. }))
        .count()
}

/// Land render windows oldest-first until the population is back under
/// [`SURFACE_DEFERRED_WINDOW_CAP`].
///
/// Through the normal choke point rather than taking entries directly:
/// `flush_intersecting` runs the fixpoint that drags in siblings overlapping the
/// same guest bytes, and taking one window out from under that would leave those
/// siblings holding stale ranges.
///
/// A window can legitimately survive its flush — a condemned backing holds its
/// obligation for `mapper::resolve` to settle — so this steps over it and tries
/// the next oldest. Stopping there would wedge the cap behind one stuck mapping
/// for every other mapping, and a window owns the frame it deferred, so the
/// leak would be a full framebuffer per stuck key.
///
/// The order is taken once and walked. Re-deriving "the oldest" after a refusal
/// returns the same stuck window forever, which is the bug this replaced.
#[cfg(feature = "backend-vulkan")]
fn evict_render_windows_to_cap<M: HostMemory + HostOps>(state: &mut DeviceState, host: &mut M) {
    for (mid, lo, hi) in render_windows_oldest_first(state) {
        if render_window_count(state) < SURFACE_DEFERRED_WINDOW_CAP {
            return;
        }
        crate::runtime::storage_flush::flush_intersecting(state, host, mid, lo, hi);
    }
}

/// Guest byte ranges of the live render windows, oldest first, for the cap's
/// eviction order. Compute windows are never chosen — they are bounded by their
/// own dispatches, and evicting one here would land content this cap was not
/// sized for.
///
/// The whole order rather than just the minimum, because a window can
/// legitimately refuse to land: a condemned backing holds its obligation for
/// `mapper::resolve` to settle, and one boot held one for 121 s. Stopping at the
/// oldest would wedge the cap behind it for *every other mapping*, and since a
/// window now owns the frame it deferred that is a full framebuffer per stuck
/// key — the "~260 stale residents pinned for the guest lifetime" shape. Step
/// over it instead.
#[cfg(feature = "backend-vulkan")]
fn render_windows_oldest_first(state: &DeviceState) -> Vec<(u32, u64, u64)> {
    let mut live: Vec<(u64, u32, u64, u64)> = state
        .compute_deferred_flush
        .iter()
        .filter_map(|(k, o)| match o {
            crate::model::DeferredOwner::Render { armed_seq, .. } => {
                Some((*armed_seq, k.mapping_id, k.surface_offset, k.span_end))
            }
            _ => None,
        })
        .collect();
    live.sort_unstable_by_key(|(seq, ..)| *seq);
    live.into_iter()
        .map(|(_, mid, lo, hi)| (mid, lo, hi))
        .collect()
}

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
        publish_surface_store(state, host, mapping_id, w, h, fmt);
    }
    wrote
}

#[cfg(all(test, feature = "backend-vulkan"))]
mod vulkan_split_tests {
    use super::*;
    use crate::model::{DeviceId, PAGE_SHIFT_X86};
    use crate::runtime::host::FakeHost;

    /// A type-11 `LOAD` whose host cache misses seeds from the surface's own
    /// guest pages, and only refuses when those cannot serve the extent.
    ///
    /// Without the guest-pages rung this returns `None`, `target_rgba8` stays
    /// unset, and `exec.rs` resolves the pass load action to `Clear` against the
    /// hardcoded `[0,0,0,0]` — so the guest's request to preserve its surface
    /// became a transparent-black wipe that the matching Store published. One
    /// x86/Vulkan boot measured 121 distinct (mapping, geometry) instances of that
    /// in ~170 s, four at the full 1920x1080 composite extent, with the host
    /// window 62-90 % near-black during a desktop drag against 0.001 % at idle.
    ///
    /// Every one of those 121 lines had `want == mapgeom` and `hostgen=0`: the
    /// cache had never held the surface and its pages were readable. That pair is
    /// what makes reading them the fix rather than a guess.
    #[test]
    fn a_type11_load_seed_falls_back_to_the_surfaces_own_guest_pages() {
        use crate::contract::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
        use crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;
        use crate::runtime::mapping_write::write_bgra8;

        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        let mid = 911u32;
        let pfn = 0x21u32;
        let gpa = (pfn as u64) << PAGE_SHIFT_X86;
        host.map_range(gpa, 0x4000, 0);
        state.map_surface(mid);
        {
            let m = state.mappings.get_mut(&mid).unwrap();
            m.mapped = true;
            m.mapping_internal = 1;
            m.page_entries = vec![(pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
        }
        let (w, h) = (4u32, 2u32);
        assert!(state.set_mapping_geom(mid, w, h, MTL_FORMAT_BGRA8_UNORM));

        // Guest-side content the compositor expects a LOAD to preserve. BGRA on
        // the wire; distinct per channel so a swizzle error cannot pass.
        let mut pages = vec![0u8; (w * h * 4) as usize];
        for px in pages.chunks_exact_mut(4) {
            px.copy_from_slice(&[0x10, 0x20, 0x30, 0xFF]);
        }
        assert!(write_bgra8(&mut state, &mut host, mid, &pages, w * 4, w, h));
        // `write_bgra8` mirrors what it wrote into the host cache, so drop that
        // mirror: the case under test is a surface whose pages hold content while
        // the cache holds nothing, which is what every one of the 121 measured
        // lines was (`hostgen=0`) — a first-ever LOAD, or a mapping whose remap
        // made `unmap_surface` evict the entry.
        crate::runtime::surface_cache::evict(&mut state, mid);
        assert!(
            crate::runtime::surface_cache::get(&state, mid, w, h).is_none(),
            "the cache must be cold: this test is about the miss path"
        );

        // Capture the always-on lines so a failure here names the check that
        // refused rather than showing a bare `None`: every rung on this ladder
        // declines by name, and the panic message is where that is worth reading.
        let cap = crate::observe::sink::FailCapture::start();
        let served = resolve_type11_load_seed(&mut state, &mut host, mid, w, h);
        let (bytes, order) = served.unwrap_or_else(|| {
            panic!(
                "a cold cache must not lose the guest's LOAD; sink said {:?}",
                cap.lines()
            )
        });
        drop(cap);
        assert_eq!(
            order,
            crate::backend::vulkan::engine::SeedOrder::Rgba8,
            "the guest-pages reader converts to RGBA8; mislabelling it swaps R and B"
        );
        assert_eq!(bytes.len(), (w * h * 4) as usize);
        assert_eq!(
            &bytes[..4],
            &[0x30, 0x20, 0x10, 0xFF],
            "BGRA guest bytes must arrive as semantic RGBA"
        );

        // A live cache entry still wins: it is the fresher copy (the last Store's
        // output) and the fallback must stay a fallback.
        let mut cached = vec![0u8; (w * h * 4) as usize];
        for px in cached.chunks_exact_mut(4) {
            px.copy_from_slice(&[0xAA, 0xBB, 0xCC, 0xFF]);
        }
        crate::runtime::surface_cache::store(&mut state, mid, w, h, cached);
        let (bytes, order) = resolve_type11_load_seed(&mut state, &mut host, mid, w, h)
            .expect("a warm cache must serve");
        assert_eq!(order, crate::backend::vulkan::engine::SeedOrder::Bgra8);
        assert_eq!(&bytes[..4], &[0xAA, 0xBB, 0xCC, 0xFF]);

        // An extent the surface is not latched at cannot be served by either rung,
        // and refusing is right: a seed of the wrong length is rejected by the
        // engine anyway, and the decline names both geometries.
        assert!(
            resolve_type11_load_seed(&mut state, &mut host, mid, w, h + 1).is_none(),
            "a mismatched extent must refuse by name, not seed something else"
        );
    }

    /// The type-11 `LOAD` seed branch reports both ways, and the miss arm names
    /// the geometry the cache actually holds.
    ///
    /// The miss is a whole-layer loss, not a degradation: with no seed the engine
    /// resolves the pass load action to `Clear` against the hardcoded
    /// `[0,0,0,0]`, so the guest's request to preserve its surface becomes a
    /// transparent-black wipe that the matching Store publishes. It reported
    /// nothing at all before this.
    ///
    /// The hit arm is asserted too, because a zero on the miss arm has to be
    /// readable: without a hit line beside it, "the cache always hit" and "this
    /// branch never ran" produce the same empty grep.
    ///
    /// Mapping ids here are chosen not to collide with any other test's, because
    /// `first_sight` latches per `(reason, discriminant)` for the life of the
    /// process and never resets.
    #[test]
    fn the_type11_load_seed_branch_reports_both_ways() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mid = 909u32;
        state.map_surface(mid);
        {
            let m = state.mappings.get_mut(&mid).unwrap();
            m.mapped = true;
            m.has_geom = true;
            m.width = 8;
            m.height = 4;
            m.map_generation = 3;
        }
        crate::runtime::surface_cache::store(&mut state, mid, 8, 4, vec![0u8; 8 * 4 * 4]);

        // The captured lines carry the sink's `OFF `/`FAIL ` severity prefix, so
        // match on the event token rather than on the first word.
        let only = |cap: &crate::observe::sink::FailCapture| -> String {
            let hits: Vec<String> = cap
                .lines()
                .into_iter()
                .filter(|l| l.contains("type11_load_seed"))
                .collect();
            assert_eq!(hits.len(), 1, "expected exactly one line, got {hits:?}");
            hits.into_iter().next().unwrap_or_default()
        };

        let cap = crate::observe::sink::FailCapture::start();
        note_type11_load_seed(&state, mid, 8, 4, Some(Type11SeedRung::Cache));
        let hit = only(&cap);
        assert!(hit.contains("outcome=cache_hit"), "{hit}");
        assert!(hit.contains("mapgeom=8x4"), "{hit}");
        assert!(hit.contains("mapgen=3"), "{hit}");
        drop(cap);

        // The recovered arm is its own outcome, not folded into `cache_hit`: its
        // rate is the only thing that prices the guest-pages fallback, and fusing
        // it would make the fix unmeasurable the moment it worked.
        let cap = crate::observe::sink::FailCapture::start();
        note_type11_load_seed(&state, mid, 4, 4, Some(Type11SeedRung::GuestPages));
        let pages = only(&cap);
        assert!(pages.contains("outcome=guest_pages"), "{pages}");
        drop(cap);

        // Same mapping, a geometry the cache does not hold: the entry's own
        // geometry is the load-bearing field, since it says a Store at another
        // extent orphaned every window still living at this one.
        let cap = crate::observe::sink::FailCapture::start();
        note_type11_load_seed(&state, mid, 8, 1, None);
        let geom = only(&cap);
        assert!(geom.contains("reason=type11_seed_cache_geom"), "{geom}");
        assert!(geom.contains("have=8x4"), "{geom}");
        assert!(geom.contains("want=8x1"), "{geom}");
        drop(cap);

        // A mapping the cache has never held reports absence, not a geometry.
        let cap = crate::observe::sink::FailCapture::start();
        note_type11_load_seed(&state, 910, 8, 4, None);
        let absent = only(&cap);
        assert!(
            absent.contains("reason=type11_seed_cache_absent"),
            "{absent}"
        );
        assert!(!absent.contains("have="), "{absent}");
        drop(cap);

        // Latched per (mapping, geometry, outcome): a repeat of any of the three
        // above emits nothing, so the branch is safe to leave on forever.
        let cap = crate::observe::sink::FailCapture::start();
        note_type11_load_seed(&state, mid, 8, 4, Some(Type11SeedRung::Cache));
        note_type11_load_seed(&state, mid, 4, 4, Some(Type11SeedRung::GuestPages));
        note_type11_load_seed(&state, mid, 8, 1, None);
        note_type11_load_seed(&state, 910, 8, 4, None);
        assert!(
            cap.lines().is_empty(),
            "second sighting must be latched: {:?}",
            cap.lines()
        );
    }

    /// One window that refuses to land must not wedge the cap for every other
    /// mapping.
    ///
    /// A condemned backing holds its obligation for `mapper::resolve` to settle —
    /// one boot held one for 121 s across 13015 flush attempts — and the eviction
    /// loop used to re-derive "the oldest" each pass and stop when it did not
    /// shrink. That returns the same stuck window forever, so the population
    /// grows without bound past the cap. It was survivable while a window was
    /// just a key; now that a window owns the frame it deferred, it leaks a whole
    /// framebuffer per stuck key.
    #[test]
    fn a_stuck_oldest_window_does_not_wedge_the_cap_for_the_others() {
        use crate::contract::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();

        let arm = |state: &mut DeviceState, mid: u32, seq: u64| {
            let gpa = 0xA000_0000u64 + (mid as u64) * 0x10_0000;
            state.map_surface(mid);
            {
                let m = state.mappings.get_mut(&mid).unwrap();
                m.mapped = true;
                m.map_generation = 1;
                m.page_entries = vec![
                    (((gpa >> PAGE_SHIFT_X86) as u32) << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID,
                ];
            }
            state.compute_deferred_flush.insert(
                crate::model::ComputeStorageResidencyKey {
                    mapping_id: mid,
                    map_generation: 1,
                    surface_offset: 0,
                    surface_bpr: 64,
                    span_end: 256,
                    width: 4,
                    height: 4,
                    pixel_format: crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM,
                    texture_ref: 0,
                },
                crate::model::DeferredOwner::Render {
                    armed_seq: seq,
                    bgra: std::sync::Arc::new(vec![0u8; 4 * 4 * 4]),
                },
            );
        };

        // The oldest window sits on a condemned backing, so its flush is held.
        arm(&mut state, 1, 1);
        assert!(state.condemn_surface_backing(1), "mapping 1 must condemn");
        assert!(state.mapping_backing_condemned(1));
        // Fill past the cap with windows that can land.
        for i in 0..SURFACE_DEFERRED_WINDOW_CAP {
            arm(&mut state, 2 + i as u32, 2 + i as u64);
        }
        let before = render_window_count(&state);
        assert!(before > SURFACE_DEFERRED_WINDOW_CAP);

        evict_render_windows_to_cap(&mut state, &mut host);

        assert!(
            render_window_count(&state) < before,
            "the stuck oldest must be stepped over, not stopped on"
        );
        assert!(
            state.mapping_backing_condemned(1),
            "and the held window's mapping is left for the resolve to settle"
        );
    }

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

    /// The branch line is only worth leaving on forever if it is bounded, and
    /// the bound is the dedup: one line per distinct route per process, however
    /// many Stores take that route.
    ///
    /// The load-bearing assertion is the dedup, not the text — a per-Store line
    /// on this path is a flood (thousands per session under compositing) and
    /// would have to be removed again, which is how the tree ended up with no
    /// always-on record of this branch in the first place.
    #[test]
    fn the_store_route_line_is_one_per_route_per_process() {
        crate::observe::redirect_logs_for_tests();
        let path = crate::observe::fail_log_path();
        let mark = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0) as usize;

        // Routes distinct from every product route so this test cannot be
        // satisfied by a line some other case in this binary emitted.
        note_type11_store_route("test_route_a");
        note_type11_store_route("test_route_a");
        note_type11_store_route("test_route_a");
        note_type11_store_route("test_route_b");

        let whole = std::fs::read_to_string(path).expect("fail log");
        let appended = &whole[mark.min(whole.len())..];
        let count = |route: &str| {
            appended
                .lines()
                .filter(|l| l.contains(&format!("type11_store_route route={route}")))
                .count()
        };
        assert_eq!(count("test_route_a"), 1, "three calls, one line");
        assert_eq!(count("test_route_b"), 1, "a second route still reports");
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
