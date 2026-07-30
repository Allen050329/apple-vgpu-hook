use super::*;
use crate::model::{DeviceId, PAGE_SHIFT_ARM64E, PAGE_SHIFT_X86};
use crate::runtime::gva_mem::write_task_gva_arm64e;
use crate::runtime::host::FakeHost;

#[cfg(feature = "backend-vulkan")]
#[test]
fn m2v_draw_boundary_preserves_the_engine_vk_call_slug() {
    let req = DrawEncodeRequest {
        pipeline_ref: 73,
        ..DrawEncodeRequest::default()
    };
    let err = crate::backend::vulkan::engine::vk_call::exec_submit_device_lost_fixture();

    let line = linux_m2v_draw_failure(&err, &req).render();
    assert!(
        line.starts_with("linux_m2v_draw reason=vk_exec_submit vk_result="),
        "the delegated VkCall slug must be the boundary's primary reason: {line}"
    );
    assert!(line.contains(" pipe=73 task=0 geom=0x0"));
    assert!(
        !line.contains("reason=vk_engine_vk_untyped"),
        "the boundary must not flatten a typed VkCall back into DrawError prose: {line}"
    );
}

/// One reflection binding of `kind` at `metal_index` for the
/// `frag_declared_unbound` guard tests. Only kind + index are load-bearing.
fn rb(
    kind: metal2vulkan::reflect::ResourceKind,
    metal_index: u32,
) -> metal2vulkan::reflect::ResourceBinding {
    metal2vulkan::reflect::ResourceBinding {
        kind,
        metal_index,
        descriptor: None,
        param_index: None,
        address_space: None,
        declared_size: None,
        type_layout: None,
        type_name: None,
        texture_shape: None,
        embedded_source: None,
        access: None,
        static_sampler: None,
    }
}

/// Regression guard for the sampled zero-copy floor (kb [[paravirt-vulkan-engine]]
/// (C)). The floor is a perf crossover, not a magic constant: it MUST sit
/// strictly between the two bands a 2026-07-20 video/scroll census measured,
/// so a future edit that re-breaks either case fails here.
///
/// - Raising it back toward 256 KiB re-strands the per-frame video composite
///   surfaces on the CPU byte path (the `t11_guest=930:226 MB` bug this floor
///   drop closed): the composites clustered at ~236 KiB.
/// - Dropping it to the small-bind band (scroll glyphs ~3.6 KiB; small-UI /
///   gva_copy ~21–34 KiB) trades cheap CPU copies for many tiny GPU gathers +
///   host-import windows.
///
/// Vulkan-arm only: pins the `backend-vulkan` zero-copy byte floors.
#[cfg(feature = "backend-vulkan")]
#[test]
#[allow(
    clippy::assertions_on_constants,
    reason = "the test pins the measured crossover bands around these product constants"
)]
fn sampled_zero_copy_floor_separates_video_from_small_binds() {
    // Largest observed bind that still legitimately prefers the CPU byte
    // path (small-UI / gva_copy avg, 2026-07-20 scroll census). The floor
    // must exclude it (stay on CPU / memo).
    const LARGEST_CPU_PREFERRED_BIND: u64 = 34 * 1024;
    // Smallest observed per-frame video composite bind that must ride the
    // zero-copy gather (2026-07-20 video census clustered at ~236 KiB).
    const SMALLEST_VIDEO_COMPOSITE_BIND: u64 = 200 * 1024;
    assert!(
        ZERO_COPY_SAMPLED_MIN_BYTES > LARGEST_CPU_PREFERRED_BIND,
        "floor {ZERO_COPY_SAMPLED_MIN_BYTES} must exceed the CPU-preferred band \
             {LARGEST_CPU_PREFERRED_BIND} so small-UI/glyph binds stay on the CPU/memo path"
    );
    assert!(
        ZERO_COPY_SAMPLED_MIN_BYTES < SMALLEST_VIDEO_COMPOSITE_BIND,
        "floor {ZERO_COPY_SAMPLED_MIN_BYTES} must be below the video-composite band \
             {SMALLEST_VIDEO_COMPOSITE_BIND} so per-frame video surfaces ride zero-copy \
             (re-breaks the t11_guest=226 MB video CPU bug if raised)"
    );
    // The buffer floor is a distinct, lower crossover (small per-draw
    // uniform/vertex buffers); it is not the sampled floor and stays 16 KiB.
    assert!(ZERO_COPY_BUFFER_MIN_BYTES < ZERO_COPY_SAMPLED_MIN_BYTES);
}

#[test]
#[cfg(feature = "backend-vulkan")]
fn type11_zero_copy_declines_transient_host_mappings() {
    use crate::contract::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
    use crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
    let mid = 7u32;
    let width = 128u32;
    let height = 128u32;
    let page_count = 16u32;
    let base_pfn = 0x100u32;
    let page = 1u64 << PAGE_SHIFT_X86;
    for i in 0..page_count {
        host.map_range(((base_pfn + i) as u64) << PAGE_SHIFT_X86, page as usize, 0);
    }
    assert!(state.map_surface(mid));
    {
        let m = state.mappings.get_mut(&mid).unwrap();
        m.page_entries = (0..page_count)
            .map(|i| ((base_pfn + i) << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID)
            .collect();
    }
    assert!(state.set_mapping_geom(mid, width, height, MTL_FORMAT_BGRA8_UNORM));

    assert!(matches!(
        try_type11_sample_zero_copy(&mut state, &mut host, mid, width, height),
        Err(t11_decline::Reason::UnstableMap)
    ));
    assert_eq!(
        host.map_pages_calls, 0,
        "transient hosts must decline before creating an importable view"
    );
}

#[test]
fn cpu_portability_store_publishes_composite() {
    use crate::contract::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
    use crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    use crate::model::SurfaceWriteKind;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let (mid, width, height) = (5u32, 64u32, 48u32);
    assert!(state.map_surface(mid));
    {
        let mapping = state.mappings.get_mut(&mid).unwrap();
        mapping.mapped = true;
        mapping.has_geom = true;
        mapping.width = width;
        mapping.height = height;
        mapping.format = MTL_FORMAT_BGRA8_UNORM;
        mapping.page_entries = vec![(1 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
        mapping.content_generation = 2;
    }
    state.present.valid = true;
    state.present.width = width;
    state.present.height = height;
    state.present.frame_flush_seen = true;

    publish_surface_store(
        &mut state,
        &mut host,
        mid,
        width,
        height,
        MTL_FORMAT_BGRA8_UNORM,
    );

    assert_eq!(state.surface_write_kind(mid), SurfaceWriteKind::Composite);
    assert_eq!(state.present.early_front_mapping, mid);
}

#[test]
fn frag_unbound_scan_reports_missing_standard_kinds_and_embedded_textures() {
    use metal2vulkan::reflect::ResourceKind as K;
    // Shader declares buffer 1+2, texture 3, sampler 0, an embedded arg-buffer
    // texture (index 9), plus other synthetic kinds (color input, threadgroup
    // buffer, storage image, constexpr sampler) that reach the shader by other
    // paths and must NOT be reported as standard-unbound.
    let bindings = [
        rb(K::Buffer, 1),
        rb(K::Buffer, 2),
        rb(K::Texture, 3),
        rb(K::Sampler, 0),
        rb(K::EmbeddedArgBufferTexture, 9),
        rb(K::ColorInput, 0),
        rb(K::ThreadgroupBuffer, 5),
        rb(K::StorageImage, 4),
        rb(K::StaticSampler, 1),
    ];
    // All standard resources bound → `unbound` empty; the embedded texture is
    // always reported (render path cannot source it) regardless of binding.
    let (unbound, embedded) =
        frag_unbound_scan(&bindings, |i| [1, 2].contains(&i), |i| i == 3, |i| i == 0);
    assert!(unbound.is_empty());
    assert_eq!(embedded, vec![9]);

    // Drop the texture bind → exactly tex3 reported (synthetics stay silent).
    let (unbound, _) = frag_unbound_scan(&bindings, |i| [1, 2].contains(&i), |_| false, |i| i == 0);
    assert_eq!(unbound, vec!["tex3".to_string()]);

    // Drop buffer 2 + sampler 0 → both reported, ordered by declaration.
    let (unbound, _) = frag_unbound_scan(&bindings, |i| i == 1, |i| i == 3, |_| false);
    assert_eq!(unbound, vec!["buf2".to_string(), "smp0".to_string()]);

    // A reflection with no embedded texture returns an empty embedded list.
    let standard_only = [rb(K::Buffer, 1), rb(K::Texture, 3), rb(K::Sampler, 0)];
    let (_, embedded) = frag_unbound_scan(&standard_only, |_| true, |_| true, |_| true);
    assert!(embedded.is_empty());
}

#[cfg(feature = "backend-vulkan")]
#[test]
fn reflected_static_sampler_maps_exact_state_and_rejects_unimplemented_modes() {
    use crate::backend::vulkan::engine::{
        SamplerAddressMode as EngineAddress, SamplerFilter as EngineFilter,
        SamplerMipFilter as EngineMip,
    };
    use metal2vulkan::reflect::{
        SamplerAddressMode, SamplerBorderColor, SamplerCompareFunction, SamplerCoordinates,
        SamplerFilter, SamplerMipFilter, SamplerReduction, StaticSamplerState,
    };

    let mut state = StaticSamplerState {
        min_filter: SamplerFilter::Linear,
        mag_filter: SamplerFilter::Linear,
        mip_filter: SamplerMipFilter::None,
        address_mode_s: SamplerAddressMode::ClampToEdge,
        address_mode_t: SamplerAddressMode::ClampToEdge,
        address_mode_r: SamplerAddressMode::ClampToEdge,
        coordinates: SamplerCoordinates::Normalized,
        compare_function: SamplerCompareFunction::Never,
        max_anisotropy: 1,
        lod_min_clamp: 0.0,
        lod_max_clamp: 65504.0,
        border_color: SamplerBorderColor::TransparentBlack,
        reduction: SamplerReduction::WeightedAverage,
        lod_bias: 0.0,
        raw_words: [0x807b_ff00_0008_0a49, 0],
    };
    let mapped =
        reflected_static_sampler_resource("fragment", 65, state).expect("supported sampler");
    assert_eq!(mapped.binding, 65);
    assert_eq!(mapped.min_filter, EngineFilter::Linear);
    assert_eq!(mapped.mag_filter, EngineFilter::Linear);
    assert_eq!(mapped.mip_filter, EngineMip::NotMipmapped);
    assert_eq!(mapped.address_mode_u, EngineAddress::ClampToEdge);
    assert_eq!(mapped.address_mode_v, EngineAddress::ClampToEdge);
    assert_eq!(mapped.address_mode_w, EngineAddress::ClampToEdge);
    assert_eq!(mapped.lod_min_f32(), 0.0);
    assert_eq!(mapped.lod_max_f32(), 65504.0);
    assert!(!mapped.unnormalized_coordinates);

    state.min_filter = SamplerFilter::Nearest;
    state.mag_filter = SamplerFilter::Nearest;
    state.address_mode_s = SamplerAddressMode::Repeat;
    state.address_mode_t = SamplerAddressMode::Repeat;
    state.address_mode_r = SamplerAddressMode::Repeat;
    let repeat = reflected_static_sampler_resource("fragment", 66, state).expect("repeat sampler");
    assert_eq!(repeat.min_filter, EngineFilter::Nearest);
    assert_eq!(repeat.address_mode_u, EngineAddress::Repeat);

    state.min_filter = SamplerFilter::Bicubic;
    assert_eq!(
        reflected_static_sampler_resource("fragment", 66, state)
            .unwrap_err()
            .slug(),
        "draw_prepare_static_sampler_min_filter_unsupported"
    );
    state.min_filter = SamplerFilter::Nearest;
    state.reduction = SamplerReduction::Minimum;
    assert_eq!(
        reflected_static_sampler_resource("fragment", 66, state)
            .unwrap_err()
            .slug(),
        "draw_prepare_static_sampler_reduction_unsupported"
    );
    state.reduction = SamplerReduction::WeightedAverage;
    state.lod_bias = 1.0;
    assert_eq!(
        reflected_static_sampler_resource("fragment", 66, state)
            .unwrap_err()
            .slug(),
        "draw_prepare_static_sampler_lod_bias_unsupported"
    );
}

#[cfg(feature = "backend-vulkan")]
#[test]
fn reflected_sampled_collision_includes_sampler_band_only() {
    use metal2vulkan::reflect::ShaderStage;

    let vertex = shader_pull_reflection(&[64]);
    let mut fragment = shader_pull_reflection(&[64]);
    fragment.stage = ShaderStage::Fragment;
    assert!(reflected_sampled_binding_collision(&vertex, &fragment));

    fragment.bindings[0].descriptor.as_mut().unwrap().binding = 65;
    assert!(!reflected_sampled_binding_collision(&vertex, &fragment));

    fragment.bindings[0].descriptor.as_mut().unwrap().binding = 96;
    assert!(!reflected_sampled_binding_collision(&vertex, &fragment));
}

#[test]
fn depth_stencil_triviality_matches_no_op_state() {
    use crate::runtime::decode::resource::DepthStencilDescriptor;
    // compare Always (7), no write, no stencil → equivalent to no depth test.
    let trivial = DepthStencilDescriptor {
        depth_compare_function: 7,
        depth_write_enabled: false,
        front_stencil_enabled: false,
        back_stencil_enabled: false,
        ..Default::default()
    };
    assert!(depth_stencil_descriptor_is_trivial(&trivial));

    // A real compare function (Less=1) occludes → non-trivial.
    assert!(!depth_stencil_descriptor_is_trivial(
        &DepthStencilDescriptor {
            depth_compare_function: 1,
            ..trivial.clone()
        }
    ));
    // Depth write on → non-trivial even with compare Always.
    assert!(!depth_stencil_descriptor_is_trivial(
        &DepthStencilDescriptor {
            depth_write_enabled: true,
            ..trivial.clone()
        }
    ));
    // Either stencil face enabled → non-trivial.
    assert!(!depth_stencil_descriptor_is_trivial(
        &DepthStencilDescriptor {
            front_stencil_enabled: true,
            ..trivial.clone()
        }
    ));
    assert!(!depth_stencil_descriptor_is_trivial(
        &DepthStencilDescriptor {
            back_stencil_enabled: true,
            ..trivial
        }
    ));
}

#[cfg(feature = "backend-vulkan")]
#[test]
fn bound_depth_stencil_that_cannot_resolve_returns_named_reason() {
    // A guest that binds a depth-stencil ref (`ds_ref != 0`) whose object-list
    // entry does not resolve must surface a *specific* reason, not `None`: the
    // draw silently disables the depth test otherwise, and every other
    // depth/stencil degradation on this path is already fail-visible. With an
    // empty state the lookup misses → the entry-missing reason, which the caller
    // logs as `shader_state_degraded reason=depth_stencil_entry_missing`.
    let state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    let host = FakeHost::new();
    let err = load_depth_stencil_descriptor(&state, &host, /*task*/ 4, /*ds_ref*/ 9)
        .expect_err("unresolvable bound depth-stencil must report a reason");
    assert_eq!(err, "depth_stencil_entry_missing");
}

#[test]
fn index_load_failures_report_the_specific_reason() {
    // The Vulkan indexed-draw path collapsed eleven distinct load failures into
    // one `index_buffer_miss`; each now names the failing check so a boot log
    // says *which* one fired.
    let state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    let host = FakeHost::new();

    // Unsupported MTLIndexType (only 0=u16 / 1=u32 exist).
    let bad_type = IndexedDrawInfo {
        index_type: 5,
        index_count: 3,
        index_buffer_ref: 9,
        index_buffer_offset: 0,
    };
    assert_eq!(
        load_index_bytes_reason(&state, &host, 4, &bad_type),
        Err(IndexLoadReason::TypeUnsupported)
    );

    // Valid type + count, but the bound index buffer ref resolves to nothing on
    // an empty state → the entry-missing site, not a generic miss.
    let unresolved = IndexedDrawInfo {
        index_type: 1,
        index_count: 6,
        index_buffer_ref: 9,
        index_buffer_offset: 0,
    };
    assert_eq!(
        load_index_bytes_reason(&state, &host, 4, &unresolved),
        Err(IndexLoadReason::EntryMissing)
    );
}

/// Eleven checks, eleven names, one namespace.
///
/// Crate-wide distinctness is `observe::gate`'s job; what this asserts is the
/// *prefix*, because bare names (`out_of_bounds`, `read_fail`) would match
/// three other rails on a `grep reason=` and the reader could not tell an
/// index buffer from a blit row.
#[test]
fn every_index_load_failure_has_its_own_namespaced_name() {
    use crate::observe::Decline as _;
    const ALL: &[IndexLoadReason] = &[
        IndexLoadReason::TypeUnsupported,
        IndexLoadReason::CountOverflow,
        IndexLoadReason::CountZero,
        IndexLoadReason::EntryMissing,
        IndexLoadReason::ObjectType,
        IndexLoadReason::DescRead,
        IndexLoadReason::DescDecode,
        IndexLoadReason::BackingMissing,
        IndexLoadReason::OffsetOverflow,
        IndexLoadReason::OutOfBounds,
        IndexLoadReason::ReadFail,
    ];
    let mut slugs: Vec<&str> = ALL.iter().map(|r| r.slug()).collect();
    for s in &slugs {
        assert!(
            s.starts_with("draw_index_"),
            "{s} is not namespaced to the indexed-draw path"
        );
    }
    let n = slugs.len();
    slugs.sort_unstable();
    slugs.dedup();
    assert_eq!(slugs.len(), n, "two index-load checks answer with one name");
}

/// The status carries the check *and* the class, and cannot render a line for
/// a success.
///
/// The class is not derivable from the slug and the caller acts on it —
/// `NoMetal` makes the exec loop honour the pass clear, `WritebackFailed` does
/// not — so a reader correlating a dropped draw with a black frame needs both
/// on the line.
#[test]
fn encode_status_renders_its_check_beside_the_class_it_collapsed_to() {
    use crate::observe::{Emit, Refusal as _};
    assert_eq!(
        Emit::refusal(
            "draw_encode_fail",
            &EncodeStatus::MissingMtlb("draw_mtl_vertex_mtlb_load")
        )
        .expect("a refusal must render a line")
        .render(),
        "draw_encode_fail reason=draw_mtl_vertex_mtlb_load class=missing_mtlb"
    );
    assert_eq!(
        Emit::refusal(
            "render_icb",
            &EncodeStatus::WritebackFailed("icb_exec_writeback_none")
        )
        .expect("a refusal must render a line")
        .render(),
        "render_icb reason=icb_exec_writeback_none class=writeback_failed"
    );
    // I2's carve-out, enforced by the type: there is no line to send for a
    // success, so no call site can log one by forgetting a guard.
    assert!(
        Emit::refusal("draw_encode_fail", &EncodeStatus::Ok).is_none(),
        "Ok is control flow and must not be representable as a line"
    );
    assert_eq!(EncodeStatus::Ok.refusal(), None);
    assert_eq!(EncodeStatus::Ok.reason(), "ok");
    assert_eq!(EncodeStatus::Ok.class(), "ok");

    #[cfg(all(feature = "backend-metal", target_os = "macos"))]
    {
        let backend =
            crate::backend::metal::error::Status::execute("metal_render_command_buffer_failed");
        let carried = EncodeStatus::MetalBackend(backend);
        assert_eq!(carried.class(), "metal_execute");
        assert_eq!(
            Emit::refusal("draw_encode_fail", &carried)
                .expect("the draw carrier must retain the backend refusal")
                .render(),
            "draw_encode_fail reason=metal_render_command_buffer_failed \
                 class=execute recovery=metal_failed"
        );
    }
}

#[cfg(all(feature = "backend-metal", target_os = "macos"))]
#[test]
fn explicit_metal_sampler_and_depth_binds_return_typed_missing_entry_declines() {
    use crate::observe::Emit;

    let state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let host = FakeHost::new();

    let sampler = load_sampler(&state, &host, 4, 77, 3)
        .expect_err("a nonzero sampler ref in an empty object list must decline");
    assert_eq!(
        sampler,
        MetalStateDecline::SamplerEntryMissing {
            sampler_ref: 77,
            index: 3,
        }
    );
    assert_eq!(
        Emit::decline("metal_draw_sampler_fallback", &sampler)
            .field("task", 4)
            .field("pipe", 19)
            .field("stage", "fragment")
            .render(),
        "metal_draw_sampler_fallback reason=metal_sampler_entry_missing \
             sampler_ref=77 index=3 task=4 pipe=19 stage=fragment"
    );

    let depth = load_depth_stencil_state(&state, &host, 4, 88)
        .expect_err("a nonzero depth-stencil ref in an empty object list must decline");
    assert_eq!(
        depth,
        MetalStateDecline::DepthStencilEntryMissing {
            depth_stencil_ref: 88,
        }
    );
    assert_eq!(
        Emit::decline("metal_draw_depth_stencil_fallback", &depth)
            .field("task", 4)
            .field("pipe", 19)
            .render(),
        "metal_draw_depth_stencil_fallback reason=metal_depth_stencil_entry_missing \
             depth_stencil_ref=88 task=4 pipe=19"
    );
}

#[cfg(all(feature = "backend-metal", target_os = "macos"))]
#[test]
fn metal_state_decode_declines_delegate_the_exact_resource_decoder_reason() {
    use crate::observe::{Decline as _, Emit};

    let sampler = MetalStateDecline::SamplerDecode {
        sampler_ref: 41,
        index: 6,
        reason: DecodeStatus::ErrShort("res_sampler_short"),
    };
    assert_eq!(sampler.slug(), "res_sampler_short");
    assert_eq!(
        Emit::decline("metal_draw_sampler_fallback", &sampler).render(),
        "metal_draw_sampler_fallback reason=res_sampler_short \
             class=short sampler_ref=41 index=6"
    );

    let depth = MetalStateDecline::DepthStencilDecode {
        depth_stencil_ref: 52,
        reason: DecodeStatus::ErrShort("res_depth_stencil_short"),
    };
    assert_eq!(depth.slug(), "res_depth_stencil_short");
    assert_eq!(
        Emit::decline("metal_draw_depth_stencil_fallback", &depth).render(),
        "metal_draw_depth_stencil_fallback reason=res_depth_stencil_short \
             class=short depth_stencil_ref=52"
    );
}

#[cfg(all(feature = "backend-metal", target_os = "macos"))]
#[test]
fn metal_icb_inheritance_preflight_rejects_invalid_explicit_binds_but_not_unbound_slots() {
    let desc = crate::runtime::decode::resource::IndirectCommandBufferDescriptor {
        inherit_pipeline_state: true,
        ..Default::default()
    };

    let missing_parent_pipeline =
        validate_icb_inheritance_bind_shape(&DrawEncodeRequest::default(), &desc)
            .expect_err("inheritPipelineState requires a parent pipeline");
    assert_eq!(
        missing_parent_pipeline,
        MetalIcbInheritanceDecline::PipelineRefZero
    );

    let req = DrawEncodeRequest {
        pipeline_ref: 7,
        vertex_buffers: vec![BufferBind {
            index: MAX_BIND_SLOTS,
            buffer_ref: 93,
            ..BufferBind::default()
        }],
        ..DrawEncodeRequest::default()
    };
    assert_eq!(
        validate_icb_inheritance_bind_shape(&req, &desc),
        Err(MetalIcbInheritanceDecline::VertexBufferIndexOutOfRange {
            buffer_ref: 93,
            index: MAX_BIND_SLOTS,
        })
    );

    let unbound = DrawEncodeRequest {
        pipeline_ref: 7,
        vertex_buffers: vec![BufferBind {
            index: u32::MAX,
            buffer_ref: 0,
            ..BufferBind::default()
        }],
        ..DrawEncodeRequest::default()
    };
    assert_eq!(
        validate_icb_inheritance_bind_shape(&unbound, &desc),
        Ok(()),
        "ref==0 is an unbound slot, not a refusal or a log event"
    );
}

#[cfg(all(feature = "backend-metal", target_os = "macos"))]
#[test]
fn every_metal_icb_inheritance_check_is_unique_namespaced_and_log_safe() {
    use crate::observe::Decline as _;

    let all = vec![
        MetalIcbInheritanceDecline::CullModeUnsupported { value: 3 },
        MetalIcbInheritanceDecline::FrontFacingUnsupported { value: 2 },
        MetalIcbInheritanceDecline::VertexBufferIndexOutOfRange {
            buffer_ref: 1,
            index: 31,
        },
        MetalIcbInheritanceDecline::FragmentBufferIndexOutOfRange {
            buffer_ref: 2,
            index: 32,
        },
        MetalIcbInheritanceDecline::VertexBufferMissing {
            buffer_ref: 3,
            index: 4,
            offset: 5,
        },
        MetalIcbInheritanceDecline::FragmentBufferMissing {
            buffer_ref: 6,
            index: 7,
            offset: 8,
        },
        MetalIcbInheritanceDecline::VertexTextureIndexOutOfRange {
            texture_ref: 9,
            index: 33,
        },
        MetalIcbInheritanceDecline::FragmentTextureIndexOutOfRange {
            texture_ref: 10,
            index: 34,
        },
        MetalIcbInheritanceDecline::VertexTextureMissing {
            texture_ref: 11,
            index: 12,
            detail: "no list entry".into(),
        },
        MetalIcbInheritanceDecline::FragmentTextureMissing {
            texture_ref: 13,
            index: 14,
            detail: "guest\nread failed".into(),
        },
        MetalIcbInheritanceDecline::VertexSamplerIndexOutOfRange {
            sampler_ref: 15,
            index: 35,
        },
        MetalIcbInheritanceDecline::FragmentSamplerIndexOutOfRange {
            sampler_ref: 16,
            index: 36,
        },
        MetalIcbInheritanceDecline::PipelineRefZero,
        MetalIcbInheritanceDecline::PipelineMissing { pipeline_ref: 17 },
        MetalIcbInheritanceDecline::VertexMtlbMissing { function_ref: 18 },
        MetalIcbInheritanceDecline::FragmentMtlbMissing { function_ref: 19 },
        MetalIcbInheritanceDecline::VertexLibraryLoad {
            function_ref: 20,
            detail: "Metal error".into(),
        },
        MetalIcbInheritanceDecline::FragmentLibraryLoad {
            function_ref: 21,
            detail: "Metal error".into(),
        },
        MetalIcbInheritanceDecline::VertexFunctionCount {
            function_ref: 22,
            count: 2,
        },
        MetalIcbInheritanceDecline::FragmentFunctionCount {
            function_ref: 23,
            count: 0,
        },
        MetalIcbInheritanceDecline::VertexFunctionGet {
            function_ref: 24,
            detail: "function missing".into(),
        },
        MetalIcbInheritanceDecline::FragmentFunctionGet {
            function_ref: 25,
            detail: "function missing".into(),
        },
        MetalIcbInheritanceDecline::VertexDescriptorMissing {
            pipeline_ref: 26,
            attribute_count: 3,
        },
        MetalIcbInheritanceDecline::RenderPipelineCreate {
            pipeline_ref: 27,
            detail: "pipeline failed".into(),
        },
    ];

    let mut slugs = all.iter().map(Decline::slug).collect::<Vec<_>>();
    assert_eq!(slugs.len(), 24, "the fixture must cover every check");
    for decline in &all {
        assert!(decline.slug().starts_with("metal_icb_inherit_"));
        for (key, value) in decline.fields() {
            assert!(!key.is_empty());
            assert!(
                !value.chars().any(char::is_whitespace),
                "{} rendered a non-token field {key}={value:?}",
                decline.slug()
            );
        }
    }
    slugs.sort_unstable();
    slugs.dedup();
    assert_eq!(slugs.len(), all.len(), "two ICB checks share one reason");
}

#[cfg(all(feature = "backend-metal", target_os = "macos"))]
#[test]
fn metal_icb_inheritance_line_keeps_pipeline_and_sanitized_driver_detail() {
    use crate::observe::Emit;

    let decline = MetalIcbInheritanceDecline::RenderPipelineCreate {
        pipeline_ref: 71,
        detail: "Error Domain=MTLLibrary Code=3".into(),
    };
    assert_eq!(
        Emit::decline("metal_icb_inheritance", &decline)
            .field("task", 2)
            .field("pipe", 71)
            .field("icb", 99)
            .render(),
        "metal_icb_inheritance reason=metal_icb_inherit_render_pipeline_create \
             pipeline_ref=71 detail=Error_Domain=MTLLibrary_Code=3 \
             task=2 pipe=71 icb=99"
    );
}

/// A vertex reflection that trips the shader-pull coverage gate: writes
/// Position, reads VertexIndex, binds a Buffer at each of `bindings`.
fn shader_pull_reflection(bindings: &[u32]) -> metal2vulkan::reflect::ShaderReflection {
    use metal2vulkan::reflect::{
        DescriptorLocation, ResourceBinding, ResourceKind, ShaderReflection, ShaderStage,
        VertexBuiltins, REFLECTION_VERSION,
    };
    ShaderReflection {
        reflection_version: REFLECTION_VERSION,
        stage: ShaderStage::Vertex,
        entry_point: None,
        bindings: bindings
            .iter()
            .map(|&binding| ResourceBinding {
                kind: ResourceKind::Buffer,
                metal_index: binding,
                descriptor: Some(DescriptorLocation { set: 0, binding }),
                param_index: None,
                address_space: None,
                declared_size: None,
                type_layout: None,
                type_name: None,
                texture_shape: None,
                embedded_source: None,
                access: None,
                static_sampler: None,
            })
            .collect(),
        vertex_attributes: vec![],
        varyings: vec![],
        render_targets: vec![],
        depth_members: vec![],
        stencil_members: vec![],
        local_size: None,
        vertex_builtins: Some(VertexBuiltins {
            uses_vertex_index: true,
            uses_instance_index: false,
            writes_position: true,
        }),
        imageblock_layouts: vec![],
        datalayout: None,
        function_constants: vec![],
    }
}

/// Minimal SPIR-V that declares one StorageBuffer variable at `binding`, with
/// pointer provenance reaching a leaf (mirrors spirv_bind's own fixture).
#[cfg(feature = "backend-vulkan")]
fn storage_buffer_spirv(binding: u32) -> Vec<u32> {
    const OP_VARIABLE: u32 = 59;
    const OP_DECORATE: u32 = 71;
    const OP_ACCESS_CHAIN: u32 = 65;
    const STORAGE_CLASS_STORAGE_BUFFER: u32 = 12;
    const DECORATION_BINDING: u32 = 33;
    let mut w = vec![0x0723_0203, 0x0001_0000, 0, 12, 0];
    w.extend([(4 << 16) | OP_VARIABLE, 1, 2, STORAGE_CLASS_STORAGE_BUFFER]);
    w.extend([(4 << 16) | OP_DECORATE, 2, DECORATION_BINDING, binding]);
    w.extend([(5 << 16) | OP_ACCESS_CHAIN, 3, 4, 2, 5]);
    w
}

/// Regression: WebKit's glyph vertex shader declares stride-48 stage-in on
/// buffer 1 but never reads it as an attribute — it indexes the same buffer
/// as a per-glyph `StorageBuffer` (binding 1) by gl_InstanceIndex. Skipping
/// every stage-in buffer left binding 1 unbound, so glyphs collapsed to
/// zero-area quads (blank Safari body text). A stage-in buffer whose binding
/// the vertex SPIR-V declares as a StorageBuffer must still be bound.
#[cfg(feature = "backend-vulkan")]
/// `swap_rb_channels` must be byte-identical to the `src.to_vec()` +
/// in-place `chunks_exact_mut(4)` swizzle it replaces — including the tail
/// (a non-multiple-of-4 remainder copied through unchanged) — and be its own
/// inverse (BGRA<->RGBA).
#[test]
fn swap_rb_channels_matches_two_pass_and_preserves_tail() {
    fn two_pass(src: &[u8]) -> Vec<u8> {
        let mut v = src.to_vec();
        for px in v.chunks_exact_mut(4) {
            px.swap(0, 2);
        }
        v
    }
    for len in [0usize, 4, 8, 5, 7, 9, 260, 263] {
        let src: Vec<u8> = (0..len).map(|i| (i * 7 + 3) as u8).collect();
        let got = swap_rb_channels(&src);
        assert_eq!(
            got,
            two_pass(&src),
            "len={len} must match the two-pass idiom"
        );
        assert_eq!(got.len(), src.len(), "len={len} length preserved");
        // Round-trip: swapping twice restores the original bytes exactly.
        assert_eq!(
            swap_rb_channels(&got),
            src,
            "len={len} swap is its own inverse"
        );
    }
}

/// `reorder_rb_in_place` must touch nothing when the order it holds is already
/// the order asked for, and match `swap_rb_channels` when it is not.
///
/// The no-op half is the whole point of threading the order rather than
/// normalizing: a type-11 composite Store's readback now arrives BGRA, so this is
/// the call that used to be a 776 us whole-frame pass and is now a compare. A
/// future edit that made it exchange unconditionally would restore that cost
/// silently — the pixels would still be right.
#[test]
fn reorder_rb_in_place_is_a_no_op_when_the_orders_already_agree() {
    for len in [0usize, 4, 8, 5, 260, 263] {
        let src: Vec<u8> = (0..len).map(|i| (i * 7 + 3) as u8).collect();
        for order in [false, true] {
            let mut same = src.clone();
            crate::runtime::metal_draw::reorder_rb_in_place(&mut same, order, order);
            assert_eq!(
                same, src,
                "len={len} order={order}: agreement must not copy"
            );
        }
        // Disagreement in either direction is exactly the established swizzle,
        // tail included.
        let mut to_bgra = src.clone();
        crate::runtime::metal_draw::reorder_rb_in_place(&mut to_bgra, false, true);
        assert_eq!(to_bgra, swap_rb_channels(&src), "len={len} rgba->bgra");
        let mut to_rgba = src.clone();
        crate::runtime::metal_draw::reorder_rb_in_place(&mut to_rgba, true, false);
        assert_eq!(to_rgba, swap_rb_channels(&src), "len={len} bgra->rgba");
    }
}

/// Vulkan-arm only: SPIR-V storage-binding reflection has no Metal analogue.
#[cfg(feature = "backend-vulkan")]
#[test]
fn stage_in_buffer_read_as_ssbo_is_bound_as_storage() {
    let words = storage_buffer_spirv(1);
    // Buffer 1 is stage-in AND read as SSBO -> must be exposed as storage.
    assert!(vertex_buffer_needs_storage_binding(&words, 1, true));
    // A plain non-stage-in buffer is always storage.
    assert!(vertex_buffer_needs_storage_binding(&words, 2, false));
    // A stage-in buffer the shader does NOT read as an SSBO stays stage-in only.
    assert!(!vertex_buffer_needs_storage_binding(&words, 3, true));
}

/// Resident GVA chain wiring: the identity is built only for GVA color0
/// (never type-11), with req dims preferred over attachment dims.
#[cfg(feature = "backend-vulkan")]
#[test]
fn gva_chain_identity_rules() {
    use crate::backend::vulkan::engine::TargetIdentity;
    let mut req = DrawEncodeRequest {
        width: 64,
        height: 32,
        ..Default::default()
    };
    assert_eq!(gva_chain_identity(&req), None, "no colors → no identity");
    req.colors.push(ColorRtRequest {
        slot: 0,
        texture_ref: 9,
        mapping_id: 0,
        target_gva: 0x1234_0000,
        width: 16,
        height: 16,
        ..Default::default()
    });
    assert_eq!(
        gva_chain_identity(&req),
        Some(TargetIdentity::Gva {
            gva: 0x1234_0000,
            width: 64,
            height: 32,
            generation: 0,
        }),
        "req dims win when nonzero"
    );
    req.width = 0;
    req.height = 0;
    assert_eq!(
        gva_chain_identity(&req).map(|i| (i.width(), i.height())),
        Some((16, 16)),
        "fallback to color0 dims"
    );
    req.colors[0].mapping_id = 5;
    assert_eq!(
        gva_chain_identity(&req),
        None,
        "type-11 targets never take the GVA identity"
    );
    req.colors[0].mapping_id = 0;
    req.colors[0].target_gva = 0;
    assert_eq!(gva_chain_identity(&req), None, "gva=0 → no identity");
}

#[cfg(feature = "backend-vulkan")]
#[test]
fn render_chain_identity_covers_type11_and_gva_targets() {
    use crate::backend::vulkan::engine::TargetIdentity;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    assert!(state.map_surface(5));
    let mut req = DrawEncodeRequest {
        width: 64,
        height: 32,
        ..Default::default()
    };
    req.colors.push(ColorRtRequest {
        slot: 0,
        texture_ref: 9,
        mapping_id: 5,
        width: 64,
        height: 32,
        ..Default::default()
    });
    assert!(matches!(
        render_chain_identity(&state, &req),
        Some(TargetIdentity::Surface {
            id: 5,
            width: 64,
            height: 32,
            ..
        })
    ));

    req.colors[0].mapping_id = 0;
    req.colors[0].target_gva = 0x1234_0000;
    assert_eq!(
        render_chain_identity(&state, &req),
        Some(TargetIdentity::Gva {
            gva: 0x1234_0000,
            width: 64,
            height: 32,
            generation: 0,
        })
    );
}

/// The last record of a resident render-pass chain is both the chain's consumer
/// and the packet's guest-visible Store, and it must name the resident it loads
/// from so it can skip its own readback.
///
/// Refusing `chain_from_resident` here cost the entire remaining composite
/// readback population — `t11_keep_chain_from_resident` measured equal to
/// `surface_deferred` in every window of one boot. The assertion that matters is
/// the *equality*: `retarget_render_pass_draw` builds every record of a packet
/// from one attachment template, so the record that loads from the resident is by
/// construction the record that renders into it, and a Store naming a different
/// slot than its own LOAD would pin an image its frame is not in.
#[cfg(feature = "backend-vulkan")]
#[test]
fn a_chained_composite_store_names_the_resident_it_loads_from() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    assert!(state.map_surface(7));
    let mut req = DrawEncodeRequest {
        width: 128,
        height: 64,
        ..Default::default()
    };
    req.colors.push(ColorRtRequest {
        slot: 0,
        texture_ref: 3,
        mapping_id: 7,
        width: 128,
        height: 64,
        load_action: PASS_LOAD_ACTION_LOAD,
        store_action: PASS_STORE_ACTION_STORE,
        ..Default::default()
    });

    let unchained = type11_store_identity(&state, &req, true);
    assert!(
        unchained.is_some(),
        "an unchained composite Store resolves its resident"
    );

    req.chain_from_resident = true;
    assert_eq!(
        type11_store_identity(&state, &req, true),
        unchained,
        "a chained Store must name the same resident an unchained one does — it is \
         the same attachment template, so the chain cannot move the slot"
    );
    assert_eq!(
        type11_store_identity(&state, &req, true),
        render_chain_identity(&state, &req),
        "the Store identity and the LoadFromTarget identity must be one slot"
    );

    // The gates that are still refusals. `writeback_guest` is the one that
    // separates the packet's last record from its intermediates, and an
    // intermediate has no guest Store to defer.
    assert_eq!(
        type11_store_identity(&state, &req, false),
        None,
        "an intermediate record stores nothing guest-visible"
    );
    req.colors[0].store_action = crate::runtime::decode::render::PASS_STORE_ACTION_DONT_CARE;
    assert_eq!(
        type11_store_identity(&state, &req, true),
        None,
        "a record that discards its target has no frame to defer"
    );
    req.colors[0].store_action = PASS_STORE_ACTION_STORE;
    req.colors[0].mapping_id = 0;
    assert_eq!(
        type11_store_identity(&state, &req, true),
        None,
        "a GVA target is the other rail's; this one requires a mapping"
    );
}

/// An intermediate record renders into the surface resident too, so it must be
/// able to ask whether that image is already current — even though it has no
/// guest Store of its own to defer.
///
/// Keying the LOAD's currency check on the *Store* identity broke this, and the
/// cost was not a lost elision but a loop. Record 1 of a chain has
/// `writeback_guest == false`, so the check never ran; its LOAD fell through to a
/// CPU seed; the seed found the host cache ceded to the resident rail and read the
/// mapping's guest pages; and reading them landed the window the rail had just
/// armed, which advanced the epoch and cost the *next* LOAD its elision too. One
/// boot measured `surface_flush / surface_resident` at 1369/1373 — one flush per
/// arm.
#[cfg(feature = "backend-vulkan")]
#[test]
fn an_intermediate_record_can_still_ask_about_the_resident_it_renders_into() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    assert!(state.map_surface(7));
    let mut req = DrawEncodeRequest {
        width: 128,
        height: 64,
        ..Default::default()
    };
    req.colors.push(ColorRtRequest {
        slot: 0,
        texture_ref: 3,
        mapping_id: 7,
        width: 128,
        height: 64,
        load_action: PASS_LOAD_ACTION_LOAD,
        store_action: PASS_STORE_ACTION_STORE,
        ..Default::default()
    });

    // The query the LOAD actually asks. It takes no `writeback_guest`, so an
    // intermediate and a final record get the same answer by construction — which
    // is the property, and it is structural rather than asserted.
    let (identity, mapping_epoch) = type11_load_currency_query(&state, &req)
        .expect("a LOAD into a mapped type-11 surface is a candidate the resident could serve");
    assert_eq!(
        Some(identity.clone()),
        render_chain_identity(&state, &req),
        "the LOAD must ask about the slot the record actually renders into"
    );
    assert_eq!(
        Some(identity),
        type11_store_identity(&state, &req, true),
        "the Store identity is the same slot, restricted — not a different one"
    );
    assert_eq!(
        mapping_epoch,
        Some(0),
        "a freshly mapped surface has published nothing, and 0 is that value — the \
         `is_some` guard in `type11_resident_is_current` is what keeps it from \
         matching an unstamped slot"
    );
    assert_eq!(
        type11_store_identity(&state, &req, false),
        None,
        "…while only the packet's last record may leave its frame on the resident"
    );

    // The refusals. A LOAD the resident cannot serve must not produce a query at
    // all, or the counters below it would divide all draws instead of candidates.
    req.colors[0].load_action = crate::runtime::decode::render::PASS_LOAD_ACTION_CLEAR;
    assert!(
        type11_load_currency_query(&state, &req).is_none(),
        "a CLEAR has no prior content to be current"
    );
    req.colors[0].load_action = PASS_LOAD_ACTION_LOAD;
    req.colors[0].target_seed_rgba = Some(vec![0u8; 128 * 64 * 4]);
    assert!(
        type11_load_currency_query(&state, &req).is_none(),
        "an explicit seed was already selected by RT provenance"
    );
    req.colors[0].target_seed_rgba = None;
    req.colors[0].store_action = crate::runtime::decode::render::PASS_STORE_ACTION_DONT_CARE;
    assert!(
        type11_load_currency_query(&state, &req).is_none(),
        "a record that discards its target renders into no resident worth naming"
    );
    req.colors[0].store_action = PASS_STORE_ACTION_STORE;
    req.colors[0].mapping_id = 0;
    assert!(
        type11_load_currency_query(&state, &req).is_none(),
        "a GVA target is the other rail's"
    );
}

/// Records 2+ of an armed resident chain bind the attachment alias from
/// the resident target; unarmed LOAD without a seed stays None (guest
/// reload would expose pre-pass bytes — existing contract).
#[cfg(feature = "backend-vulkan")]
#[test]
fn attachment_alias_resident_chain_selection() {
    let mut req = DrawEncodeRequest {
        width: 8,
        height: 8,
        ..Default::default()
    };
    req.colors.push(ColorRtRequest {
        slot: 0,
        texture_ref: 42,
        mapping_id: 0,
        target_gva: 0x9000,
        width: 8,
        height: 8,
        load_action: PASS_LOAD_ACTION_LOAD,
        ..Default::default()
    });
    assert_eq!(
        fragment_attachment_alias_sample(&req, 0, 42),
        None,
        "unarmed LOAD without seed must not alias"
    );
    req.chain_from_resident = true;
    assert_eq!(
        fragment_attachment_alias_sample(&req, 0, 42),
        Some((8, 8, AttachmentAliasSample::ResidentChain)),
        "armed chain aliases from the resident target"
    );
    // CPU seed still wins when present (record after a non-resident hop).
    let seed = vec![0u8; 8 * 8 * 4];
    req.colors[0].target_seed_rgba = Some(seed);
    assert!(matches!(
        fragment_attachment_alias_sample(&req, 0, 42),
        Some((8, 8, AttachmentAliasSample::Seed(_)))
    ));
}

/// A sampled bind takes the deferred window's resident target only for
/// the exact window content: same geometry and an owner gate the
/// post-flush cache layer would also pass. Mismatches must flush.
#[cfg(feature = "backend-vulkan")]
#[test]
fn deferred_gva_sample_eligibility_rules() {
    let win = crate::model::GvaDeferredEntry {
        task_id: 1,
        texture_ref: 7,
        producer_object_type: 2,
        width: 32,
        height: 16,
        row_stride: 128,
        format: MTL_FORMAT_BGRA8_UNORM,
        armed_seq: 0,
        pages: Default::default(),
    };
    assert!(
        deferred_gva_sample_eligible(&win, 32, 16, 2),
        "exact geometry + same object type binds the resident"
    );
    assert!(
        deferred_gva_sample_eligible(&win, 32, 16, 0),
        "unknown sampler type retains the cache-layer owner behavior"
    );
    assert!(
        !deferred_gva_sample_eligible(&win, 16, 16, 2),
        "geometry mismatch must land the window instead"
    );
    assert!(
        !deferred_gva_sample_eligible(&win, 32, 8, 2),
        "height mismatch must land the window instead"
    );
    assert!(
        deferred_gva_sample_eligible(&win, 32, 16, OBJECT_TYPE_TEXTURE_VARIANT),
        "type-2/type-3 wrappers share linear texture storage"
    );
    assert!(
        !deferred_gva_sample_eligible(&win, 32, 16, OBJECT_TYPE_TEXTURE_VIEW),
        "unrelated object-type transitions must land the window instead"
    );
}

#[test]
fn linear_sampled_memo_serves_only_exact_generation_and_geometry() {
    let mut state = DeviceState::new(DeviceId(7), PAGE_SHIFT_ARM64E);
    let rgba = std::sync::Arc::new(vec![9u8, 8, 7, 255]);
    let entry_bytes = rgba.len();
    state.linear_sampled_memo.insert(
        (3, 44),
        crate::model::LinearSampledMemo {
            gva: 0x30_2000,
            host_gen: 5,
            width: 1,
            height: 1,
            rgba: rgba.clone(),
        },
        entry_bytes,
    );
    let hit = linear_sampled_memo_reuse(&state, 3, 44, 0x30_2000, 5, 1, 1)
        .expect("exact match serves the memo");
    assert!(std::sync::Arc::ptr_eq(&hit, &rgba), "no copy on reuse");
    // Any drifted axis skips the memo: generation, gva, geometry, key.
    assert!(linear_sampled_memo_reuse(&state, 3, 44, 0x30_2000, 6, 1, 1).is_none());
    assert!(linear_sampled_memo_reuse(&state, 3, 44, 0x30_3000, 5, 1, 1).is_none());
    assert!(linear_sampled_memo_reuse(&state, 3, 44, 0x30_2000, 5, 2, 1).is_none());
    assert!(linear_sampled_memo_reuse(&state, 3, 45, 0x30_2000, 5, 1, 1).is_none());
}

#[test]
fn gva_cache_owner_object_type_transitions_are_named() {
    assert!(gva_cache_owner_allows_object_type(
        0,
        OBJECT_TYPE_TEXTURE_VARIANT
    ));
    assert!(gva_cache_owner_allows_object_type(
        OBJECT_TYPE_TEXTURE,
        OBJECT_TYPE_TEXTURE
    ));
    assert!(gva_cache_owner_allows_object_type(
        OBJECT_TYPE_TEXTURE,
        OBJECT_TYPE_TEXTURE_VARIANT
    ));
    assert!(gva_cache_owner_allows_object_type(
        OBJECT_TYPE_TEXTURE_VARIANT,
        OBJECT_TYPE_TEXTURE
    ));
    assert!(!gva_cache_owner_allows_object_type(
        OBJECT_TYPE_TEXTURE,
        OBJECT_TYPE_TEXTURE_VIEW
    ));
}

/// Vulkan-arm only: `AttachmentAliasSample` and its resolver are
/// `backend-vulkan` items.
#[cfg(feature = "backend-vulkan")]
#[test]
fn gva_attachment_alias_samples_the_in_process_chain() {
    let task_id = std::process::id();
    let texture_ref = 0xe000_0000u32.wrapping_add(task_id);
    let target_gva = 0x0abc_d000;
    let seed = vec![10, 0, 0, 255, 0, 0, 0, 0];
    let mut req = DrawEncodeRequest {
        task_id,
        colors: vec![ColorRtRequest {
            slot: 0,
            texture_ref,
            mapping_id: 0,
            target_gva,
            width: 2,
            height: 1,
            load_action: PASS_LOAD_ACTION_LOAD,
            target_seed_rgba: Some(seed.clone()),
            ..Default::default()
        }],
        ..Default::default()
    };

    let (width, height, sample) =
        fragment_attachment_alias_sample(&req, 0, texture_ref).expect("GVA alias");
    assert_eq!((width, height), (2, 1));
    let AttachmentAliasSample::Seed(actual) = sample else {
        panic!("Load alias must use the chained seed");
    };
    assert_eq!(actual, seed);
    assert!(fragment_attachment_alias_sample(&req, 1, texture_ref).is_none());
    assert!(fragment_attachment_alias_sample(&req, 0, texture_ref + 1).is_none());

    req.colors[0].mapping_id = 9;
    assert!(fragment_attachment_alias_sample(&req, 0, texture_ref).is_none());
    req.colors[0].mapping_id = 0;
    req.colors[0].load_action = PASS_LOAD_ACTION_DONT_CARE;
    assert!(fragment_attachment_alias_sample(&req, 0, texture_ref).is_none());
    req.colors[0].load_action = PASS_LOAD_ACTION_CLEAR;
    req.colors[0].clear_color = [0.25, 0.5, 0.75, 1.0];
    assert_eq!(
        fragment_attachment_alias_sample(&req, 0, texture_ref),
        Some((2, 1, AttachmentAliasSample::Clear([0.25, 0.5, 0.75, 1.0])))
    );
}

#[test]
fn tight_linear_load_uses_one_bulk_read_and_converts_rows() {
    let mut calls = 0;
    let (rgba, fmt) = load_tight_linear_rgba_with(2, 2, MTL_FORMAT_BGRA8_UNORM, false, |native| {
        calls += 1;
        assert_eq!(native.len(), 16);
        native.copy_from_slice(&[3, 2, 1, 255, 6, 5, 4, 255, 9, 8, 7, 255, 12, 11, 10, 255]);
        true
    })
    .expect("tight sample loads");

    assert_eq!(calls, 1);
    assert_eq!(fmt, TexelLayout::Rgba8);
    assert_eq!(
        rgba,
        [1, 2, 3, 255, 4, 5, 6, 255, 7, 8, 9, 255, 10, 11, 12, 255,]
    );
}

/// A native BGRA8 upload keeps the guest bytes verbatim (no CPU channel
/// swap) and reports `Bgra8` so the engine binds a BGRA8 image — the
/// Safari-scroll fallback hot path. Same read count as the swizzled path.
#[test]
fn tight_linear_native_bgra8_keeps_bytes_and_reports_bgra8() {
    let bgra = [3, 2, 1, 255, 6, 5, 4, 255, 9, 8, 7, 255, 12, 11, 10, 255];
    let mut calls = 0;
    let (bytes, fmt) = load_tight_linear_rgba_with(2, 2, MTL_FORMAT_BGRA8_UNORM, true, |native| {
        calls += 1;
        native.copy_from_slice(&bgra);
        true
    })
    .expect("tight native sample loads");
    assert_eq!(calls, 1);
    assert_eq!(fmt, TexelLayout::Bgra8);
    assert_eq!(bytes, bgra, "native BGRA8 upload must not swizzle");
}

#[test]
fn tight_rgba_linear_load_preserves_native_bytes() {
    let native = [1, 2, 3, 4, 5, 6, 7, 8];
    let (rgba, fmt) =
        load_tight_linear_rgba_with(2, 1, pixel_format::MTL_FORMAT_RGBA8_UNORM, false, |dst| {
            dst.copy_from_slice(&native);
            true
        })
        .expect("tight RGBA sample loads");
    assert_eq!(fmt, TexelLayout::Rgba8);
    assert_eq!(rgba, native);
}

/// **The sRGB-fold regression gate for the CPU upload rails.** These two
/// paths reach a linear byte layout from an sRGB Metal format, which is the
/// right layout — the two share one — but silently lost the transfer
/// function until the census was wired in. The layout must stay identical
/// to the linear sibling's *and* the loss must be counted; either half
/// alone is the old bug.
#[test]
fn the_cpu_upload_rails_count_every_srgb_downgrade() {
    use crate::runtime::census::srgb_census;
    srgb_census::reset_for_tests();
    // The sink is append-only and shared with every other test in the binary,
    // so this asserts a delta, not an absolute count.
    // Count LINES, not substring hits: the slug appears twice per line, once as
    // the event prefix and once as `reason=`.
    let downgrade_lines = || {
        std::fs::read_to_string(crate::observe::fail_log_path())
            .map(|l| {
                l.lines()
                    .filter(|l| l.starts_with("srgb_downgraded "))
                    .count()
            })
            .unwrap_or(0)
    };
    let before = downgrade_lines();

    // Native-upload rail: sRGB resolves exactly as its linear sibling.
    assert_eq!(
        linear_native_upload_format(pixel_format::MTL_FORMAT_RGBA8_UNORM_SRGB, false),
        linear_native_upload_format(pixel_format::MTL_FORMAT_RGBA8_UNORM, false),
    );
    assert_eq!(
        linear_native_upload_format(pixel_format::MTL_FORMAT_BGRA8_UNORM_SRGB, true),
        Some(TexelLayout::Bgra8),
    );
    // Tight-load rail: same layout, and the BGRA swap still happens when
    // the caller did not opt into a native BGRA8 upload.
    let native = [1u8, 2, 3, 4, 5, 6, 7, 8];
    let (bytes, fmt) = load_tight_linear_rgba_with(
        2,
        1,
        pixel_format::MTL_FORMAT_BGRA8_UNORM_SRGB,
        false,
        |dst| {
            dst.copy_from_slice(&native);
            true
        },
    )
    .expect("tight sRGB BGRA sample loads");
    assert_eq!(fmt, TexelLayout::Rgba8);
    assert_eq!(
        bytes,
        [3, 2, 1, 4, 7, 6, 5, 8],
        "channel swap still applied"
    );

    // Three distinct (site, format) pairs were downgraded above — two on the
    // native-upload rail (RGBA8 and BGRA8 sRGB) and one on the tight-load rail
    // — so the sink must carry three lines and name both rails. Read off the
    // log rather than a counter: the line is what a boot has to show.
    let log = std::fs::read_to_string(crate::observe::fail_log_path()).expect("fail log");
    assert_eq!(
        downgrade_lines() - before,
        3,
        "every downgrade named, none swallowed"
    );
    assert!(log.contains(&format!("site={}", srgb_census::site::LINEAR_NATIVE_UPLOAD)));
    assert!(log.contains(&format!("site={}", srgb_census::site::TIGHT_LINEAR_LOAD)));

    // A linear source must never touch the census, or the proxy floods and
    // stops distinguishing anything.
    srgb_census::reset_for_tests();
    let _ = linear_native_upload_format(pixel_format::MTL_FORMAT_RGBA8_UNORM, false);
    let _ = load_tight_linear_rgba_with(2, 1, pixel_format::MTL_FORMAT_BGRA8_UNORM, true, |dst| {
        dst.copy_from_slice(&native);
        true
    });
    assert_eq!(
        downgrade_lines() - before,
        3,
        "a linear source must add no line"
    );
    srgb_census::reset_for_tests();
}

#[test]
fn color_target_diag_names_every_mrt_slot() {
    let colors = vec![
        ColorRtRequest {
            slot: 0,
            texture_ref: 11,
            mapping_id: 1,
            width: 1920,
            height: 1080,
            format: MTL_FORMAT_BGRA8_UNORM,
            load_action: PASS_LOAD_ACTION_LOAD,
            store_action: PASS_STORE_ACTION_STORE,
            ..Default::default()
        },
        ColorRtRequest {
            slot: 2,
            texture_ref: 17,
            target_gva: 0x1234_5000,
            width: 960,
            height: 540,
            format: pixel_format::MTL_FORMAT_RGBA16_FLOAT,
            load_action: PASS_LOAD_ACTION_CLEAR,
            store_action: PASS_STORE_ACTION_STORE,
            ..Default::default()
        },
    ];
    assert_eq!(
        color_target_diag(&colors),
        "s0:r11:mid1:gva=0x0:1920x1080:fmt=0x50:l1:s1,\
s2:r17:mid0:gva=0x12345000:960x540:fmt=0x73:l2:s1"
    );
}

/// Vulkan-arm only: `binding_hex_prefix` is a `backend-vulkan` fn.
#[cfg(feature = "backend-vulkan")]
#[test]
fn binding_hex_prefix_selects_binding_and_bounds_output() {
    let storage = vec![
        (1, vec![0xaa; 3].into()),
        (4, (0u8..80).collect::<Vec<u8>>().into()),
    ];
    assert_eq!(binding_hex_prefix(&storage, 4, 4), "00010203");
    assert_eq!(binding_hex_prefix(&storage, 1, 64), "aaaaaa");
    assert_eq!(binding_hex_prefix(&storage, 9, 64), "");
}

#[test]
fn missing_pipeline_is_soft() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    assert!(state.define_task(1, 0x1000, 2));
    let mut host = FakeHost::new();
    let req = DrawEncodeRequest {
        task_id: 1,
        pipeline_ref: 99,
        color_texture_ref: 1,
        mapping_id: 3,
        width: 4,
        height: 4,
        format: MTL_FORMAT_BGRA8_UNORM,
        vertex_count: 3,
        instance_count: 1,
        primitive_type: 3, // triangle
        first_vertex: 0,
        target_seed_rgba: None,
        ..Default::default()
    };
    let mut req = req;
    let st = encode_draw_and_writeback(&mut state, &mut host, &mut req);
    assert!(matches!(
        st,
        EncodeStatus::MissingPipeline(_) | EncodeStatus::MissingMtlb(_) | EncodeStatus::NoMetal(_)
    ));
    let _ = pixel_format::RGBA8_BPP;
}

#[test]
fn buffer_bind_slot_count() {
    assert_eq!(MAX_BIND_SLOTS, 31);
    // No product byte-size budget: host_alloc_len only rejects >usize/isize.
    assert_eq!(host_alloc_len(64 << 20), Some(64 << 20));
    assert_eq!(host_alloc_len(0), Some(0));
}

#[cfg(feature = "backend-vulkan")]
#[test]
fn vulkan_sampler_missing_entry_returns_exact_decline() {
    let state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    let host = FakeHost::new();
    let error = load_vulkan_sampler(&state, &host, 7, 11, 64)
        .expect_err("an empty object list cannot resolve sampler 11");
    assert_eq!(error.slug(), "draw_prepare_sampler_entry_missing");
    assert_eq!(
        error.fields(),
        vec![("sampler_ref", "11".into()), ("binding", "64".into()),]
    );
}

#[cfg(feature = "backend-vulkan")]
#[test]
fn vertex_attribute_preparation_returns_exact_declines() {
    use crate::runtime::decode::resource::VertexAttribute;

    let mut attribute = VertexAttribute {
        location: 3,
        format: 99,
        buffer_index: 2,
        stride: 16,
        ..VertexAttribute::default()
    };
    let format = prepare_vertex_attribute_format(&attribute)
        .expect_err("unknown MTLVertexFormat must be typed before request validation");
    assert_eq!(format.slug(), "draw_prepare_vertex_attribute_format");
    assert_eq!(
        format.fields(),
        vec![
            ("location", "3".into()),
            ("buffer_index", "2".into()),
            ("raw_format", "99".into()),
            ("value", "99".into()),
        ]
    );

    attribute.format = 30;
    attribute.has_step_function = true;
    attribute.step_function = 9;
    let step = prepare_vertex_step_function(&attribute)
        .expect_err("unknown MTLVertexStepFunction must be typed before request validation");
    assert_eq!(step.slug(), "draw_prepare_vertex_step_function_unsupported");
    assert_eq!(
        step.fields(),
        vec![
            ("location", "3".into()),
            ("buffer_index", "2".into()),
            ("value", "9".into()),
        ]
    );
}

#[cfg(feature = "backend-vulkan")]
#[test]
fn vulkan_sampler_preserves_guest_coordinate_and_filter_state() {
    use crate::backend::vulkan::engine::{
        SamplerAddressMode, SamplerBorderColor, SamplerCompareFunction, SamplerFilter,
        SamplerMipFilter,
    };
    use crate::runtime::decode::resource::SamplerDescriptor;

    let decoded = SamplerDescriptor {
        min_filter: 0,
        mag_filter: 1,
        mip_filter: 2,
        s_address: 2,
        t_address: 3,
        r_address: 5,
        max_anisotropy: 4,
        lod_min_clamp: 1.25,
        lod_max_clamp: 7.5,
        compare_function: 3,
        border_color: 2,
        normalized_coordinates: false,
        support_argument_buffers: false,
        lod_average: false,
    };
    let sampler = vulkan_sampler_resource(5, 67, &decoded).expect("supported sampler");

    assert_eq!(sampler.binding, 67);
    assert_eq!(sampler.min_filter, SamplerFilter::Nearest);
    assert_eq!(sampler.mag_filter, SamplerFilter::Linear);
    assert_eq!(sampler.mip_filter, SamplerMipFilter::Linear);
    assert_eq!(sampler.address_mode_u, SamplerAddressMode::Repeat);
    assert_eq!(sampler.address_mode_v, SamplerAddressMode::MirrorRepeat);
    assert_eq!(
        sampler.address_mode_w,
        SamplerAddressMode::ClampToBorderColor
    );
    assert_eq!(sampler.border_color, SamplerBorderColor::OpaqueWhite);
    assert_eq!(sampler.compare_function, SamplerCompareFunction::LessEqual);
    assert_eq!(sampler.lod_min, 1.25f32.to_bits());
    assert_eq!(sampler.lod_max, 7.5f32.to_bits());
    assert_eq!(sampler.max_anisotropy, 4);
    assert!(sampler.unnormalized_coordinates);

    let mut bad = decoded;
    bad.min_filter = 9;
    let min = vulkan_sampler_resource(5, 67, &bad).expect_err("unknown min filter");
    assert_eq!(min.slug(), "draw_prepare_sampler_min_filter_translation");
    assert_eq!(
        min.fields(),
        vec![
            ("sampler_ref", "5".into()),
            ("binding", "67".into()),
            ("value", "9".into()),
        ]
    );

    bad.min_filter = 0;
    bad.mag_filter = 9;
    let mag = vulkan_sampler_resource(5, 67, &bad).expect_err("unknown mag filter");
    assert_eq!(mag.slug(), "draw_prepare_sampler_mag_filter_translation");
}

/// qemu-shim Store policy: Clear/DontCare/force_full full-write; Load+seed
/// may diff-only. Prevents Clear+partial logo-mid residual.
#[test]
fn store_seed_policy_clear_full_load_diff() {
    let seed = [1u8, 2, 3, 4];
    assert!(store_seed_policy(false, PASS_LOAD_ACTION_CLEAR, Some(&seed)).is_none());
    assert!(store_seed_policy(false, PASS_LOAD_ACTION_DONT_CARE, Some(&seed)).is_none());
    assert!(store_seed_policy(true, PASS_LOAD_ACTION_LOAD, Some(&seed)).is_none());
    assert_eq!(
        store_seed_policy(false, PASS_LOAD_ACTION_LOAD, Some(&seed)),
        Some(seed.as_slice())
    );
    assert!(store_seed_policy(false, PASS_LOAD_ACTION_LOAD, None).is_none());
}

/// Premult One/OneMinusSrcAlpha Load: transparent draw keeps seed; opaque black wins.
#[test]
fn load_composite_premult_restores_seed_under_transparent() {
    let mut draw = vec![0u8; 8];
    draw[0..4].copy_from_slice(&[255, 255, 255, 255]); // chrome
    draw[4..8].copy_from_slice(&[0, 0, 0, 0]); // uncovered (clear A=0)
    let mut seed = vec![0u8; 8];
    seed[0..4].copy_from_slice(&[203, 203, 203, 255]);
    seed[4..8].copy_from_slice(&[203, 203, 203, 255]);
    let (out, blended) = load_composite_premult_one_omsa(&draw, &seed);
    assert_eq!(blended, 1);
    assert_eq!(&out[0..4], &[255, 255, 255, 255]);
    assert_eq!(&out[4..8], &[203, 203, 203, 255], "A=0 keeps Load seed");
    draw[4..8].copy_from_slice(&[0, 0, 0, 255]);
    let (out2, _) = load_composite_premult_one_omsa(&draw, &seed);
    assert_eq!(&out2[4..8], &[0, 0, 0, 255], "opaque black stays black");
}

#[test]
fn a8_sample_preserves_alpha_coverage() {
    let native = [0, 17, 255];
    let (rgba, fmt) =
        load_tight_linear_rgba_with(3, 1, pixel_format::MTL_FORMAT_A8_UNORM, true, |dst| {
            dst.copy_from_slice(&native);
            true
        })
        .expect("A8 sample loads");
    assert_eq!(
        fmt,
        TexelLayout::Rgba8,
        "A8 needs a real convert; native flag does not apply"
    );
    assert_eq!(
        rgba,
        [0, 0, 0, 0, 0, 0, 0, 17, 0, 0, 0, 255],
        "A8 has no RGB channels; its alpha is the sampled mask payload"
    );
}

/// Metal blend factors/ops must map into engine blend types (Linux path was silent-None).
#[cfg(feature = "backend-vulkan")]
#[test]
fn blend_state_maps_src_alpha_one_minus() {
    let b = translate::blend::state(
        4, // SrcAlpha
        5, // OneMinusSrcAlpha
        0, // Add
        1, // One
        5, // OneMinusSrcAlpha
        0, // Add
        [0.0; 4],
    )
    .expect("map");
    assert_eq!(
        b.src_color,
        crate::backend::vulkan::engine::BlendFactor::SrcAlpha
    );
    assert_eq!(
        b.dst_color,
        crate::backend::vulkan::engine::BlendFactor::OneMinusSrcAlpha
    );
    assert_eq!(b.color_op, crate::backend::vulkan::engine::BlendOp::Add);
    assert_eq!(
        b.src_alpha,
        crate::backend::vulkan::engine::BlendFactor::One
    );
    assert!(translate::blend::factor(99).is_err());
    assert!(translate::blend::operation(9).is_err());
}

/// qemu-shim: guest Load with unresolvable type-11 pages still encodes
/// (archive NULL seed / Metal Clear invent) — does not drop the pass.
#[test]
fn mrt_draw_request_load_seed_miss_still_encodes() {
    use crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    use crate::runtime::decode::render::ColorAttachment;
    use crate::runtime::host::FakeHost;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    assert!(state.define_task(1, 0x1000, 2));
    // Type-11 registered with geom but empty page table → seed read fails.
    assert!(state.map_surface(9));
    assert!(state.set_mapping_geom(9, 8, 8, MTL_FORMAT_BGRA8_UNORM));
    // gen must be non-zero for Load path to attempt a snapshot (archive).
    state.mappings.get_mut(&9).unwrap().content_generation = 1;
    state.texture_to_mapping.insert((1, 42), 9);
    let att = ColorAttachment {
        present: true,
        texture_ref: 42,
        resolve_texture_ref: 0,
        level: 0,
        load_action: PASS_LOAD_ACTION_LOAD,
        store_action: PASS_STORE_ACTION_STORE,
        clear_color: [1.0, 1.0, 1.0, 1.0], // would paint solid white if Clear invented
    };
    let slots = [(0u32, att)];
    let req = mrt_draw_request(&mut state, &mut host, 1, 1, &slots, &[], 3, 1, 3, 0);
    // Archive: seed miss still builds the job (NULL seed). Product must not
    // drop the pass — that freezes lagging dual-mid on stale logo.
    let req = req.expect("Load seed miss must still encode (archive NULL seed)");
    assert!(
        req.target_seed_rgba.is_none()
            || req
                .colors
                .first()
                .map(|c| c.target_seed_rgba.is_none())
                .unwrap_or(true),
        "seed miss leaves seed None (Metal Clear invent, full Store)"
    );
    assert_eq!(req.colors[0].load_action, PASS_LOAD_ACTION_LOAD);
}

/// qemu-shim: type-8 view of type-11 is a valid color RT (archive
/// resource_resolve_texture view chain). Without this, App Store UI pipes
/// that bind a view as color attachment drop the entire MRT pass.
#[test]
fn mrt_draw_request_type8_view_of_type11_as_color_rt() {
    use crate::contract::endian::{st16, st32, st64};
    use crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    use crate::runtime::decode::render::ColorAttachment;
    use crate::runtime::decode::resource::{
        list_object_entry_offset, OBJECT_LIST_ENTRY_LEN, OBJECT_TYPE_TEXTURE_VIEW,
        TEXTURE_VIEW_DESC_BASE_REF, TEXTURE_VIEW_DESC_LEN, TEXTURE_VIEW_DESC_LEVEL_BASE,
        TEXTURE_VIEW_DESC_LEVEL_COUNT, TEXTURE_VIEW_DESC_OPCODE, TEXTURE_VIEW_DESC_PIXEL_FORMAT,
        TEXTURE_VIEW_DESC_SLICE_BASE, TEXTURE_VIEW_DESC_SLICE_COUNT, TEXTURE_VIEW_DESC_TEXTURE_REF,
        TEXTURE_VIEW_DESC_TEXTURE_TYPE, TEXTURE_VIEW_MIN_RANGED, TEXTURE_VIEW_MTL_TYPE_2D,
        TEXTURE_VIEW_OPCODE_RANGED,
    };
    use crate::runtime::host::FakeHost;

    // One-level page table: GVA pages 0..7 → data PFNs (blit_exec pattern).

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    gva_mem::define_task_pages_arm64e(&mut host, &mut state, 4, 8);
    // Object list at GVA page 0; count covers live residual slot 211.
    assert!(state.set_object_list(1, 0, 256));

    // Base type-11 mid 9 latched as texture ref 3.
    assert!(state.map_surface(9));
    assert!(state.set_mapping_geom(9, 64, 64, MTL_FORMAT_BGRA8_UNORM));
    state.mappings.get_mut(&9).unwrap().content_generation = 1;
    state.texture_to_mapping.insert((1, 3), 9);

    // Type-8 view ref 211 → base 3 (identity, level 0) — live residual slot.
    let view_ref = 211u32;
    let base_ref = 3u32;
    let len = TEXTURE_VIEW_MIN_RANGED;
    let mut desc = vec![0u8; len];
    st32(
        &mut desc[TEXTURE_VIEW_DESC_OPCODE..],
        TEXTURE_VIEW_OPCODE_RANGED,
    );
    st32(&mut desc[TEXTURE_VIEW_DESC_LEN..], len as u32);
    st32(&mut desc[TEXTURE_VIEW_DESC_TEXTURE_REF..], view_ref);
    st32(&mut desc[TEXTURE_VIEW_DESC_BASE_REF..], base_ref);
    st16(
        &mut desc[TEXTURE_VIEW_DESC_PIXEL_FORMAT..],
        MTL_FORMAT_BGRA8_UNORM,
    );
    st16(
        &mut desc[TEXTURE_VIEW_DESC_TEXTURE_TYPE..],
        TEXTURE_VIEW_MTL_TYPE_2D,
    );
    st64(&mut desc[TEXTURE_VIEW_DESC_LEVEL_BASE..], 0);
    st64(&mut desc[TEXTURE_VIEW_DESC_LEVEL_COUNT..], 1);
    st64(&mut desc[TEXTURE_VIEW_DESC_SLICE_BASE..], 0);
    st64(&mut desc[TEXTURE_VIEW_DESC_SLICE_COUNT..], 1);
    let desc_gva = 0x280u64;
    write_task_gva_arm64e(&mut host, &state.tasks[1], desc_gva, &desc);
    let off = list_object_entry_offset(view_ref, 256).unwrap();
    let mut list_entry = [0u8; OBJECT_LIST_ENTRY_LEN];
    let packed = (OBJECT_TYPE_TEXTURE_VIEW as u32) | ((len as u32) << 8);
    st32(&mut list_entry[0..], packed);
    list_entry[4..12].copy_from_slice(&desc_gva.to_le_bytes());
    write_task_gva_arm64e(&mut host, &state.tasks[1], off, &list_entry);

    let att = ColorAttachment {
        present: true,
        texture_ref: view_ref,
        resolve_texture_ref: 0,
        level: 0,
        load_action: PASS_LOAD_ACTION_CLEAR,
        store_action: PASS_STORE_ACTION_STORE,
        clear_color: [0.0, 0.0, 0.0, 1.0],
    };
    let req = mrt_draw_request(
        &mut state,
        &mut host,
        1,
        12,
        &[(0u32, att)],
        &[],
        3,
        1,
        3,
        0,
    )
    .expect("type-8 view of type-11 must resolve as color RT");
    assert_eq!(req.colors[0].mapping_id, 9);
    assert_eq!(req.colors[0].width, 64);
    assert_eq!(req.colors[0].height, 64);
    assert_eq!(req.colors[0].texture_ref, view_ref);
}

/// Archive `REIMS_VGPU_RESOURCE_RESOLVE_MAX_VIEW_CHAIN`: nested type-8 → type-8 →
/// type-11 must collapse to the non-view base. One-hop resolve left the mid
/// base as type-8 and dropped the MRT pass (`view_base_or_swizzle`).
#[test]
fn mrt_draw_request_nested_type8_view_chain_to_type11() {
    use crate::contract::endian::{st16, st32, st64};
    use crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    use crate::runtime::decode::render::ColorAttachment;
    use crate::runtime::decode::resource::{
        list_object_entry_offset, OBJECT_LIST_ENTRY_LEN, OBJECT_TYPE_TEXTURE_VIEW,
        TEXTURE_VIEW_DESC_BASE_REF, TEXTURE_VIEW_DESC_LEN, TEXTURE_VIEW_DESC_LEVEL_BASE,
        TEXTURE_VIEW_DESC_LEVEL_COUNT, TEXTURE_VIEW_DESC_OPCODE, TEXTURE_VIEW_DESC_PIXEL_FORMAT,
        TEXTURE_VIEW_DESC_SLICE_BASE, TEXTURE_VIEW_DESC_SLICE_COUNT, TEXTURE_VIEW_DESC_TEXTURE_REF,
        TEXTURE_VIEW_DESC_TEXTURE_TYPE, TEXTURE_VIEW_MIN_RANGED, TEXTURE_VIEW_MTL_TYPE_2D,
        TEXTURE_VIEW_OPCODE_RANGED,
    };
    use crate::runtime::host::FakeHost;

    fn write_type8_view(
        host: &mut FakeHost,
        state: &DeviceState,
        view_ref: u32,
        base_ref: u32,
        desc_gva: u64,
    ) {
        let len = TEXTURE_VIEW_MIN_RANGED;
        let mut desc = vec![0u8; len];
        st32(
            &mut desc[TEXTURE_VIEW_DESC_OPCODE..],
            TEXTURE_VIEW_OPCODE_RANGED,
        );
        st32(&mut desc[TEXTURE_VIEW_DESC_LEN..], len as u32);
        st32(&mut desc[TEXTURE_VIEW_DESC_TEXTURE_REF..], view_ref);
        st32(&mut desc[TEXTURE_VIEW_DESC_BASE_REF..], base_ref);
        st16(
            &mut desc[TEXTURE_VIEW_DESC_PIXEL_FORMAT..],
            MTL_FORMAT_BGRA8_UNORM,
        );
        st16(
            &mut desc[TEXTURE_VIEW_DESC_TEXTURE_TYPE..],
            TEXTURE_VIEW_MTL_TYPE_2D,
        );
        st64(&mut desc[TEXTURE_VIEW_DESC_LEVEL_BASE..], 0);
        st64(&mut desc[TEXTURE_VIEW_DESC_LEVEL_COUNT..], 1);
        st64(&mut desc[TEXTURE_VIEW_DESC_SLICE_BASE..], 0);
        st64(&mut desc[TEXTURE_VIEW_DESC_SLICE_COUNT..], 1);
        write_task_gva_arm64e(&mut *host, &state.tasks[1], desc_gva, &desc);
        let off = list_object_entry_offset(view_ref, 256).unwrap();
        let mut list_entry = [0u8; OBJECT_LIST_ENTRY_LEN];
        let packed = (OBJECT_TYPE_TEXTURE_VIEW as u32) | ((len as u32) << 8);
        st32(&mut list_entry[0..], packed);
        list_entry[4..12].copy_from_slice(&desc_gva.to_le_bytes());
        write_task_gva_arm64e(&mut *host, &state.tasks[1], off, &list_entry);
    }

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    gva_mem::define_task_pages_arm64e(&mut host, &mut state, 4, 8);
    assert!(state.set_object_list(1, 0, 256));

    // type-11 mid 9 as texture ref 3.
    assert!(state.map_surface(9));
    assert!(state.set_mapping_geom(9, 64, 64, MTL_FORMAT_BGRA8_UNORM));
    state.mappings.get_mut(&9).unwrap().content_generation = 1;
    state.texture_to_mapping.insert((1, 3), 9);

    // Inner view 8 → base 3 (type-11); outer view 211 → base 8 (type-8).
    write_type8_view(&mut host, &state, 8, 3, 0x280);
    write_type8_view(&mut host, &state, 211, 8, 0x300);

    let att = ColorAttachment {
        present: true,
        texture_ref: 211,
        resolve_texture_ref: 0,
        level: 0,
        load_action: PASS_LOAD_ACTION_CLEAR,
        store_action: PASS_STORE_ACTION_STORE,
        clear_color: [0.0, 0.0, 0.0, 1.0],
    };
    let req = mrt_draw_request(
        &mut state,
        &mut host,
        1,
        12,
        &[(0u32, att)],
        &[],
        3,
        1,
        3,
        0,
    )
    .expect("nested type-8→type-8→type-11 must resolve as color RT");
    assert_eq!(req.colors[0].mapping_id, 9);
    assert_eq!(req.colors[0].width, 64);
    assert_eq!(req.colors[0].height, 64);
    assert_eq!(req.colors[0].texture_ref, 211);
}

/// Archive resolve_texture rejects non-identity swizzle for RT resolve.
#[test]
fn mrt_draw_request_type8_swizzled_view_rejected_as_color_rt() {
    use crate::contract::endian::{st16, st32, st64};
    use crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    use crate::runtime::decode::render::ColorAttachment;
    use crate::runtime::decode::resource::{
        list_object_entry_offset, OBJECT_LIST_ENTRY_LEN, OBJECT_TYPE_TEXTURE_VIEW,
        TEXTURE_VIEW_DESC_BASE_REF, TEXTURE_VIEW_DESC_LEN, TEXTURE_VIEW_DESC_LEVEL_BASE,
        TEXTURE_VIEW_DESC_LEVEL_COUNT, TEXTURE_VIEW_DESC_OPCODE, TEXTURE_VIEW_DESC_PIXEL_FORMAT,
        TEXTURE_VIEW_DESC_SLICE_BASE, TEXTURE_VIEW_DESC_SLICE_COUNT, TEXTURE_VIEW_DESC_SWIZZLE,
        TEXTURE_VIEW_DESC_TEXTURE_REF, TEXTURE_VIEW_DESC_TEXTURE_TYPE, TEXTURE_VIEW_MIN_SWIZZLE,
        TEXTURE_VIEW_MTL_TYPE_2D, TEXTURE_VIEW_OPCODE_SWIZZLE,
    };
    use crate::runtime::host::FakeHost;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    gva_mem::define_task_pages_arm64e(&mut host, &mut state, 4, 8);
    assert!(state.set_object_list(1, 0, 32));
    assert!(state.map_surface(9));
    assert!(state.set_mapping_geom(9, 64, 64, MTL_FORMAT_BGRA8_UNORM));
    state.texture_to_mapping.insert((1, 3), 9);

    let view_ref = 8u32;
    let len = TEXTURE_VIEW_MIN_SWIZZLE;
    let mut desc = vec![0u8; len];
    st32(
        &mut desc[TEXTURE_VIEW_DESC_OPCODE..],
        TEXTURE_VIEW_OPCODE_SWIZZLE,
    );
    st32(&mut desc[TEXTURE_VIEW_DESC_LEN..], len as u32);
    st32(&mut desc[TEXTURE_VIEW_DESC_TEXTURE_REF..], view_ref);
    st32(&mut desc[TEXTURE_VIEW_DESC_BASE_REF..], 3);
    st16(
        &mut desc[TEXTURE_VIEW_DESC_PIXEL_FORMAT..],
        MTL_FORMAT_BGRA8_UNORM,
    );
    st16(
        &mut desc[TEXTURE_VIEW_DESC_TEXTURE_TYPE..],
        TEXTURE_VIEW_MTL_TYPE_2D,
    );
    st64(&mut desc[TEXTURE_VIEW_DESC_LEVEL_BASE..], 0);
    st64(&mut desc[TEXTURE_VIEW_DESC_LEVEL_COUNT..], 1);
    st64(&mut desc[TEXTURE_VIEW_DESC_SLICE_BASE..], 0);
    st64(&mut desc[TEXTURE_VIEW_DESC_SLICE_COUNT..], 1);
    // Non-identity BGRA → RGBA channel remap.
    desc[TEXTURE_VIEW_DESC_SWIZZLE..TEXTURE_VIEW_DESC_SWIZZLE + 4].copy_from_slice(&[2u8, 1, 0, 3]);
    let desc_gva = 0x280u64;
    write_task_gva_arm64e(&mut host, &state.tasks[1], desc_gva, &desc);
    let off = list_object_entry_offset(view_ref, 32).unwrap();
    let mut list_entry = [0u8; OBJECT_LIST_ENTRY_LEN];
    let packed = (OBJECT_TYPE_TEXTURE_VIEW as u32) | ((len as u32) << 8);
    st32(&mut list_entry[0..], packed);
    list_entry[4..12].copy_from_slice(&desc_gva.to_le_bytes());
    write_task_gva_arm64e(&mut host, &state.tasks[1], off, &list_entry);

    let att = ColorAttachment {
        present: true,
        texture_ref: view_ref,
        resolve_texture_ref: 0,
        level: 0,
        load_action: PASS_LOAD_ACTION_CLEAR,
        store_action: PASS_STORE_ACTION_STORE,
        clear_color: [0.0, 0.0, 0.0, 1.0],
    };
    assert!(
        mrt_draw_request(
            &mut state,
            &mut host,
            1,
            12,
            &[(0u32, att)],
            &[],
            3,
            1,
            3,
            0
        )
        .is_none(),
        "swizzled type-8 must not resolve as color RT"
    );
}

/// qemu-shim: type-2 linear RGBA16Float is a valid color RT. Stale
/// `texture_to_mapping` from a prior type-11 at the same ref must not
/// fail-closed (live residual ref=199 type=2 fmt=0x73).
#[test]
fn mrt_draw_request_type2_rgba16f_as_color_rt_despite_stale_t11_latch() {
    use crate::contract::endian::{st16, st32, st64};
    use crate::contract::pixel_format::{MTL_FORMAT_BGRA8_UNORM, MTL_FORMAT_RGBA16_FLOAT};
    use crate::runtime::decode::render::ColorAttachment;
    use crate::runtime::decode::resource::{
        list_object_entry_offset, LINEAR_DESC_HANDLE, LINEAR_DESC_SIZE, OBJECT_LIST_ENTRY_LEN,
        OBJECT_TYPE_TEXTURE, RESOURCE_PAGE_SHIFT, TEXTURE_DESC_BASE_LEN, TEXTURE_DESC_HEIGHT,
        TEXTURE_DESC_PIXEL_FORMAT, TEXTURE_DESC_ROW_STRIDE, TEXTURE_DESC_WIDTH,
    };
    use crate::runtime::host::FakeHost;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    gva_mem::define_task_pages_arm64e(&mut host, &mut state, 4, 16);
    assert!(state.set_object_list(1, 0, 256));

    // Stale type-11 latch at ref 199 (guest recycled the ref to type-2).
    assert!(state.map_surface(99));
    assert!(state.set_mapping_geom(99, 64, 64, MTL_FORMAT_BGRA8_UNORM));
    state.texture_to_mapping.insert((1, 199), 99);

    // Live type-2 RGBA16Float 480×64 bpr=3840 (live residual shape).
    let tex_ref = 199u32;
    let w = 480u32;
    let h = 64u32;
    let bpr = 3840u32;
    let handle = 8u32; // GVA page under setup_task_pages data
    let alloc = (bpr as u64) * (h as u64);
    let mut desc = vec![0u8; TEXTURE_DESC_BASE_LEN];
    st64(&mut desc[LINEAR_DESC_SIZE..], alloc);
    st32(&mut desc[LINEAR_DESC_HANDLE..], handle);
    st32(&mut desc[TEXTURE_DESC_ROW_STRIDE..], bpr);
    st32(&mut desc[TEXTURE_DESC_WIDTH..], w);
    st32(&mut desc[TEXTURE_DESC_HEIGHT..], h);
    st16(
        &mut desc[TEXTURE_DESC_PIXEL_FORMAT..],
        MTL_FORMAT_RGBA16_FLOAT,
    );
    let desc_gva = 0x280u64;
    write_task_gva_arm64e(&mut host, &state.tasks[1], desc_gva, &desc);
    let off = list_object_entry_offset(tex_ref, 256).unwrap();
    let mut list_entry = [0u8; OBJECT_LIST_ENTRY_LEN];
    let packed = (OBJECT_TYPE_TEXTURE as u32) | ((TEXTURE_DESC_BASE_LEN as u32) << 8);
    st32(&mut list_entry[0..], packed);
    list_entry[4..12].copy_from_slice(&desc_gva.to_le_bytes());
    write_task_gva_arm64e(&mut host, &state.tasks[1], off, &list_entry);

    let att = ColorAttachment {
        present: true,
        texture_ref: tex_ref,
        resolve_texture_ref: 0,
        level: 0,
        load_action: PASS_LOAD_ACTION_CLEAR,
        store_action: PASS_STORE_ACTION_STORE,
        clear_color: [0.0, 0.0, 0.0, 1.0],
    };
    let req = mrt_draw_request(
        &mut state,
        &mut host,
        1,
        12,
        &[(0u32, att)],
        &[],
        3,
        1,
        3,
        0,
    )
    .expect("type-2 RGBA16F RT must resolve despite stale type-11 latch");
    assert_eq!(req.colors[0].mapping_id, 0);
    assert_eq!(req.colors[0].width, w);
    assert_eq!(req.colors[0].height, h);
    assert_eq!(req.colors[0].format, MTL_FORMAT_RGBA16_FLOAT);
    assert_eq!(
        req.colors[0].target_gva,
        (handle as u64) << RESOURCE_PAGE_SHIFT
    );
    // Stale latch must be dropped.
    assert!(!state.texture_to_mapping.contains_key(&(1, tex_ref)));
}

/// Live type-11 descriptor mapping_id wins over a stale texture_to_mapping
/// latch (dual-mid recycled-ref residual: full desktop Store must land on
/// the mid named by the live descriptor, not a prior latch).
#[test]
fn mrt_draw_request_type11_live_mapping_overrides_stale_latch() {
    use crate::contract::endian::{st16, st32};
    use crate::contract::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
    use crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    use crate::runtime::decode::render::ColorAttachment;
    use crate::runtime::host::FakeHost;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    // 1-level page table: GVA page 0 → data pfn 4.
    let dir_gpa = 2u64 << PAGE_SHIFT_ARM64E;
    let root_gpa = 3u64 << PAGE_SHIFT_ARM64E;
    let data_gpa = 4u64 << PAGE_SHIFT_ARM64E;
    host.map_range(dir_gpa, 0x20, 0);
    host.map_range(root_gpa, 0x4000, 0);
    host.map_range(data_gpa, 0x200, 0);
    let mut d = [0u8; 8];
    st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 3);
    st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
    let _ = host.write_gpa(dir_gpa, &d);
    st32(&mut d[..4], 4);
    let _ = host.write_gpa(root_gpa, &d[..4]);
    assert!(state.define_task(1, 0x1000, 2));
    assert!(state.set_object_list(1, 0, 8));
    // Live type-11 at ref=1 → mapping_id=4 (descriptor first u32).
    let mut entry = [0u8; 12];
    st32(&mut entry[0..], 11u32 | (0x20u32 << 8));
    entry[4..12].copy_from_slice(&0x40u64.to_le_bytes());
    let _ = host.write_gpa(data_gpa + 12, &entry);
    let mut desc = [0u8; 0x20];
    st32(&mut desc[0..], 4); // live mapping_id = 4
    st16(&mut desc[0x16..], MTL_FORMAT_BGRA8_UNORM);
    st32(&mut desc[0x18..], 64);
    st32(&mut desc[0x1c..], 32);
    let _ = host.write_gpa(data_gpa + 0x40, &desc);

    // Both mids exist; stale latch points ref 1 at mid 3.
    assert!(state.map_surface(3));
    assert!(state.set_mapping_geom(3, 64, 32, MTL_FORMAT_BGRA8_UNORM));
    assert!(state.map_surface(4));
    assert!(state.set_mapping_geom(4, 64, 32, MTL_FORMAT_BGRA8_UNORM));
    state.texture_to_mapping.insert((1, 1), 3);

    let att = ColorAttachment {
        present: true,
        texture_ref: 1,
        resolve_texture_ref: 0,
        level: 0,
        load_action: PASS_LOAD_ACTION_CLEAR,
        store_action: PASS_STORE_ACTION_STORE,
        clear_color: [0.0, 0.0, 0.0, 1.0],
    };
    let req = mrt_draw_request(
        &mut state,
        &mut host,
        1,
        12,
        &[(0u32, att)],
        &[],
        3,
        1,
        3,
        0,
    )
    .expect("live type-11 RT must resolve");
    assert_eq!(
        req.colors[0].mapping_id, 4,
        "live descriptor mapping_id=4 must beat stale latch mid=3"
    );
    assert_eq!(state.texture_to_mapping.get(&(1, 1)).copied(), Some(4));
}

/// Color RT materialization does not rematerialize non-zero view mips.
#[test]
fn mrt_draw_request_type8_nonzero_level_rejected_as_color_rt() {
    use crate::contract::endian::{st16, st32, st64};
    use crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    use crate::runtime::decode::render::ColorAttachment;
    use crate::runtime::decode::resource::{
        list_object_entry_offset, OBJECT_LIST_ENTRY_LEN, OBJECT_TYPE_TEXTURE_VIEW,
        TEXTURE_VIEW_DESC_BASE_REF, TEXTURE_VIEW_DESC_LEN, TEXTURE_VIEW_DESC_LEVEL_BASE,
        TEXTURE_VIEW_DESC_LEVEL_COUNT, TEXTURE_VIEW_DESC_OPCODE, TEXTURE_VIEW_DESC_PIXEL_FORMAT,
        TEXTURE_VIEW_DESC_SLICE_BASE, TEXTURE_VIEW_DESC_SLICE_COUNT, TEXTURE_VIEW_DESC_TEXTURE_REF,
        TEXTURE_VIEW_DESC_TEXTURE_TYPE, TEXTURE_VIEW_MIN_RANGED, TEXTURE_VIEW_MTL_TYPE_2D,
        TEXTURE_VIEW_OPCODE_RANGED,
    };
    use crate::runtime::host::FakeHost;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    gva_mem::define_task_pages_arm64e(&mut host, &mut state, 4, 8);
    assert!(state.set_object_list(1, 0, 32));
    assert!(state.map_surface(9));
    assert!(state.set_mapping_geom(9, 64, 64, MTL_FORMAT_BGRA8_UNORM));
    state.texture_to_mapping.insert((1, 3), 9);

    let view_ref = 8u32;
    let len = TEXTURE_VIEW_MIN_RANGED;
    let mut desc = vec![0u8; len];
    st32(
        &mut desc[TEXTURE_VIEW_DESC_OPCODE..],
        TEXTURE_VIEW_OPCODE_RANGED,
    );
    st32(&mut desc[TEXTURE_VIEW_DESC_LEN..], len as u32);
    st32(&mut desc[TEXTURE_VIEW_DESC_TEXTURE_REF..], view_ref);
    st32(&mut desc[TEXTURE_VIEW_DESC_BASE_REF..], 3);
    st16(
        &mut desc[TEXTURE_VIEW_DESC_PIXEL_FORMAT..],
        MTL_FORMAT_BGRA8_UNORM,
    );
    st16(
        &mut desc[TEXTURE_VIEW_DESC_TEXTURE_TYPE..],
        TEXTURE_VIEW_MTL_TYPE_2D,
    );
    st64(&mut desc[TEXTURE_VIEW_DESC_LEVEL_BASE..], 1); // mip 1
    st64(&mut desc[TEXTURE_VIEW_DESC_LEVEL_COUNT..], 1);
    st64(&mut desc[TEXTURE_VIEW_DESC_SLICE_BASE..], 0);
    st64(&mut desc[TEXTURE_VIEW_DESC_SLICE_COUNT..], 1);
    let desc_gva = 0x280u64;
    write_task_gva_arm64e(&mut host, &state.tasks[1], desc_gva, &desc);
    let off = list_object_entry_offset(view_ref, 32).unwrap();
    let mut list_entry = [0u8; OBJECT_LIST_ENTRY_LEN];
    let packed = (OBJECT_TYPE_TEXTURE_VIEW as u32) | ((len as u32) << 8);
    st32(&mut list_entry[0..], packed);
    list_entry[4..12].copy_from_slice(&desc_gva.to_le_bytes());
    write_task_gva_arm64e(&mut host, &state.tasks[1], off, &list_entry);

    let att = ColorAttachment {
        present: true,
        texture_ref: view_ref,
        resolve_texture_ref: 0,
        level: 0,
        load_action: PASS_LOAD_ACTION_CLEAR,
        store_action: PASS_STORE_ACTION_STORE,
        clear_color: [0.0, 0.0, 0.0, 1.0],
    };
    assert!(
        mrt_draw_request(
            &mut state,
            &mut host,
            1,
            12,
            &[(0u32, att)],
            &[],
            3,
            1,
            3,
            0
        )
        .is_none(),
        "type-8 level_base!=0 must not resolve as color RT"
    );
}

/// Archive collapses a type-8 view's mip level into linear geometry:
/// a level-1 view of a type-2 texture is a color RT at that level's
/// plane (offset/dims/stride from the descriptor's level record) —
/// compositor blur/backdrop pyramids render into successive mips.
#[test]
fn mrt_draw_request_type8_mip_level_view_of_linear_as_color_rt() {
    use crate::contract::endian::{st16, st32, st64};
    use crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    use crate::runtime::decode::render::ColorAttachment;
    use crate::runtime::decode::resource::{
        list_object_entry_offset, OBJECT_LIST_ENTRY_LEN, OBJECT_TYPE_TEXTURE,
        OBJECT_TYPE_TEXTURE_VIEW, TEXTURE_DESC_BASE_LEN, TEXTURE_DESC_LEVEL_RECORDS,
        TEXTURE_DESC_MIPMAP_LEVEL_COUNT, TEXTURE_DESC_MIP_LEVEL_RECORD_LEN,
        TEXTURE_DESC_PIXEL_FORMAT, TEXTURE_DESC_ROW_STRIDE, TEXTURE_DESC_USED_SIZE,
        TEXTURE_DESC_WIDTH, TEXTURE_LEVEL_HEIGHT, TEXTURE_LEVEL_OFFSET, TEXTURE_LEVEL_ROW_STRIDE,
        TEXTURE_LEVEL_SIZE, TEXTURE_LEVEL_WIDTH, TEXTURE_VIEW_DESC_BASE_REF, TEXTURE_VIEW_DESC_LEN,
        TEXTURE_VIEW_DESC_LEVEL_BASE, TEXTURE_VIEW_DESC_LEVEL_COUNT, TEXTURE_VIEW_DESC_OPCODE,
        TEXTURE_VIEW_DESC_PIXEL_FORMAT, TEXTURE_VIEW_DESC_SLICE_BASE,
        TEXTURE_VIEW_DESC_SLICE_COUNT, TEXTURE_VIEW_DESC_TEXTURE_REF,
        TEXTURE_VIEW_DESC_TEXTURE_TYPE, TEXTURE_VIEW_MIN_RANGED, TEXTURE_VIEW_MTL_TYPE_2D,
        TEXTURE_VIEW_OPCODE_RANGED,
    };
    use crate::runtime::host::FakeHost;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    gva_mem::define_task_pages_arm64e(&mut host, &mut state, 4, 8);
    assert!(state.set_object_list(1, 0, 32));

    // Type-2 base with 2 mips: L0 64x32 bpr 256; L1 at +0x2000, 32x16 bpr 128.
    let base_ref = 5u32;
    let body = TEXTURE_DESC_BASE_LEN + TEXTURE_DESC_MIP_LEVEL_RECORD_LEN;
    let mut b = vec![0u8; body];
    st64(&mut b[0..], 0x20000); // allocation_size
    st32(&mut b[8..], 0x20); // handle
    st16(&mut b[TEXTURE_DESC_MIPMAP_LEVEL_COUNT..], 2);
    st32(&mut b[TEXTURE_DESC_USED_SIZE..], 64 * 32 * 4);
    st32(&mut b[TEXTURE_DESC_ROW_STRIDE..], 256);
    st32(&mut b[TEXTURE_DESC_WIDTH..], 64);
    st32(&mut b[TEXTURE_DESC_WIDTH + 4..], 32); // height
    let rec = TEXTURE_DESC_LEVEL_RECORDS;
    st64(&mut b[rec + TEXTURE_LEVEL_OFFSET..], 0x2000);
    st64(&mut b[rec + TEXTURE_LEVEL_SIZE..], 32 * 16 * 4);
    st64(&mut b[rec + TEXTURE_LEVEL_ROW_STRIDE..], 128);
    st32(&mut b[rec + TEXTURE_LEVEL_WIDTH..], 32);
    st32(&mut b[rec + TEXTURE_LEVEL_HEIGHT..], 16);
    st32(&mut b[rec + TEXTURE_LEVEL_HEIGHT + 4..], 1); // depth
    st16(
        &mut b[TEXTURE_DESC_PIXEL_FORMAT + TEXTURE_DESC_MIP_LEVEL_RECORD_LEN..],
        MTL_FORMAT_BGRA8_UNORM,
    );
    let base_desc_gva = 0x200u64;
    write_task_gva_arm64e(&mut host, &state.tasks[1], base_desc_gva, &b);
    let off = list_object_entry_offset(base_ref, 32).unwrap();
    let mut le = [0u8; OBJECT_LIST_ENTRY_LEN];
    st32(
        &mut le[0..],
        (OBJECT_TYPE_TEXTURE as u32) | ((body as u32) << 8),
    );
    le[4..12].copy_from_slice(&base_desc_gva.to_le_bytes());
    write_task_gva_arm64e(&mut host, &state.tasks[1], off, &le);

    // Type-8 view: level_base=1 over the type-2 base.
    let view_ref = 8u32;
    let len = TEXTURE_VIEW_MIN_RANGED;
    let mut desc = vec![0u8; len];
    st32(
        &mut desc[TEXTURE_VIEW_DESC_OPCODE..],
        TEXTURE_VIEW_OPCODE_RANGED,
    );
    st32(&mut desc[TEXTURE_VIEW_DESC_LEN..], len as u32);
    st32(&mut desc[TEXTURE_VIEW_DESC_TEXTURE_REF..], view_ref);
    st32(&mut desc[TEXTURE_VIEW_DESC_BASE_REF..], base_ref);
    st16(
        &mut desc[TEXTURE_VIEW_DESC_PIXEL_FORMAT..],
        MTL_FORMAT_BGRA8_UNORM,
    );
    st16(
        &mut desc[TEXTURE_VIEW_DESC_TEXTURE_TYPE..],
        TEXTURE_VIEW_MTL_TYPE_2D,
    );
    st64(&mut desc[TEXTURE_VIEW_DESC_LEVEL_BASE..], 1);
    st64(&mut desc[TEXTURE_VIEW_DESC_LEVEL_COUNT..], 1);
    st64(&mut desc[TEXTURE_VIEW_DESC_SLICE_BASE..], 0);
    st64(&mut desc[TEXTURE_VIEW_DESC_SLICE_COUNT..], 1);
    let desc_gva = 0x400u64;
    write_task_gva_arm64e(&mut host, &state.tasks[1], desc_gva, &desc);
    let off = list_object_entry_offset(view_ref, 32).unwrap();
    let mut le = [0u8; OBJECT_LIST_ENTRY_LEN];
    st32(
        &mut le[0..],
        (OBJECT_TYPE_TEXTURE_VIEW as u32) | ((len as u32) << 8),
    );
    le[4..12].copy_from_slice(&desc_gva.to_le_bytes());
    write_task_gva_arm64e(&mut host, &state.tasks[1], off, &le);

    let att = ColorAttachment {
        present: true,
        texture_ref: view_ref,
        resolve_texture_ref: 0,
        level: 0,
        load_action: PASS_LOAD_ACTION_CLEAR,
        store_action: PASS_STORE_ACTION_STORE,
        clear_color: [0.0, 0.0, 0.0, 1.0],
    };
    let req = mrt_draw_request(
        &mut state,
        &mut host,
        1,
        12,
        &[(0u32, att)],
        &[],
        3,
        1,
        3,
        0,
    )
    .expect("mip-1 view of linear texture must resolve as color RT");
    let c0 = &req.colors[0];
    assert_eq!(c0.mapping_id, 0);
    assert_eq!(
        c0.target_gva,
        ((0x20u64) << PAGE_SHIFT_ARM64E) + 0x2000,
        "RT gva = allocation base + level-1 offset"
    );
    assert_eq!((c0.width, c0.height), (32, 16), "level-1 dims");
    assert_eq!(c0.row_stride, 128, "level-1 row stride");
}

#[test]
fn view_swizzle_remaps_rgba8_pixels() {
    // Every CPU remap must report itself: this is the path the Vulkan
    // pathway replaced with a component mapping, and an unreported
    // invocation is a texture that silently lost its zero-copy crossing.
    crate::runtime::census::view_swizzle_census::reset_for_tests();
    // Reims VGPU selectors: 0=zero 1=one 2=R 3=G 4=B 5=A → BGRA order + forced alpha one.
    let plan = pixel_format::swizzle_plan(&[4, 3, 2, 1]).unwrap();
    let mut rgba = vec![10u8, 20, 30, 40, 50, 60, 70, 80];
    apply_view_swizzle_rgba8(&mut rgba, Some(&plan), 1).unwrap();
    assert_eq!(&rgba[0..4], &[30, 20, 10, 255]);
    assert_eq!(&rgba[4..8], &[70, 60, 50, 255]);
    // Identity is a no-op.
    let id = pixel_format::swizzle_identity();
    let before = rgba.clone();
    apply_view_swizzle_rgba8(&mut rgba, Some(&id), 1).unwrap();
    assert_eq!(rgba, before);
    // No plan leaves buffer untouched.
    apply_view_swizzle_rgba8(&mut rgba, None, 1).unwrap();
    assert_eq!(rgba, before);
    // Odd length fails visibly.
    let mut bad = vec![1u8, 2, 3];
    assert!(apply_view_swizzle_rgba8(&mut bad, Some(&plan), 1).is_none());
    // One non-identity remap ran and said so; the identity and None calls did
    // not, and neither did the length-rejected one. Read off the always-on sink
    // rather than a counter: the line is what a boot actually has to show.
    let log = std::fs::read_to_string(crate::observe::fail_log_path()).expect("fail log");
    assert_eq!(
        log.match_indices("view_swizzle_cpu_remap").count(),
        1,
        "exactly one CPU remap must be reported"
    );
    crate::runtime::census::view_swizzle_census::reset_for_tests();
}

#[test]
fn view_format_reinterprets_bgra_storage_as_rgba() {
    // Physical BGRA bytes B,G,R,A = 10,20,30,40.
    // As BGRA8 → RGBA sample: (30,20,10,40).
    // As RGBA8 view override → sample: (10,20,30,40) (byte reinterpret).
    let raw = [10u8, 20, 30, 40];
    let mut as_bgra = [0u8; 4];
    assert!(pixel_format::convert_row_to_rgba8(
        MTL_FORMAT_BGRA8_UNORM,
        &raw,
        1,
        &mut as_bgra
    ));
    assert_eq!(as_bgra, [30, 20, 10, 40]);
    let mut as_rgba = [0u8; 4];
    assert!(pixel_format::convert_row_to_rgba8(
        pixel_format::MTL_FORMAT_RGBA8_UNORM,
        &raw,
        1,
        &mut as_rgba
    ));
    assert_eq!(as_rgba, [10, 20, 30, 40]);
    // Combined path uses effective format.
    let fmt = effective_view_sample_format(
        MTL_FORMAT_BGRA8_UNORM,
        Some(pixel_format::MTL_FORMAT_RGBA8_UNORM),
    )
    .unwrap();
    let mut out = [0u8; 4];
    assert!(pixel_format::convert_row_to_rgba8(fmt, &raw, 1, &mut out));
    assert_eq!(out, [10, 20, 30, 40]);
}

/// Regression: type-2/3 GVA Stores must walk with device page_shift (x86=12).
/// Using the arm64e-default fallback made every `linux_m2v_store gva=… ok=0`
/// on Ventura/Tahoe x86 product boots.
#[test]
fn write_gva_rgba8_uses_device_page_shift_x86() {
    use crate::contract::endian::st32;
    use crate::contract::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
    use crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    use crate::runtime::host::FakeHost;

    let page_shift = PAGE_SHIFT_X86;
    let mut host = FakeHost::new();
    let dir_gpa = 2u64 << page_shift;
    let root_gpa = 3u64 << page_shift;
    // data for GVA page 1 (write_gva_rgba8 rejects gva==0 as "no target")
    let data_gpa = 5u64 << page_shift;
    host.map_range(dir_gpa, 0x20, 0);
    host.map_range(root_gpa, 0x1000, 0);
    host.map_range(data_gpa, 0x1000, 0);
    let mut d = [0u8; 8];
    st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 3);
    st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
    let _ = host.write_gpa(dir_gpa, &d);
    // PTE index 1 → pfn 5 (GVA 0x1000)
    st32(&mut d[..4], 5);
    let _ = host.write_gpa(root_gpa + 4, &d[..4]);

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    state.page_shift = page_shift;
    assert!(state.define_task(1, 0x1000, 2));

    let gva = 1u64 << page_shift; // 0x1000
                                  // Baseline walker under x86 tables.
    let probe = [0x11u8, 0x22, 0x33, 0x44];
    assert!(
        gva_mem::write_task_gva(&mut host, &state.tasks[1], gva, &probe, page_shift).is_ok(),
        "direct x86 GVA write must work"
    );

    // Tight RGBA8 2×2 → BGRA rows at GVA 0x1000.
    let rgba = [
        10u8, 20, 30, 255, // R G B A
        40, 50, 60, 255, //
        70, 80, 90, 255, //
        100, 110, 120, 255,
    ];
    assert!(
        write_gva_rgba8(
            &mut state,
            &mut host,
            1,
            gva,
            2,
            2,
            8, // bpr = 2*4
            MTL_FORMAT_BGRA8_UNORM,
            &rgba,
        )
        .is_ok(),
        "x86 page_shift=12 GVA store must succeed"
    );
    let mut back = [0u8; 8];
    assert!(gva_mem::read_task_gva(&host, &state.tasks[1], gva, &mut back, page_shift).is_ok());
    // BGRA row0: B,G,R,A = 30,20,10,255
    assert_eq!(&back[..4], &[30, 20, 10, 255]);
}

/// A render-target Store outside the writing task's MapMemory2 spans still
/// reaches guest RAM, and that is deliberate.
///
/// This rail was the first to be exempted, by measurement, and it was right: on
/// a driven x86/Vulkan boot the gate read `exact=1155 no_spans=0 outside=893`
/// over 2048 Stores, so refusing here drops 44% of them and blanks the screen.
/// MapMemory2 does not describe render targets — see the module note on
/// `write_gva_rgba8` for the span enumeration and for why the `owners=` field
/// cannot be used as a weaker gate either.
///
/// Every other rail has since been exempted too, for the same reason arrived at
/// from the other end: a notification the guest sends *after* installing the
/// PTEs and using the memory cannot authorise anything. `WriteGate::Undeclared`
/// is now a reported reading everywhere rather than a refusal anywhere, and
/// `44%` is the number that says how normal it is.
///
/// This test exists so that adding the gate back fails loudly rather than
/// silently costing half the frame. The fixture declares a span for the writing
/// task that deliberately does *not* cover the target, which is exactly the arm
/// that would refuse.
#[test]
fn an_rgba8_store_outside_the_tasks_declared_span_still_reaches_guest_ram() {
    use crate::contract::endian::st32;
    use crate::contract::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
    use crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    use crate::runtime::host::FakeHost;

    let page_shift = PAGE_SHIFT_X86;
    let mut host = FakeHost::new();
    let dir_gpa = 2u64 << page_shift;
    let root_gpa = 3u64 << page_shift;
    let data_gpa = 5u64 << page_shift;
    host.map_range(dir_gpa, 0x20, 0);
    host.map_range(root_gpa, 0x1000, 0);
    host.map_range(data_gpa, 0x1000, 0);
    let mut d = [0u8; 8];
    st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 3);
    st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
    let _ = host.write_gpa(dir_gpa, &d);
    st32(&mut d[..4], 5);
    let _ = host.write_gpa(root_gpa + 4, &d[..4]);

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    state.page_shift = page_shift;
    assert!(state.define_task(1, 0x1000, 2));

    let gva = 1u64 << page_shift; // 0x1000

    // Task 1 declares a span far away from the target. The registry is now
    // non-empty for this task, so the gate has a real bounds check to fail.
    state.note_task_map(1, 0x9_0000, 0x1000);
    assert_eq!(
        state.gva_write_gate(1, gva, 2 * 8),
        crate::model::WriteGate::Undeclared,
        "fixture is not real: the gate must land on the arm under test"
    );

    let rgba = [
        10u8, 20, 30, 255, //
        40, 50, 60, 255, //
        70, 80, 90, 255, //
        100, 110, 120, 255,
    ];
    assert!(
        write_gva_rgba8(
            &mut state,
            &mut host,
            1,
            gva,
            2,
            2,
            8,
            MTL_FORMAT_BGRA8_UNORM,
            &rgba,
        )
        .is_ok(),
        "this writer is deliberately ungated: an Outside store must still reach \
         guest RAM, or 44% of render Stores are lost"
    );

    // …and the bytes really landed, so this is not passing on a write that
    // failed for some unrelated reason.
    let mut back = [0u8; 8];
    assert!(gva_mem::read_task_gva(&host, &state.tasks[1], gva, &mut back, page_shift).is_ok());
    assert_eq!(&back[..4], &[30, 20, 10, 255]);
}

/// Type-2/3 GVA wallpaper layers must be sampleable from texture_ref host
/// cache (not surface_id mid map) after encode Store.
#[test]
fn gva_layer_host_cache_roundtrip_for_sample() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let tex_ref = 54u32;
    let gva = 0x2c48000u64;
    let w = 4u32;
    let h = 3u32;
    // Sky-blue solid (pipe-59 class): R G B A
    let mut rgba = vec![0u8; (w * h * 4) as usize];
    for px in rgba.chunks_exact_mut(4) {
        px[0] = 81;
        px[1] = 126;
        px[2] = 185;
        px[3] = 255;
    }
    host_cache_store_gva_layer(&mut state, tex_ref, OBJECT_TYPE_TEXTURE, gva, w, h, &rgba);
    let cached = crate::runtime::surface_cache::get_texture(&state, tex_ref, w, h)
        .expect("texture_ref encode cache");
    // BGRA storage
    assert_eq!(&cached[0..4], &[185, 126, 81, 255]);
    assert!(crate::runtime::surface_cache::get(&state, tex_ref, w, h).is_none());
    assert_eq!(
        crate::runtime::surface_cache::get_gva(&state, gva, w, h).unwrap()[0],
        185
    );
    // Sample path must hit texture cache without guest GVA walk / object list.
    let mut host = crate::runtime::host::FakeHost::new();
    let (sw, sh, sampled_mid, sampled) =
        resolve_sampled_source(&mut state, &mut host, 0, tex_ref, None).expect("sample from cache");
    assert_eq!((sw, sh), (w, h));
    assert_eq!(sampled_mid, 0, "linear cache sample is not a type-11 edge");
    let SampledSourceRequest::Bytes(sampled, _, _) = sampled else {
        panic!("cache-only fixture unexpectedly resolved resident target");
    };
    assert_eq!(&sampled[0..4], &[81, 126, 185, 255]);
}

#[test]
fn type3_linear_sample_uses_type2_gva_storage_cache() {
    use crate::contract::endian::{st16, st32, st64};
    use crate::contract::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
    use crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    use crate::runtime::decode::resource::{
        list_object_entry_offset, OBJECT_LIST_ENTRY_LEN, TEXTURE_DESC_BASE_LEN,
        TEXTURE_DESC_MIPMAP_LEVEL_COUNT, TEXTURE_DESC_PIXEL_FORMAT, TEXTURE_DESC_ROW_STRIDE,
        TEXTURE_DESC_USED_SIZE, TEXTURE_DESC_WIDTH,
    };

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let dir_pfn = 2u32;
    let root_pfn = 3u32;
    let dir_gpa = (dir_pfn as u64) << PAGE_SHIFT_ARM64E;
    let root_gpa = (root_pfn as u64) << PAGE_SHIFT_ARM64E;
    host.map_range(dir_gpa, 0x20, 0);
    host.map_range(root_gpa, 0x4000, 0);
    let mut d = [0u8; 8];
    st32(&mut d[DIRECTORY_ROOT_PFN as usize..], root_pfn);
    st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
    assert!(host.write_gpa(dir_gpa, &d).is_ok());
    for i in 0..2u32 {
        let pfn = 4 + i;
        host.map_range((pfn as u64) << PAGE_SHIFT_ARM64E, 0x4000, 0);
        let mut pte = [0u8; 4];
        st32(&mut pte, pfn);
        assert!(host.write_gpa(root_gpa + (i as u64) * 4, &pte).is_ok());
    }
    assert!(state.define_task(1, 0x1000, dir_pfn));
    assert!(state.set_object_list(1, 0, 32));

    let tex_ref = 19u32;
    let w = 4u32;
    let h = 2u32;
    let bpr = 16u32;
    let body = TEXTURE_DESC_BASE_LEN;
    let mut desc = vec![0u8; body];
    st64(&mut desc[0..], (bpr as u64) * (h as u64));
    st32(&mut desc[8..], 1); // handle -> base gva 1 << page_shift
    st16(&mut desc[TEXTURE_DESC_MIPMAP_LEVEL_COUNT..], 1);
    st32(&mut desc[TEXTURE_DESC_USED_SIZE..], bpr * h);
    st32(&mut desc[TEXTURE_DESC_ROW_STRIDE..], bpr);
    st32(&mut desc[TEXTURE_DESC_WIDTH..], w);
    st32(&mut desc[TEXTURE_DESC_WIDTH + 4..], h);
    st16(
        &mut desc[TEXTURE_DESC_PIXEL_FORMAT..],
        MTL_FORMAT_BGRA8_UNORM,
    );
    let desc_gva = 0x200u64;
    write_task_gva_arm64e(&mut host, &state.tasks[1], desc_gva, &desc);
    let off = list_object_entry_offset(tex_ref, 32).unwrap();
    let mut list_entry = [0u8; OBJECT_LIST_ENTRY_LEN];
    st32(
        &mut list_entry[0..],
        (OBJECT_TYPE_TEXTURE_VARIANT as u32) | ((body as u32) << 8),
    );
    list_entry[4..12].copy_from_slice(&desc_gva.to_le_bytes());
    write_task_gva_arm64e(&mut host, &state.tasks[1], off, &list_entry);

    let texel_gva = 1u64 << PAGE_SHIFT_ARM64E;
    let bgra = [40u8, 20, 10, 255].repeat((w * h) as usize);
    crate::runtime::surface_cache::store_gva_owned(
        &mut state,
        texel_gva,
        w,
        h,
        bgra,
        OBJECT_TYPE_TEXTURE,
    );

    let le_entry = objects::lookup_list_entry(&state, &host, 1, tex_ref)
        .expect("object-list entry must resolve");
    let td = decode_texture_descriptor(
        &objects::read_descriptor(&state, &host, 1, &le_entry).expect("descriptor must read"),
    )
    .expect("descriptor must decode");
    let (sw, sh, rgba, identity, fmt) =
        load_linear_from_host_caches(&mut state, &mut host, 1, tex_ref, &le_entry, &td)
            .expect("type-3 sample must use type-2-produced GVA cache");

    assert_eq!((sw, sh), (w, h));
    assert_eq!(fmt, TexelLayout::Rgba8);
    assert_eq!(&rgba[..4], &[10, 20, 40, 255]);
    let identity = identity.expect("GVA cache identity");
    assert_eq!(identity.key, texel_gva);
    assert_eq!(identity.generation, 1);
}

/// Guest-CPU-produced tight linear textures: unchanged native bytes must
/// reuse the memoized RGBA Arc under a stable >u32::MAX generation
/// identity; a guest write must be observed and produce a new generation.
#[test]
fn guest_linear_memo_reuses_arc_and_observes_guest_writes() {
    use crate::contract::endian::{st16, st32, st64};
    use crate::contract::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
    use crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    use crate::runtime::decode::resource::{
        list_object_entry_offset, OBJECT_LIST_ENTRY_LEN, TEXTURE_DESC_BASE_LEN,
        TEXTURE_DESC_MIPMAP_LEVEL_COUNT, TEXTURE_DESC_PIXEL_FORMAT, TEXTURE_DESC_ROW_STRIDE,
        TEXTURE_DESC_USED_SIZE, TEXTURE_DESC_WIDTH,
    };

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let dir_pfn = 2u32;
    let root_pfn = 3u32;
    let dir_gpa = (dir_pfn as u64) << PAGE_SHIFT_ARM64E;
    let root_gpa = (root_pfn as u64) << PAGE_SHIFT_ARM64E;
    host.map_range(dir_gpa, 0x20, 0);
    host.map_range(root_gpa, 0x4000, 0);
    let mut d = [0u8; 8];
    st32(&mut d[DIRECTORY_ROOT_PFN as usize..], root_pfn);
    st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
    assert!(host.write_gpa(dir_gpa, &d).is_ok());
    for i in 0..4u32 {
        let pfn = 4 + i;
        host.map_range((pfn as u64) << PAGE_SHIFT_ARM64E, 0x4000, 0);
        let mut pte = [0u8; 4];
        st32(&mut pte, pfn);
        assert!(host.write_gpa(root_gpa + (i as u64) * 4, &pte).is_ok());
    }
    assert!(state.define_task(1, 0x1000, dir_pfn));
    assert!(state.set_object_list(1, 0, 32));

    // Tight 4x2 BGRA8: bpr 16, texels at handle-page 1 (gva 0x4000).
    let tex_ref = 6u32;
    let body = TEXTURE_DESC_BASE_LEN;
    let mut b = vec![0u8; body];
    st64(&mut b[0..], 0x1000); // allocation_size
    st32(&mut b[8..], 1); // handle -> base gva 1 << page_shift
    st16(&mut b[TEXTURE_DESC_MIPMAP_LEVEL_COUNT..], 1);
    st32(&mut b[TEXTURE_DESC_USED_SIZE..], 16 * 2);
    st32(&mut b[TEXTURE_DESC_ROW_STRIDE..], 16);
    st32(&mut b[TEXTURE_DESC_WIDTH..], 4);
    st32(&mut b[TEXTURE_DESC_WIDTH + 4..], 2); // height
    st16(&mut b[TEXTURE_DESC_PIXEL_FORMAT..], MTL_FORMAT_BGRA8_UNORM);
    let desc_gva = 0x200u64;
    write_task_gva_arm64e(&mut host, &state.tasks[1], desc_gva, &b);
    let off = list_object_entry_offset(tex_ref, 32).unwrap();
    let mut le = [0u8; OBJECT_LIST_ENTRY_LEN];
    st32(
        &mut le[0..],
        (OBJECT_TYPE_TEXTURE as u32) | ((body as u32) << 8),
    );
    le[4..12].copy_from_slice(&desc_gva.to_le_bytes());
    write_task_gva_arm64e(&mut host, &state.tasks[1], off, &le);
    let texel_gva = 1u64 << PAGE_SHIFT_ARM64E;
    let bgra = [7u8, 5, 3, 255].repeat(8);
    write_task_gva_arm64e(&mut host, &state.tasks[1], texel_gva, &bgra);

    // The caller resolves the object-list entry + decodes the descriptor
    // once and threads them in; the list is immutable for the draw.
    let le_entry = objects::lookup_list_entry(&state, &host, 1, tex_ref)
        .expect("object-list entry must resolve");
    let td = decode_texture_descriptor(
        &objects::read_descriptor(&state, &host, 1, &le_entry).expect("descriptor must read"),
    )
    .expect("descriptor must decode");

    let (w, h, rgba1, id1, fmt1) =
        load_linear_from_host_caches(&mut state, &mut host, 1, tex_ref, &le_entry, &td)
            .expect("guest tight linear must load");
    assert_eq!((w, h), (4, 2));
    assert_eq!(
        fmt1,
        TexelLayout::Bgra8,
        "the tight guest-memo path uploads native BGRA8 (no CPU swizzle)"
    );
    assert_eq!(&rgba1[..4], &[7, 5, 3, 255], "native BGRA8, unswizzled");
    let id1 = id1.expect("guest memo path must carry an identity");
    assert_eq!(id1.key, texel_gva);
    assert!(
        id1.generation > u32::MAX as u64,
        "guest generations must not alias host_gen: {}",
        id1.generation
    );

    let (_, _, rgba2, id2, _) =
        load_linear_from_host_caches(&mut state, &mut host, 1, tex_ref, &le_entry, &td)
            .expect("repeat load must succeed");
    assert!(
        std::sync::Arc::ptr_eq(&rgba1, &rgba2),
        "unchanged native bytes must reuse the memoized Arc"
    );
    assert_eq!(id2.expect("identity").generation, id1.generation);

    // A direct guest write must be observed on the very next load.
    let bgra_new = [90u8, 60, 30, 255].repeat(8);
    write_task_gva_arm64e(&mut host, &state.tasks[1], texel_gva, &bgra_new);
    let (_, _, rgba3, id3, _) =
        load_linear_from_host_caches(&mut state, &mut host, 1, tex_ref, &le_entry, &td)
            .expect("post-write load must succeed");
    assert!(!std::sync::Arc::ptr_eq(&rgba1, &rgba3));
    assert_eq!(&rgba3[..4], &[90, 60, 30, 255], "native BGRA8, unswizzled");
    assert_ne!(id3.expect("identity").generation, id1.generation);
}

/// Padded-stride BGRA8 (the Safari-scroll former `lin_guest_fb` hot path)
/// now rides the guest-linear memo (gva recurrence measured ~99% under
/// scroll). Assert it uploads the guest's NATIVE BGRA8 bytes (`byte_format
/// == Bgra8`, no CPU channel swap), carries a memo identity so the engine
/// skips its content hash + upload, and that the row gather takes exactly
/// the tight texels — skipping the padding — into the tight output.
#[test]
fn padded_bgra8_memoized_uploads_native_without_swizzle() {
    use crate::contract::endian::{st16, st32, st64};
    use crate::contract::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
    use crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    use crate::runtime::decode::resource::{
        list_object_entry_offset, OBJECT_LIST_ENTRY_LEN, TEXTURE_DESC_BASE_LEN,
        TEXTURE_DESC_MIPMAP_LEVEL_COUNT, TEXTURE_DESC_PIXEL_FORMAT, TEXTURE_DESC_ROW_STRIDE,
        TEXTURE_DESC_USED_SIZE, TEXTURE_DESC_WIDTH,
    };

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let dir_pfn = 2u32;
    let root_pfn = 3u32;
    let dir_gpa = (dir_pfn as u64) << PAGE_SHIFT_ARM64E;
    let root_gpa = (root_pfn as u64) << PAGE_SHIFT_ARM64E;
    host.map_range(dir_gpa, 0x20, 0);
    host.map_range(root_gpa, 0x4000, 0);
    let mut d = [0u8; 8];
    st32(&mut d[DIRECTORY_ROOT_PFN as usize..], root_pfn);
    st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
    assert!(host.write_gpa(dir_gpa, &d).is_ok());
    for i in 0..4u32 {
        let pfn = 4 + i;
        host.map_range((pfn as u64) << PAGE_SHIFT_ARM64E, 0x4000, 0);
        let mut pte = [0u8; 4];
        st32(&mut pte, pfn);
        assert!(host.write_gpa(root_gpa + (i as u64) * 4, &pte).is_ok());
    }
    assert!(state.define_task(1, 0x1000, dir_pfn));
    assert!(state.set_object_list(1, 0, 32));

    // 4x2 BGRA8 with a PADDED row stride: tight = 16, bpr = 24 (8 pad bytes
    // per row). Padding declines the tight-stride memo loader.
    let tex_ref = 6u32;
    let (w, h) = (4u32, 2u32);
    let tight = 16u32;
    let bpr = 24u32;
    let body = TEXTURE_DESC_BASE_LEN;
    let mut b = vec![0u8; body];
    st64(&mut b[0..], 0x1000); // allocation_size
    st32(&mut b[8..], 1); // handle -> base gva 1 << page_shift
    st16(&mut b[TEXTURE_DESC_MIPMAP_LEVEL_COUNT..], 1);
    st32(&mut b[TEXTURE_DESC_USED_SIZE..], bpr * h);
    st32(&mut b[TEXTURE_DESC_ROW_STRIDE..], bpr);
    st32(&mut b[TEXTURE_DESC_WIDTH..], w);
    st32(&mut b[TEXTURE_DESC_WIDTH + 4..], h);
    st16(&mut b[TEXTURE_DESC_PIXEL_FORMAT..], MTL_FORMAT_BGRA8_UNORM);
    let desc_gva = 0x200u64;
    write_task_gva_arm64e(&mut host, &state.tasks[1], desc_gva, &b);
    let off = list_object_entry_offset(tex_ref, 32).unwrap();
    let mut le = [0u8; OBJECT_LIST_ENTRY_LEN];
    st32(
        &mut le[0..],
        (OBJECT_TYPE_TEXTURE as u32) | ((body as u32) << 8),
    );
    le[4..12].copy_from_slice(&desc_gva.to_le_bytes());
    write_task_gva_arm64e(&mut host, &state.tasks[1], off, &le);

    // Write two padded rows: each 16 tight BGRA bytes then 8 pad bytes.
    let texel_gva = 1u64 << PAGE_SHIFT_ARM64E;
    let row0: Vec<u8> = [1u8, 2, 3, 255].repeat(4); // 16 bytes
    let row1: Vec<u8> = [10u8, 20, 30, 255].repeat(4);
    let pad = [0xEEu8; 8];
    let mut backing = Vec::new();
    backing.extend_from_slice(&row0);
    backing.extend_from_slice(&pad);
    backing.extend_from_slice(&row1);
    backing.extend_from_slice(&pad);
    assert_eq!(backing.len(), (bpr * h) as usize);
    write_task_gva_arm64e(&mut host, &state.tasks[1], texel_gva, &backing);

    let le_entry = objects::lookup_list_entry(&state, &host, 1, tex_ref)
        .expect("object-list entry must resolve");
    let td = decode_texture_descriptor(
        &objects::read_descriptor(&state, &host, 1, &le_entry).expect("descriptor must read"),
    )
    .expect("descriptor must decode");

    let (gw, gh, rgba, identity, fmt) =
        load_linear_from_host_caches(&mut state, &mut host, 1, tex_ref, &le_entry, &td)
            .expect("padded BGRA8 must load via the memo");
    assert_eq!((gw, gh), (w, h));
    assert_eq!(
        fmt,
        TexelLayout::Bgra8,
        "padded BGRA8 must upload native (no CPU swizzle)"
    );
    let id = identity.expect("the padded memo path carries a producer identity");
    assert!(
        id.generation > u32::MAX as u64,
        "guest generations must not alias host_gen: {}",
        id.generation
    );
    // Tight output = the two source rows concatenated, native BGRA order,
    // padding stripped. Length is w*h*4 regardless of format.
    let mut want = Vec::new();
    want.extend_from_slice(&row0);
    want.extend_from_slice(&row1);
    assert_eq!(
        &rgba[..],
        &want[..],
        "native bytes gathered, padding skipped"
    );
    assert_eq!(rgba.len(), (tight * h) as usize);

    // A repeat bind of unchanged content reuses the memoized Arc (the whole
    // point — the engine then skips its content hash + upload).
    let (_, _, rgba2, id2, fmt2) =
        load_linear_from_host_caches(&mut state, &mut host, 1, tex_ref, &le_entry, &td)
            .expect("repeat padded load must succeed");
    assert!(
        std::sync::Arc::ptr_eq(&rgba, &rgba2),
        "unchanged padded bytes must reuse the memoized Arc"
    );
    assert_eq!(fmt2, TexelLayout::Bgra8);
    assert_eq!(id2.expect("identity").generation, id.generation);
}

/// Black-load-seed-discard regression: GVA identity wins over colliding
/// texture/surface namespaces, and a zero-RGB result remains valid.
#[test]
fn color_load_seed_uses_provenance_and_preserves_black() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let task_id = std::process::id();
    let texture_ref = 0xe000_0000u32.wrapping_add(task_id);
    let target_gva = 0x5000_0000u64 + ((task_id as u64) << 12);
    let (w, h) = (2, 1);

    // Same numeric ref in the surface_id namespace must be irrelevant.
    crate::runtime::surface_cache::store(
        &mut state,
        texture_ref,
        w,
        h,
        vec![0, 0, 200, 255, 0, 0, 200, 255],
    );
    crate::runtime::surface_cache::store_texture(
        &mut state,
        texture_ref,
        w,
        h,
        vec![0, 180, 0, 255, 0, 180, 0, 255],
    );
    crate::runtime::surface_cache::store_gva(
        &mut state,
        target_gva,
        w,
        h,
        vec![0, 0, 0, 255, 0, 0, 0, 255],
    );

    let seed = seed_color_load(
        &mut state,
        &mut host,
        task_id,
        texture_ref,
        target_gva,
        w,
        h,
    )
    .expect("exact GVA cache seed");
    assert_eq!(seed, vec![0, 0, 0, 255, 0, 0, 0, 255]);

    // Without a GVA match, use the texture namespace (green), never the
    // colliding surface namespace (red).
    let texture_seed = seed_color_load(
        &mut state,
        &mut host,
        task_id,
        texture_ref,
        target_gva + 0x1000,
        w,
        h,
    )
    .expect("texture-ref cache seed");
    assert_eq!(texture_seed, vec![0, 180, 0, 255, 0, 180, 0, 255]);
}

/// A type-5 ref is not itself a surface id. The descriptor's surface_id
/// remains authoritative even when the numeric ref collides with another
/// live display mapping (live app-launch ref=2 -> sid=71 class).
#[test]
fn type5_sample_uses_descriptor_surface_id_not_ref_collision() {
    use crate::contract::endian::st32;
    use crate::contract::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
    use crate::runtime::decode::resource::{list_object_entry_offset, OBJECT_LIST_ENTRY_LEN};
    use crate::runtime::gva_mem;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();

    // One-level x86 GVA table: pages 0..2 -> data PFNs 4..6.
    let dir_pfn = 2u32;
    let root_pfn = 3u32;
    let dir_gpa = (dir_pfn as u64) << PAGE_SHIFT_X86;
    let root_gpa = (root_pfn as u64) << PAGE_SHIFT_X86;
    host.map_range(dir_gpa, 0x20, 0);
    host.map_range(root_gpa, 0x1000, 0);
    let mut dir = [0u8; 8];
    st32(&mut dir[DIRECTORY_ROOT_PFN as usize..], root_pfn);
    st32(&mut dir[DIRECTORY_DEPTH as usize..], 1);
    assert!(host.write_gpa(dir_gpa, &dir).is_ok());
    for i in 0..3u32 {
        let pfn = 4 + i;
        host.map_range((pfn as u64) << PAGE_SHIFT_X86, 0x1000, 0);
        let mut pte = [0u8; 4];
        st32(&mut pte, pfn);
        assert!(host.write_gpa(root_gpa + (i as u64) * 4, &pte).is_ok());
    }
    assert!(state.define_task(1, 0x1000, dir_pfn));
    assert!(state.set_object_list(1, 0, 32));

    let texture_ref = 2u32;
    let surface_id = 71u32;
    let desc_gva = 0x1000u64;
    let mut desc = vec![0u8; objects::TYPE5_MIN_LEN];
    st32(&mut desc[objects::TYPE5_SURFACE_ID..], surface_id);
    assert!(
        gva_mem::write_task_gva(&mut host, &state.tasks[1], desc_gva, &desc, PAGE_SHIFT_X86,)
            .is_ok()
    );
    let list_off = list_object_entry_offset(texture_ref, 32).unwrap();
    let mut list_entry = [0u8; OBJECT_LIST_ENTRY_LEN];
    let packed = (objects::OBJECT_TYPE_REF_TEXTURE as u32) | ((objects::TYPE5_MIN_LEN as u32) << 8);
    st32(&mut list_entry, packed);
    list_entry[4..12].copy_from_slice(&desc_gva.to_le_bytes());
    assert!(gva_mem::write_task_gva(
        &mut host,
        &state.tasks[1],
        list_off,
        &list_entry,
        PAGE_SHIFT_X86,
    )
    .is_ok());

    // The lower numeric ref intentionally collides with an unrelated map.
    assert!(state.map_surface(texture_ref));
    assert!(state.set_mapping_geom(texture_ref, 8, 8, MTL_FORMAT_BGRA8_UNORM));
    state.mappings.get_mut(&texture_ref).unwrap().page_entries = vec![1];
    crate::runtime::surface_cache::store(
        &mut state,
        texture_ref,
        8,
        8,
        [0u8, 0, 255, 255].repeat(8 * 8),
    );

    assert!(state.map_surface(surface_id));
    assert!(state.set_mapping_geom(surface_id, 4, 3, MTL_FORMAT_BGRA8_UNORM));
    state.mappings.get_mut(&surface_id).unwrap().page_entries = vec![1];
    crate::runtime::surface_cache::store(
        &mut state,
        surface_id,
        4,
        3,
        [255u8, 0, 0, 255].repeat(4 * 3),
    );

    let (width, height, sampled_mid, sampled) =
        resolve_sampled_source(&mut state, &mut host, 1, texture_ref, None)
            .expect("type-5 descriptor surface must sample");
    assert_eq!((width, height, sampled_mid), (4, 3, surface_id));
    let SampledSourceRequest::Bytes(sampled, _, _) = sampled else {
        panic!("cache-backed fixture unexpectedly resolved a resident target");
    };
    assert_eq!(&sampled[..4], &[0, 0, 255, 255]);

    // Regression guard for the resolve-once optimization
    // (SAMPLED-BIND-RESOLVE-ONCE): threading the caller-resolved object-list
    // entry must produce a byte-identical sample to a fresh internal lookup.
    // This is only sound because the guest object list is immutable for the
    // life of the draw; if a future change ever made a threaded entry diverge
    // from a fresh lookup (stale-content class), this fails.
    let threaded_entry = objects::lookup_list_entry(&state, &host, 1, texture_ref);
    assert!(
        threaded_entry.is_some(),
        "type-5 fixture must expose an object-list entry to thread"
    );
    let (tw, th, tmid, tsrc) =
        resolve_sampled_source(&mut state, &mut host, 1, texture_ref, threaded_entry)
            .expect("threaded-entry sample must resolve");
    assert_eq!(
        (tw, th, tmid),
        (width, height, sampled_mid),
        "threaded entry changed the resolved geometry/mid"
    );
    let SampledSourceRequest::Bytes(tsampled, _, _) = tsrc else {
        panic!("threaded-entry sample changed the source variant");
    };
    assert_eq!(
        tsampled, sampled,
        "threaded entry must yield byte-identical sampled content"
    );
}

/// Live Safari app-launch class: the type-4 base carries an unknown
/// 2-byte IOSurface FourCC (`LA08`) while the type-5 descriptor carries
/// the exact RG8 Metal view. Defaulting the base to BGRA asks for a
/// 632-byte row against the wire's 320-byte row and drops the draw.
#[test]
fn type5_sample_uses_serialized_rg8_view_over_unknown_surface_fourcc() {
    use crate::contract::endian::{st16, st32, st64};
    use crate::contract::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
    use crate::contract::iosurface_pages::{
        DEVICE_DESC_ALLOC_SIZE, DEVICE_DESC_BPE, DEVICE_DESC_BPR, DEVICE_DESC_DIMS,
        DEVICE_DESC_LEN, DEVICE_DESC_PIXEL_FORMAT, PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID,
    };
    use crate::contract::pixel_format::MTL_FORMAT_RG8_UNORM;
    use crate::runtime::decode::resource::{list_object_entry_offset, OBJECT_LIST_ENTRY_LEN};
    use crate::runtime::gva_mem;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();

    // One-level x86 GVA table for the object list and type-5 descriptor.
    let dir_pfn = 2u32;
    let root_pfn = 3u32;
    let dir_gpa = (dir_pfn as u64) << PAGE_SHIFT_X86;
    let root_gpa = (root_pfn as u64) << PAGE_SHIFT_X86;
    host.map_range(dir_gpa, 0x20, 0);
    host.map_range(root_gpa, 0x1000, 0);
    let mut dir = [0u8; 8];
    st32(&mut dir[DIRECTORY_ROOT_PFN as usize..], root_pfn);
    st32(&mut dir[DIRECTORY_DEPTH as usize..], 1);
    assert!(host.write_gpa(dir_gpa, &dir).is_ok());
    for i in 0..3u32 {
        let pfn = 4 + i;
        host.map_range((pfn as u64) << PAGE_SHIFT_X86, 0x1000, 0);
        let mut pte = [0u8; 4];
        st32(&mut pte, pfn);
        assert!(host.write_gpa(root_gpa + (i as u64) * 4, &pte).is_ok());
    }
    assert!(state.define_task(1, 0x1000, dir_pfn));
    assert!(state.set_object_list(1, 0, 256));

    let texture_ref = 248u32;
    let surface_id = 9u32;
    let width = 158u32;
    let height = 154u32;
    let surface_bpr = 320u32;
    let desc_gva = 0x1000u64;
    let desc_len = objects::TYPE5_ARG_RECORD + objects::TYPE5_RECORD_MIN_LEN;
    let mut desc = vec![0u8; desc_len];
    st32(&mut desc[objects::TYPE5_SURFACE_ID..], surface_id);
    st32(&mut desc[objects::TYPE5_ARG_OWN_REF..], texture_ref);
    desc[objects::TYPE5_ARG_RECORD] = objects::TYPE5_RECORD_TAG;
    st16(
        &mut desc[objects::TYPE5_ARG_RECORD + objects::TYPE5_RECORD_FORMAT..],
        MTL_FORMAT_RG8_UNORM,
    );
    st32(
        &mut desc[objects::TYPE5_ARG_RECORD + objects::TYPE5_RECORD_WIDTH..],
        width,
    );
    st32(
        &mut desc[objects::TYPE5_ARG_RECORD + objects::TYPE5_RECORD_HEIGHT..],
        height,
    );
    st32(
        &mut desc[objects::TYPE5_ARG_RECORD + objects::TYPE5_RECORD_DEPTH..],
        1,
    );
    assert!(
        gva_mem::write_task_gva(&mut host, &state.tasks[1], desc_gva, &desc, PAGE_SHIFT_X86,)
            .is_ok()
    );
    let list_off = list_object_entry_offset(texture_ref, 256).unwrap();
    let mut list_entry = [0u8; OBJECT_LIST_ENTRY_LEN];
    let packed = (objects::OBJECT_TYPE_REF_TEXTURE as u32) | ((desc_len as u32) << 8);
    st32(&mut list_entry, packed);
    list_entry[4..12].copy_from_slice(&desc_gva.to_le_bytes());
    assert!(gva_mem::write_task_gva(
        &mut host,
        &state.tasks[1],
        list_off,
        &list_entry,
        PAGE_SHIFT_X86,
    )
    .is_ok());

    // Exact live geometry: 13 x86 pages, 320-byte rows, two bytes/texel.
    let page = 1u64 << PAGE_SHIFT_X86;
    let page_count = 13u32;
    let gpa0 = 0x5100_0000u64;
    host.map_range(gpa0, (page * page_count as u64) as usize, 0);
    let mut native = vec![0u8; (surface_bpr * height) as usize];
    for y in 0..height as usize {
        for x in 0..width as usize {
            let off = y * surface_bpr as usize + x * 2;
            native[off] = (x % 251) as u8 + 1;
            native[off + 1] = (y % 251) as u8 + 1;
        }
    }
    assert!(host.write_gpa(gpa0, &native).is_ok());

    assert!(state.map_surface(surface_id));
    {
        let m = state.mappings.get_mut(&surface_id).unwrap();
        m.mapped = true;
        m.page_entries = (0..page_count)
            .map(|i| {
                let pfn = ((gpa0 >> PAGE_SHIFT_X86) as u32) + i;
                (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID
            })
            .collect();
    }
    assert!(state.set_mapping_geom(surface_id, width, height, 0));
    let mut device_desc = vec![0u8; DEVICE_DESC_LEN];
    st32(&mut device_desc[DEVICE_DESC_PIXEL_FORMAT..], 0x4c41_3038);
    st32(
        &mut device_desc[DEVICE_DESC_ALLOC_SIZE..],
        (page * page_count as u64) as u32,
    );
    st64(
        &mut device_desc[DEVICE_DESC_DIMS..],
        ((width as u64) << 8) | ((height as u64) << 40),
    );
    st32(&mut device_desc[DEVICE_DESC_BPR..], surface_bpr);
    st16(&mut device_desc[DEVICE_DESC_BPE..], 2);
    assert!(state.set_mapping_device_desc(surface_id, &device_desc));

    let (sample_w, sample_h, sample_mid, sampled) =
        resolve_sampled_source(&mut state, &mut host, 1, texture_ref, None)
            .expect("serialized RG8 view must sample the 2-byte surface");
    assert_eq!(
        (sample_w, sample_h, sample_mid),
        (width, height, surface_id)
    );
    let SampledSourceRequest::Bytes(sampled, _, byte_format) = sampled else {
        panic!("serialized view unexpectedly resolved a resident target");
    };
    // Native RG8 upload: two bytes per texel, tight rows (an R8G8_UNORM
    // Vulkan image samples these identically to the old CPU (r,g,0,255)
    // RGBA8 expansion).
    assert_eq!(byte_format, TexelLayout::Rg8);
    assert_eq!(sampled.len(), (width * height * 2) as usize);
    assert_eq!(&sampled[..4], &[1, 1, 2, 1]);
    let last = ((height - 1) as usize * width as usize + (width - 1) as usize) * 2;
    assert_eq!(
        &sampled[last..last + 2],
        &[158, 154],
        "row padding must not enter the RG8 view"
    );
}

/// The type-5 view memo: unchanged plane bytes reuse the converted Arc and
/// carry a stable content identity (engine upload skipped); a guest write
/// to the plane is observed on the next bind and mints a new generation.
#[test]
fn type5_view_memo_reuses_unchanged_planes_and_invalidates_on_write() {
    use crate::contract::endian::{st16, st32, st64};
    use crate::contract::iosurface_pages::{
        DEVICE_DESC_ALLOC_SIZE, DEVICE_DESC_BPE, DEVICE_DESC_BPR, DEVICE_DESC_DIMS,
        DEVICE_DESC_LEN, DEVICE_DESC_PIXEL_FORMAT, PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID,
    };
    use crate::contract::pixel_format::MTL_FORMAT_RG8_UNORM;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
    let surface_id = 9u32;
    let width = 158u32;
    let height = 154u32;
    let surface_bpr = 320u32;
    let page = 1u64 << PAGE_SHIFT_X86;
    let page_count = 13u32;
    let gpa0 = 0x5100_0000u64;
    host.map_range(gpa0, (page * page_count as u64) as usize, 0);
    let mut native = vec![0u8; (surface_bpr * height) as usize];
    for y in 0..height as usize {
        for x in 0..width as usize {
            let off = y * surface_bpr as usize + x * 2;
            native[off] = (x % 251) as u8 + 1;
            native[off + 1] = (y % 251) as u8 + 1;
        }
    }
    assert!(host.write_gpa(gpa0, &native).is_ok());
    assert!(state.map_surface(surface_id));
    {
        let m = state.mappings.get_mut(&surface_id).unwrap();
        m.mapped = true;
        m.page_entries = (0..page_count)
            .map(|i| {
                let pfn = ((gpa0 >> PAGE_SHIFT_X86) as u32) + i;
                (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID
            })
            .collect();
    }
    assert!(state.set_mapping_geom(surface_id, width, height, 0));
    let mut device_desc = vec![0u8; DEVICE_DESC_LEN];
    st32(&mut device_desc[DEVICE_DESC_PIXEL_FORMAT..], 0x4c41_3038);
    st32(
        &mut device_desc[DEVICE_DESC_ALLOC_SIZE..],
        (page * page_count as u64) as u32,
    );
    st64(
        &mut device_desc[DEVICE_DESC_DIMS..],
        ((width as u64) << 8) | ((height as u64) << 40),
    );
    st32(&mut device_desc[DEVICE_DESC_BPR..], surface_bpr);
    st16(&mut device_desc[DEVICE_DESC_BPE..], 2);
    assert!(state.set_mapping_device_desc(surface_id, &device_desc));
    let view = objects::Type5TextureView {
        pixel_format: MTL_FORMAT_RG8_UNORM,
        width,
        height,
        depth: 1,
        plane_index: 0,
    };

    let (w1, h1, rgba1, id1, fmt1) =
        load_type5_view_rgba(&mut state, &mut host, 1, 248, surface_id, view)
            .expect("first materialization");
    assert_eq!((w1, h1), (width, height));
    assert_eq!(
        fmt1,
        TexelLayout::Rg8,
        "an RG8 chroma plane uploads at native footprint, not CPU-expanded to RGBA8"
    );
    // Native footprint: two bytes per texel, tight rows (no RGBA8 expand).
    assert_eq!(rgba1.len(), (width * height * 2) as usize);
    assert!(
        id1.generation > (1u64 << 32),
        "type-5 identities share the guest-linear generation namespace"
    );
    assert_eq!(
        id1.key,
        (1u64 << 63) | surface_id as u64,
        "identity key namespaces type-5 content above GVA keys"
    );

    let (_, _, rgba2, id2, _) =
        load_type5_view_rgba(&mut state, &mut host, 1, 248, surface_id, view)
            .expect("memo revalidation");
    assert!(
        std::sync::Arc::ptr_eq(&rgba1, &rgba2),
        "unchanged plane bytes must reuse the native allocation"
    );
    assert_eq!(id1, id2, "unchanged content keeps its identity");

    // Guest CPU writes one texel; the next bind must observe it.
    assert!(host.write_gpa(gpa0 + 6, &[0xAA, 0xBB]).is_ok());
    let (_, _, rgba3, id3, _) =
        load_type5_view_rgba(&mut state, &mut host, 1, 248, surface_id, view)
            .expect("re-materialization after guest write");
    assert!(
        id3.generation > id2.generation,
        "guest write mints a new generation"
    );
    // Native RG8: texel 3 sits at tight byte offset 3*2 = 6 (an R8G8_UNORM
    // Vulkan image samples this to (0xAA/255, 0xBB/255, 0, 1), identical to
    // the CPU-expanded (0xAA,0xBB,0,255) RGBA8 the old path produced).
    assert_eq!(
        &rgba3[6..8],
        &[0xAA, 0xBB],
        "the new native plane bytes must be observed"
    );
    assert!(!std::sync::Arc::ptr_eq(&rgba1, &rgba3));
}

#[test]
fn type5_view_materializes_only_when_base_identity_differs() {
    use crate::contract::pixel_format::MTL_FORMAT_RG8_UNORM;

    let exact = objects::Type5TextureView {
        pixel_format: MTL_FORMAT_BGRA8_UNORM,
        width: 1920,
        height: 1080,
        depth: 1,
        plane_index: 0,
    };
    assert!(!type5_view_requires_materialization(
        true,
        1920,
        1080,
        MTL_FORMAT_BGRA8_UNORM,
        exact
    ));
    assert!(type5_view_requires_materialization(
        true, 1920, 1080, 0, exact
    ));

    let rg8_view = objects::Type5TextureView {
        pixel_format: MTL_FORMAT_RG8_UNORM,
        width: 158,
        height: 154,
        depth: 1,
        plane_index: 0,
    };
    assert!(type5_view_requires_materialization(
        true,
        158,
        154,
        MTL_FORMAT_BGRA8_UNORM,
        rg8_view
    ));
    assert!(type5_view_requires_materialization(
        false,
        158,
        154,
        MTL_FORMAT_RG8_UNORM,
        rg8_view
    ));
    let volume = objects::Type5TextureView { depth: 2, ..exact };
    assert!(type5_view_requires_materialization(
        true,
        1920,
        1080,
        MTL_FORMAT_BGRA8_UNORM,
        volume
    ));
}

#[test]
fn texture_view_declines_are_specific_and_log_safe() {
    let cases = [
        TextureViewDecline::HopEntryMissing { texture_ref: 1 },
        TextureViewDecline::HopObjectNotView {
            texture_ref: 1,
            object_type: 2,
        },
        TextureViewDecline::HopDescriptorMissing {
            texture_ref: 1,
            descriptor_length: 4,
        },
        TextureViewDecline::HopDecode {
            texture_ref: 1,
            opcode: 9,
            declared: 4,
            descriptor_len: 4,
            bytes_hex: "01020304".into(),
            reason: DecodeStatus::ErrShort("res_texture_view_short"),
        },
        TextureViewDecline::HopZeroBase {
            texture_ref: 1,
            opcode: 9,
        },
        TextureViewDecline::HopLevelOverflow {
            texture_ref: 1,
            level_base: u64::MAX,
        },
        TextureViewDecline::HopSwizzleInvalid {
            texture_ref: 1,
            selectors: [0, 1, 2, 9],
        },
        TextureViewDecline::ChainSelfOrZero {
            base: 1,
            next: 1,
            depth: 1,
        },
        TextureViewDecline::ChainOverflow { base: 1, depth: 8 },
    ];
    let mut slugs = std::collections::HashSet::new();
    for decline in cases {
        assert!(slugs.insert(decline.slug()), "duplicate {}", decline.slug());
        for (_, value) in decline.fields() {
            assert!(!value.contains(char::is_whitespace));
        }
    }
    assert_eq!(slugs.len(), 9);
}

#[test]
fn texture_view_decline_preserves_decode_leaf_and_chain_identity() {
    let decode = TextureViewDecline::HopDecode {
        texture_ref: 7,
        opcode: 9,
        declared: 12,
        descriptor_len: 8,
        bytes_hex: "01020304".into(),
        reason: DecodeStatus::ErrShort("res_texture_view_short"),
    };
    assert_eq!(decode.slug(), "res_texture_view_short");
    let fields = decode.fields();
    assert!(fields.contains(&("texture_ref", "7".into())));
    assert!(fields.contains(&("opcode", "0x9".into())));
    assert!(fields.contains(&("declared", "12".into())));
    assert!(fields.contains(&("descriptor_len", "8".into())));
    assert!(fields.contains(&("bytes", "01020304".into())));

    let chain = TextureViewDecline::ChainSelfOrZero {
        base: 11,
        next: 11,
        depth: 3,
    };
    assert_eq!(chain.slug(), "texture_view_chain_self_or_zero");
    let fields = chain.fields();
    assert!(fields.contains(&("base", "11".into())));
    assert!(fields.contains(&("next", "11".into())));
    assert!(fields.contains(&("depth", "3".into())));
}

/// Every type-5 view refusal names its rail (`type5_view_`), renders
/// whitespace-free fields, and is distinct — the same discipline the
/// capture and import rails took, so `grep reason=type5_view_…` stays
/// answerable against the blit rail's `t5_*` copy vocabulary next door.
#[test]
fn every_type5_view_reason_is_namespaced_distinct_and_log_safe() {
    use crate::observe::Decline as _;
    const ALL: &[Type5ViewDecline] = &[
        Type5ViewDecline::UnsupportedDepth { depth: 0 },
        Type5ViewDecline::Unresolved,
        Type5ViewDecline::FormatBpp,
        Type5ViewDecline::NoMapping,
        Type5ViewDecline::SampleWindow {
            base_w: 0,
            base_h: 0,
            base_fmt: 0,
            desc: None,
        },
        Type5ViewDecline::Span {
            pages: 0,
            page_bytes: 0,
            span_end: 0,
            bpr: 0,
        },
        Type5ViewDecline::TightOverflow { bpp: 0 },
        Type5ViewDecline::NativeLen { tight: 0 },
        Type5ViewDecline::Read {
            base_w: 0,
            base_h: 0,
            base_fmt: 0,
            off: 0,
            bpr: 0,
            span_end: 0,
            pages: 0,
        },
        Type5ViewDecline::RgbaStride,
        Type5ViewDecline::RgbaLen { stride: 0 },
        Type5ViewDecline::Convert { row: 0, bpp: 0 },
    ];
    let mut slugs: Vec<&str> = Vec::new();
    for d in ALL {
        assert!(
            d.slug().starts_with("type5_view_"),
            "{} is not namespaced to the type-5 view rail",
            d.slug()
        );
        for (k, v) in d.fields() {
            assert!(!k.contains(' ') && !v.contains(' '), "{k}={v}");
        }
        slugs.push(d.slug());
    }
    slugs.sort_unstable();
    let before = slugs.len();
    slugs.dedup();
    assert_eq!(before, slugs.len(), "duplicate Type5ViewDecline slug");
}

/// `SampleWindow` is the only variant carrying transcribed field logic: the
/// base geometry plus the decoded device descriptor, or `desc=missing` when
/// the descriptor could not be decoded. Both branches must render exactly
/// what the old ad-hoc `detail` string did.
#[test]
fn sample_window_renders_the_descriptor_or_its_absence() {
    let present = Type5ViewDecline::SampleWindow {
        base_w: 320,
        base_h: 240,
        base_fmt: 0x50,
        desc: Some((64, 64, 0x4c41_3038, 256, 4096)),
    };
    assert_eq!(
            crate::observe::Emit::decline("type5_draw_view", &present).render(),
            "type5_draw_view reason=type5_view_sample_window base=320x240 base_fmt=0x50 desc=64x64 desc_fmt=0x4c413038 bpr=256 alloc=4096"
        );

    let missing = Type5ViewDecline::SampleWindow {
        base_w: 320,
        base_h: 240,
        base_fmt: 0x50,
        desc: None,
    };
    assert_eq!(
        crate::observe::Emit::decline("type5_draw_view", &missing).render(),
        "type5_draw_view reason=type5_view_sample_window base=320x240 base_fmt=0x50 desc=missing"
    );
}

/// A secondary MRT attachment binds **its own** blend, not slot 0's and not
/// "unblended because secondaries are always masks".
///
/// The regression this locks: `caches.rs` forced every secondary attachment
/// `blend_enable(false)`, justified by a comment claiming the decode side
/// carried no per-attachment blend state. It carried it all along — the
/// Metal arm reads exactly these fields per slot — so a guest MRT pipeline
/// that asked to blend slot 1 got a raw store instead.
#[cfg(feature = "backend-vulkan")]
#[test]
fn a_secondary_mrt_slot_binds_its_own_blend() {
    use crate::runtime::decode::resource::{PipelineColorAttachment, RenderPipelineDescriptor};
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);

    // Slot 0 blends src-alpha over; slot 1 blends ONE/ONE (additive). Two
    // different blends so borrowing slot 0's would be visible.
    let pipeline = RenderPipelineDescriptor {
        color_attachments: vec![
            PipelineColorAttachment {
                slot: 0,
                blending_enabled: true,
                src_rgb: 4, // MTLBlendFactorSourceAlpha
                dst_rgb: 5, // MTLBlendFactorOneMinusSourceAlpha
                op_rgb: 0,  // MTLBlendOperationAdd
                src_alpha: 4,
                dst_alpha: 5,
                op_alpha: 0,
                ..PipelineColorAttachment::default()
            },
            PipelineColorAttachment {
                slot: 1,
                blending_enabled: true,
                src_rgb: 1, // MTLBlendFactorOne
                dst_rgb: 1, // MTLBlendFactorOne
                op_rgb: 0,
                src_alpha: 1,
                dst_alpha: 1,
                op_alpha: 0,
                ..PipelineColorAttachment::default()
            },
        ],
        ..RenderPipelineDescriptor::default()
    };

    let colors = vec![
        ColorRtRequest {
            slot: 0,
            texture_ref: 10,
            target_gva: 0x1000,
            width: 64,
            height: 64,
            format: MTL_FORMAT_BGRA8_UNORM,
            ..ColorRtRequest::default()
        },
        ColorRtRequest {
            slot: 1,
            texture_ref: 11,
            target_gva: 0x2000,
            width: 64,
            height: 64,
            format: crate::contract::pixel_format::MTL_FORMAT_RG16_FLOAT,
            ..ColorRtRequest::default()
        },
    ];
    let primary = crate::backend::vulkan::engine::TargetIdentity::Gva {
        gva: 0x1000,
        width: 64,
        height: 64,
        generation: 0,
    };

    let secs = build_secondary_targets(&state, &colors, &pipeline, &primary, 64, 64, [0.0; 4]);
    assert_eq!(secs.len(), 1, "one secondary attachment expected");
    let blend = secs[0].blend.expect(
        "slot 1 declares blending_enabled — before this fix every secondary \
             was forced unblended",
    );
    use crate::backend::vulkan::engine::{BlendFactor, BlendOp};
    assert_eq!(blend.src_color, BlendFactor::One, "slot 1's own src factor");
    assert_eq!(blend.dst_color, BlendFactor::One, "slot 1's own dst factor");
    assert_eq!(blend.color_op, BlendOp::Add);
    // The tell that it is not slot 0's: slot 0 asked for SrcAlpha/OneMinusSrcAlpha.
    assert_ne!(blend.src_color, BlendFactor::SrcAlpha);

    // A slot the pipeline does not blend stays unblended rather than
    // inheriting slot 0's — there is no `or_else(first())` fallback here.
    let unblended = RenderPipelineDescriptor {
        color_attachments: vec![
            pipeline.color_attachments[0],
            PipelineColorAttachment {
                slot: 1,
                blending_enabled: false,
                ..PipelineColorAttachment::default()
            },
        ],
        ..RenderPipelineDescriptor::default()
    };
    let secs = build_secondary_targets(&state, &colors, &unblended, &primary, 64, 64, [0.0; 4]);
    assert_eq!(secs.len(), 1);
    assert!(
        secs[0].blend.is_none(),
        "slot 1 declares no blend; it must not inherit slot 0's"
    );
    let _ = &mut state;
}

#[cfg(feature = "backend-vulkan")]
#[test]
fn fixed_state_gap_names_every_unrepresented_field() {
    // In-contract cull (Back) + winding (CCW) are HONORED by the Vulkan
    // raster state; the depth test + attachment and the stencil test
    // (op state + reference + clear) are honored via `resources.depth` — so
    // cull/front/depth_stencil/depth_attach/stencil_ref/stencil_attach are no
    // longer gaps. Only depth bias remains unrepresented.
    // MTLCullModeBack / MTLWindingCounterClockwise, per `translate::raster`,
    // which is now the only place those SDK values are spelled.
    let mut req = DrawEncodeRequest {
        cull_mode: Some(2),
        front_facing: Some(1),
        depth_bias: Some([1.25, 2.5, 0.0]),
        depth_stencil_ref: 77,
        stencil_ref: Some((3, 4)),
        depth_attach: Some(DepthAttachment::default()),
        stencil_attach: Some(StencilAttachment::default()),
        ..DrawEncodeRequest::default()
    };
    assert_eq!(vulkan_fixed_state_gap(&req), "bias:1.250/2.500/0.000");

    // An OUT-OF-CONTRACT cull/winding value stays a gap (fail-visible), never
    // coerced to a face that silently draws or drops geometry.
    req.depth_bias = None;
    req.depth_stencil_ref = 0;
    req.stencil_ref = None;
    req.depth_attach = None;
    req.stencil_attach = None;
    req.cull_mode = Some(9);
    req.front_facing = Some(7);
    assert_eq!(vulkan_fixed_state_gap(&req), "cull:9,front:7");

    req.cull_mode = None;
    req.front_facing = None;
    assert_eq!(vulkan_fixed_state_gap(&req), "");
}

/// Recycled texture_ref must not serve a prior full-frame encode as a
/// different-sized linear sample (namespace / geom-match class).
#[test]
fn texture_ref_cache_geom_mismatch_does_not_hit_get_texture() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let tex_ref = 53u32;
    let mut full = vec![0u8; 1920 * 1152 * 4];
    full[3] = 255;
    host_cache_store_rgba8(&mut state, tex_ref, 1920, 1152, &full);
    // Exact geom hit
    assert!(crate::runtime::surface_cache::get_texture(&state, tex_ref, 1920, 1152).is_some());
    // Wrong geom (type-3 L0 recycle) miss
    assert!(crate::runtime::surface_cache::get_texture(&state, tex_ref, 115, 16).is_none());
    // surface_id map must stay empty for texture_ref stores
    assert!(crate::runtime::surface_cache::get(&state, tex_ref, 1920, 1152).is_none());
}

/// Slicing memoized whole-backing runs must honor offsets that skip whole
/// runs, split within a run, and cross run boundaries; an over-long slice
/// fails closed.
#[test]
#[cfg(feature = "backend-vulkan")]
fn slice_runs_to_engine_crosses_run_boundaries() {
    use crate::model::GuestRunSpan;
    let spans = [
        GuestRunSpan {
            host_ptr: 0x1000,
            len: 0x100,
        },
        GuestRunSpan {
            host_ptr: 0x9000,
            len: 0x80,
        },
        GuestRunSpan {
            host_ptr: 0x2_0000,
            len: 0x200,
        },
    ];
    // Slice inside the first run.
    let s = slice_runs_to_engine(&spans, 0x10, 0x20).unwrap();
    assert_eq!(s.len(), 1);
    assert_eq!((s[0].host_ptr, s[0].len), (0x1010, 0x20));
    // Slice crossing run 1 → run 2 → into run 3.
    let s = slice_runs_to_engine(&spans, 0xf0, 0x100).unwrap();
    assert_eq!(s.len(), 3);
    assert_eq!((s[0].host_ptr, s[0].len), (0x10f0, 0x10));
    assert_eq!((s[1].host_ptr, s[1].len), (0x9000, 0x80));
    assert_eq!((s[2].host_ptr, s[2].len), (0x2_0000, 0x70));
    // Offset skipping the first two runs entirely.
    let s = slice_runs_to_engine(&spans, 0x180, 0x200).unwrap();
    assert_eq!(s.len(), 1);
    assert_eq!((s[0].host_ptr, s[0].len), (0x2_0000, 0x200));
    // Over-long slice fails closed.
    assert!(slice_runs_to_engine(&spans, 0x180, 0x201).is_none());
    assert!(slice_runs_to_engine(&spans, 0, 0).is_none());
}

/// A memo hit returns the cached Arc without walking the page table; the
/// Unmap-overlap retirement drops exactly the aliasing entry (the
/// gva_host_views invalidation contract).
#[test]
#[cfg(feature = "backend-vulkan")]
fn guest_run_memo_hit_and_unmap_retirement() {
    use crate::model::{GuestRunMemoEntry, GuestRunSpan};
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
    host.stable_map_pages = true;
    let runs = std::sync::Arc::new(vec![GuestRunSpan {
        host_ptr: 0xabc000,
        len: 0x40000,
    }]);
    state.guest_run_memo.push_back(GuestRunMemoEntry {
        task_id: 3,
        gva: 0x50_0000,
        length: 0x40000,
        runs: runs.clone(),
    });
    // Hit: same Arc back, no PT walk attempted (FakeHost has no task 3 PT
    // — a walk would return None), hit counter bumps.
    let got = guest_runs_memoized(&mut state, &mut host, 3, 0x50_0000, 0x40000).unwrap();
    assert!(std::sync::Arc::ptr_eq(&got, &runs));
    assert_eq!(state.tranche.run_memo_hit, 1);
    assert_eq!(state.tranche.run_memo_miss, 0);
    // Different span → miss path → walk fails (no PT) → None, no entry.
    assert!(guest_runs_memoized(&mut state, &mut host, 3, 0x50_0000, 0x20000).is_none());
    // Unmap overlapping the span retires the entry; a disjoint unmap does not.
    crate::runtime::gva_view::retire_gva_views_overlapping(&mut state, 3, 0x60_0000, 0x1000);
    assert_eq!(state.guest_run_memo.len(), 1);
    crate::runtime::gva_view::retire_gva_views_overlapping(&mut state, 3, 0x52_0000, 0x1000);
    assert!(state.guest_run_memo.is_empty());
}

/// A cached guest-run host pointer is only valid when the host has declared
/// `map_pages` views stable. Arm64 MMIO remap views are transient, so the
/// runtime must decline before using even a memoized span.
#[test]
#[cfg(feature = "backend-vulkan")]
fn guest_run_memo_declines_on_unstable_host_mappings() {
    use crate::model::{GuestRunMemoEntry, GuestRunSpan};
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
    state.guest_run_memo.push_back(GuestRunMemoEntry {
        task_id: 3,
        gva: 0x50_0000,
        length: 0x40000,
        runs: std::sync::Arc::new(vec![GuestRunSpan {
            host_ptr: 0xabc000,
            len: 0x40000,
        }]),
    });

    assert!(guest_runs_memoized(&mut state, &mut host, 3, 0x50_0000, 0x40000).is_none());
    assert_eq!(state.tranche.run_memo_hit, 0);
    assert_eq!(state.tranche.run_memo_miss, 0);
}

/// The 1-in-64 sampled staleness verify: a PT rewire the notify hooks
/// missed is detected on the sampled hit, fail-logged, self-heals the
/// entry to the fresh runs, and counts as rmemo_stale; a dead span
/// (walk no longer resolves) retires the entry.
#[test]
#[cfg(feature = "backend-vulkan")]
fn guest_run_memo_stale_probe_detects_pt_rewire() {
    use crate::contract::endian::st32;
    use crate::contract::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
    let page_shift = crate::model::PAGE_SHIFT_X86;
    let page = 1u64 << page_shift;
    let mut host = FakeHost::new();
    host.strict_linux_map = true;
    host.stable_map_pages = true;
    let dir_gpa = 2u64 << page_shift;
    let root_gpa = 3u64 << page_shift;
    let data0 = 4u64 << page_shift;
    let data1 = 10u64 << page_shift;
    host.map_range(dir_gpa, page as usize, 0);
    host.map_range(root_gpa, page as usize, 0);
    host.map_range(data0, page as usize, 0);
    host.map_range(data1, page as usize, 0);
    let mut d = [0u8; 8];
    st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 3);
    st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
    host.write_gpa(dir_gpa, &d).unwrap();
    let mut pte = [0u8; 4];
    st32(&mut pte, 4);
    host.write_gpa(root_gpa, &pte).unwrap();

    let mut state = DeviceState::new(DeviceId(1), page_shift);
    assert!(state.define_task(1, page, 2));
    let gva = 8u64;
    // Miss → walk → entry memoized against data0.
    let first = guest_runs_memoized(&mut state, &mut host, 1, gva, 16).unwrap();
    assert_eq!(state.tranche.run_memo_miss, 1);
    let ptr0 = host.map_pages(&[data0], page as usize).unwrap() + gva as usize;
    assert_eq!(first[0].host_ptr, ptr0);

    // Rewire the PTE to data1 with no Unmap notify; force the sampled
    // verify (hit counter lands on a multiple of 64).
    st32(&mut pte, 10);
    host.write_gpa(root_gpa, &pte).unwrap();
    state.tranche.run_memo_hit = 63;
    let healed = guest_runs_memoized(&mut state, &mut host, 1, gva, 16).unwrap();
    assert_eq!(
        state.tranche.run_memo_stale, 1,
        "probe must flag the rewire"
    );
    let ptr1 = host.map_pages(&[data1], page as usize).unwrap() + gva as usize;
    assert_eq!(
        healed[0].host_ptr, ptr1,
        "entry must self-heal to fresh runs"
    );
    // Healed entry serves subsequent (unsampled) hits.
    let again = guest_runs_memoized(&mut state, &mut host, 1, gva, 16).unwrap();
    assert!(std::sync::Arc::ptr_eq(&again, &healed));

    // Dead span: PTE cleared to an unmapped page → sampled verify walks
    // None → entry retired, caller falls back.
    st32(&mut pte, 0x7fff);
    host.write_gpa(root_gpa, &pte).unwrap();
    state.tranche.run_memo_hit = 127;
    assert!(guest_runs_memoized(&mut state, &mut host, 1, gva, 16).is_none());
    assert_eq!(state.tranche.run_memo_stale, 2);
    assert!(state.guest_run_memo.is_empty());
}
