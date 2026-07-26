//! The crate-wide decline vocabulary: one trait, one registry.
//!
//! # What a decline is
//!
//! A **decline** is any path that rejects, drops, degrades or mis-executes a
//! decoded guest command. `AGENTS.md` I2 requires each one to name a *specific*
//! reason — not a coarse status a dozen distinct checks collapse into. The
//! canonical failure this prevents is already in the ground rules: a 16 MiB cap
//! returning a bare `Unsupported` alongside six other checks, invisible for a
//! day because the log could not tell them apart.
//!
//! # Why a registry and not a derive
//!
//! The registry is a `const` table, deliberately. A derive would keep the slugs
//! correct but leave them uncountable: you could not read the file and learn
//! what the crate can refuse, and you could not pin a census that only moves
//! when someone means it. This is the same choice `translate/coverage.rs`
//! makes for field dispositions, for the same reason — the table *is* the
//! document.
//!
//! Registering a type is therefore a deliberate act with a cost: you name where
//! it is defined and where it reaches the sink, and `super::gate` checks both.
//!
//! # Adding a decline type
//!
//! 1. Implement [`Decline`] on it — one slug per distinct check, never shared.
//! 2. Add a [`DeclineClass`] row below naming its file, its emission sites and
//!    every slug it can produce.
//! 3. Emit it through [`super::emit::Emit`], which cannot render a line without
//!    a slug.
//!
//! The gates then hold you to it: slugs are unique crate-wide, log-safe, and
//! every registered type demonstrably reaches the sink.

/// A typed refusal that can name itself in the always-on log.
///
/// Modelled on `TranslateReason`, which had the right shape before this trait
/// existed: the payload rides along with the variant, and rendering produces
/// `reason=<slug>` plus the values that caused it.
pub trait Decline {
    /// Stable snake_case slug for `reason=` in `/tmp/reims-vgpu-fail.log`.
    ///
    /// **One slug per distinct check.** Two checks sharing a slug is the exact
    /// defect this vocabulary exists to prevent: you grep the log, watch the
    /// slug fire, and still cannot tell which check refused.
    fn slug(&self) -> &'static str;

    /// The load-bearing values behind this decline — refs, dims, formats,
    /// offsets, sizes, caps. Rendered after `reason=` as `k=v` pairs.
    ///
    /// A decline that names only its class leaves the reader without the value
    /// that caused it, which is half a diagnostic. Allocation here is fine:
    /// declines are rare by construction, and a flood is a bug the sink's own
    /// detector will report.
    fn fields(&self) -> Vec<(&'static str, String)> {
        Vec::new()
    }
}

/// A status enum that mixes success with refusal.
///
/// [`Decline`] is for a value that is *always* a refusal — a `DrawError`, a
/// `TranslateReason`. Most of this crate's older vocabulary is not shaped that
/// way: `DecodeStatus`, `BlitStatus`, `IcbStatus` and their siblings carry `Ok`
/// and `Done` alongside a dozen genuine refusals, and the difference matters
/// because I2's carve-out lives exactly there. A resolver answering "not ready
/// yet" every poll must **not** reach the log; a malformed length must.
///
/// That judgement is one no gate can make, so this trait makes it once, in an
/// exhaustive `match` the compiler forces you to revisit when a variant
/// appears. A new status variant cannot compile until its author has said which
/// side of the line it falls on.
pub trait Refusal {
    /// The registered slug when this value refused; `None` when it is control
    /// flow — success, "done", "not ready yet" — that must stay out of the log.
    fn refusal(&self) -> Option<&'static str>;

    /// The load-bearing values behind the refusal, as for [`Decline::fields`].
    /// Returning nothing is normal here: a status usually carries no payload and
    /// the failing site adds the refs and sizes with [`super::Emit::field`].
    fn fields(&self) -> Vec<(&'static str, String)> {
        Vec::new()
    }
}

/// Whether a registered decline actually reaches `/tmp/reims-vgpu-fail.log`.
///
/// This is the field that closes the unchecked handoff `translate/` and `caps/`
/// otherwise leave open: they are pure and correctly log nothing, but *someone*
/// must, and until this existed nothing checked that anyone did.
#[derive(Clone, Copy, Debug)]
pub enum Emission {
    /// `(file, event)` pairs where a value of this type meets an `observe::`
    /// emitter — `event` being the first argument of the `Emit::decline` /
    /// `Emit::refusal` call that logs it.
    ///
    /// The event token is what makes this a claim about a *line*. Naming only
    /// the file let a row pass on any mention of the type name anywhere in it,
    /// including an unused `use` — four `DecodeStatus` rows were doing exactly
    /// that, satisfied by `DecodeStatus as StreamStatus` in an import list, and
    /// would have kept passing if their emission had been deleted outright.
    At(&'static [(&'static str, &'static str)]),
    /// Nothing can log it because nothing calls it: the type is returned only
    /// by methods that have no callers. Carries the argument, and is a standing
    /// invitation to delete the dead surface rather than a place to park types.
    Unreachable(&'static str),
}

/// One registered decline type: what it is, where it lives, where it is logged,
/// and every slug it can produce.
#[derive(Clone, Copy, Debug)]
pub struct DeclineClass {
    /// The Rust type name, e.g. `"BackendError"`.
    pub type_name: &'static str,
    /// Defining file, relative to `src/`. Checked to exist and to contain the
    /// type.
    pub defined_in: &'static str,
    /// Where — or whether — a value of this type reaches the sink.
    pub emission: Emission,
    /// Extra `(file, anchor)` blocks the gate must also read, for a vocabulary
    /// delegated to a helper — `BackendError::Unsupported` forwards its nine
    /// slugs to `BackendOp::slug`, so the gate is told to read `impl BackendOp`
    /// as well. `anchor` is the text that opens the block; the gate walks its
    /// braces.
    ///
    /// Empty is the normal case: the slugs are written in the type's own
    /// `Decline`/`Refusal` impl in [`Self::defined_in`], which the gate always
    /// reads. Naming a delegate is what keeps a forwarded slug counted instead
    /// of silently exempt.
    pub slug_blocks: &'static [(&'static str, &'static str)],
    /// `(file, call)` pairs whose *construction sites* write this type's
    /// vocabulary, for a reason that lives in the value or in a side channel
    /// rather than in a `match` arm — `FenceStatus::Unsupported(
    /// "fence_domain_unknown")`, or the blit rail's `br(status, "slug")`. The
    /// gate reads the first string literal inside each call's parentheses.
    ///
    /// Empty is the normal case. This exists because a 178-slug vocabulary is
    /// never written as 178 match arms, and a census that could only read match
    /// arms would have to exempt exactly the largest rails.
    pub slug_calls: &'static [(&'static str, &'static str)],
    /// Every slug this type can produce.
    pub slugs: &'static [&'static str],
}

/// Every decline type the crate reports through [`Decline`].
///
/// Ordered by subsystem so a reader can see at a glance which subsystems can
/// refuse and which are still silent — the latter being, as of this phase, the
/// remaining work rather than a claim of completeness.
pub const REGISTRY: &[DeclineClass] = &[
    DeclineClass {
        type_name: "BackendError",
        defined_in: "backend/mod.rs",
        // `Unsupported` forwards its nine slugs to `BackendOp::slug`.
        slug_blocks: &[("backend/mod.rs", "impl BackendOp")],
        // Measured 2026-07-25: nothing in the crate calls a fallible `Backend`
        // method. `Device<B>` invokes only `reset()`, which returns `()`; the
        // runtime drives `engine::execute_draw_request` directly rather than
        // going through this trait. So the type `AGENTS.md` cites as the
        // canonical silent-failure example turns out to sit on a dead path —
        // typing it removes the payload-free variant, but no emission can be
        // exercised until the trait is driven or its fallible surface deleted.
        emission: Emission::Unreachable(
            "no caller: Device<B> uses only Backend::reset(); the fallible \
             trait surface is vestigial",
        ),
        slug_calls: &[],
        slugs: &[
            // One per declined operation — the 19 construction sites of the
            // old payload-free `Unsupported` collapse to these nine checks,
            // disambiguated by the `backend=` field.
            "unsupported_write_texture",
            "unsupported_read_texture",
            "unsupported_set_pipeline_library",
            "unsupported_execute_blit",
            "unsupported_execute_compute",
            "unsupported_execute_render",
            "unsupported_render_draw",
            "unsupported_present",
            "unsupported_encode_simple_draw",
            "backend_invalid_argument",
            "backend_resource_missing",
            "backend_shader_error",
            "backend_device_lost",
            "backend_other",
        ],
    },
    DeclineClass {
        type_name: "FailEvent",
        defined_in: "model/state.rs",
        // The malformed-packet and unsupported-exec variants forward to their
        // fault enums, which is where thirteen former `&'static str` literals
        // now live as checked variants.
        slug_blocks: &[
            ("model/state.rs", "impl PacketFault"),
            ("model/state.rs", "impl ExecFault"),
        ],
        // `DeviceState::record_fail` is the single funnel: every FailEvent in the
        // crate is constructed and handed to it, so one site covers the type.
        emission: Emission::At(&[("model/state.rs", "fail_event")]),
        slug_calls: &[],
        slugs: &[
            "unknown_root_opcode",
            "unknown_child_opcode",
            "bad_mmio_access",
            // PacketFault — one per distinct malformed-packet check.
            "packet_desynced_head_tail",
            "packet_bad_size",
            "packet_desynced",
            "packet_root_header_read",
            "packet_root_snap_read",
            "packet_root_stamp_writeback",
            "packet_child_header_read",
            "packet_child_regs_base_read",
            "packet_child_regs_head_read",
            "packet_child_regs_stamp_read",
            "packet_child_snap_read",
            "packet_child_tail_read",
            "packet_child_head_writeback",
            // ExecFault.
            "exec_indirect2_short",
        ],
    },
    DeclineClass {
        type_name: "StateMutationDecline",
        defined_in: "model/state.rs",
        slug_blocks: &[],
        // Rejected state mutations log at the state invariant itself. Expected
        // lookup/lifecycle misses (absent unmap, duplicate delete, cache miss)
        // do not construct this type and remain silent control flow.
        emission: Emission::At(&[("model/state.rs", "model_state_mutation")]),
        slug_calls: &[],
        slugs: &[
            "model_define_task_id_range",
            "model_delete_task_id_range",
            "model_set_object_list_task_id_range",
            "model_set_object_list_task_inactive",
            "model_insert_object_task_id_range",
            "model_insert_object_task_inactive",
            "model_map_surface_id_range",
            "model_unmap_surface_id_range",
            "model_attach_mapping_id_range",
            "model_attach_mapping_internal_zero",
            "model_mapping_device_desc_id_range",
            "model_mapping_device_desc_empty",
            "model_mapping_geom_id_range",
            "model_mapping_geom_width_zero",
            "model_mapping_geom_height_zero",
            "model_mapping_geom_width_range",
            "model_mapping_geom_height_range",
        ],
    },
    DeclineClass {
        type_name: "QueryError",
        defined_in: "runtime/heap_query.rs",
        slug_blocks: &[],
        // Representative, not exhaustive: `runtime/compute_exec/mod.rs` also emits
        // one, through the same builder.
        emission: Emission::At(&[("runtime/drain/mod.rs", "heap_texture_query")]),
        slug_calls: &[],
        slugs: &[
            "heap_query_short_payload",
            "heap_query_bad_reply_length",
            "heap_query_bad_serializer_length",
            "heap_query_unknown_serializer_tag",
            "heap_query_bad_descriptor_length",
            "heap_query_unknown_texture_type",
            "heap_query_unknown_pixel_format",
            "heap_query_unknown_usage",
            "heap_query_unknown_resource_options",
            "heap_query_unsupported_protection_options",
            "heap_query_no_metal_device",
            "heap_query_zero_requirement",
            "heap_query_bad_task",
        ],
    },
    DeclineClass {
        type_name: "TranslateReason",
        defined_in: "backend/vulkan/translate/reason.rs",
        slug_blocks: &[],
        // `translate/` is pure by design and logs nothing itself. This is the
        // caller that turns a returned reason into a line — the handoff the
        // registry exists to make checkable.
        emission: Emission::At(&[("runtime/metal_draw/vulkan.rs", "shader_state_degraded")]),
        slug_calls: &[],
        slugs: &[
            "unknown_pixel_format",
            "no_storage_image_format",
            "unknown_storage_selector",
            "srgb_downgraded",
            "no_sampled_layout",
            "no_color_attachment_format",
            "unknown_vertex_format",
            "unknown_vertex_step_function",
            "unknown_primitive_type",
            "unknown_blend_factor",
            "unknown_blend_operation",
            "unknown_compare_function",
            "unknown_stencil_operation",
            "unknown_cull_mode",
            "unknown_winding",
            "unknown_sampler_filter",
            "unknown_sampler_mip_filter",
            "unknown_sampler_address_mode",
            "unknown_sampler_border_color",
            "unknown_swizzle_selector",
            "format_not_vertex_buffer",
        ],
    },
    DeclineClass {
        type_name: "DrawReason",
        defined_in: "backend/vulkan/engine/reason.rs",
        slug_blocks: &[],
        // The sampler cache is where a capability decline is named at the check.
        // The rest of `DrawReason`'s surface still travels inside `DrawError` and
        // reaches the log only if a caller renders it — that is the remaining
        // half of this migration, not a claim of completeness.
        emission: Emission::At(&[("backend/vulkan/engine/caches.rs", "vk_engine_sampler")]),
        // `VertexFormat(TranslateReason)` is absent on purpose: it forwards to
        // the translation layer's slug rather than inventing a second name for
        // one event, and that slug is counted on `TranslateReason`'s row.
        slug_calls: &[],
        slugs: &[
            "multi_viewport_array",
            "resident_sampled_not_2d",
            "guest_run_sampled_not_2d",
            "secondary_attachment_cap",
            "depth_with_secondary_attachments",
            "sampler_anisotropy_unsupported",
            "sampler_mirror_clamp_to_edge_unsupported",
            "constant_vertex_attribute",
            "instance_rate_divisor_unsupported",
            "instance_rate_divisor_over_limit",
            "no_combined_graphics_compute_queue",
            "host_pointer_import_unavailable",
            "no_importable_host_memory_type",
            "no_host_visible_memory_for_staging",
            "no_host_visible_memory_for_readback",
            "no_host_visible_memory_for_stats",
            "no_device_local_memory_for_storage_image",
            "no_device_local_memory_for_slab",
            "no_device_local_memory_for_mrt_secondary",
            "no_device_local_memory_for_depth",
            "no_memory_type_for_scanout_export",
            "no_memory_type_for_dmabuf_import",
            "dmabuf_export_unavailable",
            "scanout_export_unavailable",
            "present_export_unavailable",
            "present_export_resident_not_bgra",
            "present_host_ptr_import_unavailable",
            "host_import_resolve",
            "runs_unstable",
            "present_scatter_resident_not_bgra",
            "swapchain_unavailable",
            "queue_cannot_present",
            "swapchain_lacks_transfer_dst",
            "swapchain_no_surface_format",
            "swapchain_no_composite_alpha",
        ],
    },
    // The three runtime present-adjacent rails. All wrote bare reason strings
    // that another rail also claimed: `unmapped` and `short_view` each had two
    // claimants (console capture vs guest-page import), and `no_mapping` three
    // (capture, import, and the type-5 view loader). Prefixing is what makes
    // `grep reason=unmapped` / `reason=no_mapping` answerable.
    DeclineClass {
        type_name: "CaptureDecline",
        defined_in: "runtime/scanout.rs",
        slug_blocks: &[],
        emission: Emission::At(&[("runtime/scanout.rs", "scanout_paint_mapping")]),
        slug_calls: &[],
        // 20 slugs from 16 former names: `short_view` was one `if` with three
        // `||`-ed conditions, and two more names each covered two sites whose
        // bounds differ.
        slugs: &[
            "capture_no_mapping",
            "capture_unmapped",
            "capture_no_pages",
            "capture_no_geom",
            "capture_geom_mismatch",
            "capture_bpp_unknown",
            "capture_tight_row_unknown",
            "capture_no_sample_window",
            "capture_bpr_below_tight",
            "capture_contig_view_null",
            "capture_contig_view_short",
            "capture_base_beyond_span",
            "capture_multi_read_failed",
            "capture_dst_overflow",
            "capture_convert_row_oob",
            "capture_convert_row_missing",
            "capture_convert_to_rgba",
            "capture_convert_from_rgba",
            "capture_direct_row_oob",
            "capture_direct_row_missing",
        ],
    },
    DeclineClass {
        type_name: "ImportDecline",
        defined_in: "runtime/import_present.rs",
        slug_blocks: &[],
        emission: Emission::At(&[("runtime/import_present.rs", "import_present")]),
        slug_calls: &[],
        slugs: &[
            "import_map_gen_drift",
            "import_unmapped",
            "import_no_sample_window",
            "import_bpr_below_tight",
            "import_short_view",
            "import_base_off_overflow",
            "import_revalidate",
            "import_short_table",
            "import_no_runs",
            "import_map_run_failed",
            "import_no_intersect",
        ],
    },
    DeclineClass {
        type_name: "TextureViewDecline",
        defined_in: "runtime/metal_draw/texture_view.rs",
        slug_blocks: &[],
        // Both the render/sample diagnostic and compute staging boundary carry
        // the exact chain or delegated descriptor-decode reason.
        emission: Emission::At(&[
            ("runtime/metal_draw/mod.rs", "sample_view_resolve"),
            (
                "runtime/compute_exec/mod.rs",
                "compute_stage_tex_view_resolve",
            ),
        ]),
        slug_calls: &[],
        // HopDecode delegates to resource::DecodeStatus, whose `res_*` slugs
        // are owned by that type's registry row rather than duplicated here.
        slugs: &[
            "texture_view_hop_entry_missing",
            "texture_view_hop_object_not_view",
            "texture_view_hop_descriptor_missing",
            "texture_view_hop_zero_base",
            "texture_view_hop_level_overflow",
            "texture_view_hop_swizzle_invalid",
            "texture_view_chain_self_or_zero",
            "texture_view_chain_overflow",
        ],
    },
    DeclineClass {
        // The type-5 serialized-view sampler. Its slugs carry a `type5_view_`
        // prefix rather than the blit rail's `t5_`, because `BlitStatus` already
        // owns a `t5_*` vocabulary for the type-5 *copy* path and four of these
        // checks are the same words (`no_mapping`, `sample_window`, format-bpp,
        // unmapped). The `fail` closure is the single emission funnel.
        type_name: "Type5ViewDecline",
        defined_in: "runtime/metal_draw/vulkan.rs",
        slug_blocks: &[],
        emission: Emission::At(&[("runtime/metal_draw/vulkan.rs", "type5_draw_view")]),
        slug_calls: &[],
        slugs: &[
            "type5_view_unsupported_depth",
            "type5_view_unresolved",
            "type5_view_format_bpp",
            "type5_view_no_mapping",
            "type5_view_sample_window",
            "type5_view_span",
            "type5_view_tight_overflow",
            "type5_view_native_len",
            "type5_view_read",
            "type5_view_rgba_stride",
            "type5_view_rgba_len",
            "type5_view_convert",
        ],
    },
    DeclineClass {
        type_name: "HostPresentDecline",
        defined_in: "backend/vulkan/engine/reason.rs",
        slug_blocks: &[],
        // The runtime's three import paths all classify on this type now, and
        // all three emit through `import_present`.
        emission: Emission::At(&[("runtime/import_present.rs", "import_present")]),
        slug_calls: &[],
        slugs: &[
            "read_target_unknown_identity",
            "read_target_no_ready_content",
            "host_ptr_unknown_identity",
            "host_ptr_no_ready_content",
            "host_ptr_bad_row_bytes",
            "host_ptr_short",
            "host_runs_empty",
            "host_runs_tight_row_overflow",
            "host_runs_bad_row_bytes",
            "host_runs_null_or_empty",
            "host_runs_len_exceeds_ptr",
            "host_runs_end_overflow",
            "host_runs_out_of_order",
            "host_runs_row_offset_overflow",
            "host_runs_sample_offset_overflow",
            "host_runs_row_end_overflow",
            "host_runs_scatter_oob",
            "host_runs_scatter_run_index_oob",
            "host_runs_scatter_buffer_index_oob",
            "host_runs_scatter_zero_texels",
            "host_runs_scatter_source_oob",
            "host_runs_scatter_span_end_overflow",
            "host_runs_scatter_buffer_offset_overflow",
            "host_runs_uncovered_row",
        ],
    },
    DeclineClass {
        type_name: "DrawError",
        defined_in: "backend/vulkan/engine/types.rs",
        slug_blocks: &[],
        // `import_present` classifies a failed present on the `DrawError` it
        // gets back. `linux_m2v_draw` is the Metal→Vulkan draw boundary in
        // `runtime/metal_draw/mod.rs`: `try_metal2vulkan_draw` returns `DrawError`
        // now (its pipeline/stage preparation failures and the engine's own
        // draw failures propagate as typed variants; the remaining staging
        // backlog is `Invalid`), so the boundary emits the specific check as
        // the primary `reason=`. Before this it flattened the error into a
        // free-text `err={e}` with no `reason=` at all.
        emission: Emission::At(&[
            ("backend/vulkan/engine/mod.rs", "present_capture"),
            ("backend/vulkan/engine/mod.rs", "vk_engine_probe"),
            ("backend/vulkan/engine/mod.rs", "vk_device_recreate"),
            ("runtime/import_present.rs", "import_present"),
            ("runtime/metal_draw/mod.rs", "linux_m2v_draw"),
            ("runtime/compute_exec/mod.rs", "compute_linux_engine"),
            (
                "backend/vulkan/engine/pools/host_import_and_teardown.rs",
                "host_import_fail",
            ),
            (
                "backend/vulkan/engine/exec_compute.rs",
                "compute_direct_writeback",
            ),
        ]),
        slug_calls: &[],
        // `Present` and `Unsupported` are absent on purpose: they forward to
        // `HostPresentDecline` and `DrawReason` rather than minting a second
        // name for one event, and those slugs are counted on their own rows.
        //
        // Every payload delegates to its typed decline. Fence timeout is the
        // only reason owned directly by `DrawError`.
        slugs: &["vk_engine_fence_timeout"],
    },
    DeclineClass {
        type_name: "DrawPreparationDecline",
        defined_in: "backend/vulkan/engine/draw_preparation.rs",
        slug_blocks: &[],
        // `try_metal2vulkan_draw` returns these through `DrawError`, whose
        // `Decline` implementation delegates the exact reason to this boundary.
        emission: Emission::At(&[
            ("runtime/metal_draw/mod.rs", "linux_m2v_draw"),
            ("runtime/compute_exec/mod.rs", "compute_linux_sampler"),
        ]),
        slug_calls: &[],
        slugs: &[
            "draw_prepare_pipeline_missing",
            "draw_prepare_vertex_mtlb_missing",
            "draw_prepare_fragment_mtlb_missing",
            "draw_prepare_geometry_unsupported",
            "draw_prepare_vertex_buffer_missing",
            "draw_prepare_fragment_buffer_missing",
            "draw_prepare_vertex_attribute_format",
            "draw_prepare_stage_in_bytes_missing",
            "draw_prepare_vertex_step_function_unsupported",
            "draw_prepare_color_input_mrt_unsupported",
            "draw_prepare_attachment_alias_identity_missing",
            "draw_prepare_attachment_alias_resident_not_ready",
            "draw_prepare_texture_resolve_missing",
            "draw_prepare_texture_dimension_unsupported",
            "draw_prepare_chain_resident_not_ready",
            "draw_prepare_chain_resident_identity_missing",
            "draw_prepare_sampler_entry_missing",
            "draw_prepare_sampler_object_type",
            "draw_prepare_sampler_descriptor_missing",
            "draw_prepare_sampler_descriptor_short",
            "draw_prepare_sampler_descriptor_bad_length",
            "draw_prepare_sampler_descriptor_unknown_type",
            "draw_prepare_sampler_descriptor_unsupported",
            "draw_prepare_sampler_min_filter_translation",
            "draw_prepare_sampler_mag_filter_translation",
            "draw_prepare_sampler_mip_filter_translation",
            "draw_prepare_sampler_address_s_translation",
            "draw_prepare_sampler_address_t_translation",
            "draw_prepare_sampler_address_r_translation",
            "draw_prepare_sampler_border_color_translation",
            "draw_prepare_sampler_compare_function_translation",
            "draw_prepare_static_sampler_reduction_unsupported",
            "draw_prepare_static_sampler_lod_bias_unsupported",
            "draw_prepare_static_sampler_min_filter_unsupported",
            "draw_prepare_static_sampler_mag_filter_unsupported",
            "draw_prepare_static_sampler_reflection_descriptor_missing",
            "draw_prepare_static_sampler_reflection_state_missing",
        ],
    },
    DeclineClass {
        type_name: "InitDecline",
        defined_in: "backend/vulkan/engine/init_decline.rs",
        slug_blocks: &[],
        // Most initialization failures propagate through `DrawError` to its
        // existing draw/import/window boundaries. Device selection also emits
        // immediately because no later GPU command may exist when discovery
        // itself fails; this typed site mechanically proves the vocabulary
        // reaches the always-on sink.
        emission: Emission::At(&[
            ("backend/vulkan/engine/context.rs", "vk_device_select_fail"),
            ("backend/vulkan/engine/context.rs", "vk_loader_version"),
        ]),
        slug_calls: &[],
        slugs: &[
            "vk_init_load_loader",
            "vk_init_enumerate_instance_version",
            "vk_init_enumerate_instance_extensions",
            "vk_init_create_instance",
            "vk_init_enumerate_physical_devices",
            "vk_init_no_physical_device",
            "vk_init_below_api_floor",
            "vk_init_no_graphics_queue_family",
            "vk_init_enumerate_device_extensions",
            "vk_init_create_device",
            "vk_init_create_pipeline_cache",
        ],
    },
    DeclineClass {
        type_name: "DeviceLostDecline",
        defined_in: "backend/vulkan/engine/device_lost.rs",
        slug_blocks: &[],
        // Device loss propagates inside `DrawError`; an open draw batch can
        // also surface it through this direct typed boundary.
        emission: Emission::At(&[("backend/vulkan/engine/mod.rs", "vk_batch_flush")]),
        slug_calls: &[],
        slugs: &[
            "vk_device_lost_recreate_cap_exhausted",
            "vk_device_lost_recreate_failed",
            "vk_device_lost_forced_draw",
            "vk_device_lost_forced_compute",
            "vk_device_lost_exec_submit",
            "vk_device_lost_compute_exec_submit",
            "vk_device_lost_pools_wait_fences_retire",
            "vk_device_lost_pools_fence_status_begin_entry",
            "vk_device_lost_pools_wait_fences_entry",
            "vk_device_lost_pools_submit_batch",
        ],
    },
    DeclineClass {
        type_name: "M2vCacheDecline",
        defined_in: "runtime/m2v_cache.rs",
        slug_blocks: &[],
        // Async translation emits directly. Render delegates through
        // DrawPreparationDecline/DrawError to linux_m2v_draw; compute emits the
        // cache decline before preserving its existing recovery status.
        emission: Emission::At(&[
            ("runtime/m2v_cache.rs", "linux_m2v_async_done"),
            ("runtime/metal_draw/mod.rs", "linux_m2v_draw"),
            ("runtime/compute_exec/mod.rs", "compute_linux_m2v"),
        ]),
        slug_calls: &[],
        slugs: &[
            "m2v_vertex_scratch_write",
            "m2v_fragment_scratch_write",
            "m2v_kernel_scratch_write",
            "m2v_vertex_translate",
            "m2v_fragment_translate",
            "m2v_kernel_translate",
            "m2v_reflection_datalayout_missing",
            "m2v_translation_pending_at_sync_boundary",
            "m2v_kernel_local_size_zero",
        ],
    },
    DeclineClass {
        type_name: "SpirvLayoutDecline",
        defined_in: "runtime/spirv_layout.rs",
        slug_blocks: &[],
        // The layout repair is cached as an M2vCacheDecline, which delegates
        // this exact reason through render, compute, and async cache boundaries.
        emission: Emission::At(&[
            ("runtime/m2v_cache.rs", "linux_m2v_async_done"),
            ("runtime/metal_draw/mod.rs", "linux_m2v_draw"),
            ("runtime/compute_exec/mod.rs", "compute_linux_m2v"),
        ]),
        slug_calls: &[],
        slugs: &[
            "spirv_layout_datalayout_vector_alignment_missing",
            "spirv_layout_length_misaligned",
            "spirv_layout_header_invalid",
            "spirv_layout_vector_width_overflow",
            "spirv_layout_type_vector_alignment_missing",
            "spirv_layout_allocation_round_up_overflow",
            "spirv_layout_instruction_malformed",
            "spirv_layout_duplicate_member_offset",
            "spirv_layout_initial_member_offset_overflow",
            "spirv_layout_following_member_offset_overflow",
        ],
    },
    DeclineClass {
        type_name: "MtlbDecline",
        defined_in: "runtime/mtlb.rs",
        slug_blocks: &[],
        // Render delegates through DrawPreparationDecline/DrawError; compute
        // emits the extractor refusal before keeping its recovery class.
        emission: Emission::At(&[
            ("runtime/metal_draw/mod.rs", "linux_m2v_draw"),
            ("runtime/compute_exec/mod.rs", "compute_linux_air_extract"),
        ]),
        slug_calls: &[],
        slugs: &[
            "mtlb_wrapped_air_missing",
            "mtlb_wrapper_header_truncated",
            "mtlb_blob_out_of_bounds",
        ],
    },
    DeclineClass {
        type_name: "VertexEvalDecline",
        defined_in: "runtime/spirv_vertex_eval.rs",
        slug_blocks: &[],
        // The evaluator is a fail-closed coverage proof. Its caller delegates
        // the exact check to the always-on coverage-gap notice.
        emission: Emission::At(&[("runtime/metal_draw/vulkan.rs", "linear_coverage_gap")]),
        slug_calls: &[],
        slugs: &[
            "spirv_vertex_eval_access_chain_array_offset_overflow",
            "spirv_vertex_eval_access_chain_array_stride_missing",
            "spirv_vertex_eval_access_chain_base_not_pointer",
            "spirv_vertex_eval_access_chain_base_unknown",
            "spirv_vertex_eval_access_chain_struct_index_out_of_range",
            "spirv_vertex_eval_access_chain_struct_member_offset_missing",
            "spirv_vertex_eval_access_chain_struct_offset_overflow",
            "spirv_vertex_eval_access_chain_type_unsupported",
            "spirv_vertex_eval_access_chain_vector_index_out_of_range",
            "spirv_vertex_eval_access_chain_vector_offset_overflow",
            "spirv_vertex_eval_array_variable_unsupported",
            "spirv_vertex_eval_binary_float_operand_shape_mismatch",
            "spirv_vertex_eval_bitcast_element_type_unsupported",
            "spirv_vertex_eval_bitcast_type_unsupported",
            "spirv_vertex_eval_boolean_binary_opcode_unsupported",
            "spirv_vertex_eval_boolean_binary_shape_mismatch",
            "spirv_vertex_eval_branch_condition_not_bool",
            "spirv_vertex_eval_branch_label_unknown",
            "spirv_vertex_eval_buffer_binding_missing",
            "spirv_vertex_eval_buffer_load_type_unsupported",
            "spirv_vertex_eval_buffer_offset_does_not_fit_usize",
            "spirv_vertex_eval_buffer_range_overflow",
            "spirv_vertex_eval_buffer_read_out_of_bounds",
            "spirv_vertex_eval_buffer_size_does_not_fit_usize",
            "spirv_vertex_eval_buffer_store_unsupported",
            "spirv_vertex_eval_buffer_variable_binding_missing",
            "spirv_vertex_eval_composite_constant_forward_reference",
            "spirv_vertex_eval_composite_extract_from_scalar",
            "spirv_vertex_eval_composite_extract_index_out_of_range",
            "spirv_vertex_eval_composite_insert_index_out_of_range",
            "spirv_vertex_eval_composite_insert_into_scalar",
            "spirv_vertex_eval_constant_type_missing",
            "spirv_vertex_eval_constant_type_unsupported",
            "spirv_vertex_eval_dot_shape_mismatch",
            "spirv_vertex_eval_entry_function_body_missing_during_parse",
            "spirv_vertex_eval_entry_function_body_missing_during_run",
            "spirv_vertex_eval_ext_argument_missing",
            "spirv_vertex_eval_ext_inst_set_unknown",
            "spirv_vertex_eval_extended_opcode_unsupported",
            "spirv_vertex_eval_float_buffer_read_width_mismatch",
            "spirv_vertex_eval_float_compare_opcode_unsupported",
            "spirv_vertex_eval_float_compare_shape_mismatch",
            "spirv_vertex_eval_float_vector_expected",
            "spirv_vertex_eval_float_vector_member_type_mismatch",
            "spirv_vertex_eval_function_fell_off_end",
            "spirv_vertex_eval_function_instruction_malformed",
            "spirv_vertex_eval_function_variable_storage_class_invalid",
            "spirv_vertex_eval_function_variable_type_not_pointer",
            "spirv_vertex_eval_global_variable_type_missing",
            "spirv_vertex_eval_global_variable_type_not_pointer",
            "spirv_vertex_eval_input_variable_unsupported",
            "spirv_vertex_eval_int_operand_type_mismatch",
            "spirv_vertex_eval_int_result_type_mismatch",
            "spirv_vertex_eval_int_vector_element_type_mismatch",
            "spirv_vertex_eval_integer_binary_opcode_unsupported",
            "spirv_vertex_eval_integer_compare_opcode_unsupported",
            "spirv_vertex_eval_integer_compare_shape_mismatch",
            "spirv_vertex_eval_integer_operand_shape_mismatch",
            "spirv_vertex_eval_load_pointer_unknown",
            "spirv_vertex_eval_load_source_not_pointer",
            "spirv_vertex_eval_logical_not_operand_not_bool",
            "spirv_vertex_eval_logical_not_vector_member_not_bool",
            "spirv_vertex_eval_main_instruction_budget_exhausted",
            "spirv_vertex_eval_malformed_header",
            "spirv_vertex_eval_map_float_to_int_operand_type_mismatch",
            "spirv_vertex_eval_map_int_operand_type_mismatch",
            "spirv_vertex_eval_map_int_to_float_operand_type_mismatch",
            "spirv_vertex_eval_matrix_scalar_type_mismatch",
            "spirv_vertex_eval_matrix_times_matrix_empty_matrix",
            "spirv_vertex_eval_matrix_times_matrix_left_not_composite",
            "spirv_vertex_eval_matrix_times_matrix_right_not_composite",
            "spirv_vertex_eval_matrix_times_matrix_shape_mismatch",
            "spirv_vertex_eval_matrix_times_vector_column_count_mismatch",
            "spirv_vertex_eval_matrix_times_vector_column_height_mismatch",
            "spirv_vertex_eval_matrix_times_vector_empty_matrix",
            "spirv_vertex_eval_matrix_times_vector_matrix_not_composite",
            "spirv_vertex_eval_memory_load_path_into_scalar",
            "spirv_vertex_eval_memory_load_path_out_of_range",
            "spirv_vertex_eval_memory_load_variable_unset",
            "spirv_vertex_eval_memory_store_path_into_scalar",
            "spirv_vertex_eval_memory_store_path_out_of_range",
            "spirv_vertex_eval_memory_store_variable_unset",
            "spirv_vertex_eval_module_instruction_malformed",
            "spirv_vertex_eval_null_type_unsupported",
            "spirv_vertex_eval_opcode_unsupported",
            "spirv_vertex_eval_phi_instruction_budget_exhausted",
            "spirv_vertex_eval_phi_instruction_malformed",
            "spirv_vertex_eval_phi_outside_block_entry",
            "spirv_vertex_eval_phi_predecessor_missing",
            "spirv_vertex_eval_position_component_not_finite",
            "spirv_vertex_eval_position_component_undefined",
            "spirv_vertex_eval_position_member_never_stored",
            "spirv_vertex_eval_position_output_missing",
            "spirv_vertex_eval_position_struct_never_stored",
            "spirv_vertex_eval_position_value_not_composite",
            "spirv_vertex_eval_position_variable_never_stored",
            "spirv_vertex_eval_position_vector_length_invalid",
            "spirv_vertex_eval_scalar_size_type_unsupported",
            "spirv_vertex_eval_select_condition_not_bool",
            "spirv_vertex_eval_select_values_not_composite",
            "spirv_vertex_eval_select_vector_condition_not_bool",
            "spirv_vertex_eval_select_vector_length_mismatch",
            "spirv_vertex_eval_signed_compare_width_unknown",
            "spirv_vertex_eval_signed_convert_source_width_unknown",
            "spirv_vertex_eval_signed_division_by_zero",
            "spirv_vertex_eval_signed_remainder_by_zero",
            "spirv_vertex_eval_signed_to_float_source_width_unknown",
            "spirv_vertex_eval_storage_class_unsupported",
            "spirv_vertex_eval_store_pointer_unknown",
            "spirv_vertex_eval_store_target_not_pointer",
            "spirv_vertex_eval_switch_operands_malformed",
            "spirv_vertex_eval_switch_selector_width_unknown",
            "spirv_vertex_eval_ternary_float_operand_shape_mismatch",
            "spirv_vertex_eval_transpose_matrix_not_composite",
            "spirv_vertex_eval_transpose_matrix_ragged",
            "spirv_vertex_eval_type_missing",
            "spirv_vertex_eval_unary_float_operand_type_mismatch",
            "spirv_vertex_eval_unexpected_terminator",
            "spirv_vertex_eval_unpack_unorm_operand_not_int",
            "spirv_vertex_eval_unsigned_division_by_zero",
            "spirv_vertex_eval_unsigned_modulo_by_zero",
            "spirv_vertex_eval_value_id_unset",
            "spirv_vertex_eval_vector_scalar_type_mismatch",
            "spirv_vertex_eval_vector_shuffle_index_out_of_range",
            "spirv_vertex_eval_vector_shuffle_left_not_vector",
            "spirv_vertex_eval_vector_shuffle_right_not_vector",
            "spirv_vertex_eval_vector_times_matrix_matrix_not_composite",
            "spirv_vertex_eval_vector_times_matrix_shape_mismatch",
            "spirv_vertex_eval_vertex_entry_point_missing",
        ],
    },
    DeclineClass {
        type_name: "ShaderPulledCoverageDecline",
        defined_in: "runtime/metal_draw/vulkan.rs",
        slug_blocks: &[],
        emission: Emission::At(&[("runtime/metal_draw/vulkan.rs", "linear_coverage_gap")]),
        slug_calls: &[],
        slugs: &[
            "shader_pulled_coverage_zero_target",
            "shader_pulled_coverage_partial_viewport_or_scissor",
            "shader_pulled_coverage_index_stream_invalid",
            "shader_pulled_coverage_too_few_indices",
            "shader_pulled_coverage_position_w_degenerate",
            "shader_pulled_coverage_position_not_finite",
            "shader_pulled_coverage_triangle_gap",
            "shader_pulled_coverage_partial_bounds",
        ],
    },
    DeclineClass {
        type_name: "DrawValidationDecline",
        defined_in: "backend/vulkan/engine/draw_validation.rs",
        slug_blocks: &[],
        // The validator returns through `DrawError`; the product draw boundary
        // emits that delegated reason before any GPU work begins.
        emission: Emission::At(&[("runtime/metal_draw/mod.rs", "linux_m2v_draw")]),
        slug_calls: &[],
        slugs: &[
            "vk_draw_validate_vertex_guest_runs_row_stride",
            "vk_draw_validate_storage_guest_runs_row_stride",
            "vk_draw_validate_vertex_guest_runs_coverage",
            "vk_draw_validate_storage_guest_runs_coverage",
            "vk_draw_validate_zero_target_geometry",
            "vk_draw_validate_empty_vertex_spirv",
            "vk_draw_validate_empty_fragment_spirv",
            "vk_draw_validate_non_finite_viewport",
            "vk_draw_validate_non_positive_viewport",
            "vk_draw_validate_non_finite_blend_constants",
            "vk_draw_validate_target_seed_length",
            "vk_draw_validate_load_seed_length",
            "vk_draw_validate_seed_missing_target_identity",
            "vk_draw_validate_seed_conflicts_cpu_seed",
            "vk_draw_validate_seed_conflicts_load_from_target",
            "vk_draw_validate_seed_equals_target",
            "vk_draw_validate_seed_also_sampled",
            "vk_draw_validate_index_bytes_short",
            "vk_draw_validate_indexed_vertex_range",
            "vk_draw_validate_duplicate_vertex_location",
            "vk_draw_validate_duplicate_vertex_binding",
            "vk_draw_validate_zero_vertex_step_rate",
            "vk_draw_validate_vertex_stride_too_small",
            "vk_draw_validate_vertex_offset_overflow",
            "vk_draw_validate_vertex_element_exceeds_stride",
            "vk_draw_validate_vertex_range_overflow",
            "vk_draw_validate_instance_range_overflow",
            "vk_draw_validate_vertex_byte_range_overflow",
            "vk_draw_validate_vertex_data_short",
            "vk_draw_validate_constant_step_guest_runs",
            "vk_draw_validate_duplicate_storage_descriptor_binding",
            "vk_draw_validate_duplicate_sampled_descriptor_binding",
            "vk_draw_validate_duplicate_sampler_descriptor_binding",
            "vk_draw_validate_sampled_zero_geometry",
            "vk_draw_validate_sampled_shape_conflict",
            "vk_draw_validate_sampled_cube_geometry",
            "vk_draw_validate_sampled_nonarray_layers",
            "vk_draw_validate_sampled_no_linear_texel_footprint",
            "vk_draw_validate_sampled_bytes_length",
            "vk_draw_validate_resident_sample_geometry",
            "vk_draw_validate_guest_sample_row_stride",
            "vk_draw_validate_guest_sample_length",
            "vk_draw_validate_guest_sample_coverage",
            "vk_draw_validate_invalid_sampler_lod",
        ],
    },
    DeclineClass {
        type_name: "DrawExecutionDecline",
        defined_in: "backend/vulkan/engine/draw_execution.rs",
        slug_blocks: &[],
        // The product draw boundary emits the delegated `DrawError` reason.
        emission: Emission::At(&[("runtime/metal_draw/mod.rs", "linux_m2v_draw")]),
        slug_calls: &[],
        slugs: &[
            "vk_draw_exec_buffer_guest_run_import_missing",
            "vk_draw_exec_constant_vertex_requires_cpu_bytes",
            "vk_draw_exec_constant_vertex_base_instance_overflow",
            "vk_draw_exec_constant_vertex_allocation_overflow",
            "vk_draw_exec_load_target_content_not_ready",
            "vk_draw_exec_seed_resident_missing",
            "vk_draw_exec_seed_resident_not_ready",
            "vk_draw_exec_seed_geometry_mismatch",
            "vk_draw_exec_seed_format_mismatch",
            "vk_draw_exec_sampled_resident_missing",
            "vk_draw_exec_sampled_resident_not_ready",
            "vk_draw_exec_sampled_resident_geometry_mismatch",
            "vk_draw_exec_sampled_guest_run_import_missing",
            "vk_draw_exec_unsupported_tracked_layout",
        ],
    },
    DeclineClass {
        type_name: "ComputeValidationDecline",
        defined_in: "backend/vulkan/engine/compute_validation.rs",
        slug_blocks: &[],
        // The Vulkan compute boundary emits the delegated `DrawError` reason
        // after a failed engine call and before mapping it to `ComputeStatus`.
        emission: Emission::At(&[("runtime/compute_exec/mod.rs", "compute_linux_engine")]),
        slug_calls: &[],
        slugs: &[
            "vk_compute_validate_empty_spirv",
            "vk_compute_validate_empty_entry",
            "vk_compute_validate_entry_interior_nul",
            "vk_compute_validate_zero_grid",
            "vk_compute_validate_duplicate_storage_buffer_binding",
            "vk_compute_validate_empty_storage_buffer",
            "vk_compute_validate_duplicate_sampled_image_binding",
            "vk_compute_validate_sampled_zero_geometry",
            "vk_compute_validate_sampled_1d_height",
            "vk_compute_validate_sampled_nonarray_layers",
            "vk_compute_validate_sampled_bytes_length",
            "vk_compute_validate_invalid_sampler_lod",
            "vk_compute_validate_duplicate_sampler_binding",
            "vk_compute_validate_duplicate_storage_image_binding",
            "vk_compute_validate_storage_zero_geometry",
            "vk_compute_validate_storage_1d_height",
            "vk_compute_validate_storage_nonarray_layers",
            "vk_compute_validate_storage_bytes_length",
        ],
    },
    DeclineClass {
        type_name: "ComputeExecutionDecline",
        defined_in: "backend/vulkan/engine/compute_execution.rs",
        slug_blocks: &[],
        // The Vulkan compute boundary emits the delegated `DrawError` reason;
        // the direct-writeback rail emits its exact structural degradation
        // before taking the correct-but-slower readback fallback.
        emission: Emission::At(&[
            (
                "backend/vulkan/engine/exec_compute.rs",
                "compute_direct_writeback",
            ),
            ("runtime/compute_exec/mod.rs", "compute_linux_engine"),
        ]),
        slug_calls: &[],
        slugs: &[
            "vk_compute_exec_resident_sample_absent",
            "vk_compute_exec_resident_sample_generation_mismatch",
            "vk_compute_exec_resident_sample_byte_shape_mismatch",
            "vk_compute_exec_resident_sample_source_layers_unsupported",
            "vk_compute_exec_resident_sample_resource_layers_unsupported",
            "vk_compute_exec_resident_sample_1d_unsupported",
            "vk_compute_exec_resident_sample_arrayed_unsupported",
            "vk_compute_exec_resident_sample_volume_unsupported",
            "vk_compute_exec_seed_skipped_without_residency",
            "vk_compute_exec_resident_seed_generation_lost",
            "vk_compute_exec_resident_allocator_live_slot_missing",
            "vk_compute_exec_direct_writeback_capability_lost",
            "vk_compute_exec_direct_writeback_shape_mismatch",
            "vk_compute_exec_direct_writeback_row_bytes_not_texel_aligned",
            "vk_compute_exec_direct_writeback_row_bytes_too_short",
            "vk_compute_exec_direct_writeback_buffer_offset_not_texel_aligned",
            "vk_compute_exec_direct_writeback_buffer_offset_not_four_aligned",
            "vk_compute_exec_direct_writeback_null_pointer",
            "vk_compute_exec_direct_writeback_pointer_misaligned",
            "vk_compute_exec_direct_writeback_row_start_overflow",
            "vk_compute_exec_direct_writeback_required_span_overflow",
            "vk_compute_exec_direct_writeback_import_size_overflow",
            "vk_compute_exec_direct_writeback_window_too_short",
        ],
    },
    DeclineClass {
        type_name: "EngineFacadeDecline",
        defined_in: "backend/vulkan/engine/facade_decline.rs",
        slug_blocks: &[],
        // Each façade entry point propagates through the boundary that owns its
        // fallback or host-window event, with DrawError delegating this slug.
        emission: Emission::At(&[
            ("host_window/present.rs", "host_window_present"),
            ("runtime/storage_flush.rs", "deferred_flush_lost"),
            ("runtime/drain/mod.rs", "scanout_gl_export_fail"),
            ("lib.rs", "export_present"),
        ]),
        slug_calls: &[],
        slugs: &[
            "vk_engine_window_presenter_not_attached",
            "vk_engine_storage_read_resident_absent",
            "vk_engine_storage_read_generation_mismatch",
            "vk_engine_export_scanout_zero_geometry",
            "vk_engine_export_scanout_length_mismatch",
            "vk_engine_export_present_unknown_identity",
            "vk_engine_export_present_not_ready",
            "vk_engine_scatter_present_unknown_identity",
            "vk_engine_scatter_present_not_ready",
            "vk_engine_window_source_disappeared_before_pin",
            "vk_engine_window_peer_disappeared_before_pin",
        ],
    },
    DeclineClass {
        // The typed target of `DrawError::Vulkan(String)`. Each engine file's
        // free-text Vulkan-call sites move to `DrawError::VkCall(VkCall)`, which
        // delegates its slug here. The transcription grew this vocabulary file
        // by file until the coarse `Vulkan(String)` variant could be deleted.
        // First the stats-reduction pool's eight calls, then
        // `mod.rs`'s two import_present-reaching rails (readback + host present),
        // its storage-flush rail (`read_resident_storage`), and its two dmabuf
        // export rails (`export_scanout_from_bgra`, `export_present_*`).
        type_name: "VkCall",
        defined_in: "backend/vulkan/engine/vk_call.rs",
        slug_blocks: &[],
        // The stats pool used to swallow these with `.is_err()`/`.ok()`, so a
        // reduction that could not build its buffers went blind with no line;
        // `arm`/`pick_slot` now emit the typed decline at those swallow points.
        // The `mod.rs` readback (`read_target`/`read_target_inner`) and packed-
        // contig host-present (`present_into_host_ptr_strided`) rails already
        // reach the sink through `runtime/import_present.rs`'s
        // `Emit::decline("import_present", &e)` — `DrawError` delegates its slug
        // and fields here, so those calls render `reason=vk_...` there for free.
        // The storage-flush rail (`read_resident_storage`) reaches
        // `runtime/storage_flush.rs`'s `deferred_flush_lost` line, which the same
        // migration converted from a bare-string `err={e}` to `Emit::decline`.
        // The two dmabuf export rails reach `runtime/drain/mod.rs`'s
        // `scanout_gl_export_fail` (CPU-capture scanout) and `lib.rs`'s
        // `export_present` (zero-copy present) — the latter a silent
        // `Err(_) => None` the same migration closed. `dmabuf_export.rs`'s own
        // `export_bgra_scanout_dmabuf` (the low-level exportable VkImage those
        // two rails are built on: create/alloc/bind/get_fd) propagates up
        // through `ScanoutExportCache`/`ScanoutExportRing` to the very same two
        // sinks, so its four `vk_dmabuf_export_*` slugs surface there too.
        // Its consumer half, `import_bgra_dmabuf_image` (the host window
        // importing that dmabuf fd as a sampleable image: create/fd-props/
        // alloc/bind), is now `vk_dmabuf_import_*`. Its sink is
        // `host_window/present.rs`'s `direct_present_degrade
        // reason=import_failed` census (an `observe::off` count, not an
        // `Emit::decline`), which used to render the error with `{e:?}` — Debug
        // prints the variant name, dropping the slug. `import_ring_slot` now
        // renders `op=<slug>` plus the decline's fields, so the specific failing
        // call is greppable (`op=vk_dmabuf_import_*`) under the census's
        // `reason=import_failed` bucket — the same fix also recovered the
        // `NoMemoryTypeForDmabufImport` refusal's slug, which that `{e:?}` had
        // dropped too.
        //
        // `caches.rs`'s seven object-cache creates (`get_or_create_shader` …
        // `get_or_create_compute_pipeline`) cache their failure *negatively* as
        // a `DrawError`, so both the create and the cheap re-attempt replay the
        // same typed reason. `context.rs` (guest host-pointer import),
        // `desc_arena.rs` (descriptor-set arena), `exec.rs` (the draw
        // command-buffer record/submit/readback rail) and `exec_compute.rs`
        // (the compute one — a distinct queue submission) add the rest of the
        // engine's fallible ash calls. They flow up the engine's
        // draw/present/compute rails; on the draw path they reach the
        // `metal_draw.rs` boundary (`linux_m2v_draw`), which now propagates the
        // engine's `DrawError` unchanged and emits `Emit::decline`, so a VkCall
        // slug delegated through `DrawError::VkCall` renders there as the primary
        // `reason=vk_...`. `exec.rs`'s `vk_exec_map_readback` and the `mod.rs`
        // readback rail also reach `import_present` directly.
        // `host_scatter.rs` adds the eight setup/record/submit/wait calls on the
        // GPU-direct Store rail. They propagate through `present_into_host_runs`
        // to `runtime/import_present.rs` without a boolean/coarse wrapper.
        // `pools/mod.rs`'s resource pools contribute the command-buffer/fence
        // machinery, batch submit, and staging + readback buffer rails
        // (`vk_pools_*`); its `wait_error` fence helper now takes a `VkOp` so a
        // timeout still maps to `FenceTimeout` but a real wait/status failure
        // names its call. Part B adds its target / sampled / storage / registry
        // / MRT-secondary / depth image + view + framebuffer allocation rails,
        // so the whole `pools/mod.rs` file is typed.
        //
        // `window_present.rs`'s macOS host-window MoltenVK swapchain presenter
        // (`vk_window_*`) is the last engine file: surface + swapchain bring-up
        // and the per-present acquire/blit/submit/present. Its capability-
        // selection refusals were already `DrawReason`, so these are its raw
        // `vk::Result` calls. Its `DrawError` reaches the sink two ways: the
        // present path (`window_present_frame`) through `host_window/present.rs`'s
        // `host_window_present` line, converted here from a bare-string
        // `reason=engine_resident_present error={e}` double-reason to
        // `Emit::decline`; and the attach path (`window_present_attach`) through
        // `WindowError::AttachEngine`, where the slug rides the Display flatten
        // (`vk_engine_vk: reason=vk_window_… vk_result=…`) into that type's own
        // `host_window_init` line.
        emission: Emission::At(&[
            ("backend/vulkan/engine/stats_reduce.rs", "stats_reduce"),
            ("backend/vulkan/engine/context.rs", "vk_pipeline_cache_save"),
            ("backend/vulkan/engine/desc_arena.rs", "desc_arena_free"),
            (
                "backend/vulkan/engine/pools/host_import_and_teardown.rs",
                "vk_pools_destroy",
            ),
            ("backend/vulkan/engine/mod.rs", "vulkan_guest_reset"),
            (
                "backend/vulkan/engine/window_present.rs",
                "host_window_destroy",
            ),
            ("runtime/import_present.rs", "import_present"),
            ("runtime/storage_flush.rs", "deferred_flush_lost"),
            ("runtime/drain/mod.rs", "scanout_gl_export_fail"),
            ("lib.rs", "export_present"),
            ("host_window/present.rs", "host_window_present"),
            ("runtime/metal_draw/mod.rs", "linux_m2v_draw"),
            (
                "backend/vulkan/engine/pools/host_import_and_teardown.rs",
                "host_import_fail",
            ),
            (
                "backend/vulkan/engine/exec_compute.rs",
                "compute_direct_writeback",
            ),
        ]),
        slug_calls: &[],
        slugs: &[
            "vk_stats_desc_pool",
            "vk_stats_sampler",
            "vk_stats_create_buffer",
            "vk_stats_alloc",
            "vk_stats_bind",
            "vk_stats_map",
            "vk_stats_alloc_cb",
            "vk_stats_create_fence",
            "vk_stats_fence_status_reclaim",
            "vk_stats_alloc_descriptor_set",
            "vk_stats_reset_fence",
            "vk_stats_reset_command_buffer",
            "vk_stats_begin_command_buffer",
            "vk_stats_end_command_buffer",
            "vk_stats_queue_submit",
            "vk_stats_fence_status_consume",
            "vk_stats_wait_fence_blocking",
            "vk_stats_wait_fence_destroy",
            "vk_readback_reset_cb",
            "vk_readback_begin_cb",
            "vk_readback_end_cb",
            "vk_readback_submit",
            "vk_readback_map",
            "vk_host_present_reset_cb",
            "vk_host_present_begin_cb",
            "vk_host_present_end_cb",
            "vk_host_present_submit",
            "vk_storage_read_reset_cb",
            "vk_storage_read_begin_cb",
            "vk_storage_read_end_cb",
            "vk_storage_read_submit",
            "vk_storage_read_map",
            "vk_export_scanout_map_staging",
            "vk_export_scanout_reset_cb",
            "vk_export_scanout_begin_cb",
            "vk_export_scanout_end_cb",
            "vk_export_scanout_submit",
            "vk_export_present_reset_cb",
            "vk_export_present_begin_cb",
            "vk_export_present_end_cb",
            "vk_export_present_submit",
            "vk_dmabuf_export_create_image",
            "vk_dmabuf_export_alloc",
            "vk_dmabuf_export_bind",
            "vk_dmabuf_export_get_fd",
            "vk_dmabuf_import_create_image",
            "vk_dmabuf_import_fd_props",
            "vk_dmabuf_import_alloc",
            "vk_dmabuf_import_bind",
            "vk_caches_create_shader_module",
            "vk_caches_create_descriptor_set_layout",
            "vk_caches_create_pipeline_layout",
            "vk_caches_create_render_pass",
            "vk_caches_create_sampler",
            "vk_caches_create_graphics_pipelines",
            "vk_caches_create_compute_pipelines",
            "vk_context_host_ptr_props",
            "vk_context_import_host_ptr_alloc",
            "vk_context_pipeline_cache_get_data",
            "vk_desc_arena_create_pool",
            "vk_desc_arena_alloc_sets",
            "vk_desc_arena_alloc_sets_grown",
            "vk_desc_arena_free_sets",
            "vk_exec_reset_cb",
            "vk_exec_begin_cb",
            "vk_exec_end_cb",
            "vk_exec_submit",
            "vk_exec_map_readback",
            "vk_compute_exec_reset_cb",
            "vk_compute_exec_begin_cb",
            "vk_compute_exec_end_cb",
            "vk_compute_exec_submit",
            "vk_compute_exec_map_storage_readback",
            "vk_compute_exec_map_image_readback",
            "vk_compute_direct_writeback_create_buffer",
            "vk_compute_direct_writeback_bind_buffer",
            "vk_host_scatter_alloc_command_buffer",
            "vk_host_scatter_create_fence",
            "vk_host_scatter_reset_fence",
            "vk_host_scatter_reset_command_buffer",
            "vk_host_scatter_begin_command_buffer",
            "vk_host_scatter_end_command_buffer",
            "vk_host_scatter_queue_submit",
            "vk_host_scatter_wait_fence",
            "vk_guest_reset_device_wait_idle",
            "vk_slab_allocate_memory",
            "vk_pools_create_command_pool",
            "vk_pools_alloc_command_buffers",
            "vk_pools_create_fence",
            "vk_pools_wait_fences_retire",
            "vk_pools_fence_status_begin_entry",
            "vk_pools_wait_fences_entry",
            "vk_pools_wait_fences_destroy",
            "vk_pools_reset_fences_retire",
            "vk_pools_end_cb_batch",
            "vk_pools_submit_batch",
            "vk_pools_host_import_create_buffer",
            "vk_pools_host_import_bind_buffer",
            "vk_pools_create_staging",
            "vk_pools_alloc_staging",
            "vk_pools_bind_staging",
            "vk_pools_map_staging",
            "vk_pools_create_readback",
            "vk_pools_alloc_readback",
            "vk_pools_bind_readback",
            "vk_pools_create_readback_extra",
            "vk_pools_alloc_readback_extra",
            "vk_pools_bind_readback_extra",
            "vk_pools_create_target_image",
            "vk_pools_bind_target",
            "vk_pools_create_target_view",
            "vk_pools_create_framebuffer",
            "vk_pools_create_sampled_image",
            "vk_pools_bind_sampled",
            "vk_pools_create_sampled_view",
            "vk_pools_create_storage_image",
            "vk_pools_alloc_storage_image",
            "vk_pools_bind_storage_image",
            "vk_pools_create_storage_image_view",
            "vk_pools_create_registry_framebuffer",
            "vk_pools_create_registry_target",
            "vk_pools_bind_registry_target",
            "vk_pools_create_registry_view",
            "vk_pools_create_mrt_secondary_target",
            "vk_pools_alloc_mrt_secondary",
            "vk_pools_bind_mrt_secondary",
            "vk_pools_create_mrt_secondary_view",
            "vk_pools_create_depth_image",
            "vk_pools_alloc_depth",
            "vk_pools_bind_depth",
            "vk_pools_create_depth_view",
            "vk_pools_create_mrt_framebuffer",
            "vk_window_create_surface",
            "vk_window_surface_support",
            "vk_window_create_command_pool",
            "vk_window_alloc_command_buffer",
            "vk_window_create_acquire_semaphore",
            "vk_window_create_render_semaphore",
            "vk_window_create_fence",
            "vk_window_fence_status",
            "vk_window_queue_wait_idle",
            "vk_window_surface_caps",
            "vk_window_surface_formats",
            "vk_window_create_swapchain",
            "vk_window_get_swapchain_images",
            "vk_window_acquire_image",
            "vk_window_reset_fence",
            "vk_window_reset_command_buffer",
            "vk_window_begin_command_buffer",
            "vk_window_end_command_buffer",
            "vk_window_submit_present",
            "vk_window_queue_present",
            "vk_window_destroy_queue_wait_idle",
        ],
    },
    DeclineClass {
        // The last two `DrawError::Vulkan(String)` sites in `engine/mod.rs` were
        // not Vulkan calls: both zero-copy export rails `dup(2)` the cached
        // exportable-image fd so the importer owns its own copy, and that
        // `try_clone_to_owned` returns `std::io::Error` (an errno), not a
        // `vk::Result` — so they are `FdDupDecline`, not `VkCall`. Carried by
        // `DrawError::FdDup`, which delegates its slug/fields here. Reaches the
        // same two export sinks the `vk_export_{scanout,present}_*` rails do,
        // because it fails inside the very same two functions.
        type_name: "FdDupDecline",
        defined_in: "backend/vulkan/engine/fd_dup.rs",
        slug_blocks: &[],
        emission: Emission::At(&[
            ("runtime/drain/mod.rs", "scanout_gl_export_fail"),
            ("lib.rs", "export_present"),
        ]),
        slug_calls: &[],
        slugs: &["fd_dup_export_scanout", "fd_dup_export_present"],
    },
    DeclineClass {
        type_name: "ZeroCopyDecline",
        defined_in: "backend/vulkan/caps/zero_copy.rs",
        slug_blocks: &[],
        // Named at device bring-up, one line per degraded rail. It also appears
        // inside the one-shot `vk_caps` summary, but a bracketed field in a
        // twenty-field line is not something `grep reason=` finds.
        emission: Emission::At(&[(
            "backend/vulkan/engine/context.rs",
            "vk_caps_zero_copy_declined",
        )]),
        slug_calls: &[],
        slugs: &["no_host_pointer_import"],
    },
    DeclineClass {
        type_name: "HandoffDecline",
        defined_in: "backend/vulkan/caps/frame_interop.rs",
        slug_blocks: &[],
        emission: Emission::At(&[(
            "backend/vulkan/engine/context.rs",
            "vk_caps_handoff_declined",
        )]),
        slug_calls: &[],
        slugs: &["no_dmabuf_extensions", "no_engine_swapchain"],
    },
    DeclineClass {
        // Host-owned window bring-up. The old three String variants
        // (`EventLoop`/`Vulkan`/`Handle`) collapsed 26 checks into 3 grep
        // prefixes — the Linux `VkState::new` device build alone hid 17 ash
        // calls behind `Vulkan(String)`, its two `semaphore: {e}` sites naming
        // the same prose for two objects. Each check now has its own slug.
        type_name: "WindowError",
        defined_in: "host_window/present.rs",
        slug_blocks: &[],
        // Four boundaries carry it. `resumed` (present.rs) logs both the macOS
        // engine attach and the Linux VkState bring-up under `host_window_init`;
        // the two macOS main-thread lifecycle entries and the Linux
        // window-thread join live in lib.rs. On macOS the join is never taken
        // (the window runs on the process main thread), so `host_window_run`
        // fires only on the Linux spawn path.
        emission: Emission::At(&[
            ("host_window/present.rs", "host_window_init"),
            ("host_window/present.rs", "host_window_present"),
            ("host_window/present.rs", "direct_present_degrade"),
            ("lib.rs", "host_window_start"),
            ("lib.rs", "host_window_main"),
            ("lib.rs", "host_window_run"),
        ]),
        slug_calls: &[],
        slugs: &[
            "window_event_loop_build",
            "window_run_app",
            "window_main_loop_run",
            "window_already_owned",
            "window_no_registered_window",
            "window_wrong_owner",
            "window_create_native_window",
            "window_attach_display_handle",
            "window_attach_window_handle",
            "window_attach_engine",
            "window_vk_load_loader",
            "window_vk_display_handle",
            "window_vk_window_handle",
            "window_vk_required_exts",
            "window_vk_enumerate_instance_exts",
            "window_vk_create_instance",
            "window_vk_create_surface",
            "window_vk_enumerate_physical_devices",
            "window_vk_no_usable_device",
            "window_vk_enumerate_device_exts",
            "window_vk_no_swapchain_extension",
            "window_vk_create_device",
            "window_vk_command_pool",
            "window_vk_alloc_cmd",
            "window_vk_semaphore_image_available",
            "window_vk_semaphore_render_finished",
            "window_vk_fence",
            "window_present_acquire",
            "window_present_reset_fence",
            "window_present_reset_command_buffer",
            "window_present_begin_command_buffer",
            "window_present_end_command_buffer",
            "window_present_queue_submit",
            "window_present_queue",
            "window_staging_create_image",
            "window_staging_memory_type_unavailable",
            "window_staging_allocate_memory",
            "window_staging_bind_memory",
            "window_staging_map_memory",
            "window_dmabuf_import_extensions_missing",
            "window_dmabuf_ring_index_out_of_range",
        ],
    },
    DeclineClass {
        type_name: "ImageFormatSpecializeError",
        defined_in: "runtime/spirv_bind.rs",
        slug_blocks: &[],
        emission: Emission::At(&[(
            "runtime/compute_exec/mod.rs",
            "compute_linux_storage_format",
        )]),
        slug_calls: &[],
        slugs: &[
            "spirv_format_specialize_malformed",
            "spirv_format_specialize_missing_binding",
            "spirv_format_specialize_ambiguous_binding",
        ],
    },
    DeclineClass {
        type_name: "HostImportDecline",
        defined_in: "backend/vulkan/engine/host_import_decline.rs",
        slug_blocks: &[],
        emission: Emission::At(&[
            (
                "backend/vulkan/engine/pools/host_import_and_teardown.rs",
                "host_import_fail",
            ),
            (
                "backend/vulkan/engine/exec_compute.rs",
                "compute_direct_writeback",
            ),
        ]),
        slug_calls: &[],
        slugs: &[
            "host_import_region_count_cap",
            "host_import_total_byte_cap",
            "host_import_zero_length_span",
            "host_import_extension_absent",
            "host_import_pointer_misaligned",
            "host_import_size_misaligned",
            "host_import_range_overflow",
            "host_import_no_valid_window",
        ],
    },
    DeclineClass {
        type_name: "MipmapStatus",
        defined_in: "runtime/mipmap.rs",
        slug_blocks: &[],
        emission: Emission::At(&[("runtime/exec.rs", "blit_generate_mipmaps")]),
        slug_calls: &[],
        slugs: &[
            "mipmap_missing_texture",
            "mipmap_single_level",
            "mipmap_incomplete_layout",
            "mipmap_unsupported_format",
            "mipmap_capacity",
            "mipmap_guest_io",
        ],
    },
    DeclineClass {
        type_name: "MetalMipmapError",
        defined_in: "backend/metal/mipmap.rs",
        slug_blocks: &[],
        // Final Metal failures delegate through `MipmapStatus` to the existing
        // exec boundary. A missing device takes the correct-but-slower CPU
        // filter path and emits its own always-on degradation notice.
        emission: Emission::At(&[
            ("runtime/exec.rs", "blit_generate_mipmaps"),
            ("runtime/mipmap.rs", "mipmap_metal_fallback"),
        ]),
        slug_calls: &[],
        slugs: &[
            "metal_mipmap_device_unavailable",
            "metal_mipmap_format_unsupported",
            "metal_mipmap_width_zero",
            "metal_mipmap_height_zero",
            "metal_mipmap_level_count_too_small",
            "metal_mipmap_base_span_overflow",
            "metal_mipmap_level0_too_short",
            "metal_mipmap_level_count_rejected",
            "metal_mipmap_command_buffer_failed",
            "metal_mipmap_level_span_overflow",
        ],
    },
    DeclineClass {
        type_name: "MetalPipelineDecline",
        defined_in: "backend/metal/raw_metal.rs",
        slug_blocks: &[],
        // The reflection probe is genuine when compute texture reflection
        // requested a PSO. A separate argument-buffer probe uses the same
        // helper speculatively and correctly remains silent on absence.
        emission: Emission::At(&[
            ("backend/metal/compute.rs", "metal_compute_reflection_pso"),
            (
                "runtime/compute_session.rs",
                "compute_icb_argument_buffer_reflection",
            ),
        ]),
        slug_calls: &[],
        slugs: &["metal_compute_reflection_pipeline_create"],
    },
    DeclineClass {
        type_name: "MetalStateDecline",
        defined_in: "runtime/metal_draw/mod.rs",
        slug_blocks: &[],
        // A nonzero sampler or depth-stencil ref is an explicit guest bind.
        // Direct draws retain their prior default/disabled recovery, but emit
        // the exact resolver check. ICB execution does the same for sampler
        // recovery and refuses depth/stencil state it cannot encode rather
        // than silently executing a different render pass.
        emission: Emission::At(&[
            ("runtime/metal_draw/mod.rs", "metal_draw_sampler_fallback"),
            (
                "runtime/metal_draw/mod.rs",
                "metal_draw_depth_stencil_fallback",
            ),
            (
                "runtime/metal_draw/metal_icb.rs",
                "metal_icb_sampler_fallback",
            ),
            (
                "runtime/metal_draw/metal_icb.rs",
                "metal_icb_depth_stencil_refused",
            ),
        ]),
        slug_calls: &[],
        // Descriptor decode variants delegate the existing `DecodeStatus`
        // resource slugs, whose registry row owns those names.
        slugs: &[
            "metal_sampler_entry_missing",
            "metal_sampler_object_type",
            "metal_sampler_descriptor_missing",
            "metal_depth_stencil_entry_missing",
            "metal_depth_stencil_object_type",
            "metal_depth_stencil_descriptor_missing",
            "metal_icb_depth_stencil_unsupported",
        ],
    },
    DeclineClass {
        type_name: "MetalIcbInheritanceDecline",
        defined_in: "runtime/metal_draw/metal_icb.rs",
        slug_blocks: &[],
        // The parent encoder must supply all state that classic indirect render
        // commands cannot record. Each failed bind or inherited-pipeline check
        // returns here and is emitted before the ICB execute is abandoned.
        emission: Emission::At(&[("runtime/metal_draw/metal_icb.rs", "metal_icb_inheritance")]),
        slug_calls: &[],
        slugs: &[
            "metal_icb_inherit_cull_mode_unsupported",
            "metal_icb_inherit_front_facing_unsupported",
            "metal_icb_inherit_vertex_buffer_index_out_of_range",
            "metal_icb_inherit_fragment_buffer_index_out_of_range",
            "metal_icb_inherit_vertex_buffer_missing",
            "metal_icb_inherit_fragment_buffer_missing",
            "metal_icb_inherit_vertex_texture_index_out_of_range",
            "metal_icb_inherit_fragment_texture_index_out_of_range",
            "metal_icb_inherit_vertex_texture_missing",
            "metal_icb_inherit_fragment_texture_missing",
            "metal_icb_inherit_vertex_sampler_index_out_of_range",
            "metal_icb_inherit_fragment_sampler_index_out_of_range",
            "metal_icb_inherit_pipeline_ref_zero",
            "metal_icb_inherit_pipeline_missing",
            "metal_icb_inherit_vertex_mtlb_missing",
            "metal_icb_inherit_fragment_mtlb_missing",
            "metal_icb_inherit_vertex_library_load",
            "metal_icb_inherit_fragment_library_load",
            "metal_icb_inherit_vertex_function_count",
            "metal_icb_inherit_fragment_function_count",
            "metal_icb_inherit_vertex_function_get",
            "metal_icb_inherit_fragment_function_get",
            "metal_icb_inherit_vertex_descriptor_missing",
            "metal_icb_inherit_render_pipeline_create",
        ],
    },
    DeclineClass {
        type_name: "Status",
        defined_in: "backend/metal/error.rs",
        slug_blocks: &[],
        // `ffi.rs` emits the status directly. The product render and compute
        // rails call the same helpers without crossing C: they retain Status
        // inside EncodeStatus / ComputeStatus, whose final record boundaries
        // delegate its exact slug and structured fields to the sink.
        emission: Emission::At(&[
            ("backend/metal/ffi.rs", "metal_ffi"),
            ("runtime/exec.rs", "draw_encode_fail"),
            ("runtime/exec.rs", "compute_record"),
            ("backend/metal/runtime.rs", "metal_buffer_copy_fallback"),
            (
                "runtime/metal_draw/metal_icb.rs",
                "metal_icb_sampler_fallback",
            ),
            (
                "runtime/metal_draw/mod.rs",
                "metal_guest_attachment_fallback",
            ),
        ]),
        // The vocabulary lives at the check sites rather than in the status
        // carrier. Removing the payload-free ARGS/EXECUTE constants makes a
        // new refusal impossible without adding one of these literal calls.
        slug_calls: &[
            ("backend/metal/compute.rs", "Status::args"),
            ("backend/metal/compute.rs", "Status::execute"),
            ("backend/metal/ffi.rs", "Status::args"),
            ("backend/metal/ffi.rs", "Status::execute"),
            ("backend/metal/function.rs", "Status::args"),
            ("backend/metal/function.rs", "Status::execute"),
            ("backend/metal/render.rs", "Status::args"),
            ("backend/metal/render.rs", "Status::execute"),
            ("backend/metal/runtime.rs", "Status::args"),
            ("backend/metal/runtime.rs", "Status::execute"),
            ("backend/metal/samplers.rs", "Status::args"),
            ("backend/metal/stage_input.rs", "Status::args"),
            ("backend/metal/stage_input.rs", "Status::execute"),
        ],
        slugs: &[
            "metal_compute_attribute_stride_without_dynamic_layout",
            "metal_compute_backing_length_zero",
            "metal_compute_backing_offset_out_of_range",
            "metal_compute_backing_span_out_of_range",
            "metal_compute_buffer_binding_duplicate",
            "metal_compute_buffer_binding_out_of_range",
            "metal_compute_buffer_create_failed",
            "metal_compute_buffer_data_missing",
            "metal_compute_buffer_length_zero",
            "metal_compute_command_buffer_failed",
            "metal_compute_device_unavailable",
            "metal_compute_dispatch_kind_invalid",
            "metal_compute_dispatch_type_invalid",
            "metal_compute_dynamic_layout_without_attribute_stride",
            "metal_compute_grid_x_zero",
            "metal_compute_grid_y_zero",
            "metal_compute_grid_z_zero",
            "metal_compute_index_buffer_missing",
            "metal_compute_pso_create_failed",
            "metal_compute_reflection_cached_capacity_exceeded",
            "metal_compute_reflection_count_output_missing",
            "metal_compute_reflection_device_unavailable",
            "metal_compute_reflection_mtlb_empty",
            "metal_compute_reflection_pso_create_failed",
            "metal_compute_reflection_texture_access_unsupported",
            "metal_compute_reflection_texture_binding_duplicate",
            "metal_compute_reflection_texture_capacity_exceeded",
            "metal_compute_reflection_texture_index_exceeded",
            "metal_compute_reflection_unavailable",
            "metal_compute_reflection_usage_output_missing",
            "metal_compute_sampled_binding_duplicate",
            "metal_compute_sampled_binding_invalid",
            "metal_compute_sampled_data_missing",
            "metal_compute_sampled_data_too_short",
            "metal_compute_sampled_format_unsupported",
            "metal_compute_sampled_geometry_invalid",
            "metal_compute_sampled_swizzle_invalid",
            "metal_compute_sampled_swizzle_view_create_failed",
            "metal_compute_sampler_binding_duplicate",
            "metal_compute_sampler_binding_invalid",
            "metal_compute_sampler_count_exceeded",
            "metal_compute_stage_input_index_buffer_missing",
            "metal_compute_stage_input_pso_create_failed",
            "metal_compute_storage_binding_duplicate",
            "metal_compute_storage_binding_invalid",
            "metal_compute_storage_data_missing",
            "metal_compute_storage_data_too_short",
            "metal_compute_storage_format_unsupported",
            "metal_compute_storage_geometry_invalid",
            "metal_compute_threadgroup_limit_exceeded",
            "metal_compute_threadgroup_x_zero",
            "metal_compute_threadgroup_y_zero",
            "metal_compute_threadgroup_z_zero",
            "metal_compute_writeback_buffer_count_mismatch",
            "metal_compute_writeback_image_count_mismatch",
            "metal_compute_writeback_storage_format_unsupported",
            "metal_ffi_cache_stats_output_null",
            "metal_ffi_slice_pointer_null",
            "metal_ffi_status_entry_panicked",
            "metal_ffi_void_entry_panicked",
            "metal_function_count_not_one",
            "metal_function_library_create_failed",
            "metal_function_lookup_failed",
            "metal_function_mtlb_empty",
            "metal_buffer_no_copy_length_mismatch",
            "metal_type11_alias_device_unavailable",
            "metal_type11_alias_height_zero",
            "metal_type11_alias_mapping_zero",
            "metal_type11_alias_offset_unaligned",
            "metal_type11_alias_row_bytes_unaligned",
            "metal_type11_alias_span_out_of_range",
            "metal_type11_alias_span_overflow",
            "metal_type11_alias_view_length_zero",
            "metal_type11_alias_view_pointer_null",
            "metal_type11_alias_width_zero",
            "metal_render_back_stencil_state_unsupported",
            "metal_render_blend_alpha_operation_unsupported",
            "metal_render_blend_dst_alpha_unsupported",
            "metal_render_blend_dst_rgb_unsupported",
            "metal_render_blend_rgb_operation_unsupported",
            "metal_render_blend_src_alpha_unsupported",
            "metal_render_blend_src_rgb_unsupported",
            "metal_render_buffer_binding_out_of_range",
            "metal_render_buffer_create_failed",
            "metal_render_buffer_data_missing",
            "metal_render_buffer_length_zero",
            "metal_render_color_format_unsupported",
            "metal_render_color_height_zero",
            "metal_render_color_output_length_mismatch",
            "metal_render_color_seed_length_mismatch",
            "metal_render_color_slot_out_of_range",
            "metal_render_color_span_overflow",
            "metal_render_color_target_count_exceeded",
            "metal_render_color_targets_empty",
            "metal_render_color_width_zero",
            "metal_render_command_buffer_failed",
            "metal_render_cull_mode_unsupported",
            "metal_render_depth_clip_mode_unsupported",
            "metal_render_depth_compare_unsupported",
            "metal_render_depth_data_length_mismatch",
            "metal_render_depth_data_required",
            "metal_render_depth_format_unsupported",
            "metal_render_depth_geometry_invalid",
            "metal_render_depth_length_without_data",
            "metal_render_depth_load_action_unsupported",
            "metal_render_depth_store_action_unsupported",
            "metal_render_device_unavailable",
            "metal_render_fill_mode_unsupported",
            "metal_render_front_stencil_state_unsupported",
            "metal_render_index_buffer_too_short",
            "metal_render_index_byte_count_overflow",
            "metal_render_index_count_zero",
            "metal_render_index_data_empty",
            "metal_render_index_data_missing",
            "metal_render_index_type_unsupported",
            "metal_render_indexed_indirect_arguments_missing",
            "metal_render_indexed_indirect_arguments_too_short",
            "metal_render_indexed_indirect_buffer_too_short",
            "metal_render_indexed_indirect_byte_count_overflow",
            "metal_render_indexed_indirect_range_overflow",
            "metal_render_indirect_and_indexed_conflict",
            "metal_render_instance_count_zero",
            "metal_render_line_width_api_unavailable",
            "metal_render_primitive_indirect_arguments_missing",
            "metal_render_primitive_indirect_arguments_too_short",
            "metal_render_primitive_type_unsupported",
            "metal_render_pso_create_failed",
            "metal_render_sampled_binding_invalid",
            "metal_render_sampled_height_zero",
            "metal_render_sampled_native_data_empty",
            "metal_render_sampled_native_data_missing",
            "metal_render_sampled_native_data_too_short",
            "metal_render_sampled_native_format_unsupported",
            "metal_render_sampled_native_span_overflow",
            "metal_render_sampled_rgba_data_missing",
            "metal_render_sampled_rgba_data_too_short",
            "metal_render_sampled_rgba_geometry_invalid",
            "metal_render_sampled_width_zero",
            "metal_render_sampler_binding_duplicate",
            "metal_render_sampler_binding_invalid",
            "metal_render_scissor_count_exceeded",
            "metal_render_stencil_data_length_mismatch",
            "metal_render_stencil_data_required",
            "metal_render_stencil_format_unsupported",
            "metal_render_stencil_geometry_invalid",
            "metal_render_stencil_length_without_data",
            "metal_render_stencil_load_action_unsupported",
            "metal_render_stencil_store_action_unsupported",
            "metal_render_vertex_attribute_buffer_out_of_range",
            "metal_render_vertex_attribute_count_exceeded",
            "metal_render_vertex_attribute_location_out_of_range",
            "metal_render_vertex_attribute_stride_zero",
            "metal_render_vertex_buffer_count_exceeded",
            "metal_render_vertex_buffer_create_failed",
            "metal_render_vertex_buffer_index_conflict",
            "metal_render_vertex_count_zero",
            "metal_render_vertex_step_function_unsupported",
            "metal_render_vertex_step_rate_zero",
            "metal_render_viewport_count_exceeded",
            "metal_render_winding_unsupported",
            "metal_sampler_address_r_unsupported",
            "metal_sampler_address_s_unsupported",
            "metal_sampler_address_t_unsupported",
            "metal_sampler_anisotropy_zero",
            "metal_sampler_border_color_unsupported",
            "metal_sampler_compare_function_unsupported",
            "metal_sampler_mag_filter_unsupported",
            "metal_sampler_min_filter_unsupported",
            "metal_sampler_mip_filter_unsupported",
            "metal_stage_input_attribute_buffer_out_of_range",
            "metal_stage_input_attribute_count_exceeded",
            "metal_stage_input_attribute_duplicate",
            "metal_stage_input_attribute_format_unsupported",
            "metal_stage_input_attribute_layout_missing",
            "metal_stage_input_attribute_out_of_range",
            "metal_stage_input_attributes_unavailable",
            "metal_stage_input_index_buffer_out_of_range",
            "metal_stage_input_index_type_unsupported",
            "metal_stage_input_layout_buffer_duplicate",
            "metal_stage_input_layout_buffer_out_of_range",
            "metal_stage_input_layout_count_exceeded",
            "metal_stage_input_layouts_unavailable",
            "metal_stage_input_step_function_unsupported",
        ],
    },
    DeclineClass {
        type_name: "ComputeSpirvDecline",
        defined_in: "runtime/compute_exec/mod.rs",
        slug_blocks: &[],
        emission: Emission::At(&[("runtime/compute_exec/mod.rs", "compute_linux_spirv_parse")]),
        slug_calls: &[],
        slugs: &[
            "compute_spirv_header_too_short",
            "compute_spirv_length_misaligned",
        ],
    },
    DeclineClass {
        type_name: "ComputeStatus",
        defined_in: "runtime/compute_exec/mod.rs",
        slug_blocks: &[],
        // The rail boundary: `exec.rs` turns a refused record into one line
        // naming the check, and closes the segment-end session the same way.
        // Every construction site below already logs its own detail line; what
        // these two add is the *registered* reason on the record that was lost.
        emission: Emission::At(&[
            ("runtime/exec.rs", "compute_record"),
            ("runtime/exec.rs", "compute_session_finish"),
        ]),
        // Nine of eleven variants used to be payload-free, and 129 sites
        // collapsed into them — `MetalFailed` spoke for 38 checks alone. The
        // reason now rides in the value, so the vocabulary is written at the
        // construction sites rather than in a match arm.
        slug_calls: &[
            (
                "runtime/compute_exec/mod.rs",
                "ComputeStatus::MissingPipeline(",
            ),
            ("runtime/compute_exec/mod.rs", "ComputeStatus::MissingMtlb("),
            (
                "runtime/compute_exec/mod.rs",
                "ComputeStatus::MissingBuffer(",
            ),
            (
                "runtime/compute_exec/mod.rs",
                "ComputeStatus::MissingTexture(",
            ),
            (
                "runtime/compute_exec/mod.rs",
                "ComputeStatus::MissingSampler(",
            ),
            ("runtime/compute_exec/mod.rs", "ComputeStatus::BadGrid("),
            ("runtime/compute_exec/mod.rs", "ComputeStatus::GuestIo("),
            ("runtime/compute_exec/mod.rs", "ComputeStatus::MetalFailed("),
            ("runtime/compute_exec/mod.rs", "ComputeStatus::NoMetal("),
            ("runtime/compute_exec/mod.rs", "ComputeStatus::Unsupported("),
            (
                "runtime/compute_session.rs",
                "ComputeStatus::MissingPipeline(",
            ),
            ("runtime/compute_session.rs", "ComputeStatus::MissingMtlb("),
            (
                "runtime/compute_session.rs",
                "ComputeStatus::MissingBuffer(",
            ),
            (
                "runtime/compute_session.rs",
                "ComputeStatus::MissingSampler(",
            ),
            ("runtime/compute_session.rs", "ComputeStatus::GuestIo("),
            ("runtime/compute_session.rs", "ComputeStatus::MetalFailed("),
            ("runtime/compute_session.rs", "ComputeStatus::NoMetal("),
            ("runtime/compute_session.rs", "ComputeStatus::Unsupported("),
        ],
        slugs: &[
            "buffer_spirv_ambiguous_binding",
            "buffer_spirv_pointer_escape",
            "compute_buf_win_gva_overflow",
            "compute_buf_win_no_backing",
            "compute_buf_win_oob",
            "compute_buf_win_read",
            "compute_buffer_texture_unsupported",
            "compute_control_cond_offset_overflow",
            "compute_control_end_do_while",
            "compute_control_end_if",
            "compute_control_end_while",
            "compute_control_no_vulkan_path",
            "compute_dispatch_no_backend",
            "compute_grid_dim_range",
            "compute_heap_fmt_bytes",
            "compute_heap_fmt_storage",
            "compute_heap_host_len",
            "compute_heap_shape",
            "compute_heap_use_offset",
            "compute_icb_inherit_ab_sampler_bad_tag",
            "compute_icb_inherit_ab_sampler_decode",
            "compute_icb_inherit_ab_sampler_make",
            "compute_icb_inherit_ab_sampler_no_desc",
            "compute_icb_inherit_ab_sampler_no_entry",
            "compute_icb_inherit_ab_sampler_wrong_type",
            "compute_icb_inherit_ab_zero_len",
            "compute_icb_inherit_bind_sampled",
            "compute_icb_inherit_bind_samplers",
            "compute_icb_inherit_bind_storage",
            "compute_icb_inherit_buffer_alloc",
            "compute_icb_inherit_function",
            "compute_icb_inherit_library",
            "compute_icb_inherit_mtlb_load",
            "compute_icb_inherit_pipeline_load",
            "compute_icb_inherit_pipeline_ref_zero",
            "compute_icb_inherit_sampler_bad_tag",
            "compute_icb_inherit_sampler_decode",
            "compute_icb_inherit_sampler_no_desc",
            "compute_icb_inherit_sampler_no_entry",
            "compute_icb_inherit_sampler_wrong_type",
            "compute_icb_inherit_storage_image_missing",
            "compute_icb_inherit_tex_mtlb_load",
            "compute_icb_inherit_tex_pipeline_load",
            "compute_icb_inherit_tex_pipeline_ref_zero",
            "compute_icb_inherit_texture_short",
            "compute_icb_no_vulkan_path",
            "compute_icb_ref_zero",
            "compute_linear_tex_desc_decode",
            "compute_linear_tex_no_desc",
            "compute_linear_tex_no_entry",
            "compute_linear_tex_no_level",
            "compute_linear_tex_not_texture",
            "compute_linear_tex_stride_lt_tight",
            "compute_linear_tex_zero_geom",
            "compute_mtl_mtlb_load",
            "compute_mtl_pipeline_load",
            "compute_mtl_pipeline_ref_zero",
            "compute_mtl_retain_image_count",
            "compute_mtl_sampler_bad_tag",
            "compute_mtl_sampler_decode",
            "compute_mtl_sampler_no_desc",
            "compute_mtl_sampler_no_entry",
            "compute_mtl_sampler_wrong_type",
            "compute_nested_no_vulkan_path",
            "compute_nested_writeback_metal",
            "compute_restage_heap_lost",
            "compute_restage_linear_lost",
            "compute_restage_no_skipped_resource",
            "compute_restage_read",
            "compute_restage_sampled_heap_lost",
            "compute_restage_sampled_linear_lost",
            "compute_restage_sampled_read",
            "compute_restage_skip_without_identity",
            "compute_session_command_buffer_error",
            "compute_session_no_metal_device",
            "compute_session_no_vulkan_path",
            "compute_stage_buf_decode",
            "compute_stage_buf_gva_overflow",
            "compute_stage_buf_gva_read",
            "compute_stage_buf_no_backing",
            "compute_stage_buf_no_desc",
            "compute_stage_buf_no_entry",
            "compute_stage_buf_off_oob",
            "compute_stage_buf_want_bad",
            "compute_stage_buf_wrong_type",
            "compute_stage_tex_heap_bad_len",
            "compute_stage_tex_heap_desc_decode",
            "compute_stage_tex_heap_resident_lost",
            "compute_stage_tex_heap_zero_ref",
            "compute_stage_tex_linear_row_gva",
            "compute_stage_tex_linear_row_offset",
            "compute_stage_tex_linear_row_read",
            "compute_stage_tex_mapping_gone",
            "compute_stage_tex_mapping_no_geom",
            "compute_stage_tex_type11_no_geom",
            "compute_stage_tex_type11_read",
            "compute_stage_tex_type11_span",
            "compute_stage_tex_type11_window",
            "compute_stage_tex_type5_no_map",
            "compute_stage_tex_view_no_desc",
            "compute_stage_tex_view_resolve",
            "compute_stage_tex_zero_geom",
            "compute_view_format",
            "compute_view_swizzle_unsupported",
            "compute_view_type11_mip",
            "compute_vk_air_extract",
            "compute_vk_deferred_identity",
            "compute_vk_deferred_linear_note",
            "compute_vk_direct_non_type11",
            "compute_vk_engine_run",
            "compute_vk_mtlb_load",
            "compute_vk_pipeline_load",
            "compute_vk_pipeline_ref_zero",
            "compute_vk_readback_binding",
            "compute_vk_readback_count",
            "compute_vk_sampler_load",
            "compute_vk_spirv_parse",
            "compute_vk_translate",
            "compute_vk_zero_dims",
            "compute_wb_buf_task_gva_write",
            "compute_wb_tex_linear_cache_store",
            "compute_wb_tex_linear_guest_write",
            "compute_wb_tex_linear_layout",
            "compute_wb_tex_linear_span_overflow",
            "compute_wb_tex_type11_write",
            "control_flow_unknown_kind",
            "dispatch_in_sequencing_block",
            "engine_run_unsupported",
            "icb_ab_bind_count_mismatch",
            "icb_buffer_index_exceeds_max",
            "icb_encode_unknown_kind",
            "icb_library_function_count",
            "icb_range_exceeds_size",
            "icb_storage_selector_missing",
            "icb_texture_image_len_overflow",
            "icb_texture_selector_missing",
            "icb_texture_selector_unsupported",
            "linear_tex_fmt_bytes",
            "linear_tex_fmt_storage",
            "linear_tex_need_overflow",
            "linear_tex_no_fmt",
            "linear_tex_tight_overflow",
            "linear_tex_view_format",
            "linux_stage_in_imageblock",
            "metal_no_backend_selector",
            "resolve_dims_unknown_kind",
            "sampled_format_unsupported",
            "sequencing_block_active",
            "sequencing_unknown_kind",
            "stage_tex_fmt_bytes",
            "stage_tex_fmt_storage",
            "stage_tex_fmt_unknown",
            "stage_tex_host_len",
            "stage_tex_multiplane_no_plane",
            "stage_tex_need_overflow",
            "stage_tex_tight_bpr_overflow",
            "storage_format_specialize_error",
            "storage_format_specialize_internal",
            "storage_format_specialize_mismatch",
            "storage_no_selector_specialize",
            "storage_no_selector_writeback",
            "storage_selector_unknown_specialize",
            "storage_selector_unknown_writeback",
            "storage_spirv_format_missing",
            "storage_spirv_format_unsupported",
            "texture_spirv_storage_access_missing",
            "texture_spirv_storage_ambiguous_binding",
        ],
    },
    DeclineClass {
        type_name: "DecodeStatus",
        // One of the seven enums wearing this name, one per `runtime/decode/`
        // module. Each gets its own row and its own slug prefix, because five of
        // the seven have an `ErrShort` meaning a different read.
        defined_in: "runtime/decode/blit.rs",
        slug_blocks: &[],
        emission: Emission::At(&[("runtime/exec.rs", "blit_decode")]),
        slug_calls: &[],
        slugs: &[
            "blit_decode_args",
            "blit_decode_short",
            "blit_decode_unknown_opcode",
            "blit_decode_unsupported_opcode",
        ],
    },
    DeclineClass {
        type_name: "DecodeStatus",
        defined_in: "runtime/decode/render.rs",
        slug_blocks: &[],
        emission: Emission::At(&[("runtime/exec.rs", "render_decode")]),
        slug_calls: &[],
        slugs: &[
            "render_decode_args",
            "render_decode_short",
            "render_decode_unknown_opcode",
            "render_decode_unsupported_opcode",
            "render_decode_bad_length",
        ],
    },
    DeclineClass {
        type_name: "DecodeStatus",
        defined_in: "runtime/decode/compute.rs",
        slug_blocks: &[],
        emission: Emission::At(&[("runtime/exec.rs", "compute_decode")]),
        slug_calls: &[],
        slugs: &[
            "compute_decode_args",
            "compute_decode_short",
            "compute_decode_unknown_opcode",
            "compute_decode_unsupported_opcode",
            "compute_decode_too_many_bindings",
        ],
    },
    DeclineClass {
        type_name: "DecodeStatus",
        // The outermost decoder: it frames every guest command, so a refusal here
        // means a whole stream or the tail of a segment went unexecuted.
        defined_in: "runtime/decode/stream.rs",
        slug_blocks: &[],
        emission: Emission::At(&[
            ("runtime/exec.rs", "stream_frame_fail"),
            ("runtime/exec.rs", "stream_record_fail"),
        ]),
        // The reason rides in the variant, so the vocabulary is written where the
        // refusals are constructed rather than in `refusal()` — whose single arm
        // forwards all three variants' payloads and names no slug itself.
        slug_calls: &[
            ("runtime/decode/stream.rs", "DecodeStatus::ErrArgs("),
            ("runtime/decode/stream.rs", "DecodeStatus::ErrShort("),
            ("runtime/decode/stream.rs", "DecodeStatus::ErrBadLength("),
        ],
        slugs: &[
            "stream_bytes_len_overflow",
            "stream_index_walk_short_header",
            "stream_index_walk_cursor_overflow",
            "stream_index_walk_seg_len",
            "stream_index_target_offset_not_found",
            "stream_seg_cursor_past_end",
            "stream_seg_short_header",
            "stream_seg_len_below_header",
            "stream_seg_len_past_buffer_end",
            "stream_seg_cursor_overflow",
            "stream_reval_len_below_header",
            "stream_reval_span_oob",
            "stream_reval_header_mismatch",
            "stream_reval_command_span_mismatch",
            "stream_reval_command_offset_overflow",
            "stream_reval_command_end_oob",
            "stream_rec_cursor_out_of_segment",
            "stream_rec_protection_cursor_misaligned",
            "stream_rec_short_header",
            "stream_rec_len_below_header",
            "stream_rec_len_past_segment_end",
            "stream_rec_cursor_overflow",
            "stream_protection_wrong_segment_type",
            "stream_protection_payload_len",
        ],
    },
    DeclineClass {
        type_name: "SegmentDisposition",
        defined_in: "runtime/decode/stream.rs",
        slug_blocks: &[],
        emission: Emission::At(&[("runtime/exec.rs", "stream_segment")]),
        slug_calls: &[],
        slugs: &["stream_segment_type_unknown"],
    },
    DeclineClass {
        type_name: "DecodeStatus",
        defined_in: "runtime/decode/event.rs",
        slug_blocks: &[],
        emission: Emission::At(&[("runtime/fence_exec.rs", "event_decode")]),
        slug_calls: &[],
        slugs: &[
            "event_decode_args",
            "event_decode_short",
            "event_decode_bad_length",
            "event_decode_unknown_opcode",
            "event_decode_rejected_opcode",
        ],
    },
    DeclineClass {
        type_name: "BlitStatus",
        defined_in: "runtime/blit_exec.rs",
        // `BlitOptionError`'s three slugs arrive through
        // `br(BlitStatus::Unsupported, e.slug())`, so they are counted here
        // rather than left exempt: the channel is what carries them to the sink.
        slug_blocks: &[("runtime/decode/blit.rs", "Decline for BlitOptionError")],
        emission: Emission::At(&[
            ("runtime/exec.rs", "blit_fail"),
            ("runtime/exec.rs", "blit_fence_fail"),
        ]),
        // 182 `br(status, "slug")` sites write 177 distinct reasons. The
        // vocabulary lives at the call sites because the reason travels in a
        // thread-local channel beside the coarse status, not inside it.
        slug_calls: &[("runtime/blit_exec.rs", "br(")],
        slugs: &[
            "b2b_dst_gva_overflow",
            "b2b_overlap",
            "b2b_range_oob",
            "b2b_src_gva_overflow",
            "b2t_dst_alloc_oob",
            "b2t_dst_bpi_overflow",
            "b2t_dst_gva_overflow",
            "b2t_dst_span_overflow",
            "b2t_dst_texel_oob",
            "b2t_origin_oob",
            "b2t_repack_gva_overflow",
            "b2t_row_bytes_overflow",
            "b2t_src_bpi_overflow",
            "b2t_src_bpr_lt_row",
            "b2t_src_gva_overflow",
            "b2t_src_span_oob",
            "b2t_src_span_overflow",
            "b2t_t11_read_io",
            "b2t_t11_src_gva_overflow",
            "b2t_t11_src_span_oob",
            "b2t_t11_src_span_overflow",
            "b2t_t11_z_or_depth",
            "blit_kind_unsupported",
            "buf_desc_decode",
            "buf_desc_read",
            "buf_no_backing",
            "buf_no_list_entry",
            "buf_ref_zero",
            "buf_wrong_type",
            "copy_bytes_dst_overflow",
            "copy_bytes_read_io",
            "copy_bytes_src_overflow",
            "copy_bytes_write_io",
            "copy_kind_none",
            "copy_region_dst_plane_overflow",
            "copy_region_dst_row_overflow",
            "copy_region_read_io",
            "copy_region_row_alloc",
            "copy_region_row_gt_stride",
            "copy_region_src_plane_overflow",
            "copy_region_src_row_overflow",
            "copy_region_total_overflow",
            "copy_region_write_io",
            "fence_bad_opcode",
            "fence_missing",
            "fence_wrong_kind",
            "fill_gva_advance_overflow",
            "fill_gva_overflow",
            "fill_range_oob",
            "fill_write_io",
            "rd_row_buf_cap",
            "rd_row_gva_overflow",
            "rd_row_linear_io",
            "rd_row_t11_coord_range",
            "rd_row_t11_io",
            "rd_row_t11_y_overflow",
            "rd_row_t11_z",
            "rd_row_texel_oob",
            "rd_row_y_overflow",
            "sl_bpp_mismatch",
            "sl_depth_mismatch",
            "sl_dim_mismatch",
            "sl_dst_bpi_zero",
            "sl_dst_gva_overflow",
            "sl_dst_level_overflow",
            "sl_dst_slice_overflow",
            "sl_dst_texel_oob",
            "sl_format_mismatch",
            "sl_inner_dim_mismatch",
            "sl_inner_dst_slice_overflow",
            "sl_inner_src_slice_overflow",
            "sl_missing_ref",
            "sl_overlap",
            "sl_row_bytes_overflow",
            "sl_slice_count_underflow",
            "sl_slice_stride_zero",
            "sl_src_bpi_zero",
            "sl_src_gva_overflow",
            "sl_src_level_overflow",
            "sl_src_slice_overflow",
            "sl_src_texel_oob",
            "sl_volume_mixed",
            "sl_volume_slice_constraint",
            "sl_volume_t11",
            "sl_zero_geom",
            "t11_desc_decode",
            "t11_desc_read",
            "t11_fmt_bpp",
            "t11_level_slice",
            "t11_no_mapping",
            "t11_sample_window",
            "t11_unmapped",
            "t11_zero_geom",
            "t2b_dst_bpi_overflow",
            "t2b_dst_bpr_lt_row",
            "t2b_dst_gva_overflow",
            "t2b_dst_span_oob",
            "t2b_dst_span_overflow",
            "t2b_origin_oob",
            "t2b_repack_gva_overflow",
            "t2b_row_bytes_overflow",
            "t2b_src_bpi_overflow",
            "t2b_src_gva_overflow",
            "t2b_src_texel_oob",
            "t2b_stage_dst_gva_overflow",
            "t2b_stage_dst_span_oob",
            "t2b_stage_dst_span_overflow",
            "t2b_stage_write_io",
            "t2b_t11_z_or_depth",
            "t2t_bpp_mismatch",
            "t2t_dst_bpi_overflow",
            "t2t_dst_gva_overflow",
            "t2t_dst_texel_oob",
            "t2t_extract_plane",
            "t2t_format_mismatch",
            "t2t_insert_plane",
            "t2t_origin_oob",
            "t2t_overlap",
            "t2t_row_bytes_overflow",
            "t2t_src_bpi_overflow",
            "t2t_src_gva_overflow",
            "t2t_src_texel_oob",
            "t2t_t11_volume",
            "t2t_t11_z",
            "t5_desc_read",
            "t5_desc_short",
            "t5_fmt_bpp",
            "t5_level_slice",
            "t5_no_mapping",
            "t5_no_sid",
            "t5_sample_window",
            "t5_unmapped",
            "t5_view_decode",
            "tex_bad_bpp",
            "tex_desc_decode",
            "tex_desc_read",
            "tex_level_gva",
            "tex_level_offset_underflow",
            "tex_no_base_gva",
            "tex_no_list_entry",
            "tex_no_pixel_format",
            "tex_ref_zero",
            "tex_slice_bounds",
            "tex_slice_overflow",
            "tex_slice_single",
            "tex_slice_stride",
            "tex_view_depth_cap",
            "tex_wrong_type",
            "tex_zero_geom",
            "view_1d_height",
            "view_2d_slice",
            "view_3d_slice",
            "view_base_ref_zero",
            "view_desc_decode",
            "view_desc_read",
            "view_fmt_bpp",
            "view_fmt_incompat",
            "view_level_oob",
            "view_level_overflow",
            "view_level_u16",
            "view_slice_oob",
            "view_slice_overflow",
            "view_slice_u16",
            "view_swizzle_nonident",
            "view_swizzle_plan",
            "view_t11_level_slice",
            "view_t11_type",
            "view_type_unsupported",
            "wr_row_buf_cap",
            "wr_row_gva_overflow",
            "wr_row_linear_io",
            "wr_row_t11_coord_range",
            "wr_row_t11_io",
            "wr_row_t11_y_overflow",
            "wr_row_t11_z",
            "wr_row_texel_oob",
            "wr_row_y_overflow",
            // Forwarded from `BlitOptionError` through
            // `br(BlitStatus::Unsupported, e.slug())`; owned by the anchor in
            // `slug_blocks` above, counted here.
            "blit_options_unknown_bits",
            "blit_options_row_linear_pvrtc",
            "blit_options_conflicting_aspects",
            // The channel was empty at a refusing site: the line used to render a
            // bare `reason=` with nothing after it. Written by `refusal()`, not
            // by a `br(` site.
            "blit_unattributed",
        ],
    },
    DeclineClass {
        type_name: "FenceStatus",
        defined_in: "runtime/fence_exec.rs",
        slug_blocks: &[],
        emission: Emission::At(&[("runtime/fence_exec.rs", "fence_exec_fail")]),
        // The reason rides in `Unsupported(&str)`, so the vocabulary is written
        // at the construction sites rather than in `refusal()`.
        slug_calls: &[("runtime/fence_exec.rs", "FenceStatus::Unsupported(")],
        slugs: &[
            "fence_domain_unknown",
            "fence_event_in_fence_path",
            "fence_wait_timeout_unsupported",
            "fence_plan_invalid",
            "event_kind_unknown",
            "event_wait_timeout_unsupported",
            "event_plan_invalid",
            // The guard on the forwarding arm in `execute_event_bytes`:
            // unreachable while every event `DecodeStatus` error refuses, and
            // registered so that if one stops, the gap is a named log line.
            "event_decode_unclassified",
        ],
    },
    DeclineClass {
        type_name: "ResolveStatus",
        defined_in: "contract/gva_resolve.rs",
        slug_blocks: &[],
        // `contract/` is pure and logs nothing — correctly. The walk's fifteen
        // reasons now ride inside `MemError::Unresolved` to the one place that
        // does log, which is what makes the handoff checkable rather than
        // assumed.
        emission: Emission::At(&[
            ("runtime/gva_mem.rs", "gva_write"),
            ("qemu/host_ops.rs", "qemu_host_callback"),
        ]),
        slug_calls: &[],
        slugs: &[
            "gva_args",
            "gva_inactive_task",
            "gva_no_directory",
            "gva_directory_read",
            "gva_zero_root_pfn",
            "gva_zero_depth",
            "gva_depth_too_deep",
            "gva_address_out_of_range",
            "gva_page_table_read",
            "gva_zero_pfn",
            "gva_malformed_pte",
            "gva_span_overflow",
            "gva_visitor_stopped",
            "gva_unsupported_geometry",
            "gva_span_too_large",
        ],
    },
    DeclineClass {
        type_name: "Status",
        defined_in: "contract/iosurface_pages.rs",
        slug_blocks: &[],
        // `contract/` remains pure. The mapper boundary preserves the exact
        // contract check through capture, revalidation, descriptor fallback,
        // and page-table resolve rather than flattening it to `mapper_*`.
        emission: Emission::At(&[
            ("runtime/mapper.rs", "mapper_capture_fail"),
            ("runtime/mapper.rs", "mapper_revalidate_fallback"),
            ("runtime/mapper.rs", "mapper_device_descriptor_fallback"),
            ("runtime/mapper.rs", "mapper_resolve_fail"),
        ]),
        // `Status` mixes `Ok` with 17 reason-carrying refusal classes. The
        // vocabulary is written at construction sites; `Refusal::refusal`
        // forwards the carried slug.
        slug_calls: &[
            ("contract/iosurface_pages.rs", "Status::ErrArgs("),
            ("contract/iosurface_pages.rs", "Status::ErrShortDescriptor("),
            (
                "contract/iosurface_pages.rs",
                "Status::ErrUnsupportedFormat(",
            ),
            ("contract/iosurface_pages.rs", "Status::ErrZeroDimension("),
            ("contract/iosurface_pages.rs", "Status::ErrOverflow("),
            ("contract/iosurface_pages.rs", "Status::ErrMappingIdRange("),
            ("contract/iosurface_pages.rs", "Status::ErrNotKernelVa("),
            ("contract/iosurface_pages.rs", "Status::ErrInternalRead("),
            ("contract/iosurface_pages.rs", "Status::ErrInternalOwner("),
            (
                "contract/iosurface_pages.rs",
                "Status::ErrInternalMappingId(",
            ),
            ("contract/iosurface_pages.rs", "Status::ErrInternalSize("),
            ("contract/iosurface_pages.rs", "Status::ErrInternalFields("),
            ("contract/iosurface_pages.rs", "Status::ErrPageCount("),
            ("contract/iosurface_pages.rs", "Status::ErrPageTableRead("),
            ("contract/iosurface_pages.rs", "Status::ErrPageEntry("),
            ("contract/iosurface_pages.rs", "Status::ErrNoPageTable("),
            ("contract/iosurface_pages.rs", "Status::ErrSpanRange("),
        ],
        slugs: &[
            "iosurface_cached_table_span_uncovered",
            "iosurface_geometry_format_unsupported",
            "iosurface_geometry_height_zero",
            "iosurface_geometry_last_row_end_overflow",
            "iosurface_geometry_last_row_start_overflow",
            "iosurface_geometry_mapping_id_config_range",
            "iosurface_geometry_mapping_id_truncated",
            "iosurface_geometry_mapping_id_u64_range",
            "iosurface_geometry_page_shift_invalid",
            "iosurface_geometry_prior_rows_overflow",
            "iosurface_geometry_sample_window_invalid",
            "iosurface_geometry_tight_row_overflow",
            "iosurface_geometry_width_zero",
            "iosurface_mapper_device_desc_pointer_invalid",
            "iosurface_mapper_device_desc_pointer_read",
            "iosurface_mapper_device_desc_pointer_zero",
            "iosurface_mapper_device_kva_invalid",
            "iosurface_mapper_internal_kva_invalid",
            "iosurface_mapper_internal_mapping_id_read",
            "iosurface_mapper_internal_owner_read",
            "iosurface_mapper_internal_size_read",
            "iosurface_mapper_page_count_read",
            "iosurface_mapper_page_field_48_read",
            "iosurface_mapper_page_field_50_read",
            "iosurface_mapper_request_short",
            "iosurface_page_count_host_addressability",
            "iosurface_page_count_invalid",
            "iosurface_page_table_candidate_missing",
            "iosurface_page_table_entry_invalid",
            "iosurface_page_table_entry_read",
            "iosurface_page_table_failure_unattributed",
            "iosurface_page_table_fields_invalid",
            "iosurface_page_table_gpa_not_ram",
            "iosurface_page_table_pointer_48_invalid",
            "iosurface_page_table_pointer_48_read",
            "iosurface_page_table_pointer_50_invalid",
            "iosurface_page_table_pointer_50_read",
            "iosurface_span_chunk_length_overflow",
            "iosurface_span_end_overflow",
            "iosurface_span_gpa_not_ram",
            "iosurface_span_gpa_overflow",
            "iosurface_span_out_of_range",
            "iosurface_span_page_entry_invalid",
            "iosurface_span_page_index_out_of_range",
            "iosurface_span_page_shift_invalid",
            "iosurface_table_first_entry_invalid",
            "iosurface_table_first_entry_missing",
            "iosurface_table_first_gpa_not_ram",
            "iosurface_texture_descriptor_short",
            "iosurface_validate_internal_kva_invalid",
            "iosurface_validate_internal_owner_mismatch",
            "iosurface_validate_internal_size_mismatch",
            "iosurface_validate_mapper_device_kva_invalid",
            "iosurface_validate_mapping_id_mismatch",
        ],
    },
    DeclineClass {
        type_name: "MapperDecline",
        defined_in: "runtime/mapper.rs",
        slug_blocks: &[],
        emission: Emission::At(&[
            ("runtime/mapper.rs", "mapper_capture_fail"),
            ("runtime/mapper.rs", "mapper_device_descriptor_fallback"),
        ]),
        slug_calls: &[],
        slugs: &[
            "mapper_capture_mapper_xreg_read",
            "mapper_capture_request_type_xreg_read",
            "mapper_capture_internal_xreg_read",
            "mapper_capture_request_type_mismatch",
            "mapper_capture_internal_zero",
            "mapper_capture_internal_kva_invalid",
            "mapper_capture_mapper_kva_invalid",
            "mapper_device_descriptor_read",
        ],
    },
    DeclineClass {
        type_name: "QemuHostDecline",
        defined_in: "qemu/host_ops.rs",
        slug_blocks: &[],
        // These adapter methods cannot return a typed error without changing
        // the long-standing HostOps contract, so the failing callback boundary
        // emits directly. Optional `notify_actions` and `is_ram_gpa` callbacks
        // do not construct this type.
        emission: Emission::At(&[("qemu/host_ops.rs", "qemu_host_adapter")]),
        slug_calls: &[],
        slugs: &[
            "qemu_mono_ns_callback_missing",
            "qemu_schedule_bh_callback_missing",
            "qemu_map_pages_callback_missing",
            "qemu_map_pages_callback_failed",
            "qemu_map_pages_null_pointer",
            "qemu_unmap_pages_callback_missing",
        ],
    },
    DeclineClass {
        type_name: "MemError",
        defined_in: "runtime/host.rs",
        slug_blocks: &[],
        emission: Emission::At(&[
            ("runtime/gva_mem.rs", "gva_write"),
            ("qemu/host_ops.rs", "qemu_host_callback"),
        ]),
        // `Unresolved` delegates to `ResolveStatus`, whose fifteen slugs are
        // counted on its own row rather than restated here.
        slug_calls: &[],
        slugs: &[
            "mem_unmapped",
            "mem_no_cpu",
            "mem_overflow",
            "mem_bad_args",
            "mem_qemu_read_gpa_callback_missing",
            "mem_qemu_read_gpa_callback_failed",
            "mem_qemu_write_gpa_callback_missing",
            "mem_qemu_write_gpa_callback_failed",
            "mem_qemu_read_kva_callback_missing",
            "mem_qemu_read_kva_callback_failed",
            "mem_xreg_unavailable",
            "mem_qemu_read_xreg_callback_missing",
            "mem_qemu_read_xreg_callback_failed",
            "mem_unresolved_ok",
            "mem_no_task_directory",
            "mem_unsupported_page_shift",
            "mem_task_root_read",
            "mem_no_such_task",
            "mem_outside_map",
            "mem_not_contiguous",
        ],
    },
    DeclineClass {
        type_name: "IcbStatus",
        defined_in: "runtime/icb/mod.rs",
        slug_blocks: &[],
        // Two rails carry an ICB refusal to the sink. The compute rail forwards
        // it through `ComputeStatus` (see the `From` impl beside the type), so
        // the reason prints on `exec.rs`'s existing boundary line; the render
        // rail has no reason-carrying status at all, so `metal_draw.rs` names
        // the check itself before collapsing to `EncodeStatus`.
        emission: Emission::At(&[
            ("runtime/exec.rs", "compute_record"),
            ("runtime/exec.rs", "icb_backing"),
            ("runtime/metal_draw/metal_icb.rs", "render_icb"),
        ]),
        // Five variants spoke for 153 checks — `Args` alone for 84 — so the
        // reason rides in the payload and the vocabulary is written at the
        // construction sites, not in a `match` arm.
        slug_calls: &[
            ("runtime/icb/mod.rs", "IcbStatus::Missing("),
            ("runtime/icb/mod.rs", "IcbStatus::BadDescriptor("),
            ("runtime/icb/mod.rs", "IcbStatus::MetalFailed("),
            ("runtime/icb/mod.rs", "IcbStatus::NoMetal("),
            ("runtime/icb/mod.rs", "IcbStatus::Args("),
        ],
        slugs: &[
            "icb_apply_info_no_gpu_address",
            "icb_apply_info_zero_layout_span",
            "icb_associate_buffer_too_small",
            "icb_associate_ref_zero",
            "icb_associate_zero_layout_span",
            "icb_attribute_stride_no_slot",
            "icb_attribute_stride_offset_oob",
            "icb_bind_memory_bad_args",
            "icb_bind_memory_no_vulkan_path",
            "icb_bind_memory_not_cached",
            "icb_dcs_pipeline_offset_oob",
            "icb_dcs_slot_short",
            "icb_dcs_tg_args_oob",
            "icb_dcs_threads_args_oob",
            "icb_dcs_type_offset_oob",
            "icb_dcs_unknown_command_type",
            "icb_desc_no_list_entry",
            "icb_desc_not_icb_body",
            "icb_desc_read",
            "icb_desc_ref_zero",
            "icb_desc_type7_decode",
            "icb_desc_wrong_type",
            "icb_drs_control_point_ref_zero",
            "icb_drs_draw_args_oob",
            "icb_drs_index_buffer_ref_zero",
            "icb_drs_indexed_args_oob",
            "icb_drs_indexed_patches_args_oob",
            "icb_drs_mesh_threadgroups_args_oob",
            "icb_drs_mesh_threads_args_oob",
            "icb_drs_patches_args_oob",
            "icb_drs_pipeline_offset_oob",
            "icb_drs_pipeline_ref_zero",
            "icb_drs_slot_short",
            "icb_drs_type_offset_oob",
            "icb_drs_unknown_command_type",
            "icb_ecs_barrier_offset_oob",
            "icb_ecs_bind_offset_oob",
            "icb_ecs_dispatch_args_oob",
            "icb_ecs_pipeline_offset_oob",
            "icb_ecs_tg_offset_oob",
            "icb_ecs_type_offset_oob",
            "icb_ecs_zero_command_size",
            "icb_ers_bind_offset_oob",
            "icb_ers_draw_args_oob",
            "icb_ers_indexed_args_oob",
            "icb_ers_indexed_patches_args_oob",
            "icb_ers_mesh_threadgroups_args_oob",
            "icb_ers_mesh_threads_args_oob",
            "icb_ers_no_object_tg_table",
            "icb_ers_object_tg_offset_oob",
            "icb_ers_patches_args_oob",
            "icb_ers_zero_command_size",
            "icb_fcc_bind_host_buffer",
            "icb_fcc_bind_index_past_max",
            "icb_fcc_bind_ref_zero",
            "icb_fcc_bind_stage_guest_io",
            "icb_fcc_bind_stage_missing",
            "icb_fcc_bind_stage_other",
            "icb_fcc_command_index_past_capacity",
            "icb_fcc_mtlb_load",
            "icb_fcc_no_metal",
            "icb_fcc_no_vulkan_path",
            "icb_fcc_not_cached",
            "icb_fcc_pipeline_load",
            "icb_fcc_pipeline_ref_zero",
            "icb_fcc_ref_zero",
            "icb_fcc_tg_length_alignment",
            "icb_fcc_threadgroups_zero_dims",
            "icb_fcc_threads_zero_dims",
            "icb_fill_command_memory_read",
            "icb_fill_no_command_memory",
            "icb_fill_no_vulkan_path",
            "icb_fill_not_cached",
            "icb_fill_range_past_capacity",
            "icb_fill_range_past_memory",
            "icb_fill_zero_command_size",
            "icb_frc_base_vertex_range",
            "icb_frc_bind_host_buffer",
            "icb_frc_bind_stage_buffer",
            "icb_frc_command_index_past_capacity",
            "icb_frc_draw_primitive_type",
            "icb_frc_dual_function_get",
            "icb_frc_dual_library_empty",
            "icb_frc_dual_library_load",
            "icb_frc_fragment_function_count",
            "icb_frc_fragment_function_get",
            "icb_frc_fragment_library_load",
            "icb_frc_function_blob_empty",
            "icb_frc_function_blob_read",
            "icb_frc_function_blob_too_large",
            "icb_frc_function_desc_decode",
            "icb_frc_function_desc_read",
            "icb_frc_function_no_list_entry",
            "icb_frc_function_wrong_type",
            "icb_frc_index_span_overflow",
            "icb_frc_index_span_zero",
            "icb_frc_index_type_unknown",
            "icb_frc_indexed_index_type",
            "icb_frc_indexed_no_index_buffer",
            "icb_frc_indexed_patches_no_control_points",
            "icb_frc_indexed_patches_no_tess_buffer",
            "icb_frc_indexed_patches_zero_count",
            "icb_frc_indexed_primitive_type",
            "icb_frc_mesh_library_empty",
            "icb_frc_mesh_library_load",
            "icb_frc_mesh_pipeline_state",
            "icb_frc_mesh_single_function_get",
            "icb_frc_mesh_threadgroups_zero_dims",
            "icb_frc_mesh_threads_zero_dims",
            "icb_frc_mesh_typed_function_get",
            "icb_frc_no_fragment_function",
            "icb_frc_no_mesh_function_resolved",
            "icb_frc_no_mesh_or_vertex_function",
            "icb_frc_no_metal",
            "icb_frc_no_vertex_function",
            "icb_frc_no_vulkan_path",
            "icb_frc_not_cached",
            "icb_frc_object_library_load",
            "icb_frc_object_tg_length_alignment",
            "icb_frc_patches_no_tess_buffer",
            "icb_frc_patches_zero_count",
            "icb_frc_pipeline_desc_decode",
            "icb_frc_pipeline_desc_read",
            "icb_frc_pipeline_no_list_entry",
            "icb_frc_pipeline_ref_zero",
            "icb_frc_pipeline_wrong_type",
            "icb_frc_ref_zero",
            "icb_frc_render_pipeline_state",
            "icb_frc_tess_factor_ref_zero",
            "icb_frc_type1_host_buffer",
            "icb_frc_type1_ref_zero",
            "icb_frc_type1_stage_buffer",
            "icb_frc_vertex_function_count",
            "icb_frc_vertex_function_get",
            "icb_frc_vertex_library_load",
            "icb_host_resource_info_ref_zero",
            "icb_host_resource_info_short",
            "icb_materialize_no_metal",
            "icb_materialize_no_vulkan_path",
            "icb_materialize_zero_command_count",
            "icb_pso_function_count",
            "icb_pso_function_get",
            "icb_pso_library_load",
            "icb_pso_pipeline_state",
            "icb_resolve_no_vulkan_path",
            "icb_type1_desc_decode",
            "icb_type1_desc_read",
            "icb_type1_no_backing",
            "icb_type1_no_list_entry",
            "icb_type1_wrong_type",
            "icb_wire_va_below_base",
            "icb_wire_va_past_end",
            "icb_write_tess_factor_oob",
        ],
    },
    DeclineClass {
        type_name: "EncodeStatus",
        defined_in: "runtime/metal_draw/mod.rs",
        slug_blocks: &[],
        // Two boundaries, both in the exec loop that drives the render rail: the
        // per-record draw counter and the ICB-execute arm. Both used to render
        // the *variant* — `reason=bad_args`, or a Debug-printed `st=BadArgs` with
        // no `reason=` at all — so the rail's 27 checks arrived as six names.
        emission: Emission::At(&[
            ("runtime/exec.rs", "draw_encode_fail"),
            ("runtime/exec.rs", "render_icb"),
        ]),
        // Six payload-free variants spoke for 27 checks, so the reason rides in
        // the payload and the vocabulary is written where the refusals are
        // constructed. Two families, split by which encoder refused — the same
        // `compute_vk_*` / `compute_mtl_*` split the compute rail needed, for the
        // same reason: "no color target" is one check on the Metal encoder and a
        // different one on the Vulkan rail, and a shared slug would make a boot
        // log unable to say which encoder dropped the draw.
        slug_calls: &[
            ("runtime/metal_draw/mod.rs", "EncodeStatus::BadArgs("),
            ("runtime/metal_draw/vulkan.rs", "EncodeStatus::BadArgs("),
            ("runtime/metal_draw/metal_icb.rs", "EncodeStatus::BadArgs("),
            (
                "runtime/metal_draw/mod.rs",
                "EncodeStatus::MissingPipeline(",
            ),
            ("runtime/metal_draw/mod.rs", "EncodeStatus::MissingMtlb("),
            ("runtime/metal_draw/mod.rs", "EncodeStatus::MetalFailed("),
            (
                "runtime/metal_draw/metal_icb.rs",
                "EncodeStatus::MetalFailed(",
            ),
            (
                "runtime/metal_draw/mod.rs",
                "EncodeStatus::WritebackFailed(",
            ),
            (
                "runtime/metal_draw/metal_icb.rs",
                "EncodeStatus::WritebackFailed(",
            ),
            ("runtime/metal_draw/vulkan.rs", "EncodeStatus::NoMetal("),
            ("runtime/metal_draw/metal_icb.rs", "EncodeStatus::NoMetal("),
        ],
        slugs: &[
            // `encode_draw_chain_inner` — the Metal encoder.
            "draw_mtl_no_color_target",
            "draw_mtl_zero_geom",
            "draw_mtl_mrt_geom_mismatch",
            "draw_mtl_no_vertices",
            "draw_mtl_pipeline_load",
            "draw_mtl_vertex_mtlb_load",
            "draw_mtl_fragment_mtlb_load",
            "draw_mtl_vertex_buffer_miss",
            "draw_mtl_fragment_buffer_miss",
            "draw_mtl_vertex_texture_miss",
            "draw_mtl_fragment_texture_miss",
            "draw_mtl_guest_attachment_window",
            "draw_mtl_writeback_none",
            // `encode_draw_chain` — the Vulkan rail. Its other refusals travel
            // inside `DrawError`'s typed variants; these two are the ones that
            // reach the exec loop as a status.
            "draw_vk_no_color_target",
            "draw_vk_nothing_stored",
            // `encode_icb_execute_and_writeback`.
            "icb_exec_ref_zero",
            "icb_exec_no_color_target",
            "icb_exec_geom_mismatch",
            "icb_exec_no_metal_device",
            "icb_exec_range_past_size",
            "icb_exec_command_buffer_error",
            "icb_exec_writeback_none",
            "icb_exec_no_metal_build",
        ],
    },
    DeclineClass {
        type_name: "IndexLoadReason",
        defined_in: "runtime/metal_draw/mod.rs",
        slug_blocks: &[],
        // The Metal rail's indexed-draw site, which was the render rail's one
        // fully silent refusal: `load_index_bytes` was an `.ok()` adapter over
        // the reasoned loader, so all eleven checks dropped the draw with no line
        // at all. The Vulkan rail consumes the same reasons inside `DrawError`
        // (`index_buffer_miss:<slug>`); this row claims the Metal site only.
        emission: Emission::At(&[("runtime/metal_draw/mod.rs", "metal_draw_index")]),
        slug_calls: &[],
        slugs: &[
            "draw_index_type_unsupported",
            "draw_index_count_overflow",
            "draw_index_count_zero",
            "draw_index_entry_missing",
            "draw_index_object_type",
            "draw_index_desc_read",
            "draw_index_desc_decode",
            "draw_index_backing_missing",
            "draw_index_offset_overflow",
            "draw_index_out_of_bounds",
            "draw_index_read_fail",
        ],
    },
    DeclineClass {
        type_name: "DecodeStatus",
        // The sixth and last of the enums wearing this name, and the only one
        // upstream of every rail: blit, compute, render, mipmap and resource
        // registration all reach a guest object through it.
        defined_in: "runtime/decode/resource.rs",
        slug_blocks: &[],
        // Two sites where the reason previously died with no line at all. Most
        // *other* callers wrap it in their own registered slug (`tex_desc_decode`,
        // `compute_stage_tex_heap_desc_decode`, ...), which names the context but
        // not the check; these two named neither.
        emission: Emission::At(&[
            ("runtime/texture.rs", "type11_register"),
            ("runtime/mipmap.rs", "mipmap_texture_desc"),
        ]),
        // The reason rides in the payload — `slug()` forwards all four variants
        // and names nothing itself — so the vocabulary is written where the
        // refusals are constructed. 29 of the 40 sites are `ErrShort`.
        slug_calls: &[
            ("runtime/decode/resource.rs", "DecodeStatus::ErrShort("),
            ("runtime/decode/resource.rs", "DecodeStatus::ErrBadLength("),
            (
                "runtime/decode/resource.rs",
                "DecodeStatus::ErrUnknownType(",
            ),
            (
                "runtime/decode/resource.rs",
                "DecodeStatus::ErrUnsupported(",
            ),
        ],
        slugs: &[
            "res_buffer_desc_short",
            "res_buffer_texture_declared_len",
            "res_buffer_texture_opcode",
            "res_buffer_texture_short",
            "res_compute_pipeline_declared_len",
            "res_compute_pipeline_short",
            "res_compute_pipeline_tag",
            "res_depth_stencil_short",
            "res_depth_stencil_tag",
            "res_function_desc_short",
            "res_icb_desc_short",
            "res_icb_desc_tag",
            "res_icb_layout_short",
            "res_iosurface_short",
            "res_list_entry_short",
            "res_object_entry_short",
            "res_object_type_unknown",
            "res_render_pipeline_declared_len",
            "res_render_pipeline_short",
            "res_render_pipeline_tag",
            "res_sampler_short",
            "res_sampler_tag",
            "res_stage_input_section_oob",
            "res_texture_desc_short",
            "res_texture_view_declared_len",
            "res_texture_view_opcode",
            "res_texture_view_short",
            "res_tlv_header_short",
            "res_tlv_offset_past_end",
            "res_tlv_value_short",
            "res_type7_short",
            "res_type7_subtype_unknown",
            "res_vertex_attr_count_oob",
            "res_vertex_attr_entry_oob",
            "res_vertex_attr_offset_oob",
            "res_vertex_layout_count_oob",
            "res_vertex_layout_entry_oob",
            "res_vertex_layout_offset_oob",
            "res_wide_tlv_bad_length",
            "res_wide_tlv_trailing_bytes",
        ],
    },
    // The always-on censuses. They are not declines in the "I refused your
    // command" sense — a census counts — but the ones below carry a `reason=`,
    // and a `reason=` that is not a registered slug is the shape that teaches a
    // reader to ignore the field. The censuses whose lines carry only *counts*
    // are correctly absent from this table.
    DeclineClass {
        type_name: "TileComposite",
        defined_in: "runtime/census/present_proxy.rs",
        slug_blocks: &[],
        emission: Emission::At(&[("runtime/census/present_proxy.rs", "tile_composite")]),
        slug_calls: &[],
        // A `Refusal`: `Applied` and `NoPeerRequested` are not skips, and the
        // line used to render `reason=applied` for the first of them.
        slugs: &[
            "tile_peer_empty_rects",
            "tile_peer_empty_regions",
            "tile_peer_same_identity",
            "tile_peer_missing",
            "tile_peer_not_ready",
            "tile_peer_not_bgra",
            "tile_peer_geom_mismatch",
        ],
    },
    DeclineClass {
        type_name: "MrtDrop",
        defined_in: "runtime/census/present_proxy.rs",
        slug_blocks: &[],
        emission: Emission::At(&[("runtime/census/present_proxy.rs", "secondary_mrt_drop")]),
        slug_calls: &[],
        slugs: &[
            "mrt_drop_non_contiguous_slot",
            "mrt_drop_geometry_mismatch",
            "mrt_drop_unknown_format",
            "mrt_drop_no_identity",
            "mrt_drop_aliases_primary",
        ],
    },
    DeclineClass {
        type_name: "MaskBindMiss",
        defined_in: "runtime/census/present_proxy.rs",
        slug_blocks: &[],
        emission: Emission::At(&[("runtime/census/present_proxy.rs", "mrt_mask_bind_miss")]),
        slug_calls: &[],
        // `mask_bind_geometry_mismatch` and `MrtDrop`'s
        // `mrt_drop_geometry_mismatch` were both the bare string
        // `"geometry_mismatch"` — two checks in two proxies under one name, which
        // crate-wide uniqueness is exactly the gate for.
        slugs: &[
            "mask_bind_geometry_mismatch",
            "mask_bind_resident_not_ready",
        ],
    },
    DeclineClass {
        type_name: "WindowPublishDrop",
        defined_in: "runtime/census/present_proxy.rs",
        slug_blocks: &[],
        emission: Emission::At(&[("runtime/census/present_proxy.rs", "window_publish")]),
        slug_calls: &[],
        slugs: &["window_publish_resident_not_ready"],
    },
    DeclineClass {
        type_name: "SwizzleDecline",
        defined_in: "runtime/census/view_swizzle_census.rs",
        slug_blocks: &[],
        emission: Emission::At(&[
            (
                "runtime/census/view_swizzle_census.rs",
                "view_swizzle_cpu_remap",
            ),
            (
                "runtime/census/view_swizzle_census.rs",
                "view_swizzle_declined",
            ),
        ]),
        slug_calls: &[],
        slugs: &["swizzle_cpu_remap", "swizzle_resident_direct_bind"],
    },
    DeclineClass {
        type_name: "PipelineCacheDecline",
        defined_in: "backend/vulkan/engine/context.rs",
        slug_blocks: &[],
        emission: Emission::At(&[
            ("backend/vulkan/engine/context.rs", "vk_pipeline_cache_load"),
            ("backend/vulkan/engine/context.rs", "vk_pipeline_cache_save"),
        ]),
        slug_calls: &[],
        slugs: &[
            "vk_pipeline_cache_read",
            "vk_pipeline_cache_incompatible",
            "vk_pipeline_cache_warm_create",
            "vk_pipeline_cache_write",
            "vk_pipeline_cache_rename",
        ],
    },
    DeclineClass {
        type_name: "StatsReduceDecline",
        defined_in: "backend/vulkan/engine/stats_reduce.rs",
        slug_blocks: &[],
        emission: Emission::At(&[
            ("backend/vulkan/engine/mod.rs", "stats_reduce"),
            ("backend/vulkan/engine/stats_reduce.rs", "stats_reduce"),
        ]),
        slug_calls: &[],
        slugs: &[
            "vk_stats_reduce_zero_sequence",
            "vk_stats_reduce_zero_geometry",
        ],
    },
    DeclineClass {
        type_name: "VertexFormatWidenDecline",
        defined_in: "backend/vulkan/engine/caches.rs",
        slug_blocks: &[],
        emission: Emission::At(&[("backend/vulkan/engine/caches.rs", "vk_engine_vertex_format")]),
        slug_calls: &[],
        slugs: &["vk_vertex_format_widened"],
    },
    DeclineClass {
        type_name: "SlabDecline",
        defined_in: "backend/vulkan/engine/slab.rs",
        slug_blocks: &[],
        emission: Emission::At(&[
            ("backend/vulkan/engine/slab.rs", "slab"),
            ("runtime/metal_draw/mod.rs", "linux_m2v_draw"),
            ("runtime/compute_exec/mod.rs", "compute_linux_engine"),
        ]),
        slug_calls: &[],
        slugs: &[
            "vk_slab_free_list_invariant",
            "vk_slab_zero_size",
            "vk_slab_fresh_block_carve",
            "vk_slab_image_already_registered",
            "vk_slab_release_block_missing",
            "vk_slab_release_zero_size",
            "vk_slab_release_range_overflow",
            "vk_slab_release_range_out_of_bounds",
            "vk_slab_release_range_already_free",
        ],
    },
    DeclineClass {
        type_name: "WindowPresentDecline",
        defined_in: "backend/vulkan/engine/window_present.rs",
        slug_blocks: &[],
        emission: Emission::At(&[(
            "backend/vulkan/engine/window_present.rs",
            "host_window_present",
        )]),
        slug_calls: &[],
        slugs: &[
            "window_present_peer_rect_out_of_bounds",
            "window_present_suboptimal_persistent",
        ],
    },
    DeclineClass {
        type_name: "SlateReason",
        defined_in: "backend/vulkan/engine/window_present.rs",
        slug_blocks: &[],
        // A slate present is the window showing nothing — on arm64/MoltenVK the
        // whole blank-window class — so the start of a run is a genuine drop and
        // its end is a census line. Both name the same reason.
        emission: Emission::At(&[
            (
                "backend/vulkan/engine/window_present.rs",
                "host_window_slate",
            ),
            (
                "backend/vulkan/engine/window_present.rs",
                "host_window_slate_end",
            ),
        ]),
        slug_calls: &[],
        slugs: &[
            "slate_no_source",
            "slate_no_resident",
            "slate_content_not_ready",
            "slate_not_bgra",
            "slate_geom_mismatch",
        ],
    },
    // `BlitOptionError` is deliberately NOT here yet. Its three slugs now flow
    // into `blit_exec.rs`'s thread-local reason channel instead of being
    // discarded by `map_err(|_| ..)`, which is a real improvement — but the
    // channel reaches the sink through the dispatch-site line rather than
    // through `Emit`, and `every_registered_type_reaches_the_sink` correctly
    // refuses to certify that hop. Registering it would mean either lying in the
    // row or adding an `Emission` variant to excuse an indirection nothing
    // checks. It lands with `BlitStatus`, when the channel itself is typed; the
    // blit rail's debt is already counted in `gate::STAGED`.
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_rows_are_locally_well_formed() {
        assert!(!REGISTRY.is_empty());
        let retired_flat_modules = [
            "runtime/metal_draw.rs",
            "runtime/compute_exec.rs",
            "runtime/drain.rs",
            "runtime/icb.rs",
        ];
        for class in REGISTRY {
            assert!(!class.type_name.trim().is_empty());
            assert!(class.defined_in.ends_with(".rs"));
            assert!(
                !retired_flat_modules.contains(&class.defined_in),
                "{} still names retired defining file {}",
                class.type_name,
                class.defined_in
            );
            assert!(!class.slugs.is_empty(), "{} has no slugs", class.type_name);
            assert!(
                class.slugs.iter().all(|slug| !slug.trim().is_empty()),
                "{} carries a blank slug",
                class.type_name
            );
            match class.emission {
                Emission::At(sites) => {
                    assert!(
                        !sites.is_empty(),
                        "{} has no emission site",
                        class.type_name
                    );
                    assert!(sites
                        .iter()
                        .all(|(file, event)| file.ends_with(".rs") && !event.is_empty()));
                    assert!(
                        sites
                            .iter()
                            .all(|(file, _)| !retired_flat_modules.contains(file)),
                        "{} still names a retired emission file",
                        class.type_name
                    );
                }
                Emission::Unreachable(reason) => {
                    assert!(
                        !reason.trim().is_empty(),
                        "{} has no unreachable rationale",
                        class.type_name
                    );
                }
            }
            assert!(class
                .slug_blocks
                .iter()
                .all(|(file, _)| !retired_flat_modules.contains(file)));
            assert!(class
                .slug_calls
                .iter()
                .all(|(file, _)| !retired_flat_modules.contains(file)));
        }
    }

    #[test]
    fn registry_has_no_duplicate_type_rows_within_a_defining_module() {
        let mut identities = REGISTRY
            .iter()
            .map(|class| (class.defined_in, class.type_name))
            .collect::<Vec<_>>();
        identities.sort_unstable();
        let before = identities.len();
        identities.dedup();
        assert_eq!(identities.len(), before);
    }
}
