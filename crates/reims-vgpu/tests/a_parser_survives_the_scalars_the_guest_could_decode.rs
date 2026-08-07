//! No decoded guest scalar may kill this device.
//!
//! `a_decoder_survives_bytes_the_guest_could_write` fuzzes every `pub fn` on the
//! parser surface that takes a `&[u8]`. That is the front door, and it is not
//! the whole surface: **76 more `pub fn` across the decoders, `contract` and
//! the Vulkan translation tables take nothing but plain scalars**, and every one
//! of those scalars is a field some byte-slice parser has already pulled out of
//! a guest record. A `u32` opcode, a `u16` pixel format, a `u64` address, a mip
//! level, a page shift, a count and an entry size. The byte-slice fuzzer reaches
//! them only where a decoder happens to route to one, with whatever values that
//! route happens to produce.
//!
//! Three trees, each for its own reason. The **decoders** are where the fields
//! are lifted. **`contract`** *is* the decoded API contract — pixel formats,
//! pass actions, extents, page geometry — and is backend-independent, so it
//! drives on both arms. **`backend::vulkan::translate`** is where a guest
//! ordinal becomes a host enum, which is the position `AGENTS.md` names as
//! undefined behaviour on the Metal side; it is gated on the arm that compiles
//! it, in the scan as well as in the table, so the two agree per arm instead of
//! the table reading short.
//!
//! They are the arithmetic half of the parser. Between them these compute
//! shifts, products, offsets and extents out of guest values, which is where a
//! debug build panics and a release build silently answers wrong — Rust masks
//! an over-wide shift rather than trapping it, so the release consequence is not
//! a crash but an offset into somewhere else.
//!
//! # What this drives
//!
//! Seventy-two of the 76 are called under `catch_unwind` against a cross-product
//! of adversarial scalars. The assertion is only that it **returns** — any
//! value, any `None`, any refusal is a pass. What fails is an unwind. The other
//! four are [`EXEMPT`], which is a checked claim rather than a written one.
//!
//! It earned itself twice, which is the argument for it. On the decoder surface,
//! three `page_shift` functions in `contract::iosurface_pages` unwound with
//! "attempt to shift left with overflow"; on `contract`, a fourth
//! (`gva::pfn_to_gpa`) did the same, and `extent::mip_extent` unwound on a mip
//! **level** — a decoded guest field with no such excuse, which is now total.
//!
//! The corpus is not random. Uniform `u64`s never produce 63, 64, 65 or
//! `u32::MAX` in a slot that matters, and those are the whole population of
//! interesting inputs to a shift or a width. So [`EDGES`] is the values a
//! decoded field goes wrong at, spelled out, and every argument takes every one
//! of them.
//!
//! For arity up to [`FULL_PRODUCT_ARITY`] that is the complete cross-product,
//! which covers every function here today — the widest takes three. The larger
//! sweep below it exists for the next one and is not exhaustive: every argument
//! walked through every edge against every uniform background, plus the
//! all-same-edge corners. A bug needing two arguments extreme *and different* is
//! not covered, and this says so rather than implying otherwise.
//!
//! # Why the population is derived and not listed
//!
//! [`every_all_scalar_parser_is_driven`] re-reads the same trees the byte-slice
//! harness reads and asserts that every `pub fn` there taking only scalars is in
//! the table. A hand-kept list of parsers is a list of the parsers somebody
//! remembered, and this crate has already measured that gap twice.

use std::collections::BTreeSet;
use std::panic::catch_unwind;

mod source_scan;

/// Values a decoded guest field goes wrong at.
///
/// Chosen for the operations these functions perform rather than for coverage
/// of the number line. The shift widths (`31`, `32`, `63`, `64`, `65`) are the
/// ones an over-wide shift panics at; the powers of two either side of a sign
/// bit are where a widening cast changes meaning; `u32::MAX` and `u64::MAX` are
/// where a `+ 1` or a `- 1` leaves the type.
const EDGES: &[u64] = &[
    0,
    1,
    2,
    3,
    7,
    8,
    12,
    14,
    15,
    16,
    31,
    32,
    63,
    64,
    65,
    0x7F,
    0x80,
    0xFF,
    0x100,
    0xFFFF,
    0x1_0000,
    0x7FFF_FFFF,
    0x8000_0000,
    0xFFFF_FFFF,
    0x1_0000_0000,
    u64::MAX - 1,
    u64::MAX,
];

/// Above this many arguments the full cross-product stops being runnable.
///
/// Three arguments is 27^3 = 19 683 calls, which is instant, and three is the
/// widest scalar-only parser on this surface — so every target today takes the
/// full product. Four would be half a million and six would be 387 million,
/// which is what the sampled sweep above this arity is for.
const FULL_PRODUCT_ARITY: usize = 3;

/// One scalar-only parser and how to call it.
struct ScalarTarget {
    /// `module::function`, matching what [`every_all_scalar_parser_is_driven`]
    /// derives from the source.
    name: &'static str,
    arity: usize,
    /// Called with `arity` values drawn from [`EDGES`], cast to the argument
    /// types at the call site.
    run: &'static (dyn Fn(&[u64]) + Sync + std::panic::RefUnwindSafe),
}

fn targets() -> Vec<ScalarTarget> {
    vec![
        ScalarTarget {
            name: "iosurface_pages::dims_extent",
            arity: 1,
            run: &|v| {
                let _ = reims_vgpu::contract::iosurface_pages::dims_extent(v[0]);
            },
        },
        ScalarTarget {
            name: "iosurface_pages::arm_kernel_va",
            arity: 1,
            run: &|v| {
                let _ = reims_vgpu::contract::iosurface_pages::arm_kernel_va(v[0]);
            },
        },
        ScalarTarget {
            name: "iosurface_pages::x86_kernel_va",
            arity: 1,
            run: &|v| {
                let _ = reims_vgpu::contract::iosurface_pages::x86_kernel_va(v[0]);
            },
        },
        ScalarTarget {
            name: "iosurface_pages::guest_kernel_va",
            arity: 1,
            run: &|v| {
                let _ = reims_vgpu::contract::iosurface_pages::guest_kernel_va(v[0]);
            },
        },
        ScalarTarget {
            name: "iosurface_pages::format_bytes_per_pixel",
            arity: 1,
            run: &|v| {
                let _ = reims_vgpu::contract::iosurface_pages::format_bytes_per_pixel(v[0] as u16);
            },
        },
        ScalarTarget {
            name: "iosurface_pages::packed_span_estimate",
            arity: 3,
            run: &|v| {
                let _ = reims_vgpu::contract::iosurface_pages::packed_span_estimate(
                    v[0] as u16,
                    v[1] as u32,
                    v[2] as u32,
                );
            },
        },
        ScalarTarget {
            name: "iosurface_pages::mapper_request_entry_offset",
            arity: 1,
            run: &|v| {
                let _ =
                    reims_vgpu::contract::iosurface_pages::mapper_request_entry_offset(v[0] as u32);
            },
        },
        ScalarTarget {
            name: "iosurface_pages::mapper_request_published_entry_offset",
            arity: 1,
            run: &|v| {
                let _ =
                    reims_vgpu::contract::iosurface_pages::mapper_request_published_entry_offset(
                        v[0] as u32,
                    );
            },
        },
        ScalarTarget {
            name: "blit::parse_blit_options",
            arity: 2,
            run: &|v| {
                let _ = reims_vgpu::runtime::decode::blit::parse_blit_options(
                    v[0] & 1 != 0,
                    v[1] as u32,
                );
            },
        },
        ScalarTarget {
            name: "compute::opcode_supported",
            arity: 1,
            run: &|v| {
                let _ = reims_vgpu::runtime::decode::compute::opcode_supported(v[0] as u32);
            },
        },
        ScalarTarget {
            name: "compute::opcode_apple_rejected",
            arity: 1,
            run: &|v| {
                let _ = reims_vgpu::runtime::decode::compute::opcode_apple_rejected(v[0] as u32);
            },
        },
        ScalarTarget {
            name: "compute::opcode_confidence",
            arity: 1,
            run: &|v| {
                let _ = reims_vgpu::runtime::decode::compute::opcode_confidence(v[0] as u32);
            },
        },
        ScalarTarget {
            name: "event::opcode_rejected_by_deserializer",
            arity: 1,
            run: &|v| {
                let _ = reims_vgpu::runtime::decode::event::opcode_rejected_by_deserializer(
                    v[0] as u32,
                );
            },
        },
        ScalarTarget {
            name: "fifo::display_refresh_hz_1616",
            arity: 1,
            run: &|v| {
                let _ = reims_vgpu::runtime::decode::fifo::display_refresh_hz_1616(v[0] as u32);
            },
        },
        ScalarTarget {
            name: "fifo::display_timing_entry_offset",
            arity: 2,
            run: &|v| {
                let _ = reims_vgpu::runtime::decode::fifo::display_timing_entry_offset(
                    v[0] as u32,
                    v[1],
                );
            },
        },
        ScalarTarget {
            name: "render::bind_record_len",
            arity: 2,
            run: &|v| {
                let _ = reims_vgpu::runtime::decode::render::bind_record_len(
                    v[0] as u32,
                    v[1] as usize,
                );
            },
        },
        ScalarTarget {
            name: "render::opcode_above_the_encoder_window",
            arity: 1,
            run: &|v| {
                let _ = reims_vgpu::runtime::decode::render::opcode_above_the_encoder_window(
                    v[0] as u32,
                );
            },
        },
        ScalarTarget {
            name: "render::opcode_supported",
            arity: 1,
            run: &|v| {
                let _ = reims_vgpu::runtime::decode::render::opcode_supported(v[0] as u32);
            },
        },
        ScalarTarget {
            name: "resource::texture_view_type_supported",
            arity: 1,
            run: &|v| {
                let _ =
                    reims_vgpu::runtime::decode::resource::texture_view_type_supported(v[0] as u16);
            },
        },
        ScalarTarget {
            name: "resource::texture_view_type_uses_slices",
            arity: 1,
            run: &|v| {
                let _ = reims_vgpu::runtime::decode::resource::texture_view_type_uses_slices(
                    v[0] as u16,
                );
            },
        },
        ScalarTarget {
            name: "resource::texture_view_type_is_3d",
            arity: 1,
            run: &|v| {
                let _ = reims_vgpu::runtime::decode::resource::texture_view_type_is_3d(v[0] as u16);
            },
        },
        ScalarTarget {
            name: "resource::list_object_entry_offset",
            arity: 2,
            run: &|v| {
                let _ = reims_vgpu::runtime::decode::resource::list_object_entry_offset(
                    v[0] as u32,
                    v[1] as u32,
                );
            },
        },
        ScalarTarget {
            name: "stream::segment_type_name",
            arity: 1,
            run: &|v| {
                let _ = reims_vgpu::runtime::decode::stream::segment_type_name(v[0] as u32);
            },
        },
        ScalarTarget {
            name: "stream::segment_disposition",
            arity: 1,
            run: &|v| {
                let _ = reims_vgpu::runtime::decode::stream::segment_disposition(v[0] as u8);
            },
        },
        ScalarTarget {
            name: "checked::checked_add_u64",
            arity: 2,
            run: &|v| {
                let _ = reims_vgpu::contract::checked::checked_add_u64(v[0], v[1]);
            },
        },
        ScalarTarget {
            name: "checked::checked_mul_u64",
            arity: 2,
            run: &|v| {
                let _ = reims_vgpu::contract::checked::checked_mul_u64(v[0], v[1]);
            },
        },
        ScalarTarget {
            name: "checked::align_up_u64",
            arity: 2,
            run: &|v| {
                let _ = reims_vgpu::contract::checked::align_up_u64(v[0], v[1]);
            },
        },
        ScalarTarget {
            name: "checked::size_fits_u32",
            arity: 1,
            run: &|v| {
                let _ = reims_vgpu::contract::checked::size_fits_u32(v[0] as usize);
            },
        },
        ScalarTarget {
            name: "dispatch::is_declared_dispatch_type",
            arity: 1,
            run: &|v| {
                let _ = reims_vgpu::contract::dispatch::is_declared_dispatch_type(v[0] as u32);
            },
        },
        ScalarTarget {
            name: "extent::mip_extent",
            arity: 2,
            run: &|v| {
                let _ = reims_vgpu::contract::extent::mip_extent(v[0] as u32, v[1] as u32);
            },
        },
        ScalarTarget {
            name: "extent::tight_image_bytes",
            arity: 3,
            run: &|v| {
                let _ = reims_vgpu::contract::extent::tight_image_bytes(
                    v[0] as u32,
                    v[1] as u32,
                    v[2] as usize,
                );
            },
        },
        ScalarTarget {
            name: "extent::tight_layered_image_bytes",
            arity: 4,
            run: &|v| {
                let _ = reims_vgpu::contract::extent::tight_layered_image_bytes(
                    v[0] as u32,
                    v[1] as u32,
                    v[2] as u32,
                    v[3] as usize,
                );
            },
        },
        ScalarTarget {
            name: "extent::tight_image_layout",
            arity: 3,
            run: &|v| {
                let _ = reims_vgpu::contract::extent::tight_image_layout(
                    v[0] as u32,
                    v[1] as u32,
                    v[2] as u32,
                );
            },
        },
        ScalarTarget {
            name: "fnv::fold_u64",
            arity: 2,
            run: &|v| {
                let _ = reims_vgpu::contract::fnv::fold_u64(v[0], v[1]);
            },
        },
        ScalarTarget {
            name: "mipmap::filterable_bpp",
            arity: 1,
            run: &|v| {
                let _ = reims_vgpu::contract::mipmap::filterable_bpp(v[0] as u16);
            },
        },
        ScalarTarget {
            name: "mipmap::plan_level0",
            arity: 5,
            run: &|v| {
                let _ = reims_vgpu::contract::mipmap::plan_level0(
                    v[0] as u16,
                    v[1] as u32,
                    v[2] as u32,
                    v[3] as u32,
                    v[4] as usize,
                );
            },
        },
        ScalarTarget {
            name: "pass_action::is_declared_load_action",
            arity: 1,
            run: &|v| {
                let _ = reims_vgpu::contract::pass_action::is_declared_load_action(v[0] as u16);
            },
        },
        ScalarTarget {
            name: "pass_action::is_declared_store_action",
            arity: 1,
            run: &|v| {
                let _ = reims_vgpu::contract::pass_action::is_declared_store_action(v[0] as u16);
            },
        },
        ScalarTarget {
            name: "pixel_format::bytes_per_pixel",
            arity: 1,
            run: &|v| {
                let _ = reims_vgpu::contract::pixel_format::bytes_per_pixel(v[0] as u16);
            },
        },
        ScalarTarget {
            name: "pixel_format::format_has_depth_aspect",
            arity: 1,
            run: &|v| {
                let _ = reims_vgpu::contract::pixel_format::format_has_depth_aspect(v[0] as u16);
            },
        },
        ScalarTarget {
            name: "pixel_format::format_has_stencil_aspect",
            arity: 1,
            run: &|v| {
                let _ = reims_vgpu::contract::pixel_format::format_has_stencil_aspect(v[0] as u16);
            },
        },
        ScalarTarget {
            name: "pixel_format::depth_stencil_packing",
            arity: 1,
            run: &|v| {
                let _ = reims_vgpu::contract::pixel_format::depth_stencil_packing(v[0] as u16);
            },
        },
        ScalarTarget {
            name: "pixel_format::is_srgb",
            arity: 1,
            run: &|v| {
                let _ = reims_vgpu::contract::pixel_format::is_srgb(v[0] as u16);
            },
        },
        ScalarTarget {
            name: "pixel_format::sampled_class",
            arity: 1,
            run: &|v| {
                let _ = reims_vgpu::contract::pixel_format::sampled_class(v[0] as u16);
            },
        },
        ScalarTarget {
            name: "pixel_format::storage_selector",
            arity: 1,
            run: &|v| {
                let _ = reims_vgpu::contract::pixel_format::storage_selector(v[0] as u16);
            },
        },
        ScalarTarget {
            name: "pixel_format::render_target_bpp",
            arity: 1,
            run: &|v| {
                let _ = reims_vgpu::contract::pixel_format::render_target_bpp(v[0] as u16);
            },
        },
        ScalarTarget {
            name: "pixel_format::tight_row_bytes",
            arity: 2,
            run: &|v| {
                let _ =
                    reims_vgpu::contract::pixel_format::tight_row_bytes(v[0] as u32, v[1] as u16);
            },
        },
        ScalarTarget {
            name: "pixel_format::f64_to_unorm8",
            arity: 1,
            run: &|v| {
                let _ = reims_vgpu::contract::pixel_format::f64_to_unorm8(f64::from_bits(v[0]));
            },
        },
        ScalarTarget {
            name: "pixel_format::f16_to_f32",
            arity: 1,
            run: &|v| {
                let _ = reims_vgpu::contract::pixel_format::f16_to_f32(v[0] as u16);
            },
        },
        ScalarTarget {
            name: "vertex_step::step_rate_in_contract",
            arity: 2,
            run: &|v| {
                let _ = reims_vgpu::contract::vertex_step::step_rate_in_contract(
                    v[0] as u32,
                    v[1] as u32,
                );
            },
        },
        #[cfg(feature = "backend-vulkan")]
        ScalarTarget {
            name: "blend::factor",
            arity: 1,
            run: &|v| {
                let _ = reims_vgpu::backend::vulkan::translate::blend::factor(v[0] as u32);
            },
        },
        #[cfg(feature = "backend-vulkan")]
        ScalarTarget {
            name: "blend::operation",
            arity: 1,
            run: &|v| {
                let _ = reims_vgpu::backend::vulkan::translate::blend::operation(v[0] as u32);
            },
        },
        #[cfg(feature = "backend-vulkan")]
        ScalarTarget {
            name: "pixel::translate",
            arity: 1,
            run: &|v| {
                let _ = reims_vgpu::backend::vulkan::translate::pixel::translate(v[0] as u16);
            },
        },
        #[cfg(feature = "backend-vulkan")]
        ScalarTarget {
            name: "pixel::is_srgb",
            arity: 1,
            run: &|v| {
                let _ = reims_vgpu::backend::vulkan::translate::pixel::is_srgb(v[0] as u16);
            },
        },
        #[cfg(feature = "backend-vulkan")]
        ScalarTarget {
            name: "pixel::sampled_pixels",
            arity: 1,
            run: &|v| {
                let _ = reims_vgpu::backend::vulkan::translate::pixel::sampled_pixels(v[0] as u16);
            },
        },
        #[cfg(feature = "backend-vulkan")]
        ScalarTarget {
            name: "pixel::color_attachment",
            arity: 1,
            run: &|v| {
                let _ =
                    reims_vgpu::backend::vulkan::translate::pixel::color_attachment(v[0] as u16);
            },
        },
        #[cfg(feature = "backend-vulkan")]
        ScalarTarget {
            name: "pixel::storage_image_from_selector",
            arity: 1,
            run: &|v| {
                let _ = reims_vgpu::backend::vulkan::translate::pixel::storage_image_from_selector(
                    v[0] as u32,
                );
            },
        },
        #[cfg(feature = "backend-vulkan")]
        ScalarTarget {
            name: "pixel::storage_image",
            arity: 1,
            run: &|v| {
                let _ = reims_vgpu::backend::vulkan::translate::pixel::storage_image(v[0] as u16);
            },
        },
        #[cfg(feature = "backend-vulkan")]
        ScalarTarget {
            name: "pixel::resident_color",
            arity: 1,
            run: &|v| {
                let _ =
                    reims_vgpu::backend::vulkan::translate::pixel::resident_color(v[0] & 1 != 0);
            },
        },
        #[cfg(feature = "backend-vulkan")]
        ScalarTarget {
            name: "pixel::has_identity_components",
            arity: 1,
            run: &|v| {
                let _ = reims_vgpu::backend::vulkan::translate::pixel::has_identity_components(
                    v[0] as u16,
                );
            },
        },
        #[cfg(feature = "backend-vulkan")]
        ScalarTarget {
            name: "raster::primitive_topology",
            arity: 1,
            run: &|v| {
                let _ =
                    reims_vgpu::backend::vulkan::translate::raster::primitive_topology(v[0] as u32);
            },
        },
        #[cfg(feature = "backend-vulkan")]
        ScalarTarget {
            name: "raster::cull_mode",
            arity: 1,
            run: &|v| {
                let _ = reims_vgpu::backend::vulkan::translate::raster::cull_mode(v[0] as u32);
            },
        },
        #[cfg(feature = "backend-vulkan")]
        ScalarTarget {
            name: "raster::front_face_ccw",
            arity: 1,
            run: &|v| {
                let _ = reims_vgpu::backend::vulkan::translate::raster::front_face_ccw(v[0] as u32);
            },
        },
        #[cfg(feature = "backend-vulkan")]
        ScalarTarget {
            name: "raster::fill_mode",
            arity: 1,
            run: &|v| {
                let _ = reims_vgpu::backend::vulkan::translate::raster::fill_mode(v[0] as u32);
            },
        },
        #[cfg(feature = "backend-vulkan")]
        ScalarTarget {
            name: "raster::visibility_result_mode",
            arity: 1,
            run: &|v| {
                let _ = reims_vgpu::backend::vulkan::translate::raster::visibility_result_mode(
                    v[0] as u32,
                );
            },
        },
        #[cfg(feature = "backend-vulkan")]
        ScalarTarget {
            name: "raster::depth_clip_mode",
            arity: 1,
            run: &|v| {
                let _ = reims_vgpu::backend::vulkan::translate::raster::depth_clip_mode(v[0] as u32);
            },
        },
        #[cfg(feature = "backend-vulkan")]
        ScalarTarget {
            name: "raster::compare_function",
            arity: 1,
            run: &|v| {
                let _ =
                    reims_vgpu::backend::vulkan::translate::raster::compare_function(v[0] as u32);
            },
        },
        #[cfg(feature = "backend-vulkan")]
        ScalarTarget {
            name: "raster::stencil_operation",
            arity: 1,
            run: &|v| {
                let _ =
                    reims_vgpu::backend::vulkan::translate::raster::stencil_operation(v[0] as u32);
            },
        },
        #[cfg(feature = "backend-vulkan")]
        ScalarTarget {
            name: "raster::index_type",
            arity: 1,
            run: &|v| {
                let _ = reims_vgpu::backend::vulkan::translate::raster::index_type(v[0] as u32);
            },
        },
        #[cfg(feature = "backend-vulkan")]
        ScalarTarget {
            name: "raster::vk_front_face",
            arity: 1,
            run: &|v| {
                let _ =
                    reims_vgpu::backend::vulkan::translate::raster::vk_front_face(v[0] & 1 != 0);
            },
        },
        #[cfg(feature = "backend-vulkan")]
        ScalarTarget {
            name: "sampler::filter",
            arity: 1,
            run: &|v| {
                let _ = reims_vgpu::backend::vulkan::translate::sampler::filter(v[0] as u32);
            },
        },
        #[cfg(feature = "backend-vulkan")]
        ScalarTarget {
            name: "sampler::mip_filter",
            arity: 1,
            run: &|v| {
                let _ = reims_vgpu::backend::vulkan::translate::sampler::mip_filter(v[0] as u32);
            },
        },
        #[cfg(feature = "backend-vulkan")]
        ScalarTarget {
            name: "sampler::address_mode",
            arity: 1,
            run: &|v| {
                let _ = reims_vgpu::backend::vulkan::translate::sampler::address_mode(v[0] as u32);
            },
        },
        #[cfg(feature = "backend-vulkan")]
        ScalarTarget {
            name: "sampler::border_color",
            arity: 1,
            run: &|v| {
                let _ = reims_vgpu::backend::vulkan::translate::sampler::border_color(v[0] as u32);
            },
        },
        #[cfg(feature = "backend-vulkan")]
        ScalarTarget {
            name: "vertex::attribute_format",
            arity: 1,
            run: &|v| {
                let _ =
                    reims_vgpu::backend::vulkan::translate::vertex::attribute_format(v[0] as u32);
            },
        },
    ]
}

/// Every argument combination this target is driven with.
fn corpus(arity: usize) -> Vec<Vec<u64>> {
    let mut out: Vec<Vec<u64>> = Vec::new();
    if arity <= FULL_PRODUCT_ARITY {
        let mut acc: Vec<Vec<u64>> = vec![Vec::new()];
        for _ in 0..arity {
            let mut next = Vec::with_capacity(acc.len() * EDGES.len());
            for prefix in &acc {
                for e in EDGES {
                    let mut v = prefix.clone();
                    v.push(*e);
                    next.push(v);
                }
            }
            acc = next;
        }
        return acc;
    }
    // Every uniform corner.
    for e in EDGES {
        out.push(vec![*e; arity]);
    }
    // Every argument taken to every edge in turn, against every uniform
    // background. One argument extreme at a time is the shape a width bug
    // takes; a bug needing two arguments extreme *and different* is not
    // covered here and this is where that is admitted.
    for background in EDGES {
        for slot in 0..arity {
            for e in EDGES {
                let mut v = vec![*background; arity];
                v[slot] = *e;
                out.push(v);
            }
        }
    }
    out
}

/// No scalar-only parser unwinds on any decoded value.
#[test]
fn no_parser_panics_on_scalars_the_guest_could_decode() {
    let mut failures: Vec<String> = Vec::new();
    let mut calls = 0usize;
    for target in targets() {
        for args in corpus(target.arity) {
            calls += 1;
            let run = target.run;
            let a = args.clone();
            if catch_unwind(move || run(&a)).is_err() {
                failures.push(format!("{} {:x?}", target.name, args));
                // One line per target is the useful report; the rest of the
                // corpus for a target that already panicked says nothing new.
                break;
            }
        }
    }
    assert!(calls > 10_000, "the corpus collapsed to {calls} calls");
    assert!(
        failures.is_empty(),
        "a decoded guest scalar unwound a parser. Each of these is a call this \
         device would lose, and in release the same input answers wrong rather \
         than trapping:\n{}",
        failures.join("\n")
    );
}

/// The scanned surface, as `module::function`.
///
/// The same trees and files `a_decoder_survives_bytes_the_guest_could_write`
/// reads, filtered the other way: a `pub fn` every one of whose arguments is a
/// plain scalar.
fn declared_scalar_parsers() -> BTreeSet<String> {
    const SCALARS: &[&str] = &[
        "u8", "u16", "u32", "u64", "usize", "i8", "i16", "i32", "i64", "isize", "bool", "f32",
        "f64",
    ];
    const TREES: [&str; 3] = [
        "crates/reims-vgpu/src/runtime/decode",
        "crates/reims-vgpu/src/runtime/icb",
        // `contract` is the decoded API contract itself — pixel formats, pass
        // actions, extents, page geometry. Every argument it takes is a field
        // some decoder has already lifted out of a guest record, and it is
        // backend-independent, so it can be driven on both arms.
        // `iosurface_pages.rs` lives here and was already named below; it stays
        // named there so the two harnesses agree on that file's surface.
        "crates/reims-vgpu/src/contract",
    ];
    // Every guest ordinal that becomes a Vulkan enum passes through here, which
    // is the position `AGENTS.md` calls out as undefined behaviour on the Metal
    // side. Gated on the arm that compiles it, in the scan as well as in the
    // table, so the two agree per arm rather than the table reading short on
    // `backend-metal`.
    #[cfg(feature = "backend-vulkan")]
    const VULKAN_TREES: [&str; 1] = ["crates/reims-vgpu/src/backend/vulkan/translate"];
    #[cfg(not(feature = "backend-vulkan"))]
    const VULKAN_TREES: [&str; 0] = [];
    const FILES: [&str; 2] = [
        "crates/reims-vgpu/src/runtime/heap_query.rs",
        "crates/reims-vgpu/src/runtime/mtlb.rs",
    ];
    let root = source_scan::workspace_root();
    let mut files: Vec<std::path::PathBuf> = TREES
        .iter()
        .chain(VULKAN_TREES.iter())
        .flat_map(|d| source_scan::rust_sources(&root.join(d)))
        .collect();
    files.extend(FILES.iter().map(|f| root.join(f)));

    let mut found = BTreeSet::new();
    for path in files {
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or_default();
        if stem == "tests" || stem.ends_with("_tests") {
            continue;
        }
        let module = if stem == "mod" {
            path.parent()
                .and_then(|p| p.file_name())
                .and_then(|s| s.to_str())
                .unwrap_or_default()
                .to_string()
        } else {
            stem.to_string()
        };
        let Ok(raw) = std::fs::read_to_string(&path) else {
            continue;
        };
        // `blank_test_items` as well as `blank_comments`: several
        // `#[cfg(test)]` fixture builders in `decode::resource` are scalar-only
        // `pub fn`s, and driving one would be driving the test suite's own
        // encoder rather than the device.
        let text = source_scan::blank_test_items(&source_scan::blank_comments(&raw));
        for (leaf, args) in public_fn_signatures(&text) {
            let args = args.trim_start_matches('(');
            let parts: Vec<&str> = args
                .split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .collect();
            if parts.is_empty() {
                continue;
            }
            if parts.iter().all(|p| {
                p.split_once(':')
                    .is_some_and(|(_, ty)| SCALARS.contains(&ty.trim()))
            }) {
                found.insert(format!("{module}::{leaf}"));
            }
        }
    }
    found
}

/// Item-level `pub fn` name and parenthesised argument list, skipping any
/// declaration a `#[cfg(test)]` attribute stands in front of.
///
/// A copy of the byte-slice harness's reader rather than a shared helper,
/// deliberately: the two tests must be able to disagree about the surface if one
/// of them is changed, and a shared parser would make a narrowing in one silent
/// in the other. It has one addition, and it is load-bearing here and not there.
///
/// `source_scan::blank_test_items` blanks a `#[cfg(test)] mod`; it does not
/// blank a `#[cfg(test)] pub fn`. That costs the byte-slice harness nothing —
/// no gated function on that surface takes a `&[u8]` — and it costs this one
/// eleven false entries, because `decode::resource` declares eleven gated
/// `*_icb_layout` fixture builders whose arguments are all `u16`. Driving one
/// would be fuzzing the test suite's own encoder and calling it the device, and
/// they cannot even be named from here: they are configured out of the build
/// this test links against.
fn public_fn_signatures(text: &str) -> Vec<(String, String)> {
    let chars: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut at = 0usize;
    while let Some(rel) = text[at..].find("\npub fn ") {
        let start = at + rel + "\npub fn ".len();
        at = start;
        let leaf: String = text[start..]
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect();
        if leaf.is_empty() {
            continue;
        }
        if gated_by_cfg_test(text, start) {
            continue;
        }
        let Some(open) = text[start..].find('(') else {
            continue;
        };
        let open = start + open;
        let mut depth = 0usize;
        let mut close = open;
        for (i, c) in chars.iter().enumerate().skip(open) {
            match c {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        close = i;
                        break;
                    }
                }
                _ => {}
            }
        }
        if close > open {
            out.push((leaf, text[open + 1..close].to_string()));
        }
    }
    out
}

/// Scalar-only parsers that are **not** driven, and the check that makes each
/// safe to leave out.
///
/// An exemption list is where a harness like this rots, so there is exactly one
/// admissible reason to be here — *a check elsewhere makes the bad input
/// unreachable* — and
/// [`the_exemptions_rest_on_a_check_that_still_runs`] runs that check rather
/// than trusting this sentence.
///
/// All four take a `page_shift: u32`, and all four shift by it. A shift of 64
/// or more is a panic in a debug build and, worse, a *masked* shift in a release
/// one — Rust does not trap it, so the release consequence is an offset into
/// somewhere else rather than a crash. There is no total form: making
/// `page_size_of` answer `0` for an impossible shift only moves the panic to the
/// `page_size - 1` its callers do.
///
/// What makes them safe is that a page shift is not an arbitrary `u32` on any
/// path. It enters this crate once, across the C ABI at
/// `reims_vgpu_qemu_device_create`, and `device::device_create` answers `None`
/// to anything that is not 12 or 14 before a `DeviceState` exists to carry it.
const EXEMPT: &[(&str, &str)] = &[
    (
        "iosurface_pages::page_size_of",
        "`1u64 << page_shift`; unreachable above 63 because device_create \
         refuses every shift but 12 and 14",
    ),
    (
        "iosurface_pages::span_page_count_shift",
        "`(min_size - 1) >> page_shift`; same domain, same constructor",
    ),
    (
        "iosurface_pages::entry_gpa_shift",
        "`(pfn as u64) << page_shift`; same domain, same constructor",
    ),
    (
        "gva::pfn_to_gpa",
        "`(pfn as u64) << page_shift`, the same shift one module down; same \
         domain, same constructor",
    ),
];

/// The exemptions rest on a check that still runs.
///
/// `device_create` is the only way a `page_shift` reaches the rest of this
/// crate, and it is what makes the three functions in [`EXEMPT`] unreachable
/// with a shift they cannot survive. So this drives the same corpus at *it*, and
/// asserts it refuses everything outside the two supported shifts.
///
/// Driven at the **C entry point** rather than at `device::device_create`, which
/// is a private module. That is the better place anyway: it is the boundary the
/// QEMU shim actually crosses, so this exercises the argument check and the
/// create path a real attach takes.
///
/// Only the refusing direction is driven. A successful create registers a device
/// in a process-wide table and would have to be torn down; the two accepted
/// values are asserted to be exactly the two the model declares, which is what
/// makes "everything else refuses" a complete statement about the domain.
#[test]
fn the_exemptions_rest_on_a_check_that_still_runs() {
    use reims_vgpu::qemu::abi::{
        reims_vgpu_qemu_device_create, ReimsVgpuQemuCreateInfo, ReimsVgpuQemuDevice,
        REIMS_VGPU_QEMU_ABI_VERSION, REIMS_VGPU_QEMU_ERR_ARGS,
    };

    assert_eq!(reims_vgpu::contract::gva::PAGE_SHIFT_X86, 12);
    assert_eq!(reims_vgpu::contract::gva::PAGE_SHIFT_ARM64E, 14);

    let create = |shift: u32| -> i32 {
        let info = ReimsVgpuQemuCreateInfo {
            abi_version: REIMS_VGPU_QEMU_ABI_VERSION,
            struct_size: std::mem::size_of::<ReimsVgpuQemuCreateInfo>() as u32,
            host_ops: std::ptr::null(),
            guest_page_shift: shift,
        };
        let mut out = ReimsVgpuQemuDevice {
            abi_version: 0,
            struct_size: 0,
            handle: 0,
        };
        // SAFETY: both pointers are to live locals for the length of the call.
        unsafe { reims_vgpu_qemu_device_create(&info, &mut out) }
    };

    let mut refused = 0usize;
    for shift in EDGES {
        let shift = *shift as u32;
        if shift == 12 || shift == 14 {
            continue;
        }
        assert_eq!(
            create(shift),
            REIMS_VGPU_QEMU_ERR_ARGS,
            "the create entry accepted page_shift {shift}, so the exemptions in \
             EXEMPT no longer hold and those three must be driven or made total"
        );
        refused += 1;
    }
    // Every shift the fuzzer found a panic at, spelled out.
    for shift in [64u32, 65, 0xFF, u32::MAX] {
        assert_eq!(create(shift), REIMS_VGPU_QEMU_ERR_ARGS);
    }
    assert!(refused > 10, "the corpus collapsed to {refused} refusals");
}

/// Whether the item declared at `at` carries a `#[cfg(test)]`, reading back over
/// the doc comments and attributes above it.
///
/// Back over *both*, because an attribute is very often not the line
/// immediately above the declaration: the gated builders in `decode::resource`
/// carry six-line doc comments between `#[cfg(test)]` and `pub fn`.
fn gated_by_cfg_test(text: &str, at: usize) -> bool {
    let head = &text[..at];
    for line in head.lines().rev().skip(1) {
        let line = line.trim();
        if line.starts_with("///") || line.starts_with("//") || line.is_empty() {
            continue;
        }
        if line.starts_with("#[") {
            if line.contains("cfg(test)") {
                return true;
            }
            continue;
        }
        return false;
    }
    false
}

/// Every scalar-only parser on the surface is in the table, and every table
/// entry is still one.
#[test]
fn every_all_scalar_parser_is_driven() {
    let declared = declared_scalar_parsers();
    let driven: BTreeSet<String> = targets().into_iter().map(|t| t.name.to_string()).collect();

    assert!(
        !declared.is_empty(),
        "the scan found no scalar-only `pub fn` at all; it is reading the wrong \
         shape or the wrong tree"
    );

    let exempt: BTreeSet<String> = EXEMPT.iter().map(|(n, _)| (*n).to_string()).collect();
    let covered: BTreeSet<String> = driven.union(&exempt).cloned().collect();

    let missing: Vec<&String> = declared.difference(&covered).collect();
    assert!(
        missing.is_empty(),
        "these compute something out of decoded guest scalars and nothing proves \
         they survive them — add each to `targets()`, or to EXEMPT with the check \
         that makes it unreachable: {missing:?}"
    );

    // An exemption for a function that no longer exists is a sentence nobody
    // will ever re-read, and it hides the fact that the population shrank.
    let ghosts: Vec<&String> = exempt.difference(&declared).collect();
    assert!(
        ghosts.is_empty(),
        "these are exempted and are no longer scalar-only `pub fn`s on the \
         surface: {ghosts:?}"
    );

    let stale: Vec<&String> = driven.difference(&declared).collect();
    assert!(
        stale.is_empty(),
        "these targets name no scalar-only `pub fn` on the scanned surface; the \
         table has drifted from the tree: {stale:?}"
    );
}
