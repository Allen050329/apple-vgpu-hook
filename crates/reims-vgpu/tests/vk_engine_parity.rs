//! Off-VM regression suite for the internal Vulkan engine (no external render crate).
//!
//! Drives `engine::execute_draw_request` with representative inputs and asserts
//! known-correct pixels (metal2vulkan fixtures) plus warm create/alloc = 0 and
//! device-loss policy. Requires a working Vulkan ICD; skips cleanly if init fails.
//!
//! **Serial:** the engine is process-global; all tests take the suite lock.

#![cfg(feature = "backend-vulkan")]

use metal2vulkan::passes::Stage;
use reims_vgpu::backend::vulkan::engine::{
    self, BlendFactor, BlendOp, BlendStateResource, CullMode, DepthState, DrawRequest, IndexType,
    IndexedDrawResource, LoadOp, PrimitiveTopology, SampledContentIdentity, SampledImageResource,
    SampledSource, SamplerCompareFunction, SamplerResource, ScissorResource, SecondaryColorTarget,
    StencilFaceOps, StencilOp, StencilState, StorageBufferResource, TargetIdentity,
    VertexAttributeFormat, VertexAttributeResource, VertexStepFunction, ViewportResource,
    MAX_DEVICE_RECREATES,
};
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

fn engine_test_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| {
        // Never share the live product logs with a concurrent boot.
        reims_vgpu::observe::redirect_logs_for_tests();
        Mutex::new(())
    })
}

fn fixtures() -> PathBuf {
    // Minimal AIR fixture subset owned by reims-vgpu's Vulkan engine tests.
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/air")
}

fn translate_words(name: &str, stage: Stage) -> Vec<u32> {
    let tmp = std::env::temp_dir().join(format!(
        "paravirt_engine_{}_{}_{:?}",
        std::process::id(),
        name,
        stage
    ));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).expect("tmp");
    let path = fixtures().join(name);
    assert!(path.exists(), "missing reims-vgpu AIR fixture: {name}");
    let spv = metal2vulkan::translate(path.to_str().unwrap(), stage, &tmp)
        .unwrap_or_else(|e| panic!("translate {name}: {e}"));
    spv.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn triangle_spirv() -> (Vec<u32>, Vec<u32>) {
    (
        translate_words("render_tri.air", Stage::Vertex),
        translate_words("render_frag.air", Stage::Fragment),
    )
}

fn skip_if_no_gpu(err: &str) -> bool {
    let lower = err.to_ascii_lowercase();
    lower.contains("no vulkan")
        || lower.contains("load vulkan")
        || lower.contains("create_instance")
        || lower.contains("no graphics")
        || lower.contains("vk_engine_init")
}

fn engine_req(vert: &[u32], frag: &[u32], w: u32, h: u32) -> DrawRequest {
    DrawRequest {
        vert_spirv: std::sync::Arc::new(vert.to_vec()),
        frag_spirv: std::sync::Arc::new(frag.to_vec()),
        width: w,
        height: h,
        vertex_count: 3,
        flip_viewport_y: true,
        first_vertex: 0,
        instance_count: Some(1),
        base_instance: 0,
        primitive_topology: PrimitiveTopology::Triangle,
        ..Default::default()
    }
}

/// float4(0.25, 0.5, 0.75, 1.0) → unorm8 ≈ (64, 128, 191, 255); allow ±1 LSB.
fn near(got: u8, want: u8) -> bool {
    (got as i32 - want as i32).abs() <= 1
}

fn assert_fullscreen_fragment_color(label: &str, px: &[u8], w: u32, h: u32) {
    assert_eq!(px.len(), (w * h * 4) as usize, "{label}: readback size");
    let i = ((h / 2) * w + w / 2) as usize * 4;
    let (r, g, b, a) = (px[i], px[i + 1], px[i + 2], px[i + 3]);
    assert!(
        near(r, 64) && near(g, 128) && near(b, 191) && near(a, 255),
        "{label}: center RGBA=({r},{g},{b},{a}); expected ~(64,128,191,255)"
    );
    let all = (0..(w * h) as usize).all(|p| near(px[p * 4], 64) && near(px[p * 4 + 1], 128));
    assert!(
        all,
        "{label}: fullscreen triangle did not cover viewport (clear showing through)"
    );
}

fn draw_or_skip(label: &str, req: &DrawRequest) -> Option<Vec<u8>> {
    match engine::execute_draw_request(req) {
        Ok(o) => Some(o.pixels),
        Err(e) => {
            let s = e.to_string();
            if skip_if_no_gpu(&s) {
                eprintln!("SKIP {label}: no GPU ({s})");
                None
            } else {
                panic!("{label}: {s}");
            }
        }
    }
}

#[test]
fn plain_triangle_known_color() {
    let _g = engine_test_lock().lock().unwrap_or_else(|e| e.into_inner());
    engine::test_reset_engine();
    let (v, f) = triangle_spirv();
    let req = engine_req(&v, &f, 16, 16);
    if let Some(px) = draw_or_skip("plain_triangle", &req) {
        assert_fullscreen_fragment_color("plain_triangle", &px, 16, 16);
    }
}

#[test]
fn viewport_scissor_known_color() {
    let _g = engine_test_lock().lock().unwrap_or_else(|e| e.into_inner());
    engine::test_reset_engine();
    let (v, f) = triangle_spirv();
    let mut req = engine_req(&v, &f, 32, 32);
    req.viewports.push(ViewportResource {
        x: 0.0,
        y: 0.0,
        width: 32.0,
        height: 32.0,
        min_depth: 0.0,
        max_depth: 1.0,
    });
    req.scissors.push(ScissorResource {
        x: 0,
        y: 0,
        width: 32,
        height: 32,
    });
    if let Some(px) = draw_or_skip("viewport_scissor", &req) {
        assert_fullscreen_fragment_color("viewport_scissor", &px, 32, 32);
    }
}

/// True when the fixture triangle covered the target center (fragment color),
/// false when the clear black shows through (the triangle was culled).
fn triangle_covered(px: &[u8], w: u32, h: u32) -> bool {
    let i = ((h / 2) * w + w / 2) as usize * 4;
    // Fragment is ~(64,128,191); clear is (0,0,0). Green channel discriminates.
    px[i + 1] > 32
}

/// Face culling is honored by the Vulkan raster state and is wired correctly
/// through the Metal winding + Y-flip. On-GPU behavioral checks (no guest 3D
/// needed): the whole assertion set is a truth table that only holds if cull is
/// actually applied AND the front-facing winding selects the right face.
#[test]
fn cull_mode_honored_and_winding_correct() {
    let _g = engine_test_lock().lock().unwrap_or_else(|e| e.into_inner());
    engine::test_reset_engine();
    let (v, f) = triangle_spirv();
    let (w, h) = (16u32, 16u32);

    let variant = |cull: CullMode, ccw: bool| -> Option<bool> {
        let mut req = engine_req(&v, &f, w, h);
        req.cull_mode = cull;
        req.front_face_ccw = ccw;
        draw_or_skip("cull", &req).map(|px| triangle_covered(&px, w, h))
    };

    // cull=None must stay byte-identical to the no-cull path: full coverage.
    let Some(none_cov) = variant(CullMode::None, false) else {
        return; // no GPU
    };
    assert!(
        none_cov,
        "cull=None must draw both faces (fullscreen coverage)"
    );

    let back_cw = variant(CullMode::Back, false).unwrap();
    let front_cw = variant(CullMode::Front, false).unwrap();
    let back_ccw = variant(CullMode::Back, true).unwrap();
    let front_ccw = variant(CullMode::Front, true).unwrap();

    // A single triangle presents one face to the viewer: culling Front and Back
    // are complementary — exactly one keeps it. If cull were ignored, both would
    // stay covered and this fails.
    assert_ne!(
        back_cw, front_cw,
        "Front/Back cull must be complementary for one triangle (cull not applied?)"
    );
    // Flipping the front-facing winding swaps which face is front, i.e. swaps the
    // effect of Front vs Back. If winding were ignored, back_ccw would equal
    // back_cw instead.
    assert_eq!(
        back_ccw, front_cw,
        "flipping winding must swap Back into Front behavior (winding not wired?)"
    );
    assert_eq!(
        front_ccw, back_cw,
        "flipping winding must swap Front into Back behavior (winding not wired?)"
    );
}

/// Depth test is honored end to end: a transient depth buffer is attached, the
/// compare op + clear value are wired, and the 2D path (`depth: None`) stays
/// byte-identical (proven by every other test here running with no depth).
///
/// Vehicle: a full-screen textured quad whose per-vertex z is fed via storage
/// buffer 0 (so depth is controllable), sampling one solid color. Assertions are
/// mostly convention-independent RELATIONSHIPS (depth applied, compare matters,
/// clear matters) plus the absolute Never/Always anchors — so the test proves
/// the wiring without depending on the exact depth compare operand order.
#[test]
fn depth_test_honored_compare_and_clear_wired() {
    let _g = engine_test_lock().lock().unwrap_or_else(|e| e.into_inner());
    engine::test_reset_engine();
    let vert = translate_words("textured_quad.air", Stage::Vertex);
    let frag = translate_words("textured_quad.air", Stage::Fragment);
    let (w, h) = (16u32, 16u32);
    let encode_f32 = |values: &[f32]| {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>()
    };
    // Full-screen quad, all vertices at NDC z = 0.5.
    let quad_z = |z: f32| -> [[f32; 4]; 6] {
        [
            [-1.0, -1.0, z, 1.0],
            [1.0, -1.0, z, 1.0],
            [-1.0, 1.0, z, 1.0],
            [-1.0, 1.0, z, 1.0],
            [1.0, -1.0, z, 1.0],
            [1.0, 1.0, z, 1.0],
        ]
    };
    let uvs: [[f32; 2]; 6] = [
        [0.0, 1.0],
        [1.0, 1.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [1.0, 1.0],
        [1.0, 0.0],
    ];
    // Sampled color with a strong green channel so `triangle_covered` (green>32)
    // discriminates covered vs cleared-black.
    let rgba = [17u8, 140, 203, 255];

    // Returns Some(covered) for a fragment at z=0.5 with the given depth state.
    let variant = |compare: SamplerCompareFunction, clear: f32| -> Option<bool> {
        let mut req = engine_req(&vert, &frag, w, h);
        req.vertex_count = 6;
        req.load_op = Some(LoadOp::Clear([0.0, 0.0, 0.0, 0.0]));
        req.storage_buffers.push(StorageBufferResource {
            binding: 0,
            content: encode_f32(&quad_z(0.5).into_iter().flatten().collect::<Vec<_>>()).into(),
        });
        req.storage_buffers.push(StorageBufferResource {
            binding: 1,
            content: encode_f32(&uvs.into_iter().flatten().collect::<Vec<_>>()).into(),
        });
        req.sampled_images.push(SampledImageResource {
            binding: 32,
            width: 2,
            height: 2,
            layers: 1,
            arrayed: false,
            volume: false,
            cube: false,
            one_dim: false,
            source: SampledSource::Bytes(std::sync::Arc::new(rgba.repeat(4))),
            format: ash::vk::Format::R8G8B8A8_UNORM,
            identity: None,
            swizzle: Default::default(),
        });
        req.samplers.push(SamplerResource::normalized_default(64));
        req.depth = Some(DepthState {
            test_enable: true,
            write_enable: true,
            compare,
            clear_value: clear,
            load: false,
            stencil: None,
        });
        draw_or_skip("depth", &req).map(|px| triangle_covered(&px, w, h))
    };

    // Absolute anchors (convention-independent): Never never draws, Always always
    // draws. If depth state were ignored, Never would still cover.
    let Some(never) = variant(SamplerCompareFunction::Never, 1.0) else {
        return; // no GPU
    };
    assert!(!never, "compare=Never must discard every fragment");
    assert!(
        variant(SamplerCompareFunction::Always, 1.0).unwrap(),
        "compare=Always must keep every fragment"
    );

    // Relationships (convention-independent): with fragment z=0.5,
    let less_hi = variant(SamplerCompareFunction::Less, 1.0).unwrap();
    let less_lo = variant(SamplerCompareFunction::Less, 0.0).unwrap();
    let greater_lo = variant(SamplerCompareFunction::Greater, 0.0).unwrap();
    let greater_hi = variant(SamplerCompareFunction::Greater, 1.0).unwrap();
    // The compare op matters: Less vs Greater against the same reference differ.
    assert_ne!(
        less_hi, greater_hi,
        "Less vs Greater against clear=1.0 must differ (compare op not applied?)"
    );
    // The clear value matters: same op, different reference → different result.
    assert_ne!(
        less_hi, less_lo,
        "clear value must feed the depth reference (Less@1.0 vs Less@0.0)"
    );
    // Consistency: flipping BOTH the op and the reference gives the same outcome.
    assert_eq!(
        less_hi, greater_lo,
        "Less@1.0 and Greater@0.0 must agree (depth compare wired consistently)"
    );
    assert_eq!(less_lo, greater_hi, "Less@0.0 and Greater@1.0 must agree");
}

/// Same depth wiring proof as `depth_test_honored_compare_and_clear_wired`, but
/// through the RESIDENT target path — the product Store path (`target_identity`
/// with `skip_readback` and `read_target`, which builds its own ad-hoc [color,depth]
/// framebuffer in the registry_ensure branch (exec.rs) separate from the pooled
/// path exercised above. Without this, the resident depth branch was reachable
/// only in production; a dispose-order or framebuffer bug there (the exact class
/// that caused the MRT/depth device-lost fixes) would surface as a device loss
/// with no test to catch it. Uses BGRA output like a real Surface target; the
/// green channel discriminator survives the R/B swap (index 1 unchanged).
#[test]
fn depth_test_honored_on_resident_target_path() {
    let _g = engine_test_lock().lock().unwrap_or_else(|e| e.into_inner());
    engine::test_reset_engine();
    let vert = translate_words("textured_quad.air", Stage::Vertex);
    let frag = translate_words("textured_quad.air", Stage::Fragment);
    let (w, h) = (16u32, 16u32);
    let encode_f32 = |values: &[f32]| {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>()
    };
    let quad_z = |z: f32| -> [[f32; 4]; 6] {
        [
            [-1.0, -1.0, z, 1.0],
            [1.0, -1.0, z, 1.0],
            [-1.0, 1.0, z, 1.0],
            [-1.0, 1.0, z, 1.0],
            [1.0, -1.0, z, 1.0],
            [1.0, 1.0, z, 1.0],
        ]
    };
    let uvs: [[f32; 2]; 6] = [
        [0.0, 1.0],
        [1.0, 1.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [1.0, 1.0],
        [1.0, 0.0],
    ];
    let rgba = [17u8, 140, 203, 255];

    // Each variant renders to a FRESH resident surface (distinct id) so no stale
    // content leaks between variants, then reads it back via the product path.
    let mut surface_id = 900u32;
    let mut variant = |compare: SamplerCompareFunction, clear: f32| -> Option<bool> {
        surface_id += 1;
        let identity = TargetIdentity::Surface {
            id: surface_id,
            width: w,
            height: h,
            generation: 1,
        };
        let mut req = engine_req(&vert, &frag, w, h);
        req.vertex_count = 6;
        req.target_identity = Some(identity.clone());
        req.output_bgra = true;
        req.skip_readback = true;
        req.load_op = Some(LoadOp::Clear([0.0, 0.0, 0.0, 0.0]));
        req.storage_buffers.push(StorageBufferResource {
            binding: 0,
            content: encode_f32(&quad_z(0.5).into_iter().flatten().collect::<Vec<_>>()).into(),
        });
        req.storage_buffers.push(StorageBufferResource {
            binding: 1,
            content: encode_f32(&uvs.into_iter().flatten().collect::<Vec<_>>()).into(),
        });
        req.sampled_images.push(SampledImageResource {
            binding: 32,
            width: 2,
            height: 2,
            layers: 1,
            arrayed: false,
            volume: false,
            cube: false,
            one_dim: false,
            source: SampledSource::Bytes(std::sync::Arc::new(rgba.repeat(4))),
            format: ash::vk::Format::R8G8B8A8_UNORM,
            identity: None,
            swizzle: Default::default(),
        });
        req.samplers.push(SamplerResource::normalized_default(64));
        req.depth = Some(DepthState {
            test_enable: true,
            write_enable: true,
            compare,
            clear_value: clear,
            load: false,
            stencil: None,
        });
        match engine::execute_draw_request(&req) {
            Ok(_) => {}
            Err(e) if skip_if_no_gpu(&e.to_string()) => {
                eprintln!("SKIP resident depth: {e}");
                return None;
            }
            Err(e) => panic!("resident depth draw: {e}"),
        }
        let px = engine::read_target(&identity).expect("read resident depth target");
        Some(triangle_covered(&px, w, h))
    };

    let Some(never) = variant(SamplerCompareFunction::Never, 1.0) else {
        return; // no GPU
    };
    assert!(
        !never,
        "resident: compare=Never must discard every fragment"
    );
    assert!(
        variant(SamplerCompareFunction::Always, 1.0).unwrap(),
        "resident: compare=Always must keep every fragment"
    );

    let less_hi = variant(SamplerCompareFunction::Less, 1.0).unwrap();
    let greater_hi = variant(SamplerCompareFunction::Greater, 1.0).unwrap();
    let greater_lo = variant(SamplerCompareFunction::Greater, 0.0).unwrap();
    let less_lo = variant(SamplerCompareFunction::Less, 0.0).unwrap();
    assert_ne!(
        less_hi, greater_hi,
        "resident: Less vs Greater against clear=1.0 must differ"
    );
    assert_ne!(
        less_hi, less_lo,
        "resident: clear value must feed the depth reference"
    );
    assert_eq!(
        less_hi, greater_lo,
        "resident: Less@1.0 and Greater@0.0 must agree"
    );
}

/// Proves the Vulkan stencil test is wired end-to-end: a single full-screen quad
/// with depth compare Always (depth never gates) and a stencil face whose
/// compare/reference/read-mask decide coverage against the transient stencil
/// buffer's clear value. Mirrors the depth proof's single-draw structure — the
/// transient depth-stencil is per-draw CLEAR-only, so this covers the compare
/// path (enable + compareOp + reference + compareMask + stencil clear); the
/// stencil *ops* (fail/pass/depthFail, write_mask) need a persistent buffer to
/// observe and are the documented follow-up gap.
#[test]
fn stencil_test_honored_compare_ref_and_clear_wired() {
    let _g = engine_test_lock().lock().unwrap_or_else(|e| e.into_inner());
    engine::test_reset_engine();
    let vert = translate_words("textured_quad.air", Stage::Vertex);
    let frag = translate_words("textured_quad.air", Stage::Fragment);
    let (w, h) = (16u32, 16u32);
    let encode_f32 = |values: &[f32]| {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>()
    };
    let quad_z = |z: f32| -> [[f32; 4]; 6] {
        [
            [-1.0, -1.0, z, 1.0],
            [1.0, -1.0, z, 1.0],
            [-1.0, 1.0, z, 1.0],
            [-1.0, 1.0, z, 1.0],
            [1.0, -1.0, z, 1.0],
            [1.0, 1.0, z, 1.0],
        ]
    };
    let uvs: [[f32; 2]; 6] = [
        [0.0, 1.0],
        [1.0, 1.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [1.0, 1.0],
        [1.0, 0.0],
    ];
    let rgba = [17u8, 140, 203, 255];

    // Both faces identical (the quad is one winding); depth compare Always so
    // only the stencil test gates. `read_mask` masks both operands.
    let variant = |compare: SamplerCompareFunction,
                   reference: u32,
                   clear: u32,
                   read_mask: u32|
     -> Option<bool> {
        let face = StencilFaceOps {
            compare,
            fail_op: StencilOp::Keep,
            depth_fail_op: StencilOp::Keep,
            pass_op: StencilOp::Keep,
            read_mask,
            write_mask: 0,
        };
        let mut req = engine_req(&vert, &frag, w, h);
        req.vertex_count = 6;
        req.load_op = Some(LoadOp::Clear([0.0, 0.0, 0.0, 0.0]));
        req.storage_buffers.push(StorageBufferResource {
            binding: 0,
            content: encode_f32(&quad_z(0.5).into_iter().flatten().collect::<Vec<_>>()).into(),
        });
        req.storage_buffers.push(StorageBufferResource {
            binding: 1,
            content: encode_f32(&uvs.into_iter().flatten().collect::<Vec<_>>()).into(),
        });
        req.sampled_images.push(SampledImageResource {
            binding: 32,
            width: 2,
            height: 2,
            layers: 1,
            arrayed: false,
            volume: false,
            cube: false,
            one_dim: false,
            source: SampledSource::Bytes(std::sync::Arc::new(rgba.repeat(4))),
            format: ash::vk::Format::R8G8B8A8_UNORM,
            identity: None,
            swizzle: Default::default(),
        });
        req.samplers.push(SamplerResource::normalized_default(64));
        req.depth = Some(DepthState {
            test_enable: true,
            write_enable: false,
            compare: SamplerCompareFunction::Always,
            clear_value: 1.0,
            load: false,
            stencil: Some(StencilState {
                front: face,
                back: face,
                reference_front: reference,
                reference_back: reference,
                clear_value: clear,
            }),
        });
        draw_or_skip("stencil", &req).map(|px| triangle_covered(&px, w, h))
    };

    // Absolute anchors: Always keeps every fragment, Never discards every one —
    // independent of reference/clear. If stencil state were ignored, Never would
    // still cover (the color-only pipeline always draws).
    let Some(never) = variant(SamplerCompareFunction::Never, 0, 0, 0xFF) else {
        return; // no GPU
    };
    assert!(!never, "stencil compare=Never must discard every fragment");
    assert!(
        variant(SamplerCompareFunction::Always, 0, 0, 0xFF).unwrap(),
        "stencil compare=Always must keep every fragment"
    );

    // Equal: coverage iff (reference & mask) == (clearValue & mask).
    assert!(
        variant(SamplerCompareFunction::Equal, 0, 0, 0xFF).unwrap(),
        "Equal with reference==clear must keep every fragment"
    );
    assert!(
        !variant(SamplerCompareFunction::Equal, 1, 0, 0xFF).unwrap(),
        "Equal with reference!=clear must discard (reference/clear wired?)"
    );
    // read_mask (compareMask) masks both operands: with mask 0 the differing
    // low bit is erased so Equal passes again — proves the mask is applied.
    assert!(
        variant(SamplerCompareFunction::Equal, 1, 0, 0x00).unwrap(),
        "read_mask=0 must erase the reference/clear difference (compareMask wired?)"
    );
}

#[test]
fn load_seed_preserves_uncovered_and_draws() {
    let _g = engine_test_lock().lock().unwrap_or_else(|e| e.into_inner());
    engine::test_reset_engine();
    let (v, f) = triangle_spirv();
    let mut req = engine_req(&v, &f, 8, 8);
    // Fullscreen triangle covers everything; seed is Load base then overdrawn.
    req.target_rgba8 = Some([10, 20, 30, 255].repeat(8 * 8));
    if let Some(px) = draw_or_skip("load_seed", &req) {
        assert_fullscreen_fragment_color("load_seed", &px, 8, 8);
    }
}

#[test]
fn blend_src_alpha_known_color() {
    let _g = engine_test_lock().lock().unwrap_or_else(|e| e.into_inner());
    engine::test_reset_engine();
    let (v, f) = triangle_spirv();
    let mut req = engine_req(&v, &f, 8, 8);
    req.target_rgba8 = Some([0, 0, 0, 255].repeat(8 * 8));
    req.blend = Some(BlendStateResource {
        src_color: BlendFactor::SrcAlpha,
        dst_color: BlendFactor::OneMinusSrcAlpha,
        color_op: BlendOp::Add,
        src_alpha: BlendFactor::One,
        dst_alpha: BlendFactor::OneMinusSrcAlpha,
        alpha_op: BlendOp::Add,
        constants: [0.0; 4],
    });
    if let Some(px) = draw_or_skip("blend_src_alpha", &req) {
        // Fragment alpha=1 → same as replace over black seed.
        assert_fullscreen_fragment_color("blend_src_alpha", &px, 8, 8);
    }
}

#[test]
fn indexed_u16_known_color() {
    let _g = engine_test_lock().lock().unwrap_or_else(|e| e.into_inner());
    engine::test_reset_engine();
    let (v, f) = triangle_spirv();
    let mut req = engine_req(&v, &f, 16, 16);
    req.indexed = Some(IndexedDrawResource {
        index_type: IndexType::U16,
        index_count: 3,
        vertex_offset: 0,
        indices: {
            let mut b = Vec::new();
            for i in [0u16, 1, 2] {
                b.extend_from_slice(&i.to_le_bytes());
            }
            b
        },
    });
    if let Some(px) = draw_or_skip("indexed_u16", &req) {
        assert_fullscreen_fragment_color("indexed_u16", &px, 16, 16);
    }
}

#[test]
fn storage_buffer_binding_still_renders() {
    let _g = engine_test_lock().lock().unwrap_or_else(|e| e.into_inner());
    engine::test_reset_engine();
    let (v, f) = triangle_spirv();
    let mut req = engine_req(&v, &f, 8, 8);
    req.storage_buffers.push(StorageBufferResource {
        binding: 0,
        content: vec![0u8; 64].into(),
    });
    match engine::execute_draw_request(&req) {
        Ok(o) => assert_fullscreen_fragment_color("storage", &o.pixels, 8, 8),
        Err(e) if skip_if_no_gpu(&e.to_string()) => eprintln!("SKIP storage: {e}"),
        Err(e) => {
            // Unused binding may fail pipeline create on some SPIR-V/ICD combos — named only.
            let s = e.to_string();
            assert!(
                s.contains("vk_engine") || s.contains("pipeline") || s.contains("shader"),
                "unexpected storage path error: {s}"
            );
            eprintln!("storage path named failure (ok): {s}");
        }
    }
}

#[test]
fn sampled_and_sampler_still_renders() {
    let _g = engine_test_lock().lock().unwrap_or_else(|e| e.into_inner());
    engine::test_reset_engine();
    let (v, f) = triangle_spirv();
    let mut req = engine_req(&v, &f, 8, 8);
    req.sampled_images.push(SampledImageResource {
        binding: 1,
        width: 2,
        height: 2,
        layers: 1,
        arrayed: false,
        volume: false,
        cube: false,
        one_dim: false,
        source: SampledSource::Bytes(std::sync::Arc::new(vec![
            255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 0, 255,
        ])),
        format: ash::vk::Format::R8G8B8A8_UNORM,
        identity: None,
        swizzle: Default::default(),
    });
    req.samplers.push(SamplerResource::normalized_default(2));
    match engine::execute_draw_request(&req) {
        Ok(o) => {
            assert_fullscreen_fragment_color("sampled", &o.pixels, 8, 8);
            let warm_before = engine::counter_snapshot();
            let warm = engine::execute_draw_request(&req).expect("exact sampled cache hit");
            assert_eq!(warm.pixels, o.pixels, "cache hit must preserve draw bytes");
            let warm_delta = engine::counter_snapshot().delta_since(&warm_before);
            assert_eq!(
                warm_delta.sampled_cache_hits, 1,
                "hit proxy: {warm_delta:?}"
            );
            assert_eq!(
                warm_delta.sampled_cache_hit_bytes, 16,
                "hit-byte proxy: {warm_delta:?}"
            );
            assert_eq!(
                warm_delta.sampled_cache_misses, 0,
                "hit proxy: {warm_delta:?}"
            );
            assert_eq!(warm_delta.sampled_reuploads, 0, "no upload: {warm_delta:?}");
            assert_eq!(
                warm_delta.sampled_reupload_bytes, 0,
                "no upload bytes: {warm_delta:?}"
            );

            let changed_len = {
                let SampledSource::Bytes(bytes) = &mut req.sampled_images[0].source else {
                    unreachable!()
                };
                std::sync::Arc::make_mut(bytes)[0] ^= 0xff;
                bytes.len() as u64
            };
            let changed_before = engine::counter_snapshot();
            let changed = engine::execute_draw_request(&req).expect("changed sampled upload");
            assert_eq!(
                changed.pixels, o.pixels,
                "test shader remains a solid-color oracle"
            );
            let changed_delta = engine::counter_snapshot().delta_since(&changed_before);
            assert_eq!(
                changed_delta.sampled_cache_hits, 0,
                "miss proxy: {changed_delta:?}"
            );
            assert_eq!(
                changed_delta.sampled_cache_misses, 1,
                "miss proxy: {changed_delta:?}"
            );
            assert_eq!(
                changed_delta.sampled_reuploads, 1,
                "upload: {changed_delta:?}"
            );
            assert_eq!(
                changed_delta.sampled_reupload_bytes, changed_len,
                "upload-byte proxy: {changed_delta:?}"
            );
        }
        Err(e) if skip_if_no_gpu(&e.to_string()) => eprintln!("SKIP sampled: {e}"),
        Err(e) => {
            let s = e.to_string();
            assert!(
                s.contains("vk_engine") || s.contains("pipeline") || s.contains("shader"),
                "unexpected sampled path error: {s}"
            );
            eprintln!("sampled path named failure (ok): {s}");
        }
    }
}

/// A resident type-11 sample stays on the GPU: no source readback, staging
/// upload, or temporary sampled image. The tracked layout must still permit a
/// later LoadFromTarget draw on the source identity.
#[test]
fn resident_sample_bind_avoids_roundtrip_and_remains_loadable() {
    let _g = engine_test_lock().lock().unwrap_or_else(|e| e.into_inner());
    engine::test_reset_engine();
    let (v, f) = triangle_spirv();
    let source = TargetIdentity::Surface {
        id: 0x51,
        width: 16,
        height: 16,
        generation: 1,
    };

    let mut make_source = engine_req(&v, &f, 16, 16);
    make_source.target_identity = Some(source.clone());
    match engine::execute_draw_request(&make_source) {
        Ok(o) => assert_fullscreen_fragment_color("resident_sample_source", &o.pixels, 16, 16),
        Err(e) if skip_if_no_gpu(&e.to_string()) => {
            eprintln!("SKIP resident_sample_bind: {e}");
            return;
        }
        Err(e) => panic!("resident source: {e}"),
    }

    let mut consume = engine_req(&v, &f, 16, 16);
    consume.sampled_images.push(SampledImageResource {
        binding: 1,
        width: 16,
        height: 16,
        layers: 1,
        arrayed: false,
        volume: false,
        cube: false,
        one_dim: false,
        source: SampledSource::Target(source.clone()),
        format: ash::vk::Format::R8G8B8A8_UNORM,
        identity: None,
        swizzle: Default::default(),
    });
    engine::reset_draw_counters();
    let before = engine::counter_snapshot();
    let consumed = engine::execute_draw_request(&consume).expect("bind resident sample");
    assert_fullscreen_fragment_color("resident_sample_consumer", &consumed.pixels, 16, 16);
    let delta = engine::counter_snapshot().delta_since(&before);
    assert_eq!(delta.sampled_gpu_binds, 1, "direct-bind proxy: {delta:?}");
    assert_eq!(delta.sampled_reuploads, 0, "no sampled reupload: {delta:?}");
    assert_eq!(
        delta.readbacks, 1,
        "only the consumer target may read back: {delta:?}"
    );

    let mut load_again = engine_req(&v, &f, 16, 16);
    load_again.target_identity = Some(source.clone());
    load_again.load_op = Some(LoadOp::LoadFromTarget);
    let loaded = engine::execute_draw_request(&load_again).expect("load after direct sample");
    assert_fullscreen_fragment_color("resident_sample_reloaded", &loaded.pixels, 16, 16);
}

/// Attachment feedback is represented as a GPU snapshot, matching Metal's
/// prior-content sampling without binding one image for read and write at once.
#[test]
fn resident_sample_alias_uses_gpu_snapshot_without_roundtrip() {
    let _g = engine_test_lock().lock().unwrap_or_else(|e| e.into_inner());
    engine::test_reset_engine();
    let (v, f) = triangle_spirv();
    let identity = TargetIdentity::Surface {
        id: 0x52,
        width: 16,
        height: 16,
        generation: 1,
    };
    let mut cold = engine_req(&v, &f, 16, 16);
    cold.target_identity = Some(identity.clone());
    match engine::execute_draw_request(&cold) {
        Ok(_) => {}
        Err(e) if skip_if_no_gpu(&e.to_string()) => {
            eprintln!("SKIP resident_sample_alias: {e}");
            return;
        }
        Err(e) => panic!("resident alias source: {e}"),
    }

    let mut alias = engine_req(&v, &f, 16, 16);
    alias.target_identity = Some(identity.clone());
    alias.load_op = Some(LoadOp::LoadFromTarget);
    alias.sampled_images.push(SampledImageResource {
        binding: 1,
        width: 16,
        height: 16,
        layers: 1,
        arrayed: false,
        volume: false,
        cube: false,
        one_dim: false,
        source: SampledSource::Target(identity),
        format: ash::vk::Format::R8G8B8A8_UNORM,
        identity: None,
        swizzle: Default::default(),
    });
    engine::reset_draw_counters();
    let before = engine::counter_snapshot();
    let out = engine::execute_draw_request(&alias).expect("resident alias GPU snapshot");
    assert_fullscreen_fragment_color("resident_sample_alias", &out.pixels, 16, 16);
    let delta = engine::counter_snapshot().delta_since(&before);
    assert_eq!(delta.sampled_gpu_binds, 1, "GPU snapshot proxy: {delta:?}");
    assert_eq!(delta.sampled_reuploads, 0, "no host reupload: {delta:?}");
    assert_eq!(delta.readbacks, 1, "only target readback: {delta:?}");
}

#[test]
fn vertex_float2_attr_still_renders() {
    let _g = engine_test_lock().lock().unwrap_or_else(|e| e.into_inner());
    engine::test_reset_engine();
    let (v, f) = triangle_spirv();
    let mut req = engine_req(&v, &f, 8, 8);
    req.vertex_attributes.push(VertexAttributeResource {
        location: 0,
        binding: 0,
        format: VertexAttributeFormat::Float2,
        offset: 0,
        stride: 8,
        step_function: VertexStepFunction::PerVertex,
        step_rate: 1,
        content: vec![0u8; 24].into(),
    });
    match engine::execute_draw_request(&req) {
        Ok(o) => assert_fullscreen_fragment_color("attr", &o.pixels, 8, 8),
        Err(e) if skip_if_no_gpu(&e.to_string()) => eprintln!("SKIP attr: {e}"),
        Err(e) => {
            let s = e.to_string();
            assert!(
                s.contains("vk_engine") || s.contains("pipeline") || s.contains("shader"),
                "unexpected attr path error: {s}"
            );
            eprintln!("attr path named failure (ok): {s}");
        }
    }
}

/// Identity-keyed sampled rebind: same producer key+generation binds the
/// retained image without hashing/comparing content; a bumped generation with
/// changed bytes falls back to the content-addressed path (miss + reupload).
#[test]
fn sampled_identity_fast_path_skips_content_compare() {
    let _g = engine_test_lock().lock().unwrap_or_else(|e| e.into_inner());
    engine::test_reset_engine();
    let (v, f) = triangle_spirv();
    let mut req = engine_req(&v, &f, 8, 8);
    req.sampled_images.push(SampledImageResource {
        binding: 1,
        width: 2,
        height: 2,
        layers: 1,
        arrayed: false,
        volume: false,
        cube: false,
        one_dim: false,
        source: SampledSource::Bytes(std::sync::Arc::new(vec![
            255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 0, 255,
        ])),
        format: ash::vk::Format::R8G8B8A8_UNORM,
        identity: Some(SampledContentIdentity {
            key: 0x1234_5000,
            generation: 1,
        }),
        swizzle: Default::default(),
    });
    req.samplers.push(SamplerResource::normalized_default(2));
    match engine::execute_draw_request(&req) {
        Ok(_) => {}
        Err(e) if skip_if_no_gpu(&e.to_string()) => {
            eprintln!("SKIP sampled identity: {e}");
            return;
        }
        Err(e) => panic!("cold identity draw: {e}"),
    }

    // Same identity + generation: identity hit, no content hash/compare
    // accounting (cache_hit_bytes stays zero), no reupload.
    let warm_before = engine::counter_snapshot();
    engine::execute_draw_request(&req).expect("identity rebind");
    let d = engine::counter_snapshot().delta_since(&warm_before);
    assert_eq!(d.sampled_identity_hits, 1, "identity hit: {d:?}");
    assert_eq!(d.sampled_cache_hits, 0, "no content hit: {d:?}");
    assert_eq!(d.sampled_cache_hit_bytes, 0, "no content bytes: {d:?}");
    assert_eq!(d.sampled_reuploads, 0, "no upload: {d:?}");

    // Bumped generation + changed bytes: identity misses, content path
    // misses, upload happens, and the NEW generation is adopted.
    {
        let img = &mut req.sampled_images[0];
        img.source = SampledSource::Bytes(std::sync::Arc::new(vec![
            1, 0, 0, 255, 0, 1, 0, 255, 0, 0, 1, 255, 1, 1, 0, 255,
        ]));
        img.identity = Some(SampledContentIdentity {
            key: 0x1234_5000,
            generation: 2,
        });
    }
    let changed_before = engine::counter_snapshot();
    engine::execute_draw_request(&req).expect("gen bump upload");
    let d = engine::counter_snapshot().delta_since(&changed_before);
    assert_eq!(d.sampled_identity_hits, 0, "gen bump no identity: {d:?}");
    assert_eq!(d.sampled_cache_misses, 1, "gen bump miss: {d:?}");
    assert_eq!(d.sampled_reuploads, 1, "gen bump upload: {d:?}");

    // The new generation now identity-hits.
    let settle_before = engine::counter_snapshot();
    engine::execute_draw_request(&req).expect("settled identity rebind");
    let d = engine::counter_snapshot().delta_since(&settle_before);
    assert_eq!(d.sampled_identity_hits, 1, "settled identity: {d:?}");
    assert_eq!(d.sampled_reuploads, 0, "settled no upload: {d:?}");
}

#[test]
fn warm_identical_draw_zero_creates_and_allocs() {
    let _g = engine_test_lock().lock().unwrap_or_else(|e| e.into_inner());
    engine::test_reset_engine();
    let (v, f) = triangle_spirv();
    let req = engine_req(&v, &f, 16, 16);
    match engine::execute_draw_request(&req) {
        Ok(_) => {}
        Err(e) if skip_if_no_gpu(&e.to_string()) => {
            eprintln!("SKIP warm: {e}");
            return;
        }
        Err(e) => panic!("cold draw: {e}"),
    }
    engine::execute_draw_request(&req).expect("warm-up draw");
    engine::reset_draw_counters();
    let before = engine::counter_snapshot();
    engine::execute_draw_request(&req).expect("warm draw");
    let after = engine::counter_snapshot();
    let d = after.delta_since(&before);
    assert_eq!(
        d.creates, 0,
        "warm draw must perform zero vkCreate* (got creates={d:?})"
    );
    assert_eq!(
        d.allocs, 0,
        "warm draw must perform zero vkAllocateMemory (got allocs={d:?})"
    );
    assert!(
        d.shader_hits + d.layout_hits + d.pass_hits + d.pipeline_hits > 0,
        "expected cache hits on warm path, got {d:?}"
    );
}

#[test]
fn warm_draw_byte_identical_hot_cache() {
    let _g = engine_test_lock().lock().unwrap_or_else(|e| e.into_inner());
    engine::test_reset_engine();
    let (v, f) = triangle_spirv();
    let req = engine_req(&v, &f, 16, 16);
    let first = match engine::execute_draw_request(&req) {
        Ok(o) => o.pixels,
        Err(e) if skip_if_no_gpu(&e.to_string()) => {
            eprintln!("SKIP hot: {e}");
            return;
        }
        Err(e) => panic!("{e}"),
    };
    assert_fullscreen_fragment_color("hot_first", &first, 16, 16);
    for n in 1..=8 {
        let px = engine::execute_draw_request(&req)
            .unwrap_or_else(|e| panic!("hot #{n}: {e}"))
            .pixels;
        assert_eq!(px, first, "hot draw #{n} diverged");
    }
}

/// Warm non-Store resident draw: zero readbacks, zero seed uploads, zero creates, zero allocs.
#[test]
fn warm_non_store_zero_readback_seed_create_alloc() {
    let _g = engine_test_lock().lock().unwrap_or_else(|e| e.into_inner());
    engine::test_reset_engine();
    let (v, f) = triangle_spirv();
    let identity = TargetIdentity::Surface {
        id: 42,
        width: 16,
        height: 16,
        generation: 1,
    };
    // Cold: seed import + draw with readback so we can verify content, mark ready.
    let mut cold = engine_req(&v, &f, 16, 16);
    cold.target_identity = Some(identity.clone());
    cold.load_op = Some(LoadOp::Clear([0.0, 0.0, 0.0, 0.0]));
    cold.skip_readback = false;
    match engine::execute_draw_request(&cold) {
        Ok(o) => assert_fullscreen_fragment_color("resident_cold", &o.pixels, 16, 16),
        Err(e) if skip_if_no_gpu(&e.to_string()) => {
            eprintln!("SKIP warm_non_store: {e}");
            return;
        }
        Err(e) => panic!("cold resident: {e}"),
    }
    // Warm non-Store: LoadFromTarget, skip readback.
    let mut warm = engine_req(&v, &f, 16, 16);
    warm.target_identity = Some(identity.clone());
    warm.load_op = Some(LoadOp::LoadFromTarget);
    warm.skip_readback = true;
    // One warm-up under residency.
    engine::execute_draw_request(&warm).expect("resident warm-up");
    engine::reset_draw_counters();
    let before = engine::counter_snapshot();
    engine::execute_draw_request(&warm).expect("resident warm non-Store");
    let after = engine::counter_snapshot();
    let d = after.delta_since(&before);
    assert_eq!(
        d.readbacks, 0,
        "warm non-Store must do zero readbacks: {d:?}"
    );
    assert_eq!(
        d.seed_uploads, 0,
        "warm non-Store must do zero seed uploads: {d:?}"
    );
    assert_eq!(d.creates, 0, "warm non-Store must do zero creates: {d:?}");
    assert_eq!(d.allocs, 0, "warm non-Store must do zero allocs: {d:?}");
    assert_eq!(
        d.render_post_wait_skips, 1,
        "no-readback draw must skip the post-submit fence wait: {d:?}"
    );
    // Boundary materialization still works: read_target waits the shared
    // fence first, so it returns the exact content of the skipped-wait draw.
    let px = engine::read_target(&identity).expect("read_target after warm");
    assert_fullscreen_fragment_color("read_target", &px, 16, 16);
}

/// Workstream E: present a resident target straight into imported host memory
/// (VK_EXT_external_memory_host) — the GPU DMAs the frame into caller pages with
/// no CPU readback copy. Off-VM stand-in for guest scanout pages: we supply our
/// own page-aligned buffer and assert the GPU-written bytes appear through the
/// raw pointer. Skips if the ICD lacks the extension.
#[test]
fn present_into_host_ptr_writes_frame_zero_copy() {
    use std::alloc::{alloc_zeroed, dealloc, Layout};
    let _g = engine_test_lock().lock().unwrap_or_else(|e| e.into_inner());
    engine::test_reset_engine();
    let (v, f) = triangle_spirv();
    let w = 16u32;
    let h = 16u32;
    let identity = TargetIdentity::Surface {
        id: 77,
        width: w,
        height: h,
        generation: 1,
    };
    // Render the known triangle color into the resident target (readback so we
    // know the GPU content is right before we test the import path).
    let mut cold = engine_req(&v, &f, w, h);
    cold.target_identity = Some(identity.clone());
    cold.load_op = Some(LoadOp::Clear([0.0, 0.0, 0.0, 0.0]));
    cold.skip_readback = false;
    match engine::execute_draw_request(&cold) {
        Ok(o) => assert_fullscreen_fragment_color("import_present_cold", &o.pixels, w, h),
        Err(e) if skip_if_no_gpu(&e.to_string()) => {
            eprintln!("SKIP present_into_host_ptr: {e}");
            return;
        }
        Err(e) => panic!("cold resident: {e}"),
    }

    // Page-aligned host buffer (guest scanout page stand-in). 64 KiB alignment
    // dominates any driver min_imported_host_pointer_alignment (≤ page size).
    let frame = (w * h * 4) as usize;
    let align = 65536usize;
    let cap = frame.div_ceil(align) * align;
    let layout = Layout::from_size_align(cap, align).unwrap();
    // SAFETY: non-zero layout; freed below with the same layout.
    let ptr = unsafe { alloc_zeroed(layout) };
    assert!(!ptr.is_null(), "host alloc failed");

    engine::reset_draw_counters();
    let before = engine::counter_snapshot();
    // SAFETY: ptr backs `cap` valid bytes, exclusive for this call.
    let res =
        unsafe { engine::present_into_host_ptr_strided(&identity, ptr as *mut _, cap as u64, 0) };
    match res {
        Ok(_) => {
            let after = engine::counter_snapshot();
            let d = after.delta_since(&before);
            assert_eq!(d.import_presents, 1, "import_present counter: {d:?}");
            assert_eq!(
                d.readbacks, 0,
                "import present must not do a CPU readback: {d:?}"
            );
            // Read the frame back through the raw host pointer — these bytes were
            // written by the GPU with no CPU copy.
            // SAFETY: ptr backs at least `frame` initialized bytes post-present.
            let seen = unsafe { std::slice::from_raw_parts(ptr, frame) };
            assert_fullscreen_fragment_color("import_present_host_ptr", seen, w, h);
        }
        Err(e)
            if e.to_string().contains("external_memory_host") || skip_if_no_gpu(&e.to_string()) =>
        {
            eprintln!("SKIP present_into_host_ptr (unsupported): {e}");
        }
        Err(e) => {
            // SAFETY: same ptr/layout from the alloc above.
            unsafe { dealloc(ptr, layout) };
            panic!("present_into_host_ptr: {e}");
        }
    }
    // SAFETY: same ptr/layout from the alloc above.
    unsafe { dealloc(ptr, layout) };
}

/// Deferred render Stores pin their resident target: the registry LRU sweep
/// must skip a pinned slot even when the cap forces evictions, and the pin
/// must refuse absent identities (the runtime then falls back to the
/// synchronous Store).
#[test]
fn pinned_resident_target_survives_registry_cap_sweep() {
    let _g = engine_test_lock().lock().unwrap_or_else(|e| e.into_inner());
    engine::test_reset_engine();
    let (v, f) = triangle_spirv();

    let absent = TargetIdentity::Surface {
        id: 0x9999,
        width: 16,
        height: 16,
        generation: 1,
    };
    assert!(
        !engine::pin_resident_target(&absent),
        "pin must refuse an absent identity"
    );

    let pinned = TargetIdentity::Surface {
        id: 0x600,
        width: 16,
        height: 16,
        generation: 1,
    };
    let mut make = engine_req(&v, &f, 16, 16);
    make.target_identity = Some(pinned.clone());
    match engine::execute_draw_request(&make) {
        Ok(_) => {}
        Err(e) if skip_if_no_gpu(&e.to_string()) => {
            eprintln!("SKIP pinned_resident_target: {e}");
            return;
        }
        Err(e) => panic!("pinned target draw: {e}"),
    }
    assert!(engine::pin_resident_target(&pinned), "pin ready target");

    let unpinned = TargetIdentity::Surface {
        id: 0x601,
        width: 16,
        height: 16,
        generation: 1,
    };
    let mut make2 = engine_req(&v, &f, 16, 16);
    make2.target_identity = Some(unpinned.clone());
    engine::execute_draw_request(&make2).expect("unpinned target draw");

    // Blow past the non-pinned REGISTRY_CAP: the LRU sweep must evict the
    // unpinned early target (the oldest non-pinned) but rotate over the pinned
    // one. Derive the count from the LIVE cap rather than hard-coding it — this
    // test previously fixed 70 fillers against a cap that was later retuned
    // 64 -> 320, so no eviction ever fired and the "unpinned was evicted" assert
    // below could not hold. `+16` clears the cap with margin so the oldest
    // non-pinned is definitely swept.
    let cap = engine::cap_pressure_snapshot().registry_cap as u32;
    for i in 0..(cap + 16) {
        let mut filler = engine_req(&v, &f, 16, 16);
        filler.target_identity = Some(TargetIdentity::Surface {
            id: 0x700 + i,
            width: 16,
            height: 16,
            generation: 1,
        });
        engine::execute_draw_request(&filler).expect("filler draw");
    }
    assert!(
        engine::resident_content_ready(&pinned),
        "pinned target evicted by the cap sweep"
    );
    assert!(
        !engine::resident_content_ready(&unpinned),
        "unpinned early target should have been LRU-evicted"
    );

    // Unpin → a further sweep may evict it (no assert on timing; just verify
    // the unpin API keeps the slot registered right now).
    engine::unpin_resident_target(&pinned);
    assert!(engine::resident_content_ready(&pinned));
}

/// Workstream E: a BGRA resident target lands guest scanout byte order directly
/// (no CPU RGBA→BGRA swizzle), so import-present writes correct bytes. Same
/// triangle as the RGBA test, but the center pixel bytes are the byte-swap:
/// RGBA ~(64,128,191,255) stored as BGRA ~(191,128,64,255).
#[test]
fn present_into_host_ptr_bgra_target_lands_guest_byte_order() {
    use std::alloc::{alloc_zeroed, dealloc, Layout};
    let _g = engine_test_lock().lock().unwrap_or_else(|e| e.into_inner());
    engine::test_reset_engine();
    let (v, f) = triangle_spirv();
    let w = 16u32;
    let h = 16u32;
    let identity = TargetIdentity::Surface {
        id: 78,
        width: w,
        height: h,
        generation: 1,
    };
    // Resident BGRA target, no readback (content_ready set regardless).
    let mut req = engine_req(&v, &f, w, h);
    req.target_identity = Some(identity.clone());
    req.load_op = Some(LoadOp::Clear([0.0, 0.0, 0.0, 0.0]));
    req.output_bgra = true;
    req.skip_readback = true;
    match engine::execute_draw_request(&req) {
        Ok(_) => {}
        Err(e) if skip_if_no_gpu(&e.to_string()) => {
            eprintln!("SKIP bgra import-present: {e}");
            return;
        }
        Err(e) => panic!("bgra resident draw: {e}"),
    }

    let frame = (w * h * 4) as usize;
    let align = 65536usize;
    let cap = frame.div_ceil(align) * align;
    let layout = Layout::from_size_align(cap, align).unwrap();
    // SAFETY: non-zero layout; freed below with the same layout.
    let ptr = unsafe { alloc_zeroed(layout) };
    assert!(!ptr.is_null(), "host alloc failed");

    // SAFETY: ptr backs `cap` valid bytes, exclusive for this call.
    let res =
        unsafe { engine::present_into_host_ptr_strided(&identity, ptr as *mut _, cap as u64, 0) };
    match res {
        Ok(()) => {
            // SAFETY: ptr backs at least `frame` initialized bytes post-present.
            let seen = unsafe { std::slice::from_raw_parts(ptr, frame) };
            let i = ((h / 2) * w + w / 2) as usize * 4;
            let (b, g, r, a) = (seen[i], seen[i + 1], seen[i + 2], seen[i + 3]);
            assert!(
                near(b, 191) && near(g, 128) && near(r, 64) && near(a, 255),
                "bgra import-present center BGRA=({b},{g},{r},{a}); expected ~(191,128,64,255)"
            );
        }
        Err(e)
            if e.to_string().contains("external_memory_host") || skip_if_no_gpu(&e.to_string()) =>
        {
            eprintln!("SKIP bgra import-present (unsupported): {e}");
        }
        Err(e) => {
            // SAFETY: same ptr/layout from the alloc above.
            unsafe { dealloc(ptr, layout) };
            panic!("bgra present_into_host_ptr: {e}");
        }
    }
    // SAFETY: same ptr/layout from the alloc above.
    unsafe { dealloc(ptr, layout) };
}

/// Direct-present zero-copy export (host-window route B, B2): a content-ready
/// BGRA resident exports straight to a dmabuf fd via a GPU→GPU blit — no CPU
/// readback. Prove the end-to-end product path runs on a real resident: it
/// returns a usable fd + correct geometry, and it leaves the resident intact and
/// readable (the blit restored TRANSFER_SRC_OPTIMAL, so a later `read_target`
/// still sees the same content). The byte-fidelity of the exported pixels is
/// pinned separately by the in-crate `blit_present_into_export_is_byte_identical`
/// (the test crate cannot reach the `pub(crate)` dmabuf import helper).
#[test]
fn export_present_from_resident_returns_usable_fd_and_preserves_resident() {
    use std::os::fd::FromRawFd;
    let _g = engine_test_lock().lock().unwrap_or_else(|e| e.into_inner());
    engine::test_reset_engine();
    let (v, f) = triangle_spirv();
    let w = 16u32;
    let h = 16u32;
    let identity = TargetIdentity::Surface {
        id: 91,
        width: w,
        height: h,
        generation: 1,
    };
    let mut req = engine_req(&v, &f, w, h);
    req.target_identity = Some(identity.clone());
    req.load_op = Some(LoadOp::Clear([0.0, 0.0, 0.0, 0.0]));
    req.output_bgra = true;
    req.skip_readback = true;
    match engine::execute_draw_request(&req) {
        Ok(_) => {}
        Err(e) if skip_if_no_gpu(&e.to_string()) => {
            eprintln!("SKIP export-present resident: {e}");
            return;
        }
        Err(e) => panic!("bgra resident draw: {e}"),
    }

    // Center color the fullscreen triangle wrote (BGRA), read BEFORE the export.
    let before = engine::read_target(&identity).expect("read resident before export");
    let i = ((h / 2) * w + w / 2) as usize * 4;
    let center_before = [before[i], before[i + 1], before[i + 2], before[i + 3]];

    match unsafe { engine::export_present_from_resident_fd_policy(&identity, |_, _, _| true) } {
        Ok((fd, pitch, ew, eh, ring_idx)) => {
            let fd = fd.expect("always-true fd policy must return an fd");
            assert!(fd >= 0, "export returned an invalid fd");
            assert_eq!((ew, eh), (w, h), "export geometry must match the resident");
            assert!(
                pitch >= (w as u64) * 4,
                "LINEAR row pitch {pitch} < tight {}",
                w * 4
            );
            assert!(ring_idx < 3, "ring index {ring_idx} out of range");
            // Own + close the dmabuf fd so the test does not leak it.
            drop(unsafe { std::os::fd::OwnedFd::from_raw_fd(fd) });
        }
        Err(e)
            if e.to_string().contains("extensions unavailable")
                || skip_if_no_gpu(&e.to_string()) =>
        {
            eprintln!("SKIP export-present resident (no dmabuf export): {e}");
            return;
        }
        Err(e) => panic!("export_present_from_resident_fd_policy: {e}"),
    }

    // The export blit must not have corrupted the resident: the same content is
    // still readable (and the tracked layout was restored so the readback's own
    // barrier starts from the right place).
    let after = engine::read_target(&identity).expect("read resident after export");
    let center_after = [after[i], after[i + 1], after[i + 2], after[i + 3]];
    assert_eq!(
        center_before, center_after,
        "resident content changed across the export blit",
    );
    assert!(
        near(center_after[0], 191)
            && near(center_after[1], 128)
            && near(center_after[2], 64)
            && near(center_after[3], 255),
        "resident center BGRA={center_after:?}; expected ~(191,128,64,255)"
    );
}

/// The live window-transition failure samples CPU-decoded RGBA bytes and stores
/// into a resident BGRA target. Lock that complete format chain independently
/// of guest descriptors: shader-visible R/G/B must land as physical B/G/R.
#[test]
fn sampled_rgba_upload_to_bgra_target_preserves_semantic_channels() {
    let _g = engine_test_lock().lock().unwrap_or_else(|e| e.into_inner());
    engine::test_reset_engine();
    let vert = translate_words("textured_quad.air", Stage::Vertex);
    let frag = translate_words("textured_quad.air", Stage::Fragment);
    let w = 16u32;
    let h = 16u32;
    let identity = TargetIdentity::Surface {
        id: 82,
        width: w,
        height: h,
        generation: 1,
    };
    let mut req = engine_req(&vert, &frag, w, h);
    req.vertex_count = 6;
    req.target_identity = Some(identity.clone());
    req.output_bgra = true;
    req.skip_readback = true;
    req.load_op = Some(LoadOp::Clear([0.0, 0.0, 0.0, 0.0]));

    let positions: [[f32; 4]; 6] = [
        [-1.0, -1.0, 0.0, 1.0],
        [1.0, -1.0, 0.0, 1.0],
        [-1.0, 1.0, 0.0, 1.0],
        [-1.0, 1.0, 0.0, 1.0],
        [1.0, -1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 1.0],
    ];
    let uvs: [[f32; 2]; 6] = [
        [0.0, 1.0],
        [1.0, 1.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [1.0, 1.0],
        [1.0, 0.0],
    ];
    let encode_f32 = |values: &[f32]| {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>()
    };
    req.storage_buffers.push(StorageBufferResource {
        binding: 0,
        content: encode_f32(&positions.into_iter().flatten().collect::<Vec<_>>()).into(),
    });
    req.storage_buffers.push(StorageBufferResource {
        binding: 1,
        content: encode_f32(&uvs.into_iter().flatten().collect::<Vec<_>>()).into(),
    });
    let rgba = [17u8, 91, 203, 255];
    req.sampled_images.push(SampledImageResource {
        binding: 32,
        width: 2,
        height: 2,
        layers: 1,
        arrayed: false,
        volume: false,
        cube: false,
        one_dim: false,
        source: SampledSource::Bytes(std::sync::Arc::new(rgba.repeat(4))),
        format: ash::vk::Format::R8G8B8A8_UNORM,
        identity: None,
        swizzle: Default::default(),
    });
    req.samplers.push(SamplerResource::normalized_default(64));

    match engine::execute_draw_request(&req) {
        Ok(_) => {}
        Err(e) if skip_if_no_gpu(&e.to_string()) => {
            eprintln!("SKIP sampled RGBA to BGRA target: {e}");
            return;
        }
        Err(e) => panic!("sampled RGBA to BGRA target: {e}"),
    }
    let raw = engine::read_target(&identity).expect("read BGRA target");
    let center = ((h / 2) * w + w / 2) as usize * 4;
    assert_eq!(
        &raw[center..center + 4],
        &[rgba[2], rgba[1], rgba[0], rgba[3]],
        "shader RGBA must land in guest-visible BGRA byte order"
    );

    let warm_before = engine::counter_snapshot();
    engine::execute_draw_request(&req).expect("exact sampled-content cache hit");
    let warm_delta = engine::counter_snapshot().delta_since(&warm_before);
    assert_eq!(warm_delta.sampled_cache_hits, 1, "warm hit: {warm_delta:?}");
    assert_eq!(
        warm_delta.sampled_reuploads, 0,
        "warm upload: {warm_delta:?}"
    );
    let warm_raw = engine::read_target(&identity).expect("read warm BGRA target");
    assert_eq!(
        &warm_raw[center..center + 4],
        &[rgba[2], rgba[1], rgba[0], rgba[3]],
        "cache hit must preserve sampled shader output"
    );

    let changed_rgba = [201u8, 77, 31, 255];
    req.sampled_images[0].source =
        SampledSource::Bytes(std::sync::Arc::new(changed_rgba.repeat(4)));
    let changed_before = engine::counter_snapshot();
    engine::execute_draw_request(&req).expect("changed sampled-content cache miss");
    let changed_delta = engine::counter_snapshot().delta_since(&changed_before);
    assert_eq!(
        changed_delta.sampled_cache_misses, 1,
        "changed miss: {changed_delta:?}"
    );
    assert_eq!(
        changed_delta.sampled_reuploads, 1,
        "changed upload: {changed_delta:?}"
    );
    let changed_raw = engine::read_target(&identity).expect("read changed BGRA target");
    assert_eq!(
        &changed_raw[center..center + 4],
        &[
            changed_rgba[2],
            changed_rgba[1],
            changed_rgba[0],
            changed_rgba[3],
        ],
        "changed sampled bytes must replace cached shader input"
    );
}

/// A constexpr Metal sampler has no guest sampler object. The translator
/// reflects its packed AIR state and the product creates the corresponding
/// Vulkan descriptor. Exercise that whole handoff on a real engine: leaving the
/// reflected sampler unbound makes this shader return black (or fault on
/// MoltenVK), while the exact binding samples the known source color.
#[test]
fn reflected_static_sampler_descriptor_samples_texture() {
    use metal2vulkan::reflect::ResourceKind;

    let _g = engine_test_lock().lock().unwrap_or_else(|e| e.into_inner());
    engine::test_reset_engine();
    let vert = translate_words("textured_quad.air", Stage::Vertex);
    let tmp = std::env::temp_dir().join(format!(
        "paravirt_engine_{}_static_sampler",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).expect("tmp");
    let air = fixtures().join("render_frag_static_sampler.air");
    let (frag, reflection) =
        metal2vulkan::translate_reflected(air.to_str().unwrap(), Stage::Fragment, &tmp)
            .expect("translate static sampler fixture");
    let mut frag = frag
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect::<Vec<_>>();
    assert_eq!(
        reims_vgpu::runtime::spirv_bind::offset_fragment_sampled_resource_bindings(&mut frag),
        2,
        "fixture has one sampled image and one constexpr sampler"
    );
    let reflected = reflection
        .bindings
        .iter()
        .find(|binding| binding.kind == ResourceKind::StaticSampler)
        .expect("reflected constexpr sampler");
    let descriptor = reflected.descriptor.expect("static sampler descriptor");
    let state = reflected.static_sampler.expect("static sampler state");

    let (w, h) = (16u32, 16u32);
    let mut req = engine_req(&vert, &frag, w, h);
    req.vertex_count = 6;
    let positions: [[f32; 4]; 6] = [
        [-1.0, -1.0, 0.0, 1.0],
        [1.0, -1.0, 0.0, 1.0],
        [-1.0, 1.0, 0.0, 1.0],
        [-1.0, 1.0, 0.0, 1.0],
        [1.0, -1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 1.0],
    ];
    let uvs: [[f32; 2]; 6] = [
        [0.0, 1.0],
        [1.0, 1.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [1.0, 1.0],
        [1.0, 0.0],
    ];
    let encode_f32 = |values: &[f32]| {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>()
    };
    req.storage_buffers.push(StorageBufferResource {
        binding: 0,
        content: encode_f32(&positions.into_iter().flatten().collect::<Vec<_>>()).into(),
    });
    req.storage_buffers.push(StorageBufferResource {
        binding: 1,
        content: encode_f32(&uvs.into_iter().flatten().collect::<Vec<_>>()).into(),
    });
    let rgba = [17u8, 91, 203, 255];
    req.sampled_images.push(SampledImageResource {
        binding: 32 + reims_vgpu::runtime::spirv_bind::FRAG_SAMPLED_RESOURCE_BINDING_OFFSET,
        width: 2,
        height: 2,
        layers: 1,
        arrayed: false,
        volume: false,
        cube: false,
        one_dim: false,
        source: SampledSource::Bytes(std::sync::Arc::new(rgba.repeat(4))),
        format: ash::vk::Format::R8G8B8A8_UNORM,
        identity: None,
        swizzle: Default::default(),
    });
    req.samplers.push(
        reims_vgpu::runtime::metal_draw::reflected_static_sampler_resource(
            "fragment",
            descriptor.binding
                + reims_vgpu::runtime::spirv_bind::FRAG_SAMPLED_RESOURCE_BINDING_OFFSET,
            state,
        )
        .expect("map reflected static sampler"),
    );

    let Some(pixels) = draw_or_skip("reflected static sampler", &req) else {
        return;
    };
    for (index, pixel) in pixels.chunks_exact(4).enumerate() {
        assert_eq!(pixel, rgba, "static sampler pixel {index}");
    }
}

/// Safety net for the deferred "upload host-cache BGRA bytes as native Bgra8"
/// optimization (skip the CPU R/B swizzle): a `SampledSource::Bytes` tagged
/// `ash::vk::Format::B8G8R8A8_UNORM` must sample the SAME semantic color as the equivalent
/// RGBA upload — i.e. `B8G8R8A8_UNORM` bytes `[b,g,r,a]` land in the shader as
/// `(r,g,b,a)`, identical to `Rgba8` bytes `[r,g,b,a]`. Proves the Bytes rail (not
/// just the zero-copy GuestRuns rail) is color-correct for Bgra8 before any
/// loader is switched to stop swizzling.
#[test]
fn sampled_bgra8_bytes_upload_matches_rgba8_semantic_color() {
    let _g = engine_test_lock().lock().unwrap_or_else(|e| e.into_inner());
    engine::test_reset_engine();
    let vert = translate_words("textured_quad.air", Stage::Vertex);
    let frag = translate_words("textured_quad.air", Stage::Fragment);
    let w = 16u32;
    let h = 16u32;

    // Same geometry/UV setup as the RGBA-to-BGRA parity test above.
    let positions: [[f32; 4]; 6] = [
        [-1.0, -1.0, 0.0, 1.0],
        [1.0, -1.0, 0.0, 1.0],
        [-1.0, 1.0, 0.0, 1.0],
        [-1.0, 1.0, 0.0, 1.0],
        [1.0, -1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 1.0],
    ];
    let uvs: [[f32; 2]; 6] = [
        [0.0, 1.0],
        [1.0, 1.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [1.0, 1.0],
        [1.0, 0.0],
    ];
    let encode_f32 = |values: &[f32]| {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>()
    };

    // The single semantic color under test, expressed in each byte order.
    let rgba = [17u8, 91, 203, 255];
    let bgra = [rgba[2], rgba[1], rgba[0], rgba[3]];

    // Render the color once via an Rgba8 upload and once via a Bgra8 upload; the
    // sampled output must be byte-identical.
    let render = |bytes: Vec<u8>, format: ash::vk::Format, id: u32| {
        let identity = TargetIdentity::Surface {
            id,
            width: w,
            height: h,
            generation: 1,
        };
        let mut req = engine_req(&vert, &frag, w, h);
        req.vertex_count = 6;
        req.target_identity = Some(identity.clone());
        req.output_bgra = true;
        req.skip_readback = true;
        req.load_op = Some(LoadOp::Clear([0.0, 0.0, 0.0, 0.0]));
        req.storage_buffers.push(StorageBufferResource {
            binding: 0,
            content: encode_f32(&positions.into_iter().flatten().collect::<Vec<_>>()).into(),
        });
        req.storage_buffers.push(StorageBufferResource {
            binding: 1,
            content: encode_f32(&uvs.into_iter().flatten().collect::<Vec<_>>()).into(),
        });
        req.sampled_images.push(SampledImageResource {
            binding: 32,
            width: 2,
            height: 2,
            layers: 1,
            arrayed: false,
            volume: false,
            cube: false,
            one_dim: false,
            source: SampledSource::Bytes(std::sync::Arc::new(bytes)),
            format,
            identity: None,
            swizzle: Default::default(),
        });
        req.samplers.push(SamplerResource::normalized_default(64));
        match engine::execute_draw_request(&req) {
            Ok(_) => Some(engine::read_target(&identity).expect("read BGRA target")),
            Err(e) if skip_if_no_gpu(&e.to_string()) => {
                eprintln!("SKIP bgra8 bytes upload: {e}");
                None
            }
            Err(e) => panic!("bgra8 bytes upload: {e}"),
        }
    };

    let Some(rgba_out) = render(rgba.repeat(4), ash::vk::Format::R8G8B8A8_UNORM, 90) else {
        return;
    };
    let bgra_out =
        render(bgra.repeat(4), ash::vk::Format::B8G8R8A8_UNORM, 91).expect("bgra8 render");

    let center = ((h / 2) * w + w / 2) as usize * 4;
    // Both uploads carry the identical semantic color, so the guest-visible BGRA
    // target center must be `[b,g,r,a]` in each — and equal to each other.
    assert_eq!(
        &rgba_out[center..center + 4],
        &bgra,
        "rgba8 upload lands as guest-visible BGRA"
    );
    assert_eq!(
        &bgra_out[center..center + 4],
        &bgra,
        "bgra8 upload must sample the SAME semantic color as rgba8"
    );
    assert_eq!(
        &rgba_out[center..center + 4],
        &bgra_out[center..center + 4],
        "bgra8 and rgba8 uploads of one color must render byte-identically"
    );
}

/// **L3's proof.** A decoded type-8 view swizzle must be performed by the image
/// view's component mapping, on the GPU, at sample time — not by rewriting
/// texels, which would force every swizzled texture onto the CPU upload path
/// and cost it the zero-copy crossing.
///
/// Renders one identical RGBA source twice: once with the identity plan and
/// once with a plan that reads `(b, g, r, 1)`. Same bytes, same upload, same
/// everything but the view — so a difference in the output can only have come
/// from the mapping. If the mapping were dropped the two would be equal, which
/// is exactly the silent failure this asserts against.
#[test]
fn a_view_swizzle_is_performed_by_the_image_view_not_the_cpu() {
    use reims_vgpu::contract::pixel_format;

    let _g = engine_test_lock().lock().unwrap_or_else(|e| e.into_inner());
    engine::test_reset_engine();
    let vert = translate_words("textured_quad.air", Stage::Vertex);
    let frag = translate_words("textured_quad.air", Stage::Fragment);
    let w = 16u32;
    let h = 16u32;

    let positions: [[f32; 4]; 6] = [
        [-1.0, -1.0, 0.0, 1.0],
        [1.0, -1.0, 0.0, 1.0],
        [-1.0, 1.0, 0.0, 1.0],
        [-1.0, 1.0, 0.0, 1.0],
        [1.0, -1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 1.0],
    ];
    let uvs: [[f32; 2]; 6] = [
        [0.0, 1.0],
        [1.0, 1.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [1.0, 1.0],
        [1.0, 0.0],
    ];
    let encode_f32 = |values: &[f32]| {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>()
    };

    // A colour whose three channels are all distinct, so any remap is visible.
    let source = [17u8, 91, 203, 255];

    let render = |plan: pixel_format::SwizzlePlan, id: u32| {
        let identity = TargetIdentity::Surface {
            id,
            width: w,
            height: h,
            generation: 1,
        };
        let mut req = engine_req(&vert, &frag, w, h);
        req.vertex_count = 6;
        req.target_identity = Some(identity.clone());
        req.output_bgra = true;
        req.skip_readback = true;
        req.load_op = Some(LoadOp::Clear([0.0, 0.0, 0.0, 0.0]));
        req.storage_buffers.push(StorageBufferResource {
            binding: 0,
            content: encode_f32(&positions.into_iter().flatten().collect::<Vec<_>>()).into(),
        });
        req.storage_buffers.push(StorageBufferResource {
            binding: 1,
            content: encode_f32(&uvs.into_iter().flatten().collect::<Vec<_>>()).into(),
        });
        req.sampled_images.push(SampledImageResource {
            binding: 32,
            width: 2,
            height: 2,
            layers: 1,
            arrayed: false,
            volume: false,
            cube: false,
            one_dim: false,
            source: SampledSource::Bytes(std::sync::Arc::new(source.repeat(4))),
            format: ash::vk::Format::R8G8B8A8_UNORM,
            identity: None,
            swizzle: plan,
        });
        req.samplers.push(SamplerResource::normalized_default(64));
        match engine::execute_draw_request(&req) {
            Ok(_) => Some(engine::read_target(&identity).expect("read target")),
            Err(e) if skip_if_no_gpu(&e.to_string()) => {
                eprintln!("SKIP view swizzle: {e}");
                None
            }
            Err(e) => panic!("view swizzle draw: {e}"),
        }
    };

    let Some(plain) = render(pixel_format::swizzle_identity(), 140) else {
        return;
    };
    // Selectors: 4=B, 3=G, 2=R, 1=One  →  the view reads (b, g, r, 1).
    let reversed = pixel_format::swizzle_plan(&[4, 3, 2, 1]).expect("swizzle plan");
    let swizzled = render(reversed, 141).expect("swizzled render");

    let center = ((h / 2) * w + w / 2) as usize * 4;
    let plain_px = &plain[center..center + 4];
    let swizzled_px = &swizzled[center..center + 4];

    // The target is guest-visible BGRA, so the identity render shows [b,g,r,a].
    assert_eq!(
        plain_px,
        [source[2], source[1], source[0], source[3]],
        "identity plan must leave the sampled colour alone"
    );
    // Reading (b,g,r,1) instead of (r,g,b,a) swaps R and B before the BGRA
    // store, so the stored bytes come back with R and B exchanged again.
    assert_eq!(
        swizzled_px,
        [source[0], source[1], source[2], 255],
        "the view swizzle must reach the sampler"
    );
    assert_ne!(
        plain_px, swizzled_px,
        "identical bytes rendered identically means the mapping was dropped"
    );
}

/// A semantic RGBA Load seed must be converted to the native BGRA attachment
/// order before upload. A partial draw makes the untouched seed observable;
/// fullscreen tests overwrite the bad upload and cannot catch this class.
#[test]
fn partial_draw_preserves_rgba_seed_on_bgra_target() {
    let _g = engine_test_lock().lock().unwrap_or_else(|e| e.into_inner());
    engine::test_reset_engine();
    let (vert, frag) = triangle_spirv();
    let (w, h) = (16u32, 16u32);
    let identity = TargetIdentity::Surface {
        id: 83,
        width: w,
        height: h,
        generation: 1,
    };
    let seed_rgba = [17u8, 91, 203, 255];
    let mut req = engine_req(&vert, &frag, w, h);
    req.target_identity = Some(identity.clone());
    req.output_bgra = true;
    req.skip_readback = true;
    req.load_op = Some(LoadOp::LoadSeed(seed_rgba.repeat((w * h) as usize)));
    req.scissors.push(ScissorResource {
        x: 0,
        y: 0,
        width: 1,
        height: 1,
    });

    match engine::execute_draw_request(&req) {
        Ok(_) => {}
        Err(e) if skip_if_no_gpu(&e.to_string()) => {
            eprintln!("SKIP BGRA partial seed: {e}");
            return;
        }
        Err(e) => panic!("BGRA partial seed: {e}"),
    }
    let raw = engine::read_target(&identity).expect("read BGRA partial-seed target");
    let outside = ((h / 2) * w + w / 2) as usize * 4;
    assert_eq!(
        &raw[outside..outside + 4],
        &[seed_rgba[2], seed_rgba[1], seed_rgba[0], seed_rgba[3]],
        "untouched semantic RGBA seed must remain correct in native BGRA storage"
    );
}

/// Strided import-present: guest IOSurface BPR > width*4 (live mids 1/5 class).
/// GPU must honor buffer_row_length so row N starts at N*bpr, not N*tight.
#[test]
fn present_into_host_ptr_strided_honors_guest_bpr() {
    use std::alloc::{alloc_zeroed, dealloc, Layout};
    let _g = engine_test_lock().lock().unwrap_or_else(|e| e.into_inner());
    engine::test_reset_engine();
    let (v, f) = triangle_spirv();
    let w = 16u32;
    let h = 8u32;
    let tight = w * 4;
    let bpr = tight + 16; // 16 bytes pad per row (matches padded scanout class)
    let identity = TargetIdentity::Surface {
        id: 79,
        width: w,
        height: h,
        generation: 1,
    };
    let mut req = engine_req(&v, &f, w, h);
    req.target_identity = Some(identity.clone());
    req.load_op = Some(LoadOp::Clear([0.0, 0.0, 0.0, 0.0]));
    req.output_bgra = true;
    req.skip_readback = true;
    match engine::execute_draw_request(&req) {
        Ok(_) => {}
        Err(e) if skip_if_no_gpu(&e.to_string()) => {
            eprintln!("SKIP strided import-present: {e}");
            return;
        }
        Err(e) => panic!("strided resident draw: {e}"),
    }

    let frame = (bpr * h) as usize;
    let align = 65536usize;
    let cap = frame.div_ceil(align) * align;
    let layout = Layout::from_size_align(cap, align).unwrap();
    // SAFETY: non-zero layout; freed below.
    let ptr = unsafe { alloc_zeroed(layout) };
    assert!(!ptr.is_null());

    // SAFETY: ptr backs `cap` exclusive bytes for the DMA.
    let res =
        unsafe { engine::present_into_host_ptr_strided(&identity, ptr as *mut _, cap as u64, bpr) };
    match res {
        Ok(_) => {
            // SAFETY: GPU wrote `frame` bytes into ptr.
            let seen = unsafe { std::slice::from_raw_parts(ptr, frame) };
            // Pad region at end of row 0 must stay zero (not spilled image bytes).
            let pad0 = &seen[tight as usize..bpr as usize];
            assert!(
                pad0.iter().all(|&b| b == 0),
                "row pad must stay zero under strided DMA; got {pad0:?}"
            );
            // Center of image (row h/2, col w/2) via guest BPR layout.
            let row = (h / 2) as usize;
            let col = (w / 2) as usize;
            let i = row * (bpr as usize) + col * 4;
            let (b, g, r, a) = (seen[i], seen[i + 1], seen[i + 2], seen[i + 3]);
            assert!(
                near(b, 191) && near(g, 128) && near(r, 64) && near(a, 255),
                "strided center BGRA=({b},{g},{r},{a}); expected ~(191,128,64,255)"
            );
        }
        Err(e)
            if e.to_string().contains("external_memory_host") || skip_if_no_gpu(&e.to_string()) =>
        {
            eprintln!("SKIP strided import-present (unsupported): {e}");
        }
        Err(e) => {
            unsafe { dealloc(ptr, layout) };
            panic!("strided present_into_host_ptr: {e}");
        }
    }
    unsafe { dealloc(ptr, layout) };
}

/// Premult One/OMSA: GPU Load+blend matches the retired software composite oracle.
#[test]
fn premult_one_omsa_gpu_blend_matches_software_oracle() {
    let _g = engine_test_lock().lock().unwrap_or_else(|e| e.into_inner());
    engine::test_reset_engine();
    let (v, f) = triangle_spirv();
    // Seed: solid gray base (128,128,128,255)
    let w = 16u32;
    let h = 16u32;
    let seed: Vec<u8> = (0..(w * h)).flat_map(|_| [128u8, 128, 128, 255]).collect();
    // GPU path: Load seed + One/OMSA blend (fullscreen frag writes opaque color).
    let mut gpu = engine_req(&v, &f, w, h);
    gpu.target_rgba8 = Some(seed.clone());
    gpu.blend = Some(BlendStateResource {
        src_color: BlendFactor::One,
        dst_color: BlendFactor::OneMinusSrcAlpha,
        color_op: BlendOp::Add,
        src_alpha: BlendFactor::One,
        dst_alpha: BlendFactor::OneMinusSrcAlpha,
        alpha_op: BlendOp::Add,
        constants: [0.0; 4],
    });
    let gpu_px = match engine::execute_draw_request(&gpu) {
        Ok(o) => o.pixels,
        Err(e) if skip_if_no_gpu(&e.to_string()) => {
            eprintln!("SKIP premult: {e}");
            return;
        }
        Err(e) => panic!("premult gpu: {e}"),
    };
    // Software oracle: draw over black with same blend, then composite.
    let mut black = engine_req(&v, &f, w, h);
    black.blend = gpu.blend;
    let over_black = engine::execute_draw_request(&black)
        .expect("over black")
        .pixels;
    let (soft, _) =
        reims_vgpu::runtime::metal_draw::load_composite_premult_one_omsa(&over_black, &seed);
    // Allow ±1 LSB for unorm rounding differences between GPU blend and CPU composite.
    assert_eq!(gpu_px.len(), soft.len());
    for (i, (g, s)) in gpu_px.iter().zip(soft.iter()).enumerate() {
        assert!(
            (*g as i32 - *s as i32).abs() <= 1,
            "premult mismatch at byte {i}: gpu={g} soft={s}"
        );
    }
}

/// Class A zero-copy wipe lock: after a skip_readback Store (no CPU pixels,
/// host_cache would be empty/evicted), the next pass must LoadFromTarget so
/// progressive multi-pass content stays on the resident image. Engine Clear
/// (default when load_op/target_rgba8 are None) would black the target.
/// Product choice is also locked by `type11_load_ready_uses_resident_not_clear`.
#[test]
fn skip_readback_store_then_load_from_target_preserves_content() {
    let _g = engine_test_lock().lock().unwrap_or_else(|e| e.into_inner());
    engine::test_reset_engine();
    let (v, f) = triangle_spirv();
    let identity = TargetIdentity::Surface {
        id: 91,
        width: 16,
        height: 16,
        generation: 1,
    };
    // Pass 1: product zero-copy Store shape — resident path uses skip_readback
    // so no CPU pixels land in host_cache.
    let mut store1 = engine_req(&v, &f, 16, 16);
    store1.target_identity = Some(identity.clone());
    store1.load_op = Some(LoadOp::Clear([0.0, 0.0, 0.0, 0.0]));
    store1.skip_readback = true;
    match engine::execute_draw_request(&store1) {
        Ok(_) => {}
        Err(e) if skip_if_no_gpu(&e.to_string()) => {
            eprintln!("SKIP skip_readback_load_preserve: {e}");
            return;
        }
        Err(e) => panic!("store1: {e}"),
    }
    assert!(
        engine::resident_content_ready(&identity),
        "store1 must mark content_ready"
    );
    // Pass 2: LOAD after host_cache miss — LoadFromTarget, no CPU seed.
    let mut store2 = engine_req(&v, &f, 16, 16);
    store2.target_identity = Some(identity.clone());
    store2.load_op = Some(LoadOp::LoadFromTarget);
    store2.skip_readback = true;
    store2.target_rgba8 = None;
    engine::execute_draw_request(&store2).expect("store2 LoadFromTarget");
    let px = engine::read_target(&identity).expect("read_target after progressive Stores");
    assert_fullscreen_fragment_color("progressive_skip_readback", &px, 16, 16);
    // No seed_uploads on pass 2 (LoadFromTarget, not CPU seed).
    // Counters are process-global; just ensure content survived.
    assert!(engine::resident_content_ready(&identity));
}

/// Cross-boot retained-frame lock: a device reset must evict identity-keyed
/// resident images even when the next guest reuses the same id/generation.
#[test]
fn guest_reset_evicts_resident_targets_without_destroying_context() {
    let _g = engine_test_lock().lock().unwrap_or_else(|e| e.into_inner());
    engine::test_reset_engine();
    let (v, f) = triangle_spirv();
    let identity = TargetIdentity::Surface {
        id: 91,
        width: 16,
        height: 16,
        generation: 1,
    };
    let mut draw = engine_req(&v, &f, 16, 16);
    draw.target_identity = Some(identity.clone());
    draw.skip_readback = true;
    match engine::execute_draw_request(&draw) {
        Ok(_) => {}
        Err(e) if skip_if_no_gpu(&e.to_string()) => {
            eprintln!("SKIP guest_reset_evicts_resident: {e}");
            return;
        }
        Err(e) => panic!("resident setup: {e}"),
    }
    assert!(engine::resident_content_ready(&identity));

    let stats = engine::reset_guest_state();
    assert_eq!(stats.resident_targets, 1);
    assert!(stats.had_context);
    assert!(!engine::resident_content_ready(&identity));
}

/// Chain byte-parity: LoadFromTarget chain matches CPU-seed chain.
#[test]
fn chain_load_from_target_byte_parity_vs_cpu_seed() {
    let _g = engine_test_lock().lock().unwrap_or_else(|e| e.into_inner());
    engine::test_reset_engine();
    let (v, f) = triangle_spirv();
    // CPU-seed chain: draw1 clear → pixels → draw2 LoadSeed(pixels) → pixels2
    let d1 = engine_req(&v, &f, 16, 16);
    let p1 = match engine::execute_draw_request(&d1) {
        Ok(o) => o.pixels,
        Err(e) if skip_if_no_gpu(&e.to_string()) => {
            eprintln!("SKIP chain: {e}");
            return;
        }
        Err(e) => panic!("chain d1: {e}"),
    };
    let mut d2_cpu = engine_req(&v, &f, 16, 16);
    d2_cpu.target_rgba8 = Some(p1.clone());
    let p2_cpu = engine::execute_draw_request(&d2_cpu)
        .expect("cpu seed chain")
        .pixels;

    // GPU-resident chain: same identity LoadFromTarget.
    engine::test_reset_engine();
    let identity = TargetIdentity::Surface {
        id: 7,
        width: 16,
        height: 16,
        generation: 1,
    };
    let mut g1 = engine_req(&v, &f, 16, 16);
    g1.target_identity = Some(identity.clone());
    g1.skip_readback = true;
    engine::execute_draw_request(&g1).expect("gpu chain d1");
    let mut g2 = engine_req(&v, &f, 16, 16);
    g2.target_identity = Some(identity.clone());
    g2.load_op = Some(LoadOp::LoadFromTarget);
    g2.skip_readback = false; // read back for compare
    let p2_gpu = engine::execute_draw_request(&g2)
        .expect("gpu chain d2")
        .pixels;
    assert_eq!(
        p2_gpu, p2_cpu,
        "LoadFromTarget chain must match CPU-seed chain"
    );
}

/// Resident GVA chain (type-2/3 rail): a 3-record chain keeps intermediate
/// content on the engine target — exactly one readback (the final contract
/// Store), zero CPU seed uploads, two post-submit wait skips — and the final
/// pixels byte-match the CPU round-trip chain (readback → LoadSeed re-upload
/// per record) it replaces.
#[test]
fn gva_chain_resident_single_readback_matches_cpu_seed_chain() {
    let _g = engine_test_lock().lock().unwrap_or_else(|e| e.into_inner());
    engine::test_reset_engine();
    let (v, f) = triangle_spirv();
    // CPU round-trip reference chain: every record reads back, next record
    // re-uploads the pixels as its seed (the legacy GVA chain rail).
    let d1 = engine_req(&v, &f, 16, 16);
    let p1 = match engine::execute_draw_request(&d1) {
        Ok(o) => o.pixels,
        Err(e) if skip_if_no_gpu(&e.to_string()) => {
            eprintln!("SKIP gva_chain: {e}");
            return;
        }
        Err(e) => panic!("gva_chain d1: {e}"),
    };
    let mut d2 = engine_req(&v, &f, 16, 16);
    d2.target_rgba8 = Some(p1);
    let p2 = engine::execute_draw_request(&d2).expect("cpu d2").pixels;
    let mut d3 = engine_req(&v, &f, 16, 16);
    d3.target_rgba8 = Some(p2);
    let p3_cpu = engine::execute_draw_request(&d3).expect("cpu d3").pixels;

    // Resident chain on a Gva identity: intermediates never touch the CPU.
    engine::test_reset_engine();
    let identity = TargetIdentity::Gva {
        gva: 0x2f00_0000,
        width: 16,
        height: 16,
        generation: 0,
    };
    engine::reset_draw_counters();
    let before = engine::counter_snapshot();
    let mut g1 = engine_req(&v, &f, 16, 16);
    g1.target_identity = Some(identity.clone());
    g1.skip_readback = true;
    engine::execute_draw_request(&g1).expect("gva chain g1");
    let mut g2 = engine_req(&v, &f, 16, 16);
    g2.target_identity = Some(identity.clone());
    g2.load_op = Some(LoadOp::LoadFromTarget);
    g2.skip_readback = true;
    engine::execute_draw_request(&g2).expect("gva chain g2");
    let mut g3 = engine_req(&v, &f, 16, 16);
    g3.target_identity = Some(identity.clone());
    g3.load_op = Some(LoadOp::LoadFromTarget);
    g3.skip_readback = false; // final record: contract Store readback
    let p3_gpu = engine::execute_draw_request(&g3)
        .expect("gva chain g3")
        .pixels;
    let d = engine::counter_snapshot().delta_since(&before);
    assert_eq!(d.readbacks, 1, "only the final record reads back: {d:?}");
    assert_eq!(
        d.seed_uploads, 0,
        "no CPU seed on the resident chain: {d:?}"
    );
    assert_eq!(
        d.render_post_wait_skips, 2,
        "both intermediates skip the fence wait: {d:?}"
    );
    assert_eq!(
        p3_gpu, p3_cpu,
        "resident GVA chain must byte-match the CPU round-trip chain"
    );
}

/// Deferred GVA Store (single/final record): the draw renders into the
/// registry resident with skip_readback — zero readbacks and one post-wait
/// skip on the stamp path — and the flush-on-access `read_target` returns
/// byte-identical pixels to the synchronous readback Store it replaces.
#[test]
fn gva_deferred_store_flush_read_matches_sync_store() {
    let _g = engine_test_lock().lock().unwrap_or_else(|e| e.into_inner());
    engine::test_reset_engine();
    let (v, f) = triangle_spirv();
    // Sync reference: the legacy Store readback.
    let d_sync = engine_req(&v, &f, 16, 16);
    let p_sync = match engine::execute_draw_request(&d_sync) {
        Ok(o) => o.pixels,
        Err(e) if skip_if_no_gpu(&e.to_string()) => {
            eprintln!("SKIP gva_deferred_store: {e}");
            return;
        }
        Err(e) => panic!("sync store: {e}"),
    };

    // Deferred Store shape: registry Gva resident, no stamp-path readback.
    engine::test_reset_engine();
    let identity = TargetIdentity::Gva {
        gva: 0x3a00_0000,
        width: 16,
        height: 16,
        generation: 0,
    };
    engine::reset_draw_counters();
    let before = engine::counter_snapshot();
    let mut g = engine_req(&v, &f, 16, 16);
    g.target_identity = Some(identity.clone());
    g.skip_readback = true;
    engine::execute_draw_request(&g).expect("deferred store draw");
    let d = engine::counter_snapshot().delta_since(&before);
    assert_eq!(d.readbacks, 0, "deferred Store must not read back: {d:?}");
    assert_eq!(
        d.render_post_wait_skips, 1,
        "deferred Store skips the fence wait: {d:?}"
    );
    assert!(engine::pin_resident_target(&identity), "window pin");

    // Flush-on-access landing: one readback, byte parity with the sync Store.
    let before_flush = engine::counter_snapshot();
    let p_flush = engine::read_target(&identity).expect("flush read_target");
    let df = engine::counter_snapshot().delta_since(&before_flush);
    assert_eq!(df.readbacks, 1, "flush is the single readback: {df:?}");
    engine::unpin_resident_target(&identity);
    assert_eq!(
        p_flush, p_sync,
        "deferred flush bytes must match the sync Store readback"
    );
}

#[test]
fn device_loss_named_and_recreate_bounded() {
    let _g = engine_test_lock().lock().unwrap_or_else(|e| e.into_inner());
    engine::test_reset_engine();
    let (v, f) = triangle_spirv();
    let req = engine_req(&v, &f, 8, 8);
    match engine::execute_draw_request(&req) {
        Ok(_) => {}
        Err(e) if skip_if_no_gpu(&e.to_string()) => {
            eprintln!("SKIP device_loss: {e}");
            return;
        }
        Err(e) => panic!("{e}"),
    }
    engine::test_force_device_lost_once();
    let err = engine::execute_draw_request(&req).expect_err("forced loss");
    let s = err.to_string();
    assert!(
        s.contains("reason=vk_device_lost_forced_draw"),
        "the forced draw rail must retain its exact typed reason, got: {s}"
    );
    let mut saw_named = true;
    for _ in 0..MAX_DEVICE_RECREATES + 2 {
        engine::test_poison_and_flush();
        match engine::execute_draw_request(&req) {
            Ok(_) => {}
            Err(e) => {
                let es = e.to_string();
                assert!(
                    es.contains("device_lost")
                        || es.contains("DeviceLost")
                        || es.contains("recreate"),
                    "unexpected error after poison: {es}"
                );
                saw_named = true;
            }
        }
    }
    assert!(saw_named);
    assert!(
        engine::device_recreate_count() <= MAX_DEVICE_RECREATES + 3,
        "recreate count unbounded: {}",
        engine::device_recreate_count()
    );
    let snap = engine::counter_snapshot();
    assert!(
        snap.device_lost >= 1,
        "device_lost counter must fire, got {}",
        snap.device_lost
    );
}

/// In-flight ring lock (3-deep): consecutive no-readback resident draws land
/// on separate ring slots without retiring each other; only the entry that
/// wraps onto a still-pending slot pays the retire (ring_retire_blocks). A
/// boundary read then retires everything and sees the exact final content.
/// If RING_DEPTH changes, the wrap arithmetic below must follow.
#[test]
fn ring_overlaps_in_flight_no_readback_draws() {
    let _g = engine_test_lock().lock().unwrap_or_else(|e| e.into_inner());
    engine::test_reset_engine();
    let (v, f) = triangle_spirv();
    let id_a = TargetIdentity::Surface {
        id: 91,
        width: 16,
        height: 16,
        generation: 1,
    };
    let id_b = TargetIdentity::Surface {
        id: 92,
        width: 16,
        height: 16,
        generation: 1,
    };
    // Cold sync draws mark both targets ready (content verified).
    for (label, identity) in [("ring_cold_a", &id_a), ("ring_cold_b", &id_b)] {
        let mut cold = engine_req(&v, &f, 16, 16);
        cold.target_identity = Some((*identity).clone());
        cold.load_op = Some(LoadOp::Clear([0.0, 0.0, 0.0, 0.0]));
        match engine::execute_draw_request(&cold) {
            Ok(o) => assert_fullscreen_fragment_color(label, &o.pixels, 16, 16),
            Err(e) if skip_if_no_gpu(&e.to_string()) => {
                eprintln!("SKIP ring_overlaps: {e}");
                return;
            }
            Err(e) => panic!("{label}: {e}"),
        }
    }
    // Warm both once, then quiesce so the measured draws start with an idle ring.
    for identity in [&id_a, &id_b] {
        let mut warm = engine_req(&v, &f, 16, 16);
        warm.target_identity = Some((*identity).clone());
        warm.load_op = Some(LoadOp::LoadFromTarget);
        warm.skip_readback = true;
        engine::execute_draw_request(&warm).expect("ring warm-up");
    }
    engine::read_target(&id_a).expect("ring quiesce");
    engine::reset_draw_counters();
    let before = engine::counter_snapshot();
    // Four async draws: the first three occupy every slot in flight; the
    // fourth wraps onto the first slot and must retire it.
    for (n, identity) in [&id_a, &id_b, &id_a, &id_b].into_iter().enumerate() {
        let mut warm = engine_req(&v, &f, 16, 16);
        warm.target_identity = Some((*identity).clone());
        warm.load_op = Some(LoadOp::LoadFromTarget);
        warm.skip_readback = true;
        engine::execute_draw_request(&warm).unwrap_or_else(|e| panic!("ring async #{n}: {e}"));
    }
    let d = engine::counter_snapshot().delta_since(&before);
    assert_eq!(
        d.render_post_wait_skips, 4,
        "all four draws must skip the post-submit wait: {d:?}"
    );
    // Deferred submit: each alternating-target draw opens its own batch; the
    // next draw's begin_entry flushes it (submit), so three flushes land in
    // the window and the fourth batch is still open (flushed by the boundary
    // read below). Same-target runs sharing one CB are covered by
    // vk_engine_batch.rs.
    assert_eq!(d.batch_opens, 4, "each draw opens a batch: {d:?}");
    assert_eq!(d.batch_joins, 0, "alternating targets never join: {d:?}");
    assert_eq!(
        d.batch_flushes, 3,
        "each subsequent draw flushes its predecessor's batch: {d:?}"
    );
    // Boundary reads retire the in-flight work and see the final content.
    let px = engine::read_target(&id_a).expect("ring boundary read a");
    assert_fullscreen_fragment_color("ring_read_a", &px, 16, 16);
    let px = engine::read_target(&id_b).expect("ring boundary read b");
    assert_fullscreen_fragment_color("ring_read_b", &px, 16, 16);
}

/// Present-boundary GPU seed: `seed_from_target` copies another ready
/// resident's content into the draw target on the GPU (no CPU seed upload),
/// and the pass loads it. A zero-invocation draw then reads back the source
/// content byte-exactly.
#[test]
fn seed_from_target_gpu_copies_front_frame() {
    let _g = engine_test_lock().lock().unwrap_or_else(|e| e.into_inner());
    engine::test_reset_engine();
    let (v, f) = triangle_spirv();
    let front = TargetIdentity::Surface {
        id: 71,
        width: 16,
        height: 16,
        generation: 1,
    };
    let back = TargetIdentity::Surface {
        id: 72,
        width: 16,
        height: 16,
        generation: 1,
    };
    // Render known content into the "front frame" resident.
    let mut cold = engine_req(&v, &f, 16, 16);
    cold.target_identity = Some(front.clone());
    cold.load_op = Some(LoadOp::Clear([0.0, 0.0, 0.0, 0.0]));
    let front_pixels = match engine::execute_draw_request(&cold) {
        Ok(o) => {
            assert_fullscreen_fragment_color("gpu_seed_front", &o.pixels, 16, 16);
            o.pixels
        }
        Err(e) if skip_if_no_gpu(&e.to_string()) => {
            eprintln!("SKIP seed_from_target: {e}");
            return;
        }
        Err(e) => panic!("front draw: {e}"),
    };
    engine::reset_draw_counters();
    let before = engine::counter_snapshot();
    // Zero-invocation draw into a different identity, seeded from the front
    // resident: the readback must be the front content, with zero CPU seed
    // uploads and exactly one GPU seed copy.
    let mut seeded = engine_req(&v, &f, 16, 16);
    seeded.vertex_count = 0;
    seeded.target_identity = Some(back.clone());
    seeded.seed_from_target = Some(front.clone());
    let out = engine::execute_draw_request(&seeded).expect("gpu-seeded draw");
    assert_eq!(
        out.pixels, front_pixels,
        "GPU seed copy must reproduce the front content byte-exactly"
    );
    let d = engine::counter_snapshot().delta_since(&before);
    assert_eq!(d.seed_gpu_copies, 1, "one GPU seed copy: {d:?}");
    assert_eq!(d.seed_uploads, 0, "no CPU seed upload: {d:?}");
    // Named-error rails: src==dst and missing resident fail closed.
    let mut self_seed = engine_req(&v, &f, 16, 16);
    self_seed.target_identity = Some(back.clone());
    self_seed.seed_from_target = Some(back.clone());
    assert!(engine::execute_draw_request(&self_seed).is_err());
    let absent = TargetIdentity::Surface {
        id: 73,
        width: 16,
        height: 16,
        generation: 9,
    };
    let mut missing = engine_req(&v, &f, 16, 16);
    missing.target_identity = Some(back.clone());
    missing.seed_from_target = Some(absent);
    assert!(engine::execute_draw_request(&missing).is_err());
}

/// True N-attachment MRT: a draw with a secondary color attachment renders the
/// primary (slot 0) normally AND leaves the secondary as a ready, sampleable
/// resident that a later draw can bind via `SampledSource::Target`. This is the
/// mechanism that produces a fragment shader's secondary output (e.g. the
/// vibrancy coverage mask) instead of silently discarding it.
#[test]
fn mrt_secondary_attachment_becomes_sampleable_resident() {
    let _g = engine_test_lock().lock().unwrap_or_else(|e| e.into_inner());
    engine::test_reset_engine();
    let (v, f) = triangle_spirv();
    let primary = TargetIdentity::Surface {
        id: 0x60,
        width: 16,
        height: 16,
        generation: 1,
    };
    let secondary = TargetIdentity::Surface {
        id: 0x61,
        width: 16,
        height: 16,
        generation: 1,
    };

    let mut mrt = engine_req(&v, &f, 16, 16);
    mrt.target_identity = Some(primary.clone());
    mrt.secondary_targets.push(SecondaryColorTarget {
        identity: secondary.clone(),
        width: 16,
        height: 16,
        format: ash::vk::Format::R8G8B8A8_UNORM,
        clear: [0.0, 0.0, 1.0, 1.0],
        load: false,
        // Unblended: this parity case checks the attachment is written at
        // all, not how it composites.
        blend: None,
    });
    match engine::execute_draw_request(&mrt) {
        // Slot 0 (primary) still receives the shader's location-0 output.
        Ok(o) => assert_fullscreen_fragment_color("mrt_primary", &o.pixels, 16, 16),
        Err(e) if skip_if_no_gpu(&e.to_string()) => {
            eprintln!("SKIP mrt_secondary: {e}");
            return;
        }
        Err(e) => panic!("mrt draw: {e}"),
    }

    // The secondary attachment persisted as its own resident.
    assert!(
        engine::resident_content_ready(&secondary),
        "secondary MRT attachment must be a ready resident"
    );

    // A later draw binds the secondary as a sampled resident — the exact path
    // the CC vibrancy pipe=25 draw uses to read its coverage mask.
    let consumer_target = TargetIdentity::Surface {
        id: 0x62,
        width: 16,
        height: 16,
        generation: 1,
    };
    let mut consume = engine_req(&v, &f, 16, 16);
    consume.target_identity = Some(consumer_target);
    consume.sampled_images.push(SampledImageResource {
        binding: 1,
        width: 16,
        height: 16,
        layers: 1,
        arrayed: false,
        volume: false,
        cube: false,
        one_dim: false,
        source: SampledSource::Target(secondary.clone()),
        format: ash::vk::Format::R8G8B8A8_UNORM,
        identity: None,
        swizzle: Default::default(),
    });
    engine::reset_draw_counters();
    let before = engine::counter_snapshot();
    engine::execute_draw_request(&consume).expect("bind MRT secondary as sampled resident");
    let delta = engine::counter_snapshot().delta_since(&before);
    assert_eq!(
        delta.sampled_gpu_binds, 1,
        "secondary must bind directly with no CPU reupload: {delta:?}"
    );
    assert_eq!(delta.sampled_reuploads, 0, "no host reupload: {delta:?}");
}

/// The vibrancy coverage mask is Metal RG16Float (0x41). Exercise the real
/// secondary format end-to-end: the RG16Float render pass / pipeline / resident
/// image build and render without error, and the mask persists as a resident.
#[test]
fn mrt_rg16float_secondary_builds_and_renders() {
    let _g = engine_test_lock().lock().unwrap_or_else(|e| e.into_inner());
    engine::test_reset_engine();
    let (v, f) = triangle_spirv();
    let primary = TargetIdentity::Surface {
        id: 0x63,
        width: 32,
        height: 32,
        generation: 1,
    };
    let mask = TargetIdentity::Gva {
        gva: 0x3cf5000,
        width: 32,
        height: 32,
        generation: 0,
    };
    let mut mrt = engine_req(&v, &f, 32, 32);
    mrt.target_identity = Some(primary.clone());
    mrt.secondary_targets.push(SecondaryColorTarget {
        identity: mask.clone(),
        width: 32,
        height: 32,
        format: ash::vk::Format::R16G16_SFLOAT,
        clear: [1.0, 0.5, 0.0, 0.0],
        load: false,
        // Unblended: this is the vibrancy coverage-mask shape, and a mask is a
        // raw store. Which is exactly why every secondary used to be forced
        // unblended — one real case generalized into a rule for all of them.
        blend: None,
    });
    match engine::execute_draw_request(&mrt) {
        Ok(o) => assert_fullscreen_fragment_color("mrt_rg16f_primary", &o.pixels, 32, 32),
        Err(e) if skip_if_no_gpu(&e.to_string()) => {
            eprintln!("SKIP mrt_rg16float: {e}");
            return;
        }
        Err(e) => panic!("mrt rg16float draw: {e}"),
    }
    assert!(
        engine::resident_content_ready(&mask),
        "RG16Float mask must be a ready resident after the MRT draw"
    );
}

/// Firewall: an empty `secondary_targets` leaves the classic single-attachment
/// path untouched — same fragment color, zero MRT residents created.
#[test]
fn single_rt_draw_unaffected_by_mrt_path() {
    let _g = engine_test_lock().lock().unwrap_or_else(|e| e.into_inner());
    engine::test_reset_engine();
    let (v, f) = triangle_spirv();
    let target = TargetIdentity::Surface {
        id: 0x64,
        width: 16,
        height: 16,
        generation: 1,
    };
    let mut req = engine_req(&v, &f, 16, 16);
    req.target_identity = Some(target.clone());
    assert!(req.secondary_targets.is_empty());
    match engine::execute_draw_request(&req) {
        Ok(o) => assert_fullscreen_fragment_color("single_rt_guard", &o.pixels, 16, 16),
        Err(e) if skip_if_no_gpu(&e.to_string()) => {
            eprintln!("SKIP single_rt_guard: {e}");
            return;
        }
        Err(e) => panic!("single-rt guard: {e}"),
    }
    // A neighbouring MRT-secondary identity was never materialized.
    let never = TargetIdentity::Gva {
        gva: 0xdead000,
        width: 16,
        height: 16,
        generation: 0,
    };
    assert!(!engine::resident_content_ready(&never));
}

/// Framebuffer fetch (`color_input`): the fragment shader reads its own
/// destination pixel through the attachment-0 subpass input at the m2v
/// ColorInput binding (96) and inverts RGB. Seeding the target and drawing the
/// fullscreen triangle must yield the inverted seed — which proves the input
/// attachment was bound and read (an unbound input reads zero and would output
/// solid 255,255,255). This is the exact structural shape of the live
/// WindowServer composite (`air.render_target` INPUT param `dest_0`) whose
/// unbound read was the arm64 MoltenVK GPU-address-fault class.
#[test]
fn framebuffer_fetch_reads_destination_via_input_attachment() {
    let _g = engine_test_lock().lock().unwrap_or_else(|e| e.into_inner());
    let (v, _) = triangle_spirv();
    let f = translate_words("render_frag_fetch.air", Stage::Fragment);
    let (w, h) = (16u32, 16u32);
    let mut req = engine_req(&v, &f, w, h);
    req.color_input = true;
    // Seed (64, 128, 191, 255) → expect ~(191, 127, 64, 255).
    req.target_rgba8 = Some([64, 128, 191, 255].repeat((w * h) as usize));
    let Some(px) = draw_or_skip("framebuffer_fetch", &req) else {
        return;
    };
    assert_eq!(px.len(), (w * h * 4) as usize, "fetch: readback size");
    for p in 0..(w * h) as usize {
        let (r, g, b, a) = (px[p * 4], px[p * 4 + 1], px[p * 4 + 2], px[p * 4 + 3]);
        assert!(
            near(r, 191) && near(g, 127) && near(b, 64) && near(a, 255),
            "fetch: pixel {p} RGBA=({r},{g},{b},{a}); expected ~(191,127,64,255)"
        );
    }
}
