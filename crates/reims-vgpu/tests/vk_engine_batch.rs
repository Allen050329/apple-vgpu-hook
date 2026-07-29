//! Off-VM regression suite for draw batching increment 1 (deferred submit).
//!
//! Proves that same-target skip_readback draws share one command buffer
//! (opener + joiners), that content composes correctly across the open pass
//! boundary (LoadFromTarget inside the batch), and that every consumer path
//! (read_target, cross-target draw) flushes the open batch before touching
//! GPU content. Requires a working Vulkan ICD; skips cleanly if init fails.
//!
//! **Serial:** the engine is process-global; all tests take the suite lock.

#![cfg(feature = "backend-vulkan")]

use metal2vulkan::passes::Stage;
use reims_vgpu::backend::vulkan::engine::{
    self, BufferContent, DrawRequest, GuestRun, GuestRunSource, LoadOp, PrimitiveTopology,
    SampledImageResource, SampledSource, SamplerResource, ScissorResource, StorageBufferResource,
    TargetIdentity,
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
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/air")
}

fn translate_words(name: &str, stage: Stage) -> Vec<u32> {
    let tmp = std::env::temp_dir().join(format!(
        "paravirt_engine_batch_{}_{}_{:?}",
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

/// float4(0.25, 0.5, 0.75, 1.0) → unorm8 ≈ (64, 128, 191, 255); allow ±1 LSB.
fn near(got: u8, want: u8) -> bool {
    (got as i32 - want as i32).abs() <= 1
}

fn is_frag_color(px: &[u8]) -> bool {
    near(px[0], 64) && near(px[1], 128) && near(px[2], 191) && near(px[3], 255)
}

fn is_zero(px: &[u8]) -> bool {
    px == [0, 0, 0, 0]
}

const W: u32 = 64;
const H: u32 = 64;

fn batch_req(
    vert: &[u32],
    frag: &[u32],
    identity: &TargetIdentity,
    load: LoadOp,
    scissor: ScissorResource,
) -> DrawRequest {
    DrawRequest {
        vert_spirv: std::sync::Arc::new(vert.to_vec()),
        frag_spirv: std::sync::Arc::new(frag.to_vec()),
        width: W,
        height: H,
        vertex_count: 3,
        flip_viewport_y: true,
        first_vertex: 0,
        instance_count: Some(1),
        base_instance: 0,
        primitive_topology: PrimitiveTopology::Triangle,
        target_identity: Some(identity.clone()),
        load_op: Some(load),
        skip_readback: true,
        scissors: vec![scissor],
        ..Default::default()
    }
}

fn half_scissor(left: bool) -> ScissorResource {
    ScissorResource {
        x: if left { 0 } else { W / 2 },
        y: 0,
        width: W / 2,
        height: H,
    }
}

/// Opener (Clear, left half) + joiner (LoadFromTarget, right half) share one
/// CB; the flush at read_target submits both and the readback shows BOTH
/// halves colored — the joiner's LOAD preserved the opener's half across the
/// intra-CB pass boundary.
#[test]
fn batched_draws_compose_and_flush_on_read() {
    let _guard = engine_test_lock().lock().unwrap();
    let (vert, frag) = triangle_spirv();
    let identity = TargetIdentity::Surface {
        id: 990_101,
        width: W,
        height: H,
        generation: 1,
    };

    let before = engine::counter_snapshot();
    let opener = batch_req(
        &vert,
        &frag,
        &identity,
        LoadOp::Clear([0.0, 0.0, 0.0, 0.0]),
        half_scissor(true),
    );
    match engine::execute_draw_request(&opener) {
        Ok(out) => assert!(out.pixels.is_empty(), "skip_readback returns no pixels"),
        Err(e) => {
            let msg = e.to_string();
            if skip_if_no_gpu(&msg) {
                eprintln!("skipping: {msg}");
                return;
            }
            panic!("opener draw: {msg}");
        }
    }
    let joiner = batch_req(
        &vert,
        &frag,
        &identity,
        LoadOp::LoadFromTarget,
        half_scissor(false),
    );
    engine::execute_draw_request(&joiner).expect("joiner draw");
    let mid = engine::counter_snapshot().delta_since(&before);
    assert_eq!(mid.batch_opens, 1, "first draw opens the batch");
    assert_eq!(mid.batch_joins, 1, "second draw joins the open CB");
    assert_eq!(mid.batch_flushes, 0, "no flush before a consumer arrives");

    let px = engine::read_target(&identity).expect("read_target flushes the batch");
    let after = engine::counter_snapshot().delta_since(&before);
    assert_eq!(after.batch_flushes, 1, "read_target submitted the batch");
    assert_eq!(
        after.batch_flush_draws, 2,
        "the one submit carried both draws"
    );

    assert_eq!(px.len(), (W * H * 4) as usize);
    for y in [0u32, H / 2, H - 1] {
        for x in [0u32, W / 4, W / 2, 3 * W / 4, W - 1] {
            let i = ((y * W + x) * 4) as usize;
            assert!(
                is_frag_color(&px[i..i + 4]),
                "batched composite at ({x},{y}) = {:?}",
                &px[i..i + 4]
            );
        }
    }
}

/// A draw to a DIFFERENT target must not join; claiming its slot flushes the
/// open batch first, so the first target's content is complete when read.
#[test]
fn cross_target_draw_flushes_open_batch() {
    let _guard = engine_test_lock().lock().unwrap();
    let (vert, frag) = triangle_spirv();
    let a = TargetIdentity::Surface {
        id: 990_201,
        width: W,
        height: H,
        generation: 1,
    };
    let b = TargetIdentity::Surface {
        id: 990_202,
        width: W,
        height: H,
        generation: 1,
    };

    let before = engine::counter_snapshot();
    let opener = batch_req(
        &vert,
        &frag,
        &a,
        LoadOp::Clear([0.0, 0.0, 0.0, 0.0]),
        half_scissor(true),
    );
    match engine::execute_draw_request(&opener) {
        Ok(_) => {}
        Err(e) => {
            let msg = e.to_string();
            if skip_if_no_gpu(&msg) {
                eprintln!("skipping: {msg}");
                return;
            }
            panic!("opener draw: {msg}");
        }
    }
    // Different identity: not joinable — begin_entry flushes A's batch, and
    // this draw opens a batch of its own.
    let other = batch_req(
        &vert,
        &frag,
        &b,
        LoadOp::Clear([0.0, 0.0, 0.0, 0.0]),
        half_scissor(false),
    );
    engine::execute_draw_request(&other).expect("cross-target draw");
    let mid = engine::counter_snapshot().delta_since(&before);
    assert_eq!(mid.batch_opens, 2, "each target opened its own batch");
    assert_eq!(mid.batch_joins, 0, "cross-target draws never join");
    assert_eq!(mid.batch_flushes, 1, "claiming B's slot flushed A's batch");

    // A: left half colored, right half untouched clear — single-draw batch
    // content is exact after its flush.
    let px = engine::read_target(&a).expect("read A");
    let left = ((10 * W + 8) * 4) as usize;
    let right = ((10 * W + W / 2 + 8) * 4) as usize;
    assert!(
        is_frag_color(&px[left..left + 4]),
        "A left half = {:?}",
        &px[left..left + 4]
    );
    assert!(
        is_zero(&px[right..right + 4]),
        "A right half = {:?}",
        &px[right..right + 4]
    );
}

/// The prefetch pool submits its GPU→host copy on a dedicated CB/fence,
/// bypassing begin_entry — arming MUST flush the open batch first, or the
/// copy would be queued ahead of the batched draws producing the content.
#[test]
fn prefetch_arm_flushes_open_batch() {
    let _guard = engine_test_lock().lock().unwrap();
    let (vert, frag) = triangle_spirv();
    let identity = TargetIdentity::Surface {
        id: 990_301,
        width: W,
        height: H,
        generation: 1,
    };

    let before = engine::counter_snapshot();
    let opener = batch_req(
        &vert,
        &frag,
        &identity,
        LoadOp::Clear([0.0, 0.0, 0.0, 0.0]),
        half_scissor(true),
    );
    match engine::execute_draw_request(&opener) {
        Ok(_) => {}
        Err(e) => {
            let msg = e.to_string();
            if skip_if_no_gpu(&msg) {
                eprintln!("skipping: {msg}");
                return;
            }
            panic!("opener draw: {msg}");
        }
    }
    let mid = engine::counter_snapshot().delta_since(&before);
    assert_eq!(mid.batch_opens, 1, "draw opened a batch");
    assert_eq!(mid.batch_flushes, 0, "batch still open before the arm");
}

/// A batch refuses joiners at BATCH_MAX_DRAWS (8): draw 9 flushes + reopens,
/// draw 10 joins the second batch. Keeps the GPU fed and the staging pool
/// recycling instead of hoarding a whole run in one pending ring entry.
#[test]
fn batch_length_cap_flushes_and_reopens() {
    let _guard = engine_test_lock().lock().unwrap();
    let (vert, frag) = triangle_spirv();
    let identity = TargetIdentity::Surface {
        id: 990_401,
        width: W,
        height: H,
        generation: 1,
    };

    let before = engine::counter_snapshot();
    let opener = batch_req(
        &vert,
        &frag,
        &identity,
        LoadOp::Clear([0.0, 0.0, 0.0, 0.0]),
        half_scissor(true),
    );
    match engine::execute_draw_request(&opener) {
        Ok(_) => {}
        Err(e) => {
            let msg = e.to_string();
            if skip_if_no_gpu(&msg) {
                eprintln!("skipping: {msg}");
                return;
            }
            panic!("opener draw: {msg}");
        }
    }
    for n in 1..10 {
        let joiner = batch_req(
            &vert,
            &frag,
            &identity,
            LoadOp::LoadFromTarget,
            half_scissor(n % 2 == 0),
        );
        engine::execute_draw_request(&joiner).unwrap_or_else(|e| panic!("draw #{n}: {e}"));
    }
    let d = engine::counter_snapshot().delta_since(&before);
    assert_eq!(d.batch_opens, 2, "cap at 8 forces a second batch: {d:?}");
    assert_eq!(
        d.batch_joins, 8,
        "7 join the first batch, 1 the second: {d:?}"
    );
    assert_eq!(d.batch_flushes, 1, "the cap flushed exactly once: {d:?}");
    assert_eq!(
        d.batch_flush_draws, 8,
        "the full first batch flushed: {d:?}"
    );
    engine::test_quiesce_ring();
}

/// A deferred-submit draw whose storage buffer is `BufferContent::GuestRuns`
/// must snapshot the runs on the CPU at record time — a flush-time GPU gather
/// would read guest RAM after ack-fast let the guest repaint it (the
/// black-band class, live A/B 2026-07-19). No host-import window exists in
/// this process, so the legacy gather path would fail with
/// `buffer_guest_run_import_missing`; the snapshot path succeeds and the
/// backing can even be dropped before the flush.
#[test]
fn batched_guest_runs_buffer_snapshots_at_record() {
    let _guard = engine_test_lock().lock().unwrap();
    let (vert, frag) = triangle_spirv();
    let identity = TargetIdentity::Surface {
        id: 990_401,
        width: W,
        height: H,
        generation: 1,
    };

    let before = engine::counter_snapshot();
    let backing = vec![7u8; 64];
    let mut opener = batch_req(
        &vert,
        &frag,
        &identity,
        LoadOp::Clear([0.0, 0.0, 0.0, 0.0]),
        half_scissor(true),
    );
    opener.storage_buffers.push(StorageBufferResource {
        binding: 0,
        content: BufferContent::GuestRuns(GuestRunSource {
            runs: std::sync::Arc::new(vec![GuestRun {
                host_ptr: backing.as_ptr() as usize,
                len: backing.len() as u64,
            }]),
            total_len: backing.len() as u64,
            row_length_texels: 0,
        }),
    });
    match engine::execute_draw_request(&opener) {
        Ok(out) => assert!(out.pixels.is_empty(), "skip_readback returns no pixels"),
        Err(e) => {
            let msg = e.to_string();
            if skip_if_no_gpu(&msg) {
                eprintln!("skipping: {msg}");
                return;
            }
            assert!(
                !msg.contains("buffer_guest_run_import_missing"),
                "batched GuestRuns must snapshot at record time, not gather at flush: {msg}"
            );
            panic!("batched GuestRuns opener: {msg}");
        }
    }
    drop(backing);

    let mid = engine::counter_snapshot().delta_since(&before);
    assert_eq!(mid.batch_opens, 1, "snapshotted draw still opens a batch");
    assert_eq!(
        mid.buffer_snapshot_binds, 1,
        "GuestRuns content was CPU-snapshotted"
    );

    let px = engine::read_target(&identity).expect("read_target flushes the batch");
    for y in [0u32, H / 2, H - 1] {
        let i = ((y * W + W / 4) * 4) as usize;
        assert!(
            is_frag_color(&px[i..i + 4]),
            "left half at (.,{y}) = {:?}",
            &px[i..i + 4]
        );
    }
    engine::test_quiesce_ring();
}

/// **The guest-run sampled rail's first content coverage.**
///
/// `SampledSource::GuestRuns` names texels that live in guest RAM. The device
/// used to read them itself, through a `VK_EXT_external_memory_host` import of
/// the guest pages; it now gathers them on the CPU into pooled staging. What
/// has to survive that swap is the only thing the rail was ever for: the texels
/// the guest wrote must be the texels the fragment shader samples.
///
/// Nothing tested that before, in either mechanism. Every other sampled case in
/// the tree builds `SampledSource::Bytes`, so this rail had no executing test at
/// all — its two counters were the pair the failure-census flagged as
/// "asserted, never nonzero". The predecessor of this case asserted a *refusal*
/// by slug and one `== 0` counter, which is the shape that hid the sampled-cache
/// defect: it passes whether the path works or is entirely broken.
///
/// The fixture is what makes the assertion possible. `textured_quad` samples
/// binding 32 and writes the sampled colour out, so the output pixels *are* the
/// guest bytes — a full-screen quad over a uniform 2x2 texture means every
/// covered pixel must equal the one colour written into the host page. The
/// colour is deliberately not the fragment-shader constant every other case
/// here checks, so a draw that ignored the sampler could not pass.
///
/// The runs are two halves of one page written separately, which also exercises
/// the multi-run concatenation `write_staging_from_runs` performs: a single-run
/// case would pass with the offset arithmetic broken.
#[test]
fn sampled_guest_runs_land_the_guest_bytes_the_shader_samples() {
    let _guard = engine_test_lock().lock().unwrap();
    let vert = translate_words("textured_quad.air", Stage::Vertex);
    let frag = translate_words("textured_quad.air", Stage::Fragment);
    let identity = TargetIdentity::Surface {
        id: 990_402,
        width: W,
        height: H,
        generation: 1,
    };

    // One x86 guest page holding a uniform 2x2 RGBA8 texture, written as two
    // adjacent runs of two texels each.
    const TEXEL: [u8; 4] = [17, 140, 203, 255];
    let layout = std::alloc::Layout::from_size_align(4096, 4096).unwrap();
    let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
    assert!(!ptr.is_null());
    // SAFETY: `ptr` backs 4096 zeroed bytes; 16 texel bytes fit.
    unsafe { std::ptr::copy_nonoverlapping(TEXEL.repeat(4).as_ptr(), ptr, 16) };

    let encode_f32 = |values: &[f32]| {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>()
    };
    let quad: [[f32; 4]; 6] = [
        [-1.0, -1.0, 0.5, 1.0],
        [1.0, -1.0, 0.5, 1.0],
        [-1.0, 1.0, 0.5, 1.0],
        [-1.0, 1.0, 0.5, 1.0],
        [1.0, -1.0, 0.5, 1.0],
        [1.0, 1.0, 0.5, 1.0],
    ];
    let uvs: [[f32; 2]; 6] = [
        [0.0, 1.0],
        [1.0, 1.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [1.0, 1.0],
        [1.0, 0.0],
    ];

    let mut req = DrawRequest {
        vert_spirv: std::sync::Arc::new(vert),
        frag_spirv: std::sync::Arc::new(frag),
        width: W,
        height: H,
        vertex_count: 6,
        flip_viewport_y: true,
        first_vertex: 0,
        instance_count: Some(1),
        primitive_topology: PrimitiveTopology::Triangle,
        target_identity: Some(identity.clone()),
        load_op: Some(LoadOp::Clear([0.0, 0.0, 0.0, 0.0])),
        skip_readback: true,
        ..Default::default()
    };
    req.storage_buffers.push(StorageBufferResource {
        binding: 0,
        content: encode_f32(&quad.into_iter().flatten().collect::<Vec<_>>()).into(),
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
        source: SampledSource::GuestRuns(GuestRunSource {
            runs: std::sync::Arc::new(vec![
                GuestRun {
                    host_ptr: ptr as usize,
                    len: 8,
                },
                GuestRun {
                    host_ptr: ptr as usize + 8,
                    len: 8,
                },
            ]),
            total_len: 16,
            row_length_texels: 0,
        }),
        format: ash::vk::Format::R8G8B8A8_UNORM,
        identity: None,
        swizzle: Default::default(),
    });
    req.samplers.push(SamplerResource::normalized_default(64));

    let outcome = engine::execute_draw_request(&req);
    if let Err(e) = &outcome {
        if skip_if_no_gpu(&e.to_string()) {
            eprintln!("skipping: {e}");
            // SAFETY: `ptr` is still live here and nothing reads it after this.
            unsafe { std::alloc::dealloc(ptr, layout) };
            return;
        }
    }
    outcome.expect("a CPU-gathered guest-run sampled draw must execute");
    let px = engine::read_target(&identity).expect("read_target flushes the batch");
    engine::test_quiesce_ring();
    // The gather reads the page during `execute_draw_request`, so the page must
    // outlive that call and only that call.
    // SAFETY: `ptr` is still live here and nothing reads it after this.
    unsafe { std::alloc::dealloc(ptr, layout) };

    // Every pixel of the full-screen quad samples the same uniform texel. ±1 LSB
    // covers the unorm round-trip through the sampler's filtering.
    for y in [0u32, H / 2, H - 1] {
        for x in [0u32, W / 2, W - 1] {
            let i = ((y * W + x) * 4) as usize;
            let got = &px[i..i + 4];
            assert!(
                got.iter()
                    .zip(TEXEL.iter())
                    .all(|(g, w)| (*g as i32 - *w as i32).abs() <= 1),
                "guest-run texels did not reach the shader at ({x},{y}): \
                 got {got:?}, wrote {TEXEL:?}"
            );
        }
    }
}
