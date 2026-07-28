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
    SampledImageResource, SampledSource, ScissorResource, StorageBufferResource, TargetIdentity,
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
/// would read guest RAM after ack-fast let the guest repaint it (rect_void
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
    assert_eq!(
        mid.buffer_zerocopy_binds, 0,
        "no flush-time gather was recorded"
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

/// A draw sampling `SampledSource::GuestRuns` reads guest RAM when its CB
/// executes, so it must NOT defer its submit (no open, no join) — the
/// immediate submit keeps the record→execute window at its validated
/// pre-batching size. Requires a real host-pointer import; skips if the ICD
/// refuses the heap allocation.
#[test]
fn sampled_guest_runs_draw_does_not_batch() {
    let _guard = engine_test_lock().lock().unwrap();
    let (vert, frag) = triangle_spirv();
    let identity = TargetIdentity::Surface {
        id: 990_402,
        width: W,
        height: H,
        generation: 1,
    };

    // Page-aligned backing so VK_EXT_external_memory_host can import it.
    let layout = std::alloc::Layout::from_size_align(4096, 4096).unwrap();
    let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
    assert!(!ptr.is_null());
    if !engine::ensure_host_imports(&[engine::GuestRun {
        host_ptr: ptr as usize,
        len: 4096,
    }]) {
        eprintln!("skipping: ICD refused host-pointer import of heap memory");
        unsafe { std::alloc::dealloc(ptr, layout) };
        return;
    }

    let before = engine::counter_snapshot();
    let mut req = batch_req(
        &vert,
        &frag,
        &identity,
        LoadOp::Clear([0.0, 0.0, 0.0, 0.0]),
        half_scissor(true),
    );
    req.sampled_images.push(SampledImageResource {
        binding: 0,
        width: 4,
        height: 4,
        layers: 1,
        arrayed: false,
        volume: false,
        cube: false,
        one_dim: false,
        source: SampledSource::GuestRuns(GuestRunSource {
            runs: std::sync::Arc::new(vec![GuestRun {
                host_ptr: ptr as usize,
                len: 64,
            }]),
            total_len: 64,
            row_length_texels: 0,
        }),
        format: ash::vk::Format::R8G8B8A8_UNORM,
        identity: None,
        swizzle: Default::default(),
    });
    match engine::execute_draw_request(&req) {
        Ok(_) => {}
        Err(e) => {
            let msg = e.to_string();
            if skip_if_no_gpu(&msg) {
                eprintln!("skipping: {msg}");
                unsafe { std::alloc::dealloc(ptr, layout) };
                return;
            }
            panic!("zc-sampled draw: {msg}");
        }
    }
    let d = engine::counter_snapshot().delta_since(&before);
    assert_eq!(
        d.batch_opens, 0,
        "zc-sampled draw must not open a batch: {d:?}"
    );
    assert_eq!(
        d.batch_joins, 0,
        "zc-sampled draw must not join a batch: {d:?}"
    );
    engine::test_quiesce_ring();
    unsafe { std::alloc::dealloc(ptr, layout) };
}
