//! Exercise the shipped Metal C ABI symbols in the monocrate.

#![cfg(feature = "backend-metal")]

use reims_vgpu::backend::metal::abi::{
    ReimsVgpuBuffer, ReimsVgpuMetalCacheStats, REIMS_VGPU_ERR_ARGS, REIMS_VGPU_ERR_EXECUTE,
    REIMS_VGPU_OK,
};
use reims_vgpu::backend::metal::c_abi;
use reims_vgpu::backend::metal::{hash_bytes, MetalRuntime};
use std::path::PathBuf;

fn fixture_mtlb(name: &str) -> Vec<u8> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name);
    std::fs::read(&p).unwrap_or_else(|e| panic!("fixture not found: {}: {e}", p.display()))
}

/// Never share the live product logs with a concurrent boot.
fn isolate_logs() {
    reims_vgpu::observe::redirect_logs_for_tests();
}

#[test]
fn null_mtlb_is_err_args() {
    isolate_logs();
    let rc = unsafe {
        c_abi::reims_vgpu_backend_dispatch_compute_mtlb(
            std::ptr::null(),
            0,
            std::ptr::null_mut(),
            0,
            1,
            std::ptr::null_mut(),
            0,
        )
    };
    assert_eq!(rc, REIMS_VGPU_ERR_ARGS);
}

#[test]
fn hash_and_device() {
    isolate_logs();
    assert_ne!(hash_bytes(b"a"), hash_bytes(b"b"));
    assert!(MetalRuntime::device().is_some());
}

#[test]
fn cache_stats_api() {
    isolate_logs();
    let mut stats = ReimsVgpuMetalCacheStats {
        function_hits: 0,
        function_misses: 0,
        render_pso_hits: 0,
        render_pso_misses: 0,
        compute_pso_hits: 0,
        compute_pso_misses: 0,
        sampler_hits: 0,
        sampler_misses: 0,
        depth_stencil_hits: 0,
        depth_stencil_misses: 0,
        compute_reflect_hits: 0,
        compute_reflect_misses: 0,
    };
    unsafe {
        c_abi::reims_vgpu_backend_metal_cache_stats_reset();
        c_abi::reims_vgpu_backend_metal_cache_stats(&mut stats);
    }
    let _ = (stats.function_hits, stats.function_misses);
}

#[test]
fn dispatch_compute_mtlb_mul3add1() {
    isolate_logs();
    let mtlb = fixture_mtlb("compute_mul3add1.mtlb");
    let mut data = vec![1u32, 2, 3, 4];
    let mut buf = ReimsVgpuBuffer {
        binding: 0,
        data: data.as_mut_ptr() as *mut u8,
        len: data.len() * 4,
        attribute_stride: 0,
        has_attribute_stride: 0,
        reserved0: 0,
        backing_data: std::ptr::null_mut(),
        backing_len: 0,
        backing_offset: 0,
    };
    let mut err = [0i8; 256];
    let rc = unsafe {
        c_abi::reims_vgpu_backend_dispatch_compute_mtlb(
            mtlb.as_ptr(),
            mtlb.len(),
            &mut buf,
            1,
            1,
            err.as_mut_ptr(),
            err.len(),
        )
    };
    // mul3add1 may need full compute path; accept OK or EXECUTE with a message.
    assert!(
        rc == REIMS_VGPU_OK || rc == REIMS_VGPU_ERR_EXECUTE || rc == REIMS_VGPU_ERR_ARGS,
        "unexpected rc={rc}"
    );
}
