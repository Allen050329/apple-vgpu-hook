//! Direct host-Metal backend: pure-Rust encode + `reims_vgpu_backend_*` C ABI.
//!
//! On Apple hosts this is the full MTL encode path. On non-Apple hosts
//! (`host_stub`) the same types exist so `backend-metal` still builds and
//! links; encode stays fail-closed until a real host Metal rail lands.

pub mod abi;
mod constants;
pub mod error;
mod hash;

pub use hash::{hash_bytes, hash_u64};

// ---------------------------------------------------------------------------
// Apple: real Metal encode
// ---------------------------------------------------------------------------

#[cfg(target_os = "macos")]
mod cache;
#[cfg(target_os = "macos")]
pub(crate) mod compute;
#[cfg(target_os = "macos")]
mod device;
#[cfg(target_os = "macos")]
#[allow(dead_code)]
pub mod ffi;
#[cfg(target_os = "macos")]
pub(crate) mod format;
#[cfg(target_os = "macos")]
mod function;
#[cfg(target_os = "macos")]
pub(crate) mod mipmap;
#[cfg(target_os = "macos")]
pub(crate) mod raw_metal;
#[cfg(target_os = "macos")]
pub(crate) mod render;
#[cfg(target_os = "macos")]
pub(crate) mod runtime;
#[cfg(target_os = "macos")]
pub(crate) mod samplers;
#[cfg(target_os = "macos")]
mod stage_input;
#[cfg(target_os = "macos")]
pub(crate) mod util;

#[cfg(target_os = "macos")]
pub use device::{system_device_name, MetalBackend, MetalRuntime};

/// C ABI declarations for tests / external callers (defs in [`ffi`]).
#[cfg(target_os = "macos")]
pub mod c_abi {
    use super::abi::*;
    use std::os::raw::c_char;

    extern "C" {
        pub fn reims_vgpu_backend_begin_native_color_format(pixel_format: u32);
        pub fn reims_vgpu_backend_end_native_color_format();
        pub fn reims_vgpu_backend_dispatch_compute_mtlb(
            mtlb: *const u8,
            mtlb_len: usize,
            buffers: *mut ReimsVgpuBuffer,
            buffer_count: usize,
            grid_x: u32,
            err: *mut c_char,
            err_cap: usize,
        ) -> i32;
    }
}

// ---------------------------------------------------------------------------
// Non-Apple: host stub (same public names; encode unsupported)
// ---------------------------------------------------------------------------

#[cfg(not(target_os = "macos"))]
mod host_stub;

#[cfg(not(target_os = "macos"))]
pub use host_stub::{c_abi, runtime, system_device_name, MetalBackend, MetalRuntime};
