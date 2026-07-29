//! Direct host-Metal backend: pure-Rust encode + `reims_vgpu_backend_*` C ABI.
//!
//! macOS only. `backend-metal` on any other target is rejected by the
//! `compile_error!` in `lib.rs`, so there is no non-Apple arm of this module
//! and every `target_os = "macos"` gate below is a statement of that fact
//! rather than a branch.

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

