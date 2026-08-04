//! Direct host-Metal backend: pure-Rust Metal encode driven from `runtime/`.
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
pub(crate) mod format;
#[cfg(target_os = "macos")]
mod function;
#[cfg(target_os = "macos")]
pub(crate) mod mipmap;
#[cfg(target_os = "macos")]
pub(crate) mod mtl_enum;
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
