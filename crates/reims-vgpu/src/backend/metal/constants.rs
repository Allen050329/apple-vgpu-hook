//! Caps matching `reims_vgpu_backend_metal.m`.

use crate::backend::metal::abi::{
    REIMS_VGPU_BINDING_SAMPLER_BASE, REIMS_VGPU_BINDING_TEXTURE_BASE,
};

pub const REIMS_VGPU_METAL_MAX_ATTRS: usize = 31;
pub const REIMS_VGPU_METAL_MAX_BUFFERS: usize = 31;
pub const REIMS_VGPU_METAL_MAX_TEXTURES: usize = 31;
pub const REIMS_VGPU_METAL_MAX_SAMPLERS: usize = 16;
/// Metal max color attachments per render pass / PSO.
pub const REIMS_VGPU_METAL_MAX_COLOR_RTS: usize = 8;

// Each cap is the `cap` of a `ClockCache`, whose replacement arm computes its
// slot as `clock % cap`. A zero would divide by zero there rather than merely
// caching nothing, so the pin sits beside each declaration — the constant is
// what a future edit touches, and this is the check that edit has to survive.
pub const REIMS_VGPU_FN_CACHE_CAP: usize = 96;
const _: () = assert!(REIMS_VGPU_FN_CACHE_CAP > 0);
pub const REIMS_VGPU_RENDER_PSO_CACHE_CAP: usize = 64;
const _: () = assert!(REIMS_VGPU_RENDER_PSO_CACHE_CAP > 0);
pub const REIMS_VGPU_COMPUTE_PSO_CACHE_CAP: usize = 64;
const _: () = assert!(REIMS_VGPU_COMPUTE_PSO_CACHE_CAP > 0);
pub const REIMS_VGPU_SAMPLER_CACHE_CAP: usize = 32;
const _: () = assert!(REIMS_VGPU_SAMPLER_CACHE_CAP > 0);
pub const REIMS_VGPU_DEPTH_STENCIL_CACHE_CAP: usize = 16;
const _: () = assert!(REIMS_VGPU_DEPTH_STENCIL_CACHE_CAP > 0);
pub const REIMS_VGPU_COMPUTE_REFLECT_CACHE_CAP: usize = 64;
const _: () = assert!(REIMS_VGPU_COMPUTE_REFLECT_CACHE_CAP > 0);

/// Metal `MTLBufferLayoutStrideDynamic` == `NSUIntegerMax`.
pub const MTL_BUFFER_LAYOUT_STRIDE_DYNAMIC: u64 = u64::MAX;
const _: () = assert!(MTL_BUFFER_LAYOUT_STRIDE_DYNAMIC == u64::MAX);

// The three binding bands do not overlap. A `const` assertion rather than a
// `#[test]`, for the reason the buffer-bind-limit pin below spells out.
const _: () = assert!(REIMS_VGPU_METAL_MAX_BUFFERS as u32 <= REIMS_VGPU_BINDING_TEXTURE_BASE);
const _: () = assert!(
    REIMS_VGPU_BINDING_TEXTURE_BASE + REIMS_VGPU_METAL_MAX_TEXTURES as u32
        <= REIMS_VGPU_BINDING_SAMPLER_BASE
);

/// The buffer argument table is one Metal limit, so the two spellings of it
/// must stay equal.
///
/// Four bind paths gate on it and they must all refuse the same index: direct
/// compute (`backend::metal::compute` via
/// [`crate::backend::metal::util::valid_buffer_binding`], which reads
/// `REIMS_VGPU_METAL_MAX_BUFFERS`), direct render and render ICB inheritance
/// (both `metal_draw::MAX_BIND_SLOTS`), and compute ICB inheritance
/// (`valid_buffer_binding`). Letting the two constants drift would leave one
/// pair of paths passing an index to `setBuffer:offset:atIndex:` that the other
/// pair rejects, and Metal answers an out-of-range index with an exception that
/// aborts the process rather than a status this device can decline.
///
/// # Why these are `const` assertions and not tests
///
/// They were tests, and on the host this project is developed from they never
/// ran even once. This module is `backend-metal`-gated, so its `#[cfg(test)]`
/// block is compiled out of the Vulkan arm entirely, and `AGENTS.md` runs the
/// `backend-metal` `--lib` suite on Apple hosts only. The check standing between
/// four bind paths and a process-aborting Metal exception was therefore live on
/// no machine that anybody edits this code on.
///
/// A `const` assertion is evaluated by `rustc` whenever this file is compiled,
/// which includes the cross-compiled `--target aarch64-apple-darwin` clippy arm
/// that `AGENTS.md` requires from Linux. Same guarantee, checked everywhere,
/// and it fails the build rather than a suite nobody on this pathway runs.
const _: () =
    assert!(REIMS_VGPU_METAL_MAX_BUFFERS as u32 == crate::runtime::metal_draw::MAX_BIND_SLOTS);
