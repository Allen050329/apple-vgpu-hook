//! Metal argument-table caps.
//!
//! These are the sizes this backend's encoders accept, and each is checked
//! before the `setBuffer:`/`setTexture:`/`setSamplerState:` call it guards —
//! Metal answers an out-of-range index with an exception that aborts the
//! process rather than a status this device can decline.

use crate::backend::metal::abi::{
    REIMS_VGPU_BINDING_SAMPLER_BASE, REIMS_VGPU_BINDING_TEXTURE_BASE,
};

pub const REIMS_VGPU_METAL_MAX_ATTRS: usize = 31;
pub const REIMS_VGPU_METAL_MAX_BUFFERS: usize = 31;
/// The texture argument table: Metal's own, and Apple's serializer's.
///
/// It was 32 — the width of the descriptor binding band, not a Metal fact —
/// because a texture at index 32 would have carried
/// [`REIMS_VGPU_BINDING_SAMPLER_BASE`], sampler 0's number, and
/// [`texture_index`](crate::backend::metal::util::texture_index) could not have
/// told the two apart. The sampler band moved up
/// (`spirv_bind::widen_sampled_bands`) so the texture band is 128 wide, and the
/// band assertion below is what holds the two in step.
pub const REIMS_VGPU_METAL_MAX_TEXTURES: usize = 128;
pub const REIMS_VGPU_METAL_MAX_SAMPLERS: usize = 16;
/// Metal max color attachments per render pass / PSO.
pub const REIMS_VGPU_METAL_MAX_COLOR_RTS: usize = 8;

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
/// (both `draw::MAX_BUFFER_BIND_SLOTS`), and compute ICB inheritance
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
    assert!(REIMS_VGPU_METAL_MAX_BUFFERS as u32 == crate::runtime::draw::MAX_BUFFER_BIND_SLOTS);

// The texture table and the accumulator's texture bound are one number — the
// band width — reached from two directions. `apply_binds` keeps a slot this
// table cannot hold, or this table holds one no bind record can reach, if they
// part.
const _: () =
    assert!(REIMS_VGPU_METAL_MAX_TEXTURES as u32 == crate::runtime::draw::MAX_TEXTURE_BIND_SLOTS);

// This backend's two band bases are mirrors of `runtime::spirv_bind`'s, which is
// where the widening that set them is written. A mirror that drifts would have
// the two arms encode the same guest bind as two different descriptor bindings,
// and nothing else in the toolchain compares them.
const _: () =
    assert!(REIMS_VGPU_BINDING_TEXTURE_BASE == crate::runtime::spirv_bind::TEXTURE_BINDING_BASE);
const _: () =
    assert!(REIMS_VGPU_BINDING_SAMPLER_BASE == crate::runtime::spirv_bind::SAMPLER_BINDING_BASE);
