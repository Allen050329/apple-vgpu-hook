//! Shared validation/helpers (error, lengths, binding bases).

use crate::backend::metal::abi::{
    REIMS_VGPU_BINDING_SAMPLER_BASE, REIMS_VGPU_BINDING_TEXTURE_BASE,
};
use crate::backend::metal::constants::{
    REIMS_VGPU_METAL_MAX_BUFFERS, REIMS_VGPU_METAL_MAX_SAMPLERS, REIMS_VGPU_METAL_MAX_TEXTURES,
};
use crate::backend::metal::error::write_err;
pub(crate) use crate::backend::metal::error::Status;
use std::os::raw::c_char;

pub type ErrOut<'a> = (*mut c_char, usize);

pub fn set_err(err: ErrOut<'_>, msg: impl AsRef<str>) {
    // SAFETY: `ErrOut` is the `(char *err, size_t err_cap)` pair the shim hands
    // in, and `reims_vgpu_qemu_abi.h` requires it to be null or valid for
    // `err_cap` bytes. `write_err` checks both null and a zero capacity itself,
    // so this is the one place the ABI's promise is taken at face value.
    unsafe { write_err(err.0, err.1, msg.as_ref()) };
}

pub fn clear_err(err: ErrOut<'_>) {
    if !err.0.is_null() && err.1 > 0 {
        unsafe {
            *err.0 = 0;
        }
    }
}

pub fn rgba_len(width: u32, height: u32) -> Option<usize> {
    if width == 0 || height == 0 {
        return None;
    }
    (width as u64)
        .checked_mul(height as u64)?
        .checked_mul(4)?
        .try_into()
        .ok()
}

pub fn image_len(width: u32, height: u32, bytes_per_pixel: usize) -> Option<usize> {
    if width == 0 || height == 0 || bytes_per_pixel == 0 {
        return None;
    }
    (width as u64)
        .checked_mul(height as u64)?
        .checked_mul(bytes_per_pixel as u64)?
        .try_into()
        .ok()
}

pub fn valid_buffer_binding(binding: u32) -> bool {
    (binding as usize) < REIMS_VGPU_METAL_MAX_BUFFERS
}

pub fn texture_index(binding: u32) -> Option<usize> {
    if binding < REIMS_VGPU_BINDING_TEXTURE_BASE {
        return None;
    }
    let raw = (binding - REIMS_VGPU_BINDING_TEXTURE_BASE) as usize;
    if raw >= REIMS_VGPU_METAL_MAX_TEXTURES {
        None
    } else {
        Some(raw)
    }
}

pub fn sampler_index(binding: u32) -> Option<usize> {
    if binding < REIMS_VGPU_BINDING_SAMPLER_BASE {
        return None;
    }
    let raw = (binding - REIMS_VGPU_BINDING_SAMPLER_BASE) as usize;
    if raw >= REIMS_VGPU_METAL_MAX_SAMPLERS {
        None
    } else {
        Some(raw)
    }
}

pub fn f32_from_bits(bits: u32) -> f32 {
    f32::from_bits(bits)
}

/// As-bytes view of a `repr(C)` value for content hashing (matches ObjC `sizeof`).
pub fn bytes_of<T>(v: &T) -> &[u8] {
    unsafe { std::slice::from_raw_parts((v as *const T).cast::<u8>(), std::mem::size_of::<T>()) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_lengths_reject_empty_geometry() {
        assert_eq!(rgba_len(2, 3), Some(24));
        assert_eq!(rgba_len(0, 3), None);
        assert_eq!(image_len(2, 3, 8), Some(48));
        assert_eq!(image_len(2, 3, 0), None);
    }

    #[test]
    fn binding_bands_accept_exactly_the_backend_capacity() {
        assert!(valid_buffer_binding(0));
        assert!(valid_buffer_binding(
            REIMS_VGPU_METAL_MAX_BUFFERS as u32 - 1
        ));
        assert!(!valid_buffer_binding(REIMS_VGPU_METAL_MAX_BUFFERS as u32));

        assert_eq!(texture_index(REIMS_VGPU_BINDING_TEXTURE_BASE), Some(0));
        assert_eq!(
            texture_index(
                REIMS_VGPU_BINDING_TEXTURE_BASE + REIMS_VGPU_METAL_MAX_TEXTURES as u32 - 1
            ),
            Some(REIMS_VGPU_METAL_MAX_TEXTURES - 1)
        );
        assert_eq!(
            texture_index(REIMS_VGPU_BINDING_TEXTURE_BASE + REIMS_VGPU_METAL_MAX_TEXTURES as u32),
            None
        );

        assert_eq!(sampler_index(REIMS_VGPU_BINDING_SAMPLER_BASE), Some(0));
        assert_eq!(
            sampler_index(REIMS_VGPU_BINDING_SAMPLER_BASE + REIMS_VGPU_METAL_MAX_SAMPLERS as u32),
            None
        );
    }

    #[test]
    fn float_bits_are_preserved_exactly() {
        for bits in [0, 1, f32::INFINITY.to_bits(), f32::NAN.to_bits(), u32::MAX] {
            assert_eq!(f32_from_bits(bits).to_bits(), bits);
        }
    }
}
