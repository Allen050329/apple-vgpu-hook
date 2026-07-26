//! Exact color occupancy gathered while copying mapped readback bytes.

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Color8ContentStats {
    pub rgb_nz: usize,
    pub rgb_max: u8,
    pub alpha_nz: usize,
    pub alpha_opaque: usize,
}

fn copy_color8_with_stats_scalar(src: &[u8], dst: &mut [u8]) -> Color8ContentStats {
    let mut stats = Color8ContentStats::default();
    for (src_pixel, dst_pixel) in src.chunks_exact(4).zip(dst.chunks_exact_mut(4)) {
        dst_pixel.copy_from_slice(src_pixel);
        let rgb_max = src_pixel[0].max(src_pixel[1]).max(src_pixel[2]);
        stats.rgb_nz += usize::from(rgb_max != 0);
        stats.rgb_max = stats.rgb_max.max(rgb_max);
        stats.alpha_nz += usize::from(src_pixel[3] != 0);
        stats.alpha_opaque += usize::from(src_pixel[3] == u8::MAX);
    }
    stats
}

#[cfg(target_arch = "x86_64")]
unsafe fn copy_color8_with_stats_x86(src: &[u8], dst: &mut [u8]) -> Color8ContentStats {
    use std::arch::x86_64::*;

    const RGB_LANES_MASK: i32 = 0x00ff_ffff;
    const ALPHA_BYTE_BITS: u32 = 0x8888;

    let simd_len = src.len() & !15usize;
    let zero = _mm_setzero_si128();
    let opaque = _mm_set1_epi8(-1);
    let rgb_mask = _mm_set1_epi32(RGB_LANES_MASK);
    let mut max_rgb = zero;
    let mut stats = Color8ContentStats::default();
    for offset in (0..simd_len).step_by(16) {
        let pixels = _mm_loadu_si128(src.as_ptr().add(offset).cast());
        _mm_storeu_si128(dst.as_mut_ptr().add(offset).cast(), pixels);

        let rgb = _mm_and_si128(pixels, rgb_mask);
        max_rgb = _mm_max_epu8(max_rgb, rgb);
        let zero_rgb = _mm_movemask_ps(_mm_castsi128_ps(_mm_cmpeq_epi32(rgb, zero))) as u32;
        stats.rgb_nz += 4usize - zero_rgb.count_ones() as usize;

        let zero_bytes = _mm_movemask_epi8(_mm_cmpeq_epi8(pixels, zero)) as u32;
        stats.alpha_nz += 4usize - (zero_bytes & ALPHA_BYTE_BITS).count_ones() as usize;
        let opaque_bytes = _mm_movemask_epi8(_mm_cmpeq_epi8(pixels, opaque)) as u32;
        stats.alpha_opaque += (opaque_bytes & ALPHA_BYTE_BITS).count_ones() as usize;
    }
    let mut maxima = [0u8; 16];
    _mm_storeu_si128(maxima.as_mut_ptr().cast(), max_rgb);
    stats.rgb_max = maxima.into_iter().max().unwrap_or(0);

    let tail = copy_color8_with_stats_scalar(&src[simd_len..], &mut dst[simd_len..]);
    stats.rgb_nz += tail.rgb_nz;
    stats.rgb_max = stats.rgb_max.max(tail.rgb_max);
    stats.alpha_nz += tail.alpha_nz;
    stats.alpha_opaque += tail.alpha_opaque;
    stats
}

pub(super) fn copy_color8_with_stats(src: &[u8], dst: &mut [u8]) -> Color8ContentStats {
    assert_eq!(src.len(), dst.len(), "content copy length mismatch");
    assert_eq!(src.len() % 4, 0, "content copy is not whole RGBA8 pixels");
    #[cfg(target_arch = "x86_64")]
    unsafe {
        return copy_color8_with_stats_x86(src, dst);
    }
    #[cfg(not(target_arch = "x86_64"))]
    copy_color8_with_stats_scalar(src, dst)
}

pub(super) fn copy_color8_with_stats_to_vec(src: &[u8]) -> (Vec<u8>, Color8ContentStats) {
    assert_eq!(src.len() % 4, 0, "content copy is not whole RGBA8 pixels");
    let mut out = Box::<[u8]>::new_uninit_slice(src.len());
    // SAFETY: the temporary slice covers the allocation exactly. The copy
    // routine writes every byte because the asserted length is whole RGBA8
    // pixels, so assuming initialization after it returns is valid.
    let dst = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<u8>(), out.len()) };
    let stats = copy_color8_with_stats(src, dst);
    let out = unsafe { out.assume_init() }.into_vec();
    (out, stats)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fused_content_copy_matches_scalar_for_vector_and_tail_pixels() {
        let mut src = Vec::new();
        for i in 0..37u8 {
            src.extend_from_slice(&[
                i.wrapping_mul(17),
                i.wrapping_mul(29),
                i.wrapping_mul(43),
                if i % 3 == 0 { u8::MAX } else { i },
            ]);
        }
        let mut scalar = vec![0u8; src.len()];
        let expected = copy_color8_with_stats_scalar(&src, &mut scalar);
        let mut fused = vec![0u8; src.len()];
        let actual = copy_color8_with_stats(&src, &mut fused);
        assert_eq!(actual, expected);
        assert_eq!(fused, src);
        assert_eq!(scalar, src);

        let (allocated, allocated_stats) = copy_color8_with_stats_to_vec(&src);
        assert_eq!(allocated_stats, expected);
        assert_eq!(allocated, src);
    }
}
