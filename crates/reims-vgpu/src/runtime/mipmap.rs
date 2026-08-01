//! `generateMipmaps` (blit opcode `0x133`) for multi-mip type-2/3 linear textures.
//!
//! Type-11 IOSurface textures are out of scope: Metal forbids mipmapped
//! IOSurface textures, so there is no legal multi-mip type-11 body to generate
//! into. Non-type-2/3 refs fail as missing/unsupported at resolve.
//!
//! Primary path: host Metal `generateMipmapsForTexture:` on a temporary Shared
//! multi-level texture (native guest pixel format). CPU box-filter remains as a
//! no-device fallback for filterable unorm formats that convert through RGBA8.
//! Single-level textures fail visibly (Metal rejects `mipmapLevelCount == 1`).

use crate::contract::pixel_format::{self, RGBA8_BPP};
use crate::model::DeviceState;
use crate::runtime::decode::resource::{
    decode_texture_descriptor, OBJECT_TYPE_TEXTURE, OBJECT_TYPE_TEXTURE_VARIANT,
    TEXTURE_MAX_MIP_LEVELS,
};
use crate::runtime::gva_mem;
use crate::runtime::host::{HostMemory, HostOps};
use crate::runtime::metal_draw::host_alloc_len;
use crate::runtime::objects;

#[cfg(all(feature = "backend-metal", target_os = "macos"))]
use crate::backend::metal::mipmap as metal_mip;

/// Outcome of a generateMipmaps attempt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MipmapStatus {
    Ok,
    /// Missing object, wrong kind, short descriptor, or zero base ref.
    MissingTexture,
    /// `mipmapLevelCount <= 1` — not a valid Metal no-op.
    SingleLevel,
    /// Level layouts incomplete / out of range / zero geometry.
    IncompleteLayout,
    /// Pixel format has no filterable Metal path (and no CPU fallback).
    UnsupportedFormat,
    /// Pathological size or host buffer cap.
    Capacity,
    /// Guest GVA read/write failed.
    GuestIo,
    /// The exact Metal-side check that rejected generation.
    #[cfg(all(feature = "backend-metal", target_os = "macos"))]
    Metal(metal_mip::MetalMipmapError),
}

impl crate::observe::Refusal for MipmapStatus {
    /// `Ok` is the only value that is not a refusal.
    ///
    /// Every other variant means a decoded `generateMipmaps` was not carried
    /// out, so the guest keeps a texture whose upper levels are stale or
    /// undefined — silently, before this. The dispatch site logged
    /// `st={st:?}` with no `reason=` field at all, so none of the seven was
    /// greppable and the Debug spelling was the only handle.
    fn refusal(&self) -> Option<&'static str> {
        Some(match self {
            Self::Ok => return None,
            Self::MissingTexture => "mipmap_missing_texture",
            Self::SingleLevel => "mipmap_single_level",
            Self::IncompleteLayout => "mipmap_incomplete_layout",
            Self::UnsupportedFormat => "mipmap_unsupported_format",
            Self::Capacity => "mipmap_capacity",
            Self::GuestIo => "mipmap_guest_io",
            #[cfg(all(feature = "backend-metal", target_os = "macos"))]
            Self::Metal(error) => return Some(crate::observe::Decline::slug(error)),
        })
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            #[cfg(all(feature = "backend-metal", target_os = "macos"))]
            Self::Metal(error) => crate::observe::Decline::fields(error),
            _ => Vec::new(),
        }
    }
}

/// Box-filter downsample of a tight RGBA8 image to `dst_w × dst_h`.
///
/// Each destination texel averages the source region that maps onto it under a
/// uniform grid (integer bounds). Dimensions must be non-zero; destination may
/// not exceed the source in either axis.
///
/// Kept as the no-device fallback and pure unit-test helper; product path prefers
/// Metal-filtered generation.
fn downsample_rgba8_box(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
) -> Option<Vec<u8>> {
    if src_w == 0 || src_h == 0 || dst_w == 0 || dst_h == 0 {
        return None;
    }
    if dst_w > src_w || dst_h > src_h {
        return None;
    }
    let src_need = (src_w as usize)
        .checked_mul(src_h as usize)?
        .checked_mul(RGBA8_BPP as usize)?;
    if src.len() < src_need {
        return None;
    }
    let dst_need = (dst_w as usize)
        .checked_mul(dst_h as usize)?
        .checked_mul(RGBA8_BPP as usize)?;
    let mut dst = vec![0u8; dst_need];
    for dy in 0..dst_h {
        let y0 = ((dy as u64) * (src_h as u64) / (dst_h as u64)) as u32;
        let y1 = (((dy as u64 + 1) * (src_h as u64) / (dst_h as u64)) as u32).max(y0 + 1);
        let y1 = y1.min(src_h);
        for dx in 0..dst_w {
            let x0 = ((dx as u64) * (src_w as u64) / (dst_w as u64)) as u32;
            let x1 = (((dx as u64 + 1) * (src_w as u64) / (dst_w as u64)) as u32).max(x0 + 1);
            let x1 = x1.min(src_w);
            let mut acc = [0u64; 4];
            let mut count = 0u64;
            for y in y0..y1 {
                let row = (y as usize) * (src_w as usize) * 4;
                for x in x0..x1 {
                    let o = row + (x as usize) * 4;
                    acc[0] += src[o] as u64;
                    acc[1] += src[o + 1] as u64;
                    acc[2] += src[o + 2] as u64;
                    acc[3] += src[o + 3] as u64;
                    count += 1;
                }
            }
            if count == 0 {
                return None;
            }
            let out_o = ((dy as usize) * (dst_w as usize) + (dx as usize)) * 4;
            // Round half-up: (sum + count/2) / count.
            for c in 0..4 {
                dst[out_o + c] = ((acc[c] + count / 2) / count) as u8;
            }
        }
    }
    Some(dst)
}

struct ResolvedTexture {
    tex: crate::runtime::decode::resource::TextureDescriptor,
    levels: usize,
    fmt: u16,
}

fn resolve_multi_mip_texture<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    texture_ref: u32,
) -> Result<ResolvedTexture, MipmapStatus> {
    if texture_ref == 0 {
        return Err(MipmapStatus::MissingTexture);
    }
    let Some(entry) = objects::lookup_list_entry(state, host, task_id, texture_ref) else {
        return Err(MipmapStatus::MissingTexture);
    };
    if entry.object_type != OBJECT_TYPE_TEXTURE && entry.object_type != OBJECT_TYPE_TEXTURE_VARIANT
    {
        return Err(MipmapStatus::MissingTexture);
    }
    let Some(desc_bytes) = objects::read_descriptor(state, host, task_id, &entry) else {
        return Err(MipmapStatus::MissingTexture);
    };
    let tex = match decode_texture_descriptor(&desc_bytes) {
        Ok(t) => t,
        Err(e) => {
            // Not missing — malformed. `MipmapStatus::MissingTexture` is the
            // coarse class four checks above already answer with, so without
            // this line a truncated descriptor and an unbound ref reach the
            // sink wearing the same name.
            crate::observe::Emit::decline("mipmap_texture_desc", &e)
                .field("task", task_id)
                .field("tex", texture_ref)
                .field("len", desc_bytes.len())
                .fail_once(texture_ref as u64);
            return Err(MipmapStatus::MissingTexture);
        }
    };
    if !tex.has_pixel_format {
        return Err(MipmapStatus::UnsupportedFormat);
    }
    let levels = if tex.mipmap_level_count > 0 {
        tex.mipmap_level_count as usize
    } else {
        1
    };
    if levels <= 1 {
        return Err(MipmapStatus::SingleLevel);
    }
    if levels > TEXTURE_MAX_MIP_LEVELS || tex.levels.len() < levels {
        return Err(MipmapStatus::IncompleteLayout);
    }
    let fmt = tex.pixel_format;
    if pixel_format::bytes_per_pixel(fmt).is_none() {
        return Err(MipmapStatus::UnsupportedFormat);
    }
    // Guest level sizes must match Metal's standard pyramid.
    let l0 = tex.levels[0];
    for level in 0..levels {
        let layout = &tex.levels[level];
        let exp_w = metal_mip_extent_local(l0.width, level as u32);
        let exp_h = metal_mip_extent_local(l0.height, level as u32);
        if layout.width != exp_w
            || layout.height != exp_h
            || layout.width == 0
            || layout.height == 0
        {
            return Err(MipmapStatus::IncompleteLayout);
        }
    }
    Ok(ResolvedTexture { tex, levels, fmt })
}

fn metal_mip_extent_local(base: u32, level: u32) -> u32 {
    #[cfg(all(feature = "backend-metal", target_os = "macos"))]
    {
        metal_mip::metal_mip_extent(base, level)
    }
    #[cfg(feature = "backend-vulkan")]
    {
        if base == 0 {
            0
        } else {
            (base >> level).max(1)
        }
    }
}

/// Load one level as tightly packed native-format bytes.
fn load_level_tight_native<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    tex: &crate::runtime::decode::resource::TextureDescriptor,
    fmt: u16,
    level: u32,
) -> Result<Vec<u8>, MipmapStatus> {
    let Some((gva, layout)) = tex.level_gva(level, state.page_shift) else {
        return Err(MipmapStatus::IncompleteLayout);
    };
    let w = layout.width;
    let h = layout.height;
    let bpr = layout.row_stride;
    if bpr > u32::MAX as u64 {
        return Err(MipmapStatus::Capacity);
    }
    let Some(tight_row) = pixel_format::tight_row_bytes(w, fmt) else {
        return Err(MipmapStatus::UnsupportedFormat);
    };
    if (bpr as u32) < tight_row {
        return Err(MipmapStatus::IncompleteLayout);
    }
    let need = match (tight_row as u64).checked_mul(h as u64) {
        Some(v) => v,
        None => return Err(MipmapStatus::Capacity),
    };
    let need = host_alloc_len(need).ok_or(MipmapStatus::Capacity)?;
    // Row-by-row of `tight_row` bytes below, so the bound is the extent read
    // rather than `bpr * h` — see `TextureLevelLayout::read_span`.
    let span = layout.read_span(tight_row).ok_or(MipmapStatus::Capacity)?;
    if tex.allocation_size != 0 && layout.offset.saturating_add(span) > tex.allocation_size {
        return Err(MipmapStatus::IncompleteLayout);
    }
    let mut out = vec![0u8; need];
    let mut row = vec![0u8; tight_row as usize];
    for y in 0..h {
        let Some(row_gva) = gva.checked_add((y as u64).saturating_mul(bpr)) else {
            return Err(MipmapStatus::GuestIo);
        };
        if gva_mem::read_task_gva_by_id(
            host,
            &state.tasks,
            task_id,
            row_gva,
            &mut row,
            state.page_shift,
        )
        .is_err()
        {
            return Err(MipmapStatus::GuestIo);
        }
        let dst_off = (y as usize) * (tight_row as usize);
        out[dst_off..dst_off + tight_row as usize].copy_from_slice(&row);
    }
    Ok(out)
}

/// Write tightly packed native rows into a decoded level layout.
fn store_level_tight_native<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    tex: &crate::runtime::decode::resource::TextureDescriptor,
    fmt: u16,
    level: u32,
    tight: &[u8],
) -> Result<(), MipmapStatus> {
    let Some((gva, layout)) = tex.level_gva(level, state.page_shift) else {
        return Err(MipmapStatus::IncompleteLayout);
    };
    let w = layout.width;
    let h = layout.height;
    let bpr = layout.row_stride;
    if bpr > u32::MAX as u64 {
        return Err(MipmapStatus::Capacity);
    }
    let Some(tight_row) = pixel_format::tight_row_bytes(w, fmt) else {
        return Err(MipmapStatus::UnsupportedFormat);
    };
    if (bpr as u32) < tight_row {
        return Err(MipmapStatus::IncompleteLayout);
    }
    let need = (tight_row as usize)
        .checked_mul(h as usize)
        .ok_or(MipmapStatus::Capacity)?;
    if tight.len() < need {
        return Err(MipmapStatus::IncompleteLayout);
    }
    // Same rule on the write side: each row writes `tight_row` bytes at
    // `gva + y * bpr`, so a trailing stride is never touched.
    let span = layout.read_span(tight_row).ok_or(MipmapStatus::Capacity)?;
    if tex.allocation_size != 0 && layout.offset.saturating_add(span) > tex.allocation_size {
        return Err(MipmapStatus::IncompleteLayout);
    }
    for y in 0..h {
        let src_off = (y as usize) * (tight_row as usize);
        let row = &tight[src_off..src_off + tight_row as usize];
        let Some(row_gva) = gva.checked_add((y as u64).saturating_mul(bpr)) else {
            return Err(MipmapStatus::GuestIo);
        };
        if gva_mem::write_task_gva_product(state, host, task_id, row_gva, row).is_err() {
            return Err(MipmapStatus::GuestIo);
        }
    }
    Ok(())
}

#[cfg(all(feature = "backend-metal", target_os = "macos"))]
fn generate_via_metal(
    fmt: u16,
    width: u32,
    height: u32,
    levels: u32,
    level0: &[u8],
) -> Result<Vec<(u32, u32, Vec<u8>)>, metal_mip::MetalMipmapError> {
    metal_mip::generate_mipmaps_filtered(fmt, width, height, levels, level0).map(|chain| {
        chain
            .into_iter()
            .map(|level| (level.width, level.height, level.tight_bytes))
            .collect()
    })
}

/// No-device fallback: RGBA8 box filter for formats with row conversion.
fn generate_via_box_filter(
    fmt: u16,
    width: u32,
    height: u32,
    levels: usize,
    level0_native: &[u8],
) -> Result<Vec<(u32, u32, Vec<u8>)>, MipmapStatus> {
    // Only formats that convert through RGBA8 (unorm 8-bit family).
    let Some(tight0) = pixel_format::tight_row_bytes(width, fmt) else {
        return Err(MipmapStatus::UnsupportedFormat);
    };
    let need0 = (tight0 as usize)
        .checked_mul(height as usize)
        .ok_or(MipmapStatus::Capacity)?;
    if level0_native.len() < need0 {
        return Err(MipmapStatus::Capacity);
    }
    // Convert L0 native → tight RGBA8.
    let rgba_need = (width as usize)
        .checked_mul(height as usize)
        .and_then(|v| v.checked_mul(RGBA8_BPP as usize))
        .ok_or(MipmapStatus::Capacity)?;
    let mut prev = vec![0u8; rgba_need];
    for y in 0..height {
        let src_off = (y as usize) * (tight0 as usize);
        let dst_off = (y as usize) * (width as usize) * 4;
        if !pixel_format::convert_row_to_rgba8(
            fmt,
            &level0_native[src_off..src_off + tight0 as usize],
            width,
            &mut prev[dst_off..],
        ) {
            return Err(MipmapStatus::UnsupportedFormat);
        }
    }
    let mut out = Vec::with_capacity(levels);
    // Level 0 native copy.
    out.push((width, height, level0_native[..need0].to_vec()));
    let mut prev_w = width;
    let mut prev_h = height;
    for level in 1..levels {
        let dw = metal_mip_extent_local(width, level as u32);
        let dh = metal_mip_extent_local(height, level as u32);
        let Some(next_rgba) = downsample_rgba8_box(&prev, prev_w, prev_h, dw, dh) else {
            return Err(MipmapStatus::IncompleteLayout);
        };
        let Some(tight) = pixel_format::tight_row_bytes(dw, fmt) else {
            return Err(MipmapStatus::UnsupportedFormat);
        };
        let need = (tight as usize)
            .checked_mul(dh as usize)
            .ok_or(MipmapStatus::Capacity)?;
        let mut native = vec![0u8; need];
        for y in 0..dh {
            let src_off = (y as usize) * (dw as usize) * 4;
            let dst_off = (y as usize) * (tight as usize);
            if !pixel_format::convert_rgba8_to_row(
                fmt,
                &next_rgba[src_off..],
                dw,
                &mut native[dst_off..dst_off + tight as usize],
            ) {
                return Err(MipmapStatus::UnsupportedFormat);
            }
        }
        out.push((dw, dh, native));
        prev = next_rgba;
        prev_w = dw;
        prev_h = dh;
    }
    Ok(out)
}

/// Execute blit `0x133 generateMipmaps` for a type-2/3 multi-mip linear texture.
pub fn generate_mipmaps_linear<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    texture_ref: u32,
) -> MipmapStatus {
    let resolved = match resolve_multi_mip_texture(state, host, task_id, texture_ref) {
        Ok(r) => r,
        Err(st) => return st,
    };
    let ResolvedTexture { tex, levels, fmt } = resolved;
    let l0_w = tex.levels[0].width;
    let l0_h = tex.levels[0].height;

    let level0 = match load_level_tight_native(state, host, task_id, &tex, fmt, 0) {
        Ok(v) => v,
        Err(st) => return st,
    };

    // Prefer Metal-filtered generation in the guest's native pixel format.
    #[cfg(all(feature = "backend-metal", target_os = "macos"))]
    let chain = match generate_via_metal(fmt, l0_w, l0_h, levels as u32, &level0) {
        Ok(chain) => chain,
        Err(error @ metal_mip::MetalMipmapError::NoDevice) => {
            // Correct but slower: retain the CPU box-filter fallback, and make
            // the missing Metal device visible as a typed degradation.
            crate::observe::Emit::decline("mipmap_metal_fallback", &error)
                .field("texture", texture_ref)
                .field("format", format!("{fmt:#x}"))
                .field("width", l0_w)
                .field("height", l0_h)
                .off();
            // Soft fallback only when no MTL device is available.
            match generate_via_box_filter(fmt, l0_w, l0_h, levels, &level0) {
                Ok(c) => c,
                Err(st) => return st,
            }
        }
        Err(error) => return MipmapStatus::Metal(error),
    };
    #[cfg(feature = "backend-vulkan")]
    let chain = match generate_via_box_filter(fmt, l0_w, l0_h, levels, &level0) {
        Ok(c) => c,
        Err(st) => return st,
    };

    if chain.len() != levels {
        return MipmapStatus::IncompleteLayout;
    }
    // Write levels 1.. only (L0 is source).
    for (level, (w, h, tight)) in chain.iter().enumerate().skip(1) {
        let layout = &tex.levels[level];
        if *w != layout.width || *h != layout.height {
            return MipmapStatus::IncompleteLayout;
        }
        if let Err(st) =
            store_level_tight_native(state, host, task_id, &tex, fmt, level as u32, tight)
        {
            return st;
        }
    }
    MipmapStatus::Ok
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::pixel_format::MTL_FORMAT_RGBA8_UNORM;
    use crate::observe::Refusal;

    /// `Ok` must produce no line, every other outcome must produce a distinct
    /// one. Before this the dispatch site logged `st={st:?}` with no `reason=`
    /// field, so none of the eight was greppable.
    #[test]
    fn every_mipmap_outcome_but_ok_names_its_own_check() {
        const ALL: &[MipmapStatus] = &[
            MipmapStatus::Ok,
            MipmapStatus::MissingTexture,
            MipmapStatus::SingleLevel,
            MipmapStatus::IncompleteLayout,
            MipmapStatus::UnsupportedFormat,
            MipmapStatus::Capacity,
            MipmapStatus::GuestIo,
        ];
        assert_eq!(MipmapStatus::Ok.refusal(), None, "Ok is not a refusal");
        let mut slugs: Vec<&str> = ALL.iter().filter_map(|s| s.refusal()).collect();
        assert_eq!(slugs.len(), ALL.len() - 1, "every non-Ok outcome refuses");
        slugs.sort_unstable();
        let n = slugs.len();
        slugs.dedup();
        assert_eq!(slugs.len(), n, "two mipmap outcomes share a slug");

        #[cfg(all(feature = "backend-metal", target_os = "macos"))]
        {
            let status = MipmapStatus::Metal(metal_mip::MetalMipmapError::Level0TooShort {
                len: 15,
                expected: 16,
            });
            assert_eq!(status.refusal(), Some("metal_mipmap_level0_too_short"));
            assert_eq!(
                status.fields(),
                vec![("len", "15".to_string()), ("expected", "16".to_string())],
                "the runtime wrapper must retain the Metal leaf's structured facts"
            );
        }
    }

    #[test]
    fn box_filter_2x2_to_1x1() {
        // Four pixels: average (10,20,30,40)… → mean.
        let src = [
            10u8, 20, 30, 40, //
            20, 40, 60, 80, //
            30, 60, 90, 120, //
            40, 80, 120, 160,
        ];
        let out = downsample_rgba8_box(&src, 2, 2, 1, 1).unwrap();
        // (10+20+30+40)/4=25, (20+40+60+80)/4=50, (30+60+90+120)/4=75, (40+80+120+160)/4=100
        assert_eq!(out, vec![25, 50, 75, 100]);
    }

    #[test]
    fn box_filter_power_of_two_chain() {
        let mut src = vec![0u8; 4 * 4 * 4];
        for i in 0..16 {
            let o = i * 4;
            src[o] = 255;
            src[o + 3] = 255;
        }
        let mid = downsample_rgba8_box(&src, 4, 4, 2, 2).unwrap();
        assert_eq!(mid.len(), 2 * 2 * 4);
        assert_eq!(&mid[0..4], &[255, 0, 0, 255]);
        let lo = downsample_rgba8_box(&mid, 2, 2, 1, 1).unwrap();
        assert_eq!(lo, vec![255, 0, 0, 255]);
    }

    #[test]
    fn box_filter_rejects_upsized() {
        let src = [1u8, 2, 3, 4];
        assert!(downsample_rgba8_box(&src, 1, 1, 2, 2).is_none());
    }

    #[test]
    fn format_roundtrip_helpers_exist() {
        // Ensure convert path used by generate is available for RGBA8.
        let row = [1u8, 2, 3, 4, 5, 6, 7, 8];
        let mut rgba = [0u8; 8];
        assert!(pixel_format::convert_row_to_rgba8(
            MTL_FORMAT_RGBA8_UNORM,
            &row,
            2,
            &mut rgba
        ));
        let mut back = [0u8; 8];
        assert!(pixel_format::convert_rgba8_to_row(
            MTL_FORMAT_RGBA8_UNORM,
            &rgba,
            2,
            &mut back
        ));
        assert_eq!(back, row);
    }

    #[test]
    fn box_filter_chain_native_rgba8() {
        let w = 4u32;
        let h = 4u32;
        let mut l0 = vec![0u8; (w * h * 4) as usize];
        for px in l0.chunks_exact_mut(4) {
            px[0] = 100;
            px[1] = 150;
            px[2] = 200;
            px[3] = 255;
        }
        let chain = generate_via_box_filter(MTL_FORMAT_RGBA8_UNORM, w, h, 3, &l0).unwrap();
        assert_eq!(chain.len(), 3);
        assert_eq!(chain[2].0, 1);
        assert_eq!(chain[2].1, 1);
        assert_eq!(chain[2].2, vec![100, 150, 200, 255]);
    }

    /// Regression guard for `metal_mip_extent_local` (the Vulkan-build mip
    /// sizing formula). This computes each level's expected dimension and the
    /// resolver rejects any layout whose stored extent disagrees, so a wrong
    /// formula either falsely rejects valid mip chains (IncompleteLayout) or
    /// accepts a mismatched layout that then samples out of bounds. Lock the
    /// Metal contract `max(1, base >> level)` with the base==0 degenerate case.
    #[test]
    fn metal_mip_extent_local_halves_and_floors_at_one() {
        // Degenerate base stays zero (an empty axis has no levels).
        assert_eq!(metal_mip_extent_local(0, 0), 0);
        assert_eq!(metal_mip_extent_local(0, 3), 0);

        // Power-of-two base halves each level and floors at 1, never 0.
        assert_eq!(metal_mip_extent_local(8, 0), 8);
        assert_eq!(metal_mip_extent_local(8, 1), 4);
        assert_eq!(metal_mip_extent_local(8, 2), 2);
        assert_eq!(metal_mip_extent_local(8, 3), 1);
        assert_eq!(
            metal_mip_extent_local(8, 4),
            1,
            "past the last level clamps to 1"
        );
        assert_eq!(
            metal_mip_extent_local(8, 20),
            1,
            "huge level never underflows to 0"
        );

        // Non-power-of-two base uses integer right-shift (floor), matching Metal.
        assert_eq!(metal_mip_extent_local(100, 1), 50);
        assert_eq!(metal_mip_extent_local(100, 2), 25);
        assert_eq!(metal_mip_extent_local(100, 3), 12);
    }
}
