//! Write host BGRA8 into a guest IOSurface mapping (render writeback).
//!
//! Product writes go **only** through a revalidated contiguous HostOps view
//! (`map_pages`) — never `write_gpa` fragment walks over cached PFNs (freelist
//! `0xff000000ff000000` class). Always bumps [`DeviceState::mark_mapping_written`]
//! on success.

use crate::contract::iosurface_pages::{sample_window_prefer_device, DEVICE_DESC_LEN};
use crate::contract::pixel_format::{
    self, convert_rgba8_to_row, convert_row_to_rgba8, MTL_FORMAT_BGRA8_UNORM, RGBA8_BPP,
};
use crate::model::{DeviceState, MappingEntry, MAX_SCANOUT_DIM};
use crate::runtime::host::{HostMemory, HostOps};
use crate::runtime::mapper;

/// Resolve sample window for a type-11 texture binding on a mapping.
///
/// Prefers the guest `sIOSurfaceDeviceDescriptor` when cached: single-plane
/// uses surface-level base/bpr; multi-plane selects the unique plane whose
/// dims and bpe match the texture (`sample_window_prefer_device`). Falls back
/// to packed invent `ALIGN_UP(w×bpp, 128)` when the descriptor is missing or
/// rejects. Returns `(surface_offset, bytes_per_row, span_end)`.
///
/// # The invent fallback is reported
///
/// Type-11 is the case with **no wire plane index** — unlike
/// [`type5_sample_window`], nothing on the wire names which plane the texture
/// wants, so a multi-plane surface is resolved by matching width, height and
/// bytes-per-element. That scan takes the plane only when **exactly one**
/// matches; zero matches and two-or-more matches both fall through to the
/// invented packed window, which is plane 0's bytes at offset 0. On a
/// multi-plane surface that is a bind of the wrong plane, and it is the case the
/// geometry scan cannot detect by construction.
///
/// So the fallback is not silent. `type11_window_invent` is emitted through the
/// always-on channel, deduped per (mapping, geometry, format), carrying the
/// surface's plane count so a reader can tell "no descriptor yet" (plane_count
/// unknown) from "the scan could not pick a plane" (plane_count > 1). The three
/// `mapper.rs` callers of `sample_window_prefer_device` legitimately ignore this
/// — they want `span_end` only, as a floor on how many pages to map.
pub fn type11_sample_window(
    m: &MappingEntry,
    mapping_id: u32,
    width: u32,
    height: u32,
    format: u16,
) -> Option<(u64, u32, u64)> {
    let desc = if m.device_desc.len() >= DEVICE_DESC_LEN {
        Some(m.device_desc.as_slice())
    } else {
        None
    };
    let (offset, bpr, end, from_device) =
        sample_window_prefer_device(desc, None, format, width, height)?;
    if !from_device
        && crate::observe::first_sight(
            "type11_window_invent",
            u64::from(mapping_id) << 48
                | u64::from(width) << 32
                | u64::from(height) << 16
                | u64::from(format),
        )
    {
        let planes = desc
            .and_then(crate::contract::iosurface_pages::decode_device_surface)
            .map(|s| i64::from(s.plane_count))
            .unwrap_or(-1);
        crate::observe::off(format!(
            "type11_window_invent mapping={mapping_id} {width}x{height} fmt={format:#x} \
             planes={planes} offset={offset} bpr={bpr} span_end={end} (no wire plane index and \
             the device descriptor did not resolve one; this window is plane 0 packed)"
        ));
    }
    Some((offset, bpr, end))
}

/// Sample window for a type-5 serialized view, which — unlike type-11 —
/// carries the IOSurface plane index on the wire (type-5 record `+0x20`).
/// The index names the device plane record directly; same-geometry planes
/// (v0a8 Y plane 0 vs alpha plane 2) are indistinguishable by geometry scan.
pub fn type5_sample_window(
    m: &MappingEntry,
    plane_index: u32,
    width: u32,
    height: u32,
    format: u16,
) -> Option<(u64, u32, u64, bool)> {
    let desc = if m.device_desc.len() >= DEVICE_DESC_LEN {
        Some(m.device_desc.as_slice())
    } else {
        None
    };
    sample_window_prefer_device(desc, Some(plane_index), format, width, height)
}

/// Revalidate + packed contig host view covering at least `span_end` bytes.
///
/// Returns `None` when the mapping is fragmented on Linux (use
/// [`mapper::write_mapping_bytes`] / [`mapper::read_mapping_bytes`]).
fn contig_for_span<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    mapping_id: u32,
    span_end: u64,
) -> Option<(usize, usize)> {
    let (ptr, len) = mapper::ensure_contig_view(state, host, mapping_id)?;
    if (len as u64) < span_end {
        crate::observe::fail(format!(
            "mapping_write contig mid={mapping_id} reason=short_view len={len} need={span_end}"
        ));
        return None;
    }
    Some((ptr, len))
}

/// One past the last mapping byte a rect transfer touches: the last texel of its
/// last row, at `bpr` pitch, `x_off` bytes into the row.
///
/// Both the raw-pointer read and the raw-pointer write below must compare this
/// against `span_end`, because `contig_for_span` guarantees the view covers
/// `span_end` and nothing more — past it a read takes unrelated QEMU heap and a
/// write smashes unrelated guest pages, both trace-lessly. Written once because
/// duplicated arithmetic is the only reason the two sides could disagree, and
/// they did: the write side was hardened for this bound and the read side
/// shipped without it. Each caller still names its own slug — `read_overrun` and
/// `writeback_overrun` are different losses.
fn rect_extent_end(
    base_off: u64,
    origin_y: u32,
    height: u32,
    bpr: usize,
    x_off: u64,
    rb: usize,
) -> u64 {
    base_off
        .saturating_add(
            (origin_y as u64)
                .saturating_add(height as u64)
                .saturating_sub(1)
                .saturating_mul(bpr as u64),
        )
        .saturating_add(x_off)
        .saturating_add(rb as u64)
}

/// Write a tight BGRA8 image into the mapping's guest pages.
///
/// Packed contig HostOps view when possible; else multi-import maximal packed
/// page runs ([`mapper::write_mapping_bytes`]). Never `write_gpa`.
///
/// `src` is row-major BGRA8 with `src_stride` bytes/row. Geometry must match
/// the latched mapping size (or width/height args when has_geom is set).
pub fn write_bgra8<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mapping_id: u32,
    src: &[u8],
    src_stride: u32,
    width: u32,
    height: u32,
) -> bool {
    if width == 0
        || height == 0
        || width > MAX_SCANOUT_DIM
        || height > MAX_SCANOUT_DIM
        || src_stride < width.saturating_mul(RGBA8_BPP)
    {
        return false;
    }
    let Some(m) = state.mappings.get(&mapping_id) else {
        return false;
    };
    if !m.mapped || m.page_entries.is_empty() {
        return false;
    }
    let (mw, mh, format) = if m.has_geom {
        (
            m.width,
            m.height,
            if m.format != 0 {
                m.format
            } else {
                MTL_FORMAT_BGRA8_UNORM
            },
        )
    } else {
        (width, height, MTL_FORMAT_BGRA8_UNORM)
    };
    if mw != width || mh != height {
        return false;
    }
    let Some((base_off, bpr_u32, span_end)) = type11_sample_window(m, mapping_id, mw, mh, format)
    else {
        return false;
    };
    // Deferred-writeback flush-on-access: land pending resident content in
    // these pages before touching them.
    crate::runtime::storage_flush::flush_intersecting(state, host, mapping_id, base_off, span_end);
    let bpr = bpr_u32 as usize;
    let Some(tight) = pixel_format::tight_row_bytes(mw, format) else {
        return false;
    };
    let tight = tight as usize;

    let mut row = vec![0u8; tight];
    let mut rgba = if format == MTL_FORMAT_BGRA8_UNORM
        || format == pixel_format::MTL_FORMAT_BGRA8_UNORM_SRGB
    {
        None
    } else {
        Some(vec![0u8; (mw as usize) * (RGBA8_BPP as usize)])
    };

    // Fast path: one packed view, poke rows in place.
    if let Some((ptr, _)) = contig_for_span(state, host, mapping_id, span_end) {
        // SAFETY: contig covers span_end; revalidated in ensure_contig_view.
        let base = unsafe { (ptr as *mut u8).add(base_off as usize) };
        for y in 0..mh {
            let src_off = (y as usize) * (src_stride as usize);
            let src_row_len = (mw as usize) * (RGBA8_BPP as usize);
            if src_off + src_row_len > src.len() {
                return false;
            }
            let src_row = &src[src_off..src_off + src_row_len];
            if let Some(ref mut rgba_row) = rgba {
                if !convert_row_to_rgba8(MTL_FORMAT_BGRA8_UNORM, src_row, mw, rgba_row) {
                    return false;
                }
                if !convert_rgba8_to_row(format, rgba_row, mw, &mut row) {
                    return false;
                }
            } else {
                let n = src_row_len.min(row.len());
                row[..n].copy_from_slice(&src_row[..n]);
            }
            let dst = unsafe { base.add((y as usize).saturating_mul(bpr)) };
            unsafe {
                std::ptr::copy_nonoverlapping(row.as_ptr(), dst, tight);
            }
        }
    } else {
        // Fragmented: stage native rows then multi-import (one map_pages pass set).
        // The sample window ends at the final row's last texel, not at
        // `bpr * height`; padding after the final row is outside the texture
        // contract and may belong to another guest allocation.
        let Some(frame_len) = (mh as usize)
            .checked_sub(1)
            .and_then(|rows| bpr.checked_mul(rows))
            .and_then(|prefix| prefix.checked_add(tight))
        else {
            return false;
        };
        let mut frame = vec![0u8; frame_len];
        for y in 0..mh {
            let src_off = (y as usize) * (src_stride as usize);
            let src_row_len = (mw as usize) * (RGBA8_BPP as usize);
            if src_off + src_row_len > src.len() {
                return false;
            }
            let src_row = &src[src_off..src_off + src_row_len];
            if let Some(ref mut rgba_row) = rgba {
                if !convert_row_to_rgba8(MTL_FORMAT_BGRA8_UNORM, src_row, mw, rgba_row) {
                    return false;
                }
                if !convert_rgba8_to_row(format, rgba_row, mw, &mut row) {
                    return false;
                }
            } else {
                let n = src_row_len.min(row.len());
                row[..n].copy_from_slice(&src_row[..n]);
            }
            let dst_off = (y as usize).saturating_mul(bpr);
            if dst_off + tight > frame.len() {
                return false;
            }
            frame[dst_off..dst_off + tight].copy_from_slice(&row[..tight]);
        }
        if !mapper::write_mapping_bytes(state, host, mapping_id, base_off, &frame) {
            return false;
        }
    }
    state.invalidate_storage_residency_window(mapping_id, base_off, span_end);
    let _ = state.mark_mapping_written(mapping_id);
    let mut cache = vec![0u8; (mw as usize).saturating_mul(mh as usize).saturating_mul(4)];
    let row_src = width.saturating_mul(RGBA8_BPP) as usize;
    for y in 0..mh as usize {
        let so = y * (src_stride as usize);
        let doff = y * row_src;
        if so + row_src <= src.len() && doff + row_src <= cache.len() {
            cache[doff..doff + row_src].copy_from_slice(&src[so..so + row_src]);
        }
    }
    crate::runtime::surface_cache::store(state, mapping_id, mw, mh, cache);
    // This write just made the host copy and the guest pages agree, so it is the
    // moment the copy's currency can be pinned. Nothing else arms this mapping:
    // the type-4 sampled ladder's first census read `gw_no_stamp` 14 092 against
    // `gw_clean` 0 because only the Vulkan Store rails ever stamped, and the
    // copy that rung serves is written here. Unstamped, the reader cannot tell a
    // surface the guest has rewritten from one it has not, and must assume the
    // worst on every bind.
    crate::runtime::mapper::stamp_guest_write_gen(state, host, mapping_id);
    true
}

/// Write a tight RGBA8 image into a type-11 mapping, optionally as changed-spans.
///
/// Archive `apple_pv_gpu_write_type11_image_changed`: when `seed_rgba` is present
/// (same layout as `rgba`), only contiguous native-format spans that differ from
/// the seed are written. Equivalent to a full `storeAction=Store` when the seed
/// was the Metal Load attachment content (unchanged texels match guest), without
/// rewriting multi-MiB of identical bytes on every damage pass. `seed_rgba = None`
/// always writes every row (Clear / multi-draw final / force-full).
pub fn write_rgba8_image_changed<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mapping_id: u32,
    rgba: &[u8],
    seed_rgba: Option<&[u8]>,
    width: u32,
    height: u32,
) -> bool {
    if width == 0 || height == 0 || width > MAX_SCANOUT_DIM || height > MAX_SCANOUT_DIM {
        return false;
    }
    let rgba_stride = width.saturating_mul(RGBA8_BPP);
    let need = (height as usize).saturating_mul(rgba_stride as usize);
    if rgba.len() < need {
        return false;
    }
    if let Some(seed) = seed_rgba {
        if seed.len() < need {
            return false;
        }
    }
    let Some(m) = state.mappings.get(&mapping_id) else {
        return false;
    };
    if !m.mapped || m.page_entries.is_empty() {
        return false;
    }
    let (mw, mh, format) = if m.has_geom {
        (
            m.width,
            m.height,
            if m.format != 0 {
                m.format
            } else {
                MTL_FORMAT_BGRA8_UNORM
            },
        )
    } else {
        (width, height, MTL_FORMAT_BGRA8_UNORM)
    };
    if mw != width || mh != height {
        return false;
    }
    let Some((base_off, bpr_u32, span_end)) = type11_sample_window(m, mapping_id, mw, mh, format)
    else {
        return false;
    };
    let bpr = bpr_u32 as u64;
    let Some(tight) = pixel_format::tight_row_bytes(mw, format) else {
        return false;
    };
    let bpr_usize = bpr as usize;
    let tight = tight as usize;
    let mut native = vec![0u8; tight];
    let mut seed_native = vec![0u8; tight];
    let contig = contig_for_span(state, host, mapping_id, span_end);
    // SAFETY: when Some, contig covers span_end.
    let base = contig.map(|(ptr, _)| unsafe { (ptr as *mut u8).add(base_off as usize) });
    for y in 0..mh as usize {
        let src_off = y * rgba_stride as usize;
        let src_row = &rgba[src_off..src_off + rgba_stride as usize];
        if !rgba8_row_to_native(format, src_row, mw, &mut native) {
            return false;
        }
        let seed_row = if let Some(seed) = seed_rgba {
            let s = &seed[src_off..src_off + rgba_stride as usize];
            if !rgba8_row_to_native(format, s, mw, &mut seed_native) {
                return false;
            }
            Some(seed_native.as_slice())
        } else {
            None
        };
        if let Some(srow) = seed_row {
            if srow == native.as_slice() {
                continue;
            }
        }
        let row_moff = base_off.saturating_add((y as u64).saturating_mul(bpr));
        if let Some(base) = base {
            let dst = unsafe { base.add(y.saturating_mul(bpr_usize)) };
            if let Some(seed) = seed_row {
                // Changed spans only within the row.
                let mut x = 0usize;
                while x < tight {
                    while x < tight && native[x] == seed[x] {
                        x += 1;
                    }
                    if x >= tight {
                        break;
                    }
                    let start = x;
                    while x < tight && native[x] != seed[x] {
                        x += 1;
                    }
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            native.as_ptr().add(start),
                            dst.add(start),
                            x - start,
                        );
                    }
                }
            } else {
                unsafe {
                    std::ptr::copy_nonoverlapping(native.as_ptr(), dst, tight);
                }
            }
        } else if let Some(seed) = seed_row {
            let mut x = 0usize;
            while x < tight {
                while x < tight && native[x] == seed[x] {
                    x += 1;
                }
                if x >= tight {
                    break;
                }
                let start = x;
                while x < tight && native[x] != seed[x] {
                    x += 1;
                }
                if !mapper::write_mapping_bytes(
                    state,
                    host,
                    mapping_id,
                    row_moff.saturating_add(start as u64),
                    &native[start..x],
                ) {
                    return false;
                }
            }
        } else if !mapper::write_mapping_bytes(state, host, mapping_id, row_moff, &native) {
            return false;
        }
    }
    state.invalidate_storage_residency_window(mapping_id, base_off, span_end);
    let _ = state.mark_mapping_written(mapping_id);
    // Host render-cache (Linux §8.5): full-frame BGRA from the Store rgba.
    let mut cache = vec![0u8; need];
    for y in 0..mh as usize {
        let so = y * rgba_stride as usize;
        let doff = y * rgba_stride as usize;
        let src_row = &rgba[so..so + rgba_stride as usize];
        // rgba → bgra for host cache (same as write_bgra8 source convention).
        for x in 0..mw as usize {
            let i = x * 4;
            cache[doff + i] = src_row[i + 2];
            cache[doff + i + 1] = src_row[i + 1];
            cache[doff + i + 2] = src_row[i];
            cache[doff + i + 3] = src_row[i + 3];
        }
    }
    crate::runtime::surface_cache::store(state, mapping_id, mw, mh, cache);
    // This write just made the host copy and the guest pages agree, so it is the
    // moment the copy's currency can be pinned. Nothing else arms this mapping:
    // the type-4 sampled ladder's first census read `gw_no_stamp` 14 092 against
    // `gw_clean` 0 because only the Vulkan Store rails ever stamped, and the
    // copy that rung serves is written here. Unstamped, the reader cannot tell a
    // surface the guest has rewritten from one it has not, and must assume the
    // worst on every bind.
    crate::runtime::mapper::stamp_guest_write_gen(state, host, mapping_id);
    true
}

fn rgba8_row_to_native(format: u16, rgba_row: &[u8], width: u32, native: &mut [u8]) -> bool {
    if format == MTL_FORMAT_BGRA8_UNORM || format == pixel_format::MTL_FORMAT_BGRA8_UNORM_SRGB {
        if rgba_row.len() < native.len() || native.len() < (width as usize) * 4 {
            return false;
        }
        for i in 0..(width as usize) {
            let o = i * 4;
            native[o] = rgba_row[o + 2];
            native[o + 1] = rgba_row[o + 1];
            native[o + 2] = rgba_row[o];
            native[o + 3] = rgba_row[o + 3];
        }
        return true;
    }
    convert_rgba8_to_row(format, rgba_row, width, native)
}

/// Write tightly packed raw rows into a mapping (depth32float / stencil8).
///
/// Contig HostOps view when possible; else multi-import (no write_gpa).
#[allow(
    clippy::too_many_arguments,
    reason = "the mapping API keeps source rows and destination geometry explicit"
)]
pub fn write_raw_rows<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mapping_id: u32,
    src: &[u8],
    src_stride: u32,
    row_bytes: u32,
    width: u32,
    height: u32,
) -> bool {
    if width == 0
        || height == 0
        || width > MAX_SCANOUT_DIM
        || height > MAX_SCANOUT_DIM
        || row_bytes == 0
        || src_stride < row_bytes
    {
        return false;
    }
    let need = (height as u64).saturating_mul(src_stride as u64) as usize;
    if src.len() < need {
        return false;
    }
    // Deferred-writeback flush-on-access (coarse: whole mapping — this entry
    // resolves its window only later and is off the hot compute path).
    crate::runtime::storage_flush::flush_intersecting(state, host, mapping_id, 0, u64::MAX);
    let Some(m) = state.mappings.get(&mapping_id) else {
        return false;
    };
    if !m.mapped || m.page_entries.is_empty() {
        return false;
    }
    if m.has_geom && (m.width != width || m.height != height) {
        return false;
    }
    let span_end = (row_bytes as u64).saturating_mul(height as u64);
    let rb = row_bytes as usize;
    if let Some((ptr, _)) = contig_for_span(state, host, mapping_id, span_end) {
        // SAFETY: contig covers span_end from offset 0.
        let base = ptr as *mut u8;
        for y in 0..height as usize {
            let src_off = y * src_stride as usize;
            let dst = unsafe { base.add(y * rb) };
            unsafe {
                std::ptr::copy_nonoverlapping(src.as_ptr().add(src_off), dst, rb);
            }
        }
    } else {
        for y in 0..height as usize {
            let src_off = y * src_stride as usize;
            let moff = (y as u64).saturating_mul(row_bytes as u64);
            if !mapper::write_mapping_bytes(
                state,
                host,
                mapping_id,
                moff,
                &src[src_off..src_off + rb],
            ) {
                return false;
            }
        }
    }
    state.invalidate_storage_residency_window(mapping_id, 0, span_end);
    let _ = state.mark_mapping_written(mapping_id);
    true
}

/// Read tightly packed raw rows from a mapping (depth32float / stencil8 LOAD).
/// Contig HostOps view when possible; else multi-import.
#[allow(
    clippy::too_many_arguments,
    reason = "the mapping API keeps source rows and destination geometry explicit"
)]
pub fn read_raw_rows<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mapping_id: u32,
    dst: &mut [u8],
    dst_stride: u32,
    row_bytes: u32,
    width: u32,
    height: u32,
) -> bool {
    if width == 0
        || height == 0
        || width > MAX_SCANOUT_DIM
        || height > MAX_SCANOUT_DIM
        || row_bytes == 0
        || dst_stride < row_bytes
    {
        return false;
    }
    let need = (height as u64).saturating_mul(dst_stride as u64) as usize;
    if dst.len() < need {
        return false;
    }
    // Deferred-writeback flush-on-access (coarse: whole mapping — this entry
    // resolves its window only later and is off the hot compute path).
    crate::runtime::storage_flush::flush_intersecting(state, host, mapping_id, 0, u64::MAX);
    let Some(m) = state.mappings.get(&mapping_id) else {
        return false;
    };
    if !m.mapped || m.page_entries.is_empty() {
        return false;
    }
    if m.has_geom && (m.width != width || m.height != height) {
        return false;
    }
    let span_end = (row_bytes as u64).saturating_mul(height as u64);
    let rb = row_bytes as usize;
    if let Some((ptr, _)) = contig_for_span(state, host, mapping_id, span_end) {
        // SAFETY: contig covers span_end from offset 0.
        let base = ptr as *const u8;
        for y in 0..height as usize {
            let dst_off = y * dst_stride as usize;
            let src = unsafe { base.add(y * rb) };
            unsafe {
                std::ptr::copy_nonoverlapping(src, dst[dst_off..].as_mut_ptr(), rb);
            }
        }
    } else {
        for y in 0..height as usize {
            let dst_off = y * dst_stride as usize;
            let moff = (y as u64).saturating_mul(row_bytes as u64);
            if !mapper::read_mapping_bytes(
                state,
                host,
                mapping_id,
                moff,
                &mut dst[dst_off..dst_off + rb],
            ) {
                return false;
            }
        }
    }
    true
}

/// Read a rectangular texel region from a mapped type-11 IOSurface.
/// Contig HostOps view when possible; else multi-import.
#[allow(
    clippy::too_many_arguments,
    reason = "the mapping API mirrors the decoded texture rectangle"
)]
#[cfg(test)]
pub fn read_rect_raw<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mapping_id: u32,
    origin_x: u32,
    origin_y: u32,
    width: u32,
    height: u32,
    dst: &mut [u8],
    dst_stride: u32,
) -> bool {
    let Some(m) = state.mappings.get(&mapping_id) else {
        return false;
    };
    if !m.has_geom {
        return false;
    }
    let format = if m.format != 0 {
        m.format
    } else {
        MTL_FORMAT_BGRA8_UNORM
    };
    let Some((base_off, bpr, span_end)) =
        type11_sample_window(m, mapping_id, m.width, m.height, format)
    else {
        return false;
    };
    let Some(bpp) = pixel_format::bytes_per_pixel(format) else {
        return false;
    };
    if origin_x.saturating_add(width) > m.width || origin_y.saturating_add(height) > m.height {
        return false;
    }
    read_rect_raw_at(
        state, host, mapping_id, base_off, bpr, span_end, origin_x, origin_y, width, height, bpp,
        dst, dst_stride,
    )
}

/// Read a rect using an explicit sample window (plane base + bpr + span).
/// Contig HostOps view when possible; else multi-import.
#[allow(
    clippy::too_many_arguments,
    reason = "the explicit-plane API mirrors its sample window and rectangle"
)]
pub fn read_rect_raw_at<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mapping_id: u32,
    base_off: u64,
    surface_bpr: u32,
    span_end: u64,
    origin_x: u32,
    origin_y: u32,
    width: u32,
    height: u32,
    bpp: u32,
    dst: &mut [u8],
    dst_stride: u32,
) -> bool {
    if width == 0 || height == 0 || width > MAX_SCANOUT_DIM || height > MAX_SCANOUT_DIM || bpp == 0
    {
        return false;
    }
    // Deferred-writeback flush-on-access, for the same reason
    // `mapper::read_mapping_bytes` does it: this read must observe the deferred
    // Store's pixels, not the stale pre-Store guest bytes.
    //
    // It has to be here rather than at the callers because only one of the two
    // paths below was ever covered. The fragmented path ends in
    // `read_mapping_bytes`, which flushes; the `contig_for_span` path is a raw
    // `copy_nonoverlapping` out of the mapped span and flushes nothing — so
    // whether a type-11 surface read observed the deferred Store depended on
    // whether its guest pages happened to be contiguous. Three callers read
    // guest pages through here with no flush of their own: the type-5 view
    // loader, a blit reading a type-11 texture backing, and the compute sample
    // stage.
    //
    // `flush_intersecting` returns immediately when nothing is armed, so this
    // costs a map-empty check per read. It must also precede `contig_for_span`:
    // the flush writes through the mapping and can retire the cached view.
    crate::runtime::storage_flush::flush_intersecting(state, host, mapping_id, base_off, span_end);
    let Some(m) = state.mappings.get(&mapping_id) else {
        return false;
    };
    if !m.mapped || m.page_entries.is_empty() {
        return false;
    }
    let Some(row_bytes) = width.checked_mul(bpp) else {
        return false;
    };
    if dst_stride < row_bytes {
        return false;
    }
    let need = (height as u64).saturating_mul(dst_stride as u64) as usize;
    if dst.len() < need {
        return false;
    }
    let x_off = (origin_x as u64).saturating_mul(bpp as u64);
    if x_off.saturating_add(row_bytes as u64) > surface_bpr as u64 {
        return false;
    }
    let rb = row_bytes as usize;
    let bpr = surface_bpr as usize;
    if let Some((ptr, _)) = contig_for_span(state, host, mapping_id, span_end) {
        // The fragmented branch below goes through `mapper::read_mapping_bytes`,
        // which is bounded already. A correctly-sized read satisfies this exactly
        // (dense tight read: `read_end == span_end`), so it drops ONLY a genuine
        // overrun.
        let read_end = rect_extent_end(base_off, origin_y, height, bpr, x_off, rb);
        if read_end > span_end {
            crate::observe::fail(format!(
                "mapping_read fail reason=read_overrun mid={mapping_id} base_off={base_off} origin_y={origin_y} height={height} bpr={surface_bpr} x_off={x_off} rb={rb} read_end={read_end} span_end={span_end}"
            ));
            return false;
        }
        // SAFETY: contig covers span_end, and read_end ≤ span_end (checked).
        let base = unsafe { (ptr as *const u8).add(base_off as usize) };
        if x_off == 0 && rb == bpr && dst_stride as usize == rb {
            // Dense rows: identical byte range as the loop, one copy.
            let src = unsafe { base.add((origin_y as usize).saturating_mul(bpr)) };
            let len = (height as usize) * rb;
            unsafe {
                std::ptr::copy_nonoverlapping(src, dst.as_mut_ptr(), len);
            }
        } else {
            for y in 0..height as usize {
                let dst_off = y * dst_stride as usize;
                let row_off = ((origin_y as usize) + y)
                    .saturating_mul(bpr)
                    .saturating_add(x_off as usize);
                let src = unsafe { base.add(row_off) };
                unsafe {
                    std::ptr::copy_nonoverlapping(src, dst[dst_off..].as_mut_ptr(), rb);
                }
            }
        }
    } else {
        // Exact full-plane row layout: the tight texture bytes are already the
        // mapping byte window. Import fragmented GPA runs directly into the
        // caller's Vulkan staging vector instead of allocating another
        // full-plane window and copying it row by row.
        let direct_len = (height as usize).checked_mul(dst_stride as usize);
        let window_len = span_end
            .checked_sub(base_off)
            .and_then(|len| usize::try_from(len).ok());
        let direct_len = direct_len.filter(|direct_len| {
            origin_x == 0
                && origin_y == 0
                && row_bytes == surface_bpr
                && dst_stride == surface_bpr
                && Some(*direct_len) == window_len
        });
        if let Some(direct_len) = direct_len {
            crate::observe::off(format!(
                "mapping_read full_tight_direct mid={mapping_id} bytes={direct_len} bpr={surface_bpr} rows={height}"
            ));
            return mapper::read_mapping_bytes(
                state,
                host,
                mapping_id,
                base_off,
                &mut dst[..direct_len],
            );
        }
        // Materialize the fragmented sample window once. Calling
        // read_mapping_bytes for every row revalidates every page and rebuilds
        // all packed GPA runs each time (O(height × pages)); fullscreen
        // compute textures then strand every channel behind staging.
        let window_len_u64 = span_end.saturating_sub(base_off);
        let Ok(window_len) = usize::try_from(window_len_u64) else {
            return false;
        };
        let mut window = vec![0u8; window_len];
        if !mapper::read_mapping_bytes(state, host, mapping_id, base_off, &mut window) {
            return false;
        }
        for y in 0..height as usize {
            let dst_off = y * dst_stride as usize;
            let row_off = ((origin_y as usize) + y)
                .saturating_mul(bpr)
                .saturating_add(x_off as usize);
            let row_end = row_off.saturating_add(rb);
            let Some(row) = window.get(row_off..row_end) else {
                return false;
            };
            dst[dst_off..dst_off + rb].copy_from_slice(row);
        }
    }
    true
}

/// Write a rectangular texel region into a mapped type-11 IOSurface.
///
/// Uses latched mapping geom + [`type11_sample_window`]. Prefer
/// [`write_rect_raw_at`] for an explicit plane window.
#[allow(
    clippy::too_many_arguments,
    reason = "the mapping API mirrors the decoded texture rectangle"
)]
pub fn write_rect_raw<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mapping_id: u32,
    origin_x: u32,
    origin_y: u32,
    width: u32,
    height: u32,
    src: &[u8],
    src_stride: u32,
) -> bool {
    let Some(m) = state.mappings.get(&mapping_id) else {
        return false;
    };
    if !m.has_geom {
        return false;
    }
    let format = if m.format != 0 {
        m.format
    } else {
        MTL_FORMAT_BGRA8_UNORM
    };
    let Some((base_off, bpr, span_end)) =
        type11_sample_window(m, mapping_id, m.width, m.height, format)
    else {
        return false;
    };
    let Some(bpp) = pixel_format::bytes_per_pixel(format) else {
        return false;
    };
    if origin_x.saturating_add(width) > m.width || origin_y.saturating_add(height) > m.height {
        return false;
    }
    write_rect_raw_at(
        state, host, mapping_id, base_off, bpr, span_end, origin_x, origin_y, width, height, bpp,
        src, src_stride,
    )
}

/// Write a rect using an explicit sample window (plane base + bpr + span).
/// Contig HostOps view when possible; else multi-import.
#[allow(
    clippy::too_many_arguments,
    reason = "the explicit-plane API mirrors its sample window and rectangle"
)]
pub fn write_rect_raw_at<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mapping_id: u32,
    base_off: u64,
    surface_bpr: u32,
    span_end: u64,
    origin_x: u32,
    origin_y: u32,
    width: u32,
    height: u32,
    bpp: u32,
    src: &[u8],
    src_stride: u32,
) -> bool {
    write_rect_raw_at_impl(
        state,
        host,
        mapping_id,
        base_off,
        surface_bpr,
        span_end,
        origin_x,
        origin_y,
        width,
        height,
        bpp,
        src,
        src_stride,
        false,
    )
}

/// Write a complete explicit texture plane. Fragmented mappings import each
/// maximal packed GPA run once instead of re-importing for every image row.
#[allow(
    clippy::too_many_arguments,
    reason = "the full-plane API mirrors its mapping window and row layout"
)]
pub fn write_full_rect_raw_at<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mapping_id: u32,
    base_off: u64,
    surface_bpr: u32,
    span_end: u64,
    width: u32,
    height: u32,
    bpp: u32,
    src: &[u8],
    src_stride: u32,
) -> bool {
    write_rect_raw_at_impl(
        state,
        host,
        mapping_id,
        base_off,
        surface_bpr,
        span_end,
        0,
        0,
        width,
        height,
        bpp,
        src,
        src_stride,
        true,
    )
}

#[allow(clippy::too_many_arguments)]
fn write_rect_raw_at_impl<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mapping_id: u32,
    base_off: u64,
    surface_bpr: u32,
    span_end: u64,
    origin_x: u32,
    origin_y: u32,
    width: u32,
    height: u32,
    bpp: u32,
    src: &[u8],
    src_stride: u32,
    full_plane: bool,
) -> bool {
    if width == 0 || height == 0 || width > MAX_SCANOUT_DIM || height > MAX_SCANOUT_DIM || bpp == 0
    {
        return false;
    }
    let Some(m) = state.mappings.get(&mapping_id) else {
        return false;
    };
    if !m.mapped || m.page_entries.is_empty() {
        return false;
    }
    let Some(row_bytes) = width.checked_mul(bpp) else {
        return false;
    };
    if src_stride < row_bytes {
        return false;
    }
    let need = (height as u64).saturating_mul(src_stride as u64) as usize;
    if src.len() < need {
        return false;
    }
    let x_off = (origin_x as u64).saturating_mul(bpp as u64);
    if x_off.saturating_add(row_bytes as u64) > surface_bpr as u64 {
        return false;
    }
    let rb = row_bytes as usize;
    let bpr = surface_bpr as usize;
    if let Some((ptr, _)) = contig_for_span(state, host, mapping_id, span_end) {
        // The fragmented full-plane branch below already rejects on the same bound
        // (`frame_end > span_end`); enforce it here too so the contig fast paths
        // can never overrun. A correctly-sized writeback satisfies this exactly
        // (dense tight write: `write_end == span_end`), so it drops ONLY a genuine
        // overrun — named, never silent.
        let write_end = rect_extent_end(base_off, origin_y, height, bpr, x_off, rb);
        if write_end > span_end {
            crate::observe::fail(format!(
                "mapping_write fail reason=writeback_overrun mid={mapping_id} base_off={base_off} origin_y={origin_y} height={height} bpr={surface_bpr} x_off={x_off} rb={rb} write_end={write_end} span_end={span_end}"
            ));
            return false;
        }
        // SAFETY: contig covers span_end, and write_end ≤ span_end (checked).
        let base = unsafe { (ptr as *mut u8).add(base_off as usize) };
        if x_off == 0 && rb == bpr && src_stride as usize == rb {
            // Dense rows: identical byte range as the loop, one copy.
            let dst = unsafe { base.add((origin_y as usize).saturating_mul(bpr)) };
            let len = (height as usize) * rb;
            unsafe {
                std::ptr::copy_nonoverlapping(src.as_ptr(), dst, len);
            }
        } else {
            for y in 0..height as usize {
                let src_off = y * src_stride as usize;
                let row_off = ((origin_y as usize) + y)
                    .saturating_mul(bpr)
                    .saturating_add(x_off as usize);
                let dst = unsafe { base.add(row_off) };
                unsafe {
                    std::ptr::copy_nonoverlapping(src.as_ptr().add(src_off), dst, rb);
                }
            }
        }
    } else if full_plane {
        // Fragmented full-plane write: stage the native row layout and import
        // each maximal packed GPA run once. Calling write_mapping_bytes once
        // per row turns a 1928-row storage-texture writeback into thousands of
        // QEMU memory-region imports (the live compute_writeback_amplification
        // class). Native row padding is not texture content and is zeroed, as
        // in write_bgra8's fragmented path above.
        // `span_end` ends at the final row's last texel. It deliberately does
        // not include padding after the final row, so staging bpr * height
        // rejects every exact-span surface whose row pitch exceeds row_bytes.
        let frame_len = match (height as usize)
            .checked_sub(1)
            .and_then(|rows| (surface_bpr as usize).checked_mul(rows))
            .and_then(|prefix| prefix.checked_add(rb))
        {
            Some(v) => v,
            None => return false,
        };
        let Some(frame_end) = base_off.checked_add(frame_len as u64) else {
            return false;
        };
        if frame_end > span_end {
            return false;
        }
        // With no physical row padding, the engine's tight result is already
        // the exact mapping byte window. Write it through the fragmented-run
        // importer directly; a second frame allocation/copy is redundant.
        let window_len = span_end
            .checked_sub(base_off)
            .and_then(|len| usize::try_from(len).ok());
        if origin_x == 0
            && origin_y == 0
            && rb == bpr
            && src_stride == surface_bpr
            && Some(frame_len) == window_len
        {
            crate::observe::off(format!(
                "mapping_write full_tight_direct mid={mapping_id} bytes={frame_len} bpr={surface_bpr} rows={height}"
            ));
            if !mapper::write_mapping_bytes(state, host, mapping_id, base_off, &src[..frame_len]) {
                return false;
            }
            let _ = state.mark_mapping_written(mapping_id);
            return true;
        }
        let mut frame = vec![0u8; frame_len];
        for y in 0..height as usize {
            let src_off = y * src_stride as usize;
            let dst_off = ((origin_y as usize) + y)
                .saturating_mul(bpr)
                .saturating_add(x_off as usize);
            if dst_off + rb > frame.len() {
                return false;
            }
            frame[dst_off..dst_off + rb].copy_from_slice(&src[src_off..src_off + rb]);
        }
        if !mapper::write_mapping_bytes(state, host, mapping_id, base_off, &frame) {
            return false;
        }
    } else {
        for y in 0..height as usize {
            let src_off = y * src_stride as usize;
            let moff = base_off
                .saturating_add(((origin_y as u64) + y as u64).saturating_mul(surface_bpr as u64))
                .saturating_add(x_off);
            if !mapper::write_mapping_bytes(
                state,
                host,
                mapping_id,
                moff,
                &src[src_off..src_off + rb],
            ) {
                return false;
            }
        }
    }
    state.invalidate_storage_residency_window(mapping_id, base_off, span_end);
    let _ = state.mark_mapping_written(mapping_id);
    true
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::contract::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
    use crate::model::{DeviceId, PAGE_SHIFT_ARM64E};
    use crate::runtime::host::FakeHost;

    #[test]
    fn write_bumps_generation() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let mut host = FakeHost::new();
        let pfn = 0x10u32;
        let gpa = (pfn as u64) << PAGE_SHIFT_ARM64E;
        host.map_range(gpa, 0x4000, 0);
        let entry = (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
        state.map_surface(3);
        state.attach_mapping_internal(3, 0); // leave internal 0; set pages manually
        let m = state.mappings.get_mut(&3).unwrap();
        m.mapping_internal = 1;
        m.page_entries = vec![entry];
        assert!(state.set_mapping_geom(3, 2, 2, MTL_FORMAT_BGRA8_UNORM));
        let src = [0x11u8, 0x22, 0x33, 0x44, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        // 2x2 BGRA, stride 8
        assert!(write_bgra8(&mut state, &mut host, 3, &src, 8, 2, 2));
        assert_eq!(state.mappings.get(&3).unwrap().content_generation, 1);
    }

    /// The write that makes the host copy authoritative must also arm the
    /// witness for it.
    ///
    /// This function writes the guest pages and then stores the host render
    /// cache, so at this instant the two agree — the one moment the copy's
    /// currency can be pinned. Nothing else armed it: the type-4 sampled
    /// ladder's first census read `t11rung_host_cache_gw_no_stamp` 14 092
    /// against `gw_clean` 0, because only the Vulkan Store rails ever stamped
    /// while the copy that rung serves is written here. Unstamped, the reader
    /// cannot tell a surface the guest has rewritten from one it has not, and
    /// has to assume the worst on every bind.
    #[test]
    fn a_host_cache_write_arms_the_guest_write_witness_for_the_copy() {
        use crate::runtime::host::HostOps;

        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let mut host = FakeHost::new();
        let pfn = 0x30u32;
        let gpa = (pfn as u64) << PAGE_SHIFT_ARM64E;
        host.map_range(gpa, 0x4000, 0);
        let entry = (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
        state.map_surface(11);
        state.attach_mapping_internal(11, 0);
        let m = state.mappings.get_mut(&11).unwrap();
        m.mapping_internal = 1;
        m.page_entries = vec![entry];
        assert!(state.set_mapping_geom(11, 2, 2, MTL_FORMAT_BGRA8_UNORM));

        assert_eq!(
            state.mappings[&11].guest_write_token, 0,
            "nothing has armed this mapping yet"
        );
        let src = [0x11u8, 0x22, 0x33, 0x44, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        assert!(write_bgra8(&mut state, &mut host, 11, &src, 8, 2, 2));

        let token = state.mappings[&11].guest_write_token;
        assert_ne!(token, 0, "the store must register the pages it copied");
        assert_eq!(
            host.guest_write_gen(token),
            Some(state.mappings[&11].guest_write_gen_at_store),
            "the recorded generation must be the one the copy is current as of"
        );

        // A guest CPU store into the surface, with no device operation: the
        // recorded generation no longer matches, which is exactly what the
        // sampled ladder reads to refuse the copy.
        host.guest_wrote_page(gpa);
        assert_ne!(
            host.guest_write_gen(token),
            Some(state.mappings[&11].guest_write_gen_at_store),
            "a guest write must move the host's generation away from the stamp"
        );
    }

    /// A guest write drops only the storage-residency mirror windows it
    /// intersects; disjoint sibling windows (ping-pong canvases) survive.
    #[test]
    fn mapping_write_invalidates_intersecting_residency_windows_only() {
        use crate::model::ComputeStorageResidencyKey;

        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let mut host = FakeHost::new();
        let pfn = 0x20u32;
        let gpa = (pfn as u64) << PAGE_SHIFT_ARM64E;
        host.map_range(gpa, 0x4000, 0);
        let entry = (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
        state.map_surface(7);
        state.attach_mapping_internal(7, 0);
        let m = state.mappings.get_mut(&7).unwrap();
        m.mapping_internal = 1;
        m.page_entries = vec![entry];
        let window = |surface_offset: u64, span_end: u64| ComputeStorageResidencyKey {
            mapping_id: 7,
            map_generation: state.mappings[&7].map_generation,
            surface_offset,
            surface_bpr: 32,
            span_end,
            width: 8,
            height: 2,
            pixel_format: MTL_FORMAT_BGRA8_UNORM,
            texture_ref: 0,
        };
        let hit = window(0, 64);
        let survivor = window(1024, 1088);
        state.compute_storage_residency.insert(hit, 5);
        state.compute_storage_residency.insert(survivor, 5);
        assert!(mapper::write_mapping_bytes(
            &mut state, &mut host, 7, 16, &[0u8; 32]
        ));
        assert!(!state.compute_storage_residency.contains_key(&hit));
        assert!(state.compute_storage_residency.contains_key(&survivor));
    }

    /// compute_writeback_amplification: fragmented texture writeback imports
    /// once per maximal GPA run, not once per image row.
    #[test]
    fn fragmented_raw_rect_bulk_imports_runs_not_rows() {
        use crate::model::PAGE_SHIFT_X86;

        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        host.strict_linux_map = true;
        let page = 1usize << PAGE_SHIFT_X86;
        let gpa0 = 0x1000_0000u64;
        let gpa1 = 0x2000_0000u64;
        host.map_range(gpa0, page, 0x7e);
        host.map_range(gpa1, page, 0x7e);
        let mid = 19;
        state.map_surface(mid);
        {
            let m = state.mappings.get_mut(&mid).unwrap();
            m.page_entries = vec![
                (((gpa0 >> PAGE_SHIFT_X86) as u32) << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID,
                (((gpa1 >> PAGE_SHIFT_X86) as u32) << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID,
            ];
        }
        let src = vec![0x2a; 4 * 16];
        // Contract span ends at the last texel, excluding trailing row
        // padding: (height - 1) * bpr + tight = 3 * 2048 + 16.
        assert!(write_full_rect_raw_at(
            &mut state, &mut host, mid, 0, 2048, 6160, 4, 4, 4, &src, 16,
        ));
        // One successful import per maximal GPA run, and nothing else: the
        // fragmented page list fails `is_single_packed_run` in Rust, so the
        // packed-view fast path never spends a call the host can only refuse.
        // The old row loop took nine attempts for these four rows and scaled
        // with height.
        assert_eq!(host.map_pages_calls, 2);
        let calls_after_write = host.map_pages_calls;

        let mut row = [0u8; 16];
        assert!(mapper::read_mapping_bytes(
            &mut state, &mut host, mid, 4096, &mut row,
        ));
        assert_eq!(row, [0x2a; 16]);
        assert_eq!(calls_after_write, 2);
    }

    /// Linux product: non-packed page list still lands BGRA via multi-import.
    #[test]
    fn write_bgra8_fragmented_pages_multi_import() {
        use crate::model::PAGE_SHIFT_X86;
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        host.strict_linux_map = true;
        let page = 1u64 << PAGE_SHIFT_X86;
        // 2×2 BGRA needs 16 bytes → one page; use two non-adjacent pages so
        // ensure_contig_view fails and multi-import is forced.
        let gpa0 = 0x3000_0000u64;
        let gpa1 = 0x4000_0000u64;
        host.map_range(gpa0, page as usize, 0);
        host.map_range(gpa1, page as usize, 0);
        let pfn0 = (gpa0 >> PAGE_SHIFT_X86) as u32;
        let pfn1 = (gpa1 >> PAGE_SHIFT_X86) as u32;
        let mid = 11u32;
        state.map_surface(mid);
        {
            let m = state.mappings.get_mut(&mid).unwrap();
            m.mapped = true;
            m.mapping_internal = 1;
            m.page_entries = vec![
                (pfn0 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID,
                (pfn1 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID,
            ];
        }
        assert!(state.set_mapping_geom(mid, 2, 2, MTL_FORMAT_BGRA8_UNORM));
        let src = [
            0x11u8, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee,
            0xff, 0x10,
        ];
        assert!(write_bgra8(&mut state, &mut host, mid, &src, 8, 2, 2));
        let mut first = [0u8; 4];
        assert!(host.read_gpa(gpa0, &mut first).is_ok());
        assert_eq!(&first, &src[..4]);
        assert!(mapper::ensure_contig_view(&mut state, &mut host, mid).is_none());
    }

    /// The fragmented BGRA write path must stop at the final row's last texel.
    /// Writing `bpr * height` includes padding after the final row, which is not
    /// texture content and can overrun an exact IOSurface allocation.
    #[test]
    fn write_bgra8_fragmented_skips_final_row_padding() {
        use crate::model::PAGE_SHIFT_X86;

        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        host.strict_linux_map = true;
        let page = 1u64 << PAGE_SHIFT_X86;
        let gpa0 = 0x3500_0000u64;
        let gpa1 = 0x4600_0000u64;
        host.map_range(gpa0, page as usize, 0xCC);
        host.map_range(gpa1, page as usize, 0xCC);
        let pfn0 = (gpa0 >> PAGE_SHIFT_X86) as u32;
        let pfn1 = (gpa1 >> PAGE_SHIFT_X86) as u32;
        let mid = 13u32;
        state.map_surface(mid);
        {
            let m = state.mappings.get_mut(&mid).unwrap();
            m.mapped = true;
            m.mapping_internal = 1;
            m.page_entries = vec![
                (pfn0 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID,
                (pfn1 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID,
            ];
        }
        assert!(state.set_mapping_geom(mid, 2, 2, MTL_FORMAT_BGRA8_UNORM));
        let src = [
            0x11u8, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee,
            0xff, 0x10,
        ];
        assert!(write_bgra8(&mut state, &mut host, mid, &src, 8, 2, 2));

        let mut final_row = [0u8; 8];
        assert!(host.read_gpa(gpa0 + 128, &mut final_row).is_ok());
        assert_eq!(final_row, src[8..16]);

        let mut final_padding = [0u8; 4];
        assert!(host.read_gpa(gpa0 + 128 + 8, &mut final_padding).is_ok());
        assert_eq!(
            final_padding, [0xCC; 4],
            "padding after the final row must remain untouched"
        );
    }

    /// The packed-contig BGRA write pokes rows straight into a raw host pointer,
    /// so its only bound is the sample window `contig_for_span` validated. The
    /// fragmented path's equivalent is checked by
    /// `write_bgra8_fragmented_skips_final_row_padding`; this pins the same
    /// contract on the pointer path, where an overrun is a write into whatever
    /// guest allocation follows rather than a refused import.
    ///
    /// Asserted as "every byte outside the window is unchanged", not just the
    /// final row's padding: inter-row padding belongs to the same class and a
    /// stride bug hits it first.
    #[test]
    fn write_bgra8_contig_writes_only_inside_the_sample_window() {
        use crate::model::PAGE_SHIFT_X86;

        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        host.strict_linux_map = true;
        let page = 1u64 << PAGE_SHIFT_X86;
        let gpa = 0x7300_0000u64;
        host.map_range(gpa, page as usize, 0xCC);
        let pfn = (gpa >> PAGE_SHIFT_X86) as u32;
        let mid = 21u32;
        state.map_surface(mid);
        {
            let m = state.mappings.get_mut(&mid).unwrap();
            m.mapped = true;
            m.mapping_internal = 1;
            m.page_entries = vec![(pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
        }
        assert!(state.set_mapping_geom(mid, 2, 2, MTL_FORMAT_BGRA8_UNORM));
        // No device descriptor, so the invented window applies: tight = 2 × 4,
        // bpr = ALIGN_UP(8, ROW_BYTES_ALIGN) = 128, two rows.
        let tight = 8usize;
        let bpr = 128usize;
        let src: Vec<u8> = (0..16u8).map(|i| i.wrapping_mul(17)).collect();
        assert!(
            mapper::ensure_contig_view(&mut state, &mut host, mid).is_some(),
            "one packed page must take the contig path this test is about"
        );
        assert!(write_bgra8(&mut state, &mut host, mid, &src, 8, 2, 2));

        let mut got = vec![0u8; page as usize];
        assert!(host.read_gpa(gpa, &mut got).is_ok());
        let mut want = vec![0xCCu8; page as usize];
        want[..tight].copy_from_slice(&src[..tight]);
        want[bpr..bpr + tight].copy_from_slice(&src[tight..]);
        let first_diff = got.iter().zip(want.iter()).position(|(a, b)| a != b);
        assert_eq!(
            first_diff, None,
            "byte {first_diff:?} outside the sample window was modified"
        );
    }

    /// Fragmented compute staging materializes the sample window once and
    /// preserves padded-row addressing across non-contiguous guest pages.
    #[test]
    fn read_rect_raw_fragmented_pages_with_padded_rows() {
        use crate::model::PAGE_SHIFT_X86;

        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        host.strict_linux_map = true;
        let page = 1u64 << PAGE_SHIFT_X86;
        let gpa0 = 0x5100_0000u64;
        let gpa1 = 0x6200_0000u64;
        host.map_range(gpa0, page as usize, 0);
        host.map_range(gpa1, page as usize, 0);
        let row0 = [1u8, 2, 3, 4, 5, 6, 7, 8];
        let row1 = [9u8, 10, 11, 12, 13, 14, 15, 16];
        host.write_gpa(gpa0, &row0).unwrap();
        host.write_gpa(gpa1, &row1).unwrap();

        let pfn0 = (gpa0 >> PAGE_SHIFT_X86) as u32;
        let pfn1 = (gpa1 >> PAGE_SHIFT_X86) as u32;
        let mid = 12u32;
        state.map_surface(mid);
        {
            let m = state.mappings.get_mut(&mid).unwrap();
            m.mapped = true;
            m.mapping_internal = 1;
            m.page_entries = vec![
                (pfn0 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID,
                (pfn1 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID,
            ];
        }
        let mut dst = [0u8; 16];
        assert!(read_rect_raw_at(
            &mut state,
            &mut host,
            mid,
            0,
            page as u32,
            page + row1.len() as u64,
            0,
            0,
            2,
            2,
            4,
            &mut dst,
            8,
        ));
        assert_eq!(&dst[..8], &row0);
        assert_eq!(&dst[8..], &row1);
    }

    /// compute_full_tight_scratch: an exact-pitch fragmented compute plane
    /// reads and writes directly through the caller's tight buffer. The
    /// always-on proxy proves this class is selected on a live dispatch.
    #[test]
    fn fragmented_full_tight_rect_uses_direct_mapping_window() {
        use crate::model::PAGE_SHIFT_X86;

        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        host.strict_linux_map = true;
        let page = 1u64 << PAGE_SHIFT_X86;
        let gpa0 = 0x7100_0000u64;
        let gpa1 = 0x8200_0000u64;
        host.map_range(gpa0, page as usize, 0x31);
        host.map_range(gpa1, page as usize, 0x42);
        let mid = 29;
        assert!(state.map_surface(mid));
        {
            let m = state.mappings.get_mut(&mid).unwrap();
            m.mapped = true;
            m.mapping_internal = 1;
            m.page_entries = vec![
                (((gpa0 >> PAGE_SHIFT_X86) as u32) << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID,
                (((gpa1 >> PAGE_SHIFT_X86) as u32) << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID,
            ];
        }

        let bpr = page as u32;
        let span = page * 2;
        let mut tight = vec![0u8; span as usize];
        assert!(read_rect_raw_at(
            &mut state,
            &mut host,
            mid,
            0,
            bpr,
            span,
            0,
            0,
            bpr / 4,
            2,
            4,
            &mut tight,
            bpr,
        ));
        assert!(tight[..page as usize].iter().all(|&v| v == 0x31));
        assert!(tight[page as usize..].iter().all(|&v| v == 0x42));

        tight.fill(0x5a);
        assert!(write_full_rect_raw_at(
            &mut state,
            &mut host,
            mid,
            0,
            bpr,
            span,
            bpr / 4,
            2,
            4,
            &tight,
            bpr,
        ));
        let mut check = vec![0u8; span as usize];
        assert!(mapper::read_mapping_bytes(
            &mut state, &mut host, mid, 0, &mut check,
        ));
        assert_eq!(check, tight);

        let log = std::fs::read_to_string(crate::observe::fail_log_path()).expect("fail log");
        assert!(log.contains(&format!(
            "OFF mapping_read full_tight_direct mid={mid} bytes={span}"
        )));
        assert!(log.contains(&format!(
            "OFF mapping_write full_tight_direct mid={mid} bytes={span}"
        )));
    }

    /// qemu-shim: guest page write IS the surface content (unified memory) —
    /// bytes land in pages and the generation advances; nothing else exists.
    #[test]
    fn write_bgra8_lands_in_pages_and_bumps_gen() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let mut host = FakeHost::new();
        let pfn = 0x18u32;
        let gpa = (pfn as u64) << PAGE_SHIFT_ARM64E;
        host.map_range(gpa, 0x4000, 0);
        let entry = (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
        state.map_surface(8);
        {
            let m = state.mappings.get_mut(&8).unwrap();
            m.mapped = true;
            m.mapping_internal = 1;
            m.page_entries = vec![entry];
        }
        assert!(state.set_mapping_geom(8, 2, 2, MTL_FORMAT_BGRA8_UNORM));
        // BGRA red pixel + zeros
        let src = [0x00u8, 0x00, 0xFF, 0xFF, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        assert!(write_bgra8(&mut state, &mut host, 8, &src, 8, 2, 2));
        let m = state.mappings.get(&8).unwrap();
        assert_eq!(m.content_generation, 1);
        let mut first_px = [0u8; 4];
        assert!(host.read_gpa(gpa, &mut first_px).is_ok());
        assert_eq!(&first_px, &[0x00, 0x00, 0xFF, 0xFF], "pages hold the write");
    }

    #[test]
    fn raw_rows_roundtrip() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let mut host = FakeHost::new();
        let pfn = 0x11u32;
        let gpa = (pfn as u64) << PAGE_SHIFT_ARM64E;
        host.map_range(gpa, 0x4000, 0);
        let entry = (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
        state.map_surface(4);
        let m = state.mappings.get_mut(&4).unwrap();
        m.mapped = true;
        m.mapping_internal = 1;
        m.page_entries = vec![entry];
        assert!(state.set_mapping_geom(4, 2, 2, 0));
        // 2x2 depth32 floats: 1.0, 0.5 / 0.25, 0.0
        let mut src = Vec::new();
        for f in [1.0f32, 0.5, 0.25, 0.0] {
            src.extend_from_slice(&f.to_bits().to_le_bytes());
        }
        assert!(write_raw_rows(&mut state, &mut host, 4, &src, 8, 8, 2, 2));
        let gen = state.mappings.get(&4).unwrap().content_generation;
        assert!(gen >= 1);
        let mut dst = vec![0u8; 16];
        assert!(read_raw_rows(
            &mut state, &mut host, 4, &mut dst, 8, 8, 2, 2
        ));
        assert_eq!(dst, src);
        // Read does not bump generation.
        assert_eq!(state.mappings.get(&4).unwrap().content_generation, gen);
    }

    /// The read side of the same bound. A rect read whose geometry exceeds what
    /// `span_end` allows must be REJECTED, not run past the contig view.
    ///
    /// `contig_for_span` guarantees the view covers `span_end` and nothing more,
    /// so an oversized `height` reads whatever is next in the QEMU process —
    /// unrelated memory sampled into a texture, or a SIGSEGV that takes the VM
    /// down with no guest-side trace. The write side has carried this guard for a
    /// while; the read side did not, which is the asymmetry to watch for when a
    /// raw-pointer fast path is added beside a checked slow path.
    ///
    /// A correctly-sized read (read_end == span_end) still succeeds.
    #[test]
    fn oversized_height_rect_read_is_rejected_not_overrun() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let mut host = FakeHost::new();
        let pfn = 0x23u32;
        let gpa = (pfn as u64) << PAGE_SHIFT_ARM64E;
        // A full 16 KiB page, so `contig_for_span` succeeds and the guard — not
        // the view length — is what has to stop the overrun.
        host.map_range(gpa, 0x4000, 0xCC);
        let entry = (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
        state.map_surface(11);
        {
            let m = state.mappings.get_mut(&11).unwrap();
            m.mapped = true;
            m.mapping_internal = 1;
            m.page_entries = vec![entry];
        }
        // The source allows exactly 2 rows of bpr=8.
        let bpr = 8u32;
        let (width, bpp) = (2u32, 4u32); // row_bytes = 8 == bpr (dense path)
        let span_end = 16u64;

        // 100 rows: read_end = (100-1)*8 + 8 = 800 > 16.
        let mut big = vec![0u8; 100 * bpr as usize];
        let cap = crate::observe::FailCapture::start();
        assert!(
            !read_rect_raw_at(
                &mut state, &mut host, 11, 0, bpr, span_end, 0, 0, width, 100, bpp, &mut big, bpr,
            ),
            "an oversized-height read must be rejected"
        );
        assert!(
            cap.one("mapping_read").contains("reason=read_overrun"),
            "the refusal must name itself"
        );
        assert!(
            big.iter().all(|&b| b == 0),
            "a rejected read must not have copied anything into the caller's buffer"
        );
        drop(cap);

        // A correctly-sized 2-row read (read_end == span_end) still succeeds.
        let mut ok = vec![0u8; 2 * bpr as usize];
        assert!(
            read_rect_raw_at(
                &mut state, &mut host, 11, 0, bpr, span_end, 0, 0, width, 2, bpp, &mut ok, bpr,
            ),
            "a read whose extent equals span_end must succeed"
        );
        assert_eq!(ok, vec![0xCC; 2 * bpr as usize], "and must read the page");
    }

    /// A writeback whose source `height` exceeds what the destination `span_end`
    /// allows must be REJECTED, not run past the contig view into adjacent guest
    /// pages (the trace-less heap smash behind).
    /// A correctly-sized write (write_end == span_end) still succeeds.
    #[test]
    fn oversized_height_writeback_is_rejected_not_overrun() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let mut host = FakeHost::new();
        let pfn = 0x21u32;
        let gpa = (pfn as u64) << PAGE_SHIFT_ARM64E;
        // Map a full 16 KiB page so contig_for_span succeeds; the guard, not the
        // view length, must be what stops the overrun.
        host.map_range(gpa, 0x4000, 0xCC); // 0xCC canary fills the page
        let entry = (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
        state.map_surface(9);
        {
            let m = state.mappings.get_mut(&9).unwrap();
            m.mapped = true;
            m.mapping_internal = 1;
            m.page_entries = vec![entry];
        }
        // Destination allows exactly 2 rows of bpr=8 (span_end = 2*8 = 16).
        let bpr = 8u32;
        let (width, bpp) = (2u32, 4u32); // row_bytes = 8 == bpr (dense path)
        let span_end = 16u64;
        // Oversized source: 100 rows. write_end = (100-1)*8 + 8 = 800 > 16.
        let big = vec![0x2a; 100 * bpr as usize];
        assert!(
            !write_full_rect_raw_at(
                &mut state, &mut host, 9, 0, bpr, span_end, width, 100, bpp, &big, bpr,
            ),
            "an oversized-height write must be rejected"
        );
        // Nothing past span_end was written — the canary survives at offset 100.
        let mut probe = [0u8; 4];
        assert!(mapper::read_mapping_bytes(
            &mut state, &mut host, 9, 100, &mut probe
        ));
        assert_eq!(
            probe, [0xCC; 4],
            "guest bytes past span_end must be untouched"
        );
        // A correctly-sized 2-row write (write_end == span_end) still succeeds.
        let ok = vec![0x2a; 2 * bpr as usize];
        assert!(
            write_full_rect_raw_at(
                &mut state, &mut host, 9, 0, bpr, span_end, width, 2, bpp, &ok, bpr,
            ),
            "a write whose extent equals span_end must succeed"
        );
    }

    /// Clear+partial Store: seed=None (full write) must overwrite prior guest
    /// content outside the scissor — logo-mid residual when seed=clear skipped.
    #[test]
    fn clear_store_full_write_overwrites_prior_guest_outside_scissor() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let mut host = FakeHost::new();
        let pfn = 0x14u32;
        let gpa = (pfn as u64) << PAGE_SHIFT_ARM64E;
        host.map_range(gpa, 0x4000, 0);
        let entry = (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
        state.map_surface(7);
        let m = state.mappings.get_mut(&7).unwrap();
        m.mapped = true;
        m.mapping_internal = 1;
        m.page_entries = vec![entry];
        // 4x2 BGRA
        assert!(state.set_mapping_geom(7, 4, 2, MTL_FORMAT_BGRA8_UNORM));
        // Prior guest content: "logo" non-zero all pixels.
        let mut logo = vec![0u8; 4 * 2 * 4];
        for px in logo.chunks_exact_mut(4) {
            px.copy_from_slice(&[0x10, 0x20, 0x30, 0xFF]); // BGRA
        }
        assert!(write_bgra8(&mut state, &mut host, 7, &logo, 16, 4, 2));
        // Metal RT after Clear+partial toolbar: clear everywhere, one red pixel
        // at (1,0) as the drawn strip. Full Store (seed=None).
        let mut rgba = vec![0u8; 4 * 2 * 4]; // clear = zeros RGBA
        rgba[4] = 255; // R
        rgba[4 + 3] = 255; // A
        assert!(write_rgba8_image_changed(
            &mut state, &mut host, 7, &rgba,
            None, // Clear Store: not image_changed vs clear seed
            4, 2
        ));
        let mut row = vec![0u8; 16];
        assert!(read_rect_raw(
            &mut state, &mut host, 7, 0, 0, 4, 1, &mut row, 16
        ));
        // Outside scissor pixel 0 must be clear (not logo).
        assert_eq!(
            &row[0..4],
            &[0, 0, 0, 0],
            "Clear Store must wipe prior guest"
        );
        // Drawn pixel 1 red in BGRA.
        assert_eq!(&row[4..8], &[0, 0, 255, 255]);
        // Contrast: Load seed=logo + same rgba would leave logo where equal —
        // not tested here; store_seed_policy gates that path.
    }

    #[test]
    fn rgba8_image_changed_writes_only_diff_spans() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let mut host = FakeHost::new();
        // 4x2 BGRA: invent bpr 128 → one page.
        let pfn = 0x13u32;
        let gpa = (pfn as u64) << PAGE_SHIFT_ARM64E;
        host.map_range(gpa, 0x4000, 0);
        let entry = (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
        state.map_surface(6);
        let m = state.mappings.get_mut(&6).unwrap();
        m.mapped = true;
        m.mapping_internal = 1;
        m.page_entries = vec![entry];
        assert!(state.set_mapping_geom(6, 4, 2, MTL_FORMAT_BGRA8_UNORM));
        // Seed: all zeros.
        let seed = vec![0u8; 4 * 2 * 4];
        // Image: one red pixel at (1,0), rest zero.
        let mut rgba = seed.clone();
        rgba[4] = 255; // R
        rgba[4 + 3] = 255; // A
        assert!(write_rgba8_image_changed(
            &mut state,
            &mut host,
            6,
            &rgba,
            Some(&seed),
            4,
            2
        ));
        // Read back first row of mapping (BGRA native).
        let mut row = vec![0u8; 16];
        assert!(read_rect_raw(
            &mut state, &mut host, 6, 0, 0, 4, 1, &mut row, 16
        ));
        // Pixel 1 is red in BGRA: B=0 G=0 R=255 A=255
        assert_eq!(&row[4..8], &[0, 0, 255, 255]);
        assert_eq!(&row[0..4], &[0, 0, 0, 0]);
    }

    #[test]
    fn rect_raw_roundtrip_subregion() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let mut host = FakeHost::new();
        // 4x2 BGRA needs 4*4=16 tight, aligned bpr = 128 (ROW_BYTES_ALIGN).
        // One page is enough for 2 rows of 128.
        let pfn = 0x12u32;
        let gpa = (pfn as u64) << PAGE_SHIFT_ARM64E;
        host.map_range(gpa, 0x4000, 0);
        let entry = (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
        state.map_surface(5);
        let m = state.mappings.get_mut(&5).unwrap();
        m.mapped = true;
        m.mapping_internal = 1;
        m.page_entries = vec![entry];
        assert!(state.set_mapping_geom(5, 4, 2, MTL_FORMAT_BGRA8_UNORM));
        // Write a 2x1 rect at (1,1): two BGRA pixels.
        let src = [0x11u8, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88];
        assert!(write_rect_raw(
            &mut state, &mut host, 5, 1, 1, 2, 1, &src, 8
        ));
        let mut dst = [0u8; 8];
        assert!(read_rect_raw(
            &mut state, &mut host, 5, 1, 1, 2, 1, &mut dst, 8
        ));
        assert_eq!(dst, src);
        // OOB origin fails.
        assert!(!write_rect_raw(
            &mut state, &mut host, 5, 3, 0, 2, 1, &src, 8
        ));
    }

    /// A rect read through the **contiguous** path must observe a deferred
    /// type-11 Store, not the stale guest bytes underneath it.
    ///
    /// `read_rect_raw_at` has two paths and only one of them was ever covered.
    /// The fragmented path ends in `mapper::read_mapping_bytes`, which flushes;
    /// the `contig_for_span` path is a raw `copy_nonoverlapping` out of the
    /// mapped span and flushed nothing. So whether a type-11 surface read saw
    /// the deferred Store depended on whether its guest pages happened to be
    /// contiguous — and three callers read guest pages through here with no
    /// flush of their own (the type-5 view loader, a blit reading a type-11
    /// texture backing, and the compute sample stage). On screen that is a
    /// sampled layer rendering its pre-Store contents.
    #[cfg(feature = "backend-vulkan")]
    #[test]
    fn a_contiguous_rect_read_flushes_the_deferred_store_first() {
        use crate::model::PAGE_SHIFT_X86;

        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        let page = 1u64 << PAGE_SHIFT_X86;
        let gpa = 0x9100_0000u64;
        host.map_range(gpa, page as usize, 0);
        // Stale guest bytes: what a reader saw before the Store landed.
        host.write_gpa(gpa, &[0x22u8; 256]).unwrap();

        let mid = 21u32;
        state.map_surface(mid);
        {
            let m = state.mappings.get_mut(&mid).unwrap();
            m.mapped = true;
            m.mapping_internal = 1;
            m.map_generation = 1;
            m.has_geom = true;
            m.width = 4;
            m.height = 4;
            m.format = crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;
            m.page_entries =
                vec![(((gpa >> PAGE_SHIFT_X86) as u32) << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
        }
        let (base_off, bpr, span_end) = {
            let m = state.mappings.get(&mid).unwrap();
            type11_sample_window(
                m,
                mid,
                4,
                4,
                crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM,
            )
            .expect("the mapping has a type-11 sample window")
        };
        // The Store the guest issued, deferred rather than written.
        let frame = vec![0xE3u8; 4 * 4 * 4];
        state.compute_deferred_flush.insert(
            crate::model::ComputeStorageResidencyKey {
                mapping_id: mid,
                map_generation: 1,
                surface_offset: base_off,
                surface_bpr: bpr,
                span_end,
                width: 4,
                height: 4,
                pixel_format: crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM,
                texture_ref: 0,
            },
            crate::model::DeferredOwner::Render {
                armed_seq: 1,
                source: crate::model::RenderWindowSource::Owned(std::sync::Arc::new(frame.clone())),
            },
        );

        let mut dst = vec![0u8; 4 * 4 * 4];
        assert!(read_rect_raw_at(
            &mut state, &mut host, mid, base_off, bpr, span_end, 0, 0, 4, 4, 4, &mut dst, 16,
        ));
        assert_eq!(
            dst, frame,
            "the read must observe the deferred Store, not the stale guest bytes"
        );
        assert!(
            state.compute_deferred_flush.is_empty(),
            "the read is a flush trigger, so it must consume the window"
        );
    }
}
