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

/// Take the write proof at the head of a writer, naming the rail that wanted it.
///
/// [`mapper::vouch_mapping_pages_verdict`] already fail-logs *why* a walk refused, with
/// the page and both translations. This adds the one fact that line cannot
/// carry: which writer was about to use the list. Four rails write through
/// `page_entries` and they fail for different reasons at different rates, so a
/// single undifferentiated refusal total would not say which one to read next.
///
/// # Measured: this rail carries the traffic and none of the drift
///
/// One 300 s crash-hunt boot, x86 / Vulkan: `mapw_pages_vouched` 29 002,
/// `mapw_pages_refused` **0**, while the deferred flush rail on the same boot
/// scored `mapping_pages_ours` 25 741 and `mapping_pages_drifted` 9. So these
/// four writers do more writing than the flush rail does, and on this workload
/// not one of them found a contradicted list. The guard here is currently inert;
/// say so rather than counting it as the repair.
///
/// The split is not noise, and the reason is structural: a *deferred* frame is
/// armed at one time and landed at another, and the interval is precisely the
/// window in which the guest can re-point the surface underneath it. These
/// writers vouch and write in the same breath, so their window is nearly zero.
/// **Deferral is the exposure.** That predicts the measurement rather than
/// explaining it after the fact, and it says where to look next: shortening the
/// arm-to-land interval should move `mapping_pages_drifted`, and nothing else
/// here should.
///
/// The drift rate is also not stable boot to boot — 22 on the preceding boot, 9
/// on this one, same workload — so a single boot cannot score it and neither can
/// a pair.
fn vouch_for_write<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mapping_id: u32,
    writer: &'static str,
) -> Option<mapper::PagesVouched> {
    let (verdict, vouched) = mapper::vouch_mapping_pages_verdict(state, host, mapping_id);
    match verdict {
        mapper::PagesVerdict::Ours => {
            crate::runtime::drain::note_store_route("mapw_pages_vouched");
        }
        // The write proceeds exactly as for `Ours`; only the counter differs.
        // `mapw_pages_vouched` used to carry both, so its companion zero
        // (`mapw_pages_refused`) could not distinguish a guard that passed from
        // one that was never armed. Read the two together: `vouched` is the
        // guard's coverage and `unwitnessed` is the hole in it.
        mapper::PagesVerdict::Unwitnessed(why) => {
            crate::runtime::drain::note_store_route("mapw_pages_unwitnessed");
            crate::runtime::drain::note_store_route(match why {
                "no_walk" => "mapw_unwit_no_walk",
                "walk_superseded" => "mapw_unwit_superseded",
                "no_pages" => "mapw_unwit_no_pages",
                _ => "mapw_unwit_no_mapping",
            });
        }
        mapper::PagesVerdict::Drifted => {
            crate::observe::fail(format!(
                "mapping_write fail reason=pages_not_vouched mid={mapping_id} writer={writer}"
            ));
            crate::runtime::drain::note_store_route("mapw_pages_refused");
        }
    }
    vouched
}

/// [`contig_for_span`] for a caller that is about to write through the pointer.
///
/// The view `ensure_contig_view` hands back is a live `mach_vm_remap` of guest
/// PFNs, cached on the mapping and returned again on every later call. Its own
/// doc states the contract — "always revalidate first so a cached contig never
/// aliases PFNs after ReplacePhysical / guest recycle" — but the revalidation it
/// names cannot deliver that for a type-4 surface: with no `MappingInternal` it
/// re-resolves nothing and answers "resolvable" on a non-empty list alone. So a
/// writer holding this pointer is holding whatever those PFNs became, and a
/// full white frame poked through it is the `0xff`-filled freed guest heap the
/// crash census reads back.
///
/// Reads keep [`contig_for_span`]: a read through a drifted view returns another
/// process's bytes, which is a wrong picture and not a corrupted guest, and the
/// two losses want separate slugs.
fn contig_for_write<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    mapping_id: u32,
    span_end: u64,
    vouched: &mapper::PagesVouched,
    src: &[u8],
) -> Option<(usize, usize)> {
    if !vouched.covers(state, mapping_id) {
        crate::observe::fail(format!(
            "mapping_write contig mid={mapping_id} reason=vouch_stale need={span_end} \
             (the page list was cleared or replaced between the walk and this write)"
        ));
        return None;
    }
    let view = contig_for_span(state, host, mapping_id, span_end)?;
    // Every raw-pointer write in this file goes through here, and none of them
    // goes through `mapper::write_mapping_bytes` — they poke rows straight into
    // the view. So this is where those writes enter `observe::footprint`, and
    // without it the *largest* guest-write rail in the device would be missing
    // from a set whose whole use is answering "did we write there?".
    //
    // Marked over `[0, span_end)` because that is the extent this function
    // guarantees and the callers' row offsets are not visible here. That
    // over-marks the pages before a rect's first row — the surface's own pages,
    // never anyone else's, since the marking walks this mapping's page list — and
    // over-marking can only turn a miss into a hit. Under-marking would
    // manufacture the clean "we never wrote there" the set must never invent.
    mapper::note_mapping_write_footprint(state, mapping_id, 0, span_end);
    // And the payload census, for the same reason and with the same blind spot
    // if it is left out. The peer rails sample once per call in
    // `mapper::write_mapping_bytes` and `gva_view::write_gva_bytes`; this file
    // reaches neither, so without this the census reporting "no white payload in
    // any sample" would be reporting it about every rail except the one carrying
    // the pixels.
    //
    // `src` is the whole source image, and a partial write — skip ranges, a
    // changed-span pass — puts only some of it in guest RAM. That over-reports,
    // and the direction is deliberate: over-reporting a white run makes the
    // payload a *weaker* discriminator, while under-reporting would manufacture
    // the "we do not write white" that would clear this device on no evidence.
    crate::observe::footprint::note_written_payload(src);
    Some(view)
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

/// Mapping byte ranges a writeback must leave alone, ascending and disjoint.
///
/// Offsets are from the mapping's page 0, the same space `base_off`/`span_end`
/// are in, so a caller holding guest *page* addresses converts once with
/// [`crate::runtime::mapper::mapping_offsets_of_pages`] and everything below
/// stays in one coordinate system.
pub type SkipRanges<'a> = &'a [(u64, u64)];

/// The sub-ranges of `[start, end)` that are not covered by `skip`.
///
/// `skip` is ascending and disjoint, so one forward walk answers it. Kept
/// separate from the two writers below because they lay their bytes out
/// differently — one pokes a host view in place, the other stages a frame and
/// hands runs to the mapper — and the only thing they must agree on is *which
/// bytes are excluded*. Two open-coded walks would be two chances to disagree.
fn unskipped(start: u64, end: u64, skip: SkipRanges<'_>) -> Vec<(u64, u64)> {
    if skip.is_empty() {
        return if start < end {
            vec![(start, end)]
        } else {
            vec![]
        };
    }
    let mut out = Vec::new();
    let mut cur = start;
    for &(s, e) in skip {
        if e <= cur {
            continue;
        }
        if s >= end {
            break;
        }
        if s > cur {
            out.push((cur, s.min(end)));
        }
        cur = cur.max(e);
        if cur >= end {
            return out;
        }
    }
    if cur < end {
        out.push((cur, end));
    }
    out
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
    write_bgra8_skipping(state, host, mapping_id, src, src_stride, width, height, &[])
}

/// [`write_bgra8`], leaving `skip` untouched.
///
/// A deferred writeback holds a frame the device rendered and lands it in the
/// guest's pages later. If the guest CPU wrote some of those pages in between,
/// writing the whole frame loses the guest's stores and dropping the frame loses
/// the device's; `skip` is how the caller expresses the third answer, one page
/// at a time, from the hypervisor's own per-page report.
///
/// Everything else is unchanged, deliberately — including the cache refresh and
/// the epoch bump at the tail. The frame *is* what the device rendered, and the
/// host-side copies of it stay that; what `skip` decides is only which of those
/// bytes the guest's own memory is allowed to keep instead.
#[allow(
    clippy::too_many_arguments,
    reason = "the geometry the frame is in, plus the ranges its owner may not overwrite"
)]
pub fn write_bgra8_skipping<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mapping_id: u32,
    src: &[u8],
    src_stride: u32,
    width: u32,
    height: u32,
    skip: SkipRanges<'_>,
) -> bool {
    write_bgra8_inner(
        state,
        host,
        mapping_id,
        src,
        CacheOutcome::Publish(None),
        src_stride,
        width,
        height,
        skip,
    )
}

/// [`write_bgra8_skipping`] for a caller that owns its frame behind an `Arc`.
///
/// The tail of every non-skipping writeback publishes the frame to
/// [`crate::runtime::surface_cache`], and a caller holding only a borrow has to
/// build a second whole-frame buffer for it to keep. On the 8.29 MB composite
/// that copy costs 1.21 ms about 100 times a second — more than landing the same
/// bytes in the guest's own pages does. The cache already stores its frames
/// behind an `Arc` so that an entry and a deferred window can name one
/// allocation, so a caller that arrives holding one can publish it rather than
/// duplicate it.
///
/// The sharing conditions are checked rather than assumed, because the cache's
/// contract is a tight BGRA8 frame at the entry's geometry and an allocation that
/// is not one would be handed to every later reader as though it were: the
/// pointer has to be the one the rest of this write read from, the pitch has to
/// be the packed row length, and the allocation has to cover the whole frame.
/// Anything else takes the copying publish.
#[allow(
    clippy::too_many_arguments,
    reason = "the geometry the frame is in, plus the ranges its owner may not overwrite"
)]
pub fn write_bgra8_owned<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mapping_id: u32,
    src: &std::sync::Arc<Vec<u8>>,
    src_stride: u32,
    width: u32,
    height: u32,
    skip: SkipRanges<'_>,
) -> bool {
    write_bgra8_inner(
        state,
        host,
        mapping_id,
        src.as_slice(),
        CacheOutcome::Publish(Some(src)),
        src_stride,
        width,
        height,
        skip,
    )
}

/// [`write_bgra8_skipping`] for a caller whose frame is about to stop existing.
///
/// The other writers end by publishing the frame to
/// [`crate::runtime::surface_cache`], which is nearly free when the caller
/// already owns an `Arc` and a whole-frame copy when it does not. A caller
/// holding borrowed bytes — the deferred render flush reading a resident
/// through `engine::LeasedFrame`, which is a Vulkan staging buffer it gives
/// back a moment later — would pay that copy purely to fill a cache entry, and
/// `render_flush_cache_used` prices those entries at 0.4 %: 15 reads against
/// 3751 that nothing touched before the next flush replaced them.
///
/// So this writer drops the entry instead of refreshing it, and dropping is the
/// only correct alternative. Leaving the previous frame behind would serve a
/// later reader an old frame with nothing saying so, which is the stale-tile
/// class the fence binding exists to close. Every reader that misses falls
/// through to a source that does hold this frame — the surface's own guest
/// pages, which this write has just landed, or the resident it came out of —
/// so the miss costs a slower route to the same pixels and never wrong ones.
#[allow(
    clippy::too_many_arguments,
    reason = "the geometry the frame is in, plus the ranges its owner may not overwrite"
)]
pub fn write_bgra8_uncached<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mapping_id: u32,
    src: &[u8],
    src_stride: u32,
    width: u32,
    height: u32,
    skip: SkipRanges<'_>,
) -> bool {
    write_bgra8_inner(
        state,
        host,
        mapping_id,
        src,
        CacheOutcome::Invalidate,
        src_stride,
        width,
        height,
        skip,
    )
}

/// Land every deferred window overlapping the sample window a BGRA8 writeback
/// of this mapping would cover.
///
/// The same flush [`write_bgra8_uncached`] and its siblings make on their own
/// behalf, exposed so a caller can make it *before* it acquires the bytes it
/// intends to write.
///
/// `write_bgra8_uncached`'s frame is borrowed from the engine's readback buffer
/// under a lease, and a lease holder must not re-enter the engine: a teardown
/// waiting for that lease to come back holds the engine lock, so a holder that
/// asks for it stalls until the teardown's quiesce budget expires and then
/// reads freed memory. A deferred flush reached from inside the write is
/// exactly such a re-entry — `flush_render_one` reads a resident. Running the
/// flush first leaves the writer's own call nothing to find, because nothing
/// between the two arms a window: only a guest Store does, and no guest command
/// is decoded inside a writeback.
///
/// Answers false for a geometry that has no sample window, which is the same
/// geometry the writeback itself would refuse.
pub fn flush_windows_under_bgra8_write<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mapping_id: u32,
    width: u32,
    height: u32,
) -> bool {
    let Some(m) = state.mappings.get(&mapping_id) else {
        return false;
    };
    let (mw, mh, format) = mapping_write_geometry(m, width, height);
    let Some((base_off, _, span_end)) = type11_sample_window(m, mapping_id, mw, mh, format) else {
        return false;
    };
    crate::runtime::storage_flush::flush_intersecting(state, host, mapping_id, base_off, span_end);
    true
}

/// The geometry and pixel format a writeback to this mapping must land in.
///
/// A mapping that has declared its own (`has_geom`) owns the answer, and a
/// zero format there means BGRA8; one that has not takes the caller's geometry.
/// Factored out because the writeback and the pre-flush above must resolve it
/// identically — a pre-flush computed at a different extent than the write
/// would leave exactly the windows the write is about to land on.
fn mapping_write_geometry(m: &MappingEntry, width: u32, height: u32) -> (u32, u32, u16) {
    if m.has_geom {
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
    }
}

/// What a writeback leaves in the host surface cache when it is done.
enum CacheOutcome<'a> {
    /// Publish this frame as the mapping's entry, sharing the caller's
    /// allocation when it is one the cache's contract allows sharing.
    Publish(Option<&'a std::sync::Arc<Vec<u8>>>),
    /// Drop the mapping's entry. For a caller that cannot leave the cache
    /// naming its frame, because the memory holding it is about to be reused.
    Invalidate,
}

#[allow(
    clippy::too_many_arguments,
    reason = "the geometry the frame is in, its owner when it has one, plus the \
              ranges that owner may not overwrite"
)]
fn write_bgra8_inner<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mapping_id: u32,
    src: &[u8],
    cache: CacheOutcome<'_>,
    src_stride: u32,
    width: u32,
    height: u32,
    skip: SkipRanges<'_>,
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
    let (mw, mh, format) = mapping_write_geometry(m, width, height);
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
    // Taken after the flush, because the flush can invalidate this mapping, and
    // once for the whole frame, because the loop below writes a row at a time.
    let Some(vouched) = vouch_for_write(state, host, mapping_id, "bgra8") else {
        return false;
    };
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
    // Whether a row can go straight from `src` to the guest without passing
    // through `row`.
    //
    // `row` is the *conversion* destination: when the mapping's format is
    // already BGRA8 there is nothing to convert, and staging through it copied
    // every byte of the frame a second time — an extra 8 MB memcpy per flush on
    // the composite surface, ~106 times a second.
    //
    // Removing it is strictly less work for identical bytes, but do not go
    // looking for it in `readback_split`. It was landed on the prediction that
    // `write_us` would drop by the ~0.8 ms the same byte count costs elsewhere,
    // and a live driven boot then measured 2.79 ms per flush against 2.68 ms
    // before — no change outside run-to-run noise. The prediction was wrong
    // about *which* copy is expensive: `row` is a few KiB and stays in L1, so
    // filling it is nearly free, while the copy into cold guest pages is the
    // one that costs and is still there. `write_us` runs at ~3 GB/s against
    // ~9 GB/s for the readback's own memcpy of the identical frame, which is
    // the shape of a cache-cold scattered write, not of an avoidable pass.
    //
    // The consequence for whoever shrinks this next: the cost is bytes landing
    // in guest RAM, so fewer bytes helps proportionally and fewer staging hops
    // does not.
    //
    // Only sound while the row is byte-identical, which is why `tight` is
    // compared rather than assumed: `tight_row_bytes` is the format's own
    // packed row length, and if it ever disagrees with the source's `mw * 4`
    // the staged path still runs. That also keeps `row`'s reuse across rows
    // safe — a short source row would otherwise leave the previous row's bytes
    // in its tail.
    let direct_rows = rgba.is_none() && tight == (mw as usize) * (RGBA8_BPP as usize);

    use crate::runtime::drain::{
        SurfaceWritePhase, note_surface_write_path, note_surface_write_phase,
    };
    let frame_bytes = (mh as u64).saturating_mul(tight as u64);

    // Fast path: one packed view, poke rows in place.
    if let Some((ptr, _)) = contig_for_write(state, host, mapping_id, span_end, &vouched, src) {
        note_surface_write_path(true, frame_bytes);
        let land_started = std::time::Instant::now();
        // SAFETY: contig covers span_end; revalidated in ensure_contig_view.
        let base = unsafe { (ptr as *mut u8).add(base_off as usize) };
        for y in 0..mh {
            let src_off = (y as usize) * (src_stride as usize);
            let src_row_len = (mw as usize) * (RGBA8_BPP as usize);
            if src_off + src_row_len > src.len() {
                return false;
            }
            let src_row = &src[src_off..src_off + src_row_len];
            let row_bytes: &[u8] = if direct_rows {
                &src_row[..tight]
            } else {
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
                &row
            };
            // The row's destination in mapping-offset space, so the skip list —
            // which is in that space — is subtracted before any pointer exists.
            let row_off = base_off.saturating_add((y as u64).saturating_mul(bpr as u64));
            for (lo, hi) in unskipped(row_off, row_off.saturating_add(tight as u64), skip) {
                let within = (lo - row_off) as usize;
                let len = (hi - lo) as usize;
                let dst = unsafe { base.add((y as usize).saturating_mul(bpr) + within) };
                // SAFETY: `within + len <= tight <= row_bytes.len()`, and the
                // view covers span_end which is at or past this row's last byte.
                unsafe {
                    std::ptr::copy_nonoverlapping(row_bytes.as_ptr().add(within), dst, len);
                }
            }
        }
        note_surface_write_phase(
            SurfaceWritePhase::Land,
            land_started.elapsed().as_micros() as u64,
        );
    } else {
        note_surface_write_path(false, frame_bytes);
        let stage_started = std::time::Instant::now();
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
        // The staged buffer is `src` itself whenever the layout it would be
        // built into is the layout `src` already has: no conversion (the rows go
        // through untouched), the mapping's row pitch equal to the packed row
        // length (so there is no inter-row padding to zero-fill), and a source
        // pitch equal to it too (so the rows are already consecutive). Under all
        // three, byte `i` of the staged frame is byte `i` of `src` for every
        // `i < frame_len`, and building it copies 8 MB to produce a slice we are
        // holding.
        //
        // Each condition is load-bearing rather than defensive. Drop the pitch
        // equality and the gaps between rows — which the guest's allocator may
        // have given to something else — stop being zeroed and start carrying
        // whatever `src` has at those offsets. Drop `direct_rows` and the rows
        // are supposed to be converted on the way through.
        let staged: std::borrow::Cow<'_, [u8]> = if direct_rows
            && bpr == tight
            && src_stride as usize == tight
            && src.len() >= frame_len
        {
            std::borrow::Cow::Borrowed(&src[..frame_len])
        } else {
            let mut frame = vec![0u8; frame_len];
            for y in 0..mh {
                let src_off = (y as usize) * (src_stride as usize);
                let src_row_len = (mw as usize) * (RGBA8_BPP as usize);
                if src_off + src_row_len > src.len() {
                    return false;
                }
                let src_row = &src[src_off..src_off + src_row_len];
                let row_bytes: &[u8] = if direct_rows {
                    &src_row[..tight]
                } else {
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
                    &row
                };
                let dst_off = (y as usize).saturating_mul(bpr);
                if dst_off + tight > frame.len() {
                    return false;
                }
                frame[dst_off..dst_off + tight].copy_from_slice(&row_bytes[..tight]);
            }
            note_surface_write_phase(
                SurfaceWritePhase::Stage,
                stage_started.elapsed().as_micros() as u64,
            );
            std::borrow::Cow::Owned(frame)
        };
        let frame: &[u8] = staged.as_ref();
        let land_started = std::time::Instant::now();
        // One call per surviving run rather than one for the whole frame: the
        // mapper writes the bytes it is handed, so a skipped page has to be
        // absent from the slice rather than masked inside it.
        for (lo, hi) in unskipped(base_off, base_off.saturating_add(frame.len() as u64), skip) {
            let within = (lo - base_off) as usize;
            let len = (hi - lo) as usize;
            if !mapper::write_mapping_bytes(
                state,
                host,
                mapping_id,
                lo,
                &frame[within..within + len],
                &vouched,
            ) {
                return false;
            }
        }
        note_surface_write_phase(
            SurfaceWritePhase::Land,
            land_started.elapsed().as_micros() as u64,
        );
    }
    state.invalidate_storage_residency_window(mapping_id, base_off, span_end);
    let _ = state.mark_mapping_written(mapping_id);
    if !skip.is_empty() {
        // A skipping write leaves the guest's pages holding `src` everywhere
        // except the ranges the guest itself owns, so the pages are now the only
        // complete copy of this surface and no host-side copy of `src` is its
        // content any more. Both of them have to be told so.
        //
        // The byte cache would answer the type-11 LOAD seed, which prefers it
        // over the surface's own pages, and hand back exactly the bytes the
        // guest's stores were preserved *from*.
        crate::runtime::surface_cache::forget(state, mapping_id);
        // The resident `src` was read out of does not have the guest's stores
        // either, but it is NOT retired here: it is also the source other
        // deferred windows on the same identity are still going to flush from,
        // and withdrawing its content mid-drain loses their frames outright
        // (`chain_resident_land_fail reason=read_target`, `deferred_flush_lost
        // reason=resident_epoch_drift live=None`, both on 1920x1080 scanout
        // surfaces — measured, and a black screen).
        //
        // What disqualifies it instead is the `mark_mapping_written` above,
        // which advances `surface_content_epoch` past the resident's stamp. Both
        // rails that would bind a resident in place of this surface compare that
        // pair — the attachment LOAD elision always did, and the sampled ladder's
        // resident rung now does too. The caller that produced `src` must
        // therefore not hand the stamp back after a skipping write; see
        // `storage_flush::flush_render_one`.
        //
        // The guest-write stamp is re-taken, because the device has *adopted*
        // the guest's stores: they are in the pages it just wrote around.
        //
        // Withholding it was the defect this rail shipped with. The stamp is the
        // `since` every later `guest_written_pages` call is asked against, and
        // `page_gen` records the harvest that last saw each page written, never
        // resetting per consumer. So a stamp that does not move makes the skip
        // set grow monotonically: one full CPU repaint of a window marks every
        // page of it, and from then on every deferred flush of that surface
        // skips the entire extent and the device's composite never reaches guest
        // memory again. Measured live as a desktop that goes black and stays
        // black, at `render_flush_preserved_guest_write` ~65 a second, on a boot
        // whose sampled resident rung was gated off — so it was this rail and
        // not that one.
        //
        // It is honest as well as necessary: the stamp says "no host-side copy
        // is known stale relative to these pages", and after the two retirements
        // above there is no host-side copy at all.
        crate::runtime::mapper::stamp_guest_write_gen(state, host, mapping_id);
        return true;
    }
    let cache_started = std::time::Instant::now();
    let tight_frame = (mw as usize)
        .saturating_mul(mh as usize)
        .saturating_mul(RGBA8_BPP as usize);
    match cache {
        CacheOutcome::Invalidate => crate::runtime::surface_cache::forget(state, mapping_id),
        CacheOutcome::Publish(shared) => match shared.filter(|owner| {
            src_stride == mw.saturating_mul(RGBA8_BPP)
                && owner.len() >= tight_frame
                && std::ptr::eq(owner.as_ptr(), src.as_ptr())
        }) {
            Some(owner) => {
                crate::runtime::surface_cache::store_shared(state, mapping_id, mw, mh, owner.clone())
            }
            None => {
                crate::runtime::surface_cache::store_rows(state, mapping_id, mw, mh, src, src_stride)
            }
        },
    }
    note_surface_write_phase(
        SurfaceWritePhase::Cache,
        cache_started.elapsed().as_micros() as u64,
    );
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
    let (mw, mh, format) = mapping_write_geometry(m, width, height);
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
    // One proof for the whole image: the changed-span loop below writes each
    // differing row separately, and the walk is a translation per page.
    let Some(vouched) = vouch_for_write(state, host, mapping_id, "rgba8_changed") else {
        return false;
    };
    let contig = contig_for_write(state, host, mapping_id, span_end, &vouched, rgba);
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
                    &vouched,
                ) {
                    return false;
                }
            }
        } else if !mapper::write_mapping_bytes(state, host, mapping_id, row_moff, &native, &vouched)
        {
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
    let Some(vouched) = vouch_for_write(state, host, mapping_id, "raw_rows") else {
        return false;
    };
    if let Some((ptr, _)) = contig_for_write(state, host, mapping_id, span_end, &vouched, src) {
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
                &vouched,
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

/// Resolve the window a mapping's own latched geometry names, for a rectangle
/// addressed in that geometry rather than in an explicit plane window.
///
/// The resolution [`read_rect_raw`] and [`write_rect_raw`] share: the latched
/// format (BGRA8 when the mapping never declared one), the plane window
/// [`type11_sample_window`] decodes for it, and the texel size. Returns
/// `(base_offset, bytes_per_row, span_end, bytes_per_texel)`, or `None` when
/// the mapping is gone, carries no latched geometry, has no decodable window,
/// has an unknown format, or the rectangle leaves the surface.
fn mapping_geom_window(
    state: &DeviceState,
    mapping_id: u32,
    origin_x: u32,
    origin_y: u32,
    width: u32,
    height: u32,
) -> Option<(u64, u32, u64, u32)> {
    let m = state.mappings.get(&mapping_id)?;
    if !m.has_geom {
        return None;
    }
    let format = if m.format != 0 {
        m.format
    } else {
        MTL_FORMAT_BGRA8_UNORM
    };
    let (base_off, bpr, span_end) = type11_sample_window(m, mapping_id, m.width, m.height, format)?;
    let bpp = pixel_format::bytes_per_pixel(format)?;
    if origin_x.saturating_add(width) > m.width || origin_y.saturating_add(height) > m.height {
        return None;
    }
    Some((base_off, bpr, span_end, bpp))
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
    let Some((base_off, bpr, span_end, bpp)) =
        mapping_geom_window(state, mapping_id, origin_x, origin_y, width, height)
    else {
        return false;
    };
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
    let Some((base_off, bpr, span_end, bpp)) =
        mapping_geom_window(state, mapping_id, origin_x, origin_y, width, height)
    else {
        return false;
    };
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
    let Some(vouched) = vouch_for_write(state, host, mapping_id, "rect_raw") else {
        return false;
    };
    if let Some((ptr, _)) = contig_for_write(state, host, mapping_id, span_end, &vouched, src) {
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
            if !mapper::write_mapping_bytes(
                state,
                host,
                mapping_id,
                base_off,
                &src[..frame_len],
                &vouched,
            ) {
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
        if !mapper::write_mapping_bytes(state, host, mapping_id, base_off, &frame, &vouched) {
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
                &vouched,
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

    /// The gap walk is what both writeback paths subtract their skipped ranges
    /// with, so it is tested on its own: an off-by-one here is a row of pixels
    /// written into a page the guest owns, or a row of the guest's own bytes
    /// overwritten, and neither is visible in a frame until much later.
    #[test]
    fn unskipped_returns_exactly_the_bytes_no_range_covers() {
        assert_eq!(unskipped(0, 10, &[]), vec![(0, 10)]);
        assert_eq!(unskipped(10, 10, &[]), vec![]);
        // Interior, leading, trailing, and total cover.
        assert_eq!(unskipped(0, 10, &[(4, 6)]), vec![(0, 4), (6, 10)]);
        assert_eq!(unskipped(0, 10, &[(0, 4)]), vec![(4, 10)]);
        assert_eq!(unskipped(0, 10, &[(6, 10)]), vec![(0, 6)]);
        assert_eq!(unskipped(0, 10, &[(0, 10)]), vec![]);
        // Ranges outside the window contribute nothing, on either side.
        assert_eq!(unskipped(10, 20, &[(0, 5), (30, 40)]), vec![(10, 20)]);
        // Partial overlap at both ends, and several ranges in one window.
        assert_eq!(
            unskipped(10, 30, &[(5, 12), (16, 18), (28, 35)]),
            vec![(12, 16), (18, 28)]
        );
    }

    /// Every row must land at its own offset, with content that can tell rows
    /// apart.
    ///
    /// The skip test below fills the frame with one repeated byte, so it proves
    /// which *pages* were written and nothing at all about which row went
    /// where: a writeback that repeated row 0 sixty-four times, or that shifted
    /// every row by one, passes it unchanged. That is not hypothetical — a
    /// BGRA8 row no longer passes through the conversion scratch buffer on its
    /// way to the guest, so the source offset, the tight row length and the
    /// destination stride are now read from three places that used to be two.
    ///
    /// Both storage shapes, because they place rows by different means, and a
    /// non-BGRA format so the staged path is exercised beside the direct one.
    #[test]
    fn a_writeback_lands_every_row_at_its_own_offset() {
        use crate::model::PAGE_SHIFT_X86;
        const PAGE: u64 = 1 << PAGE_SHIFT_X86;
        const W: u32 = 64;
        const H: u32 = 64;
        const BPR: usize = (W * 4) as usize;

        for packed in [true, false] {
            for format in [MTL_FORMAT_BGRA8_UNORM, pixel_format::MTL_FORMAT_RGBA8_UNORM] {
                let mut state = DeviceState::new(DeviceId(2), PAGE_SHIFT_X86);
                let mut host = FakeHost::new();
                host.strict_linux_map = !packed;
                let base_pfn = 0x40u32;
                host.map_range((base_pfn as u64) << PAGE_SHIFT_X86, 8 * PAGE as usize, 0);
                let order: Vec<u32> = if packed {
                    (0..4).collect()
                } else {
                    vec![3, 2, 1, 0]
                };
                let entries: Vec<u32> = order
                    .iter()
                    .map(|i| ((base_pfn + i) << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID)
                    .collect();
                state.map_surface(4);
                state.attach_mapping_internal(4, 0);
                let m = state.mappings.get_mut(&4).unwrap();
                m.mapping_internal = 1;
                m.page_entries = entries;
                assert!(state.set_mapping_geom(4, W, H, format));

                // Row y is filled with the byte y in every channel, so a row
                // that lands at the wrong offset is visible as the wrong value
                // and a duplicated row is visible as a repeat. Channel order
                // does not matter to that, which is what lets one frame test
                // both formats.
                let mut frame = vec![0u8; (W * H * 4) as usize];
                for y in 0..H as usize {
                    frame[y * BPR..(y + 1) * BPR].fill(y as u8);
                }
                assert!(write_bgra8_skipping(
                    &mut state, &mut host, 4, &frame, W * 4, W, H, &[]
                ));

                for y in 0..H as usize {
                    // Which guest page this row's first byte lives in, and where
                    // in it, walking the same page list the mapping declares.
                    let off = y * BPR;
                    let gpa = ((base_pfn as u64 + order[off / PAGE as usize] as u64)
                        << PAGE_SHIFT_X86)
                        + (off as u64 % PAGE);
                    let mut got = [0u8; 4];
                    host.read_gpa(gpa, &mut got).unwrap();
                    assert!(
                        got.iter().all(|b| *b == y as u8),
                        "packed={packed} fmt={format:#x} row {y} must read {y:#x}, got {got:?}"
                    );
                }
            }
        }
    }

    /// A writeback told to skip a page leaves that page exactly as the guest
    /// left it, writes every other page, and stops claiming the frame is the
    /// mapping's content.
    ///
    /// Both storage shapes are exercised, because they place bytes by different
    /// means — the packed one pokes a host view through a raw pointer, the
    /// fragmented one stages a frame and hands runs to the mapper — and a skip
    /// honoured by one and not the other is a defect that only appears on
    /// whichever guest allocation happens to be scattered.
    ///
    /// Dropping the host cache at the tail is half the test: it would otherwise
    /// answer the type-11 LOAD seed with the very bytes the guest's stores were
    /// preserved from.
    #[test]
    fn a_skipping_writeback_leaves_the_guest_its_own_pages() {
        use crate::model::PAGE_SHIFT_X86;
        const PAGE: u64 = 1 << PAGE_SHIFT_X86;
        // 64x64 BGRA8 is 256 bytes a row, 16 KiB in four x86 pages.
        const W: u32 = 64;
        const H: u32 = 64;

        for packed in [true, false] {
            let mut state = DeviceState::new(DeviceId(2), PAGE_SHIFT_X86);
            let mut host = FakeHost::new();
            host.strict_linux_map = !packed;
            let base_pfn = 0x40u32;
            host.map_range((base_pfn as u64) << PAGE_SHIFT_X86, 8 * PAGE as usize, 0x55);
            // Reversed page order is a fragmented list to `map_pages`; the same
            // four pages either way, so the guest bytes are comparable.
            let order: Vec<u32> = if packed {
                (0..4).collect()
            } else {
                vec![3, 2, 1, 0]
            };
            let entries: Vec<u32> = order
                .iter()
                .map(|i| ((base_pfn + i) << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID)
                .collect();
            state.map_surface(4);
            state.attach_mapping_internal(4, 0);
            let m = state.mappings.get_mut(&4).unwrap();
            m.mapping_internal = 1;
            m.page_entries = entries;
            assert!(state.set_mapping_geom(4, W, H, MTL_FORMAT_BGRA8_UNORM));
            crate::runtime::surface_cache::store(
                &mut state,
                4,
                W,
                H,
                vec![0u8; (W * H * 4) as usize],
            );

            let frame = vec![0xAAu8; (W * H * 4) as usize];
            // Page 2 of the surface, in mapping-offset space.
            let skip = [(2 * PAGE, 3 * PAGE)];
            assert!(write_bgra8_skipping(
                &mut state,
                &mut host,
                4,
                &frame,
                W * 4,
                W,
                H,
                &skip
            ));

            for page in 0..4u64 {
                let gpa = (base_pfn as u64 + order[page as usize] as u64) << PAGE_SHIFT_X86;
                let mut got = [0u8; 8];
                host.read_gpa(gpa, &mut got).unwrap();
                let want = if page == 2 { 0x55 } else { 0xAA };
                assert!(
                    got.iter().all(|b| *b == want),
                    "packed={packed} surface page {page} must read {want:#x}, got {got:?}"
                );
            }
            assert!(
                crate::runtime::surface_cache::get(&state, 4, W, H).is_none(),
                "packed={packed}: the cache must stop answering for a mapping the frame no longer describes"
            );
        }
    }

    /// A skipping write must re-take the guest-write stamp, or the set of pages
    /// it skips grows monotonically until it covers the whole surface and the
    /// device's composite never reaches guest memory again.
    ///
    /// `since` for every later `guest_written_pages` call is this stamp, and the
    /// host's `page_gen` records the harvest that last saw each page written and
    /// never resets per consumer. So a stamp that does not move keeps naming
    /// every page the guest has *ever* written since it was taken. One full CPU
    /// repaint of a window marks all of them, and from then on each deferred
    /// flush of that surface skips the entire extent.
    ///
    /// That is what this rail shipped with, and it is a desktop that goes black
    /// and stays black — reproduced live at `render_flush_preserved_guest_write`
    /// ~65 a second on a boot with the sampled resident rung gated off, which is
    /// what placed it here rather than there.
    ///
    /// The stamp is honest because of what runs beside it: the byte cache is
    /// dropped and the resident's content claim withdrawn, so when it says "no
    /// host-side copy is known stale relative to these pages" there is no
    /// host-side copy at all.
    #[test]
    fn a_skipping_writeback_re_takes_the_stamp_so_the_skip_set_cannot_only_grow() {
        use crate::model::PAGE_SHIFT_X86;
        const PAGE: u64 = 1 << PAGE_SHIFT_X86;
        const W: u32 = 64;
        const H: u32 = 64;

        let mut state = DeviceState::new(DeviceId(3), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        let base_pfn = 0x40u32;
        host.map_range((base_pfn as u64) << PAGE_SHIFT_X86, 8 * PAGE as usize, 0x55);
        let entries: Vec<u32> = (0..4)
            .map(|i| ((base_pfn + i) << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID)
            .collect();
        state.map_surface(4);
        state.attach_mapping_internal(4, 0);
        let m = state.mappings.get_mut(&4).unwrap();
        m.mapping_internal = 1;
        m.page_entries = entries;
        assert!(state.set_mapping_geom(4, W, H, MTL_FORMAT_BGRA8_UNORM));

        // A Store arms the witness, then the guest paints page 2 of the surface.
        crate::runtime::mapper::stamp_guest_write_gen(&mut state, &mut host, 4);
        let token = state.mappings[&4].guest_write_token;
        assert_ne!(token, 0, "the fake host must be able to watch these pages");
        host.guest_wrote_page((base_pfn as u64 + 2) << PAGE_SHIFT_X86);

        let stamped_before = state.mappings[&4].guest_write_gen_at_store;
        assert!(
            !host
                .guest_written_pages(token, stamped_before)
                .unwrap()
                .is_empty(),
            "the guest's write must be visible against the stamp the flush would use"
        );

        let frame = vec![0xAAu8; (W * H * 4) as usize];
        assert!(write_bgra8_skipping(
            &mut state,
            &mut host,
            4,
            &frame,
            W * 4,
            W,
            H,
            &[(2 * PAGE, 3 * PAGE)]
        ));

        let stamped_after = state.mappings[&4].guest_write_gen_at_store;
        assert_ne!(
            stamped_after, 0,
            "a skipping write that leaves the stamp at 0 makes every later flush skip everything"
        );
        assert_eq!(
            host.guest_written_pages(token, stamped_after),
            Some(Vec::new()),
            "the write the device just honoured must not be named again by the next flush"
        );
    }

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
        let vouched = mapper::vouch_mapping_pages_verdict(&mut state, &host, 7)
            .1
            .expect("no walk to contradict");
        assert!(mapper::write_mapping_bytes(
            &mut state, &mut host, 7, 16, &[0u8; 32], &vouched
        ));
        assert!(!state.compute_storage_residency.contains_key(&hit));
        assert!(state.compute_storage_residency.contains_key(&survivor));
    }

    /// A direct type-11 writeback must not land in a page the guest re-pointed
    /// away, and this asserts it in the currency of the bug: the bytes of the
    /// page the surface moved to.
    ///
    /// The page-drift witness shipped with exactly one caller — the deferred
    /// render flush — so this rail, which writes a full frame of pixels through
    /// `MappingEntry::page_entries`, was unguarded. The crash reports are the
    /// receipt: WindowServer aborting in `small_free_list_remove_ptr_no_clear`,
    /// and guest-kernel kalloc poison finding whole freed elements filled with
    /// `0xff` from offset 0 — opaque white BGRA in memory already handed to
    /// somebody else.
    ///
    /// So the fixture recycles a page the way the guest does: adopt a list walked
    /// through a live task page table, write a frame (which must land), re-point
    /// the PTE with no packet — nothing bumps `map_generation`, which is the
    /// whole defect — and require the second write to refuse. `data1` stands for
    /// whatever the guest gave those pages to next, seeded with a pattern rather
    /// than zeroes so "refused" and "wrote zeroes" cannot be confused.
    #[test]
    fn a_repointed_surface_refuses_the_write_and_leaves_the_new_owner_alone() {
        use crate::contract::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
        use crate::model::{Type4Walk, PAGE_SHIFT_X86};

        let page = 1u64 << PAGE_SHIFT_X86;
        let mut host = FakeHost::new();
        let dir_gpa = 2u64 << PAGE_SHIFT_X86;
        let root_gpa = 3u64 << PAGE_SHIFT_X86;
        let data0 = 4u64 << PAGE_SHIFT_X86;
        let data1 = 10u64 << PAGE_SHIFT_X86;
        host.map_range(dir_gpa, page as usize, 0);
        host.map_range(root_gpa, page as usize, 0);
        host.map_range(data0, page as usize, 0);
        host.map_range(data1, page as usize, 0);

        let st32 = |b: &mut [u8], v: u32| b[..4].copy_from_slice(&v.to_le_bytes());
        let mut d = [0u8; 8];
        st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 3);
        st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
        host.write_gpa(dir_gpa, &d).unwrap();
        let mut pte = [0u8; 4];
        st32(&mut pte, (data0 >> PAGE_SHIFT_X86) as u32);
        host.write_gpa(root_gpa, &pte).unwrap();

        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        assert!(state.define_task(1, page, 2));
        let mid = 6;
        state.map_surface(mid);
        {
            let m = state.mappings.get_mut(&mid).unwrap();
            m.mapped = true;
            m.page_entries = vec![
                (((data0 >> PAGE_SHIFT_X86) as u32) << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID,
            ];
            m.type4_walk = Some(Type4Walk {
                task_id: 1,
                backing_pfn: 0,
                map_generation: m.map_generation,
            });
        }

        // A tight 8x4 BGRA8 frame of opaque white — the payload the crash census
        // reads back out of freed guest memory.
        let (w, h) = (8u32, 4u32);
        let frame = vec![0xffu8; (w * h * RGBA8_BPP) as usize];
        let stride = w * RGBA8_BPP;
        assert!(
            write_bgra8(&mut state, &mut host, mid, &frame, stride, w, h),
            "the list was just walked from this page table, so the frame lands"
        );
        let mut landed = [0u8; 16];
        host.read_gpa(data0, &mut landed).unwrap();
        assert_eq!(landed, [0xffu8; 16], "the first frame reached data0");

        // Now the guest reclaims that page and hands it to something else, which
        // writes its own bytes there — a malloc small-zone region, say, whose
        // free-list pointers live *inside* the freed blocks. `0x5a` stands for
        // them. This is the step the crash reports are about: the corruption
        // lands in the page the surface *left*, not the one it moved to, because
        // the device's cached contig view is a `mach_vm_remap` of the old PFNs
        // and keeps resolving there.
        host.write_gpa(data0, &[0x5au8; 16]).unwrap();

        // And the surface is re-pointed. No MapMemory2, no UnmapMemory, no
        // ReplacePhysical — nothing on the wire, so nothing bumps the
        // incarnation.
        let generation_before = state.mappings.get(&mid).unwrap().map_generation;
        st32(&mut pte, (data1 >> PAGE_SHIFT_X86) as u32);
        host.write_gpa(root_gpa, &pte).unwrap();
        assert_eq!(
            state.mappings.get(&mid).unwrap().map_generation,
            generation_before,
            "no packet arrived, so nothing bumped the incarnation"
        );

        let refused = !write_bgra8(&mut state, &mut host, mid, &frame, stride, w, h);
        // The memory assertion comes first deliberately. A return value is this
        // crate's opinion about what it did; `data0` is what the guest will
        // actually find in its heap, and that is the claim the crash reports
        // dispute. Asserting the opinion first would let a rail that returns
        // false *after* writing pass the stronger half unread.
        let mut recycled = [0u8; 16];
        host.read_gpa(data0, &mut recycled).unwrap();
        assert_eq!(
            recycled, [0x5au8; 16],
            "the page the guest took away must still hold its new owner's bytes \
             — this is the guest heap corruption the whole goal is about"
        );
        assert!(
            refused,
            "and the caller is told, so a lost frame is never read as a landed one"
        );

        // Refusing once is not enough: `page_entries` is what every later reader
        // and writer resolves through, so a list a fresh walk has contradicted
        // has to stop being believed rather than be skipped once.
        assert!(
            state.mappings.get(&mid).unwrap().page_entries.is_empty(),
            "the contradicted list is dropped, so the next bind re-resolves"
        );
        assert_ne!(
            state.mappings.get(&mid).unwrap().map_generation,
            generation_before,
            "and every window still armed against the old incarnation refuses \
             on the map_generation check it already had"
        );
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

    /// The BGRA row writers reach `observe::footprint`.
    ///
    /// This is the rail the first cut of the footprint's completeness gate
    /// missed, and it is the biggest one in the device. These writers never call
    /// `mapper::write_mapping_bytes` and never call `HostOps::map_pages`: they
    /// take a contig view through `contig_for_write` and poke rows straight into
    /// it. A gate that scanned for `map_pages` callers therefore scored this
    /// file as reaching guest RAM by no mechanism at all — which was true of the
    /// needle and false of the file.
    ///
    /// A missing mark here would have left the footprint empty of nearly every
    /// pixel this device writes, and an empty set answers "we never wrote there"
    /// to every panic it is asked about.
    #[test]
    fn a_bgra_row_write_marks_the_footprint_through_its_contig_view() {
        use crate::model::PAGE_SHIFT_X86;
        use crate::observe::footprint;

        let _fp = footprint::exclusive_for_tests();
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        let page = 1u64 << PAGE_SHIFT_X86;
        // Adjacent so the contig view packs — this is the path production takes.
        let gpa0 = 0x5000_0000u64;
        host.map_range(gpa0, 2 * page as usize, 0);
        let mid = 12u32;
        state.map_surface(mid);
        {
            let m = state.mappings.get_mut(&mid).unwrap();
            m.mapped = true;
            m.mapping_internal = 1;
            m.page_entries = vec![
                (((gpa0 >> PAGE_SHIFT_X86) as u32) << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID,
                ((((gpa0 + page) >> PAGE_SHIFT_X86) as u32) << PAGE_ENTRY_PFN_SHIFT)
                    | PAGE_ENTRY_VALID,
            ];
        }
        assert!(state.set_mapping_geom(mid, 2, 2, MTL_FORMAT_BGRA8_UNORM));
        assert!(
            mapper::ensure_contig_view(&mut state, &mut host, mid).is_some(),
            "the fixture must take the contig path or it tests the other rail"
        );
        let src = [0xFFu8; 16];
        assert!(write_bgra8(&mut state, &mut host, mid, &src, 8, 2, 2));

        assert!(
            footprint::wrote_gpa(gpa0),
            "the surface's first frame must be in the set"
        );
        assert!(
            !footprint::wrote_gpa(gpa0 + 8 * page),
            "and nothing beyond the surface"
        );
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
                armed_stamp_seq: 0,
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
