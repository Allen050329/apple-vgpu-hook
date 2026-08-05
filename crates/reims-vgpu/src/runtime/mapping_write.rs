//! Write host BGRA8 into a guest IOSurface mapping (render writeback).
//!
//! Product writes go **only** through a revalidated contiguous HostOps view
//! (`map_pages`) — never `write_gpa` fragment walks over cached PFNs (freelist
//! `0xff000000ff000000` class). Always bumps [`DeviceState::mark_mapping_written`]
//! on success.

use crate::contract::iosurface_pages::{packed_span_estimate, sample_window_from_device_desc};
use crate::contract::pixel_format::{
    self, convert_rgba8_to_row, convert_row_to_rgba8, MTL_FORMAT_BGRA8_UNORM, RGBA8_BPP,
};
use crate::model::{scanout_extent_ok, DeviceState, MappingEntry, MAX_SCANOUT_DIM};
use crate::runtime::host::{HostMemory, HostOps};
use crate::runtime::mapper;

/// Why one render writeback did not land its frame in the guest's pages.
///
/// [`write_bgra8_inner`] has fifteen refusal sites and used to answer all of them
/// with a bare `false`, which its caller rendered as a single
/// `deferred_flush_lost reason=write_refused`. That is the defect the decline
/// vocabulary exists to prevent: the composite surface is the largest frame this
/// device moves, and a reader watching that slug fire could tell that the
/// wallpaper had been dropped but not whether the mapping had gone, its geometry
/// had moved under the armed window, the page walk had refused, or the source
/// buffer was short. Those have four different fixes.
///
/// One variant per check, carrying the values that decide it. The class currently
/// reads **zero across every accumulated boot log**, so this is an instrument for
/// a failure that is not happening rather than a repair for one that is — which is
/// exactly when it is cheap to install and exactly when nobody remembers to.
#[derive(Debug)]
pub enum SurfaceWriteRefusal {
    /// A zero or over-large rect. `MAX_SCANOUT_DIM` is the bound.
    Geometry { width: u32, height: u32 },
    /// The source's row pitch cannot hold `width` BGRA8 texels.
    SourceStride { src_stride: u32, width: u32 },
    /// No such mapping. The surface went away between the arm and the landing.
    MappingAbsent,
    /// The mapping is unmapped or has no page list, so there is nowhere to write.
    MappingNotResident,
    /// **The mapping's latched geometry is not the geometry of the frame being
    /// landed.** A deferred window carries the rect it was armed with, and a
    /// wallpaper or appearance change re-publishes the surface at another one.
    /// Landing the old frame at the new pitch would skew it, so it is refused.
    GeometryMoved {
        latched_width: u32,
        latched_height: u32,
        frame_width: u32,
        frame_height: u32,
    },
    /// The sample window could not be resolved from the surface descriptor.
    WindowUnresolved {
        width: u32,
        height: u32,
        format: u16,
    },
    /// The page walk refused to vouch for the mapping's page list.
    PagesNotOurs,
    /// The format has no packed row length, so there is no rect to write.
    FormatRowLength { format: u16 },
    /// The source buffer ends before the row this write is up to.
    SourceShort { need: usize, have: usize, row: u32 },
    /// A row would not convert into the mapping's pixel format.
    RowConvert { format: u16, row: u32 },
    /// The staged frame's extent overflowed, so the rows do not describe a buffer.
    FrameExtent { bpr: usize, height: u32 },
    /// The staged frame ends before the row being placed in it.
    StagedShort { need: usize, have: usize, row: u32 },
    /// The mapper refused to write a run of the frame into the guest's pages.
    MapperWrite { lo: u64, len: usize },
    /// The seed (previous-frame) buffer ends before the frame it must diff
    /// against. Distinct from [`Self::SourceShort`] because the two buffers come
    /// from different producers, and a log that conflated them could not say
    /// which one to go and look at.
    SeedShort { need: usize, have: usize },
    /// A seed row would not convert into the mapping's pixel format. Same
    /// distinction from [`Self::RowConvert`], and the same reason.
    SeedRowConvert { format: u16, row: u32 },
}

impl crate::observe::decline::Decline for SurfaceWriteRefusal {
    fn slug(&self) -> &'static str {
        match self {
            Self::Geometry { .. } => "surface_write_geometry",
            Self::SourceStride { .. } => "surface_write_source_stride",
            Self::MappingAbsent => "surface_write_mapping_absent",
            Self::MappingNotResident => "surface_write_mapping_not_resident",
            Self::GeometryMoved { .. } => "surface_write_geometry_moved",
            Self::WindowUnresolved { .. } => "surface_write_window_unresolved",
            Self::PagesNotOurs => "surface_write_pages_not_ours",
            Self::FormatRowLength { .. } => "surface_write_format_row_length",
            Self::SourceShort { .. } => "surface_write_source_short",
            Self::RowConvert { .. } => "surface_write_row_convert",
            Self::FrameExtent { .. } => "surface_write_frame_extent",
            Self::StagedShort { .. } => "surface_write_staged_short",
            Self::MapperWrite { .. } => "surface_write_mapper_write",
            Self::SeedShort { .. } => "surface_write_seed_short",
            Self::SeedRowConvert { .. } => "surface_write_seed_row_convert",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::Geometry { width, height } => vec![
                ("geom", format!("{width}x{height}")),
                ("max", MAX_SCANOUT_DIM.to_string()),
            ],
            Self::SourceStride { src_stride, width } => vec![
                ("src_stride", src_stride.to_string()),
                ("need", (width.saturating_mul(RGBA8_BPP)).to_string()),
            ],
            Self::MappingAbsent | Self::MappingNotResident | Self::PagesNotOurs => Vec::new(),
            Self::GeometryMoved {
                latched_width,
                latched_height,
                frame_width,
                frame_height,
            } => vec![
                ("latched", format!("{latched_width}x{latched_height}")),
                ("frame", format!("{frame_width}x{frame_height}")),
            ],
            Self::WindowUnresolved {
                width,
                height,
                format,
            } => vec![
                ("geom", format!("{width}x{height}")),
                ("fmt", format!("{format:#x}")),
            ],
            Self::FormatRowLength { format } => vec![("fmt", format!("{format:#x}"))],
            Self::SourceShort { need, have, row } => vec![
                ("need", need.to_string()),
                ("have", have.to_string()),
                ("row", row.to_string()),
            ],
            Self::RowConvert { format, row } => {
                vec![("fmt", format!("{format:#x}")), ("row", row.to_string())]
            }
            Self::FrameExtent { bpr, height } => {
                vec![("bpr", bpr.to_string()), ("height", height.to_string())]
            }
            Self::StagedShort { need, have, row } => vec![
                ("need", need.to_string()),
                ("have", have.to_string()),
                ("row", row.to_string()),
            ],
            Self::MapperWrite { lo, len } => {
                vec![("lo", format!("{lo:#x}")), ("len", len.to_string())]
            }
            Self::SeedShort { need, have } => {
                vec![("need", need.to_string()), ("have", have.to_string())]
            }
            Self::SeedRowConvert { format, row } => {
                vec![("fmt", format!("{format:#x}")), ("row", row.to_string())]
            }
        }
    }
}

/// Report one writeback refusal and answer `false` for the caller to return.
///
/// Latched per `(check, mapping)`: a surface whose geometry has moved refuses
/// every frame until something re-arms it, and the second line says nothing the
/// first did not. The route beside it carries the magnitude, which is what
/// [`crate::observe::emit::Emit::fail_once`]'s contract asks for.
fn refuse(mapping_id: u32, why: SurfaceWriteRefusal) -> bool {
    use crate::observe::decline::Decline;
    crate::runtime::drain::note_store_route(why.slug());
    crate::observe::emit::Emit::decline("surface_write", &why)
        .field("mid", mapping_id)
        .fail_once(u64::from(mapping_id));
    false
}

/// Resolve the sample window a texture of this geometry occupies inside its
/// mapping, for both wire families.
///
/// Two states the device used to answer identically, and the whole point of this
/// function is that they are not the same:
///
/// - **The mapping has published no descriptor.** `MappingInternal.descriptor`
///   reads zero until the guest fills it, which `mapper::resolve` documents as a
///   real state rather than a failure, and the geometry then comes from the
///   type-11 texture object instead. There are no plane records to confuse here;
///   the single unknown is the pitch, and [`packed_span_estimate`]'s aligned row
///   stands in for it over a surface starting at offset 0.
/// - **The descriptor is published and resolves nothing.** Here the guest *has*
///   told us the layout and the texture cannot be placed in it: its geometry
///   matched no plane record, or — the case that matters — it matched more than
///   one. A v0a8 surface's Y and alpha planes are both R8 at the luma geometry,
///   so the scan cannot tell them apart *by construction*, and the packed window
///   over plane 0 is a coin flip that reads as success at every layer above.
///
/// The second case declines, and callers answer it with a named refusal. That is
/// the difference between a bind that is lost visibly and one that samples luma
/// for alpha with nothing in the device able to say so.
///
/// Neither case is reached on a healthy x86 desktop. Measured on driven Ventura
/// boots with a Safari window drag, both with the dma-buf import available and
/// with `REIMS_VGPU_DMABUF=off` — which is the run that matters here, because a
/// capable host takes the import for every guest window and leaves the copying
/// rails at zero. With the gate closed (`dma_buf_import=disabled_by_env`,
/// `guest_dmabuf_*` absent) the copying rails carried the whole workload —
/// `rt_type5_view_same` 7396, `t11rung_resident` 19177, `surface_flush` 7396
/// against 55080 draws — and **no window failed to resolve**. Every bind came
/// from a published descriptor, so the estimate above is the state before the
/// guest fills one rather than a rung this device leans on.
fn sample_window(
    m: &MappingEntry,
    plane_index: Option<u32>,
    width: u32,
    height: u32,
    format: u16,
) -> Option<(u64, u32, u64)> {
    let Some(desc) = m.device_desc_complete() else {
        let end = packed_span_estimate(format, width, height)?;
        // The estimate is a whole number of aligned rows, so dividing it back
        // out is the row it was built from rather than a second derivation.
        return Some((0, (end / u64::from(height)) as u32, end));
    };
    sample_window_from_device_desc(Some(desc), plane_index, format, width, height)
}

/// Resolve the sample window for a type-11 texture binding on a mapping.
///
/// Type-11 is the case with **no wire plane index**: nothing on the wire names
/// which plane the texture wants, so a multi-plane surface is resolved by
/// matching width, height and bytes-per-element, and the plane is taken only
/// when exactly one matches. See [`sample_window`] for what each outcome means.
pub fn type11_sample_window(
    m: &MappingEntry,
    width: u32,
    height: u32,
    format: u16,
) -> Option<(u64, u32, u64)> {
    sample_window(m, None, width, height, format)
}

/// Resolve the sample window for a type-5 serialized view, which — unlike
/// type-11 — carries the IOSurface plane index on the wire (type-5 record
/// `+0x20`).
///
/// Every type-5 consumer must come through here rather than through
/// [`type11_sample_window`], and the distinction is not cosmetic: the wire index
/// names the plane record directly, and it is the only key that separates
/// same-geometry planes. Handing a type-5 view's geometry to the type-11 scan
/// drops that index, so a bind the wire said was alpha resolves against
/// whichever same-geometry plane the scan happens to reach.
pub fn type5_sample_window(
    m: &MappingEntry,
    plane_index: u32,
    width: u32,
    height: u32,
    format: u16,
) -> Option<(u64, u32, u64)> {
    sample_window(m, Some(plane_index), width, height, format)
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
    // The other reader of these writes, and for the same reason: the hypervisor's
    // dirty bitmap witnesses guest stores only, so a copy vouched for by "the
    // guest has not written" is stale the moment this rail runs. Recorded beside
    // the footprint mark rather than in each caller, so the two cannot drift and
    // a new caller inherits both.
    state.note_host_wrote_mapping(mapping_id);
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
/// Writes the frame whole. Its one caller is the deferred render flush, which
/// preserves nothing by design — see
/// [`crate::runtime::storage_flush`]'s `note_render_flush_over_guest_write` for
/// why the witness a narrowing would rest on cannot answer. A caller that does
/// need to skip has [`write_bgra8_skipping`]; adding the parameter back here
/// belongs with the caller that can fill it.
pub fn write_bgra8_owned<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mapping_id: u32,
    src: &std::sync::Arc<Vec<u8>>,
    src_stride: u32,
    width: u32,
    height: u32,
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
        &[],
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
/// Writes the frame whole, for the same reason [`write_bgra8_owned`] does.
pub fn write_bgra8_uncached<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mapping_id: u32,
    src: &[u8],
    src_stride: u32,
    width: u32,
    height: u32,
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
        &[],
    )
}

/// A check that stopped a resident's frame from reaching the guest's pages
/// without a host copy, so the flush owes the copying rail instead.
///
/// Every variant is a routing answer and not a loss — the caller still lands the
/// frame — but each one is a whole frame's worth of memcpy the device paid twice
/// over, on the rail that is 69% of the drain worker's time. So they are named
/// individually: "the GPU writeback declined" cannot tell a host with no
/// `/dev/udmabuf` from a surface whose row pitch is not a whole texel, and those
/// have different fixes.
#[cfg(feature = "backend-vulkan")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GpuWritebackDecline {
    /// A zero or over-large rect, or a mapping that is gone or unmapped. The
    /// copying rail refuses these too, by its own [`SurfaceWriteRefusal`]; this
    /// path declines before making the guest pay two refusals for one cause.
    NotWritable,
    /// The mapping's declared geometry is not the frame's. The copying rail
    /// reports this as `GeometryMoved`; here it means the same thing and the
    /// same rail will say so.
    GeometryMoved {
        latched_width: u32,
        latched_height: u32,
        frame_width: u32,
        frame_height: u32,
    },
    /// No sample window resolves for this geometry, so there is no destination
    /// offset or row pitch to copy against.
    WindowUnresolved {
        width: u32,
        height: u32,
        format: u16,
    },
    /// The mapping's pixel format is not the one the resident holds, so landing
    /// it needs a per-row conversion. A buffer→image copy performs none, which
    /// is why this is a routing answer rather than something to work around.
    FormatNeedsConversion { format: u16 },
    /// The guest's row pitch is not a whole number of texels, so it cannot be
    /// expressed as `bufferRowLength`.
    PitchNotTexels { bpr: u32 },
    /// The frame's first texel does not start on a 4-byte boundary within its
    /// page. `VkBufferImageCopy::bufferOffset` must be a multiple of the texel
    /// block size, and a copy that ignored this is undefined rather than
    /// misaligned.
    OffsetNotTexelAligned { in_page: u64 },
    /// The mapping's page list does not cover the sample window.
    PageListShort { need: usize, have: usize },
    /// A page in the window carries no valid entry, so there is no guest frame
    /// to name in the export list.
    PageUnbacked { index: usize },
    /// The page walk refused: these are no longer provably the mapping's pages.
    /// The copying rail refuses for the same reason and reports it.
    PagesNotOurs,
    /// This host cannot export the window as a dma-buf. Latched and reported by
    /// [`crate::runtime::guest_dmabuf`], which knows whether the reason is the
    /// host or this window.
    NoDmaBuf,
    /// The engine declined or the copy failed; the inner error names which.
    Engine {
        inner: crate::backend::vulkan::engine::DrawError,
    },
}

#[cfg(feature = "backend-vulkan")]
impl crate::observe::Decline for GpuWritebackDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::NotWritable => "gpuwb_not_writable",
            Self::GeometryMoved { .. } => "gpuwb_geometry_moved",
            Self::WindowUnresolved { .. } => "gpuwb_window_unresolved",
            Self::FormatNeedsConversion { .. } => "gpuwb_format_needs_conversion",
            Self::PitchNotTexels { .. } => "gpuwb_pitch_not_texels",
            Self::OffsetNotTexelAligned { .. } => "gpuwb_offset_not_texel_aligned",
            Self::PageListShort { .. } => "gpuwb_page_list_short",
            Self::PageUnbacked { .. } => "gpuwb_page_unbacked",
            Self::PagesNotOurs => "gpuwb_pages_not_ours",
            Self::NoDmaBuf => "gpuwb_no_dmabuf",
            // The engine's own slug, so a driver that refuses the fd and a
            // resident in the wrong channel order stay as distinguishable here
            // as they are where they were decided.
            Self::Engine { inner } => crate::observe::Decline::slug(inner),
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::NotWritable | Self::PagesNotOurs | Self::NoDmaBuf => Vec::new(),
            Self::GeometryMoved {
                latched_width,
                latched_height,
                frame_width,
                frame_height,
            } => vec![
                ("latched", format!("{latched_width}x{latched_height}")),
                ("frame", format!("{frame_width}x{frame_height}")),
            ],
            Self::WindowUnresolved {
                width,
                height,
                format,
            } => vec![
                ("geom", format!("{width}x{height}")),
                ("fmt", format!("{format:#x}")),
            ],
            Self::FormatNeedsConversion { format } => vec![("fmt", format!("{format:#x}"))],
            Self::PitchNotTexels { bpr } => vec![("bpr", bpr.to_string())],
            Self::OffsetNotTexelAligned { in_page } => vec![("in_page", in_page.to_string())],
            Self::PageListShort { need, have } => {
                vec![("need", need.to_string()), ("have", have.to_string())]
            }
            Self::PageUnbacked { index } => vec![("page", index.to_string())],
            Self::Engine { inner } => crate::observe::Decline::fields(inner),
        }
    }
}

#[cfg(feature = "backend-vulkan")]
crate::observe::decline::decline_display!(GpuWritebackDecline);

/// Which of a mapping's pages a writeback's texels live in, and where inside
/// them the first one is.
///
/// A dma-buf names whole pages and starts at a page boundary; a sample window
/// starts wherever the guest's plane descriptor put it. This is the translation
/// between the two, and getting it wrong lands a frame at the wrong offset in
/// the guest's memory — which is a visibly shifted surface at best and another
/// allocation's bytes at worst.
#[cfg(feature = "backend-vulkan")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GuestWindowPlan {
    /// Indices into `page_entries` of the first and last page the frame touches.
    first_page: usize,
    last_page: usize,
    /// Byte offset of the frame's first texel within page `first_page`, which is
    /// therefore its offset within the dma-buf.
    in_page: u64,
    /// Guest row pitch in texels (`bufferRowLength`).
    row_length_texels: u32,
}

#[cfg(feature = "backend-vulkan")]
impl GuestWindowPlan {
    fn pages(&self) -> usize {
        self.last_page - self.first_page + 1
    }
}

/// Resolve a sample window against a mapping's page list.
///
/// Pure, and separate from its one caller for that reason: every value it
/// produces feeds a `VkBufferImageCopy` whose failure mode is silent — Vulkan
/// will happily write a frame at the wrong offset — and none of the surrounding
/// device state is needed to decide any of them.
#[cfg(feature = "backend-vulkan")]
fn plan_guest_window(
    page_entries: usize,
    page_size: u64,
    base_off: u64,
    span_end: u64,
    bpr: u32,
    width: u32,
) -> Result<GuestWindowPlan, GpuWritebackDecline> {
    // `bufferRowLength` is in texels, so a pitch that is not a whole number of
    // them has no spelling. Checked rather than assumed: the value comes from
    // the guest's own device descriptor.
    //
    // The second half is a Vulkan validity rule rather than an arithmetic one:
    // `bufferRowLength` must be zero or at least `imageExtent.width`, so a pitch
    // narrower than the frame is an invalid copy and not a tight one. It cannot
    // happen for a well-formed plane, which is exactly why nothing would notice
    // if it did.
    if !bpr.is_multiple_of(RGBA8_BPP) || bpr / RGBA8_BPP < width {
        return Err(GpuWritebackDecline::PitchNotTexels { bpr });
    }
    if span_end <= base_off || page_size == 0 {
        return Err(GpuWritebackDecline::NotWritable);
    }
    let first_page = (base_off / page_size) as usize;
    let last_page = ((span_end - 1) / page_size) as usize;
    if page_entries <= last_page {
        return Err(GpuWritebackDecline::PageListShort {
            need: last_page + 1,
            have: page_entries,
        });
    }
    // The dma-buf starts at a page boundary, so the frame's first texel sits
    // this far into it. Whole texels only, which is what `bufferOffset` requires
    // and what a guest pitch in texels already implies for every row but the
    // first.
    let in_page = base_off % page_size;
    if !in_page.is_multiple_of(u64::from(RGBA8_BPP)) {
        return Err(GpuWritebackDecline::OffsetNotTexelAligned { in_page });
    }
    Ok(GuestWindowPlan {
        first_page,
        last_page,
        in_page,
        row_length_texels: bpr / RGBA8_BPP,
    })
}

/// Copy a resident target straight into the guest's pages, with the frame never
/// existing on the host.
///
/// # What this is for
///
/// The copying rail this replaces moves the frame twice after the GPU has
/// already written it once: the resident is read into a `HOST_VISIBLE` staging
/// buffer, and the CPU then scatters that buffer into guest RAM row by row.
/// `readback_split` prices the pair at 0.83 ms of staging map plus 2.68 ms of
/// guest-page write inside a 6.9 ms flush, and the flush rail is 69% of the
/// drain worker's second. Making the guest's own pages the copy's destination
/// leaves only the copy that always had to happen.
///
/// # What it still owes
///
/// Everything [`write_bgra8_inner`] does *besides* moving bytes, because those
/// obligations are about the guest's pages having changed and not about who
/// changed them. In particular the guest-write witness
/// ([`DeviceState::note_host_wrote_mapping`]) and the page footprint: a rail that
/// lands frames without recording that it did makes
/// [`crate::runtime::gather_witness`] attribute its own writes to the guest, and
/// the type-11 resident rung above it then refuses residents and gathers whole
/// surfaces per bind. That failure is measured and it costs more than this rail
/// saves — see the ledger in [`crate::runtime::storage_flush`].
///
/// # Errors
///
/// Every decline is a routing answer: the caller still owes the frame and takes
/// the copying rail. `Ok(())` means the pixels are in the guest's pages and the
/// GPU has finished writing them.
#[cfg(feature = "backend-vulkan")]
pub fn write_bgra8_from_resident_gpu<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mapping_id: u32,
    identity: &crate::backend::vulkan::engine::TargetIdentity,
    width: u32,
    height: u32,
) -> Result<u64, GpuWritebackDecline> {
    if !scanout_extent_ok(width, height) {
        return Err(GpuWritebackDecline::NotWritable);
    }
    let Some(m) = state.mappings.get(&mapping_id) else {
        return Err(GpuWritebackDecline::NotWritable);
    };
    if !m.mapped || m.page_entries.is_empty() {
        return Err(GpuWritebackDecline::NotWritable);
    }
    let (mw, mh, format) = mapping_write_geometry(m, width, height);
    if mw != width || mh != height {
        return Err(GpuWritebackDecline::GeometryMoved {
            latched_width: mw,
            latched_height: mh,
            frame_width: width,
            frame_height: height,
        });
    }
    // A buffer→image copy moves bytes and converts nothing, so the mapping's
    // format has to be the one the resident already holds. `into_bgra8` on the
    // copying rail is where a semantic-RGBA resident is exchanged, and the
    // engine refuses one of those on its own account too.
    if format != MTL_FORMAT_BGRA8_UNORM && format != pixel_format::MTL_FORMAT_BGRA8_UNORM_SRGB {
        return Err(GpuWritebackDecline::FormatNeedsConversion { format });
    }
    let Some((base_off, bpr, span_end)) = type11_sample_window(m, mw, mh, format) else {
        return Err(GpuWritebackDecline::WindowUnresolved {
            width: mw,
            height: mh,
            format,
        });
    };
    // Deferred-writeback flush-on-access, before the vouch and before the page
    // list is read: this can invalidate the mapping — that is exactly what its
    // own drift check does — and every value taken after it is taken against
    // whatever it left behind.
    crate::runtime::storage_flush::flush_intersecting(state, host, mapping_id, base_off, span_end);
    // Nothing below can land a frame on a host whose GPU cannot import guest
    // pages, so the walks below are skipped rather than run and discarded. Asked
    // *after* `flush_intersecting` and not before it: that call is a side effect
    // this rail owes whether or not it goes on to write anything, and the
    // copying arm that takes over from this decline expects the state it leaves.
    //
    // Not a second gate — `guest_dmabuf::dmabuf_for` still decides, and would
    // decline these same pages a few hundred microseconds later. This only
    // declines sooner, and on the pathways where the answer never changes that
    // is a page-table walk per flush for the life of the process.
    if !crate::runtime::guest_dmabuf::export_available() {
        return Err(GpuWritebackDecline::NoDmaBuf);
    }
    // Timed on its own because it is the largest `O(pages)` step left and its
    // fix is not the other one's. `vouch_for_write` re-walks every page of the
    // mapping through the guest's page table — the check that licenses writing
    // to them at all — and until the host copies were removed that cost was
    // hidden inside a millisecond of memcpy.
    use crate::runtime::drain::{note_readback_phase, ReadbackPhase};
    let vouch_started = std::time::Instant::now();
    let vouched = vouch_for_write(state, host, mapping_id, "gpu_writeback");
    note_readback_phase(
        ReadbackPhase::Vouch,
        vouch_started.elapsed().as_micros() as u64,
    );
    if vouched.is_none() {
        return Err(GpuWritebackDecline::PagesNotOurs);
    }
    let resolve_started = std::time::Instant::now();
    let page_size = state.page_size();
    let page_shift = state.page_shift;
    let Some(m) = state.mappings.get(&mapping_id) else {
        return Err(GpuWritebackDecline::NotWritable);
    };
    let plan = plan_guest_window(m.page_entries.len(), page_size, base_off, span_end, bpr, mw)?;
    let mut gpas = Vec::with_capacity(plan.pages());
    for (i, &entry) in m.page_entries[plan.first_page..=plan.last_page]
        .iter()
        .enumerate()
    {
        let Some(gpa) = crate::contract::iosurface_pages::entry_gpa_shift(entry, page_shift) else {
            return Err(GpuWritebackDecline::PageUnbacked {
                index: plan.first_page + i,
            });
        };
        gpas.push(gpa);
    }
    let Some(window) = crate::runtime::guest_dmabuf::dmabuf_for(host, &gpas, page_size as u32)
    else {
        return Err(GpuWritebackDecline::NoDmaBuf);
    };
    let target = crate::backend::vulkan::engine::GuestPageTarget {
        window,
        mapped_bytes: gpas.len() as u64 * page_size,
        offset: plan.in_page,
        row_length_texels: plan.row_length_texels,
        width: mw,
        height: mh,
    };
    // Both witnesses before the copy rather than after it, matching
    // `contig_for_write`: a refused write costs a spurious bump, which makes a
    // reader re-read bytes that did not change, while the opposite error hands
    // out a stale copy as fresh.
    //
    // Marked over `[base_off, span_end)` — the extent the copy names — rather
    // than from zero, because unlike the contig view this rail knows its own
    // rows and has no pointer whose coverage it is restating.
    mapper::note_mapping_write_footprint(state, mapping_id, base_off, span_end - base_off);
    state.note_host_wrote_mapping(mapping_id);
    note_readback_phase(
        ReadbackPhase::Resolve,
        resolve_started.elapsed().as_micros() as u64,
    );
    crate::backend::vulkan::engine::copy_target_to_guest_pages(identity, &target)
        .map_err(|inner| GpuWritebackDecline::Engine { inner })?;
    state.invalidate_storage_residency_window(mapping_id, base_off, span_end);
    let _ = state.mark_mapping_written(mapping_id);
    // Nothing here leaves a host copy of the frame, so the surface cache must
    // not go on naming one: its entry, if any, is a previous flush's bytes and
    // the guest's pages are now the only place this frame exists. Same reason
    // `write_bgra8_uncached` invalidates rather than publishes.
    crate::runtime::surface_cache::forget(state, mapping_id);
    Ok(span_end - base_off)
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
    let Some((base_off, _, span_end)) = type11_sample_window(m, mw, mh, format) else {
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
    if !scanout_extent_ok(width, height) {
        return refuse(mapping_id, SurfaceWriteRefusal::Geometry { width, height });
    }
    if src_stride < width.saturating_mul(RGBA8_BPP) {
        return refuse(
            mapping_id,
            SurfaceWriteRefusal::SourceStride { src_stride, width },
        );
    }
    let Some(m) = state.mappings.get(&mapping_id) else {
        return refuse(mapping_id, SurfaceWriteRefusal::MappingAbsent);
    };
    if !m.mapped || m.page_entries.is_empty() {
        return refuse(mapping_id, SurfaceWriteRefusal::MappingNotResident);
    }
    let (mw, mh, format) = mapping_write_geometry(m, width, height);
    if mw != width || mh != height {
        return refuse(
            mapping_id,
            SurfaceWriteRefusal::GeometryMoved {
                latched_width: mw,
                latched_height: mh,
                frame_width: width,
                frame_height: height,
            },
        );
    }
    let Some((base_off, bpr_u32, span_end)) = type11_sample_window(m, mw, mh, format) else {
        return refuse(
            mapping_id,
            SurfaceWriteRefusal::WindowUnresolved {
                width: mw,
                height: mh,
                format,
            },
        );
    };
    // Deferred-writeback flush-on-access: land pending resident content in
    // these pages before touching them.
    crate::runtime::storage_flush::flush_intersecting(state, host, mapping_id, base_off, span_end);
    // Taken after the flush, because the flush can invalidate this mapping, and
    // once for the whole frame, because the loop below writes a row at a time.
    let Some(vouched) = vouch_for_write(state, host, mapping_id, "bgra8") else {
        return refuse(mapping_id, SurfaceWriteRefusal::PagesNotOurs);
    };
    let bpr = bpr_u32 as usize;
    let Some(tight) = pixel_format::tight_row_bytes(mw, format) else {
        return refuse(mapping_id, SurfaceWriteRefusal::FormatRowLength { format });
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
        note_surface_write_path, note_surface_write_phase, SurfaceWritePhase,
    };
    let frame_bytes = (mh as u64).saturating_mul(tight as u64);

    // Fast path: one packed view, poke rows in place.
    if let Some((ptr, _)) = contig_for_write(state, host, mapping_id, span_end, &vouched) {
        note_surface_write_path(true, frame_bytes);
        let land_started = std::time::Instant::now();
        // SAFETY: contig covers span_end; revalidated in ensure_contig_view.
        let base = unsafe { (ptr as *mut u8).add(base_off as usize) };
        for y in 0..mh {
            let src_off = (y as usize) * (src_stride as usize);
            let src_row_len = (mw as usize) * (RGBA8_BPP as usize);
            if src_off + src_row_len > src.len() {
                return refuse(
                    mapping_id,
                    SurfaceWriteRefusal::SourceShort {
                        need: src_off + src_row_len,
                        have: src.len(),
                        row: y,
                    },
                );
            }
            let src_row = &src[src_off..src_off + src_row_len];
            let row_bytes: &[u8] = if direct_rows {
                &src_row[..tight]
            } else {
                if let Some(ref mut rgba_row) = rgba {
                    if !convert_row_to_rgba8(MTL_FORMAT_BGRA8_UNORM, src_row, mw, rgba_row)
                        || !convert_rgba8_to_row(format, rgba_row, mw, &mut row)
                    {
                        return refuse(
                            mapping_id,
                            SurfaceWriteRefusal::RowConvert { format, row: y },
                        );
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
            return refuse(
                mapping_id,
                SurfaceWriteRefusal::FrameExtent { bpr, height: mh },
            );
        };
        // The staged buffer is `src` itself whenever the layout it would be
        // built into is the layout `src` already has: no conversion (the rows go
        // through untouched) and a source pitch equal to the mapping's row pitch
        // (so row `y` is already at `y * bpr`). Under both, byte `i` of the
        // staged frame is byte `i` of `src` for every `i < frame_len`, and
        // building it copies 8 MB to produce a slice we are holding.
        //
        // What `src` has in the gaps between rows does not enter into it: the
        // store below names the texel runs only, so those bytes are never read
        // out of this buffer whichever way it was built.
        let staged: std::borrow::Cow<'_, [u8]> =
            if direct_rows && bpr == src_stride as usize && src.len() >= frame_len {
                std::borrow::Cow::Borrowed(&src[..frame_len])
            } else {
                let mut frame = vec![0u8; frame_len];
                for y in 0..mh {
                    let src_off = (y as usize) * (src_stride as usize);
                    let src_row_len = (mw as usize) * (RGBA8_BPP as usize);
                    if src_off + src_row_len > src.len() {
                        return refuse(
                            mapping_id,
                            SurfaceWriteRefusal::SourceShort {
                                need: src_off + src_row_len,
                                have: src.len(),
                                row: y,
                            },
                        );
                    }
                    let src_row = &src[src_off..src_off + src_row_len];
                    let row_bytes: &[u8] = if direct_rows {
                        &src_row[..tight]
                    } else {
                        if let Some(ref mut rgba_row) = rgba {
                            if !convert_row_to_rgba8(MTL_FORMAT_BGRA8_UNORM, src_row, mw, rgba_row)
                                || !convert_rgba8_to_row(format, rgba_row, mw, &mut row)
                            {
                                return refuse(
                                    mapping_id,
                                    SurfaceWriteRefusal::RowConvert { format, row: y },
                                );
                            }
                        } else {
                            let n = src_row_len.min(row.len());
                            row[..n].copy_from_slice(&src_row[..n]);
                        }
                        &row
                    };
                    let dst_off = (y as usize).saturating_mul(bpr);
                    if dst_off + tight > frame.len() {
                        return refuse(
                            mapping_id,
                            SurfaceWriteRefusal::StagedShort {
                                need: dst_off + tight,
                                have: frame.len(),
                                row: y,
                            },
                        );
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
        // One call for the whole frame, carrying the runs it should store,
        // rather than one call per surviving run: every call re-runs
        // `flush_intersecting` over the deferred windows and re-resolves the
        // mapping's page list, both `O(pages)`, so the per-run shape pays that
        // twice-over walk for each hole the skip list cuts. The selection
        // travels into the walk instead, so the resolution happens once and
        // each imported page run moves only the parts of itself the runs name.
        //
        // The runs are the `tight` bytes at the head of each of `mh` rows, not
        // the frame's whole extent. A row pitch wider than the packed row leaves
        // padding between rows, and that padding is not a texel this call was
        // given: the contig arm above writes row by row and never touches it,
        // so storing the staged frame entire would zero it here and leave it
        // alone there — the same call landing different guest memory depending
        // only on whether the guest's pages happened to be adjacent.
        //
        // Those bytes do belong to this plane (`sample_window_from_device_plane`
        // requires the plane's own `plane_size` to cover `bpr * (mh - 1) +
        // tight`), so this is not an overrun into a neighbouring allocation. It
        // is content the guest put there and the device was never asked to
        // replace.
        let mut runs: Vec<(u64, u64)> = Vec::new();
        for y in 0..mh {
            let row_lo = base_off.saturating_add((y as u64).saturating_mul(bpr as u64));
            for (lo, hi) in unskipped(row_lo, row_lo.saturating_add(tight as u64), skip) {
                match runs.last_mut() {
                    // A packed pitch makes consecutive rows adjacent; coalescing
                    // keeps that frame the single run it was before the split,
                    // which is the shape the hot 8 MB composite surface takes.
                    Some(last) if last.1 == lo => last.1 = hi,
                    _ => runs.push((lo, hi)),
                }
            }
        }
        if !mapper::write_mapping_bytes_only(
            state,
            host,
            mapping_id,
            base_off,
            frame,
            Some(&runs),
            &vouched,
        ) {
            return refuse(
                mapping_id,
                SurfaceWriteRefusal::MapperWrite {
                    lo: base_off,
                    len: frame.len(),
                },
            );
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
        // `storage_flush::land::flush_render_one`.
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
            Some(owner) => crate::runtime::surface_cache::store_shared(
                state,
                mapping_id,
                mw,
                mh,
                owner.clone(),
            ),
            None => crate::runtime::surface_cache::store_rows(
                state, mapping_id, mw, mh, src, src_stride,
            ),
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
    if !scanout_extent_ok(width, height) {
        return refuse(mapping_id, SurfaceWriteRefusal::Geometry { width, height });
    }
    let rgba_stride = width.saturating_mul(RGBA8_BPP);
    let need = (height as usize).saturating_mul(rgba_stride as usize);
    if rgba.len() < need {
        return refuse(
            mapping_id,
            SurfaceWriteRefusal::SourceShort {
                need,
                have: rgba.len(),
                row: 0,
            },
        );
    }
    if let Some(seed) = seed_rgba {
        if seed.len() < need {
            return refuse(
                mapping_id,
                SurfaceWriteRefusal::SeedShort {
                    need,
                    have: seed.len(),
                },
            );
        }
    }
    let Some(m) = state.mappings.get(&mapping_id) else {
        return refuse(mapping_id, SurfaceWriteRefusal::MappingAbsent);
    };
    if !m.mapped || m.page_entries.is_empty() {
        return refuse(mapping_id, SurfaceWriteRefusal::MappingNotResident);
    }
    let (mw, mh, format) = mapping_write_geometry(m, width, height);
    if mw != width || mh != height {
        return refuse(
            mapping_id,
            SurfaceWriteRefusal::GeometryMoved {
                latched_width: mw,
                latched_height: mh,
                frame_width: width,
                frame_height: height,
            },
        );
    }
    let Some((base_off, bpr_u32, span_end)) = type11_sample_window(m, mw, mh, format) else {
        return refuse(
            mapping_id,
            SurfaceWriteRefusal::WindowUnresolved {
                width: mw,
                height: mh,
                format,
            },
        );
    };
    let bpr = bpr_u32 as u64;
    let Some(tight) = pixel_format::tight_row_bytes(mw, format) else {
        return refuse(mapping_id, SurfaceWriteRefusal::FormatRowLength { format });
    };
    let bpr_usize = bpr as usize;
    let tight = tight as usize;
    let mut native = vec![0u8; tight];
    let mut seed_native = vec![0u8; tight];
    // Deferred-writeback flush-on-access, as at every other read/write entry in
    // this file — the module doc for `storage_flush` names them all as choke
    // points. It has to be here rather than on one arm: the fragmented arm ends
    // in `mapper::write_mapping_bytes`, which flushes, while the
    // `contig_for_write` arm is a raw `copy_nonoverlapping` into the mapped span
    // and flushes nothing. Whether an armed window landed before or after this
    // write therefore depended on whether the guest's pages happened to be
    // contiguous, and landing after puts an older frame on top of this one —
    // which `mapper::write_mapping_bytes_only` states as its own reason for
    // flushing here.
    crate::runtime::storage_flush::flush_intersecting(state, host, mapping_id, base_off, span_end);
    // One proof for the whole image: the changed-span loop below writes each
    // differing row separately, and the walk is a translation per page.
    let Some(vouched) = vouch_for_write(state, host, mapping_id, "rgba8_changed") else {
        return refuse(mapping_id, SurfaceWriteRefusal::PagesNotOurs);
    };
    let contig = contig_for_write(state, host, mapping_id, span_end, &vouched);
    // SAFETY: when Some, contig covers span_end.
    let base = contig.map(|(ptr, _)| unsafe { (ptr as *mut u8).add(base_off as usize) });
    for y in 0..mh as usize {
        let src_off = y * rgba_stride as usize;
        let src_row = &rgba[src_off..src_off + rgba_stride as usize];
        if !rgba8_row_to_native(format, src_row, mw, &mut native) {
            return refuse(
                mapping_id,
                SurfaceWriteRefusal::RowConvert {
                    format,
                    row: y as u32,
                },
            );
        }
        let seed_row = if let Some(seed) = seed_rgba {
            let s = &seed[src_off..src_off + rgba_stride as usize];
            if !rgba8_row_to_native(format, s, mw, &mut seed_native) {
                return refuse(
                    mapping_id,
                    SurfaceWriteRefusal::SeedRowConvert {
                        format,
                        row: y as u32,
                    },
                );
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
                    return refuse(
                        mapping_id,
                        SurfaceWriteRefusal::MapperWrite {
                            lo: row_moff.saturating_add(start as u64),
                            len: x - start,
                        },
                    );
                }
            }
        } else if !mapper::write_mapping_bytes(state, host, mapping_id, row_moff, &native, &vouched)
        {
            return refuse(
                mapping_id,
                SurfaceWriteRefusal::MapperWrite {
                    lo: row_moff,
                    len: native.len(),
                },
            );
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
    if !scanout_extent_ok(width, height) {
        return refuse(mapping_id, SurfaceWriteRefusal::Geometry { width, height });
    }
    if row_bytes == 0 || src_stride < row_bytes {
        return refuse(
            mapping_id,
            SurfaceWriteRefusal::SourceStride { src_stride, width },
        );
    }
    let need = (height as u64).saturating_mul(src_stride as u64) as usize;
    if src.len() < need {
        return refuse(
            mapping_id,
            SurfaceWriteRefusal::SourceShort {
                need,
                have: src.len(),
                row: 0,
            },
        );
    }
    // Deferred-writeback flush-on-access (coarse: whole mapping — this entry
    // resolves its window only later and is off the hot compute path).
    crate::runtime::storage_flush::flush_intersecting(state, host, mapping_id, 0, u64::MAX);
    let Some(m) = state.mappings.get(&mapping_id) else {
        return refuse(mapping_id, SurfaceWriteRefusal::MappingAbsent);
    };
    if !m.mapped || m.page_entries.is_empty() {
        return refuse(mapping_id, SurfaceWriteRefusal::MappingNotResident);
    }
    if m.has_geom && (m.width != width || m.height != height) {
        return refuse(
            mapping_id,
            SurfaceWriteRefusal::GeometryMoved {
                latched_width: m.width,
                latched_height: m.height,
                frame_width: width,
                frame_height: height,
            },
        );
    }
    let span_end = (row_bytes as u64).saturating_mul(height as u64);
    let rb = row_bytes as usize;
    let Some(vouched) = vouch_for_write(state, host, mapping_id, "raw_rows") else {
        return refuse(mapping_id, SurfaceWriteRefusal::PagesNotOurs);
    };
    if let Some((ptr, _)) = contig_for_write(state, host, mapping_id, span_end, &vouched) {
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
                return refuse(
                    mapping_id,
                    SurfaceWriteRefusal::MapperWrite { lo: moff, len: rb },
                );
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
    if !scanout_extent_ok(width, height) || row_bytes == 0 || dst_stride < row_bytes {
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
/// Where a mapped type-11 surface's texels sit, and how wide one is.
///
/// `mapping_geom_window` used to return this as `Option<(u64, u32, u64, u32)>`,
/// a shape whose meaning existed only in the destructuring patterns of the two
/// callers that unpacked it — and which they then splatted straight into four
/// parameters of the `_at` functions, split around the rectangle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SurfaceWindow {
    /// Byte offset of the plane's first texel within the mapping.
    pub base_off: u64,
    /// Bytes per row of the surface, which is not `width * bpp`.
    pub bpr: u32,
    /// One past the last byte the window may touch.
    pub span_end: u64,
    /// Bytes per texel.
    pub bpp: u32,
}

/// A texel rectangle within a surface.
///
/// The four fields are `u32` and were adjacent in five signatures here, so
/// every permutation of them compiled and no call site could object. One test
/// call read `..., 0, 0, 4, 1, 1, ...` — five bare numbers spanning the origin,
/// the extent and the bytes per texel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rect {
    pub origin_x: u32,
    pub origin_y: u32,
    pub width: u32,
    pub height: u32,
}

fn mapping_geom_window(state: &DeviceState, mapping_id: u32, rect: Rect) -> Option<SurfaceWindow> {
    let Rect {
        origin_x,
        origin_y,
        width,
        height,
    } = rect;
    let m = state.mappings.get(&mapping_id)?;
    if !m.has_geom {
        return None;
    }
    let format = if m.format != 0 {
        m.format
    } else {
        MTL_FORMAT_BGRA8_UNORM
    };
    let (base_off, bpr, span_end) = type11_sample_window(m, m.width, m.height, format)?;
    let bpp = pixel_format::bytes_per_pixel(format)?;
    if origin_x.saturating_add(width) > m.width || origin_y.saturating_add(height) > m.height {
        return None;
    }
    Some(SurfaceWindow {
        base_off,
        bpr,
        span_end,
        bpp,
    })
}

/// Read a rectangular texel region from a mapped type-11 IOSurface.
/// Contig HostOps view when possible; else multi-import.
#[cfg(test)]
pub fn read_rect_raw<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mapping_id: u32,
    rect: Rect,
    dst: &mut [u8],
    dst_stride: u32,
) -> bool {
    let Some(window) = mapping_geom_window(state, mapping_id, rect) else {
        return false;
    };
    read_rect_raw_at(state, host, mapping_id, window, rect, dst, dst_stride)
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
    window: SurfaceWindow,
    rect: Rect,
    dst: &mut [u8],
    dst_stride: u32,
) -> bool {
    let SurfaceWindow {
        base_off,
        bpr: surface_bpr,
        span_end,
        bpp,
    } = window;
    let Rect {
        origin_x,
        origin_y,
        width,
        height,
    } = rect;
    if !scanout_extent_ok(width, height) || bpp == 0 {
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
    // The rect must end inside the sample window, and that is asked once for both
    // arms below rather than inside either. A correctly-sized read satisfies it
    // exactly (a dense tight read has `read_end == span_end`), so it drops only a
    // genuine overrun.
    //
    // It used to sit inside the contig arm, on the reasoning that the fragmented
    // arm was bounded anyway — which is true, but only by its own slice bounds:
    // that arm reads the window and then indexes rows into it, so an overrunning
    // rect came back as a bare `false` from a `get` that returned `None`. Both
    // callers do name that (`rd_row_t11_io`, `Type5ViewDecline::Read`), so it was
    // never a silent loss, but neither can say the rect left the window, and the
    // fragmented arm is the one a driven x86 boot actually takes. One check above
    // the split gives both arms the same refusal and the same line.
    let read_end = rect_extent_end(base_off, origin_y, height, bpr, x_off, rb);
    if read_end > span_end {
        crate::observe::fail(format!(
            "mapping_read fail reason=read_overrun mid={mapping_id} base_off={base_off} origin_y={origin_y} height={height} bpr={surface_bpr} x_off={x_off} rb={rb} read_end={read_end} span_end={span_end}"
        ));
        return false;
    }
    if let Some((ptr, _)) = contig_for_span(state, host, mapping_id, span_end) {
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
    rect: Rect,
    src: &[u8],
    src_stride: u32,
) -> bool {
    let Some(window) = mapping_geom_window(state, mapping_id, rect) else {
        return false;
    };
    write_rect_raw_at(state, host, mapping_id, window, rect, src, src_stride)
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
    window: SurfaceWindow,
    rect: Rect,
    src: &[u8],
    src_stride: u32,
) -> bool {
    let SurfaceWindow {
        base_off,
        bpr: surface_bpr,
        span_end,
        bpp,
    } = window;
    let Rect {
        origin_x,
        origin_y,
        width,
        height,
    } = rect;
    write_rect_raw_at_impl(
        state,
        host,
        mapping_id,
        SurfaceWindow {
            base_off,
            bpr: surface_bpr,
            span_end,
            bpp,
        },
        Rect {
            origin_x,
            origin_y,
            width,
            height,
        },
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
        SurfaceWindow {
            base_off,
            bpr: surface_bpr,
            span_end,
            bpp,
        },
        Rect {
            origin_x: 0,
            origin_y: 0,
            width,
            height,
        },
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
    window: SurfaceWindow,
    rect: Rect,
    src: &[u8],
    src_stride: u32,
    full_plane: bool,
) -> bool {
    let SurfaceWindow {
        base_off,
        bpr: surface_bpr,
        span_end,
        bpp,
    } = window;
    let Rect {
        origin_x,
        origin_y,
        width,
        height,
    } = rect;
    if !scanout_extent_ok(width, height) || bpp == 0 {
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
    // The destination bound, before the branch, because all three arms below
    // write guest memory and only two of them used to check it. The per-row
    // fragmented arm went through `mapper::write_mapping_bytes`, which bounds
    // against the *whole mapping's* page span and not this plane's window, so an
    // over-tall rect landed in whatever follows the window — on a multi-plane
    // IOSurface that is the next plane's pixels — and said nothing.
    //
    // `rect_extent_end` is the shared expression for exactly this reason: its own
    // doc records that the read and write sides disagreed while each computed it
    // separately. A third caller computing its own variant is how that happens
    // again, so the bound is taken once here and the arms carry none of their own.
    // A correctly-sized writeback satisfies it exactly (a dense tight write gives
    // `write_end == span_end`), so this drops ONLY a genuine overrun — named,
    // never silent.
    let write_end = rect_extent_end(base_off, origin_y, height, bpr, x_off, rb);
    if write_end > span_end {
        crate::observe::fail(format!(
            "mapping_write fail reason=writeback_overrun mid={mapping_id} base_off={base_off} origin_y={origin_y} height={height} bpr={surface_bpr} x_off={x_off} rb={rb} write_end={write_end} span_end={span_end}"
        ));
        return false;
    }
    // Deferred-writeback flush-on-access, for the same reason and on the same
    // split as `write_rgba8_image_changed`: the fragmented arms below flush
    // through `mapper::write_mapping_bytes` and the contiguous one does not, so
    // without this an armed window could land on top of this rect on packed
    // surfaces only. Safe to call from inside a flush — the storage rail reaches
    // this function through `write_full_rect_raw_at`, and `flush_intersecting`
    // removes intersecting windows up front so the nested call finds nothing.
    crate::runtime::storage_flush::flush_intersecting(state, host, mapping_id, base_off, span_end);
    let Some(vouched) = vouch_for_write(state, host, mapping_id, "rect_raw") else {
        return false;
    };
    if let Some((ptr, _)) = contig_for_write(state, host, mapping_id, span_end, &vouched) {
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
        // class).
        //
        // The store names each row's `rb` texel bytes rather than the frame's
        // whole extent, for the reason `write_bgra8_inner` states at its own
        // staging branch: a row pitch wider than the packed row leaves padding
        // between rows, the contig arm twenty lines above writes row by row and
        // never touches it, so storing the staged frame entire would zero it
        // here and leave it alone there — the same call landing different guest
        // memory depending only on whether the guest's pages happened to be
        // adjacent. Those bytes belong to this plane, so it was never an overrun
        // into a neighbour; it is content the guest put there and this call was
        // not asked to replace.
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
        // No `frame_end > span_end` here: this arm used to take its own variant
        // of the bound, computed without `x_off` and so looser than the one
        // `rect_extent_end` gives above. The overflow check on `frame_len` stays
        // because it guards the allocation on the next lines.
        if base_off.checked_add(frame_len as u64).is_none() {
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
        // Built alongside the fill so the two cannot describe different rows.
        // Adjacent runs coalesce, so a packed pitch collapses to the single run
        // it was before the split and moves exactly the same bytes.
        let mut runs: Vec<(u64, u64)> = Vec::new();
        for y in 0..height as usize {
            let src_off = y * src_stride as usize;
            let dst_off = ((origin_y as usize) + y)
                .saturating_mul(bpr)
                .saturating_add(x_off as usize);
            if dst_off + rb > frame.len() {
                return false;
            }
            frame[dst_off..dst_off + rb].copy_from_slice(&src[src_off..src_off + rb]);
            let lo = base_off.saturating_add(dst_off as u64);
            let hi = lo.saturating_add(rb as u64);
            match runs.last_mut() {
                Some(last) if last.1 == lo => last.1 = hi,
                _ => runs.push((lo, hi)),
            }
        }
        if !mapper::write_mapping_bytes_only(
            state,
            host,
            mapping_id,
            base_off,
            &frame,
            Some(&runs),
            &vouched,
        ) {
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

    /// `mapping_geom_window` puts each measurement in the field of its own name.
    ///
    /// `SurfaceWindow`'s four fields are two `u64`s and two `u32`s, so
    /// `base_off`/`span_end` can cross silently and so can `bpr`/`bpp`. The
    /// mapping below is chosen so all four read differently, which is what
    /// makes a crossing observable at all.
    ///
    /// The row pitch is asserted by its relationships, not as a number: a
    /// 4-wide BGRA8 surface reports `bpr = 128` against a tight row of 16,
    /// because `type11_sample_window` aligns the pitch up. Hard-coding either
    /// value would make this a test of that alignment rather than of which
    /// field holds what.
    #[test]
    fn the_surface_window_names_which_measurement_is_which() {
        use crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let mid = 30u32;
        state.map_surface(mid);
        {
            let m = state.mappings.get_mut(&mid).unwrap();
            m.mapped = true;
            m.mapping_internal = 1;
            m.map_generation = 1;
            m.has_geom = true;
            m.width = 4;
            m.height = 4;
            m.format = MTL_FORMAT_BGRA8_UNORM;
        }
        let rect = Rect {
            origin_x: 0,
            origin_y: 0,
            width: 4,
            height: 4,
        };
        let w = mapping_geom_window(&state, mid, rect).expect("a geometry window");
        assert_eq!(w.bpp, 4, "BGRA8 is four bytes per texel");
        assert!(
            w.bpr >= 4 * w.bpp,
            "a row holds at least the four texels: bpr={} bpp={}",
            w.bpr,
            w.bpp
        );
        assert_ne!(
            w.bpr, w.bpp,
            "the two must differ here or this test could not see them swapped"
        );
        assert_eq!(
            w.span_end - w.base_off,
            u64::from(w.bpr) * 4,
            "the span reaches exactly the four rows the rectangle asked for"
        );
        // A rectangle past the declared extent has no window at all.
        assert!(mapping_geom_window(
            &state,
            mid,
            Rect {
                origin_x: 1,
                ..rect
            }
        )
        .is_none());
    }

    /// A tight full-page-aligned surface names exactly the pages its bytes
    /// occupy, and no more.
    ///
    /// The last page is the one holding the last *texel*, not the one holding
    /// `bpr * height`. A plan that rounded up to the row pitch would pin — and
    /// hand the GPU write access to — a page past the surface on every flush of
    /// a padded layout, and the guest owns whatever is in it.
    #[cfg(feature = "backend-vulkan")]
    #[test]
    fn a_tight_window_names_the_pages_its_texels_occupy() {
        // 1920x1080 BGRA8, tight, starting at offset 0 of a 4 KiB-page guest.
        let (page, bpr) = (4096u64, 1920 * 4u32);
        let span = u64::from(bpr) * 1080;
        let plan =
            plan_guest_window(usize::MAX, page, 0, span, bpr, 1920).expect("a tight window plans");
        assert_eq!(plan.first_page, 0);
        assert_eq!(plan.last_page, ((span - 1) / page) as usize);
        assert_eq!(plan.in_page, 0);
        assert_eq!(plan.row_length_texels, 1920);
        // Exactly the pages the bytes are in: 1920*4*1080 is a whole number of
        // 4 KiB pages, so the last texel is the last byte of the last one.
        assert_eq!(plan.pages() as u64, span / page);
    }

    /// A window starting part-way into a page reports that offset, and the page
    /// it starts in is the first the dma-buf names.
    ///
    /// This is the whole reason the plan exists. The fd starts at a page
    /// boundary and the sample window does not, so a copy that took the window's
    /// mapping offset as its `bufferOffset` would land the frame `first_page *
    /// page_size` bytes early — off the front of the export entirely for any
    /// surface past the first page.
    #[cfg(feature = "backend-vulkan")]
    #[test]
    fn a_window_starting_inside_a_page_carries_the_offset_and_not_the_mapping_one() {
        let (page, bpr) = (4096u64, 256 * 4u32);
        let base = 3 * page + 512;
        let span = base + u64::from(bpr) * 8;
        let plan = plan_guest_window(usize::MAX, page, base, span, bpr, 256).expect("plans");
        assert_eq!(plan.first_page, 3);
        assert_eq!(plan.in_page, 512);
        // Not the mapping offset: that is the bug this asserts against.
        assert_ne!(plan.in_page, base);
    }

    /// Page shift is explicit, so the same window plans differently on the two
    /// guests. A helper that assumed 4 KiB would name four times too many pages
    /// on arm64 and export three quarters of a surface it was never asked for.
    #[cfg(feature = "backend-vulkan")]
    #[test]
    fn the_same_window_spans_fewer_pages_on_a_sixteen_kilobyte_guest() {
        let bpr = 1024 * 4u32;
        let span = u64::from(bpr) * 64;
        let x86 = plan_guest_window(usize::MAX, 4096, 0, span, bpr, 1024).expect("plans on x86");
        let arm = plan_guest_window(usize::MAX, 16384, 0, span, bpr, 1024).expect("plans on arm64");
        assert_eq!(x86.pages(), arm.pages() * 4);
    }

    /// A padded guest pitch travels as texels, because that is what
    /// `bufferRowLength` is. The inter-row bytes are never named, so the guest's
    /// own content in the padding survives the flush — matching the copying
    /// rail, which writes row by row and skips it too.
    #[cfg(feature = "backend-vulkan")]
    #[test]
    fn a_padded_pitch_becomes_a_row_length_in_texels() {
        let bpr = 2048 * 4u32;
        let plan =
            plan_guest_window(usize::MAX, 4096, 0, u64::from(bpr) * 4, bpr, 1600).expect("plans");
        assert_eq!(plan.row_length_texels, 2048);
    }

    /// Every value a `VkBufferImageCopy` cannot express declines by name rather
    /// than being rounded into one it can.
    ///
    /// `bufferOffset` must be a multiple of the texel block size and
    /// `bufferRowLength` is counted in texels; a copy submitted with either one
    /// wrong is undefined behaviour, not a misplaced frame, so neither may be
    /// silently repaired.
    #[cfg(feature = "backend-vulkan")]
    #[test]
    fn a_geometry_the_copy_cannot_express_declines_by_name() {
        // A row pitch that is not a whole number of texels.
        assert_eq!(
            plan_guest_window(usize::MAX, 4096, 0, 4096, 1023, 1),
            Err(GpuWritebackDecline::PitchNotTexels { bpr: 1023 })
        );
        // A window starting on an odd byte inside its page.
        assert_eq!(
            plan_guest_window(usize::MAX, 4096, 2, 4096, 4, 1),
            Err(GpuWritebackDecline::OffsetNotTexelAligned { in_page: 2 })
        );
        // A page list that stops before the window does. Writing anyway would
        // export whatever the shorter list's tail happens to name.
        assert_eq!(
            plan_guest_window(2, 4096, 0, 3 * 4096, 4, 1),
            Err(GpuWritebackDecline::PageListShort { need: 3, have: 2 })
        );
        // An empty or inverted window has no destination at all.
        assert_eq!(
            plan_guest_window(usize::MAX, 4096, 100, 100, 4, 1),
            Err(GpuWritebackDecline::NotWritable)
        );
        // A pitch narrower than the frame. Vulkan requires `bufferRowLength` to
        // be zero or at least the extent's width, so this is an invalid copy
        // rather than a tight one — and a plan that let it through would submit
        // it, because nothing else in the path re-derives the row length.
        assert_eq!(
            plan_guest_window(usize::MAX, 4096, 0, 4096, 4 * 8, 9),
            Err(GpuWritebackDecline::PitchNotTexels { bpr: 32 })
        );
    }

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
    /// Both write entries drain the mapping's armed deferred windows first, on
    /// **both** storage shapes.
    ///
    /// `storage_flush`'s module doc names "`mapping_write` read/write entries"
    /// as choke points, and five of them took the flush. These two did not, and
    /// the consequence was shape-dependent rather than absent: their fragmented
    /// arm reaches `mapper::write_mapping_bytes`, which flushes, while their
    /// contiguous arm is a raw `copy_nonoverlapping` that does not. So on a
    /// packed surface the window stayed armed and landed *after* the write,
    /// putting an older frame on top of the one just written — the outcome
    /// `mapper::write_mapping_bytes_only` gives as its own reason for flushing.
    ///
    /// Asserting on the armed window rather than on pixels is deliberate: the
    /// window landing late is the mechanism, and a pixel assertion would depend
    /// on which frame the fixture happened to arm.
    #[test]
    fn the_write_entries_drain_armed_windows_on_both_storage_shapes() {
        use crate::model::PAGE_SHIFT_X86;
        const PAGE: u64 = 1 << PAGE_SHIFT_X86;
        const W: u32 = 16;
        const H: u32 = 16;
        const BPR: u32 = W * 4;

        let arm = |state: &mut DeviceState| {
            let key = crate::model::ComputeStorageResidencyKey {
                mapping_id: 5,
                map_generation: state
                    .mappings
                    .get(&5)
                    .map(|m| m.map_generation)
                    .unwrap_or(0),
                surface_offset: 0,
                surface_bpr: BPR,
                span_end: (W * H * 4) as u64,
                width: W,
                height: H,
                pixel_format: MTL_FORMAT_BGRA8_UNORM,
                texture_ref: 0,
            };
            state.compute_deferred_flush.insert(
                key,
                crate::model::DeferredOwner::Render {
                    armed_seq: 0,
                    armed_stamp_seq: 0,
                    source: crate::model::RenderWindowSource::Owned(std::sync::Arc::new(vec![
                        0u8;
                        (W * H * 4) as usize
                    ])),
                },
            );
        };

        for packed in [true, false] {
            for entry in ["rect", "rgba8_changed"] {
                let mut state = DeviceState::new(DeviceId(4), PAGE_SHIFT_X86);
                let mut host = FakeHost::new();
                host.strict_linux_map = !packed;
                let base_pfn = 0x80u32;
                host.map_range((base_pfn as u64) << PAGE_SHIFT_X86, 4 * PAGE as usize, 0);
                let order: Vec<u32> = if packed {
                    (0..4).collect()
                } else {
                    vec![3, 2, 1, 0]
                };
                let entries: Vec<u32> = order
                    .iter()
                    .map(|i| ((base_pfn + i) << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID)
                    .collect();
                state.map_surface(5);
                state.attach_mapping_internal(5, 0);
                let m = state.mappings.get_mut(&5).unwrap();
                m.mapping_internal = 1;
                m.page_entries = entries;
                assert!(state.set_mapping_geom(5, W, H, MTL_FORMAT_BGRA8_UNORM));
                arm(&mut state);
                assert!(!state.compute_deferred_flush.is_empty());

                let src = vec![0x33u8; (W * H * 4) as usize];
                let _ = match entry {
                    "rect" => write_rect_raw_at(
                        &mut state,
                        &mut host,
                        5,
                        SurfaceWindow {
                            base_off: 0,
                            bpr: BPR,
                            span_end: (W * H * 4) as u64,
                            bpp: 4,
                        },
                        Rect {
                            origin_x: 0,
                            origin_y: 0,
                            width: W,
                            height: H,
                        },
                        &src,
                        BPR,
                    ),
                    _ => write_rgba8_image_changed(&mut state, &mut host, 5, &src, None, W, H),
                };

                assert!(
                    state.compute_deferred_flush.is_empty(),
                    "packed={packed} entry={entry}: the write must drain the armed \
                     window, or it lands afterwards and puts an older frame on top"
                );
            }
        }
    }

    /// A rect taller than the window it names is refused on **both** storage
    /// shapes, and writes nothing past the window.
    ///
    /// `write_rect_raw_at_impl` has three arms that all write guest memory, and
    /// the bound used to be on two of them. The per-row fragmented arm reached
    /// `mapper::write_mapping_bytes`, which bounds against the whole mapping's
    /// page span rather than this plane's window, so an over-tall rect landed in
    /// whatever follows the window — on a multi-plane IOSurface, the next plane's
    /// pixels — with no fail line. The packed arm refused the same call.
    ///
    /// So the loop over `packed` is the test: an assertion on one shape alone
    /// passed throughout, which is how the hole survived. `span_end` is set short
    /// of the mapping's real extent on purpose, because that gap is exactly the
    /// region the unbounded arm wrote into and the bounded one did not.
    #[test]
    fn a_rect_taller_than_its_window_is_refused_on_both_storage_shapes() {
        use crate::model::PAGE_SHIFT_X86;
        const PAGE: u64 = 1 << PAGE_SHIFT_X86;
        const W: u32 = 16;
        const BPR: u32 = W * 4;

        for packed in [true, false] {
            let mut state = DeviceState::new(DeviceId(3), PAGE_SHIFT_X86);
            let mut host = FakeHost::new();
            host.strict_linux_map = !packed;
            let base_pfn = 0x60u32;
            host.map_range((base_pfn as u64) << PAGE_SHIFT_X86, 4 * PAGE as usize, 0xee);
            let order: Vec<u32> = if packed {
                (0..4).collect()
            } else {
                vec![3, 2, 1, 0]
            };
            let entries: Vec<u32> = order
                .iter()
                .map(|i| ((base_pfn + i) << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID)
                .collect();
            state.map_surface(5);
            state.attach_mapping_internal(5, 0);
            let m = state.mappings.get_mut(&5).unwrap();
            m.mapping_internal = 1;
            m.page_entries = entries;

            // The window is one page: 64 rows of 64 bytes. The rect asks for 80.
            let span_end = PAGE;
            let rows = (PAGE / BPR as u64) as u32;
            let over = rows + 16;
            let src = vec![0x5au8; (over as usize) * BPR as usize];

            assert!(
                !write_rect_raw_at(
                    &mut state,
                    &mut host,
                    5,
                    SurfaceWindow {
                        base_off: 0,
                        bpr: BPR,
                        span_end,
                        bpp: 4
                    },
                    Rect {
                        origin_x: 0,
                        origin_y: 0,
                        width: W,
                        height: over
                    },
                    &src,
                    BPR
                ),
                "packed={packed}: a rect past the window's last row must be refused"
            );

            // The bytes after the window still hold the fill the mapping was
            // seeded with, on both shapes.
            let mut after = [0u8; 16];
            assert!(mapper::read_mapping_bytes(
                &mut state, &mut host, 5, span_end, &mut after
            ));
            assert_eq!(
                after, [0xeeu8; 16],
                "packed={packed}: the refused rect must not have written past the window"
            );
        }
    }

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
                    &mut state,
                    &mut host,
                    4,
                    &frame,
                    W * 4,
                    W,
                    H,
                    &[]
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

    /// A host write into guest RAM must announce itself, because the hypervisor's
    /// dirty bitmap is defined not to see it.
    ///
    /// The bitmap witnesses guest CPU stores. Everything this device puts into
    /// the same pages is invisible to it, so a reader holding "the guest has not
    /// written since I looked" would keep serving a copy that *we* superseded.
    /// `DeviceState::host_writes` is the only thing that separates the two, and a
    /// writer that forgets to record there is silent in exactly the way that
    /// matters.
    ///
    /// The read half is the other half of the contract: reads share the same
    /// mapping walk, and a read that moved the record would make every reader
    /// re-fetch on account of a reader.
    /// A dropped writeback must say which check dropped it.
    ///
    /// The composite surface is the largest frame this device moves, and losing
    /// one is a wrong picture that then persists. Sixteen refusal sites used to
    /// answer with a bare `false` that the caller rendered as one
    /// `reason=write_refused`, so a reader could tell that the frame had been
    /// dropped and nothing else.
    ///
    /// `GeometryMoved` is the one this test drives because it is the one a
    /// deferred window can reach without anything being broken: the frame is
    /// armed at one rect and the surface is re-published at another — which is
    /// what an appearance change or a wallpaper switch does. Asserting the
    /// *specific* route is the whole point; asserting `!ok` would pass with every
    /// site sharing a slug again.
    #[test]
    fn a_writeback_refused_because_the_geometry_moved_says_so_by_name() {
        use crate::model::PAGE_SHIFT_X86;
        const PAGE: u64 = 1 << PAGE_SHIFT_X86;

        let mut state = DeviceState::new(DeviceId(3), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        let base_pfn = 0x40u32;
        host.map_range((base_pfn as u64) << PAGE_SHIFT_X86, 8 * PAGE as usize, 0x55);
        state.map_surface(4);
        state.attach_mapping_internal(4, 0);
        let m = state.mappings.get_mut(&4).unwrap();
        m.mapping_internal = 1;
        m.page_entries = (0..4)
            .map(|i| ((base_pfn + i) << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID)
            .collect();
        // Latched at 64x64; the frame below is armed at 32x32, as a window armed
        // before a re-publish would be.
        assert!(state.set_mapping_geom(4, 64, 64, MTL_FORMAT_BGRA8_UNORM));

        // Deltas, not absolutes: the route counters are process-global and every
        // other test in this binary shares them, so a sibling reading nonzero
        // says nothing about this write.
        const SIBLINGS: [&str; 3] = [
            "surface_write_mapping_absent",
            "surface_write_pages_not_ours",
            "surface_write_source_short",
        ];
        let before = crate::runtime::drain::store_route_count("surface_write_geometry_moved");
        let before_siblings: Vec<u64> = SIBLINGS
            .iter()
            .map(|r| crate::runtime::drain::store_route_count(r))
            .collect();

        let frame = vec![0xAAu8; 32 * 32 * 4];
        assert!(
            !write_bgra8(&mut state, &mut host, 4, &frame, 32 * 4, 32, 32),
            "a frame whose rect is not the surface's must not land"
        );
        assert_eq!(
            crate::runtime::drain::store_route_count("surface_write_geometry_moved"),
            before + 1,
            "the writeback was dropped without naming the check that dropped it"
        );
        // And no sibling check moved, or the slug is not discriminating.
        for (route, was) in SIBLINGS.iter().zip(before_siblings) {
            assert_eq!(
                crate::runtime::drain::store_route_count(route),
                was,
                "{route} fired for a geometry mismatch"
            );
        }
    }

    #[test]
    fn writing_guest_pages_moves_the_host_write_record_and_reading_them_does_not() {
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

        let before = state.host_writes.epoch();
        let frame = vec![0xAAu8; (W * H * 4) as usize];
        assert!(write_bgra8(&mut state, &mut host, 4, &frame, W * 4, W, H));
        assert_ne!(
            state.host_writes.epoch(),
            before,
            "a write into the guest's pages went unannounced"
        );

        let after_write = state.host_writes.epoch();
        let mut out = vec![0u8; (W * H * 4) as usize];
        assert!(crate::runtime::mapper::read_mapping_bytes(
            &mut state, &mut host, 4, 0, &mut out
        ));
        assert_eq!(
            state.host_writes.epoch(),
            after_write,
            "a read moved the write record, so every reader now invalidates every reader"
        );
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

    /// The fragmented BGRA write path must land the texels and nothing else —
    /// the same contract `write_bgra8_contig_writes_only_inside_the_sample_window`
    /// pins on the pointer arm, asserted the same way so the two arms cannot
    /// drift apart. Padding after the final row would overrun an exact IOSurface
    /// allocation; padding *between* rows is inside the plane but is still
    /// content the guest put there and this call never named.
    /// The staged full-plane rect write must leave inter-row padding alone, on
    /// the same assertion as its two siblings.
    ///
    /// `write_rect_raw_at_impl` has three arms that must land identical guest
    /// memory. The contiguous one pokes each row's texel bytes through a raw
    /// pointer, the per-row fragmented one writes each row's bytes on its own,
    /// and the staged one built a pitch-wide zeroed frame and stored it entire —
    /// so every padding byte between rows was zeroed in the guest's pages. That
    /// is the defect `write_bgra8_inner` was fixed for, in the sibling function,
    /// and this arm's own comment cited the pre-fix behaviour as its
    /// justification.
    ///
    /// Asserted as "every byte outside the texel runs is unchanged", the same
    /// whole-page comparison the BGRA arms are pinned by, so all three cannot
    /// drift apart again. The fixture forces the staged arm two ways: two
    /// non-adjacent pages (no contiguous view) and `full_plane`, and a pitch
    /// wider than the packed row so `full_tight_direct` cannot take it either.
    #[test]
    fn write_full_rect_raw_staged_leaves_inter_row_padding_alone() {
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
        let mid = 14u32;
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
        assert!(
            mapper::ensure_contig_view(&mut state, &mut host, mid).is_none(),
            "two non-adjacent pages must take the staged path this test is about"
        );

        // 2x2 BGRA8: 8 tight bytes per row at a 128-byte pitch, so 120 bytes of
        // guest content sit between the rows.
        let (w, h, bpp) = (2u32, 2u32, 4u32);
        let tight = (w * bpp) as usize;
        let bpr = 128u32;
        let span_end = (bpr as u64) * (h as u64 - 1) + tight as u64;
        let src = [
            0x11u8, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee,
            0xff, 0x10,
        ];
        assert!(write_full_rect_raw_at(
            &mut state,
            &mut host,
            mid,
            0,
            bpr,
            span_end,
            w,
            h,
            bpp,
            &src,
            tight as u32,
        ));

        let mut got = vec![0u8; page as usize];
        assert!(host.read_gpa(gpa0, &mut got).is_ok());
        let mut want = vec![0xCCu8; page as usize];
        want[..tight].copy_from_slice(&src[..tight]);
        want[bpr as usize..bpr as usize + tight].copy_from_slice(&src[tight..]);
        let first_diff = got.iter().zip(want.iter()).position(|(a, b)| a != b);
        assert_eq!(
            first_diff, None,
            "byte {first_diff:?} outside the texel runs was modified"
        );
    }

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
        assert!(
            mapper::ensure_contig_view(&mut state, &mut host, mid).is_none(),
            "two non-adjacent pages must take the staged path this test is about"
        );
        assert!(write_bgra8(&mut state, &mut host, mid, &src, 8, 2, 2));

        // No device descriptor, so the invented window applies: tight = 2 × 4,
        // bpr = ALIGN_UP(8, ROW_BYTES_ALIGN) = 128, two rows — a pitch wider
        // than the packed row, so there is inter-row padding to get wrong.
        let tight = 8usize;
        let bpr = 128usize;
        let mut got = vec![0u8; page as usize];
        assert!(host.read_gpa(gpa0, &mut got).is_ok());
        let mut want = vec![0xCCu8; page as usize];
        want[..tight].copy_from_slice(&src[..tight]);
        want[bpr..bpr + tight].copy_from_slice(&src[tight..]);
        let first_diff = got.iter().zip(want.iter()).position(|(a, b)| a != b);
        assert_eq!(
            first_diff, None,
            "byte {first_diff:?} outside the sample window was modified"
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
            SurfaceWindow {
                base_off: 0,
                bpr: page as u32,
                span_end: page + row1.len() as u64,
                bpp: 4
            },
            Rect {
                origin_x: 0,
                origin_y: 0,
                width: 2,
                height: 2
            },
            &mut dst,
            8
        ));
        assert_eq!(&dst[..8], &row0);
        assert_eq!(&dst[8..], &row1);
    }

    /// A rect ending past the sample window must be refused the same way and
    /// named the same way whichever arm reads it. The bound used to live inside
    /// the contig arm, so the fragmented arm — the one a driven x86 boot takes —
    /// answered an overrun with a bare `false` from a slice index and no line
    /// saying the rect had left the window.
    ///
    /// Run over both arms from one body so the two cannot drift: a single packed
    /// page takes the contig arm, two scattered pages take the fragmented one.
    #[test]
    fn a_rect_past_the_sample_window_is_named_on_both_read_arms() {
        use crate::model::PAGE_SHIFT_X86;

        for scattered in [false, true] {
            let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
            let mut host = FakeHost::new();
            host.strict_linux_map = true;
            let page = 1u64 << PAGE_SHIFT_X86;
            let gpa0 = 0x5100_0000u64;
            let gpa1 = 0x6200_0000u64;
            host.map_range(gpa0, page as usize, 0);
            host.map_range(gpa1, page as usize, 0);
            let pfn0 = (gpa0 >> PAGE_SHIFT_X86) as u32;
            let pfn1 = (gpa1 >> PAGE_SHIFT_X86) as u32;
            let mid = 12u32;
            state.map_surface(mid);
            {
                let m = state.mappings.get_mut(&mid).unwrap();
                m.mapped = true;
                m.mapping_internal = 1;
                // One page is a packed view; two distant ones cannot be.
                m.page_entries = if scattered {
                    vec![
                        (pfn0 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID,
                        (pfn1 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID,
                    ]
                } else {
                    vec![(pfn0 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID]
                };
            }
            assert_eq!(
                mapper::ensure_contig_view(&mut state, &mut host, mid).is_some(),
                !scattered,
                "scattered={scattered} must select the arm this iteration is about"
            );

            // The window is one page. Asking for four rows at a one-page pitch
            // puts the fourth row's last byte three pages past its end.
            let mut dst = [0u8; 4 * 8];
            let cap = crate::observe::FailCapture::start();
            let ok = read_rect_raw_at(
                &mut state,
                &mut host,
                mid,
                SurfaceWindow {
                    base_off: 0,
                    bpr: page as u32,
                    span_end: page,
                    bpp: 4,
                },
                Rect {
                    origin_x: 0,
                    origin_y: 0,
                    width: 2,
                    height: 4,
                },
                &mut dst,
                8,
            );
            let overruns: Vec<String> = cap
                .lines()
                .into_iter()
                .filter(|l| l.contains("reason=read_overrun"))
                .collect();
            assert!(!ok, "scattered={scattered}: the read must refuse");
            assert_eq!(
                overruns.len(),
                1,
                "scattered={scattered}: the refusal must name the bound it broke, \
                 not leave the caller's decline to stand for it: {overruns:?}"
            );
        }
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
            SurfaceWindow {
                base_off: 0,
                bpr,
                span_end: span,
                bpp: 4
            },
            Rect {
                origin_x: 0,
                origin_y: 0,
                width: bpr / 4,
                height: 2
            },
            &mut tight,
            bpr
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

    /// A published descriptor that resolves no plane is not the same state as no
    /// descriptor at all, and the device must not answer them alike.
    ///
    /// Both used to return the packed window over offset 0. For the second that
    /// is the only layout information anyone has; for the first the guest has
    /// already said where its planes are and this texture matched two of them —
    /// a v0a8 surface's Y and alpha planes share format and geometry, so the
    /// scan cannot separate them and plane 0's bytes would be bound for a sample
    /// the wire meant for alpha, silently.
    #[test]
    fn an_ambiguous_descriptor_declines_where_an_absent_one_still_sizes_a_window() {
        use crate::contract::endian::{st16, st32, st64};
        use crate::contract::iosurface_pages::{
            DEVICE_DESC_ALLOC_SIZE, DEVICE_DESC_PLANES, DEVICE_DESC_PLANE_COUNT, DEVICE_PLANE_BPE,
            DEVICE_PLANE_BPR, DEVICE_PLANE_DESC_LEN, DEVICE_PLANE_DIMS, DEVICE_PLANE_OFFSET,
            DEVICE_PLANE_SIZE,
        };
        use crate::contract::pixel_format::MTL_FORMAT_R8_UNORM;

        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        state.map_surface(8);

        // No descriptor yet: geometry came from the type-11 texture object and
        // the aligned row stands in for the pitch. 4 R8 texels align to 128.
        let m = state.mappings.get(&8).expect("mapping");
        assert_eq!(
            type11_sample_window(m, 4, 2, MTL_FORMAT_R8_UNORM),
            Some((0, 128, 256)),
            "with nothing published there are no planes to confuse"
        );

        // Publish a v0a8-shaped descriptor: planes 0 and 2 are both R8 4x2.
        let mut desc = vec![0u8; crate::contract::iosurface_pages::DEVICE_DESC_LEN];
        st32(&mut desc[DEVICE_DESC_ALLOC_SIZE..], 0x2000);
        desc[DEVICE_DESC_PLANE_COUNT] = 3;
        let pack = |w: u32, h: u32| ((w as u64 & 0xffffff) << 8) | ((h as u64 & 0xffffff) << 40);
        for (i, (off, w, h, bpe)) in [(512u32, 4u32, 2u32, 1u16), (1024, 2, 1, 2), (1536, 4, 2, 1)]
            .iter()
            .enumerate()
        {
            let p = DEVICE_DESC_PLANES + i * DEVICE_PLANE_DESC_LEN;
            st32(&mut desc[p + DEVICE_PLANE_OFFSET..], *off);
            st32(&mut desc[p + DEVICE_PLANE_SIZE..], 256);
            st64(&mut desc[p + DEVICE_PLANE_DIMS..], pack(*w, *h));
            st32(&mut desc[p + DEVICE_PLANE_BPR..], 64);
            st16(&mut desc[p + DEVICE_PLANE_BPE..], *bpe);
        }
        assert!(state.set_mapping_device_desc(8, &desc));

        let m = state.mappings.get(&8).expect("mapping");
        assert_eq!(
            type11_sample_window(m, 4, 2, MTL_FORMAT_R8_UNORM),
            None,
            "two planes match and neither is the answer, so nothing is bound"
        );
        // The wire index is the only thing that separates them, and it reaches
        // each of the two directly.
        assert_eq!(
            type5_sample_window(m, 0, 4, 2, MTL_FORMAT_R8_UNORM).map(|w| w.0),
            Some(512)
        );
        assert_eq!(
            type5_sample_window(m, 2, 4, 2, MTL_FORMAT_R8_UNORM).map(|w| w.0),
            Some(1536)
        );
        // An index past the plane count resolves nothing rather than falling
        // back onto plane 0's bytes.
        assert_eq!(type5_sample_window(m, 7, 4, 2, MTL_FORMAT_R8_UNORM), None);
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
                &mut state,
                &mut host,
                11,
                SurfaceWindow {
                    base_off: 0,
                    bpr,
                    span_end,
                    bpp
                },
                Rect {
                    origin_x: 0,
                    origin_y: 0,
                    width,
                    height: 100
                },
                &mut big,
                bpr
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
                &mut state,
                &mut host,
                11,
                SurfaceWindow {
                    base_off: 0,
                    bpr,
                    span_end,
                    bpp
                },
                Rect {
                    origin_x: 0,
                    origin_y: 0,
                    width,
                    height: 2
                },
                &mut ok,
                bpr
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
            &mut state,
            &mut host,
            7,
            Rect {
                origin_x: 0,
                origin_y: 0,
                width: 4,
                height: 1
            },
            &mut row,
            16
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

    /// The depth/stencil writeback must name its refusals too.
    ///
    /// `write_raw_rows` is the third guest-memory writer in this file and was
    /// the last one still answering every refusal with a bare `false`. It is
    /// worse placed than the others to be silent: both callers discard its
    /// result outright, so nothing above it could report a reason even if it
    /// wanted to, and the guest work it drops is a `MTLStoreActionStore` on a
    /// depth/stencil attachment - the mapping simply keeps stale bytes and the
    /// pass reports success. The colour writeback twenty lines from its caller
    /// emits for the analogous condition.
    #[test]
    fn every_raw_rows_refusal_names_itself() {
        use crate::runtime::drain::store_route_count;
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
        assert!(state.set_mapping_geom(7, 4, 2, MTL_FORMAT_BGRA8_UNORM));
        let rows = vec![0u8; 4 * 2 * 4];

        // A zero dimension is not a rect.
        let n = store_route_count("surface_write_geometry");
        assert!(!write_raw_rows(
            &mut state, &mut host, 7, &rows, 16, 16, 0, 2
        ));
        assert_eq!(store_route_count("surface_write_geometry"), n + 1);

        // A source pitch that cannot hold one row.
        let n = store_route_count("surface_write_source_stride");
        assert!(!write_raw_rows(
            &mut state, &mut host, 7, &rows, 4, 16, 4, 2
        ));
        assert_eq!(store_route_count("surface_write_source_stride"), n + 1);

        // The source ends before the rows it declares.
        let n = store_route_count("surface_write_source_short");
        assert!(!write_raw_rows(
            &mut state,
            &mut host,
            7,
            &rows[..8],
            16,
            16,
            4,
            2
        ));
        assert_eq!(store_route_count("surface_write_source_short"), n + 1);

        // No such mapping.
        let n = store_route_count("surface_write_mapping_absent");
        assert!(!write_raw_rows(
            &mut state, &mut host, 4242, &rows, 16, 16, 4, 2
        ));
        assert_eq!(store_route_count("surface_write_mapping_absent"), n + 1);

        // The latched geometry is not this frame's.
        let n = store_route_count("surface_write_geometry_moved");
        let big = vec![0u8; 8 * 8 * 4];
        assert!(!write_raw_rows(
            &mut state, &mut host, 7, &big, 32, 32, 8, 8
        ));
        assert_eq!(store_route_count("surface_write_geometry_moved"), n + 1);

        // Unmapped: there is nowhere to write.
        let n = store_route_count("surface_write_mapping_not_resident");
        state.mappings.get_mut(&7).unwrap().mapped = false;
        assert!(!write_raw_rows(
            &mut state, &mut host, 7, &rows, 16, 16, 4, 2
        ));
        assert_eq!(
            store_route_count("surface_write_mapping_not_resident"),
            n + 1
        );
    }

    /// Every refusal in `write_rgba8_image_changed` must name itself.
    ///
    /// This writeback is the guest's own copy of a rendered frame, and it is
    /// reached on the live x86/Vulkan sync-store route. Every one of its
    /// refusals used to be a bare `false`, so a frame the guest never received
    /// left no trace at all — while the sibling writer in this same file
    /// answered the identical conditions through `SurfaceWriteRefusal`. The
    /// vocabulary was already complete; only this arm did not use it.
    ///
    /// Asserting on the route slugs rather than on the fail lines is what makes
    /// this a regression test: `refuse` latches its line per `(check, mapping)`
    /// but always counts, so a reverted arm shows up as a counter that stops
    /// moving even on a mapping that has already refused once.
    #[test]
    fn every_rgba8_image_changed_refusal_names_itself() {
        use crate::runtime::drain::store_route_count;
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
        assert!(state.set_mapping_geom(7, 4, 2, MTL_FORMAT_BGRA8_UNORM));
        let frame = vec![0u8; 4 * 2 * 4];

        // Each case: (slug, the call that must take that arm).
        let before = |slug: &str| store_route_count(slug);

        // A zero dimension is not a rect.
        let n = before("surface_write_geometry");
        assert!(!write_rgba8_image_changed(
            &mut state, &mut host, 7, &frame, None, 0, 2
        ));
        assert_eq!(store_route_count("surface_write_geometry"), n + 1);

        // The source ends before the frame it declares.
        let n = before("surface_write_source_short");
        assert!(!write_rgba8_image_changed(
            &mut state,
            &mut host,
            7,
            &frame[..4],
            None,
            4,
            2
        ));
        assert_eq!(store_route_count("surface_write_source_short"), n + 1);

        // The seed ends before it: a different buffer, so a different slug.
        let n = before("surface_write_seed_short");
        assert!(!write_rgba8_image_changed(
            &mut state,
            &mut host,
            7,
            &frame,
            Some(&frame[..4]),
            4,
            2
        ));
        assert_eq!(
            store_route_count("surface_write_seed_short"),
            n + 1,
            "a short seed must not be reported as a short source"
        );

        // No such mapping: the surface went away between the arm and the landing.
        let n = before("surface_write_mapping_absent");
        assert!(!write_rgba8_image_changed(
            &mut state, &mut host, 4242, &frame, None, 4, 2
        ));
        assert_eq!(store_route_count("surface_write_mapping_absent"), n + 1);

        // The latched geometry is not the frame's: landing it would skew.
        let n = before("surface_write_geometry_moved");
        let big = vec![0u8; 8 * 8 * 4];
        assert!(!write_rgba8_image_changed(
            &mut state, &mut host, 7, &big, None, 8, 8
        ));
        assert_eq!(store_route_count("surface_write_geometry_moved"), n + 1);

        // Unmapped: there is nowhere to write.
        let n = before("surface_write_mapping_not_resident");
        state.mappings.get_mut(&7).unwrap().mapped = false;
        assert!(!write_rgba8_image_changed(
            &mut state, &mut host, 7, &frame, None, 4, 2
        ));
        assert_eq!(
            store_route_count("surface_write_mapping_not_resident"),
            n + 1
        );
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
            &mut state,
            &mut host,
            6,
            Rect {
                origin_x: 0,
                origin_y: 0,
                width: 4,
                height: 1
            },
            &mut row,
            16
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
            &mut state,
            &mut host,
            5,
            Rect {
                origin_x: 1,
                origin_y: 1,
                width: 2,
                height: 1
            },
            &src,
            8
        ));
        let mut dst = [0u8; 8];
        assert!(read_rect_raw(
            &mut state,
            &mut host,
            5,
            Rect {
                origin_x: 1,
                origin_y: 1,
                width: 2,
                height: 1
            },
            &mut dst,
            8
        ));
        assert_eq!(dst, src);
        // OOB origin fails.
        assert!(!write_rect_raw(
            &mut state,
            &mut host,
            5,
            Rect {
                origin_x: 3,
                origin_y: 0,
                width: 2,
                height: 1
            },
            &src,
            8
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
            &mut state,
            &mut host,
            mid,
            SurfaceWindow {
                base_off,
                bpr,
                span_end,
                bpp: 4
            },
            Rect {
                origin_x: 0,
                origin_y: 0,
                width: 4,
                height: 4
            },
            &mut dst,
            16
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
