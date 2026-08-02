//! How much of a render writeback is already in the guest's pages.
//!
//! # Why
//!
//! The writeback rail is the largest cost in this device and it is priced in
//! bytes. On a driven x86/PCI drag second it moves 2.4 GB into guest RAM across
//! 290 landings, and that costs 398 ms of GPU fence — 310 ms of which is the
//! copy executing — plus 285 ms of CPU stores, out of a 1040 ms busy second.
//! Removing the rail outright runs the same workload at 2.9x the guest frame
//! rate, so anything that removes *bytes* from it is worth close to 1:1.
//!
//! One reduction has already been measured and rejected: landing only the
//! rectangle the guest declared damaged. `note_store_damage_coverage` reads
//! `store_damage_texels / store_attach_texels` at **99.34%**, because the Store
//! that ends a full-screen composite declares the full screen. The whole
//! declared rect is worth 0.66%.
//!
//! **That is a different question from this one.** A guest that re-composites
//! the whole desktop every frame declares the whole screen damaged and then
//! produces, for most of it, the bytes that are already there — the wallpaper
//! under a moving window does not change because the window moved. Bytes that
//! are bit-identical to what the page already holds need not be written at all,
//! and no line here has ever counted them.
//!
//! # Why the skip it prices would be sound
//!
//! Not writing a byte that already has the value being written leaves memory in
//! the same state, so this is an identity rather than a heuristic — there is no
//! rect to guess, no content pattern to match, and no observation to overfit.
//! The two witnesses the writeback feeds stay sound with it, which is the
//! requirement [`crate::runtime::storage_flush::flush_mapping_windows_before_fence`]
//! records after the counterfactual boot traded a 2.26 GB/s writeback for an
//! 8 MB-per-bind gather:
//!
//! - [`crate::runtime::host_writes`] is this device's page-exact record of its
//!   own stores, which [`crate::runtime::gather_witness`] subtracts to tell them
//!   from the guest's. A page this rail declines to write was not written, so
//!   *not* recording it is the accurate answer, not a gap.
//! - The type-11 resident rung asks whether the guest replaced the surface. A
//!   skipped page still holds the frame this landing would have put there, so
//!   the rung sees exactly what an eager landing would have left it.
//!
//! That is the difference from `REIMS_VGPU_PROBE_NO_RENDER_WRITEBACK`, which
//! skips landings whose content *does* differ and leaves pages holding neither
//! frame.
//!
//! # What this measures, and what it is not
//!
//! It is a counter, not a rail. On one write in [`AUDIT_STRIDE`] it compares the
//! bytes about to be stored against the bytes already in the guest's pages and
//! reports how many matched, at two granularities, because the granularity is
//! what decides which rail is worth building:
//!
//! - **`page`** — the guest page. The unit a CPU-side skip would work in, and
//!   the unit the write witness above is page-exact in.
//! - **`fine`** — [`FINE_TILE`] bytes, 64 BGRA8 texels. The unit a GPU-side
//!   compaction would work in. That is the only route at the 310 ms of copy,
//!   because a CPU compare happens after the bytes have already crossed the
//!   bus — and, per the refutation below, it turned out to be the only route at
//!   the CPU scatter too.
//!
//! The two differ where change is dense but narrow. A window moving sideways
//! changes a vertical band, so it touches nearly every row of the frame: whole
//! *rows* would report almost nothing redundant while fine tiles report most of
//! each row untouched. Reporting one granularity would have answered the wrong
//! question in either direction, which is why both are here.
//!
//! # What it measured, and what that is worth
//!
//! Settled x86/PCI guest, `window-drag-probe --seconds 15` moving a 1000x640
//! Safari window, host GPU at its own clock. Five landings a second audited over
//! fourteen consecutive seconds:
//!
//! ```text
//! fine  (256 B)   85.1  85.1  86.1  85.7  85.1  87.6  85.5  86.5  90.0  85.3
//!                 83.9  87.0  89.8  86.3      median 86.1%
//! page  (4 KiB)   43.6  38.6  45.0  44.3  39.5  39.0  39.9  45.1  45.9  35.9
//!                 36.2  49.0  49.3  43.1      median 43.1%
//! ```
//!
//! **86% of what this rail writes is already in the page**, and it is stable to
//! a few points across every second of the drag. The undriven desktop reads
//! 2025/2025 pages identical, so the idle case is total.
//!
//! The gap between the two rows decides which unit a rail should work in: a
//! page-granular one leaves half the redundancy on the table, and a row-granular
//! one would find almost none.
//!
//! # The CPU rail this licensed was built, and it is refuted
//!
//! The obvious next step was to compare each tile in the scatter and store only
//! the runs that differ. That was built, and measured on one settled x86/PCI
//! guest with the same stressor:
//!
//! ```text
//! run     tile rail   bytes declined   land_us per landing
//! drag1   off         -                med 744   (732-760)
//! drag2   off         -                med 769   (737-788)
//! drag4   on          91.6-91.8%       med 802   (791-956)
//! ```
//!
//! **Declining 92 % of the stores made the scatter slower**, and the ranges do
//! not overlap. The cause is that a full-cache-line store does not read its
//! destination — the hardware elides the read-for-ownership — so a store that is
//! declined never cost a read to begin with, and the compare adds a whole 8 MB
//! read of guest RAM that the eager path never paid. What it saves is DRAM write
//! bandwidth, which is not what `land_us` is bound by.
//!
//! That run did confirm the audit, to the decimal: `same_fine` 91.6/91.8/91.8 %
//! against bytes actually declined 91.6/91.8/91.8 %. The measurement and the
//! mechanism agree; it is the mechanism's *economics* that were wrong.
//!
//! # What the refutation leaves, and it is the number the GPU pass is priced on
//!
//! The failure is entirely the compare, not the skipping, and the same run
//! separates them. With a full landing at 744 µs = read `src` + write `dst`, a
//! scatter handed the changed-tile set from outside does read `0.08 * src` and
//! write `0.08 * dst` and nothing else:
//!
//! ```text
//! 0.082 * 744 us  =  ~61 us per landing,  saving ~683 us
//! 683 us * 272 landings/s  =  ~186 ms/s   of a ~990 ms busy second
//! ```
//!
//! So a **GPU-side pass is the only route, and now for two reasons rather than
//! one**: it is the only thing that can decline the copy across the bus, which
//! is 78 % of the readback fence — and it is also the only way to get the
//! scatter's own saving, because the CPU cannot derive the tile set for less
//! than the saving is worth. Both halves need the same bitmap and neither is
//! reachable without it.
//!
//! # Which writers it covers, and the one it does not
//!
//! The numbers above were taken through a single hook on
//! [`crate::runtime::mapper`]'s `copy_mapping_runs`, which is one of the
//! guest-RAM writers [`crate::observe::gate`]'s `MAP_PAGES_SITES` classifies and
//! not all of them. Two more now have their own hook, and reading the table
//! rather than grepping is what found them:
//!
//! - `mapping_write`'s BGRA row writers poke rows into a contig view and reach
//!   the mapper not at all. The boot above read `write_split contig=0 frag=272`,
//!   so it happened to miss nothing — but on a host whose mappings are
//!   host-contiguous the same build would have reported `contig=N frag=0` and
//!   this line would have gone **silent rather than wrong**, which is the harder
//!   failure to notice.
//! - The raw task-GVA leg, which `store_routes` reads at
//!   `gvaw_fence_flush=444` against `mapw_fence_flush=288` on a driven drag
//!   second — the *larger* of the two by flush count, and never measured at
//!   all until now. It needs **two** hooks, and the first cut had only one:
//!   `metal_draw::write_gva_rgba8_within` writes rows through a packed
//!   `map_fresh_span_within` when the span resolves and falls to
//!   `gva_view::write_span_multi` a row at a time when it does not, and on a
//!   driven boot it is almost always the latter. With only the packed half
//!   hooked the leg reported **7 runs and 2268 bytes in a whole second**
//!   against those 444 flushes, which reads as "this leg moves nothing"
//!   when it meant "this hook is not where the bytes go".
//!
//! The sampling unit differs between the two legs and the fractions are still
//! comparable. A mapping walk is a landing, so its stride samples whole frames;
//! a `write_span_multi` call is one row, so the GVA leg's stride samples rows
//! scattered across many windows. For a *fraction* over many samples that is
//! fine — it is an unbiased estimate either way. It is not fine for any claim
//! about a particular frame, and only the mapping leg can support one.
//!
//! The two legs are reported on separate lines ([`Leg`]) because they are
//! separate rails with separate arm and flush paths, and one blended fraction
//! over both would describe neither. Note which leg a number came from before
//! carrying it: only the mapping one is what
//! `REIMS_VGPU_PROBE_NO_RENDER_WRITEBACK` dropped to measure 2.9x.
//!
//! Still uncovered: `gva_view::write_span_multi` and the `FreshSpan` writers in
//! `compute_exec`. Neither is on the render writeback's fence path, which is
//! what this measures.
//!
//! Partial chunks at the ends of a compared range are counted in `bytes` and in
//! neither chunk total, so a chunk count is always of whole chunks.
//!
//! There is no byte-exact match count. It would need its own pass — a `==` on
//! whole chunks vectorises and stops at the first difference, a per-byte tally
//! does neither — and no rail can decline a byte, so the number would have cost
//! a third of the audit's budget to answer a question nobody can act on. The
//! page total is derived from the fine one instead of walking the range twice,
//! which is exact because every guest page size this runs on is a whole multiple
//! of [`FINE_TILE`] and both are aligned in the same space.

use std::sync::atomic::{AtomicU64, Ordering};

/// Audit one write in this many.
///
/// The audit reads the destination range it is about to compare, so an audited
/// landing costs one extra pass over ~8 MB, ~1-2 ms. At the measured 290
/// landings a second that is ~4.5 audits and under 1% of the drain worker's
/// second — the same order as
/// [`crate::runtime::gather_witness::AUDIT_STRIDE`], and for the same reason:
/// an audit that has to be afforded is one that gets turned off.
pub const AUDIT_STRIDE: u64 = 64;

/// The fine granularity, in bytes: 64 texels of a BGRA8 surface.
///
/// A GPU compaction pass emits whole tiles, so the tile is what its saving is
/// quantised to. 64 texels is one row of a 64x64 tile and is a multiple of every
/// cache line this runs on, so a fine chunk never straddles one.
pub const FINE_TILE: usize = 256;

/// Free-running, and deliberately **not** cleared by [`take_window`].
///
/// The stride is a property of the write stream, not of the reporting window. A
/// counter the census zeroed would restart the stride every second, so the first
/// write after each window boundary would be due — which on an idle desktop
/// landing four windows a second means auditing all four, and on a driven one
/// silently changes the sample rate with the load.
static STRIDE_TICK: AtomicU64 = AtomicU64::new(0);
static CALLS: AtomicU64 = AtomicU64::new(0);
static AUDITS: AtomicU64 = AtomicU64::new(0);

/// Which writeback leg a compared range belongs to.
///
/// The two are separate rails with separate arm and flush paths, and
/// `store_routes` counts them apart (`mapw_fence_flush` against
/// `gvaw_fence_flush`, 288 against 432 on a driven drag second). One blended
/// fraction over both would be a number describing neither, and a rail is built
/// against one leg at a time.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Leg {
    /// The mapping-keyed rail: `mapper::copy_mapping_runs` and
    /// `mapping_write`'s contig row writers. The leg
    /// `REIMS_VGPU_PROBE_NO_RENDER_WRITEBACK` drops to measure 2.9x.
    Mapping = 0,
    /// The raw task-GVA rail: `metal_draw::write_gva_rgba8_within` through a
    /// fresh span. Never measured before this counter existed.
    Gva = 1,
}

impl Leg {
    const ALL: [Self; 2] = [Self::Mapping, Self::Gva];

    pub fn label(self) -> &'static str {
        match self {
            Self::Mapping => "mapping",
            Self::Gva => "gva",
        }
    }
}

const LEGS: usize = 2;

static RUNS: [AtomicU64; LEGS] = [const { AtomicU64::new(0) }; LEGS];
static BYTES: [AtomicU64; LEGS] = [const { AtomicU64::new(0) }; LEGS];
static PAGES: [AtomicU64; LEGS] = [const { AtomicU64::new(0) }; LEGS];
static SAME_PAGES: [AtomicU64; LEGS] = [const { AtomicU64::new(0) }; LEGS];
static FINE: [AtomicU64; LEGS] = [const { AtomicU64::new(0) }; LEGS];
static SAME_FINE: [AtomicU64; LEGS] = [const { AtomicU64::new(0) }; LEGS];

/// One leg's window of the audit, as taken by the per-second census.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct LandRedundancyWindow {
    /// Walks audited, against `calls` offered, across **both** legs — the
    /// stride is one sequence over every writer, so these two cannot be
    /// attributed to a leg and are the same on each line.
    pub audits: u64,
    pub calls: u64,
    /// Contiguous ranges compared in this leg. A fragmented landing is hundreds
    /// of runs of one frame, so this is not a second frame count — keeping it
    /// separate is what stops the two being read as one.
    pub runs: u64,
    pub bytes: u64,
    pub pages: u64,
    pub same_pages: u64,
    pub fine: u64,
    pub same_fine: u64,
}

/// Take and clear the window, one entry per leg that compared anything.
///
/// A leg with no ranges is left out rather than emitted as zeros: a line of
/// zeros reads as "nothing was redundant" when it means "nothing was measured",
/// and the two call for opposite conclusions.
pub fn take_window() -> Vec<(Leg, LandRedundancyWindow)> {
    let audits = AUDITS.swap(0, Ordering::Relaxed);
    let calls = CALLS.swap(0, Ordering::Relaxed);
    Leg::ALL
        .into_iter()
        .filter_map(|leg| {
            let i = leg as usize;
            let w = LandRedundancyWindow {
                audits,
                calls,
                runs: RUNS[i].swap(0, Ordering::Relaxed),
                bytes: BYTES[i].swap(0, Ordering::Relaxed),
                pages: PAGES[i].swap(0, Ordering::Relaxed),
                same_pages: SAME_PAGES[i].swap(0, Ordering::Relaxed),
                fine: FINE[i].swap(0, Ordering::Relaxed),
                same_fine: SAME_FINE[i].swap(0, Ordering::Relaxed),
            };
            (w.runs > 0).then_some((leg, w))
        })
        .collect()
}

/// Whether this write is the one in [`AUDIT_STRIDE`] that gets compared.
///
/// Counted per call to the walk rather than per run, so one landing is audited
/// whole and the fraction it reports is a fraction of a frame. A per-run stride
/// would sample scattered pieces of different landings and report their mean as
/// though it described one.
pub fn audit_due() -> bool {
    CALLS.fetch_add(1, Ordering::Relaxed);
    let due = STRIDE_TICK
        .fetch_add(1, Ordering::Relaxed)
        .is_multiple_of(AUDIT_STRIDE);
    if due {
        AUDITS.fetch_add(1, Ordering::Relaxed);
    }
    due
}

/// Compare `src` against the `src.len()` bytes at `dst` and charge the match to
/// the window.
///
/// `map_off` is where the write lands in mapping-linear space, and it is what
/// the page chunks are aligned to — not the pointer, whose alignment says
/// nothing about which guest page a byte belongs to. `page_size` comes from the
/// caller because guest page geometry is never assumed here.
///
/// # Safety
///
/// `dst` must be readable for `src.len()` bytes. Every caller is about to write
/// exactly that range through the same pointer.
pub unsafe fn note_write(leg: Leg, map_off: u64, dst: *const u8, src: &[u8], page_size: u64) {
    if src.is_empty() {
        return;
    }
    // SAFETY: the caller guarantees `dst` is readable for `src.len()`.
    let dst = unsafe { std::slice::from_raw_parts(dst, src.len()) };
    let i = leg as usize;
    RUNS[i].fetch_add(1, Ordering::Relaxed);
    BYTES[i].fetch_add(src.len() as u64, Ordering::Relaxed);
    let tally = compare(map_off, src, dst, page_size);
    PAGES[i].fetch_add(tally.pages, Ordering::Relaxed);
    SAME_PAGES[i].fetch_add(tally.same_pages, Ordering::Relaxed);
    FINE[i].fetch_add(tally.fine, Ordering::Relaxed);
    SAME_FINE[i].fetch_add(tally.same_fine, Ordering::Relaxed);
}

/// What one compared range contributed, before it reaches the atomics.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct Tally {
    pages: u64,
    same_pages: u64,
    fine: u64,
    same_fine: u64,
}

/// Compare `src` against `dst` in one pass over the bytes and report both
/// granularities.
///
/// Chunks are aligned in **mapping-linear space**, not to the start of the
/// slice. That is what makes the page count mean "guest pages": a run copy can
/// begin part-way into a page, and chunking from the slice start would report
/// chunks straddling two of them as though a rail could decline either. Partial
/// chunks at both ends fall out of the totals for the same reason.
///
/// The page total is folded from the fine one — a page matches exactly when
/// every fine tile in it does — rather than walked separately. Both guest page
/// sizes this device runs on (4 KiB at shift 12, 16 KiB at shift 14) are whole
/// multiples of [`FINE_TILE`] and share its alignment, so the fold is exact
/// rather than an approximation; a page size that was not is reported as no
/// pages rather than as wrong ones.
fn compare(map_off: u64, src: &[u8], dst: &[u8], page_size: u64) -> Tally {
    let mut t = Tally::default();
    let fine = FINE_TILE as u64;
    let Ok(head) = usize::try_from(map_off.next_multiple_of(fine) - map_off) else {
        return t;
    };
    if head >= src.len() {
        return t;
    }
    // Fine tiles per page, and where in a page the first whole tile sits, so a
    // page is closed on the tile that ends it rather than on a count of tiles
    // seen since the range began.
    let per_page = if page_size.is_multiple_of(fine) {
        page_size / fine
    } else {
        0
    };
    let first_tile = (map_off + head as u64) / fine;
    let mut page_run = 0u64;
    let mut page_same = true;
    for (n, (s, d)) in src[head..]
        .chunks_exact(FINE_TILE)
        .zip(dst[head..].chunks_exact(FINE_TILE))
        .enumerate()
    {
        let tile_index = first_tile + n as u64;
        let same = s == d;
        t.fine += 1;
        t.same_fine += u64::from(same);
        if per_page != 0 {
            // A whole page is one whose first tile is its own page-aligned
            // first: `page_run` only starts counting there.
            if tile_index.is_multiple_of(per_page) {
                page_run = 0;
                page_same = true;
            }
            page_same &= same;
            page_run += 1;
            if page_run == per_page {
                t.pages += 1;
                t.same_pages += u64::from(page_same);
                page_run = 0;
            }
        }
    }
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    const PAGE: u64 = 4096;

    /// The single leg a test exercised, so a test reads one window rather than
    /// indexing a vector whose length is the assertion.
    fn only_leg() -> Option<LandRedundancyWindow> {
        let mut legs = take_window();
        assert!(legs.len() <= 1, "a test touched both legs: {legs:?}");
        legs.pop().map(|(_, w)| w)
    }

    fn reset() {
        let _ = take_window();
        CALLS.store(0, Ordering::Relaxed);
        AUDITS.store(0, Ordering::Relaxed);
        STRIDE_TICK.store(0, Ordering::Relaxed);
    }

    /// An idle second emits nothing rather than a row of zeros.
    #[test]
    fn a_window_with_no_audit_is_none() {
        reset();
        assert!(take_window().is_empty());
    }

    /// Identical content reports every byte and every chunk matched. This is the
    /// reading the whole route is priced on, so it is worth a test that it is
    /// reachable at all.
    #[test]
    fn identical_content_matches_wholly() {
        reset();
        let src = vec![0xABu8; 3 * PAGE as usize];
        let dst = src.clone();
        unsafe { note_write(Leg::Mapping, 0, dst.as_ptr(), &src, PAGE) };
        let w = only_leg().expect("one audit");
        assert_eq!(w.bytes, 3 * PAGE, "{w:?}");
        assert_eq!(w.pages, 3, "{w:?}");
        assert_eq!(w.same_pages, 3, "{w:?}");
        assert_eq!(w.fine, 3 * PAGE / FINE_TILE as u64, "{w:?}");
        assert_eq!(w.same_fine, w.fine, "{w:?}");
    }

    /// A single differing byte kills its page and its fine tile and no others.
    /// The two granularities exist to be different, and a rail built on the
    /// coarse count would decline a page the fine count says is 94% reusable.
    #[test]
    fn one_differing_byte_kills_only_its_own_chunks() {
        reset();
        let src = vec![0u8; 2 * PAGE as usize];
        let mut dst = src.clone();
        dst[PAGE as usize + 10] = 1;
        unsafe { note_write(Leg::Mapping, 0, dst.as_ptr(), &src, PAGE) };
        let w = only_leg().expect("one audit");
        assert_eq!((w.pages, w.same_pages), (2, 1), "{w:?}");
        let fine = 2 * PAGE / FINE_TILE as u64;
        assert_eq!((w.fine, w.same_fine), (fine, fine - 1), "{w:?}");
    }

    /// Chunks are aligned to mapping-linear space, not to the slice. A run that
    /// starts mid-page contributes only the whole guest pages it covers, so a
    /// page count never includes one this rail could not decline as a unit.
    #[test]
    fn chunks_align_to_mapping_space_not_to_the_slice() {
        reset();
        let src = vec![7u8; 2 * PAGE as usize];
        let dst = src.clone();
        // Starts 1 KiB into a page: pages 1 and 2 of the range are whole, the
        // head and the tail are not.
        unsafe { note_write(Leg::Mapping, PAGE + 1024, dst.as_ptr(), &src, PAGE) };
        let w = only_leg().expect("one audit");
        assert_eq!(w.pages, 1, "{w:?}");
        assert_eq!(w.same_pages, 1, "{w:?}");
        assert_eq!(w.bytes, 2 * PAGE, "{w:?}");
    }

    /// A range shorter than one aligned chunk reports bytes and no chunks,
    /// rather than reporting a partial chunk as a whole one.
    #[test]
    fn a_range_below_one_aligned_chunk_reports_no_chunk() {
        reset();
        let src = [1u8; 64];
        let dst = [2u8; 64];
        unsafe { note_write(Leg::Mapping, PAGE - 32, dst.as_ptr(), &src, PAGE) };
        let w = only_leg().expect("one audit");
        assert_eq!((w.pages, w.fine), (0, 0), "{w:?}");
        assert_eq!(w.bytes, 64, "{w:?}");
    }

    /// A page whose fine tiles all match is a matching page, and one whose tiles
    /// differ only outside it is not charged for them. The page total is folded
    /// from the fine walk rather than measured, so the fold is what is tested.
    #[test]
    fn the_page_fold_closes_pages_on_their_own_boundaries() {
        reset();
        let src = vec![0u8; 4 * PAGE as usize];
        let mut dst = src.clone();
        // One byte in the last tile of page 0 and one in the first tile of
        // page 3. Pages 1 and 2 are untouched and must survive.
        dst[PAGE as usize - 1] = 9;
        dst[3 * PAGE as usize] = 9;
        unsafe { note_write(Leg::Mapping, 0, dst.as_ptr(), &src, PAGE) };
        let w = only_leg().expect("one audit");
        assert_eq!((w.pages, w.same_pages), (4, 2), "{w:?}");
        let fine = 4 * PAGE / FINE_TILE as u64;
        assert_eq!((w.fine, w.same_fine), (fine, fine - 2), "{w:?}");
    }

    /// A guest page size that is not a whole multiple of the fine tile reports
    /// no pages rather than pages folded on the wrong boundary. Neither shipped
    /// page shift is such a size; this fixes what happens if one ever is.
    #[test]
    fn a_page_size_the_fold_cannot_divide_reports_no_pages() {
        reset();
        let src = vec![0u8; 4 * PAGE as usize];
        let dst = src.clone();
        unsafe { note_write(Leg::Mapping, 0, dst.as_ptr(), &src, FINE_TILE as u64 + 1) };
        let w = only_leg().expect("one audit");
        assert_eq!((w.pages, w.same_pages), (0, 0), "{w:?}");
        assert!(w.fine > 0, "{w:?}");
    }

    /// The two legs are tallied apart and emitted apart. They are separate
    /// rails at separate rates — `gvaw_fence_flush=432` against
    /// `mapw_fence_flush=288` on a driven second — so a blended fraction would
    /// be a number describing neither, weighted by whichever leg happened to
    /// write more bytes.
    #[test]
    fn the_two_legs_are_never_blended() {
        reset();
        let src = vec![0u8; 4 * PAGE as usize];
        let same = src.clone();
        let mut differs = src.clone();
        for b in differs.iter_mut() {
            *b = 1;
        }
        unsafe { note_write(Leg::Mapping, 0, same.as_ptr(), &src, PAGE) };
        unsafe { note_write(Leg::Gva, 0, differs.as_ptr(), &src, PAGE) };
        let legs = take_window();
        assert_eq!(legs.len(), 2, "{legs:?}");
        let get = |want: Leg| legs.iter().find(|(l, _)| *l == want).expect("leg").1;
        let (m, g) = (get(Leg::Mapping), get(Leg::Gva));
        assert_eq!(m.same_fine, m.fine, "the mapping leg was wholly redundant");
        assert_eq!(g.same_fine, 0, "the gva leg matched nothing");
        assert_eq!(m.runs, 1, "{m:?}");
        assert_eq!(g.runs, 1, "{g:?}");
    }

    /// A leg that measured nothing emits nothing. A row of zeros reads as
    /// "measured and not redundant", which is the opposite conclusion from
    /// "not measured", and both legs are hooked on paths a given host may never
    /// take.
    #[test]
    fn a_leg_that_measured_nothing_is_left_out() {
        reset();
        let src = [7u8; FINE_TILE];
        let dst = src;
        unsafe { note_write(Leg::Gva, 0, dst.as_ptr(), &src, PAGE) };
        let legs = take_window();
        assert_eq!(legs.len(), 1, "{legs:?}");
        assert_eq!(legs[0].0, Leg::Gva);
    }

    /// One write in `AUDIT_STRIDE` is compared, and `calls` records the rest so
    /// a reader can tell a sampled fraction from a total.
    #[test]
    fn the_stride_admits_one_call_in_stride() {
        reset();
        let mut due = 0;
        for _ in 0..(AUDIT_STRIDE * 3) {
            due += u64::from(audit_due());
        }
        assert_eq!(due, 3);
    }

    /// The stride survives the census taking its window. It is a property of the
    /// write stream, and a counter the census zeroed would make every window's
    /// first write due — which on an idle desktop is every landing, and on a
    /// driven one is a sample rate that moves with the load.
    #[test]
    fn taking_the_window_does_not_restart_the_stride() {
        reset();
        // Tick 0 is due, then half a stride of ticks that are not, then a
        // census, then the rest of the stride: the next due write must still be
        // tick `AUDIT_STRIDE` and not the first one after the boundary.
        assert!(audit_due(), "the stride's first tick is due");
        for _ in 1..(AUDIT_STRIDE / 2) {
            assert!(!audit_due());
        }
        let _ = take_window();
        for _ in (AUDIT_STRIDE / 2)..AUDIT_STRIDE {
            assert!(!audit_due(), "the window boundary made a write due");
        }
        assert!(audit_due(), "the stride's own tick did not come due");
    }

    /// `audits` counts walks and `runs` counts the ranges inside them. A
    /// fragmented landing is hundreds of runs of one frame, so reporting one
    /// number for both would read as hundreds of frames sampled a second.
    #[test]
    fn walks_and_the_runs_inside_them_are_counted_apart() {
        reset();
        assert!(audit_due());
        let src = [3u8; FINE_TILE * 2];
        let dst = src;
        for run in 0..4u64 {
            unsafe { note_write(Leg::Mapping, run * FINE_TILE as u64 * 2, dst.as_ptr(), &src, PAGE) };
        }
        let w = only_leg().expect("one audited walk");
        assert_eq!((w.audits, w.runs), (1, 4), "{w:?}");
        assert_eq!(w.bytes, 4 * 2 * FINE_TILE as u64, "{w:?}");
    }
}
