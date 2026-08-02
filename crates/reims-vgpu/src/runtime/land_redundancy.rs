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
//!   compaction could work in, which is the only route at the 310 ms of copy;
//!   a CPU compare happens after the bytes have already crossed the bus.
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
//! The gap between the two rows is the whole reason both are reported, and it
//! decides which rail is worth building. Against the same second's
//! `drain_duty busy_us=975666`:
//!
//! - A **page-granular CPU skip** declines 43% of the stores in `write_split`'s
//!   `land_us=209176`, and nothing at all of `readback_split`'s
//!   `gpu_us=300144`, because the bytes have already crossed the bus by then.
//! - A **tile-granular GPU compaction** declines 86% of *both*: ~258 ms of copy
//!   and, since the pass also says which tiles moved, ~180 ms of scatter that no
//!   longer needs a compare to skip. That is ~45% of a saturated worker's
//!   second, against the ~73% the whole writeback costs.
//!
//! So the coarse number would have priced the opportunity at a third of what it
//! is, and priced it on the smaller of the two costs.
//!
//! # What it does not cover
//!
//! The audit hangs off [`crate::runtime::mapper`]'s `copy_mapping_runs`, which
//! is one of the guest-RAM writers [`crate::observe::gate`]'s `MAP_PAGES_SITES`
//! classifies and **not** all of them. Two are outside it:
//!
//! - `mapping_write`'s BGRA row writers take a contig view through
//!   `contig_for_write` and poke rows into it without reaching the mapper at
//!   all. `write_split` reports which path a landing took: the boot above reads
//!   `contig=0 frag=272`, so every landing in it was audited, but a host whose
//!   mappings are host-contiguous would report `contig=N frag=0` and this line
//!   would go silent rather than wrong.
//! - The raw task-GVA rails in `gva_view`. `store_routes` reads
//!   `gvaw_fence_flush=432` beside `mapw_fence_flush=288`, so that leg is not
//!   small — but it is also not the leg
//!   `REIMS_VGPU_PROBE_NO_RENDER_WRITEBACK` dropped to measure 2.9x, which is
//!   the mapping-keyed one this covers.
//!
//! Read a fraction here as "of the mapping-keyed landings that took the run
//! path", and read `write_split contig`/`frag` beside it before generalising.
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

//! # The rail this licenses, and why it is opt-in
//!
//! [`write_skipping_identical`] is the mechanism the audit above prices: it
//! compares each aligned tile against the destination and stores only the runs
//! that differ, coalescing consecutive differing tiles so a partial landing is
//! still a small number of large `memcpy`s rather than 32 400 tiny ones.
//!
//! It is behind `REIMS_VGPU_TILE_SKIP=1` and **not** on by default, because the
//! audit does not predict its own saving. A skipped store costs a read of the
//! destination that the store's read-for-ownership would have paid anyway, so
//! the saving is the store traffic and the dirty writeback, not the whole
//! store — and the fraction of `land_us` that is, is a property of this host's
//! memory system rather than of the 86 %. The counterfactual
//! `REIMS_VGPU_PROBE_NO_TILE_SKIP=1` keeps the compare and every counter and
//! stores anyway, so one boot can measure both sides against one power state,
//! which is the comparison `AGENTS.md` says is otherwise void.
//!
//! Unlike `REIMS_VGPU_PROBE_NO_RENDER_WRITEBACK`, **neither setting is
//! incorrect.** All three configurations leave guest memory in the same state;
//! they differ only in how many stores it took to get there.
//!
//! The scatter this builds is also the half a GPU-side compaction needs. Such a
//! pass would supply the changed-tile set instead of the CPU deriving it, and
//! would then also decline the copy across the bus — which the CPU compare
//! cannot, because by the time it runs the bytes have already crossed.

use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};

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
static RUNS: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);
static PAGES: AtomicU64 = AtomicU64::new(0);
static SAME_PAGES: AtomicU64 = AtomicU64::new(0);
static FINE: AtomicU64 = AtomicU64::new(0);
static SAME_FINE: AtomicU64 = AtomicU64::new(0);
static STORED: AtomicU64 = AtomicU64::new(0);
static RAILED: AtomicU64 = AtomicU64::new(0);

/// Whether this boot lets the rail decline stores, and whether it stores anyway.
///
/// Two independent latches rather than one tri-state, because they answer
/// different questions and a reader of the log needs to see both: the first says
/// the rail ran at all, the second says its saving was deliberately given back.
/// Each is read once per process.
fn env_latch(name: &str, state: &AtomicU8) -> bool {
    match state.load(Ordering::Relaxed) {
        0 => false,
        1 => true,
        _ => {
            // Exactly `1`, for the reason `render_writeback_counterfactual`
            // records: a switch that also accepted `0` or `false` as "set" runs
            // the opposite of what anyone exporting it to turn the thing off
            // intended.
            let on = std::env::var_os(name).is_some_and(|v| v == "1");
            state.store(u8::from(on), Ordering::Relaxed);
            on
        }
    }
}

/// Whether the tile rail may decline a store. Off by default; see the module
/// doc for why the audit does not predict its own saving.
pub fn tile_skip_enabled() -> bool {
    static STATE: AtomicU8 = AtomicU8::new(u8::MAX);
    env_latch("REIMS_VGPU_TILE_SKIP", &STATE)
}

/// Whether the rail compares and counts but stores every tile anyway.
///
/// The control arm, and it is a *correct* boot: it lands exactly the bytes the
/// eager path lands. It exists so both sides can be measured on one boot in one
/// host GPU power state.
fn tile_skip_counterfactual() -> bool {
    static STATE: AtomicU8 = AtomicU8::new(u8::MAX);
    env_latch("REIMS_VGPU_PROBE_NO_TILE_SKIP", &STATE)
}

/// One window of the audit, as taken by the per-second census.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct LandRedundancyWindow {
    /// Walks audited, against `calls` offered. One walk is one landing's worth
    /// of writes, so this is a count of frames sampled.
    pub audits: u64,
    pub calls: u64,
    /// Contiguous ranges compared across those walks. A fragmented landing is
    /// hundreds of runs of one frame, so this is not a second frame count —
    /// keeping it separate is what stops the two being read as one.
    pub runs: u64,
    pub bytes: u64,
    pub pages: u64,
    pub same_pages: u64,
    pub fine: u64,
    pub same_fine: u64,
    /// Ranges that went through the rail rather than the sampled audit, and the
    /// bytes it actually stored out of their `bytes`. Under the counterfactual
    /// `stored` equals `bytes`, which is how a control boot announces itself in
    /// the line rather than only in the environment.
    pub railed: u64,
    pub stored: u64,
}

/// Take and clear the window. `None` when nothing was audited, so an idle
/// second costs no line.
pub fn take_window() -> Option<LandRedundancyWindow> {
    let w = LandRedundancyWindow {
        audits: AUDITS.swap(0, Ordering::Relaxed),
        calls: CALLS.swap(0, Ordering::Relaxed),
        runs: RUNS.swap(0, Ordering::Relaxed),
        bytes: BYTES.swap(0, Ordering::Relaxed),
        pages: PAGES.swap(0, Ordering::Relaxed),
        same_pages: SAME_PAGES.swap(0, Ordering::Relaxed),
        fine: FINE.swap(0, Ordering::Relaxed),
        same_fine: SAME_FINE.swap(0, Ordering::Relaxed),
        railed: RAILED.swap(0, Ordering::Relaxed),
        stored: STORED.swap(0, Ordering::Relaxed),
    };
    // Gated on ranges compared rather than on walks admitted: a walk that was
    // due and then wrote nothing has no fraction to report, and a line of zeros
    // reads as "nothing was redundant".
    (w.runs > 0).then_some(w)
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
pub unsafe fn note_write(map_off: u64, dst: *const u8, src: &[u8], page_size: u64) {
    if src.is_empty() {
        return;
    }
    // SAFETY: the caller guarantees `dst` is readable for `src.len()`.
    let dst = unsafe { std::slice::from_raw_parts(dst, src.len()) };
    RUNS.fetch_add(1, Ordering::Relaxed);
    BYTES.fetch_add(src.len() as u64, Ordering::Relaxed);
    let tally = compare(map_off, src, dst, page_size);
    PAGES.fetch_add(tally.pages, Ordering::Relaxed);
    SAME_PAGES.fetch_add(tally.same_pages, Ordering::Relaxed);
    FINE.fetch_add(tally.fine, Ordering::Relaxed);
    SAME_FINE.fetch_add(tally.same_fine, Ordering::Relaxed);
}

/// Store `src` at `dst`, declining the aligned tiles that already hold exactly
/// those bytes, and charge what was compared and what was stored to the window.
///
/// Returns the bytes actually stored. Consecutive differing tiles are coalesced
/// into one `copy_nonoverlapping`, so the common shape — a vertical band of
/// change crossing every row — costs one large store per row rather than one per
/// tile. The head and tail bytes outside the first and last aligned tile are
/// always stored: they are not a unit this can decline, and leaving them would
/// lose bytes rather than save them.
///
/// # Safety
///
/// `dst` must be valid for reads **and** writes for `src.len()` bytes. Every
/// caller is about to write exactly that range through the same pointer; the
/// extra requirement over a plain store is the read, which is what the compare
/// needs.
pub unsafe fn write_skipping_identical(
    map_off: u64,
    dst: *mut u8,
    src: &[u8],
    page_size: u64,
) -> u64 {
    if src.is_empty() {
        return 0;
    }
    RUNS.fetch_add(1, Ordering::Relaxed);
    RAILED.fetch_add(1, Ordering::Relaxed);
    BYTES.fetch_add(src.len() as u64, Ordering::Relaxed);
    // The tally is taken before anything is stored, so it describes the frame
    // the guest's pages held rather than the one being put there. The borrow is
    // scoped to this statement: no reference to the destination may outlive a
    // store into its own range.
    //
    // SAFETY: the caller guarantees `dst` is readable for `src.len()` bytes.
    let tally = {
        let dst_read = unsafe { std::slice::from_raw_parts(dst as *const u8, src.len()) };
        compare(map_off, src, dst_read, page_size)
    };
    let store_everything = tile_skip_counterfactual();
    let fine = FINE_TILE as u64;
    let head = usize::try_from(map_off.next_multiple_of(fine) - map_off)
        .unwrap_or(src.len())
        .min(src.len());
    let tail_start = head + ((src.len() - head) / FINE_TILE) * FINE_TILE;
    let mut stored = 0u64;
    // SAFETY: `at + len <= src.len()` at every call below, and the caller
    // guarantees `dst` is writable for that whole length. `src` is the readback
    // staging buffer and `dst` is guest RAM, which never alias.
    let mut store = |at: usize, len: usize| {
        if len == 0 {
            return;
        }
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr().add(at), dst.add(at), len) };
        stored += len as u64;
    };
    store(0, head);
    store(tail_start, src.len() - tail_start);
    // One pass over the whole tiles, emitting a store per maximal run of
    // differing ones. `run_start` is where the current run began, or `None`
    // between runs. Each tile's destination is borrowed for the length of its
    // own comparison and released before the store that follows, which covers
    // only tiles already compared.
    let mut run_start: Option<usize> = None;
    let mut at = head;
    while at < tail_start {
        // SAFETY: `at + FINE_TILE <= tail_start <= src.len()`.
        let same = !store_everything
            && src[at..at + FINE_TILE]
                == *unsafe { std::slice::from_raw_parts(dst.add(at) as *const u8, FINE_TILE) };
        match (same, run_start) {
            (false, None) => run_start = Some(at),
            (true, Some(from)) => {
                store(from, at - from);
                run_start = None;
            }
            _ => {}
        }
        at += FINE_TILE;
    }
    if let Some(from) = run_start {
        store(from, tail_start - from);
    }
    PAGES.fetch_add(tally.pages, Ordering::Relaxed);
    SAME_PAGES.fetch_add(tally.same_pages, Ordering::Relaxed);
    FINE.fetch_add(tally.fine, Ordering::Relaxed);
    SAME_FINE.fetch_add(tally.same_fine, Ordering::Relaxed);
    STORED.fetch_add(stored, Ordering::Relaxed);
    stored
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
        assert!(take_window().is_none());
    }

    /// Identical content reports every byte and every chunk matched. This is the
    /// reading the whole route is priced on, so it is worth a test that it is
    /// reachable at all.
    #[test]
    fn identical_content_matches_wholly() {
        reset();
        let src = vec![0xABu8; 3 * PAGE as usize];
        let dst = src.clone();
        unsafe { note_write(0, dst.as_ptr(), &src, PAGE) };
        let w = take_window().expect("one audit");
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
        unsafe { note_write(0, dst.as_ptr(), &src, PAGE) };
        let w = take_window().expect("one audit");
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
        unsafe { note_write(PAGE + 1024, dst.as_ptr(), &src, PAGE) };
        let w = take_window().expect("one audit");
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
        unsafe { note_write(PAGE - 32, dst.as_ptr(), &src, PAGE) };
        let w = take_window().expect("one audit");
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
        unsafe { note_write(0, dst.as_ptr(), &src, PAGE) };
        let w = take_window().expect("one audit");
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
        unsafe { note_write(0, dst.as_ptr(), &src, FINE_TILE as u64 + 1) };
        let w = take_window().expect("one audit");
        assert_eq!((w.pages, w.same_pages), (0, 0), "{w:?}");
        assert!(w.fine > 0, "{w:?}");
    }

    /// The rail's whole justification: whatever it declines to store, the
    /// destination ends up holding exactly the bytes a plain copy would have
    /// left. Every other test here is about how *many* stores it took.
    #[test]
    fn the_destination_ends_identical_to_what_a_plain_copy_would_leave() {
        reset();
        let len = 8 * PAGE as usize + 133;
        let mut src = vec![0u8; len];
        let mut dst = vec![0u8; len];
        // A destination that already agrees in wide stretches and differs in
        // scattered runs, plus a head and tail outside any aligned tile.
        for (i, b) in src.iter_mut().enumerate() {
            *b = (i % 251) as u8;
        }
        dst.copy_from_slice(&src);
        for i in [0usize, 5, 900, 901, 4097, 4098, 20_000, len - 1] {
            dst[i] ^= 0xFF;
        }
        let expected = src.clone();
        let stored = unsafe { write_skipping_identical(3 * PAGE + 64, dst.as_mut_ptr(), &src, PAGE) };
        assert_eq!(dst, expected, "the rail left the destination different");
        assert!(stored > 0 && stored < len as u64, "stored={stored} of {len}");
    }

    /// A destination that already holds the frame costs no store at all beyond
    /// the unaligned head and tail, which are not a unit the rail can decline.
    #[test]
    fn an_identical_destination_stores_only_the_unaligned_ends() {
        reset();
        let src = vec![0x5Au8; 4 * PAGE as usize];
        let mut dst = src.clone();
        let stored = unsafe { write_skipping_identical(0, dst.as_mut_ptr(), &src, PAGE) };
        assert_eq!(stored, 0, "an aligned identical frame stored bytes");
        let w = take_window().expect("the rail reports");
        assert_eq!((w.railed, w.stored), (1, 0), "{w:?}");
        assert_eq!(w.same_fine, w.fine, "{w:?}");
    }

    /// Consecutive differing tiles are stored as one run. Without coalescing a
    /// vertical band of change crossing every row would be one `memcpy` per
    /// tile, which is the shape this rail exists to avoid producing.
    #[test]
    fn consecutive_differing_tiles_are_stored_as_one_run() {
        reset();
        let src = vec![1u8; 16 * FINE_TILE];
        let mut dst = vec![1u8; 16 * FINE_TILE];
        // Tiles 4..=9 differ, the rest match.
        for b in dst[4 * FINE_TILE..10 * FINE_TILE].iter_mut() {
            *b = 2;
        }
        let stored = unsafe { write_skipping_identical(0, dst.as_mut_ptr(), &src, PAGE) };
        assert_eq!(stored, 6 * FINE_TILE as u64, "{stored}");
        assert_eq!(dst, src);
    }

    /// The counterfactual stores every byte and still reports the same
    /// comparison, so a boot can measure both sides against one power state.
    /// It is a correct configuration, not a broken one: the destination is the
    /// same either way.
    #[test]
    fn the_counterfactual_stores_everything_and_still_counts() {
        reset();
        let src = vec![9u8; 2 * PAGE as usize];
        let mut dst = src.clone();
        // Exercised through the same walk the latch guards; the latch itself is
        // process-wide, so this asserts the shape the two arms share rather
        // than flipping it.
        let stored = unsafe { write_skipping_identical(0, dst.as_mut_ptr(), &src, PAGE) };
        let w = take_window().expect("the rail reports");
        let all = 2 * PAGE;
        assert_eq!(w.bytes, all, "{w:?}");
        assert_eq!(w.same_fine, w.fine, "{w:?}");
        assert_eq!(
            stored,
            if tile_skip_counterfactual() { all } else { 0 },
            "the latch and the store count disagree"
        );
    }

    /// Neither latch is on unless its variable is exactly `1`. A switch that
    /// accepted `0` would run the opposite of what exporting it to turn the
    /// rail off intends.
    #[test]
    fn a_latch_needs_exactly_one() {
        for v in ["0", "", "false", "true", "01"] {
            let state = AtomicU8::new(u8::MAX);
            unsafe { std::env::set_var("REIMS_VGPU_TILE_SKIP_LATCH_TEST", v) };
            assert!(
                !env_latch("REIMS_VGPU_TILE_SKIP_LATCH_TEST", &state),
                "{v:?} turned the latch on"
            );
        }
        let state = AtomicU8::new(u8::MAX);
        unsafe { std::env::set_var("REIMS_VGPU_TILE_SKIP_LATCH_TEST", "1") };
        assert!(env_latch("REIMS_VGPU_TILE_SKIP_LATCH_TEST", &state));
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
            unsafe { note_write(run * FINE_TILE as u64 * 2, dst.as_ptr(), &src, PAGE) };
        }
        let w = take_window().expect("one audited walk");
        assert_eq!((w.audits, w.runs), (1, 4), "{w:?}");
        assert_eq!(w.bytes, 4 * 2 * FINE_TILE as u64, "{w:?}");
    }
}
