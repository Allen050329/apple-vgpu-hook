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

static CALLS: AtomicU64 = AtomicU64::new(0);
static AUDITS: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);
static PAGES: AtomicU64 = AtomicU64::new(0);
static SAME_PAGES: AtomicU64 = AtomicU64::new(0);
static FINE: AtomicU64 = AtomicU64::new(0);
static SAME_FINE: AtomicU64 = AtomicU64::new(0);

/// One window of the audit, as taken by the per-second census.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct LandRedundancyWindow {
    /// Writes that reached the compare, against `calls` offered.
    pub audits: u64,
    pub calls: u64,
    pub bytes: u64,
    pub pages: u64,
    pub same_pages: u64,
    pub fine: u64,
    pub same_fine: u64,
}

/// Take and clear the window. `None` when nothing was audited, so an idle
/// second costs no line.
pub fn take_window() -> Option<LandRedundancyWindow> {
    let audits = AUDITS.swap(0, Ordering::Relaxed);
    let w = LandRedundancyWindow {
        audits,
        calls: CALLS.swap(0, Ordering::Relaxed),
        bytes: BYTES.swap(0, Ordering::Relaxed),
        pages: PAGES.swap(0, Ordering::Relaxed),
        same_pages: SAME_PAGES.swap(0, Ordering::Relaxed),
        fine: FINE.swap(0, Ordering::Relaxed),
        same_fine: SAME_FINE.swap(0, Ordering::Relaxed),
    };
    (audits > 0).then_some(w)
}

/// Whether this write is the one in [`AUDIT_STRIDE`] that gets compared.
///
/// Counted per call to the walk rather than per run, so one landing is audited
/// whole and the fraction it reports is a fraction of a frame. A per-run stride
/// would sample scattered pieces of different landings and report their mean as
/// though it described one.
pub fn audit_due() -> bool {
    CALLS
        .fetch_add(1, Ordering::Relaxed)
        .is_multiple_of(AUDIT_STRIDE)
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
    AUDITS.fetch_add(1, Ordering::Relaxed);
    BYTES.fetch_add(src.len() as u64, Ordering::Relaxed);
    let tally = compare(map_off, src, dst, page_size);
    PAGES.fetch_add(tally.pages, Ordering::Relaxed);
    SAME_PAGES.fetch_add(tally.same_pages, Ordering::Relaxed);
    FINE.fetch_add(tally.fine, Ordering::Relaxed);
    SAME_FINE.fetch_add(tally.same_fine, Ordering::Relaxed);
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
}
