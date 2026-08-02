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
//! It is a counter, not a rail. On one write in
//! [`crate::runtime::land_redundancy::AUDIT_STRIDE`] it compares the
//! bytes about to be stored against the bytes already in the guest's pages and
//! reports how many matched, at two granularities, because the granularity is
//! what decides which rail is worth building:
//!
//! - **`page`** — the guest page. The unit a CPU-side skip would work in, and
//!   the unit the write witness above is page-exact in.
//! - **`fine`** — [`crate::runtime::land_redundancy::FINE_TILE`] bytes, 64 BGRA8
//!   texels. The unit a GPU-side
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
//! The sampling unit differs between the two legs. A mapping walk is a
//! landing, so its stride samples whole frames; a `write_span_multi` call is one
//! row, so the GVA leg samples rows scattered across many windows. For a
//! fraction over many samples both are unbiased; neither the GVA leg nor a
//! blend of the two can support a claim about a particular frame.
//!
//! # The stride is per leg, and it had to be — the shared one aliased
//!
//! Each leg keeps its own `STRIDE_TICK`, `CALLS` and `AUDITS`. That is not
//! tidiness. The first version shared one tick across every writer, and adding
//! the GVA hook — which fires **37 000 times a second** against the mapping
//! leg's ~290 — moved the mapping leg's own answer:
//!
//! ```text
//! run     hooks           mapping same_fine   mapping same_pages   landings audited
//! drag5   mapping only    med 90.75 %         med 51.78 %          4
//! drag6   + gva           med 99.60 %         med 97.85 %          5
//! ```
//!
//! Same stressor, same guest, comparable motion (1853 against 1655
//! repositions), `duty` 0.97 both, the same number of landings sampled — and a
//! nine-point move in the answer. A fixed stride over a stream one source
//! dominates 130:1 stops landing where it used to: it aliases onto whichever
//! phase of the other source's cycle the arithmetic happens to select. The
//! guest emits ~8.5 composites per frame and they are not alike — static layer
//! surfaces are near-totally redundant while the final composite is not — so
//! *which* of the 8.5 the stride picks is most of the answer.
//!
//! **Both readings are therefore biased samples of a mixed population, and the
//! per-leg stride fixes only half of it.** It stops one hook perturbing
//! another's sampling, which is what made the defect visible; it does not make
//! any of them a population estimate, because the mapping leg still mixes
//! surfaces whose redundancy differs by tens of points.
//!
//! Six driven runs at tile granularity, in the order taken:
//!
//! ```text
//! 86.1   89.9   91.8   90.75   99.60   78.20      (medians, per run)
//! ```
//!
//! The last is the first run with the corrected stride, and it is the *lowest*
//! of the six — so the fix removed a known bias without converging the answer.
//! **Quote a range: roughly 70 – 99 %, worst observed second 68 %.** That is
//! enough to justify the GPU pass, since even the worst second declines ~68 %
//! of both costs, and not enough to predict its saving to better than a factor.
//! Splitting the audit per surface is the first move for anyone who needs it
//! tighter.
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
//! # A mean cannot say whether the redundancy is concentrated
//!
//! Everything above is a fraction of tiles over a whole window, and two
//! populations that call for completely different builds produce the same one.
//! Seven wholly-unchanged surfaces beside one wholly-changed surface reads 87.5%
//! redundant; eight surfaces each 87.5% redundant reads 87.5% too. The first is
//! collected by hashing each landing and declining it entire — no tile bitmap,
//! no frame-sized shadow of the previous landing, no compaction — and the second
//! is not collected by anything short of the whole tile apparatus.
//!
//! [`crate::runtime::land_redundancy::Walk`] answers it. Each audited walk is bucketed by its own
//! `same_fine / fine`, and its bytes are charged to the bucket as well as to the
//! window, so the shape of the distribution is reported beside its mean:
//!
//! - `whole` — every whole tile in the walk matched. `whole_bytes` is what a
//!   landing-granular skip would decline.
//! - `over_90`, `over_50`, `under_50` — everything else, by decile of the
//!   fraction.
//!
//! Buckets sum to `audits` less any walk that compared no whole tile, which is
//! why they are reported beside it rather than instead of it.
//!
//! # It is spread, and that refutes the cheap build
//!
//! Two 15-second drags on one settled x86/PCI guest, 114 audited landings on
//! the mapping leg:
//!
//! ```text
//!         landings  whole  over_90  over_50  under_50   whole bytes   same_fine
//! drag1      61       10      49       2        0          16.4 %       97.60 %
//! drag2      53        2      32      19        0           3.8 %       92.61 %
//! ```
//!
//! **Almost no landing is wholly redundant, and no landing is under half.**
//! Declining whole landings — a hash, a four-byte readback, no tile bitmap and
//! no per-target shadow of the previous frame — would collect **4 – 16 %** of
//! the bytes. Tile compaction collects **93 – 98 %**. That is a factor of six to
//! twenty-four between the cheap build and the expensive one, so the cheap one
//! is not a first step towards the other; it is a different, much smaller thing.
//!
//! **`under_50` is single-valued on a driven boot, and it is not a dead field.**
//! `constant-fields.sh` reports a `key=value` that only ever takes one value,
//! and this one takes only `0` on every driven second measured so far. That
//! reading *is* the result — it says no landing is under half redundant — and
//! the bucket it would be traded for does not exist, because the four partition
//! the range. Deleting it would delete the finding. A window with no motion at
//! all is where a non-zero could come from, and nobody has driven one that way.
//!
//! The `under_50` column reading **zero** across all 114 is the second finding.
//! Every landing this rail carries is at least half already-correct and most are
//! over nine tenths, so a tile pass would collect near-uniformly rather than
//! well on some frames and not at all on others. There is no worst-case landing
//! to design around.
//!
//! Both are shapes of a distribution rather than point estimates: 114 landings
//! out of ~7800 is a 1.5 % sample, and the window mean still moved 97.6 → 92.6
//! between two runs minutes apart. The *shape* did not — `whole` is rare and
//! `under_50` is empty in both.
//!
//! The GVA leg's shape is the opposite and does not transfer, because its walk
//! is a row rather than a landing. It reads `whole` 122-579 against `under_50`
//! 0-449 and nothing in between, and its rows are ~250 bytes — one fine tile —
//! so "the walk" and "the tile" are the same unit there and the buckets can only
//! say matched or not. That leg is ~0.4 % of the writeback's bytes and is not
//! what any of this is priced on.
//!
//! `whole` is at tile granularity, so a walk whose only differing bytes are in a
//! partial chunk at one end lands in it. A landing-granular rail would have to
//! compare those bytes too; on the frame-sized, tile-aligned landings this leg
//! carries there are none, and `whole_bytes` is an upper bound rather than a
//! promise.
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
//! of [`crate::runtime::land_redundancy::FINE_TILE`] and both are aligned in the
//! same space.

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

/// Free-running per leg, and deliberately **not** cleared by [`take_window`].
///
/// Two independent reasons, both of them measured defects rather than taste.
/// The stride is a property of the write stream and not of the reporting
/// window: a counter the census zeroed would restart it every second, making
/// the first write after each boundary due — all four landings on an idle
/// desktop, and a rate that moves with the load on a driven one. And it is per
/// leg because one shared tick let the 37 000-a-second GVA writer alias the
/// ~290-a-second mapping writer's sample onto a different set of surfaces,
/// moving that leg's reported redundancy nine points with no change to the
/// workload. See the module doc.
static STRIDE_TICK: [AtomicU64; LEGS] = [const { AtomicU64::new(0) }; LEGS];
static CALLS: [AtomicU64; LEGS] = [const { AtomicU64::new(0) }; LEGS];
static AUDITS: [AtomicU64; LEGS] = [const { AtomicU64::new(0) }; LEGS];
static RUNS: [AtomicU64; LEGS] = [const { AtomicU64::new(0) }; LEGS];
static BYTES: [AtomicU64; LEGS] = [const { AtomicU64::new(0) }; LEGS];
static PAGES: [AtomicU64; LEGS] = [const { AtomicU64::new(0) }; LEGS];
static SAME_PAGES: [AtomicU64; LEGS] = [const { AtomicU64::new(0) }; LEGS];
static FINE: [AtomicU64; LEGS] = [const { AtomicU64::new(0) }; LEGS];
static SAME_FINE: [AtomicU64; LEGS] = [const { AtomicU64::new(0) }; LEGS];
static BUCKET_WALKS: [[AtomicU64; LEGS]; BUCKETS] =
    [const { [const { AtomicU64::new(0) }; LEGS] }; BUCKETS];
static BUCKET_BYTES: [[AtomicU64; LEGS]; BUCKETS] =
    [const { [const { AtomicU64::new(0) }; LEGS] }; BUCKETS];

/// How redundant one whole audited walk turned out to be.
///
/// The window's mean cannot distinguish a few wholly-unchanged surfaces from
/// every surface being mostly unchanged, and those call for different rails —
/// see the module doc. A walk is placed by its own `same_fine / fine`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Bucket {
    /// Every whole tile in the walk matched: a landing-granular skip collects it.
    Whole = 0,
    Over90 = 1,
    Over50 = 2,
    Under50 = 3,
}

const BUCKETS: usize = 4;

impl Bucket {
    const ALL: [Self; BUCKETS] = [Self::Whole, Self::Over90, Self::Over50, Self::Under50];

    /// Which bucket a walk that compared `fine` whole tiles, `same` of them
    /// matching, belongs to.
    ///
    /// Compared as integers so no walk's placement depends on a rounding: a
    /// walk is `Over90` when nine tenths of its tiles matched exactly, and the
    /// `Whole` test is equality rather than a fraction of 1.0.
    fn of(same: u64, fine: u64) -> Option<Self> {
        if fine == 0 {
            return None;
        }
        Some(if same == fine {
            Self::Whole
        } else if same * 10 >= fine * 9 {
            Self::Over90
        } else if same * 2 >= fine {
            Self::Over50
        } else {
            Self::Under50
        })
    }

    fn label(self) -> &'static str {
        match self {
            Self::Whole => "whole",
            Self::Over90 => "over_90",
            Self::Over50 => "over_50",
            Self::Under50 => "under_50",
        }
    }
}

/// One leg's window of the audit, as taken by the per-second census.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct LandRedundancyWindow {
    /// Walks audited against walks offered, **for this leg only**. The stride
    /// is per leg because a shared one let a 37 000-a-second writer move a
    /// 290-a-second writer's answer by nine points; see the module doc.
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
    /// Walks in each [`Bucket`], and the bytes they carried, in `Bucket::ALL`
    /// order. They sum to `audits` less the walks that compared no whole tile.
    pub bucket_walks: [u64; BUCKETS],
    pub bucket_bytes: [u64; BUCKETS],
}

impl LandRedundancyWindow {
    /// The buckets with their labels, so a caller emitting them does not
    /// re-derive the order the arrays are in.
    pub fn buckets(&self) -> impl Iterator<Item = (&'static str, u64, u64)> + '_ {
        Bucket::ALL
            .into_iter()
            .map(|b| (b.label(), self.bucket_walks[b as usize], self.bucket_bytes[b as usize]))
    }
}

/// Take and clear the window, one entry per leg that compared anything.
///
/// A leg with no ranges is left out rather than emitted as zeros: a line of
/// zeros reads as "nothing was redundant" when it means "nothing was measured",
/// and the two call for opposite conclusions.
pub fn take_window() -> Vec<(Leg, LandRedundancyWindow)> {
    Leg::ALL
        .into_iter()
        .filter_map(|leg| {
            let i = leg as usize;
            let w = LandRedundancyWindow {
                audits: AUDITS[i].swap(0, Ordering::Relaxed),
                calls: CALLS[i].swap(0, Ordering::Relaxed),
                runs: RUNS[i].swap(0, Ordering::Relaxed),
                bytes: BYTES[i].swap(0, Ordering::Relaxed),
                pages: PAGES[i].swap(0, Ordering::Relaxed),
                same_pages: SAME_PAGES[i].swap(0, Ordering::Relaxed),
                fine: FINE[i].swap(0, Ordering::Relaxed),
                same_fine: SAME_FINE[i].swap(0, Ordering::Relaxed),
                bucket_walks: Bucket::ALL.map(|b| BUCKET_WALKS[b as usize][i].swap(0, Ordering::Relaxed)),
                bucket_bytes: Bucket::ALL.map(|b| BUCKET_BYTES[b as usize][i].swap(0, Ordering::Relaxed)),
            };
            (w.runs > 0).then_some((leg, w))
        })
        .collect()
}

/// Begin the one walk in [`AUDIT_STRIDE`] that gets compared, or `None`.
///
/// The stride is counted per walk rather than per run, so one landing is audited
/// whole and the fraction it reports is a fraction of a frame. A per-run stride
/// would sample scattered pieces of different landings and report their mean as
/// though it described one.
///
/// The returned handle is what makes "one landing" a value rather than a
/// convention: every run of the walk is charged to it, and it commits to the
/// window when it drops. That is what lets the walk be bucketed by its *own*
/// redundancy — a `bool` and a set of free functions could only ever add to the
/// window's mean, which is the distinction the buckets exist to draw.
pub fn begin_audit(leg: Leg) -> Option<Walk> {
    let i = leg as usize;
    CALLS[i].fetch_add(1, Ordering::Relaxed);
    let due = STRIDE_TICK[i]
        .fetch_add(1, Ordering::Relaxed)
        .is_multiple_of(AUDIT_STRIDE);
    if !due {
        return None;
    }
    AUDITS[i].fetch_add(1, Ordering::Relaxed);
    Some(Walk {
        leg,
        runs: 0,
        bytes: 0,
        tally: Tally::default(),
    })
}

/// One audited walk in progress: the runs of a single landing, tallied together.
///
/// Commits on drop rather than through a `commit` call, so a caller that returns
/// early — every one of the four hooked walks has a refusal path that does —
/// contributes what it compared instead of silently dropping it.
#[derive(Debug)]
pub struct Walk {
    leg: Leg,
    runs: u64,
    bytes: u64,
    tally: Tally,
}

impl Walk {
    /// Compare `src` against the `src.len()` bytes at `dst` and charge the match
    /// to this walk.
    ///
    /// `map_off` is where the write lands in mapping-linear space, and it is
    /// what the page chunks are aligned to — not the pointer, whose alignment
    /// says nothing about which guest page a byte belongs to. `page_size` comes
    /// from the caller because guest page geometry is never assumed here.
    ///
    /// # Safety
    ///
    /// `dst` must be readable for `src.len()` bytes. Every caller is about to
    /// write exactly that range through the same pointer.
    pub unsafe fn note_write(&mut self, map_off: u64, dst: *const u8, src: &[u8], page_size: u64) {
        if src.is_empty() {
            return;
        }
        // SAFETY: the caller guarantees `dst` is readable for `src.len()`.
        let dst = unsafe { std::slice::from_raw_parts(dst, src.len()) };
        self.runs += 1;
        self.bytes += src.len() as u64;
        let t = compare(map_off, src, dst, page_size);
        self.tally.pages += t.pages;
        self.tally.same_pages += t.same_pages;
        self.tally.fine += t.fine;
        self.tally.same_fine += t.same_fine;
    }
}

impl Drop for Walk {
    fn drop(&mut self) {
        let i = self.leg as usize;
        RUNS[i].fetch_add(self.runs, Ordering::Relaxed);
        BYTES[i].fetch_add(self.bytes, Ordering::Relaxed);
        PAGES[i].fetch_add(self.tally.pages, Ordering::Relaxed);
        SAME_PAGES[i].fetch_add(self.tally.same_pages, Ordering::Relaxed);
        FINE[i].fetch_add(self.tally.fine, Ordering::Relaxed);
        SAME_FINE[i].fetch_add(self.tally.same_fine, Ordering::Relaxed);
        if let Some(b) = Bucket::of(self.tally.same_fine, self.tally.fine) {
            BUCKET_WALKS[b as usize][i].fetch_add(1, Ordering::Relaxed);
            BUCKET_BYTES[b as usize][i].fetch_add(self.bytes, Ordering::Relaxed);
        }
    }
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

    /// A walk that is due whatever the stride says, so a test can take several
    /// in a row without arranging 64 undue calls between them. Only
    /// [`begin_audit`] is the stride, and the tests that are about the stride
    /// call it directly.
    fn forced_walk(leg: Leg) -> Walk {
        AUDITS[leg as usize].fetch_add(1, Ordering::Relaxed);
        Walk {
            leg,
            runs: 0,
            bytes: 0,
            tally: Tally::default(),
        }
    }

    /// Compare one range as a whole walk, the way a caller with one run does.
    ///
    /// # Safety
    ///
    /// Same contract as [`Walk::note_write`].
    unsafe fn note_one(leg: Leg, map_off: u64, dst: *const u8, src: &[u8], page_size: u64) {
        let mut walk = forced_walk(leg);
        unsafe { walk.note_write(map_off, dst, src, page_size) };
    }

    fn reset() {
        let _ = take_window();
        for i in 0..LEGS {
            CALLS[i].store(0, Ordering::Relaxed);
            AUDITS[i].store(0, Ordering::Relaxed);
            STRIDE_TICK[i].store(0, Ordering::Relaxed);
        }
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
        unsafe { note_one(Leg::Mapping, 0, dst.as_ptr(), &src, PAGE) };
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
        unsafe { note_one(Leg::Mapping, 0, dst.as_ptr(), &src, PAGE) };
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
        unsafe { note_one(Leg::Mapping, PAGE + 1024, dst.as_ptr(), &src, PAGE) };
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
        unsafe { note_one(Leg::Mapping, PAGE - 32, dst.as_ptr(), &src, PAGE) };
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
        unsafe { note_one(Leg::Mapping, 0, dst.as_ptr(), &src, PAGE) };
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
        unsafe { note_one(Leg::Mapping, 0, dst.as_ptr(), &src, FINE_TILE as u64 + 1) };
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
        unsafe { note_one(Leg::Mapping, 0, same.as_ptr(), &src, PAGE) };
        unsafe { note_one(Leg::Gva, 0, differs.as_ptr(), &src, PAGE) };
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
        unsafe { note_one(Leg::Gva, 0, dst.as_ptr(), &src, PAGE) };
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
            due += u64::from(begin_audit(Leg::Mapping).is_some());
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
        let due = || begin_audit(Leg::Mapping).is_some();
        assert!(due(), "the stride's first tick is due");
        for _ in 1..(AUDIT_STRIDE / 2) {
            assert!(!due());
        }
        let _ = take_window();
        for _ in (AUDIT_STRIDE / 2)..AUDIT_STRIDE {
            assert!(!due(), "the window boundary made a write due");
        }
        assert!(due(), "the stride's own tick did not come due");
    }

    /// `audits` counts walks and `runs` counts the ranges inside them. A
    /// fragmented landing is hundreds of runs of one frame, so reporting one
    /// number for both would read as hundreds of frames sampled a second.
    #[test]
    fn walks_and_the_runs_inside_them_are_counted_apart() {
        reset();
        let mut walk = begin_audit(Leg::Mapping).expect("due");
        let src = [3u8; FINE_TILE * 2];
        let dst = src;
        for run in 0..4u64 {
            unsafe { walk.note_write(run * FINE_TILE as u64 * 2, dst.as_ptr(), &src, PAGE) };
        }
        drop(walk);
        let w = only_leg().expect("one audited walk");
        assert_eq!((w.audits, w.runs), (1, 4), "{w:?}");
        assert_eq!(w.bytes, 4 * 2 * FINE_TILE as u64, "{w:?}");
    }

    /// A walk's runs are bucketed together, by the walk's own fraction rather
    /// than each run's. A landing is hundreds of runs, and a rail that declines
    /// a whole landing needs every one of them to have matched.
    #[test]
    fn a_walk_is_bucketed_by_its_own_fraction_not_its_runs() {
        reset();
        let src = [0u8; FINE_TILE];
        let same = src;
        let differs = [1u8; FINE_TILE];
        let mut walk = begin_audit(Leg::Mapping).expect("due");
        for run in 0..9u64 {
            unsafe { walk.note_write(run * FINE_TILE as u64, same.as_ptr(), &src, PAGE) };
        }
        unsafe { walk.note_write(9 * FINE_TILE as u64, differs.as_ptr(), &src, PAGE) };
        drop(walk);
        let w = only_leg().expect("one audited walk");
        // Nine of ten tiles matched: one walk in `over_90`, and none in `whole`
        // even though nine of its runs were wholly redundant on their own.
        assert_eq!(w.same_fine, 9, "{w:?}");
        assert_eq!(w.bucket_walks, [0, 1, 0, 0], "{w:?}");
        assert_eq!(w.bucket_bytes, [0, 10 * FINE_TILE as u64, 0, 0], "{w:?}");
    }

    /// The buckets separate a distribution the window's mean cannot. Two walks
    /// wholly redundant and two wholly changed reads the same `same_fine` as
    /// four walks half redundant, and only the first is collected by declining
    /// whole landings — which is the build the buckets exist to choose.
    #[test]
    fn the_buckets_separate_concentrated_from_spread() {
        let tiles = 4usize;
        let src = vec![0u8; FINE_TILE * tiles];
        let mut half = src.clone();
        half[FINE_TILE * 2..].fill(1);
        let differs = vec![1u8; FINE_TILE * tiles];

        let concentrated = {
            reset();
            for dst in [&src, &src, &differs, &differs] {
                unsafe { note_one(Leg::Mapping, 0, dst.as_ptr(), &src, PAGE) };
            }
            only_leg().expect("four walks")
        };
        let spread = {
            reset();
            for _ in 0..4 {
                unsafe { note_one(Leg::Mapping, 0, half.as_ptr(), &src, PAGE) };
            }
            only_leg().expect("four walks")
        };

        assert_eq!(
            (concentrated.same_fine, concentrated.fine),
            (spread.same_fine, spread.fine),
            "the mean cannot tell them apart, which is the point",
        );
        assert_eq!(concentrated.bucket_walks, [2, 0, 0, 2], "{concentrated:?}");
        assert_eq!(spread.bucket_walks, [0, 0, 4, 0], "{spread:?}");
        // Only the concentrated shape offers bytes a landing-granular skip can
        // decline, and it is exactly the `whole` walks' bytes.
        assert_eq!(
            concentrated.bucket_bytes[0],
            2 * (FINE_TILE * tiles) as u64,
            "{concentrated:?}",
        );
        assert_eq!(spread.bucket_bytes[0], 0, "{spread:?}");
    }

    /// A walk that compared no whole tile is counted in `audits` and bucketed
    /// nowhere. Charging it to `under_50` would report a run too short to align
    /// a tile as a landing that changed, which is the opposite of what it says.
    #[test]
    fn a_walk_that_compared_no_whole_tile_is_bucketed_nowhere() {
        reset();
        let src = [1u8; 64];
        let dst = [2u8; 64];
        unsafe { note_one(Leg::Mapping, PAGE - 32, dst.as_ptr(), &src, PAGE) };
        let w = only_leg().expect("one audit");
        assert_eq!((w.audits, w.fine), (1, 0), "{w:?}");
        assert_eq!(w.bucket_walks, [0; BUCKETS], "{w:?}");
    }

    /// The boundaries are inclusive at the tenth and the half, and `whole` is
    /// equality rather than a rounded fraction — 999 of 1000 tiles is `over_90`
    /// and not `whole`, so a walk one tile short of skippable never reads as
    /// skippable.
    #[test]
    fn the_bucket_boundaries_are_exact() {
        assert_eq!(Bucket::of(0, 0), None);
        assert_eq!(Bucket::of(1000, 1000), Some(Bucket::Whole));
        assert_eq!(Bucket::of(999, 1000), Some(Bucket::Over90));
        assert_eq!(Bucket::of(900, 1000), Some(Bucket::Over90));
        assert_eq!(Bucket::of(899, 1000), Some(Bucket::Over50));
        assert_eq!(Bucket::of(500, 1000), Some(Bucket::Over50));
        assert_eq!(Bucket::of(499, 1000), Some(Bucket::Under50));
        assert_eq!(Bucket::of(0, 1000), Some(Bucket::Under50));
    }
}
