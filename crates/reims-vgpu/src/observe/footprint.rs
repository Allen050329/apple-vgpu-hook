//! Which guest page frames this device has written, for the whole boot.
//!
//! # The question this exists to answer
//!
//! `AGENTS.md` records twelve guest kernel panics whose victims are unrelated
//! subsystems — an apfs btree node, an ifnet function pointer, a HID driver's
//! heap element, a malloc small-zone free list — several of them filled with
//! `0xffffffffffffffff`, which is what opaque white BGRA looks like to a reader
//! who is not expecting pixels. The standing reading is that some write of this
//! device's landed at an address it did not own. That reading has never been
//! more than a shape match, and the document says so in as many words: "This is
//! a coincidence of shape, not an attribution."
//!
//! It stayed that way because nothing this device emitted could be compared
//! against what a panic actually names. XNU's `pmap_page_protect` panic prints a
//! **guest physical page number** (`pn=0x46b53b`), and this device knew its own
//! write destinations only as transient locals. "Did we write there?" was not a
//! hard question — it was an unasked one.
//!
//! This is the set that answers it: one bit per guest frame, set by every rail
//! that puts bytes into guest RAM, accumulated for the life of the boot and
//! dumped to the fail log as run-length spans. A panic's `pn` is then a lookup.
//!
//! # Read a hit and a miss differently
//!
//! They are not symmetric and a scorer must not treat them as such.
//!
//! A **miss** is strong: this device demonstrably never wrote that frame, so
//! whatever corrupted it was not these write rails. That exonerates.
//!
//! A **hit** is evidence proportional to the footprint's density. A boot that
//! wrote 34 000 distinct frames of a 16 GiB guest has touched 0.8 % of it, so an
//! unrelated victim lands inside by chance about one time in 125. That is
//! informative and it is not proof: the device is *supposed* to write those
//! frames, and one it legitimately owned a moment ago may be one the guest has
//! since freed. `pages` is on every summary line precisely so a reader can
//! compute that ratio rather than assume it.
//!
//! # Frames are 4 KiB regardless of the guest's page size
//!
//! [`FRAME_SHIFT`] is fixed rather than taking the device's `page_shift`, for
//! two reasons. It removes a `page_shift` parameter from every hook, several of
//! which sit at layers that have no business knowing the guest's page geometry
//! (`gpa_map`, the QEMU host shim). And 4 KiB is at least as fine as any guest
//! page this project supports — arm64's 16 KiB page marks four frames and stays
//! exact — so nothing is rounded up into a frame no byte reached.
//!
//! # Whole boot, not a window
//!
//! A panic reports the state of memory, not the time it was corrupted, and the
//! damaging write can predate it by minutes — the malloc free-list class is
//! discovered by a *later* allocation, not by the write that broke it. A
//! footprint that forgot would answer a question nobody is asking.
//!
//! # No silent cap
//!
//! The bit array is fixed and covers frames below [`MAX_FRAME`]. A mark above
//! that is counted in `dropped` and reported on every summary line, because a
//! footprint that quietly failed to record a write produces exactly the "miss"
//! that reads as an exoneration.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// Frames are 4 KiB. See the module header — this is deliberately not the
/// guest's page size.
pub const FRAME_SHIFT: u32 = 12;

/// Frames the set can represent: 16 Mi of them, so 64 GiB of guest-physical
/// space, against a rig that boots a 16 GiB guest — the PCI hole and any high
/// BAR aperture sit well inside it. Costs one bit each: 2 MiB, once per process.
pub const MAX_FRAME: u64 = 16 * 1024 * 1024;

const WORDS: usize = (MAX_FRAME / 64) as usize;

/// Emit the run-length dump no more often than this. The summary line is
/// per-census (once a second); the runs are the expensive part.
const DUMP_INTERVAL_MS: u64 = 30_000;

/// Runs per `guest_write_footprint_runs` line. Keeps a line to a width a human
/// can read while keeping the part count low enough that reassembly is obvious.
const RUNS_PER_LINE: usize = 48;

struct Footprint {
    bits: Box<[AtomicU64]>,
    /// Frames whose bit went 0 → 1. Maintained incrementally so the summary
    /// costs no scan; the dump recomputes runs by scanning, which is why the
    /// dump is rate-limited and the summary is not.
    pages: AtomicU64,
    /// Marks for a frame at or above [`MAX_FRAME`]. Reported, never swallowed.
    dropped: AtomicU64,
    last_dump_ms: AtomicU64,
    last_dump_pages: AtomicU64,
    dump_seq: AtomicUsize,
}

impl Footprint {
    fn new() -> Self {
        let mut bits = Vec::with_capacity(WORDS);
        bits.resize_with(WORDS, || AtomicU64::new(0));
        Self {
            bits: bits.into_boxed_slice(),
            pages: AtomicU64::new(0),
            dropped: AtomicU64::new(0),
            last_dump_ms: AtomicU64::new(0),
            last_dump_pages: AtomicU64::new(u64::MAX),
            dump_seq: AtomicUsize::new(0),
        }
    }

    fn mark(&self, frame: u64) {
        if frame >= MAX_FRAME {
            self.dropped.fetch_add(1, Ordering::Relaxed);
            return;
        }
        let word = (frame / 64) as usize;
        let bit = 1u64 << (frame % 64);
        // `fetch_or` hands back the previous word, so the 0 → 1 transition is
        // detectable without a second load and without a lock. Every rail that
        // writes guest RAM calls this, some of them per row, so marking has to
        // stay at one relaxed read-modify-write.
        let prev = self.bits[word].fetch_or(bit, Ordering::Relaxed);
        if prev & bit == 0 {
            self.pages.fetch_add(1, Ordering::Relaxed);
        }
        // Every guest write in the device funnels through here, which is what
        // makes this the right place for the check: a rail cannot reach guest
        // RAM without being asked whether the frame is still a surface's.
        if let Some((rword, rbit)) = retired_word(frame) {
            if rword.load(Ordering::Relaxed) & rbit != 0 {
                RETIRED.hits.fetch_add(1, Ordering::Relaxed);
                // Latched per frame AND capped in total. Per-frame alone is not
                // enough: a rail writing a whole 1080p surface into retired
                // pages has ~2 000 distinct frames to report, all of them the
                // same finding, and both the log and `first_sight`'s own set
                // would grow with the defect rather than with the information.
                //
                // The cap is on the *lines*, never on the counting — the census
                // keeps every hit — and the boundary line says the suppression
                // happened, because a log that quietly stopped reporting would
                // understate a defect exactly when it is worst.
                let logged = RETIRED.logged.fetch_add(1, Ordering::Relaxed);
                if logged < MAX_RETIRE_LINES {
                    if crate::observe::first_sight("write_after_retire", frame) {
                        crate::observe::fail(format!(
                            "write_after_retire frame={frame:#x} gpa={:#x} \
                             (the guest said these pages stopped being a \
                             surface's, and no mapping has adopted them since)",
                            frame << FRAME_SHIFT
                        ));
                    } else {
                        // A repeat of a frame already reported is not a new
                        // line, so it must not spend one of the budget.
                        RETIRED.logged.fetch_sub(1, Ordering::Relaxed);
                    }
                } else if logged == MAX_RETIRE_LINES {
                    crate::observe::fail(format!(
                        "write_after_retire suppressed after {MAX_RETIRE_LINES} \
                         distinct frames; the count continues in \
                         guest_write_footprint write_after_retire="
                    ));
                }
            }
        }
    }

    fn get(&self, frame: u64) -> bool {
        if frame >= MAX_FRAME {
            return false;
        }
        let word = (frame / 64) as usize;
        self.bits[word].load(Ordering::Relaxed) & (1u64 << (frame % 64)) != 0
    }

    /// Inclusive `[start, end]` frame runs, ascending.
    fn runs(&self) -> Vec<(u64, u64)> {
        let mut out: Vec<(u64, u64)> = Vec::new();
        for (w, cell) in self.bits.iter().enumerate() {
            let mut word = cell.load(Ordering::Relaxed);
            if word == 0 {
                continue;
            }
            let base = (w as u64) * 64;
            loop {
                let lo = word.trailing_zeros() as u64;
                // The run inside this word ends at the first clear bit at or
                // above `lo`. When the word is set through bit 63 there is no
                // such bit, and `trailing_zeros` of the resulting zero is 64 —
                // a length measured from bit 0, not from `lo`. Clamping to what
                // remains of the word is the difference between reporting
                // frames 60..=63 and claiming 60..=123, sixty frames this
                // device never wrote.
                let len = u64::from((!word >> lo).trailing_zeros()).min(64 - lo);
                let (s, e) = (base + lo, base + lo + len - 1);
                match out.last_mut() {
                    // Runs are found per 64-bit word, so a span crossing a word
                    // boundary arrives as two adjacent runs. Rejoin them, or the
                    // dump reports a fragmentation that is an artefact of the
                    // container rather than a fact about the device.
                    Some(last) if last.1 + 1 == s => last.1 = e,
                    _ => out.push((s, e)),
                }
                if lo + len >= 64 {
                    break;
                }
                word &= !(((1u64 << len) - 1) << lo);
                if word == 0 {
                    break;
                }
            }
        }
        out
    }

    #[cfg(test)]
    fn reset(&self) {
        for cell in self.bits.iter() {
            cell.store(0, Ordering::Relaxed);
        }
        self.pages.store(0, Ordering::Relaxed);
        self.dropped.store(0, Ordering::Relaxed);
        self.last_dump_ms.store(0, Ordering::Relaxed);
        self.last_dump_pages.store(u64::MAX, Ordering::Relaxed);
        self.dump_seq.store(0, Ordering::Relaxed);
    }
}

static FOOTPRINT: std::sync::LazyLock<Footprint> = std::sync::LazyLock::new(Footprint::new);

/// Frames this device has been told are no longer any surface's, and has not
/// since been told are a surface's again.
///
/// # Why this is not the drift guard again
///
/// The page-drift witness asks the *guest's page table* whether a mapping's
/// cached list still resolves the same way. That is the right question and it
/// has a blind spot with exactly the shape of the crash class: a surface the
/// guest has destroyed can keep its translations for as long as the address
/// space lives, so the walk agrees, the guard passes, and a write lands in
/// memory the guest handed to something else. `mapping_pages_verdict` cannot see
/// that, because nothing in the page table changed.
///
/// This asks a different question, out of this device's own bookkeeping: the
/// guest *told* us those pages stopped being a surface's, in a packet. A write
/// to one of them afterwards is write-after-teardown, and it is detectable on a
/// live boot with no panic, no guest crash and no post-mortem — which is what
/// every other instrument here has needed.
///
/// # Aliases are the false positive to avoid
///
/// Two mappings can name the same guest pages, so tearing one down does not
/// retire pages the other still holds. Frames still in any live mapping's list
/// are excluded at retire time; marking them would report the survivor's own
/// legitimate writes as a defect, and a detector whose first finding is noise
/// gets switched off.
struct Retired {
    bits: Box<[AtomicU64]>,
    frames: AtomicU64,
    /// Writes that landed in a retired frame. The finding.
    hits: AtomicU64,
    /// Retire events, and the total pages walked to answer them.
    ///
    /// Excluding an aliased page needs the *other* live mappings' lists, so a
    /// retire costs one pass over everything currently mapped. That runs on the
    /// drain worker, which `drain_duty` already shows at 0.93-0.99, and this
    /// project's standing rule is not to add work there on the assumption it is
    /// small. These two say how much it actually is: `scan_pages / scans` is the
    /// per-Unmap cost and `scans` is the rate. If the product turns out to
    /// matter, it is measured before it is optimised rather than after.
    scans: AtomicU64,
    scan_pages: AtomicU64,
    /// Distinct frames already reported by a fail line. See the cap at the
    /// emission site.
    logged: AtomicU64,
}

static RETIRED: std::sync::LazyLock<Retired> = std::sync::LazyLock::new(|| {
    let mut bits = Vec::with_capacity(WORDS);
    bits.resize_with(WORDS, || AtomicU64::new(0));
    Retired {
        bits: bits.into_boxed_slice(),
        frames: AtomicU64::new(0),
        hits: AtomicU64::new(0),
        scans: AtomicU64::new(0),
        scan_pages: AtomicU64::new(0),
        logged: AtomicU64::new(0),
    }
});

/// Distinct `write_after_retire` frames that get their own fail line before the
/// rest are suppressed.
///
/// This detector has never fired outside a unit test, so it is landing without a
/// live upper bound on how often it *could* fire. Sixty-four lines is enough to
/// see the shape of a real finding — which surfaces, which addresses — and few
/// enough that a detector that turns out to be wrong cannot take the log, the
/// census or `first_sight`'s set down with it.
const MAX_RETIRE_LINES: u64 = 64;

/// One Unmap's retire scan: how many pages it had to walk to exclude aliases.
pub fn note_retire_scan(pages_walked: u64) {
    RETIRED.scans.fetch_add(1, Ordering::Relaxed);
    RETIRED
        .scan_pages
        .fetch_add(pages_walked, Ordering::Relaxed);
}

fn retired_word(frame: u64) -> Option<(&'static AtomicU64, u64)> {
    if frame >= MAX_FRAME {
        return None;
    }
    Some((&RETIRED.bits[(frame / 64) as usize], 1u64 << (frame % 64)))
}

/// The guest said these pages stopped being a surface's. Call with the pages a
/// mapping is losing, already filtered to those no live mapping still holds.
pub fn note_pages_retired<I: IntoIterator<Item = u64>>(gpas: I, page_size: u64) {
    let step = page_size.max(1 << FRAME_SHIFT);
    for gpa in gpas {
        let first = gpa >> FRAME_SHIFT;
        let last = gpa.saturating_add(step - 1) >> FRAME_SHIFT;
        for frame in first..=last {
            if let Some((word, bit)) = retired_word(frame) {
                if word.fetch_or(bit, Ordering::Relaxed) & bit == 0 {
                    RETIRED.frames.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
    }
}

/// A mapping adopted these pages, so they are a surface's again.
///
/// Un-retiring on adoption is what keeps this from decaying into "every frame
/// the boot ever used": the guest recycles physical pages between surfaces
/// constantly, and a set that only ever grew would flag every one of those
/// perfectly ordinary reuses.
pub fn note_pages_authorized<I: IntoIterator<Item = u64>>(gpas: I, page_size: u64) {
    let step = page_size.max(1 << FRAME_SHIFT);
    for gpa in gpas {
        let first = gpa >> FRAME_SHIFT;
        let last = gpa.saturating_add(step - 1) >> FRAME_SHIFT;
        for frame in first..=last {
            if let Some((word, bit)) = retired_word(frame) {
                if word.fetch_and(!bit, Ordering::Relaxed) & bit != 0 {
                    RETIRED.frames.fetch_sub(1, Ordering::Relaxed);
                }
            }
        }
    }
}

/// `(frames currently retired, writes that landed in one)`.
pub fn retired_counts() -> (u64, u64) {
    (
        RETIRED.frames.load(Ordering::Relaxed),
        RETIRED.hits.load(Ordering::Relaxed),
    )
}

/// `(retire scans, pages walked by them)`.
pub fn retire_scan_counts() -> (u64, u64) {
    (
        RETIRED.scans.load(Ordering::Relaxed),
        RETIRED.scan_pages.load(Ordering::Relaxed),
    )
}

/// Record that `len` bytes starting at guest-physical `gpa` were written.
///
/// Every frame the byte range touches is marked, including a partial first and
/// last: the question is which frames this device put bytes into, not how many
/// bytes it put in each.
pub fn note_written_range(gpa: u64, len: u64) {
    if len == 0 {
        return;
    }
    let first = gpa >> FRAME_SHIFT;
    let last = gpa.saturating_add(len - 1) >> FRAME_SHIFT;
    let fp = &*FOOTPRINT;
    for frame in first..=last {
        fp.mark(frame);
    }
}

/// Record a written page for each GPA in `gpas`, each covering `page_size`
/// bytes. The scatter form: a mapping's page list is not contiguous, so a range
/// over its hull would claim frames between the pages that no write reached.
pub fn note_written_pages<I: IntoIterator<Item = u64>>(gpas: I, page_size: u64) {
    for gpa in gpas {
        note_written_range(gpa, page_size.max(1));
    }
}

/// One in this many guest writes is scanned for its payload shape.
///
/// Sampled rather than exhaustive because the scan is over the whole payload —
/// framebuffer-sized on the store rails, at the 28-111 stores/s `store_routes`
/// measures, on a drain worker `drain_duty` already shows at duty 0.93-0.99.
/// A deterministic counter rather than a random draw so two boots of the same
/// workload sample the same writes.
const PAYLOAD_SAMPLE_EVERY: u64 = 64;

/// The shortest all-`0xff` run that could have produced a report in the panic
/// census, and therefore the shortest one worth counting.
///
/// The two `kalloc` poison reports in `AGENTS.md` read "element modified after
/// free (off:0, val:0xffffffffffffffff, sz:6144)" and the same at `sz:256`: the
/// kernel found a **whole freed element** filled with `0xff` from offset 0. So a
/// write that could have produced the smaller of them put at least 256
/// consecutive `0xff` bytes into guest RAM. The number is that element size, not
/// a threshold picked to fit an observation.
const FF_RUN_MIN: usize = 256;

struct PayloadCensus {
    calls: AtomicU64,
    sampled: AtomicU64,
    all_ff: AtomicU64,
    all_zero: AtomicU64,
    bytes_sampled: AtomicU64,
    bytes_all_ff: AtomicU64,
    /// Sampled buffers carrying at least one run of [`FF_RUN_MIN`] `0xff` bytes.
    ff_run: AtomicU64,
    /// The longest such run seen. Exact at and above [`FF_RUN_MIN`]; shorter
    /// runs are deliberately not searched for, so a value below the threshold
    /// never appears.
    ff_run_max: AtomicU64,
}

static PAYLOAD: PayloadCensus = PayloadCensus {
    calls: AtomicU64::new(0),
    sampled: AtomicU64::new(0),
    all_ff: AtomicU64::new(0),
    all_zero: AtomicU64::new(0),
    bytes_sampled: AtomicU64::new(0),
    bytes_all_ff: AtomicU64::new(0),
    ff_run: AtomicU64::new(0),
    ff_run_max: AtomicU64::new(0),
};

/// The longest run of `0xff` bytes in `buf`, searched only for runs of at least
/// [`FF_RUN_MIN`]; `0` when there is none.
///
/// Probing every [`FF_RUN_MIN`]th byte is exact for the runs being looked for
/// and costs `len / 256` loads on a buffer that has none: any run of
/// `FF_RUN_MIN` consecutive bytes contains at least one index that is a multiple
/// of `FF_RUN_MIN`, so a run long enough to matter cannot hide between probes.
/// Only a probe that lands on `0xff` pays to expand.
fn longest_ff_run(buf: &[u8]) -> usize {
    let mut best = 0usize;
    // Exclusive end of the run last expanded. Probes stay on the fixed
    // `FF_RUN_MIN` grid — moving them to the end of a run would break the
    // alignment the correctness argument rests on — so this is what stops a
    // long run being re-expanded once per probe that lands inside it.
    let mut measured_end = 0usize;
    let mut i = 0usize;
    while i < buf.len() {
        if buf[i] == 0xFF && i >= measured_end {
            let mut lo = i;
            while lo > 0 && buf[lo - 1] == 0xFF {
                lo -= 1;
            }
            let mut hi = i + 1;
            while hi < buf.len() && buf[hi] == 0xFF {
                hi += 1;
            }
            if hi - lo >= FF_RUN_MIN {
                best = best.max(hi - lo);
            }
            measured_end = hi;
        }
        i += FF_RUN_MIN;
    }
    best
}

/// Record the *shape* of a guest write's payload, sampled.
///
/// # The assumption this exists to test
///
/// The panic census in `AGENTS.md` finds its victims filled with
/// `0xffffffffffffffff`, and the standing reading is that this is "almost
/// certainly a legitimate white frame landing at the wrong address — the defect
/// is *where*, not *what*, so do not go looking for a source of white".
///
/// That is a reasonable inference and it has never been measured. It is also
/// load-bearing: if this device writes all-`0xff` payloads constantly — a white
/// browser page is exactly that — then the payload tells a reader nothing, and
/// a victim full of `0xff` is no more likely to be ours than any other. If it
/// almost never does, the payload is a far sharper discriminator than the
/// footprint alone, which is only as strong as its density.
///
/// Both answers are worth having and neither is available today, so this counts
/// rather than concluding. `all_zero` is the control on the counter itself: a
/// scanner that reported everything as uniform would show both climbing
/// together, and a freshly allocated surface really is zero-filled, so the two
/// populations are known to be distinct in the guest.
///
/// # `all_ff` alone answers a question nobody asked
///
/// It requires the **whole** buffer to be `0xff`, and the rails here hand over
/// whole frames and whole source images. A white browser page has a menu bar, a
/// scrollbar and text in it, so a device rendering it faithfully writes
/// megabytes of white and `all_ff` still reads zero. The first live reading —
/// `all_ff=0` over 25 529 samples — was recorded with that caveat unstated, and
/// it cannot bear the weight it looked like it could.
///
/// [`FF_RUN_MIN`] is the predicate the panic census actually implies: a run long
/// enough to have filled the smaller of the two poisoned `kalloc` elements. Both
/// numbers are kept — `all_ff` is the strict form and stays comparable with the
/// boots already recorded — but `ff_run` is the one to read.
///
/// The uniform scans short-circuit on the first byte that differs, and the run
/// probe costs `len / 256` loads on a buffer with no long white span, so a
/// photograph pays almost nothing either way.
pub fn note_written_payload(buf: &[u8]) {
    if buf.is_empty() {
        return;
    }
    let n = PAYLOAD.calls.fetch_add(1, Ordering::Relaxed);
    if !n.is_multiple_of(PAYLOAD_SAMPLE_EVERY) {
        return;
    }
    PAYLOAD.sampled.fetch_add(1, Ordering::Relaxed);
    PAYLOAD
        .bytes_sampled
        .fetch_add(buf.len() as u64, Ordering::Relaxed);
    if buf.iter().all(|&b| b == 0xFF) {
        PAYLOAD.all_ff.fetch_add(1, Ordering::Relaxed);
        PAYLOAD
            .bytes_all_ff
            .fetch_add(buf.len() as u64, Ordering::Relaxed);
    } else if buf.iter().all(|&b| b == 0x00) {
        PAYLOAD.all_zero.fetch_add(1, Ordering::Relaxed);
    }
    let run = longest_ff_run(buf) as u64;
    if run > 0 {
        PAYLOAD.ff_run.fetch_add(1, Ordering::Relaxed);
        PAYLOAD.ff_run_max.fetch_max(run, Ordering::Relaxed);
    }
}

/// `(calls, sampled, all_ff, all_zero, bytes_sampled, bytes_all_ff)`.
pub fn payload_counts() -> (u64, u64, u64, u64, u64, u64) {
    (
        PAYLOAD.calls.load(Ordering::Relaxed),
        PAYLOAD.sampled.load(Ordering::Relaxed),
        PAYLOAD.all_ff.load(Ordering::Relaxed),
        PAYLOAD.all_zero.load(Ordering::Relaxed),
        PAYLOAD.bytes_sampled.load(Ordering::Relaxed),
        PAYLOAD.bytes_all_ff.load(Ordering::Relaxed),
    )
}

/// `(sampled buffers carrying a run of at least [`FF_RUN_MIN`], longest run)`.
pub fn ff_run_counts() -> (u64, u64) {
    (
        PAYLOAD.ff_run.load(Ordering::Relaxed),
        PAYLOAD.ff_run_max.load(Ordering::Relaxed),
    )
}

/// Whether this device has written the frame containing `gpa` at any point in
/// this boot. The scorer's question, exposed so a test can ask it directly.
pub fn wrote_gpa(gpa: u64) -> bool {
    FOOTPRINT.get(gpa >> FRAME_SHIFT)
}

/// Distinct frames written, and marks discarded for being at or above
/// [`MAX_FRAME`].
pub fn counts() -> (u64, u64) {
    (
        FOOTPRINT.pages.load(Ordering::Relaxed),
        FOOTPRINT.dropped.load(Ordering::Relaxed),
    )
}

/// The per-census summary line, and — at most every [`DUMP_INTERVAL_MS`], and
/// only when the set has grown since the last one — the run-length dump.
///
/// Returns the lines rather than emitting them, so the caller keeps the choice
/// of sink and a test can read them without a log fixture.
pub fn census_lines(now_ms: u64) -> Vec<String> {
    let fp = &*FOOTPRINT;
    let (pages, dropped) = counts();
    let kib = (pages << FRAME_SHIFT) / 1024;
    let (calls, sampled, all_ff, all_zero, bytes_sampled, bytes_all_ff) = payload_counts();
    // Levels, not per-interval: these are running totals for the boot, like the
    // frame count beside them and unlike `store_routes`. Summing them across
    // census lines multiplies by the cadence — the 100x error AGENTS.md records.
    let (retired_frames, retired_hits) = retired_counts();
    let (retire_scans, retire_scan_pages) = retire_scan_counts();
    let (ff_run, ff_run_max) = ff_run_counts();
    let mut out = vec![format!(
        "guest_write_footprint pages={pages} kib={kib} dropped={dropped} \
         frame_shift={FRAME_SHIFT} writes={calls} sampled={sampled} \
         all_ff={all_ff} all_zero={all_zero} samp_bytes={bytes_sampled} \
         ff_bytes={bytes_all_ff} ff_run={ff_run} ff_run_max={ff_run_max} \
         ff_run_min={FF_RUN_MIN} retired={retired_frames} \
         write_after_retire={retired_hits} retire_scans={retire_scans} \
         retire_scan_pages={retire_scan_pages} (levels, not per-interval)"
    )];

    let last_ms = fp.last_dump_ms.load(Ordering::Relaxed);
    let last_pages = fp.last_dump_pages.load(Ordering::Relaxed);
    let due = last_pages == u64::MAX || now_ms.saturating_sub(last_ms) >= DUMP_INTERVAL_MS;
    if !due || pages == last_pages {
        return out;
    }
    fp.last_dump_ms.store(now_ms, Ordering::Relaxed);
    fp.last_dump_pages.store(pages, Ordering::Relaxed);
    let seq = fp.dump_seq.fetch_add(1, Ordering::Relaxed);

    let runs = fp.runs();
    let parts = runs.len().div_ceil(RUNS_PER_LINE).max(1);
    for (i, chunk) in runs.chunks(RUNS_PER_LINE).enumerate() {
        let spans: Vec<String> = chunk
            .iter()
            .map(|(a, b)| format!("{a:#x}-{b:#x}"))
            .collect();
        out.push(format!(
            "guest_write_footprint_runs seq={seq} part={}/{parts} runs={} {}",
            i + 1,
            runs.len(),
            spans.join(" ")
        ));
    }
    out
}

/// One test at a time over the process-global set, cleared on entry.
///
/// The set is deliberately global — it is a property of the boot, not of any
/// object — so a test asserting "this rail marked frame X" is only meaningful
/// if no other test is marking concurrently. `--test-threads=1` is the project
/// rule and this does not depend on it: a global whose correctness rests on the
/// runner's flags breaks for whoever runs a single test by name.
#[cfg(test)]
static TEST_GUARD: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Take the set exclusively for a test and clear it. Held for the caller's scope.
#[cfg(test)]
pub(crate) fn exclusive_for_tests() -> std::sync::MutexGuard<'static, ()> {
    let g = TEST_GUARD.lock().unwrap_or_else(|e| e.into_inner());
    FOOTPRINT.reset();
    for cell in RETIRED.bits.iter() {
        cell.store(0, Ordering::Relaxed);
    }
    RETIRED.frames.store(0, Ordering::Relaxed);
    RETIRED.hits.store(0, Ordering::Relaxed);
    RETIRED.scans.store(0, Ordering::Relaxed);
    RETIRED.scan_pages.store(0, Ordering::Relaxed);
    RETIRED.logged.store(0, Ordering::Relaxed);
    for cell in [
        &PAYLOAD.calls,
        &PAYLOAD.sampled,
        &PAYLOAD.all_ff,
        &PAYLOAD.all_zero,
        &PAYLOAD.bytes_sampled,
        &PAYLOAD.bytes_all_ff,
        &PAYLOAD.ff_run,
        &PAYLOAD.ff_run_max,
    ] {
        cell.store(0, Ordering::Relaxed);
    }
    g
}

#[cfg(test)]
mod tests {
    use super::*;

    use super::exclusive_for_tests as fresh;

    #[test]
    fn a_written_range_marks_every_frame_it_touches_including_partial_ends() {
        let _g = fresh();
        // Starts mid-frame and ends mid-frame: 0x1800..=0x37ff is three frames,
        // not the one the start address names.
        note_written_range(0x1800, 0x2000);
        assert!(wrote_gpa(0x1000), "the partial first frame counts");
        assert!(wrote_gpa(0x2000));
        assert!(wrote_gpa(0x3000), "the partial last frame counts");
        assert!(!wrote_gpa(0x0));
        assert!(!wrote_gpa(0x4000));
        assert_eq!(counts().0, 3);
    }

    #[test]
    fn a_zero_length_write_marks_nothing() {
        let _g = fresh();
        // Without the guard, `first..=last` with last == first claims a frame no
        // byte reached — inflating the footprint, which weakens every later hit.
        note_written_range(0x9000, 0);
        assert_eq!(counts(), (0, 0));
        assert!(!wrote_gpa(0x9000));
    }

    #[test]
    fn marking_the_same_frame_twice_counts_it_once() {
        let _g = fresh();
        note_written_pages([0x5000, 0x5000], 0x1000);
        note_written_range(0x5fff, 1);
        assert_eq!(counts().0, 1, "distinct frames, not marks");
    }

    #[test]
    fn a_scatter_list_does_not_claim_the_frames_between_its_pages() {
        let _g = fresh();
        // A fragmented surface's page list. A range over its hull would mark
        // 0x2000..0x8000 as well, which is memory belonging to someone else —
        // and every one of those frames would then read as a hit.
        note_written_pages([0x1000, 0x9000], 0x1000);
        assert_eq!(counts().0, 2);
        assert!(!wrote_gpa(0x5000), "the gap is not ours to claim");
    }

    #[test]
    fn an_arm64_page_marks_its_four_frames_exactly() {
        let _g = fresh();
        note_written_pages([0x4000], 1 << 14);
        assert_eq!(counts().0, 4, "16 KiB is four 4 KiB frames");
        for f in 4..8u64 {
            assert!(wrote_gpa(f << 12));
        }
        assert!(!wrote_gpa(0x8000), "and not the fifth");
    }

    #[test]
    fn a_frame_past_the_end_of_the_set_is_dropped_loudly_and_never_reads_back_as_written() {
        let _g = fresh();
        let past = MAX_FRAME << FRAME_SHIFT;
        note_written_range(past, 0x1000);
        assert_eq!(counts(), (0, 1), "counted as dropped, not as a page");
        assert!(
            !wrote_gpa(past),
            "an unrecorded write must not answer `true`; a false hit invents evidence"
        );
        let line = &census_lines(0)[0];
        assert!(
            line.contains("dropped=1"),
            "a dropped mark has to reach the log, or the miss it causes reads as \
             an exoneration: {line}"
        );
    }

    #[test]
    fn runs_rejoin_across_word_boundaries_and_report_each_gap() {
        let _g = fresh();
        // 60..=70 crosses the 64-bit word boundary, which the per-word scan
        // finds as 60..=63 and 64..=70. Reported unjoined, the dump would claim
        // a fragmentation the device never produced.
        for frame in 60u64..=70 {
            note_written_range(frame << FRAME_SHIFT, 1);
        }
        note_written_range(200 << FRAME_SHIFT, 1);
        assert_eq!(FOOTPRINT.runs(), vec![(60, 70), (200, 200)]);
    }

    #[test]
    fn a_word_set_end_to_end_is_one_run_and_does_not_spin() {
        let _g = fresh();
        // `len == 64` is the case where the shift clearing the consumed bits
        // would be undefined. A wrong guard here hangs the census thread rather
        // than reporting a wrong number, which is the worse failure.
        note_written_range(0, 128 << FRAME_SHIFT);
        assert_eq!(FOOTPRINT.runs(), vec![(0, 127)]);
        assert_eq!(counts().0, 128);
    }

    #[test]
    fn a_run_ending_at_the_top_bit_of_a_word_terminates() {
        let _g = fresh();
        // Sets bits 32..=63 of word 0 and nothing in word 1: the scan must stop
        // at the end of the word rather than shifting past it.
        note_written_range(32 << FRAME_SHIFT, 32 << FRAME_SHIFT);
        assert_eq!(FOOTPRINT.runs(), vec![(32, 63)]);
    }

    #[test]
    fn the_payload_census_separates_uniform_white_from_uniform_black_and_from_content() {
        let _g = fresh();
        // The sampler takes call 0 and then every 64th, so drive it in blocks of
        // PAYLOAD_SAMPLE_EVERY and assert on what it sampled, not on what it saw.
        let white = vec![0xFFu8; 4096];
        let black = vec![0x00u8; 4096];
        let mut content = vec![0xFFu8; 4096];
        content[4095] = 0xFE; // uniform but for one byte: not white
        for buf in [&white, &black, &content] {
            for _ in 0..PAYLOAD_SAMPLE_EVERY {
                note_written_payload(buf);
            }
        }
        let (calls, sampled, all_ff, all_zero, samp_bytes, ff_bytes) = payload_counts();
        assert_eq!(calls, 3 * PAYLOAD_SAMPLE_EVERY);
        assert_eq!(sampled, 3, "one sample per block of {PAYLOAD_SAMPLE_EVERY}");
        assert_eq!((all_ff, all_zero), (1, 1));
        assert_eq!(samp_bytes, 3 * 4096);
        assert_eq!(ff_bytes, 4096, "only the white buffer's bytes");
    }

    #[test]
    fn a_payload_one_byte_short_of_white_is_not_white() {
        let _g = fresh();
        // The discriminator is worth nothing if it rounds. A frame the guest
        // will read as white-but-for-a-pixel is not the all-0xff fill the panic
        // census reports, and counting it as one would inflate the very number
        // this census exists to decide a hypothesis on.
        let mut nearly = vec![0xFFu8; 1024];
        nearly[0] = 0xFE;
        note_written_payload(&nearly);
        assert_eq!(payload_counts().2, 0);
    }

    /// The case `all_ff` is blind to, and the reason the run predicate exists.
    ///
    /// A white browser page has a menu bar, a scrollbar and text in it, so the
    /// frame this device writes is never uniform — and `all_ff` reads zero on
    /// exactly the workload the white-frame hypothesis is about. A run of
    /// [`FF_RUN_MIN`] is what the `kalloc` poison reports imply, and it is
    /// present in that frame by the megabyte.
    #[test]
    fn a_mostly_white_frame_scores_no_all_ff_and_a_long_ff_run() {
        let _g = fresh();
        let mut frame = vec![0xFFu8; 64 * 1024];
        // Chrome at the top and a scrollbar column: enough to break uniformity,
        // nowhere near enough to break up the white.
        for b in frame.iter_mut().take(1024) {
            *b = 0x20;
        }
        frame[40_000] = 0x00;
        note_written_payload(&frame);
        assert_eq!(
            payload_counts().2,
            0,
            "not uniform, so `all_ff` cannot see it — which is the defect in \
             reading `all_ff=0` as 'this device does not write white'"
        );
        let (ff_run, ff_run_max) = ff_run_counts();
        assert_eq!(ff_run, 1);
        assert_eq!(
            ff_run_max,
            (40_000 - 1024) as u64,
            "the longer of the two white spans the dark pixel splits: chrome to \
             it (38 976) beats it to the end (25 535)"
        );
    }

    /// The negative, and it has to be checked at the probe stride: a scan that
    /// steps 256 bytes must not report a run assembled from separate ones.
    #[test]
    fn short_ff_runs_and_content_score_nothing() {
        let _g = fresh();
        // Runs of 255 — one byte short — at every probe point, so a scan that
        // rounded up or joined across the gap would score them.
        let mut buf = vec![0x11u8; 64 * FF_RUN_MIN];
        for chunk in buf.chunks_mut(FF_RUN_MIN + 1) {
            let n = chunk.len().min(FF_RUN_MIN - 1);
            for b in chunk.iter_mut().take(n) {
                *b = 0xFF;
            }
        }
        assert_eq!(longest_ff_run(&buf), 0, "255 is not 256");
        note_written_payload(&buf);
        assert_eq!(ff_run_counts(), (0, 0));

        // And a photograph: no long uniform anything.
        let noise: Vec<u8> = (0..8192u32).map(|i| (i.wrapping_mul(37) % 251) as u8).collect();
        assert_eq!(longest_ff_run(&noise), 0);
    }

    /// Exactly at the threshold, and unaligned to the probe grid.
    ///
    /// The correctness argument for probing every [`FF_RUN_MIN`]th byte is that
    /// any run that long contains a multiple of [`FF_RUN_MIN`]. A run placed to
    /// straddle two probes with its start between them is where an off-by-one in
    /// that argument would show, so it is checked at every offset in a stride.
    #[test]
    fn a_threshold_length_run_is_found_at_every_alignment() {
        for off in 0..FF_RUN_MIN {
            let mut buf = vec![0x00u8; FF_RUN_MIN * 4];
            for b in buf.iter_mut().skip(off).take(FF_RUN_MIN) {
                *b = 0xFF;
            }
            assert_eq!(
                longest_ff_run(&buf),
                FF_RUN_MIN,
                "a {FF_RUN_MIN}-byte run starting at {off} must be found"
            );
        }
    }

    /// Two long runs in one buffer report the longer, and neither is double
    /// counted into a length that was never written.
    #[test]
    fn the_longest_of_several_runs_is_reported_and_none_are_joined() {
        let _g = fresh();
        let mut buf = vec![0x00u8; 4096];
        for b in buf.iter_mut().skip(100).take(300) {
            *b = 0xFF;
        }
        for b in buf.iter_mut().skip(1000).take(900) {
            *b = 0xFF;
        }
        assert_eq!(longest_ff_run(&buf), 900);
        note_written_payload(&buf);
        assert_eq!(ff_run_counts(), (1, 900), "one buffer, longest run 900");
    }

    #[test]
    fn an_empty_payload_is_not_counted_as_uniform() {
        let _g = fresh();
        // `[].iter().all(..)` is vacuously true for both tests, so an empty
        // buffer would score as white AND as black at once.
        note_written_payload(&[]);
        let (calls, sampled, all_ff, all_zero, _, _) = payload_counts();
        assert_eq!((calls, sampled, all_ff, all_zero), (0, 0, 0, 0));
    }

    #[test]
    fn a_write_into_a_retired_frame_is_counted_and_adoption_stops_it() {
        let _g = fresh();
        note_pages_retired([0x8000u64], 0x1000);
        assert_eq!(retired_counts(), (1, 0));

        // A write elsewhere is not a finding.
        note_written_range(0x9000, 0x1000);
        assert_eq!(retired_counts().1, 0);

        note_written_range(0x8000, 0x10);
        assert_eq!(retired_counts().1, 1, "the write into it is the finding");

        // Adoption puts the frame back in service. Without this the set only
        // grows, and the guest recycles physical pages between surfaces
        // constantly, so every ordinary reuse would read as a defect.
        note_pages_authorized([0x8000u64], 0x1000);
        assert_eq!(retired_counts().0, 0);
        note_written_range(0x8000, 0x10);
        assert_eq!(retired_counts().1, 1, "no new hit after adoption");
    }

    #[test]
    fn the_line_cap_bounds_the_log_without_bounding_the_count() {
        let _g = fresh();
        // A rail writing a whole 1080p surface into retired pages has ~2 000
        // distinct frames to report and every one is the same finding. This
        // detector has never fired on a live boot, so it lands without any
        // measured upper bound on how often it *could* — and an unverified
        // detector that can take the log down with it is worse than none.
        let n = MAX_RETIRE_LINES + 500;
        let frames: Vec<u64> = (0..n).map(|i| (i + 0x1_0000) << FRAME_SHIFT).collect();
        note_pages_retired(frames.iter().copied(), 1 << FRAME_SHIFT);
        for &gpa in &frames {
            note_written_range(gpa, 8);
        }
        assert_eq!(
            retired_counts().1,
            n,
            "every hit is counted; the cap is on lines, never on the census"
        );
        assert!(
            RETIRED.logged.load(Ordering::Relaxed) > MAX_RETIRE_LINES,
            "the counter must pass the cap so the boundary line fires exactly once"
        );
    }

    #[test]
    fn a_repeat_hit_on_a_reported_frame_does_not_spend_a_line_of_the_budget() {
        let _g = fresh();
        // Rewriting one retired frame every frame of a boot is one finding, not
        // thousands. If a repeat consumed budget, a single stuck surface would
        // exhaust the cap and suppress every *other* frame's line — losing the
        // spread, which is the part of this class that has always been the
        // diagnosis.
        note_pages_retired([0x30000u64], 1 << FRAME_SHIFT);
        for _ in 0..(MAX_RETIRE_LINES * 4) {
            note_written_range(0x30000, 8);
        }
        assert_eq!(
            RETIRED.logged.load(Ordering::Relaxed),
            1,
            "one distinct frame, one line spent"
        );
        assert_eq!(retired_counts().1, MAX_RETIRE_LINES * 4);
    }

    #[test]
    fn retiring_a_frame_twice_counts_it_once_and_adopting_an_unretired_one_is_a_no_op() {
        let _g = fresh();
        // Both directions of the counter have to be idempotent, or the level
        // drifts against the bits and `retired=` on the census stops meaning
        // "frames currently retired".
        note_pages_retired([0x2000u64, 0x2000], 0x1000);
        assert_eq!(retired_counts().0, 1);
        note_pages_authorized([0x7000u64], 0x1000);
        assert_eq!(retired_counts().0, 1, "adopting a live frame changes nothing");
        note_pages_authorized([0x2000u64, 0x2000], 0x1000);
        assert_eq!(retired_counts().0, 0);
    }

    #[test]
    fn a_guest_page_larger_than_a_frame_retires_all_of_its_frames() {
        let _g = fresh();
        // arm64. Retiring only the first frame of a 16 KiB page would leave
        // three quarters of every torn-down surface undetectable.
        note_pages_retired([0x4000u64], 1 << 14);
        assert_eq!(retired_counts().0, 4);
        note_written_range(0x4000 + 3 * 0x1000, 4);
        assert_eq!(retired_counts().1, 1);
    }

    #[test]
    fn the_dump_is_rate_limited_but_the_summary_is_not() {
        let _g = fresh();
        note_written_range(0x1000, 0x1000);
        let first = census_lines(0);
        assert!(
            first
                .iter()
                .any(|l| l.starts_with("guest_write_footprint_runs")),
            "the first census must carry a dump, or a panic in the first 30 s has \
             nothing to be scored against: {first:?}"
        );

        note_written_range(0x9000, 0x1000);
        let soon = census_lines(1_000);
        assert_eq!(soon.len(), 1, "summary only inside the interval: {soon:?}");
        assert!(soon[0].contains("pages=2"), "{}", soon[0]);

        let later = census_lines(DUMP_INTERVAL_MS);
        assert!(
            later.iter().any(|l| l.contains("0x9-0x9")),
            "the growth must appear once the interval elapses: {later:?}"
        );
    }

    #[test]
    fn a_dump_is_skipped_when_the_set_did_not_grow() {
        let _g = fresh();
        note_written_range(0x1000, 0x1000);
        let _ = census_lines(0);
        // The same frame again leaves the set unchanged, so re-emitting an
        // identical run list every 30 s would be pure log volume.
        note_written_range(0x1000, 0x1000);
        let idle = census_lines(10 * DUMP_INTERVAL_MS);
        assert_eq!(idle.len(), 1, "{idle:?}");
    }

    #[test]
    fn every_run_of_a_dump_is_reachable_from_the_part_lines_alone() {
        let _g = fresh();
        // More runs than fit on one line, so reassembly is what is under test.
        let n = RUNS_PER_LINE as u64 * 2 + 5;
        for i in 0..n {
            note_written_range((i * 4) << FRAME_SHIFT, 1);
        }
        let lines = census_lines(0);
        let parts: Vec<&String> = lines
            .iter()
            .filter(|l| l.starts_with("guest_write_footprint_runs"))
            .collect();
        assert!(parts.len() > 2, "expected several parts: {}", parts.len());
        let seen: usize = parts
            .iter()
            .map(|line| {
                line.split_whitespace()
                    .filter(|t| t.starts_with("0x") && t.contains('-'))
                    .count()
            })
            .sum();
        assert_eq!(
            seen, n as usize,
            "the chunks must sum to the whole set, or a scorer reassembling them \
             reports a smaller footprint than the device has"
        );
    }
}
