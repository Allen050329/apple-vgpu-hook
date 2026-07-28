//! Always-on per-branch census for sampled-texture resolution.
//!
//! `resolve_sampled_source` / `load_linear_from_host_caches` resolve a
//! sampled bind through ~10 distinct branches; only some carry a content
//! identity that lets the runtime skip the per-draw copy+swizzle and the
//! engine skip hash+memcmp. This census counts every resolution by branch
//! (count + RGBA bytes actually materialized on the CPU) so one boot log
//! shows which branch dominates per-draw `setup_tex` cost and where the
//! identity rail must extend next. Measure-only — never gates behavior.
//!
//! Cumulative `sampled_branch_census` lines land in `/tmp/reims-vgpu-fail.log`
//! (always-on `observe::off` sink) every [`EMIT_EVERY`] resolutions; the
//! last line of a boot is the whole-boot census.

use std::sync::atomic::{AtomicU64, Ordering};

/// Which return path of the sampled-load resolver produced the bind.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Branch {
    /// Type-5 serialized view materialized from guest pages.
    Type5View,
    /// Resident GPU target bound directly (no CPU bytes).
    Resident,
    /// Type-11 host surface cache hit (copy+swizzle).
    T11Cache,
    /// Type-11 guest-page load.
    T11Guest,
    /// Deferred-Store GVA window: resident GPU target bound directly
    /// (no flush, no CPU bytes).
    GvaResident,
    /// Linear GVA-cache memo reuse (Arc clone, no copy).
    GvaMemo,
    /// Linear GVA-cache hit, copy+swizzle (memoized after).
    GvaCopy,
    /// Linear texture-ref cache hit with generation.
    RefCache,
    /// Linear guest-page load, tight-stride memo MISS: full re-read + fresh
    /// allocation + per-row convert + memo store (`load_linear_guest_memoized`).
    /// The Safari-scroll / video hot branch — content changes each frame so the
    /// memcmp fails and this pays alloc+convert per bind.
    LinGuest,
    /// Linear guest-page load via the non-memo FALLBACK loader
    /// (`load_linear_texture_rgba_host`, reached only when the tight-stride memo
    /// loader declines: padded/mismatched stride). In-place-optimal then a
    /// `.to_vec()` slice. Split from [`Branch::LinGuest`] so a boot shows which
    /// sub-path the bulk of `lin_guest` binds actually take.
    LinGuestFallback,
    /// Linear guest-page memo reuse (native re-read + byte-exact match; no
    /// conversion, no new allocation).
    LinMemo,
    /// Type-5 view memo reuse (native window re-read + byte-exact match; no
    /// conversion, no new allocation).
    T5Memo,
    /// Zero-copy guest gather: GPU reads imported guest RAM in the draw CB
    /// (no CPU bytes at all).
    LinZeroCopy,
    /// Type-11 mapping-backed zero-copy guest gather (GPU reads imported
    /// guest RAM; no CPU bytes).
    T11ZeroCopy,
    /// Type-5 serialized-view zero-copy guest gather — the video plane rail
    /// (R8/Rg8/BGRA8/RGBA8 plane imported directly; no CPU read+upload).
    T5ZeroCopy,
    /// Any-size texture-ref cache fallback.
    TexrefAny,
    /// Tail static linear/view loader with descriptor-recovered geometry.
    StaticTail,
    /// Type-11 mapping-backed memo reuse (native BGRA re-read + byte-exact
    /// match; no convert, no new allocation, engine skips re-hash+upload via
    /// the returned content identity). The dock-magnification hot branch.
    T11Memo,
}

const N: usize = 18;

impl Branch {
    const fn idx(self) -> usize {
        match self {
            Branch::Type5View => 0,
            Branch::Resident => 1,
            Branch::T11Cache => 2,
            Branch::T11Guest => 3,
            Branch::GvaResident => 4,
            Branch::GvaMemo => 5,
            Branch::GvaCopy => 6,
            Branch::RefCache => 7,
            Branch::LinGuest => 8,
            Branch::LinMemo => 9,
            Branch::T5Memo => 10,
            Branch::LinZeroCopy => 11,
            Branch::T11ZeroCopy => 12,
            Branch::TexrefAny => 13,
            Branch::StaticTail => 14,
            Branch::LinGuestFallback => 15,
            Branch::T5ZeroCopy => 16,
            Branch::T11Memo => 17,
        }
    }
}

const NAMES: [&str; N] = [
    "t5_view",
    "resident",
    "t11_cache",
    "t11_guest",
    "gva_resident",
    "gva_memo",
    "gva_copy",
    "ref_cache",
    "lin_guest",
    "lin_memo",
    "t5_memo",
    "lin_zc",
    "t11_zc",
    "texref_any",
    "static_tail",
    "lin_guest_fb",
    "t5_zc",
    "t11_memo",
];

static COUNTS: [AtomicU64; N] = [const { AtomicU64::new(0) }; N];
static BYTES: [AtomicU64; N] = [const { AtomicU64::new(0) }; N];
static TOTAL: AtomicU64 = AtomicU64::new(0);
/// Resolve wall-clock charged to each branch. See [`note_resolve_us`].
static MICROS: [AtomicU64; N] = [const { AtomicU64::new(0) }; N];
/// Resolves that ended without any branch naming itself, and their time.
static UNNAMED: AtomicU64 = AtomicU64::new(0);
static UNNAMED_US: AtomicU64 = AtomicU64::new(0);

thread_local! {
    /// Branch noted by the resolve currently on this thread's stack.
    static PENDING: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

/// One cumulative census line per this many sampled resolutions.
const EMIT_EVERY: u64 = 256;

/// Record one sampled-bind resolution. `copied_bytes` is the RGBA byte count
/// materialized on the CPU for this bind (0 for resident/memo reuse paths).
pub fn note(branch: Branch, copied_bytes: usize) {
    let i = branch.idx();
    COUNTS[i].fetch_add(1, Ordering::Relaxed);
    BYTES[i].fetch_add(copied_bytes as u64, Ordering::Relaxed);
    PENDING.with(|p| p.set(Some(i)));
    let total = TOTAL.fetch_add(1, Ordering::Relaxed) + 1;
    if total.is_multiple_of(EMIT_EVERY) {
        crate::observe::off(format_line(&snapshot()));
    }
}

/// Charge one bind's whole resolve wall-clock to the branch that resolved it.
///
/// The count/bytes columns say which branch ran and how many bytes it copied;
/// neither says what it cost. A mean over all binds cannot answer that either,
/// because these branches differ by three orders of magnitude in bytes — the
/// byte-heavy misses are a per-cent of the count and could be most of the time,
/// or none of it, and the existing line reads identically both ways.
///
/// [`note`] runs at each branch terminus and the caller's resolve timer stops
/// just after, so the branch is known by the time the total is. Attributing here
/// costs one thread-local instead of a timer at each of the twenty termini, and
/// it cannot disagree with the counts because it charges the same event.
///
/// A resolve that named no branch is charged to `unnamed` rather than to
/// whichever branch ran last — otherwise a silent path would quietly inflate its
/// neighbour and the line would look complete while being wrong.
pub fn note_resolve_us(us: u64) {
    match PENDING.with(|p| p.take()) {
        Some(i) => MICROS[i].fetch_add(us, Ordering::Relaxed),
        None => {
            UNNAMED.fetch_add(1, Ordering::Relaxed);
            UNNAMED_US.fetch_add(us, Ordering::Relaxed)
        }
    };
}

/// Cumulative resolution count for one branch (monotonic; never reset). Used by
/// tests to assert an always-on branch fired without depending on a per-draw log
/// line (those are REIMS_VGPU_DRAW_LOG-gated to keep the fail log uncluttered).
pub fn count(branch: Branch) -> u64 {
    COUNTS[branch.idx()].load(Ordering::Relaxed)
}

/// A sub-step *inside* one branch's resolve.
///
/// The per-branch µs column says which branch is expensive. It cannot say which
/// part of that branch is, and for the type-11 rails that is now the open
/// question: they are 2.4 % of binds and 68 % of resolve time, and each runs
/// four or more sub-steps whose costs plausibly differ by three orders of
/// magnitude. One total reads identically whichever of them dominates, so a fix
/// aimed from it would be a guess.
///
/// Each variant times exactly one call site, so `count` here is the number of
/// times that site ran — not the number of binds. The map/import steps run once
/// per *guest run* (hundreds per bind), which is the distinction the per-bind
/// total cannot express and the reason a per-run fixed cost is invisible to it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Step {
    /// `try_type11_sample_zero_copy`: deferred-writeback flush over the surface.
    T11ZcFlush,
    /// `try_type11_sample_zero_copy`: `mapper::mapping_page_gpas` (revalidate +
    /// page-entry vector + control-page collision set).
    T11ZcGpas,
    /// `try_type11_sample_zero_copy`: one `HostOps::map_pages` per coalesced run.
    T11ZcMap,
    /// `try_type11_sample_zero_copy`: one `engine::ensure_host_import` per run.
    T11ZcImport,
    /// `load_type11_rgba_memoized`: native BGRA re-read of the whole surface.
    T11MemoRead,
    /// `load_type11_rgba_memoized`: memcmp of the re-read against the memo.
    T11MemoCmp,
    /// `load_type11_rgba_memoized`: BGRA→RGBA row conversion on a memo miss.
    T11MemoConvert,
}

const STEP_N: usize = 7;

impl Step {
    const fn idx(self) -> usize {
        match self {
            Step::T11ZcFlush => 0,
            Step::T11ZcGpas => 1,
            Step::T11ZcMap => 2,
            Step::T11ZcImport => 3,
            Step::T11MemoRead => 4,
            Step::T11MemoCmp => 5,
            Step::T11MemoConvert => 6,
        }
    }
}

const STEP_NAMES: [&str; STEP_N] = [
    "t11zc_flush",
    "t11zc_gpas",
    "t11zc_map",
    "t11zc_import",
    "t11m_read",
    "t11m_cmp",
    "t11m_convert",
];

static STEP_COUNTS: [AtomicU64; STEP_N] = [const { AtomicU64::new(0) }; STEP_N];
static STEP_US: [AtomicU64; STEP_N] = [const { AtomicU64::new(0) }; STEP_N];

/// Charge one execution of `step` with `us` microseconds.
pub fn note_step_us(step: Step, us: u64) {
    let i = step.idx();
    STEP_COUNTS[i].fetch_add(1, Ordering::Relaxed);
    STEP_US[i].fetch_add(us, Ordering::Relaxed);
}

/// Time `f`, charge it to `step`, and return its value.
pub fn timed<T>(step: Step, f: impl FnOnce() -> T) -> T {
    let started = std::time::Instant::now();
    let out = f();
    note_step_us(step, started.elapsed().as_micros() as u64);
    out
}

/// Cumulative (count, microseconds) for one step. Test accessor.
pub fn step_snapshot(step: Step) -> (u64, u64) {
    let i = step.idx();
    (
        STEP_COUNTS[i].load(Ordering::Relaxed),
        STEP_US[i].load(Ordering::Relaxed),
    )
}

/// Cumulative (count, bytes) per branch, indexed as [`NAMES`]; last = total.
pub fn snapshot() -> ([(u64, u64); N], u64) {
    let mut s = [(0u64, 0u64); N];
    for (i, slot) in s.iter_mut().enumerate() {
        *slot = (
            COUNTS[i].load(Ordering::Relaxed),
            BYTES[i].load(Ordering::Relaxed),
        );
    }
    (s, TOTAL.load(Ordering::Relaxed))
}

fn format_line(&(ref s, total): &([(u64, u64); N], u64)) -> String {
    use std::fmt::Write as _;
    let mut line = format!("sampled_branch_census total={total}");
    for (i, name) in NAMES.iter().enumerate() {
        // count:bytes:microseconds — the third column is what the branch cost,
        // which the first two cannot imply.
        let _ = write!(
            line,
            " {name}={}:{}:{}",
            s[i].0,
            s[i].1,
            MICROS[i].load(Ordering::Relaxed)
        );
    }
    let _ = write!(
        line,
        " unnamed={}:{}",
        UNNAMED.load(Ordering::Relaxed),
        UNNAMED_US.load(Ordering::Relaxed)
    );
    for (i, name) in STEP_NAMES.iter().enumerate() {
        // executions:microseconds — an execution is one call of that step, which
        // for the per-run steps is not one bind.
        let _ = write!(
            line,
            " {name}={}:{}",
            STEP_COUNTS[i].load(Ordering::Relaxed),
            STEP_US[i].load(Ordering::Relaxed)
        );
    }
    line
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn note_accumulates_per_branch_and_line_names_every_branch() {
        // Global counters are shared across parallel tests: assert deltas only.
        let (before, before_total) = snapshot();
        note(Branch::GvaCopy, 4096);
        note(Branch::GvaMemo, 0);
        let (after, after_total) = snapshot();
        assert_eq!(after_total - before_total, 2);
        let gc = Branch::GvaCopy.idx();
        let gm = Branch::GvaMemo.idx();
        assert_eq!(after[gc].0 - before[gc].0, 1);
        assert_eq!(after[gc].1 - before[gc].1, 4096);
        assert_eq!(after[gm].0 - before[gm].0, 1);
        assert_eq!(after[gm].1, before[gm].1);

        let line = format_line(&(after, after_total));
        assert!(line.starts_with("sampled_branch_census total="));
        for name in NAMES.iter().chain(STEP_NAMES.iter()) {
            assert!(
                line.contains(&format!(" {name}=")),
                "census line missing column {name}: {line}"
            );
        }
    }

    /// A step charge lands on its own step and is counted per execution, not per
    /// bind — the per-run steps run hundreds of times inside one resolve, which is
    /// the quantity the per-branch total cannot express.
    #[test]
    fn step_charges_land_per_execution_on_their_own_step() {
        let map = Step::T11ZcMap;
        let import = Step::T11ZcImport;
        let (map_n, map_us) = step_snapshot(map);
        let (import_n, import_us) = step_snapshot(import);

        note_step_us(map, 5);
        note_step_us(map, 7);
        assert_eq!(step_snapshot(map), (map_n + 2, map_us + 12));
        assert_eq!(step_snapshot(import), (import_n, import_us));

        // `timed` returns the closure's value and charges the elapsed time.
        let out = timed(import, || 42u32);
        assert_eq!(out, 42);
        assert_eq!(step_snapshot(import).0, import_n + 1);
    }

    /// A resolve's time lands on the branch that resolved it, and a resolve that
    /// named no branch lands on `unnamed` rather than on whoever ran last.
    ///
    /// The second half is the one worth pinning: without it a silent path
    /// inflates its neighbour, and the line reads as a complete attribution while
    /// being wrong about which branch is expensive — the exact question this
    /// column was added to answer.
    #[test]
    fn resolve_time_lands_on_its_own_branch_and_a_silent_one_is_named_unnamed() {
        // Globals are shared with the other tests in this module; drain any
        // pending branch first so this test measures only its own deltas.
        note_resolve_us(0);
        let zc = Branch::LinZeroCopy.idx();
        let before = MICROS[zc].load(Ordering::Relaxed);
        let before_unnamed = UNNAMED.load(Ordering::Relaxed);
        let before_unnamed_us = UNNAMED_US.load(Ordering::Relaxed);

        note(Branch::LinZeroCopy, 0);
        note_resolve_us(37);
        assert_eq!(MICROS[zc].load(Ordering::Relaxed) - before, 37);
        assert_eq!(UNNAMED.load(Ordering::Relaxed), before_unnamed);

        // No `note` this time: the pending slot was consumed above, so the charge
        // must not fall through to LinZeroCopy again.
        note_resolve_us(11);
        assert_eq!(
            MICROS[zc].load(Ordering::Relaxed) - before,
            37,
            "an unnamed resolve was charged to the previous branch"
        );
        assert_eq!(UNNAMED.load(Ordering::Relaxed) - before_unnamed, 1);
        assert_eq!(UNNAMED_US.load(Ordering::Relaxed) - before_unnamed_us, 11);
    }
}
