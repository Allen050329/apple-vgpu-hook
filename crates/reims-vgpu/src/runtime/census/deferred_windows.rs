//! Whether the three deferred-window caps ever bind.
//!
//! `GVA_DEFERRED_WINDOW_CAP` (16), `SURFACE_DEFERRED_WINDOW_CAP` (16) and
//! `STORAGE_RESIDENCY_WINDOWS_PER_MAPPING` (8) each bound a population of
//! deferred windows, and each window pins an engine-resident target. The bound
//! is real — an unpinned population is the "~260 stale residents (~516 MiB)
//! pinned for the guest lifetime" shape — but none of the three numbers came
//! from a measurement, and `STORAGE_RESIDENCY_WINDOWS_PER_MAPPING` says so at
//! its definition.
//!
//! Nothing could answer the prior question either. All three caps evict
//! silently, so a boot could not say whether a cap had ever bound, let alone
//! how close a population came to it. `deferred_flush_lost` reports a landing
//! that failed, not a landing the cap forced.
//!
//! So: `peak` is the high-water population each rail reached, and `evicted`
//! counts the windows a cap forced to land early. Both are **levels** — they
//! never reset, so a reader takes the last line of a boot, never a sum.
//!
//! Reading it:
//!
//! - `peak` well under the cap, `evicted=0` — the cap never bound. It is then
//!   headroom that has never been reached, and the number is unfalsifiable
//!   rather than justified.
//! - `peak` pinned at the cap with `evicted` climbing — the cap is the working
//!   set's actual limit and the guest is being throttled by it.
//!
//! The distinction matters because these are the last three unjustified
//! constants on the deferred-store path.

use std::sync::atomic::{AtomicU64, Ordering};

/// Which deferred-window population a sample belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rail {
    /// GVA render Stores — `state.gva_deferred_flush`.
    Gva,
    /// Type-11 surface render windows.
    Surface,
    /// Compute storage-residency windows, per mapping.
    Storage,
}

struct RailCounters {
    peak: AtomicU64,
    evicted: AtomicU64,
    /// The cap as the enforcing site sees it, recorded there rather than read
    /// back from the constant here — a reading is only interpretable against
    /// the bound that actually applied, and the two cannot drift if only one
    /// of them exists.
    cap: AtomicU64,
}

impl RailCounters {
    const fn new() -> Self {
        Self {
            peak: AtomicU64::new(0),
            evicted: AtomicU64::new(0),
            cap: AtomicU64::new(0),
        }
    }
}

static GVA: RailCounters = RailCounters::new();
static SURFACE: RailCounters = RailCounters::new();
static STORAGE: RailCounters = RailCounters::new();

fn rail(which: Rail) -> &'static RailCounters {
    match which {
        Rail::Gva => &GVA,
        Rail::Surface => &SURFACE,
        Rail::Storage => &STORAGE,
    }
}

/// Record a rail's live window population, keeping the high-water mark.
///
/// Called where the population is already known, so it costs one relaxed load
/// and at most one compare-exchange per arming.
pub fn note_population(which: Rail, live: usize, cap: usize) {
    let counters = rail(which);
    counters.cap.store(cap as u64, Ordering::Relaxed);
    let live = live as u64;
    let mut peak = counters.peak.load(Ordering::Relaxed);
    while live > peak {
        match counters.peak.compare_exchange_weak(
            peak,
            live,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => break,
            Err(seen) => peak = seen,
        }
    }
}

/// Record that a cap forced `n` windows to land early.
pub fn note_evicted(which: Rail, n: usize) {
    if n != 0 {
        rail(which)
            .evicted
            .fetch_add(n as u64, Ordering::Relaxed);
    }
}

/// One census line, or `None` while every rail is still empty — a boot that
/// armed no deferred window at all has nothing to say here.
pub fn census_line() -> Option<String> {
    let read = |c: &RailCounters| {
        (
            c.peak.load(Ordering::Relaxed),
            c.cap.load(Ordering::Relaxed),
            c.evicted.load(Ordering::Relaxed),
        )
    };
    let (gva_peak, gva_cap, gva_evicted) = read(&GVA);
    let (surf_peak, surf_cap, surf_evicted) = read(&SURFACE);
    let (stor_peak, stor_cap, stor_evicted) = read(&STORAGE);
    if gva_peak == 0 && surf_peak == 0 && stor_peak == 0 {
        return None;
    }
    Some(format!(
        "deferred_windows gva_peak={gva_peak} gva_cap={gva_cap} gva_evicted={gva_evicted} \
         surface_peak={surf_peak} surface_cap={surf_cap} surface_evicted={surf_evicted} \
         storage_peak={stor_peak} storage_cap={stor_cap} storage_evicted={stor_evicted} \
         (levels, not per-interval)"
    ))
}

/// Test-only reset; the counters are process-global levels.
#[cfg(test)]
pub fn reset_for_test() {
    for c in [&GVA, &SURFACE, &STORAGE] {
        c.peak.store(0, Ordering::Relaxed);
        c.evicted.store(0, Ordering::Relaxed);
        c.cap.store(0, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `peak` is a high-water mark, so a later smaller population must not
    /// lower it — a cap that bound once and then relaxed still bound.
    #[test]
    fn peak_keeps_the_high_water_mark() {
        reset_for_test();
        note_population(Rail::Gva, 3, 16);
        note_population(Rail::Gva, 9, 16);
        note_population(Rail::Gva, 1, 16);
        note_evicted(Rail::Gva, 2);
        note_evicted(Rail::Gva, 0);
        let line = census_line().expect("a rail was populated");
        assert!(line.contains("gva_peak=9"), "{line}");
        assert!(line.contains("gva_evicted=2"), "{line}");
    }

    /// A boot that armed no window says nothing, so the family's presence in a
    /// log is itself evidence that the rail ran.
    #[test]
    fn silent_until_a_window_is_armed() {
        reset_for_test();
        assert_eq!(census_line(), None);
        note_population(Rail::Storage, 1, 8);
        assert!(census_line().is_some());
    }

    /// The caps are reported beside the populations so a reader never has to go
    /// find them, and so a changed constant cannot silently invalidate a
    /// recorded reading.
    #[test]
    fn line_carries_the_caps_it_is_measured_against() {
        reset_for_test();
        note_population(Rail::Surface, 2, 16);
        let line = census_line().expect("a rail was populated");
        for field in ["gva_cap=", "surface_cap=", "storage_cap="] {
            assert!(line.contains(field), "missing {field}: {line}");
        }
    }
}
