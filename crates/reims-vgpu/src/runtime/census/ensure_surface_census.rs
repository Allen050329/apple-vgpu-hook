//! Always-on count census of the per-bind `ensure_surface_for_present` /
//! `resolve_type4_surface` task scan.
//!
//! Every sampled bind of a type-11 IOSurface resolves its backing through
//! `ensure_surface_for_present(mid)`. When the mapping is already mapped this
//! takes the *force* path (`resolve_type4_surface_force`), which walks the task
//! table (up to [`crate::model::regs::MAX_TASKS`]) doing a guest
//! `lookup_list_entry` per active task, looking for `mid` registered as a
//! type-4 `OBJECT_TYPE_SURFACE`. A type-11 IOSurface mapping has **no** type-4
//! owner, so that scan probes every active task, finds nothing, and returns —
//! pure per-bind waste that scales with the active-task count (i.e. with
//! compositor load). This census measures exactly that waste.
//!
//! Counts are scheduling-independent (unlike the `*_us` tranche fields), so the
//! ratio `no_owner / owner_found` and `probes_per_scan` are trustworthy even
//! under the SCHED_IDLE agent harness. It accumulates on the drain worker (off
//! the QEMU main core) and emits one cumulative `ensure_surface_census` line
//! every [`EMIT_EVERY`] calls via the always-on `observe::off` sink.
//! Measure-only — never gates behavior.

use std::sync::atomic::{AtomicU64, Ordering};

static CALLS: AtomicU64 = AtomicU64::new(0);
static FORCE_SCANS: AtomicU64 = AtomicU64::new(0);
static SCAN_PROBES: AtomicU64 = AtomicU64::new(0);
static NO_OWNER: AtomicU64 = AtomicU64::new(0);
static OWNER_FOUND: AtomicU64 = AtomicU64::new(0);
static HINT_HITS: AtomicU64 = AtomicU64::new(0);

/// One cumulative census line per this many `ensure_surface_for_present` calls.
const EMIT_EVERY: u64 = 16384;

/// Record one `ensure_surface_for_present` entry. Also the emit trigger.
pub fn note_call() {
    let calls = CALLS.fetch_add(1, Ordering::Relaxed) + 1;
    if calls.is_multiple_of(EMIT_EVERY) {
        crate::observe::off(format_line(&snapshot()));
    }
}

/// Record the outcome of one `resolve_type4_surface_ex` task scan: how many
/// active-task `lookup_list_entry` probes it performed, whether it found a
/// type-4 owner, and whether the owner-task hint short-circuited it (owner
/// resolved on the first hinted probe rather than walking the table).
pub fn note_scan(probes: u64, owner_found: bool, hint_hit: bool) {
    FORCE_SCANS.fetch_add(1, Ordering::Relaxed);
    SCAN_PROBES.fetch_add(probes, Ordering::Relaxed);
    if owner_found {
        OWNER_FOUND.fetch_add(1, Ordering::Relaxed);
    } else {
        NO_OWNER.fetch_add(1, Ordering::Relaxed);
    }
    if hint_hit {
        HINT_HITS.fetch_add(1, Ordering::Relaxed);
    }
}

/// Cumulative `(calls, force_scans, probes, no_owner, owner_found, hint_hits)`.
pub fn snapshot() -> (u64, u64, u64, u64, u64, u64) {
    (
        CALLS.load(Ordering::Relaxed),
        FORCE_SCANS.load(Ordering::Relaxed),
        SCAN_PROBES.load(Ordering::Relaxed),
        NO_OWNER.load(Ordering::Relaxed),
        OWNER_FOUND.load(Ordering::Relaxed),
        HINT_HITS.load(Ordering::Relaxed),
    )
}

fn format_line(
    &(calls, force_scans, probes, no_owner, owner_found, hint_hits): &(
        u64,
        u64,
        u64,
        u64,
        u64,
        u64,
    ),
) -> String {
    let per = |n: u64, d: u64| n.checked_div(d).unwrap_or(0);
    format!(
        "ensure_surface_census calls={calls} force_scans={force_scans} probes={probes} \
         no_owner={no_owner} owner_found={owner_found} hint_hits={hint_hits} \
         probes_per_scan={}",
        per(probes, force_scans),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scan_accumulates_and_line_reports_probes_per_scan() {
        let (_, f0, p0, n0, o0, h0) = snapshot();
        note_scan(5, false, false); // no owner: full scan of 5 active tasks
        note_scan(2, true, true); // owner found on hinted probe
        let (_, f1, p1, n1, o1, h1) = snapshot();
        assert_eq!(f1 - f0, 2);
        assert_eq!(p1 - p0, 7);
        assert_eq!(n1 - n0, 1);
        assert_eq!(o1 - o0, 1);
        assert_eq!(h1 - h0, 1);
        let line = format_line(&(0, 2, 7, 1, 1, 1));
        assert!(line.starts_with("ensure_surface_census calls=0"));
        assert!(line.contains("probes_per_scan=3"));
        assert!(line.contains("no_owner=1"));
    }
}
