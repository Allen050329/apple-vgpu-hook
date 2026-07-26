//! Always-on timing sub-split of the per-bind `setup_tex` phase.
//!
//! `setup_tex` (the `push_tex` sampled-binding loop in `metal_draw`) is the
//! dominant per-draw cost under compositor load (~82% of `draw_total_us`,
//! 287us/bind under active testufo+cursor load — grounded in a windowed
//! `setup_tex_split` delta). The always-on `sampled_branch_census` shows *which
//! branch* each bind takes (count) but not *where the time goes*: a
//! resident/zero-copy bind still pays the full per-bind resolution even though
//! it copies no bytes.
//!
//! This census splits each bind's resolver wall-clock into named buckets so one
//! boot log reveals which sub-op to cut:
//! - `ref_us`   — `resolve_type11_ref` (object-list + descriptor guest reads +
//!   type-11 registration) for the type-11 sampled path.
//! - `ensure_us`— `ensure_surface_for_present` (the type-4 task scan +
//!   MappingInternal resolve).
//! - the remainder of `resolve_us` is the byte load / resident bind / cache
//!   lookup that produced the source.
//! - `stats_us` — the post-resolve `rgba_rgb_a0_stats` scan (RGBA8 CPU binds).
//!
//! Both a **cumulative** and a **window** (since the previous emit) per-bind
//! value are reported: the window tracks the *current* load state (spikes,
//! steady testufo), the cumulative smooths it. It accumulates on the drain
//! worker (off the QEMU main core) and emits one `setup_tex_split` line every
//! [`EMIT_EVERY`] binds via the always-on `observe::off` sink. Measure-only —
//! never gates behavior.
//!
//! Wall-clock buckets are SCHED_IDLE-contaminated under the agent harness, so
//! trust the RATIO between the buckets (all preempted the same way), not the
//! absolute us — the ratio names which sub-op to cut.

use std::sync::atomic::{AtomicU64, Ordering};

static RESOLVE_US: AtomicU64 = AtomicU64::new(0);
static STATS_US: AtomicU64 = AtomicU64::new(0);
static REF_US: AtomicU64 = AtomicU64::new(0);
static ENSURE_US: AtomicU64 = AtomicU64::new(0);
// The resident-bind / host-cache / guest-page byte-load block — the "remainder"
// the module doc names. Isolating it from the descriptor-decode prelude (which
// stays in `resolve - ref - ensure - bind`) tells us whether a resolve spike is
// engine-side (resident readiness / sample) or object-list-decode side.
static BIND_US: AtomicU64 = AtomicU64::new(0);
static BINDS: AtomicU64 = AtomicU64::new(0);

// Snapshot at the previous emit, for window (recent) per-bind deltas.
static LAST_RESOLVE_US: AtomicU64 = AtomicU64::new(0);
static LAST_REF_US: AtomicU64 = AtomicU64::new(0);
static LAST_ENSURE_US: AtomicU64 = AtomicU64::new(0);
static LAST_BIND_US: AtomicU64 = AtomicU64::new(0);
static LAST_BINDS: AtomicU64 = AtomicU64::new(0);
// Engine-lock-wait window: previous cumulative snapshot so each emit reports the
// wait accrued (and acquisitions taken) since the last line.
static LAST_ENG_WAIT_NS: AtomicU64 = AtomicU64::new(0);
static LAST_ENG_ACQ: AtomicU64 = AtomicU64::new(0);

/// One cumulative census line per this many sampled binds.
const EMIT_EVERY: u64 = 512;

/// Record one bind's resolver time (the `resolve_sampled_source` /
/// attachment-alias resolution that produced the source). Called once per
/// bind, so this is also the bind counter + emit trigger.
pub fn note_resolve(us: u64) {
    RESOLVE_US.fetch_add(us, Ordering::Relaxed);
    let binds = BINDS.fetch_add(1, Ordering::Relaxed) + 1;
    if binds.is_multiple_of(EMIT_EVERY) {
        crate::observe::off(format_line());
    }
}

/// Record one bind's post-resolve stats-scan time (the fused
/// `rgba_rgb_a0_stats` pass + empty-layer/seed diagnostics). Zero for
/// resident/zero-copy binds that skip the scan.
pub fn note_stats(us: u64) {
    STATS_US.fetch_add(us, Ordering::Relaxed);
}

/// Record one bind's `resolve_type11_ref` time (a sub-slice of `resolve_us`).
pub fn note_ref(us: u64) {
    REF_US.fetch_add(us, Ordering::Relaxed);
}

/// Record one bind's `ensure_surface_for_present` time (a sub-slice of
/// `resolve_us`, summed across all surface candidates of the bind).
pub fn note_ensure(us: u64) {
    ENSURE_US.fetch_add(us, Ordering::Relaxed);
}

/// Record one bind's resident-bind / host-cache / guest-page byte-load block
/// time (a sub-slice of `resolve_us`). This is the "remainder" the doc names —
/// the section that acquires the engine lock (`resident_content_ready`,
/// `try_sample_resident_surface`).
pub fn note_bind(us: u64) {
    BIND_US.fetch_add(us, Ordering::Relaxed);
}

/// Cumulative `(resolve_us, stats_us, ref_us, ensure_us, binds)`.
pub fn snapshot() -> (u64, u64, u64, u64, u64) {
    (
        RESOLVE_US.load(Ordering::Relaxed),
        STATS_US.load(Ordering::Relaxed),
        REF_US.load(Ordering::Relaxed),
        ENSURE_US.load(Ordering::Relaxed),
        BINDS.load(Ordering::Relaxed),
    )
}

fn format_line() -> String {
    let (resolve_us, stats_us, ref_us, ensure_us, binds) = snapshot();
    let bind_us = BIND_US.load(Ordering::Relaxed);
    let per = |n: u64, d: u64| n.checked_div(d).unwrap_or(0);
    // Window deltas since the previous emit → current load state.
    let w_binds = binds - LAST_BINDS.swap(binds, Ordering::Relaxed);
    let w_resolve = resolve_us - LAST_RESOLVE_US.swap(resolve_us, Ordering::Relaxed);
    let w_ref = ref_us - LAST_REF_US.swap(ref_us, Ordering::Relaxed);
    let w_ensure = ensure_us - LAST_ENSURE_US.swap(ensure_us, Ordering::Relaxed);
    let w_bind = bind_us - LAST_BIND_US.swap(bind_us, Ordering::Relaxed);
    // Engine-lock-wait window: how much of this window's bind time was spent
    // WAITING for the global engine mutex (vs doing work under it). A high
    // `eng_wait_per_acq_us` during a dock-hover freeze proves the drain worker
    // is lock-bound behind a present-path holder, not compute-bound.
    let (eng_wait_ns, eng_acq, eng_max_ns) = engine_lock_snapshot();
    let w_eng_wait_ns = eng_wait_ns - LAST_ENG_WAIT_NS.swap(eng_wait_ns, Ordering::Relaxed);
    let w_eng_acq = eng_acq - LAST_ENG_ACQ.swap(eng_acq, Ordering::Relaxed);
    format!(
        "setup_tex_split binds={binds} resolve_per_bind_us={} ref_per_bind_us={} \
         ensure_per_bind_us={} bind_per_bind_us={} stats_per_bind_us={} | \
         win_resolve_us={} win_ref_us={} win_ensure_us={} win_bind_us={} | \
         eng_wait_per_acq_us={} eng_acq={} eng_max_us={}",
        per(resolve_us, binds),
        per(ref_us, binds),
        per(ensure_us, binds),
        per(bind_us, binds),
        per(stats_us, binds),
        per(w_resolve, w_binds),
        per(w_ref, w_binds),
        per(w_ensure, w_binds),
        per(w_bind, w_binds),
        per(w_eng_wait_ns / 1000, w_eng_acq),
        w_eng_acq,
        eng_max_ns / 1000,
    )
}

/// Engine-lock-wait snapshot bridge (`(total_wait_ns, acquisitions, max_wait_ns)`).
/// Returns zeros when the Vulkan backend is not compiled in.
fn engine_lock_snapshot() -> (u64, u64, u64) {
    #[cfg(feature = "backend-vulkan")]
    {
        crate::backend::vulkan::engine::engine_lock_wait_snapshot()
    }
    #[cfg(not(feature = "backend-vulkan"))]
    {
        (0, 0, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn note_accumulates_and_line_reports_per_bind_averages() {
        let (r0, s0, rf0, e0, b0) = snapshot();
        note_resolve(100);
        note_stats(40);
        note_ref(30);
        note_ensure(10);
        note_resolve(200);
        note_stats(60);
        note_ref(50);
        note_ensure(20);
        let (r1, s1, rf1, e1, b1) = snapshot();
        assert_eq!(r1 - r0, 300);
        assert_eq!(s1 - s0, 100);
        assert_eq!(rf1 - rf0, 80);
        assert_eq!(e1 - e0, 30);
        assert_eq!(b1 - b0, 2);
        let line = format_line();
        assert!(line.starts_with("setup_tex_split"));
        assert!(line.contains("ref_per_bind_us="));
        assert!(line.contains("ensure_per_bind_us="));
        assert!(line.contains("win_resolve_us="));
    }
}
