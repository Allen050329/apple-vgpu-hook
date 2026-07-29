//! Always-on census: are deferred present-store writebacks ever consumed?
//!
//! A composite/present Store defers its guest-page writeback (ack-fast:
//! `try_defer_present_store` pins the resident and arms an async readback
//! prefetch instead of a ~5 ms synchronous scatter-DMA on the stamp path). The
//! deferred window then ends one of two ways:
//!
//!  * **flushed** — a consumer touched the pages (`flush_intersecting` from a
//!    full present capture, a guest sample, or `SynchronizeResources`), so
//!    `render_flush_one` replays the scatter-DMA into guest pages. The writeback
//!    was *needed*.
//!  * **superseded** — the very next present defers a fresher window over the
//!    same mapping before anything read the old one (`try_defer_present_store`'s
//!    drop loop / a synchronous Store superseding it). The old window's pages
//!    were *never re-read*: its writeback (and the prefetch readback armed for
//!    it) was pure overhead.
//!
//! Under a dmabuf-carried display (route B, `dmabuf_active`) the window publish
//! reads the GPU resident directly and the present capture skips its readback,
//! so nothing on the present hot path consumes the guest pages — the deferred
//! windows are expected to be almost entirely superseded. This census measures
//! exactly that ratio so a future session can decide, *evidence-first*, whether
//! the per-present writeback (and its armed prefetch readback) is elidable while
//! dmabuf carries the display. It is **measure-only** — it never gates behavior;
//! the elision decision it informs stays a separate, explicit change.
//!
//! Counts (not wall-clock) → trustworthy under the SCHED_IDLE agent boot. Emits
//! `writeback_consume …` to `/tmp/reims-vgpu-fail.log` (always-on `observe::off`) once
//! per ~1 s window from whichever hot-path event closes the window.

use crate::observe;
use std::sync::atomic::{AtomicU64, Ordering};

/// Deferred present-store windows armed (`try_defer_present_store` success).
static ARMED: AtomicU64 = AtomicU64::new(0);
/// Of `ARMED`, how many were armed while the display was dmabuf-carried.
static DMABUF_ARMED: AtomicU64 = AtomicU64::new(0);
/// Deferred windows consumed by a flush (`render_flush_one` success).
static FLUSHED: AtomicU64 = AtomicU64::new(0);
/// Deferred windows dropped by a fresher defer / a synchronous Store before any
/// flush — never re-read.
static SUPERSEDED: AtomicU64 = AtomicU64::new(0);
/// Present Stores that ran the synchronous scatter-DMA (no defer happened).
static SYNC: AtomicU64 = AtomicU64::new(0);
static WINDOW_START_MS: AtomicU64 = AtomicU64::new(0);

const WINDOW_MS: u64 = 1000;

/// One deferred present-store window was armed. `dmabuf` is the present's
/// display-carry state at defer time.
pub fn note_armed(dmabuf: bool) {
    ARMED.fetch_add(1, Ordering::Relaxed);
    if dmabuf {
        DMABUF_ARMED.fetch_add(1, Ordering::Relaxed);
    }
    maybe_emit();
}

/// One deferred window was flushed into guest pages (a consumer read them).
pub fn note_flushed() {
    FLUSHED.fetch_add(1, Ordering::Relaxed);
}

/// `n` deferred windows were dropped before any flush (superseded / lifecycle).
pub fn note_superseded(n: u64) {
    if n != 0 {
        SUPERSEDED.fetch_add(n, Ordering::Relaxed);
    }
}

/// One present Store ran the synchronous scatter-DMA (no defer).
pub fn note_sync(dmabuf: bool) {
    SYNC.fetch_add(1, Ordering::Relaxed);
    let _ = dmabuf;
    maybe_emit();
}

fn maybe_emit() {
    if let Some(line) = maybe_line_at(observe::elapsed_ms() as u64) {
        observe::off(line);
    }
}

fn maybe_line_at(now: u64) -> Option<String> {
    let start = WINDOW_START_MS.load(Ordering::Relaxed);
    if start == 0 {
        let _ = WINDOW_START_MS.compare_exchange(0, now, Ordering::Relaxed, Ordering::Relaxed);
        return None;
    }
    let dt = now.saturating_sub(start);
    if dt < WINDOW_MS {
        return None;
    }
    if WINDOW_START_MS
        .compare_exchange(start, now, Ordering::Relaxed, Ordering::Relaxed)
        .is_err()
    {
        return None;
    }
    let armed = ARMED.swap(0, Ordering::Relaxed);
    let dmabuf_armed = DMABUF_ARMED.swap(0, Ordering::Relaxed);
    let flushed = FLUSHED.swap(0, Ordering::Relaxed);
    let superseded = SUPERSEDED.swap(0, Ordering::Relaxed);
    let sync = SYNC.swap(0, Ordering::Relaxed);
    if armed == 0 && flushed == 0 && superseded == 0 && sync == 0 {
        return None;
    }
    Some(format_line(
        dt,
        armed,
        dmabuf_armed,
        flushed,
        superseded,
        sync,
    ))
}

fn format_line(
    dt: u64,
    armed: u64,
    dmabuf_armed: u64,
    flushed: u64,
    superseded: u64,
    sync: u64,
) -> String {
    // consume_ratio: of the deferred windows that resolved this window
    // (flushed + superseded), the fraction actually read back. Near 0 while
    // dmabuf_armed≈armed = the writeback is dead weight under dmabuf.
    let resolved = flushed.saturating_add(superseded);
    let consume_ratio = if resolved == 0 {
        0.0
    } else {
        flushed as f64 / resolved as f64
    };
    let armed_hz = armed.saturating_mul(1000) as f64 / dt.max(1) as f64;
    format!(
        "writeback_consume window_ms={dt} armed={armed} dmabuf_armed={dmabuf_armed} \
         flushed={flushed} superseded={superseded} sync={sync} \
         consume_ratio={consume_ratio:.3} armed_hz={armed_hz:.1}"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reset() {
        for a in [
            &ARMED,
            &DMABUF_ARMED,
            &FLUSHED,
            &SUPERSEDED,
            &SYNC,
            &WINDOW_START_MS,
        ] {
            a.store(0, Ordering::Relaxed);
        }
    }

    // Drive counters directly rather than the public `note_*` (which read the
    // real monotonic clock via `maybe_emit`) so the window timing is
    // deterministic under a controlled `now`.
    #[test]
    fn first_observation_seeds_window_and_stays_silent() {
        reset();
        ARMED.fetch_add(1, Ordering::Relaxed);
        // `now=0` is the unseeded sentinel; the first real observation must be
        // ≥1 to seed the window (else it never starts).
        assert_eq!(maybe_line_at(1), None);
        // Still inside the first window → no line.
        assert!(maybe_line_at(10).is_none());
    }

    #[test]
    fn ratio_is_zero_when_all_superseded_and_one_when_all_flushed() {
        // All superseded (dmabuf steady state): consume_ratio → 0.
        let all_dropped = format_line(1000, 120, 120, 0, 118, 0);
        assert!(all_dropped.contains("consume_ratio=0.000"));
        assert!(all_dropped.contains("dmabuf_armed=120"));
        // All flushed (nothing dmabuf, every writeback consumed): ratio → 1.
        let all_flushed = format_line(1000, 60, 0, 60, 0, 0);
        assert!(all_flushed.contains("consume_ratio=1.000"));
        // Half/half.
        let mixed = format_line(1000, 40, 40, 20, 20, 0);
        assert!(mixed.contains("consume_ratio=0.500"));
    }

    #[test]
    fn window_emits_once_past_the_interval_then_resets() {
        reset();
        // Seed the window at t=1 (0 is the unseeded sentinel).
        assert_eq!(maybe_line_at(1), None);
        ARMED.fetch_add(2, Ordering::Relaxed);
        SUPERSEDED.fetch_add(2, Ordering::Relaxed);
        // Before the window closes: nothing.
        assert!(maybe_line_at(WINDOW_MS).is_none());
        // At/after the window: exactly one line, counters drained.
        let line = maybe_line_at(1 + WINDOW_MS + 5).expect("window line");
        assert!(line.contains("armed=2"));
        assert!(line.contains("superseded=2"));
        // Counters were swapped to 0; a fresh window with no events is silent.
        assert!(maybe_line_at(1 + 2 * WINDOW_MS + 10).is_none());
    }
}
