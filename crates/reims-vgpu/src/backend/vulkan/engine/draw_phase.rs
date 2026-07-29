//! Where one draw's wall clock goes, split at the boundaries that mean
//! different fixes.
//!
//! `drain_duty` established that `draw_us` is 93-99% of the drain worker's busy
//! time, and `engine_delta` priced the bytes that cross the bus. Neither can
//! separate the two shapes a slow draw comes in, and they need opposite work:
//!
//! - **Bytes.** Seed upload and readback copy dominate → the fix is to move
//!   less, which is the deferred-rail and `output_bgra` family.
//! - **Latency.** The post-submit fence wait dominates → the fix is to stop
//!   round-tripping the GPU per draw, and moving bytes faster buys nothing.
//!
//! One draw's total is charged to exactly one phase at a time, so the six
//! numbers sum to the draw. The split points are the calls that change what the
//! CPU is doing:
//!
//! | phase | from | to |
//! |---|---|---|
//! | `prep` | entry | `begin_entry` returns a ring slot |
//! | `setup` | there | the descriptor set is written |
//! | `record` | there | the CB is ended |
//! | `submit` | there | `queue_submit` returns |
//! | `wait` | there | this draw's fence signals |
//! | `readback` | there | the mapped buffer is copied out |
//!
//! `setup` is split out from `record` because every `vkCreateImage`,
//! `vkAllocateMemory`, slab block and pipeline compile a draw needs happens
//! there, after the ring slot is held and before a single command is recorded.
//! Fusing the two would leave "the driver allocated" and "we encoded a lot of
//! commands" in one number, and the pool-trim policy is decided by which.
//!
//! A draw that returns early — a decline, a batched deferred submit, a
//! `skip_readback` target — charges its remainder to whichever phase was open,
//! because [`DrawTimer`] commits from `Drop`. That is deliberate: an exit is not
//! a phase, and threading a commit through every `?` would be the one thing
//! guaranteed to go stale.
//!
//! # Why this is a tally and not a decline
//!
//! Per `AGENTS.md`, a census must not be the only record that guest work was
//! lost. Nothing here reports a loss: a slow draw still draws. The one line this
//! module emits outside the per-second aggregate is the *stall* report, and that
//! one is bounded per boot rather than latched per key, because the distribution
//! is the signal — one 950 ms draw and two hundred of them are different bugs.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// A draw at or above this is a stall rather than a slow frame: at 60 Hz it has
/// already cost six frames, and the guest's compositor blocks on the same lock
/// for the duration. Reported individually with its phase split.
const STALL_US: u64 = 100_000;

/// Cap on individual stall reports per boot. The aggregate below keeps counting
/// after this; only the per-event lines stop, so a pathological boot cannot
/// flood the sink it is diagnosed through.
const STALL_REPORT_CAP: u64 = 256;

/// Phase slots, in the order a draw passes through them.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Phase {
    Prep = 0,
    Setup = 1,
    Record = 2,
    Submit = 3,
    Wait = 4,
    Readback = 5,
}

const PHASES: usize = 6;

static ACC: [AtomicU64; PHASES] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];
static DRAWS: AtomicU64 = AtomicU64::new(0);
static MAX_US: AtomicU64 = AtomicU64::new(0);
static STALLS: AtomicU64 = AtomicU64::new(0);
static STALL_LINES: AtomicU64 = AtomicU64::new(0);

/// One window of the split, as taken by the per-second census.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DrawPhaseWindow {
    pub prep_us: u64,
    pub setup_us: u64,
    pub record_us: u64,
    pub submit_us: u64,
    pub wait_us: u64,
    pub readback_us: u64,
    pub draws: u64,
    pub max_us: u64,
    pub stalls: u64,
}

/// Take and clear the window. `None` when no draw ran, so an idle second costs
/// no line.
pub fn take_window() -> Option<DrawPhaseWindow> {
    let draws = DRAWS.swap(0, Ordering::Relaxed);
    let w = DrawPhaseWindow {
        prep_us: ACC[Phase::Prep as usize].swap(0, Ordering::Relaxed),
        setup_us: ACC[Phase::Setup as usize].swap(0, Ordering::Relaxed),
        record_us: ACC[Phase::Record as usize].swap(0, Ordering::Relaxed),
        submit_us: ACC[Phase::Submit as usize].swap(0, Ordering::Relaxed),
        wait_us: ACC[Phase::Wait as usize].swap(0, Ordering::Relaxed),
        readback_us: ACC[Phase::Readback as usize].swap(0, Ordering::Relaxed),
        draws,
        max_us: MAX_US.swap(0, Ordering::Relaxed),
        stalls: STALLS.swap(0, Ordering::Relaxed),
    };
    (draws > 0).then_some(w)
}

/// Charges a draw's wall clock to one phase at a time.
///
/// Held by value in `execute_draw_inner`; [`DrawTimer::enter`] closes the open
/// phase and opens the next. The commit is in `Drop` so every exit — including
/// the `?` on a decline — lands its time somewhere.
pub(crate) struct DrawTimer {
    started: Instant,
    last: Instant,
    open: Phase,
    us: [u64; PHASES],
    /// Set once the draw knows what it is drawing, so a stall report can say
    /// what was on screen rather than only how long it took.
    geom: (u32, u32),
    readback_bytes: u64,
}

impl DrawTimer {
    pub(crate) fn start() -> Self {
        let now = Instant::now();
        Self {
            started: now,
            last: now,
            open: Phase::Prep,
            us: [0; PHASES],
            geom: (0, 0),
            readback_bytes: 0,
        }
    }

    /// Close the open phase and open `next`.
    pub(crate) fn enter(&mut self, next: Phase) {
        let now = Instant::now();
        self.us[self.open as usize] += now.duration_since(self.last).as_micros() as u64;
        self.last = now;
        self.open = next;
    }

    /// Context for a stall report. Cheap enough to set unconditionally.
    pub(crate) fn note_target(&mut self, width: u32, height: u32, readback_bytes: u64) {
        self.geom = (width, height);
        self.readback_bytes = readback_bytes;
    }
}

impl Drop for DrawTimer {
    fn drop(&mut self) {
        let now = Instant::now();
        self.us[self.open as usize] += now.duration_since(self.last).as_micros() as u64;
        let total = now.duration_since(self.started).as_micros() as u64;
        for (slot, acc) in ACC.iter().enumerate() {
            acc.fetch_add(self.us[slot], Ordering::Relaxed);
        }
        DRAWS.fetch_add(1, Ordering::Relaxed);
        MAX_US.fetch_max(total, Ordering::Relaxed);
        if total < STALL_US {
            return;
        }
        STALLS.fetch_add(1, Ordering::Relaxed);
        let line = STALL_LINES.fetch_add(1, Ordering::Relaxed);
        if line >= STALL_REPORT_CAP {
            return;
        }
        let (w, h) = self.geom;
        let latched = if line + 1 == STALL_REPORT_CAP {
            " (last: report cap reached)"
        } else {
            ""
        };
        crate::observe::off(format!(
            "draw_stall us={total} prep_us={} setup_us={} record_us={} submit_us={} \
             wait_us={} readback_us={} geom={w}x{h} readback_bytes={} exit={:?}{latched}",
            self.us[Phase::Prep as usize],
            self.us[Phase::Setup as usize],
            self.us[Phase::Record as usize],
            self.us[Phase::Submit as usize],
            self.us[Phase::Wait as usize],
            self.us[Phase::Readback as usize],
            self.readback_bytes,
            self.open,
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every exit commits, and the phase left open at the exit is the one the
    /// remainder is charged to. This is the property that lets `?` returns keep
    /// their time — the alternative (a commit before each return) would silently
    /// drop a phase the next time someone adds an early return.
    #[test]
    fn an_early_exit_charges_its_remainder_to_the_open_phase() {
        let _ = take_window();
        {
            let mut t = DrawTimer::start();
            t.enter(Phase::Record);
            t.enter(Phase::Wait);
            // Dropped here with `Wait` open — as a `?` on a failed fence would.
        }
        let w = take_window().expect("a dropped timer counts a draw");
        assert_eq!(w.draws, 1);
        // Readback never opened, so it must be exactly zero rather than
        // inheriting the tail.
        assert_eq!(w.readback_us, 0);
        assert_eq!(w.submit_us, 0);
    }

    /// An idle second must produce no line at all: the census divides against
    /// `drain_duty`, and a zero row there is already reported by `draws=0`.
    #[test]
    fn a_window_with_no_draw_is_none() {
        let _ = take_window();
        assert_eq!(take_window(), None);
    }

    /// The window is a delta, not a running total — two reads of one draw must
    /// not both report it.
    #[test]
    fn taking_the_window_clears_it() {
        let _ = take_window();
        drop(DrawTimer::start());
        assert_eq!(take_window().map(|w| w.draws), Some(1));
        assert_eq!(take_window(), None);
    }
}
