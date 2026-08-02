//! Where one draw's wall clock goes, split at the boundaries that mean
//! different fixes.
//!
//! `drain_duty` established that `draw_us` is 93-99% of the drain worker's busy
//! time, and `engine_delta` priced the bytes that cross the bus. Neither can
//! separate the two shapes a slow draw comes in, and they need opposite work:
//!
//! - **Bytes.** `stage` and `readback` dominate → the fix is to move less, which
//!   is the deferred-rail and `output_bgra` family.
//! - **Latency.** `wait` dominates → the fix is to stop round-tripping the GPU
//!   per draw, and moving bytes faster buys nothing.
//!
//! Measured on one x86/Vulkan boot under the standing soak (442 206 draws over
//! 342 s): `wait` 43%, the four setup phases 37%, `readback` 13%, and `record` —
//! encoding the commands — 1.1%. Joined per window against `engine_delta`, only
//! **14%** of draws read back at all and each of those blocks **1.2 ms** in the
//! fence wait, while the readback copy itself runs at 8.9 GB/s. So the cost is
//! latency, not bytes: 61 847 readbacks spent 74 s of that boot waiting for a
//! single queue to drain.
//!
//! One draw's total is charged to exactly one phase at a time, so the nine
//! numbers sum to the draw. The split points are the calls that change what the
//! CPU is doing:
//!
//! | phase | from | to |
//! |---|---|---|
//! | `prep` | entry | `begin_entry` returns a ring slot |
//! | `pipeline` | there | shaders, layout, pass and pipeline are resolved |
//! | `stage` | there | vertex/index/storage/seed bytes are in staging |
//! | `stage_pass` | there | the primary render pass is resolved |
//! | `acquire` | there | the render target, its framebuffer and any transient depth are held |
//! | `acquire_sampled` | there | every sampled image the draw binds is held |
//! | `acquire_readback` | there | this draw's readback buffer is held |
//! | `descriptors` | there | the descriptor set is written |
//! | `record` | there | the CB is ended |
//! | `submit` | there | `queue_submit` returns |
//! | `wait` | there | this draw's fence signals |
//! | `readback` | there | the mapped buffer is copied out |
//!
//! The four middle phases are what a first pass called `setup`, split because
//! that one number came out at 37% of all draw time while `record` — encoding
//! the actual commands — came out at 1%, and the four have nothing in common
//! but their position. `stage` is host memcpy into mapped staging and scales
//! with bytes; `pipeline` is driver compiles; `descriptors` is pool pressure.
//! Each has a different fix and one bar cannot choose between them.
//!
//! # Why `acquire` is three numbers and not one
//!
//! `acquire` used to cover the render target, the sampled images and the
//! draw's readback buffer, and its description here said it "scales with
//! churn" — meaning
//! `vkCreateImage`/`vkAllocateMemory`/slab. On a driven x86/Vulkan boot that
//! description sent a reader after the wrong cost, so the phase was split where
//! the two populations meet.
//!
//! What convicted the churn reading is a regression of the two counters against
//! each other across 32 driven one-second windows. If `acquire` were creates,
//! `acquire_us / creates` would be flat; it is not, and it moves the wrong way:
//!
//! ```text
//! draws   acquire_us   creates   us/create   us/draw
//!   980        81715        96         851      83.4
//!   118        17205       168         102     145.8
//! ```
//!
//! The window with **more** creates and an eighth of the draws spent a fifth of
//! the time. Across all 32 windows `us/create` ranges 402-1141 while `us/draw`
//! holds 73-146, so the phase is paid per draw, not per creation — and it is
//! paid on the cache-*hit* path, because those windows report `gen_mismatch=0`,
//! `target_evicts=0` and every `*_misses` counter at 0. `registry_ensure`'s hit
//! arm is a `HashMap` get and a touch, which cannot be 85 us.
//!
//! That leaves the rest of the phase, and splitting it is what distinguishes
//! "the target was expensive to hold" from "the textures were" — the same
//! argument that split `setup`.
//!
//! **The first split's reading settled the target half and nothing else.** On a
//! driven x86/Vulkan boot, five consecutive one-second windows at ~660 draws
//! read `acquire_us` 0, 0, 0, 0, 52 against `acquire_sampled_us` 66798, 66579,
//! 63803, 67030, 64430. So holding the render target, its framebuffer and its
//! depth costs *nothing* — the churn reading is refuted outright, not merely
//! doubted — and the entire ~100 us per draw is downstream of it.
//!
//! That reading is also what forced the third number. The block after the
//! sampled loop ends with `acquire_readback`, which holds a `width * height * 4`
//! buffer — 8.3 MB at 1920x1080 — once per draw that reads back. Leaving it
//! inside `acquire_sampled` would have charged an 8 MB buffer acquisition to
//! "sampled textures" and invited exactly the misreading this section exists to
//! correct, so it gets its own slot. The two are separated by what fixes them:
//! `acquire_sampled` is per bound texture and is attacked by binding fewer or
//! caching better, `acquire_readback` is per full-frame buffer and is attacked
//! by not reading back.
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
    Pipeline = 1,
    Stage = 2,
    StagePass = 3,
    Acquire = 4,
    AcquireSampled = 5,
    AcquireReadback = 6,
    Descriptors = 7,
    Record = 8,
    Submit = 9,
    Wait = 10,
    Readback = 11,
}

const PHASES: usize = 12;

static ACC: [AtomicU64; PHASES] = [const { AtomicU64::new(0) }; PHASES];
static DRAWS: AtomicU64 = AtomicU64::new(0);
static MAX_US: AtomicU64 = AtomicU64::new(0);
static STALLS: AtomicU64 = AtomicU64::new(0);
static STALL_LINES: AtomicU64 = AtomicU64::new(0);

/// One window of the split, as taken by the per-second census.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DrawPhaseWindow {
    pub prep_us: u64,
    pub pipeline_us: u64,
    pub stage_us: u64,
    pub stage_pass_us: u64,
    pub acquire_us: u64,
    pub acquire_sampled_us: u64,
    pub acquire_readback_us: u64,
    pub descriptors_us: u64,
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
        pipeline_us: ACC[Phase::Pipeline as usize].swap(0, Ordering::Relaxed),
        stage_us: ACC[Phase::Stage as usize].swap(0, Ordering::Relaxed),
        stage_pass_us: ACC[Phase::StagePass as usize].swap(0, Ordering::Relaxed),
        acquire_us: ACC[Phase::Acquire as usize].swap(0, Ordering::Relaxed),
        acquire_sampled_us: ACC[Phase::AcquireSampled as usize].swap(0, Ordering::Relaxed),
        acquire_readback_us: ACC[Phase::AcquireReadback as usize].swap(0, Ordering::Relaxed),
        descriptors_us: ACC[Phase::Descriptors as usize].swap(0, Ordering::Relaxed),
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
            "draw_stall us={total} prep_us={} pipeline_us={} stage_us={} stage_pass_us={} \
             acquire_us={} acquire_sampled_us={} acquire_readback_us={} descriptors_us={} \
             record_us={} submit_us={} wait_us={} readback_us={} geom={w}x{h} \
             readback_bytes={} exit={:?}{latched}",
            self.us[Phase::Prep as usize],
            self.us[Phase::Pipeline as usize],
            self.us[Phase::Stage as usize],
            self.us[Phase::StagePass as usize],
            self.us[Phase::Acquire as usize],
            self.us[Phase::AcquireSampled as usize],
            self.us[Phase::AcquireReadback as usize],
            self.us[Phase::Descriptors as usize],
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

    /// Holding the render target and holding the sampled images are separate
    /// accumulators, so a draw that spends its time in the sampled loop cannot
    /// be read as an expensive target.
    ///
    /// This is the whole point of the split: the pooled number said "acquire"
    /// and was read as `vkCreateImage` churn, when the per-draw regression said
    /// it could not be. Re-pooling the two would restore exactly that ambiguity.
    ///
    /// The failure this actually guards is a **readout** mis-wiring —
    /// `acquire_sampled_us` taken from `ACC[Phase::Acquire]` — which compiles,
    /// and which reads as a sampled loop that costs nothing. Verified by making
    /// that edit: the assertion below fires. A duplicate discriminant is *not*
    /// what this covers, because rustc rejects it (E0081) before the test runs.
    #[test]
    fn the_sampled_loop_is_charged_apart_from_the_target() {
        // The phases themselves take nanoseconds, so without a measurable sleep
        // every slot reads 0 and the assertions below would pass under a slot
        // collision too. The sleep is what gives the test its power: it is spent
        // entirely inside the sampled loop, so only that slot may carry it.
        const SAMPLED_SLEEP: std::time::Duration = std::time::Duration::from_millis(4);
        let _ = take_window();
        {
            let mut t = DrawTimer::start();
            t.enter(Phase::Acquire);
            t.enter(Phase::AcquireSampled);
            std::thread::sleep(SAMPLED_SLEEP);
            // Dropped with the sampled loop open, as a decline on a texture
            // that cannot be resolved would leave it.
        }
        let w = take_window().expect("a dropped timer counts a draw");
        assert_eq!(w.draws, 1);
        // The sleep landed, so the three slots are genuinely being compared.
        assert!(
            w.acquire_sampled_us >= 2_000,
            "sampled loop lost its own time: {w:?}"
        );
        // The remainder belongs to the phase that was open. `Acquire` closed
        // when the sampled loop opened, so it must not have absorbed it.
        assert_eq!(
            w.acquire_us, 0,
            "target acquisition charged time the sampled loop spent: {w:?}"
        );
        // Nor may the readback buffer, which is the slot the sampled loop's own
        // time would land in if the two were re-pooled the other way.
        assert_eq!(
            w.acquire_readback_us, 0,
            "readback acquisition charged time the sampled loop spent: {w:?}"
        );
        // Every later phase stays clean, so the tail did not simply spill.
        assert_eq!(w.descriptors_us, 0);
        assert_eq!(w.record_us, 0);
        assert_eq!(w.readback_us, 0);
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
