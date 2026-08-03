//! What `draw_phase`'s `stage_us` is, split by the mechanism that would fix it.
//!
//! `stage_us` is the largest single column in the device. On a driven Safari
//! drag, settled x86/PCI, host GPU at P8: 3978 draws in one second spend
//! **200 ms** there, which is 83 % of `draw_phase`'s whole second and level
//! with the difference pass's `pass_us`. Everything else in that phase together
//! — pipeline, record, sampled, prep, descriptors, submit — is ~41 ms.
//!
//! One bar cannot choose between the fixes, which is the same argument that
//! split `setup` into four and `binds_us` into three. The phase covers five
//! populations and they want opposite work:
//!
//! | part | what it is | what would fix it |
//! |---|---|---|
//! | `acquire` | [`ResourcePools::acquire_staging`] | pool sizing; a miss creates a buffer and allocates memory |
//! | `bytes` | `write_staging` from a `BufferContent::Bytes` | **the second copy** — those bytes were already assembled out of guest RAM by `load_buffer_content`, which `bind_phase` charges separately |
//! | `runs` | `write_staging_from_runs` | moving fewer bytes; this arm is already one copy, guest RAM straight into mapped staging |
//! | `swap` | `write_staging_swap_rb` on a seed | nothing — it is the copy that had to happen, with a byte exchange folded in |
//! | `shift` | the `base_instance` prefix a Constant-step vertex stream needs | keeping those binds off the CPU path |
//!
//! The `bytes`/`runs` division is the one to read first, because it prices a
//! lever nobody has costed: `BufferContent::Bytes` arrives as an
//! `Arc<Vec<u8>>` that `load_buffer_content` filled from guest memory, and
//! staging it copies the same bytes a second time. `BufferContent::GuestRuns`
//! does not — `write_staging_from_runs` exists precisely because the deferred
//! snapshot path used to `cpu_bytes()` into a heap `Vec` and then
//! `write_staging` that, and removing the intermediate was worth two copies and
//! an allocation per bind. Whether the *other* arm is still paying that is what
//! `bytes_us` against `runs_us` answers, and the byte counters beside them say
//! at what rate.
//!
//! # Why the call sites and not the pool
//!
//! The four pool functions are also called from the sampled-image path, which
//! `draw_phase` charges to `acquire_sampled` and `sampled_upload`. Instrumenting
//! the pool would mix those in and the parts would no longer sum to `stage_us`.
//! `stage_buffer_content` is called from inside the `Stage` span and nowhere
//! else, so wrapping it and the four open-coded sites in that span is exact.
//!
//! # What the census costs
//!
//! Two `Instant::now()` per span, and a span per staging operation rather than
//! per draw — a draw with many distinct vertex streams opens several. Measured
//! against the sum: if `acquire + bytes + runs + swap + shift` starts exceeding
//! `draw_phase`'s `stage_us`, the census is the difference and should be read as
//! such. `AGENTS.md` records an audit that moved `land_us` 328 → 380 µs by
//! reading its own subject, so this is not hypothetical.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// The staging operations inside one draw's `Stage` phase.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Part {
    /// A staging slot was taken from the pool, or created.
    Acquire = 0,
    /// Host bytes already in a `Vec` were copied into mapped staging.
    Bytes = 1,
    /// Guest RAM was gathered straight into mapped staging.
    Runs = 2,
    /// A seed was copied with red and blue exchanged.
    Swap = 3,
    /// A Constant-step vertex stream was rebuilt behind its `base_instance`
    /// prefix. This is the one part that is neither a pool call nor a staging
    /// write — it is a `Vec` allocation and a copy before either.
    Shift = 4,
}

const PARTS: usize = 5;

/// Nanoseconds, per [`crate::observe::phase_clock`]. This census opens a span
/// per staging operation at tens of thousands a second, which is exactly the
/// population a microsecond accumulator reports as free.
static NS: [AtomicU64; PARTS] = [const { AtomicU64::new(0) }; PARTS];
static N: [AtomicU64; PARTS] = [const { AtomicU64::new(0) }; PARTS];
static BYTES: [AtomicU64; PARTS] = [const { AtomicU64::new(0) }; PARTS];

/// One window of the split, as taken by the per-second census.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct StagePhaseWindow {
    pub acquire_us: u64,
    pub acquires: u64,
    pub bytes_us: u64,
    pub bytes_n: u64,
    pub bytes_b: u64,
    pub runs_us: u64,
    pub runs_n: u64,
    pub runs_b: u64,
    pub swap_us: u64,
    pub swap_n: u64,
    pub swap_b: u64,
    pub shift_us: u64,
    pub shift_n: u64,
    pub shift_b: u64,
}

/// Take and clear the window. `None` when nothing staged, so an idle second
/// costs no line.
pub fn take_window() -> Option<StagePhaseWindow> {
    let us = |p: Part| crate::observe::phase_clock::to_us(NS[p as usize].swap(0, Ordering::Relaxed));
    let n = |p: Part| N[p as usize].swap(0, Ordering::Relaxed);
    let b = |p: Part| BYTES[p as usize].swap(0, Ordering::Relaxed);
    let w = StagePhaseWindow {
        acquire_us: us(Part::Acquire),
        acquires: n(Part::Acquire),
        bytes_us: us(Part::Bytes),
        bytes_n: n(Part::Bytes),
        bytes_b: b(Part::Bytes),
        runs_us: us(Part::Runs),
        runs_n: n(Part::Runs),
        runs_b: b(Part::Runs),
        swap_us: us(Part::Swap),
        swap_n: n(Part::Swap),
        swap_b: b(Part::Swap),
        shift_us: us(Part::Shift),
        shift_n: n(Part::Shift),
        shift_b: b(Part::Shift),
    };
    // `Acquire` carries no bytes, so it is swapped and dropped rather than
    // left to accumulate into a number nothing reads.
    let _ = b(Part::Acquire);
    let staged = w.acquires + w.bytes_n + w.runs_n + w.swap_n + w.shift_n;
    (staged > 0).then_some(w)
}

/// Charges one staging operation to one part, from `open` to `Drop`.
pub(crate) struct Span {
    part: Part,
    bytes: u64,
    started: Instant,
}

impl Span {
    /// A part with no byte count of its own — the pool call.
    pub(crate) fn open(part: Part) -> Self {
        Self {
            part,
            bytes: 0,
            started: Instant::now(),
        }
    }

    /// A part that moves `bytes`, so the window can state a rate rather than
    /// only a duration.
    pub(crate) fn moving(part: Part, bytes: u64) -> Self {
        Self {
            part,
            bytes,
            started: Instant::now(),
        }
    }
}

impl Drop for Span {
    fn drop(&mut self) {
        let slot = self.part as usize;
        NS[slot].fetch_add(
            crate::observe::phase_clock::charge_ns(self.started.elapsed()),
            Ordering::Relaxed,
        );
        N[slot].fetch_add(1, Ordering::Relaxed);
        BYTES[slot].fetch_add(self.bytes, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A window reports every part it was given, and clears itself so the next
    /// second starts from zero rather than from the boot's running total.
    #[test]
    fn a_window_takes_what_was_charged_and_leaves_nothing() {
        // Other tests in this binary share these statics, so start from a
        // known-empty window rather than assuming one.
        let _ = take_window();
        drop(Span::open(Part::Acquire));
        drop(Span::moving(Part::Bytes, 4096));
        drop(Span::moving(Part::Bytes, 1024));
        drop(Span::moving(Part::Runs, 8192));
        let w = take_window().expect("something was staged");
        assert_eq!((w.acquires, w.bytes_n, w.bytes_b), (1, 2, 5120));
        assert_eq!((w.runs_n, w.runs_b), (1, 8192));
        assert_eq!((w.swap_n, w.shift_n), (0, 0));
        assert_eq!(take_window(), None, "a taken window must not repeat");
    }

    /// A staging operation is sub-microsecond and there are tens of thousands
    /// of them a second, which is the population
    /// [`crate::observe::phase_clock`] exists for: under a microsecond-
    /// truncating accumulator every span here charges exactly zero and the
    /// column this split was built to divide reads free.
    ///
    /// Threshold measured, not guessed — see `runtime::bind_phase`'s twin,
    /// where 20 000 empty spans read 302-308 µs against 3 truncating.
    #[test]
    fn twenty_thousand_sub_microsecond_spans_are_not_free() {
        let _ = take_window();
        for _ in 0..20_000 {
            drop(Span::moving(Part::Bytes, 1));
        }
        let w = take_window().expect("something was staged");
        assert_eq!(w.bytes_n, 20_000);
        assert!(w.bytes_us > 100, "{w:?}");
    }

    /// The byte counters are per part. A part charged bytes must not leak them
    /// into another, or `bytes_us` and `runs_us` cannot be compared at all —
    /// which is the whole reason this split exists.
    #[test]
    fn bytes_do_not_cross_between_parts() {
        let _ = take_window();
        drop(Span::moving(Part::Swap, 777));
        drop(Span::moving(Part::Shift, 88));
        let w = take_window().expect("something was staged");
        assert_eq!((w.swap_b, w.shift_b), (777, 88));
        assert_eq!((w.bytes_b, w.runs_b), (0, 0));
    }
}
