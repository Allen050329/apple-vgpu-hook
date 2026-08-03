//! The split of [`crate::runtime::chain_phase`]'s `binds_us`, over the same
//! window.
//!
//! # Why
//!
//! With the render writeback removed, the draw path is what caps this device,
//! and `binds_us` is its largest column: 23.5 µs of a 103 µs draw on a driven
//! x86/PCI control second, within a microsecond of `draw_phase`'s `stage_us`.
//! One column covering `load_buffer_content` for every vertex buffer, the same
//! for every fragment buffer, and the whole stage-in attribute walk is three
//! costs with three different fixes, and no line could tell them apart.
//!
//! This stands to `chain_phase`'s `binds_us` exactly as `draw_phase` stands to
//! its `engine_us`: a division, emitted on the same cadence, read against the
//! line above it.
//!
//! # What it does not report
//!
//! No `rest_us`. The three parts do **not** claim to sum to `binds_us` — the
//! phase also clones two shader `Arc`s and builds a `BTreeSet` outside them —
//! and a computed remainder would go silently wrong the moment a fourth cost
//! were added between them. Divide against `chain_phase`'s `binds_us` by hand;
//! what the parts do not cover is the answer to "is there a fourth".
//!
//! Like every phase census here it reports no loss. A slow bind is not a
//! declined one, and a chain that returns early from inside a span charges its
//! remainder to that span, because the commit is in `Drop`.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use crate::observe::phase_clock::{charge_ns, to_us};

/// The parts of the bind phase that are worth telling apart.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Part {
    /// `load_buffer_content` over `req.vertex_buffers`.
    VertexLoad = 0,
    /// `load_buffer_content` over `req.fragment_buffers`.
    FragmentLoad = 1,
    /// The stage-in attribute walk over the pipeline's vertex block.
    Attrs = 2,
}

const PARTS: usize = 3;

/// Nanoseconds, per [`crate::observe::phase_clock`]. The attribute walk is the
/// reason: it is sub-microsecond per draw at tens of thousands of draws a
/// second, so a microsecond accumulator reported it as free.
static ACC: [AtomicU64; PARTS] = [const { AtomicU64::new(0) }; PARTS];
static BINDS: AtomicU64 = AtomicU64::new(0);

/// One window of the split, as taken by the per-second census.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct BindPhaseWindow {
    pub vertex_us: u64,
    pub fragment_us: u64,
    pub attrs_us: u64,
    /// Bind phases entered in the window — the denominator the three share.
    pub binds: u64,
}

/// Take and clear the window. `None` when no bind phase ran, so an idle second
/// costs no line.
pub fn take_window() -> Option<BindPhaseWindow> {
    let binds = BINDS.swap(0, Ordering::Relaxed);
    let w = BindPhaseWindow {
        vertex_us: to_us(ACC[Part::VertexLoad as usize].swap(0, Ordering::Relaxed)),
        fragment_us: to_us(ACC[Part::FragmentLoad as usize].swap(0, Ordering::Relaxed)),
        attrs_us: to_us(ACC[Part::Attrs as usize].swap(0, Ordering::Relaxed)),
        binds,
    };
    (binds > 0).then_some(w)
}

/// Count one entry into the bind phase, so the parts have a denominator that
/// is theirs rather than `chain_phase`'s `chains`.
///
/// Separate from the spans because a draw with no vertex buffers still entered
/// the phase, and dividing by a count that only rose when a span opened would
/// read as though every draw loaded one.
pub fn note_bind() {
    BINDS.fetch_add(1, Ordering::Relaxed);
}

/// Charges the wall clock of one scope to one [`Part`].
///
/// A plain RAII span rather than the open/close phase machine
/// [`crate::runtime::chain_phase`] uses, because the bind phase is a straight
/// sequence: the parts are lexical scopes and nothing needs to switch phase
/// from inside a call several frames down.
pub struct Span {
    part: Part,
    started: Instant,
}

impl Span {
    pub fn open(part: Part) -> Self {
        Self {
            part,
            started: Instant::now(),
        }
    }
}

impl Drop for Span {
    fn drop(&mut self) {
        ACC[self.part as usize].fetch_add(charge_ns(self.started.elapsed()), Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An idle second emits nothing rather than a row of zeros.
    #[test]
    fn a_window_with_no_bind_is_none() {
        let _ = take_window();
        assert!(take_window().is_none());
    }

    /// Each part is charged only its own scope. The whole value of the split is
    /// that a vertex-load cost cannot hide inside the attribute column.
    #[test]
    fn each_part_is_charged_only_its_own_scope() {
        let _ = take_window();
        note_bind();
        {
            let _s = Span::open(Part::VertexLoad);
            std::thread::sleep(std::time::Duration::from_millis(4));
        }
        {
            let _s = Span::open(Part::Attrs);
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        let w = take_window().expect("a bind was noted");
        assert_eq!(w.binds, 1);
        assert!(w.vertex_us >= 3_000, "{w:?}");
        assert_eq!(w.fragment_us, 0, "{w:?}");
        assert!(w.attrs_us >= 1_000 && w.attrs_us < w.vertex_us, "{w:?}");
    }

    /// The denominator counts phases entered, not spans opened. A draw that
    /// binds nothing still entered the phase, and dividing by a span count
    /// would report every draw as having loaded a buffer.
    #[test]
    fn the_denominator_counts_phases_not_spans() {
        let _ = take_window();
        note_bind();
        note_bind();
        {
            let _s = Span::open(Part::FragmentLoad);
        }
        let w = take_window().expect("binds were noted");
        assert_eq!(w.binds, 2);
    }

    /// A large population of sub-microsecond spans has to sum to something.
    /// This is the shape the attribute walk is, and it is the whole reason
    /// [`crate::observe::phase_clock`] exists: an empty span here is a pair of
    /// `Instant::now()` calls, which a microsecond-truncating accumulator
    /// charges exactly zero.
    ///
    /// The threshold is measured rather than guessed. Nanosecond accumulation
    /// reads 302-308 µs over three runs (~15 ns a span); truncating
    /// accumulation reads 3, from the handful of spans a scheduling hiccup
    /// pushed over a microsecond. 100 sits a factor of three below the true
    /// reading and thirty above the false one, and load can only raise the
    /// true reading.
    #[test]
    fn twenty_thousand_sub_microsecond_spans_are_not_free() {
        let _ = take_window();
        for _ in 0..20_000 {
            note_bind();
            let _s = Span::open(Part::Attrs);
        }
        let w = take_window().expect("binds were noted");
        assert!(w.attrs_us > 100, "{w:?}");
    }

    /// Taking the window resets it, so the line is a rate and not a running
    /// total since boot.
    #[test]
    fn taking_the_window_resets_it() {
        let _ = take_window();
        note_bind();
        {
            let _s = Span::open(Part::VertexLoad);
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        assert!(take_window().is_some());
        assert!(take_window().is_none());
        note_bind();
        let w = take_window().expect("the second window emits");
        assert_eq!(w.vertex_us, 0, "{w:?}");
    }
}
