//! The split of [`crate::runtime::chain_phase`]'s `sampled_us`, over the same
//! window.
//!
//! # Why
//!
//! `sampled_us` is the largest column in `chain_phase` that nothing divides.
//! `binds_us` is larger and already splits into [`crate::runtime::bind_phase`];
//! `engine_us` is larger and already splits into `draw_phase`, whose twelve
//! phases sum to it exactly. On a driven x86/PCI second, coverage-instrumented:
//!
//! ```text
//! chain_phase  binds_us=115363  engine_us=88749  sampled_us=54106  store_us=14895
//! bind_phase   vertex_us=65972  fragment_us=43398  attrs_us=732
//! ```
//!
//! So the sampled phase is 17% of the draw and 3.6x the Store routing, and one
//! `sampled_us` bar could not choose between the four things inside it. That is
//! the same mistake `draw_phase`'s doc records having made once with `setup_us`,
//! and the same one `bind_phase` was built to undo for `binds_us`.
//!
//! A stale comment inside `push_tex` asked for exactly this division — it
//! described a "measure-only setup_tex sub-split" against a post-resolve stats
//! scan that no longer exists. This is that measurement, against the code as it
//! is now.
//!
//! # Why these split points
//!
//! Same rule the other two use: split where the fix changes, not where the code
//! happens to be indented.
//!
//! | part | what it brackets | what would fix it |
//! |---|---|---|
//! | [`Part::Lookup`] | `lookup_list_entry` + `resolve_texture_view`, per texture bind | caching the guest object-list walk and the type-8 view descriptor read |
//! | [`Part::Resolve`] | the attachment-alias branch and `resolve_sampled_source`, per texture bind | the sampled content cache and the gather witness |
//! | [`Part::Samplers`] | `load_vulkan_sampler` over the record's own sampler binds | a sampler object cache keyed on the guest sampler ref |
//! | [`Part::Reflect`] | the AIR constexpr static-sampler walk and the residual SPIR-V `sampler_bindings` scan | computing both at translate time and holding them in `m2v_cache` |
//!
//! The last two are one part on purpose. They are different data structures —
//! a small reflection `Vec` and a full SPIR-V word array — but they answer the
//! same question ("which sampler bindings has nothing provisioned yet") and they
//! have the same fix, which is to answer it once per translated shader instead
//! of once per draw. Splitting them would produce two bars a reader could not
//! act on separately.
//!
//! # What it does not report
//!
//! No `rest_us`, for the reason [`crate::runtime::bind_phase`] gives: the four
//! parts do **not** claim to sum to `sampled_us`. The phase also reads the
//! shader reflection for each bind's image dimensionality, folds a 1D image's
//! axes and pushes the engine resources, none of which is bracketed. Divide
//! against `chain_phase`'s `sampled_us` by hand; what the parts do not cover is
//! the answer to "is there a fifth".
//!
//! Like every phase census here it reports no loss. A slow resolve is not a
//! declined one, and the decline paths inside the phase keep their own typed
//! reasons. A bind that returns early from inside a span charges its remainder
//! to that span, because the commit is in `Drop` — deliberate, for the reason
//! `chain_phase` states: an exit is not a phase.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use crate::observe::phase_clock::{charge_ns, to_us};

/// The parts of the sampled phase that are worth telling apart.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Part {
    /// The per-bind guest reads: `objects::lookup_list_entry` for the texture's
    /// object-list entry, and `resolve_texture_view` for a type-8 view's
    /// channel remap.
    Lookup = 0,
    /// Where the bound texture's texels come from: the fragment
    /// attachment-alias branch, or `resolve_sampled_source`.
    Resolve = 1,
    /// `load_vulkan_sampler` over the record's vertex and fragment sampler
    /// binds.
    Samplers = 2,
    /// The two per-shader walks that provision what the guest did not name:
    /// AIR constexpr static samplers out of reflection, then the residual
    /// SPIR-V `sampler_bindings` scan.
    Reflect = 3,
}

const PARTS: usize = 4;

/// Nanoseconds, per [`crate::observe::phase_clock`]. The lookup is the reason:
/// it is a pair of map reads per bind at tens of thousands of binds a second,
/// so a microsecond-truncating accumulator would report it as free — the same
/// shape `bind_phase`'s attribute walk has.
static ACC: [AtomicU64; PARTS] = [const { AtomicU64::new(0) }; PARTS];
static SAMPLED: AtomicU64 = AtomicU64::new(0);

/// One window of the split, as taken by the per-second census.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SampledPhaseWindow {
    pub lookup_us: u64,
    pub resolve_us: u64,
    pub samplers_us: u64,
    pub reflect_us: u64,
    /// Sampled phases entered in the window — the denominator the four share.
    pub sampled: u64,
}

/// Take and clear the window. `None` when no sampled phase ran, so an idle
/// second costs no line.
pub fn take_window() -> Option<SampledPhaseWindow> {
    let sampled = SAMPLED.swap(0, Ordering::Relaxed);
    let w = SampledPhaseWindow {
        lookup_us: to_us(ACC[Part::Lookup as usize].swap(0, Ordering::Relaxed)),
        resolve_us: to_us(ACC[Part::Resolve as usize].swap(0, Ordering::Relaxed)),
        samplers_us: to_us(ACC[Part::Samplers as usize].swap(0, Ordering::Relaxed)),
        reflect_us: to_us(ACC[Part::Reflect as usize].swap(0, Ordering::Relaxed)),
        sampled,
    };
    (sampled > 0).then_some(w)
}

/// Count one entry into the sampled phase, so the parts have a denominator that
/// is theirs rather than `chain_phase`'s `chains`.
///
/// Separate from the spans for the reason `bind_phase::note_bind` gives: a draw
/// that samples nothing still entered the phase, and dividing by a count that
/// only rose when a span opened would report every draw as having bound a
/// texture.
pub fn note_sampled() {
    SAMPLED.fetch_add(1, Ordering::Relaxed);
}

/// Charges the wall clock of one scope to one [`Part`].
///
/// A plain RAII span rather than the open/close phase machine
/// [`crate::runtime::chain_phase`] uses, for the reason
/// [`crate::runtime::bind_phase::Span`] gives: the parts are lexical scopes and
/// nothing needs to switch part from inside a call several frames down.
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
    fn a_window_with_no_sampled_phase_is_none() {
        let _ = take_window();
        assert!(take_window().is_none());
    }

    /// Each part is charged only its own scope. The whole value of the split is
    /// that a resolve cost cannot hide inside the lookup column, which is what
    /// a single `sampled_us` bar let it do.
    #[test]
    fn each_part_is_charged_only_its_own_scope() {
        let _ = take_window();
        note_sampled();
        {
            let _s = Span::open(Part::Resolve);
            std::thread::sleep(std::time::Duration::from_millis(4));
        }
        {
            let _s = Span::open(Part::Lookup);
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        let w = take_window().expect("a sampled phase was noted");
        assert_eq!(w.sampled, 1);
        assert!(w.resolve_us >= 3_000, "{w:?}");
        assert_eq!(w.samplers_us, 0, "{w:?}");
        assert_eq!(w.reflect_us, 0, "{w:?}");
        assert!(w.lookup_us >= 1_000 && w.lookup_us < w.resolve_us, "{w:?}");
    }

    /// The denominator counts phases entered, not spans opened. A draw that
    /// binds no texture still entered the phase, and dividing by a span count
    /// would report every draw as having resolved one.
    #[test]
    fn the_denominator_counts_phases_not_spans() {
        let _ = take_window();
        note_sampled();
        note_sampled();
        {
            let _s = Span::open(Part::Resolve);
        }
        let w = take_window().expect("sampled phases were noted");
        assert_eq!(w.sampled, 2);
    }

    /// A texture bind opens two spans and a draw binds several, so this
    /// population is large and individually sub-microsecond. It has to sum to
    /// something, which is the whole reason [`crate::observe::phase_clock`]
    /// accumulates nanoseconds: an empty span is a pair of `Instant::now()`
    /// calls, and a microsecond-truncating accumulator charges that exactly
    /// zero however many times it happens.
    ///
    /// The threshold carries `bind_phase`'s measured basis rather than a fresh
    /// guess: the same span shape reads ~15 ns there, so 20 000 of them is a
    /// few hundred microseconds under nanosecond accumulation and single
    /// digits under truncation. 100 sits well below the true reading and well
    /// above the false one, and load can only raise the true one.
    #[test]
    fn twenty_thousand_sub_microsecond_spans_are_not_free() {
        let _ = take_window();
        for _ in 0..20_000 {
            note_sampled();
            let _s = Span::open(Part::Lookup);
        }
        let w = take_window().expect("sampled phases were noted");
        assert!(w.lookup_us > 100, "{w:?}");
    }

    /// Taking the window resets it, so the line is a rate and not a running
    /// total since boot.
    #[test]
    fn taking_the_window_resets_it() {
        let _ = take_window();
        note_sampled();
        {
            let _s = Span::open(Part::Samplers);
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        assert!(take_window().is_some());
        assert!(take_window().is_none());
        note_sampled();
        let w = take_window().expect("the second window emits");
        assert_eq!(w.samplers_us, 0, "{w:?}");
    }
}
