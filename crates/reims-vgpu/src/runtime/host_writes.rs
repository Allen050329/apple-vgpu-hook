//! Which guest pages this device has written, and when.
//!
//! The hypervisor's dirty bitmap witnesses guest CPU stores and nothing else, so
//! a host-side copy vouched for by "the guest has not written since I looked" can
//! still be stale — because *we* wrote. This is the missing half of that witness.
//!
//! # Why pages and not mappings
//!
//! Three candidate rules were scored against a full content fold before this one
//! was built, each by its own census counter. Those counters are gone with the
//! rules they scored — [`crate::runtime::gather_witness`] takes only the page-exact answer
//! now — so the names below are what the readings were called at the time and are
//! not greppable in a current log.
//!
//! A per-mapping count was measured first and it leaks. One driven boot read
//! fifteen binds where the sampled window's own mapping had not been written, the
//! guest had not written, and the bytes moved anyway. Guest pages are reachable
//! under more than one mapping id — `deferred_alias_pages` is the rail built for
//! exactly that — so "mapping 12 was not written" is not "these pages were not
//! written", and a cache keyed on the former serves stale pixels fifteen times a
//! minute.
//!
//! The same boot read zero counterexamples for a *global* count, which moves for
//! every write anywhere — because it moves for every write including the ones a
//! narrower rule fails to attribute. Sound, and it invalidates a texture because
//! an unrelated scanout was composited.
//!
//! # Where it stands
//!
//! Once every writer records here — which took a second pass, because
//! `map_fresh_span_within`'s callers write through a raw alias and were invisible
//! to a hand-picked list of call sites — a driven boot reads **zero** binds where
//! the page-exact rule vouched and the bytes had moved, alongside zero for both
//! wider rules. Of the binds where the guest was quiet and the bytes were
//! identical, this rule serves 93 %; the rest are windows whose page set had just
//! moved.
//!
//! That measurement is what the fold is still there for. It runs on one bind in
//! [`crate::runtime::gather_witness::AUDIT_STRIDE`] rather than all of them, and its
//! counterexample cell is `gw_audit_unsound`: a standing alarm on the rule this
//! module exists to make sound, rather than the per-bind decision it began as.
//!
//! What that licenses is a cache over the zero-copy sampled gathers, valid iff
//! the hypervisor's guest generation has not moved **and** this says the pages
//! were not written. Neither half is sufficient alone and the measurements above
//! are what say so, rather than an argument that they ought to be.
//!
//! Built and measured live on a driven x86/PCI boot: **5852 gathers skipped
//! against 4167 taken, 14.25 GB not read against 4.56 GB read — 75.8 % of the
//! rail's bytes gone** — with all three unsound cells still zero and a Wikipedia
//! page rendering correctly under scroll.
//!
//! # Shape
//!
//! A ring of recent writes rather than a per-page map. A per-page map costs the
//! writer one insert per page written — a 1920x1080 scanout is ~2000 of them — and
//! the reader one lookup per page read, on every bind. The ring costs the writer
//! O(1) and costs the reader nothing at all in the common case, because between
//! two binds of the same window (~8 ms apart) there is usually no host write to
//! compare against.
//!
//! Everything here fails closed. A write that cannot name its pages, a ring that
//! has dropped the entry a reader is asking about, and a mapping re-pointed since
//! the write that named it all answer "assume written".

use std::collections::BTreeSet;

/// What one host write touched.
#[derive(Clone, Debug)]
enum Wrote {
    /// Every page of a mapping, as its page list stood at `map_generation`.
    ///
    /// The pages are resolved at read time rather than copied at write time,
    /// which is what makes recording a write O(1). `map_generation` is what makes
    /// that safe: a mapping re-pointed since the write has a different page list,
    /// and testing against the new one would answer about pages the write never
    /// touched.
    Mapping { mid: u32, map_generation: u32 },
    /// An explicit page set, for writers that walk the guest page tables and so
    /// name no mapping at all.
    Pages(Vec<u64>),
    /// A writer that could not say. Every reader older than this must assume its
    /// pages were among them.
    Unknown,
}

/// Recent host writes into guest RAM, newest last.
#[derive(Default, Debug)]
pub struct HostWrites {
    /// Monotonic stamp; a reader records the value current at its own read and
    /// asks later whether anything newer touched its pages. Never 0 once any
    /// write has happened, so 0 is usable as "never looked".
    epoch: u64,
    recent: std::collections::VecDeque<(u64, Wrote)>,
    /// Oldest mark the ring can still answer for.
    ///
    /// A reader with mark `s` asks about writes with epoch **greater than** `s`,
    /// so the ring can answer it exactly when it holds every such write — that
    /// is, when `s` is at least one below the oldest epoch retained. A reader
    /// below that is asking about writes that have been dropped, and gets
    /// "assume written".
    answers_from: u64,
}

/// How many writes the ring remembers.
///
/// Bounds the reader's scan, not memory: a `Mapping` entry is two words and
/// resolves its pages on demand. Sized well above the number of host writes that
/// can fall between two binds of the same sampled window — one driven boot read
/// ~28 host writes a second against ~330 gathers a second, so the usual answer is
/// zero entries to scan and the tail is single digits.
const RING: usize = 64;

impl HostWrites {
    /// The stamp a reader records beside a copy it has just taken.
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Record a write covering every page of `mapping_id`, as its page list
    /// stands now.
    pub fn note_mapping(&mut self, mapping_id: u32, map_generation: u32) {
        self.push(Wrote::Mapping {
            mid: mapping_id,
            map_generation,
        });
    }

    /// Record a write covering exactly `pages` (page-aligned guest addresses).
    pub fn note_pages(&mut self, pages: Vec<u64>) {
        self.push(Wrote::Pages(pages));
    }

    /// Record a write whose pages are not known. Invalidates every reader older
    /// than it.
    pub fn note_unknown(&mut self) {
        self.push(Wrote::Unknown);
    }

    fn push(&mut self, what: Wrote) {
        self.epoch = self.epoch.wrapping_add(1);
        self.recent.push_back((self.epoch, what));
        while self.recent.len() > RING {
            self.recent.pop_front();
        }
        self.answers_from = self
            .recent
            .front()
            .map(|(epoch, _)| epoch.saturating_sub(1))
            .unwrap_or(self.epoch);
    }

    /// Has this device written any of `pages` since `since`?
    ///
    /// `since` is a value previously returned by [`Self::epoch`]. Answers `true`
    /// for everything it cannot decide: a dropped ring entry, an unknown write,
    /// or a mapping whose page list has moved since the write named it.
    pub fn wrote_any_since(
        &self,
        state: &crate::model::DeviceState,
        since: u64,
        pages: &[u64],
    ) -> bool {
        if since < self.answers_from {
            return true;
        }
        let mut asked: Option<BTreeSet<u64>> = None;
        for (epoch, what) in self.recent.iter().rev() {
            if *epoch <= since {
                break;
            }
            let want = asked.get_or_insert_with(|| pages.iter().copied().collect());
            match what {
                Wrote::Unknown => return true,
                Wrote::Pages(written) => {
                    if written.iter().any(|p| want.contains(p)) {
                        return true;
                    }
                }
                Wrote::Mapping {
                    mid,
                    map_generation,
                } => {
                    let Some(m) = state.mappings.get(mid) else {
                        // The mapping is gone, so its page list cannot be
                        // reconstructed to be ruled out.
                        return true;
                    };
                    if m.map_generation != *map_generation {
                        return true;
                    }
                    let shift = state.page_shift;
                    if m.page_entries.iter().any(|&e| {
                        crate::contract::iosurface_pages::entry_gpa_shift(e, shift)
                            .is_some_and(|gpa| want.contains(&gpa))
                    }) {
                        return true;
                    }
                }
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const P: u64 = 4096;

    #[test]
    fn a_write_to_other_pages_leaves_this_window_quiet() {
        let state = crate::model::DeviceState::new(
            crate::model::DeviceId(1),
            crate::model::PAGE_SHIFT_X86,
        );
        let mut w = HostWrites::default();
        let mark = w.epoch();
        w.note_pages(vec![9 * P, 10 * P]);
        assert!(!w.wrote_any_since(&state, mark, &[3 * P, 4 * P]));
        assert!(w.wrote_any_since(&state, mark, &[4 * P, 10 * P]));
    }

    /// The reader asks about writes *after* its own mark, so the write it
    /// already accounted for must not invalidate it forever.
    #[test]
    fn a_write_the_reader_already_saw_does_not_answer_for_a_later_mark() {
        let state = crate::model::DeviceState::new(
            crate::model::DeviceId(1),
            crate::model::PAGE_SHIFT_X86,
        );
        let mut w = HostWrites::default();
        w.note_pages(vec![4 * P]);
        let after = w.epoch();
        assert!(!w.wrote_any_since(&state, after, &[4 * P]));
        w.note_pages(vec![4 * P]);
        assert!(w.wrote_any_since(&state, after, &[4 * P]));
    }

    /// A write that could not name its pages must invalidate everything, and a
    /// reader older than the ring must be told the ring cannot answer.
    #[test]
    fn what_the_ring_cannot_decide_reads_as_written() {
        let state = crate::model::DeviceState::new(
            crate::model::DeviceId(1),
            crate::model::PAGE_SHIFT_X86,
        );
        let mut w = HostWrites::default();
        let mark = w.epoch();
        w.note_unknown();
        assert!(w.wrote_any_since(&state, mark, &[999 * P]));

        let mut w = HostWrites::default();
        let stale = w.epoch();
        for i in 0..(RING as u64 + 5) {
            w.note_pages(vec![(100 + i) * P]);
        }
        assert!(
            w.wrote_any_since(&state, stale, &[3 * P]),
            "a mark older than the ring must not be answered from what is left of it"
        );
        let fresh = w.epoch();
        assert!(!w.wrote_any_since(&state, fresh, &[3 * P]));
    }

    /// A mapping-named write is resolved through the mapping's live page list, so
    /// a mapping re-pointed since must not be tested against its new pages.
    #[test]
    fn a_mapping_re_pointed_since_the_write_cannot_be_ruled_out() {
        use crate::contract::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
        let mut state = crate::model::DeviceState::new(
            crate::model::DeviceId(1),
            crate::model::PAGE_SHIFT_X86,
        );
        state.map_surface(4);
        state.attach_mapping_internal(4, 0);
        let m = state.mappings.get_mut(&4).expect("just mapped");
        m.page_entries = vec![(7u32 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
        let generation = m.map_generation;

        let mut w = HostWrites::default();
        let mark = w.epoch();
        w.note_mapping(4, generation);
        assert!(w.wrote_any_since(&state, mark, &[7 * P]));
        assert!(!w.wrote_any_since(&state, mark, &[8 * P]));

        // Re-point the mapping at a page the write never touched. The write's
        // page set is no longer reconstructible, so it can rule out nothing.
        let m = state.mappings.get_mut(&4).expect("still mapped");
        m.page_entries = vec![(8u32 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
        m.map_generation = generation.wrapping_add(1);
        assert!(
            w.wrote_any_since(&state, mark, &[3 * P]),
            "a write named by a mapping that has since moved must not be ruled out"
        );
    }
}
