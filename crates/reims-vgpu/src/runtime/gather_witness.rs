//! Is the hypervisor's guest-write generation a sound "these texels did not
//! change" witness for the zero-copy sampled gathers?
//!
//! The three zero-copy sampled producers ([`super::metal_draw::vulkan`]'s
//! linear, type-11 and type-5 rails) hand the engine a
//! [`crate::backend::vulkan::engine::SampledSource::GuestRuns`], and the engine's
//! only byte-moving arm gathers the whole window out of guest RAM into a staging
//! buffer on every single bind. That arm has no content cache — measured on a
//! driven x86/PCI boot at 360 gathers and **842.4 MB per second**, both figures
//! repeating to the digit across eight consecutive windows, which is the shape of
//! the same unchanged content being re-read every frame rather than of a working
//! set that moves.
//!
//! The obvious fix is a cache keyed on a witness that says the guest has not
//! written those pages since the last gather, and the hypervisor dirty bitmap
//! ([`crate::runtime::host::HostOps::guest_write_gen`]) is the only witness for a
//! guest CPU store. But a *false* "unwritten" serves stale pixels, which is a
//! wrong frame that then persists — the failure mode that turned the screen black
//! once already. So the witness is measured before anything is built on it, and
//! this module is that measurement.
//!
//! # What it measures
//!
//! For each distinct sampled window it arms its own tracking token over exactly
//! the pages the gather reads, and on every later bind of that window it records
//! two independent answers:
//!
//! - the **generation**: did the hypervisor observe a guest store into these
//!   pages since the previous bind?
//! - the **content**: did the bytes actually change, by a full fold over the
//!   window?
//!
//! Crossing them gives a 2x2 whose cells are the whole result, reported through
//! [`crate::runtime::drain::note_store_route`]:
//!
//! | route | meaning |
//! |---|---|
//! | `gw_clean_same` | generation still, bytes still — a gather a cache would skip |
//! | `gw_clean_diff` | **generation still, bytes moved** — the witness is unsound |
//! | `gw_moved_same` | generation moved, bytes still — a missed skip, not an error |
//! | `gw_moved_diff` | generation moved, bytes moved — a real re-gather |
//! | `gw_unarmed` | no token, or a generation not yet readable — no answer |
//! | `gw_rearm` | the window's page set changed, so nothing to compare against |
//!
//! `gw_clean_diff` is the cell that decides it. A nonzero count means some writer
//! reaches these pages without the bitmap seeing it — the device's own deferred
//! writeback into guest RAM is the obvious candidate — and names the hole that
//! must be closed before a cache can key on this. A zero across a driven boot is
//! what licenses building one.
//!
//! `gw_clean_same_kb` accumulates the bytes the `gw_clean_same` cell would not
//! have moved, so the payoff is measured in the same units as the cost.
//!
//! # What it costs
//!
//! One extra cold read of every gathered window, because the content fold has to
//! see every byte — a sampled hash could miss exactly the change `gw_clean_diff`
//! exists to catch. The fold is a word-wise mix rather than a cryptographic
//! digest so the pass runs at memory speed; it defends against incidental change,
//! which is the only kind on offer here.

use std::collections::BTreeMap;

/// Which zero-copy sampled producer built the window.
///
/// The 2x2 below says whether the witness is sound; this says whose gathers it
/// would be sound *for*. The aggregate reading that opened this — 360 gathers and
/// 842.4 MB a second — is the sum over all three rails and has never been split,
/// so which of them to fix is not yet known.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GatherRail {
    /// Linear guest texture addressed through task GVA.
    Linear,
    /// Type-11 mapping-backed sampled bind.
    Type11,
    /// Type-5 serialized IOSurface plane view (the video path).
    Type5,
}

impl GatherRail {
    /// Census names for the rail's gather count and its gathered kilobytes.
    fn names(self) -> (&'static str, &'static str) {
        match self {
            Self::Linear => ("gw_rail_linear", "gw_rail_linear_kb"),
            Self::Type11 => ("gw_rail_t11", "gw_rail_t11_kb"),
            Self::Type5 => ("gw_rail_t5", "gw_rail_t5_kb"),
        }
    }
}

/// Which sampled window a witness entry describes.
///
/// The two shapes are the two ways the producers name a window: a task-GVA span
/// (the linear texture rail, which has no mapping) and a mapping-relative offset
/// (the type-11 and type-5 rails). Those two rails can name the same
/// `(mid, base_off)` for a single-plane surface, and that is harmless — same
/// mapping, same offset and same span is the same bytes.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum GatherKey {
    /// A texture window addressed through a task's GVA space.
    TaskGva { task_id: u32, gva: u64 },
    /// A window at a byte offset into a mapping's page list.
    Mapping { mid: u32, base_off: u64 },
}

/// What the last bind of one window observed.
#[derive(Clone, Debug)]
struct Entry {
    /// The exact page set the gather read, in window order. A change here means
    /// the window was re-pointed and there is nothing to compare against.
    gpas: Vec<u64>,
    /// Byte length of the window (a geometry change is also a re-point).
    span: u64,
    /// Tracking token armed over `gpas`, or 0 when the host refused one.
    token: u64,
    /// Generation read at the previous bind; 0 means "was not readable".
    gen: u64,
    /// Content fold over the window at the previous bind.
    fold: u128,
    /// [`crate::model::DeviceState::host_guest_write_seq`] at the previous bind,
    /// so a content change can be attributed to this device's own writes rather
    /// than to a guest store the bitmap missed.
    host_seq: u64,
    /// The same, narrowed to writes that could have landed in *this* window's
    /// mapping (see `DeviceState::host_wrote_mapping_seq`). The global count
    /// moves whenever any surface anywhere is written, so the two together say
    /// how much of the global count's invalidation is other people's work.
    scoped_seq: u64,
    /// Bind ordinal of the last sight of this window, for LRU eviction.
    last_seen: u64,
}

/// Per-device witness state: one entry per sampled window seen.
#[derive(Default, Debug)]
pub struct GatherWitness {
    entries: BTreeMap<GatherKey, Entry>,
    /// Monotonic bind ordinal, stamped into [`Entry::last_seen`].
    binds: u64,
}

/// Upper bound on tracked windows.
///
/// Not a memory bound — a hypervisor harvest bound. `reims_vgpu_dirty_harvest`
/// walks every page of every tracked set on the BQL thread at each register write
/// that hands the device work, so each armed window adds its page count to a cost
/// the whole VM pays. A driven Safari boot re-presents on the order of sixty
/// distinct sampled keys, so this sits just above the observed working set rather
/// than wherever memory would run out.
///
/// The first driven boot hit the cap twice during a hard scroll, so the working
/// set does reach it. Overflow evicts the least recently bound window rather than
/// dropping the map: a full drop costs a `gw_rearm` for every live window at once,
/// which is precisely the population whose answers are wanted.
const MAX_TRACKED_WINDOWS: usize = 256;

impl GatherWitness {
    /// Release every tracking token this witness armed.
    ///
    /// The tokens are host resources keyed to page sets; dropping the map without
    /// this would leak them for the life of the VM.
    pub fn release<M: crate::runtime::host::HostOps>(&mut self, host: &mut M) {
        for entry in self.entries.values() {
            if entry.token != 0 {
                host.untrack_guest_writes(entry.token);
            }
        }
        self.entries.clear();
    }

    /// Drop the least recently bound window, releasing its token.
    fn evict_oldest<M: crate::runtime::host::HostOps>(&mut self, host: &mut M) {
        let Some(victim) = self
            .entries
            .iter()
            .min_by_key(|(_, entry)| entry.last_seen)
            .map(|(key, _)| *key)
        else {
            return;
        };
        if let Some(entry) = self.entries.remove(&victim) {
            if entry.token != 0 {
                host.untrack_guest_writes(entry.token);
            }
        }
    }
}

/// Fold `span` bytes of a gathered window into a 128-bit value.
///
/// Word-wise rather than byte-wise, and two accumulators mixed differently so the
/// result is position-sensitive: a fold that only summed words would call any
/// permutation of a window unchanged, and a scrolled tile atlas is exactly a
/// permutation of itself.
///
/// # Safety
/// Every run's `host_ptr` must be a live mapping of at least `len` bytes — the
/// same precondition the gather itself relies on, read at the same point in the
/// draw.
unsafe fn fold_runs(runs: &[crate::backend::vulkan::engine::GuestRun], span: u64) -> u128 {
    let mut a: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut b: u64 = 0xc2b2_ae3d_27d4_eb4f;
    let mut remaining = span;
    for run in runs {
        if remaining == 0 {
            break;
        }
        let n = run.len.min(remaining) as usize;
        remaining -= n as u64;
        // SAFETY: caller's precondition — `host_ptr` is a stable RAMBlock alias
        // valid for at least `run.len` bytes, and `n <= run.len`.
        let bytes = unsafe { std::slice::from_raw_parts(run.host_ptr as *const u8, n) };
        let (words, tail) = bytes.split_at(n & !7);
        for chunk in words.chunks_exact(8) {
            let w = u64::from_le_bytes(chunk.try_into().expect("chunks_exact(8) yields 8 bytes"));
            a = (a ^ w).rotate_left(29).wrapping_mul(0x9e37_79b9_7f4a_7c15);
            b = b.rotate_left(7).wrapping_add(w ^ a);
        }
        for (i, &byte) in tail.iter().enumerate() {
            a ^= (byte as u64) << (8 * i);
        }
        // Fold the run boundary in so two windows with the same bytes split into
        // different runs are still distinguishable.
        b = b.wrapping_mul(0xff51_afd7_ed55_8ccd) ^ (n as u64);
    }
    ((a as u128) << 64) | b as u128
}

/// The resolved window one gather will read.
///
/// The pages and the host spans over them are both needed and neither implies
/// the other: guest-write tracking registers a page set, and the content fold
/// reads through the coalesced host pointers.
pub struct GatherWindow<'a> {
    /// Page-aligned guest addresses the window covers, in window order.
    pub gpas: &'a [u64],
    /// Coalesced host spans the gather reads, covering `span` bytes in order.
    pub runs: &'a [crate::backend::vulkan::engine::GuestRun],
    /// Byte length of the window.
    pub span: u64,
    /// Guest page size the `gpas` are expressed in.
    pub page_size: usize,
    /// [`crate::model::DeviceState::host_guest_write_seq`] as of this bind.
    pub host_seq: u64,
    /// `DeviceState::host_wrote_mapping_seq` for this window's mapping, or the
    /// global count where the window names no mapping.
    pub scoped_seq: u64,
}

/// What one bind of a window observed, crossing the hypervisor's answer with the
/// bytes'.
///
/// Returned rather than only counted so a test can drive the witness against a
/// host whose writes it controls, and so the census emission is one place instead
/// of five.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GatherVerdict {
    /// First sight of the window, or its page set / span moved. Nothing to
    /// compare against; the entry now holds this bind's answers.
    Rearmed,
    /// No readable generation on one side of the comparison, so the hypervisor
    /// says nothing. The fold answer still stands on its own.
    Unarmed { fold_same: bool },
    /// Generation still, bytes still — a gather a cache keyed on this witness
    /// would skip, correctly.
    ///
    /// The two flags score the two invalidation rules a cache could use against
    /// the host writes the bitmap cannot see. `global_quiet` is the rule "this
    /// device wrote nothing anywhere"; `scoped_quiet` is "wrote nothing that
    /// could have reached this window". Both are sound; scoped is strictly
    /// larger, and the gap between them is how much of the global rule's
    /// invalidation is other surfaces' work.
    CleanSame {
        global_quiet: bool,
        scoped_quiet: bool,
    },
    /// Generation still, bytes moved. The witness is unsound for this window:
    /// something wrote these pages without the bitmap seeing it.
    ///
    /// The two flags are what each candidate invalidation rule would have
    /// concluded, and each one reading quiet here is that rule being unsound.
    /// `host_wrote` false means this device wrote nowhere at all and the bytes
    /// moved anyway, which leaves only a guest store the bitmap did not see.
    /// `scoped_wrote` false means it wrote nothing that could have reached this
    /// window — the narrower claim, and the one a per-mapping rule rests on.
    ///
    /// The global count moves for every write including the scoped ones, so
    /// `!host_wrote` implies `!scoped_wrote`: the scoped rule's unsound set
    /// contains the global rule's.
    CleanDiff {
        host_wrote: bool,
        scoped_wrote: bool,
    },
    /// Generation moved, bytes still — a skip the witness gives up, not an error.
    MovedSame,
    /// Generation moved, bytes moved — a re-gather that had to happen.
    MovedDiff,
}

impl GatherVerdict {
    /// Did the window's bytes stay put, where that is known?
    ///
    /// `None` for [`Self::Rearmed`], which has no previous bind to compare with.
    fn fold_same(self) -> Option<bool> {
        match self {
            Self::Rearmed => None,
            Self::Unarmed { fold_same } => Some(fold_same),
            Self::CleanSame { .. } | Self::MovedSame => Some(true),
            Self::CleanDiff { .. } | Self::MovedDiff => Some(false),
        }
    }
}

/// Record one zero-copy sampled gather against the guest-write witness, and
/// report it to the census.
///
/// Called from the producers with the window already resolved, so it adds a
/// page-set compare and one content fold and changes no behaviour.
pub fn note_gather<M: crate::runtime::host::HostOps>(
    witness: &mut GatherWitness,
    host: &mut M,
    rail: GatherRail,
    key: GatherKey,
    window: GatherWindow<'_>,
) {
    use crate::runtime::drain::{note_store_route, note_store_route_n};

    let span = window.span;
    let (rail_count, rail_kb) = rail.names();
    note_store_route(rail_count);
    note_store_route_n(rail_kb, span / 1024);

    let verdict = observe(witness, host, key, window);

    // The fold answer on its own, independent of whether a token was armed: this
    // is the ceiling on what *any* content cache could skip, and unlike the
    // cross-tab below it is available for every re-presented window.
    match verdict.fold_same() {
        Some(true) => {
            note_store_route("gw_fold_same");
            note_store_route_n("gw_fold_same_kb", span / 1024);
        }
        Some(false) => note_store_route("gw_fold_diff"),
        None => {}
    }
    match verdict {
        GatherVerdict::Rearmed => note_store_route("gw_rearm"),
        GatherVerdict::Unarmed { .. } => note_store_route("gw_unarmed"),
        GatherVerdict::CleanSame {
            global_quiet,
            scoped_quiet,
        } => {
            note_store_route("gw_clean_same");
            note_store_route_n("gw_clean_same_kb", span / 1024);
            if global_quiet {
                note_store_route("gw_hit_global");
                note_store_route_n("gw_hit_global_kb", span / 1024);
            }
            if scoped_quiet {
                note_store_route("gw_hit_scoped");
                note_store_route_n("gw_hit_scoped_kb", span / 1024);
            }
        }
        // Three names rather than one, because they condemn different rules.
        GatherVerdict::CleanDiff {
            host_wrote: false, ..
        } => note_store_route("gw_clean_diff"),
        GatherVerdict::CleanDiff {
            scoped_wrote: false,
            ..
        } => note_store_route("gw_clean_diff_scoped_quiet"),
        GatherVerdict::CleanDiff { .. } => note_store_route("gw_clean_diff_host_wrote"),
        GatherVerdict::MovedSame => note_store_route("gw_moved_same"),
        GatherVerdict::MovedDiff => note_store_route("gw_moved_diff"),
    }
}

/// The witness itself: fold the window, compare both answers against the last
/// bind of the same window, and leave the entry describing this one.
fn observe<M: crate::runtime::host::HostOps>(
    witness: &mut GatherWitness,
    host: &mut M,
    key: GatherKey,
    window: GatherWindow<'_>,
) -> GatherVerdict {
    let GatherWindow {
        gpas,
        runs,
        span,
        page_size,
        host_seq,
        scoped_seq,
    } = window;

    // SAFETY: `runs` describe the window this draw is about to gather from, so
    // their pointers are live here for the same reason they are live there.
    let fold = unsafe { fold_runs(runs, span) };

    witness.binds = witness.binds.wrapping_add(1);
    while witness.entries.len() >= MAX_TRACKED_WINDOWS && !witness.entries.contains_key(&key) {
        crate::runtime::drain::note_store_route("gw_window_overflow");
        witness.evict_oldest(host);
    }

    let stale = match witness.entries.get(&key) {
        Some(entry) => entry.gpas != gpas || entry.span != span,
        None => true,
    };
    if stale {
        if let Some(old) = witness.entries.remove(&key) {
            if old.token != 0 {
                host.untrack_guest_writes(old.token);
            }
        }
        let token = host.track_guest_writes(gpas, page_size).unwrap_or(0);
        let gen = if token == 0 {
            0
        } else {
            host.guest_write_gen(token).unwrap_or(0)
        };
        witness.entries.insert(
            key,
            Entry {
                gpas: gpas.to_vec(),
                span,
                token,
                gen,
                fold,
                host_seq,
                scoped_seq,
                last_seen: witness.binds,
            },
        );
        return GatherVerdict::Rearmed;
    }

    let entry = witness
        .entries
        .get_mut(&key)
        .expect("the stale branch above returns for every absent key");
    let fold_same = fold == entry.fold;
    let gen = if entry.token == 0 {
        0
    } else {
        host.guest_write_gen(entry.token).unwrap_or(0)
    };
    // A generation of 0 on either side is "cannot tell": the token is unarmed,
    // was released with its pages, or has not survived the two harvests the
    // dirty adapter needs before it can answer at all.
    let verdict = if gen == 0 || entry.gen == 0 {
        GatherVerdict::Unarmed { fold_same }
    } else {
        match (gen == entry.gen, fold_same) {
            (true, true) => GatherVerdict::CleanSame {
                global_quiet: host_seq == entry.host_seq,
                scoped_quiet: scoped_seq == entry.scoped_seq,
            },
            (true, false) => GatherVerdict::CleanDiff {
                host_wrote: host_seq != entry.host_seq,
                scoped_wrote: scoped_seq != entry.scoped_seq,
            },
            (false, true) => GatherVerdict::MovedSame,
            (false, false) => GatherVerdict::MovedDiff,
        }
    };
    entry.gen = gen;
    entry.fold = fold;
    entry.host_seq = host_seq;
    entry.scoped_seq = scoped_seq;
    entry.last_seen = witness.binds;
    verdict
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::vulkan::engine::GuestRun;

    const KEY: GatherKey = GatherKey::Mapping {
        mid: 11,
        base_off: 0,
    };
    const PAGE: usize = 4096;
    const GPAS: [u64; 1] = [8 * PAGE as u64];

    /// A one-page window over `runs`, at `gpas`, with no host write since the
    /// device started.
    fn one_page<'a>(gpas: &'a [u64], runs: &'a [GuestRun]) -> GatherWindow<'a> {
        host_wrote_since(gpas, runs, 0)
    }

    /// A window that saw no host write at either scope.
    const HOST_QUIET: GatherVerdict = GatherVerdict::CleanSame {
        global_quiet: true,
        scoped_quiet: true,
    };

    /// The same window, presented as of host-write sequence `host_seq`.
    fn host_wrote_since<'a>(
        gpas: &'a [u64],
        runs: &'a [GuestRun],
        host_seq: u64,
    ) -> GatherWindow<'a> {
        at_seqs(gpas, runs, host_seq, host_seq)
    }

    /// The same, with the two host-write counts set independently — which is how
    /// a write to some *other* mapping looks.
    fn at_seqs<'a>(
        gpas: &'a [u64],
        runs: &'a [GuestRun],
        host_seq: u64,
        scoped_seq: u64,
    ) -> GatherWindow<'a> {
        GatherWindow {
            gpas,
            runs,
            span: PAGE as u64,
            page_size: PAGE,
            host_seq,
            scoped_seq,
        }
    }

    fn run_over(buf: &[u8]) -> GuestRun {
        GuestRun {
            host_ptr: buf.as_ptr() as usize,
            len: buf.len() as u64,
        }
    }

    #[test]
    fn the_fold_sees_a_single_changed_byte_anywhere_in_the_window() {
        let mut buf = vec![7u8; 4096 + 3];
        let base = unsafe { fold_runs(&[run_over(&buf)], buf.len() as u64) };
        for at in [0usize, 1, 8, 1000, 4095, 4096, 4098] {
            let saved = buf[at];
            buf[at] ^= 0x40;
            let moved = unsafe { fold_runs(&[run_over(&buf)], buf.len() as u64) };
            assert_ne!(base, moved, "a flipped byte at {at} folded the same");
            buf[at] = saved;
        }
        assert_eq!(base, unsafe {
            fold_runs(&[run_over(&buf)], buf.len() as u64)
        });
    }

    #[test]
    fn the_fold_is_position_sensitive_so_a_permuted_window_is_not_unchanged() {
        // Distinct bytes at the two swapped indices, or the "permutation" is the
        // identity and the test proves nothing.
        let a: Vec<u8> = (0..512u32).map(|i| (i / 2) as u8).collect();
        let mut b = a.clone();
        assert_ne!(a[0], a[256]);
        b.swap(0, 256);
        assert_ne!(
            unsafe { fold_runs(&[run_over(&a)], a.len() as u64) },
            unsafe { fold_runs(&[run_over(&b)], b.len() as u64) },
            "swapping two words folded the same, so the fold sums rather than orders"
        );
    }

    /// A window whose bytes and pages both stand still, bound twice: the whole
    /// point of the exercise, and the cell whose count says what a cache would
    /// save.
    #[test]
    fn a_window_nothing_writes_reads_clean_and_unchanged_on_the_second_bind() {
        let mut host = crate::runtime::host::FakeHost::new();
        let mut w = GatherWitness::default();
        let buf = vec![0xa5u8; PAGE];
        let runs = [run_over(&buf)];
        assert_eq!(
            observe(&mut w, &mut host, KEY, one_page(&GPAS, &runs)),
            GatherVerdict::Rearmed,
            "first sight has nothing to compare against"
        );
        assert_eq!(
            observe(&mut w, &mut host, KEY, one_page(&GPAS, &runs)),
            HOST_QUIET
        );
    }

    /// The hypervisor saw the store, and so did the fold. Both halves agree, which
    /// is what a sound witness looks like on content that really moved.
    #[test]
    fn a_guest_store_into_the_window_moves_the_generation_and_the_fold_together() {
        let mut host = crate::runtime::host::FakeHost::new();
        let mut w = GatherWitness::default();
        let mut buf = vec![0xa5u8; PAGE];
        let runs = [run_over(&buf)];
        assert_eq!(
            observe(&mut w, &mut host, KEY, one_page(&GPAS, &runs)),
            GatherVerdict::Rearmed
        );
        buf[100] ^= 0xff;
        host.guest_wrote_page(GPAS[0]);
        assert_eq!(
            observe(&mut w, &mut host, KEY, one_page(&GPAS, &runs)),
            GatherVerdict::MovedDiff
        );
    }

    /// The unsound cell, produced deliberately: bytes changed under a page the
    /// hypervisor never saw written. This is the shape a host-side writer into
    /// guest RAM would make, and the reason the probe exists — so if a driven boot
    /// ever reports `gw_clean_diff`, this test says what that means.
    #[test]
    fn bytes_that_move_without_the_hypervisor_seeing_it_read_as_clean_and_different() {
        let mut host = crate::runtime::host::FakeHost::new();
        let mut w = GatherWitness::default();
        let mut buf = vec![0xa5u8; PAGE];
        let runs = [run_over(&buf)];
        assert_eq!(
            observe(&mut w, &mut host, KEY, one_page(&GPAS, &runs)),
            GatherVerdict::Rearmed
        );
        // No `guest_wrote_page`: the bytes move with the bitmap none the wiser.
        buf[7] ^= 0xff;
        assert_eq!(
            observe(&mut w, &mut host, KEY, one_page(&GPAS, &runs)),
            GatherVerdict::CleanDiff {
                host_wrote: false,
                scoped_wrote: false
            },
            "no host write moved the sequence, so this is a guest store the \
             bitmap did not see"
        );
    }

    /// The cell that would condemn a per-mapping invalidation rule: this device
    /// wrote *somewhere*, but nothing that could have reached this window, and
    /// the bytes moved anyway.
    ///
    /// A scoped rule reads that as quiet and serves the stale copy. It is a
    /// distinct finding from the global cell — a global rule would have
    /// invalidated here and been right — so the two are counted apart rather
    /// than folded into one "unsound" number that cannot say which rule to build.
    #[test]
    fn bytes_moving_while_only_another_mapping_was_written_condemns_the_scoped_rule() {
        let mut host = crate::runtime::host::FakeHost::new();
        let mut w = GatherWitness::default();
        let mut buf = vec![0xa5u8; PAGE];
        let runs = [run_over(&buf)];
        assert_eq!(
            observe(&mut w, &mut host, KEY, at_seqs(&GPAS, &runs, 1, 1)),
            GatherVerdict::Rearmed
        );
        // The global count moves; this mapping's does not.
        buf[9] ^= 0xff;
        assert_eq!(
            observe(&mut w, &mut host, KEY, at_seqs(&GPAS, &runs, 2, 1)),
            GatherVerdict::CleanDiff {
                host_wrote: true,
                scoped_wrote: false
            }
        );
    }

    /// A host that cannot observe guest writes must never produce a clean verdict,
    /// however still the bytes are. Fail closed: `Unarmed` carries the fold answer
    /// for the census and vouches for nothing.
    #[test]
    fn a_host_that_cannot_watch_guest_writes_never_reads_clean() {
        let mut host = crate::runtime::host::FakeHost::new();
        host.guest_writes_unobservable = true;
        let mut w = GatherWitness::default();
        let buf = vec![0xa5u8; PAGE];
        let runs = [run_over(&buf)];
        assert_eq!(
            observe(&mut w, &mut host, KEY, one_page(&GPAS, &runs)),
            GatherVerdict::Rearmed
        );
        assert_eq!(
            observe(&mut w, &mut host, KEY, one_page(&GPAS, &runs)),
            GatherVerdict::Unarmed { fold_same: true }
        );
    }

    /// A window re-pointed at different pages has no predecessor, even though its
    /// key repeats. Comparing across the move would compare two different surfaces.
    #[test]
    fn a_window_whose_pages_move_rearms_rather_than_comparing_across_the_move() {
        let mut host = crate::runtime::host::FakeHost::new();
        let mut w = GatherWitness::default();
        let buf = vec![0xa5u8; PAGE];
        let runs = [run_over(&buf)];
        let moved = [9 * PAGE as u64];
        assert_eq!(
            observe(&mut w, &mut host, KEY, one_page(&GPAS, &runs)),
            GatherVerdict::Rearmed
        );
        assert_eq!(
            observe(&mut w, &mut host, KEY, one_page(&moved, &runs)),
            GatherVerdict::Rearmed,
            "same key, different pages: nothing to compare"
        );
        assert_eq!(
            observe(&mut w, &mut host, KEY, one_page(&moved, &runs)),
            HOST_QUIET
        );
    }

    #[test]
    fn the_fold_stops_at_span_even_when_the_runs_are_longer() {
        let buf = vec![3u8; 256];
        let short = unsafe { fold_runs(&[run_over(&buf)], 64) };
        let head = vec![3u8; 64];
        assert_eq!(short, unsafe { fold_runs(&[run_over(&head)], 64) });
    }
}
