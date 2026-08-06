//! Always-on create/alloc and cache hit/miss counters (reuse-gate proxies).
//!
//! # The vocabulary is declared once
//!
//! [`engine_counters!`] takes the counter names and generates the five things
//! that used to spell them out separately: the atomic [`EngineCounters`], the
//! plain-`u64` [`CounterSnapshot`], and the three whole-vocabulary walks
//! [`EngineCounters::snapshot`], [`EngineCounters::reset`] and
//! [`CounterSnapshot::delta_since`].
//!
//! Writing seventy names five times is how a counter silently stops working, and
//! neither failure mode is a compile error or a log line:
//!
//! * missing from `reset` — the counter reports a lifetime total into a reader
//!   that asked for a window, so a per-second rate reads as monotonically rising;
//! * missing from `delta_since` — the field reads **zero in every delta**, which
//!   is indistinguishable from "this path never ran". That is the
//!   "an event count is not a state" trap in `AGENTS.md` with the count itself
//!   broken.
//!
//! All five lists were checked against each other before this collapse and all
//! five agreed, so the macro changes no behaviour. What it changes is that they
//! can no longer disagree.
//!
//! The three groups are a real distinction, not a formatting one. `windowed` is
//! zeroed by `reset()`; `cumulative` deliberately survives it, because a
//! device-loss count is a fact about the boot and not about the measurement
//! window, and only `reset_all()` clears it; `pool_sourced` has no atomic at all
//! and is merged in from `ResourcePools` by `engine::counter_snapshot`.
//!
//! # A field with no named reader is not dead
//!
//! [`CounterSnapshot`] is consumed only by the integration tests in `tests/`.
//! No product code reads it and no log line emits it, so a sweep for "fields
//! nobody references" reports most of this struct. Twenty-seven of the
//! seventy-one came back that way. Do not act on that sweep: the struct derives
//! `Debug` and every assertion in those tests prints the *whole* snapshot on
//! failure (`"...: {d:?}"`), so the unasserted fields are the diagnostic context
//! that makes a failing assertion readable.
//!
//! That is not hypothetical. The ring-wrap defect in the sampled content cache
//! was diagnosed from `sampled_free_allocs` and `sampled_recycle_cap_drops` in
//! one such dump — neither is asserted by any test, and both named the
//! mechanism. Deleting them would have cost more than the lines saved.
//!
//! The exception, and the only thing removed on those grounds, is a field **no
//! code path increments**: it is zero by construction, so it carries no
//! information in a dump either. `seed_imports` and `target_stale_import` were
//! removed for exactly that. Before deleting any other field here, check that
//! something can still make it nonzero.
//!
//! Adding one is still governed by the census-versus-decline rule in
//! `AGENTS.md`: a counter must not be the only record that guest work was lost.
//! These are reuse and cache proxies — tallies of successful work — and the
//! refusal paths they sit near report themselves with typed declines.

use std::sync::atomic::{AtomicU64, Ordering};

/// How much of its render target one draw could have written.
///
/// The standing instrument for "would bounding what a flush copies pay
/// anything?" — the question `runtime::storage_flush` carried as its largest
/// named lever until it was built, measured at zero, and removed. It is a
/// property of the *guest's* draws and not of any rail here, which is why it
/// outlives the rail: the answer changes with the workload, and nothing else in
/// this device can be read for it.
///
/// See [`EngineCounters::note_draw_coverage`] for the arithmetic that turns
/// these into a verdict.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DrawCoverage {
    /// The pass did not load the target, so it rewrote the whole attachment
    /// before the draw began — a CLEAR load action, or a whole-frame CPU seed
    /// standing in for one.
    Full,
    /// The pass loaded the target and bound a scissor covering all of it. The
    /// draw could have written any texel.
    LoadedFullScissor,
    /// The pass loaded the target and bound a scissor smaller than it. The only
    /// arm whose writes are bounded by anything.
    LoadedPartialScissor,
}

/// Declare the engine counter vocabulary once; see the module docs for why.
///
/// Doc comments written on a name here land on *both* the atomic field and its
/// snapshot field, which is why the snapshot no longer has to repeat them.
macro_rules! engine_counters {
    (
        windowed { $($(#[$wm:meta])* $win:ident,)* }
        cumulative { $($(#[$cm:meta])* $cum:ident,)* }
        pool_sourced { $($(#[$pm:meta])* $pool:ident,)* }
    ) => {
        /// Process-wide product-path counters (resettable for tests).
        #[derive(Debug, Default)]
        pub struct EngineCounters {
            $($(#[$wm])* pub $win: AtomicU64,)*
            $($(#[$cm])* pub $cum: AtomicU64,)*
        }

        /// One reading of [`EngineCounters`], plus the pool-owned tallies
        /// `engine::counter_snapshot` merges in.
        #[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
        pub struct CounterSnapshot {
            $($(#[$wm])* pub $win: u64,)*
            $($(#[$cm])* pub $cum: u64,)*
            $($(#[$pm])* pub $pool: u64,)*
        }

        impl EngineCounters {
            /// Read every counter at once. The `pool_sourced` fields have no
            /// atomic here and stay zero; `engine::counter_snapshot` fills them
            /// from `ResourcePools` immediately after calling this.
            pub fn snapshot(&self) -> CounterSnapshot {
                CounterSnapshot {
                    $($win: self.$win.load(Ordering::Relaxed),)*
                    $($cum: self.$cum.load(Ordering::Relaxed),)*
                    $($pool: 0,)*
                }
            }

            /// Zero the windowed counters, leaving the cumulative ones alone.
            pub fn reset(&self) {
                $(self.$win.store(0, Ordering::Relaxed);)*
            }

            /// Zero everything, including the counters `reset` preserves.
            pub fn reset_all(&self) {
                self.reset();
                $(self.$cum.store(0, Ordering::Relaxed);)*
            }
        }

        impl CounterSnapshot {
            /// This reading minus an earlier one, field by field. Saturating
            /// because a `reset` between the two readings makes `earlier`
            /// larger, and a window of "no work" must read 0 rather than wrap.
            pub fn delta_since(&self, earlier: &CounterSnapshot) -> CounterSnapshot {
                CounterSnapshot {
                    $($win: self.$win.saturating_sub(earlier.$win),)*
                    $($cum: self.$cum.saturating_sub(earlier.$cum),)*
                    $($pool: self.$pool.saturating_sub(earlier.$pool),)*
                }
            }
        }
    };
}

engine_counters! {
    windowed {
        creates,
        allocs,
        shader_hits,
        shader_misses,
        layout_hits,
        layout_misses,
        pass_hits,
        pass_misses,
        pipeline_hits,
        pipeline_misses,
        sampler_hits,
        sampler_misses,

        // --- compute ---
        compute_pipeline_hits,
        compute_pipeline_misses,
        dispatches,
        fence_timeouts,
        /// Compute sampled-image bytes staged for host→device upload.
        compute_sampled_uploads,
        compute_sampled_upload_bytes,
        /// Compute storage-image seed bytes staged for host→device upload.
        compute_storage_seed_uploads,
        compute_storage_seed_upload_bytes,
        /// Sampled inputs seeded by a device-local copy of a resident storage
        /// image (copy-on-sample) — bytes are the elided host upload size.
        compute_sampled_resident_copies,
        compute_sampled_resident_copy_bytes,
        /// Compute storage images whose post-dispatch readback was deferred —
        /// the pinned resident stays authoritative; bytes are the elided
        /// device→host readback size (the CPU writeback of the same size is
        /// elided too).
        compute_deferred_writebacks,
        compute_deferred_writeback_bytes,
        /// Deferred-flush reads (read_resident_storage): the on-access GPU→host
        /// copy that lands deferred content in guest pages.
        compute_deferred_flushes,
        compute_deferred_flush_bytes,

        // --- residency / oracle I/O ---
        /// Device→host copies taken as the tail of a draw or a compute dispatch,
        /// i.e. work a submission did for itself.
        ///
        /// Deliberately *not* pooled with `target_reads`. A composite Store that
        /// takes `skip_readback` moves its copy from here to there rather than
        /// deleting it, so one number over both populations cannot say whether the
        /// deferral worked — it reads the same either way. On a desktop workload
        /// `computes` is 0, so this is the draw rail alone.
        readbacks,
        readback_bytes,
        /// Full-frame reads of a pinned resident through `read_target`: the present
        /// capture and the deferred render window's on-access flush.
        ///
        /// These are the copies a deferred rail *keeps*, paid once when a consumer
        /// asks instead of once per Store. `target_reads / readbacks` is what
        /// separates "the readback moved" from "the readback went away".
        target_reads,
        target_read_bytes,
        seed_uploads,
        seed_upload_bytes,
        /// Present-boundary seeds satisfied by a GPU resident→target image copy
        /// (no CPU front-frame read, no seed upload); bytes = elided upload size.
        seed_gpu_copies,
        seed_gpu_copy_bytes,
        sampled_reuploads,
        sampled_reupload_bytes,
        /// Sampled binds served by gathering scattered guest pages into staging
        /// (`SampledSource::GuestRuns`), and the bytes those gathers moved.
        ///
        /// Every other arm of the sampled loop already reported itself and this
        /// one did not, which is how `acquire_sampled` came to be measured at
        /// the whole of a draw's acquire cost with no counter accounting for
        /// it. See `draw_phase`'s "What the sampled loop's own cost is *not*".
        sampled_gathers,
        sampled_gather_bytes,
        /// Sampled binds that would have gathered and did not, because both
        /// halves of the guest-write witness vouched that the retained image's
        /// bytes could not have moved. Bytes = the gather that did not happen,
        /// so `sampled_gather_bytes + sampled_gather_skip_bytes` is what this
        /// rail would cost with no cache.
        sampled_gather_skips,
        sampled_gather_skip_bytes,
        /// Sampled binds the GPU read straight out of the guest's own pages
        /// through an imported dma-buf — no CPU gather, no staging scratch.
        ///
        /// The third disposition of a `SampledSource::GuestRuns` bind, ranked
        /// against `sampled_gather_skips` (bound a retained image, moved
        /// nothing) and `sampled_gathers` (the CPU packed the texels). Bytes are
        /// what the copy names, which is what the CPU no longer moves.
        sampled_guest_imports,
        sampled_guest_import_bytes,
        /// How much of its target each draw could have written, split three
        /// ways. See [`DrawCoverage`] and [`EngineCounters::note_draw_coverage`].
        draw_cover_full,
        draw_cover_loaded_full_scissor,
        draw_cover_loaded_partial_scissor,
        /// Vertex/storage buffer binds the draw pointed straight at the guest's
        /// own pages through an imported dma-buf, with no copy in either
        /// direction. Ranked against `buffer_snapshot_binds` and the
        /// `stage_phase` `runs_*` bars, which are what the CPU still gathers.
        buffer_guest_imports,
        buffer_guest_import_bytes,
        sampled_cache_hits,
        sampled_identity_hits,
        sampled_cache_hit_bytes,
        sampled_cache_misses,
        sampled_gpu_binds,
        /// Batched-draw guest-run buffer binds the CPU had to gather, because
        /// the host could not export the pages or the span sits at an offset
        /// this device will not bind at.
        ///
        /// A subset of the `stage_phase` `runs_*` bars, distinguished by *when*
        /// the bytes were read: a batched CB reads them at record time and an
        /// immediate one effectively at submit, which is a real difference in
        /// how stale a snapshot can be. `buffer_guest_imports` is the other
        /// disposition of the same bind, where nothing was read at all.
        buffer_snapshot_binds,
        gpu_load_hits,
        target_evicts,
        /// Descriptor-arena growth events: a new pool block was appended because
        /// every existing block was exhausted (cap-pressure signal; 0 = no growth).
        desc_pool_grow,
        gen_mismatch,
        /// Post-submit fence waits skipped by all-deferred compute dispatches.
        compute_post_wait_skips,
        /// Post-submit fence waits skipped by no-readback (resident-target) draws.
        render_post_wait_skips,
        /// Entries that found the ring full and had to block on the oldest
        /// in-flight fence in begin_entry. This fires only when RING_DEPTH
        /// consecutive no-wait entries outrun the GPU.
        ring_retire_blocks,
        /// Draw batching (deferred submit): draws that OPENED a batch (left their
        /// CB recording), draws that JOINED an open batch (skipped
        /// begin_entry+submit entirely), batch submits, and total draws carried by
        /// those submits (avg batch length = batch_flush_draws / batch_flushes).
        batch_opens,
        batch_joins,
        batch_flushes,
        batch_flush_draws,
        /// Readbacks that appended their copy to a batch that was still
        /// recording, and so were submitted with it instead of behind it.
        ///
        /// This counted the opposite before the append path existed: the same
        /// population, as *flushes a readback forced*. A driven boot read it at
        /// 58.8 % of all `batch_flushes`, with batches averaging 1.77 draws
        /// against a `BATCH_MAX_DRAWS` of 8 — so nearly every readback was
        /// ending a run of draws to buy itself a second `vkQueueSubmit`. Each
        /// one counted here is now one submission rather than two.
        ///
        /// Read against `batch_flushes` for the share that collapses. Do **not**
        /// expect `batch_flush_draws / batch_flushes` to move with it — it read
        /// 1.77 before the append path and 1.78 after, because a readback still
        /// ends the batch it joined. A readback arriving with no batch open is
        /// not counted and has nothing to collapse.
        ///
        /// Counted at the readback sites rather than inside `batch_flush`,
        /// because that function cannot see who called it and threading a reason
        /// through `begin_entry` would put a diagnostic in the signature of the
        /// device's hottest slot claim.
        batch_readback_joins,
    }

    cumulative {
        /// Cumulative across the boot: a device loss is a fact about this run,
        /// not about the measurement window, so `reset()` leaves it standing.
        device_lost,
        /// Cumulative across the boot, for the same reason as `device_lost`.
        recreates,
    }

    pool_sourced {
        /// Sampled-cache pool recycle diagnostics (workstream D lag tail). These
        /// four come from `ResourcePools`, not the atomic counters — merged in by
        /// `engine::counter_snapshot`. `free_hits` = `acquire_sampled` reused a
        /// recycled slot (no `vkAllocateMemory`); `free_allocs` = it had to create
        /// a fresh image; `recycle_admits` = an evicted slot rejoined the per-key
        /// free list; `recycle_cap_drops` = an evicted slot was destroyed because
        /// the per-key cap was full (raising the cap would have kept it). A high
        /// `free_allocs` with a high `recycle_cap_drops` means the cap is the
        /// limiter; a high `free_allocs` with low admits means the drain timing is.
        sampled_free_hits,
        sampled_free_allocs,
        sampled_recycle_admits,
        sampled_recycle_cap_drops,
        /// Resident render-target recycle diagnostics (same shape as the sampled
        /// ones). `target_free_hits` = a create reused a recycled image (no
        /// `vkCreateImage`/`vkAllocateMemory`); `target_free_allocs` = it had to
        /// allocate fresh; `target_recycle_admits`/`target_recycle_cap_drops` =
        /// displaced images that rejoined / overflowed the per-key free list. Owned
        /// by `ResourcePools`; merged in by `engine::counter_snapshot` (zero here).
        target_free_hits,
        target_free_allocs,
        target_recycle_admits,
        target_recycle_cap_drops,
        /// High-water mark of the non-pinned resident population, against which
        /// `REGISTRY_CAP` is the ceiling — read as a band, `peak/cap`.
        ///
        /// The reach, not the drops. `target_registry_cap_evictions` reading zero
        /// says only that the cap did not bind on the workload that ran; it
        /// cannot distinguish a peak of 40 from a peak of 319, and those are
        /// opposite answers to whether the bound has headroom. This is the one
        /// that separates them, which is why it exists as a peak rather than as
        /// an instantaneous population: the cap's whole purpose is to survive a
        /// compositing *burst*, and a burst that rises and drains between two
        /// census samples is exactly what an instantaneous reading misses.
        ///
        /// Cumulative and never reset by the windowed reset, because the
        /// question is "how close did this boot ever come", not "how close is it
        /// now". `EngineCounters::reset_all` clears it for tests.
        ///
        /// Sampled where the cap is enforced, so every admission that could grow
        /// the population is seen. Two prior readings are quoted in this
        /// module's neighbours — a non-pinned peak of ~260 under a YouTube
        /// page-load, and `reg=512/512 evicts=168` before pinned slots were
        /// excluded — and **neither is reproducible today**: nothing in the tree
        /// emitted them, so they are historical probe output rather than
        /// something a boot can be asked for. That is the gap this closes.
        registry_non_pinned_peak,
        /// Residents destroyed by the capacity walk, cumulative.
        ///
        /// Paired with the peak above because the two are only interpretable
        /// together, and this is the half that counts loss: a retired resident's
        /// pixels existed only on the GPU and nothing recreates one except a
        /// draw rendering into the same identity, so a non-zero reading here is
        /// guest content destroyed rather than a cache asked to refill.
        target_registry_cap_evictions,
        /// Compute-storage residents destroyed by `COMPUTE_STORAGE_REGISTRY_CAP`,
        /// cumulative.
        ///
        /// The same quantity as `target_registry_cap_evictions` over the other
        /// registry, and it counts the same thing: lost guest work. Nothing
        /// recreates a compute-storage resident's content either, so a dispatch
        /// that later reads a destroyed identity refuses with
        /// `ResidentSampleAbsent` or `ResidentSeedGenerationLost`.
        ///
        /// It exists because that sweep incremented nothing at all, which left
        /// its 64 unfalsifiable — the cap could have been biting on every
        /// compute-heavy boot and no counter, route or log line would have
        /// moved. Kept separate from the target count rather than summed with
        /// it: the two registries are bounded by different constants over
        /// different populations, and a boot needs to know which one bit.
        compute_storage_cap_evictions,
        /// The same high-water in attachment bytes, sampled from the same
        /// population at the same instant as `registry_non_pinned_peak`.
        ///
        /// `REGISTRY_CAP` bounds slots while its own doc says "slots are cheap;
        /// the real VRAM guard is per-image bytes" — so the cap does not measure
        /// the resource it is defending. 320 slots is 5 MiB of 16x16 scratch or
        /// 10 GiB of 4K, and nothing in this device could tell those apart until
        /// this counter. A lower bound on VRAM (attachment footprint, no tiling
        /// padding, and a format with no single texel size contributes nothing),
        /// which is the safe direction for a figure that exists to decide
        /// whether a bound is too loose.
        registry_non_pinned_peak_bytes,
    }
}
/// The `note_*` helpers: the increments that are not a bare `fetch_add(1)` at
/// the call site, because they move a count and a byte total together.
impl EngineCounters {
    pub fn note_create(&self) {
        self.creates.fetch_add(1, Ordering::Relaxed);
    }

    pub fn note_alloc(&self) {
        self.allocs.fetch_add(1, Ordering::Relaxed);
    }

    pub fn note_readback(&self, bytes: u64) {
        self.readbacks.fetch_add(1, Ordering::Relaxed);
        self.readback_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    pub fn note_target_read(&self, bytes: u64) {
        self.target_reads.fetch_add(1, Ordering::Relaxed);
        self.target_read_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    pub fn note_seed_upload(&self, bytes: u64) {
        self.seed_uploads.fetch_add(1, Ordering::Relaxed);
        self.seed_upload_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    pub fn note_sampled_reupload(&self, bytes: u64) {
        self.sampled_reuploads.fetch_add(1, Ordering::Relaxed);
        self.sampled_reupload_bytes
            .fetch_add(bytes, Ordering::Relaxed);
    }

    pub fn note_sampled_gather(&self, bytes: u64) {
        self.sampled_gathers.fetch_add(1, Ordering::Relaxed);
        self.sampled_gather_bytes
            .fetch_add(bytes, Ordering::Relaxed);
    }

    pub fn note_buffer_guest_import(&self, bytes: u64) {
        self.buffer_guest_imports.fetch_add(1, Ordering::Relaxed);
        self.buffer_guest_import_bytes
            .fetch_add(bytes, Ordering::Relaxed);
    }

    /// Record how much of its target one draw could have written.
    ///
    /// # Reading the three against `surface_flush`
    ///
    /// A damage rect can only pay when a flushed surface receives *no*
    /// whole-surface write between two flushes, because any one of those makes
    /// the union total. So the verdict is a comparison of rates, not a ratio of
    /// these three to each other:
    ///
    /// ```text
    ///   (draw_cover_full + draw_cover_loaded_full_scissor) per second
    ///   ---------------------------------------------------------- << 1
    ///                    flushes per second
    /// ```
    ///
    /// On a driven x86 Safari window-drag boot it was **4.4**, not «1: 840
    /// clears and 1 637 full-scissor draws against 560 flushes, so every flush
    /// interval held several whole-surface writes and a rect built over this
    /// would have copied whole surfaces anyway. That is what was measured when
    /// the rail existed, by a `flush_rows` / `flush_surface_rows` pair that read
    /// exactly equal on every census line of the boot.
    ///
    /// `draw_cover_loaded_partial_scissor` was 1 718 in the same second — 41% of
    /// all draws — which is why the ratio and not that number is the test. A
    /// workload can bind mostly-partial scissors and still leave nothing to
    /// narrow.
    pub fn note_draw_coverage(&self, coverage: DrawCoverage) {
        let field = match coverage {
            DrawCoverage::Full => &self.draw_cover_full,
            DrawCoverage::LoadedFullScissor => &self.draw_cover_loaded_full_scissor,
            DrawCoverage::LoadedPartialScissor => &self.draw_cover_loaded_partial_scissor,
        };
        field.fetch_add(1, Ordering::Relaxed);
    }

    pub fn note_sampled_guest_import(&self, bytes: u64) {
        self.sampled_guest_imports.fetch_add(1, Ordering::Relaxed);
        self.sampled_guest_import_bytes
            .fetch_add(bytes, Ordering::Relaxed);
    }

    pub fn note_sampled_gather_skipped(&self, bytes: u64) {
        self.sampled_gather_skips.fetch_add(1, Ordering::Relaxed);
        self.sampled_gather_skip_bytes
            .fetch_add(bytes, Ordering::Relaxed);
    }

    pub fn note_compute_sampled_upload(&self, bytes: u64) {
        self.compute_sampled_uploads.fetch_add(1, Ordering::Relaxed);
        self.compute_sampled_upload_bytes
            .fetch_add(bytes, Ordering::Relaxed);
    }

    pub fn note_compute_storage_seed_upload(&self, bytes: u64) {
        self.compute_storage_seed_uploads
            .fetch_add(1, Ordering::Relaxed);
        self.compute_storage_seed_upload_bytes
            .fetch_add(bytes, Ordering::Relaxed);
    }

    pub fn note_compute_sampled_resident_copy(&self, bytes: u64) {
        self.compute_sampled_resident_copies
            .fetch_add(1, Ordering::Relaxed);
        self.compute_sampled_resident_copy_bytes
            .fetch_add(bytes, Ordering::Relaxed);
    }

    pub fn note_compute_deferred_writeback(&self, bytes: u64) {
        self.compute_deferred_writebacks
            .fetch_add(1, Ordering::Relaxed);
        self.compute_deferred_writeback_bytes
            .fetch_add(bytes, Ordering::Relaxed);
    }

    pub fn note_compute_deferred_flush(&self, bytes: u64) {
        self.compute_deferred_flushes
            .fetch_add(1, Ordering::Relaxed);
        self.compute_deferred_flush_bytes
            .fetch_add(bytes, Ordering::Relaxed);
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn note_helpers_update_event_and_byte_counters_together() {
        let counters = EngineCounters::default();
        counters.note_create();
        counters.note_alloc();
        counters.note_readback(4096);
        counters.note_seed_upload(1024);
        counters.note_sampled_gather(2048);
        counters.note_sampled_gather_skipped(512);

        let snapshot = counters.snapshot();
        assert_eq!(snapshot.creates, 1);
        assert_eq!(snapshot.allocs, 1);
        assert_eq!((snapshot.readbacks, snapshot.readback_bytes), (1, 4096));
        assert_eq!(
            (snapshot.seed_uploads, snapshot.seed_upload_bytes),
            (1, 1024)
        );
        // The gather is the sampled loop's only byte-moving arm, and it went
        // uncounted long enough to hide the whole of `acquire_sampled`. Pairing
        // it here keeps the event and its bytes from drifting apart the way a
        // count-only counter would.
        assert_eq!(
            (snapshot.sampled_gathers, snapshot.sampled_gather_bytes),
            (1, 2048)
        );
        // And the gathers that did not happen, whose bytes are the other half of
        // what this rail would cost with no cache.
        assert_eq!(
            (
                snapshot.sampled_gather_skips,
                snapshot.sampled_gather_skip_bytes
            ),
            (1, 512)
        );
    }

    #[test]
    fn reset_clears_draw_gate_counters_but_preserves_lifetime_loss_counts() {
        let counters = EngineCounters::default();
        counters.readbacks.store(4, Ordering::Relaxed);
        counters.desc_pool_grow.store(3, Ordering::Relaxed);
        counters.device_lost.store(2, Ordering::Relaxed);
        counters.recreates.store(1, Ordering::Relaxed);

        counters.reset();
        let reset = counters.snapshot();
        assert_eq!(reset.readbacks, 0);
        assert_eq!(reset.desc_pool_grow, 0);
        assert_eq!(reset.device_lost, 2);
        assert_eq!(reset.recreates, 1);

        counters.reset_all();
        assert_eq!(counters.snapshot(), CounterSnapshot::default());
    }

    #[test]
    fn snapshot_delta_saturates_after_a_counter_reset() {
        let earlier = CounterSnapshot {
            creates: 10,
            readback_bytes: 4096,
            ..Default::default()
        };
        let later = CounterSnapshot {
            creates: 13,
            readback_bytes: 1024,
            ..Default::default()
        };

        let delta = later.delta_since(&earlier);
        assert_eq!(delta.creates, 3);
        assert_eq!(delta.readback_bytes, 0);
        assert_eq!(delta.allocs, 0);
    }

    #[test]
    fn atomic_snapshot_leaves_pool_owned_counters_for_the_pool_merge() {
        let snapshot = EngineCounters::default().snapshot();
        assert_eq!(snapshot.sampled_free_hits, 0);
        assert_eq!(snapshot.sampled_free_allocs, 0);
        assert_eq!(snapshot.target_free_hits, 0);
        assert_eq!(snapshot.target_free_allocs, 0);
    }
}
