//! Always-on create/alloc and cache hit/miss counters (reuse-gate proxies).

use std::sync::atomic::{AtomicU64, Ordering};

/// Process-wide product-path counters (resettable for tests).
#[derive(Debug, Default)]
pub struct EngineCounters {
    pub creates: AtomicU64,
    pub allocs: AtomicU64,
    pub shader_hits: AtomicU64,
    pub shader_misses: AtomicU64,
    pub layout_hits: AtomicU64,
    pub layout_misses: AtomicU64,
    pub pass_hits: AtomicU64,
    pub pass_misses: AtomicU64,
    pub pipeline_hits: AtomicU64,
    pub pipeline_misses: AtomicU64,
    pub sampler_hits: AtomicU64,
    pub sampler_misses: AtomicU64,
    pub device_lost: AtomicU64,
    pub recreates: AtomicU64,
    // --- compute (workstream C) ---
    pub compute_pipeline_hits: AtomicU64,
    pub compute_pipeline_misses: AtomicU64,
    pub dispatches: AtomicU64,
    pub fence_timeouts: AtomicU64,
    /// Compute sampled-image bytes staged for host→device upload.
    pub compute_sampled_uploads: AtomicU64,
    pub compute_sampled_upload_bytes: AtomicU64,
    /// Compute storage-image seed bytes staged for host→device upload.
    pub compute_storage_seed_uploads: AtomicU64,
    pub compute_storage_seed_upload_bytes: AtomicU64,
    /// Compute storage images the GPU copied straight into the caller's
    /// imported guest window (VK_EXT_external_memory_host) — no host-visible
    /// readback buffer, no CPU writeback copy.
    pub compute_direct_writebacks: AtomicU64,
    pub compute_direct_writeback_bytes: AtomicU64,
    /// Requested direct writebacks that fell back to the readback path
    /// (import/bind failure at exec time).
    pub compute_direct_writeback_fallbacks: AtomicU64,
    /// Sampled inputs seeded by a device-local copy of a resident storage
    /// image (copy-on-sample) — bytes are the elided host upload size.
    pub compute_sampled_resident_copies: AtomicU64,
    pub compute_sampled_resident_copy_bytes: AtomicU64,
    /// Subset of the resident copies that crossed vk formats through the
    /// image→buffer→image byte-reinterpret hop (row-byte-identical views).
    pub compute_sampled_reinterpret_copies: AtomicU64,
    pub compute_sampled_reinterpret_copy_bytes: AtomicU64,
    /// Compute storage images whose post-dispatch readback was deferred —
    /// the pinned resident stays authoritative; bytes are the elided
    /// device→host readback size (the CPU writeback of the same size is
    /// elided too).
    pub compute_deferred_writebacks: AtomicU64,
    pub compute_deferred_writeback_bytes: AtomicU64,
    /// Deferred-flush reads (read_resident_storage): the on-access GPU→host
    /// copy that lands deferred content in guest pages.
    pub compute_deferred_flushes: AtomicU64,
    pub compute_deferred_flush_bytes: AtomicU64,
    // --- residency / oracle I/O (workstream D) ---
    pub readbacks: AtomicU64,
    pub readback_bytes: AtomicU64,
    pub seed_uploads: AtomicU64,
    pub seed_upload_bytes: AtomicU64,
    /// Present-boundary seeds satisfied by a GPU resident→target image copy
    /// (no CPU front-frame read, no seed upload); bytes = elided upload size.
    pub seed_gpu_copies: AtomicU64,
    pub seed_gpu_copy_bytes: AtomicU64,
    pub sampled_reuploads: AtomicU64,
    pub sampled_reupload_bytes: AtomicU64,
    pub sampled_cache_hits: AtomicU64,
    pub sampled_identity_hits: AtomicU64,
    pub sampled_cache_hit_bytes: AtomicU64,
    pub sampled_cache_misses: AtomicU64,
    pub sampled_gpu_binds: AtomicU64,
    /// Zero-copy guest-run sampled binds (GPU gathered from imported guest RAM).
    pub sampled_zerocopy_binds: AtomicU64,
    /// Zero-copy guest-run vertex/storage buffer binds (GPU gathered from
    /// imported guest RAM into the pooled staging slot the bind then uses).
    pub buffer_zerocopy_binds: AtomicU64,
    /// Guest-run buffer binds snapshotted on the CPU at record time because
    /// the draw defers its submit (batched CB must not read volatile guest
    /// RAM at flush time).
    pub buffer_snapshot_binds: AtomicU64,
    pub gpu_load_hits: AtomicU64,
    pub seed_imports: AtomicU64,
    pub target_evicts: AtomicU64,
    /// Descriptor-arena growth events: a new pool block was appended because
    /// every existing block was exhausted (cap-pressure signal; 0 = no growth).
    pub desc_pool_grow: AtomicU64,
    pub target_stale_import: AtomicU64,
    pub gen_mismatch: AtomicU64,
    /// Present-boundary frames copied straight into imported guest memory via
    /// VK_EXT_external_memory_host (workstream E) — zero CPU readback copy.
    pub import_presents: AtomicU64,
    // timing splits (microseconds, cumulative between resets)
    /// Post-submit fence waits skipped by all-deferred compute dispatches.
    pub compute_post_wait_skips: AtomicU64,
    /// Post-submit fence waits skipped by no-readback (resident-target) draws.
    pub render_post_wait_skips: AtomicU64,
    /// Entries that found the ring full and had to block on the oldest
    /// in-flight fence in begin_entry. This fires only when RING_DEPTH
    /// consecutive no-wait entries outrun the GPU.
    pub ring_retire_blocks: AtomicU64,
    /// Draw batching (deferred submit): draws that OPENED a batch (left their
    /// CB recording), draws that JOINED an open batch (skipped
    /// begin_entry+submit entirely), batch submits, and total draws carried by
    /// those submits (avg batch length = batch_flush_draws / batch_flushes).
    pub batch_opens: AtomicU64,
    pub batch_joins: AtomicU64,
    pub batch_flushes: AtomicU64,
    pub batch_flush_draws: AtomicU64,
}

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

    pub fn note_import_present(&self) {
        self.import_presents.fetch_add(1, Ordering::Relaxed);
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

    pub fn note_compute_direct_writeback(&self, bytes: u64) {
        self.compute_direct_writebacks
            .fetch_add(1, Ordering::Relaxed);
        self.compute_direct_writeback_bytes
            .fetch_add(bytes, Ordering::Relaxed);
    }

    pub fn note_compute_direct_writeback_fallback(&self) {
        self.compute_direct_writeback_fallbacks
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn note_compute_sampled_resident_copy(&self, bytes: u64) {
        self.compute_sampled_resident_copies
            .fetch_add(1, Ordering::Relaxed);
        self.compute_sampled_resident_copy_bytes
            .fetch_add(bytes, Ordering::Relaxed);
    }

    pub fn note_compute_sampled_reinterpret_copy(&self, bytes: u64) {
        self.compute_sampled_reinterpret_copies
            .fetch_add(1, Ordering::Relaxed);
        self.compute_sampled_reinterpret_copy_bytes
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

    pub fn snapshot(&self) -> CounterSnapshot {
        CounterSnapshot {
            creates: self.creates.load(Ordering::Relaxed),
            allocs: self.allocs.load(Ordering::Relaxed),
            shader_hits: self.shader_hits.load(Ordering::Relaxed),
            shader_misses: self.shader_misses.load(Ordering::Relaxed),
            layout_hits: self.layout_hits.load(Ordering::Relaxed),
            layout_misses: self.layout_misses.load(Ordering::Relaxed),
            pass_hits: self.pass_hits.load(Ordering::Relaxed),
            pass_misses: self.pass_misses.load(Ordering::Relaxed),
            pipeline_hits: self.pipeline_hits.load(Ordering::Relaxed),
            pipeline_misses: self.pipeline_misses.load(Ordering::Relaxed),
            sampler_hits: self.sampler_hits.load(Ordering::Relaxed),
            sampler_misses: self.sampler_misses.load(Ordering::Relaxed),
            device_lost: self.device_lost.load(Ordering::Relaxed),
            recreates: self.recreates.load(Ordering::Relaxed),
            compute_pipeline_hits: self.compute_pipeline_hits.load(Ordering::Relaxed),
            compute_pipeline_misses: self.compute_pipeline_misses.load(Ordering::Relaxed),
            dispatches: self.dispatches.load(Ordering::Relaxed),
            fence_timeouts: self.fence_timeouts.load(Ordering::Relaxed),
            compute_sampled_uploads: self.compute_sampled_uploads.load(Ordering::Relaxed),
            compute_sampled_upload_bytes: self.compute_sampled_upload_bytes.load(Ordering::Relaxed),
            compute_storage_seed_uploads: self.compute_storage_seed_uploads.load(Ordering::Relaxed),
            compute_storage_seed_upload_bytes: self
                .compute_storage_seed_upload_bytes
                .load(Ordering::Relaxed),
            compute_direct_writebacks: self.compute_direct_writebacks.load(Ordering::Relaxed),
            compute_direct_writeback_bytes: self
                .compute_direct_writeback_bytes
                .load(Ordering::Relaxed),
            compute_direct_writeback_fallbacks: self
                .compute_direct_writeback_fallbacks
                .load(Ordering::Relaxed),
            compute_sampled_resident_copies: self
                .compute_sampled_resident_copies
                .load(Ordering::Relaxed),
            compute_sampled_resident_copy_bytes: self
                .compute_sampled_resident_copy_bytes
                .load(Ordering::Relaxed),
            compute_sampled_reinterpret_copies: self
                .compute_sampled_reinterpret_copies
                .load(Ordering::Relaxed),
            compute_sampled_reinterpret_copy_bytes: self
                .compute_sampled_reinterpret_copy_bytes
                .load(Ordering::Relaxed),
            compute_deferred_writebacks: self.compute_deferred_writebacks.load(Ordering::Relaxed),
            compute_deferred_writeback_bytes: self
                .compute_deferred_writeback_bytes
                .load(Ordering::Relaxed),
            compute_deferred_flushes: self.compute_deferred_flushes.load(Ordering::Relaxed),
            compute_deferred_flush_bytes: self.compute_deferred_flush_bytes.load(Ordering::Relaxed),
            readbacks: self.readbacks.load(Ordering::Relaxed),
            readback_bytes: self.readback_bytes.load(Ordering::Relaxed),
            seed_uploads: self.seed_uploads.load(Ordering::Relaxed),
            seed_upload_bytes: self.seed_upload_bytes.load(Ordering::Relaxed),
            seed_gpu_copies: self.seed_gpu_copies.load(Ordering::Relaxed),
            seed_gpu_copy_bytes: self.seed_gpu_copy_bytes.load(Ordering::Relaxed),
            sampled_reuploads: self.sampled_reuploads.load(Ordering::Relaxed),
            sampled_reupload_bytes: self.sampled_reupload_bytes.load(Ordering::Relaxed),
            sampled_cache_hits: self.sampled_cache_hits.load(Ordering::Relaxed),
            sampled_identity_hits: self.sampled_identity_hits.load(Ordering::Relaxed),
            sampled_cache_hit_bytes: self.sampled_cache_hit_bytes.load(Ordering::Relaxed),
            sampled_cache_misses: self.sampled_cache_misses.load(Ordering::Relaxed),
            sampled_gpu_binds: self.sampled_gpu_binds.load(Ordering::Relaxed),
            sampled_zerocopy_binds: self.sampled_zerocopy_binds.load(Ordering::Relaxed),
            buffer_zerocopy_binds: self.buffer_zerocopy_binds.load(Ordering::Relaxed),
            buffer_snapshot_binds: self.buffer_snapshot_binds.load(Ordering::Relaxed),
            gpu_load_hits: self.gpu_load_hits.load(Ordering::Relaxed),
            seed_imports: self.seed_imports.load(Ordering::Relaxed),
            target_evicts: self.target_evicts.load(Ordering::Relaxed),
            desc_pool_grow: self.desc_pool_grow.load(Ordering::Relaxed),
            target_stale_import: self.target_stale_import.load(Ordering::Relaxed),
            gen_mismatch: self.gen_mismatch.load(Ordering::Relaxed),
            import_presents: self.import_presents.load(Ordering::Relaxed),
            compute_post_wait_skips: self.compute_post_wait_skips.load(Ordering::Relaxed),
            render_post_wait_skips: self.render_post_wait_skips.load(Ordering::Relaxed),
            ring_retire_blocks: self.ring_retire_blocks.load(Ordering::Relaxed),
            batch_opens: self.batch_opens.load(Ordering::Relaxed),
            batch_joins: self.batch_joins.load(Ordering::Relaxed),
            batch_flushes: self.batch_flushes.load(Ordering::Relaxed),
            batch_flush_draws: self.batch_flush_draws.load(Ordering::Relaxed),
            // Owned by ResourcePools, not the atomics — engine::counter_snapshot
            // overwrites these from pools.recycle_stats(); zero here.
            sampled_free_hits: 0,
            sampled_free_allocs: 0,
            sampled_recycle_admits: 0,
            sampled_recycle_cap_drops: 0,
            target_free_hits: 0,
            target_free_allocs: 0,
            target_recycle_admits: 0,
            target_recycle_cap_drops: 0,
        }
    }

    pub fn reset(&self) {
        self.creates.store(0, Ordering::Relaxed);
        self.allocs.store(0, Ordering::Relaxed);
        self.shader_hits.store(0, Ordering::Relaxed);
        self.shader_misses.store(0, Ordering::Relaxed);
        self.layout_hits.store(0, Ordering::Relaxed);
        self.layout_misses.store(0, Ordering::Relaxed);
        self.pass_hits.store(0, Ordering::Relaxed);
        self.pass_misses.store(0, Ordering::Relaxed);
        self.pipeline_hits.store(0, Ordering::Relaxed);
        self.pipeline_misses.store(0, Ordering::Relaxed);
        self.sampler_hits.store(0, Ordering::Relaxed);
        self.sampler_misses.store(0, Ordering::Relaxed);
        self.compute_pipeline_hits.store(0, Ordering::Relaxed);
        self.compute_pipeline_misses.store(0, Ordering::Relaxed);
        self.dispatches.store(0, Ordering::Relaxed);
        self.fence_timeouts.store(0, Ordering::Relaxed);
        self.compute_sampled_uploads.store(0, Ordering::Relaxed);
        self.compute_sampled_upload_bytes
            .store(0, Ordering::Relaxed);
        self.compute_storage_seed_uploads
            .store(0, Ordering::Relaxed);
        self.compute_storage_seed_upload_bytes
            .store(0, Ordering::Relaxed);
        self.compute_direct_writebacks.store(0, Ordering::Relaxed);
        self.compute_direct_writeback_bytes
            .store(0, Ordering::Relaxed);
        self.compute_direct_writeback_fallbacks
            .store(0, Ordering::Relaxed);
        self.compute_sampled_resident_copies
            .store(0, Ordering::Relaxed);
        self.compute_sampled_resident_copy_bytes
            .store(0, Ordering::Relaxed);
        self.compute_sampled_reinterpret_copies
            .store(0, Ordering::Relaxed);
        self.compute_sampled_reinterpret_copy_bytes
            .store(0, Ordering::Relaxed);
        self.compute_deferred_writebacks.store(0, Ordering::Relaxed);
        self.compute_deferred_writeback_bytes
            .store(0, Ordering::Relaxed);
        self.compute_deferred_flushes.store(0, Ordering::Relaxed);
        self.compute_deferred_flush_bytes
            .store(0, Ordering::Relaxed);
        self.readbacks.store(0, Ordering::Relaxed);
        self.readback_bytes.store(0, Ordering::Relaxed);
        self.seed_uploads.store(0, Ordering::Relaxed);
        self.seed_upload_bytes.store(0, Ordering::Relaxed);
        self.seed_gpu_copies.store(0, Ordering::Relaxed);
        self.seed_gpu_copy_bytes.store(0, Ordering::Relaxed);
        self.sampled_reuploads.store(0, Ordering::Relaxed);
        self.sampled_reupload_bytes.store(0, Ordering::Relaxed);
        self.sampled_cache_hits.store(0, Ordering::Relaxed);
        self.sampled_identity_hits.store(0, Ordering::Relaxed);
        self.sampled_cache_hit_bytes.store(0, Ordering::Relaxed);
        self.sampled_cache_misses.store(0, Ordering::Relaxed);
        self.sampled_gpu_binds.store(0, Ordering::Relaxed);
        self.sampled_zerocopy_binds.store(0, Ordering::Relaxed);
        self.buffer_zerocopy_binds.store(0, Ordering::Relaxed);
        self.buffer_snapshot_binds.store(0, Ordering::Relaxed);
        self.gpu_load_hits.store(0, Ordering::Relaxed);
        self.seed_imports.store(0, Ordering::Relaxed);
        self.target_evicts.store(0, Ordering::Relaxed);
        self.desc_pool_grow.store(0, Ordering::Relaxed);
        self.target_stale_import.store(0, Ordering::Relaxed);
        self.gen_mismatch.store(0, Ordering::Relaxed);
        self.import_presents.store(0, Ordering::Relaxed);
        self.compute_post_wait_skips.store(0, Ordering::Relaxed);
        self.render_post_wait_skips.store(0, Ordering::Relaxed);
        self.ring_retire_blocks.store(0, Ordering::Relaxed);
        self.batch_opens.store(0, Ordering::Relaxed);
        self.batch_joins.store(0, Ordering::Relaxed);
        self.batch_flushes.store(0, Ordering::Relaxed);
        self.batch_flush_draws.store(0, Ordering::Relaxed);
        // device_lost / recreates are cumulative across boot; do not zero on draw-gate reset
    }

    pub fn reset_all(&self) {
        self.reset();
        self.device_lost.store(0, Ordering::Relaxed);
        self.recreates.store(0, Ordering::Relaxed);
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CounterSnapshot {
    pub creates: u64,
    pub allocs: u64,
    pub shader_hits: u64,
    pub shader_misses: u64,
    pub layout_hits: u64,
    pub layout_misses: u64,
    pub pass_hits: u64,
    pub pass_misses: u64,
    pub pipeline_hits: u64,
    pub pipeline_misses: u64,
    pub sampler_hits: u64,
    pub sampler_misses: u64,
    pub device_lost: u64,
    pub recreates: u64,
    pub compute_pipeline_hits: u64,
    pub compute_pipeline_misses: u64,
    pub dispatches: u64,
    pub fence_timeouts: u64,
    pub compute_sampled_uploads: u64,
    pub compute_sampled_upload_bytes: u64,
    pub compute_storage_seed_uploads: u64,
    pub compute_storage_seed_upload_bytes: u64,
    pub compute_direct_writebacks: u64,
    pub compute_direct_writeback_bytes: u64,
    pub compute_direct_writeback_fallbacks: u64,
    pub compute_sampled_resident_copies: u64,
    pub compute_sampled_resident_copy_bytes: u64,
    pub compute_sampled_reinterpret_copies: u64,
    pub compute_sampled_reinterpret_copy_bytes: u64,
    pub compute_deferred_writebacks: u64,
    pub compute_deferred_writeback_bytes: u64,
    pub compute_deferred_flushes: u64,
    pub compute_deferred_flush_bytes: u64,
    pub readbacks: u64,
    pub readback_bytes: u64,
    pub seed_uploads: u64,
    pub seed_upload_bytes: u64,
    pub seed_gpu_copies: u64,
    pub seed_gpu_copy_bytes: u64,
    pub sampled_reuploads: u64,
    pub sampled_reupload_bytes: u64,
    pub sampled_cache_hits: u64,
    pub sampled_identity_hits: u64,
    pub sampled_cache_hit_bytes: u64,
    pub sampled_cache_misses: u64,
    pub sampled_gpu_binds: u64,
    pub sampled_zerocopy_binds: u64,
    pub buffer_zerocopy_binds: u64,
    pub buffer_snapshot_binds: u64,
    pub gpu_load_hits: u64,
    pub seed_imports: u64,
    pub target_evicts: u64,
    pub desc_pool_grow: u64,
    pub target_stale_import: u64,
    pub gen_mismatch: u64,
    pub import_presents: u64,
    pub compute_post_wait_skips: u64,
    pub render_post_wait_skips: u64,
    pub ring_retire_blocks: u64,
    pub batch_opens: u64,
    pub batch_joins: u64,
    pub batch_flushes: u64,
    pub batch_flush_draws: u64,
    /// Sampled-cache pool recycle diagnostics (workstream D lag tail). These
    /// four come from `ResourcePools`, not the atomic counters — merged in by
    /// `engine::counter_snapshot`. `free_hits` = `acquire_sampled` reused a
    /// recycled slot (no `vkAllocateMemory`); `free_allocs` = it had to create
    /// a fresh image; `recycle_admits` = an evicted slot rejoined the per-key
    /// free list; `recycle_cap_drops` = an evicted slot was destroyed because
    /// the per-key cap was full (raising the cap would have kept it). A high
    /// `free_allocs` with a high `recycle_cap_drops` means the cap is the
    /// limiter; a high `free_allocs` with low admits means the drain timing is.
    pub sampled_free_hits: u64,
    pub sampled_free_allocs: u64,
    pub sampled_recycle_admits: u64,
    pub sampled_recycle_cap_drops: u64,
    /// Resident render-target recycle diagnostics (same shape as the sampled
    /// ones). `target_free_hits` = a create reused a recycled image (no
    /// `vkCreateImage`/`vkAllocateMemory`); `target_free_allocs` = it had to
    /// allocate fresh; `target_recycle_admits`/`target_recycle_cap_drops` =
    /// displaced images that rejoined / overflowed the per-key free list. Owned
    /// by `ResourcePools`; merged in by `engine::counter_snapshot` (zero here).
    pub target_free_hits: u64,
    pub target_free_allocs: u64,
    pub target_recycle_admits: u64,
    pub target_recycle_cap_drops: u64,
}

impl CounterSnapshot {
    pub fn delta_since(&self, earlier: &CounterSnapshot) -> CounterSnapshot {
        CounterSnapshot {
            creates: self.creates.saturating_sub(earlier.creates),
            allocs: self.allocs.saturating_sub(earlier.allocs),
            shader_hits: self.shader_hits.saturating_sub(earlier.shader_hits),
            shader_misses: self.shader_misses.saturating_sub(earlier.shader_misses),
            layout_hits: self.layout_hits.saturating_sub(earlier.layout_hits),
            layout_misses: self.layout_misses.saturating_sub(earlier.layout_misses),
            pass_hits: self.pass_hits.saturating_sub(earlier.pass_hits),
            pass_misses: self.pass_misses.saturating_sub(earlier.pass_misses),
            pipeline_hits: self.pipeline_hits.saturating_sub(earlier.pipeline_hits),
            pipeline_misses: self.pipeline_misses.saturating_sub(earlier.pipeline_misses),
            sampler_hits: self.sampler_hits.saturating_sub(earlier.sampler_hits),
            sampler_misses: self.sampler_misses.saturating_sub(earlier.sampler_misses),
            device_lost: self.device_lost.saturating_sub(earlier.device_lost),
            recreates: self.recreates.saturating_sub(earlier.recreates),
            compute_pipeline_hits: self
                .compute_pipeline_hits
                .saturating_sub(earlier.compute_pipeline_hits),
            compute_pipeline_misses: self
                .compute_pipeline_misses
                .saturating_sub(earlier.compute_pipeline_misses),
            dispatches: self.dispatches.saturating_sub(earlier.dispatches),
            fence_timeouts: self.fence_timeouts.saturating_sub(earlier.fence_timeouts),
            compute_sampled_uploads: self
                .compute_sampled_uploads
                .saturating_sub(earlier.compute_sampled_uploads),
            compute_sampled_upload_bytes: self
                .compute_sampled_upload_bytes
                .saturating_sub(earlier.compute_sampled_upload_bytes),
            compute_storage_seed_uploads: self
                .compute_storage_seed_uploads
                .saturating_sub(earlier.compute_storage_seed_uploads),
            compute_storage_seed_upload_bytes: self
                .compute_storage_seed_upload_bytes
                .saturating_sub(earlier.compute_storage_seed_upload_bytes),
            compute_direct_writebacks: self
                .compute_direct_writebacks
                .saturating_sub(earlier.compute_direct_writebacks),
            compute_direct_writeback_bytes: self
                .compute_direct_writeback_bytes
                .saturating_sub(earlier.compute_direct_writeback_bytes),
            compute_direct_writeback_fallbacks: self
                .compute_direct_writeback_fallbacks
                .saturating_sub(earlier.compute_direct_writeback_fallbacks),
            compute_sampled_resident_copies: self
                .compute_sampled_resident_copies
                .saturating_sub(earlier.compute_sampled_resident_copies),
            compute_sampled_resident_copy_bytes: self
                .compute_sampled_resident_copy_bytes
                .saturating_sub(earlier.compute_sampled_resident_copy_bytes),
            compute_sampled_reinterpret_copies: self
                .compute_sampled_reinterpret_copies
                .saturating_sub(earlier.compute_sampled_reinterpret_copies),
            compute_sampled_reinterpret_copy_bytes: self
                .compute_sampled_reinterpret_copy_bytes
                .saturating_sub(earlier.compute_sampled_reinterpret_copy_bytes),
            compute_deferred_writebacks: self
                .compute_deferred_writebacks
                .saturating_sub(earlier.compute_deferred_writebacks),
            compute_deferred_writeback_bytes: self
                .compute_deferred_writeback_bytes
                .saturating_sub(earlier.compute_deferred_writeback_bytes),
            compute_deferred_flushes: self
                .compute_deferred_flushes
                .saturating_sub(earlier.compute_deferred_flushes),
            compute_deferred_flush_bytes: self
                .compute_deferred_flush_bytes
                .saturating_sub(earlier.compute_deferred_flush_bytes),
            readbacks: self.readbacks.saturating_sub(earlier.readbacks),
            readback_bytes: self.readback_bytes.saturating_sub(earlier.readback_bytes),
            seed_uploads: self.seed_uploads.saturating_sub(earlier.seed_uploads),
            seed_gpu_copies: self.seed_gpu_copies.saturating_sub(earlier.seed_gpu_copies),
            seed_gpu_copy_bytes: self
                .seed_gpu_copy_bytes
                .saturating_sub(earlier.seed_gpu_copy_bytes),
            seed_upload_bytes: self
                .seed_upload_bytes
                .saturating_sub(earlier.seed_upload_bytes),
            sampled_reuploads: self
                .sampled_reuploads
                .saturating_sub(earlier.sampled_reuploads),
            sampled_reupload_bytes: self
                .sampled_reupload_bytes
                .saturating_sub(earlier.sampled_reupload_bytes),
            sampled_cache_hits: self
                .sampled_cache_hits
                .saturating_sub(earlier.sampled_cache_hits),
            sampled_identity_hits: self
                .sampled_identity_hits
                .saturating_sub(earlier.sampled_identity_hits),
            sampled_cache_hit_bytes: self
                .sampled_cache_hit_bytes
                .saturating_sub(earlier.sampled_cache_hit_bytes),
            sampled_cache_misses: self
                .sampled_cache_misses
                .saturating_sub(earlier.sampled_cache_misses),
            sampled_gpu_binds: self
                .sampled_gpu_binds
                .saturating_sub(earlier.sampled_gpu_binds),
            sampled_zerocopy_binds: self
                .sampled_zerocopy_binds
                .saturating_sub(earlier.sampled_zerocopy_binds),
            buffer_zerocopy_binds: self
                .buffer_zerocopy_binds
                .saturating_sub(earlier.buffer_zerocopy_binds),
            buffer_snapshot_binds: self
                .buffer_snapshot_binds
                .saturating_sub(earlier.buffer_snapshot_binds),
            gpu_load_hits: self.gpu_load_hits.saturating_sub(earlier.gpu_load_hits),
            seed_imports: self.seed_imports.saturating_sub(earlier.seed_imports),
            target_evicts: self.target_evicts.saturating_sub(earlier.target_evicts),
            desc_pool_grow: self.desc_pool_grow.saturating_sub(earlier.desc_pool_grow),
            target_stale_import: self
                .target_stale_import
                .saturating_sub(earlier.target_stale_import),
            gen_mismatch: self.gen_mismatch.saturating_sub(earlier.gen_mismatch),
            import_presents: self.import_presents.saturating_sub(earlier.import_presents),
            compute_post_wait_skips: self
                .compute_post_wait_skips
                .saturating_sub(earlier.compute_post_wait_skips),
            render_post_wait_skips: self
                .render_post_wait_skips
                .saturating_sub(earlier.render_post_wait_skips),
            ring_retire_blocks: self
                .ring_retire_blocks
                .saturating_sub(earlier.ring_retire_blocks),
            batch_opens: self.batch_opens.saturating_sub(earlier.batch_opens),
            batch_joins: self.batch_joins.saturating_sub(earlier.batch_joins),
            batch_flushes: self.batch_flushes.saturating_sub(earlier.batch_flushes),
            batch_flush_draws: self
                .batch_flush_draws
                .saturating_sub(earlier.batch_flush_draws),
            sampled_free_hits: self
                .sampled_free_hits
                .saturating_sub(earlier.sampled_free_hits),
            sampled_free_allocs: self
                .sampled_free_allocs
                .saturating_sub(earlier.sampled_free_allocs),
            sampled_recycle_admits: self
                .sampled_recycle_admits
                .saturating_sub(earlier.sampled_recycle_admits),
            sampled_recycle_cap_drops: self
                .sampled_recycle_cap_drops
                .saturating_sub(earlier.sampled_recycle_cap_drops),
            target_free_hits: self
                .target_free_hits
                .saturating_sub(earlier.target_free_hits),
            target_free_allocs: self
                .target_free_allocs
                .saturating_sub(earlier.target_free_allocs),
            target_recycle_admits: self
                .target_recycle_admits
                .saturating_sub(earlier.target_recycle_admits),
            target_recycle_cap_drops: self
                .target_recycle_cap_drops
                .saturating_sub(earlier.target_recycle_cap_drops),
        }
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
        counters.note_compute_direct_writeback(512);

        let snapshot = counters.snapshot();
        assert_eq!(snapshot.creates, 1);
        assert_eq!(snapshot.allocs, 1);
        assert_eq!((snapshot.readbacks, snapshot.readback_bytes), (1, 4096));
        assert_eq!(
            (snapshot.seed_uploads, snapshot.seed_upload_bytes),
            (1, 1024)
        );
        assert_eq!(
            (
                snapshot.compute_direct_writebacks,
                snapshot.compute_direct_writeback_bytes
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
