//! Staging / target / readback / command / descriptor pools for warm-path reuse.

#![allow(unsafe_op_in_unsafe_fn)]

mod host_import;
use host_import::{host_import_candidates, HOST_IMPORT_WINDOW_CAP};

use ash::vk;
use ash::vk::Handle;
use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::atomic::Ordering;
use std::time::Instant;

use super::caches::{BindingSig, ComputePipelineKey, LayoutKey, ObjectCaches};
use super::compute_execution::ComputeExecutionDecline;
use super::context::{DeviceContext, FENCE_TIMEOUT_NS};
use super::counters::EngineCounters;
use super::desc_arena::{DescriptorArena, DESC_BLOCK_MAX_SETS};
use super::device_lost::{DeviceLostDecline, DeviceLostOp};
use super::host_import_decline::HostImportDecline;
use super::host_scatter;
use super::stats_reduce;
use super::types::{DrawError, StorageImageFormat, TargetIdentity};
use super::vk_call::{VkCall, VkOp};
use crate::backend::vulkan::caps::MemoryClass;
use crate::backend::vulkan::translate;
use crate::model::ComputeStorageResidencyKey;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub(crate) struct TargetKey {
    pub width: u32,
    pub height: u32,
    pub with_transfer_dst: bool,
}

#[derive(Clone)]
pub(crate) struct BufferSlot {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub size: u64,
}

pub(crate) struct TargetSlot {
    pub image: vk::Image,
    pub memory: vk::DeviceMemory,
    pub view: vk::ImageView,
    pub framebuffer: vk::Framebuffer,
}

pub(crate) struct ResourcePools {
    /// Size-bucketed free host-visible buffers (TRANSFER_SRC | VERTEX | INDEX | STORAGE).
    staging_free: HashMap<u64, Vec<BufferSlot>>,
    /// In-use staging slots returned after submit/wait.
    staging_live: Vec<BufferSlot>,
    /// Staging free-list hits / misses and the miss bucket histogram; see
    /// `note_staging_miss`. Measure-only.
    staging_hits: u64,
    staging_misses: u64,
    staging_miss_bins: [usize; STAGING_BUCKET_BINS],
    staging_miss_us_bins: [u64; STAGING_BUCKET_BINS],
    /// Target images + framebuffers keyed by geometry + render_pass identity.
    targets: HashMap<(TargetKey, u64), TargetSlot>, // u64 = render_pass as u64
    target_order: Vec<(TargetKey, u64)>,
    /// Readback buffers by size.
    readback_free: HashMap<u64, Vec<BufferSlot>>,
    readback_live: Option<BufferSlot>,
    /// Extra live readbacks (compute multi-image / multi-buffer).
    readback_multi_live: Vec<BufferSlot>,
    /// Transient sampled-image pool, keyed by exact image and view geometry.
    sampled_free: HashMap<SampledKey, Vec<SampledSlot>>,
    sampled_live: Vec<SampledSlot>,
    /// Exact-content sampled images retained across draw calls. Hash narrows
    /// candidates only; a hit always requires full byte equality.
    sampled_cache: Vec<ResidentSampledSlot>,
    sampled_cache_bytes: usize,
    /// Storage-image pool for compute.
    storage_image_free: HashMap<StorageImageKey, Vec<StorageImageSlot>>,
    storage_image_live: Vec<StorageImageSlot>,
    /// Protocol-identity keyed compute storage images retained across calls.
    compute_storage_registry: HashMap<ComputeStorageResidencyKey, ResidentStorageImageSlot>,
    /// LRU order for [`Self::compute_storage_registry`], oldest at the front.
    /// A `VecDeque` for the same reason as [`Self::registry_order`]: the cap
    /// sweep's front pop / rotate-to-back is O(1), keeping the sweep O(n).
    compute_storage_order: VecDeque<ComputeStorageResidencyKey>,
    /// Identity-keyed resident target registry (workstream D).
    registry: HashMap<TargetIdentity, ResidentTargetSlot>,
    /// LRU order for [`Self::registry`], oldest at the front. A `VecDeque` so the
    /// cap-eviction sweep's front pop / rotate-to-back is O(1) — the sweep is
    /// then O(n), not the O(n²) a `Vec` front-`remove(0)` would make it under a
    /// large pinned population (measured `reg=512` under multi-4K load).
    registry_order: VecDeque<TargetIdentity>,
    /// Monotonic wall-clock milliseconds for the resident-target idle drain, fed
    /// from the poll heartbeat and each publish ([`Self::advance_registry_touch_and_drain`]).
    /// Each admit/hit/present stamps its slot's `last_touch_ms` with this value;
    /// the drain reclaims non-pinned residents whose stamp is `IDLE_TARGET_AGE_MS`
    /// behind. Wall-clock (not a publish counter) so it keeps advancing when the
    /// guest stops publishing, returning idle VRAM to baseline on a static page.
    idle_clock_ms: u64,
    /// Wall-clock ms of the last reclaim pass — enforces `IDLE_DRAIN_INTERVAL_MS`
    /// spacing so the ~244 Hz poll cadence cannot empty the registry at once.
    last_drain_ms: u64,
    /// Consecutive fired idle-drain passes that reclaimed **zero** registry
    /// residents. A pass that drains ≥1 victim means the working set is still
    /// churning (active video keeps aging out old frame RTs), so we reset to 0.
    /// The HOST_VISIBLE buffer pool trim (a full `vkAllocateMemory` re-alloc on
    /// the upload hot path when it refills) only fires once this crosses
    /// `SETTLED_PASSES_FOR_BUFFER_TRIM`, so a single quiet pass mid-video cannot
    /// steal a 64 MiB staging buffer and spike the next upload's latency. The
    /// image/slab trims stay ungated — they refill via cheap slab suballocation.
    settled_drain_passes: u32,
    /// Persistent command pool; each ring slot owns one primary CB.
    cmd_pool: vk::CommandPool,
    /// Growable descriptor-pool arena (FREE_DESCRIPTOR_SET blocks). Grows a new
    /// block on exhaustion instead of hard-failing the draw; sets are freed
    /// per entry, paired with their owning block. See [`DescriptorArena`].
    desc_arena: DescriptorArena,
    /// N-deep in-flight ring: each slot is one CB + fence + the cleanup it
    /// owes. Entries rotate through slots; a slot is reused only after its
    /// fence retires (begin_entry blocks on the oldest when the ring is full).
    slots: Vec<CmdSlot>,
    /// Slot the current (or most recently begun) entry records into.
    cur: usize,
    /// Submitted-but-unretired slot count. While nonzero, destroying any GPU
    /// object a prior CB may reference is unsafe — dispose() defers those
    /// handles to `graveyard` until every in-flight fence retires.
    in_flight: usize,
    /// Handles displaced (cache eviction, registry replace) while a CB was in
    /// flight; destroyed once in_flight returns to 0.
    graveyard: Vec<DeferredHandle>,
    /// Cumulative sampled-cache recycle diagnostics (surfaced via
    /// `recycle_stats` into `CounterSnapshot`; see there for semantics).
    /// Plain u64 — `ResourcePools` is only ever touched under the engine lock.
    sampled_free_hits: u64,
    sampled_free_allocs: u64,
    sampled_recycle_admits: u64,
    sampled_recycle_cap_drops: u64,
    /// Resident-target recycle pool: images displaced from the identity registry
    /// (generation bump / geometry change / LRU), held by (geometry, format) for
    /// reuse instead of destroyed. Kills the per-frame `vkCreateImage`+
    /// `vkAllocateMemory` storm a per-frame-generation target (video) would
    /// otherwise pay (see [`TargetRecycleKey`]). Bounded per key.
    target_free: HashMap<TargetRecycleKey, Vec<FreeTargetImage>>,
    /// Cumulative resident-target recycle diagnostics (plain u64 — touched only
    /// under the engine lock; surfaced via `target_recycle_stats`).
    target_free_hits: u64,
    target_free_allocs: u64,
    target_recycle_admits: u64,
    target_recycle_cap_drops: u64,
    /// Transient compute-storage recycle pool diagnostics (plain u64 — touched
    /// only under the engine lock). `admits` counts slots returned to
    /// `storage_image_free` for reuse; `cap_drops` counts slots destroyed because
    /// a per-key or the global cap was full (an all-new-geometry compute burst).
    storage_recycle_admits: u64,
    storage_recycle_cap_drops: u64,
    /// VK_EXT_external_memory_host imports over guest-RAM host VAs (direct
    /// RAMBlock aliases — stable for the VM lifetime). Each entry carries a
    /// TRANSFER_SRC buffer bound over the whole import for zero-copy guest
    /// gathers. Entries are admitted only while [`HOST_IMPORT_WINDOW_BUDGET`]
    /// has room and released only by the idle sweep, never to make room for
    /// another.
    host_imports: Vec<HostImportRegion>,
    /// Monotonic resolve counter stamped into `HostImportRegion::last_touch`.
    host_import_touch: u64,
    /// Submission epoch stamped into `HostImportRegion::last_epoch`. Advanced by
    /// [`Self::finish_entry_async`] — the one place a CB goes in flight — so
    /// "touched since the last submit" needs no begin/end bracketing at the
    /// five `host_import_resolve` call sites.
    host_import_epoch: u64,
    /// Per-bucket-base occupancy of the windows this pool has imported, kept
    /// across evict/re-import so it describes the guest's working set rather
    /// than the thrash rate. Bounded by guest RAM / [`HOST_IMPORT_WINDOW_CAP`].
    host_import_occupancy: std::collections::BTreeMap<usize, host_import::WindowOccupancy>,
    /// Windowed create/evict census. Equal rates mean the working set does not
    /// fit the budget and the pool is re-importing what it just released.
    host_import_creates: u64,
    host_import_evictions: u64,
    /// One-shot guards for fail-visible import-budget declines.
    host_import_count_cap_logged: bool,
    host_import_zero_len_logged: bool,
    host_import_no_ext_logged: bool,
    host_import_byte_cap_logged: bool,
    /// GPU-side present-proxy stats reduction (the zero-copy oracle).
    stats_reduce: stats_reduce::StatsReducePool,
    host_scatter: host_scatter::HostScatterPool,
    /// Open draw batch: a ring slot whose CB is still recording deferred
    /// same-target draws (submit pending). While `Some`, that CB references
    /// live GPU objects exactly like an in-flight CB, so dispose/graveyard
    /// treat it as in flight; every path that claims a slot or quiesces the
    /// ring flushes it first ([`Self::batch_flush`]).
    open_batch: Option<OpenBatch>,
    /// Offset suballocator for DEVICE_LOCAL optimal images (targets, sampled,
    /// storage, resident registry). Sub-allocates many image binds from a few
    /// large `VkDeviceMemory` blocks to collapse the per-image
    /// `vkAllocateMemory` storm of a layer-tree reflow burst
    /// ([[present-thrash-proxies]] bug #2). Live sub-allocations are keyed by
    /// `vk::Image` handle, so the free path routes through it with just the
    /// image.
    slab: super::slab::SlabPool,
    initialized: bool,
}

/// State of the deferred-submit draw batch (draw-batching increment 1): the
/// opener's ring slot CB stays in recording state across joinable same-target
/// draws; per-draw descriptor sets and sampled-cache admissions accumulate
/// here and seal as ONE entry at flush.
pub(crate) struct OpenBatch {
    cb: vk::CommandBuffer,
    fence: vk::Fence,
    identity: TargetIdentity,
    width: u32,
    height: u32,
    bgra: bool,
    draws: u64,
    /// Per-draw descriptor sets paired with the arena block they were allocated
    /// from, so the flush-time free routes each set to its owning pool.
    dsets: Vec<(vk::DescriptorSet, vk::DescriptorPool)>,
    sampled_retains: Vec<(
        vk::Image,
        std::sync::Arc<Vec<u8>>,
        Option<crate::backend::vulkan::engine::SampledContentIdentity>,
    )>,
}

/// One cached VK_EXT_external_memory_host import over a guest-RAM host-VA
/// range, with a TRANSFER_SRC buffer bound over the whole range. `base`/`len`
/// are the aligned import bounds; any request fully inside them resolves to
/// `(buffer, ptr - base)`.
pub(crate) struct HostImportRegion {
    base: usize,
    len: u64,
    memory: vk::DeviceMemory,
    buffer: vk::Buffer,
    /// Monotonic resolve stamp ([`ResourcePools::host_import_touch`]), bumped on
    /// every hit and on create. Strict recency order for LRU eviction — the
    /// wall-clock stamp below cannot serve this, because a burst resolves
    /// hundreds of runs inside one millisecond and would tie.
    last_touch: u64,
    /// Submission epoch ([`ResourcePools::host_import_epoch`]) at `last_touch`.
    /// Equal to the live epoch ⇒ this region was resolved since the last submit,
    /// so its buffer may sit in a caller's not-yet-recorded run list. See
    /// [`ResourcePools::plan_host_import_eviction`].
    last_epoch: u64,
    /// Wall-clock stamp from `idle_clock_ms` for the idle sweep's age cutoff,
    /// matching the resident-target / sampled-cache / compute-storage pools.
    last_touch_ms: u64,
}

/// Arm's retained `mach_vm_remap` views are separate small VM regions, so a
/// desktop legitimately exceeds the old 32-entry Linux/RAMBlock assumption.
/// Bound both object count and total registered bytes: the count limits Vulkan
/// objects while the byte cap prevents many maximum-size windows from pinning
/// the guest's whole RAM allocation.
const HOST_IMPORT_REGION_CAP: usize = 512;

/// How many maximal [`HOST_IMPORT_WINDOW_CAP`] windows may be resident at once.
///
/// The byte cap has to admit **more than one** maximal window or the window
/// bucketing is self-defeating: `capped_import_window` deliberately rounds a
/// span up to its whole 1 GiB VMA bucket, so the first import spends the entire
/// budget and every later span outside that one bucket declines
/// `host_import_total_byte_cap` for the VM's lifetime. Regions are never
/// evicted, so that decline is permanent — a live x86 boot showed exactly this:
/// one `host_import_region len=0x40000000 regions=1`, then every import-present
/// store falling back to `run_unimportable` and each deferred render window
/// dying as `deferred_flush_lost kind=render reason=host_import_resolve`, which
/// is what put the compositor's frames on screen as black.
///
/// The floor is structural: an import-present store maps every run of one
/// surface and imports them all before its DMA, so a surface whose pages
/// straddle several buckets needs all of them resident simultaneously.
///
/// This bounds **how much is resident, not how much can be served.** A span
/// outside the resident set is not refused work: `host_import_resolve` declines,
/// and the caller writes it through the CPU byte path. So the budget is a
/// hit-rate knob, and the resolve path admits only when admission is free —
/// releasing a resident window to make room costs a whole 1 GiB re-import to
/// save a few megabytes of `memcpy`, and then costs it again next frame because
/// the released window is in the same working set.
///
/// 8 is the measured steady state of an x86 macOS desktop (Finder + Safari on
/// apple.com): eight *consecutive* 1 GiB buckets. A real browsing session is
/// wider than that and no window size closes the gap. One measured run —
/// twelve heavy pages kept live, then sixty seconds of sustained compositing —
/// touched **14** buckets, and `host_import_density` put 6.4 GiB of pages inside
/// them at 46 % occupancy per bucket, so a finer `HOST_IMPORT_WINDOW_CAP` would
/// halve the bytes at best while multiplying the region count by hundreds. The
/// spread is the guest's, and the overflow has to be served rather than shuffled.
///
/// The `host_import_region` / `host_import_evict` census stays load-bearing, and
/// now reads the other way round: `creates` should approach the bucket count and
/// `evictions` should come only from the idle sweep. Comparable rates would mean
/// something started evicting under pressure again. The same session measured
/// 2247 creates against 2246 evictions before that stopped, with victims logged
/// `age_ms=0`, an import costing 19.3 ms — longer than a 60 Hz frame — and one
/// drain tranche spending 1342 ms inside `zc_import_us`.
///
/// The cost of *not* evicting is that the resident set is whichever buckets
/// arrived first and it only turns over on the idle sweep, so a long session can
/// hold eight lukewarm windows while a hotter ninth pays `memcpy` every frame.
/// That is bounded by construction — the fallback is a copy of the span, not of
/// the window — and it is the cheaper side of the trade by more than an order of
/// magnitude.
const HOST_IMPORT_WINDOW_BUDGET: u64 = 8;

/// Ceiling on host pages pinned for GPU DMA. Bounded pinning is the standing
/// rule (AGENTS.md): this stays a small multiple of the window, never the whole
/// guest RAMBlock.
const HOST_IMPORT_TOTAL_BYTE_CAP: u64 = HOST_IMPORT_WINDOW_CAP * HOST_IMPORT_WINDOW_BUDGET;

fn host_import_budget(
    region_count: usize,
    imported_bytes: u64,
    candidate_bytes: u64,
) -> Result<(), HostImportDecline> {
    if region_count >= HOST_IMPORT_REGION_CAP {
        return Err(HostImportDecline::RegionCount);
    }
    if imported_bytes.saturating_add(candidate_bytes) > HOST_IMPORT_TOTAL_BYTE_CAP {
        return Err(HostImportDecline::TotalBytes);
    }
    Ok(())
}

impl ResourcePools {
    /// Indices of import windows the idle sweep may release: not in the live
    /// resolve epoch, and untouched since `cutoff`. At most `max` per pass, so a
    /// fired pass cannot empty the set in one go.
    ///
    /// Device-free for the same reason as [`Self::plan_host_import_eviction`],
    /// and it repeats that function's epoch guard rather than trusting the age
    /// cutoff alone: `last_touch_ms` comes from the poll-driven `idle_clock_ms`,
    /// which does not advance during a burst of resolves, so a whole
    /// all-or-nothing run list can share one stale millisecond stamp.
    fn plan_host_import_idle_release(&self, cutoff: u64, max: usize) -> Vec<usize> {
        (0..self.host_imports.len())
            .filter(|&i| {
                let r = &self.host_imports[i];
                r.last_epoch != self.host_import_epoch && r.last_touch_ms <= cutoff
            })
            .take(max)
            .collect()
    }

    /// Idle-sweep entry point for host-import windows: the same planner as
    /// [`Self::plan_host_import_idle_release`], but with the cutoff derived from
    /// the dedicated [`HOST_IMPORT_IDLE_AGE_MS`] rather than the cheap-VRAM
    /// [`IDLE_TARGET_AGE_MS`]. Keeping the cutoff computation here — not at the
    /// drain call site — is what lets the thrash regression test exercise the
    /// real age gate a window is held to.
    fn plan_host_import_idle_sweep(&self, now_ms: u64, max: usize) -> Vec<usize> {
        let cutoff = now_ms.saturating_sub(HOST_IMPORT_IDLE_AGE_MS);
        self.plan_host_import_idle_release(cutoff, max)
    }

    /// Total bytes currently pinned by import windows.
    fn host_import_bytes(&self) -> u64 {
        self.host_imports
            .iter()
            .fold(0u64, |total, region| total.saturating_add(region.len))
    }
}

fn terminal_host_import_error(
    last_error: Option<DrawError>,
    host_ptr: usize,
    len: u64,
    alignment: u64,
) -> DrawError {
    last_error.unwrap_or_else(|| {
        DrawError::HostImport(HostImportDecline::NoValidWindow {
            host_ptr,
            len,
            alignment,
        })
    })
}

fn resolve_scatter_regions(
    runs: &[host_scatter::ScatterRun],
    spans: &[host_scatter::ScatterSpan],
    buffers: &[(vk::Buffer, u64)],
    width: u32,
    height: u32,
) -> Result<Vec<host_scatter::ScatterRegion>, super::reason::HostPresentDecline> {
    let mut regions = Vec::with_capacity(spans.len());
    for (span_index, span) in spans.iter().enumerate() {
        let Some(run) = runs.get(span.run_index) else {
            return Err(
                super::reason::HostPresentDecline::RunsScatterRunIndexOutOfBounds {
                    span_index,
                    run_index: span.run_index,
                    run_count: runs.len(),
                },
            );
        };
        let Some(&(buffer, window_offset)) = buffers.get(span.run_index) else {
            return Err(
                super::reason::HostPresentDecline::RunsScatterBufferIndexOutOfBounds {
                    span_index,
                    run_index: span.run_index,
                    buffer_count: buffers.len(),
                },
            );
        };
        if span.texels == 0 {
            return Err(super::reason::HostPresentDecline::RunsScatterZeroTexels { span_index });
        }
        let source_end = span.x.checked_add(span.texels);
        if source_end.is_none_or(|end| end > width) || span.y >= height {
            return Err(
                super::reason::HostPresentDecline::RunsScatterSourceOutOfBounds {
                    span_index,
                    x: span.x,
                    y: span.y,
                    texels: span.texels,
                    width,
                    height,
                },
            );
        }
        let len = u64::from(span.texels) * 4;
        let span_end = span.dst_offset.checked_add(len).ok_or(
            super::reason::HostPresentDecline::RunsScatterSpanEndOverflow {
                span_index,
                dst_offset: span.dst_offset,
                len,
            },
        )?;
        if span_end > run.ptr_len {
            return Err(super::reason::HostPresentDecline::RunsScatterOob {
                dst_offset: span.dst_offset,
                len,
                cap: run.ptr_len,
            });
        }
        let buffer_offset = window_offset.checked_add(span.dst_offset).ok_or(
            super::reason::HostPresentDecline::RunsScatterBufferOffsetOverflow {
                span_index,
                window_offset,
                dst_offset: span.dst_offset,
            },
        )?;
        regions.push(host_scatter::ScatterRegion {
            buffer,
            buffer_offset,
            x: span.x,
            y: span.y,
            texels: span.texels,
        });
    }
    Ok(regions)
}

#[derive(Clone, Copy, Debug)]
enum PresentStatsSetup {
    Shader,
    Layout,
    Pipeline,
}

impl PresentStatsSetup {
    fn name(self) -> &'static str {
        match self {
            Self::Shader => "shader",
            Self::Layout => "layout",
            Self::Pipeline => "pipeline",
        }
    }

    fn discriminant(self) -> u64 {
        match self {
            Self::Shader => 1,
            Self::Layout => 2,
            Self::Pipeline => 3,
        }
    }
}

fn present_stats_setup_decline(
    setup: PresentStatsSetup,
    error: &DrawError,
) -> crate::observe::Emit {
    crate::observe::Emit::decline("stats_reduce", error).field("setup", setup.name())
}

/// One in-flight ring slot: a primary CB, its fence (created unsignaled;
/// reset immediately after every successful wait), and — while the CB is in
/// flight — the cleanup its entry owes.
struct CmdSlot {
    cmd_buf: vk::CommandBuffer,
    fence: vk::Fence,
    pending: Option<PendingGpuCleanup>,
}

/// In-flight ring depth: the next draw/dispatch records + submits while
/// previous no-readback CBs still run, removing the retire-before-acquire
/// stall of the single-slot engine. Cross-CB ordering needs no barriers
/// beyond the recorded ones — one queue, tracked layouts — so depth only
/// trades burst headroom against cleanup latency.
///
/// Depth 8 (2026-07-19): once the bufprep staging fix cut per-draw CPU prepare,
/// the draw path blocked in `begin_entry` ~61 µs/draw on slot N+1's fence — the
/// CPU outran the 3-deep GPU pipeline under Safari fast-scroll. Deepening the ring
/// lets the CPU stay ahead so the GPU stays fed: verified `retire_wait` 61 → 17
/// µs/draw, `present_hz` ~40 → ~50, correctness clean (residue byte-flat,
/// zc_flush_stale = rmemo_stale = mid_sw = 0). It was submit/fence-bubble-bound,
/// not GPU-compute-bound. Cost is 8 command buffers + fences + up to 8 slots'
/// pooled staging live at once — bounded, pooled. `retire_wait` still ~17 µs, so
/// a deeper ring or render-pass batching (only ~37 % of draws join a shared pass)
/// may reclaim more.
const RING_DEPTH: usize = 8;

/// A GPU object displaced while a CB may still reference it. Destroyed only
/// once every in-flight fence has retired.
pub(crate) enum DeferredHandle {
    Image {
        image: vk::Image,
        view: vk::ImageView,
        memory: vk::DeviceMemory,
    },
    /// A sampled-cache slot evicted by the LRU/byte cap. Instead of destroying
    /// it, the drain returns it to `sampled_free` for reuse (bounded per key) so
    /// a content-changing sampled input (live tile / video frame) re-uploads
    /// into a recycled image instead of paying a fresh `vkAllocateMemory` every
    /// frame. Routed through the same in-flight-safe deferral as destroys: an
    /// in-flight CB may still sample the evicted image, so it only rejoins the
    /// free list once `in_flight == 0`.
    RecycleSampled(SampledSlot),
    /// A resident render-target image displaced from the registry (generation
    /// bump / geometry change / LRU). Instead of destroying it, the drain
    /// returns it to `target_free` for reuse (bounded per key) so a per-frame
    /// content-changing target (video output) re-renders into a recycled image
    /// instead of paying a fresh `vkCreateImage`+`vkAllocateMemory` every
    /// frame. Same in-flight-safe deferral as destroys: an in-flight CB may
    /// still reference the displaced image, so it only rejoins the free list
    /// once `in_flight == 0`.
    RecycleTarget(FreeTargetImage),
    /// A `VK_EXT_external_memory_host` import window evicted by the LRU/byte
    /// budget or released by the idle sweep. Terminal destroy, never recycled:
    /// the memory is an alias of guest RAM at one fixed host VA, so it is useful
    /// to exactly one base address and there is nothing to hand a later import.
    /// Routed through the same in-flight-safe deferral as every other displaced
    /// handle — a submitted CB may still be DMAing through this buffer.
    HostImport {
        buffer: vk::Buffer,
        memory: vk::DeviceMemory,
    },
    Framebuffer(vk::Framebuffer),
    Pipeline(vk::Pipeline),
    PipelineLayout(vk::PipelineLayout),
    DescriptorSetLayout(vk::DescriptorSetLayout),
    RenderPass(vk::RenderPass),
    ShaderModule(vk::ShaderModule),
    Sampler(vk::Sampler),
}

impl ResourcePools {
    /// Terminal destroy of a deferred handle. Image variants free their backing
    /// memory through the slab suballocator (`free_image` releases the
    /// sub-range; a non-slab image falls back to a raw `vkFreeMemory` so mixed
    /// slab/1:1 images both free correctly). Non-memory objects are destroyed
    /// directly.
    unsafe fn destroy_deferred_handle(&mut self, device: &ash::Device, handle: DeferredHandle) {
        match handle {
            DeferredHandle::Image {
                image,
                view,
                memory,
            } => {
                // Destroy the image before releasing its memory: `free_image`
                // may `vkFreeMemory` the whole block if this was its last live
                // sub-allocation, and freeing memory under a live image is UB.
                device.destroy_image_view(view, None);
                device.destroy_image(image, None);
                if !self.slab.free_image(device, image) {
                    device.free_memory(memory, None);
                }
            }
            DeferredHandle::RecycleSampled(slot) => {
                device.destroy_image_view(slot.view, None);
                device.destroy_image(slot.image, None);
                if !self.slab.free_image(device, slot.image) {
                    device.free_memory(slot.memory, None);
                }
            }
            DeferredHandle::RecycleTarget(img) => {
                device.destroy_image_view(img.view, None);
                device.destroy_image(img.image, None);
                if !self.slab.free_image(device, img.image) {
                    device.free_memory(img.memory, None);
                }
            }
            DeferredHandle::HostImport { buffer, memory } => {
                // Same order and the same lack of slab involvement as the
                // teardown sweep in `destroy_all`: imported memory is not a slab
                // sub-allocation, so it is freed directly once its buffer is
                // gone. The host pages themselves belong to QEMU and outlive us.
                device.destroy_buffer(buffer, None);
                device.free_memory(memory, None);
            }
            DeferredHandle::Framebuffer(fb) => device.destroy_framebuffer(fb, None),
            DeferredHandle::Pipeline(p) => device.destroy_pipeline(p, None),
            DeferredHandle::PipelineLayout(pl) => device.destroy_pipeline_layout(pl, None),
            DeferredHandle::DescriptorSetLayout(dsl) => {
                device.destroy_descriptor_set_layout(dsl, None)
            }
            DeferredHandle::RenderPass(rp) => device.destroy_render_pass(rp, None),
            DeferredHandle::ShaderModule(s) => device.destroy_shader_module(s, None),
            DeferredHandle::Sampler(s) => device.destroy_sampler(s, None),
        }
    }
}

/// Cleanup owed by an entry that skipped its post-submit fence wait: the
/// descriptor set, every transient pool slot the CB references (moved out of
/// the live lists at seal time so a concurrent entry cannot recycle them),
/// and the render path's sampled-content cache admissions — deferred because
/// admission can EVICT (destroy) cache images the in-flight CB may sample.
pub(crate) struct PendingGpuCleanup {
    dsets: Vec<(vk::DescriptorSet, vk::DescriptorPool)>,
    staging: Vec<BufferSlot>,
    readback: Vec<BufferSlot>,
    sampled: Vec<SampledSlot>,
    storage_images: Vec<StorageImageSlot>,
    sampled_retains: Vec<(
        vk::Image,
        std::sync::Arc<Vec<u8>>,
        Option<crate::backend::vulkan::engine::SampledContentIdentity>,
    )>,
}

pub(crate) struct SampledSlot {
    pub image: vk::Image,
    pub memory: vk::DeviceMemory,
    pub view: vk::ImageView,
    pub width: u32,
    pub height: u32,
    pub layers: u32,
    pub volume: bool,
    pub cube: bool,
    pub arrayed: bool,
    /// The image was created as a Vulkan 1D (`TYPE_1D` / `TYPE_1D_ARRAY`) image
    /// because the shader's sampled binding reflects a Metal `texture1d` /
    /// `texture1d_array` (color-transfer LUTs). Part of the pool key: a 1D view
    /// and a `height==1` 2D view are byte-identical images but incompatible
    /// descriptor types, so a recycled slot must never cross that boundary.
    pub one_dim: bool,
    pub format: ash::vk::Format,
    /// The view's component mapping, from the decoded type-8 swizzle. Part of
    /// the pool key because it is baked into the `VkImageView`: a recycled slot
    /// whose view swizzles differently would silently remap a later bind's
    /// channels. Identity is the overwhelmingly common case and keeps its own
    /// free list, so a rare swizzled bind cannot fragment the hot one.
    pub swizzle: crate::contract::pixel_format::SwizzlePlan,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
struct SampledKey {
    width: u32,
    height: u32,
    layers: u32,
    volume: bool,
    cube: bool,
    arrayed: bool,
    one_dim: bool,
    format: ash::vk::Format,
    swizzle: crate::contract::pixel_format::SwizzlePlan,
}

impl SampledSlot {
    fn key(&self) -> SampledKey {
        SampledKey {
            width: self.width,
            height: self.height,
            layers: self.layers,
            volume: self.volume,
            cube: self.cube,
            arrayed: self.arrayed,
            one_dim: self.one_dim,
            format: self.format,
            swizzle: self.swizzle,
        }
    }

    fn handles(&self) -> Self {
        Self {
            image: self.image,
            memory: self.memory,
            view: self.view,
            width: self.width,
            height: self.height,
            layers: self.layers,
            volume: self.volume,
            cube: self.cube,
            arrayed: self.arrayed,
            one_dim: self.one_dim,
            format: self.format,
            swizzle: self.swizzle,
        }
    }
}

struct ResidentSampledSlot {
    slot: SampledSlot,
    /// 128-bit fingerprint of the retained content (see [`sampled_content_hash`]).
    /// This *is* the match key on the content-fallback path — no byte copy is
    /// kept, so a hit binds the retained image without a full-frame `memcmp`.
    content_hash: u128,
    /// Byte length of the content this slot was admitted with, kept only for the
    /// LRU byte-cap accounting (the bytes themselves are not retained).
    content_len: usize,
    /// Producer identity of the retained content; lets a same-identity,
    /// same-generation rebind skip the content hash + compare entirely.
    identity: Option<crate::backend::vulkan::engine::SampledContentIdentity>,
    /// Value of [`ResourcePools::idle_clock_ms`] at this entry's last use (admit
    /// or `find_cached_sampled` hit). The idle drain
    /// ([`ResourcePools::advance_registry_touch_and_drain`]) reclaims an entry
    /// once its touch falls `IDLE_TARGET_AGE_MS` behind the clock — so a settled
    /// video session's frame textures (the ≤128 MiB sampled cache) are returned
    /// to the driver at idle instead of pinned for the guest lifetime, while an
    /// actively-sampled entry (hit every frame) never ages out. Mirrors the
    /// resident-target registry drain; the sampled cache is the analogous
    /// upload-side pool the buffer/target idle trims already cover.
    last_touch_ms: u64,
}

/// Geometry+format key for storage-image pool free lists.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub(crate) struct StorageImageKey {
    pub width: u32,
    pub height: u32,
    pub layers: u32,
    pub format: StorageImageFormat,
    pub one_dim: bool,
    pub arrayed: bool,
    pub volume: bool,
    /// Read-only sampled descriptor instead of writable storage descriptor.
    pub sampled_only: bool,
}

#[derive(Clone, Copy)]
pub(crate) struct StorageImageSlot {
    pub image: vk::Image,
    pub memory: vk::DeviceMemory,
    pub view: vk::ImageView,
    pub key: StorageImageKey,
    pub array_layers: u32,
    pub extent_depth: u32,
}

pub(crate) struct ResidentStorageImageUse {
    pub slot: StorageImageSlot,
    pub layout: vk::ImageLayout,
    pub generation_match: bool,
}

struct ResidentStorageImageSlot {
    slot: StorageImageSlot,
    generation: u32,
    layout: vk::ImageLayout,
    /// Deferred-writeback pin: the resident is the only copy of this content
    /// (guest pages are stale) — LRU eviction must skip it until the caller
    /// flushes and unpins.
    pinned: bool,
    /// Value of `ResourcePools::idle_clock_ms` (wall-clock ms) at this resident's
    /// last use (admit or `acquire_resident_storage_image` hit). The idle drain
    /// ([`ResourcePools::advance_registry_touch_and_drain`]) reclaims a non-pinned
    /// resident once its touch falls `IDLE_TARGET_AGE_MS` behind the clock — so a
    /// compute-heavy burst's stale residents (a settled page's blur/decode storage
    /// images) are returned to the driver instead of pinning up to
    /// `COMPUTE_STORAGE_REGISTRY_CAP` standalone VkDeviceMemory allocations for the
    /// guest lifetime, while an actively-dispatched resident (touched every pass)
    /// never ages out. Mirrors [`ResidentTargetSlot::last_touch_ms`].
    last_touch_ms: u64,
}

/// Persistent GPU render-target slot (identity-keyed registry, workstream D).
pub(crate) struct ResidentTargetSlot {
    pub image: vk::Image,
    pub memory: vk::DeviceMemory,
    pub view: vk::ImageView,
    pub framebuffer: vk::Framebuffer,
    pub render_pass: vk::RenderPass,
    pub width: u32,
    pub height: u32,
    pub generation: u64,
    pub content_ready: bool,
    /// Last known layout (tracked for correct barriers).
    pub layout: vk::ImageLayout,
    /// Attachment format: true = B8G8R8A8_UNORM (guest scanout order), false =
    /// R8G8B8A8_UNORM. A format change forces image recreate (not just FB).
    pub bgra: bool,
    /// Concrete Vulkan attachment format. For the primary single-RT path this
    /// is derived from `bgra`; MRT secondary residents (e.g. the RG16Float
    /// vibrancy mask) carry a format `bgra` cannot express, so reuse is keyed
    /// on this exact format.
    pub color_format: vk::Format,
    /// Deferred render-Store pin count: this target's content exists only on
    /// the GPU (guest pages stale). The registry LRU sweep skips slots with a
    /// nonzero count. A count (not a bool) because a shared `OutputGroup`
    /// identity is pinned independently by each member's deferred window —
    /// the first member's flush must not expose the image to eviction while
    /// a peer's window is still armed.
    pub pin_count: u32,
    /// Value of `ResourcePools::idle_clock_ms` (wall-clock ms) at this target's
    /// last use (admit, `registry_ensure` hit, or present touch). The idle drain
    /// ([`ResourcePools::advance_registry_touch_and_drain`]) reclaims a non-pinned
    /// resident once its touch falls `IDLE_TARGET_AGE_MS` behind the current
    /// clock — so a burst's stale targets (a settled YouTube page's thumbnail RTs)
    /// are reclaimed instead of pinning VRAM at the high `REGISTRY_CAP` for the
    /// guest lifetime, while an actively-drawn target (touched every frame) never
    /// ages out.
    pub last_touch_ms: u64,
}

/// Geometry+format key for the resident-target recycle pool (`target_free`).
/// The registry keys targets by [`TargetIdentity`], which folds `generation`
/// into Hash/Eq — a per-frame content-changing target (video output, a live
/// compositor RT) bumps its generation every frame, so every frame is a *new*
/// registry key, a `registry_ensure` miss, and a full `vkCreateImage` +
/// `vkAllocateMemory`. Recycling by (geometry, format) — which is stable across
/// those generation bumps — lets the freed image+memory+view be reused instead
/// of reallocated, so the per-frame alloc storm collapses to alloc-once.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
struct TargetRecycleKey {
    width: u32,
    height: u32,
    format: vk::Format,
}

/// A resident-target image+memory+view displaced from the registry (generation
/// bump / geometry change / LRU eviction) and held for reuse instead of
/// destroyed. The framebuffer is NOT retained — it binds one specific
/// `render_pass`, is disposed separately, and a reused image builds a fresh
/// one. Carries its own geometry so [`ResourcePools::try_recycle_target`] can
/// bucket it without a separate key argument (mirrors [`SampledSlot`]).
pub(crate) struct FreeTargetImage {
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    width: u32,
    height: u32,
    format: vk::Format,
}

impl FreeTargetImage {
    fn key(&self) -> TargetRecycleKey {
        TargetRecycleKey {
            width: self.width,
            height: self.height,
            format: self.format,
        }
    }
}

/// Measure-only snapshot of what the target registry holds at one geometry,
/// produced by [`ResourcePools::registry_geom_census`] to classify an
/// `export_present_miss outcome=…` census event. Not a product-control input.
#[derive(Clone, Debug, Default)]
pub(crate) struct RegistryGeomCensus {
    /// Total registry occupancy (all geometries).
    pub total: usize,
    /// `Some(content_ready)` if an `OutputGroup` resident exists at this geom.
    pub group: Option<bool>,
    /// Every `Surface` resident at this geom: `(id, generation, content_ready)`.
    pub surfaces: Vec<(u32, u64, bool)>,
    /// Count of `Gva` residents at this geom.
    pub gva: usize,
}

/// One reading of the pool occupancy that feeds the always-on cap-pressure
/// census ([`ResourcePools::cap_pressure_occupancy`]). `registry_pinned` is the
/// count of resident targets held against LRU eviction by a deferred write
/// window — when it approaches `registry_len`, the registry has soft-exceeded
/// its slot cap (the LRU sweep cannot evict pinned slots) and the non-pinned
/// tail thrashes.
pub(crate) struct CapPressureOccupancy {
    pub registry_len: usize,
    pub registry_cap: usize,
    pub registry_pinned: usize,
    /// Live descriptor-arena pool blocks (1 = the pool never grew). A value > 1
    /// means a draw/dispatch burst exhausted a block and the arena grew rather
    /// than dropping the draw — the descriptor cap-pressure signal.
    pub desc_blocks: usize,
    pub sampled_len: usize,
    pub sampled_cap: usize,
    pub sampled_bytes: usize,
    pub sampled_byte_cap: usize,
    pub graveyard_len: usize,
    /// Physical VkDeviceMemory the slab holds from the driver, and how much of it
    /// is unbound — the direct VRAM footprint + fragmentation signal.
    pub slab: crate::backend::vulkan::engine::slab::SlabOccupancy,
    /// Images cached in the resident-target / sampled recycle pools (each pins a
    /// live slab sub-allocation, so they keep blocks from emptying).
    pub target_free_imgs: usize,
    pub sampled_free_imgs: usize,
    /// Transient compute-storage images cached in `storage_image_free`. Each is a
    /// standalone (non-slab) VkDeviceMemory, so — unlike the two pools above — it
    /// does NOT show up in the slab `resident_mb`/`live_subs` census; surfaced
    /// separately (`stfree`) so a compute-storage recycle leak is visible.
    pub storage_free_imgs: usize,
    /// Cumulative compute-storage recycle admits / cap-drops (`st_admit`/`st_drop`
    /// on the census). A rising `st_drop` under a flat `stfree` is a cap-bounded
    /// leak — the workload keeps producing new-geometry storage images the cap
    /// destroys rather than a genuinely reused working set.
    pub storage_recycle_admits: u64,
    pub storage_recycle_cap_drops: u64,
    /// Live compute-storage RESIDENTS (`compute_storage_registry`) and how many are
    /// pinned (deferred-writeback, only-copy-on-GPU). Bounded by
    /// `COMPUTE_STORAGE_REGISTRY_CAP=64` with LRU eviction AND a render-registry-style
    /// idle-age drain ([`ResourcePools::trim_aged_compute_storage`]) that returns
    /// non-pinned residents `IDLE_TARGET_AGE_MS` after last use — each resident is a
    /// standalone (non-slab) VkDeviceMemory invisible to the slab census, so without
    /// the drain a settled compute-heavy session would pin up to 64 whole allocations
    /// for the guest lifetime. Surfaced (`st_res`/`st_pin`) so the drain's effect (and
    /// any pinned-resident leak it cannot reclaim) stays visible at idle.
    pub storage_resident: usize,
    pub storage_resident_pinned: usize,
    /// HOST_VISIBLE staging/readback buffers cached for reuse. System RAM on a
    /// discrete GPU, but shared-with-the-guest RAM on an iGPU (portability
    /// target), so their bytes are the "least host memory" signal there. Bucketed
    /// by size, so bounded by concurrency — measured to confirm no unbounded
    /// growth under many large (4K-frame) uploads.
    pub staging_free_bytes: u64,
    pub readback_free_bytes: u64,
}

/// Cap on the **non-pinned** (LRU-evictable) resident-target population — the
/// active render working set. Pinned slots (deferred-write windows, each holding
/// content only on the GPU, bounded separately by
/// `import_present::RENDER_DEFERRED_WINDOW_CAP`) are **excluded** from this count
/// (see the eviction loops): counting them would force the still-in-use active
/// targets out whenever a compositing burst pins hundreds, thrashing exactly the
/// targets a draw is about to reuse (measured `reg=512/512 evicts=168` under a
/// YouTube page-load, ~320 pinned). Excluding them lets the active set keep its
/// full cap regardless of the pinned burst, so a burst is *absorbed* (evicts≈0)
/// instead of thrashing. Total registry is bounded by construction —
/// `REGISTRY_CAP` non-pinned + `RENDER_DEFERRED_WINDOW_CAP` pinned. VRAM does not
/// stay pinned at this ceiling: the idle drain
/// ([`ResourcePools::advance_registry_touch_and_drain`]) reclaims a burst's stale
/// leftovers ~2 s after last use, returning the resident set to the ~56 idle
/// working set once the burst ends. So this is sized to absorb the burst's *live*
/// working set (measured non-pinned peak ~260 during a YouTube page-load), not to
/// hold it forever. Slots are cheap; the real VRAM guard is per-image bytes.
const REGISTRY_CAP: usize = 320;
/// Wall-clock milliseconds a non-pinned resident may go untouched before the
/// idle drain reclaims it. An actively-drawn target is touched every frame (and
/// the presented target is touched every poll) so it never ages out, while a
/// burst's stale targets (a settled page's thumbnail RTs) are reclaimed ~2 s
/// after last use — so `REGISTRY_CAP` can be high enough to absorb a burst (no
/// eviction thrash) without pinning that VRAM for the guest lifetime.
///
/// **Wall-clock, not publish-count:** the drain clock is fed from the poll
/// heartbeat (`device_poll`, ~244 Hz), which ticks even when the guest stops
/// compositing and issuing present publishes. A publish-count clock froze on a
/// static page (measured `present_import used_hz=0`), so a burst's ~260 stale
/// residents (~516 MiB) never aged out and VRAM never returned to the ~1005 MiB
/// idle baseline. Real time keeps advancing regardless of guest activity.
const IDLE_TARGET_AGE_MS: u64 = 2000;
/// Wall-clock milliseconds a host-import window may go untouched before the idle
/// sweep releases it — deliberately far longer than [`IDLE_TARGET_AGE_MS`],
/// which governs cheap-to-recreate VRAM residents.
///
/// A host-import window is not VRAM: it is a `VK_EXT_external_memory_host`
/// registration pinning up to [`HOST_IMPORT_WINDOW_CAP`] (1 GiB) of guest RAM,
/// and re-pinning one costs 100–290 ms of `dma_us` on a live x86 boot (the
/// registration + first fault-in of a 1 GiB span), versus microseconds to
/// recreate a VRAM resident. The two must not share a cutoff.
///
/// The generic 2 s cutoff thrashed the whole working set. During steady route-B
/// (dmabuf direct) presentation the guest presents for many seconds without a
/// single import-present resolve — that is the *point* of route B — yet those
/// eight consecutive windows (see [`HOST_IMPORT_WINDOW_BUDGET`]) are still the
/// hot set the next app-switch or Launchpad open will resolve through. At 2 s
/// the sweep evicted all eight mid-session (measured `creates=109
/// evictions=106` across one Safari→apple.com→Launchpad session, the same eight
/// bases cycling evict→re-create), and the next import-present re-pinned the
/// whole set at once — ~2.3 s of stalls that froze the display on a stale frame
/// (the app-switch "corrupted background" class). The correctness bound is the
/// byte budget ([`ResourcePools::plan_host_import_eviction`], unchanged); this
/// long cutoff only returns pinned host RAM once the VM is genuinely quiescent,
/// not on the constant sub-second lulls of interactive use. 30 s is an order of
/// magnitude above the full-set re-pin cost and comfortably above the gaps
/// between resolves during active use, so a returning user pays at most one
/// re-pin rather than a per-lull thrash.
const HOST_IMPORT_IDLE_AGE_MS: u64 = 30_000;
/// Minimum wall-clock spacing between reclaim passes. The poll path calls the
/// drain ~244×/s; without this it would empty the whole registry in well under a
/// second. At `IDLE_TARGET_DRAIN_MAX_PER_CALL` per pass this bounds reclaim to
/// ~40 residents/s — a ~260-target burst drains to baseline over ~6.5 s, gently
/// (no dispose storm that would itself be a P3 hitch).
const IDLE_DRAIN_INTERVAL_MS: u64 = 100;
/// Max non-pinned residents the idle drain reclaims per pass — bounds each drain
/// pass so a large stale set (a ~600-target burst) drains gradually instead of
/// stalling one call with hundreds of image destroys.
const IDLE_TARGET_DRAIN_MAX_PER_CALL: usize = 4;
/// Cap for the separate compute-storage resident registry. Kept at its own value
/// (independent of the target `REGISTRY_CAP` retune) — compute storage residents
/// have their own pin lifecycle and working-set profile, and were never part of
/// the deferred-present pin-burst class that motivated the target-cap change.
const COMPUTE_STORAGE_REGISTRY_CAP: usize = 64;
const SAMPLED_CACHE_CAP: usize = 64;
const SAMPLED_CACHE_BYTE_CAP: usize = 128 * 1024 * 1024;
/// Max recycled sampled slots retained per geometry key in `sampled_free`. A
/// content-changing input only needs a few live at once (the CB ring is 3-deep
/// plus the one being acquired); beyond that a recycled slot is destroyed so a
/// one-off geometry cannot pin memory for the whole guest lifetime. Bounds total
/// retained memory to ~(distinct geometries × cap × slot size).
const SAMPLED_FREE_CAP_PER_KEY: usize = 4;
/// **Global** cap on `sampled_free` across all keys. The per-key cap alone does
/// not bound a *diverse* burst: a YouTube page-load evicts hundreds of distinct
/// sampled geometries (thumbnails), each ≤ the per-key cap, so the pool grew to
/// ~593 images (measured `vram sfree=593`), each pinning a slab sub-allocation so
/// no block could ever empty (`block_frees=0`) — the VRAM-return stall. This
/// global cap keeps the recycle pool from exceeding the working set; evictions
/// past it are destroyed (freeing their slab range) instead of cached.
const SAMPLED_FREE_CAP_TOTAL: usize = 64;
/// Max recycled resident-target images retained per (geometry, format) key in
/// `target_free`. A per-frame content-changing target only needs a few live at
/// once (the CB ring is 3-deep plus the frame being acquired); beyond that a
/// recycled image is destroyed so a one-off geometry cannot pin VRAM for the
/// guest lifetime. Bounds retained memory to ~(distinct geometries × cap ×
/// image size).
const TARGET_FREE_CAP_PER_KEY: usize = 4;
/// **Global** cap on `target_free` across all keys — same diverse-burst reasoning
/// as `SAMPLED_FREE_CAP_TOTAL`.
const TARGET_FREE_CAP_TOTAL: usize = 32;
/// Max recycled transient compute-storage images retained per geometry key in
/// `storage_image_free`. Same reuse logic as the render recycle pools: a
/// same-geometry compute dispatch reuses a pooled image instead of a fresh
/// `vkAllocateMemory`, but a diverse workload cannot hoard more than this per
/// geometry.
const STORAGE_IMAGE_FREE_CAP_PER_KEY: usize = 4;
/// **Global** cap on `storage_image_free` across all keys. Lower than the render
/// pools (`SAMPLED_FREE_CAP_TOTAL=64` / `TARGET_FREE_CAP_TOTAL=32`) because
/// compute-storage residency churns far less than per-frame render targets — and,
/// crucially, unlike the slab-backed render pools each storage slot is a
/// standalone `vkAllocateMemory` (not a slab sub-range), so an *uncapped* pool
/// leaks whole device allocations, not just slab fragmentation. Before this cap
/// the per-dispatch retire path (`drain_cleanup`) pushed unconditionally, so an
/// all-new-geometry compute workload (a diff-heavy / CoreImage / blur burst) grew
/// the pool without bound. Past the cap the displaced slot is destroyed, freeing
/// its VkDeviceMemory.
const STORAGE_IMAGE_FREE_CAP_TOTAL: usize = 16;
/// Images destroyed from the recycle pools per idle-drain pass. The recycle pools
/// exist for *active* per-frame reuse; at idle (the drain only fires after
/// `IDLE_TARGET_AGE_MS` of no touch) they are pure retained VRAM, so each pass
/// also trims them toward empty. Bounded like the registry drain so a large pool
/// drains gradually (no dispose storm) and refills a few-per-frame when activity
/// resumes (no re-alloc hitch).
const IDLE_RECYCLE_TRIM_PER_PASS: usize = 8;

/// Consecutive zero-victim idle-drain passes required before the HOST_VISIBLE
/// buffer pools (`staging_free`/`readback_free`) are trimmed. Unlike the image
/// pools (cheap slab suballocation refill), a trimmed staging buffer costs a
/// full `vkAllocateMemory` when the next upload refills it — on the upload hot
/// path that spikes inter-VBL latency. Gating on N consecutive settled passes
/// (drain interval `IDLE_DRAIN_INTERVAL_MS`) ensures a single quiet pass during
/// active video — where old frame RTs mostly but not always age out each pass —
/// cannot trigger a mid-playback buffer re-alloc. At true idle the counter
/// climbs and the buffers drain to zero within a few hundred ms of settling.
const SETTLED_PASSES_FOR_BUFFER_TRIM: u32 = 3;

/// Empty slab blocks retained at idle. `slab::SLAB_KEEP_EMPTY` (2) is the churn
/// buffer the hot release path keeps mid-burst; at *settled* idle the drain
/// trims all the way to zero so no empty `SLAB_SIZE` block sits resident for a
/// long idle desktop. The hot-path buffer still absorbs active churn (blocks
/// full of live content are never empty, so this never frees a working block);
/// only a block that has genuinely gone empty and stayed empty across the drain
/// interval is returned. Re-allocating on the next burst is measured hitch-free
/// (block allocation during a quad-4K load never moved the per-frame hitch
/// proxy), and at true idle no burst reuses a spare — so a retained spare is
/// pure waste. Minimising idle VRAM is the explicit goal.
const IDLE_SLAB_KEEP_EMPTY: usize = 0;

/// Pop one entry from the LARGEST non-empty bucket of a size-keyed recycle pool.
///
/// The buffer pools are keyed by power-of-two byte size, and the idle trim that
/// drains them exists to return host memory. Taking an arbitrary bucket returns
/// an arbitrary number of bytes per destroy, and `HashMap` order is effectively
/// random, so a pass budgeted at N destroys can spend all of them on 64-byte
/// slots and return nothing. That is not hypothetical: the staging census put
/// **11 792 of 26 624** misses in the 64-byte bucket and 1 462 more at 128 bytes
/// — over half the pool's re-allocations, each costing a ~1.4 ms
/// `vkAllocateMemory` on the upload hot path, to reclaim 64 bytes.
///
/// Largest-first makes each destroy return the most it can, so the trim reaches
/// its memory target in the fewest destroys and leaves the small, cheap-to-hold,
/// constantly-reused slots alone.
fn pop_largest_pool_entry<V>(pool: &mut HashMap<u64, Vec<V>>) -> Option<V> {
    let key = *pool
        .iter()
        .filter(|(_, v)| !v.is_empty())
        .max_by_key(|(k, _)| **k)
        .map(|(k, _)| k)?;
    let bucket = pool.get_mut(&key)?;
    let item = bucket.pop();
    if bucket.is_empty() {
        pool.remove(&key);
    }
    item
}

/// Pop one entry from any non-empty bucket of a keyed recycle pool, removing the
/// bucket when it empties so the pool does not accumulate empty `Vec`s. `None`
/// when the whole pool is empty.
fn pop_any_pool_entry<K, V>(pool: &mut HashMap<K, Vec<V>>) -> Option<V>
where
    K: Clone + std::hash::Hash + Eq,
{
    let key = pool
        .iter()
        .find(|(_, v)| !v.is_empty())
        .map(|(k, _)| k.clone())?;
    let bucket = pool.get_mut(&key)?;
    let item = bucket.pop();
    if bucket.is_empty() {
        pool.remove(&key);
    }
    item
}
/// Bucket bins for the staging census: one per power of two up to 2^31.
pub(crate) const STAGING_BUCKET_BINS: usize = 32;
/// One `staging_pool` line per this many misses.
const STAGING_MISS_EMIT_EVERY: u64 = 512;

/// Graveyard size at which begin_entry force-quiesces the ring to destroy
/// deferred handles (pure-async streak backstop).
const GRAVEYARD_FORCE_DRAIN: usize = 256;
/// Max draws per deferred-submit batch before `batch_slot` refuses joiners
/// and the run flushes + reopens. Bounds GPU-idle latency and staging-slot
/// hoarding (see `batch_slot`) while amortizing per-draw submit overhead.
const BATCH_MAX_DRAWS: u64 = 8;

/// 128-bit content fingerprint for the sampled cache.
///
/// The sampled cache matches an incoming blob to a retained VkImage by this
/// fingerprint alone — it no longer keeps a byte copy to `memcmp` against, so
/// the width must make an accidental collision (different content, identical
/// digest, identical geometry/format key) astronomically unlikely: at 128 bits
/// the birthday bound across the 64-entry cache is ~2^-116, far below the host
/// GPU's own soft-error rate. Two independently salted `DefaultHasher`
/// (SipHash-1-3) passes over the *warm* source bytes are still strictly cheaper
/// than the old one-hash-plus-cold-full-frame-`memcmp` (which pulled the
/// retained 8 MiB copy back through DRAM on every hit).
fn sampled_content_hash(bytes: &[u8]) -> u128 {
    let mut lo = std::collections::hash_map::DefaultHasher::new();
    bytes.hash(&mut lo);
    let mut hi = std::collections::hash_map::DefaultHasher::new();
    // Distinct salt so the two digests are independent (else both hashers see
    // the same input and finish() to correlated values, collapsing to 64 bits).
    hi.write_u64(0x9e37_79b9_7f4a_7c15);
    bytes.hash(&mut hi);
    ((hi.finish() as u128) << 64) | lo.finish() as u128
}

/// Which pool asked for a `vkAllocateMemory`.
///
/// `engine_memory_alloc_us` is 40-91 % of every drain tranche over the 25 ms
/// outlier threshold, at ~940 µs an allocation. One fused counter cannot say
/// which of the seven allocating pools spends it, and they have different fixes:
/// a staging bucket that misses its free list, a per-frame sampled image, and a
/// transient depth attachment are three different defects wearing one number.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum AllocSite {
    StorageImage,
    ResidentColor,
    TransientDepth,
    Staging,
    Readback,
    ReadbackMulti,
    SlabBlock,
}

const ALLOC_SITE_N: usize = 7;

impl AllocSite {
    const fn idx(self) -> usize {
        match self {
            AllocSite::StorageImage => 0,
            AllocSite::ResidentColor => 1,
            AllocSite::TransientDepth => 2,
            AllocSite::Staging => 3,
            AllocSite::Readback => 4,
            AllocSite::ReadbackMulti => 5,
            AllocSite::SlabBlock => 6,
        }
    }
}

const ALLOC_SITE_NAMES: [&str; ALLOC_SITE_N] = [
    "storage_image",
    "resident_color",
    "transient_depth",
    "staging",
    "readback",
    "readback_multi",
    "slab_block",
];

static ALLOC_SITE_COUNT: [std::sync::atomic::AtomicU64; ALLOC_SITE_N] = [const { std::sync::atomic::AtomicU64::new(0) }; ALLOC_SITE_N];
static ALLOC_SITE_US: [std::sync::atomic::AtomicU64; ALLOC_SITE_N] = [const { std::sync::atomic::AtomicU64::new(0) }; ALLOC_SITE_N];
static ALLOC_SITE_BYTES: [std::sync::atomic::AtomicU64; ALLOC_SITE_N] = [const { std::sync::atomic::AtomicU64::new(0) }; ALLOC_SITE_N];
/// Accumulated allocation wall-clock since the last emit; one line per second of
/// it, so the rate is self-clocked and an idle boot stays silent.
static ALLOC_WINDOW_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
const ALLOC_WINDOW_EMIT_US: u64 = 1_000_000;

pub(crate) unsafe fn allocate_memory_timed(
    ctx: &DeviceContext,
    info: &vk::MemoryAllocateInfo<'_>,
    counters: &EngineCounters,
    site: AllocSite,
) -> Result<vk::DeviceMemory, vk::Result> {
    let started = Instant::now();
    let result = ctx.device.allocate_memory(info, None);
    let us = started.elapsed().as_micros() as u64;
    counters.memory_alloc_us.fetch_add(us, Ordering::Relaxed);
    let i = site.idx();
    ALLOC_SITE_COUNT[i].fetch_add(1, Ordering::Relaxed);
    ALLOC_SITE_US[i].fetch_add(us, Ordering::Relaxed);
    ALLOC_SITE_BYTES[i].fetch_add(info.allocation_size, Ordering::Relaxed);
    if ALLOC_WINDOW_US.fetch_add(us, Ordering::Relaxed) + us >= ALLOC_WINDOW_EMIT_US {
        ALLOC_WINDOW_US.store(0, Ordering::Relaxed);
        emit_alloc_site_census();
    }
    result
}

/// Cumulative per-site allocation census: `count:microseconds:mebibytes`.
fn emit_alloc_site_census() {
    use std::fmt::Write as _;
    let mut line = String::from("vk_alloc_sites");
    for (i, name) in ALLOC_SITE_NAMES.iter().enumerate() {
        let _ = write!(
            line,
            " {name}={}:{}:{}",
            ALLOC_SITE_COUNT[i].load(Ordering::Relaxed),
            ALLOC_SITE_US[i].load(Ordering::Relaxed),
            ALLOC_SITE_BYTES[i].load(Ordering::Relaxed) >> 20,
        );
    }
    crate::observe::off(line);
}

include!("submission_and_buffers.rs");
include!("images_and_registry.rs");
include!("host_import_and_teardown.rs");

#[cfg(test)]
mod host_import_budget_tests {
    use super::{
        host_import_budget, host_scatter, present_stats_setup_decline, resolve_scatter_regions,
        terminal_host_import_error, DrawError, HostImportDecline, HostImportRegion,
        PresentStatsSetup, ResourcePools, VkCall, VkOp, HOST_IMPORT_IDLE_AGE_MS,
        HOST_IMPORT_REGION_CAP, HOST_IMPORT_TOTAL_BYTE_CAP, HOST_IMPORT_WINDOW_BUDGET,
        HOST_IMPORT_WINDOW_CAP, IDLE_RECYCLE_TRIM_PER_PASS, IDLE_TARGET_AGE_MS,
    };
    use crate::observe::Decline;
    use ash::vk;

    /// A pool holding `regions` maximal windows, each stamped with the given
    /// `(last_touch, last_epoch, last_touch_ms)`. Handles stay null: every
    /// function under test here is device-free by construction and never
    /// dereferences them.
    fn pool_with_windows(stamps: &[(u64, u64, u64)], epoch: u64) -> ResourcePools {
        let mut pools = ResourcePools::new();
        pools.host_import_epoch = epoch;
        pools.host_import_touch = stamps.iter().map(|s| s.0).max().unwrap_or(0);
        for (i, &(last_touch, last_epoch, last_touch_ms)) in stamps.iter().enumerate() {
            pools.host_imports.push(HostImportRegion {
                base: (i + 1) * HOST_IMPORT_WINDOW_CAP as usize,
                len: HOST_IMPORT_WINDOW_CAP,
                memory: vk::DeviceMemory::null(),
                buffer: vk::Buffer::null(),
                last_touch,
                last_epoch,
                last_touch_ms,
            });
        }
        pools
    }

    /// The torn-scatter guard. `present_scatter_gpu` resolves EVERY run of a
    /// surface into a local buffer list before recording any of them, and
    /// flushes the open batch first — so mid-loop the pool looks quiescent
    /// (`in_flight == 0`, no open batch) while the caller still holds live
    /// handles. `dispose`'s in-flight check cannot see those. Releasing a window
    /// resolved in the live epoch would hand the GPU a destroyed buffer and tear
    /// the guest surface, so the idle sweep must refuse one however old its
    /// wall-clock stamp looks — `last_touch_ms` rides the poll-driven idle
    /// clock, which does not tick during a resolve burst, so an entire live run
    /// list can share one stale millisecond stamp.
    #[test]
    fn a_region_touched_this_epoch_is_never_evicted() {
        let epoch = 7;
        // Every window resolved in the live epoch: the all-or-nothing run list.
        let stamps: Vec<(u64, u64, u64)> = (0..HOST_IMPORT_WINDOW_BUDGET)
            .map(|i| (i + 1, epoch, 0))
            .collect();
        let pools = pool_with_windows(&stamps, epoch);
        assert!(
            pools.plan_host_import_idle_release(u64::MAX, 16).is_empty(),
            "the age cutoff alone must not be able to release a live window"
        );
    }

    /// The idle sweep releases windows the guest has stopped resolving through,
    /// bounded per pass, and leaves recent ones alone. Without it the budget
    /// only ratchets up to its high-water mark and holds pinned host RAM until
    /// teardown.
    #[test]
    fn idle_sweep_releases_only_cold_windows_and_bounds_the_pass() {
        let pools = pool_with_windows(&[(1, 1, 100), (2, 1, 200), (3, 1, 9_000)], 5);
        assert_eq!(
            pools.plan_host_import_idle_release(500, 16),
            vec![0, 1],
            "only windows untouched since the cutoff"
        );
        assert_eq!(
            pools.plan_host_import_idle_release(500, 1).len(),
            1,
            "a fired pass must not empty the set in one go"
        );
        assert!(
            pools.plan_host_import_idle_release(0, 16).is_empty(),
            "nothing is cold before the first cutoff"
        );
    }

    /// The idle sweep holds host-import windows to [`HOST_IMPORT_IDLE_AGE_MS`],
    /// NOT the cheap-VRAM [`IDLE_TARGET_AGE_MS`]. This is the app-switch-freeze
    /// regression: the eight-window working set that steady route-B presentation
    /// leaves untouched for seconds must survive the generic 2 s cutoff, or it is
    /// evicted mid-session and re-pinned all at once on the next import-present
    /// (~2.3 s of stalls). It is released only after a genuinely long quiescence.
    #[test]
    fn host_import_idle_sweep_uses_the_long_cutoff_not_the_vram_one() {
        // Whole working set last resolved at a realistic post-boot clock stamp,
        // cold epoch (route B has since advanced the submit epoch without
        // resolving through these windows). `TOUCH` is nonzero because the live
        // idle clock is minutes-large by the time the desktop is up.
        const TOUCH: u64 = 100_000;
        let stamps: Vec<(u64, u64, u64)> = (0..HOST_IMPORT_WINDOW_BUDGET)
            .map(|i| (i + 1, 1, TOUCH))
            .collect();
        let pools = pool_with_windows(&stamps, 9);
        // A lull just past the generic VRAM cutoff but well within the
        // host-import cutoff: the old code (sampled_cutoff = now - 2000) released
        // everything here; the fix must release nothing.
        let now = TOUCH + IDLE_TARGET_AGE_MS + 500;
        assert_eq!(
            pools
                .plan_host_import_idle_release(now.saturating_sub(IDLE_TARGET_AGE_MS), 16)
                .len(),
            HOST_IMPORT_WINDOW_BUDGET as usize,
            "sanity: the generic 2 s cutoff WOULD evict the whole set — the thrash"
        );
        assert!(
            pools.plan_host_import_idle_sweep(now, 16).is_empty(),
            "the hot working set must survive a sub-30 s lull under the long cutoff"
        );
        // Once genuinely quiescent past the long cutoff, the sweep does release
        // (bounded per pass), so pinned host RAM still returns on real idle.
        let quiescent = TOUCH + HOST_IMPORT_IDLE_AGE_MS + 1;
        assert_eq!(
            pools
                .plan_host_import_idle_sweep(quiescent, IDLE_RECYCLE_TRIM_PER_PASS)
                .len(),
            IDLE_RECYCLE_TRIM_PER_PASS.min(HOST_IMPORT_WINDOW_BUDGET as usize),
            "a genuinely idle VM still returns its pinned windows, bounded per pass"
        );
    }

    /// The budget must admit more than one maximal window. `capped_import_window`
    /// rounds a span up to its whole 1 GiB VMA bucket, so with a one-window
    /// budget the first import spends all of it and every span in another bucket
    /// declines forever (regions are never evicted). That is what left the x86
    /// desktop's deferred render flushes dying on `host_import_resolve` and the
    /// screen black.
    #[test]
    fn budget_admits_a_second_window_after_a_maximal_one() {
        assert_eq!(
            host_import_budget(1, HOST_IMPORT_WINDOW_CAP, HOST_IMPORT_WINDOW_CAP),
            Ok(()),
            "a second maximal window must still fit after the first"
        );
        // Bounded pinning is still the point of the window cap: the budget is a
        // ceiling, not an invitation to grow into whole-RAMBlock territory.
        assert_eq!(
            host_import_budget(
                HOST_IMPORT_WINDOW_BUDGET as usize,
                HOST_IMPORT_TOTAL_BYTE_CAP,
                1
            ),
            Err(HostImportDecline::TotalBytes),
            "the budget must still be a ceiling once every window is resident"
        );
        // A full budget declines rather than shuffling: the resolve path admits
        // only when admission is free, so this typed, latched decline is what an
        // over-subscribed working set produces, and the caller serves the span
        // from the CPU byte path.
        let epoch = 4;
        let stamps: Vec<(u64, u64, u64)> = (0..HOST_IMPORT_WINDOW_BUDGET)
            .map(|i| (i + 1, epoch, 0))
            .collect();
        let pools = pool_with_windows(&stamps, epoch);
        assert_eq!(
            host_import_budget(
                pools.host_imports.len(),
                pools.host_import_bytes(),
                HOST_IMPORT_WINDOW_CAP
            ),
            Err(HostImportDecline::TotalBytes),
            "an exhausted, fully pinned budget stays fail-visible"
        );
    }

    #[test]
    fn scatter_region_resolution_refuses_every_invalid_span_before_recording() {
        let runs = [host_scatter::ScatterRun {
            host_ptr: 0x1000,
            ptr_len: 16,
        }];
        let buffers = [(vk::Buffer::null(), 8)];
        let base = host_scatter::ScatterSpan {
            run_index: 0,
            dst_offset: 0,
            x: 0,
            y: 0,
            texels: 1,
        };
        let regions = resolve_scatter_regions(&runs, &[base], &buffers, 4, 1).unwrap();
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].buffer_offset, 8);

        let slug = |span, runs: &[host_scatter::ScatterRun], buffers: &[(vk::Buffer, u64)]| {
            resolve_scatter_regions(runs, &[span], buffers, 4, 1)
                .unwrap_err()
                .slug()
        };
        assert_eq!(
            slug(
                host_scatter::ScatterSpan {
                    run_index: 1,
                    ..base
                },
                &runs,
                &buffers
            ),
            "host_runs_scatter_run_index_oob"
        );
        assert_eq!(slug(base, &runs, &[]), "host_runs_scatter_buffer_index_oob");
        assert_eq!(
            slug(
                host_scatter::ScatterSpan { texels: 0, ..base },
                &runs,
                &buffers
            ),
            "host_runs_scatter_zero_texels"
        );
        assert_eq!(
            slug(host_scatter::ScatterSpan { x: 4, ..base }, &runs, &buffers),
            "host_runs_scatter_source_oob"
        );
        assert_eq!(
            slug(
                host_scatter::ScatterSpan {
                    dst_offset: u64::MAX,
                    ..base
                },
                &runs,
                &buffers
            ),
            "host_runs_scatter_span_end_overflow"
        );
        assert_eq!(
            slug(
                host_scatter::ScatterSpan {
                    dst_offset: 13,
                    ..base
                },
                &runs,
                &buffers
            ),
            "host_runs_scatter_oob"
        );
        let large_run = [host_scatter::ScatterRun {
            host_ptr: 0x1000,
            ptr_len: u64::MAX,
        }];
        assert_eq!(
            slug(
                host_scatter::ScatterSpan {
                    dst_offset: 4,
                    ..base
                },
                &large_run,
                &[(vk::Buffer::null(), u64::MAX)]
            ),
            "host_runs_scatter_buffer_offset_overflow"
        );
    }

    /// Each cause owns its latch, so a second cause is never reported as a repeat
    /// of the first. A single shared flag is what would have made the extension
    /// and zero-length cases indistinguishable at the sink.
    #[test]
    fn each_host_import_cause_latches_separately() {
        let all = [
            HostImportDecline::RegionCount,
            HostImportDecline::TotalBytes,
            HostImportDecline::ZeroLength,
            HostImportDecline::ExtensionAbsent,
        ];
        let slugs: std::collections::BTreeSet<&str> = all.iter().map(|d| d.slug()).collect();
        assert_eq!(slugs.len(), all.len(), "each cause needs its own slug");
    }

    #[test]
    fn count_and_total_caps_report_distinct_reasons() {
        assert_eq!(
            host_import_budget(HOST_IMPORT_REGION_CAP, 1, 1),
            Err(HostImportDecline::RegionCount)
        );
        assert_eq!(
            host_import_budget(1, HOST_IMPORT_TOTAL_BYTE_CAP, 1),
            Err(HostImportDecline::TotalBytes)
        );
    }

    #[test]
    fn terminal_failure_preserves_the_last_attempted_import_cause() {
        let attempted = DrawError::VkCall(VkCall::new(
            VkOp::PoolsHostImportBindBuffer,
            ash::vk::Result::ERROR_OUT_OF_HOST_MEMORY,
        ));
        let error = terminal_host_import_error(Some(attempted.clone()), 0x1000, 4096, 4096);
        assert_eq!(error, attempted);
        assert_eq!(error.slug(), "vk_pools_host_import_bind_buffer");

        let no_attempt = terminal_host_import_error(None, 0x1001, 4096, 4096);
        assert_eq!(no_attempt.slug(), "host_import_no_valid_window");
        assert_eq!(
            no_attempt.fields(),
            vec![
                ("host_ptr", "0x1001".into()),
                ("len", "4096".into()),
                ("alignment", "4096".into()),
            ]
        );
    }

    #[test]
    fn present_stats_setup_failures_preserve_leaf_and_setup_stage() {
        let error = DrawError::VkCall(VkCall::new(
            VkOp::CachesCreateComputePipelines,
            ash::vk::Result::ERROR_OUT_OF_DEVICE_MEMORY,
        ));
        for setup in [
            PresentStatsSetup::Shader,
            PresentStatsSetup::Layout,
            PresentStatsSetup::Pipeline,
        ] {
            let line = present_stats_setup_decline(setup, &error).render();
            assert!(line
                .starts_with("stats_reduce reason=vk_caches_create_compute_pipelines vk_result="));
            assert!(line.ends_with(&format!(" setup={}", setup.name())));
        }
    }
}

#[cfg(test)]
mod content_hash_tests {
    use super::sampled_content_hash;

    /// Identical bytes must fingerprint identically — this is what lets a repeat
    /// bind hit the retained image without the (now removed) full-frame memcmp.
    #[test]
    fn identical_content_hashes_equal() {
        let a = vec![0x11u8; 4096];
        let b = vec![0x11u8; 4096];
        assert_eq!(sampled_content_hash(&a), sampled_content_hash(&b));
    }

    /// A single differing byte must change the digest — a stale bind is the
    /// regression this guards (dropping the memcmp made the digest the sole
    /// arbiter of "same content").
    #[test]
    fn single_byte_change_flips_digest() {
        let mut a = vec![0x11u8; 4096];
        let base = sampled_content_hash(&a);
        a[2048] = 0x12;
        assert_ne!(base, sampled_content_hash(&a));
    }

    /// The two 64-bit halves must be independent: if the high half were just a
    /// copy of the low half the fingerprint would collapse to 64 bits and the
    /// birthday bound the memcmp removal relies on would not hold. Distinct
    /// content that happened to collide on 64 bits must still differ on 128.
    #[test]
    fn halves_are_independent() {
        // Different lengths and contents: high and low halves must not mirror.
        for bytes in [
            vec![0u8; 1],
            vec![0xffu8; 64],
            (0..=255u8).collect::<Vec<_>>(),
        ] {
            let h = sampled_content_hash(&bytes);
            let lo = h as u64;
            let hi = (h >> 64) as u64;
            assert_ne!(lo, hi, "halves collapsed for len={}", bytes.len());
        }
    }

    /// Length alone must not decide the digest (content matters within a fixed
    /// geometry/format key, where all blobs share one length).
    #[test]
    fn same_length_different_content_differs() {
        let a = vec![0xa0u8; 1024];
        let mut b = vec![0xa0u8; 1024];
        b[0] = 0xa1;
        assert_ne!(sampled_content_hash(&a), sampled_content_hash(&b));
    }
}

#[cfg(test)]
mod pool_trim_order_tests {
    use super::{pop_any_pool_entry, pop_largest_pool_entry};
    use std::collections::HashMap;

    /// The buffer trim must return the most bytes it can per destroy.
    ///
    /// The pools are keyed by power-of-two byte size and the trim's budget is a
    /// COUNT of destroys, so which bucket it takes from decides how much memory a
    /// pass reclaims. `HashMap` iteration order is effectively random, so the
    /// arbitrary-bucket form could spend a whole pass on 64-byte slots — measured
    /// as 11 792 of 26 624 staging misses in that one bucket, each costing a
    /// ~1.4 ms `vkAllocateMemory` to recreate, to reclaim 64 bytes.
    ///
    /// Asserting the descending ORDER rather than one pop is the point: a single
    /// pop passes by luck with a random-order pool, which is exactly how the bug
    /// stayed invisible.
    #[test]
    fn buffer_trim_drains_the_largest_buckets_first() {
        let mut pool: HashMap<u64, Vec<u32>> = HashMap::new();
        for (bucket, n) in [(64u64, 40), (4096, 3), (1 << 22, 2), (256, 7)] {
            pool.insert(bucket, vec![bucket as u32; n]);
        }
        let mut order = Vec::new();
        while let Some(v) = pop_largest_pool_entry(&mut pool) {
            order.push(v as u64);
        }
        assert_eq!(order.len(), 52);
        assert!(
            order.windows(2).all(|w| w[0] >= w[1]),
            "trim order is not descending by bucket: {order:?}"
        );
        assert_eq!(order[0], 1 << 22);
        assert_eq!(order[order.len() - 1], 64);
        assert!(pool.is_empty(), "emptied buckets must be removed");

        // The arbitrary-bucket helper is still used by the image pools, whose
        // budget is not about bytes; it must keep draining everything.
        let mut pool: HashMap<u64, Vec<u32>> = HashMap::new();
        pool.insert(64, vec![1, 2]);
        pool.insert(4096, vec![3]);
        let mut n = 0;
        while pop_any_pool_entry(&mut pool).is_some() {
            n += 1;
        }
        assert_eq!(n, 3);
        assert!(pool.is_empty());
    }
}
