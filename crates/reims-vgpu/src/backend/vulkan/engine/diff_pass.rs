//! The render writeback's difference pass: which bytes actually changed.
//!
//! # What it is for
//!
//! The writeback rail copies a whole composited frame out of a render target
//! and scatters it into guest RAM, and 70-99% of the bytes it writes are
//! already at the destination. Two separate costs follow from moving them
//! anyway:
//!
//! - The copy out of the target is 87-91% of the readback fence, because on a
//!   discrete part it crosses the bus into host-visible memory.
//! - The scatter into guest pages is ~744 µs per landing on the CPU, and the
//!   drain worker is saturated.
//!
//! A CPU-side compare was built to collect on the second and is refuted: a
//! full-cache-line store never reads its destination, so a declined store never
//! cost a read, and comparing adds a whole frame's read the eager path never
//! paid. The GPU is the only place the comparison is free enough, because it is
//! *already* reading the frame in order to copy it.
//!
//! # The shape
//!
//! ```text
//! target image --(copy, device-local)--> cur
//! cur, prev --(tile_diff)--> out (host-visible, sparse), bits (host-visible)
//! ```
//!
//! `prev` is a device-local copy of what the guest's pages hold. The pass
//! writes `out` only for the 256-byte tiles that differ, and records those
//! tiles in `bits`. The scatter then lands the set tiles and nothing else.
//!
//! Not writing a byte that already holds the value being written is an
//! **identity**, not a heuristic: memory ends in the same state either way. So
//! the pass needs no damage rect, no witness and no guess — but it does need
//! `prev` to be true, which is a promotion problem rather than a shader one and
//! is documented at the promotion site rather than here.
//!
//! # What it costs where
//!
//! The saving is not uniform across the support matrix and should not be
//! claimed as though it were.
//!
//! On a **discrete** host the eager path reads the target and writes a whole
//! frame across the bus. This path reads the target and writes a whole frame
//! into device-local memory — local bandwidth, roughly two orders of magnitude
//! cheaper per byte — then reads two frames locally and writes a fraction of
//! one across the bus. The bus traffic falls by whatever fraction of tiles are
//! unchanged.
//!
//! On a **unified-memory** host there is no bus to decline, and the extra local
//! reads of `cur` and `prev` are real work the eager path did not do. What is
//! still collected there is the CPU scatter: the bitmap turns a full-frame
//! scatter into a scatter of the changed tiles, and that is the cost the drain
//! worker is saturated by. Expect a smaller win, not the same one.

use ash::vk;

use super::caches::{BindingSig, ComputePipelineKey, LayoutKey, ObjectCaches};
use super::context::DeviceContext;
use super::counters::EngineCounters;
use super::pools::{AllocSite, ResourcePools, allocate_memory_timed};
use super::vk_call::{VkCall, VkOp};
use super::{DrawError, reason};
use crate::backend::vulkan::spirv_emit::{self, TILE_DIFF_WORDS_PER_TILE, tile_diff_binding};

/// A device-local buffer this device allocates for its own passes.
///
/// Neither existing allocator fits: `acquire_staging` is hard-wired to
/// `MemoryClass::Upload` and `create_readback_buffer` to `Readback`, and both
/// are pools keyed by size whose slots are recycled between rails. These are
/// neither — they are long-lived, device-local, and their *contents* are the
/// point, so a slot handed to another rail between two frames would silently
/// destroy the comparison.
pub(crate) struct ScratchBuffer {
    pub buffer: vk::Buffer,
    memory: vk::DeviceMemory,
}

impl ScratchBuffer {
    /// Allocate `size` bytes of device-local memory bound to a new buffer.
    ///
    /// Mirrors the image path in `pools::images_and_registry`: ask the object
    /// for its own `memory_type_bits` rather than assuming, and unwind the
    /// half-built object on every failure edge.
    pub(crate) unsafe fn create(
        ctx: &DeviceContext,
        size: u64,
        usage: vk::BufferUsageFlags,
        counters: &EngineCounters,
    ) -> Result<Self, DrawError> {
        let buffer = ctx
            .device
            .create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size.max(4))
                    .usage(usage)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )
            .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::PoolsCreateDiffScratch, e)))?;
        counters.note_create();
        let req = ctx.device.get_buffer_memory_requirements(buffer);
        let mt = ctx
            .memory_type_for(req.memory_type_bits, crate::backend::vulkan::caps::memory_topology::MemoryClass::DeviceLocal)
            .ok_or_else(|| {
                ctx.device.destroy_buffer(buffer, None);
                DrawError::Unsupported(reason::DrawReason::NoDeviceLocalMemoryForDiffScratch {
                    memory_type_bits: req.memory_type_bits,
                })
            })?;
        let memory = allocate_memory_timed(
            ctx,
            &vk::MemoryAllocateInfo::default()
                .allocation_size(req.size)
                .memory_type_index(mt),
            AllocSite::DiffScratch,
        )
        .map_err(|e| {
            ctx.device.destroy_buffer(buffer, None);
            DrawError::VkCall(VkCall::new(VkOp::PoolsAllocDiffScratch, e))
        })?;
        counters.note_alloc();
        ctx.device
            .bind_buffer_memory(buffer, memory, 0)
            .map_err(|e| {
                ctx.device.free_memory(memory, None);
                ctx.device.destroy_buffer(buffer, None);
                DrawError::VkCall(VkCall::new(VkOp::PoolsBindDiffScratch, e))
            })?;
        Ok(Self { buffer, memory })
    }

    /// The usage a scratch buffer needs to be both half of the comparison and
    /// the destination of the copy out of the render target.
    pub(crate) const USAGE: vk::BufferUsageFlags = vk::BufferUsageFlags::from_raw(
        vk::BufferUsageFlags::STORAGE_BUFFER.as_raw()
            | vk::BufferUsageFlags::TRANSFER_DST.as_raw()
            | vk::BufferUsageFlags::TRANSFER_SRC.as_raw(),
    );

    /// Destroy the buffer and free its memory.
    ///
    /// Takes `self` by value: a scratch buffer is not pooled, so there is no
    /// state to return it to and no second caller who could destroy it twice.
    /// The caller owes the usual in-flight rule — no submitted command buffer
    /// may still reference it.
    pub(crate) unsafe fn destroy(self, device: &ash::Device) {
        device.destroy_buffer(self.buffer, None);
        device.free_memory(self.memory, None);
    }
}

/// The four buffers [`record_tile_diff`] binds, in binding order.
pub(crate) struct TileDiffBuffers {
    /// This frame. Device-local; the copy out of the render target fills it.
    pub cur: vk::Buffer,
    /// The frame the guest's pages hold. Device-local.
    pub prev: vk::Buffer,
    /// Host-visible. Written only for the tiles that differ.
    pub out: vk::Buffer,
    /// Host-visible. One bit per tile. Zeroed by the pass itself.
    pub bits: vk::Buffer,
}

/// How the pass is sized for one frame.
///
/// Derived once from the readback's byte length and the device's own workgroup
/// limit, so the module's baked-in bound, the dispatch and the bitmap length
/// cannot come from three different readings of the same frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TileDiffPlan {
    /// 32-bit words the pass covers. Whole words only; see [`Self::for_bytes`].
    pub words: u32,
    pub grid: [u32; 3],
    pub words_per_row: u32,
    /// Words of `bits`, one bit per tile.
    pub bits_words: u32,
}

impl TileDiffPlan {
    /// Plan the pass for a readback of `rb_size` bytes, or decline.
    ///
    /// `None` when the frame cannot be covered by whole words. `rb_size` is not
    /// always a multiple of four — the storage-image rail derives it from
    /// `bytes_per_texel()` and formats narrower than four bytes exist — and a
    /// `rb_size / 4` that truncated would leave a tail undiffed, which is a
    /// silently wrong frame rather than a declined one.
    ///
    /// `None` also when the frame does not fit in a `u32` of words, since the
    /// shader's bound is a 32-bit constant.
    pub(crate) fn for_bytes(rb_size: u64, max_groups_x: u32) -> Option<Self> {
        if !rb_size.is_multiple_of(4) || rb_size == 0 {
            return None;
        }
        let words = u32::try_from(rb_size / 4).ok()?;
        let (grid, words_per_row) =
            spirv_emit::tile_diff_grid(words, TILE_DIFF_WORDS_PER_TILE, max_groups_x);
        let tiles = words.div_ceil(TILE_DIFF_WORDS_PER_TILE);
        Some(Self {
            words,
            grid,
            words_per_row,
            bits_words: tiles.div_ceil(32),
        })
    }

    /// Bytes of `bits` the pass writes.
    pub(crate) fn bits_bytes(&self) -> u64 {
        u64::from(self.bits_words) * 4
    }
}

/// Record the difference pass into an already-open command buffer.
///
/// The caller owes:
/// - `cur` filled by a transfer earlier in **this** command buffer. The pass
///   emits the barrier that makes that write visible to the dispatch, so the
///   caller must not have emitted one itself.
/// - `out` and `bits` at least `plan.words * 4` and `plan.bits_bytes()` long.
/// - `seal_entry` for the returned descriptor set, so it is freed when the
///   entry retires rather than while the dispatch may still be reading it.
///
/// `bits` is zeroed here rather than by the caller because the pass is the only
/// thing that knows how much of it is live: a recycled readback slot is larger
/// than the bitmap and holds a previous tenant's bytes, and zeroing the whole
/// slot would cost more than the bitmap is.
pub(crate) unsafe fn record_tile_diff(
    ctx: &DeviceContext,
    caches: &mut ObjectCaches,
    pools: &mut ResourcePools,
    counters: &EngineCounters,
    cb: vk::CommandBuffer,
    buffers: &TileDiffBuffers,
    plan: TileDiffPlan,
) -> Result<(vk::DescriptorSet, vk::DescriptorPool), DrawError> {
    let words = spirv_emit::tile_diff(plan.words, TILE_DIFF_WORDS_PER_TILE, plan.words_per_row);
    let (digest, module) = caches.get_or_create_shader(ctx, &words, counters, pools)?;

    let layout_key = LayoutKey {
        bindings: [
            tile_diff_binding::CUR,
            tile_diff_binding::PREV,
            tile_diff_binding::OUT,
            tile_diff_binding::BITS,
        ]
        .into_iter()
        .map(|binding| BindingSig {
            binding,
            ty: vk::DescriptorType::STORAGE_BUFFER.as_raw() as u32,
            stages: vk::ShaderStageFlags::COMPUTE.as_raw(),
        })
        .collect(),
    };
    let (dsl, pipeline_layout) = caches.get_or_create_layout(ctx, &layout_key, counters, pools)?;
    let pipeline = caches.get_or_create_compute_pipeline(
        ctx,
        &ComputePipelineKey {
            spirv: digest,
            entry: "main".into(),
            layout: layout_key,
        },
        module,
        pipeline_layout,
        counters,
        pools,
    )?;
    let (dset, dset_pool) = pools.alloc_descriptor_set(&ctx.device, dsl, counters)?;

    // Every bit starts clear: the shader only ever ORs bits in, so a slot
    // carrying a previous tenant's bytes would report tiles as changed that
    // this frame never looked at.
    ctx.device
        .cmd_fill_buffer(cb, buffers.bits, 0, plan.bits_bytes(), 0);

    // One barrier for two writers: the caller's copy into `cur`, and the fill
    // above. Both are TRANSFER writes and both must be visible to the dispatch,
    // so they take one dependency rather than two.
    let to_shader = [vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)];
    ctx.device.cmd_pipeline_barrier(
        cb,
        vk::PipelineStageFlags::TRANSFER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::DependencyFlags::empty(),
        &to_shader,
        &[],
        &[],
    );

    let infos: Vec<vk::DescriptorBufferInfo> = [
        (tile_diff_binding::CUR, buffers.cur),
        (tile_diff_binding::PREV, buffers.prev),
        (tile_diff_binding::OUT, buffers.out),
        (tile_diff_binding::BITS, buffers.bits),
    ]
    .iter()
    .map(|&(_, buffer)| {
        vk::DescriptorBufferInfo::default()
            .buffer(buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE)
    })
    .collect();
    let writes: Vec<vk::WriteDescriptorSet<'_>> = [
        tile_diff_binding::CUR,
        tile_diff_binding::PREV,
        tile_diff_binding::OUT,
        tile_diff_binding::BITS,
    ]
    .iter()
    .enumerate()
    .map(|(i, &binding)| {
        vk::WriteDescriptorSet::default()
            .dst_set(dset)
            .dst_binding(binding)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&infos[i]))
    })
    .collect();
    ctx.device.update_descriptor_sets(&writes, &[]);

    ctx.device
        .cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pipeline);
    ctx.device.cmd_bind_descriptor_sets(
        cb,
        vk::PipelineBindPoint::COMPUTE,
        pipeline_layout,
        0,
        &[dset],
        &[],
    );
    ctx.device
        .cmd_dispatch(cb, plan.grid[0], plan.grid[1], plan.grid[2]);

    // The host reads both outputs after the fence. A fence orders execution;
    // it does not make a shader's writes visible to a host mapping, which is
    // what this asks for. Non-coherent memory still owes an invalidate, which
    // the readback slot's own reader does.
    let to_host = [vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::HOST_READ)];
    ctx.device.cmd_pipeline_barrier(
        cb,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::HOST,
        vk::DependencyFlags::empty(),
        &to_host,
        &[],
        &[],
    );

    Ok((dset, dset_pool))
}

/// Which tiles a bitmap reports as changed, as `(byte_offset, byte_len)` runs.
///
/// Adjacent set tiles are merged, so a frame whose changes are contiguous costs
/// the scatter one run rather than one per tile. The last run is clipped to
/// `bytes`, because the final tile is partial whenever the frame is not a whole
/// number of tiles.
///
/// Deliberately **not** converted into the scatter's `SkipRanges`: that list is
/// rescanned from its start for every row segment, which is fine for the
/// handful of guest-written ranges it carries and quadratic for a bitmap's
/// thousands of runs.
pub fn changed_runs(bits: &[u32], bytes: u64) -> Vec<(u64, u64)> {
    const TILE_BYTES: u64 = TILE_DIFF_WORDS_PER_TILE as u64 * 4;
    let mut runs: Vec<(u64, u64)> = Vec::new();
    for (word_index, word) in bits.iter().enumerate() {
        if *word == 0 {
            continue;
        }
        for bit in 0..32 {
            if word & (1 << bit) == 0 {
                continue;
            }
            let start = (word_index as u64 * 32 + bit) * TILE_BYTES;
            if start >= bytes {
                continue;
            }
            let end = (start + TILE_BYTES).min(bytes);
            match runs.last_mut() {
                Some(last) if last.0 + last.1 == start => last.1 += end - start,
                _ => runs.push((start, end - start)),
            }
        }
    }
    runs
}

/// Which destination a `prev` frame is a claim about.
///
/// Deliberately **not** [`super::types::TargetIdentity`]. `prev` claims that the
/// guest's pages hold a particular frame, so its identity is the mapping window
/// a landing writes into, not the render target the pixels came out of. Two
/// targets can land at one window, one target's identity carries a generation
/// the guest bumps for reasons that have nothing to do with where its pixels go,
/// and a claim keyed on the wrong one of those is a claim about the wrong bytes.
///
/// `texture_ref` is excluded for the same reason: it names which texture
/// produced the frame, and the destination does not care. Including it would
/// give two textures alternating into one window a `prev` each, so each would
/// be invalidated by the other's landing every frame — correct, because the
/// host-write witness would catch it, and worthless, because the rail would
/// then pay for a comparison it can never use.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct LandingIdentity {
    pub mapping_id: u32,
    pub map_generation: u32,
    pub surface_offset: u64,
    pub surface_bpr: u32,
    pub width: u32,
    pub height: u32,
    pub pixel_format: u16,
}

impl LandingIdentity {
    /// The destination a deferred render window will land at.
    pub fn of(key: &crate::model::ComputeStorageResidencyKey) -> Self {
        Self {
            mapping_id: key.mapping_id,
            map_generation: key.map_generation,
            surface_offset: key.surface_offset,
            surface_bpr: key.surface_bpr,
            width: key.width,
            height: key.height,
            pixel_format: key.pixel_format,
        }
    }
}

/// The `cur`/`prev` pair one destination compares through.
struct RailEntry {
    cur: ScratchBuffer,
    prev: ScratchBuffer,
    bytes: u64,
    /// `prev` holds the frame this destination's guest pages hold.
    ///
    /// False until a landing has been acknowledged against it. The first frame
    /// of a destination compares against a buffer nothing wrote, so it is
    /// landed whole and only the *next* one can be narrowed.
    seeded: bool,
    /// Last use, on the rail's own monotonic clock. Eviction is by this.
    used: u64,
    /// The host-write epoch current when this destination was last
    /// acknowledged, as the runtime's own [`crate::runtime::host_writes`] ring
    /// counts it. The runtime asks that ring whether anything has written these
    /// pages since, which is the half of the witness the engine cannot see.
    stamp: u64,
}

/// Per-destination `cur`/`prev` scratch for the render writeback's difference
/// pass.
///
/// Bounded in **bytes**, because that is what the pass actually spends: an
/// entry is two whole frames of device-local memory, and a count would let a
/// 4K destination cost sixteen times what a thumbnail does under the same
/// bound. Eviction is least-recently-used and evicts one entry at a time —
/// dropping the whole map, which the census does, would re-seed every live
/// destination at once and cost a full landing each.
#[derive(Default)]
pub(crate) struct DiffRail {
    entries: std::collections::HashMap<LandingIdentity, RailEntry>,
    /// Device-local bytes currently held across all entries.
    held: u64,
    clock: u64,
}

/// Device-local bytes the rail may hold, given the device it is running on.
///
/// A sixteenth of the largest device-local heap. **That fraction is chosen, not
/// derived, and is not offered as one**: the quantity that would size this is
/// the number of distinct landing destinations live at once during motion, and
/// nothing has measured it — `dr_targets_sum` is emitted per diffed readback so
/// the next driven run states it rather than leaving it inferred from whether
/// an eviction happened to fire. What the fraction *is* chosen against is the
/// rest of the heap's claimants: the resident target registry, the sampled
/// pools and the readback ring all draw on it, and a pass that competes with
/// the images it exists to read would cost more than it collects.
///
/// Reading the heap rather than baking a constant is what keeps this honest on
/// the low end of the support matrix. A discrete part and an iGPU with a small
/// carve-out get budgets that differ by two orders of magnitude, and a constant
/// sized for the first would evict the second's resident targets.
fn scratch_byte_cap(ctx: &DeviceContext) -> u64 {
    ctx.caps.memory.device_local_bytes / 16
}

impl DiffRail {
    /// The pair this destination compares through, and whether `prev` is usable.
    ///
    /// `prev_offered` is the caller's answer to "do the guest's pages still hold
    /// what we last landed there" — it covers writers this module cannot see,
    /// and a `false` here only ever costs a full landing. `None` is returned
    /// when the rail cannot serve the frame at all, which is always a reason to
    /// take the undiffed readback rather than a failure.
    unsafe fn attach(
        &mut self,
        ctx: &DeviceContext,
        counters: &EngineCounters,
        key: &LandingIdentity,
        bytes: u64,
        prev_offered: bool,
    ) -> Option<(vk::Buffer, vk::Buffer, bool)> {
        use crate::runtime::drain::note_store_route_n;
        self.clock = self.clock.wrapping_add(1);
        let clock = self.clock;

        // A destination whose frame size changed under one identity has a pair
        // describing a different shape, so there is nothing to compare against.
        if self.entries.get(key).is_some_and(|e| e.bytes != bytes) {
            self.drop_entry(&ctx.device, key);
        }
        if !self.entries.contains_key(key) {
            let pair = bytes.saturating_mul(2);
            let cap = scratch_byte_cap(ctx);
            if pair > cap {
                // One destination alone does not fit the budget. Nothing to
                // evict would help, and re-asking every frame would churn.
                note_store_route_n("dr_declined_budget", 1);
                return None;
            }
            let live: Vec<(LandingIdentity, u64, u64)> = self
                .entries
                .iter()
                .map(|(k, e)| (k.clone(), e.used, e.bytes.saturating_mul(2)))
                .collect();
            for victim in eviction_order(&live, self.held, pair, cap) {
                self.drop_entry(&ctx.device, &victim);
                note_store_route_n("dr_evict", 1);
            }
            let cur = ScratchBuffer::create(ctx, bytes, ScratchBuffer::USAGE, counters).ok()?;
            let Ok(prev) = ScratchBuffer::create(ctx, bytes, ScratchBuffer::USAGE, counters) else {
                cur.destroy(&ctx.device);
                note_store_route_n("dr_declined_scratch", 1);
                return None;
            };
            self.held = self.held.saturating_add(pair);
            self.entries.insert(
                key.clone(),
                RailEntry {
                    cur,
                    prev,
                    bytes,
                    seeded: false,
                    used: clock,
                    stamp: 0,
                },
            );
        }
        let entry = self.entries.get_mut(key)?;
        entry.used = clock;
        // A destination the caller cannot vouch for stops being seeded rather
        // than being asked again next frame: the frame about to be landed whole
        // re-establishes the claim, and leaving the flag set would let a later
        // frame compare against a `prev` the guest has since overwritten.
        if !prev_offered {
            entry.seeded = false;
        }
        Some((entry.cur.buffer, entry.prev.buffer, entry.seeded))
    }

    /// This destination's guest pages now hold what `cur` holds.
    ///
    /// Called only after a landing has been acknowledged, because until then
    /// nothing establishes that the bytes reached guest RAM. The swap is what
    /// makes this frame the next one's predecessor.
    fn promote(&mut self, key: &LandingIdentity, stamp: u64) {
        if let Some(entry) = self.entries.get_mut(key) {
            std::mem::swap(&mut entry.cur, &mut entry.prev);
            entry.seeded = true;
            entry.stamp = stamp;
        }
    }

    /// The host-write epoch this destination was last acknowledged at, or
    /// `None` when there is no `prev` to ask about.
    fn stamp(&self, key: &LandingIdentity) -> Option<u64> {
        self.entries
            .get(key)
            .filter(|e| e.seeded)
            .map(|e| e.stamp)
    }

    /// This destination's guest pages no longer hold what we last landed.
    fn unseed(&mut self, key: &LandingIdentity) {
        if let Some(entry) = self.entries.get_mut(key) {
            entry.seeded = false;
        }
    }

    unsafe fn drop_entry(&mut self, device: &ash::Device, key: &LandingIdentity) {
        if let Some(entry) = self.entries.remove(key) {
            self.held = self.held.saturating_sub(entry.bytes.saturating_mul(2));
            entry.cur.destroy(device);
            entry.prev.destroy(device);
        }
    }

    /// Live destinations, for the counter that reports the working set.
    fn len(&self) -> usize {
        self.entries.len()
    }

    /// Release every buffer. The caller owes the in-flight rule.
    pub(crate) unsafe fn destroy_all(&mut self, device: &ash::Device) {
        for (_, entry) in self.entries.drain() {
            entry.cur.destroy(device);
            entry.prev.destroy(device);
        }
        self.held = 0;
    }
}

/// The device-local pair a diffed readback compares through, resolved for one
/// frame, plus the caches the pass needs to build its pipeline.
pub(crate) struct DiffAttach<'a> {
    pub cur: vk::Buffer,
    pub prev: vk::Buffer,
    pub plan: TileDiffPlan,
    pub caches: &'a mut ObjectCaches,
}

/// Set up one diffed readback, or decline it.
///
/// Splitting the resolution from the submission is what lets the caller hold
/// `pools` mutably for the readback while this holds the rail: everything the
/// pass needs is reduced to two buffer handles and a plan before the command
/// buffer is opened.
///
/// Answers `None` for every reason a comparison cannot be made — an odd frame
/// size, a budget that will not hold the pair, a device that refuses the
/// scratch — and each of those is a reason to take the plain readback, not a
/// failure. Also answers the second element `false` when `prev` holds nothing
/// yet, which the caller must read as "this frame lands whole".
pub(crate) unsafe fn attach_for_readback(
    ctx: &DeviceContext,
    counters: &EngineCounters,
    rail: &mut DiffRail,
    key: &LandingIdentity,
    rb_size: u64,
    prev_offered: bool,
) -> Option<(vk::Buffer, vk::Buffer, TileDiffPlan, bool)> {
    use crate::runtime::drain::note_store_route_n;
    let Some(plan) = TileDiffPlan::for_bytes(rb_size, ctx.max_compute_work_group_count_x) else {
        note_store_route_n("dr_declined_geometry", 1);
        return None;
    };
    let (cur, prev, seeded) = rail.attach(ctx, counters, key, rb_size, prev_offered)?;
    note_store_route_n("dr_targets_sum", rail.len() as u64);
    Some((cur, prev, plan, seeded))
}

/// Which entries must go before `need` more bytes fit under `cap`.
///
/// Least-recently-used first, and no further than necessary — a rail asked for
/// one more destination evicts one, not the map. Dropping everything is what
/// the census does, and it costs a full landing for every live destination at
/// once; the census can afford that because it is a probe and the rail cannot.
///
/// Returned as a list rather than evicted in place because the buffers an entry
/// holds are the part that needs a device and the *choice* of entry is the part
/// that can be wrong. `live` is `(key, last use, bytes the pair holds)`.
fn eviction_order(
    live: &[(LandingIdentity, u64, u64)],
    held: u64,
    need: u64,
    cap: u64,
) -> Vec<LandingIdentity> {
    let mut order: Vec<&(LandingIdentity, u64, u64)> = live.iter().collect();
    order.sort_by_key(|(_, used, _)| *used);
    let mut freed = 0u64;
    let mut out = Vec::new();
    for (key, _, bytes) in order {
        if held.saturating_sub(freed).saturating_add(need) <= cap {
            break;
        }
        freed = freed.saturating_add(*bytes);
        out.push(key.clone());
    }
    out
}

/// Record what one comparison found, so the rail's saving is a measured
/// quantity rather than an inferred one.
///
/// `dr_tiles_changed / dr_tiles` is the fraction of the writeback this rail
/// still sends. It is the same quantity the census counts under
/// `REIMS_VGPU_PROBE_TILE_DIFF_CENSUS`, but taken on the product path and over
/// the frames actually landed, so a boot can be read for it without the census
/// probe's extra submission deforming the timings.
pub(crate) fn note_diff_result(bits: &[u32], bytes: u64) {
    use crate::runtime::drain::note_store_route_n;
    let tiles = (bytes.div_ceil(4) as u32).div_ceil(TILE_DIFF_WORDS_PER_TILE);
    let changed: u32 = bits.iter().map(|w| w.count_ones()).sum();
    note_store_route_n("dr_frames", 1);
    note_store_route_n("dr_tiles", u64::from(tiles));
    note_store_route_n("dr_tiles_changed", u64::from(changed.min(tiles)));
}

/// Promote a destination's `cur` to `prev` after its landing was acknowledged.
pub(crate) fn promote(rail: &mut DiffRail, key: &LandingIdentity, stamp: u64) {
    rail.promote(key, stamp);
}

/// The host-write epoch a destination's `prev` was established at.
pub(crate) fn prev_stamp(rail: &DiffRail, key: &LandingIdentity) -> Option<u64> {
    rail.stamp(key)
}

/// Withdraw a destination's `prev`, because its guest pages no longer hold it.
pub(crate) fn unseed(rail: &mut DiffRail, key: &LandingIdentity) {
    rail.unseed(key);
}

/// Is the tile-difference census probe on for this boot?
///
/// Off by default, and it must stay that way: the census copies a second whole
/// frame into device-local memory and runs the pass on its own submission, per
/// readback. That is the price of a *complete* count rather than a sample, and
/// it is not a price the product should pay to measure itself.
pub(crate) fn census_requested() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static STATE: AtomicU8 = AtomicU8::new(u8::MAX);
    match STATE.load(Ordering::Relaxed) {
        0 => false,
        1 => true,
        _ => {
            let on = matches!(
                std::env::var_os("REIMS_VGPU_PROBE_TILE_DIFF_CENSUS").as_deref(),
                Some(v) if !v.is_empty() && v != "0"
            );
            STATE.store(u8::from(on), Ordering::Relaxed);
            if on {
                crate::observe::fail(
                    "PROBE tile_diff_census=on reason=REIMS_VGPU_PROBE_TILE_DIFF_CENSUS — \
                     every resident target readback runs a second submission that copies \
                     the frame again and diffs it. Screen output is unaffected and timing \
                     numbers from this boot are not comparable to a boot without it.",
                );
            }
            on
        }
    }
}

/// The `cur`/`prev` pair the census keeps for one render target.
struct CensusEntry {
    cur: ScratchBuffer,
    prev: ScratchBuffer,
    bytes: u64,
    /// False until a second frame of this target has been seen, so the first is
    /// not counted as "every tile changed" against a `prev` holding nothing.
    seeded: bool,
}

/// Per-target scratch for the tile-difference census.
///
/// Bounded by target count rather than by bytes, because each entry is two
/// whole frames of device-local memory and an unbounded map on a guest that
/// cycles render targets would exhaust VRAM rather than degrade.
#[derive(Default)]
pub(crate) struct DiffCensus {
    entries: std::collections::HashMap<super::types::TargetIdentity, CensusEntry>,
    /// The shader's `out`, shared by every target and never read.
    ///
    /// One buffer rather than one per target: the census wants the bitmap, and
    /// `out` exists only because the shader writes it. Device-local, so the
    /// probe does not pay the bus crossing that the eventual rail exists to
    /// decline — which would be measuring the thing it is trying to avoid.
    sink: Option<ScratchBuffer>,
    sink_bytes: u64,
}

/// Targets the census holds scratch for at once.
///
/// Overflow drops the whole map rather than evicting: this is a probe, and an
/// LRU here would be a mechanism nothing in the product needs.
///
/// The first value was 8, reasoned from "a driven drag composites ~8 surfaces
/// per guest frame". That reasoning was wrong in a way worth recording: the
/// composites-per-frame figure counts *flushes*, several of which are the same
/// surface flushed again, while the population this bounds is distinct
/// identities — a different quantity that nothing had measured. The live run
/// read `tdc_overflow` 5 times in 11 seconds against that bound, and every
/// overflow re-seeds the whole map, which accounted for all 39 of that run's
/// seeds.
///
/// 16 is not a derived number either, and is not offered as one: it is one
/// doubling, bounding the probe at 32 frames of device-local memory.
/// `tdc_targets` is what replaces the guess — it reports the live map size, so
/// the next run states the working set rather than leaving it inferred from
/// whether an overflow happened to fire.
const MAX_CENSUS_TARGETS: usize = 16;

impl DiffCensus {
    /// Release every buffer. The caller owes the in-flight rule.
    pub(crate) unsafe fn destroy_all(&mut self, device: &ash::Device) {
        for (_, entry) in self.entries.drain() {
            entry.cur.destroy(device);
            entry.prev.destroy(device);
        }
        if let Some(sink) = self.sink.take() {
            sink.destroy(device);
        }
        self.sink_bytes = 0;
    }
}

/// Count how many of this frame's tiles differ from the frame before it.
///
/// Runs on its own submission after the readback rather than inside it, so a
/// boot with the probe off is byte-identical on the hot path — and a boot with
/// it on cannot be read for timing, which the announcement says.
///
/// What it answers, over *every* landing rather than a sample and per target
/// rather than pooled, is how much of a composited frame is already what the
/// frame before it was. That per-surface split is the measurement the tile
/// rail's saving was only ever bounded to a factor without.
///
/// Declines quietly and often — an odd frame size, a device that refuses the
/// scratch, a target whose geometry just changed. A census is never worth
/// failing a readback for, so every path returns `()` and names its decline.
#[allow(
    clippy::too_many_arguments,
    reason = "a probe that re-describes the readback it follows"
)]
pub(crate) unsafe fn census_target(
    ctx: &DeviceContext,
    caches: &mut ObjectCaches,
    pools: &mut ResourcePools,
    counters: &EngineCounters,
    census: &mut DiffCensus,
    identity: &super::types::TargetIdentity,
    image: vk::Image,
    width: u32,
    height: u32,
    rb_size: u64,
) {
    use crate::runtime::drain::note_store_route_n;

    let Some(plan) = TileDiffPlan::for_bytes(rb_size, ctx.max_compute_work_group_count_x) else {
        note_store_route_n("tdc_declined_geometry", 1);
        return;
    };

    // A geometry change under one identity means the pair describes a different
    // frame shape, so there is nothing to compare against.
    if census.entries.get(identity).is_some_and(|e| e.bytes != rb_size) {
        if let Some(e) = census.entries.remove(identity) {
            e.cur.destroy(&ctx.device);
            e.prev.destroy(&ctx.device);
        }
    }
    if census.sink_bytes < rb_size {
        if let Some(old) = census.sink.take() {
            old.destroy(&ctx.device);
        }
        let Ok(sink) = ScratchBuffer::create(ctx, rb_size, ScratchBuffer::USAGE, counters) else {
            note_store_route_n("tdc_declined_scratch", 1);
            census.sink_bytes = 0;
            return;
        };
        census.sink = Some(sink);
        census.sink_bytes = rb_size;
    }
    if !census.entries.contains_key(identity) {
        if census.entries.len() >= MAX_CENSUS_TARGETS {
            note_store_route_n("tdc_overflow", 1);
            for (_, e) in census.entries.drain() {
                e.cur.destroy(&ctx.device);
                e.prev.destroy(&ctx.device);
            }
        }
        let Ok(cur) = ScratchBuffer::create(ctx, rb_size, ScratchBuffer::USAGE, counters) else {
            note_store_route_n("tdc_declined_scratch", 1);
            return;
        };
        let Ok(prev) = ScratchBuffer::create(ctx, rb_size, ScratchBuffer::USAGE, counters) else {
            cur.destroy(&ctx.device);
            note_store_route_n("tdc_declined_scratch", 1);
            return;
        };
        census.entries.insert(
            identity.clone(),
            CensusEntry {
                cur,
                prev,
                bytes: rb_size,
                seeded: false,
            },
        );
    }

    let (Some(entry), Some(sink)) = (census.entries.get(identity), census.sink.as_ref()) else {
        return;
    };
    let bits = match census_submit(
        ctx,
        caches,
        pools,
        counters,
        image,
        width,
        height,
        [entry.cur.buffer, entry.prev.buffer, sink.buffer],
        plan,
    ) {
        Ok(bits) => bits,
        Err(_) => {
            note_store_route_n("tdc_declined_submit", 1);
            return;
        }
    };

    let tiles = plan.words.div_ceil(TILE_DIFF_WORDS_PER_TILE);
    let changed: u32 = bits.iter().map(|w| w.count_ones()).sum();
    // A **sum**, because `note_store_route_n` only ever adds: this is the live
    // map size accumulated once per censused readback, so the population is
    // `tdc_targets_sum / (tdc_frames + tdc_seed)` and the name says so. Emitted
    // as a sum rather than a high-water mark because a mark that only grew
    // could not fall when the guest settles, and the question is whether
    // `MAX_CENSUS_TARGETS` binds *during the motion*.
    note_store_route_n("tdc_targets_sum", census.entries.len() as u64);
    if let Some(e) = census.entries.get_mut(identity) {
        if std::mem::replace(&mut e.seeded, true) {
            note_store_route_n("tdc_frames", 1);
            note_store_route_n("tdc_tiles", u64::from(tiles));
            note_store_route_n("tdc_tiles_changed", u64::from(changed.min(tiles)));
        } else {
            // The first frame of a target diffs against a buffer nothing wrote,
            // so its answer is noise. Counted so a run that is all seeds — a
            // guest cycling identities faster than it reuses them — reads as
            // one rather than as a suspiciously high change rate.
            note_store_route_n("tdc_seed", 1);
        }
        // This frame becomes the next one's predecessor.
        std::mem::swap(&mut e.cur, &mut e.prev);
    }
}

/// Copy the target into `cur`, run the pass, and read the bitmap back.
///
/// `cur_prev_out` is `[cur, prev, out]`. The fourth binding is not among them:
/// the bitmap slot is acquired here, because it is the only output that leaves
/// this function and its size comes from `plan`.
#[allow(
    clippy::too_many_arguments,
    reason = "one submission assembled from the readback's own arguments"
)]
unsafe fn census_submit(
    ctx: &DeviceContext,
    caches: &mut ObjectCaches,
    pools: &mut ResourcePools,
    counters: &EngineCounters,
    image: vk::Image,
    width: u32,
    height: u32,
    cur_prev_out: [vk::Buffer; 3],
    plan: TileDiffPlan,
) -> Result<Vec<u32>, DrawError> {
    let [cur, prev, out] = cur_prev_out;
    let (cb, fence) = pools.begin_entry(ctx, counters)?;
    let bits = pools.acquire_readback_extra(ctx, plan.bits_bytes(), counters)?;
    ctx.device
        .reset_command_buffer(cb, vk::CommandBufferResetFlags::empty())
        .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::ReadbackResetCb, e)))?;
    ctx.device
        .begin_command_buffer(
            cb,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )
        .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::ReadbackBeginCb, e)))?;
    // The readback this follows left the image in TRANSFER_SRC_OPTIMAL, so the
    // transition half is a no-op and the dependency half is the point — exactly
    // as it is for the readback's own barrier.
    let barrier = [vk::ImageMemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::TRANSFER_READ)
        .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
        .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
        .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
        .image(image)
        .subresource_range(super::color_subresource_range())];
    ctx.device.cmd_pipeline_barrier(
        cb,
        vk::PipelineStageFlags::ALL_COMMANDS,
        vk::PipelineStageFlags::TRANSFER,
        vk::DependencyFlags::empty(),
        &[],
        &[],
        &barrier,
    );
    let region = [vk::BufferImageCopy::default()
        .image_subresource(super::color_subresource_layers())
        .image_extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        })];
    ctx.device.cmd_copy_image_to_buffer(
        cb,
        image,
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        cur,
        &region,
    );
    let (dset, dset_pool) = record_tile_diff(
        ctx,
        caches,
        pools,
        counters,
        cb,
        &TileDiffBuffers {
            cur,
            prev,
            out,
            bits: bits.buffer,
        },
        plan,
    )?;
    ctx.device
        .end_command_buffer(cb)
        .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::ReadbackEndCb, e)))?;
    let cbs = [cb];
    ctx.device
        .queue_submit(
            ctx.queue(),
            &[vk::SubmitInfo::default().command_buffers(&cbs)],
            fence,
        )
        .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::ReadbackSubmit, e)))?;
    pools.wait_entry_fence(ctx, counters, fence)?;
    let bytes = super::pools::read_back_slot(
        ctx,
        &bits,
        plan.bits_bytes(),
        VkOp::ReadbackMap,
        VkOp::ReadbackInvalidate,
    )?;
    let cleanup = pools.seal_entry(vec![(dset, dset_pool)], Vec::new());
    pools.finish_entry_async(cleanup);
    Ok(bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// What [`probe_tile_diff`] read back off the device.
pub struct TileDiffProbe {
    /// The whole `out` slot, `cur.len()` bytes, seeded with
    /// [`probe_tile_diff`]'s sentinel wherever the pass declined to write.
    pub out: Vec<u8>,
    /// The tile bitmap.
    pub bits: Vec<u32>,
}

/// Run the difference pass once over two host byte strings.
///
/// This exists for the integration tests, which cannot otherwise reach a
/// `DeviceContext` — and it exercises the *real* plumbing rather than a
/// stand-in: device-local scratch through [`ScratchBuffer`], `out` and `bits`
/// in recycled readback slots, and the same [`record_tile_diff`] the writeback
/// rail calls. A harness that uploaded four staging buffers instead would pass
/// while the production bindings were wrong.
///
/// `out` is filled with `sentinel` before the dispatch, so a tile the pass
/// declined is distinguishable from one it wrote with the value already there.
///
/// # Errors
///
/// Declines rather than panics on every path, because it runs on whatever
/// thread the test is on and shares the engine mutex with a live device.
pub fn probe_tile_diff(
    cur: &[u8],
    prev: &[u8],
    sentinel: u32,
) -> Result<TileDiffProbe, DrawError> {
    if cur.len() != prev.len() {
        return Err(DrawError::Unsupported(
            reason::DrawReason::NoCombinedGraphicsComputeQueue,
        ));
    }
    let mut guard = super::lock_engine();
    let super::EngineState {
        ref mut owner,
        ref mut caches,
        ref mut pools,
        ref counters,
        ..
    } = &mut *guard;
    unsafe {
        let ctx = owner.ensure(counters)?;
        if !ctx.compute_capable {
            return Err(DrawError::Unsupported(
                reason::DrawReason::NoCombinedGraphicsComputeQueue,
            ));
        }
        pools.ensure_init(ctx, counters)?;
        let bytes = cur.len() as u64;
        let Some(plan) = TileDiffPlan::for_bytes(bytes, ctx.max_compute_work_group_count_x) else {
            return Err(DrawError::Unsupported(
                reason::DrawReason::NoCombinedGraphicsComputeQueue,
            ));
        };

        let scratch_cur = ScratchBuffer::create(ctx, bytes, ScratchBuffer::USAGE, counters)?;
        let scratch_prev = ScratchBuffer::create(ctx, bytes, ScratchBuffer::USAGE, counters)?;
        let result = probe_tile_diff_inner(
            ctx,
            caches,
            pools,
            counters,
            &scratch_cur,
            &scratch_prev,
            cur,
            prev,
            sentinel,
            plan,
        );
        // The submission below is fenced before this returns, so nothing can
        // still be reading them — including on the error paths, which all
        // return after their own `wait_entry_fence` or before any submit.
        scratch_cur.destroy(&ctx.device);
        scratch_prev.destroy(&ctx.device);
        result
    }
}

#[allow(
    clippy::too_many_arguments,
    reason = "a test entry point that assembles the rail's own pieces by hand"
)]
unsafe fn probe_tile_diff_inner(
    ctx: &DeviceContext,
    caches: &mut ObjectCaches,
    pools: &mut ResourcePools,
    counters: &EngineCounters,
    scratch_cur: &ScratchBuffer,
    scratch_prev: &ScratchBuffer,
    cur: &[u8],
    prev: &[u8],
    sentinel: u32,
    plan: TileDiffPlan,
) -> Result<TileDiffProbe, DrawError> {
    let bytes = cur.len() as u64;
    let (cb, fence) = pools.begin_entry(ctx, counters)?;
    let up_cur = pools.acquire_staging(ctx, bytes, vk::BufferUsageFlags::TRANSFER_SRC, counters)?;
    pools.write_staging(ctx, &up_cur, cur)?;
    let up_prev = pools.acquire_staging(ctx, bytes, vk::BufferUsageFlags::TRANSFER_SRC, counters)?;
    pools.write_staging(ctx, &up_prev, prev)?;
    let out = pools.acquire_readback(ctx, bytes, counters)?;
    let bits = pools.acquire_readback_extra(ctx, plan.bits_bytes(), counters)?;

    ctx.device
        .reset_command_buffer(cb, vk::CommandBufferResetFlags::empty())
        .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::ReadbackResetCb, e)))?;
    ctx.device
        .begin_command_buffer(
            cb,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )
        .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::ReadbackBeginCb, e)))?;
    let whole = [vk::BufferCopy::default().size(bytes)];
    ctx.device
        .cmd_copy_buffer(cb, up_cur.buffer, scratch_cur.buffer, &whole);
    ctx.device
        .cmd_copy_buffer(cb, up_prev.buffer, scratch_prev.buffer, &whole);
    // `vkCmdFillBuffer` takes a whole word and writes it repeatedly, which is
    // exactly the sentinel's shape.
    ctx.device.cmd_fill_buffer(cb, out.buffer, 0, bytes, sentinel);
    let (dset, dset_pool) = record_tile_diff(
        ctx,
        caches,
        pools,
        counters,
        cb,
        &TileDiffBuffers {
            cur: scratch_cur.buffer,
            prev: scratch_prev.buffer,
            out: out.buffer,
            bits: bits.buffer,
        },
        plan,
    )?;
    ctx.device
        .end_command_buffer(cb)
        .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::ReadbackEndCb, e)))?;
    let cbs = [cb];
    ctx.device
        .queue_submit(
            ctx.queue(),
            &[vk::SubmitInfo::default().command_buffers(&cbs)],
            fence,
        )
        .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::ReadbackSubmit, e)))?;
    pools.wait_entry_fence(ctx, counters, fence)?;

    let out_bytes = super::pools::read_back_slot(
        ctx,
        &out,
        bytes,
        VkOp::ReadbackMap,
        VkOp::ReadbackInvalidate,
    )?;
    let bits_bytes = super::pools::read_back_slot(
        ctx,
        &bits,
        plan.bits_bytes(),
        VkOp::ReadbackMap,
        VkOp::ReadbackInvalidate,
    )?;
    // After both reads: sealing hands the slots back to the free pool.
    let cleanup = pools.seal_entry(vec![(dset, dset_pool)], Vec::new());
    pools.finish_entry_async(cleanup);
    Ok(TileDiffProbe {
        out: out_bytes,
        bits: bits_bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn landing(mapping_id: u32) -> LandingIdentity {
        LandingIdentity {
            mapping_id,
            map_generation: 1,
            surface_offset: 0,
            surface_bpr: 7680,
            width: 1920,
            height: 1080,
            pixel_format: 0x50,
        }
    }

    /// The destination is the guest window a landing writes into, so which
    /// texture produced the frame must not enter the identity: two textures
    /// alternating into one window would each get a `prev` the other's landing
    /// invalidates, and the rail would pay for comparisons it can never use.
    #[test]
    fn the_landing_identity_is_the_destination_and_not_the_source() {
        let mut key = crate::model::ComputeStorageResidencyKey {
            mapping_id: 9,
            map_generation: 3,
            surface_offset: 4096,
            surface_bpr: 7680,
            span_end: 1 << 24,
            width: 1920,
            height: 1080,
            pixel_format: 0x50,
            texture_ref: 11,
        };
        let a = LandingIdentity::of(&key);
        key.texture_ref = 12;
        // `span_end` is a hull over the same window, so it is destination
        // geometry the identity already names through offset and pitch.
        key.span_end = 1 << 25;
        assert_eq!(a, LandingIdentity::of(&key));
        // Anything that moves where the pixels land is a different destination.
        key.map_generation = 4;
        assert_ne!(a, LandingIdentity::of(&key));
    }

    /// A rail asked for one more destination evicts one, not the map.
    #[test]
    fn eviction_takes_the_least_recently_used_and_stops_when_it_fits() {
        const PAIR: u64 = 16;
        let live = [
            (landing(1), 30u64, PAIR),
            (landing(2), 10, PAIR),
            (landing(3), 20, PAIR),
        ];
        // Room already: nothing goes.
        assert!(eviction_order(&live, 3 * PAIR, 0, 4 * PAIR).is_empty());
        // One pair short: exactly the oldest goes.
        assert_eq!(
            eviction_order(&live, 3 * PAIR, PAIR, 3 * PAIR),
            vec![landing(2)]
        );
        // Two pairs short: the two oldest, in age order, and no more.
        assert_eq!(
            eviction_order(&live, 3 * PAIR, 2 * PAIR, 3 * PAIR),
            vec![landing(2), landing(3)]
        );
        // A request that cannot fit even an empty rail evicts everything it
        // has rather than looping; the caller declines above this.
        assert_eq!(eviction_order(&live, 3 * PAIR, 99 * PAIR, PAIR).len(), 3);
    }

    /// Entries of different sizes free different amounts, so the loop must stop
    /// on bytes freed and not on a count.
    #[test]
    fn eviction_counts_bytes_rather_than_entries() {
        let live = [(landing(1), 1u64, 64u64), (landing(2), 2, 8)];
        // Evicting the oldest alone frees 64, which is enough for a 32-byte
        // pair under a 40-byte cap.
        assert_eq!(eviction_order(&live, 72, 32, 40), vec![landing(1)]);
        // Reverse the ages and the small one goes first but does not suffice,
        // so the large one follows.
        let live = [(landing(1), 2u64, 64u64), (landing(2), 1, 8)];
        assert_eq!(eviction_order(&live, 72, 32, 40), vec![landing(2), landing(1)]);
    }

    /// A frame that is not a whole number of words is declined rather than
    /// truncated. `rb_size / 4` on a 3-bytes-per-texel format would leave the
    /// tail of every row out of the comparison, and an undiffed tail reads as
    /// a frame whose right edge never updates.
    #[test]
    fn a_frame_that_is_not_whole_words_is_declined() {
        assert!(TileDiffPlan::for_bytes(1920 * 1080 * 3, 65_535).is_some());
        assert!(TileDiffPlan::for_bytes(1919 * 3, 65_535).is_none());
        assert!(TileDiffPlan::for_bytes(0, 65_535).is_none());
        assert!(TileDiffPlan::for_bytes(2, 65_535).is_none());
    }

    /// The bitmap is exactly long enough for the tiles the dispatch covers, and
    /// the dispatch covers every word.
    #[test]
    fn the_plan_covers_the_frame_and_the_bitmap_covers_the_plan() {
        for bytes in [4u64, 256, 1920 * 1080 * 4, 3840 * 2160 * 4, 8_192] {
            let plan = TileDiffPlan::for_bytes(bytes, 65_535).expect("whole words");
            let covered = u64::from(plan.grid[0]) * u64::from(plan.grid[1]) * 64;
            assert!(covered >= u64::from(plan.words), "{bytes} bytes uncovered");
            let tiles = plan.words.div_ceil(64);
            assert!(
                u64::from(plan.bits_words) * 32 >= u64::from(tiles),
                "{bytes} bytes: {} bits for {tiles} tiles",
                plan.bits_words * 32,
            );
            // And not wastefully long: one word more would be a word nothing
            // can set, which reads in a dump as a region that never changes.
            assert_eq!(plan.bits_words, tiles.div_ceil(32));
        }
    }

    /// Set tiles come back as byte runs, adjacent ones merged, and the last one
    /// clipped to the frame.
    #[test]
    fn changed_runs_merges_neighbours_and_clips_the_tail() {
        assert_eq!(changed_runs(&[0], 4096), vec![]);
        // Tiles 0 and 1 are adjacent: one run of 512 bytes, not two of 256.
        assert_eq!(changed_runs(&[0b11], 4096), vec![(0, 512)]);
        // Tile 0 and tile 2, with 1 clear: two runs.
        assert_eq!(changed_runs(&[0b101], 4096), vec![(0, 256), (512, 256)]);
        // Across a word boundary, tiles 31 and 32 are still adjacent.
        assert_eq!(
            changed_runs(&[1 << 31, 1], 32 * 256 + 256),
            vec![(31 * 256, 512)]
        );
        // A frame of 300 bytes has a partial second tile: the run stops at the
        // frame, not at the tile.
        assert_eq!(changed_runs(&[0b11], 300), vec![(0, 300)]);
        // A bit past the end of the frame names no bytes at all.
        assert_eq!(changed_runs(&[0b100], 300), vec![]);
    }
}
