//! The render writeback's difference pass: which bytes actually changed.
//!
//! # What it is for
//!
//! The writeback rail copies a whole composited frame out of a render target
//! and scatters it into guest RAM, and 70-99% of the bytes it writes are
//! already at the destination (see [`crate::runtime::land_redundancy`]). Two
//! separate costs follow from moving them anyway:
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
