//! Growable descriptor-pool arena.
//!
//! A single fixed `VkDescriptorPool` hard-fails once its `max_sets` or per-type
//! descriptor budget is exhausted: `vkAllocateDescriptorSets` returns
//! `ERROR_OUT_OF_POOL_MEMORY` / `ERROR_FRAGMENTED_POOL`, the draw's descriptor
//! set never allocates, and the whole draw is dropped — a classic cap-cliff
//! (the exact "a blown cap drops render to a crawl" failure mode this project
//! fights). The prior pool was sized `max_sets=64`, which is only
//! `RING_DEPTH × BATCH_MAX_DRAWS` single-binding sets and is shared with every
//! concurrent compute dispatch and every multi-texture draw (each sampled image
//! consumes one of the 64 `SAMPLED_IMAGE` descriptors) — under a heavy
//! many-4K-video tab the ceiling is reachable.
//!
//! This arena keeps a list of equally-sized pool BLOCKS and appends a fresh
//! block on exhaustion, so descriptor capacity grows with demand instead of
//! dropping draws. Each allocated set remembers its owning block so frees route
//! back to the allocating pool — Vulkan requires `vkFreeDescriptorSets` to
//! target the pool the set came from. Blocks are never destroyed until engine
//! teardown; a `VkDescriptorPool` is host-side bookkeeping (negligible GPU
//! VRAM), and keeping the high-water-mark of blocks avoids destroy/recreate
//! thrash (which would itself cause the hitches the project fights). Growth is
//! a rare event (zero under normal load); the block count is the steady-state
//! cap-pressure signal.

use ash::vk;
use ash::vk::Handle as _;

use super::types::DrawError;
use super::vk_call::{VkCall, VkOp};

fn free_sets_call(result: Result<(), vk::Result>) -> Result<(), VkCall> {
    result.map_err(|error| VkCall::new(VkOp::DescArenaFreeSets, error))
}

/// Per-block `max_sets`. One block sustains a full ring of batched draws
/// (`RING_DEPTH` × `BATCH_MAX_DRAWS` single-binding sets); heavier per-set
/// binding counts or concurrent compute simply grow another block rather than
/// dropping the draw.
pub(crate) const DESC_BLOCK_MAX_SETS: u32 = 64;
/// Per-type descriptor budget within one block, matched to `DESC_BLOCK_MAX_SETS`
/// so a block of single-binding sets exhausts `max_sets` and per-type budget
/// together. A draw that binds N sampled images consumes N of this type's
/// budget; exhausting it before `max_sets` still triggers a clean grow.
const DESC_BLOCK_PER_TYPE: u32 = 64;

/// A growable set of same-sized descriptor-pool blocks. `blocks[0]` is created
/// eagerly at engine init; later blocks appear only on allocation exhaustion.
pub(crate) struct DescriptorArena {
    blocks: Vec<vk::DescriptorPool>,
}

impl DescriptorArena {
    /// Empty arena with no blocks. `create_first_block` (called from the pool
    /// init that owns a device) brings up `blocks[0]`; before that the arena
    /// reports a null primary pool, mirroring the old `desc_pool == null`
    /// pre-init state.
    pub(crate) fn empty() -> Self {
        Self { blocks: Vec::new() }
    }

    /// Bring up the first block. Called once, from the device-owning pool init.
    pub(crate) unsafe fn create_first_block(
        &mut self,
        device: &ash::Device,
    ) -> Result<(), DrawError> {
        debug_assert!(self.blocks.is_empty(), "arena already initialized");
        let pool = Self::create_block(device)?;
        self.blocks.push(pool);
        Ok(())
    }

    unsafe fn create_block(device: &ash::Device) -> Result<vk::DescriptorPool, DrawError> {
        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(DESC_BLOCK_PER_TYPE),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(DESC_BLOCK_PER_TYPE),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(DESC_BLOCK_PER_TYPE),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::SAMPLER)
                .descriptor_count(DESC_BLOCK_PER_TYPE),
        ];
        device
            .create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
                    .max_sets(DESC_BLOCK_MAX_SETS)
                    .pool_sizes(&pool_sizes),
                None,
            )
            .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::DescArenaCreatePool, e)))
    }

    /// Allocate one descriptor set for `dsl`. Tries each existing block
    /// (most-recent first — most likely to have room); on exhaustion of all
    /// blocks it appends a fresh block and allocates from it. Returns the set,
    /// its owning pool (so the caller can pair it for a correct later free), and
    /// whether a new block was grown (the caller emits the cap-pressure proxy).
    ///
    /// A genuine allocation error (out of host/device memory) propagates; only
    /// the two pool-exhaustion codes trigger growth.
    pub(crate) unsafe fn allocate(
        &mut self,
        device: &ash::Device,
        dsl: vk::DescriptorSetLayout,
    ) -> Result<(vk::DescriptorSet, vk::DescriptorPool, bool), DrawError> {
        let layouts = [dsl];
        for &pool in self.blocks.iter().rev() {
            let info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(pool)
                .set_layouts(&layouts);
            match device.allocate_descriptor_sets(&info) {
                Ok(sets) => return Ok((sets[0], pool, false)),
                Err(vk::Result::ERROR_OUT_OF_POOL_MEMORY)
                | Err(vk::Result::ERROR_FRAGMENTED_POOL) => continue,
                Err(e) => return Err(DrawError::VkCall(VkCall::new(VkOp::DescArenaAllocSets, e))),
            }
        }
        // Every existing block is full (or the arena is empty pre-init and this
        // is the first allocation) — grow.
        let pool = Self::create_block(device)?;
        self.blocks.push(pool);
        let info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(&layouts);
        let sets = device
            .allocate_descriptor_sets(&info)
            .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::DescArenaAllocSetsGrown, e)))?;
        Ok((sets[0], pool, true))
    }

    /// Free a batch of `(set, owning_pool)` pairs, grouping by owning pool so
    /// each `vkFreeDescriptorSets` targets the pool the set was allocated from.
    pub(crate) unsafe fn free(
        &self,
        device: &ash::Device,
        sets: &[(vk::DescriptorSet, vk::DescriptorPool)],
    ) {
        if sets.is_empty() {
            return;
        }
        for (pool, group) in group_by_pool(&self.blocks, sets) {
            if !group.is_empty() {
                if let Err(error) = free_sets_call(device.free_descriptor_sets(pool, &group)) {
                    crate::observe::Emit::decline("desc_arena_free", &error)
                        .field("sets", group.len())
                        .fail_once(pool.as_raw());
                }
            }
        }
    }

    /// Number of live blocks — the steady-state cap-pressure signal (1 = no
    /// growth ever occurred).
    pub(crate) fn block_count(&self) -> usize {
        self.blocks.len()
    }

    pub(crate) unsafe fn destroy(&mut self, device: &ash::Device) {
        for pool in self.blocks.drain(..) {
            device.destroy_descriptor_pool(pool, None);
        }
    }
}

/// Group `(set, owning_pool)` pairs by owning pool, in `blocks` order. A pair
/// whose pool is not among `blocks` is dropped (cannot happen for arena-owned
/// sets; guards against a stale handle rather than freeing to a wrong pool).
/// Pure (no Vulkan calls) so the free-routing invariant is unit-testable.
fn group_by_pool(
    blocks: &[vk::DescriptorPool],
    sets: &[(vk::DescriptorSet, vk::DescriptorPool)],
) -> Vec<(vk::DescriptorPool, Vec<vk::DescriptorSet>)> {
    blocks
        .iter()
        .map(|&block| {
            let owned: Vec<vk::DescriptorSet> = sets
                .iter()
                .filter(|(_, p)| *p == block)
                .map(|(s, _)| *s)
                .collect();
            (block, owned)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk::Handle;

    fn pool(raw: u64) -> vk::DescriptorPool {
        vk::DescriptorPool::from_raw(raw)
    }
    fn set(raw: u64) -> vk::DescriptorSet {
        vk::DescriptorSet::from_raw(raw)
    }

    #[test]
    fn frees_route_to_each_owning_pool() {
        let blocks = [pool(1), pool(2)];
        let sets = [
            (set(10), pool(1)),
            (set(11), pool(2)),
            (set(12), pool(1)),
            (set(13), pool(2)),
        ];
        let groups = group_by_pool(&blocks, &sets);
        assert_eq!(groups.len(), 2, "one group per block");
        assert_eq!(groups[0].0, pool(1));
        assert_eq!(
            groups[0].1,
            vec![set(10), set(12)],
            "block-1 sets, in order"
        );
        assert_eq!(groups[1].0, pool(2));
        assert_eq!(
            groups[1].1,
            vec![set(11), set(13)],
            "block-2 sets, in order"
        );
    }

    #[test]
    fn single_block_groups_all_together() {
        let blocks = [pool(7)];
        let sets = [(set(1), pool(7)), (set(2), pool(7))];
        let groups = group_by_pool(&blocks, &sets);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].1, vec![set(1), set(2)]);
    }

    #[test]
    fn set_from_unknown_pool_is_dropped_not_misrouted() {
        // A set whose pool is not a live block must never be freed to a
        // different block (that is UB). group_by_pool simply omits it.
        let blocks = [pool(1)];
        let sets = [(set(10), pool(1)), (set(99), pool(42))];
        let groups = group_by_pool(&blocks, &sets);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].1, vec![set(10)], "only the block-1 set is freed");
    }

    #[test]
    fn descriptor_free_failure_names_the_free_operation() {
        use crate::observe::Decline as _;
        let error = free_sets_call(Err(vk::Result::ERROR_OUT_OF_HOST_MEMORY))
            .expect_err("synthetic free failure must decline");
        assert_eq!(error.slug(), "vk_desc_arena_free_sets");
        let line = crate::observe::Emit::decline("desc_arena_free", &error)
            .field("sets", 3)
            .render();
        assert!(line.starts_with("desc_arena_free reason=vk_desc_arena_free_sets "));
        assert!(line.contains(" sets=3"));
    }
}

#[cfg(test)]
mod device_tests {
    use super::*;
    use crate::backend::vulkan::engine::context::DeviceContext;
    use ash::vk::Handle;

    /// The growth mechanism against a real driver: exhausting a block's
    /// `max_sets` must append a fresh block and keep allocating (not drop the
    /// draw), every set must be a distinct valid handle spanning both blocks,
    /// and freeing must route each set to its owning block without a driver
    /// fault. Skips cleanly when no GPU / Vulkan is present.
    #[test]
    fn arena_grows_on_exhaustion_and_frees_route_correctly() {
        crate::observe::redirect_logs_for_tests();
        let mut ctx = match unsafe { DeviceContext::create() } {
            Ok(c) => c,
            Err(e) => {
                eprintln!("SKIP desc arena growth: no device ({e})");
                return;
            }
        };
        // One STORAGE_BUFFER binding per set: each allocation consumes one set
        // slot + one per-type descriptor from a block.
        let bindings = [vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)];
        let dsl = unsafe {
            ctx.device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )
        }
        .expect("create test descriptor set layout");

        let mut arena = DescriptorArena::empty();
        unsafe { arena.create_first_block(&ctx.device) }.expect("first block");
        assert_eq!(arena.block_count(), 1, "one block before any growth");

        // Allocate two blocks' worth to guarantee at least one growth even if the
        // driver holds a few extra sets per pool beyond max_sets.
        let n = (DESC_BLOCK_MAX_SETS as usize) * 2;
        let mut sets = Vec::with_capacity(n);
        let mut grew_seen = false;
        for i in 0..n {
            let (s, pool, grew) = unsafe { arena.allocate(&ctx.device, dsl) }
                .unwrap_or_else(|e| panic!("allocate {i}: {e:?}"));
            assert!(!s.is_null(), "allocated set {i} is non-null");
            grew_seen |= grew;
            sets.push((s, pool));
        }
        assert!(grew_seen, "exhausting a block must report growth");
        assert!(
            arena.block_count() >= 2,
            "arena grew at least one overflow block (blocks={})",
            arena.block_count()
        );
        let distinct_sets: std::collections::HashSet<u64> =
            sets.iter().map(|(s, _)| s.as_raw()).collect();
        assert_eq!(distinct_sets.len(), n, "every allocated set is distinct");
        let distinct_pools: std::collections::HashSet<u64> =
            sets.iter().map(|(_, p)| p.as_raw()).collect();
        assert!(
            distinct_pools.len() >= 2,
            "sets span at least two blocks (pools={})",
            distinct_pools.len()
        );

        // Route every set back to its owning block; a misroute would fault here.
        unsafe { arena.free(&ctx.device, &sets) };
        unsafe {
            ctx.device.destroy_descriptor_set_layout(dsl, None);
            arena.destroy(&ctx.device);
            ctx.destroy();
        }
    }
}
