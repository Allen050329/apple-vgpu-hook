//! Offset suballocator for the HOST_VISIBLE upload (staging) buffers.
//!
//! The staging recycle pool hits ~99.6 % of the time, so what it costs is
//! decided entirely by its misses — and a miss used to be a full
//! `vkAllocateMemory`. Measured on a driven x86/PCI boot, `staging_pool`'s
//! per-bucket mean miss cost is **~0.4-0.6 ms whatever the size**: a 64-byte
//! staging miss read 421 us against 644 us for a 256 KiB one (120 and 548
//! samples respectively). Size adds on top of that floor but does not explain
//! it, so the floor is a per-`VkDeviceMemory` cost, not a per-byte one. At the
//! ~1 500 misses a boot the pool actually takes, that floor alone is most of a
//! second of stall — and it lands in exactly the wrong place, because a miss
//! happens under the engine lock on the first composite after idle, which is
//! the cold-window hitch.
//!
//! So this module does for upload buffers what [`super::slab`] does for
//! DEVICE_LOCAL images: sub-allocate many buffer binds out of a few large
//! blocks. A miss becomes `vkCreateBuffer` + [`BlockPlan::carve`] +
//! `vkBindBufferMemory` — no allocation, and no `vkMapMemory` either, because
//! the block is mapped once when it is created and every sub-allocation is a
//! pointer into that one mapping.
//!
//! # Why this is a separate pool from [`super::slab::SlabPool`]
//!
//! `SlabPool` is `DEVICE_LOCAL` optimal-tiled images and says so: it aligns
//! every carve to `bufferImageGranularity` because it must never let a linear
//! resource share a granularity window with a non-linear one. These blocks hold
//! **only linear buffers**, so that rule does not apply to them at all and
//! aligning to it would round a 64-byte staging carve up to a page. The two
//! also want different memory classes ([`MemoryClass::Upload`] vs
//! `DeviceLocal`), different size classes, and different lifetime keys — a
//! staging slot carries its own token, where an image is looked up by handle.
//! What they genuinely share is the free-list core, and that is what is shared:
//! [`BlockPlan`].
//!
//! # What it measured
//!
//! Driven x86/PCI boot, against the same `staging_pool` census line at the same
//! miss count (1 536) from a boot before this module existed — so the two are
//! directly comparable rather than scaled:
//!
//! ```text
//!              misses   total     mean    64 B   256 KiB   16 MiB
//! before         1536   1231 ms   801 us  421 us   644 us  4802 us
//! after          1536    198 ms   129 us    0 us    50 us     5 us
//! ```
//!
//! Six times less total, and the shape of the win is the point: the floor is
//! gone. The 256 KiB bucket is 920 of those 1 536 misses and fell 13-fold; the
//! sub-4 KiB buckets round to zero. `vk_alloc_sites` read `staging_block=9:130`
//! (count:MiB) where the same census before read `staging=242:134` — the same
//! bytes through 9 allocations instead of 242. `draw_phase stage_us` per draw in
//! a driven window read 55 us against the 3 525 us/draw of the cold window that
//! `vkAllocateMemory` was first caught in.
//!
//! What remains is priced honestly by the same census: the buckets that still
//! cost something (`8388608:2252`) are the misses that landed on a new block, so
//! the residual *is* the block allocations and nothing else. Blocks churned 13
//! allocations against 10 frees over five minutes, which is the working set
//! crossing [`HOST_SLAB_SIZE`] rather than a leak.

use ash::vk;

use super::context::DeviceContext;
use super::counters::EngineCounters;
use super::pools::{allocate_memory_timed, AllocSite};
use super::slab::{BlockPlan, SlabDecline};
use super::types::DrawError;
use super::vk_call::{VkCall, VkOp};
use crate::backend::vulkan::caps::MemoryClass;

/// Block size for the large size class.
///
/// Derived from the two measured quantities this allocator trades between. The
/// *cost* side: host-visible `vkAllocateMemory` on the measured host prices a
/// 16 MiB request at 4.8-18.4 ms, which is what the existing pool already pays
/// for its single largest bucket — so a block never costs more than the worst
/// allocation this code path made before it existed, and it is paid a handful
/// of times a boot instead of ~1 500. The *capacity* side: `staging_pool` reads
/// `live=3..28` slots with the bucket histogram dominated by 256 KiB, so the
/// whole live+free upload working set is tens of MiB — a handful of these
/// blocks holds it. Going larger would buy fewer allocations at the price of a
/// worse single stall and more resident host RAM, which on an iGPU is shared
/// guest RAM.
const HOST_SLAB_SIZE: u64 = 16 << 20;

/// Carves strictly below this bind from the **small** class. Same reasoning as
/// [`super::slab`]'s split: the staging bucket histogram is strongly bimodal
/// (hundreds of ≤64 KiB carves for vertex/index/uniform runs, plus a handful of
/// multi-MiB frame-sized ones), and mixed first-fit lets a stable small carve
/// sit in the middle of a large block and stop it ever emptying. 128 KiB is
/// above the small cluster's top bucket and below every frame-sized upload.
const HOST_SMALL_CLASS_MAX: u64 = 128 << 10;

/// Block size for the small class. 2 MiB holds 32 max-size (64 KiB) small
/// carves, or hundreds of the 64-byte to 8 KiB ones that dominate the histogram
/// by count — the whole small working set in one block, at ~2 ms to allocate.
const HOST_SMALL_SLAB_SIZE: u64 = 2 << 20;

/// Fully-empty shared blocks kept resident rather than freed on release.
///
/// One, not [`super::slab`]'s two: a block here is host RAM held from the guest
/// (shared guest RAM on an iGPU), and unlike the image slab the pool in front of
/// this one already absorbs steady-state churn at a 99.6 % hit rate — a carve
/// only reaches this allocator when that pool missed. So the spare exists to
/// stop a single empty↔full cycle re-paying a block, and no more.
const HOST_SLAB_KEEP_EMPTY: usize = 1;

/// The size classes only segregate anything while these hold, and a later edit
/// to one constant that breaks one of them would show up as fragmentation
/// rather than as an error. A build-time check is the right shape for a
/// relationship between constants: there is nothing to run.
const _: () = {
    // A small block must hold many small carves, or the class buys nothing over
    // giving each small carve its own block.
    assert!(HOST_SMALL_SLAB_SIZE / HOST_SMALL_CLASS_MAX >= 16);
    // The large class must be able to host a carve that is not small, or every
    // non-small carve goes dedicated.
    assert!(HOST_SLAB_SIZE > HOST_SMALL_CLASS_MAX);
};

/// A live sub-allocation: which block, at what offset, backed by which
/// `VkDeviceMemory`, and where in the block's single persistent mapping it
/// starts.
#[derive(Clone, Copy)]
pub(crate) struct HostSlabToken {
    block: u32,
    offset: u64,
    size: u64,
    /// Bind target for `vkBindBufferMemory`, shared with every other carve in
    /// the same block. Never passed to `vkFreeMemory` by the holder.
    pub memory: vk::DeviceMemory,
    /// Host address of this carve: the block's mapping base plus [`Self::offset`].
    ///
    /// A `usize` rather than a pointer for the same reason
    /// [`super::pools::BufferSlot::mapped`] is one — the engine state stays
    /// `Send`.
    pub mapped: usize,
}

impl HostSlabToken {
    /// Bind offset for `vkBindBufferMemory`.
    pub(crate) fn offset(&self) -> u64 {
        self.offset
    }
}

struct HostBlock {
    memory: vk::DeviceMemory,
    plan: BlockPlan,
    mem_type: u32,
    /// Host address of the whole block, mapped once at allocation and left
    /// mapped for the block's life. Vulkan permits a memory object to stay
    /// mapped until it is freed, and `vkFreeMemory` unmaps implicitly — which is
    /// what makes a carve's `mapped` free to hand out.
    base: usize,
    /// A carve larger than [`HOST_SLAB_SIZE`] got a block of exactly its own
    /// size; never shared, and freed the moment it empties.
    dedicated: bool,
    /// Small-size-class block ([`HOST_SMALL_SLAB_SIZE`], holds only carves
    /// below [`HOST_SMALL_CLASS_MAX`]).
    small: bool,
    /// Set once the free-list invariant is seen violated: the block is leaked
    /// (never carved from, never freed) so a logic bug cannot alias two live
    /// buffers onto the same bytes.
    poisoned: bool,
}

/// Block pool for HOST_VISIBLE upload memory.
///
/// Safety net, as in [`super::slab`]: every mutation is followed by a
/// [`BlockPlan::well_formed`] check, and a violation fail-logs and poisons the
/// block. An allocator bug degrades to a leak plus a fresh block rather than to
/// two staging buffers silently writing the same bytes.
pub(crate) struct HostSlabPool {
    blocks: Vec<Option<HostBlock>>,
    block_allocs: u64,
    block_frees: u64,
    sub_allocs: u64,
    sub_frees: u64,
    invariant_violations: u64,
}

impl HostSlabPool {
    pub(crate) fn new() -> Self {
        Self {
            blocks: Vec::new(),
            block_allocs: 0,
            block_frees: 0,
            sub_allocs: 0,
            sub_frees: 0,
            invariant_violations: 0,
        }
    }

    /// Carve a sub-range for a buffer with these memory requirements.
    ///
    /// Reuses a resident block of the same memory type and size class when the
    /// request fits; else allocates one. `note_alloc`/timing is charged only on
    /// a real block allocation, so `vk_alloc_sites staging_block` counts true
    /// `vkAllocateMemory` calls and collapses as the pool warms.
    pub(crate) unsafe fn acquire(
        &mut self,
        ctx: &DeviceContext,
        req: &vk::MemoryRequirements,
        counters: &EngineCounters,
    ) -> Result<HostSlabToken, DrawError> {
        let mem_type = ctx
            .memory_type_for(req.memory_type_bits, MemoryClass::Upload)
            .ok_or({
                DrawError::Unsupported(super::reason::DrawReason::NoHostVisibleMemoryForStaging {
                    memory_type_bits: req.memory_type_bits,
                })
            })?;
        let size = req.size;
        if size == 0 {
            return Err(DrawError::Slab(SlabDecline::ZeroSize {
                memory_type_bits: req.memory_type_bits,
            }));
        }
        // Buffers only, so `bufferImageGranularity` does not apply: the spec's
        // padding rule governs an adjacent *linear/non-linear* pair, and there
        // is no non-linear resource in any of these blocks. The bind's own
        // requirement is the whole constraint.
        let align = req.alignment.max(1);

        if size > HOST_SLAB_SIZE {
            return self.new_block(ctx, size, mem_type, align, size, true, false, counters);
        }

        let want_small = size < HOST_SMALL_CLASS_MAX;
        for i in 0..self.blocks.len() {
            let hit = match &mut self.blocks[i] {
                Some(b)
                    if !b.poisoned
                        && !b.dedicated
                        && b.small == want_small
                        && b.mem_type == mem_type =>
                {
                    b.plan.carve(size, align)
                }
                _ => None,
            };
            if let Some(offset) = hit {
                if !self.check_block(i) {
                    // The carve corrupted the free list — poisoned; try elsewhere.
                    continue;
                }
                let b = self.blocks[i].as_ref().expect("just carved");
                self.sub_allocs += 1;
                return Ok(HostSlabToken {
                    block: i as u32,
                    offset,
                    size,
                    memory: b.memory,
                    mapped: b.base + offset as usize,
                });
            }
        }

        let block_size = if want_small {
            HOST_SMALL_SLAB_SIZE
        } else {
            HOST_SLAB_SIZE
        };
        self.new_block(
            ctx, block_size, mem_type, align, size, false, want_small, counters,
        )
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn new_block(
        &mut self,
        ctx: &DeviceContext,
        block_size: u64,
        mem_type: u32,
        align: u64,
        carve: u64,
        dedicated: bool,
        small: bool,
        counters: &EngineCounters,
    ) -> Result<HostSlabToken, DrawError> {
        let memory = allocate_memory_timed(
            ctx,
            &vk::MemoryAllocateInfo::default()
                .allocation_size(block_size)
                .memory_type_index(mem_type),
            AllocSite::StagingBlock,
        )
        .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::PoolsAllocStaging, e)))?;
        counters.note_alloc();
        self.block_allocs += 1;
        let base = match ctx
            .device
            .map_memory(memory, 0, block_size, vk::MemoryMapFlags::empty())
        {
            Ok(p) => p as usize,
            Err(e) => {
                ctx.device.free_memory(memory, None);
                self.block_frees += 1;
                return Err(DrawError::VkCall(VkCall::new(VkOp::PoolsMapStaging, e)));
            }
        };
        let mut plan = BlockPlan::new(block_size);
        let offset = match plan.carve(carve, align) {
            Some(o) => o,
            None => {
                // A fresh block that cannot host its own reason-for-being is an
                // align/size logic error; free it rather than leak it.
                ctx.device.free_memory(memory, None);
                self.block_frees += 1;
                return Err(DrawError::Slab(SlabDecline::FreshBlockCarve {
                    block_size,
                    carve,
                    alignment: align,
                }));
            }
        };
        let idx = self.insert_block(HostBlock {
            memory,
            plan,
            mem_type,
            base,
            dedicated,
            small,
            poisoned: false,
        });
        self.sub_allocs += 1;
        // Always-on and low-frequency by construction: a block event fires a
        // handful of times a boot, not once per staging acquire, so
        // `sub_allocs` climbing while `block_allocs` stays flat is the whole
        // claim of this module in one line.
        crate::observe::off(format!(
            "host_slab ev=alloc size={block_size} dedicated={} small={} block_allocs={} \
             block_frees={} sub_allocs={} sub_frees={} resident_mb={} live_mb={} violations={}",
            dedicated as u8,
            small as u8,
            self.block_allocs,
            self.block_frees,
            self.sub_allocs,
            self.sub_frees,
            self.resident_bytes() >> 20,
            self.live_bytes() >> 20,
            self.invariant_violations,
        ));
        Ok(HostSlabToken {
            block: idx,
            offset,
            size: carve,
            memory,
            mapped: base + offset as usize,
        })
    }

    fn insert_block(&mut self, block: HostBlock) -> u32 {
        if let Some(i) = self.blocks.iter().position(Option::is_none) {
            self.blocks[i] = Some(block);
            i as u32
        } else {
            self.blocks.push(Some(block));
            (self.blocks.len() - 1) as u32
        }
    }

    /// Return a carve to its block. The caller has already destroyed the
    /// `VkBuffer` bound to it and must never `vkFreeMemory` the token's memory.
    pub(crate) unsafe fn release(&mut self, device: &ash::Device, token: HostSlabToken) {
        let idx = token.block as usize;
        match self.release_preflight(token.block, token.offset, token.size) {
            Ok(true) => {}
            Ok(false) => return,
            Err(decline) => {
                crate::observe::Emit::decline("host_slab", &decline).fail_once(u64::from(token.block));
                return;
            }
        }
        let b = self.blocks[idx]
            .as_mut()
            .expect("release preflight proved the block exists");
        b.plan.release(token.offset, token.size);
        self.sub_frees += 1;
        let (empty, dedicated) = (b.plan.is_empty(), b.dedicated);
        if !self.check_block(idx) {
            // A corrupt release poisoned the block; leak it (already logged).
            return;
        }
        if empty {
            if dedicated {
                // A dedicated block exists for one oversized carve and can never
                // host another; holding it as a spare would just be a leak with
                // a nicer name.
                self.free_block(device, idx);
            } else {
                self.trim_empty_blocks(device, HOST_SLAB_KEEP_EMPTY);
            }
        }
    }

    /// Validate a token's block and range without touching Vulkan state.
    /// `Ok(false)` is the deliberate leak policy for an already-poisoned block;
    /// the poisoning event itself was logged when it happened.
    fn release_preflight(&self, block: u32, offset: u64, size: u64) -> Result<bool, SlabDecline> {
        match self.blocks.get(block as usize).and_then(Option::as_ref) {
            Some(b) if b.poisoned => Ok(false),
            Some(b) => {
                b.plan.release_preflight(block, offset, size)?;
                Ok(true)
            }
            None => Err(SlabDecline::ReleaseBlockMissing {
                block,
                block_slots: self.blocks.len(),
            }),
        }
    }

    unsafe fn free_block(&mut self, device: &ash::Device, idx: usize) {
        let Some(b) = self.blocks[idx].take() else {
            return;
        };
        // `vkFreeMemory` unmaps implicitly, so the block's persistent mapping
        // needs no separate `vkUnmapMemory`.
        device.free_memory(b.memory, None);
        self.block_frees += 1;
        crate::observe::off(format!(
            "host_slab ev=free block_allocs={} block_frees={} sub_allocs={} sub_frees={} \
             resident_mb={} live_mb={}",
            self.block_allocs,
            self.block_frees,
            self.sub_allocs,
            self.sub_frees,
            self.resident_bytes() >> 20,
            self.live_bytes() >> 20,
        ));
    }

    /// Free fully-empty shared blocks beyond `keep` spares per size class,
    /// returning the count freed.
    ///
    /// Called from [`Self::release`] rather than from the idle drain, and the
    /// difference from [`super::slab::SlabPool`] is worth stating: that one
    /// *retains* on release and needs an idle sweep to give the retained blocks
    /// back later. This one settles to its policy on every release, and a block
    /// can only become empty by a release, so there is nothing left for an idle
    /// pass to find.
    unsafe fn trim_empty_blocks(&mut self, device: &ash::Device, keep: usize) -> usize {
        let victims = self.empty_block_victims(keep);
        let mut freed = 0;
        for idx in victims {
            if self.blocks[idx].is_some() {
                self.free_block(device, idx);
                freed += 1;
            }
        }
        freed
    }

    /// Indices of surplus empty shared blocks to free — `keep` spares **of each
    /// size class**, not `keep` in total.
    ///
    /// Per class because the spare exists to stop the next carve re-paying a
    /// block allocation, and a carve can only reuse a block of its own class. A
    /// total budget would let the small block be the only survivor and leave the
    /// first frame-sized upload after idle allocating anyway, which is the
    /// cold-window hitch this allocator is for.
    ///
    /// Pure — split out so the selection is unit-testable without a device.
    fn empty_block_victims(&self, keep: usize) -> Vec<usize> {
        let mut victims = Vec::new();
        for class in [true, false] {
            let empties = self.blocks.iter().enumerate().filter_map(|(i, s)| match s {
                Some(b) if !b.dedicated && !b.poisoned && b.small == class && b.plan.is_empty() => {
                    Some(i)
                }
                _ => None,
            });
            victims.extend(empties.skip(keep));
        }
        victims.sort_unstable();
        victims
    }

    /// Host bytes this pool currently holds from the system, block padding and
    /// all — what an iGPU's shared-guest-RAM budget actually loses to it.
    pub(crate) fn resident_bytes(&self) -> u64 {
        self.blocks
            .iter()
            .flatten()
            .map(|b| b.plan.size())
            .sum::<u64>()
    }

    /// Bytes currently carved out. `resident_bytes() - live_bytes()` is the
    /// slack a further carve can use without a new block.
    pub(crate) fn live_bytes(&self) -> u64 {
        self.blocks
            .iter()
            .flatten()
            .map(|b| b.plan.size() - b.plan.free_bytes())
            .sum::<u64>()
    }

    /// Verify the block's free-list invariant after a mutation; on violation,
    /// fail-log once and poison the block. Returns whether the block is usable.
    fn check_block(&mut self, idx: usize) -> bool {
        if let Some(Some(b)) = self.blocks.get_mut(idx) {
            if b.plan.well_formed() {
                return true;
            }
            b.poisoned = true;
            self.invariant_violations += 1;
            let decline = SlabDecline::FreeListInvariant {
                block: idx,
                size: b.plan.size(),
                free_bytes: b.plan.free_bytes(),
            };
            crate::observe::Emit::decline("host_slab", &decline).fail_once(idx as u64);
            return false;
        }
        false
    }

    /// Destroy every remaining block (device teardown / recreate). The caller
    /// has already destroyed every `VkBuffer` bound into these blocks.
    pub(crate) unsafe fn destroy_all(&mut self, device: &ash::Device) {
        for b in self.blocks.drain(..).flatten() {
            device.free_memory(b.memory, None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A block with no live carves and one spare already retained is surplus;
    /// the first `keep` are not. Mirrors what the idle drain asks for.
    #[test]
    fn empty_block_victims_keeps_the_first_n() {
        let block = |free: bool| {
            Some(HostBlock {
                memory: vk::DeviceMemory::null(),
                plan: {
                    let mut p = BlockPlan::new(1024);
                    if !free {
                        p.carve(64, 1).expect("fresh block hosts a 64-byte carve");
                    }
                    p
                },
                mem_type: 0,
                base: 0,
                dedicated: false,
                small: false,
                poisoned: false,
            })
        };
        let pool = HostSlabPool {
            blocks: vec![block(true), block(false), block(true), block(true)],
            block_allocs: 4,
            block_frees: 0,
            sub_allocs: 1,
            sub_frees: 0,
            invariant_violations: 0,
        };
        assert_eq!(pool.empty_block_victims(1), vec![2, 3]);
        assert_eq!(pool.empty_block_victims(0), vec![0, 2, 3]);
        assert!(pool.empty_block_victims(3).is_empty());
    }

    /// The occupied block is never a victim however low `keep` goes, and the
    /// resident/live split reports the carve.
    #[test]
    fn occupied_blocks_are_never_trimmed_and_are_counted_live() {
        let mut plan = BlockPlan::new(4096);
        let off = plan.carve(256, 256).expect("carve fits");
        assert_eq!(off, 0);
        let pool = HostSlabPool {
            blocks: vec![Some(HostBlock {
                memory: vk::DeviceMemory::null(),
                plan,
                mem_type: 0,
                base: 0,
                dedicated: false,
                small: false,
                poisoned: false,
            })],
            block_allocs: 1,
            block_frees: 0,
            sub_allocs: 1,
            sub_frees: 0,
            invariant_violations: 0,
        };
        assert!(pool.empty_block_victims(0).is_empty());
        assert_eq!(pool.resident_bytes(), 4096);
        assert_eq!(pool.live_bytes(), 256);
    }

    /// The spare budget is per size class, so a small block surviving never
    /// costs the large class its spare.
    ///
    /// This is the case a total budget gets wrong, and it gets it wrong in the
    /// direction that matters: with one spare shared between them, the small
    /// block (allocated first, since small carves are the common ones) is the
    /// survivor and the first frame-sized upload after idle allocates a block
    /// anyway — which is the stall the whole allocator exists to remove.
    #[test]
    fn the_empty_spare_budget_is_per_size_class() {
        let block = |small: bool| {
            Some(HostBlock {
                memory: vk::DeviceMemory::null(),
                plan: BlockPlan::new(1024),
                mem_type: 0,
                base: 0,
                dedicated: false,
                small,
                poisoned: false,
            })
        };
        // Two of each class, all empty: one of each survives, the other two go.
        let pool = HostSlabPool {
            blocks: vec![block(true), block(false), block(true), block(false)],
            block_allocs: 4,
            block_frees: 0,
            sub_allocs: 0,
            sub_frees: 0,
            invariant_violations: 0,
        };
        assert_eq!(pool.empty_block_victims(1), vec![2, 3]);
        assert_eq!(pool.empty_block_victims(0), vec![0, 1, 2, 3]);
        assert!(pool.empty_block_victims(2).is_empty());
    }

    /// A dedicated block is excluded from the empty-spare accounting entirely:
    /// it frees itself on release, so counting it would let a shared spare be
    /// trimmed in its place.
    #[test]
    fn dedicated_and_poisoned_blocks_are_not_spares() {
        let block = |dedicated: bool, poisoned: bool| {
            Some(HostBlock {
                memory: vk::DeviceMemory::null(),
                plan: BlockPlan::new(1024),
                mem_type: 0,
                base: 0,
                dedicated,
                small: false,
                poisoned,
            })
        };
        let pool = HostSlabPool {
            blocks: vec![block(true, false), block(false, true), block(false, false)],
            block_allocs: 3,
            block_frees: 0,
            sub_allocs: 0,
            sub_frees: 0,
            invariant_violations: 1,
        };
        assert!(pool.empty_block_victims(0) == vec![2]);
        assert!(pool.empty_block_victims(1).is_empty());
    }

    /// `release_preflight` refuses a token pointing at a freed block slot and at
    /// a range that is already free, rather than inserting an overlap that would
    /// let two carves alias.
    #[test]
    fn release_preflight_refuses_missing_block_and_double_release() {
        let mut plan = BlockPlan::new(4096);
        let off = plan.carve(256, 1).expect("carve fits");
        let pool = HostSlabPool {
            blocks: vec![
                None,
                Some(HostBlock {
                    memory: vk::DeviceMemory::null(),
                    plan,
                    mem_type: 0,
                    base: 0,
                    dedicated: false,
                    small: false,
                    poisoned: false,
                }),
            ],
            block_allocs: 1,
            block_frees: 1,
            sub_allocs: 1,
            sub_frees: 0,
            invariant_violations: 0,
        };
        assert!(matches!(
            pool.release_preflight(0, 0, 256),
            Err(SlabDecline::ReleaseBlockMissing { .. })
        ));
        assert!(matches!(pool.release_preflight(1, off, 256), Ok(true)));
        // The tail past the carve is free, so releasing it would overlap.
        assert!(matches!(
            pool.release_preflight(1, 256, 256),
            Err(SlabDecline::ReleaseRangeAlreadyFree { .. })
        ));
    }

}
