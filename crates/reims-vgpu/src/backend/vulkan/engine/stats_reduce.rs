//! GPU-side present-proxy stats reduction — the zero-copy oracle.
//!
//! The always-on present proxies need to *measure* the finished frame, not to
//! own its pixels. This module dispatches `shaders/present_stats.comp` over the
//! resident the display already holds and reads back a **32-byte** stats block,
//! so no full-frame GPU→CPU copy happens on any path — default boot or not.
//!
//! # Why a dedicated pool rather than the guest compute path
//!
//! `execute_compute_inner` would give pipeline/descriptor/barrier handling for
//! free, but it is built for guest dispatches: it has no affordance for binding
//! a [`TargetIdentity`] resident (only `ComputeStorageResidencyKey`), its
//! sampled-image validation demands a full `w*h*bpt` CPU byte payload — a
//! display-sized allocation we exist to avoid — and its writable-SSBO path ends
//! in a synchronous `retire_all` that quiesces the whole ring. All three defeat
//! the purpose.
//!
//! So this mirrors [`super::prefetch`] instead: a dedicated command buffer, a
//! dedicated fence, and a persistently-mapped host-coherent buffer per slot,
//! with an `arm` / `consume` / `cancel` lifecycle keyed by
//! `(identity, generation, seq)`. The pipeline itself still comes from the
//! shared object caches, which are content-keyed (`ComputePipelineKey` is
//! documented "Never funcId"), so a host-authored kernel coexists with guest
//! ones without special-casing.
//!
//! # Zero cost on the present path
//!
//! `arm` records and submits, then returns — it never waits. `consume` polls
//! with a non-blocking `vkGetFenceStatus` and, on a miss, reports "not ready"
//! rather than blocking. The present drain therefore pays a dispatch submit and
//! nothing else.
//!
//! Stats may consequently arrive a frame or more after the present they
//! describe. That is sound *because the block is labelled*: every slot carries
//! the `(identity, generation, seq)` it captured, and `consume` matches on the
//! exact triple, so a proxy is never handed stats belonging to a different
//! frame. This is the same a/b-glitch guard [`super::prefetch`] documents. It is
//! the reason a lagged **stats** readback is safe where a lagged **pixel**
//! readback was not.
//!
//! # Source binding
//!
//! The resident is bound SAMPLED (`slot.view`), not STORAGE: registry targets
//! are already created with `SAMPLED` usage, whereas adding `STORAGE` would
//! change the usage set on every registry target and break the `target_free`
//! recycle pool's identical-usage assumption.

use super::context::DeviceContext;
use super::counters::EngineCounters;
use super::types::{DrawError, TargetIdentity};
use super::vk_call::{VkCall, VkOp};
use crate::backend::vulkan::caps::MemoryClass;
use crate::backend::vulkan::translate;
use ash::vk;

fn stats_call<T>(op: VkOp, result: Result<T, vk::Result>) -> Result<T, DrawError> {
    result.map_err(|error| DrawError::VkCall(VkCall::new(op, error)))
}

/// Structural refusal before a stats-reduction dispatch reaches Vulkan. Both
/// values violate the arm contract; slot saturation and an unfinished fence
/// remain expected non-blocking control flow and are deliberately absent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum StatsReduceDecline {
    ZeroSequence,
    ZeroGeometry { width: u32, height: u32 },
}

impl crate::observe::Decline for StatsReduceDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::ZeroSequence => "vk_stats_reduce_zero_sequence",
            Self::ZeroGeometry { .. } => "vk_stats_reduce_zero_geometry",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::ZeroSequence => Vec::new(),
            Self::ZeroGeometry { width, height } => {
                vec![("width", width.to_string()), ("height", height.to_string())]
            }
        }
    }
}

fn validate_arm(seq: u64, width: u32, height: u32) -> Result<(), StatsReduceDecline> {
    if seq == 0 {
        Err(StatsReduceDecline::ZeroSequence)
    } else if width == 0 || height == 0 {
        Err(StatsReduceDecline::ZeroGeometry { width, height })
    } else {
        Ok(())
    }
}

/// Reduced present-frame statistics — what the proxies consume instead of an
/// 8 MiB frame. Field-for-field the output of `shaders/present_stats.comp`.
///
/// Byte-exact with the CPU reference implementations
/// (`observe::bgra_present_stats`, `present_proxy::edge_energy_bgra`); the
/// equality is asserted by `present_stats_gpu_matches_cpu_reference`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PresentStats {
    /// Nonzero bytes across all four channels.
    pub byte_nz: u32,
    /// Max byte over all four channels.
    pub byte_max: u8,
    /// Pixels whose `max(B,G,R)` is nonzero.
    pub rgb_nz: u32,
    /// Max of B/G/R over all pixels.
    pub max_rgb: u8,
    /// First pixel, memory order `[B, G, R, A]`.
    pub px0: [u8; 4],
    /// Edge energy, already `>> 8` to match `edge_energy_bgra`'s return.
    pub edge_energy: u32,
    /// Geometry the kernel actually reduced (echoed by the shader).
    pub width: u32,
    pub height: u32,
    /// Pixels whose alpha is nonzero.
    pub alpha_nz: u32,
    /// Pixels whose alpha is exactly 255.
    pub alpha_opaque: u32,
}

impl From<PresentStats> for super::content_stats::Color8ContentStats {
    /// The store-scatter path's content stats, sourced from the GPU reduction
    /// instead of a full-frame CPU readback. `rgb_max` is the shader's
    /// `max_rgb`: both are `max(B,G,R)` over all pixels.
    fn from(s: PresentStats) -> Self {
        Self {
            rgb_nz: s.rgb_nz as usize,
            rgb_max: s.max_rgb,
            alpha_nz: s.alpha_nz as usize,
            alpha_opaque: s.alpha_opaque as usize,
        }
    }
}

/// Raw shader output block: 10 `u32`s, std430, matching `StatsBuf` in the GLSL.
const STATS_WORDS: usize = 10;
const STATS_BYTES: u64 = (STATS_WORDS * 4) as u64;

/// Concurrent in-flight reductions. Small: one present is in flight at a time in
/// practice, and a stats block is 32 bytes, so this only bounds pathological
/// pile-ups.
const MAX_STATS_SLOTS: usize = 4;

/// One in-flight (or idle-reusable) reduction.
struct StatsSlot {
    cmd_buf: vk::CommandBuffer,
    fence: vk::Fence,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    /// Persistent host-coherent map of `buffer`; valid while `buffer != null`.
    mapped: *mut u8,
    descriptor_set: vk::DescriptorSet,
    /// Identity/gen/seq this dispatch captured, while `in_flight`.
    identity: Option<TargetIdentity>,
    generation: u64,
    seq: u64,
    /// True from submit until consumed/reclaimed; the GPU may still reference
    /// the buffer/CB/descriptor set while set.
    in_flight: bool,
}

impl StatsSlot {
    fn idle(cmd_buf: vk::CommandBuffer, fence: vk::Fence) -> Self {
        Self {
            cmd_buf,
            fence,
            buffer: vk::Buffer::null(),
            memory: vk::DeviceMemory::null(),
            mapped: std::ptr::null_mut(),
            descriptor_set: vk::DescriptorSet::null(),
            identity: None,
            generation: 0,
            seq: 0,
            in_flight: false,
        }
    }
}

/// Dedicated pool for stats reductions.
pub(crate) struct StatsReducePool {
    slots: Vec<StatsSlot>,
    /// Private descriptor pool: the shared one frees sets only when its ring
    /// slot retires, which would couple this lifecycle to the draw ring.
    desc_pool: vk::DescriptorPool,
    /// Point sampler. `texelFetch` ignores sampler state, but a `sampler`
    /// descriptor must still be bound for the separate texture/sampler layout.
    sampler: vk::Sampler,
    /// Cumulative diagnostics: `(arms, hits, misses, not_ready, saturated)`.
    arms: u64,
    hits: u64,
    misses: u64,
    not_ready: u64,
    saturated: u64,
}

// SAFETY: `mapped` is a raw pointer into host-coherent memory owned by this
// pool; it is only dereferenced under the engine mutex, exactly as
// `PrefetchPool` does for the same reason.
unsafe impl Send for StatsReducePool {}

impl StatsReducePool {
    pub(crate) fn new() -> Self {
        Self {
            slots: Vec::new(),
            desc_pool: vk::DescriptorPool::null(),
            sampler: vk::Sampler::null(),
            arms: 0,
            hits: 0,
            misses: 0,
            not_ready: 0,
            saturated: 0,
        }
    }

    /// `(arms, hits, misses, not_ready, saturated)` for the always-on census.
    pub(crate) fn stats(&self) -> (u64, u64, u64, u64, u64) {
        (
            self.arms,
            self.hits,
            self.misses,
            self.not_ready,
            self.saturated,
        )
    }

    unsafe fn ensure_shared(&mut self, ctx: &DeviceContext) -> Result<(), DrawError> {
        if self.desc_pool == vk::DescriptorPool::null() {
            let sizes = [
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::SAMPLED_IMAGE)
                    .descriptor_count(MAX_STATS_SLOTS as u32),
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::SAMPLER)
                    .descriptor_count(MAX_STATS_SLOTS as u32),
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(MAX_STATS_SLOTS as u32),
            ];
            self.desc_pool = ctx
                .device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(MAX_STATS_SLOTS as u32)
                        .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
                        .pool_sizes(&sizes),
                    None,
                )
                .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::StatsDescPool, e)))?;
        }
        if self.sampler == vk::Sampler::null() {
            self.sampler = ctx
                .device
                .create_sampler(
                    &vk::SamplerCreateInfo::default()
                        .mag_filter(translate::sampler::EXACT_TEXEL_FILTER)
                        .min_filter(translate::sampler::EXACT_TEXEL_FILTER)
                        .mipmap_mode(translate::sampler::EXACT_TEXEL_MIPMAP_MODE)
                        .address_mode_u(translate::sampler::EXACT_TEXEL_ADDRESS_MODE)
                        .address_mode_v(translate::sampler::EXACT_TEXEL_ADDRESS_MODE)
                        .address_mode_w(translate::sampler::EXACT_TEXEL_ADDRESS_MODE),
                    None,
                )
                .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::StatsSampler, e)))?;
        }
        Ok(())
    }

    unsafe fn ensure_buffer(
        slot: &mut StatsSlot,
        ctx: &DeviceContext,
        counters: &EngineCounters,
    ) -> Result<(), DrawError> {
        if slot.buffer != vk::Buffer::null() {
            return Ok(());
        }
        let buffer = ctx
            .device
            .create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(STATS_BYTES)
                    .usage(
                        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                    )
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )
            .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::StatsCreateBuffer, e)))?;
        counters.note_create();
        let req = ctx.device.get_buffer_memory_requirements(buffer);
        let mt = ctx
            .memory_type_for(req.memory_type_bits, MemoryClass::Readback)
            .ok_or_else(|| {
                DrawError::Unsupported(super::reason::DrawReason::NoHostVisibleMemoryForStats {
                    memory_type_bits: req.memory_type_bits,
                })
            })?;
        let memory = ctx
            .device
            .allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req.size)
                    .memory_type_index(mt),
                None,
            )
            .map_err(|e| {
                ctx.device.destroy_buffer(buffer, None);
                DrawError::VkCall(VkCall::new(VkOp::StatsAlloc, e))
            })?;
        counters.note_alloc();
        ctx.device
            .bind_buffer_memory(buffer, memory, 0)
            .map_err(|e| {
                ctx.device.free_memory(memory, None);
                ctx.device.destroy_buffer(buffer, None);
                DrawError::VkCall(VkCall::new(VkOp::StatsBind, e))
            })?;
        let mapped = ctx
            .device
            .map_memory(memory, 0, req.size, vk::MemoryMapFlags::empty())
            .map_err(|e| {
                ctx.device.free_memory(memory, None);
                ctx.device.destroy_buffer(buffer, None);
                DrawError::VkCall(VkCall::new(VkOp::StatsMap, e))
            })? as *mut u8;
        slot.buffer = buffer;
        slot.memory = memory;
        slot.mapped = mapped;
        Ok(())
    }

    unsafe fn alloc_slot(
        &mut self,
        ctx: &DeviceContext,
        cmd_pool: vk::CommandPool,
    ) -> Result<usize, DrawError> {
        let cmd_buf = ctx
            .device
            .allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(cmd_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
            .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::StatsAllocCb, e)))?[0];
        let fence = ctx
            .device
            .create_fence(&vk::FenceCreateInfo::default(), None)
            .map_err(|e| {
                ctx.device.free_command_buffers(cmd_pool, &[cmd_buf]);
                DrawError::VkCall(VkCall::new(VkOp::StatsCreateFence, e))
            })?;
        self.slots.push(StatsSlot::idle(cmd_buf, fence));
        Ok(self.slots.len() - 1)
    }

    /// Pick an idle slot, growing up to `MAX_STATS_SLOTS`, else reclaim one
    /// whose fence has already signalled. Returns `None` when saturated.
    unsafe fn pick_slot(
        &mut self,
        ctx: &DeviceContext,
        cmd_pool: vk::CommandPool,
    ) -> Option<usize> {
        if let Some(i) = self.slots.iter().position(|s| !s.in_flight) {
            return Some(i);
        }
        if self.slots.len() < MAX_STATS_SLOTS {
            return match self.alloc_slot(ctx, cmd_pool) {
                Ok(i) => Some(i),
                Err(e) => {
                    crate::observe::Emit::decline("stats_reduce", &e).fail_once(0);
                    None
                }
            };
        }
        // Non-blocking reclaim of a finished-but-unconsumed slot.
        for i in 0..self.slots.len() {
            match stats_call(
                VkOp::StatsFenceStatusReclaim,
                ctx.device.get_fence_status(self.slots[i].fence),
            ) {
                Ok(true) => {
                    self.slots[i].in_flight = false;
                    self.slots[i].identity = None;
                    return Some(i);
                }
                Ok(false) => {}
                Err(error) => {
                    crate::observe::Emit::decline("stats_reduce", &error).fail_once(0);
                }
            }
        }
        None
    }

    /// Record + submit a reduction of `image` (currently `old_layout`) keyed by
    /// `(identity, generation, seq)`. Never waits. Returns whether it armed.
    ///
    /// The caller has already verified the resident is content-ready and BGRA,
    /// and must record the layout this leaves behind
    /// (`SHADER_READ_ONLY_OPTIMAL`) via `registry_set_layout` — the pool cannot
    /// reach the registry.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn arm(
        &mut self,
        ctx: &DeviceContext,
        counters: &EngineCounters,
        cmd_pool: vk::CommandPool,
        pipeline: vk::Pipeline,
        pipeline_layout: vk::PipelineLayout,
        set_layout: vk::DescriptorSetLayout,
        identity: &TargetIdentity,
        generation: u64,
        seq: u64,
        image: vk::Image,
        view: vk::ImageView,
        old_layout: vk::ImageLayout,
        width: u32,
        height: u32,
    ) -> bool {
        if let Err(decline) = validate_arm(seq, width, height) {
            crate::observe::Emit::decline("stats_reduce", &decline)
                .fail_once((u64::from(width) << 32) | u64::from(height));
            return false;
        }
        // These Vulkan-call failures were swallowed by `.is_err()`: a stats
        // reduction that could not build its shared pool/sampler/buffer or grow a
        // slot returned `false` here with no line, so the present proxies went
        // blind (or, for `alloc_slot`, the miss was miscounted as saturation)
        // with nothing in `/tmp/reims-vgpu-fail.log`. Each names its call now, latched
        // per reason because `arm` runs every present.
        if let Err(e) = self.ensure_shared(ctx) {
            crate::observe::Emit::decline("stats_reduce", &e).fail_once(0);
            return false;
        }
        let Some(idx) = self.pick_slot(ctx, cmd_pool) else {
            self.saturated = self.saturated.wrapping_add(1);
            return false;
        };
        {
            let slot = &mut self.slots[idx];
            if let Err(e) = Self::ensure_buffer(slot, ctx, counters) {
                crate::observe::Emit::decline("stats_reduce", &e).fail_once(0);
                return false;
            }
        }
        // Descriptor set: allocate once per slot and rewrite each arm (the image
        // view changes with geometry / identity).
        if self.slots[idx].descriptor_set == vk::DescriptorSet::null() {
            let layouts = [set_layout];
            match ctx.device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(self.desc_pool)
                    .set_layouts(&layouts),
            ) {
                Ok(sets) => self.slots[idx].descriptor_set = sets[0],
                Err(result) => {
                    let error =
                        DrawError::VkCall(VkCall::new(VkOp::StatsAllocDescriptorSet, result));
                    crate::observe::Emit::decline("stats_reduce", &error).fail_once(0);
                    return false;
                }
            }
        }
        let dset = self.slots[idx].descriptor_set;
        let buffer = self.slots[idx].buffer;
        let img_info = [vk::DescriptorImageInfo::default()
            .image_view(view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let smp_info = [vk::DescriptorImageInfo::default().sampler(self.sampler)];
        let buf_info = [vk::DescriptorBufferInfo::default()
            .buffer(buffer)
            .offset(0)
            .range(STATS_BYTES)];
        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(dset)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .image_info(&img_info),
            vk::WriteDescriptorSet::default()
                .dst_set(dset)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::SAMPLER)
                .image_info(&smp_info),
            vk::WriteDescriptorSet::default()
                .dst_set(dset)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&buf_info),
        ];
        ctx.device.update_descriptor_sets(&writes, &[]);

        let slot = &mut self.slots[idx];
        if let Err(error) = stats_call(
            VkOp::StatsResetFence,
            ctx.device.reset_fences(&[slot.fence]),
        ) {
            crate::observe::Emit::decline("stats_reduce", &error).fail_once(0);
            return false;
        }
        let cb = slot.cmd_buf;
        if let Err(error) = stats_call(
            VkOp::StatsResetCommandBuffer,
            ctx.device
                .reset_command_buffer(cb, vk::CommandBufferResetFlags::empty()),
        ) {
            crate::observe::Emit::decline("stats_reduce", &error).fail_once(0);
            return false;
        }
        if let Err(error) = stats_call(
            VkOp::StatsBeginCommandBuffer,
            ctx.device.begin_command_buffer(
                cb,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            ),
        ) {
            crate::observe::Emit::decline("stats_reduce", &error).fail_once(0);
            return false;
        }

        // Zero the accumulators: the shader only ever atomicAdd/atomicMax into
        // them, so a stale block would accumulate across frames.
        ctx.device.cmd_fill_buffer(cb, buffer, 0, STATS_BYTES, 0);
        let fill_barrier = [vk::BufferMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .buffer(buffer)
            .offset(0)
            .size(STATS_BYTES)];
        ctx.device.cmd_pipeline_barrier(
            cb,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &fill_barrier,
            &[],
        );

        // Bring the resident into SHADER_READ_ONLY_OPTIMAL for the compute read.
        // Broad source scope (as `read_target_inner` uses) because the resident
        // may arrive from a colour-attachment write, a transfer, or a prior read.
        if old_layout != vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL {
            let img_barrier = [vk::ImageMemoryBarrier::default()
                .src_access_mask(
                    vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                        | vk::AccessFlags::TRANSFER_WRITE
                        | vk::AccessFlags::SHADER_WRITE,
                )
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .old_layout(old_layout)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(0)
                        .level_count(1)
                        .base_array_layer(0)
                        .layer_count(1),
                )];
            ctx.device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &img_barrier,
            );
        }

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
        // 16x16 workgroups, matching `local_size` in the shader.
        ctx.device
            .cmd_dispatch(cb, width.div_ceil(16), height.div_ceil(16), 1);

        let post = [vk::BufferMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::HOST_READ)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .buffer(buffer)
            .offset(0)
            .size(STATS_BYTES)];
        ctx.device.cmd_pipeline_barrier(
            cb,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(),
            &[],
            &post,
            &[],
        );

        if let Err(error) = stats_call(
            VkOp::StatsEndCommandBuffer,
            ctx.device.end_command_buffer(cb),
        ) {
            crate::observe::Emit::decline("stats_reduce", &error).fail_once(0);
            return false;
        }
        let cbs = [cb];
        let si = vk::SubmitInfo::default().command_buffers(&cbs);
        if let Err(error) = stats_call(
            VkOp::StatsQueueSubmit,
            ctx.device.queue_submit(ctx.queue(), &[si], slot.fence),
        ) {
            crate::observe::Emit::decline("stats_reduce", &error).fail_once(0);
            return false;
        }
        slot.identity = Some(identity.clone());
        slot.generation = generation;
        slot.seq = seq;
        slot.in_flight = true;
        self.arms = self.arms.wrapping_add(1);
        true
    }

    /// Take the reduction for `(identity, seq)` if it has completed.
    ///
    /// Returns `None` when there is no matching armed slot (`miss`) **or** when
    /// its fence has not signalled yet (`not_ready`) — this never blocks, so a
    /// caller on the present path is free to try again next present.
    pub(crate) unsafe fn consume(
        &mut self,
        ctx: &DeviceContext,
        identity: &TargetIdentity,
        generation: u64,
        seq: u64,
    ) -> Option<PresentStats> {
        let idx = self.slots.iter().position(|s| {
            s.in_flight
                && s.seq == seq
                && s.generation == generation
                && s.identity.as_ref() == Some(identity)
        })?;
        match stats_call(
            VkOp::StatsFenceStatusConsume,
            ctx.device.get_fence_status(self.slots[idx].fence),
        ) {
            Ok(true) => {}
            Ok(false) => {
                self.not_ready = self.not_ready.wrapping_add(1);
                return None;
            }
            Err(error) => {
                crate::observe::Emit::decline("stats_reduce", &error).fail_once(0);
                return None;
            }
        }
        let slot = &mut self.slots[idx];
        let mut words = [0u32; STATS_WORDS];
        std::ptr::copy_nonoverlapping(slot.mapped as *const u32, words.as_mut_ptr(), STATS_WORDS);
        slot.in_flight = false;
        slot.identity = None;
        self.hits = self.hits.wrapping_add(1);
        Some(decode_stats(&words))
    }

    /// Blocking [`Self::consume`]: wait the matching slot's fence first.
    ///
    /// For the synchronous store path, which needs this frame's content stats
    /// before it returns — unlike the present proxies, which are happy to pick
    /// up a lagged block on a later drain.
    pub(crate) unsafe fn consume_blocking(
        &mut self,
        ctx: &DeviceContext,
        identity: &TargetIdentity,
        generation: u64,
        seq: u64,
    ) -> Option<PresentStats> {
        let idx = self.slots.iter().position(|s| {
            s.in_flight
                && s.seq == seq
                && s.generation == generation
                && s.identity.as_ref() == Some(identity)
        })?;
        if let Err(error) = stats_call(
            VkOp::StatsWaitFenceBlocking,
            ctx.device
                .wait_for_fences(&[self.slots[idx].fence], true, u64::MAX),
        ) {
            crate::observe::Emit::decline("stats_reduce", &error).fail_once(0);
            return None;
        }
        self.consume(ctx, identity, generation, seq)
    }

    /// Make `seq` unmatchable: the present that armed it is gone. The slot stays
    /// `in_flight` so GPU-safe reclamation still applies.
    pub(crate) fn cancel(&mut self, seq: u64) {
        for s in self.slots.iter_mut() {
            if s.in_flight && s.seq == seq {
                s.identity = None;
            }
        }
    }

    pub(crate) unsafe fn destroy_all(&mut self, device: &ash::Device, cmd_pool: vk::CommandPool) {
        for s in self.slots.iter() {
            if s.in_flight {
                if let Err(error) = stats_call(
                    VkOp::StatsWaitFenceDestroy,
                    device.wait_for_fences(&[s.fence], true, 1_000_000_000),
                ) {
                    crate::observe::Emit::decline("stats_reduce", &error).fail_once(0);
                }
            }
        }
        for s in self.slots.drain(..) {
            if s.buffer != vk::Buffer::null() {
                device.unmap_memory(s.memory);
                device.destroy_buffer(s.buffer, None);
                device.free_memory(s.memory, None);
            }
            device.destroy_fence(s.fence, None);
            device.free_command_buffers(cmd_pool, &[s.cmd_buf]);
        }
        if self.desc_pool != vk::DescriptorPool::null() {
            device.destroy_descriptor_pool(self.desc_pool, None);
            self.desc_pool = vk::DescriptorPool::null();
        }
        if self.sampler != vk::Sampler::null() {
            device.destroy_sampler(self.sampler, None);
            self.sampler = vk::Sampler::null();
        }
    }
}

/// Decode the shader's std430 block into [`PresentStats`].
///
/// Split out so the byte-exactness unit can exercise it without a GPU. The
/// `>> 8` on `edge_sum` happens here, matching `edge_energy_bgra`'s return
/// (the shader accumulates the raw sum).
pub(crate) fn decode_stats(words: &[u32; STATS_WORDS]) -> PresentStats {
    let px = words[5];
    PresentStats {
        byte_nz: words[0],
        byte_max: (words[1] & 0xFF) as u8,
        rgb_nz: words[2],
        max_rgb: (words[3] & 0xFF) as u8,
        px0: [
            (px & 0xFF) as u8,
            ((px >> 8) & 0xFF) as u8,
            ((px >> 16) & 0xFF) as u8,
            ((px >> 24) & 0xFF) as u8,
        ],
        edge_energy: (words[4] >> 8),
        width: words[6],
        height: words[7],
        alpha_nz: words[8],
        alpha_opaque: words[9],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observe::Decline as _;

    #[test]
    fn invalid_stats_arms_are_typed_while_valid_and_saturation_stay_control_flow() {
        assert_eq!(validate_arm(1, 64, 32), Ok(()));
        let zero_seq = validate_arm(0, 64, 32).unwrap_err();
        assert_eq!(zero_seq.slug(), "vk_stats_reduce_zero_sequence");
        assert!(zero_seq.fields().is_empty());
        let zero_geometry = validate_arm(1, 0, 32).unwrap_err();
        assert_eq!(zero_geometry.slug(), "vk_stats_reduce_zero_geometry");
        assert_eq!(
            zero_geometry.fields(),
            vec![("width", "0".into()), ("height", "32".into())]
        );
    }

    #[test]
    fn every_stats_runtime_vk_failure_preserves_its_operation() {
        use crate::observe::Decline as _;
        let cases = [
            (
                VkOp::StatsFenceStatusReclaim,
                "vk_stats_fence_status_reclaim",
            ),
            (
                VkOp::StatsAllocDescriptorSet,
                "vk_stats_alloc_descriptor_set",
            ),
            (VkOp::StatsResetFence, "vk_stats_reset_fence"),
            (
                VkOp::StatsResetCommandBuffer,
                "vk_stats_reset_command_buffer",
            ),
            (
                VkOp::StatsBeginCommandBuffer,
                "vk_stats_begin_command_buffer",
            ),
            (VkOp::StatsEndCommandBuffer, "vk_stats_end_command_buffer"),
            (VkOp::StatsQueueSubmit, "vk_stats_queue_submit"),
            (
                VkOp::StatsFenceStatusConsume,
                "vk_stats_fence_status_consume",
            ),
            (VkOp::StatsWaitFenceBlocking, "vk_stats_wait_fence_blocking"),
            (VkOp::StatsWaitFenceDestroy, "vk_stats_wait_fence_destroy"),
        ];
        for (op, slug) in cases {
            let error = stats_call::<()>(op, Err(vk::Result::ERROR_DEVICE_LOST))
                .expect_err("synthetic stats failure must decline");
            assert_eq!(error.slug(), slug);
        }
    }

    /// The decode must mirror the GLSL block layout exactly: field order, the
    /// `px0` byte packing (`B | G<<8 | R<<16 | A<<24`, i.e. memory order), and
    /// the `>> 8` that turns the shader's raw edge sum into `edge_energy_bgra`'s
    /// return value.
    #[test]
    fn decode_stats_matches_shader_block_layout() {
        let words = [
            1234u32,        // byte_nz
            0xFF,           // byte_max
            567,            // rgb_nz
            0x80,           // max_rgb
            (9000u32) << 8, // edge_sum: decodes to exactly 9000 after >> 8
            0x44_33_22_11,  // px0: B=0x11 G=0x22 R=0x33 A=0x44
            1920,
            1080,
            4242, // alpha_nz
            777,  // alpha_opaque
        ];
        let s = decode_stats(&words);
        assert_eq!(s.byte_nz, 1234);
        assert_eq!(s.byte_max, 0xFF);
        assert_eq!(s.rgb_nz, 567);
        assert_eq!(s.max_rgb, 0x80);
        assert_eq!(s.edge_energy, 9000);
        assert_eq!(s.px0, [0x11, 0x22, 0x33, 0x44], "px0 is memory order BGRA");
        assert_eq!((s.width, s.height), (1920, 1080));
        assert_eq!((s.alpha_nz, s.alpha_opaque), (4242, 777));

        // The scatter path consumes these as `Color8ContentStats`; `max_rgb`
        // must land in `rgb_max` (same quantity, different field name).
        let c = super::super::content_stats::Color8ContentStats::from(s);
        assert_eq!(
            (c.rgb_nz, c.rgb_max, c.alpha_nz, c.alpha_opaque),
            (567, 0x80, 4242, 777)
        );
    }

    /// The embedded SPIR-V must be what `shaders/present_stats.comp` compiles
    /// to. Without this the array silently drifts from the GLSL it documents.
    /// Skips when `glslc` is unavailable, matching the repo's other shader tests.
    #[test]
    fn present_stats_spirv_matches_shader_source() {
        use std::path::PathBuf;
        use std::process::Command;

        let comp = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("shaders/present_stats.comp");
        let spv = std::env::temp_dir().join(format!(
            "reims_vgpu_present_stats_check_{}.spv",
            std::process::id()
        ));
        let status = Command::new("glslc")
            .args([
                "-fshader-stage=comp",
                comp.to_str().unwrap(),
                "-o",
                spv.to_str().unwrap(),
            ])
            .status();
        if !matches!(status, Ok(s) if s.success()) {
            eprintln!("SKIP present_stats_spirv_matches_shader_source: no glslc");
            return;
        }
        let bytes = std::fs::read(&spv).expect("read compiled spv");
        let _ = std::fs::remove_file(&spv);
        let fresh: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let embedded = super::super::present_stats_spv::PRESENT_STATS_SPIRV;
        assert_eq!(
            fresh.len(),
            embedded.len(),
            "embedded SPIR-V is {} words, shader compiles to {}; refresh \
             present_stats_spv.rs from shaders/present_stats.comp",
            embedded.len(),
            fresh.len()
        );
        assert!(
            fresh == embedded,
            "embedded SPIR-V differs from shaders/present_stats.comp"
        );
    }
}
