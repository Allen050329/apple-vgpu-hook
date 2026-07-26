//! Typed Vulkan-call failures for the engine.
//!
//! # Why this is not `DrawError::Vulkan(String)`
//!
//! The engine's fallible ash calls used to fail as
//! `DrawError::Vulkan(format!("<operation>: {e}"))` — free-text prose rendered
//! through the coarse `vk_engine_vk_untyped` slug (see
//! [`super::types::DrawError`]). With 148 such sites across eleven engine files,
//! a grep of `/tmp/reims-vgpu-fail.log` could not say which of the ~113 distinct
//! fallible Vulkan calls the crate makes had actually refused: `create buffer`
//! failed on six rails, and `reason=vk_engine_vk_untyped` named none of them.
//!
//! [`VkCall`] types one call, carrying the driver's raw [`vk::Result`] and a
//! [`VkOp`] identifying the *(rail, operation)* pair. Its slug is
//! `vk_<rail>_<operation>`, so each call names its own reason.
//!
//! # Why (rail, operation), not just operation
//!
//! Collapsing to a bare operation would lose what the prose already carried: six
//! rails create a buffer, and a shared `vk_create_buffer` could not tell the
//! stats-reduction pool's buffer from the descriptor pool's. So the vocabulary
//! is one variant per call the crate actually makes, disambiguated by the
//! enclosing subsystem. The generic operations (`queue_submit`,
//! `reset_command_buffer`, …) appear on several rails and are split the same way.
//!
//! # Grown file by file
//!
//! [`VkOp`] grows as engine files move their `DrawError::Vulkan(String)` sites
//! to typed operations. `Vulkan(String)` should shrink while `VkOp` grows in
//! lockstep, one reviewable commit per file.

use ash::vk;

use crate::observe::Decline;

/// One fallible Vulkan operation the engine makes, named by *(rail, operation)*.
///
/// Each variant maps to exactly one call site; the rail prefix is the enclosing
/// subsystem so a call spelled the same way on two rails (e.g. `create buffer`)
/// stays distinguishable in the log.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VkOp {
    // ---- stats_reduce.rs — the present-proxy GPU stats-reduction pool ----
    /// `vkCreateDescriptorPool` for the private stats descriptor pool.
    StatsDescPool,
    /// `vkCreateSampler` for the point sampler `texelFetch` binds.
    StatsSampler,
    /// `vkCreateBuffer` for a slot's 32-byte readback buffer.
    StatsCreateBuffer,
    /// `vkAllocateMemory` for that buffer.
    StatsAlloc,
    /// `vkBindBufferMemory` for that buffer.
    StatsBind,
    /// `vkMapMemory` for that buffer's persistent host-coherent map.
    StatsMap,
    /// `vkAllocateCommandBuffers` for a slot's command buffer.
    StatsAllocCb,
    /// `vkCreateFence` for a slot's completion fence.
    StatsCreateFence,
    /// `vkGetFenceStatus` while reclaiming a saturated slot.
    StatsFenceStatusReclaim,
    /// `vkAllocateDescriptorSets` for a stats slot.
    StatsAllocDescriptorSet,
    /// `vkResetFences` before one stats dispatch.
    StatsResetFence,
    /// `vkResetCommandBuffer` before one stats dispatch.
    StatsResetCommandBuffer,
    /// `vkBeginCommandBuffer` for one stats dispatch.
    StatsBeginCommandBuffer,
    /// `vkEndCommandBuffer` after one stats dispatch.
    StatsEndCommandBuffer,
    /// `vkQueueSubmit` of one stats dispatch.
    StatsQueueSubmit,
    /// `vkGetFenceStatus` while non-blockingly consuming a stats block.
    StatsFenceStatusConsume,
    /// `vkWaitForFences` for the synchronous Store stats consumer.
    StatsWaitFenceBlocking,
    /// Bounded `vkWaitForFences` before destroying an in-flight stats slot.
    StatsWaitFenceDestroy,

    // ---- mod.rs `read_target_inner` — the full-frame CPU readback rail ----
    /// `vkResetCommandBuffer` before recording the readback copy.
    ReadbackResetCb,
    /// `vkBeginCommandBuffer` for the readback copy.
    ReadbackBeginCb,
    /// `vkEndCommandBuffer` closing the readback copy.
    ReadbackEndCb,
    /// `vkQueueSubmit` of the readback copy.
    ReadbackSubmit,
    /// `vkMapMemory` of the readback buffer to copy the pixels out.
    ReadbackMap,

    // ---- mod.rs `present_into_host_ptr_strided` — the packed-contig host
    //      present rail (guest page imported, GPU DMA straight into it) ----
    /// `vkResetCommandBuffer` before recording the host-ptr present copy.
    HostPresentResetCb,
    /// `vkBeginCommandBuffer` for the host-ptr present copy.
    HostPresentBeginCb,
    /// `vkEndCommandBuffer` closing the host-ptr present copy.
    HostPresentEndCb,
    /// `vkQueueSubmit` of the host-ptr present copy.
    HostPresentSubmit,

    // ---- mod.rs `read_resident_storage` — the pinned deferred-writeback
    //      storage-image flush rail (GPU→host tight copy, then unpin) ----
    /// `vkResetCommandBuffer` before recording the storage flush copy.
    StorageReadResetCb,
    /// `vkBeginCommandBuffer` for the storage flush copy.
    StorageReadBeginCb,
    /// `vkEndCommandBuffer` closing the storage flush copy.
    StorageReadEndCb,
    /// `vkQueueSubmit` of the storage flush copy.
    StorageReadSubmit,
    /// `vkMapMemory` of the storage readback buffer to copy the bytes out.
    StorageReadMap,

    // ---- mod.rs `export_scanout_from_bgra` — the CPU-capture → dmabuf export
    //      scanout rail (staging copy of `frame_bgra`, then blit + export) ----
    /// `vkMapMemory` of the staging buffer to upload the captured BGRA.
    ExportScanoutMapStaging,
    /// `vkResetCommandBuffer` before recording the scanout export blit.
    ExportScanoutResetCb,
    /// `vkBeginCommandBuffer` for the scanout export blit.
    ExportScanoutBeginCb,
    /// `vkEndCommandBuffer` closing the scanout export blit.
    ExportScanoutEndCb,
    /// `vkQueueSubmit` of the scanout export blit.
    ExportScanoutSubmit,

    // ---- mod.rs `export_present_from_resident_composited_fd_policy` — the
    //      zero-copy resident → dmabuf present export rail ----
    /// `vkResetCommandBuffer` before recording the present export blit.
    ExportPresentResetCb,
    /// `vkBeginCommandBuffer` for the present export blit.
    ExportPresentBeginCb,
    /// `vkEndCommandBuffer` closing the present export blit.
    ExportPresentEndCb,
    /// `vkQueueSubmit` of the present export blit.
    ExportPresentSubmit,

    // ---- dmabuf_export.rs `export_bgra_scanout_dmabuf` — the low-level dmabuf
    //      scanout image export (the exportable VkImage the two mod.rs export
    //      rails above are built on), create + alloc + bind + get_fd ----
    /// `vkCreateImage` for the exportable LINEAR scanout image.
    DmabufExportCreateImage,
    /// `vkAllocateMemory` for that image (device-local preferred, exportable).
    DmabufExportAlloc,
    /// `vkBindImageMemory` binding that memory to the image.
    DmabufExportBind,
    /// `vkGetMemoryFdKHR` exporting the dmabuf fd for that memory.
    DmabufExportGetFd,

    // ---- dmabuf_export.rs `import_bgra_dmabuf_image` — the consumer half: the
    //      host window imports the engine's exported dmabuf fd as a sampleable
    //      `VkImage` for its zero-copy present blit, create + fd-props + alloc +
    //      bind. Sibling of the export ops above, on the other device. ----
    /// `vkCreateImage` for the LINEAR image backing the imported dmabuf.
    DmabufImportCreateImage,
    /// `vkGetMemoryFdPropertiesKHR` for the imported fd's allowed memory types.
    DmabufImportFdProps,
    /// `vkAllocateMemory` importing the dmabuf fd as dedicated device memory.
    DmabufImportAlloc,
    /// `vkBindImageMemory` binding the imported memory to the image.
    DmabufImportBind,

    // ---- caches.rs `ObjectCaches` — the L2–L7 immutable-object create caches.
    //      Each op is cached negatively as a `VkCall`, so the cheap re-attempt
    //      replays the same typed reason rather than a re-formatted string ----
    /// `vkCreateShaderModule` for a translated SPIR-V shader.
    CachesCreateShaderModule,
    /// `vkCreateDescriptorSetLayout` for a pipeline's descriptor layout.
    CachesCreateDescriptorSetLayout,
    /// `vkCreatePipelineLayout` for a pipeline's layout.
    CachesCreatePipelineLayout,
    /// `vkCreateRenderPass` for a target's render pass.
    CachesCreateRenderPass,
    /// `vkCreateSampler` for a resolved sampler state.
    CachesCreateSampler,
    /// `vkCreateGraphicsPipelines` for a draw pipeline.
    CachesCreateGraphicsPipelines,
    /// `vkCreateComputePipelines` for a dispatch pipeline.
    CachesCreateComputePipelines,

    // ---- context.rs — the guest host-pointer import rail (VK_EXT_external_
    //      memory_host: address the guest RAM window in place) ----
    /// `vkGetMemoryHostPointerPropertiesEXT` for an imported host pointer.
    ContextHostPtrProps,
    /// `vkAllocateMemory` importing that host pointer as device memory.
    ContextImportHostPtrAlloc,
    /// `vkGetPipelineCacheData` before persisting a grown pipeline cache.
    ContextPipelineCacheGetData,

    // ---- desc_arena.rs — the per-frame descriptor-set arena ----
    /// `vkCreateDescriptorPool` for the arena's pool.
    DescArenaCreatePool,
    /// `vkAllocateDescriptorSets` from the arena.
    DescArenaAllocSets,
    /// `vkAllocateDescriptorSets` after growing the arena.
    DescArenaAllocSetsGrown,
    /// `vkFreeDescriptorSets` returning one retired batch to its owning block.
    DescArenaFreeSets,

    // ---- exec.rs — the draw command-buffer record/submit/readback rail ----
    /// `vkResetCommandBuffer` before recording the draw batch.
    ExecResetCb,
    /// `vkBeginCommandBuffer` for the draw batch.
    ExecBeginCb,
    /// `vkEndCommandBuffer` closing the draw batch.
    ExecEndCb,
    /// `vkQueueSubmit` of the draw batch.
    ExecSubmit,
    /// `vkMapMemory` of the draw readback buffer.
    ExecMapReadback,

    // ---- exec_compute.rs — the compute command-buffer record/submit/readback
    //      rail (a distinct queue submission from the draw rail above) ----
    /// `vkResetCommandBuffer` before recording the dispatch.
    ComputeExecResetCb,
    /// `vkBeginCommandBuffer` for the dispatch.
    ComputeExecBeginCb,
    /// `vkEndCommandBuffer` closing the dispatch.
    ComputeExecEndCb,
    /// `vkQueueSubmit` of the dispatch.
    ComputeExecSubmit,
    /// `vkMapMemory` of the storage-buffer readback after the dispatch.
    ComputeExecMapStorageReadback,
    /// `vkMapMemory` of the storage-image readback after the dispatch.
    ComputeExecMapImageReadback,
    /// `vkCreateBuffer` over imported guest memory for direct compute writeback.
    ComputeDirectWritebackCreateBuffer,
    /// `vkBindBufferMemory` over imported guest memory for direct compute writeback.
    ComputeDirectWritebackBindBuffer,

    // ---- host_scatter.rs — the GPU-direct resident → guest-page Store rail ----
    /// `vkAllocateCommandBuffers` for the private scatter command buffer.
    HostScatterAllocCommandBuffer,
    /// `vkCreateFence` for synchronous scatter completion.
    HostScatterCreateFence,
    /// `vkResetFences` before one scatter submission.
    HostScatterResetFence,
    /// `vkResetCommandBuffer` before recording one scatter.
    HostScatterResetCommandBuffer,
    /// `vkBeginCommandBuffer` for one scatter.
    HostScatterBeginCommandBuffer,
    /// `vkEndCommandBuffer` after recording one scatter.
    HostScatterEndCommandBuffer,
    /// `vkQueueSubmit` of one scatter.
    HostScatterQueueSubmit,
    /// `vkWaitForFences` that makes the guest-page write synchronous.
    HostScatterWaitFence,

    // ---- mod.rs guest reset — quiesce before dropping guest-owned state ----
    /// `vkDeviceWaitIdle` before a guest reset destroys device-derived objects.
    GuestResetDeviceWaitIdle,

    // ---- slab.rs — shared DEVICE_LOCAL optimal-image memory ----
    /// `vkAllocateMemory` for a new shared or dedicated image slab block.
    SlabAllocateMemory,

    // ---- pools/mod.rs — the resource pools: command-buffer/fence machinery,
    //      the batch submit, and the staging + readback buffer rails ----
    /// `vkCreateCommandPool` for the engine's command pool.
    PoolsCreateCommandPool,
    /// `vkAllocateCommandBuffers` for the pool's slots.
    PoolsAllocCommandBuffers,
    /// `vkCreateFence` for a slot's completion fence.
    PoolsCreateFence,
    /// `vkWaitForFences` retiring a slot (non-timeout, non-device-lost result).
    PoolsWaitFencesRetire,
    /// `vkGetFenceStatus` when beginning a batch entry.
    PoolsFenceStatusBeginEntry,
    /// `vkWaitForFences` draining an in-flight batch entry.
    PoolsWaitFencesEntry,
    /// Bounded `vkWaitForFences` before destroying in-flight pool slots.
    PoolsWaitFencesDestroy,
    /// `vkResetFences` retiring a slot.
    PoolsResetFencesRetire,
    /// `vkEndCommandBuffer` closing a submit batch.
    PoolsEndCbBatch,
    /// `vkQueueSubmit` of a submit batch.
    PoolsSubmitBatch,
    /// `vkCreateBuffer` over a cached host-import window.
    PoolsHostImportCreateBuffer,
    /// `vkBindBufferMemory` over a cached host-import window.
    PoolsHostImportBindBuffer,
    /// `vkCreateBuffer` for a staging slot.
    PoolsCreateStaging,
    /// `vkAllocateMemory` for a staging slot.
    PoolsAllocStaging,
    /// `vkBindBufferMemory` for a staging slot.
    PoolsBindStaging,
    /// `vkMapMemory` of a staging slot to upload guest bytes.
    PoolsMapStaging,
    /// `vkCreateBuffer` for a readback slot.
    PoolsCreateReadback,
    /// `vkAllocateMemory` for a readback slot.
    PoolsAllocReadback,
    /// `vkBindBufferMemory` for a readback slot.
    PoolsBindReadback,
    /// `vkCreateBuffer` for the secondary ("extra") readback slot.
    PoolsCreateReadbackExtra,
    /// `vkAllocateMemory` for the extra readback slot.
    PoolsAllocReadbackExtra,
    /// `vkBindBufferMemory` for the extra readback slot.
    PoolsBindReadbackExtra,

    // ---- pools/mod.rs — the target / sampled / storage / registry / MRT-
    //      secondary / depth image + view + framebuffer allocation rails ----
    /// `vkCreateImage` for a render-target color image.
    PoolsCreateTargetImage,
    /// `vkBindImageMemory` for a render-target image.
    PoolsBindTarget,
    /// `vkCreateImageView` for a render-target image.
    PoolsCreateTargetView,
    /// `vkCreateFramebuffer` for a render target.
    PoolsCreateFramebuffer,
    /// `vkCreateImage` for a sampled (texture) image.
    PoolsCreateSampledImage,
    /// `vkBindImageMemory` for a sampled image.
    PoolsBindSampled,
    /// `vkCreateImageView` for a sampled image.
    PoolsCreateSampledView,
    /// `vkCreateImage` for a compute storage image.
    PoolsCreateStorageImage,
    /// `vkAllocateMemory` for a storage image.
    PoolsAllocStorageImage,
    /// `vkBindImageMemory` for a storage image.
    PoolsBindStorageImage,
    /// `vkCreateImageView` for a storage image.
    PoolsCreateStorageImageView,
    /// `vkCreateFramebuffer` for a resident-registry target.
    PoolsCreateRegistryFramebuffer,
    /// `vkCreateImage` for a resident-registry target.
    PoolsCreateRegistryTarget,
    /// `vkBindImageMemory` for a resident-registry target.
    PoolsBindRegistryTarget,
    /// `vkCreateImageView` for a resident-registry target.
    PoolsCreateRegistryView,
    /// `vkCreateImage` for a secondary (MRT slot 1..) color image.
    PoolsCreateMrtSecondaryTarget,
    /// `vkAllocateMemory` for a secondary MRT image.
    PoolsAllocMrtSecondary,
    /// `vkBindImageMemory` for a secondary MRT image.
    PoolsBindMrtSecondary,
    /// `vkCreateImageView` for a secondary MRT image.
    PoolsCreateMrtSecondaryView,
    /// `vkCreateImage` for a depth attachment.
    PoolsCreateDepthImage,
    /// `vkAllocateMemory` for a depth image.
    PoolsAllocDepth,
    /// `vkBindImageMemory` for a depth image.
    PoolsBindDepth,
    /// `vkCreateImageView` for a depth image.
    PoolsCreateDepthView,
    /// `vkCreateFramebuffer` for an MRT (multi-attachment) render target.
    PoolsCreateMrtFramebuffer,

    // ---- window_present.rs — the macOS host-window MoltenVK swapchain
    //      presenter: surface + swapchain bring-up and the per-present
    //      acquire/blit/submit/present. The capability-selection refusals on
    //      this rail (no present-capable queue, no surface format, no composite
    //      alpha, …) are already `DrawReason`; these are its raw `vk::Result`
    //      calls. The image-available and render-finished semaphores are split
    //      (distinct objects, distinct fixes), but the same create call on the
    //      initial-bring-up and swapchain-recreate paths shares one op. ----
    /// `ash_window::create_surface` for the host window.
    WindowCreateSurface,
    /// `vkGetPhysicalDeviceSurfaceSupportKHR` for the present queue family.
    WindowSurfaceSupport,
    /// `vkCreateCommandPool` for the presenter's blit command pool.
    WindowCreateCommandPool,
    /// `vkAllocateCommandBuffers` for the presenter's blit command buffer.
    WindowAllocCommandBuffer,
    /// `vkCreateSemaphore` for the image-available (acquire) semaphore.
    WindowCreateAcquireSemaphore,
    /// `vkCreateSemaphore` for the render-finished (present) semaphore.
    WindowCreateRenderSemaphore,
    /// `vkCreateFence` for the presenter's in-flight fence.
    WindowCreateFence,
    /// `vkGetFenceStatus` retiring the previous present's in-flight fence.
    WindowFenceStatus,
    /// `vkQueueWaitIdle` before recreating the swapchain.
    WindowQueueWaitIdle,
    /// `vkGetPhysicalDeviceSurfaceCapabilitiesKHR` sizing the swapchain.
    WindowSurfaceCaps,
    /// `vkGetPhysicalDeviceSurfaceFormatsKHR` choosing the swapchain format.
    WindowSurfaceFormats,
    /// `vkCreateSwapchainKHR` for the host-window swapchain.
    WindowCreateSwapchain,
    /// `vkGetSwapchainImagesKHR` for the swapchain's images.
    WindowGetSwapchainImages,
    /// `vkAcquireNextImageKHR` (non-`NOT_READY`/`TIMEOUT`/`OUT_OF_DATE` result).
    WindowAcquireImage,
    /// `vkResetFences` on the in-flight fence before recording the present blit.
    WindowResetFence,
    /// `vkResetCommandBuffer` before recording the present blit.
    WindowResetCommandBuffer,
    /// `vkBeginCommandBuffer` for the present blit.
    WindowBeginCommandBuffer,
    /// `vkEndCommandBuffer` closing the present blit.
    WindowEndCommandBuffer,
    /// `vkQueueSubmit` of the present blit.
    WindowSubmitPresent,
    /// `vkQueuePresentKHR` (non-`OUT_OF_DATE` result) presenting the image.
    WindowQueuePresent,
    /// `vkQueueWaitIdle` before destroying the window presenter.
    WindowDestroyQueueWaitIdle,
}

/// A failed Vulkan call: which operation refused, and the driver's `vk::Result`.
///
/// Carried by [`super::types::DrawError::VkCall`], which delegates its slug and
/// fields here so one event has one name at every layer — the same rule
/// [`super::reason::DrawReason::VertexFormat`] follows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VkCall {
    pub op: VkOp,
    pub result: vk::Result,
}

impl VkCall {
    pub fn new(op: VkOp, result: vk::Result) -> Self {
        Self { op, result }
    }
}

impl Decline for VkCall {
    /// One slug per distinct call, `vk_<rail>_<operation>`. The match is here
    /// rather than on [`VkOp`] so the crate-wide census reads the literals from
    /// this type's own `Decline` impl.
    fn slug(&self) -> &'static str {
        match self.op {
            VkOp::StatsDescPool => "vk_stats_desc_pool",
            VkOp::StatsSampler => "vk_stats_sampler",
            VkOp::StatsCreateBuffer => "vk_stats_create_buffer",
            VkOp::StatsAlloc => "vk_stats_alloc",
            VkOp::StatsBind => "vk_stats_bind",
            VkOp::StatsMap => "vk_stats_map",
            VkOp::StatsAllocCb => "vk_stats_alloc_cb",
            VkOp::StatsCreateFence => "vk_stats_create_fence",
            VkOp::StatsFenceStatusReclaim => "vk_stats_fence_status_reclaim",
            VkOp::StatsAllocDescriptorSet => "vk_stats_alloc_descriptor_set",
            VkOp::StatsResetFence => "vk_stats_reset_fence",
            VkOp::StatsResetCommandBuffer => "vk_stats_reset_command_buffer",
            VkOp::StatsBeginCommandBuffer => "vk_stats_begin_command_buffer",
            VkOp::StatsEndCommandBuffer => "vk_stats_end_command_buffer",
            VkOp::StatsQueueSubmit => "vk_stats_queue_submit",
            VkOp::StatsFenceStatusConsume => "vk_stats_fence_status_consume",
            VkOp::StatsWaitFenceBlocking => "vk_stats_wait_fence_blocking",
            VkOp::StatsWaitFenceDestroy => "vk_stats_wait_fence_destroy",

            VkOp::ReadbackResetCb => "vk_readback_reset_cb",
            VkOp::ReadbackBeginCb => "vk_readback_begin_cb",
            VkOp::ReadbackEndCb => "vk_readback_end_cb",
            VkOp::ReadbackSubmit => "vk_readback_submit",
            VkOp::ReadbackMap => "vk_readback_map",

            VkOp::HostPresentResetCb => "vk_host_present_reset_cb",
            VkOp::HostPresentBeginCb => "vk_host_present_begin_cb",
            VkOp::HostPresentEndCb => "vk_host_present_end_cb",
            VkOp::HostPresentSubmit => "vk_host_present_submit",

            VkOp::StorageReadResetCb => "vk_storage_read_reset_cb",
            VkOp::StorageReadBeginCb => "vk_storage_read_begin_cb",
            VkOp::StorageReadEndCb => "vk_storage_read_end_cb",
            VkOp::StorageReadSubmit => "vk_storage_read_submit",
            VkOp::StorageReadMap => "vk_storage_read_map",

            VkOp::ExportScanoutMapStaging => "vk_export_scanout_map_staging",
            VkOp::ExportScanoutResetCb => "vk_export_scanout_reset_cb",
            VkOp::ExportScanoutBeginCb => "vk_export_scanout_begin_cb",
            VkOp::ExportScanoutEndCb => "vk_export_scanout_end_cb",
            VkOp::ExportScanoutSubmit => "vk_export_scanout_submit",

            VkOp::ExportPresentResetCb => "vk_export_present_reset_cb",
            VkOp::ExportPresentBeginCb => "vk_export_present_begin_cb",
            VkOp::ExportPresentEndCb => "vk_export_present_end_cb",
            VkOp::ExportPresentSubmit => "vk_export_present_submit",

            VkOp::DmabufExportCreateImage => "vk_dmabuf_export_create_image",
            VkOp::DmabufExportAlloc => "vk_dmabuf_export_alloc",
            VkOp::DmabufExportBind => "vk_dmabuf_export_bind",
            VkOp::DmabufExportGetFd => "vk_dmabuf_export_get_fd",

            VkOp::DmabufImportCreateImage => "vk_dmabuf_import_create_image",
            VkOp::DmabufImportFdProps => "vk_dmabuf_import_fd_props",
            VkOp::DmabufImportAlloc => "vk_dmabuf_import_alloc",
            VkOp::DmabufImportBind => "vk_dmabuf_import_bind",

            VkOp::CachesCreateShaderModule => "vk_caches_create_shader_module",
            VkOp::CachesCreateDescriptorSetLayout => "vk_caches_create_descriptor_set_layout",
            VkOp::CachesCreatePipelineLayout => "vk_caches_create_pipeline_layout",
            VkOp::CachesCreateRenderPass => "vk_caches_create_render_pass",
            VkOp::CachesCreateSampler => "vk_caches_create_sampler",
            VkOp::CachesCreateGraphicsPipelines => "vk_caches_create_graphics_pipelines",
            VkOp::CachesCreateComputePipelines => "vk_caches_create_compute_pipelines",

            VkOp::ContextHostPtrProps => "vk_context_host_ptr_props",
            VkOp::ContextImportHostPtrAlloc => "vk_context_import_host_ptr_alloc",
            VkOp::ContextPipelineCacheGetData => "vk_context_pipeline_cache_get_data",

            VkOp::DescArenaCreatePool => "vk_desc_arena_create_pool",
            VkOp::DescArenaAllocSets => "vk_desc_arena_alloc_sets",
            VkOp::DescArenaAllocSetsGrown => "vk_desc_arena_alloc_sets_grown",
            VkOp::DescArenaFreeSets => "vk_desc_arena_free_sets",

            VkOp::ExecResetCb => "vk_exec_reset_cb",
            VkOp::ExecBeginCb => "vk_exec_begin_cb",
            VkOp::ExecEndCb => "vk_exec_end_cb",
            VkOp::ExecSubmit => "vk_exec_submit",
            VkOp::ExecMapReadback => "vk_exec_map_readback",

            VkOp::ComputeExecResetCb => "vk_compute_exec_reset_cb",
            VkOp::ComputeExecBeginCb => "vk_compute_exec_begin_cb",
            VkOp::ComputeExecEndCb => "vk_compute_exec_end_cb",
            VkOp::ComputeExecSubmit => "vk_compute_exec_submit",
            VkOp::ComputeExecMapStorageReadback => "vk_compute_exec_map_storage_readback",
            VkOp::ComputeExecMapImageReadback => "vk_compute_exec_map_image_readback",
            VkOp::ComputeDirectWritebackCreateBuffer => "vk_compute_direct_writeback_create_buffer",
            VkOp::ComputeDirectWritebackBindBuffer => "vk_compute_direct_writeback_bind_buffer",

            VkOp::HostScatterAllocCommandBuffer => "vk_host_scatter_alloc_command_buffer",
            VkOp::HostScatterCreateFence => "vk_host_scatter_create_fence",
            VkOp::HostScatterResetFence => "vk_host_scatter_reset_fence",
            VkOp::HostScatterResetCommandBuffer => "vk_host_scatter_reset_command_buffer",
            VkOp::HostScatterBeginCommandBuffer => "vk_host_scatter_begin_command_buffer",
            VkOp::HostScatterEndCommandBuffer => "vk_host_scatter_end_command_buffer",
            VkOp::HostScatterQueueSubmit => "vk_host_scatter_queue_submit",
            VkOp::HostScatterWaitFence => "vk_host_scatter_wait_fence",

            VkOp::GuestResetDeviceWaitIdle => "vk_guest_reset_device_wait_idle",

            VkOp::SlabAllocateMemory => "vk_slab_allocate_memory",

            VkOp::PoolsCreateCommandPool => "vk_pools_create_command_pool",
            VkOp::PoolsAllocCommandBuffers => "vk_pools_alloc_command_buffers",
            VkOp::PoolsCreateFence => "vk_pools_create_fence",
            VkOp::PoolsWaitFencesRetire => "vk_pools_wait_fences_retire",
            VkOp::PoolsFenceStatusBeginEntry => "vk_pools_fence_status_begin_entry",
            VkOp::PoolsWaitFencesEntry => "vk_pools_wait_fences_entry",
            VkOp::PoolsWaitFencesDestroy => "vk_pools_wait_fences_destroy",
            VkOp::PoolsResetFencesRetire => "vk_pools_reset_fences_retire",
            VkOp::PoolsEndCbBatch => "vk_pools_end_cb_batch",
            VkOp::PoolsSubmitBatch => "vk_pools_submit_batch",
            VkOp::PoolsHostImportCreateBuffer => "vk_pools_host_import_create_buffer",
            VkOp::PoolsHostImportBindBuffer => "vk_pools_host_import_bind_buffer",
            VkOp::PoolsCreateStaging => "vk_pools_create_staging",
            VkOp::PoolsAllocStaging => "vk_pools_alloc_staging",
            VkOp::PoolsBindStaging => "vk_pools_bind_staging",
            VkOp::PoolsMapStaging => "vk_pools_map_staging",
            VkOp::PoolsCreateReadback => "vk_pools_create_readback",
            VkOp::PoolsAllocReadback => "vk_pools_alloc_readback",
            VkOp::PoolsBindReadback => "vk_pools_bind_readback",
            VkOp::PoolsCreateReadbackExtra => "vk_pools_create_readback_extra",
            VkOp::PoolsAllocReadbackExtra => "vk_pools_alloc_readback_extra",
            VkOp::PoolsBindReadbackExtra => "vk_pools_bind_readback_extra",

            VkOp::PoolsCreateTargetImage => "vk_pools_create_target_image",
            VkOp::PoolsBindTarget => "vk_pools_bind_target",
            VkOp::PoolsCreateTargetView => "vk_pools_create_target_view",
            VkOp::PoolsCreateFramebuffer => "vk_pools_create_framebuffer",
            VkOp::PoolsCreateSampledImage => "vk_pools_create_sampled_image",
            VkOp::PoolsBindSampled => "vk_pools_bind_sampled",
            VkOp::PoolsCreateSampledView => "vk_pools_create_sampled_view",
            VkOp::PoolsCreateStorageImage => "vk_pools_create_storage_image",
            VkOp::PoolsAllocStorageImage => "vk_pools_alloc_storage_image",
            VkOp::PoolsBindStorageImage => "vk_pools_bind_storage_image",
            VkOp::PoolsCreateStorageImageView => "vk_pools_create_storage_image_view",
            VkOp::PoolsCreateRegistryFramebuffer => "vk_pools_create_registry_framebuffer",
            VkOp::PoolsCreateRegistryTarget => "vk_pools_create_registry_target",
            VkOp::PoolsBindRegistryTarget => "vk_pools_bind_registry_target",
            VkOp::PoolsCreateRegistryView => "vk_pools_create_registry_view",
            VkOp::PoolsCreateMrtSecondaryTarget => "vk_pools_create_mrt_secondary_target",
            VkOp::PoolsAllocMrtSecondary => "vk_pools_alloc_mrt_secondary",
            VkOp::PoolsBindMrtSecondary => "vk_pools_bind_mrt_secondary",
            VkOp::PoolsCreateMrtSecondaryView => "vk_pools_create_mrt_secondary_view",
            VkOp::PoolsCreateDepthImage => "vk_pools_create_depth_image",
            VkOp::PoolsAllocDepth => "vk_pools_alloc_depth",
            VkOp::PoolsBindDepth => "vk_pools_bind_depth",
            VkOp::PoolsCreateDepthView => "vk_pools_create_depth_view",
            VkOp::PoolsCreateMrtFramebuffer => "vk_pools_create_mrt_framebuffer",

            VkOp::WindowCreateSurface => "vk_window_create_surface",
            VkOp::WindowSurfaceSupport => "vk_window_surface_support",
            VkOp::WindowCreateCommandPool => "vk_window_create_command_pool",
            VkOp::WindowAllocCommandBuffer => "vk_window_alloc_command_buffer",
            VkOp::WindowCreateAcquireSemaphore => "vk_window_create_acquire_semaphore",
            VkOp::WindowCreateRenderSemaphore => "vk_window_create_render_semaphore",
            VkOp::WindowCreateFence => "vk_window_create_fence",
            VkOp::WindowFenceStatus => "vk_window_fence_status",
            VkOp::WindowQueueWaitIdle => "vk_window_queue_wait_idle",
            VkOp::WindowSurfaceCaps => "vk_window_surface_caps",
            VkOp::WindowSurfaceFormats => "vk_window_surface_formats",
            VkOp::WindowCreateSwapchain => "vk_window_create_swapchain",
            VkOp::WindowGetSwapchainImages => "vk_window_get_swapchain_images",
            VkOp::WindowAcquireImage => "vk_window_acquire_image",
            VkOp::WindowResetFence => "vk_window_reset_fence",
            VkOp::WindowResetCommandBuffer => "vk_window_reset_command_buffer",
            VkOp::WindowBeginCommandBuffer => "vk_window_begin_command_buffer",
            VkOp::WindowEndCommandBuffer => "vk_window_end_command_buffer",
            VkOp::WindowSubmitPresent => "vk_window_submit_present",
            VkOp::WindowQueuePresent => "vk_window_queue_present",
            VkOp::WindowDestroyQueueWaitIdle => "vk_window_destroy_queue_wait_idle",
        }
    }

    /// The driver's result code — the load-bearing value the old prose carried
    /// as `{e}`. `vk::Result`'s `Display` is whitespace-free for known codes, but
    /// it is squashed defensively so a future unknown code cannot break the
    /// space-separated log line.
    fn fields(&self) -> Vec<(&'static str, String)> {
        vec![(
            "vk_result",
            self.result.to_string().replace(char::is_whitespace, "_"),
        )]
    }
}

impl std::fmt::Display for VkCall {
    /// `reason=<slug> vk_result=<code>` — what a `{e}` in someone else's
    /// `format!` produces, matching the fields the emitter renders.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "reason={} vk_result={}",
            self.slug(),
            self.result.to_string().replace(char::is_whitespace, "_")
        )
    }
}

/// Test fixture for callers that must prove a typed engine failure survives a
/// layer boundary without naming Vulkan state outside `backend/vulkan/`.
#[cfg(test)]
pub(crate) fn exec_submit_device_lost_fixture() -> super::types::DrawError {
    super::types::DrawError::VkCall(VkCall::new(VkOp::ExecSubmit, vk::Result::ERROR_DEVICE_LOST))
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: &[VkOp] = &[
        VkOp::StatsDescPool,
        VkOp::StatsSampler,
        VkOp::StatsCreateBuffer,
        VkOp::StatsAlloc,
        VkOp::StatsBind,
        VkOp::StatsMap,
        VkOp::StatsAllocCb,
        VkOp::StatsCreateFence,
        VkOp::StatsFenceStatusReclaim,
        VkOp::StatsAllocDescriptorSet,
        VkOp::StatsResetFence,
        VkOp::StatsResetCommandBuffer,
        VkOp::StatsBeginCommandBuffer,
        VkOp::StatsEndCommandBuffer,
        VkOp::StatsQueueSubmit,
        VkOp::StatsFenceStatusConsume,
        VkOp::StatsWaitFenceBlocking,
        VkOp::StatsWaitFenceDestroy,
        VkOp::ReadbackResetCb,
        VkOp::ReadbackBeginCb,
        VkOp::ReadbackEndCb,
        VkOp::ReadbackSubmit,
        VkOp::ReadbackMap,
        VkOp::HostPresentResetCb,
        VkOp::HostPresentBeginCb,
        VkOp::HostPresentEndCb,
        VkOp::HostPresentSubmit,
        VkOp::StorageReadResetCb,
        VkOp::StorageReadBeginCb,
        VkOp::StorageReadEndCb,
        VkOp::StorageReadSubmit,
        VkOp::StorageReadMap,
        VkOp::ExportScanoutMapStaging,
        VkOp::ExportScanoutResetCb,
        VkOp::ExportScanoutBeginCb,
        VkOp::ExportScanoutEndCb,
        VkOp::ExportScanoutSubmit,
        VkOp::ExportPresentResetCb,
        VkOp::ExportPresentBeginCb,
        VkOp::ExportPresentEndCb,
        VkOp::ExportPresentSubmit,
        VkOp::DmabufExportCreateImage,
        VkOp::DmabufExportAlloc,
        VkOp::DmabufExportBind,
        VkOp::DmabufExportGetFd,
        VkOp::DmabufImportCreateImage,
        VkOp::DmabufImportFdProps,
        VkOp::DmabufImportAlloc,
        VkOp::DmabufImportBind,
        VkOp::CachesCreateShaderModule,
        VkOp::CachesCreateDescriptorSetLayout,
        VkOp::CachesCreatePipelineLayout,
        VkOp::CachesCreateRenderPass,
        VkOp::CachesCreateSampler,
        VkOp::CachesCreateGraphicsPipelines,
        VkOp::CachesCreateComputePipelines,
        VkOp::ContextHostPtrProps,
        VkOp::ContextImportHostPtrAlloc,
        VkOp::ContextPipelineCacheGetData,
        VkOp::DescArenaCreatePool,
        VkOp::DescArenaAllocSets,
        VkOp::DescArenaAllocSetsGrown,
        VkOp::DescArenaFreeSets,
        VkOp::ExecResetCb,
        VkOp::ExecBeginCb,
        VkOp::ExecEndCb,
        VkOp::ExecSubmit,
        VkOp::ExecMapReadback,
        VkOp::ComputeExecResetCb,
        VkOp::ComputeExecBeginCb,
        VkOp::ComputeExecEndCb,
        VkOp::ComputeExecSubmit,
        VkOp::ComputeExecMapStorageReadback,
        VkOp::ComputeExecMapImageReadback,
        VkOp::ComputeDirectWritebackCreateBuffer,
        VkOp::ComputeDirectWritebackBindBuffer,
        VkOp::HostScatterAllocCommandBuffer,
        VkOp::HostScatterCreateFence,
        VkOp::HostScatterResetFence,
        VkOp::HostScatterResetCommandBuffer,
        VkOp::HostScatterBeginCommandBuffer,
        VkOp::HostScatterEndCommandBuffer,
        VkOp::HostScatterQueueSubmit,
        VkOp::HostScatterWaitFence,
        VkOp::GuestResetDeviceWaitIdle,
        VkOp::SlabAllocateMemory,
        VkOp::PoolsCreateCommandPool,
        VkOp::PoolsAllocCommandBuffers,
        VkOp::PoolsCreateFence,
        VkOp::PoolsWaitFencesRetire,
        VkOp::PoolsFenceStatusBeginEntry,
        VkOp::PoolsWaitFencesEntry,
        VkOp::PoolsWaitFencesDestroy,
        VkOp::PoolsResetFencesRetire,
        VkOp::PoolsEndCbBatch,
        VkOp::PoolsSubmitBatch,
        VkOp::PoolsHostImportCreateBuffer,
        VkOp::PoolsHostImportBindBuffer,
        VkOp::PoolsCreateStaging,
        VkOp::PoolsAllocStaging,
        VkOp::PoolsBindStaging,
        VkOp::PoolsMapStaging,
        VkOp::PoolsCreateReadback,
        VkOp::PoolsAllocReadback,
        VkOp::PoolsBindReadback,
        VkOp::PoolsCreateReadbackExtra,
        VkOp::PoolsAllocReadbackExtra,
        VkOp::PoolsBindReadbackExtra,
        VkOp::PoolsCreateTargetImage,
        VkOp::PoolsBindTarget,
        VkOp::PoolsCreateTargetView,
        VkOp::PoolsCreateFramebuffer,
        VkOp::PoolsCreateSampledImage,
        VkOp::PoolsBindSampled,
        VkOp::PoolsCreateSampledView,
        VkOp::PoolsCreateStorageImage,
        VkOp::PoolsAllocStorageImage,
        VkOp::PoolsBindStorageImage,
        VkOp::PoolsCreateStorageImageView,
        VkOp::PoolsCreateRegistryFramebuffer,
        VkOp::PoolsCreateRegistryTarget,
        VkOp::PoolsBindRegistryTarget,
        VkOp::PoolsCreateRegistryView,
        VkOp::PoolsCreateMrtSecondaryTarget,
        VkOp::PoolsAllocMrtSecondary,
        VkOp::PoolsBindMrtSecondary,
        VkOp::PoolsCreateMrtSecondaryView,
        VkOp::PoolsCreateDepthImage,
        VkOp::PoolsAllocDepth,
        VkOp::PoolsBindDepth,
        VkOp::PoolsCreateDepthView,
        VkOp::PoolsCreateMrtFramebuffer,
        VkOp::WindowCreateSurface,
        VkOp::WindowSurfaceSupport,
        VkOp::WindowCreateCommandPool,
        VkOp::WindowAllocCommandBuffer,
        VkOp::WindowCreateAcquireSemaphore,
        VkOp::WindowCreateRenderSemaphore,
        VkOp::WindowCreateFence,
        VkOp::WindowFenceStatus,
        VkOp::WindowQueueWaitIdle,
        VkOp::WindowSurfaceCaps,
        VkOp::WindowSurfaceFormats,
        VkOp::WindowCreateSwapchain,
        VkOp::WindowGetSwapchainImages,
        VkOp::WindowAcquireImage,
        VkOp::WindowResetFence,
        VkOp::WindowResetCommandBuffer,
        VkOp::WindowBeginCommandBuffer,
        VkOp::WindowEndCommandBuffer,
        VkOp::WindowSubmitPresent,
        VkOp::WindowQueuePresent,
        VkOp::WindowDestroyQueueWaitIdle,
    ];

    /// Two calls sharing a slug means a grep of the fail log cannot tell which
    /// Vulkan call refused — the exact defect this type replaces. Every slug is
    /// also `vk_`-prefixed (its rail) and log-safe.
    #[test]
    fn every_vk_call_names_its_operation_log_safe() {
        let mut slugs: Vec<&str> = ALL
            .iter()
            .map(|op| VkCall::new(*op, vk::Result::SUCCESS).slug())
            .collect();
        for s in &slugs {
            assert!(
                s.starts_with("vk_"),
                "slug {s:?} must carry its rail prefix"
            );
            assert!(
                s.bytes()
                    .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_'),
                "slug {s:?} must be lowercase snake_case"
            );
        }
        slugs.sort_unstable();
        let before = slugs.len();
        slugs.dedup();
        assert_eq!(before, slugs.len(), "duplicate VkOp slug");
    }

    /// The result code the old prose carried as `{e}` must reach the line as a
    /// whitespace-free `vk_result=` field — the property that keeps the
    /// space-separated log parseable, which the crate-wide gate cannot see
    /// because it does not read field values.
    #[test]
    fn the_line_carries_the_driver_result_code() {
        let c = VkCall::new(
            VkOp::StatsCreateBuffer,
            vk::Result::ERROR_OUT_OF_DEVICE_MEMORY,
        );
        let line = crate::observe::Emit::decline("stats_reduce", &c).render();
        assert!(
            line.starts_with("stats_reduce reason=vk_stats_create_buffer vk_result="),
            "{line}"
        );
        assert!(
            !line.trim_end().ends_with("vk_result="),
            "the result code must be present: {line}"
        );
        // No field value may carry whitespace — the log is split on spaces.
        for field in line.split(' ').skip(1) {
            assert!(!field.is_empty(), "double space in {line:?}");
        }
    }

    #[test]
    fn phase5_lifecycle_and_cache_calls_keep_their_exact_rail() {
        for (op, slug) in [
            (
                VkOp::ContextPipelineCacheGetData,
                "vk_context_pipeline_cache_get_data",
            ),
            (
                VkOp::GuestResetDeviceWaitIdle,
                "vk_guest_reset_device_wait_idle",
            ),
            (VkOp::PoolsWaitFencesDestroy, "vk_pools_wait_fences_destroy"),
            (
                VkOp::WindowDestroyQueueWaitIdle,
                "vk_window_destroy_queue_wait_idle",
            ),
        ] {
            let decline = VkCall::new(op, vk::Result::ERROR_DEVICE_LOST);
            assert_eq!(decline.slug(), slug);
            let expected_result = vk::Result::ERROR_DEVICE_LOST
                .to_string()
                .replace(char::is_whitespace, "_");
            assert_eq!(decline.fields(), vec![("vk_result", expected_result)]);
        }
    }
}
