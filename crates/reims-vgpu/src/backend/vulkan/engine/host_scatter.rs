//! GPU-direct scatter of a resident present into imported guest pages.
//!
//! The CPU store path reads the whole resident back into a `Vec<u8>` and then
//! memcpys it into the fragmented guest runs — two full-frame CPU copies per
//! present (~8 MiB each at 1920x1080). This module removes both: guest RAM is
//! already importable as a `VK_EXT_external_memory_host` buffer, so the same
//! per-row copy plan can be issued as `vkCmdCopyImageToBuffer` regions writing
//! straight into the guest's pages. Nothing crosses the bus but the fence.
//!
//! Each planned span is one contiguous run of texels inside one guest row, so
//! it maps to exactly one `VkBufferImageCopy` with a 1-row extent. Spans landing
//! in the same 1 GiB import window share a `VkBuffer`, so a fragmented surface
//! costs regions, not allocations — which is what makes this viable now: the
//! windowed resolver (`HOST_IMPORT_WINDOW_CAP`) buckets spans into a handful of
//! shared windows instead of the per-surface allocation storm that made an
//! earlier attempt at this back out.
//!
//! Portability: this is a strict opt-in fast path. Every reason it can't run —
//! no extension, no VMA (non-Linux), alignment the driver won't take, region cap
//! — resolves to `None` and the caller stays on the CPU scatter.

use ash::vk;

use super::context::DeviceContext;
use super::types::DrawError;
use super::vk_call::{VkCall, VkOp};

/// Attach this rail's exact operation to a failed ash call.
///
/// Split out so the same conversion used by the live command-buffer path can
/// be fired synthetically without a Vulkan device.
fn scatter_call<T>(op: VkOp, result: Result<T, vk::Result>) -> Result<T, DrawError> {
    result.map_err(|result| DrawError::VkCall(VkCall::new(op, result)))
}

/// A guest-page run to resolve against the host-import windows.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ScatterRun {
    pub host_ptr: usize,
    pub ptr_len: u64,
}

/// One planned copy: `texels` pixels starting at image `(x, y)`, landing at
/// `dst_offset` bytes into run `run_index`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ScatterSpan {
    pub run_index: usize,
    pub dst_offset: u64,
    pub x: u32,
    pub y: u32,
    pub texels: u32,
}

/// One contiguous horizontal texel run copied into a guest page span.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ScatterRegion {
    /// Imported-window buffer the guest bytes live in.
    pub buffer: vk::Buffer,
    /// Byte offset of this span's first texel within `buffer`.
    pub buffer_offset: u64,
    /// Source texel coordinates in the resident image.
    pub x: u32,
    pub y: u32,
    /// Texel count (span length / 4).
    pub texels: u32,
}

/// Single command buffer + fence for the synchronous scatter submit.
///
/// One in flight at a time by construction: the caller waits the fence before
/// returning, because the guest is told the present landed immediately after.
#[derive(Default)]
pub(crate) struct HostScatterPool {
    cmd_buf: vk::CommandBuffer,
    fence: vk::Fence,
}

impl HostScatterPool {
    unsafe fn ensure(
        &mut self,
        ctx: &DeviceContext,
        cmd_pool: vk::CommandPool,
    ) -> Result<(), DrawError> {
        if self.cmd_buf == vk::CommandBuffer::null() {
            let info = vk::CommandBufferAllocateInfo::default()
                .command_pool(cmd_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            self.cmd_buf = scatter_call(
                VkOp::HostScatterAllocCommandBuffer,
                ctx.device.allocate_command_buffers(&info),
            )?[0];
        }
        if self.fence == vk::Fence::null() {
            self.fence = scatter_call(
                VkOp::HostScatterCreateFence,
                ctx.device
                    .create_fence(&vk::FenceCreateInfo::default(), None),
            )?;
        }
        Ok(())
    }

    /// Copy `image` into the imported guest buffers described by `regions` and
    /// wait for completion. Leaves the image in `TRANSFER_SRC_OPTIMAL`; the
    /// caller must sync its tracked layout.
    ///
    /// Returns the exact failed operation if setup, recording, submit, or wait
    /// failed — the caller must then fall back to the CPU scatter, because the
    /// guest pages may hold partial content.
    pub(crate) unsafe fn scatter(
        &mut self,
        ctx: &DeviceContext,
        cmd_pool: vk::CommandPool,
        image: vk::Image,
        old_layout: vk::ImageLayout,
        regions: &[ScatterRegion],
    ) -> Result<(), DrawError> {
        if regions.is_empty() {
            // The public planner rejects uncovered/empty run sets before this
            // private rail. Keep an empty defensive call side-effect free.
            return Ok(());
        }
        self.ensure(ctx, cmd_pool)?;
        let dev = &ctx.device;
        scatter_call(VkOp::HostScatterResetFence, dev.reset_fences(&[self.fence]))?;
        let cb = self.cmd_buf;
        scatter_call(
            VkOp::HostScatterResetCommandBuffer,
            dev.reset_command_buffer(cb, vk::CommandBufferResetFlags::empty()),
        )?;
        scatter_call(
            VkOp::HostScatterBeginCommandBuffer,
            dev.begin_command_buffer(
                cb,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            ),
        )?;

        // Resident -> TRANSFER_SRC_OPTIMAL. Broad source scope, matching
        // `read_target_inner`: the frame may arrive from a colour-attachment
        // write, a transfer, or a shader write.
        if old_layout != vk::ImageLayout::TRANSFER_SRC_OPTIMAL {
            let barrier = [vk::ImageMemoryBarrier::default()
                .src_access_mask(
                    vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                        | vk::AccessFlags::TRANSFER_WRITE
                        | vk::AccessFlags::SHADER_WRITE,
                )
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .old_layout(old_layout)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
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
            dev.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &barrier,
            );
        }

        // Group consecutive regions by buffer so each `cmd_copy_image_to_buffer`
        // carries as many regions as possible. The planner emits spans in row
        // order, and a fragmented surface's runs are ordered by linear base, so
        // runs of same-buffer spans are long in practice.
        let mut start = 0usize;
        while start < regions.len() {
            let buffer = regions[start].buffer;
            let mut end = start + 1;
            while end < regions.len() && regions[end].buffer == buffer {
                end += 1;
            }
            let copies: Vec<vk::BufferImageCopy> = regions[start..end]
                .iter()
                .map(|r| {
                    vk::BufferImageCopy::default()
                        .buffer_offset(r.buffer_offset)
                        // Tight: this span is exactly one row of `texels`.
                        .buffer_row_length(r.texels)
                        .buffer_image_height(1)
                        .image_subresource(
                            vk::ImageSubresourceLayers::default()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .mip_level(0)
                                .base_array_layer(0)
                                .layer_count(1),
                        )
                        .image_offset(vk::Offset3D {
                            x: r.x as i32,
                            y: r.y as i32,
                            z: 0,
                        })
                        .image_extent(vk::Extent3D {
                            width: r.texels,
                            height: 1,
                            depth: 1,
                        })
                })
                .collect();
            dev.cmd_copy_image_to_buffer(
                cb,
                image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                buffer,
                &copies,
            );
            start = end;
        }

        // The guest reads these pages with the CPU once we return, so the
        // transfer writes must be visible to host reads.
        let host_vis = [vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::HOST_READ)];
        dev.cmd_pipeline_barrier(
            cb,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(),
            &host_vis,
            &[],
            &[],
        );

        scatter_call(
            VkOp::HostScatterEndCommandBuffer,
            dev.end_command_buffer(cb),
        )?;
        let cbs = [cb];
        let si = vk::SubmitInfo::default().command_buffers(&cbs);
        scatter_call(
            VkOp::HostScatterQueueSubmit,
            dev.queue_submit(ctx.queue(), &[si], self.fence),
        )?;
        // Synchronous: the store's completion is what tells the guest its
        // surface is readable. Same fence wait the CPU path already paid inside
        // `read_target_inner`; what we skip is the two 8 MiB memcpys after it.
        scatter_call(
            VkOp::HostScatterWaitFence,
            dev.wait_for_fences(&[self.fence], true, u64::MAX),
        )
    }

    pub(crate) unsafe fn destroy_all(&mut self, device: &ash::Device, cmd_pool: vk::CommandPool) {
        if self.cmd_buf != vk::CommandBuffer::null() {
            device.free_command_buffers(cmd_pool, &[self.cmd_buf]);
            self.cmd_buf = vk::CommandBuffer::null();
        }
        if self.fence != vk::Fence::null() {
            device.destroy_fence(self.fence, None);
            self.fence = vk::Fence::null();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observe::Decline as _;

    /// This fires the exact adapter every live ash call uses. The old boolean
    /// rail made all eight cases indistinguishable as `submit_failed`.
    #[test]
    fn every_scatter_vk_failure_preserves_its_operation() {
        let cases = [
            (
                VkOp::HostScatterAllocCommandBuffer,
                "vk_host_scatter_alloc_command_buffer",
            ),
            (VkOp::HostScatterCreateFence, "vk_host_scatter_create_fence"),
            (VkOp::HostScatterResetFence, "vk_host_scatter_reset_fence"),
            (
                VkOp::HostScatterResetCommandBuffer,
                "vk_host_scatter_reset_command_buffer",
            ),
            (
                VkOp::HostScatterBeginCommandBuffer,
                "vk_host_scatter_begin_command_buffer",
            ),
            (
                VkOp::HostScatterEndCommandBuffer,
                "vk_host_scatter_end_command_buffer",
            ),
            (VkOp::HostScatterQueueSubmit, "vk_host_scatter_queue_submit"),
            (VkOp::HostScatterWaitFence, "vk_host_scatter_wait_fence"),
        ];
        for (op, expected) in cases {
            let error = scatter_call::<()>(op, Err(vk::Result::ERROR_DEVICE_LOST))
                .expect_err("synthetic driver failure must decline");
            assert_eq!(error.slug(), expected);
            let fields = error.fields();
            assert_eq!(fields.len(), 1);
            assert_eq!(fields[0].0, "vk_result");
            assert!(!fields[0].1.is_empty());
            assert!(!fields[0].1.contains(char::is_whitespace));
        }
    }
}
