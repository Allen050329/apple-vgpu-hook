//! Engine-device WSI presentation for the macOS Vulkan host window.
//!
//! The final compositor resident stays on the engine `VkDevice`. A short
//! queue-ordered blit writes it into the acquired MoltenVK swapchain image; no
//! host readback, staging upload, dmabuf export, or second Vulkan device exists
//! on this pathway.

#![allow(unsafe_op_in_unsafe_fn)]

use ash::vk;
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};
use std::time::Instant;

use super::context::DeviceContext;
use super::counters::EngineCounters;
use super::facade_decline::EngineFacadeDecline;
use super::pools::ResourcePools;
use super::types::{DrawError, PresentRect, TargetIdentity, WindowPresentSource};
use super::vk_call::{VkCall, VkOp};
use crate::backend::vulkan::translate;

/// Consecutive suboptimal-flagged presents (each of which arms a swapchain
/// recreation) before the always-on alarm names the class. Recreation normally
/// clears the flag on the next frame, and a live user resize clears the streak
/// whenever the extent actually changes. A streak this long at an unchanged
/// extent means recreation is not converging and the window may be presenting
/// invisibly (the CAMetalLayer drawableSize-clobber class).
const SUBOPTIMAL_ALARM_STREAK: u32 = 60;

/// The pre-content / letterbox-bar clear color (linear BGRA channels).
const SLATE_CLEAR: [f32; 4] = [0.05, 0.06, 0.08, 1.0];

/// A host-window present degradation that does not abort the whole present.
///
/// These are not [`SlateReason`]s: a malformed peer rect skips only that
/// correction tile, and a persistent suboptimal flag still queues presents
/// while warning that swapchain recreation is not converging.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WindowPresentDecline {
    PeerRectOutOfBounds {
        rect: PresentRect,
        width: u32,
        height: u32,
    },
    SuboptimalPersistent {
        streak: u32,
        width: u32,
        height: u32,
    },
}

impl crate::observe::Decline for WindowPresentDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::PeerRectOutOfBounds { .. } => "window_present_peer_rect_out_of_bounds",
            Self::SuboptimalPersistent { .. } => "window_present_suboptimal_persistent",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::PeerRectOutOfBounds {
                rect: (x0, y0, x1, y1),
                width,
                height,
            } => vec![
                ("x0", x0.to_string()),
                ("y0", y0.to_string()),
                ("x1", x1.to_string()),
                ("y1", y1.to_string()),
                ("width", width.to_string()),
                ("height", height.to_string()),
            ],
            Self::SuboptimalPersistent {
                streak,
                width,
                height,
            } => vec![
                ("streak", streak.to_string()),
                ("width", width.to_string()),
                ("height", height.to_string()),
            ],
        }
    }
}

/// Why a present cleared to slate instead of blitting a guest resident.
///
/// A slate present is the window showing *nothing* — on the arm64 MoltenVK
/// pathway it is the whole "blank window" failure class, and it used to happen
/// with no log line at all: the caller only reported the FIRST direct present,
/// so a later regression into slate was invisible except as a drop in
/// `direct_frac`. Every slate run now names its cause.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SlateReason {
    /// No frame source was published for this present. Expected before the
    /// present boundary and while the guest is idle.
    NoSource,
    /// The source named candidate identities but none is in the resident
    /// registry — the resident was evicted, or never created.
    NoResident,
    /// A resident exists but its content has not landed yet.
    ContentNotReady,
    /// A resident exists and is ready, but is not BGRA. The present blit does
    /// no format conversion, so it cannot be shown.
    NotBgra,
    /// A resident exists and is ready, but at different dimensions than the
    /// source claims — presenting it would show a torn or scaled frame.
    GeomMismatch,
}

impl crate::observe::Decline for SlateReason {
    /// Slugs carry a `slate_` prefix.
    ///
    /// They were bare (`no_source`, `geom_mismatch`, …) while this type was an
    /// island with its own `slug()`. Crate-wide they read as claims about the
    /// whole present path rather than about the window's blit choice, and
    /// `geom_mismatch` is also a `THRASH` proxy name while `no_resident` sits
    /// one word away from the capture rail's `no_resident_content`. A grep for
    /// a bare one would mix three different subsystems.
    fn slug(&self) -> &'static str {
        match self {
            Self::NoSource => "slate_no_source",
            Self::NoResident => "slate_no_resident",
            Self::ContentNotReady => "slate_content_not_ready",
            Self::NotBgra => "slate_not_bgra",
            Self::GeomMismatch => "slate_geom_mismatch",
        }
    }
}

/// What the registry knows about one candidate identity, flattened so the
/// classification below is pure and testable without a GPU.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct CandidateState {
    /// The identity resolved to a registry slot.
    pub resident: bool,
    pub content_ready: bool,
    pub bgra: bool,
    pub width: u32,
    pub height: u32,
}

/// Name why no candidate could be presented.
///
/// Reports the blocker closest to success: a resident that is ready and BGRA
/// but the wrong size is a more actionable diagnosis than a sibling candidate
/// that was never created. Ordering matters — collapsing these into one
/// "no_resident" is the exact "N distinct checks share one status" trap the
/// failure-logging rules call out.
pub(crate) fn classify_slate(
    source_present: bool,
    want: (u32, u32),
    candidates: &[CandidateState],
) -> SlateReason {
    if !source_present {
        return SlateReason::NoSource;
    }
    let resident: Vec<_> = candidates.iter().filter(|c| c.resident).collect();
    if resident.is_empty() {
        return SlateReason::NoResident;
    }
    if resident
        .iter()
        .any(|c| c.content_ready && c.bgra && (c.width, c.height) != want)
    {
        return SlateReason::GeomMismatch;
    }
    if resident.iter().any(|c| c.content_ready && !c.bgra) {
        return SlateReason::NotBgra;
    }
    SlateReason::ContentNotReady
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WindowPresentOutcome {
    Busy,
    Presented {
        direct: bool,
        width: u32,
        height: u32,
        swapchain_images: usize,
        /// The surface reported suboptimal at acquire or present, so a
        /// recreation is armed. The window must schedule another redraw
        /// promptly instead of waiting for the next guest frame — boot-era
        /// presents can be seconds apart, which would leave a mismatched
        /// drawable on screen for that long.
        suboptimal: bool,
    },
}

pub(crate) struct WindowPresenter {
    surface_loader: ash::khr::surface::Instance,
    surface: vk::SurfaceKHR,
    swapchain_loader: ash::khr::swapchain::Device,
    swapchain: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    extent: vk::Extent2D,
    desired_extent: vk::Extent2D,
    recreate_pending: bool,
    /// Why the next recreation was armed — carried into the always-on
    /// `host_window_swapchain` line so a live log separates guest/user resizes
    /// from suboptimal-surface self-heals.
    recreate_reason: &'static str,
    /// Consecutive presents whose acquire or present reported a suboptimal
    /// surface. Each one arms a recreation; see [`SUBOPTIMAL_ALARM_STREAK`].
    suboptimal_streak: u32,
    /// Latch for the malformed peer-rect fail line — once per presenter, not
    /// per frame, because one bad upstream producer would flood otherwise.
    peer_rect_warned: bool,
    /// Reason for the slate run currently in progress, `None` while presenting
    /// guest content. A line is emitted when a run STARTS or its reason
    /// CHANGES, and a summary when it ends — so a window blank for a minute at
    /// 120 Hz costs two lines, not 7200.
    slate_reason: Option<SlateReason>,
    /// Consecutive slate presents in the current run.
    slate_run: u64,
    cmd_pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    image_available: vk::Semaphore,
    render_finished: vk::Semaphore,
    in_flight: vk::Fence,
    submitted: bool,
    pinned: Vec<TargetIdentity>,
    cadence_started: Instant,
    cadence_presents: u64,
    cadence_direct: u64,
    cadence_busy: u64,
}

impl WindowPresenter {
    pub(crate) unsafe fn create(
        ctx: &DeviceContext,
        display: RawDisplayHandle,
        window: RawWindowHandle,
        width: u32,
        height: u32,
    ) -> Result<Self, DrawError> {
        if !ctx.swapchain {
            return Err(DrawError::Unsupported(
                super::reason::DrawReason::SwapchainUnavailable,
            ));
        }
        let surface = ash_window::create_surface(&ctx._entry, &ctx.instance, display, window, None)
            .map_err(|error| DrawError::VkCall(VkCall::new(VkOp::WindowCreateSurface, error)))?;
        let surface_loader = ash::khr::surface::Instance::new(&ctx._entry, &ctx.instance);
        let present_capable = surface_loader
            .get_physical_device_surface_support(ctx.pd, ctx.gq, surface)
            .map_err(|error| {
                surface_loader.destroy_surface(surface, None);
                DrawError::VkCall(VkCall::new(VkOp::WindowSurfaceSupport, error))
            })?;
        if !present_capable {
            surface_loader.destroy_surface(surface, None);
            return Err(DrawError::Unsupported(
                super::reason::DrawReason::QueueCannotPresent {
                    queue_family: ctx.gq,
                },
            ));
        }

        let cmd_pool = match ctx.device.create_command_pool(
            &vk::CommandPoolCreateInfo::default()
                .queue_family_index(ctx.gq)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
            None,
        ) {
            Ok(pool) => pool,
            Err(error) => {
                surface_loader.destroy_surface(surface, None);
                return Err(DrawError::VkCall(VkCall::new(
                    VkOp::WindowCreateCommandPool,
                    error,
                )));
            }
        };
        let cmd = match ctx.device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(cmd_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1),
        ) {
            Ok(buffers) => buffers[0],
            Err(error) => {
                ctx.device.destroy_command_pool(cmd_pool, None);
                surface_loader.destroy_surface(surface, None);
                return Err(DrawError::VkCall(VkCall::new(
                    VkOp::WindowAllocCommandBuffer,
                    error,
                )));
            }
        };
        let image_available = match ctx
            .device
            .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
        {
            Ok(semaphore) => semaphore,
            Err(error) => {
                ctx.device.destroy_command_pool(cmd_pool, None);
                surface_loader.destroy_surface(surface, None);
                return Err(DrawError::VkCall(VkCall::new(
                    VkOp::WindowCreateAcquireSemaphore,
                    error,
                )));
            }
        };
        let render_finished = match ctx
            .device
            .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
        {
            Ok(semaphore) => semaphore,
            Err(error) => {
                ctx.device.destroy_semaphore(image_available, None);
                ctx.device.destroy_command_pool(cmd_pool, None);
                surface_loader.destroy_surface(surface, None);
                return Err(DrawError::VkCall(VkCall::new(
                    VkOp::WindowCreateRenderSemaphore,
                    error,
                )));
            }
        };
        let in_flight = match ctx.device.create_fence(
            &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
            None,
        ) {
            Ok(fence) => fence,
            Err(error) => {
                ctx.device.destroy_semaphore(render_finished, None);
                ctx.device.destroy_semaphore(image_available, None);
                ctx.device.destroy_command_pool(cmd_pool, None);
                surface_loader.destroy_surface(surface, None);
                return Err(DrawError::VkCall(VkCall::new(
                    VkOp::WindowCreateFence,
                    error,
                )));
            }
        };

        let mut presenter = Self {
            surface_loader,
            surface,
            swapchain_loader: ash::khr::swapchain::Device::new(&ctx.instance, &ctx.device),
            swapchain: vk::SwapchainKHR::null(),
            images: Vec::new(),
            extent: vk::Extent2D::default(),
            desired_extent: vk::Extent2D {
                width: width.max(1),
                height: height.max(1),
            },
            recreate_pending: true,
            recreate_reason: "init",
            suboptimal_streak: 0,
            peer_rect_warned: false,
            slate_reason: None,
            slate_run: 0,
            cmd_pool,
            cmd,
            image_available,
            render_finished,
            in_flight,
            submitted: false,
            pinned: Vec::new(),
            cadence_started: Instant::now(),
            cadence_presents: 0,
            cadence_direct: 0,
            cadence_busy: 0,
        };
        if let Err(error) = presenter.recreate_swapchain(ctx) {
            presenter.destroy(ctx, None);
            return Err(error);
        }
        Ok(presenter)
    }

    pub(crate) fn resize(&mut self, width: u32, height: u32) {
        let requested = vk::Extent2D {
            width: width.max(1),
            height: height.max(1),
        };
        if requested != self.desired_extent {
            self.recreate_pending = true;
            self.recreate_reason = "resize";
        }
        self.desired_extent = requested;
    }

    unsafe fn retire(
        &mut self,
        ctx: &DeviceContext,
        pools: &mut ResourcePools,
    ) -> Result<bool, DrawError> {
        if !self.submitted {
            return Ok(true);
        }
        let signaled = ctx
            .device
            .get_fence_status(self.in_flight)
            .map_err(|error| DrawError::VkCall(VkCall::new(VkOp::WindowFenceStatus, error)))?;
        if !signaled {
            return Ok(false);
        }
        for identity in self.pinned.drain(..) {
            let _ = pools.pin_resident_target(&identity, false);
        }
        self.submitted = false;
        Ok(true)
    }

    unsafe fn recreate_swapchain(&mut self, ctx: &DeviceContext) -> Result<(), DrawError> {
        ctx.device
            .queue_wait_idle(ctx.queue())
            .map_err(|error| DrawError::VkCall(VkCall::new(VkOp::WindowQueueWaitIdle, error)))?;
        let caps = self
            .surface_loader
            .get_physical_device_surface_capabilities(ctx.pd, self.surface)
            .map_err(|error| DrawError::VkCall(VkCall::new(VkOp::WindowSurfaceCaps, error)))?;
        if !caps
            .supported_usage_flags
            .contains(vk::ImageUsageFlags::TRANSFER_DST)
        {
            return Err(DrawError::Unsupported(
                super::reason::DrawReason::SwapchainLacksTransferDst,
            ));
        }
        let formats = self
            .surface_loader
            .get_physical_device_surface_formats(ctx.pd, self.surface)
            .map_err(|error| DrawError::VkCall(VkCall::new(VkOp::WindowSurfaceFormats, error)))?;
        let format = formats
            .iter()
            .find(|format| {
                format.format == translate::pixel::SCANOUT_FORMAT
                    && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .or_else(|| formats.first())
            .copied()
            .ok_or(DrawError::Unsupported(
                super::reason::DrawReason::SwapchainNoSurfaceFormat,
            ))?;
        let extent = if caps.current_extent.width != u32::MAX {
            caps.current_extent
        } else {
            vk::Extent2D {
                width: self
                    .desired_extent
                    .width
                    .clamp(caps.min_image_extent.width, caps.max_image_extent.width),
                height: self
                    .desired_extent
                    .height
                    .clamp(caps.min_image_extent.height, caps.max_image_extent.height),
            }
        };
        let mut image_count = caps.min_image_count.saturating_add(1);
        if caps.max_image_count != 0 {
            image_count = image_count.min(caps.max_image_count);
        }
        let composite_alpha = [
            vk::CompositeAlphaFlagsKHR::OPAQUE,
            vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED,
            vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED,
            vk::CompositeAlphaFlagsKHR::INHERIT,
        ]
        .into_iter()
        .find(|flag| caps.supported_composite_alpha.contains(*flag))
        .ok_or(DrawError::Unsupported(
            super::reason::DrawReason::SwapchainNoCompositeAlpha,
        ))?;
        // Destroy the old swapchain BEFORE creating its replacement, and create
        // the replacement without `old_swapchain`. MoltenVK (verified against
        // v1.4.1 MVKSwapchain.mm) works around a Metal present-callback
        // regression by setting the CAMetalLayer drawableSize to {1,1} when a
        // swapchain that still has 1-2 unpresented images is retired; with
        // `old_swapchain`, that clobber runs AFTER the new swapchain has
        // already configured the layer, and nothing restores the size — every
        // later present then succeeds (flagged suboptimal only) while the
        // window displays a single stretched pixel. Destroy-first makes the new
        // swapchain's layer configuration the final write, the ordering that
        // workaround assumes. The queue idled above, so no submitted work
        // references the old swapchain.
        let from = self.extent;
        if self.swapchain != vk::SwapchainKHR::null() {
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
            self.swapchain = vk::SwapchainKHR::null();
            self.images.clear();
        }
        let swapchain = self
            .swapchain_loader
            .create_swapchain(
                &vk::SwapchainCreateInfoKHR::default()
                    .surface(self.surface)
                    .min_image_count(image_count)
                    .image_format(format.format)
                    .image_color_space(format.color_space)
                    .image_extent(extent)
                    .image_array_layers(1)
                    .image_usage(vk::ImageUsageFlags::TRANSFER_DST)
                    .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .pre_transform(caps.current_transform)
                    .composite_alpha(composite_alpha)
                    .present_mode(vk::PresentModeKHR::FIFO)
                    .clipped(true),
                None,
            )
            .map_err(|error| DrawError::VkCall(VkCall::new(VkOp::WindowCreateSwapchain, error)))?;
        let images = self
            .swapchain_loader
            .get_swapchain_images(swapchain)
            .map_err(|error| {
                self.swapchain_loader.destroy_swapchain(swapchain, None);
                DrawError::VkCall(VkCall::new(VkOp::WindowGetSwapchainImages, error))
            })?;
        // Fresh per-recreation semaphores: an acquire whose submit later failed
        // leaves `image_available` with a signal nobody consumed, which is
        // invalid to reuse on the new swapchain's first acquire. Created before
        // the old pair is destroyed so a failure leaves the presenter
        // consistent.
        let image_available = match ctx
            .device
            .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
        {
            Ok(semaphore) => semaphore,
            Err(error) => {
                self.swapchain_loader.destroy_swapchain(swapchain, None);
                return Err(DrawError::VkCall(VkCall::new(
                    VkOp::WindowCreateAcquireSemaphore,
                    error,
                )));
            }
        };
        let render_finished = match ctx
            .device
            .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
        {
            Ok(semaphore) => semaphore,
            Err(error) => {
                ctx.device.destroy_semaphore(image_available, None);
                self.swapchain_loader.destroy_swapchain(swapchain, None);
                return Err(DrawError::VkCall(VkCall::new(
                    VkOp::WindowCreateRenderSemaphore,
                    error,
                )));
            }
        };
        ctx.device.destroy_semaphore(self.image_available, None);
        ctx.device.destroy_semaphore(self.render_finished, None);
        self.image_available = image_available;
        self.render_finished = render_finished;
        self.swapchain = swapchain;
        self.images = images;
        self.extent = extent;
        self.desired_extent = extent;
        self.recreate_pending = false;
        if extent != from {
            // A geometry change is progress; only a same-extent suboptimal
            // loop should keep accumulating toward the alarm.
            self.suboptimal_streak = 0;
        }
        crate::observe::off(swapchain_recreated_line(from, extent, self.recreate_reason));
        Ok(())
    }

    pub(crate) unsafe fn present(
        &mut self,
        ctx: &DeviceContext,
        pools: &mut ResourcePools,
        counters: &EngineCounters,
        source: Option<&WindowPresentSource>,
    ) -> Result<WindowPresentOutcome, DrawError> {
        if !self.retire(ctx, pools)? {
            self.note_cadence(false, false);
            return Ok(WindowPresentOutcome::Busy);
        }
        if self.swapchain == vk::SwapchainKHR::null() || self.recreate_pending {
            self.recreate_swapchain(ctx)?;
        }
        let (image_index, acquire_suboptimal) = match self.swapchain_loader.acquire_next_image(
            self.swapchain,
            0,
            self.image_available,
            vk::Fence::null(),
        ) {
            Ok((index, suboptimal)) => (index, suboptimal),
            Err(vk::Result::NOT_READY) | Err(vk::Result::TIMEOUT) => {
                self.note_cadence(false, false);
                return Ok(WindowPresentOutcome::Busy);
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.recreate_pending = true;
                self.recreate_reason = "acquire_out_of_date";
                self.note_cadence(false, false);
                return Ok(WindowPresentOutcome::Busy);
            }
            Err(error) => {
                return Err(DrawError::VkCall(VkCall::new(
                    VkOp::WindowAcquireImage,
                    error,
                )));
            }
        };

        pools.batch_flush(ctx, counters)?;
        let selected = source.and_then(|source| {
            source.candidates.iter().find_map(|identity| {
                let slot = pools.registry_get(identity)?;
                (slot.content_ready
                    && slot.bgra
                    && slot.width == source.width
                    && slot.height == source.height)
                    .then(|| {
                        (
                            identity.clone(),
                            slot.image,
                            slot.layout,
                            slot.width,
                            slot.height,
                        )
                    })
            })
        });
        if selected.is_some() {
            self.note_slate_end();
        } else {
            // Failure path only: re-walk the candidates to name WHY nothing
            // could be presented. Cheap because it never runs on a good frame.
            let states: Vec<CandidateState> = source
                .map(|source| {
                    source
                        .candidates
                        .iter()
                        .map(|identity| match pools.registry_get(identity) {
                            Some(slot) => CandidateState {
                                resident: true,
                                content_ready: slot.content_ready,
                                bgra: slot.bgra,
                                width: slot.width,
                                height: slot.height,
                            },
                            None => CandidateState::default(),
                        })
                        .collect()
                })
                .unwrap_or_default();
            let want = source.map_or((0, 0), |s| (s.width, s.height));
            self.note_slate(
                classify_slate(source.is_some(), want, &states),
                want,
                &states,
            );
        }
        let peer = selected
            .as_ref()
            .and_then(|(identity, _, _, width, height)| {
                if !matches!(identity, TargetIdentity::Surface { .. }) {
                    return None;
                }
                let (peer_identity, rects) = source?.peer.as_ref()?;
                let slot = pools.registry_get(peer_identity)?;
                (slot.content_ready
                    && slot.bgra
                    && slot.width == *width
                    && slot.height == *height
                    && !rects.is_empty())
                .then(|| {
                    (
                        peer_identity.clone(),
                        slot.image,
                        slot.layout,
                        rects.as_slice(),
                    )
                })
            });

        let mut pinned = Vec::with_capacity(2);
        if let Some((identity, _, _, _, _)) = selected.as_ref() {
            if !pools.pin_resident_target(identity, true) {
                return Err(DrawError::Facade(
                    EngineFacadeDecline::WindowSourceDisappearedBeforePin {
                        identity: identity.clone(),
                    },
                ));
            }
            pinned.push(identity.clone());
        }
        if let Some((identity, _, _, _)) = peer.as_ref() {
            if !pools.pin_resident_target(identity, true) {
                for pinned_identity in pinned.drain(..) {
                    let _ = pools.pin_resident_target(&pinned_identity, false);
                }
                return Err(DrawError::Facade(
                    EngineFacadeDecline::WindowPeerDisappearedBeforePin {
                        identity: identity.clone(),
                    },
                ));
            }
            pinned.push(identity.clone());
        }

        let submit_result = (|| {
            ctx.device
                .reset_fences(&[self.in_flight])
                .map_err(|error| DrawError::VkCall(VkCall::new(VkOp::WindowResetFence, error)))?;
            ctx.device
                .reset_command_buffer(self.cmd, vk::CommandBufferResetFlags::empty())
                .map_err(|error| {
                    DrawError::VkCall(VkCall::new(VkOp::WindowResetCommandBuffer, error))
                })?;
            ctx.device
                .begin_command_buffer(
                    self.cmd,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .map_err(|error| {
                    DrawError::VkCall(VkCall::new(VkOp::WindowBeginCommandBuffer, error))
                })?;

            let color_range = vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .level_count(1)
                .layer_count(1);
            let dst = self.images[image_index as usize];
            image_barrier(
                &ctx.device,
                self.cmd,
                dst,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::AccessFlags::empty(),
                vk::AccessFlags::TRANSFER_WRITE,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
            );
            if let Some((identity, image, layout, base_width, base_height)) = selected.as_ref() {
                // Aspect-fit placement: the guest frame keeps its aspect ratio
                // inside whatever drawable exists right now (a guest-driven
                // native resize normally makes this the full window within
                // milliseconds). The window input path maps pointer positions
                // through this same transform.
                let vp = crate::host_window::viewport::aspect_fit(
                    (*base_width, *base_height),
                    (self.extent.width, self.extent.height),
                );
                if !vp.covers((self.extent.width, self.extent.height)) {
                    // Letterbox bars: clear the whole image first so stale
                    // swapchain pixels never frame the guest content.
                    ctx.device.cmd_clear_color_image(
                        self.cmd,
                        dst,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &vk::ClearColorValue {
                            float32: SLATE_CLEAR,
                        },
                        &[color_range],
                    );
                }
                transition_source(&ctx.device, self.cmd, *image, *layout);
                blit_rect(
                    &ctx.device,
                    self.cmd,
                    *image,
                    dst,
                    (0, 0, *base_width, *base_height),
                    (vp.x, vp.y, vp.x + vp.width, vp.y + vp.height),
                );
                pools.registry_set_layout(identity, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
                if let Some((peer_identity, peer_image, peer_layout, rects)) = peer.as_ref() {
                    transition_source(&ctx.device, self.cmd, *peer_image, *peer_layout);
                    for &rect in *rects {
                        // The divergent-tile contract is endpoint rects
                        // (x0,y0,x1,y1) — the same form `read_target_inner`
                        // consumes on the CPU capture route. Reading them as
                        // (x,y,w,h) displaced every non-edge peer region.
                        let Some((src_rect, dst_rect)) =
                            map_peer_rect(rect, (*base_width, *base_height), &vp)
                        else {
                            if !self.peer_rect_warned {
                                let decline = WindowPresentDecline::PeerRectOutOfBounds {
                                    rect,
                                    width: *base_width,
                                    height: *base_height,
                                };
                                crate::observe::Emit::decline("host_window_present", &decline)
                                    .fail();
                                self.peer_rect_warned = true;
                            }
                            continue;
                        };
                        blit_rect(&ctx.device, self.cmd, *peer_image, dst, src_rect, dst_rect);
                    }
                    pools.registry_set_layout(peer_identity, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
                }
            } else {
                ctx.device.cmd_clear_color_image(
                    self.cmd,
                    dst,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &vk::ClearColorValue {
                        float32: SLATE_CLEAR,
                    },
                    &[color_range],
                );
            }
            image_barrier(
                &ctx.device,
                self.cmd,
                dst,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::PRESENT_SRC_KHR,
                vk::AccessFlags::TRANSFER_WRITE,
                vk::AccessFlags::empty(),
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            );
            ctx.device.end_command_buffer(self.cmd).map_err(|error| {
                DrawError::VkCall(VkCall::new(VkOp::WindowEndCommandBuffer, error))
            })?;
            let waits = [self.image_available];
            let wait_stages = [vk::PipelineStageFlags::TRANSFER];
            let signals = [self.render_finished];
            let commands = [self.cmd];
            ctx.device
                .queue_submit(
                    ctx.queue(),
                    &[vk::SubmitInfo::default()
                        .wait_semaphores(&waits)
                        .wait_dst_stage_mask(&wait_stages)
                        .command_buffers(&commands)
                        .signal_semaphores(&signals)],
                    self.in_flight,
                )
                .map_err(|error| DrawError::VkCall(VkCall::new(VkOp::WindowSubmitPresent, error)))
        })();
        if let Err(error) = submit_result {
            for identity in pinned.drain(..) {
                let _ = pools.pin_resident_target(&identity, false);
            }
            return Err(error);
        }
        self.pinned = pinned;
        self.submitted = true;

        let swapchains = [self.swapchain];
        let indices = [image_index];
        let waits = [self.render_finished];
        match self.swapchain_loader.queue_present(
            ctx.queue(),
            &vk::PresentInfoKHR::default()
                .wait_semaphores(&waits)
                .swapchains(&swapchains)
                .image_indices(&indices),
        ) {
            Ok(present_suboptimal) => {
                // ash reports VK_SUBOPTIMAL_KHR as `Ok(true)` (a success code),
                // never through the `Err` arm. MoltenVK returns it from both
                // acquire and present for as long as the CAMetalLayer's
                // drawable or natural size diverges from the swapchain extent —
                // including after a retired swapchain clobbered the layer's
                // drawableSize — so ignoring the flag leaves an invisible
                // window that still counts successful presents.
                let suboptimal = acquire_suboptimal || present_suboptimal;
                if suboptimal {
                    self.recreate_pending = true;
                    self.recreate_reason = "suboptimal";
                    self.suboptimal_streak = self.suboptimal_streak.saturating_add(1);
                    if self.suboptimal_streak == SUBOPTIMAL_ALARM_STREAK {
                        let decline = WindowPresentDecline::SuboptimalPersistent {
                            streak: self.suboptimal_streak,
                            width: self.extent.width,
                            height: self.extent.height,
                        };
                        crate::observe::Emit::decline("host_window_present", &decline).fail();
                    }
                } else {
                    self.suboptimal_streak = 0;
                }
                let direct = selected.is_some();
                self.note_cadence(true, direct);
                Ok(WindowPresentOutcome::Presented {
                    direct,
                    width: self.extent.width,
                    height: self.extent.height,
                    swapchain_images: self.images.len(),
                    suboptimal,
                })
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.recreate_pending = true;
                self.recreate_reason = "present_out_of_date";
                self.note_cadence(false, false);
                Ok(WindowPresentOutcome::Busy)
            }
            Err(error) => Err(DrawError::VkCall(VkCall::new(
                VkOp::WindowQueuePresent,
                error,
            ))),
        }
    }

    /// Record a slate present. Emits a fail-visible line when a run starts or
    /// its reason changes; silent for every repeat within a run.
    fn note_slate(&mut self, reason: SlateReason, want: (u32, u32), states: &[CandidateState]) {
        if self.slate_reason == Some(reason) {
            self.slate_run = self.slate_run.saturating_add(1);
            return;
        }
        if self.slate_reason.is_some() {
            self.note_slate_end();
        }
        self.slate_reason = Some(reason);
        self.slate_run = 1;
        let seen = states
            .iter()
            .map(|c| {
                if c.resident {
                    format!(
                        "{}x{}/{}{}",
                        c.width, c.height, c.content_ready as u8, c.bgra as u8
                    )
                } else {
                    "absent".to_string()
                }
            })
            .collect::<Vec<_>>()
            .join(",");
        crate::observe::Emit::decline("host_window_slate", &reason)
            .field("want", format!("{}x{}", want.0, want.1))
            .field("candidates", states.len())
            .field("seen", format!("[{seen}]"))
            .fail();
    }

    /// Close an in-progress slate run, reporting how long the window was blank.
    fn note_slate_end(&mut self) {
        let Some(reason) = self.slate_reason.take() else {
            return;
        };
        // `off()`, not `fail()`: the run *ending* is the window recovering, so
        // it is a census line rather than a drop, per the curated-fail rule.
        crate::observe::Emit::decline("host_window_slate_end", &reason)
            .field("frames", self.slate_run)
            .off();
        self.slate_run = 0;
    }

    fn note_cadence(&mut self, presented: bool, direct: bool) {
        if presented {
            self.cadence_presents = self.cadence_presents.saturating_add(1);
            self.cadence_direct = self.cadence_direct.saturating_add(u64::from(direct));
        } else {
            self.cadence_busy = self.cadence_busy.saturating_add(1);
        }
        let elapsed = self.cadence_started.elapsed();
        if elapsed.as_millis() < 1_000 {
            return;
        }
        crate::observe::off(window_cadence_line(
            elapsed.as_millis() as u64,
            self.cadence_presents,
            self.cadence_direct,
            self.cadence_busy,
        ));
        self.cadence_started = Instant::now();
        self.cadence_presents = 0;
        self.cadence_direct = 0;
        self.cadence_busy = 0;
    }

    pub(crate) fn release_pins_after_idle(&mut self, pools: &mut ResourcePools) {
        for identity in self.pinned.drain(..) {
            let _ = pools.pin_resident_target(&identity, false);
        }
        self.submitted = false;
    }

    pub(crate) unsafe fn destroy(
        &mut self,
        ctx: &DeviceContext,
        pools: Option<&mut ResourcePools>,
    ) {
        if let Err(error) = ctx.device.queue_wait_idle(ctx.queue()) {
            let decline = VkCall::new(VkOp::WindowDestroyQueueWaitIdle, error);
            crate::observe::Emit::decline("host_window_destroy", &decline).fail_once(0);
        }
        if let Some(pools) = pools {
            for identity in self.pinned.drain(..) {
                let _ = pools.pin_resident_target(&identity, false);
            }
        } else {
            self.pinned.clear();
        }
        self.submitted = false;
        ctx.device.destroy_fence(self.in_flight, None);
        ctx.device.destroy_semaphore(self.render_finished, None);
        ctx.device.destroy_semaphore(self.image_available, None);
        ctx.device.destroy_command_pool(self.cmd_pool, None);
        if self.swapchain != vk::SwapchainKHR::null() {
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
            self.swapchain = vk::SwapchainKHR::null();
        }
        self.surface_loader.destroy_surface(self.surface, None);
    }
}

/// Clip an endpoint-form `(x0,y0,x1,y1)` damage rect to the source bounds and
/// scale it into the destination viewport. `None` for a rect that is empty or
/// entirely out of range after clipping — the caller treats that as a
/// fail-visible upstream contract violation, not silent control flow.
fn map_peer_rect(
    rect: (u32, u32, u32, u32),
    src: (u32, u32),
    vp: &crate::host_window::viewport::Viewport,
) -> Option<(PresentRect, PresentRect)> {
    let (x0, y0, x1, y1) = rect;
    let x1 = x1.min(src.0);
    let y1 = y1.min(src.1);
    if x0 >= x1 || y0 >= y1 {
        return None;
    }
    let dst_rect = (
        vp.x + scale_edge(x0, src.0, vp.width),
        vp.y + scale_edge(y0, src.1, vp.height),
        vp.x + scale_edge_ceil(x1, src.0, vp.width),
        vp.y + scale_edge_ceil(y1, src.1, vp.height),
    );
    Some(((x0, y0, x1, y1), dst_rect))
}

fn swapchain_recreated_line(from: vk::Extent2D, to: vk::Extent2D, reason: &str) -> String {
    format!(
        "host_window_swapchain status=recreated from={}x{} to={}x{} trigger={reason}",
        from.width, from.height, to.width, to.height
    )
}

fn window_cadence_line(window_ms: u64, presents: u64, direct: u64, busy: u64) -> String {
    let hz = presents as f64 * 1_000.0 / window_ms.max(1) as f64;
    let direct_fraction = direct as f64 / presents.max(1) as f64;
    format!(
        "host_window_cadence window_ms={window_ms} presents={presents} direct={direct} \
         busy={busy} present_hz={hz:.1} direct_frac={direct_fraction:.2}"
    )
}

unsafe fn transition_source(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    old_layout: vk::ImageLayout,
) {
    if old_layout != vk::ImageLayout::TRANSFER_SRC_OPTIMAL {
        image_barrier(
            device,
            cmd,
            image,
            old_layout,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                | vk::AccessFlags::SHADER_WRITE
                | vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::TRANSFER_READ,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::PipelineStageFlags::TRANSFER,
        );
    }
}

#[allow(
    clippy::too_many_arguments,
    reason = "the helper mirrors the complete Vulkan image barrier state"
)]
unsafe fn image_barrier(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    src_access: vk::AccessFlags,
    dst_access: vk::AccessFlags,
    src_stage: vk::PipelineStageFlags,
    dst_stage: vk::PipelineStageFlags,
) {
    device.cmd_pipeline_barrier(
        cmd,
        src_stage,
        dst_stage,
        vk::DependencyFlags::empty(),
        &[],
        &[],
        &[vk::ImageMemoryBarrier::default()
            .old_layout(old_layout)
            .new_layout(new_layout)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .level_count(1)
                    .layer_count(1),
            )
            .src_access_mask(src_access)
            .dst_access_mask(dst_access)],
    );
}

unsafe fn blit_rect(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    src: vk::Image,
    dst: vk::Image,
    src_rect: PresentRect,
    dst_rect: PresentRect,
) {
    let layers = vk::ImageSubresourceLayers::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .layer_count(1);
    device.cmd_blit_image(
        cmd,
        src,
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        dst,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        &[vk::ImageBlit::default()
            .src_subresource(layers)
            .src_offsets([
                vk::Offset3D {
                    x: src_rect.0 as i32,
                    y: src_rect.1 as i32,
                    z: 0,
                },
                vk::Offset3D {
                    x: src_rect.2 as i32,
                    y: src_rect.3 as i32,
                    z: 1,
                },
            ])
            .dst_subresource(layers)
            .dst_offsets([
                vk::Offset3D {
                    x: dst_rect.0 as i32,
                    y: dst_rect.1 as i32,
                    z: 0,
                },
                vk::Offset3D {
                    x: dst_rect.2 as i32,
                    y: dst_rect.3 as i32,
                    z: 1,
                },
            ])],
        crate::backend::vulkan::translate::sampler::PRESENT_BLIT_FILTER,
    );
}

fn scale_edge(value: u32, source: u32, destination: u32) -> u32 {
    (value as u64)
        .saturating_mul(destination as u64)
        .checked_div(source.max(1) as u64)
        .unwrap_or(0) as u32
}

fn scale_edge_ceil(value: u32, source: u32, destination: u32) -> u32 {
    let numerator = (value as u64).saturating_mul(destination as u64);
    numerator
        .saturating_add(source.max(1) as u64 - 1)
        .checked_div(source.max(1) as u64)
        .unwrap_or(0)
        .min(destination as u64) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scaled_damage_edges_cover_the_source_rect() {
        assert_eq!(scale_edge(1, 3, 10), 3);
        assert_eq!(scale_edge_ceil(2, 3, 10), 7);
        assert_eq!(scale_edge(0, 0, 10), 0);
        assert_eq!(scale_edge_ceil(5, 0, 10), 10);
    }

    fn full_vp(width: u32, height: u32) -> crate::host_window::viewport::Viewport {
        crate::host_window::viewport::Viewport {
            x: 0,
            y: 0,
            width,
            height,
        }
    }

    #[test]
    fn peer_rects_are_endpoints_not_extents() {
        // A lower-right-quadrant endpoint rect maps to itself at equal
        // source/destination geometry.
        assert_eq!(
            map_peer_rect((960, 540, 1920, 1080), (1920, 1080), &full_vp(1920, 1080)),
            Some(((960, 540, 1920, 1080), (960, 540, 1920, 1080)))
        );
        // A non-edge rect stays in place — the historical (x,y,w,h) misread
        // turned this into a displaced (100,100,300,300) blit.
        assert_eq!(
            map_peer_rect((100, 100, 200, 200), (1920, 1080), &full_vp(1920, 1080)),
            Some(((100, 100, 200, 200), (100, 100, 200, 200)))
        );
    }

    #[test]
    fn peer_rects_clip_to_bounds_and_reject_empty() {
        assert_eq!(
            map_peer_rect((1000, 500, 4000, 4000), (1920, 1080), &full_vp(1920, 1080)),
            Some(((1000, 500, 1920, 1080), (1000, 500, 1920, 1080)))
        );
        // Scaling into a half-size destination keeps endpoint coverage.
        assert_eq!(
            map_peer_rect((0, 0, 1920, 1080), (1920, 1080), &full_vp(960, 540)),
            Some(((0, 0, 1920, 1080), (0, 0, 960, 540)))
        );
        assert_eq!(
            map_peer_rect((5, 5, 5, 9), (1920, 1080), &full_vp(1920, 1080)),
            None
        );
        assert_eq!(
            map_peer_rect((2000, 0, 2100, 50), (1920, 1080), &full_vp(1920, 1080)),
            None
        );
    }

    #[test]
    fn peer_rects_offset_into_a_letterboxed_viewport() {
        // 1440x1080 guest pillarboxed into a 1920x1080 drawable at x=240.
        let vp = crate::host_window::viewport::aspect_fit((1440, 1080), (1920, 1080));
        assert_eq!(
            map_peer_rect((0, 0, 1440, 1080), (1440, 1080), &vp),
            Some(((0, 0, 1440, 1080), (240, 0, 1680, 1080)))
        );
        assert_eq!(
            map_peer_rect((100, 100, 200, 200), (1440, 1080), &vp),
            Some(((100, 100, 200, 200), (340, 100, 440, 200)))
        );
    }

    #[test]
    fn swapchain_recreation_line_names_geometry_and_reason() {
        let from = vk::Extent2D {
            width: 1920,
            height: 1080,
        };
        let to = vk::Extent2D {
            width: 1440,
            height: 1080,
        };
        assert_eq!(
            swapchain_recreated_line(from, to, "resize"),
            "host_window_swapchain status=recreated from=1920x1080 to=1440x1080 trigger=resize"
        );
    }

    #[test]
    fn cadence_proxy_reports_actual_queue_presents_and_direct_fraction() {
        let line = window_cadence_line(1_000, 120, 119, 131);
        assert!(line.contains("presents=120"), "{line}");
        assert!(line.contains("direct=119"), "{line}");
        assert!(line.contains("busy=131"), "{line}");
        assert!(line.contains("present_hz=120.0"), "{line}");
        assert!(line.contains("direct_frac=0.99"), "{line}");
    }

    fn ready(width: u32, height: u32) -> CandidateState {
        CandidateState {
            resident: true,
            content_ready: true,
            bgra: true,
            width,
            height,
        }
    }

    /// No published source is the expected pre-boundary / idle case and must be
    /// distinguishable from a source whose residents are missing.
    #[test]
    fn slate_without_a_source_is_named_separately() {
        assert_eq!(classify_slate(false, (0, 0), &[]), SlateReason::NoSource);
        assert_eq!(
            classify_slate(true, (1440, 1080), &[CandidateState::default()]),
            SlateReason::NoResident
        );
    }

    /// A resident that exists but has not landed content yet is the boot-era
    /// case; it must not be reported as a missing resident.
    #[test]
    fn unready_resident_reports_content_not_ready() {
        let pending = CandidateState {
            resident: true,
            content_ready: false,
            bgra: true,
            width: 1440,
            height: 1080,
        };
        assert_eq!(
            classify_slate(true, (1440, 1080), &[pending]),
            SlateReason::ContentNotReady
        );
    }

    /// The blocker CLOSEST to success wins: a ready BGRA resident at the wrong
    /// geometry outranks a sibling that was never created, because the geometry
    /// is the actionable fact.
    #[test]
    fn geometry_mismatch_outranks_a_missing_sibling() {
        let states = [CandidateState::default(), ready(1920, 1080)];
        assert_eq!(
            classify_slate(true, (1440, 1080), &states),
            SlateReason::GeomMismatch
        );
    }

    /// A ready non-BGRA resident is its own class — the present blit does no
    /// format conversion, so collapsing it into content_not_ready would send a
    /// reader hunting the wrong bug.
    #[test]
    fn non_bgra_resident_is_its_own_reason() {
        let states = [CandidateState {
            resident: true,
            content_ready: true,
            bgra: false,
            width: 1440,
            height: 1080,
        }];
        assert_eq!(
            classify_slate(true, (1440, 1080), &states),
            SlateReason::NotBgra
        );
    }

    /// Every reason has a distinct, `slate_`-prefixed slug.
    ///
    /// Distinctness is now also covered crate-wide by
    /// `observe::gate::every_registered_slug_is_unique_crate_wide`; what stays
    /// local is the **prefix**, which is what keeps a grep for this window's
    /// blit choice from also matching the capture rail's `no_resident_content`
    /// and the `THRASH geom_mismatch` proxy.
    #[test]
    fn slate_reason_slugs_are_distinct_and_namespaced() {
        use crate::observe::Decline;
        let mut slugs = [
            SlateReason::NoSource,
            SlateReason::NoResident,
            SlateReason::ContentNotReady,
            SlateReason::NotBgra,
            SlateReason::GeomMismatch,
        ]
        .map(|r| r.slug());
        for s in slugs {
            assert!(s.starts_with("slate_"), "{s} is not namespaced");
        }
        slugs.sort_unstable();
        let unique = slugs.len();
        let mut dedup = slugs.to_vec();
        dedup.dedup();
        assert_eq!(dedup.len(), unique);
    }

    #[test]
    fn non_aborting_present_degradations_keep_exact_geometry() {
        use crate::observe::Decline as _;
        let peer = WindowPresentDecline::PeerRectOutOfBounds {
            rect: (1, 2, 65, 66),
            width: 64,
            height: 64,
        };
        assert_eq!(
            crate::observe::Emit::decline("host_window_present", &peer).render(),
            "host_window_present reason=window_present_peer_rect_out_of_bounds \
             x0=1 y0=2 x1=65 y1=66 width=64 height=64"
        );

        let suboptimal = WindowPresentDecline::SuboptimalPersistent {
            streak: 60,
            width: 1440,
            height: 1080,
        };
        assert_eq!(suboptimal.slug(), "window_present_suboptimal_persistent");
        assert_eq!(
            suboptimal.fields(),
            vec![
                ("streak", "60".into()),
                ("width", "1440".into()),
                ("height", "1080".into()),
            ]
        );
    }

    /// A matching resident is never classified as slate — the classifier only
    /// runs on the failure path, but a caller reordering that check must not
    /// silently start reporting healthy frames.
    #[test]
    fn a_matching_ready_resident_still_reports_only_lesser_blockers() {
        // With one perfect candidate and one absent sibling, no
        // higher-severity reason fires; ContentNotReady is the residual.
        let states = [ready(1440, 1080), CandidateState::default()];
        assert_eq!(
            classify_slate(true, (1440, 1080), &states),
            SlateReason::ContentNotReady
        );
    }
}
