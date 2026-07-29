//! dmabuf scanout export/import — carry a finished BGRA present image between
//! the engine device and the host window's device as a Linux dmabuf fd, keeping
//! the frame on the GPU (no guest-page read + CPU surface copy on the display
//! path). This is the transport under direct-present route B.
//!
//! Both halves live here. The **producer** creates an exportable image and pulls
//! its dmabuf fd ([`export_bgra_scanout_dmabuf`], ringed by
//! [`ScanoutExportRing`]); the finished present target is blitted into it by
//! [`record_blit_present_into_export`]. The **consumer**
//! ([`import_bgra_dmabuf_image`]) imports that fd on the window's own device as a
//! `TRANSFER_SRC` image to blit into the swapchain.
//!
//! The image is `LINEAR`-tiled so the exported dmabuf carries the implicit
//! `DRM_FORMAT_MOD_LINEAR` modifier — a known stride with no explicit-modifier
//! negotiation. Optimal-tiled export via `VK_EXT_image_drm_format_modifier` is a
//! later refinement.
//!
//! Capability is gated on [`DeviceContext::ext_external_memory_fd`]
//! (`VK_KHR_external_memory_fd` + `VK_EXT_external_memory_dma_buf` — both are the
//! portable, Mesa/Intel/AMD/NVIDIA-supported dmabuf path; verified present on the
//! RTX 5080). When absent, callers keep the CPU staging path.
//!
//! Portability note: LINEAR-tiled export uses the implicit `DRM_FORMAT_MOD_LINEAR`
//! modifier, which every driver exports and imports. Explicit optimal-tiling
//! modifiers (`VK_EXT_image_drm_format_modifier`) differ per vendor, so LINEAR is
//! the portable baseline (see the AGENTS.md portability ground rule).
#![allow(dead_code)]

use ash::vk;

use super::context::DeviceContext;
use super::types::DrawError;
use super::vk_call::{VkCall, VkOp};
use crate::backend::vulkan::caps::MemoryClass;
use crate::backend::vulkan::translate;

/// An exportable BGRA8 scanout image plus its dmabuf handle. The `fd` is an
/// OWNED file descriptor (the caller must `close(2)` it once QEMU has imported
/// the dmabuf, or on teardown) and `image`/`memory` must be destroyed via
/// [`ExportedScanoutImage::destroy`].
#[derive(Debug)]
pub(crate) struct ExportedScanoutImage {
    pub image: vk::Image,
    pub memory: vk::DeviceMemory,
    /// Owned dmabuf fd (`DRM_FORMAT_MOD_LINEAR`). `-1` never occurs on success.
    pub fd: i32,
    /// Bytes per row of the LINEAR image (the EGL import stride).
    pub row_pitch: u64,
    /// Total allocation size (bytes).
    pub size: u64,
    pub width: u32,
    pub height: u32,
}

impl ExportedScanoutImage {
    /// Destroy the Vulkan image + memory. Does NOT close `fd` — the dmabuf fd is
    /// an independent kernel handle whose lifetime is the importer's (QEMU);
    /// close it separately once the importer is done.
    ///
    /// # Safety
    /// `device` must be the context that produced this image, and no in-flight
    /// command buffer may still reference it.
    pub(crate) unsafe fn destroy(self, device: &ash::Device) {
        device.destroy_image(self.image, None);
        device.free_memory(self.memory, None);
    }
}

/// Create a `width`x`height` BGRA8 `LINEAR` image backed by exportable memory and
/// return it together with its dmabuf fd. `DrawError::Unsupported` when the
/// device lacks the export extensions; `DrawError::Vulkan` on a Vulkan failure.
///
/// The image usage is `TRANSFER_DST` only: the finished present target is blitted
/// INTO it, and the importer (EGL→GL) consumes it — Vulkan itself never samples
/// or renders it, so no color-attachment/sampled usage is needed (and `LINEAR`
/// images do not support color-attachment on most drivers).
///
/// # Safety
/// `ctx` must be a live device context; the caller owns the returned resources.
pub(crate) unsafe fn export_bgra_scanout_dmabuf(
    ctx: &DeviceContext,
    width: u32,
    height: u32,
) -> Result<ExportedScanoutImage, DrawError> {
    let fd_loader = ctx.ext_external_memory_fd.as_ref().ok_or({
        DrawError::Unsupported(super::reason::DrawReason::DmabufExportUnavailable)
    })?;
    let handle = vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT;

    let mut ext_img = vk::ExternalMemoryImageCreateInfo::default().handle_types(handle);
    let image = ctx
        .device
        .create_image(
            &vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(translate::pixel::SCANOUT_FORMAT)
                .extent(vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::LINEAR)
                .usage(vk::ImageUsageFlags::TRANSFER_DST)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .push_next(&mut ext_img),
            None,
        )
        .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::DmabufExportCreateImage, e)))?;

    let req = ctx.device.get_image_memory_requirements(image);
    // Prefer device-local (NVIDIA exports device-local dmabufs); fall back to any
    // compatible type so the export never fails purely on placement.
    let mem_type = ctx
        .memory_type_for(req.memory_type_bits, MemoryClass::DeviceLocalPreferred)
        .ok_or_else(|| {
            ctx.device.destroy_image(image, None);
            DrawError::Unsupported(super::reason::DrawReason::NoMemoryTypeForScanoutExport {
                memory_type_bits: req.memory_type_bits,
            })
        })?;

    let mut export_info = vk::ExportMemoryAllocateInfo::default().handle_types(handle);
    let memory = ctx
        .device
        .allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(req.size)
                .memory_type_index(mem_type)
                .push_next(&mut export_info),
            None,
        )
        .map_err(|e| {
            ctx.device.destroy_image(image, None);
            DrawError::VkCall(VkCall::new(VkOp::DmabufExportAlloc, e))
        })?;

    if let Err(e) = ctx.device.bind_image_memory(image, memory, 0) {
        ctx.device.free_memory(memory, None);
        ctx.device.destroy_image(image, None);
        return Err(DrawError::VkCall(VkCall::new(VkOp::DmabufExportBind, e)));
    }

    // LINEAR row stride the importer needs (implicit-modifier dmabuf).
    let layout = ctx.device.get_image_subresource_layout(
        image,
        vk::ImageSubresource {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            array_layer: 0,
        },
    );

    let fd = fd_loader
        .get_memory_fd(
            &vk::MemoryGetFdInfoKHR::default()
                .memory(memory)
                .handle_type(handle),
        )
        .map_err(|e| {
            ctx.device.free_memory(memory, None);
            ctx.device.destroy_image(image, None);
            DrawError::VkCall(VkCall::new(VkOp::DmabufExportGetFd, e))
        })?;

    Ok(ExportedScanoutImage {
        image,
        memory,
        fd,
        row_pitch: layout.row_pitch,
        size: req.size,
        width,
        height,
    })
}

/// An imported dmabuf as a `TRANSFER_SRC` BGRA8 `LINEAR` VkImage — the consumer
/// half of [`export_bgra_scanout_dmabuf`]. Used by the host window (route B of the
/// direct-present increment) to import the engine's exported present frame and
/// blit it into the swapchain, with no CPU readback/upload. Destroy via
/// [`ImportedDmabufImage::destroy`].
#[derive(Debug)]
pub(crate) struct ImportedDmabufImage {
    pub image: vk::Image,
    pub memory: vk::DeviceMemory,
    pub width: u32,
    pub height: u32,
}

impl ImportedDmabufImage {
    /// Destroy the image + memory. Freeing `memory` also closes the imported
    /// dmabuf fd (Vulkan took ownership at import), so do NOT close that fd
    /// separately.
    ///
    /// # Safety
    /// `device` must be the one that imported this image, and no in-flight
    /// command buffer may still reference it.
    pub(crate) unsafe fn destroy(self, device: &ash::Device) {
        device.destroy_image(self.image, None);
        device.free_memory(self.memory, None);
    }
}

/// Import a LINEAR BGRA8 dmabuf `fd` (as produced by
/// [`export_bgra_scanout_dmabuf`]) as a `TRANSFER_SRC` VkImage on `device`, so it
/// can be blitted straight into a swapchain image. `fd_loader` is the device's
/// `VK_KHR_external_memory_fd` loader (for `vkGetMemoryFdPropertiesKHR`) and
/// `mem_props` its physical-device memory properties.
///
/// On **success** Vulkan takes ownership of `fd` (freed with the returned
/// memory); the caller must not touch it. On **failure** the fd is left untouched
/// so the caller can close it. `DrawError::Vulkan` on any Vulkan failure.
///
/// # Safety
/// `device` / `fd_loader` / `mem_props` must belong to the same live physical
/// device, and `fd` must be a dmabuf fd for a `width`x`height` LINEAR BGRA8 image.
pub(crate) unsafe fn import_bgra_dmabuf_image(
    device: &ash::Device,
    fd_loader: &ash::khr::external_memory_fd::Device,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    fd: i32,
    width: u32,
    height: u32,
) -> Result<ImportedDmabufImage, DrawError> {
    let handle = vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT;
    let mut ext_img = vk::ExternalMemoryImageCreateInfo::default().handle_types(handle);
    let image = device
        .create_image(
            &vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(translate::pixel::SCANOUT_FORMAT)
                .extent(vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::LINEAR)
                .usage(vk::ImageUsageFlags::TRANSFER_SRC)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .push_next(&mut ext_img),
            None,
        )
        .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::DmabufImportCreateImage, e)))?;

    let req = device.get_image_memory_requirements(image);
    // The kernel restricts which memory types can back THIS dmabuf; the valid
    // set is the intersection with the image's own requirements (Vulkan spec).
    let mut fd_props = vk::MemoryFdPropertiesKHR::default();
    fd_loader
        .get_memory_fd_properties(handle, fd, &mut fd_props)
        .map_err(|e| {
            device.destroy_image(image, None);
            DrawError::VkCall(VkCall::new(VkOp::DmabufImportFdProps, e))
        })?;
    let allowed = req.memory_type_bits & fd_props.memory_type_bits;
    let mem_type = (0..mem_props.memory_type_count)
        .find(|&i| (allowed & (1 << i)) != 0)
        .ok_or_else(|| {
            device.destroy_image(image, None);
            DrawError::Unsupported(super::reason::DrawReason::NoMemoryTypeForDmabufImport {
                memory_type_bits: allowed,
            })
        })?;

    // dmabuf image import needs a dedicated allocation on most drivers; importing
    // an fd transfers its ownership to Vulkan on a successful allocate.
    let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().image(image);
    let mut import = vk::ImportMemoryFdInfoKHR::default()
        .handle_type(handle)
        .fd(fd);
    let memory = device
        .allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(req.size)
                .memory_type_index(mem_type)
                .push_next(&mut dedicated)
                .push_next(&mut import),
            None,
        )
        .map_err(|e| {
            device.destroy_image(image, None);
            DrawError::VkCall(VkCall::new(VkOp::DmabufImportAlloc, e))
        })?;
    if let Err(e) = device.bind_image_memory(image, memory, 0) {
        device.free_memory(memory, None);
        device.destroy_image(image, None);
        return Err(DrawError::VkCall(VkCall::new(VkOp::DmabufImportBind, e)));
    }
    Ok(ImportedDmabufImage {
        image,
        memory,
        width,
        height,
    })
}

/// Record a same-size BGRA copy from the finished present target into the
/// exportable scanout image, on an already-begun command buffer `cb`. This is
/// the producer step the display path runs each present: the frame stays on the
/// GPU (target image → export image), no guest-page read + CPU surface copy.
///
/// `src_layout` is the present target's current layout (the pass leaves slot 0
/// at `TRANSFER_SRC_OPTIMAL`, so the caller normally passes that — a redundant
/// same-layout barrier is legal and harmless). `vkCmdCopyImage` (not blit) is
/// used because source and export share dimensions and the BGRA8 format, so no
/// scaling/filtering is needed; a copy also works on the `LINEAR` export image,
/// which most drivers do not accept as a blit destination.
///
/// After the copy the export image is left in `GENERAL` layout with its transfer
/// write made available to later reads — valid for a subsequent
/// `vkCmdCopyImageToBuffer` readback AND for an external dmabuf consumer. (A real
/// boot-path present additionally releases queue-family ownership to
/// `QUEUE_FAMILY_EXTERNAL` before the EGL import; that transfer is added when the
/// display half is wired, since it needs the importing queue on the QEMU side.)
///
/// # Safety
/// `cb` must be in the recording state on `device`'s graphics queue; `src_image`
/// and `export.image` must be live and not otherwise in flight.
pub(crate) unsafe fn record_blit_present_into_export(
    device: &ash::Device,
    cb: vk::CommandBuffer,
    src_image: vk::Image,
    src_layout: vk::ImageLayout,
    export: &ExportedScanoutImage,
) {
    let color_range = vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_mip_level(0)
        .level_count(1)
        .base_array_layer(0)
        .layer_count(1);

    // Export UNDEFINED → TRANSFER_DST (we overwrite every texel, so UNDEFINED
    // discards its prior contents), and source → TRANSFER_SRC for the read.
    let pre = [
        vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(export.image)
            .subresource_range(color_range),
        vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
            .old_layout(src_layout)
            .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(src_image)
            .subresource_range(color_range),
    ];
    device.cmd_pipeline_barrier(
        cb,
        vk::PipelineStageFlags::ALL_COMMANDS,
        vk::PipelineStageFlags::TRANSFER,
        vk::DependencyFlags::empty(),
        &[],
        &[],
        &pre,
    );

    let sub = vk::ImageSubresourceLayers::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .mip_level(0)
        .base_array_layer(0)
        .layer_count(1);
    let copy = vk::ImageCopy::default()
        .src_subresource(sub)
        .dst_subresource(sub)
        .extent(vk::Extent3D {
            width: export.width,
            height: export.height,
            depth: 1,
        });
    device.cmd_copy_image(
        cb,
        src_image,
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        export.image,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        &[copy],
    );

    // Export → GENERAL, making the transfer write available to any later read
    // (readback copy in tests; the external dmabuf consumer at boot).
    let post = [vk::ImageMemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::MEMORY_READ)
        .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .new_layout(vk::ImageLayout::GENERAL)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(export.image)
        .subresource_range(color_range)];
    device.cmd_pipeline_barrier(
        cb,
        vk::PipelineStageFlags::TRANSFER,
        vk::PipelineStageFlags::ALL_COMMANDS,
        vk::DependencyFlags::empty(),
        &[],
        &[],
        &post,
    );
}

/// Number of exportable images the direct-present ring keeps. The engine blits
/// frame N into slot `N % RING` while the window is still reading slot
/// `(N-1) % RING`, so a slot is only reused RING presents after it was last
/// read — the window's blit of it is long GPU-complete by then. This is what
/// makes cross-device dmabuf sharing tear-safe WITHOUT a shared semaphore: no
/// slot is ever written by the producer while the consumer reads it. Matches
/// the swapchain depth so the window can hold one import per slot.
pub(crate) const SCANOUT_EXPORT_RING: usize = 3;

/// A small ring of exportable scanout images for the direct-present path
/// ([[host-window]] route B). It hands a DIFFERENT slot each present so the
/// engine never overwrites the slot the host window is mid-blit reading — a
/// single reused image would tear against a consumer on its own frame pacing.
///
/// Lifecycle is caller-owned by design, so the ring stays decoupled from the
/// engine's fence/ring bookkeeping: [`acquire_next`] returns any images
/// displaced by a geometry change for the caller to dispose once no in-flight
/// work references them; the ring never frees a live image.
pub(crate) struct ScanoutExportRing {
    slots: [Option<ExportedScanoutImage>; SCANOUT_EXPORT_RING],
    /// Next slot to hand out.
    next: usize,
    width: u32,
    height: u32,
}

impl Default for ScanoutExportRing {
    fn default() -> Self {
        Self::new()
    }
}

impl ScanoutExportRing {
    pub(crate) fn new() -> Self {
        Self {
            slots: Default::default(),
            next: 0,
            width: 0,
            height: 0,
        }
    }

    /// Advance to the next ring slot at `width`x`height`, creating that slot's
    /// image if absent. On a geometry change every existing slot is retired and
    /// returned (with its canonical fd) for the caller to destroy + close once no
    /// in-flight command buffer references it; the ring rebuilds lazily. Returns
    /// `(slot_index, retired)`. After this returns `Ok`, [`slot`](Self::slot)
    /// with the returned index is the image to blit into + export.
    ///
    /// # Safety
    /// `ctx` must be a live device context; retired images are the caller's to
    /// destroy and whose fds to close.
    pub(crate) unsafe fn acquire_next(
        &mut self,
        ctx: &DeviceContext,
        width: u32,
        height: u32,
    ) -> Result<(usize, Vec<ExportedScanoutImage>), DrawError> {
        let mut retired = Vec::new();
        if self.width != width || self.height != height {
            for s in self.slots.iter_mut() {
                if let Some(img) = s.take() {
                    retired.push(img);
                }
            }
            self.width = width;
            self.height = height;
            self.next = 0;
        }
        let idx = self.next;
        if self.slots[idx].is_none() {
            self.slots[idx] = Some(export_bgra_scanout_dmabuf(ctx, width, height)?);
        }
        self.next = (self.next + 1) % SCANOUT_EXPORT_RING;
        Ok((idx, retired))
    }

    /// The image at `idx` (must have been established by a prior
    /// [`acquire_next`] that returned `idx`).
    pub(crate) fn slot(&self, idx: usize) -> &ExportedScanoutImage {
        self.slots[idx]
            .as_ref()
            .expect("ring slot established by acquire_next")
    }

    /// Destroy every slot image + free its memory, returning their dmabuf fds for
    /// the caller to `close(2)`.
    ///
    /// # Safety
    /// `device` must be the context that produced the images and no in-flight
    /// command buffer may still reference them.
    pub(crate) unsafe fn destroy(&mut self, device: &ash::Device) -> Vec<i32> {
        let mut fds = Vec::new();
        for s in self.slots.iter_mut() {
            if let Some(img) = s.take() {
                fds.push(img.fd);
                img.destroy(device);
            }
        }
        self.width = 0;
        self.height = 0;
        self.next = 0;
        fds
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Proves the host GPU actually hands us a usable dmabuf fd for a BGRA
    /// scanout image — the biggest technical unknown for the GL/dmabuf scanout
    /// export lever (NVIDIA dmabuf export is historically limited). Skips cleanly
    /// when no GPU / no export extension is present, so it is safe in CI.
    #[test]
    fn exports_usable_bgra_dmabuf_fd() {
        crate::observe::redirect_logs_for_tests();
        let mut ctx = match unsafe { DeviceContext::create() } {
            Ok(c) => c,
            Err(e) => {
                eprintln!("SKIP dmabuf export: no device ({e})");
                return;
            }
        };
        if ctx.ext_external_memory_fd.is_none() {
            eprintln!("SKIP dmabuf export: extensions not enabled on this device");
            unsafe { ctx.destroy() };
            return;
        }
        let (w, h) = (256u32, 128u32);
        let exported = unsafe { export_bgra_scanout_dmabuf(&ctx, w, h) }
            .expect("dmabuf export must succeed when the extensions are enabled");
        assert!(exported.fd >= 0, "exported dmabuf fd must be valid");
        assert_eq!(exported.width, w);
        assert_eq!(exported.height, h);
        // LINEAR BGRA8: one row is at least width*4 bytes.
        assert!(
            exported.row_pitch >= (w as u64) * 4,
            "row_pitch {} must cover {}px * 4 bytes",
            exported.row_pitch,
            w
        );
        assert!(
            exported.size >= exported.row_pitch * (h as u64),
            "allocation {} must cover the full LINEAR image",
            exported.size
        );
        // The fd is a real kernel handle we own — wrap it in an OwnedFd so it is
        // closed on drop (proves it is a genuine, closeable descriptor without a
        // libc dependency).
        {
            use std::os::fd::FromRawFd;
            let _owned = unsafe { std::os::fd::OwnedFd::from_raw_fd(exported.fd) };
        }
        unsafe {
            exported.destroy(&ctx.device);
            ctx.destroy();
        }
    }

    /// Proves the CONSUMER half of route B works on the host GPU: a BGRA dmabuf
    /// fd this device exported re-imports as a usable `TRANSFER_SRC` VkImage on the
    /// SAME device. This is the key unknown for host-window direct-present route B
    /// (can the window import the engine's exported present frame as an image);
    /// self-import is the strictest same-driver case. Skips cleanly with no GPU /
    /// no export extensions.
    #[test]
    fn import_exported_dmabuf_as_image_roundtrip() {
        crate::observe::redirect_logs_for_tests();
        let mut ctx = match unsafe { DeviceContext::create() } {
            Ok(c) => c,
            Err(e) => {
                eprintln!("SKIP dmabuf import: no device ({e})");
                return;
            }
        };
        if ctx.ext_external_memory_fd.is_none() {
            eprintln!("SKIP dmabuf import: extensions not enabled on this device");
            unsafe { ctx.destroy() };
            return;
        }
        let (w, h) = (128u32, 64u32);
        let exported = unsafe { export_bgra_scanout_dmabuf(&ctx, w, h) }
            .expect("export must succeed when the extensions are enabled");
        assert!(exported.fd >= 0, "exported dmabuf fd must be valid");
        let mem_props = unsafe { ctx.instance.get_physical_device_memory_properties(ctx.pd) };
        let fd_loader = ctx.ext_external_memory_fd.as_ref().unwrap();
        // Importing consumes `exported.fd` (Vulkan owns it after success); the
        // `exported` image's own memory is separate and freed below.
        let imported = unsafe {
            import_bgra_dmabuf_image(&ctx.device, fd_loader, &mem_props, exported.fd, w, h)
        }
        .expect("importing our own exported dmabuf as an image must succeed");
        assert_eq!(imported.width, w);
        assert_eq!(imported.height, h);
        assert_ne!(imported.image, vk::Image::null());
        assert_ne!(imported.memory, vk::DeviceMemory::null());
        unsafe {
            imported.destroy(&ctx.device); // frees memory + closes the imported fd
            exported.destroy(&ctx.device); // image + memory only (fd now Vulkan's)
            ctx.destroy();
        }
    }

    /// Proves the FULL route-B chain across TWO independent logical devices — the
    /// real host-window topology, where the window owns a `VkDevice` distinct from
    /// the engine's. Device A (engine) writes a known BGRA pattern into its
    /// exported dmabuf image; device B (window) imports that fd on a SEPARATE
    /// `VkDevice` on the same physical GPU and reads the shared image back
    /// BYTE-IDENTICAL. Self-import (same device) can hide driver-internal aliasing
    /// a distinct device would expose, so this is the strict viability gate. It
    /// also pins the consumer acquire-barrier that the window's `blit_ring_slot`
    /// must use: an EXTERNAL→graphics ownership acquire from `GENERAL` (the
    /// producer's layout) — NOT `UNDEFINED`, which would discard the imported
    /// content. Skips cleanly with no GPU / no export extensions.
    #[test]
    fn cross_device_dmabuf_import_is_byte_identical() {
        crate::observe::redirect_logs_for_tests();
        let mut ctx = match unsafe { DeviceContext::create() } {
            Ok(c) => c,
            Err(e) => {
                eprintln!("SKIP cross-device dmabuf: no device ({e})");
                return;
            }
        };
        if ctx.ext_external_memory_fd.is_none() {
            eprintln!("SKIP cross-device dmabuf: extensions not enabled on this device");
            unsafe { ctx.destroy() };
            return;
        }
        let (w, h) = (64u32, 48u32);
        let texels = (w * h) as usize;
        let bytes = texels * 4;
        // Deterministic BGRA pattern (no Math::random in tests).
        let src_pixels: Vec<u8> = (0..texels)
            .flat_map(|i| {
                [
                    (i & 0xff) as u8,
                    ((i >> 8) & 0xff) as u8,
                    ((i * 13) & 0xff) as u8,
                    0xff,
                ]
            })
            .collect();

        unsafe {
            let color_layers = vk::ImageSubresourceLayers::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .layer_count(1);
            let color_range = vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .level_count(1)
                .layer_count(1);

            // === Device A: write the known pattern into the exported dmabuf ===
            let dev_a = &ctx.device;
            let src_image = dev_a
                .create_image(
                    &vk::ImageCreateInfo::default()
                        .image_type(vk::ImageType::TYPE_2D)
                        .format(translate::pixel::SCANOUT_FORMAT)
                        .extent(vk::Extent3D {
                            width: w,
                            height: h,
                            depth: 1,
                        })
                        .mip_levels(1)
                        .array_layers(1)
                        .samples(vk::SampleCountFlags::TYPE_1)
                        .tiling(vk::ImageTiling::OPTIMAL)
                        .usage(
                            vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC,
                        )
                        .initial_layout(vk::ImageLayout::UNDEFINED),
                    None,
                )
                .expect("create A source image");
            let src_req = dev_a.get_image_memory_requirements(src_image);
            let src_mt = ctx
                .memory_type_for(src_req.memory_type_bits, MemoryClass::DeviceLocalPreferred)
                .expect("A source memory type");
            let src_mem = dev_a
                .allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(src_req.size)
                        .memory_type_index(src_mt),
                    None,
                )
                .expect("alloc A source mem");
            dev_a
                .bind_image_memory(src_image, src_mem, 0)
                .expect("bind A source");
            // Host-visible upload buffer, seeded with the pattern (same pd, so
            // ctx.find_memory_type serves both devices' host-visible needs).
            let up_buf = dev_a
                .create_buffer(
                    &vk::BufferCreateInfo::default()
                        .size(bytes as u64)
                        .usage(vk::BufferUsageFlags::TRANSFER_SRC),
                    None,
                )
                .expect("create A upload buffer");
            let up_req = dev_a.get_buffer_memory_requirements(up_buf);
            let up_mt = ctx
                .memory_type_for(up_req.memory_type_bits, MemoryClass::Upload)
                .expect("A upload memory type");
            let up_mem = dev_a
                .allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(up_req.size)
                        .memory_type_index(up_mt),
                    None,
                )
                .expect("alloc A upload mem");
            dev_a
                .bind_buffer_memory(up_buf, up_mem, 0)
                .expect("bind A upload");
            let up_ptr = dev_a
                .map_memory(up_mem, 0, up_req.size, vk::MemoryMapFlags::empty())
                .expect("map A upload") as *mut u8;
            std::ptr::copy_nonoverlapping(src_pixels.as_ptr(), up_ptr, bytes);
            dev_a.unmap_memory(up_mem);

            let exported = export_bgra_scanout_dmabuf(&ctx, w, h).expect("export image on A");

            let pool_a = dev_a
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default().queue_family_index(ctx.gq),
                    None,
                )
                .expect("A cmd pool");
            let cb_a = dev_a
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(pool_a)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
                .expect("alloc A cb")[0];
            dev_a
                .begin_command_buffer(
                    cb_a,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .expect("begin A cb");
            dev_a.cmd_pipeline_barrier(
                cb_a,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[vk::ImageMemoryBarrier::default()
                    .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(src_image)
                    .subresource_range(color_range)],
            );
            dev_a.cmd_copy_buffer_to_image(
                cb_a,
                up_buf,
                src_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[vk::BufferImageCopy::default()
                    .image_subresource(color_layers)
                    .image_extent(vk::Extent3D {
                        width: w,
                        height: h,
                        depth: 1,
                    })],
            );
            record_blit_present_into_export(
                dev_a,
                cb_a,
                src_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &exported,
            );
            dev_a.end_command_buffer(cb_a).expect("end A cb");
            let fence_a = dev_a
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .expect("A fence");
            dev_a
                .queue_submit(
                    ctx.queue(),
                    &[vk::SubmitInfo::default().command_buffers(&[cb_a])],
                    fence_a,
                )
                .expect("submit A");
            dev_a
                .wait_for_fences(&[fence_a], true, 5_000_000_000)
                .expect("wait A fence");

            // === Device B: a SEPARATE logical device imports the fd + reads it ===
            let dev_exts = [
                vk::KHR_EXTERNAL_MEMORY_NAME.as_ptr(),
                vk::KHR_EXTERNAL_MEMORY_FD_NAME.as_ptr(),
                vk::EXT_EXTERNAL_MEMORY_DMA_BUF_NAME.as_ptr(),
            ];
            let prio = [1.0f32];
            let qci = [vk::DeviceQueueCreateInfo::default()
                .queue_family_index(ctx.gq)
                .queue_priorities(&prio)];
            let dev_b = ctx
                .instance
                .create_device(
                    ctx.pd,
                    &vk::DeviceCreateInfo::default()
                        .queue_create_infos(&qci)
                        .enabled_extension_names(&dev_exts),
                    None,
                )
                .expect("create device B");
            let queue_b = dev_b.get_device_queue(ctx.gq, 0);
            let fd_loader_b = ash::khr::external_memory_fd::Device::new(&ctx.instance, &dev_b);
            let mem_props_b = ctx.instance.get_physical_device_memory_properties(ctx.pd);
            let imported =
                import_bgra_dmabuf_image(&dev_b, &fd_loader_b, &mem_props_b, exported.fd, w, h)
                    .expect("import exported dmabuf on device B");

            let rb_buf = dev_b
                .create_buffer(
                    &vk::BufferCreateInfo::default()
                        .size(bytes as u64)
                        .usage(vk::BufferUsageFlags::TRANSFER_DST),
                    None,
                )
                .expect("create B readback buffer");
            let rb_req = dev_b.get_buffer_memory_requirements(rb_buf);
            let rb_mt = ctx
                .memory_type_for(rb_req.memory_type_bits, MemoryClass::Readback)
                .expect("B readback memory type");
            let rb_mem = dev_b
                .allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(rb_req.size)
                        .memory_type_index(rb_mt),
                    None,
                )
                .expect("alloc B readback mem");
            dev_b
                .bind_buffer_memory(rb_buf, rb_mem, 0)
                .expect("bind B readback");

            let pool_b = dev_b
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default().queue_family_index(ctx.gq),
                    None,
                )
                .expect("B cmd pool");
            let cb_b = dev_b
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(pool_b)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
                .expect("alloc B cb")[0];
            dev_b
                .begin_command_buffer(
                    cb_b,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .expect("begin B cb");
            // Acquire ownership from the external producer, preserving content:
            // old_layout=GENERAL (what `record_blit_present_into_export` left the
            // shared memory in) NOT UNDEFINED. This is the exact barrier the
            // window's `blit_ring_slot` uses.
            dev_b.cmd_pipeline_barrier(
                cb_b,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::empty())
                    .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                    .old_layout(vk::ImageLayout::GENERAL)
                    .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_EXTERNAL)
                    .dst_queue_family_index(ctx.gq)
                    .image(imported.image)
                    .subresource_range(color_range)],
            );
            dev_b.cmd_copy_image_to_buffer(
                cb_b,
                imported.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                rb_buf,
                &[vk::BufferImageCopy::default()
                    .image_subresource(color_layers)
                    .image_extent(vk::Extent3D {
                        width: w,
                        height: h,
                        depth: 1,
                    })],
            );
            dev_b.end_command_buffer(cb_b).expect("end B cb");
            let fence_b = dev_b
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .expect("B fence");
            dev_b
                .queue_submit(
                    queue_b,
                    &[vk::SubmitInfo::default().command_buffers(&[cb_b])],
                    fence_b,
                )
                .expect("submit B");
            dev_b
                .wait_for_fences(&[fence_b], true, 5_000_000_000)
                .expect("wait B fence");

            let rptr = dev_b
                .map_memory(rb_mem, 0, rb_req.size, vk::MemoryMapFlags::empty())
                .expect("map B readback") as *const u8;
            let readback = std::slice::from_raw_parts(rptr, bytes).to_vec();
            dev_b.unmap_memory(rb_mem);
            assert_eq!(
                readback, src_pixels,
                "device B must read the engine's exported dmabuf byte-identically"
            );

            // Cleanup device B first (frees imported memory + closes the fd), then A.
            dev_b.destroy_fence(fence_b, None);
            dev_b.destroy_command_pool(pool_b, None);
            dev_b.destroy_buffer(rb_buf, None);
            dev_b.free_memory(rb_mem, None);
            imported.destroy(&dev_b);
            dev_b.destroy_device(None);

            dev_a.destroy_fence(fence_a, None);
            dev_a.destroy_command_pool(pool_a, None);
            dev_a.destroy_buffer(up_buf, None);
            dev_a.free_memory(up_mem, None);
            dev_a.destroy_image(src_image, None);
            dev_a.free_memory(src_mem, None);
            // The fd is now owned+closed by device B's import; exported.destroy
            // releases only A's image + memory.
            exported.destroy(&ctx.device);
            ctx.destroy();
        }
    }

    /// Proves the producer step is content-correct: a known BGRA pattern written
    /// into a present-target-shaped image, copied into the exportable scanout
    /// image via [`record_blit_present_into_export`], reads back byte-identical.
    /// This is the "readback == source" gate for the display half — the frame
    /// stays on the GPU end to end and the copy loses nothing. Skips cleanly with
    /// no GPU / no export extension.
    #[test]
    fn blit_present_into_export_is_byte_identical() {
        crate::observe::redirect_logs_for_tests();
        let mut ctx = match unsafe { DeviceContext::create() } {
            Ok(c) => c,
            Err(e) => {
                eprintln!("SKIP blit export: no device ({e})");
                return;
            }
        };
        if ctx.ext_external_memory_fd.is_none() {
            eprintln!("SKIP blit export: extensions not enabled on this device");
            unsafe { ctx.destroy() };
            return;
        }
        let (w, h) = (64u32, 48u32);
        let texels = (w * h) as usize;
        let bytes = texels * 4;
        // Deterministic BGRA pattern (no Math::random in scripts/tests).
        let src_pixels: Vec<u8> = (0..texels)
            .flat_map(|i| {
                [
                    (i & 0xff) as u8,
                    ((i >> 8) & 0xff) as u8,
                    ((i * 7) & 0xff) as u8,
                    0xff,
                ]
            })
            .collect();

        unsafe {
            let dev = &ctx.device;
            // --- source present-target-shaped image (OPTIMAL BGRA, xfer both ways)
            let src_image = dev
                .create_image(
                    &vk::ImageCreateInfo::default()
                        .image_type(vk::ImageType::TYPE_2D)
                        .format(translate::pixel::SCANOUT_FORMAT)
                        .extent(vk::Extent3D {
                            width: w,
                            height: h,
                            depth: 1,
                        })
                        .mip_levels(1)
                        .array_layers(1)
                        .samples(vk::SampleCountFlags::TYPE_1)
                        .tiling(vk::ImageTiling::OPTIMAL)
                        .usage(
                            vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC,
                        )
                        .initial_layout(vk::ImageLayout::UNDEFINED),
                    None,
                )
                .expect("create source image");
            let src_req = dev.get_image_memory_requirements(src_image);
            let src_mt = ctx
                .memory_type_for(src_req.memory_type_bits, MemoryClass::DeviceLocalPreferred)
                .expect("source memory type");
            let src_mem = dev
                .allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(src_req.size)
                        .memory_type_index(src_mt),
                    None,
                )
                .expect("alloc source mem");
            dev.bind_image_memory(src_image, src_mem, 0)
                .expect("bind source");

            // --- host-visible staging (upload) + readback buffers
            let make_buffer = |size: u64, usage: vk::BufferUsageFlags| {
                let buf = dev
                    .create_buffer(
                        &vk::BufferCreateInfo::default().size(size).usage(usage),
                        None,
                    )
                    .expect("create buffer");
                let req = dev.get_buffer_memory_requirements(buf);
                let mt = ctx
                    .memory_type_for(req.memory_type_bits, MemoryClass::Upload)
                    .expect("host-visible memory type");
                let mem = dev
                    .allocate_memory(
                        &vk::MemoryAllocateInfo::default()
                            .allocation_size(req.size)
                            .memory_type_index(mt),
                        None,
                    )
                    .expect("alloc buffer mem");
                dev.bind_buffer_memory(buf, mem, 0).expect("bind buffer");
                (buf, mem, req.size)
            };
            let (up_buf, up_mem, up_sz) =
                make_buffer(bytes as u64, vk::BufferUsageFlags::TRANSFER_SRC);
            let ptr = dev
                .map_memory(up_mem, 0, up_sz, vk::MemoryMapFlags::empty())
                .expect("map upload") as *mut u8;
            std::ptr::copy_nonoverlapping(src_pixels.as_ptr(), ptr, bytes);
            dev.unmap_memory(up_mem);
            let (rb_buf, rb_mem, rb_sz) =
                make_buffer(bytes as u64, vk::BufferUsageFlags::TRANSFER_DST);

            let exported = export_bgra_scanout_dmabuf(&ctx, w, h).expect("export image");

            // --- one-shot command buffer: upload → blit → readback
            let pool = dev
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default().queue_family_index(ctx.gq),
                    None,
                )
                .expect("cmd pool");
            let cb = dev
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
                .expect("alloc cb")[0];
            dev.begin_command_buffer(
                cb,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .expect("begin cb");

            let color_range = vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .level_count(1)
                .layer_count(1);
            // source UNDEFINED → TRANSFER_DST, upload it.
            dev.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[vk::ImageMemoryBarrier::default()
                    .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(src_image)
                    .subresource_range(color_range)],
            );
            let sub = vk::ImageSubresourceLayers::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .layer_count(1);
            dev.cmd_copy_buffer_to_image(
                cb,
                up_buf,
                src_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[vk::BufferImageCopy::default()
                    .image_subresource(sub)
                    .image_extent(vk::Extent3D {
                        width: w,
                        height: h,
                        depth: 1,
                    })],
            );
            // The producer step under test (source now at TRANSFER_DST).
            record_blit_present_into_export(
                dev,
                cb,
                src_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &exported,
            );
            // Readback: export (GENERAL) → host buffer, tightly packed.
            dev.cmd_copy_image_to_buffer(
                cb,
                exported.image,
                vk::ImageLayout::GENERAL,
                rb_buf,
                &[vk::BufferImageCopy::default()
                    .image_subresource(sub)
                    .image_extent(vk::Extent3D {
                        width: w,
                        height: h,
                        depth: 1,
                    })],
            );
            dev.end_command_buffer(cb).expect("end cb");

            let fence = dev
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .expect("fence");
            let cbs = [cb];
            dev.queue_submit(
                ctx.queue(),
                &[vk::SubmitInfo::default().command_buffers(&cbs)],
                fence,
            )
            .expect("submit");
            dev.wait_for_fences(&[fence], true, 5_000_000_000)
                .expect("wait fence");

            let rptr = dev
                .map_memory(rb_mem, 0, rb_sz, vk::MemoryMapFlags::empty())
                .expect("map readback") as *const u8;
            let readback = std::slice::from_raw_parts(rptr, bytes).to_vec();
            dev.unmap_memory(rb_mem);
            assert_eq!(
                readback, src_pixels,
                "export image must be byte-identical to source"
            );

            // Cleanup.
            dev.destroy_fence(fence, None);
            dev.destroy_command_pool(pool, None);
            dev.destroy_buffer(up_buf, None);
            dev.free_memory(up_mem, None);
            dev.destroy_buffer(rb_buf, None);
            dev.free_memory(rb_mem, None);
            dev.destroy_image(src_image, None);
            dev.free_memory(src_mem, None);
            {
                use std::os::fd::FromRawFd;
                let _owned = std::os::fd::OwnedFd::from_raw_fd(exported.fd);
            }
            exported.destroy(&ctx.device);
            ctx.destroy();
        }
    }

    /// The direct-present ring hands a DIFFERENT slot each present (tear-safety:
    /// the engine never overwrites the slot the window is mid-blit reading),
    /// cycles back after `SCANOUT_EXPORT_RING` slots reusing the same fd (no
    /// per-frame churn), and retires ALL slots on a geometry change.
    #[test]
    fn export_ring_alternates_distinct_slots_and_retires_on_resize() {
        use std::os::fd::{FromRawFd, OwnedFd};
        crate::observe::redirect_logs_for_tests();
        let mut ctx = match unsafe { DeviceContext::create() } {
            Ok(c) => c,
            Err(e) => {
                eprintln!("SKIP export ring: no device ({e})");
                return;
            }
        };
        if ctx.ext_external_memory_fd.is_none() {
            eprintln!("SKIP export ring: extensions not enabled on this device");
            unsafe { ctx.destroy() };
            return;
        }
        unsafe {
            let mut ring = ScanoutExportRing::new();
            // Fill every slot once: distinct indices, distinct images + fds,
            // nothing retired.
            let mut images = Vec::new();
            let mut fds = Vec::new();
            for expect_idx in 0..SCANOUT_EXPORT_RING {
                let (idx, retired) = ring.acquire_next(&ctx, 128, 64).expect("acquire fill");
                assert_eq!(idx, expect_idx, "ring hands slots in order while filling");
                assert!(retired.is_empty(), "filling a fresh slot retires nothing");
                let s = ring.slot(idx);
                assert_eq!((s.width, s.height), (128, 64));
                assert!(s.fd >= 0);
                images.push(s.image);
                fds.push(s.fd);
            }
            for i in 0..SCANOUT_EXPORT_RING {
                for j in (i + 1)..SCANOUT_EXPORT_RING {
                    assert_ne!(images[i], images[j], "ring slots are distinct images");
                    assert_ne!(fds[i], fds[j], "ring slots have distinct fds");
                }
            }

            // Cycle: the next acquire returns slot 0 again, same image+fd (no
            // churn), nothing retired.
            let (idx, retired) = ring.acquire_next(&ctx, 128, 64).expect("acquire cycle");
            assert_eq!(idx, 0, "ring cycles back to slot 0");
            assert!(retired.is_empty(), "reused slot retires nothing");
            assert_eq!(
                ring.slot(0).image,
                images[0],
                "cycled slot reuses its image"
            );
            assert_eq!(ring.slot(0).fd, fds[0], "cycled slot reuses its fd");

            // Geometry change retires every populated slot for disposal.
            let (idx, retired) = ring.acquire_next(&ctx, 256, 128).expect("acquire resize");
            assert_eq!(idx, 0, "resize restarts at slot 0");
            assert_eq!(
                retired.len(),
                SCANOUT_EXPORT_RING,
                "resize retires every prior slot"
            );
            for old in retired {
                let fd = old.fd;
                old.destroy(&ctx.device);
                drop(OwnedFd::from_raw_fd(fd));
            }
            assert_eq!(ring.slot(0).width, 256);

            for fd in ring.destroy(&ctx.device) {
                drop(OwnedFd::from_raw_fd(fd));
            }
            assert!(
                ring.destroy(&ctx.device).is_empty(),
                "destroy on an empty ring is a no-op"
            );
            ctx.destroy();
        }
    }
}
