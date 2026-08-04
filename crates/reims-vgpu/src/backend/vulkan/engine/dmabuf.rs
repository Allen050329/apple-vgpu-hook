//! Import a dma-buf fd as `VkDeviceMemory` and bind a `VkBuffer` to it.
//!
//! This is the one place a foreign allocation becomes something the engine can
//! bind. Everything about *which* pages the fd covers is decided before it gets
//! here — see [`crate::qemu::host_ops`] for how a guest page run becomes an fd,
//! and [`super::super::caps::external_memory`] for why a dma-buf is admissible
//! where a host-pointer import is not.
//!
//! # The fd ownership rule
//!
//! `vkAllocateMemory` with a chained `VkImportMemoryFdInfoKHR` **takes ownership
//! of the fd on success** and leaves it with the caller on failure. Get that
//! backwards in either direction and the bug is invisible until it is fatal: a
//! double close eventually closes an unrelated fd this process has since opened,
//! and a missed close leaks one per imported buffer until the process hits its
//! limit mid-frame.
//!
//! [`import_buffer`] takes an [`OwnedFd`] by value so the type system carries the
//! rule. On the success path the fd is released with [`IntoRawFd::into_raw_fd`]
//! and deliberately not closed; on every failure path it is simply dropped,
//! which closes it. There is no path that does both and none that does neither.

use std::os::fd::{AsRawFd, IntoRawFd, OwnedFd};

use ash::vk;

use crate::observe::Decline;

/// A guest page run living on the GPU as a bindable buffer, with no copy
/// between it and the guest's own view of those pages.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ImportedDmaBuf {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    /// Bytes the fd covers. The buffer spans all of it.
    pub size: vk::DeviceSize,
}

impl ImportedDmaBuf {
    /// Release both halves. Freeing the memory is what ends the GPU's access to
    /// the guest pages, so this is the revocation the capability doc promises —
    /// it must run even on a teardown path that is otherwise giving up.
    ///
    /// # Safety
    ///
    /// No submission may still reference `buffer`, and `device` must be the one
    /// the import was made against.
    pub(crate) unsafe fn destroy(self, device: &ash::Device) {
        unsafe {
            device.destroy_buffer(self.buffer, None);
            device.free_memory(self.memory, None);
        }
    }
}

/// A check that stopped a dma-buf from becoming a bindable buffer.
///
/// Every variant is a distinct check with its own slug: an import that fails
/// costs the frame a copy at best and the guest's work at worst, and "the import
/// declined" without saying which step is half a diagnostic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DmaBufDecline {
    /// The device is not on the `Supported` rung. Carries the rung so the log
    /// says whether this host lacks the extension or declines the handle type.
    Unsupported {
        rung: crate::backend::vulkan::caps::DmaBufImport,
    },
    /// `vkGetMemoryFdPropertiesKHR` failed. The fd is not one this device can
    /// answer about — a non-dma-buf fd reaches here, as does a dma-buf whose
    /// exporter this driver cannot talk to.
    FdProperties { result: vk::Result },
    /// The device accepts the fd but reports no memory type for it. Distinct
    /// from [`Self::NoCommonMemoryType`]: here the *fd* matched nothing, before
    /// the buffer's own requirements narrowed anything.
    NoImportableMemoryType,
    /// `vkCreateBuffer` failed for a buffer declared external.
    CreateBuffer { result: vk::Result },
    /// The memory types that accept the fd and the memory types the buffer
    /// accepts are disjoint, so no single index satisfies both.
    NoCommonMemoryType { fd_bits: u32, buffer_bits: u32 },
    /// The buffer needs more bytes than the fd covers. A guest page run shorter
    /// than the resource that named it lands here rather than binding a buffer
    /// that reads past the end of the import.
    TooSmall { required: u64, available: u64 },
    /// `vkAllocateMemory` failed. The fd is still owned by this process and has
    /// been closed.
    AllocateMemory { result: vk::Result },
    /// `vkBindBufferMemory` failed after a successful import.
    BindBuffer { result: vk::Result },
}

impl Decline for DmaBufDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::Unsupported { .. } => "vk_dmabuf_unsupported",
            Self::FdProperties { .. } => "vk_dmabuf_fd_properties",
            Self::NoImportableMemoryType => "vk_dmabuf_no_importable_memory_type",
            Self::CreateBuffer { .. } => "vk_dmabuf_create_buffer",
            Self::NoCommonMemoryType { .. } => "vk_dmabuf_no_common_memory_type",
            Self::TooSmall { .. } => "vk_dmabuf_too_small",
            Self::AllocateMemory { .. } => "vk_dmabuf_allocate_memory",
            Self::BindBuffer { .. } => "vk_dmabuf_bind_buffer",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::Unsupported { rung } => vec![("rung", rung.slug().to_string())],
            Self::FdProperties { result }
            | Self::CreateBuffer { result }
            | Self::AllocateMemory { result }
            | Self::BindBuffer { result } => {
                vec![("vk_result", format!("{result:?}").replace(char::is_whitespace, "_"))]
            }
            Self::NoImportableMemoryType => Vec::new(),
            Self::NoCommonMemoryType {
                fd_bits,
                buffer_bits,
            } => vec![
                ("fd_bits", format!("{fd_bits:#x}")),
                ("buffer_bits", format!("{buffer_bits:#x}")),
            ],
            Self::TooSmall {
                required,
                available,
            } => vec![
                ("required", required.to_string()),
                ("available", available.to_string()),
            ],
        }
    }
}

crate::observe::decline::decline_display!(DmaBufDecline);

/// Turn a dma-buf fd covering `size` bytes of guest pages into a bound
/// `VkBuffer`.
///
/// The buffer is created with [`GUEST_IMPORT_USAGE`], which is the same usage
/// the capability query asked the device about — so a device on the `Supported`
/// rung has already answered yes to this exact combination.
///
/// The allocation is dedicated. One import is one buffer by construction, which
/// is precisely the dedicated model, and some drivers require it for dma-buf
/// imports rather than merely preferring it.
///
/// # Safety
///
/// `fd` must be a dma-buf whose mapping covers at least `size` bytes, and it
/// must remain valid for the lifetime of the returned import — which it does,
/// because the import takes ownership of it.
///
/// [`GUEST_IMPORT_USAGE`]: crate::backend::vulkan::caps::external_memory::GUEST_IMPORT_USAGE
pub(crate) unsafe fn import_buffer(
    ctx: &super::context::DeviceContext,
    fd: OwnedFd,
    size: u64,
) -> Result<ImportedDmaBuf, DmaBufDecline> {
    use crate::backend::vulkan::caps::external_memory::GUEST_IMPORT_USAGE;

    let Some(loader) = ctx.external_memory_fd.as_ref() else {
        return Err(DmaBufDecline::Unsupported {
            rung: ctx.caps.dma_buf_import,
        });
    };
    const HANDLE_TYPE: vk::ExternalMemoryHandleTypeFlags =
        vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT;

    // Which memory types will accept *this* fd. Asked before anything is
    // created, because the answer is a property of the exporter and can differ
    // between two fds on one device — guest RAM and a GPU-exported buffer are
    // both dma-bufs and need not land in the same heap.
    let mut fd_props = vk::MemoryFdPropertiesKHR::default();
    unsafe { loader.get_memory_fd_properties(HANDLE_TYPE, fd.as_raw_fd(), &mut fd_props) }
        .map_err(|result| DmaBufDecline::FdProperties { result })?;
    if fd_props.memory_type_bits == 0 {
        return Err(DmaBufDecline::NoImportableMemoryType);
    }

    let mut external = vk::ExternalMemoryBufferCreateInfo::default().handle_types(HANDLE_TYPE);
    let create = vk::BufferCreateInfo::default()
        .size(size)
        .usage(GUEST_IMPORT_USAGE)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .push_next(&mut external);
    let buffer = unsafe { ctx.device.create_buffer(&create, None) }
        .map_err(|result| DmaBufDecline::CreateBuffer { result })?;

    // From here every failure must destroy the buffer before returning, so the
    // work is done in a closure and the cleanup happens once at the end.
    let bound = (|| {
        let reqs = unsafe { ctx.device.get_buffer_memory_requirements(buffer) };
        if reqs.size > size {
            return Err(DmaBufDecline::TooSmall {
                required: reqs.size,
                available: size,
            });
        }
        let common = reqs.memory_type_bits & fd_props.memory_type_bits;
        if common == 0 {
            return Err(DmaBufDecline::NoCommonMemoryType {
                fd_bits: fd_props.memory_type_bits,
                buffer_bits: reqs.memory_type_bits,
            });
        }
        // The lowest common index. There is no preference to express: the
        // exporter already decided where these pages live, so the memory-class
        // machinery that ranks heaps for owned allocations has nothing to rank
        // here — every bit in `common` names the same underlying pages.
        let memory_type_index = common.trailing_zeros();

        let mut import = vk::ImportMemoryFdInfoKHR::default()
            .handle_type(HANDLE_TYPE)
            .fd(fd.as_raw_fd());
        let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().buffer(buffer);
        let allocate = vk::MemoryAllocateInfo::default()
            .allocation_size(size)
            .memory_type_index(memory_type_index)
            .push_next(&mut import)
            .push_next(&mut dedicated);
        let memory = unsafe { ctx.device.allocate_memory(&allocate, None) }
            .map_err(|result| DmaBufDecline::AllocateMemory { result })?;

        match unsafe { ctx.device.bind_buffer_memory(buffer, memory, 0) } {
            Ok(()) => Ok(ImportedDmaBuf {
                buffer,
                memory,
                size,
            }),
            Err(result) => {
                // The import succeeded, so the fd is the driver's now and
                // freeing the memory is what releases it.
                unsafe { ctx.device.free_memory(memory, None) };
                Err(DmaBufDecline::BindBuffer { result })
            }
        }
    })();

    match bound {
        Ok(import) => {
            // Ownership moved to the driver at `vkAllocateMemory`. Releasing the
            // `OwnedFd` without closing it is what keeps this from being a
            // double close the moment the driver closes its copy.
            let _ = fd.into_raw_fd();
            Ok(import)
        }
        Err(decline) => {
            unsafe { ctx.device.destroy_buffer(buffer, None) };
            // `fd` drops here and closes. Correct on every arm above: either
            // `vkAllocateMemory` was never reached, or it failed — and a failed
            // import leaves the fd with the caller.
            Err(decline)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn every_decline() -> Vec<DmaBufDecline> {
        vec![
            DmaBufDecline::Unsupported {
                rung: crate::backend::vulkan::caps::DmaBufImport::NoDmaBufExtension,
            },
            DmaBufDecline::FdProperties {
                result: vk::Result::ERROR_INVALID_EXTERNAL_HANDLE,
            },
            DmaBufDecline::NoImportableMemoryType,
            DmaBufDecline::CreateBuffer {
                result: vk::Result::ERROR_OUT_OF_HOST_MEMORY,
            },
            DmaBufDecline::NoCommonMemoryType {
                fd_bits: 0x9,
                buffer_bits: 0x6,
            },
            DmaBufDecline::TooSmall {
                required: 8192,
                available: 4096,
            },
            DmaBufDecline::AllocateMemory {
                result: vk::Result::ERROR_OUT_OF_DEVICE_MEMORY,
            },
            DmaBufDecline::BindBuffer {
                result: vk::Result::ERROR_INVALID_EXTERNAL_HANDLE,
            },
        ]
    }

    /// One slug per check. Two of these sharing one would mean watching the
    /// import fail in the log and being unable to tell a driver that refused the
    /// fd from a page run that was simply too short.
    #[test]
    fn every_decline_has_its_own_log_safe_slug() {
        let mut slugs: Vec<_> = every_decline().iter().map(|d| d.slug()).collect();
        let count = slugs.len();
        slugs.sort_unstable();
        slugs.dedup();
        assert_eq!(slugs.len(), count, "two dma-buf declines share a slug");
        for slug in slugs {
            assert!(
                slug.starts_with("vk_dmabuf_"),
                "{slug} is not greppable as a dma-buf decline"
            );
            assert!(slug
                .bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_'));
        }
    }

    /// A decline that names only its class leaves out the value that caused it.
    /// The two that carry a size and the two that carry a bitmask are the ones
    /// where the number *is* the diagnosis.
    #[test]
    fn declines_carry_the_values_behind_them() {
        for decline in every_decline() {
            let rendered = decline.to_string();
            assert!(rendered.contains(decline.slug()), "{rendered}");
            assert!(
                !rendered.contains(' ') || rendered.split(' ').count() > 1,
                "{rendered}"
            );
        }
        let short = DmaBufDecline::TooSmall {
            required: 8192,
            available: 4096,
        };
        assert!(short.to_string().contains("required=8192"));
        assert!(short.to_string().contains("available=4096"));
        let disjoint = DmaBufDecline::NoCommonMemoryType {
            fd_bits: 0x9,
            buffer_bits: 0x6,
        };
        assert!(disjoint.to_string().contains("fd_bits=0x9"));
        assert!(disjoint.to_string().contains("buffer_bits=0x6"));
    }

    /// The unsupported decline reports *which* rung refused, not just that one
    /// did — an Apple host and a Linux host with an old ICD are different bug
    /// reports and must not render identically.
    #[test]
    fn the_unsupported_decline_names_the_rung() {
        use crate::backend::vulkan::caps::DmaBufImport;
        let apple = DmaBufDecline::Unsupported {
            rung: DmaBufImport::NoDmaBufExtension,
        };
        let declined = DmaBufDecline::Unsupported {
            rung: DmaBufImport::NotImportable,
        };
        assert!(apple.to_string().contains("rung=no_dma_buf_extension"));
        assert!(declined.to_string().contains("rung=not_importable"));
        assert_ne!(apple.to_string(), declined.to_string());
    }
}
