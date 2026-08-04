//! Whether this device can import a Linux dma-buf as `VkDeviceMemory`, and the
//! extensions that answer implies.
//!
//! # Why this is not `VK_EXT_external_memory_host` wearing a different name
//!
//! That extension is banned crate-wide and stays banned. It imports an arbitrary
//! **host virtual address range** belonging to this process, which over guest RAM
//! is the whole guest: the pointer is unbounded by anything the guest asked for,
//! it aliases whatever the QEMU RAMBlock mapping happens to cover, and it cannot
//! be revoked once handed to the driver.
//!
//! A dma-buf is a different object with different properties, and the difference
//! is what makes it admissible where the host-pointer import is not:
//!
//! * **Bounded by construction.** The fd names an explicit list of page-sized
//!   ranges chosen when it was created. There is no pointer to stray from and no
//!   surrounding mapping to reach; a page not named at create time is not
//!   reachable through the fd at all.
//! * **Revocable.** Closing the fd and freeing the `VkDeviceMemory` ends the
//!   GPU's access. A host-pointer import has no such handle.
//! * **Kernel-mediated.** The importing driver takes a reference on pages the
//!   kernel is tracking, rather than being handed a raw address it must trust.
//!
//! That is the same mechanism upstream QEMU's virtio-gpu blob path uses for
//! exactly this purpose, and it is what lets guest pages be a GPU source *and*
//! destination without the copy crossings that dominate this device's cost.
//!
//! # Query and enable together
//!
//! Same rule as [`super::device_features`], for the same reason: the extensions
//! are named here and nowhere else, so no site can bind an import the device was
//! never asked to support. [`DmaBufImport::required_extensions`] is the only
//! producer of those strings.

use ash::vk;

/// Buffer usage every imported guest-memory buffer is created with, and
/// therefore the usage [`query`] asks the device about.
///
/// Importability is a property of the handle type **and the usage**, so asking
/// about a narrower set than the import site binds is a query that can answer
/// yes to a bind the driver then refuses. The set is the union of both
/// directions the rail runs in:
///
/// * `TRANSFER_SRC` — guest pages as the source of an upload into a device-local
///   image or buffer.
/// * `TRANSFER_DST` — guest pages as the destination of a render/compute result,
///   which is the writeback the deferred-flush rail otherwise stages through the
///   CPU.
/// * `VERTEX_BUFFER` / `INDEX_BUFFER` / `STORAGE_BUFFER` / `UNIFORM_BUFFER` —
///   guest pages bound directly to a draw, with no copy at all.
pub const GUEST_IMPORT_USAGE: vk::BufferUsageFlags = vk::BufferUsageFlags::from_raw(
    vk::BufferUsageFlags::TRANSFER_SRC.as_raw()
        | vk::BufferUsageFlags::TRANSFER_DST.as_raw()
        | vk::BufferUsageFlags::VERTEX_BUFFER.as_raw()
        | vk::BufferUsageFlags::INDEX_BUFFER.as_raw()
        | vk::BufferUsageFlags::STORAGE_BUFFER.as_raw()
        | vk::BufferUsageFlags::UNIFORM_BUFFER.as_raw(),
);

/// Whether guest pages can reach this device as a dma-buf import, and when they
/// cannot, which check said so.
///
/// Rungs rather than a bool because every negative rung is a different host and
/// a different answer for a bug report: an Apple host has no dma-buf at all, a
/// Linux host with an old ICD may have the fd extension without it, and a device
/// can advertise both and still decline the handle type for this usage.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DmaBufImport {
    /// Both extensions advertised and the device reports DMA_BUF importable for
    /// [`GUEST_IMPORT_USAGE`]. The only rung on which the import path may run.
    Supported,
    /// No [`query`] has been run. The default, so a [`super::HostGpuCaps`] built
    /// without one never claims a capability nothing checked for.
    #[default]
    Unqueried,
    /// `VK_KHR_external_memory_fd` absent. Without it there is no
    /// `vkGetMemoryFdProperties` to ask which memory types accept the fd, and no
    /// `VkImportMemoryFdInfoKHR` to import through.
    NoExternalMemoryFd,
    /// `VK_EXT_external_memory_dma_buf` absent. Expected on every non-Linux ICD,
    /// MoltenVK included — the handle type names a Linux kernel object.
    NoDmaBufExtension,
    /// Both extensions advertised, and the device still declines DMA_BUF as an
    /// importable handle type for [`GUEST_IMPORT_USAGE`].
    NotImportable,
}

impl DmaBufImport {
    /// The one place the import path asks whether it may run.
    pub fn is_available(self) -> bool {
        matches!(self, Self::Supported)
    }

    /// Stable slug for the `reason=` field of a decline, and for the capability
    /// line. Named per rung so a log says which check refused.
    pub fn slug(self) -> &'static str {
        match self {
            Self::Supported => "supported",
            Self::Unqueried => "unqueried",
            Self::NoExternalMemoryFd => "no_external_memory_fd",
            Self::NoDmaBufExtension => "no_dma_buf_extension",
            Self::NotImportable => "not_importable",
        }
    }

    /// Device extension names this rung requires. Only [`Self::Supported`]
    /// requires any: asking for an extension on a rung that cannot use it would
    /// fail device creation on the hosts that do not have it, which is every
    /// host the negative rungs describe.
    pub fn required_extensions(self) -> Vec<*const std::os::raw::c_char> {
        if self.is_available() {
            vec![
                vk::KHR_EXTERNAL_MEMORY_FD_NAME.as_ptr(),
                vk::EXT_EXTERNAL_MEMORY_DMA_BUF_NAME.as_ptr(),
            ]
        } else {
            Vec::new()
        }
    }
}

/// Resolve dma-buf importability against one physical device.
///
/// `has_extension` is passed in rather than enumerated here because the caller
/// already enumerates device extensions for the other capability queries.
///
/// # Safety
///
/// `pd` must be a physical device belonging to `instance`.
pub unsafe fn query(
    instance: &ash::Instance,
    pd: vk::PhysicalDevice,
    has_extension: &dyn Fn(&std::ffi::CStr) -> bool,
) -> DmaBufImport {
    if !has_extension(vk::KHR_EXTERNAL_MEMORY_FD_NAME) {
        return DmaBufImport::NoExternalMemoryFd;
    }
    if !has_extension(vk::EXT_EXTERNAL_MEMORY_DMA_BUF_NAME) {
        return DmaBufImport::NoDmaBufExtension;
    }
    // `vkGetPhysicalDeviceExternalBufferProperties` is Vulkan 1.1 core and the
    // baseline is 1.2, so this is always answerable once the handle type itself
    // is spelled by an advertised extension.
    let info = vk::PhysicalDeviceExternalBufferInfo::default()
        .usage(GUEST_IMPORT_USAGE)
        .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
    let mut props = vk::ExternalBufferProperties::default();
    unsafe { instance.get_physical_device_external_buffer_properties(pd, &info, &mut props) };
    if props
        .external_memory_properties
        .external_memory_features
        .contains(vk::ExternalMemoryFeatureFlags::IMPORTABLE)
    {
        DmaBufImport::Supported
    } else {
        DmaBufImport::NotImportable
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Only the supported rung asks for extension strings. Requesting
    /// `VK_EXT_external_memory_dma_buf` on a host that does not advertise it
    /// fails `vkCreateDevice` outright — so a negative rung that still named its
    /// extensions would turn "no zero-copy here" into "no Vulkan here", on
    /// exactly the hosts the rung exists to describe.
    #[test]
    fn only_the_supported_rung_names_extensions() {
        assert_eq!(DmaBufImport::Supported.required_extensions().len(), 2);
        for rung in [
            DmaBufImport::Unqueried,
            DmaBufImport::NoExternalMemoryFd,
            DmaBufImport::NoDmaBufExtension,
            DmaBufImport::NotImportable,
        ] {
            assert!(
                rung.required_extensions().is_empty(),
                "{rung:?} must not request an extension it cannot use"
            );
            assert!(!rung.is_available(), "{rung:?} must not gate the rail open");
        }
    }

    /// The default rung is "nobody asked", not "no". Both refuse the rail, but
    /// only one of them is honest about a `HostGpuCaps` that was never queried —
    /// and the slug is what a reader greps to tell them apart.
    #[test]
    fn the_default_rung_says_it_was_never_queried() {
        assert_eq!(DmaBufImport::default(), DmaBufImport::Unqueried);
        assert_eq!(DmaBufImport::default().slug(), "unqueried");
    }

    /// One slug per rung: two rungs sharing one would mean watching the slug
    /// fire in the log and still not knowing which check refused.
    #[test]
    fn every_rung_has_its_own_slug() {
        let rungs = [
            DmaBufImport::Supported,
            DmaBufImport::Unqueried,
            DmaBufImport::NoExternalMemoryFd,
            DmaBufImport::NoDmaBufExtension,
            DmaBufImport::NotImportable,
        ];
        let mut slugs: Vec<_> = rungs.iter().map(|r| r.slug()).collect();
        slugs.sort_unstable();
        let count = slugs.len();
        slugs.dedup();
        assert_eq!(slugs.len(), count, "two rungs share a slug");
        assert!(slugs.iter().all(|s| s
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b == b'_')));
    }

    /// The usage set queried is the usage set bound. Asking about a narrower set
    /// than the import site creates its buffer with is a query that can answer
    /// yes to a bind the driver then refuses — the failure would land at
    /// `vkCreateBuffer` on a guest frame rather than at capability time.
    #[test]
    fn the_queried_usage_covers_both_directions_of_the_rail() {
        // Guest pages as a GPU *source* — the upload the CPU otherwise gathers.
        assert!(GUEST_IMPORT_USAGE.contains(vk::BufferUsageFlags::TRANSFER_SRC));
        // Guest pages as a GPU *destination* — the writeback the deferred-flush
        // rail otherwise stages through the CPU, and the larger half of the cost.
        assert!(GUEST_IMPORT_USAGE.contains(vk::BufferUsageFlags::TRANSFER_DST));
        // Bound straight to a draw, with no copy in either direction.
        for direct in [
            vk::BufferUsageFlags::VERTEX_BUFFER,
            vk::BufferUsageFlags::INDEX_BUFFER,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
        ] {
            assert!(GUEST_IMPORT_USAGE.contains(direct));
        }
    }
}
