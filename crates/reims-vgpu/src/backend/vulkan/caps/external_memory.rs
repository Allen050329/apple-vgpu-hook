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
//!
//! # Optional, and switchable off
//!
//! Neither extension is a requirement. A host that advertises neither reaches
//! [`DmaBufImport::NoDmaBufExtension`], asks for nothing at `vkCreateDevice`, and
//! runs every guest-memory rail through the copying path it used before this
//! capability existed. That is the whole support story for Apple hosts, for a
//! Linux ICD without the extension, and for a device that declines the handle
//! type — three different hosts, three different rungs, one behavior.
//!
//! [`crate::env::DMABUF`] adds a fourth: an operator can take a host that *is*
//! capable down to [`DmaBufImport::DisabledByEnv`]. This is the only way to
//! exercise the copying rails on a machine where the import works, so a
//! regression in them is findable without hunting for hardware that lacks the
//! extension. It cannot run the other way — see [`crate::env`] for why no
//! variable may widen a measured capability.

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
    /// `vkGetMemoryFdProperties` to ask which memory types accept the fd, and
    /// nothing to chain onto `vkAllocateMemory` to import through.
    NoExternalMemoryFd,
    /// `VK_EXT_external_memory_dma_buf` absent. Expected on every non-Linux ICD,
    /// MoltenVK included — the handle type names a Linux kernel object.
    NoDmaBufExtension,
    /// Both extensions advertised, and the device still declines DMA_BUF as an
    /// importable handle type for [`GUEST_IMPORT_USAGE`].
    NotImportable,
    /// [`crate::env::DMABUF`] was set off. The only rung that is a statement
    /// about policy rather than about the host: this device may well be capable,
    /// and the operator asked for the copying rails anyway. Distinct from every
    /// rung above precisely so a log does not read as a hardware limitation when
    /// it is a switch someone left set.
    DisabledByEnv,
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
            Self::DisabledByEnv => "disabled_by_env",
        }
    }

    /// Stable byte encoding for [`LATCHED`].
    ///
    /// Both directions are written out rather than cast, so a rung added later
    /// is a non-exhaustive-match error here instead of a value that decodes as a
    /// neighbouring rung — which would be a log naming the wrong check and, on
    /// the `Supported` code, a rail opened by an arithmetic accident.
    fn code(self) -> u8 {
        match self {
            Self::Unqueried => 0,
            Self::Supported => 1,
            Self::NoExternalMemoryFd => 2,
            Self::NoDmaBufExtension => 3,
            Self::NotImportable => 4,
            Self::DisabledByEnv => 5,
        }
    }

    /// Inverse of [`Self::code`]. An unknown byte is [`Self::Unqueried`] — the
    /// rung that claims nothing — so a decode that somehow goes wrong closes the
    /// rail's *export* side rather than opening it.
    fn from_code(code: u8) -> Self {
        match code {
            1 => Self::Supported,
            2 => Self::NoExternalMemoryFd,
            3 => Self::NoDmaBufExtension,
            4 => Self::NotImportable,
            5 => Self::DisabledByEnv,
            _ => Self::Unqueried,
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

/// The rung this process resolved, readable without the engine lock and
/// without a device context in hand.
///
/// # Why the answer is published rather than only held on `HostGpuCaps`
///
/// The rail has two halves in two crates' worth of distance from each other:
/// the runtime asks the host to *export* a dma-buf over guest pages, and the
/// engine *imports* it. Only the engine held the answer, so the export ran
/// unconditionally and the import declined afterwards — which on any host
/// without the extension meant a `UDMABUF_CREATE_LIST` per window, a cached fd,
/// and up to [`crate::runtime::guest_dmabuf::MAX_PINNED_BYTES`] of the guest's
/// RAM made unswappable, all of it to feed an import that could never happen.
/// Pinning guest memory for a capability the device does not have is the cost
/// this publishes to avoid.
///
/// Zero — [`DmaBufImport::Unqueried`] — until a device is created, which is the
/// honest answer and the one the export side treats as "not settled, let the
/// import site decide".
static LATCHED: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Publish the rung a freshly created device resolved to.
///
/// Called once per `vkCreateDevice`, including each recreate, so a rebuilt
/// device republishes rather than leaving the previous one's answer standing.
pub fn latch(rung: DmaBufImport) {
    LATCHED.store(rung.code(), std::sync::atomic::Ordering::Relaxed);
}

/// The published rung. [`DmaBufImport::Unqueried`] before any device exists.
///
/// `Relaxed` on both sides: this gates whether an optimization is attempted, and
/// a reader that sees the previous value takes the copying path for one window.
/// Nothing here orders access to any other memory.
pub fn latched() -> DmaBufImport {
    DmaBufImport::from_code(LATCHED.load(std::sync::atomic::Ordering::Relaxed))
}

/// What [`crate::env::DMABUF`] says about running this rail at all.
///
/// `None` to go on and ask the device. `Some` short-circuits [`query`], which
/// is deliberate on two counts: the device is not asked about a handle type
/// nothing will import, and neither extension is then named at `vkCreateDevice`,
/// so the switch produces exactly the device a host without them would get
/// rather than a capable device with one gate closed.
///
/// [`crate::env::Switch::On`] is not a way to turn the rail on — no variable may
/// widen a measured capability — but it is not ignored either: an operator who
/// set it has stated an expectation, and if the device then refuses, the `vk_caps`
/// line names the rung that refused. Only the unrecognized case is reported here,
/// because that is the one an operator would otherwise read as "the switch did
/// nothing" with no way to tell a typo from a device that declined.
fn env_override() -> Option<DmaBufImport> {
    match crate::env::read(crate::env::DMABUF) {
        (crate::env::Switch::Off, _) => Some(DmaBufImport::DisabledByEnv),
        (crate::env::Switch::Unrecognized, value) => {
            crate::observe::fail(format!(
                "vk_dmabuf_env_unrecognized var={} value={:?} (expected on|off; the rail is left \
                 to the device)",
                crate::env::DMABUF,
                value.unwrap_or_default()
            ));
            None
        }
        (crate::env::Switch::On | crate::env::Switch::Unset, _) => None,
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
    if let Some(disabled) = env_override() {
        return disabled;
    }
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

    /// Every rung, so a rung added without a test here fails to compile rather
    /// than quietly skipping the invariants below.
    const RUNGS: [DmaBufImport; 6] = [
        DmaBufImport::Supported,
        DmaBufImport::Unqueried,
        DmaBufImport::NoExternalMemoryFd,
        DmaBufImport::NoDmaBufExtension,
        DmaBufImport::NotImportable,
        DmaBufImport::DisabledByEnv,
    ];

    /// Only the supported rung asks for extension strings. Requesting
    /// `VK_EXT_external_memory_dma_buf` on a host that does not advertise it
    /// fails `vkCreateDevice` outright — so a negative rung that still named its
    /// extensions would turn "no zero-copy here" into "no Vulkan here", on
    /// exactly the hosts the rung exists to describe.
    #[test]
    fn only_the_supported_rung_names_extensions() {
        assert_eq!(DmaBufImport::Supported.required_extensions().len(), 2);
        for rung in RUNGS.into_iter().filter(|r| !r.is_available()) {
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
        let mut slugs: Vec<_> = RUNGS.iter().map(|r| r.slug()).collect();
        slugs.sort_unstable();
        let count = slugs.len();
        slugs.dedup();
        assert_eq!(slugs.len(), count, "two rungs share a slug");
        assert!(slugs
            .iter()
            .all(|s| s.bytes().all(|b| b.is_ascii_lowercase() || b == b'_')));
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

    /// Set [`crate::env::DMABUF`] to `value`, read the override, and restore.
    /// One test at a time: the variable is process-global.
    fn with_env(value: Option<&str>) -> Option<DmaBufImport> {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: the lock serializes every mutation of this variable in this
        // process, and the only reader is `env_override`, called below.
        unsafe {
            match value {
                Some(v) => std::env::set_var(crate::env::DMABUF, v),
                None => std::env::remove_var(crate::env::DMABUF),
            }
        }
        let out = env_override();
        unsafe { std::env::remove_var(crate::env::DMABUF) };
        out
    }

    /// The switch turns the rail off, and lands on a rung that says a person did
    /// it. Reading `no_dma_buf_extension` for a host that has the extension would
    /// send the next bug report hunting for a driver problem.
    #[test]
    fn the_env_switch_takes_a_capable_host_down() {
        assert_eq!(with_env(Some("0")), Some(DmaBufImport::DisabledByEnv));
        assert_eq!(with_env(Some("off")), Some(DmaBufImport::DisabledByEnv));
        assert!(!DmaBufImport::DisabledByEnv.is_available());
        assert!(DmaBufImport::DisabledByEnv.required_extensions().is_empty());
        assert_eq!(DmaBufImport::DisabledByEnv.slug(), "disabled_by_env");
    }

    /// The switch has no on direction. Setting it affirmatively hands the answer
    /// straight back to the device — which is the whole rule from [`crate::env`]:
    /// a variable may narrow what this device does and may never widen it, because
    /// binding an extension the host does not advertise fails `vkCreateDevice` and
    /// importing a handle type it declines is undefined behavior in the driver.
    #[test]
    fn the_env_switch_cannot_turn_the_rail_on() {
        for on in ["1", "on", "true", "yes"] {
            assert_eq!(with_env(Some(on)), None, "{on} must not preempt the query");
        }
        assert_eq!(with_env(None), None);
    }

    /// A misspelled value leaves the device to decide, rather than guessing at
    /// an intent. `env::read` keeps the raw value so the line above names it.
    #[test]
    fn an_unrecognized_value_does_not_change_the_answer() {
        assert_eq!(with_env(Some("maybe")), None);
    }

    /// Every rung survives the byte encoding the published answer travels
    /// through, and no two share a code. A collision would let the export side
    /// read `Supported` for a host that refused — pinning guest pages for an
    /// import that then declines, which is the exact cost the latch exists to
    /// avoid.
    #[test]
    fn every_rung_round_trips_through_its_code() {
        let mut codes: Vec<u8> = RUNGS.iter().map(|r| r.code()).collect();
        for rung in RUNGS {
            assert_eq!(DmaBufImport::from_code(rung.code()), rung, "{rung:?}");
        }
        let count = codes.len();
        codes.sort_unstable();
        codes.dedup();
        assert_eq!(codes.len(), count, "two rungs share a code");
    }

    /// An unrecognized byte decodes to the rung that claims nothing, so a decode
    /// that goes wrong closes the export side rather than opening it. Zero is
    /// pinned separately because it is the atomic's initializer, written as a
    /// literal where `code()` cannot be called.
    #[test]
    fn an_unknown_code_is_the_rung_that_claims_nothing() {
        assert_eq!(DmaBufImport::Unqueried.code(), 0);
        for code in [6u8, 7, 100, u8::MAX] {
            assert_eq!(DmaBufImport::from_code(code), DmaBufImport::Unqueried);
        }
    }

    /// The published answer starts at "nobody asked" and follows the last
    /// device created — a recreate republishes rather than leaving a stale rung
    /// standing.
    #[test]
    fn the_published_rung_follows_the_last_device() {
        assert_eq!(DmaBufImport::from_code(0), DmaBufImport::Unqueried);
        for rung in RUNGS {
            latch(rung);
            assert_eq!(latched(), rung, "{rung:?}");
        }
        latch(DmaBufImport::Unqueried);
    }
}
