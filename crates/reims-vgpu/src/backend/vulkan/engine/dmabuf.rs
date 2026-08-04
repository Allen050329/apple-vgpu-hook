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
use std::sync::Arc;

use ash::vk;

use super::types::GuestDmaBuf;
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
pub enum DmaBufDecline {
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
    /// The runtime's fd could not be duplicated, so there was nothing to hand
    /// `vkAllocateMemory` that this process could also keep. Distinct from every
    /// check above because it fails before the device is asked anything at all —
    /// a process at its descriptor limit reaches here, and no Vulkan result
    /// describes that.
    CloneFd { errno: i32 },
    /// The pinned-bytes bound is full of imports the command buffer now being
    /// recorded already names, so nothing can be displaced to make room. Not a
    /// device refusal and not a page-list problem: one draw asked for more guest
    /// memory at once than the rail is allowed to pin, and the caller gathers on
    /// the CPU instead.
    BoundInUse { held: u64, incoming: u64 },
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
            Self::CloneFd { .. } => "vk_dmabuf_clone_fd",
            Self::BoundInUse { .. } => "vk_dmabuf_bound_in_use",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::Unsupported { rung } => vec![("rung", rung.slug().to_string())],
            Self::FdProperties { result }
            | Self::CreateBuffer { result }
            | Self::AllocateMemory { result }
            | Self::BindBuffer { result } => {
                vec![(
                    "vk_result",
                    format!("{result:?}").replace(char::is_whitespace, "_"),
                )]
            }
            Self::NoImportableMemoryType => Vec::new(),
            Self::CloneFd { errno } => vec![("errno", errno.to_string())],
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
            Self::BoundInUse { held, incoming } => vec![
                ("held", held.to_string()),
                ("incoming", incoming.to_string()),
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

/// A check that stopped a resident's frame from being copied straight into the
/// guest's own pages, so the flush took the CPU route instead.
///
/// Every one of these is a *routing* answer rather than a loss — the copying
/// rail still lands the frame — but each is a whole flush's worth of memcpy that
/// the device paid and did not have to, so they are named individually and
/// counted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GuestWriteDecline {
    /// The device cannot import a dma-buf at all. Carries the rung so the log
    /// says which check refused; expected on every non-Linux host.
    Unsupported {
        rung: crate::backend::vulkan::caps::DmaBufImport,
    },
    /// The resident's physical channel order is not the guest's scanout order,
    /// so landing it would need an R/B exchange — which a buffer→image copy
    /// cannot perform. The copying rail's `into_bgra8` is where that lives.
    NotScanoutOrder,
    /// The resident's geometry is not the geometry the window promised the
    /// guest. Copying anyway would land one extent's pixels under another's row
    /// pitch.
    GeometryMoved {
        resident_width: u32,
        resident_height: u32,
        want_width: u32,
        want_height: u32,
    },
    /// The frame's last byte falls past the end of the imported window. A short
    /// page list reaches here rather than writing past the pages the fd names.
    WindowTooSmall { need: u64, have: u64 },
    /// The import itself declined; the inner reason names the step.
    Import { inner: DmaBufDecline },
}

impl Decline for GuestWriteDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::Unsupported { .. } => "gpu_writeback_unsupported",
            Self::NotScanoutOrder => "gpu_writeback_not_scanout_order",
            Self::GeometryMoved { .. } => "gpu_writeback_geometry_moved",
            Self::WindowTooSmall { .. } => "gpu_writeback_window_too_small",
            // The inner decline's own slug, so a driver that refuses the fd and
            // a page run that is too short stay as distinguishable here as they
            // are at the import site.
            Self::Import { inner } => inner.slug(),
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::Unsupported { rung } => vec![("rung", rung.slug().to_string())],
            Self::NotScanoutOrder => Vec::new(),
            Self::GeometryMoved {
                resident_width,
                resident_height,
                want_width,
                want_height,
            } => vec![
                ("resident", format!("{resident_width}x{resident_height}")),
                ("want", format!("{want_width}x{want_height}")),
            ],
            Self::WindowTooSmall { need, have } => {
                vec![("need", need.to_string()), ("have", have.to_string())]
            }
            Self::Import { inner } => inner.fields(),
        }
    }
}

crate::observe::decline::decline_display!(GuestWriteDecline);

/// Imports of guest page windows, kept across frames.
///
/// # Why this is a second cache and not the runtime's
///
/// [`crate::runtime::guest_dmabuf`] caches the *fd* — the result of the page
/// walk and the pin. This caches the `VkDeviceMemory` and `VkBuffer` made from
/// one, which is a different object with a different owner: the fd belongs to
/// this process and the device memory belongs to the driver, and only the
/// importer can release the second. Two caches, each bounding what it actually
/// holds, is the arrangement that keeps an eviction in either from being a
/// use-after-free in the other.
///
/// # What bounds it
///
/// The same guest bytes the runtime cache pins, because an import references
/// exactly the pages its fd names and no others. Making the two bounds equal is
/// what keeps a window that the runtime is still caching from being re-imported
/// every frame — a smaller bound here would pay `vkAllocateMemory` per flush to
/// hold pages that are pinned regardless.
///
/// # When an entry may be freed
///
/// `vkFreeMemory` on memory an in-flight command buffer references is undefined
/// behaviour, so this cache never frees an eviction itself: [`Self::get_or_import`]
/// hands the displaced imports back and
/// [`super::pools::ResourcePools::import_guest_window`] routes each through the
/// ring's graveyard, which destroys it once every slot open at that instant has
/// retired. That is what lets a *draw* import a window — it records a copy and
/// submits without waiting, so the "no consumer outlives its own fence" rule the
/// writeback rail relied on is no longer true of every caller.
///
/// The graveyard cannot see the command buffer currently being **recorded**,
/// though, because its slot is not pending until submit. So an eviction is also
/// refused for any entry the open recording has already been handed, tracked by
/// [`Self::epoch`]; a draw whose own imports do not fit under the bound is
/// declined ([`DmaBufDecline::BoundInUse`]) and gathers on the CPU instead.
#[derive(Default)]
pub(crate) struct ImportCache {
    entries: Vec<ImportEntry>,
    bytes: u64,
    clock: u64,
    /// Bumped by [`Self::begin_recording`] at the start of each command buffer.
    /// An entry stamped with the current value has been handed to the CB being
    /// recorded right now, which no fence and no graveyard mask can yet name.
    epoch: u64,
}

struct ImportEntry {
    /// [`GuestDmaBuf::id`], which is monotonic and never reused. The fd *number*
    /// cannot be the key: it is recycled the instant one closes, so a freed
    /// window and a fresh one can wear the same number and the second would bind
    /// the first's memory.
    window: u64,
    /// Holding the runtime's window keeps its id from being retired while this
    /// entry names it, so the key cannot go stale under the entry it identifies.
    _window_fd: Arc<GuestDmaBuf>,
    import: ImportedDmaBuf,
    used: u64,
    /// [`ImportCache::epoch`] when this entry was last handed out. Equal to the
    /// live epoch means the command buffer still recording may already name it.
    epoch: u64,
}

/// Most guest memory this device will hold imported at once. Equal to the
/// runtime cache's pin bound by construction; see [`ImportCache`].
const MAX_IMPORTED_BYTES: u64 = crate::runtime::guest_dmabuf::MAX_PINNED_BYTES;

/// Index of the entry to drop next: the one least recently handed out, skipping
/// every entry the command buffer now recording has already been given.
///
/// Separate from the eviction loop so the choice can be tested without a Vulkan
/// device — the loop's other half is handing the import to the graveyard, and
/// getting *which* entry it displaces wrong is either a re-import every frame or
/// a copy reading freed memory.
fn lru_victim(entries: &[ImportEntry], live_epoch: u64) -> Option<usize> {
    entries
        .iter()
        .enumerate()
        .filter(|(_, e)| e.epoch != live_epoch)
        .min_by_key(|(_, e)| e.used)
        .map(|(i, _)| i)
}

impl ImportCache {
    /// Start of a command buffer's recording. Every entry handed out from here
    /// until the next call is pinned against eviction, because nothing else can
    /// name a CB that has not been submitted.
    pub(crate) fn begin_recording(&mut self) {
        self.epoch += 1;
    }

    /// The bound `window`'s pages reach the device through, imported if this is
    /// the first time this window has been asked for.
    ///
    /// `size` is the whole extent the fd covers, so one window is one buffer
    /// whatever part of it a given copy names — a caller wanting a sub-range
    /// says so with the copy's own offset rather than by importing twice.
    ///
    /// Imports displaced to make room are **appended to `displaced` rather than
    /// freed**: the caller owns disposing of them safely, which for a submitting
    /// caller means the ring's graveyard. See [`ImportCache`].
    ///
    /// # Safety
    ///
    /// `device` must be the one `ctx` owns.
    pub(crate) unsafe fn get_or_import(
        &mut self,
        ctx: &super::context::DeviceContext,
        window: &Arc<GuestDmaBuf>,
        size: u64,
        displaced: &mut Vec<ImportedDmaBuf>,
    ) -> Result<ImportedDmaBuf, DmaBufDecline> {
        self.clock += 1;
        let clock = self.clock;
        let epoch = self.epoch;
        if let Some(entry) = self.entries.iter_mut().find(|e| e.window == window.id) {
            entry.used = clock;
            entry.epoch = epoch;
            return Ok(entry.import);
        }
        // Before the import rather than after it, so the bound is restored
        // before it is added to rather than after.
        self.evict_to_bound(size, displaced);
        if self.bytes.saturating_add(size) > MAX_IMPORTED_BYTES {
            // Everything left is pinned by the CB now recording, so there is no
            // eviction that would not free memory a recorded copy reads. The
            // caller still has its CPU gather; taking it is slower than this
            // rail and correct, which is the whole reason the fallback is kept.
            return Err(DmaBufDecline::BoundInUse {
                held: self.bytes,
                incoming: size,
            });
        }
        // The driver takes the fd it is given, and the runtime keeps the
        // original for every other importer and for the revocation that closing
        // it performs. So the import gets a duplicate; `import_buffer` then owns
        // exactly one fd and its ownership rule stays a local property.
        let fd = window.fd.try_clone().map_err(|e| DmaBufDecline::CloneFd {
            errno: e.raw_os_error().unwrap_or(0),
        })?;
        let import = unsafe { import_buffer(ctx, fd, size) }?;
        self.entries.push(ImportEntry {
            window: window.id,
            _window_fd: Arc::clone(window),
            import,
            used: clock,
            epoch,
        });
        self.bytes += size;
        Ok(import)
    }

    /// Displace least-recently-used imports into `displaced` until `incoming`
    /// more bytes would fit under [`MAX_IMPORTED_BYTES`], or until everything
    /// left is pinned by the open recording.
    fn evict_to_bound(&mut self, incoming: u64, displaced: &mut Vec<ImportedDmaBuf>) {
        while self.bytes.saturating_add(incoming) > MAX_IMPORTED_BYTES {
            let Some(victim) = lru_victim(&self.entries, self.epoch) else {
                return;
            };
            let entry = self.entries.swap_remove(victim);
            self.bytes = self.bytes.saturating_sub(entry.import.size);
            displaced.push(entry.import);
        }
    }

    /// Release every import. The revocation this rail's capability doc promises,
    /// so it must run on a teardown that is otherwise giving up.
    ///
    /// # Safety
    ///
    /// No submission may still reference any import, and `device` must be the
    /// one they were made against.
    pub(crate) unsafe fn destroy_all(&mut self, device: &ash::Device) {
        for entry in self.entries.drain(..) {
            unsafe { entry.import.destroy(device) };
        }
        self.bytes = 0;
    }

    /// Guest memory currently reachable by the device through imports. Census
    /// only.
    pub(crate) fn imported_bytes(&self) -> u64 {
        self.bytes
    }

    /// Windows currently imported. Census only.
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
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
            DmaBufDecline::CloneFd { errno: 24 },
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

    fn entry(window: u64, size: u64, used: u64) -> ImportEntry {
        entry_at_epoch(window, size, used, 0)
    }

    fn entry_at_epoch(window: u64, size: u64, used: u64, epoch: u64) -> ImportEntry {
        let (read, _write) = std::io::pipe().expect("a pipe for a placeholder fd");
        ImportEntry {
            window,
            _window_fd: Arc::new(GuestDmaBuf {
                id: window,
                fd: std::os::fd::OwnedFd::from(read),
            }),
            // Never destroyed: no test here reaches `vkFreeMemory`, and the
            // cache's own bookkeeping is what is under test.
            import: ImportedDmaBuf {
                buffer: vk::Buffer::null(),
                memory: vk::DeviceMemory::null(),
                size,
            },
            used,
            epoch,
        }
    }

    /// The least recently handed-out import is the one that goes.
    ///
    /// Getting this backwards does not fail anything: the cache still answers
    /// correctly, and the only symptom is `vkAllocateMemory` on the hot window
    /// every frame while a cold one is kept — a performance defect that looks
    /// exactly like the rail working.
    #[test]
    fn eviction_takes_the_least_recently_used_import() {
        let entries = vec![entry(1, 4096, 30), entry(2, 4096, 7), entry(3, 4096, 19)];
        assert_eq!(lru_victim(&entries, 1), Some(1));
        assert_eq!(lru_victim(&[], 1), None);
    }

    /// An import the command buffer now recording was already handed is not a
    /// candidate however cold it looks, because nothing else can see that CB.
    ///
    /// A submitted CB is named by its ring slot, so the graveyard holds its
    /// imports until the slot retires. One still being *recorded* has no slot
    /// bit set and no fence, so an eviction would `vkFreeMemory` guest pages a
    /// recorded `vkCmdCopyBufferToImage` is about to read — the frame comes back
    /// as whatever the driver put there, or the submit faults.
    #[test]
    fn an_import_the_open_recording_holds_is_never_the_victim() {
        let entries = vec![
            entry_at_epoch(1, 4096, 1, 7),
            entry_at_epoch(2, 4096, 2, 6),
            entry_at_epoch(3, 4096, 3, 7),
        ];
        assert_eq!(
            lru_victim(&entries, 7),
            Some(1),
            "the coldest entry is pinned by the live epoch; the next one goes"
        );
        assert_eq!(
            lru_victim(&entries, 8),
            Some(0),
            "a later recording pins none of them, so the plain LRU answer returns"
        );
    }

    /// Displacement hands the imports back rather than freeing them, and stops
    /// as soon as everything left belongs to the open recording.
    ///
    /// Both halves are load-bearing. Freeing here would be a `vkFreeMemory`
    /// under a possibly in-flight command buffer, which is why the caller routes
    /// them through the ring's graveyard instead. And a loop that could not stop
    /// would either spin or evict the recording's own pages.
    #[test]
    fn displacement_yields_the_imports_and_stops_at_the_pinned_ones() {
        let mut cache = ImportCache {
            entries: vec![
                entry_at_epoch(1, MAX_IMPORTED_BYTES / 2, 1, 4),
                entry_at_epoch(2, MAX_IMPORTED_BYTES / 2, 2, 3),
            ],
            bytes: MAX_IMPORTED_BYTES,
            clock: 2,
            epoch: 4,
        };

        let mut displaced = Vec::new();
        cache.evict_to_bound(MAX_IMPORTED_BYTES / 2, &mut displaced);
        assert_eq!(
            displaced.len(),
            1,
            "the unpinned entry is the one displaced"
        );
        assert_eq!(cache.entries.len(), 1);
        assert_eq!(cache.entries[0].window, 1);
        assert_eq!(cache.bytes, MAX_IMPORTED_BYTES / 2);

        // Nothing unpinned is left, so a second ask cannot be satisfied and must
        // leave the survivor alone rather than loop or free it.
        let mut again = Vec::new();
        cache.evict_to_bound(MAX_IMPORTED_BYTES, &mut again);
        assert!(again.is_empty());
        assert_eq!(cache.entries.len(), 1);
    }

    /// The two caches bound the same physical guest pages, so they carry the
    /// same number.
    ///
    /// A smaller bound here would evict imports of windows the runtime is still
    /// pinning, which pays `vkAllocateMemory` per flush to reach pages that stay
    /// unswappable either way — the cost without the saving.
    #[test]
    fn the_import_bound_matches_the_pin_bound() {
        assert_eq!(
            MAX_IMPORTED_BYTES,
            crate::runtime::guest_dmabuf::MAX_PINNED_BYTES
        );
    }

    /// The key is the window's monotonic id, never the fd number.
    ///
    /// An fd number is recycled the instant one closes, so a cache keyed on it
    /// would hand a fresh window the previous occupant's `VkDeviceMemory` — a
    /// flush landing one surface's pixels in another's pages, which no counter
    /// in this device would report.
    #[test]
    fn the_cache_key_is_the_window_id_and_not_the_fd() {
        let first = entry(1, 4096, 1);
        let second = entry(2, 4096, 2);
        assert_ne!(first.window, second.window);
        // Two live pipes can share neither number, but a closed one's is free
        // for the next open — which is exactly why the id, and not this, is the
        // key.
        assert_eq!(first.window, first._window_fd.id);
        assert_eq!(second.window, second._window_fd.id);
    }
}
