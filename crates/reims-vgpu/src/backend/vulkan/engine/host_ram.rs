//! Import a bounded span of a RAMBlock's host mapping as `VkDeviceMemory` and
//! bind a `VkBuffer` over it.
//!
//! This is the one place guest memory becomes something the engine can bind.
//! Which *bytes* a draw reaches is decided before it gets here and is carried by
//! a [`GuestRef`], whose bound cannot be skipped — see
//! [`crate::runtime::guest_ram`] for why that is a type and not a review rule,
//! and [`super::super::caps::host_pointer`] for the capability that gates the
//! whole rail.
//!
//! # Windows, and why the whole block is not one of them
//!
//! A window is [`GUEST_IMPORT_WINDOW_BYTES`] of one import, and every reference
//! the window holds binds against the one buffer it already made. The size is
//! not an amortisation knob: registering a range makes the DRM driver re-walk
//! all of it whenever an MMU notifier fires over it, so the recurring cost of a
//! window is its size, on submissions that have nothing to do with the bytes
//! that moved. [`crate::runtime::guest_ram::ImportWindow`] carries the profile
//! that fixed the number. **Never widen a window to a whole RAMBlock**, which on
//! this pathway is 16 GiB of that walk.
//!
//! # Bounded, and never evicted
//!
//! [`HostRamImports`] grows to [`WINDOW_COUNT_CAP`] windows or
//! [`WINDOW_TOTAL_BYTE_CAP`] bytes, whichever binds first, and then declines. It
//! does not evict: a decline routes that reference through the copying rails,
//! which land the same frame, whereas evicting a window a live submission still
//! names would land undefined behavior in the driver. The caps are what keep the
//! host's locked memory and the driver's walk bounded; the census in
//! [`super::pools::ResourcePools::host_ram_import_census`] is where a boot says
//! how close it came.
//!
//! # Every submission is charged for the whole live set
//!
//! RADV's amdgpu winsys treats a host-pointer import as unconditionally
//! resident: every live one is appended to every command submission's BO list,
//! whether or not that submission binds it. An ioctl trace shows a submission
//! that binds no guest memory at all still carrying the full set, and the charge
//! tracks bytes rather than windows. Measured on a Radeon RX 9070 XT (RADV
//! GFX1201), submit plus fence wait, min of five runs:
//!
//! ```text
//! live imported    per submit
//!         0 MiB      27.7 us
//!        16 MiB      47.2 us
//!       256 MiB     307.4 us
//!      1024 MiB     869.7 us
//! ```
//!
//! About 27.7 us plus 1 us per MiB live, or 4 ns per 4 KiB page. The control is
//! the same allocation count and the same dirty anonymous footprint with no
//! import, which stays at 27.7 us — so this is the import, not the memory.
//! Splitting one gibibyte across four times as many windows moved it by 1.7 %:
//! [`WINDOW_COUNT_CAP`] is not the knob, [`WINDOW_TOTAL_BYTE_CAP`] is.
//!
//! So the byte cap is a standing per-submission tax on the whole device, paid by
//! draws that never touch guest memory. A boot that reaches the cap and holds it
//! — which is what the `guest_import_levels` census has shown, at
//! `windows=16 mib=1024` within 30 s — is paying roughly 850 us on every
//! submission it makes. Weigh a raise against that, and read
//! `buffer_guest_imports` first: while it is zero the rail is removing no copies
//! at all, and the tax is the only thing it is doing.
//!
//! Whether another driver charges the same way is unmeasured; nothing here has
//! run the experiment off amdgpu.
//!
//! # What the import does not promise
//!
//! Freeing the memory ends the GPU's access, but nothing in the extension's
//! specification says the pages were pinned while it lived. amdgpu and the
//! NVIDIA driver call `get_user_pages` at import time in practice; that is an
//! observation about two drivers rather than a contract. The honest statement is
//! in [`crate::runtime::guest_ram`]'s module doc and is not repeated as a
//! guarantee here.

use ash::vk;

use crate::observe::Decline;
use crate::runtime::guest_ram::{GuestRamError, GuestRef, ImportWindow};

/// Bytes of one import a single window covers, where the range it was opened
/// for fits inside one bucket of that size.
///
/// 64 MiB is a ceiling on the driver's re-walk rather than a target for reuse:
/// see [`ImportWindow`] for the profile, and the module doc for why growing it
/// buys nothing. A full-screen frame at this device's largest advertised mode is
/// well under it, so a scanout-sized reference costs one window.
pub(crate) const GUEST_IMPORT_WINDOW_BYTES: u64 = 1 << 26;

/// Windows that may be live at once, across every import.
///
/// The byte cap below binds first for windows of the usual size; this one bounds
/// the small-RAMBlock case, where an import shorter than one bucket is taken
/// whole and many of them would otherwise accumulate for very few bytes.
const WINDOW_COUNT_CAP: usize = 64;

/// Guest RAM this device may have registered for DMA at once.
///
/// The number the host pays in locked memory, and the ceiling on what one
/// submission's revalidation can walk.
const WINDOW_TOTAL_BYTE_CAP: u64 = 1 << 30;

/// One window living on the GPU as a bindable buffer, with no copy between it
/// and the guest's own view of those bytes.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ImportedHostRam {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    /// Which import, and which of its bytes, the buffer spans. The buffer covers
    /// the whole window, so every range [`ImportWindow::offset_of`] admits is a
    /// valid offset into it.
    pub window: ImportWindow,
}

impl ImportedHostRam {
    /// Release both halves. Freeing the memory is what ends the GPU's access to
    /// guest RAM, so it must run even on a teardown path that is otherwise
    /// giving up.
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

/// A check that stopped guest RAM from becoming a bindable buffer.
///
/// Every variant is a distinct check with its own slug. An import that fails at
/// `vkAllocateMemory` and one the device declined a memory type for are two
/// different findings — the first is usually the driver refusing the pointer's
/// backing, the second is a memory-type intersection that came out empty — and a
/// shared reason would leave a reader unable to tell them apart.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HostRamDecline {
    /// This device cannot import a host pointer at all. Carries the rung so the
    /// log says which check refused; expected on every host without the
    /// extension and on any host where an operator turned the rail off.
    Unsupported {
        rung: crate::backend::vulkan::caps::HostPointerImport,
    },
    /// `vkGetMemoryHostPointerPropertiesEXT` declined the pointer, or no memory
    /// type it named also satisfies what this device wants for guest memory.
    NoImportableMemoryType { host_base: usize },
    /// `vkCreateBuffer` over the whole window failed.
    CreateBuffer { result: vk::Result },
    /// The buffer the driver made needs more bytes than the window has. A window
    /// is sized to bytes the RAMBlock backs and may not be rounded up: past the
    /// end of the last one lies this process's own memory.
    TooSmall { required: u64, available: u64 },
    /// `vkAllocateMemory` with the chained import failed. On most drivers this
    /// is the pointer being refused — not fd-backed, not aligned, or not a
    /// mapping the driver can take a reference on.
    AllocateMemory { result: vk::Result },
    /// `vkBindBufferMemory` failed after a successful import.
    BindBuffer { result: vk::Result },
    /// The reference did not survive its own bound. Carries the check that
    /// refused, from [`crate::runtime::guest_ram`].
    Bound { inner: GuestRamError },
    /// [`WINDOW_COUNT_CAP`] windows are already live. Expected only where the
    /// shim reports many RAMBlocks shorter than one bucket, since a bucket-sized
    /// window reaches [`Self::WindowByteCap`] long before this.
    WindowCountCap { windows: usize },
    /// A window of this size would take the total past
    /// [`WINDOW_TOTAL_BYTE_CAP`]. The reference falls to the copying rails,
    /// which land the same frame for a memcpy.
    WindowByteCap { imported: u64, want: u64 },
    /// A window that does not hold the range it was opened for.
    ///
    /// A healthy zero: [`crate::runtime::guest_ram::GuestRamImport::window`]
    /// extends a window forward until it covers the range, so a firing means
    /// that derivation broke. Checked before the import rather than after, so
    /// the break costs arithmetic instead of a `get_user_pages` over the window.
    WindowMissesRange { offset: u64, len: u64 },
}

impl Decline for HostRamDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::Unsupported { .. } => "host_ram_import_unsupported",
            Self::NoImportableMemoryType { .. } => "host_ram_import_no_importable_memory_type",
            Self::CreateBuffer { .. } => "host_ram_import_create_buffer",
            Self::TooSmall { .. } => "host_ram_import_too_small",
            Self::AllocateMemory { .. } => "host_ram_import_allocate_memory",
            Self::BindBuffer { .. } => "host_ram_import_bind_buffer",
            Self::WindowCountCap { .. } => "host_ram_import_window_count_cap",
            Self::WindowByteCap { .. } => "host_ram_import_window_byte_cap",
            Self::WindowMissesRange { .. } => "host_ram_import_window_misses_range",
            // The inner check is the diagnosis. Forwarding rather than adding a
            // slug keeps one name per check across the two modules.
            Self::Bound { inner } => inner.slug(),
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::Unsupported { rung } => vec![("rung", rung.slug().to_string())],
            Self::NoImportableMemoryType { host_base } => {
                vec![("host_base", format!("{host_base:#x}"))]
            }
            Self::CreateBuffer { result }
            | Self::AllocateMemory { result }
            | Self::BindBuffer { result } => vec![("result", format!("{result:?}"))],
            Self::TooSmall {
                required,
                available,
            } => vec![
                ("required", required.to_string()),
                ("available", available.to_string()),
            ],
            Self::WindowCountCap { windows } => vec![
                ("windows", windows.to_string()),
                ("cap", WINDOW_COUNT_CAP.to_string()),
            ],
            Self::WindowByteCap { imported, want } => vec![
                ("imported", imported.to_string()),
                ("want", want.to_string()),
                ("cap", WINDOW_TOTAL_BYTE_CAP.to_string()),
            ],
            Self::WindowMissesRange { offset, len } => vec![
                ("offset", format!("{offset:#x}")),
                ("len", len.to_string()),
            ],
            Self::Bound { inner } => inner.fields(),
        }
    }
}

crate::observe::decline_display!(HostRamDecline);

/// A guest-memory range the engine can bind right now.
///
/// `offset` is into `buffer`, which spans the window the range fell in rather
/// than the whole RAMBlock, so it is not the range's offset in its import.
/// Producing it is the one place the two are related.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct BoundGuestRam {
    pub buffer: vk::Buffer,
    pub offset: vk::DeviceSize,
    pub len: vk::DeviceSize,
    /// Bytes from `offset` to the first byte the caller asked for, added by
    /// widening the range to the device's import granularity. A caller that
    /// binds `offset` and reads from byte zero of it reads this many bytes early.
    pub head: vk::DeviceSize,
}

/// Every window this device has imported.
///
/// A `Vec` because the lookup is containment rather than equality — a range is
/// served by whichever live window holds it — and because the caps keep the scan
/// short enough that its cost is below the bounds check that precedes it.
#[derive(Default)]
pub(crate) struct HostRamImports {
    live: Vec<ImportedHostRam>,
}

impl HostRamImports {
    /// Resolve `guest_ref` to a bindable range, opening a window over it if no
    /// live window already holds those bytes.
    ///
    /// # Safety
    ///
    /// `ctx` must be the live device context, and the import's host base must
    /// still be a mapping in this process — which it is for the VM's lifetime,
    /// because it is QEMU's own RAMBlock mapping.
    pub(crate) unsafe fn bind(
        &mut self,
        ctx: &super::context::DeviceContext,
        guest_ref: &GuestRef,
    ) -> Result<BoundGuestRam, HostRamDecline> {
        // The bound first, so a reference that cannot name its own bytes never
        // reaches an import call.
        let range = guest_ref
            .bound()
            .map_err(|inner| HostRamDecline::Bound { inner })?;
        let import = guest_ref.import();
        let id = import.id();
        let head = guest_ref.head();

        if let Some((buffer, offset)) = self
            .live
            .iter()
            .find_map(|l| Some((l.buffer, l.window.offset_of(id, range)?)))
        {
            return Ok(BoundGuestRam {
                buffer,
                offset,
                len: range.len,
                head,
            });
        }

        let window = import
            .window(range, GUEST_IMPORT_WINDOW_BYTES)
            .map_err(|inner| HostRamDecline::Bound { inner })?;
        // Before the import, so a window that does not cover its own range costs
        // arithmetic rather than a page walk over every byte of it.
        let offset =
            window
                .offset_of(id, range)
                .ok_or(HostRamDecline::WindowMissesRange {
                    offset: range.offset,
                    len: range.len,
                })?;
        self.affordable(window.len())?;

        let made = unsafe { import_window(ctx, &window) }?;
        self.live.push(made);
        crate::observe::off(format!(
            "host_ram_window import={id} offset={:#x} len={:#x} windows={} bytes={:#x}",
            window.offset(),
            window.len(),
            self.live.len(),
            self.imported_bytes(),
        ));
        Ok(BoundGuestRam {
            buffer: made.buffer,
            offset,
            len: range.len,
            head,
        })
    }

    /// Whether one more window of `bytes` fits inside both caps.
    ///
    /// Separate refusals rather than one: a boot that stopped importing because
    /// it ran out of window slots and one that stopped because it ran out of
    /// bytes want different numbers changed.
    fn affordable(&self, bytes: u64) -> Result<(), HostRamDecline> {
        if self.live.len() >= WINDOW_COUNT_CAP {
            return Err(HostRamDecline::WindowCountCap {
                windows: self.live.len(),
            });
        }
        let imported = self.imported_bytes();
        if imported.saturating_add(bytes) > WINDOW_TOTAL_BYTE_CAP {
            return Err(HostRamDecline::WindowByteCap {
                imported,
                want: bytes,
            });
        }
        Ok(())
    }

    /// Release every window. Called on device teardown, before the device goes.
    ///
    /// # Safety
    ///
    /// No submission may still reference any imported buffer.
    pub(crate) unsafe fn destroy_all(&mut self, device: &ash::Device) {
        for live in self.live.drain(..) {
            unsafe { live.destroy(device) };
        }
    }

    /// Bytes of guest RAM this device currently has registered for DMA, for the
    /// census. Bounded by [`WINDOW_TOTAL_BYTE_CAP`].
    pub(crate) fn imported_bytes(&self) -> u64 {
        self.live.iter().map(|l| l.window.len()).sum()
    }

    /// How many windows are live. Rises with the span of guest RAM the workload
    /// touches rather than with the number of references into it — a count that
    /// tracks draws is the per-resource import the model exists to avoid, and a
    /// count parked at [`WINDOW_COUNT_CAP`] is a boot spending every later
    /// reference on the copying rails.
    pub(crate) fn len(&self) -> usize {
        self.live.len()
    }
}

/// Import one window's host span.
///
/// # Safety
///
/// As [`HostRamImports::bind`].
unsafe fn import_window(
    ctx: &super::context::DeviceContext,
    window: &ImportWindow,
) -> Result<ImportedHostRam, HostRamDecline> {
    use crate::backend::vulkan::caps::host_pointer::GUEST_IMPORT_USAGE;

    let Some(loader) = ctx.external_memory_host.as_ref() else {
        return Err(HostRamDecline::Unsupported {
            rung: ctx.caps.host_pointer.rung,
        });
    };
    const HANDLE_TYPE: vk::ExternalMemoryHandleTypeFlags =
        vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT;

    let host_base = window.host_ptr();
    let size = window.len();

    // Which memory types will accept *this* pointer. Asked before anything is
    // created, because the answer is a property of the mapping rather than of
    // the device — and it goes through the one memory-type selector this
    // backend has, so the ranking is not restated here.
    //
    // `Upload` is the class: guest RAM is host memory the GPU reaches, which is
    // exactly what that preference describes. On a discrete host the selector
    // will land on a host-visible type, and the copy into VRAM is a separate
    // decision made by the caller, not by this import.
    let req = ctx
        .caps
        .memory_request(crate::backend::vulkan::caps::MemoryClass::Upload);
    let memory_type_index = unsafe {
        crate::backend::vulkan::caps::host_pointer::import_memory_type(
            loader,
            &ctx.memory_properties,
            host_base as *const std::ffi::c_void,
            &req,
        )
    }
    .ok_or(HostRamDecline::NoImportableMemoryType { host_base })?;

    let mut external = vk::ExternalMemoryBufferCreateInfo::default().handle_types(HANDLE_TYPE);
    let create = vk::BufferCreateInfo::default()
        .size(size)
        .usage(GUEST_IMPORT_USAGE)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .push_next(&mut external);
    let buffer = unsafe { ctx.device.create_buffer(&create, None) }
        .map_err(|result| HostRamDecline::CreateBuffer { result })?;

    // From here every failure must destroy the buffer before returning, so the
    // work is done in a closure and the cleanup happens once at the end.
    let bound = (|| {
        let reqs = unsafe { ctx.device.get_buffer_memory_requirements(buffer) };
        if reqs.size > size {
            // Not rounded up. Past the end of the last window lies memory the
            // RAMBlock does not back, and handing the GPU write access to it is
            // the one stray the bound exists to prevent.
            return Err(HostRamDecline::TooSmall {
                required: reqs.size,
                available: size,
            });
        }
        if reqs.memory_type_bits & (1u32 << memory_type_index) == 0 {
            return Err(HostRamDecline::NoImportableMemoryType { host_base });
        }

        let mut host_import = vk::ImportMemoryHostPointerInfoEXT::default()
            .handle_type(HANDLE_TYPE)
            .host_pointer(host_base as *mut std::ffi::c_void);
        let allocate = vk::MemoryAllocateInfo::default()
            .allocation_size(size)
            .memory_type_index(memory_type_index)
            .push_next(&mut host_import);
        let memory = unsafe { ctx.device.allocate_memory(&allocate, None) }
            .map_err(|result| HostRamDecline::AllocateMemory { result })?;

        match unsafe { ctx.device.bind_buffer_memory(buffer, memory, 0) } {
            Ok(()) => Ok(ImportedHostRam {
                buffer,
                memory,
                window: *window,
            }),
            Err(result) => {
                // Freeing the memory is what ends the GPU's access to the
                // pointer, so it happens even on this failure path.
                unsafe { ctx.device.free_memory(memory, None) };
                Err(HostRamDecline::BindBuffer { result })
            }
        }
    })();

    if bound.is_err() {
        unsafe { ctx.device.destroy_buffer(buffer, None) };
    }
    bound
}

/// Channel order as a decline field reads, so the two sides of an
/// [`GuestWriteDecline::OrderMismatch`] name themselves rather than printing
/// `true`/`false` a reader has to know the polarity of.
fn order_name(bgra: bool) -> &'static str {
    if bgra {
        "bgra"
    } else {
        "rgba"
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
    /// The device cannot import guest RAM at all. Carries the rung so the log
    /// says which check refused; expected on every host without the extension.
    Unsupported {
        rung: crate::backend::vulkan::caps::HostPointerImport,
    },
    /// The resident's physical channel order is not the order the destination
    /// stores, so landing it would need an R/B exchange — which an image→buffer
    /// copy cannot perform. The copying rail's per-row conversion is where that
    /// lives.
    ///
    /// Stated as a disagreement between the two rather than as "the resident is
    /// not BGRA", because both orders reach this call: a type-11 mapping's pages
    /// are guest scanout order and a GVA render target's are whatever the guest
    /// declared for it. A rail that spelled the rule as one fixed order refused
    /// every RGBA destination it could have served unchanged.
    OrderMismatch { resident_bgra: bool, want_bgra: bool },
    /// The resident's geometry is not the geometry the window promised the
    /// guest. Copying anyway would land one extent's pixels under another's row
    /// pitch.
    GeometryMoved {
        resident_width: u32,
        resident_height: u32,
        want_width: u32,
        want_height: u32,
    },
    /// The frame's last byte falls past the end of the range the runtime asked
    /// for.
    ///
    /// Not the same check as the import bound, and that is why it is kept: the
    /// runtime sizes the request from the mapping's page plan and the engine
    /// computes the extent from the resident's own geometry and row pitch. Two
    /// independently-derived numbers, and a disagreement between them is a
    /// frame that would land under the wrong pitch.
    WindowTooSmall { need: u64, have: u64 },
    /// The import itself declined; the inner reason names the step.
    Import { inner: HostRamDecline },
}

impl Decline for GuestWriteDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::Unsupported { .. } => "gpu_writeback_unsupported",
            Self::OrderMismatch { .. } => "gpu_writeback_order_mismatch",
            Self::GeometryMoved { .. } => "gpu_writeback_geometry_moved",
            Self::WindowTooSmall { .. } => "gpu_writeback_window_too_small",
            // The inner decline's own slug, so a driver that refuses the pointer
            // and a range that is too short stay as distinguishable here as they
            // are at the import site.
            Self::Import { inner } => inner.slug(),
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::Unsupported { rung } => vec![("rung", rung.slug().to_string())],
            Self::OrderMismatch {
                resident_bgra,
                want_bgra,
            } => vec![
                ("resident", order_name(*resident_bgra).to_string()),
                ("want", order_name(*want_bgra).to_string()),
            ],
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::vulkan::caps::HostPointerImport;

    /// One slug per check. Two sharing one would mean watching a slug fire and
    /// still not knowing whether the driver refused the pointer or the memory
    /// type intersection came out empty.
    #[test]
    fn every_decline_has_its_own_slug() {
        let all = [
            HostRamDecline::Unsupported {
                rung: HostPointerImport::Unqueried,
            },
            HostRamDecline::NoImportableMemoryType { host_base: 0 },
            HostRamDecline::CreateBuffer {
                result: vk::Result::ERROR_UNKNOWN,
            },
            HostRamDecline::TooSmall {
                required: 0,
                available: 0,
            },
            HostRamDecline::AllocateMemory {
                result: vk::Result::ERROR_UNKNOWN,
            },
            HostRamDecline::BindBuffer {
                result: vk::Result::ERROR_UNKNOWN,
            },
            HostRamDecline::WindowCountCap { windows: 0 },
            HostRamDecline::WindowByteCap {
                imported: 0,
                want: 0,
            },
            HostRamDecline::WindowMissesRange { offset: 0, len: 0 },
        ];
        let mut slugs: Vec<_> = all.iter().map(|d| d.slug()).collect();
        let count = slugs.len();
        slugs.sort_unstable();
        slugs.dedup();
        assert_eq!(slugs.len(), count, "two checks share a slug");
        for slug in slugs {
            assert!(slug.starts_with("host_ram_import_"), "{slug}");
        }
    }

    /// A bound refusal keeps the inner check's name rather than being renamed
    /// at the boundary. The two modules are one rail and a reader greps one
    /// vocabulary.
    #[test]
    fn a_bound_refusal_forwards_the_check_that_refused() {
        let inner = GuestRamError::SliceEndPastImport {
            end: 0x1001,
            import_len: 0x1000,
        };
        let outer = HostRamDecline::Bound { inner };
        assert_eq!(outer.slug(), inner.slug());
        assert_eq!(outer.fields(), inner.fields());
    }

    /// A fresh map holds nothing and reports nothing imported.
    #[test]
    fn an_empty_map_reports_no_imports() {
        let imports = HostRamImports::default();
        assert_eq!(imports.len(), 0);
        assert_eq!(imports.imported_bytes(), 0);
    }

    /// A window's whole point is that it is a bounded slice of guest RAM. This
    /// is the arithmetic behind that claim on this pathway's shape: a 16 GiB
    /// RAMBlock and a full-screen reference into it must reach the driver as
    /// tens of megabytes, and the total across a boot must stay inside a cap the
    /// host can afford to have locked.
    ///
    /// Whole-RAMBlock import is what this replaces, and it fails here on the
    /// first assertion.
    #[test]
    fn a_window_over_this_pathway_is_a_fraction_of_the_block() {
        use crate::runtime::guest_ram::{GuestRamImport, GuestRamRegion};

        const GUEST_RAM: u64 = 16 << 30;
        // 3440x1440 at four bytes a pixel: the largest mode the display model
        // advertises, so the largest single reference a scanout can make.
        const FRAME: u64 = 3440 * 1440 * 4;

        let import = GuestRamImport::new(
            GuestRamRegion {
                gpa_base: 0x1_0000_0000,
                host_va: 0x7f00_0000_0000,
                len: GUEST_RAM,
            },
            0x1000,
        )
        .expect("a page-aligned 16 GiB block is importable");
        let slice = import
            .slice(GUEST_RAM / 2 + 0x3000, FRAME)
            .expect("a frame inside the block is sliceable");
        let range = import.resolve(&slice).expect("its own slice resolves");
        let window = import
            .window(range, GUEST_IMPORT_WINDOW_BYTES)
            .expect("a bucket-sized grain is a power of two");

        assert!(
            window.len() < GUEST_IMPORT_WINDOW_BYTES + FRAME,
            "a frame must pull one window's worth, got {:#x}",
            window.len()
        );
        // The total cap must bound guest RAM rather than restate it: a cap at or
        // above `-m` is the whole-RAMBlock import wearing a budget's name.
        const { assert!(WINDOW_TOTAL_BYTE_CAP < GUEST_RAM) };
        assert_eq!(
            window.offset_of(import.id(), range),
            Some(range.offset - window.offset()),
            "the window must hold the range it was opened for"
        );
    }

    /// Both caps refuse, and each says which one did. A boot that stopped
    /// importing wants to know whether to raise the slot count or the byte
    /// budget, and one shared refusal answers neither.
    #[test]
    fn each_cap_refuses_under_its_own_name() {
        let full = HostRamDecline::WindowCountCap {
            windows: WINDOW_COUNT_CAP,
        };
        assert_eq!(full.slug(), "host_ram_import_window_count_cap");
        assert!(full
            .fields()
            .iter()
            .any(|(k, v)| *k == "cap" && v == &WINDOW_COUNT_CAP.to_string()));

        let broke = HostRamDecline::WindowByteCap {
            imported: WINDOW_TOTAL_BYTE_CAP,
            want: GUEST_IMPORT_WINDOW_BYTES,
        };
        assert_eq!(broke.slug(), "host_ram_import_window_byte_cap");
        assert!(broke
            .fields()
            .iter()
            .any(|(k, v)| *k == "cap" && v == &WINDOW_TOTAL_BYTE_CAP.to_string()));
    }

    /// An empty map affords a window; a map already holding the byte cap does
    /// not, and refuses by the byte cap rather than the slot count. The caps are
    /// what bound the host's locked memory, so a budget that admitted one more
    /// window past them would make both numbers advisory.
    #[test]
    fn the_budget_refuses_once_a_cap_is_reached() {
        let empty = HostRamImports::default();
        assert!(empty.affordable(GUEST_IMPORT_WINDOW_BYTES).is_ok());
        assert!(empty.affordable(WINDOW_TOTAL_BYTE_CAP).is_ok());
        assert!(matches!(
            empty.affordable(WINDOW_TOTAL_BYTE_CAP + 1),
            Err(HostRamDecline::WindowByteCap { .. })
        ));
        assert!(matches!(
            empty.affordable(u64::MAX),
            Err(HostRamDecline::WindowByteCap { .. })
        ));
    }
}
