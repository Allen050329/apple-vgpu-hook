//! Register-window and FIFO wire constants from the live Reims VGPU contract.
//!
//! Sources: `apple-pv-gpu.h`, `reims_vgpu_fifo_format.h`.
//! Values are protocol constants — not content heuristics.

/// Gfx MMIO window size (16 KiB); bounds the sparse register store.
///
/// This constant is the owner. `REIMS_VGPU_GFX_MMIO_SIZE` in
/// `include/reims_vgpu_qemu_abi.h` mirrors it and both shims size their region
/// from the header — the sysbus gfx window on one, BAR0 on the other — so the
/// window the guest can address and the bound Rust indexes cannot part without
/// `the_abi_header_agrees_on_the_gfx_window_size` failing.
///
/// The iosfc window's size is not mirrored here. QEMU declares that region
/// (`REIMS_VGPU_MMIO_IOSFC_MMIO_SIZE` in `reims-vgpu-mmio.c`) and Rust only
/// needs a bound for state it keeps per offset, which the iosfc rail does not do
/// — it decodes five named registers and ignores the rest. A second unread copy
/// of the iosfc size would be a source of truth nothing checks against the one
/// that actually sizes the `MemoryRegion`.
pub const GFX_MMIO_SIZE: u64 = 0x4000;

/// Control block base inside the gfx window.
pub const REG_BASE: u64 = 0x1000;

pub const GFX_REG_CONTROL_FIFO: u64 = 0x1000;
pub const GFX_REG_FIFO_LENGTH: u64 = 0x1004;
pub const GFX_REG_FIFO_WRITTEN: u64 = 0x1008;
pub const GFX_REG_FIFO_READ: u64 = 0x100c;
pub const GFX_REG_FIFO_START: u64 = 0x1010;
pub const GFX_REG_INTR_STATUS_DISP: u64 = 0x1014;
pub const GFX_REG_INTR_STATUS_GPU: u64 = 0x1018;
pub const GFX_REG_ROOT_PAGE: u64 = 0x101c;
pub const GFX_REG_CHILD_DOORBELL: u64 = 0x1020;
pub const GFX_REG_MAIN_KICK: u64 = 0x1024;
pub const GFX_REG_CHILD_REPLAY_DOORBELL: u64 = 0x1028;
pub const GFX_REG_INTR_FAULT: u64 = 0x102c;
pub const GFX_REG_FIFO_BASE_PAGE: u64 = 0x1030;
pub const GFX_REG_VERSION: u64 = 0x1034;

pub const GFX_REG_EFI_DISPLAY: u64 = 0x1200;
pub const GFX_REG_EFI_MODE_COUNT: u64 = 0x1204;
pub const GFX_REG_EFI_MODE_SELECT: u64 = 0x1208;
pub const GFX_REG_EFI_MODE_SIZE: u64 = 0x120c;
pub const GFX_REG_EFI_FB_START: u64 = 0x1210;
pub const GFX_REG_EFI_FB_LENGTH: u64 = 0x1214;
pub const GFX_REG_EFI_FB_DEPTH: u64 = 0x1218;
pub const GFX_REG_EFI_FB_MODE: u64 = 0x121c;
pub const GFX_REG_EFI_DISPLAY_IRQ: u64 = 0x1220;
pub const GFX_REG_EFI_STRIDE_ALIGN: u64 = 0x1224;
pub const GFX_REG_EFI_FB_STRIDE: u64 = 0x1228;
pub const GFX_REG_EFI_DISPLAY_PORTS: u64 = 0x122c;
pub const GFX_REG_EFI_BUILTIN_CONNECTED: u64 = 0x1234;

pub const IOSFC_REG_RING_BASE: u64 = 0x1000;
pub const IOSFC_REG_CAPACITY: u64 = 0x1008;
pub const IOSFC_REG_DESC_TABLE: u64 = 0x1010;
pub const IOSFC_REG_PRODUCER: u64 = 0x1018;
pub const IOSFC_REG_CONSUMER: u64 = 0x1020;

/// The single EFI mode this device **advertises**, not a dimension observed in
/// a guest.
///
/// It is reported to the firmware through `GFX_REG_EFI_MODE_SIZE` as
/// `(width << EFI_MODE_WIDTH_SHIFT) | height`, with [`EFI_MODE_COUNT`] of 1, so
/// the pre-boot console geometry is this device's own declaration and the guest
/// has no other mode to select. Comparing a request against it — as
/// `scanout::paint_efi_console` does — is checking the contract this device
/// published, not special-casing a pixel size.
///
/// Recorded because the absence of this note actively misled a reader: a review
/// pass scored the `width != EFI_BOOT_WIDTH` check as a high-confidence
/// "special-cased for an observed pixel dimension" violation. The value is a
/// choice, which is fine; a choice with no stated basis is what reads as a guess.
///
/// This constant and [`EFI_BOOT_HEIGHT`] are the owners.
/// `REIMS_VGPU_EFI_BOOT_{WIDTH,HEIGHT}` in `include/reims_vgpu_qemu_abi.h`
/// mirror them, and both shims size the pre-boot `DisplaySurface` from the
/// header rather than from a private copy — because a shim painting at one
/// geometry into a console this file refuses at another is a failure with the
/// two numbers in different files.
/// `the_abi_header_agrees_on_the_efi_boot_mode` fails if they part.
pub const EFI_BOOT_WIDTH: u32 = 1920;
/// Height of the advertised EFI mode. See [`EFI_BOOT_WIDTH`].
pub const EFI_BOOT_HEIGHT: u32 = 1080;
pub const EFI_MODE_WIDTH_SHIFT: u32 = 16;
pub const EFI_MODE_COUNT: u32 = 1;
pub const EFI_STRIDE_ALIGNMENT: u32 = 64;
pub const EFI_DISPLAY_PORT_COUNT: u32 = 1;
pub const EFI_BUILTIN_CONNECTED: u32 = 1;

/// How many channel ids this device has state for, including the root FIFO.
///
/// # 32 has headroom, and that is measured rather than assumed
///
/// The guest driver does not allocate channels; it **enumerates** them, and
/// every one comes from one of exactly two places:
///
/// * four fixed channels the accelerator creates at setup, with their ids as
///   immediates — 1 `Exec`, 2 `Immediate`, 3 `Uploads`, 4 `Downloads` — alongside
///   the root FIFO at 0, which does not use a child register block at all;
/// * one per display pipe, at `pipe_index + 5`, where the pipe index is bounded
///   by the driver's own `index <= 7`.
///
/// So the highest id this guest can ever ring is **12**, and no path allocates,
/// reuses or pools an id. Command queues do not get their own: every queue binds
/// the one accelerator-wide `Exec` channel, so the count does not scale with
/// processes, Metal devices or submissions.
///
/// The pipe count is the only negotiated half, and it is negotiated *downwards*:
/// the guest reads [`GFX_REG_EFI_DISPLAY_PORTS`] and clamps it into `1..=8`
/// itself, so this device cannot widen the range by advertising more. At the
/// [`EFI_DISPLAY_PORT_COUNT`] published here it creates channels 0..=5.
///
/// The bound is therefore not tight, and it is not load-bearing either — see
/// [`accept_child_channel`] for what a refusal past it would cost. It is pinned
/// from above by the three `u32` masks below and from *use* by the layout of the
/// root page, which has room for `(page_size - CHILD_REG_BLOCK_OFFSET) /
/// CHILD_REG_BLOCK_STRIDE` blocks — 153 on a 4 KiB page — and which the guest
/// indexes with no bound check of its own.
pub const MAX_CHANNELS: usize = 32;

/// `active_child_mask`, `pending.child_mask` and `child_doorbell_rung` are each
/// a `u32` carrying one bit per channel, and every producer reaches them with a
/// bare `1u32 << channel_id`. That is only defined because a channel id is
/// bounded by 32 — a `MAX_CHANNELS` above `u32::BITS` would make every one of
/// those shifts overflow, which Rust panics on in debug and wraps in release.
/// The masks would have to widen with the constant, so this refuses the change
/// at the constant rather than at the four shift sites.
const _: () = assert!(MAX_CHANNELS <= u32::BITS as usize);

/// Whether `channel_id` names a child channel this device has.
///
/// Channel 0 is the root/main FIFO, not a child, so it is refused here for the
/// same reason an id past the end is: neither indexes `child_rings` and neither
/// has a mask bit.
///
/// The rule had seven copies in four files and two mutually inverted spellings
/// — `channel_id == 0 || channel_id as usize >= MAX_CHANNELS` in three places
/// and `ch >= 1 && (ch as usize) < MAX_CHANNELS` in four. An inverted copy is
/// the worst kind: reading the two side by side proves nothing about the two
/// you did not put side by side.
pub const fn is_child_channel(channel_id: u32) -> bool {
    channel_id >= 1 && (channel_id as usize) < MAX_CHANNELS
}

/// [`is_child_channel`], reporting the refusal when the answer is no.
///
/// The three sites that gate on a channel id are the two doorbell handlers
/// (locked and lock-free) and `ensure_child_ring`, and all three used to answer
/// this question and then say nothing: an `if` with no `else` at the first two,
/// and a `0` return at the third that is indistinguishable from "ring not valid
/// yet". A guest ringing channel 32 therefore set no mask bit, scheduled no
/// bottom half, and was never told — every command it queued there sits in the
/// ring forever, which is a stalled channel rather than a dropped record and so
/// does not even look like corruption from the guest's side.
///
/// `MAX_CHANNELS` is a bound this device imposes, not one the protocol states,
/// and nothing tells a guest what it is: `DEVICE_INFO_CAPS` advertises no
/// channel count, and the register blocks the guest indexes are inside its own
/// root page rather than in this device's MMIO window. What bounds a guest in
/// practice is its own enumeration, which stops at id 12 — see
/// [`MAX_CHANNELS`]. This report is what would say that reading had gone stale.
///
/// Latched per channel id, so a guest hammering one out-of-range doorbell costs
/// one line rather than one per ring; the census route counts every occurrence.
pub fn accept_child_channel(channel_id: u32, site: &'static str) -> bool {
    if is_child_channel(channel_id) {
        return true;
    }
    crate::runtime::drain::census::note_store_route("child_channel_out_of_range");
    if crate::observe::first_sight("channel_outside_device_range", u64::from(channel_id)) {
        crate::observe::fail(format!(
            "child_channel_out_of_range reason=channel_outside_device_range \
             site={site} channel={channel_id} max_channels={MAX_CHANNELS}"
        ));
    }
    false
}

/// Whether `mapping_id` names a mapping rather than "no mapping".
///
/// Zero is the device-wide sentinel for an unbound mapping — `runtime::draw`
/// branches on `mapping_id == 0` in more than a dozen places to mean the
/// attachment is addressed by GVA instead — so a record naming 0 is not a
/// record naming mapping 0, and creating `mappings[0]` for it produces state no
/// sentinel-aware reader will ever consult.
///
/// Four callers knew that and six did not: `runtime::texture` and the two
/// type-4 backing paths in `runtime::objects` refused zero, while the five
/// `DeviceState` mutators and `mapper::capture_published_request` bounded the
/// id from above only.
///
/// # There is no upper bound, and there never should have been one
///
/// This used to also require `mapping_id < MAX_MAPPINGS` (4096). Nothing was
/// indexed by that number. Every one of the five `DeviceState` mutators reaches
/// `self.mappings.entry(mapping_id).or_default()` on a `BTreeMap` keyed by the
/// full `u32`, and no other structure in the device is sized by mapping id — so
/// the bound allocated nothing, protected nothing, and its only effect was that
/// a guest naming id 4096 had its MAP, UNMAP, MappingInternal attach, device
/// descriptor and geometry silently refused, and every type-4 surface backing
/// with it. A mapping id is a full `u32` on the wire; the guest chooses it and
/// the device is not entitled to an opinion about how large it is.
///
/// Zero stays refused because it is a *meaning*, not a size: the sentinel is
/// read as "unbound" by the draw path, so a mapping stored under it could never
/// be served.
pub const fn is_mapping_id(mapping_id: u32) -> bool {
    mapping_id >= 1
}

/// Largest scanout / surface edge the device accepts, in pixels.
///
/// Derived from the allocation it bounds rather than from a device capability.
/// Host pixel buffers here are tightly packed BGRA8, so this edge squared times
/// 4 is the largest single surface the device can be asked to hold: 8192 gives
/// 256 MiB, which is the figure [`crate::runtime::surface_cache`]'s GVA cache
/// cap is reasoned against at its own eviction site. The wire fields are 16-bit
/// and would admit 65535 — a 16 GiB surface out of one corrupt guest word — so
/// a ceiling is required, and this is the product's.
///
/// This constant is the owner. `REIMS_VGPU_MAX_SCANOUT_DIM` in
/// `include/reims_vgpu_qemu_abi.h` mirrors it for the two QEMU shims, and
/// `the_abi_header_agrees_on_the_scanout_bound` fails if they drift.
pub const MAX_SCANOUT_DIM: u32 = 8192;

/// Which half of the scanout extent bound a `width`x`height` pair breaks.
///
/// Named rather than boolean because exactly one caller — `set_mapping_geom` —
/// has to say *which*, and it is the only one on a path where a typed decline
/// reaches the fail log. Every other caller asks
/// [`scanout_extent_fault`]`(..).is_none()`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScanoutExtentFault {
    WidthZero,
    HeightZero,
    WidthAboveBound,
    HeightAboveBound,
}

/// Whether a `width`x`height` pair is an extent this device will hold, and the
/// named reason when it is not.
///
/// One rule: an edge is at least one pixel and at most [`MAX_SCANOUT_DIM`].
/// It was written out eight times — the surface cache's four entry points, the
/// two scanout copies, `mapping_write::read_raw_rows`, and `set_mapping_geom` —
/// as a four-term `||` chain in seven of them and as four typed refusals in the
/// eighth. Eight copies of a bound whose whole job is to keep a corrupt 16-bit
/// guest word from sizing a 16 GiB allocation is eight places for the ceiling to
/// be raised in seven.
///
/// The zero test belongs here with the ceiling and not beside each caller's own
/// argument checks: a zero edge is not a small image, it is one whose row count
/// or row length makes every downstream `need` computation zero, so a buffer of
/// any size "fits" it and the copy silently writes nothing.
pub const fn scanout_extent_fault(width: u32, height: u32) -> Option<ScanoutExtentFault> {
    if width == 0 {
        Some(ScanoutExtentFault::WidthZero)
    } else if height == 0 {
        Some(ScanoutExtentFault::HeightZero)
    } else if width > MAX_SCANOUT_DIM {
        Some(ScanoutExtentFault::WidthAboveBound)
    } else if height > MAX_SCANOUT_DIM {
        Some(ScanoutExtentFault::HeightAboveBound)
    } else {
        None
    }
}

/// Shorthand for the seven callers that only need the verdict.
pub const fn scanout_extent_ok(width: u32, height: u32) -> bool {
    scanout_extent_fault(width, height).is_none()
}

// Single source of truth for the shifts: `contract::gva::PAGE_SHIFT_*`,
// re-exported rather than restated. There is **no** bare `PAGE_SIZE` /
// `PAGE_SHIFT` — those names defaulted to arm16K and caused x86 wild writes
// (stamp slots). Product code uses `state.page_size()` / `state.page_shift`;
// fixtures pick an arch-qualified name.
//
// The two `PAGE_SIZE_*` below are NOT the `contract::gva` constants of the same
// name: those are `u32` for page-offset masking, these are the `u64` widening
// the device's address arithmetic and its fixtures want. Both derive from the
// one re-exported shift, so they cannot disagree in value — but the names do
// collide, so import from one module or the other on purpose.
pub(crate) use crate::contract::gva::{pfn_to_gpa, PAGE_SHIFT_ARM64E, PAGE_SHIFT_X86};
// Gated with the fixtures that want them. That is the same statement the
// comment above makes, now enforced rather than asserted: no product path
// names either one, because product code goes through `state.page_size()`.
#[cfg(test)]
pub const PAGE_SIZE_ARM64E: u64 = 1u64 << PAGE_SHIFT_ARM64E;
#[cfg(test)]
pub const PAGE_SIZE_X86: u64 = 1u64 << PAGE_SHIFT_X86;

pub const PACKET_OPCODE: usize = 0x00;
pub const PACKET_STAMP_COUNT: usize = 0x02;
pub const PACKET_TOTAL_SIZE: usize = 0x04;
pub const PACKET_COMPLETION_STAMP: usize = 0x08;
pub const PACKET_HEADER_LEN: u32 = 12;
pub const PACKET_STAMP_LEN: u32 = 8;
pub const STAMP_INDEX_MASK: u32 = 0xffff;
pub const STAMP_SLOT_LEN: u32 = 4;

pub const ROOT_OP_WRAPPER: u16 = 0x01;
/// PVG `CmdDeleteTask` — live UnknownRootOpcode flood was root op `0x20`
/// total_size=16 (header 12 + task_id u32). Same id as child DeleteTask.
pub const ROOT_OP_DELETE_TASK: u16 = 0x20;
pub const ROOT_OP_DEVICE_INFO_MONTEREY: u16 = 0x2d;
pub const ROOT_OP_DEFINE_FIFO: u16 = 0x30;
pub const ROOT_OP_FREE_FIFO: u16 = 0x31;
pub const ROOT_OP_SET_OBJECT_LIST: u16 = 0x33;
pub const ROOT_OP_DEFINE_TASK2: u16 = 0x38;
pub const ROOT_OP_DEVICE_INFO_TAHOE: u16 = 0x3a;

pub const CHILD_OP_SETUP_SHARED_STATE: u16 = 0x01;
pub const CHILD_OP_ONLINE_ACK: u16 = 0x02;
pub const CHILD_OP_CURSOR_GLYPH: u16 = 0x04;
pub const CHILD_OP_CURSOR_SHOW: u16 = 0x05;
/// x86 Ventura/Tahoe display-pipe present (ch5): 12B `[disp][surface_id][stamp]`.
pub const CHILD_OP_PRESENT_X86: u16 = 0x06;
/// x86 present-with-gamma (surface_id typically @+8).
pub const CHILD_OP_PRESENT_GAMMA_X86: u16 = 0x07;
pub const CHILD_OP_DISPLAY_SWAP: u16 = 0x08;
/// PVG `CmdDeleteTask` (same opcode as root [`ROOT_OP_DELETE_TASK`]).
pub const CHILD_OP_DELETE_TASK: u16 = 0x20;
/// PVG `CmdUnmapMemory` (not map — [`CHILD_OP_MAP_MEMORY2`] is `0x39`).
pub const CHILD_OP_UNMAP_MEMORY: u16 = 0x22;
/// Display-channel flush: a fence carrying **stamps and no payload**.
///
/// The guest's display pipe emits this from the failure and teardown legs of a
/// present, on the pipe's own channel, to fence work it is about to abandon.
/// Alone among the child commands it allocates no command bytes at all, so its
/// packet is header plus stamps and a non-empty payload would be a contract
/// violation rather than a longer form.
///
/// There is nothing for this device to execute — retiring the stamps *is* the
/// whole obligation, and the drain does that for every accepted packet. It is
/// named here only so it stops reading as an unknown opcode: it was landing in
/// the `UnknownChildOpcode` arm, which reports a real Apple command as a device
/// defect, on the display channel, exactly when the present path is already in
/// trouble and the log is being read.
pub const CHILD_OP_FLUSH_CHANNEL_EVENT: u16 = 0x1e;
pub const CHILD_OP_DELETE_OBJECT: u16 = 0x25;
pub const CHILD_OP_PRESENT_FRAME: u16 = 0x28;
pub const CHILD_OP_SET_OBJECT_LIST: u16 = 0x33;
/// PVG CmdInvalidateResources.
pub const CHILD_OP_INVALIDATE_RESOURCES: u16 = 0x34;
/// PVG CmdSynchronizeResources.
pub const CHILD_OP_SYNCHRONIZE_RESOURCES: u16 = 0x35;
/// PVG CmdDeleteIOSurfaceBacking2.
pub const CHILD_OP_DELETE_IOSURFACE_BACKING2: u16 = 0x36;
pub const CHILD_OP_EXEC_INDIRECT2: u16 = 0x37;
pub const CHILD_OP_DEFINE_TASK2: u16 = 0x38;
/// PVG CmdMapMemory2 (task GPU-VA map).
pub const CHILD_OP_MAP_MEMORY2: u16 = 0x39;
/// PVG CmdGetComputeInfo (query). Archive material calls the same opcode
/// `present-frame-flush`; that reading is wrong and has no constant here.
pub const CHILD_OP_GET_COMPUTE_INFO: u16 = 0x3b;
/// PVG CmdReplacePhysical (`replacePhysical` → `{taskID, objectID}`).
/// Live fail log: ch2 op 0x3c total_size=20 (header+payload). Stamp-complete;
/// full rebind RE is open — accept so guest is not blocked on UnknownChildOpcode.
pub const CHILD_OP_REPLACE_PHYSICAL: u16 = 0x3c;
/// The guest asking what a heap texture would cost: its handler decodes a
/// `heap_query` request and writes size and alignment back through the reply
/// GVA the request names. A query rather than a state change — nothing in the
/// device model moves — but the guest blocks on the reply, so a refusal is a
/// stall and not a dropped command.
pub const CHILD_OP_CONFIG_40: u16 = 0x40;

/// `CmdDisplayTransaction3` (op 6) trailer `[pipe][surface][task]`: surface id
/// offset. `CmdDisplaySwapMapping` (op 8) is a different command with a
/// different payload — see `DISPLAY_SWAP_MAPPING`, which is not this offset.
pub const PRESENT_X86_SURFACE_ID: usize = 0x04;
/// The gamma variant (op 7) trailer is `[pipe][task][surface][gamma…]`, so its
/// surface and task words are swapped relative to op 6's.
pub const PRESENT_GAMMA_X86_SURFACE_ID: usize = 0x08;

pub const CHILD_REG_BLOCK_OFFSET: u64 = 0x400;
pub const CHILD_REG_BLOCK_STRIDE: u64 = 0x14;
pub const CHILD_REG_TAIL: u64 = 0x00;
pub const CHILD_REG_HEAD: u64 = 0x04;
/// The one word of the five-word child block nothing reads.
///
/// `drain::child` takes head, tail, stamp index and base PFN out of this block
/// every doorbell; this word sits between head and stamp index and no product
/// path touches it. It stays named for the reason [`DISPLAY_SWAP_DISPLAY`]
/// does — a guest field with no name is the one nobody notices being ignored —
/// and because deleting it leaves the block map skipping `0x08` with nothing
/// saying what lives there, which is how a later offset gets read from the
/// wrong word.
///
/// Whether ignoring it is correct is **not established**. It is named `CONTROL`
/// from the register block's shape rather than from a decoded write, and no
/// boot on this rig has sampled what the guest puts in it. `dead-state` reports
/// it on both arms every run; this comment is the triage, not a licence to cut.
#[allow(dead_code)] // named on purpose and read by nothing — see above.
pub const CHILD_REG_CONTROL: u64 = 0x08;
pub const CHILD_REG_STAMP_INDEX: u64 = 0x0c;
pub const CHILD_REG_BASE_PFN: u64 = 0x10;
pub const CHILD_RING_PFN_ENTRY_LEN: u64 = 4;

pub const DEVICE_INFO_REPLY_PAIR_LEN: usize = 8;
pub const DEVICE_INFO_TAHOE_COUNT: usize = 0x04;
pub const DEVICE_INFO_TAHOE_REPLY_PFN: usize = 0x08;
pub const DEVICE_INFO_MONTEREY_COUNT: usize = 0x00;
pub const DEVICE_INFO_MONTEREY_REPLY_PFN: usize = 0x04;

pub const DEFINE_TASK_RAW_ID: usize = 0x00;
pub const DEFINE_TASK_LENGTH: usize = 0x04;
pub const DEFINE_TASK_DIRECTORY_PFN: usize = 0x0c;
pub const DEFINE_TASK_LEN: usize = 16;
pub const DEFINE_TASK_ID_SHIFT: u32 = 1;

pub const SET_OBJECT_LIST_TASK_ID: usize = 0x00;
pub const SET_OBJECT_LIST_PFN: usize = 0x04;
pub const SET_OBJECT_LIST_COUNT: usize = 0x08;
pub const SET_OBJECT_LIST_LEN: usize = 12;

// `CmdDisplaySwapMapping`'s trailer is `[display][_][mapping]`. Only the
// mapping is read; the display index has no reader here, and it stays named
// because a decoded guest field with no name is the one nobody notices being
// ignored. Whether ignoring it is correct is not established — it needs the
// display count this device advertises, which no boot on this rig has
// measured. Its own length lives at `display_txn_trailer_len`.
#[allow(dead_code)] // named on purpose and read by nothing — see above.
pub const DISPLAY_SWAP_DISPLAY: usize = 0x00;
pub const DISPLAY_SWAP_MAPPING: usize = 0x08;

pub const CHILD_SHARED_STATE_INDEX: usize = 0x00;
pub const CHILD_SHARED_STATE_PFN: usize = 0x04;
pub const CHILD_SHARED_STATE_LEN: usize = 8;

pub const DISPLAY_SHARED_PENDING: u64 = 0x100;
pub const DISPLAY_SHARED_ENABLE_MASK: u64 = 0x104;
/// VBL pending bit (`signalVBLInterrupt`).:
/// ~60 Hz vblank = page+0x100 bit0 + 0x1014 + MSI so the compositor keeps
/// presenting. Without this the guest can stick presenting clear-only flip
/// buffers while content lands on intermediate mids.
pub const DISPLAY_VBL_EVENT_MASK: u32 = 1 << 0;
/// Present-complete pending bit (PVG `_presentMappedSurface` completion block:
/// set bit 1 at `sharedState+0x100`, read `+0x104`, `displayPokePort:` when
/// the guest asked to be notified). Guest `handleHostInterrupt` read-clears
/// the pending word; bit index convention matches ONLINE (bit 2).
pub const DISPLAY_PRESENT_EVENT_MASK: u32 = 1 << 1;
pub const DISPLAY_ONLINE_EVENT_MASK: u32 = 1 << 2;
/// Cursor position / show in the display shared-state page (GPA).
pub const DISPLAY_SHARED_CURSOR_POS: u64 = 0xe00;
pub const DISPLAY_SHARED_CURSOR_SHOW: u64 = 0xe04;
/// Host-advertised HW cursor capability (archive / PVG).
pub const DISPLAY_SHARED_CURSOR_MAX_WH: u64 = 0x18;
pub const DISPLAY_SHARED_CURSOR_FEATURES: u64 = 0x20;
pub const DISPLAY_CURSOR_FEATURE_SHOW: u32 = 1;
pub const DISPLAY_CURSOR_FEATURE_MOVE: u32 = 2;
pub const DISPLAY_CURSOR_FEATURE_HW: u32 =
    DISPLAY_CURSOR_FEATURE_SHOW | DISPLAY_CURSOR_FEATURE_MOVE;

/// Display descriptor shared page.
pub const DISPLAY_DESC_SERIAL: u64 = 0x00;
pub const DISPLAY_DESC_PRODUCT_NAME: u64 = 0x04;
pub const DISPLAY_DESC_INDEX: u64 = 0x12;
pub const DISPLAY_DESC_WIDTH_MM: u64 = 0x14;
pub const DISPLAY_DESC_HEIGHT_MM: u64 = 0x16;
pub const DISPLAY_DESC_FEATURES: u64 = 0x1c;
/// Timing-element **count** (not a pixel width — large values hang the guest).
pub const DISPLAY_DESC_TIMING_COUNT: u64 = 0x208;

/// Modes advertised to AppleParavirtDisplay (archive apple_pv_gpu_display_setup).
pub const DISPLAY_MODE_EFI_W: u16 = 1920;
pub const DISPLAY_MODE_EFI_H: u16 = 1080;
pub const DISPLAY_MODE1_W: u16 = 1440;
pub const DISPLAY_MODE1_H: u16 = 1080;
pub const DISPLAY_MODE2_W: u16 = 1280;
pub const DISPLAY_MODE2_H: u16 = 1024;
/// 4K UHD, advertised at `DISPLAY_REFRESH_HZ` (120) like every other mode so the
/// guest can select native 3840×2160 @ 120 Hz. 3840 < `MAX_SCANOUT_DIM` (8192);
/// a 4K BGRA8 surface is 33 MiB (slab bucket 6). The scanout/present/host-window
/// geometry is dynamic (follows the presented surface), so no other constant
/// changes to run the desktop at 4K — see [[scanout-bridge]] mode-switch contract.
pub const DISPLAY_MODE3_W: u16 = 3840;
pub const DISPLAY_MODE3_H: u16 = 2160;
pub const DISPLAY_SERIAL_NUMBER: u32 = 1;
pub const DISPLAY_WIDTH_MM: u16 = 400;
pub const DISPLAY_HEIGHT_MM: u16 = 300;
/// Advertised refresh of every timing element. macOS paces CoreAnimation /
/// rAF to the display's advertised rate, so 60 here caps the guest at 60 fps
/// regardless of how fast VBL is signalled. 120 requests ProMotion-class
/// pacing; it must be matched by the VBL limiter and enough poll opportunities
/// (`REIMS_VGPU_PCI_HEARTBEAT_MS` = 4). The limiter now *derives* its interval
/// from this constant (`DISPLAY_VBL_MIN_INTERVAL_US`) rather than restating it,
/// because the two were allowed to drift apart: a hardcoded 8 ms delivered
/// 125 Hz against the 120 advertised here, and the guest paces to what is
/// delivered.
pub const DISPLAY_REFRESH_HZ: u32 = 120;
pub const DISPLAY_PRODUCT_NAME: &[u8] = b"QEMU display\0";
/// Archive: ~30s of ONLINE asserts at ~200ms (poll_ctr % 50, 4ms poll).
pub const DISPLAY_ONLINE_MAX_TRIES: u32 = 150;
pub const DISPLAY_ONLINE_POLL_DIVISOR: u32 = 50;

pub const CURSOR_GLYPH_BPP: u32 = 4;
/// Largest cursor sprite edge this device will accept a glyph for, in pixels.
///
/// # Derived from the consumer, which is the only thing that can refuse
///
/// A `SetCursorGlyph` past this bound is dropped whole — the guest's pointer
/// keeps whatever image it had, or none — so the number has to be the one that
/// actually cannot be served rather than a round guess. It was **256**, with no
/// stated basis, and the wire carries `width`/`height` as `u16`.
///
/// The consumer is QEMU's `cursor_alloc`, which both shims call and which is
/// the only thing downstream that refuses a size:
///
/// ```c
/// /* Modern physical hardware typically uses 512x512 sprites */
/// if (width > 512 || height > 512) {
///     return NULL;
/// }
/// ```
///
/// So 256 was half of what the host would have taken, and every cursor between
/// 257 and 512 was refused here for nothing. macOS reaches that band without
/// anything exotic: the accessibility pointer-size slider scales the sprite
/// several times over, and a Retina backing store doubles it again.
///
/// `the_cursor_bound_matches_the_sprite_size_qemu_will_allocate` reads the guard
/// out of `vendor/qemu/ui/cursor.c` and compares it to this, because the two
/// numbers live in different languages in different trees and nothing else
/// relates them. Being *above* QEMU's bound is the worse direction: `cursor_alloc`
/// answers `NULL`, and both shims drop the glyph on that path with no line at
/// all — a silent loss in C, which is exactly what this constant existing in
/// Rust is supposed to prevent.
pub const CURSOR_MAX_DIM: u32 = 512;
pub const CURSOR_GLYPH_PAYLOAD_LEN: usize = 0x2c;

pub const MMIO_U32: u32 = 4;
pub const MMIO_U64: u32 = 8;

/// Highest protocol version this host implements.
///
/// The guest writes a **fixed** 4 to `GFX_REG_VERSION` — not the highest
/// version it speaks, which is 62 — then reads the register back and switches on
/// *what it read* to fill its feature struct:
/// object tables, the child doorbell, EFI display, heaps, buffer-from-IOSurface,
/// the FIFO depth and the D32S8 stencil byte count. The ladder's top rung is 62;
/// **every value above 62 falls into the guest's default arm, which turns every
/// one of those features off**. So echoing a number this host does not implement
/// is not a no-op — it silently degrades the guest to a near-empty device.
///
/// 62 is not a fitted constant: it is the top rung of the guest's own switch and
/// the clamp Apple's host-side implementation applies to the same register.
///
/// Because the guest switches on the read-back rather than on what it wrote, the
/// effective rung is whatever the host leaves here, and a host may legally land
/// the guest **above** the 4 it asked for. Apple's own host does not: it clamps
/// down and never up, so a stock guest runs at rung 4 with `metalHeaps` and
/// `bufferFromIOSurface` off. Raising it is therefore in-mechanism but
/// out-of-contract, and it does not reach Metal's device families: the guest's
/// Metal plugin answers those from its own tables, not from anything this rung
/// unlocks.
pub const PROTOCOL_VERSION_MAX: u32 = 62;

/// What this host writes back for a guest-requested protocol version.
///
/// Clamping is the whole point. A guest newer than this host asks for a version
/// past the top rung, and an echo would hand that number straight back and land
/// the guest in its all-features-off default; clamping lands it on the newest
/// rung both sides implement.
#[inline]
pub fn negotiate_protocol_version(requested: u32) -> u32 {
    requested.min(PROTOCOL_VERSION_MAX)
}

/// Wire keys of the device-info reply whose value is a property of the GPU that
/// actually executes the guest's work, not of the protocol.
///
/// The guest driver stores the reply into a capability struct and hands it to
/// its Metal plugin, which answers `maxThreadsPerThreadgroup`,
/// `maxThreadgroupMemoryLength`, `supportsSampleCount:` and
/// `isDepth24Stencil8PixelFormatSupported` straight out of it. Every one of
/// those is an instruction to the guest about what it may build. Answering
/// higher than the host can execute does not degrade gracefully — the guest
/// sizes a threadgroup, declares threadgroup memory, or names a depth format,
/// and the host then refuses the pipeline that comes back.
pub const DEVICE_INFO_KEY_MAX_SAMPLE_COUNT: u32 = 1;
pub const DEVICE_INFO_KEY_D24_STENCIL8: u32 = 2;
pub const DEVICE_INFO_KEY_MAX_THREADS_W: u32 = 3;
pub const DEVICE_INFO_KEY_MAX_THREADS_H: u32 = 4;
pub const DEVICE_INFO_KEY_MAX_THREADS_D: u32 = 5;
pub const DEVICE_INFO_KEY_THREADGROUP_MEMORY: u32 = 6;
pub const DEVICE_INFO_KEY_NATIVE_FP16: u32 = 9;

/// Wire key 12 — whether the guest may build two-plane (biplanar YUV) textures.
///
/// Unlike its neighbours this is not a property of the host GPU. It is one of
/// the feature bools the guest's own driver selects from the negotiated
/// protocol version, echoed back over the wire; see
/// [`protocol_dual_plane_textures`].
pub const DEVICE_INFO_KEY_DUAL_PLANE_TEXTURES: u32 = 12;

/// Wire key 7 — whether the guest may read the framebuffer inside a fragment
/// shader (`isFramebufferReadSupported`).
///
/// Read as a **byte**, not a word, so any value whose low byte is zero reads as
/// false however large the `u32` is. Same for keys 8 and 9.
pub const DEVICE_INFO_KEY_FRAMEBUFFER_READ: u32 = 7;

/// Wire key 8 — `isRGB10A2GammaSupported`. A byte, like key 7.
pub const DEVICE_INFO_KEY_RGB10A2_GAMMA: u32 = 8;

/// Wire key 13 — `linearTextureAlignmentBytes`, the alignment the guest applies
/// when it builds a linear texture over a buffer.
///
/// **Arithmetic, not a boolean.** The guest computes with this number; absent,
/// it uses 16. Whatever this device publishes is what the guest's rows are laid
/// out on, so it is a promise about what this device can address rather than a
/// capability it can decline later.
///
/// Distinct from [`crate::contract::iosurface_pages::ROW_BYTES_ALIGN`], which is
/// an IOSurface row estimate for counting pages and is deliberately not a pitch.
/// Do not relate the two: one is Metal's linear-texture rule and the other is
/// IOSurface's allocator, and nothing says they are the same number.
pub const DEVICE_INFO_KEY_LINEAR_TEXTURE_ALIGN: u32 = 13;

/// Wire key 14 — one of the three terms of the guest's `supportsHeaps`.
///
/// The guest answers `supportsHeaps` with `key14 != 0 && key16 != 0 && f`, where
/// `f` is a feature bool selected by the negotiated protocol version. So neither
/// key alone turns heaps on and neither alone turns them off; what key 14
/// separately means, if anything, is **not established**.
pub const DEVICE_INFO_KEY_HEAPS: u32 = 14;

/// Wire key 15 — the granularity `heapBufferSizeAndAlignWithLength:options:`
/// rounds a heap buffer up to, and returns as its alignment.
///
/// Arithmetic again, and absent it is 256. This device publishes something
/// *smaller* than that fallback, which is the direction that asks more of the
/// host: a guest told 32 may place heap buffers 32 bytes apart.
pub const DEVICE_INFO_KEY_HEAP_BUFFER_GRANULARITY: u32 = 15;

/// Wire key 16 — whether a heap may back a texture.
///
/// Zero makes the guest's `[MTLHeap newTextureWithDescriptor:]` return nil
/// outright, so this is the key that decides whether heap textures exist at all.
/// Also the second term of `supportsHeaps`; see [`DEVICE_INFO_KEY_HEAPS`].
pub const DEVICE_INFO_KEY_HEAP_TEXTURES: u32 = 16;

/// Wire key 17 — `supportsBufferWithIOSurface`, gated on a protocol-version
/// feature bool as well.
pub const DEVICE_INFO_KEY_BUFFER_WITH_IOSURFACE: u32 = 17;

/// Wire key 10 — the **serializer feature version**, and the widest-reaching
/// number in this table after the protocol version itself.
///
/// The guest's Metal plugin passes this value straight into the command-stream
/// serializer's initialiser, which clamps it to 8 and turns it into five
/// independent feature bools by *rung*:
///
/// | rung | feature |
/// |---|---|
/// | `>= 3` | reflection serialization |
/// | `>= 5` | shared textures |
/// | `>= 6` | **OpenGL** |
/// | `>= 7` | IOSurface texture with rotation |
/// | `>= 8` | correct base vertex |
///
/// Absent, the plugin takes a two-argument initialiser and every rung is off.
///
/// So one number moves five gates at once, and lowering it to close one closes
/// every rung above it. That is why [`DEVICE_INFO_SERIALIZER_VERSION`] is a
/// named constant with this table beside it rather than a literal in
/// [`DEVICE_INFO_CAPS`].
///
/// # The OpenGL rung is on, and this device decodes none of what it unlocks
///
/// At `>= 6` the serializer's render encoder stops asserting on fifteen
/// GL-shaped selectors and emits them instead, `0x8a`..=`0x98`:
/// alpha-test reference, point size, clip plane, vertex/fragment sampler with
/// an LOD *bias*, viewport-transform enable, provoking-vertex mode, primitive
/// restart, two-sided fill mode, transform-feedback state, depth/stencil
/// "cleared", and the three colour/depth/stencil **resolve texture** setters.
/// `runtime::decode::render` has an arm for none of them; all fifteen are inside
/// the accepted opcode window, so each becomes `Kind::OtherAccepted` and is
/// reported once by the unimplemented-opcode path and executed by nothing.
///
/// Reachability is not the same on the two pathways, and the difference is in
/// the guest, not here. The PCI plugin's `MTLDevice` answers `supportsOpenGL`
/// with a hardcoded no whatever this key says, so nothing selects that device
/// for GL and the fifteen cannot arrive. The sysbus plugin's device forwards the
/// serializer's answer, so on that pathway they can. Neither has been driven
/// with a GL client here, and the unimplemented-opcode report is what would say
/// so — a driven x86/PCI boot under `web-content-probe -n 10 --churn 1` emits
/// none of it, which is the expected reading for the pathway where the guest's
/// own device declines GL, and not evidence about the other one.
pub const DEVICE_INFO_KEY_SERIALIZER_VERSION: u32 = 10;

/// The value served for [`DEVICE_INFO_KEY_SERIALIZER_VERSION`].
///
/// 8 is the serializer's own ceiling — it clamps anything higher — so this is
/// "every rung on", which is what the capture carried and what a real device
/// sends. It is *not* narrowed to the rungs this device implements, and the
/// reason is the coupling above: the OpenGL rung sits at 6, under the base-vertex
/// rung at 8, so declining OpenGL by lowering this number would also tell the
/// guest to apply its incorrect-base-vertex workaround and to stop using shared
/// textures. Which of those trades is right needs a driven boot on the sysbus
/// pathway that this checkout cannot take.
pub const DEVICE_INFO_SERIALIZER_VERSION: u32 = 8;

/// Wire key 11 — bitmask of the `MTLPrimitiveType` values the guest may draw.
///
/// Not a count and not a maximum: the guest's `supportsPrimitiveType:` tests
/// **bit `type`** of this value for any `type <= 8`, and answers `type < 5` when
/// the key is absent. See [`crate::contract::draw::EXECUTABLE_PRIMITIVE_TYPES`]
/// for what this device puts in it and why that is narrower than the capture.
///
/// Narrowing it changed nothing on the x86 pathway, as expected and no more
/// than that: a driven boot after the change passed the colour gate on all ten
/// captures with no `unknown_primitive_type` and no unimplemented-opcode line.
/// The PCI plugin's `MTLDevice` has no getter for this key at all, so that boot
/// exercises the guest *parsing* the narrower value and nothing reading it. The
/// pathway where it is read is the sysbus one, which this checkout cannot
/// drive.
pub const DEVICE_INFO_KEY_PRIMITIVE_TYPE_MASK: u32 = 11;

/// Whether protocol `version` enables two-plane textures.
///
/// The guest driver switches the negotiated version into a feature struct, and
/// this is the bool at offset 10 of it. The guest uses it **ungated**: a
/// two-plane pixel format makes its texture object allocate three sub-resources
/// and a doubled descriptor array instead of two. So answering a version that
/// does not have the feature that it does is not cosmetic — it changes the
/// shape of every biplanar video texture the guest builds.
///
/// The rung set is the guest's own switch, and it is **not monotonic**: the
/// feature appears at 31, is explicitly turned back off at 40, and returns at
/// 41. Both ends of the protocol encode that same gap, so it is a contract and
/// not an accident. Do not rewrite this as `version >= 31`.
#[inline]
pub fn protocol_dual_plane_textures(version: u32) -> bool {
    matches!(version, 31 | 41 | 42 | 43 | 60 | 61 | 62)
}

/// What the host GPU can actually do, for the keys above.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceInfoLimits {
    pub max_sample_count: u32,
    pub d24_stencil8: bool,
    pub max_threads_per_threadgroup: [u32; 3],
    pub max_threadgroup_memory_bytes: u32,
    pub native_fp16: bool,
}

/// The device-info reply for this host: GPU-dependent keys reduced to what
/// `limits` says the host can execute, and version-dependent keys answered from
/// the negotiated protocol `version`.
///
/// The GPU-dependent keys are reduction only. [`DEVICE_INFO_CAPS`] is the set of
/// answers this pathway has been exercised at, so it stays the ceiling: a host
/// that can do more is not a reason to promise the guest something no boot here
/// has ever run, and a host that can do less must not be promised anything at
/// all.
///
/// Key 12 is neither reduced nor served from the table. It is a protocol
/// feature bit, so the only correct answer is the one the negotiated version
/// selects — the table's fixed `1` is right for some versions and wrong for the
/// rest.
pub fn device_info_caps(limits: &DeviceInfoLimits, version: u32) -> Vec<(u32, u32)> {
    DEVICE_INFO_CAPS
        .iter()
        .map(|&(key, value)| {
            let host = match key {
                DEVICE_INFO_KEY_MAX_SAMPLE_COUNT => limits.max_sample_count,
                DEVICE_INFO_KEY_D24_STENCIL8 => u32::from(limits.d24_stencil8),
                DEVICE_INFO_KEY_MAX_THREADS_W => limits.max_threads_per_threadgroup[0],
                DEVICE_INFO_KEY_MAX_THREADS_H => limits.max_threads_per_threadgroup[1],
                DEVICE_INFO_KEY_MAX_THREADS_D => limits.max_threads_per_threadgroup[2],
                DEVICE_INFO_KEY_THREADGROUP_MEMORY => limits.max_threadgroup_memory_bytes,
                DEVICE_INFO_KEY_NATIVE_FP16 => u32::from(limits.native_fp16),
                DEVICE_INFO_KEY_DUAL_PLANE_TEXTURES => {
                    return (key, u32::from(protocol_dual_plane_textures(version)));
                }
                _ => return (key, value),
            };
            (key, value.min(host))
        })
        .collect()
}

/// Device-info capability table (key, value) — wire ABI from live bring-up.
///
/// Keys 1..=17 are the ones the macOS 13.7.8 guest driver parses; its reply
/// walker discards anything above 17 and stops at key 0. The higher keys are
/// kept because they came from a live capture and a newer guest may read them —
/// removing values whose meaning is not established would be trading one guess
/// for another. `reply_device_info` writes the zero terminator, which is what
/// stops the guest walking the rest of the page.
///
/// # Every key at or below 17 is named, and that is a rule rather than tidiness
///
/// The guest's walker is a jump table with one arm per key, and each arm stores
/// into a distinct field of a capability struct its Metal plugin then answers
/// from. So a value here is not a description of this device — it is an
/// **instruction to the guest** about what it may build, and one it cannot
/// re-ask about later. Key 11 is what that costs when it is left as a number:
/// it read `1023` from a capture for as long as this table existed, which
/// authorised four primitive types both backends refuse, and nothing said so
/// because the entry was a pair of integers.
///
/// A new entry at or below 17 therefore gets a `DEVICE_INFO_KEY_*` constant
/// whose doc says what the guest does with it. Above 17 the guest discards the
/// pair, so a bare number there promises nothing.
///
/// **Key 0 is not a key.** It terminates the walk and discards every remaining
/// pair, so an entry keyed 0 would silently truncate this table at its position.
/// `no_key_in_the_table_terminates_the_guest_walk` refuses one. A code span
/// rather than a link: rustdoc documents no `cfg(test)` item, so a link to a
/// test cannot resolve on any arm and reads as rot in the intra-doc pass.
///
/// The GPU-dependent subset is not served from here directly: see
/// [`device_info_caps`].
pub const DEVICE_INFO_CAPS: &[(u32, u32)] = &[
    (1, 8),
    (2, 1),
    (3, 1024),
    (4, 1024),
    (5, 64),
    (6, 32768),
    (DEVICE_INFO_KEY_FRAMEBUFFER_READ, 1),
    (DEVICE_INFO_KEY_RGB10A2_GAMMA, 1),
    (9, 1),
    (
        DEVICE_INFO_KEY_SERIALIZER_VERSION,
        DEVICE_INFO_SERIALIZER_VERSION,
    ),
    (
        DEVICE_INFO_KEY_PRIMITIVE_TYPE_MASK,
        crate::contract::draw::EXECUTABLE_PRIMITIVE_TYPES,
    ),
    (12, 1),
    (DEVICE_INFO_KEY_LINEAR_TEXTURE_ALIGN, 256),
    (DEVICE_INFO_KEY_HEAPS, 1),
    (DEVICE_INFO_KEY_HEAP_BUFFER_GRANULARITY, 32),
    (DEVICE_INFO_KEY_HEAP_TEXTURES, 1),
    (DEVICE_INFO_KEY_BUFFER_WITH_IOSURFACE, 1),
    (18, 131079),
    (19, 1),
    (21, 1),
    (23, 1),
    (24, 1),
    (25, 1),
    (26, 1),
    (27, 1),
    (28, 7),
    (29, 1024),
    (30, 32768),
    (31, 32768),
    (32, 16),
    (33, 4095),
    (34, 8),
    (35, 2048),
    (36, 16),
    (37, 1009),
    (40, 256),
    (41, 7),
    (42, 2),
    (44, 127),
];

#[inline]
pub fn child_reg_block_offset(channel_id: u32) -> Option<u64> {
    if !is_child_channel(channel_id) {
        return None;
    }
    Some(CHILD_REG_BLOCK_OFFSET + (channel_id as u64 - 1) * CHILD_REG_BLOCK_STRIDE)
}

#[inline]
pub fn main_ring_data_size(fifo_length: u32, fifo_start: u32) -> u32 {
    if fifo_length > fifo_start {
        fifo_length - fifo_start
    } else {
        fifo_length
    }
}

#[inline]
pub fn published_byte_count(head: u32, tail: u32, ring_size: u32) -> Option<u32> {
    let count = tail.wrapping_sub(head);
    if count <= ring_size {
        Some(count)
    } else {
        None
    }
}

#[inline]
pub fn stamp_slot_count(page_bytes: u64) -> u32 {
    (page_bytes / STAMP_SLOT_LEN as u64).min(u32::MAX as u64) as u32
}

#[inline]
pub fn stamp_slot_index(raw: u32) -> u32 {
    raw & STAMP_INDEX_MASK
}

/// Byte offset of stamp slot `index` within the stamp page.
///
/// `page_bytes` **must** be the guest stamp page size for this device
/// (`state.page_size()` — 4 KiB on x86, 16 KiB on arm64e). Never use a
/// hard-coded arm 16 KiB constant: that allowed indices 1024..4095 on x86
/// and wrote past the 4 KiB stamp page into adjacent guest RAM (wild write).
#[inline]
pub fn stamp_slot_offset(index: u32, page_bytes: u64) -> Option<u64> {
    let slots = stamp_slot_count(page_bytes);
    if index >= slots {
        None
    } else {
        Some((index as u64) * STAMP_SLOT_LEN as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A host that can do more than the table does not raise a single answer.
    ///
    /// The table is the set of values this pathway has been exercised at. The
    /// derivation exists to stop over-promising a weak host, not to start
    /// promising a strong one something no boot here has run.
    /// A version whose feature set matches the table, so only the host varies.
    ///
    /// 31 is the lowest rung with two-plane textures on, which is the table's
    /// own value for key 12; using it keeps this test about the host reduction
    /// and nothing else.
    const VERSION_WITH_DUAL_PLANE: u32 = 31;

    #[test]
    fn a_more_capable_host_never_raises_a_device_info_answer() {
        let generous = DeviceInfoLimits {
            max_sample_count: 64,
            d24_stencil8: true,
            max_threads_per_threadgroup: [4096, 4096, 4096],
            max_threadgroup_memory_bytes: 262_144,
            native_fp16: true,
        };
        assert_eq!(
            device_info_caps(&generous, VERSION_WITH_DUAL_PLANE),
            DEVICE_INFO_CAPS.to_vec()
        );
    }

    /// Two-plane textures follow the negotiated version, not a constant.
    ///
    /// The guest acts on this ungated: a two-plane format allocates three
    /// sub-resources rather than two when the bit is set. Serving a fixed 1
    /// tells a version-4 guest that a feature its own driver has switched off
    /// is available, and every biplanar video texture is then built to the
    /// wrong shape.
    #[test]
    fn key_12_is_the_negotiated_dual_plane_bit_not_a_constant() {
        let host = DeviceInfoLimits {
            max_sample_count: 64,
            d24_stencil8: true,
            max_threads_per_threadgroup: [4096, 4096, 4096],
            max_threadgroup_memory_bytes: 262_144,
            native_fp16: true,
        };
        let key12 = |version: u32| -> u32 {
            device_info_caps(&host, version)
                .into_iter()
                .find(|&(k, _)| k == DEVICE_INFO_KEY_DUAL_PLANE_TEXTURES)
                .expect("key 12 is served")
                .1
        };

        // What this guest actually negotiates. The table said 1; the guest's
        // own driver says 0.
        assert_eq!(key12(4), 0, "version 4 has no two-plane textures");
        assert_eq!(key12(31), 1);
        assert_eq!(key12(62), 1);
    }

    /// The rung set is the guest's switch, and that switch is not monotonic.
    ///
    /// Two-plane textures appear at 31, are turned back off at 40, and return
    /// at 41. Both ends of the protocol encode the same gap. A future
    /// simplification to `version >= 31` would answer 40 wrongly, so the gap is
    /// asserted directly.
    #[test]
    fn dual_plane_textures_is_off_at_version_40() {
        assert!(protocol_dual_plane_textures(31));
        assert!(!protocol_dual_plane_textures(40), "40 explicitly clears it");
        assert!(protocol_dual_plane_textures(41));

        // Every rung, so the whole contract is pinned rather than three points
        // of it. Versions that are not rungs land in the guest's default arm,
        // which has all features off.
        for version in 0..=PROTOCOL_VERSION_MAX {
            let want = matches!(version, 31 | 41 | 42 | 43 | 60 | 61 | 62);
            assert_eq!(
                protocol_dual_plane_textures(version),
                want,
                "version {version}"
            );
        }
    }

    /// A host at the Vulkan floor is told to the guest as the floor.
    ///
    /// Each of these is an instruction the guest acts on: it sizes threadgroups
    /// from the thread maxima, declares threadgroup memory up to the limit, and
    /// names a packed depth/stencil format if told one exists. Answering the
    /// table's fixed values on such a host is how a guest builds work the host
    /// then refuses.
    #[test]
    fn a_host_at_the_vulkan_floor_reduces_every_gpu_dependent_answer() {
        let floor = DeviceInfoLimits {
            max_sample_count: 1,
            d24_stencil8: false,
            max_threads_per_threadgroup: [128, 128, 64],
            max_threadgroup_memory_bytes: 16384,
            native_fp16: false,
        };
        let served: std::collections::BTreeMap<u32, u32> =
            device_info_caps(&floor, VERSION_WITH_DUAL_PLANE)
                .into_iter()
                .collect();
        assert_eq!(served[&DEVICE_INFO_KEY_MAX_SAMPLE_COUNT], 1);
        assert_eq!(served[&DEVICE_INFO_KEY_D24_STENCIL8], 0);
        assert_eq!(served[&DEVICE_INFO_KEY_MAX_THREADS_W], 128);
        assert_eq!(served[&DEVICE_INFO_KEY_MAX_THREADS_H], 128);
        assert_eq!(served[&DEVICE_INFO_KEY_MAX_THREADS_D], 64);
        assert_eq!(served[&DEVICE_INFO_KEY_THREADGROUP_MEMORY], 16384);
        assert_eq!(served[&DEVICE_INFO_KEY_NATIVE_FP16], 0);

        // Keys that describe the protocol rather than the GPU are untouched by
        // any host: reducing the serializer version or the primitive-type mask
        // would be answering a question the host GPU was never asked.
        let table: std::collections::BTreeMap<u32, u32> =
            DEVICE_INFO_CAPS.iter().copied().collect();
        for key in [7u32, 8, 10, 11, 12, 13, 14, 15, 16, 17] {
            assert_eq!(
                served[&key], table[&key],
                "key {key} must not depend on host"
            );
        }
    }

    /// No entry is keyed 0, and no key repeats.
    ///
    /// Zero is the walk's terminator, not a key: the guest's jump table sends it
    /// to a `ret`, so a `(0, x)` pair here would discard every pair after it and
    /// the loss would be invisible from this side — the reply is well formed and
    /// shorter than intended. A duplicate is the milder version of the same
    /// thing: the later arm overwrites the earlier field with no complaint from
    /// either end.
    ///
    /// Neither is expressible as a `const` assertion over a slice of tuples, so
    /// this is a test rather than a build failure.
    #[test]
    fn no_key_in_the_table_terminates_the_guest_walk() {
        let mut seen = std::collections::BTreeSet::new();
        for &(key, _) in DEVICE_INFO_CAPS {
            assert_ne!(key, 0, "key 0 ends the guest's walk; it cannot be an entry");
            assert!(seen.insert(key), "key {key} appears twice");
        }
    }

    #[test]
    fn child_register_blocks_cover_only_real_child_channels() {
        assert_eq!(child_reg_block_offset(0), None);
        assert_eq!(child_reg_block_offset(1), Some(CHILD_REG_BLOCK_OFFSET));
        assert_eq!(
            child_reg_block_offset((MAX_CHANNELS - 1) as u32),
            Some(CHILD_REG_BLOCK_OFFSET + (MAX_CHANNELS as u64 - 2) * CHILD_REG_BLOCK_STRIDE)
        );
        assert_eq!(child_reg_block_offset(MAX_CHANNELS as u32), None);
    }

    #[test]
    fn ring_publication_rejects_counts_larger_than_the_ring() {
        assert_eq!(main_ring_data_size(0x1000, 0x100), 0xf00);
        assert_eq!(main_ring_data_size(0x80, 0x100), 0x80);
        assert_eq!(published_byte_count(10, 25, 16), Some(15));
        assert_eq!(published_byte_count(10, 27, 16), None);
        assert_eq!(published_byte_count(u32::MAX - 3, 4, 8), Some(8));
    }

    #[test]
    fn stamp_bounds_follow_the_selected_guest_page_size() {
        let x86_slots = stamp_slot_count(PAGE_SIZE_X86);
        let arm_slots = stamp_slot_count(PAGE_SIZE_ARM64E);
        assert_eq!(x86_slots, (PAGE_SIZE_X86 / STAMP_SLOT_LEN as u64) as u32);
        assert_eq!(arm_slots, (PAGE_SIZE_ARM64E / STAMP_SLOT_LEN as u64) as u32);
        assert!(arm_slots > x86_slots);
        assert_eq!(
            stamp_slot_offset(x86_slots - 1, PAGE_SIZE_X86),
            Some(PAGE_SIZE_X86 - STAMP_SLOT_LEN as u64)
        );
        assert_eq!(stamp_slot_offset(x86_slots, PAGE_SIZE_X86), None);
        assert_eq!(stamp_slot_index(0xabcd_1234), 0x1234);
    }

    /// The C shims bound guest geometry against the same ceiling this module
    /// owns, and they get it from `REIMS_VGPU_MAX_SCANOUT_DIM` in the shared ABI
    /// header. Both shims carried a private `8192u` before that define existed,
    /// which is the shape `reims-vgpu-shim.h` warns about in as many words: a
    /// duplicated table is a table that can drift, and a drift between the two
    /// shims is a bug the guest sees on exactly one pathway. Here it would be a
    /// geometry that one attach accepts and the other drops.
    ///
    /// Nothing in the toolchain compares the two — Rust does not read the header
    /// and the shims do not read Rust — so this test is the only thing that
    /// does. It parses the header rather than restating the literal, because a
    /// second copy of the number in an assertion is the same defect one level
    /// up.
    #[test]
    fn the_abi_header_agrees_on_the_scanout_bound() {
        assert_eq!(
            crate::qemu::abi::header_define("REIMS_VGPU_MAX_SCANOUT_DIM"),
            MAX_SCANOUT_DIM,
            "the QEMU shims bound guest geometry against the header's value; \
             it has drifted from the Rust constant that owns the bound"
        );
    }

    /// The bound rejects both edges at both ends, and names which broke.
    ///
    /// `MAX_SCANOUT_DIM` itself is legal and one past it is not — the edge is a
    /// ceiling, not an exclusive limit, and seven of the eight copies this
    /// replaced spelled it `>` while nothing checked that they all did.
    #[test]
    fn the_scanout_extent_bound_holds_at_both_edges() {
        use ScanoutExtentFault as F;
        assert_eq!(scanout_extent_fault(1, 1), None);
        assert_eq!(
            scanout_extent_fault(MAX_SCANOUT_DIM, MAX_SCANOUT_DIM),
            None,
            "the largest edge the device accepts must be accepted"
        );
        assert_eq!(scanout_extent_fault(0, 16), Some(F::WidthZero));
        assert_eq!(scanout_extent_fault(16, 0), Some(F::HeightZero));
        assert_eq!(
            scanout_extent_fault(MAX_SCANOUT_DIM + 1, 16),
            Some(F::WidthAboveBound)
        );
        assert_eq!(
            scanout_extent_fault(16, MAX_SCANOUT_DIM + 1),
            Some(F::HeightAboveBound)
        );
        assert!(!scanout_extent_ok(0, 0));
        assert!(scanout_extent_ok(1920, 1080));
    }

    /// Channel 0 and channel `MAX_CHANNELS` are both refused, and the last real
    /// channel is not.
    ///
    /// The off-by-one at the top matters more than it looks: the id is used
    /// directly as both a `child_rings` index and a `1u32 <<` shift distance, so
    /// admitting `MAX_CHANNELS` would index one past a fixed-size array and
    /// shift a `u32` by 32.
    #[test]
    fn the_channel_id_bound_refuses_the_root_and_the_end() {
        assert!(
            !is_child_channel(0),
            "channel 0 is the root FIFO, not a child"
        );
        assert!(is_child_channel(1));
        assert!(is_child_channel(MAX_CHANNELS as u32 - 1));
        assert!(!is_child_channel(MAX_CHANNELS as u32));
        assert!(!is_child_channel(u32::MAX));
        assert_eq!(child_reg_block_offset(0), None);
        assert_eq!(child_reg_block_offset(MAX_CHANNELS as u32), None);
    }

    /// Zero is refused because it is the "no mapping" sentinel — and that is the
    /// *only* id this predicate refuses.
    ///
    /// The five `DeviceState` mutators and `mapper::capture_published_request`
    /// bounded the id from above only, so a guest MAP naming mapping 0 — which
    /// the mapper decodes straight out of the iosfc ring — would have created
    /// `mappings[0]`. Every `runtime::draw` reader treats `mapping_id == 0` as "this
    /// attachment is addressed by GVA", so that entry is state nothing goes on
    /// to consult.
    ///
    /// The upper half of this test used to assert a 4096 ceiling. Its storage is
    /// a `BTreeMap` keyed by the full `u32`, so the ceiling refused ids the map
    /// would have held; `u32::MAX` is asserted *accepted* now, in the same place
    /// it was once asserted refused, so a reinstated bound fails here.
    /// `CURSOR_MAX_DIM` must be the sprite size `cursor_alloc` will actually
    /// allocate.
    ///
    /// The two numbers live in different languages in different trees and
    /// nothing else relates them, which is how they came to differ by a factor
    /// of two: this was 256 with no stated basis while QEMU's `cursor_alloc`
    /// refuses only above 512, so every cursor sprite between 257 and 512 was
    /// dropped here for nothing. A `SetCursorGlyph` past the bound is dropped
    /// whole — the guest's pointer keeps whatever image it had.
    ///
    /// Equality rather than `<=`, and the direction matters both ways. Below
    /// QEMU's bound is the loss above. **Above** it is worse: `cursor_alloc`
    /// answers `NULL`, and both shims discard the glyph on that path with no
    /// line at all — a silent loss in C, which is what bounding this in Rust
    /// exists to prevent.
    ///
    /// Read out of the guard itself rather than pinned to a literal here, so a
    /// QEMU bump that moves the sprite limit fails this test instead of
    /// silently re-opening one of those two gaps. The parse asserts it found the
    /// guard before believing anything: a scan that matches nothing must fail,
    /// not pass.
    ///
    /// Beside the constant rather than in `tests/qemu_shims_agree.rs`, where the
    /// other shim-agreement scans live, because `model::regs` is a private
    /// module — an integration test cannot name `CURSOR_MAX_DIM`, and widening
    /// the crate's public surface to be testable is the wrong trade. Nothing
    /// here is backend-gated, so one arm executing it is enough.
    #[test]
    fn the_cursor_bound_matches_the_sprite_size_qemu_will_allocate() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../vendor/qemu/ui/cursor.c");
        let src = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("{} must be readable: {e}", path.display()));

        // The body of `cursor_alloc`, so a `> N` elsewhere in the file cannot
        // answer for the one that refuses an allocation.
        let body = src
            .split_once("QEMUCursor *cursor_alloc(")
            .map(|(_, tail)| tail)
            .unwrap_or_else(|| panic!("{} must define cursor_alloc", path.display()));
        let body = body
            .split_once("\n}")
            .map(|(head, _)| head)
            .expect("cursor_alloc must have a body");

        let bounds: Vec<u32> = body
            .match_indices("width > ")
            .chain(body.match_indices("height > "))
            .filter_map(|(at, pat)| {
                body[at + pat.len()..]
                    .split(|c: char| !c.is_ascii_digit())
                    .next()
                    .filter(|d| !d.is_empty())
                    .and_then(|d| d.parse().ok())
            })
            .collect();

        assert_eq!(
            bounds.len(),
            2,
            "expected cursor_alloc to bound width and height; found {bounds:?}. \
             The guard moved or was reworded — read it and re-derive \
             CURSOR_MAX_DIM rather than deleting this test."
        );
        assert!(
            bounds.iter().all(|&b| b == bounds[0]),
            "cursor_alloc bounds width and height differently ({bounds:?}); \
             CURSOR_MAX_DIM is one number and cannot express that"
        );
        assert_eq!(
            CURSOR_MAX_DIM, bounds[0],
            "CURSOR_MAX_DIM must be the sprite edge cursor_alloc will allocate: \
             below it drops guest cursors this host would have shown, above it \
             reaches a NULL both shims discard without a line"
        );
    }

    #[test]
    fn the_mapping_id_bound_refuses_only_the_no_mapping_sentinel() {
        assert!(
            !is_mapping_id(0),
            "0 names no mapping; it must not open an entry"
        );
        assert!(is_mapping_id(1));
        assert!(
            is_mapping_id(u32::MAX),
            "a mapping id is a full u32 on the wire and its storage is a map"
        );
    }

    /// The pre-boot console geometry, which both shims size a `DisplaySurface`
    /// with and which `scanout::paint_efi_console` refuses a paint against.
    ///
    /// A drift does not degrade — it makes the refusal fire on every early
    /// console paint, into a surface the shim itself just created at the other
    /// size, with the two numbers never appearing in the same file.
    #[test]
    fn the_abi_header_agrees_on_the_efi_boot_mode() {
        use crate::qemu::abi::header_define as define;
        assert_eq!(
            (
                define("REIMS_VGPU_EFI_BOOT_WIDTH"),
                define("REIMS_VGPU_EFI_BOOT_HEIGHT")
            ),
            (EFI_BOOT_WIDTH, EFI_BOOT_HEIGHT),
            "the shims paint the early console at the header's mode; it has \
             drifted from the mode this device advertises to the firmware"
        );
    }

    /// The gfx window the guest can address, against the bound Rust indexes.
    ///
    /// One direction is the one that loses work: a window wider than
    /// [`GFX_MMIO_SIZE`] is guest-addressable space whose accesses reach a
    /// register store that has no slot for them. Asserted as equality because
    /// the two have no reason to differ and an inequality check would let the
    /// harmless direction hide the accumulation.
    #[test]
    fn the_abi_header_agrees_on_the_gfx_window_size() {
        assert_eq!(
            u64::from(crate::qemu::abi::header_define("REIMS_VGPU_GFX_MMIO_SIZE")),
            GFX_MMIO_SIZE,
            "the shims size the gfx MMIO region from the header; it has drifted \
             from the bound Rust's sparse register store is indexed against"
        );
    }

    /// The header's four guest-page constants agree with Rust's page geometry.
    ///
    /// These cross the boundary and nothing in the toolchain compared them:
    /// `reims-vgpu-mmio.c` sizes its `mach_vm_remap` view, its alignment mask
    /// and its packed-contiguity stride from the header's arm64e page size,
    /// while every Rust reader derives the same number from
    /// `contract::gva::PAGE_SHIFT_ARM64E`. A drift is a view built at one
    /// stride and addressed at another, on one pathway, with no failure at the
    /// seam.
    ///
    /// Four rather than the two the shims read today: the shift pair is what a
    /// portable caller is supposed to take, so pinning only the sizes would
    /// leave the names most likely to be reached for next unchecked.
    #[test]
    fn the_abi_header_is_pinned_to_the_rust_page_geometry() {
        use crate::qemu::abi::header_define;
        assert_eq!(
            header_define("REIMS_VGPU_GUEST_PAGE_SHIFT_ARM64E"),
            PAGE_SHIFT_ARM64E,
            "the arm64e page shift has drifted from the contract"
        );
        assert_eq!(
            header_define("REIMS_VGPU_GUEST_PAGE_SHIFT_X86_64"),
            PAGE_SHIFT_X86,
            "the x86_64 page shift has drifted from the contract"
        );
        assert_eq!(
            u64::from(header_define("REIMS_VGPU_GUEST_PAGE_SIZE_ARM64E")),
            PAGE_SIZE_ARM64E,
            "the arm64e page size has drifted; the mmio shim strides its \
             mach_vm_remap view by this"
        );
        assert_eq!(
            u64::from(header_define("REIMS_VGPU_GUEST_PAGE_SIZE_X86_64")),
            PAGE_SIZE_X86,
            "the x86_64 page size has drifted from the contract"
        );
    }

    /// No two child-command names may carry the same opcode.
    ///
    /// The drain dispatches on these in one `match`, so a collision makes the
    /// later arm dead and hands one guest command to the other's handler —
    /// silently, because a duplicated *constant* in a pattern is not what
    /// `unreachable_patterns` is looking for. Every number here was assigned by
    /// reading Apple's command table, which is exactly the process that produces
    /// a transcription collision.
    ///
    /// Root and child are deliberately not crossed: they are separate spaces and
    /// `CmdDeleteTask` really is `0x20` in both.
    #[test]
    fn no_two_child_opcodes_share_a_number() {
        let table = [
            ("SETUP_SHARED_STATE", CHILD_OP_SETUP_SHARED_STATE),
            ("ONLINE_ACK", CHILD_OP_ONLINE_ACK),
            ("CURSOR_GLYPH", CHILD_OP_CURSOR_GLYPH),
            ("CURSOR_SHOW", CHILD_OP_CURSOR_SHOW),
            ("PRESENT_X86", CHILD_OP_PRESENT_X86),
            ("PRESENT_GAMMA_X86", CHILD_OP_PRESENT_GAMMA_X86),
            ("DISPLAY_SWAP", CHILD_OP_DISPLAY_SWAP),
            ("FLUSH_CHANNEL_EVENT", CHILD_OP_FLUSH_CHANNEL_EVENT),
            ("DELETE_TASK", CHILD_OP_DELETE_TASK),
            ("UNMAP_MEMORY", CHILD_OP_UNMAP_MEMORY),
            ("DELETE_OBJECT", CHILD_OP_DELETE_OBJECT),
            ("PRESENT_FRAME", CHILD_OP_PRESENT_FRAME),
            ("SET_OBJECT_LIST", CHILD_OP_SET_OBJECT_LIST),
            ("INVALIDATE_RESOURCES", CHILD_OP_INVALIDATE_RESOURCES),
            ("SYNCHRONIZE_RESOURCES", CHILD_OP_SYNCHRONIZE_RESOURCES),
            (
                "DELETE_IOSURFACE_BACKING2",
                CHILD_OP_DELETE_IOSURFACE_BACKING2,
            ),
            ("EXEC_INDIRECT2", CHILD_OP_EXEC_INDIRECT2),
            ("DEFINE_TASK2", CHILD_OP_DEFINE_TASK2),
            ("MAP_MEMORY2", CHILD_OP_MAP_MEMORY2),
            ("GET_COMPUTE_INFO", CHILD_OP_GET_COMPUTE_INFO),
            ("REPLACE_PHYSICAL", CHILD_OP_REPLACE_PHYSICAL),
            ("CONFIG_40", CHILD_OP_CONFIG_40),
        ];
        for (i, (name, op)) in table.iter().enumerate() {
            for (other, other_op) in &table[i + 1..] {
                assert_ne!(
                    op, other_op,
                    "CHILD_OP_{name} and CHILD_OP_{other} are both {op:#04x}; \
                     the drain's match would give both commands one handler"
                );
            }
        }
    }

    #[test]
    fn advertised_modes_fit_the_scanout_contract() {
        for (width, height) in [
            (DISPLAY_MODE_EFI_W, DISPLAY_MODE_EFI_H),
            (DISPLAY_MODE1_W, DISPLAY_MODE1_H),
            (DISPLAY_MODE2_W, DISPLAY_MODE2_H),
            (DISPLAY_MODE3_W, DISPLAY_MODE3_H),
        ] {
            assert!(u32::from(width) <= MAX_SCANOUT_DIM);
            assert!(u32::from(height) <= MAX_SCANOUT_DIM);
        }
        assert_eq!(DISPLAY_PRODUCT_NAME.last(), Some(&0));
    }
}
