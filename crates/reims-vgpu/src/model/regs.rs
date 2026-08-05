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
/// [`the_abi_header_agrees_on_the_gfx_window_size`] failing.
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
/// [`the_abi_header_agrees_on_the_efi_boot_mode`] fails if they part.
pub const EFI_BOOT_WIDTH: u32 = 1920;
/// Height of the advertised EFI mode. See [`EFI_BOOT_WIDTH`].
pub const EFI_BOOT_HEIGHT: u32 = 1080;
pub const EFI_MODE_WIDTH_SHIFT: u32 = 16;
pub const EFI_MODE_COUNT: u32 = 1;
pub const EFI_STRIDE_ALIGNMENT: u32 = 64;
pub const EFI_DISPLAY_PORT_COUNT: u32 = 1;
pub const EFI_BUILTIN_CONNECTED: u32 = 1;

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

/// Task ids run from 0, unlike mapping ids — see [`is_mapping_id`]. Task 0 is
/// the task every type-4 surface this device has resolved lived in, and
/// `objects::type4_probe_order` probes it first by name.
pub const MAX_TASKS: usize = 256;
pub const MAX_MAPPINGS: usize = 4096;

/// Whether `mapping_id` names a mapping slot rather than "no mapping".
///
/// Zero is the device-wide sentinel for an unbound mapping — `metal_draw`
/// branches on `mapping_id == 0` in more than a dozen places to mean the
/// attachment is addressed by GVA instead — so a record naming 0 is not a
/// record naming slot 0, and creating `mappings[0]` for it produces state no
/// sentinel-aware reader will ever consult.
///
/// Four callers knew that and six did not: `runtime::texture` and the two
/// type-4 backing paths in `runtime::objects` refused zero, while the five
/// `DeviceState` mutators and `mapper::capture_published_request` bounded the
/// id from above only.
pub const fn is_mapping_id(mapping_id: u32) -> bool {
    mapping_id >= 1 && (mapping_id as usize) < MAX_MAPPINGS
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
/// [`the_abi_header_agrees_on_the_scanout_bound`] fails if they drift.
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
pub use crate::contract::gva::{pfn_to_gpa, PAGE_SHIFT_ARM64E, PAGE_SHIFT_X86};
pub const PAGE_SIZE_ARM64E: u64 = 1u64 << PAGE_SHIFT_ARM64E;
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
pub const CURSOR_MAX_DIM: u32 = 256;
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
/// out-of-contract, and it does not reach Metal's device families — see
/// `AGENTS.md`, "What The Guest Driver Puts Out Of Reach".
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
/// The GPU-dependent subset is not served from here directly: see
/// [`device_info_caps`].
pub const DEVICE_INFO_CAPS: &[(u32, u32)] = &[
    (1, 8),
    (2, 1),
    (3, 1024),
    (4, 1024),
    (5, 64),
    (6, 32768),
    (7, 1),
    (8, 1),
    (9, 1),
    (10, 8),
    (11, 1023),
    (12, 1),
    (13, 256),
    (14, 1),
    (15, 32),
    (16, 1),
    (17, 1),
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

    /// Zero is refused because it is the "no mapping" sentinel, not because it
    /// is out of range.
    ///
    /// The five `DeviceState` mutators and `mapper::capture_published_request`
    /// bounded the id from above only, so a guest MAP naming mapping 0 — which
    /// the mapper decodes straight out of the iosfc ring — would have created
    /// `mappings[0]`. Every `metal_draw` reader treats `mapping_id == 0` as "this
    /// attachment is addressed by GVA", so that entry is state nothing goes on
    /// to consult.
    #[test]
    fn the_mapping_id_bound_refuses_the_no_mapping_sentinel() {
        assert!(
            !is_mapping_id(0),
            "0 names no mapping; it must not open a slot"
        );
        assert!(is_mapping_id(1));
        assert!(is_mapping_id(MAX_MAPPINGS as u32 - 1));
        assert!(!is_mapping_id(MAX_MAPPINGS as u32));
        assert!(!is_mapping_id(u32::MAX));
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
