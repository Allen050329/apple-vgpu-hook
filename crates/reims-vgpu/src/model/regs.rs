//! Register-window and FIFO wire constants from the live Reims VGPU contract.
//!
//! Sources: `apple-pv-gpu.h`, `reims_vgpu_fifo_format.h`.
//! Values are protocol constants — not content heuristics.

/// Gfx MMIO window size (16 KiB); bounds the sparse register store.
///
/// The iosfc window's size is not mirrored here. QEMU declares both regions
/// (`REIMS_VGPU_MMIO_{GFX,IOSFC}_MMIO_SIZE` in `reims-vgpu-mmio.c`) and Rust
/// only needs a bound for state it keeps per offset, which the iosfc rail does
/// not do — it decodes five named registers and ignores the rest. A second
/// unread copy of the iosfc size would be a source of truth nothing checks
/// against the one that actually sizes the `MemoryRegion`.
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
pub const EFI_BOOT_WIDTH: u32 = 1920;
/// Height of the advertised EFI mode. See [`EFI_BOOT_WIDTH`].
pub const EFI_BOOT_HEIGHT: u32 = 1080;
pub const EFI_MODE_WIDTH_SHIFT: u32 = 16;
pub const EFI_MODE_COUNT: u32 = 1;
pub const EFI_STRIDE_ALIGNMENT: u32 = 64;
pub const EFI_DISPLAY_PORT_COUNT: u32 = 1;
pub const EFI_BUILTIN_CONNECTED: u32 = 1;

pub const MAX_CHANNELS: usize = 32;
pub const MAX_TASKS: usize = 256;
pub const MAX_MAPPINGS: usize = 4096;
pub const MAX_SCANOUT_DIM: u32 = 8192;

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
pub use crate::contract::gva::{PAGE_SHIFT_ARM64E, PAGE_SHIFT_X86, pfn_to_gpa};
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
/// PVG CmdUnmapMemory.
/// PVG `CmdDeleteTask` (same opcode as root [`ROOT_OP_DELETE_TASK`]).
pub const CHILD_OP_DELETE_TASK: u16 = 0x20;
pub const CHILD_OP_UNMAP_MEMORY: u16 = 0x22;
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

pub const DISPLAY_SWAP_DISPLAY: usize = 0x00;
pub const DISPLAY_SWAP_MAPPING: usize = 0x08;
pub const DISPLAY_SWAP_MIN_LEN: usize = 12;

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
/// pacing; it must be matched by the VBL limiter (`DISPLAY_VBL_MIN_INTERVAL_MS`
/// = 8) and enough poll opportunities (`REIMS_VGPU_PCI_HEARTBEAT_MS` = 4).
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

pub const MAPPER_REQUEST_MAP: u32 = 1;
pub const MAPPER_REQUEST_UNMAP: u32 = 2;
pub const MAPPER_REQUEST_ENTRY_LEN: usize = 16;

/// Device-info capability table (key, value) — wire ABI from live bring-up.
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
    if channel_id == 0 || channel_id as usize >= MAX_CHANNELS {
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
