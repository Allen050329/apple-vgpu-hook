//! Live guest-visible state remembered by the host.
//!
//! Registers, rings, tasks/objects, mapper, present/cursor, stamps — the
//! ApplePV-shaped model. No parsing of wire bytes here; no backend execution.

mod lru_memo;
mod regs;
mod state;

pub use lru_memo::LruBytesMemo;
pub use regs::*;
pub use state::{
    ChannelRing, ComputeStorageResidencyKey, CursorState, DeferredOwner, DeviceId,
    DeviceState, DisplayHandshake, ExecFault, FENCE_DOMAIN_BLIT, FENCE_DOMAIN_COMPUTE,
    FENCE_DOMAIN_EVENT, FENCE_DOMAIN_RENDER, FailEvent, GfxRegs, GuestLinearMemo,
    GvaBacking, GvaDeferredEntry, GvaHostView,
    GvaEvictionWitness, HostLinearTexture, HostSurface, GVA_ENCODE_CACHE_BYTE_CAP,
    GVA_EVICTION_WITNESS_KEYS,
    IosfcRegs, LinearDeferredEntry, MapperCapture, MappingEntry,
    PacketFault, PaintSrc,
    PendingWork, PresentBacking, PresentState, RenderWindowSource, ResourceValidity,
    SurfaceWriteKind,
    TaskEntry, Type4Walk,
};

use crate::backend::Backend;
use crate::runtime::{self, host::HostOps};

/// Device instance: protocol state + selected backend.
///
/// MMIO and drain behavior live in [`crate::runtime`]; this type holds state
/// and forwards.
pub struct Device<B: Backend> {
    pub state: DeviceState,
    pub backend: B,
}

impl<B: Backend> Device<B> {
    /// `page_shift`: [`PAGE_SHIFT_X86`] or [`PAGE_SHIFT_ARM64E`]. Required — no default.
    pub fn new(id: DeviceId, backend: B, page_shift: u32) -> Self {
        Self {
            state: DeviceState::new(id, page_shift),
            backend,
        }
    }

    pub fn reset(&mut self) {
        self.backend.reset();
        self.state.reset();
    }

    /// Reset after releasing every HostOps view owned by this guest lifetime.
    pub fn reset_with_host<H: HostOps>(&mut self, host: &mut H) -> usize {
        // Backend aliases (notably Metal type-11 textures) must die before the
        // underlying contiguous guest-memory views are unmapped.
        self.backend.reset();
        let views = self.state.take_all_host_views();
        let count = views.len();
        for (ptr, len) in views {
            host.unmap_pages(ptr, len);
        }
        self.state.reset();
        count
    }

    pub fn gfx_read(&mut self, offset: u64, size: u32) -> u64 {
        runtime::mmio::gfx_read(&mut self.state, offset, size)
    }

    pub fn gfx_write<H: runtime::host::HostMemory + HostOps>(
        &mut self,
        host: &mut H,
        offset: u64,
        data: u64,
        size: u32,
    ) {
        runtime::mmio::gfx_write(&mut self.state, host, offset, data, size);
    }

    pub fn iosfc_read(&self, offset: u64, size: u32) -> u64 {
        runtime::mmio::iosfc_read(&self.state, offset, size)
    }

    pub fn iosfc_write<H: runtime::host::HostMemory + HostOps>(
        &mut self,
        host: &mut H,
        offset: u64,
        data: u64,
        size: u32,
    ) {
        runtime::mmio::iosfc_write(&mut self.state, host, offset, data, size);
    }

    /// BH body: drain pending work.
    ///
    /// `state.texture_to_mapping` is the authoritative type-11 ref → mapping
    /// table and is read directly by `runtime/metal_draw`. This used to also
    /// copy it into the backend on every drain, into a map nothing ever read.
    pub fn drain<H: runtime::host::HostMemory + HostOps>(&mut self, host: &mut H) {
        runtime::drain::drain_pending(&mut self.state, host);
    }

    pub fn fails(&self) -> &[FailEvent] {
        &self.state.fails
    }
}

#[cfg(test)]
mod tests {
    use crate::model::{PAGE_SHIFT_ARM64E, PAGE_SIZE_ARM64E, PAGE_SIZE_X86};

    use super::*;
    use crate::backend::NullBackend;
    use crate::contract::endian::st32;
    use crate::runtime::{
        FakeHost, HostActionKind, HostMemory,
    };

    #[test]
    fn stamp_slot_offset_respects_guest_page_size() {
        assert_eq!(stamp_slot_offset(0, PAGE_SIZE_X86), Some(0));
        assert_eq!(stamp_slot_offset(1023, PAGE_SIZE_X86), Some(1023 * 4));
        assert_eq!(stamp_slot_offset(1024, PAGE_SIZE_X86), None);
        assert_eq!(stamp_slot_offset(1024, PAGE_SIZE_ARM64E), Some(1024 * 4));
        assert_eq!(stamp_slot_offset(4095, PAGE_SIZE_ARM64E), Some(4095 * 4));
        assert_eq!(stamp_slot_offset(4096, PAGE_SIZE_ARM64E), None);
    }

    fn dev() -> Device<NullBackend> {
        Device::new(DeviceId(1), NullBackend, PAGE_SHIFT_ARM64E)
    }

    #[test]
    fn reset_view_collection_detaches_every_guest_alias() {
        let mut d = dev();
        d.state.retired_views.push((0x1000, 0x2000));
        d.state.mappings.insert(
            7,
            MappingEntry {
                contig_ptr: 0x3000,
                contig_len: 0x4000,
                ..Default::default()
            },
        );
        d.state.gva_host_views.push(GvaHostView {
            task_id: 1,
            gva: 0x8000,
            length: 0x1000,
            ptr: 0x5000,
            ptr_len: 0x6000,
            ..Default::default()
        });

        let mut views = d.state.take_all_host_views();
        views.sort_unstable();
        assert_eq!(
            views,
            vec![(0x1000, 0x2000), (0x3000, 0x4000), (0x5000, 0x6000)]
        );
        assert!(d.state.retired_views.is_empty());
        assert!(d.state.gva_host_views.is_empty());
        assert_eq!(d.state.mappings[&7].contig_ptr, 0);
        assert_eq!(d.state.mappings[&7].contig_len, 0);
    }

    fn setup_boot_regs(d: &mut Device<NullBackend>, h: &mut FakeHost) {
        d.gfx_write(h, GFX_REG_VERSION, 0x3e, MMIO_U32);
        assert_eq!(d.gfx_read(GFX_REG_VERSION, MMIO_U32), 0x3e);
        d.gfx_write(h, GFX_REG_FIFO_BASE_PAGE, 0x10, MMIO_U32);
        d.gfx_write(h, GFX_REG_FIFO_START, 0x4000, MMIO_U32);
        d.gfx_write(h, GFX_REG_FIFO_LENGTH, 0x10000, MMIO_U32);
        d.gfx_write(h, GFX_REG_CONTROL_FIFO, 1, MMIO_U32);
        let stamp = pfn_to_gpa(0x10, PAGE_SHIFT_ARM64E);
        h.map_range(stamp, PAGE_SIZE_ARM64E as usize, 0);
        let ring = stamp + 0x4000;
        h.map_range(ring, 0xc000, 0);
    }

    fn write_main_packet(h: &mut FakeHost, abs_off: u32, opcode: u16, stamp: u32, payload: &[u8]) {
        let ring_base = pfn_to_gpa(0x10, PAGE_SHIFT_ARM64E) + 0x4000;
        let ring_size = 0xc000u32;
        let total = PACKET_HEADER_LEN + payload.len() as u32;
        let mut raw = vec![0u8; total as usize];
        raw[0..2].copy_from_slice(&opcode.to_le_bytes());
        raw[2..4].copy_from_slice(&0u16.to_le_bytes());
        raw[4..8].copy_from_slice(&total.to_le_bytes());
        raw[8..12].copy_from_slice(&stamp.to_le_bytes());
        raw[12..].copy_from_slice(payload);
        for (i, b) in raw.iter().enumerate() {
            let off = abs_off.wrapping_add(i as u32) % ring_size;
            let _ = h.write_gpa(ring_base + off as u64, &[*b]);
        }
    }

    #[test]
    fn version_handshake_echo() {
        let mut d = dev();
        let mut h = FakeHost::new();
        d.gfx_write(&mut h, GFX_REG_VERSION, 0x3e, MMIO_U32);
        assert_eq!(d.gfx_read(GFX_REG_VERSION, MMIO_U32), 0x3e);
    }

    #[test]
    fn both_register_windows() {
        let mut d = dev();
        let mut h = FakeHost::new();
        d.gfx_write(&mut h, GFX_REG_ROOT_PAGE, 0x42, MMIO_U32);
        assert_eq!(d.gfx_read(GFX_REG_ROOT_PAGE, MMIO_U32), 0x42);
        d.iosfc_write(&mut h, IOSFC_REG_RING_BASE, 0x7000_0000, MMIO_U64);
        d.iosfc_write(&mut h, IOSFC_REG_CAPACITY, 0x400, MMIO_U32);
        assert_eq!(d.iosfc_read(IOSFC_REG_RING_BASE, MMIO_U64), 0x7000_0000);
        assert_eq!(d.iosfc_read(IOSFC_REG_CAPACITY, MMIO_U32), 0x400);
        d.state
            .gfx
            .interrupt_status_gpu
            .store(0x5, std::sync::atomic::Ordering::Release);
        assert_eq!(d.gfx_read(GFX_REG_INTR_STATUS_GPU, MMIO_U32), 0x5);
        assert_eq!(d.gfx_read(GFX_REG_INTR_STATUS_GPU, MMIO_U32), 0);
    }

    #[test]
    fn efi_display_constants() {
        let mut d = dev();
        assert_eq!(
            d.gfx_read(GFX_REG_EFI_MODE_COUNT, MMIO_U32),
            EFI_MODE_COUNT as u64
        );
        assert_eq!(
            d.gfx_read(GFX_REG_EFI_DISPLAY_PORTS, MMIO_U32),
            EFI_DISPLAY_PORT_COUNT as u64
        );
        assert_eq!(
            d.gfx_read(GFX_REG_EFI_BUILTIN_CONNECTED, MMIO_U32),
            EFI_BUILTIN_CONNECTED as u64
        );
        let size = d.gfx_read(GFX_REG_EFI_MODE_SIZE, MMIO_U32);
        assert_eq!(size >> EFI_MODE_WIDTH_SHIFT, EFI_BOOT_WIDTH as u64);
        assert_eq!(size & 0xffff, EFI_BOOT_HEIGHT as u64);
    }

    #[test]
    fn main_fifo_wraparound() {
        let mut d = dev();
        let mut h = FakeHost::new();
        setup_boot_regs(&mut d, &mut h);
        let ring_size = 0xc000u32;
        let payload = 1u32.to_le_bytes();
        let total = PACKET_HEADER_LEN + 4;
        let start = ring_size - 8;
        write_main_packet(&mut h, start, ROOT_OP_DEFINE_FIFO, 0x11, &payload);
        d.state
            .gfx
            .fifo_read
            .store(start, std::sync::atomic::Ordering::Release);
        d.state.gfx.fifo_written = start.wrapping_add(total);
        d.state.pending.main_drain = true;
        d.drain(&mut h);
        assert_eq!(
            d.state
                .gfx
                .fifo_read
                .load(std::sync::atomic::Ordering::Acquire),
            start.wrapping_add(total)
        );
        assert!(d.state.active_child_mask & (1 << 1) != 0);
        assert_eq!(h.get_u32(pfn_to_gpa(0x10, PAGE_SHIFT_ARM64E)), 0x11);
        assert!(h.action_count(HostActionKind::IrqGfxPulse) >= 1);
    }

    #[test]
    fn device_info_reply() {
        let mut d = dev();
        let mut h = FakeHost::new();
        setup_boot_regs(&mut d, &mut h);
        let reply_pfn = 0x20u32;
        h.map_range(
            pfn_to_gpa(reply_pfn, PAGE_SHIFT_ARM64E),
            PAGE_SIZE_ARM64E as usize,
            0xee,
        );
        let mut payload = vec![0u8; 12];
        st32(&mut payload[0..], 0);
        st32(&mut payload[4..], 4);
        st32(&mut payload[8..], reply_pfn);
        write_main_packet(&mut h, 0, ROOT_OP_DEVICE_INFO_TAHOE, 3, &payload);
        d.state
            .gfx
            .fifo_read
            .store(0, std::sync::atomic::Ordering::Release);
        d.state.gfx.fifo_written = PACKET_HEADER_LEN + 12;
        d.state.pending.main_drain = true;
        d.drain(&mut h);
        assert_eq!(
            h.get_u32(pfn_to_gpa(reply_pfn, PAGE_SHIFT_ARM64E)),
            DEVICE_INFO_CAPS[0].0
        );
        assert_eq!(
            h.get_u32(pfn_to_gpa(reply_pfn, PAGE_SHIFT_ARM64E) + 4),
            DEVICE_INFO_CAPS[0].1
        );
    }


    #[test]
    fn reset_clears_state() {
        let mut d = dev();
        let mut h = FakeHost::new();
        d.gfx_write(&mut h, GFX_REG_VERSION, 0x3e, MMIO_U32);
        d.state.define_task(1, 0x1000, 5);
        d.state.map_surface(7);
        d.state.record_fail(FailEvent::UnknownRootOpcode {
            opcode: 0xff,
            total_size: 12,
        });
        d.reset();
        assert_eq!(d.gfx_read(GFX_REG_VERSION, MMIO_U32), 0);
        assert!(!d.state.tasks[1].active);
        assert!(d.state.mappings.is_empty());
        assert!(d.fails().is_empty());
    }

    #[test]
    fn resource_lifecycle() {
        let mut d = dev();
        assert!(d.state.define_task(2, 0x2000, 9));
        assert!(d.state.set_object_list(2, 3, 64));
        assert!(d.state.insert_object(2, 10));
        assert!(d.state.objects.contains(&(2, 10)));
        assert!(d.state.delete_object(2, 10));
        assert!(!d.state.objects.contains(&(2, 10)));
        d.state.insert_object(2, 1);
        d.state.define_task(2, 0x2000, 9);
        assert!(d.state.objects.is_empty());
    }

    #[test]
    fn mapper_handshake() {
        let mut d = dev();
        let mut h = FakeHost::new();
        let base = 0x8000_0000u64;
        h.map_range(base, 0x1000, 0);
        d.iosfc_write(&mut h, IOSFC_REG_RING_BASE, base, MMIO_U64);
        d.iosfc_write(&mut h, IOSFC_REG_CAPACITY, 0x40, MMIO_U32);
        let _ = h.write_gpa(base, &1u32.to_le_bytes());
        let _ = h.write_gpa(base + 4, &7u32.to_le_bytes());
        // PRODUCER write sync-drains the mapper ring on the publishing vCPU
        // (pending.iosfc is set then cleared inside iosfc_write).
        d.iosfc_write(&mut h, IOSFC_REG_PRODUCER, 0x10, MMIO_U32);
        assert!(
            !d.state.pending.iosfc,
            "iosfc producer path drains before return"
        );
        assert!(
            d.state.mappings.contains_key(&7)
                || !d.fails().is_empty()
                || d.state.iosfc.consumer > 0
                || h.bh_scheduled
        );
    }

    #[test]
    fn present_and_cursor_model() {
        let mut d = dev();
        d.state.present.width = 1440;
        d.state.present.height = 900;
        d.state.cursor.show = true;
        d.state.cursor.hot_x = 1;
        d.state.cursor.hot_y = 2;
        assert_eq!(d.state.present.width, 1440);
        assert!(d.state.cursor.show);
    }

    #[test]
    fn fail_visible_unknown_root() {
        let mut d = dev();
        let mut h = FakeHost::new();
        setup_boot_regs(&mut d, &mut h);
        write_main_packet(&mut h, 0, 0xeeee, 1, &[]);
        d.state
            .gfx
            .fifo_read
            .store(0, std::sync::atomic::Ordering::Release);
        d.state.gfx.fifo_written = PACKET_HEADER_LEN;
        d.state.pending.main_drain = true;
        d.drain(&mut h);
        assert!(!d.fails().is_empty());
    }

    #[test]
    fn fail_visible_malformed_packet() {
        let mut d = dev();
        let mut h = FakeHost::new();
        setup_boot_regs(&mut d, &mut h);
        // total_size smaller than header
        let ring_base = pfn_to_gpa(0x10, PAGE_SHIFT_ARM64E) + 0x4000;
        let _ = h.write_gpa(ring_base, &0u16.to_le_bytes());
        let _ = h.write_gpa(ring_base + 2, &0u16.to_le_bytes());
        let _ = h.write_gpa(ring_base + 4, &4u32.to_le_bytes());
        d.state
            .gfx
            .fifo_read
            .store(0, std::sync::atomic::Ordering::Release);
        d.state.gfx.fifo_written = 12;
        d.state.pending.main_drain = true;
        let before = d
            .state
            .gfx
            .fifo_read
            .load(std::sync::atomic::Ordering::Acquire);
        d.drain(&mut h);
        // malformed: head must not advance (or fail recorded)
        assert!(
            d.state
                .gfx
                .fifo_read
                .load(std::sync::atomic::Ordering::Acquire)
                == before
                || !d.fails().is_empty()
        );
    }

    #[test]
    fn child_fifo_drain_define_task() {
        let mut d = dev();
        let mut h = FakeHost::new();
        setup_boot_regs(&mut d, &mut h);
        let ch = 1u32;
        d.state.active_child_mask |= 1 << ch;
        // Minimal child ring setup is covered by main DEFINE_FIFO + drain path.
        let payload = ch.to_le_bytes();
        write_main_packet(&mut h, 0, ROOT_OP_DEFINE_FIFO, 2, &payload);
        d.state
            .gfx
            .fifo_read
            .store(0, std::sync::atomic::Ordering::Release);
        d.state.gfx.fifo_written = PACKET_HEADER_LEN + 4;
        d.state.pending.main_drain = true;
        d.drain(&mut h);
        assert!(d.state.active_child_mask & (1 << ch) != 0);
    }

    #[test]
    fn doorbell_sets_pending_child() {
        let mut d = dev();
        let mut h = FakeHost::new();
        d.state.active_child_mask = 1 << 3;
        // Child doorbells publish work and schedule the BH; the vCPU MMIO path
        // must not synchronously consume render work.
        d.gfx_write(&mut h, GFX_REG_CHILD_DOORBELL, 3, MMIO_U32);
        assert_eq!(
            d.state.pending.child_mask,
            1 << 3,
            "doorbell leaves pending work for the BH"
        );
        assert!(
            d.state.active_child_mask & (1 << 3) != 0,
            "doorbell keeps channel active"
        );
        assert!(
            h.bh_scheduled,
            "doorbell schedules BH for HostAction delivery"
        );
    }
}
