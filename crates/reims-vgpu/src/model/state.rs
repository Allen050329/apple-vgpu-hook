//! Device-owned state: registers, rings, tasks, mapper, present, fail log.

use crate::model::{
    LruBytesMemo, DEFAULT_OBJECT_LIST_COUNT, DEFAULT_OBJECT_LIST_PFN, GFX_MMIO_SIZE, MAX_CHANNELS,
    MAX_MAPPINGS, MAX_TASKS,
};
use std::collections::{BTreeMap, VecDeque};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

/// Opaque device instance id (QEMU handle).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct DeviceId(pub u64);

/// Which check found a FIFO packet malformed.
///
/// One variant per distinct check, because the whole point of the vocabulary is
/// that `malformed packet` is not a diagnosis. These were thirteen hyphenated
/// `&'static str` literals passed by hand — informative to read, but not
/// greppable as slugs, not enumerable, and not countable, so nothing could tell
/// you whether the guest's ring had desynced or whether a header read had simply
/// failed.
///
/// Root-only and child-only checks are separate variants rather than one shared
/// slug plus a `channel=` field: they are genuinely different reads against
/// different registers, and collapsing them would put us back where we started.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PacketFault {
    /// Producer/consumer counters cannot describe a published byte range.
    DesyncedHeadTail,
    /// `total_size` outside `[header, ring]`, or short of its stamp list.
    BadSize,
    /// The decoder classified the ring position as desynced.
    Desynced,
    /// Guest read failed: root packet header.
    RootHeaderRead,
    /// Guest read failed: root packet snapshot.
    RootSnapRead,
    /// Guest write failed: root completion-stamp writeback.
    RootStampWriteback,
    /// Guest read failed: child packet header.
    ChildHeaderRead,
    /// Guest read failed: child ring register base.
    ChildRegsBaseRead,
    /// Guest read failed: child ring head register.
    ChildRegsHeadRead,
    /// Guest read failed: child ring stamp register.
    ChildRegsStampRead,
    /// Guest read failed: child packet snapshot.
    ChildSnapRead,
    /// Guest read failed: child ring tail.
    ChildTailRead,
    /// Guest write failed: child ring head writeback.
    ChildHeadWriteback,
}

impl PacketFault {
    pub fn slug(self) -> &'static str {
        match self {
            Self::DesyncedHeadTail => "packet_desynced_head_tail",
            Self::BadSize => "packet_bad_size",
            Self::Desynced => "packet_desynced",
            Self::RootHeaderRead => "packet_root_header_read",
            Self::RootSnapRead => "packet_root_snap_read",
            Self::RootStampWriteback => "packet_root_stamp_writeback",
            Self::ChildHeaderRead => "packet_child_header_read",
            Self::ChildRegsBaseRead => "packet_child_regs_base_read",
            Self::ChildRegsHeadRead => "packet_child_regs_head_read",
            Self::ChildRegsStampRead => "packet_child_regs_stamp_read",
            Self::ChildSnapRead => "packet_child_snap_read",
            Self::ChildTailRead => "packet_child_tail_read",
            Self::ChildHeadWriteback => "packet_child_head_writeback",
        }
    }
}

/// Which check refused to execute a decoded child-channel command.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecFault {
    /// A type-2 indirect exec packet shorter than its declared descriptor.
    Indirect2Short,
}

impl ExecFault {
    pub fn slug(self) -> &'static str {
        match self {
            Self::Indirect2Short => "exec_indirect2_short",
        }
    }
}

/// Fail-visible protocol event (unknown/malformed). Never invents semantics.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FailEvent {
    UnknownRootOpcode {
        opcode: u16,
        total_size: u32,
    },
    UnknownChildOpcode {
        channel: u32,
        opcode: u16,
        total_size: u32,
    },
    MalformedRootPacket {
        fault: PacketFault,
        head: u32,
    },
    MalformedChildPacket {
        channel: u32,
        fault: PacketFault,
        head: u32,
    },
    UnsupportedExec {
        channel: u32,
        fault: ExecFault,
    },
    BadMmioAccess {
        window: MmioWindow,
        offset: u64,
        size: u32,
    },
}

impl crate::observe::Decline for FailEvent {
    fn slug(&self) -> &'static str {
        match self {
            Self::UnknownRootOpcode { .. } => "unknown_root_opcode",
            Self::UnknownChildOpcode { .. } => "unknown_child_opcode",
            // The malformed variants delegate: the specific check *is* the
            // fault, so forwarding keeps one slug per check instead of two
            // coarse ones that the reader would then have to disambiguate by
            // hand from the fields.
            Self::MalformedRootPacket { fault, .. } | Self::MalformedChildPacket { fault, .. } => {
                fault.slug()
            }
            Self::UnsupportedExec { fault, .. } => fault.slug(),
            Self::BadMmioAccess { .. } => "bad_mmio_access",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::UnknownRootOpcode { opcode, total_size } => vec![
                ("opcode", format!("{opcode:#x}")),
                ("total_size", total_size.to_string()),
            ],
            Self::UnknownChildOpcode {
                channel,
                opcode,
                total_size,
            } => vec![
                ("ch", channel.to_string()),
                ("opcode", format!("{opcode:#x}")),
                ("total_size", total_size.to_string()),
            ],
            Self::MalformedRootPacket { head, .. } => vec![("head", head.to_string())],
            Self::MalformedChildPacket { channel, head, .. } => {
                vec![("ch", channel.to_string()), ("head", head.to_string())]
            }
            Self::UnsupportedExec { channel, .. } => vec![("ch", channel.to_string())],
            Self::BadMmioAccess {
                window,
                offset,
                size,
            } => vec![
                (
                    "window",
                    match window {
                        MmioWindow::Gfx => "gfx",
                        MmioWindow::Iosfc => "iosfc",
                    }
                    .to_string(),
                ),
                ("offset", format!("{offset:#x}")),
                ("size", size.to_string()),
            ],
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MmioWindow {
    Gfx,
    Iosfc,
}

/// Gfx named registers + sparse backing for unnamed offsets.
#[derive(Clone, Debug)]
pub struct GfxRegs {
    pub version: u32,
    pub control_fifo: u32,
    pub fifo_length: u32,
    pub fifo_written: u32,
    /// Main-FIFO consumer byte counter (0x100c), host-advanced. Lock-free
    /// `Arc<AtomicU32>` shared with the registry slot: the guest `writeFifo`
    /// producer spins on this register, so it must observe drain progress
    /// live while the drain worker owns the device lock.
    pub fifo_read: Arc<AtomicU32>,
    pub fifo_start: u32,
    pub root_page: u32,
    pub fifo_base_page: u32,
    /// Read-to-clear interrupt status (0x1014). Lock-free `Arc<AtomicU32>` so
    /// the guest ISR MMIO read observes live bits even while the drain worker
    /// owns the device lock (ack fast: a cached/stale mask loses signals).
    /// The `Arc` is shared with the device registry slot and survives reset.
    pub interrupt_status_disp: Arc<AtomicU32>,
    /// Read-to-clear stamp-signal status (0x1018). Same lock-free contract.
    pub interrupt_status_gpu: Arc<AtomicU32>,
    /// Fault interrupt status (0x102c), host-set, guest-read (not r2c). Same
    /// lock-free read rail (the guest ISR reads it right after 0x1018).
    pub interrupt_fault: Arc<AtomicU32>,
    pub efi_display: u32,
    pub efi_mode_select: u32,
    pub efi_fb_start: u64,
    pub efi_fb_length: u32,
    pub efi_fb_depth: u32,
    pub efi_fb_mode: u32,
    pub efi_fb_stride: u32,
    /// Backing for offsets without dedicated fields (word index).
    pub sparse: BTreeMap<u32, u32>,
}

impl Default for GfxRegs {
    fn default() -> Self {
        Self {
            version: 0,
            control_fifo: 0,
            fifo_length: 0,
            fifo_written: 0,
            fifo_read: Arc::new(AtomicU32::new(0)),
            fifo_start: 0,
            root_page: 0,
            fifo_base_page: 0,
            interrupt_status_disp: Arc::new(AtomicU32::new(0)),
            interrupt_status_gpu: Arc::new(AtomicU32::new(0)),
            interrupt_fault: Arc::new(AtomicU32::new(0)),
            efi_display: 0,
            efi_mode_select: 0,
            efi_fb_start: 0,
            efi_fb_length: 0,
            efi_fb_depth: 0,
            efi_fb_mode: 0,
            efi_fb_stride: 0,
            sparse: BTreeMap::new(),
        }
    }
}

impl GfxRegs {
    pub fn reset(&mut self) {
        // Preserve the shared interrupt-status atomics: the registry slot holds
        // clones for lock-free ISR reads; replacing them would detach that rail.
        let disp = Arc::clone(&self.interrupt_status_disp);
        let gpu = Arc::clone(&self.interrupt_status_gpu);
        let fault = Arc::clone(&self.interrupt_fault);
        let fifo_read = Arc::clone(&self.fifo_read);
        disp.store(0, Ordering::Release);
        gpu.store(0, Ordering::Release);
        fault.store(0, Ordering::Release);
        fifo_read.store(0, Ordering::Release);
        *self = Self {
            interrupt_status_disp: disp,
            interrupt_status_gpu: gpu,
            interrupt_fault: fault,
            fifo_read,
            ..Self::default()
        };
    }

    pub fn sparse_get(&self, offset: u64) -> u32 {
        let idx = (offset / 4) as u32;
        self.sparse.get(&idx).copied().unwrap_or(0)
    }

    pub fn sparse_set(&mut self, offset: u64, val: u32) {
        if offset < GFX_MMIO_SIZE {
            self.sparse.insert((offset / 4) as u32, val);
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct IosfcRegs {
    pub ring_base: u64,
    pub capacity: u32,
    pub desc_table: u64,
    pub producer: u32,
    pub consumer: u32,
}

impl IosfcRegs {
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// Per-channel child ring cache (page list decoded from base_pfn).
#[derive(Clone, Debug, Default)]
pub struct ChannelRing {
    pub valid: bool,
    pub base_pfn: u32,
    pub length: u32,
    pub page_gpas: Vec<u64>,
}

/// Ordered completion stamp slot (submission order per channel).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StampSlot {
    pub stamp_index: u32,
    pub stamp_value: u32,
    /// False while an async job owns this slot.
    pub ready: bool,
    /// Deferred job id (opaque); None = sync stamp.
    pub job_id: Option<u64>,
    /// Type-11 color/write target for async draw/compute jobs (`0` = none).
    /// Archive `ApplePVGPUDrawJob.mapping_id` / compute image mapping — used by
    /// `render_wait_surface` / product `surface_inflight` only.
    pub target_mapping: u32,
}

#[derive(Clone, Debug, Default)]
pub struct ChannelStamps {
    pub queue: VecDeque<StampSlot>,
}

impl ChannelStamps {
    pub fn reset(&mut self) {
        self.queue.clear();
    }

    /// Enqueue a stamp. Fires immediately only if ready and queue was empty of pending.
    pub fn push(&mut self, slot: StampSlot) {
        self.queue.push_back(slot);
    }

    /// Mark the first slot with `job_id` ready.
    pub fn mark_ready(&mut self, job_id: u64) -> bool {
        for s in self.queue.iter_mut() {
            if s.job_id == Some(job_id) {
                s.ready = true;
                return true;
            }
        }
        false
    }

    /// Pop all leading ready slots (in order).
    pub fn drain_ready(&mut self) -> Vec<StampSlot> {
        let mut out = Vec::new();
        while let Some(front) = self.queue.front() {
            if !front.ready {
                break;
            }
            out.push(self.queue.pop_front().unwrap());
        }
        out
    }
}

/// Task directory / object-list ownership.
#[derive(Clone, Debug, Default)]
pub struct TaskEntry {
    pub active: bool,
    pub length: u64,
    pub directory_pfn: u32,
    pub object_list_pfn: u32,
    pub object_list_count: u32,
}

/// Guest-declared MapMemory2 span (notify-only; no host PTE invent).
///
/// Used to fail-closed product GVA writes outside any recorded map when the
/// task has received at least one MapMemory2 (audit: cap write length against
/// known map range). Empty registry for a task ⇒ no cap (fixtures / pre-map).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TaskMapSpan {
    pub task_id: u32,
    pub gva: u64,
    pub length: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StateMutationDecline {
    DefineTaskIdRange { task_id: u32 },
    DeleteTaskIdRange { task_id: u32 },
    SetObjectListTaskIdRange { task_id: u32 },
    SetObjectListTaskInactive { task_id: u32 },
    InsertObjectTaskIdRange { task_id: u32, object_ref: u32 },
    InsertObjectTaskInactive { task_id: u32, object_ref: u32 },
    MapSurfaceIdRange { mapping_id: u32 },
    UnmapSurfaceIdRange { mapping_id: u32 },
    AttachMappingIdRange { mapping_id: u32 },
    AttachMappingInternalZero { mapping_id: u32 },
    MappingDeviceDescIdRange { mapping_id: u32 },
    MappingDeviceDescEmpty { mapping_id: u32 },
    MappingGeomIdRange { mapping_id: u32 },
    MappingGeomWidthZero { mapping_id: u32 },
    MappingGeomHeightZero { mapping_id: u32 },
    MappingGeomWidthRange { mapping_id: u32, width: u32 },
    MappingGeomHeightRange { mapping_id: u32, height: u32 },
}

impl crate::observe::Decline for StateMutationDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::DefineTaskIdRange { .. } => "model_define_task_id_range",
            Self::DeleteTaskIdRange { .. } => "model_delete_task_id_range",
            Self::SetObjectListTaskIdRange { .. } => "model_set_object_list_task_id_range",
            Self::SetObjectListTaskInactive { .. } => "model_set_object_list_task_inactive",
            Self::InsertObjectTaskIdRange { .. } => "model_insert_object_task_id_range",
            Self::InsertObjectTaskInactive { .. } => "model_insert_object_task_inactive",
            Self::MapSurfaceIdRange { .. } => "model_map_surface_id_range",
            Self::UnmapSurfaceIdRange { .. } => "model_unmap_surface_id_range",
            Self::AttachMappingIdRange { .. } => "model_attach_mapping_id_range",
            Self::AttachMappingInternalZero { .. } => "model_attach_mapping_internal_zero",
            Self::MappingDeviceDescIdRange { .. } => "model_mapping_device_desc_id_range",
            Self::MappingDeviceDescEmpty { .. } => "model_mapping_device_desc_empty",
            Self::MappingGeomIdRange { .. } => "model_mapping_geom_id_range",
            Self::MappingGeomWidthZero { .. } => "model_mapping_geom_width_zero",
            Self::MappingGeomHeightZero { .. } => "model_mapping_geom_height_zero",
            Self::MappingGeomWidthRange { .. } => "model_mapping_geom_width_range",
            Self::MappingGeomHeightRange { .. } => "model_mapping_geom_height_range",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        let mut fields = match self {
            Self::DefineTaskIdRange { task_id }
            | Self::DeleteTaskIdRange { task_id }
            | Self::SetObjectListTaskIdRange { task_id }
            | Self::SetObjectListTaskInactive { task_id } => {
                vec![("task", task_id.to_string())]
            }
            Self::InsertObjectTaskIdRange {
                task_id,
                object_ref,
            }
            | Self::InsertObjectTaskInactive {
                task_id,
                object_ref,
            } => vec![
                ("task", task_id.to_string()),
                ("ref", object_ref.to_string()),
            ],
            Self::MapSurfaceIdRange { mapping_id }
            | Self::UnmapSurfaceIdRange { mapping_id }
            | Self::AttachMappingIdRange { mapping_id }
            | Self::AttachMappingInternalZero { mapping_id }
            | Self::MappingDeviceDescIdRange { mapping_id }
            | Self::MappingDeviceDescEmpty { mapping_id }
            | Self::MappingGeomIdRange { mapping_id }
            | Self::MappingGeomWidthZero { mapping_id }
            | Self::MappingGeomHeightZero { mapping_id }
            | Self::MappingGeomWidthRange { mapping_id, .. }
            | Self::MappingGeomHeightRange { mapping_id, .. } => {
                vec![("mapping", mapping_id.to_string())]
            }
        };
        match self {
            Self::MappingGeomWidthRange { width, .. } => {
                fields.push(("width", width.to_string()));
            }
            Self::MappingGeomHeightRange { height, .. } => {
                fields.push(("height", height.to_string()));
            }
            _ => {}
        }
        fields
    }
}

impl StateMutationDecline {
    fn emit(self, discriminant: u64) {
        crate::observe::Emit::decline("model_state_mutation", &self).fail_once(discriminant);
    }
}

impl TaskMapSpan {
    /// True if half-open `[gva, gva+len)` is fully inside this span.
    #[inline]
    pub fn covers(&self, gva: u64, len: u64) -> bool {
        if self.length == 0 || len == 0 {
            return false;
        }
        let end = gva.saturating_add(len);
        let span_end = self.gva.saturating_add(self.length);
        gva >= self.gva && end <= span_end
    }
}

impl TaskEntry {
    pub fn define(length: u64, directory_pfn: u32) -> Self {
        Self {
            active: true,
            length,
            directory_pfn,
            object_list_pfn: DEFAULT_OBJECT_LIST_PFN,
            object_list_count: DEFAULT_OBJECT_LIST_COUNT,
        }
    }
}

/// Object-list entry (type + descriptor GVA) keyed by (task, ref).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ObjectEntry {
    pub object_type: u8,
    pub desc_gva: u64,
    pub desc_len: u32,
}

/// Directed mapper capture from guest xregs at iosfc producer write.
#[derive(Clone, Copy, Debug, Default)]
pub struct MapperCapture {
    /// Producer index that published this request (entry = producer - 1).
    pub producer: u32,
    pub mapper_device_kva: u64,
    pub request_type: u32,
    /// Guest kernel VA of MappingInternal.
    pub mapping_internal: u64,
}

/// IOSurface mapper registry entry keyed by mapping_id.
#[derive(Clone, Debug, Default)]
pub struct MappingEntry {
    pub mapped: bool,
    pub has_geom: bool,
    pub width: u32,
    pub height: u32,
    pub format: u16,
    pub content_generation: u32,
    /// Global store-order stamp: `DeviceState::store_seq` value at this
    /// mapping's most recent decoded guest write (`mark_mapping_written`).
    /// Unlike per-mapping `content_generation`, this is comparable ACROSS
    /// mappings — the compositor-output member with the highest stamp is the
    /// buffer the guest most recently finished (structural write history,
    /// never content).
    pub last_store_seq: u64,
    /// Bumped whenever the guest page list / map lifetime changes (MAP, UNMAP,
    /// ReplacePhysical, MappingInternal reattach, page-table refresh that
    /// changes PFNs). Used as [`TargetIdentity`] generation for resident
    /// import-present so a recycled mid never reuses a stale GPU target, and
    /// as a fail-closed check before zero-copy DMA into contig views.
    pub map_generation: u32,
    /// Guest page-table entries (valid bit + PFN); empty until resolved.
    pub page_entries: Vec<u32>,
    /// Page entries retired by a trailing `DeleteIOSurfaceBacking2` while the
    /// id may already carry a NEW incarnation (the delete trails the guest
    /// CPU-side release asynchronously; ids recycle within ~20 ms under
    /// scroll). Fingerprint for the next resolve: an identical re-resolved
    /// plan is the SAME incarnation (stale delete — keep generation, resident,
    /// deferred windows); a different plan is a genuine new incarnation
    /// (bump + drop condemned windows). Cleared by every explicit lifecycle
    /// event (fresh MAP, unmap, MappingInternal reattach, ReplacePhysical).
    pub condemned_entries: Option<Vec<u32>>,
    /// Guest KVA of MappingInternal (from capture or recover).
    pub mapping_internal: u64,
    pub page_table_kva: u64,
    /// Cached `sIOSurfaceDeviceDescriptor` (0x200) from MappingInternal+0x38.
    /// Used for biplanar plane selection by texture geometry; empty when unknown.
    pub device_desc: Vec<u8>,
    /// Contiguous host-VA view over `page_entries` (`HostOps::map_pages`,
    /// mach_vm_remap of guest RAM). 0 = not built. This is the unified-memory
    /// surface storage for the guest mapping: Metal
    /// render targets and samples are created directly on this view, so GPU
    /// Load/Store, guest CPU writes, and host page reads all see ONE copy —
    /// no writeback mirrors, no seed/capture ranking. Retired (never freed in
    /// place) whenever `page_entries` change; see `DeviceState::retired_views`.
    pub contig_ptr: usize,
    pub contig_len: usize,
    /// Task id that last owned this surface as a type-4 `OBJECT_TYPE_SURFACE`
    /// object (0 = no non-trivial hint; task 0 is always probed first anyway).
    /// `resolve_type4_surface_ex` probes this task right after task 0 so a
    /// per-bind present-path scan short-circuits instead of walking all 256
    /// task slots. Purely a search-order hint — a stale/wrong value only costs
    /// one extra probe before the full-table fallback re-finds the owner.
    pub owner_task_hint: u32,
}

/// Exact protocol-backed compute storage-image view eligible for residency.
///
/// `map_generation` separates recycled mapping lifetimes. The remaining fields
/// distinguish Metal texture views over one IOSurface; equal mapping ids alone
/// are not enough when formats or plane windows differ.
///
/// Three window kinds share this shape (`texture_ref` appended last so the
/// `(mapping_id, …)` ordering prefix — and every mapping-keyed range scan —
/// is unchanged):
/// - **Surface window** (`mapping_id != 0`): a type-11 IOSurface view;
///   `texture_ref == 0`.
/// - **Linear window** (`mapping_id == 0`): a type-2/3 raw task-GVA texture,
///   identity-matched to its `host_linear_textures` cache entry —
///   `map_generation` holds the task id, `surface_offset` the level-0 GVA,
///   `surface_bpr` the row stride, `span_end` `row_stride * height`, and
///   `texture_ref` the object-list ref. Mapping-keyed scans never see these
///   (real mapping ids are nonzero).
/// - **Heap texture** (`mapping_id == 0`, `surface_offset == 0`): a host-only
///   opcode-0x15 texture. `map_generation` holds the task id and `texture_ref`
///   the heap-texture object ref. It has no guest GVA to flush or restage.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ComputeStorageResidencyKey {
    pub mapping_id: u32,
    pub map_generation: u32,
    pub surface_offset: u64,
    pub surface_bpr: u32,
    pub span_end: u64,
    pub width: u32,
    pub height: u32,
    pub pixel_format: u16,
    pub texture_ref: u32,
}

impl ComputeStorageResidencyKey {
    /// Identity of a linear (type-2/3 raw task-GVA) texture window.
    #[allow(
        clippy::too_many_arguments,
        reason = "the key constructor names every wire-derived identity component"
    )]
    pub fn linear(
        task_id: u32,
        texture_ref: u32,
        gva: u64,
        row_stride: u32,
        span_end: u64,
        width: u32,
        height: u32,
        pixel_format: u16,
    ) -> Self {
        Self {
            mapping_id: 0,
            map_generation: task_id,
            surface_offset: gva,
            surface_bpr: row_stride,
            span_end,
            width,
            height,
            pixel_format,
            texture_ref,
        }
    }

    /// Identity of a host-only opcode-0x15 heap texture.
    pub fn heap(
        task_id: u32,
        texture_ref: u32,
        width: u32,
        height: u32,
        pixel_format: u16,
    ) -> Self {
        Self {
            mapping_id: 0,
            map_generation: task_id,
            surface_offset: 0,
            surface_bpr: 0,
            span_end: 0,
            width,
            height,
            pixel_format,
            texture_ref,
        }
    }

    /// True for a linear task-GVA window (see the struct doc).
    pub fn is_linear(&self) -> bool {
        self.mapping_id == 0 && self.surface_offset != 0
    }

    /// True for a host-only opcode-0x15 heap texture.
    pub fn is_heap(&self) -> bool {
        self.mapping_id == 0 && self.surface_offset == 0
    }
}

/// Deferred render-Store window: guest pages of this type-11 view are STALE —
/// the pinned engine resident render target (identity reconstructed as
/// `TargetIdentity::Surface { id: mapping_id, width, height, generation:
/// map_generation }`) is the authoritative content until a flush lands it.
///
/// Keyed like [`ComputeStorageResidencyKey`] so intersecting range scans share
/// one shape; ordered by `(mapping_id, surface_offset, span_end)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RenderDeferredKey {
    pub mapping_id: u32,
    pub surface_offset: u64,
    pub span_end: u64,
}

/// Everything a later flush needs to replay the deferred import-present Store.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RenderDeferredEntry {
    pub width: u32,
    pub height: u32,
    /// Mapping lifetime at defer time — flush drops on drift (recycled pages).
    pub map_generation: u64,
    /// Bounds mode the original Store would have used.
    pub full_quad_bounds: bool,
    /// The deferred draw targeted the unified compositor-output group
    /// resident (`TargetIdentity::OutputGroup`), not the per-mid Surface
    /// identity. The flush rebuilds exactly the identity that was pinned;
    /// `map_generation` still records the MEMBER's page lifetime so the
    /// recycled-pages drop guard keeps working for grouped windows.
    pub grouped: bool,
    /// Arm order for oldest-first flush when the render-deferred window cap is
    /// hit — mirrors [`GvaDeferredEntry::armed_seq`]. Bounds the pinned resident
    /// population so a compositing burst (YouTube page-load) cannot balloon the
    /// registry far past its slot cap.
    pub armed_seq: u64,
}

/// Everything a later flush needs to land a deferred **GVA render Store**
/// (type-2/3 color0 with `target_gva != 0`): the engine resident
/// `TargetIdentity::Gva { gva, width, height, generation: 0 }` holds the
/// authoritative pixels; guest pages + `host_gva_surfaces` are stale until a
/// flush lands them. One window per `gva` — a newer Store at the same GVA
/// supersedes (same geometry) or flushes (different geometry) the older one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GvaDeferredEntry {
    pub task_id: u32,
    pub texture_ref: u32,
    /// Producer object type captured at defer time (the task/object list may
    /// be gone by flush time) — `host_gva_surfaces` owner-gating input.
    pub producer_object_type: u8,
    pub width: u32,
    pub height: u32,
    /// Guest row stride the sync Store would have written with.
    pub row_stride: u32,
    pub format: u16,
    /// Arm order for oldest-first flush when the window cap is hit.
    pub armed_seq: u64,
    /// Defer-time physical page GPAs of the guest window — raw task-GVA reads
    /// aliasing these flush first (`storage_flush::flush_intersecting_task_gva`).
    pub pages: std::collections::HashSet<u64>,
}

impl GvaDeferredEntry {
    /// Guest byte span the flush writes: `row_stride * height`.
    pub fn span(&self) -> u64 {
        (self.row_stride as u64).saturating_mul(self.height as u64)
    }
}

/// HostOps view over a **task GVA range** (MapMemory2 / UnmapMemory lifecycle).
///
/// Distinct from [`MappingEntry::contig_ptr`] (iosfc `mapping_id` page list).
/// Created on demand via [`crate::runtime::gva_view::ensure_gva_view`]; torn
/// down on overlapping UnmapMemory / MapMemory2 / delete_task so we never keep
/// a host alias after the guest drops the GPU page-table mapping (Apple
/// `unmapMemory` analogue). Does **not** own discrete encode content
/// (`host_gva_surfaces`) — that cache is retained across Unmap (wallpaper class).
#[derive(Clone, Debug, Default)]
pub struct GvaHostView {
    /// Task slot the walk used when the view was built (resolved active id).
    pub task_id: u32,
    /// Guest VA base of the registered span (not necessarily page-aligned).
    pub gva: u64,
    /// Byte length of the registered GVA span.
    pub length: u64,
    /// Host pointer from [`crate::runtime::host::HostOps::map_pages`].
    pub ptr: usize,
    /// Host view length in bytes (`gpas.len() * page_size`).
    pub ptr_len: usize,
    /// Leaf GPA of the first/last page at build time — the sampled reuse
    /// verify re-translates these and retires the view on mismatch (stale
    /// cached-view read class). `0` = unverifiable (fixtures), skip.
    pub first_gpa: u64,
    pub last_gpa: u64,
}

/// Host-owned BGRA8 frame for a surface_id (Linux/Vulkan render-cache, §8.5).
#[derive(Clone, Debug, Default)]
pub struct HostSurface {
    pub width: u32,
    pub height: u32,
    /// Tight BGRA8, stride = width * 4.
    pub bgra: Vec<u8>,
    /// Monotonic host store generation (independent of guest content_generation).
    pub host_gen: u32,
    /// Producer identity for GVA-keyed type-2/3 encodes. Zero for surface/ref
    /// caches and for legacy/compute stores that did not record an owner.
    pub producer_task_id: u32,
    pub producer_texture_ref: u32,
    pub producer_object_type: u8,
}

/// Raw type-2/3 texture content retained by the discrete backend.
///
/// Unlike [`HostSurface`], bytes stay in the guest Metal pixel format and are
/// tightly row-packed. The key is `(task_id, texture_ref)`; descriptor fields
/// below reject stale hits after a ref is rebound. UnmapMemory drops the guest
/// page-table alias, not this GPU-private texture body.
#[derive(Clone, Debug, Default)]
pub struct HostLinearTexture {
    pub gva: u64,
    pub pixel_format: u16,
    pub width: u32,
    pub height: u32,
    pub row_stride: u64,
    pub bytes: Vec<u8>,
    pub host_gen: u32,
    /// Nonzero ⇒ the engine's pinned resident storage image at this generation
    /// is the authoritative content and `bytes` is empty (deferred linear
    /// writeback). Cleared by any bytes store.
    pub resident_gen: u32,
}

/// Geometry/source proven for a compositor-output mapping (see
/// [`PresentState::compositor_output_members`]).
#[derive(Clone, Copy, Debug, Default)]
pub struct CompositorOutputMember {
    pub width: u32,
    pub height: u32,
    /// Sampled type-11 source mapping, or `0` for the linear-input class.
    pub source: u32,
}

/// Which sub-path `paint_mapping` used to fill the present frame.
///
/// Measure-only provenance for the per-present `paint_us` cost: a deferred-Store
/// flush reuse and a cold fragmented guest-page read are indistinguishable from
/// `capture_present_frame` (both leave `from_host_cache == false`), yet the reuse
/// is a cheap memcpy of an in-hand readback while the cold read is the ~12 ms/
/// present fragmented multi-import. Collapsing both to `src=guest_pages` hid that
/// the fast path already covers the overwhelming majority of captures.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PaintSrc {
    /// No `paint_mapping` provenance recorded (host_cache path, or not yet run).
    #[default]
    None,
    /// Reused the byte-identical readback the deferred Store flush just scattered
    /// into the guest pages during this capture's own `flush_intersecting`.
    ReuseStore,
    /// Read straight out of the GPU resident (`read_resident_bgra`) with **no**
    /// guest-page scatter — the oracle frame source. The guest-page writeback
    /// stays deferred on the `render_deferred_flush` rail for a real guest read.
    Resident,
    /// Contiguous HostOps view read (packed mapping — one host span).
    GuestPagesContig,
    /// Multi-import fragmented guest-page read (the cold ~12 ms/present path).
    GuestPagesFragmented,
}

/// Occupancy stats of the retained `frame_bgra` snapshot, computed **once** by
/// [`crate::runtime::scanout::capture_present_frame`]'s fused scan.
///
/// The console paint ([`crate::runtime::scanout::copy_to_bgra8`]) runs under the
/// device lock on the QEMU display thread and previously re-scanned the same
/// 8 MiB `frame_bgra` twice (a byte-nz pass + an rgb-nz pass) purely to fill its
/// diagnostic `present_paint` lines — contending with the worker/VBL for the
/// lock. Since `frame_bgra` is frozen at capture time (only writer is
/// `capture_present_frame`), those stats are already known; the paint reuses
/// them, guarded by `mapping`/`generation` so a mismatch falls back to a scan.
#[derive(Clone, Copy, Debug, Default)]
pub struct FrameStats {
    /// The `frame_mapping` these stats were computed for.
    pub mapping: u32,
    /// The `frame_generation` these stats were computed for.
    pub generation: u32,
    /// Nonzero **bytes** across all four channels (`nonzero_stats`).
    pub byte_nz: usize,
    /// Max byte value across all four channels.
    pub byte_max: u8,
    /// Pixels with `max(B,G,R) > 0` (`bgra_rgb_stats`).
    pub rgb_nz: usize,
    /// Max of B/G/R across the frame.
    pub max_rgb: u8,
    /// First pixel BGRA.
    pub px0: [u8; 4],
}

/// Coarse tile grid for the per-mid damage-epoch map ([`PresentState::tile_gen`]).
/// Deliberately the SAME pitch as the `present_proxy` DAMAGE_GRID (32×18) so one
/// grid definition governs both damage measurement and cross-mid divergence — a
/// 1920×1080 target tiles at 60×60 px. Resolution-independent (proportional).
pub(crate) const TILE_GEN_GRID_W: usize = 32;
pub(crate) const TILE_GEN_GRID_H: usize = 18;
pub(crate) const TILE_GEN_TILES: usize = TILE_GEN_GRID_W * TILE_GEN_GRID_H;

/// Present / scanout model state.
#[derive(Clone, Debug, Default)]
pub struct PresentState {
    pub valid: bool,
    pub mapping_id: u32,
    pub width: u32,
    pub height: u32,
    pub bpr: u32,
    /// Content generation observed at last DisplaySwap enqueue.
    pub generation: u32,
    /// A host-owned presentation window is live (device_drain refreshes this
    /// from the window link each tranche). When false the QEMU console is the
    /// display: every present must enqueue a CPU `ScanoutUpdate` and the
    /// present-completion ack belongs to the console paint
    /// (`device_scanout_copy`), never the drain tail.
    pub window_active: bool,
    /// Mapping id of the last successful console paint (0 = never).
    /// Paired with `painted_generation` so dual-mid DisplaySwap cannot
    /// Unchanged-skip when both mids share the same generation counter.
    pub painted_mapping: u32,
    /// Content generation of the last successful paint (skip if matches).
    pub painted_generation: u32,
    pub present_mapping: u32,
    pub host_mapping: u32,
    pub frame_flush_seen: bool,
    /// Latest type-11 **Composite** writeback mid (logo/desktop content).
    /// Pre-boundary: sticky early feed for gfx_update when present_mapping is a
    /// ClearOnly flip buffer (dual-mid buffer-setup thrash class).
    /// Post-boundary: dual-mid *peer* tracker — x86 present often names ClearOnly
    /// mid 2/3 while Stores land on Composite mid 1/4/5; ClearOnly present
    /// captures this peer into +0x188 (not the black clear mid).
    pub early_front_mapping: u32,
    pub early_front_generation: u32,
    /// Latest successful full-geometry compositor edge `source -> output`.
    ///
    /// A non-self type-11 sample, or a same-geometry type-2/3 linear sample,
    /// establishes that `compositor_output_mapping` is downstream of a decoded
    /// texture input.  `compositor_output_source == 0` names the linear-input
    /// case (linear textures do not have a surface mapping id).  ClearOnly
    /// DisplaySwap uses this protocol/resource relationship to distinguish a
    /// completed compositor output from unrelated full-frame writers; pixel
    /// occupancy is never consulted.
    pub compositor_output_mapping: u32,
    pub compositor_output_source: u32,
    pub compositor_output_generation: u32,
    pub compositor_output_width: u32,
    pub compositor_output_height: u32,
    /// Mappings proven compositor outputs by a decoded edge (full-coverage
    /// linear sample or non-self type-11 sample), keyed by mapping id with the
    /// geometry/source recorded at proof time.  Steady-state WindowServer
    /// composite passes are damage passes (partial quads) that never re-prove
    /// full coverage, so membership is sticky: a later Composite-class
    /// writeback into a member at the proven geometry refreshes
    /// `compositor_output_mapping` to that member, following the guest's
    /// double-buffer alternation.  Removed on unmap.
    pub compositor_output_members: BTreeMap<u32, CompositorOutputMember>,
    /// Present/scanout evidence: mapping → latest geometry it was displayed
    /// at (a `capture_present_frame` action or a retained-frame re-show).
    /// OutputGroup unification requires this — the copy-swap contract is a
    /// *presented* double-buffer property.
    /// Sampled sub-surfaces (WebKit content tiles, scrollbars) publish full
    /// frames every paint but are never presented; without this gate any two
    /// same-geometry publishers unified and distinct surfaces chained one
    /// resident (the Safari-scroll black-band class).
    /// Removed on unmap with membership.
    /// Stamped with the mapping lifetime that presented it — see
    /// [`DeviceState::presented_geom_live`], which is the only way to read it.
    pub presented_geoms: BTreeMap<u32, PresentedGeom>,
    /// Display geometries proven to be a multi-buffer compositor swapchain:
    /// latched `true` the first time two distinct compositor-output members are
    /// *presented* at that geometry (the same condition `output_group_for`
    /// arms on). Sticky per boot — a resolution that was ever double-buffered
    /// stays one. WindowServer recycles swapchain buffers continuously (new mid
    /// ids, old ones unmapped, so `presented_geoms`/membership churn), which
    /// dropped the *concurrently* presented member count to one and collapsed
    /// the group — a fresh or recycled buffer then resolved to a per-mid
    /// resident that never held the accumulated full frame, so the guest's
    /// damage-only draw left everything outside the damaged rect black (the
    /// black-background / desktop-residue class). Latching the geometry keeps
    /// every presented member at a proven swapchain resolution unified through
    /// those recycles. Not gated on measured content; the per-mid
    /// member+`presented_at` checks in `output_group_for` still keep never
    /// presented publish-only tiles (WebKit content surfaces) out of the group.
    pub output_group_geoms: std::collections::BTreeSet<(u32, u32)>,
    /// Protocol-structural dense-frame tracking (measure-only, never gates a
    /// present decision): per compositor-output member, the value of
    /// [`Self::dense_frame_counter`] at the last full-frame (whole-`w`×`h`)
    /// Store **naming that mapping id** — the completeness proof in
    /// [`Self::note_compositor_member_published`], which is the only site that
    /// advances it. A presented member whose seq lags a same-geometry peer by a
    /// margin is the a/b inter-buffer retention gap
    /// ([`Self::dense_retention_gap`]). Cleared with membership on unmap.
    ///
    /// **What this is keyed on, and what that means it cannot see.** The advance
    /// is a function of the mapping id the Store named and nothing else; it does
    /// not consult [`DeviceState::output_group_for`] or any resident handle. So a
    /// full frame the guest sent for a member, whose draws were routed to a
    /// *different* resident than the one that member's present will read, still
    /// advances the seq. That routing failure is the black-desktop mechanism
    /// (§8.80 of the local KB), and the gate below is structurally blind to it —
    /// `present_identity_flip` is what catches it. It is also keyed per member
    /// while unified members share ONE resident, so a full frame stored through
    /// one member does not mark its siblings backed even though they hold the
    /// same pixels.
    pub dense_frame_seq: BTreeMap<u32, u64>,
    /// Per compositor-output member: the [`Self::dense_frame_seq`] value that
    /// member held the last time it was PRESENTED.
    ///
    /// A member whose seq is unchanged across two of its own presents received
    /// no full-frame Store naming it in between. That is the always-on
    /// `present_unbacked` gate — the loss itself, reported on the mid the guest
    /// named, rather than a rate at which we papered over it. Keyed per member
    /// (not globally) so healthy a/b alternation, where each buffer legitimately
    /// advances on its own turn, stays quiet. Cleared with membership on unmap.
    ///
    /// The "or an inter-buffer seed" half of this condition is gone: `62587b1`
    /// deleted the a/b peer front seed, because unified members share one
    /// resident and a seed between them is a copy onto itself. Nothing else
    /// advances [`Self::dense_frame_counter`].
    pub presented_dense_seq: BTreeMap<u32, u64>,
    /// Monotonic source for [`Self::dense_frame_seq`] (one bump per full-frame
    /// Store). Never reset except on device reset.
    pub dense_frame_counter: u64,
    /// Coarse per-mid per-tile damage-epoch grid.
    /// `TILE_GEN_GRID_W × TILE_GEN_GRID_H` tiles; each cell holds the
    /// [`Self::tile_epoch`] at which THIS mid last drew into that tile. Absent
    /// map entry / cell value 0 = never drawn. Cross-mid comparable because
    /// every cell is stamped from the single monotonic `tile_epoch` (never the
    /// per-mid, non-comparable `content_generation`). This is the SPATIAL
    /// refinement of the whole-frame [`Self::dense_frame_seq`]: a tile the guest
    /// erased in a peer has a higher epoch there than in a presented mid still
    /// showing the stale content (the a/b damage-coverage residue). Pruned on
    /// unmap in [`Self::forget_compositor_mapping`]. Feeds the `tile_divergence`
    /// proxy and the route-B cross-mid correction rects; the correction result is
    /// separately classified by the always-on `tile_composite` proxy.
    pub tile_gen: BTreeMap<u32, Box<[u64; TILE_GEN_TILES]>>,
    /// Monotonic frame clock for [`Self::tile_gen`], advanced ONCE PER PRESENT
    /// cycle (never per draw — a per-draw clock makes an actively-repainted
    /// video tile look perpetually divergent from a peer 1 draw behind, the
    /// thrash the reverted pixel-scan prototype hit). So two mids both
    /// repainting a tile every frame stamp it within ~1 epoch of each other
    /// (below [`crate::runtime::census::present_proxy::RETENTION_GAP_MARGIN`]), while a
    /// genuinely stale tile lags many epochs. Never reset except on device reset.
    pub tile_epoch: u64,
    /// Latest presentFrame retain (PGDisplay +0x188) — most recent DisplaySwap.
    /// Tight packed BGRA8, stride = `frame_width * 4`.
    pub frame_bgra: Vec<u8>,
    pub frame_mapping: u32,
    pub frame_width: u32,
    pub frame_height: u32,
    pub frame_generation: u32,
    pub frame_valid: bool,
    /// True only when DisplaySwap capture failed; first host paint retries.
    pub frame_encode_pending: bool,
    /// DisplaySwaps accepted since the last host paint of +0x188.
    ///
    /// apple-gfx `pending_frames` / PGDisplay `waitForPendingFrames` entry gate:
    /// when this is ≥ [`crate::runtime::drain::MAX_UNPAINTED_PRESENTS`], the
    /// child drain **holds** the next CmdDisplaySwap at channel head (no stamp)
    /// until paint clears the count. Accepted presents still stamp at retain.
    pub unpainted_presents: u32,
    /// Suppress repeated fail-log lines while the same present packet remains
    /// held at the pending-frames entry gate.
    pub backpressure_hold_active: bool,
    pub backpressure_hold_channel: u32,
    pub backpressure_hold_head: u32,
    /// Always-on diagnostic counter for distinct pending-frames hold episodes.
    pub backpressure_hold_count: u64,
    /// Sub-path the most recent `paint_mapping` used (measure-only provenance).
    pub last_paint_src: PaintSrc,
    /// Occupancy stats of the current `frame_bgra`, computed once at capture so
    /// the console paint does not re-scan the 8 MiB frame under the device lock.
    pub frame_stats: FrameStats,
    /// Recycled scratch for the present-capture frame buffer.
    ///
    /// `capture_present_frame` previously did `vec![0u8; need]` on **every**
    /// present — a fresh 8 MiB allocation that is zeroed and then fully
    /// overwritten, faulting in fresh anon pages each time (a large part of the
    /// per-present `paint_us`). Instead the capture takes this warm buffer,
    /// resizes (no realloc at steady geometry), fills it, and on success swaps
    /// the **old** `frame_bgra` back in here — so exactly two 8 MiB buffers
    /// cycle forever with no per-present malloc/zero/fault. On capture failure
    /// the buffer is returned here unchanged so the prior `frame_bgra` retain is
    /// untouched (keep-prior contract). Serialized with the console paint by the
    /// device lock; never read as content.
    pub capture_scratch: Vec<u8>,
    /// True when the previous present's window publish exported the frame as a
    /// zero-copy dmabuf (direct present, route B) rather than falling back to the
    /// CPU staging upload. Set by `publish_window_frame` each present (same drain
    /// worker, one present after the capture reads it — dmabuf state is stable
    /// across steady-state presents). When true, the display is carried by the
    /// GPU resident and does NOT consume the CPU `frame_bgra`, so
    /// `capture_present_frame` skips the expensive guest-page readback. The GPU
    /// stats oracle feeds the proxies without a framebuffer copy. Always false
    /// on non-host-window / non-import-capable builds, so those keep the
    /// per-present readback unchanged.
    pub dmabuf_active: bool,
    /// Always-on census: full (readback ran) vs light (dmabuf-carried, readback
    /// skipped) captures, so the readback-elision ratio is visible.
    pub full_captures: u64,
    pub light_captures: u64,
    /// The GPU stats reduction armed by the previous present, awaiting consume.
    ///
    /// The zero-copy oracle is asynchronous: a present arms a reduction and a
    /// later present collects it, so nothing ever blocks on the GPU. This
    /// carries the key needed to rebuild the exact identity that was armed.
    pub pending_stats: Option<PendingStats>,
    /// Monotonic arm counter for [`PendingStats::seq`]; never 0 once armed.
    pub stats_seq: u64,
}

/// One surface's present/scanout evidence: the geometry the guest named it at,
/// and the mapping incarnation that was current when it did.
///
/// The generation is what makes the evidence expire on its own. Without it the
/// entry outlives the surface it describes, and every site that recycles a
/// mapping has to remember to prune it — five did, and the one that pruned too
/// eagerly cost the desktop (see [`DeviceState::presented_geom_live`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PresentedGeom {
    pub width: u32,
    pub height: u32,
    pub map_generation: u32,
}

/// Why [`DeviceState::output_group_resolve`] refused to unify a surface into
/// the compositor output group. Produced by the check that refused, so a caller
/// can never supply the word itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputGroupMiss {
    /// The guest has not named this surface as plane 0 of a display transaction
    /// at this geometry. `presented` is what `presented_geoms` holds for the
    /// mapping instead: `None` when the entry was pruned (fresh MAP, unmap,
    /// condemned backing, new `MappingInternal`), `Some(other)` when the guest
    /// last presented this surface at a different geometry.
    NotPresentedHere { presented: Option<(u32, u32)> },
    /// Presented here, but the geometry is not a latched swapchain and no other
    /// surface is presented at it right now — nothing to unify with.
    NoPeer,
    /// There IS evidence at this geometry, but it was recorded by a prior
    /// incarnation of this mapping id: the guest recycled the id into a
    /// different surface, which must re-earn its qualification.
    ///
    /// Kept distinct from [`Self::NotPresentedHere`] because a wrong prune used
    /// to manufacture the latter out of a surface that was still the same
    /// incarnation, and the two want opposite responses.
    PriorIncarnation {
        presented: (u32, u32),
        entry_gen: u32,
        current_gen: u32,
    },
}

/// Key for an in-flight GPU stats reduction (the zero-copy present oracle).
///
/// Carries everything needed to rebuild the resident identity that was armed,
/// **without re-resolving it** — same rule as `render_deferred_identity`: a
/// compositor-membership change between arm and consume must not make us look
/// up a different image than the one the dispatch actually read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PendingStats {
    pub mapping_id: u32,
    pub width: u32,
    pub height: u32,
    /// Protocol content generation, for proxy attribution.
    pub generation: u32,
    /// Identity generation (0 for a unified `OutputGroup`).
    pub map_generation: u64,
    /// True when the armed identity was the unified compositor `OutputGroup`.
    pub grouped: bool,
    pub seq: u64,
}

/// Hardware cursor model.
#[derive(Clone, Debug, Default)]
pub struct CursorState {
    pub show: bool,
    pub x: u16,
    pub y: u16,
    pub width: u16,
    pub height: u16,
    pub hot_x: u16,
    pub hot_y: u16,
    /// QEMUCursor pixels as 0xAARRGGBB (guest BGRA reordered).
    pub pixels: Vec<u32>,
    /// True when `pixels` holds a complete glyph for the host console.
    pub glyph_ready: bool,
}

/// Display shared-state handshake (archive setupSharedState + online poll).
#[derive(Clone, Debug, Default)]
pub struct DisplayHandshake {
    pub shared_gpa: u64,
    pub display_index: u32,
    pub online_acked: bool,
    pub online_tries: u32,
    /// Cadence counter for ONLINE re-drive (archive display_poll_ctr).
    pub poll_ctr: u32,
    /// Samples already logged per observed display-transaction wire shape,
    /// keyed by `(opcode, payload_len, pipe_index, task_field_is_set)`.
    ///
    /// Backs the `display_txn_payload` measurement. A live x86 session showed the
    /// payload is trailer-only and its length never varies, so keying on length
    /// alone spent the whole budget inside the first 400ms of display activity
    /// and stayed silent afterwards. The remaining trailer words are what still
    /// carry news: `pipe_index` changes when a second display pipe appears, and
    /// the task field is zero through early bring-up, so its first non-zero value
    /// re-arms the probe exactly once at the transition into steady-state
    /// compositing.
    ///
    /// The plane-0 surface id is deliberately *not* part of the key: it is
    /// expected to change every frame, so keying on it would make the probe
    /// unbounded. Whether the task field is likewise per-frame is answered by
    /// comparing the samples within its bucket.
    pub txn_payload_samples: BTreeMap<(u16, usize, u32, bool), u32>,
}

/// Last **command-class** write to a surface mid (not pixel occupancy).
///
/// Used so a DisplaySwap of a mid that only received Clear (no composite Store)
/// does not overwrite a finished +0x188 retain — dual-mid clear flip of empty
/// display buffers while content lives on intermediate mids. This is protocol
/// history (Clear vs Store), not an rgb_nz / content-shape gate (AGENTS).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SurfaceWriteKind {
    #[default]
    Unknown,
    /// Only clear-only streams / software CLEAR Stores since last present.
    ClearOnly,
    /// At least one draw/composite Store (m2v encode, non-clear writeback).
    Composite,
}

/// Pending drain flags (MMIO path only sets bits; drain consumes).
#[derive(Clone, Debug, Default)]
pub struct PendingWork {
    pub main_drain: bool,
    pub child_mask: u32,
    pub iosfc: bool,
    /// A present queued a host scanout action. The ordered worker must return
    /// before consuming more guest work so QEMU can apply that action without
    /// blocking on the device lock. Cleared when the action is consumed.
    pub host_action_yield: bool,
}

/// Byte cap for the guest-CPU-produced content memos (`guest_linear_memo`,
/// `type5_view_memo`, `type11_memo`). A cap crossing evicts the coldest entries
/// down to a low-water mark — never a bulk clear — so the hot working set (and
/// its avoided re-decode/re-convert cost) survives.
pub const GUEST_LINEAR_MEMO_BYTE_CAP: usize = 128 << 20;

/// Byte cap for the authoritative-cache linear-sampled reuse memo
/// (`linear_sampled_memo`). Bounds host RAM by real bytes rather than a raw
/// entry count — 4K frames are ~33 MiB each, so a byte cap is the honest bound.
pub const LINEAR_SAMPLED_MEMO_BYTE_CAP: usize = 128 << 20;

/// See [`DeviceState::linear_sampled_memo`].
#[derive(Clone, Debug)]
pub struct LinearSampledMemo {
    pub gva: u64,
    pub host_gen: u32,
    pub width: u32,
    pub height: u32,
    pub rgba: std::sync::Arc<Vec<u8>>,
}

/// See [`DeviceState::guest_linear_memo`].
#[derive(Clone, Debug)]
pub struct GuestLinearMemo {
    /// Native guest rows (row-stride bytes as read, pre-conversion) at the last
    /// content change. Padding is included so a write anywhere in the span is
    /// observed by the byte-compare.
    pub native: Vec<u8>,
    /// Tight upload bytes of `native`: swizzled RGBA8, or — when `bgra8` — the
    /// guest's native BGRA8 order (uploaded into a BGRA8 image, no CPU swap).
    pub rgba: std::sync::Arc<Vec<u8>>,
    /// `rgba` holds native BGRA8 texels (upload as `Bgra8`) rather than RGBA8.
    pub bgra8: bool,
    /// Content generation: bumps only when the native bytes change.
    pub generation: u64,
}

/// Per-drain-tranche timing accumulator (diagnostic only — never gates behavior).
///
/// A drain tranche runs on QEMU's main-loop BH and holds the device lock for its
/// whole duration; a long tranche both delays completion stamps (guest present
/// stalls) and blocks QEMU's main loop (delayed host display refresh). The
/// opaque `drain_tranche_us` outlier is attributed here so a hitch can be read as
/// compile/convert/wait/readback-bound without fragile per-draw log correlation.
/// Accumulated per draw at the `log_linux_m2v_timing` site; `take`n every tranche
/// in `device_drain`, which emits the breakdown only on the >25 ms outlier line.
#[derive(Debug, Default, Clone, Copy)]
pub struct TrancheStats {
    pub draws: u64,
    pub draw_total_us: u64,
    pub load_us: u64,
    pub m2v_us: u64,
    pub setup_us: u64,
    pub setup_bufs_us: u64,
    pub setup_tex_us: u64,
    pub setup_seed_us: u64,
    pub setup_asm_us: u64,
    pub engine_us: u64,
    /// Engine per-draw cost split (subset of `engine_us`), so the dominant
    /// per-draw engine cost is attributable off-main-core without the per-draw
    /// `linux_m2v_timing` line (which needs `REIMS_VGPU_DRAW_LOG` and inflates timing).
    /// `engine_resource_us` = the whole resource-prep phase (samplers, staging /
    /// target resources, descriptor set) — measured ~65 µs/draw, the dominant
    /// engine cost; `engine_descriptor_us` = the descriptor subset
    /// (`vkAllocateDescriptorSets` + `vkUpdateDescriptorSets` + free) — measured
    /// ~0.1 µs/draw, NOT a lever; `engine_record_us` = command-buffer recording
    /// (barriers + staging copies + the draw); `engine_submit_us` = queue submit.
    pub engine_resource_us: u64,
    pub engine_descriptor_us: u64,
    pub engine_record_us: u64,
    pub engine_submit_us: u64,
    /// Resource-prep decomposition (subset of `engine_resource_us`), to locate
    /// the dominant sub-phase of the ~65 µs/draw resource cost off-main-core.
    /// `engine_target_us` = render-target image acquire/manage; `engine_sampled_us`
    /// = sampled-texture load/import; `engine_bufprep_us` = the
    /// vertex/index/storage/seed/sampler staging-prep loops. The remainder
    /// (`engine_resource_us` minus these three, `engine_descriptor_us`, and
    /// readback prep) is the resource-phase unattributed residual.
    pub engine_target_us: u64,
    pub engine_sampled_us: u64,
    pub engine_bufprep_us: u64,
    /// GPU-object churn proxy (scheduling-independent, unlike the `*_us` fields):
    /// `engine_creates` = `vkCreateImage`/framebuffer/view creates this tranche,
    /// `engine_allocs` = `vkAllocateMemory` calls. On a steady workload these
    /// stay near zero (targets/samplers are reused). When they track `draws`,
    /// GPU-backed resources are being recreated per draw — the render-target
    /// recreate-per-generation path (`registry_ensure`) that makes Safari video
    /// playback crawl (a full image alloc/free every frame). This is the always-on
    /// proxy for that bug class; it must not flood (one line per tranche).
    pub engine_creates: u64,
    pub engine_allocs: u64,
    /// Resident render-target images reused from the recycle pool this tranche
    /// (`target_free` hits). The counterpart to `engine_allocs` for the
    /// video-realloc fix: when a per-frame-generation target recurs, its image
    /// is popped from the recycle pool instead of reallocated, so under video
    /// `target_reuse` tracks `draws` while `engine_allocs` collapses to ~0. A
    /// high `engine_allocs` with a low `target_reuse` means the recycle pool is
    /// missing (geometry/format churn or the per-key cap).
    pub target_reuse: u64,
    /// Wall-clock spent inside `vkAllocateMemory` this tranche (the timed subset
    /// of `engine_allocs`; a slice of `engine_resource_us`). Always-on so a
    /// layout-reflow burst (a fullscreen transition recreates the whole layer
    /// tree at new geometry: `engine_creates`≈150 `engine_allocs`≈85
    /// `target_reuse`≈0, which no geometry-keyed recycle pool can absorb) shows
    /// whether the hitch is *allocation*-bound — i.e. whether a suballocating
    /// memory pool (one big `vkAllocateMemory`, many image binds) would help —
    /// without needing the `REIMS_VGPU_DRAW_LOG` per-draw `linux_m2v_timing` line. Like
    /// every `*_us` field it is SCHED_IDLE-contaminated under the agent harness;
    /// read it as a fraction of `drain_tranche_us`, and trust the count
    /// (`engine_allocs`) for the scheduling-independent burst size.
    pub engine_memory_alloc_us: u64,
    pub wait_us: u64,
    pub retire_wait_us: u64,
    pub readback_us: u64,
    pub readbacks: u64,
    pub reuploads: u64,
    pub reupload_bytes: u64,
    /// Zero-copy vertex/storage buffer binds (engine GPU-gathered the span
    /// from imported guest RAM instead of a CPU staging read).
    pub buf_zc: u64,
    /// Guest-run buffer binds CPU-snapshotted at record time (deferred-submit
    /// draw: the batched CB must not read volatile guest RAM at flush).
    pub buf_snap: u64,
    /// Compute dispatches (not render draws): `execute_dispatch_linux` total_us.
    pub compute_n: u64,
    pub compute_us: u64,
    /// Store/flush GPU readback+guest-writeback: import-present present captures
    /// (`import_present` map+dma+post) and deferred render/GVA flushes
    /// (`storage_flush::render_deferred_flush`). Neither is a render draw.
    pub store_n: u64,
    pub store_us: u64,
    /// Present/scanout capture lock hold: `capture_present_frame` (8 MiB alloc +
    /// host-cache/guest-page paint + the fused occupancy scan). Runs on the
    /// present drain, not on a render draw — previously invisible inside the
    /// opaque `other_us` residual, so a capture-bound present hitch could not be
    /// distinguished from FIFO-parse / mapper-walk time.
    pub capture_n: u64,
    pub capture_us: u64,
    /// Draw-batching ceiling census (measure-only, never gates behavior):
    /// `batch_same_target` = this draw's (identity, geometry, bgra) equals the
    /// previous draw's in the same packet; `batch_joinable` = additionally the
    /// load folds to LoadFromTarget with no seed, no readback, no MRT
    /// secondaries, and the draw does not sample its own target — i.e. it
    /// could have been recorded into the previous draw's render pass.
    pub batch_same_target: u64,
    pub batch_joinable: u64,
    /// ACTUAL deferred-submit batching outcomes (engine-reported, vs the
    /// census prediction above): `batch_opened` = draws that left their CB
    /// recording for successors; `batch_joined` = draws appended to an open
    /// CB, skipping slot claim + submit entirely.
    pub batch_opened: u64,
    pub batch_joined: u64,
    /// Guest-run memo effectiveness (draw-time zero-copy binds): a hit skips
    /// the per-bind task page-table walk entirely.
    pub run_memo_hit: u64,
    pub run_memo_miss: u64,
    /// Sampled memo-hit verifies (1-in-64) that found the fresh walk
    /// disagreeing with the memoized runs — each one is a draw that would
    /// have read stale guest pages. Zero on a healthy boot; nonzero means
    /// the invalidation contract has a hole (fail-logged as `rmemo_stale`).
    pub run_memo_stale: u64,
    /// Draw-time buffer-bind CPU cost split (subset of `setup_bufs_us`), so the
    /// dominant per-bind cost is attributable off-main-core without a per-draw
    /// log. Accumulated in **nanoseconds** (each bind is sub-microsecond; a
    /// per-call `as_micros()` truncates every sample to 0), emitted as µs:
    /// `buf_resolve` = object-list lookup + descriptor read/decode + backing
    /// resolve (FFI PT walks); `buf_read` = the byte materialization (host
    /// per-page FFI read; host-pointer memcpy on a future fast path).
    pub buf_resolve_ns: u64,
    pub buf_read_ns: u64,
    /// Wall time of the two stream buffer-load loops (vertex + fragment
    /// `load_buffer_content`) within `setup_bufs`, in nanoseconds. The
    /// remainder `setup_bufs_us - bufs_load` is the stage-in attribute build +
    /// fragment SPIR-V reloc + storage-binding classification.
    pub bufs_load_ns: u64,
    /// Draw-time zero-copy-attempt cost split (nanoseconds), for binds that ride
    /// `try_buffer_zero_copy_resolved`. `zc_flush` = the intersecting-deferred
    /// store flush (per-bind page walk when any deferred surface is live);
    /// `zc_import` = the `ensure_host_import` engine-lock + window resolve loop.
    /// `zc_fail_import` counts ZC attempts that fell back to the CPU read
    /// because a run was not coverable by a host import window.
    pub zc_flush_ns: u64,
    pub zc_import_ns: u64,
    pub zc_fail_import: u64,
    /// Deferred-window flushes that actually fired (a buffer bind aliased a
    /// live deferred-writeback window). Near-zero under pure compositing means
    /// the per-bind flush walk is detection overhead, not real coherence work.
    pub zc_flush_hits: u64,
    /// Per-bind flush walks skipped via the no-intersection memo (the win). A
    /// skip replaces a full per-page task-PT walk with a signature compare +
    /// set lookup.
    pub zc_flush_skip: u64,
    /// Sampled full walks (1-in-64 of memo skips) that found a real
    /// intersection the memo had marked non-intersecting — a missed deferred
    /// signature change. MUST stay 0 on a healthy boot; each one is
    /// fail-logged `zc_flush_stale` and self-heals (memo entry dropped, window
    /// flushed).
    pub zc_flush_stale: u64,
    /// zc_flush sub-path census (subset accounting for `zc_flush_ns`), so the
    /// dominant cost inside the per-bind flush is attributable off-main-core.
    /// `zc_flush_walk` = full per-page task-PT walks actually run (the expensive
    /// path — each per-page FFI translate dominates when the memo key
    /// `(task,gva,span)` churns and never hits); `zc_flush_walk_ns` = wall time
    /// inside those walks; `zc_flush_recheck` = cheap `deferred_pages_intersect`
    /// rechecks (a memo entry whose deferred signature changed, re-validated
    /// against cached pages with NO PT walk). A high `zc_flush_walk` relative to
    /// `zc_flush_skip` means the memo is churning (widen the key), not that the
    /// signature compare is slow.
    pub zc_flush_walk: u64,
    pub zc_flush_walk_ns: u64,
    pub zc_flush_recheck: u64,
    /// Wall time (ns) inside `deferred_flush_signature()` — the per-bind O(windows)
    /// hash over every live deferred window, computed once per flush call
    /// (~8×/draw). Prime suspect for the non-walk zc_flush residual when the memo
    /// churns. `zc_flush_isect_ns` = wall time inside the cheap
    /// `deferred_pages_intersect` rechecks (O(cached_pages × windows), no PT walk).
    pub zc_flush_sig_ns: u64,
    pub zc_flush_isect_ns: u64,
    /// Number of `flush_intersecting_task_gva` invocations made from the
    /// zero-copy-resolved buffer attempt (`try_buffer_zero_copy_resolved`), i.e.
    /// the exact population whose wall time is `zc_flush_ns`. Divides
    /// `zc_flush_ns` to a true per-call cost so the residual can be attributed
    /// (the sub-timers below span all flush call sites, a different population).
    pub zc_flush_calls: u64,
    /// Zero-copy flush calls whose wall time exceeded 100 µs — a preemption /
    /// off-CPU spike, not steady per-call work. If a few of these dominate
    /// `zc_flush_ns` the residual is scheduler noise (irreducible); if
    /// near-zero while the mean stays high the residual is a uniform per-call
    /// memory stall (reducible by cutting flush-call count / working set).
    pub zc_flush_slow: u64,
    /// Largest single zero-copy flush call wall time (ns) seen in the tranche —
    /// separates a bimodal spike distribution from a uniform-slow one.
    pub zc_flush_max_ns: u64,
    /// Wall time (ns) inside the exact-window fast path `flush_gva_exact` (a
    /// buffer bind whose base GVA is itself a live deferred render window → a
    /// synchronous engine `read_target` GPU readback). Isolates that stall from
    /// the cheap detection sub-paths so a rare-but-expensive readback cannot hide
    /// inside `zc_flush_ns`.
    pub zc_flush_exact_ns: u64,
}

/// One host-VA run of a memoized guest span (model mirror of the engine's
/// `GuestRun` so [`DeviceState`] stays backend-independent).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GuestRunSpan {
    pub host_ptr: usize,
    pub len: u64,
}

/// Memoized task-GVA→host-run resolution for draw-time zero-copy binds.
///
/// The per-draw page-table walk (`task_gva_guest_runs`) was the dominant
/// setup cost (~68 µs/draw under Safari scroll — walking ~60 PT leaves per
/// 260 KB buffer bind, ~5 binds/draw). The resolved runs are stable until the
/// guest changes the task page table, so they carry **exactly the
/// `gva_host_views` invalidation contract**: retired on UnmapMemory /
/// MapMemory2 overlap, task redefine, and task delete. Entries own no OS
/// resources (product-Linux `map_pages` is a RAMBlock alias), so retirement
/// is a plain drop.
#[derive(Debug, Clone)]
pub struct GuestRunMemoEntry {
    pub task_id: u32,
    pub gva: u64,
    pub length: u64,
    pub runs: Arc<Vec<GuestRunSpan>>,
}

/// One ExecIndirect2 packet's summary counters, folded into [`ExecAggStats`].
/// Mirror of the runtime `ExecResult` fields the old per-packet
/// `exec_indirect2` line carried (model stays independent of runtime types).
#[derive(Debug, Default, Clone, Copy)]
pub struct ExecPacketSample {
    pub streams: u64,
    pub saw_draw: bool,
    pub clears: u64,
    pub draws_ok: u64,
    pub draws_fail: u64,
    pub rt_resolves: u64,
    pub guest_stores: u64,
    pub icb_ok: u64,
    pub icb_fail: u64,
    pub compute_ctrl_fail: u64,
    pub compute_icb_fail: u64,
    pub load_us: u64,
    pub render_us: u64,
    pub blit_us: u64,
    pub compute_us: u64,
    pub event_us: u64,
    pub info_us: u64,
    pub finish_us: u64,
    pub total_us: u64,
}

/// Aggregated ExecIndirect2 packet telemetry (diagnostic only — never gates
/// behavior). The per-packet `exec_indirect2` line ran ~1k/s under Safari
/// scroll (the dominant always-on flood after the per-draw telemetry was
/// verbose-gated); healthy packets fold in here and one `exec_indirect2_agg`
/// summary per ~1 s window keeps rate/shares/tail visible. Packets carrying
/// failure counters still log per-packet on the always-on sink at the drain
/// site, so failure visibility is unchanged.
#[derive(Debug, Default)]
pub struct ExecAggStats {
    pub packets: u64,
    pub saw_draw: u64,
    pub streams: u64,
    pub clears: u64,
    pub draws_ok: u64,
    pub draws_fail: u64,
    pub rt_resolves: u64,
    pub guest_stores: u64,
    pub icb_ok: u64,
    pub icb_fail: u64,
    pub compute_ctrl_fail: u64,
    pub compute_icb_fail: u64,
    pub load_us: u64,
    pub render_us: u64,
    pub blit_us: u64,
    pub compute_us: u64,
    pub event_us: u64,
    pub info_us: u64,
    pub finish_us: u64,
    pub finish_us_max: u64,
    pub total_us: u64,
    pub total_us_max: u64,
    /// finish_us histogram: <1 ms, 1–4 ms, 4–16 ms, 16–64 ms, ≥64 ms.
    pub finish_hist: [u64; 5],
    window_start: Option<std::time::Instant>,
}

/// Aggregation window for the `exec_indirect2_agg` summary line.
const EXEC_AGG_WINDOW: std::time::Duration = std::time::Duration::from_secs(1);

impl ExecAggStats {
    pub fn note(&mut self, s: &ExecPacketSample) {
        if self.window_start.is_none() {
            self.window_start = Some(std::time::Instant::now());
        }
        self.packets = self.packets.saturating_add(1);
        self.saw_draw = self.saw_draw.saturating_add(s.saw_draw as u64);
        self.streams = self.streams.saturating_add(s.streams);
        self.clears = self.clears.saturating_add(s.clears);
        self.draws_ok = self.draws_ok.saturating_add(s.draws_ok);
        self.draws_fail = self.draws_fail.saturating_add(s.draws_fail);
        self.rt_resolves = self.rt_resolves.saturating_add(s.rt_resolves);
        self.guest_stores = self.guest_stores.saturating_add(s.guest_stores);
        self.icb_ok = self.icb_ok.saturating_add(s.icb_ok);
        self.icb_fail = self.icb_fail.saturating_add(s.icb_fail);
        self.compute_ctrl_fail = self.compute_ctrl_fail.saturating_add(s.compute_ctrl_fail);
        self.compute_icb_fail = self.compute_icb_fail.saturating_add(s.compute_icb_fail);
        self.load_us = self.load_us.saturating_add(s.load_us);
        self.render_us = self.render_us.saturating_add(s.render_us);
        self.blit_us = self.blit_us.saturating_add(s.blit_us);
        self.compute_us = self.compute_us.saturating_add(s.compute_us);
        self.event_us = self.event_us.saturating_add(s.event_us);
        self.info_us = self.info_us.saturating_add(s.info_us);
        self.finish_us = self.finish_us.saturating_add(s.finish_us);
        self.finish_us_max = self.finish_us_max.max(s.finish_us);
        self.total_us = self.total_us.saturating_add(s.total_us);
        self.total_us_max = self.total_us_max.max(s.total_us);
        let bucket = match s.finish_us {
            0..=999 => 0,
            1_000..=3_999 => 1,
            4_000..=15_999 => 2,
            16_000..=63_999 => 3,
            _ => 4,
        };
        self.finish_hist[bucket] = self.finish_hist[bucket].saturating_add(1);
    }

    /// When the window has run ≥1 s, return the formatted summary line and
    /// reset. Called after each note() — cadence is packet-driven, so an idle
    /// device emits nothing.
    pub fn flush_if_due(&mut self) -> Option<String> {
        let started = self.window_start?;
        let elapsed = started.elapsed();
        if elapsed < EXEC_AGG_WINDOW {
            return None;
        }
        let line = format!(
            "exec_indirect2_agg n={} window_ms={} saw_draw={} streams={} clears={} draws_ok={} draws_fail={} rt_resolves={} guest_stores={} icb_ok={} icb_fail={} compute_ctrl_fail={} compute_icb_fail={} load_us={} render_us={} blit_us={} compute_us={} event_us={} info_us={} finish_us={} finish_us_max={} total_us={} total_us_max={} finish_ms_hist={}/{}/{}/{}/{}",
            self.packets,
            elapsed.as_millis(),
            self.saw_draw,
            self.streams,
            self.clears,
            self.draws_ok,
            self.draws_fail,
            self.rt_resolves,
            self.guest_stores,
            self.icb_ok,
            self.icb_fail,
            self.compute_ctrl_fail,
            self.compute_icb_fail,
            self.load_us,
            self.render_us,
            self.blit_us,
            self.compute_us,
            self.event_us,
            self.info_us,
            self.finish_us,
            self.finish_us_max,
            self.total_us,
            self.total_us_max,
            self.finish_hist[0],
            self.finish_hist[1],
            self.finish_hist[2],
            self.finish_hist[3],
            self.finish_hist[4],
        );
        *self = Self::default();
        Some(line)
    }
}

impl TrancheStats {
    /// Fold a compute dispatch's lock hold in (not a render draw).
    pub fn note_compute(&mut self, us: u64) {
        self.compute_n = self.compute_n.saturating_add(1);
        self.compute_us = self.compute_us.saturating_add(us);
    }

    /// Fold a store/flush readback+writeback's lock hold in (not a render draw).
    pub fn note_store(&mut self, us: u64) {
        self.store_n = self.store_n.saturating_add(1);
        self.store_us = self.store_us.saturating_add(us);
    }

    /// Fold a present/scanout capture's lock hold in (not a render draw).
    pub fn note_capture(&mut self, us: u64) {
        self.capture_n = self.capture_n.saturating_add(1);
        self.capture_us = self.capture_us.saturating_add(us);
    }

    /// Fold one draw's timing delta in (saturating; all fields are cumulative).
    pub fn add(&mut self, d: TrancheStats) {
        self.draws = self.draws.saturating_add(d.draws);
        self.draw_total_us = self.draw_total_us.saturating_add(d.draw_total_us);
        self.load_us = self.load_us.saturating_add(d.load_us);
        self.m2v_us = self.m2v_us.saturating_add(d.m2v_us);
        self.setup_us = self.setup_us.saturating_add(d.setup_us);
        self.setup_bufs_us = self.setup_bufs_us.saturating_add(d.setup_bufs_us);
        self.setup_tex_us = self.setup_tex_us.saturating_add(d.setup_tex_us);
        self.setup_seed_us = self.setup_seed_us.saturating_add(d.setup_seed_us);
        self.setup_asm_us = self.setup_asm_us.saturating_add(d.setup_asm_us);
        self.engine_us = self.engine_us.saturating_add(d.engine_us);
        self.engine_resource_us = self.engine_resource_us.saturating_add(d.engine_resource_us);
        self.engine_descriptor_us = self
            .engine_descriptor_us
            .saturating_add(d.engine_descriptor_us);
        self.engine_record_us = self.engine_record_us.saturating_add(d.engine_record_us);
        self.engine_submit_us = self.engine_submit_us.saturating_add(d.engine_submit_us);
        self.engine_target_us = self.engine_target_us.saturating_add(d.engine_target_us);
        self.engine_sampled_us = self.engine_sampled_us.saturating_add(d.engine_sampled_us);
        self.engine_bufprep_us = self.engine_bufprep_us.saturating_add(d.engine_bufprep_us);
        self.engine_creates = self.engine_creates.saturating_add(d.engine_creates);
        self.engine_allocs = self.engine_allocs.saturating_add(d.engine_allocs);
        self.target_reuse = self.target_reuse.saturating_add(d.target_reuse);
        self.engine_memory_alloc_us = self
            .engine_memory_alloc_us
            .saturating_add(d.engine_memory_alloc_us);
        self.wait_us = self.wait_us.saturating_add(d.wait_us);
        self.retire_wait_us = self.retire_wait_us.saturating_add(d.retire_wait_us);
        self.readback_us = self.readback_us.saturating_add(d.readback_us);
        self.readbacks = self.readbacks.saturating_add(d.readbacks);
        self.reuploads = self.reuploads.saturating_add(d.reuploads);
        self.reupload_bytes = self.reupload_bytes.saturating_add(d.reupload_bytes);
        self.buf_zc = self.buf_zc.saturating_add(d.buf_zc);
        self.buf_snap = self.buf_snap.saturating_add(d.buf_snap);
        self.compute_n = self.compute_n.saturating_add(d.compute_n);
        self.compute_us = self.compute_us.saturating_add(d.compute_us);
        self.store_n = self.store_n.saturating_add(d.store_n);
        self.store_us = self.store_us.saturating_add(d.store_us);
        self.capture_n = self.capture_n.saturating_add(d.capture_n);
        self.capture_us = self.capture_us.saturating_add(d.capture_us);
        self.batch_same_target = self.batch_same_target.saturating_add(d.batch_same_target);
        self.batch_joinable = self.batch_joinable.saturating_add(d.batch_joinable);
        self.batch_opened = self.batch_opened.saturating_add(d.batch_opened);
        self.batch_joined = self.batch_joined.saturating_add(d.batch_joined);
        self.run_memo_hit = self.run_memo_hit.saturating_add(d.run_memo_hit);
        self.run_memo_miss = self.run_memo_miss.saturating_add(d.run_memo_miss);
        self.run_memo_stale = self.run_memo_stale.saturating_add(d.run_memo_stale);
        self.buf_resolve_ns = self.buf_resolve_ns.saturating_add(d.buf_resolve_ns);
        self.buf_read_ns = self.buf_read_ns.saturating_add(d.buf_read_ns);
        self.bufs_load_ns = self.bufs_load_ns.saturating_add(d.bufs_load_ns);
        self.zc_flush_ns = self.zc_flush_ns.saturating_add(d.zc_flush_ns);
        self.zc_import_ns = self.zc_import_ns.saturating_add(d.zc_import_ns);
        self.zc_fail_import = self.zc_fail_import.saturating_add(d.zc_fail_import);
        self.zc_flush_hits = self.zc_flush_hits.saturating_add(d.zc_flush_hits);
        self.zc_flush_skip = self.zc_flush_skip.saturating_add(d.zc_flush_skip);
        self.zc_flush_stale = self.zc_flush_stale.saturating_add(d.zc_flush_stale);
        self.zc_flush_walk = self.zc_flush_walk.saturating_add(d.zc_flush_walk);
        self.zc_flush_walk_ns = self.zc_flush_walk_ns.saturating_add(d.zc_flush_walk_ns);
        self.zc_flush_recheck = self.zc_flush_recheck.saturating_add(d.zc_flush_recheck);
        self.zc_flush_sig_ns = self.zc_flush_sig_ns.saturating_add(d.zc_flush_sig_ns);
        self.zc_flush_isect_ns = self.zc_flush_isect_ns.saturating_add(d.zc_flush_isect_ns);
        self.zc_flush_calls = self.zc_flush_calls.saturating_add(d.zc_flush_calls);
        self.zc_flush_slow = self.zc_flush_slow.saturating_add(d.zc_flush_slow);
        self.zc_flush_max_ns = self.zc_flush_max_ns.max(d.zc_flush_max_ns);
        self.zc_flush_exact_ns = self.zc_flush_exact_ns.saturating_add(d.zc_flush_exact_ns);
    }

    /// Reset to empty, returning the accumulated tranche total.
    pub fn take(&mut self) -> TrancheStats {
        std::mem::take(self)
    }
}

/// A map of deferred writeback windows whose page sets feed the union index
/// [`DeviceState::deferred_page_refs`].
///
/// Read-only outside this module: [`Deref`](std::ops::Deref) exposes the whole
/// `BTreeMap` read API, and there is deliberately **no** `DerefMut`, so the
/// inner map can only be mutated where the paired refcount update lives. That
/// is what makes the union index exact by construction — a site that armed or
/// disarmed a window without touching the index would not compile, so nothing
/// has to sample the index at runtime and repair it.
#[derive(Debug)]
pub struct DeferredWindows<K, V>(BTreeMap<K, V>);

impl<K, V> DeferredWindows<K, V> {
    fn new() -> Self {
        Self(BTreeMap::new())
    }
}

impl<K, V> std::ops::Deref for DeferredWindows<K, V> {
    type Target = BTreeMap<K, V>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// `for (k, v) in &windows`, which auto-deref does not reach on its own.
impl<'a, K, V> IntoIterator for &'a DeferredWindows<K, V> {
    type Item = (&'a K, &'a V);
    type IntoIter = std::collections::btree_map::Iter<'a, K, V>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

/// Full device model state (backend-independent).
#[derive(Debug)]
pub struct DeviceState {
    pub id: DeviceId,
    /// Guest page shift for PFN↔GPA wire math (12 = x86, 14 = arm64e).
    pub page_shift: u32,
    pub gfx: GfxRegs,
    pub iosfc: IosfcRegs,
    pub is_tahoe: bool,
    pub active_child_mask: u32,
    /// Child channels whose head `EXEC_INDIRECT2` packet is held while an
    /// immutable AIR translation is still loading. The packet head and stamp
    /// remain untouched until retry, so this is scheduler state rather than a
    /// submitted async GPU job.
    pub translation_deferred_mask: u32,
    /// Root/child FIFO timelines held behind a cold-translation EXEC. Bit 0 is
    /// the root FIFO; child channel N uses bit N. This is diagnostic scheduler
    /// ownership, not a guest-visible protocol mask.
    pub translation_order_hold_mask: u32,
    /// Distinct cross-FIFO hold episodes (retries of one episode do not grow it).
    pub translation_order_holds: u64,
    /// Display transactions held while another channel remained blocked on
    /// translation after the transaction's rescue drains. This counts hold
    /// episodes, not poll retries of the same packet.
    pub present_translation_holds: u64,
    /// Display channels whose FIFO head is already held for
    /// `translation_deferred_mask`. Suppresses fail-log flooding while the
    /// same head is retried and is cleared with channel lifecycle state.
    pub present_translation_hold_mask: u32,
    pub pending: PendingWork,
    pub child_rings: [ChannelRing; MAX_CHANNELS],
    pub child_stamps: [ChannelStamps; MAX_CHANNELS],
    pub tasks: [TaskEntry; MAX_TASKS],
    /// Count of MapMemory2/UnmapMemory packets (measure census).
    pub map_family_events: u64,
    /// Live MapMemory2 spans per task (wire notify ranges). Cleared on Unmap /
    /// delete_task / redefine. Product GVA writes check coverage when non-empty.
    pub task_map_spans: Vec<TaskMapSpan>,
    /// Sparse object table: (task_id, ref) -> entry.
    pub objects: BTreeMap<(u32, u32), ObjectEntry>,
    /// Type-11 texture object ref → mapping_id: (task_id, ref) -> mapping_id.
    pub texture_to_mapping: BTreeMap<(u32, u32), u32>,
    pub mappings: BTreeMap<u32, MappingEntry>,
    /// Host render-cache keyed by surface_id / mapping_id (Linux/Vulkan rail).
    /// See [`crate::runtime::surface_cache`] and kb tahoe-x86-host-reims_vgpu §8.5.
    /// **Surface_id namespace only** — never texture_ref (object list ids collide).
    pub host_surfaces: BTreeMap<u32, HostSurface>,
    /// Discrete encode cache for type-2/3 GVA color targets, keyed by texture
    /// object ref. Separate from [`Self::host_surfaces`] so list ids cannot
    /// clobber type-4 present mids (live: sky `tex_ref=24` vs mid 24).
    pub host_texture_surfaces: BTreeMap<u32, HostSurface>,
    /// Same type-2/3 encode content keyed by target GVA — survives texture_ref
    /// rebinding / small-atlas overwrite of the ref slot.
    pub host_gva_surfaces: BTreeMap<u64, HostSurface>,
    /// Raw compute encode for type-2/3 textures. Retained across GVA unmap;
    /// evicted on task/object lifetime end or descriptor mismatch.
    pub host_linear_textures: BTreeMap<(u32, u32), HostLinearTexture>,
    /// Perf memo: one swizzled RGBA copy per (gva, generation) for linear
    /// sampled textures on the render path; repeat draws clone the Arc
    /// instead of re-copying + re-swizzling the cache bytes. Coherence is
    /// re-established on every lookup by matching the authoritative
    /// [`Self::host_gva_surfaces`] entry's gva/generation/geometry - a stale
    /// entry can never be served, only skipped. Keyed by (task_id, ref).
    /// Byte-bounded LRU ([`LINEAR_SAMPLED_MEMO_BYTE_CAP`]): a cap crossing evicts
    /// the coldest entries, never the whole map (no re-copy cliff).
    pub linear_sampled_memo: LruBytesMemo<(u32, u32), LinearSampledMemo>,
    /// Perf memo for guest-CPU-produced linear textures (no host cache entry,
    /// so no producer generation exists). Coherence is re-established on
    /// every lookup by re-reading the native guest rows and comparing them
    /// byte-exact against the memoized copy — a guest write is always seen;
    /// only the swizzle+alloc (and the engine's content hash+memcmp, via the
    /// generation identity) are skipped on unchanged content. Keyed by
    /// (task_id, level-0 gva, width, height, sample format). Byte-bounded LRU
    /// ([`GUEST_LINEAR_MEMO_BYTE_CAP`]): a cap crossing evicts the least-recently
    /// -used entries down to a low-water mark, never bulk-clearing the hot set.
    pub guest_linear_memo: LruBytesMemo<(u32, u64, u32, u32, u16), GuestLinearMemo>,
    /// Monotonic content-generation source for [`Self::guest_linear_memo`]
    /// and [`Self::type5_view_memo`] (shared so a generation value never
    /// repeats across the two producers).
    pub guest_linear_gen: u64,
    /// Reusable native-row read buffer for the guest-linear memo path.
    pub guest_linear_scratch: Vec<u8>,
    /// Byte-exact revalidated memo for type-5 serialized texture views
    /// (media IOSurface planes). Same contract as
    /// [`Self::guest_linear_memo`]: every bind re-reads the native plane
    /// window; conversion + upload (via the returned content identity) are
    /// skipped on unchanged bytes. Keyed by
    /// (mapping_id, plane, width, height, view pixel format). Byte-bounded LRU
    /// ([`GUEST_LINEAR_MEMO_BYTE_CAP`]).
    pub type5_view_memo: LruBytesMemo<(u32, u32, u32, u32, u16), GuestLinearMemo>,
    /// Byte-exact revalidated memo for the type-11 mapping-backed sampled path
    /// (`load_type11_rgba_static` — small IOSurface textures below the zero-copy
    /// floor, e.g. dock icons under magnification). Same contract as
    /// [`Self::guest_linear_memo`]: every bind re-reads the native BGRA rect;
    /// the BGRA->RGBA convert + the two per-bind allocs + the engine's content
    /// hash+upload (via the returned content identity) are skipped on unchanged
    /// bytes. A dock-magnification burst re-binds the same static icons ~1000x,
    /// so this collapses the `t11_guest` CPU copies that otherwise saturate the
    /// serial drain worker (dock-hover freeze). Keyed by (mapping_id, w, h).
    /// Byte-bounded LRU ([`GUEST_LINEAR_MEMO_BYTE_CAP`]).
    pub type11_memo: LruBytesMemo<(u32, u32, u32), GuestLinearMemo>,
    /// Reusable native BGRA read buffer for the type-11 memo re-read.
    pub type11_memo_scratch: Vec<u8>,
    /// Measurement-only: last guest-visible generation produced by a compute
    /// storage-image writeback for an exact type-11 view. This does not select
    /// engine behavior; it measures safe residency opportunities.
    pub compute_storage_residency: BTreeMap<ComputeStorageResidencyKey, u32>,
    /// Deferred compute writebacks: windows whose guest pages are STALE — the
    /// pinned engine resident at this generation is the authoritative content.
    /// Every host-side read or write of intersecting mapping bytes must flush
    /// first (`runtime::storage_flush::flush_intersecting`). Value = the
    /// engine resident generation to flush.
    pub compute_deferred_flush: BTreeMap<ComputeStorageResidencyKey, u32>,
    /// Deferred render Stores: type-11 windows whose guest pages are STALE —
    /// the pinned engine resident render target is authoritative. Same
    /// flush-on-access contract as [`Self::compute_deferred_flush`]
    /// (`runtime::storage_flush::flush_intersecting`); lifetime boundaries
    /// drop fail-visibly, never write.
    pub render_deferred_flush: BTreeMap<RenderDeferredKey, RenderDeferredEntry>,
    /// Physical page bases of each mapping with live deferred windows, for the
    /// raw task-GVA sampling guard (`storage_flush::flush_intersecting_task_gva`).
    /// Built at defer time from the just-resolved `page_entries`
    /// ([`Self::index_deferred_alias_pages`] — per-sample resolution cost the
    /// boot-19 setup_us regression measured at ~1.4 s/boot); entries drop when
    /// the mapping's last deferred window is taken. A stale entry after a PFN
    /// change costs one spurious no-op flush call, never a wrong flush — the
    /// windows map stays the single flush authority.
    pub deferred_alias_pages: DeferredWindows<u32, std::collections::HashSet<u64>>,
    /// Per-mid last write **command class** (ClearOnly vs Composite) — present path.
    pub surface_write_kind: BTreeMap<u32, SurfaceWriteKind>,
    /// RGB-nonzero pixel count of each mapping's last full-frame Store import
    /// (`import_present ok_runs` resident stats). Diagnostic memory for the
    /// `fullquad_store_noop` proxy — a full-quad draw whose resident content
    /// stats do not move names the incomplete-swap-base class. Never gates
    /// behavior.
    pub import_rgb_nz: BTreeMap<u32, usize>,
    pub present: PresentState,
    pub cursor: CursorState,
    pub display: DisplayHandshake,
    pub fails: Vec<FailEvent>,
    /// Next async job id for ordered stamps.
    pub next_job_id: u64,
    /// Last successful directed mapper capture (consumed on matching MAP/UNMAP).
    pub mapper_capture: Option<MapperCapture>,
    /// Cached IOSurfaceParavirtMapperDevice KVA from capture.
    pub mapper_device_kva: u64,
    /// Sync value table for event + encoder fence domains.
    ///
    /// Key: `(task_id, domain_tag, ref)` → value (event: explicit signal value;
    /// fence: monotonic generation). Domain tags match
    /// [`crate::runtime::plan::event_sync::Domain`] as `u8` (`1` = event,
    /// `2` = blitFence, `3` = computeFence, `4` = renderFence). Stored as a
    /// plain map so `model` does not depend on the planner types.
    pub fence_generations: BTreeMap<(u32, u8, u32), u64>,
    /// Child channel currently being drained (0 = none). Convenience for
    /// single-level skip; prefer [`Self::draining_mask`] for nested drains.
    pub draining_channel: u32,
    /// Bitmask of child channels mid-`drain_child_fifo` (stack). Nested
    /// `drain_other_child_fifos` must skip **all** bits set — otherwise it can
    /// re-enter a mid-packet channel and re-process the same head.
    pub draining_mask: u32,
    /// Contiguous mapping views (`MappingEntry::contig_ptr`) whose page tables
    /// changed. `DeviceState` cannot unmap (no HostOps); the runtime flushes
    /// these via `HostOps::unmap_pages` after dropping the Metal objects that
    /// alias them (`mapper::flush_retired_views`).
    pub retired_views: Vec<(usize, usize)>,
    /// Task-GVA HostOps views (zero-copy import substrate). Dropped on
    /// overlapping UnmapMemory/MapMemory2; flushed via `retired_views`.
    pub gva_host_views: Vec<GvaHostView>,
    /// Linear-window residency keys whose `host_linear_textures` entry died
    /// (task/object delete). `DeviceState` cannot reach the engine; the
    /// runtime unpins these (`storage_flush::retire_linear_residents`) so the
    /// pinned images become LRU-evictable instead of leaking.
    pub retired_linear_residents: Vec<ComputeStorageResidencyKey>,
    /// Deferred linear windows whose guest pages the superseded sync path
    /// WOULD have written (GVA-mapped at defer time): generation + defer-time
    /// page-GPA index. A raw task-GVA read aliasing these pages flushes the
    /// resident into the cache entry and guest pages first
    /// (`storage_flush::flush_intersecting_task_gva`). Cache-only-shaped
    /// windows never enter — their sync path never wrote guest pages either.
    pub linear_deferred_flush:
        DeferredWindows<ComputeStorageResidencyKey, (u32, std::collections::HashSet<u64>)>,
    /// Deferred GVA render-Store windows (type-2/3 color0, `target_gva != 0`)
    /// whose guest bytes + `host_gva_surfaces` encode the superseded sync path
    /// WOULD have written. The engine resident `TargetIdentity::Gva` is the
    /// authoritative content until `storage_flush::flush_gva_one` lands it.
    pub gva_deferred_flush: DeferredWindows<u64, GvaDeferredEntry>,
    /// Monotonic arm counter for [`Self::gva_deferred_flush`] oldest-first cap.
    pub gva_deferred_seq: u64,
    /// Monotonic arm counter for the [`Self::render_deferred_flush`]
    /// oldest-first cap (bounds the pinned resident population).
    pub render_deferred_seq: u64,
    /// GVAs rendered this guest lifetime as an **MRT secondary attachment**
    /// (e.g. the vibrancy RG16Float coverage mask) → its (width, height). The
    /// producer records the identity + geometry here; a later draw sampling a
    /// type-2/3 texture at the same GVA binds the engine resident directly
    /// (`TargetIdentity::Gva{gva,w,h,0}`) instead of reading zero. Coherent by
    /// construction: only GVAs we actively rendered as secondaries are eligible,
    /// and the geometry must match the sampler's descriptor. Cleared at guest
    /// reset with the rest of the lifetime state.
    pub mrt_secondary_gvas: std::collections::HashMap<u64, (u32, u32)>,
    /// GVA windows whose task died (`delete_task`) — the GVA walk is gone, so
    /// the runtime lands these **cache-only** (no guest write) and unpins
    /// (`storage_flush::retire_gva_windows`).
    pub retired_gva_windows: Vec<(u64, GvaDeferredEntry)>,
    /// Mappings presented (CmdDisplaySwap capture) since our last LOAD draw
    /// into them. The present declares the guest pages the finished frame and
    /// the guest may CPU-write them afterwards (inter-buffer damage
    /// forward-copy — no device command, no 0x35), so the first LOAD draw
    /// after a mapping's own present seeds from guest pages instead of
    /// chaining the resident (dual-mid strobe class). Consumed at the type-11
    /// Load seed decision (`metal_draw::resolve_type11_load_choice`).
    pub presented_needs_guest_seed: std::collections::BTreeSet<u32>,
    /// Global monotonic decoded-write counter feeding
    /// [`MappingEntry::last_store_seq`] — the cross-mapping store order the
    /// ClearOnly dual-mid peer-select uses to find the latest-finished
    /// compositor member.
    pub store_seq: u64,
    /// FIFO of proven compositor-output members that received a Composite
    /// full-FB writeback and are not yet matched to a present. The guest
    /// pipelines its double buffer in ring order (store B, store A, present,
    /// present), so each ClearOnly present pairs with the OLDEST unconsumed
    /// member store. Capturing the newest member for every present drops the
    /// other member's frame each cycle and displays a member against the
    /// wrong present slot (dual-mid residue class). Entries are
    /// (mapping_id, content_generation at enqueue); bounded by
    /// [`PRESENT_STORE_FIFO_CAP`] (oldest dropped).
    pub present_store_fifo: std::collections::VecDeque<(u32, u32)>,
    /// Per-tranche draw-timing breakdown (diagnostic). See [`TrancheStats`].
    pub tranche: TrancheStats,
    /// Aggregated ExecIndirect2 packet telemetry (diagnostic). See [`ExecAggStats`].
    pub exec_agg: ExecAggStats,
    /// Batch-ceiling census key of the previous engine draw in the current
    /// packet: (hash of the engine target identity, width, height, bgra).
    /// Measure-only — feeds TrancheStats batch_same_target/batch_joinable;
    /// reset at every ExecIndirect2 packet start. The identity is stored as a
    /// std-hash so the model stays independent of backend engine types.
    pub last_draw_batch_key: Option<(u64, u32, u32, bool)>,
    /// Draw-time zero-copy run memo. See [`GuestRunMemoEntry`] for the
    /// invalidation contract (mirrors `gva_host_views` exactly). A `VecDeque`
    /// so the FIFO cap evict is an O(1) `pop_front` rather than a `Vec`
    /// `remove(0)` that shifts all `GUEST_RUN_MEMO_CAP` (512) entries on every
    /// miss once full.
    pub guest_run_memo: std::collections::VecDeque<GuestRunMemoEntry>,
    /// Covering-view reuse counter — drives the 1-in-32 sampled staleness
    /// verify in `ensure_gva_view` (stale cached-view read class).
    pub view_verify_ctr: u64,
    /// Total stale views the sampled verify caught (fail-logged as
    /// `gva_view_stale`; the view self-heals via retire + rebuild).
    pub view_stale_reads: u64,
    /// Draw-time buffer-bind coherence-flush no-intersection memo. Maps a
    /// full-walked `(task_id, gva, span)` bind to `(validated_signature,
    /// gpa_pages)`: the deferred-index signature the walk ran at, and the exact
    /// guest physical pages the bind's span resolved to.
    /// `storage_flush::flush_intersecting_task_gva` skips its per-page task-PT
    /// walk when the signature is unchanged, and on a signature change re-checks
    /// the cached `gpa_pages` against the current deferred windows **without a PT
    /// walk** (the FFI page translate is the expensive part) — only a real
    /// intersection falls back to the full walk. The pages stay valid until the
    /// task PT remaps the gva range, which invalidates the entry exactly where
    /// `guest_run_memo` is. A 1-in-64 sampled full walk ([`flush_verify_ctr`])
    /// self-heals a missed PT remap. Only fully-probed spans are cached (a
    /// strided walk's page set is incomplete, so it is never stored).
    pub flush_nohit_memo: std::collections::HashMap<(u32, u64, u64), (u64, Vec<u64>)>,
    pub flush_verify_ctr: u64,
    /// Refcounted union of every live deferred window's physical page GPAs — the
    /// fast index behind [`deferred_pages_intersect`]. Maintained incrementally
    /// at window arm/disarm (only the changed window's pages are touched, never
    /// a 24k-page rebuild), so the per-bind recheck is `bind_pages` O(1) lookups
    /// into ONE map instead of the old O(bind_pages × num_windows) scan over
    /// every window's HashSet — the dominant `zc_flush` cost. Refcounts (not a
    /// plain set) because windows share physical pages; a page leaves the index
    /// only when its last window disarms.
    ///
    /// Exactness is a property of the type, not of a repair pass: the three
    /// source maps are [`DeferredWindows`], which hands out no mutable access
    /// outside this module, so every arm and disarm necessarily runs through the
    /// method that also moves the refcount here.
    deferred_page_refs: std::collections::HashMap<u64, u32>,
}

/// Bound for [`DeviceState::present_store_fifo`]: the guest pipelines at most
/// a few frames ahead (double/triple buffer); a deeper backlog means presents
/// stopped consuming (present-style switch) and old entries are stale.
pub const PRESENT_STORE_FIFO_CAP: usize = 8;

/// Domain tag for ch-event segment events (matches event_sync::Domain::Event).
pub const FENCE_DOMAIN_EVENT: u8 = 1;
/// Domain tag for blit fences (matches event_sync::Domain::BlitFence).
pub const FENCE_DOMAIN_BLIT: u8 = 2;
/// Domain tag for compute fences.
pub const FENCE_DOMAIN_COMPUTE: u8 = 3;
/// Domain tag for render fences.
pub const FENCE_DOMAIN_RENDER: u8 = 4;

impl DeviceState {
    /// GPA for a guest PFN under this device's page size.
    #[inline]
    pub fn pfn_gpa(&self, pfn: u32) -> u64 {
        (pfn as u64) << self.page_shift
    }

    #[inline]
    pub fn page_size(&self) -> u64 {
        1u64 << self.page_shift
    }

    /// Create device state for a guest with the given page shift.
    ///
    /// `page_shift` must be **12** (x86_64 / Tahoe) or **14** (arm64e). There
    /// is no default — product create and tests must choose explicitly.
    pub fn new(id: DeviceId, page_shift: u32) -> Self {
        Self {
            id,
            page_shift,
            gfx: GfxRegs::default(),
            iosfc: IosfcRegs::default(),
            is_tahoe: false,
            active_child_mask: 0,
            translation_deferred_mask: 0,
            translation_order_hold_mask: 0,
            translation_order_holds: 0,
            present_translation_holds: 0,
            present_translation_hold_mask: 0,
            pending: PendingWork::default(),
            child_rings: std::array::from_fn(|_| ChannelRing::default()),
            child_stamps: std::array::from_fn(|_| ChannelStamps::default()),
            tasks: std::array::from_fn(|_| TaskEntry::default()),
            map_family_events: 0,
            task_map_spans: Vec::new(),
            objects: BTreeMap::new(),
            texture_to_mapping: BTreeMap::new(),
            mappings: BTreeMap::new(),
            host_surfaces: BTreeMap::new(),
            host_texture_surfaces: BTreeMap::new(),
            host_gva_surfaces: BTreeMap::new(),
            host_linear_textures: BTreeMap::new(),
            compute_storage_residency: BTreeMap::new(),
            compute_deferred_flush: BTreeMap::new(),
            render_deferred_flush: BTreeMap::new(),
            deferred_alias_pages: DeferredWindows::new(),
            surface_write_kind: BTreeMap::new(),
            import_rgb_nz: BTreeMap::new(),
            present: PresentState::default(),
            cursor: CursorState {
                show: true,
                ..Default::default()
            },
            mapper_capture: None,
            mapper_device_kva: 0,
            display: DisplayHandshake::default(),
            fails: Vec::new(),
            next_job_id: 1,
            fence_generations: BTreeMap::new(),
            draining_channel: 0,
            draining_mask: 0,
            retired_views: Vec::new(),
            retired_linear_residents: Vec::new(),
            linear_deferred_flush: DeferredWindows::new(),
            gva_deferred_flush: DeferredWindows::new(),
            mrt_secondary_gvas: std::collections::HashMap::new(),
            gva_deferred_seq: 0,
            render_deferred_seq: 0,
            retired_gva_windows: Vec::new(),
            linear_sampled_memo: LruBytesMemo::new(LINEAR_SAMPLED_MEMO_BYTE_CAP),
            guest_linear_memo: LruBytesMemo::new(GUEST_LINEAR_MEMO_BYTE_CAP),
            guest_linear_gen: 0,
            guest_linear_scratch: Vec::new(),
            type5_view_memo: LruBytesMemo::new(GUEST_LINEAR_MEMO_BYTE_CAP),
            type11_memo: LruBytesMemo::new(GUEST_LINEAR_MEMO_BYTE_CAP),
            type11_memo_scratch: Vec::new(),
            presented_needs_guest_seed: std::collections::BTreeSet::new(),
            store_seq: 0,
            present_store_fifo: std::collections::VecDeque::new(),
            gva_host_views: Vec::new(),
            tranche: TrancheStats::default(),
            exec_agg: ExecAggStats::default(),
            last_draw_batch_key: None,
            guest_run_memo: std::collections::VecDeque::new(),
            view_verify_ctr: 0,
            view_stale_reads: 0,
            flush_nohit_memo: std::collections::HashMap::new(),
            flush_verify_ctr: 0,
            deferred_page_refs: std::collections::HashMap::new(),
        }
    }

    /// Cheap structural signature of the three deferred-writeback indices that
    /// [`crate::runtime::storage_flush::flush_intersecting_task_gva`] consults.
    /// `O(deferred entries)` (tiny), NOT `O(pages)` — `HashSet::len` is `O(1)`.
    /// Changes with overwhelming probability on any add/remove/re-arm; a same-
    /// signature content change is caught by the flush's 1-in-64 sampled walk.
    pub fn deferred_flush_signature(&self) -> u64 {
        const P: u64 = 0x100000001b3;
        let mut s: u64 = self.deferred_alias_pages.len() as u64;
        for (mid, pages) in &self.deferred_alias_pages {
            s = s
                .wrapping_mul(P)
                .wrapping_add(*mid as u64)
                .wrapping_add((pages.len() as u64) << 20);
        }
        s = s
            .wrapping_mul(P)
            .wrapping_add(self.linear_deferred_flush.len() as u64);
        for (key, (generation, pages)) in &self.linear_deferred_flush {
            s = s
                .wrapping_mul(P)
                .wrapping_add(key.texture_ref as u64)
                .wrapping_add((key.map_generation as u64) << 12)
                .wrapping_add((*generation as u64) << 28)
                .wrapping_add((pages.len() as u64) << 44);
        }
        s = s
            .wrapping_mul(P)
            .wrapping_add(self.gva_deferred_flush.len() as u64);
        for (gva, entry) in &self.gva_deferred_flush {
            s = s
                .wrapping_mul(P)
                .wrapping_add(*gva)
                .wrapping_add(entry.armed_seq << 8);
        }
        s
    }

    /// True if any of `gpa_pages` (a bind's resolved physical pages) falls in a
    /// live deferred-writeback window. Same membership test as the flush walk's
    /// visitor, but over already-resolved pages — no task page-table walk. Used
    /// to re-validate a `flush_nohit_memo` entry after a deferred-signature
    /// change without paying the per-page FFI translate again.
    ///
    /// O(`gpa_pages`) lookups into the refcounted [`deferred_page_refs`] index,
    /// not the old O(`gpa_pages` × num_windows) scan over every window's set.
    pub fn deferred_pages_intersect(&self, gpa_pages: &[u64]) -> bool {
        gpa_pages
            .iter()
            .any(|p| self.deferred_page_refs.contains_key(p))
    }

    /// Add one reference to each of `pages` in the deferred-union index.
    fn deferred_ref_add_pages(&mut self, pages: &std::collections::HashSet<u64>) {
        for &p in pages {
            *self.deferred_page_refs.entry(p).or_insert(0) += 1;
        }
    }

    /// Drop one reference from each of `pages`; a page leaves the index when its
    /// last referencing window disarms.
    fn deferred_ref_sub_pages(&mut self, pages: &std::collections::HashSet<u64>) {
        for &p in pages {
            if let Some(c) = self.deferred_page_refs.get_mut(&p) {
                *c -= 1;
                if *c == 0 {
                    self.deferred_page_refs.remove(&p);
                }
            }
        }
    }

    /// Arm (or re-arm) a deferred GVA render-Store window, keeping the union
    /// index in sync (re-arm subtracts the superseded page set first).
    pub fn arm_gva_deferred_window(&mut self, gva: u64, entry: GvaDeferredEntry) {
        if let Some(old) = self.gva_deferred_flush.get(&gva) {
            let old = old.pages.clone();
            self.deferred_ref_sub_pages(&old);
        }
        self.deferred_ref_add_pages(&entry.pages);
        self.gva_deferred_flush.0.insert(gva, entry);
    }

    /// Arm (or re-arm) a linear compute-storage deferred window, keeping the
    /// union index in sync.
    pub fn arm_linear_deferred_window(
        &mut self,
        key: ComputeStorageResidencyKey,
        generation: u32,
        pages: std::collections::HashSet<u64>,
    ) {
        if let Some((_, old)) = self.linear_deferred_flush.get(&key) {
            let old = old.clone();
            self.deferred_ref_sub_pages(&old);
        }
        self.deferred_ref_add_pages(&pages);
        self.linear_deferred_flush.0.insert(key, (generation, pages));
    }

    /// Disarm a linear compute-storage deferred window, keeping the union index
    /// in sync. Returns whether an entry was present.
    pub fn disarm_linear_deferred_window(&mut self, key: &ComputeStorageResidencyKey) -> bool {
        if let Some((_, pages)) = self.linear_deferred_flush.0.remove(key) {
            self.deferred_ref_sub_pages(&pages);
            true
        } else {
            false
        }
    }

    /// Detach `e`'s contiguous view for later unmap (page table changed).
    /// Returns the retired (ptr, len) to push into `retired_views`.
    fn take_mapping_view(e: &mut MappingEntry) -> Option<(usize, usize)> {
        if e.contig_ptr == 0 {
            return None;
        }
        let v = (e.contig_ptr, e.contig_len);
        e.contig_ptr = 0;
        e.contig_len = 0;
        Some(v)
    }

    /// Detach every HostOps mapping owned by the current guest lifetime.
    ///
    /// Device reset is a lifetime boundary even when QEMU itself remains alive.
    /// Returning the views lets the runtime invalidate backend aliases first,
    /// then release them through the bound HostOps implementation.
    pub fn take_all_host_views(&mut self) -> Vec<(usize, usize)> {
        let mut views = std::mem::take(&mut self.retired_views);
        for mapping in self.mappings.values_mut() {
            if let Some(view) = Self::take_mapping_view(mapping) {
                views.push(view);
            }
        }
        views.extend(self.gva_host_views.drain(..).filter_map(|view| {
            (view.ptr != 0 && view.ptr_len != 0).then_some((view.ptr, view.ptr_len))
        }));
        self.guest_run_memo.clear();
        self.flush_nohit_memo.clear();
        views
    }

    /// Snapshot fence generation if present.
    pub fn fence_generation(&self, task_id: u32, domain: u8, fence_ref: u32) -> Option<u64> {
        self.fence_generations
            .get(&(task_id, domain, fence_ref))
            .copied()
    }

    /// Store fence generation (monotonic update owned by the planner).
    pub fn set_fence_generation(&mut self, task_id: u32, domain: u8, fence_ref: u32, value: u64) {
        if fence_ref == 0 {
            return;
        }
        self.fence_generations
            .insert((task_id, domain, fence_ref), value);
    }

    /// Record a clear-only write to `mapping_id` (display_clear / CLEAR Store).
    pub fn note_surface_clear(&mut self, mapping_id: u32) {
        if mapping_id == 0 {
            return;
        }
        // Guest Clear wipes the surface: next present of this mid must not be
        // treated as a finished composite (unless a later Draw Store re-marks
        // Composite).
        self.surface_write_kind
            .insert(mapping_id, SurfaceWriteKind::ClearOnly);
    }

    /// Record a composite/draw Store to `mapping_id`.
    pub fn note_surface_composite(&mut self, mapping_id: u32) {
        if mapping_id == 0 {
            return;
        }
        self.surface_write_kind
            .insert(mapping_id, SurfaceWriteKind::Composite);
    }

    /// Record the latest successful non-self full-geometry type-11 edge.
    pub fn note_compositor_output(
        &mut self,
        source_mapping: u32,
        output_mapping: u32,
        width: u32,
        height: u32,
        generation: u32,
    ) {
        if source_mapping == 0
            || output_mapping == 0
            || source_mapping == output_mapping
            || width == 0
            || height == 0
        {
            return;
        }
        self.present.compositor_output_mapping = output_mapping;
        self.present.compositor_output_source = source_mapping;
        self.present.compositor_output_generation = generation;
        self.present.compositor_output_width = width;
        self.present.compositor_output_height = height;
        self.present.compositor_output_members.insert(
            output_mapping,
            CompositorOutputMember {
                width,
                height,
                source: source_mapping,
            },
        );
    }

    /// A draw Store published a **complete** frame for `mapping_id` into guest
    /// pages (full-frame resident writeback, `import_present ok_runs`): grant
    /// compositor-output membership at that geometry. Unlike the one-shot
    /// full-coverage draw edges, this fires on every steady-state composite
    /// pass, so both halves of a guest double buffer qualify on any boot.
    /// Does not move the output pin itself — only Composite writebacks on a
    /// member do (see [`Self::refresh_compositor_output_member`]).
    pub fn note_compositor_member_published(&mut self, mapping_id: u32, width: u32, height: u32) {
        if mapping_id == 0 || width == 0 || height == 0 {
            return;
        }
        let member = self
            .present
            .compositor_output_members
            .entry(mapping_id)
            .or_insert_with(|| {
                crate::observe::off(format!(
                    "compositor_member_grant mid={mapping_id} {width}x{height} reason=full_frame_publish"
                ));
                CompositorOutputMember {
                    width,
                    height,
                    source: 0,
                }
            });
        // Geometry change (mode switch / re-geom): follow the newly published
        // geometry; keep the proven source edge when unchanged.
        if member.width != width || member.height != height {
            member.width = width;
            member.height = height;
            member.source = 0;
        }
        // Protocol-structural dense marker: this member now holds a complete
        // full-frame. Advance its `dense_frame_seq` so a peer that never got a
        // full frame (nor a seed) shows up as lagging to the peer seed and to
        // the retention proxies. The counter is monotonic per full-frame Store
        // across all members.
        self.present.dense_frame_counter = self.present.dense_frame_counter.saturating_add(1);
        let seq = self.present.dense_frame_counter;
        self.present.dense_frame_seq.insert(mapping_id, seq);
    }

    /// Advance the per-present tile-epoch clock and return the new value. Call
    /// EXACTLY ONCE per present cycle (see [`PresentState::tile_epoch`]); draws
    /// between two presents stamp the value current at draw time.
    pub fn advance_tile_epoch(&mut self) -> u64 {
        self.present.tile_epoch = self.present.tile_epoch.saturating_add(1);
        self.present.tile_epoch
    }

    /// Stamp every tile a pixel-space damage rect covers on `mid` with the
    /// current [`PresentState::tile_epoch`]. `rect` = `(x0,y0,x1,y1)` in target
    /// pixels (half-open); clamped to the `w`×`h` target. No-op on a degenerate
    /// rect or unknown geometry. Allocates the mid's tile array lazily. O(tiles
    /// in rect) — a menu strip / tooltip covers a handful; a full-frame store
    /// covers all [`TILE_GEN_TILES`].
    pub fn bump_tile_gen(&mut self, mid: u32, rect: (u32, u32, u32, u32), w: u32, h: u32) {
        if mid == 0 || w == 0 || h == 0 {
            return;
        }
        let (x0, y0, x1, y1) = rect;
        if x1 <= x0 || y1 <= y0 {
            return;
        }
        let tw = w as usize;
        let th = h as usize;
        let x0 = (x0 as usize).min(tw.saturating_sub(1));
        let y0 = (y0 as usize).min(th.saturating_sub(1));
        // Half-open [x0,x1): the last covered pixel is x1-1.
        let xe = (x1 as usize).min(tw).saturating_sub(1);
        let ye = (y1 as usize).min(th).saturating_sub(1);
        let gx0 = (x0 * TILE_GEN_GRID_W / tw).min(TILE_GEN_GRID_W - 1);
        let gx1 = (xe * TILE_GEN_GRID_W / tw).min(TILE_GEN_GRID_W - 1);
        let gy0 = (y0 * TILE_GEN_GRID_H / th).min(TILE_GEN_GRID_H - 1);
        let gy1 = (ye * TILE_GEN_GRID_H / th).min(TILE_GEN_GRID_H - 1);
        let epoch = self.present.tile_epoch;
        let arr = self
            .present
            .tile_gen
            .entry(mid)
            .or_insert_with(|| Box::new([0u64; TILE_GEN_TILES]));
        for gy in gy0..=gy1 {
            let row = gy * TILE_GEN_GRID_W;
            for gx in gx0..=gx1 {
                arr[row + gx] = epoch;
            }
        }
    }

    /// Count the tiles where `peer_mid` is fresher than `presented_mid` by at
    /// least [`crate::runtime::census::present_proxy::RETENTION_GAP_MARGIN`] epochs — the
    /// divergent (residue) tiles the guest erased in the peer but our presented
    /// mid still shows stale. Pure generation compare over [`TILE_GEN_TILES`]
    /// `u64`s (no pixel scan, no allocation). Returns 0 when either mid has no
    /// tile map yet (steady-state / bootstrap). Never counts a peer tile the peer
    /// never drew (epoch 0 ≤ presented) — that is the by-construction guard that a
    /// damage-only peer can never override the presented mid's good background.
    /// Returns `(count, [gx0, gy0, gx1, gy1])` — the divergent-tile count and the
    /// inclusive tile-grid bounding box of the divergent region (all zero when
    /// count is 0). The bbox lets a proxy confirm the divergence LOCALIZES to the
    /// visible residue (e.g. the menu strip / a rubber-band trail) rather than
    /// smearing whole-frame.
    pub fn divergent_tile_count(&self, presented_mid: u32, peer_mid: u32) -> (u32, [u32; 4]) {
        if presented_mid == peer_mid {
            return (0, [0; 4]);
        }
        let (Some(pres), Some(peer)) = (
            self.present.tile_gen.get(&presented_mid),
            self.present.tile_gen.get(&peer_mid),
        ) else {
            return (0, [0; 4]);
        };
        let margin = crate::runtime::census::present_proxy::RETENTION_GAP_MARGIN;
        let mut n = 0u32;
        let (mut gx0, mut gy0, mut gx1, mut gy1) = (u32::MAX, u32::MAX, 0u32, 0u32);
        for i in 0..TILE_GEN_TILES {
            if peer[i] >= pres[i].saturating_add(margin) {
                n += 1;
                let gx = (i % TILE_GEN_GRID_W) as u32;
                let gy = (i / TILE_GEN_GRID_W) as u32;
                gx0 = gx0.min(gx);
                gy0 = gy0.min(gy);
                gx1 = gx1.max(gx);
                gy1 = gy1.max(gy);
            }
        }
        if n == 0 {
            (0, [0; 4])
        } else {
            (n, [gx0, gy0, gx1, gy1])
        }
    }

    /// A Composite-class writeback landed on a proven compositor-output member
    /// at its proven geometry: move the graph output pin to that member so
    /// ClearOnly presents follow the guest's buffer alternation (damage passes
    /// never re-prove full coverage). Returns `Some(previous_output_mapping)`
    /// when the pin moved to a different mapping, `None` when only refreshed
    /// in place or not a matching member.
    pub fn refresh_compositor_output_member(
        &mut self,
        mapping_id: u32,
        width: u32,
        height: u32,
        generation: u32,
    ) -> Option<u32> {
        if mapping_id == 0 || width == 0 || height == 0 {
            return None;
        }
        let member = *self.present.compositor_output_members.get(&mapping_id)?;
        if member.width != width || member.height != height {
            return None;
        }
        let prev = self.present.compositor_output_mapping;
        self.present.compositor_output_mapping = mapping_id;
        self.present.compositor_output_source = member.source;
        self.present.compositor_output_generation = generation;
        self.present.compositor_output_width = width;
        self.present.compositor_output_height = height;
        (prev != mapping_id).then_some(prev)
    }

    /// Whether `mapping_id` has been proven a compositor output (any geometry).
    pub fn is_compositor_output_member(&self, mapping_id: u32) -> bool {
        self.present
            .compositor_output_members
            .contains_key(&mapping_id)
    }

    /// The same-geometry presented peer of `mapping_id` holding the freshest
    /// full frame: `(peer_mid, presented_seq, peer_seq)`, when that peer is
    /// strictly ahead.
    ///
    /// The lag this reports is **not** a staleness measure on its own.
    /// `dense_frame_counter` is a single global counter bumped once per
    /// full-frame publish by *any* member at *any* geometry, so a member that
    /// just took its turn already sits a whole turnaround behind the one that
    /// published last with nothing stale about either. It must never decide
    /// *which* surface to present: the transaction names plane 0's surface id
    /// and that is the only correct capture source.
    ///
    /// Its readers are the present census and the ClearOnly torn-capture
    /// substitution in the drain (`stale_present_substitute`), which is measured
    /// dead on the x86 PCI pathway and unmeasurable on arm64. Nothing on the
    /// guest-named present path reads it: the a/b peer seed used to, and was
    /// deleted once resident unification was shown to make it a copy onto
    /// itself for every target the guest actually displays.
    ///
    /// The peer set is scoped to same-geometry members that have themselves been
    /// **presented** at this geometry ([`Self::presented_at`]) — the SAME gate
    /// [`Self::output_group_for`] uses to keep never-displayed full-frame
    /// publishers (WebKit content tiles, offscreen scratch surfaces) out of an
    /// output group. Without it, two *distinct logical outputs* at the same
    /// resolution — the desktop swapchain and a never-presented publisher that
    /// happens to composite full 1920×1080 frames — would share one peer set, and
    /// a publisher's frame could be seeded into a real output on a transition
    /// (the intermittent a/b residue / stale-tile class). The invariant is
    /// pathway-independent: a buffer the guest has never chosen to display is
    /// never a valid seed *source* into one. Genuine swapchain siblings all get
    /// presented as they alternate as front, so they stay in the set.
    pub fn dense_retention_gap(&self, mapping_id: u32, w: u32, h: u32) -> Option<(u32, u64, u64)> {
        if mapping_id == 0 || w == 0 || h == 0 {
            return None;
        }
        let member = self.present.compositor_output_members.get(&mapping_id)?;
        if member.width != w || member.height != h {
            return None;
        }
        let presented_seq = self
            .present
            .dense_frame_seq
            .get(&mapping_id)
            .copied()
            .unwrap_or(0);
        let (peer_mid, peer_seq) = self
            .present
            .compositor_output_members
            .iter()
            .filter(|(&mid, m)| {
                mid != mapping_id && m.width == w && m.height == h && self.presented_at(mid, w, h)
            })
            .filter_map(|(mid, _)| self.present.dense_frame_seq.get(mid).map(|s| (*mid, *s)))
            .max_by_key(|(_, s)| *s)?;
        (peer_seq > presented_seq).then_some((peer_mid, presented_seq, peer_seq))
    }

    /// The same-geometry compositor-output peer of `mapping_id` holding the
    /// freshest full frame (`dense_frame_seq`). Unlike [`Self::dense_retention_gap`]
    /// this does NOT gate on a whole-frame seq lag — tile-level residue is exactly
    /// the case where the whole-frame seqs match but individual tiles diverge, so
    /// the tile-divergence path needs the peer regardless of whole-frame lag.
    pub fn compositor_geometry_peer(&self, mapping_id: u32, w: u32, h: u32) -> Option<u32> {
        if mapping_id == 0 || w == 0 || h == 0 {
            return None;
        }
        let member = self.present.compositor_output_members.get(&mapping_id)?;
        if member.width != w || member.height != h {
            return None;
        }
        self.present
            .compositor_output_members
            .iter()
            .filter(|(mid, m)| **mid != mapping_id && m.width == w && m.height == h)
            .map(|(mid, _)| {
                (
                    *mid,
                    self.present.dense_frame_seq.get(mid).copied().unwrap_or(0),
                )
            })
            .max_by_key(|(_, s)| *s)
            .map(|(mid, _)| mid)
    }

    /// Fill `out` (cleared first) with the pixel-space rects `(x0,y0,x1,y1)` of
    /// the tiles where `peer_mid` is fresher than `presented_mid` by the retention
    /// margin — the residue tiles to composite from the peer's resident. Adjacent
    /// divergent tiles in a row are coalesced into one rect to cut the copy-region
    /// count. Tiling matches the `present_proxy` DAMAGE_GRID. Returns the number
    /// of divergent TILES (not rects). Reuses `out` so the present path allocates
    /// only on growth. A peer tile the peer never drew (epoch 0 ≤ presented) is
    /// never emitted — the by-construction guard against pulling black over the
    /// presented mid's good background.
    pub fn collect_divergent_tile_rects(
        &self,
        presented_mid: u32,
        peer_mid: u32,
        w: u32,
        h: u32,
        out: &mut Vec<(u32, u32, u32, u32)>,
    ) -> u32 {
        out.clear();
        if presented_mid == peer_mid || w == 0 || h == 0 {
            return 0;
        }
        let (Some(pres), Some(peer)) = (
            self.present.tile_gen.get(&presented_mid),
            self.present.tile_gen.get(&peer_mid),
        ) else {
            return 0;
        };
        let margin = crate::runtime::census::present_proxy::RETENTION_GAP_MARGIN;
        let mut count = 0u32;
        for gy in 0..TILE_GEN_GRID_H {
            let y0 = (gy * h as usize / TILE_GEN_GRID_H) as u32;
            let y1 = ((gy + 1) * h as usize / TILE_GEN_GRID_H) as u32;
            let row = gy * TILE_GEN_GRID_W;
            let mut run_start: Option<usize> = None;
            for gx in 0..TILE_GEN_GRID_W {
                let divergent = peer[row + gx] >= pres[row + gx].saturating_add(margin);
                if divergent {
                    count += 1;
                    if run_start.is_none() {
                        run_start = Some(gx);
                    }
                } else if let Some(start) = run_start.take() {
                    let x0 = (start * w as usize / TILE_GEN_GRID_W) as u32;
                    let x1 = (gx * w as usize / TILE_GEN_GRID_W) as u32;
                    out.push((x0, y0, x1, y1));
                }
            }
            if let Some(start) = run_start.take() {
                let x0 = (start * w as usize / TILE_GEN_GRID_W) as u32;
                out.push((x0, y0, w, y1));
            }
        }
        count
    }

    /// For a presented `mapping_id`, find its freshest same-geometry peer and
    /// count the tiles where that peer is fresher by the retention margin — the
    /// per-tile damage-coverage residue. `Some((peer_mid,
    /// divergent_count, tile_bbox))` when a peer exists; `None` when there is no
    /// peer. Pure generation compare, allocation-free.
    pub fn tile_divergence_vs_peer(
        &self,
        mapping_id: u32,
        w: u32,
        h: u32,
    ) -> Option<(u32, u32, [u32; 4])> {
        let peer = self.compositor_geometry_peer(mapping_id, w, h)?;
        let (count, bbox) = self.divergent_tile_count(mapping_id, peer);
        Some((peer, count, bbox))
    }

    /// Present/scanout evidence that `mapping_id` was displayed at `w`x`h`
    /// **by its current incarnation** (see [`Self::note_presented_geom`]).
    pub fn presented_at(&self, mapping_id: u32, w: u32, h: u32) -> bool {
        self.presented_geom_live(mapping_id) == Some((w, h))
    }

    /// The geometry `mapping_id` was presented at, or `None` when there is no
    /// evidence *or* the evidence belongs to a prior incarnation of this id.
    ///
    /// The generation compare is what makes the evidence self-invalidating. It
    /// replaced five scattered `presented_geoms.remove` sites, every one of
    /// which was a place that had to *remember* to forget — and one of them was
    /// wrong in a way that blacked out the desktop. `objects.rs`'s type-4
    /// attach calls `map_surface` (which pruned unconditionally) and only
    /// *afterwards* runs the fingerprint compare that decides whether this is a
    /// new incarnation at all. On an identical page plan that compare says "the
    /// SAME incarnation, deferred windows and the resident survive" — but the
    /// presented evidence was already gone, so a proven swapchain buffer was
    /// demoted to a private per-mid resident that had never held the
    /// accumulated full frame, and every draw until the next present landed
    /// there. The screen kept only the damaged rects and went black elsewhere.
    ///
    /// Tying the evidence to `map_generation` makes each prune site's own rule
    /// hold automatically: the sites that mean "genuinely new surface"
    /// (`unmap_surface`, `attach_mapping_internal`, and `objects.rs`'s refresh,
    /// which prunes only on the branch that bumps) invalidate it, and the ones
    /// that deliberately keep the lifetime undecided pending a fingerprint
    /// compare (`map_surface`, `condemn_surface_backing`) keep it — which is
    /// exactly what they say they intend for every other piece of state.
    fn presented_geom_live(&self, mapping_id: u32) -> Option<(u32, u32)> {
        let seen = self.present.presented_geoms.get(&mapping_id)?;
        (seen.map_generation == self.map_generation_or_zero(mapping_id))
            .then_some((seen.width, seen.height))
    }

    /// A mapping's lifetime counter, or 0 when we hold no entry for the id.
    ///
    /// Same convention as `surface_identity`'s per-mid generation. An id we
    /// have never mapped is not a *different* incarnation, it is the absence of
    /// one, and must compare equal to itself — the guest can name a surface in
    /// a display transaction before our mapping entry for it exists, and that
    /// evidence must not be silently discarded.
    fn map_generation_or_zero(&self, mapping_id: u32) -> u32 {
        self.mappings
            .get(&mapping_id)
            .map(|m| m.map_generation)
            .unwrap_or(0)
    }

    /// Record a present/scanout action displaying `mapping_id` at `w`x`h`.
    /// This is the only evidence class that can qualify a mapping for
    /// OutputGroup unification.
    pub fn note_presented_geom(&mut self, mapping_id: u32, w: u32, h: u32) {
        if mapping_id == 0 || w == 0 || h == 0 {
            return;
        }
        // Stamp the incarnation this evidence belongs to, so it expires on its
        // own when the id is recycled into a different surface.
        let map_generation = self.map_generation_or_zero(mapping_id);
        self.present.presented_geoms.insert(
            mapping_id,
            PresentedGeom {
                width: w,
                height: h,
                map_generation,
            },
        );
        // Latch the geometry as a proven multi-buffer swapchain the first time
        // the guest has named two distinct surfaces there (see
        // `output_group_geoms`). The latch is sticky, so a later buffer recycle
        // that momentarily leaves a single presented surface cannot collapse
        // the group.
        if !self.present.output_group_geoms.contains(&(w, h))
            && self.presented_count(w, h) >= 2
        {
            self.present.output_group_geoms.insert((w, h));
        }
    }

    /// Number of distinct surfaces the guest has named in a display transaction
    /// at `w`x`h` (the arming condition for [`Self::output_group_for`]).
    fn presented_count(&self, w: u32, h: u32) -> usize {
        self.present
            .presented_geoms
            .keys()
            .filter(|&&mid| self.presented_geom_live(mid) == Some((w, h)))
            .count()
    }

    /// Group id when `mapping_id` has been **presented** at `w`x`h` and at least
    /// one OTHER surface has been presented at the same geometry — the two then
    /// act as alternating storage for ONE logical framebuffer (guest copy-swap
    /// contract) and every one of them resolves to the shared
    /// `TargetIdentity::OutputGroup`. A geometry only ever presented from one
    /// surface stays per-mid: no pair, nothing to unify.
    ///
    /// **The admission criterion is the decoded display transaction, and nothing
    /// else.** [`Self::presented_at`] records that the guest named this surface
    /// as plane 0 of a display transaction at this geometry, which is the guest
    /// stating outright that the surface is the scanout source for that frame.
    /// There is no stronger evidence available, and in particular it is stronger
    /// than `compositor_output_members`, which is *inferred* from our own
    /// full-frame-publish detector and resource-graph edges. Requiring both
    /// excluded surfaces the guest genuinely scans out but which never tripped
    /// the publish detector: they resolved to a private `Surface` identity with
    /// no resident behind it, and the export declined
    /// (`export_present_miss outcome=orphan … group=ready` — the group holding
    /// the desktop was ready at the same geometry the whole time) leaving a
    /// desktop black everywhere except the rect the guest had just damaged.
    ///
    /// It also keeps sampled sub-surfaces out, and for a better reason than
    /// before: a WebKit content tile or scrollbar publishes full frames but is
    /// never *named in a display transaction*, so it has no `presented_at` entry
    /// and cannot join. Unifying those would chain distinct surfaces onto one
    /// resident (the Safari-scroll black-band class).
    ///
    /// The geometry latch ([`PresentState::output_group_geoms`]) keeps a surface
    /// unified across buffer recycles that momentarily drop the concurrent peer
    /// count to one and would otherwise re-expose a per-mid resident (the
    /// black-background class). It also *arms* on the very first frame two
    /// surfaces appear together, because `note_presented_geom` evaluates the
    /// arming condition immediately after its own insert. It is therefore the
    /// only thing that decides membership at a geometry: there is no live peer
    /// recount to disagree with it.
    pub fn output_group_for(&self, mapping_id: u32, w: u32, h: u32) -> Option<u32> {
        self.output_group_resolve(mapping_id, w, h).ok()
    }

    /// [`Self::output_group_for`], plus the reading taken by whichever check
    /// refused. This is the only implementation; `output_group_for` is its
    /// `.ok()`, so the two can never disagree about the admission decision.
    ///
    /// A caller that only learns "not a member" has to *guess* which of the two
    /// admission conditions failed, and both have said the wrong thing here
    /// before. The miss carries the state it read instead.
    pub fn output_group_resolve(
        &self,
        mapping_id: u32,
        w: u32,
        h: u32,
    ) -> Result<u32, OutputGroupMiss> {
        if !self.presented_at(mapping_id, w, h) {
            let current = self.map_generation_or_zero(mapping_id);
            return Err(match self.present.presented_geoms.get(&mapping_id) {
                // An entry from a superseded incarnation: the id was recycled
                // into a genuinely different surface, which must re-earn its
                // qualification. Distinct from having no entry at all, because
                // this is the state a wrong prune used to manufacture.
                Some(g) if g.map_generation != current => OutputGroupMiss::PriorIncarnation {
                    presented: (g.width, g.height),
                    entry_gen: g.map_generation,
                    current_gen: current,
                },
                _ => OutputGroupMiss::NotPresentedHere {
                    presented: self.presented_geom_live(mapping_id),
                },
            });
        }
        // One question, one answer: has this geometry been proven a multi-buffer
        // swapchain? A live recount of same-geometry peers used to sit under
        // this, and given the `presented_at` gate above it tested the *identical*
        // predicate the latch arms on (`presented_count(w, h) >= 2`) — a second
        // mechanism re-deriving a permanent fact from transient state.
        //
        // It could only have admitted anyone the latch had missed, which needs
        // the arming predicate to become true with no `note_presented_geom`
        // running to observe it. Its inputs are `presented_geoms` (one insert
        // site, latch check immediately after) and each mid's `map_generation`,
        // and evidence never comes back to life
        // (`a_superseded_presented_entry_never_becomes_live_again`), so the count
        // rises only inside the call that checks the latch. Measured before
        // deleting: a probe on that branch emitted nothing across boot 87, whose
        // `resident_identity_shared` lines show this resolver admitting mids
        // 2, 4, 5 and 6 at 1920x1080 the whole time.
        if self.present.output_group_geoms.contains(&(w, h)) {
            Ok(1)
        } else {
            Err(OutputGroupMiss::NoPeer)
        }
    }

    /// Queue a proven member's Composite full-FB writeback for present↔store
    /// FIFO pairing (see [`DeviceState::present_store_fifo`]). Consecutive
    /// writebacks into the same member coalesce (multi-pass stores of one
    /// frame must not shift the pairing); the queue drops its oldest entry
    /// past [`PRESENT_STORE_FIFO_CAP`]. Returns false when `mapping_id` is
    /// not a member at this geometry (nothing queued).
    pub fn note_member_store(
        &mut self,
        mapping_id: u32,
        width: u32,
        height: u32,
        generation: u32,
    ) -> bool {
        let Some(member) = self.present.compositor_output_members.get(&mapping_id) else {
            return false;
        };
        if member.width != width || member.height != height {
            return false;
        }
        if let Some(back) = self.present_store_fifo.back_mut() {
            if back.0 == mapping_id {
                back.1 = generation;
                return true;
            }
        }
        if self.present_store_fifo.len() >= PRESENT_STORE_FIFO_CAP {
            self.present_store_fifo.pop_front();
        }
        self.present_store_fifo.push_back((mapping_id, generation));
        true
    }

    /// Record that `mapping_id` is being presented and report whether a
    /// full-frame Store **named it** since its own previous present.
    ///
    /// Returns `Some(seq)` — the unchanged [`PresentState::dense_frame_seq`] —
    /// when none did. `None` on the member's first present (no prior witness)
    /// and whenever the seq advanced.
    ///
    /// Structural only: decoded Store bookkeeping, never measured content, and
    /// never the resident. Say what that leaves out, because the name reads
    /// broader than the check: a `None` here means the guest sent a frame for
    /// this mid, **not** that the resident this present will read holds it. When
    /// the two disagree — draws routed to a per-mid resident while the present
    /// reads the shared one — the seq advances all the same and this stays
    /// quiet. `present_identity_flip` is the gate for that; see
    /// [`PresentState::dense_frame_seq`].
    ///
    /// Records the witness on every call, so a member that stays unbacked
    /// reports once per present rather than once per lifetime.
    pub fn note_present_backing(&mut self, mapping_id: u32) -> Option<u64> {
        if mapping_id == 0 {
            return None;
        }
        let seq = self
            .present
            .dense_frame_seq
            .get(&mapping_id)
            .copied()
            .unwrap_or(0);
        let previous = self.present.presented_dense_seq.insert(mapping_id, seq);
        match previous {
            Some(prev) if prev == seq => Some(seq),
            _ => None,
        }
    }

    fn forget_compositor_mapping(&mut self, mapping_id: u32) {
        crate::runtime::census::present_proxy::forget_display_store_sample(mapping_id);
        self.present.compositor_output_members.remove(&mapping_id);
        // Prune the dense-frame seq: a recycled mapping id must not inherit a
        // stale predecessor's dense seq.
        self.present.dense_frame_seq.remove(&mapping_id);
        // Same rule for the presented-seq witness: a recycled id must not
        // compare its first present against a predecessor's seq.
        self.present.presented_dense_seq.remove(&mapping_id);
        // Prune the per-mid tile-epoch grid too (mirrors dense_frame_seq): a
        // recycled mapping id must not inherit a predecessor's tile epochs, or a
        // logically-unrelated surface would show phantom cross-mid divergence.
        self.present.tile_gen.remove(&mapping_id);
        // `presented_geoms` is NOT pruned here. It carries the incarnation that
        // presented it, so an unmap (which bumps `map_generation`) invalidates
        // it automatically, while a condemned backing — which keeps the
        // generation on purpose, pending the fingerprint compare — keeps its
        // evidence too, exactly as it keeps the deferred windows.
        // Prune the present-boundary seed flag too. It marks "this mid was just
        // presented, so its next LOAD re-seeds from the front" — a per-lifetime
        // signal that MUST NOT survive a teardown. Left stale, a recycled
        // mapping_id (a new, logically-unrelated surface reusing this id after
        // DeleteIOSurfaceBacking2) would have its FIRST LOAD draw consume this
        // flag, take the present-boundary seed path, and bleed the CURRENT
        // retained front frame (a different surface's pixels at +0x188) over its
        // own ready resident — the "background/window content doesn't clear
        // cleanly" residue class.
        self.presented_needs_guest_seed.remove(&mapping_id);
        if self.present.compositor_output_mapping == mapping_id
            || self.present.compositor_output_source == mapping_id
        {
            self.present.compositor_output_mapping = 0;
            self.present.compositor_output_source = 0;
            self.present.compositor_output_generation = 0;
            self.present.compositor_output_width = 0;
            self.present.compositor_output_height = 0;
        }
    }

    /// Last write class for present keep-prior decisions.
    pub fn surface_write_kind(&self, mapping_id: u32) -> SurfaceWriteKind {
        self.surface_write_kind
            .get(&mapping_id)
            .copied()
            .unwrap_or(SurfaceWriteKind::Unknown)
    }

    pub fn reset(&mut self) {
        // A translation hold that is still standing here never resolved. The
        // hold itself is control flow — the FIFO is parked until an AIR module
        // finishes loading and the packet is retried, not consumed — so it is
        // census. THIS is the failure: the device went away with guest packets
        // still parked behind a load that never completed, and those packets are
        // lost. Reading it at the lifetime boundary needs no age, depth or
        // timeout; the guest's own teardown is the deadline.
        if self.translation_order_hold_mask != 0 || self.translation_deferred_mask != 0 {
            crate::observe::fail(format!(
                "translation_hold_unreleased held_mask={:#x} producer_mask={:#x} episodes={} \
                 (device reset with guest packets still parked behind an AIR load)",
                self.translation_order_hold_mask,
                self.translation_deferred_mask,
                self.translation_order_holds
            ));
        }
        let id = self.id;
        let page_shift = self.page_shift;
        // Keep the interrupt-status Arcs wired to the registry slot: the
        // lock-free ISR read rail clones them once at device create.
        let intr_disp = Arc::clone(&self.gfx.interrupt_status_disp);
        let intr_gpu = Arc::clone(&self.gfx.interrupt_status_gpu);
        let intr_fault = Arc::clone(&self.gfx.interrupt_fault);
        let fifo_read = Arc::clone(&self.gfx.fifo_read);
        intr_disp.store(0, Ordering::Release);
        intr_gpu.store(0, Ordering::Release);
        intr_fault.store(0, Ordering::Release);
        fifo_read.store(0, Ordering::Release);
        *self = Self::new(id, page_shift);
        self.gfx.interrupt_status_disp = intr_disp;
        self.gfx.interrupt_status_gpu = intr_gpu;
        self.gfx.interrupt_fault = intr_fault;
        self.gfx.fifo_read = fifo_read;
    }

    pub fn alloc_job_id(&mut self) -> u64 {
        let id = self.next_job_id;
        self.next_job_id = self.next_job_id.saturating_add(1);
        id
    }

    /// Queue the engine-unpin for a dying linear cache entry that still owns a
    /// resident image (see `retired_linear_residents`).
    fn retire_linear_resident(&mut self, task_id: u32, texture_ref: u32, e: &HostLinearTexture) {
        if e.resident_gen == 0 || e.row_stride > u32::MAX as u64 {
            return;
        }
        self.retired_linear_residents
            .push(ComputeStorageResidencyKey::linear(
                task_id,
                texture_ref,
                e.gva,
                e.row_stride as u32,
                e.row_stride.saturating_mul(e.height as u64),
                e.width,
                e.height,
                e.pixel_format,
            ));
    }

    fn retire_task_linear_residents(&mut self, task_id: u32) {
        let doomed: Vec<(u32, HostLinearTexture)> = self
            .host_linear_textures
            .iter()
            .filter(|((t, _), e)| *t == task_id && e.resident_gen != 0)
            .map(|((_, r), e)| {
                (
                    *r,
                    HostLinearTexture {
                        bytes: Vec::new(),
                        ..e.clone()
                    },
                )
            })
            .collect();
        for (r, e) in doomed {
            self.retire_linear_resident(task_id, r, &e);
        }
    }

    /// Deferred GVA render-Store windows lose their GVA walk with the task
    /// (walks try `task_id` then `task_id >> 1`) — hand them to the runtime
    /// for a cache-only landing (`storage_flush::retire_gva_windows`); never
    /// write guest pages from teardown.
    fn retire_task_gva_windows(&mut self, task_id: u32) {
        let doomed: Vec<u64> = self
            .gva_deferred_flush
            .iter()
            .filter(|(_, e)| e.task_id == task_id || e.task_id >> 1 == task_id)
            .map(|(&gva, _)| gva)
            .collect();
        for gva in doomed {
            if let Some(entry) = self.gva_deferred_flush.0.remove(&gva) {
                self.deferred_ref_sub_pages(&entry.pages);
                self.retired_gva_windows.push((gva, entry));
            }
        }
    }

    /// Take the deferred GVA window at exactly `gva`, if any.
    pub fn take_gva_deferred_window(&mut self, gva: u64) -> Option<GvaDeferredEntry> {
        let entry = self.gva_deferred_flush.0.remove(&gva)?;
        self.deferred_ref_sub_pages(&entry.pages);
        Some(entry)
    }

    /// Take the oldest-armed deferred GVA window (window-cap eviction).
    pub fn take_oldest_gva_deferred_window(&mut self) -> Option<(u64, GvaDeferredEntry)> {
        let gva = self
            .gva_deferred_flush
            .iter()
            .min_by_key(|(_, e)| e.armed_seq)
            .map(|(&gva, _)| gva)?;
        let entry = self.gva_deferred_flush.0.remove(&gva)?;
        self.deferred_ref_sub_pages(&entry.pages);
        Some((gva, entry))
    }

    /// Remove the least-recently-armed render-deferred window, for the
    /// oldest-first flush that bounds the window population (and thus the pinned
    /// resident count) under a compositing burst. Mirrors
    /// [`Self::take_oldest_gva_deferred_window`]; prunes the alias index for the
    /// window's mapping so the incremental deferred-page refcount stays exact.
    pub fn take_oldest_render_deferred_window(
        &mut self,
    ) -> Option<(RenderDeferredKey, RenderDeferredEntry)> {
        let key = *self
            .render_deferred_flush
            .iter()
            .min_by_key(|(_, e)| e.armed_seq)
            .map(|(k, _)| k)?;
        let entry = self.render_deferred_flush.remove(&key)?;
        self.prune_alias_index(key.mapping_id);
        Some((key, entry))
    }

    pub fn define_task(&mut self, task_id: u32, length: u64, directory_pfn: u32) -> bool {
        if task_id as usize >= MAX_TASKS {
            StateMutationDecline::DefineTaskIdRange { task_id }.emit(u64::from(task_id));
            return false;
        }
        // Drop objects for this task on redefine.
        self.objects.retain(|&(t, _), _| t != task_id);
        self.retire_task_linear_residents(task_id);
        self.retire_task_gva_windows(task_id);
        self.host_linear_textures.retain(|&(t, _), _| t != task_id);
        self.clear_task_map_spans(task_id);
        // New directory ⇒ old GVA HostOps views alias the wrong PT — retire.
        let mut i = 0;
        while i < self.gva_host_views.len() {
            if self.gva_host_views[i].task_id == task_id {
                let v = self.gva_host_views.swap_remove(i);
                if v.ptr != 0 && v.ptr_len != 0 {
                    self.retired_views.push((v.ptr, v.ptr_len));
                }
            } else {
                i += 1;
            }
        }
        self.guest_run_memo.retain(|e| e.task_id != task_id);
        self.flush_nohit_memo.retain(|&(t, _, _), _| t != task_id);
        self.tasks[task_id as usize] = TaskEntry::define(length, directory_pfn);
        true
    }

    /// Record a MapMemory2 span (guest already installed PTEs; notify only).
    pub fn note_task_map(&mut self, task_id: u32, gva: u64, length: u64) {
        if gva == 0 || length == 0 {
            return;
        }
        // Replace exact duplicate; keep other spans.
        self.task_map_spans
            .retain(|s| !(s.task_id == task_id && s.gva == gva && s.length == length));
        self.task_map_spans.push(TaskMapSpan {
            task_id,
            gva,
            length,
        });
    }

    /// Drop MapMemory2 spans overlapping Unmap `[gva, gva+length)`.
    pub fn note_task_unmap(&mut self, task_id: u32, gva: u64, length: u64) {
        if gva == 0 || length == 0 {
            return;
        }
        let end = gva.saturating_add(length);
        self.task_map_spans.retain(|s| {
            if s.task_id != task_id {
                return true;
            }
            let s_end = s.gva.saturating_add(s.length);
            // Keep spans that do not overlap the unmap range.
            !(s.gva < end && gva < s_end)
        });
    }

    pub fn clear_task_map_spans(&mut self, task_id: u32) {
        self.task_map_spans.retain(|s| s.task_id != task_id);
    }

    /// Product GVA write gate: if this task has any recorded MapMemory2 spans,
    /// `[gva, gva+len)` must be fully covered by one span. Empty registry ⇒ allow
    /// (unit fixtures and pre-Map paths).
    pub fn gva_write_allowed(&self, task_id: u32, gva: u64, len: u64) -> bool {
        if len == 0 {
            return true;
        }
        let has_any = self.task_map_spans.iter().any(|s| {
            s.task_id == task_id || s.task_id == (task_id >> 1) || (s.task_id << 1) == task_id
        });
        if !has_any {
            return true;
        }
        self.task_map_spans.iter().any(|s| {
            (s.task_id == task_id || s.task_id == (task_id >> 1) || (s.task_id << 1) == task_id)
                && s.covers(gva, len)
        })
    }

    /// PVG `CmdDeleteTask` (op `0x20`): drop task directory + object list entries.
    /// Guest reuses task ids; leaving stale active tasks corrupts GVA walks.
    pub fn delete_task(&mut self, task_id: u32) -> bool {
        if task_id as usize >= MAX_TASKS {
            StateMutationDecline::DeleteTaskIdRange { task_id }.emit(u64::from(task_id));
            return false;
        }
        if !self.tasks[task_id as usize].active {
            return false;
        }
        self.objects.retain(|&(t, _), _| t != task_id);
        self.retire_task_linear_residents(task_id);
        self.retire_task_gva_windows(task_id);
        self.host_linear_textures.retain(|&(t, _), _| t != task_id);
        self.clear_task_map_spans(task_id);
        // Clear texture→mapping latches for this task.
        let doomed_refs: Vec<u32> = self
            .texture_to_mapping
            .keys()
            .filter_map(|&(t, r)| if t == task_id { Some(r) } else { None })
            .collect();
        self.texture_to_mapping.retain(|&(t, _), _| t != task_id);
        // Drop texture-ref encode slots that were latched for this task. Other
        // refs are left (cache is ref-keyed without task); delete_object also
        // evicts. GVA encode cache retained until Unmap of that range.
        for r in doomed_refs {
            self.host_texture_surfaces.remove(&r);
        }
        // Task teardown ≡ all GPU VA maps for this task go away — retire any
        // HostOps views we held (does not touch host_gva_surfaces encode).
        // Runtime flushes retired_views via HostOps::unmap_pages.
        let mut i = 0;
        while i < self.gva_host_views.len() {
            if self.gva_host_views[i].task_id == task_id {
                let v = self.gva_host_views.swap_remove(i);
                if v.ptr != 0 && v.ptr_len != 0 {
                    self.retired_views.push((v.ptr, v.ptr_len));
                }
            } else {
                i += 1;
            }
        }
        self.guest_run_memo.retain(|e| e.task_id != task_id);
        self.flush_nohit_memo.retain(|&(t, _, _), _| t != task_id);
        self.tasks[task_id as usize] = TaskEntry::default();
        true
    }

    pub fn set_object_list(&mut self, task_id: u32, pfn: u32, count: u32) -> bool {
        if task_id as usize >= MAX_TASKS {
            StateMutationDecline::SetObjectListTaskIdRange { task_id }.emit(u64::from(task_id));
            return false;
        }
        if !self.tasks[task_id as usize].active {
            StateMutationDecline::SetObjectListTaskInactive { task_id }.emit(u64::from(task_id));
            return false;
        }
        self.tasks[task_id as usize].object_list_pfn = pfn;
        self.tasks[task_id as usize].object_list_count = count;
        true
    }

    pub fn insert_object(&mut self, task_id: u32, ref_: u32, entry: ObjectEntry) -> bool {
        let discriminant = (u64::from(task_id) << 32) | u64::from(ref_);
        if task_id as usize >= MAX_TASKS {
            StateMutationDecline::InsertObjectTaskIdRange {
                task_id,
                object_ref: ref_,
            }
            .emit(discriminant);
            return false;
        }
        if !self.tasks[task_id as usize].active {
            StateMutationDecline::InsertObjectTaskInactive {
                task_id,
                object_ref: ref_,
            }
            .emit(discriminant);
            return false;
        }
        self.objects.insert((task_id, ref_), entry);
        true
    }

    pub fn delete_object(&mut self, task_id: u32, ref_: u32) -> bool {
        let removed = self.objects.remove(&(task_id, ref_)).is_some();
        if removed {
            self.host_texture_surfaces.remove(&ref_);
            if let Some(e) = self.host_linear_textures.remove(&(task_id, ref_)) {
                self.retire_linear_resident(task_id, ref_, &e);
            }
            self.texture_to_mapping.remove(&(task_id, ref_));
        }
        removed
    }

    /// Bump [`MappingEntry::map_generation`] (never 0 after first bump).
    ///
    /// The bump orphans any generation-keyed resident for the mapping — for a
    /// large surface that moment is load-bearing when diagnosing sample
    /// fallback classes, so it traces under `REIMS_VGPU_DRAW_LOG=1`.
    pub fn bump_map_generation(mapping_id: u32, e: &mut MappingEntry) {
        e.map_generation = e.map_generation.wrapping_add(1);
        if e.map_generation == 0 {
            e.map_generation = 1;
        }
        if crate::observe::enabled() && (e.width as u64) * (e.height as u64) >= 250_000 {
            crate::observe::line(format!(
                "map_gen_bump mid={mapping_id} {}x{} gen={}",
                e.width, e.height, e.map_generation
            ));
        }
    }

    /// Drop compute storage-residency mirror entries whose byte window
    /// `[surface_offset, span_end)` intersects a guest write of
    /// `[lo, hi)` on this mapping. The mirror claims "guest pages still hold
    /// exactly the resident's content for this window" — any intersecting
    /// write breaks that claim; disjoint windows (ping-pong canvases) survive.
    pub fn invalidate_storage_residency_window(&mut self, mapping_id: u32, lo: u64, hi: u64) {
        self.compute_storage_residency.retain(|key, _| {
            key.mapping_id != mapping_id || key.span_end <= lo || key.surface_offset >= hi
        });
    }

    /// Remove and return every deferred-writeback window intersecting
    /// `[lo, hi)` on this mapping. The caller owns flushing each returned
    /// entry (or reporting the loss) — once taken, the map no longer names it.
    pub fn take_deferred_flush_windows(
        &mut self,
        mapping_id: u32,
        lo: u64,
        hi: u64,
    ) -> Vec<(ComputeStorageResidencyKey, u32)> {
        let keys: Vec<ComputeStorageResidencyKey> = self
            .compute_deferred_flush
            .keys()
            .filter(|key| {
                key.mapping_id == mapping_id && key.span_end > lo && key.surface_offset < hi
            })
            .cloned()
            .collect();
        let taken: Vec<(ComputeStorageResidencyKey, u32)> = keys
            .into_iter()
            .filter_map(|key| {
                self.compute_deferred_flush
                    .remove(&key)
                    .map(|gen| (key, gen))
            })
            .collect();
        if !taken.is_empty() {
            self.prune_alias_index(mapping_id);
        }
        taken
    }

    /// Remove and return every deferred render-Store window intersecting
    /// `[lo, hi)` on this mapping. Same take-then-flush ownership contract as
    /// [`Self::take_deferred_flush_windows`].
    pub fn take_render_deferred_windows(
        &mut self,
        mapping_id: u32,
        lo: u64,
        hi: u64,
    ) -> Vec<(RenderDeferredKey, RenderDeferredEntry)> {
        let keys: Vec<RenderDeferredKey> = self
            .render_deferred_flush
            .keys()
            .filter(|key| {
                key.mapping_id == mapping_id && key.span_end > lo && key.surface_offset < hi
            })
            .cloned()
            .collect();
        let taken: Vec<(RenderDeferredKey, RenderDeferredEntry)> = keys
            .into_iter()
            .filter_map(|key| {
                self.render_deferred_flush
                    .remove(&key)
                    .map(|entry| (key, entry))
            })
            .collect();
        if !taken.is_empty() {
            self.prune_alias_index(mapping_id);
        }
        taken
    }

    /// Record the physical page bases of `mapping_id` in the raw-GVA alias
    /// index. Called at defer time, when `page_entries` are freshly resolved
    /// (the Store/dispatch just targeted them) — never at sample time.
    pub fn index_deferred_alias_pages(&mut self, mapping_id: u32) {
        let page_shift = self.page_shift;
        let page = self.page_size();
        let Some(m) = self.mappings.get(&mapping_id) else {
            return;
        };
        let set: std::collections::HashSet<u64> = m
            .page_entries
            .iter()
            .filter_map(|&e| crate::contract::iosurface_pages::entry_gpa_shift(e, page_shift))
            .map(|gpa| gpa & !(page - 1))
            .collect();
        // Re-index: drop the superseded page set's refs before adding the fresh
        // one so the union index tracks exactly the live pages.
        if let Some(old) = self.deferred_alias_pages.get(&mapping_id) {
            let old = old.clone();
            self.deferred_ref_sub_pages(&old);
        }
        if set.is_empty() {
            self.deferred_alias_pages.0.remove(&mapping_id);
        } else {
            self.deferred_ref_add_pages(&set);
            self.deferred_alias_pages.0.insert(mapping_id, set);
        }
    }

    /// Drop the alias-index entry once no deferred window (compute or render)
    /// names this mapping anymore.
    fn prune_alias_index(&mut self, mapping_id: u32) {
        let live = self
            .compute_deferred_flush
            .keys()
            .any(|k| k.mapping_id == mapping_id)
            || self
                .render_deferred_flush
                .keys()
                .any(|k| k.mapping_id == mapping_id);
        if !live {
            if let Some(old) = self.deferred_alias_pages.0.remove(&mapping_id) {
                self.deferred_ref_sub_pages(&old);
            }
        }
    }

    /// Drop cached page list + contig view without unmapping the slot.
    ///
    /// Used on ReplacePhysical / rebind: guest may have recycled PFNs into the
    /// zone freelist; the next Store must re-resolve before any host write or
    /// import-present DMA (freelist `0xff000000ff000000` class).
    pub fn invalidate_mapping_pages(&mut self, mapping_id: u32) -> bool {
        let Some(e) = self.mappings.get_mut(&mapping_id) else {
            return false;
        };
        let had = !e.page_entries.is_empty() || e.contig_ptr != 0;
        e.page_entries.clear();
        e.page_table_kva = 0;
        e.condemned_entries = None;
        Self::bump_map_generation(mapping_id, e);
        let retired = Self::take_mapping_view(e);
        if let Some(v) = retired {
            self.retired_views.push(v);
        }
        #[cfg(all(feature = "backend-metal", target_os = "macos"))]
        crate::backend::metal::runtime::type11_guest_texture_invalidate(mapping_id);
        had
    }

    /// Trailing `DeleteIOSurfaceBacking2`: retire the page bindings — nothing
    /// may write through possibly-recycled pages (boot-16 PTE-corruption
    /// rule) — but KEEP content state (map_generation, geometry, resident
    /// identity, deferred windows). The deleted backing may belong to a PRIOR
    /// incarnation of a recycled id whose slot already carries a live surface
    /// with an unflushed paint (black-band class): the next page resolve
    /// compares against the stashed fingerprint and either reprieves (same
    /// plan) or bumps + drops (different plan). Returns whether a fingerprint
    /// was stashed; on `false` the caller should fall back to full teardown.
    pub fn condemn_surface_backing(&mut self, mapping_id: u32) -> bool {
        self.forget_compositor_mapping(mapping_id);
        self.host_surfaces.remove(&mapping_id);
        if let Some(old) = self.deferred_alias_pages.0.remove(&mapping_id) {
            self.deferred_ref_sub_pages(&old);
        }
        let Some(e) = self.mappings.get_mut(&mapping_id) else {
            return false;
        };
        if e.page_entries.is_empty() {
            return false;
        }
        e.condemned_entries = Some(std::mem::take(&mut e.page_entries));
        e.page_table_kva = 0;
        let retired = Self::take_mapping_view(e);
        if let Some(v) = retired {
            self.retired_views.push(v);
        }
        #[cfg(all(feature = "backend-metal", target_os = "macos"))]
        crate::backend::metal::runtime::type11_guest_texture_invalidate(mapping_id);
        true
    }

    /// Whether `mapping_id` sits in the condemned state (backing deleted, no
    /// resolve since). A second delete in this state is genuinely dead — the
    /// caller tears down for real.
    pub fn mapping_backing_condemned(&self, mapping_id: u32) -> bool {
        self.mappings
            .get(&mapping_id)
            .is_some_and(|e| e.condemned_entries.is_some())
    }

    pub fn map_surface(&mut self, mapping_id: u32) -> bool {
        if mapping_id as usize >= MAX_MAPPINGS {
            StateMutationDecline::MapSurfaceIdRange { mapping_id }.emit(u64::from(mapping_id));
            return false;
        }
        let e = self.mappings.entry(mapping_id).or_default();
        e.mapped = true;
        // Fresh MAP invalidates any previous page table / geom for this slot.
        // Stale has_geom after 1920→1440 remap blocks writebacks (size mismatch)
        // and freezes host console at the old mode. The MAP notify often TRAILS
        // our eager resolve of the same surface (a Store discovers the mapping
        // before the guest's notification drains) — so never bump eagerly:
        // stash the page fingerprint and let the next resolve decide (same
        // plan = same incarnation, generation and deferred windows survive;
        // different plan = genuine new surface, bump + drop there). Geometry
        // stays cleared either way — samples fail-closed until re-resolve, so
        // a genuinely new surface can never be served the old resident.
        if !e.page_entries.is_empty() && e.condemned_entries.is_none() {
            e.condemned_entries = Some(std::mem::take(&mut e.page_entries));
        } else {
            e.page_entries.clear();
        }
        e.page_table_kva = 0;
        e.device_desc.clear();
        e.content_generation = 0;
        e.has_geom = false;
        e.width = 0;
        e.height = 0;
        e.format = 0;
        let retired = Self::take_mapping_view(e);
        if let Some(v) = retired {
            self.retired_views.push(v);
        }
        #[cfg(all(feature = "backend-metal", target_os = "macos"))]
        crate::backend::metal::runtime::type11_guest_texture_invalidate(mapping_id);
        // Fresh MAP: prior host-cache for this surface_id is stale, and so is
        // any present evidence — the slot may hold a NEW surface (a stale
        // presented_geoms entry could wrongly qualify a recycled publish-only
        // surface for OutputGroup unification).
        self.host_surfaces.remove(&mapping_id);
        // Present evidence is stamped with the incarnation and deliberately NOT
        // dropped here. A fresh MAP does not yet know whether this is a new
        // surface — that is what the fingerprint compare decides, bumping the
        // generation when it is. Dropping it eagerly demoted a proven swapchain
        // buffer to a private resident for every draw until its next present,
        // which is the black-desktop class.
        crate::runtime::census::present_proxy::forget_display_store_sample(mapping_id);
        true
    }

    pub fn unmap_surface(&mut self, mapping_id: u32) -> bool {
        if mapping_id as usize >= MAX_MAPPINGS {
            StateMutationDecline::UnmapSurfaceIdRange { mapping_id }.emit(u64::from(mapping_id));
            return false;
        }
        self.forget_compositor_mapping(mapping_id);
        if let Some(e) = self.mappings.get_mut(&mapping_id) {
            e.mapped = false;
            e.page_entries.clear();
            e.page_table_kva = 0;
            e.condemned_entries = None;
            e.mapping_internal = 0;
            e.device_desc.clear();
            Self::bump_map_generation(mapping_id, e);
            e.has_geom = false;
            e.width = 0;
            e.height = 0;
            e.format = 0;
            let retired = Self::take_mapping_view(e);
            if let Some(v) = retired {
                self.retired_views.push(v);
            }
            #[cfg(all(feature = "backend-metal", target_os = "macos"))]
            crate::backend::metal::runtime::type11_guest_texture_invalidate(mapping_id);
            self.host_surfaces.remove(&mapping_id);
            true
        } else {
            false
        }
    }

    /// Attach directed MappingInternal capture to a mapped slot.
    pub fn attach_mapping_internal(&mut self, mapping_id: u32, mapping_internal: u64) -> bool {
        if mapping_id as usize >= MAX_MAPPINGS {
            StateMutationDecline::AttachMappingIdRange { mapping_id }.emit(u64::from(mapping_id));
            return false;
        }
        if mapping_internal == 0 {
            StateMutationDecline::AttachMappingInternalZero { mapping_id }
                .emit(u64::from(mapping_id));
            return false;
        }
        let e = self.mappings.entry(mapping_id).or_default();
        // A re-statement of the SAME MappingInternal (notify trailing our
        // eager resolve) is not a new surface: keep bindings, generation,
        // resident, and deferred windows untouched.
        if e.mapping_internal == mapping_internal {
            e.mapped = true;
            return true;
        }
        e.mapped = true;
        e.mapping_internal = mapping_internal;
        e.page_entries.clear();
        e.page_table_kva = 0;
        e.condemned_entries = None;
        e.device_desc.clear();
        e.content_generation = 0;
        Self::bump_map_generation(mapping_id, e);
        // New MappingInternal ⇒ new surface; force device-desc re-resolve.
        e.has_geom = false;
        e.width = 0;
        e.height = 0;
        e.format = 0;
        let retired = Self::take_mapping_view(e);
        if let Some(v) = retired {
            self.retired_views.push(v);
        }
        #[cfg(all(feature = "backend-metal", target_os = "macos"))]
        crate::backend::metal::runtime::type11_guest_texture_invalidate(mapping_id);
        // New MappingInternal ⇒ new surface, and the `bump_map_generation`
        // above is what retires the stale present evidence: it is stamped with
        // the incarnation that recorded it, so the recycled slot cannot inherit
        // an OutputGroup qualification it did not earn.
        true
    }

    /// Cache the 0x200-byte guest device descriptor for plane/surface sample windows.
    pub fn set_mapping_device_desc(&mut self, mapping_id: u32, desc: &[u8]) -> bool {
        if mapping_id as usize >= MAX_MAPPINGS {
            StateMutationDecline::MappingDeviceDescIdRange { mapping_id }
                .emit(u64::from(mapping_id));
            return false;
        }
        if desc.is_empty() {
            StateMutationDecline::MappingDeviceDescEmpty { mapping_id }.emit(u64::from(mapping_id));
            return false;
        }
        let e = self.mappings.entry(mapping_id).or_default();
        e.device_desc = desc.to_vec();
        true
    }

    pub fn set_mapping_geom(
        &mut self,
        mapping_id: u32,
        width: u32,
        height: u32,
        format: u16,
    ) -> bool {
        if mapping_id as usize >= MAX_MAPPINGS {
            StateMutationDecline::MappingGeomIdRange { mapping_id }.emit(u64::from(mapping_id));
            return false;
        }
        if width == 0 {
            StateMutationDecline::MappingGeomWidthZero { mapping_id }.emit(u64::from(mapping_id));
            return false;
        }
        if height == 0 {
            StateMutationDecline::MappingGeomHeightZero { mapping_id }.emit(u64::from(mapping_id));
            return false;
        }
        if width > crate::model::MAX_SCANOUT_DIM {
            StateMutationDecline::MappingGeomWidthRange { mapping_id, width }
                .emit((u64::from(mapping_id) << 32) | u64::from(width));
            return false;
        }
        if height > crate::model::MAX_SCANOUT_DIM {
            StateMutationDecline::MappingGeomHeightRange { mapping_id, height }
                .emit((u64::from(mapping_id) << 32) | u64::from(height));
            return false;
        }
        let e = self.mappings.entry(mapping_id).or_default();
        // Geom change (mode switch / rematerialize) is a new surface identity:
        // reset content_generation and drop the cached Metal texture object
        // (its descriptor no longer matches; the guest pages stay authoritative).
        if e.width != width || e.height != height {
            e.content_generation = 0;
            #[cfg(all(feature = "backend-metal", target_os = "macos"))]
            crate::backend::metal::runtime::type11_guest_texture_invalidate(mapping_id);
        }
        e.has_geom = true;
        e.width = width;
        e.height = height;
        e.format = format;
        true
    }

    /// Bump content generation after a write into the mapping (0 never skips).
    pub fn mark_mapping_written(&mut self, mapping_id: u32) -> u32 {
        let seq = self.store_seq.wrapping_add(1);
        let Some(m) = self.mappings.get_mut(&mapping_id) else {
            return 0;
        };
        self.store_seq = seq;
        m.last_store_seq = seq;
        m.content_generation = m.content_generation.wrapping_add(1);
        if m.content_generation == 0 {
            m.content_generation = 1;
        }
        m.content_generation
    }

    pub fn record_fail(&mut self, ev: FailEvent) {
        // Fail-visible (I2): decode/contract gaps must reach the always-on fail
        // log, not only the in-memory test vec — silently dropped commands
        // (e.g. unknown display-channel opcodes) otherwise leave no trace in a
        // live boot.
        //
        // Through `Emit` rather than `format!("{ev:?}")`: the debug rendering
        // carried the same facts but spelled them `MalformedRootPacket { reason:
        // "bad-packet-size", head: 4096 }`, which is neither `reason=<slug>` nor
        // greppable by the vocabulary every other subsystem uses.
        crate::observe::Emit::decline("fail_event", &ev).fail();
        self.fails.push(ev);
    }
}

/// The sticky geometry latch is the only thing that decides output-group
/// membership.
///
/// It used to share that job with a live recount of same-geometry presented
/// peers, which tested the identical predicate the latch arms on — two
/// mechanisms answering one question, so one of them had to go. These tests hold
/// the line: the latch alone admits, and the state the recount used to admit on
/// is refused.
#[cfg(test)]
mod output_group_membership_tests {
    use super::*;
    use crate::model::PAGE_SHIFT_X86;
    use crate::runtime::import_present::OUTPUT_GROUP_ID;

    fn state() -> DeviceState {
        DeviceState::new(DeviceId(1), PAGE_SHIFT_X86)
    }

    /// Presented peers with no latch do not make a group.
    ///
    /// This state is not reachable through `note_presented_geom` — it checks the
    /// latch immediately after its own insert — so it is built by writing
    /// `presented_geoms` directly. That is the point: the recount's admission
    /// could only ever come from bookkeeping the arming path never saw, and the
    /// contract says a swapchain is proven at the moment two surfaces are
    /// presented together, not inferred from a later recount.
    #[test]
    fn presented_peers_without_the_latch_are_not_a_group() {
        let mut s = state();
        for mid in [4u32, 9u32] {
            s.present.presented_geoms.insert(
                mid,
                PresentedGeom {
                    width: 1920,
                    height: 1080,
                    map_generation: 0,
                },
            );
        }
        assert!(!s.present.output_group_geoms.contains(&(1920, 1080)));
        assert_eq!(
            s.output_group_resolve(4, 1920, 1080),
            Err(OutputGroupMiss::NoPeer),
            "a live recount must not admit what the arming path never proved"
        );
    }

    /// The guest's own ordering arms the latch before anything can resolve, so
    /// nothing else is needed to admit the first pair.
    ///
    /// `note_presented_geom` checks the arming condition immediately after its
    /// own insert, which is what made the recount below it unreachable: the
    /// count cannot pass 2 anywhere else.
    #[test]
    fn the_second_present_arms_the_latch_that_admits_both_members() {
        let mut s = state();
        s.map_surface(4);
        s.note_presented_geom(4, 1920, 1080);
        assert_eq!(
            s.output_group_resolve(4, 1920, 1080),
            Err(OutputGroupMiss::NoPeer),
            "one presented surface is not a swapchain"
        );
        s.map_surface(9);
        s.note_presented_geom(9, 1920, 1080);
        assert!(
            s.present.output_group_geoms.contains(&(1920, 1080)),
            "the second present must arm the latch inside note_presented_geom"
        );
        for mid in [4u32, 9u32] {
            assert_eq!(s.output_group_resolve(mid, 1920, 1080), Ok(OUTPUT_GROUP_ID));
        }
    }

    /// The lemma the deletion rests on: presented evidence never comes back to
    /// life, so the peer count can only rise inside `note_presented_geom` — and
    /// that is the one place the latch condition is checked.
    ///
    /// `bump_map_generation` wraps but explicitly skips 0, and no site removes a
    /// `mappings` entry, so `map_generation_or_zero` leaves 0 exactly once and
    /// never revisits any value it has left. An entry stamped with a superseded
    /// generation is therefore dead permanently, not until the next recycle.
    #[test]
    fn a_superseded_presented_entry_never_becomes_live_again() {
        let mut s = state();
        s.map_surface(4);
        s.note_presented_geom(4, 1920, 1080);
        assert!(s.presented_at(4, 1920, 1080));

        let mut seen = vec![s.map_generation_or_zero(4)];
        for _ in 0..64 {
            let e = s.mappings.get_mut(&4).expect("mappings entries are not removed");
            DeviceState::bump_map_generation(4, e);
            let now = s.map_generation_or_zero(4);
            assert_ne!(now, 0, "a bump never lands back on the never-mapped value");
            assert!(
                !seen.contains(&now),
                "generation {now} recurred; the evidence it stamped would revive"
            );
            seen.push(now);
            assert!(
                !s.presented_at(4, 1920, 1080),
                "evidence from generation {} must stay dead",
                seen[0]
            );
        }
    }
}

#[cfg(test)]
mod fail_vocabulary_tests {
    use super::*;
    use crate::observe::{Decline, REGISTRY};

    /// Every `FailEvent` names a *specific* check. Written as one assertion per
    /// variant rather than a loop so the expected slug is visible next to the
    /// value that produces it — this table is the thing a reader checks against
    /// `/tmp/reims-vgpu-fail.log`.
    #[test]
    fn every_fail_event_variant_names_its_own_check() {
        assert_eq!(
            FailEvent::UnknownRootOpcode {
                opcode: 0x20,
                total_size: 16
            }
            .slug(),
            "unknown_root_opcode"
        );
        assert_eq!(
            FailEvent::UnknownChildOpcode {
                channel: 5,
                opcode: 6,
                total_size: 32
            }
            .slug(),
            "unknown_child_opcode"
        );
        assert_eq!(
            FailEvent::BadMmioAccess {
                window: MmioWindow::Gfx,
                offset: 0x1000,
                size: 2
            }
            .slug(),
            "bad_mmio_access"
        );
        // The malformed variants forward to the fault, so two different checks
        // on the same variant must not share a slug — that collapse is the
        // defect the vocabulary exists to prevent.
        let desync = FailEvent::MalformedRootPacket {
            fault: PacketFault::DesyncedHeadTail,
            head: 0,
        };
        let header = FailEvent::MalformedRootPacket {
            fault: PacketFault::RootHeaderRead,
            head: 0,
        };
        assert_eq!(desync.slug(), "packet_desynced_head_tail");
        assert_eq!(header.slug(), "packet_root_header_read");
        assert_ne!(desync.slug(), header.slug());
        assert_eq!(
            FailEvent::UnsupportedExec {
                channel: 3,
                fault: ExecFault::Indirect2Short
            }
            .slug(),
            "exec_indirect2_short"
        );
    }

    /// A slug without the value that caused it is half a diagnostic. The fields
    /// carry the load-bearing numbers, and the root/child distinction shows up
    /// as the presence of `ch=`.
    #[test]
    fn fail_event_fields_carry_the_load_bearing_values() {
        let line = crate::observe::Emit::decline(
            "fail_event",
            &FailEvent::UnknownChildOpcode {
                channel: 5,
                opcode: 6,
                total_size: 32,
            },
        )
        .render();
        assert_eq!(
            line,
            "fail_event reason=unknown_child_opcode ch=5 opcode=0x6 total_size=32"
        );

        let root = crate::observe::Emit::decline(
            "fail_event",
            &FailEvent::MalformedRootPacket {
                fault: PacketFault::BadSize,
                head: 4096,
            },
        )
        .render();
        assert_eq!(root, "fail_event reason=packet_bad_size head=4096");

        let child = crate::observe::Emit::decline(
            "fail_event",
            &FailEvent::MalformedChildPacket {
                channel: 2,
                fault: PacketFault::BadSize,
                head: 4096,
            },
        )
        .render();
        assert_eq!(child, "fail_event reason=packet_bad_size ch=2 head=4096");
    }

    /// Thirteen distinct malformed-packet checks used to be thirteen hyphenated
    /// string literals passed by hand. They are now variants, and no two may
    /// answer with the same slug — otherwise a child tail read and a child head
    /// writeback look identical in the log.
    #[test]
    fn the_thirteen_packet_faults_all_differ() {
        const ALL: &[PacketFault] = &[
            PacketFault::DesyncedHeadTail,
            PacketFault::BadSize,
            PacketFault::Desynced,
            PacketFault::RootHeaderRead,
            PacketFault::RootSnapRead,
            PacketFault::RootStampWriteback,
            PacketFault::ChildHeaderRead,
            PacketFault::ChildRegsBaseRead,
            PacketFault::ChildRegsHeadRead,
            PacketFault::ChildRegsStampRead,
            PacketFault::ChildSnapRead,
            PacketFault::ChildTailRead,
            PacketFault::ChildHeadWriteback,
        ];
        let mut slugs: Vec<&str> = ALL.iter().map(|f| f.slug()).collect();
        slugs.sort_unstable();
        let count = slugs.len();
        slugs.dedup();
        assert_eq!(slugs.len(), count, "two packet faults share a slug");

        // And every one of them is registered, so the crate-wide uniqueness and
        // log-safety gates actually cover them.
        let row = REGISTRY
            .iter()
            .find(|c| c.type_name == "FailEvent")
            .expect("FailEvent is registered");
        for f in ALL {
            assert!(
                row.slugs.contains(&f.slug()),
                "{} is not in the registry row",
                f.slug()
            );
        }
    }

    #[test]
    fn every_state_mutation_check_has_its_own_registered_reason() {
        let declines = [
            StateMutationDecline::DefineTaskIdRange { task_id: 64 },
            StateMutationDecline::DeleteTaskIdRange { task_id: 64 },
            StateMutationDecline::SetObjectListTaskIdRange { task_id: 64 },
            StateMutationDecline::SetObjectListTaskInactive { task_id: 1 },
            StateMutationDecline::InsertObjectTaskIdRange {
                task_id: 64,
                object_ref: 3,
            },
            StateMutationDecline::InsertObjectTaskInactive {
                task_id: 1,
                object_ref: 3,
            },
            StateMutationDecline::MapSurfaceIdRange { mapping_id: 8192 },
            StateMutationDecline::UnmapSurfaceIdRange { mapping_id: 8192 },
            StateMutationDecline::AttachMappingIdRange { mapping_id: 8192 },
            StateMutationDecline::AttachMappingInternalZero { mapping_id: 1 },
            StateMutationDecline::MappingDeviceDescIdRange { mapping_id: 8192 },
            StateMutationDecline::MappingDeviceDescEmpty { mapping_id: 1 },
            StateMutationDecline::MappingGeomIdRange { mapping_id: 8192 },
            StateMutationDecline::MappingGeomWidthZero { mapping_id: 1 },
            StateMutationDecline::MappingGeomHeightZero { mapping_id: 1 },
            StateMutationDecline::MappingGeomWidthRange {
                mapping_id: 1,
                width: crate::model::MAX_SCANOUT_DIM + 1,
            },
            StateMutationDecline::MappingGeomHeightRange {
                mapping_id: 1,
                height: crate::model::MAX_SCANOUT_DIM + 1,
            },
        ];
        let row = REGISTRY
            .iter()
            .find(|class| class.type_name == "StateMutationDecline")
            .expect("state mutation declines are registered");
        let mut slugs = std::collections::HashSet::new();
        for decline in declines {
            assert!(slugs.insert(decline.slug()), "duplicate {}", decline.slug());
            assert!(
                row.slugs.contains(&decline.slug()),
                "{} is not registered",
                decline.slug()
            );
        }
        assert_eq!(slugs.len(), row.slugs.len());
        assert_eq!(
            crate::observe::Emit::decline(
                "model_state_mutation",
                &StateMutationDecline::MappingGeomWidthRange {
                    mapping_id: 7,
                    width: 65_535,
                },
            )
            .render(),
            "model_state_mutation reason=model_mapping_geom_width_range \
             mapping=7 width=65535"
        );
    }

    #[test]
    fn invalid_mapping_geometry_cannot_create_an_out_of_range_slot() {
        let mut state = DeviceState::new(DeviceId(1), crate::model::PAGE_SHIFT_X86);
        let bad_mapping = MAX_MAPPINGS as u32;
        assert!(!state.set_mapping_geom(bad_mapping, 64, 64, 0x50));
        assert!(!state.mappings.contains_key(&bad_mapping));
        assert!(!state.set_mapping_geom(1, 0, 64, 0x50));
        assert!(!state.set_mapping_geom(1, 64, 0, 0x50));
        assert!(!state.mappings.contains_key(&1));
    }
}
