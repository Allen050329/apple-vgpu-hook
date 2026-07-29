//! Device-owned state: registers, rings, tasks, mapper, present, fail log.

use crate::model::{LruBytesMemo, GFX_MMIO_SIZE, MAX_CHANNELS, MAX_MAPPINGS, MAX_TASKS};
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

/// Why the GVA write gate let a write through — or that it did not.
///
/// The gate is the **only** bounds check on host→guest writes, and it has three
/// separate ways to say yes that a `bool` cannot tell apart. That matters
/// because `gva_mem.rs` logs `reason=mem_outside_map` on the refusal while
/// nothing at all reads the allows, so "the gate passed" has never
/// distinguished *checked and covered* from *nothing to check against*.
///
/// `AGENTS.md` names this shape directly: a reason the caller writes is not a
/// reading, and the collapse regrows wherever a `-> bool` crosses a module
/// boundary. This carries the answer out of the check that made it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WriteGate {
    /// A span recorded for this exact task covers the whole range.
    Exact,
    /// This task has no recorded spans at all, so the gate allowed by default.
    /// `delete_task` calls `clear_task_map_spans`, so a write arriving after a
    /// teardown lands here.
    NoSpans,
    /// Spans exist for this task and none covers the range.
    Outside,
}

impl crate::observe::Decline for WriteGate {
    fn slug(&self) -> &'static str {
        match self {
            Self::Exact => "write_gate_exact",
            Self::NoSpans => "write_gate_no_spans",
            Self::Outside => "write_gate_outside",
        }
    }
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
    /// A task the guest has defined but not yet given an object list.
    ///
    /// `object_list_pfn` and `object_list_count` are **zero** because
    /// `DefineTask2` does not carry them. `SetObjectList` (`0x33`) does, and
    /// until it arrives the correct answer to "what object does ref N name" is
    /// "the guest has not said".
    ///
    /// This used to invent `pfn = 1, count = 0x100000` — a page frame the guest
    /// never named and a list of a million entries. Measured on the x86/Vulkan
    /// rail: `lookup_list_entry` then computed entry addresses of `0x1000 + off`
    /// for every task with no list, walked them, and failed with `gva_zero_pfn`
    /// because nothing is mapped there — after which the guest-read fallback
    /// walked the *neighbouring task's* page table at the same address and
    /// decoded whatever it found as this task's object-list entry. Seven such
    /// substitutions per boot, every boot, all from that one lookup.
    pub fn define(length: u64, directory_pfn: u32) -> Self {
        Self {
            active: true,
            length,
            directory_pfn,
            object_list_pfn: 0,
            object_list_count: 0,
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
    /// mach_vm_remap of guest RAM). 0 = not built. This is the surface storage
    /// for the guest mapping, and it is read and written by the **CPU only**:
    /// Metal render targets used to be created directly on this view, which
    /// gave the host GPU a handle on guest RAM, and that alias is deleted.
    /// Guest CPU writes and host page reads still see one copy; a GPU Store
    /// reaches it through the writeback. Retired (never freed in place)
    /// whenever `page_entries` change; see `DeviceState::retired_views`.
    pub contig_ptr: usize,
    pub contig_len: usize,
    /// `map_generation` whose page list was measured non-packed, so no
    /// contiguous view can exist over it. `None` = not measured for the
    /// current list.
    ///
    /// "Packed or not" is a pure function of `page_entries`, and
    /// `map_generation` names that list — the same key that makes `contig_ptr`
    /// above safe to cache. Without it every caller on a fragmented mapping
    /// re-collected the whole page-GPA vector and re-scanned it only to reach
    /// the answer it reached last time.
    pub contig_fragmented_gen: Option<u32>,
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

/// Why a present is not backed by guest work, as reported by
/// [`DeviceState::note_present_backing`].
///
/// Two distinct losses, and the callee names which so the caller cannot supply
/// the word. They differ in what the viewer sees: `Restaled` shows the previous
/// frame again, `NeverStored` shows an uninitialized surface — a black screen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PresentBacking {
    /// Presented again with no full-frame Store naming this mapping since its
    /// own previous present. Carries the unchanged `dense_frame_seq`.
    Restaled { seq: u64 },
    /// First present since this mapping was created, and no full-frame Store has
    /// ever named it. The surface is uninitialized.
    NeverStored,
}

/// Which rail holds the authoritative pixels of a mapping-keyed deferred
/// window, and therefore how a flush must read them.
///
/// Both kinds live in one map — [`DeviceState::compute_deferred_flush`] — and
/// that is the point. The dangerous half of any deferred rail is the set of
/// guest-page readers that must drain it first; a reader that misses one window
/// makes the guest read stale pixels with nothing logged. Sharing the key type
/// means both kinds share the range scan
/// ([`DeviceState::take_deferred_flush_windows`]), the raw-GVA alias index
/// ([`DeviceState::deferred_alias_pages`]), the teardown drop and every
/// existing trigger, so a rail cannot be covered for one kind and missed for
/// the other. A second map keyed the same way would have had to re-derive
/// "does any window still name this mapping" in
/// [`DeviceState::prune_alias_index`], and getting that wrong drops the alias
/// index out from under a live window.
///
/// What genuinely differs is only where the pixels are. Everything else the
/// flush needs — mapping id, geometry, format, guest byte range — is already in
/// the key, which is why neither variant carries geometry of its own.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DeferredOwner {
    /// Compute rail: a *storage* resident keyed by this same
    /// `ComputeStorageResidencyKey`, read with
    /// `engine::read_resident_storage(key, generation)`. The generation is the
    /// resident's **content** generation, unrelated to `key.map_generation`.
    Storage { generation: u32 },
    /// Type-11 render Store rail: the window **owns the frame it deferred**,
    /// tight BGRA8 at `key.width x key.height`, shared with the
    /// [`crate::runtime::surface_cache`] entry that was stored from the same
    /// readback.
    ///
    /// Owning it is what makes the obligation landable. The flush used to source
    /// its pixels from `surface_cache::get(mapping_id, key.width, key.height)`,
    /// and that cache holds exactly **one** entry per mapping: a later Store at a
    /// different geometry replaces it, and every window still armed at the old
    /// geometry then misses and reports `deferred_flush_lost reason=cache_miss`.
    /// One boot lost 15 whole layers that way — a 1920x1080 desktop surface, a
    /// 1920x24 menu bar, several window-sized rects — which is a compositing
    /// layer rendering solid black with the loss reported only after the fact.
    /// An `Arc` clone costs nothing at arm time and cannot be orphaned.
    Render {
        armed_seq: u64,
        bgra: std::sync::Arc<Vec<u8>>,
    },
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
    ///
    /// Shared rather than owned so a [`DeferredOwner::Render`] window can hold
    /// the exact frame it deferred without copying it: the window and this entry
    /// point at one allocation, and replacing the entry leaves the window's
    /// pixels intact instead of orphaning them.
    pub bgra: std::sync::Arc<Vec<u8>>,
    /// Monotonic host store generation (independent of guest content_generation).
    pub host_gen: u32,
    /// Decoded object type that produced a GVA-keyed type-2/3 encode. Zero for
    /// surface/ref caches and for stores that did not record an owner.
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
    /// guest-page scatter — the oracle frame source. Nothing is owed to the
    /// guest pages by this read: a type-11 Store writes them on its own path
    /// (`mapping_write::write_rgba8_image_changed`).
    Resident,
    /// Contiguous HostOps view read (packed mapping — one host span).
    GuestPagesContig,
    /// Multi-import fragmented guest-page read (the cold ~12 ms/present path).
    GuestPagesFragmented,
}

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
    /// Post-boundary: dual-mid *peer* tracker, read only by the failure/census
    /// lines (`front_wb`, `present_order_hold`) — x86 present often names
    /// ClearOnly mid 2/3 while Stores land on Composite mid 1/4/5, and naming
    /// the peer there is what makes that split visible in a boot log.
    pub early_front_mapping: u32,
    pub early_front_generation: u32,
    /// Present/scanout evidence: mapping → latest geometry it was displayed
    /// at (a `capture_present_frame` action or a retained-frame re-show). The
    /// decoded display transaction naming this surface as plane 0 is the only
    /// thing that writes it, so it separates a scanout buffer from a sampled
    /// sub-surface (a WebKit content tile publishes full frames every paint and
    /// is never presented).
    /// Protocol-structural dense-frame tracking (measure-only, never gates a
    /// present decision): per mapping id, the value of
    /// [`Self::dense_frame_counter`] at the last full-frame (whole-`w`×`h`)
    /// Store **naming that mapping id** — the completeness proof in
    /// [`DeviceState::note_dense_frame_published`], which is the only site that
    /// advances it. Read only by [`DeviceState::note_present_backing`], the
    /// `present_unbacked` gate. Cleared on unmap.
    ///
    /// **What this is keyed on, and what that means it cannot see.** The advance
    /// is a function of the mapping id the Store named and nothing else; it
    /// consults no resident handle. So a full frame the guest sent for a
    /// surface, whose draws were routed to a *different* resident than the one
    /// that surface's present will read, still advances the seq — the gate below
    /// is structurally blind to that. It is also keyed per mapping
    /// id while unified surfaces share ONE resident, so a full frame stored
    /// through one of them does not mark its siblings backed even though they
    /// hold the same pixels.
    pub dense_frame_seq: BTreeMap<u32, u64>,
    /// Per mapping id: the [`Self::dense_frame_seq`] value that mapping held
    /// the last time it was PRESENTED.
    ///
    /// A surface whose seq is unchanged across two of its own presents received
    /// no full-frame Store naming it in between. That is the always-on
    /// `present_unbacked` gate — the loss itself, reported on the mid the guest
    /// named, rather than a rate at which we papered over it. Keyed per mapping
    /// id (not globally) so healthy a/b alternation, where each buffer
    /// legitimately advances on its own turn, stays quiet. Cleared on unmap.
    ///
    /// The "or an inter-buffer seed" half of this condition is gone: `62587b1`
    /// deleted the a/b peer front seed, because unified members share one
    /// resident and a seed between them is a copy onto itself. Nothing else
    /// advances [`Self::dense_frame_counter`].
    pub presented_dense_seq: BTreeMap<u32, u64>,
    /// Monotonic source for [`Self::dense_frame_seq`] (one bump per full-frame
    /// Store). Never reset except on device reset.
    pub dense_frame_counter: u64,
    /// Monotonic present counter, advanced exactly once per present cycle at the
    /// present boundary ([`DeviceState::advance_present_epoch`]). Its only
    /// consumer is the macOS window-publish dedup key, which includes it so that
    /// every present republishes the frame even when the mapping id and resource
    /// generation repeat (an in-place update of the same resident). Never reset
    /// except on device reset.
    pub present_epoch: u64,
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
    /// True when the previous present's window publish handed the window a GPU
    /// resident rather than CPU pixels — the macOS engine-swapchain handoff, which
    /// presents the compositor's resident through the engine's own MoltenVK
    /// swapchain and never reads `frame_bgra`. Set by `publish_window_frame` each
    /// present (same drain worker, one present after the capture reads it; the
    /// handoff is stable across steady-state presents). When true,
    /// `capture_present_frame` skips the expensive guest-page readback.
    ///
    /// Always false where the window owns its own swapchain and uploads CPU pixels
    /// — every non-macOS host — so those keep the per-present readback unchanged.
    pub display_from_resident: bool,
    /// Always-on census: full (readback ran) vs light (resident-carried, readback
    /// skipped) captures, so the readback-elision ratio is visible.
    pub full_captures: u64,
    pub light_captures: u64,
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
/// Counters for the two content memos whose only observable effect is a
/// skipped re-read: the guest-run signature memo (`run_memo_*`) and the
/// zero-copy flush signature memo (`zc_flush_*`).
///
/// These name product behavior, not cost: a memo that stops hitting silently
/// doubles the work per bind, and `stale` is the memo serving bytes the guest
/// has since rewritten. The tests for both paths assert on these.
#[derive(Debug, Default, Clone, Copy)]
pub struct MemoCounters {
    pub run_memo_hit: u64,
    pub run_memo_miss: u64,
    pub run_memo_stale: u64,
    /// Deferred windows a flush-signature check found already landed.
    pub zc_flush_hits: u64,
    /// Binds the signature memo answered without walking the window map.
    pub zc_flush_skip: u64,
    /// Memo answers invalidated by an intervening arm/disarm.
    pub zc_flush_stale: u64,
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
    /// Deferred mapping-keyed writebacks: windows whose guest pages are STALE —
    /// a pinned engine resident is the authoritative content. Every host-side
    /// read or write of intersecting mapping bytes must flush first
    /// (`runtime::storage_flush::flush_intersecting`). The value says which
    /// rail owns the pixels; see [`DeferredOwner`].
    pub compute_deferred_flush: BTreeMap<ComputeStorageResidencyKey, DeferredOwner>,
    /// Arm order for [`DeferredOwner::Render`] windows, so the population cap
    /// can evict oldest-first. Compute windows are bounded by the dispatches
    /// that create them; render windows are armed once per composite Store and
    /// each one pins a display-sized image, so they need their own bound.
    pub surface_deferred_seq: u64,
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
    /// Content-memo hit/miss/stale counters. See [`MemoCounters`].
    pub tranche: MemoCounters,
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
            surface_deferred_seq: 0,
            deferred_alias_pages: DeferredWindows::new(),
            surface_write_kind: BTreeMap::new(),
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
            retired_gva_windows: Vec::new(),
            linear_sampled_memo: LruBytesMemo::new(LINEAR_SAMPLED_MEMO_BYTE_CAP),
            guest_linear_memo: LruBytesMemo::new(GUEST_LINEAR_MEMO_BYTE_CAP),
            guest_linear_gen: 0,
            guest_linear_scratch: Vec::new(),
            type5_view_memo: LruBytesMemo::new(GUEST_LINEAR_MEMO_BYTE_CAP),
            type11_memo: LruBytesMemo::new(GUEST_LINEAR_MEMO_BYTE_CAP),
            type11_memo_scratch: Vec::new(),
            presented_needs_guest_seed: std::collections::BTreeSet::new(),
            gva_host_views: Vec::new(),
            tranche: MemoCounters::default(),
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
        self.linear_deferred_flush
            .0
            .insert(key, (generation, pages));
    }

    /// Disarm a linear compute-storage deferred window, keeping the union index
    /// in sync.
    ///
    /// Returns the page set the window was armed against, so a caller about to
    /// write those guest pages can check they still belong to this window (see
    /// `runtime::storage_flush::deferred_pages_still_ours`). This used to return
    /// a bare `bool` and drop the pages on the floor, which left the flush with
    /// no way to tell that the guest had re-pointed the span since defer time —
    /// the same hazard the GVA rail already guards. `Some` still means "an entry
    /// was present", so the presence test is unchanged for callers that only
    /// want that.
    pub fn disarm_linear_deferred_window(
        &mut self,
        key: &ComputeStorageResidencyKey,
    ) -> Option<std::collections::HashSet<u64>> {
        let (_, pages) = self.linear_deferred_flush.0.remove(key)?;
        self.deferred_ref_sub_pages(&pages);
        Some(pages)
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

    /// A draw Store published a **complete** frame for `mapping_id` into guest
    /// pages (full-frame resident writeback, `import_present ok_runs`).
    ///
    /// Protocol-structural dense marker: this mapping now holds a complete full
    /// frame, so advance its [`PresentState::dense_frame_seq`] off the global
    /// [`PresentState::dense_frame_counter`]. A surface presented twice with no
    /// advance in between received no full frame of its own, which is the
    /// `present_unbacked` gate in [`Self::note_present_backing`] — the only
    /// reader. The counter is monotonic per full-frame Store across all
    /// mappings, so the value is a witness of "something was published for this
    /// mid", never a staleness measure on its own.
    pub fn note_dense_frame_published(&mut self, mapping_id: u32, width: u32, height: u32) {
        if mapping_id == 0 || width == 0 || height == 0 {
            return;
        }
        self.present.dense_frame_counter = self.present.dense_frame_counter.saturating_add(1);
        let seq = self.present.dense_frame_counter;
        self.present.dense_frame_seq.insert(mapping_id, seq);
    }

    /// Advance the per-present epoch counter and return the new value. Call
    /// EXACTLY ONCE per present cycle (see [`PresentState::present_epoch`]).
    pub fn advance_present_epoch(&mut self) -> u64 {
        self.present.present_epoch = self.present.present_epoch.saturating_add(1);
        self.present.present_epoch
    }

    /// Record that `mapping_id` is being presented and report whether the guest
    /// ever sent a full-frame Store **naming it** for what is about to be shown.
    ///
    /// Structural only: decoded Store bookkeeping, never measured content, and
    /// never the resident. Say what that leaves out, because the name reads
    /// broader than the check: a `None` here means the guest sent a frame for
    /// this mid, **not** that the resident this present will read holds it. See
    /// [`PresentState::dense_frame_seq`].
    ///
    /// Records the witness on every call, so a member that stays unbacked
    /// reports once per present rather than once per lifetime — except
    /// [`PresentBacking::NeverStored`], which by construction can only be
    /// reported on a mapping's first present since it was created.
    pub fn note_present_backing(&mut self, mapping_id: u32) -> Option<PresentBacking> {
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
            Some(prev) if prev == seq => Some(PresentBacking::Restaled { seq }),
            // First present since this mapping was created. `dense_frame_seq` is
            // pruned by `forget_compositor_mapping`, so a *re-created* surface
            // arrives here with no witness and no seq — and this arm is the only
            // thing that can see it.
            //
            // It matters because that is the worst version of this class rather
            // than a corner of it: a surface nothing has ever Stored into is
            // uninitialized, so presenting it shows a fully black screen, not a
            // stale one. Measured on a live boot: the guest re-created its
            // scanout surfaces (`gen` reset 82 → 0) and we presented mid 6 at
            // `gen=0` with `px0=[0,0,0,0]` and `rgb_nz=4254` of 2 073 600 — a
            // black screen — for the three presents that followed.
            // `present_unbacked` fired **zero** times during that whole boot.
            //
            // The guest was awake for all of it. An earlier reading of this
            // boot blamed display sleep and it does not survive the log: the
            // 86 s the guest went quiet is bracketed by seven
            // `sync_exec_lock_hold` events of 935-979 ms each, one guest exec
            // packet apiece, on an otherwise idle device. The surface
            // re-creation is downstream of the stall, not of a power
            // transition. What causes the stall is a separate question and is
            // measured by `draw_phase`.
            //
            // The old shape could not have caught it. It compared this present's
            // seq against the previous present's, which is a check for a
            // *repeat* — a transition — while "this surface has never been
            // written" is a *state*. The state was sitting in `dense_frame_seq`
            // the whole time as an absent key.
            None if seq == 0 => Some(PresentBacking::NeverStored),
            _ => None,
        }
    }

    fn forget_compositor_mapping(&mut self, mapping_id: u32) {
        // Prune the dense-frame seq: a recycled mapping id must not inherit a
        // stale predecessor's dense seq.
        self.present.dense_frame_seq.remove(&mapping_id);
        // Same rule for the presented-seq witness: a recycled id must not
        // compare its first present against a predecessor's seq.
        self.present.presented_dense_seq.remove(&mapping_id);
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
        self.retire_task_gva_views(task_id);
        self.tasks[task_id as usize] = TaskEntry::define(length, directory_pfn);
        true
    }

    /// Retire every GVA HostOps view registered under `task_id`, plus the two
    /// memos that carry the same invalidation contract.
    ///
    /// Both entry points that end a task's page table — `define_task` on a
    /// redefine and `delete_task` on teardown — owe exactly this: the views hold
    /// host pointers into pages the guest is about to recycle, so leaving one
    /// live is a read of memory that no longer belongs to the surface (the
    /// WindowServer SIGSEGV class `write_span` documents). `retired_views` is
    /// drained by `mapper::flush_retired_views` through `HostOps::unmap_pages`.
    fn retire_task_gva_views(&mut self, task_id: u32) {
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
    ///
    /// Returns [`WriteGate`] rather than `bool` so the always-on line can name
    /// the arm that decided instead of the caller's assumption about it.
    pub fn gva_write_gate(&self, task_id: u32, gva: u64, len: u64) -> WriteGate {
        if len == 0 {
            return WriteGate::Exact;
        }
        // Spans filed by `task_id` and nothing else. This used to also accept a
        // span filed under `task_id >> 1`, which is the wire-word halving that
        // `runtime::task_slot` already refuted and removed from the command
        // resolvers: `MapMemory2` files spans under slot ids (`0x5`, `0x7`,
        // `0x9`), which the `DefineTask2` wire space does not contain, so
        // halving a slot id names a **different task**. The gate was therefore
        // authorising a write with a different task's map — and the write that
        // followed walked the *named* task's page tables, because
        // `gva_view::resolve_task_for_walk` never halves. Authorisation and
        // destination came from two different address spaces.
        let mut saw_any = false;
        for s in &self.task_map_spans {
            if s.task_id != task_id {
                continue;
            }
            saw_any = true;
            if s.covers(gva, len) {
                return WriteGate::Exact;
            }
        }
        if saw_any {
            WriteGate::Outside
        } else {
            WriteGate::NoSpans
        }
    }

    /// Every task id holding a span that covers `[gva, gva+len)`, ascending and
    /// deduplicated. **Readout only** — [`Self::gva_write_gate`] does not call
    /// this and nothing may branch on it.
    ///
    /// This exists because the gate cannot answer it. The gate considers spans
    /// filed under the writing task and no other, which is the whole point of
    /// it; what a refusal leaves unknown is whether some *other* task declared
    /// this range, and the only way to see that is to look without the filter.
    /// A refused write whose `owners` names one specific other task every time
    /// is a decode question about which key space that opcode's word is in — it
    /// is not licence to write through the named task's page tables anyway,
    /// which is what the removed alias arm did.
    ///
    /// Ambiguity is a property of the registry at the instant of the write, so
    /// it is read from the registry rather than counted from map/unmap events: a
    /// count of registrations would say how many spans were filed, never which
    /// of them authorise this particular range right now.
    pub fn tasks_covering(&self, gva: u64, len: u64) -> Vec<u32> {
        if len == 0 {
            return Vec::new();
        }
        let mut owners: Vec<u32> = self
            .task_map_spans
            .iter()
            .filter(|s| s.covers(gva, len))
            .map(|s| s.task_id)
            .collect();
        owners.sort_unstable();
        owners.dedup();
        owners
    }

    /// How many MapMemory2 spans the registry holds in total, across all tasks.
    ///
    /// Pairs with [`WriteGate::NoSpans`], whose name overstates what it saw: it
    /// means "no span for this task or its one alias", which is not the same as
    /// an empty registry, and only one of those two is "the gate did not run".
    pub fn task_map_span_count(&self) -> usize {
        self.task_map_spans.len()
    }

    /// How many spans `task_id` has filed of its own, ignoring every other task.
    ///
    /// Separates the two things [`WriteGate::Aliased`] currently collapses. If
    /// the writing task has registered nothing, the honest reading is that its
    /// bounds check did not run and a neighbour's span was found by the alias
    /// search — an ordering fact. If it has spans and none covers, the range is
    /// one the guest never mapped for it — a bounds fact. Neither is visible
    /// from the arm alone, and they call for opposite fixes.
    pub fn task_own_span_count(&self, task_id: u32) -> usize {
        self.task_map_spans
            .iter()
            .filter(|s| s.task_id == task_id)
            .count()
    }

    /// Whether the gate permits the write. See [`Self::gva_write_gate`] for why.
    pub fn gva_write_allowed(&self, task_id: u32, gva: u64, len: u64) -> bool {
        self.gva_write_gate(task_id, gva, len) != WriteGate::Outside
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
        self.retire_task_gva_views(task_id);
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
    /// The bump orphans any generation-keyed resident for the mapping.
    pub fn bump_map_generation(e: &mut MappingEntry) {
        e.map_generation = e.map_generation.wrapping_add(1);
        if e.map_generation == 0 {
            e.map_generation = 1;
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

    /// Remove one deferred window by exact key, pruning the alias index with it.
    ///
    /// For supersede: a writer that fully covers a window's guest range drops
    /// the obligation instead of landing it, and must not disturb the
    /// intersecting siblings [`Self::take_deferred_flush_windows`] would also
    /// take. Going through here rather than `compute_deferred_flush.remove`
    /// keeps the raw-GVA alias index in step — a mapping whose last window
    /// leaves must lose its page refs, or the union index keeps counting pages
    /// nothing defers on.
    pub fn take_deferred_flush_window_exact(
        &mut self,
        key: &ComputeStorageResidencyKey,
    ) -> Option<DeferredOwner> {
        let owner = self.compute_deferred_flush.remove(key)?;
        self.prune_alias_index(key.mapping_id);
        Some(owner)
    }

    /// Remove and return every deferred-writeback window intersecting
    /// `[lo, hi)` on this mapping. The caller owns flushing each returned
    /// entry (or reporting the loss) — once taken, the map no longer names it.
    pub fn take_deferred_flush_windows(
        &mut self,
        mapping_id: u32,
        lo: u64,
        hi: u64,
    ) -> Vec<(ComputeStorageResidencyKey, DeferredOwner)> {
        let keys: Vec<ComputeStorageResidencyKey> = self
            .compute_deferred_flush
            .keys()
            .filter(|key| {
                key.mapping_id == mapping_id && key.span_end > lo && key.surface_offset < hi
            })
            .cloned()
            .collect();
        let taken: Vec<(ComputeStorageResidencyKey, DeferredOwner)> = keys
            .into_iter()
            .filter_map(|key| {
                self.compute_deferred_flush
                    .remove(&key)
                    .map(|owner| (key, owner))
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

    /// Drop the alias-index entry once no mapping-keyed deferred window names
    /// this mapping anymore.
    fn prune_alias_index(&mut self, mapping_id: u32) {
        let live = self
            .compute_deferred_flush
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
        Self::bump_map_generation(e);
        let retired = Self::take_mapping_view(e);
        if let Some(v) = retired {
            self.retired_views.push(v);
        }
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
        // Fresh MAP: prior host-cache for this surface_id is stale, and so is
        // any present evidence — the slot may hold a NEW surface.
        self.host_surfaces.remove(&mapping_id);
        // Present evidence is stamped with the incarnation and deliberately NOT
        // dropped here. A fresh MAP does not yet know whether this is a new
        // surface — that is what the fingerprint compare decides, bumping the
        // generation when it is. Dropping it eagerly demoted a proven swapchain
        // buffer to a private resident for every draw until its next present,
        // which is the black-desktop class.
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
            Self::bump_map_generation(e);
            e.has_geom = false;
            e.width = 0;
            e.height = 0;
            e.format = 0;
            let retired = Self::take_mapping_view(e);
            if let Some(v) = retired {
                self.retired_views.push(v);
            }
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
        Self::bump_map_generation(e);
        // New MappingInternal ⇒ new surface; force device-desc re-resolve.
        e.has_geom = false;
        e.width = 0;
        e.height = 0;
        e.format = 0;
        let retired = Self::take_mapping_view(e);
        if let Some(v) = retired {
            self.retired_views.push(v);
        }
        // New MappingInternal ⇒ new surface, and the `bump_map_generation`
        // above is what retires the stale present evidence: it is stamped with
        // the incarnation that recorded it, so the recycled slot cannot inherit
        // a display-plane qualification it did not earn.
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
        // reset content_generation (the guest pages stay authoritative).
        if e.width != width || e.height != height {
            e.content_generation = 0;
        }
        e.has_geom = true;
        e.width = width;
        e.height = height;
        e.format = format;
        true
    }

    /// Bump content generation after a write into the mapping (0 never skips).
    pub fn mark_mapping_written(&mut self, mapping_id: u32) -> u32 {
        let Some(m) = self.mappings.get_mut(&mapping_id) else {
            return 0;
        };
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

#[cfg(test)]
mod fail_vocabulary_tests {
    use super::*;
    use crate::observe::Decline;

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
        let mut slugs = std::collections::HashSet::new();
        for decline in declines {
            assert!(slugs.insert(decline.slug()), "duplicate {}", decline.slug());
        }
        assert_eq!(
            slugs.len(),
            17,
            "every state mutation check has its own slug"
        );
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
