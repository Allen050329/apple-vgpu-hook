//! Product-path MTLIndirectCommandBuffer materialization + compute command fills.
//!
//! ## Create (wire)
//!
//! Guest create is serialized by
//! `PGSerializer newIndirectCommandBufferWithDescriptor:layout:maxCommandCount:options:allocator:`
//! into an 88-byte type-7 body (tag `0x36`) including a 52-byte command
//! **layout** at `+0x1c`. Product materializes a host ICB and caches it per
//! `(task_id, icb_ref)`.
//!
//! ## Command fills — buffer-backed, not stream opcodes
//!
//! RE of `AppleParavirtGPUMetalIOGPUFamily` (2026-07-11): there is **no** Reims VGPU
//! compute-stream opcode for `indirectComputeCommandAtIndex` fills. Guest CPU
//! writes into an ICB backing buffer via
//! `PGSerializerIndirectComputeCommand` (setPSO / setKernelBuffer /
//! concurrentDispatch*). Command slots use the layout from create:
//! - type `0x20` = concurrentDispatchThreadgroups, `0x40` = …Threads
//! - pipeline object-list ref at `pipelineStateOffset`
//! - kernel binds at `kernelBufferBindOffset` (0x14 B: ref@0, va@4, gpuva@0xc)
//! - dispatch args at `commandArgumentsOffset` (3×u64 grid + 3×u64 tptg)
//!
//! Product decode of that buffer → [`fill_compute_command`]. Execute (`0xe4`/
//! `0xe5`) re-fills from registered command memory when present. Host Metal
//! fill API remains for tests without guest backing.

use crate::contract::endian::{ld32, ld64}; // ld64: 0x1d1 gpu_address + dispatch args
use crate::model::DeviceState;
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
use crate::runtime::decode::resource::TYPE7_OBJECT_ICB;
use crate::runtime::decode::resource::{
    decode_type7_descriptor, icb_layout_attribute_stride_slot_count,
    icb_layout_kernel_tg_slot_count, Descriptor as ResourceDescriptor, IcbCommandLayout,
    IndirectCommandBufferDescriptor, ICB_ATTRIBUTE_STRIDE_ENTRY_SIZE, ICB_BUFFER_BIND_STRIDE,
    ICB_CMD_TYPE_CONCURRENT_DISPATCH_THREADGROUPS, ICB_CMD_TYPE_CONCURRENT_DISPATCH_THREADS,
    ICB_CMD_TYPE_DRAW, ICB_CMD_TYPE_DRAW_INDEXED, ICB_CMD_TYPE_DRAW_INDEXED_PATCHES,
    ICB_CMD_TYPE_DRAW_MESH_THREADGROUPS, ICB_CMD_TYPE_DRAW_MESH_THREADS, ICB_CMD_TYPE_DRAW_PATCHES,
    ICB_CONCURRENT_DISPATCH_ARGS_LEN, ICB_DRAW_INDEXED_PATCHES_ARGS_LEN, ICB_DRAW_MESH_ARGS_LEN,
    ICB_DRAW_PATCHES_ARGS_LEN, ICB_TESSELLATION_FACTOR_LEN, ICB_TG_MEMORY_STRIDE,
    OBJECT_TYPE_TYPE7,
}; // ICB_TG_MEMORY_STRIDE: object + kernel TG length tables
#[cfg(test)]
use crate::runtime::decode::resource::{
    MTL_INDIRECT_CMD_DRAW, MTL_INDIRECT_CMD_DRAW_INDEXED_PATCHES, MTL_INDIRECT_CMD_DRAW_PATCHES,
}; // slot-encoder fixtures only
use crate::runtime::host::{HostMemory, HostOps};
use crate::runtime::objects;
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
use std::collections::HashMap;
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
use std::sync::OnceLock;

/// A refusal on the indirect-command-buffer rail.
///
/// Every variant carries the registered slug naming **which** check refused.
/// Before that payload existed the five variants spoke for 153 checks — `Args`
/// alone for 84 — so a guest whose ICB never executed produced a log line
/// indistinguishable from thirty other bugs. The variant is the class; the slug
/// is the check.
///
/// There is deliberately no `Ok`: every function on this rail returns
/// `Result<_, IcbStatus>`, so success is `Ok(..)` and this type is *always* a
/// refusal. The old `Ok` variant was never constructed anywhere in the crate —
/// it survived only as an unreachable `Err(IcbStatus::Ok)` match arm — and
/// keeping it would have forced [`crate::observe::Refusal`] where
/// [`crate::observe::Decline`] is the honest shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IcbStatus {
    /// A guest object, a cache entry, or a required buffer is not there.
    Missing(&'static str),
    /// A descriptor was found but is the wrong type or does not decode.
    BadDescriptor(&'static str),
    /// A host Metal call failed.
    MetalFailed(&'static str),
    /// No Metal device on this pathway — the Vulkan build's stubs, and the
    /// `system_device()` miss.
    NoMetal(&'static str),
    /// The decoded arguments do not satisfy the contract: a span past the end,
    /// a zero count, an unknown wire tag.
    Args(&'static str),
}

impl crate::observe::Decline for IcbStatus {
    fn slug(&self) -> &'static str {
        match self {
            Self::Missing(s)
            | Self::BadDescriptor(s)
            | Self::MetalFailed(s)
            | Self::NoMetal(s)
            | Self::Args(s) => s,
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        vec![(
            "class",
            match self {
                Self::Missing(_) => "missing",
                Self::BadDescriptor(_) => "bad_descriptor",
                Self::MetalFailed(_) => "metal_failed",
                Self::NoMetal(_) => "no_metal",
                Self::Args(_) => "args",
            }
            .to_string(),
        )]
    }
}

/// Carry an ICB refusal onto the compute rail without losing its name.
///
/// The compute rail is where an ICB refusal actually reaches the sink: the
/// session hands a [`crate::runtime::compute_exec::ComputeStatus`] back to
/// `exec.rs`, which logs it as `compute_record reason=<slug>`. Before this
/// conversion existed each boundary invented its own coarse literal —
/// `icb_resolve_bad_descriptor_or_args` spoke for `BadDescriptor` *and* `Args`,
/// i.e. for 93 of this file's checks at once — so the reason died one frame
/// short of the log. Forwarding `self.slug()` means the check that refused is
/// the reason that prints.
impl From<IcbStatus> for crate::runtime::compute_exec::ComputeStatus {
    fn from(e: IcbStatus) -> Self {
        use crate::observe::Decline;
        let slug = e.slug();
        match e {
            IcbStatus::Missing(_) => Self::MissingBuffer(slug),
            IcbStatus::BadDescriptor(_) | IcbStatus::Args(_) => Self::Unsupported(slug),
            IcbStatus::MetalFailed(_) => Self::MetalFailed(slug),
            IcbStatus::NoMetal(_) => Self::NoMetal(slug),
        }
    }
}

/// One kernel-buffer bind for an ICB compute command fill (Metal setKernelBuffer).
#[derive(Clone, Debug, Default)]
pub struct IcbKernelBufferBind {
    pub index: u32,
    pub buffer_ref: u32,
    /// Byte offset into the type-1 buffer (host fill API, or resolved from [`Self::wire_va`]).
    pub offset: u64,
    /// Absolute guest VA from bind record `va@+4` (PGSerializer: base+offset).
    /// `0` means host-only fill / ref-at-base. Resolved to [`Self::offset`] before stage.
    pub wire_va: u64,
    /// Dynamic attribute stride (`setKernelBuffer:offset:attributeStride:atIndex:`).
    /// Wire: u64 at `attributeStrideOffset + index*8`; 0 = no stride API / default.
    pub attribute_stride: u64,
    pub has_attribute_stride: bool,
}

/// Dispatch form recorded into an ICB compute command (Metal concurrent* only).
#[derive(Clone, Copy, Debug)]
pub enum IcbFillDispatch {
    /// `concurrentDispatchThreadgroups:threadsPerThreadgroup:`
    ConcurrentThreadgroups {
        grid_x: u32,
        grid_y: u32,
        grid_z: u32,
        tg_x: u32,
        tg_y: u32,
        tg_z: u32,
    },
    /// `concurrentDispatchThreads:threadsPerThreadgroup:`
    ConcurrentThreads {
        threads_x: u32,
        threads_y: u32,
        threads_z: u32,
        tg_x: u32,
        tg_y: u32,
        tg_z: u32,
    },
}

/// One kernel-threadgroup-memory length for an ICB compute command fill.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct IcbThreadgroupMemory {
    pub index: u32,
    /// Byte length (`setThreadgroupMemoryLength:atIndex:`); 0 clears the slot.
    pub length: u64,
}

/// Arguments for product-path fill of one compute command slot.
#[derive(Clone, Debug)]
pub struct IcbComputeFill {
    pub command_index: u32,
    /// Type-7 compute pipeline object-list ref (kernel function + optional stage-in).
    pub pipeline_ref: u32,
    pub buffers: Vec<IcbKernelBufferBind>,
    /// `setThreadgroupMemoryLength:atIndex:` entries (wire: u64 lengths table).
    pub threadgroup_memory: Vec<IcbThreadgroupMemory>,
    /// `setBarrier` when true, `clearBarrier` when false (wire: u32 at barrierOffset).
    pub barrier: bool,
    pub dispatch: IcbFillDispatch,
}

/// Stage for a render ICB buffer bind (layout table + Metal set*Buffer API).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum IcbRenderBindStage {
    #[default]
    Vertex,
    Fragment,
    /// Object-shader stage (`setObjectBuffer`); wire at `objectBufferBindOffset`.
    Object,
    /// Mesh-shader stage (`setMeshBuffer`); wire at `meshBufferBindOffset`.
    Mesh,
}

/// One buffer bind for a render ICB command fill.
#[derive(Clone, Debug, Default)]
pub struct IcbRenderBufferBind {
    pub index: u32,
    pub buffer_ref: u32,
    /// Byte offset into the type-1 buffer (host fill API, or resolved from [`Self::wire_va`]).
    pub offset: u64,
    /// Absolute guest VA from bind record `va@+4` (PGSerializer: base+offset).
    /// `0` means host-only fill / ref-at-base. Resolved to [`Self::offset`] before stage.
    pub wire_va: u64,
    /// Dynamic attribute stride for vertex binds
    /// (`setVertexBuffer:offset:attributeStride:atIndex:`). Wire: u64 at
    /// `attributeStrideOffset + index*8`. Non-vertex stages ignore this field.
    pub attribute_stride: u64,
    pub has_attribute_stride: bool,
    /// Legacy convenience: `true` means [`IcbRenderBindStage::Fragment`].
    /// Prefer [`Self::stage`]; when `stage` is default Vertex and this is true,
    /// treat as Fragment (older call sites).
    pub is_fragment: bool,
    /// Bind stage (vertex / fragment / object / mesh). When `Object` or `Mesh`,
    /// overrides `is_fragment`.
    pub stage: IcbRenderBindStage,
}

impl IcbRenderBufferBind {
    /// Effective stage after reconciling `stage` and legacy `is_fragment`.
    pub fn effective_stage(&self) -> IcbRenderBindStage {
        match self.stage {
            IcbRenderBindStage::Object | IcbRenderBindStage::Mesh => self.stage,
            IcbRenderBindStage::Fragment => IcbRenderBindStage::Fragment,
            IcbRenderBindStage::Vertex if self.is_fragment => IcbRenderBindStage::Fragment,
            IcbRenderBindStage::Vertex => IcbRenderBindStage::Vertex,
        }
    }
}

/// Tessellation-factor buffer recorded at layout `tessellationFactorOffset`.
/// Host RE 2026-07-12: u32 ref@0 · u64 va@4 · u64 gpuva@0xc · u64 instanceStride@0x14.
#[derive(Clone, Copy, Debug, Default)]
pub struct IcbTessellationFactor {
    pub buffer_ref: u32,
    pub offset: u64,
    pub wire_va: u64,
    pub instance_stride: u64,
}

/// Draw form recorded into a render ICB command.
#[derive(Clone, Copy, Debug)]
pub enum IcbRenderDraw {
    /// command type `0x1` — drawPrimitives
    Primitives {
        primitive_type: u16,
        vertex_start: u64,
        vertex_count: u64,
        instance_count: u64,
        base_instance: u64,
    },
    /// command type `0x2` — drawIndexedPrimitives (PGSerializer layout).
    Indexed {
        primitive_type: u16,
        /// MTLIndexType (UInt16=0, UInt32=1).
        index_type: u16,
        index_buffer_ref: u32,
        index_count: u64,
        /// Byte offset into the index type-1 buffer (host fill or resolved from wire VA).
        index_buffer_offset: u64,
        /// Absolute guest VA of the index range (`va@+0x10` in DrawIndexed args); `0` = base.
        index_wire_va: u64,
        instance_count: u64,
        base_vertex: i64,
        base_instance: u64,
    },
    /// command type `0x4` — drawPatches (host RE PGSerializerIndirectRenderCommand).
    Patches {
        number_of_patch_control_points: u16,
        patch_start: u64,
        patch_count: u64,
        /// Optional patch-index buffer object-list ref (`0` = none / null Metal buffer).
        patch_index_buffer_ref: u32,
        patch_index_buffer_offset: u64,
        patch_index_wire_va: u64,
        instance_count: u64,
        base_instance: u64,
        tessellation_factor: IcbTessellationFactor,
    },
    /// command type `0x8` — drawIndexedPatches.
    IndexedPatches {
        number_of_patch_control_points: u16,
        patch_start: u64,
        patch_count: u64,
        patch_index_buffer_ref: u32,
        patch_index_buffer_offset: u64,
        patch_index_wire_va: u64,
        control_point_index_buffer_ref: u32,
        control_point_index_buffer_offset: u64,
        control_point_index_wire_va: u64,
        instance_count: u64,
        base_instance: u64,
        tessellation_factor: IcbTessellationFactor,
    },
    /// command type `0x100` — drawMeshThreads. `grid` is threadsPerGrid.
    MeshThreads(IcbMeshDraw),
    /// command type `0x80` — drawMeshThreadgroups. `grid` is
    /// threadgroupsPerGrid.
    MeshThreadgroups(IcbMeshDraw),
}

/// The record both mesh draw commands serialize.
///
/// Wire: three MTLSize as 3×u64 each at `commandArgumentsOffset`, total
/// [`ICB_DRAW_MESH_ARGS_LEN`] (`0x48` from host `setupCommandLayout`). Field
/// order matches Metal SPI `MTLIndirectDrawMesh*Arguments` — grid @0, object TG
/// @0x18, mesh TG @0x30. Fill IMPs are stubs; layout is RE'd from
/// setupCommandLayout + concurrent-dispatch packing + SPI field order.
///
/// `drawMeshThreads` (`0x100`) and `drawMeshThreadgroups` (`0x80`) write byte-
/// identical records; the only difference is what the first MTLSize counts,
/// which the two [`IcbRenderDraw`] variants carry. One record, two meanings.
#[derive(Clone, Copy, Debug)]
pub struct IcbMeshDraw {
    /// threadsPerGrid or threadgroupsPerGrid, per the owning variant.
    pub grid: [u32; 3],
    pub object_tg: [u32; 3],
    pub mesh_tg: [u32; 3],
}

impl IcbMeshDraw {
    /// Read the nine u64 dimensions at `args` within `slot`. The caller has
    /// already proved `args + ICB_DRAW_MESH_ARGS_LEN <= slot.len()`.
    fn decode(slot: &[u8], args: usize) -> Self {
        let at = |off: usize| ld64(&slot[args + off..]) as u32;
        Self {
            grid: [at(0), at(8), at(0x10)],
            object_tg: [at(0x18), at(0x20), at(0x28)],
            mesh_tg: [at(0x30), at(0x38), at(0x40)],
        }
    }

    /// Write the nine dimensions at `args` within `slot`. The caller has
    /// already proved `args + ICB_DRAW_MESH_ARGS_LEN <= size`. Test-only, like
    /// its only caller [`encode_render_command_slot`].
    #[cfg(test)]
    fn encode(&self, slot: &mut [u8], args: usize) {
        use crate::contract::endian::st64;
        for (i, v) in self
            .grid
            .iter()
            .chain(&self.object_tg)
            .chain(&self.mesh_tg)
            .enumerate()
        {
            st64(&mut slot[args + i * 8..], u64::from(*v));
        }
    }
}

/// Arguments for product-path fill of one render command slot.
#[derive(Clone, Debug)]
pub struct IcbRenderFill {
    pub command_index: u32,
    pub pipeline_ref: u32,
    pub buffers: Vec<IcbRenderBufferBind>,
    /// Object-stage TG memory lengths (`setObjectThreadgroupMemoryLength:atIndex:`).
    /// Wire: u64 table at `objectThreadgroupMemoryLengthOffset + index*8`.
    pub object_threadgroup_memory: Vec<IcbThreadgroupMemory>,
    pub draw: IcbRenderDraw,
}

/// Guest ICB command-memory association (backing buffer for CPU fills).
#[derive(Clone, Copy, Debug)]
pub struct IcbCommandMemory {
    pub gva: u64,
    pub byte_len: u64,
}

/// Decode one filled compute command slot from ICB backing bytes.
///
/// Returns `None` if the slot is empty/reset (command type 0).
pub fn decode_compute_command_slot(
    layout: &IcbCommandLayout,
    slot: &[u8],
    max_kernel_binds: u16,
) -> Result<Option<IcbComputeFill>, IcbStatus> {
    let cmd_size = layout.command_size as usize;
    if cmd_size == 0 || slot.len() < cmd_size {
        return Err(IcbStatus::Args("icb_dcs_slot_short"));
    }
    let type_off = layout.command_type_offset as usize;
    if type_off + 4 > slot.len() {
        return Err(IcbStatus::Args("icb_dcs_type_offset_oob"));
    }
    let cmd_type = ld32(&slot[type_off..]);
    if cmd_type == 0 {
        return Ok(None);
    }
    let dispatch = match cmd_type {
        ICB_CMD_TYPE_CONCURRENT_DISPATCH_THREADGROUPS
        | ICB_CMD_TYPE_CONCURRENT_DISPATCH_THREADS => {
            // Both commands serialize the same two MTLSize (grid, threadgroup)
            // as 6xu64; only what the first counts differs.
            let threadgroups = cmd_type == ICB_CMD_TYPE_CONCURRENT_DISPATCH_THREADGROUPS;
            let args = layout.command_arguments_offset as usize;
            if args + ICB_CONCURRENT_DISPATCH_ARGS_LEN > slot.len() {
                return Err(IcbStatus::Args(if threadgroups {
                    "icb_dcs_tg_args_oob"
                } else {
                    "icb_dcs_threads_args_oob"
                }));
            }
            let d = |off: usize| ld64(&slot[args + off..]) as u32;
            let (x, y, z, tg_x, tg_y, tg_z) = (d(0), d(8), d(16), d(24), d(32), d(40));
            if threadgroups {
                IcbFillDispatch::ConcurrentThreadgroups {
                    grid_x: x,
                    grid_y: y,
                    grid_z: z,
                    tg_x,
                    tg_y,
                    tg_z,
                }
            } else {
                IcbFillDispatch::ConcurrentThreads {
                    threads_x: x,
                    threads_y: y,
                    threads_z: z,
                    tg_x,
                    tg_y,
                    tg_z,
                }
            }
        }
        _ => return Err(IcbStatus::Args("icb_dcs_unknown_command_type")),
    };

    // Pipeline ref may be 0 when the ICB was created with inheritPipelineState —
    // PSO then comes from the parent compute encoder at execute (Metal contract).
    let mut pipeline_ref = 0u32;
    if layout.pipeline_state_offset != 0 {
        let off = layout.pipeline_state_offset as usize;
        if off + 4 > slot.len() {
            return Err(IcbStatus::Args("icb_dcs_pipeline_offset_oob"));
        }
        pipeline_ref = ld32(&slot[off..]);
    }

    let mut buffers = Vec::new();
    if layout.kernel_buffer_bind_offset != 0 && max_kernel_binds > 0 {
        let base = layout.kernel_buffer_bind_offset as usize;
        for i in 0..max_kernel_binds as usize {
            let off = base + i * ICB_BUFFER_BIND_STRIDE;
            if off + 4 > slot.len() {
                break;
            }
            let buffer_ref = ld32(&slot[off..]);
            if buffer_ref == 0 {
                continue;
            }
            // Bind record 0x14 B: ref@0, va@4, gpuva@0xc. Offset is not a
            // separate field — guest writes base+offset into va (resolved later).
            let wire_va = if off + 0xc <= slot.len() {
                ld64(&slot[off + 4..])
            } else {
                0
            };
            // Attribute stride table (separate from bind record).
            let (attribute_stride, has_attribute_stride) =
                read_attribute_stride(layout, slot, i as u32);
            buffers.push(IcbKernelBufferBind {
                index: i as u32,
                buffer_ref,
                offset: 0,
                wire_va,
                attribute_stride,
                has_attribute_stride,
            });
        }
    }

    // Barrier: u32 at barrierOffset (setBarrier writes 1, clearBarrier writes 0).
    // Host RE 2026-07-12 PGSerializerIndirectComputeCommand setBarrier/clearBarrier.
    let mut barrier = false;
    if layout.barrier_offset != 0 {
        let bo = layout.barrier_offset as usize;
        if bo + 4 <= slot.len() {
            barrier = ld32(&slot[bo..]) != 0;
        }
    }

    // Threadgroup memory: u64 length table at threadgroupMemoryLengthOffset,
    // entry i at + i*8 (setThreadgroupMemoryLength:atIndex:).
    let mut threadgroup_memory = Vec::new();
    let tg_slots = icb_layout_kernel_tg_slot_count(layout);
    if tg_slots > 0 && layout.threadgroup_memory_length_offset != 0 {
        let base = layout.threadgroup_memory_length_offset as usize;
        for i in 0..tg_slots as usize {
            let off = base + i * ICB_TG_MEMORY_STRIDE;
            if off + 8 > slot.len() {
                break;
            }
            let length = ld64(&slot[off..]);
            if length != 0 {
                threadgroup_memory.push(IcbThreadgroupMemory {
                    index: i as u32,
                    length,
                });
            }
        }
    }

    Ok(Some(IcbComputeFill {
        command_index: 0, // caller sets index
        pipeline_ref,
        buffers,
        threadgroup_memory,
        barrier,
        dispatch,
    }))
}

/// Decode one filled **render** command slot (Draw / DrawIndexed) from ICB backing.
pub fn decode_render_command_slot(
    layout: &IcbCommandLayout,
    slot: &[u8],
    max_vertex_binds: u16,
    max_fragment_binds: u16,
) -> Result<Option<IcbRenderFill>, IcbStatus> {
    let cmd_size = layout.command_size as usize;
    if cmd_size == 0 || slot.len() < cmd_size {
        return Err(IcbStatus::Args("icb_drs_slot_short"));
    }
    let type_off = layout.command_type_offset as usize;
    if type_off + 4 > slot.len() {
        return Err(IcbStatus::Args("icb_drs_type_offset_oob"));
    }
    let cmd_type = ld32(&slot[type_off..]);
    if cmd_type == 0 {
        return Ok(None);
    }

    let mut pipeline_ref = 0u32;
    if layout.pipeline_state_offset != 0 {
        let off = layout.pipeline_state_offset as usize;
        if off + 4 > slot.len() {
            return Err(IcbStatus::Args("icb_drs_pipeline_offset_oob"));
        }
        pipeline_ref = ld32(&slot[off..]);
    }
    if pipeline_ref == 0 {
        return Err(IcbStatus::Missing("icb_drs_pipeline_ref_zero"));
    }

    let tessellation_factor = read_tessellation_factor(layout, slot);

    let mut buffers = Vec::new();
    let push_binds = |buffers: &mut Vec<IcbRenderBufferBind>,
                      base_off: u32,
                      count: u16,
                      stage: IcbRenderBindStage| {
        if base_off == 0 || count == 0 {
            return;
        }
        let base = base_off as usize;
        for i in 0..count as usize {
            let off = base + i * ICB_BUFFER_BIND_STRIDE;
            if off + 4 > slot.len() {
                break;
            }
            let buffer_ref = ld32(&slot[off..]);
            if buffer_ref == 0 {
                continue;
            }
            let wire_va = if off + 0xc <= slot.len() {
                ld64(&slot[off + 4..])
            } else {
                0
            };
            let (attribute_stride, has_attribute_stride) = if stage == IcbRenderBindStage::Vertex {
                read_attribute_stride(layout, slot, i as u32)
            } else {
                (0, false)
            };
            buffers.push(IcbRenderBufferBind {
                index: i as u32,
                buffer_ref,
                offset: 0,
                wire_va,
                attribute_stride,
                has_attribute_stride,
                is_fragment: stage == IcbRenderBindStage::Fragment,
                stage,
            });
        }
    };
    push_binds(
        &mut buffers,
        layout.vertex_buffer_bind_offset,
        max_vertex_binds,
        IcbRenderBindStage::Vertex,
    );
    push_binds(
        &mut buffers,
        layout.fragment_buffer_bind_offset,
        max_fragment_binds,
        IcbRenderBindStage::Fragment,
    );
    // Object/mesh bind table sizes from layout offsets (setupCommandLayout order).
    let max_object = icb_layout_stage_bind_count(
        layout.object_buffer_bind_offset,
        layout.mesh_buffer_bind_offset,
    );
    let max_mesh = icb_layout_stage_bind_count(
        layout.mesh_buffer_bind_offset,
        layout.kernel_buffer_bind_offset,
    );
    push_binds(
        &mut buffers,
        layout.object_buffer_bind_offset,
        max_object,
        IcbRenderBindStage::Object,
    );
    push_binds(
        &mut buffers,
        layout.mesh_buffer_bind_offset,
        max_mesh,
        IcbRenderBindStage::Mesh,
    );

    let args = layout.command_arguments_offset as usize;
    let draw = match cmd_type {
        ICB_CMD_TYPE_DRAW => {
            // Pack: u16 prim @0, u64 vertexStart @2, u64 vertexCount @0xa,
            // u64 instanceCount @0x12, u64 baseInstance @0x1a.
            if args + 0x22 > slot.len() {
                return Err(IcbStatus::Args("icb_drs_draw_args_oob"));
            }
            let prim = u16::from_le_bytes([slot[args], slot[args + 1]]);
            IcbRenderDraw::Primitives {
                primitive_type: prim,
                vertex_start: ld64(&slot[args + 2..]),
                vertex_count: ld64(&slot[args + 0xa..]),
                instance_count: ld64(&slot[args + 0x12..]),
                base_instance: ld64(&slot[args + 0x1a..]),
            }
        }
        ICB_CMD_TYPE_DRAW_INDEXED => {
            // DrawIndexed (PGSerializer): u16 prim @0, u16 indexType @2,
            // u32 indexBufferRef @4, u64 indexCount @8, u64 va @0x10, u64 gpuva @0x18,
            // u64 instanceCount @0x20, u64 baseVertex @0x28 (signed bit pattern),
            // u64 baseInstance @0x30.
            if args + 0x38 > slot.len() {
                return Err(IcbStatus::Args("icb_drs_indexed_args_oob"));
            }
            let prim = u16::from_le_bytes([slot[args], slot[args + 1]]);
            let index_type = u16::from_le_bytes([slot[args + 2], slot[args + 3]]);
            let index_buffer_ref = ld32(&slot[args + 4..]);
            if index_buffer_ref == 0 {
                return Err(IcbStatus::Missing("icb_drs_index_buffer_ref_zero"));
            }
            let index_wire_va = ld64(&slot[args + 0x10..]);
            IcbRenderDraw::Indexed {
                primitive_type: prim,
                index_type,
                index_buffer_ref,
                index_count: ld64(&slot[args + 8..]),
                index_buffer_offset: 0, // resolved from index_wire_va when non-zero
                index_wire_va,
                instance_count: ld64(&slot[args + 0x20..]),
                base_vertex: ld64(&slot[args + 0x28..]) as i64,
                base_instance: ld64(&slot[args + 0x30..]),
            }
        }
        ICB_CMD_TYPE_DRAW_PATCHES => {
            // Host RE: u16 controlPoints@0, u64 patchStart@2, u64 patchCount@0xa,
            // u32 patchIndexRef@0x12, u64 va@0x16, u64 gpuva@0x1e,
            // u64 instanceCount@0x26, u64 baseInstance@0x2e.
            if args + ICB_DRAW_PATCHES_ARGS_LEN as usize > slot.len() {
                return Err(IcbStatus::Args("icb_drs_patches_args_oob"));
            }
            let cps = u16::from_le_bytes([slot[args], slot[args + 1]]);
            let patch_index_buffer_ref = ld32(&slot[args + 0x12..]);
            let patch_index_wire_va = ld64(&slot[args + 0x16..]);
            IcbRenderDraw::Patches {
                number_of_patch_control_points: cps,
                patch_start: ld64(&slot[args + 2..]),
                patch_count: ld64(&slot[args + 0xa..]),
                patch_index_buffer_ref,
                patch_index_buffer_offset: 0,
                patch_index_wire_va,
                instance_count: ld64(&slot[args + 0x26..]),
                base_instance: ld64(&slot[args + 0x2e..]),
                tessellation_factor,
            }
        }
        ICB_CMD_TYPE_DRAW_INDEXED_PATCHES => {
            // Host RE: like DrawPatches through patchIndex, then
            // u32 controlPointIndexRef@0x26, u64 va@0x2a, u64 gpuva@0x32,
            // u64 instanceCount@0x3a, u64 baseInstance@0x42.
            if args + ICB_DRAW_INDEXED_PATCHES_ARGS_LEN as usize > slot.len() {
                return Err(IcbStatus::Args("icb_drs_indexed_patches_args_oob"));
            }
            let cps = u16::from_le_bytes([slot[args], slot[args + 1]]);
            let patch_index_buffer_ref = ld32(&slot[args + 0x12..]);
            let patch_index_wire_va = ld64(&slot[args + 0x16..]);
            let control_point_index_buffer_ref = ld32(&slot[args + 0x26..]);
            if control_point_index_buffer_ref == 0 {
                return Err(IcbStatus::Missing("icb_drs_control_point_ref_zero"));
            }
            let control_point_index_wire_va = ld64(&slot[args + 0x2a..]);
            IcbRenderDraw::IndexedPatches {
                number_of_patch_control_points: cps,
                patch_start: ld64(&slot[args + 2..]),
                patch_count: ld64(&slot[args + 0xa..]),
                patch_index_buffer_ref,
                patch_index_buffer_offset: 0,
                patch_index_wire_va,
                control_point_index_buffer_ref,
                control_point_index_buffer_offset: 0,
                control_point_index_wire_va,
                instance_count: ld64(&slot[args + 0x3a..]),
                base_instance: ld64(&slot[args + 0x42..]),
                tessellation_factor,
            }
        }
        ICB_CMD_TYPE_DRAW_MESH_THREADS | ICB_CMD_TYPE_DRAW_MESH_THREADGROUPS => {
            let threads = cmd_type == ICB_CMD_TYPE_DRAW_MESH_THREADS;
            if args + ICB_DRAW_MESH_ARGS_LEN as usize > slot.len() {
                return Err(IcbStatus::Args(if threads {
                    "icb_drs_mesh_threads_args_oob"
                } else {
                    "icb_drs_mesh_threadgroups_args_oob"
                }));
            }
            let mesh = IcbMeshDraw::decode(slot, args);
            if threads {
                IcbRenderDraw::MeshThreads(mesh)
            } else {
                IcbRenderDraw::MeshThreadgroups(mesh)
            }
        }
        _ => return Err(IcbStatus::Args("icb_drs_unknown_command_type")),
    };

    // Object TG memory length table (setupCommandLayout: before kernel TG).
    let mut object_threadgroup_memory = Vec::new();
    let obj_tg_slots = icb_layout_object_tg_slot_count(layout);
    if obj_tg_slots > 0 && layout.object_threadgroup_memory_length_offset != 0 {
        let base = layout.object_threadgroup_memory_length_offset as usize;
        for i in 0..obj_tg_slots as usize {
            let off = base + i * ICB_TG_MEMORY_STRIDE;
            if off + 8 > slot.len() {
                break;
            }
            let length = ld64(&slot[off..]);
            if length != 0 {
                object_threadgroup_memory.push(IcbThreadgroupMemory {
                    index: i as u32,
                    length,
                });
            }
        }
    }

    Ok(Some(IcbRenderFill {
        command_index: 0,
        pipeline_ref,
        buffers,
        object_threadgroup_memory,
        draw,
    }))
}

/// Object-TG length table slot count between layout offsets.
fn icb_layout_object_tg_slot_count(layout: &IcbCommandLayout) -> u16 {
    let start = layout.object_threadgroup_memory_length_offset;
    let end = layout.threadgroup_memory_length_offset;
    if end <= start {
        return 0;
    }
    ((end - start) / ICB_TG_MEMORY_STRIDE as u32) as u16
}

/// Read tessellation-factor table at `tessellationFactorOffset` (host RE).
fn read_tessellation_factor(layout: &IcbCommandLayout, slot: &[u8]) -> IcbTessellationFactor {
    if layout.tessellation_factor_offset == 0 {
        return IcbTessellationFactor::default();
    }
    let off = layout.tessellation_factor_offset as usize;
    if off + ICB_TESSELLATION_FACTOR_LEN > slot.len() {
        return IcbTessellationFactor::default();
    }
    IcbTessellationFactor {
        buffer_ref: ld32(&slot[off..]),
        wire_va: ld64(&slot[off + 4..]),
        offset: 0,
        instance_stride: ld64(&slot[off + 0x14..]),
    }
}

#[cfg(test)]
fn write_tessellation_factor(
    layout: &IcbCommandLayout,
    slot: &mut [u8],
    tf: &IcbTessellationFactor,
) -> Result<(), IcbStatus> {
    if layout.tessellation_factor_offset == 0 {
        return Ok(());
    }
    let off = layout.tessellation_factor_offset as usize;
    if off + ICB_TESSELLATION_FACTOR_LEN > slot.len() {
        return Err(IcbStatus::Args("icb_write_tess_factor_oob"));
    }
    use crate::contract::endian::{st32, st64};
    st32(&mut slot[off..], tf.buffer_ref);
    let va = if tf.wire_va != 0 { tf.wire_va } else { 0 };
    st64(&mut slot[off + 4..], va);
    st64(&mut slot[off + 0xc..], va);
    st64(&mut slot[off + 0x14..], tf.instance_stride);
    Ok(())
}

/// Bind-table slot count between two layout offsets (`count × 0x14`).
fn icb_layout_stage_bind_count(start: u32, end: u32) -> u16 {
    if end <= start {
        return 0;
    }
    ((end - start) / ICB_BUFFER_BIND_STRIDE as u32) as u16
}

/// Encode one render Draw / DrawIndexed command slot (tests / fixtures).
#[cfg(test)]
pub fn encode_render_command_slot(
    layout: &IcbCommandLayout,
    fill: &IcbRenderFill,
) -> Result<Vec<u8>, IcbStatus> {
    use crate::contract::endian::{st16, st32, st64};
    let size = layout.command_size as usize;
    if size == 0 {
        return Err(IcbStatus::Args("icb_ers_zero_command_size"));
    }
    let mut slot = vec![0u8; size];
    let type_off = layout.command_type_offset as usize;
    if layout.pipeline_state_offset != 0 {
        st32(
            &mut slot[layout.pipeline_state_offset as usize..],
            fill.pipeline_ref,
        );
    }
    for b in &fill.buffers {
        let base = match b.effective_stage() {
            IcbRenderBindStage::Vertex => layout.vertex_buffer_bind_offset,
            IcbRenderBindStage::Fragment => layout.fragment_buffer_bind_offset,
            IcbRenderBindStage::Object => layout.object_buffer_bind_offset,
            IcbRenderBindStage::Mesh => layout.mesh_buffer_bind_offset,
        } as usize;
        let off = base + (b.index as usize) * ICB_BUFFER_BIND_STRIDE;
        if off + ICB_BUFFER_BIND_STRIDE > size {
            return Err(IcbStatus::Args("icb_ers_bind_offset_oob"));
        }
        st32(&mut slot[off..], b.buffer_ref);
        // Wire VA = absolute GVA (base+offset). Prefer explicit wire_va; else 0
        // (host fill uses offset without a wire VA). Same 0x14 packing as
        // setVertexBuffer / setFragmentBuffer (ref@0 · va@4 · gpuva@0xc).
        let va = if b.wire_va != 0 { b.wire_va } else { 0 };
        st64(&mut slot[off + 4..], va);
        st64(&mut slot[off + 0xc..], va); // gpuva same as va for fixtures
        if b.effective_stage() == IcbRenderBindStage::Vertex && b.has_attribute_stride {
            write_attribute_stride(layout, &mut slot, b.index, b.attribute_stride)?;
        }
    }
    // Object TG memory lengths (u64 at objectThreadgroupMemoryLengthOffset + i*8).
    for tg in &fill.object_threadgroup_memory {
        if layout.object_threadgroup_memory_length_offset == 0 {
            return Err(IcbStatus::Args("icb_ers_no_object_tg_table"));
        }
        let off = layout.object_threadgroup_memory_length_offset as usize
            + (tg.index as usize) * ICB_TG_MEMORY_STRIDE;
        if off + 8 > size {
            return Err(IcbStatus::Args("icb_ers_object_tg_offset_oob"));
        }
        st64(&mut slot[off..], tg.length);
    }
    let args = layout.command_arguments_offset as usize;
    match fill.draw {
        IcbRenderDraw::Primitives {
            primitive_type,
            vertex_start,
            vertex_count,
            instance_count,
            base_instance,
        } => {
            if args + 0x22 > size {
                return Err(IcbStatus::Args("icb_ers_draw_args_oob"));
            }
            st32(&mut slot[type_off..], ICB_CMD_TYPE_DRAW);
            st16(&mut slot[args..], primitive_type);
            st64(&mut slot[args + 2..], vertex_start);
            st64(&mut slot[args + 0xa..], vertex_count);
            st64(&mut slot[args + 0x12..], instance_count);
            st64(&mut slot[args + 0x1a..], base_instance);
        }
        IcbRenderDraw::Indexed {
            primitive_type,
            index_type,
            index_buffer_ref,
            index_count,
            index_buffer_offset: _,
            index_wire_va,
            instance_count,
            base_vertex,
            base_instance,
        } => {
            if args + 0x38 > size {
                return Err(IcbStatus::Args("icb_ers_indexed_args_oob"));
            }
            st32(&mut slot[type_off..], ICB_CMD_TYPE_DRAW_INDEXED);
            st16(&mut slot[args..], primitive_type);
            st16(&mut slot[args + 2..], index_type);
            st32(&mut slot[args + 4..], index_buffer_ref);
            st64(&mut slot[args + 8..], index_count);
            st64(&mut slot[args + 0x10..], index_wire_va);
            st64(&mut slot[args + 0x18..], index_wire_va);
            st64(&mut slot[args + 0x20..], instance_count);
            st64(&mut slot[args + 0x28..], base_vertex as u64);
            st64(&mut slot[args + 0x30..], base_instance);
        }
        IcbRenderDraw::Patches {
            number_of_patch_control_points,
            patch_start,
            patch_count,
            patch_index_buffer_ref,
            patch_index_buffer_offset: _,
            patch_index_wire_va,
            instance_count,
            base_instance,
            tessellation_factor,
        } => {
            if args + ICB_DRAW_PATCHES_ARGS_LEN as usize > size {
                return Err(IcbStatus::Args("icb_ers_patches_args_oob"));
            }
            st32(&mut slot[type_off..], ICB_CMD_TYPE_DRAW_PATCHES);
            st16(&mut slot[args..], number_of_patch_control_points);
            st64(&mut slot[args + 2..], patch_start);
            st64(&mut slot[args + 0xa..], patch_count);
            st32(&mut slot[args + 0x12..], patch_index_buffer_ref);
            st64(&mut slot[args + 0x16..], patch_index_wire_va);
            st64(&mut slot[args + 0x1e..], patch_index_wire_va);
            st64(&mut slot[args + 0x26..], instance_count);
            st64(&mut slot[args + 0x2e..], base_instance);
            write_tessellation_factor(layout, &mut slot, &tessellation_factor)?;
        }
        IcbRenderDraw::IndexedPatches {
            number_of_patch_control_points,
            patch_start,
            patch_count,
            patch_index_buffer_ref,
            patch_index_buffer_offset: _,
            patch_index_wire_va,
            control_point_index_buffer_ref,
            control_point_index_buffer_offset: _,
            control_point_index_wire_va,
            instance_count,
            base_instance,
            tessellation_factor,
        } => {
            if args + ICB_DRAW_INDEXED_PATCHES_ARGS_LEN as usize > size {
                return Err(IcbStatus::Args("icb_ers_indexed_patches_args_oob"));
            }
            st32(&mut slot[type_off..], ICB_CMD_TYPE_DRAW_INDEXED_PATCHES);
            st16(&mut slot[args..], number_of_patch_control_points);
            st64(&mut slot[args + 2..], patch_start);
            st64(&mut slot[args + 0xa..], patch_count);
            st32(&mut slot[args + 0x12..], patch_index_buffer_ref);
            st64(&mut slot[args + 0x16..], patch_index_wire_va);
            st64(&mut slot[args + 0x1e..], patch_index_wire_va);
            st32(&mut slot[args + 0x26..], control_point_index_buffer_ref);
            st64(&mut slot[args + 0x2a..], control_point_index_wire_va);
            st64(&mut slot[args + 0x32..], control_point_index_wire_va);
            st64(&mut slot[args + 0x3a..], instance_count);
            st64(&mut slot[args + 0x42..], base_instance);
            write_tessellation_factor(layout, &mut slot, &tessellation_factor)?;
        }
        IcbRenderDraw::MeshThreads(mesh) | IcbRenderDraw::MeshThreadgroups(mesh) => {
            let threads = matches!(fill.draw, IcbRenderDraw::MeshThreads(_));
            if args + ICB_DRAW_MESH_ARGS_LEN as usize > size {
                return Err(IcbStatus::Args(if threads {
                    "icb_ers_mesh_threads_args_oob"
                } else {
                    "icb_ers_mesh_threadgroups_args_oob"
                }));
            }
            st32(
                &mut slot[type_off..],
                if threads {
                    ICB_CMD_TYPE_DRAW_MESH_THREADS
                } else {
                    ICB_CMD_TYPE_DRAW_MESH_THREADGROUPS
                },
            );
            mesh.encode(&mut slot, args);
        }
    }
    let _ = (
        MTL_INDIRECT_CMD_DRAW,
        MTL_INDIRECT_CMD_DRAW_PATCHES,
        MTL_INDIRECT_CMD_DRAW_INDEXED_PATCHES,
    );
    Ok(slot)
}

/// Encode one compute command slot into ICB backing bytes (tests / fixtures).
#[cfg(test)]
pub fn encode_compute_command_slot(
    layout: &IcbCommandLayout,
    fill: &IcbComputeFill,
) -> Result<Vec<u8>, IcbStatus> {
    use crate::contract::endian::{st32, st64};
    let size = layout.command_size as usize;
    if size == 0 {
        return Err(IcbStatus::Args("icb_ecs_zero_command_size"));
    }
    let mut slot = vec![0u8; size];
    let (cmd_type, gx, gy, gz, tx, ty, tz) = match fill.dispatch {
        IcbFillDispatch::ConcurrentThreadgroups {
            grid_x,
            grid_y,
            grid_z,
            tg_x,
            tg_y,
            tg_z,
        } => (
            ICB_CMD_TYPE_CONCURRENT_DISPATCH_THREADGROUPS,
            grid_x,
            grid_y,
            grid_z,
            tg_x,
            tg_y,
            tg_z,
        ),
        IcbFillDispatch::ConcurrentThreads {
            threads_x,
            threads_y,
            threads_z,
            tg_x,
            tg_y,
            tg_z,
        } => (
            ICB_CMD_TYPE_CONCURRENT_DISPATCH_THREADS,
            threads_x,
            threads_y,
            threads_z,
            tg_x,
            tg_y,
            tg_z,
        ),
    };
    let type_off = layout.command_type_offset as usize;
    if type_off + 4 > size {
        return Err(IcbStatus::Args("icb_ecs_type_offset_oob"));
    }
    st32(&mut slot[type_off..], cmd_type);
    if layout.pipeline_state_offset != 0 {
        let off = layout.pipeline_state_offset as usize;
        if off + 4 > size {
            return Err(IcbStatus::Args("icb_ecs_pipeline_offset_oob"));
        }
        st32(&mut slot[off..], fill.pipeline_ref);
    }
    for b in &fill.buffers {
        let off =
            layout.kernel_buffer_bind_offset as usize + (b.index as usize) * ICB_BUFFER_BIND_STRIDE;
        if off + ICB_BUFFER_BIND_STRIDE > size {
            return Err(IcbStatus::Args("icb_ecs_bind_offset_oob"));
        }
        st32(&mut slot[off..], b.buffer_ref);
        let va = if b.wire_va != 0 { b.wire_va } else { 0 };
        st64(&mut slot[off + 4..], va);
        st64(&mut slot[off + 0xc..], va);
        if b.has_attribute_stride {
            write_attribute_stride(layout, &mut slot, b.index, b.attribute_stride)?;
        }
    }
    // Barrier u32 (1 = setBarrier, 0 = clear).
    if layout.barrier_offset != 0 {
        let bo = layout.barrier_offset as usize;
        if bo + 4 > size {
            return Err(IcbStatus::Args("icb_ecs_barrier_offset_oob"));
        }
        st32(&mut slot[bo..], if fill.barrier { 1 } else { 0 });
    }
    // Threadgroup memory length table (u64 per index).
    for tg in &fill.threadgroup_memory {
        let off = layout.threadgroup_memory_length_offset as usize
            + (tg.index as usize) * ICB_TG_MEMORY_STRIDE;
        if off + 8 > size {
            return Err(IcbStatus::Args("icb_ecs_tg_offset_oob"));
        }
        st64(&mut slot[off..], tg.length);
    }
    let args = layout.command_arguments_offset as usize;
    if args + ICB_CONCURRENT_DISPATCH_ARGS_LEN > size {
        return Err(IcbStatus::Args("icb_ecs_dispatch_args_oob"));
    }
    st64(&mut slot[args..], gx as u64);
    st64(&mut slot[args + 8..], gy as u64);
    st64(&mut slot[args + 16..], gz as u64);
    st64(&mut slot[args + 24..], tx as u64);
    st64(&mut slot[args + 32..], ty as u64);
    st64(&mut slot[args + 40..], tz as u64);
    Ok(slot)
}

/// Load and decode a type-7 ICB descriptor for `icb_ref` on the task object list.
pub fn load_icb_descriptor<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    icb_ref: u32,
) -> Result<IndirectCommandBufferDescriptor, IcbStatus> {
    if icb_ref == 0 {
        return Err(IcbStatus::Missing("icb_desc_ref_zero"));
    }
    let entry = objects::lookup_list_entry(state, host, task_id, icb_ref)
        .ok_or(IcbStatus::Missing("icb_desc_no_list_entry"))?;
    if entry.object_type != OBJECT_TYPE_TYPE7 {
        return Err(IcbStatus::BadDescriptor("icb_desc_wrong_type"));
    }
    let desc = objects::read_descriptor(state, host, task_id, &entry)
        .ok_or(IcbStatus::Missing("icb_desc_read"))?;
    match decode_type7_descriptor(&desc) {
        Ok(ResourceDescriptor::IndirectCommandBuffer(icb)) => Ok(icb),
        Ok(_) => Err(IcbStatus::BadDescriptor("icb_desc_not_icb_body")),
        Err(_) => Err(IcbStatus::BadDescriptor("icb_desc_type7_decode")),
    }
}

/// Materialize a host Metal ICB from a decoded create descriptor (uncached).
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
pub fn materialize_metal_icb(
    desc: &IndirectCommandBufferDescriptor,
) -> Result<metal::IndirectCommandBuffer, IcbStatus> {
    use crate::backend::metal::runtime::system_device;
    use metal::{IndirectCommandBufferDescriptor, MTLResourceOptions};

    if desc.max_command_count == 0 {
        return Err(IcbStatus::Args("icb_materialize_zero_command_count"));
    }
    let Some(device) = system_device() else {
        return Err(IcbStatus::NoMetal("icb_materialize_no_metal"));
    };
    let mtl_desc = IndirectCommandBufferDescriptor::new();
    // Pass wire commandTypes bits through as-is (SDK layout). Do not use
    // metal-0.33's MTLIndirectCommandType bitflags: ConcurrentDispatch is
    // mis-shifted and mesh bits (1<<7 / 1<<8) are omitted — from_bits_truncate
    // drops unknown bits and yields an empty ICB.
    crate::backend::metal::raw_metal::icb_descriptor_set_command_types(
        mtl_desc.as_ref(),
        desc.command_types as u64,
    );
    mtl_desc.set_inherit_buffers(desc.inherit_buffers);
    mtl_desc.set_inherit_pipeline_state(desc.inherit_pipeline_state);
    mtl_desc.set_max_vertex_buffer_bind_count(desc.max_vertex_buffer_bind_count as u64);
    mtl_desc.set_max_fragment_buffer_bind_count(desc.max_fragment_buffer_bind_count as u64);
    mtl_desc.set_max_kernel_buffer_bind_count(desc.max_kernel_buffer_bind_count as u64);
    // Prefer create-body count; fall back to layout-implied TG slot count.
    let max_tg = desc
        .max_kernel_threadgroup_memory_bind_count
        .max(icb_layout_kernel_tg_slot_count(&desc.layout));
    if max_tg > 0 {
        crate::backend::metal::raw_metal::set_max_kernel_threadgroup_memory_bind_count(
            mtl_desc.as_ref(),
            max_tg as u64,
        );
    }
    // Mesh / object bind counts from create body (macOS 14+).
    crate::backend::metal::raw_metal::set_max_mesh_buffer_bind_count(
        mtl_desc.as_ref(),
        desc.max_mesh_buffer_bind_count as u64,
    );
    crate::backend::metal::raw_metal::set_max_object_buffer_bind_count(
        mtl_desc.as_ref(),
        desc.max_object_buffer_bind_count as u64,
    );
    if desc.max_object_threadgroup_memory_bind_count > 0 {
        crate::backend::metal::raw_metal::set_max_object_threadgroup_memory_bind_count(
            mtl_desc.as_ref(),
            desc.max_object_threadgroup_memory_bind_count as u64,
        );
    }

    let options = MTLResourceOptions::from_bits_truncate(desc.options as u64);
    let icb = device.new_indirect_command_buffer_with_descriptor(
        &mtl_desc,
        desc.max_command_count as u64,
        options,
    );
    let _ = TYPE7_OBJECT_ICB;
    Ok(icb)
}

#[cfg(feature = "backend-vulkan")]
pub fn materialize_metal_icb(_desc: &IndirectCommandBufferDescriptor) -> Result<(), IcbStatus> {
    Err(IcbStatus::NoMetal("icb_materialize_no_vulkan_path"))
}

// ---------------------------------------------------------------------------
// Host ICB cache: (task_id, icb_ref) → filled Metal ICB + retained resources
// ---------------------------------------------------------------------------

#[cfg(all(feature = "backend-metal", target_os = "macos"))]
struct HostIcbEntry {
    desc: IndirectCommandBufferDescriptor,
    icb: metal::IndirectCommandBuffer,
    /// Keep compute PSOs alive while command slots reference them.
    retained_psos: Vec<metal::ComputePipelineState>,
    /// Keep render PSOs alive while command slots reference them.
    retained_psos_render: Vec<metal::RenderPipelineState>,
    retained_buffers: Vec<metal::Buffer>,
    /// GVA writeback descriptors for buffers bound into filled commands.
    writebacks: Vec<IcbWriteback>,
    /// Guest ICB backing buffer (CPU-filled command slots).
    command_memory: Option<IcbCommandMemory>,
    /// True once at least one host fill or guest-memory fill has landed.
    has_fills: bool,
}

#[cfg(all(feature = "backend-metal", target_os = "macos"))]
struct IcbWriteback {
    bind: crate::runtime::compute_exec::ComputeBufferBind,
    gva: u64,
    /// Host staging length (GPU result copied here after execute, then to GVA).
    len: usize,
    /// The staging walk's page set, carried from the [`StagedBuffer`] this slot
    /// was recorded from. A cached ICB replays long after the stage, so the
    /// writeback has to be bounded by where the buffer resolved *then*; a walk
    /// taken at replay time answers where the GVA points now, which is the
    /// question that lets a recycled page take the write.
    pages: std::collections::HashSet<u64>,
    mtl: metal::Buffer,
}

#[cfg(all(feature = "backend-metal", target_os = "macos"))]
fn icb_cache() -> &'static parking_lot::Mutex<HashMap<(u32, u32), HostIcbEntry>> {
    static CACHE: OnceLock<parking_lot::Mutex<HashMap<(u32, u32), HostIcbEntry>>> = OnceLock::new();
    CACHE.get_or_init(|| parking_lot::Mutex::new(HashMap::new()))
}

/// Drop all cached host ICBs (tests / task teardown).
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
pub fn clear_icb_cache() {
    icb_cache().lock().clear();
}

#[cfg(feature = "backend-vulkan")]
pub fn clear_icb_cache() {}

/// Resolve guest ICB ref → host Metal ICB, reusing the per-(task,ref) cache.
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
pub fn resolve_metal_icb<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    icb_ref: u32,
) -> Result<
    (
        IndirectCommandBufferDescriptor,
        metal::IndirectCommandBuffer,
    ),
    IcbStatus,
> {
    let desc = load_icb_descriptor(state, host, task_id, icb_ref)?;
    let mut cache = icb_cache().lock();
    if let Some(entry) = cache.get(&(task_id, icb_ref)) {
        // Descriptor must still match the create body we materialize from.
        if entry.desc.max_command_count == desc.max_command_count
            && entry.desc.command_types == desc.command_types
        {
            return Ok((entry.desc.clone(), entry.icb.clone()));
        }
    }
    let icb = materialize_metal_icb(&desc)?;
    cache.insert(
        (task_id, icb_ref),
        HostIcbEntry {
            desc: desc.clone(),
            icb: icb.clone(),
            retained_psos: Vec::new(),
            retained_psos_render: Vec::new(),
            retained_buffers: Vec::new(),
            writebacks: Vec::new(),
            command_memory: None,
            has_fills: false,
        },
    );
    Ok((desc, icb))
}

/// Info-segment opcode for `PGSerializerInfoCommandEncoder icbHostResourceInfo:info:`.
///
/// Full wire record length `0x18` (8 B header + 16 B payload). Payload:
/// `icb_ref:u32 @0`, `buffer_ref:u32 @4`, `gpu_address:u64 @8`.
/// Host RE 2026-07-11 AppleParavirtGPUMetalIOGPUFamily.
pub const INFO_OP_ICB_HOST_RESOURCE: u32 = 0x1d1;
pub const INFO_OP_ICB_HOST_RESOURCE_RECORD_LEN: u32 = 0x18;
pub const INFO_OP_ICB_HOST_RESOURCE_PAYLOAD_LEN: usize = 0x10;

/// Decoded `0x1d1` icbHostResourceInfo payload (sync path for ICB backing).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IcbHostResourceInfo {
    pub icb_ref: u32,
    /// Type-1 object-list ref of the ICB command backing buffer.
    pub buffer_ref: u32,
    /// Guest GPU/VA of the backing (`AppleParavirtBuffer._gpuAddress`), or 0.
    pub gpu_address: u64,
}

/// Decode `0x1d1` payload (16 bytes) or full record (24 bytes including header).
pub fn decode_icb_host_resource_info(bytes: &[u8]) -> Result<IcbHostResourceInfo, IcbStatus> {
    let payload = if bytes.len() >= INFO_OP_ICB_HOST_RESOURCE_RECORD_LEN as usize
        && ld32(&bytes[0..]) == INFO_OP_ICB_HOST_RESOURCE
    {
        &bytes[8..8 + INFO_OP_ICB_HOST_RESOURCE_PAYLOAD_LEN]
    } else if bytes.len() >= INFO_OP_ICB_HOST_RESOURCE_PAYLOAD_LEN {
        &bytes[..INFO_OP_ICB_HOST_RESOURCE_PAYLOAD_LEN]
    } else {
        return Err(IcbStatus::Args("icb_host_resource_info_short"));
    };
    let icb_ref = ld32(&payload[0..]);
    let buffer_ref = ld32(&payload[4..]);
    let gpu_address = ld64(&payload[8..]);
    if icb_ref == 0 {
        return Err(IcbStatus::Args("icb_host_resource_info_ref_zero"));
    }
    Ok(IcbHostResourceInfo {
        icb_ref,
        buffer_ref,
        gpu_address,
    })
}

/// Register guest command-memory GVA for an ICB (backing buffer for CPU fills).
///
/// Fills are not stream opcodes; the guest writes this buffer via
/// `PGSerializerIndirectComputeCommand`. Product re-decodes it at execute.
pub fn bind_icb_command_memory(
    task_id: u32,
    icb_ref: u32,
    mem: IcbCommandMemory,
) -> Result<(), IcbStatus> {
    if icb_ref == 0 || mem.gva == 0 || mem.byte_len == 0 {
        return Err(IcbStatus::Args("icb_bind_memory_bad_args"));
    }
    #[cfg(feature = "backend-vulkan")]
    {
        let _ = (task_id, icb_ref, mem);
        Err(IcbStatus::NoMetal("icb_bind_memory_no_vulkan_path"))
    }
    #[cfg(all(feature = "backend-metal", target_os = "macos"))]
    {
        let mut cache = icb_cache().lock();
        let entry = cache
            .get_mut(&(task_id, icb_ref))
            .ok_or(IcbStatus::Missing("icb_bind_memory_not_cached"))?;
        entry.command_memory = Some(mem);
        Ok(())
    }
}

/// Associate ICB command memory from a type-1 buffer object-list ref (sync path).
///
/// Resolves buffer GVA/size via the type-1 descriptor (`handle << PAGE_SHIFT`).
/// Byte length is min(buffer size, command_size × max_command_count) from the
/// ICB create layout so oversize type-1 allocations are truncated to the ICB.
pub fn associate_icb_backing_buffer_ref<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    icb_ref: u32,
    buffer_ref: u32,
) -> Result<IcbCommandMemory, IcbStatus> {
    if icb_ref == 0 || buffer_ref == 0 {
        return Err(IcbStatus::Args("icb_associate_ref_zero"));
    }
    // Materialize host ICB + layout from type-7 create if needed.
    let (desc, _) = resolve_metal_icb(state, host, task_id, icb_ref)?;
    let (gva, buf_size) = type1_buffer_gva_size(state, host, task_id, buffer_ref)?;
    let need = (desc.layout.command_size as u64).saturating_mul(desc.max_command_count as u64);
    if need == 0 {
        return Err(IcbStatus::Args("icb_associate_zero_layout_span"));
    }
    if buf_size < need {
        return Err(IcbStatus::Args("icb_associate_buffer_too_small"));
    }
    let mem = IcbCommandMemory {
        gva,
        byte_len: need,
    };
    bind_icb_command_memory(task_id, icb_ref, mem)?;
    Ok(mem)
}

/// Apply info-segment `0x1d1` (icbHostResourceInfo) — auto-bind backing GVA.
///
/// Prefers type-1 `buffer_ref` for GVA; if buffer_ref is 0, uses `gpu_address`
/// with size from the ICB create layout.
pub fn apply_icb_host_resource_info<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    info: &IcbHostResourceInfo,
) -> Result<IcbCommandMemory, IcbStatus> {
    if info.buffer_ref != 0 {
        return associate_icb_backing_buffer_ref(
            state,
            host,
            task_id,
            info.icb_ref,
            info.buffer_ref,
        );
    }
    if info.gpu_address == 0 {
        return Err(IcbStatus::Args("icb_apply_info_no_gpu_address"));
    }
    let (desc, _) = resolve_metal_icb(state, host, task_id, info.icb_ref)?;
    let need = (desc.layout.command_size as u64).saturating_mul(desc.max_command_count as u64);
    if need == 0 {
        return Err(IcbStatus::Args("icb_apply_info_zero_layout_span"));
    }
    let mem = IcbCommandMemory {
        gva: info.gpu_address,
        byte_len: need,
    };
    bind_icb_command_memory(task_id, info.icb_ref, mem)?;
    Ok(mem)
}

/// Read attribute-stride u64 at `attributeStrideOffset + index*8`.
///
/// Returns `(stride, has)` — `has` is true when a stride table slot exists and
/// the stored value is non-zero, or when the slot exists and we treat any
/// stored value as authoritative (including 0 from host encode of
/// `has_attribute_stride` with stride 0 is rare; product uses non-zero for has).
fn read_attribute_stride(layout: &IcbCommandLayout, slot: &[u8], index: u32) -> (u64, bool) {
    let slots = icb_layout_attribute_stride_slot_count(layout);
    if slots == 0 || index >= slots as u32 || layout.attribute_stride_offset == 0 {
        return (0, false);
    }
    let off = layout.attribute_stride_offset as usize
        + (index as usize) * ICB_ATTRIBUTE_STRIDE_ENTRY_SIZE;
    if off + 8 > slot.len() {
        return (0, false);
    }
    let stride = ld64(&slot[off..]);
    // Non-zero stride means the attributeStride API was used. Zero means unset
    // (plain setKernelBuffer/setVertexBuffer does not touch this table).
    if stride != 0 {
        (stride, true)
    } else {
        (0, false)
    }
}

#[cfg(test)]
fn write_attribute_stride(
    layout: &IcbCommandLayout,
    slot: &mut [u8],
    index: u32,
    stride: u64,
) -> Result<(), IcbStatus> {
    use crate::contract::endian::st64;
    let slots = icb_layout_attribute_stride_slot_count(layout);
    if slots == 0 || index >= slots as u32 || layout.attribute_stride_offset == 0 {
        return Err(IcbStatus::Args("icb_attribute_stride_no_slot"));
    }
    let off = layout.attribute_stride_offset as usize
        + (index as usize) * ICB_ATTRIBUTE_STRIDE_ENTRY_SIZE;
    if off + 8 > slot.len() {
        return Err(IcbStatus::Args("icb_attribute_stride_offset_oob"));
    }
    st64(&mut slot[off..], stride);
    Ok(())
}

fn type1_buffer_gva_size<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    buffer_ref: u32,
) -> Result<(u64, u64), IcbStatus> {
    use crate::runtime::decode::resource::{decode_buffer_descriptor, OBJECT_TYPE_BUFFER};
    let entry = objects::lookup_list_entry(state, host, task_id, buffer_ref)
        .ok_or(IcbStatus::Missing("icb_type1_no_list_entry"))?;
    if entry.object_type != OBJECT_TYPE_BUFFER {
        return Err(IcbStatus::BadDescriptor("icb_type1_wrong_type"));
    }
    let desc_bytes = objects::read_descriptor(state, host, task_id, &entry)
        .ok_or(IcbStatus::Missing("icb_type1_desc_read"))?;
    let desc = decode_buffer_descriptor(&desc_bytes)
        .map_err(|_| IcbStatus::BadDescriptor("icb_type1_desc_decode"))?;
    desc.backing_gva_size(state.page_shift)
        .ok_or(IcbStatus::Missing("icb_type1_no_backing"))
}

/// Convert absolute bind VA → offset into type-1 allocation (`handle << page_shift`).
///
/// PGSerializer stores `base+offset` in the bind VA field (not a separate offset).
/// `wire_va == 0` means base (offset 0). Fail-closed if VA is below base or past size.
fn offset_from_wire_va<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    buffer_ref: u32,
    wire_va: u64,
) -> Result<u64, IcbStatus> {
    if wire_va == 0 {
        return Ok(0);
    }
    let (base, size) = type1_buffer_gva_size(state, host, task_id, buffer_ref)?;
    if wire_va < base {
        return Err(IcbStatus::Args("icb_wire_va_below_base"));
    }
    let off = wire_va - base;
    if off >= size {
        return Err(IcbStatus::Args("icb_wire_va_past_end"));
    }
    Ok(off)
}

/// Resolve wire VAs on a compute fill into type-1 bind offsets (mutates in place).
pub fn resolve_compute_fill_offsets<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    fill: &mut IcbComputeFill,
) -> Result<(), IcbStatus> {
    for b in &mut fill.buffers {
        if b.wire_va != 0 {
            b.offset = offset_from_wire_va(state, host, task_id, b.buffer_ref, b.wire_va)?;
        }
    }
    Ok(())
}

/// Resolve wire VAs on a render fill into type-1 bind / index offsets (mutates in place).
pub fn resolve_render_fill_offsets<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    fill: &mut IcbRenderFill,
) -> Result<(), IcbStatus> {
    for b in &mut fill.buffers {
        if b.wire_va != 0 {
            b.offset = offset_from_wire_va(state, host, task_id, b.buffer_ref, b.wire_va)?;
        }
    }
    match &mut fill.draw {
        IcbRenderDraw::Indexed {
            index_buffer_ref,
            index_wire_va,
            index_buffer_offset,
            ..
        } => {
            if *index_wire_va != 0 {
                *index_buffer_offset =
                    offset_from_wire_va(state, host, task_id, *index_buffer_ref, *index_wire_va)?;
            }
        }
        IcbRenderDraw::Patches {
            patch_index_buffer_ref,
            patch_index_wire_va,
            patch_index_buffer_offset,
            tessellation_factor,
            ..
        } => {
            if *patch_index_wire_va != 0 && *patch_index_buffer_ref != 0 {
                *patch_index_buffer_offset = offset_from_wire_va(
                    state,
                    host,
                    task_id,
                    *patch_index_buffer_ref,
                    *patch_index_wire_va,
                )?;
            }
            if tessellation_factor.wire_va != 0 && tessellation_factor.buffer_ref != 0 {
                tessellation_factor.offset = offset_from_wire_va(
                    state,
                    host,
                    task_id,
                    tessellation_factor.buffer_ref,
                    tessellation_factor.wire_va,
                )?;
            }
        }
        IcbRenderDraw::IndexedPatches {
            patch_index_buffer_ref,
            patch_index_wire_va,
            patch_index_buffer_offset,
            control_point_index_buffer_ref,
            control_point_index_wire_va,
            control_point_index_buffer_offset,
            tessellation_factor,
            ..
        } => {
            if *patch_index_wire_va != 0 && *patch_index_buffer_ref != 0 {
                *patch_index_buffer_offset = offset_from_wire_va(
                    state,
                    host,
                    task_id,
                    *patch_index_buffer_ref,
                    *patch_index_wire_va,
                )?;
            }
            if *control_point_index_wire_va != 0 {
                *control_point_index_buffer_offset = offset_from_wire_va(
                    state,
                    host,
                    task_id,
                    *control_point_index_buffer_ref,
                    *control_point_index_wire_va,
                )?;
            }
            if tessellation_factor.wire_va != 0 && tessellation_factor.buffer_ref != 0 {
                tessellation_factor.offset = offset_from_wire_va(
                    state,
                    host,
                    task_id,
                    tessellation_factor.buffer_ref,
                    tessellation_factor.wire_va,
                )?;
            }
        }
        IcbRenderDraw::Primitives { .. }
        | IcbRenderDraw::MeshThreads(_)
        | IcbRenderDraw::MeshThreadgroups(_) => {}
    }
    Ok(())
}

/// Decode guest command memory into host ICB fills for the given index range.
///
/// Called from execute when the ICB has registered command memory. Dispatches
/// compute vs render fills from wire `commandTypes` / slot command-type tags.
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
pub fn fill_icb_from_command_memory<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    icb_ref: u32,
    range_location: u64,
    range_length: u64,
) -> Result<(), IcbStatus> {
    use crate::runtime::decode::resource::{
        MTL_INDIRECT_CMD_CONCURRENT_DISPATCH, MTL_INDIRECT_CMD_CONCURRENT_DISPATCH_THREADS,
        MTL_INDIRECT_CMD_DRAW, MTL_INDIRECT_CMD_DRAW_INDEXED,
        MTL_INDIRECT_CMD_DRAW_INDEXED_PATCHES, MTL_INDIRECT_CMD_DRAW_MESH_THREADGROUPS,
        MTL_INDIRECT_CMD_DRAW_MESH_THREADS, MTL_INDIRECT_CMD_DRAW_PATCHES,
    };
    use crate::runtime::gva_mem;

    let (layout, max_kernel, max_vertex, max_fragment, command_types, max_cmds, mem) = {
        let cache = icb_cache().lock();
        let entry = cache
            .get(&(task_id, icb_ref))
            .ok_or(IcbStatus::Missing("icb_fill_not_cached"))?;
        let mem = entry
            .command_memory
            .ok_or(IcbStatus::Missing("icb_fill_no_command_memory"))?;
        (
            entry.desc.layout,
            entry.desc.max_kernel_buffer_bind_count,
            entry.desc.max_vertex_buffer_bind_count,
            entry.desc.max_fragment_buffer_bind_count,
            entry.desc.command_types,
            entry.desc.max_command_count as u64,
            mem,
        )
    };
    if layout.command_size == 0 {
        return Err(IcbStatus::Args("icb_fill_zero_command_size"));
    }
    let end = range_location.saturating_add(range_length);
    if end > max_cmds {
        return Err(IcbStatus::Args("icb_fill_range_past_capacity"));
    }
    let need = (end as u64).saturating_mul(layout.command_size as u64);
    if need > mem.byte_len {
        return Err(IcbStatus::Args("icb_fill_range_past_memory"));
    }
    let mut bytes = vec![0u8; need as usize];
    gva_mem::read_task_gva_by_id(
        host,
        &state.tasks,
        task_id,
        mem.gva,
        &mut bytes,
        state.page_shift,
    )
    .map_err(|_| IcbStatus::MetalFailed("icb_fill_command_memory_read"))?;

    let is_compute = command_types
        & (MTL_INDIRECT_CMD_CONCURRENT_DISPATCH | MTL_INDIRECT_CMD_CONCURRENT_DISPATCH_THREADS)
        != 0;
    let is_render = command_types
        & (MTL_INDIRECT_CMD_DRAW
            | MTL_INDIRECT_CMD_DRAW_INDEXED
            | MTL_INDIRECT_CMD_DRAW_PATCHES
            | MTL_INDIRECT_CMD_DRAW_INDEXED_PATCHES
            | MTL_INDIRECT_CMD_DRAW_MESH_THREADGROUPS
            | MTL_INDIRECT_CMD_DRAW_MESH_THREADS)
        != 0;

    for i in range_location..end {
        let off = (i as usize) * (layout.command_size as usize);
        let slot = &bytes[off..off + layout.command_size as usize];
        if is_compute || (!is_render && !is_compute) {
            // Prefer compute when ConcurrentDispatch bits are set; empty slots skip.
            if let Some(mut fill) = decode_compute_command_slot(&layout, slot, max_kernel)? {
                fill.command_index = i as u32;
                resolve_compute_fill_offsets(state, host, task_id, &mut fill)?;
                fill_compute_command(state, host, task_id, icb_ref, &fill)?;
                continue;
            }
        }
        if is_render {
            if let Some(mut fill) =
                decode_render_command_slot(&layout, slot, max_vertex, max_fragment)?
            {
                fill.command_index = i as u32;
                resolve_render_fill_offsets(state, host, task_id, &mut fill)?;
                fill_render_command(state, host, task_id, icb_ref, &fill)?;
            }
        }
    }
    Ok(())
}

/// Build an `MTLVertexDescriptor` from the type-7 pipeline vertex-input block.
///
/// Returns `None` when there are no usable attributes (no stage-in / SSBO-only
/// pipelines). Format `0` or stride `0` entries are skipped as incomplete.
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
pub(crate) fn metal_vertex_descriptor_from_attrs(
    attrs: &[crate::runtime::decode::resource::VertexAttribute],
) -> Option<metal::VertexDescriptor> {
    metal_vertex_descriptor_from_attrs_for_draw(attrs, false)
}

/// Build `MTLVertexDescriptor` from type-7 vertex attributes.
///
/// When `for_patches` is true and a layout lacks an explicit step function,
/// use `PerPatchControlPoint` (SDK value 4) so post-tessellation vertex
/// functions receive control-point attributes correctly.
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
pub(crate) fn metal_vertex_descriptor_from_attrs_for_draw(
    attrs: &[crate::runtime::decode::resource::VertexAttribute],
    for_patches: bool,
) -> Option<metal::VertexDescriptor> {
    use metal::{MTLVertexStepFunction, VertexDescriptor};

    if attrs.is_empty() {
        return None;
    }
    let vd = VertexDescriptor::new().to_owned();
    let mut any = false;
    for a in attrs {
        if a.format == 0 || a.stride == 0 {
            continue;
        }
        any = true;
        if let Some(attr) = vd.attributes().object_at(a.location as u64) {
            // MTLVertexFormat is NSUInteger-sized on Apple platforms.
            attr.set_format(unsafe { std::mem::transmute(a.format as u64) });
            attr.set_offset(a.offset as u64);
            attr.set_buffer_index(a.buffer_index as u64);
        }
        if let Some(layout) = vd.layouts().object_at(a.buffer_index as u64) {
            layout.set_stride(a.stride as u64);
            let step = if a.has_step_function {
                a.step_function
            } else if for_patches {
                MTLVertexStepFunction::PerPatchControlPoint as u32
            } else {
                MTLVertexStepFunction::PerVertex as u32
            };
            layout.set_step_function(unsafe { std::mem::transmute(step as u64) });
            let rate = if a.has_step_rate {
                a.step_rate.max(1)
            } else {
                1
            };
            layout.set_step_rate(rate as u64);
        }
    }
    if any {
        Some(vd)
    } else {
        None
    }
}

/// The five `MTLPrimitiveType` values the ICB wire encodes, by SDK ordinal.
/// `slug` names the caller so a refused value still says which draw form it
/// came from — the Draw and DrawIndexed arms shared this mapping verbatim and
/// differed only in that slug.
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
fn icb_primitive_type(
    primitive_type: u16,
    slug: &'static str,
) -> Result<metal::MTLPrimitiveType, IcbStatus> {
    use metal::MTLPrimitiveType;
    match primitive_type {
        0 => Ok(MTLPrimitiveType::Point),
        1 => Ok(MTLPrimitiveType::Line),
        2 => Ok(MTLPrimitiveType::LineStrip),
        3 => Ok(MTLPrimitiveType::Triangle),
        4 => Ok(MTLPrimitiveType::TriangleStrip),
        _ => Err(IcbStatus::Args(slug)),
    }
}

/// Fill one **render** command slot on a cached host ICB (Metal IndirectRenderCommand).
///
/// Builds an ICB-capable render PSO from the type-7 render pipeline's vertex/
/// fragment MTLBs (color0 = BGRA8Unorm, matching product mapping/scanout).
/// When the type-7 body carries a vertex-input block, attaches an
/// `MTLVertexDescriptor` so `[[stage_in]]` attributes bind correctly.
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
pub fn fill_render_command<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    icb_ref: u32,
    fill: &IcbRenderFill,
) -> Result<(), IcbStatus> {
    use crate::backend::metal::runtime::{new_buffer_from_host, system_device};
    use crate::runtime::compute_exec::{stage_buffer, ComputeBufferBind};
    use crate::runtime::decode::resource::{
        decode_function_descriptor, decode_render_pipeline_descriptor, FunctionDescriptor,
        OBJECT_TYPE_FUNCTION, OBJECT_TYPE_TYPE7,
    };
    use crate::runtime::gva_mem;
    use metal::{
        MTLIndexType, MTLPixelFormat, MeshRenderPipelineDescriptor, RenderPipelineDescriptor,
    };

    if icb_ref == 0 {
        return Err(IcbStatus::Args("icb_frc_ref_zero"));
    }
    let Some(device) = system_device() else {
        return Err(IcbStatus::NoMetal("icb_frc_no_metal"));
    };
    // Need create flags before staging: pipeline is required on the ICB
    // command unless inheritPipelineState (parent encoder supplies PSO).
    let (icb_desc, _) = resolve_metal_icb(state, host, task_id, icb_ref)?;
    // Host-fill path may already set offset; wire-decode path sets wire_va.
    let mut fill_resolved = fill.clone();
    resolve_render_fill_offsets(state, host, task_id, &mut fill_resolved)?;
    let fill = &fill_resolved;

    let is_patches = matches!(
        fill.draw,
        IcbRenderDraw::Patches { .. } | IcbRenderDraw::IndexedPatches { .. }
    );
    let is_mesh = matches!(
        fill.draw,
        IcbRenderDraw::MeshThreads(_) | IcbRenderDraw::MeshThreadgroups(_)
    );

    // Pipeline is required on the ICB command unless inheritPipelineState.
    // Mirrors fill_compute_command: when inherit, parent encoder supplies PSO
    // at execute (metal_draw::apply_icb_encoder_inheritance).
    let pso = if !icb_desc.inherit_pipeline_state {
        if fill.pipeline_ref == 0 {
            return Err(IcbStatus::Args("icb_frc_pipeline_ref_zero"));
        }
        let entry = objects::lookup_list_entry(state, host, task_id, fill.pipeline_ref)
            .ok_or(IcbStatus::Missing("icb_frc_pipeline_no_list_entry"))?;
        if entry.object_type != OBJECT_TYPE_TYPE7 {
            return Err(IcbStatus::BadDescriptor("icb_frc_pipeline_wrong_type"));
        }
        let desc_bytes = objects::read_descriptor(state, host, task_id, &entry)
            .ok_or(IcbStatus::Missing("icb_frc_pipeline_desc_read"))?;
        let rp = decode_render_pipeline_descriptor(&desc_bytes)
            .map_err(|_| IcbStatus::BadDescriptor("icb_frc_pipeline_desc_decode"))?;
        let load_fn = |func_ref: u32| -> Result<Vec<u8>, IcbStatus> {
            let e = objects::lookup_list_entry(state, host, task_id, func_ref)
                .ok_or(IcbStatus::Missing("icb_frc_function_no_list_entry"))?;
            if e.object_type != OBJECT_TYPE_FUNCTION {
                return Err(IcbStatus::BadDescriptor("icb_frc_function_wrong_type"));
            }
            let d = objects::read_descriptor(state, host, task_id, &e)
                .ok_or(IcbStatus::Missing("icb_frc_function_desc_read"))?;
            let f: FunctionDescriptor = decode_function_descriptor(&d)
                .map_err(|_| IcbStatus::BadDescriptor("icb_frc_function_desc_decode"))?;
            if f.blob_gva == 0 || f.blob_size < 4 {
                return Err(IcbStatus::Args("icb_frc_function_blob_empty"));
            }
            // Guest blob_size is authoritative — no product 1 MiB MTLB ceiling.
            let len = crate::runtime::metal_draw::host_alloc_len(f.blob_size as u64)
                .ok_or(IcbStatus::Args("icb_frc_function_blob_too_large"))?;
            let mut mtlb = vec![0u8; len];
            gva_mem::read_task_gva_by_id(
                host,
                &state.tasks,
                task_id,
                f.blob_gva,
                &mut mtlb,
                state.page_shift,
            )
            .map_err(|_| IcbStatus::MetalFailed("icb_frc_function_blob_read"))?;
            Ok(mtlb)
        };

        if rp.fragment_func_ref == 0 {
            return Err(IcbStatus::Missing("icb_frc_no_fragment_function"));
        }
        if is_mesh {
            // Mesh stage: mesh SPI `mesh_func_ref` (tag 0x02 under shape 0x14)
            // or classic `vertex_func_ref` (mesh-only / dual-export metallib).
            if rp.mesh_func_ref == 0 && rp.vertex_func_ref == 0 {
                return Err(IcbStatus::Missing("icb_frc_no_mesh_or_vertex_function"));
            }
        } else if rp.vertex_func_ref == 0 {
            return Err(IcbStatus::Missing("icb_frc_no_vertex_function"));
        }

        let frag = load_fn(rp.fragment_func_ref)?;
        let flib = device
            .new_library_with_data(&frag)
            .map_err(|_| IcbStatus::MetalFailed("icb_frc_fragment_library_load"))?;
        let fnames = flib.function_names();
        if fnames.len() != 1 {
            return Err(IcbStatus::Args("icb_frc_fragment_function_count"));
        }
        let ff = flib
            .get_function(&fnames[0], None)
            .map_err(|_| IcbStatus::MetalFailed("icb_frc_fragment_function_get"))?;

        // Mesh draws need MTLMeshRenderPipelineDescriptor + mesh descriptor factory.
        // Prefer mesh SPI type-7 shape (tag 0x14; 0x01 object / 0x02 mesh / 0x03 frag);
        // else dual-export or mesh-only metallib in classic `vertex_func_ref`.
        let built = if is_mesh {
            use crate::backend::metal::raw_metal::{
                function_type, MTL_FUNCTION_TYPE_MESH, MTL_FUNCTION_TYPE_OBJECT,
            };
            use metal::Library;

            let pick_typed = |lib: &Library,
                              want: u64,
                              allow_single: bool|
             -> Result<Option<metal::Function>, IcbStatus> {
                let names = lib.function_names();
                if names.is_empty() {
                    return Err(IcbStatus::Args("icb_frc_mesh_library_empty"));
                }
                let mut typed = None;
                for name in names.iter() {
                    let f = lib
                        .get_function(name, None)
                        .map_err(|_| IcbStatus::MetalFailed("icb_frc_mesh_typed_function_get"))?;
                    if function_type(f.as_ref()) == want {
                        typed = Some(f);
                        break;
                    }
                }
                if typed.is_none() && allow_single && names.len() == 1 {
                    typed =
                        Some(lib.get_function(&names[0], None).map_err(|_| {
                            IcbStatus::MetalFailed("icb_frc_mesh_single_function_get")
                        })?);
                }
                Ok(typed)
            };

            let mut mesh_fn = None;
            let mut object_fn = None;
            // Keep libraries alive for the Function refs they own.
            let mut mesh_lib_keep = None;
            let mut object_lib_keep = None;
            let mut dual_lib_keep = None;

            if rp.mesh_func_ref != 0 {
                let mtlb = load_fn(rp.mesh_func_ref)?;
                let lib = device
                    .new_library_with_data(&mtlb)
                    .map_err(|_| IcbStatus::MetalFailed("icb_frc_mesh_library_load"))?;
                mesh_fn = pick_typed(&lib, MTL_FUNCTION_TYPE_MESH, true)?;
                mesh_lib_keep = Some(lib);
            }
            if rp.object_func_ref != 0 {
                let otlb = load_fn(rp.object_func_ref)?;
                let lib = device
                    .new_library_with_data(&otlb)
                    .map_err(|_| IcbStatus::MetalFailed("icb_frc_object_library_load"))?;
                object_fn = pick_typed(&lib, MTL_FUNCTION_TYPE_OBJECT, true)?;
                object_lib_keep = Some(lib);
            }
            // Dual-export / mesh-only fallback when mesh tag absent, or object tag
            // absent and dual-export can supply the object stage.
            if (mesh_fn.is_none() || object_fn.is_none()) && rp.vertex_func_ref != 0 {
                let vtlb = load_fn(rp.vertex_func_ref)?;
                let lib = device
                    .new_library_with_data(&vtlb)
                    .map_err(|_| IcbStatus::MetalFailed("icb_frc_dual_library_load"))?;
                let names = lib.function_names();
                if names.is_empty() {
                    return Err(IcbStatus::Args("icb_frc_dual_library_empty"));
                }
                for name in names.iter() {
                    let f = lib
                        .get_function(name, None)
                        .map_err(|_| IcbStatus::MetalFailed("icb_frc_dual_function_get"))?;
                    match function_type(f.as_ref()) {
                        MTL_FUNCTION_TYPE_MESH if mesh_fn.is_none() => mesh_fn = Some(f),
                        MTL_FUNCTION_TYPE_OBJECT if object_fn.is_none() => object_fn = Some(f),
                        _ if mesh_fn.is_none() && names.len() == 1 => mesh_fn = Some(f),
                        _ => {}
                    }
                }
                dual_lib_keep = Some(lib);
            }

            let Some(mesh_f) = mesh_fn else {
                return Err(IcbStatus::Args("icb_frc_no_mesh_function_resolved"));
            };
            let mdesc = MeshRenderPipelineDescriptor::new();
            mdesc.set_mesh_function(Some(mesh_f.as_ref()));
            if let Some(ref of) = object_fn {
                mdesc.set_object_function(Some(of.as_ref()));
            }
            mdesc.set_fragment_function(Some(&ff));
            crate::backend::metal::raw_metal::mesh_pipeline_set_support_indirect_command_buffers(
                mdesc.as_ref(),
                true,
            );
            if let Some(ca) = mdesc.color_attachments().object_at(0) {
                ca.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
            }
            // Keep mesh_f / object_fn / libraries alive until PSO is built.
            let pso = device
                .new_mesh_render_pipeline_state(&mdesc)
                .map_err(|_| IcbStatus::MetalFailed("icb_frc_mesh_pipeline_state"))?;
            drop(object_fn);
            drop(mesh_f);
            drop(mesh_lib_keep);
            drop(object_lib_keep);
            drop(dual_lib_keep);
            pso
        } else {
            let vert = load_fn(rp.vertex_func_ref)?;
            let vlib = device
                .new_library_with_data(&vert)
                .map_err(|_| IcbStatus::MetalFailed("icb_frc_vertex_library_load"))?;
            let vnames = vlib.function_names();
            if vnames.len() != 1 {
                return Err(IcbStatus::Args("icb_frc_vertex_function_count"));
            }
            let vf = vlib
                .get_function(&vnames[0], None)
                .map_err(|_| IcbStatus::MetalFailed("icb_frc_vertex_function_get"))?;
            let pdesc = RenderPipelineDescriptor::new();
            pdesc.set_vertex_function(Some(&vf));
            pdesc.set_fragment_function(Some(&ff));
            pdesc.set_support_indirect_command_buffers(true);
            if let Some(ca) = pdesc.color_attachments().object_at(0) {
                ca.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
            }
            // Tessellation PSO fields required for drawPatches / drawIndexedPatches
            // (metal-0.33 leaves these as TODOs — raw msg_send).
            if is_patches {
                let cp_index_ty = match fill.draw {
                    IcbRenderDraw::IndexedPatches { .. } => {
                        // UInt16 control-point indices (product fill stages type-1 bytes).
                        crate::backend::metal::raw_metal::MTL_TESSELLATION_CONTROL_POINT_INDEX_UINT16
                    }
                    _ => {
                        crate::backend::metal::raw_metal::MTL_TESSELLATION_CONTROL_POINT_INDEX_NONE
                    }
                };
                crate::backend::metal::raw_metal::configure_tessellation_pipeline(
                    pdesc.as_ref(),
                    16,
                    cp_index_ty,
                );
            }
            // Stage-in / control-point: type-7 vertex-input block → MTLVertexDescriptor.
            // Patch draws force PerPatchControlPoint when the layout does not already
            // carry a step function (host tessellation oracle fixture).
            if let Some(vd) =
                metal_vertex_descriptor_from_attrs_for_draw(&rp.vertex_attributes, is_patches)
            {
                pdesc.set_vertex_descriptor(Some(vd.as_ref()));
            }
            device
                .new_render_pipeline_state(&pdesc)
                .map_err(|_| IcbStatus::MetalFailed("icb_frc_render_pipeline_state"))?
        };
        Some(built)
    } else {
        None
    };

    // (index, stage, has_vertex_stride, stride, buffer)
    let mut staged: Vec<(u32, IcbRenderBindStage, bool, u64, metal::Buffer)> = Vec::new();
    for b in &fill.buffers {
        let stage = b.effective_stage();
        let bind = ComputeBufferBind {
            index: b.index,
            buffer_ref: b.buffer_ref,
            offset: b.offset,
            attribute_stride: b.attribute_stride,
            has_attribute_stride: b.has_attribute_stride,
        };
        let s = stage_buffer(state, host, task_id, &bind)
            .map_err(|_| IcbStatus::Missing("icb_frc_bind_stage_buffer"))?;
        let mtl = new_buffer_from_host(device, s.bytes.as_ptr(), s.bytes.len())
            .ok_or(IcbStatus::MetalFailed("icb_frc_bind_host_buffer"))?;
        staged.push((
            b.index,
            stage,
            b.has_attribute_stride && stage == IcbRenderBindStage::Vertex,
            b.attribute_stride,
            mtl,
        ));
    }

    // Stage index / patch / tessellation factor buffers by object-list ref.
    let stage_type1 = |buffer_ref: u32, offset: u64| -> Result<metal::Buffer, IcbStatus> {
        if buffer_ref == 0 {
            return Err(IcbStatus::Args("icb_frc_type1_ref_zero"));
        }
        let bind = ComputeBufferBind {
            index: 0,
            buffer_ref,
            offset,
            attribute_stride: 0,
            has_attribute_stride: false,
        };
        let s = stage_buffer(state, host, task_id, &bind)
            .map_err(|_| IcbStatus::Missing("icb_frc_type1_stage_buffer"))?;
        new_buffer_from_host(device, s.bytes.as_ptr(), s.bytes.len())
            .ok_or(IcbStatus::MetalFailed("icb_frc_type1_host_buffer"))
    };

    let index_mtl = match fill.draw {
        IcbRenderDraw::Indexed {
            index_buffer_ref,
            index_buffer_offset,
            index_type,
            index_count,
            ..
        } => {
            let elem = match index_type {
                0 => 2usize, // MTLIndexTypeUInt16
                1 => 4usize, // MTLIndexTypeUInt32
                _ => return Err(IcbStatus::Args("icb_frc_index_type_unknown")),
            };
            let need = (index_count as usize)
                .checked_mul(elem)
                .ok_or(IcbStatus::Args("icb_frc_index_span_overflow"))?;
            if need == 0 {
                return Err(IcbStatus::Args("icb_frc_index_span_zero"));
            }
            let mtl = stage_type1(index_buffer_ref, index_buffer_offset)?;
            // Product stages the index window at offset 0 in the retained buffer.
            let _ = need;
            Some(mtl)
        }
        IcbRenderDraw::Primitives { .. }
        | IcbRenderDraw::Patches { .. }
        | IcbRenderDraw::IndexedPatches { .. }
        | IcbRenderDraw::MeshThreads(_)
        | IcbRenderDraw::MeshThreadgroups(_) => None,
    };

    // Optional patch-index buffer (nullable in Metal API).
    let patch_index_mtl = match fill.draw {
        IcbRenderDraw::Patches {
            patch_index_buffer_ref,
            patch_index_buffer_offset,
            ..
        }
        | IcbRenderDraw::IndexedPatches {
            patch_index_buffer_ref,
            patch_index_buffer_offset,
            ..
        } if patch_index_buffer_ref != 0 => Some(stage_type1(
            patch_index_buffer_ref,
            patch_index_buffer_offset,
        )?),
        _ => None,
    };

    let control_point_index_mtl = match fill.draw {
        IcbRenderDraw::IndexedPatches {
            control_point_index_buffer_ref,
            control_point_index_buffer_offset,
            ..
        } => Some(stage_type1(
            control_point_index_buffer_ref,
            control_point_index_buffer_offset,
        )?),
        _ => None,
    };

    // Tessellation factor buffer is required by Metal for drawPatches variants.
    let tess_factor_mtl = match fill.draw {
        IcbRenderDraw::Patches {
            tessellation_factor,
            ..
        }
        | IcbRenderDraw::IndexedPatches {
            tessellation_factor,
            ..
        } => {
            if tessellation_factor.buffer_ref == 0 {
                return Err(IcbStatus::Args("icb_frc_tess_factor_ref_zero"));
            }
            Some(stage_type1(
                tessellation_factor.buffer_ref,
                tessellation_factor.offset,
            )?)
        }
        _ => None,
    };

    let mut cache = icb_cache().lock();
    let entry = cache
        .get_mut(&(task_id, icb_ref))
        .ok_or(IcbStatus::Missing("icb_frc_not_cached"))?;
    if fill.command_index as u64 >= entry.icb.size() {
        return Err(IcbStatus::Args("icb_frc_command_index_past_capacity"));
    }
    let cmd = entry
        .icb
        .indirect_render_command_at_index(fill.command_index as u64);
    if let Some(ref pso) = pso {
        cmd.set_render_pipeline_state(pso);
    }
    // When inheritBuffers, vertex/fragment buffers come from the parent encoder
    // at execute (see metal_draw::encode_icb_execute_and_writeback).
    if !entry.desc.inherit_buffers {
        for (idx, stage, has_stride, stride, mtl) in &staged {
            match stage {
                IcbRenderBindStage::Fragment => {
                    cmd.set_fragment_buffer(*idx as u64, Some(mtl.as_ref()), 0);
                }
                IcbRenderBindStage::Mesh => {
                    crate::backend::metal::raw_metal::icb_set_mesh_buffer(
                        cmd,
                        Some(mtl.as_ref()),
                        0,
                        *idx as u64,
                    );
                }
                IcbRenderBindStage::Object => {
                    crate::backend::metal::raw_metal::icb_set_object_buffer(
                        cmd,
                        Some(mtl.as_ref()),
                        0,
                        *idx as u64,
                    );
                }
                IcbRenderBindStage::Vertex if *has_stride => {
                    crate::backend::metal::raw_metal::icb_set_vertex_buffer_attribute_stride(
                        cmd,
                        Some(mtl.as_ref()),
                        0,
                        *stride,
                        *idx as u64,
                    );
                }
                IcbRenderBindStage::Vertex => {
                    cmd.set_vertex_buffer(*idx as u64, Some(mtl.as_ref()), 0);
                }
            }
        }
    }
    // Object-stage threadgroup memory (mesh pipelines with objectFunction).
    for tg in &fill.object_threadgroup_memory {
        // Metal requires length multiple of 16 when non-zero (same as compute TG).
        if tg.length != 0 && tg.length % 16 != 0 {
            return Err(IcbStatus::Args("icb_frc_object_tg_length_alignment"));
        }
        crate::backend::metal::raw_metal::icb_set_object_threadgroup_memory_length(
            cmd,
            tg.length,
            tg.index as u64,
        );
    }
    match fill.draw {
        IcbRenderDraw::Primitives {
            primitive_type,
            vertex_start,
            vertex_count,
            instance_count,
            base_instance,
        } => {
            let prim = icb_primitive_type(primitive_type, "icb_frc_draw_primitive_type")?;
            cmd.draw_primitives(
                prim,
                vertex_start,
                vertex_count,
                instance_count,
                base_instance,
            );
        }
        IcbRenderDraw::Indexed {
            primitive_type,
            index_type,
            index_count,
            index_buffer_offset: _,
            instance_count,
            base_vertex,
            base_instance,
            ..
        } => {
            let prim = icb_primitive_type(primitive_type, "icb_frc_indexed_primitive_type")?;
            let ity = match index_type {
                0 => MTLIndexType::UInt16,
                1 => MTLIndexType::UInt32,
                _ => return Err(IcbStatus::Args("icb_frc_indexed_index_type")),
            };
            let idx_buf = index_mtl
                .as_ref()
                .ok_or(IcbStatus::Missing("icb_frc_indexed_no_index_buffer"))?;
            // SDK: baseVertex is NSInteger (signed). Wire stores a u64 bit pattern
            // of the signed value (ld64 as i64). metal-0.33 types the ICB method
            // as NSUInteger — use raw msg_send with NSInteger.
            // Fail-closed only when the value does not fit NSInteger (platform width).
            let base_vertex_ns = base_vertex as metal::NSInteger;
            if base_vertex_ns as i64 != base_vertex {
                return Err(IcbStatus::Args("icb_frc_base_vertex_range"));
            }
            crate::backend::metal::raw_metal::icb_draw_indexed_primitives(
                cmd,
                prim,
                index_count,
                ity,
                idx_buf.as_ref(),
                0, // staged window starts at offset 0
                instance_count,
                base_vertex_ns,
                base_instance,
            );
        }
        IcbRenderDraw::Patches {
            number_of_patch_control_points,
            patch_start,
            patch_count,
            instance_count,
            base_instance,
            tessellation_factor,
            ..
        } => {
            if patch_count == 0 || number_of_patch_control_points == 0 {
                return Err(IcbStatus::Args("icb_frc_patches_zero_count"));
            }
            let tess = tess_factor_mtl
                .as_ref()
                .ok_or(IcbStatus::Missing("icb_frc_patches_no_tess_buffer"))?;
            // patchIndexBuffer is nullable in the SDK; product uses raw msg_send.
            crate::backend::metal::raw_metal::icb_draw_patches(
                cmd,
                number_of_patch_control_points as u64,
                patch_start,
                patch_count,
                patch_index_mtl.as_ref().map(|b| b.as_ref()),
                0,
                instance_count,
                base_instance,
                tess.as_ref(),
                0,
                tessellation_factor.instance_stride,
            );
        }
        IcbRenderDraw::IndexedPatches {
            number_of_patch_control_points,
            patch_start,
            patch_count,
            instance_count,
            base_instance,
            tessellation_factor,
            ..
        } => {
            if patch_count == 0 || number_of_patch_control_points == 0 {
                return Err(IcbStatus::Args("icb_frc_indexed_patches_zero_count"));
            }
            let tess = tess_factor_mtl
                .as_ref()
                .ok_or(IcbStatus::Missing("icb_frc_indexed_patches_no_tess_buffer"))?;
            let cp = control_point_index_mtl.as_ref().ok_or(IcbStatus::Missing(
                "icb_frc_indexed_patches_no_control_points",
            ))?;
            // patchIndexBuffer is nullable in the SDK.
            crate::backend::metal::raw_metal::icb_draw_indexed_patches(
                cmd,
                number_of_patch_control_points as u64,
                patch_start,
                patch_count,
                patch_index_mtl.as_ref().map(|b| b.as_ref()),
                0,
                cp.as_ref(),
                0,
                instance_count,
                base_instance,
                tess.as_ref(),
                0,
                tessellation_factor.instance_stride,
            );
        }
        IcbRenderDraw::MeshThreads(mesh) | IcbRenderDraw::MeshThreadgroups(mesh) => {
            use crate::backend::metal::raw_metal;
            let threads = matches!(fill.draw, IcbRenderDraw::MeshThreads(_));
            if mesh.grid[0] == 0 || mesh.mesh_tg[0] == 0 {
                return Err(IcbStatus::Args(if threads {
                    "icb_frc_mesh_threads_zero_dims"
                } else {
                    "icb_frc_mesh_threadgroups_zero_dims"
                }));
            }
            let size = |d: [u32; 3]| raw_metal::mtl_size(d[0] as u64, d[1] as u64, d[2] as u64);
            let grid = size(mesh.grid);
            // object TG size is still required by the selector when objectFunction is nil.
            let obj_tg = size(mesh.object_tg.map(|d| d.max(1)));
            let mesh_tg = size(mesh.mesh_tg);
            if threads {
                raw_metal::icb_draw_mesh_threads(cmd, grid, obj_tg, mesh_tg);
            } else {
                raw_metal::icb_draw_mesh_threadgroups(cmd, grid, obj_tg, mesh_tg);
            }
        }
    }
    if let Some(pso) = pso {
        entry.retained_psos_render.push(pso);
    }
    for (_, _, _, _, mtl) in staged {
        entry.retained_buffers.push(mtl);
    }
    if let Some(mtl) = index_mtl {
        entry.retained_buffers.push(mtl);
    }
    if let Some(mtl) = patch_index_mtl {
        entry.retained_buffers.push(mtl);
    }
    if let Some(mtl) = control_point_index_mtl {
        entry.retained_buffers.push(mtl);
    }
    if let Some(mtl) = tess_factor_mtl {
        entry.retained_buffers.push(mtl);
    }
    entry.has_fills = true;
    Ok(())
}

#[cfg(feature = "backend-vulkan")]
pub fn fill_render_command<M: HostMemory + HostOps>(
    _state: &DeviceState,
    _host: &M,
    _task_id: u32,
    _icb_ref: u32,
    _fill: &IcbRenderFill,
) -> Result<(), IcbStatus> {
    Err(IcbStatus::NoMetal("icb_frc_no_vulkan_path"))
}

#[cfg(feature = "backend-vulkan")]
pub fn fill_icb_from_command_memory<M: HostMemory + HostOps>(
    _state: &DeviceState,
    _host: &M,
    _task_id: u32,
    _icb_ref: u32,
    _range_location: u64,
    _range_length: u64,
) -> Result<(), IcbStatus> {
    Err(IcbStatus::NoMetal("icb_fill_no_vulkan_path"))
}

#[cfg(feature = "backend-vulkan")]
pub fn resolve_metal_icb<M: HostMemory + HostOps>(
    _state: &DeviceState,
    _host: &M,
    _task_id: u32,
    _icb_ref: u32,
) -> Result<(IndirectCommandBufferDescriptor, ()), IcbStatus> {
    Err(IcbStatus::NoMetal("icb_resolve_no_vulkan_path"))
}

/// Clone writeback slots for a cached ICB into a session nested job (after execute).
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
pub fn export_icb_writeback_job(
    task_id: u32,
    icb_ref: u32,
) -> Option<crate::runtime::compute_exec::NestedDispatchJob> {
    use crate::runtime::compute_exec::{nested_job_from_icb_buffers, StagedBuffer};

    let cache = icb_cache().lock();
    let entry = cache.get(&(task_id, icb_ref))?;
    if entry.writebacks.is_empty() {
        return None;
    }
    let mut staged = Vec::with_capacity(entry.writebacks.len());
    let mut mtl = Vec::with_capacity(entry.writebacks.len());
    for w in &entry.writebacks {
        staged.push(StagedBuffer {
            bind: w.bind.clone(),
            gva: w.gva,
            bytes: vec![0u8; w.len],
            pages: w.pages.clone(),
        });
        mtl.push(w.mtl.clone());
    }
    Some(nested_job_from_icb_buffers(staged, mtl))
}

/// Build a compute PSO with `supportIndirectCommandBuffers` (required for ICB
/// fills and for parent-encoder `inheritPipelineState`).
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
pub(crate) fn new_icb_compute_pso(
    device: &metal::Device,
    mtlb: &[u8],
) -> Result<metal::ComputePipelineState, IcbStatus> {
    use metal::ComputePipelineDescriptor;

    // Load sole function from MTLB (same contract as product compute path).
    let library = device
        .new_library_with_data(mtlb)
        .map_err(|_| IcbStatus::MetalFailed("icb_pso_library_load"))?;
    let names = library.function_names();
    if names.len() != 1 {
        return Err(IcbStatus::Args("icb_pso_function_count"));
    }
    let function = library
        .get_function(&names[0], None)
        .map_err(|_| IcbStatus::MetalFailed("icb_pso_function_get"))?;
    let desc = ComputePipelineDescriptor::new();
    desc.set_compute_function(Some(&function));
    desc.set_support_indirect_command_buffers(true);
    device
        .new_compute_pipeline_state(&desc)
        .map_err(|_| IcbStatus::MetalFailed("icb_pso_pipeline_state"))
}

/// Fill one compute command slot on a cached host ICB from guest object-list state.
///
/// Mirrors Metal: `indirectComputeCommandAtIndex` → set PSO / kernel buffers /
/// concurrent dispatch. Stages type-1 buffer contents into shared Metal buffers
/// and records GVA writebacks for post-execute flush.
///
/// When the ICB was created with `inheritPipelineState` / `inheritBuffers`, those
/// resources are **not** recorded into the slot — the parent compute encoder
/// supplies them at `executeCommandsInBuffer` (see
/// [`crate::runtime::compute_session::ComputeSession::encode_icb`]).
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
pub fn fill_compute_command<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    icb_ref: u32,
    fill: &IcbComputeFill,
) -> Result<(), IcbStatus> {
    use crate::backend::metal::raw_metal::mtl_size;
    use crate::backend::metal::runtime::{new_buffer_from_host, system_device};
    use crate::runtime::compute_exec::{
        load_compute_pipeline, load_mtlb, stage_buffer, ComputeBufferBind,
    };

    if icb_ref == 0 {
        return Err(IcbStatus::Args("icb_fcc_ref_zero"));
    }
    let Some(device) = system_device() else {
        return Err(IcbStatus::NoMetal("icb_fcc_no_metal"));
    };

    // Ensure the host ICB exists in the cache; need create flags before staging.
    let (desc, _) = resolve_metal_icb(state, host, task_id, icb_ref)?;
    let mut fill_resolved = fill.clone();
    resolve_compute_fill_offsets(state, host, task_id, &mut fill_resolved)?;
    let fill = &fill_resolved;

    // Pipeline is required on the ICB command unless inheritPipelineState.
    let pso = if !desc.inherit_pipeline_state {
        if fill.pipeline_ref == 0 {
            return Err(IcbStatus::Args("icb_fcc_pipeline_ref_zero"));
        }
        let pipeline = load_compute_pipeline(state, host, task_id, fill.pipeline_ref)
            .ok_or(IcbStatus::Missing("icb_fcc_pipeline_load"))?;
        let mtlb = load_mtlb(state, host, task_id, pipeline.kernel_func_ref)
            .ok_or(IcbStatus::Missing("icb_fcc_mtlb_load"))?;
        Some(new_icb_compute_pso(device, &mtlb)?)
    } else {
        None
    };

    // Kernel buffers: stage only when not inheritBuffers (parent encoder owns them).
    let mut staged_binds: Vec<(
        u32,
        bool,
        u64,
        metal::Buffer,
        crate::runtime::compute_exec::StagedBuffer,
    )> = Vec::new();
    if !desc.inherit_buffers {
        for b in &fill.buffers {
            if b.buffer_ref == 0 {
                return Err(IcbStatus::Args("icb_fcc_bind_ref_zero"));
            }
            let bind = ComputeBufferBind {
                index: b.index,
                buffer_ref: b.buffer_ref,
                offset: b.offset,
                attribute_stride: b.attribute_stride,
                has_attribute_stride: b.has_attribute_stride,
            };
            // The slug is dropped by this remap, not lost: `stage_buffer`
            // fail-logs the check that refused before returning, so the line is
            // already on the sink under `compute_stage_buf`. `IcbStatus` gets
            // its own vocabulary when it is registered.
            let staged = stage_buffer(state, host, task_id, &bind).map_err(|e| match e {
                crate::runtime::compute_exec::ComputeStatus::MissingBuffer(_) => {
                    IcbStatus::Missing("icb_fcc_bind_stage_missing")
                }
                crate::runtime::compute_exec::ComputeStatus::GuestIo(_) => {
                    IcbStatus::MetalFailed("icb_fcc_bind_stage_guest_io")
                }
                _ => IcbStatus::Args("icb_fcc_bind_stage_other"),
            })?;
            let mtl = new_buffer_from_host(device, staged.bytes.as_ptr(), staged.bytes.len())
                .ok_or(IcbStatus::MetalFailed("icb_fcc_bind_host_buffer"))?;
            staged_binds.push((
                b.index,
                b.has_attribute_stride,
                b.attribute_stride,
                mtl,
                staged,
            ));
        }
    }

    let mut cache = icb_cache().lock();
    let entry = cache
        .get_mut(&(task_id, icb_ref))
        .ok_or(IcbStatus::Missing("icb_fcc_not_cached"))?;
    if fill.command_index as u64 >= entry.icb.size() {
        return Err(IcbStatus::Args("icb_fcc_command_index_past_capacity"));
    }
    // maxKernelBufferBindCount: reject binds past the create descriptor.
    if !entry.desc.inherit_buffers {
        for (idx, _, _, _, _) in &staged_binds {
            if *idx as u64 >= entry.desc.max_kernel_buffer_bind_count as u64 {
                return Err(IcbStatus::Args("icb_fcc_bind_index_past_max"));
            }
        }
    }

    let cmd = entry
        .icb
        .indirect_compute_command_at_index(fill.command_index as u64);
    if let Some(ref pso) = pso {
        cmd.set_compute_pipeline_state(pso);
    }
    // When inheritBuffers, kernel buffers come from the parent compute encoder.
    if !entry.desc.inherit_buffers {
        for (idx, has_stride, stride, mtl, _) in &staged_binds {
            if *has_stride {
                crate::backend::metal::raw_metal::icb_set_kernel_buffer_attribute_stride(
                    cmd,
                    Some(mtl.as_ref()),
                    0,
                    *stride,
                    *idx as u64,
                );
            } else {
                cmd.set_kernel_buffer(*idx as u64, Some(mtl.as_ref()), 0);
            }
        }
    }
    for tg in &fill.threadgroup_memory {
        // Metal requires length multiple of 16 when non-zero; zero clears.
        if tg.length != 0 && tg.length % 16 != 0 {
            return Err(IcbStatus::Args("icb_fcc_tg_length_alignment"));
        }
        cmd.set_threadgroup_memory_length(tg.index as u64, tg.length);
    }
    if fill.barrier {
        cmd.set_barrier();
    } else {
        cmd.clear_barrier();
    }
    match fill.dispatch {
        IcbFillDispatch::ConcurrentThreadgroups {
            grid_x,
            grid_y,
            grid_z,
            tg_x,
            tg_y,
            tg_z,
        } => {
            if grid_x == 0 || grid_y == 0 || grid_z == 0 || tg_x == 0 || tg_y == 0 || tg_z == 0 {
                return Err(IcbStatus::Args("icb_fcc_threadgroups_zero_dims"));
            }
            cmd.concurrent_dispatch_threadgroups(
                mtl_size(grid_x as u64, grid_y as u64, grid_z as u64),
                mtl_size(tg_x as u64, tg_y as u64, tg_z as u64),
            );
        }
        IcbFillDispatch::ConcurrentThreads {
            threads_x,
            threads_y,
            threads_z,
            tg_x,
            tg_y,
            tg_z,
        } => {
            if threads_x == 0
                || threads_y == 0
                || threads_z == 0
                || tg_x == 0
                || tg_y == 0
                || tg_z == 0
            {
                return Err(IcbStatus::Args("icb_fcc_threads_zero_dims"));
            }
            cmd.concurrent_dispatch_threads(
                mtl_size(threads_x as u64, threads_y as u64, threads_z as u64),
                mtl_size(tg_x as u64, tg_y as u64, tg_z as u64),
            );
        }
    }

    if let Some(pso) = pso {
        entry.retained_psos.push(pso);
    }
    // Writebacks only for buffers recorded into the ICB (not inheritBuffers).
    for (_, _, _, mtl, staged) in staged_binds {
        entry.writebacks.push(IcbWriteback {
            bind: staged.bind.clone(),
            gva: staged.gva,
            len: staged.bytes.len(),
            pages: staged.pages.clone(),
            mtl: mtl.clone(),
        });
        entry.retained_buffers.push(mtl);
    }
    entry.has_fills = true;
    Ok(())
}

#[cfg(feature = "backend-vulkan")]
pub fn fill_compute_command<M: HostMemory + HostOps>(
    _state: &DeviceState,
    _host: &M,
    _task_id: u32,
    _icb_ref: u32,
    _fill: &IcbComputeFill,
) -> Result<(), IcbStatus> {
    Err(IcbStatus::NoMetal("icb_fcc_no_vulkan_path"))
}

#[cfg(test)]
mod tests;
