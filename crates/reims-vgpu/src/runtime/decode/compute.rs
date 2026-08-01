//! Compute command decoder (port of `host/utils/reims-vgpu-compute-decode`).

use crate::contract::endian::{ld16, ld32, ld64};

pub const MAX_BIND_ENTRIES: usize = 128;
pub const HEADER_LEN: usize = 8;
pub const SIZE3_SIZE: usize = 24;

pub const OP_USE_HEAPS: u32 = 0x86;
pub const OP_USE_RESOURCES: u32 = 0x87;
pub const OP_DISPATCH_THREADGROUPS: u32 = 0xc8;
pub const OP_DISPATCH_THREADGROUPS_INDIRECT: u32 = 0xc9;
pub const OP_DISPATCH_THREADS: u32 = 0xca;
pub const OP_SET_BUFFERS: u32 = 0xcb;
pub const OP_SET_SAMPLERS: u32 = 0xcc;
pub const OP_SET_SAMPLERS_LOD: u32 = 0xcd;
pub const OP_SET_TEXTURES: u32 = 0xce;
pub const OP_SET_BUFFER_OFFSET: u32 = 0xcf;
pub const OP_SET_PIPELINE: u32 = 0xd0;
pub const OP_SET_STAGE_IN_REGION: u32 = 0xd1;
pub const OP_SET_STAGE_IN_REGION_INDIRECT: u32 = 0xd2;
pub const OP_SET_THREADGROUP_MEMORY_LENGTH: u32 = 0xd3;
pub const OP_UPDATE_FENCE: u32 = 0xd4;
pub const OP_WAIT_FENCE: u32 = 0xd5;
pub const OP_BARRIER_RESOURCES: u32 = 0xd6;
pub const OP_BARRIER_SCOPE: u32 = 0xd7;
pub const OP_SET_IMAGEBLOCK_DIMENSIONS: u32 = 0xd8;
pub const OP_SET_BUFFERS_ATTRIBUTE_STRIDE: u32 = 0xd9;
pub const OP_SET_BUFFER_OFFSET_ATTRIBUTE_STRIDE: u32 = 0xda;
pub const OP_DISPATCH_TYPE: u32 = 0xdb;
pub const OP_ENCODE_START_DO_WHILE: u32 = 0xdc;
pub const OP_ENCODE_END_DO_WHILE: u32 = 0xdd;
pub const OP_ENCODE_START_WHILE: u32 = 0xde;
pub const OP_ENCODE_END_WHILE: u32 = 0xdf;
pub const OP_ENCODE_START_IF: u32 = 0xe0;
pub const OP_ENCODE_START_ELSE: u32 = 0xe1;
pub const OP_ENCODE_END_IF: u32 = 0xe2;
pub const OP_INSERT_COMPRESSED_TEXTURE_FLUSH: u32 = 0xe3;
pub const OP_EXECUTE_COMMANDS_IN_BUFFER: u32 = 0xe4;
pub const OP_EXECUTE_COMMANDS_IN_BUFFER_INDIRECT: u32 = 0xe5;
pub const OP_DISPATCH_THREADS_INDIRECT: u32 = 0xe6;

pub const REJECTED_85: u32 = 0x85;
pub const REJECTED_88: u32 = 0x88;
pub const REJECTED_C7: u32 = 0xc7;

const COUNT_BASE: usize = HEADER_LEN + 4;
const BIND_BASE: usize = HEADER_LEN + 8;
const REF_SIZE: usize = 4;
const BUF_ENTRY: usize = 12;
const BUF_STRIDE_ENTRY: usize = 20;
const SAMPLER_LOD_ENTRY: usize = 12;
const BUF_OFF_LEN: usize = 0x14;
const BUF_OFF_STRIDE_LEN: usize = 0x1c;
const DISPATCH_DIRECT_LEN: usize = HEADER_LEN + 2 * SIZE3_SIZE;
const DISPATCH_INDIRECT_LEN: usize = HEADER_LEN + SIZE3_SIZE + 8 + 4;
const DISPATCH_THREADS_INDIRECT_LEN: usize = 0x14;
const STAGE_IN_LEN: usize = HEADER_LEN + 2 * SIZE3_SIZE;
const STAGE_IN_INDIRECT_LEN: usize = 0x14;
const TG_MEM_LEN: usize = 0x14;
const FENCE_LEN: usize = HEADER_LEN + 4;
const BARRIER_SCOPE_LEN: usize = HEADER_LEN + 4;
const IMAGEBLOCK_LEN: usize = 0x10;
const DISPATCH_TYPE_LEN: usize = HEADER_LEN + 4;
const CONDITION_LEN: usize = 0x1c;
const EXECUTE_LEN: usize = 0x1c;
const EXECUTE_INDIRECT_LEN: usize = 0x18;
const EMPTY_LEN: usize = HEADER_LEN;
const PIPELINE_LEN: usize = HEADER_LEN + 4;

/// Why the compute decoder refused a command.
///
/// No `Ok` and no `ErrArgs`, for the reason recorded on `blit::DecodeStatus`:
/// success is the result's own `Ok`, and a bad argument here is a payload
/// shorter than the field, which `ErrShort` already names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeStatus {
    ErrShort,
    ErrUnknownOpcode,
    ErrUnsupportedOpcode,
    ErrTooManyBindings,
}

impl crate::observe::Refusal for DecodeStatus {
    /// Slugs carry a `compute_decode_` prefix: seven modules under
    /// `runtime/decode/` define a type called `DecodeStatus`, and five of them
    /// have an `ErrShort` that means a different read. Without the prefix the
    /// crate-wide uniqueness gate could not tell the compute decoder's refusals
    /// from any other's.
    fn refusal(&self) -> Option<&'static str> {
        Some(match self {
            Self::ErrShort => "compute_decode_short",
            Self::ErrUnknownOpcode => "compute_decode_unknown_opcode",
            Self::ErrUnsupportedOpcode => "compute_decode_unsupported_opcode",
            Self::ErrTooManyBindings => "compute_decode_too_many_bindings",
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum OpcodeConfidence {
    #[default]
    Unknown = 0,
    AppleEmittedConfirmed,
    AppleRejected,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Kind {
    #[default]
    Unknown = 0,
    UseHeaps,
    UseResources,
    Pipeline,
    BufferBind,
    BufferOffset,
    TextureBind,
    SamplerBind,
    SamplerLod,
    DispatchThreadgroups,
    DispatchThreadgroupsIndirect,
    DispatchThreads,
    StageInRegion,
    StageInRegionIndirect,
    ThreadgroupMemory,
    UpdateFence,
    WaitFence,
    BarrierResources,
    BarrierScope,
    ImageblockDimensions,
    BufferBindAttributeStride,
    BufferOffsetAttributeStride,
    DispatchType,
    DispatchThreadsIndirect,
    ControlStartDoWhile,
    ControlEndDoWhile,
    ControlStartWhile,
    ControlEndWhile,
    ControlStartIf,
    ControlStartElse,
    ControlEndIf,
    CompressedTextureFlush,
    ExecuteCommandsInBuffer,
    ExecuteCommandsInBufferIndirect,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Size3 {
    pub x: u64,
    pub y: u64,
    pub z: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BufferBinding {
    pub ref_: u32,
    pub offset: u64,
    pub attribute_stride: u64,
    pub has_attribute_stride: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RefBinding {
    pub ref_: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SamplerBinding {
    pub ref_: u32,
    pub lod_min_bits: u32,
    pub lod_max_bits: u32,
    pub has_lod_clamp: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Region3 {
    pub origin: Size3,
    pub size: Size3,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Command {
    pub opcode: u32,
    pub command_length: u32,
    pub kind: Kind,
    pub confidence: OpcodeConfidence,
    pub pipeline_ref: u32,
    pub first: u32,
    pub count: u32,
    pub buffers: Vec<BufferBinding>,
    pub textures: Vec<RefBinding>,
    pub samplers: Vec<SamplerBinding>,
    pub resources: Vec<RefBinding>,
    pub heaps: Vec<RefBinding>,
    pub grid: Size3,
    pub threads_per_threadgroup: Size3,
    pub indirect_buffer_ref: u32,
    pub indirect_buffer_offset: u64,
    pub buffer_offset: u64,
    pub attribute_stride: u64,
    pub imageblock_width: u32,
    pub imageblock_height: u32,
    pub dispatch_type: u32,
    pub stage_in_region: Region3,
    pub stage_in_indirect_buffer_ref: u32,
    pub stage_in_indirect_buffer_offset: u64,
    pub threadgroup_memory_length: u64,
    pub threadgroup_memory_index: u32,
    pub resource_usage: u32,
    pub fence_ref: u32,
    pub barrier_scope: u16,
    pub barrier_scope_reserved: u16,
    pub condition_buffer_ref: u32,
    pub condition_buffer_offset: u64,
    pub condition_comparison: u32,
    pub condition_reference_value: u32,
    pub indirect_command_buffer_ref: u32,
    pub indirect_command_range_location: u64,
    pub indirect_command_range_length: u64,
    pub indirect_command_arguments_buffer_ref: u32,
    pub indirect_command_arguments_buffer_offset: u64,
}

pub fn opcode_supported(opcode: u32) -> bool {
    matches!(
        opcode,
        OP_USE_HEAPS
            | OP_USE_RESOURCES
            | OP_DISPATCH_THREADGROUPS
            | OP_DISPATCH_THREADGROUPS_INDIRECT
            | OP_DISPATCH_THREADS
            | OP_SET_BUFFERS
            | OP_SET_SAMPLERS
            | OP_SET_SAMPLERS_LOD
            | OP_SET_TEXTURES
            | OP_SET_BUFFER_OFFSET
            | OP_SET_PIPELINE
            | OP_SET_STAGE_IN_REGION
            | OP_SET_STAGE_IN_REGION_INDIRECT
            | OP_SET_THREADGROUP_MEMORY_LENGTH
            | OP_UPDATE_FENCE
            | OP_WAIT_FENCE
            | OP_BARRIER_RESOURCES
            | OP_BARRIER_SCOPE
            | OP_SET_IMAGEBLOCK_DIMENSIONS
            | OP_SET_BUFFERS_ATTRIBUTE_STRIDE
            | OP_SET_BUFFER_OFFSET_ATTRIBUTE_STRIDE
            | OP_DISPATCH_TYPE
            | OP_ENCODE_START_DO_WHILE
            | OP_ENCODE_END_DO_WHILE
            | OP_ENCODE_START_WHILE
            | OP_ENCODE_END_WHILE
            | OP_ENCODE_START_IF
            | OP_ENCODE_START_ELSE
            | OP_ENCODE_END_IF
            | OP_INSERT_COMPRESSED_TEXTURE_FLUSH
            | OP_EXECUTE_COMMANDS_IN_BUFFER
            | OP_EXECUTE_COMMANDS_IN_BUFFER_INDIRECT
            | OP_DISPATCH_THREADS_INDIRECT
    )
}

pub fn opcode_apple_rejected(opcode: u32) -> bool {
    matches!(opcode, REJECTED_85 | REJECTED_88 | REJECTED_C7)
}

pub fn opcode_confidence(opcode: u32) -> OpcodeConfidence {
    if opcode_supported(opcode) {
        OpcodeConfidence::AppleEmittedConfirmed
    } else if opcode_apple_rejected(opcode) {
        OpcodeConfidence::AppleRejected
    } else {
        OpcodeConfidence::Unknown
    }
}

fn decode_size3(p: &[u8]) -> Size3 {
    Size3 {
        x: ld64(&p[0..]),
        y: ld64(&p[8..]),
        z: ld64(&p[16..]),
    }
}

fn var_len(cmd_len: usize, base: usize, count: u32, stride: usize) -> bool {
    let expected = base as u64 + (count as u64) * (stride as u64);
    expected <= u32::MAX as u64 && cmd_len == expected as usize
}

/// Transactional compute command decode.
pub fn decode(command: &[u8]) -> Result<Command, DecodeStatus> {
    if command.len() < HEADER_LEN {
        return Err(DecodeStatus::ErrShort);
    }
    let opcode = ld32(&command[0..]);
    let command_length = ld32(&command[4..]) as usize;
    let confidence = opcode_confidence(opcode);
    if command_length < HEADER_LEN || command_length > command.len() {
        return Err(DecodeStatus::ErrShort);
    }
    if confidence == OpcodeConfidence::Unknown {
        return Err(DecodeStatus::ErrUnknownOpcode);
    }
    if !opcode_supported(opcode) {
        return Err(DecodeStatus::ErrUnsupportedOpcode);
    }
    let payload = &command[HEADER_LEN..command_length];
    let mut out = Command {
        opcode,
        command_length: command_length as u32,
        confidence,
        ..Default::default()
    };

    match opcode {
        OP_USE_HEAPS => {
            if command_length < COUNT_BASE {
                return Err(DecodeStatus::ErrShort);
            }
            let count = ld32(&payload[0..]);
            if count as usize > MAX_BIND_ENTRIES {
                return Err(DecodeStatus::ErrTooManyBindings);
            }
            if !var_len(command_length, COUNT_BASE, count, REF_SIZE) {
                return Err(DecodeStatus::ErrShort);
            }
            out.kind = Kind::UseHeaps;
            out.count = count;
            for i in 0..count as usize {
                out.heaps.push(RefBinding {
                    ref_: ld32(&payload[4 + i * REF_SIZE..]),
                });
            }
            Ok(out)
        }
        OP_USE_RESOURCES => {
            if command_length < BIND_BASE {
                return Err(DecodeStatus::ErrShort);
            }
            let count = ld32(&payload[0..]);
            if count as usize > MAX_BIND_ENTRIES {
                return Err(DecodeStatus::ErrTooManyBindings);
            }
            if !var_len(command_length, BIND_BASE, count, REF_SIZE) {
                return Err(DecodeStatus::ErrShort);
            }
            out.kind = Kind::UseResources;
            out.count = count;
            out.resource_usage = ld32(&payload[4..]);
            for i in 0..count as usize {
                out.resources.push(RefBinding {
                    ref_: ld32(&payload[8 + i * REF_SIZE..]),
                });
            }
            Ok(out)
        }
        OP_SET_PIPELINE => {
            if command_length != PIPELINE_LEN {
                return Err(DecodeStatus::ErrShort);
            }
            out.kind = Kind::Pipeline;
            out.pipeline_ref = ld32(payload);
            Ok(out)
        }
        OP_SET_BUFFERS => {
            if command_length < BIND_BASE {
                return Err(DecodeStatus::ErrShort);
            }
            let count = ld32(&payload[4..]);
            if count as usize > MAX_BIND_ENTRIES {
                return Err(DecodeStatus::ErrTooManyBindings);
            }
            if !var_len(command_length, BIND_BASE, count, BUF_ENTRY) {
                return Err(DecodeStatus::ErrShort);
            }
            out.kind = Kind::BufferBind;
            out.first = ld32(&payload[0..]);
            out.count = count;
            for i in 0..count as usize {
                let e = &payload[8 + i * BUF_ENTRY..];
                out.buffers.push(BufferBinding {
                    ref_: ld32(&e[0..]),
                    offset: ld64(&e[4..]),
                    ..Default::default()
                });
            }
            Ok(out)
        }
        OP_SET_BUFFERS_ATTRIBUTE_STRIDE => {
            if command_length < BIND_BASE {
                return Err(DecodeStatus::ErrShort);
            }
            let count = ld32(&payload[4..]);
            if count as usize > MAX_BIND_ENTRIES {
                return Err(DecodeStatus::ErrTooManyBindings);
            }
            if !var_len(command_length, BIND_BASE, count, BUF_STRIDE_ENTRY) {
                return Err(DecodeStatus::ErrShort);
            }
            out.kind = Kind::BufferBindAttributeStride;
            out.first = ld32(&payload[0..]);
            out.count = count;
            for i in 0..count as usize {
                let e = &payload[8 + i * BUF_STRIDE_ENTRY..];
                out.buffers.push(BufferBinding {
                    ref_: ld32(&e[0..]),
                    offset: ld64(&e[4..]),
                    attribute_stride: ld64(&e[12..]),
                    has_attribute_stride: true,
                });
            }
            Ok(out)
        }
        OP_SET_SAMPLERS => {
            if command_length < BIND_BASE {
                return Err(DecodeStatus::ErrShort);
            }
            let count = ld32(&payload[4..]);
            if count as usize > MAX_BIND_ENTRIES {
                return Err(DecodeStatus::ErrTooManyBindings);
            }
            if !var_len(command_length, BIND_BASE, count, REF_SIZE) {
                return Err(DecodeStatus::ErrShort);
            }
            out.kind = Kind::SamplerBind;
            out.first = ld32(&payload[0..]);
            out.count = count;
            for i in 0..count as usize {
                out.samplers.push(SamplerBinding {
                    ref_: ld32(&payload[8 + i * REF_SIZE..]),
                    ..Default::default()
                });
            }
            Ok(out)
        }
        OP_SET_TEXTURES => {
            if command_length < BIND_BASE {
                return Err(DecodeStatus::ErrShort);
            }
            let count = ld32(&payload[4..]);
            if count as usize > MAX_BIND_ENTRIES {
                return Err(DecodeStatus::ErrTooManyBindings);
            }
            if !var_len(command_length, BIND_BASE, count, REF_SIZE) {
                return Err(DecodeStatus::ErrShort);
            }
            out.kind = Kind::TextureBind;
            out.first = ld32(&payload[0..]);
            out.count = count;
            for i in 0..count as usize {
                out.textures.push(RefBinding {
                    ref_: ld32(&payload[8 + i * REF_SIZE..]),
                });
            }
            Ok(out)
        }
        OP_SET_SAMPLERS_LOD => {
            if command_length < BIND_BASE {
                return Err(DecodeStatus::ErrShort);
            }
            let count = ld32(&payload[4..]);
            if count as usize > MAX_BIND_ENTRIES {
                return Err(DecodeStatus::ErrTooManyBindings);
            }
            if !var_len(command_length, BIND_BASE, count, SAMPLER_LOD_ENTRY) {
                return Err(DecodeStatus::ErrShort);
            }
            out.kind = Kind::SamplerLod;
            out.first = ld32(&payload[0..]);
            out.count = count;
            for i in 0..count as usize {
                let e = &payload[8 + i * SAMPLER_LOD_ENTRY..];
                out.samplers.push(SamplerBinding {
                    ref_: ld32(&e[0..]),
                    lod_min_bits: ld32(&e[4..]),
                    lod_max_bits: ld32(&e[8..]),
                    has_lod_clamp: true,
                });
            }
            Ok(out)
        }
        OP_SET_BUFFER_OFFSET => {
            if command_length != BUF_OFF_LEN {
                return Err(DecodeStatus::ErrShort);
            }
            out.kind = Kind::BufferOffset;
            out.first = ld32(&payload[0..]);
            out.buffer_offset = ld64(&payload[4..]);
            Ok(out)
        }
        OP_SET_BUFFER_OFFSET_ATTRIBUTE_STRIDE => {
            if command_length != BUF_OFF_STRIDE_LEN {
                return Err(DecodeStatus::ErrShort);
            }
            out.kind = Kind::BufferOffsetAttributeStride;
            out.first = ld32(&payload[0..]);
            out.buffer_offset = ld64(&payload[4..]);
            out.attribute_stride = ld64(&payload[12..]);
            Ok(out)
        }
        OP_DISPATCH_THREADGROUPS | OP_DISPATCH_THREADS => {
            if command_length != DISPATCH_DIRECT_LEN {
                return Err(DecodeStatus::ErrShort);
            }
            out.kind = if opcode == OP_DISPATCH_THREADGROUPS {
                Kind::DispatchThreadgroups
            } else {
                Kind::DispatchThreads
            };
            out.grid = decode_size3(&payload[0..]);
            out.threads_per_threadgroup = decode_size3(&payload[SIZE3_SIZE..]);
            Ok(out)
        }
        OP_DISPATCH_THREADGROUPS_INDIRECT => {
            if command_length != DISPATCH_INDIRECT_LEN {
                return Err(DecodeStatus::ErrShort);
            }
            out.kind = Kind::DispatchThreadgroupsIndirect;
            out.threads_per_threadgroup = decode_size3(&payload[0..]);
            out.indirect_buffer_offset = ld64(&payload[SIZE3_SIZE..]);
            out.indirect_buffer_ref = ld32(&payload[SIZE3_SIZE + 8..]);
            Ok(out)
        }
        OP_DISPATCH_THREADS_INDIRECT => {
            if command_length != DISPATCH_THREADS_INDIRECT_LEN {
                return Err(DecodeStatus::ErrShort);
            }
            out.kind = Kind::DispatchThreadsIndirect;
            out.indirect_buffer_offset = ld64(&payload[0..]);
            out.indirect_buffer_ref = ld32(&payload[8..]);
            Ok(out)
        }
        OP_SET_STAGE_IN_REGION => {
            if command_length != STAGE_IN_LEN {
                return Err(DecodeStatus::ErrShort);
            }
            out.kind = Kind::StageInRegion;
            out.stage_in_region.size = decode_size3(&payload[0..]);
            out.stage_in_region.origin = decode_size3(&payload[SIZE3_SIZE..]);
            Ok(out)
        }
        OP_SET_STAGE_IN_REGION_INDIRECT => {
            if command_length != STAGE_IN_INDIRECT_LEN {
                return Err(DecodeStatus::ErrShort);
            }
            out.kind = Kind::StageInRegionIndirect;
            out.stage_in_indirect_buffer_ref = ld32(&payload[0..]);
            out.stage_in_indirect_buffer_offset = ld64(&payload[4..]);
            Ok(out)
        }
        OP_SET_THREADGROUP_MEMORY_LENGTH => {
            if command_length != TG_MEM_LEN {
                return Err(DecodeStatus::ErrShort);
            }
            out.kind = Kind::ThreadgroupMemory;
            out.threadgroup_memory_length = ld64(&payload[0..]);
            out.threadgroup_memory_index = ld32(&payload[8..]);
            Ok(out)
        }
        OP_UPDATE_FENCE | OP_WAIT_FENCE => {
            if command_length != FENCE_LEN {
                return Err(DecodeStatus::ErrShort);
            }
            out.kind = if opcode == OP_UPDATE_FENCE {
                Kind::UpdateFence
            } else {
                Kind::WaitFence
            };
            out.fence_ref = ld32(payload);
            Ok(out)
        }
        OP_BARRIER_RESOURCES => {
            if command_length < BIND_BASE {
                return Err(DecodeStatus::ErrShort);
            }
            let count = ld32(&payload[0..]);
            if count as usize > MAX_BIND_ENTRIES {
                return Err(DecodeStatus::ErrTooManyBindings);
            }
            if !var_len(command_length, BIND_BASE, count, REF_SIZE) {
                return Err(DecodeStatus::ErrShort);
            }
            out.kind = Kind::BarrierResources;
            out.count = count;
            out.resource_usage = ld32(&payload[4..]);
            for i in 0..count as usize {
                out.resources.push(RefBinding {
                    ref_: ld32(&payload[8 + i * REF_SIZE..]),
                });
            }
            Ok(out)
        }
        OP_BARRIER_SCOPE => {
            if command_length != BARRIER_SCOPE_LEN {
                return Err(DecodeStatus::ErrShort);
            }
            out.kind = Kind::BarrierScope;
            out.barrier_scope = ld16(&payload[0..]);
            out.barrier_scope_reserved = ld16(&payload[2..]);
            Ok(out)
        }
        OP_SET_IMAGEBLOCK_DIMENSIONS => {
            if command_length != IMAGEBLOCK_LEN {
                return Err(DecodeStatus::ErrShort);
            }
            out.kind = Kind::ImageblockDimensions;
            out.imageblock_width = ld32(&payload[0..]);
            out.imageblock_height = ld32(&payload[4..]);
            Ok(out)
        }
        OP_DISPATCH_TYPE => {
            if command_length != DISPATCH_TYPE_LEN {
                return Err(DecodeStatus::ErrShort);
            }
            out.kind = Kind::DispatchType;
            out.dispatch_type = ld32(payload);
            Ok(out)
        }
        // Condition payloads live on start-while / start-if / *end*-do-while
        // (MetalSerializer + Reims VGPU surface: start-do-while is empty; end-do-while carries
        // buffer/offset/comparison/referenceValue). See compute-surface-manifest.
        OP_ENCODE_START_WHILE | OP_ENCODE_START_IF | OP_ENCODE_END_DO_WHILE => {
            if command_length != CONDITION_LEN {
                return Err(DecodeStatus::ErrShort);
            }
            out.kind = match opcode {
                OP_ENCODE_START_WHILE => Kind::ControlStartWhile,
                OP_ENCODE_START_IF => Kind::ControlStartIf,
                _ => Kind::ControlEndDoWhile,
            };
            out.condition_buffer_ref = ld32(&payload[0..]);
            out.condition_buffer_offset = ld64(&payload[4..]);
            out.condition_comparison = ld32(&payload[12..]);
            out.condition_reference_value = ld32(&payload[16..]);
            Ok(out)
        }
        OP_ENCODE_START_DO_WHILE
        | OP_ENCODE_END_WHILE
        | OP_ENCODE_START_ELSE
        | OP_ENCODE_END_IF
        | OP_INSERT_COMPRESSED_TEXTURE_FLUSH => {
            if command_length != EMPTY_LEN {
                return Err(DecodeStatus::ErrShort);
            }
            out.kind = match opcode {
                OP_ENCODE_START_DO_WHILE => Kind::ControlStartDoWhile,
                OP_ENCODE_END_WHILE => Kind::ControlEndWhile,
                OP_ENCODE_START_ELSE => Kind::ControlStartElse,
                OP_ENCODE_END_IF => Kind::ControlEndIf,
                _ => Kind::CompressedTextureFlush,
            };
            Ok(out)
        }
        OP_EXECUTE_COMMANDS_IN_BUFFER => {
            if command_length != EXECUTE_LEN {
                return Err(DecodeStatus::ErrShort);
            }
            out.kind = Kind::ExecuteCommandsInBuffer;
            out.indirect_command_buffer_ref = ld32(&payload[0..]);
            out.indirect_command_range_location = ld64(&payload[4..]);
            out.indirect_command_range_length = ld64(&payload[12..]);
            Ok(out)
        }
        OP_EXECUTE_COMMANDS_IN_BUFFER_INDIRECT => {
            if command_length != EXECUTE_INDIRECT_LEN {
                return Err(DecodeStatus::ErrShort);
            }
            out.kind = Kind::ExecuteCommandsInBufferIndirect;
            out.indirect_command_buffer_ref = ld32(&payload[0..]);
            out.indirect_command_arguments_buffer_ref = ld32(&payload[4..]);
            out.indirect_command_arguments_buffer_offset = ld64(&payload[8..]);
            Ok(out)
        }
        _ => Err(DecodeStatus::ErrUnknownOpcode),
    }
}

#[cfg(test)]
mod tests {

    /// A malformed compute command used to be dropped at the dispatch site with no
    /// log line at all — indistinguishable from a segment carrying no compute
    /// work. Each check names itself now, `Ok` still produces nothing, and the
    /// prefix keeps them apart from the six sibling `DecodeStatus` enums.
    #[test]
    fn every_compute_decode_failure_names_its_own_check() {
        use crate::observe::Refusal;
        const ERRS: &[DecodeStatus] = &[
            DecodeStatus::ErrShort,
            DecodeStatus::ErrUnknownOpcode,
            DecodeStatus::ErrUnsupportedOpcode,
            DecodeStatus::ErrTooManyBindings,
        ];
        let mut slugs: Vec<&str> = ERRS.iter().filter_map(|s| s.refusal()).collect();
        assert_eq!(slugs.len(), ERRS.len(), "every error variant refuses");
        assert!(slugs.iter().all(|s| s.starts_with("compute_decode_")));
        slugs.sort_unstable();
        let n = slugs.len();
        slugs.dedup();
        assert_eq!(slugs.len(), n, "two compute decode checks share a slug");
    }
    use super::*;
    use crate::contract::endian::st32;

    fn hdr(op: u32, len: usize) -> Vec<u8> {
        let mut v = vec![0u8; len];
        st32(&mut v[0..4], op);
        st32(&mut v[4..8], len as u32);
        v
    }

    #[test]
    fn pipeline_and_dispatch() {
        let mut v = hdr(OP_SET_PIPELINE, PIPELINE_LEN);
        st32(&mut v[8..], 42);
        let c = decode(&v).unwrap();
        assert_eq!(c.kind, Kind::Pipeline);
        assert_eq!(c.pipeline_ref, 42);

        let v = hdr(OP_DISPATCH_THREADGROUPS, DISPATCH_DIRECT_LEN);
        let c = decode(&v).unwrap();
        assert_eq!(c.kind, Kind::DispatchThreadgroups);
    }

    #[test]
    fn rejected_and_matrix() {
        assert!(opcode_supported(OP_SET_PIPELINE));
        assert!(opcode_apple_rejected(REJECTED_85));
        assert_eq!(
            decode(&hdr(REJECTED_85, 16)).unwrap_err(),
            DecodeStatus::ErrUnsupportedOpcode
        );
        assert_eq!(
            decode(&hdr(0x999, 16)).unwrap_err(),
            DecodeStatus::ErrUnknownOpcode
        );
    }

    #[test]
    fn set_buffers() {
        let count = 2u32;
        let len = BIND_BASE + (count as usize) * BUF_ENTRY;
        let mut v = hdr(OP_SET_BUFFERS, len);
        st32(&mut v[8..], 1); // first
        st32(&mut v[12..], count);
        st32(&mut v[16..], 10);
        st32(&mut v[28..], 11);
        let c = decode(&v).unwrap();
        assert_eq!(c.count, 2);
        assert_eq!(c.buffers[0].ref_, 10);
        assert_eq!(c.buffers[1].ref_, 11);
    }

    #[test]
    fn property_fuzz_opcodes() {
        for op in 0x80u32..0xf0 {
            for len in [8, 12, 16, 20, 28, 0x14, 0x1c, 0x38, 0x40] {
                let _ = decode(&hdr(op, len));
            }
        }
    }

    #[test]
    fn control_do_while_start_empty_end_condition() {
        // Wire contract: start-do-while is empty; end-do-while carries condition.
        let v = hdr(OP_ENCODE_START_DO_WHILE, EMPTY_LEN);
        let c = decode(&v).unwrap();
        assert_eq!(c.kind, Kind::ControlStartDoWhile);

        let mut v = hdr(OP_ENCODE_END_DO_WHILE, CONDITION_LEN);
        st32(&mut v[8..], 1201); // buffer ref
                                 // offset u64 @ +4 payload = absolute +12
        v[12..20].copy_from_slice(&0x640u64.to_le_bytes());
        st32(&mut v[20..], 2); // comparison Equal
        st32(&mut v[24..], 0x1234_5678);
        let c = decode(&v).unwrap();
        assert_eq!(c.kind, Kind::ControlEndDoWhile);
        assert_eq!(c.condition_buffer_ref, 1201);
        assert_eq!(c.condition_buffer_offset, 0x640);
        assert_eq!(c.condition_comparison, 2);
        assert_eq!(c.condition_reference_value, 0x1234_5678);

        // Swapped lengths must fail closed.
        assert!(decode(&hdr(OP_ENCODE_START_DO_WHILE, CONDITION_LEN)).is_err());
        assert!(decode(&hdr(OP_ENCODE_END_DO_WHILE, EMPTY_LEN)).is_err());
    }

    #[test]
    fn control_if_while_and_icb_lengths() {
        let mut v = hdr(OP_ENCODE_START_IF, CONDITION_LEN);
        st32(&mut v[8..], 7);
        let c = decode(&v).unwrap();
        assert_eq!(c.kind, Kind::ControlStartIf);
        assert_eq!(c.condition_buffer_ref, 7);

        assert_eq!(
            decode(&hdr(OP_ENCODE_START_ELSE, EMPTY_LEN)).unwrap().kind,
            Kind::ControlStartElse
        );
        assert_eq!(
            decode(&hdr(OP_ENCODE_END_IF, EMPTY_LEN)).unwrap().kind,
            Kind::ControlEndIf
        );

        let mut v = hdr(OP_EXECUTE_COMMANDS_IN_BUFFER, EXECUTE_LEN);
        st32(&mut v[8..], 1301);
        v[12..20].copy_from_slice(&3u64.to_le_bytes());
        v[20..28].copy_from_slice(&7u64.to_le_bytes());
        let c = decode(&v).unwrap();
        assert_eq!(c.kind, Kind::ExecuteCommandsInBuffer);
        assert_eq!(c.indirect_command_buffer_ref, 1301);
        assert_eq!(c.indirect_command_range_location, 3);
        assert_eq!(c.indirect_command_range_length, 7);
    }
}
