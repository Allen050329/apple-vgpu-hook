//! Resource descriptor decode (port of `host/utils/reims-vgpu-resource-decode`).

use crate::contract::endian::{ld16, ld32, ld64}; // ld64: texture-view level base/count
#[cfg(test)]
use crate::contract::endian::{st16, st32}; // ICB layout fixture encoder only

/// A refusal from the descriptor decoder.
///
/// This decoder sits upstream of every rail — blit, compute, render, mipmap and
/// resource registration all reach a guest object through it — so a refusal
/// here means some object the guest created is unusable everywhere. The payload
/// is the registered slug naming which check refused: **29 of the 40 sites were
/// `ErrShort`**, one name for twenty-nine different reads, from a 12-byte
/// object-list entry to a vertex-attribute table offset inside a type-7 body.
///
/// Slugs carry a `res_` prefix. Six modules under `runtime/decode/` define a
/// type called `DecodeStatus` and five of them have an `ErrShort` meaning a
/// different read, so without the prefix the crate-wide uniqueness gate could
/// not tell this decoder's refusals from the stream framer's.
///
/// There is deliberately no `Ok`: every entry point returns
/// `Result<_, DecodeStatus>`. `Ok`, `ErrArgs` and `ErrOverflow` were **never
/// constructed anywhere in the crate** — they existed only as arms of a
/// `decode_status_name` helper that itself had no callers — so they are gone
/// and this is a [`crate::observe::Decline`], not a `Refusal`.
///
/// `ErrBadLength` went the same way. Every length disagreement this decoder can
/// see is a read that ran off the end, and all three sites of the compact-TLV
/// walk already say so with their own slug (`res_tlv_offset_past_end`,
/// `res_tlv_header_short`, `res_tlv_value_short`). A second class for the same
/// condition would only split one check's census across two names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeStatus {
    /// The blob is shorter than the field being read.
    ErrShort(&'static str),
    /// An object-list type tag this host has no contract for.
    ErrUnknownType(&'static str),
    /// A well-formed blob whose tag/opcode names a variant the decoder does not
    /// implement.
    ErrUnsupported(&'static str),
}

impl crate::observe::Decline for DecodeStatus {
    fn slug(&self) -> &'static str {
        match self {
            Self::ErrShort(s) | Self::ErrUnknownType(s) | Self::ErrUnsupported(s) => s,
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        vec![(
            "class",
            match self {
                Self::ErrShort(_) => "short",
                Self::ErrUnknownType(_) => "unknown_type",
                Self::ErrUnsupported(_) => "unsupported",
            }
            .to_string(),
        )]
    }
}

/// Live object-list type tags (`reims_vgpu_resource_decode.h` / arm contract).
///
/// Type 3 carries the same geometry prefix as type 2 (WindowServer composite
/// and glyph sources); type 7 is the container for sampler, depth-stencil and
/// render/compute pipeline descriptors.
pub const OBJECT_TYPE_BUFFER: u8 = 1;
pub const OBJECT_TYPE_TEXTURE: u8 = 2;
pub const OBJECT_TYPE_TEXTURE_VARIANT: u8 = 3;
pub const OBJECT_TYPE_FUNCTION: u8 = 6;
pub const OBJECT_TYPE_TYPE7: u8 = 7;
pub const OBJECT_TYPE_TEXTURE_VIEW: u8 = 8;
pub const OBJECT_TYPE_IOSURFACE: u8 = 11;

/// Type-7 first dword subtypes.
pub const TYPE7_OBJECT_SAMPLER: u32 = 0x03;
pub const TYPE7_OBJECT_DEPTH_STENCIL: u32 = 0x04;
pub const TYPE7_OBJECT_COMPUTE_PIPELINE: u32 = 0x0b;
pub const TYPE7_OBJECT_RENDER_PIPELINE: u32 = 0x0e;
/// Indirect command buffer create body from
/// `PGSerializer newIndirectCommandBufferWithDescriptor:layout:maxCommandCount:options:allocator:`.
pub const TYPE7_OBJECT_ICB: u32 = 0x36;
/// End of the 16-byte type-7 header, which is also where its first TLV
/// starts — one boundary, so one name.
pub const TYPE7_FIRST_TLVS: usize = 16;
/// Serialized ICB descriptor length (allocateOperationBytes 0x58).
pub const ICB_DESC_LEN: usize = 0x58;
pub const ICB_DESC_TAG: usize = 0;
pub const ICB_DESC_DECLARED_LEN: usize = 4;
pub const ICB_DESC_COMMAND_TYPES: usize = 8;
/// Per-stage max bind counts are single bytes (PGSerializer create body).
/// `newIndirectCommandBufferWithDescriptor:…` strb order:
/// +0xc vertex · +0xd fragment · +0xe kernel · +0xf object · +0x10 mesh ·
/// +0x11 kernelTG · +0x12 objectTG.
pub const ICB_DESC_MAX_VERTEX_BINDS: usize = 0x0c; // u8
pub const ICB_DESC_MAX_FRAGMENT_BINDS: usize = 0x0d; // u8
/// maxKernelBufferBindCount.
pub const ICB_DESC_MAX_KERNEL_BINDS: usize = 0x0e; // u8
/// maxObjectBufferBindCount (mesh object stage).
pub const ICB_DESC_MAX_OBJECT_BINDS: usize = 0x0f; // u8
/// maxMeshBufferBindCount.
pub const ICB_DESC_MAX_MESH_BINDS: usize = 0x10; // u8
/// maxKernelThreadgroupMemoryBindCount.
pub const ICB_DESC_MAX_KERNEL_TG_BINDS: usize = 0x11; // u8
/// maxObjectThreadgroupMemoryBindCount.
pub const ICB_DESC_MAX_OBJECT_TG_BINDS: usize = 0x12; // u8
pub const ICB_DESC_FLAGS: usize = 0x16; // u16 feature flags
/// Bytes per ICB kernel-threadgroup-memory length slot (`u64` length at index).
pub const ICB_TG_MEMORY_STRIDE: usize = 8;
/// Bytes per ICB attribute-stride table entry (`u64` stride at buffer index).
/// `setKernelBuffer:offset:attributeStride:atIndex:` and
/// `setVertexBuffer:offset:attributeStride:atIndex:` store at
/// `attributeStrideOffset + index*8`.
pub const ICB_ATTRIBUTE_STRIDE_ENTRY_SIZE: usize = 8;
/// Flags at +0x16 from create, in the order the command layout sets them up:
/// bit0 = inheritPipelineState, bit1 = inheritBuffers.
pub const ICB_FLAG_INHERIT_PIPELINE_STATE: u16 = 1 << 0;
pub const ICB_FLAG_INHERIT_BUFFERS: u16 = 1 << 1;
/// Embedded ICB command layout (52 B) at +0x1c in the create body.
pub const ICB_DESC_LAYOUT: usize = 0x1c;
pub const ICB_LAYOUT_LEN: usize = 0x34;
pub const ICB_DESC_MAX_COMMAND_COUNT: usize = 0x50;
pub const ICB_DESC_OPTIONS: usize = 0x54;
/// Command-type values written by PGSerializerIndirect*Command fills.
pub const ICB_CMD_TYPE_DRAW: u32 = 0x1;
pub const ICB_CMD_TYPE_DRAW_INDEXED: u32 = 0x2;
/// `drawPatches` stores wire type `4`.
pub const ICB_CMD_TYPE_DRAW_PATCHES: u32 = 0x4;
/// `drawIndexedPatches` stores wire type `8`.
pub const ICB_CMD_TYPE_DRAW_INDEXED_PATCHES: u32 = 0x8;
pub const ICB_CMD_TYPE_CONCURRENT_DISPATCH_THREADGROUPS: u32 = 0x20;
pub const ICB_CMD_TYPE_CONCURRENT_DISPATCH_THREADS: u32 = 0x40;
/// Wire command type = SDK bit value (same pattern as Draw/Patches).
/// `setupCommandLayout:` uses `1<<7` / `1<<8` for mesh args size.
/// Fill IMPs are stubs; type value follows the bit-pattern convention.
pub const ICB_CMD_TYPE_DRAW_MESH_THREADGROUPS: u32 = 0x80;
pub const ICB_CMD_TYPE_DRAW_MESH_THREADS: u32 = 0x100;
/// Bytes per kernel/vertex/fragment buffer bind slot in the command layout.
pub const ICB_BUFFER_BIND_STRIDE: usize = 0x14;
/// Tessellation-factor table used size (u32 ref + 3×u64) at `tessellationFactorOffset`.
pub const ICB_TESSELLATION_FACTOR_LEN: usize = 0x1c;
/// Concurrent-dispatch args size: two `MTLSize`, grid then threadgroup, at
/// 3xu64 each — 2 * 3 * 8 = 0x30. Matches the `ConcurrentDispatch` bit's
/// allocation in host RE `setupCommandLayout:`.
pub const ICB_CONCURRENT_DISPATCH_ARGS_LEN: usize = 0x30;
/// DrawPatches args size: `setupCommandLayout` allocates 0x38, and the fill IMP
/// writes through `baseInstance` — a u64 *starting* at 0x2e, so ending at 0x36.
/// The two bytes between are the allocation's slack, exactly as
/// [`ICB_DRAW_INDEXED_PATCHES_ARGS_LEN`] documents for its own 0x4a/0x4c pair.
/// (This doc used to read "baseInstance ends at +0x2e", which reads as though
/// 0x38 were the fill extent and makes the constant look two bytes wrong.)
pub const ICB_DRAW_PATCHES_ARGS_LEN: u32 = 0x38;
/// DrawIndexedPatches args size (baseInstance u64 @0x42 → end 0x4a).
/// Note: `setupCommandLayout` allocates max `0x4c` for this bit; fill IMP uses through `0x4a`.
pub const ICB_DRAW_INDEXED_PATCHES_ARGS_LEN: u32 = 0x4a;
/// Mesh drawMeshThreadgroups / drawMeshThreads args size.
/// `setupCommandLayout:`: both mesh create bits take **0x48** —
/// three `MTLSize` (3×u64 each) matching Metal SPI
/// `MTLIndirectDrawMesh{Threadgroups,Threads}Arguments` field order:
/// grid / threadsPerGrid @0, object TG @0x18, mesh TG @0x30.
pub const ICB_DRAW_MESH_ARGS_LEN: u32 = 0x48;
/// SDK MTLIndirectCommandType bits (not metal-0.33's shifted ConcurrentDispatch).
pub const MTL_INDIRECT_CMD_DRAW: u32 = 1 << 0;
pub const MTL_INDIRECT_CMD_DRAW_INDEXED: u32 = 1 << 1;
pub const MTL_INDIRECT_CMD_DRAW_PATCHES: u32 = 1 << 2;
pub const MTL_INDIRECT_CMD_DRAW_INDEXED_PATCHES: u32 = 1 << 3;
pub const MTL_INDIRECT_CMD_CONCURRENT_DISPATCH: u32 = 1 << 5;
pub const MTL_INDIRECT_CMD_CONCURRENT_DISPATCH_THREADS: u32 = 1 << 6;
/// Mesh create bits (SDK). Wire args size from setupCommandLayout; fill IMPs stubbed.
pub const MTL_INDIRECT_CMD_DRAW_MESH_THREADGROUPS: u32 = 1 << 7;
pub const MTL_INDIRECT_CMD_DRAW_MESH_THREADS: u32 = 1 << 8;

/// Compact type-7 TLV field: `[tag:u8][length:u8][value…]` after a field-count byte.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CompactTlv {
    pub tag: u8,
    pub length: u8,
    pub value_offset: usize,
    pub value_u32: u32,
    pub has_u32: bool,
}

/// Linear buffer descriptor (type-1): allocation size + guest page handle.
///
/// Contract: `reims_vgpu_resource_format.h` `REIMS_VGPU_RESOURCE_LINEAR_DESC_*`.
/// Backing GVA = `(handle as u64) << PAGE_SHIFT` (14 on arm64e guest).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BufferDescriptor {
    pub allocation_size: u64,
    pub handle64: u64,
    pub handle: u32,
}

/// Linear descriptor offsets (shared with type-2 texture prefix).
pub const LINEAR_DESC_MIN_LEN: usize = 16;
pub const LINEAR_DESC_SIZE: usize = 0;
pub const LINEAR_DESC_HANDLE: usize = 8;
/// Arm fixture alias of [`crate::contract::gva::PAGE_SHIFT_ARM64E`].
/// Prefer `PAGE_SHIFT_ARM64E` / `PAGE_SHIFT_X86` at new call sites. Product
/// paths must pass `DeviceState::page_shift`, not a fixed arch constant.
pub const RESOURCE_PAGE_SHIFT: u32 = crate::contract::gva::PAGE_SHIFT_ARM64E;

impl BufferDescriptor {
    /// Guest VA of buffer backing at the given page_shift, or None when
    /// size/handle/shift is invalid. Callers must pass arm (14) or x86 (12)
    /// explicitly — there is no default.
    pub fn backing_gva_size(&self, page_shift: u32) -> Option<(u64, u64)> {
        if self.allocation_size == 0 || self.handle == 0 || page_shift == 0 || page_shift > 30 {
            return None;
        }
        Some(((self.handle as u64) << page_shift, self.allocation_size))
    }
}

/// One mip level layout inside a type-2/3 texture allocation.
///
/// Level 0 comes from the geometry prefix; levels 1..N-1 are 36-byte records at
/// `TEXTURE_DESC_LEVEL_RECORDS` (offset/size/row_stride from allocation base).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TextureLevelLayout {
    /// Byte offset from allocation base (`handle << PAGE_SHIFT`).
    pub offset: u64,
    pub size: u64,
    pub row_stride: u64,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
}

impl TextureLevelLayout {
    /// Bytes from [`Self::offset`] that a row-by-row reader or writer of this
    /// level actually touches, given one tight row of its format.
    ///
    /// `row_stride * height` is the obvious answer and it overcounts by one
    /// row of trailing padding: the last row occupies `tight_row` bytes, and
    /// every reader in this crate walks `base + y * row_stride` for `tight_row`
    /// bytes. Used as a bound against `TextureDescriptor::allocation_size`,
    /// the looser form rejects allocations the guest sized correctly.
    ///
    /// Measured: a 27x27 `RG8Unorm` window-corner mask at offset 0x850 with a
    /// 384-byte stride in a 12 288-byte allocation scores 12 496 under
    /// `stride * height` and is refused, while its true extent is 12 166. That
    /// refusal dropped the WindowServer's whole full-screen composite draw, so
    /// the guest's rounded window corners and drop shadows rendered square.
    ///
    /// `None` for zero height (no rows) or on overflow. A bound this feeds must
    /// treat `None` as a refusal, never as "no limit".
    pub fn read_span(&self, tight_row: u32) -> Option<u64> {
        self.height
            .checked_sub(1)
            .map(u64::from)?
            .checked_mul(self.row_stride)?
            .checked_add(u64::from(tight_row))
    }

    /// Bytes from [`Self::offset`] that a reader of one whole array slice /
    /// cube face of this level touches, when `depth` planes are packed
    /// contiguously inside the slice at `row_stride * height` each.
    ///
    /// Every plane below the last is walked in full; the last one ends at
    /// [`Self::read_span`], because the padding after its final row is no more
    /// read here than it is in the 2D case. `depth` 0 and 1 both mean one
    /// plane, matching `TextureDescriptor`'s "0 means 2D" encoding.
    pub fn slice_read_span(&self, tight_row: u32, depth: u32) -> Option<u64> {
        u64::from(depth.max(1) - 1)
            .checked_mul(self.row_stride)?
            .checked_mul(u64::from(self.height))?
            .checked_add(self.read_span(tight_row)?)
    }
}

/// Type-2/3 linear texture geometry (`REIMS_VGPU_RESOURCE_TEXTURE_DESC_*`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TextureDescriptor {
    pub allocation_size: u64,
    pub handle: u32,
    pub mipmap_level_count: u32,
    pub data_offset: u32,
    pub bytes_per_element: u8,
    pub used_size: u32,
    pub row_stride: u32,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub pixel_format: u16,
    pub has_row_stride: bool,
    pub has_width: bool,
    pub has_height: bool,
    pub has_pixel_format: bool,
    /// Per-mip layouts (index 0 = L0). Empty if geometry incomplete.
    pub levels: Vec<TextureLevelLayout>,
}

impl TextureDescriptor {
    /// Allocation base GVA (`handle << page_shift`), not including data_offset.
    /// `page_shift` must be chosen explicitly (12 or 14).
    pub fn allocation_base_gva(&self, page_shift: u32) -> Option<u64> {
        if self.handle == 0 || page_shift == 0 || page_shift > 30 {
            return None;
        }
        Some((self.handle as u64) << page_shift)
    }

    /// Level-0 texel base GVA and allocation size at the given page_shift.
    pub fn backing_gva_size(&self, page_shift: u32) -> Option<(u64, u64)> {
        if self.allocation_size == 0 || self.handle == 0 {
            return None;
        }
        let base = self.allocation_base_gva(page_shift)?;
        Some((base + self.data_offset as u64, self.allocation_size))
    }

    /// Layout for mip `level` (0-based).
    pub fn level(&self, level: u32) -> Option<&TextureLevelLayout> {
        self.levels.get(level as usize)
    }

    /// Guest VA of mip `level` texel base + layout at the given page_shift.
    pub fn level_gva(&self, level: u32, page_shift: u32) -> Option<(u64, &TextureLevelLayout)> {
        let base = self.allocation_base_gva(page_shift)?;
        let layout = self.level(level)?;
        if layout.width == 0 || layout.height == 0 || layout.row_stride == 0 {
            return None;
        }
        // Offset must leave room for at least one row within the allocation.
        if self.allocation_size != 0
            && (layout.offset >= self.allocation_size
                || self.allocation_size - layout.offset < layout.row_stride)
        {
            return None;
        }
        Some((base.checked_add(layout.offset)?, layout))
    }
}

/// Texture descriptor field offsets (geometry prefix + format trailer).
pub const TEXTURE_DESC_GEOMETRY_LEN: usize = 68;
pub const TEXTURE_DESC_MIPMAP_LEVEL_COUNT: usize = 12;
pub const TEXTURE_DESC_DATA_OFFSET: usize = 16;
pub const TEXTURE_DESC_BYTES_PER_ELEMENT: usize = 35;
pub const TEXTURE_DESC_USED_SIZE: usize = 44;
pub const TEXTURE_DESC_ROW_STRIDE: usize = 52;
pub const TEXTURE_DESC_WIDTH: usize = 60;
pub const TEXTURE_DESC_HEIGHT: usize = 64;
pub const TEXTURE_DESC_DEPTH: usize = 68;
pub const TEXTURE_DESC_LEVEL_RECORDS: usize = 72;
pub const TEXTURE_DESC_MIP_LEVEL_RECORD_LEN: usize = 36;
pub const TEXTURE_LEVEL_OFFSET: usize = 0;
pub const TEXTURE_LEVEL_SIZE: usize = 8;
pub const TEXTURE_LEVEL_ROW_STRIDE: usize = 16;
pub const TEXTURE_LEVEL_WIDTH: usize = 24;
pub const TEXTURE_LEVEL_HEIGHT: usize = 28;
pub const TEXTURE_LEVEL_DEPTH: usize = 32;
pub const TEXTURE_DESC_PIXEL_FORMAT: usize = 86;
pub const TEXTURE_DESC_BASE_LEN: usize = 116;
pub const TEXTURE_MAX_MIP_LEVELS: usize = 16;

/// Vertex attribute from a type-7 render-pipeline vertex-input block.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VertexAttribute {
    pub location: u32,
    pub format: u32,
    pub offset: u32,
    pub buffer_index: u32,
    pub stride: u32,
    pub has_step_function: bool,
    pub step_function: u32,
    pub has_step_rate: bool,
    pub step_rate: u32,
}

/// `MTLColorWriteMask` for one attachment, in Metal's own bit order.
///
/// A newtype rather than a bare `u32` for one reason: the value that means
/// "write every channel" is `0xf`, and the value a derived `Default` would
/// produce is `0` — which means *write nothing*. `PipelineColorAttachment` is
/// built with `..Default::default()` in the decoder and defaulted outright at
/// several call sites, so a bare field would make an omitted mask a black
/// attachment. Here the omission is unwritable: `Default` is `all`, which is
/// also what an entry that does not carry tag `0x09` means on the wire.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ColorWriteMask {
    /// Private so the only ways to obtain one are [`ColorWriteMask::new`],
    /// which range-checks, and `Default`, which is `all`. A `pub` field would
    /// also make this a decode field needing its own coverage disposition,
    /// when the disposition that matters is `PipelineColorAttachment`'s.
    bits: u32,
}

impl Default for ColorWriteMask {
    fn default() -> Self {
        Self {
            bits: MTL_COLOR_WRITE_MASK_ALL,
        }
    }
}

impl ColorWriteMask {
    /// `None` for a value outside `MTLColorWriteMask`'s four bits — see
    /// [`ColorWriteMaskOutOfRange`], which is what the decoder reports for it.
    pub fn new(bits: u32) -> Option<Self> {
        (bits <= MTL_COLOR_WRITE_MASK_ALL).then_some(Self { bits })
    }

    pub fn bits(self) -> u32 {
        self.bits
    }
}

/// One pipeline color-attachment entry (format + blend) from the type-7 color section.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PipelineColorAttachment {
    pub slot: u32,
    pub has_pixel_format: bool,
    pub pixel_format: u32,
    pub blending_enabled: bool,
    pub src_rgb: u32,
    pub dst_rgb: u32,
    pub op_rgb: u32,
    pub src_alpha: u32,
    pub dst_alpha: u32,
    pub op_alpha: u32,
    /// Which channels this attachment writes. Independent of blending: a
    /// masked attachment with blending off still leaves the unwritten channels
    /// alone, so this cannot ride inside the blend state.
    pub write_mask: ColorWriteMask,
}

/// Decoded type-7 render pipeline (functions + optional stage-in attrs).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RenderPipelineDescriptor {
    pub object_id: u32,
    pub word3: u32,
    pub vertex_func_ref: u32,
    pub fragment_func_ref: u32,
    /// Object-stage function ref (mesh SPI shape: tag `0x01`).
    ///
    /// Zero on classic pipelines. Mesh-only/dual-export fixtures may leave this
    /// zero and put the metallib in `vertex_func_ref` instead.
    pub object_func_ref: u32,
    /// Mesh-stage function ref (mesh SPI shape: tag `0x02`).
    ///
    /// Zero on classic pipelines. When zero, product mesh fill falls back to
    /// dual-export / mesh-only metallib in `vertex_func_ref`.
    pub mesh_func_ref: u32,
    /// Byte offset from end of 16-byte header to color-attachment section (tag 0x08).
    pub color_attachment_offset: u32,
    pub has_color_attachment_offset: bool,
    pub vertex_attributes: Vec<VertexAttribute>,
    /// First color attachment (compat / color0).
    pub color0: PipelineColorAttachment,
    /// All color attachments with Metal slot indices (0..count-1 by entry order).
    pub color_attachments: Vec<PipelineColorAttachment>,
}

/// Compute stage-input attribute from type-7 compute pipeline compact block.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ComputeStageInputAttribute {
    pub raw_bits: u32,
    pub location: u32,
    pub format: u32,
    pub offset: u32,
    pub buffer_index: u32,
}

/// Compute stage-input layout from type-7 compute pipeline compact block.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ComputeStageInputLayout {
    pub raw_bits: u32,
    pub buffer_index: u32,
    pub step_function: u32,
    pub step_rate: u32,
    pub stride: u64,
}

/// Decoded compute `stageInputDescriptor` from a type-7 compute pipeline.
///
/// Layout is the MetalSerializer compact block after the first TLV record:
/// `word0`, `header0` (payload len + counts + index metadata), `header1`
/// (layout/attribute section offsets), then packed layout/attribute entries.
/// See `reims_vgpu_resource_format.h` COMPUTE_STAGE_INPUT_* and resource-surface-manifest.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ComputeStageInputDescriptor {
    pub word0: u32,
    pub header0: u32,
    pub header1: u32,
    pub index_type: u32,
    pub index_buffer_index: u32,
    pub attributes: Vec<ComputeStageInputAttribute>,
    pub layouts: Vec<ComputeStageInputLayout>,
    /// Attributes beyond [`MAX_COMPUTE_STAGE_INPUT_ATTRS`] (fail product handoff).
    pub dropped_attributes: u32,
    /// Layouts beyond [`MAX_COMPUTE_STAGE_INPUT_LAYOUTS`] (fail product handoff).
    pub dropped_layouts: u32,
}

/// Decoded type-7 compute pipeline (kernel function + optional stage-input).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ComputePipelineDescriptor {
    pub kernel_func_ref: u32,
    pub stage_input: Option<ComputeStageInputDescriptor>,
}

/// Caps matching `REIMS_VGPU_RESOURCE_MAX_COMPUTE_STAGE_INPUT_*` / backend ABI.
pub const MAX_COMPUTE_STAGE_INPUT_ATTRS: usize = 16;
pub const MAX_COMPUTE_STAGE_INPUT_LAYOUTS: usize = 16;

// MetalSerializer compute stage-input compact block (offsets relative to block start).
pub const COMPUTE_STAGE_INPUT_WORD0: usize = 0;
pub const COMPUTE_STAGE_INPUT_HEADER0: usize = 4;
pub const COMPUTE_STAGE_INPUT_HEADER1: usize = 8;
pub const COMPUTE_STAGE_INPUT_MIN_LEN: usize = 12;
pub const COMPUTE_STAGE_INPUT_HEADER0_LEN_MASK: u32 = 0xffff;
pub const COMPUTE_STAGE_INPUT_HEADER0_INDEX_TYPE_SHIFT: u32 = 16;
pub const COMPUTE_STAGE_INPUT_HEADER0_INDEX_TYPE_MASK: u32 = 0x1;
pub const COMPUTE_STAGE_INPUT_HEADER0_INDEX_BUFFER_SHIFT: u32 = 17;
pub const COMPUTE_STAGE_INPUT_HEADER0_INDEX_BUFFER_MASK: u32 = 0x1f;
pub const COMPUTE_STAGE_INPUT_HEADER0_ATTR_COUNT_SHIFT: u32 = 22;
pub const COMPUTE_STAGE_INPUT_HEADER0_LAYOUT_COUNT_SHIFT: u32 = 27;
pub const COMPUTE_STAGE_INPUT_HEADER0_COUNT_MASK: u32 = 0x1f;
pub const COMPUTE_STAGE_INPUT_HEADER1_LAYOUT_OFFSET_MASK: u32 = 0xffff;
pub const COMPUTE_STAGE_INPUT_HEADER1_ATTR_OFFSET_SHIFT: u32 = 16;
/// Offsets in header1 are relative to header0 (not word0).
pub const COMPUTE_STAGE_INPUT_HEADER1_OFFSET_BASE: usize = COMPUTE_STAGE_INPUT_HEADER0;
pub const COMPUTE_STAGE_INPUT_LAYOUT_ENTRY_SIZE: usize = 16;
pub const COMPUTE_STAGE_INPUT_LAYOUT_BITS_BUFFER_MASK: u32 = 0x1f;
pub const COMPUTE_STAGE_INPUT_LAYOUT_BITS_STEP_SHIFT: u32 = 5;
pub const COMPUTE_STAGE_INPUT_LAYOUT_BITS_STEP_MASK: u32 = 0x1f;
pub const COMPUTE_STAGE_INPUT_LAYOUT_STEP_RATE: usize = 4;
pub const COMPUTE_STAGE_INPUT_LAYOUT_STRIDE: usize = 8;
pub const COMPUTE_STAGE_INPUT_ATTR_ENTRY_SIZE: usize = 8;
pub const COMPUTE_STAGE_INPUT_ATTR_BITS_LOCATION_MASK: u32 = 0x1f;
pub const COMPUTE_STAGE_INPUT_ATTR_BITS_BUFFER_SHIFT: u32 = 5;
pub const COMPUTE_STAGE_INPUT_ATTR_BITS_BUFFER_MASK: u32 = 0x1f;
pub const COMPUTE_STAGE_INPUT_ATTR_BITS_FORMAT_SHIFT: u32 = 10;
pub const COMPUTE_STAGE_INPUT_ATTR_BITS_FORMAT_MASK: u32 = 0x3f;
pub const COMPUTE_STAGE_INPUT_ATTR_OFFSET: usize = 4;

/// Type-8 texture view (base texture + optional format/level/slice/swizzle).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TextureViewDescriptor {
    pub view_opcode: u32,
    pub view_texture_ref: u32,
    pub base_texture_ref: u32,
    pub pixel_format: u16,
    pub has_pixel_format: bool,
    /// `MTLTextureType` when present on ranged/swizzle forms (`@18`).
    pub texture_type: u16,
    pub has_texture_type: bool,
    pub level_base: u64,
    pub level_count: u64,
    pub has_levels: bool,
    pub slice_base: u64,
    pub slice_count: u64,
    pub has_slices: bool,
    pub swizzle: [u8; 4],
    pub has_swizzle: bool,
}

pub const TEXTURE_VIEW_DESC_OPCODE: usize = 0;
pub const TEXTURE_VIEW_DESC_LEN: usize = 4;
pub const TEXTURE_VIEW_DESC_TEXTURE_REF: usize = 8;
pub const TEXTURE_VIEW_DESC_BASE_REF: usize = 12;
pub const TEXTURE_VIEW_DESC_PIXEL_FORMAT: usize = 16;
pub const TEXTURE_VIEW_DESC_TEXTURE_TYPE: usize = 18;
pub const TEXTURE_VIEW_DESC_LEVEL_BASE: usize = 20;
pub const TEXTURE_VIEW_DESC_LEVEL_COUNT: usize = 28;
pub const TEXTURE_VIEW_DESC_SLICE_BASE: usize = 36;
pub const TEXTURE_VIEW_DESC_SLICE_COUNT: usize = 44;
pub const TEXTURE_VIEW_DESC_SWIZZLE: usize = 52;
pub const TEXTURE_VIEW_MIN_SIMPLE: usize = 20;
pub const TEXTURE_VIEW_MIN_RANGED: usize = 52;
pub const TEXTURE_VIEW_MIN_SWIZZLE: usize = 56;
pub const TEXTURE_VIEW_OPCODE_SIMPLE: u32 = 7;
pub const TEXTURE_VIEW_OPCODE_RANGED: u32 = 8;
pub const TEXTURE_VIEW_OPCODE_SWIZZLE: u32 = 0x1b;
// Heap-backed texture (`newTextureWithDescriptor:heap:offset:useOffset:
// allocator:`). It shares the type-8 object tag, but is a complete texture
// resource rather than a view. The 32-byte PGSerializedTextureDescriptor body
// starts at +0x10, followed by `useOffset` and the heap byte offset.
pub const HEAP_TEXTURE_OPCODE: u32 = 0x15;
pub const HEAP_TEXTURE_LEN: usize = 60;
pub const HEAP_TEXTURE_HEAP_REF: usize = 12;
pub const HEAP_TEXTURE_DESCRIPTOR: usize = 16;
pub const HEAP_TEXTURE_USE_OFFSET: usize = 48;
pub const HEAP_TEXTURE_OFFSET: usize = 52;
// Opcode 9 is NOT a view: it is a buffer-backed texture (`newTextureWithBuffer:
// descriptor:offset:bytesPerRow:`) serialized by `-[PGSerializer newTextureWith
// Buffer:...]`.
// It shares only the type-8 object tag + 16-byte header (opcode@0, len@4,
// self-ref@8, source-ref@0xc); the source ref @0xc is a BUFFER, not a texture,
// and the body is {u64 offset, u64 bytesPerRow, 32-byte embedded MTLTextureDescriptor}.
pub const TEXTURE_VIEW_OPCODE_BUFFER_TEXTURE: u32 = 9;
pub const BUF_TEX_DESC_BUFFER_REF: usize = 12;
pub const BUF_TEX_DESC_OFFSET: usize = 16;
pub const BUF_TEX_DESC_BYTES_PER_ROW: usize = 24;
// The embedded MTLTextureDescriptor base is @0x20, which is where the flags
// word sits — one offset, read under the name the decoder uses for it.
pub const BUF_TEX_DESC_FLAGS: usize = 32; // bits[0:4]=textureType, bits[16:32]=pixelFormat
pub const BUF_TEX_DESC_WIDTH: usize = 36;
pub const BUF_TEX_DESC_HEIGHT: usize = 40;
pub const BUF_TEX_DESC_DEPTH: usize = 44;
pub const BUF_TEX_DESC_MIP_COUNT: usize = 48; // u16
pub const BUF_TEX_DESC_SAMPLE_COUNT: usize = 50; // u16
pub const BUF_TEX_DESC_ARRAY_LEN: usize = 52; // u16
pub const BUF_TEX_MIN_LEN: usize = 64;
// MTLTextureType values (Metal.framework Headers/MTLTextureType.h).
pub const TEXTURE_VIEW_MTL_TYPE_1D: u16 = 0;
pub const TEXTURE_VIEW_MTL_TYPE_1D_ARRAY: u16 = 1;
pub const TEXTURE_VIEW_MTL_TYPE_2D: u16 = 2;
pub const TEXTURE_VIEW_MTL_TYPE_2D_ARRAY: u16 = 3;
pub const TEXTURE_VIEW_MTL_TYPE_2D_MULTISAMPLE: u16 = 4;
pub const TEXTURE_VIEW_MTL_TYPE_CUBE: u16 = 5;
pub const TEXTURE_VIEW_MTL_TYPE_CUBE_ARRAY: u16 = 6;
pub const TEXTURE_VIEW_MTL_TYPE_3D: u16 = 7;

/// Whether a type-8 view `texture_type` is supported for product-path blit/sample.
pub fn texture_view_type_supported(texture_type: u16) -> bool {
    matches!(
        texture_type,
        TEXTURE_VIEW_MTL_TYPE_1D
            | TEXTURE_VIEW_MTL_TYPE_1D_ARRAY
            | TEXTURE_VIEW_MTL_TYPE_2D
            | TEXTURE_VIEW_MTL_TYPE_2D_ARRAY
            | TEXTURE_VIEW_MTL_TYPE_CUBE
            | TEXTURE_VIEW_MTL_TYPE_CUBE_ARRAY
            | TEXTURE_VIEW_MTL_TYPE_3D
    )
}

/// Types that use the Metal array-slice dimension (not 3D depth).
pub fn texture_view_type_uses_slices(texture_type: u16) -> bool {
    matches!(
        texture_type,
        TEXTURE_VIEW_MTL_TYPE_1D_ARRAY
            | TEXTURE_VIEW_MTL_TYPE_2D_ARRAY
            | TEXTURE_VIEW_MTL_TYPE_CUBE
            | TEXTURE_VIEW_MTL_TYPE_CUBE_ARRAY
    )
}

/// 3D volume type (uses z depth planes; array slice must be 0).
pub fn texture_view_type_is_3d(texture_type: u16) -> bool {
    texture_type == TEXTURE_VIEW_MTL_TYPE_3D
}

// Colour-attachment TLV tags. `0x01..=0x09` are `MTLRenderPipelineColorAttach\
// mentDescriptor`'s nine properties in the order `MTLRenderPipeline.h` declares
// them — pixelFormat, blendingEnabled, sourceRGBBlendFactor,
// destinationRGBBlendFactor, rgbBlendOperation, sourceAlphaBlendFactor,
// destinationAlphaBlendFactor, alphaBlendOperation, writeMask — so the tag is
// the property's one-based header index. Tag `0x00` sits before the first
// property and is not one; it rides every entry with value 0 in every workload
// measured and is reported by `note_color_entry_fields` as unconsumed.
pub const COLOR_ATTACHMENT_TAG_PIXEL_FORMAT: u8 = 0x01;
pub const COLOR_ATTACHMENT_TAG_BLEND_ENABLE: u8 = 0x02;
pub const COLOR_ATTACHMENT_TAG_SRC_RGB: u8 = 0x03;
pub const COLOR_ATTACHMENT_TAG_DST_RGB: u8 = 0x04;
pub const COLOR_ATTACHMENT_TAG_RGB_OP: u8 = 0x05;
pub const COLOR_ATTACHMENT_TAG_SRC_ALPHA: u8 = 0x06;
pub const COLOR_ATTACHMENT_TAG_DST_ALPHA: u8 = 0x07;
pub const COLOR_ATTACHMENT_TAG_ALPHA_OP: u8 = 0x08;
/// `MTLColorWriteMask`, the ninth and last property.
///
/// Read off a live x86/Vulkan guest on 2026-07-30: the tag appears with
/// `len=4 value=1` on a pipeline whose entry is `[00, 01, 02, 06, 09]`, and
/// `value=1` is [`MTL_COLOR_WRITE_MASK_ALPHA`] — an alpha-only attachment,
/// which is how a compositor punches a shape into a surface's alpha without
/// touching its colour. Serialized entries omit properties left at their
/// default, which is why only the one non-`all` mask in that boot appeared.
pub const COLOR_ATTACHMENT_TAG_WRITE_MASK: u8 = 0x09;
pub const BLEND_FACTOR_ZERO: u32 = 0;
pub const BLEND_FACTOR_ONE: u32 = 1;
pub const BLEND_OP_ADD: u32 = 0;

// `MTLColorWriteMask` (Metal.framework Headers/MTLRenderPipeline.h). The bits
// run alpha-first from the low end, which is the reverse of the RGBA reading
// order the name suggests — `Red` is `1 << 3`, not `1 << 0`.
pub const MTL_COLOR_WRITE_MASK_NONE: u32 = 0;
pub const MTL_COLOR_WRITE_MASK_ALPHA: u32 = 1 << 0;
pub const MTL_COLOR_WRITE_MASK_BLUE: u32 = 1 << 1;
pub const MTL_COLOR_WRITE_MASK_GREEN: u32 = 1 << 2;
pub const MTL_COLOR_WRITE_MASK_RED: u32 = 1 << 3;
pub const MTL_COLOR_WRITE_MASK_ALL: u32 = 0xf;

/// Sampler descriptor length (type-7 subtype 0x03).
pub const SAMPLER_DESC_LEN: usize = 36;
pub const SAMPLER_DESC_TAG: usize = 0;
pub const SAMPLER_DESC_DECLARED_LEN: usize = 4;
pub const SAMPLER_DESC_ID: usize = 8;
pub const SAMPLER_DESC_STATE_BITS: usize = 12;
pub const SAMPLER_DESC_WORD16: usize = 16;
pub const SAMPLER_DESC_WORD20: usize = 20;
pub const SAMPLER_DESC_LOD_MAX: usize = 24;

/// Live function descriptor (reims_vgpu_resource_format.h).
pub const FUNCTION_DESC_BLOB_GVA: usize = 0;
pub const FUNCTION_DESC_BLOB_SIZE: usize = 8;
pub const FUNCTION_DESC_FUNCTION_ID: usize = 0x14;
pub const FUNCTION_DESC_MIN_LEN: usize = 12;
pub const FUNCTION_DESC_READ_LEN: usize = 32;

/// Compact first-subrecord tags (u8) on type-7 pipelines.
pub const PIPELINE_TAG_KERNEL_FUNC: u8 = 0x00;
/// Classic: vertex function. Mesh SPI: object function.
pub const PIPELINE_TAG_VERTEX_FUNC: u8 = 0x01;
/// Classic: fragment function. Mesh SPI: mesh function.
pub const PIPELINE_TAG_FRAGMENT_FUNC: u8 = 0x02;
/// Mesh SPI only: fragment function (classic tag 0x03 is a different field).
pub const PIPELINE_TAG_MESH_FRAGMENT_FUNC: u8 = 0x03;
/// Offset (from header end) to color-attachment section; vertex block lives before it.
pub const PIPELINE_TAG_COLOR_ATTACH_OFFSET: u8 = 0x08;
/// Mesh SPI section offset (analog of classic [`PIPELINE_TAG_COLOR_ATTACH_OFFSET`]).
///
/// Live host Metal `-[_MTLDevice serializeMeshRenderPipelineDescriptor:]`
/// differentials (2026-07-12, Apple M3 Max): same compact first-subrecord
/// grammar as classic type-7 (`[fieldCount]×[tag][0x04][u32]`). Presence of
/// this tag selects the mesh role map for tags 0x01/0x02/0x03.
pub const PIPELINE_TAG_MESH_SECTION_OFFSET: u8 = 0x14;
/// Mesh object-stage function — same wire tag as classic vertex (`0x01`).
pub const PIPELINE_TAG_OBJECT_FUNC: u8 = PIPELINE_TAG_VERTEX_FUNC;
/// Mesh mesh-stage function — same wire tag as classic fragment (`0x02`).
pub const PIPELINE_TAG_MESH_FUNC: u8 = PIPELINE_TAG_FRAGMENT_FUNC;

pub const VERTEX_DESC_TAG_ATTRIBUTES: u8 = 0x00;
pub const VERTEX_DESC_TAG_LAYOUTS: u8 = 0x01;
pub const VERTEX_ATTR_TAG_LOCATION: u8 = 0x00;
pub const VERTEX_ATTR_TAG_FORMAT: u8 = 0x01;
pub const VERTEX_ATTR_TAG_OFFSET: u8 = 0x02;
pub const VERTEX_ATTR_TAG_BUFFER_INDEX: u8 = 0x03;
pub const VERTEX_LAYOUT_TAG_BUFFER_INDEX: u8 = 0x00;
pub const VERTEX_LAYOUT_TAG_STEP_FUNCTION: u8 = 0x01;
pub const VERTEX_LAYOUT_TAG_STEP_RATE: u8 = 0x02;
pub const VERTEX_LAYOUT_TAG_STRIDE: u8 = 0x03;
pub const MAX_VERTEX_ATTRS: usize = 31;
pub const MAX_VERTEX_LAYOUTS: usize = 31;
pub const VERTEX_LABEL_MIN_ASCII: u8 = 0x20;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FunctionDescriptor {
    pub blob_gva: u64,
    pub blob_size: u32,
    pub function_id: u32,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SamplerDescriptor {
    pub min_filter: u32,
    pub mag_filter: u32,
    pub mip_filter: u32,
    pub s_address: u32,
    pub t_address: u32,
    pub r_address: u32,
    pub max_anisotropy: u32,
    pub lod_min_clamp: f32,
    pub lod_max_clamp: f32,
    pub compare_function: u32,
    pub border_color: u32,
    pub normalized_coordinates: bool,
    pub support_argument_buffers: bool,
    pub lod_average: bool,
}

/// Type-7 depth-stencil face (12 bytes).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DepthStencilFace {
    pub compare_function: u32,
    pub stencil_failure_operation: u32,
    pub depth_failure_operation: u32,
    pub depth_stencil_pass_operation: u32,
    pub read_mask: u32,
    pub write_mask: u32,
}

/// Type-7 depth-stencil state (40 bytes).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DepthStencilDescriptor {
    pub depth_stencil_id: u32,
    pub depth_compare_function: u32,
    pub depth_write_enabled: bool,
    pub front_stencil_enabled: bool,
    pub back_stencil_enabled: bool,
    pub front_face: DepthStencilFace,
    pub back_face: DepthStencilFace,
}

pub const DEPTH_STENCIL_DESC_LEN: usize = 40;
pub const DEPTH_STENCIL_DESC_STATE_BITS: usize = 12;
pub const DEPTH_STENCIL_DESC_ID: usize = 8;
pub const DEPTH_STENCIL_DESC_FRONT_FACE: usize = 16;
pub const DEPTH_STENCIL_DESC_BACK_FACE: usize = 28;
pub const DEPTH_STENCIL_DEPTH_COMPARE_MASK: u32 = 0x7;
pub const DEPTH_STENCIL_DEPTH_WRITE: u32 = 1 << 3;
pub const DEPTH_STENCIL_FRONT_STENCIL_ENABLED: u32 = 1 << 4;
pub const DEPTH_STENCIL_BACK_STENCIL_ENABLED: u32 = 1 << 5;

/// Per-command-slot layout offsets inside the ICB backing buffer.
///
/// Type encoding from `AppleParavirtIndirectCommandBuffer._commandLayout`,
/// embedded at create-body `+0x1c` (52 bytes).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct IcbCommandLayout {
    pub command_type_offset: u16,
    pub barrier_offset: u16,
    pub kernel_dispatch_arguments_offset: u16,
    pub tessellation_factor_offset: u16,
    pub pipeline_state_offset: u32,
    pub vertex_buffer_bind_offset: u32,
    pub fragment_buffer_bind_offset: u32,
    pub object_buffer_bind_offset: u32,
    pub mesh_buffer_bind_offset: u32,
    pub kernel_buffer_bind_offset: u32,
    pub attribute_stride_offset: u32,
    pub object_threadgroup_memory_length_offset: u32,
    pub threadgroup_memory_length_offset: u32,
    pub command_arguments_offset: u32,
    pub command_size: u32,
}

/// Decoded type-7 ICB create descriptor (88-byte MetalSerializer body).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct IndirectCommandBufferDescriptor {
    pub command_types: u32,
    pub max_vertex_buffer_bind_count: u16,
    pub max_fragment_buffer_bind_count: u16,
    pub max_kernel_buffer_bind_count: u16,
    /// `maxObjectBufferBindCount` create body `+0x0f`.
    pub max_object_buffer_bind_count: u16,
    /// `maxMeshBufferBindCount` create body `+0x10`.
    pub max_mesh_buffer_bind_count: u16,
    /// `maxKernelThreadgroupMemoryBindCount` create body `+0x11`.
    pub max_kernel_threadgroup_memory_bind_count: u16,
    /// `maxObjectThreadgroupMemoryBindCount` create body `+0x12`.
    pub max_object_threadgroup_memory_bind_count: u16,
    pub inherit_buffers: bool,
    pub inherit_pipeline_state: bool,
    pub max_command_count: u32,
    pub options: u32,
    /// Layout for decoding filled command slots in the ICB backing buffer.
    pub layout: IcbCommandLayout,
}

/// One decoded object-list descriptor.
///
/// There is deliberately no `Unknown`: an object type this host has no contract
/// for is [`DecodeStatus::ErrUnknownType`], and a type-7 subtype it does not
/// implement is [`DecodeStatus::ErrUnsupported`]. Both name the check that
/// refused. An `Unknown` variant would let the same condition arrive as a
/// successful decode carrying nothing, which every consumer would then have to
/// re-refuse without knowing why.
#[derive(Clone, Debug, PartialEq)]
pub enum Descriptor {
    Buffer(BufferDescriptor),
    Texture(TextureDescriptor),
    Sampler(SamplerDescriptor),
    Function(FunctionDescriptor),
    RenderPipeline(RenderPipelineDescriptor),
    ComputePipeline(ComputePipelineDescriptor),
    DepthStencil(DepthStencilDescriptor),
    TextureView(TextureViewDescriptor),
    IOSurfaceTexture {
        mapping_id: u32,
        object_ref: u32,
        pixel_format: u16,
        width: u32,
        height: u32,
    },
    IndirectCommandBuffer(IndirectCommandBufferDescriptor),
}

/// Live Reims VGPU object-list entry size (kb + reims-vgpu-resource-format).
pub const OBJECT_LIST_ENTRY_LEN: usize = 12;
pub const OBJECT_LIST_ENTRY_HEADER: usize = 0;
pub const OBJECT_LIST_ENTRY_DESC_GVA: usize = 4;
pub const OBJECT_TYPE_MASK: u32 = 0xff;
pub const OBJECT_DESC_LEN_SHIFT: u32 = 8;

/// Wire object-list entry: `[type:u8 | desc_len:u24]<< packed u32` + `desc_gva:u64`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ListObjectEntry {
    pub object_type: u8,
    pub descriptor_length: u32,
    pub descriptor_gva: u64,
}

/// Decode one 12-byte object-list entry (live arm Reims VGPU contract).
pub fn decode_list_object_entry(bytes: &[u8]) -> Result<ListObjectEntry, DecodeStatus> {
    if bytes.len() < OBJECT_LIST_ENTRY_LEN {
        return Err(DecodeStatus::ErrShort("res_list_entry_short"));
    }
    let first = ld32(&bytes[OBJECT_LIST_ENTRY_HEADER..]);
    Ok(ListObjectEntry {
        object_type: (first & OBJECT_TYPE_MASK) as u8,
        descriptor_length: first >> OBJECT_DESC_LEN_SHIFT,
        descriptor_gva: ld64(&bytes[OBJECT_LIST_ENTRY_DESC_GVA..]),
    })
}

/// Byte offset of object-list slot `ref_` (0-based index; ref_ < entry_count).
pub fn list_object_entry_offset(ref_: u32, entry_count: u32) -> Option<u64> {
    if ref_ >= entry_count {
        return None;
    }
    (ref_ as u64).checked_mul(OBJECT_LIST_ENTRY_LEN as u64)
}

pub fn decode_buffer_descriptor(bytes: &[u8]) -> Result<BufferDescriptor, DecodeStatus> {
    if bytes.len() < LINEAR_DESC_MIN_LEN {
        return Err(DecodeStatus::ErrShort("res_buffer_desc_short"));
    }
    let handle64 = ld64(&bytes[LINEAR_DESC_HANDLE..]);
    Ok(BufferDescriptor {
        allocation_size: ld64(&bytes[LINEAR_DESC_SIZE..]),
        handle64,
        handle: handle64 as u32,
    })
}

pub fn decode_texture_descriptor(bytes: &[u8]) -> Result<TextureDescriptor, DecodeStatus> {
    if bytes.len() < TEXTURE_DESC_GEOMETRY_LEN {
        return Err(DecodeStatus::ErrShort("res_texture_desc_short"));
    }
    let mut out = TextureDescriptor {
        allocation_size: ld64(&bytes[LINEAR_DESC_SIZE..]),
        handle: ld32(&bytes[LINEAR_DESC_HANDLE..]),
        ..Default::default()
    };
    if bytes.len() >= TEXTURE_DESC_MIPMAP_LEVEL_COUNT + 2 {
        out.mipmap_level_count = ld16(&bytes[TEXTURE_DESC_MIPMAP_LEVEL_COUNT..]) as u32;
    }
    if bytes.len() >= TEXTURE_DESC_DATA_OFFSET + 4 {
        out.data_offset = ld32(&bytes[TEXTURE_DESC_DATA_OFFSET..]);
    }
    if bytes.len() > TEXTURE_DESC_BYTES_PER_ELEMENT {
        out.bytes_per_element = bytes[TEXTURE_DESC_BYTES_PER_ELEMENT];
    }
    if bytes.len() >= TEXTURE_DESC_USED_SIZE + 4 {
        out.used_size = ld32(&bytes[TEXTURE_DESC_USED_SIZE..]);
    }
    if bytes.len() >= TEXTURE_DESC_ROW_STRIDE + 4 {
        out.row_stride = ld32(&bytes[TEXTURE_DESC_ROW_STRIDE..]);
        out.has_row_stride = out.row_stride != 0;
    }
    if bytes.len() >= TEXTURE_DESC_WIDTH + 4 {
        out.width = ld32(&bytes[TEXTURE_DESC_WIDTH..]);
        out.has_width = out.width != 0;
    }
    if bytes.len() >= TEXTURE_DESC_HEIGHT + 4 {
        out.height = ld32(&bytes[TEXTURE_DESC_HEIGHT..]);
        out.has_height = out.height != 0;
    }
    if bytes.len() >= TEXTURE_DESC_DEPTH + 4 {
        out.depth = ld32(&bytes[TEXTURE_DESC_DEPTH..]);
        if out.depth == 0 {
            out.depth = 1;
        }
    } else {
        out.depth = 1;
    }

    // Level layouts: L0 from geometry prefix; L1.. from records at +72.
    let declared_levels = if out.mipmap_level_count > 0 {
        out.mipmap_level_count
    } else {
        1
    };
    if out.has_width && out.has_height {
        let l0_size = if out.used_size != 0 {
            out.used_size as u64
        } else if out.has_row_stride && out.height > 0 {
            (out.row_stride as u64).saturating_mul(out.height as u64)
        } else {
            0
        };
        out.levels.push(TextureLevelLayout {
            offset: out.data_offset as u64,
            size: l0_size,
            row_stride: out.row_stride as u64,
            width: out.width,
            height: out.height,
            depth: if out.depth == 0 { 1 } else { out.depth },
        });
        if declared_levels > 1 {
            let mut rec_off = TEXTURE_DESC_LEVEL_RECORDS;
            let max_extra = (declared_levels as usize - 1).min(TEXTURE_MAX_MIP_LEVELS - 1);
            // Both truncations below leave `mipmap_level_count` at what the
            // guest declared while `levels` holds fewer, so `level(n)` answers
            // `None` for a level the descriptor named. That is a level of a
            // texture this device will not sample or blit, and it has to be
            // legible as a drop rather than as an absence.
            if declared_levels as usize - 1 > max_extra
                && crate::observe::first_sight(
                    "texture_desc_levels_over_cap",
                    u64::from(declared_levels),
                )
            {
                crate::observe::fail(format!(
                    "texture_desc_levels_over_cap declared={declared_levels} \
                     cap={TEXTURE_MAX_MIP_LEVELS}"
                ));
            }
            for _ in 0..max_extra {
                if rec_off + TEXTURE_DESC_MIP_LEVEL_RECORD_LEN > bytes.len() {
                    if crate::observe::first_sight(
                        "texture_desc_level_record_short",
                        u64::from(declared_levels),
                    ) {
                        crate::observe::fail(format!(
                            "texture_desc_level_record_short declared={declared_levels} \
                             decoded={} rec_off={rec_off} len={} \
                             (body ends before a level record the descriptor named)",
                            out.levels.len(),
                            bytes.len()
                        ));
                    }
                    break;
                }
                let rec = &bytes[rec_off..rec_off + TEXTURE_DESC_MIP_LEVEL_RECORD_LEN];
                let mut depth = ld32(&rec[TEXTURE_LEVEL_DEPTH..]);
                if depth == 0 {
                    depth = 1;
                }
                out.levels.push(TextureLevelLayout {
                    offset: ld64(&rec[TEXTURE_LEVEL_OFFSET..]),
                    size: ld64(&rec[TEXTURE_LEVEL_SIZE..]),
                    row_stride: ld64(&rec[TEXTURE_LEVEL_ROW_STRIDE..]),
                    width: ld32(&rec[TEXTURE_LEVEL_WIDTH..]),
                    height: ld32(&rec[TEXTURE_LEVEL_HEIGHT..]),
                    depth,
                });
                rec_off += TEXTURE_DESC_MIP_LEVEL_RECORD_LEN;
            }
        }
    }

    // Format trailer: shift by (levels-1)*36 for multi-mip bodies.
    let levels = declared_levels;
    let format_shift = if levels > 1 {
        (levels as usize - 1).saturating_mul(TEXTURE_DESC_MIP_LEVEL_RECORD_LEN)
    } else {
        0
    };
    let pf_off = TEXTURE_DESC_PIXEL_FORMAT + format_shift;
    if bytes.len() >= pf_off + 2 {
        out.pixel_format = ld16(&bytes[pf_off..]);
        out.has_pixel_format = out.pixel_format != 0;
    } else if crate::observe::first_sight("texture_desc_format_unreachable", levels as u64) {
        // No fallback to the unshifted offset. The fallback's own length test
        // was `TEXTURE_DESC_PIXEL_FORMAT + 2`, so for a single-mip body it
        // guarded the same two bytes the branch above already read. It
        // was reachable only when `format_shift > 0` — and there offset 86 is
        // not the format at all: the level records start at 72 and run 36 bytes
        // each, so 86..88 is inside level record 1. The fallback therefore
        // produced a format only in the case where it was guaranteed to be
        // reading something else, and then set `has_pixel_format`, which is
        // what seven downstream gates fail closed on. Better to have no format.
        crate::observe::fail(format!(
            "texture_desc_format_unreachable levels={levels} pf_off={pf_off} len={} \
             (multi-mip body too short for its shifted format trailer)",
            bytes.len()
        ));
    }
    Ok(out)
}

pub fn decode_function_descriptor(bytes: &[u8]) -> Result<FunctionDescriptor, DecodeStatus> {
    if bytes.len() < FUNCTION_DESC_MIN_LEN {
        return Err(DecodeStatus::ErrShort("res_function_desc_short"));
    }
    let blob_gva = ld64(&bytes[FUNCTION_DESC_BLOB_GVA..]);
    let blob_size = ld32(&bytes[FUNCTION_DESC_BLOB_SIZE..]);
    let function_id = if bytes.len() >= FUNCTION_DESC_FUNCTION_ID + 4 {
        ld32(&bytes[FUNCTION_DESC_FUNCTION_ID..])
    } else {
        0
    };
    Ok(FunctionDescriptor {
        blob_gva,
        blob_size,
        function_id,
    })
}

/// Compact type-7 first sub-record: `[fieldCount:u8]` × `[tag:u8][len:u8][value…]`.
pub fn decode_compact_tlv_record(
    bytes: &[u8],
    offset: usize,
) -> Result<(Vec<CompactTlv>, usize), DecodeStatus> {
    if offset >= bytes.len() {
        return Err(DecodeStatus::ErrShort("res_tlv_offset_past_end"));
    }
    let field_count = bytes[offset] as usize;
    let mut p = offset + 1;
    let mut out = Vec::with_capacity(field_count);
    for _ in 0..field_count {
        if p + 2 > bytes.len() {
            return Err(DecodeStatus::ErrShort("res_tlv_header_short"));
        }
        let tag = bytes[p];
        let field_len = bytes[p + 1] as usize;
        if p + 2 + field_len > bytes.len() {
            return Err(DecodeStatus::ErrShort("res_tlv_value_short"));
        }
        let value_offset = p + 2;
        let (has_u32, value_u32) = if field_len >= 4 {
            (true, ld32(&bytes[value_offset..]))
        } else {
            (false, 0)
        };
        out.push(CompactTlv {
            tag,
            length: field_len as u8,
            value_offset,
            value_u32,
            has_u32,
        });
        p += 2 + field_len;
    }
    Ok((out, p - offset))
}

fn compact_tlv_u32(fields: &[CompactTlv], tag: u8) -> Option<u32> {
    fields
        .iter()
        .find(|f| f.tag == tag && f.has_u32)
        .map(|f| f.value_u32)
}

/// [`entry_tag_u32_present`] with a value for "the entry does not carry `tag`".
///
/// The two used to be written out separately, with identical control flow and
/// five identical bounds checks, differing only in what they returned when the
/// walk fell through. A walk written twice is a walk that can be fixed once.
fn entry_tag_u32(bytes: &[u8], len: usize, entry_off: usize, tag: u8, default: u32) -> u32 {
    entry_tag_u32_present(bytes, len, entry_off, tag).unwrap_or(default)
}

fn entry_tag_u32_present(bytes: &[u8], len: usize, entry_off: usize, tag: u8) -> Option<u32> {
    if entry_off >= len {
        return None;
    }
    let field_count = bytes[entry_off] as usize;
    let mut p = entry_off + 1;
    for _ in 0..field_count {
        if p + 2 > len {
            return None;
        }
        let t = bytes[p];
        let field_len = bytes[p + 1] as usize;
        if p + 2 + field_len > len {
            return None;
        }
        if t == tag && field_len >= 4 {
            return Some(ld32(&bytes[p + 2..]));
        }
        p += 2 + field_len;
    }
    None
}

fn skip_optional_label_and_pad(bytes: &[u8], end: usize, mut off: usize) -> usize {
    if off < end && bytes[off] >= VERTEX_LABEL_MIN_ASCII {
        while off < end && bytes[off] != 0 {
            off += 1;
        }
    }
    while off < end && bytes[off] == 0 {
        off += 1;
    }
    off
}

/// Parse vertex-input block between first TLVs and color-attachment section.
pub fn parse_vertex_block(
    bytes: &[u8],
    block_start: usize,
    block_end: usize,
) -> Result<Vec<VertexAttribute>, DecodeStatus> {
    if block_start >= block_end || block_end > bytes.len() {
        return Ok(Vec::new());
    }
    let bo = skip_optional_label_and_pad(bytes, block_end, block_start);
    if bo >= block_end {
        return Ok(Vec::new());
    }
    let attr_off = entry_tag_u32(bytes, block_end, bo, VERTEX_DESC_TAG_ATTRIBUTES, u32::MAX);
    let layout_off = entry_tag_u32(bytes, block_end, bo, VERTEX_DESC_TAG_LAYOUTS, u32::MAX);
    if attr_off == u32::MAX || layout_off == u32::MAX {
        return Ok(Vec::new());
    }

    let mut strides = [0u32; MAX_VERTEX_LAYOUTS];
    let mut have_stride = [false; MAX_VERTEX_LAYOUTS];
    let mut layout_steps: Vec<(u32, bool, u32, bool, u32)> = Vec::new(); // bi, has_sf, sf, has_sr, sr

    let layout_section = bo.saturating_add(layout_off as usize);
    if layout_section + 4 > block_end {
        return Err(DecodeStatus::ErrShort("res_vertex_layout_count_oob"));
    }
    let layout_count = ld32(&bytes[layout_section..]) as usize;
    for i in 0..layout_count {
        let offloc = layout_section + 4 + i * 4;
        if offloc + 4 > block_end {
            return Err(DecodeStatus::ErrShort("res_vertex_layout_offset_oob"));
        }
        let entry = layout_section + ld32(&bytes[offloc..]) as usize;
        if entry >= block_end {
            return Err(DecodeStatus::ErrShort("res_vertex_layout_entry_oob"));
        }
        let buffer_index =
            entry_tag_u32(bytes, block_end, entry, VERTEX_LAYOUT_TAG_BUFFER_INDEX, 0);
        let stride = entry_tag_u32(bytes, block_end, entry, VERTEX_LAYOUT_TAG_STRIDE, 0);
        let has_step_function =
            entry_tag_u32_present(bytes, block_end, entry, VERTEX_LAYOUT_TAG_STEP_FUNCTION);
        let has_step_rate =
            entry_tag_u32_present(bytes, block_end, entry, VERTEX_LAYOUT_TAG_STEP_RATE);
        if (buffer_index as usize) < MAX_VERTEX_LAYOUTS && stride != 0 {
            strides[buffer_index as usize] = stride;
            have_stride[buffer_index as usize] = true;
        }
        layout_steps.push((
            buffer_index,
            has_step_function.is_some(),
            has_step_function.unwrap_or(0),
            has_step_rate.is_some(),
            has_step_rate.unwrap_or(0),
        ));
    }

    let attr_section = bo.saturating_add(attr_off as usize);
    if attr_section + 4 > block_end {
        return Err(DecodeStatus::ErrShort("res_vertex_attr_count_oob"));
    }
    let attr_count = ld32(&bytes[attr_section..]) as usize;
    let mut attrs = Vec::new();
    for i in 0..attr_count {
        if attrs.len() >= MAX_VERTEX_ATTRS {
            break;
        }
        let offloc = attr_section + 4 + i * 4;
        if offloc + 4 > block_end {
            return Err(DecodeStatus::ErrShort("res_vertex_attr_offset_oob"));
        }
        let entry = attr_section + ld32(&bytes[offloc..]) as usize;
        if entry >= block_end {
            return Err(DecodeStatus::ErrShort("res_vertex_attr_entry_oob"));
        }
        let location = entry_tag_u32(bytes, block_end, entry, VERTEX_ATTR_TAG_LOCATION, i as u32);
        let format = entry_tag_u32(bytes, block_end, entry, VERTEX_ATTR_TAG_FORMAT, 0);
        let offset = entry_tag_u32(bytes, block_end, entry, VERTEX_ATTR_TAG_OFFSET, 0);
        let buffer_index = entry_tag_u32(bytes, block_end, entry, VERTEX_ATTR_TAG_BUFFER_INDEX, 0);
        let stride =
            if (buffer_index as usize) < MAX_VERTEX_LAYOUTS && have_stride[buffer_index as usize] {
                strides[buffer_index as usize]
            } else {
                0
            };
        let (has_sf, sf, has_sr, sr) = layout_steps
            .iter()
            .find(|(bi, ..)| *bi == buffer_index)
            .map(|(_, has_sf, sf, has_sr, sr)| (*has_sf, *sf, *has_sr, *sr))
            .unwrap_or((false, 0, false, 0));
        attrs.push(VertexAttribute {
            location,
            format,
            offset,
            buffer_index,
            stride,
            has_step_function: has_sf,
            step_function: sf,
            has_step_rate: has_sr,
            step_rate: sr,
        });
    }
    Ok(attrs)
}

fn parse_depth_stencil_face(bytes: &[u8]) -> DepthStencilFace {
    let bits = ld32(&bytes[0..]);
    DepthStencilFace {
        compare_function: bits & 0x7,
        stencil_failure_operation: (bits >> 3) & 0x7,
        depth_failure_operation: (bits >> 6) & 0x7,
        depth_stencil_pass_operation: (bits >> 9) & 0x7,
        read_mask: ld32(&bytes[4..]),
        write_mask: ld32(&bytes[8..]),
    }
}

pub fn decode_depth_stencil_descriptor(
    bytes: &[u8],
) -> Result<DepthStencilDescriptor, DecodeStatus> {
    if bytes.len() < DEPTH_STENCIL_DESC_LEN {
        return Err(DecodeStatus::ErrShort("res_depth_stencil_short"));
    }
    let tag = ld32(&bytes[0..]);
    let declared = ld32(&bytes[4..]);
    if tag != TYPE7_OBJECT_DEPTH_STENCIL || declared as usize != bytes.len() {
        return Err(DecodeStatus::ErrUnsupported("res_depth_stencil_tag"));
    }
    let state_bits = ld32(&bytes[DEPTH_STENCIL_DESC_STATE_BITS..]);
    Ok(DepthStencilDescriptor {
        depth_stencil_id: ld32(&bytes[DEPTH_STENCIL_DESC_ID..]),
        depth_compare_function: state_bits & DEPTH_STENCIL_DEPTH_COMPARE_MASK,
        depth_write_enabled: (state_bits & DEPTH_STENCIL_DEPTH_WRITE) != 0,
        front_stencil_enabled: (state_bits & DEPTH_STENCIL_FRONT_STENCIL_ENABLED) != 0,
        back_stencil_enabled: (state_bits & DEPTH_STENCIL_BACK_STENCIL_ENABLED) != 0,
        front_face: parse_depth_stencil_face(&bytes[DEPTH_STENCIL_DESC_FRONT_FACE..]),
        back_face: parse_depth_stencil_face(&bytes[DEPTH_STENCIL_DESC_BACK_FACE..]),
    })
}

pub fn decode_sampler_descriptor(bytes: &[u8]) -> Result<SamplerDescriptor, DecodeStatus> {
    if bytes.len() < SAMPLER_DESC_LEN {
        return Err(DecodeStatus::ErrShort("res_sampler_short"));
    }
    let tag = ld32(&bytes[SAMPLER_DESC_TAG..]);
    let declared = ld32(&bytes[SAMPLER_DESC_DECLARED_LEN..]);
    if tag != TYPE7_OBJECT_SAMPLER || declared as usize != bytes.len() {
        return Err(DecodeStatus::ErrUnsupported("res_sampler_tag"));
    }
    let bits = ld32(&bytes[SAMPLER_DESC_STATE_BITS..]);
    const MIN_MAG_MASK: u32 = 0x1;
    const MIP_MASK: u32 = 0x3;
    const COMPARE_MASK: u32 = 0x7;
    const ADDR_MASK: u32 = 0x7;
    const BORDER_MASK: u32 = 0x3;
    const ANISO_MASK: u32 = 0x1f;
    const LOD_AVERAGE: u32 = 1 << 4;
    const NORM_COORDS: u32 = 0x8000_0000;
    let word16 = ld32(&bytes[SAMPLER_DESC_WORD16..]);
    Ok(SamplerDescriptor {
        min_filter: bits & MIN_MAG_MASK,
        mag_filter: (bits >> 2) & MIN_MAG_MASK,
        mip_filter: (bits >> 24) & MIP_MASK,
        s_address: (bits >> 8) & ADDR_MASK,
        t_address: (bits >> 12) & ADDR_MASK,
        r_address: (bits >> 16) & ADDR_MASK,
        max_anisotropy: ((bits >> 26) & ANISO_MASK).max(1),
        lod_min_clamp: f32::from_bits(ld32(&bytes[SAMPLER_DESC_WORD20..])),
        lod_max_clamp: f32::from_bits(ld32(&bytes[SAMPLER_DESC_LOD_MAX..])),
        compare_function: (bits >> 5) & COMPARE_MASK,
        border_color: (bits >> 20) & BORDER_MASK,
        normalized_coordinates: (bits & NORM_COORDS) != 0,
        support_argument_buffers: (word16 & 1) != 0,
        lod_average: (bits & LOD_AVERAGE) != 0,
    })
}

pub fn decode_render_pipeline_descriptor(
    bytes: &[u8],
) -> Result<RenderPipelineDescriptor, DecodeStatus> {
    if bytes.len() < TYPE7_MIN_LEN {
        return Err(DecodeStatus::ErrShort("res_render_pipeline_short"));
    }
    let obj_type = ld32(&bytes[0..]);
    let declared = ld32(&bytes[4..]) as usize;
    if obj_type != TYPE7_OBJECT_RENDER_PIPELINE {
        return Err(DecodeStatus::ErrUnsupported("res_render_pipeline_tag"));
    }
    if declared != bytes.len() || declared < TYPE7_MIN_LEN {
        return Err(DecodeStatus::ErrShort("res_render_pipeline_declared_len"));
    }
    let mut out = RenderPipelineDescriptor {
        object_id: ld32(&bytes[8..]),
        word3: ld32(&bytes[12..]),
        ..Default::default()
    };
    let (fields, consumed) = decode_compact_tlv_record(bytes, TYPE7_FIRST_TLVS)?;
    let tag01 = compact_tlv_u32(&fields, PIPELINE_TAG_VERTEX_FUNC).unwrap_or(0);
    let tag02 = compact_tlv_u32(&fields, PIPELINE_TAG_FRAGMENT_FUNC).unwrap_or(0);
    let tag03 = compact_tlv_u32(&fields, PIPELINE_TAG_MESH_FRAGMENT_FUNC).unwrap_or(0);
    // Mesh SPI shape: tag 0x14 section offset (host serializeMeshRenderPipelineDescriptor).
    // Classic type-7 uses tag 0x08. Roles for 0x01/0x02/0x03 differ by shape.
    if let Some(off) = compact_tlv_u32(&fields, PIPELINE_TAG_MESH_SECTION_OFFSET) {
        out.object_func_ref = tag01;
        out.mesh_func_ref = tag02;
        out.fragment_func_ref = tag03;
        out.vertex_func_ref = 0;
        out.color_attachment_offset = off;
        out.has_color_attachment_offset = true;
    } else {
        out.vertex_func_ref = tag01;
        out.fragment_func_ref = tag02;
        out.object_func_ref = 0;
        out.mesh_func_ref = 0;
        if let Some(off) = compact_tlv_u32(&fields, PIPELINE_TAG_COLOR_ATTACH_OFFSET) {
            out.color_attachment_offset = off;
            out.has_color_attachment_offset = true;
        }
    }
    let first_tlv_end = TYPE7_FIRST_TLVS + consumed;
    if out.has_color_attachment_offset {
        let color_abs = TYPE7_FIRST_TLVS + out.color_attachment_offset as usize;
        if color_abs <= declared && first_tlv_end < color_abs {
            out.vertex_attributes = parse_vertex_block(bytes, first_tlv_end, color_abs)?;
        }
        if color_abs < declared {
            out.color_attachments = parse_color_attachments(bytes, declared, color_abs);
            if let Some(c0) = out.color_attachments.first().copied() {
                out.color0 = c0;
            }
        }
    }
    Ok(out)
}

/// A colour-attachment TLV field this decoder does not read.
///
/// The entry is `[field_count][tag][len][value…]*` and eight tags are consumed
/// (`COLOR_ATTACHMENT_TAG_PIXEL_FORMAT` through `..._ALPHA_OP`). Anything else
/// is a field the guest serialized and we dropped on the floor — including,
/// per `translate/coverage.rs`, `colorAttachments[n].writeMask`, whose position
/// in this block is unknown and is an RE question rather than a guess.
struct ColorAttachDropped {
    tag: u8,
}

impl crate::observe::Decline for ColorAttachDropped {
    fn slug(&self) -> &'static str {
        "color_attachment_field_dropped"
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        vec![("tag", format!("0x{:02x}", self.tag))]
    }
}

const COLOR_ATTACHMENT_TAGS_CONSUMED: [u8; 9] = [
    COLOR_ATTACHMENT_TAG_PIXEL_FORMAT,
    COLOR_ATTACHMENT_TAG_BLEND_ENABLE,
    COLOR_ATTACHMENT_TAG_SRC_RGB,
    COLOR_ATTACHMENT_TAG_DST_RGB,
    COLOR_ATTACHMENT_TAG_RGB_OP,
    COLOR_ATTACHMENT_TAG_SRC_ALPHA,
    COLOR_ATTACHMENT_TAG_DST_ALPHA,
    COLOR_ATTACHMENT_TAG_ALPHA_OP,
    COLOR_ATTACHMENT_TAG_WRITE_MASK,
];

/// A colour-attachment `writeMask` outside `MTLColorWriteMask`'s four bits.
///
/// This is the standing check on the tag identification itself. Tag `0x09` is
/// `writeMask` because it is the ninth property in `MTLRenderPipeline.h` and
/// tags `0x01..=0x08` are the first eight in order — an argument from the
/// header, not from the one observed value. If the tag is something else, it
/// will eventually carry a value no four-bit mask can hold, and that value
/// arrives here by name instead of quietly masking channels off.
struct ColorWriteMaskOutOfRange;

impl crate::observe::Decline for ColorWriteMaskOutOfRange {
    fn slug(&self) -> &'static str {
        "color_write_mask_out_of_range"
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        Vec::new()
    }
}

/// The largest field value that rides in the dedup key.
///
/// A dropped field's *value distribution* is the identifying signal — an
/// `MTLColorWriteMask` only ever takes 0..=15, which names the field from the
/// wire alone without knowing its tag in advance. So small values each get
/// their own line. A field carrying wide values would otherwise make this
/// unbounded, so above the cap the key collapses to the tag and only the first
/// value seen is printed.
const COLOR_ATTACH_DROP_VALUE_CAP: u32 = 64;

/// Report the shape of one colour-attachment entry and every field in it this
/// decoder does not consume.
///
/// Two lines with different jobs, because a silent census cannot distinguish
/// "the guest sends nothing but the eight tags we read" from "this walk never
/// ran on a live guest":
///
/// * `type7_color_attach_shape` is the *branch*, deduped per distinct `(tag,
///   len)` sequence. A boot with entries but no drop line is then a positive
///   reading — the entries were seen and carried only consumed tags — rather
///   than an absence.
/// * `color_attachment_field_dropped` is the *loss*, one typed decline per
///   dropped `(tag, len, value)`. A serialized field we never read is guest
///   intent discarded, which the ground rules say must not be silent.
///
/// Pipeline descriptors are decoded once per distinct pipeline and cached, so
/// this walk is not on a per-draw path.
fn note_color_entry_fields(bytes: &[u8], len: usize, entry: usize, slot: u32) {
    if entry >= len {
        return;
    }
    let field_count = bytes[entry] as usize;
    let mut p = entry + 1;
    let mut shape = String::new();
    let mut shape_key: u64 = 0;
    let mut dropped: Vec<(u8, u8, u32)> = Vec::new();
    for _ in 0..field_count {
        if p + 2 > len {
            break;
        }
        let tag = bytes[p];
        let field_len = bytes[p + 1] as usize;
        if p + 2 + field_len > len {
            break;
        }
        let value = if field_len >= 4 {
            ld32(&bytes[p + 2..])
        } else {
            0
        };
        let consumed = COLOR_ATTACHMENT_TAGS_CONSUMED.contains(&tag);
        let sep = if shape.is_empty() { "" } else { "," };
        let star = if consumed { "" } else { "*" };
        let _ = std::fmt::Write::write_fmt(
            &mut shape,
            format_args!("{sep}{tag:02x}:{field_len}{star}"),
        );
        // Order-sensitive so a reordered entry reads as a different shape; the
        // tag and length are what the walk depends on, the value is not.
        shape_key = shape_key.rotate_left(9) ^ (u64::from(tag) << 8) ^ (field_len as u64);
        if !consumed {
            dropped.push((tag, field_len as u8, value));
        }
        p += 2 + field_len;
    }
    if crate::observe::first_sight("type7_color_attach_shape", shape_key) {
        crate::observe::off(format!(
            "type7_color_attach_shape slot={slot} nfields={field_count} \
             tags=[{shape}] unconsumed={}",
            dropped.len()
        ));
    }
    for (tag, field_len, value) in dropped {
        let keyed_value = if value <= COLOR_ATTACH_DROP_VALUE_CAP {
            u64::from(value)
        } else {
            u64::from(COLOR_ATTACH_DROP_VALUE_CAP) + 1
        };
        let disc = (u64::from(tag) << 40) | (u64::from(field_len) << 32) | keyed_value;
        if !crate::observe::first_sight("color_attachment_field_dropped", disc) {
            continue;
        }
        crate::observe::Emit::decline("type7_color_attach", &ColorAttachDropped { tag })
            .field("slot", slot)
            .field("len", field_len)
            .field("value", value)
            .fail();
    }
}

fn parse_one_color_entry(
    bytes: &[u8],
    len: usize,
    entry: usize,
    slot: u32,
) -> PipelineColorAttachment {
    note_color_entry_fields(bytes, len, entry, slot);
    let mut out = PipelineColorAttachment {
        slot,
        src_rgb: BLEND_FACTOR_ONE,
        dst_rgb: BLEND_FACTOR_ZERO,
        op_rgb: BLEND_OP_ADD,
        src_alpha: BLEND_FACTOR_ONE,
        dst_alpha: BLEND_FACTOR_ZERO,
        op_alpha: BLEND_OP_ADD,
        ..Default::default()
    };
    let pf = entry_tag_u32(
        bytes,
        len,
        entry,
        COLOR_ATTACHMENT_TAG_PIXEL_FORMAT,
        u32::MAX,
    );
    if pf != u32::MAX {
        out.has_pixel_format = true;
        out.pixel_format = pf;
    }
    out.blending_enabled =
        entry_tag_u32(bytes, len, entry, COLOR_ATTACHMENT_TAG_BLEND_ENABLE, 0) != 0;
    out.src_rgb = entry_tag_u32(
        bytes,
        len,
        entry,
        COLOR_ATTACHMENT_TAG_SRC_RGB,
        BLEND_FACTOR_ONE,
    );
    out.dst_rgb = entry_tag_u32(
        bytes,
        len,
        entry,
        COLOR_ATTACHMENT_TAG_DST_RGB,
        BLEND_FACTOR_ZERO,
    );
    out.op_rgb = entry_tag_u32(bytes, len, entry, COLOR_ATTACHMENT_TAG_RGB_OP, BLEND_OP_ADD);
    out.src_alpha = entry_tag_u32(
        bytes,
        len,
        entry,
        COLOR_ATTACHMENT_TAG_SRC_ALPHA,
        BLEND_FACTOR_ONE,
    );
    out.dst_alpha = entry_tag_u32(
        bytes,
        len,
        entry,
        COLOR_ATTACHMENT_TAG_DST_ALPHA,
        BLEND_FACTOR_ZERO,
    );
    out.op_alpha = entry_tag_u32(
        bytes,
        len,
        entry,
        COLOR_ATTACHMENT_TAG_ALPHA_OP,
        BLEND_OP_ADD,
    );
    // An entry that omits the tag left the property at its default, which for
    // `MTLColorWriteMask` is `all` — the same thing `ColorWriteMask::default()`
    // says, so the absent case needs no branch.
    if let Some(mask) = entry_tag_u32_present(bytes, len, entry, COLOR_ATTACHMENT_TAG_WRITE_MASK) {
        if let Some(decoded) = ColorWriteMask::new(mask) {
            out.write_mask = decoded;
        } else if crate::observe::first_sight("color_write_mask_out_of_range", u64::from(mask)) {
            crate::observe::Emit::decline("type7_color_attach", &ColorWriteMaskOutOfRange)
                .field("slot", slot)
                .field("value", mask)
                .fail();
        }
    }
    out
}

/// A color-attachment section that named more entries than it delivered.
///
/// The section is `[count:u32][entry_offset:u32 × count]`, each offset relative
/// to the section start. `count` above [`MAX_COLOR_ATTACHMENTS`], an offset word
/// running past the descriptor, or an entry offset resolving outside it, all
/// mean the same thing: the pixel format and blend state the guest serialized
/// for a slot never reaches the pipeline, and that slot silently takes
/// `parse_one_color_entry`'s defaults — opaque `ONE`/`ZERO`, blending off.
///
/// Named because the alternative is indistinguishable downstream from a guest
/// that declared fewer attachments, which is the shape a wrong blend or a
/// missing render target would arrive in.
struct ColorAttachTableTruncated {
    /// `None` when the section header itself did not fit, so the count was never
    /// readable and an unknown number of attachments were lost.
    declared: Option<usize>,
    decoded: usize,
}

impl crate::observe::Decline for ColorAttachTableTruncated {
    fn slug(&self) -> &'static str {
        "color_attachment_table_truncated"
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        vec![
            (
                "declared",
                self.declared
                    .map_or_else(|| "unreadable".to_string(), |d| d.to_string()),
            ),
            ("decoded", self.decoded.to_string()),
        ]
    }
}

/// Report a color-attachment table that lost entries. Deduped per distinct
/// (declared, decoded) pair — a malformed descriptor replayed every frame would
/// otherwise flood the log with one line per draw.
fn note_color_table_truncated(
    declared: Option<usize>,
    decoded: usize,
    section_off: usize,
    len: usize,
) {
    let disc = ((declared.unwrap_or(usize::MAX) as u64) << 32) | decoded as u64;
    if !crate::observe::first_sight("color_attachment_table_truncated", disc) {
        return;
    }
    crate::observe::Emit::decline(
        "type7_color_attach",
        &ColorAttachTableTruncated { declared, decoded },
    )
    .field("section_off", section_off)
    .field("desc_len", len)
    .fail();
}

/// `MTLRenderPipelineDescriptor.colorAttachments` is an eight-slot array, so a
/// section naming more than eight is malformed rather than something we chose
/// not to read. Same bound as `render::PASS_MAX_COLOR_ATTACHMENTS`, stated here
/// because this is the pipeline-descriptor side of the same Metal limit.
const MAX_COLOR_ATTACHMENTS: usize = 8;

/// Parse all color-attachment entries (slot = entry index in the section).
///
/// `section_off == 0` is the descriptor saying it has no color section at all —
/// expected control flow, and quiet. Every other early exit is a loss and says
/// so through [`ColorAttachTableTruncated`].
pub fn parse_color_attachments(
    bytes: &[u8],
    len: usize,
    section_off: usize,
) -> Vec<PipelineColorAttachment> {
    let mut out = Vec::new();
    if section_off == 0 {
        return out;
    }
    // The header is the count plus the first entry's offset word. A section the
    // descriptor cannot contain loses an unreadable number of attachments, which
    // the count mismatch below cannot see, so it is reported here.
    if section_off + 8 > len {
        note_color_table_truncated(None, 0, section_off, len);
        return out;
    }
    let declared = ld32(&bytes[section_off..]) as usize;
    for i in 0..declared.min(MAX_COLOR_ATTACHMENTS) {
        let offloc = section_off + 4 + i * 4;
        if offloc + 4 > len {
            break;
        }
        let entry_rel = ld32(&bytes[offloc..]) as usize;
        let entry = match section_off.checked_add(entry_rel) {
            Some(e) if e < len => e,
            _ => break,
        };
        out.push(parse_one_color_entry(bytes, len, entry, i as u32));
    }
    if out.len() != declared {
        note_color_table_truncated(Some(declared), out.len(), section_off, len);
    }
    out
}

pub fn decode_texture_view_descriptor(bytes: &[u8]) -> Result<TextureViewDescriptor, DecodeStatus> {
    if bytes.len() < TEXTURE_VIEW_MIN_SIMPLE {
        return Err(DecodeStatus::ErrShort("res_texture_view_short"));
    }
    let view_opcode = ld32(&bytes[TEXTURE_VIEW_DESC_OPCODE..]);
    let declared = ld32(&bytes[TEXTURE_VIEW_DESC_LEN..]) as usize;
    let min_len = match view_opcode {
        TEXTURE_VIEW_OPCODE_SIMPLE => TEXTURE_VIEW_MIN_SIMPLE,
        TEXTURE_VIEW_OPCODE_RANGED => TEXTURE_VIEW_MIN_RANGED,
        TEXTURE_VIEW_OPCODE_SWIZZLE => TEXTURE_VIEW_MIN_SWIZZLE,
        _ => return Err(DecodeStatus::ErrUnsupported("res_texture_view_opcode")),
    };
    if declared < min_len || declared != bytes.len() {
        return Err(DecodeStatus::ErrShort("res_texture_view_declared_len"));
    }
    let pixel_format = ld16(&bytes[TEXTURE_VIEW_DESC_PIXEL_FORMAT..]);
    let mut out = TextureViewDescriptor {
        view_opcode,
        view_texture_ref: ld32(&bytes[TEXTURE_VIEW_DESC_TEXTURE_REF..]),
        base_texture_ref: ld32(&bytes[TEXTURE_VIEW_DESC_BASE_REF..]),
        pixel_format,
        // `!= 0`, the same rule `decode_texture_descriptor` uses, and not the
        // unconditional `true` this used to carry. `MTLPixelFormatInvalid` is 0,
        // so a zero here is a format the descriptor did not state — and a flag
        // named `has_pixel_format` that answers yes for a format nobody named is
        // a trap for the first caller to gate on it. Every current reader of
        // this flag is on `TextureDescriptor`, so the two must not disagree
        // about what it means.
        has_pixel_format: pixel_format != 0,
        ..Default::default()
    };
    if view_opcode >= TEXTURE_VIEW_OPCODE_RANGED && bytes.len() >= TEXTURE_VIEW_MIN_RANGED {
        out.has_texture_type = true;
        out.texture_type = ld16(&bytes[TEXTURE_VIEW_DESC_TEXTURE_TYPE..]);
        out.has_levels = true;
        out.level_base = ld64(&bytes[TEXTURE_VIEW_DESC_LEVEL_BASE..]);
        out.level_count = ld64(&bytes[TEXTURE_VIEW_DESC_LEVEL_COUNT..]);
        out.has_slices = true;
        out.slice_base = ld64(&bytes[TEXTURE_VIEW_DESC_SLICE_BASE..]);
        out.slice_count = ld64(&bytes[TEXTURE_VIEW_DESC_SLICE_COUNT..]);
    }
    if view_opcode == TEXTURE_VIEW_OPCODE_SWIZZLE && bytes.len() >= TEXTURE_VIEW_MIN_SWIZZLE {
        out.has_swizzle = true;
        out.swizzle = [
            bytes[TEXTURE_VIEW_DESC_SWIZZLE],
            bytes[TEXTURE_VIEW_DESC_SWIZZLE + 1],
            bytes[TEXTURE_VIEW_DESC_SWIZZLE + 2],
            bytes[TEXTURE_VIEW_DESC_SWIZZLE + 3],
        ];
    }
    Ok(out)
}

/// A texture aliased over an MTLBuffer's storage — object type 8, view_opcode 9
/// (`newTextureWithDescriptor:offset:bytesPerRow:`). Distinct from a texture view:
/// the source ref is a BUFFER and the sampled bytes come straight from that
/// buffer's guest storage at `offset`, `bytes_per_row` stride.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BufferTextureDescriptor {
    pub new_texture_ref: u32,
    pub buffer_ref: u32,
    pub offset: u64,
    pub bytes_per_row: u64,
    pub texture_type: u16,
    pub pixel_format: u16,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub mipmap_level_count: u16,
    pub sample_count: u16,
    pub array_length: u16,
}

/// Decode the opcode-9 (buffer-backed texture) type-8 descriptor — the
/// serialized form of
/// `newTextureWithBuffer:descriptor:offset:bytesPerRow:allocator:`.
pub fn decode_buffer_texture_descriptor(
    bytes: &[u8],
) -> Result<BufferTextureDescriptor, DecodeStatus> {
    if bytes.len() < BUF_TEX_MIN_LEN {
        return Err(DecodeStatus::ErrShort("res_buffer_texture_short"));
    }
    let opcode = ld32(&bytes[TEXTURE_VIEW_DESC_OPCODE..]);
    if opcode != TEXTURE_VIEW_OPCODE_BUFFER_TEXTURE {
        return Err(DecodeStatus::ErrUnsupported("res_buffer_texture_opcode"));
    }
    let declared = ld32(&bytes[TEXTURE_VIEW_DESC_LEN..]) as usize;
    if declared < BUF_TEX_MIN_LEN || declared != bytes.len() {
        return Err(DecodeStatus::ErrShort("res_buffer_texture_declared_len"));
    }
    // Embedded MTLTextureDescriptor flags word: bits[0:4]=textureType, bits[16:32]=pixelFormat.
    let flags = ld32(&bytes[BUF_TEX_DESC_FLAGS..]);
    let texture_type = (flags & 0xf) as u16;
    let pixel_format = ((flags >> 16) & 0xffff) as u16;
    Ok(BufferTextureDescriptor {
        new_texture_ref: ld32(&bytes[TEXTURE_VIEW_DESC_TEXTURE_REF..]),
        buffer_ref: ld32(&bytes[BUF_TEX_DESC_BUFFER_REF..]),
        offset: ld64(&bytes[BUF_TEX_DESC_OFFSET..]),
        bytes_per_row: ld64(&bytes[BUF_TEX_DESC_BYTES_PER_ROW..]),
        texture_type,
        pixel_format,
        width: ld32(&bytes[BUF_TEX_DESC_WIDTH..]),
        height: ld32(&bytes[BUF_TEX_DESC_HEIGHT..]),
        depth: ld32(&bytes[BUF_TEX_DESC_DEPTH..]),
        mipmap_level_count: ld16(&bytes[BUF_TEX_DESC_MIP_COUNT..]),
        sample_count: ld16(&bytes[BUF_TEX_DESC_SAMPLE_COUNT..]),
        array_length: ld16(&bytes[BUF_TEX_DESC_ARRAY_LEN..]),
    })
}

/// Peek the view_opcode of a type-8 descriptor (opcode 9 = buffer-backed texture,
/// 7/8/0x1b = texture view). Returns `None` for a blob too short to hold a header.
pub fn texture_type8_opcode(bytes: &[u8]) -> Option<u32> {
    if bytes.len() < TEXTURE_VIEW_MIN_SIMPLE {
        return None;
    }
    Some(ld32(&bytes[TEXTURE_VIEW_DESC_OPCODE..]))
}

const TYPE7_MIN_LEN: usize = 17;

/// **Unused on the x86 PCI pathway.** A probe placed at the top of this function
/// — before the length check, so a short record would also report — emitted
/// nothing across a full interactive session. Type-11 geometry on that pathway
/// is latched from the **type-4** surface backing descriptor instead
/// (`runtime/objects.rs`, `decode_type4_surface` -> `set_mapping_geom`). Do not
/// reason about what the guest tells us at surface-create time from this
/// decoder without re-confirming it runs; measure `decode_type4_surface`.
/// Offsets in the type-11 IOSurface-texture descriptor. Named rather than
/// written as hex at the read sites, which is the convention every other
/// decoder in this file already follows — an offset that appears only as a
/// literal cannot be found by a reader checking whether the layout still holds.
pub const IOSURFACE_TEX_MAPPING_ID: usize = 0x00;
pub const IOSURFACE_TEX_OBJECT_REF: usize = 0x10;
pub const IOSURFACE_TEX_PIXEL_FORMAT: usize = 0x16;
pub const IOSURFACE_TEX_WIDTH: usize = 0x18;
pub const IOSURFACE_TEX_HEIGHT: usize = 0x1c;
/// Through `height`. Live blobs run longer (0x38 / 0x58); the tail past this is
/// not decoded, and the comment on the decoder says why.
pub const IOSURFACE_TEX_MIN_LEN: usize = 0x20;

pub fn decode_iosurface_texture_descriptor(bytes: &[u8]) -> Result<Descriptor, DecodeStatus> {
    // Matches reims-vgpu-iosurface-pages texture descriptor min layout (mappingID,
    // object self-ref, format, width, height). Live type-11 blobs are longer
    // (0x38/0x58); multi-mip level records are **not** part of this object type
    // — Metal forbids mipmapped IOSurface textures
    // (`newTextureWithDescriptor:iosurface:` rejects mipmapLevelCount > 1),
    // and product resolve fail-closes non-zero levels rather than inventing
    // a pyramid packing in the mapping (see blit_exec::Type11Texture).
    if bytes.len() < IOSURFACE_TEX_MIN_LEN {
        return Err(DecodeStatus::ErrShort("res_iosurface_short"));
    }
    Ok(Descriptor::IOSurfaceTexture {
        mapping_id: ld32(&bytes[IOSURFACE_TEX_MAPPING_ID..]),
        object_ref: ld32(&bytes[IOSURFACE_TEX_OBJECT_REF..]),
        pixel_format: ld16(&bytes[IOSURFACE_TEX_PIXEL_FORMAT..]),
        width: ld32(&bytes[IOSURFACE_TEX_WIDTH..]),
        height: ld32(&bytes[IOSURFACE_TEX_HEIGHT..]),
    })
}

fn section_range_fits(len: usize, start: u64, count: u32, entry_size: usize) -> bool {
    if count == 0 {
        return true;
    }
    let bytes = (count as u64).saturating_mul(entry_size as u64);
    start
        .checked_add(bytes)
        .map(|end| end <= len as u64)
        .unwrap_or(false)
}

fn section_ranges_overlap(a_start: u64, a_len: u64, b_start: u64, b_len: u64) -> bool {
    if a_len == 0 || b_len == 0 {
        return false;
    }
    let a_end = a_start.saturating_add(a_len);
    let b_end = b_start.saturating_add(b_len);
    a_start < b_end && b_start < a_end
}

/// Parse optional MetalSerializer compute stage-input block after first TLVs.
///
/// Returns `Ok(None)` when no valid block is present (short / length mismatch).
/// Returns `Err` only when the block header claims a valid payload but entry
/// ranges are out of bounds or overlapping (fail-closed structural error).
pub fn parse_compute_stage_input_block(
    bytes: &[u8],
    block_start: usize,
) -> Result<Option<ComputeStageInputDescriptor>, DecodeStatus> {
    if block_start >= bytes.len() {
        return Ok(None);
    }
    let bo = skip_optional_label_and_pad(bytes, bytes.len(), block_start);
    if bo >= bytes.len() {
        return Ok(None);
    }
    let block_len = bytes.len() - bo;
    if block_len < COMPUTE_STAGE_INPUT_MIN_LEN {
        return Ok(None);
    }
    let word0 = ld32(&bytes[bo + COMPUTE_STAGE_INPUT_WORD0..]);
    let header0 = ld32(&bytes[bo + COMPUTE_STAGE_INPUT_HEADER0..]);
    let header1 = ld32(&bytes[bo + COMPUTE_STAGE_INPUT_HEADER1..]);
    let declared_payload = header0 & COMPUTE_STAGE_INPUT_HEADER0_LEN_MASK;
    // header0 low16 is payload length after word0; total block = word0 + payload.
    if (declared_payload as u64).saturating_add(4) != block_len as u64 {
        return Ok(None);
    }

    let attr_count = (header0 >> COMPUTE_STAGE_INPUT_HEADER0_ATTR_COUNT_SHIFT)
        & COMPUTE_STAGE_INPUT_HEADER0_COUNT_MASK;
    let layout_count = (header0 >> COMPUTE_STAGE_INPUT_HEADER0_LAYOUT_COUNT_SHIFT)
        & COMPUTE_STAGE_INPUT_HEADER0_COUNT_MASK;
    let index_type = (header0 >> COMPUTE_STAGE_INPUT_HEADER0_INDEX_TYPE_SHIFT)
        & COMPUTE_STAGE_INPUT_HEADER0_INDEX_TYPE_MASK;
    let index_buffer_index = (header0 >> COMPUTE_STAGE_INPUT_HEADER0_INDEX_BUFFER_SHIFT)
        & COMPUTE_STAGE_INPUT_HEADER0_INDEX_BUFFER_MASK;
    let layout_rel = header1 & COMPUTE_STAGE_INPUT_HEADER1_LAYOUT_OFFSET_MASK;
    let attr_rel = header1 >> COMPUTE_STAGE_INPUT_HEADER1_ATTR_OFFSET_SHIFT;

    let offset_base = (bo + COMPUTE_STAGE_INPUT_HEADER1_OFFSET_BASE) as u64;
    let min_entries = (bo + COMPUTE_STAGE_INPUT_MIN_LEN) as u64;
    let layout_section = offset_base.saturating_add(layout_rel as u64);
    let attr_section = offset_base.saturating_add(attr_rel as u64);
    let layout_bytes = (layout_count as u64) * (COMPUTE_STAGE_INPUT_LAYOUT_ENTRY_SIZE as u64);
    let attr_bytes = (attr_count as u64) * (COMPUTE_STAGE_INPUT_ATTR_ENTRY_SIZE as u64);
    if (layout_count != 0 && layout_section < min_entries)
        || (attr_count != 0 && attr_section < min_entries)
        || !section_range_fits(
            bytes.len(),
            layout_section,
            layout_count,
            COMPUTE_STAGE_INPUT_LAYOUT_ENTRY_SIZE,
        )
        || !section_range_fits(
            bytes.len(),
            attr_section,
            attr_count,
            COMPUTE_STAGE_INPUT_ATTR_ENTRY_SIZE,
        )
        || section_ranges_overlap(layout_section, layout_bytes, attr_section, attr_bytes)
    {
        return Err(DecodeStatus::ErrShort("res_stage_input_section_oob"));
    }

    let mut out = ComputeStageInputDescriptor {
        word0,
        header0,
        header1,
        index_type,
        index_buffer_index,
        attributes: Vec::new(),
        layouts: Vec::new(),
        dropped_attributes: 0,
        dropped_layouts: 0,
    };

    for i in 0..layout_count {
        let entry = layout_section + (i as u64) * (COMPUTE_STAGE_INPUT_LAYOUT_ENTRY_SIZE as u64);
        let entry = entry as usize;
        let raw_bits = ld32(&bytes[entry..]);
        if out.layouts.len() < MAX_COMPUTE_STAGE_INPUT_LAYOUTS {
            out.layouts.push(ComputeStageInputLayout {
                raw_bits,
                buffer_index: raw_bits & COMPUTE_STAGE_INPUT_LAYOUT_BITS_BUFFER_MASK,
                step_function: (raw_bits >> COMPUTE_STAGE_INPUT_LAYOUT_BITS_STEP_SHIFT)
                    & COMPUTE_STAGE_INPUT_LAYOUT_BITS_STEP_MASK,
                step_rate: ld32(&bytes[entry + COMPUTE_STAGE_INPUT_LAYOUT_STEP_RATE..]),
                stride: ld64(&bytes[entry + COMPUTE_STAGE_INPUT_LAYOUT_STRIDE..]),
            });
        } else {
            out.dropped_layouts += 1;
        }
    }
    for i in 0..attr_count {
        let entry = attr_section + (i as u64) * (COMPUTE_STAGE_INPUT_ATTR_ENTRY_SIZE as u64);
        let entry = entry as usize;
        let raw_bits = ld32(&bytes[entry..]);
        if out.attributes.len() < MAX_COMPUTE_STAGE_INPUT_ATTRS {
            out.attributes.push(ComputeStageInputAttribute {
                raw_bits,
                location: raw_bits & COMPUTE_STAGE_INPUT_ATTR_BITS_LOCATION_MASK,
                buffer_index: (raw_bits >> COMPUTE_STAGE_INPUT_ATTR_BITS_BUFFER_SHIFT)
                    & COMPUTE_STAGE_INPUT_ATTR_BITS_BUFFER_MASK,
                format: (raw_bits >> COMPUTE_STAGE_INPUT_ATTR_BITS_FORMAT_SHIFT)
                    & COMPUTE_STAGE_INPUT_ATTR_BITS_FORMAT_MASK,
                offset: ld32(&bytes[entry + COMPUTE_STAGE_INPUT_ATTR_OFFSET..]),
            });
        } else {
            out.dropped_attributes += 1;
        }
    }
    Ok(Some(out))
}

/// Decode type-7 compute pipeline (`objType=0x0b`): kernel TLV + optional stage-input.
pub fn decode_compute_pipeline_descriptor(
    bytes: &[u8],
) -> Result<ComputePipelineDescriptor, DecodeStatus> {
    if bytes.len() < TYPE7_MIN_LEN {
        return Err(DecodeStatus::ErrShort("res_compute_pipeline_short"));
    }
    if ld32(&bytes[0..]) != TYPE7_OBJECT_COMPUTE_PIPELINE {
        return Err(DecodeStatus::ErrUnsupported("res_compute_pipeline_tag"));
    }
    let declared = ld32(&bytes[4..]) as usize;
    if declared != bytes.len() || declared < TYPE7_MIN_LEN {
        return Err(DecodeStatus::ErrShort("res_compute_pipeline_declared_len"));
    }
    let (fields, consumed) = decode_compact_tlv_record(bytes, TYPE7_FIRST_TLVS)?;
    let first_tlv_end = TYPE7_FIRST_TLVS + consumed;
    let stage_input = parse_compute_stage_input_block(bytes, first_tlv_end)?;
    Ok(ComputePipelineDescriptor {
        kernel_func_ref: compact_tlv_u32(&fields, PIPELINE_TAG_KERNEL_FUNC).unwrap_or(0),
        stage_input,
    })
}

/// Decode the 52-byte ICB command layout (create body `+0x1c` or live object).
pub fn decode_icb_command_layout(bytes: &[u8]) -> Result<IcbCommandLayout, DecodeStatus> {
    if bytes.len() < ICB_LAYOUT_LEN {
        return Err(DecodeStatus::ErrShort("res_icb_layout_short"));
    }
    Ok(IcbCommandLayout {
        command_type_offset: ld16(&bytes[0..]),
        barrier_offset: ld16(&bytes[2..]),
        kernel_dispatch_arguments_offset: ld16(&bytes[4..]),
        tessellation_factor_offset: ld16(&bytes[6..]),
        pipeline_state_offset: ld32(&bytes[8..]),
        vertex_buffer_bind_offset: ld32(&bytes[0xc..]),
        fragment_buffer_bind_offset: ld32(&bytes[0x10..]),
        object_buffer_bind_offset: ld32(&bytes[0x14..]),
        mesh_buffer_bind_offset: ld32(&bytes[0x18..]),
        kernel_buffer_bind_offset: ld32(&bytes[0x1c..]),
        attribute_stride_offset: ld32(&bytes[0x20..]),
        object_threadgroup_memory_length_offset: ld32(&bytes[0x24..]),
        threadgroup_memory_length_offset: ld32(&bytes[0x28..]),
        command_arguments_offset: ld32(&bytes[0x2c..]),
        command_size: ld32(&bytes[0x30..]),
    })
}

/// Encode layout into 52 bytes (tests / fixtures).
#[cfg(test)]
pub fn encode_icb_command_layout(layout: &IcbCommandLayout) -> [u8; ICB_LAYOUT_LEN] {
    let mut b = [0u8; ICB_LAYOUT_LEN];
    st16(&mut b[0..], layout.command_type_offset);
    st16(&mut b[2..], layout.barrier_offset);
    st16(&mut b[4..], layout.kernel_dispatch_arguments_offset);
    st16(&mut b[6..], layout.tessellation_factor_offset);
    st32(&mut b[8..], layout.pipeline_state_offset);
    st32(&mut b[0xc..], layout.vertex_buffer_bind_offset);
    st32(&mut b[0x10..], layout.fragment_buffer_bind_offset);
    st32(&mut b[0x14..], layout.object_buffer_bind_offset);
    st32(&mut b[0x18..], layout.mesh_buffer_bind_offset);
    st32(&mut b[0x1c..], layout.kernel_buffer_bind_offset);
    st32(&mut b[0x20..], layout.attribute_stride_offset);
    st32(
        &mut b[0x24..],
        layout.object_threadgroup_memory_length_offset,
    );
    st32(&mut b[0x28..], layout.threadgroup_memory_length_offset);
    st32(&mut b[0x2c..], layout.command_arguments_offset);
    st32(&mut b[0x30..], layout.command_size);
    b
}

/// Render-only layout for Draw / DrawIndexed / patches / mesh, no inherit.
///
/// `setupCommandLayout:` (pipeline `0x60`): bind tables in order
/// vertex → fragment → **object → mesh** → kernel, each `count × 0x14`, then
/// attribute-stride table (`maxVertex × 8` when dynamic stride), object-TG
/// lengths, kernel-TG lengths, then args.
#[cfg(test)]
pub fn render_icb_layout(
    max_vertex: u16,
    max_fragment: u16,
    command_types: u32,
) -> IcbCommandLayout {
    render_icb_layout_ex(max_vertex, max_fragment, 0, 0, 0, command_types)
}

/// Like [`render_icb_layout`] with object/mesh bind tables and object-TG lengths.
#[cfg(test)]
pub fn render_icb_layout_ex(
    max_vertex: u16,
    max_fragment: u16,
    max_object: u16,
    max_mesh: u16,
    max_object_tg: u16,
    command_types: u32,
) -> IcbCommandLayout {
    let pipeline = 0x60u32;
    let vertex_bind = 0x64u32;
    let after_vertex = vertex_bind + (max_vertex as u32) * (ICB_BUFFER_BIND_STRIDE as u32);
    let fragment_bind = after_vertex;
    let after_fragment = fragment_bind + (max_fragment as u32) * (ICB_BUFFER_BIND_STRIDE as u32);
    let object_bind = after_fragment;
    let after_object = object_bind + (max_object as u32) * (ICB_BUFFER_BIND_STRIDE as u32);
    let mesh_bind = after_object;
    let after_mesh = mesh_bind + (max_mesh as u32) * (ICB_BUFFER_BIND_STRIDE as u32);
    // No kernel binds on pure render ICBs (maxKernel=0).
    let free_after_binds = after_mesh;
    // RE setupCommandLayout: attribute-stride table is maxVertex × 8 after binds
    // when supportDynamicAttributeStride (product always reserves for max_vertex).
    let stride_off = free_after_binds;
    let after_stride = stride_off + (max_vertex as u32) * (ICB_ATTRIBUTE_STRIDE_ENTRY_SIZE as u32);
    // Object TG length table then kernel TG (kernel 0 for pure render).
    let object_tg_off = after_stride;
    let after_object_tg = object_tg_off + (max_object_tg as u32) * (ICB_TG_MEMORY_STRIDE as u32);
    let kernel_tg_off = after_object_tg;
    let after_tg = after_object_tg;
    // setupCommandLayout (host RE): max of per-type argument region sizes.
    // Draw=0x24, DrawIndexed/DrawPatches=0x38, DrawIndexedPatches fill=0x4a
    // (layout alloc may use 0x4c), Mesh=0x48, ConcurrentDispatch=0x30.
    let mut args_size = 0u32;
    if command_types & MTL_INDIRECT_CMD_DRAW != 0 {
        args_size = 0x24;
    }
    if command_types & MTL_INDIRECT_CMD_DRAW_INDEXED != 0 {
        args_size = args_size.max(0x38);
    }
    if command_types & MTL_INDIRECT_CMD_DRAW_PATCHES != 0 {
        args_size = args_size.max(ICB_DRAW_PATCHES_ARGS_LEN);
    }
    if command_types & MTL_INDIRECT_CMD_DRAW_INDEXED_PATCHES != 0 {
        args_size = args_size.max(ICB_DRAW_INDEXED_PATCHES_ARGS_LEN);
    }
    if command_types
        & (MTL_INDIRECT_CMD_DRAW_MESH_THREADGROUPS | MTL_INDIRECT_CMD_DRAW_MESH_THREADS)
        != 0
    {
        args_size = args_size.max(ICB_DRAW_MESH_ARGS_LEN);
    }
    if args_size == 0 {
        // Default Draw if no bits (tests); keep deterministic.
        args_size = 0x24;
    }
    IcbCommandLayout {
        command_type_offset: 0,
        barrier_offset: 4,
        kernel_dispatch_arguments_offset: 8,
        tessellation_factor_offset: 0x40,
        pipeline_state_offset: pipeline,
        vertex_buffer_bind_offset: vertex_bind,
        fragment_buffer_bind_offset: fragment_bind,
        object_buffer_bind_offset: object_bind,
        mesh_buffer_bind_offset: mesh_bind,
        kernel_buffer_bind_offset: free_after_binds,
        attribute_stride_offset: stride_off,
        object_threadgroup_memory_length_offset: object_tg_off,
        threadgroup_memory_length_offset: kernel_tg_off,
        command_arguments_offset: after_tg,
        command_size: after_tg + args_size,
    }
}

/// Draw-only convenience (commandTypes Draw).
#[cfg(test)]
pub fn render_only_icb_layout(max_vertex: u16) -> IcbCommandLayout {
    render_icb_layout(max_vertex, 0, MTL_INDIRECT_CMD_DRAW)
}

/// DrawIndexed-only convenience.
#[cfg(test)]
pub fn render_draw_indexed_icb_layout(max_vertex: u16) -> IcbCommandLayout {
    render_icb_layout(max_vertex, 0, MTL_INDIRECT_CMD_DRAW_INDEXED)
}

/// DrawPatches-only convenience (args 0x38 + tessellation factor table at 0x40).
#[cfg(test)]
pub fn render_draw_patches_icb_layout(max_vertex: u16) -> IcbCommandLayout {
    render_icb_layout(max_vertex, 0, MTL_INDIRECT_CMD_DRAW_PATCHES)
}

/// DrawIndexedPatches-only convenience (args 0x4a).
#[cfg(test)]
pub fn render_draw_indexed_patches_icb_layout(max_vertex: u16) -> IcbCommandLayout {
    render_icb_layout(max_vertex, 0, MTL_INDIRECT_CMD_DRAW_INDEXED_PATCHES)
}

/// DrawMeshThreads-only convenience (args 0x48, optional mesh bind slots).
#[cfg(test)]
pub fn render_draw_mesh_threads_icb_layout(max_mesh: u16) -> IcbCommandLayout {
    render_icb_layout_ex(0, 0, 0, max_mesh, 0, MTL_INDIRECT_CMD_DRAW_MESH_THREADS)
}

/// DrawMeshThreadgroups-only convenience (args 0x48).
#[cfg(test)]
pub fn render_draw_mesh_threadgroups_icb_layout() -> IcbCommandLayout {
    render_icb_layout(0, 0, MTL_INDIRECT_CMD_DRAW_MESH_THREADGROUPS)
}

/// Mesh draw with object + mesh bind tables.
#[cfg(test)]
pub fn render_draw_mesh_threads_icb_layout_with_binds(
    max_object: u16,
    max_mesh: u16,
) -> IcbCommandLayout {
    render_icb_layout_ex(
        0,
        0,
        max_object,
        max_mesh,
        0,
        MTL_INDIRECT_CMD_DRAW_MESH_THREADS,
    )
}

/// Object+mesh drawMeshThreadgroups with optional object TG memory slots.
#[cfg(test)]
pub fn render_draw_mesh_threadgroups_icb_layout_ex(
    max_object: u16,
    max_mesh: u16,
    max_object_tg: u16,
) -> IcbCommandLayout {
    render_icb_layout_ex(
        0,
        0,
        max_object,
        max_mesh,
        max_object_tg,
        MTL_INDIRECT_CMD_DRAW_MESH_THREADGROUPS,
    )
}

/// Compute-only layout for ConcurrentDispatch with `max_kernel` binds, no inherit.
///
/// Matches `AppleParavirtIndirectCommandBuffer setupCommandLayout:` for the
/// common product case (commandTypes=`1<<5`, inheritBuffers=false,
/// inheritPipelineState=false). Threadgroup-memory table size 0 (no TG binds).
#[cfg(test)]
pub fn compute_only_icb_layout(max_kernel: u16) -> IcbCommandLayout {
    compute_icb_layout(max_kernel, 0)
}

/// Compute ICB layout with optional TG-memory table and attribute-stride table.
///
/// `setupCommandLayout:` order after kernel binds:
/// 1. `max_kernel × 8` attribute-stride u64s at `attributeStrideOffset`
/// 2. `max_kernel_tg × 8` TG-memory length u64s at `threadgroupMemoryLengthOffset`
/// 3. dispatch args. Barrier is u32 at `barrierOffset` (typically 4).
#[cfg(test)]
pub fn compute_icb_layout(max_kernel: u16, max_kernel_tg: u16) -> IcbCommandLayout {
    let pipeline = 0x60u32;
    let kernel_bind = 0x64u32;
    let free_after_binds = kernel_bind + (max_kernel as u32) * (ICB_BUFFER_BIND_STRIDE as u32);
    let stride_off = free_after_binds;
    let after_stride = stride_off + (max_kernel as u32) * (ICB_ATTRIBUTE_STRIDE_ENTRY_SIZE as u32);
    let tg_off = after_stride;
    let after_tg = tg_off + (max_kernel_tg as u32) * (ICB_TG_MEMORY_STRIDE as u32);
    // ConcurrentDispatch-only args size = 0x30 (3×u64 grid + 3×u64 tptg).
    let args_size = 0x30u32;
    IcbCommandLayout {
        command_type_offset: 0,
        barrier_offset: 4,
        kernel_dispatch_arguments_offset: 8,
        tessellation_factor_offset: 0x40,
        pipeline_state_offset: pipeline,
        vertex_buffer_bind_offset: kernel_bind,
        fragment_buffer_bind_offset: kernel_bind,
        object_buffer_bind_offset: kernel_bind,
        mesh_buffer_bind_offset: kernel_bind,
        kernel_buffer_bind_offset: kernel_bind,
        attribute_stride_offset: stride_off,
        object_threadgroup_memory_length_offset: after_tg,
        threadgroup_memory_length_offset: tg_off,
        command_arguments_offset: after_tg,
        command_size: after_tg + args_size,
    }
}

/// Number of kernel-threadgroup-memory length slots implied by layout offsets.
pub fn icb_layout_kernel_tg_slot_count(layout: &IcbCommandLayout) -> u16 {
    let start = layout.threadgroup_memory_length_offset;
    let end = layout.command_arguments_offset;
    if end <= start {
        return 0;
    }
    ((end - start) / ICB_TG_MEMORY_STRIDE as u32) as u16
}

/// Number of attribute-stride table entries implied by layout offsets.
///
/// Table is `[attribute_stride_offset, next_region)` in u64 slots, where
/// `next_region` is the earliest of object-TG / kernel-TG / command-args that
/// lies strictly after the stride table start.
pub fn icb_layout_attribute_stride_slot_count(layout: &IcbCommandLayout) -> u16 {
    let start = layout.attribute_stride_offset;
    if start == 0 {
        return 0;
    }
    let end = [
        layout.object_threadgroup_memory_length_offset,
        layout.threadgroup_memory_length_offset,
        layout.command_arguments_offset,
    ]
    .into_iter()
    .filter(|&e| e > start)
    .min()
    .unwrap_or(start);
    if end <= start {
        return 0;
    }
    ((end - start) / ICB_ATTRIBUTE_STRIDE_ENTRY_SIZE as u32) as u16
}

/// Decode type-7 ICB create descriptor (tag 0x36, length 0x58).
///
/// Field map from PGSerializer
/// `newIndirectCommandBufferWithDescriptor:layout:maxCommandCount:options:allocator:`
/// local emission + layout memcpy at `+0x1c` (2026-07-11 arm64 host RE).
pub fn decode_icb_descriptor(
    bytes: &[u8],
) -> Result<IndirectCommandBufferDescriptor, DecodeStatus> {
    if bytes.len() < ICB_DESC_LEN {
        return Err(DecodeStatus::ErrShort("res_icb_desc_short"));
    }
    let tag = ld32(&bytes[ICB_DESC_TAG..]);
    let declared = ld32(&bytes[ICB_DESC_DECLARED_LEN..]);
    if tag != TYPE7_OBJECT_ICB
        || declared as usize != ICB_DESC_LEN
        || declared as usize != bytes.len()
    {
        return Err(DecodeStatus::ErrUnsupported("res_icb_desc_tag"));
    }
    let command_types = ld32(&bytes[ICB_DESC_COMMAND_TYPES..]);
    let max_vertex = bytes[ICB_DESC_MAX_VERTEX_BINDS] as u16;
    let max_fragment = bytes[ICB_DESC_MAX_FRAGMENT_BINDS] as u16;
    let max_kernel_byte = bytes[ICB_DESC_MAX_KERNEL_BINDS] as u16;
    let max_object = bytes[ICB_DESC_MAX_OBJECT_BINDS] as u16;
    let max_mesh = bytes[ICB_DESC_MAX_MESH_BINDS] as u16;
    let max_kernel_tg = bytes[ICB_DESC_MAX_KERNEL_TG_BINDS] as u16;
    let max_object_tg = bytes[ICB_DESC_MAX_OBJECT_TG_BINDS] as u16;
    let flags = ld16(&bytes[ICB_DESC_FLAGS..]);
    // `maxFragmentBufferBindCount` is reported as the guest wrote it at +0x0d.
    // It used to be forced to 0 whenever `command_types` named a dispatch and no
    // draw, on the reading that the guest "meant" a compute-only buffer. That is
    // an inference about intent overriding a field the descriptor states
    // outright, and it was silent — a decoded value was discarded with nothing
    // recorded. The command layout at +0x1c independently carries
    // `fragment_buffer_bind_offset`, so nothing downstream needs the count to be
    // re-derived from the command mask.
    // Create body packs maxKernel at +0x0e (third bind-count byte).
    let max_kernel = max_kernel_byte;
    let layout =
        decode_icb_command_layout(&bytes[ICB_DESC_LAYOUT..ICB_DESC_LAYOUT + ICB_LAYOUT_LEN])?;
    Ok(IndirectCommandBufferDescriptor {
        command_types,
        max_vertex_buffer_bind_count: max_vertex,
        max_fragment_buffer_bind_count: max_fragment,
        max_kernel_buffer_bind_count: max_kernel,
        max_object_buffer_bind_count: max_object,
        max_mesh_buffer_bind_count: max_mesh,
        max_kernel_threadgroup_memory_bind_count: max_kernel_tg,
        max_object_threadgroup_memory_bind_count: max_object_tg,
        inherit_buffers: (flags & ICB_FLAG_INHERIT_BUFFERS) != 0,
        inherit_pipeline_state: (flags & ICB_FLAG_INHERIT_PIPELINE_STATE) != 0,
        max_command_count: ld32(&bytes[ICB_DESC_MAX_COMMAND_COUNT..]),
        options: ld32(&bytes[ICB_DESC_OPTIONS..]),
        layout,
    })
}

/// Decode type-7 container (sampler / depth-stencil / pipelines / ICB).
pub fn decode_type7_descriptor(bytes: &[u8]) -> Result<Descriptor, DecodeStatus> {
    if bytes.len() < 4 {
        return Err(DecodeStatus::ErrShort("res_type7_short"));
    }
    let first = ld32(&bytes[0..]);
    match first {
        TYPE7_OBJECT_SAMPLER => Ok(Descriptor::Sampler(decode_sampler_descriptor(bytes)?)),
        TYPE7_OBJECT_DEPTH_STENCIL => Ok(Descriptor::DepthStencil(
            decode_depth_stencil_descriptor(bytes)?,
        )),
        TYPE7_OBJECT_RENDER_PIPELINE => Ok(Descriptor::RenderPipeline(
            decode_render_pipeline_descriptor(bytes)?,
        )),
        TYPE7_OBJECT_COMPUTE_PIPELINE => Ok(Descriptor::ComputePipeline(
            decode_compute_pipeline_descriptor(bytes)?,
        )),
        TYPE7_OBJECT_ICB => Ok(Descriptor::IndirectCommandBuffer(decode_icb_descriptor(
            bytes,
        )?)),
        _ => Err(DecodeStatus::ErrUnsupported("res_type7_subtype_unknown")),
    }
}

pub fn decode_descriptor(object_type: u8, bytes: &[u8]) -> Result<Descriptor, DecodeStatus> {
    match object_type {
        OBJECT_TYPE_BUFFER => Ok(Descriptor::Buffer(decode_buffer_descriptor(bytes)?)),
        OBJECT_TYPE_TEXTURE | OBJECT_TYPE_TEXTURE_VARIANT => {
            Ok(Descriptor::Texture(decode_texture_descriptor(bytes)?))
        }
        OBJECT_TYPE_FUNCTION => Ok(Descriptor::Function(decode_function_descriptor(bytes)?)),
        OBJECT_TYPE_TYPE7 => decode_type7_descriptor(bytes),
        OBJECT_TYPE_TEXTURE_VIEW => Ok(Descriptor::TextureView(decode_texture_view_descriptor(
            bytes,
        )?)),
        OBJECT_TYPE_IOSURFACE => decode_iosurface_texture_descriptor(bytes),
        _ => Err(DecodeStatus::ErrUnknownType("res_object_type_unknown")),
    }
}

#[cfg(test)]
mod tests {
    use crate::model::PAGE_SHIFT_ARM64E;

    use super::*;
    use crate::contract::endian::st32;

    /// Short reads on different descriptors name different checks.
    ///
    /// This is the collapse the payload closes: **29 of the decoder's 40 sites
    /// were `ErrShort`**, one name for twenty-nine different reads spanning a
    /// 12-byte object-list entry, a type-7 TLV walk and a vertex-attribute
    /// table offset. Asserting the class alone would pass on any of them.
    #[test]
    fn every_short_read_names_the_field_it_ran_out_on() {
        use crate::observe::Decline;
        let cases: &[(&str, &'static str)] = &[
            ("list entry", "res_list_entry_short"),
            ("buffer", "res_buffer_desc_short"),
            ("sampler", "res_sampler_short"),
            ("icb layout", "res_icb_layout_short"),
        ];
        let got = [
            decode_list_object_entry(&[0u8; 1]).unwrap_err(),
            decode_buffer_descriptor(&[0u8; 1]).unwrap_err(),
            decode_sampler_descriptor(&[0u8; 1]).unwrap_err(),
            decode_icb_command_layout(&[0u8; 1]).unwrap_err(),
        ];
        for ((what, want), e) in cases.iter().zip(got) {
            assert_eq!(e.slug(), *want, "{what} short read lost its name");
        }

        // The other three classes, so the whole vocabulary is exercised rather
        // than just the one that used to swallow everything.
        assert_eq!(
            decode_descriptor(0xfe, &[0u8; 64]).unwrap_err().slug(),
            "res_object_type_unknown"
        );
        let mut type7 = [0u8; 64];
        st32(&mut type7[0..], 0xdead_beef);
        assert_eq!(
            decode_type7_descriptor(&type7).unwrap_err().slug(),
            "res_type7_subtype_unknown"
        );
    }

    #[test]
    fn icb_descriptor_from_serializer_fixture() {
        // PGSerializer emission: ConcurrentDispatch, maxKernel=4, maxCmd=8, options=0.
        let mut b = [0u8; ICB_DESC_LEN];
        st32(&mut b[0..], TYPE7_OBJECT_ICB);
        st32(&mut b[4..], ICB_DESC_LEN as u32);
        st32(&mut b[8..], MTL_INDIRECT_CMD_CONCURRENT_DISPATCH);
        // bind counts as u8: vertex, fragment, kernel, …
        b[ICB_DESC_MAX_VERTEX_BINDS] = 0;
        b[ICB_DESC_MAX_FRAGMENT_BINDS] = 0;
        b[ICB_DESC_MAX_KERNEL_BINDS] = 4;
        st16(&mut b[ICB_DESC_FLAGS..], 0); // no inherit
        let layout = compute_only_icb_layout(4);
        b[ICB_DESC_LAYOUT..ICB_DESC_LAYOUT + ICB_LAYOUT_LEN]
            .copy_from_slice(&encode_icb_command_layout(&layout));
        st32(&mut b[ICB_DESC_MAX_COMMAND_COUNT..], 8);
        st32(&mut b[ICB_DESC_OPTIONS..], 0);
        let icb = decode_icb_descriptor(&b).unwrap();
        assert_eq!(icb.command_types, MTL_INDIRECT_CMD_CONCURRENT_DISPATCH);
        assert_eq!(icb.max_kernel_buffer_bind_count, 4);
        assert_eq!(icb.max_command_count, 8);
        assert!(!icb.inherit_buffers);
        assert!(!icb.inherit_pipeline_state);
    }

    /// A dispatch-only `command_types` does not license discarding the
    /// fragment bind count the descriptor states.
    ///
    /// The decoder used to zero `max_fragment_buffer_bind_count` whenever the
    /// command mask named a dispatch and no draw. That is an inference about
    /// what the guest meant, overriding a byte the guest wrote at +0x0d, and it
    /// was silent — so a descriptor built by a guest that reserves fragment
    /// binds on a buffer it happens to fill with dispatches had the reservation
    /// dropped with nothing recorded. Metal is handed this count directly
    /// (`icb::materialize`), so the drop is guest-visible.
    #[test]
    fn a_dispatch_only_command_mask_keeps_the_stated_fragment_bind_count() {
        let mut b = [0u8; ICB_DESC_LEN];
        st32(&mut b[0..], TYPE7_OBJECT_ICB);
        st32(&mut b[4..], ICB_DESC_LEN as u32);
        // Dispatch bits only — no draw bit anywhere in the mask.
        st32(
            &mut b[8..],
            MTL_INDIRECT_CMD_CONCURRENT_DISPATCH | MTL_INDIRECT_CMD_CONCURRENT_DISPATCH_THREADS,
        );
        b[ICB_DESC_MAX_VERTEX_BINDS] = 0;
        b[ICB_DESC_MAX_FRAGMENT_BINDS] = 6;
        b[ICB_DESC_MAX_KERNEL_BINDS] = 4;
        st16(&mut b[ICB_DESC_FLAGS..], 0);
        let layout = compute_only_icb_layout(4);
        b[ICB_DESC_LAYOUT..ICB_DESC_LAYOUT + ICB_LAYOUT_LEN]
            .copy_from_slice(&encode_icb_command_layout(&layout));
        st32(&mut b[ICB_DESC_MAX_COMMAND_COUNT..], 8);
        st32(&mut b[ICB_DESC_OPTIONS..], 0);

        let icb = decode_icb_descriptor(&b).unwrap();
        assert_eq!(
            icb.max_fragment_buffer_bind_count, 6,
            "the wire byte at +0x0d is the answer, not the command mask"
        );
        assert_eq!(icb.max_kernel_buffer_bind_count, 4);
        assert_eq!(icb.layout.command_size, layout.command_size);
        assert_eq!(icb.layout.kernel_buffer_bind_offset, 0x64);
        match decode_type7_descriptor(&b).unwrap() {
            Descriptor::IndirectCommandBuffer(d) => assert_eq!(d.max_command_count, 8),
            _ => panic!("expected ICB"),
        }
        // inherit both (bit0 = pipeline, bit1 = buffers)
        st16(
            &mut b[ICB_DESC_FLAGS..],
            ICB_FLAG_INHERIT_BUFFERS | ICB_FLAG_INHERIT_PIPELINE_STATE,
        );
        let icb = decode_icb_descriptor(&b).unwrap();
        assert!(icb.inherit_buffers && icb.inherit_pipeline_state);
    }

    /// Dedicated create-body max-count matrix: decode offsets +0x0f..+0x12 and
    /// layout table sizing for object/mesh/objectTG/kernelTG.
    #[test]
    fn icb_create_body_max_count_matrix() {
        use crate::contract::endian::st16;

        // --- Decode: single-byte fields at RE offsets ---
        let mut b = [0u8; ICB_DESC_LEN];
        st32(&mut b[0..], TYPE7_OBJECT_ICB);
        st32(&mut b[4..], ICB_DESC_LEN as u32);
        st32(
            &mut b[8..],
            MTL_INDIRECT_CMD_DRAW_MESH_THREADGROUPS | MTL_INDIRECT_CMD_DRAW,
        );
        b[ICB_DESC_MAX_VERTEX_BINDS] = 2;
        b[ICB_DESC_MAX_FRAGMENT_BINDS] = 3;
        b[ICB_DESC_MAX_KERNEL_BINDS] = 0;
        b[ICB_DESC_MAX_OBJECT_BINDS] = 4; // +0x0f
        b[ICB_DESC_MAX_MESH_BINDS] = 5; // +0x10
        b[ICB_DESC_MAX_KERNEL_TG_BINDS] = 6; // +0x11
        b[ICB_DESC_MAX_OBJECT_TG_BINDS] = 7; // +0x12
        st16(&mut b[ICB_DESC_FLAGS..], 0);
        let layout = render_icb_layout_ex(
            2,
            3,
            4,
            5,
            7,
            MTL_INDIRECT_CMD_DRAW_MESH_THREADGROUPS | MTL_INDIRECT_CMD_DRAW,
        );
        b[ICB_DESC_LAYOUT..ICB_DESC_LAYOUT + ICB_LAYOUT_LEN]
            .copy_from_slice(&encode_icb_command_layout(&layout));
        st32(&mut b[ICB_DESC_MAX_COMMAND_COUNT..], 1);
        st32(&mut b[ICB_DESC_OPTIONS..], 0);

        let icb = decode_icb_descriptor(&b).unwrap();
        assert_eq!(icb.max_vertex_buffer_bind_count, 2);
        assert_eq!(icb.max_fragment_buffer_bind_count, 3);
        assert_eq!(icb.max_kernel_buffer_bind_count, 0);
        assert_eq!(icb.max_object_buffer_bind_count, 4);
        assert_eq!(icb.max_mesh_buffer_bind_count, 5);
        assert_eq!(icb.max_kernel_threadgroup_memory_bind_count, 6);
        assert_eq!(icb.max_object_threadgroup_memory_bind_count, 7);

        // --- Layout sizing: bind table order vertex → fragment → object → mesh ---
        // vertex @0x64, 2 × 0x14 = 0x28 → fragment @0x8c
        assert_eq!(layout.vertex_buffer_bind_offset, 0x64);
        assert_eq!(layout.fragment_buffer_bind_offset, 0x64 + 2 * 0x14);
        assert_eq!(
            layout.object_buffer_bind_offset,
            layout.fragment_buffer_bind_offset + 3 * 0x14
        );
        assert_eq!(
            layout.mesh_buffer_bind_offset,
            layout.object_buffer_bind_offset + 4 * 0x14
        );
        // After mesh: attribute stride max_vertex × 8
        let after_mesh = layout.mesh_buffer_bind_offset + 5 * 0x14;
        assert_eq!(layout.attribute_stride_offset, after_mesh);
        let after_stride = after_mesh + 2 * ICB_ATTRIBUTE_STRIDE_ENTRY_SIZE as u32;
        // Object TG table: max_object_tg × 8
        assert_eq!(layout.object_threadgroup_memory_length_offset, after_stride);
        assert_eq!(
            layout.threadgroup_memory_length_offset,
            after_stride + 7 * ICB_TG_MEMORY_STRIDE as u32
        );
        // Pure render: kernel TG slots empty (object TG ends at args)
        assert_eq!(
            layout.command_arguments_offset,
            layout.object_threadgroup_memory_length_offset + 7 * ICB_TG_MEMORY_STRIDE as u32
        );
        assert_eq!(
            layout.command_size,
            layout.command_arguments_offset + ICB_DRAW_MESH_ARGS_LEN
        );

        // --- Compute: kernelTG table size from max_kernel_tg ---
        let cl = compute_icb_layout(3, 2);
        assert_eq!(icb_layout_kernel_tg_slot_count(&cl), 2);
        assert_eq!(
            cl.threadgroup_memory_length_offset + 2 * ICB_TG_MEMORY_STRIDE as u32,
            cl.command_arguments_offset
        );

        // --- Zero counts: tables collapse (no object/mesh/objectTG) ---
        let zero = render_icb_layout_ex(1, 0, 0, 0, 0, MTL_INDIRECT_CMD_DRAW);
        assert_eq!(
            zero.object_buffer_bind_offset,
            zero.fragment_buffer_bind_offset
        );
        assert_eq!(zero.mesh_buffer_bind_offset, zero.object_buffer_bind_offset);
        assert_eq!(
            zero.object_threadgroup_memory_length_offset,
            zero.attribute_stride_offset + ICB_ATTRIBUTE_STRIDE_ENTRY_SIZE as u32
        );
        assert_eq!(
            zero.command_arguments_offset,
            zero.object_threadgroup_memory_length_offset
        );
    }

    #[test]
    fn compute_pipeline_stage_input_fixture() {
        // Local MetalSerializer fixture: dynamic Float4 stage-input layout.
        // From reims_vgpu_resource_resolve_test make_compute_stage_input_pipeline.
        let fixture: [u8; 60] = [
            0x0b, 0x00, 0x00, 0x00, 0x3c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x2c, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
            0x20, 0x00, 0x40, 0x08, 0x08, 0x00, 0x18, 0x00, 0xa0, 0x00, 0x00, 0x00, 0x01, 0x00,
            0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x7c, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00,
        ];
        let cp = decode_compute_pipeline_descriptor(&fixture).unwrap();
        // First TLV is empty (field_count=0 at +16); kernel not present in this fixture.
        assert_eq!(cp.kernel_func_ref, 0);
        let si = cp.stage_input.expect("stage-input block");
        assert_eq!(si.header0, 0x0840_0020);
        assert_eq!(si.header1, 0x0018_0008);
        assert_eq!(si.index_type, 0);
        assert_eq!(si.index_buffer_index, 0);
        assert_eq!(si.layouts.len(), 1);
        assert_eq!(si.layouts[0].raw_bits, 0xa0);
        assert_eq!(si.layouts[0].step_function, 5);
        assert_eq!(si.layouts[0].stride, u64::MAX);
        assert_eq!(si.attributes.len(), 1);
        assert_eq!(si.attributes[0].raw_bits, 0x7c00);
        // format bits 10..15 of 0x7c00 = 0x1f = 31 (Float4).
        assert_eq!(si.attributes[0].format, 31);
        assert_eq!(si.dropped_attributes, 0);
        assert_eq!(si.dropped_layouts, 0);
    }

    #[test]
    fn list_entry_and_buffer() {
        // Live list offset: ref * 12
        assert_eq!(list_object_entry_offset(3, 10), Some(36));

        let mut list = [0u8; 12];
        st32(&mut list[0..], 11u32 | (0x20u32 << 8));
        // desc_gva
        list[4] = 0x80;
        let le = decode_list_object_entry(&list).unwrap();
        assert_eq!(le.object_type, 11);
        assert_eq!(le.descriptor_length, 0x20);
        assert_eq!(le.descriptor_gva, 0x80);

        let mut buf = vec![0u8; LINEAR_DESC_MIN_LEN];
        // allocation_size = 256, handle = 0x1234
        buf[0] = 0;
        buf[1] = 1;
        buf[8] = 0x34;
        buf[9] = 0x12;
        let d = decode_buffer_descriptor(&buf).unwrap();
        assert_eq!(d.allocation_size, 256);
        assert_eq!(d.handle, 0x1234);
        assert_eq!(
            d.backing_gva_size(PAGE_SHIFT_ARM64E),
            Some(((0x1234u64) << RESOURCE_PAGE_SHIFT, 256))
        );
    }

    #[test]
    fn iosurface_type11() {
        let mut b = [0u8; 0x20];
        b[0] = 2;
        b[0x16] = 0x50;
        b[0x18] = 64;
        b[0x1c] = 32;
        match decode_descriptor(11, &b).unwrap() {
            Descriptor::IOSurfaceTexture {
                mapping_id,
                width,
                height,
                pixel_format,
                ..
            } => {
                assert_eq!(mapping_id, 2);
                assert_eq!(width, 64);
                assert_eq!(height, 32);
                assert_eq!(pixel_format, 0x50);
            }
            _ => panic!("wrong kind"),
        }
    }

    #[test]
    fn linear_texture_geometry() {
        use crate::contract::endian::{st16, st64};
        use crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;
        let mut b = vec![0u8; TEXTURE_DESC_BASE_LEN];
        st64(&mut b[0..], 0x10000);
        st32(&mut b[8..], 0x10);
        st32(&mut b[TEXTURE_DESC_ROW_STRIDE..], 256);
        st32(&mut b[TEXTURE_DESC_WIDTH..], 64);
        st32(&mut b[TEXTURE_DESC_HEIGHT..], 32);
        st16(&mut b[TEXTURE_DESC_PIXEL_FORMAT..], MTL_FORMAT_BGRA8_UNORM);
        let d = decode_texture_descriptor(&b).unwrap();
        assert_eq!(d.width, 64);
        assert_eq!(d.height, 32);
        assert_eq!(d.row_stride, 256);
        assert_eq!(d.pixel_format, MTL_FORMAT_BGRA8_UNORM);
        assert_eq!(
            d.backing_gva_size(PAGE_SHIFT_ARM64E),
            Some(((0x10u64) << RESOURCE_PAGE_SHIFT, 0x10000))
        );
        assert_eq!(d.levels.len(), 1);
        assert_eq!(d.level(0).unwrap().width, 64);
    }

    #[test]
    fn multi_mip_level_layouts() {
        use crate::contract::endian::{st16, st32, st64};
        use crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;
        // 2 mips: L0 64x32 + L1 record + format trailer shifted by 36.
        let levels = 2u32;
        let body = TEXTURE_DESC_BASE_LEN + TEXTURE_DESC_MIP_LEVEL_RECORD_LEN; // 116+36=152
        let mut b = vec![0u8; body];
        st64(&mut b[0..], 0x20000);
        st32(&mut b[8..], 0x20);
        st16(&mut b[TEXTURE_DESC_MIPMAP_LEVEL_COUNT..], levels as u16);
        st32(&mut b[TEXTURE_DESC_DATA_OFFSET..], 0);
        st32(&mut b[TEXTURE_DESC_USED_SIZE..], 64 * 32 * 4);
        st32(&mut b[TEXTURE_DESC_ROW_STRIDE..], 256);
        st32(&mut b[TEXTURE_DESC_WIDTH..], 64);
        st32(&mut b[TEXTURE_DESC_HEIGHT..], 32);
        // L1 record at +72
        let rec = TEXTURE_DESC_LEVEL_RECORDS;
        st64(&mut b[rec + TEXTURE_LEVEL_OFFSET..], 0x2000);
        st64(&mut b[rec + TEXTURE_LEVEL_SIZE..], 32 * 16 * 4);
        st64(&mut b[rec + TEXTURE_LEVEL_ROW_STRIDE..], 128);
        st32(&mut b[rec + TEXTURE_LEVEL_WIDTH..], 32);
        st32(&mut b[rec + TEXTURE_LEVEL_HEIGHT..], 16);
        st32(&mut b[rec + TEXTURE_LEVEL_DEPTH..], 1);
        // Format at 86 + 36
        let pf_off = TEXTURE_DESC_PIXEL_FORMAT + TEXTURE_DESC_MIP_LEVEL_RECORD_LEN;
        st16(&mut b[pf_off..], MTL_FORMAT_BGRA8_UNORM);
        let d = decode_texture_descriptor(&b).unwrap();
        assert_eq!(d.mipmap_level_count, 2);
        assert_eq!(d.levels.len(), 2);
        assert_eq!(d.pixel_format, MTL_FORMAT_BGRA8_UNORM);
        let l0 = d.level(0).unwrap();
        assert_eq!((l0.width, l0.height, l0.row_stride), (64, 32, 256));
        let l1 = d.level(1).unwrap();
        assert_eq!((l1.width, l1.height), (32, 16));
        assert_eq!(l1.offset, 0x2000);
        assert_eq!(l1.row_stride, 128);
        let (gva1, lay1) = d.level_gva(1, PAGE_SHIFT_ARM64E).unwrap();
        assert_eq!(gva1, ((0x20u64) << RESOURCE_PAGE_SHIFT) + 0x2000);
        assert_eq!(lay1.width, 32);
        assert!(d.level_gva(2, PAGE_SHIFT_ARM64E).is_none());
    }

    /// A mip level the descriptor named but the body does not reach is a drop,
    /// and it says so.
    ///
    /// `mipmap_level_count` keeps what the guest declared while `levels` holds
    /// fewer, so `level(n)` answers `None` for a level that was named — the same
    /// answer it gives for a level that was never named at all. Without a line
    /// here the two are indistinguishable, and the first is a texture level this
    /// device will not sample or blit.
    ///
    /// The unshifted-format fallback that used to sit under this case is gone
    /// too, so a body this short reports no format rather than reading bytes
    /// 86..88 — which for a multi-mip body are inside level record 1, not the
    /// format trailer.
    #[test]
    fn a_level_record_the_body_does_not_reach_is_reported_not_dropped() {
        use crate::contract::endian::{st16, st32, st64};
        // Declares 3 levels but carries only L0's geometry prefix and one
        // record's worth of room — L2's record runs past the end.
        let body = TEXTURE_DESC_LEVEL_RECORDS + TEXTURE_DESC_MIP_LEVEL_RECORD_LEN;
        let mut b = vec![0u8; body];
        st64(&mut b[0..], 0x20000);
        st32(&mut b[8..], 0x20);
        st16(&mut b[TEXTURE_DESC_MIPMAP_LEVEL_COUNT..], 3);
        st32(&mut b[TEXTURE_DESC_USED_SIZE..], 64 * 32 * 4);
        st32(&mut b[TEXTURE_DESC_ROW_STRIDE..], 256);
        st32(&mut b[TEXTURE_DESC_WIDTH..], 64);
        st32(&mut b[TEXTURE_DESC_HEIGHT..], 32);
        let rec = TEXTURE_DESC_LEVEL_RECORDS;
        st32(&mut b[rec + TEXTURE_LEVEL_WIDTH..], 32);
        st32(&mut b[rec + TEXTURE_LEVEL_HEIGHT..], 16);

        let cap = crate::observe::FailCapture::start();
        let d = decode_texture_descriptor(&b).unwrap();
        assert_eq!(d.mipmap_level_count, 3, "the declaration is preserved");
        assert_eq!(d.levels.len(), 2, "only two records are reachable");
        assert!(d.level(2).is_none());
        let short = cap
            .lines()
            .into_iter()
            .find(|l| l.starts_with("texture_desc_level_record_short"))
            .expect("a level the body does not reach must be reported");
        assert!(
            short.contains("declared=3") && short.contains("decoded=2"),
            "the line must name both counts: {short}"
        );
        // Same body: the format trailer sits past the end, so there is no
        // format rather than two bytes read out of a level record.
        assert!(!d.has_pixel_format, "no format is better than a wrong one");
    }

    #[test]
    fn compact_render_pipeline_funcs() {
        use crate::contract::endian::st32;
        // Minimal type-7 render pipeline: header + fieldCount=2 with vert/frag refs.
        let mut b = vec![0u8; 16 + 1 + 6 + 6];
        let blen = b.len() as u32;
        st32(&mut b[0..], TYPE7_OBJECT_RENDER_PIPELINE);
        st32(&mut b[4..], blen);
        st32(&mut b[8..], 9);
        b[16] = 2;
        b[17] = PIPELINE_TAG_VERTEX_FUNC;
        b[18] = 4;
        st32(&mut b[19..], 2);
        b[23] = PIPELINE_TAG_FRAGMENT_FUNC;
        b[24] = 4;
        st32(&mut b[25..], 1);
        let p = decode_render_pipeline_descriptor(&b).unwrap();
        assert_eq!(p.vertex_func_ref, 2);
        assert_eq!(p.fragment_func_ref, 1);
        assert_eq!(p.object_func_ref, 0);
        assert_eq!(p.mesh_func_ref, 0);
        assert_eq!(p.object_id, 9);
    }

    #[test]
    fn compact_render_pipeline_object_mesh_funcs() {
        use crate::contract::endian::st32;
        // Mesh SPI shape: tag 0x14 section offset + 0x01 object / 0x02 mesh / 0x03 frag.
        // (Host serializeMeshRenderPipelineDescriptor differentials, 2026-07-12.)
        let mut b = vec![0u8; 16 + 1 + 6 * 4];
        let blen = b.len() as u32;
        st32(&mut b[0..], TYPE7_OBJECT_RENDER_PIPELINE);
        st32(&mut b[4..], blen);
        st32(&mut b[8..], 7);
        b[16] = 4;
        b[17] = PIPELINE_TAG_MESH_SECTION_OFFSET;
        b[18] = 4;
        st32(&mut b[19..], 24); // section offset from header end
        b[23] = PIPELINE_TAG_OBJECT_FUNC; // 0x01
        b[24] = 4;
        st32(&mut b[25..], 4); // object fn ref
        b[29] = PIPELINE_TAG_MESH_FUNC; // 0x02
        b[30] = 4;
        st32(&mut b[31..], 5); // mesh fn ref
        b[35] = PIPELINE_TAG_MESH_FRAGMENT_FUNC; // 0x03
        b[36] = 4;
        st32(&mut b[37..], 3); // frag fn ref
        let p = decode_render_pipeline_descriptor(&b).unwrap();
        assert_eq!(p.object_func_ref, 4);
        assert_eq!(p.mesh_func_ref, 5);
        assert_eq!(p.fragment_func_ref, 3);
        assert_eq!(p.vertex_func_ref, 0);
        assert!(p.has_color_attachment_offset);
        assert_eq!(p.color_attachment_offset, 24);
        assert_eq!(p.object_id, 7);
    }

    #[test]
    fn depth_stencil_object_decode() {
        use crate::contract::endian::st32;
        let mut b = vec![0u8; DEPTH_STENCIL_DESC_LEN];
        st32(&mut b[0..], TYPE7_OBJECT_DEPTH_STENCIL);
        st32(&mut b[4..], DEPTH_STENCIL_DESC_LEN as u32);
        st32(&mut b[DEPTH_STENCIL_DESC_ID..], 5);
        // compare Less=1, write enabled, both stencil enabled
        let bits = 1u32
            | DEPTH_STENCIL_DEPTH_WRITE
            | DEPTH_STENCIL_FRONT_STENCIL_ENABLED
            | DEPTH_STENCIL_BACK_STENCIL_ENABLED;
        st32(&mut b[DEPTH_STENCIL_DESC_STATE_BITS..], bits);
        st32(&mut b[DEPTH_STENCIL_DESC_FRONT_FACE + 4..], 0xff);
        st32(&mut b[DEPTH_STENCIL_DESC_FRONT_FACE + 8..], 0xff);
        let d = decode_depth_stencil_descriptor(&b).unwrap();
        assert_eq!(d.depth_stencil_id, 5);
        assert_eq!(d.depth_compare_function, 1);
        assert!(d.depth_write_enabled);
        assert!(d.front_stencil_enabled);
        assert_eq!(d.front_face.read_mask, 0xff);
    }

    #[test]
    fn color_attachment0_blend_section() {
        use crate::contract::endian::st32;
        use crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;
        // Place section at off=16 (nonzero; 0 means "absent" for callers).
        // Section: count=1, entry_rel=8, entry with fieldCount + tags.
        let off = 16usize;
        let mut buf = vec![0u8; off + 8 + 1 + 6 * 3];
        st32(&mut buf[off..], 1);
        st32(&mut buf[off + 4..], 8);
        let entry = off + 8;
        buf[entry] = 3;
        buf[entry + 1] = COLOR_ATTACHMENT_TAG_PIXEL_FORMAT;
        buf[entry + 2] = 4;
        st32(&mut buf[entry + 3..], MTL_FORMAT_BGRA8_UNORM as u32);
        buf[entry + 7] = COLOR_ATTACHMENT_TAG_BLEND_ENABLE;
        buf[entry + 8] = 4;
        st32(&mut buf[entry + 9..], 1);
        buf[entry + 13] = COLOR_ATTACHMENT_TAG_DST_RGB;
        buf[entry + 14] = 4;
        st32(&mut buf[entry + 15..], 5); // OneMinusSourceAlpha
        let all = parse_color_attachments(&buf, buf.len(), off);
        let c = all.first().copied().unwrap_or_default();
        assert!(c.has_pixel_format);
        assert_eq!(c.pixel_format, MTL_FORMAT_BGRA8_UNORM as u32);
        assert!(c.blending_enabled);
        assert_eq!(c.dst_rgb, 5);
        assert_eq!(c.src_rgb, BLEND_FACTOR_ONE);
        assert_eq!(c.slot, 0);
        let all = parse_color_attachments(&buf, buf.len(), off);
        assert_eq!(all.len(), 1);
    }

    /// A section that declares three attachments and delivers one is a pipeline
    /// whose other two slots take opaque `ONE`/`ZERO` defaults, which downstream
    /// cannot tell from a guest that declared one. The loss has to say so.
    ///
    /// The entry offset for slot 1 points past the descriptor, so the walk stops
    /// after slot 0 — the same `break` the truncated-offset-word and
    /// out-of-range-entry cases take.
    #[test]
    fn a_colour_attachment_table_that_loses_entries_says_how_many() {
        use crate::contract::endian::st32;
        use crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;
        let off = 16usize;
        // Header (count + 3 offset words) then one entry: one tag, 6 bytes.
        let mut buf = vec![0u8; off + 4 + 3 * 4 + 1 + 6];
        st32(&mut buf[off..], 3);
        st32(&mut buf[off + 4..], 16); // slot 0: entry at off+16, in range
        st32(&mut buf[off + 8..], 0xffff); // slot 1: resolves past the descriptor
        st32(&mut buf[off + 12..], 0xffff); // slot 2: never reached
        let entry = off + 16;
        buf[entry] = 1;
        buf[entry + 1] = COLOR_ATTACHMENT_TAG_PIXEL_FORMAT;
        buf[entry + 2] = 4;
        st32(&mut buf[entry + 3..], MTL_FORMAT_BGRA8_UNORM as u32);

        let cap = crate::observe::FailCapture::start();
        let all = parse_color_attachments(&buf, buf.len(), off);
        let lines = cap.lines();
        assert_eq!(all.len(), 1, "only slot 0's entry is in range");

        let truncated: Vec<&String> = lines
            .iter()
            .filter(|l| l.contains("reason=color_attachment_table_truncated"))
            .collect();
        assert_eq!(
            truncated.len(),
            1,
            "one decline for the truncated table: {lines:?}"
        );
        assert!(
            truncated[0].contains("declared=3") && truncated[0].contains("decoded=1"),
            "the decline names how many were promised and how many arrived: {}",
            truncated[0]
        );
    }

    /// A colour-attachment field this decoder does not read is guest intent
    /// dropped, and the shape line beside it is what makes a boot with *no*
    /// drops readable as a measurement rather than as silence.
    #[test]
    fn an_unconsumed_colour_attachment_field_reports_its_tag_and_value() {
        use crate::contract::endian::st32;
        use crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;
        // A tag no other test in this process uses, so `first_sight` cannot
        // have latched it already.
        const UNKNOWN_TAG: u8 = 0x7f;
        const UNKNOWN_VALUE: u32 = 13;
        let off = 16usize;
        let mut buf = vec![0u8; off + 8 + 1 + 2 * 6];
        st32(&mut buf[off..], 1);
        st32(&mut buf[off + 4..], 8);
        let entry = off + 8;
        buf[entry] = 2;
        buf[entry + 1] = COLOR_ATTACHMENT_TAG_PIXEL_FORMAT;
        buf[entry + 2] = 4;
        st32(&mut buf[entry + 3..], MTL_FORMAT_BGRA8_UNORM as u32);
        buf[entry + 7] = UNKNOWN_TAG;
        buf[entry + 8] = 4;
        st32(&mut buf[entry + 9..], UNKNOWN_VALUE);

        let cap = crate::observe::FailCapture::start();
        let all = parse_color_attachments(&buf, buf.len(), off);
        let lines = cap.lines();
        // The consumed field still decodes; the census does not disturb it.
        assert_eq!(
            all.first().map(|c| c.pixel_format),
            Some(u32::from(MTL_FORMAT_BGRA8_UNORM))
        );

        let shape: Vec<&String> = lines
            .iter()
            .filter(|l| l.contains("type7_color_attach_shape"))
            .collect();
        assert_eq!(
            shape.len(),
            1,
            "one shape line per distinct entry: {lines:?}"
        );
        assert!(
            shape[0].contains("tags=[01:4,7f:4*]") && shape[0].contains("unconsumed=1"),
            "the shape line names every tag and stars the unread ones: {}",
            shape[0]
        );

        let drop: Vec<&String> = lines
            .iter()
            .filter(|l| l.contains("reason=color_attachment_field_dropped"))
            .collect();
        assert_eq!(drop.len(), 1, "one decline per dropped field: {lines:?}");
        assert!(
            drop[0].contains("tag=0x7f") && drop[0].contains("value=13"),
            "the decline carries the tag and the value, which is what \
             identifies the field: {}",
            drop[0]
        );

        // Latched: a second identical entry reports neither line again.
        let cap2 = crate::observe::FailCapture::start();
        let _ = parse_color_attachments(&buf, buf.len(), off);
        assert!(
            cap2.lines().is_empty(),
            "the census is deduped per shape and per (tag, len, value): {:?}",
            cap2.lines()
        );
    }

    /// The ninth colour-attachment tag is `MTLColorWriteMask`, and an entry
    /// that omits it left the property at `all`.
    #[test]
    fn a_colour_attachment_write_mask_decodes_and_defaults_to_all() {
        use crate::contract::endian::st32;
        use crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;

        // `[fieldCount][01 pixelFormat][09 writeMask]`, the shape the live
        // guest sent (alpha-only, value 1).
        let off = 16usize;
        let mut buf = vec![0u8; off + 8 + 1 + 2 * 6];
        st32(&mut buf[off..], 1);
        st32(&mut buf[off + 4..], 8);
        let entry = off + 8;
        buf[entry] = 2;
        buf[entry + 1] = COLOR_ATTACHMENT_TAG_PIXEL_FORMAT;
        buf[entry + 2] = 4;
        st32(&mut buf[entry + 3..], MTL_FORMAT_BGRA8_UNORM as u32);
        buf[entry + 7] = COLOR_ATTACHMENT_TAG_WRITE_MASK;
        buf[entry + 8] = 4;
        st32(&mut buf[entry + 9..], MTL_COLOR_WRITE_MASK_ALPHA);
        let masked = parse_color_attachments(&buf, buf.len(), off);
        assert_eq!(
            masked.first().map(|c| c.write_mask),
            Some(ColorWriteMask::new(MTL_COLOR_WRITE_MASK_ALPHA).unwrap())
        );
        assert_ne!(masked[0].write_mask.bits, MTL_COLOR_WRITE_MASK_ALL);

        // Same entry with the tag dropped: `all`, not `none`. This is the arm
        // a derived `Default` on a bare `u32` would have made a black
        // attachment, and every pipeline in the tree takes it.
        buf[entry] = 1;
        let plain = parse_color_attachments(&buf, buf.len(), off);
        assert_eq!(plain[0].write_mask.bits, MTL_COLOR_WRITE_MASK_ALL);
        assert_eq!(plain[0].write_mask, ColorWriteMask::default());
    }

    /// The tag identification is an argument from `MTLRenderPipeline.h`'s
    /// property order, so it needs a standing check that it still holds. A
    /// value no four-bit mask can carry refuses by name rather than masking
    /// channels off on a guess.
    #[test]
    fn a_write_mask_outside_the_four_bits_refuses_by_name() {
        use crate::contract::endian::st32;
        let off = 16usize;
        let mut buf = vec![0u8; off + 8 + 1 + 6];
        st32(&mut buf[off..], 1);
        st32(&mut buf[off + 4..], 8);
        let entry = off + 8;
        buf[entry] = 1;
        buf[entry + 1] = COLOR_ATTACHMENT_TAG_WRITE_MASK;
        buf[entry + 2] = 4;
        st32(&mut buf[entry + 3..], 0x1234_5678);

        let cap = crate::observe::FailCapture::start();
        let all = parse_color_attachments(&buf, buf.len(), off);
        assert!(
            all[0].write_mask.bits == MTL_COLOR_WRITE_MASK_ALL,
            "a refused mask leaves the attachment writing every channel, \
             which is the pre-decode behaviour rather than a new failure"
        );
        let lines = cap.lines();
        assert!(
            lines
                .iter()
                .any(|l| l.contains("reason=color_write_mask_out_of_range")
                    && l.contains("value=305419896")),
            "the refusal names the value that refuted it: {lines:?}"
        );
    }

    /// The measured case, from an x86/Vulkan boot in Dark appearance: a 27x27
    /// `RG8Unorm` corner mask at offset 0x850, 384-byte rows, in a 12 288-byte
    /// allocation. `row_stride * height` scores 12 496 and refuses it; the
    /// bytes actually read end at 12 166, with 122 to spare.
    ///
    /// The guest's allocation is exactly right, and the old bound demanded
    /// trailing padding that no row occupies — so the refusal dropped the
    /// WindowServer's whole composite draw and the window rendered with square
    /// corners and no shadow.
    #[test]
    fn a_levels_read_span_stops_at_the_last_row_not_a_last_stride() {
        const OFFSET: u64 = 0x850;
        const STRIDE: u64 = 384;
        const HEIGHT: u32 = 27;
        const TIGHT: u32 = 27 * 2;
        const ALLOCATION: u64 = 12288;

        let span = TextureLevelLayout {
            offset: OFFSET,
            size: 0,
            row_stride: STRIDE,
            width: 27,
            height: HEIGHT,
            depth: 1,
        }
        .read_span(TIGHT)
        .unwrap();
        assert_eq!(span, 26 * STRIDE + TIGHT as u64);
        assert_eq!(OFFSET + span, 12166);
        assert!(
            OFFSET + span <= ALLOCATION,
            "the guest sized this allocation for exactly this image"
        );
        // The bound this replaced, stated so the regression is visible here
        // rather than only on a live guest.
        assert!(OFFSET + STRIDE * HEIGHT as u64 > ALLOCATION);

        // A tight image (no padding) is unchanged: the two forms agree.
        let tight_span = TextureLevelLayout {
            offset: OFFSET,
            size: 0,
            row_stride: TIGHT as u64,
            width: 27,
            height: HEIGHT,
            depth: 1,
        }
        .read_span(TIGHT)
        .unwrap();
        assert_eq!(tight_span, (TIGHT as u64) * HEIGHT as u64);

        // A single row is its own tight length, with no stride charged at all.
        assert_eq!(
            TextureLevelLayout {
                offset: OFFSET,
                size: 0,
                row_stride: STRIDE,
                width: 27,
                height: 1,
                depth: 1
            }
            .read_span(TIGHT),
            Some(TIGHT as u64)
        );

        // Zero height has no rows and therefore no span; the caller rejects
        // that extent separately, and this must not underflow into a huge one.
        assert_eq!(
            TextureLevelLayout {
                offset: OFFSET,
                size: 0,
                row_stride: STRIDE,
                width: 27,
                height: 0,
                depth: 1
            }
            .read_span(TIGHT),
            None
        );
    }

    /// The array/volume form of the same rule. A slice is charged for every
    /// plane below its last one in full, and for its last plane only as far as
    /// the last row reaches — so an allocation sized exactly for N slices is
    /// accepted rather than refused for the padding after the very last row.
    #[test]
    fn a_slice_read_span_charges_full_planes_and_a_tight_last_row() {
        const STRIDE: u64 = 384;
        const HEIGHT: u32 = 27;
        const TIGHT: u32 = 27 * 2;
        let layout = TextureLevelLayout {
            offset: 0,
            size: 0,
            row_stride: STRIDE,
            width: 27,
            height: HEIGHT,
            depth: 4,
        };

        // Depth 0 and 1 are both one plane, and then this is exactly `read_span`.
        let flat = layout.read_span(TIGHT).unwrap();
        assert_eq!(layout.slice_read_span(TIGHT, 1), Some(flat));
        assert_eq!(layout.slice_read_span(TIGHT, 0), Some(flat));

        // Three whole planes, then the fourth's rows.
        let plane = STRIDE * HEIGHT as u64;
        assert_eq!(layout.slice_read_span(TIGHT, 4), Some(3 * plane + flat));

        // The stride form this replaced overcounts by exactly one row's padding,
        // whatever the plane count.
        for depth in [1u32, 2, 4] {
            assert_eq!(
                plane * u64::from(depth) - layout.slice_read_span(TIGHT, depth).unwrap(),
                STRIDE - TIGHT as u64
            );
        }

        // Zero height has no rows, so no span — and must not underflow.
        assert_eq!(
            TextureLevelLayout {
                height: 0,
                ..layout
            }
            .slice_read_span(TIGHT, 4),
            None
        );
    }

    #[test]
    fn texture_view_simple() {
        use crate::contract::endian::{st16, st32};
        let mut b = vec![0u8; TEXTURE_VIEW_MIN_SIMPLE];
        st32(
            &mut b[TEXTURE_VIEW_DESC_OPCODE..],
            TEXTURE_VIEW_OPCODE_SIMPLE,
        );
        st32(
            &mut b[TEXTURE_VIEW_DESC_LEN..],
            TEXTURE_VIEW_MIN_SIMPLE as u32,
        );
        st32(&mut b[TEXTURE_VIEW_DESC_TEXTURE_REF..], 10);
        st32(&mut b[TEXTURE_VIEW_DESC_BASE_REF..], 3);
        st16(&mut b[TEXTURE_VIEW_DESC_PIXEL_FORMAT..], 0x50);
        let v = decode_texture_view_descriptor(&b).unwrap();
        assert_eq!(v.base_texture_ref, 3);
        assert_eq!(v.view_texture_ref, 10);
        assert_eq!(v.pixel_format, 0x50);
        assert!(v.has_pixel_format);
        assert!(!v.has_swizzle);

        // A view that states no format must not claim one. This used to be an
        // unconditional `true`, which disagreed with `decode_texture_descriptor`
        // — the decoder every current reader of this flag goes through — about
        // what the flag means. `MTLPixelFormatInvalid` is 0, so a zero here is
        // an absent format and the gates that fail closed on it must see that.
        st16(&mut b[TEXTURE_VIEW_DESC_PIXEL_FORMAT..], 0);
        let none = decode_texture_view_descriptor(&b).unwrap();
        assert_eq!(none.pixel_format, 0);
        assert!(
            !none.has_pixel_format,
            "format 0 is MTLPixelFormatInvalid, not a format the view named"
        );
    }

    #[test]
    fn texture_view_swizzle_form() {
        use crate::contract::endian::{st16, st32, st64};
        let mut b = vec![0u8; TEXTURE_VIEW_MIN_SWIZZLE];
        st32(
            &mut b[TEXTURE_VIEW_DESC_OPCODE..],
            TEXTURE_VIEW_OPCODE_SWIZZLE,
        );
        st32(
            &mut b[TEXTURE_VIEW_DESC_LEN..],
            TEXTURE_VIEW_MIN_SWIZZLE as u32,
        );
        st32(&mut b[TEXTURE_VIEW_DESC_TEXTURE_REF..], 11);
        st32(&mut b[TEXTURE_VIEW_DESC_BASE_REF..], 4);
        st16(&mut b[TEXTURE_VIEW_DESC_PIXEL_FORMAT..], 0x46);
        st16(
            &mut b[TEXTURE_VIEW_DESC_TEXTURE_TYPE..],
            TEXTURE_VIEW_MTL_TYPE_2D,
        );
        st64(&mut b[TEXTURE_VIEW_DESC_LEVEL_BASE..], 1);
        st64(&mut b[TEXTURE_VIEW_DESC_LEVEL_COUNT..], 1);
        st64(&mut b[TEXTURE_VIEW_DESC_SLICE_BASE..], 0);
        st64(&mut b[TEXTURE_VIEW_DESC_SLICE_COUNT..], 1);
        // Selectors: B,G,R,A (4,3,2,5)
        b[TEXTURE_VIEW_DESC_SWIZZLE] = 4;
        b[TEXTURE_VIEW_DESC_SWIZZLE + 1] = 3;
        b[TEXTURE_VIEW_DESC_SWIZZLE + 2] = 2;
        b[TEXTURE_VIEW_DESC_SWIZZLE + 3] = 5;
        let v = decode_texture_view_descriptor(&b).unwrap();
        assert_eq!(v.view_opcode, TEXTURE_VIEW_OPCODE_SWIZZLE);
        assert_eq!(v.base_texture_ref, 4);
        assert!(v.has_levels);
        assert_eq!(v.level_base, 1);
        assert!(v.has_slices);
        assert_eq!((v.slice_base, v.slice_count), (0, 1));
        assert!(v.has_texture_type);
        assert_eq!(v.texture_type, TEXTURE_VIEW_MTL_TYPE_2D);
        assert!(v.has_swizzle);
        assert_eq!(v.swizzle, [4, 3, 2, 5]);
    }

    #[test]
    fn texture_view_ranged_form() {
        use crate::contract::endian::{st16, st32, st64};
        let mut b = vec![0u8; TEXTURE_VIEW_MIN_RANGED];
        st32(
            &mut b[TEXTURE_VIEW_DESC_OPCODE..],
            TEXTURE_VIEW_OPCODE_RANGED,
        );
        st32(
            &mut b[TEXTURE_VIEW_DESC_LEN..],
            TEXTURE_VIEW_MIN_RANGED as u32,
        );
        st32(&mut b[TEXTURE_VIEW_DESC_TEXTURE_REF..], 12);
        st32(&mut b[TEXTURE_VIEW_DESC_BASE_REF..], 5);
        st16(&mut b[TEXTURE_VIEW_DESC_PIXEL_FORMAT..], 0x50);
        st16(
            &mut b[TEXTURE_VIEW_DESC_TEXTURE_TYPE..],
            TEXTURE_VIEW_MTL_TYPE_2D,
        );
        st64(&mut b[TEXTURE_VIEW_DESC_LEVEL_BASE..], 2);
        st64(&mut b[TEXTURE_VIEW_DESC_LEVEL_COUNT..], 1);
        st64(&mut b[TEXTURE_VIEW_DESC_SLICE_BASE..], 0);
        st64(&mut b[TEXTURE_VIEW_DESC_SLICE_COUNT..], 1);
        let v = decode_texture_view_descriptor(&b).unwrap();
        assert_eq!(v.view_opcode, TEXTURE_VIEW_OPCODE_RANGED);
        assert_eq!(v.level_base, 2);
        assert_eq!(v.level_count, 1);
        assert!(!v.has_swizzle);
    }

    #[test]
    fn decodes_opcode9_buffer_texture_live_blobs() {
        // Two real 64-byte opcode-9 descriptors captured from a live x86
        // reims-vgpu-pci boot (Notification Center widget-tile sampled inputs,
        // pipe=51/53). See journal 2026-07-17 Reims VGPU-VIEW-RESOLVE-OPCODE9.
        let b1 = hex_to_bytes(
            "0900000040000000090000000800000000000000000000000005000000000000\
             421150001c0100001c0100000100000001000100010010000000000000000000",
        );
        let d1 = decode_buffer_texture_descriptor(&b1).unwrap();
        assert_eq!(d1.new_texture_ref, 9);
        assert_eq!(d1.buffer_ref, 8);
        assert_eq!(d1.offset, 0);
        assert_eq!(d1.bytes_per_row, 1280);
        assert_eq!(d1.pixel_format, 0x50); // BGRA8_UNORM
        assert_eq!(d1.texture_type, TEXTURE_VIEW_MTL_TYPE_2D);
        assert_eq!((d1.width, d1.height), (284, 284));
        assert_eq!(d1.depth, 1);
        assert_eq!(d1.mipmap_level_count, 1);
        assert_eq!(d1.sample_count, 1);
        assert_eq!(d1.array_length, 1);

        let b2 = hex_to_bytes(
            "09000000400000004c0000004b000000000000000000000000010000000000004\
             211500040000000400000000100000001000100010010000000000000000000",
        );
        let d2 = decode_buffer_texture_descriptor(&b2).unwrap();
        assert_eq!(d2.new_texture_ref, 76);
        assert_eq!(d2.buffer_ref, 75);
        assert_eq!(d2.bytes_per_row, 256);
        assert_eq!(d2.pixel_format, 0x50);
        assert_eq!((d2.width, d2.height), (64, 64));

        // A real texture-VIEW (opcode 8) is NOT a buffer texture.
        let mut view = vec![0u8; TEXTURE_VIEW_MIN_RANGED];
        crate::contract::endian::st32(
            &mut view[TEXTURE_VIEW_DESC_OPCODE..],
            TEXTURE_VIEW_OPCODE_RANGED,
        );
        crate::contract::endian::st32(
            &mut view[TEXTURE_VIEW_DESC_LEN..],
            TEXTURE_VIEW_MIN_RANGED as u32,
        );
        assert!(decode_buffer_texture_descriptor(&view).is_err());
        assert_eq!(
            texture_type8_opcode(&view),
            Some(TEXTURE_VIEW_OPCODE_RANGED)
        );
        assert_eq!(
            texture_type8_opcode(&b1),
            Some(TEXTURE_VIEW_OPCODE_BUFFER_TEXTURE)
        );
    }

    fn hex_to_bytes(s: &str) -> Vec<u8> {
        let clean: String = s.chars().filter(|c| c.is_ascii_hexdigit()).collect();
        (0..clean.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&clean[i..i + 2], 16).unwrap())
            .collect()
    }

    #[test]
    fn property_fuzz_types() {
        for t in 0u8..16 {
            let bytes = vec![0u8; 128];
            let _ = decode_descriptor(t, &bytes);
        }
    }
}
