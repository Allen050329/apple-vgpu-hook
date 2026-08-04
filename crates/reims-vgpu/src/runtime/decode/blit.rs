//! Blit command decoder (port of `host/utils/reims-vgpu-blit-decode`).

use crate::contract::endian::{ld16, ld32, ld64};
use core::mem::offset_of;
use reims_vgpu_wire::ops::blit as wire;

/// Bytes before a record's payload, from [`reims_vgpu_wire::OP_HEADER_LEN`].
///
/// One fact that used to be declared six times — here, in `render`, `compute`,
/// `blit` and twice in `stream` — all agreeing, with nothing comparing them.
/// Re-exporting makes drift impossible rather than merely detectable, which is
/// the move `contract::gva` already made for the page-table format.
pub use reims_vgpu_wire::OP_HEADER_LEN as HEADER_LEN;

/// Map a `reims-vgpu-wire` view onto a payload, translating its refusal.
///
/// The ICB layouts live in that crate, derived from Apple's own serializer and
/// pinned by fixtures, so this module reads them rather than restating them --
/// the same arrangement `decode::render` uses. A view that does not fit is a
/// payload shorter than the record claims, which is `ErrShort`.
#[inline]
fn wire_view<T: reims_vgpu_wire::Wire>(payload: &[u8]) -> Result<&T, DecodeStatus> {
    reims_vgpu_wire::view::<T>(payload).map_err(|_| DecodeStatus::ErrShort)
}

/// The three indirect-command-buffer records, which this decoder reads and no
/// executor applies.
///
/// **Apple's serializer emits all three**, so calling them "rejected" -- which
/// this module did -- was a claim about Apple that the serializer contradicts:
/// `copyIndirectCommandBuffer:sourceRange:destination:destinationIndex:` writes
/// `0x131`, `optimizeIndirectCommandBuffer:withRange:` writes `0x138` and
/// `resetCommandsInBuffer:withRange:` writes `0x139`, each pinned by a fixture
/// in `reims_vgpu_wire::ops::blit` (`CopyIcb`, `IcbRange`).
///
/// They sit inside this encoder's opcode space, between `0x12c` and `0x13e`, so
/// they are not a foreign encoder's numbers arriving in a blit segment either.
///
/// They used to be refused whole, under one shared reason. Refusing them said
/// three different things with one word, and the three are not equivalent:
/// `0x138` is Metal's *optimization hint*, so skipping it is semantically
/// correct and costs only speed, while `0x139` leaves commands live that the
/// guest asked to be reset and `0x131` leaves a destination buffer holding
/// whatever it held before. Two of the three are lost work and one is not.
/// Decoding them lets `runtime::exec` say which; see the arms there.
pub const OP_COPY_ICB: u32 = wire::OPCODE_COPY_ICB;
pub const OP_OPTIMIZE_ICB: u32 = wire::OPCODE_OPTIMIZE_ICB;
pub const OP_RESET_ICB: u32 = wire::OPCODE_RESET_ICB;

pub const OP_COPY_BUFFER_TO_TEXTURE: u32 = wire::OPCODE_COPY_BUFFER_TO_TEXTURE;
pub const OP_COPY_BUFFER_TO_BUFFER: u32 = wire::OPCODE_COPY_BUFFER_TO_BUFFER;
pub const OP_COPY_TEXTURE_TO_BUFFER: u32 = wire::OPCODE_COPY_TEXTURE_TO_BUFFER;
pub const OP_COPY_TEXTURE_TO_TEXTURE: u32 = wire::OPCODE_COPY_TEXTURE_REGION;
pub const OP_COPY_TEXTURE_TO_TEXTURE_OPTIONS: u32 = wire::OPCODE_COPY_TEXTURE_REGION_OPTIONS;
pub const OP_FILL_BUFFER: u32 = wire::OPCODE_FILL_BUFFER;
pub const OP_GENERATE_MIPMAPS: u32 = wire::OPCODE_GENERATE_MIPMAPS;
pub const OP_OPTIMIZE_CPU: u32 = wire::OPCODE_OPTIMIZE_FOR_CPU;
pub const OP_OPTIMIZE_GPU: u32 = wire::OPCODE_OPTIMIZE_FOR_GPU;
pub const OP_OPTIMIZE_IMAGE_CPU: u32 = wire::OPCODE_OPTIMIZE_FOR_CPU_SLICE_LEVEL;
pub const OP_OPTIMIZE_IMAGE_GPU: u32 = wire::OPCODE_OPTIMIZE_FOR_GPU_SLICE_LEVEL;
pub const OP_SYNCHRONIZE_RESOURCE: u32 = wire::OPCODE_SYNCHRONIZE_RESOURCE;
pub const OP_SYNCHRONIZE_TEXTURE_IMAGE: u32 = wire::OPCODE_SYNCHRONIZE_TEXTURE;
pub const OP_UPDATE_FENCE: u32 = wire::OPCODE_UPDATE_FENCE;
pub const OP_WAIT_FENCE: u32 = wire::OPCODE_WAIT_FOR_FENCE;
pub const OP_COPY_TEXTURE_TO_TEXTURE_SLICE_LEVEL: u32 = wire::OPCODE_COPY_TEXTURE_SLICES;

/// The five records `-setSupportsBlitEncoderSPI:` gates.
///
/// This encoder's opcode run does not stop at `0x13e`. It was read that way for
/// as long as the wire capture drove the class with every capability at its
/// default, and all sixteen of those default off -- so five real records looked
/// like selectors Apple never emits, and this decoder answered every one of
/// them with `ErrUnknownOpcode`.
///
/// Each is pinned by a fixture in [`reims_vgpu_wire::ops::blit`], and the three
/// fills are writes to guest-visible memory: dropping one leaves the
/// destination holding whatever it held before, which the guest then reads back
/// as content it believes it just wrote.
///
/// `0x142` and `0x143` are the [`wire::Ref`] and [`wire::RefSliceLevel`] shapes
/// exactly, and this module deliberately does **not** fold them into the arms
/// that already decode those shapes: those arms end in the executor's
/// unconditional no-op, and a compressed-texture invalidate that silently
/// joined them would be indistinguishable from a `synchronizeTexture:` that
/// genuinely needs nothing done.
pub const OP_FILL_BUFFER_PATTERN4: u32 = wire::OPCODE_FILL_BUFFER_PATTERN4;
pub const OP_FILL_TEXTURE_BYTES: u32 = wire::OPCODE_FILL_TEXTURE_BYTES;
pub const OP_FILL_TEXTURE_COLOR: u32 = wire::OPCODE_FILL_TEXTURE_COLOR;
pub const OP_INVALIDATE_COMPRESSED_TEXTURE: u32 = wire::OPCODE_INVALIDATE_COMPRESSED_TEXTURE;
pub const OP_INVALIDATE_COMPRESSED_TEXTURE_SLICE_LEVEL: u32 =
    wire::OPCODE_INVALIDATE_COMPRESSED_TEXTURE_SLICE_LEVEL;

// MTLBlitOption (Metal.framework Headers/MTLBlitCommandEncoder.h).
pub const MTL_BLIT_OPTION_NONE: u32 = 0;
pub const MTL_BLIT_OPTION_DEPTH_FROM_DEPTH_STENCIL: u32 = 1 << 0;
pub const MTL_BLIT_OPTION_STENCIL_FROM_DEPTH_STENCIL: u32 = 1 << 1;
pub const MTL_BLIT_OPTION_ROW_LINEAR_PVRTC: u32 = 1 << 2;
/// All bits defined by the Metal SDK for `MTLBlitOption`.
pub const MTL_BLIT_OPTION_KNOWN_MASK: u32 = MTL_BLIT_OPTION_DEPTH_FROM_DEPTH_STENCIL
    | MTL_BLIT_OPTION_STENCIL_FROM_DEPTH_STENCIL
    | MTL_BLIT_OPTION_ROW_LINEAR_PVRTC;

/// Selected texture aspect for a buffer↔texture / options-bearing copy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlitAspect {
    /// Full texel (options None / zero).
    Full,
    /// Depth plane of a depth or depth-stencil texture.
    Depth,
    /// Stencil plane of a stencil or depth-stencil texture.
    Stencil,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BlitOptionError {
    UnknownBits,
    RowLinearPvrtc,
    ConflictingAspects,
}

impl crate::observe::Decline for BlitOptionError {
    fn slug(&self) -> &'static str {
        match self {
            Self::UnknownBits => "blit_options_unknown_bits",
            Self::RowLinearPvrtc => "blit_options_row_linear_pvrtc",
            Self::ConflictingAspects => "blit_options_conflicting_aspects",
        }
    }
}

/// Parse wire `MTLBlitOption` bits into a product-path aspect selection.
///
/// - Zero / absent options → [`BlitAspect::Full`]
/// - Depth and stencil bits are mutually exclusive
/// - `RowLinearPVRTC` and unknown bits fail (no PVRTC rail; unknown stays unknown)
pub fn parse_blit_options(has_options: bool, options: u32) -> Result<BlitAspect, BlitOptionError> {
    if !has_options || options == 0 {
        return Ok(BlitAspect::Full);
    }
    if options & !MTL_BLIT_OPTION_KNOWN_MASK != 0 {
        return Err(BlitOptionError::UnknownBits);
    }
    if options & MTL_BLIT_OPTION_ROW_LINEAR_PVRTC != 0 {
        // Compressed PVRTC row-linear layout is not on the product path.
        return Err(BlitOptionError::RowLinearPvrtc);
    }
    let depth = options & MTL_BLIT_OPTION_DEPTH_FROM_DEPTH_STENCIL != 0;
    let stencil = options & MTL_BLIT_OPTION_STENCIL_FROM_DEPTH_STENCIL != 0;
    match (depth, stencil) {
        (true, false) => Ok(BlitAspect::Depth),
        (false, true) => Ok(BlitAspect::Stencil),
        (false, false) => Ok(BlitAspect::Full),
        (true, true) => Err(BlitOptionError::ConflictingAspects),
    }
}

// format offsets (payload-relative)
const REF_SOURCE: usize = 0;
const REF_DESTINATION: usize = 4;
const POINT_X: usize = 0;
const POINT_Y: usize = 8;
const POINT_Z: usize = 16;

const CBT_SRC_OFF: usize = offset_of!(wire::CopyBufferToTexture, source_offset);
const CBT_SRC_BPR: usize = offset_of!(wire::CopyBufferToTexture, source_bytes_per_row);
const CBT_SRC_BPI: usize = offset_of!(wire::CopyBufferToTexture, source_bytes_per_image);
const CBT_SRC_SIZE: usize = offset_of!(wire::CopyBufferToTexture, size_width);
const CBT_DST_ORIGIN: usize = offset_of!(wire::CopyBufferToTexture, dest_origin_x);
const CBT_DST_SLICE: usize = offset_of!(wire::CopyBufferToTexture, dest_slice);
const CBT_DST_LEVEL: usize = offset_of!(wire::CopyBufferToTexture, dest_level);
const CBT_OPTIONS: usize = offset_of!(wire::CopyBufferToTexture, options);
const CBT_LEN: usize = wire::COPY_BUFFER_TO_TEXTURE_TOTAL_LEN as usize;

const CBB_SRC_OFF: usize = offset_of!(wire::BufferToBuffer, source_offset);
const CBB_DST_OFF: usize = offset_of!(wire::BufferToBuffer, dest_offset);
const CBB_SIZE: usize = offset_of!(wire::BufferToBuffer, size);
const CBB_LEN: usize = wire::COPY_BUFFER_TO_BUFFER_TOTAL_LEN as usize;

const CTB_SRC_ORIGIN: usize = offset_of!(wire::CopyTextureToBuffer, source_origin_x);
const CTB_SRC_SIZE: usize = offset_of!(wire::CopyTextureToBuffer, size_width);
const CTB_DST_OFF: usize = offset_of!(wire::CopyTextureToBuffer, dest_offset);
const CTB_DST_BPR: usize = offset_of!(wire::CopyTextureToBuffer, dest_bytes_per_row);
const CTB_DST_BPI: usize = offset_of!(wire::CopyTextureToBuffer, dest_bytes_per_image);
const CTB_SRC_SLICE: usize = offset_of!(wire::CopyTextureToBuffer, source_slice);
const CTB_SRC_LEVEL: usize = offset_of!(wire::CopyTextureToBuffer, source_level);
const CTB_OPTIONS: usize = offset_of!(wire::CopyTextureToBuffer, options);
const CTB_LEN: usize = wire::COPY_TEXTURE_TO_BUFFER_TOTAL_LEN as usize;

const CTT_SRC_ORIGIN: usize = offset_of!(wire::CopyTextureRegion, source_origin_x);
const CTT_SRC_SIZE: usize = offset_of!(wire::CopyTextureRegion, size_width);
const CTT_DST_ORIGIN: usize = offset_of!(wire::CopyTextureRegion, dest_origin_x);
const CTT_SRC_SLICE: usize = offset_of!(wire::CopyTextureRegion, source_slice);
const CTT_SRC_LEVEL: usize = offset_of!(wire::CopyTextureRegion, source_level);
const CTT_DST_SLICE: usize = offset_of!(wire::CopyTextureRegion, dest_slice);
const CTT_DST_LEVEL: usize = offset_of!(wire::CopyTextureRegion, dest_level);
const CTT_OPTIONS: usize = offset_of!(wire::CopyTextureRegionOptions, options);
const CTT_LEN: usize = wire::COPY_TEXTURE_REGION_TOTAL_LEN as usize;
const CTT_OPTIONS_LEN: usize = wire::COPY_TEXTURE_REGION_OPTIONS_TOTAL_LEN as usize;

const CTTSL_SRC_SLICE: usize = offset_of!(wire::CopyTextureSlices, source_slice);
const CTTSL_SRC_LEVEL: usize = offset_of!(wire::CopyTextureSlices, source_level);
const CTTSL_DST_SLICE: usize = offset_of!(wire::CopyTextureSlices, dest_slice);
const CTTSL_DST_LEVEL: usize = offset_of!(wire::CopyTextureSlices, dest_level);
const CTTSL_SLICE_COUNT: usize = offset_of!(wire::CopyTextureSlices, slice_count);
const CTTSL_LEVEL_COUNT: usize = offset_of!(wire::CopyTextureSlices, level_count);
const CTTSL_LEN: usize = wire::COPY_TEXTURE_SLICES_TOTAL_LEN as usize;

const FILL_REF: usize = offset_of!(wire::FillBuffer, buffer_ref);
const FILL_RANGE_LOC: usize = offset_of!(wire::FillBuffer, range_location);
const FILL_RANGE_LEN: usize = offset_of!(wire::FillBuffer, range_length);
const FILL_VALUE: usize = offset_of!(wire::FillBuffer, value);
const FILL_LEN: usize = wire::FILL_BUFFER_TOTAL_LEN as usize;

const RESOURCE_REF: usize = offset_of!(wire::Ref, object_ref);
const RESOURCE_LEN: usize = wire::REF_TOTAL_LEN as usize;
const IMAGE_TEXTURE: usize = offset_of!(wire::RefSliceLevel, texture_ref);
const IMAGE_SLICE: usize = offset_of!(wire::RefSliceLevel, slice);
const IMAGE_LEVEL: usize = offset_of!(wire::RefSliceLevel, level);
const IMAGE_LEN: usize = wire::REF_SLICE_LEVEL_TOTAL_LEN as usize;
const FENCE_REF: usize = offset_of!(wire::Ref, object_ref);
const FENCE_LEN: usize = wire::REF_TOTAL_LEN as usize;

/// Why the blit decoder refused a command.
///
/// There is deliberately no `Ok`: `decode` returns `Result<Command, _>`, so
/// success is the `Ok` arm of the result and not a variant here. `ErrArgs` went
/// with it — every argument this decoder rejects is a payload too short for the
/// field it was about to read, which is `ErrShort`. Both were constructed only
/// by the test that listed them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeStatus {
    ErrShort,
    ErrUnknownOpcode,
}

impl crate::observe::Refusal for DecodeStatus {
    /// Slugs carry a `blit_decode_` prefix.
    ///
    /// `DecodeStatus` is **seven separate enums** in `runtime/decode/`, one per
    /// module, and five of them have an `ErrShort`. Without the prefix they
    /// would all answer with the same name for five different reads, which is
    /// the collapse the crate-wide uniqueness gate exists to refuse.
    fn refusal(&self) -> Option<&'static str> {
        Some(match self {
            Self::ErrShort => "blit_decode_short",
            Self::ErrUnknownOpcode => "blit_decode_unknown_opcode",
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Kind {
    #[default]
    Unknown = 0,
    Copy,
    FillBuffer,
    Resource,
    Image,
    Fence,
    /// `optimizeIndirectCommandBuffer:withRange:` and
    /// `resetCommandsInBuffer:withRange:`, which name one ICB and a command
    /// range. Which of the two is [`Command::opcode`]; the buffer is
    /// [`Command::resource`] and the range is [`Command::range_location`] and
    /// [`Command::range_length`].
    IcbRange,
    /// `copyIndirectCommandBuffer:sourceRange:destination:destinationIndex:`.
    /// Source and destination are [`Command::source`] and
    /// [`Command::destination`], the source range is the shared range pair, and
    /// the destination command index is [`Command::destination_index`].
    IcbCopy,
    /// `fillBuffer:range:pattern4:` — [`Kind::FillBuffer`] with a repeating
    /// 32-bit unit in [`Command::fill_pattern`] instead of the single byte in
    /// [`Command::fill_value`]. Separate kinds rather than one kind with two
    /// value fields, because an executor that read the wrong one would write
    /// plausible bytes rather than fail.
    FillBufferPattern4,
    /// `fillTexture:level:slice:region:color:[pixelFormat:]` and
    /// `fillTexture:level:slice:region:bytes:length:`, told apart by
    /// [`Command::fill_source`].
    FillTexture,
    /// `invalidateCompressedTexture:` and its `slice:level:` form. The texture
    /// is [`Command::texture`]; the whole-texture form leaves
    /// [`Command::slice`] and [`Command::level`] at zero, and
    /// [`Command::opcode`] is what distinguishes "slice 0, level 0" from "every
    /// slice and every level".
    InvalidateCompressedTexture,
}

/// Where a [`Kind::FillTexture`] takes the value it writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum FillSource {
    #[default]
    None = 0,
    /// A clear colour, in [`Command::fill_color_raw`], interpreted in
    /// [`Command::fill_pixel_format`].
    Color,
    /// A pixel pattern the serializer staged into a buffer rather than putting
    /// it on the wire: [`Command::fill_bytes_ref`],
    /// [`Command::fill_bytes_offset`] and [`Command::fill_bytes_length`]. The
    /// record carries no pixel data.
    Bytes,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum CopyKind {
    #[default]
    None = 0,
    BufferToTexture,
    BufferToBuffer,
    TextureToBuffer,
    TextureToTexture,
    TextureToTextureSliceLevel,
}

/// What a blit command's source/destination/resource ref names.
///
/// There is deliberately no `Fence`: the two fence opcodes carry their ref in
/// [`Command::fence`] under [`Kind::Fence`] and leave every `RefKind` field at
/// `None`, so a `RefKind::Fence` was a second name for a ref this decoder puts
/// somewhere else.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum RefKind {
    #[default]
    None = 0,
    Buffer,
    Texture,
    Resource,
    /// An `MTLIndirectCommandBuffer`. Not folded into [`Self::Resource`]: the
    /// three ICB records reach it through `indirectCommandBufferRef`, which is
    /// a different accessor from every other resource in this encoder, and a
    /// consumer that resolved one as a generic resource would look it up in the
    /// wrong table.
    IndirectCommandBuffer,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Point {
    pub x: u64,
    pub y: u64,
    pub z: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Size {
    pub width: u64,
    pub height: u64,
    pub depth: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Command {
    pub opcode: u32,
    pub command_length: u32,
    pub kind: Kind,
    pub copy_kind: CopyKind,
    pub source_kind: RefKind,
    pub destination_kind: RefKind,
    pub source: u32,
    pub destination: u32,
    pub source_offset: u64,
    pub source_bytes_per_row: u64,
    pub source_bytes_per_image: u64,
    pub source_origin: Point,
    pub source_size: Size,
    pub destination_offset: u64,
    pub destination_bytes_per_row: u64,
    pub destination_bytes_per_image: u64,
    pub destination_origin: Point,
    pub size: u64,
    pub source_slice: u16,
    pub source_level: u16,
    pub destination_slice: u16,
    pub destination_level: u16,
    pub slice_count: u16,
    pub level_count: u16,
    pub has_options: bool,
    pub options: u32,
    pub resource: u32,
    pub resource_kind: RefKind,
    pub buffer: u32,
    pub range_location: u64,
    pub range_length: u64,
    /// Destination command index of a [`Kind::IcbCopy`].
    ///
    /// A `u64` of its own rather than one of the `u16` subresource indices
    /// beside it: this counts *commands* in the destination buffer, and the
    /// record keeps the two kinds of index apart.
    pub destination_index: u64,
    pub fill_value: u8,
    /// The repeating 32-bit unit of a [`Kind::FillBufferPattern4`].
    pub fill_pattern: u32,
    pub texture: u32,
    pub slice: u16,
    pub level: u16,
    pub fence: u32,

    // --- Kind::FillTexture ---------------------------------------------------
    pub fill_source: FillSource,
    /// The filled region. Named apart from the copy records' origin and size
    /// because a fill has one texture and no direction: reusing
    /// `source_origin`/`destination_origin` here would make a reader ask which
    /// end of a copy this was.
    ///
    /// The wire stores the region **size before its origin**, reversing
    /// `MTLRegion` and the selector's own type encoding. That is the wire
    /// crate's business and this decoder reads it through the pinned view, but
    /// it is the reason not to hand-roll the offsets here.
    pub fill_origin: Point,
    pub fill_size: Size,
    /// The clear colour's four components as the serializer wrote them.
    ///
    /// IEEE-754 `double` bit patterns rather than `f64`, for two reasons:
    /// [`Command`] is `Eq`, which `f64` cannot be, and nothing in this device
    /// converts them yet — so storing the bits keeps the record exact instead of
    /// committing to a conversion no consumer has asked for.
    pub fill_color_raw: [u64; 4],
    /// The format the colour is interpreted in. Comes from the guest's argument
    /// on the `pixelFormat:` form and from the *texture* on the other, and the
    /// wire record cannot tell you which — [`Command::opcode`] is the same for
    /// both.
    pub fill_pixel_format: u16,
    /// The staging buffer a [`FillSource::Bytes`] fill named, its offset, and
    /// the number of pattern bytes there. An executor must fetch them; the
    /// record carries no pixel data and the length is the guest's rather than
    /// anything derivable from the region.
    pub fill_bytes_ref: u32,
    pub fill_bytes_offset: u64,
    pub fill_bytes_length: u64,
}

fn decode_origin(p: &[u8]) -> Point {
    Point {
        x: ld64(&p[POINT_X..]),
        y: ld64(&p[POINT_Y..]),
        z: ld64(&p[POINT_Z..]),
    }
}

fn decode_size(p: &[u8]) -> Size {
    Size {
        width: ld64(&p[POINT_X..]),
        height: ld64(&p[POINT_Y..]),
        depth: ld64(&p[POINT_Z..]),
    }
}

/// Transactional decode of one blit command record.
pub fn decode(command: &[u8]) -> Result<Command, DecodeStatus> {
    if command.len() < HEADER_LEN {
        return Err(DecodeStatus::ErrShort);
    }
    let opcode = ld32(&command[0..]);
    let command_length = ld32(&command[4..]) as usize;
    if command_length < HEADER_LEN || command_length > command.len() {
        return Err(DecodeStatus::ErrShort);
    }
    let payload = &command[HEADER_LEN..command_length];
    let mut out = Command {
        opcode,
        command_length: command_length as u32,
        ..Default::default()
    };

    match opcode {
        OP_COPY_BUFFER_TO_TEXTURE => {
            if command_length != CBT_LEN {
                return Err(DecodeStatus::ErrShort);
            }
            out.kind = Kind::Copy;
            out.copy_kind = CopyKind::BufferToTexture;
            out.source_kind = RefKind::Buffer;
            out.destination_kind = RefKind::Texture;
            out.source = ld32(&payload[REF_SOURCE..]);
            out.destination = ld32(&payload[REF_DESTINATION..]);
            out.source_offset = ld64(&payload[CBT_SRC_OFF..]);
            out.source_bytes_per_row = ld64(&payload[CBT_SRC_BPR..]);
            out.source_bytes_per_image = ld64(&payload[CBT_SRC_BPI..]);
            out.source_size = decode_size(&payload[CBT_SRC_SIZE..]);
            out.destination_origin = decode_origin(&payload[CBT_DST_ORIGIN..]);
            out.destination_slice = ld16(&payload[CBT_DST_SLICE..]);
            out.destination_level = ld16(&payload[CBT_DST_LEVEL..]);
            out.has_options = true;
            out.options = ld32(&payload[CBT_OPTIONS..]);
            Ok(out)
        }
        OP_COPY_BUFFER_TO_BUFFER => {
            if command_length != CBB_LEN {
                return Err(DecodeStatus::ErrShort);
            }
            out.kind = Kind::Copy;
            out.copy_kind = CopyKind::BufferToBuffer;
            out.source_kind = RefKind::Buffer;
            out.destination_kind = RefKind::Buffer;
            out.source = ld32(&payload[REF_SOURCE..]);
            out.destination = ld32(&payload[REF_DESTINATION..]);
            out.source_offset = ld64(&payload[CBB_SRC_OFF..]);
            out.destination_offset = ld64(&payload[CBB_DST_OFF..]);
            out.size = ld64(&payload[CBB_SIZE..]);
            Ok(out)
        }
        OP_COPY_TEXTURE_TO_BUFFER => {
            if command_length != CTB_LEN {
                return Err(DecodeStatus::ErrShort);
            }
            out.kind = Kind::Copy;
            out.copy_kind = CopyKind::TextureToBuffer;
            out.source_kind = RefKind::Texture;
            out.destination_kind = RefKind::Buffer;
            out.source = ld32(&payload[REF_SOURCE..]);
            out.destination = ld32(&payload[REF_DESTINATION..]);
            out.source_origin = decode_origin(&payload[CTB_SRC_ORIGIN..]);
            out.source_size = decode_size(&payload[CTB_SRC_SIZE..]);
            out.destination_offset = ld64(&payload[CTB_DST_OFF..]);
            out.destination_bytes_per_row = ld64(&payload[CTB_DST_BPR..]);
            out.destination_bytes_per_image = ld64(&payload[CTB_DST_BPI..]);
            out.source_slice = ld16(&payload[CTB_SRC_SLICE..]);
            out.source_level = ld16(&payload[CTB_SRC_LEVEL..]);
            out.has_options = true;
            // **Two bytes, not four.** This is the one copy record that narrows
            // `options`: `blit_copy_texture_to_buffer_options` came back
            // `04 00 AA AA` at `+0x54` against the oracle's poison, where the
            // buffer-to-texture and region forms both fill all four
            // (`reims_vgpu_wire::ops::blit::CopyTextureToBuffer` reads a `u16`
            // and its siblings a `u32`). Reading four here took two bytes of
            // whatever the guest's ring last held into the high half, and
            // `parse_blit_options` refuses any bit outside `MTLBlitOption`'s
            // three — so a plain full-texel copy, whose two written bytes are
            // zero, was declined whenever those stale bytes happened to be
            // non-zero.
            out.options = ld16(&payload[CTB_OPTIONS..]) as u32;
            Ok(out)
        }
        OP_COPY_TEXTURE_TO_TEXTURE | OP_COPY_TEXTURE_TO_TEXTURE_OPTIONS => {
            let want = if opcode == OP_COPY_TEXTURE_TO_TEXTURE_OPTIONS {
                CTT_OPTIONS_LEN
            } else {
                CTT_LEN
            };
            if command_length != want {
                return Err(DecodeStatus::ErrShort);
            }
            out.kind = Kind::Copy;
            out.copy_kind = CopyKind::TextureToTexture;
            out.source_kind = RefKind::Texture;
            out.destination_kind = RefKind::Texture;
            out.source = ld32(&payload[REF_SOURCE..]);
            out.destination = ld32(&payload[REF_DESTINATION..]);
            out.source_origin = decode_origin(&payload[CTT_SRC_ORIGIN..]);
            out.source_size = decode_size(&payload[CTT_SRC_SIZE..]);
            out.destination_origin = decode_origin(&payload[CTT_DST_ORIGIN..]);
            out.source_slice = ld16(&payload[CTT_SRC_SLICE..]);
            out.source_level = ld16(&payload[CTT_SRC_LEVEL..]);
            out.destination_slice = ld16(&payload[CTT_DST_SLICE..]);
            out.destination_level = ld16(&payload[CTT_DST_LEVEL..]);
            if opcode == OP_COPY_TEXTURE_TO_TEXTURE_OPTIONS {
                out.has_options = true;
                out.options = ld32(&payload[CTT_OPTIONS..]);
            }
            Ok(out)
        }
        OP_COPY_TEXTURE_TO_TEXTURE_SLICE_LEVEL => {
            if command_length != CTTSL_LEN {
                return Err(DecodeStatus::ErrShort);
            }
            out.kind = Kind::Copy;
            out.copy_kind = CopyKind::TextureToTextureSliceLevel;
            out.source_kind = RefKind::Texture;
            out.destination_kind = RefKind::Texture;
            out.source = ld32(&payload[REF_SOURCE..]);
            out.destination = ld32(&payload[REF_DESTINATION..]);
            out.source_slice = ld16(&payload[CTTSL_SRC_SLICE..]);
            out.source_level = ld16(&payload[CTTSL_SRC_LEVEL..]);
            out.destination_slice = ld16(&payload[CTTSL_DST_SLICE..]);
            out.destination_level = ld16(&payload[CTTSL_DST_LEVEL..]);
            out.slice_count = ld16(&payload[CTTSL_SLICE_COUNT..]);
            out.level_count = ld16(&payload[CTTSL_LEVEL_COUNT..]);
            Ok(out)
        }
        OP_FILL_BUFFER => {
            if command_length != FILL_LEN {
                return Err(DecodeStatus::ErrShort);
            }
            out.kind = Kind::FillBuffer;
            out.buffer = ld32(&payload[FILL_REF..]);
            out.range_location = ld64(&payload[FILL_RANGE_LOC..]);
            out.range_length = ld64(&payload[FILL_RANGE_LEN..]);
            out.fill_value = payload[FILL_VALUE];
            Ok(out)
        }
        OP_GENERATE_MIPMAPS | OP_OPTIMIZE_CPU | OP_OPTIMIZE_GPU => {
            if command_length != RESOURCE_LEN {
                return Err(DecodeStatus::ErrShort);
            }
            out.kind = Kind::Resource;
            out.resource_kind = RefKind::Texture;
            out.resource = ld32(&payload[RESOURCE_REF..]);
            Ok(out)
        }
        OP_OPTIMIZE_IMAGE_CPU | OP_OPTIMIZE_IMAGE_GPU | OP_SYNCHRONIZE_TEXTURE_IMAGE => {
            if command_length != IMAGE_LEN {
                return Err(DecodeStatus::ErrShort);
            }
            out.kind = Kind::Image;
            out.texture = ld32(&payload[IMAGE_TEXTURE..]);
            out.slice = ld16(&payload[IMAGE_SLICE..]);
            out.level = ld16(&payload[IMAGE_LEVEL..]);
            Ok(out)
        }
        OP_SYNCHRONIZE_RESOURCE => {
            if command_length != RESOURCE_LEN {
                return Err(DecodeStatus::ErrShort);
            }
            out.kind = Kind::Resource;
            out.resource_kind = RefKind::Resource;
            out.resource = ld32(&payload[RESOURCE_REF..]);
            Ok(out)
        }
        OP_UPDATE_FENCE | OP_WAIT_FENCE => {
            if command_length != FENCE_LEN {
                return Err(DecodeStatus::ErrShort);
            }
            out.kind = Kind::Fence;
            out.fence = ld32(&payload[FENCE_REF..]);
            Ok(out)
        }
        OP_OPTIMIZE_ICB | OP_RESET_ICB => {
            if command_length != wire::ICB_RANGE_TOTAL_LEN as usize {
                return Err(DecodeStatus::ErrShort);
            }
            let r = wire_view::<wire::IcbRange>(payload)?;
            out.kind = Kind::IcbRange;
            out.resource_kind = RefKind::IndirectCommandBuffer;
            out.resource = r.icb_ref.get();
            // Two `u64` in declaration order with no narrowing, which is what
            // separates this shape from the ref-slice-level records that are
            // also "a ref and two numbers".
            out.range_location = r.range_location.get();
            out.range_length = r.range_length.get();
            Ok(out)
        }
        OP_COPY_ICB => {
            if command_length != wire::COPY_ICB_TOTAL_LEN as usize {
                return Err(DecodeStatus::ErrShort);
            }
            let c = wire_view::<wire::CopyIcb>(payload)?;
            out.kind = Kind::IcbCopy;
            out.source_kind = RefKind::IndirectCommandBuffer;
            out.destination_kind = RefKind::IndirectCommandBuffer;
            out.source = c.source_ref.get();
            out.destination = c.dest_ref.get();
            out.range_location = c.range_location.get();
            out.range_length = c.range_length.get();
            out.destination_index = c.dest_index.get();
            Ok(out)
        }
        OP_FILL_BUFFER_PATTERN4 => {
            if command_length != wire::FILL_BUFFER_PATTERN4_TOTAL_LEN as usize {
                return Err(DecodeStatus::ErrShort);
            }
            let f = wire_view::<wire::FillBufferPattern4>(payload)?;
            out.kind = Kind::FillBufferPattern4;
            out.buffer = f.buffer_ref.get();
            out.range_location = f.range_location.get();
            out.range_length = f.range_length.get();
            out.fill_pattern = f.pattern.get();
            Ok(out)
        }
        OP_FILL_TEXTURE_COLOR => {
            if command_length != wire::FILL_TEXTURE_COLOR_TOTAL_LEN as usize {
                return Err(DecodeStatus::ErrShort);
            }
            let f = wire_view::<wire::FillTextureColor>(payload)?;
            out.kind = Kind::FillTexture;
            out.fill_source = FillSource::Color;
            out.texture = f.texture_ref.get();
            out.level = f.level.get();
            out.slice = f.slice.get();
            out.fill_origin = Point {
                x: f.origin_x.get(),
                y: f.origin_y.get(),
                z: f.origin_z.get(),
            };
            out.fill_size = Size {
                width: f.size_width.get(),
                height: f.size_height.get(),
                depth: f.size_depth.get(),
            };
            out.fill_color_raw = [
                f.color_red.get().to_bits(),
                f.color_green.get().to_bits(),
                f.color_blue.get().to_bits(),
                f.color_alpha.get().to_bits(),
            ];
            out.fill_pixel_format = f.pixel_format.get();
            Ok(out)
        }
        OP_FILL_TEXTURE_BYTES => {
            if command_length != wire::FILL_TEXTURE_BYTES_TOTAL_LEN as usize {
                return Err(DecodeStatus::ErrShort);
            }
            let f = wire_view::<wire::FillTextureBytes>(payload)?;
            out.kind = Kind::FillTexture;
            out.fill_source = FillSource::Bytes;
            out.texture = f.texture_ref.get();
            out.level = f.level.get();
            out.slice = f.slice.get();
            out.fill_origin = Point {
                x: f.origin_x.get(),
                y: f.origin_y.get(),
                z: f.origin_z.get(),
            };
            out.fill_size = Size {
                width: f.size_width.get(),
                height: f.size_height.get(),
                depth: f.size_depth.get(),
            };
            out.fill_bytes_ref = f.bytes_ref.get();
            out.fill_bytes_offset = f.bytes_offset.get();
            out.fill_bytes_length = f.length.get();
            Ok(out)
        }
        OP_INVALIDATE_COMPRESSED_TEXTURE => {
            if command_length != wire::REF_TOTAL_LEN as usize {
                return Err(DecodeStatus::ErrShort);
            }
            let r = wire_view::<wire::Ref>(payload)?;
            out.kind = Kind::InvalidateCompressedTexture;
            out.texture = r.object_ref.get();
            Ok(out)
        }
        OP_INVALIDATE_COMPRESSED_TEXTURE_SLICE_LEVEL => {
            if command_length != wire::REF_SLICE_LEVEL_TOTAL_LEN as usize {
                return Err(DecodeStatus::ErrShort);
            }
            let r = wire_view::<wire::RefSliceLevel>(payload)?;
            out.kind = Kind::InvalidateCompressedTexture;
            out.texture = r.texture_ref.get();
            out.slice = r.slice.get();
            out.level = r.level.get();
            Ok(out)
        }
        _ => Err(DecodeStatus::ErrUnknownOpcode),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::endian::{st16, st32, st64};

    fn hdr(opcode: u32, len: u32) -> Vec<u8> {
        let mut v = vec![0u8; len as usize];
        st32(&mut v[0..4], opcode);
        st32(&mut v[4..8], len);
        v
    }

    /// A blit record that fails to decode used to be `Err(_) => return` at the
    /// dispatch site — a dropped guest command indistinguishable from a segment
    /// carrying no blit work. Each of the four checks now names itself, and `Ok`
    /// still produces no line.
    #[test]
    fn every_blit_decode_failure_names_its_own_check() {
        use crate::observe::{Decline, Refusal};
        const ALL: &[DecodeStatus] = &[DecodeStatus::ErrShort, DecodeStatus::ErrUnknownOpcode];
        let mut slugs: Vec<&str> = ALL.iter().filter_map(|s| s.refusal()).collect();
        assert_eq!(slugs.len(), ALL.len(), "every variant refuses");
        slugs.sort_unstable();
        slugs.dedup();
        assert_eq!(slugs.len(), ALL.len(), "two blit decode checks share a slug");
        // The prefix is load-bearing: seven modules define a `DecodeStatus` and
        // five of them have an `ErrShort` meaning a different read.
        assert!(slugs.iter().all(|s| s.starts_with("blit_decode_")));

        // The three option checks used to be discarded by `map_err(|_| ..)`.
        let mut opts: Vec<&str> = [
            BlitOptionError::UnknownBits,
            BlitOptionError::RowLinearPvrtc,
            BlitOptionError::ConflictingAspects,
        ]
        .iter()
        .map(|e| e.slug())
        .collect();
        opts.sort_unstable();
        opts.dedup();
        assert_eq!(opts.len(), 3, "two blit option checks share a slug");
    }

    #[test]
    fn buffer_to_buffer() {
        let mut v = hdr(OP_COPY_BUFFER_TO_BUFFER, CBB_LEN as u32);
        st32(&mut v[8..], 1);
        st32(&mut v[12..], 2);
        st64(&mut v[8 + CBB_SRC_OFF..], 0x10);
        st64(&mut v[8 + CBB_DST_OFF..], 0x20);
        st64(&mut v[8 + CBB_SIZE..], 0x30);
        let c = decode(&v).unwrap();
        assert_eq!(c.copy_kind, CopyKind::BufferToBuffer);
        assert_eq!(c.source, 1);
        assert_eq!(c.destination, 2);
        assert_eq!(c.size, 0x30);
    }

    #[test]
    fn fill_buffer() {
        let mut v = hdr(OP_FILL_BUFFER, FILL_LEN as u32);
        st32(&mut v[8 + FILL_REF..], 9);
        st64(&mut v[8 + FILL_RANGE_LOC..], 4);
        st64(&mut v[8 + FILL_RANGE_LEN..], 8);
        v[8 + FILL_VALUE] = 0xAB;
        let c = decode(&v).unwrap();
        assert_eq!(c.kind, Kind::FillBuffer);
        assert_eq!(c.fill_value, 0xAB);
        assert_eq!(c.range_length, 8);
    }

    /// The pattern fill is the byte fill with a wider last field, and this
    /// reads it as one.
    ///
    /// The pattern is asymmetric so a decoder that read it big-endian, or read
    /// only its low byte into `fill_value`, fails rather than producing a
    /// plausible number.
    #[test]
    fn fill_buffer_pattern4() {
        let total = wire::FILL_BUFFER_PATTERN4_TOTAL_LEN;
        let mut v = hdr(OP_FILL_BUFFER_PATTERN4, total);
        st32(&mut v[HEADER_LEN..], 9);
        st64(&mut v[HEADER_LEN + 4..], 0x3300);
        st64(&mut v[HEADER_LEN + 12..], 0x4400);
        st32(&mut v[HEADER_LEN + 20..], 0x89ab_cdef);
        let c = decode(&v).unwrap();
        assert_eq!(c.kind, Kind::FillBufferPattern4);
        assert_eq!(c.buffer, 9);
        assert_eq!(c.range_location, 0x3300);
        assert_eq!(c.range_length, 0x4400);
        assert_eq!(c.fill_pattern, 0x89ab_cdef);
        // The two fills are distinct kinds precisely so this cannot happen: an
        // executor reading `fill_value` off a pattern record would write one
        // plausible byte instead of failing.
        assert_eq!(c.fill_value, 0);
    }

    /// The two texture fills share every field up to their tail, and the tail
    /// is what `fill_source` names.
    ///
    /// The region is checked component by component with no two values equal,
    /// because the wire stores the size **before** the origin — reversing
    /// `MTLRegion` — and a decoder that took them the other way round would
    /// read back a region that is the right shape and in the wrong place.
    #[test]
    fn both_texture_fills_decode_their_region_size_before_origin() {
        let region = |v: &mut [u8]| {
            let p = HEADER_LEN;
            st32(&mut v[p..], 4242);
            st16(&mut v[p + 4..], 3);
            st16(&mut v[p + 6..], 5);
            st64(&mut v[p + 8..], 0x44);
            st64(&mut v[p + 16..], 0x55);
            st64(&mut v[p + 24..], 0x66);
            st64(&mut v[p + 32..], 0x11);
            st64(&mut v[p + 40..], 0x22);
            st64(&mut v[p + 48..], 0x33);
        };
        let expect_shared = |c: &Command| {
            assert_eq!(c.kind, Kind::FillTexture);
            assert_eq!(c.texture, 4242);
            assert_eq!(c.level, 3);
            assert_eq!(c.slice, 5);
            assert_eq!(
                c.fill_size,
                Size {
                    width: 0x44,
                    height: 0x55,
                    depth: 0x66
                }
            );
            assert_eq!(
                c.fill_origin,
                Point {
                    x: 0x11,
                    y: 0x22,
                    z: 0x33
                }
            );
        };

        let mut v = hdr(OP_FILL_TEXTURE_COLOR, wire::FILL_TEXTURE_COLOR_TOTAL_LEN);
        region(&mut v);
        for (i, bits) in [0.25f64, 0.5, 0.75, 1.0].iter().enumerate() {
            st64(&mut v[HEADER_LEN + 56 + i * 8..], bits.to_bits());
        }
        st16(&mut v[HEADER_LEN + 88..], 80);
        let c = decode(&v).unwrap();
        expect_shared(&c);
        assert_eq!(c.fill_source, FillSource::Color);
        assert_eq!(
            c.fill_color_raw,
            [
                0.25f64.to_bits(),
                0.5f64.to_bits(),
                0.75f64.to_bits(),
                1.0f64.to_bits()
            ],
            "the four colour components must keep their order"
        );
        assert_eq!(c.fill_pixel_format, 80);
        assert_eq!(c.fill_bytes_ref, 0);

        let mut v = hdr(OP_FILL_TEXTURE_BYTES, wire::FILL_TEXTURE_BYTES_TOTAL_LEN);
        region(&mut v);
        st32(&mut v[HEADER_LEN + 56..], 8181);
        st64(&mut v[HEADER_LEN + 60..], 0x9999);
        st64(&mut v[HEADER_LEN + 68..], 8);
        let c = decode(&v).unwrap();
        expect_shared(&c);
        assert_eq!(c.fill_source, FillSource::Bytes);
        // The pattern is staged, not inline: the record names a buffer and an
        // offset and carries no pixel data.
        assert_eq!(c.fill_bytes_ref, 8181);
        assert_eq!(c.fill_bytes_offset, 0x9999);
        assert_eq!(c.fill_bytes_length, 8);
        assert_eq!(c.fill_color_raw, [0; 4]);
    }

    /// Both compressed-texture invalidates reach one kind, and the `slice:level:`
    /// form is the only one that carries a subresource.
    ///
    /// These share their wire shapes with `generateMipmapsForTexture:` and
    /// `synchronizeTexture:slice:level:`, whose arms end in the executor's
    /// unconditional no-op. Landing there would be correct behaviour and an
    /// unusable measurement — "this workload issues no compressed-texture
    /// invalidates" would be unprovable — so the kind is separate.
    #[test]
    fn both_compressed_texture_invalidates_are_their_own_kind() {
        let mut v = hdr(OP_INVALIDATE_COMPRESSED_TEXTURE, wire::REF_TOTAL_LEN);
        st32(&mut v[HEADER_LEN..], 4242);
        let c = decode(&v).unwrap();
        assert_eq!(c.kind, Kind::InvalidateCompressedTexture);
        assert_eq!(c.texture, 4242);
        assert_eq!((c.slice, c.level), (0, 0));

        let mut v = hdr(
            OP_INVALIDATE_COMPRESSED_TEXTURE_SLICE_LEVEL,
            wire::REF_SLICE_LEVEL_TOTAL_LEN,
        );
        st32(&mut v[HEADER_LEN..], 4242);
        st16(&mut v[HEADER_LEN + 4..], 3);
        st16(&mut v[HEADER_LEN + 6..], 5);
        let c = decode(&v).unwrap();
        assert_eq!(c.kind, Kind::InvalidateCompressedTexture);
        assert_eq!(c.texture, 4242);
        assert_eq!(c.slice, 3);
        assert_eq!(c.level, 5);
        // Not folded into `Kind::Image`, which the same wire shape reaches for
        // three other opcodes. `opcode` is what tells the two forms apart,
        // because "slice 0, level 0" and "every slice, every level" read alike.
        assert_eq!(c.opcode, OP_INVALIDATE_COMPRESSED_TEXTURE_SLICE_LEVEL);
    }

    /// Every one of the five refuses a record one byte short of its length.
    ///
    /// The length check is what stops a truncated record being read as a whole
    /// one, and each of these lengths is a constant from the wire crate rather
    /// than a number restated here — so a serializer that changed one fails
    /// this rather than decoding into the wrong fields.
    #[test]
    fn each_spi_record_refuses_a_length_one_byte_short() {
        for (op, total) in [
            (
                OP_FILL_BUFFER_PATTERN4,
                wire::FILL_BUFFER_PATTERN4_TOTAL_LEN,
            ),
            (OP_FILL_TEXTURE_BYTES, wire::FILL_TEXTURE_BYTES_TOTAL_LEN),
            (OP_FILL_TEXTURE_COLOR, wire::FILL_TEXTURE_COLOR_TOTAL_LEN),
            (OP_INVALIDATE_COMPRESSED_TEXTURE, wire::REF_TOTAL_LEN),
            (
                OP_INVALIDATE_COMPRESSED_TEXTURE_SLICE_LEVEL,
                wire::REF_SLICE_LEVEL_TOTAL_LEN,
            ),
        ] {
            let v = hdr(op, total - 1);
            assert_eq!(
                decode(&v),
                Err(DecodeStatus::ErrShort),
                "op {op:#x} accepted a record one byte short of {total}"
            );
        }
    }

    #[test]
    fn parse_blit_options_aspects() {
        assert_eq!(parse_blit_options(false, 0), Ok(BlitAspect::Full));
        assert_eq!(parse_blit_options(true, 0), Ok(BlitAspect::Full));
        assert_eq!(
            parse_blit_options(true, MTL_BLIT_OPTION_DEPTH_FROM_DEPTH_STENCIL),
            Ok(BlitAspect::Depth)
        );
        assert_eq!(
            parse_blit_options(true, MTL_BLIT_OPTION_STENCIL_FROM_DEPTH_STENCIL),
            Ok(BlitAspect::Stencil)
        );
        // Both depth+stencil forbidden.
        assert!(parse_blit_options(
            true,
            MTL_BLIT_OPTION_DEPTH_FROM_DEPTH_STENCIL | MTL_BLIT_OPTION_STENCIL_FROM_DEPTH_STENCIL
        )
        .is_err());
        // PVRTC not on product path.
        assert!(parse_blit_options(true, MTL_BLIT_OPTION_ROW_LINEAR_PVRTC).is_err());
        // Unknown bits fail.
        assert!(parse_blit_options(true, 1 << 8).is_err());
    }

    /// `copyFromTexture:toBuffer:` reads two bytes of `options`, not four.
    ///
    /// It is the one copy record that narrows the field: the serializer wrote
    /// `04 00 AA AA` there against the oracle's poison, where the
    /// buffer-to-texture and region forms fill all four. The two bytes past it
    /// belong to no field, so on a guest's wire they hold whatever the ring
    /// last contained — which is what the poison stands in for here.
    ///
    /// Both halves matter. A plain copy writes zero into the two live bytes and
    /// must still parse as `Full`; a depth-aspect copy must reach `Depth`
    /// rather than being refused for bits that were never written.
    #[test]
    fn a_texture_to_buffer_copy_reads_no_byte_past_its_options() {
        for (written, want) in [
            (0u16, BlitAspect::Full),
            (MTL_BLIT_OPTION_DEPTH_FROM_DEPTH_STENCIL as u16, BlitAspect::Depth),
        ] {
            let mut v = hdr(OP_COPY_TEXTURE_TO_BUFFER, CTB_LEN as u32);
            // Everything past the options field is ring content, not record.
            for b in v.iter_mut().skip(HEADER_LEN + CTB_OPTIONS + 2) {
                *b = 0xAA;
            }
            st16(&mut v[HEADER_LEN + CTB_OPTIONS..], written);

            let c = decode(&v).expect("a well-formed copy must decode");
            assert_eq!(
                c.options, written as u32,
                "options picked up a byte the serializer never wrote"
            );
            assert_eq!(parse_blit_options(c.has_options, c.options), Ok(want));
        }

        // The sibling that really is four bytes wide keeps reading four, so
        // this is a per-record narrowing rather than a family rule.
        let mut v = hdr(OP_COPY_BUFFER_TO_TEXTURE, CBT_LEN as u32);
        st32(&mut v[HEADER_LEN + CBT_OPTIONS..], 0x0001_0000);
        assert_eq!(decode(&v).unwrap().options, 0x0001_0000);
    }

    /// A record we decline and a number we do not recognise are different
    /// refusals, and the three ICB opcodes are the first kind.
    ///
    /// The opcodes come from `reims-vgpu-wire`, where fixtures pin them against
    /// bytes the serializer produced. Asserting them against that crate rather
    /// than against literals is what keeps this from drifting back into a claim
    /// that Apple never emits them.
    #[test]
    fn an_icb_blit_record_is_read_rather_than_refused_whole() {
        use crate::contract::endian::st64;
        use reims_vgpu_wire::ops::blit as wire;

        for (op, wire_op) in [
            (OP_COPY_ICB, wire::OPCODE_COPY_ICB),
            (OP_OPTIMIZE_ICB, wire::OPCODE_OPTIMIZE_ICB),
            (OP_RESET_ICB, wire::OPCODE_RESET_ICB),
        ] {
            assert_eq!(op, wire_op, "the serializer writes a different opcode");
        }
        // Still inside this encoder's opcode space, which is why "a foreign
        // encoder's number arrived here" was never the explanation.
        for op in [OP_COPY_ICB, OP_OPTIMIZE_ICB, OP_RESET_ICB] {
            assert!(
                (OP_COPY_BUFFER_TO_TEXTURE..=OP_COPY_TEXTURE_TO_TEXTURE_SLICE_LEVEL).contains(&op),
                "op {op:#x} is outside the blit opcode space"
            );
        }

        // The two range forms: a ref then two `u64` in declaration order. The
        // range values differ from each other so a record that carried the same
        // number twice could not read back correct.
        for op in [OP_OPTIMIZE_ICB, OP_RESET_ICB] {
            let total = wire::ICB_RANGE_TOTAL_LEN;
            let mut v = hdr(op, total);
            st32(&mut v[HEADER_LEN..], 6161);
            st64(&mut v[HEADER_LEN + 4..], 0x3300);
            st64(&mut v[HEADER_LEN + 12..], 0x4400);
            let c = decode(&v).unwrap_or_else(|e| panic!("op {op:#x}: {e:?}"));
            assert_eq!(c.kind, Kind::IcbRange, "op {op:#x}");
            assert_eq!(c.resource_kind, RefKind::IndirectCommandBuffer, "op {op:#x}");
            assert_eq!(c.resource, 6161, "op {op:#x}");
            assert_eq!(
                (c.range_location, c.range_length),
                (0x3300, 0x4400),
                "op {op:#x}: the range is crossed"
            );
            assert_eq!(
                decode(&hdr(op, total - 4)).unwrap_err(),
                DecodeStatus::ErrShort,
                "op {op:#x} accepted a record four bytes short"
            );
        }

        // The copy: both refs lead, and the destination index is a `u64` beside
        // the source range rather than a `u16` at the tail — it counts commands,
        // not subresources. Four distinct values, so no pair can be swapped
        // without the assertion seeing it.
        let total = wire::COPY_ICB_TOTAL_LEN;
        let mut v = hdr(OP_COPY_ICB, total);
        st32(&mut v[HEADER_LEN..], 7171);
        st32(&mut v[HEADER_LEN + 4..], 7272);
        st64(&mut v[HEADER_LEN + 8..], 0x1100);
        st64(&mut v[HEADER_LEN + 16..], 0x2200);
        st64(&mut v[HEADER_LEN + 24..], 0x3300);
        let c = decode(&v).expect("copy icb");
        assert_eq!(c.kind, Kind::IcbCopy);
        assert_eq!((c.source, c.destination), (7171, 7272));
        assert_eq!(c.source_kind, RefKind::IndirectCommandBuffer);
        assert_eq!(c.destination_kind, RefKind::IndirectCommandBuffer);
        assert_eq!((c.range_location, c.range_length), (0x1100, 0x2200));
        assert_eq!(
            c.destination_index, 0x3300,
            "the destination index read one of the range words"
        );
        assert_eq!(
            decode(&hdr(OP_COPY_ICB, total - 4)).unwrap_err(),
            DecodeStatus::ErrShort
        );

        assert_eq!(
            decode(&hdr(0x999, 16)).unwrap_err(),
            DecodeStatus::ErrUnknownOpcode
        );
    }

    #[test]
    fn fence_and_resource() {
        let mut v = hdr(OP_UPDATE_FENCE, FENCE_LEN as u32);
        st32(&mut v[8..], 3);
        assert_eq!(decode(&v).unwrap().fence, 3);
        let mut v = hdr(OP_GENERATE_MIPMAPS, RESOURCE_LEN as u32);
        st32(&mut v[8..], 5);
        let c = decode(&v).unwrap();
        assert_eq!(c.resource, 5);
        assert_eq!(c.resource_kind, RefKind::Texture);
    }

    #[test]
    fn texture_to_texture_options_len() {
        let v = hdr(OP_COPY_TEXTURE_TO_TEXTURE_OPTIONS, CTT_OPTIONS_LEN as u32);
        // zeros decode fine
        let c = decode(&v).unwrap();
        assert!(c.has_options);
        let bad = hdr(OP_COPY_TEXTURE_TO_TEXTURE_OPTIONS, CTT_LEN as u32);
        assert_eq!(decode(&bad).unwrap_err(), DecodeStatus::ErrShort);
    }

    /// Every blit opcode Apple's serializer emits has a constant here, and this
    /// module names no opcode Apple does not emit.
    ///
    /// The blit arm is the one where both directions of this have already cost
    /// guest work. Three records the serializer emits were called records it
    /// refuses, and `0x13f`–`0x143` — the six `BlitEncoderSPI` selectors — all
    /// reached `blit_decode_unknown_opcode` until the capability was forced,
    /// because `ops::blit`'s own doc said the run stopped at `0x13e` and no
    /// capture had contradicted it. A run's upper end is a statement about what
    /// has been driven, so nothing here may take one on faith.
    ///
    /// Unlike the compute sibling this is plain set equality: the blit encoder
    /// has no inherited numbers and no refused-but-decoded arms, so a mismatch
    /// in either direction is a defect rather than an exception to record.
    #[test]
    fn the_blit_opcode_table_is_exactly_apples_blit_manifest() {
        let device: &[(u32, &str)] = &[
            (OP_COPY_BUFFER_TO_TEXTURE, "OP_COPY_BUFFER_TO_TEXTURE"),
            (OP_COPY_BUFFER_TO_BUFFER, "OP_COPY_BUFFER_TO_BUFFER"),
            (OP_COPY_TEXTURE_TO_BUFFER, "OP_COPY_TEXTURE_TO_BUFFER"),
            (OP_COPY_TEXTURE_TO_TEXTURE, "OP_COPY_TEXTURE_TO_TEXTURE"),
            (
                OP_COPY_TEXTURE_TO_TEXTURE_OPTIONS,
                "OP_COPY_TEXTURE_TO_TEXTURE_OPTIONS",
            ),
            (OP_COPY_ICB, "OP_COPY_ICB"),
            (OP_FILL_BUFFER, "OP_FILL_BUFFER"),
            (OP_GENERATE_MIPMAPS, "OP_GENERATE_MIPMAPS"),
            (OP_OPTIMIZE_CPU, "OP_OPTIMIZE_CPU"),
            (OP_OPTIMIZE_GPU, "OP_OPTIMIZE_GPU"),
            (OP_OPTIMIZE_IMAGE_CPU, "OP_OPTIMIZE_IMAGE_CPU"),
            (OP_OPTIMIZE_IMAGE_GPU, "OP_OPTIMIZE_IMAGE_GPU"),
            (OP_OPTIMIZE_ICB, "OP_OPTIMIZE_ICB"),
            (OP_RESET_ICB, "OP_RESET_ICB"),
            (OP_SYNCHRONIZE_RESOURCE, "OP_SYNCHRONIZE_RESOURCE"),
            (OP_SYNCHRONIZE_TEXTURE_IMAGE, "OP_SYNCHRONIZE_TEXTURE_IMAGE"),
            (OP_UPDATE_FENCE, "OP_UPDATE_FENCE"),
            (OP_WAIT_FENCE, "OP_WAIT_FENCE"),
            (
                OP_COPY_TEXTURE_TO_TEXTURE_SLICE_LEVEL,
                "OP_COPY_TEXTURE_TO_TEXTURE_SLICE_LEVEL",
            ),
            (OP_FILL_BUFFER_PATTERN4, "OP_FILL_BUFFER_PATTERN4"),
            (OP_FILL_TEXTURE_BYTES, "OP_FILL_TEXTURE_BYTES"),
            (OP_FILL_TEXTURE_COLOR, "OP_FILL_TEXTURE_COLOR"),
            (
                OP_INVALIDATE_COMPRESSED_TEXTURE,
                "OP_INVALIDATE_COMPRESSED_TEXTURE",
            ),
            (
                OP_INVALIDATE_COMPRESSED_TEXTURE_SLICE_LEVEL,
                "OP_INVALIDATE_COMPRESSED_TEXTURE_SLICE_LEVEL",
            ),
        ];

        let mut apple: Vec<u32> = reims_vgpu_wire::manifest::MANIFEST
            .iter()
            .filter(|e| e.class == "PGSerializerBlitCommandEncoder")
            .flat_map(|e| e.opcodes.iter().copied())
            .collect();
        apple.sort_unstable();
        apple.dedup();

        for (op, name) in device {
            assert!(
                apple.contains(op),
                "{name} = {op:#x} is not an opcode Apple's blit manifest lists, \
                 so no capture supports it"
            );
        }
        for op in &apple {
            assert!(
                device.iter().any(|(d, _)| d == op),
                "Apple's serializer emits blit opcode {op:#x} and this module \
                 names no constant for it, so every copy or fill carrying it is \
                 declined as an opcode Apple does not write"
            );
        }
        assert_eq!(
            device.len(),
            apple.len(),
            "the roster has a duplicate entry"
        );
    }

    #[test]
    fn property_fuzz_opcodes() {
        for op in 0x120u32..0x150 {
            let mut v = hdr(op, 0x80);
            let _ = decode(&v);
            // also exact common lengths
            for len in [0x0c, 0x10, 0x1c, 0x20, 0x28, 0x60, 0x64] {
                v = hdr(op, len);
                let _ = decode(&v);
            }
        }
    }
}
