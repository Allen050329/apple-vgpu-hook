//! Blit command decoder (port of `host/utils/reims-vgpu-blit-decode`).

use crate::contract::endian::{ld16, ld32, ld64};

pub const HEADER_LEN: usize = 8;
pub const REJECTED_131: u32 = 0x131;
pub const REJECTED_138: u32 = 0x138;
pub const REJECTED_139: u32 = 0x139;

pub const OP_COPY_BUFFER_TO_TEXTURE: u32 = 0x12c;
pub const OP_COPY_BUFFER_TO_BUFFER: u32 = 0x12d;
pub const OP_COPY_TEXTURE_TO_BUFFER: u32 = 0x12e;
pub const OP_COPY_TEXTURE_TO_TEXTURE: u32 = 0x12f;
pub const OP_COPY_TEXTURE_TO_TEXTURE_OPTIONS: u32 = 0x130;
pub const OP_FILL_BUFFER: u32 = 0x132;
pub const OP_GENERATE_MIPMAPS: u32 = 0x133;
pub const OP_OPTIMIZE_CPU: u32 = 0x134;
pub const OP_OPTIMIZE_GPU: u32 = 0x135;
pub const OP_OPTIMIZE_IMAGE_CPU: u32 = 0x136;
pub const OP_OPTIMIZE_IMAGE_GPU: u32 = 0x137;
pub const OP_SYNCHRONIZE_RESOURCE: u32 = 0x13a;
pub const OP_SYNCHRONIZE_TEXTURE_IMAGE: u32 = 0x13b;
pub const OP_UPDATE_FENCE: u32 = 0x13c;
pub const OP_WAIT_FENCE: u32 = 0x13d;
pub const OP_COPY_TEXTURE_TO_TEXTURE_SLICE_LEVEL: u32 = 0x13e;

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

const CBT_SRC_OFF: usize = 0x08;
const CBT_SRC_BPR: usize = 0x10;
const CBT_SRC_BPI: usize = 0x18;
const CBT_SRC_SIZE: usize = 0x20;
const CBT_DST_ORIGIN: usize = 0x38;
const CBT_DST_SLICE: usize = 0x50;
const CBT_DST_LEVEL: usize = 0x52;
const CBT_OPTIONS: usize = 0x54;
const CBT_LEN: usize = 0x60;

const CBB_SRC_OFF: usize = 0x08;
const CBB_DST_OFF: usize = 0x10;
const CBB_SIZE: usize = 0x18;
const CBB_LEN: usize = 0x28;

const CTB_SRC_ORIGIN: usize = 0x08;
const CTB_SRC_SIZE: usize = 0x20;
const CTB_DST_OFF: usize = 0x38;
const CTB_DST_BPR: usize = 0x40;
const CTB_DST_BPI: usize = 0x48;
const CTB_SRC_SLICE: usize = 0x50;
const CTB_SRC_LEVEL: usize = 0x52;
const CTB_OPTIONS: usize = 0x54;
const CTB_LEN: usize = 0x60;

const CTT_SRC_ORIGIN: usize = 0x08;
const CTT_SRC_SIZE: usize = 0x20;
const CTT_DST_ORIGIN: usize = 0x38;
const CTT_SRC_SLICE: usize = 0x50;
const CTT_SRC_LEVEL: usize = 0x52;
const CTT_DST_SLICE: usize = 0x54;
const CTT_DST_LEVEL: usize = 0x56;
const CTT_OPTIONS: usize = 0x58;
const CTT_LEN: usize = 0x60;
const CTT_OPTIONS_LEN: usize = 0x64;

const CTTSL_SRC_SLICE: usize = 0x08;
const CTTSL_SRC_LEVEL: usize = 0x0a;
const CTTSL_DST_SLICE: usize = 0x0c;
const CTTSL_DST_LEVEL: usize = 0x0e;
const CTTSL_SLICE_COUNT: usize = 0x10;
const CTTSL_LEVEL_COUNT: usize = 0x12;
const CTTSL_LEN: usize = 0x1c;

const FILL_REF: usize = 0;
const FILL_RANGE_LOC: usize = 0x04;
const FILL_RANGE_LEN: usize = 0x0c;
const FILL_VALUE: usize = 0x14;
const FILL_LEN: usize = 0x20;

const RESOURCE_REF: usize = 0;
const RESOURCE_LEN: usize = 0x0c;
const IMAGE_TEXTURE: usize = 0;
const IMAGE_SLICE: usize = 0x04;
const IMAGE_LEVEL: usize = 0x06;
const IMAGE_LEN: usize = 0x10;
const FENCE_REF: usize = 0;
const FENCE_LEN: usize = 0x0c;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeStatus {
    Ok = 0,
    ErrArgs,
    ErrShort,
    ErrUnknownOpcode,
    ErrUnsupportedOpcode,
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
            Self::Ok => return None,
            Self::ErrArgs => "blit_decode_args",
            Self::ErrShort => "blit_decode_short",
            Self::ErrUnknownOpcode => "blit_decode_unknown_opcode",
            Self::ErrUnsupportedOpcode => "blit_decode_unsupported_opcode",
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum RefKind {
    #[default]
    None = 0,
    Buffer,
    Texture,
    Resource,
    Fence,
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
    pub fill_value: u8,
    pub texture: u32,
    pub slice: u16,
    pub level: u16,
    pub fence: u32,
}

pub fn opcode_supported(opcode: u32) -> bool {
    matches!(
        opcode,
        OP_COPY_BUFFER_TO_TEXTURE
            | OP_COPY_BUFFER_TO_BUFFER
            | OP_COPY_TEXTURE_TO_BUFFER
            | OP_COPY_TEXTURE_TO_TEXTURE
            | OP_COPY_TEXTURE_TO_TEXTURE_OPTIONS
            | OP_FILL_BUFFER
            | OP_GENERATE_MIPMAPS
            | OP_OPTIMIZE_CPU
            | OP_OPTIMIZE_GPU
            | OP_OPTIMIZE_IMAGE_CPU
            | OP_OPTIMIZE_IMAGE_GPU
            | OP_SYNCHRONIZE_RESOURCE
            | OP_SYNCHRONIZE_TEXTURE_IMAGE
            | OP_UPDATE_FENCE
            | OP_WAIT_FENCE
            | OP_COPY_TEXTURE_TO_TEXTURE_SLICE_LEVEL
    )
}

pub fn opcode_apple_rejected(opcode: u32) -> bool {
    matches!(opcode, REJECTED_131 | REJECTED_138 | REJECTED_139)
}

pub fn opcode_name(opcode: u32) -> &'static str {
    match opcode {
        OP_COPY_BUFFER_TO_TEXTURE => "decodeCopyFromBufferToTexture",
        OP_COPY_BUFFER_TO_BUFFER => "decodeCopyFromBufferToBuffer",
        OP_COPY_TEXTURE_TO_BUFFER => "decodeCopyFromTextureToBuffer",
        OP_COPY_TEXTURE_TO_TEXTURE => "decodeCopyFromTextureToTexture",
        OP_COPY_TEXTURE_TO_TEXTURE_OPTIONS => "decodeCopyFromTextureToTextureWithOptions",
        OP_FILL_BUFFER => "decodeFillBuffer",
        OP_GENERATE_MIPMAPS => "decodeGenerateMipmaps",
        OP_OPTIMIZE_CPU | OP_OPTIMIZE_GPU => "decodeOptimize",
        OP_OPTIMIZE_IMAGE_CPU | OP_OPTIMIZE_IMAGE_GPU => "decodeOptimizeImage",
        OP_SYNCHRONIZE_RESOURCE => "decodeSynchronizeResource",
        OP_SYNCHRONIZE_TEXTURE_IMAGE => "decodeSynchronizeTextureImage",
        OP_UPDATE_FENCE => "decodeBlitUpdateFence",
        OP_WAIT_FENCE => "decodeBlitWaitForFence",
        OP_COPY_TEXTURE_TO_TEXTURE_SLICE_LEVEL => "decodeCopyFromTextureToTextureWithNumSliceLevel",
        _ if opcode_apple_rejected(opcode) => "AppleException",
        _ => "unknown",
    }
}

pub fn kind_name(kind: Kind) -> &'static str {
    match kind {
        Kind::Copy => "copy",
        Kind::FillBuffer => "fillBuffer",
        Kind::Resource => "resource",
        Kind::Image => "image",
        Kind::Fence => "fence",
        Kind::Unknown => "unknown",
    }
}

pub fn copy_kind_name(kind: CopyKind) -> &'static str {
    match kind {
        CopyKind::BufferToTexture => "copyFromBuffer:toTexture",
        CopyKind::BufferToBuffer => "copyFromBuffer:toBuffer",
        CopyKind::TextureToBuffer => "copyFromTexture:toBuffer",
        CopyKind::TextureToTexture => "copyFromTexture:toTexture",
        CopyKind::TextureToTextureSliceLevel => "copyFromTexture:toTexture:sliceLevel",
        CopyKind::None => "none",
    }
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
    if opcode_apple_rejected(opcode) {
        return Err(DecodeStatus::ErrUnsupportedOpcode);
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
            out.options = ld32(&payload[CTB_OPTIONS..]);
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
    fn every_blit_decode_failure_but_ok_names_its_own_check() {
        use crate::observe::{Decline, Refusal};
        const ALL: &[DecodeStatus] = &[
            DecodeStatus::Ok,
            DecodeStatus::ErrArgs,
            DecodeStatus::ErrShort,
            DecodeStatus::ErrUnknownOpcode,
            DecodeStatus::ErrUnsupportedOpcode,
        ];
        assert_eq!(DecodeStatus::Ok.refusal(), None);
        let mut slugs: Vec<&str> = ALL.iter().filter_map(|s| s.refusal()).collect();
        assert_eq!(slugs.len(), 4);
        slugs.sort_unstable();
        slugs.dedup();
        assert_eq!(slugs.len(), 4, "two blit decode checks share a slug");
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

    #[test]
    fn rejected_and_unknown() {
        assert_eq!(
            decode(&hdr(REJECTED_131, 16)).unwrap_err(),
            DecodeStatus::ErrUnsupportedOpcode
        );
        assert_eq!(
            decode(&hdr(0x999, 16)).unwrap_err(),
            DecodeStatus::ErrUnknownOpcode
        );
        assert!(opcode_supported(OP_FILL_BUFFER));
        assert!(!opcode_supported(REJECTED_131));
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
        let _ = st16;
    }
}
