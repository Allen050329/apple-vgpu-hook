//! Render command decoder (port of `host/utils/reims-vgpu-render-decode`).
//!
//! Full opcode matrix is preserved for supported/rejected classification.
//! Per-opcode payload layouts for the highest-traffic families (set pipeline,
//! buffer/texture binds, draw, viewport/scissor, barriers, fences) are decoded;
//! remaining accepted opcodes are recognized and returned as typed kinds with
//! raw length validation where the contract specifies fixed sizes.

use crate::contract::endian::{ld16, ld32, ld64}; // ld16: render-pass load/store actions

pub const HEADER_LEN: usize = 8;

// Contract opcodes from reims_vgpu_render_decode.h / reims_vgpu_render_format.h.
/// Wide `drawPrimitives:vertexStart:vertexCount:` — `alloc(0, 0x14)`. Its payload
/// layout is unverified, so it is declined by name rather than decoded.
pub const OP_DRAW_WIDE: u32 = 0x00;
/// Compact `drawPrimitives:vertexStart:vertexCount:` — `alloc(1, 8)`, wire sz 0x10.
pub const OP_DRAW: u32 = 0x01;
/// Wire record length and payload length of the compact form, from the encoder's
/// own `alloc(1, 8)`. Checked exactly: a `0x1` record of any other size is not a
/// form this contract knows.
pub const DRAW_COMPACT_PAYLOAD_LEN: usize = 8;
pub const DRAW_COMPACT_CMD_LEN: usize = HEADER_LEN + DRAW_COMPACT_PAYLOAD_LEN;
/// Compact `drawPrimitives:vertexStart:vertexCount:instanceCount:` (wire sz 0x10).
pub const OP_DRAW_INST_COMPACT: u32 = 0x03;
/// Wide drawIndexedPrimitives form (wire size 0x20).
pub const OP_DRAW_INDEXED_WIDE: u32 = 0x06;
pub const OP_DRAW_INDEXED: u32 = 0x07;
/// Last opcode in the accepted render-command enum window.
pub const OP_ACCEPTED_LAST: u32 = 0x98;
pub const OP_DRAW_INDEXED_INST: u32 = 0x09;
pub const OP_EXECUTE_COMMANDS_INDIRECT: u32 = 0x14;
pub const OP_EXECUTE_COMMANDS_RANGE: u32 = 0x15;
/// Full record lengths (header + payload) from reims_vgpu_render_format.h.
pub const EXECUTE_INDIRECT_CMD_LEN: usize = 0x18;
pub const EXECUTE_RANGE_CMD_LEN: usize = 0x1c;
pub const EXECUTE_INDIRECT_COMMAND_BUFFER_REF: usize = 0;
pub const EXECUTE_INDIRECT_BUFFER_REF: usize = 4;
pub const EXECUTE_INDIRECT_BUFFER_OFFSET: usize = 8;
pub const EXECUTE_RANGE_COMMAND_BUFFER_REF: usize = 0;
pub const EXECUTE_RANGE_LOCATION: usize = 4;
pub const EXECUTE_RANGE_LENGTH: usize = 0x0c;
pub const OP_RESOURCE_BARRIER: u32 = 0x16;
pub const OP_MEMORY_BARRIER: u32 = 0x17;
pub const OP_UPDATE_FENCE: u32 = 0x18;
pub const OP_WAIT_FENCE: u32 = 0x19;
pub const OP_RENDER_PASS: u32 = 0x1a;

/// Live render-pass attachment layout (reims_vgpu_render_format.h).
pub const PASS_DEPTH_ATTACH_OFF: usize = 0x00;
pub const PASS_STENCIL_ATTACH_OFF: usize = 0x28;
pub const PASS_DEPTH_STENCIL_ATTACH_STRIDE: usize = 0x28;
pub const PASS_COLOR_ATTACH_OFF: usize = 0x4c;
pub const PASS_COLOR_ATTACH_STRIDE: usize = 0x3c;
pub const PASS_ATTACH_TEXREF: usize = 0x00;
pub const PASS_ATTACH_RESOLVEREF: usize = 0x04;
pub const PASS_ATTACH_LEVEL: usize = 0x08;
pub const PASS_ATTACH_LOAD_ACTION: usize = 0x14;
pub const PASS_ATTACH_STORE_ACTION: usize = 0x16;
pub const PASS_ATTACH_CLEAR_COLOR: usize = 0x1c;
pub const PASS_DEPTH_ATTACH_CLEAR_DEPTH: usize = 0x1c;
pub const PASS_STENCIL_ATTACH_CLEAR_STENCIL: usize = 0x1c;
pub const PASS_MAX_COLOR_ATTACHMENTS: usize = 8;
pub const PASS_LOAD_ACTION_DONT_CARE: u16 = 0;
pub const PASS_LOAD_ACTION_LOAD: u16 = 1;
pub const PASS_LOAD_ACTION_CLEAR: u16 = 2;
pub const PASS_STORE_ACTION_DONT_CARE: u16 = 0;
pub const PASS_STORE_ACTION_STORE: u16 = 1;
pub const PASS_MIN_PAYLOAD: usize = PASS_COLOR_ATTACH_OFF + PASS_COLOR_ATTACH_STRIDE;
pub const OP_SET_BLEND_COLOR: u32 = 0x65;
pub const OP_SET_DEPTH_STENCIL: u32 = 0x68;
pub const OP_SET_CULL_MODE: u32 = 0x6b;
pub const OP_SET_DEPTH_BIAS: u32 = 0x6c;
pub const OP_SET_FRAGMENT_BUFFER: u32 = 0x6e;
pub const OP_SET_FRAGMENT_BUFFER_OFFSET: u32 = 0x6f;
pub const OP_SET_FRAGMENT_SAMPLER: u32 = 0x70;
pub const OP_SET_FRAGMENT_TEXTURE: u32 = 0x72;
pub const OP_SET_FRONT_FACING: u32 = 0x73;
pub const OP_SET_PIPELINE: u32 = 0x74;
pub const OP_SET_SCISSOR: u32 = 0x75;
pub const OP_SET_STENCIL_REF: u32 = 0x77;
pub const OP_SET_VERTEX_BUFFER: u32 = 0x7d;
pub const OP_SET_VERTEX_BUFFER_OFFSET: u32 = 0x7e;
pub const OP_SET_VERTEX_SAMPLER: u32 = 0x7f;
pub const OP_SET_VERTEX_TEXTURE: u32 = 0x81;
pub const OP_SET_VIEWPORT: u32 = 0x82;
pub const OP_USE_HEAP: u32 = 0x86;
pub const OP_USE_RESOURCE: u32 = 0x87;

/// Multi-entry bind header (reims_vgpu_render_format.h): first:u32 @0, count:u32 @4.
pub const BIND_FIRST: usize = 0;
pub const BIND_COUNT: usize = 4;
pub const BIND_ENTRIES: usize = 8;
pub const BUFFER_BIND_ENTRY_SIZE: usize = 12;
pub const BUFFER_BIND_ENTRY_REF: usize = 0;
pub const BUFFER_BIND_ENTRY_OFFSET: usize = 4;
pub const REF_BIND_ENTRY_SIZE: usize = 4;
pub const MAX_BIND_ENTRIES: u32 = 32;
/// set*BufferOffset: index:u32 @0, offset:u64 @4 (payload 12; full cmd 0x14).
pub const BUFFER_OFFSET_INDEX: usize = 0;
pub const BUFFER_OFFSET_VALUE: usize = 4;
pub const BUFFER_OFFSET_PAYLOAD_LEN: usize = 12;
/// setScissorRect: four u64 fields (archive REIMS_VGPU_RENDER_SCISSOR_*).
pub const SCISSOR_X: usize = 0;
pub const SCISSOR_Y: usize = 8;
pub const SCISSOR_WIDTH: usize = 0x10;
pub const SCISSOR_HEIGHT: usize = 0x18;
pub const SCISSOR_PAYLOAD_LEN: usize = 0x20;

// Supported window is the full C-accepted encoder range 0x00..=0x98 minus rejected.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeStatus {
    Ok,
    ErrArgs,
    ErrShort,
    ErrUnknownOpcode,
    ErrUnsupportedOpcode,
    ErrBadLength,
}

impl crate::observe::Refusal for DecodeStatus {
    /// Slugs carry a `render_decode_` prefix: seven modules under
    /// `runtime/decode/` define a type called `DecodeStatus`, and five of them
    /// have an `ErrShort` that means a different read. Without the prefix the
    /// crate-wide uniqueness gate could not tell the render decoder's refusals
    /// from any other's.
    fn refusal(&self) -> Option<&'static str> {
        Some(match self {
            Self::Ok => return None,
            Self::ErrArgs => "render_decode_args",
            Self::ErrShort => "render_decode_short",
            Self::ErrUnknownOpcode => "render_decode_unknown_opcode",
            Self::ErrUnsupportedOpcode => "render_decode_unsupported_opcode",
            Self::ErrBadLength => "render_decode_bad_length",
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Kind {
    #[default]
    Unknown,
    SetPipeline,
    SetBuffer,
    /// setVertexBufferOffset / setFragmentBufferOffset (0x7e / 0x6f).
    SetBufferOffset,
    SetTexture,
    SetSampler,
    Draw,
    SetViewport,
    SetScissor,
    SetDepthStencil,
    SetBlendColor,
    SetCullMode,
    SetFrontFacing,
    SetDepthBias,
    SetStencilReference,
    Fence,
    Barrier,
    UseResource,
    UseHeap,
    ExecuteCommands,
    RenderPass,
    OtherAccepted,
}

/// One color attachment from a render-pass descriptor (0x1a).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ColorAttachment {
    pub present: bool,
    pub texture_ref: u32,
    pub resolve_texture_ref: u32,
    pub level: u32,
    pub load_action: u16,
    pub store_action: u16,
    /// MTLClearColor as RGBA doubles in [0,1].
    pub clear_color: [f64; 4],
}

/// Depth attachment from a render-pass descriptor (slot @0x00).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DepthAttachment {
    pub present: bool,
    pub texture_ref: u32,
    pub resolve_texture_ref: u32,
    pub level: u32,
    pub load_action: u16,
    pub store_action: u16,
    pub clear_depth: f64,
}

/// Stencil attachment from a render-pass descriptor (slot @0x28).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StencilAttachment {
    pub present: bool,
    pub texture_ref: u32,
    pub resolve_texture_ref: u32,
    pub level: u32,
    pub load_action: u16,
    pub store_action: u16,
    pub clear_stencil: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Stage {
    #[default]
    Unknown,
    Vertex,
    Fragment,
    Object,
    Mesh,
    Tile,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Command {
    pub opcode: u32,
    pub command_length: u32,
    pub kind: Kind,
    pub stage: Stage,
    pub pipeline_ref: u32,
    pub first: u32,
    pub count: u32,
    pub buffer_ref: u32,
    pub buffer_offset: u64,
    /// Multi-entry buffer binds: (object_ref, offset) for slots first..first+count.
    pub buffer_binds: Vec<(u32, u64)>,
    pub texture_ref: u32,
    /// Multi-entry texture/sampler refs for slots first..first+count.
    pub ref_binds: Vec<u32>,
    pub sampler_ref: u32,
    pub primitive_type: u32,
    pub vertex_start: u32,
    pub vertex_count: u32,
    pub instance_count: u32,
    pub index_count: u32,
    pub index_type: u32,
    pub index_buffer_ref: u32,
    pub index_buffer_offset: u64,
    pub viewport: [f64; 6],
    pub scissor_x: u32,
    pub scissor_y: u32,
    pub scissor_w: u32,
    pub scissor_h: u32,
    pub fence_ref: u32,
    pub resource_ref: u32,
    pub raw_payload_len: usize,
    /// Color attachment[0] when kind is RenderPass (boot clear path).
    pub color0: ColorAttachment,
    pub depth: DepthAttachment,
    pub stencil: StencilAttachment,
    /// setBlendColor RGBA floats (when kind is SetBlendColor).
    pub blend_color: [f32; 4],
    pub has_blend_color: bool,
    /// setCullMode
    pub cull_mode: u32,
    pub has_cull_mode: bool,
    /// setFrontFacingWinding
    pub front_facing: u32,
    pub has_front_facing: bool,
    /// setDepthBias (depthBias, slopeScale, clamp) as f32.
    pub depth_bias: [f32; 3],
    pub has_depth_bias: bool,
    /// setDepthStencilState object ref
    pub depth_stencil_ref: u32,
    /// setStencilReference front/back
    pub stencil_ref_front: u32,
    pub stencil_ref_back: u32,
    pub has_stencil_ref: bool,
    /// `0x14`/`0x15` executeCommandsInBuffer.
    pub indirect_command_buffer_ref: u32,
    /// `0x15` range form (unaligned after ICB ref).
    pub icb_range_location: u64,
    pub icb_range_length: u64,
    /// `0x14` indirect range buffer form.
    pub icb_args_buffer_ref: u32,
    pub icb_args_buffer_offset: u64,
    /// True when kind is ExecuteCommands with the range layout (`0x15`).
    pub icb_is_range: bool,
}

pub fn opcode_is_apple_rejected(opcode: u32) -> bool {
    // Outside the known accepted encoder surface (C matrix uses explicit accepted list).
    // Unknown high opcodes fail closed as unsupported when listed rejected in C tests.
    opcode > OP_ACCEPTED_LAST
}

pub fn opcode_supported(opcode: u32) -> bool {
    if opcode_is_apple_rejected(opcode) {
        return false;
    }
    // Full accepted window from reims_vgpu_render_decode.h enum range.
    opcode <= OP_ACCEPTED_LAST
}

/// Decode color attachment slot `index` from a render-pass payload.
pub fn decode_color_attachment(payload: &[u8], index: usize) -> ColorAttachment {
    let mut out = ColorAttachment::default();
    let base = PASS_COLOR_ATTACH_OFF + index * PASS_COLOR_ATTACH_STRIDE;
    if payload.len() < base + PASS_COLOR_ATTACH_STRIDE {
        return out;
    }
    let slot = &payload[base..base + PASS_COLOR_ATTACH_STRIDE];
    out.texture_ref = ld32(&slot[PASS_ATTACH_TEXREF..]);
    out.resolve_texture_ref = ld32(&slot[PASS_ATTACH_RESOLVEREF..]);
    out.level = ld32(&slot[PASS_ATTACH_LEVEL..]);
    out.load_action = ld16(&slot[PASS_ATTACH_LOAD_ACTION..]);
    out.store_action = ld16(&slot[PASS_ATTACH_STORE_ACTION..]);
    for i in 0..4 {
        let bits = ld64(&slot[PASS_ATTACH_CLEAR_COLOR + i * 8..]);
        out.clear_color[i] = f64::from_bits(bits);
    }
    out.present = out.texture_ref != 0;
    out
}

/// Decode the depth attachment (fixed slot @0).
pub fn decode_depth_attachment(payload: &[u8]) -> DepthAttachment {
    let mut out = DepthAttachment::default();
    if payload.len() < PASS_DEPTH_ATTACH_OFF + PASS_DEPTH_STENCIL_ATTACH_STRIDE {
        return out;
    }
    let slot =
        &payload[PASS_DEPTH_ATTACH_OFF..PASS_DEPTH_ATTACH_OFF + PASS_DEPTH_STENCIL_ATTACH_STRIDE];
    out.texture_ref = ld32(&slot[PASS_ATTACH_TEXREF..]);
    out.resolve_texture_ref = ld32(&slot[PASS_ATTACH_RESOLVEREF..]);
    // Archive uses u16 for depth/stencil level (color uses u32).
    out.level = ld16(&slot[PASS_ATTACH_LEVEL..]) as u32;
    out.load_action = ld16(&slot[PASS_ATTACH_LOAD_ACTION..]);
    out.store_action = ld16(&slot[PASS_ATTACH_STORE_ACTION..]);
    out.clear_depth = f64::from_bits(ld64(&slot[PASS_DEPTH_ATTACH_CLEAR_DEPTH..]));
    out.present = out.texture_ref != 0;
    out
}

/// Decode the stencil attachment (fixed slot @0x28).
pub fn decode_stencil_attachment(payload: &[u8]) -> StencilAttachment {
    let mut out = StencilAttachment::default();
    if payload.len() < PASS_STENCIL_ATTACH_OFF + PASS_DEPTH_STENCIL_ATTACH_STRIDE {
        return out;
    }
    let slot = &payload
        [PASS_STENCIL_ATTACH_OFF..PASS_STENCIL_ATTACH_OFF + PASS_DEPTH_STENCIL_ATTACH_STRIDE];
    out.texture_ref = ld32(&slot[PASS_ATTACH_TEXREF..]);
    out.resolve_texture_ref = ld32(&slot[PASS_ATTACH_RESOLVEREF..]);
    out.level = ld16(&slot[PASS_ATTACH_LEVEL..]) as u32;
    out.load_action = ld16(&slot[PASS_ATTACH_LOAD_ACTION..]);
    out.store_action = ld16(&slot[PASS_ATTACH_STORE_ACTION..]);
    out.clear_stencil = ld32(&slot[PASS_STENCIL_ATTACH_CLEAR_STENCIL..]);
    out.present = out.texture_ref != 0;
    out
}

pub fn stage_name(s: Stage) -> &'static str {
    match s {
        Stage::Vertex => "vertex",
        Stage::Fragment => "fragment",
        Stage::Object => "object",
        Stage::Mesh => "mesh",
        Stage::Tile => "tile",
        Stage::Unknown => "unknown",
    }
}

/// Transactional render command decode.
pub fn decode(command: &[u8]) -> Result<Command, DecodeStatus> {
    if command.len() < HEADER_LEN {
        return Err(DecodeStatus::ErrShort);
    }
    let opcode = ld32(&command[0..]);
    let command_length = ld32(&command[4..]) as usize;
    if command_length < HEADER_LEN || command_length > command.len() {
        return Err(DecodeStatus::ErrShort);
    }
    if opcode_is_apple_rejected(opcode) {
        return Err(DecodeStatus::ErrUnsupportedOpcode);
    }
    if !opcode_supported(opcode) {
        return Err(DecodeStatus::ErrUnknownOpcode);
    }
    let payload = &command[HEADER_LEN..command_length];
    let mut out = Command {
        opcode,
        command_length: command_length as u32,
        raw_payload_len: payload.len(),
        ..Default::default()
    };

    match opcode {
        OP_SET_PIPELINE => {
            if payload.len() < 4 {
                return Err(DecodeStatus::ErrShort);
            }
            out.kind = Kind::SetPipeline;
            out.pipeline_ref = ld32(payload);
            Ok(out)
        }
        OP_SET_VERTEX_BUFFER | OP_SET_FRAGMENT_BUFFER => {
            // Archive layout: [first:u32][count:u32][ {ref:u32, offset:u64} × count ]
            if payload.len() < BIND_ENTRIES {
                return Err(DecodeStatus::ErrShort);
            }
            out.kind = Kind::SetBuffer;
            out.stage = if opcode == OP_SET_VERTEX_BUFFER {
                Stage::Vertex
            } else {
                Stage::Fragment
            };
            out.first = ld32(&payload[BIND_FIRST..]);
            out.count = ld32(&payload[BIND_COUNT..]);
            if out.count == 0 || out.count > MAX_BIND_ENTRIES {
                return Err(DecodeStatus::ErrBadLength);
            }
            let need = BIND_ENTRIES + (out.count as usize) * BUFFER_BIND_ENTRY_SIZE;
            if payload.len() < need {
                return Err(DecodeStatus::ErrShort);
            }
            out.buffer_binds.clear();
            for i in 0..out.count as usize {
                let e = BIND_ENTRIES + i * BUFFER_BIND_ENTRY_SIZE;
                let r = ld32(&payload[e + BUFFER_BIND_ENTRY_REF..]);
                let o = ld64(&payload[e + BUFFER_BIND_ENTRY_OFFSET..]);
                out.buffer_binds.push((r, o));
            }
            // Convenience: first entry.
            if let Some(&(r, o)) = out.buffer_binds.first() {
                out.buffer_ref = r;
                out.buffer_offset = o;
            }
            Ok(out)
        }
        OP_SET_VERTEX_TEXTURE | OP_SET_FRAGMENT_TEXTURE => {
            // Archive layout: [first:u32][count:u32][ ref:u32 × count ]
            if payload.len() < BIND_ENTRIES {
                return Err(DecodeStatus::ErrShort);
            }
            out.kind = Kind::SetTexture;
            out.stage = if opcode == OP_SET_VERTEX_TEXTURE {
                Stage::Vertex
            } else {
                Stage::Fragment
            };
            out.first = ld32(&payload[BIND_FIRST..]);
            out.count = ld32(&payload[BIND_COUNT..]);
            if out.count == 0 || out.count > MAX_BIND_ENTRIES {
                return Err(DecodeStatus::ErrBadLength);
            }
            let need = BIND_ENTRIES + (out.count as usize) * REF_BIND_ENTRY_SIZE;
            if payload.len() < need {
                return Err(DecodeStatus::ErrShort);
            }
            out.ref_binds.clear();
            for i in 0..out.count as usize {
                let e = BIND_ENTRIES + i * REF_BIND_ENTRY_SIZE;
                out.ref_binds.push(ld32(&payload[e..]));
            }
            if let Some(&r) = out.ref_binds.first() {
                out.texture_ref = r;
            }
            Ok(out)
        }
        OP_SET_VERTEX_SAMPLER | OP_SET_FRAGMENT_SAMPLER => {
            // Archive layout: [first:u32][count:u32][ ref:u32 × count ]
            if payload.len() < BIND_ENTRIES {
                return Err(DecodeStatus::ErrShort);
            }
            out.kind = Kind::SetSampler;
            out.stage = if opcode == OP_SET_VERTEX_SAMPLER {
                Stage::Vertex
            } else {
                Stage::Fragment
            };
            out.first = ld32(&payload[BIND_FIRST..]);
            out.count = ld32(&payload[BIND_COUNT..]);
            if out.count == 0 || out.count > MAX_BIND_ENTRIES {
                return Err(DecodeStatus::ErrBadLength);
            }
            let need = BIND_ENTRIES + (out.count as usize) * REF_BIND_ENTRY_SIZE;
            if payload.len() < need {
                return Err(DecodeStatus::ErrShort);
            }
            out.ref_binds.clear();
            for i in 0..out.count as usize {
                let e = BIND_ENTRIES + i * REF_BIND_ENTRY_SIZE;
                out.ref_binds.push(ld32(&payload[e..]));
            }
            if let Some(&r) = out.ref_binds.first() {
                out.sampler_ref = r;
            }
            Ok(out)
        }
        OP_DRAW => {
            // `-[…drawPrimitives:vertexStart:vertexCount:]` calls the command
            // allocator as `alloc(1, 8)`: opcode `0x1` is the COMPACT form and
            // its payload is exactly 8 bytes —
            //   u32 primitiveType@0x00 · u16 vertexStart@0x04 · u16 vertexCount@0x06
            // The wide form of the same selector is a different opcode (`0x0`,
            // `alloc(0, 0x14)`) and is declined below rather than guessed.
            //
            // This used to read four u32s behind `payload.len() < 16`, which is
            // neither form. The only test for it was a synthetic 24-byte fixture
            // built to match the code, so nothing caught it — and every live
            // compact draw was rejected `ErrShort` and dropped. Silently, until
            // the decode refusal was named: one fired on the first arm64 boot
            // that could report it.
            if command_length != DRAW_COMPACT_CMD_LEN || payload.len() < DRAW_COMPACT_PAYLOAD_LEN {
                return Err(DecodeStatus::ErrBadLength);
            }
            out.kind = Kind::Draw;
            out.primitive_type = ld32(&payload[0..]);
            out.vertex_start = ld16(&payload[4..]) as u32;
            out.vertex_count = ld16(&payload[6..]) as u32;
            // Not on the wire: this selector is the non-instanced one, and
            // Metal draws it once.
            out.instance_count = 1;
            Ok(out)
        }
        // Wide `drawPrimitives:vertexStart:vertexCount:` — `alloc(0, 0x14)`.
        // The opcode is inside the accepted window, so without this arm it falls
        // through to `Kind::OtherAccepted` and the draw is dropped as a no-op:
        // a silent lost draw wearing the same shape as an accepted state-set.
        //
        // The payload layout is NOT decoded, because it is NOT verified. The
        // wide siblings `0x2` and `0x4` are `u64 vertexStart@0 · u64
        // vertexCount@8 · …`, so `0x14` is consistent with `u64 · u64 · u32
        // primitiveType@0x10` — consistent is not confirmed, and unknown wire
        // format stays unknown. Declining by name is the honest answer and puts
        // the gap in the log instead of in a guess.
        OP_DRAW_WIDE => Err(DecodeStatus::ErrUnsupportedOpcode),
        // Compact `drawPrimitives:vertexStart:vertexCount:instanceCount:` (0x03).
        // Wire sz 0x10 → 8-byte payload. Layout is DISTINCT from the 0x01 form:
        //   u16 vertexStart@0 · u16 vertexCount@2 · u16 instanceCount@4 · u16 primitiveType@6
        // (contract + live x86 WebKit bytes `00000400 0d000400` = vs0 vc4 inst13 primTriStrip).
        // This is WebKit's instanced glyph/rect batch; the non-instanced 0x01/indexed 0x07 forms
        // render chrome text, which is why chrome rendered while page content stayed blank.
        OP_DRAW_INST_COMPACT => {
            if payload.len() < 8 {
                return Err(DecodeStatus::ErrShort);
            }
            out.kind = Kind::Draw;
            out.vertex_start = ld16(&payload[0..]) as u32;
            out.vertex_count = ld16(&payload[2..]) as u32;
            out.instance_count = (ld16(&payload[4..]) as u32).max(1);
            out.primitive_type = ld16(&payload[6..]) as u32;
            Ok(out)
        }
        OP_DRAW_INDEXED_WIDE => {
            // Native/Rosetta MetalSerializer wide indexed form:
            // u16 primitiveType@0, u16 indexType@2, u32 indexBufferRef@4,
            // u32 indexCount@8, u32 pad@0xc, u32 indexBufferOffset@0x10,
            // u32 trailing pad@0x14. Full wire record is exactly 0x20 bytes.
            const PAYLOAD_LEN: usize = 0x18;
            if command_length != HEADER_LEN + PAYLOAD_LEN || payload.len() < PAYLOAD_LEN {
                return Err(DecodeStatus::ErrBadLength);
            }
            out.kind = Kind::Draw;
            out.primitive_type = ld16(payload) as u32;
            out.index_type = ld16(&payload[2..]) as u32;
            out.index_buffer_ref = ld32(&payload[4..]);
            out.index_count = ld32(&payload[8..]);
            out.index_buffer_offset = ld32(&payload[0x10..]) as u64;
            out.instance_count = 1;
            Ok(out)
        }
        // Live arm boot (and reims_vgpu_render_format.h): compact indexed draws.
        // Full record lengths: 0x07 → 0x14, 0x09 → 0x18 (8 B header + payload).
        // Payload layout (REIMS_VGPU_RENDER_INDEXED_ARM_*):
        //   u32 prim@0 · u32 indexBufferRef@4 · u16 count@8 · u16 offset@0xa
        //   · u16 instanceCount@0xc when instanced (0x09 only)
        // Index type is fixed UINT16 on this layout (not on the wire).
        OP_DRAW_INDEXED | OP_DRAW_INDEXED_INST => {
            const ARM_COMPACT_PAYLOAD: usize = 0x0c; // 0x14 total − 8 header
            const ARM_INST_PAYLOAD: usize = 0x10; // 0x18 total − 8 header
            const WIDE_PAYLOAD: usize = 28;
            out.kind = Kind::Draw;
            if payload.len() >= WIDE_PAYLOAD {
                // Prior wide interpretation (x86-ish / long form).
                out.primitive_type = ld32(&payload[0..]);
                out.index_count = ld32(&payload[4..]);
                out.index_type = ld32(&payload[8..]);
                out.index_buffer_ref = ld32(&payload[12..]);
                out.index_buffer_offset = ld64(&payload[16..]);
                out.instance_count = ld32(&payload[24..]).max(1);
                Ok(out)
            } else if payload.len() >= ARM_COMPACT_PAYLOAD {
                // ARM compact (live vmapple boot logo path).
                out.primitive_type = ld32(&payload[0..]);
                out.index_buffer_ref = ld32(&payload[4..]);
                out.index_count = ld16(&payload[8..]) as u32;
                out.index_buffer_offset = ld16(&payload[0x0a..]) as u64;
                out.index_type = 0; // MTLIndexTypeUInt16
                out.instance_count =
                    if opcode == OP_DRAW_INDEXED_INST && payload.len() >= ARM_INST_PAYLOAD {
                        ld16(&payload[0x0c..]) as u32
                    } else {
                        1
                    }
                    .max(1);
                Ok(out)
            } else {
                Err(DecodeStatus::ErrShort)
            }
        }
        OP_SET_VERTEX_BUFFER_OFFSET | OP_SET_FRAGMENT_BUFFER_OFFSET => {
            // Archive: index:u32 @0, offset:u64 @4 (REIMS_VGPU_RENDER_BUFFER_OFFSET_*).
            if payload.len() < BUFFER_OFFSET_PAYLOAD_LEN {
                return Err(DecodeStatus::ErrShort);
            }
            out.kind = Kind::SetBufferOffset;
            out.stage = if opcode == OP_SET_VERTEX_BUFFER_OFFSET {
                Stage::Vertex
            } else {
                Stage::Fragment
            };
            out.first = ld32(&payload[BUFFER_OFFSET_INDEX..]);
            out.buffer_offset = ld64(&payload[BUFFER_OFFSET_VALUE..]);
            Ok(out)
        }
        OP_SET_VIEWPORT => {
            if payload.len() < 48 {
                return Err(DecodeStatus::ErrShort);
            }
            out.kind = Kind::SetViewport;
            for i in 0..6 {
                let bits = ld64(&payload[i * 8..]);
                out.viewport[i] = f64::from_bits(bits);
            }
            Ok(out)
        }
        OP_SET_SCISSOR => {
            // Archive: four u64 fields at 0/8/0x10/0x18 (full payload 0x20).
            // Product previously mis-read four u32s — wrong for live op 0x75 len=40.
            if payload.len() < SCISSOR_PAYLOAD_LEN {
                return Err(DecodeStatus::ErrShort);
            }
            out.kind = Kind::SetScissor;
            out.scissor_x = ld64(&payload[SCISSOR_X..]) as u32;
            out.scissor_y = ld64(&payload[SCISSOR_Y..]) as u32;
            out.scissor_w = ld64(&payload[SCISSOR_WIDTH..]) as u32;
            out.scissor_h = ld64(&payload[SCISSOR_HEIGHT..]) as u32;
            Ok(out)
        }
        OP_SET_BLEND_COLOR => {
            if payload.len() < 16 {
                return Err(DecodeStatus::ErrShort);
            }
            out.kind = Kind::SetBlendColor;
            out.has_blend_color = true;
            for i in 0..4 {
                out.blend_color[i] = f32::from_bits(ld32(&payload[i * 4..]));
            }
            Ok(out)
        }
        OP_SET_CULL_MODE => {
            if payload.len() < 4 {
                return Err(DecodeStatus::ErrShort);
            }
            out.kind = Kind::SetCullMode;
            out.has_cull_mode = true;
            out.cull_mode = ld32(payload);
            Ok(out)
        }
        OP_SET_FRONT_FACING => {
            if payload.len() < 4 {
                return Err(DecodeStatus::ErrShort);
            }
            out.kind = Kind::SetFrontFacing;
            out.has_front_facing = true;
            out.front_facing = ld32(payload);
            Ok(out)
        }
        OP_SET_DEPTH_BIAS => {
            if payload.len() < 12 {
                return Err(DecodeStatus::ErrShort);
            }
            out.kind = Kind::SetDepthBias;
            out.has_depth_bias = true;
            out.depth_bias[0] = f32::from_bits(ld32(&payload[0..]));
            out.depth_bias[1] = f32::from_bits(ld32(&payload[4..]));
            out.depth_bias[2] = f32::from_bits(ld32(&payload[8..]));
            Ok(out)
        }
        OP_SET_DEPTH_STENCIL => {
            if payload.len() < 4 {
                return Err(DecodeStatus::ErrShort);
            }
            out.kind = Kind::SetDepthStencil;
            out.depth_stencil_ref = ld32(payload);
            Ok(out)
        }
        OP_SET_STENCIL_REF => {
            if payload.len() < 8 {
                return Err(DecodeStatus::ErrShort);
            }
            out.kind = Kind::SetStencilReference;
            out.has_stencil_ref = true;
            out.stencil_ref_front = ld32(&payload[0..]);
            out.stencil_ref_back = ld32(&payload[4..]);
            Ok(out)
        }
        OP_UPDATE_FENCE | OP_WAIT_FENCE => {
            if payload.len() < 4 {
                return Err(DecodeStatus::ErrShort);
            }
            out.kind = Kind::Fence;
            out.fence_ref = ld32(payload);
            Ok(out)
        }
        OP_USE_RESOURCE | OP_USE_HEAP => {
            if payload.len() < 4 {
                return Err(DecodeStatus::ErrShort);
            }
            out.kind = if opcode == OP_USE_RESOURCE {
                Kind::UseResource
            } else {
                Kind::UseHeap
            };
            out.resource_ref = ld32(payload);
            Ok(out)
        }
        OP_RESOURCE_BARRIER | OP_MEMORY_BARRIER => {
            out.kind = Kind::Barrier;
            Ok(out)
        }
        OP_EXECUTE_COMMANDS_INDIRECT => {
            if command_length != EXECUTE_INDIRECT_CMD_LEN || payload.len() < 0x10 {
                return Err(DecodeStatus::ErrBadLength);
            }
            out.kind = Kind::ExecuteCommands;
            out.icb_is_range = false;
            out.indirect_command_buffer_ref = ld32(&payload[EXECUTE_INDIRECT_COMMAND_BUFFER_REF..]);
            out.icb_args_buffer_ref = ld32(&payload[EXECUTE_INDIRECT_BUFFER_REF..]);
            out.icb_args_buffer_offset = ld64(&payload[EXECUTE_INDIRECT_BUFFER_OFFSET..]);
            Ok(out)
        }
        OP_EXECUTE_COMMANDS_RANGE => {
            if command_length != EXECUTE_RANGE_CMD_LEN || payload.len() < 0x14 {
                return Err(DecodeStatus::ErrBadLength);
            }
            out.kind = Kind::ExecuteCommands;
            out.icb_is_range = true;
            out.indirect_command_buffer_ref = ld32(&payload[EXECUTE_RANGE_COMMAND_BUFFER_REF..]);
            // Unaligned u64s after the ICB ref (contract).
            out.icb_range_location = ld64(&payload[EXECUTE_RANGE_LOCATION..]);
            out.icb_range_length = ld64(&payload[EXECUTE_RANGE_LENGTH..]);
            Ok(out)
        }
        OP_RENDER_PASS => {
            if payload.len() < PASS_MIN_PAYLOAD {
                return Err(DecodeStatus::ErrShort);
            }
            out.kind = Kind::RenderPass;
            out.depth = decode_depth_attachment(payload);
            out.stencil = decode_stencil_attachment(payload);
            out.color0 = decode_color_attachment(payload, 0);
            if out.color0.texture_ref != 0 {
                out.texture_ref = out.color0.texture_ref;
            }
            Ok(out)
        }
        _ => {
            out.kind = Kind::OtherAccepted;
            Ok(out)
        }
    }
}

#[cfg(test)]
mod tests {

    /// A malformed render command used to be dropped at the dispatch site with no
    /// log line at all — indistinguishable from a segment carrying no render
    /// work. Each check names itself now, `Ok` still produces nothing, and the
    /// prefix keeps them apart from the six sibling `DecodeStatus` enums.
    #[test]
    fn every_render_decode_failure_but_ok_names_its_own_check() {
        use crate::observe::Refusal;
        const ERRS: &[DecodeStatus] = &[
            DecodeStatus::ErrArgs,
            DecodeStatus::ErrShort,
            DecodeStatus::ErrUnknownOpcode,
            DecodeStatus::ErrUnsupportedOpcode,
            DecodeStatus::ErrBadLength,
        ];
        assert_eq!(DecodeStatus::Ok.refusal(), None, "Ok is not a refusal");
        let mut slugs: Vec<&str> = ERRS.iter().filter_map(|s| s.refusal()).collect();
        assert_eq!(slugs.len(), ERRS.len(), "every error variant refuses");
        assert!(slugs.iter().all(|s| s.starts_with("render_decode_")));
        slugs.sort_unstable();
        let n = slugs.len();
        slugs.dedup();
        assert_eq!(slugs.len(), n, "two render decode checks share a slug");
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
    fn pipeline_and_draw() {
        let mut v = hdr(OP_SET_PIPELINE, 12);
        st32(&mut v[8..], 9);
        let c = decode(&v).unwrap();
        assert_eq!(c.pipeline_ref, 9);

        // The compact draw form gets its own test below, against captured
        // bytes rather than a fixture shaped like the code.
    }

    /// Opcode `0x1` is the COMPACT `drawPrimitives:vertexStart:vertexCount:` —
    /// `alloc(1, 8)`, so wire sz `0x10` and an 8-byte payload of
    /// `u32 primitiveType · u16 vertexStart · u16 vertexCount`.
    ///
    /// These payload bytes are the contract's, from the RE'd encoder plus the
    /// checked-in corpus record: `03 00 00 00 00 00 06 00` = triangle list,
    /// vertexStart 0, vertexCount 6.
    ///
    /// This is the test that fails without the fix. The old fixture was a
    /// synthetic 24-byte record with four u32s, which is neither the compact nor
    /// the wide form — so the decoder rejected every real compact draw as
    /// `ErrShort` and the test agreed with it.
    #[test]
    fn compact_draw_layout_is_the_contracts_eight_byte_payload() {
        let v: [u8; 16] = [
            0x01, 0x00, 0x00, 0x00, // opcode 0x1
            0x10, 0x00, 0x00, 0x00, // sz 0x10
            0x03, 0x00, 0x00, 0x00, // primitiveType 3 (triangle list)
            0x00, 0x00, // vertexStart 0
            0x06, 0x00, // vertexCount 6
        ];
        let c = decode(&v).expect("the contract's compact draw must decode");
        assert_eq!(c.kind, Kind::Draw);
        assert_eq!(c.primitive_type, 3);
        assert_eq!(c.vertex_start, 0);
        assert_eq!(c.vertex_count, 6);
        assert_eq!(c.instance_count, 1, "the non-instanced selector draws once");

        // A nonzero start must survive: the device offsets both the stage-in
        // fetch and `[[vertex_id]]` from it, so reading it from the wrong offset
        // renders the wrong vertices rather than nothing.
        let mut v2 = v;
        v2[12] = 0x02;
        v2[14] = 0x04;
        let c2 = decode(&v2).expect("nonzero start decodes");
        assert_eq!((c2.vertex_start, c2.vertex_count), (2, 4));
    }

    /// Any other length for opcode `0x1` is not a form this contract knows, so
    /// it is refused by name rather than read at a guessed offset.
    #[test]
    fn a_compact_draw_of_the_wrong_length_is_refused_not_guessed() {
        let mut wide = hdr(OP_DRAW, 24);
        st32(&mut wide[8..], 3);
        assert_eq!(decode(&wide), Err(DecodeStatus::ErrBadLength));
        let short = hdr(OP_DRAW, 12);
        assert_eq!(decode(&short), Err(DecodeStatus::ErrBadLength));
    }

    /// The wide form is a *different opcode* and its payload layout is
    /// unverified. It must decline by name: falling through to
    /// `Kind::OtherAccepted` would drop the draw while looking like an accepted
    /// state-set, which is the silent class this vocabulary exists to end.
    #[test]
    fn the_wide_draw_form_declines_rather_than_passing_as_other_accepted() {
        let wide = hdr(OP_DRAW_WIDE, 8 + 0x14);
        assert_eq!(decode(&wide), Err(DecodeStatus::ErrUnsupportedOpcode));
    }

    #[test]
    fn compact_instanced_draw_layout_live_webkit_bytes() {
        // Live x86 WebKit content record (aneesiqbal.ai), boot serial-20260717-161608:
        //   03000000 10000000 | 00000400 0d000400
        // op 0x3, sz 0x10, payload = vs0 vc4 inst13 primTriStrip(4).
        let v: [u8; 16] = [
            0x03, 0x00, 0x00, 0x00, // opcode 0x3
            0x10, 0x00, 0x00, 0x00, // sz 0x10
            0x00, 0x00, // vertexStart 0
            0x04, 0x00, // vertexCount 4
            0x0d, 0x00, // instanceCount 13
            0x04, 0x00, // primitiveType 4 (triangle strip)
        ];
        let c = decode(&v).expect("compact instanced draw");
        assert_eq!(c.kind, Kind::Draw);
        assert_eq!(c.vertex_start, 0);
        assert_eq!(c.vertex_count, 4);
        assert_eq!(c.instance_count, 13);
        assert_eq!(c.primitive_type, 4);
        // Not misread as an indexed draw.
        assert_eq!(c.index_count, 0);
        assert_eq!(c.index_buffer_ref, 0);
    }

    #[test]
    fn wide_indexed_draw_layout() {
        use crate::contract::endian::st16;

        let mut v = hdr(OP_DRAW_INDEXED_WIDE, 0x20);
        st16(&mut v[8..], 3); // triangle
        st16(&mut v[10..], 0); // UInt16
        st32(&mut v[12..], 0x3e); // index buffer ref
        st32(&mut v[16..], 6); // index count
        st32(&mut v[24..], 0x10100); // byte offset

        let c = decode(&v).expect("wide indexed draw");
        assert_eq!(c.kind, Kind::Draw);
        assert_eq!(c.primitive_type, 3);
        assert_eq!(c.index_type, 0);
        assert_eq!(c.index_buffer_ref, 0x3e);
        assert_eq!(c.index_count, 6);
        assert_eq!(c.index_buffer_offset, 0x10100);
        assert_eq!(c.instance_count, 1);
    }

    #[test]
    fn execute_commands_range_and_indirect() {
        use crate::contract::endian::st64;
        // 0x15 withRange: ref + unaligned location/length
        let mut v = hdr(OP_EXECUTE_COMMANDS_RANGE, EXECUTE_RANGE_CMD_LEN);
        st32(&mut v[8..], 0x3333);
        st64(&mut v[12..], 5);
        st64(&mut v[20..], 7);
        let c = decode(&v).unwrap();
        assert_eq!(c.kind, Kind::ExecuteCommands);
        assert!(c.icb_is_range);
        assert_eq!(c.indirect_command_buffer_ref, 0x3333);
        assert_eq!(c.icb_range_location, 5);
        assert_eq!(c.icb_range_length, 7);
        // 0x14 indirect buffer form
        let mut v = hdr(OP_EXECUTE_COMMANDS_INDIRECT, EXECUTE_INDIRECT_CMD_LEN);
        st32(&mut v[8..], 0x1111);
        st32(&mut v[12..], 0x2222);
        st64(&mut v[16..], 0x40);
        let c = decode(&v).unwrap();
        assert!(!c.icb_is_range);
        assert_eq!(c.indirect_command_buffer_ref, 0x1111);
        assert_eq!(c.icb_args_buffer_ref, 0x2222);
        assert_eq!(c.icb_args_buffer_offset, 0x40);
    }

    #[test]
    fn depth_and_stencil_pass_slots() {
        use crate::contract::endian::{st16, st32, st64};
        let mut payload = vec![0u8; PASS_MIN_PAYLOAD];
        // depth @0
        st32(
            &mut payload[PASS_DEPTH_ATTACH_OFF + PASS_ATTACH_TEXREF..],
            77,
        );
        st16(
            &mut payload[PASS_DEPTH_ATTACH_OFF + PASS_ATTACH_LOAD_ACTION..],
            PASS_LOAD_ACTION_CLEAR,
        );
        st16(
            &mut payload[PASS_DEPTH_ATTACH_OFF + PASS_ATTACH_STORE_ACTION..],
            PASS_STORE_ACTION_STORE,
        );
        st64(
            &mut payload[PASS_DEPTH_ATTACH_OFF + PASS_DEPTH_ATTACH_CLEAR_DEPTH..],
            0.5f64.to_bits(),
        );
        // stencil @0x28
        st32(
            &mut payload[PASS_STENCIL_ATTACH_OFF + PASS_ATTACH_TEXREF..],
            88,
        );
        st32(
            &mut payload[PASS_STENCIL_ATTACH_OFF + PASS_STENCIL_ATTACH_CLEAR_STENCIL..],
            9,
        );
        let d = decode_depth_attachment(&payload);
        assert!(d.present);
        assert_eq!(d.texture_ref, 77);
        assert!((d.clear_depth - 0.5).abs() < 1e-9);
        let s = decode_stencil_attachment(&payload);
        assert!(s.present);
        assert_eq!(s.texture_ref, 88);
        assert_eq!(s.clear_stencil, 9);
    }

    #[test]
    fn blend_color_and_cull() {
        let mut v = hdr(OP_SET_BLEND_COLOR, 24);
        // RGBA as f32 bits
        st32(&mut v[8..], 1.0f32.to_bits());
        st32(&mut v[12..], 0.0f32.to_bits());
        st32(&mut v[16..], 0.0f32.to_bits());
        st32(&mut v[20..], 1.0f32.to_bits());
        let c = decode(&v).unwrap();
        assert!(c.has_blend_color);
        assert!((c.blend_color[0] - 1.0).abs() < 1e-6);

        let mut v = hdr(OP_SET_CULL_MODE, 12);
        st32(&mut v[8..], 2);
        let c = decode(&v).unwrap();
        assert!(c.has_cull_mode);
        assert_eq!(c.cull_mode, 2);
    }

    #[test]
    fn rejected_unknown() {
        assert!(opcode_is_apple_rejected(0xff));
        assert_eq!(
            decode(&hdr(0xff, 16)).unwrap_err(),
            DecodeStatus::ErrUnsupportedOpcode
        );
        // 0x99 is just above the accepted max 0x98
        assert_eq!(
            decode(&hdr(0x99, 16)).unwrap_err(),
            DecodeStatus::ErrUnsupportedOpcode
        );
    }

    #[test]
    fn property_fuzz() {
        for op in 0u32..0x120 {
            for len in [8usize, 12, 16, 24, 32, 48, 64] {
                let _ = decode(&hdr(op, len));
            }
        }
    }
}
