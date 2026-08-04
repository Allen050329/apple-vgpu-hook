//! Render command decoder (port of `host/utils/reims-vgpu-render-decode`).
//!
//! Full opcode matrix is preserved for supported/rejected classification.
//! Per-opcode payload layouts for the highest-traffic families (set pipeline,
//! buffer/texture binds, draw, viewport/scissor, barriers, fences) are decoded;
//! remaining accepted opcodes are recognized and returned as typed kinds with
//! raw length validation where the contract specifies fixed sizes.

use reims_vgpu_wire::ops::render as wire;
use reims_vgpu_wire::ops::render_pass as wire_pass;
use reims_vgpu_wire::ops::tile as wire_tile;

use reims_vgpu_wire::OP_HEADER_LEN;

/// Map a `reims-vgpu-wire` view onto a payload, translating its refusal.
///
/// The draw layouts live in that crate, derived from Apple's own serializer and
/// pinned by fixtures, so this module reads them rather than restating them —
/// one declaration, and drift is impossible rather than merely detectable. What
/// stays here is everything the crate cannot know: which `Kind` a record is,
/// what the runtime's `Command` calls each field, and how a refusal is named.
#[inline]
fn wire_view<T: reims_vgpu_wire::Wire>(payload: &[u8]) -> Result<&T, DecodeStatus> {
    reims_vgpu_wire::view::<T>(payload).map_err(|_| DecodeStatus::ErrShort)
}

/// Narrow a wide draw's 64-bit count to the 32 bits `Command` carries.
///
/// The wide forms exist because the guest had a value above 16 bits, not above
/// 32: a vertex or index count of four billion is not a draw any GPU completes.
/// Truncating one would draw the wrong geometry in silence, so it is refused by
/// name instead.
#[inline]
fn narrow_count(value: u64) -> Result<u32, DecodeStatus> {
    u32::try_from(value).map_err(|_| DecodeStatus::ErrCountOutOfRange)
}

// Layout lengths for fixed-size records and bind tables. Opcodes live in
// `reims_vgpu_wire::ops::{render,render_pass,tile}`; this module maps them into
// product `Kind`/`Command` and does not re-export wire opcode constants.
/// Compact `drawPrimitives:vertexStart:vertexCount:` payload length (`alloc(1, 8)`).
/// Checked exactly: a `0x1` record of any other size is not a form this contract knows.
pub const DRAW_COMPACT_PAYLOAD_LEN: usize = 8;
/// Compact draw total length including the shared op header.
pub const DRAW_COMPACT_CMD_LEN: usize = OP_HEADER_LEN + DRAW_COMPACT_PAYLOAD_LEN;
/// Fixed total lengths for ICB execute records (header + payload). Prefer
/// `wire::EXECUTE_COMMANDS_*_TOTAL_LEN` at new call sites; these remain for
/// historical tests and local checks that already use them.
pub const EXECUTE_INDIRECT_CMD_LEN: usize = 0x18;
pub const EXECUTE_RANGE_CMD_LEN: usize = 0x1c;
pub const EXECUTE_INDIRECT_COMMAND_BUFFER_REF: usize = 0;
pub const EXECUTE_INDIRECT_BUFFER_REF: usize = 4;
pub const EXECUTE_INDIRECT_BUFFER_OFFSET: usize = 8;
pub const EXECUTE_RANGE_COMMAND_BUFFER_REF: usize = 0;
pub const EXECUTE_RANGE_LOCATION: usize = 4;
pub const EXECUTE_RANGE_LENGTH: usize = 0x0c;

/// Render-pass attachment layout, taken from the wire structs' own fields.
///
/// The three sections are contiguous, so each record's extent is the distance to
/// the one after it and is never written down separately: depth is
/// `[0x00, 0x28)`, stencil is `[0x28, 0x4c)`, and the color slots run from 0x4c
/// at `PASS_COLOR_ATTACH_STRIDE` each. A single "depth/stencil stride" constant
/// used to state both of the first two as 0x28, which is right for depth and
/// 4 bytes too long for stencil — that spare word is color slot 0's texture ref.
///
/// Offsets are `offset_of!` / `size_of!` on
/// [`reims_vgpu_wire::ops::render_pass`]. Attachment decode maps wire attachment
/// bodies rather than hand-loading fields; `level` is sixteen bits with `slice`
/// immediately above it (a former product colour-arm u32 load swallowed the
/// slice).
pub const PASS_DEPTH_ATTACH_OFF: usize = 0x00;
pub const PASS_STENCIL_ATTACH_OFF: usize = core::mem::size_of::<wire_pass::DepthAttachmentBody>();
pub const PASS_COLOR_ATTACH_OFF: usize =
    PASS_STENCIL_ATTACH_OFF + core::mem::size_of::<wire_pass::StencilAttachmentBody>();
pub const PASS_COLOR_ATTACH_STRIDE: usize = core::mem::size_of::<wire_pass::ColorAttachmentBody>();
pub const PASS_ATTACH_TEXREF: usize =
    core::mem::offset_of!(wire_pass::AttachmentPrefix, texture_ref);
pub const PASS_ATTACH_RESOLVEREF: usize =
    core::mem::offset_of!(wire_pass::AttachmentPrefix, resolve_texture_ref);
pub const PASS_ATTACH_LEVEL: usize = core::mem::offset_of!(wire_pass::AttachmentPrefix, level);
pub const PASS_ATTACH_SLICE: usize = core::mem::offset_of!(wire_pass::AttachmentPrefix, slice);
pub const PASS_ATTACH_DEPTH_PLANE: usize =
    core::mem::offset_of!(wire_pass::AttachmentPrefix, depth_plane);
pub const PASS_ATTACH_LOAD_ACTION: usize =
    core::mem::offset_of!(wire_pass::AttachmentPrefix, load_action);
pub const PASS_ATTACH_STORE_ACTION: usize =
    core::mem::offset_of!(wire_pass::AttachmentPrefix, store_action);
pub const PASS_ATTACH_CLEAR_COLOR: usize =
    core::mem::offset_of!(wire_pass::ColorAttachmentBody, clear_color_bits);
pub const PASS_DEPTH_ATTACH_CLEAR_DEPTH: usize =
    core::mem::offset_of!(wire_pass::DepthAttachmentBody, clear_depth_bits);
pub const PASS_STENCIL_ATTACH_CLEAR_STENCIL: usize =
    core::mem::offset_of!(wire_pass::StencilAttachmentBody, clear_stencil);
pub const PASS_MAX_COLOR_ATTACHMENTS: usize = wire_pass::RENDER_PASS_COLOR_ATTACHMENTS;

/// Offset of the pass-level tail, past the last colour slot.
///
/// Four fields this device decodes and does not apply. They are the guest's
/// explicit statement about the pass's extent and its occlusion query buffer,
/// and none of them can be recovered from the attachments: a guest may bind a
/// 4096-wide texture and ask for a 640-wide pass.
pub const PASS_TAIL_OFF: usize =
    PASS_COLOR_ATTACH_OFF + PASS_MAX_COLOR_ATTACHMENTS * PASS_COLOR_ATTACH_STRIDE;
pub const PASS_TAIL_VISIBILITY_BUFFER_REF: usize = 0x00;
pub const PASS_TAIL_ARRAY_LENGTH: usize = 0x04;
pub const PASS_TAIL_TARGET_WIDTH: usize = 0x0c;
pub const PASS_TAIL_TARGET_HEIGHT: usize = 0x14;
pub const PASS_LOAD_ACTION_DONT_CARE: u16 = 0;
pub const PASS_LOAD_ACTION_LOAD: u16 = 1;
pub const PASS_LOAD_ACTION_CLEAR: u16 = 2;
pub const PASS_STORE_ACTION_DONT_CARE: u16 = 0;
pub const PASS_STORE_ACTION_STORE: u16 = 1;
pub const PASS_MIN_PAYLOAD: usize = PASS_COLOR_ATTACH_OFF + PASS_COLOR_ATTACH_STRIDE;
/// Count width of `setScissorRects:count:` — eight bytes, not the four used by
/// `setViewports:count:`. The element is the singular scissor payload.
pub const SCISSOR_RECTS_COUNT_LEN: usize = 8;
/// Bytes one LOD-bearing sampler entry occupies: ref, then two `f32` clamps.
pub const SAMPLER_LOD_BIND_ENTRY_SIZE: usize = 12;
/// Count width of `setViewports:count:` — four bytes (see [`SCISSOR_RECTS_COUNT_LEN`]).
pub const VIEWPORTS_COUNT_LEN: usize = 4;

/// Residency record head: `count:u32` at `+0` on both forms.
///
/// Wire opcodes are `wire::OPCODE_USE_HEAP` (`0x1b`) and
/// `wire::OPCODE_USE_RESOURCE` (`0x89`); the old `0x86`/`0x87` pair is not
/// residency (see `the_residency_opcodes_are_the_ones_apples_serializer_writes`).
pub const RESIDENCY_COUNT: usize = 0;
/// `useResource:` packs `usage` and `stages` into the word at `+4` as two
/// `u16`s, so its refs begin at `+8`.
pub const USE_RESOURCE_REFS: usize = 8;
/// `useHeap:` has no `usage` at all: `stages` sits alone at `+4` as a `u16` and
/// the refs begin at `+6`. That offset is deliberately not a multiple of four —
/// reading this record with the resource record's layout skips the first heap.
pub const USE_HEAP_REFS: usize = 6;

/// Multi-entry bind header (reims_vgpu_render_format.h): first:u32 @0, count:u32 @4.
pub const BIND_FIRST: usize = 0;
pub const BIND_COUNT: usize = 4;
pub const BIND_ENTRIES: usize = 8;
pub const BUFFER_BIND_ENTRY_SIZE: usize = 12;
/// The same entry with a `u64` attribute stride appended. See
/// [`wire::OPCODE_SET_VERTEX_BUFFER_STRIDE`].
pub const BUFFER_STRIDE_BIND_ENTRY_SIZE: usize = 20;
/// `setVertexAmplificationCount:viewMappings:`: a four-byte count, then one
/// `MTLVertexAmplificationViewMapping` (two `u32`) per view.
pub const AMPLIFICATION_COUNT_LEN: usize = 4;
pub const AMPLIFICATION_MAPPING_SIZE: usize = 8;
pub const BUFFER_BIND_ENTRY_REF: usize = 0;
pub const BUFFER_BIND_ENTRY_OFFSET: usize = 4;
pub const REF_BIND_ENTRY_SIZE: usize = 4;

/// Bytes a bind record needs for `count` entries of `entry_size`, or `None` if
/// no record could be that long.
///
/// **A bind record is bounded by its own length and by nothing else.** This
/// replaced a `MAX_BIND_ENTRIES = 32` cap that had no citation and was not
/// Apple's: `setVertexTextures:withRange:` over a range of 40 produces a
/// 176-byte record (fixture `render_set_vertex_textures_range_40`), which that
/// cap refused with `ErrBadLength` — dropping all forty binds rather than the
/// eight that would not fit a table. Metal's own limit is 128 textures per
/// stage, so 32 was not even the API's number.
///
/// The count stays guest-controlled and is never trusted before this check:
/// nothing is allocated or read until the entries are known to be inside the
/// record the guest itself sized.
#[inline]
pub fn bind_record_len(count: u32, entry_size: usize) -> Option<usize> {
    (count as usize)
        .checked_mul(entry_size)
        .and_then(|n| n.checked_add(BIND_ENTRIES))
}
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

/// Why the render decoder refused a command.
///
/// No `Ok` and no `ErrArgs`, for the reason recorded on `blit::DecodeStatus`:
/// success is the result's own `Ok`, and a bad argument here is a payload
/// shorter than the field, which `ErrShort` already names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeStatus {
    ErrShort,
    ErrUnknownOpcode,
    ErrUnsupportedOpcode,
    ErrBadLength,
    /// A wide draw's 64-bit count does not fit the 32 bits `Command` carries.
    /// See [`narrow_count`] for why that is refused rather than truncated.
    ErrCountOutOfRange,
}

impl crate::observe::Refusal for DecodeStatus {
    /// Slugs carry a `render_decode_` prefix: seven modules under
    /// `runtime/decode/` define a type called `DecodeStatus`, and five of them
    /// have an `ErrShort` that means a different read. Without the prefix the
    /// crate-wide uniqueness gate could not tell the render decoder's refusals
    /// from any other's.
    fn refusal(&self) -> Option<&'static str> {
        Some(match self {
            Self::ErrShort => "render_decode_short",
            Self::ErrUnknownOpcode => "render_decode_unknown_opcode",
            Self::ErrUnsupportedOpcode => "render_decode_unsupported_opcode",
            Self::ErrBadLength => "render_decode_bad_length",
            Self::ErrCountOutOfRange => "render_decode_count_out_of_range",
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
    /// `setTriangleFillMode:` / `setDepthClipMode:`. The value is in
    /// [`Command::mode`]; which state it sets is [`Command::opcode`], as on the
    /// wire.
    SetRasterState,
    /// `setLineWidth:` / `setTessellationFactorScale:`, value in
    /// [`Command::float_value`].
    SetFloatState,
    /// `setColorStoreAction:atIndex:` and the depth and stencil forms. The
    /// colour form carries an index; the other two have one attachment each and
    /// carry none.
    SetStoreAction,
    UseResource,
    UseHeap,
    ExecuteCommands,
    RenderPass,
    /// `drawPrimitives:indirectBuffer:` and its indexed sibling. Which of the
    /// two is [`Command::opcode`], as on the wire. The buffer holding the
    /// counts is [`Command::indirect_buffer_ref`] at
    /// [`Command::indirect_buffer_offset`]; the indexed form also fills
    /// [`Command::index_type`], [`Command::index_buffer_ref`] and
    /// [`Command::index_buffer_offset`].
    DrawIndirect,
    /// `setVisibilityResultMode:offset:`. Mode in [`Command::mode`], offset in
    /// [`Command::visibility_result_offset`].
    SetVisibilityResultMode,
    /// `setVertexAmplificationMode:value:` fills [`Command::mode`] and
    /// [`Command::amplification_value`]; `setVertexAmplificationCount:viewMappings:`
    /// fills [`Command::count`]. Which of the two is [`Command::opcode`], as on
    /// the wire. The view mappings are not lifted — see
    /// [`wire::OPCODE_SET_VERTEX_AMPLIFICATION_MODE`].
    SetVertexAmplification,
    /// A bind against the **tile** argument tables: `0x9d`/`0x9e` (buffer and
    /// offset), `0x9f`/`0xa0` (sampler, plain and LOD-bearing), `0xa1`
    /// (texture). Which of the five is [`Command::opcode`], as on the wire;
    /// the slots are [`Command::first`] and [`Command::count`].
    ///
    /// **Deliberately not `SetBuffer`/`SetTexture`/`SetSampler` with a third
    /// [`Stage`].** The records are those records byte for byte, so folding
    /// them in is the tempting shape — but every existing executor arm reads
    /// `Stage` as vertex-or-fragment, and a tile texture routed through one
    /// would bind into the *fragment* table. That is worse than dropping it:
    /// the guest's fragment shader would sample a texture it never bound. A
    /// kind nothing else matches cannot be mis-applied by an arm that has not
    /// been taught about tiles.
    TileBind,
    /// A tile-shader dispatch: `0x9b`, or `0xa2`/`0xa3` bounded to a region.
    /// Threads per tile in [`Command::tile_threads`].
    TileDispatch,
    /// `getTileDimensions:` (`0xa4`), the guest asking the host to report tile
    /// geometry into [`Command::buffer_ref`] at [`Command::buffer_offset`].
    TileDimensionsQuery,
    /// `set{Color,Depth,Stencil}StoreActionOptions:` (`0x67`/`0x6a`/`0x79`).
    /// Options in [`Command::mode`]; the colour form also fills
    /// [`Command::first`] with the attachment index. Distinct from
    /// [`Kind::SetStoreAction`], which is the action rather than its options
    /// and is a different record at a different width.
    SetStoreActionOptions,
    /// `setTessellationFactorBuffer:offset:instanceStride:` (`0x7a`), in
    /// [`Command::buffer_ref`] and [`Command::buffer_offset`].
    SetTessellationFactorBuffer,
    /// A tessellated draw: [`wire::OPCODE_DRAW_PATCHES`] and its four siblings. Which
    /// form is [`Command::opcode`], as on the wire — except that `0x0c` is two
    /// records and [`Command::command_length`] is what separates them.
    ///
    /// No field is lifted, because nothing here tessellates and a `patch_count`
    /// with no consumer is worse than its absence. The record is still fully
    /// bounds-checked, so a truncated one is refused rather than reported as a
    /// smaller draw.
    DrawPatches,
    /// One pass property `writeDescriptor` emits as a record of its own rather
    /// than as a field of the pass descriptor: `0x1e` the default raster sample
    /// count, `0x20` the programmable sample positions, `0x21` the
    /// rasterization rate map, `0x22`/`0x23` the imageblock and threadgroup
    /// memory lengths, `0x24` the tile size. Which one is [`Command::opcode`],
    /// as on the wire; the scalar is [`Command::mode`] and the rate map's ref is
    /// [`Command::object_ref`].
    ///
    /// Decoded and counted rather than applied. Each of the six is behind a
    /// capability that defaults off, so a non-zero count is the first evidence
    /// this project would have that any guest negotiates one — which is a thing
    /// nothing in this device currently observes.
    RenderPassProperty,
    OtherAccepted,
}

/// One color attachment from a render-pass descriptor (0x1a).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ColorAttachment {
    pub present: bool,
    pub texture_ref: u32,
    pub resolve_texture_ref: u32,
    pub level: u32,
    /// Array slice, sixteen bits on the wire directly above `level`.
    pub slice: u32,
    /// Depth plane of a 3D attachment, sixteen bits above `slice`.
    pub depth_plane: u32,
    pub load_action: u16,
    pub store_action: u16,
    /// MTLClearColor as RGBA doubles in `[0,1]`.
    pub clear_color: [f64; 4],
}

/// Depth attachment from a render-pass descriptor (slot @0x00).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DepthAttachment {
    pub present: bool,
    pub texture_ref: u32,
    pub resolve_texture_ref: u32,
    pub level: u32,
    /// Array slice and depth plane, the two sixteen-bit fields above `level`.
    ///
    /// All three attachment shapes share one 28-byte prefix — that is what
    /// `reims_vgpu_wire::ops::render_pass::AttachmentPrefix` is — so these are
    /// here for the same reason they are on [`ColorAttachment`]: a depth buffer
    /// bound at slice 5 is as real as a colour target bound there, and a field
    /// nothing decodes is a field nothing can report.
    pub slice: u32,
    pub depth_plane: u32,
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
    /// See [`DepthAttachment::slice`]; the prefix is the same 28 bytes.
    pub slice: u32,
    pub depth_plane: u32,
    pub load_action: u16,
    pub store_action: u16,
    pub clear_stencil: u32,
}

/// Which encoder table a render bind record names.
///
/// Derived from the opcode, not from a wire field:
/// `wire::OPCODE_SET_VERTEX_*` versus `wire::OPCODE_SET_FRAGMENT_*`. The render
/// opcode set expresses no other stage, so there are no other variants — an
/// object/mesh/tile bind reaches the device through the indirect-command-buffer
/// path and carries [`crate::runtime::icb::IcbRenderBindStage`], which is a
/// different vocabulary with a different wire encoding.
///
/// Keeping this exhaustive is the point. With unreachable variants present,
/// every `match` over it needed a catch-all, and a catch-all is what would
/// swallow a genuinely new stage in silence.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Stage {
    /// The record named no stage, or the opcode was not a stage-bearing one.
    #[default]
    Unknown,
    Vertex,
    Fragment,
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
    /// Metal `baseInstance` / Vulkan `firstInstance`. Zero on the draw forms
    /// whose selector has no such argument, which is what Metal defaults to.
    pub base_instance: u32,
    /// Metal `baseVertex` / Vulkan `vertexOffset`, on the indexed forms that
    /// carry one. Signed: Metal declares it `NSInteger`, Apple's serializer
    /// declares it `q` where every count beside it is `Q`, and a negative
    /// offset read as unsigned becomes a large index rather than an error.
    pub base_vertex: i64,
    pub viewport: [f64; 6],
    pub scissor_x: u32,
    pub scissor_y: u32,
    pub scissor_w: u32,
    pub scissor_h: u32,
    pub fence_ref: u32,
    /// Value of a [`Kind::SetRasterState`], [`Kind::SetStoreAction`] or
    /// [`Kind::SetVisibilityResultMode`] record.
    pub mode: u64,
    /// Buffer a [`Kind::DrawIndirect`] record reads its counts from, and the
    /// byte offset of the arguments structure within it. Distinct from
    /// [`Command::index_buffer_ref`], which the indexed form fills as well —
    /// an indexed indirect draw names two buffers and two offsets, and a
    /// decoder that crossed them would replay the wrong one.
    pub indirect_buffer_ref: u32,
    pub indirect_buffer_offset: u64,
    /// Byte offset a [`Kind::SetVisibilityResultMode`] record writes its
    /// occlusion counter to, within the pass's visibility result buffer.
    pub visibility_result_offset: u64,
    /// `value` of a [`Kind::SetVertexAmplification`] mode record. Thirty-two
    /// bits: the selector declares it `Q` and the serializer narrows it, which
    /// only the capture shows.
    pub amplification_value: u32,
    /// Threads per tile of a [`Kind::TileDispatch`], as width/height/depth.
    ///
    /// Unnarrowed `u64` — the serializer writes all three at full width, unlike
    /// almost every other count in this protocol. Read by `runtime::exec` both
    /// to tell an empty dispatch from a real one and to say in the fail line
    /// how much work was dropped.
    pub tile_threads: [u64; 3],
    /// Value of a [`Kind::SetFloatState`] record.
    pub float_value: f32,
    /// The sampler bind carried per-entry LOD clamps this decoder did not lift.
    /// Only ever true on [`Kind::SetSampler`]; see [`wire::OPCODE_SET_VERTEX_SAMPLER_LOD`].
    pub has_sampler_lod: bool,
    /// The vertex buffer bind carried a per-entry attribute stride this decoder
    /// did not lift. True on [`Kind::SetBuffer`] and [`Kind::SetBufferOffset`];
    /// see [`wire::OPCODE_SET_VERTEX_BUFFER_STRIDE`]. The buffer still binds — what is
    /// missing is the stride the guest wanted the vertex fetch to use.
    pub has_attribute_stride: bool,
    pub raw_payload_len: usize,
    /// Color attachment[0] when kind is RenderPass (boot clear path).
    pub color0: ColorAttachment,
    pub depth: DepthAttachment,
    pub stencil: StencilAttachment,
    /// The pass's own tail, on [`Kind::RenderPass`]. Decoded, not applied.
    ///
    /// `visibility_result_buffer_ref` is the buffer
    /// `setVisibilityResultMode:offset:` indexes — that record carries the mode
    /// and the offset and *only* the pass record names the buffer, so the two
    /// halves of an occlusion query arrive on different records. The three
    /// geometry fields are the guest's explicit statement about the pass extent
    /// and layer count, which cannot be recovered from the attachments: a guest
    /// may bind a 4096-wide texture and ask for a 640-wide pass.
    pub pass_visibility_result_buffer_ref: u32,
    pub pass_render_target_array_length: u64,
    pub pass_render_target_width: u64,
    pub pass_render_target_height: u64,
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

/// Whether an opcode is above every one Apple's serializer writes here.
///
/// Named for what it measures. Its predecessor was `opcode_is_apple_rejected`,
/// which asserted the serializer would never emit anything above the window --
/// and it emitted `0xa5` and `0xa6`, so this device refused four vertex binds as
/// records Apple does not produce. That is the same correction
/// `decode::blit::opcode_unimplemented_here` needed, and the same lesson: the
/// highest opcode *this project has driven* is not the highest Apple writes.
///
/// The bound comes from [`wire::OPCODE_SET_VERTEX_BUFFER_OFFSET_STRIDE`], which is now derived from
/// `reims_vgpu_wire`'s manifest rather than from observation.
/// An opcode inside the accepted window that no decode arm claims.
///
/// Two tests need one -- this module's catch-all test and `runtime::exec`'s
/// fail-visible test -- and both used to hardcode it. Both went stale, twice:
/// `wire::OPCODE_SET_VERTEX_BUFFER_OFFSET_STRIDE` stopped working when that bound was corrected to `0xa6`,
/// and its replacement `0x99` lasted one commit until
/// `setVertexAmplificationMode:value:` turned out to be exactly that number.
/// Searching keeps them honest as arms are added, because what they test is
/// that the catch-all exists and reports, not that any number is in it.
#[cfg(test)]
pub(crate) fn unclaimed_accepted_opcode() -> u32 {
    (0..=wire::OPCODE_SET_VERTEX_BUFFER_OFFSET_STRIDE)
        .find(|&op| {
            let mut v = vec![0u8; OP_HEADER_LEN];
            crate::contract::endian::st32(&mut v[0..4], op);
            crate::contract::endian::st32(&mut v[4..8], OP_HEADER_LEN as u32);
            matches!(decode(&v), Ok(c) if c.kind == Kind::OtherAccepted)
        })
        .expect("every opcode in the window is decoded; the catch-all is unreachable")
}

pub fn opcode_above_the_encoder_window(opcode: u32) -> bool {
    opcode > wire::OPCODE_SET_VERTEX_BUFFER_OFFSET_STRIDE
}

pub fn opcode_supported(opcode: u32) -> bool {
    if opcode_above_the_encoder_window(opcode) {
        return false;
    }
    // Full accepted window from reims_vgpu_render_decode.h enum range.
    opcode <= wire::OPCODE_SET_VERTEX_BUFFER_OFFSET_STRIDE
}

/// Decode color attachment slot `index` from a render-pass payload.
///
/// # `level` is sixteen bits and `slice` is the sixteen above it
///
/// This read was `ld32` at `PASS_ATTACH_LEVEL` for as long as the function has
/// existed, with a comment on [`decode_depth_attachment`] stating the rule as
/// "the archive uses u16 for depth/stencil level (color uses u32)". Nothing had
/// ever checked that; Apple's own bytes say all three shapes are identical here,
/// and the four bytes at `+0x08` are `level` then `slice`.
///
/// So a pass rendering into array slice 1 — a cube face, a texture-array layer,
/// a layered shadow map — reported mip level 65536 and lost its slice entirely.
/// Both are decoded now.
fn color_from_wire(c: &wire_pass::ColorAttachmentBody) -> ColorAttachment {
    let p = &c.prefix;
    let texture_ref = p.texture_ref.get();
    ColorAttachment {
        texture_ref,
        resolve_texture_ref: p.resolve_texture_ref.get(),
        level: u32::from(p.level.get()),
        slice: u32::from(p.slice.get()),
        depth_plane: u32::from(p.depth_plane.get()),
        load_action: p.load_action.get(),
        store_action: p.store_action.get(),
        clear_color: c.clear_color(),
        present: texture_ref != 0,
        ..Default::default()
    }
}

fn depth_from_wire(d: &wire_pass::DepthAttachmentBody) -> DepthAttachment {
    let p = &d.prefix;
    let texture_ref = p.texture_ref.get();
    DepthAttachment {
        texture_ref,
        resolve_texture_ref: p.resolve_texture_ref.get(),
        level: u32::from(p.level.get()),
        slice: u32::from(p.slice.get()),
        depth_plane: u32::from(p.depth_plane.get()),
        load_action: p.load_action.get(),
        store_action: p.store_action.get(),
        clear_depth: d.clear_depth(),
        present: texture_ref != 0,
        ..Default::default()
    }
}

fn stencil_from_wire(s: &wire_pass::StencilAttachmentBody) -> StencilAttachment {
    let p = &s.prefix;
    let texture_ref = p.texture_ref.get();
    StencilAttachment {
        texture_ref,
        resolve_texture_ref: p.resolve_texture_ref.get(),
        level: u32::from(p.level.get()),
        slice: u32::from(p.slice.get()),
        depth_plane: u32::from(p.depth_plane.get()),
        load_action: p.load_action.get(),
        store_action: p.store_action.get(),
        clear_stencil: s.clear_stencil.get(),
        present: texture_ref != 0,
        ..Default::default()
    }
}

pub fn decode_color_attachment(payload: &[u8], index: usize) -> ColorAttachment {
    let base = PASS_COLOR_ATTACH_OFF + index * PASS_COLOR_ATTACH_STRIDE;
    match reims_vgpu_wire::view_at::<wire_pass::ColorAttachmentBody>(payload, base) {
        Ok(c) => color_from_wire(c),
        Err(_) => ColorAttachment::default(),
    }
}

/// Decode the depth attachment (fixed slot @0).
pub fn decode_depth_attachment(payload: &[u8]) -> DepthAttachment {
    match reims_vgpu_wire::view::<wire_pass::DepthAttachmentBody>(payload) {
        Ok(d) if payload.len() >= PASS_STENCIL_ATTACH_OFF => depth_from_wire(d),
        _ => DepthAttachment::default(),
    }
}

/// Decode the stencil attachment (fixed slot after depth).
pub fn decode_stencil_attachment(payload: &[u8]) -> StencilAttachment {
    match reims_vgpu_wire::view_at::<wire_pass::StencilAttachmentBody>(
        payload,
        PASS_STENCIL_ATTACH_OFF,
    ) {
        Ok(s) => stencil_from_wire(s),
        Err(_) => StencilAttachment::default(),
    }
}

/// Transactional render command decode.
pub fn decode(command: &[u8]) -> Result<Command, DecodeStatus> {
    let op = reims_vgpu_wire::op(command, 0).map_err(|_| DecodeStatus::ErrShort)?;
    let opcode = op.opcode();
    let command_length = op.length() as usize;
    if opcode_above_the_encoder_window(opcode) {
        return Err(DecodeStatus::ErrUnsupportedOpcode);
    }
    if !opcode_supported(opcode) {
        return Err(DecodeStatus::ErrUnknownOpcode);
    }
    let payload = op.payload;
    let mut out = Command {
        opcode,
        command_length: command_length as u32,
        raw_payload_len: payload.len(),
        ..Default::default()
    };

    match opcode {
        wire::OPCODE_SET_RENDER_PIPELINE_STATE => {
            let r = wire::state_ref(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::SetPipeline;
            out.pipeline_ref = r.object_ref.get();
            Ok(out)
        }
        wire::OPCODE_SET_VERTEX_BUFFER | wire::OPCODE_SET_FRAGMENT_BUFFER => {
            let (head, entries) = wire::buffer_binds(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::SetBuffer;
            out.stage = if opcode == wire::OPCODE_SET_FRAGMENT_BUFFER {
                Stage::Fragment
            } else {
                Stage::Vertex
            };
            out.first = head.first.get();
            out.count = head.count.get();
            if out.count == 0 {
                return Err(DecodeStatus::ErrBadLength);
            }
            // Exact length: product refuses slack the guest did not size for.
            match bind_record_len(out.count, BUFFER_BIND_ENTRY_SIZE) {
                Some(need) if payload.len() >= need => {}
                _ => return Err(DecodeStatus::ErrShort),
            }
            out.buffer_binds.clear();
            for e in entries {
                out.buffer_binds.push((e.buffer_ref.get(), e.offset.get()));
            }
            if let Some(&(r, o)) = out.buffer_binds.first() {
                out.buffer_ref = r;
                out.buffer_offset = o;
            }
            Ok(out)
        }
        wire::OPCODE_SET_VERTEX_BUFFER_STRIDE => {
            // Attribute-stride form: twenty-byte entries. Stride is not lifted
            // (nothing applies it), but the bind itself must decode.
            let (head, entries) =
                wire::buffer_stride_binds(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::SetBuffer;
            out.stage = Stage::Vertex;
            out.has_attribute_stride = true;
            out.first = head.first.get();
            out.count = head.count.get();
            if out.count == 0 {
                return Err(DecodeStatus::ErrBadLength);
            }
            match bind_record_len(out.count, BUFFER_STRIDE_BIND_ENTRY_SIZE) {
                Some(need) if payload.len() >= need => {}
                _ => return Err(DecodeStatus::ErrShort),
            }
            out.buffer_binds.clear();
            for e in entries {
                out.buffer_binds.push((e.buffer_ref.get(), e.offset.get()));
            }
            if let Some(&(r, o)) = out.buffer_binds.first() {
                out.buffer_ref = r;
                out.buffer_offset = o;
            }
            Ok(out)
        }
        wire::OPCODE_SET_VERTEX_TEXTURE
        | wire::OPCODE_SET_FRAGMENT_TEXTURE
        | wire::OPCODE_SET_VERTEX_SAMPLER
        | wire::OPCODE_SET_FRAGMENT_SAMPLER => {
            let (head, entries) = wire::ref_binds(&op).map_err(|_| DecodeStatus::ErrShort)?;
            let textures = opcode == wire::OPCODE_SET_VERTEX_TEXTURE
                || opcode == wire::OPCODE_SET_FRAGMENT_TEXTURE;
            out.kind = if textures {
                Kind::SetTexture
            } else {
                Kind::SetSampler
            };
            out.stage = if opcode == wire::OPCODE_SET_VERTEX_TEXTURE
                || opcode == wire::OPCODE_SET_VERTEX_SAMPLER
            {
                Stage::Vertex
            } else {
                Stage::Fragment
            };
            out.first = head.first.get();
            out.count = head.count.get();
            if out.count == 0 {
                return Err(DecodeStatus::ErrBadLength);
            }
            match bind_record_len(out.count, REF_BIND_ENTRY_SIZE) {
                Some(need) if payload.len() >= need => {}
                _ => return Err(DecodeStatus::ErrShort),
            }
            out.ref_binds.clear();
            for e in entries {
                out.ref_binds.push(e.object_ref.get());
            }
            if let Some(&r) = out.ref_binds.first() {
                if textures {
                    out.texture_ref = r;
                } else {
                    out.sampler_ref = r;
                }
            }
            Ok(out)
        }
        wire::OPCODE_SET_VERTEX_SAMPLER_LOD | wire::OPCODE_SET_FRAGMENT_SAMPLER_LOD => {
            // LOD clamps are not lifted (nothing applies them), but the bind
            // itself is read through the wire layout rather than dropped.
            let (head, entries) =
                wire::sampler_lod_binds(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::SetSampler;
            out.stage = if opcode == wire::OPCODE_SET_VERTEX_SAMPLER_LOD {
                Stage::Vertex
            } else {
                Stage::Fragment
            };
            out.has_sampler_lod = true;
            out.first = head.first.get();
            out.count = head.count.get();
            if out.count == 0 {
                return Err(DecodeStatus::ErrBadLength);
            }
            match bind_record_len(out.count, SAMPLER_LOD_BIND_ENTRY_SIZE) {
                Some(need) if payload.len() >= need => {}
                _ => return Err(DecodeStatus::ErrShort),
            }
            out.ref_binds.clear();
            for e in entries {
                out.ref_binds.push(e.sampler_ref.get());
            }
            if let Some(&r) = out.ref_binds.first() {
                out.sampler_ref = r;
            }
            Ok(out)
        }
        wire::OPCODE_DRAW => {
            // Compact `drawPrimitives:vertexStart:vertexCount:`: an 8-byte
            // payload of `u32 primitiveType · u16 vertexStart · u16 vertexCount`.
            //
            // This used to read four u32s behind `payload.len() < 16`, which is
            // neither of the selector's two forms. The only test for it was a
            // synthetic 24-byte fixture built to match the code, so nothing
            // caught it — and every live compact draw was rejected `ErrShort`
            // and dropped. Silently, until the decode refusal was named: one
            // fired on the first arm64 boot that could report it. The layout is
            // now `reims_vgpu_wire::ops::render::Draw`, pinned by fixtures
            // `render_draw_primitives` and `render_draw_primitives_strip`.
            if command_length != DRAW_COMPACT_CMD_LEN {
                return Err(DecodeStatus::ErrBadLength);
            }
            let d = wire::draw(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::Draw;
            out.primitive_type = d.primitive_type.get();
            out.vertex_start = d.vertex_start.get() as u32;
            out.vertex_count = d.vertex_count.get() as u32;
            // Not on the wire: this selector is the non-instanced one, and
            // Metal draws it once.
            out.instance_count = 1;
            Ok(out)
        }
        // Wide `drawPrimitives:vertexStart:vertexCount:`, which the guest emits
        // instead of `0x01` when either count exceeds 16 bits.
        //
        // This arm used to decline by name, and it was right to: the layout it
        // would have guessed — `u64 · u64 · u32 primitiveType@0x10`, by analogy
        // with the wide instanced siblings — is wrong. `primitiveType` leads and
        // is 32-bit, exactly as in the compact form. Fixtures
        // `render_draw_primitives_wide`, `..._count_over_16bit` and
        // `..._start_over_16bit` settle it.
        wire::OPCODE_DRAW_WIDE => {
            if command_length != wire::DRAW_WIDE_TOTAL_LEN as usize {
                return Err(DecodeStatus::ErrBadLength);
            }
            let d = wire::draw_wide(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::Draw;
            out.primitive_type = d.primitive_type.get();
            out.vertex_start = narrow_count(d.vertex_start.get())?;
            out.vertex_count = narrow_count(d.vertex_count.get())?;
            out.instance_count = 1;
            Ok(out)
        }
        // Compact `drawPrimitives:vertexStart:vertexCount:instanceCount:`.
        //
        // The layout is DISTINCT from the `0x01` form — the counts lead and
        // `primitiveType` is last and 16-bit. Derived here from live x86 WebKit
        // bytes (`00000400 0d000400` = vs0 vc4 inst13 primTriStrip) before the
        // oracle existed, and `render_draw_primitives_instanced` later agreed
        // field for field. This is WebKit's instanced glyph/rect batch; the
        // non-instanced `0x01` and indexed `0x07` forms render chrome text,
        // which is why chrome rendered while page content stayed blank.
        wire::OPCODE_DRAW_INSTANCED => {
            let d = wire::draw_instanced(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::Draw;
            out.vertex_start = d.vertex_start.get() as u32;
            out.vertex_count = d.vertex_count.get() as u32;
            out.instance_count = (d.instance_count.get() as u32).max(1);
            out.primitive_type = d.primitive_type.get() as u32;
            Ok(out)
        }
        // Wide `drawPrimitives:…:instanceCount:` — the counts widen together
        // when any one of them passes 16 bits, so the whole record is 64-bit
        // even where two of the three would have fitted.
        wire::OPCODE_DRAW_INSTANCED_WIDE => {
            if command_length != wire::DRAW_INSTANCED_WIDE_TOTAL_LEN as usize {
                return Err(DecodeStatus::ErrBadLength);
            }
            let d = wire::draw_instanced_wide(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::Draw;
            out.vertex_start = narrow_count(d.vertex_start.get())?;
            out.vertex_count = narrow_count(d.vertex_count.get())?;
            out.instance_count = narrow_count(d.instance_count.get())?.max(1);
            out.primitive_type = d.primitive_type.get() as u32;
            Ok(out)
        }
        // `drawPrimitives:…:instanceCount:baseInstance:`, both encodings.
        //
        // Neither was decoded at all until now: both fall inside the accepted
        // window, so they reached `Kind::OtherAccepted` and executed nothing —
        // an entire Metal draw selector dropped, wearing the shape of an
        // accepted state-set.
        wire::OPCODE_DRAW_INSTANCED_BASE => {
            if command_length != wire::DRAW_INSTANCED_BASE_TOTAL_LEN as usize {
                return Err(DecodeStatus::ErrBadLength);
            }
            let d = wire::draw_instanced_base(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::Draw;
            out.vertex_start = d.vertex_start.get() as u32;
            out.vertex_count = d.vertex_count.get() as u32;
            out.instance_count = (d.instance_count.get() as u32).max(1);
            out.base_instance = d.base_instance.get() as u32;
            out.primitive_type = d.primitive_type.get() as u32;
            Ok(out)
        }
        wire::OPCODE_DRAW_INSTANCED_BASE_WIDE => {
            if command_length != wire::DRAW_INSTANCED_BASE_WIDE_TOTAL_LEN as usize {
                return Err(DecodeStatus::ErrBadLength);
            }
            let d = wire::draw_instanced_base_wide(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::Draw;
            out.vertex_start = narrow_count(d.vertex_start.get())?;
            out.vertex_count = narrow_count(d.vertex_count.get())?;
            out.instance_count = narrow_count(d.instance_count.get())?.max(1);
            out.base_instance = narrow_count(d.base_instance.get())?;
            out.primitive_type = d.primitive_type.get() as u32;
            Ok(out)
        }
        wire::OPCODE_DRAW_INDEXED_WIDE => {
            // The wide indexed form. This arm's head was already right; what it
            // called `u32 indexCount@8, u32 pad@0xc` and `u32
            // indexBufferOffset@0x10, u32 pad@0x14` are the two halves of two
            // 64-bit fields, which reads the same below 2³² and differently
            // above it. Fixtures `render_draw_indexed_count_over_16bit` and
            // `..._offset_over_16bit`.
            if command_length != wire::DRAW_INDEXED_WIDE_TOTAL_LEN as usize {
                return Err(DecodeStatus::ErrBadLength);
            }
            let d = wire::draw_indexed_wide(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::Draw;
            out.primitive_type = d.primitive_type.get() as u32;
            out.index_type = d.index_type.get() as u32;
            out.index_buffer_ref = d.index_buffer_ref.get();
            out.index_count = narrow_count(d.index_count.get())?;
            out.index_buffer_offset = d.index_buffer_offset.get();
            out.instance_count = 1;
            Ok(out)
        }
        // Compact indexed draws: `0x07` is 20 bytes on the wire and `0x09` is
        // 24, the second appending a 16-bit instance count.
        //
        // Two things here were wrong and could not be seen from a boot. The
        // first four bytes were read as one `u32 primitiveType`, which absorbs
        // `indexType` at `+2`; and `index_type` was then hardcoded to
        // `MTLIndexTypeUInt16`. Both are right exactly while the guest uses
        // 16-bit indices, because that ordinal is 0. Fixture
        // `render_draw_indexed_uint32` is the case that separates them: with
        // `MTLIndexTypeUInt32` the word reads `04 00 01 00`, so the old arm
        // produced `primitiveType = 0x10004` — no such Metal primitive —
        // alongside a 32-bit index buffer drawn as 16-bit.
        //
        // The `payload.len() >= 28` branch that used to sit in front of this is
        // gone. These two records are 12 and 16 bytes of payload and never
        // anything else; the wide forms are separate opcodes (`0x06`, `0x08`).
        // It was unreachable, and the layout it carried was an invention.
        wire::OPCODE_DRAW_INDEXED | wire::OPCODE_DRAW_INDEXED_INSTANCED => {
            out.kind = Kind::Draw;
            // Shared compact body; the instanced form only adds instance_count.
            // Use the layout view (not draw_indexed's opcode assert) so both arms share it.
            let d = wire_view::<wire::DrawIndexed>(payload)?;
            out.primitive_type = d.primitive_type.get() as u32;
            out.index_type = d.index_type.get() as u32;
            out.index_buffer_ref = d.index_buffer_ref.get();
            out.index_count = d.index_count.get() as u32;
            out.index_buffer_offset = d.index_buffer_offset.get() as u64;
            out.instance_count = if opcode == wire::OPCODE_DRAW_INDEXED_INSTANCED {
                let i = wire::draw_indexed_instanced(&op).map_err(|_| DecodeStatus::ErrShort)?;
                (i.instance_count.get() as u32).max(1)
            } else {
                1
            };
            Ok(out)
        }
        wire::OPCODE_DRAW_INDEXED_INSTANCED_WIDE => {
            if command_length != wire::DRAW_INDEXED_INSTANCED_WIDE_TOTAL_LEN as usize {
                return Err(DecodeStatus::ErrBadLength);
            }
            let d = wire::draw_indexed_instanced_wide(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::Draw;
            out.primitive_type = d.primitive_type.get() as u32;
            out.index_type = d.index_type.get() as u32;
            out.index_buffer_ref = d.index_buffer_ref.get();
            out.index_count = narrow_count(d.index_count.get())?;
            out.index_buffer_offset = d.index_buffer_offset.get();
            out.instance_count = narrow_count(d.instance_count.get())?.max(1);
            Ok(out)
        }
        // The full indexed draw, with a base vertex and a base instance.
        //
        // These two are the *only* records in the family that put the buffer
        // offset before the index count. Reading them with the siblings' order
        // swaps a guest's index count and its buffer offset, which draws from
        // the wrong place in the wrong amount — so the field order here is not
        // a copy of the arm above and must not be made one.
        wire::OPCODE_DRAW_INDEXED_INSTANCED_BASE => {
            if command_length != wire::DRAW_INDEXED_INSTANCED_BASE_TOTAL_LEN as usize {
                return Err(DecodeStatus::ErrBadLength);
            }
            let d = wire::draw_indexed_instanced_base(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::Draw;
            out.primitive_type = d.primitive_type.get() as u32;
            out.index_type = d.index_type.get() as u32;
            out.index_buffer_ref = d.index_buffer_ref.get();
            out.index_count = d.index_count.get() as u32;
            out.index_buffer_offset = d.index_buffer_offset.get() as u64;
            out.instance_count = (d.instance_count.get() as u32).max(1);
            out.base_instance = d.base_instance.get() as u32;
            out.base_vertex = d.base_vertex.get() as i64;
            Ok(out)
        }
        wire::OPCODE_DRAW_INDEXED_INSTANCED_BASE_WIDE => {
            if command_length != wire::DRAW_INDEXED_INSTANCED_BASE_WIDE_TOTAL_LEN as usize {
                return Err(DecodeStatus::ErrBadLength);
            }
            let d =
                wire::draw_indexed_instanced_base_wide(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::Draw;
            out.primitive_type = d.primitive_type.get() as u32;
            out.index_type = d.index_type.get() as u32;
            out.index_buffer_ref = d.index_buffer_ref.get();
            out.index_count = narrow_count(d.index_count.get())?;
            out.index_buffer_offset = d.index_buffer_offset.get();
            out.instance_count = narrow_count(d.instance_count.get())?.max(1);
            out.base_instance = narrow_count(d.base_instance.get())?;
            out.base_vertex = d.base_vertex.get();
            Ok(out)
        }
        wire::OPCODE_SET_VERTEX_BUFFER_OFFSET | wire::OPCODE_SET_FRAGMENT_BUFFER_OFFSET => {
            let b = wire::buffer_offset(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::SetBufferOffset;
            out.stage = if opcode == wire::OPCODE_SET_FRAGMENT_BUFFER_OFFSET {
                Stage::Fragment
            } else {
                Stage::Vertex
            };
            out.first = b.index.get();
            out.buffer_offset = b.offset.get();
            Ok(out)
        }
        wire::OPCODE_SET_VERTEX_BUFFER_OFFSET_STRIDE => {
            let b = wire::buffer_offset_stride(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::SetBufferOffset;
            out.stage = Stage::Vertex;
            out.has_attribute_stride = true;
            out.first = b.index.get();
            out.buffer_offset = b.offset.get();
            Ok(out)
        }
        wire::OPCODE_SET_VIEWPORT => {
            let v = wire::set_viewport(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::SetViewport;
            out.count = 1;
            out.viewport = [
                v.origin_x.get(),
                v.origin_y.get(),
                v.width.get(),
                v.height.get(),
                v.znear.get(),
                v.zfar.get(),
            ];
            Ok(out)
        }
        wire::OPCODE_SET_VIEWPORTS => {
            // Count is kept: the executor models one viewport; further rects are
            // a named loss. First viewport is still lifted.
            let (head, ports) = wire::set_viewports(&op).map_err(|_| DecodeStatus::ErrShort)?;
            let count = head.count.get();
            if count == 0 {
                return Err(DecodeStatus::ErrBadLength);
            }
            out.kind = Kind::SetViewport;
            out.count = count;
            let v = ports.first().ok_or(DecodeStatus::ErrShort)?;
            out.viewport = [
                v.origin_x.get(),
                v.origin_y.get(),
                v.width.get(),
                v.height.get(),
                v.znear.get(),
                v.zfar.get(),
            ];
            Ok(out)
        }
        wire::OPCODE_SET_SCISSOR => {
            let r = wire::set_scissor(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::SetScissor;
            out.count = 1;
            out.scissor_x = r.x.get() as u32;
            out.scissor_y = r.y.get() as u32;
            out.scissor_w = r.width.get() as u32;
            out.scissor_h = r.height.get() as u32;
            Ok(out)
        }
        wire::OPCODE_SET_SCISSOR_RECTS => {
            let (head, rects) = wire::set_scissor_rects(&op).map_err(|_| DecodeStatus::ErrShort)?;
            let count = head.count.get();
            if count == 0 {
                return Err(DecodeStatus::ErrBadLength);
            }
            let Ok(count) = u32::try_from(count) else {
                return Err(DecodeStatus::ErrCountOutOfRange);
            };
            let r = rects.first().ok_or(DecodeStatus::ErrShort)?;
            out.kind = Kind::SetScissor;
            out.count = count;
            out.scissor_x = r.x.get() as u32;
            out.scissor_y = r.y.get() as u32;
            out.scissor_w = r.width.get() as u32;
            out.scissor_h = r.height.get() as u32;
            Ok(out)
        }
        wire::OPCODE_SET_BLEND_COLOR => {
            let b = wire::set_blend_color(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::SetBlendColor;
            out.has_blend_color = true;
            out.blend_color = [b.red.get(), b.green.get(), b.blue.get(), b.alpha.get()];
            Ok(out)
        }
        wire::OPCODE_SET_CULL_MODE => {
            let m = wire::set_cull_mode(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::SetCullMode;
            out.has_cull_mode = true;
            out.cull_mode = m.mode.get() as u32;
            Ok(out)
        }
        wire::OPCODE_SET_FRONT_FACING => {
            let m = wire::set_front_facing(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::SetFrontFacing;
            out.has_front_facing = true;
            out.front_facing = m.mode.get() as u32;
            Ok(out)
        }
        wire::OPCODE_SET_DEPTH_BIAS => {
            let d = wire::set_depth_bias(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::SetDepthBias;
            out.has_depth_bias = true;
            out.depth_bias = [d.bias.get(), d.slope_scale.get(), d.clamp.get()];
            Ok(out)
        }
        wire::OPCODE_SET_DEPTH_STENCIL_STATE => {
            let r = wire::state_ref(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::SetDepthStencil;
            out.depth_stencil_ref = r.object_ref.get();
            Ok(out)
        }
        wire::OPCODE_SET_STENCIL_REFERENCE => {
            let s = wire::set_stencil_reference(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::SetStencilReference;
            out.has_stencil_ref = true;
            out.stencil_ref_front = s.front.get();
            out.stencil_ref_back = s.back.get();
            Ok(out)
        }
        wire::OPCODE_UPDATE_FENCE | wire::OPCODE_WAIT_FOR_FENCE => {
            let f = wire::fence(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::Fence;
            out.fence_ref = f.fence_ref.get();
            Ok(out)
        }
        wire::OPCODE_USE_RESOURCE => {
            // Refs are not lifted (exec no-ops residency); count bounds the
            // record via the wire layout (usage+stages pack to 4 bytes, refs
            // at +8).
            let (head, refs) = wire::use_resource(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::UseResource;
            out.count = head.count.get();
            if out.count as usize != refs.len() {
                return Err(DecodeStatus::ErrShort);
            }
            Ok(out)
        }
        wire::OPCODE_USE_HEAP => {
            // Heap form: no usage word, stages u16, refs at +6 (align-1).
            let (head, refs) = wire::use_heap(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::UseHeap;
            out.count = head.count.get();
            if out.count as usize != refs.len() {
                return Err(DecodeStatus::ErrShort);
            }
            Ok(out)
        }
        wire::OPCODE_MEMORY_BARRIER_RESOURCES
        | wire::OPCODE_MEMORY_BARRIER_SCOPE
        | wire::OPCODE_TEXTURE_BARRIER => {
            out.kind = Kind::Barrier;
            Ok(out)
        }
        wire::OPCODE_SET_DEPTH_CLIP_MODE | wire::OPCODE_SET_TRIANGLE_FILL_MODE => {
            let m = wire::mode_state(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::SetRasterState;
            out.mode = m.mode.get();
            Ok(out)
        }
        wire::OPCODE_DRAW_INDIRECT => {
            if command_length != wire::DRAW_INDIRECT_TOTAL_LEN as usize {
                return Err(DecodeStatus::ErrBadLength);
            }
            let d = wire::draw_indirect(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::DrawIndirect;
            // 16 bits here, where the direct draws give the same field 32. The
            // two bytes above it are never written by the serializer, so a
            // wider read takes the guest's stale ring.
            out.primitive_type = d.primitive_type.get() as u32;
            out.indirect_buffer_ref = d.indirect_buffer_ref.get();
            out.indirect_buffer_offset = d.indirect_buffer_offset.get();
            Ok(out)
        }
        wire::OPCODE_DRAW_INDEXED_INDIRECT => {
            if command_length != wire::DRAW_INDEXED_INDIRECT_TOTAL_LEN as usize {
                return Err(DecodeStatus::ErrBadLength);
            }
            let d = wire::draw_indexed_indirect(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::DrawIndirect;
            out.primitive_type = d.primitive_type.get() as u32;
            // Its own 16-bit field beside `primitive_type`, not the upper half
            // of a 32-bit one — reading a `u32` at `+0` would absorb it, which
            // is the bug the compact indexed draw had.
            out.index_type = d.index_type.get() as u32;
            out.index_buffer_ref = d.index_buffer_ref.get();
            out.index_buffer_offset = d.index_buffer_offset.get();
            out.indirect_buffer_ref = d.indirect_buffer_ref.get();
            out.indirect_buffer_offset = d.indirect_buffer_offset.get();
            Ok(out)
        }
        wire_tile::OPCODE_SET_TILE_THREADGROUP_MEMORY => {
            if command_length != wire_tile::SET_TILE_THREADGROUP_MEMORY_TOTAL_LEN as usize {
                return Err(DecodeStatus::ErrBadLength);
            }
            let m = wire_tile::tile_threadgroup_memory(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::TileBind;
            out.first = m.index.get();
            out.count = 1;
            // The length and the offset are read by the view and then not
            // lifted, exactly as the other tile binds' entries are not: nothing
            // downstream allocates imageblock memory, so a field carrying the
            // size would have no consumer. `tile_threads` in particular is a
            // dispatch's grid and must not be borrowed for it.
            Ok(out)
        }
        wire_tile::OPCODE_SET_TILE_BUFFER => {
            // Entries are not lifted (no tile table consumer); first/count and
            // length come from the wire bind walk.
            let (head, _entries) =
                wire_tile::tile_buffer_binds(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::TileBind;
            out.first = head.first.get();
            out.count = head.count.get();
            if out.count == 0 {
                return Err(DecodeStatus::ErrBadLength);
            }
            match bind_record_len(out.count, BUFFER_BIND_ENTRY_SIZE) {
                Some(need) if payload.len() >= need => Ok(out),
                _ => Err(DecodeStatus::ErrShort),
            }
        }
        wire_tile::OPCODE_SET_TILE_TEXTURE => {
            let (head, _entries) =
                wire_tile::tile_texture_binds(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::TileBind;
            out.first = head.first.get();
            out.count = head.count.get();
            if out.count == 0 {
                return Err(DecodeStatus::ErrBadLength);
            }
            match bind_record_len(out.count, REF_BIND_ENTRY_SIZE) {
                Some(need) if payload.len() >= need => Ok(out),
                _ => Err(DecodeStatus::ErrShort),
            }
        }
        wire_tile::OPCODE_SET_TILE_SAMPLER => {
            let (head, _entries) =
                wire_tile::tile_sampler_binds(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::TileBind;
            out.first = head.first.get();
            out.count = head.count.get();
            if out.count == 0 {
                return Err(DecodeStatus::ErrBadLength);
            }
            match bind_record_len(out.count, REF_BIND_ENTRY_SIZE) {
                Some(need) if payload.len() >= need => Ok(out),
                _ => Err(DecodeStatus::ErrShort),
            }
        }
        wire_tile::OPCODE_SET_TILE_SAMPLER_LOD => {
            let (head, _entries) =
                wire_tile::tile_sampler_lod_binds(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::TileBind;
            out.first = head.first.get();
            out.count = head.count.get();
            if out.count == 0 {
                return Err(DecodeStatus::ErrBadLength);
            }
            match bind_record_len(out.count, SAMPLER_LOD_BIND_ENTRY_SIZE) {
                Some(need) if payload.len() >= need => Ok(out),
                _ => Err(DecodeStatus::ErrShort),
            }
        }
        wire_tile::OPCODE_SET_TILE_BUFFER_OFFSET => {
            if command_length != wire_tile::SET_TILE_BUFFER_OFFSET_TOTAL_LEN as usize {
                return Err(DecodeStatus::ErrBadLength);
            }
            let b = wire_tile::tile_buffer_offset(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::TileBind;
            out.first = b.index.get();
            out.count = 1;
            out.buffer_offset = b.offset.get();
            Ok(out)
        }
        wire_tile::OPCODE_DISPATCH_THREADS_PER_TILE => {
            if command_length != wire_tile::DISPATCH_THREADS_PER_TILE_TOTAL_LEN as usize {
                return Err(DecodeStatus::ErrBadLength);
            }
            let d =
                wire_tile::dispatch_threads_per_tile(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::TileDispatch;
            out.tile_threads = [d.width.get(), d.height.get(), d.depth.get()];
            Ok(out)
        }
        wire_tile::OPCODE_DISPATCH_THREADS_PER_TILE_IN_REGION
        | wire_tile::OPCODE_DISPATCH_THREADS_PER_TILE_IN_REGION_RT_INDEX => {
            if command_length != wire_tile::DISPATCH_THREADS_PER_TILE_IN_REGION_TOTAL_LEN as usize {
                return Err(DecodeStatus::ErrBadLength);
            }
            let d = wire_tile::dispatch_threads_per_tile_in_region(&op)
                .map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::TileDispatch;
            out.tile_threads = [d.width.get(), d.height.get(), d.depth.get()];
            // Region / RT index not lifted — see wire tile module.
            Ok(out)
        }
        wire_tile::OPCODE_GET_TILE_DIMENSIONS => {
            if command_length != wire_tile::GET_TILE_DIMENSIONS_TOTAL_LEN as usize {
                return Err(DecodeStatus::ErrBadLength);
            }
            let g = wire_tile::get_tile_dimensions(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::TileDimensionsQuery;
            out.buffer_ref = g.buffer_ref.get();
            out.buffer_offset = g.offset.get();
            Ok(out)
        }
        wire::OPCODE_SET_VERTEX_AMPLIFICATION_MODE => {
            if command_length != wire::SET_VERTEX_AMPLIFICATION_MODE_TOTAL_LEN as usize {
                return Err(DecodeStatus::ErrBadLength);
            }
            let m = wire::vertex_amplification_mode(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::SetVertexAmplification;
            out.mode = m.mode.get() as u64;
            out.amplification_value = m.value.get();
            Ok(out)
        }
        wire::OPCODE_SET_VERTEX_AMPLIFICATION_COUNT => {
            // Four-byte count head (not BindHeader); mappings follow and are
            // not lifted — nothing downstream amplifies. Wire parser bounds
            // entries to the record length.
            let (head, mappings) =
                wire::vertex_amplification_count(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::SetVertexAmplification;
            out.count = head.count.get();
            if out.count as usize != mappings.len() {
                return Err(DecodeStatus::ErrShort);
            }
            let _ = mappings; // unlifted by design
            Ok(out)
        }
        wire::OPCODE_SET_VISIBILITY_RESULT_MODE => {
            if command_length != wire::SET_VISIBILITY_RESULT_MODE_TOTAL_LEN as usize {
                return Err(DecodeStatus::ErrBadLength);
            }
            let v = wire::set_visibility_result_mode(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::SetVisibilityResultMode;
            // Offset first, mode second. See `wire::OPCODE_SET_VISIBILITY_RESULT_MODE`.
            out.visibility_result_offset = v.offset.get();
            out.mode = v.mode.get();
            Ok(out)
        }
        wire::OPCODE_SET_LINE_WIDTH | wire::OPCODE_SET_TESSELLATION_FACTOR_SCALE => {
            let f = wire::float_state(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::SetFloatState;
            out.float_value = f.value.get();
            Ok(out)
        }
        wire::OPCODE_SET_COLOR_STORE_ACTION => {
            let a = wire::set_color_store_action(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::SetStoreAction;
            out.mode = u64::from(a.store_action.get());
            out.first = a.index.get();
            Ok(out)
        }
        wire::OPCODE_SET_DEPTH_STORE_ACTION | wire::OPCODE_SET_STENCIL_STORE_ACTION => {
            // Depth/stencil store actions share the one-NSUInteger mode shape.
            let m = wire::mode_state(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::SetStoreAction;
            out.mode = m.mode.get();
            Ok(out)
        }
        wire::OPCODE_SET_COLOR_STORE_ACTION_OPTIONS
        | wire::OPCODE_SET_DEPTH_STORE_ACTION_OPTIONS
        | wire::OPCODE_SET_STENCIL_STORE_ACTION_OPTIONS => {
            // The same three-attachment split one opcode higher, and the widths
            // do *not* carry over: the options are a `u64` where the store
            // action is a `u32`, so the colour form's index sits at `+8` rather
            // than `+4` and the record is 20 bytes rather than 16.
            if opcode == wire::OPCODE_SET_COLOR_STORE_ACTION_OPTIONS {
                if command_length != wire::SET_COLOR_STORE_ACTION_OPTIONS_TOTAL_LEN as usize {
                    return Err(DecodeStatus::ErrBadLength);
                }
                let a = wire::set_color_store_action_options(&op)
                    .map_err(|_| DecodeStatus::ErrShort)?;
                out.mode = a.options.get();
                out.first = a.index.get();
            } else {
                if command_length != wire::SET_STORE_ACTION_OPTIONS_TOTAL_LEN as usize {
                    return Err(DecodeStatus::ErrBadLength);
                }
                let a = wire::set_store_action_options(&op).map_err(|_| DecodeStatus::ErrShort)?;
                out.mode = a.options.get();
            }
            out.kind = Kind::SetStoreActionOptions;
            Ok(out)
        }
        wire::OPCODE_DRAW_PATCHES
        | wire::OPCODE_DRAW_PATCHES_WIDE
        | wire::OPCODE_DRAW_INDEXED_PATCHES
        | wire::OPCODE_DRAW_PATCHES_INDIRECT
        | wire::OPCODE_DRAW_INDEXED_PATCHES_INDIRECT => {
            // Six records across five opcodes, because `0x0c` is two: the plain
            // wide draw at 56 bytes and the indexed wide draw at 68. Dispatched
            // on the length rather than guessed, and a `0x0c` at any other
            // length is refused rather than read as whichever is closer -- the
            // two bodies disagree from their tenth byte on.
            //
            // Every form is length-checked exactly, so a truncated patch draw
            // is an `ErrBadLength` rather than a draw with invented counts.
            //
            // None of the fields is lifted. Nothing here tessellates, so a
            // `patch_count` in `Command` would be a producer with no consumer;
            // what `runtime::exec` needs is that a patch draw happened and
            // which form, and the opcode carries both.
            let want = match opcode {
                wire::OPCODE_DRAW_PATCHES => wire::DRAW_PATCHES_TOTAL_LEN as usize,
                wire::OPCODE_DRAW_INDEXED_PATCHES => wire::DRAW_INDEXED_PATCHES_TOTAL_LEN as usize,
                wire::OPCODE_DRAW_PATCHES_INDIRECT => {
                    wire::DRAW_PATCHES_INDIRECT_TOTAL_LEN as usize
                }
                wire::OPCODE_DRAW_INDEXED_PATCHES_INDIRECT => {
                    wire::DRAW_INDEXED_PATCHES_INDIRECT_TOTAL_LEN as usize
                }
                // `0x0c`: the two wide forms, and nothing else.
                _ => match command_length {
                    n if n == wire::DRAW_PATCHES_WIDE_TOTAL_LEN as usize => n,
                    n if n == wire::DRAW_INDEXED_PATCHES_WIDE_TOTAL_LEN as usize => n,
                    _ => return Err(DecodeStatus::ErrBadLength),
                },
            };
            if command_length != want {
                return Err(DecodeStatus::ErrBadLength);
            }
            // Viewed, so a record whose declared length outran its bytes is
            // refused here rather than by whoever reads it next.
            match opcode {
                wire::OPCODE_DRAW_PATCHES => {
                    wire::draw_patches(&op).map_err(|_| DecodeStatus::ErrShort)?;
                }
                wire::OPCODE_DRAW_INDEXED_PATCHES => {
                    wire::draw_indexed_patches(&op).map_err(|_| DecodeStatus::ErrShort)?;
                }
                wire::OPCODE_DRAW_PATCHES_INDIRECT => {
                    wire::draw_patches_indirect(&op).map_err(|_| DecodeStatus::ErrShort)?;
                }
                wire::OPCODE_DRAW_INDEXED_PATCHES_INDIRECT => {
                    wire::draw_indexed_patches_indirect(&op).map_err(|_| DecodeStatus::ErrShort)?;
                }
                _ if command_length == wire::DRAW_PATCHES_WIDE_TOTAL_LEN as usize => {
                    wire::draw_patches_wide(&op).map_err(|_| DecodeStatus::ErrShort)?;
                }
                _ => {
                    wire::draw_indexed_patches_wide(&op).map_err(|_| DecodeStatus::ErrShort)?;
                }
            }
            out.kind = Kind::DrawPatches;
            Ok(out)
        }
        wire::OPCODE_SET_TESSELLATION_FACTOR_BUFFER => {
            // Not a bind: one buffer per encoder, so there is no slot and no
            // count -- the ref and its two `u64` sit directly in the payload.
            if command_length != wire::SET_TESSELLATION_FACTOR_BUFFER_TOTAL_LEN as usize {
                return Err(DecodeStatus::ErrBadLength);
            }
            let t =
                wire::set_tessellation_factor_buffer(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::SetTessellationFactorBuffer;
            out.buffer_ref = t.buffer_ref.get();
            out.buffer_offset = t.offset.get();
            Ok(out)
        }
        wire::OPCODE_EXECUTE_COMMANDS_INDIRECT => {
            if command_length != wire::EXECUTE_COMMANDS_INDIRECT_TOTAL_LEN as usize {
                return Err(DecodeStatus::ErrBadLength);
            }
            let e = wire::execute_commands_indirect(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::ExecuteCommands;
            out.icb_is_range = false;
            out.indirect_command_buffer_ref = e.icb_ref.get();
            out.icb_args_buffer_ref = e.indirect_buffer_ref.get();
            out.icb_args_buffer_offset = e.indirect_buffer_offset.get();
            Ok(out)
        }
        wire::OPCODE_EXECUTE_COMMANDS_RANGE => {
            if command_length != wire::EXECUTE_COMMANDS_RANGE_TOTAL_LEN as usize {
                return Err(DecodeStatus::ErrBadLength);
            }
            let e = wire::execute_commands_range(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::ExecuteCommands;
            out.icb_is_range = true;
            out.indirect_command_buffer_ref = e.icb_ref.get();
            out.icb_range_location = e.range_location.get();
            out.icb_range_length = e.range_length.get();
            Ok(out)
        }
        wire_pass::OPCODE_RENDER_PASS => {
            if payload.len() < PASS_MIN_PAYLOAD {
                return Err(DecodeStatus::ErrShort);
            }
            out.kind = Kind::RenderPass;
            // Full Apple record: one wire body. Shorter product fixtures still
            // decode through the attachment views (same wire layouts at offsets).
            if let Ok(body) = wire_pass::render_pass(&op) {
                out.depth = depth_from_wire(&body.depth);
                out.stencil = stencil_from_wire(&body.stencil);
                out.color0 = color_from_wire(&body.color[0]);
                out.pass_visibility_result_buffer_ref = body.visibility_result_buffer_ref.get();
                out.pass_render_target_array_length = body.render_target_array_length.get();
                out.pass_render_target_width = body.render_target_width.get();
                out.pass_render_target_height = body.render_target_height.get();
            } else {
                out.depth = decode_depth_attachment(payload);
                out.stencil = decode_stencil_attachment(payload);
                out.color0 = decode_color_attachment(payload, 0);
            }
            if out.color0.texture_ref != 0 {
                out.texture_ref = out.color0.texture_ref;
            }
            Ok(out)
        }
        wire_pass::OPCODE_RASTERIZATION_RATE_MAP => {
            let r = wire_pass::pass_rate_map(&op).map_err(|_| DecodeStatus::ErrBadLength)?;
            out.kind = Kind::RenderPassProperty;
            out.texture_ref = r.rate_map_ref.get();
            Ok(out)
        }
        wire_pass::OPCODE_DEFAULT_RASTER_SAMPLE_COUNT => {
            let c = wire_pass::default_raster_sample_count(&op)
                .map_err(|_| DecodeStatus::ErrBadLength)?;
            out.kind = Kind::RenderPassProperty;
            out.mode = u64::from(c.count.get());
            Ok(out)
        }
        wire_pass::OPCODE_IMAGEBLOCK_SAMPLE_LENGTH
        | wire_pass::OPCODE_THREADGROUP_MEMORY_LENGTH => {
            let m = wire_pass::tile_memory(&op).map_err(|_| DecodeStatus::ErrBadLength)?;
            out.kind = Kind::RenderPassProperty;
            out.mode = u64::from(m.length.get());
            Ok(out)
        }
        wire_pass::OPCODE_TILE_SIZE => {
            let s = wire_pass::tile_size(&op).map_err(|_| DecodeStatus::ErrBadLength)?;
            out.kind = Kind::RenderPassProperty;
            // width | height << 16 — product packing, not a second layout.
            out.mode = u64::from(s.width.get()) | (u64::from(s.height.get()) << 16);
            Ok(out)
        }
        wire_pass::OPCODE_SAMPLE_POSITIONS => {
            let (head, positions) =
                wire_pass::sample_positions(&op).map_err(|_| DecodeStatus::ErrBadLength)?;
            out.kind = Kind::RenderPassProperty;
            out.count = head.count.get();
            if out.count as usize != positions.len() {
                return Err(DecodeStatus::ErrBadLength);
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
    fn every_render_decode_failure_names_its_own_check() {
        use crate::observe::Refusal;
        const ERRS: &[DecodeStatus] = &[
            DecodeStatus::ErrShort,
            DecodeStatus::ErrUnknownOpcode,
            DecodeStatus::ErrUnsupportedOpcode,
            DecodeStatus::ErrBadLength,
        ];
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
        let mut v = hdr(wire::OPCODE_SET_RENDER_PIPELINE_STATE, 12);
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
    /// These payload bytes are the contract's, from the encoder's field order
    /// plus the checked-in corpus record: `03 00 00 00 00 00 06 00` = triangle list,
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
        let mut wide = hdr(wire::OPCODE_DRAW, 24);
        st32(&mut wide[8..], 3);
        assert_eq!(decode(&wide), Err(DecodeStatus::ErrBadLength));
        let short = hdr(wire::OPCODE_DRAW, 12);
        assert_eq!(decode(&short), Err(DecodeStatus::ErrBadLength));
    }

    /// The wide form is a *different opcode*, emitted when either count does
    /// not fit 16 bits, and it keeps the compact form's field order rather than
    /// the instanced forms': `primitiveType` leads and is 32-bit.
    ///
    /// This arm used to decline by name, which lost the draw but said so. The
    /// layout its comment proposed by analogy — two `u64`s then a trailing
    /// `primitiveType` — was the wrong one of the two candidates, so decoding on
    /// that reasoning would have drawn a wrong primitive from a wrong offset.
    /// The bytes below are `reims-vgpu-wire`'s `render_draw_primitives_wide`
    /// fixture shape: Apple's serializer emits exactly this for
    /// `(Triangle, start 0x11111, count 0x22222)`.
    #[test]
    fn the_wide_draw_form_decodes_with_the_compact_forms_field_order() {
        use crate::contract::endian::st64;

        let mut v = hdr(wire::OPCODE_DRAW_WIDE, wire::DRAW_WIDE_TOTAL_LEN as usize);
        st32(&mut v[8..], 3); // primitiveType, 32-bit and FIRST
        st64(&mut v[12..], 0x11111); // vertexStart
        st64(&mut v[20..], 0x22222); // vertexCount
        let c = decode(&v).expect("the wide draw decodes");
        assert_eq!(c.kind, Kind::Draw);
        assert_eq!(c.primitive_type, 3);
        assert_eq!(c.vertex_start, 0x11111);
        assert_eq!(c.vertex_count, 0x22222);
        assert_eq!(c.instance_count, 1, "the non-instanced selector draws once");

        // Reading it with the instanced forms' order would have taken
        // `primitiveType` from the last four bytes, which are a count's high
        // half and read zero. A regression to that guess shows up here.
        assert_ne!(c.primitive_type, 0);

        let short = hdr(
            wire::OPCODE_DRAW_WIDE,
            wire::DRAW_WIDE_TOTAL_LEN as usize - 4,
        );
        assert_eq!(decode(&short), Err(DecodeStatus::ErrBadLength));
    }

    /// A wide count above 32 bits is refused rather than truncated.
    ///
    /// `Command` carries 32-bit counts, and the wide encoding exists because a
    /// value passed 16 bits, not 32 — so this cannot arise from a real draw.
    /// Truncating would silently draw different geometry, which is the class
    /// this decoder's named refusals exist to prevent.
    #[test]
    fn a_wide_count_that_cannot_fit_the_commands_field_is_refused_not_truncated() {
        use crate::contract::endian::st64;

        let mut v = hdr(wire::OPCODE_DRAW_WIDE, wire::DRAW_WIDE_TOTAL_LEN as usize);
        st32(&mut v[8..], 3);
        st64(&mut v[12..], 0);
        st64(&mut v[20..], 0x1_0000_0000);
        assert_eq!(decode(&v), Err(DecodeStatus::ErrCountOutOfRange));
    }

    /// The compact indexed draw carries its index type on the wire, in the two
    /// bytes that used to be read as the upper half of `primitiveType`.
    ///
    /// Both readings agree while the guest uses 16-bit indices, because
    /// `MTLIndexTypeUInt16` is ordinal 0. With `MTLIndexTypeUInt32` the head
    /// reads `04 00 01 00`, and the old arm produced `primitiveType = 0x10004`
    /// — no such Metal primitive — while separately reporting UInt16 for a
    /// 32-bit index buffer. Apple's serializer emits exactly these bytes for
    /// `(TriangleStrip, count 0x1111, UInt32, ref, offset 0x2222)`; the wire
    /// crate's `render_draw_indexed_uint32` fixture is the same capture.
    #[test]
    fn a_compact_indexed_draw_reads_its_index_type_from_the_wire() {
        use crate::contract::endian::st16;

        let mut v = hdr(
            wire::OPCODE_DRAW_INDEXED,
            wire::DRAW_INDEXED_TOTAL_LEN as usize,
        );
        st16(&mut v[8..], 4); // primitiveType, 16-bit
        st16(&mut v[10..], 1); // indexType UInt32
        st32(&mut v[12..], 0x141f); // index buffer ref
        st16(&mut v[16..], 0x1111); // index count
        st16(&mut v[18..], 0x2222); // index buffer offset

        let c = decode(&v).expect("compact indexed draw");
        assert_eq!(c.kind, Kind::Draw);
        assert_eq!(
            c.primitive_type, 4,
            "primitiveType must not absorb indexType"
        );
        assert_eq!(c.index_type, 1, "UInt32 must survive rather than reading 0");
        assert_eq!(c.index_buffer_ref, 0x141f);
        assert_eq!(c.index_count, 0x1111);
        assert_eq!(c.index_buffer_offset, 0x2222);
        assert_eq!(c.instance_count, 1);

        // The instanced sibling appends a 16-bit instance count and changes
        // nothing before it.
        let mut w = hdr(
            wire::OPCODE_DRAW_INDEXED_INSTANCED,
            wire::DRAW_INDEXED_INSTANCED_TOTAL_LEN as usize,
        );
        w[8..20].copy_from_slice(&v[8..20]);
        st16(&mut w[20..], 0x3333);
        let ci = decode(&w).expect("compact instanced indexed draw");
        assert_eq!(ci.primitive_type, 4);
        assert_eq!(ci.index_type, 1);
        assert_eq!(ci.index_count, 0x1111);
        assert_eq!(ci.instance_count, 0x3333);
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

    /// The six draw forms that carry a base instance used to reach
    /// `Kind::OtherAccepted` and execute nothing — two whole Metal selectors
    /// dropped, plus the wide encodings of two more, each wearing the shape of
    /// an accepted state-set.
    ///
    /// This walks all twelve draw opcodes and asserts none of them lands in the
    /// catch-all. A new opcode in `0x00..=0x0b` that nobody decodes fails here
    /// rather than going quiet.
    #[test]
    fn no_draw_opcode_falls_through_to_the_accepted_catch_all() {
        for (opcode, total) in [
            (wire::OPCODE_DRAW_WIDE, wire::DRAW_WIDE_TOTAL_LEN),
            (wire::OPCODE_DRAW, wire::DRAW_TOTAL_LEN),
            (
                wire::OPCODE_DRAW_INSTANCED_WIDE,
                wire::DRAW_INSTANCED_WIDE_TOTAL_LEN,
            ),
            (wire::OPCODE_DRAW_INSTANCED, wire::DRAW_INSTANCED_TOTAL_LEN),
            (
                wire::OPCODE_DRAW_INSTANCED_BASE_WIDE,
                wire::DRAW_INSTANCED_BASE_WIDE_TOTAL_LEN,
            ),
            (
                wire::OPCODE_DRAW_INSTANCED_BASE,
                wire::DRAW_INSTANCED_BASE_TOTAL_LEN,
            ),
            (
                wire::OPCODE_DRAW_INDEXED_WIDE,
                wire::DRAW_INDEXED_WIDE_TOTAL_LEN,
            ),
            (wire::OPCODE_DRAW_INDEXED, wire::DRAW_INDEXED_TOTAL_LEN),
            (
                wire::OPCODE_DRAW_INDEXED_INSTANCED_WIDE,
                wire::DRAW_INDEXED_INSTANCED_WIDE_TOTAL_LEN,
            ),
            (
                wire::OPCODE_DRAW_INDEXED_INSTANCED,
                wire::DRAW_INDEXED_INSTANCED_TOTAL_LEN,
            ),
            (
                wire::OPCODE_DRAW_INDEXED_INSTANCED_BASE_WIDE,
                wire::DRAW_INDEXED_INSTANCED_BASE_WIDE_TOTAL_LEN,
            ),
            (
                wire::OPCODE_DRAW_INDEXED_INSTANCED_BASE,
                wire::DRAW_INDEXED_INSTANCED_BASE_TOTAL_LEN,
            ),
        ] {
            let v = hdr(opcode, total as usize);
            let c = decode(&v).unwrap_or_else(|e| panic!("opcode {opcode:#x} refused: {e:?}"));
            assert_eq!(
                c.kind,
                Kind::Draw,
                "opcode {opcode:#x} is a draw and must not decode as {:?}",
                c.kind
            );
        }
    }

    /// `drawPrimitives:…:instanceCount:baseInstance:`, the compact form.
    ///
    /// Metal offsets `[[instance_id]]` and every per-instance vertex fetch from
    /// `baseInstance`, so dropping it draws the same instance repeatedly rather
    /// than drawing nothing — which is why this was invisible until the
    /// selector was decoded at all.
    #[test]
    fn a_base_instance_draw_carries_its_base_instance() {
        use crate::contract::endian::st16;

        let mut v = hdr(
            wire::OPCODE_DRAW_INSTANCED_BASE,
            wire::DRAW_INSTANCED_BASE_TOTAL_LEN as usize,
        );
        st16(&mut v[8..], 1); // vertexStart
        st16(&mut v[10..], 2); // vertexCount
        st16(&mut v[12..], 3); // instanceCount
        st16(&mut v[14..], 4); // baseInstance
        st16(&mut v[16..], 3); // primitiveType, last and 16-bit
        let c = decode(&v).expect("base-instance draw");
        assert_eq!(c.kind, Kind::Draw);
        assert_eq!(c.vertex_start, 1);
        assert_eq!(c.vertex_count, 2);
        assert_eq!(c.instance_count, 3);
        assert_eq!(c.base_instance, 4);
        assert_eq!(c.primitive_type, 3);
    }

    /// The two indexed forms with a base vertex put the buffer offset BEFORE
    /// the index count, which their four siblings do not.
    ///
    /// Reading them with the siblings' order swaps the two, drawing the wrong
    /// number of indices from the wrong place — and both are plausible values,
    /// so nothing downstream would refuse it. The base vertex is signed, so a
    /// small negative offset must not read as an index near 65535.
    #[test]
    fn the_full_indexed_draw_puts_its_offset_before_its_count_and_signs_its_base_vertex() {
        use crate::contract::endian::st16;

        let mut v = hdr(
            wire::OPCODE_DRAW_INDEXED_INSTANCED_BASE,
            wire::DRAW_INDEXED_INSTANCED_BASE_TOTAL_LEN as usize,
        );
        st16(&mut v[8..], 3); // primitiveType
        st16(&mut v[10..], 1); // indexType UInt32
        st32(&mut v[12..], 0x141f); // index buffer ref
        st16(&mut v[16..], 0x2222); // index buffer OFFSET first
        st16(&mut v[18..], 0x1111); // index count second
        st16(&mut v[20..], 0x3333); // instanceCount
        st16(&mut v[22..], 0xfffe); // baseVertex = -2, two's complement
        st16(&mut v[24..], 0x55); // baseInstance

        let c = decode(&v).expect("full indexed draw");
        assert_eq!(c.kind, Kind::Draw);
        assert_eq!(c.index_buffer_offset, 0x2222, "offset comes first here");
        assert_eq!(c.index_count, 0x1111, "count comes second here");
        assert_eq!(c.index_type, 1);
        assert_eq!(c.index_buffer_ref, 0x141f);
        assert_eq!(c.instance_count, 0x3333);
        assert_eq!(c.base_instance, 0x55);
        assert_eq!(
            c.base_vertex, -2,
            "a negative base vertex must stay negative"
        );
    }

    /// The wide form of the same record, whose base vertex is sign-extended to
    /// 64 bits rather than truncated to 16.
    #[test]
    fn the_wide_full_indexed_draw_sign_extends_its_base_vertex() {
        use crate::contract::endian::{st16, st64};

        let mut v = hdr(
            wire::OPCODE_DRAW_INDEXED_INSTANCED_BASE_WIDE,
            wire::DRAW_INDEXED_INSTANCED_BASE_WIDE_TOTAL_LEN as usize,
        );
        st16(&mut v[8..], 3);
        st16(&mut v[10..], 0);
        st32(&mut v[12..], 0x141f);
        st64(&mut v[16..], 0x2222); // offset first
        st64(&mut v[24..], 0x1111); // count second
        st64(&mut v[32..], 0x10000); // instanceCount, the argument that widened it
        st64(&mut v[40..], (-70000i64) as u64); // baseVertex
        st64(&mut v[48..], 0x55); // baseInstance

        let c = decode(&v).expect("wide full indexed draw");
        assert_eq!(c.index_buffer_offset, 0x2222);
        assert_eq!(c.index_count, 0x1111);
        assert_eq!(c.instance_count, 0x10000);
        assert_eq!(c.base_vertex, -70000);
        assert_eq!(c.base_instance, 0x55);
    }

    #[test]
    fn wide_indexed_draw_layout() {
        use crate::contract::endian::st16;

        let mut v = hdr(wire::OPCODE_DRAW_INDEXED_WIDE, 0x20);
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
        let mut v = hdr(wire::OPCODE_EXECUTE_COMMANDS_RANGE, EXECUTE_RANGE_CMD_LEN);
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
        let mut v = hdr(
            wire::OPCODE_EXECUTE_COMMANDS_INDIRECT,
            EXECUTE_INDIRECT_CMD_LEN,
        );
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

    /// Each of the first two records ends where the next one begins: depth
    /// `[0x00, 0x28)`, stencil `[0x28, 0x4c)`. A payload that carries both in
    /// full — and not one byte of the color section — must decode both.
    ///
    /// A shared `PASS_DEPTH_STENCIL_ATTACH_STRIDE = 0x28` used to give the
    /// stencil record the depth record's length, so the decoder demanded 0x50
    /// bytes to read a 0x24-byte record and sliced 4 bytes past its end, over
    /// color slot 0's texture ref. This payload is exactly `PASS_COLOR_ATTACH_OFF`
    /// long, so the old guard rejected it and returned a defaulted attachment.
    #[test]
    fn depth_and_stencil_records_end_where_the_next_section_begins() {
        use crate::contract::endian::{st32, st64};
        let mut payload = vec![0u8; PASS_COLOR_ATTACH_OFF];
        st32(
            &mut payload[PASS_DEPTH_ATTACH_OFF + PASS_ATTACH_TEXREF..],
            31,
        );
        st64(
            &mut payload[PASS_DEPTH_ATTACH_OFF + PASS_DEPTH_ATTACH_CLEAR_DEPTH..],
            0.25f64.to_bits(),
        );
        st32(
            &mut payload[PASS_STENCIL_ATTACH_OFF + PASS_ATTACH_TEXREF..],
            32,
        );
        st32(
            &mut payload[PASS_STENCIL_ATTACH_OFF + PASS_STENCIL_ATTACH_CLEAR_STENCIL..],
            0xfe,
        );
        let d = decode_depth_attachment(&payload);
        assert!(
            d.present,
            "depth record is complete at {PASS_STENCIL_ATTACH_OFF} bytes"
        );
        assert_eq!(d.texture_ref, 31);
        assert!((d.clear_depth - 0.25).abs() < 1e-9);
        let s = decode_stencil_attachment(&payload);
        assert!(
            s.present,
            "stencil record is complete at {PASS_COLOR_ATTACH_OFF} bytes"
        );
        assert_eq!(s.texture_ref, 32);
        assert_eq!(s.clear_stencil, 0xfe);
    }

    #[test]
    fn blend_color_and_cull() {
        let mut v = hdr(wire::OPCODE_SET_BLEND_COLOR, 24);
        // RGBA as f32 bits
        st32(&mut v[8..], 1.0f32.to_bits());
        st32(&mut v[12..], 0.0f32.to_bits());
        st32(&mut v[16..], 0.0f32.to_bits());
        st32(&mut v[20..], 1.0f32.to_bits());
        let c = decode(&v).unwrap();
        assert!(c.has_blend_color);
        assert!((c.blend_color[0] - 1.0).abs() < 1e-6);

        // Mode state is one NSUInteger on the wire (SET_MODE_TOTAL_LEN = 16).
        let mut v = hdr(
            wire::OPCODE_SET_CULL_MODE,
            wire::SET_MODE_TOTAL_LEN as usize,
        );
        st32(&mut v[8..], 2);
        let c = decode(&v).unwrap();
        assert!(c.has_cull_mode);
        assert_eq!(c.cull_mode, 2);
    }

    /// Above the window is refused; inside it but unclaimed is not.
    ///
    /// These are different answers and the difference is the whole reason the
    /// bound moved. `0x99` used to be refused here because it was one past
    /// `0x98`, the highest opcode this project had *seen* -- and `0xa5`/`0xa6`
    /// were refused with it, which lost four real vertex binds. `0x99` is now
    /// inside the encoder's range and unclaimed, so it reaches the catch-all
    /// and is reported as a gap rather than denied.
    #[test]
    fn an_opcode_above_the_window_is_refused_and_one_inside_it_is_not() {
        assert!(opcode_above_the_encoder_window(0xff));
        assert_eq!(
            decode(&hdr(0xff, 16)).unwrap_err(),
            DecodeStatus::ErrUnsupportedOpcode
        );
        // One past the highest opcode Apple's serializer writes here.
        assert!(opcode_above_the_encoder_window(
            wire::OPCODE_SET_VERTEX_BUFFER_OFFSET_STRIDE + 1
        ));
        assert_eq!(
            decode(&hdr(wire::OPCODE_SET_VERTEX_BUFFER_OFFSET_STRIDE + 1, 16)).unwrap_err(),
            DecodeStatus::ErrUnsupportedOpcode
        );
        // Inside the range, claimed by no arm. `OtherAccepted` is what says so.
        //
        // Found rather than named. A literal here goes stale the moment that
        // opcode is decoded, which already happened once: `0x99` was the
        // unclaimed probe until `setVertexAmplificationMode:value:` turned out
        // to be exactly that number.
        let unclaimed = super::unclaimed_accepted_opcode();
        assert_eq!(
            decode(&hdr(unclaimed, 16))
                .unwrap_or_else(|e| panic!("op {unclaimed:#x}: {e:?}"))
                .kind,
            Kind::OtherAccepted
        );
    }

    /// Every bind opcode refuses `count == 0` and refuses more entries than the
    /// slot table holds, so a decoded bind record ALWAYS carries at least one
    /// entry.
    ///
    /// `exec::apply_record` relies on this: it walks `buffer_binds` / `ref_binds`
    /// directly, with no single-entry wire form to fall back to. If a zero-count
    /// record ever decoded successfully, those loops would silently bind nothing.
    #[test]
    fn a_bind_record_never_decodes_to_zero_entries() {
        for (op, entry_size) in [
            (wire::OPCODE_SET_VERTEX_BUFFER, BUFFER_BIND_ENTRY_SIZE),
            (wire::OPCODE_SET_FRAGMENT_BUFFER, BUFFER_BIND_ENTRY_SIZE),
            (wire::OPCODE_SET_VERTEX_TEXTURE, REF_BIND_ENTRY_SIZE),
            (wire::OPCODE_SET_FRAGMENT_TEXTURE, REF_BIND_ENTRY_SIZE),
            (wire::OPCODE_SET_VERTEX_SAMPLER, REF_BIND_ENTRY_SIZE),
            (wire::OPCODE_SET_FRAGMENT_SAMPLER, REF_BIND_ENTRY_SIZE),
        ] {
            let hdr_len = 8;
            let body = |count: u32, entries: usize| {
                let mut v = hdr(op, hdr_len + BIND_ENTRIES + entries * entry_size);
                st32(&mut v[hdr_len + BIND_FIRST..], 0);
                st32(&mut v[hdr_len + BIND_COUNT..], count);
                v
            };
            assert_eq!(
                decode(&body(0, 0)).unwrap_err(),
                DecodeStatus::ErrBadLength,
                "op {op:#x} accepted count=0"
            );
            // A count with no entries behind it. The record is the guest's own
            // length claim, so this is the bound — and it must not wrap.
            assert_eq!(
                decode(&body(4, 3)).unwrap_err(),
                DecodeStatus::ErrShort,
                "op {op:#x} accepted a record one entry shorter than its count"
            );
            assert_eq!(
                decode(&body(u32::MAX, 1)).unwrap_err(),
                DecodeStatus::ErrShort,
                "op {op:#x} accepted a count whose entries cannot exist"
            );
            let c = decode(&body(1, 1)).unwrap_or_else(|e| panic!("op {op:#x}: {e:?}"));
            assert_eq!(
                c.buffer_binds.len() + c.ref_binds.len(),
                1,
                "op {op:#x} decoded one entry into neither list"
            );
            // Forty slots, which Apple produces (`setVertexTextures:withRange:`
            // over a range of 40) and a 32-entry cap used to refuse whole.
            let c =
                decode(&body(40, 40)).unwrap_or_else(|e| panic!("op {op:#x} refused 40: {e:?}"));
            assert_eq!(
                c.buffer_binds.len() + c.ref_binds.len(),
                40,
                "op {op:#x} did not decode all forty entries"
            );
        }
    }

    /// The six unapplied states decode to their own kinds, at their own widths.
    ///
    /// Two are sixteen-byte records with a `u64`, two are twelve-byte records
    /// with an `f32`, and the colour store action is the only one of the three
    /// store forms that carries an index -- depth and stencil have one
    /// attachment each. Reading a float record as a `f64` would take four bytes
    /// of whatever followed it, which is why the length is asserted here rather
    /// than assumed from the family.
    #[test]
    fn each_unapplied_state_decodes_at_its_own_width() {
        use crate::contract::endian::{st32, st64};
        use reims_vgpu_wire::ops::render as wire;

        for (op, wire_op) in [
            (
                wire::OPCODE_SET_DEPTH_CLIP_MODE,
                wire::OPCODE_SET_DEPTH_CLIP_MODE,
            ),
            (
                wire::OPCODE_SET_TRIANGLE_FILL_MODE,
                wire::OPCODE_SET_TRIANGLE_FILL_MODE,
            ),
            (wire::OPCODE_SET_LINE_WIDTH, wire::OPCODE_SET_LINE_WIDTH),
            (
                wire::OPCODE_SET_TESSELLATION_FACTOR_SCALE,
                wire::OPCODE_SET_TESSELLATION_FACTOR_SCALE,
            ),
            (
                wire::OPCODE_SET_COLOR_STORE_ACTION,
                wire::OPCODE_SET_COLOR_STORE_ACTION,
            ),
            (
                wire::OPCODE_SET_DEPTH_STORE_ACTION,
                wire::OPCODE_SET_DEPTH_STORE_ACTION,
            ),
            (
                wire::OPCODE_SET_STENCIL_STORE_ACTION,
                wire::OPCODE_SET_STENCIL_STORE_ACTION,
            ),
            (wire::OPCODE_TEXTURE_BARRIER, wire::OPCODE_TEXTURE_BARRIER),
        ] {
            assert_eq!(op, wire_op, "the serializer writes a different opcode");
        }

        // The `u64` mode records.
        for op in [
            wire::OPCODE_SET_DEPTH_CLIP_MODE,
            wire::OPCODE_SET_TRIANGLE_FILL_MODE,
        ] {
            let mut v = hdr(op, wire::SET_MODE_TOTAL_LEN as usize);
            st64(&mut v[OP_HEADER_LEN..], 1);
            let c = decode(&v).unwrap_or_else(|e| panic!("op {op:#x}: {e:?}"));
            assert_eq!(c.kind, Kind::SetRasterState, "op {op:#x}");
            assert_eq!(c.mode, 1, "op {op:#x}");
            assert_eq!(
                decode(&hdr(op, OP_HEADER_LEN + 4)).unwrap_err(),
                DecodeStatus::ErrShort,
                "op {op:#x} read a mode out of four bytes"
            );
        }

        // The `f32` records: twelve bytes, and the payload is four.
        for op in [
            wire::OPCODE_SET_LINE_WIDTH,
            wire::OPCODE_SET_TESSELLATION_FACTOR_SCALE,
        ] {
            let total = wire::SET_FLOAT_TOTAL_LEN as usize;
            assert_eq!(total, OP_HEADER_LEN + 4, "op {op:#x}");
            let mut v = hdr(op, total);
            st32(&mut v[OP_HEADER_LEN..], 2.5f32.to_bits());
            let c = decode(&v).unwrap_or_else(|e| panic!("op {op:#x}: {e:?}"));
            assert_eq!(c.kind, Kind::SetFloatState, "op {op:#x}");
            assert_eq!(c.float_value, 2.5, "op {op:#x}");
        }

        // The colour store action carries an index; the other two do not, and
        // the values are deliberately unequal so a swap is visible.
        let mut v = hdr(wire::OPCODE_SET_COLOR_STORE_ACTION, 16);
        st32(&mut v[OP_HEADER_LEN..], 2);
        st32(&mut v[OP_HEADER_LEN + 4..], 3);
        let c = decode(&v).expect("colour store action");
        assert_eq!(c.kind, Kind::SetStoreAction);
        assert_eq!((c.mode, c.first), (2, 3), "action and index are swapped");

        for op in [
            wire::OPCODE_SET_DEPTH_STORE_ACTION,
            wire::OPCODE_SET_STENCIL_STORE_ACTION,
        ] {
            let mut v = hdr(op, 16);
            st64(&mut v[OP_HEADER_LEN..], 1);
            let c = decode(&v).unwrap_or_else(|e| panic!("op {op:#x}: {e:?}"));
            assert_eq!(c.kind, Kind::SetStoreAction, "op {op:#x}");
            assert_eq!(c.mode, 1, "op {op:#x}");
            assert_eq!(c.first, 0, "op {op:#x} invented an attachment index");
        }

        // `textureBarrier` is the header alone and joins the barrier kind.
        let c = decode(&hdr(wire::OPCODE_TEXTURE_BARRIER, OP_HEADER_LEN)).expect("texture barrier");
        assert_eq!(c.kind, Kind::Barrier);
    }

    /// A vertex bind carrying an attribute stride binds the buffer.
    ///
    /// It used to be refused outright: `0xa5`/`0xa6` are above the old
    /// `wire::OPCODE_SET_VERTEX_BUFFER_OFFSET_STRIDE` of `0x98`, so `opcode_is_apple_rejected` called them
    /// records Apple does not emit -- and Apple emits them whenever the guest
    /// negotiates `supportsDynamicAttributeStride`. Every strided vertex bind
    /// was dropped and the buffer never bound, which is the sampler-LOD bug
    /// again with a worse consequence.
    ///
    /// The plural case is the load-bearing one. Its two entries are twenty
    /// bytes apart, and a decoder using the plain twelve would read the second
    /// entry out of the middle of the first -- so both offsets are distinct and
    /// both are asserted.
    #[test]
    fn a_vertex_bind_with_an_attribute_stride_binds_the_buffer_rather_than_being_refused() {
        use crate::contract::endian::{st32, st64};
        use reims_vgpu_wire::ops::render as wire;

        for (op, wire_op) in [
            (
                wire::OPCODE_SET_VERTEX_BUFFER_STRIDE,
                wire::OPCODE_SET_VERTEX_BUFFER_STRIDE,
            ),
            (
                wire::OPCODE_SET_VERTEX_BUFFER_OFFSET_STRIDE,
                wire::OPCODE_SET_VERTEX_BUFFER_OFFSET_STRIDE,
            ),
        ] {
            assert_eq!(op, wire_op, "the serializer writes a different opcode");
            assert!(
                !opcode_above_the_encoder_window(op),
                "op {op:#x} is still called a record Apple does not emit, and Apple emits it"
            );
        }

        // Two slots, twenty bytes apart: {ref u32, offset u64, stride u64}.
        let entries = 2usize;
        let total = OP_HEADER_LEN + BIND_ENTRIES + entries * BUFFER_STRIDE_BIND_ENTRY_SIZE;
        assert_eq!(total, 56, "the plural fixture is 56 bytes");
        let mut v = hdr(wire::OPCODE_SET_VERTEX_BUFFER_STRIDE, total);
        st32(&mut v[OP_HEADER_LEN + BIND_FIRST..], 9);
        st32(&mut v[OP_HEADER_LEN + BIND_COUNT..], entries as u32);
        for (i, (r, off, stride)) in [(5151u32, 0x3333u64, 0x5555u64), (5252, 0x4444, 0x6666)]
            .into_iter()
            .enumerate()
        {
            let e = OP_HEADER_LEN + BIND_ENTRIES + i * BUFFER_STRIDE_BIND_ENTRY_SIZE;
            st32(&mut v[e..], r);
            st64(&mut v[e + 4..], off);
            st64(&mut v[e + 12..], stride);
        }
        let c = decode(&v).expect("a strided vertex bind must decode");
        assert_eq!(c.kind, Kind::SetBuffer);
        assert_eq!(c.stage, Stage::Vertex, "there is no fragment stride form");
        assert!(c.has_attribute_stride);
        assert_eq!(c.first, 9);
        assert_eq!(
            c.buffer_binds,
            vec![(5151, 0x3333), (5252, 0x4444)],
            "the entry stride is 20, not the plain bind's 12"
        );

        // The plain bind must not be told it carries one, or every ordinary
        // vertex bind would report a loss it did not have.
        let plain_total = OP_HEADER_LEN + BIND_ENTRIES + BUFFER_BIND_ENTRY_SIZE;
        let mut p = hdr(wire::OPCODE_SET_VERTEX_BUFFER, plain_total);
        st32(&mut p[OP_HEADER_LEN + BIND_COUNT..], 1);
        st32(&mut p[OP_HEADER_LEN + BIND_ENTRIES..], 5151);
        let c = decode(&p).expect("plain vertex bind");
        assert!(!c.has_attribute_stride);

        // `0xa6` is the offset re-point with the stride appended: 28 bytes.
        let total = wire::SET_BUFFER_OFFSET_STRIDE_TOTAL_LEN as usize;
        let mut v = hdr(wire::OPCODE_SET_VERTEX_BUFFER_OFFSET_STRIDE, total);
        st32(&mut v[OP_HEADER_LEN..], 8);
        st64(&mut v[OP_HEADER_LEN + 4..], 0x4567);
        st64(&mut v[OP_HEADER_LEN + 12..], 0x5678);
        let c = decode(&v).expect("a strided offset re-point must decode");
        assert_eq!(c.kind, Kind::SetBufferOffset);
        assert_eq!(c.stage, Stage::Vertex);
        assert!(c.has_attribute_stride);
        assert_eq!((c.first, c.buffer_offset), (8, 0x4567));
        // Short of its own stride word it is refused, rather than read as the
        // twenty-byte record it is not.
        assert_eq!(
            decode(&hdr(
                wire::OPCODE_SET_VERTEX_BUFFER_OFFSET_STRIDE,
                total - 8
            ))
            .unwrap_err(),
            DecodeStatus::ErrShort
        );
    }

    /// Vertex amplification decodes at two widths the type encoding hides.
    ///
    /// The mode record declares both arguments `Q` and puts both on the wire at
    /// 32 bits, so a decoder written from the encoding reads a 24-byte record
    /// that is 16. The count record's head is four bytes rather than the
    /// eight-byte bind header every other counted record here uses, so reading
    /// a `BindHeader` takes the first mapping's viewport offset as the count --
    /// which is why the fixture's count and first offset are deliberately
    /// unequal and this test asserts on both.
    #[test]
    fn vertex_amplification_decodes_at_the_widths_the_serializer_wrote() {
        use crate::contract::endian::st32;
        use reims_vgpu_wire::ops::render as wire;

        for (op, wire_op) in [
            (
                wire::OPCODE_SET_VERTEX_AMPLIFICATION_MODE,
                wire::OPCODE_SET_VERTEX_AMPLIFICATION_MODE,
            ),
            (
                wire::OPCODE_SET_VERTEX_AMPLIFICATION_COUNT,
                wire::OPCODE_SET_VERTEX_AMPLIFICATION_COUNT,
            ),
        ] {
            assert_eq!(op, wire_op, "the serializer writes a different opcode");
        }

        let total = wire::SET_VERTEX_AMPLIFICATION_MODE_TOTAL_LEN as usize;
        assert_eq!(total, OP_HEADER_LEN + 8, "two u32, not two u64");
        let mut v = hdr(wire::OPCODE_SET_VERTEX_AMPLIFICATION_MODE, total);
        st32(&mut v[OP_HEADER_LEN..], 0x5555);
        st32(&mut v[OP_HEADER_LEN + 4..], 0x6666);
        let c = decode(&v).expect("amplification mode");
        assert_eq!(c.kind, Kind::SetVertexAmplification);
        assert_eq!(
            (c.mode, c.amplification_value),
            (0x5555, 0x6666),
            "mode and value are crossed"
        );

        // Two mappings, four distinct offsets. The count is 2 and the first
        // viewport offset is 0x1111, so a four-byte head cannot be confused
        // with an eight-byte one.
        let mappings = 2usize;
        let total = OP_HEADER_LEN + AMPLIFICATION_COUNT_LEN + mappings * AMPLIFICATION_MAPPING_SIZE;
        assert_eq!(total, 28, "the fixture is 28 bytes");
        let mut v = hdr(wire::OPCODE_SET_VERTEX_AMPLIFICATION_COUNT, total);
        st32(&mut v[OP_HEADER_LEN..], mappings as u32);
        for (i, (vp, rt)) in [(0x1111u32, 0x2222u32), (0x3333, 0x4444)]
            .into_iter()
            .enumerate()
        {
            let e = OP_HEADER_LEN + AMPLIFICATION_COUNT_LEN + i * AMPLIFICATION_MAPPING_SIZE;
            st32(&mut v[e..], vp);
            st32(&mut v[e + 4..], rt);
        }
        let c = decode(&v).expect("amplification count");
        assert_eq!(c.kind, Kind::SetVertexAmplification);
        assert_eq!(c.count, 2, "the head was read as eight bytes");

        // A count with no mappings behind it. The record is the guest's own
        // length claim, so this is the bound, and it must not wrap.
        let mut short = hdr(wire::OPCODE_SET_VERTEX_AMPLIFICATION_COUNT, total);
        st32(&mut short[OP_HEADER_LEN..], 3);
        assert_eq!(decode(&short).unwrap_err(), DecodeStatus::ErrShort);
        let mut huge = hdr(wire::OPCODE_SET_VERTEX_AMPLIFICATION_COUNT, total);
        st32(&mut huge[OP_HEADER_LEN..], u32::MAX);
        assert_eq!(decode(&huge).unwrap_err(), DecodeStatus::ErrShort);
    }

    /// `0x0c` is two records, and only the length says which.
    ///
    /// The plain wide patch draw is 56 bytes and the indexed one is 68, under
    /// the same opcode — the one place in this family where a wide form does
    /// not get its own number. The two bodies agree for nine bytes and diverge
    /// at the tenth, so a decoder that dispatched on the opcode and read either
    /// body unconditionally would misread the other half the time rather than
    /// refuse it.
    ///
    /// The third arm is the one that matters most: a `0x0c` at any *other*
    /// length has no reading at all, and picking the nearer of the two would
    /// take one record's buffer ref as the other's offset. It must be refused.
    #[test]
    fn the_wide_patch_draw_opcode_is_resolved_by_length_and_refused_without_one() {
        use reims_vgpu_wire::ops::render as wire;

        for (op, wire_op) in [
            (wire::OPCODE_DRAW_PATCHES, wire::OPCODE_DRAW_PATCHES),
            (
                wire::OPCODE_DRAW_PATCHES_WIDE,
                wire::OPCODE_DRAW_PATCHES_WIDE,
            ),
            (
                wire::OPCODE_DRAW_INDEXED_PATCHES,
                wire::OPCODE_DRAW_INDEXED_PATCHES,
            ),
            (
                wire::OPCODE_DRAW_PATCHES_INDIRECT,
                wire::OPCODE_DRAW_PATCHES_INDIRECT,
            ),
            (
                wire::OPCODE_DRAW_INDEXED_PATCHES_INDIRECT,
                wire::OPCODE_DRAW_INDEXED_PATCHES_INDIRECT,
            ),
        ] {
            assert_eq!(op, wire_op, "the serializer writes a different opcode");
        }

        // The two wide lengths really are different, which is what makes the
        // length usable as a discriminator at all.
        assert_ne!(
            wire::DRAW_PATCHES_WIDE_TOTAL_LEN,
            wire::DRAW_INDEXED_PATCHES_WIDE_TOTAL_LEN
        );

        for total in [
            wire::DRAW_PATCHES_WIDE_TOTAL_LEN as usize,
            wire::DRAW_INDEXED_PATCHES_WIDE_TOTAL_LEN as usize,
        ] {
            let c = decode(&hdr(wire::OPCODE_DRAW_PATCHES_WIDE, total))
                .unwrap_or_else(|e| panic!("0x0c at {total} bytes: {e:?}"));
            assert_eq!(c.kind, Kind::DrawPatches);
            assert_eq!(c.command_length as usize, total);
        }

        // Between the two, above both, and below both: none has a reading.
        for total in [
            wire::DRAW_PATCHES_WIDE_TOTAL_LEN as usize + 4,
            wire::DRAW_INDEXED_PATCHES_WIDE_TOTAL_LEN as usize + 4,
            wire::DRAW_PATCHES_WIDE_TOTAL_LEN as usize - 4,
        ] {
            assert_eq!(
                decode(&hdr(wire::OPCODE_DRAW_PATCHES_WIDE, total)).unwrap_err(),
                DecodeStatus::ErrBadLength,
                "0x0c at {total} bytes was given a reading; it has none"
            );
        }

        // The four single-length forms, each accepted at its own length and
        // refused four bytes short. A patch draw read short is invented
        // geometry, not a smaller draw.
        for (op, total) in [
            (
                wire::OPCODE_DRAW_PATCHES,
                wire::DRAW_PATCHES_TOTAL_LEN as usize,
            ),
            (
                wire::OPCODE_DRAW_INDEXED_PATCHES,
                wire::DRAW_INDEXED_PATCHES_TOTAL_LEN as usize,
            ),
            (
                wire::OPCODE_DRAW_PATCHES_INDIRECT,
                wire::DRAW_PATCHES_INDIRECT_TOTAL_LEN as usize,
            ),
            (
                wire::OPCODE_DRAW_INDEXED_PATCHES_INDIRECT,
                wire::DRAW_INDEXED_PATCHES_INDIRECT_TOTAL_LEN as usize,
            ),
        ] {
            let c = decode(&hdr(op, total)).unwrap_or_else(|e| panic!("op {op:#x}: {e:?}"));
            assert_eq!(c.kind, Kind::DrawPatches, "op {op:#x}");
            assert_eq!(
                decode(&hdr(op, total - 4)).unwrap_err(),
                DecodeStatus::ErrBadLength,
                "op {op:#x} accepted a record four bytes short"
            );
        }

        // A record whose header promises more bytes than are present is
        // refused by the framing, before any of the above runs.
        let mut truncated = hdr(
            wire::OPCODE_DRAW_PATCHES,
            wire::DRAW_PATCHES_TOTAL_LEN as usize,
        );
        truncated.truncate(truncated.len() - 4);
        assert!(decode(&truncated).is_err(), "a short buffer was accepted");
    }

    /// The store-action options and the tessellation factor buffer.
    ///
    /// `0x67`/`0x6a`/`0x79` sit one opcode above the three store actions and
    /// look like longer forms of them. They are not, and the difference is a
    /// *width*: the options are a `u64` where the action is a `u32`, so the
    /// colour form's attachment index moves from payload `+4` to `+8` and the
    /// record grows from 16 bytes to 20. A decoder that reused
    /// `ColorStoreAction` here would read the index out of the options' high
    /// half and report attachment 0 for every one of them.
    ///
    /// `0x7a` is checked in the same test because it makes the opposite
    /// mistake available: it names a buffer with an offset, so it reads like a
    /// bind, and a reader that took a `BindHeader` would call the ref `first`
    /// and the low half of the offset `count`.
    /// A colour attachment's `level` is sixteen bits, and `slice` is the
    /// sixteen above it.
    ///
    /// This arm read `ld32` at [`PASS_ATTACH_LEVEL`] for as long as it existed,
    /// under a comment on the depth arm that stated the rule as "the archive
    /// uses u16 for depth/stencil level (color uses u32)". Apple's own bytes say
    /// all three attachment shapes are identical through their prefix, so the
    /// wide read returned `level | (slice << 16)`: a pass rendering into array
    /// slice 1 reported mip level 65536 and lost the slice.
    ///
    /// The synthetic here is what a cube-face pass looks like — level 1, slice
    /// 5 — and the two fields are read apart. The `0xffff` case is the same
    /// claim at the boundary: a slice that fills its half must not reach the
    /// level at all.
    #[test]
    fn a_colour_attachments_level_does_not_swallow_its_slice() {
        for (level, slice, plane) in [(1u16, 5u16, 2u16), (0, 0xffff, 0), (0xffff, 0, 0)] {
            let total = OP_HEADER_LEN + PASS_MIN_PAYLOAD;
            let mut cmd = vec![0u8; total];
            st32(&mut cmd[0..], wire_pass::OPCODE_RENDER_PASS);
            st32(&mut cmd[4..], total as u32);
            let slot = OP_HEADER_LEN + PASS_COLOR_ATTACH_OFF;
            st32(&mut cmd[slot + PASS_ATTACH_TEXREF..], 7);
            cmd[slot + PASS_ATTACH_LEVEL..slot + PASS_ATTACH_LEVEL + 2]
                .copy_from_slice(&level.to_le_bytes());
            cmd[slot + PASS_ATTACH_SLICE..slot + PASS_ATTACH_SLICE + 2]
                .copy_from_slice(&slice.to_le_bytes());
            cmd[slot + PASS_ATTACH_DEPTH_PLANE..slot + PASS_ATTACH_DEPTH_PLANE + 2]
                .copy_from_slice(&plane.to_le_bytes());
            let att = decode_color_attachment(&cmd[OP_HEADER_LEN..], 0);
            assert_eq!(att.level, u32::from(level), "level took the slice's bits");
            assert_eq!(att.slice, u32::from(slice), "slice went unread");
            assert_eq!(att.depth_plane, u32::from(plane), "depth plane went unread");
        }
    }

    /// The pass's tail is four fields this device decodes and does not apply.
    ///
    /// A record short of the tail must leave all four at zero rather than
    /// reading past its own payload — the decoder accepts a payload as small as
    /// one colour slot, and Apple's own record is 584 bytes.
    #[test]
    fn the_render_pass_tail_is_read_only_when_the_record_carries_one() {
        use crate::contract::endian::st64;
        let full = OP_HEADER_LEN + PASS_TAIL_OFF + 0x1c;
        let mut cmd = vec![0u8; full];
        st32(&mut cmd[0..], wire_pass::OPCODE_RENDER_PASS);
        st32(&mut cmd[4..], full as u32);
        let t = OP_HEADER_LEN + PASS_TAIL_OFF;
        st32(&mut cmd[t + PASS_TAIL_VISIBILITY_BUFFER_REF..], 5151);
        st64(&mut cmd[t + PASS_TAIL_ARRAY_LENGTH..], 0x11);
        st64(&mut cmd[t + PASS_TAIL_TARGET_WIDTH..], 0x1234);
        st64(&mut cmd[t + PASS_TAIL_TARGET_HEIGHT..], 0x5678);
        let c = decode(&cmd).expect("well formed");
        assert_eq!(c.kind, Kind::RenderPass);
        assert_eq!(c.pass_visibility_result_buffer_ref, 5151);
        assert_eq!(c.pass_render_target_array_length, 0x11);
        assert_eq!(c.pass_render_target_width, 0x1234);
        assert_eq!(c.pass_render_target_height, 0x5678);

        let short = OP_HEADER_LEN + PASS_MIN_PAYLOAD;
        let mut cmd = vec![0u8; short];
        st32(&mut cmd[0..], wire_pass::OPCODE_RENDER_PASS);
        st32(&mut cmd[4..], short as u32);
        let c = decode(&cmd).expect("well formed");
        assert_eq!(c.kind, Kind::RenderPass);
        assert_eq!(c.pass_render_target_width, 0, "read past the payload");
    }

    /// The six pass-property records decode rather than reaching the catch-all.
    ///
    /// Every one sits inside the accepted opcode window, so before they were
    /// named they were accepted, dropped and silent — which is exactly the
    /// shape that hid the sampler-LOD binds.
    #[test]
    fn every_pass_property_record_reaches_an_arm_of_its_own() {
        for (op, payload_len) in [
            (wire_pass::OPCODE_DEFAULT_RASTER_SAMPLE_COUNT, 4usize),
            (wire_pass::OPCODE_RASTERIZATION_RATE_MAP, 4),
            (wire_pass::OPCODE_IMAGEBLOCK_SAMPLE_LENGTH, 4),
            (wire_pass::OPCODE_THREADGROUP_MEMORY_LENGTH, 4),
            (wire_pass::OPCODE_TILE_SIZE, 4),
        ] {
            let total = OP_HEADER_LEN + payload_len;
            let mut cmd = vec![0u8; total];
            st32(&mut cmd[0..], op);
            st32(&mut cmd[4..], total as u32);
            st32(&mut cmd[OP_HEADER_LEN..], 4);
            let c = decode(&cmd).expect("well formed");
            assert_eq!(c.kind, Kind::RenderPassProperty, "opcode {op:#x}");
            if op == wire_pass::OPCODE_RASTERIZATION_RATE_MAP {
                assert_eq!(c.texture_ref, 4, "opcode {op:#x}: ref");
            } else {
                assert_eq!(c.mode, 4, "opcode {op:#x}: scalar");
            }
            // A header that claims to be the whole record carries no scalar,
            // and that is a refusal rather than a zero.
            let mut bare = vec![0u8; OP_HEADER_LEN];
            st32(&mut bare[0..], op);
            st32(&mut bare[4..], OP_HEADER_LEN as u32);
            assert!(
                matches!(decode(&bare), Err(DecodeStatus::ErrBadLength)),
                "opcode {op:#x}"
            );
        }

        // Sample positions are head plus `count` pairs, and the count is guest
        // data: one claiming more pairs than the record holds is refused.
        let total = OP_HEADER_LEN + 4 + 2 * 8;
        let mut cmd = vec![0u8; total];
        st32(&mut cmd[0..], wire_pass::OPCODE_SAMPLE_POSITIONS);
        st32(&mut cmd[4..], total as u32);
        st32(&mut cmd[OP_HEADER_LEN..], 2);
        let c = decode(&cmd).expect("well formed");
        assert_eq!(c.kind, Kind::RenderPassProperty);
        assert_eq!(c.count, 2);
        st32(&mut cmd[OP_HEADER_LEN..], 0xffff_ffff);
        assert!(matches!(decode(&cmd), Err(DecodeStatus::ErrBadLength)));
    }

    #[test]
    fn the_store_action_options_are_not_wider_store_actions() {
        use crate::contract::endian::{st32, st64};
        use reims_vgpu_wire::ops::render as wire;

        for (op, wire_op) in [
            (
                wire::OPCODE_SET_COLOR_STORE_ACTION_OPTIONS,
                wire::OPCODE_SET_COLOR_STORE_ACTION_OPTIONS,
            ),
            (
                wire::OPCODE_SET_DEPTH_STORE_ACTION_OPTIONS,
                wire::OPCODE_SET_DEPTH_STORE_ACTION_OPTIONS,
            ),
            (
                wire::OPCODE_SET_STENCIL_STORE_ACTION_OPTIONS,
                wire::OPCODE_SET_STENCIL_STORE_ACTION_OPTIONS,
            ),
            (
                wire::OPCODE_SET_TESSELLATION_FACTOR_BUFFER,
                wire::OPCODE_SET_TESSELLATION_FACTOR_BUFFER,
            ),
        ] {
            assert_eq!(op, wire_op, "the serializer writes a different opcode");
        }

        // Each options opcode is exactly one above its store action, and the
        // two are different records at different lengths. Asserting the
        // adjacency keeps a future edit from collapsing them into one arm.
        for (action, options) in [
            (
                wire::OPCODE_SET_COLOR_STORE_ACTION,
                wire::OPCODE_SET_COLOR_STORE_ACTION_OPTIONS,
            ),
            (
                wire::OPCODE_SET_DEPTH_STORE_ACTION,
                wire::OPCODE_SET_DEPTH_STORE_ACTION_OPTIONS,
            ),
            (
                wire::OPCODE_SET_STENCIL_STORE_ACTION,
                wire::OPCODE_SET_STENCIL_STORE_ACTION_OPTIONS,
            ),
        ] {
            assert_eq!(options, action + 1);
        }

        // The colour form. The index is at +8; a `u32` read of the options
        // would leave it at +4 and find the options' own high half.
        let total = wire::SET_COLOR_STORE_ACTION_OPTIONS_TOTAL_LEN as usize;
        let mut v = hdr(wire::OPCODE_SET_COLOR_STORE_ACTION_OPTIONS, total);
        st64(&mut v[OP_HEADER_LEN..], 0x1111);
        st32(&mut v[OP_HEADER_LEN + 8..], 3);
        let c = decode(&v).expect("colour store action options");
        assert_eq!(c.kind, Kind::SetStoreActionOptions);
        assert_eq!(
            (c.mode, c.first),
            (0x1111, 3),
            "the options and the attachment index are crossed"
        );

        // Depth and stencil have one attachment each and carry no index, so
        // their record is four bytes shorter than the colour form's.
        let total = wire::SET_STORE_ACTION_OPTIONS_TOTAL_LEN as usize;
        assert_eq!(
            total + 4,
            wire::SET_COLOR_STORE_ACTION_OPTIONS_TOTAL_LEN as usize
        );
        for (op, options) in [
            (wire::OPCODE_SET_DEPTH_STORE_ACTION_OPTIONS, 0x2222u64),
            (wire::OPCODE_SET_STENCIL_STORE_ACTION_OPTIONS, 0x3333),
        ] {
            let mut v = hdr(op, total);
            st64(&mut v[OP_HEADER_LEN..], options);
            let c = decode(&v).unwrap_or_else(|e| panic!("op {op:#x}: {e:?}"));
            assert_eq!(c.kind, Kind::SetStoreActionOptions);
            assert_eq!(c.mode, options, "op {op:#x}");
            assert_eq!(c.first, 0, "op {op:#x} invented an attachment index");
        }

        // `0x7a`: ref, then two `u64` that differ, so a crossed pair shows.
        let total = wire::SET_TESSELLATION_FACTOR_BUFFER_TOTAL_LEN as usize;
        let mut v = hdr(wire::OPCODE_SET_TESSELLATION_FACTOR_BUFFER, total);
        st32(&mut v[OP_HEADER_LEN..], 5151);
        st64(&mut v[OP_HEADER_LEN + 4..], 0x3456);
        st64(&mut v[OP_HEADER_LEN + 12..], 0x4567);
        let c = decode(&v).expect("tessellation factor buffer");
        assert_eq!(c.kind, Kind::SetTessellationFactorBuffer);
        assert_eq!(
            (c.buffer_ref, c.buffer_offset),
            (5151, 0x3456),
            "read as a bind header rather than as a ref and an offset"
        );

        for (op, total) in [
            (
                wire::OPCODE_SET_COLOR_STORE_ACTION_OPTIONS,
                wire::SET_COLOR_STORE_ACTION_OPTIONS_TOTAL_LEN as usize,
            ),
            (
                wire::OPCODE_SET_DEPTH_STORE_ACTION_OPTIONS,
                wire::SET_STORE_ACTION_OPTIONS_TOTAL_LEN as usize,
            ),
            (
                wire::OPCODE_SET_STENCIL_STORE_ACTION_OPTIONS,
                wire::SET_STORE_ACTION_OPTIONS_TOTAL_LEN as usize,
            ),
            (
                wire::OPCODE_SET_TESSELLATION_FACTOR_BUFFER,
                wire::SET_TESSELLATION_FACTOR_BUFFER_TOTAL_LEN as usize,
            ),
        ] {
            assert_eq!(
                decode(&hdr(op, total - 4)).unwrap_err(),
                DecodeStatus::ErrBadLength,
                "op {op:#x} accepted a record four bytes short"
            );
        }
    }

    /// The nine tile-shader opcodes leave the catch-all.
    ///
    /// All nine were `Kind::OtherAccepted` together, so a guest running a tile
    /// shader produced one deduped line naming a number and nothing that said a
    /// dispatch or a bind had been lost. They are checked here against
    /// `reims_vgpu_wire::ops::tile`'s constants, which fixtures pin against
    /// bytes Apple's serializer produced.
    ///
    /// Every value is distinct, because four of the five bind opcodes share a
    /// record shape and differ only in entry stride: a decoder that took
    /// `0xa0`'s twelve-byte entry for `0x9f`'s four would accept a record it
    /// should refuse, and one that read `0x9e` as a bind header would take the
    /// low half of its 64-bit offset as a count.
    #[test]
    fn a_tile_record_is_decoded_rather_than_accepted_without_a_claim() {
        use crate::contract::endian::{st32, st64};
        use reims_vgpu_wire::ops::tile as wire_tile;

        // The local constants and the serializer's, held together so neither
        // can drift. This is the check that would have caught `0x86`/`0x87`.
        for (op, wire_op) in [
            (
                wire_tile::OPCODE_DISPATCH_THREADS_PER_TILE,
                wire_tile::OPCODE_DISPATCH_THREADS_PER_TILE,
            ),
            (
                wire_tile::OPCODE_SET_TILE_BUFFER,
                wire_tile::OPCODE_SET_TILE_BUFFER,
            ),
            (
                wire_tile::OPCODE_SET_TILE_BUFFER_OFFSET,
                wire_tile::OPCODE_SET_TILE_BUFFER_OFFSET,
            ),
            (
                wire_tile::OPCODE_SET_TILE_SAMPLER,
                wire_tile::OPCODE_SET_TILE_SAMPLER,
            ),
            (
                wire_tile::OPCODE_SET_TILE_SAMPLER_LOD,
                wire_tile::OPCODE_SET_TILE_SAMPLER_LOD,
            ),
            (
                wire_tile::OPCODE_SET_TILE_TEXTURE,
                wire_tile::OPCODE_SET_TILE_TEXTURE,
            ),
            (
                wire_tile::OPCODE_DISPATCH_THREADS_PER_TILE_IN_REGION,
                wire_tile::OPCODE_DISPATCH_THREADS_PER_TILE_IN_REGION,
            ),
            (
                wire_tile::OPCODE_DISPATCH_THREADS_PER_TILE_IN_REGION_RT_INDEX,
                wire_tile::OPCODE_DISPATCH_THREADS_PER_TILE_IN_REGION_RT_INDEX,
            ),
            (
                wire_tile::OPCODE_GET_TILE_DIMENSIONS,
                wire_tile::OPCODE_GET_TILE_DIMENSIONS,
            ),
            (
                wire_tile::OPCODE_SET_TILE_THREADGROUP_MEMORY,
                wire_tile::OPCODE_SET_TILE_THREADGROUP_MEMORY,
            ),
        ] {
            assert_eq!(op, wire_op, "the serializer writes a different opcode");
        }

        // `0x9c`: length, offset, index. Three distinct values, because a
        // decoder that took the compute encoder's two-field namesake would read
        // the offset's low half as the index.
        let total = wire_tile::SET_TILE_THREADGROUP_MEMORY_TOTAL_LEN as usize;
        let mut v = hdr(wire_tile::OPCODE_SET_TILE_THREADGROUP_MEMORY, total);
        st64(&mut v[OP_HEADER_LEN..], 0x1234);
        st64(&mut v[OP_HEADER_LEN + 8..], 0x2345);
        st32(&mut v[OP_HEADER_LEN + 16..], 5);
        let c = decode(&v).expect("tile threadgroup memory");
        assert_eq!(c.kind, Kind::TileBind);
        assert_eq!((c.first, c.count), (5, 1));

        // `0x9b`: three unnarrowed `u64`, none of them equal.
        let total = wire_tile::DISPATCH_THREADS_PER_TILE_TOTAL_LEN as usize;
        let mut v = hdr(wire_tile::OPCODE_DISPATCH_THREADS_PER_TILE, total);
        st64(&mut v[OP_HEADER_LEN..], 0x11);
        st64(&mut v[OP_HEADER_LEN + 8..], 0x22);
        st64(&mut v[OP_HEADER_LEN + 16..], 0x33);
        let c = decode(&v).expect("tile dispatch");
        assert_eq!(c.kind, Kind::TileDispatch);
        assert_eq!(c.tile_threads, [0x11, 0x22, 0x33]);

        // `0xa2`/`0xa3`: the same nine `u64`, the grid first and the region
        // origin-before-size. Only `0xa3` writes the trailing `u32`, so the
        // decoder must read neither -- set those four bytes and require the
        // answer not to move on either opcode.
        let total = wire_tile::DISPATCH_THREADS_PER_TILE_IN_REGION_TOTAL_LEN as usize;
        for op in [
            wire_tile::OPCODE_DISPATCH_THREADS_PER_TILE_IN_REGION,
            wire_tile::OPCODE_DISPATCH_THREADS_PER_TILE_IN_REGION_RT_INDEX,
        ] {
            let mut v = hdr(op, total);
            for (i, value) in [0x11u64, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99]
                .into_iter()
                .enumerate()
            {
                st64(&mut v[OP_HEADER_LEN + i * 8..], value);
            }
            let c = decode(&v).expect("tile region dispatch");
            assert_eq!(c.kind, Kind::TileDispatch);
            assert_eq!(
                c.tile_threads,
                [0x11, 0x22, 0x33],
                "op {op:#x} took the region's origin for the grid"
            );

            let mut noisy = v.clone();
            noisy[OP_HEADER_LEN + wire_tile::REGION_RT_INDEX_OFFSET..][..4].fill(0xff);
            assert_eq!(
                decode(&noisy).expect("tile region dispatch"),
                c,
                "op {op:#x} let the trailing render-target index reach a field; on \
                 0xa2 those four bytes are the guest's ring"
            );
        }

        // `0xa4`: a ref then a 64-bit offset, and it is a readback rather than
        // a bind -- the buffer is where the *host* writes.
        let total = wire_tile::GET_TILE_DIMENSIONS_TOTAL_LEN as usize;
        let mut v = hdr(wire_tile::OPCODE_GET_TILE_DIMENSIONS, total);
        st32(&mut v[OP_HEADER_LEN..], 5151);
        st64(&mut v[OP_HEADER_LEN + 4..], 0x9999);
        let c = decode(&v).expect("tile dimensions query");
        assert_eq!(c.kind, Kind::TileDimensionsQuery);
        assert_eq!((c.buffer_ref, c.buffer_offset), (5151, 0x9999));

        // `0x9e`: index then a 64-bit offset. Not a bind header -- a decoder
        // that read one would take the offset's low half as a count.
        let total = wire_tile::SET_TILE_BUFFER_OFFSET_TOTAL_LEN as usize;
        let mut v = hdr(wire_tile::OPCODE_SET_TILE_BUFFER_OFFSET, total);
        st32(&mut v[OP_HEADER_LEN..], 4);
        st64(&mut v[OP_HEADER_LEN + 4..], 0x2345);
        let c = decode(&v).expect("tile buffer offset");
        assert_eq!(c.kind, Kind::TileBind);
        assert_eq!((c.first, c.count, c.buffer_offset), (4, 1, 0x2345));

        // The four bind opcodes, each at its own entry stride. A two-slot
        // record is built at the right size and accepted, and the same record
        // one entry short is refused -- which is what separates "knows the
        // stride" from "accepts anything with a plausible head".
        for (op, entry_size) in [
            (wire_tile::OPCODE_SET_TILE_BUFFER, BUFFER_BIND_ENTRY_SIZE),
            (wire_tile::OPCODE_SET_TILE_TEXTURE, REF_BIND_ENTRY_SIZE),
            (wire_tile::OPCODE_SET_TILE_SAMPLER, REF_BIND_ENTRY_SIZE),
            (
                wire_tile::OPCODE_SET_TILE_SAMPLER_LOD,
                SAMPLER_LOD_BIND_ENTRY_SIZE,
            ),
        ] {
            let total = OP_HEADER_LEN + BIND_ENTRIES + 2 * entry_size;
            let mut v = hdr(op, total);
            st32(&mut v[OP_HEADER_LEN + BIND_FIRST..], 7);
            st32(&mut v[OP_HEADER_LEN + BIND_COUNT..], 2);
            let c = decode(&v).unwrap_or_else(|e| panic!("op {op:#x}: {e:?}"));
            assert_eq!(c.kind, Kind::TileBind, "op {op:#x}");
            assert_eq!((c.first, c.count), (7, 2), "op {op:#x}");

            let mut short = hdr(op, total - entry_size);
            st32(&mut short[OP_HEADER_LEN + BIND_FIRST..], 7);
            st32(&mut short[OP_HEADER_LEN + BIND_COUNT..], 2);
            assert_eq!(
                decode(&short).unwrap_err(),
                DecodeStatus::ErrShort,
                "op {op:#x} accepted a two-slot bind holding one entry"
            );

            // A zero count is not a bind; it is a record whose head does not
            // describe itself, the same refusal the other stages give.
            let mut empty = hdr(op, OP_HEADER_LEN + BIND_ENTRIES);
            st32(&mut empty[OP_HEADER_LEN + BIND_COUNT..], 0);
            assert_eq!(
                decode(&empty).unwrap_err(),
                DecodeStatus::ErrBadLength,
                "op {op:#x} accepted a bind of no slots"
            );
        }

        // The fixed-length forms are refused at any other length rather than
        // read short.
        for (op, total) in [
            (
                wire_tile::OPCODE_DISPATCH_THREADS_PER_TILE,
                wire_tile::DISPATCH_THREADS_PER_TILE_TOTAL_LEN as usize,
            ),
            (
                wire_tile::OPCODE_DISPATCH_THREADS_PER_TILE_IN_REGION,
                wire_tile::DISPATCH_THREADS_PER_TILE_IN_REGION_TOTAL_LEN as usize,
            ),
            (
                wire_tile::OPCODE_DISPATCH_THREADS_PER_TILE_IN_REGION_RT_INDEX,
                wire_tile::DISPATCH_THREADS_PER_TILE_IN_REGION_TOTAL_LEN as usize,
            ),
            (
                wire_tile::OPCODE_SET_TILE_BUFFER_OFFSET,
                wire_tile::SET_TILE_BUFFER_OFFSET_TOTAL_LEN as usize,
            ),
            (
                wire_tile::OPCODE_GET_TILE_DIMENSIONS,
                wire_tile::GET_TILE_DIMENSIONS_TOTAL_LEN as usize,
            ),
            (
                wire_tile::OPCODE_SET_TILE_THREADGROUP_MEMORY,
                wire_tile::SET_TILE_THREADGROUP_MEMORY_TOTAL_LEN as usize,
            ),
        ] {
            assert_eq!(
                decode(&hdr(op, total - 4)).unwrap_err(),
                DecodeStatus::ErrBadLength,
                "op {op:#x} accepted a record four bytes short"
            );
        }
    }

    /// The two indirect draws and the visibility mode leave the catch-all.
    ///
    /// All three were `Kind::OtherAccepted`, which is an `Ok` that means "no arm
    /// claimed this" -- so a guest's indirect draw produced one deduped line
    /// naming a raw opcode and nothing that said a draw had been lost.
    ///
    /// Every field here is a distinct value on purpose. Two of these three
    /// records put an argument on the wire in the reverse of its selector
    /// order, and `0x11` names two buffers and two offsets that are all the
    /// same widths, so a decoder that crossed any pair would read back
    /// plausible numbers and draw the wrong thing. Only distinct values catch
    /// that, and the layouts they are checked against are
    /// `reims_vgpu_wire::ops::render`'s, pinned by fixtures.
    #[test]
    fn an_indirect_draw_is_decoded_rather_than_accepted_without_a_claim() {
        use crate::contract::endian::{st16, st32, st64};
        use reims_vgpu_wire::ops::render as wire;

        for (op, wire_op) in [
            (wire::OPCODE_DRAW_INDIRECT, wire::OPCODE_DRAW_INDIRECT),
            (
                wire::OPCODE_DRAW_INDEXED_INDIRECT,
                wire::OPCODE_DRAW_INDEXED_INDIRECT,
            ),
            (
                wire::OPCODE_SET_VISIBILITY_RESULT_MODE,
                wire::OPCODE_SET_VISIBILITY_RESULT_MODE,
            ),
        ] {
            assert_eq!(op, wire_op, "the serializer writes a different opcode");
        }

        // `0x10`: offset first, then the buffer, then a 16-bit primitive type.
        let total = wire::DRAW_INDIRECT_TOTAL_LEN as usize;
        let mut v = hdr(wire::OPCODE_DRAW_INDIRECT, total);
        st64(&mut v[OP_HEADER_LEN..], 0x1111);
        st32(&mut v[OP_HEADER_LEN + 8..], 5151);
        st16(&mut v[OP_HEADER_LEN + 12..], 3);
        let c = decode(&v).expect("indirect draw");
        assert_eq!(c.kind, Kind::DrawIndirect);
        assert_eq!(c.indirect_buffer_offset, 0x1111);
        assert_eq!(c.indirect_buffer_ref, 5151);
        assert_eq!(c.primitive_type, 3);
        // The two bytes above `primitive_type` are never written by the
        // serializer, so a wider read would take them. Set them and require the
        // answer not to move; `no_decoder_reads_a_bit_apples_serializer_never_wrote`
        // makes the same check against Apple's own measured mask.
        let mut noisy = v.clone();
        noisy[OP_HEADER_LEN + 14] = 0xff;
        noisy[OP_HEADER_LEN + 15] = 0xff;
        assert_eq!(
            decode(&noisy).expect("indirect draw"),
            c,
            "the record's unwritten tail reached a field"
        );

        // `0x11`: both types lead as `u16`, both refs follow as `u32`, both
        // offsets trail as `u64` -- the blit family's shape, not `0x10`'s.
        let total = wire::DRAW_INDEXED_INDIRECT_TOTAL_LEN as usize;
        let mut v = hdr(wire::OPCODE_DRAW_INDEXED_INDIRECT, total);
        st16(&mut v[OP_HEADER_LEN..], 4);
        st16(&mut v[OP_HEADER_LEN + 2..], 1);
        st32(&mut v[OP_HEADER_LEN + 4..], 5151);
        st32(&mut v[OP_HEADER_LEN + 8..], 5252);
        st64(&mut v[OP_HEADER_LEN + 12..], 0x1111);
        st64(&mut v[OP_HEADER_LEN + 20..], 0x2222);
        let c = decode(&v).expect("indexed indirect draw");
        assert_eq!(c.kind, Kind::DrawIndirect);
        assert_eq!(c.primitive_type, 4);
        assert_eq!(c.index_type, 1, "index type read out of the primitive type");
        assert_eq!((c.index_buffer_ref, c.indirect_buffer_ref), (5151, 5252));
        assert_eq!(
            (c.index_buffer_offset, c.indirect_buffer_offset),
            (0x1111, 0x2222),
            "the two offsets are crossed"
        );

        // `0x84`: offset first, mode second, reversing the selector.
        let total = wire::SET_VISIBILITY_RESULT_MODE_TOTAL_LEN as usize;
        let mut v = hdr(wire::OPCODE_SET_VISIBILITY_RESULT_MODE, total);
        st64(&mut v[OP_HEADER_LEN..], 0x1234);
        st64(&mut v[OP_HEADER_LEN + 8..], 2);
        let c = decode(&v).expect("visibility result mode");
        assert_eq!(c.kind, Kind::SetVisibilityResultMode);
        assert_eq!(
            (c.visibility_result_offset, c.mode),
            (0x1234, 2),
            "mode and offset are swapped"
        );

        // Each is a fixed length the serializer always writes, so a record that
        // is not that length is refused rather than read short.
        for (op, total) in [
            (
                wire::OPCODE_DRAW_INDIRECT,
                wire::DRAW_INDIRECT_TOTAL_LEN as usize,
            ),
            (
                wire::OPCODE_DRAW_INDEXED_INDIRECT,
                wire::DRAW_INDEXED_INDIRECT_TOTAL_LEN as usize,
            ),
            (
                wire::OPCODE_SET_VISIBILITY_RESULT_MODE,
                wire::SET_VISIBILITY_RESULT_MODE_TOTAL_LEN as usize,
            ),
        ] {
            assert_eq!(
                decode(&hdr(op, total - 4)).unwrap_err(),
                DecodeStatus::ErrBadLength,
                "op {op:#x} accepted a record four bytes short"
            );
        }
    }

    /// The plural viewport and scissor records are the singular ones behind a
    /// count, and the two counts are not the same width.
    ///
    /// Eight bytes for scissor, four for viewport -- from selectors that both
    /// declare `Q`, so only the capture settles it. Borrowing either constant
    /// for the other reads the first entry four bytes off, which for a scissor
    /// is `x` taken from the high half of the count.
    #[test]
    fn a_plural_viewport_or_scissor_is_the_singular_record_behind_its_own_count() {
        use crate::contract::endian::{st32, st64};
        use reims_vgpu_wire::ops::render as wire;

        assert_eq!(
            wire::OPCODE_SET_SCISSOR_RECTS,
            wire::OPCODE_SET_SCISSOR_RECTS
        );
        assert_eq!(wire::OPCODE_SET_VIEWPORTS, wire::OPCODE_SET_VIEWPORTS);
        assert_ne!(
            SCISSOR_RECTS_COUNT_LEN, VIEWPORTS_COUNT_LEN,
            "the two counts are different widths; that is the whole hazard"
        );

        // Two rects, and the *first* is the one this rail keeps.
        let total = OP_HEADER_LEN + SCISSOR_RECTS_COUNT_LEN + 2 * SCISSOR_PAYLOAD_LEN;
        let mut v = hdr(wire::OPCODE_SET_SCISSOR_RECTS, total);
        st64(&mut v[OP_HEADER_LEN..], 2);
        let e0 = OP_HEADER_LEN + SCISSOR_RECTS_COUNT_LEN;
        for (i, val) in [0x11u64, 0x22, 0x33, 0x44].into_iter().enumerate() {
            st64(&mut v[e0 + i * 8..], val);
        }
        let e1 = e0 + SCISSOR_PAYLOAD_LEN;
        for (i, val) in [0x55u64, 0x66, 0x77, 0x88].into_iter().enumerate() {
            st64(&mut v[e1 + i * 8..], val);
        }
        let c = decode(&v).expect("two scissor rects");
        assert_eq!(c.kind, Kind::SetScissor);
        assert_eq!(c.count, 2);
        assert_eq!(
            (c.scissor_x, c.scissor_y, c.scissor_w, c.scissor_h),
            (0x11, 0x22, 0x33, 0x44),
            "the record was read at the viewport record's count width"
        );

        // Two viewports, four-byte count.
        let total = OP_HEADER_LEN + VIEWPORTS_COUNT_LEN + 2 * 48;
        let mut v = hdr(wire::OPCODE_SET_VIEWPORTS, total);
        st32(&mut v[OP_HEADER_LEN..], 2);
        let e0 = OP_HEADER_LEN + VIEWPORTS_COUNT_LEN;
        for i in 0..6 {
            st64(&mut v[e0 + i * 8..], (1.0f64 + i as f64).to_bits());
        }
        for i in 0..6 {
            st64(&mut v[e0 + 48 + i * 8..], (100.0f64 + i as f64).to_bits());
        }
        let c = decode(&v).expect("two viewports");
        assert_eq!(c.kind, Kind::SetViewport);
        assert_eq!(c.count, 2);
        assert_eq!(c.viewport, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        // A count of zero names no rect, and a record that cannot hold the
        // count it claims is refused rather than read short.
        let mut v = hdr(
            wire::OPCODE_SET_SCISSOR_RECTS,
            OP_HEADER_LEN + SCISSOR_RECTS_COUNT_LEN,
        );
        st64(&mut v[OP_HEADER_LEN..], 0);
        assert_eq!(decode(&v).unwrap_err(), DecodeStatus::ErrBadLength);
        let mut v = hdr(
            wire::OPCODE_SET_VIEWPORTS,
            OP_HEADER_LEN + VIEWPORTS_COUNT_LEN + 48,
        );
        st32(&mut v[OP_HEADER_LEN..], 2);
        assert_eq!(decode(&v).unwrap_err(), DecodeStatus::ErrShort);

        // The singular opcodes keep reading from offset zero and report one.
        let mut v = hdr(
            wire::OPCODE_SET_SCISSOR,
            OP_HEADER_LEN + SCISSOR_PAYLOAD_LEN,
        );
        st64(&mut v[OP_HEADER_LEN..], 0x99);
        let c = decode(&v).expect("singular scissor");
        assert_eq!(c.scissor_x, 0x99);
        assert_eq!(c.count, 1);
    }

    /// A sampler bind carrying LOD clamps is a bind, at a wider entry stride.
    ///
    /// `0x80` and `0x71` are not longer forms of `0x7f` and `0x70` — they are
    /// separate opcodes, so this decoder knowing only the plain pair did not
    /// lose the clamps, it lost the whole bind and left the slot empty. The
    /// entry is twelve bytes here against four there, which is the assertion
    /// that matters: reading a LOD record at the plain stride would take a
    /// clamp for the next slot's sampler.
    #[test]
    fn a_sampler_bind_with_lod_clamps_is_still_a_sampler_bind() {
        use crate::contract::endian::st32;
        use reims_vgpu_wire::ops::render as wire;

        assert_eq!(
            wire::OPCODE_SET_VERTEX_SAMPLER_LOD,
            wire::OPCODE_SET_VERTEX_SAMPLER_LOD
        );
        assert_eq!(
            wire::OPCODE_SET_FRAGMENT_SAMPLER_LOD,
            wire::OPCODE_SET_FRAGMENT_SAMPLER_LOD
        );

        for (op, stage) in [
            (wire::OPCODE_SET_VERTEX_SAMPLER_LOD, Stage::Vertex),
            (wire::OPCODE_SET_FRAGMENT_SAMPLER_LOD, Stage::Fragment),
        ] {
            const COUNT: u32 = 2;
            let total =
                OP_HEADER_LEN + BIND_ENTRIES + (COUNT as usize) * SAMPLER_LOD_BIND_ENTRY_SIZE;
            let mut v = hdr(op, total);
            st32(&mut v[OP_HEADER_LEN + BIND_FIRST..], 3);
            st32(&mut v[OP_HEADER_LEN + BIND_COUNT..], COUNT);
            for i in 0..COUNT as usize {
                let e = OP_HEADER_LEN + BIND_ENTRIES + i * SAMPLER_LOD_BIND_ENTRY_SIZE;
                st32(&mut v[e..], 0x6363 + i as u32);
                // Clamps this decoder does not lift. They are here so a decoder
                // reading at the plain four-byte stride would pick one up as a
                // ref and fail the assertion below.
                st32(&mut v[e + 4..], 0x3e80_0000); // 0.25
                st32(&mut v[e + 8..], 0x3f40_0000); // 0.75
            }

            let c = decode(&v).unwrap_or_else(|e| panic!("op {op:#x}: {e:?}"));
            assert_eq!(c.kind, Kind::SetSampler, "op {op:#x}");
            assert_eq!(c.stage, stage, "op {op:#x}");
            assert!(c.has_sampler_lod, "op {op:#x}");
            assert_eq!(c.first, 3, "op {op:#x}");
            assert_eq!(
                c.ref_binds,
                vec![0x6363, 0x6364],
                "op {op:#x} read the entries at the wrong stride"
            );
            assert_eq!(c.sampler_ref, 0x6363, "op {op:#x}");
        }

        // The plain forms keep the four-byte stride and say they carry no
        // clamps, so the flag is the record's and not the family's.
        let total = OP_HEADER_LEN + BIND_ENTRIES + REF_BIND_ENTRY_SIZE;
        let mut v = hdr(wire::OPCODE_SET_VERTEX_SAMPLER, total);
        st32(&mut v[OP_HEADER_LEN + BIND_COUNT..], 1);
        st32(&mut v[OP_HEADER_LEN + BIND_ENTRIES..], 0x6363);
        let c = decode(&v).expect("plain sampler bind");
        assert!(!c.has_sampler_lod);
        assert_eq!(c.ref_binds, vec![0x6363]);
    }

    /// The accepted-opcode window ends exactly where Apple's render manifest
    /// does, computed rather than transcribed.
    ///
    /// The accepted window ends at the highest opcode in the wire render
    /// manifest (`OPCODE_SET_VERTEX_BUFFER_OFFSET_STRIDE`). A capture that
    /// adds a higher opcode fails here and names itself, rather than leaving
    /// a stale bound that refuses real records as "Apple does not write".
    #[test]
    fn the_accepted_window_ends_where_apples_render_manifest_does() {
        let highest = reims_vgpu_wire::manifest::MANIFEST
            .iter()
            .filter(|e| e.class == "PGSerializerRenderCommandEncoder")
            .flat_map(|e| e.opcodes.iter().copied())
            .max()
            .expect("the render encoder has opcodes in the manifest");
        assert_eq!(
            wire::OPCODE_SET_VERTEX_BUFFER_OFFSET_STRIDE,
            highest,
            "a render opcode above the accepted window would be refused as one \
             Apple does not write"
        );
    }

    /// Every render opcode Apple's serializer emits has a constant here, and
    /// this module names no opcode Apple does not emit.
    ///
    /// [`wire::OPCODE_SET_VERTEX_BUFFER_OFFSET_STRIDE`] bounds the window and this fills it. The two catch
    /// different failures, and the one this catches is the quieter: an opcode
    /// *inside* the accepted window that no arm claims does not get refused, it
    /// gets [`Kind::OtherAccepted`] — the catch-all whose `Ok` is not a decode,
    /// and which is what hid the sampler-LOD binds `0x80`/`0x71` behind a
    /// passing run. So a capture that adds `0x9b`..`0xa4` the way `TileShaders`
    /// did lands below the window's end, breaks nothing, and loses every record
    /// it names.
    ///
    /// The roster below is references rather than numbers, so no entry can
    /// carry a wrong *value*; what this test adds is that no entry can go
    /// *missing*. The reverse direction matters as much and is the shape of the
    /// `0x86`/`0x87` residency bug: an opcode named here and absent from the
    /// manifest is a number no capture supports.
    ///
    /// **`0x1a` used to be excluded from it, on a claim that was wrong.** This
    /// doc said the render-pass descriptor was "the live descriptor this
    /// device's own framing carries, not a record
    /// `PGSerializerRenderCommandEncoder` writes, and the manifest agrees by
    /// omitting it" — and the manifest omitted it only because no case had
    /// driven `writeDescriptor`, which emits it. Six more opcodes arrived with
    /// it, all of them pass properties behind a capability, and every one was
    /// reaching the catch-all.
    ///
    /// The general form is worth keeping: **"the manifest agrees" is not
    /// evidence when the manifest's silence is what is being explained.** An
    /// opcode absent from a capture-derived roster can be an opcode nobody
    /// captured.
    #[test]
    fn the_render_opcode_table_is_exactly_apples_render_manifest() {
        let device: &[(u32, &str)] = &[
            (wire::OPCODE_DRAW_WIDE, "wire::OPCODE_DRAW_WIDE"),
            (wire::OPCODE_DRAW, "wire::OPCODE_DRAW"),
            (
                wire::OPCODE_DRAW_INSTANCED_WIDE,
                "wire::OPCODE_DRAW_INSTANCED_WIDE",
            ),
            (wire::OPCODE_DRAW_INSTANCED, "wire::OPCODE_DRAW_INSTANCED"),
            (
                wire::OPCODE_DRAW_INSTANCED_BASE_WIDE,
                "wire::OPCODE_DRAW_INSTANCED_BASE_WIDE",
            ),
            (
                wire::OPCODE_DRAW_INSTANCED_BASE,
                "wire::OPCODE_DRAW_INSTANCED_BASE",
            ),
            (
                wire::OPCODE_DRAW_INDEXED_WIDE,
                "wire::OPCODE_DRAW_INDEXED_WIDE",
            ),
            (wire::OPCODE_DRAW_INDEXED, "wire::OPCODE_DRAW_INDEXED"),
            (
                wire::OPCODE_DRAW_INDEXED_INSTANCED_WIDE,
                "wire::OPCODE_DRAW_INDEXED_INSTANCED_WIDE",
            ),
            (
                wire::OPCODE_DRAW_INDEXED_INSTANCED,
                "wire::OPCODE_DRAW_INDEXED_INSTANCED",
            ),
            (
                wire::OPCODE_DRAW_INDEXED_INSTANCED_BASE_WIDE,
                "wire::OPCODE_DRAW_INDEXED_INSTANCED_BASE_WIDE",
            ),
            (
                wire::OPCODE_DRAW_INDEXED_INSTANCED_BASE,
                "wire::OPCODE_DRAW_INDEXED_INSTANCED_BASE",
            ),
            (
                wire::OPCODE_DRAW_PATCHES_WIDE,
                "wire::OPCODE_DRAW_PATCHES_WIDE",
            ),
            (wire::OPCODE_DRAW_PATCHES, "wire::OPCODE_DRAW_PATCHES"),
            (
                wire::OPCODE_DRAW_INDEXED_PATCHES,
                "wire::OPCODE_DRAW_INDEXED_PATCHES",
            ),
            (wire::OPCODE_DRAW_INDIRECT, "wire::OPCODE_DRAW_INDIRECT"),
            (
                wire::OPCODE_DRAW_INDEXED_INDIRECT,
                "wire::OPCODE_DRAW_INDEXED_INDIRECT",
            ),
            (
                wire::OPCODE_DRAW_PATCHES_INDIRECT,
                "wire::OPCODE_DRAW_PATCHES_INDIRECT",
            ),
            (
                wire::OPCODE_DRAW_INDEXED_PATCHES_INDIRECT,
                "wire::OPCODE_DRAW_INDEXED_PATCHES_INDIRECT",
            ),
            (
                wire::OPCODE_EXECUTE_COMMANDS_INDIRECT,
                "wire::OPCODE_EXECUTE_COMMANDS_INDIRECT",
            ),
            (
                wire::OPCODE_EXECUTE_COMMANDS_RANGE,
                "wire::OPCODE_EXECUTE_COMMANDS_RANGE",
            ),
            (
                wire::OPCODE_MEMORY_BARRIER_RESOURCES,
                "wire::OPCODE_MEMORY_BARRIER_RESOURCES",
            ),
            (
                wire::OPCODE_MEMORY_BARRIER_SCOPE,
                "wire::OPCODE_MEMORY_BARRIER_SCOPE",
            ),
            (wire::OPCODE_UPDATE_FENCE, "wire::OPCODE_UPDATE_FENCE"),
            (wire::OPCODE_WAIT_FOR_FENCE, "wire::OPCODE_WAIT_FOR_FENCE"),
            (wire::OPCODE_USE_HEAP, "wire::OPCODE_USE_HEAP"),
            (wire::OPCODE_SET_BLEND_COLOR, "wire::OPCODE_SET_BLEND_COLOR"),
            (
                wire::OPCODE_SET_COLOR_STORE_ACTION,
                "wire::OPCODE_SET_COLOR_STORE_ACTION",
            ),
            (
                wire::OPCODE_SET_COLOR_STORE_ACTION_OPTIONS,
                "wire::OPCODE_SET_COLOR_STORE_ACTION_OPTIONS",
            ),
            (
                wire::OPCODE_SET_DEPTH_STENCIL_STATE,
                "wire::OPCODE_SET_DEPTH_STENCIL_STATE",
            ),
            (
                wire::OPCODE_SET_DEPTH_STORE_ACTION,
                "wire::OPCODE_SET_DEPTH_STORE_ACTION",
            ),
            (
                wire::OPCODE_SET_DEPTH_STORE_ACTION_OPTIONS,
                "wire::OPCODE_SET_DEPTH_STORE_ACTION_OPTIONS",
            ),
            (wire::OPCODE_SET_CULL_MODE, "wire::OPCODE_SET_CULL_MODE"),
            (wire::OPCODE_SET_DEPTH_BIAS, "wire::OPCODE_SET_DEPTH_BIAS"),
            (
                wire::OPCODE_SET_DEPTH_CLIP_MODE,
                "wire::OPCODE_SET_DEPTH_CLIP_MODE",
            ),
            (
                wire::OPCODE_SET_FRAGMENT_BUFFER,
                "wire::OPCODE_SET_FRAGMENT_BUFFER",
            ),
            (
                wire::OPCODE_SET_FRAGMENT_BUFFER_OFFSET,
                "wire::OPCODE_SET_FRAGMENT_BUFFER_OFFSET",
            ),
            (
                wire::OPCODE_SET_FRAGMENT_SAMPLER,
                "wire::OPCODE_SET_FRAGMENT_SAMPLER",
            ),
            (
                wire::OPCODE_SET_FRAGMENT_SAMPLER_LOD,
                "wire::OPCODE_SET_FRAGMENT_SAMPLER_LOD",
            ),
            (
                wire::OPCODE_SET_FRAGMENT_TEXTURE,
                "wire::OPCODE_SET_FRAGMENT_TEXTURE",
            ),
            (
                wire::OPCODE_SET_FRONT_FACING,
                "wire::OPCODE_SET_FRONT_FACING",
            ),
            (
                wire::OPCODE_SET_RENDER_PIPELINE_STATE,
                "wire::OPCODE_SET_RENDER_PIPELINE_STATE",
            ),
            (wire::OPCODE_SET_SCISSOR, "wire::OPCODE_SET_SCISSOR"),
            (
                wire::OPCODE_SET_SCISSOR_RECTS,
                "wire::OPCODE_SET_SCISSOR_RECTS",
            ),
            (
                wire::OPCODE_SET_STENCIL_REFERENCE,
                "wire::OPCODE_SET_STENCIL_REFERENCE",
            ),
            (
                wire::OPCODE_SET_STENCIL_STORE_ACTION,
                "wire::OPCODE_SET_STENCIL_STORE_ACTION",
            ),
            (
                wire::OPCODE_SET_STENCIL_STORE_ACTION_OPTIONS,
                "wire::OPCODE_SET_STENCIL_STORE_ACTION_OPTIONS",
            ),
            (
                wire::OPCODE_SET_TESSELLATION_FACTOR_BUFFER,
                "wire::OPCODE_SET_TESSELLATION_FACTOR_BUFFER",
            ),
            (
                wire::OPCODE_SET_TESSELLATION_FACTOR_SCALE,
                "wire::OPCODE_SET_TESSELLATION_FACTOR_SCALE",
            ),
            (
                wire::OPCODE_SET_TRIANGLE_FILL_MODE,
                "wire::OPCODE_SET_TRIANGLE_FILL_MODE",
            ),
            (
                wire::OPCODE_SET_VERTEX_BUFFER,
                "wire::OPCODE_SET_VERTEX_BUFFER",
            ),
            (
                wire::OPCODE_SET_VERTEX_BUFFER_OFFSET,
                "wire::OPCODE_SET_VERTEX_BUFFER_OFFSET",
            ),
            (
                wire::OPCODE_SET_VERTEX_SAMPLER,
                "wire::OPCODE_SET_VERTEX_SAMPLER",
            ),
            (
                wire::OPCODE_SET_VERTEX_SAMPLER_LOD,
                "wire::OPCODE_SET_VERTEX_SAMPLER_LOD",
            ),
            (
                wire::OPCODE_SET_VERTEX_TEXTURE,
                "wire::OPCODE_SET_VERTEX_TEXTURE",
            ),
            (wire::OPCODE_SET_VIEWPORT, "wire::OPCODE_SET_VIEWPORT"),
            (wire::OPCODE_SET_VIEWPORTS, "wire::OPCODE_SET_VIEWPORTS"),
            (
                wire::OPCODE_SET_VISIBILITY_RESULT_MODE,
                "wire::OPCODE_SET_VISIBILITY_RESULT_MODE",
            ),
            (wire::OPCODE_TEXTURE_BARRIER, "wire::OPCODE_TEXTURE_BARRIER"),
            (wire::OPCODE_SET_LINE_WIDTH, "wire::OPCODE_SET_LINE_WIDTH"),
            (wire::OPCODE_USE_RESOURCE, "wire::OPCODE_USE_RESOURCE"),
            (
                wire::OPCODE_SET_VERTEX_AMPLIFICATION_MODE,
                "wire::OPCODE_SET_VERTEX_AMPLIFICATION_MODE",
            ),
            (
                wire::OPCODE_SET_VERTEX_AMPLIFICATION_COUNT,
                "wire::OPCODE_SET_VERTEX_AMPLIFICATION_COUNT",
            ),
            (
                wire_tile::OPCODE_DISPATCH_THREADS_PER_TILE,
                "wire_tile::OPCODE_DISPATCH_THREADS_PER_TILE",
            ),
            (
                wire_tile::OPCODE_SET_TILE_THREADGROUP_MEMORY,
                "wire_tile::OPCODE_SET_TILE_THREADGROUP_MEMORY",
            ),
            (
                wire_tile::OPCODE_SET_TILE_BUFFER,
                "wire_tile::OPCODE_SET_TILE_BUFFER",
            ),
            (
                wire_tile::OPCODE_SET_TILE_BUFFER_OFFSET,
                "wire_tile::OPCODE_SET_TILE_BUFFER_OFFSET",
            ),
            (
                wire_tile::OPCODE_SET_TILE_SAMPLER,
                "wire_tile::OPCODE_SET_TILE_SAMPLER",
            ),
            (
                wire_tile::OPCODE_SET_TILE_SAMPLER_LOD,
                "wire_tile::OPCODE_SET_TILE_SAMPLER_LOD",
            ),
            (
                wire_tile::OPCODE_SET_TILE_TEXTURE,
                "wire_tile::OPCODE_SET_TILE_TEXTURE",
            ),
            (
                wire_tile::OPCODE_DISPATCH_THREADS_PER_TILE_IN_REGION,
                "wire_tile::OPCODE_DISPATCH_THREADS_PER_TILE_IN_REGION",
            ),
            (
                wire_tile::OPCODE_DISPATCH_THREADS_PER_TILE_IN_REGION_RT_INDEX,
                "wire_tile::OPCODE_DISPATCH_THREADS_PER_TILE_IN_REGION_RT_INDEX",
            ),
            (
                wire_tile::OPCODE_GET_TILE_DIMENSIONS,
                "wire_tile::OPCODE_GET_TILE_DIMENSIONS",
            ),
            (
                wire::OPCODE_SET_VERTEX_BUFFER_STRIDE,
                "wire::OPCODE_SET_VERTEX_BUFFER_STRIDE",
            ),
            (
                wire::OPCODE_SET_VERTEX_BUFFER_OFFSET_STRIDE,
                "wire::OPCODE_SET_VERTEX_BUFFER_OFFSET_STRIDE",
            ),
            (
                wire_pass::OPCODE_RENDER_PASS,
                "wire_pass::OPCODE_RENDER_PASS",
            ),
            (
                wire_pass::OPCODE_DEFAULT_RASTER_SAMPLE_COUNT,
                "wire_pass::OPCODE_DEFAULT_RASTER_SAMPLE_COUNT",
            ),
            (
                wire_pass::OPCODE_SAMPLE_POSITIONS,
                "wire_pass::OPCODE_SAMPLE_POSITIONS",
            ),
            (
                wire_pass::OPCODE_RASTERIZATION_RATE_MAP,
                "wire_pass::OPCODE_RASTERIZATION_RATE_MAP",
            ),
            (
                wire_pass::OPCODE_IMAGEBLOCK_SAMPLE_LENGTH,
                "wire_pass::OPCODE_IMAGEBLOCK_SAMPLE_LENGTH",
            ),
            (
                wire_pass::OPCODE_THREADGROUP_MEMORY_LENGTH,
                "wire_pass::OPCODE_THREADGROUP_MEMORY_LENGTH",
            ),
            (wire_pass::OPCODE_TILE_SIZE, "wire_pass::OPCODE_TILE_SIZE"),
        ];

        let mut apple: Vec<u32> = reims_vgpu_wire::manifest::MANIFEST
            .iter()
            .filter(|e| e.class == "PGSerializerRenderCommandEncoder")
            .flat_map(|e| e.opcodes.iter().copied())
            .collect();
        apple.sort_unstable();
        apple.dedup();

        for (op, name) in device {
            assert!(
                apple.contains(op),
                "{name} = {op:#x} is an opcode Apple's render manifest does not \
                 list, so no capture supports it"
            );
        }
        for op in &apple {
            assert!(
                device.iter().any(|(d, _)| d == op),
                "Apple's serializer emits render opcode {op:#x} and this module \
                 names no constant for it, so it reaches Kind::OtherAccepted and \
                 every record carrying it is lost without a refusal"
            );
        }
        assert_eq!(
            device.len(),
            apple.len(),
            "the roster has a duplicate entry"
        );
    }

    /// Residency uses the wire opcodes Apple's serializer writes (`0x1b` /
    /// `0x89`), not the old barrier-neighbourhood numbers `0x86`/`0x87`.
    #[test]
    fn the_residency_opcodes_are_the_ones_apples_serializer_writes() {
        use reims_vgpu_wire::ops::render as wire;
        assert_eq!(wire::OPCODE_USE_HEAP, 0x1b);
        assert_eq!(wire::OPCODE_USE_RESOURCE, 0x89);
        // Barrier neighbourhood: nothing here may claim these as residency.
        assert_ne!(wire::OPCODE_USE_HEAP, 0x86);
        assert_ne!(wire::OPCODE_USE_RESOURCE, 0x87);
    }

    /// The refs of a residency record start at a different offset on each form,
    /// and the count-led extent is checked rather than assumed.
    ///
    /// `useHeap:` puts its array at `+6`, which is not a multiple of four: a
    /// record read with `useResource:`'s `+8` accepts two bytes fewer than it
    /// needs, so the length check is what separates the two layouts.
    #[test]
    fn a_residency_record_is_bounded_by_its_own_count() {
        for (op, refs_at, kind) in [
            (
                wire::OPCODE_USE_RESOURCE,
                USE_RESOURCE_REFS,
                Kind::UseResource,
            ),
            (wire::OPCODE_USE_HEAP, USE_HEAP_REFS, Kind::UseHeap),
        ] {
            let body = |count: u32, entries: usize| {
                let mut v = hdr(op, OP_HEADER_LEN + refs_at + entries * REF_BIND_ENTRY_SIZE);
                st32(&mut v[OP_HEADER_LEN + RESIDENCY_COUNT..], count);
                v
            };

            let c = decode(&body(2, 2)).unwrap_or_else(|e| panic!("op {op:#x}: {e:?}"));
            assert_eq!(c.kind, kind, "op {op:#x}");
            assert_eq!(c.count, 2, "op {op:#x}");

            // One entry short of what the count claims.
            assert_eq!(
                decode(&body(2, 1)).unwrap_err(),
                DecodeStatus::ErrShort,
                "op {op:#x} accepted a record one ref shorter than its count"
            );
            // Past the bind cap and still well-formed. This record names no
            // table slot, so a bind-table cap is not its bound; refusing here
            // would drop a residency declaration a guest may legitimately make.
            let big = 40u32;
            let c = decode(&body(big, big as usize))
                .unwrap_or_else(|e| panic!("op {op:#x} refused {big} resources: {e:?}"));
            assert_eq!(c.count, big, "op {op:#x}");
            // A count whose byte length overflows `usize` must not wrap into a
            // bound the record satisfies.
            assert_eq!(
                decode(&body(u32::MAX, 1)).unwrap_err(),
                DecodeStatus::ErrShort,
                "op {op:#x} accepted a count whose array cannot exist"
            );
            // The head itself, with no room for the array at all.
            assert_eq!(
                decode(&hdr(op, OP_HEADER_LEN + 4)).unwrap_err(),
                DecodeStatus::ErrShort,
                "op {op:#x} accepted a record with no room for its refs"
            );
        }
    }

    /// `0x86` and `0x87` are claimed by no arm and stay visible.
    ///
    /// They are still inside the accepted window, so they decode — but as
    /// [`Kind::OtherAccepted`], which `runtime::exec` reports on the failure
    /// channel. The residency kinds had no executor arm at all, so reading them
    /// as residency was strictly worse than not decoding them: it removed them
    /// from the one net that would have named them.
    #[test]
    fn the_barrier_neighbourhood_opcodes_are_not_claimed_as_residency() {
        for op in [0x86u32, 0x87] {
            let c =
                decode(&hdr(op, OP_HEADER_LEN + 16)).unwrap_or_else(|e| panic!("{op:#x}: {e:?}"));
            assert_eq!(
                c.kind,
                Kind::OtherAccepted,
                "{op:#x} is claimed by an arm again"
            );
        }
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
