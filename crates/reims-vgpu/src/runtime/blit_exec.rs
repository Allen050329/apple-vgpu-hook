//! Product-path execution of blit fill/copy commands against guest backings.
//!
//! Supported now:
//! - `fillBuffer` (0x132) on type-1 buffers
//! - `copyFromBuffer:toBuffer:` (0x12d) on type-1 buffers
//! - Rectangular buffer↔texture / texture↔texture copies on linear type-2/3
//! - Same rectangular copies with **type-11 IOSurface** texture endpoints
//!   (level 0, slice 0, depth 1) via mapping page tables; multi-plane (biplanar)
//!   sample windows from cached `sIOSurfaceDeviceDescriptor` selected by texture
//!   geometry (width/height/bpe), not a wire plane index
//! - **Type-8 texture views** as copy endpoints: unswizzled views over type-2/3
//!   or type-11 bases; multi-level / array / non-2D Metal types when geometry matches
//!   (type-11 bases remain single-level / single-slice — see below)
//! - **`MTLBlitOption`**: None; DepthFromDepthStencil / StencilFromDepthStencil;
//!   combined DS plane packing on linear GVA; unknown bits / RowLinearPVRTC fail
//! - **`0x13e` whole-surface** texture→texture: for each level in
//!   `[sourceLevel, sourceLevel+levelCount)`:
//!   - **depth-1 (array/2D):** full `width×height` across `sliceCount` consecutive
//!     slices
//!   - **depth>1 (3D volume):** Metal requires `sliceCount==1` and slices 0;
//!     copies full `width×height×depth` of that mip (depth planes via
//!     `bytes_per_image`); linear type-2/3 only
//!   - zero `sliceCount`/`levelCount` are Metal no-ops
//! - **Fences** `0x13c` update / `0x13d` wait: blit-fence domain generation via
//!   [`crate::runtime::plan::event_sync`]; waits that are not yet satisfied are
//!   soft-pending (do not block drain), matching the unified-memory in-order path
//!
//! Not executed (fail visibly / soft miss):
//! - swizzled type-8 views (contract: blit rejects remapped swizzle materialization)
//! - multisample view types
//! - RowLinearPVRTC / unknown option bits
//! - overlapping same-buffer B2B windows
//! - type-11 multi-mip / non-zero level or slice — **not a missing feature**: Metal
//!   forbids mipmapped IOSurface textures (`newTextureWithDescriptor:iosurface:`
//!   rejects `mipmapLevelCount > 1`). Product path fail-closes; do not invent a
//!   pyramid layout in the mapping.
//! - 3D whole-surface with `sliceCount!=1`, non-zero slices, or type-11 endpoint

use crate::contract::pixel_format::{self, MTL_FORMAT_BGRA8_UNORM};
use crate::model::DeviceState;
use crate::observe::Decline;
use crate::runtime::decode::blit::{
    self, BlitAspect, Command, CopyKind, Kind, OP_UPDATE_FENCE, OP_WAIT_FENCE,
};
use crate::runtime::decode::resource::{
    decode_buffer_descriptor, decode_iosurface_texture_descriptor, decode_texture_descriptor,
    decode_texture_view_descriptor, texture_view_type_is_3d, texture_view_type_supported,
    texture_view_type_uses_slices, Descriptor as ResourceDescriptor, OBJECT_TYPE_BUFFER,
    OBJECT_TYPE_IOSURFACE, OBJECT_TYPE_TEXTURE, OBJECT_TYPE_TEXTURE_VARIANT,
    OBJECT_TYPE_TEXTURE_VIEW, TEXTURE_VIEW_MTL_TYPE_2D,
};
use crate::runtime::fence_exec::{self, FenceStatus};
use crate::runtime::gva_mem;
use crate::runtime::host::{HostMemory, HostOps};
use crate::runtime::mapper;
use crate::runtime::mapping_write;
use crate::runtime::metal_draw::{self, host_alloc_len};
use crate::runtime::objects;
use crate::runtime::plan::event_sync::{Domain as FenceDomain, FenceAction};

/// Cap on type-8 view → base → … chains (views of views).
const VIEW_RESOLVE_DEPTH_CAP: u32 = 4;

/// Chunk size for fill/copy host staging (bounded guest IO).
const CHUNK: usize = 64 * 1024;

/// Outcome of a product-path blit fill/copy/fence attempt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlitStatus {
    Ok,
    /// Missing object, wrong kind, or unreadable descriptor.
    MissingResource,
    /// Opcode / options / view / slice / 3D / type-11 not on this path.
    Unsupported,
    /// Offset/length/extent outside allocation or level bounds.
    Bounds,
    /// Guest GVA read/write failed.
    GuestIo,
    /// Pathological size or host staging cap.
    Capacity,
    /// Zero-size fill or zero-extent rectangular copy (Metal no-op → soft ok).
    ZeroExtent,
    /// Same buffer, overlapping source/destination windows.
    Overlap,
    /// Fence wait not yet satisfied (soft; does not block drain).
    FencePending,
}

impl crate::observe::Refusal for BlitStatus {
    /// The reason comes from the thread-local channel, not from the variant.
    ///
    /// This rail is the crate's largest refusal surface — **177 distinct checks
    /// across 182 sites**, collapsing into eight coarse statuses — so the specific
    /// cause has always travelled beside the value in [`BLIT_FAIL_REASON`] rather
    /// than inside it. That is a legitimate shape (a 177-arm `slug()` is not a
    /// thing anyone writes) and the registry reads the vocabulary at the `br(`
    /// sites, so every one of the 177 is counted and unique crate-wide.
    ///
    /// What was *not* legitimate: an uninstrumented site returned a coarse status
    /// with the channel still empty, and the dispatch line rendered a bare
    /// `reason=` with nothing after it — unfindable by grep and indistinguishable
    /// from a missing field. That case is now the registered `blit_unattributed`,
    /// which names the gap instead of hiding it.
    ///
    /// Read on the same thread that ran the blit, which both dispatch sites do
    /// immediately after the call. `Ok`, `ZeroExtent` and `FencePending` are
    /// control flow — the first two are the dispatch site's success arm, the third
    /// is a soft wait the guest re-polls — and this reproduces exactly the two
    /// sites' previous log conditions.
    fn refusal(&self) -> Option<&'static str> {
        match self {
            Self::Ok | Self::ZeroExtent | Self::FencePending => None,
            _ => Some(match blit_fail_reason() {
                "" => "blit_unattributed",
                slug => slug,
            }),
        }
    }
}

thread_local! {
    /// The specific reason slug for the most recent non-`Ok` [`BlitStatus`], set at
    /// the failing site so the single dispatch-site failure line can name *which* of
    /// the many checks that collapse into a coarse status actually fired. Cleared at
    /// the start of every `execute_blit`/`execute_blit_fence` so an uninstrumented
    /// site reports empty rather than a stale value from a prior command. Genuine
    /// failures only reach the dispatch log, so this never floods a healthy boot.
    static BLIT_FAIL_REASON: std::cell::Cell<&'static str> = const { std::cell::Cell::new("") };
}

/// Record `reason` for a non-`Ok` [`BlitStatus`] at the failing site and return
/// that status unchanged. Use at every `return Err(..)` / `.ok_or_else(..)` site that
/// collapses a distinct cause into a coarse status.
#[inline]
fn br(status: BlitStatus, reason: &'static str) -> BlitStatus {
    BLIT_FAIL_REASON.with(|r| r.set(reason));
    status
}

/// Read the last recorded blit-failure reason without clearing it, so several call
/// sites (a path-specific line plus the dispatch summary) can name the same cause.
/// The channel is reset at the start of the next command via [`clear_blit_fail_reason`],
/// so a stale reason cannot leak across commands. Read this only on the failure path.
pub fn blit_fail_reason() -> &'static str {
    BLIT_FAIL_REASON.with(|r| r.get())
}

/// Reset the reason channel at entry to a blit command so an uninstrumented failure
/// reports empty rather than a stale reason from a prior command.
#[inline]
fn clear_blit_fail_reason() {
    BLIT_FAIL_REASON.with(|r| r.set(""));
}

/// Dedup set for the `tex_wrong_type` enrichment line, keyed by
/// `(task_id, texture_ref, object_type)`. A blit that binds a non-texture ref
/// fails once per draw (observed ~67/six-app-launch), so the enrichment must
/// dedup — but the bare `reason=tex_wrong_type` dispatch slug hides *what* the
/// object actually is (buffer bound as texture = a decode/tracking bug vs. a
/// legit guest race), which is the load-bearing field for diagnosis.
static TEX_WRONG_TYPE_SEEN: std::sync::OnceLock<
    std::sync::Mutex<std::collections::HashSet<(u32, u32, u8)>>,
> = std::sync::OnceLock::new();

/// Emit ONE always-on `blit tex_wrong_type` line per distinct
/// `(task, ref, object_type)` naming the actual object type a blit tried to use
/// as a texture. Deduped so a per-draw repeat cannot flood. Returns whether it
/// emitted (tests use it). Diagnostic only.
fn note_tex_wrong_type(
    task_id: u32,
    texture_ref: u32,
    object_type: u8,
    level: u16,
    slice: u16,
) -> bool {
    let set = TEX_WRONG_TYPE_SEEN.get_or_init(|| std::sync::Mutex::new(Default::default()));
    if let Ok(mut g) = set.lock() {
        if !g.insert((task_id, texture_ref, object_type)) {
            return false;
        }
    }
    crate::observe::fail(format!(
        "blit tex_wrong_type task={task_id} ref={texture_ref} object_type={object_type} level={level} slice={slice}"
    ));
    true
}

#[cfg(test)]
fn reset_tex_wrong_type_dedup_for_test() {
    if let Some(set) = TEX_WRONG_TYPE_SEEN.get() {
        if let Ok(mut g) = set.lock() {
            g.clear();
        }
    }
}

/// Dedup set for the `t5_view_decode` diagnostic, keyed by surface id.
static T5_DECODE_FAIL_SEEN: std::sync::OnceLock<std::sync::Mutex<std::collections::HashSet<u32>>> =
    std::sync::OnceLock::new();

/// One always-on diagnostic per surface id when a type-5 RefTexture's view
/// record fails to decode: dumps `desc_len` + head hex so the exact blit-path
/// type-5 layout can be read offline (the decoder wants tag 0x42 at +0x14, 2D
/// nonzero geom, depth==1). Deduped so a per-draw repeat cannot flood.
fn note_t5_decode_fail(sid: u32, bytes: &[u8]) {
    let set = T5_DECODE_FAIL_SEEN.get_or_init(|| std::sync::Mutex::new(Default::default()));
    if let Ok(mut g) = set.lock() {
        if !g.insert(sid) {
            return;
        }
    }
    let n = bytes.len().min(40);
    let mut hex = String::with_capacity(n * 2);
    for b in &bytes[..n] {
        use std::fmt::Write as _;
        let _ = write!(hex, "{b:02x}");
    }
    crate::observe::fail(format!(
        "blit t5_view_decode sid={sid} desc_len={} head_hex={hex}",
        bytes.len()
    ));
}

/// Dedup set for the `t2t_overlap` enrichment, keyed by `(task, src_ref, dst_ref)`.
static T2T_OVERLAP_SEEN: std::sync::OnceLock<
    std::sync::Mutex<std::collections::HashSet<(u32, u32, u32)>>,
> = std::sync::OnceLock::new();

/// Emit ONE always-on `blit t2t_overlap` line per distinct
/// `(task, src_ref, dst_ref)` carrying the load-bearing overlap geometry so a
/// genuine self-overlap (same allocation, live bytes collide — undefined in
/// Metal, correctly rejected) can be told apart from a false positive.
///
/// The reject uses a COMPRESSED bounding span (`row_bytes*copy_h*copy_d`) that
/// ignores `row_stride` gaps, so it can misjudge a strided sub-rect copy in
/// either direction. Logging `row_bytes` vs `row_stride` and both offsets lets
/// a future boot decide whether the check needs to become stride-precise.
/// Deduped so a per-draw repeat cannot flood. Diagnostic only.
#[allow(clippy::too_many_arguments)]
fn note_t2t_overlap(
    task_id: u32,
    src_ref: u32,
    dst_ref: u32,
    src_off: u64,
    dst_off: u64,
    row_bytes: u64,
    row_stride: u64,
    copy_h: u64,
    copy_d: u64,
) -> bool {
    let set = T2T_OVERLAP_SEEN.get_or_init(|| std::sync::Mutex::new(Default::default()));
    if let Ok(mut g) = set.lock() {
        if !g.insert((task_id, src_ref, dst_ref)) {
            return false;
        }
    }
    let span = row_bytes.saturating_mul(copy_h).saturating_mul(copy_d);
    let strided = if row_bytes < row_stride { 1 } else { 0 };
    crate::observe::fail(format!(
        "blit t2t_overlap task={task_id} src_ref={src_ref} dst_ref={dst_ref} \
         src_off={src_off} dst_off={dst_off} row_bytes={row_bytes} \
         row_stride={row_stride} copy_h={copy_h} copy_d={copy_d} span={span} strided={strided}"
    ));
    true
}

/// Dedup set for the `copy_region_*_io` enrichment, keyed by
/// `(task, gva_page, is_write)`.
static COPY_REGION_IO_SEEN: std::sync::OnceLock<
    std::sync::Mutex<std::collections::HashSet<(u32, u64, bool)>>,
> = std::sync::OnceLock::new();

/// Emit ONE always-on `blit copy_region_io` line per distinct
/// `(task, failing-gva-page, is_write)` naming the exact guest address a
/// rectangular copy row could not read/write. A guest that tears down the
/// destination surface mid-copy (teardown race) shows a plausible-but-unmapped
/// gva; a decode/geometry bug shows a wild gva. Deduped per page so a strided
/// multi-row failure cannot flood. Diagnostic only.
fn note_copy_region_io(
    task_id: u32,
    is_write: bool,
    gva: u64,
    row: u64,
    image: u64,
    row_bytes: u64,
    page_shift: u32,
) -> bool {
    let set = COPY_REGION_IO_SEEN.get_or_init(|| std::sync::Mutex::new(Default::default()));
    let key = (task_id, gva >> page_shift, is_write);
    if let Ok(mut g) = set.lock() {
        if !g.insert(key) {
            return false;
        }
    }
    let dir = if is_write { "write" } else { "read" };
    crate::observe::fail(format!(
        "blit copy_region_io dir={dir} task={task_id} gva={gva:#x} row={row} image={image} row_bytes={row_bytes}"
    ));
    true
}

struct LinearBuffer {
    gva: u64,
    size: u64,
}

struct LinearTextureLevel {
    /// Allocation base GVA (`handle << page_shift` for the device).
    base_gva: u64,
    alloc_size: u64,
    level_offset: u64,
    row_stride: u64,
    /// Byte stride between array slices / cube faces at this level.
    /// 0 means single-slice (no slice offset applied).
    slice_stride: u64,
    /// Absolute array slice / cube face selected for this resolve.
    slice_index: u32,
    width: u32,
    height: u32,
    depth: u32,
    bpp: u32,
    pixel_format: u16,
}

/// Type-11 IOSurface texture (single level, 2D).
///
/// Metal rejects mipmapped IOSurface textures (`mipmapLevelCount > 1` fails
/// descriptor validation on `newTextureWithDescriptor:iosurface:plane:`). The
/// product path therefore never materializes non-zero mip levels or invents a
/// multi-mip packing inside the mapping — non-zero `level`/`slice` fails closed.
///
/// Multi-plane (biplanar 420): sample window comes from the cached guest device
/// descriptor via geometry match (texture width/height/bpe); `surface_offset` is
/// the plane base in the shared mapping.
struct Type11Texture {
    mapping_id: u32,
    width: u32,
    height: u32,
    /// Byte offset of this texture/plane in the mapping allocation.
    surface_offset: u64,
    /// IOSurface-aligned surface row stride (bytes).
    row_stride: u64,
    /// Exclusive end of the sample window (for page-span planning).
    span_end: u64,
    bpp: u32,
    pixel_format: u16,
}

enum TextureBacking {
    Linear(LinearTextureLevel),
    Type11(Type11Texture),
}

impl TextureBacking {
    fn width(&self) -> u32 {
        match self {
            TextureBacking::Linear(t) => t.width,
            TextureBacking::Type11(t) => t.width,
        }
    }
    fn height(&self) -> u32 {
        match self {
            TextureBacking::Linear(t) => t.height,
            TextureBacking::Type11(t) => t.height,
        }
    }
    fn depth(&self) -> u32 {
        match self {
            TextureBacking::Linear(t) => t.depth,
            TextureBacking::Type11(_) => 1,
        }
    }
    fn bpp(&self) -> u32 {
        match self {
            TextureBacking::Linear(t) => t.bpp,
            TextureBacking::Type11(t) => t.bpp,
        }
    }
    fn pixel_format(&self) -> u16 {
        match self {
            TextureBacking::Linear(t) => t.pixel_format,
            TextureBacking::Type11(t) => t.pixel_format,
        }
    }
    fn is_type11(&self) -> bool {
        matches!(self, TextureBacking::Type11(_))
    }
}

impl LinearTextureLevel {
    fn bytes_per_image(&self) -> Option<u64> {
        self.row_stride.checked_mul(self.height as u64)
    }

    /// Byte offset of texel origin (x,y,z) within the allocation (includes slice).
    fn texel_offset(&self, x: u64, y: u64, z: u64) -> Option<u64> {
        let bpi = self.bytes_per_image()?;
        let row = y.checked_mul(self.row_stride)?;
        let col = x.checked_mul(self.bpp as u64)?;
        let plane = z.checked_mul(bpi)?;
        let slice = if self.slice_index == 0 || self.slice_stride == 0 {
            0u64
        } else {
            (self.slice_index as u64).checked_mul(self.slice_stride)?
        };
        self.level_offset
            .checked_add(slice)?
            .checked_add(plane)?
            .checked_add(row)?
            .checked_add(col)
    }
}

/// Contiguous Metal packing for one array slice / cube face at a mip level:
/// `row_stride * height * depth` (depth planes sit inside the slice).
fn derived_slice_stride(row_stride: u64, height: u32, depth: u32) -> Option<u64> {
    let h = height.max(1) as u64;
    let d = depth.max(1) as u64;
    row_stride.checked_mul(h)?.checked_mul(d)
}

fn resolve_buffer<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    buffer_ref: u32,
) -> Result<LinearBuffer, BlitStatus> {
    if buffer_ref == 0 {
        return Err(br(BlitStatus::MissingResource, "buf_ref_zero"));
    }
    let Some(entry) = objects::lookup_list_entry(state, host, task_id, buffer_ref) else {
        return Err(br(BlitStatus::MissingResource, "buf_no_list_entry"));
    };
    if entry.object_type != OBJECT_TYPE_BUFFER {
        return Err(br(BlitStatus::MissingResource, "buf_wrong_type"));
    }
    let Some(bytes) = objects::read_descriptor(state, host, task_id, &entry) else {
        return Err(br(BlitStatus::MissingResource, "buf_desc_read"));
    };
    let Ok(buf) = decode_buffer_descriptor(&bytes) else {
        return Err(br(BlitStatus::MissingResource, "buf_desc_decode"));
    };
    let Some((gva, size)) = buf.backing_gva_size(state.page_shift) else {
        return Err(br(BlitStatus::MissingResource, "buf_no_backing"));
    };
    Ok(LinearBuffer { gva, size })
}

fn resolve_texture_backing<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    texture_ref: u32,
    level: u16,
    slice: u16,
) -> Result<TextureBacking, BlitStatus> {
    resolve_texture_backing_depth(state, host, task_id, texture_ref, level, slice, 0)
}

fn resolve_texture_backing_depth<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    texture_ref: u32,
    level: u16,
    slice: u16,
    depth: u32,
) -> Result<TextureBacking, BlitStatus> {
    if texture_ref == 0 {
        return Err(br(BlitStatus::MissingResource, "tex_ref_zero"));
    }
    if depth > VIEW_RESOLVE_DEPTH_CAP {
        return Err(br(BlitStatus::Unsupported, "tex_view_depth_cap"));
    }
    let Some(entry) = objects::lookup_list_entry(state, host, task_id, texture_ref) else {
        return Err(br(BlitStatus::MissingResource, "tex_no_list_entry"));
    };

    // Type-8 view → base texture (unswizzled; multi-level / array / non-2D allowed).
    if entry.object_type == OBJECT_TYPE_TEXTURE_VIEW {
        let Some(bytes) = objects::read_descriptor(state, host, task_id, &entry) else {
            return Err(br(BlitStatus::MissingResource, "view_desc_read"));
        };
        let Ok(view) = decode_texture_view_descriptor(&bytes) else {
            return Err(br(BlitStatus::MissingResource, "view_desc_decode"));
        };
        if view.base_texture_ref == 0 {
            return Err(br(BlitStatus::MissingResource, "view_base_ref_zero"));
        }
        // Blit rejects swizzled materialization (contract).
        if view.has_swizzle {
            let plan = pixel_format::swizzle_plan(&view.swizzle)
                .ok_or_else(|| br(BlitStatus::Unsupported, "view_swizzle_plan"))?;
            if !pixel_format::swizzle_is_identity(&plan) {
                return Err(br(BlitStatus::Unsupported, "view_swizzle_nonident"));
            }
        }
        let view_type = if view.has_texture_type {
            if !texture_view_type_supported(view.texture_type) {
                return Err(br(BlitStatus::Unsupported, "view_type_unsupported"));
            }
            view.texture_type
        } else {
            TEXTURE_VIEW_MTL_TYPE_2D
        };
        // Relative command level → absolute on the base (multi-level ranges ok).
        let rel_level = level as u64;
        let level_count = if view.has_levels {
            if view.level_count == 0 {
                1
            } else {
                view.level_count
            }
        } else {
            // Simple form: no level range; command level is absolute on base.
            u64::MAX
        };
        if view.has_levels && rel_level >= level_count {
            return Err(br(BlitStatus::Bounds, "view_level_oob"));
        }
        let abs_level = if view.has_levels {
            view.level_base
                .checked_add(rel_level)
                .ok_or_else(|| br(BlitStatus::Bounds, "view_level_overflow"))?
        } else {
            rel_level
        };
        if abs_level > u16::MAX as u64 {
            return Err(br(BlitStatus::Bounds, "view_level_u16"));
        }
        // Relative command slice → absolute (array / cube faces).
        let rel_slice = slice as u64;
        let slice_count = if view.has_slices {
            if view.slice_count == 0 {
                1
            } else {
                view.slice_count
            }
        } else {
            u64::MAX
        };
        if view.has_slices && rel_slice >= slice_count {
            return Err(br(BlitStatus::Bounds, "view_slice_oob"));
        }
        let abs_slice = if view.has_slices {
            view.slice_base
                .checked_add(rel_slice)
                .ok_or_else(|| br(BlitStatus::Bounds, "view_slice_overflow"))?
        } else {
            rel_slice
        };
        if abs_slice > u16::MAX as u64 {
            return Err(br(BlitStatus::Bounds, "view_slice_u16"));
        }
        // 3D views use depth planes, not array slices.
        if texture_view_type_is_3d(view_type) && abs_slice != 0 {
            return Err(br(BlitStatus::Unsupported, "view_3d_slice"));
        }
        // Non-array 2D/1D: only slice 0.
        if !texture_view_type_uses_slices(view_type)
            && !texture_view_type_is_3d(view_type)
            && abs_slice != 0
        {
            return Err(br(BlitStatus::Unsupported, "view_2d_slice"));
        }
        let mut backing = resolve_texture_backing_depth(
            state,
            host,
            task_id,
            view.base_texture_ref,
            abs_level as u16,
            abs_slice as u16,
            depth + 1,
        )?;
        // Geometry constraints for non-2D types.
        match &backing {
            TextureBacking::Linear(t) => {
                if matches!(
                    view_type,
                    crate::runtime::decode::resource::TEXTURE_VIEW_MTL_TYPE_1D
                        | crate::runtime::decode::resource::TEXTURE_VIEW_MTL_TYPE_1D_ARRAY
                ) && t.height != 1
                {
                    return Err(br(BlitStatus::Unsupported, "view_1d_height"));
                }
            }
            TextureBacking::Type11(_) => {
                // Metal forbids mipmapped / multi-slice IOSurface textures; see
                // Type11Texture. Fail closed rather than inventing layout.
                if abs_level != 0 || abs_slice != 0 {
                    return Err(br(BlitStatus::Unsupported, "view_t11_level_slice"));
                }
                if texture_view_type_uses_slices(view_type) || texture_view_type_is_3d(view_type) {
                    return Err(br(BlitStatus::Unsupported, "view_t11_type"));
                }
            }
        }
        // View pixel_format overrides base when bpp-compatible.
        if view.pixel_format != 0 {
            let base_fmt = backing.pixel_format();
            let eff = metal_draw::effective_view_sample_format(base_fmt, Some(view.pixel_format))
                .ok_or_else(|| br(BlitStatus::Unsupported, "view_fmt_incompat"))?;
            match &mut backing {
                TextureBacking::Linear(t) => {
                    t.pixel_format = eff;
                    t.bpp = pixel_format::bytes_per_pixel(eff)
                        .ok_or_else(|| br(BlitStatus::Unsupported, "view_fmt_bpp"))?;
                }
                TextureBacking::Type11(t) => {
                    t.pixel_format = eff;
                    t.bpp = pixel_format::bytes_per_pixel(eff)
                        .ok_or_else(|| br(BlitStatus::Unsupported, "view_fmt_bpp"))?;
                }
            }
        }
        return Ok(backing);
    }

    // Type-11 IOSurface: single level, 2D, mapping page table.
    // Non-zero level/slice is fail-closed (Metal disallows mipmapped IOSurfaces).
    // Texture object dims/format select the plane when the mapping is multi-plane.
    if entry.object_type == OBJECT_TYPE_IOSURFACE {
        if level != 0 || slice != 0 {
            return Err(br(BlitStatus::Unsupported, "t11_level_slice"));
        }
        let Some(bytes) = objects::read_descriptor(state, host, task_id, &entry) else {
            return Err(br(BlitStatus::MissingResource, "t11_desc_read"));
        };
        let Ok(ResourceDescriptor::IOSurfaceTexture {
            mapping_id,
            pixel_format: tex_fmt,
            width: tex_w,
            height: tex_h,
            ..
        }) = decode_iosurface_texture_descriptor(&bytes)
        else {
            return Err(br(BlitStatus::MissingResource, "t11_desc_decode"));
        };
        if mapping_id == 0 || tex_w == 0 || tex_h == 0 {
            return Err(br(BlitStatus::MissingResource, "t11_zero_geom"));
        }
        // Latch texture→mapping and refresh pages / device desc.
        let _ = objects::resolve_type11_ref(state, host, task_id, texture_ref);
        let _ = mapper::ensure_resolved_for_scanout(state, host, mapping_id);
        let Some(m) = state.mappings.get(&mapping_id) else {
            return Err(br(BlitStatus::MissingResource, "t11_no_mapping"));
        };
        if !m.mapped || m.page_entries.is_empty() {
            return Err(br(BlitStatus::MissingResource, "t11_unmapped"));
        }
        let format = if tex_fmt != 0 {
            tex_fmt
        } else if m.format != 0 {
            m.format
        } else {
            MTL_FORMAT_BGRA8_UNORM
        };
        let Some(bpp) = pixel_format::bytes_per_pixel(format) else {
            return Err(br(BlitStatus::Unsupported, "t11_fmt_bpp"));
        };
        let Some((surface_offset, surface_bpr, span_end)) =
            mapping_write::type11_sample_window(m, tex_w, tex_h, format)
        else {
            return Err(br(BlitStatus::Bounds, "t11_sample_window"));
        };
        return Ok(TextureBacking::Type11(Type11Texture {
            mapping_id,
            width: tex_w,
            height: tex_h,
            surface_offset,
            row_stride: surface_bpr as u64,
            span_end,
            bpp,
            pixel_format: format,
        }));
    }

    // Type-5 RefTexture: a serialized Metal texture VIEW over an IOSurface
    // (surfaceID at +0). The compute stage path already resolves these; the
    // blit path previously dropped every one as `tex_wrong_type` (~99/six-app
    // launch, all object_type=5), so a blit COPY from a video/biplanar plane
    // or a row-byte-equivalent reinterpretation view (e.g. RGBA32Uint over
    // BGRA8) never landed. Resolve it exactly like type-11 but using the
    // decoded VIEW geometry/format: `type11_sample_window` matches the actual
    // plane record by geometry+bpe (or a packed row-compatible reinterpretation)
    // and fail-closes when it cannot, so this never invents a plane window.
    if entry.object_type == objects::OBJECT_TYPE_REF_TEXTURE {
        if level != 0 || slice != 0 {
            return Err(br(BlitStatus::Unsupported, "t5_level_slice"));
        }
        let Some(bytes) = objects::read_descriptor(state, host, task_id, &entry) else {
            return Err(br(BlitStatus::MissingResource, "t5_desc_read"));
        };
        if bytes.len() < objects::TYPE5_MIN_LEN {
            return Err(br(BlitStatus::MissingResource, "t5_desc_short"));
        }
        let sid = crate::contract::endian::ld32(&bytes[objects::TYPE5_SURFACE_ID..]);
        if sid == 0 {
            return Err(br(BlitStatus::MissingResource, "t5_no_sid"));
        }
        let Some(view) = objects::decode_type5_texture_view(&bytes) else {
            // A short/zero-geom record fails closed — no fallback to base geom.
            // Capture why (len/tag/geom) deduped per sid so the exact blit-path
            // type-5 layout can be decoded without flooding.
            note_t5_decode_fail(sid, &bytes);
            return Err(br(BlitStatus::Unsupported, "t5_view_decode"));
        };
        // Surface id IS the type-4 mapping mid (never the task object-list ref —
        // those id spaces collide). Resolve the backing, then the mapping.
        let _ = objects::ensure_surface_for_present(state, host, sid);
        let _ = mapper::ensure_resolved_for_scanout(state, host, sid);
        let Some(m) = state.mappings.get(&sid) else {
            return Err(br(BlitStatus::MissingResource, "t5_no_mapping"));
        };
        if !m.mapped || m.page_entries.is_empty() {
            return Err(br(BlitStatus::MissingResource, "t5_unmapped"));
        }
        let format = view.pixel_format;
        let Some(bpp) = pixel_format::bytes_per_pixel(format) else {
            return Err(br(BlitStatus::Unsupported, "t5_fmt_bpp"));
        };
        let Some((surface_offset, surface_bpr, span_end)) =
            mapping_write::type11_sample_window(m, view.width, view.height, format)
        else {
            return Err(br(BlitStatus::Bounds, "t5_sample_window"));
        };
        return Ok(TextureBacking::Type11(Type11Texture {
            mapping_id: sid,
            width: view.width,
            height: view.height,
            surface_offset,
            row_stride: surface_bpr as u64,
            span_end,
            bpp,
            pixel_format: format,
        }));
    }

    if entry.object_type != OBJECT_TYPE_TEXTURE && entry.object_type != OBJECT_TYPE_TEXTURE_VARIANT
    {
        let _ = note_tex_wrong_type(task_id, texture_ref, entry.object_type, level, slice);
        return Err(br(BlitStatus::MissingResource, "tex_wrong_type"));
    }
    let Some(bytes) = objects::read_descriptor(state, host, task_id, &entry) else {
        return Err(br(BlitStatus::MissingResource, "tex_desc_read"));
    };
    let Ok(tex) = decode_texture_descriptor(&bytes) else {
        return Err(br(BlitStatus::MissingResource, "tex_desc_decode"));
    };
    if !tex.has_pixel_format {
        crate::observe::fail(format!(
            "blit tex no_pixel_format ref={texture_ref} w={} h={} fmt={}",
            tex.width, tex.height, tex.pixel_format
        ));
        return Err(br(BlitStatus::Unsupported, "tex_no_pixel_format"));
    }
    let Some(bpp) = pixel_format::bytes_per_pixel(tex.pixel_format) else {
        crate::observe::fail(format!(
            "blit tex bad_bpp ref={texture_ref} fmt={}",
            tex.pixel_format
        ));
        return Err(br(BlitStatus::Unsupported, "tex_bad_bpp"));
    };
    let Some((layout_gva, layout)) = tex.level_gva(level as u32, state.page_shift) else {
        crate::observe::fail(format!(
            "blit tex level_gva_shift fail ref={texture_ref} lvl={level} handle={} alloc={} mips={} page_shift={} w={} h={} fmt={:#x}",
            tex.handle,
            tex.allocation_size,
            tex.mipmap_level_count,
            state.page_shift,
            tex.width,
            tex.height,
            tex.pixel_format
        ));
        return Err(br(BlitStatus::Bounds, "tex_level_gva"));
    };
    let Some(base_gva) = tex.allocation_base_gva(state.page_shift) else {
        return Err(br(BlitStatus::MissingResource, "tex_no_base_gva"));
    };
    // level_gva already applied offset; keep offset relative to base for plane math.
    let level_offset = match layout_gva.checked_sub(base_gva) {
        Some(v) => v,
        None => {
            crate::observe::fail(format!(
                "blit tex level_offset underflow layout_gva={layout_gva:#x} base={base_gva:#x} page_shift={}",
                state.page_shift
            ));
            return Err(br(BlitStatus::Bounds, "tex_level_offset_underflow"));
        }
    };
    if layout.width == 0 || layout.height == 0 {
        crate::observe::fail(format!(
            "blit tex zero_geom ref={texture_ref} lvl={level} layout={}x{}x{}",
            layout.width, layout.height, layout.depth
        ));
        return Err(br(BlitStatus::Bounds, "tex_zero_geom"));
    }
    let depth = if layout.depth == 0 { 1 } else { layout.depth };
    // Array-slice packing: contiguous images at this mip (row_stride × height × depth).
    // Prefer level.size when it is an exact multiple of one-slice bytes (multi-slice alloc).
    let one_slice = derived_slice_stride(layout.row_stride, layout.height, depth)
        .ok_or_else(|| br(BlitStatus::Capacity, "tex_slice_stride"))?;
    let slice_stride =
        if layout.size != 0 && layout.size % one_slice == 0 && layout.size >= one_slice {
            // When size == one_slice, only one slice lives at this level offset.
            // When size is a multiple, slices are packed with stride one_slice.
            one_slice
        } else {
            one_slice
        };
    if slice != 0 {
        // Bounds: selected slice must fit in allocation when known.
        // Live x86 buffer→texture (opcode 0x12c) uses slice=1,2 with
        // size=16384x1x1 at off=64K/128K — array packing into one allocation
        // even when the L0 level record's `size` equals one_slice. Prefer
        // allocation_size over the level-size single-slice reject.
        let slice_end = (slice as u64)
            .checked_mul(slice_stride)
            .and_then(|o| o.checked_add(level_offset))
            .and_then(|o| o.checked_add(one_slice))
            .ok_or_else(|| br(BlitStatus::Bounds, "tex_slice_overflow"))?;
        if tex.allocation_size != 0 && slice_end > tex.allocation_size {
            crate::observe::fail(format!(
                "blit tex slice Bounds slice={slice} end={slice_end} alloc={} one_slice={one_slice} lvl_off={level_offset}",
                tex.allocation_size
            ));
            return Err(br(BlitStatus::Bounds, "tex_slice_bounds"));
        }
        if tex.allocation_size == 0 && layout.size != 0 && layout.size == one_slice && slice != 0 {
            // Unknown alloc and level size covers a single slice only.
            return Err(br(BlitStatus::Bounds, "tex_slice_single"));
        }
    }
    Ok(TextureBacking::Linear(LinearTextureLevel {
        base_gva,
        alloc_size: tex.allocation_size,
        level_offset,
        row_stride: layout.row_stride,
        slice_stride,
        slice_index: slice as u32,
        width: layout.width,
        height: layout.height,
        depth,
        bpp,
        pixel_format: tex.pixel_format,
    }))
}

/// Read one texture row (tight `row_bytes`) at texel (ox, oy+row_i) plane z into `buf`.
#[allow(
    clippy::too_many_arguments,
    reason = "the row helper keeps texture origin and plane geometry explicit"
)]
fn read_texture_row<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    tex: &TextureBacking,
    ox: u64,
    oy: u64,
    oz: u64,
    row_i: u64,
    row_bytes: u64,
    buf: &mut [u8],
) -> Result<(), BlitStatus> {
    if row_bytes as usize > buf.len() {
        return Err(br(BlitStatus::Capacity, "rd_row_buf_cap"));
    }
    match tex {
        TextureBacking::Linear(t) => {
            let off = t
                .texel_offset(
                    ox,
                    oy.checked_add(row_i)
                        .ok_or_else(|| br(BlitStatus::Bounds, "rd_row_y_overflow"))?,
                    oz,
                )
                .ok_or_else(|| br(BlitStatus::Bounds, "rd_row_texel_oob"))?;
            let gva = t
                .base_gva
                .checked_add(off)
                .ok_or_else(|| br(BlitStatus::Bounds, "rd_row_gva_overflow"))?;
            if gva_mem::read_task_gva_fallback(
                host,
                &state.tasks,
                task_id,
                gva,
                &mut buf[..row_bytes as usize],
                state.page_shift,
            )
            .is_err()
            {
                return Err(br(BlitStatus::GuestIo, "rd_row_linear_io"));
            }
            Ok(())
        }
        TextureBacking::Type11(t) => {
            if oz != 0 {
                return Err(br(BlitStatus::Unsupported, "rd_row_t11_z"));
            }
            let y = oy
                .checked_add(row_i)
                .ok_or_else(|| br(BlitStatus::Bounds, "rd_row_t11_y_overflow"))?;
            if y > u32::MAX as u64 || ox > u32::MAX as u64 {
                return Err(br(BlitStatus::Bounds, "rd_row_t11_coord_range"));
            }
            let pixels = (row_bytes / t.bpp as u64) as u32;
            if !mapping_write::read_rect_raw_at(
                state,
                host,
                t.mapping_id,
                t.surface_offset,
                t.row_stride as u32,
                t.span_end,
                ox as u32,
                y as u32,
                pixels,
                1,
                t.bpp,
                &mut buf[..row_bytes as usize],
                row_bytes as u32,
            ) {
                return Err(br(BlitStatus::GuestIo, "rd_row_t11_io"));
            }
            Ok(())
        }
    }
}

/// Write one texture row from `buf`.
#[allow(
    clippy::too_many_arguments,
    reason = "the row helper keeps texture origin and plane geometry explicit"
)]
fn write_texture_row<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    tex: &TextureBacking,
    ox: u64,
    oy: u64,
    oz: u64,
    row_i: u64,
    row_bytes: u64,
    buf: &[u8],
) -> Result<(), BlitStatus> {
    if row_bytes as usize > buf.len() {
        return Err(br(BlitStatus::Capacity, "wr_row_buf_cap"));
    }
    match tex {
        TextureBacking::Linear(t) => {
            let off = t
                .texel_offset(
                    ox,
                    oy.checked_add(row_i)
                        .ok_or_else(|| br(BlitStatus::Bounds, "wr_row_y_overflow"))?,
                    oz,
                )
                .ok_or_else(|| br(BlitStatus::Bounds, "wr_row_texel_oob"))?;
            let gva = t
                .base_gva
                .checked_add(off)
                .ok_or_else(|| br(BlitStatus::Bounds, "wr_row_gva_overflow"))?;
            if gva_mem::write_task_gva_product(
                state,
                host,
                task_id,
                gva,
                &buf[..row_bytes as usize],
            )
            .is_err()
            {
                return Err(br(BlitStatus::GuestIo, "wr_row_linear_io"));
            }
            Ok(())
        }
        TextureBacking::Type11(t) => {
            if oz != 0 {
                return Err(br(BlitStatus::Unsupported, "wr_row_t11_z"));
            }
            let y = oy
                .checked_add(row_i)
                .ok_or_else(|| br(BlitStatus::Bounds, "wr_row_t11_y_overflow"))?;
            if y > u32::MAX as u64 || ox > u32::MAX as u64 {
                return Err(br(BlitStatus::Bounds, "wr_row_t11_coord_range"));
            }
            let pixels = (row_bytes / t.bpp as u64) as u32;
            if !mapping_write::write_rect_raw_at(
                state,
                host,
                t.mapping_id,
                t.surface_offset,
                t.row_stride as u32,
                t.span_end,
                ox as u32,
                y as u32,
                pixels,
                1,
                t.bpp,
                &buf[..row_bytes as usize],
                row_bytes as u32,
            ) {
                return Err(br(BlitStatus::GuestIo, "wr_row_t11_io"));
            }
            Ok(())
        }
    }
}

fn range_fits(offset: u64, length: u64, size: u64) -> bool {
    offset <= size && length <= size - offset
}

fn ranges_overlap(a0: u64, a_len: u64, b0: u64, b_len: u64) -> bool {
    if a_len == 0 || b_len == 0 {
        return false;
    }
    let a1 = a0.saturating_add(a_len);
    let b1 = b0.saturating_add(b_len);
    a0 < b1 && b0 < a1
}

fn write_fill_range<M: HostMemory + HostOps>(
    host: &mut M,
    state: &mut DeviceState,
    task_id: u32,
    gva: u64,
    length: u64,
    value: u8,
) -> Result<(), BlitStatus> {
    if length == 0 {
        return Ok(());
    }
    let mut remaining = length;
    let mut cur = gva;
    let chunk = vec![value; CHUNK];
    while remaining > 0 {
        let n = remaining.min(CHUNK as u64) as usize;
        if gva_mem::write_task_gva_product(state, host, task_id, cur, &chunk[..n]).is_err() {
            return Err(br(BlitStatus::GuestIo, "fill_write_io"));
        }
        cur = cur
            .checked_add(n as u64)
            .ok_or_else(|| br(BlitStatus::Capacity, "fill_gva_advance_overflow"))?;
        remaining -= n as u64;
    }
    Ok(())
}

fn copy_bytes<M: HostMemory + HostOps>(
    host: &mut M,
    state: &mut DeviceState,
    task_id: u32,
    src_gva: u64,
    dst_gva: u64,
    length: u64,
) -> Result<(), BlitStatus> {
    if length == 0 {
        return Ok(());
    }
    let mut remaining = length;
    let mut s = src_gva;
    let mut d = dst_gva;
    let mut buf = vec![0u8; CHUNK.min(length as usize).max(1)];
    while remaining > 0 {
        let n = remaining.min(buf.len() as u64) as usize;
        if gva_mem::read_task_gva_fallback(
            host,
            &state.tasks,
            task_id,
            s,
            &mut buf[..n],
            state.page_shift,
        )
        .is_err()
        {
            return Err(br(BlitStatus::GuestIo, "copy_bytes_read_io"));
        }
        if gva_mem::write_task_gva_product(state, host, task_id, d, &buf[..n]).is_err() {
            return Err(br(BlitStatus::GuestIo, "copy_bytes_write_io"));
        }
        s = s
            .checked_add(n as u64)
            .ok_or_else(|| br(BlitStatus::Capacity, "copy_bytes_src_overflow"))?;
        d = d
            .checked_add(n as u64)
            .ok_or_else(|| br(BlitStatus::Capacity, "copy_bytes_dst_overflow"))?;
        remaining -= n as u64;
    }
    Ok(())
}

/// Copy a rectangular multi-plane region with independent source/dest strides.
#[allow(
    clippy::too_many_arguments,
    reason = "the copy helper mirrors independent source and destination row geometry"
)]
fn copy_row_region<M: HostMemory + HostOps>(
    host: &mut M,
    state: &mut DeviceState,
    task_id: u32,
    src_base: u64,
    src_row_stride: u64,
    src_image_stride: u64,
    dst_base: u64,
    dst_row_stride: u64,
    dst_image_stride: u64,
    row_bytes: u64,
    row_count: u64,
    image_count: u64,
) -> Result<(), BlitStatus> {
    if row_bytes == 0 || row_count == 0 || image_count == 0 {
        return Ok(());
    }
    // Stride/row contract only — no host MiB byte budget (chunked row I/O).
    if row_bytes > src_row_stride || row_bytes > dst_row_stride {
        return Err(br(BlitStatus::Bounds, "copy_region_row_gt_stride"));
    }
    let _total = row_bytes
        .checked_mul(row_count)
        .and_then(|v| v.checked_mul(image_count))
        .ok_or_else(|| br(BlitStatus::Capacity, "copy_region_total_overflow"))?;
    let row_len = host_alloc_len(row_bytes)
        .ok_or_else(|| br(BlitStatus::Capacity, "copy_region_row_alloc"))?;
    let mut row_buf = vec![0u8; row_len];
    for z in 0..image_count {
        let src_plane = src_base
            .checked_add(
                z.checked_mul(src_image_stride)
                    .ok_or_else(|| br(BlitStatus::Capacity, "copy_region_src_plane_overflow"))?,
            )
            .ok_or_else(|| br(BlitStatus::Capacity, "copy_region_src_plane_overflow"))?;
        let dst_plane = dst_base
            .checked_add(
                z.checked_mul(dst_image_stride)
                    .ok_or_else(|| br(BlitStatus::Capacity, "copy_region_dst_plane_overflow"))?,
            )
            .ok_or_else(|| br(BlitStatus::Capacity, "copy_region_dst_plane_overflow"))?;
        for y in 0..row_count {
            let s = src_plane
                .checked_add(
                    y.checked_mul(src_row_stride)
                        .ok_or_else(|| br(BlitStatus::Capacity, "copy_region_src_row_overflow"))?,
                )
                .ok_or_else(|| br(BlitStatus::Capacity, "copy_region_src_row_overflow"))?;
            let d = dst_plane
                .checked_add(
                    y.checked_mul(dst_row_stride)
                        .ok_or_else(|| br(BlitStatus::Capacity, "copy_region_dst_row_overflow"))?,
                )
                .ok_or_else(|| br(BlitStatus::Capacity, "copy_region_dst_row_overflow"))?;
            if gva_mem::read_task_gva_fallback(
                host,
                &state.tasks,
                task_id,
                s,
                &mut row_buf,
                state.page_shift,
            )
            .is_err()
            {
                note_copy_region_io(task_id, false, s, y, z, row_bytes, state.page_shift);
                return Err(br(BlitStatus::GuestIo, "copy_region_read_io"));
            }
            if gva_mem::write_task_gva_product(state, host, task_id, d, &row_buf).is_err() {
                note_copy_region_io(task_id, true, d, y, z, row_bytes, state.page_shift);
                return Err(br(BlitStatus::GuestIo, "copy_region_write_io"));
            }
        }
    }
    Ok(())
}

fn exec_fill_buffer<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    cmd: &Command,
) -> BlitStatus {
    if cmd.range_length == 0 {
        return BlitStatus::ZeroExtent;
    }
    let buf = match resolve_buffer(state, host, task_id, cmd.buffer) {
        Ok(b) => b,
        Err(st) => return st,
    };
    if !range_fits(cmd.range_location, cmd.range_length, buf.size) {
        return br(BlitStatus::Bounds, "fill_range_oob");
    }
    let gva = match buf.gva.checked_add(cmd.range_location) {
        Some(v) => v,
        None => return br(BlitStatus::Bounds, "fill_gva_overflow"),
    };
    match write_fill_range(host, state, task_id, gva, cmd.range_length, cmd.fill_value) {
        Ok(()) => BlitStatus::Ok,
        Err(st) => st,
    }
}

fn exec_copy_buffer_to_buffer<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    cmd: &Command,
) -> BlitStatus {
    if cmd.size == 0 {
        return BlitStatus::ZeroExtent;
    }
    let src = match resolve_buffer(state, host, task_id, cmd.source) {
        Ok(b) => b,
        Err(st) => return st,
    };
    let dst = match resolve_buffer(state, host, task_id, cmd.destination) {
        Ok(b) => b,
        Err(st) => return st,
    };
    if !range_fits(cmd.source_offset, cmd.size, src.size)
        || !range_fits(cmd.destination_offset, cmd.size, dst.size)
    {
        return br(BlitStatus::Bounds, "b2b_range_oob");
    }
    // Same allocation (same GVA base + size): reject overlapping windows.
    if src.gva == dst.gva
        && src.size == dst.size
        && ranges_overlap(
            cmd.source_offset,
            cmd.size,
            cmd.destination_offset,
            cmd.size,
        )
    {
        return br(BlitStatus::Overlap, "b2b_overlap");
    }
    let s = match src.gva.checked_add(cmd.source_offset) {
        Some(v) => v,
        None => return br(BlitStatus::Bounds, "b2b_src_gva_overflow"),
    };
    let d = match dst.gva.checked_add(cmd.destination_offset) {
        Some(v) => v,
        None => return br(BlitStatus::Bounds, "b2b_dst_gva_overflow"),
    };
    match copy_bytes(host, state, task_id, s, d, cmd.size) {
        Ok(()) => BlitStatus::Ok,
        Err(st) => st,
    }
}

fn clamp_extent(requested: u64, max: u64) -> u64 {
    if requested == 0 {
        // Metal size 0 is a no-op extent; keep 0.
        0
    } else if requested > max {
        max
    } else {
        requested
    }
}

/// Resolve `MTLBlitOption` → aspect flags + buffer-side plane bpp.
fn copy_aspect_for_options(
    texture_format: u16,
    cmd: &Command,
) -> Result<(bool, bool, u32), BlitStatus> {
    // The three option checks used to collapse into a bare `Unsupported` with
    // the reason discarded by `map_err(|_| ..)`. The blit reason channel carries
    // the specific slug to the dispatch-site line, so an unknown option bit and
    // a depth+stencil conflict no longer read identically.
    let aspect = blit::parse_blit_options(cmd.has_options, cmd.options)
        .map_err(|e: blit::BlitOptionError| br(BlitStatus::Unsupported, e.slug()))?;
    let (want_depth, want_stencil) = match aspect {
        BlitAspect::Full => (false, false),
        BlitAspect::Depth => (true, false),
        BlitAspect::Stencil => (false, true),
    };
    let bpp = pixel_format::blit_aspect_bytes_per_pixel(texture_format, want_depth, want_stencil)
        .ok_or(BlitStatus::Unsupported)?;
    Ok((want_depth, want_stencil, bpp))
}

/// Texture-side full texel bpp (storage). Plane copies use this for GVA strides.
fn texture_storage_bpp(format: u16) -> Result<u32, BlitStatus> {
    pixel_format::bytes_per_pixel(format).ok_or(BlitStatus::Unsupported)
}

/// Read one packed texture row (tight `width * storage_bpp`) at (ox, oy+row_i, oz).
#[allow(
    clippy::too_many_arguments,
    reason = "the row helper keeps packed texture coordinates and format explicit"
)]
fn read_texture_storage_row<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    tex: &TextureBacking,
    ox: u64,
    oy: u64,
    oz: u64,
    row_i: u64,
    width: u32,
    storage_bpp: u32,
    buf: &mut [u8],
) -> Result<(), BlitStatus> {
    let row_bytes = (width as u64)
        .checked_mul(storage_bpp as u64)
        .ok_or(BlitStatus::Capacity)?;
    if row_bytes as usize > buf.len() {
        return Err(BlitStatus::Capacity);
    }
    // Reuse read_texture_row but with storage row size (not plane size).
    // Temporarily: call the same GVA path with storage row_bytes.
    read_texture_row(state, host, task_id, tex, ox, oy, oz, row_i, row_bytes, buf)
}

/// Write one packed texture row.
#[allow(
    clippy::too_many_arguments,
    reason = "the row helper keeps packed texture coordinates and format explicit"
)]
fn write_texture_storage_row<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    tex: &TextureBacking,
    ox: u64,
    oy: u64,
    oz: u64,
    row_i: u64,
    width: u32,
    storage_bpp: u32,
    buf: &[u8],
) -> Result<(), BlitStatus> {
    let row_bytes = (width as u64)
        .checked_mul(storage_bpp as u64)
        .ok_or(BlitStatus::Capacity)?;
    write_texture_row(state, host, task_id, tex, ox, oy, oz, row_i, row_bytes, buf)
}

/// Copy buffer plane rows ↔ texture with optional combined-DS plane repack.
#[allow(
    clippy::too_many_arguments,
    reason = "the blit executor mirrors the decoded buffer, texture, and aspect fields"
)]
fn copy_buffer_texture_rows_aspect<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    buf_base_gva: u64,
    buf_row_stride: u64,
    buf_image_stride: u64,
    tex: &TextureBacking,
    tex_ox: u64,
    tex_oy: u64,
    tex_oz: u64,
    copy_w: u32,
    copy_h: u64,
    copy_d: u64,
    plane_bpp: u32,
    want_depth: bool,
    want_stencil: bool,
    to_texture: bool,
) -> Result<(), BlitStatus> {
    let fmt = tex.pixel_format();
    let repack = pixel_format::blit_aspect_needs_repack(fmt, want_depth, want_stencil);
    let storage_bpp = if repack {
        texture_storage_bpp(fmt)?
    } else {
        plane_bpp
    };
    let plane_row = (copy_w as u64)
        .checked_mul(plane_bpp as u64)
        .ok_or(BlitStatus::Capacity)? as usize;
    let storage_row = (copy_w as u64)
        .checked_mul(storage_bpp as u64)
        .ok_or(BlitStatus::Capacity)? as usize;
    let mut plane = vec![0u8; plane_row];
    let mut packed = vec![0u8; storage_row.max(plane_row)];
    for z in 0..copy_d {
        for y in 0..copy_h {
            let buf_gva = buf_base_gva
                .checked_add(
                    z.checked_mul(buf_image_stride)
                        .ok_or(BlitStatus::Capacity)?,
                )
                .ok_or(BlitStatus::Capacity)?
                .checked_add(y.checked_mul(buf_row_stride).ok_or(BlitStatus::Capacity)?)
                .ok_or(BlitStatus::Capacity)?;
            if to_texture {
                if gva_mem::read_task_gva_fallback(
                    host,
                    &state.tasks,
                    task_id,
                    buf_gva,
                    &mut plane,
                    state.page_shift,
                )
                .is_err()
                {
                    return Err(BlitStatus::GuestIo);
                }
                if repack {
                    // RMW: load existing packed row, insert plane, store.
                    read_texture_storage_row(
                        state,
                        host,
                        task_id,
                        tex,
                        tex_ox,
                        tex_oy,
                        tex_oz + z,
                        y,
                        copy_w,
                        storage_bpp,
                        &mut packed,
                    )?;
                    if !pixel_format::insert_plane_row(
                        fmt,
                        want_depth,
                        want_stencil,
                        &plane,
                        copy_w,
                        &mut packed[..storage_row],
                    ) {
                        return Err(BlitStatus::Unsupported);
                    }
                    write_texture_storage_row(
                        state,
                        host,
                        task_id,
                        tex,
                        tex_ox,
                        tex_oy,
                        tex_oz + z,
                        y,
                        copy_w,
                        storage_bpp,
                        &packed[..storage_row],
                    )?;
                } else {
                    write_texture_row(
                        state,
                        host,
                        task_id,
                        tex,
                        tex_ox,
                        tex_oy,
                        tex_oz + z,
                        y,
                        plane_row as u64,
                        &plane,
                    )?;
                }
            } else if repack {
                read_texture_storage_row(
                    state,
                    host,
                    task_id,
                    tex,
                    tex_ox,
                    tex_oy,
                    tex_oz + z,
                    y,
                    copy_w,
                    storage_bpp,
                    &mut packed,
                )?;
                if !pixel_format::extract_plane_row(
                    fmt,
                    want_depth,
                    want_stencil,
                    &packed[..storage_row],
                    copy_w,
                    &mut plane,
                ) {
                    return Err(BlitStatus::Unsupported);
                }
                if gva_mem::write_task_gva_product(state, host, task_id, buf_gva, &plane).is_err() {
                    return Err(BlitStatus::GuestIo);
                }
            } else {
                read_texture_row(
                    state,
                    host,
                    task_id,
                    tex,
                    tex_ox,
                    tex_oy,
                    tex_oz + z,
                    y,
                    plane_row as u64,
                    &mut plane,
                )?;
                if gva_mem::write_task_gva_product(state, host, task_id, buf_gva, &plane).is_err() {
                    return Err(BlitStatus::GuestIo);
                }
            }
        }
    }
    Ok(())
}

fn exec_copy_buffer_to_texture<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    cmd: &Command,
) -> BlitStatus {
    let src = match resolve_buffer(state, host, task_id, cmd.source) {
        Ok(b) => b,
        Err(st) => return st,
    };
    let dst = match resolve_texture_backing(
        state,
        host,
        task_id,
        cmd.destination,
        cmd.destination_level,
        cmd.destination_slice,
    ) {
        Ok(t) => t,
        Err(st) => return st,
    };
    let (want_depth, want_stencil, copy_bpp) =
        match copy_aspect_for_options(dst.pixel_format(), cmd) {
            Ok(v) => v,
            Err(st) => return st,
        };
    let repack =
        pixel_format::blit_aspect_needs_repack(dst.pixel_format(), want_depth, want_stencil);
    // Type-11 is 2D only.
    if dst.is_type11() && (cmd.destination_origin.z != 0 || cmd.source_size.depth > 1) {
        if cmd.source_size.depth == 0 {
            return BlitStatus::ZeroExtent;
        }
        if cmd.destination_origin.z != 0 || cmd.source_size.depth != 1 {
            return br(BlitStatus::Unsupported, "b2t_t11_z_or_depth");
        }
    }
    let ox = cmd.destination_origin.x;
    let oy = cmd.destination_origin.y;
    let oz = cmd.destination_origin.z;
    if ox > dst.width() as u64 || oy > dst.height() as u64 || oz > dst.depth() as u64 {
        return br(BlitStatus::Bounds, "b2t_origin_oob");
    }
    let copy_w = clamp_extent(cmd.source_size.width, dst.width() as u64 - ox);
    let copy_h = clamp_extent(cmd.source_size.height, dst.height() as u64 - oy);
    let copy_d = if cmd.source_size.depth == 0 {
        0
    } else {
        clamp_extent(cmd.source_size.depth, dst.depth() as u64 - oz)
    };
    if copy_w == 0 || copy_h == 0 || copy_d == 0 {
        return BlitStatus::ZeroExtent;
    }
    // Buffer-side plane bpp (aspect-aware).
    let row_bytes = match copy_w.checked_mul(copy_bpp as u64) {
        Some(v) => v,
        None => return br(BlitStatus::Capacity, "b2t_row_bytes_overflow"),
    };
    let src_bpr = if cmd.source_bytes_per_row != 0 {
        cmd.source_bytes_per_row
    } else {
        row_bytes
    };
    if src_bpr < row_bytes {
        return br(BlitStatus::Bounds, "b2t_src_bpr_lt_row");
    }
    let src_bpi = if cmd.source_bytes_per_image != 0 {
        cmd.source_bytes_per_image
    } else {
        match src_bpr.checked_mul(copy_h) {
            Some(v) => v,
            None => return br(BlitStatus::Capacity, "b2t_src_bpi_overflow"),
        }
    };
    // Combined DS + aspect: plane repack path (not raw GVA span).
    if repack {
        let src_gva = match src.gva.checked_add(cmd.source_offset) {
            Some(v) => v,
            None => return br(BlitStatus::Bounds, "b2t_repack_gva_overflow"),
        };
        return match copy_buffer_texture_rows_aspect(
            state,
            host,
            task_id,
            src_gva,
            src_bpr,
            src_bpi,
            &dst,
            ox,
            oy,
            oz,
            copy_w as u32,
            copy_h,
            copy_d,
            copy_bpp,
            want_depth,
            want_stencil,
            true,
        ) {
            Ok(()) => BlitStatus::Ok,
            Err(st) => st,
        };
    }
    // Prefer direct GVA row-span when both sides linear (dst only texture here).
    if let TextureBacking::Linear(ref lt) = dst {
        let dst_off = match lt.texel_offset(ox, oy, oz) {
            Some(v) => v,
            None => return br(BlitStatus::Bounds, "b2t_dst_texel_oob"),
        };
        let dst_bpi = match lt.bytes_per_image() {
            Some(v) => v,
            None => return br(BlitStatus::Capacity, "b2t_dst_bpi_overflow"),
        };
        let last = match dst_off
            .checked_add((copy_d - 1).saturating_mul(dst_bpi))
            .and_then(|v| v.checked_add((copy_h - 1).saturating_mul(lt.row_stride)))
            .and_then(|v| v.checked_add(row_bytes))
        {
            Some(v) => v,
            None => return br(BlitStatus::Bounds, "b2t_dst_span_overflow"),
        };
        if lt.alloc_size != 0 && last > lt.alloc_size {
            return br(BlitStatus::Bounds, "b2t_dst_alloc_oob");
        }
        let src_span = match cmd
            .source_offset
            .checked_add((copy_d - 1).saturating_mul(src_bpi))
            .and_then(|v| v.checked_add((copy_h - 1).saturating_mul(src_bpr)))
            .and_then(|v| v.checked_add(row_bytes))
        {
            Some(v) => v,
            None => return br(BlitStatus::Bounds, "b2t_src_span_overflow"),
        };
        if src_span > src.size {
            return br(BlitStatus::Bounds, "b2t_src_span_oob");
        }
        let src_gva = match src.gva.checked_add(cmd.source_offset) {
            Some(v) => v,
            None => return br(BlitStatus::Bounds, "b2t_src_gva_overflow"),
        };
        let dst_gva = match lt.base_gva.checked_add(dst_off) {
            Some(v) => v,
            None => return br(BlitStatus::Bounds, "b2t_dst_gva_overflow"),
        };
        return match copy_row_region(
            host,
            state,
            task_id,
            src_gva,
            src_bpr,
            src_bpi,
            dst_gva,
            lt.row_stride,
            dst_bpi,
            row_bytes,
            copy_h,
            copy_d,
        ) {
            Ok(()) => BlitStatus::Ok,
            Err(st) => st,
        };
    }
    // Type-11 destination: row-stage from buffer GVA.
    let src_span = match cmd
        .source_offset
        .checked_add((copy_d - 1).saturating_mul(src_bpi))
        .and_then(|v| v.checked_add((copy_h - 1).saturating_mul(src_bpr)))
        .and_then(|v| v.checked_add(row_bytes))
    {
        Some(v) => v,
        None => return br(BlitStatus::Bounds, "b2t_t11_src_span_overflow"),
    };
    if src_span > src.size {
        return br(BlitStatus::Bounds, "b2t_t11_src_span_oob");
    }
    let mut row = vec![0u8; row_bytes as usize];
    for z in 0..copy_d {
        for y in 0..copy_h {
            let s = match src
                .gva
                .checked_add(cmd.source_offset)
                .and_then(|b| b.checked_add(z.saturating_mul(src_bpi)))
                .and_then(|b| b.checked_add(y.saturating_mul(src_bpr)))
            {
                Some(v) => v,
                None => return br(BlitStatus::Bounds, "b2t_t11_src_gva_overflow"),
            };
            if gva_mem::read_task_gva_fallback(
                host,
                &state.tasks,
                task_id,
                s,
                &mut row,
                state.page_shift,
            )
            .is_err()
            {
                return br(BlitStatus::GuestIo, "b2t_t11_read_io");
            }
            if let Err(st) = write_texture_row(
                state,
                host,
                task_id,
                &dst,
                ox,
                oy,
                oz + z,
                y,
                row_bytes,
                &row,
            ) {
                return st;
            }
        }
    }
    BlitStatus::Ok
}

fn exec_copy_texture_to_buffer<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    cmd: &Command,
) -> BlitStatus {
    let src = match resolve_texture_backing(
        state,
        host,
        task_id,
        cmd.source,
        cmd.source_level,
        cmd.source_slice,
    ) {
        Ok(t) => t,
        Err(st) => return st,
    };
    let (want_depth, want_stencil, copy_bpp) =
        match copy_aspect_for_options(src.pixel_format(), cmd) {
            Ok(v) => v,
            Err(st) => return st,
        };
    let repack =
        pixel_format::blit_aspect_needs_repack(src.pixel_format(), want_depth, want_stencil);
    let dst = match resolve_buffer(state, host, task_id, cmd.destination) {
        Ok(b) => b,
        Err(st) => return st,
    };
    if src.is_type11() && (cmd.source_origin.z != 0 || cmd.source_size.depth > 1) {
        if cmd.source_size.depth == 0 {
            return BlitStatus::ZeroExtent;
        }
        if cmd.source_origin.z != 0 || cmd.source_size.depth != 1 {
            return br(BlitStatus::Unsupported, "t2b_t11_z_or_depth");
        }
    }
    let ox = cmd.source_origin.x;
    let oy = cmd.source_origin.y;
    let oz = cmd.source_origin.z;
    if ox > src.width() as u64 || oy > src.height() as u64 || oz > src.depth() as u64 {
        return br(BlitStatus::Bounds, "t2b_origin_oob");
    }
    let copy_w = clamp_extent(cmd.source_size.width, src.width() as u64 - ox);
    let copy_h = clamp_extent(cmd.source_size.height, src.height() as u64 - oy);
    let copy_d = if cmd.source_size.depth == 0 {
        0
    } else {
        clamp_extent(cmd.source_size.depth, src.depth() as u64 - oz)
    };
    if copy_w == 0 || copy_h == 0 || copy_d == 0 {
        return BlitStatus::ZeroExtent;
    }
    let row_bytes = match copy_w.checked_mul(copy_bpp as u64) {
        Some(v) => v,
        None => return br(BlitStatus::Capacity, "t2b_row_bytes_overflow"),
    };
    let dst_bpr = if cmd.destination_bytes_per_row != 0 {
        cmd.destination_bytes_per_row
    } else {
        row_bytes
    };
    if dst_bpr < row_bytes {
        return br(BlitStatus::Bounds, "t2b_dst_bpr_lt_row");
    }
    let dst_bpi = if cmd.destination_bytes_per_image != 0 {
        cmd.destination_bytes_per_image
    } else {
        match dst_bpr.checked_mul(copy_h) {
            Some(v) => v,
            None => return br(BlitStatus::Capacity, "t2b_dst_bpi_overflow"),
        }
    };
    if repack {
        let dst_gva = match dst.gva.checked_add(cmd.destination_offset) {
            Some(v) => v,
            None => return br(BlitStatus::Bounds, "t2b_repack_gva_overflow"),
        };
        return match copy_buffer_texture_rows_aspect(
            state,
            host,
            task_id,
            dst_gva,
            dst_bpr,
            dst_bpi,
            &src,
            ox,
            oy,
            oz,
            copy_w as u32,
            copy_h,
            copy_d,
            copy_bpp,
            want_depth,
            want_stencil,
            false,
        ) {
            Ok(()) => BlitStatus::Ok,
            Err(st) => st,
        };
    }
    if let TextureBacking::Linear(ref lt) = src {
        let src_off = match lt.texel_offset(ox, oy, oz) {
            Some(v) => v,
            None => return br(BlitStatus::Bounds, "t2b_src_texel_oob"),
        };
        let src_bpi = match lt.bytes_per_image() {
            Some(v) => v,
            None => return br(BlitStatus::Capacity, "t2b_src_bpi_overflow"),
        };
        let dst_span = match cmd
            .destination_offset
            .checked_add((copy_d - 1).saturating_mul(dst_bpi))
            .and_then(|v| v.checked_add((copy_h - 1).saturating_mul(dst_bpr)))
            .and_then(|v| v.checked_add(row_bytes))
        {
            Some(v) => v,
            None => return br(BlitStatus::Bounds, "t2b_dst_span_overflow"),
        };
        if dst_span > dst.size {
            return br(BlitStatus::Bounds, "t2b_dst_span_oob");
        }
        let src_gva = match lt.base_gva.checked_add(src_off) {
            Some(v) => v,
            None => return br(BlitStatus::Bounds, "t2b_src_gva_overflow"),
        };
        let dst_gva = match dst.gva.checked_add(cmd.destination_offset) {
            Some(v) => v,
            None => return br(BlitStatus::Bounds, "t2b_dst_gva_overflow"),
        };
        return match copy_row_region(
            host,
            state,
            task_id,
            src_gva,
            lt.row_stride,
            src_bpi,
            dst_gva,
            dst_bpr,
            dst_bpi,
            row_bytes,
            copy_h,
            copy_d,
        ) {
            Ok(()) => BlitStatus::Ok,
            Err(st) => st,
        };
    }
    let dst_span = match cmd
        .destination_offset
        .checked_add((copy_d - 1).saturating_mul(dst_bpi))
        .and_then(|v| v.checked_add((copy_h - 1).saturating_mul(dst_bpr)))
        .and_then(|v| v.checked_add(row_bytes))
    {
        Some(v) => v,
        None => return br(BlitStatus::Bounds, "t2b_stage_dst_span_overflow"),
    };
    if dst_span > dst.size {
        return br(BlitStatus::Bounds, "t2b_stage_dst_span_oob");
    }
    let mut row = vec![0u8; row_bytes as usize];
    for z in 0..copy_d {
        for y in 0..copy_h {
            if let Err(st) = read_texture_row(
                state,
                host,
                task_id,
                &src,
                ox,
                oy,
                oz + z,
                y,
                row_bytes,
                &mut row,
            ) {
                return st;
            }
            let d = match dst
                .gva
                .checked_add(cmd.destination_offset)
                .and_then(|b| b.checked_add(z.saturating_mul(dst_bpi)))
                .and_then(|b| b.checked_add(y.saturating_mul(dst_bpr)))
            {
                Some(v) => v,
                None => return br(BlitStatus::Bounds, "t2b_stage_dst_gva_overflow"),
            };
            if gva_mem::write_task_gva_product(state, host, task_id, d, &row).is_err() {
                return br(BlitStatus::GuestIo, "t2b_stage_write_io");
            }
        }
    }
    BlitStatus::Ok
}

fn exec_copy_texture_to_texture<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    cmd: &Command,
) -> BlitStatus {
    let src = match resolve_texture_backing(
        state,
        host,
        task_id,
        cmd.source,
        cmd.source_level,
        cmd.source_slice,
    ) {
        Ok(t) => t,
        Err(st) => return st,
    };
    let dst = match resolve_texture_backing(
        state,
        host,
        task_id,
        cmd.destination,
        cmd.destination_level,
        cmd.destination_slice,
    ) {
        Ok(t) => t,
        Err(st) => return st,
    };
    // Options apply to both ends; plane bpp must agree under the selected aspect.
    let (want_depth, want_stencil, src_bpp) = match copy_aspect_for_options(src.pixel_format(), cmd)
    {
        Ok(v) => v,
        Err(st) => return st,
    };
    let (_, _, dst_bpp) = match copy_aspect_for_options(dst.pixel_format(), cmd) {
        Ok(v) => v,
        Err(st) => return st,
    };
    if src_bpp != dst_bpp {
        return br(BlitStatus::Unsupported, "t2t_bpp_mismatch");
    }
    if src.pixel_format() != 0
        && dst.pixel_format() != 0
        && src.pixel_format() != dst.pixel_format()
    {
        return br(BlitStatus::Unsupported, "t2t_format_mismatch");
    }
    let copy_bpp = src_bpp;
    let repack_src =
        pixel_format::blit_aspect_needs_repack(src.pixel_format(), want_depth, want_stencil);
    let repack_dst =
        pixel_format::blit_aspect_needs_repack(dst.pixel_format(), want_depth, want_stencil);
    let any_t11 = src.is_type11() || dst.is_type11();
    if any_t11 && (cmd.source_origin.z != 0 || cmd.destination_origin.z != 0) {
        return br(BlitStatus::Unsupported, "t2t_t11_z");
    }
    let sox = cmd.source_origin.x;
    let soy = cmd.source_origin.y;
    let soz = cmd.source_origin.z;
    let dox = cmd.destination_origin.x;
    let doy = cmd.destination_origin.y;
    let doz = cmd.destination_origin.z;
    if sox > src.width() as u64
        || soy > src.height() as u64
        || soz > src.depth() as u64
        || dox > dst.width() as u64
        || doy > dst.height() as u64
        || doz > dst.depth() as u64
    {
        return br(BlitStatus::Bounds, "t2t_origin_oob");
    }
    let mut copy_w = clamp_extent(cmd.source_size.width, src.width() as u64 - sox);
    copy_w = copy_w.min(dst.width() as u64 - dox);
    let mut copy_h = clamp_extent(cmd.source_size.height, src.height() as u64 - soy);
    copy_h = copy_h.min(dst.height() as u64 - doy);
    let copy_d = if cmd.source_size.depth == 0 {
        0
    } else {
        let mut d = clamp_extent(cmd.source_size.depth, src.depth() as u64 - soz);
        d = d.min(dst.depth() as u64 - doz);
        d
    };
    if any_t11 && copy_d > 1 {
        return br(BlitStatus::Unsupported, "t2t_t11_volume");
    }
    if copy_w == 0 || copy_h == 0 || copy_d == 0 {
        return BlitStatus::ZeroExtent;
    }
    let row_bytes = match copy_w.checked_mul(copy_bpp as u64) {
        Some(v) => v,
        None => return br(BlitStatus::Capacity, "t2t_row_bytes_overflow"),
    };
    // Combined DS + aspect: extract plane from src, insert into dst (RMW).
    if repack_src || repack_dst {
        let src_storage = texture_storage_bpp(src.pixel_format()).unwrap_or(copy_bpp);
        let dst_storage = texture_storage_bpp(dst.pixel_format()).unwrap_or(copy_bpp);
        let plane_row = row_bytes as usize;
        let mut plane = vec![0u8; plane_row];
        let mut src_packed = vec![0u8; (copy_w as usize).saturating_mul(src_storage as usize)];
        let mut dst_packed = vec![0u8; (copy_w as usize).saturating_mul(dst_storage as usize)];
        for z in 0..copy_d {
            for y in 0..copy_h {
                if repack_src {
                    if let Err(st) = read_texture_storage_row(
                        state,
                        host,
                        task_id,
                        &src,
                        sox,
                        soy,
                        soz + z,
                        y,
                        copy_w as u32,
                        src_storage,
                        &mut src_packed,
                    ) {
                        return st;
                    }
                    if !pixel_format::extract_plane_row(
                        src.pixel_format(),
                        want_depth,
                        want_stencil,
                        &src_packed,
                        copy_w as u32,
                        &mut plane,
                    ) {
                        return br(BlitStatus::Unsupported, "t2t_extract_plane");
                    }
                } else if let Err(st) = read_texture_row(
                    state,
                    host,
                    task_id,
                    &src,
                    sox,
                    soy,
                    soz + z,
                    y,
                    row_bytes,
                    &mut plane,
                ) {
                    return st;
                }
                if repack_dst {
                    if let Err(st) = read_texture_storage_row(
                        state,
                        host,
                        task_id,
                        &dst,
                        dox,
                        doy,
                        doz + z,
                        y,
                        copy_w as u32,
                        dst_storage,
                        &mut dst_packed,
                    ) {
                        return st;
                    }
                    if !pixel_format::insert_plane_row(
                        dst.pixel_format(),
                        want_depth,
                        want_stencil,
                        &plane,
                        copy_w as u32,
                        &mut dst_packed,
                    ) {
                        return br(BlitStatus::Unsupported, "t2t_insert_plane");
                    }
                    if let Err(st) = write_texture_storage_row(
                        state,
                        host,
                        task_id,
                        &dst,
                        dox,
                        doy,
                        doz + z,
                        y,
                        copy_w as u32,
                        dst_storage,
                        &dst_packed,
                    ) {
                        return st;
                    }
                } else if let Err(st) = write_texture_row(
                    state,
                    host,
                    task_id,
                    &dst,
                    dox,
                    doy,
                    doz + z,
                    y,
                    row_bytes,
                    &plane,
                ) {
                    return st;
                }
            }
        }
        return BlitStatus::Ok;
    }
    // Fast path: both linear → existing GVA span copy.
    if let (TextureBacking::Linear(ref sl), TextureBacking::Linear(ref dl)) = (&src, &dst) {
        let src_off = match sl.texel_offset(sox, soy, soz) {
            Some(v) => v,
            None => return br(BlitStatus::Bounds, "t2t_src_texel_oob"),
        };
        let dst_off = match dl.texel_offset(dox, doy, doz) {
            Some(v) => v,
            None => return br(BlitStatus::Bounds, "t2t_dst_texel_oob"),
        };
        let src_bpi = match sl.bytes_per_image() {
            Some(v) => v,
            None => return br(BlitStatus::Capacity, "t2t_src_bpi_overflow"),
        };
        let dst_bpi = match dl.bytes_per_image() {
            Some(v) => v,
            None => return br(BlitStatus::Capacity, "t2t_dst_bpi_overflow"),
        };
        let src_gva = match sl.base_gva.checked_add(src_off) {
            Some(v) => v,
            None => return br(BlitStatus::Bounds, "t2t_src_gva_overflow"),
        };
        let dst_gva = match dl.base_gva.checked_add(dst_off) {
            Some(v) => v,
            None => return br(BlitStatus::Bounds, "t2t_dst_gva_overflow"),
        };
        // Exact identity self-copy is a no-op: source and destination name the
        // same guest bytes with the same layout, so every row reads and writes
        // the same address — the destination already holds the source content.
        // Observed live (Ventura x86, media apps): the guest issues
        // copyFromTexture:X toTexture:X sourceOrigin==destinationOrigin on small
        // window textures (src_ref==dst_ref, src_off==dst_off). Copying bytes
        // onto themselves changes nothing, so succeed without work rather than
        // rejecting it as Overlap (which returned a spurious error to the guest
        // blit encoder and dropped a copy the guest treats as complete). A
        // genuinely-shifted overlap (src_gva != dst_gva, undefined in Metal)
        // still falls through to the reject below.
        if src_gva == dst_gva && sl.row_stride == dl.row_stride && src_bpi == dst_bpi {
            return BlitStatus::Ok;
        }
        // Same allocation (self-copy or aliased view) but a different region:
        // safe unless the source and destination TEXEL rectangles actually
        // intersect. Two axis-aligned rectangles overlap iff they overlap on
        // EVERY axis — the correct model when src/dst share the texel layout
        // (same row stride + per-image stride, guaranteed for a same-texture
        // self-copy). The prior byte-span test collapsed row_stride into one
        // contiguous span and produced phantom overlaps for strided sub-rect
        // column copies: a 1-wide column shifted N texels right (live: src_off=0
        // dst_off=64, 4-byte rows 1024 apart) never collides row-to-row, yet the
        // span test flagged it and dropped a legitimate copy. If the layouts
        // differ (exotic aliased views with mismatched strides), texel grids are
        // incomparable — keep the conservative byte-span reject there.
        if sl.base_gva == dl.base_gva {
            let same_layout = sl.row_stride == dl.row_stride && src_bpi == dst_bpi;
            let overlaps = if same_layout {
                let x = sox < dox + copy_w && dox < sox + copy_w;
                let y = soy < doy + copy_h && doy < soy + copy_h;
                let z = soz < doz + copy_d && doz < soz + copy_d;
                x && y && z
            } else {
                let s_end =
                    src_off.saturating_add(row_bytes.saturating_mul(copy_h).saturating_mul(copy_d));
                let d_end =
                    dst_off.saturating_add(row_bytes.saturating_mul(copy_h).saturating_mul(copy_d));
                ranges_overlap(
                    src_off,
                    s_end.saturating_sub(src_off),
                    dst_off,
                    d_end.saturating_sub(dst_off),
                )
            };
            if overlaps {
                note_t2t_overlap(
                    task_id,
                    cmd.source,
                    cmd.destination,
                    src_off,
                    dst_off,
                    row_bytes,
                    sl.row_stride,
                    copy_h,
                    copy_d,
                );
                return br(BlitStatus::Overlap, "t2t_overlap");
            }
        }
        return match copy_row_region(
            host,
            state,
            task_id,
            src_gva,
            sl.row_stride,
            src_bpi,
            dst_gva,
            dl.row_stride,
            dst_bpi,
            row_bytes,
            copy_h,
            copy_d,
        ) {
            Ok(()) => BlitStatus::Ok,
            Err(st) => st,
        };
    }
    // Mixed or type-11↔type-11: stage rows.
    let mut row = vec![0u8; row_bytes as usize];
    for z in 0..copy_d {
        for y in 0..copy_h {
            if let Err(st) = read_texture_row(
                state,
                host,
                task_id,
                &src,
                sox,
                soy,
                soz + z,
                y,
                row_bytes,
                &mut row,
            ) {
                return st;
            }
            if let Err(st) = write_texture_row(
                state,
                host,
                task_id,
                &dst,
                dox,
                doy,
                doz + z,
                y,
                row_bytes,
                &row,
            ) {
                return st;
            }
        }
    }
    BlitStatus::Ok
}

/// `0x13e copyFromTexture:…sliceCount:levelCount:` — whole-surface multi-slice/level.
///
/// For each level offset in `0..level_count`:
/// - **depth == 1:** copies full `width×height` across `slice_count` consecutive
///   array slices (`origin (0,0,0)`, size `w×h×1`).
/// - **depth > 1 (3D volume):** Metal requires `sliceCount == 1` and source/
///   destination slices 0; copies full `width×height×depth` of that mip with
///   depth planes strided by `bytes_per_image`. Linear type-2/3 only.
///
/// Zero `slice_count` or `level_count` is a Metal no-op ([`BlitStatus::ZeroExtent`]).
fn exec_copy_texture_to_texture_slice_level<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    cmd: &Command,
) -> BlitStatus {
    if cmd.slice_count == 0 || cmd.level_count == 0 {
        return BlitStatus::ZeroExtent;
    }
    if cmd.source == 0 || cmd.destination == 0 {
        return br(BlitStatus::MissingResource, "sl_missing_ref");
    }
    for level_i in 0..cmd.level_count {
        let src_level = match cmd.source_level.checked_add(level_i) {
            Some(v) => v,
            None => return br(BlitStatus::Bounds, "sl_src_level_overflow"),
        };
        let dst_level = match cmd.destination_level.checked_add(level_i) {
            Some(v) => v,
            None => return br(BlitStatus::Bounds, "sl_dst_level_overflow"),
        };
        let last_slice_delta = match cmd.slice_count.checked_sub(1) {
            Some(v) => v,
            None => return br(BlitStatus::Bounds, "sl_slice_count_underflow"),
        };
        let src_last_slice = match cmd.source_slice.checked_add(last_slice_delta) {
            Some(v) => v,
            None => return br(BlitStatus::Bounds, "sl_src_slice_overflow"),
        };
        let dst_last_slice = match cmd.destination_slice.checked_add(last_slice_delta) {
            Some(v) => v,
            None => return br(BlitStatus::Bounds, "sl_dst_slice_overflow"),
        };

        // Resolve the starting slice at this level for geometry / format.
        // Volume (depth>1) forms use slice 0 only; non-zero source_slice on a
        // depth-1 packing fails at resolve (Bounds). For volumes we require
        // sliceCount==1 and slices 0 before any multi-slice last-index walk.
        let src0 = match resolve_texture_backing(
            state,
            host,
            task_id,
            cmd.source,
            src_level,
            cmd.source_slice,
        ) {
            Ok(t) => t,
            Err(st) => return st,
        };
        let dst0 = match resolve_texture_backing(
            state,
            host,
            task_id,
            cmd.destination,
            dst_level,
            cmd.destination_slice,
        ) {
            Ok(t) => t,
            Err(st) => return st,
        };
        if src0.bpp() != dst0.bpp() {
            return br(BlitStatus::Unsupported, "sl_bpp_mismatch");
        }
        if src0.pixel_format() != 0
            && dst0.pixel_format() != 0
            && src0.pixel_format() != dst0.pixel_format()
        {
            return br(BlitStatus::Unsupported, "sl_format_mismatch");
        }
        if src0.width() != dst0.width() || src0.height() != dst0.height() {
            return br(BlitStatus::Bounds, "sl_dim_mismatch");
        }
        if src0.depth() != dst0.depth() {
            return br(BlitStatus::Bounds, "sl_depth_mismatch");
        }
        let w = src0.width();
        let h = src0.height();
        let d = src0.depth();
        if w == 0 || h == 0 || d == 0 {
            return br(BlitStatus::Bounds, "sl_zero_geom");
        }
        let is_volume = d > 1;
        // Metal 3D whole-surface: sliceCount must be 1, slices 0; full depth of mip.
        if is_volume {
            if cmd.slice_count != 1 || cmd.source_slice != 0 || cmd.destination_slice != 0 {
                return br(BlitStatus::Unsupported, "sl_volume_slice_constraint");
            }
            // Type-11 is 2D (depth 1); volume endpoints are linear only.
            if src0.is_type11() || dst0.is_type11() {
                return br(BlitStatus::Unsupported, "sl_volume_t11");
            }
        } else if cmd.slice_count > 1 {
            // Array form: last slice must resolve (view / packing bounds).
            if let Err(st) =
                resolve_texture_backing(state, host, task_id, cmd.source, src_level, src_last_slice)
            {
                return st;
            }
            if let Err(st) = resolve_texture_backing(
                state,
                host,
                task_id,
                cmd.destination,
                dst_level,
                dst_last_slice,
            ) {
                return st;
            }
        }
        let bpp = src0.bpp();
        let row_bytes = match (w as u64).checked_mul(bpp as u64) {
            Some(v) => v,
            None => return br(BlitStatus::Capacity, "sl_row_bytes_overflow"),
        };

        // Linear: multi-slice (depth-1) or full volume (depth>1).
        if let (TextureBacking::Linear(ref sl), TextureBacking::Linear(ref dl)) = (&src0, &dst0) {
            if !is_volume && cmd.slice_count > 1 && (sl.slice_stride == 0 || dl.slice_stride == 0) {
                return br(BlitStatus::Unsupported, "sl_slice_stride_zero");
            }
            let src_off = match sl.texel_offset(0, 0, 0) {
                Some(v) => v,
                None => return br(BlitStatus::Bounds, "sl_src_texel_oob"),
            };
            let dst_off = match dl.texel_offset(0, 0, 0) {
                Some(v) => v,
                None => return br(BlitStatus::Bounds, "sl_dst_texel_oob"),
            };
            let src_gva = match sl.base_gva.checked_add(src_off) {
                Some(v) => v,
                None => return br(BlitStatus::Bounds, "sl_src_gva_overflow"),
            };
            let dst_gva = match dl.base_gva.checked_add(dst_off) {
                Some(v) => v,
                None => return br(BlitStatus::Bounds, "sl_dst_gva_overflow"),
            };
            // Volume: image_count = depth, stride = bytes_per_image (z planes).
            // Array: image_count = slice_count, stride = slice_stride when multi.
            let (src_img_stride, dst_img_stride, image_count) = if is_volume {
                let src_bpi = match sl.bytes_per_image() {
                    Some(v) if v > 0 => v,
                    _ => return br(BlitStatus::Bounds, "sl_src_bpi_zero"),
                };
                let dst_bpi = match dl.bytes_per_image() {
                    Some(v) if v > 0 => v,
                    _ => return br(BlitStatus::Bounds, "sl_dst_bpi_zero"),
                };
                (src_bpi, dst_bpi, d as u64)
            } else if cmd.slice_count <= 1 {
                (
                    sl.bytes_per_image().unwrap_or(row_bytes),
                    dl.bytes_per_image().unwrap_or(row_bytes),
                    1u64,
                )
            } else {
                (sl.slice_stride, dl.slice_stride, cmd.slice_count as u64)
            };
            // Same allocation overlap check (conservative).
            if sl.base_gva == dl.base_gva {
                let span = row_bytes
                    .saturating_mul(h as u64)
                    .saturating_mul(image_count);
                if ranges_overlap(src_off, span, dst_off, span) {
                    return br(BlitStatus::Overlap, "sl_overlap");
                }
            }
            if let Err(st) = copy_row_region(
                host,
                state,
                task_id,
                src_gva,
                sl.row_stride,
                src_img_stride,
                dst_gva,
                dl.row_stride,
                dst_img_stride,
                row_bytes,
                h as u64,
                image_count,
            ) {
                return st;
            }
            continue;
        }

        // Type-11 / mixed: depth-1 only (type-11 is 2D); per-slice whole-surface.
        if is_volume {
            return br(BlitStatus::Unsupported, "sl_volume_mixed");
        }
        for si in 0..cmd.slice_count {
            let ss = match cmd.source_slice.checked_add(si) {
                Some(v) => v,
                None => return br(BlitStatus::Bounds, "sl_inner_src_slice_overflow"),
            };
            let ds = match cmd.destination_slice.checked_add(si) {
                Some(v) => v,
                None => return br(BlitStatus::Bounds, "sl_inner_dst_slice_overflow"),
            };
            let src = match resolve_texture_backing(state, host, task_id, cmd.source, src_level, ss)
            {
                Ok(t) => t,
                Err(st) => return st,
            };
            let dst =
                match resolve_texture_backing(state, host, task_id, cmd.destination, dst_level, ds)
                {
                    Ok(t) => t,
                    Err(st) => return st,
                };
            if src.width() != w || src.height() != h || dst.width() != w || dst.height() != h {
                return br(BlitStatus::Bounds, "sl_inner_dim_mismatch");
            }
            let mut row = vec![0u8; row_bytes as usize];
            for y in 0..h as u64 {
                if let Err(st) =
                    read_texture_row(state, host, task_id, &src, 0, 0, 0, y, row_bytes, &mut row)
                {
                    return st;
                }
                if let Err(st) =
                    write_texture_row(state, host, task_id, &dst, 0, 0, 0, y, row_bytes, &row)
                {
                    return st;
                }
            }
        }
    }
    BlitStatus::Ok
}

/// Execute blit fence update (`0x13c`) or wait (`0x13d`) on the blit-fence domain.
///
/// See [`fence_exec::execute_fence`].
pub fn execute_blit_fence(state: &mut DeviceState, task_id: u32, cmd: &Command) -> BlitStatus {
    clear_blit_fail_reason();
    if cmd.kind != Kind::Fence {
        return br(BlitStatus::Unsupported, "fence_wrong_kind");
    }
    let action = match cmd.opcode {
        OP_UPDATE_FENCE => FenceAction::Update,
        OP_WAIT_FENCE => FenceAction::Wait,
        _ => return br(BlitStatus::Unsupported, "fence_bad_opcode"),
    };
    blit_status_from_fence(fence_exec::execute_fence(
        state,
        task_id,
        FenceDomain::BlitFence,
        cmd.fence,
        action,
        0,
    ))
}

/// Re-express a fence outcome as a blit outcome, carrying the reason across.
///
/// Named rather than inlined so the reason-forwarding is directly testable: the
/// `Unsupported` arm used to write a flat `fence_unsupported`, so all seven
/// fence/event refusals — bad domain, event-on-fence-path, either timeout form,
/// either invalid plan, unknown event kind — reached the blit dispatch line as
/// one indistinguishable reason. The forwarded slug is registered by
/// `FenceStatus`, not by this file.
pub(crate) fn blit_status_from_fence(status: FenceStatus) -> BlitStatus {
    match status {
        FenceStatus::Ok => BlitStatus::Ok,
        FenceStatus::Pending => BlitStatus::FencePending,
        FenceStatus::Missing => br(BlitStatus::MissingResource, "fence_missing"),
        FenceStatus::Unsupported(why) => br(BlitStatus::Unsupported, why),
    }
}

/// Blit into a type-11 destination writes the guest pages, and the mapping's
/// Metal texture is a view over those same bytes (unified memory) — content
/// is coherent by construction, no invalidation needed.
fn invalidate_type11_last_store(_state: &mut DeviceState, _dst: &TextureBacking) {}

/// Execute a decoded blit command on the product path.
///
/// Returns [`BlitStatus::Unsupported`] for resource/image/mipmap opcodes
/// that other modules own or that are protocol no-ops (caller should not count
/// those as copy/fill failures). Fences use [`execute_blit_fence`].
pub fn execute_blit<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    cmd: &Command,
) -> BlitStatus {
    // Fresh reason channel per command: an uninstrumented failure reports empty
    // rather than a stale slug left by a prior blit (see `br` / `blit_fail_reason`).
    clear_blit_fail_reason();
    match cmd.kind {
        Kind::FillBuffer => exec_fill_buffer(state, host, task_id, cmd),
        Kind::Copy => match cmd.copy_kind {
            CopyKind::BufferToBuffer => exec_copy_buffer_to_buffer(state, host, task_id, cmd),
            CopyKind::BufferToTexture => {
                let st = exec_copy_buffer_to_texture(state, host, task_id, cmd);
                if matches!(st, BlitStatus::Ok) {
                    // Re-resolve dest for invalidate (exec already wrote).
                    if let Ok(dst) = resolve_texture_backing(
                        state,
                        host,
                        task_id,
                        cmd.destination,
                        cmd.destination_level,
                        cmd.destination_slice,
                    ) {
                        invalidate_type11_last_store(state, &dst);
                        if dst.is_type11() {
                            crate::observe::line(format!(
                                "blit buf→tex type11 dst_ref={} mid={} {}x{} ok",
                                cmd.destination,
                                match &dst {
                                    TextureBacking::Type11(t) => t.mapping_id,
                                    _ => 0,
                                },
                                dst.width(),
                                dst.height()
                            ));
                        }
                    }
                }
                st
            }
            CopyKind::TextureToBuffer => exec_copy_texture_to_buffer(state, host, task_id, cmd),
            CopyKind::TextureToTexture => {
                let st = exec_copy_texture_to_texture(state, host, task_id, cmd);
                if matches!(st, BlitStatus::Ok) {
                    if let Ok(dst) = resolve_texture_backing(
                        state,
                        host,
                        task_id,
                        cmd.destination,
                        cmd.destination_level,
                        cmd.destination_slice,
                    ) {
                        invalidate_type11_last_store(state, &dst);
                        if dst.is_type11() {
                            crate::observe::line(format!(
                                "blit tex→tex type11 src_ref={} dst_ref={} mid={} {}x{} origin=({},{}) size=({},{}) ok",
                                cmd.source,
                                cmd.destination,
                                match &dst {
                                    TextureBacking::Type11(t) => t.mapping_id,
                                    _ => 0,
                                },
                                dst.width(),
                                dst.height(),
                                cmd.destination_origin.x,
                                cmd.destination_origin.y,
                                cmd.source_size.width,
                                cmd.source_size.height
                            ));
                        }
                    }
                }
                st
            }
            CopyKind::TextureToTextureSliceLevel => {
                let st = exec_copy_texture_to_texture_slice_level(state, host, task_id, cmd);
                if matches!(st, BlitStatus::Ok) {
                    if let Ok(dst) = resolve_texture_backing(
                        state,
                        host,
                        task_id,
                        cmd.destination,
                        cmd.destination_level,
                        cmd.destination_slice,
                    ) {
                        invalidate_type11_last_store(state, &dst);
                    }
                }
                st
            }
            CopyKind::None => br(BlitStatus::Unsupported, "copy_kind_none"),
        },
        Kind::Fence => execute_blit_fence(state, task_id, cmd),
        Kind::Resource | Kind::Image | Kind::Unknown => {
            br(BlitStatus::Unsupported, "blit_kind_unsupported")
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::field_reassign_with_default,
        clippy::too_many_arguments,
        reason = "wire fixtures are assembled field by field to keep each protocol case explicit"
    )]

    use super::*;
    use crate::contract::endian::{st16, st32, st64};
    use crate::contract::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
    use crate::contract::pixel_format::{MTL_FORMAT_BGRA8_UNORM, MTL_FORMAT_RGBA8_UNORM};
    use crate::model::{DeviceId, FENCE_DOMAIN_BLIT, PAGE_SHIFT_ARM64E};
    use crate::runtime::decode::blit::{self, OP_COPY_BUFFER_TO_BUFFER, OP_FILL_BUFFER};
    use crate::runtime::decode::resource::{
        list_object_entry_offset, LINEAR_DESC_HANDLE, LINEAR_DESC_MIN_LEN, LINEAR_DESC_SIZE,
        OBJECT_LIST_ENTRY_LEN, OBJECT_TYPE_BUFFER, RESOURCE_PAGE_SHIFT,
    };
    use crate::runtime::host::FakeHost;
    use crate::runtime::objects;
    use crate::runtime::plan::blit::{plan_from_bytes, PlannedBlit};

    /// The channel is the whole diagnostic for this rail: 177 checks collapse
    /// into eight statuses, so a refusal that reaches the dispatch line without a
    /// reason says almost nothing. An uninstrumented site used to render a bare
    /// `reason=` with nothing after it — not greppable, and indistinguishable from
    /// a missing field rather than from a missing *reason*.
    #[test]
    fn an_unattributed_refusal_names_the_gap_rather_than_rendering_an_empty_reason() {
        use crate::observe::{Emit, Refusal};

        clear_blit_fail_reason();
        assert_eq!(
            BlitStatus::Bounds.refusal(),
            Some("blit_unattributed"),
            "a refusal with an empty channel must still name something"
        );
        let line = Emit::refusal("blit", &BlitStatus::Bounds)
            .expect("a refusal produces a line")
            .render();
        assert_eq!(line, "blit reason=blit_unattributed");
        assert!(
            !line.contains("reason= "),
            "the line rendered an empty reason: {line}"
        );

        // An instrumented site names its own check instead.
        let st = br(BlitStatus::Bounds, "fill_out_of_range");
        assert_eq!(st.refusal(), Some("fill_out_of_range"));
    }

    /// `Ok`, a zero-extent no-op and a soft fence wait are control flow. The
    /// dispatch sites count them as success or as pending, and the guest re-polls
    /// the wait every drain — logging any of them floods the always-on sink.
    /// `Emit::refusal` returns `None`, so no caller can log one by accident.
    #[test]
    fn zero_extent_and_pending_fences_are_control_flow_not_refusals() {
        use crate::observe::{Emit, Refusal};

        // A stale channel value must not resurrect a success as a refusal.
        let _ = br(BlitStatus::Bounds, "fill_out_of_range");
        for ok in [
            BlitStatus::Ok,
            BlitStatus::ZeroExtent,
            BlitStatus::FencePending,
        ] {
            assert_eq!(ok.refusal(), None, "{ok:?} is not a refusal");
            assert!(
                Emit::refusal("blit", &ok).is_none(),
                "{ok:?} produced a loggable line"
            );
        }
    }

    /// One-level page table: GVA pages 0..7 → data PFNs `data_base_pfn + i`.
    fn setup_task_pages(host: &mut FakeHost, state: &mut DeviceState, data_base_pfn: u32) {
        let dir_pfn = 2u32;
        let root_pfn = 3u32;
        let dir_gpa = (dir_pfn as u64) << PAGE_SHIFT_ARM64E;
        let root_gpa = (root_pfn as u64) << PAGE_SHIFT_ARM64E;
        host.map_range(dir_gpa, 0x20, 0);
        host.map_range(root_gpa, 0x4000, 0);
        let mut d = [0u8; 8];
        st32(&mut d[DIRECTORY_ROOT_PFN as usize..], root_pfn);
        st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
        let _ = host.write_gpa(dir_gpa, &d);
        for i in 0..8u32 {
            let pfn = data_base_pfn + i;
            host.map_range((pfn as u64) << PAGE_SHIFT_ARM64E, 0x4000, 0);
            let mut pte = [0u8; 4];
            st32(&mut pte, pfn);
            let _ = host.write_gpa(root_gpa + (i as u64) * 4, &pte);
        }
        assert!(state.define_task(1, 0x1000, dir_pfn));
    }

    /// Install type-1 buffer: object-list at GVA 0, descriptor at GVA 0x100 + ref*0x20.
    fn install_buffer(
        host: &mut FakeHost,
        state: &mut DeviceState,
        obj_ref: u32,
        handle: u32,
        size: u64,
    ) {
        assert!(state.set_object_list(1, 0, 16));
        let mut desc = vec![0u8; LINEAR_DESC_MIN_LEN];
        st64(&mut desc[LINEAR_DESC_SIZE..], size);
        st64(&mut desc[LINEAR_DESC_HANDLE..], handle as u64);
        let desc_gva = 0x100u64 + (obj_ref as u64) * 0x20;
        assert!(
            gva_mem::write_task_gva(host, &state.tasks[1], desc_gva, &desc, PAGE_SHIFT_ARM64E)
                .is_ok()
        );
        let off = list_object_entry_offset(obj_ref, 16).unwrap();
        let mut entry = [0u8; OBJECT_LIST_ENTRY_LEN];
        let packed = (OBJECT_TYPE_BUFFER as u32) | ((LINEAR_DESC_MIN_LEN as u32) << 8);
        st32(&mut entry[0..], packed);
        entry[4..12].copy_from_slice(&desc_gva.to_le_bytes());
        assert!(
            gva_mem::write_task_gva(host, &state.tasks[1], off, &entry, PAGE_SHIFT_ARM64E).is_ok()
        );
        let e = objects::lookup_list_entry(state, host, 1, obj_ref).expect("entry");
        assert_eq!(e.object_type, OBJECT_TYPE_BUFFER);
    }

    #[test]
    fn range_overlap_helper() {
        assert!(ranges_overlap(0, 10, 5, 10));
        assert!(!ranges_overlap(0, 10, 10, 5));
        assert!(!ranges_overlap(0, 0, 0, 5));
    }

    #[test]
    fn range_fits_helper() {
        assert!(range_fits(0, 10, 10));
        assert!(range_fits(5, 5, 10));
        assert!(!range_fits(5, 6, 10));
        assert!(!range_fits(11, 0, 10));
    }

    #[test]
    fn decode_fill_and_plan() {
        let mut v = vec![0u8; 0x20];
        st32(&mut v[0..], OP_FILL_BUFFER);
        st32(&mut v[4..], 0x20);
        st32(&mut v[8..], 3);
        st64(&mut v[0x0c..], 0x10);
        st64(&mut v[0x14..], 8);
        v[0x1c] = 0xa5;
        let cmd = blit::decode(&v).unwrap();
        assert_eq!(cmd.kind, Kind::FillBuffer);
        assert_eq!(cmd.buffer, 3);
        assert_eq!(cmd.range_location, 0x10);
        assert_eq!(cmd.range_length, 8);
        assert_eq!(cmd.fill_value, 0xa5);
        match plan_from_bytes(&v).unwrap() {
            PlannedBlit::Fill(f) => {
                assert_eq!(f.buffer, 3);
                assert_eq!(f.fill_value, 0xa5);
            }
            _ => panic!("expected fill"),
        }
    }

    #[test]
    fn decode_b2b() {
        let mut v = vec![0u8; 0x28];
        st32(&mut v[0..], OP_COPY_BUFFER_TO_BUFFER);
        st32(&mut v[4..], 0x28);
        st32(&mut v[8..], 1);
        st32(&mut v[12..], 2);
        st64(&mut v[0x10..], 4);
        st64(&mut v[0x18..], 8);
        st64(&mut v[0x20..], 16);
        let cmd = blit::decode(&v).unwrap();
        assert_eq!(cmd.copy_kind, CopyKind::BufferToBuffer);
        assert_eq!(cmd.size, 16);
        assert_eq!(cmd.source_offset, 4);
        assert_eq!(cmd.destination_offset, 8);
    }

    #[test]
    fn fill_buffer_roundtrip() {
        let mut host = FakeHost::new();
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        setup_task_pages(&mut host, &mut state, 4);
        install_buffer(&mut host, &mut state, 7, 1, 256);
        let mut cmd = Command::default();
        cmd.kind = Kind::FillBuffer;
        cmd.buffer = 7;
        cmd.range_location = 16;
        cmd.range_length = 8;
        cmd.fill_value = 0x5a;
        assert_eq!(execute_blit(&mut state, &mut host, 1, &cmd), BlitStatus::Ok);
        let mut out = [0u8; 8];
        let gva = (1u64 << RESOURCE_PAGE_SHIFT) + 16;
        assert!(
            gva_mem::read_task_gva(&host, &state.tasks[1], gva, &mut out, PAGE_SHIFT_ARM64E)
                .is_ok()
        );
        assert_eq!(out, [0x5a; 8]);
    }

    /// The reason channel names *which* collapsed check fired for a coarse
    /// `BlitStatus`, distinguishes distinct causes, is reset per command so a stale
    /// slug never leaks across blits, and stays empty after a successful blit.
    #[test]
    fn blit_fail_reason_names_distinct_causes_and_resets_per_command() {
        let mut host = FakeHost::new();
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        setup_task_pages(&mut host, &mut state, 4);
        install_buffer(&mut host, &mut state, 7, 1, 256);

        // ref==0 → MissingResource, reason "buf_ref_zero".
        let mut cmd = Command::default();
        cmd.kind = Kind::FillBuffer;
        cmd.buffer = 0;
        cmd.range_length = 8;
        assert_eq!(
            execute_blit(&mut state, &mut host, 1, &cmd),
            BlitStatus::MissingResource
        );
        assert_eq!(blit_fail_reason(), "buf_ref_zero");

        // Unbound ref → same coarse status, DIFFERENT reason "buf_no_list_entry".
        cmd.buffer = 42; // never installed
        assert_eq!(
            execute_blit(&mut state, &mut host, 1, &cmd),
            BlitStatus::MissingResource
        );
        assert_eq!(blit_fail_reason(), "buf_no_list_entry");

        // In-bounds range on a valid buffer → the channel is reset at entry and the
        // successful blit leaves it empty (no stale "buf_no_list_entry").
        cmd.buffer = 7;
        cmd.range_location = 16;
        cmd.range_length = 8;
        assert_eq!(execute_blit(&mut state, &mut host, 1, &cmd), BlitStatus::Ok);
        assert_eq!(blit_fail_reason(), "");

        // Out-of-range fill on a valid buffer → Bounds, reason "fill_range_oob".
        cmd.range_location = 250;
        cmd.range_length = 64; // 250+64 > 256
        assert_eq!(
            execute_blit(&mut state, &mut host, 1, &cmd),
            BlitStatus::Bounds
        );
        assert_eq!(blit_fail_reason(), "fill_range_oob");
    }

    #[test]
    fn copy_buffer_to_buffer_roundtrip() {
        let mut host = FakeHost::new();
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        setup_task_pages(&mut host, &mut state, 4);
        install_buffer(&mut host, &mut state, 1, 1, 256);
        install_buffer(&mut host, &mut state, 2, 2, 256);
        let src_gva = 1u64 << RESOURCE_PAGE_SHIFT;
        let pat = [1u8, 2, 3, 4, 5, 6, 7, 8];
        assert!(gva_mem::write_task_gva(
            &mut host,
            &state.tasks[1],
            src_gva + 4,
            &pat,
            PAGE_SHIFT_ARM64E
        )
        .is_ok());
        let mut cmd = Command::default();
        cmd.kind = Kind::Copy;
        cmd.copy_kind = CopyKind::BufferToBuffer;
        cmd.source = 1;
        cmd.destination = 2;
        cmd.source_offset = 4;
        cmd.destination_offset = 8;
        cmd.size = 8;
        assert_eq!(execute_blit(&mut state, &mut host, 1, &cmd), BlitStatus::Ok);
        let mut out = [0u8; 8];
        let dst_gva = (2u64 << RESOURCE_PAGE_SHIFT) + 8;
        assert!(gva_mem::read_task_gva(
            &host,
            &state.tasks[1],
            dst_gva,
            &mut out,
            PAGE_SHIFT_ARM64E
        )
        .is_ok());
        assert_eq!(out, pat);
    }

    #[test]
    fn copy_b2b_overlap_rejected() {
        let mut host = FakeHost::new();
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        setup_task_pages(&mut host, &mut state, 4);
        install_buffer(&mut host, &mut state, 1, 1, 256);
        let mut cmd = Command::default();
        cmd.kind = Kind::Copy;
        cmd.copy_kind = CopyKind::BufferToBuffer;
        cmd.source = 1;
        cmd.destination = 1;
        cmd.source_offset = 0;
        cmd.destination_offset = 4;
        cmd.size = 16;
        assert_eq!(
            execute_blit(&mut state, &mut host, 1, &cmd),
            BlitStatus::Overlap
        );
    }

    #[test]
    fn copy_buffer_to_type11_roundtrip() {
        use crate::contract::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
        use crate::runtime::decode::resource::{
            list_object_entry_offset, OBJECT_LIST_ENTRY_LEN, OBJECT_TYPE_IOSURFACE,
        };
        let mut host = FakeHost::new();
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        setup_task_pages(&mut host, &mut state, 4);
        // Buffer with 8 BGRA pixels (one row of 2 pixels for a 2x1 copy).
        install_buffer(&mut host, &mut state, 1, 1, 256);
        let pat = [0x11u8, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88];
        let src_gva = 1u64 << RESOURCE_PAGE_SHIFT;
        assert!(gva_mem::write_task_gva(
            &mut host,
            &state.tasks[1],
            src_gva,
            &pat,
            PAGE_SHIFT_ARM64E
        )
        .is_ok());

        // Type-11 object ref 3 → mapping 9, 2x2 BGRA.
        let mapping_id = 9u32;
        let pfn = 0x20u32;
        let gpa = (pfn as u64) << PAGE_SHIFT_ARM64E;
        host.map_range(gpa, 0x4000, 0);
        let entry = (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
        state.map_surface(mapping_id);
        {
            let m = state.mappings.get_mut(&mapping_id).unwrap();
            m.mapped = true;
            m.mapping_internal = 1;
            m.page_entries = vec![entry];
        }
        assert!(state.set_mapping_geom(mapping_id, 2, 2, MTL_FORMAT_BGRA8_UNORM));

        assert!(state.set_object_list(1, 0, 16));
        let mut desc = vec![0u8; 0x20];
        st32(&mut desc[0..], mapping_id);
        // format @0x16, width @0x18, height @0x1c (iosurface desc layout)
        st16(&mut desc[0x16..], MTL_FORMAT_BGRA8_UNORM);
        st32(&mut desc[0x18..], 2);
        st32(&mut desc[0x1c..], 2);
        let desc_gva = 0x180u64;
        assert!(gva_mem::write_task_gva(
            &mut host,
            &state.tasks[1],
            desc_gva,
            &desc,
            PAGE_SHIFT_ARM64E
        )
        .is_ok());
        let off = list_object_entry_offset(3, 16).unwrap();
        let mut list_entry = [0u8; OBJECT_LIST_ENTRY_LEN];
        let packed = (OBJECT_TYPE_IOSURFACE as u32) | (0x20u32 << 8);
        st32(&mut list_entry[0..], packed);
        list_entry[4..12].copy_from_slice(&desc_gva.to_le_bytes());
        assert!(gva_mem::write_task_gva(
            &mut host,
            &state.tasks[1],
            off,
            &list_entry,
            PAGE_SHIFT_ARM64E
        )
        .is_ok());

        let mut cmd = Command::default();
        cmd.kind = Kind::Copy;
        cmd.copy_kind = CopyKind::BufferToTexture;
        cmd.source = 1;
        cmd.destination = 3;
        cmd.source_offset = 0;
        cmd.source_bytes_per_row = 8;
        cmd.source_size.width = 2;
        cmd.source_size.height = 1;
        cmd.source_size.depth = 1;
        cmd.destination_origin.x = 0;
        cmd.destination_origin.y = 1;
        assert_eq!(execute_blit(&mut state, &mut host, 1, &cmd), BlitStatus::Ok);

        // Read back the written row via mapping_write.
        let mut back = [0u8; 8];
        assert!(mapping_write::read_rect_raw(
            &mut state, &mut host, mapping_id, 0, 1, 2, 1, &mut back, 8
        ));
        assert_eq!(back, pat);
        // Blit again — unified memory: pages are the only content; gen advances.
        let gen_before = state.mappings[&mapping_id].content_generation;
        assert_eq!(execute_blit(&mut state, &mut host, 1, &cmd), BlitStatus::Ok);
        assert!(state.mappings[&mapping_id].content_generation > gen_before);
    }

    /// type-11→type-11 copy lands source bytes in dest pages (unified content).
    #[test]
    fn copy_type11_to_type11_writes_dst_pages() {
        let mut host = FakeHost::new();
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        setup_task_pages(&mut host, &mut state, 4);
        install_type11(&mut host, &mut state, 3, 3, 0x20);
        install_type11(&mut host, &mut state, 4, 4, 0x21);
        // Seed source mid=3 pages with a known pattern.
        let src_pat = [9u8, 8, 7, 6, 5, 4, 3, 2, 1, 0, 11, 12, 13, 14, 15, 16];
        assert!(mapping_write::write_rect_raw(
            &mut state, &mut host, 3, 0, 0, 2, 2, &src_pat, 8
        ));
        let mut cmd = Command::default();
        cmd.kind = Kind::Copy;
        cmd.copy_kind = CopyKind::TextureToTexture;
        cmd.source = 3;
        cmd.destination = 4;
        cmd.source_origin.x = 0;
        cmd.source_origin.y = 0;
        cmd.destination_origin.x = 0;
        cmd.destination_origin.y = 0;
        cmd.source_size.width = 2;
        cmd.source_size.height = 2;
        cmd.source_size.depth = 1;
        assert_eq!(execute_blit(&mut state, &mut host, 1, &cmd), BlitStatus::Ok);
        let mut back = [0u8; 16];
        assert!(mapping_write::read_rect_raw(
            &mut state, &mut host, 4, 0, 0, 2, 2, &mut back, 8
        ));
        assert_eq!(back, src_pat, "dest pages hold blit content (one copy)");
    }

    /// The reason channel names *which* collapsed check fired inside each of the
    /// rectangular copy executors (texture↔texture, texture→buffer, buffer→texture),
    /// distinguishes distinct causes, and is reset to empty by a subsequent success —
    /// so a `blit_fail reason=<slug> st=Bounds` dispatch line always carries the
    /// specific failing site rather than a bare coarse status.
    #[test]
    fn copy_executor_reason_slugs_name_distinct_sites() {
        let mut host = FakeHost::new();
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        setup_task_pages(&mut host, &mut state, 4);
        install_type11(&mut host, &mut state, 3, 3, 0x20); // 2×2 BGRA, mid 3
        install_type11(&mut host, &mut state, 4, 4, 0x21); // 2×2 BGRA, mid 4
        install_buffer(&mut host, &mut state, 5, 5, 4096);

        // texture→texture: destination origin past a 2×2 target → Bounds.
        let mut cmd = Command::default();
        cmd.kind = Kind::Copy;
        cmd.copy_kind = CopyKind::TextureToTexture;
        cmd.source = 3;
        cmd.destination = 4;
        cmd.destination_origin.x = 3; // > width 2
        cmd.source_size.width = 1;
        cmd.source_size.height = 1;
        cmd.source_size.depth = 1;
        assert_eq!(
            execute_blit(&mut state, &mut host, 1, &cmd),
            BlitStatus::Bounds
        );
        assert_eq!(blit_fail_reason(), "t2t_origin_oob");

        // texture→texture: a type-11 endpoint with a non-zero z origin (type-11 is
        // 2D) → Unsupported, a DIFFERENT reason under the same executor.
        let mut cmd = Command::default();
        cmd.kind = Kind::Copy;
        cmd.copy_kind = CopyKind::TextureToTexture;
        cmd.source = 3;
        cmd.destination = 4;
        cmd.source_origin.z = 1;
        cmd.source_size.width = 1;
        cmd.source_size.height = 1;
        cmd.source_size.depth = 1;
        assert_eq!(
            execute_blit(&mut state, &mut host, 1, &cmd),
            BlitStatus::Unsupported
        );
        assert_eq!(blit_fail_reason(), "t2t_t11_z");

        // texture→buffer: source origin past bounds → Bounds "t2b_origin_oob".
        let mut cmd = Command::default();
        cmd.kind = Kind::Copy;
        cmd.copy_kind = CopyKind::TextureToBuffer;
        cmd.source = 3;
        cmd.destination = 5;
        cmd.source_origin.x = 3; // > width 2
        cmd.source_size.width = 1;
        cmd.source_size.height = 1;
        cmd.source_size.depth = 1;
        assert_eq!(
            execute_blit(&mut state, &mut host, 1, &cmd),
            BlitStatus::Bounds
        );
        assert_eq!(blit_fail_reason(), "t2b_origin_oob");

        // buffer→texture: destination origin past bounds → Bounds "b2t_origin_oob".
        let mut cmd = Command::default();
        cmd.kind = Kind::Copy;
        cmd.copy_kind = CopyKind::BufferToTexture;
        cmd.source = 5;
        cmd.destination = 3;
        cmd.destination_origin.x = 3; // > width 2
        cmd.source_size.width = 1;
        cmd.source_size.height = 1;
        cmd.source_size.depth = 1;
        assert_eq!(
            execute_blit(&mut state, &mut host, 1, &cmd),
            BlitStatus::Bounds
        );
        assert_eq!(blit_fail_reason(), "b2t_origin_oob");

        // A full-target valid type-11→type-11 copy succeeds and resets the channel,
        // so no stale slug leaks into the next command's dispatch line.
        let mut cmd = Command::default();
        cmd.kind = Kind::Copy;
        cmd.copy_kind = CopyKind::TextureToTexture;
        cmd.source = 3;
        cmd.destination = 4;
        cmd.source_size.width = 2;
        cmd.source_size.height = 2;
        cmd.source_size.depth = 1;
        assert_eq!(execute_blit(&mut state, &mut host, 1, &cmd), BlitStatus::Ok);
        assert_eq!(blit_fail_reason(), "");
    }

    /// Install type-11 object-list entry + mapping pages (2×2 BGRA).
    fn install_type11(
        host: &mut FakeHost,
        state: &mut DeviceState,
        obj_ref: u32,
        mapping_id: u32,
        pfn: u32,
    ) {
        use crate::contract::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
        use crate::runtime::decode::resource::{
            list_object_entry_offset, OBJECT_LIST_ENTRY_LEN, OBJECT_TYPE_IOSURFACE,
        };
        let gpa = (pfn as u64) << PAGE_SHIFT_ARM64E;
        host.map_range(gpa, 0x4000, 0);
        let entry = (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
        state.map_surface(mapping_id);
        {
            let m = state.mappings.get_mut(&mapping_id).unwrap();
            m.mapped = true;
            m.mapping_internal = 1;
            m.page_entries = vec![entry];
        }
        assert!(state.set_mapping_geom(mapping_id, 2, 2, MTL_FORMAT_BGRA8_UNORM));
        assert!(state.set_object_list(1, 0, 16));
        let mut desc = vec![0u8; 0x20];
        st32(&mut desc[0..], mapping_id);
        st16(&mut desc[0x16..], MTL_FORMAT_BGRA8_UNORM);
        st32(&mut desc[0x18..], 2);
        st32(&mut desc[0x1c..], 2);
        let desc_gva = 0x180u64 + (obj_ref as u64) * 0x40;
        assert!(
            gva_mem::write_task_gva(host, &state.tasks[1], desc_gva, &desc, PAGE_SHIFT_ARM64E)
                .is_ok()
        );
        let off = list_object_entry_offset(obj_ref, 16).unwrap();
        let mut list_entry = [0u8; OBJECT_LIST_ENTRY_LEN];
        let packed = (OBJECT_TYPE_IOSURFACE as u32) | (0x20u32 << 8);
        st32(&mut list_entry[0..], packed);
        list_entry[4..12].copy_from_slice(&desc_gva.to_le_bytes());
        assert!(gva_mem::write_task_gva(
            host,
            &state.tasks[1],
            off,
            &list_entry,
            PAGE_SHIFT_ARM64E
        )
        .is_ok());
    }

    /// Shared biplanar mapping: plane0 Y 4×2 R8 @512 bpr=64; plane1 UV 2×1 RG8 @1024 bpr=64.
    fn install_biplanar_mapping(
        host: &mut FakeHost,
        state: &mut DeviceState,
        mapping_id: u32,
        pfn: u32,
    ) {
        use crate::contract::iosurface_pages::{
            DEVICE_DESC_ALLOC_SIZE, DEVICE_DESC_LEN, DEVICE_DESC_PLANES, DEVICE_DESC_PLANE_COUNT,
            DEVICE_PLANE_BPE, DEVICE_PLANE_BPR, DEVICE_PLANE_DESC_LEN, DEVICE_PLANE_DIMS,
            DEVICE_PLANE_OFFSET, DEVICE_PLANE_SIZE, PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID,
        };
        let gpa = (pfn as u64) << PAGE_SHIFT_ARM64E;
        host.map_range(gpa, 0x4000, 0);
        let entry = (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
        state.map_surface(mapping_id);
        {
            let m = state.mappings.get_mut(&mapping_id).unwrap();
            m.mapped = true;
            m.mapping_internal = 1;
            m.page_entries = vec![entry];
        }
        let mut device = vec![0u8; DEVICE_DESC_LEN];
        st32(&mut device[DEVICE_DESC_ALLOC_SIZE..], 0x2000);
        device[DEVICE_DESC_PLANE_COUNT] = 2;
        let pack = |w: u32, h: u32| ((w as u64 & 0xffffff) << 8) | ((h as u64 & 0xffffff) << 40);
        let p0 = DEVICE_DESC_PLANES;
        st32(&mut device[p0 + DEVICE_PLANE_OFFSET..], 512);
        st32(&mut device[p0 + DEVICE_PLANE_SIZE..], 512);
        st64(&mut device[p0 + DEVICE_PLANE_DIMS..], pack(4, 2));
        st32(&mut device[p0 + DEVICE_PLANE_BPR..], 64);
        st16(&mut device[p0 + DEVICE_PLANE_BPE..], 1);
        let p1 = DEVICE_DESC_PLANES + DEVICE_PLANE_DESC_LEN;
        st32(&mut device[p1 + DEVICE_PLANE_OFFSET..], 1024);
        st32(&mut device[p1 + DEVICE_PLANE_SIZE..], 256);
        st64(&mut device[p1 + DEVICE_PLANE_DIMS..], pack(2, 1));
        st32(&mut device[p1 + DEVICE_PLANE_BPR..], 64);
        st16(&mut device[p1 + DEVICE_PLANE_BPE..], 2);
        assert!(state.set_mapping_device_desc(mapping_id, &device));
        // Surface-level geom is not the plane; leave has_geom false until texture latch.
    }

    fn install_type11_plane(
        host: &mut FakeHost,
        state: &mut DeviceState,
        obj_ref: u32,
        mapping_id: u32,
        format: u16,
        width: u32,
        height: u32,
    ) {
        use crate::runtime::decode::resource::{
            list_object_entry_offset, OBJECT_LIST_ENTRY_LEN, OBJECT_TYPE_IOSURFACE,
        };
        assert!(state.set_object_list(1, 0, 16));
        let mut desc = vec![0u8; 0x20];
        st32(&mut desc[0..], mapping_id);
        st16(&mut desc[0x16..], format);
        st32(&mut desc[0x18..], width);
        st32(&mut desc[0x1c..], height);
        let desc_gva = 0x180u64 + (obj_ref as u64) * 0x40;
        assert!(
            gva_mem::write_task_gva(host, &state.tasks[1], desc_gva, &desc, PAGE_SHIFT_ARM64E)
                .is_ok()
        );
        let off = list_object_entry_offset(obj_ref, 16).unwrap();
        let mut list_entry = [0u8; OBJECT_LIST_ENTRY_LEN];
        let packed = (OBJECT_TYPE_IOSURFACE as u32) | (0x20u32 << 8);
        st32(&mut list_entry[0..], packed);
        list_entry[4..12].copy_from_slice(&desc_gva.to_le_bytes());
        assert!(gva_mem::write_task_gva(
            host,
            &state.tasks[1],
            off,
            &list_entry,
            PAGE_SHIFT_ARM64E
        )
        .is_ok());
    }

    /// Install a type-5 RefTexture (object_type=5) that names an IOSurface
    /// mapping via `surfaceID@+0` and a serialized 0x62 color-view record
    /// (fmt@+0x16, w@+0x18, h@+0x1c, depth@+0x20, plane@+0x34 — the live blit-
    /// source layout from `decode_type5_texture_view_live_0x62_color_window_view`).
    /// Also installs a single-page mapping at `mapping_id` so the resolve lands.
    fn install_type5(
        host: &mut FakeHost,
        state: &mut DeviceState,
        obj_ref: u32,
        mapping_id: u32,
        pfn: u32,
        format: u16,
        width: u32,
        height: u32,
    ) {
        use crate::contract::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
        use crate::runtime::decode::resource::{list_object_entry_offset, OBJECT_LIST_ENTRY_LEN};
        // Mapping (surfaceID == mapping_id): mapped, one data page, latched geom.
        let gpa = (pfn as u64) << PAGE_SHIFT_ARM64E;
        host.map_range(gpa, 0x4000, 0);
        let entry = (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
        state.map_surface(mapping_id);
        {
            let m = state.mappings.get_mut(&mapping_id).unwrap();
            m.mapped = true;
            m.mapping_internal = 1;
            m.page_entries = vec![entry];
        }
        assert!(state.set_mapping_geom(mapping_id, width, height, format));
        // Type-5 descriptor: 56-byte blob, 0x62 color-view record.
        assert!(state.set_object_list(1, 0, 16));
        let desc_len = 56usize;
        let mut desc = vec![0u8; desc_len];
        st32(&mut desc[objects::TYPE5_SURFACE_ID..], mapping_id);
        st32(&mut desc[objects::TYPE5_ARG_KIND..], 0x2f);
        st32(&mut desc[objects::TYPE5_ARG_BLOB_LEN..], 0x30);
        st32(&mut desc[objects::TYPE5_ARG_OWN_REF..], obj_ref);
        let rec = objects::TYPE5_ARG_RECORD;
        desc[rec] = objects::TYPE5_RECORD_TAG_COLOR_VIEW;
        st16(&mut desc[rec + objects::TYPE5_RECORD_FORMAT..], format);
        st32(&mut desc[rec + objects::TYPE5_RECORD_WIDTH..], width);
        st32(&mut desc[rec + objects::TYPE5_RECORD_HEIGHT..], height);
        st32(&mut desc[rec + objects::TYPE5_RECORD_DEPTH..], 1);
        let desc_gva = 0x180u64 + (obj_ref as u64) * 0x40;
        assert!(
            gva_mem::write_task_gva(host, &state.tasks[1], desc_gva, &desc, PAGE_SHIFT_ARM64E)
                .is_ok()
        );
        let off = list_object_entry_offset(obj_ref, 16).unwrap();
        let mut list_entry = [0u8; OBJECT_LIST_ENTRY_LEN];
        let packed = (objects::OBJECT_TYPE_REF_TEXTURE as u32) | ((desc_len as u32) << 8);
        st32(&mut list_entry[0..], packed);
        list_entry[4..12].copy_from_slice(&desc_gva.to_le_bytes());
        assert!(gva_mem::write_task_gva(
            host,
            &state.tasks[1],
            off,
            &list_entry,
            PAGE_SHIFT_ARM64E
        )
        .is_ok());
        let e = objects::lookup_list_entry(state, host, 1, obj_ref).expect("type5 entry");
        assert_eq!(e.object_type, objects::OBJECT_TYPE_REF_TEXTURE);
    }

    /// Regression guard for the type-5 RefTexture blit-source branch
    /// (`resolve_texture_backing_depth` ~588): a type-5 object whose 0x62 record
    /// names a BGRA8 view must resolve to a `Type11` backing carrying the VIEW
    /// geometry/format (not the base mapping's), so a blit copy from a media /
    /// window backing lands. Mirrors the type-11 install fixtures.
    #[test]
    fn type5_ref_texture_resolves_as_type11_blit_backing() {
        use crate::contract::pixel_format::bytes_per_pixel;
        let mut host = FakeHost::new();
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        setup_task_pages(&mut host, &mut state, 4);
        let mapping_id = 34u32;
        let obj_ref = 12u32;
        let (w, h, fmt) = (2u32, 2u32, MTL_FORMAT_BGRA8_UNORM);
        install_type5(&mut host, &mut state, obj_ref, mapping_id, 0x30, fmt, w, h);
        let backing = resolve_texture_backing(&mut state, &mut host, 1, obj_ref, 0, 0)
            .expect("type-5 blit source must resolve");
        match backing {
            TextureBacking::Type11(t) => {
                assert_eq!(t.mapping_id, mapping_id, "backs the named surface");
                assert_eq!((t.width, t.height), (w, h), "view geometry, not base");
                assert_eq!(t.pixel_format, fmt);
                assert_eq!(t.bpp, bytes_per_pixel(fmt).unwrap());
                assert!(t.row_stride >= (w as u64) * (t.bpp as u64));
                assert!(t.span_end >= t.row_stride * (h as u64));
            }
            TextureBacking::Linear(_) => panic!("expected Type11 backing, got Linear"),
        }
    }

    /// A type-5 record whose tag is neither 0x42 nor 0x62 is unknown wire → the
    /// blit branch must fail closed (`t5_view_decode`), never invent geometry.
    #[test]
    fn type5_unknown_record_tag_fails_closed() {
        let mut host = FakeHost::new();
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        setup_task_pages(&mut host, &mut state, 4);
        let (mapping_id, obj_ref) = (34u32, 12u32);
        install_type5(
            &mut host,
            &mut state,
            obj_ref,
            mapping_id,
            0x30,
            MTL_FORMAT_BGRA8_UNORM,
            2,
            2,
        );
        // Corrupt the record tag to an unknown value in-place.
        let desc_gva = 0x180u64 + (obj_ref as u64) * 0x40;
        let bad = [0x99u8];
        assert!(gva_mem::write_task_gva(
            &mut host,
            &state.tasks[1],
            desc_gva + objects::TYPE5_ARG_RECORD as u64,
            &bad,
            PAGE_SHIFT_ARM64E
        )
        .is_ok());
        match resolve_texture_backing(&mut state, &mut host, 1, obj_ref, 0, 0) {
            Err(st) => assert_eq!(st, BlitStatus::Unsupported),
            Ok(_) => panic!("unknown type-5 record tag must fail closed"),
        }
    }

    #[test]
    fn biplanar_type11_y_and_uv_planes_distinct() {
        use crate::contract::pixel_format::{MTL_FORMAT_R8_UNORM, MTL_FORMAT_RG8_UNORM};
        let mut host = FakeHost::new();
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        setup_task_pages(&mut host, &mut state, 4);
        let mapping_id = 7u32;
        install_biplanar_mapping(&mut host, &mut state, mapping_id, 0x30);
        // Y plane texture ref 10, UV plane texture ref 11 — same mapping_id.
        install_type11_plane(
            &mut host,
            &mut state,
            10,
            mapping_id,
            MTL_FORMAT_R8_UNORM,
            4,
            2,
        );
        install_type11_plane(
            &mut host,
            &mut state,
            11,
            mapping_id,
            MTL_FORMAT_RG8_UNORM,
            2,
            1,
        );

        // Buffer with pattern for Y (4×2 R8 tight = 8 B, use bpr 4).
        let y_pat = [0x11u8, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88];
        {
            use crate::runtime::decode::resource::{
                list_object_entry_offset, OBJECT_LIST_ENTRY_LEN, OBJECT_TYPE_BUFFER,
                RESOURCE_PAGE_SHIFT,
            };
            let mut bdesc = vec![0u8; 16];
            st64(&mut bdesc[0..], 64);
            st32(&mut bdesc[8..], 2); // handle 2
            let bgva = 0x300u64;
            assert!(gva_mem::write_task_gva(
                &mut host,
                &state.tasks[1],
                bgva,
                &bdesc,
                PAGE_SHIFT_ARM64E
            )
            .is_ok());
            let off = list_object_entry_offset(1, 16).unwrap();
            let mut le = [0u8; OBJECT_LIST_ENTRY_LEN];
            let packed = (OBJECT_TYPE_BUFFER as u32) | (16u32 << 8);
            st32(&mut le[0..], packed);
            le[4..12].copy_from_slice(&bgva.to_le_bytes());
            assert!(gva_mem::write_task_gva(
                &mut host,
                &state.tasks[1],
                off,
                &le,
                PAGE_SHIFT_ARM64E
            )
            .is_ok());
            let buf_gva = 2u64 << RESOURCE_PAGE_SHIFT;
            assert!(gva_mem::write_task_gva(
                &mut host,
                &state.tasks[1],
                buf_gva,
                &y_pat,
                PAGE_SHIFT_ARM64E
            )
            .is_ok());
        }

        let mut cmd = Command::default();
        cmd.kind = Kind::Copy;
        cmd.copy_kind = CopyKind::BufferToTexture;
        cmd.source = 1;
        cmd.destination = 10; // Y
        cmd.source_offset = 0;
        cmd.source_bytes_per_row = 4;
        cmd.source_size.width = 4;
        cmd.source_size.height = 2;
        cmd.source_size.depth = 1;
        assert_eq!(execute_blit(&mut state, &mut host, 1, &cmd), BlitStatus::Ok);

        // Y plane base 512, bpr 64: rows at 512 and 576.
        let mut row0 = [0u8; 4];
        let mut row1 = [0u8; 4];
        assert!(mapping_write::read_rect_raw_at(
            &mut state,
            &mut host,
            mapping_id,
            512,
            64,
            512 + 64 + 4,
            0,
            0,
            4,
            1,
            1,
            &mut row0,
            4
        ));
        assert!(mapping_write::read_rect_raw_at(
            &mut state,
            &mut host,
            mapping_id,
            512,
            64,
            512 + 64 + 4,
            0,
            1,
            4,
            1,
            1,
            &mut row1,
            4
        ));
        assert_eq!(row0, y_pat[0..4]);
        assert_eq!(row1, y_pat[4..8]);

        // UV plane: write 2×1 RG8 (4 B) from same buffer offset 0.
        let uv_pat = [0xaau8, 0xbb, 0xcc, 0xdd];
        {
            let buf_gva = 2u64 << crate::runtime::decode::resource::RESOURCE_PAGE_SHIFT;
            assert!(gva_mem::write_task_gva(
                &mut host,
                &state.tasks[1],
                buf_gva,
                &uv_pat,
                PAGE_SHIFT_ARM64E
            )
            .is_ok());
        }
        cmd.destination = 11;
        cmd.source_bytes_per_row = 4;
        cmd.source_size.width = 2;
        cmd.source_size.height = 1;
        assert_eq!(execute_blit(&mut state, &mut host, 1, &cmd), BlitStatus::Ok);

        let mut uv = [0u8; 4];
        assert!(mapping_write::read_rect_raw_at(
            &mut state,
            &mut host,
            mapping_id,
            1024,
            64,
            1024 + 4,
            0,
            0,
            2,
            1,
            2,
            &mut uv,
            4
        ));
        assert_eq!(uv, uv_pat);
        // Y plane must be untouched by UV write.
        assert!(mapping_write::read_rect_raw_at(
            &mut state,
            &mut host,
            mapping_id,
            512,
            64,
            512 + 64 + 4,
            0,
            0,
            4,
            1,
            1,
            &mut row0,
            4
        ));
        assert_eq!(row0, y_pat[0..4]);
    }

    fn install_type8_view(
        host: &mut FakeHost,
        state: &mut DeviceState,
        view_ref: u32,
        base_ref: u32,
        pixel_format: u16,
        level_base: u64,
        swizzle: Option<[u8; 4]>,
    ) {
        use crate::runtime::decode::resource::{
            list_object_entry_offset, OBJECT_LIST_ENTRY_LEN, OBJECT_TYPE_TEXTURE_VIEW,
            TEXTURE_VIEW_DESC_BASE_REF, TEXTURE_VIEW_DESC_LEN, TEXTURE_VIEW_DESC_LEVEL_BASE,
            TEXTURE_VIEW_DESC_LEVEL_COUNT, TEXTURE_VIEW_DESC_OPCODE,
            TEXTURE_VIEW_DESC_PIXEL_FORMAT, TEXTURE_VIEW_DESC_SLICE_BASE,
            TEXTURE_VIEW_DESC_SLICE_COUNT, TEXTURE_VIEW_DESC_SWIZZLE,
            TEXTURE_VIEW_DESC_TEXTURE_REF, TEXTURE_VIEW_DESC_TEXTURE_TYPE, TEXTURE_VIEW_MIN_RANGED,
            TEXTURE_VIEW_MIN_SWIZZLE, TEXTURE_VIEW_MTL_TYPE_2D, TEXTURE_VIEW_OPCODE_RANGED,
            TEXTURE_VIEW_OPCODE_SWIZZLE,
        };
        let (opcode, len) = if swizzle.is_some() {
            (TEXTURE_VIEW_OPCODE_SWIZZLE, TEXTURE_VIEW_MIN_SWIZZLE)
        } else {
            (TEXTURE_VIEW_OPCODE_RANGED, TEXTURE_VIEW_MIN_RANGED)
        };
        let mut desc = vec![0u8; len];
        st32(&mut desc[TEXTURE_VIEW_DESC_OPCODE..], opcode);
        st32(&mut desc[TEXTURE_VIEW_DESC_LEN..], len as u32);
        st32(&mut desc[TEXTURE_VIEW_DESC_TEXTURE_REF..], view_ref);
        st32(&mut desc[TEXTURE_VIEW_DESC_BASE_REF..], base_ref);
        st16(&mut desc[TEXTURE_VIEW_DESC_PIXEL_FORMAT..], pixel_format);
        st16(
            &mut desc[TEXTURE_VIEW_DESC_TEXTURE_TYPE..],
            TEXTURE_VIEW_MTL_TYPE_2D,
        );
        st64(&mut desc[TEXTURE_VIEW_DESC_LEVEL_BASE..], level_base);
        st64(&mut desc[TEXTURE_VIEW_DESC_LEVEL_COUNT..], 1);
        st64(&mut desc[TEXTURE_VIEW_DESC_SLICE_BASE..], 0);
        st64(&mut desc[TEXTURE_VIEW_DESC_SLICE_COUNT..], 1);
        if let Some(sw) = swizzle {
            desc[TEXTURE_VIEW_DESC_SWIZZLE..TEXTURE_VIEW_DESC_SWIZZLE + 4].copy_from_slice(&sw);
        }
        let desc_gva = 0x280u64 + (view_ref as u64) * 0x40;
        assert!(
            gva_mem::write_task_gva(host, &state.tasks[1], desc_gva, &desc, PAGE_SHIFT_ARM64E)
                .is_ok()
        );
        let off = list_object_entry_offset(view_ref, 16).unwrap();
        let mut list_entry = [0u8; OBJECT_LIST_ENTRY_LEN];
        let packed = (OBJECT_TYPE_TEXTURE_VIEW as u32) | ((len as u32) << 8);
        st32(&mut list_entry[0..], packed);
        list_entry[4..12].copy_from_slice(&desc_gva.to_le_bytes());
        assert!(gva_mem::write_task_gva(
            host,
            &state.tasks[1],
            off,
            &list_entry,
            PAGE_SHIFT_ARM64E
        )
        .is_ok());
    }

    #[test]
    fn copy_buffer_to_type8_view_of_type11() {
        let mut host = FakeHost::new();
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        setup_task_pages(&mut host, &mut state, 4);
        install_buffer(&mut host, &mut state, 1, 1, 256);
        let mapping_id = 9u32;
        install_type11(&mut host, &mut state, 3, mapping_id, 0x20);
        // View ref 8 → base 3, level 0, BGRA identity.
        install_type8_view(&mut host, &mut state, 8, 3, MTL_FORMAT_BGRA8_UNORM, 0, None);
        let pat = [0xaau8, 0xbb, 0xcc, 0xdd, 0x11, 0x22, 0x33, 0x44];
        let src_gva = 1u64 << RESOURCE_PAGE_SHIFT;
        assert!(gva_mem::write_task_gva(
            &mut host,
            &state.tasks[1],
            src_gva,
            &pat,
            PAGE_SHIFT_ARM64E
        )
        .is_ok());
        let mut cmd = Command::default();
        cmd.kind = Kind::Copy;
        cmd.copy_kind = CopyKind::BufferToTexture;
        cmd.source = 1;
        cmd.destination = 8; // type-8 view
        cmd.source_offset = 0;
        cmd.source_bytes_per_row = 8;
        cmd.source_size.width = 2;
        cmd.source_size.height = 1;
        cmd.source_size.depth = 1;
        cmd.destination_origin.y = 0;
        assert_eq!(execute_blit(&mut state, &mut host, 1, &cmd), BlitStatus::Ok);
        let mut back = [0u8; 8];
        assert!(mapping_write::read_rect_raw(
            &mut state, &mut host, mapping_id, 0, 0, 2, 1, &mut back, 8
        ));
        assert_eq!(back, pat);
    }

    #[test]
    fn type8_swizzled_view_rejected_for_blit() {
        let mut host = FakeHost::new();
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        setup_task_pages(&mut host, &mut state, 4);
        install_buffer(&mut host, &mut state, 1, 1, 256);
        install_type11(&mut host, &mut state, 3, 9, 0x20);
        // Non-identity swizzle BGRA order selectors.
        install_type8_view(
            &mut host,
            &mut state,
            8,
            3,
            MTL_FORMAT_BGRA8_UNORM,
            0,
            Some([4, 3, 2, 5]),
        );
        let mut cmd = Command::default();
        cmd.kind = Kind::Copy;
        cmd.copy_kind = CopyKind::BufferToTexture;
        cmd.source = 1;
        cmd.destination = 8;
        cmd.source_size.width = 1;
        cmd.source_size.height = 1;
        cmd.source_size.depth = 1;
        cmd.source_bytes_per_row = 4;
        assert_eq!(
            execute_blit(&mut state, &mut host, 1, &cmd),
            BlitStatus::Unsupported
        );
    }

    #[test]
    fn type8_level_base_on_type11_rejected() {
        // Metal forbids mipmapped IOSurfaces; view level_base=1 fail-closes.
        let mut host = FakeHost::new();
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        setup_task_pages(&mut host, &mut state, 4);
        install_buffer(&mut host, &mut state, 1, 1, 256);
        install_type11(&mut host, &mut state, 3, 9, 0x20);
        install_type8_view(
            &mut host,
            &mut state,
            8,
            3,
            MTL_FORMAT_BGRA8_UNORM,
            1, // level_base
            None,
        );
        let mut cmd = Command::default();
        cmd.kind = Kind::Copy;
        cmd.copy_kind = CopyKind::BufferToTexture;
        cmd.source = 1;
        cmd.destination = 8;
        cmd.source_size.width = 1;
        cmd.source_size.height = 1;
        cmd.source_size.depth = 1;
        cmd.source_bytes_per_row = 4;
        assert_eq!(
            execute_blit(&mut state, &mut host, 1, &cmd),
            BlitStatus::Unsupported
        );
    }

    #[test]
    fn texel_offset_math() {
        let t = LinearTextureLevel {
            base_gva: 0,
            alloc_size: 0x1000,
            level_offset: 0x100,
            row_stride: 16,
            slice_stride: 64,
            slice_index: 0,
            width: 4,
            height: 4,
            depth: 1,
            bpp: 4,
            pixel_format: MTL_FORMAT_RGBA8_UNORM,
        };
        // (x=1,y=2) → 0x100 + 2*16 + 1*4 = 0x124
        assert_eq!(t.texel_offset(1, 2, 0), Some(0x124));
        let mut t1 = t;
        t1.slice_index = 2;
        // + 2 * 64 slice stride
        assert_eq!(t1.texel_offset(1, 2, 0), Some(0x124 + 128));
    }

    #[test]
    fn derived_slice_stride_2d() {
        assert_eq!(derived_slice_stride(256, 32, 1), Some(256 * 32));
        assert_eq!(derived_slice_stride(16, 4, 2), Some(16 * 4 * 2));
    }

    /// Multi-mip linear texture + multi-level type-8 view selecting L1.
    #[test]
    fn copy_buffer_to_multilevel_view_l1() {
        use crate::runtime::decode::resource::{
            list_object_entry_offset, OBJECT_LIST_ENTRY_LEN, OBJECT_TYPE_TEXTURE,
            RESOURCE_PAGE_SHIFT, TEXTURE_DESC_BASE_LEN, TEXTURE_DESC_DATA_OFFSET,
            TEXTURE_DESC_HEIGHT, TEXTURE_DESC_LEVEL_RECORDS, TEXTURE_DESC_MIPMAP_LEVEL_COUNT,
            TEXTURE_DESC_MIP_LEVEL_RECORD_LEN, TEXTURE_DESC_PIXEL_FORMAT, TEXTURE_DESC_ROW_STRIDE,
            TEXTURE_DESC_USED_SIZE, TEXTURE_DESC_WIDTH, TEXTURE_LEVEL_DEPTH, TEXTURE_LEVEL_HEIGHT,
            TEXTURE_LEVEL_OFFSET, TEXTURE_LEVEL_ROW_STRIDE, TEXTURE_LEVEL_SIZE,
            TEXTURE_LEVEL_WIDTH,
        };
        let mut host = FakeHost::new();
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        setup_task_pages(&mut host, &mut state, 4);
        install_buffer(&mut host, &mut state, 1, 1, 512);

        // Type-2 texture handle=2, 2 mips: L0 4x2, L1 2x1, RGBA8 (bpp=4).
        let handle = 2u32;
        let levels = 2u32;
        let body = TEXTURE_DESC_BASE_LEN + TEXTURE_DESC_MIP_LEVEL_RECORD_LEN;
        let mut desc = vec![0u8; body];
        st64(&mut desc[0..], 0x4000); // allocation
        st32(&mut desc[8..], handle);
        st16(&mut desc[TEXTURE_DESC_MIPMAP_LEVEL_COUNT..], levels as u16);
        st32(&mut desc[TEXTURE_DESC_DATA_OFFSET..], 0);
        st32(&mut desc[TEXTURE_DESC_USED_SIZE..], 4 * 2 * 4); // L0
        st32(&mut desc[TEXTURE_DESC_ROW_STRIDE..], 16);
        st32(&mut desc[TEXTURE_DESC_WIDTH..], 4);
        st32(&mut desc[TEXTURE_DESC_HEIGHT..], 2);
        let rec = TEXTURE_DESC_LEVEL_RECORDS;
        st64(&mut desc[rec + TEXTURE_LEVEL_OFFSET..], 32); // L1 after L0 32 bytes
        st64(&mut desc[rec + TEXTURE_LEVEL_SIZE..], 8);
        st64(&mut desc[rec + TEXTURE_LEVEL_ROW_STRIDE..], 8);
        st32(&mut desc[rec + TEXTURE_LEVEL_WIDTH..], 2);
        st32(&mut desc[rec + TEXTURE_LEVEL_HEIGHT..], 1);
        st32(&mut desc[rec + TEXTURE_LEVEL_DEPTH..], 1);
        let pf_off = TEXTURE_DESC_PIXEL_FORMAT + TEXTURE_DESC_MIP_LEVEL_RECORD_LEN;
        st16(&mut desc[pf_off..], MTL_FORMAT_RGBA8_UNORM);
        let desc_gva = 0x300u64;
        assert!(gva_mem::write_task_gva(
            &mut host,
            &state.tasks[1],
            desc_gva,
            &desc,
            PAGE_SHIFT_ARM64E
        )
        .is_ok());
        let off = list_object_entry_offset(4, 16).unwrap();
        let mut list_entry = [0u8; OBJECT_LIST_ENTRY_LEN];
        let packed = (OBJECT_TYPE_TEXTURE as u32) | ((body as u32) << 8);
        st32(&mut list_entry[0..], packed);
        list_entry[4..12].copy_from_slice(&desc_gva.to_le_bytes());
        assert!(gva_mem::write_task_gva(
            &mut host,
            &state.tasks[1],
            off,
            &list_entry,
            PAGE_SHIFT_ARM64E
        )
        .is_ok());

        // View: level_base=0, level_count=2 over texture ref 4.
        install_type8_view(&mut host, &mut state, 8, 4, MTL_FORMAT_RGBA8_UNORM, 0, None);
        // Patch level_count to 2 on the installed view.
        {
            use crate::runtime::decode::resource::{
                TEXTURE_VIEW_DESC_LEVEL_COUNT, TEXTURE_VIEW_MIN_RANGED,
            };
            let view_gva = 0x280u64 + 8 * 0x40;
            let mut v = vec![0u8; TEXTURE_VIEW_MIN_RANGED];
            assert!(gva_mem::read_task_gva(
                &host,
                &state.tasks[1],
                view_gva,
                &mut v,
                PAGE_SHIFT_ARM64E
            )
            .is_ok());
            st64(&mut v[TEXTURE_VIEW_DESC_LEVEL_COUNT..], 2);
            assert!(gva_mem::write_task_gva(
                &mut host,
                &state.tasks[1],
                view_gva,
                &v,
                PAGE_SHIFT_ARM64E
            )
            .is_ok());
        }

        // Seed buffer with 2 RGBA pixels for L1 (2x1).
        let pat = [1u8, 2, 3, 4, 5, 6, 7, 8];
        let src_gva = 1u64 << RESOURCE_PAGE_SHIFT;
        assert!(gva_mem::write_task_gva(
            &mut host,
            &state.tasks[1],
            src_gva,
            &pat,
            PAGE_SHIFT_ARM64E
        )
        .is_ok());

        let mut cmd = Command::default();
        cmd.kind = Kind::Copy;
        cmd.copy_kind = CopyKind::BufferToTexture;
        cmd.source = 1;
        cmd.destination = 8;
        cmd.source_level = 1; // relative → absolute L1
        cmd.destination_level = 1;
        cmd.source_offset = 0;
        cmd.source_bytes_per_row = 8;
        cmd.source_size.width = 2;
        cmd.source_size.height = 1;
        cmd.source_size.depth = 1;
        assert_eq!(execute_blit(&mut state, &mut host, 1, &cmd), BlitStatus::Ok);

        // Read L1 from texture handle 2 GVA + 32.
        let l1_gva = ((handle as u64) << RESOURCE_PAGE_SHIFT) + 32;
        let mut back = [0u8; 8];
        assert!(gva_mem::read_task_gva(
            &host,
            &state.tasks[1],
            l1_gva,
            &mut back,
            PAGE_SHIFT_ARM64E
        )
        .is_ok());
        assert_eq!(back, pat);
    }

    #[test]
    fn multilevel_view_relative_level_oob() {
        let mut host = FakeHost::new();
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        setup_task_pages(&mut host, &mut state, 4);
        install_buffer(&mut host, &mut state, 1, 1, 64);
        install_type11(&mut host, &mut state, 3, 9, 0x20);
        // View over type-11 with level_count=1, level_base=0; command level 1 is OOB.
        install_type8_view(&mut host, &mut state, 8, 3, MTL_FORMAT_BGRA8_UNORM, 0, None);
        let mut cmd = Command::default();
        cmd.kind = Kind::Copy;
        cmd.copy_kind = CopyKind::BufferToTexture;
        cmd.source = 1;
        cmd.destination = 8;
        cmd.destination_level = 1; // relative 1 >= count 1
        cmd.source_size.width = 1;
        cmd.source_size.height = 1;
        cmd.source_size.depth = 1;
        cmd.source_bytes_per_row = 4;
        assert_eq!(
            execute_blit(&mut state, &mut host, 1, &cmd),
            BlitStatus::Bounds
        );
    }

    #[test]
    fn texture_view_type_helpers() {
        use crate::runtime::decode::resource::{
            texture_view_type_is_3d, texture_view_type_supported, texture_view_type_uses_slices,
            TEXTURE_VIEW_MTL_TYPE_2D, TEXTURE_VIEW_MTL_TYPE_2D_ARRAY,
            TEXTURE_VIEW_MTL_TYPE_2D_MULTISAMPLE, TEXTURE_VIEW_MTL_TYPE_3D,
            TEXTURE_VIEW_MTL_TYPE_CUBE,
        };
        assert!(texture_view_type_supported(TEXTURE_VIEW_MTL_TYPE_2D));
        assert!(texture_view_type_supported(TEXTURE_VIEW_MTL_TYPE_2D_ARRAY));
        assert!(texture_view_type_supported(TEXTURE_VIEW_MTL_TYPE_CUBE));
        assert!(texture_view_type_supported(TEXTURE_VIEW_MTL_TYPE_3D));
        assert!(!texture_view_type_supported(
            TEXTURE_VIEW_MTL_TYPE_2D_MULTISAMPLE
        ));
        assert!(texture_view_type_uses_slices(
            TEXTURE_VIEW_MTL_TYPE_2D_ARRAY
        ));
        assert!(texture_view_type_uses_slices(TEXTURE_VIEW_MTL_TYPE_CUBE));
        assert!(!texture_view_type_uses_slices(TEXTURE_VIEW_MTL_TYPE_2D));
        assert!(texture_view_type_is_3d(TEXTURE_VIEW_MTL_TYPE_3D));
    }

    #[test]
    fn decode_copy_slice_level_0x13e() {
        use crate::runtime::decode::blit::{self, OP_COPY_TEXTURE_TO_TEXTURE_SLICE_LEVEL};
        let mut v = vec![0u8; 0x1c];
        st32(&mut v[0..], OP_COPY_TEXTURE_TO_TEXTURE_SLICE_LEVEL);
        st32(&mut v[4..], 0x1c);
        st32(&mut v[8..], 2);
        st32(&mut v[12..], 3);
        st16(&mut v[0x10..], 1);
        st16(&mut v[0x12..], 0);
        st16(&mut v[0x14..], 0);
        st16(&mut v[0x16..], 1);
        st16(&mut v[0x18..], 2);
        st16(&mut v[0x1a..], 3);
        let c = blit::decode(&v).unwrap();
        assert_eq!(c.copy_kind, CopyKind::TextureToTextureSliceLevel);
        assert_eq!(c.source, 2);
        assert_eq!(c.destination, 3);
        assert_eq!(c.source_slice, 1);
        assert_eq!(c.destination_level, 1);
        assert_eq!(c.slice_count, 2);
        assert_eq!(c.level_count, 3);
    }

    #[test]
    fn slice_level_zero_counts_are_noop() {
        let mut host = FakeHost::new();
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        setup_task_pages(&mut host, &mut state, 4);
        let mut cmd = Command::default();
        cmd.kind = Kind::Copy;
        cmd.copy_kind = CopyKind::TextureToTextureSliceLevel;
        cmd.source = 1;
        cmd.destination = 2;
        cmd.slice_count = 0;
        cmd.level_count = 1;
        assert_eq!(
            execute_blit(&mut state, &mut host, 1, &cmd),
            BlitStatus::ZeroExtent
        );
        cmd.slice_count = 1;
        cmd.level_count = 0;
        assert_eq!(
            execute_blit(&mut state, &mut host, 1, &cmd),
            BlitStatus::ZeroExtent
        );
    }

    /// Install a simple type-2 RGBA8 texture (single level, handle → GVA).
    fn install_linear_rgba(
        host: &mut FakeHost,
        state: &mut DeviceState,
        obj_ref: u32,
        handle: u32,
        width: u32,
        height: u32,
        row_stride: u32,
    ) {
        use crate::runtime::decode::resource::{
            list_object_entry_offset, OBJECT_LIST_ENTRY_LEN, OBJECT_TYPE_TEXTURE,
            RESOURCE_PAGE_SHIFT, TEXTURE_DESC_BASE_LEN, TEXTURE_DESC_DATA_OFFSET,
            TEXTURE_DESC_HEIGHT, TEXTURE_DESC_MIPMAP_LEVEL_COUNT, TEXTURE_DESC_PIXEL_FORMAT,
            TEXTURE_DESC_ROW_STRIDE, TEXTURE_DESC_USED_SIZE, TEXTURE_DESC_WIDTH,
        };
        let _ = RESOURCE_PAGE_SHIFT;
        let mut desc = vec![0u8; TEXTURE_DESC_BASE_LEN];
        let size = (row_stride as u64) * (height as u64);
        st64(&mut desc[0..], size.max(0x1000));
        st32(&mut desc[8..], handle);
        st16(&mut desc[TEXTURE_DESC_MIPMAP_LEVEL_COUNT..], 1);
        st32(&mut desc[TEXTURE_DESC_DATA_OFFSET..], 0);
        st32(&mut desc[TEXTURE_DESC_USED_SIZE..], size as u32);
        st32(&mut desc[TEXTURE_DESC_ROW_STRIDE..], row_stride);
        st32(&mut desc[TEXTURE_DESC_WIDTH..], width);
        st32(&mut desc[TEXTURE_DESC_HEIGHT..], height);
        st16(
            &mut desc[TEXTURE_DESC_PIXEL_FORMAT..],
            MTL_FORMAT_RGBA8_UNORM,
        );
        let desc_gva = 0x200u64 + (obj_ref as u64) * 0x80;
        assert!(
            gva_mem::write_task_gva(host, &state.tasks[1], desc_gva, &desc, PAGE_SHIFT_ARM64E)
                .is_ok()
        );
        assert!(state.set_object_list(1, 0, 16));
        let off = list_object_entry_offset(obj_ref, 16).unwrap();
        let mut list_entry = [0u8; OBJECT_LIST_ENTRY_LEN];
        let packed = (OBJECT_TYPE_TEXTURE as u32) | ((TEXTURE_DESC_BASE_LEN as u32) << 8);
        st32(&mut list_entry[0..], packed);
        list_entry[4..12].copy_from_slice(&desc_gva.to_le_bytes());
        assert!(gva_mem::write_task_gva(
            host,
            &state.tasks[1],
            off,
            &list_entry,
            PAGE_SHIFT_ARM64E
        )
        .is_ok());
    }

    #[test]
    fn whole_surface_0x13e_single_level_copy() {
        let mut host = FakeHost::new();
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        setup_task_pages(&mut host, &mut state, 4);
        // src handle=2 (4×2 RGBA, stride 16), dst handle=3
        install_linear_rgba(&mut host, &mut state, 2, 2, 4, 2, 16);
        install_linear_rgba(&mut host, &mut state, 3, 3, 4, 2, 16);
        let src_gva = 2u64 << RESOURCE_PAGE_SHIFT;
        let dst_gva = 3u64 << RESOURCE_PAGE_SHIFT;
        // Fill source with pattern (2 rows × 16 B).
        let mut pat = vec![0u8; 32];
        for (i, b) in pat.iter_mut().enumerate() {
            *b = i as u8;
        }
        assert!(gva_mem::write_task_gva(
            &mut host,
            &state.tasks[1],
            src_gva,
            &pat,
            PAGE_SHIFT_ARM64E
        )
        .is_ok());

        let mut cmd = Command::default();
        cmd.kind = Kind::Copy;
        cmd.copy_kind = CopyKind::TextureToTextureSliceLevel;
        cmd.source = 2;
        cmd.destination = 3;
        cmd.source_slice = 0;
        cmd.source_level = 0;
        cmd.destination_slice = 0;
        cmd.destination_level = 0;
        cmd.slice_count = 1;
        cmd.level_count = 1;
        assert_eq!(execute_blit(&mut state, &mut host, 1, &cmd), BlitStatus::Ok);

        let mut back = vec![0u8; 32];
        assert!(gva_mem::read_task_gva(
            &host,
            &state.tasks[1],
            dst_gva,
            &mut back,
            PAGE_SHIFT_ARM64E
        )
        .is_ok());
        // Only tight 4×4=16 B per row are defined; padding in stride may be zero.
        assert_eq!(&back[0..16], &pat[0..16]);
        assert_eq!(&back[16..32], &pat[16..32]);
    }

    /// Regression guard for the identity self-copy no-op: the guest issues
    /// copyFromTexture:X toTexture:X with matching origin (observed live on
    /// Ventura x86 media apps: src_ref==dst_ref, src_off==dst_off). This copies
    /// bytes onto themselves — a no-op — and must return Ok with content
    /// unchanged, NOT reject it as Overlap (which returned a spurious error to
    /// the guest and dropped a copy it treats as complete).
    #[test]
    fn t2t_identity_self_copy_is_noop_ok() {
        let mut host = FakeHost::new();
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        setup_task_pages(&mut host, &mut state, 4);
        // One 4×2 RGBA texture (stride 16), ref==handle==2.
        install_linear_rgba(&mut host, &mut state, 2, 2, 4, 2, 16);
        let gva = 2u64 << RESOURCE_PAGE_SHIFT;
        let mut pat = vec![0u8; 32];
        for (i, b) in pat.iter_mut().enumerate() {
            *b = (0xA0 + i) as u8;
        }
        assert!(
            gva_mem::write_task_gva(&mut host, &state.tasks[1], gva, &pat, PAGE_SHIFT_ARM64E)
                .is_ok()
        );

        let mut cmd = Command::default();
        cmd.kind = Kind::Copy;
        cmd.copy_kind = CopyKind::TextureToTexture;
        cmd.source = 2;
        cmd.destination = 2; // same texture, same origin => identity
        cmd.source_size.width = 4;
        cmd.source_size.height = 2;
        cmd.source_size.depth = 1;
        assert_eq!(
            execute_blit(&mut state, &mut host, 1, &cmd),
            BlitStatus::Ok,
            "identity self-copy must succeed as a no-op, not Overlap"
        );
        // Content byte-identical: the no-op touched nothing.
        let mut back = vec![0u8; 32];
        assert!(
            gva_mem::read_task_gva(&host, &state.tasks[1], gva, &mut back, PAGE_SHIFT_ARM64E)
                .is_ok()
        );
        assert_eq!(back, pat, "identity self-copy left the bytes unchanged");
        // No overlap enrichment line was emitted (it's not a genuine overlap).
        assert!(
            note_t2t_overlap(1, 2, 2, 0, 0, 16, 16, 2, 1),
            "identity path must not have consumed the overlap dedup slot"
        );
    }

    /// Regression guard for the strided-column false positive: a self-copy of a
    /// 1-wide column shifted N texels right (src rect x[0,1), dst rect x[2,3))
    /// has strided per-row byte footprints that never collide, so it must
    /// SUCCEED and actually move the bytes — the old byte-span overlap test
    /// collapsed row_stride and dropped it as a phantom Overlap. Uses texel-
    /// rectangle overlap (disjoint on x => no overlap).
    #[test]
    fn t2t_shifted_column_self_copy_moves_bytes() {
        let mut host = FakeHost::new();
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        setup_task_pages(&mut host, &mut state, 4);
        // 4×4 RGBA, stride 16, ref==handle==2.
        install_linear_rgba(&mut host, &mut state, 2, 2, 4, 4, 16);
        let gva = 2u64 << RESOURCE_PAGE_SHIFT;
        // Distinct per-texel marker so a moved column is verifiable.
        let mut pat = vec![0u8; 64];
        for (i, b) in pat.iter_mut().enumerate() {
            *b = (0x10 + i) as u8;
        }
        assert!(
            gva_mem::write_task_gva(&mut host, &state.tasks[1], gva, &pat, PAGE_SHIFT_ARM64E)
                .is_ok()
        );
        // Copy column x=0 (4 rows) to column x=2 within the same texture.
        let mut cmd = Command::default();
        cmd.kind = Kind::Copy;
        cmd.copy_kind = CopyKind::TextureToTexture;
        cmd.source = 2;
        cmd.destination = 2;
        cmd.destination_origin.x = 2;
        cmd.source_size.width = 1;
        cmd.source_size.height = 4;
        cmd.source_size.depth = 1;
        assert_eq!(
            execute_blit(&mut state, &mut host, 1, &cmd),
            BlitStatus::Ok,
            "disjoint shifted column copy must succeed, not phantom-Overlap"
        );
        let mut back = vec![0u8; 64];
        assert!(
            gva_mem::read_task_gva(&host, &state.tasks[1], gva, &mut back, PAGE_SHIFT_ARM64E)
                .is_ok()
        );
        // For each row r: dst column x=2 (bytes [r*16+8, +4)) now equals the
        // src column x=0 (bytes [r*16, +4)) as it was in the original pattern.
        for r in 0..4usize {
            let src_texel = &pat[r * 16..r * 16 + 4];
            let dst_texel = &back[r * 16 + 8..r * 16 + 12];
            assert_eq!(dst_texel, src_texel, "row {r} column x=2 holds moved src");
        }
        // No overlap enrichment (this is not a genuine overlap).
        assert!(
            note_t2t_overlap(9, 2, 2, 0, 8, 4, 16, 4, 1),
            "shifted-column path must not have consumed the overlap dedup slot"
        );
    }

    /// Regression guard: a GENUINELY overlapping self-copy (src rect x[0,2),
    /// dst rect x[1,3) — overlap on x) is undefined in Metal and must still be
    /// rejected as Overlap, with the enrichment line emitted for diagnosis.
    #[test]
    fn t2t_overlapping_self_copy_still_rejected() {
        let mut host = FakeHost::new();
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        setup_task_pages(&mut host, &mut state, 4);
        // Unique ref (4) so the (task,src,dst) enrichment key is globally
        // distinct from the identity/shifted tests' probes.
        install_linear_rgba(&mut host, &mut state, 4, 4, 4, 4, 16);
        let mut cmd = Command::default();
        cmd.kind = Kind::Copy;
        cmd.copy_kind = CopyKind::TextureToTexture;
        cmd.source = 4;
        cmd.destination = 4;
        cmd.destination_origin.x = 1; // src x[0,2), dst x[1,3) overlap at x=1
        cmd.source_size.width = 2;
        cmd.source_size.height = 4;
        cmd.source_size.depth = 1;
        assert_eq!(
            execute_blit(&mut state, &mut host, 1, &cmd),
            BlitStatus::Overlap,
            "genuinely overlapping self-copy must be rejected"
        );
        // The enrichment slot WAS consumed (a real overlap was logged).
        assert!(
            !note_t2t_overlap(1, 4, 4, 0, 4, 8, 16, 4, 1),
            "the reject path must have logged the overlap enrichment once"
        );
    }

    #[test]
    fn whole_surface_0x13e_two_levels() {
        use crate::runtime::decode::resource::{
            list_object_entry_offset, OBJECT_LIST_ENTRY_LEN, OBJECT_TYPE_TEXTURE,
            RESOURCE_PAGE_SHIFT, TEXTURE_DESC_BASE_LEN, TEXTURE_DESC_DATA_OFFSET,
            TEXTURE_DESC_HEIGHT, TEXTURE_DESC_LEVEL_RECORDS, TEXTURE_DESC_MIPMAP_LEVEL_COUNT,
            TEXTURE_DESC_MIP_LEVEL_RECORD_LEN, TEXTURE_DESC_PIXEL_FORMAT, TEXTURE_DESC_ROW_STRIDE,
            TEXTURE_DESC_USED_SIZE, TEXTURE_DESC_WIDTH, TEXTURE_LEVEL_DEPTH, TEXTURE_LEVEL_HEIGHT,
            TEXTURE_LEVEL_OFFSET, TEXTURE_LEVEL_ROW_STRIDE, TEXTURE_LEVEL_SIZE,
            TEXTURE_LEVEL_WIDTH,
        };
        let mut host = FakeHost::new();
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        setup_task_pages(&mut host, &mut state, 4);

        // Two textures, 2 mips: L0 4×2 stride16, L1 2×1 stride8.
        for (obj_ref, handle) in [(2u32, 2u32), (3u32, 3u32)] {
            let body = TEXTURE_DESC_BASE_LEN + TEXTURE_DESC_MIP_LEVEL_RECORD_LEN;
            let mut desc = vec![0u8; body];
            st64(&mut desc[0..], 0x4000);
            st32(&mut desc[8..], handle);
            st16(&mut desc[TEXTURE_DESC_MIPMAP_LEVEL_COUNT..], 2);
            st32(&mut desc[TEXTURE_DESC_DATA_OFFSET..], 0);
            st32(&mut desc[TEXTURE_DESC_USED_SIZE..], 32);
            st32(&mut desc[TEXTURE_DESC_ROW_STRIDE..], 16);
            st32(&mut desc[TEXTURE_DESC_WIDTH..], 4);
            st32(&mut desc[TEXTURE_DESC_HEIGHT..], 2);
            let rec = TEXTURE_DESC_LEVEL_RECORDS;
            st64(&mut desc[rec + TEXTURE_LEVEL_OFFSET..], 32);
            st64(&mut desc[rec + TEXTURE_LEVEL_SIZE..], 8);
            st64(&mut desc[rec + TEXTURE_LEVEL_ROW_STRIDE..], 8);
            st32(&mut desc[rec + TEXTURE_LEVEL_WIDTH..], 2);
            st32(&mut desc[rec + TEXTURE_LEVEL_HEIGHT..], 1);
            st32(&mut desc[rec + TEXTURE_LEVEL_DEPTH..], 1);
            let pf_off = TEXTURE_DESC_PIXEL_FORMAT + TEXTURE_DESC_MIP_LEVEL_RECORD_LEN;
            st16(&mut desc[pf_off..], MTL_FORMAT_RGBA8_UNORM);
            let desc_gva = 0x200u64 + (obj_ref as u64) * 0x100;
            assert!(gva_mem::write_task_gva(
                &mut host,
                &state.tasks[1],
                desc_gva,
                &desc,
                PAGE_SHIFT_ARM64E
            )
            .is_ok());
            assert!(state.set_object_list(1, 0, 16));
            let off = list_object_entry_offset(obj_ref, 16).unwrap();
            let mut list_entry = [0u8; OBJECT_LIST_ENTRY_LEN];
            let packed = (OBJECT_TYPE_TEXTURE as u32) | ((body as u32) << 8);
            st32(&mut list_entry[0..], packed);
            list_entry[4..12].copy_from_slice(&desc_gva.to_le_bytes());
            assert!(gva_mem::write_task_gva(
                &mut host,
                &state.tasks[1],
                off,
                &list_entry,
                PAGE_SHIFT_ARM64E
            )
            .is_ok());
        }

        // Seed L0 and L1 on source handle 2.
        let base = 2u64 << RESOURCE_PAGE_SHIFT;
        let l0 = [1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let l0_row1 = [
            17u8, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
        ];
        let l1 = [0xaau8, 0xbb, 0xcc, 0xdd, 0x11, 0x22, 0x33, 0x44];
        assert!(
            gva_mem::write_task_gva(&mut host, &state.tasks[1], base, &l0, PAGE_SHIFT_ARM64E)
                .is_ok()
        );
        assert!(gva_mem::write_task_gva(
            &mut host,
            &state.tasks[1],
            base + 16,
            &l0_row1,
            PAGE_SHIFT_ARM64E
        )
        .is_ok());
        assert!(gva_mem::write_task_gva(
            &mut host,
            &state.tasks[1],
            base + 32,
            &l1,
            PAGE_SHIFT_ARM64E
        )
        .is_ok());

        let mut cmd = Command::default();
        cmd.kind = Kind::Copy;
        cmd.copy_kind = CopyKind::TextureToTextureSliceLevel;
        cmd.source = 2;
        cmd.destination = 3;
        cmd.slice_count = 1;
        cmd.level_count = 2;
        assert_eq!(execute_blit(&mut state, &mut host, 1, &cmd), BlitStatus::Ok);

        let dst = 3u64 << RESOURCE_PAGE_SHIFT;
        let mut back_l0 = [0u8; 16];
        let mut back_l1 = [0u8; 8];
        assert!(gva_mem::read_task_gva(
            &host,
            &state.tasks[1],
            dst,
            &mut back_l0,
            PAGE_SHIFT_ARM64E
        )
        .is_ok());
        assert!(gva_mem::read_task_gva(
            &host,
            &state.tasks[1],
            dst + 32,
            &mut back_l1,
            PAGE_SHIFT_ARM64E
        )
        .is_ok());
        assert_eq!(back_l0, l0);
        assert_eq!(back_l1, l1);
        let _ = RESOURCE_PAGE_SHIFT;
    }

    /// Install type-2 RGBA8 volume (single level, depth>1) at `handle<<14`.
    fn install_linear_rgba_volume(
        host: &mut FakeHost,
        state: &mut DeviceState,
        obj_ref: u32,
        handle: u32,
        width: u32,
        height: u32,
        depth: u32,
        row_stride: u32,
    ) {
        use crate::runtime::decode::resource::{
            list_object_entry_offset, OBJECT_LIST_ENTRY_LEN, OBJECT_TYPE_TEXTURE,
            RESOURCE_PAGE_SHIFT, TEXTURE_DESC_BASE_LEN, TEXTURE_DESC_DATA_OFFSET,
            TEXTURE_DESC_DEPTH, TEXTURE_DESC_HEIGHT, TEXTURE_DESC_MIPMAP_LEVEL_COUNT,
            TEXTURE_DESC_PIXEL_FORMAT, TEXTURE_DESC_ROW_STRIDE, TEXTURE_DESC_USED_SIZE,
            TEXTURE_DESC_WIDTH,
        };
        let _ = RESOURCE_PAGE_SHIFT;
        let mut desc = vec![0u8; TEXTURE_DESC_BASE_LEN];
        let plane = (row_stride as u64) * (height as u64);
        let size = plane * (depth as u64);
        st64(&mut desc[0..], size.max(0x1000));
        st32(&mut desc[8..], handle);
        st16(&mut desc[TEXTURE_DESC_MIPMAP_LEVEL_COUNT..], 1);
        st32(&mut desc[TEXTURE_DESC_DATA_OFFSET..], 0);
        st32(&mut desc[TEXTURE_DESC_USED_SIZE..], size as u32);
        st32(&mut desc[TEXTURE_DESC_ROW_STRIDE..], row_stride);
        st32(&mut desc[TEXTURE_DESC_WIDTH..], width);
        st32(&mut desc[TEXTURE_DESC_HEIGHT..], height);
        st32(&mut desc[TEXTURE_DESC_DEPTH..], depth);
        st16(
            &mut desc[TEXTURE_DESC_PIXEL_FORMAT..],
            MTL_FORMAT_RGBA8_UNORM,
        );
        let desc_gva = 0x200u64 + (obj_ref as u64) * 0x80;
        assert!(
            gva_mem::write_task_gva(host, &state.tasks[1], desc_gva, &desc, PAGE_SHIFT_ARM64E)
                .is_ok()
        );
        assert!(state.set_object_list(1, 0, 16));
        let off = list_object_entry_offset(obj_ref, 16).unwrap();
        let mut list_entry = [0u8; OBJECT_LIST_ENTRY_LEN];
        let packed = (OBJECT_TYPE_TEXTURE as u32) | ((TEXTURE_DESC_BASE_LEN as u32) << 8);
        st32(&mut list_entry[0..], packed);
        list_entry[4..12].copy_from_slice(&desc_gva.to_le_bytes());
        assert!(gva_mem::write_task_gva(
            host,
            &state.tasks[1],
            off,
            &list_entry,
            PAGE_SHIFT_ARM64E
        )
        .is_ok());
    }

    #[test]
    fn whole_surface_0x13e_volume_depth_planes() {
        let mut host = FakeHost::new();
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        setup_task_pages(&mut host, &mut state, 4);
        // 2×2×3 RGBA8, row_stride 8 → plane 16 B, volume 48 B.
        install_linear_rgba_volume(&mut host, &mut state, 2, 2, 2, 2, 3, 8);
        install_linear_rgba_volume(&mut host, &mut state, 3, 3, 2, 2, 3, 8);
        let src_gva = 2u64 << RESOURCE_PAGE_SHIFT;
        let dst_gva = 3u64 << RESOURCE_PAGE_SHIFT;
        let mut vol = vec![0u8; 48];
        for (i, b) in vol.iter_mut().enumerate() {
            *b = (i as u8).wrapping_add(1);
        }
        assert!(gva_mem::write_task_gva(
            &mut host,
            &state.tasks[1],
            src_gva,
            &vol,
            PAGE_SHIFT_ARM64E
        )
        .is_ok());

        let mut cmd = Command::default();
        cmd.kind = Kind::Copy;
        cmd.copy_kind = CopyKind::TextureToTextureSliceLevel;
        cmd.source = 2;
        cmd.destination = 3;
        cmd.source_slice = 0;
        cmd.destination_slice = 0;
        cmd.slice_count = 1;
        cmd.level_count = 1;
        assert_eq!(execute_blit(&mut state, &mut host, 1, &cmd), BlitStatus::Ok);

        let mut back = vec![0u8; 48];
        assert!(gva_mem::read_task_gva(
            &host,
            &state.tasks[1],
            dst_gva,
            &mut back,
            PAGE_SHIFT_ARM64E
        )
        .is_ok());
        assert_eq!(back, vol);
    }

    #[test]
    fn whole_surface_0x13e_volume_rejects_multi_slice() {
        let mut host = FakeHost::new();
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        setup_task_pages(&mut host, &mut state, 4);
        install_linear_rgba_volume(&mut host, &mut state, 2, 2, 2, 2, 2, 8);
        install_linear_rgba_volume(&mut host, &mut state, 3, 3, 2, 2, 2, 8);
        let mut cmd = Command::default();
        cmd.kind = Kind::Copy;
        cmd.copy_kind = CopyKind::TextureToTextureSliceLevel;
        cmd.source = 2;
        cmd.destination = 3;
        cmd.slice_count = 2; // Metal: 3D requires sliceCount==1
        cmd.level_count = 1;
        assert_eq!(
            execute_blit(&mut state, &mut host, 1, &cmd),
            BlitStatus::Unsupported
        );
    }

    #[test]
    fn whole_surface_0x13e_volume_rejects_nonzero_slice() {
        let mut host = FakeHost::new();
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        setup_task_pages(&mut host, &mut state, 4);
        install_linear_rgba_volume(&mut host, &mut state, 2, 2, 2, 2, 2, 8);
        install_linear_rgba_volume(&mut host, &mut state, 3, 3, 2, 2, 2, 8);
        let mut cmd = Command::default();
        cmd.kind = Kind::Copy;
        cmd.copy_kind = CopyKind::TextureToTextureSliceLevel;
        cmd.source = 2;
        cmd.destination = 3;
        // Non-zero slice on 3D whole-surface is fail-closed (Metal forbids).
        // Status may be Bounds (slice packing) or Unsupported (3D rule).
        cmd.source_slice = 1;
        cmd.slice_count = 1;
        cmd.level_count = 1;
        let st = execute_blit(&mut state, &mut host, 1, &cmd);
        assert!(
            matches!(st, BlitStatus::Bounds | BlitStatus::Unsupported),
            "expected Bounds or Unsupported, got {st:?}"
        );
    }

    #[test]
    fn blit_fence_update_then_wait() {
        use crate::model::FENCE_DOMAIN_BLIT;
        use crate::runtime::decode::blit::{OP_UPDATE_FENCE, OP_WAIT_FENCE};
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let mut upd = Command::default();
        upd.kind = Kind::Fence;
        upd.opcode = OP_UPDATE_FENCE;
        upd.fence = 7;
        assert_eq!(execute_blit_fence(&mut state, 1, &upd), BlitStatus::Ok);
        assert_eq!(state.fence_generation(1, FENCE_DOMAIN_BLIT, 7), Some(1));
        // Second update advances generation.
        assert_eq!(execute_blit_fence(&mut state, 1, &upd), BlitStatus::Ok);
        assert_eq!(state.fence_generation(1, FENCE_DOMAIN_BLIT, 7), Some(2));
        let mut wait = Command::default();
        wait.kind = Kind::Fence;
        wait.opcode = OP_WAIT_FENCE;
        wait.fence = 7;
        assert_eq!(execute_blit_fence(&mut state, 1, &wait), BlitStatus::Ok);
    }

    #[test]
    fn blit_fence_wait_pending_without_update() {
        use crate::runtime::decode::blit::OP_WAIT_FENCE;
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let mut wait = Command::default();
        wait.kind = Kind::Fence;
        wait.opcode = OP_WAIT_FENCE;
        wait.fence = 3;
        assert_eq!(
            execute_blit_fence(&mut state, 1, &wait),
            BlitStatus::FencePending
        );
        assert!(state.fence_generation(1, FENCE_DOMAIN_BLIT, 3).is_none());
    }

    #[test]
    fn blit_fence_zero_ref_fails() {
        use crate::runtime::decode::blit::OP_UPDATE_FENCE;
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let mut upd = Command::default();
        upd.kind = Kind::Fence;
        upd.opcode = OP_UPDATE_FENCE;
        upd.fence = 0;
        assert_eq!(
            execute_blit_fence(&mut state, 1, &upd),
            BlitStatus::MissingResource
        );
    }

    /// Regression guard for the pure blit-geometry helpers `clamp_extent`,
    /// `texture_storage_bpp`, and the aspect routing of `copy_aspect_for_options`.
    /// These feed copy bounds and row strides, so a silent break corrupts the
    /// copied region: a bad clamp reads/writes out of bounds, a wrong bpp
    /// miscomputes the row stride, and a wrong aspect flag copies the wrong
    /// depth/stencil plane. `copy_bpp_for_options` tests already cover the bpp
    /// result; this locks the extent/stride/aspect-flag contract directly.
    #[test]
    fn blit_geometry_helpers_clamp_bpp_and_aspect() {
        use crate::contract::pixel_format::{
            MTL_FORMAT_A8_UNORM, MTL_FORMAT_DEPTH32_FLOAT_STENCIL8, MTL_FORMAT_RGBA16_FLOAT,
        };
        use crate::runtime::decode::blit::{
            MTL_BLIT_OPTION_DEPTH_FROM_DEPTH_STENCIL, MTL_BLIT_OPTION_NONE,
            MTL_BLIT_OPTION_STENCIL_FROM_DEPTH_STENCIL,
        };

        // clamp_extent: zero stays a no-op extent; over-max clamps to max;
        // in-range and exactly-max pass through unchanged.
        assert_eq!(clamp_extent(0, 100), 0, "zero is a Metal no-op extent");
        assert_eq!(clamp_extent(50, 100), 50, "in-range passes through");
        assert_eq!(clamp_extent(150, 100), 100, "over-max clamps to max");
        assert_eq!(clamp_extent(100, 100), 100, "exactly max passes through");

        // texture_storage_bpp: full-texel storage size per format; unknown fails.
        assert_eq!(texture_storage_bpp(MTL_FORMAT_BGRA8_UNORM), Ok(4));
        assert_eq!(texture_storage_bpp(MTL_FORMAT_A8_UNORM), Ok(1));
        assert_eq!(texture_storage_bpp(MTL_FORMAT_RGBA16_FLOAT), Ok(8));
        assert_eq!(
            texture_storage_bpp(0xFFFF),
            Err(BlitStatus::Unsupported),
            "unknown format must fail visibly, not invent a stride",
        );

        // copy_aspect_for_options: option bit -> (want_depth, want_stencil, bpp).
        let with_opts = |opts: u32| {
            let mut cmd = Command::default();
            cmd.has_options = true;
            cmd.options = opts;
            cmd
        };
        // No option on a color format -> full aspect, no plane routing.
        assert_eq!(
            copy_aspect_for_options(MTL_FORMAT_BGRA8_UNORM, &with_opts(MTL_BLIT_OPTION_NONE)),
            Ok((false, false, 4)),
        );
        // Depth option on a depth-stencil format -> depth plane (4 B), no stencil.
        assert_eq!(
            copy_aspect_for_options(
                MTL_FORMAT_DEPTH32_FLOAT_STENCIL8,
                &with_opts(MTL_BLIT_OPTION_DEPTH_FROM_DEPTH_STENCIL),
            ),
            Ok((true, false, 4)),
        );
        // Stencil option -> stencil plane (1 B), no depth.
        assert_eq!(
            copy_aspect_for_options(
                MTL_FORMAT_DEPTH32_FLOAT_STENCIL8,
                &with_opts(MTL_BLIT_OPTION_STENCIL_FROM_DEPTH_STENCIL),
            ),
            Ok((false, true, 1)),
        );
        // Unknown option bit -> visible failure (no invented aspect).
        assert_eq!(
            copy_aspect_for_options(MTL_FORMAT_DEPTH32_FLOAT_STENCIL8, &with_opts(1 << 8)),
            Err(BlitStatus::Unsupported),
        );
    }

    /// Regression guard: the `tex_wrong_type` enrichment is deduped per
    /// `(task, ref, object_type)` — a per-draw non-texture bind must not flood
    /// the always-on sink — while distinct refs/types each report once so a
    /// buffer-bound-as-texture (decode bug) stays diagnosable.
    #[test]
    fn tex_wrong_type_enrichment_dedups_per_ref_and_type() {
        reset_tex_wrong_type_dedup_for_test();
        // First sighting of a (task, ref, type) emits; repeats are deduped.
        assert!(note_tex_wrong_type(7, 0x40, OBJECT_TYPE_BUFFER, 0, 0));
        for _ in 0..20 {
            assert!(!note_tex_wrong_type(7, 0x40, OBJECT_TYPE_BUFFER, 0, 0));
        }
        // A different ref is a distinct failure -> reports once.
        assert!(note_tex_wrong_type(7, 0x41, OBJECT_TYPE_BUFFER, 0, 0));
        // Same ref but a different actual object_type also reports (the type is
        // the diagnostic field, so a type change must not be masked).
        assert!(note_tex_wrong_type(
            7,
            0x40,
            crate::runtime::decode::resource::OBJECT_TYPE_FUNCTION,
            0,
            0
        ));
        assert!(!note_tex_wrong_type(
            7,
            0x40,
            crate::runtime::decode::resource::OBJECT_TYPE_FUNCTION,
            0,
            0
        ));
    }

    /// Regression guard: the `t2t_overlap` enrichment dedups per
    /// `(task, src_ref, dst_ref)` — a self-overlapping copy re-issued every
    /// frame must not flood — while a distinct src/dst pair reports once so a
    /// genuine drop stays diagnosable.
    #[test]
    fn t2t_overlap_enrichment_dedups_per_pair() {
        // Unique task namespace (3) so the process-global dedup set never
        // collides with other tests; the set starts empty so first-insert is
        // deterministic without a reset.
        assert!(note_t2t_overlap(3, 0x10, 0x10, 0, 4096, 256, 1024, 8, 1));
        for _ in 0..20 {
            assert!(!note_t2t_overlap(3, 0x10, 0x10, 0, 4096, 256, 1024, 8, 1));
        }
        // A distinct destination ref is a distinct failure -> reports once.
        assert!(note_t2t_overlap(3, 0x10, 0x11, 0, 4096, 256, 1024, 8, 1));
    }

    /// Regression guard: the `copy_region_io` enrichment dedups per
    /// `(task, gva_page, is_write)` — a strided multi-row failure into one page
    /// must not flood — while read vs write and distinct pages each report once.
    #[test]
    fn copy_region_io_enrichment_dedups_per_page_and_direction() {
        // Unique task namespace (2) + page base so the process-global dedup set
        // never collides with other tests; empty-set start makes first-insert
        // deterministic without a reset.
        let shift = PAGE_SHIFT_ARM64E;
        let page = 0x5000u64 << shift;
        // Rows 0..N inside the same destination page collapse to one line.
        assert!(note_copy_region_io(2, true, page, 0, 0, 256, shift));
        for y in 1..10u64 {
            assert!(!note_copy_region_io(
                2,
                true,
                page + y * 256,
                y,
                0,
                256,
                shift
            ));
        }
        // A read at the same page is a distinct direction -> reports once.
        assert!(note_copy_region_io(2, false, page, 0, 0, 256, shift));
        // A different page reports once.
        assert!(note_copy_region_io(
            2,
            true,
            page + (1u64 << shift),
            0,
            0,
            256,
            shift
        ));
    }
}
