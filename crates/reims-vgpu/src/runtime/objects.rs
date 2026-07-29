//! Object-list lookup, type-11 registration, and x86 type-4 surface backing.
//!
//! Live layout (reims-vgpu-resource-format): entry `ref` is at
//! `(object_list_pfn << PAGE_SHIFT) + ref * 12` in the task GVA space —
//! `[type|desc_len packed u32][desc_gva u64]`.
//!
//! **x86 type-4 present path (Ventura 13.7 RE):**
//! `AppleParavirtResource::allocateBackingHandle` calls
//! `ResourceHeap::addObject(type=4, objectId=IOSurface::getSurfaceID(), …)` so
//! the object-list index for a surface-backed resource **is** the present
//! `surface_id`. Descriptor layout:
//! length@0, backing_pfn@8, format@0xc, plane_count@0x10, planes@0x14.

use crate::contract::endian::{ld32, ld64, st16, st32, st64};
use crate::contract::iosurface_pages::{
    entry_gpa_shift, page_size_of, DEVICE_DESC_ALLOC_SIZE, DEVICE_DESC_BPE, DEVICE_DESC_BPR,
    DEVICE_DESC_DIMS, DEVICE_DESC_LEN, DEVICE_DESC_PIXEL_FORMAT, DEVICE_DESC_PLANES,
    DEVICE_DESC_PLANE_COUNT, DEVICE_PLANE_BPE, DEVICE_PLANE_BPR, DEVICE_PLANE_DESC_LEN,
    DEVICE_PLANE_DIMS, DEVICE_PLANE_OFFSET, DEVICE_PLANE_SIZE, PAGE_ENTRY_PFN_SHIFT,
    PAGE_ENTRY_VALID,
};
use crate::model::{DeviceState, MappingEntry, ObjectEntry, MAX_MAPPINGS, MAX_TASKS};
use crate::runtime::decode::resource::{
    decode_list_object_entry, list_object_entry_offset, ListObjectEntry, OBJECT_LIST_ENTRY_LEN,
    OBJECT_TYPE_IOSURFACE,
};
use crate::runtime::gva_mem;
use crate::runtime::host::HostMemory;
use crate::runtime::texture;

/// Fail-visible, de-duplicated per `(task_id, ref)`, for the type-11 resolve
/// blind spot: an object ref that IS a type-11 IOSurface texture but whose
/// descriptor cannot be read, cannot register a Metal/Vulkan texture, or carries
/// `mapping_id==0` used to collapse into a bare `None` → a coarse
/// `MissingTexture` at the draw site with no reason. `resolve_type11_ref` runs
/// per-draw per-ref (very hot), so a bare fail line would flood; the latch logs
/// each `(task,ref,reason)` once and is cleared when the ref resolves
/// ([`clear_type11_fail`]). Only genuine failures for a *confirmed IOSurface*
/// ref are routed here — the legitimate "ref is a different object type" and
/// unbound-slot returns stay silent. Runs on the drain worker (off the QEMU main
/// core).
type Type11Failure = (u32, u32, &'static str);
type Type11FailureSet = std::collections::HashSet<Type11Failure>;

fn type11_fail_latch() -> &'static std::sync::Mutex<Type11FailureSet> {
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<Type11FailureSet>> = OnceLock::new();
    SEEN.get_or_init(|| Mutex::new(Type11FailureSet::new()))
}

fn note_type11_fail(task_id: u32, ref_: u32, reason: &'static str, detail: String) {
    let mut guard = type11_fail_latch()
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    if guard.insert((task_id, ref_, reason)) {
        crate::observe::fail(detail);
    }
}

/// Re-arm the fail latch for a ref that just resolved, so a later genuine
/// failure on the same ref is logged again (catches flapping).
fn clear_type11_fail(task_id: u32, ref_: u32) {
    let mut guard = type11_fail_latch()
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    guard.retain(|(t, r, _)| !(*t == task_id && *r == ref_));
}

/// Fail-visible, de-duplicated per `(surface_id, reason)`, for the type-4
/// backing blind spot: a surface whose object-list descriptor decoded fine (an
/// active task, a valid `Type4Surface`) but whose page-backing construction then
/// failed — every downstream present/Store for that surface paints **stale or
/// black** with no reason. `apply_type4_backing` is reached from the per-present
/// scanout path (`ensure_surface_for_present`, ~48/s under scroll), so a persistent
/// backing failure would flood; the latch logs each `(surface_id, reason)` once
/// and re-arms when the surface next resolves cleanly ([`clear_type4_fail`]), so a
/// flapping backing is re-logged. Only genuine type-4 candidate failures are
/// routed here — the caller's speculative per-task `continue`s (surface absent
/// from this task or a non-surface object type) stay silent. Runs on the drain
/// worker (off the QEMU main core).
fn type4_fail_latch() -> &'static std::sync::Mutex<std::collections::HashSet<(u32, &'static str)>> {
    use std::collections::HashSet;
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<HashSet<(u32, &'static str)>>> = OnceLock::new();
    SEEN.get_or_init(|| Mutex::new(HashSet::new()))
}

fn note_type4_fail(surface_id: u32, reason: &'static str, detail: String) {
    let mut guard = type4_fail_latch().lock().unwrap_or_else(|e| e.into_inner());
    if guard.insert((surface_id, reason)) {
        crate::observe::fail(detail);
    }
}

/// Re-arm the type-4 fail latch for a surface that just backed cleanly, so a
/// later genuine backing failure on the same surface is logged again.
fn clear_type4_fail(surface_id: u32) {
    let mut guard = type4_fail_latch().lock().unwrap_or_else(|e| e.into_inner());
    guard.retain(|(s, _)| *s != surface_id);
}

/// Wire object type for surface / IOSurface backing (x86 Tahoe/Ventura).
pub const OBJECT_TYPE_SURFACE: u8 = 4;
/// RefTextureHandle: surfaceID@0 + cookie@4 + guest blob@8 (texture-ref 28-06-26).
pub const OBJECT_TYPE_REF_TEXTURE: u8 = 5;
/// Type-5 RefTexture descriptor (RE `allocateRefTextureHandle` + Metal
/// `initWithDevice:descriptor:iosurface:plane:field:`):
/// - `surfaceID@0` = `IOSurface::getSurfaceID()` = type-4 heap object id / mid
/// - `field@4` = device-side field dword (not plane index)
/// - `args@8..` = **serialized texture args** length `desc_len-8` (MTLTextureDescriptor
///   stream for the **plane** view; plane is applied guest-side before serialize)
///
/// See [[reims-vgpu-resource-paging]] type-5 section.
pub const TYPE5_SURFACE_ID: usize = 0x00;
pub const TYPE5_FIELD: usize = 0x04;
pub const TYPE5_ARGS: usize = 0x08;
pub const TYPE5_MIN_LEN: usize = 0x08;

/// Type-5 args blob layout (live wire census 2026-07-14, `compute_stage_tex
/// type5 … args_hex`; 48-byte blob on Ventura 13.7.8 x86):
/// - `+0` u32 kind tag (`0x2f` observed)
/// - `+4` u32 blob length (== `desc_len - TYPE5_ARGS`)
/// - `+8` u32 the type-5 object's **own ref** (same convention as the type-11
///   texture descriptor's object-ref field)
/// - `+12` serialized **plane texture record** — the guest-side
///   `newTextureWithDescriptor:iosurface:plane:` view (plane already applied
///   before serialization; see the `TYPE5_ARGS` doc above):
///   `[+0 u8 tag=0x42][+1 u8 unknown][+2 u16 MTLPixelFormat][+4 u32 width]`
///   `[+8 u32 height][+12 u32 depth][+0x10 trailer][+0x20 u32 IOSurface plane]`
///   Live: `R8 1024×1024 depth=1` = Y plane of a `'420f'` 1024×1024 surface;
///   `BGRA8 68×58`, `RGBA32Uint 482×1928` (uint4 view of a BGRA 1928×1928
///   surface — byte-identical rows) also observed.
///   The **plane index at record `+0x20`** is the
///   `newTextureWithDescriptor:iosurface:plane:` plane argument — live v0a8
///   3-plane blob census (boot 20260717-063043, 10 mappings): Y blobs carry 0,
///   the RG8 chroma blob carries 1, and the second R8 view of identical
///   geometry carries 2 (the alpha plane). Geometry cannot disambiguate Y from
///   alpha; this field is the only wire key. (The type-11 texture descriptor
///   carries no such field — that finding is unchanged.)
pub const TYPE5_ARG_KIND: usize = TYPE5_ARGS;
pub const TYPE5_ARG_BLOB_LEN: usize = TYPE5_ARGS + 0x04;
pub const TYPE5_ARG_OWN_REF: usize = TYPE5_ARGS + 0x08;
pub const TYPE5_ARG_RECORD: usize = TYPE5_ARGS + 0x0c;
pub const TYPE5_RECORD_TAG: u8 = 0x42;
/// Sibling record tag observed live on the blit copy-source path (x86 Ventura
/// 13.7.8, 2026-07-19 six-app launch): full-color texture views (BGRA8_sRGB
/// 1024×768 window backings) carry tag `0x62` where biplanar plane views carry
/// `0x42`. The record layout (format@+2, width@+4, height@+8, depth@+0xc) is
/// byte-identical — the tag distinguishes a variant, not a different geometry
/// encoding — so both decode through the same field offsets.
pub const TYPE5_RECORD_TAG_COLOR_VIEW: u8 = 0x62;
pub const TYPE5_RECORD_FORMAT: usize = 0x02;
pub const TYPE5_RECORD_WIDTH: usize = 0x04;
pub const TYPE5_RECORD_HEIGHT: usize = 0x08;
pub const TYPE5_RECORD_DEPTH: usize = 0x0c;
pub const TYPE5_RECORD_PLANE: usize = 0x20;
pub const TYPE5_RECORD_MIN_LEN: usize = 0x10;

/// Texture view named by a type-5 descriptor's serialized args record.
///
/// This is not limited to IOSurface planes. The live desktop also uses
/// row-byte-equivalent reinterpretations such as a 480-wide RGBA32Uint view
/// over a 1920-wide BGRA8 surface.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Type5TextureView {
    pub pixel_format: u16,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    /// IOSurface plane the serialized view binds (record `+0x20`); 0 when the
    /// record is too short to carry the field (pre-plane blobs, tests).
    pub plane_index: u32,
}

/// Decode the serialized texture-view record from a full type-5 descriptor.
///
/// Fail-closed: `None` unless the record tag matches and geometry is sane
/// (2D, nonzero). The record names the exact Metal view (format + geometry)
/// over the IOSurface bytes; callers must not replace it with base mapping
/// geometry merely because the surface itself is otherwise stageable.
pub fn decode_type5_texture_view(desc: &[u8]) -> Option<Type5TextureView> {
    if desc.len() < TYPE5_ARG_RECORD + TYPE5_RECORD_MIN_LEN {
        return None;
    }
    let rec = &desc[TYPE5_ARG_RECORD..];
    // Accept both the biplanar-plane record tag (0x42) and the full-color
    // texture-view variant (0x62); both share the field layout below. Any other
    // tag stays unknown → fail closed (no invented geometry).
    if rec[0] != TYPE5_RECORD_TAG && rec[0] != TYPE5_RECORD_TAG_COLOR_VIEW {
        return None;
    }
    let pixel_format = u16::from_le_bytes([rec[TYPE5_RECORD_FORMAT], rec[TYPE5_RECORD_FORMAT + 1]]);
    let width = ld32(&rec[TYPE5_RECORD_WIDTH..]);
    let height = ld32(&rec[TYPE5_RECORD_HEIGHT..]);
    let depth = ld32(&rec[TYPE5_RECORD_DEPTH..]);
    if pixel_format == 0 || width == 0 || height == 0 || depth != 1 {
        return None;
    }
    let plane_index = if rec.len() >= TYPE5_RECORD_PLANE + 4 {
        ld32(&rec[TYPE5_RECORD_PLANE..])
    } else {
        0
    };
    Some(Type5TextureView {
        pixel_format,
        width,
        height,
        depth,
        plane_index,
    })
}

/// Type-4 descriptor field offsets (RE allocateBackingHandle / tahoe §9.4).
pub const TYPE4_LEN: usize = 0x00;
pub const TYPE4_BACKING_PFN: usize = 0x08;
pub const TYPE4_PIXEL_FORMAT: usize = 0x0c;
pub const TYPE4_PLANE_COUNT: usize = 0x10;
pub const TYPE4_PLANES: usize = 0x14;
pub const TYPE4_PLANE_STRIDE: usize = 0x10;
/// Plane0 fields relative to descriptor base (after off-by-one fix in kb).
pub const TYPE4_PLANE0_WIDTH: usize = 0x18;
pub const TYPE4_PLANE0_HEIGHT: usize = 0x1c;
pub const TYPE4_PLANE0_BPR_PACKED: usize = 0x20;
pub const TYPE4_MIN_LEN: usize = 0x24;
/// Max plane records in type-4 wire / device desc (IOSurface getPlaneCount cap).
pub const TYPE4_PLANE_CAP: usize = 8;

/// CoreVideo / IOSurface biplanar 420 full-range (`'420f'`).
pub const IOSURFACE_FOURCC_420F: u32 = 0x3432_3066;
/// CoreVideo / IOSurface biplanar 420 video-range (`'420v'`).
pub const IOSURFACE_FOURCC_420V: u32 = 0x3432_3076;

/// One type-4 plane record (stride 0x10 @ +0x14): offset, w, h, bpr|bpe<<24.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Type4Plane {
    pub offset: u32,
    pub width: u32,
    pub height: u32,
    pub bytes_per_row: u32,
    /// From packed high 8 bits (`getPlaneBytesPerElement`); 0 if wire left it 0.
    pub bytes_per_element: u8,
}

/// Decoded type-4 surface backing descriptor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Type4Surface {
    pub length: u64,
    pub backing_pfn: u32,
    /// Wire `pixelFormat@0xc` — OSType FourCC or small MTL ordinal.
    pub pixel_format: u32,
    pub plane_count: u8,
    pub planes: [Type4Plane; TYPE4_PLANE_CAP],
    /// Plane0 convenience (present / single-plane geom).
    pub width: u32,
    pub height: u32,
    pub bytes_per_row: u32,
}

/// CoreVideo biplanar 8-bit 420 family — **not** a single `MTLPixelFormat`.
///
/// Metal binds planes via `newTextureWithDescriptor:iosurface:plane:` as R8 (Y)
/// and RG8 (UV). Product must not invent BGRA.
#[inline]
pub fn iosurface_fourcc_is_biplanar(pixel_format: u32) -> bool {
    matches!(pixel_format, IOSURFACE_FOURCC_420F | IOSURFACE_FOURCC_420V)
}

/// True when type-4 / mapping cannot be staged as one linear color texture.
#[inline]
pub fn type4_is_multiplanar(surf: &Type4Surface) -> bool {
    surf.plane_count > 1 || iosurface_fourcc_is_biplanar(surf.pixel_format)
}

/// Mapping has multi-plane device geometry (plane_count≥2) or biplanar FourCC.
pub fn mapping_is_multiplanar(m: &MappingEntry) -> bool {
    use crate::contract::iosurface_pages::decode_device_surface;
    if let Some(s) = decode_device_surface(&m.device_desc) {
        if s.plane_count > 1 {
            return true;
        }
        if iosurface_fourcc_is_biplanar(s.pixel_format) {
            return true;
        }
    }
    false
}

/// Map IOSurface OSType FourCC (or MTL raw) to a **single-plane** MTL pixel format.
///
/// Live x86 type-4 carries IOSurface `pixelFormat` as a FourCC (e.g. `'BGRA'` =
/// `0x42475241`). Truncating to u16 yields `0x5241` which is not a Metal format.
///
/// Returns **0** when:
/// - format is multi-plane (e.g. `'420f'` / `'420v'`) — no single MTLPixelFormat
/// - format is unknown — fail closed; **do not** invent BGRA8
///
/// Unknown formats fail closed.
pub fn iosurface_pixel_format_to_mtl(pixel_format: u32) -> u16 {
    use crate::contract::pixel_format::{
        MTL_FORMAT_BGRA8_UNORM, MTL_FORMAT_R8_UNORM, MTL_FORMAT_RG8_UNORM, MTL_FORMAT_RGBA16_FLOAT,
        MTL_FORMAT_RGBA8_UNORM,
    };
    if pixel_format == 0 {
        return 0;
    }
    // Multi-plane OSTypes are not MTLPixelFormats (Metal plane: API).
    if iosurface_fourcc_is_biplanar(pixel_format) {
        return 0;
    }
    // Already a small MTLPixelFormat ordinal (type-11 path).
    if pixel_format <= 0x200 {
        return pixel_format as u16;
    }
    match pixel_format {
        // 'BGRA' / 'ARGB' (kb: ARGB fourcc → BGRA8Unorm 0x50 for render targets)
        0x4247_5241 | 0x4152_4742 => MTL_FORMAT_BGRA8_UNORM,
        // 'RGBA'
        0x5247_4241 => MTL_FORMAT_RGBA8_UNORM,
        // 'RGhA' / half-float variants seen as AhGR in notes
        0x5247_6841 | 0x4168_4752 => MTL_FORMAT_RGBA16_FLOAT,
        // Single-plane R8 / RG8 OSTypes used as plane textures (not biplanar media fourcc).
        // 'L008' / common R8 fourccs are rare on type-4; MTL ordinals already handled above.
        // 'R8  ' / 'RG08' if ever seen as OSType:
        0x5238_2020 => MTL_FORMAT_R8_UNORM,
        0x5247_3038 => MTL_FORMAT_RG8_UNORM,
        // Unknown FourCC: 0 — callers fail closed (no BGRA invent).
        _ => 0,
    }
}

/// Decode one type-4 plane at `TYPE4_PLANES + i*TYPE4_PLANE_STRIDE`.
fn decode_type4_plane(desc: &[u8], plane_index: usize) -> Option<Type4Plane> {
    let base = TYPE4_PLANES + plane_index * TYPE4_PLANE_STRIDE;
    if desc.len() < base + TYPE4_PLANE_STRIDE {
        return None;
    }
    let offset = ld32(&desc[base..]);
    let width = ld32(&desc[base + 4..]);
    let height = ld32(&desc[base + 8..]);
    let packed = ld32(&desc[base + 12..]);
    let bytes_per_row = packed & 0x00ff_ffff;
    let bytes_per_element = ((packed >> 24) & 0xff) as u8;
    Some(Type4Plane {
        offset,
        width,
        height,
        bytes_per_row,
        bytes_per_element,
    })
}

/// The bytes of a type-4 surface descriptor that [`decode_type4_surface`] does
/// **not** read: `+0x11..0x14` and everything past the plane records it
/// consumed (`TYPE4_PLANES + plane_count * TYPE4_PLANE_STRIDE ..`).
///
/// Decoded today: `length` (+0x00), `backing_pfn` (+0x08), `pixel_format`
/// (+0x0c), `plane_count` (+0x10), and each plane's offset/width/height/packed
/// bpr. That is everything we know about a surface when the guest creates it —
/// and it is not enough to tell a desktop swapchain buffer from a same-geometry
/// offscreen render target, because a WebKit content tile is also 1920x1080
/// 'BGRA'. Membership is therefore reconstructed downstream by compositor-output
/// edges, full-frame-publish detection, output groups, presented-ness, and the
/// a/b seed.
///
/// **Measured: the guest is not telling us here.** Across one 1766 s x86/Vulkan
/// session with a real GUI login (boot `20260728-163046`), the probe below
/// emitted exactly two shapes for ≥5983 decodes over 453 distinct surface ids
/// and 154 distinct geometries — desktop swapchain buffers and never-displayed
/// content tiles alike:
///
/// ```text
/// type4_desc_shape distinct=1 1920x1080 fmt=0x42475241 planes=1 len=36 undecoded_len=3 undecoded_nz=0
/// type4_desc_shape distinct=2   320x320 fmt=0x34323066 planes=2 len=52 undecoded_len=3 undecoded_nz=0
/// ```
///
/// `len` is `TYPE4_PLANES + plane_count * TYPE4_PLANE_STRIDE` exactly, and it is
/// the *guest's* number — [`read_descriptor`] honours `descriptor_length` with no
/// clamp. The record ends where the plane array ends; the only bytes we skip are
/// the three at `+0x11`, and they were zero every time. There is nowhere in this
/// descriptor for a usage, bind, scanout or role hint to be, so no rule over
/// surface identity can classify a brand-new buffer before its first draw.
///
/// Narrow: this is the type-4 record on the x86 PCI pathway. It says nothing
/// about type-11 (`decode_iosurface_texture_descriptor`, which does not run
/// here and whose 0x38/0x58 blobs are still read only to 0x20), and a
/// create-time record we never read at all would be invisible to it.
///
/// A `plane_count` above [`TYPE4_PLANE_CAP`] is clamped by the decoder, so the
/// records past the clamp fall into this span too — which is correct: they are
/// bytes we did not read.
///
/// Public so the probe's notion of "undecoded" is pinned by a test rather than
/// restated in a log format string.
pub fn undecoded_type4_surface_bytes(desc: &[u8]) -> Vec<u8> {
    if desc.len() < TYPE4_MIN_LEN {
        return Vec::new();
    }
    let plane_count = (desc[TYPE4_PLANE_COUNT] as usize).min(TYPE4_PLANE_CAP);
    let planes_end = TYPE4_PLANES + plane_count * TYPE4_PLANE_STRIDE;
    let mut out = Vec::new();
    out.extend_from_slice(&desc[0x11..TYPE4_PLANES]);
    if planes_end < desc.len() {
        out.extend_from_slice(&desc[planes_end..]);
    }
    out
}

/// One always-on line per distinct `(len, undecoded span)`, capped.
///
/// Keyed on the **content** of the undecoded bytes, never on the record length.
/// The `display_txn_payload` probe keyed its budget on `(opcode, payload_len)`,
/// the length never varied, and it exhausted itself inside the first 400 ms —
/// it answered one question and then went blind for the rest of the session. A
/// new *value* is the interesting event here, so that is the key.
///
/// Runs before the decoder's own validity checks, so a record that fails to
/// decode still reports. An earlier version of this probe on the type-11
/// descriptor sat after its length check and emitted nothing at all on a live
/// boot; "the decoder never ran" and "the tail is constant" produced the same
/// silence, which is the reading the probe exists to rule out.
///
/// Hitting the cap is reported once. A silent truncation would read like "we
/// saw everything", which is the same class of error as a probe reporting a
/// confident constant.
fn note_type4_surface_shape(desc: &[u8]) {
    const MAX_SHAPES: usize = 24;
    const HEX_MAX: usize = 128;
    use std::sync::Mutex;
    type ShapeKey = (usize, Vec<u8>);
    static SEEN: Mutex<Option<std::collections::BTreeSet<ShapeKey>>> = Mutex::new(None);

    let undecoded = undecoded_type4_surface_bytes(desc);
    let (fresh, distinct) = {
        let mut guard = SEEN.lock().unwrap_or_else(|p| p.into_inner());
        let seen = guard.get_or_insert_with(Default::default);
        if seen.len() > MAX_SHAPES {
            return;
        }
        (seen.insert((desc.len(), undecoded.clone())), seen.len())
    };
    if !fresh {
        return;
    }
    if distinct > MAX_SHAPES {
        crate::observe::fail(format!(
            "type4_desc_shape outcome=cap_reached distinct={distinct} \
             (the undecoded span varies per surface; it is not a constant tail)"
        ));
        return;
    }
    let (w, h, fmt, pc) = if desc.len() >= TYPE4_MIN_LEN {
        (
            ld32(&desc[TYPE4_PLANES + 4..]),
            ld32(&desc[TYPE4_PLANES + 8..]),
            ld32(&desc[TYPE4_PIXEL_FORMAT..]),
            desc[TYPE4_PLANE_COUNT],
        )
    } else {
        (0, 0, 0, 0)
    };
    let hex: String = desc
        .iter()
        .take(HEX_MAX)
        .map(|b| format!("{b:02x}"))
        .collect();
    crate::observe::fail(format!(
        "type4_desc_shape distinct={distinct} {w}x{h} fmt={fmt:#x} planes={pc} len={} \
         undecoded_len={} undecoded_nz={} hex={hex}{}",
        desc.len(),
        undecoded.len(),
        undecoded.iter().filter(|&&b| b != 0).count(),
        if desc.len() > HEX_MAX { "…" } else { "" },
    ));
}

/// Decode a type-4 surface descriptor blob.
pub fn decode_type4_surface(desc: &[u8]) -> Option<Type4Surface> {
    note_type4_surface_shape(desc);
    if desc.len() < TYPE4_MIN_LEN {
        return None;
    }
    let length = ld64(&desc[TYPE4_LEN..]);
    let backing_pfn = ld32(&desc[TYPE4_BACKING_PFN..]);
    let pixel_format = ld32(&desc[TYPE4_PIXEL_FORMAT..]);
    let plane_count_raw = desc[TYPE4_PLANE_COUNT];
    if backing_pfn == 0 || length == 0 {
        return None;
    }
    let plane_count = (plane_count_raw as usize).min(TYPE4_PLANE_CAP) as u8;
    let mut planes = [Type4Plane::default(); TYPE4_PLANE_CAP];
    for (i, plane) in planes.iter_mut().enumerate().take(plane_count as usize) {
        if let Some(p) = decode_type4_plane(desc, i) {
            *plane = p;
        }
    }
    let (width, height, bpr) = if plane_count > 0 {
        let p0 = planes[0];
        (p0.width, p0.height, p0.bytes_per_row)
    } else {
        (0, 0, 0)
    };
    Some(Type4Surface {
        length,
        backing_pfn,
        pixel_format,
        plane_count,
        planes,
        width,
        height,
        bytes_per_row: bpr,
    })
}

/// Build `sIOSurfaceDeviceDescriptor` geometry from type-4 wire (no invent).
///
/// Multi-plane: plane records from type-4 planes; sample path selects by geometry
///. Single-plane: surface-level fields only
/// (`plane_count==0` path in `sample_window_prefer_device`).
fn synthesize_device_desc_from_type4(surf: &Type4Surface) -> Vec<u8> {
    let mut device_desc = vec![0u8; DEVICE_DESC_LEN];
    let multi = type4_is_multiplanar(surf);
    let mtl = iosurface_pixel_format_to_mtl(surf.pixel_format);
    // Device desc pixelFormat field: guest stores getPixelFormat() (FourCC for
    // biplanar media). Single-plane product sample uses MTL ordinal when known.
    let fmt_word = if multi {
        surf.pixel_format
    } else if mtl != 0 {
        mtl as u32
    } else {
        surf.pixel_format
    };
    st32(&mut device_desc[DEVICE_DESC_PIXEL_FORMAT..], fmt_word);
    let alloc = if surf.length > u32::MAX as u64 {
        u32::MAX
    } else {
        surf.length as u32
    };
    st32(&mut device_desc[DEVICE_DESC_ALLOC_SIZE..], alloc);
    // Surface-level dims/bpr from plane0 (same as type-4 plane0 convenience).
    let dims = ((surf.width as u64) << 8) | ((surf.height as u64) << 40);
    st64(&mut device_desc[DEVICE_DESC_DIMS..], dims);
    if surf.bytes_per_row > 0 {
        st32(&mut device_desc[DEVICE_DESC_BPR..], surf.bytes_per_row);
    }
    if multi && surf.plane_count > 0 {
        // Multi-plane: publish plane records; sample_window_prefer_device matches
        // type-11 R8/RG8 binds by (w,h,bpe). Do not invent bases from format alone.
        let n = (surf.plane_count as usize).min(TYPE4_PLANE_CAP);
        device_desc[DEVICE_DESC_PLANE_COUNT] = n as u8;
        // Surface-level bpe: plane0 element size when wire provides it.
        let bpe0 = surf.planes[0].bytes_per_element;
        if bpe0 != 0 {
            st16(&mut device_desc[DEVICE_DESC_BPE..], bpe0 as u16);
        }
        for i in 0..n {
            let p = &surf.planes[i];
            let base = DEVICE_DESC_PLANES + i * DEVICE_PLANE_DESC_LEN;
            st32(&mut device_desc[base + DEVICE_PLANE_OFFSET..], p.offset);
            // plane_size: 0 = skip size check in sample_window_from_device_plane
            // (type-4 wire has offset/w/h/bpr, not a separate size field).
            st32(&mut device_desc[base + DEVICE_PLANE_SIZE..], 0);
            let pdims = ((p.width as u64) << 8) | ((p.height as u64) << 40);
            st64(&mut device_desc[base + DEVICE_PLANE_DIMS..], pdims);
            st32(&mut device_desc[base + DEVICE_PLANE_BPR..], p.bytes_per_row);
            if p.bytes_per_element != 0 {
                st16(
                    &mut device_desc[base + DEVICE_PLANE_BPE..],
                    p.bytes_per_element as u16,
                );
            } else if iosurface_fourcc_is_biplanar(surf.pixel_format) {
                // Contract: 420 Y bpe=1, UV bpe=2 when wire high-byte is 0.
                // Only fill when FourCC is known biplanar — not a free invent for
                // arbitrary multi-plane. Matches Metal R8/RG8 plane bind bpp.
                let bpe = if i == 0 { 1u16 } else { 2u16 };
                st16(&mut device_desc[base + DEVICE_PLANE_BPE..], bpe);
            }
        }
    } else {
        // Single-plane surface-level sample path (plane_count 0).
        device_desc[DEVICE_DESC_PLANE_COUNT] = 0;
        if mtl != 0 {
            if let Some(bpp) = crate::contract::pixel_format::bytes_per_pixel(mtl) {
                st16(&mut device_desc[DEVICE_DESC_BPE..], bpp as u16);
            }
        }
    }
    device_desc
}

/// Lookup one object-list slot for `task_id` / `ref_`.
pub fn lookup_list_entry<M: HostMemory>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    ref_: u32,
) -> Option<ListObjectEntry> {
    if task_id as usize >= MAX_TASKS {
        return None;
    }
    let task = &state.tasks[task_id as usize];
    if !task.active || task.object_list_count == 0 {
        return None;
    }
    let off = list_object_entry_offset(ref_, task.object_list_count)?;
    let entry_gva = ((task.object_list_pfn as u64) << state.page_shift).checked_add(off)?;
    let mut raw = [0u8; OBJECT_LIST_ENTRY_LEN];
    gva_mem::read_task_gva_by_id(
        host,
        &state.tasks,
        task_id,
        entry_gva,
        &mut raw,
        state.page_shift,
    )
    .ok()?;
    let e = decode_list_object_entry(&raw).ok()?;
    if e.descriptor_length == 0 || e.descriptor_gva == 0 {
        return None;
    }
    Some(e)
}

/// Read the descriptor blob for a list entry.
pub fn read_descriptor<M: HostMemory>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    entry: &ListObjectEntry,
) -> Option<Vec<u8>> {
    // Guest descriptor_length is authoritative — no product 4 KiB read clamp.
    let len = crate::runtime::metal_draw::host_alloc_len(entry.descriptor_length as u64)
        .filter(|&n| n > 0)?;
    let mut buf = vec![0u8; len];
    gva_mem::read_task_gva_by_id(
        host,
        &state.tasks,
        task_id,
        entry.descriptor_gva,
        &mut buf,
        state.page_shift,
    )
    .ok()?;
    Some(buf)
}

/// Resolve object ref and, if type-11, latch mapping geometry + cache the entry.
///
/// Returns the mapping_id for type-11 textures, or None.
pub fn resolve_type11_ref<M: HostMemory>(
    state: &mut DeviceState,
    host: &M,
    task_id: u32,
    ref_: u32,
) -> Option<u32> {
    let entry = lookup_list_entry(state, host, task_id, ref_)?;
    // The list entry passed validation (descriptor_gva != 0, length != 0) but
    // its descriptor blob is unreadable — genuine, only for a bound entry.
    let Some(desc) = read_descriptor(state, host, task_id, &entry) else {
        note_type11_fail(
            task_id,
            ref_,
            "type11_desc_read",
            format!(
                "type11_resolve_fail reason=type11_desc_read task={task_id} ref={ref_} obj_type={} desc_gva={:#x} desc_len={}",
                entry.object_type, entry.descriptor_gva, entry.descriptor_length
            ),
        );
        return None;
    };
    // Cache in the sparse object table (model ObjectEntry).
    let _ = state.insert_object(
        task_id,
        ref_,
        ObjectEntry {
            object_type: entry.object_type,
            desc_gva: entry.descriptor_gva,
            desc_len: entry.descriptor_length,
        },
    );
    if entry.object_type != OBJECT_TYPE_IOSURFACE {
        // Legitimate: this ref is a different object type, not a texture. Normal
        // control flow (resolve_type11_refs skips it) — never a failure.
        return None;
    }
    if !texture::register_from_descriptor_bytes(state, OBJECT_TYPE_IOSURFACE, &desc) {
        // A confirmed IOSurface texture whose descriptor could not register —
        // the draw then samples a missing/black texture.
        note_type11_fail(
            task_id,
            ref_,
            "type11_register",
            format!(
                "type11_resolve_fail reason=type11_register task={task_id} ref={ref_} desc_len={}",
                desc.len()
            ),
        );
        return None;
    }
    // mapping_id is first u32 of type-11 desc.
    let mapping_id = u32::from_le_bytes(desc[0..4].try_into().ok()?);
    if mapping_id == 0 {
        note_type11_fail(
            task_id,
            ref_,
            "type11_mapping_zero",
            format!("type11_resolve_fail reason=type11_mapping_zero task={task_id} ref={ref_} desc_len={}", desc.len()),
        );
        return None;
    }
    state.texture_to_mapping.insert((task_id, ref_), mapping_id);
    // Resolved: re-arm so a later genuine failure on this ref logs again.
    clear_type11_fail(task_id, ref_);
    Some(mapping_id)
}

/// Apply a decoded type-4 surface as page-table backing for `surface_id`.
///
/// `backing_pfn` is a GPU-VA page (same source as type-2/3 textures). Translate
/// each consecutive GVA page through the task page table into GPA page entries
/// the scanout path already understands.
fn apply_type4_backing<M: HostMemory>(
    state: &mut DeviceState,
    host: &M,
    task_id: u32,
    surface_id: u32,
    surf: &Type4Surface,
) -> bool {
    if surface_id == 0 || surface_id as usize >= MAX_MAPPINGS {
        note_type4_fail(
            surface_id,
            "sid_oob",
            format!("type4_backing_fail reason=sid_oob sid={surface_id} task={task_id} max={MAX_MAPPINGS}"),
        );
        return false;
    }
    let page_shift = state.page_shift;
    let page_size = page_size_of(page_shift);
    if page_size == 0 {
        note_type4_fail(
            surface_id,
            "page_size_zero",
            format!("type4_backing_fail reason=page_size_zero sid={surface_id} task={task_id} page_shift={page_shift}"),
        );
        return false;
    }
    let page_count = ((surf.length.saturating_sub(1)) / page_size) + 1;
    // No host MiB budget: page count follows guest `surf.length` only.
    // Fail if zero or not host-addressable as a page-entry vector.
    if page_count == 0 || crate::runtime::metal_draw::host_alloc_len(page_count).is_none() {
        note_type4_fail(
            surface_id,
            "page_count_oob",
            format!(
                "type4_backing_fail reason=page_count_oob sid={surface_id} task={task_id} len={:#x} page_count={page_count}",
                surf.length
            ),
        );
        return false;
    }
    let task = match state.tasks.get(task_id as usize) {
        Some(t) if t.active => t,
        _ => {
            note_type4_fail(
                surface_id,
                "task_inactive",
                format!("type4_backing_fail reason=task_inactive sid={surface_id} task={task_id}"),
            );
            return false;
        }
    };

    // Contract: backing_pfn is getGPUVirtualAddress>>page_shift (GPU-VA page).
    // Translate each consecutive GVA page through the task directory.
    // Identity GPA is only used when the walk fails *and* that GPA is RAM —
    // never preferred by content heuristics (AGENTS.md).
    let mut entries = Vec::with_capacity(page_count as usize);
    let mut gva_hits = 0u32;
    let mut id_hits = 0u32;
    for i in 0..page_count {
        let gva = ((surf.backing_pfn as u64) + i) << page_shift;
        let walked = gva_mem::translate_task_gva(host, task, gva, page_shift);
        let gpa = match walked {
            Some(g) => {
                gva_hits = gva_hits.saturating_add(1);
                Some(g)
            }
            None => {
                let candidate = gva;
                let mut probe = [0u8; 1];
                if host.read_gpa(candidate, &mut probe).is_ok() {
                    id_hits = id_hits.saturating_add(1);
                    Some(candidate)
                } else {
                    None
                }
            }
        };
        let Some(gpa) = gpa else {
            crate::observe::fail(format!(
                "type4 translate fail sid={surface_id} task={task_id} page={i}/{} gva={gva:#x}",
                page_count
            ));
            return false;
        };
        let pfn = gpa >> page_shift;
        if pfn > u32::MAX as u64 {
            note_type4_fail(
                surface_id,
                "pfn_oob",
                format!("type4_backing_fail reason=pfn_oob sid={surface_id} task={task_id} page={i}/{page_count} gpa={gpa:#x} pfn={pfn:#x}"),
            );
            return false;
        }
        let entry = ((pfn as u32) << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
        // Sanity: entry_gpa must round-trip.
        if entry_gpa_shift(entry, page_shift) != Some(gpa & !(page_size - 1)) {
            note_type4_fail(
                surface_id,
                "entry_roundtrip",
                format!("type4_backing_fail reason=entry_roundtrip sid={surface_id} task={task_id} page={i}/{page_count} gpa={gpa:#x} entry={entry:#x}"),
            );
            return false;
        }
        entries.push(entry);
    }
    // Bring-up probe once per surface_id (first attach).
    let first_attach = state
        .mappings
        .get(&surface_id)
        .map(|m| m.page_entries.is_empty())
        .unwrap_or(true);
    if first_attach && page_count >= 1 {
        let g0 = entry_gpa_shift(entries[0], page_shift).unwrap_or(0);
        let g1 = entries
            .get(1)
            .and_then(|&e| entry_gpa_shift(e, page_shift))
            .unwrap_or(0);
        let g2 = entries
            .get(2)
            .and_then(|&e| entry_gpa_shift(e, page_shift))
            .unwrap_or(0);
        let mut sample = [0u8; 16];
        let snz = if host.read_gpa(g0, &mut sample).is_ok() {
            sample.iter().filter(|&&b| b != 0).count()
        } else {
            0
        };
        // Tight plane0 footprint using wire bpe when present (else 0 — log only).
        let bpe0 = if surf.planes[0].bytes_per_element != 0 {
            surf.planes[0].bytes_per_element as u64
        } else if iosurface_fourcc_is_biplanar(surf.pixel_format) {
            1
        } else if iosurface_pixel_format_to_mtl(surf.pixel_format) != 0 {
            crate::contract::pixel_format::bytes_per_pixel(iosurface_pixel_format_to_mtl(
                surf.pixel_format,
            ))
            .unwrap_or(0) as u64
        } else {
            0
        };
        let plane0_bytes = (surf.width as u64)
            .saturating_mul(surf.height as u64)
            .saturating_mul(bpe0);
        // Bring-up census (dims/fmt/sample probe), not a drop — the genuine
        // type-4 failures route through note_type4_fail with reason=. On the
        // always-on `off()` sink, not `fail()`: under surface recycling this
        // "first attach" re-fires per recycle (page_entries cleared by the
        // teardown), so on fail() it floods the curated real-error view (~4k
        // lines under a continuously-animating app, burying genuine failures).
        crate::observe::off(format!(
            "type4 pages sid={surface_id} task={task_id} n={page_count} gva_hits={gva_hits} id_hits={id_hits} gpa0={g0:#x} gpa1={g1:#x} gpa2={g2:#x} sample0_nz={snz}/16 w={} h={} bpr={} len={:#x} plane0_bytes={plane0_bytes} fmt={:#x} planes={} multi={}",
            surf.width,
            surf.height,
            surf.bytes_per_row,
            surf.length,
            surf.pixel_format,
            surf.plane_count,
            type4_is_multiplanar(surf) as u8
        ));
    }

    if !state.map_surface(surface_id) {
        note_type4_fail(
            surface_id,
            "map_surface",
            format!("type4_backing_fail reason=map_surface sid={surface_id} task={task_id} n={page_count}"),
        );
        return false;
    }
    // Device desc from type-4 wire only (single- or multi-plane). No BGRA invent.
    let device_desc = synthesize_device_desc_from_type4(surf);

    if let Some(m) = state.mappings.get_mut(&surface_id) {
        // `map_surface` above stashed the prior bindings as the incarnation
        // fingerprint (the notify-vs-eager-resolve rule): compare the fresh
        // plan against it — identical pages are the SAME incarnation (no
        // bump; deferred windows and the resident survive), a change is the
        // recycled-mid rule (bump; stale residents/views must never survive).
        let prior = m
            .condemned_entries
            .take()
            .unwrap_or_else(|| std::mem::take(&mut m.page_entries));
        let changed = prior != entries;
        let replaced = !prior.is_empty() && changed;
        if changed {
            crate::model::DeviceState::bump_map_generation(m);
        }
        if replaced {
            // Recycled-mid backing-refresh census (not a drop; the recycle rate
            // is summarized by teardown_churn). Off the curated fail() view —
            // per-recycle under animation churn it floods the real-error view.
            crate::observe::off(format!(
                "type4_pages_refreshed sid={surface_id} task={task_id} n={} map_gen={}",
                entries.len(),
                m.map_generation
            ));
            // Present evidence needs no prune here: this branch only runs when
            // the plan changed, which bumped `map_generation` just above, and
            // the evidence is stamped with the incarnation that recorded it.
            // Pruning it unconditionally is what the identical-plan path used
            // to do via `map_surface`, and that demoted a surface the compare
            // had just called the SAME incarnation.
        }
        m.page_entries = entries;
        m.mapped = true;
        m.page_table_kva = 0;
        m.device_desc = device_desc;
        // Contiguous view must be rebuilt.
        if m.contig_ptr != 0 {
            state.retired_views.push((m.contig_ptr, m.contig_len));
            m.contig_ptr = 0;
            m.contig_len = 0;
        }
    }

    // Mapping geom format: single-plane MTL only. Multi-plane / unknown → format 0
    // (stage/paint must not invent BGRA; type-11 selects planes via device_desc).
    let format = iosurface_pixel_format_to_mtl(surf.pixel_format);
    if surf.width > 0 && surf.height > 0 {
        if type4_is_multiplanar(surf) {
            // Dims from plane0 for bookkeeping; format 0 = not a single color RT.
            let _ = state.set_mapping_geom(surface_id, surf.width, surf.height, 0);
        } else if format != 0 {
            let _ = state.set_mapping_geom(surface_id, surf.width, surf.height, format);
        } else if surf.width > 0 && surf.height > 0 {
            // Unknown single-plane FourCC: keep dims, format 0 (fail closed on sample bpp).
            let _ = state.set_mapping_geom(surface_id, surf.width, surf.height, 0);
        }
    }

    // Backing built cleanly — re-arm the fail latch so a later genuine failure
    // on this surface (flapping backing) is logged again.
    clear_type4_fail(surface_id);
    true
}

/// Resolve present `surface_id` to type-4 backing pages + geometry.
///
/// Scans active tasks: object-list slot `surface_id` must be type-4 (heap is
/// indexed by IOSurface surface ID). Returns true when pages were latched.
pub fn resolve_type4_surface<M: HostMemory>(
    state: &mut DeviceState,
    host: &M,
    surface_id: u32,
) -> bool {
    resolve_type4_surface_ex(state, host, surface_id, false)
}

/// Like [`resolve_type4_surface`] but always re-reads the object list / PT.
pub fn resolve_type4_surface_force<M: HostMemory>(
    state: &mut DeviceState,
    host: &M,
    surface_id: u32,
) -> bool {
    resolve_type4_surface_ex(state, host, surface_id, true)
}

/// Latch the task that owns `surface_id` as its type-4 backing so the next
/// present-path scan tries it right after task 0.
fn record_type4_owner(state: &mut DeviceState, surface_id: u32, task_id: u32) {
    if let Some(m) = state.mappings.get_mut(&surface_id) {
        m.owner_task_hint = task_id;
    }
}

fn resolve_type4_surface_ex<M: HostMemory>(
    state: &mut DeviceState,
    host: &M,
    surface_id: u32,
    force: bool,
) -> bool {
    if surface_id == 0 || surface_id as usize >= MAX_MAPPINGS {
        return false;
    }
    // Task probe order: task 0 (kernel/global — historical type-4 home) first,
    // then the cached owner-task hint (so a hot present-path re-scan
    // short-circuits on the owning task instead of walking all 256 slots),
    // then the remaining tasks.
    let hint = state
        .mappings
        .get(&surface_id)
        .map(|m| m.owner_task_hint)
        .unwrap_or(0);
    let mut order: Vec<u32> = Vec::with_capacity(MAX_TASKS + 1);
    order.push(0);
    if hint != 0 && (hint as usize) < MAX_TASKS {
        order.push(hint);
    }
    for tid in 1..MAX_TASKS as u32 {
        if tid == hint {
            continue;
        }
        order.push(tid);
    }

    for task_id in order {
        if task_id as usize >= state.tasks.len() {
            continue;
        }
        if !state.tasks[task_id as usize].active {
            continue;
        }
        // Count the guest-read cost of one active-task object-list probe.
        let Some(entry) = lookup_list_entry(state, host, task_id, surface_id) else {
            continue;
        };
        if entry.object_type != OBJECT_TYPE_SURFACE {
            continue;
        }
        let Some(desc) = read_descriptor(state, host, task_id, &entry) else {
            note_type4_fail(
                surface_id,
                "desc_read",
                format!(
                    "type4_backing_fail reason=desc_read sid={surface_id} task={task_id} desc_gva={:#x} desc_len={}",
                    entry.descriptor_gva, entry.descriptor_length
                ),
            );
            continue;
        };
        let _ = state.insert_object(
            task_id,
            surface_id,
            ObjectEntry {
                object_type: entry.object_type,
                desc_gva: entry.descriptor_gva,
                desc_len: entry.descriptor_length,
            },
        );
        let Some(surf) = decode_type4_surface(&desc) else {
            note_type4_fail(
                surface_id,
                "desc_decode",
                format!(
                    "type4_backing_fail reason=desc_decode sid={surface_id} task={task_id} desc_len={} backing_pfn={:#x} length={:#x}",
                    desc.len(),
                    desc.get(TYPE4_BACKING_PFN..TYPE4_BACKING_PFN + 4)
                        .map(ld32)
                        .unwrap_or(0),
                    desc.get(TYPE4_LEN..TYPE4_LEN + 8)
                        .map(ld64)
                        .unwrap_or(0)
                ),
            );
            continue;
        };
        // Force path validated the cached pages are still fresh → keep them.
        let mut force_fresh = false;
        // Skip rebuild when pages already match this backing (hot present path).
        if !force {
            let same_geom = state
                .mappings
                .get(&surface_id)
                .map(|m| {
                    m.mapped
                        && !m.page_entries.is_empty()
                        && m.has_geom
                        && m.width == surf.width
                        && m.height == surf.height
                })
                .unwrap_or(false);
            if same_geom {
                // Same geom + non-empty pages: keep (guest double-buffer
                // may still rewrite page *content* without changing pfn).
                record_type4_owner(state, surface_id, task_id);
                return true;
            }
        } else if let Some(m) = state.mappings.get(&surface_id) {
            // Force: keep the cached table only while the CURRENT task
            // page-table translation of the descriptor's first and last
            // backing pages still matches it. `backing_pfn` is a GPU-VA page;
            // the guest may remap that GVA range onto new physical pages
            // without changing surface id, geometry, or length (early-boot
            // console FB vs the WindowServer reallocation). A same-size guard
            // here kept boot-time pages forever, so presents froze on pages
            // nobody writes.
            if m.mapped && !m.page_entries.is_empty() {
                let page_shift = state.page_shift;
                let page_size = page_size_of(page_shift);
                let need = ((surf.length.saturating_sub(1)) / page_size) + 1;
                if m.page_entries.len() as u64 == need && m.width == surf.width {
                    let task = state.tasks.get(task_id as usize).filter(|t| t.active);
                    let entry_fresh = |idx: u64, entry: u32| -> bool {
                        let gva = ((surf.backing_pfn as u64) + idx) << page_shift;
                        let cached = entry_gpa_shift(entry, page_shift);
                        match task
                            .and_then(|t| gva_mem::translate_task_gva(host, t, gva, page_shift))
                        {
                            Some(gpa) => cached == Some(gpa & !(page_size - 1)),
                            // Walk fails now: cached identity fallback is
                            // still the best available answer — keep it.
                            None => cached == Some(gva),
                        }
                    };
                    let last = m.page_entries.len() - 1;
                    if entry_fresh(0, m.page_entries[0])
                        && entry_fresh(last as u64, m.page_entries[last])
                    {
                        force_fresh = true;
                    } else {
                        crate::observe::fail(format!(
                            "type4_pages_stale sid={surface_id} task={task_id} n={} gpa0={:#x} (task PT translation moved; rebuilding)",
                            m.page_entries.len(),
                            entry_gpa_shift(m.page_entries[0], page_shift).unwrap_or(0)
                        ));
                    }
                }
            }
        }
        if force_fresh {
            record_type4_owner(state, surface_id, task_id);
            return true;
        }
        if apply_type4_backing(state, host, task_id, surface_id, &surf) {
            record_type4_owner(state, surface_id, task_id);
            return true;
        }
    }
    false
}

/// Ensure surface backing for present: type-4 pages when needed, else keep arm
/// MappingInternal path.
///
/// Resolves type-4 once pages are empty; guest double-buffering uses distinct
/// surface_ids (content updates land in-place on an already-mapped pfn).
pub fn ensure_surface_for_present<M: HostMemory + crate::runtime::host::HostOps>(
    state: &mut DeviceState,
    host: &M,
    surface_id: u32,
) -> bool {
    if surface_id == 0 {
        return false;
    }
    let need = state
        .mappings
        .get(&surface_id)
        .map(|m| !m.mapped || m.page_entries.is_empty())
        .unwrap_or(true);
    if need {
        let _ = resolve_type4_surface(state, host, surface_id);
    } else {
        // Opportunistic refresh if wire geom changed (mode switch).
        let _ = resolve_type4_surface_force(state, host, surface_id);
    }
    // Arm/iosfc path: MappingInternal resolve when captured.
    let _ = crate::runtime::mapper::ensure_resolved_for_scanout(state, host, surface_id);
    state
        .mappings
        .get(&surface_id)
        .map(|m| m.mapped && !m.page_entries.is_empty() && m.has_geom)
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::contract::endian::{ld32, st16, st32, st64};
    use crate::contract::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
    use crate::contract::iosurface_pages::DEVICE_DESC_PLANE_COUNT;
    use crate::model::{DeviceId, PAGE_SHIFT_ARM64E, PAGE_SHIFT_X86};
    use crate::runtime::host::FakeHost;

    #[test]
    fn type11_fail_latch_dedups_per_task_ref_and_rearms_on_clear() {
        // Flood guard for the per-draw-per-ref resolve path: a genuinely-broken
        // type-11 ref logs each reason once, isolates per (task,ref), and
        // re-arms on resolve. Unique ids so this never races real refs across
        // the process-global latch.
        let (t, r, r2) = (0xAB01u32, 0xCD01u32, 0xCD02u32);
        clear_type11_fail(t, r);
        clear_type11_fail(t, r2);
        let seen = |task: u32, rf: u32, reason: &'static str| {
            type11_fail_latch()
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .contains(&(task, rf, reason))
        };
        note_type11_fail(t, r, "type11_register", "x".into());
        assert!(seen(t, r, "type11_register"));
        // Distinct reason on the same ref tracked independently.
        note_type11_fail(t, r, "type11_desc_read", "x".into());
        assert!(seen(t, r, "type11_desc_read"));
        // A different ref is untouched.
        assert!(!seen(t, r2, "type11_register"));
        note_type11_fail(t, r2, "type11_register", "x".into());
        // Clearing r re-arms only r, leaves r2.
        clear_type11_fail(t, r);
        assert!(!seen(t, r, "type11_register"));
        assert!(!seen(t, r, "type11_desc_read"));
        assert!(seen(t, r2, "type11_register"));
        clear_type11_fail(t, r2);
    }

    fn setup_task_with_list(host: &mut FakeHost, state: &mut DeviceState) {
        // Same 1-level map as gva_mem test: GVA page 0 → data pfn 4.
        let dir_gpa = 2u64 << PAGE_SHIFT_ARM64E;
        let root_gpa = 3u64 << PAGE_SHIFT_ARM64E;
        let data_gpa = 4u64 << PAGE_SHIFT_ARM64E;
        host.map_range(dir_gpa, 0x20, 0);
        host.map_range(root_gpa, 0x4000, 0);
        host.map_range(data_gpa, 0x200, 0);
        let mut d = [0u8; 8];
        st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 3);
        st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
        let _ = host.write_gpa(dir_gpa, &d);
        st32(&mut d[..4], 4);
        let _ = host.write_gpa(root_gpa, &d[..4]);

        assert!(state.define_task(1, 0x1000, 2));
        // list base GVA 0 (pfn field 0 allowed)
        assert!(state.set_object_list(1, 0, 8));
        let mut entry = [0u8; 12];
        st32(&mut entry[0..], 11u32 | (0x20u32 << 8));
        entry[4..12].copy_from_slice(&0x40u64.to_le_bytes());
        let _ = host.write_gpa(data_gpa + 12, &entry);
        let mut desc = [0u8; 0x20];
        st32(&mut desc[0..], 9);
        st16(&mut desc[0x16..], 0x50);
        st32(&mut desc[0x18..], 64);
        st32(&mut desc[0x1c..], 32);
        let _ = host.write_gpa(data_gpa + 0x40, &desc);
    }

    #[test]
    fn resolve_type11_from_list() {
        let mut host = FakeHost::new();
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        setup_task_with_list(&mut host, &mut state);
        // Sanity: list entry readable
        let e = lookup_list_entry(&state, &host, 1, 1).expect("list entry");
        assert_eq!(e.object_type, 11);
        assert_eq!(e.descriptor_gva, 0x40);
        let mid = resolve_type11_ref(&mut state, &host, 1, 1).expect("type11");
        assert_eq!(mid, 9);
        let m = state.mappings.get(&9).unwrap();
        assert!(m.has_geom);
        assert_eq!((m.width, m.height, m.format), (64, 32, 0x50));
    }

    #[test]
    fn decode_type4_plane0() {
        let mut desc = vec![0u8; 0x30];
        st64(&mut desc[0..], 0x1000);
        st32(&mut desc[8..], 0x100); // backing pfn
        st32(&mut desc[0xc..], 0x4247_5241); // 'BGRA'
        desc[0x10] = 1;
        st32(&mut desc[0x14..], 0); // plane offset
        st32(&mut desc[0x18..], 64);
        st32(&mut desc[0x1c..], 32);
        st32(&mut desc[0x20..], 256); // bpr
        let s = decode_type4_surface(&desc).expect("type4");
        assert_eq!(s.length, 0x1000);
        assert_eq!(s.backing_pfn, 0x100);
        assert_eq!((s.width, s.height, s.bytes_per_row), (64, 32, 256));
        assert_eq!(s.plane_count, 1);
        assert_eq!(s.planes[0].offset, 0);
        assert!(!type4_is_multiplanar(&s));
        assert_eq!(
            iosurface_pixel_format_to_mtl(s.pixel_format),
            crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM
        );
    }

    #[test]
    fn fourcc_420f_not_bgra_and_multiplanar() {
        assert_eq!(iosurface_pixel_format_to_mtl(IOSURFACE_FOURCC_420F), 0);
        assert_eq!(iosurface_pixel_format_to_mtl(IOSURFACE_FOURCC_420V), 0);
        assert!(iosurface_fourcc_is_biplanar(IOSURFACE_FOURCC_420F));
        // Unknown FourCC must not invent BGRA.
        assert_eq!(iosurface_pixel_format_to_mtl(0xdead_beef), 0);
    }

    #[test]
    fn decode_type4_biplanar_420f_planes() {
        // Wire: plane0 Y 1024×1024 bpr=1024 bpe=1; plane1 UV 512×512 bpr=1024 bpe=2.
        // Live boot: fmt='420f' len=0x180000 plane0 bpr=1024.
        let mut desc = vec![0u8; 0x14 + 2 * 0x10];
        st64(&mut desc[0..], 0x180000);
        st32(&mut desc[8..], 0x200);
        st32(&mut desc[0xc..], IOSURFACE_FOURCC_420F);
        desc[0x10] = 2;
        // plane0
        st32(&mut desc[0x14..], 0); // offset
        st32(&mut desc[0x18..], 1024);
        st32(&mut desc[0x1c..], 1024);
        st32(&mut desc[0x20..], 1024 | (1 << 24)); // bpr | bpe<<24
                                                   // plane1
        st32(&mut desc[0x24..], 1024 * 1024); // offset after Y
        st32(&mut desc[0x28..], 512);
        st32(&mut desc[0x2c..], 512);
        st32(&mut desc[0x30..], 1024 | (2 << 24));
        let s = decode_type4_surface(&desc).expect("type4 420f");
        assert!(type4_is_multiplanar(&s));
        assert_eq!(s.plane_count, 2);
        assert_eq!(
            (
                s.planes[0].width,
                s.planes[0].height,
                s.planes[0].bytes_per_row
            ),
            (1024, 1024, 1024)
        );
        assert_eq!(s.planes[0].bytes_per_element, 1);
        assert_eq!(
            (
                s.planes[1].width,
                s.planes[1].height,
                s.planes[1].bytes_per_element
            ),
            (512, 512, 2)
        );
        let dev = synthesize_device_desc_from_type4(&s);
        assert_eq!(dev[DEVICE_DESC_PLANE_COUNT], 2);
        use crate::contract::iosurface_pages::{
            decode_device_surface, sample_window_prefer_device, DEVICE_DESC_PIXEL_FORMAT,
        };
        assert_eq!(
            ld32(&dev[DEVICE_DESC_PIXEL_FORMAT..]),
            IOSURFACE_FOURCC_420F
        );
        let surf = decode_device_surface(&dev).expect("device");
        assert_eq!(surf.plane_count, 2);
        assert_eq!(surf.alloc_size, 0x180000);
        // Type-11 Y plane: R8 1024×1024 matches plane0 (contract geometry key).
        let y = sample_window_prefer_device(
            Some(&dev),
            None,
            crate::contract::pixel_format::MTL_FORMAT_R8_UNORM,
            1024,
            1024,
        )
        .expect("Y window");
        assert_eq!(y.0, 0); // offset
        assert_eq!(y.1, 1024); // bpr
        assert!(y.3); // from device
                      // UV plane: RG8 half res.
        let uv = sample_window_prefer_device(
            Some(&dev),
            None,
            crate::contract::pixel_format::MTL_FORMAT_RG8_UNORM,
            512,
            512,
        )
        .expect("UV window");
        assert_eq!(uv.0, 1024 * 1024);
        assert_eq!(uv.1, 1024);
        // BGRA invent of full 1024² must still reject (alloc < invent span).
        assert!(sample_window_prefer_device(
            Some(&dev),
            None,
            crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM,
            1024,
            1024,
        )
        .is_none());
    }

    #[test]
    fn resolve_type4_identity_gpa() {
        let mut host = FakeHost::new();
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        state.page_shift = PAGE_SHIFT_X86;
        // Pages for identity GPA fallback at pfn 0x20.
        let page = 0x20u64 << PAGE_SHIFT_X86;
        host.map_range(page, 0x2000, 0x5a);
        // Task with empty directory — force identity GPA path.
        assert!(state.define_task(0, 0x1000, 0)); // directory_pfn=0 → no GVA
                                                  // Put a type-4 entry at surface_id=3: need object list in GPA (not GVA).
                                                  // With directory_pfn=0, lookup will fail GVA. Build a task with PT.
        let dir_gpa = 2u64 << PAGE_SHIFT_X86;
        let root_gpa = 3u64 << PAGE_SHIFT_X86;
        let data_gpa = 4u64 << PAGE_SHIFT_X86;
        host.map_range(dir_gpa, 0x20, 0);
        host.map_range(root_gpa, 0x1000, 0);
        host.map_range(data_gpa, 0x200, 0);
        let mut d = [0u8; 8];
        st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 3);
        st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
        let _ = host.write_gpa(dir_gpa, &d);
        st32(&mut d[..4], 4);
        let _ = host.write_gpa(root_gpa, &d[..4]);
        assert!(state.define_task(1, 0x1000, 2));
        assert!(state.set_object_list(1, 0, 8));
        // Entry at index 3 (surface_id).
        let mut entry = [0u8; 12];
        st32(&mut entry[0..], 4u32 | (0x30u32 << 8));
        entry[4..12].copy_from_slice(&0x80u64.to_le_bytes());
        let _ = host.write_gpa(data_gpa + 3 * 12, &entry);
        let mut desc = vec![0u8; 0x30];
        st64(&mut desc[0..], 0x1000);
        st32(&mut desc[8..], 0x20); // identity GPA pfn
        st32(&mut desc[0xc..], 0x50);
        desc[0x10] = 1;
        st32(&mut desc[0x18..], 16);
        st32(&mut desc[0x1c..], 16);
        st32(&mut desc[0x20..], 64);
        let _ = host.write_gpa(data_gpa + 0x80, &desc);

        assert!(resolve_type4_surface(&mut state, &host, 3));
        let m = state.mappings.get(&3).unwrap();
        assert!(m.mapped);
        assert_eq!(m.page_entries.len(), 1);
        assert!(m.has_geom);
        assert_eq!((m.width, m.height), (16, 16));
    }

    /// Force-resolve must rebuild the cached page table when the task PT
    /// translation of the backing GVA moved (same surface id, same geometry,
    /// new physical pages — the early-boot FB vs WindowServer reallocation).
    #[test]
    fn resolve_type4_force_rebuilds_when_task_translation_moves() {
        let mut host = FakeHost::new();
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        state.page_shift = PAGE_SHIFT_X86;
        let dir_gpa = 2u64 << PAGE_SHIFT_X86;
        let root_gpa = 3u64 << PAGE_SHIFT_X86;
        let data_gpa = 4u64 << PAGE_SHIFT_X86;
        let old_page = 5u64 << PAGE_SHIFT_X86;
        let new_page = 6u64 << PAGE_SHIFT_X86;
        host.map_range(dir_gpa, 0x20, 0);
        host.map_range(root_gpa, 0x1000, 0);
        host.map_range(data_gpa, 0x200, 0);
        host.map_range(old_page, 0x1000, 0x11);
        host.map_range(new_page, 0x1000, 0x22);
        let mut d = [0u8; 8];
        st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 3);
        st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
        let _ = host.write_gpa(dir_gpa, &d);
        // root[0] = data page (object list + descriptors), root[1] = old backing.
        st32(&mut d[..4], 4);
        let _ = host.write_gpa(root_gpa, &d[..4]);
        st32(&mut d[..4], 5);
        let _ = host.write_gpa(root_gpa + 4, &d[..4]);
        assert!(state.define_task(1, 0x1000, 2));
        assert!(state.set_object_list(1, 0, 8));
        // Type-4 entry at surface_id=3, descriptor at GVA 0x80.
        let mut entry = [0u8; 12];
        st32(&mut entry[0..], 4u32 | (0x30u32 << 8));
        entry[4..12].copy_from_slice(&0x80u64.to_le_bytes());
        let _ = host.write_gpa(data_gpa + 3 * 12, &entry);
        let mut desc = vec![0u8; 0x30];
        st64(&mut desc[0..], 0x1000);
        st32(&mut desc[8..], 1); // backing_pfn = GVA page 1
        st32(&mut desc[0xc..], 0x50);
        desc[0x10] = 1;
        st32(&mut desc[0x18..], 16);
        st32(&mut desc[0x1c..], 16);
        st32(&mut desc[0x20..], 64);
        let _ = host.write_gpa(data_gpa + 0x80, &desc);

        assert!(resolve_type4_surface(&mut state, &host, 3));
        {
            let m = state.mappings.get(&3).unwrap();
            assert_eq!(m.page_entries.len(), 1);
            assert_eq!(
                entry_gpa_shift(m.page_entries[0], PAGE_SHIFT_X86),
                Some(old_page)
            );
            assert_eq!(m.map_generation, 1);
        }
        // Guest remaps GVA page 1 onto a new physical page (same id/geometry).
        st32(&mut d[..4], 6);
        let _ = host.write_gpa(root_gpa + 4, &d[..4]);
        assert!(resolve_type4_surface_force(&mut state, &host, 3));
        {
            let m = state.mappings.get(&3).unwrap();
            assert_eq!(
                entry_gpa_shift(m.page_entries[0], PAGE_SHIFT_X86),
                Some(new_page),
                "force-resolve must follow the moved translation"
            );
            assert_eq!(m.map_generation, 2, "page move bumps map_generation");
        }
        // Unchanged translation: force keeps the table without a rebuild.
        assert!(resolve_type4_surface_force(&mut state, &host, 3));
        let m = state.mappings.get(&3).unwrap();
        assert_eq!(m.map_generation, 2);
        assert_eq!(
            entry_gpa_shift(m.page_entries[0], PAGE_SHIFT_X86),
            Some(new_page)
        );
    }

    /// A genuine backing failure (a surface whose descriptor decoded fine but
    /// whose page-backing construction fails) must be fail-visible with a
    /// `reason=` slug, deduped per `(surface_id, reason)`, and re-armed when the
    /// surface next backs cleanly — never a silent `return false` that paints
    /// stale/black with no log. Locks the type-4 backing blind-spot closure.
    #[test]
    fn apply_type4_backing_fail_latches_reason_and_rearms() {
        let host = FakeHost::new();
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        state.page_shift = PAGE_SHIFT_X86;
        // A surface_id other type-4 tests do not touch (they use 3).
        let sid = 11u32;
        clear_type4_fail(sid);
        assert!(!type4_fail_latch()
            .lock()
            .unwrap()
            .contains(&(sid, "task_inactive")));
        // Small valid length (page_count = 1) so the alloc-guard passes, then an
        // undefined/inactive task_id hits the `task_inactive` site — the drain
        // race where a decoded surface's owning task died before backing landed.
        let surf = Type4Surface {
            length: 0x1000,
            backing_pfn: 0x20,
            pixel_format: 0,
            plane_count: 1,
            planes: [Type4Plane::default(); TYPE4_PLANE_CAP],
            width: 16,
            height: 16,
            bytes_per_row: 64,
        };
        assert!(!apply_type4_backing(&mut state, &host, 5, sid, &surf));
        assert!(
            type4_fail_latch()
                .lock()
                .unwrap()
                .contains(&(sid, "task_inactive")),
            "genuine backing failure must latch a reason slug"
        );
        // A clean backing on the same surface re-arms the latch.
        clear_type4_fail(sid);
        assert!(
            !type4_fail_latch()
                .lock()
                .unwrap()
                .contains(&(sid, "task_inactive")),
            "clear_type4_fail must re-arm so a later failure logs again"
        );
    }

    /// A task the guest has defined but never given an object list to must
    /// resolve **nothing** — not another task's list.
    ///
    /// This reproduces, at unit scale, what the rail was measured doing on every
    /// boot. `TaskEntry::define` used to invent `object_list_pfn = 1` and
    /// `count = 0x100000`, so a task with no `SetObjectList` still computed an
    /// entry address of `0x1000 + off`. Nothing is mapped there for that task,
    /// the walk failed `gva_zero_pfn`, and `read_task_gva_by_id` then walked
    /// task `5 >> 1 == 2`'s page table at the same address — where task 2's
    /// object list genuinely lives — and decoded task 2's entry as task 5's.
    ///
    /// Task 2's own lookup is asserted first so the fixture is known to be real:
    /// a test where the donor list is unreadable would pass for the wrong reason.
    #[test]
    fn a_task_with_no_object_list_resolves_nothing_not_its_neighbours_list() {
        let mut host = FakeHost::new();
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let dir_gpa = 2u64 << PAGE_SHIFT_X86;
        let root_gpa = 3u64 << PAGE_SHIFT_X86;
        let data_gpa = 4u64 << PAGE_SHIFT_X86;
        host.map_range(dir_gpa, 0x20, 0);
        host.map_range(root_gpa, 0x1000, 0);
        host.map_range(data_gpa, 0x1000, 0);
        let mut d = [0u8; 8];
        st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 3);
        st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
        let _ = host.write_gpa(dir_gpa, &d);
        // PTE for GVA page 1 (0x1000) → pfn 4, so task 2's list is readable.
        let mut pte = [0u8; 4];
        st32(&mut pte, 4);
        let _ = host.write_gpa(root_gpa + 4, &pte);

        let mut entry = [0u8; OBJECT_LIST_ENTRY_LEN];
        st32(
            &mut entry[0..],
            (OBJECT_TYPE_SURFACE as u32) | (0x40u32 << 8),
        );
        entry[4..12].copy_from_slice(&0xdead_0000u64.to_le_bytes());
        let _ = host.write_gpa(data_gpa, &entry);

        // Task 2 owns a real list at pfn 1. Task 5 has a directory that maps
        // nothing, and `5 >> 1 == 2`.
        assert!(state.define_task(2, 0x1000, 2));
        assert!(state.set_object_list(2, 1, 4));
        assert!(state.define_task(5, 0x1000, 9));

        let donor = lookup_list_entry(&state, &host, 2, 0);
        assert!(
            donor.is_some(),
            "fixture is not real: task 2's own list must be readable"
        );

        // The behavioural claim first, so a regression fails on the corruption
        // itself rather than on the field that causes it.
        assert_eq!(
            lookup_list_entry(&state, &host, 5, 0),
            None,
            "task 5 has no object list, so it must resolve nothing — returning \
             Some here is task 2's entry answering for task 5"
        );
        assert_eq!(
            state.tasks[5].object_list_pfn, 0,
            "a defined task has no list until SetObjectList says so"
        );
        assert_eq!(state.tasks[5].object_list_count, 0);
    }

    fn setup_type4_candidate(
        host: &mut FakeHost,
        state: &mut DeviceState,
        surface_id: u32,
        desc_gva: u64,
        desc_len: u32,
    ) -> u64 {
        let dir_gpa = 2u64 << PAGE_SHIFT_X86;
        let root_gpa = 3u64 << PAGE_SHIFT_X86;
        let data_gpa = 4u64 << PAGE_SHIFT_X86;
        host.map_range(dir_gpa, 0x20, 0);
        host.map_range(root_gpa, 0x1000, 0);
        host.map_range(data_gpa, 0x1000, 0);
        let mut d = [0u8; 8];
        st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 3);
        st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
        let _ = host.write_gpa(dir_gpa, &d);
        st32(&mut d[..4], 4);
        let _ = host.write_gpa(root_gpa, &d[..4]);
        assert!(state.define_task(1, 0x1000, 2));
        assert!(state.set_object_list(1, 0, surface_id + 1));

        let mut entry = [0u8; OBJECT_LIST_ENTRY_LEN];
        st32(
            &mut entry[0..],
            (OBJECT_TYPE_SURFACE as u32) | (desc_len << 8),
        );
        entry[4..12].copy_from_slice(&desc_gva.to_le_bytes());
        let entry_gpa = data_gpa + surface_id as u64 * OBJECT_LIST_ENTRY_LEN as u64;
        let _ = host.write_gpa(entry_gpa, &entry);
        data_gpa
    }

    /// Once task-scan lookup finds an actual type-4 candidate, descriptor read
    /// failure is no longer speculative: the surface has an owner but cannot get
    /// backing. It must be fail-visible with a stable reason slug.
    #[test]
    fn resolve_type4_candidate_logs_descriptor_read_failure() {
        let mut host = FakeHost::new();
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let sid = 17u32;
        clear_type4_fail(sid);
        let _ = setup_type4_candidate(&mut host, &mut state, sid, 0x3000, 0x30);

        assert!(!resolve_type4_surface(&mut state, &host, sid));
        assert!(
            type4_fail_latch()
                .lock()
                .unwrap()
                .contains(&(sid, "desc_read")),
            "surface-type candidate with unreadable descriptor must name desc_read"
        );
        clear_type4_fail(sid);
    }

    /// A readable but invalid type-4 descriptor used to fall through to the
    /// resolver tail with no site reason. Keep it fail-visible without logging
    /// absent/non-surface speculative probes.
    #[test]
    fn resolve_type4_candidate_logs_descriptor_decode_failure() {
        let mut host = FakeHost::new();
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let sid = 18u32;
        clear_type4_fail(sid);
        let data_gpa = setup_type4_candidate(&mut host, &mut state, sid, 0x80, 0x30);
        let bad_desc = vec![0u8; 0x30];
        let _ = host.write_gpa(data_gpa + 0x80, &bad_desc);

        assert!(!resolve_type4_surface(&mut state, &host, sid));
        assert!(
            type4_fail_latch()
                .lock()
                .unwrap()
                .contains(&(sid, "desc_decode")),
            "surface-type candidate with invalid descriptor must name desc_decode"
        );
        clear_type4_fail(sid);
    }

    /// Live wire bytes (boot 093019 `compute_stage_tex type5 … args_hex`):
    /// R8 1024×1024 = Y plane view of a biplanar 1024×1024 surface.
    #[test]
    fn decode_type5_texture_view_live_r8_y_plane() {
        let mut desc = vec![0u8; 8];
        st32(&mut desc[TYPE5_SURFACE_ID..], 8);
        // args blob: kind 0x2f, len 0x30, own_ref 0x15, record R8 1024×1024 d=1.
        let args = [
            0x2fu8, 0, 0, 0, 0x30, 0, 0, 0, 0x15, 0, 0, 0, // kind, blob_len, own_ref
            0x42, 0x01, 0x0a, 0x00, // tag, unk, fmt=R8
            0x00, 0x04, 0x00, 0x00, // width 1024
            0x00, 0x04, 0x00, 0x00, // height 1024
            0x01, 0x00, 0x00, 0x00, // depth 1
            0x01, 0x00, 0x01, 0x00, 0x01, 0x00, 0x10, 0x00, // trailer (unconsumed)
        ];
        desc.extend_from_slice(&args);
        let rec = decode_type5_texture_view(&desc).expect("live R8 record decodes");
        assert_eq!(rec.pixel_format, 0x0a);
        assert_eq!((rec.width, rec.height, rec.depth), (1024, 1024, 1));
        // Short record (no +0x20 field) defaults to plane 0.
        assert_eq!(rec.plane_index, 0);
    }

    /// Live 56-byte wire blob from the BLIT copy-source path (x86 Ventura
    /// 13.7.8, 2026-07-19 `blit t5_view_decode sid=34`): a full-color
    /// texture view (BGRA8_sRGB 1024×768 window backing) carries the sibling
    /// record tag `0x62`, not the biplanar `0x42`. Same field layout — must
    /// decode, or the blit path drops the copy.
    #[test]
    fn decode_type5_texture_view_live_0x62_color_window_view() {
        // Exact leading 40 bytes observed, zero-padded to the 56-byte desc_len.
        let head: [u8; 40] = [
            0x22, 0x00, 0x00, 0x00, // surface_id = 34
            0x00, 0x00, 0x00, 0x00, // field
            0x2f, 0x00, 0x00, 0x00, // kind 0x2f
            0x30, 0x00, 0x00, 0x00, // blob_len 0x30
            0x0b, 0x00, 0x00, 0x00, // own_ref 0x0b
            0x62, 0x00, 0x51, 0x00, // tag=0x62, unk, fmt=0x51 BGRA8_sRGB
            0x00, 0x04, 0x00, 0x00, // width 1024
            0x00, 0x03, 0x00, 0x00, // height 768
            0x01, 0x00, 0x00, 0x00, // depth 1
            0x01, 0x00, 0x01, 0x00, // trailer
        ];
        let mut desc = head.to_vec();
        desc.resize(56, 0); // plane field (+0x20 in record) reads 0
        let rec = decode_type5_texture_view(&desc).expect("0x62 color view must decode");
        assert_eq!(rec.pixel_format, 0x51);
        assert_eq!((rec.width, rec.height, rec.depth), (1024, 768, 1));
        assert_eq!(rec.plane_index, 0);
    }

    /// Live 56-byte wire blob (boot 20260717-063043, v0a8 hero): the record
    /// carries the `newTextureWithDescriptor:iosurface:plane:` plane at
    /// `+0x20` — Y views carry 0, the RG8 chroma view 1, the same-geometry
    /// alpha view 2. Geometry cannot separate Y from alpha; this field does.
    #[test]
    fn decode_type5_texture_view_live_v0a8_alpha_plane_index() {
        let mut desc = vec![0u8; 8];
        st32(&mut desc[TYPE5_SURFACE_ID..], 0x6d);
        let args = [
            0x2fu8, 0, 0, 0, 0x30, 0, 0, 0, 0x82, 0x01, 0, 0, // kind, blob_len, own_ref
            0x42, 0x01, 0x0a, 0x00, // tag, unk, fmt=R8
            0xb2, 0x03, 0x00, 0x00, // width 946
            0x5e, 0x01, 0x00, 0x00, // height 350
            0x01, 0x00, 0x00, 0x00, // depth 1
            0x01, 0x00, 0x01, 0x00, 0x01, 0x00, 0x10, 0x00, // trailer
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // reserved
            0x02, 0x00, 0x00, 0x00, // IOSurface plane index = 2 (alpha)
        ];
        desc.extend_from_slice(&args);
        let rec = decode_type5_texture_view(&desc).expect("live v0a8 alpha record decodes");
        assert_eq!(rec.pixel_format, 0x0a);
        assert_eq!((rec.width, rec.height, rec.depth), (946, 350, 1));
        assert_eq!(rec.plane_index, 2);
    }

    #[test]
    fn decode_type5_texture_view_fail_closed() {
        // Short descriptor (no record).
        let mut short = vec![0u8; 8];
        st32(&mut short[TYPE5_SURFACE_ID..], 8);
        assert!(decode_type5_texture_view(&short).is_none());
        // Wrong record tag.
        let mut bad_tag = vec![0u8; 8];
        st32(&mut bad_tag[TYPE5_SURFACE_ID..], 8);
        bad_tag.extend_from_slice(&[0u8; 12]);
        bad_tag.extend_from_slice(&[
            0x41, 0x01, 0x0a, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x01, 0, 0, 0,
        ]);
        assert!(decode_type5_texture_view(&bad_tag).is_none());
        // Non-2D (depth != 1) fails closed.
        let mut vol = vec![0u8; 8];
        st32(&mut vol[TYPE5_SURFACE_ID..], 8);
        vol.extend_from_slice(&[0u8; 12]);
        vol.extend_from_slice(&[
            0x42, 0x07, 0x50, 0x00, 0x40, 0, 0, 0, 0x40, 0, 0, 0, 0x40, 0, 0, 0,
        ]);
        assert!(decode_type5_texture_view(&vol).is_none());
        // Zero width fails closed.
        let mut zw = vec![0u8; 8];
        st32(&mut zw[TYPE5_SURFACE_ID..], 8);
        zw.extend_from_slice(&[0u8; 12]);
        zw.extend_from_slice(&[
            0x42, 0x01, 0x0a, 0x00, 0, 0, 0, 0, 0x00, 0x04, 0, 0, 0x01, 0, 0, 0,
        ]);
        assert!(decode_type5_texture_view(&zw).is_none());
    }

    /// The probe's notion of "undecoded" must be exactly the bytes
    /// `decode_type4_surface` skips, and it must distinguish two surfaces on
    /// those bytes alone.
    ///
    /// This is the measurement that blocks the largest deletion in the present
    /// path: nothing decoded at surface-create time separates a desktop
    /// swapchain buffer from a same-geometry offscreen tile, so membership is
    /// reconstructed by half a dozen downstream mechanisms. If the guest is
    /// telling us in the undecoded span, the probe has to be able to see it.
    #[test]
    fn undecoded_type4_span_is_exactly_what_the_decoder_skips() {
        // One plane: the decoder consumes 0x14..0x24, so the tail starts there.
        let mut a = vec![0u8; 0x40];
        st64(&mut a[TYPE4_LEN..], 0x800000);
        st32(&mut a[TYPE4_BACKING_PFN..], 0x1234);
        st32(&mut a[TYPE4_PIXEL_FORMAT..], 0x4247_5241); // 'BGRA'
        a[TYPE4_PLANE_COUNT] = 1;
        st32(&mut a[TYPE4_PLANES..], 0); // plane0 offset
        st32(&mut a[TYPE4_PLANES + 4..], 1920);
        st32(&mut a[TYPE4_PLANES + 8..], 1080);
        st32(&mut a[TYPE4_PLANES + 12..], 1920 * 4);

        // Every decoded field can change without moving the undecoded span.
        let mut b = a.clone();
        st64(&mut b[TYPE4_LEN..], 0x900000);
        st32(&mut b[TYPE4_BACKING_PFN..], 0x9999);
        st32(&mut b[TYPE4_PIXEL_FORMAT..], 0x4c31_3062);
        st32(&mut b[TYPE4_PLANES + 4..], 1280);
        st32(&mut b[TYPE4_PLANES + 8..], 720);
        st32(&mut b[TYPE4_PLANES + 12..], 1280 * 4);
        assert_eq!(
            undecoded_type4_surface_bytes(&a),
            undecoded_type4_surface_bytes(&b),
            "changing only decoded fields must not look like a new shape"
        );

        // The span covers the three bytes after plane_count and the whole tail
        // past the plane records the decoder consumed.
        for probe in [0x11usize, 0x13, 0x24, 0x3f] {
            let mut c = a.clone();
            c[probe] ^= 0xff;
            assert_ne!(
                undecoded_type4_surface_bytes(&a),
                undecoded_type4_surface_bytes(&c),
                "byte {probe:#x} is undecoded and must be visible to the probe"
            );
        }

        // Bytes the decoder DOES read must not be in the span, or ordinary
        // surface-to-surface variation would look like a new shape forever.
        // `plane_count` (+0x10) is excluded on purpose: it is decoded AND it
        // moves the span's own boundary, which the two-plane case below pins.
        for probe in [0x00usize, 0x08, 0x0c, 0x14, 0x23] {
            let mut c = a.clone();
            c[probe] ^= 0xff;
            assert_eq!(
                undecoded_type4_surface_bytes(&a),
                undecoded_type4_surface_bytes(&c),
                "byte {probe:#x} is decoded and must stay out of the span"
            );
        }

        // A second plane moves the boundary: 0x24..0x34 becomes decoded.
        let mut two = a.clone();
        two[TYPE4_PLANE_COUNT] = 2;
        assert_eq!(
            undecoded_type4_surface_bytes(&two).len(),
            undecoded_type4_surface_bytes(&a).len() - TYPE4_PLANE_STRIDE,
            "the span shrinks by exactly one plane record"
        );

        // A record too short to decode reports nothing rather than a partial
        // span that would compare unequal against every real one.
        assert!(undecoded_type4_surface_bytes(&a[..TYPE4_MIN_LEN - 1]).is_empty());
    }
}
