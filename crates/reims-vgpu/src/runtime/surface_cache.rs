//! Host surface cache for Linux/Vulkan discrete-GPU present (kb tahoe-x86 §8.5).
//!
//! On Apple Metal hosts, GPU Stores land in guest IOSurface pages (unified
//! memory). On this Linux product rail guest type-4 pages are **not** filled by
//! the host GPU until encode writeback; historical product painted from a
//! **host render-cache** keyed by surface_id. This module is that cache.
//!
//! Namespace split (2026-07-13 live x86):
//! - [`store`] / [`get`] — **type-4 surface_id / mapping_id** only (`host_surfaces`)
//! - [`store_texture`] / [`get_texture`] — type-2/3 color targets by object ref
//! - [`store_gva_owned`] / [`get_gva`] — type-2/3 by target GVA (survives ref rebinding)
//!
//! Never put texture_ref into `host_surfaces`: list ids collide with mids and
//! recycled refs return stale full-frame blacks as multi-bind samples.
//!
//! Write paths (clear Store, metal2vulkan encode writeback) call the matching
//! store; [`crate::runtime::scanout::capture_present_frame`] prefers surface_id
//! cache content when present so scanout matches what the host executed.

use crate::contract::pixel_format::RGBA8_BPP;
use crate::model::{DeviceState, GvaBacking, HostSurface, MAX_SCANOUT_DIM};
use crate::runtime::host::HostMemory;

/// `generation` is issued by
/// [`DeviceState::next_sampled_content_generation`] and is never derived from
/// the entry being replaced: an entry-local counter restarts whenever the entry
/// is re-created, and half of this cache's identity contract is that a
/// generation names one content for the life of the device.
fn store_into(
    map: &mut std::collections::BTreeMap<u32, HostSurface>,
    id: u32,
    width: u32,
    height: u32,
    bgra: std::sync::Arc<Vec<u8>>,
    generation: u64,
) {
    if id == 0 || width == 0 || height == 0 || width > MAX_SCANOUT_DIM || height > MAX_SCANOUT_DIM {
        return;
    }
    let need = (height as usize)
        .saturating_mul(width as usize)
        .saturating_mul(RGBA8_BPP as usize);
    if bgra.len() < need {
        return;
    }
    let entry = map.entry(id).or_default();
    entry.host_gen = generation;
    entry.width = width;
    entry.height = height;
    entry.bgra = bgra;
}

fn get_from(
    map: &std::collections::BTreeMap<u32, HostSurface>,
    id: u32,
    width: u32,
    height: u32,
) -> Option<&[u8]> {
    get_from_with_gen(map, id, width, height).map(|(bgra, _)| bgra)
}

fn get_from_with_gen(
    map: &std::collections::BTreeMap<u32, HostSurface>,
    id: u32,
    width: u32,
    height: u32,
) -> Option<(&[u8], u64)> {
    let e = map.get(&id)?;
    if e.width != width || e.height != height || e.bgra.is_empty() {
        return None;
    }
    let need = (height as usize)
        .saturating_mul(width as usize)
        .saturating_mul(RGBA8_BPP as usize);
    if e.bgra.len() < need {
        return None;
    }
    Some((&e.bgra[..need], e.host_gen))
}

/// Insert/replace host-cache pixels for `surface_id` (type-4 present id).
pub fn store(state: &mut DeviceState, surface_id: u32, width: u32, height: u32, bgra: Vec<u8>) {
    store_shared(state, surface_id, width, height, std::sync::Arc::new(bgra));
}

/// [`store`] for a frame already held behind an `Arc` — the type-11 render Store
/// arms its deferred window with the same allocation, so the frame is stored
/// once and referenced twice.
pub fn store_shared(
    state: &mut DeviceState,
    surface_id: u32,
    width: u32,
    height: u32,
    bgra: std::sync::Arc<Vec<u8>>,
) {
    let generation = state.next_sampled_content_generation();
    store_into(
        &mut state.host_surfaces,
        surface_id,
        width,
        height,
        bgra,
        generation,
    );
}

/// [`store`] from rows the caller only has as a borrow, reusing the entry's own
/// allocation whenever nothing else is looking at it.
///
/// The caller that mattered is the deferred render writeback, which reaches here
/// holding a frame it does not own and had to build a whole second copy of just
/// to call [`store`]. On the composite surface that is a fresh 8.29 MB
/// `Vec` about 95 times a second, and the allocation is the expensive half, not
/// the copy: a multi-megabyte `vec![0u8; n]` comes back as untouched pages and
/// the fill faults every one of them in, then the buffer is dropped and the next
/// flush does it again. Measured at 1.21 ms per flush — 32% of the writeback,
/// against 0.72 ms for landing the frame in the guest's pages.
///
/// Reuse is conditional on [`std::sync::Arc::get_mut`], so it happens only when
/// the strong count is exactly one and no window, sampled binding or present
/// capture can observe the mutation. When something does hold the frame — a
/// [`crate::model::DeferredOwner::Render`] window armed on this allocation, the
/// case the `Arc` exists for — this allocates as before and the old bytes stay
/// intact for their holder.
pub fn store_rows(
    state: &mut DeviceState,
    surface_id: u32,
    width: u32,
    height: u32,
    src: &[u8],
    src_stride: u32,
) {
    let generation = state.next_sampled_content_generation();
    if surface_id == 0
        || width == 0
        || height == 0
        || width > MAX_SCANOUT_DIM
        || height > MAX_SCANOUT_DIM
    {
        return;
    }
    let row = (width as usize).saturating_mul(RGBA8_BPP as usize);
    let need = (height as usize).saturating_mul(row);
    if need == 0 || src.len() < (height as usize).saturating_mul(src_stride as usize) {
        return;
    }
    let entry = state.host_surfaces.entry(surface_id).or_default();
    match std::sync::Arc::get_mut(&mut entry.bgra) {
        Some(buf) if buf.len() == need => fill_tight_rows(buf, src, src_stride, row, height),
        _ => {
            let mut buf = vec![0u8; need];
            fill_tight_rows(&mut buf, src, src_stride, row, height);
            entry.bgra = std::sync::Arc::new(buf);
        }
    }
    entry.host_gen = generation;
    entry.width = width;
    entry.height = height;
}

/// Copy `height` rows of `row` bytes out of `src` at `src_stride` pitch into a
/// tightly packed `dst`.
///
/// One `copy_from_slice` when the source is already tight, which is the shape
/// every readback arrives in — a per-row loop over an 8 MB frame is 1079 calls
/// that a single memcpy expresses exactly.
fn fill_tight_rows(dst: &mut [u8], src: &[u8], src_stride: u32, row: usize, height: u32) {
    if src_stride as usize == row {
        let n = dst.len().min(src.len());
        dst[..n].copy_from_slice(&src[..n]);
        return;
    }
    for y in 0..height as usize {
        let so = y.saturating_mul(src_stride as usize);
        let doff = y.saturating_mul(row);
        if so + row <= src.len() && doff + row <= dst.len() {
            dst[doff..doff + row].copy_from_slice(&src[so..so + row]);
        }
    }
}

/// Borrow host-cache frame when geom matches request (surface_id namespace).
pub fn get(state: &DeviceState, surface_id: u32, width: u32, height: u32) -> Option<&[u8]> {
    get_from(&state.host_surfaces, surface_id, width, height)
}

/// [`get_shared`], plus the generation that names these exact bytes.
///
/// The generation is the entry's `host_gen`, and it is a sampled-content
/// identity rather than provenance: every writer of this map — [`store_into`],
/// [`store_rows`] and [`cede_surface_to_resident`] — takes a fresh
/// [`DeviceState::next_sampled_content_generation`] in the same breath as it
/// changes the bytes, and [`forget`] removes the entry outright. So a repeated
/// `(surface_id, generation)` pair is a statement that the bytes have not moved,
/// which is what lets the sampled cache skip re-hashing a frame it already holds.
///
/// The caller is the type-11 sampled ladder's host-cache rung, which without
/// this had no identity to offer and drove every bind through the content
/// digest: 116 lookups a second over 201 MB, hashed twice each. It takes the
/// frame as a handle rather than a slice because it hands the bytes to the
/// engine, which outlives the borrow of `state` — so the rung costs a refcount
/// and not a full-frame copy.
pub fn get_shared_with_gen(
    state: &DeviceState,
    surface_id: u32,
    width: u32,
    height: u32,
) -> Option<(std::sync::Arc<Vec<u8>>, u64)> {
    let (_, host_gen) = get_from_with_gen(&state.host_surfaces, surface_id, width, height)?;
    // Deliberately delegated rather than reimplemented: `get_shared` owns the
    // rule for a stored buffer carrying slop past `width * height * 4`, and a
    // second copy of that rule here could drift from it.
    Some((get_shared(state, surface_id, width, height)?, host_gen))
}

/// Cede this mapping's cached frame to the engine resident a deferred type-11
/// render Store just pinned: the entry keeps its geometry and its `host_gen`
/// lineage, and holds no bytes.
///
/// The emptiness **is** the cession, and [`get_from`]'s `bgra.is_empty()` gate is
/// what enforces it — so every reader that goes through [`get`] or [`get_shared`]
/// misses and falls through to the source that does hold the frame:
/// [`crate::runtime::scanout::capture_present_frame`] to
/// `try_capture_from_resident`, and the type-11 LOAD seed to the surface's own
/// guest pages, which lands this window first. Nothing has to be taught about a
/// new state.
///
/// Retaining the stale bytes as a fallback would be worse than missing. A Store
/// that skipped its readback has already superseded them, and a consumer served
/// the previous frame renders a whole compositing layer one frame behind with no
/// report — which is the class `deferred_flush_lost reason=cache_miss` cost 15
/// layers in one boot to close.
///
/// Returns false for a geometry this cache would not have stored anyway, so the
/// caller can refuse to arm rather than leave a live entry contradicting a
/// resident-authoritative window.
pub fn cede_surface_to_resident(
    state: &mut DeviceState,
    surface_id: u32,
    width: u32,
    height: u32,
) -> bool {
    if surface_id == 0
        || width == 0
        || height == 0
        || width > MAX_SCANOUT_DIM
        || height > MAX_SCANOUT_DIM
    {
        return false;
    }
    let generation = state.next_sampled_content_generation();
    let entry = state.host_surfaces.entry(surface_id).or_default();
    entry.host_gen = generation;
    entry.width = width;
    entry.height = height;
    entry.bgra = std::sync::Arc::new(Vec::new());
    true
}

/// Drop this mapping's cache entry outright.
///
/// Distinct from [`cede_surface_to_resident`], and the difference is which
/// source the reader is being sent to. A cession says "the engine resident holds
/// this frame"; this says "nothing host-side does — read the surface's own
/// pages". It is what a writeback that deliberately left some of the guest's own
/// bytes in place has to do, because after one of those neither the cache nor the
/// resident holds the mapping's content: they hold the frame the device rendered,
/// and the pages hold that frame with the guest's stores still in it.
///
/// Removes rather than emptying, so [`surface_ceded_to_resident`] does not read
/// the result as a cession and report a decline that names the wrong source.
pub fn forget(state: &mut DeviceState, surface_id: u32) {
    state.host_surfaces.remove(&surface_id);
}

/// Whether this mapping's cache entry is the ceded shell
/// [`cede_surface_to_resident`] leaves behind: present at exactly this geometry
/// and carrying no bytes.
///
/// Read by the type-11 LOAD seed's decline classifier so a ceded entry is named
/// as such instead of being reported as a stale-geometry hit — `get`'s miss is
/// the same either way, and the two have different fixes.
pub fn surface_ceded_to_resident(
    state: &DeviceState,
    surface_id: u32,
    width: u32,
    height: u32,
) -> bool {
    state
        .host_surfaces
        .get(&surface_id)
        .is_some_and(|e| e.bgra.is_empty() && e.width == width && e.height == height)
}

/// [`get`] as a shared handle, for a caller that needs to own the frame past the
/// borrow of `state` — a Load seed does, and taking it this way costs a refcount
/// rather than a full-framebuffer copy.
///
/// Hits exactly when [`get`] hits. A handle cannot be truncated the way [`get`]'s
/// slice is, so the refcount is only taken when the stored buffer is *exactly*
/// `width * height * 4`; a store carrying slop past that is copied instead.
///
/// The copy is not reachable today — every producer of `host_surfaces` allocates
/// exactly that — but returning `None` there would be a silent seed loss, and a
/// missing Load seed renders the pass onto a cleared target, which is a
/// compositing layer going solid black. Matching [`get`] means a future producer
/// with slop costs a copy rather than a defect.
pub fn get_shared(
    state: &DeviceState,
    surface_id: u32,
    width: u32,
    height: u32,
) -> Option<std::sync::Arc<Vec<u8>>> {
    let need = get_from(&state.host_surfaces, surface_id, width, height)?.len();
    let e = state.host_surfaces.get(&surface_id)?;
    Some(if e.bgra.len() == need {
        std::sync::Arc::clone(&e.bgra)
    } else {
        std::sync::Arc::new(e.bgra[..need].to_vec())
    })
}

/// Type-2/3 encode cache by texture object ref (not surface_id).
///
/// `source_gva` is the address the producing Store rendered into, kept so
/// [`texture_source_gva`] can tell a later serve whether this entry's pixels
/// came from the allocation it is about to be served as. Zero when the producer
/// had no GVA.
pub fn store_texture(
    state: &mut DeviceState,
    texture_ref: u32,
    width: u32,
    height: u32,
    bgra: Vec<u8>,
    source_gva: u64,
) {
    let generation = state.next_sampled_content_generation();
    store_into(
        &mut state.host_texture_surfaces,
        texture_ref,
        width,
        height,
        std::sync::Arc::new(bgra),
        generation,
    );
    if let Some(e) = state.host_texture_surfaces.get_mut(&texture_ref) {
        e.source_gva = source_gva;
    }
}

pub fn get_texture(
    state: &DeviceState,
    texture_ref: u32,
    width: u32,
    height: u32,
) -> Option<&[u8]> {
    get_from(&state.host_texture_surfaces, texture_ref, width, height)
}

/// The address the entry behind [`get_texture`] was produced over, or `None`
/// when no entry answers at this geometry.
///
/// Separate from [`get_texture`] rather than returned beside it because the
/// serve site holds a shared borrow of `state` across the pixel slice, and the
/// question is asked once per serve while the slice is used per texel.
pub fn texture_source_gva(
    state: &DeviceState,
    texture_ref: u32,
    width: u32,
    height: u32,
) -> Option<u64> {
    get_from(&state.host_texture_surfaces, texture_ref, width, height)?;
    state
        .host_texture_surfaces
        .get(&texture_ref)
        .map(|e| e.source_gva)
}

pub fn evict_texture(state: &mut DeviceState, texture_ref: u32) {
    state.host_texture_surfaces.remove(&texture_ref);
}

/// Store tight raw compute content for a type-2/3 texture object.
///
/// This is the discrete GPU-private body. It deliberately survives
/// MapMemory2/UnmapMemory; the guest GVA pages are only a pageable alias.
#[allow(clippy::too_many_arguments)]
pub fn store_linear_texture(
    state: &mut DeviceState,
    task_id: u32,
    texture_ref: u32,
    gva: u64,
    pixel_format: u16,
    width: u32,
    height: u32,
    row_stride: u64,
    bytes: &[u8],
) -> bool {
    let Some(bpp) = crate::contract::pixel_format::bytes_per_pixel(pixel_format) else {
        return false;
    };
    let Some(need) = (width as usize)
        .checked_mul(height as usize)
        .and_then(|n| n.checked_mul(bpp as usize))
    else {
        return false;
    };
    if texture_ref == 0
        || gva == 0
        || width == 0
        || height == 0
        || row_stride < (width as u64).saturating_mul(bpp as u64)
        || bytes.len() < need
    {
        return false;
    }
    let entry = state
        .host_linear_textures
        .entry((task_id, texture_ref))
        .or_default();
    entry.host_gen = entry.host_gen.wrapping_add(1);
    if entry.host_gen == 0 {
        entry.host_gen = 1;
    }
    entry.gva = gva;
    entry.pixel_format = pixel_format;
    entry.width = width;
    entry.height = height;
    entry.row_stride = row_stride;
    entry.bytes.clear();
    entry.bytes.extend_from_slice(&bytes[..need]);
    entry.resident_gen = 0;
    true
}

/// Deferred linear writeback: the engine's pinned resident storage image at
/// `generation` becomes the authoritative content for this (task, ref) window;
/// no bytes are stored. Same validation as [`store_linear_texture`].
#[allow(clippy::too_many_arguments)]
pub fn note_linear_texture_resident(
    state: &mut DeviceState,
    task_id: u32,
    texture_ref: u32,
    gva: u64,
    pixel_format: u16,
    width: u32,
    height: u32,
    row_stride: u64,
    generation: u32,
) -> bool {
    let Some(bpp) = crate::contract::pixel_format::bytes_per_pixel(pixel_format) else {
        return false;
    };
    if texture_ref == 0
        || gva == 0
        || width == 0
        || height == 0
        || generation == 0
        || row_stride < (width as u64).saturating_mul(bpp as u64)
    {
        return false;
    }
    let entry = state
        .host_linear_textures
        .entry((task_id, texture_ref))
        .or_default();
    entry.host_gen = generation;
    entry.gva = gva;
    entry.pixel_format = pixel_format;
    entry.width = width;
    entry.height = height;
    entry.row_stride = row_stride;
    entry.bytes.clear();
    entry.resident_gen = generation;
    true
}

/// Resident generation of a linear window when the current descriptor still
/// matches and the entry is resident-authoritative (deferred writeback).
#[allow(clippy::too_many_arguments)]
pub fn linear_texture_resident_gen(
    state: &DeviceState,
    task_id: u32,
    texture_ref: u32,
    gva: u64,
    pixel_format: u16,
    width: u32,
    height: u32,
    row_stride: u64,
) -> Option<u32> {
    let entry = state.host_linear_textures.get(&(task_id, texture_ref))?;
    if entry.resident_gen == 0
        || entry.gva != gva
        || entry.pixel_format != pixel_format
        || entry.width != width
        || entry.height != height
        || entry.row_stride != row_stride
    {
        return None;
    }
    Some(entry.resident_gen)
}

/// Why [`materialize_linear_resident`] did not land the flushed bytes.
///
/// The distinction the caller needs is whether the cache entry ends up holding
/// this frame. `Superseded` is the one arm where it does not matter — the defer
/// has already been overtaken, so there is no frame here to lose. Every other
/// arm means a live entry stayed empty, and `flush_linear_one` then decides
/// whether to write guest pages on the assumption that it did not.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinearMaterializeDecline {
    /// The (task, ref) entry is gone, or a newer defer replaced this
    /// generation. Expected control flow: the content this flush carries is no
    /// longer the content anything would read.
    Superseded { resident_gen: u32 },
    /// The entry's own format has no bytes-per-pixel, so the tight size it
    /// wants cannot be computed.
    FormatUnsized { pixel_format: u16 },
    /// `width * height * bpp` overflowed `usize`.
    TightSizeOverflow { width: u32, height: u32, bpp: u32 },
    /// The engine returned fewer bytes than one tight image.
    ReadbackShort { got: usize, need: usize },
}

impl crate::observe::Decline for LinearMaterializeDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::Superseded { .. } => "linear_materialize_superseded",
            Self::FormatUnsized { .. } => "linear_materialize_format_unsized",
            Self::TightSizeOverflow { .. } => "linear_materialize_tight_size_overflow",
            Self::ReadbackShort { .. } => "linear_materialize_readback_short",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::Superseded { resident_gen } => {
                vec![("resident_gen", resident_gen.to_string())]
            }
            Self::FormatUnsized { pixel_format } => {
                vec![("pixel_format", format!("{pixel_format:#x}"))]
            }
            Self::TightSizeOverflow { width, height, bpp } => vec![
                ("width", width.to_string()),
                ("height", height.to_string()),
                ("bpp", bpp.to_string()),
            ],
            Self::ReadbackShort { got, need } => {
                vec![("got", got.to_string()), ("need", need.to_string())]
            }
        }
    }
}

/// Land flushed resident bytes into the entry (tight rows), clearing the
/// resident-authoritative marker.
///
/// The `Err` is load-bearing rather than informational. `flush_linear_one` may
/// go on to refuse the guest write, and it is allowed to do that only because
/// this call left the authoritative bytes in the cache — so a caller that
/// cannot tell success from failure is asserting the one thing it needs to
/// know. Every arm but `Superseded` means a live entry was left empty.
pub fn materialize_linear_resident(
    state: &mut DeviceState,
    task_id: u32,
    texture_ref: u32,
    generation: u32,
    bytes: &[u8],
) -> Result<(), LinearMaterializeDecline> {
    let Some(entry) = state.host_linear_textures.get_mut(&(task_id, texture_ref)) else {
        return Err(LinearMaterializeDecline::Superseded { resident_gen: 0 });
    };
    if entry.resident_gen != generation {
        return Err(LinearMaterializeDecline::Superseded {
            resident_gen: entry.resident_gen,
        });
    }
    let Some(bpp) = crate::contract::pixel_format::bytes_per_pixel(entry.pixel_format) else {
        return Err(LinearMaterializeDecline::FormatUnsized {
            pixel_format: entry.pixel_format,
        });
    };
    let Some(need) = (entry.width as usize)
        .checked_mul(entry.height as usize)
        .and_then(|n| n.checked_mul(bpp as usize))
    else {
        return Err(LinearMaterializeDecline::TightSizeOverflow {
            width: entry.width,
            height: entry.height,
            bpp,
        });
    };
    if bytes.len() < need {
        return Err(LinearMaterializeDecline::ReadbackShort {
            got: bytes.len(),
            need,
        });
    }
    entry.bytes.clear();
    entry.bytes.extend_from_slice(&bytes[..need]);
    entry.resident_gen = 0;
    Ok(())
}

/// Borrow a raw compute encode only when the current descriptor still matches.
#[allow(clippy::too_many_arguments)]
pub fn get_linear_texture(
    state: &DeviceState,
    task_id: u32,
    texture_ref: u32,
    gva: u64,
    pixel_format: u16,
    width: u32,
    height: u32,
    row_stride: u64,
) -> Option<&[u8]> {
    let entry = state.host_linear_textures.get(&(task_id, texture_ref))?;
    if entry.gva != gva
        || entry.pixel_format != pixel_format
        || entry.width != width
        || entry.height != height
        || entry.row_stride != row_stride
    {
        return None;
    }
    let bpp = crate::contract::pixel_format::bytes_per_pixel(pixel_format)? as usize;
    let need = (width as usize)
        .checked_mul(height as usize)?
        .checked_mul(bpp)?;
    (entry.bytes.len() >= need).then(|| &entry.bytes[..need])
}

/// True when [`mirror_linear_color_cache`] would republish this format into
/// the BGRA render-sample caches. Deferred linear writebacks are gated on
/// `!linear_mirrorable` so render-side consumers never lose the mirror.
pub fn linear_mirrorable(pixel_format: u16) -> bool {
    use crate::contract::pixel_format::{
        MTL_FORMAT_BGRA8_UNORM, MTL_FORMAT_BGRA8_UNORM_SRGB, MTL_FORMAT_RGBA8_UNORM,
        MTL_FORMAT_RGBA8_UNORM_SRGB,
    };
    matches!(
        pixel_format,
        MTL_FORMAT_RGBA8_UNORM
            | MTL_FORMAT_RGBA8_UNORM_SRGB
            | MTL_FORMAT_BGRA8_UNORM
            | MTL_FORMAT_BGRA8_UNORM_SRGB
    )
}

/// Mirror normalized 8-bit compute output into the established BGRA sample
/// caches so a later render view over the same object/GVA observes the encode.
#[allow(
    clippy::too_many_arguments,
    reason = "the mirrored identity is the object, GVA, format, geometry, and guest backing"
)]
pub fn mirror_linear_color_cache<M: HostMemory + crate::runtime::host::HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    texture_ref: u32,
    gva: u64,
    pixel_format: u16,
    width: u32,
    height: u32,
    bytes: &[u8],
) {
    use crate::contract::pixel_format::{
        MTL_FORMAT_BGRA8_UNORM, MTL_FORMAT_BGRA8_UNORM_SRGB, MTL_FORMAT_RGBA8_UNORM,
        MTL_FORMAT_RGBA8_UNORM_SRGB,
    };
    let Some(need) = (width as usize)
        .checked_mul(height as usize)
        .and_then(|n| n.checked_mul(4))
    else {
        return;
    };
    if bytes.len() < need {
        return;
    }
    let mut bgra = bytes[..need].to_vec();
    match pixel_format {
        MTL_FORMAT_RGBA8_UNORM | MTL_FORMAT_RGBA8_UNORM_SRGB => {
            for px in bgra.chunks_exact_mut(4) {
                px.swap(0, 2);
            }
        }
        MTL_FORMAT_BGRA8_UNORM | MTL_FORMAT_BGRA8_UNORM_SRGB => {}
        _ => return,
    }
    store_texture(state, texture_ref, width, height, bgra.clone(), gva);
    let backing = gva_backing(state, host, task_id, gva, width, height);
    store_gva_owned(state, gva, width, height, bgra, 0, backing);
}

/// The guest page currently backing `gva` under `task_id`, page-aligned.
///
/// Returns `None` when the walk cannot name the backing at all — a zero or
/// degenerate geometry, a dead task, or an address that does not translate. A
/// `None` backing means the entry is simply not validatable, never that it is
/// fresh.
///
/// The same call [`gva_backing_state`] makes to check the entry later, so the
/// producer and the consumer cannot disagree about what names an allocation.
pub fn gva_backing<M: HostMemory>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    gva: u64,
    width: u32,
    height: u32,
) -> Option<GvaBacking> {
    if gva == 0 || width == 0 || height == 0 {
        return None;
    }
    // Resolved by slot index, which is what the dense walk this replaced did
    // (`visit_task_gva_pages`) and what `gva_backing_state` does when it
    // re-asks. `translate_task_gva` applies the `active`/`directory_pfn` test
    // itself.
    let task = state.tasks.get(task_id as usize)?;
    let gpa = crate::runtime::gva_mem::translate_task_gva(host, task, gva, state.page_shift)?;
    Some(GvaBacking {
        task_id,
        first_gpa: gpa & page_mask(state.page_shift),
    })
}

/// Mask that clears the page offset for this guest page geometry.
const fn page_mask(page_shift: u32) -> u64 {
    !((1u64 << page_shift) - 1)
}

/// Store a type-2/3 encode in the GVA-keyed cache, with the decoded object
/// identity that produced it.
///
/// Type-2/type-3 wrappers are the same linear texture storage family when the
/// GVA and geometry match; unrelated nonzero object-type transitions still
/// identify a different resource class.
///
/// On discrete hosts this cache is the **GPU-private** texture content for that
/// VA. Guest MapMemory2 unmap/remap changes PFNs under the same GVA but does
/// **not** destroy the encode: nothing on the Unmap path touches this map,
/// deliberately — an unmapped VA is the normal state of the wallpaper class this
/// cache holds. [`gva_backing_state`] is what says whether the key still names
/// these pages.
#[allow(
    clippy::too_many_arguments,
    reason = "the cache identity is the GVA, its geometry, its producer, and its guest backing"
)]
pub fn store_gva_owned(
    state: &mut DeviceState,
    gva: u64,
    width: u32,
    height: u32,
    bgra: Vec<u8>,
    object_type: u8,
    backing: Option<GvaBacking>,
) {
    if gva == 0 || width == 0 || height == 0 || width > MAX_SCANOUT_DIM || height > MAX_SCANOUT_DIM
    {
        return;
    }
    let need = (height as usize)
        .saturating_mul(width as usize)
        .saturating_mul(RGBA8_BPP as usize);
    if bgra.len() < need {
        return;
    }
    let generation = state.next_sampled_content_generation();
    // A store re-populates the identity, so any miss the byte cap could have
    // been charged for it is now a different question. Retiring the witness key
    // here keeps `gva_cap_wanted` a count of lookups the cap actually cost,
    // rather than one that keeps accruing against content that came back.
    state.gva_eviction_witness.note_restored(gva, width, height);
    let touch = state.next_gva_touch();
    let entry = state.host_gva_surfaces.entry(gva).or_default();
    entry.last_touch = touch;
    entry.host_gen = generation;
    entry.width = width;
    entry.height = height;
    // One of the two sites that change this map's byte total; see
    // [`DeviceState::gva_cache_bytes`]. The replaced entry's bytes are
    // reclaimed before the new ones are charged, so a store at an existing key
    // nets to the difference instead of double-counting. Applied to the device
    // below, once this borrow of the entry has ended.
    let reclaimed = entry.bgra.len();
    entry.bgra = std::sync::Arc::new(bgra);
    let charged = entry.bgra.len();
    entry.producer_object_type = object_type;
    // These bytes came from *this* backing, so it replaces whatever the
    // previous store recorded — including a `None` that says the walk could
    // not name it. Carrying the old list forward would let a validated entry
    // vouch for pixels it did not produce.
    // The old token was taken for the old page list. If these bytes came from
    // exactly the same pages it still watches exactly the right memory, and it
    // has to be kept: the host holds a freshly tracked set at "generation
    // unreadable" for a two-harvest startup window, so a token retired and
    // re-taken on every store never survives long enough to become readable.
    // Retiring unconditionally is why `gvac_gw_clean` was 0 of 201 331 lookups
    // on a 300 s boot — not because the guest had rewritten every entry, which
    // is how that zero was read, but because no set this rail ever created
    // outlived its own arming window.
    entry.backing = backing;
    charge_gva_cache_bytes(state, reclaimed, charged);
    enforce_gva_cache_cap(state, gva);
}

/// Move [`DeviceState::gva_cache_bytes`] by one entry's replacement.
///
/// Reclaim before charge so the running total never transiently exceeds the
/// real one, and saturating so a bookkeeping slip can only under-report — an
/// over-report would make the cap evict content it never needed to.
fn charge_gva_cache_bytes(state: &mut DeviceState, reclaimed: usize, charged: usize) {
    state.gva_cache_bytes = state
        .gva_cache_bytes
        .saturating_sub(reclaimed)
        .saturating_add(charged);
}

/// Hold [`DeviceState::host_gva_surfaces`] at or under
/// [`GVA_ENCODE_CACHE_BYTE_CAP`], evicting the least-recently-**used** entries
/// first.
///
/// Runs from [`store_gva_owned`], which is the map's only insert path, so the
/// bound is enforced exactly where it can be crossed. Two things it deliberately
/// does not do:
///
/// - **It never bulk-clears.** Draining to a 7/8 low-water mark, the same shape
///   [`crate::model::LruBytesMemo`] uses, means a steady insert stream evicts in
///   occasional batches with headroom instead of one-for-one at the boundary,
///   and a cap crossing never dumps the hot set — the re-encode cliff that
///   pattern exists to avoid.
/// - **It never evicts a GVA that still owes a deferred writeback.** A window in
///   `gva_deferred_flush` names this address and its flush reads this entry;
///   dropping it would turn a memory bound into lost guest pixels, which is the
///   Goal 3 loss class. That is a correctness exclusion, not a heuristic — the
///   obligation is recorded, not guessed.
///
/// `protect` is the address the store that triggered this just wrote, and it is
/// never evicted. Without it a single entry bigger than the low-water mark is
/// dropped by its own store — the map holds one entry, that entry is over, and
/// it is the only eviction candidate — so the surface is never cached at all.
/// That is reachable rather than theoretical: `MAX_SCANOUT_DIM` is 8192, so an
/// entry may be up to 256 MiB against a 112 MiB low-water mark. An oversized
/// entry therefore rides alone and over the cap, matching the sibling memo,
/// because refusing to cache a surface for being big is how a 4K wallpaper
/// stops being cached at all.
fn enforce_gva_cache_cap(state: &mut DeviceState, protect: u64) {
    let cap = state.gva_cache_byte_cap;
    let low_water = cap - cap / 8;
    // The running total, not a fresh sum: this runs on the store path, which is
    // the draw path. See [`DeviceState::gva_cache_bytes`] — the census
    // recomputes the real figure once a second and reports any divergence, so
    // trusting it here is checked rather than assumed.
    if state.gva_cache_bytes <= low_water {
        return;
    }
    // Coldest first. This only runs at the cap boundary, never on the steady
    // store path, so one ordered pass over the keys is acceptable.
    let mut by_touch: Vec<(u64, u64)> = state
        .host_gva_surfaces
        .iter()
        .filter(|(gva, _)| **gva != protect && !state.gva_deferred_flush.contains_key(gva))
        .map(|(&gva, e)| (e.last_touch, gva))
        .collect();
    by_touch.sort_unstable();
    for (_, gva) in by_touch {
        // `evict_gva` maintains the running total, so this reads the live
        // figure each round rather than tracking a second copy of it.
        if state.gva_cache_bytes <= low_water {
            break;
        }
        let Some(e) = state.host_gva_surfaces.get(&gva) else {
            continue;
        };
        let (width, height) = (e.width, e.height);
        state.gva_eviction_witness.note_evicted(gva, width, height);
        evict_gva(state, gva);
    }
}

/// The one selection rule every GVA-cache read goes through: exact key, exact
/// geometry, enough bytes for it. Returns the entry and the byte length a
/// serve would hand out.
///
/// Pure — it does **not** charge the byte cap's harm witness. Probes that ask
/// "would this hit" ([`has_gva`], [`touch_gva`]) go through here directly, so
/// only a read that actually wanted the pixels is counted as harm; charging
/// here instead would count two or three times for one frame's single logical
/// lookup and make the figure uninterpretable.
fn lookup_gva(
    state: &DeviceState,
    gva: u64,
    width: u32,
    height: u32,
) -> Option<(&crate::model::HostSurface, usize)> {
    let need = (height as usize)
        .saturating_mul(width as usize)
        .saturating_mul(RGBA8_BPP as usize);
    let e = state.host_gva_surfaces.get(&gva)?;
    (e.width == width && e.height == height && !e.bgra.is_empty() && e.bgra.len() >= need)
        .then_some((e, need))
}

/// [`lookup_gva`] for the paths that want the bytes, charging a miss to the
/// byte cap when the cap is what removed this exact identity.
///
/// A key that was never cached, or whose geometry never matched, is an ordinary
/// miss and is not counted — see [`crate::model::GvaEvictionWitness`].
fn read_gva(
    state: &DeviceState,
    gva: u64,
    width: u32,
    height: u32,
) -> Option<(&crate::model::HostSurface, usize)> {
    let hit = lookup_gva(state, gva, width, height);
    if hit.is_none() {
        state.gva_eviction_witness.note_miss(gva, width, height);
    }
    hit
}

/// Mark a GVA entry most-recently-used, so the byte cap's eviction reaches only
/// entries nothing is reading.
///
/// Call on a **confirmed serve**, not on an attempted one: this is the half of
/// the recency signal that keeps a stored-once-sampled-forever entry (the
/// retained wallpaper class) alive, and charging recency for lookups that were
/// then refused would keep entries warm that nothing can actually use.
pub fn touch_gva(state: &mut DeviceState, gva: u64, width: u32, height: u32) {
    if lookup_gva(state, gva, width, height).is_none() {
        return;
    }
    let stamp = state.next_gva_touch();
    if let Some(e) = state.host_gva_surfaces.get_mut(&gva) {
        e.last_touch = stamp;
    }
}

pub fn get_gva(state: &DeviceState, gva: u64, width: u32, height: u32) -> Option<&[u8]> {
    get_gva_with_gen(state, gva, width, height).map(|(bgra, _)| bgra)
}

/// Whether a [`get_gva`] for this key would hit, without borrowing the bytes.
///
/// Lets a caller that needs `&mut DeviceState` (backing revalidation) find out
/// first whether there is anything to revalidate.
pub fn has_gva(state: &DeviceState, gva: u64, width: u32, height: u32) -> bool {
    lookup_gva(state, gva, width, height).is_some()
}

/// Borrow a GVA encode plus its producer generation.
///
/// This is diagnostic provenance for the linear-sample loss proxy; selection
/// semantics are identical to [`get_gva`].
fn get_gva_with_gen(
    state: &DeviceState,
    gva: u64,
    width: u32,
    height: u32,
) -> Option<(&[u8], u64)> {
    let (e, need) = read_gva(state, gva, width, height)?;
    Some((&e.bgra[..need], e.host_gen))
}

/// Explicit drop (tests / object delete). Unmap does **not** come through here;
/// see [`store_gva_owned`] for why the map is retained across it.
pub fn evict_gva(state: &mut DeviceState, gva: u64) {
    if let Some(entry) = state.host_gva_surfaces.remove(&gva) {
        // The other site that changes this map's byte total; see
        // [`DeviceState::gva_cache_bytes`].
        state.gva_cache_bytes = state.gva_cache_bytes.saturating_sub(entry.bgra.len());
    }
}

/// Entry count and resident bytes of one host-side pixel cache.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CacheLevel {
    pub entries: u64,
    pub bytes: u64,
    /// Bytes held by the single largest entry — the figure that separates "many
    /// small surfaces" from "a few 4K ones", which cost ~4x a 1080p entry each.
    pub largest: u64,
}

impl CacheLevel {
    fn of<'a, K: 'a, V: 'a>(
        map: &'a std::collections::BTreeMap<K, V>,
        len: impl Fn(&V) -> usize,
    ) -> Self {
        let mut level = Self {
            entries: map.len() as u64,
            ..Self::default()
        };
        for value in map.values() {
            let bytes = len(value) as u64;
            level.bytes += bytes;
            level.largest = level.largest.max(bytes);
        }
        level
    }
}

/// Resident size of every host-side pixel cache, right now.
///
/// # These are LEVELS, not per-interval counts
///
/// The opposite convention from `store_routes`, whose every field is a count for
/// one census interval and must be summed across lines. Summing these instead
/// would multiply a steady cache by the census cadence and report a leak that is
/// not there. **Take the last line for the current size, and `peak_bytes` for the
/// high-water mark**; the trend across lines is the thing to read.
///
/// `peak_bytes` is carried because a single last line cannot show a transient
/// spike, and a spike is what a resolution change produces: every geometry
/// change orphans the previous geometry's entries until something replaces or
/// evicts them.
///
/// # Why this exists
///
/// None of these maps has a size cap, and until now none had a counter either,
/// so "the host caches grow without bound" was neither refuted nor measurable —
/// `host_surfaces` alone is keyed by surface id with `remove()` on unmap/delete
/// and no bound on how many live ids there may be. This is the proxy for that
/// class, added before any attempt to cap it, because a cap chosen without a
/// measurement is a magic number.
///
/// Measure-only. Nothing may read this back to decide what to cache or evict:
/// that would make a resource gauge into a content heuristic.
///
/// `bytes` sums `Arc<Vec<u8>>` lengths, and a deferred render window can share
/// an entry's allocation rather than copying it — so a cache figure is the size
/// of the pixels reachable through the cache, not memory additional to the
/// windows.
///
/// # The surface tier reads zero by construction, and that zero is the finding
///
/// `surfaces`/`surface_bytes`/`surface_largest` are 0 on every census line of
/// every driven boot — 4 896 samples — while `gva` and `linear` beside them hold
/// 100 MB and 70 MB. That is not a dead tier and not a broken counter. It is
/// where the census samples:
///
/// - `note_cache_levels` runs in `lib.rs` at the tail of a drain tranche, after
///   `Device::drain` has returned.
/// - Inside that drain, `storage_flush::flush_all_windows_before_fence` lands
///   every armed render window before the guest is told the work is done.
/// - Every one of those landings takes the leased frame — `render_flush_copied`
///   has never fired, `render_flush_leased` fires on every census line — so each
///   one writes through `mapping_write::write_bgra8_uncached`, whose
///   `CacheOutcome::Invalidate` calls [`forget`].
///
/// So every entry an arm put in has been reclaimed before the reader looks, and
/// the tier is guaranteed empty at exactly the instant it is measured. A
/// **non-zero** reading is therefore the alarm: it means a cache entry outlived
/// the window that armed it, which is the leak this counter was added to catch
/// and which nothing else in the device would report.
///
/// Two consequences a reader has to carry: `total_bytes` and `peak_bytes`
/// exclude this tier by construction, so they understate the host-side pixel
/// footprint by whatever the surface cache holds mid-drain (an 8.29 MB composite
/// frame, at ~95 a second). And anyone wanting the tier's real occupancy needs a
/// high-water mark maintained at insert time — a level read here cannot answer
/// it, and reading zero here is not evidence that it is small.
fn cache_levels(state: &DeviceState) -> (CacheLevel, CacheLevel, CacheLevel) {
    (
        CacheLevel::of(&state.host_surfaces, |e| e.bgra.len()),
        CacheLevel::of(&state.host_gva_surfaces, |e| e.bgra.len()),
        CacheLevel::of(&state.host_linear_textures, |e| e.bytes.len()),
    )
}

/// GVA-keyed entries whose key no longer translates to the backing the pixels
/// were produced from. Returns `(moved, unmapped, checked)`.
///
/// This replaced a `gva_cache_staleness` probe that counted dead-task and
/// unbacked entries. Both of its fields read zero on every census line of
/// every boot — 0 of 331 recorded at `model/state.rs`, and 0 across all 151
/// lines of a later driven boot — so task death cannot be the eviction rule
/// and the question moved to the backing itself. Do not reintroduce it.
///
/// A `GVA` is only a name for whatever the owning task's page table points it at
/// now. `GvaBacking::gpas` records what it pointed at when the pixels were
/// stored, and `get_gva_with_gen` serves on `(gva, exact geometry)` — so an entry
/// whose key now walks somewhere else is one no correct lookup can use: the
/// name has been handed to a different allocation. That is the same "drop only
/// what could never be served" standard the dead-task rule was reaching for,
/// applied where the evidence says to look.
///
/// - **`moved`** — the key translates, to a different page than recorded. The
///   guest reused the address.
/// - **`unmapped`** — the key does not translate at all. Counted apart because
///   the two are different guest actions and, more importantly, because a
///   transient walk failure looks exactly like this: `d455c3e`'s whole finding
///   was that the device answers before the guest has finished mapping, so a
///   *failure to translate* must never on its own authorise dropping content.
///   Only `moved` carries positive evidence that the address belongs to someone
///   else now.
/// - **`checked`** — entries with a usable backing and a live task, i.e. the
///   denominator. Without it a reader cannot tell "nothing moved" from "nothing
///   was examined", which is the failure direction that reads as a clean result.
///
/// Cost is one page-table walk per entry per census interval — the **first**
/// recorded page only, not the whole list. A whole-list walk of a 4K entry is
/// ~2 025 walks and this runs on the drain thread; the first page is enough to
/// tell a reused address from a retained one, and this is a measurement rather
/// than the authorisation for a write.
///
/// Measure-only. Nothing may evict on this yet: it exists to size the rule
/// before the rule is written.
fn gva_backing_moved<H: HostMemory>(state: &DeviceState, host: &H) -> (u64, u64, u64) {
    let (mut moved, mut unmapped, mut checked) = (0, 0, 0);
    for &gva in state.host_gva_surfaces.keys() {
        match gva_backing_state(state, host, gva) {
            GvaBackingState::Unrecorded => {}
            GvaBackingState::Same => checked += 1,
            GvaBackingState::Unmapped => {
                checked += 1;
                unmapped += 1;
            }
            GvaBackingState::Moved => {
                checked += 1;
                moved += 1;
            }
        }
    }
    (moved, unmapped, checked)
}

/// Whether one GVA-keyed entry's key still translates to the pages its pixels
/// were produced from.
///
/// The single spelling of that question. [`gva_backing_moved`] sums it over the
/// whole map for the level census; the colour LOAD seed asks it about the one
/// entry it is about to serve. Two spellings of "did this address move" would be
/// two answers, and the serve-side reading is only worth having if it is the
/// same reading the census reports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GvaBackingState {
    /// No backing was recorded, or the task that recorded it is gone. The
    /// question cannot be asked, which is not the same as answering "fresh".
    Unrecorded,
    /// The key translates to the page it was stored over.
    Same,
    /// The key does not translate at all. Not evidence of reuse on its own —
    /// `d455c3e` found the device answers before the guest has finished
    /// mapping, and a transient walk failure looks exactly like this.
    Unmapped,
    /// The key translates, to a different page than recorded: the guest handed
    /// this address to another allocation. The only state that carries positive
    /// evidence these pixels belong to someone else.
    Moved,
}

/// [`GvaBackingState`] for one key. First recorded page only — enough to tell a
/// reused address from a retained one, and a whole-list walk of a 4K entry is
/// ~2 025 walks.
pub fn gva_backing_state<H: HostMemory>(
    state: &DeviceState,
    host: &H,
    gva: u64,
) -> GvaBackingState {
    let page_shift = state.page_shift;
    let Some(entry) = state.host_gva_surfaces.get(&gva) else {
        return GvaBackingState::Unrecorded;
    };
    let Some(backing) = entry.backing.as_ref() else {
        return GvaBackingState::Unrecorded;
    };
    let recorded = backing.first_gpa;
    // Same liveness test the walk itself applies: present in the table AND
    // flagged active. A dead task's page table cannot answer the question.
    let Some(task) = state
        .tasks
        .get(backing.task_id as usize)
        .filter(|t| t.active)
    else {
        return GvaBackingState::Unrecorded;
    };
    match crate::runtime::gva_mem::translate_task_gva(host, task, gva, page_shift) {
        None => GvaBackingState::Unmapped,
        Some(live) if (live & page_mask(page_shift)) != recorded => GvaBackingState::Moved,
        Some(_) => GvaBackingState::Same,
    }
}

/// Emit [`cache_levels`] at most once per census interval.
///
/// Shares the one-second cadence the drain census already runs on, so a boot's
/// cache trend lines up row-for-row with `store_routes` and `drain_duty`.
pub fn note_cache_levels<H: HostMemory>(state: &DeviceState, host: &H) {
    use std::sync::atomic::{AtomicU64, Ordering};
    static LAST_MS: AtomicU64 = AtomicU64::new(0);
    static PEAK_BYTES: AtomicU64 = AtomicU64::new(0);

    let (surfaces, gva, linear) = cache_levels(state);
    let total = surfaces.bytes + gva.bytes + linear.bytes;
    let peak = PEAK_BYTES.fetch_max(total, Ordering::Relaxed).max(total);

    let now = crate::observe::elapsed_ms() as u64;
    let last = LAST_MS.load(Ordering::Relaxed);
    if now.saturating_sub(last) < 1000 {
        return;
    }
    // Losing the race only costs a skipped interval, never a double line.
    if LAST_MS
        .compare_exchange(last, now, Ordering::Relaxed, Ordering::Relaxed)
        .is_err()
    {
        return;
    }
    let (moved, unmapped, checked) = gva_backing_moved(state, host);
    // `gva_cap_*` are the only running totals on this line — the eviction
    // witness accumulates for the life of the device, where every other field
    // here is a level. Take the last line for all of them either way; do not
    // sum any of it.
    let (cap_evicted, cap_wanted, cap_forgotten) = state.gva_eviction_witness.counts();
    // The running total the cap actually tests against, minus the real sum this
    // census just computed for `gva_bytes`. Always 0; anything else means a
    // mutation site changed `bgra` without telling `gva_cache_bytes`, and the
    // cap is bounding a number that has stopped describing the map.
    let cap_drift = state.gva_cache_bytes as i64 - gva.bytes as i64;
    crate::observe::off(format!(
        "host_cache_levels (levels, not per-interval) total_bytes={total} peak_bytes={peak} \
         surfaces={} surface_bytes={} surface_largest={} \
         gva={} gva_bytes={} gva_largest={} \
         gva_backing_moved={moved} gva_backing_unmapped={unmapped} \
         gva_backing_checked={checked} \
         gva_cap_bytes={} gva_cap_drift={cap_drift} \
         gva_cap_evicted={cap_evicted} gva_cap_wanted={cap_wanted} \
         gva_cap_forgotten={cap_forgotten} \
         linear={} linear_bytes={} linear_largest={}",
        surfaces.entries,
        surfaces.bytes,
        surfaces.largest,
        gva.entries,
        gva.bytes,
        gva.largest,
        state.gva_cache_byte_cap,
        linear.entries,
        linear.bytes,
        linear.largest,
    ));
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::model::{DeviceId, PAGE_SHIFT_ARM64E};
    use crate::runtime::host::FakeHost;

    /// The probe has to separate a reassigned address from one that merely
    /// failed to walk, because only the first is evidence the entry is dead.
    ///
    /// `d455c3e`'s finding was that this device routinely asks before the guest
    /// has finished mapping, so a failure to translate is a transient state and
    /// not a licence to drop content — collapsing `unmapped` into `moved` would
    /// build an eviction rule on exactly that mistake.
    #[test]
    fn the_backing_probe_separates_a_reassigned_address_from_an_unmapped_one() {
        use crate::model::GvaBacking;
        let mut host = FakeHost::new();
        let mut st = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let root_gpa = setup_depth1_task(&mut host, &mut st);

        // Three entries at GVAs 1, 2 and 3 pages in, each recording the page its
        // own PTE currently names (PT_BASE + i, i.e. 5, 6, 7).
        for i in 1..=3u32 {
            let gva = (i as u64) << PAGE_SHIFT_ARM64E;
            store_gva_owned(&mut st, gva, 2, 2, vec![0u8; 2 * 2 * 4], 0, None);
            st.host_gva_surfaces.get_mut(&gva).unwrap().backing = Some(GvaBacking {
                task_id: 1,
                first_gpa: ((4 + i) as u64) << PAGE_SHIFT_ARM64E,
            });
        }

        // Nothing touched yet: every backing still agrees with the page table.
        assert_eq!(gva_backing_moved(&st, &host), (0, 0, 3));

        // The guest hands GVA page 2 to a different allocation.
        repoint_pte(&mut host, root_gpa, 2, 12);
        assert_eq!(
            gva_backing_moved(&st, &host),
            (1, 0, 3),
            "a re-pointed PTE is a moved backing"
        );

        // And drops GVA page 3 entirely (PTE 0 = not present).
        repoint_pte(&mut host, root_gpa, 3, 0);
        let (moved, unmapped, checked) = gva_backing_moved(&st, &host);
        assert_eq!(
            (moved, unmapped),
            (1, 1),
            "unmapped must not be folded into moved"
        );
        // The denominator must stay honest, or "nothing moved" and "nothing was
        // examined" become the same reading.
        assert_eq!(checked, 3);
    }

    /// The gauge has to separate the two shapes a growing cache can take, or it
    /// cannot tell "many small surfaces" from "a few 4K ones" — which is the
    /// distinction the no-size-cap question turns on, since a 4K entry is ~4x a
    /// 1080p one.
    #[test]
    fn the_cache_gauge_reports_count_bytes_and_the_largest_entry() {
        let mut st = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        assert_eq!(cache_levels(&st).0, CacheLevel::default(), "empty is zero");

        // Two small entries and one large: 4x4 and 2x2 at RGBA8, then 8x8.
        store(&mut st, 1, 4, 4, vec![0u8; 4 * 4 * 4]);
        store(&mut st, 2, 2, 2, vec![0u8; 2 * 2 * 4]);
        store(&mut st, 3, 8, 8, vec![0u8; 8 * 8 * 4]);

        let (surfaces, _, _) = cache_levels(&st);
        assert_eq!(surfaces.entries, 3);
        assert_eq!(surfaces.bytes, (4 * 4 + 2 * 2 + 8 * 8) * 4);
        // Not the sum and not the newest — the largest single entry.
        assert_eq!(surfaces.largest, 8 * 8 * 4);

        // Eviction is visible, which is what makes an unbounded map detectable:
        // a gauge that only ever rose could not tell growth from churn.
        forget(&mut st, 3);
        let (after, _, _) = cache_levels(&st);
        assert_eq!(after.entries, 2);
        assert_eq!(after.bytes, (4 * 4 + 2 * 2) * 4);
        assert_eq!(after.largest, 4 * 4 * 4);
    }

    /// A generation must name one content for the life of the device, and the
    /// hard case is the one that shipped broken: the entry is *destroyed* in
    /// between.
    ///
    /// `evict_gva` runs on every deferred GVA render Store arm, so this
    /// sequence is the routine compositor path rather than a corner. With a
    /// per-entry counter both stores report generation 1, the engine's sampled
    /// cache matches `(gva, 1)` against the image it retained for the first
    /// one, and binds the previous content — measured live as
    /// `sampled_identity_stale identity_key=0xa4c000 generation=1` over two
    /// different 64x64 icons.
    ///
    /// Asserting the two generations differ is the whole property; asserting
    /// either value would pin the counter's history instead.
    #[test]
    fn a_gva_reused_after_eviction_never_repeats_a_generation() {
        let mut st = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let (gva, w, h) = (0xa4_c000u64, 2u32, 2u32);

        let px = vec![0x11; (w * h * 4) as usize];
        store_gva_owned(&mut st, gva, w, h, px, 0, None);
        let (_, first) = get_gva_with_gen(&st, gva, w, h).expect("first store");

        evict_gva(&mut st, gva);
        assert!(
            get_gva_with_gen(&st, gva, w, h).is_none(),
            "the arm removes the entry outright"
        );

        let px = vec![0x22; (w * h * 4) as usize];
        store_gva_owned(&mut st, gva, w, h, px, 0, None);
        let (bytes, second) = get_gva_with_gen(&st, gva, w, h).expect("second store");

        assert_eq!(bytes[0], 0x22, "the cache holds the new content");
        assert_ne!(
            first, second,
            "same gva, different bytes, same generation: the sampled cache \
             would bind the first store's image for the second store's pixels"
        );
    }

    /// The same rule across producers. The generations used to live in two
    /// namespaces split by a `1 << 32` constant precisely because the counters
    /// were independent; one counter removes the constant and the failure mode
    /// it was guarding against, so nothing may reintroduce a second source.
    #[test]
    fn every_host_cache_producer_draws_from_one_generation_source() {
        let mut st = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let px = vec![0u8; 4 * 4 * 4];
        let mut seen = std::collections::HashSet::new();

        store(&mut st, 7, 4, 4, px.clone());
        seen.insert(
            get_from_with_gen(&st.host_surfaces, 7, 4, 4)
                .expect("mid store")
                .1,
        );
        store_texture(&mut st, 9, 4, 4, px.clone(), 0);
        seen.insert(
            get_from_with_gen(&st.host_texture_surfaces, 9, 4, 4)
                .expect("ref store")
                .1,
        );
        store_gva_owned(&mut st, 0x5000, 4, 4, px, 0, None);
        seen.insert(get_gva_with_gen(&st, 0x5000, 4, 4).expect("gva store").1);
        assert!(
            cede_surface_to_resident(&mut st, 7, 4, 4),
            "cession is a state change and must take a generation too"
        );
        seen.insert(st.host_surfaces.get(&7).expect("ceded entry").host_gen);

        assert_eq!(seen.len(), 4, "four stores, four distinct generations");
        assert!(
            !seen.contains(&0),
            "0 is reserved for 'no host content yet'"
        );
    }

    /// A short readback leaves a live entry empty, and that must not read the
    /// same as a supersede.
    ///
    /// `flush_linear_one` may refuse the guest write on drift and call the
    /// refusal lossless because "the cache entry keeps the authoritative
    /// bytes". When this returned a bare `bool` the caller discarded it, so the
    /// two arms below were one value: the frame was gone from both the cache
    /// and the guest pages, and nothing said so. The entry staying
    /// resident-authoritative afterwards is the proof the bytes did not land.
    #[test]
    fn a_short_resident_readback_is_distinguishable_from_a_supersede() {
        use crate::contract::pixel_format::MTL_FORMAT_RGBA16_FLOAT;
        let mut st = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let (task, r, gva, w, h, stride) = (6u32, 21u32, 0x30_2000u64, 4u32, 2u32, 32u64);
        assert!(note_linear_texture_resident(
            &mut st,
            task,
            r,
            gva,
            MTL_FORMAT_RGBA16_FLOAT,
            w,
            h,
            stride,
            2,
        ));
        let need = (w * h * 8) as usize;
        let short = vec![0xabu8; need - 1];
        assert_eq!(
            materialize_linear_resident(&mut st, task, r, 2, &short),
            Err(LinearMaterializeDecline::ReadbackShort {
                got: need - 1,
                need,
            }),
            "a live entry that could not be filled is a loss, not a supersede"
        );
        assert_eq!(
            linear_texture_resident_gen(&st, task, r, gva, MTL_FORMAT_RGBA16_FLOAT, w, h, stride),
            Some(2),
            "the entry is still resident-authoritative, so the frame is in neither place"
        );
    }

    /// Deferred linear residency lifecycle: note marks the entry
    /// resident-authoritative with empty bytes, the resident getter validates
    /// the descriptor exactly, materialize lands bytes and clears the marker,
    /// and a plain bytes store also clears it.
    #[test]
    fn linear_resident_note_materialize_and_store_clear() {
        use crate::contract::pixel_format::{MTL_FORMAT_RGBA16_FLOAT, MTL_FORMAT_RGBA8_UNORM};
        let mut st = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let (task, r, gva, w, h, stride) = (6u32, 21u32, 0x30_2000u64, 4u32, 2u32, 32u64);
        assert!(note_linear_texture_resident(
            &mut st,
            task,
            r,
            gva,
            MTL_FORMAT_RGBA16_FLOAT,
            w,
            h,
            stride,
            2,
        ));
        assert_eq!(
            linear_texture_resident_gen(&st, task, r, gva, MTL_FORMAT_RGBA16_FLOAT, w, h, stride),
            Some(2)
        );
        // Bytes consumers see nothing while resident-authoritative.
        assert!(
            get_linear_texture(&st, task, r, gva, MTL_FORMAT_RGBA16_FLOAT, w, h, stride).is_none()
        );
        // Any descriptor drift invalidates the resident claim.
        assert_eq!(
            linear_texture_resident_gen(&st, task, r, gva, MTL_FORMAT_RGBA8_UNORM, w, h, stride),
            None
        );
        assert_eq!(
            linear_texture_resident_gen(
                &st,
                task,
                r,
                gva + 0x1000,
                MTL_FORMAT_RGBA16_FLOAT,
                w,
                h,
                stride
            ),
            None
        );
        // Materialize with the wrong generation is refused; the right one
        // lands bytes and clears the marker.
        let flushed = vec![0xabu8; (w * h * 8) as usize];
        assert_eq!(
            materialize_linear_resident(&mut st, task, r, 9, &flushed),
            Err(LinearMaterializeDecline::Superseded { resident_gen: 2 }),
            "the wrong generation is a supersede, not a lost frame"
        );
        assert_eq!(
            materialize_linear_resident(&mut st, task, r, 2, &flushed),
            Ok(())
        );
        assert_eq!(
            linear_texture_resident_gen(&st, task, r, gva, MTL_FORMAT_RGBA16_FLOAT, w, h, stride),
            None
        );
        let got = get_linear_texture(&st, task, r, gva, MTL_FORMAT_RGBA16_FLOAT, w, h, stride)
            .expect("materialized bytes");
        assert!(got.iter().all(|&b| b == 0xab));
        // A later resident note supersedes; a plain store clears again.
        assert!(note_linear_texture_resident(
            &mut st,
            task,
            r,
            gva,
            MTL_FORMAT_RGBA16_FLOAT,
            w,
            h,
            stride,
            3,
        ));
        let px = vec![0x5au8; (w * h * 8) as usize];
        assert!(store_linear_texture(
            &mut st,
            task,
            r,
            gva,
            MTL_FORMAT_RGBA16_FLOAT,
            w,
            h,
            stride,
            &px,
        ));
        assert_eq!(
            linear_texture_resident_gen(&st, task, r, gva, MTL_FORMAT_RGBA16_FLOAT, w, h, stride),
            None
        );
    }

    /// Task/object deletion of a resident-authoritative entry queues the
    /// engine unpin key (the runtime drains `retired_linear_residents`).
    #[test]
    fn linear_resident_retires_on_task_and_object_delete() {
        use crate::contract::pixel_format::MTL_FORMAT_RGBA16_FLOAT;
        let mut st = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        st.define_task(6, 0x1000, 1);
        assert!(note_linear_texture_resident(
            &mut st,
            6,
            21,
            0x30_2000,
            MTL_FORMAT_RGBA16_FLOAT,
            4,
            2,
            32,
            2,
        ));
        // A pending guest-flush obligation dies with the entry (boot-16 rule:
        // never write guest pages at a lifetime boundary).
        let obligation_key = crate::model::ComputeStorageResidencyKey::linear(
            6,
            21,
            0x30_2000,
            32,
            64,
            4,
            2,
            MTL_FORMAT_RGBA16_FLOAT,
        );
        st.arm_linear_deferred_window(obligation_key, 2, std::collections::HashSet::new());
        assert!(st.delete_task(6));
        assert_eq!(st.retired_linear_residents.len(), 1);
        let key = st.retired_linear_residents[0];
        assert!(key.is_linear());
        assert_eq!(key.map_generation, 6);
        assert_eq!(key.texture_ref, 21);
        assert_eq!(key.surface_offset, 0x30_2000);
        crate::runtime::storage_flush::retire_linear_residents(&mut st);
        assert!(st.retired_linear_residents.is_empty());
        assert!(
            st.linear_deferred_flush.is_empty(),
            "retire must drop the guest-flush obligation"
        );

        st.define_task(6, 0x1000, 1);
        st.insert_object(6, 21);
        assert!(note_linear_texture_resident(
            &mut st,
            6,
            21,
            0x30_2000,
            MTL_FORMAT_RGBA16_FLOAT,
            4,
            2,
            32,
            5,
        ));
        assert!(st.delete_object(6, 21));
        assert_eq!(st.retired_linear_residents.len(), 1);
        assert_eq!(st.retired_linear_residents[0].texture_ref, 21);
        // Non-resident entries retire nothing.
        st.retired_linear_residents.clear();
        let px = vec![0u8; 4 * 2 * 8];
        st.insert_object(6, 22);
        assert!(store_linear_texture(
            &mut st,
            6,
            22,
            0x40_0000,
            MTL_FORMAT_RGBA16_FLOAT,
            4,
            2,
            32,
            &px,
        ));
        assert!(st.delete_object(6, 22));
        assert!(st.retired_linear_residents.is_empty());
    }

    #[test]
    fn store_and_get_roundtrip() {
        let mut st = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let w = 4u32;
        let h = 2u32;
        let mut px = vec![0u8; (w * h * 4) as usize];
        px[0] = 0x11;
        px[1] = 0x22;
        px[2] = 0x33;
        px[3] = 0xff;
        store(&mut st, 7, w, h, px);
        let got = get(&st, 7, w, h).expect("cached");
        assert_eq!(got[0], 0x11);
        assert_eq!(got[3], 0xff);
        assert!(get(&st, 7, 8, 8).is_none());
        forget(&mut st, 7);
        assert!(get(&st, 7, w, h).is_none());
        let _ = HostSurface::default();
    }

    #[test]
    fn texture_and_surface_namespaces_are_separate() {
        let mut st = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let w = 2u32;
        let h = 2u32;
        let mut surface = vec![0u8; 16];
        surface[0] = 1;
        let mut tex = vec![0u8; 16];
        tex[0] = 2;
        store(&mut st, 5, w, h, surface);
        store_texture(&mut st, 5, w, h, tex, 0);
        assert_eq!(get(&st, 5, w, h).unwrap()[0], 1);
        assert_eq!(get_texture(&st, 5, w, h).unwrap()[0], 2);
    }

    /// The ref cache remembers which address produced its pixels, and a store at
    /// a new address replaces that answer rather than leaving the old one.
    ///
    /// This is the only thing that separates "the GVA entry aged out of its byte
    /// cap and the ref door is serving the same allocation" from "the guest
    /// re-pointed this texture and the ref door is serving the previous
    /// allocation's picture". Both look identical at the serve site without it,
    /// and only the second is a wrong LOAD seed.
    #[test]
    fn the_ref_cache_remembers_which_address_produced_its_pixels() {
        let mut st = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let (w, h) = (2u32, 2u32);
        assert_eq!(
            texture_source_gva(&st, 5, w, h),
            None,
            "no entry, no answer — an absent entry must not read as address 0"
        );
        store_texture(&mut st, 5, w, h, vec![1u8; 16], 0x4000);
        assert_eq!(texture_source_gva(&st, 5, w, h), Some(0x4000));
        // The guest re-points the texture and stores again: the recorded address
        // must move with the pixels, or a later serve compares against a stale one.
        store_texture(&mut st, 5, w, h, vec![2u8; 16], 0x9000);
        assert_eq!(texture_source_gva(&st, 5, w, h), Some(0x9000));
        // A producer with no address records none, which is its own case at the
        // serve site: unknowable, neither "same" nor "different".
        store_texture(&mut st, 5, w, h, vec![3u8; 16], 0);
        assert_eq!(texture_source_gva(&st, 5, w, h), Some(0));
        assert_eq!(
            texture_source_gva(&st, 5, w + 1, h),
            None,
            "a geometry the entry does not answer at answers nothing here either"
        );
    }

    #[test]
    fn gva_cache_roundtrip() {
        let mut st = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let gva = 0x2c48000u64;
        let mut px = vec![0u8; 16];
        px[0] = 0xaa;
        store_gva_owned(&mut st, gva, 2, 2, px, 0, None);
        assert_eq!(get_gva(&st, gva, 2, 2).unwrap()[0], 0xaa);
        let (got, generation) = get_gva_with_gen(&st, gva, 2, 2).unwrap();
        assert_eq!(got[0], 0xaa);
        assert_eq!(generation, 1);
        assert!(get_gva_with_gen(&st, gva, 4, 1).is_none());

        let mut replacement = vec![0u8; 16];
        replacement[0] = 0xbb;
        store_gva_owned(&mut st, gva, 2, 2, replacement, 0, None);
        let (got, generation) = get_gva_with_gen(&st, gva, 2, 2).unwrap();
        assert_eq!(got[0], 0xbb);
        assert_eq!(generation, 2);

        let mut owned = vec![0u8; 16];
        owned[0] = 0xcc;
        store_gva_owned(&mut st, gva, 2, 2, owned, 2, None);
        let (got, generation) = get_gva_with_gen(&st, gva, 2, 2).unwrap();
        assert_eq!(got[0], 0xcc);
        assert_eq!(generation, 3);
        evict_gva(&mut st, gva);
        assert!(get_gva(&st, gva, 2, 2).is_none());
    }

    /// The GVA encode cache is keyed by virtual address alone, at any geometry.
    ///
    /// This is what makes "UnmapMemory retains the encode" work: the retain is
    /// the *absence* of an evict on the unmap path, so the cache has to stay
    /// readable through page-table churn without anyone re-registering it. The
    /// live x86 wallpaper class was a full sky store to `gva=0x2c22000` followed
    /// by UnmapMemory + MapMemory2 of the same VA with new PFNs; if the entry
    /// did not survive on the VA key alone, the next sample found zero guest
    /// pages and an empty cache and the pipe stored a black wipe. No size gate —
    /// a 64x48 layer takes the same path as a full-screen one.
    #[test]
    fn gva_encode_is_keyed_by_address_at_any_size() {
        let mut st = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let gva = 0x2c22000u64;
        // Small layer — same retain path as wallpaper (no W×H gate).
        let w = 64u32;
        let h = 48u32;
        let mut px = vec![0u8; (w * h * 4) as usize];
        for chunk in px.chunks_exact_mut(4) {
            chunk[0] = 185;
            chunk[1] = 126;
            chunk[2] = 81;
            chunk[3] = 255;
        }
        store_gva_owned(&mut st, gva, w, h, px, 0, None);
        let got = get_gva(&st, gva, w, h).expect("retained on the VA key");
        assert_eq!(got[0], 185);
        assert_eq!(got[2], 81);
    }

    /// Regression guard for the host-surface cache-hit validator
    /// (`get_from_with_gen` / `get_from`). This decides whether cached pixels
    /// are served straight to scanout/present, so every guard clause is
    /// load-bearing: serving a differently-sized or truncated entry paints the
    /// wrong or a torn surface (residue / framebuffer corruption). Lock:
    ///  - absent id -> None;
    ///  - width OR height mismatch -> None (never resize-serve a stale frame);
    ///  - empty or short-of-`need` bytes -> None (no partial garbage);
    ///  - exact geom + sufficient bytes -> exactly `need` bytes (over-allocated
    ///    entries are truncated to the requested extent) plus the entry host_gen;
    ///  - `get_from` returns the same bytes, dropping only the generation.
    #[test]
    fn host_surface_cache_hit_validates_geom_and_truncates_to_need() {
        use crate::contract::pixel_format::RGBA8_BPP;
        let (id, w, h) = (7u32, 4u32, 2u32);
        let need = (w * h * RGBA8_BPP) as usize; // 32
        let mut map: std::collections::BTreeMap<u32, HostSurface> = Default::default();

        // Absent id.
        assert_eq!(get_from_with_gen(&map, id, w, h), None);

        // Store an over-allocated (need + slop) buffer with a distinct host_gen.
        let mut bgra = vec![0xABu8; need + 16];
        bgra[need] = 0xCD; // a byte past `need` must never be returned
        map.insert(
            id,
            HostSurface {
                width: w,
                height: h,
                bgra: std::sync::Arc::new(bgra),
                host_gen: 9,
                ..Default::default()
            },
        );

        // Geometry mismatch on either axis must miss (no resize-serve).
        assert_eq!(get_from_with_gen(&map, id, w + 1, h), None);
        assert_eq!(get_from_with_gen(&map, id, w, h + 1), None);

        // Exact hit: exactly `need` bytes (slop truncated) + the entry host_gen.
        let (bytes, gen) = get_from_with_gen(&map, id, w, h).expect("exact geom must hit");
        assert_eq!(bytes.len(), need, "must truncate to width*height*BPP");
        assert_eq!(gen, 9, "must report the entry host_gen");
        assert!(bytes.iter().all(|&b| b == 0xAB), "no slop byte leaks in");

        // get_from is the same content, generation dropped.
        assert_eq!(get_from(&map, id, w, h), Some(bytes));

        // Empty bytes -> None even with matching geometry.
        map.get_mut(&id).unwrap().bgra = std::sync::Arc::new(Vec::new());
        assert_eq!(
            get_from_with_gen(&map, id, w, h),
            None,
            "empty entry misses"
        );

        // Non-empty but short of `need` -> None (truncated store, no partial serve).
        map.get_mut(&id).unwrap().bgra = std::sync::Arc::new(vec![0xABu8; need - 1]);
        assert_eq!(
            get_from_with_gen(&map, id, w, h),
            None,
            "under-`need` bytes must not be served",
        );
    }

    /// Ceding a mapping to its resident must stop the cache answering for it —
    /// with a miss, never with the frame the Store superseded.
    ///
    /// The whole point of the `skip_readback` rail is that no CPU copy of the new
    /// frame exists, so anything still serving the *old* one is serving a frame
    /// that is now a layer behind. `capture_present_frame` reads
    /// `surface_cache::get` **before** it tries the resident, so a cession that
    /// left the bytes in place would pin the display to the pre-Store frame for as
    /// long as the rail stayed engaged, with nothing to report it.
    ///
    /// The restore direction is asserted too: the flush writes through
    /// `mapping_write::write_bgra8`, whose tail republishes this entry, and that
    /// is what ends the cession. A cession that could not be ended would leave the
    /// mapping permanently dependent on a resident that only a pin protects.
    #[test]
    fn a_ceded_surface_serves_a_miss_and_says_it_was_ceded() {
        use crate::model::{DeviceId, PAGE_SHIFT_X86};
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let (w, h) = (4u32, 4u32);
        let need = (w * h * 4) as usize;
        store(&mut state, 7, w, h, vec![0xA1u8; need]);
        let before = state.host_surfaces.get(&7).map(|e| e.host_gen).unwrap();

        assert!(cede_surface_to_resident(&mut state, 7, w, h));
        assert_eq!(
            get(&state, 7, w, h),
            None,
            "a ceded entry must not serve the frame the Store superseded"
        );
        assert!(
            get_shared(&state, 7, w, h).is_none(),
            "the shared handle must miss wherever the slice does"
        );
        assert!(surface_ceded_to_resident(&state, 7, w, h));
        assert!(
            state.host_surfaces.get(&7).unwrap().host_gen != before,
            "the cession is a state change and must advance host_gen"
        );

        // A geometry this cache would not have stored anyway is refused, so the
        // arm can fail closed rather than leave a live entry contradicting a
        // resident-authoritative window.
        assert!(!cede_surface_to_resident(&mut state, 0, w, h));
        assert!(!cede_surface_to_resident(&mut state, 7, 0, h));
        assert!(!cede_surface_to_resident(
            &mut state,
            7,
            w,
            MAX_SCANOUT_DIM + 1
        ));

        // The flush's republish ends it.
        store(&mut state, 7, w, h, vec![0xB2u8; need]);
        assert!(!surface_ceded_to_resident(&state, 7, w, h));
        assert_eq!(get(&state, 7, w, h).map(|b| b[0]), Some(0xB2));
    }

    /// A ceded entry is not the same thing as a stale-geometry one, and the
    /// classifier must not confuse them.
    ///
    /// Both make `get` miss, and folding them together would print
    /// `have=4x4` against `want=4x4` on the LOAD-seed decline — a line that reads
    /// as a contradiction rather than as the expected cost of the rail.
    #[test]
    fn cession_is_distinguishable_from_a_stale_geometry_entry() {
        use crate::model::{DeviceId, PAGE_SHIFT_X86};
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        store(&mut state, 7, 8, 8, vec![0xA1u8; 8 * 8 * 4]);
        assert!(
            !surface_ceded_to_resident(&state, 7, 4, 4),
            "an entry at another geometry is stale, not ceded"
        );
        assert!(
            !surface_ceded_to_resident(&state, 9, 4, 4),
            "an absent entry is absent, not ceded"
        );
        assert!(cede_surface_to_resident(&mut state, 7, 4, 4));
        assert!(surface_ceded_to_resident(&state, 7, 4, 4));
        assert!(
            !surface_ceded_to_resident(&state, 7, 8, 8),
            "the cession is scoped to the geometry it was taken at"
        );
    }

    /// `get_shared` must hit exactly when `get` hits, including for a stored
    /// buffer carrying slop past `width * height * 4`.
    ///
    /// Returning `None` there would be a silent seed loss, and a missing Load
    /// seed renders the pass onto a cleared target — a compositing layer going
    /// solid black. The shared handle cannot be truncated the way `get`'s slice
    /// is, so the slop case pays a copy instead of missing.
    #[test]
    fn get_shared_hits_wherever_get_hits_and_never_serves_slop() {
        use crate::model::{DeviceId, PAGE_SHIFT_X86};
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let (w, h) = (4u32, 4u32);
        let need = (w * h * 4) as usize;

        // Exact: shared, and the same bytes `get` serves.
        store(&mut state, 7, w, h, vec![0xA1u8; need]);
        let exact = get_shared(&state, 7, w, h).expect("exact store must hit");
        assert_eq!(exact.len(), need);
        assert_eq!(&exact[..], get(&state, 7, w, h).unwrap());

        // Slop: still hits, truncated to the geometry the caller matched on —
        // the engine rejects a seed whose length is not exactly that.
        let mut slop = vec![0xB2u8; need + 16];
        slop[need] = 0xCD;
        state.host_surfaces.get_mut(&7).unwrap().bgra = std::sync::Arc::new(slop);
        let got = get_shared(&state, 7, w, h).expect("a store with slop must still hit");
        assert_eq!(got.len(), need, "must truncate to width*height*BPP");
        assert!(got.iter().all(|&b| b == 0xB2), "no slop byte leaks in");
        assert_eq!(&got[..], get(&state, 7, w, h).unwrap());

        // Geometry mismatch misses in both, identically.
        assert!(get_shared(&state, 7, w + 1, h).is_none());
        assert!(get(&state, 7, w + 1, h).is_none());
    }

    /// A depth-1 task page table where root PTE `i` points at PFN `PT_BASE + i`,
    /// so a GVA of `i << PAGE_SHIFT_ARM64E` resolves to a page this test can
    /// re-point by rewriting one PTE — which is exactly what the guest does when
    /// it hands a virtual address to a different allocation.
    fn setup_depth1_task(host: &mut FakeHost, state: &mut DeviceState) -> u64 {
        use crate::contract::endian::st32;
        use crate::contract::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
        const DIR_PFN: u32 = 2;
        const ROOT_PFN: u32 = 3;
        const PT_BASE: u32 = 4;
        let dir_gpa = (DIR_PFN as u64) << PAGE_SHIFT_ARM64E;
        let root_gpa = (ROOT_PFN as u64) << PAGE_SHIFT_ARM64E;
        host.map_range(dir_gpa, 0x20, 0);
        host.map_range(root_gpa, 0x4000, 0);
        let mut d = [0u8; 8];
        st32(&mut d[DIRECTORY_ROOT_PFN as usize..], ROOT_PFN);
        st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
        let _ = host.write_gpa(dir_gpa, &d);
        for i in 0..16u32 {
            let pfn = PT_BASE + i;
            host.map_range((pfn as u64) << PAGE_SHIFT_ARM64E, 0x4000, 0);
            let mut pte = [0u8; 4];
            st32(&mut pte, pfn);
            let _ = host.write_gpa(root_gpa + (i as u64) * 4, &pte);
        }
        assert!(state.define_task(1, 0x1000, DIR_PFN));
        root_gpa
    }

    fn repoint_pte(host: &mut FakeHost, root_gpa: u64, index: u64, pfn: u32) {
        use crate::contract::endian::st32;
        let mut pte = [0u8; 4];
        st32(&mut pte, pfn);
        let _ = host.write_gpa(root_gpa + index * 4, &pte);
    }

    /// An address that does not resolve records no backing, rather than a
    /// backing of zero.
    ///
    /// This is the shape the dense page list used to guard, carried over to the
    /// first-page identity that replaced it. The list kept a `0` slot where a
    /// page did not resolve so two mappings with holes in different places
    /// could not read as the same one. A single recorded GPA has the sharper
    /// version of that hazard: store `0` for an unresolvable page and every
    /// unresolvable page compares equal to every other, so a later `Moved`
    /// check answers `Same` for two unrelated allocations. `None` is the only
    /// honest answer, and `gva_backing_state` reads it as `Unrecorded`.
    #[test]
    fn an_address_that_does_not_resolve_records_no_backing() {
        let mut host = FakeHost::new();
        let mut st = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let root_gpa = setup_depth1_task(&mut host, &mut st);
        let page = 1u64 << PAGE_SHIFT_ARM64E;
        let (w, h) = (64u32, 64u32);
        let gva = page;
        let pte_index = gva >> PAGE_SHIFT_ARM64E;

        let full = gva_backing(&st, &host, 1, gva, w, h).expect("walk resolves");
        assert_ne!(full.first_gpa, 0, "a resolved page is not the hole value");

        // Punt this page's PTE to an invalid PFN.
        repoint_pte(&mut host, root_gpa, pte_index, 0);
        assert!(
            gva_backing(&st, &host, 1, gva, w, h).is_none(),
            "an unresolvable address must yield no backing, never first_gpa=0"
        );

        // A zero geometry cannot name a span either.
        assert!(gva_backing(&st, &host, 1, gva, 0, h).is_none());
        assert!(gva_backing(&st, &host, 1, 0, w, h).is_none());
    }

    /// "Cannot tell" is its own answer, and the serve site is where conflating
    /// it with "fresh" would cost a frame.
    ///
    /// `the_backing_probe_separates_a_reassigned_address_from_an_unmapped_one`
    /// pins Same/Moved/Unmapped through the map-wide sum, which now delegates
    /// here, so those need no second statement. What only the per-entry answer
    /// can be asked is the case the sum *skips*: an entry whose walk never
    /// resolved, and a key that was never stored. The colour LOAD seed asks
    /// about one address at a time, so it meets both, and a probe that answered
    /// `Same` for either would report a clean result for a question it never
    /// asked.
    #[test]
    fn a_backing_the_probe_cannot_read_is_not_a_fresh_one() {
        let mut host = FakeHost::new();
        let mut st = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        setup_depth1_task(&mut host, &mut st);
        let page = 1u64 << PAGE_SHIFT_ARM64E;
        // 64x64 BGRA8 is exactly one 16 KiB page.
        let (w, h) = (64u32, 64u32);
        let gva = page;

        let backing = gva_backing(&st, &host, 1, gva, w, h).expect("walk resolves");
        store_gva_owned(
            &mut st,
            gva,
            w,
            h,
            vec![0xAB; (w * h * 4) as usize],
            0,
            Some(backing),
        );
        assert_eq!(
            gva_backing_state(&st, &host, gva),
            GvaBackingState::Same,
            "control: a store whose walk resolved reads its own pages back"
        );

        // Re-store the same key with no backing: the walk did not resolve, so
        // there is nothing to compare and the entry drops out of the census
        // denominator rather than counting as fresh.
        let px = vec![0xCD; (w * h * 4) as usize];
        store_gva_owned(&mut st, gva, w, h, px, 0, None);
        assert_eq!(
            gva_backing_state(&st, &host, gva),
            GvaBackingState::Unrecorded
        );
        assert_eq!(
            gva_backing_moved(&st, &host),
            (0, 0, 0),
            "and is not counted"
        );

        assert_eq!(
            gva_backing_state(&st, &host, gva + page),
            GvaBackingState::Unrecorded,
            "a key that was never stored is not an answer about backing"
        );
    }
}

/// The GVA encode cache's byte cap: what it bounds, what it refuses to touch,
/// and what it costs.
///
/// Every test here sets [`DeviceState::gva_cache_byte_cap`] to a size a test can
/// allocate. The policy under test is identical at 128 MiB; only the arithmetic
/// scales.
#[cfg(test)]
mod cap_tests {
    use super::*;
    use crate::model::{DeviceId, GvaDeferredEntry, PAGE_SHIFT_X86};

    /// One 16x16 BGRA frame — 1 024 bytes, so a cap in the tens of KiB holds a
    /// countable number of them.
    const W: u32 = 16;
    const H: u32 = 16;
    const FRAME_BYTES: usize = (W * H * 4) as usize;

    fn state_capped(cap: usize) -> DeviceState {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        state.gva_cache_byte_cap = cap;
        state
    }

    fn store_frame(state: &mut DeviceState, gva: u64, fill: u8) {
        store_gva_owned(state, gva, W, H, vec![fill; FRAME_BYTES], 0, None);
    }

    fn deferred_entry() -> GvaDeferredEntry {
        GvaDeferredEntry {
            task_id: 0,
            texture_ref: 0,
            producer_object_type: 0,
            width: W,
            height: H,
            row_stride: W * 4,
            format: 0,
            armed_seq: 1,
            armed_stamp_seq: 0,
            pages: std::collections::HashSet::new(),
            alloc_gen: 1,
        }
    }

    /// The leak, and the bound.
    ///
    /// This map is keyed by guest *virtual* address and a store at an existing
    /// key replaces in place, so growth comes entirely from new GVAs — which is
    /// exactly what a resolution change produces. Measured on the rig, 60
    /// guest-driven mode changes took it from 26 entries to 354 without it ever
    /// decreasing once. Here that shape is reproduced with 400 distinct
    /// addresses: uncapped it holds all 400, capped it holds what the cap
    /// allows and no more.
    #[test]
    fn the_byte_cap_bounds_a_map_whose_keys_never_repeat() {
        let uncapped = {
            let mut state = state_capped(usize::MAX);
            for i in 0..400u64 {
                store_frame(&mut state, 0x1000 + i * 0x1000, i as u8);
            }
            state.host_gva_surfaces.len()
        };
        assert_eq!(
            uncapped, 400,
            "control: without a cap every abandoned address is kept forever"
        );

        let cap = 64 * FRAME_BYTES;
        let mut state = state_capped(cap);
        for i in 0..400u64 {
            store_frame(&mut state, 0x1000 + i * 0x1000, i as u8);
        }
        let bytes: usize = state.host_gva_surfaces.values().map(|e| e.bgra.len()).sum();
        assert!(
            bytes <= cap,
            "capped map holds {bytes} bytes against a {cap}-byte cap"
        );
        assert!(
            state.host_gva_surfaces.len() < 400,
            "the cap must actually have evicted something"
        );
        assert!(
            state.gva_eviction_witness.evicted > 0,
            "and it must say so: an eviction count of zero is a cap that never engaged"
        );
    }

    /// The wallpaper property, and the whole reason this is recency and not
    /// staleness.
    ///
    /// A wallpaper plane is stored **once** and sampled every frame thereafter.
    /// A rule keyed on stores would see the most-wanted entry in the map as its
    /// coldest; a rule keyed on translation would evict it too, because this
    /// cache is deliberately retained across Unmap, so "does not translate" is
    /// that entry's normal state. Touch-on-read is what makes it the hottest
    /// thing here instead.
    #[test]
    fn an_entry_read_every_frame_but_never_rewritten_survives_the_cap() {
        let cap = 16 * FRAME_BYTES;
        let mut state = state_capped(cap);
        let wallpaper = 0x9_0000u64;
        store_frame(&mut state, wallpaper, 0xAB);

        for i in 0..400u64 {
            // Sampled every frame, never rewritten — the read is the only thing
            // keeping it alive.
            assert!(
                get_gva(&state, wallpaper, W, H).is_some(),
                "wallpaper evicted at round {i}"
            );
            touch_gva(&mut state, wallpaper, W, H);
            store_frame(&mut state, 0x100_0000 + i * 0x1000, i as u8);
        }

        let served = get_gva(&state, wallpaper, W, H).expect("wallpaper survives the whole stream");
        assert!(
            served.iter().all(|&b| b == 0xAB),
            "and it is still its own pixels, not a neighbour's"
        );
        assert_eq!(
            state.gva_eviction_witness.wanted.load(Relaxed),
            0,
            "no lookup was ever charged to the cap"
        );
    }

    /// Without the touch, the same entry is evicted — so the assertion above is
    /// testing the touch and not merely a map that happens to be small.
    ///
    /// The two tests are a matched pair on one binary: identical cap, identical
    /// insert stream, and the only difference is whether the read path reports
    /// the use.
    #[test]
    fn the_same_entry_is_evicted_when_nothing_reports_reading_it() {
        let cap = 16 * FRAME_BYTES;
        let mut state = state_capped(cap);
        let wallpaper = 0x9_0000u64;
        store_frame(&mut state, wallpaper, 0xAB);
        for i in 0..400u64 {
            store_frame(&mut state, 0x100_0000 + i * 0x1000, i as u8);
        }
        assert!(
            get_gva(&state, wallpaper, W, H).is_none(),
            "an entry nothing reports using is exactly what the cap is for"
        );
        assert_eq!(
            state.gva_eviction_witness.wanted.load(Relaxed),
            1,
            "and the lookup that then wanted it is charged to the cap, not written off"
        );
    }

    /// A memory bound must never become a pixel loss.
    ///
    /// A window in `gva_deferred_flush` names this address and its flush reads
    /// this entry, so evicting it would drop guest pixels that were promised —
    /// the Goal 3 loss class. The exclusion is on a recorded obligation, not on
    /// a guess about what the guest still wants.
    #[test]
    fn an_address_that_still_owes_a_deferred_writeback_is_never_evicted() {
        let cap = 8 * FRAME_BYTES;
        let mut state = state_capped(cap);
        let owed = 0x7_0000u64;
        store_frame(&mut state, owed, 0xCD);
        state.arm_gva_deferred_window(owed, deferred_entry());

        // Never touched again, so recency alone would have evicted it long ago.
        for i in 0..400u64 {
            store_frame(&mut state, 0x200_0000 + i * 0x1000, i as u8);
        }
        assert!(
            state.host_gva_surfaces.contains_key(&owed),
            "the obligation outranks the bound"
        );
        let served = get_gva(&state, owed, W, H).expect("still servable");
        assert!(served.iter().all(|&b| b == 0xCD));
    }

    /// The harm witness must charge the cap for its own misses and nothing
    /// else, or the number cannot be read.
    ///
    /// An address that was never cached misses for the ordinary reason, and a
    /// cached address asked for at a geometry it never held is not a cap
    /// casualty either. Only a lookup that would have hit, for an identity the
    /// cap removed, is the cost of capping.
    #[test]
    fn the_witness_charges_the_cap_only_for_misses_the_cap_caused() {
        let mut state = state_capped(2 * FRAME_BYTES);
        let victim = 0x5_0000u64;
        store_frame(&mut state, victim, 0x11);
        for i in 0..64u64 {
            store_frame(&mut state, 0x300_0000 + i * 0x1000, i as u8);
        }
        assert!(state.gva_eviction_witness.evicted > 0);
        assert!(!state.host_gva_surfaces.contains_key(&victim));

        // Never cached at all.
        assert!(get_gva(&state, 0xdead_0000, W, H).is_none());
        assert_eq!(
            state.gva_eviction_witness.wanted.load(Relaxed),
            0,
            "an address this cache never held is an ordinary miss"
        );

        // Evicted, but asked for at a geometry it never had.
        assert!(get_gva(&state, victim, W * 2, H).is_none());
        assert_eq!(
            state.gva_eviction_witness.wanted.load(Relaxed),
            0,
            "the cap did not remove *that* identity"
        );

        // The real thing.
        assert!(get_gva(&state, victim, W, H).is_none());
        assert_eq!(
            state.gva_eviction_witness.wanted.load(Relaxed),
            1,
            "a lookup that would have hit but for the cap is the cost of capping"
        );
    }

    /// A probe is not a read, or one frame's single logical lookup would be
    /// counted two or three times.
    ///
    /// The sampled path asks [`has_gva`] first (so it can decide whether there
    /// is anything to revalidate) and only then reads. Charging the witness in
    /// the shared selection rule would score that frame twice and make the
    /// figure uninterpretable — inflated toward reporting harm, which is the
    /// direction that wastes a session rather than hiding a bug, but wrong.
    #[test]
    fn asking_whether_an_entry_exists_is_not_charged_as_harm() {
        let mut state = state_capped(2 * FRAME_BYTES);
        let victim = 0x5_0000u64;
        store_frame(&mut state, victim, 0x11);
        for i in 0..64u64 {
            store_frame(&mut state, 0x300_0000 + i * 0x1000, i as u8);
        }
        assert!(!has_gva(&state, victim, W, H));
        touch_gva(&mut state, victim, W, H);
        assert_eq!(
            state.gva_eviction_witness.wanted.load(Relaxed),
            0,
            "probes do not read the pixels, so they are not denied any"
        );
        assert!(get_gva(&state, victim, W, H).is_none());
        assert_eq!(state.gva_eviction_witness.wanted.load(Relaxed), 1);
    }

    /// Once the content is back, later misses are a different question.
    ///
    /// Otherwise the witness keeps charging the cap for an identity that has
    /// been re-stored since, and `gva_cap_wanted` drifts upward for reasons
    /// that have nothing to do with the bound.
    #[test]
    fn a_store_that_brings_an_evicted_identity_back_stops_charging_the_cap() {
        let mut state = state_capped(2 * FRAME_BYTES);
        let victim = 0x5_0000u64;
        store_frame(&mut state, victim, 0x11);
        for i in 0..64u64 {
            store_frame(&mut state, 0x300_0000 + i * 0x1000, i as u8);
        }
        assert!(!state.host_gva_surfaces.contains_key(&victim));

        store_frame(&mut state, victim, 0x22);
        assert!(get_gva(&state, victim, W, H).is_some());
        // Evict it again by store pressure, but this time the witness has
        // forgotten it, so the miss below is not the cap's to answer for.
        state.gva_eviction_witness = crate::model::GvaEvictionWitness::default();
        assert!(get_gva(&state, 0x5_1000, W, H).is_none());
        assert_eq!(state.gva_eviction_witness.wanted.load(Relaxed), 0);
    }

    /// The ring bound must under-report visibly, never silently.
    ///
    /// It remembers a fixed number of evicted identities, so a long boot
    /// evicts more than it can hold. That makes `wanted` a lower bound, and a
    /// reader has to be able to tell — `forgotten` is what says so.
    #[test]
    fn forgetting_an_evicted_key_is_reported_rather_than_swallowed() {
        let mut state = state_capped(2 * FRAME_BYTES);
        let overflow_by = 64u64;
        let n = crate::model::GVA_EVICTION_WITNESS_KEYS as u64 + overflow_by;
        for i in 0..n {
            store_frame(&mut state, 0x1000 + i * 0x1000, i as u8);
        }
        let (evicted, _, forgotten) = state.gva_eviction_witness.counts();
        assert!(evicted > crate::model::GVA_EVICTION_WITNESS_KEYS as u64);
        assert!(
            forgotten > 0,
            "the ring overflowed and the census must be able to say so"
        );

        // The very first address evicted is the one the ring dropped, so a
        // lookup for it is uncounted — which is the point of `forgotten`.
        assert!(get_gva(&state, 0x1000, W, H).is_none());
        assert_eq!(
            state.gva_eviction_witness.wanted.load(Relaxed),
            0,
            "uncounted, and `forgotten` is the flag that keeps that honest"
        );
    }

    /// A store must not be undone by its own cap enforcement.
    ///
    /// Enforcement runs *after* the insert, so a single entry over the
    /// low-water mark is the only candidate in an otherwise empty map and
    /// evicts itself — the surface is then never cached at all, which is the
    /// "refused for being big" behaviour the cap explicitly must not have.
    ///
    /// Reachable in production, not only at test sizes: `MAX_SCANOUT_DIM` is
    /// 8192, so one entry may be 256 MiB against a 112 MiB low-water mark.
    #[test]
    fn an_entry_bigger_than_the_cap_is_admitted_alone_not_evicted_by_its_own_store() {
        let (w, h) = (64u32, 64u32);
        let big = (w * h * 4) as usize;
        let mut state = state_capped(big / 4);
        let gva = 0x8_0000u64;
        store_gva_owned(&mut state, gva, w, h, vec![0x77; big], 0, None);

        let served = get_gva(&state, gva, w, h)
            .expect("an oversized entry rides alone rather than being refused");
        assert!(served.iter().all(|&b| b == 0x77));
        assert_eq!(
            state.gva_cache_bytes, big,
            "and the total still describes it"
        );
        assert_eq!(
            state.gva_eviction_witness.evicted, 0,
            "nothing was evicted: there was nothing else to evict"
        );

        // It also does not pin the map forever — a later store at another
        // address makes it an ordinary candidate and the coldest thing present.
        store_frame(&mut state, 0x8_1000, 0x11);
        assert!(
            get_gva(&state, gva, w, h).is_none(),
            "once it is not the store's own key it is an ordinary eviction candidate"
        );
    }

    /// The cap tests a running total, so that total has to equal the map.
    ///
    /// A second source of truth is exactly how a bound silently stops bounding:
    /// under-count and the cap never fires, over-count and it evicts content it
    /// never needed to. This drives the three transitions that can break it —
    /// a fresh key, a replace at an existing key (which must net to the
    /// difference, not double-charge), and an eviction — and holds the total to
    /// the real sum at every step. `gva_cap_drift` is the same check, live.
    #[test]
    fn the_running_byte_total_equals_the_map_after_every_transition() {
        let truth = |state: &DeviceState| -> usize {
            state.host_gva_surfaces.values().map(|e| e.bgra.len()).sum()
        };
        let mut state = state_capped(usize::MAX);
        assert_eq!(state.gva_cache_bytes, 0);

        // Fresh keys.
        for i in 0..8u64 {
            store_frame(&mut state, 0x1000 + i * 0x1000, i as u8);
            assert_eq!(state.gva_cache_bytes, truth(&state), "after insert {i}");
        }
        assert_eq!(state.gva_cache_bytes, 8 * FRAME_BYTES);

        // Replace at an existing key, same size: the total must not move.
        store_frame(&mut state, 0x1000, 0xFF);
        assert_eq!(
            state.gva_cache_bytes,
            8 * FRAME_BYTES,
            "replace double-charged"
        );
        assert_eq!(state.gva_cache_bytes, truth(&state));

        // Replace at an existing key with a *different* geometry: the old bytes
        // are reclaimed and the new ones charged.
        let (w2, h2) = (W * 2, H);
        store_gva_owned(
            &mut state,
            0x1000,
            w2,
            h2,
            vec![0x5A; (w2 * h2 * 4) as usize],
            0,
            None,
        );
        assert_eq!(state.gva_cache_bytes, truth(&state), "geometry change");

        // Eviction.
        evict_gva(&mut state, 0x1000);
        assert_eq!(state.gva_cache_bytes, truth(&state), "after evict");
        assert_eq!(state.gva_cache_bytes, 7 * FRAME_BYTES);

        // And after the cap itself has run a batch of evictions.
        state.gva_cache_byte_cap = 4 * FRAME_BYTES;
        for i in 0..64u64 {
            store_frame(&mut state, 0x400_0000 + i * 0x1000, i as u8);
            assert_eq!(
                state.gva_cache_bytes,
                truth(&state),
                "under the cap, round {i}"
            );
        }
        assert!(
            state.gva_eviction_witness.evicted > 0,
            "the cap must have run"
        );
    }

    use std::sync::atomic::Ordering::Relaxed;
}
