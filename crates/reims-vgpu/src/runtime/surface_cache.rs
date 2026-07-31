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
//! - [`store_gva`] / [`get_gva`] — type-2/3 by target GVA (survives ref rebinding)
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
    map_generation: u32,
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
    entry.map_generation_at_store = map_generation;
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
    let map_generation = live_map_generation(state, surface_id);
    store_into(
        &mut state.host_surfaces,
        surface_id,
        width,
        height,
        bgra,
        generation,
        map_generation,
    );
}

/// The mapping incarnation `surface_id` is on right now, or 0 when this crate
/// holds no mapping under that id.
///
/// Zero is "no incarnation to speak of", not "incarnation zero": entries stored
/// and read while no mapping exists compare 0 against 0 and stay usable, which
/// keeps the texture_ref and GVA namespaces — and every unit test that stores
/// into a bare cache — behaving exactly as before.
fn live_map_generation(state: &DeviceState, surface_id: u32) -> u32 {
    state
        .mappings
        .get(&surface_id)
        .map_or(0, |m| m.map_generation)
}

/// Whether a cached surface_id frame was produced from the incarnation the
/// mapping is on now. Pure question, no counter and no knob.
///
/// Kept separate from [`surface_entry_may_serve`] because they are different
/// questions and an earlier draft here answered them with one function — which
/// its own test caught, by asserting a re-pointed entry was not current and
/// getting `true` back, because the knob was off. "These bytes are from the
/// incarnation in front of us" and "the device is willing to serve them anyway"
/// must not share a return value.
///
/// # Measured: this never fires, and the reason generalises
///
/// One 300 s crash-hunt boot, x86 / Vulkan: `surfcache_gen_same` 16 186,
/// `surfcache_gen_stale` **0**. The cache does not serve a frame across a
/// re-point on this workload, so it is not the mechanism behind the Finder icon
/// corruption or the Safari scroll patch. Kept anyway — the identity is real, it
/// costs a `u32` and a compare, and its absence is exactly what let the question
/// go unasked — but not counted as a repair.
///
/// The reason is arithmetic and it rules out a whole family of hypotheses.
/// `map_generation` moves 9-22 times in a 300 s boot while these rails do 25 000
/// to 51 000 operations in the same window, so *any* defect of the form "the
/// incarnation changed underneath a cached thing" is a ~0.04 % event. The icon
/// and patch symptoms are ones a user sees constantly. A rare mechanism cannot
/// explain a common symptom, and three separate cached-thing-outlives-its-
/// incarnation hypotheses have now measured zero or near zero on this rail
/// (`mapw_pages_refused`, `surface_resident_identity_split`, this). Look for
/// something that fires at content cadence instead.
fn surface_entry_is_current(state: &DeviceState, surface_id: u32) -> bool {
    let Some(entry) = state.host_surfaces.get(&surface_id) else {
        return true;
    };
    entry.map_generation_at_store == live_map_generation(state, surface_id)
}

/// Whether a surface_id lookup may hand back what it found, counting the answer
/// so a boot carries the rate either way.
///
/// Behaviour is deliberately unchanged at this commit. "How often does this
/// cache serve a frame across a re-point?" has never been measured, and refusing
/// on a guess is the wrong way round: a withheld Load seed renders the pass onto
/// a cleared target, which is a compositing layer going solid black, and this
/// project has already paid a boot for that failure direction (`13ae46d`, 0 of
/// 14 rounds). So this counts first and `REIMS_VGPU_SURFACE_CACHE_GEN_STRICT=1`
/// is what acts on it, which keeps an arm and its control one binary apart.
fn surface_entry_may_serve(state: &DeviceState, surface_id: u32) -> bool {
    if !state.host_surfaces.contains_key(&surface_id) {
        return true;
    }
    if surface_entry_is_current(state, surface_id) {
        crate::runtime::drain::note_store_route("surfcache_gen_same");
        return true;
    }
    crate::runtime::drain::note_store_route("surfcache_gen_stale");
    if let Some(entry) = state.host_surfaces.get(&surface_id) {
        crate::observe::off(format!(
            "surface_cache_stale_incarnation mid={surface_id} stored_gen={} live_gen={} {}x{} \
             (the cached frame was read out of a page list this mapping has since replaced)",
            entry.map_generation_at_store,
            live_map_generation(state, surface_id),
            entry.width,
            entry.height
        ));
    }
    !crate::observe::surface_cache_gen_strict()
}

/// Borrow host-cache frame when geom matches request (surface_id namespace).
pub fn get(state: &DeviceState, surface_id: u32, width: u32, height: u32) -> Option<&[u8]> {
    if !surface_entry_may_serve(state, surface_id) {
        return None;
    }
    get_from(&state.host_surfaces, surface_id, width, height)
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
    if !surface_entry_may_serve(state, surface_id) {
        return None;
    }
    let need = get_from(&state.host_surfaces, surface_id, width, height)?.len();
    let e = state.host_surfaces.get(&surface_id)?;
    Some(if e.bgra.len() == need {
        std::sync::Arc::clone(&e.bgra)
    } else {
        std::sync::Arc::new(e.bgra[..need].to_vec())
    })
}

/// Type-2/3 encode cache by texture object ref (not surface_id).
pub fn store_texture(
    state: &mut DeviceState,
    texture_ref: u32,
    width: u32,
    height: u32,
    bgra: Vec<u8>,
) {
    let generation = state.next_sampled_content_generation();
    store_into(
        &mut state.host_texture_surfaces,
        texture_ref,
        width,
        height,
        std::sync::Arc::new(bgra),
        generation,
        // A texture ref is not a mapping id, so there is no incarnation to name.
        0,
    );
}

pub fn get_texture(
    state: &DeviceState,
    texture_ref: u32,
    width: u32,
    height: u32,
) -> Option<&[u8]> {
    get_from(&state.host_texture_surfaces, texture_ref, width, height)
}

/// Any size under texture_ref (sample path when descriptor geom unknown).
pub fn get_texture_any(state: &DeviceState, texture_ref: u32) -> Option<(u32, u32, &[u8])> {
    let e = state.host_texture_surfaces.get(&texture_ref)?;
    if e.width == 0 || e.height == 0 || e.bgra.is_empty() {
        return None;
    }
    let need = (e.height as usize)
        .saturating_mul(e.width as usize)
        .saturating_mul(RGBA8_BPP as usize);
    if e.bgra.len() < need {
        return None;
    }
    Some((e.width, e.height, &e.bgra[..need]))
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

/// Land flushed resident bytes into the entry (tight rows), clearing the
/// resident-authoritative marker. No-op when the entry is gone or its
/// descriptor changed since the defer.
pub fn materialize_linear_resident(
    state: &mut DeviceState,
    task_id: u32,
    texture_ref: u32,
    generation: u32,
    bytes: &[u8],
) -> bool {
    let Some(entry) = state.host_linear_textures.get_mut(&(task_id, texture_ref)) else {
        return false;
    };
    if entry.resident_gen != generation {
        return false;
    }
    let Some(bpp) = crate::contract::pixel_format::bytes_per_pixel(entry.pixel_format) else {
        return false;
    };
    let Some(need) = (entry.width as usize)
        .checked_mul(entry.height as usize)
        .and_then(|n| n.checked_mul(bpp as usize))
    else {
        return false;
    };
    if bytes.len() < need {
        return false;
    }
    entry.bytes.clear();
    entry.bytes.extend_from_slice(&bytes[..need]);
    entry.resident_gen = 0;
    true
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
    store_texture(state, texture_ref, width, height, bgra.clone());
    let backing = gva_backing(state, host, task_id, gva, width, height);
    store_gva_owned(state, gva, width, height, bgra, 0, backing);
    arm_gva_guest_write_witness(state, host, gva);
}

/// Type-2/3 encode cache by target GVA.
///
/// On discrete hosts this is the **GPU-private** texture content for that VA.
/// Guest MapMemory2 unmap/remap changes PFNs under the same GVA but does **not**
/// destroy the encode — see [`note_unmap_retain_gva`] (Unmap retains; Map notify-only).
pub fn store_gva(state: &mut DeviceState, gva: u64, width: u32, height: u32, bgra: Vec<u8>) {
    store_gva_owned(state, gva, width, height, bgra, 0, None);
}

/// Guest pages currently backing `[gva, gva + width*height*4)` under `task_id`.
///
/// Returns `None` when the walk cannot name the backing at all — no task
/// directory, an unsupported page shift, or a geometry that overflows. A `None`
/// backing means the entry is simply not validatable, never that it is fresh.
pub fn gva_backing<M: HostMemory>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    gva: u64,
    width: u32,
    height: u32,
) -> Option<GvaBacking> {
    if gva == 0 {
        return None;
    }
    let span = (width as u64)
        .checked_mul(height as u64)?
        .checked_mul(RGBA8_BPP as u64)?;
    if span == 0 {
        return None;
    }
    let gpas = crate::runtime::gva_mem::task_gva_page_gpas_dense(
        host,
        &state.tasks,
        task_id,
        gva,
        span,
        state.page_shift,
    );
    if gpas.is_empty() {
        return None;
    }
    Some(GvaBacking {
        task_id,
        span,
        gpas,
    })
}

/// Store a GVA encode with the decoded object identity that produced it.
/// Type-2/type-3 wrappers are the same linear texture storage family when the
/// GVA and geometry match; unrelated nonzero object-type transitions still
/// identify a different resource class.
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
    let entry = state.host_gva_surfaces.entry(gva).or_default();
    entry.host_gen = generation;
    entry.width = width;
    entry.height = height;
    entry.bgra = std::sync::Arc::new(bgra);
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
    let pages_unchanged = match (entry.backing.as_ref(), backing.as_ref()) {
        (Some(old), Some(new)) => old.gpas == new.gpas,
        _ => false,
    };
    entry.backing = backing;
    entry.backing_suspect = false;
    if pages_unchanged {
        // The *baseline* still goes, because it named the previous bytes and
        // these are different ones. `arm_gva_guest_write_witness` re-reads it
        // against the surviving token immediately after this returns, which is
        // the only moment a baseline may be latched: the bytes are in hand.
        entry.guest_write_gen_at_store = 0;
        return;
    }
    // Any other change retires the token, including a `None` that says the walk
    // could not name the pages. This runs in the *store*, not in the arming
    // helper, because disarming is the half that must never be forgotten: an
    // entry left holding a token for someone else's pages would report
    // "unwritten" for pages it no longer owns. Re-arming is the caller's option
    // (see [`arm_gva_guest_write_witness`]); disarming is not.
    let stale = std::mem::replace(&mut entry.guest_write_token, 0);
    entry.guest_write_gen_at_store = 0;
    if stale != 0 {
        // Only the host can free host-side tracking state, and this function
        // has no host. `flush_retired_views` drains the list.
        state.retired_guest_write_tokens.push(stale);
    }
}

/// Ask the host to watch the guest pages a freshly stored GVA entry was
/// produced from, and record the generation those bytes are current as of.
///
/// Called by the stores that have a host to ask. Split from
/// [`store_gva_owned`] rather than folded into it because the two halves have
/// opposite failure directions: a store that fails to *disarm* serves a stale
/// picture, while a store that fails to *arm* only makes the next read fall
/// through to the guest's own pages. The dangerous half is unconditional and
/// host-free; this one is best-effort.
///
/// A host with no dirty bitmap answers `None` from `track_guest_writes` and the
/// entry stays unarmed forever, which every reader takes as "assume written".
pub fn arm_gva_guest_write_witness<H: crate::runtime::host::HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    gva: u64,
) {
    let page_size = state.page_size() as usize;
    let Some(entry) = state.host_gva_surfaces.get(&gva) else {
        return;
    };
    // Already tracking this page list: `store_gva_owned` clears the token
    // whenever it replaces the backing, so a live token is by construction a
    // token for the pages the current bytes were produced from. Keep it —
    // re-registering costs a host call per store — but **re-read its
    // generation**, because this runs immediately after the bytes were stored
    // and the baseline has to name the bytes in hand.
    //
    // Latching once was the defect. `reims_vgpu_dirty_gen` holds a set's
    // generation at 0 for a deliberate two-harvest startup window ("an absence
    // of reports says nothing about the guest yet"), and this function runs
    // directly after `track_guest_writes` — always inside that window. So the
    // baseline was recorded as 0, always, and the early return meant it was
    // never revisited. A real generation is never 0, so
    // `gva_guest_wrote_since_store` could never match one, and `gvac_gw_clean`
    // was unreachable by construction: 0 of 201 331 lookups on the last boot.
    // `stamp_guest_write_gen` is the same rail done correctly on the mapping
    // side, and it re-stamps on every store for exactly this reason.
    if entry.guest_write_token != 0 {
        let token = entry.guest_write_token;
        let gen_at_store = host.guest_write_gen(token).unwrap_or(0);
        crate::runtime::drain::note_store_route(if gen_at_store == 0 {
            "gvac_gw_restamp_unarmed"
        } else {
            "gvac_gw_restamp_armed"
        });
        if let Some(entry) = state.host_gva_surfaces.get_mut(&gva) {
            entry.guest_write_gen_at_store = gen_at_store;
        }
        return;
    }
    let Some(backing) = entry.backing.as_ref() else {
        return;
    };
    if backing.gpas.is_empty() {
        return;
    }
    // `task_gva_page_gpas_dense` keeps one slot per page and writes `0` where a
    // page did not resolve, because its other caller compares whole lists as a
    // mapping identity and holes have to keep their positions. A hole is not an
    // address, so it must not be armed: `track_guest_writes` would watch guest
    // page 0 on this surface's behalf, and page 0 is busy, so the witness would
    // report writes that are nothing to do with these pixels.
    //
    // Dropping the holes and arming the rest is the other option and is worse.
    // This witness is asked whether the guest wrote a surface, and a partial
    // watch can only answer "clean" about a page it never watched — a false
    // clean is what lets the device reuse content the guest has replaced. An
    // unarmed witness says "unknown", which every reader already handles,
    // because the host shim maps token 0 to `None` during its own startup
    // window.
    if backing.gpas.contains(&0) {
        crate::runtime::drain::note_store_route("gvac_gw_hole_unarmed");
        return;
    }
    let gpas = backing.gpas.clone();
    let Some(token) = host.track_guest_writes(&gpas, page_size) else {
        return;
    };
    // Read the generation *after* registration, so the recorded value is one
    // the host could actually have produced for this token.
    let gen_at_store = host.guest_write_gen(token).unwrap_or(0);
    let Some(entry) = state.host_gva_surfaces.get_mut(&gva) else {
        // The entry vanished between the borrow above and here (no path does
        // this today, but the token is real either way and must not leak).
        state.retired_guest_write_tokens.push(token);
        return;
    };
    entry.guest_write_token = token;
    entry.guest_write_gen_at_store = gen_at_store;
}

/// Has the guest CPU written this entry's pages since its bytes were stored?
///
/// `true` is the answer for every way the question cannot be settled — no
/// entry, no token, a host that cannot observe guest writes — because the only
/// safe default for a cache of somebody else's memory is that it is stale. The
/// caller falls through to the guest's own pages, which are authoritative by
/// construction.
///
/// Each refusal is counted apart from the others: "this rail was never armed"
/// and "the guest rewrites this texture every frame" are the same fall-through
/// and completely different findings.
///
/// # What it measured, including what it did not fix
///
/// One 14-round Finder recomposite boot under load, x86 / Vulkan:
/// **`gvac_gw_clean` is zero.** Not small — zero, across every window of the
/// whole boot, against `gvac_gw_wrote` running 26-1757 per round. Every entry
/// this cache still held had been rewritten by the guest since it was stored.
/// That is independent confirmation of the probe's byte compare
/// (`gvac_content_differ` == `gvac_content_checked`) from a different
/// instrument on a different boot: the rail was serving a stale picture
/// essentially every time it served at all.
///
/// **It did not close the icon class.** Seven of fourteen rounds still
/// corrupted, so the wrong pixels were not coming from here, or not only from
/// here. What the counter did buy is the first quantity that separates a
/// corrupt round from a clean one, after a session of counters that could not:
///
/// ```text
/// clean rounds    gvac_gw_wrote  26  279  306  486  634  662
/// corrupt rounds                255  346 1445 1571 1639 1674 1752 1757
/// ```
///
/// Normalised per 1000 draws (`draw_scissor_full`, since every other draw-path
/// counter in this census is strictly proportional to round length and none of
/// them separates anything), it is **bimodal with an empty gap**:
///
/// ```text
/// low  mode  3.1  23.5  32.7  49.6  57.5  68.8   all six CLEAN rounds
///           28.4  30.4                          + two corrupt rounds (2, 7)
/// high mode 126.7 134.3 142.1 142.5 145.0 147.3  six CORRUPT rounds, no clean
/// ```
///
/// Nothing lands between 68.8 and 126.7, and no clean round reaches the high
/// mode. Read it as a load proxy and not as a cause — the rail is fully refused
/// now, so what this counts is how hard the guest is recycling and rewriting
/// texture memory *in place*. Above roughly 120 per 1000 draws the round
/// corrupts every time.
///
/// That is the condition under which naming a resource by its address stops
/// working. The engine's resident registry no longer does: `TargetIdentity::Gva`
/// carries the hash of the guest pages behind the target as its `generation`
/// (`metal_draw::vulkan::gva_alloc_generation`), so two allocations reusing one
/// address at one geometry get two slots rather than one shared image — the same
/// treatment `surface_identity` gives the `Surface` rail with `map_generation`.
/// This counter stays as the load proxy: it says how hard the guest is recycling
/// texture memory, which is the condition, not the mechanism.
///
/// The two corrupt rounds in the low mode are the reminder that this class has
/// been two defects since it was first split (see
/// [`crate::observe::sink::content_reuse_disabled`]): a high-recycling round
/// corrupts reliably, and something else corrupts occasionally regardless.
///
/// # Cross-validated on a second boot, and it earns its keep by refusing credit
///
/// The next 14-round boot scored **1 corrupt of 14**, against 7 of 14 on the
/// one above, with a fix landed in between. Read as a rate that would have been
/// a result. Scored against this counter it is not: *every round of that boot
/// was in the low mode* (peak 74.4, against a corrupting threshold of ~127), so
/// the model predicted no high-mode corruption and there was none. The
/// improvement is the load, not the change.
///
/// Pooled over both boots the split holds:
///
/// ```text
/// high mode  6 rounds   6 corrupt   100 %
/// low  mode 22 rounds   3 corrupt    14 %
/// ```
///
/// This is why the raw corrupt-round count must not be used to score a change
/// on this class, and it is the first quantity here that can say so. A boot that
/// never enters the high mode cannot confirm or refute a fix for the reliable
/// defect, and two of the three boots taken this session did not.
///
/// # RETRACTED: this counter does not predict the icon class
///
/// The model above was built on two boots and a third refutes it. Three
/// 14-round boots on one binary, after the GVA rail was bounded by the fence:
///
/// ```text
/// boot          gvac_gw_wrote/1000 draws   rounds corrupt
/// icon-fence            42.4                    0 of 14
/// icon-fence2           79.5                    0 of 14
/// hi-mode              109.9 -> see below       0 of 14   (clean arm)
/// hi-mode2              74.5                   14 of 14
/// ```
///
/// The all-corrupt boot has a **lower** normalised count than the all-clean one
/// it is paired against (74.5 against 109.9), and neither reaches the ~127 the
/// model calls the corrupting threshold. So the bimodal split does not survive
/// contact with a third and fourth boot, and no conclusion above that rests on
/// "this boot was in the low mode" should be trusted — including the caution
/// that a low-mode boot cannot score a fix. It cannot, but not for this reason.
///
/// What DOES separate those two boots is not instantaneous load at all: the
/// corrupt one had been driven for 600 s (Mission Control, Spotlight, window
/// drags) before the icon harness started, and the clean one was fresh. The
/// class tracks accumulated session history. The counters that move with it are
/// remap counters — `gvac_suspect` 15.8x, `gvac_moved` 10.5x, and `rmemo_stale`
/// 0 -> 19, the last of which had never been observed at all (previously zero
/// over 1 639 738 verified hits) and means the guest re-pointed a GPU-mapped
/// range with no notification arriving first.
///
/// That is a regime where naming a resource by its address stops working, which
/// is what the paragraph above this section already said the open work was.
pub fn gva_guest_wrote_since_store<H: crate::runtime::host::HostOps>(
    state: &DeviceState,
    host: &H,
    gva: u64,
) -> bool {
    let Some(entry) = state.host_gva_surfaces.get(&gva) else {
        crate::runtime::drain::note_store_route("gvac_gw_no_entry");
        return true;
    };
    if entry.guest_write_token == 0 {
        crate::runtime::drain::note_store_route("gvac_gw_unarmed");
        return true;
    }
    // A baseline of 0 is not a generation, it is the absence of one: the set was
    // still inside its startup window at the store that recorded it. Named apart
    // from `unreadable` because the two have opposite follow-ups — this one says
    // the *device* has no reference point, `unreadable` says the *host* cannot
    // answer right now — and pooling them is what let a boot be read as "every
    // entry had been rewritten by the guest" when most of the population had
    // simply never had a readable baseline. The mapping rail draws the same
    // distinction (`mapping_guest_write_verdict` returns `NoStamp` for
    // `guest_write_gen_at_store == 0`).
    if entry.guest_write_gen_at_store == 0 {
        crate::runtime::drain::note_store_route("gvac_gw_no_baseline");
        return true;
    }
    match host.guest_write_gen(entry.guest_write_token) {
        Some(gen_) if gen_ == entry.guest_write_gen_at_store => {
            crate::runtime::drain::note_store_route("gvac_gw_clean");
            false
        }
        Some(_) => {
            crate::runtime::drain::note_store_route("gvac_gw_wrote");
            true
        }
        None => {
            crate::runtime::drain::note_store_route("gvac_gw_unreadable");
            true
        }
    }
}

/// Whether a GVA-keyed entry's recorded backing is still what the GVA names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackingVerdict {
    /// No page-table mapping change has touched this span since the store.
    Unchanged,
    /// The guest remapped the span and the pages came back identical — the
    /// mapping churned, the allocation did not.
    Confirmed,
    /// The guest remapped the span and it now resolves to different pages:
    /// this GVA names a different allocation than the one these bytes are of.
    Moved,
    /// Nothing was recorded to compare against (store-time walk failed, or a
    /// non-GVA cache), so this entry cannot answer the question either way.
    Unrecorded,
}

/// Re-walk a suspect GVA entry's span and compare it against the pages the
/// stored bytes were produced from.
///
/// Called on the sampled read path *before* the lookup. A suspect entry costs
/// one page-table walk once per guest remap of its span, not one per sample:
/// [`BackingVerdict::Confirmed`] clears the flag.
pub fn revalidate_gva_backing<M: HostMemory>(
    state: &mut DeviceState,
    host: &M,
    task_id: u32,
    gva: u64,
) -> BackingVerdict {
    let Some(entry) = state.host_gva_surfaces.get(&gva) else {
        return BackingVerdict::Unrecorded;
    };
    if !entry.backing_suspect {
        return if entry.backing.is_some() {
            BackingVerdict::Unchanged
        } else {
            BackingVerdict::Unrecorded
        };
    }
    let Some(recorded) = entry.backing.clone() else {
        return BackingVerdict::Unrecorded;
    };
    // The walk must use the task the pixels were produced under. A GVA has no
    // meaning apart from its page table, so re-resolving a different task's
    // table would compare two unrelated address spaces and call the difference
    // a move.
    if recorded.task_id != task_id {
        return BackingVerdict::Unrecorded;
    }
    let gpas = crate::runtime::gva_mem::task_gva_page_gpas_dense(
        host,
        &state.tasks,
        task_id,
        gva,
        recorded.span,
        state.page_shift,
    );
    if gpas == recorded.gpas {
        if let Some(entry) = state.host_gva_surfaces.get_mut(&gva) {
            entry.backing_suspect = false;
        }
        return BackingVerdict::Confirmed;
    }
    BackingVerdict::Moved
}

/// Mark every GVA entry whose recorded span overlaps `[gva, gva+length)` under
/// `task_id` as needing revalidation before its next read.
///
/// The Unmap/MapMemory2 notify is the guest telling us a virtual range now
/// points somewhere else. This cache is deliberately **retained** across that
/// notify (a mapping that churns and comes back must not black out the
/// wallpaper), so the notify cannot evict — but it can record that the entry's
/// name is no longer known-good, and make the next reader prove it.
///
/// Returns the number of entries marked.
pub fn mark_gva_backing_suspect(
    state: &mut DeviceState,
    task_id: u32,
    gva: u64,
    length: u64,
) -> u32 {
    if length == 0 {
        return 0;
    }
    let mut n = 0u32;
    for (&entry_gva, entry) in state.host_gva_surfaces.iter_mut() {
        let Some(backing) = entry.backing.as_ref() else {
            continue;
        };
        // Widened task match, like every other invalidation this notify does.
        // A mark is not a verdict — [`revalidate_gva_backing`] re-walks and
        // decides — so an over-mark costs one walk that comes back
        // `Confirmed`, while a missed mark leaves a stale entry serving with
        // nobody ever asking it to prove itself.
        if !crate::runtime::gva_view::task_matches(backing.task_id, task_id) || entry.backing_suspect
        {
            continue;
        }
        if crate::runtime::gva_view::ranges_overlap(entry_gva, backing.span, gva, length) {
            entry.backing_suspect = true;
            n = n.saturating_add(1);
        }
    }
    n
}

pub fn get_gva(state: &DeviceState, gva: u64, width: u32, height: u32) -> Option<&[u8]> {
    get_gva_with_gen(state, gva, width, height).map(|(bgra, _)| bgra)
}

/// Whether a [`get_gva`] for this key would hit, without borrowing the bytes.
///
/// Lets a caller that needs `&mut DeviceState` (backing revalidation) find out
/// first whether there is anything to revalidate.
pub fn has_gva(state: &DeviceState, gva: u64, width: u32, height: u32) -> bool {
    get_gva_with_gen(state, gva, width, height).is_some()
}

/// Borrow a GVA encode plus its producer generation.
///
/// This is diagnostic provenance for the linear-sample loss proxy; selection
/// semantics are identical to [`get_gva`].
pub fn get_gva_with_gen(
    state: &DeviceState,
    gva: u64,
    width: u32,
    height: u32,
) -> Option<(&[u8], u64)> {
    let e = state.host_gva_surfaces.get(&gva)?;
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

/// Borrow a GVA encode plus its decoded producer object type.
pub fn get_gva_with_owner(
    state: &DeviceState,
    gva: u64,
    width: u32,
    height: u32,
) -> Option<(&[u8], u64, u8)> {
    let e = state.host_gva_surfaces.get(&gva)?;
    if e.width != width || e.height != height || e.bgra.is_empty() {
        return None;
    }
    let need = (height as usize)
        .saturating_mul(width as usize)
        .saturating_mul(RGBA8_BPP as usize);
    if e.bgra.len() < need {
        return None;
    }
    Some((&e.bgra[..need], e.host_gen, e.producer_object_type))
}

/// Explicit drop (tests / object delete). Prefer [`note_unmap_retain_gva`] on Unmap.
pub fn evict_gva(state: &mut DeviceState, gva: u64) {
    // The entry's guest-write token is host-side state for a page list that no
    // longer has an owner here; dropping the entry without retiring it would
    // leak one tracking set per deleted texture VA.
    if let Some(entry) = state.host_gva_surfaces.remove(&gva) {
        if entry.guest_write_token != 0 {
            state
                .retired_guest_write_tokens
                .push(entry.guest_write_token);
        }
    }
}

/// Drop host-cache entry (unmap / delete surface).
pub fn evict(state: &mut DeviceState, surface_id: u32) {
    state.host_surfaces.remove(&surface_id);
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::model::{DeviceId, PAGE_SHIFT_ARM64E};
    use crate::runtime::host::FakeHost;

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

        store_gva(&mut st, gva, w, h, vec![0x11; (w * h * 4) as usize]);
        let (_, first, _) = get_gva_with_owner(&st, gva, w, h).expect("first store");

        evict_gva(&mut st, gva);
        assert!(
            get_gva_with_owner(&st, gva, w, h).is_none(),
            "the arm removes the entry outright"
        );

        store_gva(&mut st, gva, w, h, vec![0x22; (w * h * 4) as usize]);
        let (bytes, second, _) = get_gva_with_owner(&st, gva, w, h).expect("second store");

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
        seen.insert(get_from_with_gen(&st.host_surfaces, 7, 4, 4).expect("mid store").1);
        store_texture(&mut st, 9, 4, 4, px.clone());
        seen.insert(
            get_from_with_gen(&st.host_texture_surfaces, 9, 4, 4)
                .expect("ref store")
                .1,
        );
        store_gva(&mut st, 0x5000, 4, 4, px);
        seen.insert(get_gva_with_gen(&st, 0x5000, 4, 4).expect("gva store").1);
        assert!(
            cede_surface_to_resident(&mut st, 7, 4, 4),
            "cession is a state change and must take a generation too"
        );
        seen.insert(st.host_surfaces.get(&7).expect("ceded entry").host_gen);

        assert_eq!(seen.len(), 4, "four stores, four distinct generations");
        assert!(!seen.contains(&0), "0 is reserved for 'no host content yet'");
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
        assert!(!materialize_linear_resident(&mut st, task, r, 9, &flushed));
        assert!(materialize_linear_resident(&mut st, task, r, 2, &flushed));
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
        evict(&mut st, 7);
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
        store_texture(&mut st, 5, w, h, tex);
        assert_eq!(get(&st, 5, w, h).unwrap()[0], 1);
        assert_eq!(get_texture(&st, 5, w, h).unwrap()[0], 2);
    }

    #[test]
    fn gva_cache_roundtrip() {
        let mut st = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let gva = 0x2c48000u64;
        let mut px = vec![0u8; 16];
        px[0] = 0xaa;
        store_gva(&mut st, gva, 2, 2, px);
        assert_eq!(get_gva(&st, gva, 2, 2).unwrap()[0], 0xaa);
        let (got, generation) = get_gva_with_gen(&st, gva, 2, 2).unwrap();
        assert_eq!(got[0], 0xaa);
        assert_eq!(generation, 1);
        assert!(get_gva_with_gen(&st, gva, 4, 1).is_none());

        let mut replacement = vec![0u8; 16];
        replacement[0] = 0xbb;
        store_gva(&mut st, gva, 2, 2, replacement);
        let (got, generation) = get_gva_with_gen(&st, gva, 2, 2).unwrap();
        assert_eq!(got[0], 0xbb);
        assert_eq!(generation, 2);

        let mut owned = vec![0u8; 16];
        owned[0] = 0xcc;
        store_gva_owned(&mut st, gva, 2, 2, owned, 2, None);
        let (got, generation, object_type) = get_gva_with_owner(&st, gva, 2, 2).unwrap();
        assert_eq!(got[0], 0xcc);
        assert_eq!(generation, 3);
        assert_eq!(object_type, 2);
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
        store_gva(&mut st, gva, w, h, px);
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

    /// The GVA encode cache is keyed by a *name*, and the guest reassigns names.
    ///
    /// `host_gva_surfaces` is retained across Unmap on purpose, so nothing on
    /// the notify path evicts it; without a recorded backing, an entry stored
    /// for one 64x64 icon is served for whatever the guest points that address
    /// at next, at the same geometry, indefinitely. The backing is what lets a
    /// reader tell the two apart: a mapping that churned and came back reads
    /// `Confirmed`, and a mapping now resolving elsewhere reads `Moved`.
    #[test]
    fn gva_backing_separates_a_churned_mapping_from_a_reassigned_address() {
        let mut host = FakeHost::new();
        let mut st = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let root_gpa = setup_depth1_task(&mut host, &mut st);
        // 64x64 BGRA8 is 16 KiB — exactly one arm64e page, one PTE to re-point.
        let (w, h) = (64u32, 64u32);
        let page = 1u64 << PAGE_SHIFT_ARM64E;
        let gva = page;
        let backing = gva_backing(&st, &host, 1, gva, w, h).expect("walk resolves");
        assert_eq!(backing.gpas.len(), 1, "64x64 BGRA8 covers one 16 KiB page");
        assert_eq!(backing.span, (w as u64) * (h as u64) * 4);
        store_gva_owned(
            &mut st,
            gva,
            w,
            h,
            vec![0xA5u8; (w * h * 4) as usize],
            0,
            Some(backing),
        );

        // No notify has touched the span, so there is nothing to prove and no
        // walk to pay for.
        assert_eq!(
            revalidate_gva_backing(&mut st, &host, 1, gva),
            BackingVerdict::Unchanged
        );

        // Unmap + Map2 of the same range restoring the same PFNs: the retained
        // wallpaper class. The pixels are still this allocation's pixels.
        assert_eq!(mark_gva_backing_suspect(&mut st, 1, gva, page), 1);
        assert_eq!(
            revalidate_gva_backing(&mut st, &host, 1, gva),
            BackingVerdict::Confirmed
        );
        // Confirming clears the flag: the cost is one walk per remap, not one
        // walk per sample.
        assert_eq!(
            revalidate_gva_backing(&mut st, &host, 1, gva),
            BackingVerdict::Unchanged
        );

        // The guest hands the same address to a different allocation.
        assert_eq!(mark_gva_backing_suspect(&mut st, 1, gva, page), 1);
        repoint_pte(&mut host, root_gpa, 1, 12);
        assert_eq!(
            revalidate_gva_backing(&mut st, &host, 1, gva),
            BackingVerdict::Moved
        );
        // A `Moved` entry stays suspect. Nothing about asking twice makes the
        // pixels belong to the new owner.
        assert_eq!(
            revalidate_gva_backing(&mut st, &host, 1, gva),
            BackingVerdict::Moved
        );
        // The lookup itself is unchanged by this commit: the census scores, it
        // does not gate.
        assert!(get_gva(&st, gva, w, h).is_some());
    }

    /// The suspect mark follows the span and the task, not every entry.
    ///
    /// A notify that marks unrelated entries costs a page-table walk each and
    /// makes `gvac_moved` unreadable; one that marks across address spaces
    /// would compare a GVA against a table that never named it.
    #[test]
    fn suspect_marking_is_bounded_by_span_and_task() {
        let mut host = FakeHost::new();
        let mut st = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        setup_depth1_task(&mut host, &mut st);
        let page = 1u64 << PAGE_SHIFT_ARM64E;
        let (w, h) = (64u32, 64u32);
        for i in 1..4u64 {
            let gva = i * page;
            let backing = gva_backing(&st, &host, 1, gva, w, h).expect("walk resolves");
            store_gva_owned(
                &mut st,
                gva,
                w,
                h,
                vec![0u8; (w * h * 4) as usize],
                0,
                Some(backing),
            );
        }
        // An entry stored with no recorded backing cannot be marked — there is
        // nothing to compare it against, and it reports `Unrecorded` forever.
        store_gva_owned(&mut st, 9 * page, w, h, vec![0u8; (w * h * 4) as usize], 0, None);

        // Exactly the two entries the range overlaps.
        assert_eq!(mark_gva_backing_suspect(&mut st, 1, 2 * page, 2 * page), 2);
        assert_eq!(
            revalidate_gva_backing(&mut st, &host, 1, page),
            BackingVerdict::Unchanged
        );
        assert_eq!(
            revalidate_gva_backing(&mut st, &host, 1, 2 * page),
            BackingVerdict::Confirmed
        );
        assert_eq!(
            revalidate_gva_backing(&mut st, &host, 1, 9 * page),
            BackingVerdict::Unrecorded
        );

        // An unrelated task's notify names an unrelated address space.
        assert_eq!(mark_gva_backing_suspect(&mut st, 5, page, 8 * page), 0);
        // The shift-aliased id is marked on purpose: the mark is not the
        // verdict, so over-marking costs a walk and under-marking costs a
        // stale serve that nobody ever asks to prove itself. Two of the three
        // entries in range — the one at `3 * page` is still suspect from the
        // first notify and is not counted twice.
        assert_eq!(mark_gva_backing_suspect(&mut st, 2, page, 8 * page), 2);
        // Revalidation is the strict end. A GVA has no meaning apart from its
        // page table, so a reader on another task cannot answer the question —
        // walking its table would compare two address spaces and report the
        // difference as a move.
        assert_eq!(
            revalidate_gva_backing(&mut st, &host, 2, page),
            BackingVerdict::Unrecorded
        );
    }

    /// An unresolved page is part of the identity, not a gap to be closed over.
    #[test]
    fn dense_page_walk_keeps_holes_in_place() {
        let mut host = FakeHost::new();
        let mut st = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let root_gpa = setup_depth1_task(&mut host, &mut st);
        let page = 1u64 << PAGE_SHIFT_ARM64E;
        // Three pages: 64x192 BGRA8 = 48 KiB.
        let (w, h) = (64u32, 192u32);
        let gva = page;
        let full = gva_backing(&st, &host, 1, gva, w, h).expect("walk resolves");
        assert_eq!(full.gpas.len(), 3);

        // Punt the middle page's PTE to an invalid PFN.
        repoint_pte(&mut host, root_gpa, 2, 0);
        let holed = gva_backing(&st, &host, 1, gva, w, h).expect("walk still names the span");
        assert_eq!(holed.gpas.len(), 3, "one slot per page, holes included");
        assert_eq!(holed.gpas[1], 0, "the hole sits where the page is");
        assert_ne!(holed.gpas, full.gpas);
    }

    /// Store a 64x64 BGRA8 entry at `gva` with its backing walked and its
    /// guest-write witness armed — the shape both product stores produce.
    fn store_armed(st: &mut DeviceState, host: &mut FakeHost, gva: u64, fill: u8) -> u64 {
        let (w, h) = (64u32, 64u32);
        let backing = gva_backing(st, host, 1, gva, w, h).expect("walk resolves");
        let gpa = backing.gpas[0];
        store_gva_owned(
            st,
            gva,
            w,
            h,
            vec![fill; (w * h * 4) as usize],
            0,
            Some(backing),
        );
        arm_gva_guest_write_witness(st, host, gva);
        gpa
    }

    /// The verdict this cache had could not see a guest CPU write, and that is
    /// the one way it goes wrong that nothing else catches.
    ///
    /// `backing_suspect` is driven by the Unmap/Map notify: it answers whether
    /// this GVA still *names* these pages. A guest CPU store into pages that
    /// never moved issues no notify, touches no device operation, and moves no
    /// generation on this side — so the entry passes `Unchanged` and keeps
    /// serving an icon the guest has already replaced. Measured live at 64x64
    /// with the content probe on: every audited serve of this rail disagreed
    /// with the guest's own pages (`gvac_content_differ` == `gvac_content_checked`).
    ///
    /// The hypervisor's dirty bitmap is the only witness that can see it, and
    /// asking it is what this test asserts.
    #[test]
    fn a_guest_cpu_write_invalidates_a_gva_entry_no_notify_can_see() {
        let mut host = FakeHost::new();
        let mut st = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        setup_depth1_task(&mut host, &mut st);
        let gva = 1u64 << PAGE_SHIFT_ARM64E;
        let gpa = store_armed(&mut st, &mut host, gva, 0xA5);

        // Nothing has touched the pages: the entry is current and serves.
        assert!(
            !gva_guest_wrote_since_store(&st, &host, gva),
            "a freshly stored entry whose pages nobody wrote is current"
        );
        assert_eq!(
            revalidate_gva_backing(&mut st, &host, 1, gva),
            BackingVerdict::Unchanged,
            "no notify has touched the span"
        );

        // The guest CPU rewrites the texture in place. No notify, no remap, no
        // device operation — the backing verdict cannot move, and does not.
        host.guest_wrote_page(gpa);
        assert_eq!(
            revalidate_gva_backing(&mut st, &host, 1, gva),
            BackingVerdict::Unchanged,
            "the pages did not move, so the backing verdict still says they did not"
        );
        assert!(
            gva_guest_wrote_since_store(&st, &host, gva),
            "the host observed the write, so the cached bytes are stale"
        );

        // Storing the guest's new bytes re-arms against the same pages and the
        // entry is current again — the rail recovers rather than latching off.
        store_armed(&mut st, &mut host, gva, 0x5A);
        assert!(
            !gva_guest_wrote_since_store(&st, &host, gva),
            "a store after the write re-arms the witness"
        );
    }

    /// The case this cache exists for must survive the new refusal.
    ///
    /// A mapping that churns and comes back — Unmap then Map2 restoring the
    /// same PFNs — is the retained-wallpaper class. The pixels are still this
    /// allocation's pixels, and nothing wrote them, so the entry must still
    /// serve. A witness that refused here would black out the wallpaper to fix
    /// an icon.
    #[test]
    fn a_mapping_that_churned_without_a_guest_write_still_serves() {
        let mut host = FakeHost::new();
        let mut st = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        setup_depth1_task(&mut host, &mut st);
        let page = 1u64 << PAGE_SHIFT_ARM64E;
        let gva = page;
        store_armed(&mut st, &mut host, gva, 0xA5);

        assert_eq!(mark_gva_backing_suspect(&mut st, 1, gva, page), 1);
        assert_eq!(
            revalidate_gva_backing(&mut st, &host, 1, gva),
            BackingVerdict::Confirmed,
            "the same pages came back"
        );
        assert!(
            !gva_guest_wrote_since_store(&st, &host, gva),
            "a remap is not a write; the retained encode is still the guest's picture"
        );
    }

    /// A token watches a page list, so it must not outlive the list it was
    /// taken for.
    ///
    /// `store_gva_owned` replaces `backing` unconditionally, and the token has
    /// to go with it: carried forward, it would report "unwritten" about pages
    /// this entry no longer claims — an entry vouching for itself with somebody
    /// else's witness. Disarming lives in the store for exactly this reason,
    /// while arming is the caller's option.
    #[test]
    fn replacing_the_backing_retires_the_token_taken_for_the_old_pages() {
        let mut host = FakeHost::new();
        let mut st = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let root_gpa = setup_depth1_task(&mut host, &mut st);
        let page = 1u64 << PAGE_SHIFT_ARM64E;
        let gva = page;
        let first_gpa = store_armed(&mut st, &mut host, gva, 0xA5);
        assert_eq!(host.tracked_guest_write_sets(), 1);

        // The guest points the address at a different allocation and the next
        // store records the new pages.
        repoint_pte(&mut host, root_gpa, 1, 12);
        let second_gpa = store_armed(&mut st, &mut host, gva, 0x5A);
        assert_ne!(first_gpa, second_gpa, "the address names different pages now");

        // A write to the *old* pages says nothing about this entry.
        host.guest_wrote_page(first_gpa);
        assert!(
            !gva_guest_wrote_since_store(&st, &host, gva),
            "the old page list is not this entry's page list"
        );
        // A write to the pages it actually holds does.
        host.guest_wrote_page(second_gpa);
        assert!(gva_guest_wrote_since_store(&st, &host, gva));

        // And the first token is released rather than leaked.
        crate::runtime::mapper::flush_retired_views(&mut st, &mut host);
        assert_eq!(
            host.tracked_guest_write_sets(),
            1,
            "one live set for the current page list, the superseded one released"
        );
    }

    /// Every way the question cannot be settled reads as "written".
    ///
    /// An entry stored without a resolvable backing has no page list to watch,
    /// and a host with no dirty bitmap has nothing to watch it with. Both leave
    /// the entry unarmed, and an unarmed entry must fall through to the guest's
    /// own pages — forgetting to arm costs a re-read, never a wrong picture.
    #[test]
    fn an_unarmed_gva_entry_is_read_as_written() {
        let mut st = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let mut host = FakeHost::new();
        setup_depth1_task(&mut host, &mut st);
        let gva = 1u64 << PAGE_SHIFT_ARM64E;

        // No backing recorded: nothing to register.
        store_gva(&mut st, gva, 64, 64, vec![0xA5u8; 64 * 64 * 4]);
        arm_gva_guest_write_witness(&mut st, &mut host, gva);
        assert_eq!(host.tracked_guest_write_sets(), 0);
        assert!(gva_guest_wrote_since_store(&st, &host, gva));

        // A host that cannot observe guest writes answers `None` forever.
        let mut blind = FakeHost::new();
        blind.guest_writes_unobservable = true;
        let mut st2 = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        setup_depth1_task(&mut blind, &mut st2);
        store_armed(&mut st2, &mut blind, gva, 0xA5);
        assert_eq!(blind.tracked_guest_write_sets(), 0);
        assert!(
            gva_guest_wrote_since_store(&st2, &blind, gva),
            "an incapable host leaves every entry unwitnessed, so every entry is stale"
        );

        // An address with no entry at all is not a silent pass either.
        assert!(gva_guest_wrote_since_store(&st, &host, 0xdead_0000));
    }

    /// A hole in the page list is not an address, so it must not be watched.
    ///
    /// `task_gva_page_gpas_dense` writes `0` where a page did not resolve,
    /// because its other caller compares whole lists as a mapping identity and
    /// the holes have to keep their positions. Arming that list registers guest
    /// page 0 on this surface's behalf, and page 0 is busy, so the witness would
    /// answer with traffic that has nothing to do with these pixels.
    ///
    /// Dropping the holes and arming the rest is the tempting alternative and is
    /// the worse one: a partial watch can only ever answer "clean" about a page
    /// it never watched, and a false clean is what lets the device keep content
    /// the guest has already replaced. Unarmed reads as written, which costs a
    /// re-read and never a wrong picture.
    #[test]
    fn a_backing_with_a_hole_is_not_armed() {
        let mut st = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let mut host = FakeHost::new();
        let root_gpa = setup_depth1_task(&mut host, &mut st);
        let page = 1u64 << PAGE_SHIFT_ARM64E;
        let gva = page;
        // 64x192 BGRA8 spans three pages, so there is a middle one to punch out.
        let (w, h) = (64u32, 192u32);

        let whole = gva_backing(&st, &host, 1, gva, w, h).expect("walk resolves");
        assert!(whole.gpas.iter().all(|&g| g != 0), "fixture starts complete");
        store_gva_owned(
            &mut st,
            gva,
            w,
            h,
            vec![0xA5u8; (w * h * 4) as usize],
            0,
            Some(whole),
        );
        arm_gva_guest_write_witness(&mut st, &mut host, gva);
        // Assert the entry's token, not the host's live-set count: a superseded
        // set stays live until `flush_retired_views` drains it, so the count
        // cannot distinguish "armed again" from "the old one is still around".
        assert_ne!(
            st.host_gva_surfaces.get(&gva).map(|e| e.guest_write_token),
            Some(0),
            "a complete page list is the case that must still arm"
        );

        // Punch the middle page out and re-store: the list keeps its length and
        // carries a hole where the page was.
        repoint_pte(&mut host, root_gpa, 2, 0);
        let holed = gva_backing(&st, &host, 1, gva, w, h).expect("walk still names the span");
        assert_eq!(holed.gpas[1], 0, "the hole sits where the page is");
        store_gva_owned(
            &mut st,
            gva,
            w,
            h,
            vec![0x5Au8; (w * h * 4) as usize],
            0,
            Some(holed),
        );
        arm_gva_guest_write_witness(&mut st, &mut host, gva);
        assert_eq!(
            st.host_gva_surfaces.get(&gva).map(|e| e.guest_write_token),
            Some(0),
            "the holed list must leave the entry unarmed, not watch guest page 0"
        );
        assert!(
            gva_guest_wrote_since_store(&st, &host, gva),
            "an unarmed entry reads as written"
        );
    }

    /// Dropping an entry drops its host-side tracking set with it.
    ///
    /// `evict_gva` runs on every deferred GVA render Store arm and the reset
    /// path walks the whole map, so a token left behind on either would leak
    /// one hypervisor tracking set per cached texture VA per device lifetime.
    #[test]
    fn evict_and_reset_release_gva_write_tokens() {
        let mut host = FakeHost::new();
        let mut st = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        setup_depth1_task(&mut host, &mut st);
        let page = 1u64 << PAGE_SHIFT_ARM64E;
        store_armed(&mut st, &mut host, page, 0xA5);
        store_armed(&mut st, &mut host, 2 * page, 0xA5);
        assert_eq!(host.tracked_guest_write_sets(), 2);

        evict_gva(&mut st, page);
        crate::runtime::mapper::flush_retired_views(&mut st, &mut host);
        assert_eq!(host.tracked_guest_write_sets(), 1, "the evicted entry's set");

        let _ = st.take_all_host_views();
        crate::runtime::mapper::flush_retired_views(&mut st, &mut host);
        assert_eq!(
            host.tracked_guest_write_sets(),
            0,
            "a device reset releases the GVA cache's tokens too"
        );
    }
}

#[cfg(test)]
mod incarnation_tests {
    use super::*;
    use crate::model::{DeviceId, PAGE_SHIFT_X86};

    /// The surface_id cache must be able to tell one incarnation of a mapping
    /// from the next, because nothing else in the invalidation path can.
    ///
    /// `invalidate_mapping_pages` is the device's whole response to discovering
    /// that a page list no longer names its surface: it clears `page_entries`,
    /// bumps `map_generation`, and retires the contiguous view and the
    /// guest-write token. It does not touch this cache, and this cache is keyed
    /// by mapping id alone — so the frame read out of the *old* page list stays
    /// on offer to the present capture, the sampled path and the type-11 Load
    /// seed, for as long as the geometry matches. After a re-point the geometry
    /// usually does match: it is the same window at the same size on new
    /// backing.
    ///
    /// That is the shape of the two symptoms this is being read for — a Finder
    /// icon that renders correctly for a few frames and then corrupts, and a
    /// Safari patch that is blank at one scroll position while the rest of the
    /// page is fine. Both are "correct until something serves the previous
    /// incarnation".
    ///
    /// Behaviour is unchanged at this commit and this test says so on purpose.
    /// The rate is unmeasured, and the failure direction of refusing is worse
    /// than the failure direction of serving: a withheld Load seed renders onto
    /// a cleared target, which is a compositing layer going solid black. The
    /// identity is captured and counted first; `REIMS_VGPU_SURFACE_CACHE_GEN_STRICT`
    /// is what acts on it.
    #[test]
    fn a_cached_surface_frame_knows_which_incarnation_it_came_from() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mid = 7u32;
        let (w, h) = (8u32, 4u32);
        state.map_surface(mid);
        {
            let m = state.mappings.get_mut(&mid).unwrap();
            m.mapped = true;
            m.page_entries = vec![1];
        }
        let generation_at_store = state.mappings[&mid].map_generation;

        store(&mut state, mid, w, h, vec![0xa5u8; (w * h * 4) as usize]);
        assert_eq!(
            state.host_surfaces[&mid].map_generation_at_store, generation_at_store,
            "the entry records the incarnation its bytes were read out of"
        );
        assert!(
            surface_entry_is_current(&state, mid),
            "nothing has moved, so the frame is current"
        );
        assert!(get(&state, mid, w, h).is_some());

        // The device discovers the guest re-pointed this surface and does what
        // it does about it. The cache is not in that path.
        assert!(state.invalidate_mapping_pages(mid));
        assert_ne!(
            state.mappings[&mid].map_generation, generation_at_store,
            "invalidation bumps the incarnation"
        );
        assert_eq!(
            state.host_surfaces[&mid].map_generation_at_store, generation_at_store,
            "and leaves the cached frame behind, still keyed by mapping id alone"
        );
        assert!(
            !surface_entry_is_current(&state, mid),
            "so the frame on offer is the previous incarnation's, and this is \
             the only thing that can say so"
        );

        // Unchanged by default: counted, not yet refused.
        assert!(
            get(&state, mid, w, h).is_some(),
            "REIMS_VGPU_SURFACE_CACHE_GEN_STRICT is what turns the refusal on"
        );
    }

    /// A namespace with no mapping behind it must not be collateral damage.
    ///
    /// `store_texture` keys on a texture object ref and `store_gva` on a guest
    /// virtual address; neither is a mapping id, so neither has an incarnation
    /// to compare. Both sides of that comparison are 0, which is why "no
    /// mapping" had to mean a specific value rather than an `Option` the
    /// readers would each have to interpret.
    #[test]
    fn a_cache_namespace_that_is_not_a_mapping_id_is_never_stale() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let (w, h) = (8u32, 4u32);
        let texture_ref = 12u32;
        store_texture(&mut state, texture_ref, w, h, vec![0x3cu8; (w * h * 4) as usize]);
        assert_eq!(
            state.host_texture_surfaces[&texture_ref].map_generation_at_store, 0,
            "a texture ref names no mapping incarnation"
        );
        assert!(get_texture(&state, texture_ref, w, h).is_some());

        // And a surface_id entry stored while this crate holds no mapping under
        // that id — every bare-cache unit test in this crate — stays readable.
        let orphan = 99u32;
        store(&mut state, orphan, w, h, vec![0x11u8; (w * h * 4) as usize]);
        assert!(!state.mappings.contains_key(&orphan));
        assert!(surface_entry_is_current(&state, orphan));
        assert!(get(&state, orphan, w, h).is_some());
    }
}

#[cfg(test)]
mod arming_window_tests {
    use super::*;
    use crate::model::{DeviceId, PAGE_SHIFT_X86};
    use crate::runtime::host::{FakeHost, HostOps};

    /// The GVA cache's guest-write witness must recover once the host's arming
    /// window closes, or it can never report a clean entry at all.
    ///
    /// It could not. `reims_vgpu_dirty_gen` holds a tracked set's generation at
    /// 0 for two harvests — an absence of write reports says nothing about the
    /// guest until logging has been on for a full interval — and
    /// `arm_gva_guest_write_witness` runs immediately after
    /// `track_guest_writes`, so it always read 0. It then returned early on
    /// every later store because a token already existed, so the useless
    /// baseline was never revisited. A real generation is never 0, so the
    /// comparison in `gva_guest_wrote_since_store` could not match one:
    /// `gvac_gw_clean` was **0 of 201 331 lookups** on a 300 s boot, and a
    /// previous session read that zero as a fact about the guest — "every entry
    /// this cache still held had been rewritten" — when most of the population
    /// had simply never had a baseline.
    ///
    /// Nothing caught it because `FakeHost` armed its sets instantly. A test
    /// double more generous than the host it stands for cannot fail the way
    /// production does, which is why `guest_write_startup_window` exists and why
    /// this test turns it on.
    #[test]
    fn the_gva_witness_recovers_when_the_hosts_arming_window_closes() {
        let page = 1u64 << PAGE_SHIFT_X86;
        let gpa = 0x4000_0000u64;
        let mut host = FakeHost::new();
        host.guest_write_startup_window = true;
        host.map_range(gpa, page as usize, 0);

        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let gva = 0x8000u64;
        let (w, h) = (4u32, 2u32);
        let bytes = || vec![0x7fu8; (w * h * 4) as usize];
        let backing = GvaBacking {
            gpas: vec![gpa],
            span: page,
            task_id: 0,
        };

        store_gva_owned(&mut state, gva, w, h, bytes(), 0, Some(backing.clone()));
        arm_gva_guest_write_witness(&mut state, &mut host, gva);
        let token = state.host_gva_surfaces[&gva].guest_write_token;
        assert_ne!(token, 0, "the pages are trackable, so a token is taken");
        assert_eq!(
            state.host_gva_surfaces[&gva].guest_write_gen_at_store, 0,
            "inside the window the host has no generation to give"
        );
        assert!(
            gva_guest_wrote_since_store(&state, &host, gva),
            "and with no baseline the only safe answer is that the copy is stale"
        );

        // The window closes. Nothing about the guest changed.
        host.finish_guest_write_arming();
        assert_eq!(host.guest_write_gen(token), Some(1));

        // The next store re-latches against the now-readable generation. This is
        // the step the early return used to skip.
        store_gva_owned(&mut state, gva, w, h, bytes(), 0, Some(backing));
        arm_gva_guest_write_witness(&mut state, &mut host, gva);
        assert_eq!(
            state.host_gva_surfaces[&gva].guest_write_token, token,
            "the token is kept — re-registering would cost a host call per store"
        );
        assert_eq!(
            state.host_gva_surfaces[&gva].guest_write_gen_at_store, 1,
            "but the baseline is re-read, because it must name the bytes stored"
        );
        assert!(
            !gva_guest_wrote_since_store(&state, &host, gva),
            "so an unwritten entry can finally be reported clean, which is the \
             whole point of the rail"
        );

        // And it still reports a real write, so the recovery did not buy `clean`
        // by making the witness blind.
        host.guest_wrote_page(gpa);
        assert!(
            gva_guest_wrote_since_store(&state, &host, gva),
            "a witness that only ever says clean is worse than one that never does"
        );
    }
}
