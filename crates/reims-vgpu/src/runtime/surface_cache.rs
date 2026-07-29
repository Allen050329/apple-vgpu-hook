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
use crate::model::{DeviceState, HostSurface, MAX_SCANOUT_DIM};

fn store_into(
    map: &mut std::collections::BTreeMap<u32, HostSurface>,
    id: u32,
    width: u32,
    height: u32,
    bgra: std::sync::Arc<Vec<u8>>,
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
    entry.host_gen = entry.host_gen.wrapping_add(1);
    if entry.host_gen == 0 {
        entry.host_gen = 1;
    }
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
) -> Option<(&[u8], u32)> {
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
    store_into(&mut state.host_surfaces, surface_id, width, height, bgra);
}

/// Borrow host-cache frame when geom matches request (surface_id namespace).
pub fn get(state: &DeviceState, surface_id: u32, width: u32, height: u32) -> Option<&[u8]> {
    get_from(&state.host_surfaces, surface_id, width, height)
}

/// [`get`] as a shared handle, for a caller that needs to own the frame past the
/// borrow of `state` — a Load seed does, and taking it this way costs a refcount
/// rather than a full-framebuffer copy.
///
/// A handle cannot be truncated the way [`get`]'s slice is, so this additionally
/// requires the stored buffer to be *exactly* `width * height * 4`. A store with
/// slop past that is served by [`get`] and misses here, which is the safe way
/// round: the engine rejects a seed whose length is not exactly the geometry
/// (`TargetSeedLength`), so handing one out would turn a working draw into a
/// declined one.
pub fn get_shared(
    state: &DeviceState,
    surface_id: u32,
    width: u32,
    height: u32,
) -> Option<std::sync::Arc<Vec<u8>>> {
    let need = get_from(&state.host_surfaces, surface_id, width, height)?.len();
    let e = state.host_surfaces.get(&surface_id)?;
    (e.bgra.len() == need).then(|| std::sync::Arc::clone(&e.bgra))
}

/// Type-2/3 encode cache by texture object ref (not surface_id).
pub fn store_texture(
    state: &mut DeviceState,
    texture_ref: u32,
    width: u32,
    height: u32,
    bgra: Vec<u8>,
) {
    store_into(
        &mut state.host_texture_surfaces,
        texture_ref,
        width,
        height,
        std::sync::Arc::new(bgra),
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
pub fn mirror_linear_color_cache(
    state: &mut DeviceState,
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
    store_gva(state, gva, width, height, bgra);
}

/// Type-2/3 encode cache by target GVA.
///
/// On discrete hosts this is the **GPU-private** texture content for that VA.
/// Guest MapMemory2 unmap/remap changes PFNs under the same GVA but does **not**
/// destroy the encode — see [`note_unmap_retain_gva`] (Unmap retains; Map notify-only).
pub fn store_gva(state: &mut DeviceState, gva: u64, width: u32, height: u32, bgra: Vec<u8>) {
    store_gva_owned(state, gva, width, height, bgra, 0);
}

/// Store a GVA encode with the decoded object identity that produced it.
/// Type-2/type-3 wrappers are the same linear texture storage family when the
/// GVA and geometry match; unrelated nonzero object-type transitions still
/// identify a different resource class.
pub fn store_gva_owned(
    state: &mut DeviceState,
    gva: u64,
    width: u32,
    height: u32,
    bgra: Vec<u8>,
    object_type: u8,
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
    let entry = state.host_gva_surfaces.entry(gva).or_default();
    entry.host_gen = entry.host_gen.wrapping_add(1);
    if entry.host_gen == 0 {
        entry.host_gen = 1;
    }
    entry.width = width;
    entry.height = height;
    entry.bgra = std::sync::Arc::new(bgra);
    entry.producer_object_type = object_type;
}

pub fn get_gva(state: &DeviceState, gva: u64, width: u32, height: u32) -> Option<&[u8]> {
    get_gva_with_gen(state, gva, width, height).map(|(bgra, _)| bgra)
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
) -> Option<(&[u8], u32)> {
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
) -> Option<(&[u8], u32, u8)> {
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
    state.host_gva_surfaces.remove(&gva);
}

/// Drop host-cache entry (unmap / delete surface).
pub fn evict(state: &mut DeviceState, surface_id: u32) {
    state.host_surfaces.remove(&surface_id);
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::model::{DeviceId, PAGE_SHIFT_ARM64E};

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
        st.insert_object(6, 21, crate::model::ObjectEntry::default());
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
        st.insert_object(6, 22, crate::model::ObjectEntry::default());
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
        store_gva_owned(&mut st, gva, 2, 2, owned, 2);
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
}
