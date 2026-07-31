//! IOSurface mapper capture + page-table / geometry resolve.
//!
//! Capture runs on the iosfc producer MMIO path (guest x19/x21/x22 still hold
//! the directed handoff from `do_host_mapping_gated`). Resolve builds
//! `MappingEntry.page_entries` and geometry from MappingInternal + device
//! descriptor via guest KVA reads ([`HostOps::read_kva`]).

use crate::contract::iosurface_pages::{
    self, build_table_plan, decode_device_surface, decode_mapper_request_entry, guest_kernel_va,
    mapper_request_published_entry_offset, read_internal_desc_ptr, read_mapper_identity,
    read_mapper_internal, sample_window_prefer_device, validate_mapper_internal, PagesMemory,
    DEVICE_DESC_LEN, MAPPER_CAPTURE_REG_MAPPER_DEVICE, MAPPER_CAPTURE_REG_MAPPING_INTERNAL,
    MAPPER_CAPTURE_REG_REQUEST_TYPE, MAPPER_REQUEST_ENTRY_LEN, MAPPER_REQUEST_MAP,
    MAPPER_REQUEST_UNMAP,
};
use crate::model::{DeviceState, MapperCapture, MAX_MAPPINGS};
use crate::runtime::host::{HostMemory, HostOps, MemError};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MapperDecline {
    CaptureMapperXregRead(MemError),
    CaptureRequestTypeXregRead(MemError),
    CaptureInternalXregRead(MemError),
    CaptureRequestTypeMismatch,
    CaptureInternalZero,
    CaptureInternalKvaInvalid,
    CaptureMapperKvaInvalid,
    DeviceDescriptorRead(MemError),
}

impl crate::observe::Decline for MapperDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::CaptureMapperXregRead(_) => "mapper_capture_mapper_xreg_read",
            Self::CaptureRequestTypeXregRead(_) => "mapper_capture_request_type_xreg_read",
            Self::CaptureInternalXregRead(_) => "mapper_capture_internal_xreg_read",
            Self::CaptureRequestTypeMismatch => "mapper_capture_request_type_mismatch",
            Self::CaptureInternalZero => "mapper_capture_internal_zero",
            Self::CaptureInternalKvaInvalid => "mapper_capture_internal_kva_invalid",
            Self::CaptureMapperKvaInvalid => "mapper_capture_mapper_kva_invalid",
            Self::DeviceDescriptorRead(_) => "mapper_device_descriptor_read",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::CaptureMapperXregRead(error)
            | Self::CaptureRequestTypeXregRead(error)
            | Self::CaptureInternalXregRead(error)
            | Self::DeviceDescriptorRead(error) => vec![(
                "host_reason",
                crate::observe::Decline::slug(error).to_string(),
            )],
            _ => Vec::new(),
        }
    }
}

fn refusal_reason(status: &iosurface_pages::Status) -> &'static str {
    crate::observe::Refusal::refusal(status)
        .expect("an IOSurface contract error must carry a refusal reason")
}

/// Fail-visible, **de-duplicated per `(mapping_id, reason)`**, for the
/// `resolve_mapping_backing` blind spot: a mapped surface whose page-table /
/// geometry resolve fails leaves the mapping silently un-resolved, and every
/// downstream present/Store/sample paints or writes back **black** for it with
/// no log naming why. `resolve_mapping_backing` runs on the per-present `force`
/// path (drain.rs), so a bare `observe::fail` at a failing site would flood.
/// This latch logs each `(mapping_id, reason)` **once** and is cleared for a
/// mapping the moment it resolves ([`clear_resolve_fail`]), so a genuinely
/// broken mapping logs one line, a flapping one re-logs per transition, and a
/// healthy boot fires nothing. Runs on the drain worker (off the QEMU main
/// core). Speculative/not-ready returns (unmapped, dims-not-yet-landed) are
/// **not** routed here — only genuine anomalies for an already-mapped surface.
fn resolve_fail_latch() -> &'static std::sync::Mutex<std::collections::HashSet<(u32, &'static str)>>
{
    use std::collections::HashSet;
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<HashSet<(u32, &'static str)>>> = OnceLock::new();
    SEEN.get_or_init(|| Mutex::new(HashSet::new()))
}

fn note_resolve_fail(mapping_id: u32, reason: &'static str, detail: String) {
    let mut guard = resolve_fail_latch()
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    if guard.insert((mapping_id, reason)) {
        crate::observe::fail(detail);
    }
}

fn note_resolve_keep_cached(mapping_id: u32, reason: &'static str, detail: String) {
    let mut guard = resolve_fail_latch()
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    if guard.insert((mapping_id, reason)) {
        crate::observe::off(detail);
    }
}

/// Fail-visible capture miss, sharing the same per-`(mapping_id, reason)` latch
/// as [`note_resolve_fail`]. `capture_at_producer` runs on the publishing vCPU
/// (iosfc producer MMIO write), not the drain worker, but it fires **once per
/// mapper-ring publish** — a rare setup/map event, never per-frame — so a
/// latched genuine-only line here costs nothing on the hot path. A capture miss
/// for an already-decoded MAP/UNMAP request means the mapping's `MappingInternal`
/// never attaches, and every downstream present/Store for it paints **black**
/// with no reason. Speculative returns (producer==0, ring not ready, a non
/// MAP/UNMAP request type) are **not** routed here. Sharing the latch means a
/// mapping that later resolves cleanly re-arms its capture reasons too
/// ([`clear_resolve_fail`] clears all reasons for the id).
fn note_capture_fail(mapping_id: u32, reason: &'static str, detail: String) {
    note_resolve_fail(mapping_id, reason, detail);
}

/// Re-arm every reason latch for a mapping that just resolved, so a later
/// genuine failure on the same mapping is logged again (catches flapping).
fn clear_resolve_fail(mapping_id: u32) {
    let mut guard = resolve_fail_latch()
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    guard.retain(|(mid, _)| *mid != mapping_id);
}

fn capture_xreg_failed(
    mapping_id: u32,
    producer: u32,
    decline: MapperDecline,
) -> Option<MapperCapture> {
    note_capture_fail(
        mapping_id,
        crate::observe::Decline::slug(&decline),
        crate::observe::Emit::decline("mapper_capture_fail", &decline)
            .field("mapping", mapping_id)
            .field("producer", producer)
            .render(),
    );
    None
}

/// Adapter: mapper internals are KVA; page content GPAs use HostMemory.
struct MapperMem<'a, H: HostMemory + HostOps> {
    host: &'a H,
    last_error: std::cell::Cell<Option<MemError>>,
}

impl<'a, H: HostMemory + HostOps> MapperMem<'a, H> {
    fn new(host: &'a H) -> Self {
        Self {
            host,
            last_error: std::cell::Cell::new(None),
        }
    }

    fn last_error(&self) -> Option<MemError> {
        self.last_error.get()
    }
}

impl<H: HostMemory + HostOps> PagesMemory for MapperMem<'_, H> {
    fn read(&self, address: u64, dst: &mut [u8]) -> bool {
        if guest_kernel_va(address) {
            match self.host.read_kva(address, dst) {
                Ok(()) => true,
                Err(e) => {
                    self.last_error.set(Some(e));
                    false
                }
            }
        } else {
            match self.host.read_gpa(address, dst) {
                Ok(()) => true,
                Err(e) => {
                    self.last_error.set(Some(e));
                    false
                }
            }
        }
    }
    fn is_kernel_va(&self, address: u64) -> bool {
        guest_kernel_va(address)
    }
    fn is_ram_gpa(&self, address: u64) -> bool {
        self.host.is_ram_gpa(address)
    }
}

/// Capture mapper handoff registers while still on the publishing vCPU.
///
/// Call from the iosfc producer MMIO write path before scheduling the drain BH.
pub fn capture_at_producer<H: HostMemory + HostOps>(
    state: &DeviceState,
    host: &H,
    producer: u32,
) -> Option<MapperCapture> {
    if producer == 0 || state.iosfc.ring_base == 0 {
        return None;
    }
    let entry_off = mapper_request_published_entry_offset(producer)?;
    let mut e = [0u8; MAPPER_REQUEST_ENTRY_LEN];
    host.read_gpa(state.iosfc.ring_base + entry_off, &mut e)
        .ok()?;
    let request = decode_mapper_request_entry(&e).ok()?;
    if request.request_type != MAPPER_REQUEST_MAP && request.request_type != MAPPER_REQUEST_UNMAP {
        return None;
    }
    if request.mapping_id as usize >= MAX_MAPPINGS {
        return None;
    }

    // From here the ring entry is a decoded MAP/UNMAP for a valid mapping_id, so
    // any failure below is a genuine capture miss (the handoff registers do not
    // corroborate the request), not the speculative not-ready poll — log it once
    // per (mapping_id, reason). The mapping's MappingInternal never attaches and
    // downstream present/Store paints black otherwise.
    let mid = request.mapping_id;
    let mapper = match host.read_xreg(MAPPER_CAPTURE_REG_MAPPER_DEVICE) {
        Ok(value) => value,
        Err(error) => {
            let decline = MapperDecline::CaptureMapperXregRead(error);
            return capture_xreg_failed(mid, producer, decline);
        }
    };
    let rtype = match host.read_xreg(MAPPER_CAPTURE_REG_REQUEST_TYPE) {
        Ok(value) => value as u32,
        Err(error) => {
            let decline = MapperDecline::CaptureRequestTypeXregRead(error);
            return capture_xreg_failed(mid, producer, decline);
        }
    };
    let internal = match host.read_xreg(MAPPER_CAPTURE_REG_MAPPING_INTERNAL) {
        Ok(value) => value,
        Err(error) => {
            let decline = MapperDecline::CaptureInternalXregRead(error);
            return capture_xreg_failed(mid, producer, decline);
        }
    };
    if rtype != request.request_type {
        let decline = MapperDecline::CaptureRequestTypeMismatch;
        note_capture_fail(
            mid,
            crate::observe::Decline::slug(&decline),
            crate::observe::Emit::decline("mapper_capture_fail", &decline)
                .field("mapping", mid)
                .field("rtype", rtype)
                .field("request_type", request.request_type)
                .render(),
        );
        return None;
    }
    if internal == 0 {
        let decline = MapperDecline::CaptureInternalZero;
        note_capture_fail(
            mid,
            crate::observe::Decline::slug(&decline),
            crate::observe::Emit::decline("mapper_capture_fail", &decline)
                .field("mapping", mid)
                .render(),
        );
        return None;
    }
    if !guest_kernel_va(internal) {
        let decline = MapperDecline::CaptureInternalKvaInvalid;
        note_capture_fail(
            mid,
            crate::observe::Decline::slug(&decline),
            crate::observe::Emit::decline("mapper_capture_fail", &decline)
                .field("mapping", mid)
                .field("internal", format!("{internal:#x}"))
                .render(),
        );
        return None;
    }
    if mapper != 0 && !guest_kernel_va(mapper) {
        let decline = MapperDecline::CaptureMapperKvaInvalid;
        note_capture_fail(
            mid,
            crate::observe::Decline::slug(&decline),
            crate::observe::Emit::decline("mapper_capture_fail", &decline)
                .field("mapping", mid)
                .field("mapper_kva", format!("{mapper:#x}"))
                .render(),
        );
        return None;
    }

    let mem = MapperMem::new(host);
    let fields = match read_mapper_identity(&mem, internal, mapper != 0, mapper) {
        Ok(f) => f,
        Err(status) => {
            let reason = refusal_reason(&status);
            note_capture_fail(
                mid,
                reason,
                crate::observe::Emit::refusal("mapper_capture_fail", &status)
                    .expect("the error arm cannot carry Status::Ok")
                    .field("mapping", mid)
                    .field("internal", format!("{internal:#x}"))
                    .field("mapper_kva", format!("{mapper:#x}"))
                    .render(),
            );
            return None;
        }
    };
    let status = validate_mapper_internal(&mem, mid, &fields);
    if status != iosurface_pages::Status::Ok {
        let reason = refusal_reason(&status);
        note_capture_fail(
            mid,
            reason,
            crate::observe::Emit::refusal("mapper_capture_fail", &status)
                .expect("the non-Ok branch must carry a refusal")
                .field("mapping", mid)
                .field("internal", format!("{internal:#x}"))
                .render(),
        );
        return None;
    }

    Some(MapperCapture {
        producer,
        mapper_device_kva: mapper,
        request_type: rtype,
        mapping_internal: internal,
    })
}

/// Apply a capture to the mapping named by the just-drained ring entry.
pub fn apply_capture(state: &mut DeviceState, cap: &MapperCapture, mapping_id: u32) -> bool {
    // Neither branch below releases a deferred writeback window. A type-11
    // render Store writes guest pages on its own path, so an UNMAP (or a MAP
    // that re-backs the slot with a different MappingInternal, orphaning the
    // old identity) leaves no mapping-keyed render obligation behind. Compute
    // storage windows are the surviving mapping-keyed rail and they are torn
    // down by `storage_flush::drop_windows` from the lifecycle sites that know
    // whether the pages can still be written.
    if cap.request_type == MAPPER_REQUEST_UNMAP {
        return state.unmap_surface(mapping_id);
    }
    if cap.request_type != MAPPER_REQUEST_MAP {
        return false;
    }
    if cap.mapper_device_kva != 0 {
        state.mapper_device_kva = cap.mapper_device_kva;
    }
    state.attach_mapping_internal(mapping_id, cap.mapping_internal)
}

/// Resolve page table + device-descriptor geometry for a mapped slot.
///
/// Safe to call repeatedly; refreshes pages when `mapping_internal` is set.
pub fn resolve_mapping_backing<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &H,
    mapping_id: u32,
) -> bool {
    let Some(m) = state.mappings.get(&mapping_id) else {
        return false;
    };
    if !m.mapped || m.mapping_internal == 0 {
        return false;
    }
    let internal = m.mapping_internal;
    let mapper = state.mapper_device_kva;
    let cached_pages = m.page_entries.len();
    let cached_table = m.page_table_kva;
    let had_cached_pages = cached_pages != 0;
    let mem = MapperMem::new(host);

    let fields = match read_mapper_internal(&mem, internal, mapper != 0, mapper) {
        Ok(f) => f,
        Err(status) => {
            let reason = refusal_reason(&status);
            let host_error = mem.last_error();
            let host_reason = host_error
                .map(|error| crate::observe::Decline::slug(&error))
                .unwrap_or("none");
            if had_cached_pages {
                // QEMU can stop exposing a CPU-backed KVA alias after the
                // mapper handoff while the already-validated GPA page plan
                // remains live. Likewise, a transiently unmapped debug-read
                // alias says nothing about the cached guest-physical pages.
                // Both are the normal revalidation fallback, not a decoded
                // guest-command refusal; emitting them once per recycled
                // mapping id made a healthy boot fire dozens of new Phase-5
                // lines. Keep other cached-plan failures visible because they
                // describe malformed identity fields rather than alias
                // availability.
                if matches!(host_error, Some(MemError::NoCpu | MemError::Unmapped)) {
                    return true;
                }
                note_resolve_keep_cached(
                    mapping_id,
                    reason,
                    crate::observe::Emit::refusal("mapper_revalidate_fallback", &status)
                        .expect("the error arm cannot carry Status::Ok")
                        .field("mapping", mapping_id)
                        .field("pages", cached_pages)
                        .field("table", format!("{cached_table:#x}"))
                        .field("internal", format!("{internal:#x}"))
                        .field("mapper_kva", format!("{mapper:#x}"))
                        .field("host_reason", host_reason)
                        .render(),
                );
                return true;
            }
            // A mapped surface (m.mapped, mapping_internal != 0) whose mapper
            // internal KVA is unreadable is a genuine anomaly, not the
            // not-yet-mapped poll — every downstream present/Store for this
            // mapping then paints black with no reason.
            note_resolve_fail(
                mapping_id,
                reason,
                crate::observe::Emit::refusal("mapper_resolve_fail", &status)
                    .expect("the error arm cannot carry Status::Ok")
                    .field("mapping", mapping_id)
                    .field("internal", format!("{internal:#x}"))
                    .field("mapper_kva", format!("{mapper:#x}"))
                    .field("host_reason", host_reason)
                    .render(),
            );
            return false;
        }
    };
    let status = validate_mapper_internal(&mem, mapping_id, &fields);
    if status != iosurface_pages::Status::Ok {
        let reason = refusal_reason(&status);
        note_resolve_fail(
            mapping_id,
            reason,
            crate::observe::Emit::refusal("mapper_resolve_fail", &status)
                .expect("the non-Ok branch must carry a refusal")
                .field("mapping", mapping_id)
                .field("internal", format!("{internal:#x}"))
                .render(),
        );
        return false;
    }

    // Geometry from device descriptor when present; cache full 0x200 for
    // biplanar plane selection (sample_window_prefer_device).
    let mut width = 0u32;
    let mut height = 0u32;
    let mut format = 0u16;
    // Guest page size for *this* device — never a bare arm PAGE_SIZE constant.
    let guest_page = state.page_size();
    let mut min_size = guest_page;
    let mut device_desc: Option<Vec<u8>> = None;
    match read_internal_desc_ptr(&mem, internal) {
        Ok(desc_kva) => {
            let mut desc = [0u8; DEVICE_DESC_LEN];
            if !mem.read(desc_kva, &mut desc) {
                let decline = MapperDecline::DeviceDescriptorRead(
                    mem.last_error().unwrap_or(MemError::Unmapped),
                );
                note_resolve_fail(
                    mapping_id,
                    crate::observe::Decline::slug(&decline),
                    crate::observe::Emit::decline("mapper_device_descriptor_fallback", &decline)
                        .field("mapping", mapping_id)
                        .field("internal", format!("{internal:#x}"))
                        .field("descriptor", format!("{desc_kva:#x}"))
                        .render(),
                );
            } else {
                device_desc = Some(desc.to_vec());
                if let Some(surf) = decode_device_surface(&desc) {
                    if surf.alloc_size as u64 > 0 {
                        min_size = (surf.alloc_size as u64).max(guest_page);
                    }
                    if surf.width > 0 && surf.height > 0 {
                        width = surf.width;
                        height = surf.height;
                        format = (surf.pixel_format & 0xffff) as u16;
                        if let Some((_, _, end, _)) =
                            sample_window_prefer_device(Some(&desc), None, format, width, height)
                        {
                            min_size = min_size.max(end).max(guest_page);
                        }
                    }
                }
            }
        }
        Err(status) => {
            let reason = refusal_reason(&status);
            // A zero descriptor pointer is the documented "not present" state:
            // geometry can come from the texture object. A failed read or a
            // nonzero invalid pointer is a real fallback decision.
            if reason != "iosurface_mapper_device_desc_pointer_zero" {
                note_resolve_fail(
                    mapping_id,
                    reason,
                    crate::observe::Emit::refusal("mapper_device_descriptor_fallback", &status)
                        .expect("the error arm cannot carry Status::Ok")
                        .field("mapping", mapping_id)
                        .field("internal", format!("{internal:#x}"))
                        .render(),
                );
            }
        }
    }

    // Texture-path geom (type-11 object dims) refines span for single-plane; for
    // multi-plane, prefer alloc_size already latched from the device descriptor.
    if let Some(m) = state.mappings.get(&mapping_id) {
        if m.has_geom && m.width > 0 && m.height > 0 {
            width = m.width;
            height = m.height;
            format = if m.format != 0 { m.format } else { format };
            let desc_slice = device_desc.as_deref().or({
                if m.device_desc.len() >= DEVICE_DESC_LEN {
                    Some(m.device_desc.as_slice())
                } else {
                    None
                }
            });
            if let Some((_, _, end, _)) =
                sample_window_prefer_device(desc_slice, None, format, width, height)
            {
                min_size = min_size.max(end).max(guest_page);
            }
        }
    }

    let plan = match build_table_plan(&mem, mapping_id, &fields, min_size, state.page_shift) {
        Ok(p) => p,
        Err(status) => {
            // Still latch geom / device desc if we decoded them, even without pages yet.
            if let Some(ref d) = device_desc {
                let _ = state.set_mapping_device_desc(mapping_id, d);
            }
            if width > 0 && height > 0 {
                let _ = state.set_mapping_geom(mapping_id, width, height, format);
                // Geometry IS known, yet no page table covers its
                // `min_size` span — the short-page-table → black-tile class
                // (fail-closed Store writeback / sample walk while the geom is
                // set). Distinct from the dims-not-yet-landed poll (width==0),
                // which stays silent as legitimate not-ready control flow.
                let reason = refusal_reason(&status);
                note_resolve_fail(
                    mapping_id,
                    reason,
                    crate::observe::Emit::refusal("mapper_resolve_fail", &status)
                        .expect("the error arm cannot carry Status::Ok")
                        .field("mapping", mapping_id)
                        .field("width", width)
                        .field("height", height)
                        .field("format", format!("{format:#x}"))
                        .field("min_size", min_size)
                        .render(),
                );
            }
            return false;
        }
    };

    // Read before the `get_mut` below takes `state` mutably.
    let page_shift = state.page_shift;
    let mut retired = None;
    let mut incarnation_changed = false;
    let mut reprieved = false;
    let mut pages_changed = false;
    if let Some(m) = state.mappings.get_mut(&mapping_id) {
        // A condemned slot (trailing DeleteIOSurfaceBacking2, no resolve
        // since) compares against the stashed fingerprint: the same plan is
        // the SAME incarnation — the delete was stale, keep the generation so
        // the resident and deferred windows stay live (black-band class). A
        // different plan is a genuine new incarnation.
        let condemned = m.condemned_entries.take();
        let prev_pages = m.page_entries.len();
        (pages_changed, incarnation_changed, reprieved) =
            plan_adoption_decision(condemned.as_deref(), &m.page_entries, &plan.entries);
        // New page table ⇒ the contiguous view (and any Metal texture aliasing
        // it) describe the old pages; retire them before adopting the plan.
        if m.contig_ptr != 0 && pages_changed {
            retired = Some((m.contig_ptr, m.contig_len));
            m.contig_ptr = 0;
            m.contig_len = 0;
        }
        if pages_changed {
            DeviceState::bump_map_generation(m);
        }
        // The guest-physical footprint this incarnation authorises us to write.
        //
        // A guest kernel panic names a *physical page* (`pmap_page_protect()
        // ... pn=0x46b53b`), and nothing this device emitted could be compared
        // against it — so "did we write there?" was unanswerable, and the
        // random-victim panic class in AGENTS.md stayed a signature with no way
        // to confirm or clear this device. Every mapping-rail write is bounded
        // to the page list adopted here, so the union of these spans over a boot
        // is exactly the set of pages those writes can reach. A `pn` inside it
        // is evidence; a `pn` outside every one of them exonerates the rail.
        //
        // One line per surface incarnation, not per write: the key is
        // (mapping, generation) and the generation only moves when the PFNs do,
        // so this is bounded by how often the guest rewires a surface and is
        // safe to leave on. min/max over the entries is O(pages) once per
        // incarnation, against an O(pages) table build that just ran.
        // Keyed on the ADOPTION, not on `pages_changed`. Two earlier cuts of
        // this line were silent for entire boots, and both were the
        // instrument-the-branch trap from AGENTS.md committed by a change that
        // cites it. The first logged only when a span resolved, so it could not
        // distinguish "no span" from "never ran". The second moved to
        // `pages_changed` on the reasoning that `map_gen` climbing past 100
        // proved that branch ran — but `bump_map_generation` has five other
        // call sites, so the generation was never evidence about this one.
        //
        // `pages_changed` is genuinely false here even on a first population:
        // the reprieve path (`condemned` holds the fingerprint, the plan
        // matches it) repopulates an emptied `page_entries` with
        // `pages_changed == false`. The adoption below is what every write is
        // then bounded by, so that is what has to be reported.
        //
        // Dedup is `first_sight` on the span itself, which keeps it bounded by
        // *distinct footprints* rather than by resolve rate — a mapping
        // re-resolved every frame to the same pages logs once.
        if let Some((lo, hi)) = entry_gpa_span(&plan.entries, page_shift) {
            let key = (u64::from(mapping_id) << 40) ^ (lo >> page_shift) ^ (hi << 20);
            if crate::observe::first_sight("mapping_gpa_span", key) {
                crate::observe::off(format!(
                    "mapping_gpa_span mid={mapping_id} gen={} pages={} prev_pages={prev_pages} \
                     changed={} lo={lo:#x} hi={:#x} pn_lo={:#x} pn_hi={:#x}",
                    m.map_generation,
                    plan.entries.len(),
                    pages_changed as u8,
                    hi + (1u64 << page_shift),
                    lo >> page_shift,
                    hi >> page_shift,
                ));
            }
        }
        m.page_entries = plan.entries;
        m.page_table_kva = plan.page_table_kva;
        m.mapping_internal = internal;
        m.mapped = true;
        if let Some(ref d) = device_desc {
            m.device_desc = d.clone();
        }
    }
    if let Some(v) = retired {
        state.retired_views.push(v);
    }
    if incarnation_changed {
        // The condemned backing really died and the id now carries a new
        // surface: drop the prior incarnation's deferred windows before any
        // access could flush old content through the new pages.
        crate::runtime::storage_flush::drop_windows(state, mapping_id, "incarnation_changed");
    } else if reprieved {
        // Stale trailing delete on a live incarnation — the exact black-band
        // trigger. Only note when content was actually at stake (an armed
        // deferred window survived); plain reprieves are steady id-recycle
        // control flow.
        let windows = state
            .compute_deferred_flush
            .keys()
            .filter(|k| k.mapping_id == mapping_id)
            .count();
        if windows > 0 {
            // Condemn dropped the raw-GVA alias index; the pages just
            // re-adopted are the same ones the windows defer-armed on.
            state.index_deferred_alias_pages(mapping_id);
            crate::observe::off(format!(
                "delete_backing_reprieve mapping={mapping_id} windows={windows}"
            ));
            // Wrong-PFN guard on the REPRIEVE path — the blind spot the rewire
            // guard above cannot cover. A reprieve keeps armed deferred windows
            // WITHOUT bumping map_generation (the delete looked stale: the plan
            // still fingerprints the condemned pages). But a guest that FREED the
            // backing and handed the SAME physical pages to another surface — yet
            // has not yet rewired this mapping's GPU page table away from them —
            // fingerprints identical here, so `pages_changed` is false and the
            // rewire guard never runs. The still-armed flush would then DMA into
            // pages another live surface now owns (or recycled userspace heap =
            // the WindowServer malloc free-list corruption class). A detected
            // cross-surface alias is a proven ownership violation, so fail
            // closed: name it, drop deferred writes, and invalidate the page
            // plan. Runs only on reprieve-with-armed-windows (rare), on the drain
            // worker.
            if let Some((gpa, owner)) = first_surface_page_collision(state, mapping_id) {
                let mine_pages = state
                    .mappings
                    .get(&mapping_id)
                    .map(|m| m.page_entries.len())
                    .unwrap_or(0);
                fail_closed_surface_page_collision(
                    state, mapping_id, gpa, owner, mine_pages, "reprieve",
                );
                return false;
            }
        }
    }
    if width > 0 && height > 0 {
        let _ = state.set_mapping_geom(mapping_id, width, height, format);
    }
    // Wrong-PFN rewire-race guard: a freshly adopted page plan must not alias
    // a *different* live surface's guest pages. Two distinct live IOSurface
    // mappings backing the same physical page means one holds a stale/wrong
    // PFN — a pixel writeback through it scribbles the other surface, or (if
    // the page was recycled to userspace) guest heap: the WindowServer malloc
    // free-list corruption class. A detected alias is a proven ownership
    // violation, so fail closed: name it, drop deferred writes, invalidate the
    // adopted plan, and make this resolve fail. Runs only on a genuine rewire
    // (`pages_changed`), on the drain worker.
    if pages_changed {
        if let Some((gpa, owner)) = first_surface_page_collision(state, mapping_id) {
            let mine_pages = state
                .mappings
                .get(&mapping_id)
                .map(|m| m.page_entries.len())
                .unwrap_or(0);
            fail_closed_surface_page_collision(state, mapping_id, gpa, owner, mine_pages, "rewire");
            return false;
        }
    }
    // Resolved: re-arm the fail latch so a later genuine failure (a re-map that
    // goes bad, a corrupted descriptor) is logged again rather than swallowed.
    clear_resolve_fail(mapping_id);
    true
}

/// Detect the wrong-PFN rewire-race corruption vector: the mapping `mapping_id`
/// just adopted a fresh page plan whose page base is also owned by a
/// *different* currently-live surface mapping. Two distinct live IOSurface
/// mappings must never back the same guest physical page; if they do, one holds
/// a stale/wrong PFN and a writeback through it corrupts memory it does not own
/// (see the WindowServer heap-corruption class). Measure-only — never gates a
/// write. Cost O(this_pages + Σ other live pages); called only on a rewire.
fn first_surface_page_collision(state: &DeviceState, mapping_id: u32) -> Option<(u64, u32)> {
    let page_shift = state.page_shift;
    let page = state.page_size();
    let page_base = |gpa: u64| gpa & !(page - 1);
    let m = state.mappings.get(&mapping_id)?;
    if !m.mapped || m.page_entries.is_empty() {
        return None;
    }
    let mine: std::collections::HashSet<u64> = m
        .page_entries
        .iter()
        .filter_map(|&e| crate::contract::iosurface_pages::entry_gpa_shift(e, page_shift))
        .map(page_base)
        .collect();
    if mine.is_empty() {
        return None;
    }
    for (&other_id, other) in &state.mappings {
        if other_id == mapping_id || !other.mapped || other.page_entries.is_empty() {
            continue;
        }
        for &e in &other.page_entries {
            if let Some(gpa) = crate::contract::iosurface_pages::entry_gpa_shift(e, page_shift) {
                if mine.contains(&page_base(gpa)) {
                    return Some((page_base(gpa), other_id));
                }
            }
        }
    }
    None
}

/// Always-on, deduped-per-`(mid, owner, gpa)` fail line for a cross-surface
/// page alias. Off-main-core (drain worker resolve path). Fires zero on a
/// healthy boot (distinct live surfaces never share a physical page).
fn note_surface_page_collision(
    mapping_id: u32,
    gpa: u64,
    owner: u32,
    mine_pages: usize,
    path: &str,
) {
    use std::collections::HashSet;
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<HashSet<(u32, u32, u64)>>> = OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(HashSet::new()));
    if seen
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert((mapping_id, owner, gpa))
    {
        crate::observe::fail(format!(
            "mapping_pages fail reason=surface_page_collision path={path} mid={mapping_id} \
             owner={owner} gpa={gpa:#x} mine_pages={mine_pages}"
        ));
    }
}

/// A cross-surface page alias is a structural ownership violation: any
/// host-authored pixel write through `mapping_id` can scribble another live
/// surface or, for the recycled-heap variant, a userspace heap page. Keep the
/// always-on forensic line, then fail closed by dropping deferred windows and
/// invalidating the adopted page plan so the next writer must re-resolve instead
/// of writing through known-bad pages.
fn fail_closed_surface_page_collision(
    state: &mut DeviceState,
    mapping_id: u32,
    gpa: u64,
    owner: u32,
    mine_pages: usize,
    path: &str,
) {
    note_surface_page_collision(mapping_id, gpa, owner, mine_pages, path);
    crate::runtime::storage_flush::drop_windows(state, mapping_id, "surface_page_collision");
    let _ = state.invalidate_mapping_pages(mapping_id);
}

/// Incarnation decision when adopting a freshly resolved page plan into a
/// mapping slot. `condemned` is the fingerprint a trailing
/// `DeleteIOSurfaceBacking2` stashed (None when the slot is not condemned).
/// Returns `(pages_changed, incarnation_changed, reprieved)`:
/// `incarnation_changed` = the condemned backing really died and the id now
/// carries different pages (drop the old windows); `reprieved` = the delete
/// was stale — the plan matches the fingerprint, the same incarnation lives
/// on (keep generation, resident, deferred windows).
/// Lowest and highest page-aligned GPA a page-entry list resolves to.
///
/// Invalid entries are skipped rather than failing the span: the span is a
/// *bound* on where writes through this list can land, and an entry that does
/// not resolve is one that no write can reach. `None` when nothing resolves.
///
/// The bound is inclusive of `hi` — it names the first byte of the last page,
/// not the end of it — so a caller reporting a range must add one page. The
/// page list is not sorted and is not contiguous, so `[lo, hi]` is a hull and
/// not a promise that every page inside it belongs to this surface.
pub(crate) fn entry_gpa_span(entries: &[u32], page_shift: u32) -> Option<(u64, u64)> {
    let (mut lo, mut hi) = (u64::MAX, 0u64);
    for &e in entries {
        if let Some(gpa) = crate::contract::iosurface_pages::entry_gpa_shift(e, page_shift) {
            lo = lo.min(gpa);
            hi = hi.max(gpa);
        }
    }
    (lo != u64::MAX).then_some((lo, hi))
}

pub(crate) fn plan_adoption_decision(
    condemned: Option<&[u32]>,
    current: &[u32],
    plan: &[u32],
) -> (bool, bool, bool) {
    let pages_changed = match condemned {
        Some(old) => old != plan,
        None => current != plan,
    };
    (
        pages_changed,
        condemned.is_some() && pages_changed,
        condemned.is_some() && !pages_changed,
    )
}

/// True when the cached page table covers the type-11 sample/write span for
/// the latched geom (archive table build uses the same min_size).
///
/// Early resolve often runs before type-11 object dims land (`min_size` =
/// PAGE_SIZE only). Leaving a short `page_entries` while `has_geom` is true
/// makes Store writeback and sample page walks fail-closed on tiles (Favourites
/// 249² with ~16 pages vs ~63 required) while the Metal attachment still holds
/// content. Re-resolve when the span no longer fits.
pub fn pages_cover_geom(state: &DeviceState, mapping_id: u32) -> bool {
    let Some(m) = state.mappings.get(&mapping_id) else {
        return false;
    };
    if m.page_entries.is_empty() {
        return false;
    }
    if !m.has_geom || m.width == 0 || m.height == 0 {
        // No geom yet — any non-empty table is acceptable until dims latch.
        return true;
    }
    let format = if m.format != 0 {
        m.format
    } else {
        // Match scanout/writeback default when format not latched.
        crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM
    };
    let Some((_, _, span_end, _)) = sample_window_prefer_device(
        if m.device_desc.len() >= DEVICE_DESC_LEN {
            Some(m.device_desc.as_slice())
        } else {
            None
        },
        None,
        format,
        m.width,
        m.height,
    ) else {
        return false;
    };
    let page_size = crate::contract::iosurface_pages::page_size_of(state.page_shift);
    let covered = (m.page_entries.len() as u64).saturating_mul(page_size);
    covered >= span_end.max(page_size)
}

/// Ensure pages (and geom if possible) before scanout paint / type-11 Store.
///
/// Re-resolves when the table is empty, geom is missing, **or** the cached page
/// count cannot cover the latched W×H sample window (stale early resolve).
pub fn ensure_resolved_for_scanout<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &H,
    mapping_id: u32,
) -> bool {
    let (mapped, has_internal, empty_pages, has_geom) = match state.mappings.get(&mapping_id) {
        Some(m) => (
            m.mapped,
            m.mapping_internal != 0,
            m.page_entries.is_empty(),
            m.has_geom,
        ),
        None => return false,
    };
    let needs = mapped
        && has_internal
        && (empty_pages || !has_geom || !pages_cover_geom(state, mapping_id));
    if needs {
        resolve_mapping_backing(state, host, mapping_id)
    } else {
        mapped && !empty_pages
    }
}

/// Fail-closed page-list revalidation before host writeback or import-present.
///
/// When `mapping_internal` is set **and** we previously resolved a live
/// `page_table_kva`, re-walk MappingInternal so we never write through PFNs the
/// guest recycled (zone freelist `0xff000000ff000000` class). Resolve failure
/// **invalidates** a live table rather than writing stale PFNs.
///
/// Manual / unit-test page lists (`page_table_kva == 0`) keep their entries when
/// resolve is not available — product MAP always re-resolves once KVA is known.
pub fn revalidate_mapping_pages<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &H,
    mapping_id: u32,
) -> bool {
    revalidate_mapping_reason(state, host, mapping_id).is_none()
}

/// Precise reason a revalidate missed, or `None` when the mapping is resolvable.
///
/// The bool [`revalidate_mapping_pages`] collapses four distinct outcomes into
/// one "false", which forces a caller that fail-logs a lost flush to emit a
/// single `reason=revalidate` slug — and a future reader then cannot tell a
/// benign teardown window from a genuine content-drop without hunting for a
/// paired `map_revalidate resolve_fail` line. This returns the specific slug so
/// the two never share a status (AGENTS.md: each distinct check owns its slug):
/// - `revalidate_gone` / `revalidate_unmapped` — the guest already dropped the
///   mapping (pageoff/unwire raced ahead of the flush trigger); nothing to write
///   back to, benign.
/// - `revalidate_resolve_fail` — a live page table turned unreadable; the real
///   content-drop risk, and the only one that also emits the `st=invalidate`
///   line below.
///
/// The empty-page-list outcome is not one outcome. Four different states reach
/// it, and they were all reported as `revalidate_no_pages` with a doc comment
/// calling the class "a transient (re)wire gap" — one of the four, asserted for
/// all of them. 106 render-flush losses across 73 boots carry that slug and none
/// of them says which state produced it. Each check now owns its own:
/// - `revalidate_condemned` — `DeleteIOSurfaceBacking2` moved the page list into
///   `condemned_entries` and no resolve has re-adopted it. The guest deleted the
///   backing; there is nothing safe to write through.
/// - `revalidate_no_internal` — no `MappingInternal`, so **no resolve was ever
///   attempted**, and the page list is empty for some other reason. Note that a
///   zero `mapping_internal` is NOT itself a sign of missing backing: measured on
///   the rail, 2280 render windows in one boot were armed on mappings with
///   `mapping_internal == 0` and `page_entries.len() == 2040`, and all but two
///   flushed normally, because a non-empty page list returns `None` above.
/// - `revalidate_resolve_miss` — resolve ran and missed, with no live page table
///   to condemn (so not `resolve_fail`).
/// - `revalidate_empty_after_resolve` — resolve ran and *succeeded*, and the page
///   list is still empty. The genuinely surprising one.
/// - `revalidate_unmapped_late` / `revalidate_gone_late` — the mapping was
///   mapped on entry and is unmapped or absent after resolve; teardown raced the
///   revalidate, which the entry-side `revalidate_unmapped` cannot see.
pub fn revalidate_mapping_reason<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &H,
    mapping_id: u32,
) -> Option<&'static str> {
    let Some(m) = state.mappings.get(&mapping_id) else {
        return Some("revalidate_gone");
    };
    if !m.mapped {
        return Some("revalidate_unmapped");
    }
    let has_internal = m.mapping_internal != 0;
    let had_live_table = m.page_table_kva != 0;
    let had_pages = !m.page_entries.is_empty();
    // Whether the resolve below ran at all, and whether it reported success —
    // the two facts that separate the empty-page-list outcomes from each other.
    let mut resolve_ran = false;
    let mut resolve_ok = false;
    if has_internal {
        let generation_before = m.map_generation;
        let started = std::time::Instant::now();
        let resolved = resolve_mapping_backing(state, host, mapping_id);
        resolve_ran = true;
        resolve_ok = resolved;
        let elapsed_us = started.elapsed().as_micros() as u64;
        let (pages_after, generation_after) = state
            .mappings
            .get(&mapping_id)
            .map(|entry| (entry.page_entries.len(), entry.map_generation))
            .unwrap_or((0, generation_before));
        if revalidate_timing_is_slow(elapsed_us) {
            crate::observe::off(format!(
                "map_revalidate_slow mid={mapping_id} us={elapsed_us} pages={pages_after} resolved={} generation={} changed={}",
                resolved as u8,
                generation_after,
                (generation_after != generation_before) as u8
            ));
        }
        if !resolved && had_live_table {
            // Product: table was live and is now unreadable — drop PFNs.
            if had_pages {
                let _ = state.invalidate_mapping_pages(mapping_id);
                crate::observe::fail(format!(
                    "map_revalidate mid={mapping_id} st=invalidate reason=resolve_fail"
                ));
            }
            return Some("revalidate_resolve_fail");
        }
        // No prior live KVA (first resolve miss, or test fixture with manual
        // page_entries only) — fall through to accept non-empty manual list.
    }
    match state.mappings.get(&mapping_id) {
        Some(m) if m.mapped && !m.page_entries.is_empty() => None,
        Some(m) if !m.mapped => Some("revalidate_unmapped_late"),
        Some(m) if m.condemned_entries.is_some() => Some("revalidate_condemned"),
        Some(_) if !resolve_ran => Some("revalidate_no_internal"),
        Some(_) if !resolve_ok => Some("revalidate_resolve_miss"),
        Some(_) => Some("revalidate_empty_after_resolve"),
        None => Some("revalidate_gone_late"),
    }
}

const REVALIDATE_SLOW_US: u64 = 1_000;

#[inline]
fn revalidate_timing_is_slow(elapsed_us: u64) -> bool {
    elapsed_us >= REVALIDATE_SLOW_US
}

/// Unmap contiguous views whose page tables changed. No GPU object can hold one
/// of these views: nothing on either backend aliases guest pages any more, so
/// the only readers are CPU copies that finish inside their own call.
pub fn flush_retired_views<H: HostOps>(state: &mut DeviceState, host: &mut H) {
    for (ptr, len) in state.retired_views.drain(..) {
        host.unmap_pages(ptr, len);
    }
    // Same shape and the same reason: a guest-write token is host-side state
    // for a page list that no longer exists, and only the host can free it.
    for token in state.retired_guest_write_tokens.drain(..) {
        host.untrack_guest_writes(token);
    }
}

/// The live guest-write token for this mapping's current page list, asking the
/// host for one if the list has none.
///
/// Registration is what makes the host observe writes to these pages at all,
/// so it happens where the device first cares — at the Store that publishes
/// the surface — rather than at every mapping resolve. A host that cannot
/// observe guest writes answers `None` here forever, and every consumer reads
/// that as "assume written".
///
/// Reads `page_entries` directly instead of going through
/// [`mapping_page_gpas`]: the revalidation that function performs is a guest
/// page-table walk, and the caller has already proved the mapping resolvable by
/// rendering into it.
///
/// What keys the token to the surface's *current* pages is
/// [`crate::model::MappingEntry::map_generation`], not the eager retirement in
/// the lifecycle mutators. Two writers replace `page_entries` in place without
/// going near those mutators — the mapper's own plan adoption and the type-4
/// page refresh — and both retired the contiguous view while leaving the token
/// alone. Both do bump the generation exactly when the list changes, so
/// checking it here makes a carried-over token unusable by construction instead
/// of depending on every future writer remembering a second thing to retire.
pub fn ensure_guest_write_token<H: HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    mapping_id: u32,
) -> Option<u64> {
    let page_shift = state.page_shift;
    let page_size = state.page_size() as usize;
    let m = state.mappings.get(&mapping_id)?;
    let map_generation = m.map_generation;
    if m.guest_write_token != 0 {
        if m.guest_write_token_gen == map_generation {
            return Some(m.guest_write_token);
        }
        // The list moved underneath the token. Everything recorded against it
        // describes pages this surface may no longer own, so the Store stamp
        // goes with it.
        let e = state.mappings.get_mut(&mapping_id)?;
        let stale = std::mem::replace(&mut e.guest_write_token, 0);
        e.guest_write_token_gen = 0;
        e.guest_write_gen_at_store = 0;
        state.retired_guest_write_tokens.push(stale);
    }
    let m = state.mappings.get(&mapping_id)?;
    if !m.mapped || m.page_entries.is_empty() {
        return None;
    }
    let gpas: Vec<u64> = m
        .page_entries
        .iter()
        .filter_map(|&e| crate::contract::iosurface_pages::entry_gpa_shift(e, page_shift))
        .collect();
    // A partial list would have the host watch some of the surface and report
    // "unwritten" for the rest, which is the one answer that must never be
    // invented.
    if gpas.len() != m.page_entries.len() {
        return None;
    }
    let token = host.track_guest_writes(&gpas, page_size)?;
    let e = state.mappings.get_mut(&mapping_id)?;
    e.guest_write_token = token;
    e.guest_write_token_gen = map_generation;
    Some(token)
}

/// Revalidate + collect page-aligned GPAs for a mapped surface (GVA order).
///
/// Fails closed on empty / invalid entries and known transport/control-page
/// aliases. Does not invent PFNs. Every consumer immediately passes the
/// returned GPAs to `HostOps::map_pages`, whose host callback is the
/// authoritative RAM/range validator; repeating `is_ram_gpa` once per page
/// here makes full-frame surfaces perform thousands of duplicate QEMU address
/// translations before the exact same validation in `map_pages`.
pub fn mapping_page_gpas<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    mapping_id: u32,
) -> Option<Vec<u64>> {
    if !{ revalidate_mapping_pages(state, host, mapping_id) } {
        return None;
    }
    let m = state.mappings.get(&mapping_id)?;
    if !m.mapped || m.page_entries.is_empty() {
        return None;
    }
    let page_shift = state.page_shift;
    let gpas: Vec<u64> = m
        .page_entries
        .iter()
        .filter_map(|&e| crate::contract::iosurface_pages::entry_gpa_shift(e, page_shift))
        .collect();
    if gpas.is_empty() || gpas.len() != m.page_entries.len() {
        return None;
    }
    if let Some((gpa, owner)) = { first_control_page_collision(state, &gpas) } {
        crate::observe::fail(format!(
            "mapping_pages fail reason=control_page_collision mid={mapping_id} gpa={gpa:#x} owner={owner} pages={}",
            gpas.len()
        ));
        return None;
    }
    Some(gpas)
}

/// A render surface must never alias pages that the device knows are live
/// transport or task-control structures. `is_ram_gpa` alone cannot distinguish
/// an IOSurface page from a FIFO/page-table page; reject the provable overlap
/// before either CPU or GPU writes touch it.
fn first_control_page_collision(state: &DeviceState, gpas: &[u64]) -> Option<(u64, &'static str)> {
    let page = state.page_size();
    let page_base = |gpa: u64| gpa & !(page - 1);
    // Probe the SURFACE, not the control structures. A live task can advertise a
    // million object-list slots — 4,096 x86 pages — and an object list is one
    // contiguous span, so asking "does the surface hold this page?" once per
    // control page enumerated the whole span page by page. Sorted, the same
    // question is one range query per task.
    //
    // Measured on the x86/Vulkan rail before this shape: 414 µs per call, 71 024
    // calls in a 120 s arm, 29.4 s of wall clock spent proving a full-screen
    // IOSurface does not alias a FIFO ring.
    let mut pages: Vec<u64> = gpas.iter().map(|&gpa| page_base(gpa)).collect();
    pages.sort_unstable();
    let holds = |gpa: u64| pages.binary_search(&page_base(gpa)).is_ok();
    // The lowest surface page in `[start, end)`, if any. Both bounds and every
    // entry are multiples of `page`, so a hit is exactly one of the pages the
    // per-page form would have enumerated — same page, same reported gpa.
    let holds_range = |start: u64, end: u64| -> Option<u64> {
        let i = pages.partition_point(|&p| p < start);
        pages.get(i).copied().filter(|&p| p < end)
    };

    if state.gfx.root_page != 0 && holds((state.gfx.root_page as u64) << state.page_shift) {
        return Some(((state.gfx.root_page as u64) << state.page_shift, "gfx_root"));
    }
    if state.gfx.fifo_base_page != 0 && holds((state.gfx.fifo_base_page as u64) << state.page_shift)
    {
        return Some((
            (state.gfx.fifo_base_page as u64) << state.page_shift,
            "root_fifo",
        ));
    }
    if state.iosfc.ring_base != 0 && holds(state.iosfc.ring_base) {
        return Some((page_base(state.iosfc.ring_base), "iosfc_ring"));
    }
    for ring in &state.child_rings {
        for &gpa in &ring.page_gpas {
            if holds(gpa) {
                return Some((page_base(gpa), "child_fifo"));
            }
        }
    }
    for task in &state.tasks {
        if !task.active {
            continue;
        }
        if task.directory_pfn != 0 {
            let gpa = (task.directory_pfn as u64) << state.page_shift;
            if holds(gpa) {
                return Some((gpa, "task_directory"));
            }
        }
        if task.object_list_pfn != 0 {
            let first = (task.object_list_pfn as u64) << state.page_shift;
            let bytes = (task.object_list_count as u64).saturating_mul(16);
            let count = bytes.saturating_add(page - 1) / page;
            let end = first.saturating_add(count.saturating_mul(page));
            if let Some(gpa) = holds_range(first, end) {
                return Some((gpa, "task_object_list"));
            }
        }
    }
    None
}

/// Contiguous host-VA view over the mapping's guest pages (unified memory).
///
/// Builds the view on first use via [`HostOps::map_pages`] (mach_vm_remap of
/// guest RAM). Returns `(ptr, len)`. The view is the single storage for
/// surface content: Metal textures are created directly on it, so there is
/// nothing to synchronize — resolve failure here must fail the caller visibly.
///
/// **Safe zero-copy contract:** always [`revalidate_mapping_pages`] first so a
/// cached contig never aliases PFNs after ReplacePhysical / guest recycle.
///
/// On Linux, only a **packed** sequential host run succeeds. Fragmented
/// IOSurface page lists must use [`write_mapping_bytes`] / [`read_mapping_bytes`]
/// or multi-run import-present.
/// Device-wide count of [`ensure_contig_view`] calls answered "fragmented",
/// whether the verdict was derived or served from `contig_fragmented_gen`.
/// Reported as `served=` on every `contig_view_fragmented` line so the
/// magnitude the old per-call line carried survives its deduplication.
static CONTIG_FRAGMENTED_SERVED: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

pub fn ensure_contig_view<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    mapping_id: u32,
) -> Option<(usize, usize)> {
    // Always revalidate before returning a cached contig (ReplacePhysical /
    // recycle must not leave a live view over freelist PFNs).
    if !revalidate_mapping_pages(state, host, mapping_id) {
        return None;
    }
    flush_retired_views(state, host);
    {
        let m = state.mappings.get(&mapping_id)?;
        if m.contig_ptr != 0 {
            return Some((m.contig_ptr, m.contig_len));
        }
        // The negative verdict caches on exactly the key that makes the
        // positive one above safe. Re-deriving it per call collected the page
        // GPAs and rescanned them every time, and said so in the always-on sink
        // every time: 471 757 lines in one 2 900 s boot, the sole prefix ever to
        // trip `log_flood_detected`, at up to 1 826 lines in a one-second window.
        if m.contig_fragmented_gen == Some(m.map_generation) {
            CONTIG_FRAGMENTED_SERVED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return None;
        }
    }
    let gpas = mapping_page_gpas(state, host, mapping_id)?;
    let page_sz = crate::contract::iosurface_pages::page_size_of(state.page_shift) as usize;
    // A fragmented page list can never map as one packed view, and asking
    // anyway turns documented control flow ("use write_mapping_bytes /
    // read_mapping_bytes / multi-run import-present") into a logged
    // `qemu_map_pages_callback_failed`.
    let runs = crate::runtime::gva_view::contig_run_count(&gpas, page_sz as u64);
    if runs != 1 {
        let served = CONTIG_FRAGMENTED_SERVED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let m = state.mappings.get_mut(&mapping_id)?;
        m.contig_fragmented_gen = Some(m.map_generation);
        let generation = m.map_generation;
        // One line per (mapping, page list) rather than per call. The count the
        // per-call line used to carry moves to `served`, a device-wide
        // cumulative total of fragmented answers, so a census still reads
        // magnitude off the newest line while the line count now measures
        // distinct fragmented page lists — which is what the slug claims.
        crate::observe::off(format!(
            "contig_view_fragmented mid={mapping_id} pages={} runs={runs} generation={generation} served={served}",
            gpas.len(),
        ));
        return None;
    }
    let ptr = host.map_pages(&gpas, page_sz)?;
    let len = gpas.len() * page_sz;
    let m = state.mappings.get_mut(&mapping_id)?;
    m.contig_ptr = ptr;
    m.contig_len = len;
    Some((ptr, len))
}

/// Write `buf` into mapping linear offset `off` via packed map_pages runs.
///
/// Covers fragmented page lists (Linux product): split GPAs into maximal packed
/// runs, map each, poke, unmap. No `write_gpa`. Returns false if revalidate /
/// map fails.
pub fn write_mapping_bytes<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    mapping_id: u32,
    off: u64,
    buf: &[u8],
) -> bool {
    if buf.is_empty() {
        return true;
    }
    // Deferred-writeback flush-on-access: land any pending resident content
    // in these pages first so this write applies on top of it, not under it.
    crate::runtime::storage_flush::flush_intersecting(
        state,
        host,
        mapping_id,
        off,
        off.saturating_add(buf.len() as u64),
    );
    // Exact-window residency invalidation: guest pages in this range no
    // longer mirror any resident storage image (disjoint windows survive).
    state.invalidate_storage_residency_window(
        mapping_id,
        off,
        off.saturating_add(buf.len() as u64),
    );
    // Fast path: one packed view covering the write.
    let need_end = off.saturating_add(buf.len() as u64);
    if let Some((ptr, len)) = ensure_contig_view(state, host, mapping_id) {
        if (len as u64) >= need_end && (off as usize) + buf.len() <= len {
            // SAFETY: view covers need_end.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    buf.as_ptr(),
                    (ptr as *mut u8).add(off as usize),
                    buf.len(),
                );
            }
            return true;
        }
    }
    let gpas = match mapping_page_gpas(state, host, mapping_id) {
        Some(g) => g,
        None => {
            crate::observe::fail(format!(
                "mapping_write fail reason=revalidate mid={mapping_id} off={off:#x} len={:#x}",
                buf.len()
            ));
            return false;
        }
    };
    let page_size = state.page_size();
    let page_sz = page_size as usize;
    let span_end = (gpas.len() as u64).saturating_mul(page_size);
    if need_end > span_end {
        crate::observe::fail(format!(
            "mapping_write fail reason=short_table mid={mapping_id} off={off:#x} len={:#x} span={span_end:#x}",
            buf.len()
        ));
        return false;
    }
    flush_retired_views(state, host);
    let runs = crate::runtime::gva_view::contig_page_runs(&gpas, page_size);
    let import_started = std::time::Instant::now();
    let end = need_end;
    for run in &runs {
        let run_gpas = &gpas[run.clone()];
        let run_mlo = (run.start as u64).saturating_mul(page_size);
        let run_mhi = (run.end as u64).saturating_mul(page_size);
        let copy_lo = off.max(run_mlo);
        let copy_hi = end.min(run_mhi);
        if copy_lo >= copy_hi {
            continue;
        }
        let Some(ptr) = host.map_pages(run_gpas, page_sz) else {
            crate::observe::fail(format!(
                "mapping_write fail reason=map_pages mid={mapping_id} run_pages={} mlo={run_mlo:#x}",
                run_gpas.len()
            ));
            return false;
        };
        let total = run_gpas.len().saturating_mul(page_sz);
        let buf_off = (copy_lo - off) as usize;
        let host_off = (copy_lo - run_mlo) as usize;
        let n = (copy_hi - copy_lo) as usize;
        if host_off + n > total || buf_off + n > buf.len() {
            host.unmap_pages(ptr, total);
            return false;
        }
        // SAFETY: map covers total; host_off+n in range.
        unsafe {
            std::ptr::copy_nonoverlapping(
                buf.as_ptr().add(buf_off),
                (ptr as *mut u8).add(host_off),
                n,
            );
        }
        host.unmap_pages(ptr, total);
    }
    let import_us = import_started.elapsed().as_micros() as u64;
    if mapping_run_import_is_slow(import_us) {
        crate::observe::off(format!(
            "mapping_write_runs mid={mapping_id} us={import_us} bytes={} pages={} runs={}",
            buf.len(),
            gpas.len(),
            runs.len()
        ));
    }
    true
}

/// Read mapping linear `[off, off+buf.len())` via packed map_pages runs.
pub fn read_mapping_bytes<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    mapping_id: u32,
    off: u64,
    buf: &mut [u8],
) -> bool {
    if buf.is_empty() {
        return true;
    }
    // Deferred-writeback flush-on-access: this read must observe the resident
    // content, not the stale pre-dispatch guest bytes.
    crate::runtime::storage_flush::flush_intersecting(
        state,
        host,
        mapping_id,
        off,
        off.saturating_add(buf.len() as u64),
    );
    let need_end = off.saturating_add(buf.len() as u64);
    if let Some((ptr, len)) = ensure_contig_view(state, host, mapping_id) {
        if (len as u64) >= need_end && (off as usize) + buf.len() <= len {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    (ptr as *const u8).add(off as usize),
                    buf.as_mut_ptr(),
                    buf.len(),
                );
            }
            return true;
        }
    }
    let gpas = match mapping_page_gpas(state, host, mapping_id) {
        Some(g) => g,
        None => return false,
    };
    let page_size = state.page_size();
    let page_sz = page_size as usize;
    let span_end = (gpas.len() as u64).saturating_mul(page_size);
    if need_end > span_end {
        return false;
    }
    flush_retired_views(state, host);
    let runs = crate::runtime::gva_view::contig_page_runs(&gpas, page_size);
    let import_started = std::time::Instant::now();
    let end = need_end;
    for run in &runs {
        let run_gpas = &gpas[run.clone()];
        let run_mlo = (run.start as u64).saturating_mul(page_size);
        let run_mhi = (run.end as u64).saturating_mul(page_size);
        let copy_lo = off.max(run_mlo);
        let copy_hi = end.min(run_mhi);
        if copy_lo >= copy_hi {
            continue;
        }
        let Some(ptr) = host.map_pages(run_gpas, page_sz) else {
            return false;
        };
        let total = run_gpas.len().saturating_mul(page_sz);
        let buf_off = (copy_lo - off) as usize;
        let host_off = (copy_lo - run_mlo) as usize;
        let n = (copy_hi - copy_lo) as usize;
        if host_off + n > total || buf_off + n > buf.len() {
            host.unmap_pages(ptr, total);
            return false;
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                (ptr as *const u8).add(host_off),
                buf.as_mut_ptr().add(buf_off),
                n,
            );
        }
        host.unmap_pages(ptr, total);
    }
    let import_us = import_started.elapsed().as_micros() as u64;
    if mapping_run_import_is_slow(import_us) {
        crate::observe::off(format!(
            "mapping_read_runs mid={mapping_id} us={import_us} bytes={} pages={} runs={}",
            buf.len(),
            gpas.len(),
            runs.len()
        ));
    }
    true
}

const MAPPING_RUN_IMPORT_SLOW_US: u64 = 1_000;

#[inline]
fn mapping_run_import_is_slow(elapsed_us: u64) -> bool {
    elapsed_us >= MAPPING_RUN_IMPORT_SLOW_US
}

#[cfg(test)]
mod revalidate_tests {
    use super::*;
    use crate::contract::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
    use crate::model::{DeviceId, PAGE_SHIFT_X86};
    use crate::runtime::host::FakeHost;

    #[test]
    fn page_table_revalidation_slow_proxy_threshold_is_explicit() {
        assert!(!revalidate_timing_is_slow(REVALIDATE_SLOW_US - 1));
        assert!(revalidate_timing_is_slow(REVALIDATE_SLOW_US));
    }

    #[test]
    fn fragmented_run_import_slow_proxy_threshold_is_explicit() {
        assert!(!mapping_run_import_is_slow(MAPPING_RUN_IMPORT_SLOW_US - 1));
        assert!(mapping_run_import_is_slow(MAPPING_RUN_IMPORT_SLOW_US));
    }

    #[test]
    fn revalidate_fail_closed_without_internal_and_empty_pages() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let host = FakeHost::new();
        state.map_surface(2);
        // Mapped but no MappingInternal and no pages → not writable.
        assert!(!revalidate_mapping_pages(&mut state, &host, 2));
    }

    #[test]
    fn revalidate_reason_disambiguates_the_miss() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let host = FakeHost::new();
        // Unknown id → the mapping was never created / already forgotten.
        assert_eq!(
            revalidate_mapping_reason(&mut state, &host, 7),
            Some("revalidate_gone")
        );
        // Mapped, no MappingInternal, no page list. The resolve never ran, so
        // this says nothing about whether the pages exist — which is exactly
        // what the old shared `revalidate_no_pages` slug hid behind a comment
        // calling it a benign (re)wire gap.
        state.map_surface(2);
        assert_eq!(
            revalidate_mapping_reason(&mut state, &host, 2),
            Some("revalidate_no_internal")
        );
        // Unmapped on entry is caught before the resolve and keeps its own slug.
        state.map_surface(3);
        state.mappings.get_mut(&3).unwrap().mapped = false;
        assert_eq!(
            revalidate_mapping_reason(&mut state, &host, 3),
            Some("revalidate_unmapped")
        );
        // A condemned backing is empty for a REASON the guest gave us
        // (DeleteIOSurfaceBacking2 stashed the page list), which is a different
        // answer from "the page list happens to be empty" and must not share its
        // slug — a deferred window flushing here has nothing safe to write
        // through, rather than nothing resolved yet.
        state.map_surface(5);
        state.mappings.get_mut(&5).unwrap().page_entries =
            vec![(0x200 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
        assert!(state.condemn_surface_backing(5));
        assert_eq!(
            revalidate_mapping_reason(&mut state, &host, 5),
            Some("revalidate_condemned")
        );
        // A resolvable static page list → success (None).
        state.map_surface(4);
        state.mappings.get_mut(&4).unwrap().page_entries =
            vec![(0x100 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
        assert_eq!(revalidate_mapping_reason(&mut state, &host, 4), None);
    }

    #[test]
    fn surface_page_collision_detects_only_distinct_live_alias() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let entry = |pfn: u32| (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
        let gpa = |pfn: u64| pfn << PAGE_SHIFT_X86;
        // Two distinct live surfaces on disjoint pages → no collision.
        state.map_surface(10);
        state.map_surface(20);
        state.mappings.get_mut(&10).unwrap().page_entries = vec![entry(0x100), entry(0x101)];
        state.mappings.get_mut(&20).unwrap().page_entries = vec![entry(0x200), entry(0x201)];
        assert_eq!(first_surface_page_collision(&state, 10), None);
        assert_eq!(first_surface_page_collision(&state, 20), None);
        // Surface 20 rewires onto a page surface 10 still owns → collision,
        // reported against the other owner (10) at the shared GPA.
        state.mappings.get_mut(&20).unwrap().page_entries = vec![entry(0x101), entry(0x201)];
        assert_eq!(
            first_surface_page_collision(&state, 20),
            Some((gpa(0x101), 10))
        );
        // A surface never collides with itself.
        assert_eq!(
            first_surface_page_collision(&state, 10),
            Some((gpa(0x101), 20))
        );
        // If the other owner is unmapped, the alias is legitimate (handoff) →
        // no collision.
        state.unmap_surface(10);
        assert_eq!(first_surface_page_collision(&state, 20), None);
        // Empty / unmapped self → None.
        state.mappings.get_mut(&20).unwrap().page_entries.clear();
        assert_eq!(first_surface_page_collision(&state, 20), None);
    }

    #[test]
    fn reprieve_with_aliasing_peer_is_a_detected_collision() {
        // The condemn/reprieve corruptor precondition: a mapping's backing was
        // deleted (condemn stashed its pages), the guest handed the SAME
        // physical pages to another live surface, but this mapping's page table
        // still resolves to them — so the resolve fingerprints identical and
        // REPRIEVES (pages_changed == false, no map_generation bump). The rewire
        // wrong-PFN guard is gated on pages_changed and would never run; the
        // reprieve-path guard must catch it. This asserts both halves the branch
        // composes fire together on that exact state.
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let entry = |pfn: u32| (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
        let gpa = |pfn: u64| pfn << PAGE_SHIFT_X86;

        // Mapping 3: condemned, its stashed pages == the plan it re-adopts.
        state.map_surface(3);
        {
            let m = state.mappings.get_mut(&3).unwrap();
            m.mapped = true;
            m.page_entries = vec![entry(0x300), entry(0x301)];
            m.map_generation = 4;
        }
        assert!(state.condemn_surface_backing(3));
        // The re-walked plan matches the condemned fingerprint → reprieve.
        let condemned = state.mappings.get(&3).unwrap().condemned_entries.clone();
        let plan = vec![entry(0x300), entry(0x301)];
        let (pages_changed, incarnation_changed, reprieved) =
            plan_adoption_decision(condemned.as_deref(), &[], &plan);
        assert!(
            reprieved,
            "same plan as condemned fingerprint must reprieve"
        );
        assert!(!pages_changed, "reprieve must not see a page change");
        assert!(!incarnation_changed);

        // Re-adopt the plan (as the resolve would) and stand up a DIFFERENT live
        // surface (20) that now also owns page 0x301 — the guest recycled it.
        {
            let m = state.mappings.get_mut(&3).unwrap();
            m.page_entries = plan.clone();
            m.condemned_entries = None;
        }
        state.map_surface(20);
        {
            let m = state.mappings.get_mut(&20).unwrap();
            m.mapped = true;
            m.page_entries = vec![entry(0x301), entry(0x999)];
        }
        // The reprieve-path guard's detector fires: mapping 3's re-adopted page
        // 0x301 is also owned by live surface 20 — the wrong-PFN write vector the
        // rewire-only guard would have missed (pages_changed was false).
        assert_eq!(
            first_surface_page_collision(&state, 3),
            Some((gpa(0x301), 20))
        );
    }

    #[test]
    fn surface_page_collision_invalidates_mapping_fail_closed() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let entry = |pfn: u32| (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
        let gpa = |pfn: u64| pfn << PAGE_SHIFT_X86;
        const MID: u32 = 0x0CA;
        const OWNER: u32 = 0x0BE;

        state.map_surface(MID);
        {
            let m = state.mappings.get_mut(&MID).unwrap();
            m.mapped = true;
            m.map_generation = 7;
            m.page_entries = vec![entry(0x777), entry(0x778)];
            m.page_table_kva = 0xABC0;
        }
        state.map_surface(OWNER);
        {
            let m = state.mappings.get_mut(&OWNER).unwrap();
            m.mapped = true;
            m.page_entries = vec![entry(0x778)];
        }

        let (shared_gpa, owner) =
            first_surface_page_collision(&state, MID).expect("must detect alias");
        assert_eq!((shared_gpa, owner), (gpa(0x778), OWNER));

        fail_closed_surface_page_collision(&mut state, MID, shared_gpa, owner, 2, "test");
        let m = state.mappings.get(&MID).unwrap();
        assert!(m.mapped, "surface stays mapped but unresolved");
        assert!(
            m.page_entries.is_empty(),
            "known-bad page plan must be cleared"
        );
        assert_eq!(m.page_table_kva, 0);
        assert_eq!(
            m.map_generation, 8,
            "generation bump makes any deferred writeback fail closed"
        );
        assert_eq!(first_surface_page_collision(&state, MID), None);
    }

    #[test]
    fn revalidate_accepts_static_page_list_without_internal() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let host = FakeHost::new();
        state.map_surface(4);
        {
            let m = state.mappings.get_mut(&4).unwrap();
            m.page_entries = vec![(0x100 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
        }
        assert!(revalidate_mapping_pages(&mut state, &host, 4));
    }

    #[test]
    fn mapping_io_still_rejects_non_ram_page_at_map_boundary() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        host.strict_linux_map = true;
        let mid = 6;
        assert!(state.map_surface(mid));
        {
            let m = state.mappings.get_mut(&mid).unwrap();
            m.page_entries = vec![(0x7f000 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
        }
        let mut byte = [0u8; 1];
        assert!(!read_mapping_bytes(
            &mut state, &mut host, mid, 0, &mut byte,
        ));
        assert!(!write_mapping_bytes(&mut state, &mut host, mid, 0, &[1],));
    }

    #[test]
    fn invalidate_mapping_pages_bumps_map_generation_and_clears() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        state.map_surface(5);
        {
            let m = state.mappings.get_mut(&5).unwrap();
            m.page_entries = vec![1];
            m.contig_ptr = 0xdead;
            m.contig_len = 4096;
        }
        let gen0 = state.mappings.get(&5).unwrap().map_generation;
        assert!(state.invalidate_mapping_pages(5));
        let m = state.mappings.get(&5).unwrap();
        assert!(m.page_entries.is_empty());
        assert_eq!(m.contig_ptr, 0);
        assert!(m.map_generation != gen0);
        assert_eq!(state.retired_views, vec![(0xdead, 4096)]);
    }

    /// The "cannot pack" verdict is derived once per page list, not once per
    /// call, and a new page list re-derives it.
    ///
    /// Before this cache every call on a fragmented mapping rebuilt the page-GPA
    /// vector, rescanned it for runs, and emitted a line — 471 757 of them in
    /// one 2 900 s boot, the only prefix ever to trip `log_flood_detected`. The
    /// line count is therefore the assertion: repeated calls must add none, and
    /// the magnitude the old line carried must still be readable as `served=`.
    #[test]
    fn fragmented_verdict_is_derived_once_per_page_list() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        host.strict_linux_map = true;
        let page = 1u64 << PAGE_SHIFT_X86;
        let entry =
            |gpa: u64| ((gpa >> PAGE_SHIFT_X86) as u32) << PAGE_ENTRY_PFN_SHIFT | PAGE_ENTRY_VALID;
        // Non-adjacent guest pages — never one packed run.
        let (gpa0, gpa1, gpa2) = (0x1000_0000u64, 0x2000_0000u64, 0x3000_0000u64);
        for gpa in [gpa0, gpa1, gpa2] {
            host.map_range(gpa, page as usize, 0);
        }
        let mid = 9u32;
        state.map_surface(mid);
        state.mappings.get_mut(&mid).unwrap().page_entries = vec![entry(gpa0), entry(gpa1)];

        let cap = crate::observe::FailCapture::start();
        let lines = || -> Vec<String> {
            cap.lines()
                .into_iter()
                .filter(|l| l.starts_with("OFF contig_view_fragmented"))
                .collect()
        };
        for _ in 0..16 {
            assert!(
                ensure_contig_view(&mut state, &mut host, mid).is_none(),
                "fragmented list must never pack"
            );
        }
        let first = lines();
        assert_eq!(
            first.len(),
            1,
            "16 calls on one page list must derive (and say) the verdict once: {first:?}"
        );
        assert!(
            first[0].contains(" pages=2 runs=2 "),
            "the derived line keeps its shape: {}",
            first[0]
        );

        // A different page list is a different verdict: the generation bump that
        // retires `contig_ptr` must also retire this.
        {
            let m = state.mappings.get_mut(&mid).unwrap();
            m.page_entries = vec![entry(gpa0), entry(gpa1), entry(gpa2)];
            DeviceState::bump_map_generation(m);
        }
        assert!(ensure_contig_view(&mut state, &mut host, mid).is_none());
        let after = lines();
        assert_eq!(
            after.len(),
            2,
            "a new page list must re-derive and re-report: {after:?}"
        );
        assert!(
            after[1].contains(" pages=3 runs=3 "),
            "the second line describes the second list: {}",
            after[1]
        );

        // Magnitude survives deduplication: `served` counts every fragmented
        // answer, cached ones included, so it advanced by the 16 calls between
        // the two derivations.
        let served = |l: &str| -> u64 {
            l.rsplit_once("served=")
                .and_then(|(_, v)| v.split_whitespace().next())
                .and_then(|v| v.parse().ok())
                .expect("line carries served=")
        };
        assert_eq!(
            served(&after[1]) - served(&after[0]),
            16,
            "served must count cached answers too: {after:?}"
        );
    }

    /// Product Linux: full page list is non-packed → ensure_contig_view fails;
    /// write_mapping_bytes still lands bytes via maximal packed runs.
    #[test]
    fn multi_import_fragmented_mapping_write() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        host.strict_linux_map = true;
        let page = 1u64 << PAGE_SHIFT_X86;
        // Two non-adjacent guest pages (gap in GPA → not one packed map_pages).
        let gpa0 = 0x1000_0000u64;
        let gpa1 = 0x2000_0000u64;
        host.map_range(gpa0, page as usize, 0);
        host.map_range(gpa1, page as usize, 0);
        let pfn0 = (gpa0 >> PAGE_SHIFT_X86) as u32;
        let pfn1 = (gpa1 >> PAGE_SHIFT_X86) as u32;
        let mid = 9u32;
        state.map_surface(mid);
        {
            let m = state.mappings.get_mut(&mid).unwrap();
            m.page_entries = vec![
                (pfn0 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID,
                (pfn1 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID,
            ];
        }
        assert!(
            ensure_contig_view(&mut state, &mut host, mid).is_none(),
            "fragmented list must not pack under strict_linux_map"
        );
        let payload = b"FRAG-MULTI-IMPORT-OK!!!!"; // 24 bytes
        assert!(write_mapping_bytes(&mut state, &mut host, mid, 0, payload));
        // Second page offset = page_size.
        let mut hi = [0u8; 8];
        assert!(read_mapping_bytes(
            &mut state, &mut host, mid, page, &mut hi
        ));
        // Write only touched page 0; page 1 still zero.
        assert_eq!(hi, [0u8; 8]);
        let mut lo = [0u8; 24];
        assert!(read_mapping_bytes(&mut state, &mut host, mid, 0, &mut lo));
        assert_eq!(&lo[..], &payload[..]);
        // Cross-page write spanning the gap.
        let cross = vec![0xABu8; 16];
        let off = page - 8;
        assert!(write_mapping_bytes(&mut state, &mut host, mid, off, &cross));
        let mut check = [0u8; 16];
        assert!(read_mapping_bytes(
            &mut state, &mut host, mid, off, &mut check
        ));
        assert_eq!(check, [0xABu8; 16]);
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::contract::endian::st32;
    use crate::contract::iosurface_pages::{
        MAPPING_INTERNAL_BACKPTR, MAPPING_INTERNAL_EXPECTED_SIZE, MAPPING_INTERNAL_ID,
        MAPPING_INTERNAL_SIZE, PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID,
    };

    /// The span is a hull over the *resolvable* entries, in PFN order-independent
    /// fashion, and it must not be fooled by an unsorted list or by an invalid
    /// entry sitting at either end — those are exactly the shapes a real page
    /// list has, and a span that tracked first/last instead of min/max would
    /// name a range that does not contain the pages written.
    #[test]
    fn entry_gpa_span_is_a_hull_over_resolvable_entries() {
        let valid = |pfn: u32| (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
        // Unsorted, so first/last != min/max.
        let entries = [valid(9), valid(3), valid(7)];
        assert_eq!(
            entry_gpa_span(&entries, 12),
            Some((3u64 << 12, 9u64 << 12)),
            "min/max, not first/last"
        );
        // An invalid entry is skipped, not treated as GPA 0 — a zero would drag
        // `lo` to the bottom of RAM and make the span claim pages no write can
        // reach, which is the wrong direction for a bound used as evidence.
        let with_hole = [0u32, valid(3), valid(9), 0u32];
        assert_eq!(
            entry_gpa_span(&with_hole, 12),
            Some((3u64 << 12, 9u64 << 12))
        );
        // Page shift is honoured rather than assumed to be 12 (arm64 is 14).
        assert_eq!(entry_gpa_span(&entries, 14), Some((3u64 << 14, 9u64 << 14)));
        // Nothing resolvable ⇒ no span at all, rather than (u64::MAX, 0).
        assert_eq!(entry_gpa_span(&[0, 0], 12), None);
        assert_eq!(entry_gpa_span(&[], 12), None);
    }

    #[test]
    fn plan_adoption_decision_incarnation_semantics() {
        // No condemn: plain pages-changed compare against the live entries.
        assert_eq!(
            plan_adoption_decision(None, &[1, 2], &[1, 2]),
            (false, false, false)
        );
        assert_eq!(
            plan_adoption_decision(None, &[1, 2], &[1, 3]),
            (true, false, false)
        );
        // Condemned + identical plan = stale delete reprieve: the SAME
        // incarnation lives on — no bump, no drop (black-band class).
        assert_eq!(
            plan_adoption_decision(Some(&[1, 2]), &[], &[1, 2]),
            (false, false, true)
        );
        // Condemned + different plan = the backing really died and the id
        // was re-used: bump + drop the old incarnation's windows.
        assert_eq!(
            plan_adoption_decision(Some(&[1, 2]), &[], &[7, 8]),
            (true, true, false)
        );
        // The live (cleared) entries never mask the fingerprint compare.
        assert_eq!(
            plan_adoption_decision(Some(&[1, 2]), &[7, 8], &[1, 2]),
            (false, false, true)
        );
    }

    #[test]
    fn resolve_fail_latch_dedups_per_mapping_and_rearms_on_clear() {
        // Flood guard for the per-present `resolve_mapping_backing` path: a
        // genuinely-broken mapping must log each reason once, re-arm when it
        // resolves, and never bleed across mappings. Unique ids so this never
        // races real mappings across the process-global latch.
        let mid = 0xF00D_0001u32;
        let other = 0xF00D_0002u32;
        clear_resolve_fail(mid);
        clear_resolve_fail(other);
        let seen = |m: u32, r: &'static str| {
            resolve_fail_latch()
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .contains(&(m, r))
        };
        note_resolve_fail(mid, "iosurface_validate_mapping_id_mismatch", "x".into());
        assert!(seen(mid, "iosurface_validate_mapping_id_mismatch"));
        // A different reason on the same mapping is tracked independently.
        note_resolve_fail(mid, "iosurface_mapper_internal_owner_read", "x".into());
        assert!(seen(mid, "iosurface_mapper_internal_owner_read"));
        // A different mapping is untouched by mid's failures.
        assert!(!seen(other, "iosurface_validate_mapping_id_mismatch"));
        // Clearing mid re-arms both its reasons but leaves `other` alone.
        note_resolve_fail(other, "iosurface_validate_mapping_id_mismatch", "x".into());
        clear_resolve_fail(mid);
        assert!(!seen(mid, "iosurface_validate_mapping_id_mismatch"));
        assert!(!seen(mid, "iosurface_mapper_internal_owner_read"));
        assert!(seen(other, "iosurface_validate_mapping_id_mismatch"));
        clear_resolve_fail(other);
    }

    #[test]
    fn mapper_declines_are_exact_and_log_safe() {
        use crate::observe::Decline;

        let declines = [
            MapperDecline::CaptureMapperXregRead(MemError::XregUnavailable),
            MapperDecline::CaptureRequestTypeXregRead(MemError::XregUnavailable),
            MapperDecline::CaptureInternalXregRead(MemError::XregUnavailable),
            MapperDecline::CaptureRequestTypeMismatch,
            MapperDecline::CaptureInternalZero,
            MapperDecline::CaptureInternalKvaInvalid,
            MapperDecline::CaptureMapperKvaInvalid,
            MapperDecline::DeviceDescriptorRead(MemError::Unmapped),
        ];
        let mut slugs = std::collections::HashSet::new();
        for decline in declines {
            let slug = decline.slug();
            assert!(slug.starts_with("mapper_"));
            assert!(
                slug.bytes()
                    .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_'),
                "not log-safe: {slug}"
            );
            assert!(slugs.insert(slug), "duplicate mapper decline: {slug}");
        }
        assert_eq!(
            crate::observe::Emit::decline(
                "mapper_capture_fail",
                &MapperDecline::CaptureRequestTypeMismatch,
            )
            .field("mapping", 9)
            .render(),
            "mapper_capture_fail reason=mapper_capture_request_type_mismatch mapping=9"
        );
    }

    #[test]
    fn mapper_boundary_preserves_the_iosurface_check_reason() {
        let status =
            iosurface_pages::Status::ErrInternalRead("iosurface_mapper_internal_mapping_id_read");
        assert_eq!(
            refusal_reason(&status),
            "iosurface_mapper_internal_mapping_id_read"
        );
        assert_eq!(
            crate::observe::Emit::refusal("mapper_resolve_fail", &status)
                .unwrap()
                .field("mapping", 4)
                .render(),
            "mapper_resolve_fail reason=iosurface_mapper_internal_mapping_id_read \
             class=internal_read mapping=4"
        );
    }
    use crate::model::{DeviceId, PAGE_SHIFT_ARM64E, PAGE_SIZE_ARM64E};
    use crate::runtime::host::FakeHost;

    /// arm64e kernel VA base used by the contract.
    const KVA: u64 = 0xfffffe00_10000000;

    fn put_u32(h: &mut FakeHost, gpa: u64, v: u32) {
        h.map_range(gpa, 4, 0);
        h.put_u32(gpa, v);
    }
    fn put_u64(h: &mut FakeHost, gpa: u64, v: u64) {
        h.map_range(gpa, 8, 0);
        let b = v.to_le_bytes();
        let _ = h.write_gpa(gpa, &b);
    }

    #[test]
    fn capture_validates_identity_and_ring() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let mut host = FakeHost::new();
        let ring = 0x7000_0000u64;
        state.iosfc.ring_base = ring;

        // producer=1 → entry 0: MAP mapping_id=7
        let mut entry = [0u8; 16];
        st32(&mut entry[0..], MAPPER_REQUEST_MAP);
        st32(&mut entry[4..], 7);
        host.map_range(ring, 16, 0);
        let _ = host.write_gpa(ring, &entry);

        let internal = KVA;
        let mapper = KVA + 0x1000;
        // MappingInternal identity fields
        put_u64(&mut host, internal + MAPPING_INTERNAL_BACKPTR, mapper);
        put_u32(&mut host, internal + MAPPING_INTERNAL_ID, 7);
        put_u32(
            &mut host,
            internal + MAPPING_INTERNAL_SIZE,
            MAPPING_INTERNAL_EXPECTED_SIZE,
        );

        host.set_xreg(MAPPER_CAPTURE_REG_MAPPER_DEVICE, mapper);
        host.set_xreg(MAPPER_CAPTURE_REG_REQUEST_TYPE, MAPPER_REQUEST_MAP as u64);
        host.set_xreg(MAPPER_CAPTURE_REG_MAPPING_INTERNAL, internal);

        let cap = capture_at_producer(&state, &host, 1).expect("capture");
        assert_eq!(cap.producer, 1);
        assert_eq!(cap.mapping_internal, internal);
        assert!(apply_capture(&mut state, &cap, 7));
        assert_eq!(state.mappings.get(&7).unwrap().mapping_internal, internal);
    }

    #[test]
    fn capture_handoff_mismatch_is_fail_visible_and_latched() {
        // A decoded MAP request whose captured handoff registers disagree with
        // the ring (wrong request-type in the xreg) is a genuine capture miss:
        // the mapping never attaches → downstream black. It must return None,
        // latch its reason once (no per-publish flood), and re-arm on clear.
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let mut host = FakeHost::new();
        let ring = 0x7100_0000u64;
        state.iosfc.ring_base = ring;

        // producer=1 → entry 0: MAP mapping_id=9
        let mut entry = [0u8; 16];
        st32(&mut entry[0..], MAPPER_REQUEST_MAP);
        st32(&mut entry[4..], 9);
        host.map_range(ring, 16, 0);
        let _ = host.write_gpa(ring, &entry);

        let internal = KVA;
        // xreg request-type disagrees with the ring's MAP → handoff mismatch.
        host.set_xreg(MAPPER_CAPTURE_REG_MAPPER_DEVICE, 0);
        host.set_xreg(MAPPER_CAPTURE_REG_REQUEST_TYPE, MAPPER_REQUEST_UNMAP as u64);
        host.set_xreg(MAPPER_CAPTURE_REG_MAPPING_INTERNAL, internal);

        clear_resolve_fail(9);
        let seen = |m: u32, r: &'static str| {
            resolve_fail_latch()
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .contains(&(m, r))
        };
        assert!(capture_at_producer(&state, &host, 1).is_none());
        assert!(seen(9, "mapper_capture_request_type_mismatch"));
        // A second identical publish must not add a duplicate (still one entry).
        assert!(capture_at_producer(&state, &host, 1).is_none());
        // A clean resolve of the same mapping re-arms the capture reason.
        clear_resolve_fail(9);
        assert!(!seen(9, "mapper_capture_request_type_mismatch"));
    }

    #[test]
    fn resolve_builds_page_entries() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let mut host = FakeHost::new();
        let internal = KVA;
        let mapper = KVA + 0x1000;
        let page_obj = KVA + 0x2000;
        let table = KVA + 0x3000;
        let pfn = 0x1e88c_u32;
        let page_gpa = (pfn as u64) << PAGE_SHIFT_ARM64E;

        put_u64(&mut host, internal + MAPPING_INTERNAL_BACKPTR, mapper);
        put_u32(&mut host, internal + MAPPING_INTERNAL_ID, 3);
        put_u32(
            &mut host,
            internal + MAPPING_INTERNAL_SIZE,
            MAPPING_INTERNAL_EXPECTED_SIZE,
        );
        // page fields: 0x48 points at page_obj which has table ptr at +0xb8
        put_u64(
            &mut host,
            internal + iosurface_pages::MAPPING_INTERNAL_PAGE_FIELD_48,
            page_obj,
        );
        put_u64(
            &mut host,
            internal + iosurface_pages::MAPPING_INTERNAL_PAGE_FIELD_50,
            0,
        );
        put_u64(
            &mut host,
            internal + iosurface_pages::MAPPING_INTERNAL_PAGE_COUNT,
            1,
        );
        put_u64(
            &mut host,
            page_obj + iosurface_pages::MAPPING_PAGE_TABLE_FROM_F48,
            table,
        );
        let entry = (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
        put_u32(&mut host, table, entry);
        // one page of guest RAM for the surface
        host.map_range(page_gpa, PAGE_SIZE_ARM64E as usize, 0x55);

        state.mapper_device_kva = mapper;
        assert!(state.attach_mapping_internal(3, internal));
        // The adopted page list is what bounds every mapping-rail guest write,
        // so a successful resolve must report its guest-physical footprint. An
        // earlier cut of `mapping_gpa_span` keyed on `pages_changed` and was
        // silent for whole live boots while page lists were plainly being
        // adopted; asserting it here is what makes that failure loud in the
        // suite instead of only in a log nobody diffed.
        let cap = crate::observe::sink::FailCapture::start();
        assert!(resolve_mapping_backing(&mut state, &host, 3));
        let m = state.mappings.get(&3).unwrap();
        assert_eq!(m.page_entries.len(), 1);
        assert_eq!(m.page_entries[0], entry);
        let span = cap.one("OFF");
        assert!(
            span.contains("mapping_gpa_span mid=3") && span.contains("pages=1"),
            "resolve must report its adopted footprint, got {span:?}"
        );
        // The page number is what a guest panic prints (`pmap_page_protect()
        // ... pn=0x...`), so it has to be readable without arithmetic.
        assert!(
            span.contains(&format!("pn_lo={pfn:#x}")),
            "span must name the adopted PFN as a page number, got {span:?}"
        );
    }

    struct FailingKvaHost {
        inner: FakeHost,
        err: MemError,
    }

    impl HostMemory for FailingKvaHost {
        fn read_gpa(&self, gpa: u64, buf: &mut [u8]) -> Result<(), MemError> {
            self.inner.read_gpa(gpa, buf)
        }

        fn write_gpa(&mut self, gpa: u64, buf: &[u8]) -> Result<(), MemError> {
            self.inner.write_gpa(gpa, buf)
        }
    }

    impl HostOps for FailingKvaHost {
        fn mono_ns(&self) -> u64 {
            0
        }

        fn enqueue(&mut self, _action: crate::runtime::host::HostAction) {}

        fn schedule_bh(&mut self) {}

        fn read_kva(&self, _kva: u64, _buf: &mut [u8]) -> Result<(), MemError> {
            Err(self.err)
        }

        fn map_pages(&mut self, gpas: &[u64], page_size: usize) -> Option<usize> {
            self.inner.map_pages(gpas, page_size)
        }

        fn unmap_pages(&mut self, ptr: usize, len: usize) {
            self.inner.unmap_pages(ptr, len);
        }

        fn is_ram_gpa(&self, gpa: u64) -> bool {
            self.inner.is_ram_gpa(gpa)
        }
    }

    fn assert_revalidate_error_preserves_cached_page_plan(err: MemError) {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let entry = (0x444u32 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
        assert!(state.attach_mapping_internal(3, KVA));
        assert!(state.set_mapping_geom(
            3,
            64,
            64,
            crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM
        ));
        {
            let m = state.mappings.get_mut(&3).unwrap();
            m.mapped = true;
            m.page_entries = vec![entry];
            m.page_table_kva = KVA + 0x3000;
        }

        let host = FailingKvaHost {
            inner: FakeHost::new(),
            err,
        };
        clear_resolve_fail(3);
        let log_before = std::fs::read_to_string(crate::observe::fail_log_path())
            .unwrap_or_default()
            .len();
        assert_eq!(revalidate_mapping_reason(&mut state, &host, 3), None);
        let m = state.mappings.get(&3).unwrap();
        assert_eq!(m.page_entries, vec![entry]);
        assert_eq!(m.page_table_kva, KVA + 0x3000);
        let log_after =
            std::fs::read_to_string(crate::observe::fail_log_path()).unwrap_or_default();
        assert!(
            !log_after[log_before..].contains("mapper_revalidate_fallback"),
            "an expected cached-plan alias fallback must stay silent: {}",
            &log_after[log_before..]
        );
    }

    #[test]
    fn revalidate_no_cpu_preserves_cached_page_plan() {
        assert_revalidate_error_preserves_cached_page_plan(MemError::NoCpu);
    }

    #[test]
    fn revalidate_unmapped_read_preserves_cached_page_plan() {
        assert_revalidate_error_preserves_cached_page_plan(MemError::Unmapped);
    }

    /// qemu-shim: early page resolve + late geom must re-expand the table.
    /// IOSurface PAGE_SIZE is 16 KiB (arm64e). 1440×1080 BGRA needs
    /// ALIGN_UP(1440×4,128)×1080 = 6 220 800 bytes ≈ 380 pages; a 1-page stale
    /// table must not cover (dual-mid Store after mode switch).
    #[test]
    fn pages_cover_geom_false_when_table_shorter_than_span() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        assert!(state.attach_mapping_internal(8, KVA));
        {
            let m = state.mappings.get_mut(&8).unwrap();
            m.mapped = true;
            // Stale early resolve: single PAGE_SIZE before geom latched.
            m.page_entries = vec![0x11; 1];
        }
        assert!(state.set_mapping_geom(
            8,
            1440,
            1080,
            crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM
        ));
        assert!(
            !pages_cover_geom(&state, 8),
            "1×16KiB page cannot cover 1440×1080 BGRA sample window"
        );
        let host = FakeHost::new();
        let _ = ensure_resolved_for_scanout(&mut state, &host, 8);
        assert!(!pages_cover_geom(&state, 8));
    }

    #[test]
    fn pages_cover_geom_true_for_full_table() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        assert!(state.attach_mapping_internal(3, KVA));
        assert!(state.set_mapping_geom(
            3,
            64,
            64,
            crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM
        ));
        // 64×64 BGRA packed bpr 256 → 16 KiB → 1×16KiB page covers.
        {
            let m = state.mappings.get_mut(&3).unwrap();
            m.mapped = true;
            m.page_entries = vec![0x22; 1];
        }
        assert!(pages_cover_geom(&state, 3));
    }

    /// A page table cannot make a window the guest's own allocation does not
    /// contain, however many pages it holds.
    ///
    /// `sample_window_prefer_device` refuses when the invented packed span runs
    /// past `device_desc.alloc_size` — that refusal is the only place the wire
    /// allocation bounds the window. This case is the one where a caller could
    /// answer it by calling `sample_window` directly and getting the rejected
    /// span back: the descriptor says 1.5 MiB, the packed 1024² BGRA window is
    /// 4 MiB, and the table is deliberately sized to cover the 4 MiB. Falling
    /// back would report "covered" for a surface whose own descriptor says it is
    /// a third of that size.
    #[test]
    fn a_generous_table_cannot_cover_a_window_past_the_wire_allocation() {
        use crate::contract::endian::{st32, st64};
        use crate::contract::iosurface_pages::{
            DEVICE_DESC_ALLOC_SIZE, DEVICE_DESC_BPR, DEVICE_DESC_DIMS, DEVICE_DESC_LEN,
            DEVICE_DESC_PLANE_COUNT,
        };

        let mut desc = vec![0u8; DEVICE_DESC_LEN];
        // 1.5 MiB allocation, single plane, and a bpr too small for 1024 BGRA so
        // the device-surface path refuses and the invent tail is what runs.
        st32(&mut desc[DEVICE_DESC_ALLOC_SIZE..], 0x18_0000);
        st64(
            &mut desc[DEVICE_DESC_DIMS..],
            (1024u64 << 8) | (1024u64 << 40),
        );
        st32(&mut desc[DEVICE_DESC_BPR..], 64);
        desc[DEVICE_DESC_PLANE_COUNT] = 0;

        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        assert!(state.attach_mapping_internal(9, KVA));
        assert!(state.set_mapping_device_desc(9, &desc));
        assert!(state.set_mapping_geom(
            9,
            1024,
            1024,
            crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM
        ));
        {
            let m = state.mappings.get_mut(&9).unwrap();
            m.mapped = true;
            // 256 × 16 KiB = 4 MiB, exactly the packed 1024² BGRA span, so a
            // fallback to `sample_window` would compare 4 MiB against 4 MiB and
            // pass.
            m.page_entries = vec![0x33; 256];
        }
        assert!(
            !pages_cover_geom(&state, 9),
            "alloc_size 0x18_0000 cannot hold a 4 MiB window; the page count is not the bound"
        );
    }

    /// 249² Favourites-class tiles fit in 16×16KiB pages; short-table proxy is
    /// desktop dual-mid, not tile size alone.
    #[test]
    fn pages_cover_geom_249_tile_fits_in_sixteen_16k_pages() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        assert!(state.attach_mapping_internal(8, KVA));
        {
            let m = state.mappings.get_mut(&8).unwrap();
            m.mapped = true;
            m.page_entries = vec![0x11; 16];
        }
        assert!(state.set_mapping_geom(
            8,
            249,
            249,
            crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM
        ));
        assert!(
            pages_cover_geom(&state, 8),
            "live Favourites pages=16 is enough for 249² BGRA at 16KiB pages"
        );
    }

    #[test]
    fn render_pages_reject_known_device_and_task_control_pages() {
        let mut state = DeviceState::new(DeviceId(1), crate::model::PAGE_SHIFT_X86);
        state.gfx.root_page = 0x120;
        state.child_rings[2].page_gpas = vec![0x330_000];
        assert!(state.define_task(1, 0x4000_0000, 0x440));
        assert!(state.set_object_list(1, 0x550, 1024));

        assert_eq!(
            first_control_page_collision(&state, &[0x120_000]),
            Some((0x120_000, "gfx_root"))
        );
        assert_eq!(
            first_control_page_collision(&state, &[0x330_000]),
            Some((0x330_000, "child_fifo"))
        );
        assert_eq!(
            first_control_page_collision(&state, &[0x440_000]),
            Some((0x440_000, "task_directory"))
        );
        assert_eq!(
            first_control_page_collision(&state, &[0x551_000]),
            Some((0x551_000, "task_object_list"))
        );
        assert_eq!(first_control_page_collision(&state, &[0x660_000]), None);
    }

    /// The object-list probe is a RANGE query over the surface, and the three
    /// things a range query can get wrong that a per-page enumeration cannot.
    ///
    /// The per-page form walked `first + i*page` for every slot page and returned
    /// the first one the surface held, so it reported the LOWEST colliding page
    /// and stopped exactly at `count`. A `partition_point` that is off by one
    /// entry, or an end bound computed from slots rather than pages, reproduces
    /// the same `Some(_)`/`None` on a single-page surface and diverges here.
    #[test]
    fn object_list_collision_reports_the_lowest_page_and_stops_at_the_span_end() {
        let mut state = DeviceState::new(DeviceId(1), crate::model::PAGE_SHIFT_X86);
        assert!(state.define_task(1, 0x4000_0000, 0x440));
        // 1024 slots x 16 bytes = 16 KiB = pages 0x550..0x554 at a 4 KiB shift.
        assert!(state.set_object_list(1, 0x550, 1024));

        // Several surface pages inside the span, listed out of order: the answer
        // is the lowest, not whichever the surface happens to name first.
        assert_eq!(
            first_control_page_collision(&state, &[0x553_000, 0x551_000, 0x552_000]),
            Some((0x551_000, "task_object_list"))
        );
        // Both ends of the span are inside it.
        assert_eq!(
            first_control_page_collision(&state, &[0x550_000]),
            Some((0x550_000, "task_object_list"))
        );
        assert_eq!(
            first_control_page_collision(&state, &[0x553_000]),
            Some((0x553_000, "task_object_list"))
        );
        // The page immediately after the span is not part of it.
        assert_eq!(first_control_page_collision(&state, &[0x554_000]), None);
        // Nor is the page immediately before.
        assert_eq!(first_control_page_collision(&state, &[0x54f_000]), None);
        // A surface that straddles the span without landing in it stays clean —
        // the query must not report the neighbour it binary-searched past.
        assert_eq!(
            first_control_page_collision(&state, &[0x100_000, 0x554_000]),
            None
        );
    }

    /// Priority order survives the rewrite: a surface colliding with several
    /// control structures at once names the same one it always did. The walk is
    /// per task and interleaved (task 1's object list before task 2's directory),
    /// which a flat "collect every control page then sort" would silently lose.
    #[test]
    fn a_surface_colliding_with_several_control_structures_names_the_first() {
        let mut state = DeviceState::new(DeviceId(1), crate::model::PAGE_SHIFT_X86);
        state.gfx.root_page = 0x120;
        state.gfx.fifo_base_page = 0x220;
        state.iosfc.ring_base = 0x300_000;
        state.child_rings[2].page_gpas = vec![0x330_000];
        assert!(state.define_task(1, 0x4000_0000, 0x440));
        assert!(state.set_object_list(1, 0x550, 1024));
        assert!(state.define_task(2, 0x4000_0000, 0x660));

        let all = [
            0x660_000, 0x551_000, 0x440_000, 0x330_000, 0x300_000, 0x220_000, 0x120_000,
        ];
        assert_eq!(
            first_control_page_collision(&state, &all),
            Some((0x120_000, "gfx_root"))
        );
        state.gfx.root_page = 0;
        assert_eq!(
            first_control_page_collision(&state, &all),
            Some((0x220_000, "root_fifo"))
        );
        state.gfx.fifo_base_page = 0;
        assert_eq!(
            first_control_page_collision(&state, &all),
            Some((0x300_000, "iosfc_ring"))
        );
        state.iosfc.ring_base = 0;
        assert_eq!(
            first_control_page_collision(&state, &all),
            Some((0x330_000, "child_fifo"))
        );
        state.child_rings[2].page_gpas.clear();
        assert_eq!(
            first_control_page_collision(&state, &all),
            Some((0x440_000, "task_directory"))
        );
        // Task 1's object list outranks task 2's directory: the walk is per task.
        state.tasks[1].directory_pfn = 0;
        assert_eq!(
            first_control_page_collision(&state, &all),
            Some((0x551_000, "task_object_list"))
        );
        assert!(state.set_object_list(1, 0, 0));
        assert_eq!(
            first_control_page_collision(&state, &all),
            Some((0x660_000, "task_directory"))
        );
    }
}
