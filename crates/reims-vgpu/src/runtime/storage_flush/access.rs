//! The flush-on-access choke points.
//!
//! Every host-side read or write of intersecting mapping bytes enters here
//! before it touches guest memory, so a deferred window's stale pages are
//! landed before anybody reads them. These four are the whole non-fence entry
//! surface; each decides *which* windows are owed and hands each one to
//! [`super::land`].

use super::land::{flush_gva_one, flush_linear_one, flush_one};
use crate::model::DeviceState;
use crate::runtime::host::{HostMemory, HostOps};

/// Flush every deferred window intersecting `[lo, hi)` on `mapping_id` into
/// guest pages. Returns `false` when any window could not be flushed (the
/// failure is fail-logged; the guest window keeps its stale-but-coherent
/// pre-dispatch bytes).
///
/// Re-entrancy: intersecting entries are removed from the map up front
/// (fixpoint over window unions), so the nested hook fired by the flush's own
/// `write_full_rect_raw_at` finds nothing and recurses no further.
pub fn flush_intersecting<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mapping_id: u32,
    lo: u64,
    hi: u64,
) -> bool {
    if state.compute_deferred_flush.is_empty() {
        return true;
    }
    // A condemned backing is an UNDECIDED window, not a dead one.
    // `condemn_surface_backing` deliberately keeps content state — including
    // these deferred windows — because `DeleteIOSurfaceBacking2` may name a
    // prior incarnation of a recycled id whose slot already carries a live
    // surface with an unflushed paint; `mapper::resolve` settles which by
    // comparing the stashed page fingerprint, and reprieves or drops.
    //
    // Taking the windows here defeats exactly that. The page list is stashed in
    // `condemned_entries`, so the flush cannot write (and must not — the pages
    // may be recycled, the boot-16 PTE-corruption class), and the window is
    // consumed on the way to failing: `flush_intersecting` removes it before
    // `flush_one` runs. The fingerprint decision then has nothing left to
    // reprieve, and the loss is reported as `revalidate_condemned` as though the
    // flush were at fault. Leave the obligation armed for the resolve instead.
    // A second delete with no resolve between still tears down for real
    // (`drop_windows`), and the window cap still bounds the population.
    if state.mapping_backing_condemned(mapping_id) {
        // Latched per mapping. Holding is the *expected* outcome for as long as
        // the condemnation is undecided, and a reader hits this choke point on
        // every access: one boot emitted this 15224 times, 13015 of them for a
        // single mapping that stayed condemned for 121 s, which is 7:1 against
        // every other line in the log put together. That drowns the channel this
        // device is diagnosed through, and the rate was never the signal — which
        // mapping is holding is. A real loss is still reported, by
        // `deferred_flush_lost` if the resolve reprieves and the write then
        // fails, or by `deferred_flush_dropped` if it tears down.
        if crate::observe::first_sight("deferred_flush_held", u64::from(mapping_id)) {
            crate::observe::off(format!(
                "deferred_flush_held mapping={mapping_id} reason=backing_condemned lo={lo} hi={hi} (latched)"
            ));
        }
        return true;
    }
    // Fixpoint: a taken window may extend past [lo, hi) and drag further
    // deferred compute siblings into the flush set.
    let mut pending = state.take_deferred_flush_windows(mapping_id, lo, hi);
    let (mut span_lo, mut span_hi) = (lo, hi);
    loop {
        let new_lo = pending
            .iter()
            .map(|(key, _)| key.surface_offset)
            .fold(span_lo, u64::min);
        let new_hi = pending
            .iter()
            .map(|(key, _)| key.span_end)
            .fold(span_hi, u64::max);
        if new_lo == span_lo && new_hi == span_hi {
            break;
        }
        span_lo = new_lo;
        span_hi = new_hi;
        pending.extend(state.take_deferred_flush_windows(mapping_id, span_lo, span_hi));
    }
    let mut ok = true;
    for (key, owner) in pending {
        ok &= flush_one(state, host, &key, owner);
    }
    ok
}

/// Flush deferred windows whose **physical pages** alias a raw task-GVA span.
///
/// The linear-sample fallback (`load_linear_texture_rgba_host`) reads texture
/// content through task page-table walks that never name a `mapping_id`, so it
/// bypasses every mapping-keyed flush choke point — a sample of a
/// resident-authoritative surface through its GVA alias reads the stale
/// pre-Store bytes (boot-18 `m2v_empty_layer reason=linear_sample` poisoning).
/// Resolve the span's pages, match them against each deferred window's mapping
/// pages, and flush the mappings that hit before the caller reads.
///
/// # Why the per-page walk is unconditional
///
/// The walk visits every page of the span through the task page table, and it
/// only ever runs when at least one deferred window is armed — the `is_empty()`
/// early-outs above return first otherwise. Since the four fence bindings of
/// 2026-07-31 collapsed a deferred window's lifetime to a single submission
/// (mean arm-to-flush age ~351 µs), "at least one window armed" is close to
/// never, so the walk is close to never.
///
/// A per-bind no-intersection memo used to sit here, keyed by `(task, gva,
/// span)` and skipping the walk while the deferred-window signature was
/// unchanged. Its justification was a 78 % skip rate over 408 000 calls; that
/// figure predated the fence bindings. Censusing the memo's three outcomes on a
/// driven x86/Vulkan boot — macOS desktop, 25 s of Safari window compositing
/// (2 727 pointer events at 111 Hz, drain duty 0.97, 499 draws/s), 70
/// `store_routes` lines — read `walk = 1`, `skip` never emitted, `recheck`
/// never emitted. The memo answered nothing because the early-outs answered
/// first, so it and its 1-in-64 sampled self-heal are gone. Do not re-derive
/// it: this walk is not a cost on this rail, and caching it reintroduces a
/// hole that only a sampled walk can close.
///
/// That reading is now standing rather than historical, which is the one thing
/// it was missing: it came from the memo's own counters, and deleting the memo
/// deleted them, so nothing could say whether it still held. The
/// `gva_alias_probe_*` fields on the `store_routes` line replace them and are
/// not tied to a heuristic that can be removed. A driven x86/Vulkan boot —
/// Safari window drag, drain duty 0.90, 4 972 draws/s — reads **705 988 quiet
/// calls and one walk** over 43 census windows, that walk costing 12 µs over
/// 256 pages, with `gva_alias_hit_page` never firing.
///
/// So the conclusion is unchanged and is now cheap to re-check: divide
/// `gva_alias_probe_us` by `gva_alias_probe_walked`, and read the walked count
/// against `gva_alias_probe_quiet`. A walked count that climbs toward the quiet
/// one means a rail is staying armed across binds, which puts an `O(pages)`
/// guest page-table walk on the draw path — and that, not the hit rate, is the
/// event worth alarming on.
pub fn flush_intersecting_task_gva<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    gva: u64,
    span: u64,
) {
    if span == 0
        || (state.deferred_alias_pages.is_empty()
            && state.linear_deferred_flush.is_empty()
            && state.gva_deferred_flush.is_empty())
    {
        // The whole call, walk included, costs nothing when no rail is armed.
        // Counted apart from the walking arm because the two are what decide
        // whether this probe is on the draw path's bill at all: every zero-copy
        // bind makes this call, so a quiet:walked ratio near 1 means the early
        // return is doing the work and a ratio near 0 means it is not.
        crate::runtime::drain::note_store_route("gva_alias_probe_quiet");
        return;
    }
    // Fast exact-window path: a sample of the deferred GVA surface itself
    // names the same base GVA — no page walk needed to detect it.
    if state.gva_deferred_flush.contains_key(&gva) {
        flush_gva_exact(state, host, gva, true, "gva_alias");
    }
    if state.deferred_alias_pages.is_empty()
        && state.linear_deferred_flush.is_empty()
        && state.gva_deferred_flush.is_empty()
    {
        crate::runtime::drain::note_store_route("gva_alias_probe_quiet");
        return;
    }
    // The arm that walks. Timed and counted because it is `O(pages)` guest
    // page-table reads on a call every zero-copy bind makes, and because the
    // caller immediately walks the *same* range again to build its run window —
    // so what this costs is also what collapsing the two would return.
    // `gva_alias_hit_page` below says how often the walk finds anything.
    crate::runtime::drain::note_store_route("gva_alias_probe_walked");
    let probe_started = std::time::Instant::now();
    let page = state.page_size();
    let n_pages = crate::runtime::gva_mem::pages_spanned(gva, span, page);
    let mut hits: Vec<u32> = Vec::new();
    let mut linear_hits: Vec<(crate::model::ComputeStorageResidencyKey, u32)> = Vec::new();
    let mut gva_hits: Vec<u64> = Vec::new();
    // Pages the walk actually resolved, which is not `n_pages`: the visitor
    // stops early once every armed window has been hit, and an unresolvable
    // page is skipped rather than visited. The requested reach is `n_pages`
    // beside it, so a reader can see which of the two the cost tracks.
    let mut walked_pages: u64 = 0;
    {
        let index = &state.deferred_alias_pages;
        let linear_index = &state.linear_deferred_flush;
        let gva_index = &state.gva_deferred_flush;
        let total = index.len() + linear_index.len() + gva_index.len();
        crate::runtime::gva_mem::visit_task_gva_page_gpas(
            host,
            &state.tasks,
            task_id,
            gva,
            span,
            state.page_shift,
            1,
            &mut |gpa_page| {
                walked_pages += 1;
                for (&mid, pages) in index.iter() {
                    if pages.contains(&gpa_page) && !hits.contains(&mid) {
                        hits.push(mid);
                    }
                }
                for (key, entry) in linear_index.iter() {
                    if entry.pages.contains(&gpa_page) && !linear_hits.iter().any(|(k, _)| k == key)
                    {
                        linear_hits.push((*key, entry.generation));
                    }
                }
                for (&window_gva, entry) in gva_index.iter() {
                    if entry.pages.contains(&gpa_page) && !gva_hits.contains(&window_gva) {
                        gva_hits.push(window_gva);
                    }
                }
                hits.len() + linear_hits.len() + gva_hits.len() < total
            },
        );
    }
    crate::runtime::drain::note_store_route_n("gva_alias_probe_pages", walked_pages);
    crate::runtime::drain::note_store_route_n("gva_alias_probe_reach", n_pages);
    crate::runtime::drain::note_store_route_us(
        "gva_alias_probe_us",
        probe_started.elapsed().as_micros() as u64,
    );
    let hit_ct = (hits.len() + linear_hits.len() + gva_hits.len()) as u64;
    if hit_ct == 0 {
        return;
    }
    // Always-on: a hit-producing walk is rare (six in a whole repro boot), so
    // there is no flood risk and nothing to sample.
    crate::observe::fail(format!(
        "gva_alias_hit_page task={task_id} gva={gva:#x} span={span} \
         n_pages={n_pages} hits={hit_ct}"
    ));
    for mid in hits {
        crate::observe::off(format!(
            "deferred_flush_gva_alias mid={mid} task={task_id} gva={gva:#x} span={span}"
        ));
        let _ = flush_intersecting(state, host, mid, 0, u64::MAX);
    }
    for (key, generation) in linear_hits {
        crate::observe::off(format!(
            "deferred_flush_gva_alias kind=linear task={} ref={} gva={gva:#x} span={span}",
            key.map_generation, key.texture_ref
        ));
        let _ = flush_linear_one(state, host, &key, generation);
    }
    for window_gva in gva_hits {
        crate::observe::off(format!(
            "deferred_flush_gva_alias kind=gva window={window_gva:#x} task={task_id} gva={gva:#x} span={span}"
        ));
        flush_gva_exact(state, host, window_gva, true, "gva_alias");
    }
}

/// Land every deferred window the guest is about to CPU-read on `mapping_id`.
///
/// SynchronizeResources (child op 0x35, `synchronizeForUnwire`) is the guest's
/// declaration that it will read/pageoff this resource's pages with the CPU —
/// the one host-visible choke point for guest CPU reads, which no device-side
/// flush hook can see (boot-24/25 black-wallpaper class: the fade snapshot is
/// guest-CPU-composited from device-rendered windows whose writebacks were
/// deferred). Mapping-keyed compute windows flush via [`flush_intersecting`];
/// linear task-GVA windows never name a mapping, so they flush when their
/// defer-time page index aliases the mapping's physical pages. Returns
/// `(all_ok, windows_flushed)`.
///
/// # What its counters can and cannot answer
///
/// `guest_read_declared` counts every call, `guest_read_landed` every window one
/// lands, and `guest_read_dry` the calls that found nothing armed. They exist
/// because this is the only place the guest tells us it is about to read, and
/// until now nothing counted it — so "how often does the guest declare?" had no
/// answer, and the eager fence-bound writeback
/// ([`flush_mapping_windows_before_fence`](crate::runtime::storage_flush::fence::flush_mapping_windows_before_fence)) was being weighed against an unknown.
///
/// Read them with the order of events in mind. The fence flush runs first and
/// empties the windows, so `guest_read_dry` dominating is the *expected* reading
/// and does **not** show the declaration would have been too late — it shows the
/// fence got there first, which it always does. What the pair does bound is the
/// declaration *rate*: `guest_read_declared` against the composite rate says
/// whether the guest declares once per frame it reads, rarely, or never. A rate
/// far below the flush rate means most flushes land for nobody, which is the
/// case the demand-driven route is for; a rate close to it means the eager rail
/// is doing work the guest would have asked for anyway.
pub fn flush_mapping_for_guest_read<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mapping_id: u32,
) -> (bool, u32) {
    crate::runtime::drain::note_store_route("guest_read_declared");
    // Does the declaration name a surface the eager rail actually writes back?
    // `guest_read_dry` cannot say — the fence always empties the windows first,
    // so every declaration is dry either way. This split can, and it is the
    // number that decides whether the writeback could be demand-driven at all.
    crate::runtime::drain::note_store_route(
        if state.fence_flushed_mappings.contains(&mapping_id) {
            "guest_read_on_flushed_mid"
        } else {
            "guest_read_on_other_mid"
        },
    );
    let keyed = state
        .compute_deferred_flush
        .keys()
        .filter(|k| k.mapping_id == mapping_id)
        .count();
    let mut ok = true;
    let mut flushed = keyed as u32;
    if keyed > 0 {
        ok &= flush_intersecting(state, host, mapping_id, 0, u64::MAX);
    }
    if !state.linear_deferred_flush.is_empty() || !state.gva_deferred_flush.is_empty() {
        let page = state.page_size();
        let page_shift = state.page_shift;
        if let Some(m) = state.mappings.get(&mapping_id) {
            let pages: std::collections::HashSet<u64> = m
                .page_entries
                .iter()
                .filter_map(|&e| crate::contract::iosurface_pages::entry_gpa_shift(e, page_shift))
                .map(|gpa| gpa & !(page - 1))
                .collect();
            if !pages.is_empty() {
                let hits: Vec<(crate::model::ComputeStorageResidencyKey, u32)> = state
                    .linear_deferred_flush
                    .iter()
                    .filter(|(_, entry)| !entry.pages.is_disjoint(&pages))
                    .map(|(key, entry)| (*key, entry.generation))
                    .collect();
                for (key, generation) in hits {
                    ok &= flush_linear_one(state, host, &key, generation);
                    flushed = flushed.saturating_add(1);
                }
                let gva_hits: Vec<u64> = state
                    .gva_deferred_flush
                    .iter()
                    .filter(|(_, entry)| !entry.pages.is_disjoint(&pages))
                    .map(|(&gva, _)| gva)
                    .collect();
                for gva in gva_hits {
                    ok &= flush_gva_exact(state, host, gva, true, "guest_read");
                    flushed = flushed.saturating_add(1);
                }
            }
        }
    }
    if flushed == 0 {
        crate::runtime::drain::note_store_route("guest_read_dry");
    } else {
        crate::runtime::drain::note_store_route_n("guest_read_landed", u64::from(flushed));
    }
    (ok, flushed)
}

/// Land the deferred GVA render-Store window at exactly `gva`, if armed.
///
/// `guest_write` selects the full landing (guest pages + host caches; the
/// task's PTEs must still be live) vs cache-only (unmap/remap/teardown — the
/// map-notify PTE-corruption class forbids guest writes there; the encode
/// cache alone preserves the wallpaper-retain contract). Returns `true` when
/// nothing was armed or the flush landed.
pub fn flush_gva_exact<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    gva: u64,
    guest_write: bool,
    trigger: &str,
) -> bool {
    let Some(entry) = state.take_gva_deferred_window(gva) else {
        return true;
    };
    flush_gva_one(state, host, gva, &entry, guest_write, trigger)
}
