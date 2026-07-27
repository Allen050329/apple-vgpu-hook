//! Deferred compute-writeback flush (flush-on-access).
//!
//! A resident-backed type-11 compute storage output may skip both the engine
//! readback and the CPU guest writeback on the stamp path
//! (`ComputeStorageImageResource::defer_readback`): the pinned engine resident
//! is the authoritative content and the guest window is stale. Every host-side
//! read or write of intersecting mapping bytes calls [`flush_intersecting`]
//! first; the flush copies the resident to the host once
//! (`engine::read_resident_storage`, which also unpins) and lands it in the
//! guest window, then re-establishes the residency mirror so chained seed
//! skips keep working.
//!
//! Guest CPU accesses that never cross our host paths cannot be intercepted —
//! the same accepted exposure as resident render targets under
//! `skip_readback`. Choke points: `mapping_write` read/write entries,
//! `mapper::read/write_mapping_bytes`, and the drain unmap/ReplacePhysical
//! sites (which drop-with-fail instead of writing through recycled pages).

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
    if state.compute_deferred_flush.is_empty() && state.render_deferred_flush.is_empty() {
        return true;
    }
    // Fixpoint: a taken window may extend past [lo, hi) and drag further
    // deferred siblings (compute or render) into the flush set.
    let mut pending = state.take_deferred_flush_windows(mapping_id, lo, hi);
    let mut render_pending = state.take_render_deferred_windows(mapping_id, lo, hi);
    let (mut span_lo, mut span_hi) = (lo, hi);
    loop {
        let new_lo = pending
            .iter()
            .map(|(key, _)| key.surface_offset)
            .chain(render_pending.iter().map(|(key, _)| key.surface_offset))
            .fold(span_lo, u64::min);
        let new_hi = pending
            .iter()
            .map(|(key, _)| key.span_end)
            .chain(render_pending.iter().map(|(key, _)| key.span_end))
            .fold(span_hi, u64::max);
        if new_lo == span_lo && new_hi == span_hi {
            break;
        }
        span_lo = new_lo;
        span_hi = new_hi;
        pending.extend(state.take_deferred_flush_windows(mapping_id, span_lo, span_hi));
        render_pending.extend(state.take_render_deferred_windows(mapping_id, span_lo, span_hi));
    }
    let mut ok = true;
    for (key, generation) in pending {
        ok &= flush_one(state, host, &key, generation);
    }
    for (key, entry) in render_pending {
        ok &= render_flush_one(state, host, &key, &entry);
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
/// Spans up to this many pages probe every page; larger spans probe sparsely.
/// Bound for the per-bind no-intersection memo (`DeviceState::flush_nohit_memo`).
/// The memo clears on every deferred-signature change, so it only holds the
/// distinct `(task,gva,span)` binds seen at one signature — dozens in practice.
/// The cap is a runaway guard: overflow just re-walks (correct, slower).
const FLUSH_NOHIT_MEMO_CAP: usize = 4096;

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
        return;
    }
    // Fast exact-window path: a sample of the deferred GVA surface itself
    // names the same base GVA — no page walk needed to detect it. The flush
    // itself is a synchronous engine read_target (GPU readback) — time it so a
    // rare-but-expensive stall cannot hide inside the caller's zc_flush_ns.
    if state.gva_deferred_flush.contains_key(&gva) {
        let t_exact = std::time::Instant::now();
        flush_gva_exact(state, host, gva, true, "gva_alias");
        state.tranche.zc_flush_exact_ns = state
            .tranche
            .zc_flush_exact_ns
            .saturating_add(t_exact.elapsed().as_nanos() as u64);
    }
    if state.deferred_alias_pages.is_empty()
        && state.linear_deferred_flush.is_empty()
        && state.gva_deferred_flush.is_empty()
    {
        return;
    }
    // No-intersection memo. The per-page task-PT walk below only finds work when
    // a bound buffer span aliases a live deferred window — measured
    // `zc_flush_hits == 0` over ~59k draws of pure compositing, i.e. it is
    // detection overhead. Cache each full-walked bind's resolved gpa pages; skip
    // the walk while the deferred signature is unchanged, and on a signature
    // change re-check the cached pages against the current windows WITHOUT a PT
    // walk (the per-page FFI translate is the expensive part). A 1-in-64 sampled
    // full walk (`flush_verify_ctr`) self-heals a missed task-PT remap
    // (`zc_flush_stale`, must stay 0).
    let t_sig = std::time::Instant::now();
    let sig = state.deferred_flush_signature();
    state.tranche.zc_flush_sig_ns = state
        .tranche
        .zc_flush_sig_ns
        .saturating_add(t_sig.elapsed().as_nanos() as u64);
    let memo_key = (task_id, gva, span);
    let mut sampled_verify = false;
    // On the sampled-verify path, what the cheap page recheck concluded — the
    // full walk below is ground truth we compare it against for staleness.
    let mut verify_cheap_hit = false;
    if let Some((vsig, pages)) = state.flush_nohit_memo.get(&memo_key) {
        let vsig = *vsig;
        state.flush_verify_ctr = state.flush_verify_ctr.wrapping_add(1);
        sampled_verify = state.flush_verify_ctr.is_multiple_of(64);
        if sampled_verify {
            let pages = pages.clone();
            // Self-heal the incremental deferred-page-refs index against truth on
            // the same 1-in-64 cadence: a missed arm/disarm site would otherwise
            // let the fast reject skip a live window. Rebuild + fail-log on drift
            // so a coverage hole is visible and corrected, never silent.
            if state.verify_and_heal_deferred_refs() {
                crate::observe::fail(format!(
                    "deferred_ref_drift task={task_id} gva={gva:#x} span={span} healed_pages={}",
                    state.deferred_page_refs.len()
                ));
            }
            verify_cheap_hit = state.deferred_pages_intersect(&pages);
            // fall through to the full PT walk (ground truth / self-heal)
        } else if vsig == sig {
            // Deferred set unchanged since the walk — still non-intersecting.
            state.tranche.zc_flush_skip = state.tranche.zc_flush_skip.saturating_add(1);
            return;
        } else {
            // Signature changed: re-check the cached pages against the current
            // windows without a PT walk.
            let pages = pages.clone();
            state.tranche.zc_flush_recheck = state.tranche.zc_flush_recheck.saturating_add(1);
            let t_isect = std::time::Instant::now();
            let isect = state.deferred_pages_intersect(&pages);
            state.tranche.zc_flush_isect_ns = state
                .tranche
                .zc_flush_isect_ns
                .saturating_add(t_isect.elapsed().as_nanos() as u64);
            if !isect {
                state.flush_nohit_memo.insert(memo_key, (sig, pages));
                state.tranche.zc_flush_skip = state.tranche.zc_flush_skip.saturating_add(1);
                return;
            }
            // A window now covers this bind — drop the entry and full-walk to
            // flush it below.
            state.flush_nohit_memo.remove(&memo_key);
        }
    }
    let page = state.page_size();
    let n_pages = ((gva % page) + span).div_ceil(page);
    let mut hits: Vec<u32> = Vec::new();
    let mut linear_hits: Vec<(crate::model::ComputeStorageResidencyKey, u32)> = Vec::new();
    let mut gva_hits: Vec<u64> = Vec::new();
    // Resolved gpa pages visited by the walk. Complete only on a no-hit walk
    // (the visitor early-exits once every window is hit), which is exactly when
    // it is cached.
    let mut visited_pages: Vec<u64> = Vec::new();
    // Which page produced the first hit. The walk is complete, so this is the
    // true first overlapping page rather than the next sample point after it.
    let mut first_hit_ordinal: Option<u64> = None;
    state.tranche.zc_flush_walk = state.tranche.zc_flush_walk.saturating_add(1);
    let t_walk = std::time::Instant::now();
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
                visited_pages.push(gpa_page);
                for (&mid, pages) in index.iter() {
                    if pages.contains(&gpa_page) && !hits.contains(&mid) {
                        hits.push(mid);
                    }
                }
                for (key, (generation, pages)) in linear_index.iter() {
                    if pages.contains(&gpa_page) && !linear_hits.iter().any(|(k, _)| k == key) {
                        linear_hits.push((*key, *generation));
                    }
                }
                for (&window_gva, entry) in gva_index.iter() {
                    if entry.pages.contains(&gpa_page) && !gva_hits.contains(&window_gva) {
                        gva_hits.push(window_gva);
                    }
                }
                if first_hit_ordinal.is_none()
                    && hits.len() + linear_hits.len() + gva_hits.len() > 0
                {
                    // `visited_pages` was pushed at the top of this call, so the
                    // ordinal of the page that just hit is len-1.
                    first_hit_ordinal = Some(visited_pages.len().saturating_sub(1) as u64);
                }
                hits.len() + linear_hits.len() + gva_hits.len() < total
            },
        );
    }
    state.tranche.zc_flush_walk_ns = state
        .tranche
        .zc_flush_walk_ns
        .saturating_add(t_walk.elapsed().as_nanos() as u64);
    let hit_ct = (hits.len() + linear_hits.len() + gva_hits.len()) as u64;
    if hit_ct == 0 {
        // Non-intersecting. Cache the resolved pages for the cheap re-check on a
        // later signature change. The walk visits every page, so the set is
        // complete and safe to re-check; that completeness is the whole reason
        // the memo is sound.
        if state.flush_nohit_memo.len() < FLUSH_NOHIT_MEMO_CAP {
            state
                .flush_nohit_memo
                .insert(memo_key, (sig, visited_pages));
        }
        if sampled_verify {
            state.tranche.zc_flush_skip = state.tranche.zc_flush_skip.saturating_add(1);
        }
        return;
    }
    if sampled_verify {
        // The full walk found work. If the cheap recheck had already flagged it,
        // the fast path is sound (a real, freshly-armed intersection). If the
        // cheap recheck said "clear", the cached pages were stale (a missed
        // task-PT remap) — fail-log and count it.
        if !verify_cheap_hit {
            state.tranche.zc_flush_stale = state.tranche.zc_flush_stale.saturating_add(1);
            crate::observe::fail(format!(
                "zc_flush_stale task={task_id} gva={gva:#x} span={span} hits={hit_ct}"
            ));
        }
        state.flush_nohit_memo.remove(&memo_key);
    }
    // Always-on: a hit-producing walk is rare (six in a whole repro boot; the
    // measurement this path's memo was built on saw zero over ~59k compositing
    // draws), so there is no flood risk and nothing to sample.
    crate::observe::fail(format!(
        "gva_alias_hit_page task={task_id} gva={gva:#x} span={span} first_hit_page={} \
         n_pages={n_pages} hits={hit_ct}",
        first_hit_ordinal.map_or(-1i64, |o| o as i64)
    ));
    state.tranche.zc_flush_hits = state.tranche.zc_flush_hits.saturating_add(hit_ct);
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
/// deferred). Render + compute windows on the mapping flush via
/// [`flush_intersecting`]; linear task-GVA windows never name a mapping, so
/// they flush when their defer-time page index aliases the mapping's physical
/// pages. Returns `(all_ok, windows_flushed)`.
pub fn flush_mapping_for_guest_read<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mapping_id: u32,
) -> (bool, u32) {
    let keyed = state
        .compute_deferred_flush
        .keys()
        .filter(|k| k.mapping_id == mapping_id)
        .count()
        + state
            .render_deferred_flush
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
                    .filter(|(_, (_, window_pages))| !window_pages.is_disjoint(&pages))
                    .map(|(key, (generation, _))| (*key, *generation))
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

/// Does this window's GVA still resolve to the pages it was armed with?
///
/// `entry.pages` is the whole point of the page-alias trigger: a new guest write
/// is matched against it to decide the window must land first, so that two
/// writers to the same guest memory are ordered. It was recorded when the window
/// was armed, and nothing re-checks it. If the guest has since re-pointed
/// `[gva, gva+span)` at different pages, then the alias matched pages this window
/// no longer owns *and* the write that follows lands in whatever owns `gva` now —
/// the stale-view class, with our own bookkeeping as the stale part.
///
/// §8.53/§8.54 measured only the case where the guest zeroed the PTEs, which is
/// caught by [`crate::runtime::host::MemError::is_guest_teardown`]. Whether a
/// window's pages can move while still resolving has never been measured, and a
/// guard for an unmeasured hazard is a guess. So this reads and reports; it does
/// not decide. A silent counter would be worth nothing, and this cannot flood:
/// only the alias/read triggers reach it, ten times in a repro boot.
#[cfg(feature = "backend-vulkan")]
fn report_window_page_drift<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    gva: u64,
    entry: &crate::model::GvaDeferredEntry,
    trigger: &str,
) {
    let span = entry.span();
    if span == 0 || entry.pages.is_empty() {
        return;
    }
    let mut live = std::collections::HashSet::new();
    crate::runtime::gva_mem::visit_task_gva_page_gpas(
        host,
        &state.tasks,
        entry.task_id,
        gva,
        span,
        state.page_shift,
        1,
        &mut |gpa_page| {
            live.insert(gpa_page);
            true
        },
    );
    // An empty or short walk is the teardown case the writer below names
    // precisely; reporting it here too would double-count it under a reason that
    // does not fit.
    if live.len() < entry.pages.len() {
        return;
    }
    if live == entry.pages {
        return;
    }
    crate::observe::fail(format!(
        "deferred_window_page_drift gva={gva:#x} task={} {}x{} trigger={trigger} \
         armed_pages={} live_pages={} moved={}",
        entry.task_id,
        entry.width,
        entry.height,
        entry.pages.len(),
        live.len(),
        entry.pages.difference(&live).count()
    ));
}

/// Land a taken deferred GVA render-Store window: engine resident target →
/// guest pages (when `guest_write` and the span is still map-covered) +
/// `host_gva_surfaces`/texture encode caches (always). Unpins the resident
/// either way; a lost resident is fail-visible and leaves the guest window
/// stale-but-coherent (pre-Store bytes).
#[cfg(feature = "backend-vulkan")]
pub fn flush_gva_one<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    gva: u64,
    entry: &crate::model::GvaDeferredEntry,
    guest_write: bool,
    trigger: &str,
) -> bool {
    use crate::backend::vulkan::engine::TargetIdentity;
    let started = std::time::Instant::now();
    let identity = TargetIdentity::Gva {
        gva,
        width: entry.width,
        height: entry.height,
        generation: 0,
    };
    let rgba = match crate::backend::vulkan::engine::read_target(&identity) {
        Ok(px) => px,
        Err(e) => {
            crate::backend::vulkan::engine::unpin_resident_target(&identity);
            crate::observe::fail(format!(
                "deferred_flush_lost kind=gva gva={gva:#x} {}x{} fmt={:#x} trigger={trigger} err={e}",
                entry.width, entry.height, entry.format
            ));
            return false;
        }
    };
    crate::backend::vulkan::engine::unpin_resident_target(&identity);
    let mut guest = "skip";
    if guest_write && state.gva_write_allowed(entry.task_id, gva, entry.span()) {
        report_window_page_drift(state, host, gva, entry, trigger);
        guest = match crate::runtime::metal_draw::write_gva_rgba8(
            state,
            host,
            entry.task_id,
            gva,
            entry.width,
            entry.height,
            entry.row_stride,
            entry.format,
            &rgba,
        ) {
            Ok(()) => "written",
            // The guest already tore this window down and its Unmap notify has
            // not drained yet. That is the same state the Unmap/Map notify path
            // lands cache-only for — "on Unmap the PTEs are already gone" — just
            // reached through a different door, because a page-alias flush races
            // ahead of the notify. The caches below hold the content, so the
            // obligation is discharged and nothing is lost. Expected control
            // flow: it does not belong in the failure log.
            Err(err) if err.is_guest_teardown() => "unmapped",
            // A write that refused while the target still existed. The caches
            // below keep the authoritative bytes, so guest RAM is stale rather
            // than wrong — but this one is a real loss of guest work.
            Err(err) => {
                crate::observe::Emit::decline("deferred_flush_lost", &err)
                    .field("kind", "gva")
                    .field("gva", format!("{gva:#x}"))
                    .field("dims", format!("{}x{}", entry.width, entry.height))
                    .field("bpr", entry.row_stride)
                    .field("fmt", format!("{:#x}", entry.format))
                    .field("trigger", trigger)
                    .fail();
                "write_fail"
            }
        };
    } else if guest_write {
        guest = "skip_uncovered";
    }
    crate::runtime::metal_draw::host_cache_store_gva_layer(
        state,
        entry.task_id,
        entry.texture_ref,
        entry.producer_object_type,
        gva,
        entry.width,
        entry.height,
        &rgba,
    );
    crate::observe::off(format!(
        "gva_deferred_flush gva={gva:#x} {}x{} fmt={:#x} guest={guest} trigger={trigger} bytes={} us={}",
        entry.width,
        entry.height,
        entry.format,
        rgba.len(),
        started.elapsed().as_micros()
    ));
    guest != "write_fail"
}

#[cfg(not(feature = "backend-vulkan"))]
pub fn flush_gva_one<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    _host: &mut M,
    gva: u64,
    entry: &crate::model::GvaDeferredEntry,
    _guest_write: bool,
    trigger: &str,
) -> bool {
    // No engine ⇒ nothing can have deferred; drop the obligation fail-visibly.
    let _ = state;
    crate::observe::fail(format!(
        "deferred_flush_lost kind=gva reason=no_backend gva={gva:#x} {}x{} trigger={trigger}",
        entry.width, entry.height
    ));
    false
}

/// Land GVA windows whose task died (`DeviceState::retired_gva_windows`)
/// **cache-only**: the GVA walk is gone with the task, so guest pages are
/// never written from teardown (boot-16 rule); the encode cache keeps the
/// content for later samples (wallpaper-retain contract).
pub fn retire_gva_windows<M: HostMemory + HostOps>(state: &mut DeviceState, host: &mut M) {
    if state.retired_gva_windows.is_empty() {
        return;
    }
    let retired = std::mem::take(&mut state.retired_gva_windows);
    for (gva, entry) in &retired {
        let _ = flush_gva_one(state, host, *gva, entry, false, "task_retired");
    }
}

/// Land a deferred linear window: resident → cache entry bytes
/// (`materialize_linear_resident`) → guest pages when the span is still
/// GVA-covered (fresh page-table walks; a write through changed PTEs fails
/// per-row, fail-visibly, and never touches other memory). Drops the
/// obligation either way — the cache entry keeps the authoritative bytes.
#[cfg(feature = "backend-vulkan")]
pub fn flush_linear_one<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    key: &crate::model::ComputeStorageResidencyKey,
    generation: u32,
) -> bool {
    state.disarm_linear_deferred_window(key);
    let task_id = key.map_generation;
    let texture_ref = key.texture_ref;
    let started = std::time::Instant::now();
    let (bytes, texel) =
        match crate::backend::vulkan::engine::read_resident_storage(key, generation) {
            Ok(v) => v,
            Err(e) => {
                crate::observe::Emit::decline("deferred_flush_lost", &e)
                    .field("kind", "linear")
                    .field("task", task_id)
                    .field("ref", texture_ref)
                    .field("geom", format!("{}x{}", key.width, key.height))
                    .field("fmt", format!("{:#x}", key.pixel_format))
                    .field("gen", generation)
                    .fail();
                if let Some(entry) = state.host_linear_textures.get_mut(&(task_id, texture_ref)) {
                    if entry.resident_gen == generation {
                        entry.resident_gen = 0;
                    }
                }
                return false;
            }
        };
    crate::runtime::surface_cache::materialize_linear_resident(
        state,
        task_id,
        texture_ref,
        generation,
        &bytes,
    );
    let tight = (key.width as usize).saturating_mul(texel as usize);
    let mut guest = "skip_uncovered";
    if state.gva_write_allowed(task_id, key.surface_offset, key.span_end) {
        guest = if crate::runtime::compute_exec::write_linear_guest(
            state,
            host,
            task_id,
            key.surface_offset,
            key.surface_bpr as u64,
            tight,
            key.height,
            &bytes,
            &format!("flush ref={texture_ref}"),
        ) {
            "written"
        } else {
            // The per-row failure is already fail-logged; the cache entry
            // keeps the coherent authoritative bytes.
            "write_fail"
        };
    }
    crate::observe::off(format!(
        "linear_deferred_flush task={task_id} ref={texture_ref} {}x{} fmt={:#x} gen={generation} guest={guest} bytes={} us={}",
        key.width,
        key.height,
        key.pixel_format,
        bytes.len(),
        started.elapsed().as_micros()
    ));
    true
}

#[cfg(not(feature = "backend-vulkan"))]
pub fn flush_linear_one<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    _host: &mut M,
    key: &crate::model::ComputeStorageResidencyKey,
    generation: u32,
) -> bool {
    // No engine ⇒ nothing can have deferred; drop the obligation fail-visibly.
    state.disarm_linear_deferred_window(key);
    crate::observe::fail(format!(
        "deferred_flush_lost kind=linear reason=no_backend task={} ref={} gen={generation}",
        key.map_generation, key.texture_ref
    ));
    false
}

/// Unpin engine residents whose linear cache entry died (task/object delete —
/// `DeviceState::retired_linear_residents`). The images become LRU-evictable;
/// without this a dead entry leaks its pinned VRAM image for the boot.
pub fn retire_linear_residents(state: &mut DeviceState) {
    if state.retired_linear_residents.is_empty() {
        return;
    }
    let retired = std::mem::take(&mut state.retired_linear_residents);
    for key in &retired {
        // Task teardown = the GPU VA maps are gone; never write guest pages
        // from here (boot-16 rule) — drop any pending guest-flush obligation.
        if state.disarm_linear_deferred_window(key) {
            crate::observe::off(format!(
                "linear_deferred_dropped reason=retired task={} ref={}",
                key.map_generation, key.texture_ref
            ));
        }
        #[cfg(feature = "backend-vulkan")]
        {
            crate::backend::vulkan::engine::unpin_resident_storage(key);
            crate::observe::off(format!(
                "linear_resident_retired task={} ref={} gva={:#x} {}x{} fmt={:#x}",
                key.map_generation,
                key.texture_ref,
                key.surface_offset,
                key.width,
                key.height,
                key.pixel_format
            ));
        }
    }
}

#[cfg(feature = "backend-vulkan")]
fn flush_one<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    key: &crate::model::ComputeStorageResidencyKey,
    generation: u32,
) -> bool {
    let started = std::time::Instant::now();
    // Same recycled-pages guard as the render flush: a surface window whose
    // defer-time map_generation no longer matches must not write through the
    // rewired pages.
    let current = state
        .mappings
        .get(&key.mapping_id)
        .map(|m| m.map_generation);
    if current != Some(key.map_generation) {
        crate::backend::vulkan::engine::unpin_resident_storage(key);
        crate::observe::fail(format!(
            "deferred_flush_lost mapping={} {}x{} fmt={:#x} gen={generation} reason=map_generation_drift current={current:?}",
            key.mapping_id, key.width, key.height, key.pixel_format
        ));
        return false;
    }
    let (bytes, texel) =
        match crate::backend::vulkan::engine::read_resident_storage(key, generation) {
            Ok(v) => v,
            Err(e) => {
                // The pinned resident vanished (device loss, guest reset,
                // same-identity key change). The window keeps its coherent
                // pre-dispatch bytes; name the loss.
                crate::observe::Emit::decline("deferred_flush_lost", &e)
                    .field("mapping", key.mapping_id)
                    .field("geom", format!("{}x{}", key.width, key.height))
                    .field("fmt", format!("{:#x}", key.pixel_format))
                    .field("gen", generation)
                    .fail();
                return false;
            }
        };
    let expected_bpp = crate::contract::pixel_format::bytes_per_pixel(key.pixel_format);
    if expected_bpp != Some(texel) {
        crate::observe::fail(format!(
            "deferred_flush_lost mapping={} reason=texel_mismatch engine={texel} guest={expected_bpp:?} fmt={:#x}",
            key.mapping_id, key.pixel_format
        ));
        return false;
    }
    let tight = key.width.saturating_mul(texel);
    if !crate::runtime::mapping_write::write_full_rect_raw_at(
        state,
        host,
        key.mapping_id,
        key.surface_offset,
        key.surface_bpr,
        key.span_end,
        key.width,
        key.height,
        texel,
        &bytes,
        tight,
    ) {
        crate::observe::fail(format!(
            "deferred_flush_lost mapping={} reason=guest_write {}x{} off={} bpr={} span_end={}",
            key.mapping_id,
            key.width,
            key.height,
            key.surface_offset,
            key.surface_bpr,
            key.span_end
        ));
        return false;
    }
    // Guest pages now hold exactly the resident content at `generation`:
    // re-establish the mirror entry the write's own invalidation dropped so
    // chained seed skips stay live.
    state.compute_storage_residency.insert(*key, generation);
    crate::observe::off(format!(
        "compute_deferred_flush mapping={} {}x{} fmt={:#x} gen={generation} bytes={} us={}",
        key.mapping_id,
        key.width,
        key.height,
        key.pixel_format,
        bytes.len(),
        started.elapsed().as_micros()
    ));
    true
}

#[cfg(not(feature = "backend-vulkan"))]
fn flush_one<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    _host: &mut M,
    key: &crate::model::ComputeStorageResidencyKey,
    generation: u32,
) -> bool {
    let _ = state;
    crate::observe::fail(format!(
        "deferred_flush_lost mapping={} reason=no_backend gen={generation}",
        key.mapping_id
    ));
    false
}

/// Flush one deferred render-Store window by replaying the import-present
/// Store it stood in for. The engine resident target (pinned at defer time)
/// DMAs straight into guest pages; every failure is the exact fail line the
/// synchronous Store would have emitted, plus a named `deferred_flush_lost`.
#[cfg(feature = "backend-vulkan")]
pub(crate) fn render_flush_one<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    key: &crate::model::RenderDeferredKey,
    entry: &crate::model::RenderDeferredEntry,
) -> bool {
    let started = std::time::Instant::now();
    let identity = crate::runtime::import_present::render_deferred_identity(key.mapping_id, entry);
    // Recycled-pages drop guard for EVERY window: the mapping's page table at
    // defer time must still be current or the flush would DMA old content
    // into rewired pages (grouped windows carry the group identity with no
    // member lifetime inside; ungrouped windows can outlive a lifecycle bump
    // when a fresh MAP re-uses the id before the flush trigger fires).
    {
        let current = state
            .mappings
            .get(&key.mapping_id)
            .map(|m| m.map_generation as u64);
        if current != Some(entry.map_generation) {
            crate::backend::vulkan::engine::unpin_resident_target(&identity);
            crate::observe::fail(format!(
                "deferred_flush_lost kind=render mapping={} {}x{} gen={} reason=map_generation_drift current={:?}",
                key.mapping_id, entry.width, entry.height, entry.map_generation, current
            ));
            return false;
        }
    }
    let result = crate::runtime::import_present::try_import_present_store(
        state,
        host,
        &identity,
        key.mapping_id,
        entry.width,
        entry.height,
        entry.full_quad_bounds,
    );
    crate::backend::vulkan::engine::unpin_resident_target(&identity);
    if !matches!(
        result,
        crate::runtime::import_present::ImportPresentResult::Ok
    ) {
        // A failure before the readback consume leaves the armed prefetch
        // slot live — cancel it so the orphan can never be matched later
        // (no-op when the seq was consumed or never armed).
        // The window is gone from the map either way; guest pages keep their
        // stale-but-coherent pre-Store bytes. Name the loss.
        crate::observe::fail(format!(
            "deferred_flush_lost kind=render mapping={} {}x{} gen={} reason={}",
            key.mapping_id,
            entry.width,
            entry.height,
            entry.map_generation,
            result.reason()
        ));
        return false;
    }
    // Measure-only consume census: this deferred window was actually flushed
    // into guest pages — a consumer (present capture, guest sample, or
    // SynchronizeResources) read them, so the writeback was needed.
    crate::runtime::census::writeback_census::note_flushed();
    let flush_us = started.elapsed().as_micros() as u64;
    // Per-flush success census with wall-clock `us=` — one line per deferred
    // render flush (~1.8k/25s under a continuously-animating app) and
    // SCHED_IDLE-contaminated timing. The failure path above stays fail-visible
    // (`deferred_flush_lost reason=`); the store bytes are already attributed to
    // the tranche `store_us`. Gate the success line behind REIMS_VGPU_DRAW_LOG.
    crate::observe::line(format!(
        "render_deferred_flush mapping={} {}x{} gen={} off={} span_end={} us={flush_us}",
        key.mapping_id,
        entry.width,
        entry.height,
        entry.map_generation,
        key.surface_offset,
        key.span_end,
    ));
    // Per-tranche `store_us` attribution is noted by the inner
    // `try_import_present_store` above (the zero-copy GPU→guest DMA that is
    // essentially all of `flush_us`); do NOT note again here or the flush's
    // store bytes would be double-counted.
    true
}

#[cfg(not(feature = "backend-vulkan"))]
pub(crate) fn render_flush_one<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    _host: &mut M,
    key: &crate::model::RenderDeferredKey,
    entry: &crate::model::RenderDeferredEntry,
) -> bool {
    let _ = state;
    crate::observe::fail(format!(
        "deferred_flush_lost kind=render mapping={} reason=no_backend gen={}",
        key.mapping_id, entry.map_generation
    ));
    false
}

/// Drop (without flushing) every deferred window on `mapping_id` whose pages
/// can no longer be written safely (ReplacePhysical PFN recycling, unmap
/// without host access). Each drop is fail-visible.
pub fn drop_windows(state: &mut DeviceState, mapping_id: u32, reason: &str) {
    let dropped = state.take_deferred_flush_windows(mapping_id, 0, u64::MAX);
    for (key, generation) in dropped {
        crate::observe::fail(format!(
            "deferred_flush_dropped mapping={} reason={reason} {}x{} fmt={:#x} gen={generation}",
            key.mapping_id, key.width, key.height, key.pixel_format
        ));
        #[cfg(feature = "backend-vulkan")]
        crate::backend::vulkan::engine::unpin_resident_storage(&key);
    }
    let render_dropped = state.take_render_deferred_windows(mapping_id, 0, u64::MAX);
    for (key, entry) in render_dropped {
        crate::observe::fail(format!(
            "deferred_flush_dropped kind=render mapping={} reason={reason} {}x{} gen={}",
            key.mapping_id, entry.width, entry.height, entry.map_generation
        ));
        #[cfg(feature = "backend-vulkan")]
        {
            crate::backend::vulkan::engine::unpin_resident_target(
                &crate::runtime::import_present::render_deferred_identity(key.mapping_id, &entry),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::model::{ComputeStorageResidencyKey, DeviceId, DeviceState, PAGE_SHIFT_X86};

    fn key(mapping_id: u32, lo: u64, hi: u64) -> ComputeStorageResidencyKey {
        ComputeStorageResidencyKey {
            mapping_id,
            map_generation: 1,
            surface_offset: lo,
            surface_bpr: 64,
            span_end: hi,
            width: 4,
            height: 4,
            pixel_format: 0x46,
            texture_ref: 0,
        }
    }

    #[test]
    fn take_render_deferred_windows_is_exact_intersection() {
        use crate::model::{RenderDeferredEntry, RenderDeferredKey};
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let entry = RenderDeferredEntry {
            width: 4,
            height: 4,
            map_generation: 1,
            full_quad_bounds: false,
            grouped: false,
            armed_seq: 0,
        };
        let rkey = |mapping_id: u32, lo: u64, hi: u64| RenderDeferredKey {
            mapping_id,
            surface_offset: lo,
            span_end: hi,
        };
        state.render_deferred_flush.insert(rkey(7, 0, 256), entry);
        state.render_deferred_flush.insert(rkey(7, 256, 512), entry);
        state.render_deferred_flush.insert(rkey(8, 0, 256), entry);

        assert!(state.take_render_deferred_windows(7, 512, 1024).is_empty());
        assert_eq!(state.render_deferred_flush.len(), 3);

        let taken = state.take_render_deferred_windows(7, 200, 257);
        assert_eq!(taken.len(), 2, "both mapping-7 windows intersect [200,257)");
        assert_eq!(state.render_deferred_flush.len(), 1);
        assert!(state.render_deferred_flush.contains_key(&rkey(8, 0, 256)));
    }

    #[test]
    fn condemn_keeps_content_state_and_lifecycle_clears_it() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let m = state.mappings.entry(7).or_default();
        m.mapped = true;
        m.has_geom = true;
        m.width = 100;
        m.height = 50;
        m.format = 0x46;
        m.map_generation = 4;
        m.page_entries = vec![5, 9, 13];
        assert!(state.condemn_surface_backing(7));
        let e = state.mappings.get(&7).unwrap();
        assert!(e.mapped, "condemn must not unmap");
        assert!(e.has_geom, "condemn must keep geometry");
        assert_eq!(e.map_generation, 4, "condemn must not bump the generation");
        assert!(e.page_entries.is_empty(), "live bindings must be retired");
        assert_eq!(e.condemned_entries.as_deref(), Some(&[5u32, 9, 13][..]));
        assert!(state.mapping_backing_condemned(7));
        // Second condemn with no resolve between: nothing left to stash — the
        // caller falls back to full teardown (genuinely dead).
        // (mapping_backing_condemned gates that in the drain handler.)
        // A fresh MAP notify does NOT settle the pending decision (the notify
        // may trail our eager resolve of the same surface): the fingerprint
        // survives; only a resolve (or unmap/new-internal) settles it.
        assert!(state.map_surface(7));
        assert!(state.mapping_backing_condemned(7));
        assert!(state.unmap_surface(7));
        assert!(!state.mapping_backing_condemned(7));
        // Pageless mapping: condemn declines (caller tears down).
        let m = state.mappings.entry(8).or_default();
        m.mapped = true;
        assert!(!state.condemn_surface_backing(8));
    }

    #[test]
    fn map_notify_stashes_fingerprint_instead_of_bumping() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let m = state.mappings.entry(5).or_default();
        m.mapped = true;
        m.map_generation = 7;
        m.page_entries = vec![1, 2, 3];
        // The MAP notify often trails the eager resolve that established the
        // same surface: it must not bump (the resolve-time fingerprint compare
        // decides), so a deferred paint's resident/window stay live.
        assert!(state.map_surface(5));
        let e = state.mappings.get(&5).unwrap();
        assert_eq!(e.map_generation, 7, "late MAP notify must not bump");
        assert_eq!(e.condemned_entries.as_deref(), Some(&[1u32, 2, 3][..]));
        assert!(!e.has_geom, "geometry must re-resolve after MAP");
        // Same MappingInternal re-statement: full no-op for content state.
        let m = state.mappings.entry(6).or_default();
        m.mapped = true;
        m.map_generation = 9;
        m.mapping_internal = 0xabc;
        m.page_entries = vec![4, 5];
        m.has_geom = true;
        assert!(state.attach_mapping_internal(6, 0xabc));
        let e = state.mappings.get(&6).unwrap();
        assert_eq!(e.map_generation, 9);
        assert_eq!(e.page_entries, vec![4, 5]);
        assert!(e.has_geom, "same-internal re-statement keeps geometry");
        // Different MappingInternal: genuine new surface — full reset + bump.
        assert!(state.attach_mapping_internal(6, 0xdef));
        let e = state.mappings.get(&6).unwrap();
        assert_eq!(e.map_generation, 10);
        assert!(e.page_entries.is_empty());
    }

    #[test]
    fn render_flush_refuses_map_generation_drift() {
        use crate::model::{RenderDeferredEntry, RenderDeferredKey};
        use crate::runtime::host::FakeHost;
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        // Mapping exists but was re-wired since defer time (gen 3 vs window's
        // gen 2): the flush must refuse (never DMA old content into rewired
        // pages) and consume the window fail-visibly. This guards ALL render
        // windows, not just grouped ones.
        let m = state.mappings.entry(9).or_default();
        m.mapped = true;
        m.map_generation = 3;
        state.render_deferred_flush.insert(
            RenderDeferredKey {
                mapping_id: 9,
                surface_offset: 0,
                span_end: 4096,
            },
            RenderDeferredEntry {
                width: 32,
                height: 32,
                map_generation: 2,
                full_quad_bounds: false,
                grouped: false,
                armed_seq: 0,
            },
        );
        let ok = super::flush_intersecting(&mut state, &mut host, 9, 0, u64::MAX);
        assert!(!ok, "drifted window must report the loss");
        assert!(state.render_deferred_flush.is_empty());
    }

    #[test]
    fn flush_intersecting_takes_render_windows_and_reports_loss() {
        use crate::model::{RenderDeferredEntry, RenderDeferredKey};
        use crate::runtime::host::FakeHost;
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        // Window over an unmapped mapping: the flush replay must fail closed
        // (fail-visible loss), remove the window, and return false.
        state.render_deferred_flush.insert(
            RenderDeferredKey {
                mapping_id: 9,
                surface_offset: 0,
                span_end: 4096,
            },
            RenderDeferredEntry {
                width: 32,
                height: 32,
                map_generation: 1,
                full_quad_bounds: false,
                grouped: false,
                armed_seq: 0,
            },
        );
        let ok = super::flush_intersecting(&mut state, &mut host, 9, 0, u64::MAX);
        assert!(!ok, "lost render window must report failure");
        assert!(
            state.render_deferred_flush.is_empty(),
            "taken windows never return to the map"
        );
        // Disjoint mapping id: untouched.
        state.render_deferred_flush.insert(
            RenderDeferredKey {
                mapping_id: 10,
                surface_offset: 0,
                span_end: 4096,
            },
            RenderDeferredEntry {
                width: 32,
                height: 32,
                map_generation: 1,
                full_quad_bounds: false,
                grouped: false,
                armed_seq: 0,
            },
        );
        assert!(super::flush_intersecting(
            &mut state,
            &mut host,
            11,
            0,
            u64::MAX
        ));
        assert_eq!(state.render_deferred_flush.len(), 1);
    }

    /// A raw task-GVA span whose physical pages alias a deferred window's
    /// mapping pages must take (and attempt to flush) that window; a window
    /// on non-aliased pages stays. Locks the boot-18 linear_sample poisoning
    /// channel: GVA reads bypassing the mapping-keyed hooks.
    #[test]
    fn gva_alias_takes_only_aliased_windows() {
        use crate::contract::endian::st32;
        use crate::contract::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
        use crate::model::{RenderDeferredEntry, RenderDeferredKey};
        use crate::runtime::host::{FakeHost, HostMemory};
        let page_shift = PAGE_SHIFT_X86;
        let mut host = FakeHost::new();
        // Task 1 directory at pfn 2 → root table pfn 3 → gva page 0 =
        // pfn 0x2000. Data pfns sit past the default task object list
        // (pfn 1 + 0x100000 slots = 4096 pages), which the mapping
        // control-page collision check treats as reserved.
        let dir_gpa = 2u64 << page_shift;
        let root_gpa = 3u64 << page_shift;
        let data_gpa = 0x2000u64 << page_shift;
        host.map_range(dir_gpa, 0x20, 0);
        host.map_range(root_gpa, 0x1000, 0);
        host.map_range(data_gpa, 0x1000, 0xab);
        let mut d = [0u8; 8];
        st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 3);
        st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
        host.write_gpa(dir_gpa, &d).unwrap();
        let mut pte = [0u8; 4];
        st32(&mut pte, 0x2000);
        host.write_gpa(root_gpa, &pte).unwrap();

        let mut state = DeviceState::new(DeviceId(1), page_shift);
        assert!(state.define_task(1, 0x1000, 2));
        // Mapping 9 is backed by pfn 0x2000 (the page the GVA span resolves
        // to); mapping 10 is backed by pfn 0x2001 (disjoint).
        let page_entry = |pfn: u32| (pfn << 2) | 1;
        for (mid, pfn) in [(9u32, 0x2000u32), (10, 0x2001)] {
            let m = state.mappings.entry(mid).or_default();
            m.mapped = true;
            m.page_entries = vec![page_entry(pfn)];
        }
        let entry = RenderDeferredEntry {
            width: 32,
            height: 32,
            map_generation: 1,
            full_quad_bounds: false,
            grouped: false,
            armed_seq: 0,
        };
        let rkey = |mapping_id: u32| RenderDeferredKey {
            mapping_id,
            surface_offset: 0,
            span_end: 0x1000,
        };
        state.render_deferred_flush.insert(rkey(9), entry);
        state.render_deferred_flush.insert(rkey(10), entry);
        // Product defer sites index pages at defer time.
        state.index_deferred_alias_pages(9);
        state.index_deferred_alias_pages(10);
        assert_eq!(state.deferred_alias_pages.len(), 2);

        super::flush_intersecting_task_gva(&mut state, &mut host, 1, 0, 0x100);
        assert!(
            !state.render_deferred_flush.contains_key(&rkey(9)),
            "aliased window must be taken for flush"
        );
        assert!(
            state.render_deferred_flush.contains_key(&rkey(10)),
            "non-aliased window must stay deferred"
        );
        assert!(
            !state.deferred_alias_pages.contains_key(&9),
            "alias index must drop with the mapping's last window"
        );
        assert!(
            state.deferred_alias_pages.contains_key(&10),
            "alias index for the untouched mapping must stay"
        );
    }

    /// SynchronizeResources choke point: the guest names a mapping it is
    /// about to CPU-read; every deferred window on it — mapping-keyed
    /// (compute + render) and linear windows whose defer-time page index
    /// aliases the mapping's physical pages — must be taken for flush.
    /// Windows on disjoint mappings/pages stay deferred. Locks the
    /// boot-25 black-wallpaper class (guest-CPU composite of stale pages).
    #[test]
    fn guest_read_flush_takes_keyed_and_linear_alias_windows() {
        use crate::model::{RenderDeferredEntry, RenderDeferredKey};
        use crate::runtime::host::FakeHost;
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        let page_entry = |pfn: u32| (pfn << 2) | 1;
        for (mid, pfn) in [(9u32, 0x2000u32), (10, 0x2001)] {
            let m = state.mappings.entry(mid).or_default();
            m.mapped = true;
            m.page_entries = vec![page_entry(pfn)];
        }
        state.compute_deferred_flush.insert(key(9, 0, 256), 3);
        let entry = RenderDeferredEntry {
            width: 32,
            height: 32,
            map_generation: 1,
            full_quad_bounds: false,
            grouped: false,
            armed_seq: 0,
        };
        let rkey = |mapping_id: u32| RenderDeferredKey {
            mapping_id,
            surface_offset: 0,
            span_end: 0x1000,
        };
        state.render_deferred_flush.insert(rkey(9), entry);
        state.render_deferred_flush.insert(rkey(10), entry);
        // Linear windows never name the mapping: one aliases mapping 9's
        // physical page, one sits on a disjoint page.
        let mut lin_aliased = key(0, 0, 0x1000);
        lin_aliased.texture_ref = 42;
        let mut lin_disjoint = key(0, 0, 0x1000);
        lin_disjoint.texture_ref = 43;
        let aliased_pages: std::collections::HashSet<u64> =
            [(0x2000u64) << PAGE_SHIFT_X86].into_iter().collect();
        let disjoint_pages: std::collections::HashSet<u64> =
            [(0x3000u64) << PAGE_SHIFT_X86].into_iter().collect();
        state
            .linear_deferred_flush
            .insert(lin_aliased, (1, aliased_pages));
        state
            .linear_deferred_flush
            .insert(lin_disjoint, (1, disjoint_pages));

        // No windows on mapping 11: clean no-op.
        assert_eq!(
            super::flush_mapping_for_guest_read(&mut state, &mut host, 11),
            (true, 0)
        );

        let (ok, flushed) = super::flush_mapping_for_guest_read(&mut state, &mut host, 9);
        // Nothing is engine-pinned / host-mapped in this fixture, so every
        // flush reports a fail-visible loss — but every aliased window must
        // still be taken (obligations never return to the maps).
        assert!(!ok, "losses must be reported");
        assert_eq!(flushed, 3, "compute@9 + render@9 + linear alias");
        assert!(state.compute_deferred_flush.is_empty());
        assert!(!state.render_deferred_flush.contains_key(&rkey(9)));
        assert!(
            state.render_deferred_flush.contains_key(&rkey(10)),
            "disjoint mapping's render window must stay deferred"
        );
        assert!(
            !state.linear_deferred_flush.contains_key(&lin_aliased),
            "page-aliased linear window must be taken"
        );
        assert!(
            state.linear_deferred_flush.contains_key(&lin_disjoint),
            "disjoint-page linear window must stay deferred"
        );
    }

    fn gva_entry(task_id: u32, w: u32, h: u32, pages: &[u64]) -> crate::model::GvaDeferredEntry {
        crate::model::GvaDeferredEntry {
            task_id,
            texture_ref: 5,
            producer_object_type: 2,
            width: w,
            height: h,
            row_stride: w * 4,
            format: 0x46,
            armed_seq: 0,
            pages: pages.iter().copied().collect(),
        }
    }

    /// The refcounted deferred-page union index (`deferred_page_refs`, the fast
    /// reject behind `deferred_pages_intersect`) tracks exactly the live window
    /// pages: shared pages are refcounted so a page survives until its LAST
    /// window disarms, re-arm swaps page sets cleanly, and a fresh rebuild agrees
    /// with the incrementally-maintained index.
    #[test]
    fn deferred_page_refs_track_arm_disarm_with_sharing() {
        use crate::model::{DeviceState, PAGE_SHIFT_X86};
        let mut state = DeviceState::new(crate::model::DeviceId(1), PAGE_SHIFT_X86);
        let p = |pfn: u64| pfn << PAGE_SHIFT_X86;
        // Two windows share page A; window 1 also owns B, window 2 also owns C.
        state.arm_gva_deferred_window(0x1000, gva_entry(1, 4, 4, &[p(0xA), p(0xB)]));
        state.arm_gva_deferred_window(0x2000, gva_entry(1, 4, 4, &[p(0xA), p(0xC)]));
        assert!(state.deferred_pages_intersect(&[p(0xA)]));
        assert!(state.deferred_pages_intersect(&[p(0xB)]));
        assert!(state.deferred_pages_intersect(&[p(0xC)]));
        assert!(!state.deferred_pages_intersect(&[p(0xD)]));
        // Disarm window 1: shared A stays (window 2 still holds it), B leaves.
        assert!(state.take_gva_deferred_window(0x1000).is_some());
        assert!(
            state.deferred_pages_intersect(&[p(0xA)]),
            "A shared by window 2"
        );
        assert!(
            !state.deferred_pages_intersect(&[p(0xB)]),
            "B was only window 1"
        );
        assert!(state.deferred_pages_intersect(&[p(0xC)]));
        // Re-arm window 2 onto a different page set: C leaves, E joins, A stays
        // present only if the new set keeps it — here it does not.
        state.arm_gva_deferred_window(0x2000, gva_entry(1, 4, 4, &[p(0xE)]));
        assert!(
            !state.deferred_pages_intersect(&[p(0xA)]),
            "re-arm dropped A"
        );
        assert!(
            !state.deferred_pages_intersect(&[p(0xC)]),
            "re-arm dropped C"
        );
        assert!(state.deferred_pages_intersect(&[p(0xE)]));
        // The incremental index must match a from-scratch rebuild (no drift).
        assert!(
            !state.verify_and_heal_deferred_refs(),
            "incremental index agrees with a fresh rebuild"
        );
        // Disarm the last window: index empties.
        assert!(state.take_gva_deferred_window(0x2000).is_some());
        assert!(!state.deferred_pages_intersect(&[p(0xE)]));
        assert!(!state.verify_and_heal_deferred_refs());
    }

    /// A raw task-GVA span aliasing a deferred GVA render-Store window's
    /// pages (or naming its base GVA exactly) must take the window; windows
    /// on disjoint pages stay armed. Same channel as the linear windows —
    /// GVA reads that bypass every mapping-keyed hook.
    #[test]
    fn task_gva_alias_takes_gva_store_windows() {
        use crate::contract::endian::st32;
        use crate::contract::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
        use crate::runtime::host::{FakeHost, HostMemory};
        let page_shift = PAGE_SHIFT_X86;
        let mut host = FakeHost::new();
        let dir_gpa = 2u64 << page_shift;
        let root_gpa = 3u64 << page_shift;
        let data_gpa = 0x2000u64 << page_shift;
        host.map_range(dir_gpa, 0x20, 0);
        host.map_range(root_gpa, 0x1000, 0);
        host.map_range(data_gpa, 0x1000, 0xab);
        let mut d = [0u8; 8];
        st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 3);
        st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
        host.write_gpa(dir_gpa, &d).unwrap();
        let mut pte = [0u8; 4];
        st32(&mut pte, 0x2000);
        host.write_gpa(root_gpa, &pte).unwrap();

        let mut state = DeviceState::new(DeviceId(1), page_shift);
        assert!(state.define_task(1, 0x1000, 2));
        // Window A aliases the page the span resolves to; window B does not.
        state.arm_gva_deferred_window(0x9000_0000, gva_entry(1, 4, 4, &[0x2000u64 << page_shift]));
        state.arm_gva_deferred_window(0x9100_0000, gva_entry(1, 4, 4, &[0x3000u64 << page_shift]));

        super::flush_intersecting_task_gva(&mut state, &mut host, 1, 0, 0x100);
        // No engine in this fixture: the flush reports a fail-visible loss,
        // but the aliased window must be taken (obligations never return).
        assert!(
            !state.gva_deferred_flush.contains_key(&0x9000_0000),
            "page-aliased GVA window must be taken"
        );
        assert!(
            state.gva_deferred_flush.contains_key(&0x9100_0000),
            "disjoint GVA window must stay armed"
        );

        // Exact-base fast path: a read naming the window's own GVA takes it
        // without any page walk.
        super::flush_intersecting_task_gva(&mut state, &mut host, 1, 0x9100_0000, 0x10);
        assert!(
            !state.gva_deferred_flush.contains_key(&0x9100_0000),
            "exact-base read must take the window"
        );
    }

    /// PT builder shared by the no-intersection-memo tests: task 1's GVA
    /// `0..0x1000` resolves to data page `0x2000<<shift`. Returns the host so
    /// the caller can remap the PTE to simulate a task page-table change.
    fn memo_pt_fixture() -> (crate::runtime::host::FakeHost, DeviceState, u64, u32) {
        use crate::contract::endian::st32;
        use crate::contract::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
        use crate::runtime::host::{FakeHost, HostMemory};
        let page_shift = PAGE_SHIFT_X86;
        let mut host = FakeHost::new();
        let dir_gpa = 2u64 << page_shift;
        let root_gpa = 3u64 << page_shift;
        host.map_range(dir_gpa, 0x20, 0);
        host.map_range(root_gpa, 0x1000, 0);
        host.map_range(0x2000u64 << page_shift, 0x1000, 0xab);
        host.map_range(0x3000u64 << page_shift, 0x1000, 0xcd);
        let mut d = [0u8; 8];
        st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 3);
        st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
        host.write_gpa(dir_gpa, &d).unwrap();
        let mut pte = [0u8; 4];
        st32(&mut pte, 0x2000);
        host.write_gpa(root_gpa, &pte).unwrap();
        let mut state = DeviceState::new(DeviceId(1), page_shift);
        assert!(state.define_task(1, 0x1000, 2));
        (host, state, root_gpa, page_shift)
    }

    /// A repeated buffer bind that never aliases a live deferred window walks
    /// the task PT once, then rides the no-intersection memo: the second and
    /// later identical binds skip the walk. A deferred-signature change (arm or
    /// take a window) drops the memo and forces exactly one re-walk.
    #[test]
    fn flush_nohit_memo_skips_repeat_walk_and_rechecks_on_signature_change() {
        let (mut host, mut state, _root, page_shift) = memo_pt_fixture();
        // Disjoint window on page 0x3000 keeps the deferred set non-empty but
        // never intersects the [0,0x100) bind (which resolves to page 0x2000).
        state.arm_gva_deferred_window(0x9100_0000, gva_entry(1, 4, 4, &[0x3000u64 << page_shift]));

        // First bind: full walk, no hit -> memo caches (task=1, gva=0, span=0x100)
        // with the resolved page 0x2000.
        super::flush_intersecting_task_gva(&mut state, &mut host, 1, 0, 0x100);
        let cached = state.flush_nohit_memo.get(&(1, 0, 0x100)).cloned();
        assert!(cached.is_some(), "first bind must cache its resolved pages");
        assert_eq!(
            cached.unwrap().1,
            vec![0x2000u64 << page_shift],
            "cached pages are the bind's resolved gpa pages"
        );
        assert_eq!(state.tranche.zc_flush_skip, 0, "first bind must walk");

        // Second identical bind, signature unchanged: fast skip (no walk).
        super::flush_intersecting_task_gva(&mut state, &mut host, 1, 0, 0x100);
        assert_eq!(state.tranche.zc_flush_skip, 1, "repeat bind must skip");
        assert!(state.gva_deferred_flush.contains_key(&0x9100_0000));
        assert_eq!(state.tranche.zc_flush_hits, 0);
        assert_eq!(state.tranche.zc_flush_stale, 0);

        // Arm another DISJOINT window: signature changes, but the cheap page
        // recheck (cached page 0x2000 vs windows on 0x3000) still finds no
        // intersection -> SKIP without a PT walk (the residual-fix win).
        state.arm_gva_deferred_window(0x9200_0000, gva_entry(1, 4, 4, &[0x3000u64 << page_shift]));
        super::flush_intersecting_task_gva(&mut state, &mut host, 1, 0, 0x100);
        assert_eq!(
            state.tranche.zc_flush_skip, 2,
            "signature change with disjoint window re-checks cheaply and skips"
        );
        assert!(state.flush_nohit_memo.contains_key(&(1, 0, 0x100)));

        // Task redefine drops the task's memo entries (the PT may remap).
        assert!(state.define_task(1, 0x1000, 2));
        assert!(
            !state.flush_nohit_memo.contains_key(&(1, 0, 0x100)),
            "task redefine must invalidate the flush memo"
        );
    }

    /// A large bind's alias must be found wherever it sits, not only where a
    /// sample point happens to land.
    ///
    /// This walk used to sample every 16th page once a span passed 64 pages, on
    /// the stated grounds that "real aliases are same-surface, so the first page
    /// hits". Measured on the rail, no alias hit page 0 — the three observed
    /// landed at 16, 32 and 48 of 127- and 256-page spans, i.e. partial overlaps
    /// somewhere below each sample point. So the miss window was live, and this
    /// is what falls through it: a 65-page bind overlapping a window on page 1
    /// alone, which a stride of 16 steps straight over.
    #[test]
    fn a_large_bind_alias_is_found_off_the_sample_points() {
        use crate::contract::endian::st32;
        use crate::runtime::host::HostMemory;
        let (mut host, mut state, root_gpa, page_shift) = memo_pt_fixture();
        // 65 pages, so the old rule ran a strided walk. Page i -> pfn 0x4000+i.
        const N: u64 = 65;
        for i in 0..N {
            let pfn = 0x4000 + i;
            host.map_range(pfn << page_shift, 0x1000, 0);
            let mut pte = [0u8; 4];
            st32(&mut pte, pfn as u32);
            host.write_gpa(root_gpa + 4 * i, &pte).unwrap();
        }
        // The deferred window covers page 1 and nothing else. A stride-16 walk
        // visits 0, 16, 32, 48, 64 — never 1.
        state.arm_gva_deferred_window(0x9100_0000, gva_entry(1, 4, 4, &[0x4001u64 << page_shift]));
        super::flush_intersecting_task_gva(&mut state, &mut host, 1, 0, N << page_shift);
        assert!(
            !state.gva_deferred_flush.contains_key(&0x9100_0000),
            "a window aliasing page 1 of a 65-page bind must be found and flushed"
        );
    }

    /// `gva_alias_hit_page` reports which page of a bind first overlapped a live
    /// deferred window, and the walk exists to find every such overlap — so the
    /// line has to be able to tell a page-0 hit from a later one. A probe that
    /// always said 0 would look healthy while measuring nothing, and reading this
    /// number is what refuted the "real aliases hit on the first page" claim that
    /// a 1-in-16 sampled walk used to rest on.
    ///
    /// Same fixture, same two-page span, same window geometry — only which page
    /// the window sits on differs.
    #[test]
    fn alias_hit_page_probe_separates_a_first_page_hit_from_a_later_one() {
        use crate::contract::endian::st32;
        use crate::runtime::host::HostMemory;
        let first_hit_page = |state: &mut DeviceState,
                              host: &mut crate::runtime::host::FakeHost,
                              window_page: u64|
         -> String {
            let page_shift = PAGE_SHIFT_X86;
            state.arm_gva_deferred_window(
                0x9100_0000,
                gva_entry(1, 4, 4, &[window_page << page_shift]),
            );
            crate::observe::redirect_logs_for_tests();
            let at = std::fs::read_to_string(crate::observe::fail_log_path())
                .unwrap_or_default()
                .len();
            // Two guest pages from gva 0, so page 0 resolves to 0x2000 and page 1
            // to 0x3000 — the window sits on exactly one of them.
            super::flush_intersecting_task_gva(state, host, 1, 0, 2 << page_shift);
            let body = std::fs::read_to_string(crate::observe::fail_log_path()).unwrap_or_default();
            body[at.min(body.len())..]
                .lines()
                .find(|l| l.starts_with("gva_alias_hit_page "))
                .and_then(|l| {
                    l.split_whitespace()
                        .find(|f| f.starts_with("first_hit_page="))
                })
                .unwrap_or("<no line>")
                .to_string()
        };

        let (mut host, mut state, root_gpa, page_shift) = memo_pt_fixture();
        // Wire PTE[1] -> 0x3000 so the span's second page resolves too.
        let mut pte = [0u8; 4];
        st32(&mut pte, 0x3000);
        host.write_gpa(root_gpa + 4, &pte).unwrap();
        assert_eq!(
            first_hit_page(&mut state, &mut host, 0x2000),
            "first_hit_page=0",
            "a window on the span's first page must report page 0"
        );

        let (mut host, mut state, root_gpa, _) = memo_pt_fixture();
        let mut pte = [0u8; 4];
        st32(&mut pte, 0x3000);
        host.write_gpa(root_gpa + 4, &pte).unwrap();
        assert_eq!(
            first_hit_page(&mut state, &mut host, 0x3000),
            "first_hit_page=1",
            "a window on the span's second page must report page 1, not 0"
        );
        let _ = page_shift;
    }

    /// A signature change that arms a window ON the cached bind's own page is
    /// caught by the cheap recheck (no PT walk needed) and flushed.
    #[test]
    fn flush_nohit_memo_recheck_catches_new_intersecting_window() {
        let (mut host, mut state, _root, page_shift) = memo_pt_fixture();
        // Cache a no-hit result with a disjoint window.
        state.arm_gva_deferred_window(0x9100_0000, gva_entry(1, 4, 4, &[0x3000u64 << page_shift]));
        super::flush_intersecting_task_gva(&mut state, &mut host, 1, 0, 0x100);
        assert!(state.flush_nohit_memo.contains_key(&(1, 0, 0x100)));

        // Arm a window ON the bind's resolved page 0x2000. Signature changes; the
        // cheap recheck of the cached pages finds the intersection and the full
        // walk flushes the window.
        state.arm_gva_deferred_window(0x9300_0000, gva_entry(1, 4, 4, &[0x2000u64 << page_shift]));
        super::flush_intersecting_task_gva(&mut state, &mut host, 1, 0, 0x100);
        assert!(
            !state.gva_deferred_flush.contains_key(&0x9300_0000),
            "intersecting window must be flushed"
        );
        assert_eq!(
            state.tranche.zc_flush_stale, 0,
            "a fresh arm is not a stale-page miss"
        );
        assert!(
            !state.flush_nohit_memo.contains_key(&(1, 0, 0x100)),
            "an intersecting bind must drop its no-hit entry"
        );
    }

    /// Safety net: if a task PT change that should have invalidated the memo is
    /// missed (no retire / signature change), the 1-in-64 sampled full walk
    /// catches the now-real intersection, fail-logs `zc_flush_stale`, flushes
    /// the window, and drops the stale entry.
    #[test]
    fn flush_nohit_memo_sampled_verify_selfheals_missed_pt_change() {
        use crate::contract::endian::st32;
        use crate::runtime::host::HostMemory;
        let (mut host, mut state, root_gpa, page_shift) = memo_pt_fixture();
        // Window on page 0x3000 — disjoint from the [0,0x100) bind (page 0x2000)
        // at first, so the initial walk caches a no-intersection result.
        state.arm_gva_deferred_window(0x9100_0000, gva_entry(1, 4, 4, &[0x3000u64 << page_shift]));
        super::flush_intersecting_task_gva(&mut state, &mut host, 1, 0, 0x100);
        assert!(state.flush_nohit_memo.contains_key(&(1, 0, 0x100)));

        // Simulate a MISSED invalidation: remap gva page 0 -> 0x3000 directly in
        // guest RAM (no retire_gva_views, no deferred-set change), so the bind
        // now aliases the armed window but the memo still says "no hit" and the
        // deferred signature is unchanged.
        let mut pte = [0u8; 4];
        st32(&mut pte, 0x3000);
        host.write_gpa(root_gpa, &pte).unwrap();

        // 63 skips stay stale (the bounded hole), then the 64th memo hit samples
        // a full walk and self-heals.
        for _ in 0..63 {
            super::flush_intersecting_task_gva(&mut state, &mut host, 1, 0, 0x100);
        }
        assert_eq!(
            state.tranche.zc_flush_stale, 0,
            "hole is bounded to 64 draws"
        );
        assert!(state.gva_deferred_flush.contains_key(&0x9100_0000));

        super::flush_intersecting_task_gva(&mut state, &mut host, 1, 0, 0x100);
        assert_eq!(
            state.tranche.zc_flush_stale, 1,
            "sampled verify must catch the missed intersection"
        );
        assert!(
            !state.gva_deferred_flush.contains_key(&0x9100_0000),
            "sampled verify must flush the missed window"
        );
        assert!(
            !state.flush_nohit_memo.contains_key(&(1, 0, 0x100)),
            "stale memo entry must be dropped on self-heal"
        );
    }

    /// SynchronizeResources choke point: GVA windows whose defer-time pages
    /// alias the named mapping's physical pages must be taken for flush.
    #[test]
    fn guest_read_flush_takes_gva_store_alias_windows() {
        use crate::runtime::host::FakeHost;
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        let page_entry = |pfn: u32| (pfn << 2) | 1;
        let m = state.mappings.entry(9).or_default();
        m.mapped = true;
        m.page_entries = vec![page_entry(0x2000)];
        state.arm_gva_deferred_window(
            0x9000_0000,
            gva_entry(1, 4, 4, &[0x2000u64 << PAGE_SHIFT_X86]),
        );
        state.arm_gva_deferred_window(
            0x9100_0000,
            gva_entry(1, 4, 4, &[0x3000u64 << PAGE_SHIFT_X86]),
        );

        let (ok, flushed) = super::flush_mapping_for_guest_read(&mut state, &mut host, 9);
        assert!(!ok, "engine-less flush reports the loss");
        assert_eq!(flushed, 1, "exactly the aliased GVA window");
        assert!(!state.gva_deferred_flush.contains_key(&0x9000_0000));
        assert!(state.gva_deferred_flush.contains_key(&0x9100_0000));
    }

    /// The page-drift probe must distinguish the cases it exists to separate.
    ///
    /// A probe that reports nothing is indistinguishable from a probe that
    /// cannot fire, and this codebase has already paid for three of those. So
    /// drive both controls through the same fixture: a window whose GVA still
    /// resolves to its armed pages must stay silent, and one whose pages moved
    /// under it must produce the line — same task, same geometry, only the
    /// armed set differs.
    #[test]
    fn window_page_drift_probe_fires_on_drift_and_is_silent_without_it() {
        use crate::contract::endian::st32;
        use crate::contract::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
        use crate::runtime::host::{FakeHost, HostMemory};
        let page = 1u64 << PAGE_SHIFT_X86;
        let mut host = FakeHost::new();
        let (dir_gpa, root_gpa, data0) = (2 * page, 3 * page, 4 * page);
        for gpa in [dir_gpa, root_gpa, data0] {
            host.map_range(gpa, page as usize, 0);
        }
        let mut d = [0u8; 8];
        st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 3);
        st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
        host.write_gpa(dir_gpa, &d).unwrap();
        let mut pte = [0u8; 4];
        st32(&mut pte, 4);
        host.write_gpa(root_gpa, &pte).unwrap();
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        assert!(state.define_task(1, page, 2));

        crate::observe::redirect_logs_for_tests();
        let drift_lines = |from: usize| -> usize {
            std::fs::read_to_string(crate::observe::fail_log_path())
                .unwrap_or_default()
                .get(from..)
                .unwrap_or_default()
                .lines()
                .filter(|l| l.starts_with("deferred_window_page_drift "))
                .count()
        };
        let mark = || {
            std::fs::read_to_string(crate::observe::fail_log_path())
                .unwrap_or_default()
                .len()
        };

        // Negative control: armed on the page the GVA resolves to right now.
        let at = mark();
        super::report_window_page_drift(
            &state,
            &host,
            0,
            &gva_entry(1, 4, 4, &[data0]),
            "gva_alias",
        );
        assert_eq!(drift_lines(at), 0, "a window that did not move must be quiet");

        // Positive control: same window, armed on a page it no longer maps to.
        let at = mark();
        super::report_window_page_drift(
            &state,
            &host,
            0,
            &gva_entry(1, 4, 4, &[9 * page]),
            "gva_alias",
        );
        assert_eq!(drift_lines(at), 1, "a window whose pages moved must report");
    }

    /// Task teardown moves the task's GVA windows to the retired list (model)
    /// and the runtime lands them cache-only — obligations never write guest
    /// pages from teardown and never linger.
    #[test]
    fn task_delete_retires_gva_windows_cache_only() {
        use crate::runtime::host::FakeHost;
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        assert!(state.define_task(6, 0x1000, 2));
        state.arm_gva_deferred_window(0x9000_0000, gva_entry(6, 4, 4, &[]));
        state.arm_gva_deferred_window(0x9100_0000, gva_entry(7, 4, 4, &[]));
        assert!(state.delete_task(6));
        assert!(
            !state.gva_deferred_flush.contains_key(&0x9000_0000),
            "dead task's window must leave the armed map"
        );
        assert!(
            state.gva_deferred_flush.contains_key(&0x9100_0000),
            "other task's window must stay armed"
        );
        assert_eq!(state.retired_gva_windows.len(), 1);
        super::retire_gva_windows(&mut state, &mut host);
        assert!(state.retired_gva_windows.is_empty());
    }

    /// The window cap lands the oldest-armed window first.
    #[test]
    fn oldest_gva_window_is_taken_first() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut newer = gva_entry(1, 4, 4, &[]);
        newer.armed_seq = 9;
        let mut older = gva_entry(1, 4, 4, &[]);
        older.armed_seq = 3;
        state.arm_gva_deferred_window(0x1000, newer);
        state.arm_gva_deferred_window(0x2000, older);
        let (gva, entry) = state.take_oldest_gva_deferred_window().unwrap();
        assert_eq!(gva, 0x2000);
        assert_eq!(entry.armed_seq, 3);
        assert_eq!(state.gva_deferred_flush.len(), 1);
    }

    #[test]
    fn take_deferred_windows_is_exact_intersection() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        state.compute_deferred_flush.insert(key(7, 0, 256), 3);
        state.compute_deferred_flush.insert(key(7, 256, 512), 4);
        state.compute_deferred_flush.insert(key(8, 0, 256), 5);

        // Disjoint range takes nothing.
        assert!(state.take_deferred_flush_windows(7, 512, 1024).is_empty());
        assert_eq!(state.compute_deferred_flush.len(), 3);

        // Intersecting range takes only the touching window on that mapping.
        let taken = state.take_deferred_flush_windows(7, 200, 257);
        assert_eq!(taken.len(), 2, "both mapping-7 windows intersect [200,257)");
        assert_eq!(state.compute_deferred_flush.len(), 1);
        assert!(state.compute_deferred_flush.contains_key(&key(8, 0, 256)));
    }
}
