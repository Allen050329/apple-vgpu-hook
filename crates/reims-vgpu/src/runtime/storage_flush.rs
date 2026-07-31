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
/// Bound for the per-bind no-intersection memo (`DeviceState::flush_nohit_memo`).
///
/// An entry is *revalidated* on a deferred-signature change, not dropped: the
/// cached pages are re-checked against the new windows and re-stamped with the
/// new signature when they still miss. So an entry can outlive arbitrarily many
/// signature changes, and the live set is every distinct `(task, gva, span)`
/// bind that has ever missed, not the ones seen at one signature. The cap is a
/// runaway guard on that set: overflow just re-walks (correct, slower).
///
/// The walk this memo skips visits every page of the span — a stride argument
/// with a sparse mode for large spans used to sit here and is gone.
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
    // names the same base GVA — no page walk needed to detect it.
    if state.gva_deferred_flush.contains_key(&gva) {
        flush_gva_exact(state, host, gva, true, "gva_alias");
    }
    if state.deferred_alias_pages.is_empty()
        && state.linear_deferred_flush.is_empty()
        && state.gva_deferred_flush.is_empty()
    {
        return;
    }
    // No-intersection memo. The per-page task-PT walk below only finds work when
    // a bound buffer span aliases a live deferred window, which is rare but NOT
    // dead: across five x86/Vulkan repro boots it hit once in 408k calls and
    // 89k walks (boot 89; boots 86, 87, 88 and 90 hit zero), and the fail sink
    // holds 11 `gva_alias_hit_page` lines lifetime. An earlier reading of
    // `zc_flush_hits == 0` over ~59k draws is restated here as "hits about once
    // per boot" rather than "detection overhead" — the walk cannot be removed,
    // only made cheap. Cache each full-walked bind's resolved gpa pages; skip
    // the walk while the deferred signature is unchanged, and on a signature
    // change re-check the cached pages against the current windows WITHOUT a PT
    // walk (the per-page FFI translate is the expensive part). A 1-in-64 sampled
    // full walk (`flush_verify_ctr`) self-heals a missed task-PT remap
    // (`zc_flush_stale`, must stay 0) — the memo's pages are only as good as the
    // guest telling us it remapped, which is the one invariant here that no local
    // audit can close.
    let sig = state.deferred_flush_signature();
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
            let isect = state.deferred_pages_intersect(&pages);
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
                for (key, entry) in linear_index.iter() {
                    if entry.pages.contains(&gpa_page)
                        && !linear_hits.iter().any(|(k, _)| k == key)
                    {
                        linear_hits.push((*key, entry.generation));
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
/// deferred). Mapping-keyed compute windows flush via [`flush_intersecting`];
/// linear task-GVA windows never name a mapping, so they flush when their
/// defer-time page index aliases the mapping's physical pages. Returns
/// `(all_ok, windows_flushed)`.
pub fn flush_mapping_for_guest_read<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mapping_id: u32,
) -> (bool, u32) {
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
/// window's pages can move while still resolving was the open question this used
/// to only report on, on the grounds that a guard for an unmeasured hazard is a
/// guess.
///
/// **It is measured now, and it happens.** One x86/Vulkan boot driving Finder,
/// Calendar and Safari produced fourteen of these, and in most of them *every*
/// armed page had moved — `armed_pages=73 live_pages=73 moved=73` for a 196x381
/// window, and the same total displacement at 5, 4 and 22 pages under the
/// `clear_store`, `rearm` and `gva_alias` triggers. So the guard has its
/// measurement and this decides.
///
/// It returns `true` when the window may still be written to guest RAM. Drift
/// means our own bookkeeping is the stale part: the window was armed against one
/// set of guest pages, the guest has since re-pointed `[gva, gva+span)`
/// somewhere else, and [`crate::runtime::metal_draw::write_gva_rgba8`] walks
/// fresh — so the write lands in whatever owns those pages *now*. On this rail
/// that has been observed as guest heap corruption: WindowServer aborting inside
/// `small_free_list_remove_ptr_no_clear`, and the guest kernel panicking with
/// `element modified after free` on a freed allocation overwritten with white
/// RGBA8 pixels.
///
/// Refusing costs stale bytes at a guest address the guest has already
/// repurposed; permitting costs somebody else's heap. The caller keeps the
/// content either way — `host_cache_store_gva_layer` runs unconditionally — so
/// nothing renderable is lost by refusing.
#[cfg(feature = "backend-vulkan")]
pub(crate) fn window_pages_still_ours<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    gva: u64,
    entry: &crate::model::GvaDeferredEntry,
    trigger: &str,
    outcome: &str,
) -> bool {
    deferred_pages_still_ours(
        state,
        host,
        entry.task_id,
        gva,
        entry.span(),
        &entry.pages,
        &format!("{}x{} trigger={trigger}", entry.width, entry.height),
        outcome,
    )
}

/// The drift decision itself, over any deferred window's armed page set.
///
/// Both deferred rails arm against a page set resolved at defer time and then
/// write guest RAM through a *fresh* walk at flush time, so both have the same
/// hazard and the same answer. Keeping one implementation is what stops the
/// second rail from drifting away from the first: the linear rail carried this
/// hazard with no check at all while the GVA rail had one, purely because the
/// check lived inside the GVA-shaped function.
///
/// Returns `true` when the window still names the pages it was armed on.
///
/// `outcome` names what the caller gives up when this answers `false`, because
/// the question has two consumers that lose different things. A flush asks it
/// to keep a write off somebody else's pages (`guest=refused`). The cross-pass
/// resident Load asks it to keep somebody else's pixels from being loaded as
/// this draw's own prior content (`resident=refused`) — the same drift, read
/// from the other side. One hardcoded outcome word would make one line a lie.
#[cfg(feature = "backend-vulkan")]
#[allow(
    clippy::too_many_arguments,
    reason = "the drift question names the window, its armed pages, and what the caller loses"
)]
pub(crate) fn deferred_pages_still_ours<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    gva: u64,
    span: u64,
    armed: &std::collections::HashSet<u64>,
    what: &str,
    outcome: &str,
) -> bool {
    if span == 0 || armed.is_empty() {
        return true;
    }
    let mut live = std::collections::HashSet::new();
    crate::runtime::gva_mem::visit_task_gva_page_gpas(
        host,
        &state.tasks,
        task_id,
        gva,
        span,
        state.page_shift,
        1,
        &mut |gpa_page| {
            live.insert(gpa_page);
            true
        },
    );
    // The property that makes the write safe is not "the same number of pages
    // came back", it is "every page this write can reach is one the window was
    // given". `write_gva_rgba8` resolves the destination per row from a fresh
    // walk, so the pages it can reach are exactly the ones this walk resolves —
    // and a page of `live` that is not in `armed` is a page some other owner
    // holds now.
    //
    // A subset is the benign teardown case: the guest dropped part of the range
    // and the rest is still ours, so the rows that still resolve land in our own
    // pages and the rest fail per-row on their own terms. That is what the
    // length test was reaching for, and it is not what it tested — `live` can be
    // shorter than `armed` while containing pages that were never ours, because
    // pages can disappear and reappear pointing somewhere else in the same walk.
    // The strictly-shorter arm returned "still ours" for that case, which is the
    // one arrangement of this range that corrupts another owner's memory.
    if live.iter().all(|p| armed.contains(p)) {
        return true;
    }
    crate::observe::fail(format!(
        "deferred_window_page_drift gva={gva:#x} task={task_id} {what} \
         armed_pages={} live_pages={} moved={} foreign={} {outcome}",
        armed.len(),
        live.len(),
        armed.difference(&live).count(),
        live.difference(armed).count()
    ));
    false
}

/// Land every armed GVA render-Store window, because the guest is about to be
/// told the work is finished.
///
/// This is the deferral rail's contract with the guest, and it is the one thing
/// [`deferred_pages_still_ours`] cannot substitute for. A completion stamp is
/// this device's only statement that a render is done; from the instant it lands
/// the guest may free the target, and its own allocator may hand those pages to
/// anything at all without touching a page table — so no later walk, page-set
/// comparison or content test can tell the memory apart from the target it used
/// to be. The only sound moment to write a render's bytes into guest RAM is
/// before the fence that claims they are already there.
///
/// Apple's device needs no equivalent because it has no equivalent window: the
/// render target *is* the guest allocation, so completion and "the bytes are in
/// guest memory" are the same event. This is that invariant restated for a rail
/// that has to copy.
///
/// What the deferral still buys is everything inside one fence: a chain of
/// passes rendering into the same target reuses the registry resident, and
/// `supersede_gva_window` still drops a window the same submission re-renders.
/// What it stops buying is survival across the fence, which was never the
/// device's to sell.
///
/// `REIMS_VGPU_FENCE_FLUSH_OFF=gva` (or `=1`/`=all`) restores the old unbounded
/// behaviour so an arm and its control can be measured on one binary.
#[cfg(feature = "backend-vulkan")]
pub fn flush_gva_windows_before_fence<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
) {
    if state.gva_deferred_flush.is_empty()
        || crate::observe::fence_flush_disabled(crate::observe::FenceFlushRail::Gva)
    {
        return;
    }
    // Oldest-first, so windows land in the order they were rendered: a later
    // Store at an address the guest recycled within one submission must not be
    // overwritten by the earlier one.
    while let Some((gva, entry)) = state.take_oldest_gva_deferred_window() {
        crate::runtime::drain::note_store_route("gvaw_fence_flush");
        let _ = flush_gva_one(state, host, gva, &entry, true, "fence");
    }
}

/// Metal-direct builds never arm GVA windows — nothing to land at the fence.
#[cfg(not(feature = "backend-vulkan"))]
pub fn flush_gva_windows_before_fence<M: HostMemory + HostOps>(
    _state: &mut DeviceState,
    _host: &mut M,
) {
}

/// Land every armed linear compute-storage window, for the same reason and under
/// the same contract as [`flush_gva_windows_before_fence`].
///
/// This rail writes a raw task GVA. `ComputeStorageResidencyKey::linear` sets
/// `mapping_id` to 0 and stores the *task id* in `map_generation`, so there is no
/// mapping incarnation to compare and no lifecycle notify anywhere in the wire
/// format — exactly the position the GVA render rail is in, and exactly why
/// `6bc2220` could clear `flush_render_one` and `flush_storage_one` on
/// `map_generation` drift and could not clear this one.
///
/// # Measured before it was repaired
///
/// One x86/Vulkan boot on the crash-hunt workload (Safari on three compositing
/// pages, Finder windows, then 600 s of Mission Control ×71, Spotlight ×71,
/// window drags ×142):
///
/// ```text
/// linw_stamp_same       0
/// linw_stamp_outlived   1     task=5 ref=52 gva=0x39f000 128x135 stamps=1019
/// ```
///
/// Both halves matter. The rail is late whenever it lands at all — the one
/// landing in ten minutes came 1 019 fences after the guest was told the work was
/// done. And it lands almost never, which is what makes the repair free: the
/// objection that stopped the fence repair from being applied to the render rail
/// was the cost of writing back full-screen frames ~98 % of which nothing reads,
/// and there is no such cost here. One window per ten minutes is not a writeback
/// budget.
///
/// A rate this low cannot on its own convict this rail of any guest crash, and no
/// such claim is made. What it does mean is that the correct behaviour is also
/// the cheap one, so there is nothing to trade.
///
/// `REIMS_VGPU_FENCE_FLUSH_OFF=linear` (or `=1`/`=all`) restores the unbounded
/// behaviour here, so an arm and its control stay one binary apart.
#[cfg(feature = "backend-vulkan")]
pub fn flush_linear_windows_before_fence<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
) {
    if state.linear_deferred_flush.is_empty()
        || crate::observe::fence_flush_disabled(crate::observe::FenceFlushRail::Linear)
    {
        return;
    }
    // Snapshot the keys first: `flush_linear_one` disarms its own window and may
    // flush others through the cache paths below it, so iterating the live map
    // would borrow it across a mutation. A key whose window is gone by the time
    // it comes up disarms to `None` and the flush is a no-op on the guest.
    let armed: Vec<(crate::model::ComputeStorageResidencyKey, u32)> = state
        .linear_deferred_flush
        .iter()
        .map(|(key, entry)| (*key, entry.generation))
        .collect();
    for (key, generation) in armed {
        if !state.linear_deferred_flush.contains_key(&key) {
            continue;
        }
        crate::runtime::drain::note_store_route("linw_fence_flush");
        let _ = flush_linear_one(state, host, &key, generation);
    }
}

/// Metal-direct builds never arm linear windows — nothing to land at the fence.
#[cfg(not(feature = "backend-vulkan"))]
pub fn flush_linear_windows_before_fence<M: HostMemory + HostOps>(
    _state: &mut DeviceState,
    _host: &mut M,
) {
}

/// Land every armed mapping-keyed window — type-11 render Stores and compute
/// storage alike — because the guest is about to be told the work is finished.
///
/// This is the last of the four deferred rails to be bound to the fence, and it
/// is bound for a *different* reason from the other three, which is why it was
/// measured first rather than assumed.
///
/// # The other three rails were bound because they could not name their memory
///
/// A `GvaDeferredEntry` and a `ComputeStorageResidencyKey::linear` name a raw
/// address, so a guest that frees the allocation and reuses the pages leaves the
/// window pointing at somebody else's memory with nothing to refuse on. That is
/// not this rail's position: [`flush_render_one`] and [`flush_storage_one`]
/// compare the mapping's live `map_generation` against `key.map_generation` and
/// refuse before reading, and `map_generation` moves on exactly the events that
/// let a guest reuse an IOSurface's storage. [`note_mapping_window_against_fence`]
/// records that argument in full and still holds.
///
/// # This rail is bound because the guest is entitled to the bytes at the fence
///
/// A completion stamp is this device's statement that the render is finished. A
/// guest that has been told so may map the IOSurface and read it — CoreGraphics
/// reading back a layer, a damage forward-copy from the previous buffer, any
/// CPU-side compositing step — and it reads *guest RAM*, through its own
/// mapping, without crossing a single host path this device can intercept.
/// `flush_intersecting` covers every reader that goes through us and there is no
/// mechanism that covers the ones that do not.
///
/// So a deferred window is a bet that nothing reads those pages before we land
/// them, and when the bet loses the guest composites the *pre-Store* bytes: a
/// region of the surface holding whatever was there one frame ago, or nothing at
/// all. That is a stale rectangle in an otherwise correct frame, and it is
/// indistinguishable from the corruption classes this device is chasing.
///
/// Apple's device does not take that bet and does not need to. Its render target
/// *is* the guest allocation, so "the render is complete" and "the bytes are in
/// guest memory" are one event. This is that invariant restated for a rail that
/// has to copy: the copy happens before the statement, not after it.
///
/// # And because it clobbers writes the guest itself made
///
/// A deferred window promises to replay a Store later, and that is only a replay
/// while nothing else writes those pages in between. The guest *is* something
/// else: it maps the same IOSurface and does inter-buffer damage forward-copies
/// and CoreGraphics blits into it. The writeback covers the full attachment
/// extent, so every such guest store inside the deferral interval is gone when
/// the window lands. One x86/Vulkan boot on the icon workload (Safari + Finder,
/// 300 s of Mission Control ×41 / Spotlight ×41 / window drags ×82, then four
/// Finder recomposite rounds) measured that directly:
///
/// ```text
/// surface_resident               49 706
/// surface_flush                  12 343    windows that landed
/// render_flush_over_guest_write   8 968    of those, 73 % clobbered guest bytes
/// rendw_stamp_outlived           12 343    every one landed after the fence
/// storw_stamp_outlived              101
/// ```
///
/// `deferred_flush_clobber` is 8 975 lines of that boot's fail log — the largest
/// self-declared loss of guest work anywhere in it.
///
/// [`render_flush_guest_written_ranges`] states why the obvious repair —
/// preserve the pages the guest wrote — is not available: `page_gen[p]` is
/// stamped at the *harvest* that saw page `p` dirty, not at the write, so the
/// witness cannot say whether a store happened before or after the Store this
/// window defers. Preserving on it withheld the device's own frames and turned
/// the screen black (`13ae46d`, 0 of 14 rounds).
///
/// The fence deletes the question rather than answering it. A window that lands
/// before [`crate::runtime::drain::write_stamp`] covers only the interval a
/// synchronous Store would itself have covered, so there is no interval left in
/// which a guest write can be both after the Store and before the writeback.
/// Nothing has to be preserved because nothing is clobbered.
///
/// # What the deferral still buys, and what it stops buying
///
/// Everything inside one fence survives: a chain of passes into the same surface
/// still reuses one resident, and `supersede_covered_render_windows` still drops
/// a window a later Store in the same submission fully covers. What it stops
/// buying is survival *across* the fence, and that is where this rail's cost is,
/// because unlike the linear rail it is not free.
///
/// `arm_surface_resident_store` exists to skip the whole-framebuffer GPU→host
/// readback entirely on the ~86 % of windows nothing ever flushes — `draw_phase`
/// prices that skip at 565 ms per second of wall clock. Landing every window at
/// its fence pays a readback for each: `surface_resident 49 706` against
/// `surface_flush 12 343` bounds it at 4× the current landings. That is the trade
/// this binding makes, and it is a trade rather than a regression only if the
/// measurement says so, so it is measured on one binary under
/// `REIMS_VGPU_FENCE_FLUSH_OFF=mapping` with `present_hz` and `draw_us` read both
/// ways.
///
/// The GVA rail's binding was expected to cost frame rate and paid back instead
/// (5.9 → 9.5 Hz, `draw_us` 524 ms → 156 ms), because the unbounded rail spent
/// its time in oldest-first `window_cap` eviction storms holding residents pinned
/// across hundreds of frames. `evict_render_windows_to_cap` is the same shape and
/// may go the same way, but that is a prediction and not a reading.
///
/// The endgame removes the trade rather than choosing a side of it: a resident
/// whose image memory *is* the guest pages has nothing to write back, which is
/// why Apple's device has neither this rail nor this cost. That is a backend
/// allocation change, not a scheduling one.
///
/// # Ordering
///
/// Render windows first in arm order, then whatever remains, and both through
/// [`flush_intersecting`] rather than by taking entries directly. That choke
/// point runs the fixpoint that drags in every sibling overlapping the same guest
/// bytes, so windows that overlap land together in one pass whatever order this
/// loop reaches them in — the ordering here decides only which *disjoint* window
/// goes first, and disjoint windows cannot overwrite each other.
///
/// A window may legitimately survive: `flush_intersecting` holds every window on
/// a condemned backing so `mapper::resolve` can settle whether the delete named
/// this incarnation. That hold is the existing contract and the fence does not
/// override it — such a window is not owed to guest RAM until the resolve says
/// the memory is still ours.
///
/// `REIMS_VGPU_FENCE_FLUSH_OFF=mapping` restores the unbounded behaviour on this
/// rail *only*. That matters more here than on the other two: the GVA and linear
/// bindings are already measured and the icon verdict depends on them, so an
/// `=1` control would price this rail against a control that had also given back
/// two repairs. `=1`/`=all` still revert every rail, for a bisection that wants
/// the pre-fence device back.
#[cfg(feature = "backend-vulkan")]
pub fn flush_mapping_windows_before_fence<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
) {
    if state.compute_deferred_flush.is_empty()
        || crate::observe::fence_flush_disabled(crate::observe::FenceFlushRail::Mapping)
    {
        return;
    }
    // Snapshot first: landing one window consumes its overlapping siblings
    // through the fixpoint, so iterating the live map would borrow it across a
    // mutation. A key already consumed by an earlier pass is skipped rather than
    // re-flushed.
    for key in mapping_windows_fence_order(state) {
        if !state.compute_deferred_flush.contains_key(&key) {
            continue;
        }
        crate::runtime::drain::note_store_route("mapw_fence_flush");
        flush_intersecting(state, host, key.mapping_id, key.surface_offset, key.span_end);
    }
}

/// Metal-direct builds never arm mapping-keyed windows — nothing to land.
#[cfg(not(feature = "backend-vulkan"))]
pub fn flush_mapping_windows_before_fence<M: HostMemory + HostOps>(
    _state: &mut DeviceState,
    _host: &mut M,
) {
}

/// The order [`flush_mapping_windows_before_fence`] lands windows in: render
/// windows oldest-first by `armed_seq`, then every other window.
///
/// Only the render rail carries an arm sequence, and only the render rail can
/// hold several live windows on one mapping at once (different planes, different
/// geometries at the same offset). Compute storage windows are keyed by the
/// dispatch span that produced them and are appended in key order, which is the
/// order every other flush trigger has always used.
#[cfg(feature = "backend-vulkan")]
fn mapping_windows_fence_order(
    state: &DeviceState,
) -> Vec<crate::model::ComputeStorageResidencyKey> {
    let mut render: Vec<(u64, crate::model::ComputeStorageResidencyKey)> = Vec::new();
    let mut rest: Vec<crate::model::ComputeStorageResidencyKey> = Vec::new();
    for (key, owner) in &state.compute_deferred_flush {
        match owner {
            crate::model::DeferredOwner::Render { armed_seq, .. } => render.push((*armed_seq, *key)),
            crate::model::DeferredOwner::Storage { .. } => rest.push(*key),
        }
    }
    render.sort_unstable_by_key(|(seq, _)| *seq);
    render
        .into_iter()
        .map(|(_, key)| key)
        .chain(rest)
        .collect()
}

/// Score a deferred window about to write guest RAM against the guest's fence.
///
/// [`crate::runtime::drain::write_stamp`] is the only thing this device says to
/// the guest about whether work is finished. Once it has moved, the guest is
/// entitled to free everything it allocated for that work — and the guest's own
/// allocator is then free to hand those pages to anything, without touching a
/// page table. So a window armed at stamp N and landed at stamp N+k, k > 0, is a
/// write to memory the guest was told it could reclaim k fences ago.
///
/// [`deferred_pages_still_ours`] cannot see this. It asks whether the GVA still
/// resolves to the pages the window was armed on, and free-then-reuse inside one
/// process preserves the translation exactly. That is why the guard landed and
/// the WindowServer `small_free_list_remove_ptr_no_clear` aborts continued.
///
/// The counters carry their own denominator — `gvaw_stamp_same` against
/// `gvaw_stamp_outlived` in the per-second `store_routes` line.
///
/// # Measured, and it is not a tail
///
/// One x86/Vulkan boot driving the workload the user's report names (Safari on
/// three compositing-heavy pages, Finder windows, then 600 s of Mission Control
/// ×71, Spotlight ×71 and window drags ×142 — every one of them a window-list
/// capture compositing a backdrop blur, which is the frame the report crashed
/// in):
///
/// ```text
/// gvaw_stamp_same       0
/// gvaw_stamp_outlived 810
/// ```
///
/// **Zero.** Not a minority, not a tail: every deferred GVA window that wrote
/// guest RAM on that boot wrote it after the guest had been fenced. The elapsed
/// stamp counts say how far after — over 227 latched spans, median 133 fences,
/// p90 1 099, max 1 601. The guest was told this work had finished 133 times
/// over before the device put the bytes in its memory.
///
/// The trigger breakdown says why: 215 of 227 land under `window_cap`, the
/// oldest-first eviction that runs when `GVA_DEFERRED_WINDOW_CAP` is reached. So
/// the rail's normal exit is not a flush anything asked for; it is a window
/// sitting until the cap pushes it out, hundreds of fences past the point the
/// guest was free to reclaim it.
///
/// And the geometry names the second defect as well as the first. The largest
/// single population is **64x64, 65 of 227** — a folder icon exactly, the same
/// geometry the surviving Finder icon class corrupts at. The icons that come out
/// wrong are the windows written into guest memory long after the guest was told
/// they were done.
///
/// No userspace crash fired during those 600 s, so this boot does not by itself
/// convict the rail of the WindowServer abort. What it establishes is that the
/// hazard is not rare, not a corner, and not something a page-set guard can see.
///
/// # After the repair, on the same harness
///
/// [`flush_gva_windows_before_fence`] inverts it completely:
///
/// ```text
///                      before repair   after repair
/// gvaw_stamp_same                  0         54 932
/// gvaw_stamp_outlived            810              0
/// ```
///
/// Every landing is now inside the fence that completes it, and
/// `gvaw_fence_flush` equals `gva_deferred` exactly — every window armed is a
/// window landed at the next stamp, which is the whole of the deferral the
/// contract permits.
///
/// The cost was expected to be a frame-rate loss and was the opposite. Same
/// harness, same 600 s drive, mean over ~510 one-second windows:
///
/// ```text
///                 before repair   after repair
/// present_hz                5.9            9.5
/// draw_us              523 895        156 294
/// ```
///
/// Two boots are not a benchmark and load varies, but the direction is not
/// subtle and it has a mechanism: 215 of 227 landings used to come out under
/// `window_cap`, so the old rail spent its time in oldest-first eviction storms
/// while holding residents pinned across hundreds of frames. Landing at the
/// fence keeps the window set nearly empty and the pin churn with it.
///
/// The crash itself is still unscored. `.agents/repros/crash-hunt.sh` has never
/// fired the abort in either arm, so it gates the census and not the class.
#[cfg(feature = "backend-vulkan")]
fn note_window_outlived_its_stamp(
    state: &DeviceState,
    gva: u64,
    entry: &crate::model::GvaDeferredEntry,
    trigger: &str,
) {
    let elapsed = state
        .completion_stamp_seq
        .wrapping_sub(entry.armed_stamp_seq);
    if elapsed == 0 {
        crate::runtime::drain::note_store_route("gvaw_stamp_same");
        return;
    }
    crate::runtime::drain::note_store_route("gvaw_stamp_outlived");
    // Identity, latched per span+trigger: the count says how often, and this
    // says which windows and which door they came through. A rail that only
    // ever outlives its stamp under one trigger is a different repair from one
    // that does it everywhere.
    if crate::observe::first_sight(
        "gva_window_outlived_stamp",
        gva ^ ((entry.width as u64) << 32) ^ entry.height as u64,
    ) {
        crate::observe::fail(format!(
            "gva_window_outlived_stamp gva={gva:#x} task={} {}x{} trigger={trigger} \
             stamps={elapsed} (guest was fenced before these bytes were written)",
            entry.task_id, entry.width, entry.height
        ));
    }
}

/// Score a deferred **linear compute-storage** landing against the guest's fence.
///
/// [`note_window_outlived_its_stamp`] is the same reading for the GVA render
/// rail, and the hazard is identical because the identity is identical: a
/// `ComputeStorageResidencyKey::linear` names a task and an address
/// (`mapping_id` 0, `map_generation` carrying the task id), so nothing the guest
/// does to reclaim the memory reaches this rail as a notification.
///
/// That distinction is why `6bc2220` cleared the other two deferred rails and
/// cannot clear this one. `flush_render_one` and `flush_storage_one` refuse on
/// `map_generation` drift, and `map_generation` moves on exactly the events that
/// let a guest reuse an IOSurface's storage. This rail has no such generation to
/// compare — [`deferred_pages_still_ours`] is its only guard, and free-then-reuse
/// inside one process preserves the translation the guard reads.
///
/// The rail's own flush already records what that costs when it goes wrong:
/// a `pmap_page_protect` kernel panic and userspace SIGSEGVs inside libmalloc's
/// page bookkeeping. What was missing is how often the landing is late at all,
/// which is what `linw_stamp_same` against `linw_stamp_outlived` says.
#[cfg(feature = "backend-vulkan")]
fn note_linear_window_outlived_its_stamp(
    state: &DeviceState,
    key: &crate::model::ComputeStorageResidencyKey,
    window: &crate::model::LinearDeferredEntry,
) {
    let elapsed = state
        .completion_stamp_seq
        .wrapping_sub(window.armed_stamp_seq);
    if elapsed == 0 {
        crate::runtime::drain::note_store_route("linw_stamp_same");
        return;
    }
    crate::runtime::drain::note_store_route("linw_stamp_outlived");
    if crate::observe::first_sight(
        "linear_window_outlived_stamp",
        key.surface_offset ^ ((key.width as u64) << 32) ^ key.height as u64,
    ) {
        crate::observe::fail(format!(
            "linear_window_outlived_stamp task={} ref={} gva={:#x} {}x{} stamps={elapsed} \
             (guest was fenced before these bytes were written)",
            key.map_generation, key.texture_ref, key.surface_offset, key.width, key.height
        ));
    }
}

/// Engine-resident identity a deferred GVA window is holding pinned.
///
/// Rebuilt from the window's own fields — including the
/// [`crate::model::GvaDeferredEntry::alloc_gen`] the arming draw resolved —
/// rather than from a fresh page walk. The window exists because the guest may
/// hand the address to another allocation before the flush runs; a walk taken
/// now would name that allocation, the registry lookup would miss the slot this
/// window pinned, and the deferred frame would be lost instead of landing.
///
/// Single spelling for every consumer that starts from a window
/// ([`flush_gva_one`], `metal_draw::vulkan::supersede_gva_window`,
/// `metal_draw::vulkan::try_sample_deferred_gva`) so the three cannot drift
/// apart from the producer or from each other.
#[cfg(feature = "backend-vulkan")]
pub fn gva_window_identity(
    gva: u64,
    entry: &crate::model::GvaDeferredEntry,
) -> crate::backend::vulkan::engine::TargetIdentity {
    crate::backend::vulkan::engine::TargetIdentity::Gva {
        gva,
        width: entry.width,
        height: entry.height,
        generation: entry.alloc_gen,
    }
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
    let started = std::time::Instant::now();
    let identity = gva_window_identity(gva, entry);
    // `into_rgba8` rather than the raw bytes: a GVA resident is RGBA today, so
    // this is a no-op, but the writer below (`write_gva_rgba8`) is declared in
    // semantic RGBA and the readback states its own order. Asserting the order
    // here instead would be the caller writing a fact it did not read.
    let rgba = match crate::backend::vulkan::engine::read_target(&identity) {
        Ok(rb) => rb.into_rgba8(),
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
    if guest_write {
        // Reported, not skipped. This used to be `skip_uncovered`, which dropped
        // the whole deferred window — a full compositing layer — whenever the
        // guest had not yet notified the allocation, and it fired on live boots
        // for 240x135 and 320x512 surfaces. `MapMemory2` arrives after the guest
        // has installed the PTEs and used the memory; see `WriteGate`.
        crate::runtime::gva_mem::report_undeclared_write(
            state,
            host,
            entry.task_id,
            gva,
            entry.span(),
            "gva_deferred_flush",
        );
    }
    if guest_write {
        note_window_outlived_its_stamp(state, gva, entry, trigger);
    }
    if guest_write && !window_pages_still_ours(state, host, gva, entry, trigger, "guest=refused") {
        // The window's pages moved under us. Cache-only: see
        // `window_pages_still_ours` for why writing here lands in another
        // owner's memory. This is the REPORT — it walks every page of the window
        // against the pages it was armed on and names the event with counts a
        // reader can score. The BOUND is `Some(&entry.pages)` below, which the
        // writer's own walk enforces; a decision taken before a second walk is
        // a decision about a page table the bytes do not go through.
        guest = "skip_drift";
    } else if guest_write {
        guest = match crate::runtime::metal_draw::write_gva_rgba8_within(
            state,
            host,
            entry.task_id,
            gva,
            entry.width,
            entry.height,
            entry.row_stride,
            entry.format,
            &rgba,
            Some(&entry.pages),
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
    }
    crate::runtime::metal_draw::host_cache_store_gva_layer(
        state,
        host,
        entry.task_id,
        entry.texture_ref,
        entry.producer_object_type,
        gva,
        entry.width,
        entry.height,
        &rgba,
    );
    // A flush that landed is expected control flow and stays quiet. The two
    // outcomes that are not — a refused write, and a window whose span the guest
    // had already torn down — each emit their own typed line above, so the
    // always-on view keeps the losses and drops the running commentary.
    crate::runtime::drain::note_drain_phase(crate::runtime::drain::DrainPhase::Flush, started);
    crate::observe::line(format!(
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
    let window = state.disarm_linear_deferred_window(key);
    let armed_pages = window.as_ref().map(|w| w.pages.clone());
    let task_id = key.map_generation;
    let texture_ref = key.texture_ref;
    let started = std::time::Instant::now();
    if let Some(window) = window.as_ref() {
        note_linear_window_outlived_its_stamp(state, key, window);
    }
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

    // Same hazard, same answer as the GVA rail: this window was armed against a
    // page set at defer time and `write_linear_guest` walks fresh, so a span the
    // guest has since re-pointed sends a compute-storage image into whatever
    // owns those pages now. Observed on this rail as guest heap corruption — a
    // `pmap_page_protect` kernel panic and userspace SIGSEGVs inside libmalloc's
    // own page bookkeeping. The cache entry keeps the authoritative bytes
    // (`materialize_linear_resident` ran above), so refusing loses nothing
    // renderable.
    let still_ours = match &armed_pages {
        // `span_end` is a length (`row_stride * height`) for a linear key, not
        // an end address — and the arm site walks `(surface_offset, span_end)`
        // with exactly these two values, so this walk has to as well or the two
        // page sets describe different ranges and every flush reads as drift.
        Some(pages) => deferred_pages_still_ours(
            state,
            host,
            task_id,
            key.surface_offset,
            key.span_end,
            pages,
            &format!(
                "{}x{} trigger=linear_flush ref={texture_ref}",
                key.width, key.height
            ),
            "guest=refused",
        ),
        None => true,
    };
    // Both arms assign, so this is the whole set of outcomes this rail can
    // report — `skip_uncovered` was the third and is gone.
    let guest = if !still_ours {
        "skip_drift"
    } else {
        // `skip_uncovered` used to live on this branch and discarded the whole
        // linear writeback — up to 1.3 MiB of texture — when the guest had not
        // yet notified the allocation. Reported and written now; see
        // `report_undeclared_write`.
        crate::runtime::gva_mem::report_undeclared_write(
            state,
            host,
            task_id,
            key.surface_offset,
            key.span_end,
            "linear_deferred_flush",
        );
        // Same bound as the GVA rail: the armed page set travels into the
        // writer's own walk, so the decision `still_ours` reached above cannot be
        // invalidated by the guest between that walk and this one. `None` here
        // would be a window with no armed pages, which is a window this rail
        // never bounded in the first place.
        match crate::runtime::compute_exec::write_linear_guest_within(
            state,
            host,
            task_id,
            key.surface_offset,
            key.surface_bpr as u64,
            tight,
            key.height,
            &bytes,
            &format!("flush ref={texture_ref}"),
            armed_pages.as_ref(),
        ) {
            crate::runtime::compute_exec::LinearWrite::Written => "written",
            // Nothing resolves at this GVA, so there is no guest memory to land
            // in. Distinct from `write_fail`, which means a write was attempted:
            // one is the guest having taken the pages away, the other is ours.
            crate::runtime::compute_exec::LinearWrite::Unmapped => "skip_unmapped",
            // The per-row failure is already fail-logged; the cache entry keeps
            // the coherent authoritative bytes.
            crate::runtime::compute_exec::LinearWrite::Failed => "write_fail",
        }
    };
    crate::runtime::drain::note_drain_phase(crate::runtime::drain::DrainPhase::Flush, started);
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
        if state.disarm_linear_deferred_window(key).is_some() {
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

/// Land one taken mapping-keyed window, dispatching on which rail holds its
/// pixels. The key names the guest side identically for both; only the read
/// differs (see [`crate::model::DeferredOwner`]).
fn flush_one<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    key: &crate::model::ComputeStorageResidencyKey,
    owner: crate::model::DeferredOwner,
) -> bool {
    note_mapping_window_against_fence(state, key, &owner);
    match owner {
        crate::model::DeferredOwner::Storage {
            generation,
            armed_stamp_seq: _,
        } => flush_storage_one(state, host, key, generation),
        crate::model::DeferredOwner::Render { source, .. } => {
            flush_render_one(state, host, key, &source)
        }
    }
}

/// Score a mapping-keyed deferred window against the guest's fence, exactly as
/// [`note_window_outlived_its_stamp`] scores the GVA rail.
///
/// Counted at the flush dispatcher rather than at each writer, so the two rails
/// share one denominator; whether a landing actually reached guest RAM is what
/// the existing `deferred_flush_*` lines already say.
///
/// # What this counter did NOT settle, and what did
///
/// The reading that made this rail worth measuring separately still stands. One
/// 14-round x86/Vulkan icon boot:
///
/// ```text
/// rendw_stamp_same    0     rendw_stamp_outlived 1088
/// storw_stamp_same    0     storw_stamp_outlived   24
/// elapsed over 217 latched spans: min 1, p50 66, p90 2551, max 17086
/// ```
///
/// Read as a counter that looks exactly like the GVA rail's 810-of-810, and it
/// does not mean the same thing. **The counter is not the hazard.** Outliving
/// the fence corrupts memory only if the guest can repurpose that memory without
/// the device finding out, and on these rails it cannot:
///
/// - [`flush_render_one`] and [`flush_storage_one`] both compare the mapping's
///   live `map_generation` against `key.map_generation` and refuse with
///   `deferred_flush_lost reason=map_generation_drift` before reading anything.
/// - `map_generation` is bumped by exactly the events that let the guest reuse
///   an IOSurface's storage — MAP, UNMAP, `ReplacePhysical`, MappingInternal
///   reattach, any page-table refresh that changes PFNs.
/// - A `DeleteIOSurfaceBacking2` that has not yet resolved leaves the backing
///   *condemned*, and [`flush_intersecting`] refuses to take those windows at
///   all.
///
/// So these windows name a specific mapping incarnation, and a guest that frees
/// the storage invalidates the name. That is precisely the allocation identity
/// the GVA rail did not have and could not be given: a type-2/3 target is a
/// texture handle shifted into an address, with no lifecycle notify anywhere in
/// the wire format, so `deferred_pages_still_ours` was the only guard available
/// and page identity survives free-then-reuse.
///
/// This rail is nonetheless bound to the fence now
/// ([`flush_mapping_windows_before_fence`]) — for the *other* hazard, which this
/// counter cannot see and `render_flush_over_guest_write` can: the guest holds
/// the same IOSurface mapped and writes it, and a full-extent writeback landing
/// later replaces what it wrote. 8 968 of 12 343 landings on one measured boot.
/// The free-then-reuse argument above is untouched by that and is still the
/// reason this rail needed its own evidence instead of the GVA rail's.
///
/// These counters stay as the standing check on the `map_generation` guard, and
/// as the reading of how much deferral the binding actually removed: with the
/// fence drain wired, `rendw_stamp_same` should carry the traffic and
/// `rendw_stamp_outlived` should fall to the windows a condemned backing holds.
fn note_mapping_window_against_fence(
    state: &DeviceState,
    key: &crate::model::ComputeStorageResidencyKey,
    owner: &crate::model::DeferredOwner,
) {
    let rail = match owner {
        crate::model::DeferredOwner::Storage { .. } => "storage",
        crate::model::DeferredOwner::Render { .. } => "render",
    };
    let elapsed = state
        .completion_stamp_seq
        .wrapping_sub(owner.armed_stamp_seq());
    if elapsed == 0 {
        crate::runtime::drain::note_store_route(match rail {
            "storage" => "storw_stamp_same",
            _ => "rendw_stamp_same",
        });
        return;
    }
    crate::runtime::drain::note_store_route(match rail {
        "storage" => "storw_stamp_outlived",
        _ => "rendw_stamp_outlived",
    });
    if crate::observe::first_sight(
        "mapping_window_outlived_stamp",
        u64::from(key.mapping_id) ^ ((key.width as u64) << 32) ^ key.height as u64,
    ) {
        crate::observe::fail(format!(
            "mapping_window_outlived_stamp rail={rail} mapping={} {}x{} stamps={elapsed} \
             (guest was fenced before these bytes were written)",
            key.mapping_id, key.width, key.height
        ));
    }
}

/// Land a deferred **type-11 render Store**: perform the CPU writeback into the
/// mapping's guest pages that the Store itself skipped.
///
/// The pixels come from `surface_cache`, not from the engine. The Store read
/// its target back as it always did and refreshed the cache with that frame
/// before arming; only the guest-page copy was deferred. That is deliberate and
/// it is what keeps this rail small: the engine resident for a type-11 surface
/// is not authoritative here, so nothing has to be pinned, no `content_ready`
/// has to hold across frames, and the Load seed and present capture keep
/// reading exactly what they read before.
///
/// Deferring is a win rather than a rescheduling because nothing on the
/// host-window present path reads these guest pages — `capture_present_frame`
/// takes the cache or the resident and states in situ that it "never touches
/// guest memory" — so the writeback is owed only to a guest-side reader that
/// may never come.
/// The engine resident a [`crate::model::RenderWindowSource::Resident`] window
/// pinned, rebuilt from the key.
///
/// Not stored on the window, for the same reason `flush_gva_one` rebuilds its
/// own: the key already carries every term of the identity, and two spellings of
/// one value are two things that can disagree. `key.map_generation` is the field
/// `present_identity::surface_identity` keys on, and the flush refuses on
/// generation drift before it reads anything, so the rebuild is always for the
/// generation the arm pinned.
#[cfg(feature = "backend-vulkan")]
fn render_window_identity(
    key: &crate::model::ComputeStorageResidencyKey,
) -> crate::backend::vulkan::engine::TargetIdentity {
    crate::backend::vulkan::engine::TargetIdentity::Surface {
        id: key.mapping_id,
        width: key.width,
        height: key.height,
        generation: key.map_generation as u64,
    }
}

/// Report what a landing window is about to overwrite, and preserve none of it.
///
/// A deferred window promises to replay a synchronous Store later, and that is
/// only a replay while nothing else writes the pages in between. The writeback
/// covers the whole attachment extent, so a guest CPU store into any page of it
/// — an inter-buffer damage forward-copy, a CoreGraphics blit into the same
/// IOSurface — is gone the moment this window lands. Nothing else in the flush
/// can see that: `map_generation` covers a rebind, `resident_content_epoch`
/// covers a later device draw, and neither is a witness for the surface's own
/// owner. One 14-round composite boot measured `render_flush_over_guest_write`
/// at 68 of every 99 `surface_flush`es.
///
/// This rail did preserve those pages, and it must not, because the witness it
/// would preserve them on cannot answer the question it is being asked.
///
/// `page_gen[p]` is stamped with the generation at the *harvest* that saw page
/// `p` dirty, not at the write. `reims_vgpu_dirty_harvest` returns early when
/// nothing has read a generation since the last one, and does not clear the
/// bitmap when it does, so a guest store can sit unharvested across a Store and
/// be attributed to the generation of a harvest that ran after it. Every such
/// page is then "written since the Store" when the device's own render
/// superseded it, and preserving it withholds the frame from guest memory.
///
/// Bisected on the live rail, x86 / Vulkan, four `icon-composite` rounds each,
/// one binary per arm:
///
/// ```text
/// 22a3346  preserve absent   3 of 4 rounds clean, desktop paints
/// 8178caa  preserve absent   2 of 4 rounds clean, desktop paints
/// 13ae46d  preserve present  0 of 14 rounds, screen black, 19 Hz
/// ```
///
/// So the answer this rail reaches for is the right one and the evidence it
/// would reach for it on is not sound. A full-extent landing that reports what
/// it replaced is strictly better than a partial one that silently withholds the
/// device's frame.
///
/// The ordering repair is what actually removes the loss, and it is upstream of
/// this question rather than an answer to it:
/// [`flush_mapping_windows_before_fence`] lands every armed window before the
/// guest is told the work is done, so the interval in which a guest store can be
/// both after the Store and before the writeback does not exist. Nothing needs
/// preserving because nothing is clobbered, and this function becomes the
/// standing check on that — a `render_flush_over_guest_write` after the binding
/// names a window that landed outside the fence anyway, which is a defect and
/// not a cost.
///
/// [`crate::runtime::mapping_write::write_bgra8_skipping`] and
/// `HostOps::guest_written_pages` stay: the sampled ladder's merge uses both,
/// and it errs the other way — it keeps both halves rather than choosing.
#[cfg(feature = "backend-vulkan")]
fn render_flush_guest_written_ranges<M: HostOps>(
    state: &DeviceState,
    host: &M,
    key: &crate::model::ComputeStorageResidencyKey,
) -> Vec<(u64, u64)> {
    use crate::runtime::mapper::{mapping_guest_write_verdict, GuestWriteVerdict};
    if mapping_guest_write_verdict(state, host, key.mapping_id) != GuestWriteVerdict::Wrote {
        return Vec::new();
    }
    crate::runtime::drain::note_store_route("render_flush_over_guest_write");
    crate::observe::fail(format!(
        "deferred_flush_clobber kind=render mapping={} {}x{} fmt={:#x} gen={} \
         (the guest wrote pages of this surface after the Store this window defers; \
         the full-extent writeback replaces them)",
        key.mapping_id, key.width, key.height, key.pixel_format, key.map_generation
    ));
    Vec::new()
}

#[cfg(feature = "backend-vulkan")]
fn flush_render_one<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    key: &crate::model::ComputeStorageResidencyKey,
    source: &crate::model::RenderWindowSource,
) -> bool {
    let started = std::time::Instant::now();
    // Counted on the same one-line-per-second census as the Store routes, so a
    // boot reads `surface_deferred=N surface_flush=M` on one line.
    //
    // That ratio is the only thing that separates a deferral from a
    // rescheduling, and nothing else can report it: a reader draining every
    // window every frame arms and flushes at identical rates and is
    // indistinguishable from a working rail by arm count alone. M << N is the
    // win; M ≈ N means some guest-page reader is asking for these bytes anyway
    // and the next question is which one.
    crate::runtime::drain::note_store_route("surface_flush");
    // Recycled-pages guard, identical in intent to the compute rail's below and
    // to the GVA rail's `deferred_pages_still_ours`: a mapping rebound since
    // arm time (ReplacePhysical, unmap/remap) points at pages this window's
    // pixels do not belong in, and writing them there lands a framebuffer in
    // whatever owns that memory now. Drop rather than write.
    let current = state
        .mappings
        .get(&key.mapping_id)
        .map(|m| m.map_generation);
    if current != Some(key.map_generation) {
        // Release the pin first. This arm returns before touching the frame, and
        // a `Resident` window holds a registry pin that nothing else will drop —
        // `evict_registry_to_cap` and the idle drain both skip pinned slots by
        // design, so a pin leaked here strands a whole framebuffer for the guest
        // lifetime. That is the "~260 stale residents (~516 MiB)" shape, and this
        // drift is not rare: one in 85 s on a driven boot.
        release_window_pin_for_key(key, source);
        crate::observe::fail(format!(
            "deferred_flush_lost kind=render mapping={} {}x{} fmt={:#x} gen={} reason=map_generation_drift current={current:?}",
            key.mapping_id, key.width, key.height, key.pixel_format, key.map_generation
        ));
        return false;
    }
    // Where the frame comes from, in guest scanout order either way.
    //
    // `Owned` carries its own bytes and cannot miss. It used to read
    // `surface_cache::get(mapping_id, key.width, key.height)`, and that is one
    // entry per mapping: a later Store at a different geometry replaced it and
    // every window still armed at the old geometry lost its pixels —
    // `deferred_flush_lost reason=cache_miss`, 15 whole layers in one boot, which
    // is a compositing layer going solid black. The bytes are shared with the
    // cache entry the same readback stored, so owning them costs an `Arc` clone
    // and no copy.
    //
    // `Resident` names the pinned engine image instead, and pays the readback here
    // rather than at every Store. It is checked against the epoch it was published
    // at before being believed: `registry_mark_ready` clears a slot's
    // `content_epoch` on every draw into it, so a mismatch means something rendered
    // over this surface after the Store that armed this window, and the resident no
    // longer holds the frame this window promised the guest. Declining leaves the
    // guest its pre-Store bytes — stale but coherent — where writing would land a
    // different layer's pixels in this one's pages.
    // Set when the frame below came *out of* a resident image, so the write can
    // hand the currency witness back to it. See the re-stamp after the write.
    let mut flushed_from_resident: Option<crate::backend::vulkan::engine::TargetIdentity> = None;
    let frame: std::borrow::Cow<'_, [u8]> = match source {
        crate::model::RenderWindowSource::Owned(bytes) => {
            std::borrow::Cow::Borrowed(bytes.as_slice())
        }
        crate::model::RenderWindowSource::Resident { epoch } => {
            use crate::backend::vulkan::engine::ResidentContent;
            let identity = render_window_identity(key);
            // Three outcomes, not two, and the third used to hide inside the
            // second. `resident_content_epoch` answers `None` both for a slot a
            // later draw un-stamped — expected traffic, the newer pass owns the
            // surface now — and for a slot that is not there at all, which
            // cannot happen to a pinned identity unless the arm and the flush
            // spell that identity differently. One measured boot lost ~150
            // frames here, `live=None` on every one of them, and nothing in the
            // log could say which kind they were. See `engine::ResidentContent`.
            let live = crate::backend::vulkan::engine::resident_content_state(&identity);
            if live != ResidentContent::Epoch(*epoch) {
                crate::backend::vulkan::engine::unpin_resident_target(&identity);
                let (reason, route) = match live {
                    ResidentContent::Absent => (
                        "resident_absent (a pinned slot cannot be evicted, so the arm \
                         and the flush name this target differently)",
                        "rendflush_resident_absent",
                    ),
                    ResidentContent::Unstamped => (
                        "resident_epoch_cleared (a draw landed on this surface after \
                         the Store this window defers)",
                        "rendflush_epoch_cleared",
                    ),
                    ResidentContent::Epoch(_) => {
                        ("resident_epoch_drift", "rendflush_epoch_drift")
                    }
                };
                crate::runtime::drain::note_store_route(route);
                crate::observe::fail(format!(
                    "deferred_flush_lost kind=render mapping={} {}x{} fmt={:#x} gen={} \
                     reason={reason} want={epoch} live={live:?}",
                    key.mapping_id, key.width, key.height, key.pixel_format, key.map_generation
                ));
                return false;
            }
            match crate::backend::vulkan::engine::read_target(&identity) {
                Ok(rb) => {
                    crate::backend::vulkan::engine::unpin_resident_target(&identity);
                    flushed_from_resident = Some(identity);
                    // `into_bgra8`, not the raw bytes: a `Surface` resident is BGRA
                    // so this is a no-op, and the writer below is declared in
                    // scanout order. Reading the reported order rather than
                    // asserting one is what keeps a future format change from
                    // landing R and B exchanged in guest memory.
                    std::borrow::Cow::Owned(rb.into_bgra8())
                }
                Err(e) => {
                    crate::backend::vulkan::engine::unpin_resident_target(&identity);
                    crate::observe::fail(format!(
                        "deferred_flush_lost kind=render mapping={} {}x{} fmt={:#x} gen={} \
                         reason=resident_read err={e}",
                        key.mapping_id, key.width, key.height, key.pixel_format, key.map_generation
                    ));
                    return false;
                }
            }
        }
    };
    let bgra: &[u8] = frame.as_ref();
    let preserve = render_flush_guest_written_ranges(state, host, key);
    let ok = crate::runtime::mapping_write::write_bgra8_skipping(
        state,
        host,
        key.mapping_id,
        bgra,
        key.width.saturating_mul(4),
        key.width,
        key.height,
        &preserve,
    );
    if !ok {
        crate::observe::fail(format!(
            "deferred_flush_lost kind=render mapping={} {}x{} fmt={:#x} gen={} reason=write_refused",
            key.mapping_id, key.width, key.height, key.pixel_format, key.map_generation
        ));
    }
    // Hand the currency witness back to the image the frame came out of.
    //
    // `write_bgra8` ends in `mark_mapping_written`, which advances
    // `surface_content_epoch` — correctly, since the mapping's guest pages did
    // change. But the *pixels* did not: they are the resident's, copied out of it
    // one statement ago. Leaving the stamp behind therefore invalidates a resident
    // that holds exactly the mapping's content, and on the composite rail that is
    // not a residual — it is a loop. The stale stamp costs the next LOAD its
    // elision, the CPU seed it falls back to finds the host cache ceded to this
    // rail, so it reads the mapping's guest pages, and reading them flushes the
    // window this Store just armed, which advances the epoch again. One boot
    // measured it at `surface_flush / surface_resident` = 1369/1373 — one flush per
    // arm, a rail that had become a rescheduling with a GPU round trip added.
    //
    // Only on the resident path: an `Owned` window's bytes came from an `Arc`, and
    // nothing here establishes that the slot under this identity still holds them.
    // The stamp is refused for a slot that is absent or not content_ready, and a
    // failed write leaves `flushed_from_resident` unused, so both fall back to a
    // seed rather than to a wrong frame.
    if ok {
        if let Some(identity) = flushed_from_resident {
            if let Some(epoch) = state
                .mappings
                .get(&key.mapping_id)
                .map(|m| m.surface_content_epoch)
            {
                crate::backend::vulkan::engine::stamp_resident_content_epoch(&identity, epoch);
            }
        }
    }
    crate::runtime::drain::note_drain_phase(crate::runtime::drain::DrainPhase::Flush, started);
    crate::observe::line(format!(
        "render_deferred_flush mapping={} {}x{} fmt={:#x} ok={} bytes={} us={}",
        key.mapping_id,
        key.width,
        key.height,
        key.pixel_format,
        ok as u8,
        bgra.len(),
        started.elapsed().as_micros()
    ));
    ok
}

#[cfg(not(feature = "backend-vulkan"))]
fn flush_render_one<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    _host: &mut M,
    key: &crate::model::ComputeStorageResidencyKey,
    _bgra: &[u8],
) -> bool {
    // No engine ⇒ nothing can have deferred; drop the obligation fail-visibly.
    let _ = state;
    crate::observe::fail(format!(
        "deferred_flush_lost kind=render mapping={} reason=no_backend",
        key.mapping_id
    ));
    false
}

#[cfg(feature = "backend-vulkan")]
fn flush_storage_one<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    key: &crate::model::ComputeStorageResidencyKey,
    generation: u32,
) -> bool {
    let started = std::time::Instant::now();
    // Two unrelated `u32` generations are in scope here and they must not be
    // confused in the log: `key.map_generation` is the mapping's lifetime, the
    // quantity this guard compares, and `generation` is the pinned resident's
    // *content* generation, which only `read_resident_storage` uses. The fail
    // line below printed `content_gen` in a field named `gen` next to
    // `reason=map_generation_drift`, so a live boot read out as a mapping
    // lifetime that had gone backwards (3 -> 2) when the two numbers were
    // simply not comparable. `gen=` is the compared value; the other one says
    // so in its name.
    //
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
            "deferred_flush_lost kind=compute mapping={} {}x{} fmt={:#x} gen={} content_gen={generation} reason=map_generation_drift current={current:?}",
            key.mapping_id, key.width, key.height, key.pixel_format, key.map_generation
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
                    .field("kind", "compute")
                    .field("mapping", key.mapping_id)
                    .field("geom", format!("{}x{}", key.width, key.height))
                    .field("fmt", format!("{:#x}", key.pixel_format))
                    .field("gen", key.map_generation)
                    .field("content_gen", generation)
                    .fail();
                return false;
            }
        };
    let expected_bpp = crate::contract::pixel_format::bytes_per_pixel(key.pixel_format);
    if expected_bpp != Some(texel) {
        crate::observe::fail(format!(
            "deferred_flush_lost kind=compute mapping={} reason=texel_mismatch engine={texel} guest={expected_bpp:?} fmt={:#x}",
            key.mapping_id, key.pixel_format
        ));
        return false;
    }
    let tight = key.width.saturating_mul(texel);
    if crate::observe::content_probe_enabled() {
        crate::observe::off(format!(
            "compute_content stage=flush_out mapping={} {}x{} fmt={:#x} gen={generation} {}",
            key.mapping_id,
            key.width,
            key.height,
            key.pixel_format,
            crate::observe::content_summary(&bytes, texel, key.width, key.height),
        ));
    }
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
            "deferred_flush_lost kind=compute mapping={} reason=guest_write {}x{} off={} bpr={} span_end={}",
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
    crate::runtime::drain::note_drain_phase(crate::runtime::drain::DrainPhase::Flush, started);
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
fn flush_storage_one<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    _host: &mut M,
    key: &crate::model::ComputeStorageResidencyKey,
    generation: u32,
) -> bool {
    let _ = state;
    crate::observe::fail(format!(
        "deferred_flush_lost kind=compute mapping={} content_gen={generation} reason=no_backend",
        key.mapping_id
    ));
    false
}

/// Drop (without flushing) every deferred window on `mapping_id` whose pages
/// can no longer be written safely (ReplacePhysical PFN recycling, unmap
/// without host access). Each drop is fail-visible.
pub fn drop_windows(state: &mut DeviceState, mapping_id: u32, reason: &str) {
    let dropped = state.take_deferred_flush_windows(mapping_id, 0, u64::MAX);
    for (key, owner) in dropped {
        crate::observe::fail(format!(
            "deferred_flush_dropped mapping={} reason={reason} {}x{} fmt={:#x} owner={}",
            key.mapping_id,
            key.width,
            key.height,
            key.pixel_format,
            owner_slug(&owner)
        ));
        // The two rails pin different registries, so the release has to follow
        // the owner. Unpinning storage for a render window would leave the
        // target resident pinned for the life of the boot — the "~260 stale
        // residents (~516 MiB)" shape — while reporting a clean teardown.
        #[cfg(feature = "backend-vulkan")]
        release_window_pin(&key, &owner);
    }
}

/// Drop — do not land — every render window whose guest byte range this Store
/// fully covers, releasing what each one held.
///
/// Lives here rather than at the arm site because the *release* lives here, and
/// the arm site got it wrong for exactly that reason: it took each covered window
/// with a bare `take_deferred_flush_window_exact` and discarded it, so a
/// `Resident` window's counted registry pin was never dropped. That is one leaked
/// pin per composite Store on a surface the compositor repaints — and because the
/// re-Store carries the *same* key, it is the same slot's `pin_count` climbing
/// without bound. `evict_registry_to_cap` rotates pinned slots instead of
/// evicting and the idle drain requires `pin_count == 0`, so a slot that gets
/// there can never be reclaimed again: the "~260 stale residents (~516 MiB)
/// pinned for the guest lifetime" shape, arrived at one frame at a time.
///
/// Dropping rather than flushing is what makes the rail a deferral instead of a
/// rescheduling — a compositor painting one surface re-Stores the identical range
/// every frame, so the previous window always intersects, and landing it here
/// would perform exactly the guest write the rail exists to skip. It is sound for
/// the reason it is sound on the GVA rail: those bytes were never observable
/// without a flush, since any reader would have taken the window first, and this
/// Store's pixels cover every byte of the range.
///
/// Returns the identities whose pins were released, so a caller can log them and
/// a test can read the decision. `None` for an `Owned` window is the answer, not
/// a missing one: its pixels are an `Arc` and dropping it *is* the release.
#[cfg(feature = "backend-vulkan")]
pub(crate) fn supersede_covered_render_windows(
    state: &mut DeviceState,
    key: &crate::model::ComputeStorageResidencyKey,
) -> Vec<(
    crate::model::ComputeStorageResidencyKey,
    Option<crate::backend::vulkan::engine::TargetIdentity>,
)> {
    // Matched on the guest byte range, not on geometry: a sibling Store at a
    // different size over the same span writes the same pages, so its window is
    // covered even though its key differs. `release_window_pin` therefore has to
    // rebuild the identity from the *old* key, which is why it takes one.
    let covered: Vec<crate::model::ComputeStorageResidencyKey> = state
        .compute_deferred_flush
        .iter()
        .filter(|(k, o)| {
            k.mapping_id == key.mapping_id
                && k.surface_offset == key.surface_offset
                && k.span_end == key.span_end
                && matches!(o, crate::model::DeferredOwner::Render { .. })
        })
        .map(|(k, _)| *k)
        .collect();
    let mut released = Vec::with_capacity(covered.len());
    for old in covered {
        if let Some(owner) = state.take_deferred_flush_window_exact(&old) {
            released.push((old, release_window_pin(&old, &owner)));
        }
    }
    released
}

/// Release whatever a taken window held, according to its rail.
///
/// Every site that takes a window and does not flush it must go through this
/// rather than calling `unpin_resident_storage` directly. A compute window owns
/// a storage-registry pin; a render window owns nothing on the GPU — its pixels
/// are a `surface_cache` entry, which is LRU-managed and shared with the Load
/// seed, so it must not be evicted here. Unpinning storage for a render window
/// would name a key the storage registry never held and succeed silently.
///
/// Returns the render identity it unpinned, if any. `unpin_resident_target` is a
/// silent no-op for an absent slot and the engine keeps no log of it, so without
/// this return value "the pin was released" is a claim no test and no boot can
/// read — which is how the supersede site went several commits leaking one.
#[cfg(feature = "backend-vulkan")]
pub(crate) fn release_window_pin(
    key: &crate::model::ComputeStorageResidencyKey,
    owner: &crate::model::DeferredOwner,
) -> Option<crate::backend::vulkan::engine::TargetIdentity> {
    match owner {
        crate::model::DeferredOwner::Storage { .. } => {
            crate::backend::vulkan::engine::unpin_resident_storage(key);
            None
        }
        crate::model::DeferredOwner::Render { source, .. } => {
            release_window_pin_for_key(key, source)
        }
    }
}

/// Release whatever GPU hold a render window's source carries.
///
/// An `Owned` window holds nothing — its pixels are an `Arc` and dropping it is
/// the release, so `None` here is the answer and not a miss. A `Resident` window
/// holds a counted registry pin, and **every** exit that abandons the window has
/// to drop it: `evict_registry_to_cap` and the idle drain both skip pinned slots
/// by design, so a leaked pin strands a whole framebuffer for the guest lifetime
/// rather than merely delaying a reclaim.
#[cfg(feature = "backend-vulkan")]
fn release_window_pin_for_key(
    key: &crate::model::ComputeStorageResidencyKey,
    source: &crate::model::RenderWindowSource,
) -> Option<crate::backend::vulkan::engine::TargetIdentity> {
    if !matches!(source, crate::model::RenderWindowSource::Resident { .. }) {
        return None;
    }
    let identity = render_window_identity(key);
    crate::backend::vulkan::engine::unpin_resident_target(&identity);
    Some(identity)
}

pub(crate) fn owner_slug(owner: &crate::model::DeferredOwner) -> &'static str {
    match owner {
        crate::model::DeferredOwner::Storage { .. } => "compute",
        crate::model::DeferredOwner::Render { .. } => "render",
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

    /// A render window carrying its own 4x4 BGRA frame — the geometry [`key`]
    /// names, since the flush writes `key.width x key.height` from these bytes.
    fn render_owner(armed_seq: u64) -> crate::model::DeferredOwner {
        crate::model::DeferredOwner::Render {
            armed_seq,
            armed_stamp_seq: 0,
            source: crate::model::RenderWindowSource::Owned(std::sync::Arc::new(vec![
                0u8;
                4 * 4 * 4
            ])),
        }
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

    /// The compute flush's drift line must print the generation its guard
    /// compared, not the other one in scope.
    ///
    /// `flush_one` holds two unrelated `u32`s: `key.map_generation` (the
    /// mapping lifetime it compares) and the pinned resident's *content*
    /// generation. The line printed the content generation in a field named
    /// `gen`, adjacent to `reason=map_generation_drift current=…`, and a boot
    /// was read as showing a mapping lifetime running backwards (`gen=3
    /// current=Some(2)`) when the two numbers were never comparable.
    #[test]
    fn the_compute_drift_line_names_the_generation_it_compared() {
        use crate::runtime::host::FakeHost;
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        let m = state.mappings.entry(9).or_default();
        m.mapped = true;
        // Distinct on purpose: the window's map_generation is 1 (from `key`),
        // the mapping is at 5, and the content generation is 3. Only one pair
        // of those is what the guard compares.
        m.map_generation = 5;
        state.compute_deferred_flush.insert(
            key(9, 0, 256),
            crate::model::DeferredOwner::Storage { generation: 3, armed_stamp_seq: 0 },
        );
        let cap = crate::observe::FailCapture::start();
        assert!(!super::flush_intersecting(
            &mut state,
            &mut host,
            9,
            0,
            u64::MAX
        ));
        let line = cap.one("deferred_flush_lost");
        assert!(
            line.contains("reason=map_generation_drift"),
            "wrong refusal: {line}"
        );
        assert!(
            line.contains(" gen=1 ") && line.contains("current=Some(5)"),
            "`gen=` must be the compared window generation: {line}"
        );
        assert!(
            line.contains("content_gen=3"),
            "the resident's content generation must say so in its name: {line}"
        );
        assert!(
            line.contains("kind=compute"),
            "every deferred_flush_lost names its path: {line}"
        );
    }

    /// A type-11 render window is found and landed by the *same* mapping-keyed
    /// trigger the compute rail uses, and is read as a render window.
    ///
    /// This is the property the whole deferred type-11 rail rests on. Its
    /// pixels live in a target resident that `ComputeStorageResidencyKey`
    /// cannot name, so the flush has to dispatch on the owner; if it did not,
    /// `flush_intersecting` would hand a render window to the storage read and
    /// report a compute loss for a window the compute rail never armed. Driving
    /// it through `flush_intersecting` — rather than calling the flush directly
    /// — is deliberate: that call is the choke point every guest-page reader
    /// goes through, so this also pins the trigger wiring.
    ///
    /// The map-generation drift is the cheap way to make the flush take a
    /// decisive branch with no engine present. It doubles as coverage of the
    /// recycled-pages guard: a mapping rebound since arm time must never have a
    /// stale framebuffer written through its new pages.
    #[test]
    fn a_render_window_flushes_through_the_shared_trigger_and_names_its_rail() {
        use crate::runtime::host::FakeHost;
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        let m = state.mappings.entry(9).or_default();
        m.mapped = true;
        // The window latched map_generation 1 (from `key`); the mapping has
        // since moved to 5, so its pages are not the ones the Store rendered
        // for.
        m.map_generation = 5;
        state
            .compute_deferred_flush
            .insert(key(9, 0, 256), render_owner(1));
        let cap = crate::observe::FailCapture::start();
        assert!(
            !super::flush_intersecting(&mut state, &mut host, 9, 0, u64::MAX),
            "a window that cannot be written must report the loss"
        );
        let line = cap.one("deferred_flush_lost");
        assert!(
            line.contains("kind=render"),
            "a render window must not be reported as a compute one: {line}"
        );
        assert!(
            line.contains("reason=map_generation_drift") && line.contains("current=Some(5)"),
            "the rebound mapping must be the stated refusal: {line}"
        );
        assert!(
            state.compute_deferred_flush.is_empty(),
            "the trigger must consume the window it took"
        );
    }

    /// A window landing over pages the guest wrote preserves nothing and says
    /// so; one landing over untouched pages preserves nothing and stays quiet.
    ///
    /// Both halves are the test. The report has to be keyed on the guest write
    /// and not on the landing — the writeback runs on every landing and the
    /// interesting population is the subset the guest also wrote — so the
    /// untouched arm is what makes the reporting arm mean anything.
    ///
    /// This test asserted the opposite of its first half until the rail was
    /// bisected on live boots: the preserving behaviour turned the screen black
    /// (0 of 14 rounds, against 3 of 4 and 2 of 4 clean on the two commits
    /// before it), because `page_gen` is stamped at the harvest and not at the
    /// write, so a store the device's own render superseded can still be named
    /// "written since the Store". See
    /// [`super::render_flush_guest_written_ranges`].
    #[test]
    fn a_render_window_landing_over_guest_writes_reports_them_and_preserves_nothing() {
        use crate::runtime::host::{FakeHost, HostOps};
        let page = 1u64 << PAGE_SHIFT_X86;
        for guest_wrote in [false, true] {
            let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
            let mut host = FakeHost::new();
            let token = host
                .track_guest_writes(&[page], 1usize << PAGE_SHIFT_X86)
                .unwrap();
            let stamped = host.guest_write_gen(token).unwrap();
            let m = state.mappings.entry(9).or_default();
            m.mapped = true;
            m.map_generation = 1;
            // The one tracked page IS this surface's page 0, so the report the
            // host gives back has somewhere in the mapping to land.
            let pfn = (page >> PAGE_SHIFT_X86) as u32;
            m.page_entries = vec![
                (pfn << crate::contract::iosurface_pages::PAGE_ENTRY_PFN_SHIFT)
                    | crate::contract::iosurface_pages::PAGE_ENTRY_VALID,
            ];
            m.guest_write_token = token;
            m.guest_write_token_gen = 1;
            m.guest_write_gen_at_store = stamped;
            if guest_wrote {
                host.guest_wrote_page(page);
            }
            let cap = crate::observe::FailCapture::start();
            let preserve =
                super::render_flush_guest_written_ranges(&state, &host, &key(9, 0, 256));
            let clobbers: Vec<String> = cap
                .lines()
                .into_iter()
                .filter(|l| l.split_whitespace().next() == Some("deferred_flush_clobber"))
                .collect();
            assert!(
                preserve.is_empty(),
                "guest_wrote={guest_wrote}: the landing writes its whole extent, always"
            );
            assert_eq!(
                clobbers.len(),
                usize::from(guest_wrote),
                "guest_wrote={guest_wrote} must decide whether the loss is reported: {clobbers:?}"
            );
        }
    }

    /// A `Resident` window whose resident no longer vouches for the frame
    /// declines, and leaves the guest's pages exactly as it found them.
    ///
    /// This is the whole safety argument for the `skip_readback` rail. An `Owned`
    /// window carries its pixels and cannot be wrong about them; a `Resident`
    /// window carries only a *claim* that a GPU image still holds them, and the
    /// epoch is what tests the claim. `registry_mark_ready` clears a slot's
    /// `content_epoch` on every draw into it, so a mismatch means another layer
    /// rendered over this surface after the Store that armed the window — and
    /// writing then lands that other layer's pixels in these pages, which is the
    /// black/torn-layer class rather than a merely stale frame.
    ///
    /// No engine is initialized here, so `resident_content_epoch` answers `None`
    /// for the reconstructed identity, which is the same reading an evicted slot
    /// produces and the one this arm must fail closed on. The assertion that
    /// matters is the *guest memory*: a decline that still wrote would pass a
    /// log-only check.
    #[test]
    fn a_resident_window_that_cannot_be_vouched_for_declines_without_writing() {
        use crate::contract::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
        use crate::runtime::host::{FakeHost, HostMemory};
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        let page = 1u64 << PAGE_SHIFT_X86;
        let gpa = 0x4500_0000u64;
        host.map_range(gpa, page as usize, 0);
        // A recognizable pre-Store pattern, so "did not write" is checkable
        // rather than indistinguishable from a zeroed page.
        let pre = [0x5Cu8; 256];
        host.write_gpa(gpa, &pre).unwrap();
        state.map_surface(9);
        {
            let m = state.mappings.get_mut(&9).unwrap();
            m.mapped = true;
            m.map_generation = 1;
            m.has_geom = true;
            m.width = 4;
            m.height = 4;
            m.format = crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;
            m.page_entries =
                vec![(((gpa >> PAGE_SHIFT_X86) as u32) << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
        }
        state.compute_deferred_flush.insert(
            key(9, 0, 256),
            crate::model::DeferredOwner::Render {
                armed_seq: 1,
                armed_stamp_seq: 0,
                source: crate::model::RenderWindowSource::Resident { epoch: 7 },
            },
        );
        let cap = crate::observe::FailCapture::start();
        assert!(
            !super::flush_intersecting(&mut state, &mut host, 9, 0, u64::MAX),
            "a window whose resident cannot be vouched for must report the loss"
        );
        let line = cap.one("deferred_flush_lost");
        assert!(
            line.contains("kind=render")
                && line.contains("reason=resident_epoch_drift")
                && line.contains("want=7"),
            "the epoch witness must be the stated refusal, with the value it wanted: {line}"
        );
        let mut after = [0u8; 256];
        host.read_gpa(gpa, &mut after).unwrap();
        assert_eq!(
            &after[..],
            &pre[..],
            "a declined resident window must leave the guest's own bytes untouched"
        );
        assert!(
            state.compute_deferred_flush.is_empty(),
            "the trigger must consume the window it took"
        );
    }

    /// The identity a `Resident` window's flush rebuilds from its key is the one
    /// the draw rendered into, pinned and stamped.
    ///
    /// Four separate places name this slot — the draw's `target_identity`, the
    /// arm's `pin_resident_target`, the arm's `stamp_resident_content_epoch`, and
    /// the flush's `read_target` — and all four resolve through
    /// `present_identity::surface_identity` except the last, which has only the
    /// key. If those two spellings ever disagree the pin protects one image while
    /// the flush reads another: the frame is silently the wrong one, and no
    /// assertion in the crate is watching for it because both lookups *succeed*.
    #[test]
    fn a_render_windows_key_rebuilds_the_identity_the_draw_rendered_into() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        for generation in [1u32, 5, u32::MAX] {
            let m = state.mappings.entry(9).or_default();
            m.map_generation = generation;
            let mut k = key(9, 0, 256);
            k.map_generation = generation;
            assert_eq!(
                super::render_window_identity(&k),
                crate::runtime::present_identity::surface_identity(&state, 9, k.width, k.height),
                "the flush's rebuilt identity must equal the one the draw and the pin used"
            );
        }
    }

    /// A render window lands its own pixels even when `surface_cache` has moved
    /// on to another geometry for the same mapping.
    ///
    /// The flush used to source its bytes from
    /// `surface_cache::get(mapping_id, key.width, key.height)`, and that cache
    /// holds exactly one entry per mapping. A guest that re-Stores the surface at
    /// a new size therefore orphaned every window still armed at the old one:
    /// the flush missed, emitted `deferred_flush_lost reason=cache_miss` and the
    /// guest kept its stale pixels. One boot lost 15 whole layers that way —
    /// including a 1920x1080 desktop surface and a 1920x24 menu bar — which on
    /// screen is a compositing layer rendering solid black.
    #[test]
    fn a_render_window_lands_its_own_pixels_after_the_cache_moved_geometry() {
        use crate::contract::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
        use crate::runtime::host::{FakeHost, HostMemory};
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        let page = 1u64 << PAGE_SHIFT_X86;
        let gpa = 0x4400_0000u64;
        host.map_range(gpa, page as usize, 0);
        state.map_surface(9);
        {
            let m = state.mappings.get_mut(&9).unwrap();
            m.mapped = true;
            m.map_generation = 1;
            m.has_geom = true;
            m.width = 4;
            m.height = 4;
            m.format = crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;
            m.page_entries =
                vec![(((gpa >> PAGE_SHIFT_X86) as u32) << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
        }
        // The window's own frame — every byte 0xA7.
        let frame = vec![0xA7u8; 4 * 4 * 4];
        state.compute_deferred_flush.insert(
            key(9, 0, 256),
            crate::model::DeferredOwner::Render {
                armed_seq: 1,
                armed_stamp_seq: 0,
                source: crate::model::RenderWindowSource::Owned(std::sync::Arc::new(frame.clone())),
            },
        );
        // A later Store re-Stored this mapping at 8x8, replacing the one cache
        // entry it has. The 4x4 window above is now unreachable through it.
        crate::runtime::surface_cache::store(&mut state, 9, 8, 8, vec![0x11u8; 8 * 8 * 4]);

        let cap = crate::observe::FailCapture::start();
        assert!(
            super::flush_intersecting(&mut state, &mut host, 9, 0, u64::MAX),
            "a window carrying its own pixels is always landable"
        );
        assert!(
            cap.lines()
                .iter()
                .all(|l| !l.contains("deferred_flush_lost")),
            "nothing may be lost: {:?}",
            cap.lines()
        );
        // The guest side is row-strided at the mapping's own bytes-per-row, so
        // read it the way the writeback wrote it.
        let (base_off, bpr, _) = {
            let m = state.mappings.get(&9).unwrap();
            crate::runtime::mapping_write::type11_sample_window(
                m,
                9,
                4,
                4,
                crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM,
            )
            .expect("the mapping has a type-11 sample window")
        };
        for y in 0..4u64 {
            let mut row = [0u8; 4 * 4];
            host.read_gpa(gpa + base_off + y * bpr as u64, &mut row)
                .unwrap();
            assert_eq!(
                &row[..],
                &frame[(y as usize) * 16..(y as usize) * 16 + 16],
                "row {y} of the guest pages must hold the window's frame, not the cache's"
            );
        }
    }

    /// A render window fully covered by a later writer is *dropped*, not
    /// flushed, and dropping it takes its alias-index refs with it.
    ///
    /// This is the difference between a deferral and a rescheduling. A guest
    /// compositing into one surface re-Stores the identical guest range every
    /// frame, so the previous window always intersects the new one; landing it
    /// there performs exactly the guest write the rail exists to skip, once per
    /// Store, and `surface_flush` would track `surface_deferred` at a ratio of 1.
    ///
    /// The alias-index half is the part that is easy to get wrong: taking the
    /// entry with a bare `remove` leaves `deferred_alias_pages` holding page
    /// refs for a mapping with no windows left, and the raw-GVA sampling guard
    /// then walks pages nothing defers on.
    #[test]
    fn a_superseded_render_window_is_dropped_and_releases_its_alias_pages() {
        use crate::contract::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        state.map_surface(9);
        {
            let m = state.mappings.get_mut(&9).unwrap();
            m.page_entries = vec![(0x300 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
        }
        let k = key(9, 0, 256);
        state.compute_deferred_flush.insert(k, render_owner(1));
        state.index_deferred_alias_pages(9);
        assert!(
            state.deferred_alias_pages.contains_key(&9),
            "arming indexes the mapping's pages for the raw-GVA guard"
        );

        let released = super::supersede_covered_render_windows(&mut state, &k);
        assert_eq!(
            released.iter().map(|(k, _)| *k).collect::<Vec<_>>(),
            vec![k],
            "the exact key is the one taken"
        );
        assert!(state.compute_deferred_flush.is_empty());
        assert!(
            !state.deferred_alias_pages.contains_key(&9),
            "the last window leaving must drop the mapping's alias-page refs"
        );
    }

    /// The other half of dropping a superseded window: a `Resident` one holds a
    /// counted registry pin, and the supersede is one of the exits
    /// `release_window_pin` names.
    ///
    /// The arm site got this wrong. It took each covered window with a bare
    /// `take_deferred_flush_window_exact` and discarded it, so every composite
    /// Store on a repainted surface leaked one pin — and since the re-Store
    /// carries the same key, it is the same slot's `pin_count` climbing without
    /// bound until nothing can ever reclaim it. `unpin_resident_target` is a
    /// silent no-op with no engine here, so the assertion is on the *identity*
    /// the release named: it has to be rebuilt from the superseded window's own
    /// key, since a covered sibling may carry a different geometry over the same
    /// guest range.
    #[test]
    fn superseding_a_resident_window_releases_the_pin_it_held() {
        use crate::backend::vulkan::engine::TargetIdentity;
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut k = key(9, 0, 256);
        k.map_generation = 3;
        state.compute_deferred_flush.insert(
            k,
            crate::model::DeferredOwner::Render {
                armed_seq: 1,
                armed_stamp_seq: 0,
                source: crate::model::RenderWindowSource::Resident { epoch: 11 },
            },
        );

        let released = super::supersede_covered_render_windows(&mut state, &k);
        assert_eq!(
            released,
            vec![(
                k,
                Some(TargetIdentity::Surface {
                    id: 9,
                    width: k.width,
                    height: k.height,
                    generation: 3,
                })
            )],
            "a resident window's pin must be released, under the identity its own key names"
        );

        // An `Owned` window holds nothing on the GPU, so `None` is the answer and
        // not a missed release — unpinning for one would name a slot the arm never
        // pinned and succeed silently.
        state.compute_deferred_flush.insert(k, render_owner(2));
        assert_eq!(
            super::supersede_covered_render_windows(&mut state, &k),
            vec![(k, None)],
            "an owned window releases nothing"
        );
    }

    /// Superseding one window must not disturb a sibling covering a different
    /// guest range on the same mapping — that one holds bytes the new Store does
    /// not write, and dropping it would lose them.
    #[test]
    fn superseding_one_window_leaves_a_disjoint_sibling_armed() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let covered = key(9, 0, 256);
        let sibling = key(9, 256, 512);
        state
            .compute_deferred_flush
            .insert(covered, render_owner(1));
        state
            .compute_deferred_flush
            .insert(sibling, render_owner(2));

        assert_eq!(
            super::supersede_covered_render_windows(&mut state, &covered).len(),
            1
        );
        assert!(
            state.compute_deferred_flush.contains_key(&sibling),
            "a different range is a different obligation"
        );
        assert_eq!(state.compute_deferred_flush.len(), 1);
    }

    /// Teardown must name the render rail, because the two rails pin different
    /// registries and the drop is where the pin is released.
    ///
    /// Unpinning storage for a render window succeeds silently and leaves the
    /// target resident pinned for the life of the boot — a display-sized image
    /// per window, which is the "~260 stale residents (~516 MiB)" shape. The
    /// slug on this line is the only always-on evidence that the right registry
    /// was chosen.
    #[test]
    fn dropping_a_render_window_reports_the_render_rail() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        state
            .compute_deferred_flush
            .insert(key(9, 0, 256), render_owner(7));
        state.compute_deferred_flush.insert(
            key(9, 256, 512),
            crate::model::DeferredOwner::Storage { generation: 3, armed_stamp_seq: 0 },
        );
        let cap = crate::observe::FailCapture::start();
        super::drop_windows(&mut state, 9, "unit");
        let lines: Vec<String> = cap
            .lines()
            .into_iter()
            .filter(|l| l.split_whitespace().next() == Some("deferred_flush_dropped"))
            .collect();
        assert_eq!(lines.len(), 2, "both windows drop: {lines:?}");
        assert!(
            lines.iter().any(|l| l.contains("owner=render")),
            "the render window must say so: {lines:?}"
        );
        assert!(
            lines.iter().any(|l| l.contains("owner=compute")),
            "the compute window must say so: {lines:?}"
        );
        assert!(state.compute_deferred_flush.is_empty());
    }

    /// `condemn_surface_backing` keeps a mapping's deferred windows on purpose:
    /// `DeleteIOSurfaceBacking2` may name a prior incarnation of a recycled id,
    /// and `mapper::resolve` settles it later by fingerprint compare. A flush
    /// trigger arriving inside that undecided window must therefore leave the
    /// obligation armed — consuming it destroys the very thing the fingerprint
    /// decision exists to reprieve, and reports a loss the flush did not cause.
    #[test]
    fn flush_holds_windows_while_the_backing_is_condemned() {
        use crate::contract::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
        use crate::runtime::host::FakeHost;
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        let k = key(9, 0, 4096);
        state.map_surface(9);
        {
            let m = state.mappings.get_mut(&9).unwrap();
            m.map_generation = 2;
            m.page_entries = vec![(0x300 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
        }
        state
            .compute_deferred_flush
            .insert(k, crate::model::DeferredOwner::Storage { generation: 3, armed_stamp_seq: 0 });
        // The guest deletes the backing; the window is kept for the fingerprint
        // decision and the page list moves to `condemned_entries`.
        assert!(state.condemn_surface_backing(9));
        assert!(state.mapping_backing_condemned(9));
        let ok = super::flush_intersecting(&mut state, &mut host, 9, 0, u64::MAX);
        assert!(ok, "an undecided window is not a loss");
        assert!(
            state.compute_deferred_flush.contains_key(&k),
            "the window must survive for mapper::resolve to reprieve or drop"
        );
    }

    #[test]
    fn flush_intersecting_takes_windows_and_reports_loss() {
        use crate::runtime::host::FakeHost;
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        // Window over an unmapped mapping: the flush must fail closed
        // (fail-visible loss), remove the window, and return false.
        state.compute_deferred_flush.insert(
            key(9, 0, 4096),
            crate::model::DeferredOwner::Storage { generation: 3, armed_stamp_seq: 0 },
        );
        let ok = super::flush_intersecting(&mut state, &mut host, 9, 0, u64::MAX);
        assert!(!ok, "lost window must report failure");
        assert!(
            state.compute_deferred_flush.is_empty(),
            "taken windows never return to the map"
        );
        // Disjoint mapping id: untouched.
        state.compute_deferred_flush.insert(
            key(10, 0, 4096),
            crate::model::DeferredOwner::Storage { generation: 3, armed_stamp_seq: 0 },
        );
        assert!(super::flush_intersecting(
            &mut state,
            &mut host,
            11,
            0,
            u64::MAX
        ));
        assert_eq!(state.compute_deferred_flush.len(), 1);
    }

    /// A raw task-GVA span whose physical pages alias a deferred window's
    /// mapping pages must take (and attempt to flush) that window; a window
    /// on non-aliased pages stays. Locks the boot-18 linear_sample poisoning
    /// channel: GVA reads bypassing the mapping-keyed hooks.
    #[test]
    fn gva_alias_takes_only_aliased_windows() {
        use crate::contract::endian::st32;
        use crate::contract::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
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
        let ckey = |mapping_id: u32| key(mapping_id, 0, 0x1000);
        state.compute_deferred_flush.insert(
            ckey(9),
            crate::model::DeferredOwner::Storage { generation: 3, armed_stamp_seq: 0 },
        );
        state.compute_deferred_flush.insert(
            ckey(10),
            crate::model::DeferredOwner::Storage { generation: 3, armed_stamp_seq: 0 },
        );
        // Product defer sites index pages at defer time.
        state.index_deferred_alias_pages(9);
        state.index_deferred_alias_pages(10);
        assert_eq!(state.deferred_alias_pages.len(), 2);

        super::flush_intersecting_task_gva(&mut state, &mut host, 1, 0, 0x100);
        assert!(
            !state.compute_deferred_flush.contains_key(&ckey(9)),
            "aliased window must be taken for flush"
        );
        assert!(
            state.compute_deferred_flush.contains_key(&ckey(10)),
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
    /// (compute) and linear windows whose defer-time page index aliases the
    /// mapping's physical pages — must be taken for flush.
    /// Windows on disjoint mappings/pages stay deferred. Locks the
    /// boot-25 black-wallpaper class (guest-CPU composite of stale pages).
    #[test]
    fn guest_read_flush_takes_keyed_and_linear_alias_windows() {
        use crate::runtime::host::FakeHost;
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        let page_entry = |pfn: u32| (pfn << 2) | 1;
        for (mid, pfn) in [(9u32, 0x2000u32), (10, 0x2001)] {
            let m = state.mappings.entry(mid).or_default();
            m.mapped = true;
            m.page_entries = vec![page_entry(pfn)];
        }
        state.compute_deferred_flush.insert(
            key(9, 0, 256),
            crate::model::DeferredOwner::Storage { generation: 3, armed_stamp_seq: 0 },
        );
        let disjoint = key(10, 0, 0x1000);
        state.compute_deferred_flush.insert(
            disjoint,
            crate::model::DeferredOwner::Storage { generation: 3, armed_stamp_seq: 0 },
        );
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
        state.arm_linear_deferred_window(lin_aliased, 1, aliased_pages);
        state.arm_linear_deferred_window(lin_disjoint, 1, disjoint_pages);

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
        assert_eq!(flushed, 2, "compute@9 + linear alias");
        assert!(!state.compute_deferred_flush.contains_key(&key(9, 0, 256)));
        assert!(
            state.compute_deferred_flush.contains_key(&disjoint),
            "disjoint mapping's window must stay deferred"
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
            armed_stamp_seq: 0,
            pages: pages.iter().copied().collect(),
            alloc_gen: 0,
        }
    }

    /// Independent restatement of what the union index owes its only caller:
    /// `deferred_pages_intersect` answers true for exactly the pages some live
    /// window still holds. Recomputed here from the three window maps, so the
    /// assertion never consults the index it is checking. `ever_armed` bounds
    /// the domain — every page the caller has armed at any point, so the check
    /// covers pages that must have LEFT the index as well as pages still in it.
    fn assert_index_matches_windows(state: &crate::model::DeviceState, ever_armed: &[u64]) {
        let mut live: std::collections::HashSet<u64> = std::collections::HashSet::new();
        for pages in state.deferred_alias_pages.values() {
            live.extend(pages.iter().copied());
        }
        for entry in state.linear_deferred_flush.values() {
            live.extend(entry.pages.iter().copied());
        }
        for entry in state.gva_deferred_flush.values() {
            live.extend(entry.pages.iter().copied());
        }
        for &page in ever_armed {
            assert_eq!(
                state.deferred_pages_intersect(&[page]),
                live.contains(&page),
                "page {page:#x}: fast reject disagrees with the live window sets"
            );
        }
    }

    /// The refcounted deferred-page union index (`deferred_page_refs`, the fast
    /// reject behind `deferred_pages_intersect`) tracks exactly the live window
    /// pages across both window kinds that carry their own page sets: shared
    /// pages are refcounted so a page survives until its LAST window disarms,
    /// re-arm swaps page sets cleanly, and a page shared by a GVA window and a
    /// linear window needs both to disarm before it leaves.
    ///
    /// This is the invariant the deleted 1-in-64 `deferred_ref_drift` self-heal
    /// used to repair at runtime. `DeferredWindows` now denies mutable access to
    /// the three source maps outside `model::state`, so a window can no longer
    /// be armed or disarmed without the paired refcount move; what remains
    /// testable is that the paired moves are themselves right.
    #[test]
    fn deferred_page_refs_track_arm_disarm_with_sharing() {
        use crate::model::{ComputeStorageResidencyKey, DeviceState, PAGE_SHIFT_X86};
        let mut state = DeviceState::new(crate::model::DeviceId(1), PAGE_SHIFT_X86);
        let p = |pfn: u64| pfn << PAGE_SHIFT_X86;
        let seen = [p(0xA), p(0xB), p(0xC), p(0xD), p(0xE), p(0xF)];
        // Two windows share page A; window 1 also owns B, window 2 also owns C.
        state.arm_gva_deferred_window(0x1000, gva_entry(1, 4, 4, &[p(0xA), p(0xB)]));
        state.arm_gva_deferred_window(0x2000, gva_entry(1, 4, 4, &[p(0xA), p(0xC)]));
        assert!(state.deferred_pages_intersect(&[p(0xA)]));
        assert!(state.deferred_pages_intersect(&[p(0xB)]));
        assert!(state.deferred_pages_intersect(&[p(0xC)]));
        assert!(!state.deferred_pages_intersect(&[p(0xD)]));
        assert_index_matches_windows(&state, &seen);
        // A linear window joins on A too: the page is now held by three windows
        // of two different kinds, and the refcount is the only thing that knows.
        let lin = ComputeStorageResidencyKey::linear(1, 7, 0, 4, 4, 4, 2, 0x46);
        state.arm_linear_deferred_window(lin, 1, [p(0xA), p(0xF)].into_iter().collect());
        assert_index_matches_windows(&state, &seen);
        // Disarm window 1: shared A stays, B leaves.
        assert!(state.take_gva_deferred_window(0x1000).is_some());
        assert!(
            state.deferred_pages_intersect(&[p(0xA)]),
            "A shared by window 2 and the linear window"
        );
        assert!(
            !state.deferred_pages_intersect(&[p(0xB)]),
            "B was only window 1"
        );
        assert!(state.deferred_pages_intersect(&[p(0xC)]));
        assert_index_matches_windows(&state, &seen);
        // Re-arm window 2 onto a different page set: C leaves, E joins. A stays
        // only because the linear window still holds it.
        state.arm_gva_deferred_window(0x2000, gva_entry(1, 4, 4, &[p(0xE)]));
        assert!(
            !state.deferred_pages_intersect(&[p(0xC)]),
            "re-arm dropped C"
        );
        assert!(
            state.deferred_pages_intersect(&[p(0xA)]),
            "the linear window still holds A"
        );
        assert!(state.deferred_pages_intersect(&[p(0xE)]));
        assert_index_matches_windows(&state, &seen);
        // Re-arm the linear window off A: now nothing holds it.
        state.arm_linear_deferred_window(lin, 2, [p(0xF)].into_iter().collect());
        assert!(
            !state.deferred_pages_intersect(&[p(0xA)]),
            "last holder of A re-armed away from it"
        );
        assert_index_matches_windows(&state, &seen);
        // Disarm both remaining windows: index empties.
        assert_eq!(
            state
                .disarm_linear_deferred_window(&lin)
                .map(|window| window.pages),
            Some([p(0xF)].into_iter().collect()),
            "disarm returns the pages the window was armed against"
        );
        assert!(state.take_oldest_gva_deferred_window().is_some());
        for page in seen {
            assert!(!state.deferred_pages_intersect(&[page]));
        }
        assert_index_matches_windows(&state, &seen);
    }

    /// A linear compute-storage window records the fence it was armed under, and
    /// a re-arm records the fence it was re-armed under.
    ///
    /// This rail writes a raw task GVA with no mapping incarnation to name, so
    /// the only thing that can say a landing is late is the stamp counter at arm
    /// time. Without it every linear landing is unscoreable, which is the state
    /// `6bc2220` left it in while clearing the two rails that *do* carry an
    /// allocation identity.
    #[test]
    fn a_linear_window_records_the_fence_it_was_armed_under() {
        use crate::model::{ComputeStorageResidencyKey, DeviceState, PAGE_SHIFT_X86};
        let mut state = DeviceState::new(crate::model::DeviceId(1), PAGE_SHIFT_X86);
        let p = |pfn: u64| pfn << PAGE_SHIFT_X86;
        let key = ComputeStorageResidencyKey::linear(1, 7, 0x4000, 256, 0x1000, 64, 64, 0x46);

        state.completion_stamp_seq = 41;
        state.arm_linear_deferred_window(key, 1, [p(0xA)].into_iter().collect());
        assert_eq!(
            state.linear_deferred_flush.get(&key).unwrap().armed_stamp_seq,
            41,
            "the window must carry the fence it was armed under"
        );

        // The guest is fenced twice, then the same key re-arms: the window is a
        // NEW obligation and must be scored against the new fence, not the one
        // its predecessor was armed under.
        state.completion_stamp_seq = 43;
        state.arm_linear_deferred_window(key, 2, [p(0xB)].into_iter().collect());
        let window = state.disarm_linear_deferred_window(&key).unwrap();
        assert_eq!(window.armed_stamp_seq, 43, "a re-arm re-stamps the window");
        assert_eq!(window.generation, 2);
        assert_eq!(window.pages, [p(0xB)].into_iter().collect());
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

    /// Page drift must distinguish the cases it exists to separate, and now
    /// **decide** them.
    ///
    /// A probe that reports nothing is indistinguishable from a probe that
    /// cannot fire, and this codebase has already paid for three of those. So
    /// drive both controls through the same fixture: a window whose GVA still
    /// resolves to its armed pages must stay silent and stay writable, and one
    /// whose pages moved under it must produce the line and be refused — same
    /// task, same geometry, only the armed set differs.
    ///
    /// The decision is asserted alongside the line because they are two separate
    /// claims. Logging drift while still writing is exactly what this used to
    /// do, and the guest heap corruption that allowed — WindowServer aborting in
    /// `small_free_list_remove_ptr_no_clear` — is why it decides now.
    /// The mapping-keyed rails get the same reading as the GVA rail, per rail,
    /// so a boot can say whether `map_generation` in the key is already enough
    /// to make deferral here safe — rather than the two being assumed alike.
    #[test]
    fn each_mapping_rail_is_scored_against_the_fence_under_its_own_name() {
        use crate::runtime::drain::store_route_count;
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let k = key(7, 0, 256);

        let render = render_owner(1);
        let storage = crate::model::DeferredOwner::Storage {
            generation: 3,
            armed_stamp_seq: 0,
        };

        // Inside the fence: neither rail may be reported.
        let before = [
            store_route_count("rendw_stamp_outlived"),
            store_route_count("storw_stamp_outlived"),
        ];
        super::note_mapping_window_against_fence(&state, &k, &render);
        super::note_mapping_window_against_fence(&state, &k, &storage);
        assert_eq!(
            [
                store_route_count("rendw_stamp_outlived"),
                store_route_count("storw_stamp_outlived")
            ],
            before,
            "a window landed inside its own fence is the safe case on both rails"
        );

        // Past the fence: each rail reports under its own counter, so a boot can
        // tell a render-Store window from a compute-storage one.
        state.completion_stamp_seq = 5;
        super::note_mapping_window_against_fence(&state, &k, &render);
        assert_eq!(
            [
                store_route_count("rendw_stamp_outlived"),
                store_route_count("storw_stamp_outlived")
            ],
            [before[0] + 1, before[1]],
            "the render rail must not be counted under the storage rail's name"
        );
        super::note_mapping_window_against_fence(&state, &k, &storage);
        assert_eq!(
            [
                store_route_count("rendw_stamp_outlived"),
                store_route_count("storw_stamp_outlived")
            ],
            [before[0] + 1, before[1] + 1],
            "and the storage rail must not be counted under the render rail's"
        );
    }

    /// A completion stamp is the guest's licence to free everything it allocated
    /// for the work being completed, so the stamp must leave nothing owed to
    /// guest RAM. Asserted through [`crate::runtime::drain::write_stamp`] itself
    /// rather than against the helper, because the claim that matters is the
    /// wiring: a helper nothing calls at the fence is the bug this fixes.
    #[cfg(feature = "backend-vulkan")]
    #[test]
    fn a_completion_stamp_leaves_no_window_still_owing_guest_ram() {
        use crate::runtime::host::FakeHost;
        let page = 1u64 << PAGE_SHIFT_X86;
        let mut host = FakeHost::new();
        let stamp_pfn = 9u32;
        host.map_range(u64::from(stamp_pfn) << PAGE_SHIFT_X86, page as usize, 0);
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        state.gfx.fifo_base_page = stamp_pfn;

        state.arm_gva_deferred_window(0x1000, gva_entry(1, 4, 4, &[]));
        state.arm_gva_deferred_window(0x2000, gva_entry(1, 4, 4, &[]));
        assert_eq!(state.gva_deferred_flush.len(), 2, "two windows armed");

        crate::runtime::drain::write_stamp(&mut state, &mut host, 1, 0x55);

        assert!(
            state.gva_deferred_flush.is_empty(),
            "the guest may free every one of these targets the instant it reads \
             the stamp, so none of them may still be waiting to be written"
        );
        assert_eq!(
            state.completion_stamp_seq, 1,
            "the fence the windows are measured against must have moved"
        );
    }

    /// The guest's fence is the only thing that separates a deferred write from
    /// a write into somebody else's allocation, and the page-set guard cannot
    /// see it: free-then-reuse inside one process leaves the translation
    /// identical, so `deferred_pages_still_ours` says yes to exactly the window
    /// that corrupts the guest heap.
    ///
    /// Both directions are asserted. A census that fires on every landing is as
    /// useless as one that never fires — the whole point is that it separates
    /// the windows landed inside their own fence from the ones that outlived it.
    #[cfg(feature = "backend-vulkan")]
    #[test]
    fn a_window_landed_after_its_fence_is_counted_apart_from_one_landed_inside_it() {
        use crate::runtime::drain::store_route_count;
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);

        // Negative control: armed and landed under the same stamp. The guest
        // has not been told this render finished, so it cannot have freed the
        // target, and the write is the one the Store promised.
        let mut inside = gva_entry(1, 4, 4, &[]);
        inside.armed_stamp_seq = state.completion_stamp_seq;
        let same_before = store_route_count("gvaw_stamp_same");
        let outlived_before = store_route_count("gvaw_stamp_outlived");
        super::note_window_outlived_its_stamp(&state, 0x1000, &inside, "rearm");
        assert_eq!(
            store_route_count("gvaw_stamp_same"),
            same_before + 1,
            "a window landed inside its own fence is the safe case and must be counted as one"
        );
        assert_eq!(
            store_route_count("gvaw_stamp_outlived"),
            outlived_before,
            "a guard that fires on every landing cannot price the repair"
        );

        // Positive control: the same window, landed after the guest was fenced.
        state.completion_stamp_seq = state.completion_stamp_seq.wrapping_add(3);
        let same_before = store_route_count("gvaw_stamp_same");
        super::note_window_outlived_its_stamp(&state, 0x1000, &inside, "gva_alias");
        assert_eq!(
            store_route_count("gvaw_stamp_outlived"),
            outlived_before + 1,
            "a window whose stamp moved before it landed writes memory the guest was \
             told it could reclaim, and that is the class the page-set guard is blind to"
        );
        assert_eq!(
            store_route_count("gvaw_stamp_same"),
            same_before,
            "the two outcomes must not both be counted for one landing"
        );
    }

    #[test]
    fn window_page_drift_refuses_the_guest_write_and_is_silent_without_it() {
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
        assert!(
            super::window_pages_still_ours(
                &state,
                &host,
                0,
                &gva_entry(1, 4, 4, &[data0]),
                "gva_alias",
                "guest=refused",
            ),
            "an unmoved window must stay writable — a guard that refuses every \
             flush means the guest never sees a Store"
        );
        assert_eq!(
            drift_lines(at),
            0,
            "a window that did not move must be quiet"
        );

        // Positive control: same window, armed on a page it no longer maps to.
        let at = mark();
        assert!(
            !super::window_pages_still_ours(
                &state,
                &host,
                0,
                &gva_entry(1, 4, 4, &[9 * page]),
                "gva_alias",
                "guest=refused",
            ),
            "a window whose pages moved must be refused, not merely reported"
        );
        assert_eq!(drift_lines(at), 1, "a window whose pages moved must report");

        // A window armed on TWO pages whose range now resolves ONE page, and
        // that page is not one of the two. This is the arrangement a guest
        // produces by releasing a GPU allocation and letting part of the virtual
        // range be re-pointed: fewer pages come back, and what does come back
        // belongs to somebody else.
        //
        // A guard keyed on page COUNT reads this as "shorter walk, therefore
        // teardown, therefore nothing to protect" and permits it, and the writer
        // then lands rows in `data0` — which this window never owned. Keyed on
        // membership it is refused, which is what the guest's own crash reports
        // say has to happen.
        let at = mark();
        assert!(
            !super::window_pages_still_ours(
                &state,
                &host,
                0,
                &gva_entry(1, 4, 4, &[7 * page, 8 * page]),
                "clear_store",
                "guest=refused",
            ),
            "a short walk that resolves a page the window was never armed on is \
             not a teardown — it is a write into another owner's pages"
        );
        assert_eq!(
            drift_lines(at),
            1,
            "the refusal must be visible; a silent one cannot be scored"
        );

        // The benign half of the same shape: fewer pages come back, and every
        // one of them is still ours. Refusing this would drop live Stores whose
        // destination never moved, so the guard must not simply require equal
        // sets.
        let at = mark();
        assert!(
            super::window_pages_still_ours(
                &state,
                &host,
                0,
                &gva_entry(1, 4, 4, &[data0, 8 * page]),
                "clear_store",
                "guest=refused",
            ),
            "a walk that came back short but entirely inside the armed pages is \
             the teardown case, and its rows land in this window's own memory"
        );
        assert_eq!(drift_lines(at), 0, "a subset walk must stay quiet");
    }

    /// The same window, asked by the reader instead of the writer.
    ///
    /// The cross-pass resident Load in `encode_draw_chain` trusts a GVA resident
    /// as a draw's *prior content*, gated on a deferred window existing at the
    /// address with matching geometry — conditions a different allocation
    /// reusing the address satisfies exactly. The flush path had refused that
    /// drift since it was written; the read path did not ask, which left the two
    /// sides of one window disagreeing about whether it still belonged to its
    /// name.
    ///
    /// What this pins is that the reader gets the same verdict *and its own
    /// outcome word*. A drift line is the only record either consumer leaves,
    /// and `guest=refused` on a line emitted by a Load would say guest memory
    /// was protected when what was actually refused was a stale picture.
    #[test]
    fn the_resident_load_reader_gets_the_same_drift_verdict_under_its_own_name() {
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
        let tail = |from: usize| -> String {
            std::fs::read_to_string(crate::observe::fail_log_path())
                .unwrap_or_default()
                .get(from..)
                .unwrap_or_default()
                .lines()
                .filter(|l| l.starts_with("deferred_window_page_drift "))
                .collect::<Vec<_>>()
                .join("\n")
        };
        let mark = || {
            std::fs::read_to_string(crate::observe::fail_log_path())
                .unwrap_or_default()
                .len()
        };

        // The address still names the pages the window was armed on, so the
        // resident behind it is this allocation's own prior frame and the Load
        // may take it. Refusing here would cost a seed on every chained pass.
        let at = mark();
        assert!(
            super::window_pages_still_ours(
                &state,
                &host,
                0,
                &gva_entry(1, 4, 4, &[data0]),
                "xpass_load",
                "resident=refused",
            ),
            "an unmoved window's resident is the draw's own prior content"
        );
        assert_eq!(tail(at), "", "an unmoved window must be quiet on both sides");

        // The guest handed this address to a different allocation. The resident
        // still exists, still has the geometry, and still reports content_ready
        // — every gate the Load had before this check. It holds the previous
        // allocation's pixels.
        let at = mark();
        assert!(
            !super::window_pages_still_ours(
                &state,
                &host,
                0,
                &gva_entry(1, 4, 4, &[9 * page]),
                "xpass_load",
                "resident=refused",
            ),
            "a reallocated address must not load the previous owner's pixels as \
             this draw's prior content"
        );
        let line = tail(at);
        assert!(
            line.contains("trigger=xpass_load"),
            "the line must name the reader that asked: {line}"
        );
        assert!(
            line.contains("resident=refused"),
            "the reader refuses a resident, not a guest write: {line}"
        );
        assert!(
            !line.contains("guest=refused"),
            "a Load must not claim it protected guest memory: {line}"
        );
    }

    /// The linear compute-storage rail gets the same drift decision as the GVA
    /// rail, and takes its span the way the arm site does.
    ///
    /// `flush_linear_one` needs a live Vulkan engine to reach its guest write, so
    /// this exercises the decision itself with a linear key's geometry. The span
    /// argument is the subtle part and the positive control is what pins it: a
    /// linear key's `span_end` is a *length* (`row_stride * height`), not an end
    /// address, and the arm site walks `(surface_offset, span_end)` with exactly
    /// those two values. Reading `span_end` as an end address here would make the
    /// span `page - page == 0`, the walk would come back empty, the short-walk arm
    /// would permit, and the positive control below would fail — which is the
    /// point of siting it at a nonzero offset rather than at GVA 0, where both
    /// readings coincide.
    #[test]
    fn a_linear_window_whose_pages_moved_is_refused_and_reads_its_span_as_a_length() {
        use crate::contract::endian::st32;
        use crate::contract::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
        use crate::runtime::host::{FakeHost, HostMemory};
        let page = 1u64 << PAGE_SHIFT_X86;
        let mut host = FakeHost::new();
        let (dir_gpa, root_gpa, data0, data1) = (2 * page, 3 * page, 4 * page, 5 * page);
        for gpa in [dir_gpa, root_gpa, data0, data1] {
            host.map_range(gpa, page as usize, 0);
        }
        let mut d = [0u8; 8];
        st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 3);
        st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
        host.write_gpa(dir_gpa, &d).unwrap();
        // Two PTEs: GVA page 0 → data0, GVA page 1 → data1.
        let mut ptes = [0u8; 8];
        st32(&mut ptes[0..], 4);
        st32(&mut ptes[4..], 5);
        host.write_gpa(root_gpa, &ptes).unwrap();
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        assert!(state.define_task(1, 8 * page, 2));

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
        // One page long, sited at GVA `page` so a length/end confusion is visible.
        let (offset, span) = (page, page);

        // Negative control: armed on the page GVA `page` resolves to right now.
        let at = mark();
        assert!(
            super::deferred_pages_still_ours(
                &state,
                &host,
                1,
                offset,
                span,
                &[data1].into_iter().collect(),
                "8x8 trigger=linear_flush ref=5",
                "guest=refused",
            ),
            "an unmoved linear window must stay writable — a guard that refuses \
             every flush means the guest never sees a compute Store"
        );
        assert_eq!(
            drift_lines(at),
            0,
            "a linear window that did not move must be quiet"
        );

        // Positive control: same window, armed on a page it no longer maps to.
        // This is also the assertion that the span is read as a length.
        let at = mark();
        assert!(
            !super::deferred_pages_still_ours(
                &state,
                &host,
                1,
                offset,
                span,
                &[9 * page].into_iter().collect(),
                "8x8 trigger=linear_flush ref=5",
                "guest=refused",
            ),
            "a linear window whose pages moved must be refused — and a zero-length \
             walk from misreading span_end as an end address would permit it"
        );
        assert_eq!(
            drift_lines(at),
            1,
            "a linear window whose pages moved must report"
        );
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
        state.compute_deferred_flush.insert(
            key(7, 0, 256),
            crate::model::DeferredOwner::Storage { generation: 3, armed_stamp_seq: 0 },
        );
        state.compute_deferred_flush.insert(
            key(7, 256, 512),
            crate::model::DeferredOwner::Storage { generation: 4, armed_stamp_seq: 0 },
        );
        state.compute_deferred_flush.insert(
            key(8, 0, 256),
            crate::model::DeferredOwner::Storage { generation: 5, armed_stamp_seq: 0 },
        );

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
