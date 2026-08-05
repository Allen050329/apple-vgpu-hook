//! Do the guest pages a window names still belong to the surface that armed it?
//!
//! A window records where its bytes are owed, and between arming and landing
//! the guest may unmap, remap or recycle those pages. Writing anyway lands the
//! bytes in whatever now occupies the address — the PTE-corruption class. Each
//! check here re-walks a window's pages against the live mapping and refuses,
//! fail-visibly, when the answer has changed.
//!
//! The whole module is `backend-vulkan`: what a window defers is a pinned
//! engine resident, and nothing can arm one on a build without the engine
//! (see this rail's module doc).

use crate::model::DeviceState;
use crate::runtime::host::{HostMemory, HostOps};

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
    // Same accounting the mapping rail carries: this function has three ways to
    // return `true` and only one of them checked anything, so a boot reporting
    // no drift on this rail could not say whether the guard had passed or had
    // nothing to pass on. Counted, never gated — every write that landed before
    // still lands.
    if span == 0 || armed.is_empty() {
        crate::runtime::drain::note_store_route("defw_unwit_no_armed");
        return true;
    }
    let mut live = std::collections::HashSet::new();
    live.extend(crate::runtime::gva_mem::task_gva_page_gpa_set(
        host,
        &state.tasks,
        task_id,
        gva,
        span,
        state.page_shift,
    ));
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
        // `all` over an EMPTY set is true, and that is not a verification: it
        // means the walk resolved no page of this window at all, so there was
        // nothing to compare against `armed`. Harmless — `write_gva_rgba8`
        // resolves per row from the same walk, so no row lands either — but it
        // must not be counted as the guard having agreed, which is exactly the
        // conflation that made the mapping rail's `refused = 0` unreadable.
        crate::runtime::drain::note_store_route(if live.is_empty() {
            "defw_unwit_no_live"
        } else {
            "defw_pages_verified"
        });
        return true;
    }
    crate::runtime::drain::note_store_route("defw_pages_drifted");
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

/// Whether the mapping's cached page list still names the guest memory it was
/// walked from, counted so a boot carries the rate and gated so an arm and its
/// control stay one binary apart.
///
/// The check is [`crate::runtime::mapper::type4_pages_witness`]; this is the
/// deferred rails' use of it, and it is the missing half of a guarantee the
/// raw-GVA rails already have. `gva_view::write_span` re-walks the task page
/// table at write time and fails closed, stating outright that a write through a
/// cached view "lands in whatever now owns those host pages (guest heap
/// corruption: the 2026-07-19 WindowServer SIGSEGV class)". The mapping-keyed
/// rails write through `MappingEntry::page_entries`, which for a type-4 surface
/// nothing re-walks between the resolve that filled it and the flush that uses
/// it.
///
/// That is the shape of every crash this device is chasing. The user's report is
/// WindowServer aborting inside `small_free_list_remove_ptr_no_clear` under an
/// allocation made by `AppleParavirtGPUMetal`, and the guest kernel's own poison
/// check found freed elements "filled with 0xFF from offset 0" — opaque white
/// pixels in memory the guest had already reclaimed. The twelve guest panics on
/// disk hit apfs, airportd, tccd, a HID driver and WindowServer, which is not a
/// bug in one path but a device writing where it no longer has title.
///
/// # Drift refuses this write and stops the list being believed again
///
/// Refusing the one window is not enough. The list is what every later reader
/// and writer of this mapping resolves through, so leaving it in place means the
/// next flush asks the same question and the next present serves pixels read
/// through the same wrong pages. `invalidate_mapping_pages` clears it and bumps
/// `map_generation`, which retires the contiguous view and the guest-write token
/// with it, and every window still armed against the old incarnation then
/// refuses on the `map_generation` check it already has.
///
/// Self-healing rather than terminal: the next type-4 bind re-resolves the
/// surface from the object list and adopts a fresh plan, which is the path that
/// would have discovered this eventually anyway. An actively-drawn surface
/// recovers on its next bind; an idle one stays unresolvable, which is the
/// correct state for a mapping this device can no longer name.
///
/// Deliberately NOT a forced `resolve_type4_surface` here. That would be the
/// more informative answer — it re-runs the object search and could say whether
/// the surface merely moved or is gone — but it goes through `map_surface`,
/// which clears `has_geom`, the geometry and `surface_content_epoch` before the
/// adoption restores them. Running that from inside a flush puts a mapping
/// through a destructive half-state while a writeback is in progress, to answer
/// a question the next bind answers for free.
///
/// # This is not a duplicate of the writer's vouch, and collapsing them is wrong
///
/// `flush_render_one` calls this, and then the write it performs calls
/// `mapping_write::vouch_for_write`, which reaches the same
/// `mapper::mapping_pages_verdict` — the same `O(pages)` guest page-table walk,
/// microseconds apart. It reads as pure redundancy and it is not. Both halves
/// were measured on a driven x86/Vulkan boot — `mapping_pages_ours=15686`
/// against `mapw_pages_vouched=15699` on 15 653 flushes, so each runs once per
/// flush, and `readback_split vouch_us` prices one at 55 µs, about 37 ms of a
/// 913 ms busy second — so the cost is real. Removing this call is still wrong,
/// for two independent reasons:
///
/// - **They are the same question asked at two times, and both times matter.**
///   Between them run `flush_windows_under_bgra8_write` and the writeback's own
///   `flush_intersecting`, and the guest runs on its own vCPUs across both. A
///   re-point inside that interval is visible only to the later walk; a
///   re-point before it, only to this one.
/// - **Only this one can drop the window cleanly.** It refuses before the frame
///   is acquired, releases the registry pin, and reports the loss under its own
///   name (`rendflush_page_drift`, `reason=mapping_page_drift`). The writer's
///   vouch refuses with `GpuWritebackDecline::PagesNotOurs` after the flush has
///   committed to landing, which falls through to the copying arm to be refused
///   a second time under a different name.
///
/// Hoisting instead of deleting does not work either. `PagesVouched::covers`
/// re-checks `map_generation` and nothing else, and the case this guard exists
/// for is a type-4 surface re-pointed with **no** generation bump — so a token
/// minted here and presented later would still be `covers`-valid over exactly
/// the drift it was supposed to catch.
///
/// `rendflush_page_drift` reading 0 is not evidence against any of this. It is
/// 0 on the Safari window-drag workload and non-zero on the control boot that
/// traced the class end to end (a 1225x512 WebKit tile whose backing was
/// fabricated at its own GVA, refused when the live walk disagreed);
/// `a_render_window_over_repointed_pages_is_refused_and_counted` is that boot
/// turned into a test, and it fails on the collapsed version.
fn mapping_pages_still_ours<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mapping_id: u32,
) -> bool {
    use crate::runtime::mapper::PagesVerdict;
    match crate::runtime::mapper::mapping_pages_verdict(state, host, mapping_id) {
        PagesVerdict::Ours => {
            crate::runtime::drain::note_store_route("mapping_pages_ours");
            true
        }
        // Lands exactly as `Ours` does; counted apart because it is not the
        // same claim. `mapping_pages_ours` used to include every flush this
        // witness had nothing to say about, so the ratio it appeared to give
        // against `mapping_pages_drifted` was not the guard's hit rate.
        PagesVerdict::Unwitnessed(_) => {
            crate::runtime::drain::note_store_route("mapping_pages_unwitnessed");
            true
        }
        PagesVerdict::Drifted => {
            crate::runtime::drain::note_store_route("mapping_pages_drifted");
            false
        }
    }
}

/// Which of the three questions a mapping-keyed window failed.
///
/// Two rails land such a window — the render Store rail and the compute
/// storage rail — and both must ask exactly these three, in exactly this
/// order, before writing a byte. They used to ask them twice: two copies of
/// the ladder, two copies of the fail line, and the copies had already parted
/// in one visible way (only the render copy counts its refusals on the census).
/// The order is the argument, so it belongs in one place.
pub(super) enum WindowRefusal {
    /// The guest declared its own pages authoritative after the work this
    /// window defers, so nothing is owed. Asked first because it is the prior
    /// question: the other two ask *where* the bytes would land, this asks
    /// whether they are owed at all.
    HostCopySuperseded,
    /// The mapping was rebound since arm time (ReplacePhysical, unmap/remap),
    /// so it points at pages this window's pixels do not belong in. Writing
    /// them there lands a framebuffer in whatever owns that memory now.
    MapGenerationDrift { current: Option<u32> },
    /// `map_generation` is the guest's *declared* incarnation, and a type-4
    /// surface can be re-pointed with nothing declared at all — so the pages
    /// are re-walked even when the generation matches. See
    /// [`mapping_pages_still_ours`].
    MappingPageDrift,
}

/// Which rail is landing the window.
///
/// The two ask the same three questions and their drift refusals lose the same
/// thing — guest work that nothing will re-arm — so the census names for those
/// losses come from one table rather than from each rail's own call site.
#[derive(Clone, Copy)]
pub(super) enum Rail {
    Render,
    Compute,
}

impl WindowRefusal {
    /// The `reason=` field, including whatever the reason itself carries.
    ///
    /// `defers` names the guest work this rail's window holds — a Store for the
    /// render rail, a dispatch for the compute one. It is the only per-rail
    /// word in any of the three, which is why it is a parameter rather than a
    /// second copy of the sentence.
    pub(super) fn reason(&self, defers: &str) -> String {
        match self {
            Self::HostCopySuperseded => format!(
                "host_copy_superseded (the guest declared its own pages \
                 authoritative after the {defers} this window defers)"
            ),
            Self::MapGenerationDrift { current } => {
                format!("map_generation_drift current={current:?}")
            }
            Self::MappingPageDrift => "mapping_page_drift".to_string(),
        }
    }

    /// The store-route name for a refusal that loses guest work, or `None`
    /// when nothing was lost.
    ///
    /// A lost tile has to be countable, not just loggable. `flush_intersecting`
    /// has already TAKEN the window out of `compute_deferred_flush` by the time
    /// either rail refuses, and `flush_mapping_windows_before_fence` returns
    /// `()`, so the fence advances and nothing re-arms the obligation: the
    /// pixels land nowhere, permanently. That is the event a census has to be
    /// able to score an arm against.
    ///
    /// `mapping_pages_drifted` is not a substitute for either rail — it is
    /// incremented inside [`mapping_pages_still_ours`], which more than one
    /// caller reaches, so it counts refusals rather than lost work.
    ///
    /// The render rail carried its two names and the compute rail carried
    /// none, which was not a decision anybody made: the two ladders were 400
    /// lines apart and each looked complete. Both are here now, so a third
    /// refusal added to the enum has to say what it costs on *both* rails
    /// before it compiles.
    pub(super) fn lost_work_route(&self, rail: Rail) -> Option<&'static str> {
        match (rail, self) {
            // Not a loss: the guest declared its own pages authoritative, so
            // it asked for these bytes to be dropped.
            (_, Self::HostCopySuperseded) => None,
            (Rail::Render, Self::MapGenerationDrift { .. }) => Some("rendflush_gen_drift"),
            (Rail::Render, Self::MappingPageDrift) => Some("rendflush_page_drift"),
            (Rail::Compute, Self::MapGenerationDrift { .. }) => Some("compflush_gen_drift"),
            (Rail::Compute, Self::MappingPageDrift) => Some("compflush_page_drift"),
        }
    }
}

/// Ask a mapping-keyed window's three questions, in order. `None` means the
/// bytes are still owed and may still be written where the key says.
///
/// Nothing is released here. Both callers hold something that must be dropped
/// on a refusal — a registry pin on the render rail, a pinned resident on the
/// compute one — and they are not the same thing, so the release stays at the
/// call site where the reader can see which one it is.
pub(super) fn mapping_window_refusal<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    key: &crate::model::ComputeStorageResidencyKey,
) -> Option<WindowRefusal> {
    if crate::runtime::resource_validity::writeback_refused(state, key.mapping_id) {
        return Some(WindowRefusal::HostCopySuperseded);
    }
    let current = state
        .mappings
        .get(&key.mapping_id)
        .map(|m| m.map_generation);
    if current != Some(key.map_generation) {
        return Some(WindowRefusal::MapGenerationDrift { current });
    }
    if !mapping_pages_still_ours(state, host, key.mapping_id) {
        return Some(WindowRefusal::MappingPageDrift);
    }
    None
}

/// The `deferred_flush_lost` line every mapping-keyed refusal writes.
///
/// Six copies of this prefix used to be spelled out, and the field set is
/// exactly what a reader of the log reasons from: one of them printed the
/// resident's *content* generation in a field named `gen`, next to
/// `reason=map_generation_drift`, and a boot was read as showing a mapping
/// lifetime running backwards (`gen=3 current=Some(2)`) when the two numbers
/// were never comparable. `gen=` is the mapping lifetime — the quantity the
/// drift guard compares — and `content_gen=`, when a rail has one, is the
/// pinned resident's content generation.
pub(super) fn deferred_flush_lost(
    kind: &str,
    key: &crate::model::ComputeStorageResidencyKey,
    content_gen: Option<u32>,
    reason: &str,
) -> String {
    let content = match content_gen {
        Some(g) => format!(" content_gen={g}"),
        None => String::new(),
    };
    format!(
        "deferred_flush_lost kind={kind} mapping={} {}x{} fmt={:#x} gen={}{content} reason={reason}",
        key.mapping_id, key.width, key.height, key.pixel_format, key.map_generation
    )
}
