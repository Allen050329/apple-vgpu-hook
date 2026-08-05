//! What the rail measured: batch sizes, stamp order, and who read a landing.
//!
//! None of this decides anything — the same membership rule, and the same
//! module name, as `runtime::exec::report`. Every function here records a reading the
//! rail's own cost arguments are built from — how many windows a fence pass
//! landed, whether a window outlived the stamp it was armed under, whether
//! anybody ever read what a landing wrote — and they live together so that the
//! rails above stay readable as rails.
//!
//! Two of them are the crate's entry points for the read half: `scanout`,
//! `drain` and `runtime::draw` call [`note_render_flush_cache_read`] and
//! [`note_render_flush_pages_read`] when they consume a landed surface, and
//! `note_render_flush_landed` scores the previous landing against them.

use crate::model::DeviceState;
#[cfg(feature = "backend-vulkan")]
use crate::runtime::host::HostOps;

/// How many windows one completion fence lands at once.
///
/// [`flush_gva_one`](crate::runtime::storage_flush::land::flush_gva_one) pays a private `vkQueueSubmit` and its own fence wait per
/// window, so a batch of *n* is *n* serialized GPU round trips inside one
/// completion stamp. That is the number that priced recording each copy into its
/// draw's own command buffer instead: at a batch of 1 the build saves one
/// submission's overhead per flush, and at a batch of 8 it collapses eight waits
/// into the one the tranche already owes.
///
/// **It reads 1, always** — 56 209 of 56 209 on one driven boot and 56 713 of
/// 56 713 on a second, with every other band at 0. That retires the *tranche*
/// half of the argument for that build: there is no set of round trips for a
/// shared fence to collapse. It says nothing about the latency half, which
/// [`flush_gva_one`](crate::runtime::storage_flush::land::flush_gva_one) records as open.
///
/// Kept after the verdict rather than removed, because it is the assumption the
/// verdict rests on and not a number that was consumed once. A workload that
/// ever lands two windows on one stamp reopens the question, and this is the
/// only thing that would say so — `gvaw_fence_flush` counts windows, and a
/// hundred landing one at a time reads the same as ten landing ten at a time.
/// Banded rather than summed for the reason the draw-list census is: the tail is
/// the case that matters and a mean hides it.
///
/// Zero is not counted. An empty map returns before this, so a zero band would
/// only ever record calls that had nothing to do.
///
/// Gated with its caller: `flush_gva_windows_before_fence` is Vulkan-only and a
/// Metal build arms no GVA windows, so an ungated band is dead code there.
#[cfg(feature = "backend-vulkan")]
pub(super) fn note_fence_batch_band(landed: u64) {
    if let Some(slug) = fence_batch_band(landed) {
        crate::runtime::drain::note_store_route(slug);
    }
}

/// The band [`note_fence_batch_band`] charges, split out so the boundaries can
/// be read without an emit.
#[cfg(feature = "backend-vulkan")]
pub(super) fn fence_batch_band(landed: u64) -> Option<&'static str> {
    Some(match landed {
        0 => return None,
        1 => "gvaw_fence_batch_1",
        2 => "gvaw_fence_batch_2",
        3..=4 => "gvaw_fence_batch_3_4",
        5..=8 => "gvaw_fence_batch_5_8",
        9..=16 => "gvaw_fence_batch_9_16",
        17..=64 => "gvaw_fence_batch_17_64",
        _ => "gvaw_fence_batch_over_64",
    })
}

/// Mark the host surface cache copy of `mapping_id` as taken by a host reader.
///
/// Called beside every mapping-keyed read of [`crate::runtime::surface_cache`],
/// which is the leg `land::flush_render_one` stores through. Unknown mappings are
/// ignored: a cache entry outlives its mapping, and a read of one says nothing
/// about a flush there is no longer an entry to attribute it to.
pub fn note_render_flush_cache_read(state: &mut DeviceState, mapping_id: u32) {
    if let Some(m) = state.mappings.get_mut(&mapping_id) {
        m.render_flush.cache_unread = false;
    }
}

/// Mark this mapping's guest pages as gathered by a host reader — the other leg
/// `land::flush_render_one` writes.
pub fn note_render_flush_pages_read(state: &mut DeviceState, mapping_id: u32) {
    if let Some(m) = state.mappings.get_mut(&mapping_id) {
        m.render_flush.pages_unread = false;
    }
}

/// Report what read the previous landed flush of this mapping, then arm the
/// witness for the one landing now.
///
/// The pair of counts per leg is what makes the flush's cost answerable:
/// `render_flush_cache_used` / `render_flush_cache_unread`, and the `pages_`
/// pair beside them, divide a gigabyte a second of readback into the part
/// something asked for and the part nothing did. A mapping whose first flush is
/// landing now is not counted, so an arriving surface is never scored as unread
/// work.
///
/// Only where the rail exists. A Metal-direct build never arms a mapping-keyed
/// window — `flush_mapping_windows_before_fence` is a no-op there — so there is
/// no landing to score, and the two readers above stay unconditional only
/// because clearing a flag that was never set costs nothing and keeps the
/// scanout and sampled rungs free of a cfg.
#[cfg(feature = "backend-vulkan")]
pub(crate) fn note_render_flush_landed(
    state: &mut DeviceState,
    mapping_id: u32,
    cache_stored: bool,
) -> Option<crate::model::RenderFlushWitness> {
    use crate::runtime::drain::note_store_route;
    let now_us = crate::observe::elapsed_us();
    let m = state.mappings.get_mut(&mapping_id)?;
    let prior = std::mem::replace(
        &mut m.render_flush,
        crate::model::RenderFlushWitness {
            landed: true,
            cache_stored,
            cache_unread: cache_stored,
            pages_unread: true,
            landed_us: now_us,
        },
    );
    if !prior.landed {
        return None;
    }
    // How long the flush being replaced survived. Bucketed against the VBL
    // interval, because that is what separates "the compositor repainted"
    // from "this surface was written twice inside one drain tranche".
    note_store_route(match now_us.saturating_sub(prior.landed_us) {
        0..=999 => "render_flush_age_sub_ms",
        1000..=8332 => "render_flush_age_sub_frame",
        _ => "render_flush_age_frame_plus",
    });
    // Only where the previous flush actually stored one. A borrowed-frame flush
    // publishes nothing to the cache, and counting its absent copy as unread
    // would inflate the very number that prices the cache leg.
    if prior.cache_stored {
        note_store_route(if prior.cache_unread {
            "render_flush_cache_unread"
        } else {
            "render_flush_cache_used"
        });
    }
    note_store_route(if prior.pages_unread {
        "render_flush_pages_unread"
    } else {
        "render_flush_pages_used"
    });
    Some(prior)
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
/// `guards::deferred_pages_still_ours` cannot see this. It asks whether the GVA still
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
/// [`flush_gva_windows_before_fence`](crate::runtime::storage_flush::fence::flush_gva_windows_before_fence) inverts it completely:
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
pub(super) fn note_window_outlived_its_stamp(
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
/// `note_window_outlived_its_stamp` is the same reading for the GVA render
/// rail, and the hazard is identical because the identity is identical: a
/// `ComputeStorageResidencyKey::linear` names a task and an address
/// (`mapping_id` 0, `map_generation` carrying the task id), so nothing the guest
/// does to reclaim the memory reaches this rail as a notification.
///
/// That distinction is why `6bc2220` cleared the other two deferred rails and
/// cannot clear this one. `flush_render_one` and `flush_storage_one` refuse on
/// `map_generation` drift, and `map_generation` moves on exactly the events that
/// let a guest reuse an IOSurface's storage. This rail has no such generation to
/// compare — `guards::deferred_pages_still_ours` is its only guard, and free-then-reuse
/// inside one process preserves the translation the guard reads.
///
/// The rail's own flush already records what that costs when it goes wrong:
/// a `pmap_page_protect` kernel panic and userspace SIGSEGVs inside libmalloc's
/// page bookkeeping. What was missing is how often the landing is late at all,
/// which is what `linw_stamp_same` against `linw_stamp_outlived` says.
#[cfg(feature = "backend-vulkan")]
pub(super) fn note_linear_window_outlived_its_stamp(
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

/// Score a mapping-keyed deferred window against the guest's fence, exactly as
/// `note_window_outlived_its_stamp` scores the GVA rail.
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
/// - `land::flush_render_one` and `land::flush_storage_one` both compare the mapping's
///   live `map_generation` against `key.map_generation` and refuse with
///   `deferred_flush_lost reason=map_generation_drift` before reading anything.
/// - `map_generation` is bumped by exactly the events that let the guest reuse
///   an IOSurface's storage — MAP, UNMAP, `ReplacePhysical`, MappingInternal
///   reattach, any page-table refresh that changes PFNs.
/// - A `DeleteIOSurfaceBacking2` that has not yet resolved leaves the backing
///   *condemned*, and [`flush_intersecting`](crate::runtime::storage_flush::access::flush_intersecting) refuses to take those windows at
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
/// ([`flush_mapping_windows_before_fence`](crate::runtime::storage_flush::fence::flush_mapping_windows_before_fence)) — for the *other* hazard, which this
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
pub(super) fn note_mapping_window_against_fence(
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

/// Report that a landing window is about to overwrite the guest's own stores.
///
/// This returns nothing, and that is the finding rather than an omission. It
/// used to hand back the ranges to preserve; it now preserves none of them, and
/// carrying an always-empty `Vec` out to the writeback made three signatures
/// advertise a narrowing no caller can ever ask for — a reader auditing whether
/// this rail honours guest writes would find a `skip` parameter and conclude it
/// does, when it deliberately does not and says so on every occurrence.
///
/// A deferred window promises to replay a synchronous Store later, and that is
/// only a replay while nothing else writes the pages in between. The writeback
/// covers the whole attachment extent, so a guest CPU store into any page of it
/// — an inter-buffer damage forward-copy, a CoreGraphics blit into the same
/// IOSurface — is gone the moment this window lands. Nothing else in the flush
/// can see that: `map_generation` covers a rebind, `resident_content_epoch`
/// covers a later device draw, and neither is a witness for the surface's own
/// owner.
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
/// [`flush_mapping_windows_before_fence`](crate::runtime::storage_flush::fence::flush_mapping_windows_before_fence) lands every armed window before the
/// guest is told the work is done, so the interval in which a guest store can be
/// both after the Store and before the writeback does not exist. Nothing needs
/// preserving because nothing is clobbered, and this function becomes the
/// standing check on that.
///
/// It is a **loose** check, and reading it as a tight one sends a reader after a
/// hole that need not exist. The verdict is `guest_write_gen(token) !=
/// guest_write_gen_at_store`, and that generation moves at the *harvest* that
/// saw the page dirty, not at the write —
/// [`note_render_flush_over_guest_write`] states the same rule for the same
/// reason. `reims_vgpu_dirty_gen` returns the value as of the last harvest and
/// only marks a read as owed; `reims_vgpu_dirty_harvest` then returns early
/// unless a read is owed, and runs at the drain tail. So a guest store made
/// *before* the Store, in a tranche whose harvest had not yet run, is stamped
/// into a generation that moves *after* it, and this fires. That is structural,
/// not a race: it is the same unsoundness that made preserving the pages black
/// the screen, and it points the one way that costs nothing to be wrong about.
///
/// So a surviving occurrence cannot be read as a count of clobbered windows. It
/// is, however, no longer something to tolerate a line of, because on a healthy
/// device it does not occur at all.
///
/// Over twenty recorded x86/Vulkan boots on three binaries, this route fires if
/// and only if the boot's guest-write witness latched — the state
/// `reims_vgpu_dirty_harvest` used to reach when its tracked window crossed the
/// PCI hole, in which every page of every surface reads permanently written:
///
/// ```text
/// witness healthy   15 boots   0 firings, over 69k-85k surface_flushes each
/// witness latched    5 boots   29-69 firings
/// ```
///
/// Zero overlap, and the zeroes are on boots that ran the rail tens of thousands
/// of times — a real never-fires rather than a rail the workload skipped. That
/// makes this a **healthy-zero alarm**: a firing is not a loose upper bound to
/// be discounted, it is the witness reporting that every page reads written, and
/// the guest is about to show transparent backdrops and broken popover geometry
/// until it reboots. Treat a single line as that, and read
/// `gw_refused_guest_store` next to it — the two move together.
///
/// The loose-verdict reasoning above still holds and is why the *preserve* rail
/// stays retired: it is structural, about stamping at harvest rather than at the
/// write, and it is untouched by the witness being fixed. What the fix removes
/// is only the expectation that this counter carries a standing background rate.
/// The bisect that blacked the screen was run on a latched binary, which is
/// exactly what preserving produces when the witness claims every page was
/// written, so it corroborates far less than its numbers suggest. Do not read it
/// as having independently measured the preserve rail.
///
/// The other thing that would be a defect is a `rendw_stamp_outlived` naming a
/// window that landed after [`crate::runtime::drain::write_stamp`] — that one is
/// an ordering statement the device can actually make.
///
/// [`crate::runtime::mapping_write::write_bgra8_skipping`] and
/// `HostOps::guest_written_pages` stay: the sampled ladder's merge uses both,
/// and it errs the other way — it keeps both halves rather than choosing.
#[cfg(feature = "backend-vulkan")]
pub(super) fn note_render_flush_over_guest_write<M: HostOps>(
    state: &DeviceState,
    host: &M,
    key: &crate::model::ComputeStorageResidencyKey,
) {
    use crate::runtime::mapper::{mapping_guest_write_verdict, GuestWriteVerdict};
    if mapping_guest_write_verdict(state, host, key.mapping_id) != GuestWriteVerdict::Wrote {
        return;
    }
    crate::runtime::drain::note_store_route("render_flush_over_guest_write");
    crate::observe::fail(format!(
        "deferred_flush_clobber kind=render mapping={} {}x{} fmt={:#x} gen={} \
         (a guest write to this surface was observed since the Store this window \
         defers, and the full-extent writeback replaces it; the witness moves at \
         harvest, not at the write, so this line cannot order the two — but it \
         does not occur at all on a healthy device, so a run of these means the \
         guest-write witness has latched every page as written, and \
         gw_refused_guest_store will be in the thousands)",
        key.mapping_id, key.width, key.height, key.pixel_format, key.map_generation
    ));
}
