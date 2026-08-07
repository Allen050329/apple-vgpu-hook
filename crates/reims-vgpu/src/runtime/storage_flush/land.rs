//! Landing one window: resident to host, host to guest pages.
//!
//! Given a single window a caller believes is armed, copy the pinned engine
//! resident out once (which also unpins it) and write it into the guest window,
//! then re-establish the residency mirror so chained seed skips keep working.
//! The identity functions here answer the question that must be settled before
//! any of that: is the window still describing the surface it was armed for?
//! `guards` answers the other one — are its pages still ours.

#[cfg(feature = "backend-vulkan")]
use super::guards::{
    deferred_flush_lost, deferred_pages_still_ours, mapping_window_refusal,
    window_pages_still_ours, Rail,
};
#[cfg(feature = "backend-vulkan")]
use super::lifecycle::release_window_pin_for_key;
use super::report::note_mapping_window_against_fence;
#[cfg(feature = "backend-vulkan")]
use super::report::{
    note_linear_window_outlived_its_stamp, note_render_flush_landed,
    note_render_flush_over_guest_write, note_window_outlived_its_stamp,
};
use crate::model::DeviceState;
use crate::runtime::host::{HostMemory, HostOps};

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
/// ([`flush_gva_one`], `draw::vulkan::supersede_gva_window`,
/// `draw::vulkan::try_sample_deferred_gva`) so the three cannot drift
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
///
/// # This rail is a GPU round trip that carries almost no bytes
///
/// It is the last fully-copying writeback: `read_target` submits a readback,
/// waits its own fence, maps the staging buffer and copies it into a `Vec`;
/// `write_gva_rgba8_within` then scatters that `Vec` into guest pages a row at
/// a time; and the encode cache takes a third pass. The render rail retired
/// exactly this shape by making the guest's pages the copy's destination
/// (`render_flush_gpu_direct`), and nothing equivalent exists here.
///
/// What makes it worth a separate entry in the ledger is that its cost is not
/// the bytes. One driven x86/Vulkan second, Safari window drag:
///
/// ```text
/// flush_rails     render_us=377653 render=640      gva_us=99662 gva=960
/// readback_split  fence=1680 fence_us=329043       gpu_us=205578 gpu=1680
/// ```
///
/// `gpu` counts both rails, and 672 render flushes of a 1920x1080 frame at the
/// measured PCIe rate account for **all** of `gpu_us` on its own — so this
/// rail's ~1000 flushes a second contribute no measurable copy time and its
/// ~104 µs apiece is submit-to-signal latency, a map, and three host passes over
/// a small surface. Roughly 100 ms of a 913 ms busy second.
///
/// # Why recording it at arm time is open here and closed for the render rail
///
/// [`flush_mapping_windows_before_fence`](crate::runtime::storage_flush::fence::flush_mapping_windows_before_fence) refutes "append the copy to the
/// render's own command buffer" for the mapping-keyed rail, and one of its two
/// reasons is arming rates: an icon workload measured 49 706 arms against
/// 12 343 flushes, so recording at arm time would pay a DMA for four windows in
/// five that nothing ever reads.
///
/// **That ratio is 1.000 on this rail, and it is not close.** Summed over a
/// whole driven boot:
///
/// ```text
/// gva_deferred 23177   gva_flush_guest_written 23177
/// gvaw_fence_flush 23177   gvaw_stamp_same 23177
/// ```
///
/// Every arm is flushed, by the very next fence, inside the stamp it was armed
/// in — four counters that cannot agree by accident. So the objection that
/// closes the door for the render rail does not reach this one.
///
/// # Where the 97 µs goes, and which of the two builds it chooses
///
/// Measured boot-wide over 25 853 flushes, the three spans this function now
/// brackets:
///
/// ```text
/// gva_us       2512514   97.2 us/flush   end to end
/// gva_read_us  1700067   65.8 us  (68%)  read_target: submit, own fence, map, copy to Vec
/// gva_write_us  630126   24.4 us  (25%)  CPU scatter into guest pages
/// gva_cache_us  117957    4.6 us  ( 5%)  host cache store
/// gva_write_kb  474639   18.4 KB/flush
/// ```
///
/// 18 KB a flush is why this rail contributes nothing to `gpu_us`: at the
/// measured PCIe rate that copy is about a microsecond, so essentially all of
/// `gva_read_us` is submit-to-signal latency and the staging map. The scatter's
/// 24.4 µs over 18 KB is ~750 MB/s, which is the cache-cold guest-page write
/// rate the ledger already quotes.
///
/// ## The GPU-direct arm is refuted, and not on performance grounds
///
/// Giving this rail `render_flush_gpu_direct`'s shape — guest pages as the
/// copy's destination — would delete `gva_write_us`, a quarter of the flush. It
/// cannot be done, because [`crate::runtime::draw::host_cache_store_gva_layer`]
/// below is not optional. `surface_cache`'s GVA entry is the *only* place the
/// frame survives the guest unmapping the VA — the wallpaper-retain contract —
/// and `enforce_gva_cache_cap` refuses to evict any entry that still owes a
/// deferred writeback, calling that a correctness exclusion rather than a
/// heuristic. Publishing that entry needs a host copy of the frame, which is
/// what `read_target` exists to produce.
///
/// So a GPU-direct arm would have to keep the readback *and* add a second
/// submission for the GPU-side copy: it would pay `gva_read_us` unchanged, add
/// a round trip, and save only the scatter. Strictly worse — and the reference
/// being cheap does not rescue it: naming the destination is a range check on
/// an already-held import, but what the second submission adds is another fence
/// to wait on, and that is the expensive half everywhere else in this file. The
/// two channel-order and format questions it also raises (a GVA resident is RGBA
/// where `copy_target_to_guest_pages` demands scanout order, and
/// `convert_rgba8_to_row` is a straight copy only for the two RGBA8 formats)
/// are real but do not need answering, because this one closes it first.
///
/// ## What is left is the round trip — and recording the copy does not remove it
///
/// The build this rail's numbers used to point at was: record the readback into
/// the draw's own command buffer at arm time, so it is waited against a fence
/// with the rest of the tranche to signal rather than one issued a moment ago.
/// The draw's command buffer is submitted and deliberately *not* waited
/// (`render_post_wait_skips` equals the draw count), so there was room for it.
///
/// **Measured, it is not worth building.** Two readings close it, both from one
/// driven x86/Vulkan boot under a 60 s Safari drag:
///
/// ```text
/// gvaw_fence_batch_1 56209        every other band 0
/// readback_split     submit 10.1 us   fence 213 us   map 1.8 us   (per readback)
/// ```
///
/// The first says **there is no tranche**. Every one of 56 209 completion stamps
/// landed exactly one window — the batch is 1.000, never 2 — so there is no set
/// of serialized round trips for a shared fence to collapse. That was the half
/// of the argument that could have paid for the risk, and it does not exist.
/// [`flush_gva_windows_before_fence`](crate::runtime::storage_flush::fence::flush_gva_windows_before_fence)'s own repair note explains why: landing at
/// the fence keeps the window set nearly empty by construction.
///
/// The second reading does **not** close it, and an earlier revision of this
/// section said it did. That claim is withdrawn; what follows is what the
/// numbers actually support.
///
/// `readback_split` is **pooled across this rail and the render rail**, and the
/// two are nothing alike. The render rail copies whole frames; this one copies
/// 17.1 KB a flush, which is why the section above concludes it "contributes
/// nothing to `gpu_us`" — at the measured rate that copy is about a microsecond.
/// So the pooled `gpu_us` of 131 µs is the render rail's, and applying it here
/// to conclude "the fence is mostly the copy, so folding saves only the 10 µs
/// submit" was reading the wrong rail's number. **For this rail, essentially all
/// of `gva_read_us` is submit-to-signal latency**, and folding is exactly a
/// change to how many submissions that latency is paid for.
///
/// What folding provably removes: one of the two `vkQueueSubmit`s, the draw
/// batch flush that `begin_entry` forces before every readback, and the
/// submit-to-GPU-start latency of the second command buffer. What it cannot
/// remove: the copy's own execution, and the fence-signal-to-CPU-wake.
/// `bar_us = 1.16 µs` proves the draw is already finished when the copy runs, so
/// none of the wait is the draw.
///
/// The split between those two halves of the ~85 µs was not measurable by
/// counter — the GPU and CPU timelines here are deliberately uncorrelated (see
/// `note_readback_gpu_us`: both spans are deltas on the GPU's own clock). It was
/// settled by experiment instead, and the experiment says **submission latency
/// is not the cost**.
///
/// The readback paths were changed to append their copy to the open draw batch
/// and submit once rather than twice. Submissions across a driven boot fell from
/// 287 425 to 159 901, a 44 % cut that removed one whole `vkQueueSubmit` from
/// the front of nearly every readback. `fence_us` per readback did not fall:
/// 203 µs before, 222 µs after, inside boot-to-boot variance, with total fence
/// time 21.7 s against 21.6 s over comparable census windows.
///
/// Removing a submission did not shorten the wait, so the ~85 µs is
/// fence-signal-to-CPU-wake and not submit-to-GPU-start. Folding the copy into
/// the draw's own command buffer removes a submission too, and nothing else —
/// so it would buy the same nothing, for a three-state readback decision, a
/// staging-slot lease held across the ring, and a fence handle threaded into the
/// deferred entry.
///
/// Standing verdict: **refuted on measurement.** Both halves are now closed —
/// the tranche half by `gvaw_fence_batch_1`, the latency half by the append
/// experiment. What is left of this rail's ~90 µs is the cost of blocking a
/// thread on a GPU fence at all, which is a question about *who waits* rather
/// than about how the work is submitted.
///
/// ## Where the rail's time actually goes
///
/// `fence_us` summed 20.0 s across 77.6 s of census windows, with `write_us` at
/// 0 because the GPU-direct rail already landed the copies. **A quarter of the
/// device's wall clock is spent blocked on readback fences**, and none of it is
/// the copy. That is a question about who waits, not about what is copied, and
/// no rearrangement of the copy addresses it.
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
    // Timed apart from the rest of the flush because it is the whole routing
    // question for this rail. `flush_rails gva_us` is the end-to-end cost and
    // `readback_split fence_us` pools this rail's fence with the render rail's,
    // so how much of a GVA flush is the GPU round trip and how much is the
    // three host passes has only ever been derived by algebra off `gpu_us`.
    // The two answers point at different builds: a guest-pages destination
    // removes the host passes and keeps the round trip, and recording the copy
    // into the draw's own command buffer removes the round trip and keeps the
    // passes.
    let read_started = std::time::Instant::now();
    let read_target = crate::backend::vulkan::engine::read_target(&identity);
    crate::runtime::drain::note_store_route_us(
        "gva_read_us",
        read_started.elapsed().as_micros() as u64,
    );
    crate::runtime::drain::note_store_route("gva_reads");
    let rgba = match read_target {
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
        // The CPU scatter, timed against `gva_read_us`. This is the pass a
        // guest-pages destination would delete and the round trip above is the
        // one it would keep, so the two readings are what choose between the
        // builds.
        let write_started = std::time::Instant::now();
        let written = crate::runtime::draw::write_gva_rgba8_within(
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
        );
        crate::runtime::drain::note_store_route_us(
            "gva_write_us",
            write_started.elapsed().as_micros() as u64,
        );
        crate::runtime::drain::note_store_route_n("gva_write_kb", (rgba.len() as u64) >> 10);
        guest = match written {
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
    // The host cache is stored on all five outcomes, deliberately: on the four
    // that did not reach guest RAM it is what holds the authoritative bytes. But
    // that makes the cache store a poor witness of whether the guest got them,
    // and the `guest=` word below rides `observe::line`, which is off by
    // default — so on a stock boot nothing says how the rail's writes divided.
    // Census it on the always-on counters instead. `written` is the healthy
    // majority; the other four are each explained at their arm above.
    crate::runtime::drain::note_store_route(match guest {
        "written" => "gva_flush_guest_written",
        "skip" => "gva_flush_guest_skip",
        "skip_drift" => "gva_flush_guest_skip_drift",
        "unmapped" => "gva_flush_guest_unmapped",
        _ => "gva_flush_guest_write_fail",
    });
    // The third host pass over the frame, timed for the same reason as the
    // other two: a GPU-direct arm has no host copy to publish, so this is the
    // one that has to be answered for rather than simply removed. What it costs
    // is what a reader has to weigh against whatever replaces it.
    let cache_started = std::time::Instant::now();
    crate::runtime::draw::host_cache_store_gva_layer(
        state,
        host,
        entry.task_id,
        entry.texture_ref,
        entry.producer_object_type,
        gva,
        entry.width,
        entry.height,
        &rgba,
        // The four outcomes above that did not reach guest RAM are exactly the
        // ones where this entry is the only copy, so they mark it unevictable.
        // See `crate::model::HostSurface::guest_holds_bytes` — this is the
        // condition the comment above ("on the four that did not reach guest
        // RAM it is what holds the authoritative bytes") had stated but not
        // recorded anywhere the byte cap could read.
        guest == "written",
    );
    crate::runtime::drain::note_store_route_us(
        "gva_cache_us",
        cache_started.elapsed().as_micros() as u64,
    );
    // A flush that landed is expected control flow and stays quiet. The two
    // outcomes that are not — a refused write, and a window whose span the guest
    // had already torn down — each emit their own typed line above, so the
    // always-on view keeps the losses and drops the running commentary.
    crate::runtime::drain::note_drain_phase(
        crate::runtime::drain::DrainPhase::Flush(crate::runtime::drain::FlushRail::Gva),
        started,
    );
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
    // The `skip_drift` arm below refuses the guest write and calls that
    // lossless, on the grounds that the cache entry holds the frame. That is
    // true only if this call landed it, so a failure here has to reach the log:
    // otherwise the two together drop a frame with no record, which is the
    // whole loss this rail exists to avoid. `Superseded` is the exception —
    // a newer defer already owns the entry, so there is no frame to keep.
    let cached = crate::runtime::surface_cache::materialize_linear_resident(
        state,
        task_id,
        texture_ref,
        generation,
        &bytes,
    );
    if let Err(decline) = &cached {
        if !matches!(
            decline,
            crate::runtime::surface_cache::LinearMaterializeDecline::Superseded { .. }
        ) {
            crate::observe::Emit::decline("linear_materialize_lost", decline)
                .field("task", task_id)
                .field("ref", texture_ref)
                .field("geom", format!("{}x{}", key.width, key.height))
                .field("fmt", format!("{:#x}", key.pixel_format))
                .field("gen", generation)
                .fail();
        }
    }
    let tight = (key.width as usize).saturating_mul(texel as usize);

    // Same hazard, same answer as the GVA rail: this window was armed against a
    // page set at defer time and `write_linear_guest_within` walks fresh, so a span the
    // guest has since re-pointed sends a compute-storage image into whatever
    // owns those pages now. Observed on this rail as guest heap corruption — a
    // `pmap_page_protect` kernel panic and userspace SIGSEGVs inside libmalloc's
    // own page bookkeeping. Refusing is lossless *when* the cache entry kept
    // the authoritative bytes, which is exactly `cached.is_ok()` — the refusal
    // and the store are one claim, so the emit above is what makes the pair
    // honest rather than the comment that used to state it unconditionally.
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
    crate::runtime::drain::note_drain_phase(
        crate::runtime::drain::DrainPhase::Flush(crate::runtime::drain::FlushRail::Linear),
        started,
    );
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
            // The guest deleted the object, so the image's content is not guest
            // work any more and the reclaim paths must be able to take it. An
            // unpin alone would trade this function's pinned-VRAM leak for a
            // sole-copy one.
            crate::backend::vulkan::engine::retire_resident_storage_content(key);
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
pub(super) fn flush_one<M: HostMemory + HostOps>(
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
pub fn render_window_identity(
    key: &crate::model::ComputeStorageResidencyKey,
) -> crate::backend::vulkan::engine::TargetIdentity {
    crate::backend::vulkan::engine::TargetIdentity::Surface {
        id: key.mapping_id,
        width: key.width,
        height: key.height,
        generation: key.map_generation as u64,
    }
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
    // That ratio used to be the only thing separating a deferral from a
    // rescheduling: a reader draining every window every frame arms and flushes
    // at identical rates and is indistinguishable from a working rail by arm
    // count alone, so M << N was the win and M ≈ N meant some guest-page reader
    // was asking for these bytes anyway.
    //
    // `flush_mapping_windows_before_fence` changes what the ratio means, and a
    // census read on the old rule would draw the wrong conclusion from it. Every
    // armed window now lands at the next completion stamp by design, so M ≈ N is
    // the *intended* state and says nothing about guest-page readers.
    //
    // This comment used to send the reader to `surface_resident` against
    // `surface_flush` in one second, on the reasoning that what deferral still
    // buys is coalescing inside one fence. That instrument has now been read,
    // and it cannot answer: the two are equal on **all 1 780 census lines** of
    // the accumulated x86 / Vulkan log, 193 458 each, with not one line
    // differing. Every arm gets exactly one flush, because
    // `surface_deferred_superseded` has never once fired — no later Store in the
    // whole log fully covered a live window — and every arm succeeds, because
    // `surface_resident_sync` is likewise absent. A ratio pinned at 1.0 by the
    // workload is not a measurement of coalescing; it is the statement that no
    // coalescing was available to do.
    //
    // So do not quote that ratio as this rail's payoff. What is measured is the
    // readback: the resident Store arms without reading the frame back off the
    // GPU, and `surface_resident_sync` counts the arms that had to. Coalescing
    // is a guard for a workload shape — several passes fully covering one
    // surface inside one submission — that nine driven boots, including a
    // SceneKit title and a live WebGL scene, never produced. It is kept because
    // landing a covered window costs a full-framebuffer write for nothing, not
    // because anything here has seen it pay.
    crate::runtime::drain::note_store_route("surface_flush");
    // The three questions every mapping-keyed window answers before its bytes
    // may be written; `guards` owns the ladder and the compute rail asks the
    // same one.
    if let Some(refusal) = mapping_window_refusal(state, host, key) {
        // Release the pin first. This arm returns before touching the frame, and
        // a `Resident` window holds a registry pin that nothing else will drop —
        // the allocation-failure reclaim and the idle drain both skip pinned
        // slots by design, so a pin leaked here strands a whole framebuffer for the guest
        // lifetime. That is the "~260 stale residents (~516 MiB)" shape, and the
        // generation drift is not rare: one in 85 s on a driven boot.
        release_window_pin_for_key(key, source);
        // Counted, not just logged, when the refusal loses a painted tile. Which
        // of the three do is `WindowRefusal`'s answer and not this rail's, so
        // the compute rail below cannot end up scoring the same losses
        // differently. The three resident-mismatch refusals further down carry
        // their own routes.
        if let Some(route) = refusal.lost_work_route(Rail::Render) {
            crate::runtime::drain::note_store_route(route);
        }
        crate::observe::fail(deferred_flush_lost(
            "render",
            key,
            None,
            &refusal.reason("Store"),
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
    // Land anything already armed over these pages *before* the frame is
    // acquired, not inside the write that follows.
    //
    // `write_bgra8_*` makes this call itself, and for the copying arms that is
    // where it belongs. The leased arm cannot afford it there: its frame is
    // borrowed from the engine's readback buffer, and a flush reached from
    // inside the write would read another resident — re-entering the engine
    // under a live lease, which is the one thing a holder may not do. Running it
    // here leaves the writer's own call nothing to find, because the only thing
    // that arms a window is a guest Store and no guest command is decoded inside
    // a writeback.
    //
    // Unconditional rather than gated on the arm taken below, so both arms reach
    // the write through the same state. A no-op on all but the rare mapping
    // carrying a second window.
    crate::runtime::mapping_write::flush_windows_under_bgra8_write(
        state,
        host,
        key.mapping_id,
        key.width,
        key.height,
    );
    // Owned rather than borrowed, and shared rather than owned outright: the
    // writeback's tail publishes this frame to the surface cache, and a cache
    // entry stores its frame behind an `Arc` precisely so that it and a window
    // can name one allocation. Handing the frame down as an `Arc` therefore ends
    // in one `Arc` clone where a borrow ended in a whole-frame copy — 1.21 ms of
    // memcpy per flush on the composite, about 100 times a second. `Owned`
    // already has one to clone.
    //
    // `Resident` does not go that way any more. It reads the resident back and
    // then *borrows* the staging buffer the readback landed in, because it has
    // no use for the frame after the scatter: owning it means one whole-frame
    // memcpy (`readback_split map_us`, ~0.82 ms of a 6.9 ms flush) whose only
    // consumer beyond the scatter is the host surface cache, and
    // `render_flush_cache_used` prices that consumer at 0.4 %. See
    // [`crate::backend::vulkan::engine::LeasedFrame`] for what the borrow costs
    // and [`crate::runtime::mapping_write::write_bgra8_uncached`] for what
    // happens to the cache entry instead.
    // Before the frame is acquired rather than after, so the one reading is
    // taken at the same point on every route this function has. The GPU-direct
    // arm below returns without ever reaching the copying arms, and a census
    // sampled only on the copying side would report this rail's guest-write
    // overlap as though the fast route did not exist.
    note_render_flush_over_guest_write(state, host, key);
    let frame: FlushFrame = match source {
        crate::model::RenderWindowSource::Owned(bytes) => FlushFrame::Owned(bytes.clone()),
        crate::model::RenderWindowSource::Resident { epoch } => {
            use crate::backend::vulkan::engine::ResidentContent;
            // The close of the interval `note_resident_window_armed` opened at
            // the Store. Taken before the epoch check, not after: a window that
            // the check refuses still consumed the arm, and leaving it counted
            // would make every later flush look like it had two outstanding.
            crate::runtime::drain::note_resident_window_flushed();
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
                    ResidentContent::Epoch(_) => ("resident_epoch_drift", "rendflush_epoch_drift"),
                };
                crate::runtime::drain::note_store_route(route);
                crate::observe::fail(format!(
                    "deferred_flush_lost kind=render mapping={} {}x{} fmt={:#x} gen={} \
                     reason={reason} want={epoch} live={live:?}",
                    key.mapping_id, key.width, key.height, key.pixel_format, key.map_generation
                ));
                return false;
            }
            // The frame need not exist on the host at all.
            //
            // Both arms below move it twice after the GPU has already written it
            // once: into a staging buffer, then out of that buffer into guest
            // RAM. `copy_target_to_guest_pages` makes the guest's own pages the
            // destination of the copy the GPU was going to make anyway, so the
            // two host passes — `readback_split map_us` plus `write_us`, about
            // 3.5 ms of a 6.9 ms flush — simply do not happen.
            //
            // Tried first because when it works there is nothing left for the
            // rest of this function to do, and declined for a named reason
            // otherwise. A decline is a routing answer, not a loss: the copying
            // arms below still land the frame, exactly as they did before this
            // rail existed.
            //
            // Ordered against the guest here in the one way that matters: this
            // runs inside `flush_all_windows_before_fence`, which is ordered
            // before `write_stamp`, and `write_stamp` settles every submitted
            // writeback before it moves the guest's fence. The pages hold the
            // frame before the guest is told anything about the submission that
            // produced it.
            //
            // The wait is not taken here, per window, and that is the point.
            // Doing it inline made this rail one blocking GPU round trip per
            // landed window — 369 a second, 1 360 us each, of which the device's
            // own timestamps priced 636 us as the copy and the rest as
            // submit-to-start plus signal-to-wake. Submitting them all and
            // settling once lets the queue run them back to back.
            match crate::runtime::mapping_write::write_bgra8_from_resident_gpu(
                state,
                host,
                key.mapping_id,
                &identity,
                key.width,
                key.height,
            ) {
                Ok(bytes) => {
                    // No unpin here, and that is this rail's contract rather
                    // than an omission: the copy is submitted and not yet
                    // executed, so the engine has taken the pin and releases it
                    // when the writeback settles. See
                    // `engine::copy_target_to_guest_pages`.
                    crate::runtime::drain::note_store_route("render_flush_gpu_direct");
                    return finish_render_flush(
                        state,
                        key,
                        Some(identity),
                        bytes as usize,
                        true,
                        // Nothing was published to the host surface cache: this
                        // rail never had a host copy to publish.
                        false,
                        started,
                    );
                }
                Err(decline) => {
                    // Latched per mapping as well as per reason. A host without
                    // `VK_EXT_external_memory_host` declines every flush of
                    // every surface, and a line per flush would drown the
                    // channel; a line per (reason, mapping) says which surfaces
                    // are paying the copy and why, once each.
                    crate::observe::Emit::decline("render_flush_gpu_declined", &decline)
                        .field("mapping", key.mapping_id)
                        .field("geom", format!("{}x{}", key.width, key.height))
                        .fail_once(u64::from(key.mapping_id));
                    crate::runtime::drain::note_store_route("render_flush_gpu_declined");
                }
            }
            // Borrow first, and only where the borrow needs no transformation.
            //
            // The writer below is declared in guest scanout order, so a resident
            // reporting semantic RGBA8 owes an R/B exchange before its bytes can
            // land — which is a whole-frame pass, and a pass over the staging
            // buffer at that. `into_bgra8` on an owned copy is the existing home
            // for it, so a non-BGRA resident takes the copying arm rather than
            // teaching the lease to rewrite memory it does not own. A `Surface`
            // resident is BGRA and that is the composite rail this rail's cost
            // lives on; reading the reported order rather than asserting one is
            // what keeps a future format change from landing R and B exchanged
            // in guest memory.
            match crate::backend::vulkan::engine::read_target_leased(&identity) {
                Ok(Some(leased)) if leased.bgra => {
                    crate::backend::vulkan::engine::unpin_resident_target(&identity);
                    flushed_from_resident = Some(identity);
                    crate::runtime::drain::note_store_route("render_flush_leased");
                    FlushFrame::Leased(leased)
                }
                // Either the pool declined the lease (uncached readback memory,
                // where reading the mapping in place is the *slower* shape) or
                // the resident is not in scanout order. Both take the copy, and
                // the leased frame — if there is one — is dropped first so its
                // slot is back in the pool before the second readback asks for
                // one.
                Ok(leased) => {
                    drop(leased);
                    crate::runtime::drain::note_store_route("render_flush_copied");
                    match crate::backend::vulkan::engine::read_target(&identity) {
                        Ok(rb) => {
                            crate::backend::vulkan::engine::unpin_resident_target(&identity);
                            flushed_from_resident = Some(identity);
                            FlushFrame::Owned(std::sync::Arc::new(rb.into_bgra8()))
                        }
                        Err(e) => {
                            crate::backend::vulkan::engine::unpin_resident_target(&identity);
                            crate::observe::fail(format!(
                                "deferred_flush_lost kind=render mapping={} {}x{} fmt={:#x} \
                                 gen={} reason=resident_read err={e}",
                                key.mapping_id,
                                key.width,
                                key.height,
                                key.pixel_format,
                                key.map_generation
                            ));
                            return false;
                        }
                    }
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
    let write_started = std::time::Instant::now();
    let ok = match &frame {
        FlushFrame::Owned(bytes) => crate::runtime::mapping_write::write_bgra8_owned(
            state,
            host,
            key.mapping_id,
            bytes,
            key.width.saturating_mul(4),
            key.width,
            key.height,
        ),
        FlushFrame::Leased(leased) => crate::runtime::mapping_write::write_bgra8_uncached(
            state,
            host,
            key.mapping_id,
            leased.bytes(),
            key.width.saturating_mul(4),
            key.width,
            key.height,
        ),
    };
    crate::runtime::drain::note_readback_phase(
        crate::runtime::drain::ReadbackPhase::Write,
        write_started.elapsed().as_micros() as u64,
    );
    // Whether this flush left a host surface cache copy behind, which decides
    // whether the witness has a cache leg to score at all. A borrowed frame
    // leaves none: it drops the entry because the memory holding it goes back to
    // the pool. The skipping write is the other writeback that leaves none, and
    // it is not reachable from here — this rail preserves nothing, so no store
    // it makes is a skipping one.
    let cache_stored = matches!(&frame, FlushFrame::Owned(_));
    // End the lease before anything below reaches the engine again — the
    // resident re-stamp does. A holder that blocks on the engine lock while a
    // teardown is waiting for exactly this lease is the deadlock `LeasedFrame`
    // forbids, and the frame has no reader left after the write in any case.
    let frame_len = frame.len();
    drop(frame);
    finish_render_flush(
        state,
        key,
        flushed_from_resident,
        frame_len,
        ok,
        cache_stored,
        started,
    )
}

/// The bookkeeping every landed render window owes, whichever route landed it.
///
/// Shared by the GPU-direct arm and the two copying arms because the obligations
/// are about the guest's pages having changed, not about who moved the bytes.
/// Splitting them was how the resident re-stamp below came to exist in one place
/// and be missing from another, and the symptom of that omission — a rail that
/// invalidates the resident holding exactly the content it just published — is a
/// loop rather than a wrong pixel, which is much harder to see.
#[cfg(feature = "backend-vulkan")]
#[allow(
    clippy::too_many_arguments,
    reason = "the window, what landed it, how much, and whether it succeeded"
)]
fn finish_render_flush(
    state: &mut DeviceState,
    key: &crate::model::ComputeStorageResidencyKey,
    flushed_from_resident: Option<crate::backend::vulkan::engine::TargetIdentity>,
    frame_len: usize,
    ok: bool,
    cache_stored: bool,
    started: std::time::Instant,
) -> bool {
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
            // The flush above copied this resident's pixels into the mapping's
            // guest pages, so the image has stopped being the only place they
            // exist and the reclaim paths may take it. Under `ok` only: a
            // refused write leaves the guest pages holding the previous frame,
            // and telling the registry otherwise would license destroying the
            // one copy of this one.
            crate::backend::vulkan::engine::note_resident_content_copied_out(&identity);
        }
        // Every copy this flush just made is unread until something reads it;
        // whatever was left of the previous flush's is scored now.
        let _ = note_render_flush_landed(state, key.mapping_id, cache_stored);
    }
    crate::runtime::drain::note_drain_phase(
        crate::runtime::drain::DrainPhase::Flush(crate::runtime::drain::FlushRail::Render),
        started,
    );
    crate::observe::line(format!(
        "render_deferred_flush mapping={} {}x{} fmt={:#x} ok={} bytes={} us={}",
        key.mapping_id,
        key.width,
        key.height,
        key.pixel_format,
        ok as u8,
        frame_len,
        started.elapsed().as_micros()
    ));
    ok
}

/// Where a landing render window's frame lives while it is being written.
///
/// The two differ in what the writeback may leave behind. `Owned` names an
/// allocation that outlives the flush, so the host surface cache can hold it for
/// a refcount; `Leased` names the engine's readback staging buffer, which goes
/// back to the pool a moment later and therefore cannot be what a cache entry
/// points at. See [`crate::runtime::mapping_write::write_bgra8_uncached`].
#[cfg(feature = "backend-vulkan")]
enum FlushFrame {
    Owned(std::sync::Arc<Vec<u8>>),
    Leased(crate::backend::vulkan::engine::LeasedFrame),
}

#[cfg(feature = "backend-vulkan")]
impl FlushFrame {
    fn len(&self) -> usize {
        match self {
            FlushFrame::Owned(bytes) => bytes.len(),
            FlushFrame::Leased(leased) => leased.bytes().len(),
        }
    }
}

#[cfg(not(feature = "backend-vulkan"))]
fn flush_render_one<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    _host: &mut M,
    key: &crate::model::ComputeStorageResidencyKey,
    _source: &crate::model::RenderWindowSource,
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
    // The same three questions the render rail asks, from the one ladder that
    // owns their order. Two unrelated `u32` generations are in scope while it
    // runs and they must not be confused in the log: `key.map_generation` is
    // the mapping's lifetime, which is what the drift arm compares, and
    // `generation` is the pinned resident's *content* generation, which only
    // `read_resident_storage` uses. `deferred_flush_lost` keeps them in
    // separate fields for that reason.
    if let Some(refusal) = mapping_window_refusal(state, host, key) {
        // This rail holds a pinned resident rather than a registry pin, so its
        // release is a different call from the render rail's — which is why the
        // ladder returns the refusal instead of reporting it.
        crate::backend::vulkan::engine::unpin_resident_storage(key);
        // Same census as the render rail, from the same table. This rail used
        // to count none of its three refusals while losing the dispatch output
        // just as permanently.
        if let Some(route) = refusal.lost_work_route(Rail::Compute) {
            crate::runtime::drain::note_store_route(route);
        }
        crate::observe::fail(deferred_flush_lost(
            "compute",
            key,
            Some(generation),
            &refusal.reason("dispatch"),
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
    crate::runtime::drain::note_drain_phase(
        crate::runtime::drain::DrainPhase::Flush(crate::runtime::drain::FlushRail::Storage),
        started,
    );
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
