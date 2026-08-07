//! Land a render Store's frame in the guest's pages, at the Store.
//!
//! A type-11 render Store names a mapping and a resident image the draw just
//! rendered into. This module copies the one into the other and returns. There
//! is no window, no pin held across the call, and nothing to land later.
//!
//! # Why the frame is written here rather than deferred
//!
//! It used to be deferred. A Store armed a window naming the pinned resident,
//! and the window was landed either by the next completion stamp or by a host
//! path that touched the mapping's bytes first. The argument for that shape was
//! coalescing: several passes fully covering one surface inside one submission
//! would land once instead of once each.
//!
//! That coalescing was measured and it never occurred. Arms and lands were
//! equal on every census line of an accumulated driven x86/Vulkan log — 193 458
//! each across 1 780 lines, not one line differing — because no later Store
//! ever fully covered a live window. A ratio pinned at 1.0 by the workload is
//! not coalescing; it is the statement that none was available.
//!
//! What the rail did buy is real and is kept: the Store does not read the frame
//! back off the GPU. [`crate::runtime::mapping_write::write_bgra8_from_resident_gpu`]
//! makes the guest's own pages the destination of the copy the GPU was going to
//! make anyway, so nothing crosses host memory on the arm that runs. Landing at
//! the Store keeps that and drops the window.
//!
//! # What the window cost that this cannot
//!
//! Every hazard the deferred rail had to answer came from the window outliving
//! the Store, and none of them can arise here:
//!
//! * **Resident drift.** A window promised pixels from a slot a later draw
//!   could render over, so the land compared a content epoch and refused on a
//!   mismatch — losing the frame. Here the resident is the one the draw just
//!   produced and nothing runs in between.
//! * **Pin leaks.** A window held a registry pin that the reclaim paths skip by
//!   design, so a pin dropped on any early return stranded a framebuffer for the
//!   guest's lifetime. Nothing is pinned here.
//! * **Page recycling.** The guest could hand a window's pages to a different
//!   allocation before it landed, which is the PTE-corruption class the window
//!   guards existed for. The pages cannot move inside this call.
//! * **Write ordering against the guest's own claim.** A window could hold
//!   pixels rendered *before* a guest CPU write to the same resource, and
//!   landing it afterwards clobbered the guest's bytes with stale ones. The
//!   Store and the write are now ordered by when they happen.
//!
//! # Ordering against the guest
//!
//! The copy is recorded into the engine's command stream, not waited on. It is
//! ordered before the guest can observe it by the completion stamp: the stamp
//! word is written behind an `ALL_COMMANDS -> TRANSFER` barrier and every
//! submitted guest-page write settles before the stamp moves. See
//! `backend::vulkan::engine::write_stamp_after_guest_writes`.

use crate::model::DeviceState;
#[cfg(feature = "backend-vulkan")]
use crate::runtime::host::{HostMemory, HostOps};

/// Block until every guest-page write this device has submitted has executed.
///
/// The writes above are recorded into the engine's command stream and not
/// waited on, which is what makes a Store cheap. A **host-side** reader of the
/// same guest bytes — a mapping read, a CPU seed, a present capture — is not
/// ordered against them by anything the GPU knows about, so it has to settle
/// them first or it reads the pre-Store bytes.
///
/// The guest is ordered separately and does not come through here: its
/// completion stamp is written behind a barrier that already subsumes these
/// copies (`engine::write_stamp_after_guest_writes`).
///
/// Free when nothing is outstanding — the engine keeps a debt flag and this
/// returns without touching a queue when it is clear.
pub fn settle_guest_writes() {
    #[cfg(feature = "backend-vulkan")]
    crate::backend::vulkan::engine::quiesce_guest_writes();
}

/// Release the engine residents of linear cache entries whose task or object
/// the guest deleted this drain.
///
/// Two releases, and dropping either one is a leak in the opposite direction: an
/// unpin alone leaves the image holding the only copy of content nothing may
/// reclaim, and retiring the content alone leaves a pinned slot no reclaim path
/// may take. Together they make the image ordinarily evictable.
///
/// Task teardown means the GPU VA maps are gone, so nothing here writes guest
/// pages — the deleted object's bytes are not guest work any more.
pub fn retire_linear_residents(state: &mut DeviceState) {
    if state.retired_linear_residents.is_empty() {
        return;
    }
    let retired = std::mem::take(&mut state.retired_linear_residents);
    // The engine that holds these pins is the Vulkan one; a `backend-metal`
    // build arms nothing that could have pinned them, so taking the list is the
    // whole of the work there.
    #[cfg(feature = "backend-vulkan")]
    for key in &retired {
        crate::backend::vulkan::engine::unpin_resident_storage(key);
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
    #[cfg(not(feature = "backend-vulkan"))]
    drop(retired);
}

/// Copy `identity`'s pixels into `mapping_id`'s guest pages.
///
/// `true` when the guest's pages hold the frame. `false` is a real loss and is
/// reported on the failure channel by the arm that refused — the caller has no
/// second copy to fall back to, because this rail never made one.
#[cfg(feature = "backend-vulkan")]
pub fn store_render_frame<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mapping_id: u32,
    identity: &crate::backend::vulkan::engine::TargetIdentity,
    width: u32,
    height: u32,
) -> bool {
    let started = std::time::Instant::now();
    crate::runtime::drain::note_store_route("surface_flush");
    // The GPU writes the guest's pages directly. Tried first because when it
    // works there is nothing left to do: no staging buffer is mapped and no
    // host pass over the frame happens at all.
    match crate::runtime::mapping_write::write_bgra8_from_resident_gpu(
        state, host, mapping_id, identity, width, height,
    ) {
        Ok(bytes) => {
            crate::runtime::drain::note_store_route("render_flush_gpu_direct");
            finish(state, mapping_id, identity, bytes as usize, started);
            return true;
        }
        Err(decline) => {
            // Latched per mapping as well as per reason: a host without
            // `VK_EXT_external_memory_host` declines every Store of every
            // surface, and a line each would drown the channel.
            crate::observe::Emit::decline("render_flush_gpu_declined", &decline)
                .field("mapping", mapping_id)
                .field("geom", format!("{width}x{height}"))
                .fail_once(u64::from(mapping_id));
            crate::runtime::drain::note_store_route("render_flush_gpu_declined");
        }
    }
    // The copying arms. These are the only arms on a host that cannot import
    // guest RAM, and the arm a discrete GPU takes regardless.
    //
    // Borrow the readback where it needs no transformation. The writer below is
    // declared in guest scanout order, so a resident reporting semantic RGBA8
    // owes an R/B exchange first — a whole-frame pass, and `into_bgra8` on an
    // owned copy is its home, so a non-BGRA resident takes the copy rather than
    // teaching the lease to rewrite memory it does not own.
    let bpr = width.saturating_mul(4);
    let write_started = std::time::Instant::now();
    let (ok, frame_len) = match crate::backend::vulkan::engine::read_target_leased(identity) {
        Ok(Some(leased)) if leased.bgra => {
            crate::runtime::drain::note_store_route("render_flush_leased");
            let len = leased.bytes().len();
            let ok = crate::runtime::mapping_write::write_bgra8_uncached(
                state,
                host,
                mapping_id,
                leased.bytes(),
                bpr,
                width,
                height,
            );
            // End the lease before anything below reaches the engine again: the
            // re-stamp in `finish` does, and a holder blocking on the engine
            // lock while a teardown waits for this lease is the deadlock
            // `LeasedFrame` forbids.
            drop(leased);
            (ok, len)
        }
        // Either the pool declined the lease (uncached readback memory, where
        // reading the mapping in place is the slower shape) or the resident is
        // not in scanout order. Drop any leased frame first so its slot is back
        // in the pool before the second readback asks for one.
        Ok(leased) => {
            drop(leased);
            crate::runtime::drain::note_store_route("render_flush_copied");
            match crate::backend::vulkan::engine::read_target(identity) {
                Ok(rb) => {
                    // Shared rather than owned outright: the write's tail
                    // publishes this frame to the surface cache, and a cache
                    // entry holds its frame behind an `Arc` precisely so the two
                    // can name one allocation instead of copying it.
                    let bytes = std::sync::Arc::new(rb.into_bgra8());
                    let len = bytes.len();
                    let ok = crate::runtime::mapping_write::write_bgra8_owned(
                        state, host, mapping_id, &bytes, bpr, width, height,
                    );
                    (ok, len)
                }
                Err(e) => {
                    crate::observe::fail(format!(
                        "render_store_lost mapping={mapping_id} {width}x{height} \
                         reason=resident_read err={e}"
                    ));
                    return false;
                }
            }
        }
        Err(e) => {
            crate::observe::fail(format!(
                "render_store_lost mapping={mapping_id} {width}x{height} \
                 reason=resident_read err={e}"
            ));
            return false;
        }
    };
    crate::runtime::drain::note_readback_phase(
        crate::runtime::drain::ReadbackPhase::Write,
        write_started.elapsed().as_micros() as u64,
    );
    if !ok {
        crate::observe::fail(format!(
            "render_store_lost mapping={mapping_id} {width}x{height} reason=write_refused"
        ));
        return false;
    }
    finish(state, mapping_id, identity, frame_len, started);
    true
}

/// Hand the currency witness back to the image the frame came out of, and score
/// the write.
///
/// `write_bgra8_*` ends in `mark_mapping_written`, which advances the mapping's
/// `surface_content_epoch` — correctly, because its guest pages did change. But
/// the *pixels* did not: they are this resident's, copied out of it one
/// statement ago. Leaving the stamp behind invalidates a resident that holds
/// exactly the mapping's content, which costs the next Load its elision and
/// sends it to a CPU seed for bytes it already has.
#[cfg(feature = "backend-vulkan")]
fn finish(
    state: &mut DeviceState,
    mapping_id: u32,
    identity: &crate::backend::vulkan::engine::TargetIdentity,
    frame_len: usize,
    started: std::time::Instant,
) {
    if let Some(epoch) = state
        .mappings
        .get(&mapping_id)
        .map(|m| m.surface_content_epoch)
    {
        crate::backend::vulkan::engine::stamp_resident_content_epoch(identity, epoch);
    }
    // The copy above means this image has stopped being the only place these
    // pixels exist, so the reclaim paths may take it.
    crate::backend::vulkan::engine::note_resident_content_copied_out(identity);
    crate::runtime::drain::note_drain_phase(
        crate::runtime::drain::DrainPhase::Flush(crate::runtime::drain::FlushRail::Render),
        started,
    );
    crate::observe::line(format!(
        "render_store mapping={mapping_id} bytes={frame_len} us={}",
        started.elapsed().as_micros()
    ));
}
