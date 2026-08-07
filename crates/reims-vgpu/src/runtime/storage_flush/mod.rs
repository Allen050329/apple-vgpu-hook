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
//! Guest CPU accesses that never cross our host paths cannot be intercepted by
//! anything in this device — the same accepted exposure as resident render
//! targets under `skip_readback`. That is a statement about the device and not
//! about what is possible; `flush_mapping_windows_before_fence` records what a
//! hypervisor-level witness for them would take, and why it has to be a
//! measurement before it is a mechanism. Choke points: `mapping_write` read/write entries,
//! `mapper::read/write_mapping_bytes`, and the drain unmap/ReplacePhysical
//! sites (which drop-with-fail instead of writing through recycled pages).
//!
//! # The parts
//!
//! A window is armed elsewhere, and everything that can then happen to it is
//! one of five things. Each has its own module, and the dependency edges run
//! only downward through this list:
//!
//! * [`access`] — the flush-on-access choke points. A host-side reader or
//!   writer arrives, and the windows covering its bytes are landed first.
//! * [`fence`] — the before-fence passes. A completion stamp is about to be
//!   signalled, so every window still held is landed whether or not anybody
//!   asked, because the guest may read the bytes through its own mapping.
//! * [`land`] — landing one window: resident to host, host to guest pages,
//!   plus the identity checks that decide whether the window still describes
//!   the surface it was armed for.
//! * [`guards`] — whether the guest pages a window names are still the
//!   surface's. Landing into recycled pages is the PTE-corruption class.
//! * [`lifecycle`] — the three ways a window ends *without* landing: torn
//!   down, superseded, or unpinned.
//!
//! [`report`] sits beside them all and decides nothing: it holds the readings
//! this rail's cost arguments are built from. It shares that membership rule,
//! and its name, with `runtime::exec::report`.
//!
//! # This entire rail is Vulkan-only
//!
//! It is the largest architectural fact about this rail and it used to be
//! spelled nowhere but in thirty separate `#[cfg]` attributes. Nothing here
//! names `backend::metal`; what a window defers is a *pinned engine resident*,
//! and the engine is `backend::vulkan::engine`. On a `backend-metal` build —
//! which is every build without `backend-vulkan`, since `lib.rs` requires
//! exactly one — [`guards`] does not exist at all and the seven entry points
//! across [`fence`] and [`land`] compile to stubs.
//!
//! **Nothing can arm a window on such a build**, which is why three of those
//! stubs are silently empty and the other four are fail-visible. Every
//! production site that arms one is inside `backend-vulkan`-gated code — the
//! two render Stores in `draw::vulkan` (behind that module's own gate) and
//! the two compute-storage arms in `compute_exec::execute_dispatch_linux`
//! (behind the function's) — so:
//!
//! * [`fence`]'s `flush_gva_windows_before_fence`,
//!   `flush_linear_windows_before_fence` and `flush_mapping_windows_before_fence`
//!   land *every armed window*, and on this arm the set is empty by
//!   construction. An empty pass loses nothing, so saying nothing is correct.
//! * [`land`]'s `flush_gva_one`, `flush_linear_one`, `flush_render_one` and
//!   `flush_storage_one` are handed one key a caller believes is armed. Reaching
//!   one here means someone believed in an obligation that cannot exist, so each
//!   reports `deferred_flush_lost … reason=no_backend`.
//!
//! The premise those three stubs rest on is that **every arm site sits under a
//! `backend-vulkan` gate.** A new one outside such a gate turns them from
//! unreachable into silent losses. A source scan used to hold that premise;
//! nothing does now, so check the gate when adding an arm site.

pub(crate) mod access;
pub(crate) mod fence;
#[cfg(feature = "backend-vulkan")]
pub(crate) mod guards;
pub(crate) mod land;
pub(crate) mod lifecycle;
pub(crate) mod report;

pub(crate) use access::{
    flush_gva_exact, flush_intersecting, flush_intersecting_task_gva, flush_mapping_for_guest_read,
};
pub(crate) use fence::flush_all_windows_before_fence;
pub(crate) use land::{retire_gva_windows, retire_linear_residents};
pub(crate) use lifecycle::drop_windows;
pub(crate) use report::{note_render_flush_cache_read, note_render_flush_pages_read};

// The two landing entry points join the identity helpers behind this gate:
// they are reached only from `backend-vulkan` code, because this whole
// deferred-writeback rail is the Vulkan arm's — which is why the Metal
// `*_before_fence` bodies are empty.
#[cfg(feature = "backend-vulkan")]
pub(crate) use land::{
    flush_gva_one, flush_linear_one, gva_window_identity, render_window_identity,
};
#[cfg(feature = "backend-vulkan")]
pub(crate) use lifecycle::{release_window_pin, supersede_covered_render_windows};

pub(crate) fn owner_slug(owner: &crate::model::DeferredOwner) -> &'static str {
    match owner {
        crate::model::DeferredOwner::Storage { .. } => "compute",
        crate::model::DeferredOwner::Render { .. } => "render",
    }
}

#[cfg(all(test, feature = "backend-vulkan"))]
mod render_flush_witness_tests;

#[cfg(test)]
mod tests;
