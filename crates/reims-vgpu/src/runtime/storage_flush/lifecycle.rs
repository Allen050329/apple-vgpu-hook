//! Ending a window without landing it.
//!
//! A window can also be torn down (its mapping is gone), superseded (a later
//! Store covers the same bytes) or unpinned (the resident it held is being
//! released). None of these owe the guest anything, but every one of them must
//! drop the pin the window took, which is why they live together rather than
//! beside the rail that triggers each.

#[cfg(feature = "backend-vulkan")]
use super::land::render_window_identity;
use super::owner_slug;
use crate::model::DeviceState;

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
/// without bound. Both paths that can give a resident back select it through
/// `recoverable_residents`, which requires `pin_count == 0` — the
/// allocation-failure reclaim and the idle drain — so a slot that gets there can
/// never be reclaimed again: the "~260 stale residents (~516 MiB) pinned for the
/// guest lifetime" shape, arrived at one frame at a time.
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
/// to drop it: the allocation-failure reclaim and the idle drain both skip
/// pinned slots by design, so a leaked pin strands a whole framebuffer for the
/// guest lifetime rather than merely delaying a reclaim.
#[cfg(feature = "backend-vulkan")]
pub(super) fn release_window_pin_for_key(
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
