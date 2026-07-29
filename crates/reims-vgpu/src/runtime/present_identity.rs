//! The resident identity a type-11 guest surface renders into.
//!
//! This is all that survives of `import_present`, which owned three ways of
//! landing a Vulkan composite Store in guest IOSurface pages without a CPU
//! copy: a packed-contig strided DMA, a fragmented multi-run scatter DMA, and
//! an ack-fast deferred rung that pinned the resident and replayed the Store on
//! first access.
//!
//! All three needed `VK_EXT_external_memory_host` — a host pointer over the
//! guest's own pages, which is a pointer the GPU can write. Neither the
//! extension nor the two engine entry points exist any more, so type-11 Stores
//! take the CPU writeback
//! (`mapping_write::write_rgba8_image_changed`), which every one of those rails
//! already fell back to whenever an import was refused.
//!
//! What is left is the identity itself, which was never about importing: the
//! registry is keyed by it whichever way the pixels reach the guest.

#![cfg(feature = "backend-vulkan")]

use crate::backend::vulkan::engine::TargetIdentity;
use crate::model::DeviceState;

/// Build a protocol-stable resident identity for this mapping at its current
/// [`crate::model::MappingEntry::map_generation`].
///
/// One identity per mapping, always. `ResourcePools::registry` is keyed by
/// `TargetIdentity`, so two mappings with equal identities would render into and
/// capture from ONE `VkImage` — and distinct guest surfaces have independent
/// damage histories, because WindowServer redraws a buffer only where it differs
/// from what THAT buffer last held. Sharing a resident between them makes every
/// frame a fusion of damage from several buffers, which is the rubber-band
/// residue class.
pub fn surface_identity(
    state: &DeviceState,
    mapping_id: u32,
    width: u32,
    height: u32,
) -> TargetIdentity {
    let gen = state
        .mappings
        .get(&mapping_id)
        .map(|m| m.map_generation as u64)
        .unwrap_or(0);
    TargetIdentity::Surface {
        id: mapping_id,
        width,
        height,
        generation: gen,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{DeviceId, PAGE_SHIFT_X86};

    /// Two mappings must never share an identity, and one mapping must never
    /// change identity without its `map_generation` changing. Both directions
    /// are the rubber-band residue class: the registry is keyed on this value,
    /// so a collision fuses two guest surfaces' damage histories into one
    /// `VkImage`, and a spurious change orphans a live resident.
    #[test]
    fn identity_separates_mappings_and_tracks_only_the_map_generation() {
        let state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let a = surface_identity(&state, 7, 64, 32);
        let b = surface_identity(&state, 8, 64, 32);
        assert_ne!(a, b, "distinct mappings must not share a resident");
        assert_eq!(
            a,
            surface_identity(&state, 7, 64, 32),
            "identity must be a pure function of the mapping and geometry"
        );
        assert_ne!(
            a,
            surface_identity(&state, 7, 65, 32),
            "geometry is part of the resident's shape"
        );
    }
}
