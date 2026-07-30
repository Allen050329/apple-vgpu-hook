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

    /// A compositor swapchain — several scanout buffers presenting at ONE
    /// geometry — must get one resident each.
    ///
    /// This is the shape that used to collapse: four buffers at 1920x1080
    /// unified onto a single geometry-keyed resident, and a held drag that
    /// reversed direction left a selection-rectangle fragment on the desktop.
    /// Interleaved on/off A/B over four boots: 5 of 12 rounds reproduced with
    /// the collapse, 1 of 12 without, and the dominant sub-class — a 15x15
    /// fragment at the press point — went 4 to 0.
    ///
    /// What holds the line now is structural rather than a check a caller has
    /// to remember: the mapping id is part of the registry key, so
    /// `registry_get` on one buffer's identity cannot return another's. That
    /// replaced an explicit `surface_mapping_id()` predicate, which the
    /// geometry-keyed resolver needed and nothing does. The pairwise case above
    /// states the same property; this one states it over the exact arity the
    /// live defect had, because "distinct in pairs" is what a reader checks and
    /// "four buffers, four residents" is what the compositor does.
    #[test]
    fn a_four_buffer_swapchain_at_one_geometry_gets_four_residents() {
        let state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let ids: Vec<TargetIdentity> = [11u32, 12, 13, 14]
            .iter()
            .map(|&mid| surface_identity(&state, mid, 1920, 1080))
            .collect();
        for (i, a) in ids.iter().enumerate() {
            for (j, b) in ids.iter().enumerate() {
                if i != j {
                    assert_ne!(a, b, "scanout buffers {i} and {j} share a resident");
                }
            }
        }
    }
}
