//! Host GPU capability classification — the single source of truth for what
//! the bound Vulkan device can do.
//!
//! Everything here is *measured* on the device at create time and consumed
//! either by a decision or by the one-shot `vk_caps` line. There is
//! deliberately no derived taxonomy on top of it: a classification nothing
//! branches on cannot be wrong in a way anyone notices, and the one that used to
//! live here was wrong — it printed `handoff=dmabuf_fd handoff_declined=[]` on
//! boots where the dmabuf arm was never entered, because the ladder that produced
//! the string and the branch that picks the present path were separate pieces of
//! code that had never been made to agree. The `dmabuf` bit itself went the same
//! way once the export rail it described was deleted: nothing branched on it and
//! nothing could.
//!
//! * [`memory_topology::MemoryTopology`] — `Unified` vs `Discrete` selects an
//!   allocation *preference*, never a different observable result. Live: every
//!   allocation names a [`MemoryClass`] and this module turns it into flags.
//! * [`device_features`] — which optional device features are queried and
//!   enabled, in one place, so no site can ask about one it did not request.
//! * [`DriverQuirk`] — the only place driver identity may change behavior.
//!
//! **Vulkan 1.2 is the baseline on every supported host.** See [`api_floor`]
//! for why the API version is a floor check and nothing more; `gate.rs` scans
//! the crate to keep it true.
//!
//! # Rules for adding a capability gate
//!
//! 1. Gate on the **capability**, never on a driver name, vendor id, an API
//!    version, or `VK_KHR_portability_subset`. If a driver quirk genuinely needs
//!    keying on the driver, add a named [`DriverQuirk`] with the observed
//!    failure in its doc comment — so the next reader knows it is a workaround,
//!    not a design.
//! 2. Put the field on [`HostGpuCaps`] only if something reads it. A capability
//!    that only reaches a log line is a fact, and belongs in the format string
//!    at the site that measured it.

pub mod api_floor;
pub mod device_features;
pub mod device_select;
#[cfg(test)]
mod gate;
pub mod memory_topology;

pub use device_select::{rank_physical_device, select_physical_device};
pub use memory_topology::{MemoryClass, MemoryProfile, MemoryTopology, TopologySignal};

use ash::vk;

/// Driver-identity workarounds. Each variant documents the concrete failure it
/// works around and how to retire it. This is the ONLY place driver identity is
/// allowed to change behavior — see the module rules.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DriverQuirk {
    /// MoltenVK reported `DEVICE_LOST` when a deferred draw batch was submitted
    /// by a later non-joinable draw after the target was already marked
    /// resident-ready. Retire by reproducing the batch submit on MoltenVK with
    /// validation on and fixing the ordering; the batching itself is portable.
    pub no_deferred_draw_batching: bool,
    /// GPU-only guest-visible content rails are held back where a device
    /// recreate drops registry residents and guest pages must stay
    /// authoritative. Retire once the device-loss source is closed.
    pub guest_pages_stay_authoritative: bool,
}

impl DriverQuirk {
    /// Quirks implied by a device advertising `VK_KHR_portability_subset`
    /// (in practice: MoltenVK).
    pub fn for_portability_subset(portability_subset: bool) -> Self {
        Self {
            no_deferred_draw_batching: portability_subset,
            guest_pages_stay_authoritative: portability_subset,
        }
    }
}

/// Everything the engine is allowed to know about the bound device's
/// capabilities, classified once at device create.
#[derive(Clone, Debug)]
pub struct HostGpuCaps {
    pub memory: MemoryProfile,
    pub quirks: DriverQuirk,
    /// `VK_KHR_portability_subset` was advertised. Kept for the selection log
    /// line and for constructing [`DriverQuirk`] — never gate behavior on it
    /// directly.
    pub portability_subset: bool,
    /// Device `apiVersion` as reported, for the selection log line.
    pub device_api_version: u32,
    pub device_type: vk::PhysicalDeviceType,
}

impl HostGpuCaps {
    /// Flags to request for `class` on this device.
    pub fn memory_request(&self, class: MemoryClass) -> memory_topology::MemoryRequest {
        self.memory.topology.request(class)
    }

    /// One-shot, fail-visible summary of the classification. Load-bearing for
    /// portability debugging: it names the memory topology, the signal that
    /// decided it, and the heap sizes that signal was read from. Every field is
    /// something the device reported.
    pub fn selection_line(&self, device_name: &str) -> String {
        format!(
            "vk_caps api={} baseline={} memory={} memory_signal={} device_local_mb={} host_visible_device_local_mb={} portability_subset={} type={:?} name={device_name:?}",
            api_floor::version_str(self.device_api_version),
            api_floor::version_str(api_floor::MIN_SUPPORTED_API),
            self.memory.topology.slug(),
            self.memory.signal.slug(),
            self.memory.device_local_bytes >> 20,
            self.memory.host_visible_device_local_bytes >> 20,
            self.portability_subset,
            self.device_type,
        )
    }

    /// Summary for a device that only **consumes** finished frames — the host
    /// window's `VkDevice`, which on a hybrid host may be a different physical
    /// GPU than the engine's. Omits the heap sizes: they describe how guest
    /// memory reaches the GPU, which is the engine device's job.
    pub fn consumer_line(&self, device_name: &str) -> String {
        format!(
            "vk_caps role=consumer api={} baseline={} memory={} memory_signal={} portability_subset={} type={:?} name={device_name:?}",
            api_floor::version_str(self.device_api_version),
            api_floor::version_str(api_floor::MIN_SUPPORTED_API),
            self.memory.topology.slug(),
            self.memory.signal.slug(),
            self.portability_subset,
            self.device_type,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use memory_topology::fixtures;

    fn caps(api: u32, props: &vk::PhysicalDeviceMemoryProperties) -> HostGpuCaps {
        HostGpuCaps {
            memory: memory_topology::classify_memory(props),
            quirks: DriverQuirk::default(),
            portability_subset: false,
            device_api_version: api,
            device_type: vk::PhysicalDeviceType::DISCRETE_GPU,
        }
    }

    /// The selection line names the topology, the signal that decided it, and the
    /// heap sizes that signal was read from — what a portability bug report needs.
    /// Every assertion here is a field a reader greps for.
    #[test]
    fn selection_line_carries_the_diagnosis() {
        let c = caps(vk::API_VERSION_1_3, &fixtures::nvidia_discrete());
        let line = c.selection_line("NVIDIA GeForce RTX 5080");
        assert!(line.contains("memory=discrete"), "{line}");
        assert!(line.contains("memory_signal=separate_host_heap"), "{line}");
        assert!(line.contains("device_local_mb=16384"), "{line}");
        // The baseline is stated on every line so no reader mistakes the
        // device's reported version for a requirement.
        assert!(line.contains("baseline=1.2"), "{line}");
    }

    /// The API version does not change the classification. Getting this wrong is
    /// how the retired tier axis smuggled a capability in under "is 1.3".
    #[test]
    fn the_api_version_does_not_change_the_classification() {
        let props = fixtures::intel_igpu();
        for api in [
            vk::API_VERSION_1_2,
            vk::API_VERSION_1_3,
            vk::make_api_version(0, 1, 4, 334),
        ] {
            let line = caps(api, &props).selection_line("dev");
            assert!(line.contains("memory=unified"), "{line}");
        }
    }

    /// A unified-memory host classifies as unified whatever else it reports, and
    /// the line is where "why is this host slow" starts.
    #[test]
    fn a_unified_memory_host_says_so() {
        let line =
            caps(vk::API_VERSION_1_2, &fixtures::apple_m3_max()).selection_line("Apple M3 Max");
        assert!(line.contains("memory=unified"), "{line}");
        assert!(line.contains("memory_signal="), "{line}");
    }

    /// The consumer line answers for the window's own device and does not
    /// restate the engine's heap classification, which would contradict the
    /// engine's line for the same GPU on a hybrid host.
    #[test]
    fn consumer_line_omits_the_engine_only_fields() {
        let line = caps(vk::API_VERSION_1_2, &fixtures::intel_igpu()).consumer_line("iGPU");
        assert!(line.contains("role=consumer"), "{line}");
        assert!(line.contains("memory=unified"), "{line}");
        assert!(!line.contains("device_local_mb"), "{line}");
    }

    /// Quirks are derived from portability-subset in ONE place, so no other
    /// site needs to know what that extension implies.
    #[test]
    fn quirks_derive_from_portability_subset_once() {
        let off = DriverQuirk::for_portability_subset(false);
        assert!(!off.no_deferred_draw_batching);
        assert!(!off.guest_pages_stay_authoritative);
        let on = DriverQuirk::for_portability_subset(true);
        assert!(on.no_deferred_draw_batching);
        assert!(on.guest_pages_stay_authoritative);
    }
}
