//! Host GPU capability classification — the single source of truth for what
//! the bound Vulkan device can do.
//!
//! # The support matrix
//!
//! `reims-vgpu` targets **four** host GPU configurations as first-class,
//! separately-tested rows. They are the cross product of two independent axes:
//!
//! | | **DMA** (some crossing is zero-copy) | **non-DMA** (every crossing copies) |
//! |---|---|---|
//! | **Unified memory** | Apple M-series via MoltenVK; Intel/AMD iGPU on Mesa | an iGPU exposing no sharing mechanism |
//! | **Discrete memory** | the RTX 5080 dev host | a GTX 750 Ti-class dGPU on a bare 1.2 driver |
//!
//! * [`memory_topology::MemoryTopology`] — `Unified` vs `Discrete` selects an
//!   allocation *preference*, never a different observable result.
//! * [`zero_copy::DmaSupport`] — can the GPU address memory it did not itself
//!   allocate? Resolved per rail (guest read, guest write, frame handoff), each
//!   independently feature-gated with a named decline.
//! * [`frame_interop::FrameHandoff`] — the display rail specifically: how a
//!   finished frame reaches **our own window**, which is the only frame sink on
//!   every row.
//!
//! **Vulkan 1.2 is the baseline for all four cells, not an axis.** See
//! [`api_floor`] for why the API version is a floor check and nothing more.
//!
//! # Why this module exists
//!
//! Before it, capability lived as a handful of loose booleans on the device
//! context and behavior was gated on *driver identity* (`portability_subset`,
//! i.e. "is this MoltenVK") rather than on the capability actually being
//! decided. That silently coupled unrelated properties: a driver without dmabuf
//! got the MoltenVK code path's answer to questions MoltenVK was never being
//! asked. Classification now happens once, in one place, with named reasons on
//! every decline, and the four rows are asserted by [`matrix`] tests that need
//! no GPU.
//!
//! # Rules for adding a capability gate
//!
//! 1. Gate on the **capability**, never on a driver name, vendor id, an API
//!    version, or `VK_KHR_portability_subset`. If a driver quirk genuinely needs
//!    keying on the driver, add a named [`DriverQuirk`] with the observed
//!    failure in its doc comment — so the next reader knows it is a workaround,
//!    not a design.
//! 2. Every decline emits a fail-visible line naming the missing capability.
//! 3. Add the row to [`matrix::MATRIX`] and let the completeness test prove the
//!    fallback exists on all four cells.

pub mod api_floor;
pub mod device_features;
pub mod device_select;
pub mod frame_interop;
#[cfg(test)]
mod gate;
pub mod matrix;
pub mod memory_topology;
pub mod zero_copy;

pub use device_select::{rank_physical_device, select_physical_device};
pub use frame_interop::{FrameHandoff, HandoffLadder};
pub use matrix::{FrameSink, SupportCell};
pub use memory_topology::{MemoryClass, MemoryProfile, MemoryTopology, TopologySignal};
pub use zero_copy::{DmaMechanisms, DmaSupport, GuestRead, GuestWrite, ZeroCopyProfile};

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
    /// Which crossings of the guest → GPU → display path avoid a CPU copy, and
    /// which degraded and why.
    pub zero_copy: ZeroCopyProfile,
    pub handoff: HandoffLadder,
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
    /// The matrix cell this device lands in.
    pub fn cell(&self) -> SupportCell {
        SupportCell::of(self.memory.topology, self.zero_copy.support())
    }

    /// Flags to request for `class` on this device.
    pub fn memory_request(&self, class: MemoryClass) -> memory_topology::MemoryRequest {
        self.memory.topology.request(class)
    }

    /// Whether the GPU can read and write the guest's own pages in place.
    ///
    /// Call sites ask this instead of re-checking whether
    /// `VK_EXT_external_memory_host` is enabled, so the answer cannot drift
    /// between the rails that depend on it.
    pub fn imports_guest_pages(&self) -> bool {
        self.zero_copy.guest_read.is_zero_copy()
    }

    /// One-shot, fail-visible summary of the classification. Load-bearing for
    /// portability debugging: it names the matrix cell, the signal that decided
    /// the memory topology, the rung each zero-copy rail landed on, and why any
    /// rail or handoff rung was skipped. "Why is this host slow?" should be
    /// answerable from this line alone.
    pub fn selection_line(&self, device_name: &str) -> String {
        format!(
            "vk_caps cell={} api={} baseline={} memory={} memory_signal={} dma={} dma_mechanisms={} guest_read={} guest_write={} zero_copy_declined=[{}] device_local_mb={} host_visible_device_local_mb={} handoff={} handoff_declined=[{}] sink={} portability_subset={} type={:?} name={device_name:?}",
            self.cell().slug(),
            api_floor::version_str(self.device_api_version),
            api_floor::version_str(api_floor::MIN_SUPPORTED_API),
            self.memory.topology.slug(),
            self.memory.signal.slug(),
            self.zero_copy.support().slug(),
            self.zero_copy.mechanisms.slug(),
            self.zero_copy.guest_read.slug(),
            self.zero_copy.guest_write.slug(),
            self.zero_copy.declined_summary(),
            self.memory.device_local_bytes >> 20,
            self.memory.host_visible_device_local_bytes >> 20,
            self.handoff.chosen().slug(),
            self.handoff.declined_summary(),
            FrameSink::HostOwnedWindow.slug(),
            self.portability_subset,
            self.device_type,
        )
    }

    /// Summary for a device that only **consumes** finished frames — the host
    /// window's `VkDevice`, which on a hybrid host may be a different physical
    /// GPU than the engine's.
    ///
    /// Deliberately omits the matrix cell and the memory rails. Those describe
    /// how guest memory reaches the GPU, which is the engine device's job; a
    /// consumer reporting `cell=` would contradict the engine's own line for
    /// the same GPU on any host where the consumer lacks dmabuf (every Mac).
    pub fn consumer_line(&self, device_name: &str) -> String {
        format!(
            "vk_caps role=consumer api={} baseline={} memory={} memory_signal={} {} handoff={} sink={} portability_subset={} type={:?} name={device_name:?}",
            api_floor::version_str(self.device_api_version),
            api_floor::version_str(api_floor::MIN_SUPPORTED_API),
            self.memory.topology.slug(),
            self.memory.signal.slug(),
            self.zero_copy.consumer_summary(),
            self.handoff.chosen().slug(),
            FrameSink::HostOwnedWindow.slug(),
            self.portability_subset,
            self.device_type,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use memory_topology::fixtures;

    fn caps(
        api: u32,
        props: &vk::PhysicalDeviceMemoryProperties,
        mechanisms: DmaMechanisms,
    ) -> HostGpuCaps {
        HostGpuCaps {
            memory: memory_topology::classify_memory(props),
            zero_copy: ZeroCopyProfile::resolve(mechanisms),
            handoff: HandoffLadder::resolve(&frame_interop::HandoffInputs {
                dmabuf_export: mechanisms.dmabuf_share,
                engine_swapchain: false,
            }),
            quirks: DriverQuirk::default(),
            portability_subset: false,
            device_api_version: api,
            device_type: vk::PhysicalDeviceType::DISCRETE_GPU,
        }
    }

    const HOST_PTR_ONLY: DmaMechanisms = DmaMechanisms {
        host_pointer_import: true,
        dmabuf_share: false,
    };
    const BOTH: DmaMechanisms = DmaMechanisms {
        host_pointer_import: true,
        dmabuf_share: true,
    };
    const NEITHER: DmaMechanisms = DmaMechanisms {
        host_pointer_import: false,
        dmabuf_share: false,
    };

    /// The four matrix cells are reachable from real device shapes, and the API
    /// version does NOT participate: the same 1.2 device lands in whichever
    /// cell its topology and mechanisms put it in.
    #[test]
    fn every_matrix_cell_is_reachable_from_a_real_device_shape() {
        let cells = [
            (
                fixtures::apple_m3_max(),
                HOST_PTR_ONLY,
                SupportCell::UnifiedDma,
            ),
            (fixtures::intel_igpu(), NEITHER, SupportCell::UnifiedNoDma),
            (
                fixtures::nvidia_discrete_rebar(),
                BOTH,
                SupportCell::DiscreteDma,
            ),
            (
                fixtures::nvidia_discrete(),
                NEITHER,
                SupportCell::DiscreteNoDma,
            ),
        ];
        for (props, mechanisms, expected) in cells {
            assert_eq!(
                caps(vk::API_VERSION_1_2, &props, mechanisms).cell(),
                expected
            );
        }
    }

    /// The API version is not an axis: the same device at 1.2 and at 1.4 lands
    /// in the same cell. Getting this wrong is how the old tier axis smuggled
    /// "has dmabuf" in under "is 1.3".
    #[test]
    fn the_api_version_does_not_change_the_cell() {
        let props = fixtures::apple_m3_max();
        for api in [
            vk::API_VERSION_1_2,
            vk::API_VERSION_1_3,
            vk::make_api_version(0, 1, 4, 334),
        ] {
            assert_eq!(
                caps(api, &props, HOST_PTR_ONLY).cell(),
                SupportCell::UnifiedDma,
                "api {}",
                api_floor::version_str(api)
            );
        }
    }

    /// The selection line names the cell, both signals, every rail's rung, and
    /// the handoff — what a portability bug report needs.
    #[test]
    fn selection_line_carries_the_diagnosis() {
        let c = caps(vk::API_VERSION_1_3, &fixtures::nvidia_discrete(), BOTH);
        let line = c.selection_line("NVIDIA GeForce RTX 5080");
        assert!(line.contains("cell=discrete_dma"), "{line}");
        assert!(line.contains("memory_signal=separate_host_heap"), "{line}");
        assert!(
            line.contains("dma_mechanisms=host_pointer+dmabuf"),
            "{line}"
        );
        assert!(line.contains("guest_read=imported_pages"), "{line}");
        assert!(line.contains("guest_write=gpu_direct"), "{line}");
        assert!(line.contains("handoff=dmabuf_fd"), "{line}");
        assert!(line.contains("device_local_mb=16384"), "{line}");
        // The baseline is stated on every line so no reader mistakes the
        // device's reported version for a requirement.
        assert!(line.contains("baseline=1.2"), "{line}");
    }

    /// The line always names our own window as the sink — on every row, and
    /// whatever rung was chosen.
    #[test]
    fn selection_line_always_names_the_host_window_sink() {
        for (props, mechanisms) in [
            (fixtures::apple_m3_max(), HOST_PTR_ONLY),
            (fixtures::intel_igpu(), NEITHER),
            (fixtures::nvidia_discrete_rebar(), BOTH),
            (fixtures::nvidia_discrete(), NEITHER),
        ] {
            let line = caps(vk::API_VERSION_1_2, &props, mechanisms).selection_line("dev");
            assert!(line.contains("sink=host_window"), "{line}");
        }
    }

    /// A degraded memory rail is visible in the line rather than silent — the
    /// copy-only host must announce itself.
    #[test]
    fn a_copy_only_host_names_both_degraded_rails() {
        let c = caps(vk::API_VERSION_1_2, &fixtures::nvidia_discrete(), NEITHER);
        let line = c.selection_line("GeForce GTX 750 Ti");
        assert!(line.contains("cell=discrete_no_dma"), "{line}");
        assert!(line.contains("dma=no_dma"), "{line}");
        assert!(line.contains("guest_read=staging_copy"), "{line}");
        assert!(line.contains("guest_write=cpu_readback"), "{line}");
        assert!(
            line.contains(
                "zero_copy_declined=[guest_read:no_host_pointer_import,\
                 guest_write:no_host_pointer_import]"
            ),
            "{line}"
        );
        assert!(!c.imports_guest_pages());
    }

    /// `imports_guest_pages` is the one question call sites ask, and it tracks
    /// the mechanism rather than the driver.
    #[test]
    fn imports_guest_pages_tracks_the_mechanism() {
        let props = fixtures::apple_m3_max();
        assert!(caps(vk::API_VERSION_1_2, &props, HOST_PTR_ONLY).imports_guest_pages());
        assert!(!caps(vk::API_VERSION_1_2, &props, NEITHER).imports_guest_pages());
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
