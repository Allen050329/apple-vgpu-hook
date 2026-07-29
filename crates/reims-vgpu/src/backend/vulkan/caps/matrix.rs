//! The four first-class host GPU configurations, as data.
//!
//! This module exists so the support matrix cannot quietly shrink. Every cell is
//! enumerated in [`MATRIX`]; the tests below prove the cross product is
//! complete, that each cell resolves a working memory policy and a working frame
//! handoff, and that no cell depends on a capability another cell lacks. A
//! change that drops a configuration fails here rather than on someone's
//! machine.
//!
//! # The two axes
//!
//! |  | **DMA** (some crossing is zero-copy) | **non-DMA** (every crossing copies) |
//! |---|---|---|
//! | **Unified memory** | Apple M-series via MoltenVK; Intel/AMD iGPU on Mesa | an iGPU whose driver exposes no sharing mechanism |
//! | **Discrete memory** | the RTX 5080 dev host | a GTX 750 Ti-class dGPU on a bare 1.2 driver |
//!
//! * [`MemoryTopology`] — does the GPU read the same DRAM the CPU does? Selects
//!   an allocation *preference*, never a different observable result.
//! * [`DmaSupport`] — can the GPU address memory it did not allocate? Resolved
//!   per rail by [`super::zero_copy`], which names every rail that degraded.
//!
//! # Vulkan 1.2 is the baseline for all four cells, not an axis
//!
//! The API version used to be an axis here. It described the hosts this project
//! owns rather than anything the code does — nothing in the engine uses a 1.3
//! core feature — and it wrongly implied dmabuf came with 1.3. Zero-copy
//! capability is what actually varies, and it is orthogonal to the API version:
//! a 1.2 driver can advertise `VK_EXT_external_memory_host` and a 1.3 driver can
//! lack it. See [`super::api_floor`].
//!
//! # Every cell presents to our own window
//!
//! [`FrameSink`] has exactly one variant on purpose. QEMU's console is not a
//! frame sink on any row: the product display path is the host-owned winit +
//! `VkSurfaceKHR` window on every host, and the handoff ladder only chooses
//! *how* the frame reaches that window.

use super::frame_interop::FrameHandoff;
use super::memory_topology::MemoryTopology;
use super::zero_copy::DmaSupport;

/// One cell of the support matrix: a memory topology crossed with a zero-copy
/// answer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SupportCell {
    /// Unified memory, some rail is zero-copy. An Intel/AMD iGPU on Mesa, via
    /// dmabuf on the display rail. Apple M-series through MoltenVK used to land
    /// here on the host-pointer import and now classifies [`Self::UnifiedNoDma`],
    /// because it has no dmabuf and the import is not requested.
    UnifiedDma,
    /// Unified memory, every crossing copies. Cheap copies — same DRAM, no bus
    /// transfer — but copies.
    UnifiedNoDma,
    /// Discrete memory, some rail is zero-copy. The RTX 5080 dev host.
    DiscreteDma,
    /// Discrete memory, every crossing copies. The slowest supported host, and
    /// the one whose fallbacks must actually work.
    DiscreteNoDma,
}

impl SupportCell {
    /// The cell a classified device lands in. Total over the cross product.
    pub fn of(memory: MemoryTopology, dma: DmaSupport) -> Self {
        match (memory, dma) {
            (MemoryTopology::Unified, DmaSupport::Dma) => Self::UnifiedDma,
            (MemoryTopology::Unified, DmaSupport::NoDma) => Self::UnifiedNoDma,
            (MemoryTopology::Discrete, DmaSupport::Dma) => Self::DiscreteDma,
            (MemoryTopology::Discrete, DmaSupport::NoDma) => Self::DiscreteNoDma,
        }
    }

    pub fn memory(self) -> MemoryTopology {
        match self {
            Self::UnifiedDma | Self::UnifiedNoDma => MemoryTopology::Unified,
            Self::DiscreteDma | Self::DiscreteNoDma => MemoryTopology::Discrete,
        }
    }

    pub fn dma(self) -> DmaSupport {
        match self {
            Self::UnifiedDma | Self::DiscreteDma => DmaSupport::Dma,
            Self::UnifiedNoDma | Self::DiscreteNoDma => DmaSupport::NoDma,
        }
    }

    /// Stable slug for logs, proxy lines, and test names.
    pub fn slug(self) -> &'static str {
        match self {
            Self::UnifiedDma => "unified_dma",
            Self::UnifiedNoDma => "unified_no_dma",
            Self::DiscreteDma => "discrete_dma",
            Self::DiscreteNoDma => "discrete_no_dma",
        }
    }

    /// All four cells. Iterating this is how a caller proves it handled the
    /// whole matrix rather than the row it happened to be running on.
    pub const ALL: [SupportCell; 4] = [
        Self::UnifiedDma,
        Self::UnifiedNoDma,
        Self::DiscreteDma,
        Self::DiscreteNoDma,
    ];
}

/// Where a finished frame is displayed.
///
/// One variant, deliberately. The host-owned window is the only supported frame
/// sink on every row of the matrix — QEMU runs `-display none` and owns no
/// window, and no per-present paint crosses into QEMU's address space. A second
/// variant appearing here would mean re-litigating that decision, not adding a
/// configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FrameSink {
    /// The Rust-owned winit window with its own `VkSurfaceKHR`.
    HostOwnedWindow,
}

impl FrameSink {
    pub fn slug(self) -> &'static str {
        match self {
            Self::HostOwnedWindow => "host_window",
        }
    }
}

/// Documentation-grade description of one matrix cell. Carried in the binary so
/// the tests can assert against it.
#[derive(Clone, Copy, Debug)]
pub struct MatrixCell {
    pub cell: SupportCell,
    /// Hosts known to land in this cell.
    pub representative_hosts: &'static str,
    /// What zero-copy does or does not buy on this row, for the next reader.
    pub zero_copy_note: &'static str,
    /// The handoff this cell reaches with every optional capability absent.
    /// Always the unconditional rung — that is what makes the row real rather
    /// than aspirational.
    pub guaranteed_handoff: FrameHandoff,
    /// Where the frame is displayed. The same on every row.
    pub frame_sink: FrameSink,
}

/// The support matrix. Adding a host configuration means adding a row here and
/// making the tests below pass — not adding a `cfg` to a call site.
pub const MATRIX: [MatrixCell; 4] = [
    MatrixCell {
        cell: SupportCell::UnifiedDma,
        representative_hosts:
            "Apple M-series via MoltenVK (host-pointer import, no dmabuf); Intel/AMD iGPU on Mesa",
        zero_copy_note:
            "the GPU samples and writes guest pages in place; the frame reaches the window \
             without an external-memory handle on macOS (the engine device owns the surface) \
             and through dmabuf on Mesa",
        guaranteed_handoff: FrameHandoff::HostVisibleCopy,
        frame_sink: FrameSink::HostOwnedWindow,
    },
    MatrixCell {
        cell: SupportCell::UnifiedNoDma,
        representative_hosts:
            "an iGPU whose driver advertises neither VK_EXT_external_memory_host nor the \
             dmabuf pair",
        zero_copy_note:
            "every crossing copies, but the copies are cheap: CPU and GPU share the same DRAM, \
             so nothing traverses a bus",
        guaranteed_handoff: FrameHandoff::HostVisibleCopy,
        frame_sink: FrameSink::HostOwnedWindow,
    },
    MatrixCell {
        cell: SupportCell::DiscreteDma,
        representative_hosts: "the RTX 5080 dev host; any current discrete Linux driver",
        zero_copy_note:
            "host-pointer import lets the GPU DMA guest pages over PCIe, and dmabuf carries the \
             frame to the window's device without a readback",
        guaranteed_handoff: FrameHandoff::HostVisibleCopy,
        frame_sink: FrameSink::HostOwnedWindow,
    },
    MatrixCell {
        cell: SupportCell::DiscreteNoDma,
        representative_hosts:
            "a GTX 750 Ti-class dGPU on a bare Vulkan 1.2 driver with neither extension",
        zero_copy_note:
            "the slowest supported host: every crossing copies AND every copy traverses PCIe. \
             This is the row whose fallbacks must actually work, not just compile",
        guaranteed_handoff: FrameHandoff::HostVisibleCopy,
        frame_sink: FrameSink::HostOwnedWindow,
    },
];

/// Look up a cell. Total over [`SupportCell`] by construction (asserted below).
pub fn cell(cell: SupportCell) -> &'static MatrixCell {
    MATRIX
        .iter()
        .find(|c| c.cell == cell)
        .expect("MATRIX covers every SupportCell — asserted by matrix_covers_every_cell")
}

#[cfg(test)]
mod tests {
    use super::super::frame_interop::{HandoffInputs, HandoffLadder};
    use super::super::memory_topology::{
        classify_memory, fixtures, select_memory_type, MemoryClass,
    };
    use super::super::zero_copy::{DmaMechanisms, ZeroCopyProfile};
    use super::*;

    /// Every cell paired with a representative device: its memory layout and
    /// the sharing mechanisms that host advertises.
    fn representative_devices() -> [(
        SupportCell,
        ash::vk::PhysicalDeviceMemoryProperties,
        DmaMechanisms,
    ); 4] {
        [
            (
                // A Mesa iGPU: unified memory plus dmabuf. Since the
                // host-pointer import was removed, dmabuf is the only mechanism
                // left that reaches a zero-copy rung, so this is what a unified
                // `Dma` host looks like.
                SupportCell::UnifiedDma,
                fixtures::intel_igpu(),
                DmaMechanisms { dmabuf_share: true },
            ),
            (
                // Apple/MoltenVK: unified memory, no dmabuf, and no import —
                // every crossing is a copy. Fully supported, just slower.
                SupportCell::UnifiedNoDma,
                fixtures::apple_m3_max(),
                DmaMechanisms::default(),
            ),
            (
                SupportCell::DiscreteDma,
                fixtures::nvidia_discrete_rebar(),
                DmaMechanisms { dmabuf_share: true },
            ),
            (
                SupportCell::DiscreteNoDma,
                fixtures::nvidia_discrete(),
                DmaMechanisms::default(),
            ),
        ]
    }

    /// The matrix is the full cross product: four cells, each distinct.
    #[test]
    fn matrix_covers_every_cell() {
        assert_eq!(MATRIX.len(), SupportCell::ALL.len());
        for c in SupportCell::ALL {
            assert_eq!(cell(c).cell, c);
        }
        let mut slugs: Vec<_> = SupportCell::ALL.iter().map(|c| c.slug()).collect();
        slugs.sort_unstable();
        slugs.dedup();
        assert_eq!(slugs.len(), 4, "cell slugs must be unique");
    }

    /// `SupportCell::of` is a bijection onto the cross product — no pair of
    /// (topology, dma) collapses onto the same cell, and every cell is
    /// reachable.
    #[test]
    fn cell_lookup_is_a_bijection_over_the_cross_product() {
        let mut seen = Vec::new();
        for memory in [MemoryTopology::Unified, MemoryTopology::Discrete] {
            for dma in DmaSupport::ALL {
                let c = SupportCell::of(memory, dma);
                assert_eq!(c.memory(), memory);
                assert_eq!(c.dma(), dma);
                seen.push(c);
            }
        }
        seen.sort_unstable();
        assert_eq!(seen, SupportCell::ALL.to_vec());
    }

    /// EVERY cell must keep presenting with every optional capability absent.
    /// This is the guarantee that makes the non-DMA rows real: strip dmabuf and
    /// the swapchain and the frame still reaches the window.
    #[test]
    fn every_cell_presents_with_no_optional_capability() {
        let bare = HandoffLadder::resolve(&HandoffInputs::default());
        for c in MATRIX {
            assert_eq!(
                bare.chosen(),
                c.guaranteed_handoff,
                "{} must fall back to its guaranteed rung",
                c.cell.slug()
            );
            assert!(bare.supports(c.guaranteed_handoff));
        }
    }

    /// The frame sink is the host-owned window on every row. QEMU's console is
    /// not a supported sink anywhere in the matrix, so a cell that named a
    /// different sink would be documenting a display path the project does not
    /// have.
    #[test]
    fn every_cell_presents_to_the_host_owned_window() {
        for c in MATRIX {
            assert_eq!(
                c.frame_sink,
                FrameSink::HostOwnedWindow,
                "{} must present to our own window",
                c.cell.slug()
            );
        }
    }

    /// A representative device from each cell classifies into that cell — on
    /// BOTH axes. This is the end-to-end check that a row is not just a label.
    #[test]
    fn every_representative_device_classifies_into_its_own_cell() {
        for (expected, props, mechanisms) in representative_devices() {
            let topology = classify_memory(&props).topology;
            let dma = ZeroCopyProfile::resolve(mechanisms).support();
            assert_eq!(
                SupportCell::of(topology, dma),
                expected,
                "{} representative device must classify into its own row",
                expected.slug()
            );
        }
    }

    /// Each cell's memory policy resolves against a real device layout from that
    /// cell: take its topology, ask for every memory class, and get an answer.
    #[test]
    fn every_cell_resolves_every_memory_class_on_a_representative_device() {
        for (c, props, _) in representative_devices() {
            let profile = classify_memory(&props);
            for class in [
                MemoryClass::Upload,
                MemoryClass::Readback,
                MemoryClass::DeviceLocal,
                MemoryClass::DeviceLocalPreferred,
            ] {
                let req = profile.topology.request(class);
                assert!(
                    select_memory_type(&props, !0, &req).is_some(),
                    "{}/{class:?} must resolve a memory type",
                    c.slug()
                );
            }
        }
    }

    /// The DMA column is exactly the cells whose representative device has a
    /// sharing mechanism, and the non-DMA column exactly those with none. This
    /// locks the axis to the capability rather than to a vendor or an API
    /// version — an RTX 5080 and an M3 Max share a column despite sharing
    /// neither driver, memory topology, nor dmabuf support.
    #[test]
    fn the_dma_column_tracks_mechanisms_not_vendors() {
        for (c, _, mechanisms) in representative_devices() {
            assert_eq!(
                c.dma(),
                if mechanisms.any() {
                    DmaSupport::Dma
                } else {
                    DmaSupport::NoDma
                },
                "{} column must follow its mechanisms",
                c.slug()
            );
        }
    }

    /// A cell is described for humans too — an empty host list or an empty
    /// zero-copy note means the row exists only on paper.
    #[test]
    fn every_cell_names_its_hosts_and_its_zero_copy_story() {
        for c in MATRIX {
            assert!(
                !c.representative_hosts.is_empty(),
                "{} must name a real host",
                c.cell.slug()
            );
            assert!(
                !c.zero_copy_note.is_empty(),
                "{} must say what zero-copy does or does not buy",
                c.cell.slug()
            );
        }
    }
}
