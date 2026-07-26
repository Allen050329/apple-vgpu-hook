//! How a finished guest frame leaves the engine and reaches **our own window**.
//!
//! This is the display rail of the zero-copy classification (see
//! [`super::zero_copy`] for the two memory rails). It is an **ordered ladder**,
//! not a per-platform special case. Each rung is implemented; the top rung the
//! device can actually do is chosen, and every rung that was skipped records a
//! named reason so a portability bug is one log line away from its cause.
//!
//! | Rung | Cost | Requires |
//! |---|---|---|
//! | [`FrameHandoff::DmaBufFd`] | zero-copy, crosses a device boundary | `VK_KHR_external_memory_fd` + `VK_EXT_external_memory_dma_buf` on both sides |
//! | [`FrameHandoff::EngineSwapchain`] | zero-copy, no boundary to cross | `VK_KHR_swapchain` + a platform surface on the engine's queue family |
//! | [`FrameHandoff::HostVisibleCopy`] | one GPU→host copy | nothing beyond a host-visible memory type |
//!
//! # The destination is always the host-owned window
//!
//! Every rung ends at the Rust-owned winit + `VkSurfaceKHR` window. QEMU runs
//! `-display none`, owns no window, and receives no per-present paint; its
//! console is not a frame sink on any row of the support matrix. The rungs
//! differ only in *how* the pixels get to our window, never in where they land.
//!
//! # Why dmabuf ranks above the engine swapchain
//!
//! Both are zero-copy, and in practice they are never both available, so the
//! order is documentation rather than a live tiebreak. They answer different
//! questions:
//!
//! * On macOS the engine's `VkDevice` owns the window surface, so the frame
//!   never crosses a device boundary and needs no external-memory handle at all.
//!   MoltenVK has no dmabuf, and none is wanted.
//! * On Linux the window creates its **own** `VkDevice` — possibly a different
//!   physical GPU on a hybrid host — so the frame must cross a boundary, and
//!   dmabuf is what carries it there without a readback. The engine device does
//!   not create a swapchain on that pathway.
//!
//! # The bottom rung is not optional
//!
//! [`FrameHandoff::HostVisibleCopy`] needs only a `HOST_VISIBLE` memory type,
//! which every Vulkan implementation is required to expose. It is therefore
//! **always** reachable, and [`HandoffLadder::resolve`] can never return "no
//! way to present". That is the property that makes the non-DMA rows of the
//! matrix first-class: a device with neither dmabuf nor a swapchain still shows
//! the guest desktop in our window, slower, rather than dropping frames.

use crate::observe::Decline;

/// One rung of the frame-handoff ladder, best first.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FrameHandoff {
    /// Export the present image's memory as a dmabuf fd and import it into the
    /// host window's own `VkDevice` — which on a hybrid Linux host may be a
    /// different physical GPU than the engine's. The pixels never cross that
    /// boundary as bytes.
    DmaBufFd,
    /// Blit the resident straight into a swapchain image owned by the engine's
    /// own `VkDevice`, presenting to the host window's `VkSurfaceKHR`. Same
    /// device and queue as the render work, so there is no boundary to cross
    /// and no external-memory handle is needed. This is the arm64/MoltenVK
    /// pathway.
    EngineSwapchain,
    /// Copy the present image into host-visible memory, which the host window
    /// uploads into its staging image. One GPU copy plus a CPU blit. Always
    /// available.
    HostVisibleCopy,
}

impl FrameHandoff {
    /// Stable slug for logs and proxy lines.
    pub fn slug(self) -> &'static str {
        match self {
            Self::DmaBufFd => "dmabuf_fd",
            Self::EngineSwapchain => "engine_swapchain",
            Self::HostVisibleCopy => "host_visible_copy",
        }
    }

    /// True when the frame's pixels never round-trip through host memory.
    pub fn is_zero_copy(self) -> bool {
        matches!(self, Self::DmaBufFd | Self::EngineSwapchain)
    }

    /// The ladder, best rung first. The last entry is always the rung that
    /// needs no optional capability.
    pub const LADDER: [FrameHandoff; 3] =
        [Self::DmaBufFd, Self::EngineSwapchain, Self::HostVisibleCopy];
}

/// Why a rung was not taken. Every skipped rung carries one of these so a
/// "why is my present slow" question is answerable from `/tmp/reims-vgpu-fail.log`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HandoffDecline {
    /// `VK_KHR_external_memory_fd` and/or `VK_EXT_external_memory_dma_buf` are
    /// not advertised by the device (MoltenVK, and Vulkan 1.2-era drivers
    /// without the dmabuf pair).
    NoDmabufExtensions,
    /// `VK_KHR_swapchain` or a usable platform surface is unavailable, or the
    /// engine's queue family cannot present to it.
    NoEngineSwapchain,
}

impl crate::observe::Decline for HandoffDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::NoDmabufExtensions => "no_dmabuf_extensions",
            Self::NoEngineSwapchain => "no_engine_swapchain",
        }
    }
}

/// The capability answers the ladder needs. Kept as a plain struct of booleans
/// so [`HandoffLadder::resolve`] is pure and every row of the support matrix is
/// testable without a GPU.
///
/// `VK_EXT_external_memory_host` deliberately does **not** appear here: it is a
/// memory-rail capability, not a display one, and it lives in
/// [`super::zero_copy::DmaMechanisms`]. Carrying it in both places is how the
/// two answers drift apart.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct HandoffInputs {
    /// Both dmabuf extensions present AND enabled at device create.
    pub dmabuf_export: bool,
    /// `VK_KHR_swapchain` enabled on the engine device and a surface exists.
    pub engine_swapchain: bool,
}

/// The resolved ladder: which rung was chosen and why each better rung was not.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HandoffLadder {
    chosen: FrameHandoff,
    declined: Vec<(FrameHandoff, HandoffDecline)>,
    inputs: HandoffInputs,
}

impl HandoffLadder {
    /// Walk the ladder top-down and take the first rung the device supports,
    /// recording a named decline for each rung skipped.
    pub fn resolve(inputs: &HandoffInputs) -> Self {
        let mut declined = Vec::new();
        for rung in FrameHandoff::LADDER {
            let decline = match rung {
                FrameHandoff::DmaBufFd if !inputs.dmabuf_export => {
                    Some(HandoffDecline::NoDmabufExtensions)
                }
                FrameHandoff::EngineSwapchain if !inputs.engine_swapchain => {
                    Some(HandoffDecline::NoEngineSwapchain)
                }
                _ => None,
            };
            match decline {
                Some(reason) => declined.push((rung, reason)),
                None => {
                    return Self {
                        chosen: rung,
                        declined,
                        inputs: *inputs,
                    }
                }
            }
        }
        // Unreachable: HostVisibleCopy has no capability guard. Kept total
        // rather than panicking so a future rung insertion cannot crash boot.
        Self {
            chosen: FrameHandoff::HostVisibleCopy,
            declined,
            inputs: *inputs,
        }
    }

    pub fn chosen(&self) -> FrameHandoff {
        self.chosen
    }

    pub fn inputs(&self) -> &HandoffInputs {
        &self.inputs
    }

    /// True when this rung supports the P1 zero-copy present invariant.
    pub fn is_zero_copy(&self) -> bool {
        self.chosen.is_zero_copy()
    }

    /// Whether a given rung is available, regardless of which one was chosen —
    /// a caller may want dmabuf for the console while presenting to its own
    /// swapchain.
    pub fn supports(&self, rung: FrameHandoff) -> bool {
        match rung {
            FrameHandoff::DmaBufFd => self.inputs.dmabuf_export,
            FrameHandoff::EngineSwapchain => self.inputs.engine_swapchain,
            FrameHandoff::HostVisibleCopy => true,
        }
    }

    /// Every rung that was skipped, with the decline that explains it.
    /// Exposed for the same reason as `ZeroCopyProfile::declined`.
    pub fn declined(&self) -> &[(FrameHandoff, HandoffDecline)] {
        &self.declined
    }

    /// `rung:reason` pairs for the one-shot selection log line.
    pub fn declined_summary(&self) -> String {
        self.declined
            .iter()
            .map(|(rung, reason)| format!("{}:{}", rung.slug(), reason.slug()))
            .collect::<Vec<_>>()
            .join(",")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A Linux discrete host with the dmabuf pair takes the top rung.
    #[test]
    fn dmabuf_wins_when_available() {
        let ladder = HandoffLadder::resolve(&HandoffInputs {
            dmabuf_export: true,
            engine_swapchain: true,
        });
        assert_eq!(ladder.chosen(), FrameHandoff::DmaBufFd);
        assert!(ladder.is_zero_copy());
        assert_eq!(ladder.declined_summary(), "");
    }

    /// MoltenVK: no dmabuf, but the engine device owns the window surface —
    /// still zero-copy, and the skipped rung names why. "No dmabuf" is not a
    /// degradation on this pathway; there is no device boundary to cross.
    #[test]
    fn moltenvk_falls_to_the_engine_swapchain_and_stays_zero_copy() {
        let ladder = HandoffLadder::resolve(&HandoffInputs {
            dmabuf_export: false,
            engine_swapchain: true,
        });
        assert_eq!(ladder.chosen(), FrameHandoff::EngineSwapchain);
        assert!(ladder.is_zero_copy());
        assert_eq!(ladder.declined_summary(), "dmabuf_fd:no_dmabuf_extensions");
    }

    /// THE load-bearing property: with every optional capability absent, the
    /// ladder still resolves. No device can end up unable to present.
    #[test]
    fn bottom_rung_is_always_reachable() {
        let ladder = HandoffLadder::resolve(&HandoffInputs::default());
        assert_eq!(ladder.chosen(), FrameHandoff::HostVisibleCopy);
        assert!(!ladder.is_zero_copy());
        assert_eq!(
            ladder.declined_summary(),
            "dmabuf_fd:no_dmabuf_extensions,engine_swapchain:no_engine_swapchain"
        );
    }

    /// Exhaustive: every combination of the capability inputs resolves to a
    /// rung the device actually supports.
    #[test]
    fn every_capability_combination_resolves_to_a_supported_rung() {
        for bits in 0u8..4 {
            let inputs = HandoffInputs {
                dmabuf_export: bits & 1 != 0,
                engine_swapchain: bits & 2 != 0,
            };
            let ladder = HandoffLadder::resolve(&inputs);
            assert!(
                ladder.supports(ladder.chosen()),
                "{inputs:?} chose an unsupported rung"
            );
        }
    }

    /// A caller may need to know a rung is available without it being the one
    /// chosen — the export ring is prepared on capability, not on selection.
    #[test]
    fn supports_is_independent_of_the_chosen_rung() {
        let ladder = HandoffLadder::resolve(&HandoffInputs {
            dmabuf_export: true,
            engine_swapchain: true,
        });
        assert_eq!(ladder.chosen(), FrameHandoff::DmaBufFd);
        assert!(ladder.supports(FrameHandoff::EngineSwapchain));
        assert!(ladder.supports(FrameHandoff::HostVisibleCopy));
    }

    /// The ladder is ordered best-first and ends on the unconditional rung.
    #[test]
    fn ladder_order_is_best_first_and_ends_unconditional() {
        assert_eq!(FrameHandoff::LADDER[0], FrameHandoff::DmaBufFd);
        assert_eq!(
            *FrameHandoff::LADDER.last().unwrap(),
            FrameHandoff::HostVisibleCopy
        );
        assert!(FrameHandoff::LADDER[0].is_zero_copy());
        assert!(!FrameHandoff::HostVisibleCopy.is_zero_copy());
    }
}
