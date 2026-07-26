//! Zero-copy classification — how much of the guest → GPU → display path can
//! avoid a CPU copy on this host.
//!
//! # Three independent rails, each its own ladder
//!
//! "Zero-copy" is not one capability. The guest command stream crosses the host
//! boundary three times, and each crossing has its own feature gate and its own
//! honest fallback:
//!
//! | Rail | Zero-copy rung | Fallback rung |
//! |---|---|---|
//! | [`GuestRead`] — guest textures/buffers into the GPU | GPU samples the guest's own pages | staging copy into a device buffer |
//! | [`GuestWrite`] — results back into guest memory | GPU writes the guest's own pages | CPU readback into those pages |
//! | [`super::FrameHandoff`] — finished frame to the display | same-device swapchain, or dmabuf to the window's device | host-visible copy into the window's staging image |
//!
//! The rails are independent: MoltenVK reaches the top rung on all three
//! without dmabuf existing at all (it imports host pointers and the engine
//! device owns the window surface), while a Linux host with dmabuf but no
//! `VK_EXT_external_memory_host` reaches the top rung on the display rail and
//! the bottom rung on both memory rails. Collapsing them into one flag would
//! report either host as "half zero-copy" and name neither half.
//!
//! # Why the axis is still binary in the matrix
//!
//! [`DmaSupport`] answers the coarse question the support matrix asks — does
//! this host reach *any* zero-copy rung, or does every crossing cost a copy? A
//! `NoDma` host is fully supported and fully tested; it is just slower, and the
//! log line names which rung each rail landed on so "why is this host slow" is
//! answerable without a debugger.
//!
//! # The pinning constraint (standing directive, 2026-07-18)
//!
//! `VK_EXT_external_memory_host` *registers* — pins — every page it imports for
//! the lifetime of the import. Importing the QEMU RAMBlock's whole VMA would
//! therefore pin all guest RAM on the host, which is why imports are bucketed
//! into `HOST_IMPORT_WINDOW_CAP`-sized windows rather than taken whole. The
//! top rung on the memory rails means "the GPU addresses guest pages in place",
//! never "all of guest RAM is one GPU allocation".

use crate::observe::Decline;

/// The mechanisms by which this device can address memory it did not itself
/// allocate. Detected once at device create; never re-queried.
///
/// These are *different boundaries*, not alternatives — see the module docs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DmaMechanisms {
    /// `VK_EXT_external_memory_host`: import a host virtual address range (the
    /// pages QEMU allocated for guest RAM) as `VkDeviceMemory`, so the GPU
    /// reads and writes guest memory in place.
    ///
    /// Present on MoltenVK, Mesa, and NVIDIA. This is the mechanism both memory
    /// rails ride on.
    pub host_pointer_import: bool,
    /// `VK_KHR_external_memory_fd` + `VK_EXT_external_memory_dma_buf`: export a
    /// GPU allocation as a kernel dma-buf fd another device or process imports.
    ///
    /// Linux only. It matters here because the host window creates its **own**
    /// `VkDevice`, which on a hybrid host may be a different physical GPU than
    /// the engine's — dmabuf is what lets the frame cross that boundary without
    /// a copy. macOS needs no equivalent: the engine device owns the window
    /// surface directly, so the frame never crosses a device boundary.
    pub dmabuf_share: bool,
}

impl DmaMechanisms {
    /// Stable slug naming which mechanisms are present, for the selection line.
    pub fn slug(self) -> &'static str {
        match (self.host_pointer_import, self.dmabuf_share) {
            (true, true) => "host_pointer+dmabuf",
            (true, false) => "host_pointer",
            (false, true) => "dmabuf",
            (false, false) => "none",
        }
    }

    /// True when at least one mechanism exists, i.e. some crossing can avoid a
    /// copy. This is the matrix axis.
    pub fn any(self) -> bool {
        self.host_pointer_import || self.dmabuf_share
    }
}

/// One axis of the support matrix: whether this host can avoid a CPU copy
/// anywhere on the guest → GPU → display path.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DmaSupport {
    /// At least one rail reaches its zero-copy rung.
    Dma,
    /// Every crossing costs a copy. Fully supported, just slower — every rail's
    /// fallback rung is implemented and tested.
    NoDma,
}

impl DmaSupport {
    pub fn slug(self) -> &'static str {
        match self {
            Self::Dma => "dma",
            Self::NoDma => "no_dma",
        }
    }

    /// Both columns of the matrix. Iterating this is how a caller proves it
    /// handled the copy-only host too.
    pub const ALL: [DmaSupport; 2] = [Self::Dma, Self::NoDma];
}

/// How guest memory reaches the GPU for reads (sampling a guest texture,
/// binding a guest buffer).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GuestRead {
    /// The guest's own pages are imported and read in place. No copy.
    ImportedPages,
    /// The guest's bytes are copied into a device buffer before use. Always
    /// available: it needs no optional capability.
    StagingCopy,
}

impl GuestRead {
    pub fn slug(self) -> &'static str {
        match self {
            Self::ImportedPages => "imported_pages",
            Self::StagingCopy => "staging_copy",
        }
    }

    pub fn is_zero_copy(self) -> bool {
        matches!(self, Self::ImportedPages)
    }
}

/// How GPU results reach guest memory (a compute store, a present write-back).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GuestWrite {
    /// The GPU writes the guest's imported pages directly — the GPU copy IS the
    /// write-back. No CPU readback.
    GpuDirect,
    /// The GPU writes host-visible memory the CPU then copies into guest pages.
    /// Always available.
    CpuReadback,
}

impl GuestWrite {
    pub fn slug(self) -> &'static str {
        match self {
            Self::GpuDirect => "gpu_direct",
            Self::CpuReadback => "cpu_readback",
        }
    }

    pub fn is_zero_copy(self) -> bool {
        matches!(self, Self::GpuDirect)
    }
}

/// Why a rail did not reach its zero-copy rung. Every degrade is named so a
/// "why is this host slow" question is one `/tmp/reims-vgpu-fail.log` line from its
/// cause, per the never-drop-silently rule.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ZeroCopyDecline {
    /// `VK_EXT_external_memory_host` is not advertised, so no host pointer can
    /// become `VkDeviceMemory` and both memory rails fall to a copy.
    NoHostPointerImport,
}

impl crate::observe::Decline for ZeroCopyDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::NoHostPointerImport => "no_host_pointer_import",
        }
    }
}

/// The resolved zero-copy classification for one device.
///
/// Built once at device create from [`DmaMechanisms`]; every call site that
/// used to ask the device context "is `VK_EXT_external_memory_host` enabled?"
/// asks this instead, so the answer cannot drift between rails.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ZeroCopyProfile {
    pub mechanisms: DmaMechanisms,
    pub guest_read: GuestRead,
    pub guest_write: GuestWrite,
    declined: Vec<(&'static str, ZeroCopyDecline)>,
}

impl ZeroCopyProfile {
    /// Resolve every rail against the mechanisms this device advertises.
    ///
    /// Both memory rails currently ride on the same mechanism, so they decline
    /// together. They are kept separate anyway because they fail separately in
    /// practice — a read import can succeed on a span a write import rejects
    /// (alignment, window cap) — and because a future device could expose one
    /// direction only.
    pub fn resolve(mechanisms: DmaMechanisms) -> Self {
        let mut declined = Vec::new();
        let (guest_read, guest_write) = if mechanisms.host_pointer_import {
            (GuestRead::ImportedPages, GuestWrite::GpuDirect)
        } else {
            declined.push(("guest_read", ZeroCopyDecline::NoHostPointerImport));
            declined.push(("guest_write", ZeroCopyDecline::NoHostPointerImport));
            (GuestRead::StagingCopy, GuestWrite::CpuReadback)
        };
        Self {
            mechanisms,
            guest_read,
            guest_write,
            declined,
        }
    }

    /// Classification for a device that only ever **consumes** finished frames
    /// — the host window's `VkDevice`. It never touches guest memory, so both
    /// memory rails are reported at their fallback rung with **no** decline
    /// recorded: naming a decline here would cry wolf about a rail this device
    /// is never asked to run.
    ///
    /// Such a device is not a matrix cell. The matrix classifies the engine
    /// device, which is the one that decides how guest memory reaches the GPU;
    /// report a consumer with [`ZeroCopyProfile::consumer_summary`] rather than
    /// through the full selection line.
    pub fn display_only(dmabuf_import: bool) -> Self {
        Self {
            mechanisms: DmaMechanisms {
                host_pointer_import: false,
                dmabuf_share: dmabuf_import,
            },
            guest_read: GuestRead::StagingCopy,
            guest_write: GuestWrite::CpuReadback,
            declined: Vec::new(),
        }
    }

    /// The fields worth logging for a frame-consuming device: whether it can
    /// import a dmabuf, and nothing that would contradict the engine device's
    /// own classification of the same physical GPU.
    pub fn consumer_summary(&self) -> String {
        format!("dmabuf_import={}", self.mechanisms.dmabuf_share)
    }

    /// The matrix axis: does any rail avoid a copy?
    pub fn support(&self) -> DmaSupport {
        if self.mechanisms.any() {
            DmaSupport::Dma
        } else {
            DmaSupport::NoDma
        }
    }

    /// Every rail that degraded, with the decline that explains it.
    ///
    /// Exposed so the bring-up site can give each one its own `reason=<slug>`
    /// line. The summary below reads well but is one field inside a
    /// twenty-field line, which is not something a `grep reason=` finds.
    pub fn declined(&self) -> &[(&'static str, ZeroCopyDecline)] {
        &self.declined
    }

    /// `rail:reason` pairs for the one-shot selection log line.
    pub fn declined_summary(&self) -> String {
        self.declined
            .iter()
            .map(|(rail, reason)| format!("{rail}:{}", reason.slug()))
            .collect::<Vec<_>>()
            .join(",")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// MoltenVK: host-pointer import, no dmabuf. Both memory rails reach their
    /// zero-copy rung, and the host lands in the DMA column — the display rail
    /// needs no external-memory handle there because the engine device owns the
    /// window surface.
    #[test]
    fn moltenvk_is_dma_through_host_pointer_import_alone() {
        let profile = ZeroCopyProfile::resolve(DmaMechanisms {
            host_pointer_import: true,
            dmabuf_share: false,
        });
        assert_eq!(profile.support(), DmaSupport::Dma);
        assert_eq!(profile.guest_read, GuestRead::ImportedPages);
        assert_eq!(profile.guest_write, GuestWrite::GpuDirect);
        assert_eq!(profile.declined_summary(), "");
        assert_eq!(profile.mechanisms.slug(), "host_pointer");
    }

    /// A Linux host with dmabuf but no host-pointer import is still DMA — the
    /// display rail is zero-copy — but BOTH memory rails must name why they
    /// degraded. Reporting a bare "dma" here is the failure this test locks out.
    #[test]
    fn dmabuf_without_host_pointer_import_names_both_degraded_rails() {
        let profile = ZeroCopyProfile::resolve(DmaMechanisms {
            host_pointer_import: false,
            dmabuf_share: true,
        });
        assert_eq!(profile.support(), DmaSupport::Dma);
        assert_eq!(profile.guest_read, GuestRead::StagingCopy);
        assert_eq!(profile.guest_write, GuestWrite::CpuReadback);
        assert_eq!(
            profile.declined_summary(),
            "guest_read:no_host_pointer_import,guest_write:no_host_pointer_import"
        );
        assert_eq!(profile.mechanisms.slug(), "dmabuf");
    }

    /// The copy-only host: every rail on its fallback rung, and the matrix puts
    /// it in the non-DMA column. This row is supported, not declined.
    #[test]
    fn no_mechanism_resolves_every_rail_to_its_fallback() {
        let profile = ZeroCopyProfile::resolve(DmaMechanisms::default());
        assert_eq!(profile.support(), DmaSupport::NoDma);
        assert!(!profile.guest_read.is_zero_copy());
        assert!(!profile.guest_write.is_zero_copy());
        assert_eq!(profile.mechanisms.slug(), "none");
    }

    /// Exhaustive over the mechanism cross product: the axis is exactly "any
    /// mechanism", and every combination resolves both rails to a real rung.
    #[test]
    fn every_mechanism_combination_resolves_and_agrees_with_the_axis() {
        for bits in 0u8..4 {
            let mechanisms = DmaMechanisms {
                host_pointer_import: bits & 1 != 0,
                dmabuf_share: bits & 2 != 0,
            };
            let profile = ZeroCopyProfile::resolve(mechanisms);
            let expected = if mechanisms.any() {
                DmaSupport::Dma
            } else {
                DmaSupport::NoDma
            };
            assert_eq!(profile.support(), expected, "{mechanisms:?}");
            // A rail is zero-copy exactly when its mechanism is present.
            assert_eq!(
                profile.guest_read.is_zero_copy(),
                mechanisms.host_pointer_import,
                "{mechanisms:?}"
            );
            assert_eq!(
                profile.guest_write.is_zero_copy(),
                mechanisms.host_pointer_import,
                "{mechanisms:?}"
            );
        }
    }

    /// Every mechanism combination has a distinct slug, so the selection line
    /// distinguishes "no dmabuf" from "no zero-copy at all" — two very
    /// different hosts that a shared slug would render identical.
    #[test]
    fn mechanism_slugs_are_distinct() {
        let mut slugs: Vec<_> = (0u8..4)
            .map(|bits| {
                DmaMechanisms {
                    host_pointer_import: bits & 1 != 0,
                    dmabuf_share: bits & 2 != 0,
                }
                .slug()
            })
            .collect();
        slugs.sort_unstable();
        slugs.dedup();
        assert_eq!(slugs.len(), 4);
    }

    /// A degraded rail NEVER degrades silently — the rule the whole module
    /// exists to serve.
    #[test]
    fn every_degraded_rail_is_named() {
        for bits in 0u8..4 {
            let mechanisms = DmaMechanisms {
                host_pointer_import: bits & 1 != 0,
                dmabuf_share: bits & 2 != 0,
            };
            let profile = ZeroCopyProfile::resolve(mechanisms);
            let degraded = usize::from(!profile.guest_read.is_zero_copy())
                + usize::from(!profile.guest_write.is_zero_copy());
            let named = profile
                .declined_summary()
                .split(',')
                .filter(|s| !s.is_empty())
                .count();
            assert_eq!(named, degraded, "{mechanisms:?} left a rail unnamed");
        }
    }
}
