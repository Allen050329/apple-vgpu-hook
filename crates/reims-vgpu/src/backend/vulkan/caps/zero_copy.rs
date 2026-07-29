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
//! The rails are independent, and on this backend they are also permanently
//! split: `VK_EXT_external_memory_host` is never requested, so both memory
//! rails sit on their copy rung on every host while the display rail still
//! reaches dmabuf or a same-device swapchain. Collapsing them into one flag
//! would report every host as "half zero-copy" and name neither half.
//!
//! # Why the axis is still binary in the matrix
//!
//! [`DmaSupport`] answers the coarse question the support matrix asks — does
//! this host reach *any* zero-copy rung, or does every crossing cost a copy? A
//! `NoDma` host is fully supported and fully tested; it is just slower, and the
//! log line names which rung each rail landed on so "why is this host slow" is
//! answerable without a debugger.
//!
//! # Why the memory rails have no zero-copy rung any more
//!
//! `VK_EXT_external_memory_host` *registers* — pins — every page it imports,
//! and a page the GPU can read is one it can write. Bounding that with windows,
//! byte caps and an idle sweep was tried and is gone: the mechanism was removed
//! rather than tuned, because no residency policy changes what the GPU is
//! allowed to do with a page it holds. [`GuestRead::ImportedPages`] and
//! [`GuestWrite::GpuDirect`] therefore name rungs this backend cannot select;
//! they are kept so the log line and the matrix can say which rung was *lost*
//! rather than reporting the copy as if it were the only shape that existed.

use crate::observe::Decline;

/// The mechanisms by which this device can address memory it did not itself
/// allocate. Detected once at device create; never re-queried.
///
/// These are *different boundaries*, not alternatives — see the module docs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DmaMechanisms {
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
        if self.dmabuf_share {
            "dmabuf"
        } else {
            "none"
        }
    }

    /// True when at least one mechanism exists, i.e. some crossing can avoid a
    /// copy. This is the matrix axis.
    ///
    /// Only the display rail can answer yes. Both memory rails lost their
    /// mechanism with the host-pointer import, so a host reaching [`DmaSupport::Dma`]
    /// does so on dmabuf alone — see [`ZeroCopyProfile::resolve`].
    pub fn any(self) -> bool {
        self.dmabuf_share
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
    /// No host pointer can become `VkDeviceMemory`, so both memory rails fall
    /// to a copy. Unconditional: `VK_EXT_external_memory_host` is never
    /// requested, however the driver advertises it, because a host pointer
    /// imported over guest RAM is a host GPU that can write the guest VM's
    /// memory.
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
    /// Both memory rails land on their copy rung on **every** device, because
    /// the only mechanism either ever had was `VK_EXT_external_memory_host` and
    /// this backend no longer requests it. That is a degradation, not expected
    /// control flow, so it is declined by name here — one
    /// `vk_caps_zero_copy_declined reason=no_host_pointer_import` per rail per
    /// device create, which is where "why does this host copy every guest
    /// texture" is answered.
    ///
    /// The two rails stay separate entries rather than one, because they cost
    /// different things (a per-bind staging gather vs a per-store readback) and
    /// a reader chasing one should not have to know they were ever the same
    /// mechanism.
    pub fn resolve(mechanisms: DmaMechanisms) -> Self {
        Self {
            mechanisms,
            guest_read: GuestRead::StagingCopy,
            guest_write: GuestWrite::CpuReadback,
            declined: vec![
                ("guest_read", ZeroCopyDecline::NoHostPointerImport),
                ("guest_write", ZeroCopyDecline::NoHostPointerImport),
            ],
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

    /// The arm64 pathway after the import removal: MoltenVK has no dmabuf and
    /// no host-pointer import, so every crossing costs a copy and the host
    /// lands in the non-DMA column. Supported, just slower.
    #[test]
    fn no_mechanism_resolves_every_rail_to_its_fallback() {
        let profile = ZeroCopyProfile::resolve(DmaMechanisms::default());
        assert_eq!(profile.support(), DmaSupport::NoDma);
        assert!(!profile.guest_read.is_zero_copy());
        assert!(!profile.guest_write.is_zero_copy());
        assert_eq!(profile.mechanisms.slug(), "none");
    }

    /// A Linux host with dmabuf is still DMA — the display rail is zero-copy —
    /// but BOTH memory rails must name why they degraded. Reporting a bare
    /// "dma" here is the failure this test locks out.
    #[test]
    fn dmabuf_still_names_both_degraded_memory_rails() {
        let profile = ZeroCopyProfile::resolve(DmaMechanisms { dmabuf_share: true });
        assert_eq!(profile.support(), DmaSupport::Dma);
        assert_eq!(profile.guest_read, GuestRead::StagingCopy);
        assert_eq!(profile.guest_write, GuestWrite::CpuReadback);
        assert_eq!(
            profile.declined_summary(),
            "guest_read:no_host_pointer_import,guest_write:no_host_pointer_import"
        );
        assert_eq!(profile.mechanisms.slug(), "dmabuf");
    }

    /// The memory rails decline **whatever** the device advertises. This is the
    /// property the removal has to hold, and it is stated over the whole
    /// mechanism space rather than on one fixture: no input reaches
    /// [`GuestRead::ImportedPages`] or [`GuestWrite::GpuDirect`].
    ///
    /// It is also why the display rail is checked in the same loop — a change
    /// that accidentally pinned *every* rail to its fallback would still pass a
    /// test that only asserted the memory rails, and it would silently cost the
    /// dmabuf handoff.
    #[test]
    fn no_device_shape_reaches_a_zero_copy_memory_rung() {
        for dmabuf_share in [false, true] {
            let mechanisms = DmaMechanisms { dmabuf_share };
            let profile = ZeroCopyProfile::resolve(mechanisms);
            assert!(!profile.guest_read.is_zero_copy(), "{mechanisms:?}");
            assert!(!profile.guest_write.is_zero_copy(), "{mechanisms:?}");
            assert_eq!(
                profile.support(),
                if dmabuf_share {
                    DmaSupport::Dma
                } else {
                    DmaSupport::NoDma
                },
                "the axis must track the display rail alone: {mechanisms:?}"
            );
        }
    }

    /// Each mechanism combination has a distinct slug, so the selection line
    /// distinguishes "no dmabuf" from "dmabuf" on the summary line.
    #[test]
    fn mechanism_slugs_are_distinct() {
        let mut slugs: Vec<_> = [false, true]
            .into_iter()
            .map(|dmabuf_share| DmaMechanisms { dmabuf_share }.slug())
            .collect();
        slugs.sort_unstable();
        slugs.dedup();
        assert_eq!(slugs.len(), 2);
    }

    /// A degraded rail NEVER degrades silently — the rule the whole module
    /// exists to serve. Both memory rails are degraded on every host now, so
    /// both must be named on every host.
    #[test]
    fn every_degraded_rail_is_named() {
        for dmabuf_share in [false, true] {
            let mechanisms = DmaMechanisms { dmabuf_share };
            let profile = ZeroCopyProfile::resolve(mechanisms);
            let degraded = usize::from(!profile.guest_read.is_zero_copy())
                + usize::from(!profile.guest_write.is_zero_copy());
            let named = profile
                .declined_summary()
                .split(',')
                .filter(|s| !s.is_empty())
                .count();
            assert_eq!(named, degraded, "{mechanisms:?} left a rail unnamed");
            assert_eq!(named, 2, "both memory rails degrade on every host");
        }
    }
}
