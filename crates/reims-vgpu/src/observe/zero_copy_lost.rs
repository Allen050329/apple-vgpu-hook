//! The rails that used to give the host GPU a handle on guest RAM, and no
//! longer do.
//!
//! Every variant here names one rail that had a zero-copy origin — an imported
//! host pointer, or a texture aliased straight over guest pages — and now runs
//! a CPU copy instead. The mechanism was removed rather than bounded: a host
//! pointer the GPU can read is one it can write, and no budget, granule or
//! window policy changes that.
//!
//! **This is a degradation notice, not a loss of guest work.** Each site below
//! completes the guest's request; what it lost is the copy elision. That is
//! still reportable under the never-fail-silently rule — a rail that quietly
//! costs a full-frame copy per present is exactly the kind of change that gets
//! attributed to something else six months later — but it is reported *once per
//! rail per process*, via [`crate::observe::Emit::fail_once`], because these
//! sites run per bind and per present. A per-event line here is a flood, and a
//! flood gets deleted, which is how a rail ends up with no record at all.
//!
//! Living in `observe/` rather than beside one of its callers is deliberate and
//! is the exception to this crate's colocate-the-decline rule: the whole point
//! of the vocabulary is that it spans both backends and four subsystems, so
//! `grep reason=zero_copy_lost` answers "what did the guest-RAM isolation cost"
//! in one query. Splitting it per owner would make that question unanswerable,
//! which is the same defect one-slug-per-check exists to prevent, inverted.

use super::Decline;

/// A rail that completed on the CPU because its zero-copy origin was removed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ZeroCopyLost {
    /// Draw-time vertex/storage buffer whose bytes live in guest runs. The runs
    /// are gathered into pooled staging by `write_staging_from_runs`.
    BufferGuestRuns,
    /// Draw-time sampled image whose texels live in guest runs. Same gather,
    /// then the usual buffer→image upload.
    SampledGuestRuns,
    /// Compute writeback that used to land straight in imported guest pages.
    /// Falls back to the readback-and-copy writeback.
    ComputeDirectWriteback,
    /// Present that used to DMA the finished frame into the guest's scanout
    /// pages — packed-contiguous and fragmented-scatter alike, which shared one
    /// decision point in the runtime. Falls back to the CPU writeback.
    ImportPresent,
    /// Console scanout that used to hand QEMU pixels the GPU wrote in place.
    /// Falls back to the CPU capture copy.
    ConsoleScanout,
    /// Metal texture that used to alias a `map_pages` view of guest RAM with
    /// `newBufferWithBytesNoCopy`. Falls back to a copied upload.
    MetalGuestTexture,
}

impl ZeroCopyLost {
    /// Stable per-rail discriminant for [`crate::observe::Emit::fail_once`], so
    /// each rail reports its first occurrence and then goes quiet independently
    /// of the others.
    ///
    /// Derived from the variant rather than the slug pointer: `fail_once` keys
    /// on a `u64`, and two rails colliding there would silence one of them for
    /// the process with nothing to say it had happened.
    pub fn discriminant(self) -> u64 {
        self as u64
    }

    /// Report this rail's first CPU-copy completion of the process.
    pub fn note(self) {
        super::Emit::decline("zero_copy_lost", &self).fail_once(self.discriminant());
    }
}

impl Decline for ZeroCopyLost {
    fn slug(&self) -> &'static str {
        match self {
            Self::BufferGuestRuns => "zero_copy_lost_buffer_guest_runs",
            Self::SampledGuestRuns => "zero_copy_lost_sampled_guest_runs",
            Self::ComputeDirectWriteback => "zero_copy_lost_compute_direct_writeback",
            Self::ImportPresent => "zero_copy_lost_import_present",
            Self::ConsoleScanout => "zero_copy_lost_console_scanout",
            Self::MetalGuestTexture => "zero_copy_lost_metal_guest_texture",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        vec![("fallback", self.fallback().to_string())]
    }
}

impl ZeroCopyLost {
    /// What ran instead. Named in the line because "the fast path is gone" is
    /// only half a diagnostic: a reader chasing a frame-time regression needs
    /// to know which copy they are now paying for.
    fn fallback(self) -> &'static str {
        match self {
            Self::BufferGuestRuns | Self::SampledGuestRuns => "staging_gather",
            Self::ComputeDirectWriteback => "readback_copy",
            Self::ImportPresent => "cpu_writeback",
            Self::ConsoleScanout => "cpu_capture_copy",
            Self::MetalGuestTexture => "copied_upload",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: &[ZeroCopyLost] = &[
        ZeroCopyLost::BufferGuestRuns,
        ZeroCopyLost::SampledGuestRuns,
        ZeroCopyLost::ComputeDirectWriteback,
        ZeroCopyLost::ImportPresent,
        ZeroCopyLost::ConsoleScanout,
        ZeroCopyLost::MetalGuestTexture,
    ];

    /// Two rails sharing a `fail_once` key silences one of them for the whole
    /// process, and the silence is indistinguishable from "it never ran" — the
    /// event-count-is-not-a-state trap, manufactured by the reporting itself.
    #[test]
    fn every_rail_reports_under_its_own_key_and_slug() {
        let mut keys: Vec<u64> = ALL.iter().map(|r| r.discriminant()).collect();
        keys.sort_unstable();
        keys.dedup();
        assert_eq!(keys.len(), ALL.len(), "rails collide on the fail_once key");

        let mut slugs: Vec<&str> = ALL.iter().map(|r| r.slug()).collect();
        slugs.sort_unstable();
        slugs.dedup();
        assert_eq!(slugs.len(), ALL.len(), "rails collide on the slug");
    }

    /// The line has to carry the replacement, not just the absence.
    #[test]
    fn every_rail_names_the_copy_it_now_pays() {
        for rail in ALL {
            let rendered = crate::observe::Emit::decline("zero_copy_lost", rail).render();
            assert!(
                rendered.contains(rail.slug()) && rendered.contains("fallback="),
                "{rendered}"
            );
        }
    }
}
