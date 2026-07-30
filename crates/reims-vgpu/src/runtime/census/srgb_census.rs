//! Always-on proxy for the sRGB-downgrade bug class.
//!
//! # The class
//!
//! The guest names an sRGB render target or texture; the host binds the linear
//! sibling of that format. The hardware then never applies the sRGB transfer
//! function, so blending and sampling happen in the wrong colour space. The
//! defect is not that the fold exists — several rails genuinely carry raw bytes
//! and cannot encode — it is that the fold used to be **silent**: a lost format
//! qualifier looked exactly like a supported format, at twelve independent
//! sites, with nothing in the fail log.
//!
//! # Reading it
//!
//! `/tmp/reims-vgpu-fail.log`, always-on, one line per (site, format) pair:
//!
//! * `srgb_downgraded reason=srgb_downgraded site=<site> mtl=<fmt> …`
//!
//! **No lines on a healthy boot means the guest never asked for sRGB.** A line
//! is not itself a failure; it says which rail is trading colour-space
//! correctness away, which is the only thing needed to decide where adopting
//! `VK_FORMAT_*_SRGB` would pay. The pair is the unit because a rail hit twice
//! with the same format has nothing more to say the second time.
//!
//! Measure-only: nothing here gates decode, execute or present.

use std::collections::BTreeSet;
use std::sync::Mutex;

use crate::observe;

/// The slug every downgrade line carries. Kept equal to
/// `TranslateReason::SrgbDowngraded`'s slug by a unit test in the Vulkan
/// backend, so the typed reason and the always-on line cannot drift apart.
pub const SRGB_DOWNGRADED_SLUG: &str = "srgb_downgraded";

/// The rails that can drop an sRGB qualifier, each named for the code path a
/// reader would open. One constant per site so a log line points somewhere
/// specific rather than at "the sampled path".
pub mod site {
    /// `try_linear_sample_zero_copy` — type-2/3 linear texture gathered from
    /// guest RAM straight into a sampled image.
    pub const LINEAR_SAMPLE_ZERO_COPY: &str = "linear_sample_zero_copy";
    /// `try_type11_sample_zero_copy` — type-11 surface window sampled in place.
    pub const TYPE11_SAMPLE_ZERO_COPY: &str = "type11_sample_zero_copy";
    /// `try_type5_plane_zero_copy` — type-5 multiplanar view sampled in place.
    pub const TYPE5_PLANE_ZERO_COPY: &str = "type5_plane_zero_copy";
    /// `build_secondary_targets` — MRT colour attachment beyond slot 0.
    pub const SECONDARY_COLOR_TARGET: &str = "secondary_color_target";
    /// `linear_native_upload_format` — guest bytes uploaded in their native
    /// order with no convert pass.
    pub const LINEAR_NATIVE_UPLOAD: &str = "linear_native_upload";
    /// `load_tight_linear_rgba_with` — tight-row CPU load of a linear texture.
    pub const TIGHT_LINEAR_LOAD: &str = "tight_linear_load";

    /// Every site, for the completeness test. A new site constant that is not
    /// listed here is one the census cannot report on.
    pub const ALL: &[&str] = &[
        LINEAR_SAMPLE_ZERO_COPY,
        TYPE11_SAMPLE_ZERO_COPY,
        TYPE5_PLANE_ZERO_COPY,
        SECONDARY_COLOR_TARGET,
        LINEAR_NATIVE_UPLOAD,
        TIGHT_LINEAR_LOAD,
    ];
}

/// `(site, MTLPixelFormat)` pairs already reported, so a per-draw rail costs one
/// line per distinct pair per boot. Bounded by `site::ALL` times the small set
/// of sRGB formats.
static SEEN: Mutex<BTreeSet<(&'static str, u16)>> = Mutex::new(BTreeSet::new());

/// Record that `site` bound the linear sibling of the sRGB format `mtl`.
///
/// Call this at the moment the qualifier is dropped, not at the moment the
/// format is decoded — the point is to name what was actually traded away.
pub fn note_downgrade(site: &'static str, mtl: u16) {
    let first_sight = SEEN
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert((site, mtl));
    if first_sight {
        observe::fail(format!(
            "srgb_downgraded reason={SRGB_DOWNGRADED_SLUG} site={site} mtl={mtl:#x} \
             (bound the linear sibling; hardware will not apply the sRGB transfer \
             function on this rail)"
        ));
    }
}

/// Drop the first-sight set. Test-only: it is process-global, so a test that
/// asserts a line was emitted must start from a known point.
#[cfg(test)]
pub fn reset_for_tests() {
    SEEN.lock().unwrap_or_else(|e| e.into_inner()).clear();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::pixel_format::{MTL_FORMAT_BGRA8_UNORM_SRGB, MTL_FORMAT_RGBA8_UNORM_SRGB};

    /// A per-draw rail must cost one line per distinct (site, format) pair, not
    /// one per bind — the dedup is what makes it safe to leave on forever, and
    /// a second pair on the same site is still a new event.
    #[test]
    fn each_site_and_format_pair_reports_once() {
        reset_for_tests();
        assert!(SEEN
            .lock()
            .unwrap()
            .insert((site::LINEAR_SAMPLE_ZERO_COPY, MTL_FORMAT_BGRA8_UNORM_SRGB)));
        for _ in 0..64 {
            note_downgrade(site::LINEAR_SAMPLE_ZERO_COPY, MTL_FORMAT_BGRA8_UNORM_SRGB);
        }
        assert_eq!(SEEN.lock().unwrap().len(), 1, "64 binds, one pair");
        note_downgrade(site::LINEAR_SAMPLE_ZERO_COPY, MTL_FORMAT_RGBA8_UNORM_SRGB);
        note_downgrade(site::SECONDARY_COLOR_TARGET, MTL_FORMAT_RGBA8_UNORM_SRGB);
        assert_eq!(
            SEEN.lock().unwrap().len(),
            3,
            "a new format and a new site are each a new event"
        );
        reset_for_tests();
    }

    /// Site names are distinct and log-safe — a duplicate would merge two
    /// rails' counts and a space would break the field split.
    #[test]
    fn site_names_are_distinct_and_log_safe() {
        let mut names = site::ALL.to_vec();
        names.sort_unstable();
        let before = names.len();
        names.dedup();
        assert_eq!(before, names.len(), "duplicate site name");
        for name in site::ALL {
            assert!(name
                .bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_'));
        }
    }
}
