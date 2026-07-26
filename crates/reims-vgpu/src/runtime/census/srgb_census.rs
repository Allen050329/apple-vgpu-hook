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
//! `/tmp/reims-vgpu-fail.log`, always-on:
//!
//! * `srgb_downgraded reason=srgb_downgraded site=<site> mtl=<fmt> …` — first
//!   sight of one (site, format) pair. Deduplicated, so the bound is the number
//!   of distinct pairs (a handful per boot), never per draw.
//! * `OFF srgb_census total=<n> sites=[<site>:<n> …]` — cumulative volume,
//!   emitted on a doubling schedule so a hot rail reports without flooding.
//!
//! **`total=0` on a healthy boot means the guest never asked for sRGB.** A
//! nonzero total is not itself a failure; it is the measurement that says how
//! much colour-space correctness is currently being traded away, and which rail
//! is trading it. Adopting `VK_FORMAT_*_SRGB` on a rail is only worth doing
//! where this proxy says the rail is actually hit.
//!
//! Measure-only: nothing here gates decode, execute or present.

use std::collections::BTreeMap;
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

#[derive(Default)]
struct Census {
    /// (site, MTLPixelFormat) pairs already reported once.
    seen: std::collections::BTreeSet<(&'static str, u16)>,
    per_site: BTreeMap<&'static str, u64>,
    total: u64,
    /// Next `total` at which the cumulative line is emitted.
    next_emit: u64,
}

static CENSUS: Mutex<Option<Census>> = Mutex::new(None);

fn with<R>(f: impl FnOnce(&mut Census) -> R) -> R {
    let mut guard = CENSUS.lock().unwrap_or_else(|e| e.into_inner());
    let census = guard.get_or_insert_with(|| Census {
        next_emit: 1,
        ..Census::default()
    });
    f(census)
}

/// Record that `site` bound the linear sibling of the sRGB format `mtl`.
///
/// Call this at the moment the qualifier is dropped, not at the moment the
/// format is decoded — the point is to count what was actually traded away.
pub fn note_downgrade(site: &'static str, mtl: u16) {
    let (first_sight, total, line) = with(|c| {
        c.total += 1;
        *c.per_site.entry(site).or_default() += 1;
        let first_sight = c.seen.insert((site, mtl));
        let line = if c.total >= c.next_emit {
            c.next_emit = c.total.saturating_mul(2);
            Some(summary_line(c))
        } else {
            None
        };
        (first_sight, c.total, line)
    });
    if first_sight {
        observe::fail(format!(
            "srgb_downgraded reason={SRGB_DOWNGRADED_SLUG} site={site} mtl={mtl:#x} \
             (bound the linear sibling; hardware will not apply the sRGB transfer \
             function on this rail) seen={total}"
        ));
    }
    if let Some(line) = line {
        observe::off(line);
    }
}

fn summary_line(c: &Census) -> String {
    let sites = c
        .per_site
        .iter()
        .map(|(s, n)| format!("{s}:{n}"))
        .collect::<Vec<_>>()
        .join(",");
    format!("srgb_census total={} sites=[{sites}]", c.total)
}

/// Cumulative downgrade count, total and per site. Test/diagnostic accessor.
pub fn counts() -> (u64, BTreeMap<&'static str, u64>) {
    with(|c| (c.total, c.per_site.clone()))
}

/// Drop all accumulated state. Test-only: the census is process-global, so
/// tests that assert on counts must start from a known point.
#[cfg(test)]
pub fn reset_for_tests() {
    with(|c| {
        *c = Census {
            next_emit: 1,
            ..Census::default()
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::pixel_format::{MTL_FORMAT_BGRA8_UNORM_SRGB, MTL_FORMAT_RGBA8_UNORM_SRGB};

    /// The census counts every downgrade but reports each (site, format) pair
    /// once. Both halves are load-bearing: the count is the measurement, the
    /// dedup is what keeps a per-draw rail from burying the rest of the fail
    /// log.
    #[test]
    fn counts_every_downgrade_but_names_each_pair_once() {
        reset_for_tests();
        for _ in 0..64 {
            note_downgrade(site::LINEAR_SAMPLE_ZERO_COPY, MTL_FORMAT_BGRA8_UNORM_SRGB);
        }
        note_downgrade(site::SECONDARY_COLOR_TARGET, MTL_FORMAT_RGBA8_UNORM_SRGB);
        let (total, per_site) = counts();
        assert_eq!(total, 65);
        assert_eq!(per_site[site::LINEAR_SAMPLE_ZERO_COPY], 64);
        assert_eq!(per_site[site::SECONDARY_COLOR_TARGET], 1);
        // 65 events, but only two distinct pairs were first-sighted.
        assert_eq!(
            with(|c| c.seen.len()),
            2,
            "one first-sight line per (site, format) pair"
        );
        reset_for_tests();
    }

    /// The cumulative line names the total and every contributing rail, so
    /// "which path is trading colour correctness away" is answerable from one
    /// grep.
    #[test]
    fn the_summary_line_names_total_and_every_site() {
        reset_for_tests();
        note_downgrade(site::TYPE11_SAMPLE_ZERO_COPY, MTL_FORMAT_BGRA8_UNORM_SRGB);
        note_downgrade(site::TIGHT_LINEAR_LOAD, MTL_FORMAT_RGBA8_UNORM_SRGB);
        let line = with(|c| summary_line(c));
        assert!(line.contains("srgb_census total=2"), "{line}");
        assert!(line.contains("type11_sample_zero_copy:1"), "{line}");
        assert!(line.contains("tight_linear_load:1"), "{line}");
        reset_for_tests();
    }

    /// Emission doubles, so a rail hit thousands of times per second costs a
    /// logarithmic number of lines rather than one per event.
    #[test]
    fn the_summary_emits_on_a_doubling_schedule() {
        reset_for_tests();
        let mut emits = 0;
        for _ in 0..4096 {
            let emitted = with(|c| {
                c.total += 1;
                if c.total >= c.next_emit {
                    c.next_emit = c.total.saturating_mul(2);
                    true
                } else {
                    false
                }
            });
            emits += u32::from(emitted);
        }
        assert_eq!(emits, 13, "4096 events must cost log2 lines, not 4096");
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
