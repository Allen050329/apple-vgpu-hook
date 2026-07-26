//! Always-on per-branch census for sampled-texture resolution.
//!
//! `resolve_sampled_source` / `load_linear_from_host_caches` resolve a
//! sampled bind through ~10 distinct branches; only some carry a content
//! identity that lets the runtime skip the per-draw copy+swizzle and the
//! engine skip hash+memcmp. This census counts every resolution by branch
//! (count + RGBA bytes actually materialized on the CPU) so one boot log
//! shows which branch dominates per-draw `setup_tex` cost and where the
//! identity rail must extend next. Measure-only — never gates behavior.
//!
//! Cumulative `sampled_branch_census` lines land in `/tmp/reims-vgpu-fail.log`
//! (always-on `observe::off` sink) every [`EMIT_EVERY`] resolutions; the
//! last line of a boot is the whole-boot census.

use std::sync::atomic::{AtomicU64, Ordering};

/// Which return path of the sampled-load resolver produced the bind.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Branch {
    /// Type-5 serialized view materialized from guest pages.
    Type5View,
    /// Resident GPU target bound directly (no CPU bytes).
    Resident,
    /// Type-11 host surface cache hit (copy+swizzle).
    T11Cache,
    /// Type-11 guest-page load.
    T11Guest,
    /// Deferred-Store GVA window: resident GPU target bound directly
    /// (no flush, no CPU bytes).
    GvaResident,
    /// Linear GVA-cache memo reuse (Arc clone, no copy).
    GvaMemo,
    /// Linear GVA-cache hit, copy+swizzle (memoized after).
    GvaCopy,
    /// Linear texture-ref cache hit with generation.
    RefCache,
    /// Linear guest-page load, tight-stride memo MISS: full re-read + fresh
    /// allocation + per-row convert + memo store (`load_linear_guest_memoized`).
    /// The Safari-scroll / video hot branch — content changes each frame so the
    /// memcmp fails and this pays alloc+convert per bind.
    LinGuest,
    /// Linear guest-page load via the non-memo FALLBACK loader
    /// (`load_linear_texture_rgba_host`, reached only when the tight-stride memo
    /// loader declines: padded/mismatched stride). In-place-optimal then a
    /// `.to_vec()` slice. Split from [`Branch::LinGuest`] so a boot shows which
    /// sub-path the bulk of `lin_guest` binds actually take.
    LinGuestFallback,
    /// Linear guest-page memo reuse (native re-read + byte-exact match; no
    /// conversion, no new allocation).
    LinMemo,
    /// Type-5 view memo reuse (native window re-read + byte-exact match; no
    /// conversion, no new allocation).
    T5Memo,
    /// Zero-copy guest gather: GPU reads imported guest RAM in the draw CB
    /// (no CPU bytes at all).
    LinZeroCopy,
    /// Type-11 mapping-backed zero-copy guest gather (GPU reads imported
    /// guest RAM; no CPU bytes).
    T11ZeroCopy,
    /// Type-5 serialized-view zero-copy guest gather — the video plane rail
    /// (R8/Rg8/BGRA8/RGBA8 plane imported directly; no CPU read+upload).
    T5ZeroCopy,
    /// Any-size texture-ref cache fallback.
    TexrefAny,
    /// Tail static linear/view loader with descriptor-recovered geometry.
    StaticTail,
    /// Type-11 mapping-backed memo reuse (native BGRA re-read + byte-exact
    /// match; no convert, no new allocation, engine skips re-hash+upload via
    /// the returned content identity). The dock-magnification hot branch.
    T11Memo,
}

const N: usize = 18;

impl Branch {
    const fn idx(self) -> usize {
        match self {
            Branch::Type5View => 0,
            Branch::Resident => 1,
            Branch::T11Cache => 2,
            Branch::T11Guest => 3,
            Branch::GvaResident => 4,
            Branch::GvaMemo => 5,
            Branch::GvaCopy => 6,
            Branch::RefCache => 7,
            Branch::LinGuest => 8,
            Branch::LinMemo => 9,
            Branch::T5Memo => 10,
            Branch::LinZeroCopy => 11,
            Branch::T11ZeroCopy => 12,
            Branch::TexrefAny => 13,
            Branch::StaticTail => 14,
            Branch::LinGuestFallback => 15,
            Branch::T5ZeroCopy => 16,
            Branch::T11Memo => 17,
        }
    }
}

const NAMES: [&str; N] = [
    "t5_view",
    "resident",
    "t11_cache",
    "t11_guest",
    "gva_resident",
    "gva_memo",
    "gva_copy",
    "ref_cache",
    "lin_guest",
    "lin_memo",
    "t5_memo",
    "lin_zc",
    "t11_zc",
    "texref_any",
    "static_tail",
    "lin_guest_fb",
    "t5_zc",
    "t11_memo",
];

static COUNTS: [AtomicU64; N] = [const { AtomicU64::new(0) }; N];
static BYTES: [AtomicU64; N] = [const { AtomicU64::new(0) }; N];
static TOTAL: AtomicU64 = AtomicU64::new(0);

/// One cumulative census line per this many sampled resolutions.
const EMIT_EVERY: u64 = 256;

/// Record one sampled-bind resolution. `copied_bytes` is the RGBA byte count
/// materialized on the CPU for this bind (0 for resident/memo reuse paths).
pub fn note(branch: Branch, copied_bytes: usize) {
    let i = branch.idx();
    COUNTS[i].fetch_add(1, Ordering::Relaxed);
    BYTES[i].fetch_add(copied_bytes as u64, Ordering::Relaxed);
    let total = TOTAL.fetch_add(1, Ordering::Relaxed) + 1;
    if total.is_multiple_of(EMIT_EVERY) {
        crate::observe::off(format_line(&snapshot()));
    }
}

/// Cumulative resolution count for one branch (monotonic; never reset). Used by
/// tests to assert an always-on branch fired without depending on a per-draw log
/// line (those are REIMS_VGPU_DRAW_LOG-gated to keep the fail log uncluttered).
pub fn count(branch: Branch) -> u64 {
    COUNTS[branch.idx()].load(Ordering::Relaxed)
}

/// Cumulative (count, bytes) per branch, indexed as [`NAMES`]; last = total.
pub fn snapshot() -> ([(u64, u64); N], u64) {
    let mut s = [(0u64, 0u64); N];
    for (i, slot) in s.iter_mut().enumerate() {
        *slot = (
            COUNTS[i].load(Ordering::Relaxed),
            BYTES[i].load(Ordering::Relaxed),
        );
    }
    (s, TOTAL.load(Ordering::Relaxed))
}

fn format_line(&(ref s, total): &([(u64, u64); N], u64)) -> String {
    use std::fmt::Write as _;
    let mut line = format!("sampled_branch_census total={total}");
    for (i, name) in NAMES.iter().enumerate() {
        let _ = write!(line, " {name}={}:{}", s[i].0, s[i].1);
    }
    line
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn note_accumulates_per_branch_and_line_names_every_branch() {
        // Global counters are shared across parallel tests: assert deltas only.
        let (before, before_total) = snapshot();
        note(Branch::GvaCopy, 4096);
        note(Branch::GvaMemo, 0);
        let (after, after_total) = snapshot();
        assert_eq!(after_total - before_total, 2);
        let gc = Branch::GvaCopy.idx();
        let gm = Branch::GvaMemo.idx();
        assert_eq!(after[gc].0 - before[gc].0, 1);
        assert_eq!(after[gc].1 - before[gc].1, 4096);
        assert_eq!(after[gm].0 - before[gm].0, 1);
        assert_eq!(after[gm].1, before[gm].1);

        let line = format_line(&(after, after_total));
        assert!(line.starts_with("sampled_branch_census total="));
        for name in NAMES {
            assert!(
                line.contains(&format!(" {name}=")),
                "census line missing branch {name}: {line}"
            );
        }
    }
}
