//! Always-on decline-reason census for the type-11 sampled zero-copy rail.
//!
//! When a type-11 sampled bind falls to the CPU byte loader
//! (`load_type11_rgba_static`, census branch `t11_guest`) instead of the
//! zero-copy guest gather (`t11_zc`), exactly one reason gated it out. Under
//! video playback `t11_guest` is the dominant remaining CPU copy (hundreds of
//! MB/session), so knowing *why* zero-copy declined names the next lever
//! precisely — a below-floor span wants a floor rethink, a stride/format
//! decline wants a format extension. This counts each decline by reason so one
//! boot log turns the lead into a fact.
//!
//! Measure-only — never gates behavior. Emitted through the always-on
//! `observe::off` sink from the drain worker (off the QEMU main core) every
//! [`EMIT_EVERY`] declines as a cumulative `t11_zc_decline` line.

use std::sync::atomic::{AtomicU64, Ordering};

/// Why a type-11 sampled bind declined the zero-copy guest gather and fell to
/// the CPU byte loader. One variant per distinct early-out in the rail.
///
/// There is no call-site gate to name any more: the gather is attempted for
/// every bind that reaches the guest-pages rung, because reaching it already
/// means no host-side copy served the bind.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Reason {
    /// Mapping absent, unmapped, or carries no page entries.
    Unmapped,
    /// Pixel format is not one of the byte-identical zero-copy formats
    /// (BGRA8/RGBA8) — needs a swizzle/convert the gather cannot do.
    BadFormat,
    /// `type11_sample_window` could not resolve a plane/surface window.
    NoWindow,
    /// Row stride is narrower than tight or not a whole texel multiple.
    Stride,
    /// Span is below `ZERO_COPY_SAMPLED_MIN_BYTES` (the perf floor).
    BelowFloor,
    /// HostOps map_pages returns a transient view (arm64 mach_vm_remap), so the
    /// Vulkan host-import cache must not retain the pointer.
    UnstableMap,
    /// Guest page GPAs unavailable or cover less than the window.
    Coverage,
    /// A page run failed to map to a host pointer, or the coalesced runs did
    /// not cover the full span, or a host import was refused.
    ImportFail,
}

const N: usize = 8;

impl Reason {
    const fn idx(self) -> usize {
        match self {
            Reason::Unmapped => 0,
            Reason::BadFormat => 1,
            Reason::NoWindow => 2,
            Reason::Stride => 3,
            Reason::BelowFloor => 4,
            Reason::UnstableMap => 5,
            Reason::Coverage => 6,
            Reason::ImportFail => 7,
        }
    }
}

const NAMES: [&str; N] = [
    "unmapped",
    "bad_format",
    "no_window",
    "stride",
    "below_floor",
    "unstable_map",
    "coverage",
    "import_fail",
];

static COUNTS: [AtomicU64; N] = [const { AtomicU64::new(0) }; N];
/// Sum of RGBA bytes the CPU loader materialized for each declined bind, so a
/// boot shows not just how often but how much copy each reason costs.
static BYTES: [AtomicU64; N] = [const { AtomicU64::new(0) }; N];
static TOTAL: AtomicU64 = AtomicU64::new(0);

/// One cumulative census line per this many declines.
const EMIT_EVERY: u64 = 256;

/// Record one type-11 zero-copy decline. `copied_bytes` is the RGBA byte count
/// the CPU loader then materialized for this bind.
pub fn note(reason: Reason, copied_bytes: usize) {
    let i = reason.idx();
    COUNTS[i].fetch_add(1, Ordering::Relaxed);
    BYTES[i].fetch_add(copied_bytes as u64, Ordering::Relaxed);
    let total = TOTAL.fetch_add(1, Ordering::Relaxed) + 1;
    if total.is_multiple_of(EMIT_EVERY) {
        crate::observe::off(format_line(&snapshot()));
    }
}

/// Cumulative (count, bytes) per reason, indexed as [`NAMES`]; last = total.
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
    let mut line = format!("t11_zc_decline total={total}");
    for (i, name) in NAMES.iter().enumerate() {
        let _ = write!(line, " {name}={}:{}", s[i].0, s[i].1);
    }
    line
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn note_accumulates_per_reason_and_line_names_every_reason() {
        let (before, before_total) = snapshot();
        note(Reason::BelowFloor, 243_000);
        note(Reason::Unmapped, 0);
        let (after, after_total) = snapshot();
        assert_eq!(after_total - before_total, 2);
        let bf = Reason::BelowFloor.idx();
        assert_eq!(after[bf].0 - before[bf].0, 1);
        assert_eq!(after[bf].1 - before[bf].1, 243_000);

        let line = format_line(&(after, after_total));
        assert!(line.starts_with("t11_zc_decline total="));
        for name in NAMES {
            assert!(
                line.contains(&format!(" {name}=")),
                "decline line missing reason {name}: {line}"
            );
        }
    }
}
