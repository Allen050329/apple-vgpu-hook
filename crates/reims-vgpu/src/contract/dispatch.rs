//! What a decoded compute dispatch record means: its `MTLDispatchType`, and the
//! threadgroup counts its grid resolves to.
//!
//! Both are read off the wire and neither is backend-specific, so both are
//! answered here once rather than at each backend's encode site. Each is a
//! *closed* rule with a substitution behind it, which is why they are functions
//! and not open-coded comparisons — see
//! [`crate::contract::dispatch::workgroup_counts`] for what the split version of
//! the second one cost.
//!
//! # The dispatch type
//!
//! Two ordinals, and the only interesting thing about them is that the accepted
//! set is *closed*: `MTLDispatchType` in the Metal SDK has exactly `Serial` and
//! `Concurrent`, so unlike a pixel format or a primitive type there is no
//! "value this device has no contract for yet" to leave room for. An ordinal
//! outside the pair is a malformed record or a wrong wire offset, not a guest
//! feature.
//!
//! # Why they live here rather than in the backend
//!
//! The value arrives on the wire, from the guest, and is decoded by
//! [`crate::runtime::decode::compute`] — none of which is backend-specific. It
//! was previously reachable only through `backend::metal::abi`, which is
//! `backend-metal`-gated, so the shared code that accepts the field could not
//! name the values it was accepting and the one place that narrowed it ran on a
//! single arm. `contract/` is where a number that comes from the wire and the
//! SDK belongs, per this module tree's own doc.
//!
//! `backend::metal::abi` keeps its own spelling of the pair, because that module
//! is a mirror of an archived C header and its provenance is the point. A `const`
//! assertion there pins the two spellings equal, so a divergence is a build
//! failure on every arm that compiles the mirror — including the cross-compiled
//! `--target aarch64-apple-darwin` clippy run `AGENTS.md` requires from Linux.

/// `MTLDispatchTypeSerial` — dispatches in a segment may not overlap.
pub const MTL_DISPATCH_TYPE_SERIAL: u32 = 0;
/// `MTLDispatchTypeConcurrent` — Metal may overlap dispatches in a segment.
pub const MTL_DISPATCH_TYPE_CONCURRENT: u32 = 1;

/// Whether `raw` is one of the two dispatch types the contract declares.
///
/// Beside the constants on purpose. The rule this answers used to be written
/// out at the site that consumed it, as
/// `if x == CONCURRENT { CONCURRENT } else { SERIAL }` — a comparison that reads
/// as a narrowing and is really an unreported substitution, and which nothing
/// could compare against the constants it was narrowing to.
#[must_use]
pub fn is_declared_dispatch_type(raw: u32) -> bool {
    raw == MTL_DISPATCH_TYPE_SERIAL || raw == MTL_DISPATCH_TYPE_CONCURRENT
}

/// The threadgroup counts a dispatch resolves to, or `None` if it has no work.
///
/// `grid` is the record's grid and `tg` its threads-per-threadgroup, both
/// straight off the wire. `grid_is_threads` distinguishes Metal's two spellings:
/// `DispatchThreadgroups` states the count directly, while `DispatchThreads`
/// states a total thread count that Metal divides by the threadgroup size,
/// rounding up — which is what `div_ceil` reproduces here.
///
/// # Why the zero test and the division are one function
///
/// They were two, about two hundred lines apart, and the distance is what made
/// the pair wrong: the division carried a `.max(1)` on each quotient, which the
/// zero test above it had already made unreachable. That clamp reads as
/// prudence and is the opposite. A `grid` component of zero is a guest asking
/// for no threads, and the only faithful answer is no dispatch — fabricating
/// one threadgroup runs the kernel, and a kernel that runs writes the storage
/// buffers and images bound to it. So the substitution this device must never
/// make is available only where the test that forbids it also lives.
///
/// A zero in `tg` is refused for a second reason: it is the divisor, so the
/// alternative to refusing it is a panic on a value that came off the wire.
#[must_use]
pub fn workgroup_counts(grid: [u32; 3], tg: [u32; 3], grid_is_threads: bool) -> Option<[u32; 3]> {
    if grid.iter().chain(&tg).any(|&d| d == 0) {
        return None;
    }
    if !grid_is_threads {
        return Some(grid);
    }
    Some([
        grid[0].div_ceil(tg[0]),
        grid[1].div_ceil(tg[1]),
        grid[2].div_ceil(tg[2]),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A zero anywhere in either triple means no dispatch, in both spellings.
    ///
    /// Exhaustive over which component is zero, because the refusal is what
    /// stands between a guest that asked for no threads and a kernel this
    /// device ran anyway. A predicate that checked only `grid[0]` — which is
    /// the shape the sibling mesh-draw path still has — passes a test that
    /// zeroes the first component and nothing else.
    #[test]
    fn a_zero_in_any_dimension_dispatches_nothing() {
        for grid_is_threads in [false, true] {
            for i in 0..6 {
                let mut grid = [4u32, 4, 4];
                let mut tg = [2u32, 2, 2];
                if i < 3 {
                    grid[i] = 0
                } else {
                    tg[i - 3] = 0
                }
                assert_eq!(
                    workgroup_counts(grid, tg, grid_is_threads),
                    None,
                    "grid={grid:?} tg={tg:?} threads={grid_is_threads}"
                );
            }
        }
    }

    /// `DispatchThreadgroups` passes the guest's count through untouched.
    #[test]
    fn a_threadgroup_count_is_not_divided() {
        assert_eq!(
            workgroup_counts([7, 3, 1], [8, 8, 1], false),
            Some([7, 3, 1])
        );
    }

    /// `DispatchThreads` rounds up, and never past what the guest asked for.
    ///
    /// The exact-multiple case is the one that would hide an off-by-one: 16
    /// threads in groups of 8 is two groups, not three. The partial case pins
    /// the rounding direction — Metal launches the group that covers the
    /// remainder, so a trailing thread is never dropped.
    #[test]
    fn a_thread_count_rounds_up_to_whole_threadgroups() {
        assert_eq!(
            workgroup_counts([16, 16, 1], [8, 8, 1], true),
            Some([2, 2, 1])
        );
        assert_eq!(
            workgroup_counts([17, 1, 1], [8, 1, 1], true),
            Some([3, 1, 1])
        );
        assert_eq!(
            workgroup_counts([1, 1, 1], [64, 64, 64], true),
            Some([1, 1, 1])
        );
        assert_eq!(
            workgroup_counts([u32::MAX, 1, 1], [1, 1, 1], true),
            Some([u32::MAX, 1, 1]),
            "the widest grid a u32 can carry divides without wrapping"
        );
    }

    /// The accepted set is exactly the two declared ordinals.
    ///
    /// Worth pinning rather than assuming, because the predicate's whole job is
    /// to be *closed*: the substitution it guards is chosen for every value it
    /// rejects, so a predicate that accidentally accepted a third ordinal would
    /// pass that value through to a Metal enum conversion that has no arm for
    /// it. The sweep runs past both constants in both directions.
    #[test]
    fn only_the_two_declared_dispatch_types_are_accepted() {
        assert!(is_declared_dispatch_type(MTL_DISPATCH_TYPE_SERIAL));
        assert!(is_declared_dispatch_type(MTL_DISPATCH_TYPE_CONCURRENT));
        for raw in 2..=64u32 {
            assert!(
                !is_declared_dispatch_type(raw),
                "{raw} is not a declared MTLDispatchType"
            );
        }
        assert!(!is_declared_dispatch_type(u32::MAX));
    }
}
