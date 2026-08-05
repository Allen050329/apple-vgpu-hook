//! The `MTLDispatchType` a compute segment's `WRITE_DESCRIPTOR` record carries.
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

#[cfg(test)]
mod tests {
    use super::*;

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
