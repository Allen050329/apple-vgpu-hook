//! Extents the guest's API defines: a three-dimensional compute extent, and
//! the size of one mip level of a texture.

/// A grid or threadgroup size, as three dimensions that travel together.
///
/// Its own type rather than three `u32`s because a dispatch carries **two** of
/// these side by side, built from sources that look alike — three consecutive
/// little-endian words for the indirect arms — and a transposition between them
/// dispatches a valid grid of the wrong shape, which nothing downstream can tell
/// from the right one.
///
/// It lives in `contract` rather than beside the decoder that first needed it
/// because the hazard is at the *boundary*, not at construction. The decoder
/// built two of these correctly and then destructured both back into six loose
/// `u32` parameters to reach the backend, where every one of the 720 orderings
/// compiles again — so the type protected the half of the journey that was
/// already safe and stopped exactly where the two extents become
/// interchangeable. Both sides of that call now name it.
///
/// What this does **not** close: two `Extent3` arguments are still the same
/// type, so passing the threadgroup where the grid belongs compiles. That is
/// the one remaining transposition of the 720, and it is the one the callers'
/// own `grid` / `threadgroup` bindings name at every site. Closing it needs two
/// newtypes, and the argument for them is a measurement nobody has: it has not
/// happened, whereas a `grid_y`/`grid_z` slip in a six-argument run is the kind
/// that has.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Extent3 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

/// Metal's dimension for mip `level` of an axis whose level-0 size is `base`.
///
/// Each level halves and floors, and the chain stops at 1 rather than reaching
/// 0 — `MTLTexture`'s level sizes are `max(1, base >> level)`, which is why a
/// 100-wide texture's levels run 100, 50, 25, 12, ... and its last level is 1
/// and not 0. `base == 0` is the one case that is not a clamp: an axis with no
/// texels has no levels, and answering 1 there would size a read of a texture
/// that does not exist.
///
/// Here rather than in either backend because both rails need the same answer
/// and each used to hold its own copy — `backend::metal::mipmap` for the
/// filtered generator, and a cfg-forked `metal_mip_extent_local` in
/// `runtime::mipmap` whose Vulkan arm reimplemented the line the Metal arm
/// called. That fork also decided nothing: the two arms were identical, so the
/// only thing the `#[cfg]` could ever change was which copy ran.
///
/// The runtime resolver rejects any stored mip layout whose extent disagrees
/// with this, so a wrong formula either refuses valid mip chains or accepts a
/// mismatched layout that then samples out of bounds.
pub fn mip_extent(base: u32, level: u32) -> u32 {
    if base == 0 {
        return 0;
    }
    (base >> level).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_mip_level_halves_and_floors_at_one() {
        // An axis with no texels has no levels.
        assert_eq!(mip_extent(0, 0), 0);
        assert_eq!(mip_extent(0, 3), 0);

        // Power-of-two base halves each level and floors at 1, never 0.
        assert_eq!(mip_extent(8, 0), 8);
        assert_eq!(mip_extent(8, 1), 4);
        assert_eq!(mip_extent(8, 2), 2);
        assert_eq!(mip_extent(8, 3), 1);
        assert_eq!(mip_extent(8, 4), 1, "past the last level clamps to 1");
        assert_eq!(mip_extent(8, 20), 1, "huge level never underflows to 0");

        // Non-power-of-two base right-shifts (floors), matching Metal.
        assert_eq!(mip_extent(5, 1), 2);
        assert_eq!(mip_extent(5, 2), 1);
        assert_eq!(mip_extent(3, 1), 1);
        assert_eq!(mip_extent(100, 1), 50);
        assert_eq!(mip_extent(100, 2), 25);
        assert_eq!(mip_extent(100, 3), 12);
    }
}
