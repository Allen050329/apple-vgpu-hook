//! A three-dimensional compute extent.

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

impl Extent3 {
    /// Whether any dimension is zero, which is not a dispatch.
    pub fn has_zero_dimension(self) -> bool {
        self.x == 0 || self.y == 0 || self.z == 0
    }
}
