//! The arguments of one direct draw.

/// What a `drawPrimitives` / `drawIndexedPrimitives` record asks for, as one
/// value.
///
/// Its own type for the same reason [`super::extent::Extent3`] is: the hazard is
/// at the call boundary, not at construction. These five were decoded into a
/// struct and then destructured back into loose `u32`s to cross two of them —
/// `draw::mrt_draw_request` took `(vertex_count, instance_count,
/// primitive_type, first_vertex, base_instance)` and
/// `backend::metal::render::render_core_mrt`, one call further down the same
/// draw, took the same five as `(vertex_count, first_vertex, instance_count,
/// base_instance, primitive_type)`. Two orders, both positional, both all-`u32`
/// or all-`usize`, so every one of the 120 permutations compiled at each site
/// and the two sites did not even agree with each other.
///
/// A transposition here does not fail: it draws a valid primitive of the wrong
/// shape, or the right vertices of the wrong instance, which nothing downstream
/// can distinguish from the draw the guest asked for.
///
/// What this does not close: the fields are still five `u32`s, so a *builder*
/// that names them wrongly compiles. That hazard is at construction, where the
/// field names are written out and a reader can check them against the decoder,
/// and it is not the one that has bitten.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DrawArgs {
    pub vertex_count: u32,
    pub instance_count: u32,
    pub primitive_type: u32,
    pub first_vertex: u32,
    /// Metal `baseInstance` / Vulkan `firstInstance`.
    pub base_instance: u32,
}
