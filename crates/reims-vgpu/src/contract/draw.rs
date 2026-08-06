//! The arguments of one direct draw.

/// Bit `n` set = this device can execute `MTLPrimitiveType` `n`.
///
/// # This is an advertisement, and the guest reads it as permission
///
/// The guest driver answers `-[MTLDevice supportsPrimitiveType:]` by testing
/// bit `type` of the device-info value for
/// [`crate::model::DEVICE_INFO_KEY_PRIMITIVE_TYPE_MASK`], for any `type <= 8`,
/// and falls back to `type < 5` when the key is absent. So the number this
/// device publishes decides which primitive types the guest is permitted to
/// build a draw out of — and a bit set for a type no backend can translate is a
/// draw this device refuses *after* the guest has committed to it.
///
/// The capture that table came from carried `1023`: bits 0..=9, authorising the
/// four non-public types 5..=8 on top of the public enum. Both backends refuse
/// those by name — `translate::raster::primitive_topology` answers
/// `UnknownPrimitiveType` and `backend::metal::mtl_enum::primitive_type` answers
/// `None` — so every one of those bits was a promise this device cannot keep.
/// Narrowing to what it can execute is the rule
/// [`crate::model::device_info_caps`] already applies to the GPU-dependent keys:
/// answering higher than the host can execute does not degrade gracefully.
///
/// Widening it again needs the *meaning* of 5..=8 first. They are not in the
/// public `MTLPrimitiveType` enum and nothing here has decoded one, so setting a
/// bit for one would be a number chosen to match a capture rather than a
/// contract. Each backend's translator carries the test that holds this constant
/// to the arms that actually exist.
pub const EXECUTABLE_PRIMITIVE_TYPES: u32 = 0b1_1111;

/// Whether `mtl` is a primitive type this device advertises and can execute.
#[inline]
pub const fn primitive_type_executable(mtl: u32) -> bool {
    mtl < u32::BITS && (EXECUTABLE_PRIMITIVE_TYPES >> mtl) & 1 == 1
}

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

#[cfg(test)]
mod tests {
    use super::*;

    /// The mask names the public `MTLPrimitiveType` enum and nothing above it.
    ///
    /// Stated here as well as in each backend's translator test because this is
    /// the value that leaves the device: a bit added here without an arm behind
    /// it is a guest draw refused, and the backend tests cannot both run on one
    /// host.
    #[test]
    fn the_advertised_primitive_types_stop_at_the_public_enum() {
        for mtl in 0..5 {
            assert!(primitive_type_executable(mtl), "public type {mtl}");
        }
        for mtl in 5..=8 {
            assert!(
                !primitive_type_executable(mtl),
                "type {mtl} is not in the public enum and no arm decodes it"
            );
        }
        assert!(!primitive_type_executable(u32::MAX), "no shift overflow");
    }
}
