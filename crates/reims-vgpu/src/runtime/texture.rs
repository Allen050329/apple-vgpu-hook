//! Type-11 / texture geometry registration onto the IOSurface mapping table.
//!
//! Archive: `apple_pv_gpu_register_type11_texture` latches width/height/format
//! on the mapping when a texture descriptor is resolved. Device-desc geom
//! (mapper path) and texture-path geom share [`DeviceState::set_mapping_geom`].

use crate::contract::iosurface_pages::{decode_texture_descriptor, TYPE11_DESC_MIN_LEN};
use crate::model::{is_mapping_id, DeviceState};
use crate::runtime::decode::resource::{decode_descriptor, Descriptor};

/// Register geometry from a decoded type-11 / IOSurface texture descriptor.
pub fn register_type11_geom(
    state: &mut DeviceState,
    mapping_id: u32,
    width: u32,
    height: u32,
    format: u16,
) -> bool {
    if !is_mapping_id(mapping_id) {
        return false;
    }
    if let Some(m) = state.mappings.get(&mapping_id) {
        if m.has_geom && m.width == width && m.height == height && m.format == format {
            return true;
        }
    }
    state.set_mapping_geom(mapping_id, width, height, format)
}

/// Decode descriptor bytes and, if type-11, latch mapping geometry.
pub fn register_from_descriptor_bytes(
    state: &mut DeviceState,
    object_type: u8,
    desc: &[u8],
) -> bool {
    // Also accept the raw iosurface texture layout without a type byte.
    let headerless = |state: &mut DeviceState| {
        if desc.len() >= TYPE11_DESC_MIN_LEN {
            if let Ok(t) = decode_texture_descriptor(desc) {
                return register_type11_geom(
                    state,
                    t.mapping_id,
                    t.width,
                    t.height,
                    t.pixel_format,
                );
            }
        }
        false
    };
    match decode_descriptor(object_type, desc) {
        Ok(Descriptor::IOSurfaceTexture {
            mapping_id,
            width,
            height,
            pixel_format,
            ..
        }) => register_type11_geom(state, mapping_id, width, height, pixel_format),
        // A buffer, a function, a pipeline: not a refusal. This is called for
        // every object type and only type-11 carries mapping geometry, so the
        // decoder answering "that is a different object" is the normal case and
        // must stay out of the log.
        Ok(_) => headerless(state),
        Err(e) => {
            let recovered = headerless(state);
            if !recovered {
                // Both forms refused, so a type-11 registration was genuinely
                // dropped. `_ => false` used to give this the same silence as
                // the `Ok(_)` arm above, which is the collapse: a malformed
                // descriptor and a buffer object looked identical.
                crate::observe::Emit::decline("type11_register", &e)
                    .field("obj_type", object_type)
                    .field("len", desc.len())
                    .fail_once(object_type as u64);
            }
            recovered
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::contract::endian::st16;
    use crate::contract::endian::st32;
    use crate::model::{DeviceId, PAGE_SHIFT_ARM64E};

    #[test]
    fn type11_geom_and_generation() {
        let mut s = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        assert!(register_type11_geom(&mut s, 5, 640, 480, 0x50));
        let m = s.mappings.get(&5).unwrap();
        assert!(m.has_geom);
        assert_eq!((m.width, m.height, m.format), (640, 480, 0x50));
        assert_eq!(s.mark_mapping_written(5), 1);
        assert_eq!(s.mark_mapping_written(5), 2);

        let mut desc = [0u8; 0x20];
        st32(&mut desc[0..], 9);
        st16(&mut desc[0x16..], 0x73);
        st32(&mut desc[0x18..], 100);
        st32(&mut desc[0x1c..], 50);
        assert!(register_from_descriptor_bytes(&mut s, 11, &desc));
        let m = s.mappings.get(&9).unwrap();
        assert_eq!(m.width, 100);
        assert_eq!(m.height, 50);
        assert_eq!(m.format, 0x73);
    }
}
