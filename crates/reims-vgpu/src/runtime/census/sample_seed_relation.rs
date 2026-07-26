//! Measure exact channel exchange at the sampled-input/attachment-Load boundary.
//!
//! This diagnostic never selects a resource or changes draw behavior. It keeps
//! one semantic-RGBA center row per same-geometry linear sample and stays quiet
//! unless that row is an exact red/blue exchange of the attachment Load row.

use crate::contract::pixel_format::{self, MTL_FORMAT_BGRA8_UNORM, RGBA8_BPP};
use crate::model::DeviceState;
use crate::observe;
use crate::runtime::host::{HostMemory, HostOps};
use crate::runtime::{mapper, mapping_write};

/// One semantic-RGBA center row from a same-geometry linear compositor input.
pub(crate) struct SampleRow {
    pub index: u32,
    pub texture_ref: u32,
    pub object_type: u8,
    pub pixel_format: u16,
    pub frag_stage: bool,
    pub rgba: Vec<u8>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct RelationStats {
    changed_px: usize,
    same_px: usize,
    rb_swap_px: usize,
}

fn relation_stats(sample_rgba: &[u8], seed_rgba: &[u8]) -> RelationStats {
    let mut stats = RelationStats::default();
    for (sample, seed) in sample_rgba.chunks_exact(4).zip(seed_rgba.chunks_exact(4)) {
        if sample == seed {
            stats.same_px += 1;
        } else {
            stats.changed_px += 1;
        }
        // Distinct R/B avoids counting gray pixels as an exchange.
        if sample[0] != sample[2]
            && seed[0] == sample[2]
            && seed[1] == sample[1]
            && seed[2] == sample[0]
            && seed[3] == sample[3]
        {
            stats.rb_swap_px += 1;
        }
    }
    stats
}

pub(crate) fn center_row(rgba: &[u8], width: u32, height: u32) -> Option<&[u8]> {
    let row_bytes = (width as usize).checked_mul(RGBA8_BPP as usize)?;
    let start = (height as usize / 2).checked_mul(row_bytes)?;
    let end = start.checked_add(row_bytes)?;
    rgba.get(start..end)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn log_exact_relations(
    pipeline_ref: u32,
    target_mid: u32,
    target_format: u16,
    width: u32,
    height: u32,
    seed_src: &str,
    samples: &[SampleRow],
    seed_row: &[u8],
) {
    for sample in samples {
        if sample.rgba.len() != seed_row.len() {
            continue;
        }
        let stats = relation_stats(&sample.rgba, seed_row);
        if stats.rb_swap_px == 0 {
            continue;
        }
        observe::off(format!(
            "sample_seed_relation pipe={pipeline_ref} target_mid={target_mid} target_fmt={target_format:#x} {width}x{height} seed_src={seed_src} stage={} i={} ref={} type={} sample_fmt={:#x} changed_row_px={} same_row_px={} rb_swap_row_px={}",
            if sample.frag_stage { "frag" } else { "vert" },
            sample.index,
            sample.texture_ref,
            sample.object_type,
            sample.pixel_format,
            stats.changed_px,
            stats.same_px,
            stats.rb_swap_px
        ));
    }
}

/// Read one semantic-RGBA center row from a guest-visible type-11 Store mirror.
///
/// A ready Vulkan target is attachment-Load authority, but a successful Store
/// scatters those exact BGRA bytes into the mapping before its stamp completes.
/// This reads one row, never a GPU image or full frame.
pub(crate) fn load_type11_center_row_rgba<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mapping_id: u32,
    width: u32,
    height: u32,
    format: u16,
) -> Option<Vec<u8>> {
    if width == 0 || height == 0 {
        return None;
    }
    let format = if format != 0 {
        format
    } else {
        MTL_FORMAT_BGRA8_UNORM
    };
    let _ = mapper::ensure_resolved_for_scanout(state, host, mapping_id);
    let (base_off, bpr) = {
        let mapping = state.mappings.get(&mapping_id)?;
        let (base_off, bpr, _) =
            mapping_write::type11_sample_window(mapping, width, height, format)?;
        (base_off, bpr)
    };
    let native_len = pixel_format::tight_row_bytes(width, format)? as usize;
    let row_off = base_off.checked_add((height as u64 / 2).checked_mul(bpr as u64)?)?;
    let mut native = vec![0u8; native_len];
    if !mapper::read_mapping_bytes(state, host, mapping_id, row_off, &mut native) {
        return None;
    }
    let mut rgba = vec![0u8; (width as usize).checked_mul(RGBA8_BPP as usize)?];
    pixel_format::convert_row_to_rgba8(format, &native, width, &mut rgba).then_some(rgba)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::decode::resource::OBJECT_TYPE_TEXTURE_VARIANT;

    #[test]
    fn distinguishes_exact_rb_exchange() {
        let sample = [
            10, 20, 30, 255, // exact R/B exchange below
            40, 50, 60, 255, // unchanged
            70, 80, 90, 200, // arbitrary change
            33, 44, 33, 255, // gray: must not count as an exchange
        ];
        let seed = [
            30, 20, 10, 255, 40, 50, 60, 255, 1, 2, 3, 4, 33, 44, 33, 255,
        ];
        assert_eq!(
            relation_stats(&sample, &seed),
            RelationStats {
                changed_px: 2,
                same_px: 2,
                rb_swap_px: 1,
            }
        );
        assert_eq!(center_row(&sample, 2, 2), Some(&sample[8..16]));
        assert_eq!(center_row(&sample[..15], 2, 2), None);
    }

    #[test]
    fn proxy_is_fail_visible_and_consistent_control_is_quiet() {
        let pipeline_ref = 0xf000_0000u32.wrapping_add(std::process::id());
        let seed = [30, 20, 10, 255, 40, 50, 60, 255];
        let sample = SampleRow {
            index: 3,
            texture_ref: 222,
            object_type: OBJECT_TYPE_TEXTURE_VARIANT,
            pixel_format: MTL_FORMAT_BGRA8_UNORM,
            frag_stage: true,
            rgba: vec![10, 20, 30, 255, 40, 50, 60, 255],
        };
        log_exact_relations(
            pipeline_ref,
            1,
            MTL_FORMAT_BGRA8_UNORM,
            2,
            1,
            "guest_pages",
            &[sample],
            &seed,
        );
        let marker = format!(
            "OFF sample_seed_relation pipe={pipeline_ref} target_mid=1 target_fmt=0x50 2x1 seed_src=guest_pages stage=frag i=3 ref=222 type=3 sample_fmt=0x50 changed_row_px=1 same_row_px=1 rb_swap_row_px=1"
        );
        let body = std::fs::read_to_string(crate::observe::fail_log_path())
            .expect("reims-vgpu-fail.log readable");
        assert!(body.lines().any(|line| line.starts_with(&marker)));

        let quiet_pipeline = pipeline_ref.wrapping_add(1);
        let same = SampleRow {
            index: 3,
            texture_ref: 222,
            object_type: OBJECT_TYPE_TEXTURE_VARIANT,
            pixel_format: MTL_FORMAT_BGRA8_UNORM,
            frag_stage: true,
            rgba: seed.to_vec(),
        };
        log_exact_relations(
            quiet_pipeline,
            1,
            MTL_FORMAT_BGRA8_UNORM,
            2,
            1,
            "resident_guest_mirror",
            &[same],
            &seed,
        );
        let body = std::fs::read_to_string(crate::observe::fail_log_path())
            .expect("reims-vgpu-fail.log readable");
        assert!(!body
            .lines()
            .any(|line| line.contains(&format!("sample_seed_relation pipe={quiet_pipeline} "))));
    }
}
