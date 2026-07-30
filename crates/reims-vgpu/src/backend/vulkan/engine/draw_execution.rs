//! Typed failures while materializing a validated Vulkan draw request.
//!
//! Request-shape checks belong to `DrawValidationDecline`. These reasons are
//! later failures: resident state disagreed with the validated request, a
//! constant-step vertex bind could not be shifted, or a tracked image layout
//! could not be used as a transfer source.
//!
//! Two variants used to head this list — `BufferGuestRunImportMissing` and
//! `SampledGuestRunImportMissing`, both meaning "an imported guest span
//! disappeared between the runtime's pre-check and the bind". Neither can
//! happen now: guest runs are gathered by the CPU out of the mapped span, so
//! there is no import to lose.

use ash::vk;

use super::types::TargetIdentity;
use crate::observe::Decline;

/// A specific failure while preparing or executing a validated draw.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DrawExecutionDecline {
    ConstantVertexRequiresCpuBytes {
        location: u32,
    },
    ConstantVertexBaseInstanceOverflow {
        base_instance: u32,
        stride: u32,
    },
    ConstantVertexAllocationOverflow {
        prefix: usize,
        bytes_len: usize,
    },
    LoadTargetContentNotReady {
        identity: TargetIdentity,
    },
    SeedResidentMissing {
        identity: TargetIdentity,
    },
    SeedResidentNotReady {
        identity: TargetIdentity,
    },
    SeedGeometryMismatch {
        identity: TargetIdentity,
        resident_width: u32,
        resident_height: u32,
        draw_width: u32,
        draw_height: u32,
    },
    SeedFormatMismatch {
        identity: TargetIdentity,
        resident_bgra: bool,
        draw_bgra: bool,
    },
    SampledResidentMissing {
        binding: u32,
        identity: TargetIdentity,
    },
    SampledResidentNotReady {
        binding: u32,
        identity: TargetIdentity,
    },
    SampledResidentGeometryMismatch {
        binding: u32,
        identity: TargetIdentity,
        resident_width: u32,
        resident_height: u32,
        resource_width: u32,
        resource_height: u32,
    },
    UnsupportedTrackedLayout {
        layout: vk::ImageLayout,
    },
}

impl Decline for DrawExecutionDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::ConstantVertexRequiresCpuBytes { .. } => {
                "vk_draw_exec_constant_vertex_requires_cpu_bytes"
            }
            Self::ConstantVertexBaseInstanceOverflow { .. } => {
                "vk_draw_exec_constant_vertex_base_instance_overflow"
            }
            Self::ConstantVertexAllocationOverflow { .. } => {
                "vk_draw_exec_constant_vertex_allocation_overflow"
            }
            Self::LoadTargetContentNotReady { .. } => "vk_draw_exec_load_target_content_not_ready",
            Self::SeedResidentMissing { .. } => "vk_draw_exec_seed_resident_missing",
            Self::SeedResidentNotReady { .. } => "vk_draw_exec_seed_resident_not_ready",
            Self::SeedGeometryMismatch { .. } => "vk_draw_exec_seed_geometry_mismatch",
            Self::SeedFormatMismatch { .. } => "vk_draw_exec_seed_format_mismatch",
            Self::SampledResidentMissing { .. } => "vk_draw_exec_sampled_resident_missing",
            Self::SampledResidentNotReady { .. } => "vk_draw_exec_sampled_resident_not_ready",
            Self::SampledResidentGeometryMismatch { .. } => {
                "vk_draw_exec_sampled_resident_geometry_mismatch"
            }
            Self::UnsupportedTrackedLayout { .. } => "vk_draw_exec_unsupported_tracked_layout",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::ConstantVertexRequiresCpuBytes { location } => {
                vec![("location", location.to_string())]
            }
            Self::ConstantVertexBaseInstanceOverflow {
                base_instance,
                stride,
            } => vec![
                ("base_instance", base_instance.to_string()),
                ("stride", stride.to_string()),
            ],
            Self::ConstantVertexAllocationOverflow { prefix, bytes_len } => vec![
                ("prefix", prefix.to_string()),
                ("bytes_len", bytes_len.to_string()),
            ],
            Self::LoadTargetContentNotReady { identity }
            | Self::SeedResidentMissing { identity }
            | Self::SeedResidentNotReady { identity } => identity_fields(identity),
            Self::SeedGeometryMismatch {
                identity,
                resident_width,
                resident_height,
                draw_width,
                draw_height,
            } => {
                let mut fields = identity_fields(identity);
                fields.extend([
                    ("resident_width", resident_width.to_string()),
                    ("resident_height", resident_height.to_string()),
                    ("draw_width", draw_width.to_string()),
                    ("draw_height", draw_height.to_string()),
                ]);
                fields
            }
            Self::SeedFormatMismatch {
                identity,
                resident_bgra,
                draw_bgra,
            } => {
                let mut fields = identity_fields(identity);
                fields.extend([
                    ("resident_bgra", resident_bgra.to_string()),
                    ("draw_bgra", draw_bgra.to_string()),
                ]);
                fields
            }
            Self::SampledResidentMissing { binding, identity }
            | Self::SampledResidentNotReady { binding, identity } => {
                let mut fields = vec![("binding", binding.to_string())];
                fields.extend(identity_fields(identity));
                fields
            }
            Self::SampledResidentGeometryMismatch {
                binding,
                identity,
                resident_width,
                resident_height,
                resource_width,
                resource_height,
            } => {
                let mut fields = vec![("binding", binding.to_string())];
                fields.extend(identity_fields(identity));
                fields.extend([
                    ("resident_width", resident_width.to_string()),
                    ("resident_height", resident_height.to_string()),
                    ("resource_width", resource_width.to_string()),
                    ("resource_height", resource_height.to_string()),
                ]);
                fields
            }
            Self::UnsupportedTrackedLayout { layout } => {
                vec![("layout", format!("{layout:?}"))]
            }
        }
    }
}

pub(super) fn identity_fields(identity: &TargetIdentity) -> Vec<(&'static str, String)> {
    match identity {
        TargetIdentity::Surface {
            id,
            width,
            height,
            generation,
        } => vec![
            ("identity_kind", "surface".into()),
            ("identity_id", id.to_string()),
            ("identity_width", width.to_string()),
            ("identity_height", height.to_string()),
            ("identity_generation", generation.to_string()),
        ],
        TargetIdentity::Texture {
            ref_,
            width,
            height,
            generation,
        } => vec![
            ("identity_kind", "texture".into()),
            ("identity_ref", ref_.to_string()),
            ("identity_width", width.to_string()),
            ("identity_height", height.to_string()),
            ("identity_generation", generation.to_string()),
        ],
        TargetIdentity::Gva {
            gva,
            width,
            height,
            generation,
        } => vec![
            ("identity_kind", "gva".into()),
            ("identity_gva", format!("{gva:#x}")),
            ("identity_width", width.to_string()),
            ("identity_height", height.to_string()),
            ("identity_generation", generation.to_string()),
        ],
        TargetIdentity::Anonymous { slot } => vec![
            ("identity_kind", "anonymous".into()),
            ("identity_slot", slot.to_string()),
        ],
    }
}

crate::observe::decline_display!(DrawExecutionDecline);

#[cfg(test)]
mod tests {
    use super::*;

    fn identity() -> TargetIdentity {
        TargetIdentity::Surface {
            id: 7,
            width: 64,
            height: 32,
            generation: 9,
        }
    }

    fn all() -> Vec<DrawExecutionDecline> {
        vec![
            DrawExecutionDecline::ConstantVertexRequiresCpuBytes { location: 2 },
            DrawExecutionDecline::ConstantVertexBaseInstanceOverflow {
                base_instance: u32::MAX,
                stride: u32::MAX,
            },
            DrawExecutionDecline::ConstantVertexAllocationOverflow {
                prefix: usize::MAX,
                bytes_len: 1,
            },
            DrawExecutionDecline::LoadTargetContentNotReady {
                identity: identity(),
            },
            DrawExecutionDecline::SeedResidentMissing {
                identity: identity(),
            },
            DrawExecutionDecline::SeedResidentNotReady {
                identity: identity(),
            },
            DrawExecutionDecline::SeedGeometryMismatch {
                identity: identity(),
                resident_width: 32,
                resident_height: 16,
                draw_width: 64,
                draw_height: 32,
            },
            DrawExecutionDecline::SeedFormatMismatch {
                identity: identity(),
                resident_bgra: false,
                draw_bgra: true,
            },
            DrawExecutionDecline::SampledResidentMissing {
                binding: 32,
                identity: identity(),
            },
            DrawExecutionDecline::SampledResidentNotReady {
                binding: 32,
                identity: identity(),
            },
            DrawExecutionDecline::SampledResidentGeometryMismatch {
                binding: 32,
                identity: identity(),
                resident_width: 32,
                resident_height: 16,
                resource_width: 64,
                resource_height: 32,
            },
            DrawExecutionDecline::UnsupportedTrackedLayout {
                layout: vk::ImageLayout::UNDEFINED,
            },
        ]
    }

    #[test]
    fn every_draw_execution_check_has_a_unique_log_safe_slug() {
        let mut slugs: Vec<_> = all().iter().map(Decline::slug).collect();
        for slug in &slugs {
            assert!(slug.starts_with("vk_draw_exec_"), "{slug}");
            assert!(
                slug.bytes()
                    .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_'),
                "{slug}"
            );
        }
        slugs.sort_unstable();
        let before = slugs.len();
        slugs.dedup();
        // Down from 14: the two `*_guest_run_import_missing` checks went with
        // the host-pointer import. Both meant "an imported guest span vanished
        // between the runtime's pre-check and the bind", and guest runs are now
        // gathered by the CPU out of the mapped span, so there is no import to
        // lose and no refusal to make.
        assert_eq!(before, 12, "the draw executor's reason census moved");
        assert_eq!(before, slugs.len(), "duplicate draw-execution slug");
    }

    #[test]
    fn draw_execution_fields_are_structured_and_log_safe() {
        for decline in all() {
            let line = crate::observe::Emit::decline("draw_execution_test", &decline).render();
            assert!(line.starts_with(&format!("draw_execution_test reason={}", decline.slug())));
            for field in line.split(' ').skip(1) {
                assert!(!field.is_empty(), "empty field in {line:?}");
                assert!(
                    !field.contains(char::is_whitespace),
                    "non-token field in {line:?}"
                );
            }
        }
    }

    #[test]
    fn identity_fields_preserve_each_protocol_namespace() {
        let cases = [
            (
                TargetIdentity::Surface {
                    id: 7,
                    width: 80,
                    height: 60,
                    generation: 11,
                },
                "surface",
                ("identity_id", "7"),
            ),
            (
                TargetIdentity::Texture {
                    ref_: 8,
                    width: 80,
                    height: 60,
                    generation: 11,
                },
                "texture",
                ("identity_ref", "8"),
            ),
            (
                TargetIdentity::Gva {
                    gva: 0x1234,
                    width: 80,
                    height: 60,
                    generation: 11,
                },
                "gva",
                ("identity_gva", "0x1234"),
            ),
            (
                TargetIdentity::Anonymous { slot: 9 },
                "anonymous",
                ("identity_slot", "9"),
            ),
        ];

        for (identity, expected_kind, expected_key) in cases {
            let fields = DrawExecutionDecline::SeedResidentMissing { identity }.fields();
            assert!(fields.contains(&("identity_kind", expected_kind.into())));
            assert!(fields.contains(&(expected_key.0, expected_key.1.into())));
        }
    }
}
