//! Typed failures while materializing a validated Vulkan compute dispatch.
//!
//! These checks protect persistent storage-image residency. They are later
//! than `ComputeValidationDecline`: the request is structurally valid, but the
//! resident snapshot observed at execution no longer matches what the runtime
//! staged.

use super::types::StorageImageFormat;
use crate::model::ComputeStorageResidencyKey;
use crate::observe::Decline;

/// A specific failure while preparing a validated compute dispatch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ComputeExecutionDecline {
    ResidentSampleAbsent {
        binding: u32,
        identity: ComputeStorageResidencyKey,
        width: u32,
        height: u32,
    },
    ResidentSampleGenerationMismatch {
        binding: u32,
        identity: ComputeStorageResidencyKey,
        actual_generation: u32,
        expected_generation: u32,
    },
    ResidentSampleByteShapeMismatch {
        binding: u32,
        identity: ComputeStorageResidencyKey,
        source_width: u32,
        source_height: u32,
        source_format: StorageImageFormat,
        source_row_bytes: u64,
        resource_width: u32,
        resource_height: u32,
        resource_format: StorageImageFormat,
        resource_row_bytes: u64,
    },
    ResidentSampleSourceLayersUnsupported {
        binding: u32,
        identity: ComputeStorageResidencyKey,
        layers: u32,
    },
    ResidentSampleResourceLayersUnsupported {
        binding: u32,
        identity: ComputeStorageResidencyKey,
        layers: u32,
    },
    ResidentSampleOneDimUnsupported {
        binding: u32,
        identity: ComputeStorageResidencyKey,
    },
    ResidentSampleArrayedUnsupported {
        binding: u32,
        identity: ComputeStorageResidencyKey,
    },
    ResidentSampleVolumeUnsupported {
        binding: u32,
        identity: ComputeStorageResidencyKey,
    },
    SeedSkippedWithoutResidency {
        binding: u32,
        width: u32,
        height: u32,
    },
    ResidentSeedGenerationLost {
        binding: u32,
        identity: ComputeStorageResidencyKey,
        expected_generation: u32,
    },
    ResidentAllocatorLiveSlotMissing {
        identity: ComputeStorageResidencyKey,
        width: u32,
        height: u32,
        layers: u32,
        format: StorageImageFormat,
        one_dim: bool,
        arrayed: bool,
        volume: bool,
    },
}

impl Decline for ComputeExecutionDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::ResidentSampleAbsent { .. } => "vk_compute_exec_resident_sample_absent",
            Self::ResidentSampleGenerationMismatch { .. } => {
                "vk_compute_exec_resident_sample_generation_mismatch"
            }
            Self::ResidentSampleByteShapeMismatch { .. } => {
                "vk_compute_exec_resident_sample_byte_shape_mismatch"
            }
            Self::ResidentSampleSourceLayersUnsupported { .. } => {
                "vk_compute_exec_resident_sample_source_layers_unsupported"
            }
            Self::ResidentSampleResourceLayersUnsupported { .. } => {
                "vk_compute_exec_resident_sample_resource_layers_unsupported"
            }
            Self::ResidentSampleOneDimUnsupported { .. } => {
                "vk_compute_exec_resident_sample_1d_unsupported"
            }
            Self::ResidentSampleArrayedUnsupported { .. } => {
                "vk_compute_exec_resident_sample_arrayed_unsupported"
            }
            Self::ResidentSampleVolumeUnsupported { .. } => {
                "vk_compute_exec_resident_sample_volume_unsupported"
            }
            Self::SeedSkippedWithoutResidency { .. } => {
                "vk_compute_exec_seed_skipped_without_residency"
            }
            Self::ResidentSeedGenerationLost { .. } => {
                "vk_compute_exec_resident_seed_generation_lost"
            }
            Self::ResidentAllocatorLiveSlotMissing { .. } => {
                "vk_compute_exec_resident_allocator_live_slot_missing"
            }
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::ResidentSampleAbsent {
                binding,
                identity,
                width,
                height,
            } => {
                let mut fields = binding_identity_fields(*binding, identity);
                fields.extend([
                    ("resource_width", width.to_string()),
                    ("resource_height", height.to_string()),
                ]);
                fields
            }
            Self::ResidentSampleGenerationMismatch {
                binding,
                identity,
                actual_generation,
                expected_generation,
            } => {
                let mut fields = binding_identity_fields(*binding, identity);
                fields.extend([
                    ("actual_generation", actual_generation.to_string()),
                    ("expected_generation", expected_generation.to_string()),
                ]);
                fields
            }
            Self::ResidentSampleByteShapeMismatch {
                binding,
                identity,
                source_width,
                source_height,
                source_format,
                source_row_bytes,
                resource_width,
                resource_height,
                resource_format,
                resource_row_bytes,
            } => {
                let mut fields = binding_identity_fields(*binding, identity);
                fields.extend([
                    ("source_width", source_width.to_string()),
                    ("source_height", source_height.to_string()),
                    ("source_format", format!("{source_format:?}")),
                    ("source_row_bytes", source_row_bytes.to_string()),
                    ("resource_width", resource_width.to_string()),
                    ("resource_height", resource_height.to_string()),
                    ("resource_format", format!("{resource_format:?}")),
                    ("resource_row_bytes", resource_row_bytes.to_string()),
                ]);
                fields
            }
            Self::ResidentSampleSourceLayersUnsupported {
                binding,
                identity,
                layers,
            }
            | Self::ResidentSampleResourceLayersUnsupported {
                binding,
                identity,
                layers,
            } => {
                let mut fields = binding_identity_fields(*binding, identity);
                fields.push(("layers", layers.to_string()));
                fields
            }
            Self::ResidentSampleOneDimUnsupported { binding, identity }
            | Self::ResidentSampleArrayedUnsupported { binding, identity }
            | Self::ResidentSampleVolumeUnsupported { binding, identity } => {
                binding_identity_fields(*binding, identity)
            }
            Self::SeedSkippedWithoutResidency {
                binding,
                width,
                height,
            } => vec![
                ("binding", binding.to_string()),
                ("resource_width", width.to_string()),
                ("resource_height", height.to_string()),
            ],
            Self::ResidentSeedGenerationLost {
                binding,
                identity,
                expected_generation,
            } => {
                let mut fields = binding_identity_fields(*binding, identity);
                fields.push(("expected_generation", expected_generation.to_string()));
                fields
            }
            Self::ResidentAllocatorLiveSlotMissing {
                identity,
                width,
                height,
                layers,
                format,
                one_dim,
                arrayed,
                volume,
            } => {
                let mut fields = residency_fields(identity);
                fields.extend([
                    ("resource_width", width.to_string()),
                    ("resource_height", height.to_string()),
                    ("layers", layers.to_string()),
                    ("format", format!("{format:?}")),
                    ("one_dim", one_dim.to_string()),
                    ("arrayed", arrayed.to_string()),
                    ("volume", volume.to_string()),
                ]);
                fields
            }
        }
    }
}

fn binding_identity_fields(
    binding: u32,
    identity: &ComputeStorageResidencyKey,
) -> Vec<(&'static str, String)> {
    let mut fields = vec![("binding", binding.to_string())];
    fields.extend(residency_fields(identity));
    fields
}

pub(super) fn residency_fields(
    identity: &ComputeStorageResidencyKey,
) -> Vec<(&'static str, String)> {
    vec![
        ("residency_mapping_id", identity.mapping_id.to_string()),
        (
            "residency_map_generation",
            identity.map_generation.to_string(),
        ),
        (
            "residency_surface_offset",
            format!("{:#x}", identity.surface_offset),
        ),
        ("residency_surface_bpr", identity.surface_bpr.to_string()),
        ("residency_span_end", identity.span_end.to_string()),
        ("residency_width", identity.width.to_string()),
        ("residency_height", identity.height.to_string()),
        ("residency_pixel_format", identity.pixel_format.to_string()),
        ("residency_texture_ref", identity.texture_ref.to_string()),
    ]
}

crate::observe::decline_display!(ComputeExecutionDecline);

#[cfg(test)]
mod tests {
    use super::*;

    fn identity() -> ComputeStorageResidencyKey {
        ComputeStorageResidencyKey {
            mapping_id: 7,
            map_generation: 8,
            surface_offset: 0x9000,
            surface_bpr: 256,
            span_end: 4096,
            width: 64,
            height: 32,
            pixel_format: 80,
            texture_ref: 11,
        }
    }

    fn all() -> Vec<ComputeExecutionDecline> {
        vec![
            ComputeExecutionDecline::ResidentSampleAbsent {
                binding: 32,
                identity: identity(),
                width: 64,
                height: 32,
            },
            ComputeExecutionDecline::ResidentSampleGenerationMismatch {
                binding: 32,
                identity: identity(),
                actual_generation: 8,
                expected_generation: 9,
            },
            ComputeExecutionDecline::ResidentSampleByteShapeMismatch {
                binding: 32,
                identity: identity(),
                source_width: 64,
                source_height: 32,
                source_format: StorageImageFormat::Rgba8Unorm,
                source_row_bytes: 256,
                resource_width: 32,
                resource_height: 32,
                resource_format: StorageImageFormat::Rgba8Unorm,
                resource_row_bytes: 128,
            },
            ComputeExecutionDecline::ResidentSampleSourceLayersUnsupported {
                binding: 32,
                identity: identity(),
                layers: 2,
            },
            ComputeExecutionDecline::ResidentSampleResourceLayersUnsupported {
                binding: 32,
                identity: identity(),
                layers: 2,
            },
            ComputeExecutionDecline::ResidentSampleOneDimUnsupported {
                binding: 32,
                identity: identity(),
            },
            ComputeExecutionDecline::ResidentSampleArrayedUnsupported {
                binding: 32,
                identity: identity(),
            },
            ComputeExecutionDecline::ResidentSampleVolumeUnsupported {
                binding: 32,
                identity: identity(),
            },
            ComputeExecutionDecline::SeedSkippedWithoutResidency {
                binding: 34,
                width: 64,
                height: 32,
            },
            ComputeExecutionDecline::ResidentSeedGenerationLost {
                binding: 34,
                identity: identity(),
                expected_generation: 9,
            },
            ComputeExecutionDecline::ResidentAllocatorLiveSlotMissing {
                identity: identity(),
                width: 64,
                height: 32,
                layers: 1,
                format: StorageImageFormat::Rgba8Unorm,
                one_dim: false,
                arrayed: false,
                volume: false,
            },
        ]
    }

    #[test]
    fn every_compute_execution_check_has_a_unique_log_safe_slug() {
        let mut slugs: Vec<_> = all().iter().map(Decline::slug).collect();
        for slug in &slugs {
            assert!(slug.starts_with("vk_compute_exec_"), "{slug}");
            assert!(
                slug.bytes()
                    .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_'),
                "{slug}"
            );
        }
        slugs.sort_unstable();
        let before = slugs.len();
        slugs.dedup();
        // Down from 23: twelve `direct_writeback_*` checks went out with the
        // GPU-direct compute writeback. Every one of them validated the shape
        // of a caller-supplied guest window the dispatch would DMA into —
        // alignment, row stride, offset, overflow, window length — and none of
        // them has anything left to validate now that the copy always lands in
        // a pooled readback the runtime owns.
        assert_eq!(before, 11, "the compute executor's reason census moved");
        assert_eq!(before, slugs.len(), "duplicate compute-execution slug");
    }

    #[test]
    fn compute_execution_fields_are_structured_and_log_safe() {
        for decline in all() {
            let line = crate::observe::Emit::decline("compute_execution_test", &decline).render();
            assert!(line.starts_with(&format!("compute_execution_test reason={}", decline.slug())));
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
    fn residency_fields_preserve_every_identity_component() {
        assert_eq!(
            residency_fields(&identity()),
            vec![
                ("residency_mapping_id", "7".into()),
                ("residency_map_generation", "8".into()),
                ("residency_surface_offset", "0x9000".into()),
                ("residency_surface_bpr", "256".into()),
                ("residency_span_end", "4096".into()),
                ("residency_width", "64".into()),
                ("residency_height", "32".into()),
                ("residency_pixel_format", "80".into()),
                ("residency_texture_ref", "11".into()),
            ]
        );
    }
}
