//! Typed validation failures for a Vulkan compute request.
//!
//! Validation runs before context creation or GPU work. The old rail collapsed
//! seventeen request invariants into `DrawError::Invalid(String)`, including
//! four descriptor-role checks with identical prose.

use crate::observe::Decline;

/// A specific malformed or internally inconsistent compute request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ComputeValidationDecline {
    EmptySpirv,
    EmptyEntry,
    EntryInteriorNul,
    ZeroGrid {
        grid: [u32; 3],
    },
    DuplicateStorageBufferBinding {
        binding: u32,
    },
    EmptyStorageBuffer {
        binding: u32,
    },
    DuplicateSampledImageBinding {
        binding: u32,
    },
    SampledZeroGeometry {
        binding: u32,
        width: u32,
        height: u32,
        layers: u32,
    },
    SampledOneDimHeight {
        binding: u32,
        height: u32,
    },
    SampledNonArrayLayers {
        binding: u32,
        layers: u32,
    },
    SampledBytesLength {
        binding: u32,
        actual: usize,
        expected: usize,
    },
    InvalidSamplerLod {
        binding: u32,
        lod_min_bits: u32,
        lod_max_bits: u32,
    },
    DuplicateSamplerBinding {
        binding: u32,
    },
    DuplicateStorageImageBinding {
        binding: u32,
    },
    StorageZeroGeometry {
        binding: u32,
        width: u32,
        height: u32,
        layers: u32,
    },
    StorageOneDimHeight {
        binding: u32,
        height: u32,
    },
    StorageNonArrayLayers {
        binding: u32,
        layers: u32,
    },
    StorageBytesLength {
        binding: u32,
        actual: usize,
        expected: usize,
    },
}

impl Decline for ComputeValidationDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::EmptySpirv => "vk_compute_validate_empty_spirv",
            Self::EmptyEntry => "vk_compute_validate_empty_entry",
            Self::EntryInteriorNul => "vk_compute_validate_entry_interior_nul",
            Self::ZeroGrid { .. } => "vk_compute_validate_zero_grid",
            Self::DuplicateStorageBufferBinding { .. } => {
                "vk_compute_validate_duplicate_storage_buffer_binding"
            }
            Self::EmptyStorageBuffer { .. } => "vk_compute_validate_empty_storage_buffer",
            Self::DuplicateSampledImageBinding { .. } => {
                "vk_compute_validate_duplicate_sampled_image_binding"
            }
            Self::SampledZeroGeometry { .. } => "vk_compute_validate_sampled_zero_geometry",
            Self::SampledOneDimHeight { .. } => "vk_compute_validate_sampled_1d_height",
            Self::SampledNonArrayLayers { .. } => "vk_compute_validate_sampled_nonarray_layers",
            Self::SampledBytesLength { .. } => "vk_compute_validate_sampled_bytes_length",
            Self::InvalidSamplerLod { .. } => "vk_compute_validate_invalid_sampler_lod",
            Self::DuplicateSamplerBinding { .. } => "vk_compute_validate_duplicate_sampler_binding",
            Self::DuplicateStorageImageBinding { .. } => {
                "vk_compute_validate_duplicate_storage_image_binding"
            }
            Self::StorageZeroGeometry { .. } => "vk_compute_validate_storage_zero_geometry",
            Self::StorageOneDimHeight { .. } => "vk_compute_validate_storage_1d_height",
            Self::StorageNonArrayLayers { .. } => "vk_compute_validate_storage_nonarray_layers",
            Self::StorageBytesLength { .. } => "vk_compute_validate_storage_bytes_length",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::ZeroGrid { grid } => vec![
                ("grid_x", grid[0].to_string()),
                ("grid_y", grid[1].to_string()),
                ("grid_z", grid[2].to_string()),
            ],
            Self::DuplicateStorageBufferBinding { binding }
            | Self::EmptyStorageBuffer { binding }
            | Self::DuplicateSampledImageBinding { binding }
            | Self::DuplicateSamplerBinding { binding }
            | Self::DuplicateStorageImageBinding { binding } => {
                vec![("binding", binding.to_string())]
            }
            Self::SampledZeroGeometry {
                binding,
                width,
                height,
                layers,
            }
            | Self::StorageZeroGeometry {
                binding,
                width,
                height,
                layers,
            } => vec![
                ("binding", binding.to_string()),
                ("width", width.to_string()),
                ("height", height.to_string()),
                ("layers", layers.to_string()),
            ],
            Self::SampledOneDimHeight { binding, height }
            | Self::StorageOneDimHeight { binding, height } => vec![
                ("binding", binding.to_string()),
                ("height", height.to_string()),
            ],
            Self::SampledNonArrayLayers { binding, layers }
            | Self::StorageNonArrayLayers { binding, layers } => vec![
                ("binding", binding.to_string()),
                ("layers", layers.to_string()),
            ],
            Self::SampledBytesLength {
                binding,
                actual,
                expected,
            }
            | Self::StorageBytesLength {
                binding,
                actual,
                expected,
            } => vec![
                ("binding", binding.to_string()),
                ("actual", actual.to_string()),
                ("expected", expected.to_string()),
            ],
            Self::InvalidSamplerLod {
                binding,
                lod_min_bits,
                lod_max_bits,
            } => vec![
                ("binding", binding.to_string()),
                ("lod_min", f32::from_bits(*lod_min_bits).to_string()),
                ("lod_max", f32::from_bits(*lod_max_bits).to_string()),
            ],
            Self::EmptySpirv | Self::EmptyEntry | Self::EntryInteriorNul => Vec::new(),
        }
    }
}

impl std::fmt::Display for ComputeValidationDecline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "reason={}", self.slug())?;
        for (key, value) in self.fields() {
            write!(f, " {key}={value}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all() -> Vec<ComputeValidationDecline> {
        vec![
            ComputeValidationDecline::EmptySpirv,
            ComputeValidationDecline::EmptyEntry,
            ComputeValidationDecline::EntryInteriorNul,
            ComputeValidationDecline::ZeroGrid { grid: [1, 0, 1] },
            ComputeValidationDecline::DuplicateStorageBufferBinding { binding: 0 },
            ComputeValidationDecline::EmptyStorageBuffer { binding: 0 },
            ComputeValidationDecline::DuplicateSampledImageBinding { binding: 32 },
            ComputeValidationDecline::SampledZeroGeometry {
                binding: 32,
                width: 0,
                height: 1,
                layers: 1,
            },
            ComputeValidationDecline::SampledOneDimHeight {
                binding: 32,
                height: 2,
            },
            ComputeValidationDecline::SampledNonArrayLayers {
                binding: 32,
                layers: 2,
            },
            ComputeValidationDecline::SampledBytesLength {
                binding: 32,
                actual: 3,
                expected: 4,
            },
            ComputeValidationDecline::InvalidSamplerLod {
                binding: 64,
                lod_min_bits: 2.0f32.to_bits(),
                lod_max_bits: 1.0f32.to_bits(),
            },
            ComputeValidationDecline::DuplicateSamplerBinding { binding: 64 },
            ComputeValidationDecline::DuplicateStorageImageBinding { binding: 34 },
            ComputeValidationDecline::StorageZeroGeometry {
                binding: 34,
                width: 1,
                height: 0,
                layers: 1,
            },
            ComputeValidationDecline::StorageOneDimHeight {
                binding: 34,
                height: 2,
            },
            ComputeValidationDecline::StorageNonArrayLayers {
                binding: 34,
                layers: 2,
            },
            ComputeValidationDecline::StorageBytesLength {
                binding: 34,
                actual: 3,
                expected: 4,
            },
        ]
    }

    #[test]
    fn every_compute_validation_check_has_a_unique_log_safe_slug() {
        let mut slugs: Vec<_> = all().iter().map(Decline::slug).collect();
        for slug in &slugs {
            assert!(slug.starts_with("vk_compute_validate_"), "{slug}");
            assert!(
                slug.bytes()
                    .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_'),
                "{slug}"
            );
        }
        slugs.sort_unstable();
        let before = slugs.len();
        slugs.dedup();
        assert_eq!(before, 18, "the compute validator's reason census moved");
        assert_eq!(before, slugs.len(), "duplicate compute-validation slug");
    }

    #[test]
    fn compute_validation_fields_are_structured_and_log_safe() {
        for decline in all() {
            let line = crate::observe::Emit::decline("compute_validation_test", &decline).render();
            assert!(line.starts_with(&format!(
                "compute_validation_test reason={}",
                decline.slug()
            )));
            for field in line.split(' ').skip(1) {
                assert!(!field.is_empty(), "empty field in {line:?}");
                assert!(
                    !field.contains(char::is_whitespace),
                    "non-token field in {line:?}"
                );
            }
        }
    }
}
