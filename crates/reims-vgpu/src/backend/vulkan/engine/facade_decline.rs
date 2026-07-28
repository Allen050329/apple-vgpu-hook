//! Typed failures at the Vulkan engine façade and host-window presenter seam.
//!
//! These checks are neither malformed draw/compute requests nor failed Vulkan
//! calls. They reject an engine entry point because the façade's tracked state
//! disappeared, disagreed with the caller, or could not describe the requested
//! scanout.

use super::compute_execution::residency_fields;
use super::draw_execution::identity_fields;
use super::types::TargetIdentity;
use crate::model::ComputeStorageResidencyKey;
use crate::observe::Decline;

/// A specific engine façade or host-window presenter state failure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EngineFacadeDecline {
    WindowPresenterNotAttached,
    StorageReadResidentAbsent {
        identity: ComputeStorageResidencyKey,
    },
    StorageReadGenerationMismatch {
        identity: ComputeStorageResidencyKey,
        actual_generation: u32,
        expected_generation: u32,
    },
    ExportScanoutZeroGeometry {
        width: u32,
        height: u32,
    },
    ExportScanoutLengthMismatch {
        width: u32,
        height: u32,
        actual_len: usize,
        expected_len: usize,
    },
    ExportPresentUnknownIdentity {
        identity: TargetIdentity,
    },
    ExportPresentNotReady {
        identity: TargetIdentity,
    },
    ScatterPresentUnknownIdentity {
        identity: TargetIdentity,
    },
    ScatterPresentNotReady {
        identity: TargetIdentity,
    },
    WindowSourceDisappearedBeforePin {
        identity: TargetIdentity,
    },
}

impl Decline for EngineFacadeDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::WindowPresenterNotAttached => "vk_engine_window_presenter_not_attached",
            Self::StorageReadResidentAbsent { .. } => "vk_engine_storage_read_resident_absent",
            Self::StorageReadGenerationMismatch { .. } => {
                "vk_engine_storage_read_generation_mismatch"
            }
            Self::ExportScanoutZeroGeometry { .. } => "vk_engine_export_scanout_zero_geometry",
            Self::ExportScanoutLengthMismatch { .. } => "vk_engine_export_scanout_length_mismatch",
            Self::ExportPresentUnknownIdentity { .. } => {
                "vk_engine_export_present_unknown_identity"
            }
            Self::ExportPresentNotReady { .. } => "vk_engine_export_present_not_ready",
            Self::ScatterPresentUnknownIdentity { .. } => {
                "vk_engine_scatter_present_unknown_identity"
            }
            Self::ScatterPresentNotReady { .. } => "vk_engine_scatter_present_not_ready",
            Self::WindowSourceDisappearedBeforePin { .. } => {
                "vk_engine_window_source_disappeared_before_pin"
            }
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::WindowPresenterNotAttached => Vec::new(),
            Self::StorageReadResidentAbsent { identity } => residency_fields(identity),
            Self::StorageReadGenerationMismatch {
                identity,
                actual_generation,
                expected_generation,
            } => {
                let mut fields = residency_fields(identity);
                fields.extend([
                    ("actual_generation", actual_generation.to_string()),
                    ("expected_generation", expected_generation.to_string()),
                ]);
                fields
            }
            Self::ExportScanoutZeroGeometry { width, height } => {
                vec![("width", width.to_string()), ("height", height.to_string())]
            }
            Self::ExportScanoutLengthMismatch {
                width,
                height,
                actual_len,
                expected_len,
            } => vec![
                ("width", width.to_string()),
                ("height", height.to_string()),
                ("actual_len", actual_len.to_string()),
                ("expected_len", expected_len.to_string()),
            ],
            Self::ExportPresentUnknownIdentity { identity }
            | Self::ExportPresentNotReady { identity }
            | Self::ScatterPresentUnknownIdentity { identity }
            | Self::ScatterPresentNotReady { identity }
            | Self::WindowSourceDisappearedBeforePin { identity } => identity_fields(identity),
        }
    }
}

impl std::fmt::Display for EngineFacadeDecline {
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

    fn residency() -> ComputeStorageResidencyKey {
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

    fn identity() -> TargetIdentity {
        TargetIdentity::Surface {
            id: 7,
            width: 64,
            height: 32,
            generation: 9,
        }
    }

    fn all() -> Vec<EngineFacadeDecline> {
        vec![
            EngineFacadeDecline::WindowPresenterNotAttached,
            EngineFacadeDecline::StorageReadResidentAbsent {
                identity: residency(),
            },
            EngineFacadeDecline::StorageReadGenerationMismatch {
                identity: residency(),
                actual_generation: 8,
                expected_generation: 9,
            },
            EngineFacadeDecline::ExportScanoutZeroGeometry {
                width: 0,
                height: 32,
            },
            EngineFacadeDecline::ExportScanoutLengthMismatch {
                width: 64,
                height: 32,
                actual_len: 7,
                expected_len: 8192,
            },
            EngineFacadeDecline::ExportPresentUnknownIdentity {
                identity: identity(),
            },
            EngineFacadeDecline::ExportPresentNotReady {
                identity: identity(),
            },
            EngineFacadeDecline::ScatterPresentUnknownIdentity {
                identity: identity(),
            },
            EngineFacadeDecline::ScatterPresentNotReady {
                identity: identity(),
            },
            EngineFacadeDecline::WindowSourceDisappearedBeforePin {
                identity: identity(),
            },
        ]
    }

    #[test]
    fn every_engine_facade_check_has_a_unique_log_safe_slug() {
        let mut slugs: Vec<_> = all().iter().map(Decline::slug).collect();
        for slug in &slugs {
            assert!(slug.starts_with("vk_engine_"), "{slug}");
            assert!(
                slug.bytes()
                    .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_'),
                "{slug}"
            );
        }
        slugs.sort_unstable();
        let before = slugs.len();
        slugs.dedup();
        assert_eq!(before, 10, "the engine façade reason census moved");
        assert_eq!(before, slugs.len(), "duplicate engine façade slug");
    }

    #[test]
    fn engine_facade_fields_are_structured_and_log_safe() {
        for decline in all() {
            let line = crate::observe::Emit::decline("engine_facade_test", &decline).render();
            assert!(line.starts_with(&format!("engine_facade_test reason={}", decline.slug())));
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
    fn scatter_present_state_failures_keep_identity_and_readiness_distinct() {
        let unknown = EngineFacadeDecline::ScatterPresentUnknownIdentity {
            identity: identity(),
        };
        let not_ready = EngineFacadeDecline::ScatterPresentNotReady {
            identity: identity(),
        };
        assert_ne!(unknown.slug(), not_ready.slug());
        assert_eq!(unknown.fields(), not_ready.fields());
        assert_eq!(
            unknown.fields(),
            vec![
                ("identity_kind", "surface".into()),
                ("identity_id", "7".into()),
                ("identity_width", "64".into()),
                ("identity_height", "32".into()),
                ("identity_generation", "9".into()),
            ]
        );
    }

    #[test]
    fn export_scanout_sites_return_their_exact_declines_before_gpu_work() {
        let zero = unsafe { super::super::export_scanout_from_bgra(0, 32, &[]) }
            .expect_err("zero geometry must decline");
        assert_eq!(zero.slug(), "vk_engine_export_scanout_zero_geometry");

        let short = unsafe { super::super::export_scanout_from_bgra(2, 2, &[0; 15]) }
            .expect_err("wrong byte length must decline");
        assert_eq!(short.slug(), "vk_engine_export_scanout_length_mismatch");
        assert_eq!(
            short.fields(),
            vec![
                ("width", "2".into()),
                ("height", "2".into()),
                ("actual_len", "15".into()),
                ("expected_len", "16".into()),
            ]
        );
    }
}
