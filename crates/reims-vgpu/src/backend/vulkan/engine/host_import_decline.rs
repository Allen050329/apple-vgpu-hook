//! Typed precondition failures for `VK_EXT_external_memory_host` imports.
//!
//! The resource-pool resolver also uses this vocabulary for its bounded-window
//! budget and capability preconditions. Keeping the pointer-alignment contract
//! here lets the low-level context and its higher-level callers name the same
//! check without depending on one another.

use crate::observe::Decline;

/// Every non-driver reason a host-pointer import can refuse.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HostImportDecline {
    RegionCount,
    TotalBytes,
    ZeroLength,
    ExtensionAbsent,
    PointerMisaligned {
        host_ptr: usize,
        alignment: u64,
    },
    SizeMisaligned {
        size: u64,
        alignment: u64,
    },
    RangeOverflow {
        host_ptr: usize,
        len: u64,
    },
    NoValidWindow {
        host_ptr: usize,
        len: u64,
        alignment: u64,
    },
}

impl Decline for HostImportDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::RegionCount => "host_import_region_count_cap",
            Self::TotalBytes => "host_import_total_byte_cap",
            Self::ZeroLength => "host_import_zero_length_span",
            Self::ExtensionAbsent => "host_import_extension_absent",
            Self::PointerMisaligned { .. } => "host_import_pointer_misaligned",
            Self::SizeMisaligned { .. } => "host_import_size_misaligned",
            Self::RangeOverflow { .. } => "host_import_range_overflow",
            Self::NoValidWindow { .. } => "host_import_no_valid_window",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::PointerMisaligned {
                host_ptr,
                alignment,
            } => vec![
                ("host_ptr", format!("{host_ptr:#x}")),
                ("alignment", alignment.to_string()),
            ],
            Self::SizeMisaligned { size, alignment } => vec![
                ("size", size.to_string()),
                ("alignment", alignment.to_string()),
            ],
            Self::RangeOverflow { host_ptr, len } => vec![
                ("host_ptr", format!("{host_ptr:#x}")),
                ("len", len.to_string()),
            ],
            Self::NoValidWindow {
                host_ptr,
                len,
                alignment,
            } => vec![
                ("host_ptr", format!("{host_ptr:#x}")),
                ("len", len.to_string()),
                ("alignment", alignment.to_string()),
            ],
            Self::RegionCount | Self::TotalBytes | Self::ZeroLength | Self::ExtensionAbsent => {
                Vec::new()
            }
        }
    }
}

impl std::fmt::Display for HostImportDecline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "reason={}", self.slug())?;
        for (key, value) in self.fields() {
            write!(f, " {key}={value}")?;
        }
        Ok(())
    }
}

pub(super) fn validate_host_import_alignment(
    host_ptr: usize,
    size: u64,
    alignment: u64,
) -> Result<(), HostImportDecline> {
    let alignment = alignment.max(1);
    if !(host_ptr as u64).is_multiple_of(alignment) {
        return Err(HostImportDecline::PointerMisaligned {
            host_ptr,
            alignment,
        });
    }
    if !size.is_multiple_of(alignment) {
        return Err(HostImportDecline::SizeMisaligned { size, alignment });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_host_import_cause_has_a_unique_log_safe_slug() {
        let all = [
            HostImportDecline::RegionCount,
            HostImportDecline::TotalBytes,
            HostImportDecline::ZeroLength,
            HostImportDecline::ExtensionAbsent,
            HostImportDecline::PointerMisaligned {
                host_ptr: 0x1001,
                alignment: 4096,
            },
            HostImportDecline::SizeMisaligned {
                size: 4097,
                alignment: 4096,
            },
            HostImportDecline::RangeOverflow {
                host_ptr: usize::MAX,
                len: 2,
            },
            HostImportDecline::NoValidWindow {
                host_ptr: 0x1000,
                len: 4096,
                alignment: 4096,
            },
        ];
        let mut slugs: Vec<_> = all.iter().map(Decline::slug).collect();
        for slug in &slugs {
            assert!(slug.starts_with("host_import_"), "{slug}");
            assert!(
                slug.bytes()
                    .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_'),
                "{slug}"
            );
        }
        slugs.sort_unstable();
        let before = slugs.len();
        slugs.dedup();
        assert_eq!(before, 8, "the host-import reason census moved");
        assert_eq!(before, slugs.len(), "duplicate host-import slug");
    }

    /// `host_import_resolve` returns its refusal so the caller can name it. That
    /// only buys anything if the wrapped leaf keeps its own identity: the eight
    /// causes used to collapse into one `None` at that boundary and every loss
    /// downstream reported the caller's coarse `host_import_resolve`, which is a
    /// label nobody measured. A byte-cap refusal and a driver rejection need
    /// opposite fixes, so re-collapsing them to a shared slug must fail here.
    #[test]
    fn wrapping_a_leaf_in_draw_error_preserves_its_own_slug() {
        use crate::backend::vulkan::engine::types::DrawError;

        let all = [
            HostImportDecline::RegionCount,
            HostImportDecline::TotalBytes,
            HostImportDecline::ZeroLength,
            HostImportDecline::ExtensionAbsent,
            HostImportDecline::PointerMisaligned {
                host_ptr: 0x1001,
                alignment: 4096,
            },
            HostImportDecline::SizeMisaligned {
                size: 0x2001,
                alignment: 4096,
            },
            HostImportDecline::RangeOverflow {
                host_ptr: usize::MAX,
                len: 0x1000,
            },
            HostImportDecline::NoValidWindow {
                host_ptr: 0x1000,
                len: 0x1000,
                alignment: 4096,
            },
        ];

        let mut wrapped: Vec<&'static str> = Vec::new();
        for leaf in all {
            let carried = DrawError::HostImport(leaf).slug();
            assert_eq!(
                carried,
                leaf.slug(),
                "DrawError::HostImport must carry the leaf's slug, not a class name"
            );
            // The coarse label the scatter store used to return for all eight.
            assert_ne!(
                carried, "host_import_resolve",
                "leaf collapsed back into the caller's class reason"
            );
            wrapped.push(carried);
        }
        wrapped.sort_unstable();
        let before = wrapped.len();
        wrapped.dedup();
        assert_eq!(before, 8, "the host-import reason census moved");
        assert_eq!(
            before,
            wrapped.len(),
            "two causes are indistinguishable once wrapped"
        );
    }

    #[test]
    fn alignment_contract_names_pointer_and_size_separately() {
        let pointer =
            validate_host_import_alignment(0x1001, 0x2000, 0x1000).expect_err("misaligned pointer");
        assert_eq!(pointer.slug(), "host_import_pointer_misaligned");
        assert_eq!(
            pointer.fields(),
            vec![("host_ptr", "0x1001".into()), ("alignment", "4096".into()),]
        );

        let size =
            validate_host_import_alignment(0x1000, 0x2001, 0x1000).expect_err("misaligned size");
        assert_eq!(size.slug(), "host_import_size_misaligned");
        assert_eq!(
            size.fields(),
            vec![("size", "8193".into()), ("alignment", "4096".into())]
        );
    }
}
