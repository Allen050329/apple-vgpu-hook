//! Typed fd-dup failures on the zero-copy dmabuf export path.
//!
//! # Why this is not [`super::vk_call::VkCall`]
//!
//! Both zero-copy export rails `dup` the cached exportable-image fd so the
//! importer (QEMU on Linux, the host window on the arm64 MoltenVK pathway) owns
//! and closes its own copy of the dmabuf. `dup(2)` can fail — `EMFILE`/`ENFILE`
//! under host fd pressure — and that is a POSIX syscall failure carrying an
//! `errno`, not an ash call carrying a [`ash::vk::Result`]. So it needs its own
//! decline: the shape mirrors `VkCall` (rail + the driver's result code), but the
//! result code is `std::io::Error::raw_os_error()`, not a Vulkan enum.
//!
//! These were the last two `DrawError::Vulkan(String)` sites in `engine/mod.rs`,
//! spelled `"export_{scanout,present} dup fd: {e}"` around a `try_clone_to_owned`
//! that returns [`std::io::Error`]. Carried by [`super::types::DrawError::FdDup`],
//! which delegates its slug and fields here so the export sinks
//! (`runtime/drain/mod.rs` `scanout_gl_export_fail`, `lib.rs` `export_present`) name
//! the failing dup rather than flattening it into `vk_engine_vk_untyped`.

use crate::observe::Decline;

/// Which zero-copy export rail's fd dup failed.
///
/// The two rails dup a *different* fd (the scanout export ring vs. the resident
/// present export ring), so they are distinct reasons even though the syscall is
/// the same — the [`super::vk_call::VkCall`] *(rail, operation)* principle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FdDupRail {
    /// `export_scanout_from_bgra`'s scanout export fd (the CPU-capture → dmabuf
    /// scanout rail).
    ExportScanout,
    /// `export_present_from_resident_composited_fd_policy`'s present export fd
    /// (the zero-copy resident → dmabuf present rail).
    ExportPresent,
}

/// A failed `dup` of an export dmabuf fd: which rail, and the errno.
///
/// Carried by [`super::types::DrawError::FdDup`], which delegates its slug and
/// fields here so one event has one name at every layer — the same rule
/// [`super::vk_call::VkCall`] follows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FdDupDecline {
    pub rail: FdDupRail,
    /// `std::io::Error::raw_os_error()` — the errno the failed `dup(2)` set.
    /// `None` if the platform reported no OS error code.
    pub errno: Option<i32>,
}

impl FdDupDecline {
    pub fn new(rail: FdDupRail, err: &std::io::Error) -> Self {
        Self {
            rail,
            errno: err.raw_os_error(),
        }
    }

    /// The errno as a log-safe token — the numeric code, or `none`.
    fn errno_field(&self) -> String {
        self.errno
            .map(|n| n.to_string())
            .unwrap_or_else(|| "none".to_string())
    }
}

impl Decline for FdDupDecline {
    /// One slug per export rail, `fd_dup_export_<rail>`.
    fn slug(&self) -> &'static str {
        match self.rail {
            FdDupRail::ExportScanout => "fd_dup_export_scanout",
            FdDupRail::ExportPresent => "fd_dup_export_present",
        }
    }

    /// The errno the failed `dup(2)` carried — the load-bearing value (`EMFILE`
    /// vs a real error) the old `{e}` prose held.
    fn fields(&self) -> Vec<(&'static str, String)> {
        vec![("errno", self.errno_field())]
    }
}

impl std::fmt::Display for FdDupDecline {
    /// `reason=<slug> errno=<code>` — what a `{e}` in someone else's `format!`
    /// produces, matching the fields the emitter renders.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "reason={} errno={}", self.slug(), self.errno_field())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: &[FdDupRail] = &[FdDupRail::ExportScanout, FdDupRail::ExportPresent];

    /// Two rails sharing a slug would make a grep of the fail log unable to tell
    /// which export's dup refused. Every slug is also log-safe.
    #[test]
    fn every_rail_names_its_export_log_safe() {
        let mut slugs: Vec<&str> = ALL
            .iter()
            .map(|rail| {
                FdDupDecline {
                    rail: *rail,
                    errno: Some(24),
                }
                .slug()
            })
            .collect();
        for s in &slugs {
            assert!(
                s.starts_with("fd_dup_"),
                "slug {s:?} must carry its rail prefix"
            );
            assert!(
                s.bytes()
                    .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_'),
                "slug {s:?} must be lowercase snake_case"
            );
        }
        slugs.sort_unstable();
        let before = slugs.len();
        slugs.dedup();
        assert_eq!(before, slugs.len(), "duplicate FdDupRail slug");
    }

    /// The errno the old prose carried as `{e}` must reach the line as a
    /// whitespace-free `errno=` field, and a missing errno renders `none` rather
    /// than an empty value that would collapse two space-separated fields.
    #[test]
    fn the_line_carries_the_errno_or_none() {
        let with = FdDupDecline::new(
            FdDupRail::ExportScanout,
            &std::io::Error::from_raw_os_error(24),
        );
        let line = crate::observe::Emit::decline("scanout_gl_export_fail", &with).render();
        assert!(
            line.starts_with("scanout_gl_export_fail reason=fd_dup_export_scanout errno=24"),
            "{line}"
        );

        let without = FdDupDecline::new(
            FdDupRail::ExportPresent,
            &std::io::Error::new(std::io::ErrorKind::Other, "x"),
        );
        assert_eq!(without.errno, None);
        let line = crate::observe::Emit::decline("export_present", &without).render();
        assert!(line.contains("errno=none"), "{line}");
        for field in line.split(' ').skip(1) {
            assert!(!field.is_empty(), "double space in {line:?}");
        }
    }
}
