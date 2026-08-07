//! Task GVA resolver — the algorithms live in `reims_vgpu_paging::resolve`,
//! which this module re-exports under the device's established names. What
//! stays here is the one thing that cannot move: the mapping of the walk's
//! typed statuses onto the device's failure channel.
//!
//! The guest-memory seam is the wire crate's
//! [`GuestMemory`](reims_vgpu_wire::mem::GuestMemory); the device implements
//! it over [`crate::runtime::host::HostMemory`] at each caller.

pub use reims_vgpu_paging::resolve::{
    read_task_root, resolve_status_name, translate_root, translate_root_run, Geometry,
    ResolveStatus, Task, TaskRoot, Translation, ARM64E as ARM64E_GEOMETRY,
    X86_64 as X86_64_GEOMETRY,
};

impl crate::observe::Refusal for ResolveStatus {
    /// Fifteen distinct checks in the guest page-table walk, each with its own
    /// slug.
    ///
    /// They were already distinct *variants* — the walk has been honest about
    /// which check refused since it was written. What was missing is that every
    /// caller collapsed all fifteen into one `MemError::Unmapped`, and
    /// `MemError` reaches the always-on log at no site in the crate. So "the
    /// guest asked for a GVA and we could not produce it" was
    /// indistinguishable from "the directory PFN is zero", from "the PTE is
    /// malformed", from "the span overflowed" — and none of them was visible at
    /// all.
    ///
    /// `gva_` prefix: these names (`args`, `zero_pfn`, `span_overflow`) are
    /// generic enough to collide with half the crate.
    fn refusal(&self) -> Option<&'static str> {
        Some(match self {
            Self::Ok => return None,
            Self::ErrArgs => "gva_args",
            Self::ErrInactiveTask => "gva_inactive_task",
            Self::ErrNoDirectory => "gva_no_directory",
            Self::ErrDirectoryRead => "gva_directory_read",
            Self::ErrZeroRootPfn => "gva_zero_root_pfn",
            Self::ErrZeroDepth => "gva_zero_depth",
            Self::ErrDepthTooDeep => "gva_depth_too_deep",
            Self::ErrAddressOutOfRange => "gva_address_out_of_range",
            Self::ErrPageTableRead => "gva_page_table_read",
            Self::ErrZeroPfn => "gva_zero_pfn",
            Self::ErrMalformedPte => "gva_malformed_pte",
            Self::ErrUnsupportedGeometry => "gva_unsupported_geometry",
        })
    }
}
