//! GVA page-table geometry, as this device names it.
//!
//! Nothing here is declared. The format itself lives in
//! `reims_vgpu_wire::page_table`, which owns it — so every value below is either
//! re-exported from there or computed from it, and the two cannot drift because
//! there is only one of each. What this module adds is *this device's names for them*, which the
//! rest of the crate uses and which encode a rule the wire crate has no reason
//! to care about (see the arch-prefix note below).
//!
//! An earlier version of this file declared the same constants a second time,
//! with a test asserting the two agreed. That is the right remedy where the two
//! sides cannot see each other — the QEMU ABI header, which Rust does not
//! include — but both of these are Rust, so a re-export is strictly better: it
//! makes the drift impossible rather than detectable.

use reims_vgpu_wire::page_table as wire;

/// Offsets within a task's directory page. Narrowed from the wire crate's `u64`
/// because every consumer here indexes a `u32` field set.
pub const DIRECTORY_ROOT_PFN: u32 = wire::DIRECTORY_ROOT_PFN as u32;
pub const DIRECTORY_DEPTH: u32 = wire::DIRECTORY_DEPTH as u32;

// No `MAX_SPAN_PAGES`. There was one, `1 << 20`, whose doc said the guest's page
// table could describe a longer span and this device declined instead. No such
// decline existed: the value was carried as a `Geometry` field, set from the
// constant in both of the two geometries there are, compared against the same
// constant by `validate_geometry`, and dropped by `wire_geometry` before the
// walk ever saw it. A span of any length resolved, which is the faithful
// behaviour — so the constant is gone rather than the behaviour, and this note
// stands where a reader would otherwise "restore" a refusal that never ran.

pub const PAGE_SHIFT_ARM64E: u32 = wire::ARM64E.page_shift;
pub const PAGE_SIZE_ARM64E: u32 = wire::ARM64E.page_size() as u32;
pub const ARM64E_PAGE_OFFSET_MASK: u32 = wire::ARM64E.page_offset_mask() as u32;
pub const ARM64E_INDEX_BITS: u32 = wire::ARM64E.index_bits();
pub const ARM64E_INDEX_MASK: u32 = wire::ARM64E.index_mask() as u32;
pub const ARM64E_ENTRIES_PER_TABLE: u32 = wire::ARM64E.entries_per_table() as u32;
pub const ARM64E_MAX_DEPTH: u32 = wire::ARM64E.max_depth;

pub const PAGE_SHIFT_X86: u32 = wire::X86_64.page_shift;
pub const PAGE_SIZE_X86: u32 = wire::X86_64.page_size() as u32;
pub const X86_64_PAGE_OFFSET_MASK: u32 = wire::X86_64.page_offset_mask() as u32;
pub const X86_64_INDEX_BITS: u32 = wire::X86_64.index_bits();
pub const X86_64_INDEX_MASK: u32 = wire::X86_64.index_mask() as u32;
pub const X86_64_ENTRIES_PER_TABLE: u32 = wire::X86_64.entries_per_table() as u32;
pub const X86_64_MAX_DEPTH: u32 = wire::X86_64.max_depth;

// No bare `PAGE_SHIFT`, `PAGE_SIZE`, `INDEX_BITS`, `INDEX_MASK` or
// `ENTRIES_PER_TABLE`. Every one of those silently meant arm64e and caused
// cross-arch bugs. Use the arch-prefixed name or the device `page_shift`.

/// PFN → GPA at an explicit guest page shift (12 or 14). No default.
///
/// `model::regs` re-exports this rather than restating it. It had its own copy
/// with the same body and the same doc, and the ring drains reached that one
/// while this one was reached by nothing but the round-trip test below — two
/// definitions of one shift, either of which could have been changed alone.
#[inline]
pub fn pfn_to_gpa(pfn: u32, page_shift: u32) -> u64 {
    (pfn as u64) << page_shift
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn architecture_page_geometry_is_self_consistent() {
        assert_eq!(PAGE_SIZE_ARM64E, 1 << PAGE_SHIFT_ARM64E);
        assert_eq!(ARM64E_PAGE_OFFSET_MASK, PAGE_SIZE_ARM64E - 1);
        assert_eq!(ARM64E_ENTRIES_PER_TABLE, 1 << ARM64E_INDEX_BITS);
        assert_eq!(ARM64E_INDEX_MASK, ARM64E_ENTRIES_PER_TABLE - 1);

        assert_eq!(PAGE_SIZE_X86, 1 << PAGE_SHIFT_X86);
        assert_eq!(X86_64_PAGE_OFFSET_MASK, PAGE_SIZE_X86 - 1);
        assert_eq!(X86_64_ENTRIES_PER_TABLE, 1 << X86_64_INDEX_BITS);
        assert_eq!(X86_64_INDEX_MASK, X86_64_ENTRIES_PER_TABLE - 1);
        assert_ne!(PAGE_SIZE_ARM64E, PAGE_SIZE_X86);
    }

    /// The wire crate states page geometry in `u64`; this module narrows it.
    ///
    /// The narrowing is silent, so it gets a test. There is no drift test here
    /// on purpose — these names are re-exports and computations, not a second
    /// declaration, so there is nothing that *can* disagree. What a widened
    /// page shift would do instead is truncate, and this is what catches that.
    #[test]
    fn narrowing_the_wire_geometry_to_this_devices_width_loses_nothing() {
        for g in [wire::X86_64, wire::ARM64E] {
            assert_eq!(g.page_size() as u32 as u64, g.page_size());
            assert_eq!(g.page_offset_mask() as u32 as u64, g.page_offset_mask());
            assert_eq!(g.index_mask() as u32 as u64, g.index_mask());
            assert_eq!(g.entries_per_table() as u32 as u64, g.entries_per_table());
        }
        assert_eq!(DIRECTORY_ROOT_PFN as u64, wire::DIRECTORY_ROOT_PFN);
        assert_eq!(DIRECTORY_DEPTH as u64, wire::DIRECTORY_DEPTH);
    }

    /// A PFN shifted to a GPA still names its own page at any offset inside it.
    ///
    /// Stated at both shifts because that is the whole reason `pfn_to_gpa` takes
    /// one: a helper that assumed 14 is what put x86 stamp writes on the wrong
    /// page. The inverse is written out rather than called, because no product
    /// path wanted a named `page_index` helper and an unused one is a second
    /// place this shift could be changed.
    #[test]
    fn a_pfn_shifted_to_a_gpa_names_its_own_page_at_either_shift() {
        for shift in [PAGE_SHIFT_X86, PAGE_SHIFT_ARM64E] {
            let gpa = pfn_to_gpa(0x1234, shift);
            assert_eq!((gpa + 0x321) >> shift, 0x1234);
        }
    }
}
