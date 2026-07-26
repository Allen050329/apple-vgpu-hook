//! GVA page-table geometry constants (reims_vgpu_gva_format.h).

pub const DIRECTORY_ROOT_PFN: u32 = 0x00;
pub const DIRECTORY_DEPTH: u32 = 0x04;

pub const PTE_SIZE: u32 = 4;
pub const PTE_FLAG_MASK: u32 = 0x8000_0000;
pub const PTE_PFN_MASK: u32 = 0x7fff_ffff;

pub const MAX_DEPTH: u32 = 4;
pub const MAX_SPAN_PAGES: u32 = 1 << 20;

pub const PAGE_SHIFT_ARM64E: u32 = 14;
pub const PAGE_SIZE_ARM64E: u32 = 1 << PAGE_SHIFT_ARM64E;
pub const ARM64E_PAGE_OFFSET_MASK: u32 = PAGE_SIZE_ARM64E - 1;
pub const ARM64E_INDEX_BITS: u32 = 12;
pub const ARM64E_INDEX_MASK: u32 = 0xfff;
pub const ARM64E_ENTRIES_PER_TABLE: u32 = 1 << ARM64E_INDEX_BITS;
pub const ARM64E_MAX_DEPTH: u32 = MAX_DEPTH;

pub const PAGE_SHIFT_X86: u32 = 12;
pub const PAGE_SIZE_X86: u32 = 1 << PAGE_SHIFT_X86;
pub const X86_64_PAGE_OFFSET_MASK: u32 = PAGE_SIZE_X86 - 1;
pub const X86_64_INDEX_BITS: u32 = 10;
pub const X86_64_INDEX_MASK: u32 = 0x3ff;
pub const X86_64_ENTRIES_PER_TABLE: u32 = 1 << X86_64_INDEX_BITS;
pub const X86_64_MAX_DEPTH: u32 = MAX_DEPTH;

// No bare `PAGE_SHIFT` / `PAGE_SIZE` — those silently meant arm64e and caused
// cross-arch bugs. Use PAGE_SHIFT_ARM64E / PAGE_SHIFT_X86 / device page_shift.
pub const ARM64E_PAGE_OFFSET_MASK_ALIAS: u32 = ARM64E_PAGE_OFFSET_MASK;
pub const INDEX_BITS: u32 = ARM64E_INDEX_BITS;
pub const INDEX_MASK: u32 = ARM64E_INDEX_MASK;
pub const ENTRIES_PER_TABLE: u32 = ARM64E_ENTRIES_PER_TABLE;

pub const CACHE_WAYS: usize = 8;

/// PFN → GPA at an explicit guest page shift (12 or 14).
#[inline]
pub fn pfn_to_gpa(pfn: u32, page_shift: u32) -> u64 {
    (pfn as u64) << page_shift
}

#[inline]
pub fn page_index(address: u64, page_shift: u32) -> u64 {
    address >> page_shift
}

#[inline]
pub fn page_offset(address: u64, page_shift: u32) -> u32 {
    let mask = (1u64 << page_shift) - 1;
    (address & mask) as u32
}

#[inline]
pub fn page_aligned(address: u64, page_shift: u32) -> bool {
    page_offset(address, page_shift) == 0
}

#[inline]
pub fn pages_to_bytes(page_count: u32, page_shift: u32) -> u64 {
    (page_count as u64) << page_shift
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

    #[test]
    fn explicit_shift_helpers_round_trip_page_and_offset() {
        for shift in [PAGE_SHIFT_X86, PAGE_SHIFT_ARM64E] {
            let gpa = pfn_to_gpa(0x1234, shift);
            assert!(page_aligned(gpa, shift));
            assert_eq!(page_index(gpa + 0x321, shift), 0x1234);
            assert_eq!(page_offset(gpa + 0x321, shift), 0x321);
            assert_eq!(pages_to_bytes(0x1234, shift), gpa);
        }
    }
}
