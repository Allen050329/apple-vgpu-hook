//! Task GVA resolver (port of `host/utils/reims-vgpu-gva-resolve`).

use crate::contract::endian::ld32;
use crate::contract::gva::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Geometry {
    pub page_shift: u32,
    pub page_size: u32,
    pub page_offset_mask: u32,
    pub index_bits: u32,
    pub index_mask: u32,
    pub entries_per_table: u32,
    pub pte_size: u32,
    pub pte_flag_mask: u32,
    pub pte_pfn_mask: u32,
    pub max_depth: u32,
    pub max_span_pages: u32,
}

pub const ARM64E_GEOMETRY: Geometry = Geometry {
    page_shift: PAGE_SHIFT_ARM64E,
    page_size: PAGE_SIZE_ARM64E,
    page_offset_mask: ARM64E_PAGE_OFFSET_MASK,
    index_bits: ARM64E_INDEX_BITS,
    index_mask: ARM64E_INDEX_MASK,
    entries_per_table: ARM64E_ENTRIES_PER_TABLE,
    pte_size: PTE_SIZE,
    pte_flag_mask: PTE_FLAG_MASK,
    pte_pfn_mask: PTE_PFN_MASK,
    max_depth: ARM64E_MAX_DEPTH,
    max_span_pages: MAX_SPAN_PAGES,
};

pub const X86_64_GEOMETRY: Geometry = Geometry {
    page_shift: PAGE_SHIFT_X86,
    page_size: PAGE_SIZE_X86,
    page_offset_mask: X86_64_PAGE_OFFSET_MASK,
    index_bits: X86_64_INDEX_BITS,
    index_mask: X86_64_INDEX_MASK,
    entries_per_table: X86_64_ENTRIES_PER_TABLE,
    pte_size: PTE_SIZE,
    pte_flag_mask: PTE_FLAG_MASK,
    pte_pfn_mask: PTE_PFN_MASK,
    max_depth: X86_64_MAX_DEPTH,
    max_span_pages: MAX_SPAN_PAGES,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Task {
    pub active: bool,
    pub directory_pfn: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct TaskRoot {
    pub directory_pfn: u32,
    pub root_pfn: u32,
    pub depth: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
#[derive(Default)]
pub enum ResolveStatus {
    #[default]
    Ok = 0,
    ErrArgs = 1,
    ErrInactiveTask = 2,
    ErrNoDirectory = 3,
    ErrDirectoryRead = 4,
    ErrZeroRootPfn = 5,
    ErrZeroDepth = 6,
    ErrDepthTooDeep = 7,
    ErrAddressOutOfRange = 8,
    ErrPageTableRead = 9,
    ErrZeroPfn = 10,
    ErrMalformedPte = 11,
    ErrUnsupportedGeometry = 14,
}

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
    /// `gva_` prefix: `contract/` is a shared module and these names
    /// (`args`, `zero_pfn`, `span_overflow`) are generic enough to collide with
    /// half the crate.
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheStatus {
    Disabled = 0,
    Hit,
    Miss,
    MissInserted,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Translation {
    pub status: ResolveStatus,
    pub cache_status: CacheStatus,
    pub gva: u64,
    pub gpa: u64,
    pub gva_page_index: u64,
    pub gpa_page: u64,
    pub directory_pfn: u32,
    pub root_pfn: u32,
    pub depth: u32,
    pub leaf_pfn: u32,
    pub level: u32,
    pub entry_index: u32,
    pub raw_pte: u32,
}

impl Default for Translation {
    fn default() -> Self {
        Self {
            status: ResolveStatus::Ok,
            cache_status: CacheStatus::Disabled,
            gva: 0,
            gpa: 0,
            gva_page_index: 0,
            gpa_page: 0,
            directory_pfn: 0,
            root_pfn: 0,
            depth: 0,
            leaf_pfn: 0,
            level: 0,
            entry_index: 0,
            raw_pte: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CacheEntry {
    pub valid: bool,
    pub page_shift: u32,
    pub index_bits: u32,
    pub root_pfn: u32,
    pub depth: u32,
    pub page_index: u64,
    pub gpa_page: u64,
}

#[derive(Clone, Debug)]
pub struct Cache {
    pub entries: [CacheEntry; CACHE_WAYS],
    pub next: u32,
}

impl Default for Cache {
    fn default() -> Self {
        Self {
            entries: [CacheEntry::default(); CACHE_WAYS],
            next: 0,
        }
    }
}

/// Callback: read `len` guest physical bytes at `gpa` into `dst`.
pub trait PhysReader {
    fn read_phys(&self, gpa: u64, dst: &mut [u8]) -> bool;
}

pub fn resolve_status_name(status: ResolveStatus) -> &'static str {
    match status {
        ResolveStatus::Ok => "ok",
        ResolveStatus::ErrArgs => "args",
        ResolveStatus::ErrInactiveTask => "inactive-task",
        ResolveStatus::ErrNoDirectory => "no-directory",
        ResolveStatus::ErrDirectoryRead => "directory-read",
        ResolveStatus::ErrZeroRootPfn => "zero-root-pfn",
        ResolveStatus::ErrZeroDepth => "zero-depth",
        ResolveStatus::ErrDepthTooDeep => "depth-too-deep",
        ResolveStatus::ErrAddressOutOfRange => "address-out-of-range",
        ResolveStatus::ErrPageTableRead => "page-table-read",
        ResolveStatus::ErrZeroPfn => "zero-pfn",
        ResolveStatus::ErrMalformedPte => "malformed-pte",
        ResolveStatus::ErrUnsupportedGeometry => "unsupported-geometry",
    }
}

pub fn validate_geometry(geometry: &Geometry) -> ResolveStatus {
    if geometry.page_shift >= 31 || geometry.index_bits >= 31 {
        return ResolveStatus::ErrUnsupportedGeometry;
    }
    if geometry.page_shift != PAGE_SHIFT_ARM64E && geometry.page_shift != PAGE_SHIFT_X86 {
        return ResolveStatus::ErrUnsupportedGeometry;
    }
    let page_size = 1u32 << geometry.page_shift;
    let entries_per_table = 1u32 << geometry.index_bits;
    if geometry.page_size != page_size
        || geometry.page_offset_mask != page_size - 1
        || geometry.entries_per_table != entries_per_table
        || geometry.index_mask != entries_per_table - 1
        || geometry.pte_size != PTE_SIZE
        || geometry.pte_flag_mask != PTE_FLAG_MASK
        || geometry.pte_pfn_mask != PTE_PFN_MASK
        || geometry.max_depth == 0
        || geometry.max_depth > MAX_DEPTH
        || geometry.max_span_pages == 0
        || geometry.max_span_pages > MAX_SPAN_PAGES
    {
        return ResolveStatus::ErrUnsupportedGeometry;
    }
    if (geometry.entries_per_table as u64) * (geometry.pte_size as u64) != geometry.page_size as u64
    {
        return ResolveStatus::ErrUnsupportedGeometry;
    }
    if geometry.page_shift != geometry.index_bits + 2 {
        return ResolveStatus::ErrUnsupportedGeometry;
    }
    ResolveStatus::Ok
}

fn pfn_to_gpa(geometry: &Geometry, pfn: u32) -> Option<u64> {
    if geometry.page_shift >= 64 {
        return None;
    }
    if (pfn as u64) > (u64::MAX >> geometry.page_shift) {
        return None;
    }
    Some((pfn as u64) << geometry.page_shift)
}

fn read_u32_phys(reader: &dyn PhysReader, gpa: u64) -> Option<u32> {
    let mut bytes = [0u8; 4];
    if !reader.read_phys(gpa, &mut bytes) {
        return None;
    }
    Some(ld32(&bytes))
}

fn cache_lookup(
    cache: Option<&Cache>,
    geometry: &Geometry,
    root_pfn: u32,
    depth: u32,
    page_index: u64,
) -> Option<u64> {
    let cache = cache?;
    for entry in &cache.entries {
        if entry.valid
            && entry.page_shift == geometry.page_shift
            && entry.index_bits == geometry.index_bits
            && entry.root_pfn == root_pfn
            && entry.depth == depth
            && entry.page_index == page_index
        {
            return Some(entry.gpa_page);
        }
    }
    None
}

fn cache_insert(
    cache: &mut Cache,
    geometry: &Geometry,
    root_pfn: u32,
    depth: u32,
    page_index: u64,
    gpa_page: u64,
) {
    let slot = cache.next as usize % CACHE_WAYS;
    cache.next = (cache.next + 1) % CACHE_WAYS as u32;
    cache.entries[slot] = CacheEntry {
        valid: true,
        page_shift: geometry.page_shift,
        index_bits: geometry.index_bits,
        root_pfn,
        depth,
        page_index,
        gpa_page,
    };
}

pub fn read_task_root(
    reader: &dyn PhysReader,
    task: &Task,
    geometry: &Geometry,
) -> Result<TaskRoot, ResolveStatus> {
    let gs = validate_geometry(geometry);
    if gs != ResolveStatus::Ok {
        return Err(gs);
    }
    if !task.active {
        return Err(ResolveStatus::ErrInactiveTask);
    }
    if task.directory_pfn == 0 {
        return Err(ResolveStatus::ErrNoDirectory);
    }
    let dir_gpa =
        pfn_to_gpa(geometry, task.directory_pfn).ok_or(ResolveStatus::ErrAddressOutOfRange)?;
    let root_pfn = read_u32_phys(reader, dir_gpa + DIRECTORY_ROOT_PFN as u64)
        .ok_or(ResolveStatus::ErrDirectoryRead)?;
    let depth = read_u32_phys(reader, dir_gpa + DIRECTORY_DEPTH as u64)
        .ok_or(ResolveStatus::ErrDirectoryRead)?;
    Ok(TaskRoot {
        directory_pfn: task.directory_pfn,
        root_pfn,
        depth,
    })
}

pub fn translate_root(
    reader: &dyn PhysReader,
    geometry: &Geometry,
    root_pfn: u32,
    depth: u32,
    gva: u64,
    cache: Option<&mut Cache>,
) -> Translation {
    let mut out = Translation {
        gva,
        root_pfn,
        depth,
        ..Default::default()
    };
    let gs = validate_geometry(geometry);
    if gs != ResolveStatus::Ok {
        out.status = gs;
        return out;
    }
    out.gva_page_index = gva >> geometry.page_shift;
    if root_pfn == 0 {
        out.status = ResolveStatus::ErrZeroRootPfn;
        return out;
    }
    if depth == 0 {
        out.status = ResolveStatus::ErrZeroDepth;
        return out;
    }
    if depth > geometry.max_depth {
        out.status = ResolveStatus::ErrDepthTooDeep;
        return out;
    }

    let page_off = gva & geometry.page_offset_mask as u64;
    if let Some(cached_gpa_page) = cache_lookup(
        cache.as_deref(),
        geometry,
        root_pfn,
        depth,
        out.gva_page_index,
    ) {
        out.status = ResolveStatus::Ok;
        out.cache_status = CacheStatus::Hit;
        out.gpa_page = cached_gpa_page;
        if u64::MAX - cached_gpa_page < page_off {
            out.status = ResolveStatus::ErrAddressOutOfRange;
            return out;
        }
        out.gpa = cached_gpa_page + page_off;
        out.leaf_pfn = (cached_gpa_page >> geometry.page_shift) as u32;
        return out;
    }
    out.cache_status = if cache.is_none() {
        CacheStatus::Disabled
    } else {
        CacheStatus::Miss
    };

    let page_index = out.gva_page_index;
    let mut current_pfn = root_pfn;
    for level in 0..depth {
        let shift = (depth - 1 - level) * geometry.index_bits;
        let entry_idx = ((page_index >> shift) & geometry.index_mask as u64) as u32;
        let table_gpa = match pfn_to_gpa(geometry, current_pfn) {
            Some(v) => v,
            None => {
                out.status = ResolveStatus::ErrAddressOutOfRange;
                return out;
            }
        };
        let entry_offset = (entry_idx as u64) * (geometry.pte_size as u64);
        if u64::MAX - table_gpa < entry_offset {
            out.status = ResolveStatus::ErrAddressOutOfRange;
            return out;
        }
        let pte_gpa = table_gpa + entry_offset;
        out.level = level;
        out.entry_index = entry_idx;
        let pte = match read_u32_phys(reader, pte_gpa) {
            Some(v) => v,
            None => {
                out.status = ResolveStatus::ErrPageTableRead;
                return out;
            }
        };
        out.raw_pte = pte;
        let next_pfn = pte & geometry.pte_pfn_mask;
        if next_pfn == 0 {
            out.status = if pte == 0 {
                ResolveStatus::ErrZeroPfn
            } else {
                ResolveStatus::ErrMalformedPte
            };
            return out;
        }
        if level + 1 == depth {
            let gpa_page = match pfn_to_gpa(geometry, next_pfn) {
                Some(v) => v,
                None => {
                    out.status = ResolveStatus::ErrAddressOutOfRange;
                    return out;
                }
            };
            if u64::MAX - gpa_page < page_off {
                out.status = ResolveStatus::ErrAddressOutOfRange;
                return out;
            }
            out.status = ResolveStatus::Ok;
            out.leaf_pfn = next_pfn;
            out.gpa_page = gpa_page;
            out.gpa = gpa_page + page_off;
            if let Some(c) = cache {
                cache_insert(c, geometry, root_pfn, depth, page_index, gpa_page);
                out.cache_status = CacheStatus::MissInserted;
            }
            return out;
        }
        current_pfn = next_pfn;
    }
    out.status = ResolveStatus::ErrZeroDepth;
    out
}

#[allow(
    clippy::too_many_arguments,
    reason = "the walker exposes each page-table and span input explicitly"
)]
#[cfg(test)]
mod tests {

    use super::*;
    use crate::model::PAGE_SHIFT_ARM64E;
    use std::collections::HashMap;

    struct MapReader {
        map: HashMap<u64, u8>,
    }

    impl MapReader {
        fn new() -> Self {
            Self {
                map: HashMap::new(),
            }
        }
        fn put_u32(&mut self, gpa: u64, v: u32) {
            for (i, b) in v.to_le_bytes().iter().enumerate() {
                self.map.insert(gpa + i as u64, *b);
            }
        }
    }

    impl PhysReader for MapReader {
        fn read_phys(&self, gpa: u64, dst: &mut [u8]) -> bool {
            for (i, slot) in dst.iter_mut().enumerate() {
                match self.map.get(&(gpa + i as u64)) {
                    Some(b) => *slot = *b,
                    None => return false,
                }
            }
            true
        }
    }

    #[test]
    fn geometry_defaults() {
        assert_eq!(validate_geometry(&ARM64E_GEOMETRY), ResolveStatus::Ok);
        assert_eq!(validate_geometry(&X86_64_GEOMETRY), ResolveStatus::Ok);
        let mut bad = ARM64E_GEOMETRY;
        bad.page_shift = 13;
        assert_eq!(
            validate_geometry(&bad),
            ResolveStatus::ErrUnsupportedGeometry
        );
    }

    #[test]
    fn task_root_reads_directory_root_and_depth() {
        let mut r = MapReader::new();
        let dir_gpa = (2u64) << PAGE_SHIFT_ARM64E;
        r.put_u32(dir_gpa + DIRECTORY_ROOT_PFN as u64, 1);
        r.put_u32(dir_gpa + DIRECTORY_DEPTH as u64, 1);
        let task = Task {
            active: true,
            directory_pfn: 2,
        };
        let root = read_task_root(&r, &task, &ARM64E_GEOMETRY).unwrap();
        assert_eq!(root.directory_pfn, 2);
        assert_eq!(root.root_pfn, 1);
        assert_eq!(root.depth, 1);
    }

    #[test]
    fn translate_one_level() {
        // depth=1, root_pfn=1, GVA page 0 -> leaf pfn 5
        let mut r = MapReader::new();
        // table at pfn 1: entry 0 = pfn 5
        let table_gpa = (1u64) << PAGE_SHIFT_ARM64E;
        r.put_u32(table_gpa, 5);
        let t = translate_root(&r, &ARM64E_GEOMETRY, 1, 1, 0x100, None);
        assert_eq!(t.status, ResolveStatus::Ok);
        assert_eq!(t.leaf_pfn, 5);
        assert_eq!(t.gpa, ((5u64) << PAGE_SHIFT_ARM64E) + 0x100);
        assert_eq!(t.cache_status, CacheStatus::Disabled);
    }

    #[test]
    fn cache_hit_miss() {
        let mut r = MapReader::new();
        let table_gpa = (1u64) << PAGE_SHIFT_ARM64E;
        r.put_u32(table_gpa, 5);
        let mut cache = Cache::default();
        let t1 = translate_root(&r, &ARM64E_GEOMETRY, 1, 1, 0x100, Some(&mut cache));
        assert_eq!(t1.cache_status, CacheStatus::MissInserted);
        let t2 = translate_root(&r, &ARM64E_GEOMETRY, 1, 1, 0x200, Some(&mut cache));
        assert_eq!(t2.cache_status, CacheStatus::Hit);
        assert_eq!(t2.gpa, ((5u64) << PAGE_SHIFT_ARM64E) + 0x200);
    }

    #[test]
    fn zero_pfn_and_inactive() {
        let r = MapReader::new();
        let task = Task {
            active: false,
            directory_pfn: 1,
        };
        assert_eq!(
            read_task_root(&r, &task, &ARM64E_GEOMETRY).unwrap_err(),
            ResolveStatus::ErrInactiveTask
        );
        let t = translate_root(&r, &ARM64E_GEOMETRY, 0, 1, 0, None);
        assert_eq!(t.status, ResolveStatus::ErrZeroRootPfn);
    }
}
