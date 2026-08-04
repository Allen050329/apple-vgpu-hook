//! Task GVA resolver (port of `host/utils/reims-vgpu-gva-resolve`).

use crate::contract::endian::ld32;
use crate::contract::gva::*;
use reims_vgpu_wire::mem as wire_mem;
use reims_vgpu_wire::page_table as wire_page_table;

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

/// Presents a [`PhysReader`] as the wire crate's guest-memory seam.
///
/// Deliberately borrows rather than owns, and implements only `read_at`: a
/// `PhysReader` cannot hand back a borrow of guest bytes, so the default
/// `slice_at` — which refuses every borrow — is the correct behaviour and not
/// an omission.
struct PhysAsGuestMemory<'a>(&'a dyn PhysReader);

impl wire_mem::GuestMemory for PhysAsGuestMemory<'_> {
    fn read_at(&self, addr: u64, out: &mut [u8]) -> bool {
        self.0.read_phys(addr, out)
    }
}

/// This module's geometry as the wire crate's.
///
/// The wire crate derives fan-out, masks and page size from the page shift
/// because a node is one page of four-byte entries. This module carries them as
/// separate fields and [`validate_geometry`] checks they agree, so the
/// conversion drops the derived ones rather than translating them.
fn wire_geometry(geometry: &Geometry) -> wire_page_table::Geometry {
    wire_page_table::Geometry {
        page_shift: geometry.page_shift,
        max_depth: geometry.max_depth,
    }
}

/// Wire walk failures as this device's typed refusals.
///
/// The names differ in one place worth stating: the wire crate calls a zero
/// entry `NotPresent`, because Apple's builder writes zero for absent. This
/// device calls the same thing `ErrZeroPfn` and reports it as `gva_zero_pfn`,
/// which is the guest saying "not mapped here" rather than a device defect. The
/// slug is unchanged so the fail log reads the same.
fn resolve_status_of(error: wire_page_table::WalkError) -> ResolveStatus {
    use wire_page_table::WalkError as W;
    match error {
        W::UnsupportedGeometry => ResolveStatus::ErrUnsupportedGeometry,
        W::ZeroRootPfn => ResolveStatus::ErrZeroRootPfn,
        W::ZeroDepth => ResolveStatus::ErrZeroDepth,
        W::DepthTooDeep => ResolveStatus::ErrDepthTooDeep,
        W::TableRead => ResolveStatus::ErrPageTableRead,
        W::NotPresent => ResolveStatus::ErrZeroPfn,
        W::MalformedPte => ResolveStatus::ErrMalformedPte,
    }
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
    let dir_gpa = pfn_to_gpa(task.directory_pfn, geometry.page_shift);
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
        out.gpa = cached_gpa_page + page_off;
        out.leaf_pfn = (cached_gpa_page >> geometry.page_shift) as u32;
        return out;
    }
    out.cache_status = if cache.is_none() {
        CacheStatus::Disabled
    } else {
        CacheStatus::Miss
    };

    // The descent itself lives in `reims_vgpu_wire::page_table`, which owns the
    // format. Keeping it there means there is one declaration rather than two
    // that could drift apart silently, and the tree walk gets exercised by that
    // crate's tests as well as by this module's.
    //
    // What stays here is everything that is not byte interpretation: the
    // translation cache above, and the typed refusal statuses below that the
    // device's failure channel reports.
    let page_index = out.gva_page_index;
    let mem = PhysAsGuestMemory(reader);
    let walked = wire_page_table::walk(&mem, wire_geometry(geometry), root_pfn, depth, gva);

    let w = match walked {
        Ok(w) => w,
        Err(f) => {
            out.level = f.level;
            out.entry_index = f.entry_index;
            out.raw_pte = f.raw_pte;
            out.status = resolve_status_of(f.error);
            return out;
        }
    };

    // On success the walker reports the deepest level it read, which is where
    // the loop this replaced left these fields.
    out.level = depth - 1;
    out.entry_index = (page_index & geometry.index_mask as u64) as u32;
    out.raw_pte = w.raw_pte;
    let gpa_page = w.addr_page;
    out.status = ResolveStatus::Ok;
    out.leaf_pfn = w.leaf_pfn;
    out.gpa_page = gpa_page;
    out.gpa = gpa_page + page_off;
    debug_assert_eq!(out.gpa, w.addr);
    if let Some(c) = cache {
        cache_insert(c, geometry, root_pfn, depth, page_index, gpa_page);
        out.cache_status = CacheStatus::MissInserted;
    }
    out
}

/// Translate a run of consecutive pages under one root, calling `visit` with
/// each page's GPA or `None` for a page the table cannot translate.
///
/// The run form of [`translate_root`], and it exists for one reason: the
/// per-page form re-reads every upper level of the tree for every page, because
/// the [`Cache`] it consults holds finished translations keyed by the exact page
/// index and a run visits each index once. A 1080p surface's licence check walks
/// 2 025 consecutive pages and paid `depth` guest-memory reads for each; the
/// upper levels of all of them are the same three or four entries.
///
/// The descent still lives in `reims_vgpu_wire::page_table`, which owns the
/// format — [`wire_page_table::walk_run`] is the same walk with the repeated
/// upper reads elided, and that crate's tests assert it answers identically to
/// [`wire_page_table::walk`] page for page.
///
/// The visitor stops the run by answering `false`. It is not called at all when
/// the root or geometry is unusable, so a caller must compare what it saw
/// against what it expected rather than reading a quiet return as agreement —
/// the same contract [`translate_root`] has by returning a status.
pub fn translate_root_run(
    reader: &dyn PhysReader,
    geometry: &Geometry,
    root_pfn: u32,
    depth: u32,
    first_gva: u64,
    pages: u64,
    visit: &mut dyn FnMut(u64, Option<u64>) -> bool,
) {
    if validate_geometry(geometry) != ResolveStatus::Ok || root_pfn == 0 || depth == 0 {
        return;
    }
    let mem = PhysAsGuestMemory(reader);
    wire_page_table::walk_run(
        &mem,
        wire_geometry(geometry),
        root_pfn,
        depth,
        first_gva,
        pages,
        &mut |index, walked| visit(index, walked.ok().map(|w| w.addr)),
    );
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

    /// A full-depth walk descends every level and takes the leaf from the last
    /// one.
    ///
    /// Every level of the walk reads its PTE identically, so the leaf is just
    /// the PFN the final level named. Nothing tested a walk deeper than one
    /// level before, which left that "just" unchecked: a walk that stopped one
    /// level early, or that took the leaf from the table PFN instead of the
    /// entry, resolves `depth == 1` correctly and every deeper `depth` wrong.
    /// The four indices here are distinct so an off-by-one level lands on the
    /// wrong table and reads nothing.
    #[test]
    fn a_four_level_walk_takes_its_leaf_from_the_deepest_entry() {
        const IDX: [u64; 4] = [1, 2, 3, 4];
        const TABLE_PFN: [u32; 4] = [10, 11, 12, 13];
        const LEAF_PFN: u32 = 0x555;
        const PAGE_OFF: u64 = 0x1234;

        let g = ARM64E_GEOMETRY;
        let page_index = (IDX[0] << 36) | (IDX[1] << 24) | (IDX[2] << 12) | IDX[3];

        let mut r = MapReader::new();
        for level in 0..4usize {
            let next = if level == 3 {
                // Flag bit set: the walk must mask it off, not fold it into the PFN.
                PTE_FLAG_MASK | LEAF_PFN
            } else {
                TABLE_PFN[level + 1]
            };
            let tbl = (TABLE_PFN[level] as u64) << g.page_shift;
            r.put_u32(tbl + IDX[level] * PTE_SIZE as u64, next);
        }

        let gva = (page_index << g.page_shift) | PAGE_OFF;
        let t = translate_root(&r, &g, TABLE_PFN[0], 4, gva, None);

        assert_eq!(t.status, ResolveStatus::Ok);
        assert_eq!(t.leaf_pfn, LEAF_PFN);
        assert_eq!(t.gpa, ((LEAF_PFN as u64) << g.page_shift) + PAGE_OFF);
        assert_eq!(t.gpa_page, (LEAF_PFN as u64) << g.page_shift);
        assert_eq!(t.level, 3, "the last level walked");
        assert_eq!(t.entry_index as u64, IDX[3]);
        assert_eq!(t.raw_pte, PTE_FLAG_MASK | LEAF_PFN);
    }

    /// No address a walk can form overflows, at any geometry `validate_geometry`
    /// accepts.
    ///
    /// The walk used to carry five `u64::MAX - x < y` guards and a fallible
    /// PFN-to-GPA helper, all of which were dead: a PFN is a `u32` and the
    /// accepted page shifts are 12 and 14, so the widest address the walk can
    /// name is under 2^46 and every addend is under 2^17. Those guards are gone,
    /// which makes this the only thing holding the premise. It fails if a
    /// geometry with a wider shift is ever accepted, which is exactly when the
    /// guards would have needed to come back.
    #[test]
    fn accepted_geometries_cannot_form_an_address_that_overflows() {
        for g in [ARM64E_GEOMETRY, X86_64_GEOMETRY] {
            assert_eq!(validate_geometry(&g), ResolveStatus::Ok);
            // The widest table or leaf base a `u32` PFN can name.
            let max_base = pfn_to_gpa(u32::MAX, g.page_shift);
            let max_entry_off = (g.index_mask as u64) * (g.pte_size as u64);
            let max_page_off = g.page_offset_mask as u64;
            assert!(max_base.checked_add(max_entry_off).is_some());
            assert!(max_base.checked_add(max_page_off).is_some());
        }

        // And nothing wider is accepted. The shift alone decides this, so the
        // rest of the geometry is left consistent-for-arm64e on purpose.
        for shift in 0..64u32 {
            let mut g = ARM64E_GEOMETRY;
            g.page_shift = shift;
            if validate_geometry(&g) == ResolveStatus::Ok {
                assert_eq!(shift, PAGE_SHIFT_ARM64E, "unexpected page shift accepted");
            }
        }
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
