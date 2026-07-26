//! Task GVA resolver (port of `host/utils/reims-vgpu-gva-resolve`).

use crate::contract::endian::ld32;
use crate::contract::gva::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Architecture {
    Arm64e = 0,
    X86_64 = 1,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Geometry {
    pub architecture: Architecture,
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
    architecture: Architecture::Arm64e,
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
    architecture: Architecture::X86_64,
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
    ErrSpanOverflow = 12,
    ErrVisitorStopped = 13,
    ErrUnsupportedGeometry = 14,
    ErrSpanTooLarge = 15,
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
            Self::ErrSpanOverflow => "gva_span_overflow",
            Self::ErrVisitorStopped => "gva_visitor_stopped",
            Self::ErrUnsupportedGeometry => "gva_unsupported_geometry",
            Self::ErrSpanTooLarge => "gva_span_too_large",
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SpanKind {
    #[default]
    Empty = 0,
    SinglePage,
    MultiPage,
    Overflow,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Span {
    pub status: ResolveStatus,
    pub kind: SpanKind,
    pub gva: u64,
    pub length: u64,
    pub first_page_index: u64,
    pub last_page_index: u64,
    pub page_count: u64,
    pub first_page_offset: u32,
    pub first_chunk_length: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpanChunk {
    pub gva: u64,
    pub gpa: u64,
    pub length: u64,
    pub page_index: u64,
    pub page_offset: u32,
    pub chunk_index: u64,
    pub chunk_count: u64,
    pub cache_status: CacheStatus,
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

impl Cache {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn invalidate(&mut self) {
        *self = Self::default();
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
        ResolveStatus::ErrSpanOverflow => "span-overflow",
        ResolveStatus::ErrVisitorStopped => "visitor-stopped",
        ResolveStatus::ErrUnsupportedGeometry => "unsupported-geometry",
        ResolveStatus::ErrSpanTooLarge => "span-too-large",
    }
}

pub fn cache_status_name(status: CacheStatus) -> &'static str {
    match status {
        CacheStatus::Disabled => "disabled",
        CacheStatus::Hit => "hit",
        CacheStatus::Miss => "miss",
        CacheStatus::MissInserted => "miss-inserted",
    }
}

pub fn span_kind_name(kind: SpanKind) -> &'static str {
    match kind {
        SpanKind::Empty => "empty",
        SpanKind::SinglePage => "single-page",
        SpanKind::MultiPage => "multi-page",
        SpanKind::Overflow => "overflow",
    }
}

pub fn arch_geometry(architecture: Architecture) -> &'static Geometry {
    match architecture {
        Architecture::Arm64e => &ARM64E_GEOMETRY,
        Architecture::X86_64 => &X86_64_GEOMETRY,
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

/// **Arm64e only** (16 KiB geometry). Prefer [`classify_span_with_geometry`].
pub fn classify_span_arm64e(gva: u64, length: u64) -> Span {
    classify_span_with_geometry(&ARM64E_GEOMETRY, gva, length)
}

pub fn classify_span_with_geometry(geometry: &Geometry, gva: u64, length: u64) -> Span {
    let mut out = Span {
        status: ResolveStatus::Ok,
        kind: SpanKind::Empty,
        gva,
        length,
        ..Default::default()
    };
    let gs = validate_geometry(geometry);
    if gs != ResolveStatus::Ok {
        out.status = gs;
        return out;
    }
    if length == 0 {
        out.kind = SpanKind::Empty;
        return out;
    }
    if u64::MAX - gva < length - 1 {
        out.status = ResolveStatus::ErrSpanOverflow;
        out.kind = SpanKind::Overflow;
        return out;
    }
    let last_byte = gva + (length - 1);
    out.first_page_index = gva >> geometry.page_shift;
    out.last_page_index = last_byte >> geometry.page_shift;
    out.page_count = out.last_page_index - out.first_page_index + 1;
    if out.page_count > geometry.max_span_pages as u64 {
        out.status = ResolveStatus::ErrSpanTooLarge;
        return out;
    }
    out.first_page_offset = (gva & geometry.page_offset_mask as u64) as u32;
    let mut first_chunk = geometry.page_size as u64 - out.first_page_offset as u64;
    if first_chunk > length {
        first_chunk = length;
    }
    out.first_chunk_length = first_chunk as u32;
    out.kind = if out.page_count == 1 {
        SpanKind::SinglePage
    } else {
        SpanKind::MultiPage
    };
    out
}

/// **Arm64e only.** Prefer span over [`classify_span_with_geometry`].
pub fn span_page_count_arm64e(gva: u64, length: u64) -> u32 {
    let span = classify_span_arm64e(gva, length);
    if span.status != ResolveStatus::Ok || span.page_count > u32::MAX as u64 {
        u32::MAX
    } else {
        span.page_count as u32
    }
}

pub fn strided_span_len(stride: u64, row_bytes: u32, rows: u32) -> Option<u64> {
    if row_bytes == 0 || rows == 0 {
        return None;
    }
    let repeat = (rows as u64) - 1;
    if stride != 0 && repeat > (u64::MAX - row_bytes as u64) / stride {
        return Some(u64::MAX);
    }
    Some(repeat * stride + row_bytes as u64)
}

pub fn span_first_page_slot(span: &Span, page_slot_count: u32) -> Option<u32> {
    if span.status != ResolveStatus::Ok
        || span.first_chunk_length == 0
        || page_slot_count == 0
        || span.first_page_index >= page_slot_count as u64
    {
        return None;
    }
    Some(span.first_page_index as u32)
}

pub fn span_chunk_inserted_page_gpa(chunk: &SpanChunk) -> Option<u64> {
    if chunk.cache_status != CacheStatus::MissInserted || chunk.gpa < chunk.page_offset as u64 {
        return None;
    }
    Some(chunk.gpa - chunk.page_offset as u64)
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

pub fn translate_task(
    reader: &dyn PhysReader,
    task: &Task,
    geometry: &Geometry,
    gva: u64,
    cache: Option<&mut Cache>,
) -> Translation {
    let mut out = Translation {
        gva,
        ..Default::default()
    };
    let root = match read_task_root(reader, task, geometry) {
        Ok(r) => r,
        Err(e) => {
            out.status = e;
            return out;
        }
    };
    let mut t = translate_root(reader, geometry, root.root_pfn, root.depth, gva, cache);
    t.directory_pfn = root.directory_pfn;
    t
}

#[allow(
    clippy::too_many_arguments,
    reason = "the walker exposes each page-table and span input explicitly"
)]
pub fn walk_span_root<F>(
    reader: &dyn PhysReader,
    geometry: &Geometry,
    root_pfn: u32,
    depth: u32,
    gva: u64,
    length: u64,
    cache: Option<&mut Cache>,
    mut visitor: F,
) -> Result<(), (ResolveStatus, Translation)>
where
    F: FnMut(&SpanChunk) -> bool,
{
    let span = classify_span_with_geometry(geometry, gva, length);
    if span.status != ResolveStatus::Ok {
        let failure = Translation {
            status: span.status,
            gva,
            ..Translation::default()
        };
        return Err((span.status, failure));
    }
    if span.kind == SpanKind::Empty {
        return Ok(());
    }

    let mut done = 0u64;
    let mut chunk_index = 0u64;
    // Re-borrow cache for each iteration without consuming.
    // Use raw split: store cache as Option pointer via rebind.
    let mut cache = cache;
    while done < length {
        let current_gva = gva + done;
        let page_off = (current_gva & geometry.page_offset_mask as u64) as u32;
        let mut chunk_len = geometry.page_size as u64 - page_off as u64;
        if chunk_len > length - done {
            chunk_len = length - done;
        }
        let translation = {
            let cache_ref = cache.as_deref_mut();
            translate_root(reader, geometry, root_pfn, depth, current_gva, cache_ref)
        };
        if translation.status != ResolveStatus::Ok {
            return Err((translation.status, translation));
        }
        let chunk = SpanChunk {
            gva: current_gva,
            gpa: translation.gpa,
            length: chunk_len,
            page_index: translation.gva_page_index,
            page_offset: page_off,
            chunk_index,
            chunk_count: span.page_count,
            cache_status: translation.cache_status,
        };
        if !visitor(&chunk) {
            let failure = Translation {
                status: ResolveStatus::ErrVisitorStopped,
                gva: current_gva,
                ..Translation::default()
            };
            return Err((ResolveStatus::ErrVisitorStopped, failure));
        }
        done += chunk_len;
        chunk_index += 1;
    }
    Ok(())
}

pub fn walk_span_task<F>(
    reader: &dyn PhysReader,
    task: &Task,
    geometry: &Geometry,
    gva: u64,
    length: u64,
    cache: Option<&mut Cache>,
    visitor: F,
) -> Result<(), (ResolveStatus, Translation)>
where
    F: FnMut(&SpanChunk) -> bool,
{
    let span = classify_span_with_geometry(geometry, gva, length);
    if span.status != ResolveStatus::Ok {
        let failure = Translation {
            status: span.status,
            gva,
            ..Translation::default()
        };
        return Err((span.status, failure));
    }
    if span.kind == SpanKind::Empty {
        return Ok(());
    }
    let root = match read_task_root(reader, task, geometry) {
        Ok(r) => r,
        Err(e) => {
            let failure = Translation {
                status: e,
                gva,
                ..Translation::default()
            };
            return Err((e, failure));
        }
    };
    match walk_span_root(
        reader,
        geometry,
        root.root_pfn,
        root.depth,
        gva,
        length,
        cache,
        visitor,
    ) {
        Ok(()) => Ok(()),
        Err((s, mut t)) => {
            t.directory_pfn = root.directory_pfn;
            Err((s, t))
        }
    }
}

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
    fn classify_span_basic() {
        let s = classify_span_arm64e(0x100, 0x10);
        assert_eq!(s.status, ResolveStatus::Ok);
        assert_eq!(s.kind, SpanKind::SinglePage);
        assert_eq!(s.first_chunk_length, 0x10);

        let s = classify_span_arm64e(0, 0);
        assert_eq!(s.kind, SpanKind::Empty);

        let s = classify_span_arm64e(PAGE_SIZE_ARM64E as u64 - 1, 2);
        assert_eq!(s.kind, SpanKind::MultiPage);
        assert_eq!(s.page_count, 2);
    }

    #[test]
    fn strided_and_slot() {
        assert_eq!(strided_span_len(256, 128, 2), Some(256 + 128));
        assert!(strided_span_len(0, 0, 1).is_none());
        let s = classify_span_arm64e(0, 16);
        assert_eq!(span_first_page_slot(&s, 4), Some(0));
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
        let mut cache = Cache::new();
        let t1 = translate_root(&r, &ARM64E_GEOMETRY, 1, 1, 0x100, Some(&mut cache));
        assert_eq!(t1.cache_status, CacheStatus::MissInserted);
        let t2 = translate_root(&r, &ARM64E_GEOMETRY, 1, 1, 0x200, Some(&mut cache));
        assert_eq!(t2.cache_status, CacheStatus::Hit);
        assert_eq!(t2.gpa, ((5u64) << PAGE_SHIFT_ARM64E) + 0x200);
    }

    #[test]
    fn task_root_and_walk() {
        let mut r = MapReader::new();
        let dir_gpa = (2u64) << PAGE_SHIFT_ARM64E;
        r.put_u32(dir_gpa + DIRECTORY_ROOT_PFN as u64, 1);
        r.put_u32(dir_gpa + DIRECTORY_DEPTH as u64, 1);
        let table_gpa = (1u64) << PAGE_SHIFT_ARM64E;
        r.put_u32(table_gpa, 7);
        let task = Task {
            active: true,
            directory_pfn: 2,
        };
        let root = read_task_root(&r, &task, &ARM64E_GEOMETRY).unwrap();
        assert_eq!(root.root_pfn, 1);
        assert_eq!(root.depth, 1);

        let mut chunks = Vec::new();
        walk_span_task(&r, &task, &ARM64E_GEOMETRY, 0x10, 0x20, None, |c| {
            chunks.push(*c);
            true
        })
        .unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].gpa, ((7u64) << PAGE_SHIFT_ARM64E) + 0x10);
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

    #[test]
    fn property_span_page_count_arm64e() {
        // Fuzz-ish: page counts for boundary addresses.
        for off in [0u64, 1, PAGE_SIZE_ARM64E as u64 - 1] {
            for len in [
                1u64,
                2,
                PAGE_SIZE_ARM64E as u64,
                PAGE_SIZE_ARM64E as u64 + 1,
            ] {
                let s = classify_span_arm64e(off, len);
                if s.status == ResolveStatus::Ok && len > 0 {
                    assert!(s.page_count >= 1);
                    assert_eq!(span_page_count_arm64e(off, len) as u64, s.page_count);
                }
            }
        }
    }
}
