//! Task GVA resolver (port of `host/utils/reims-vgpu-gva-resolve`).
//!
//! The page-table format and the descent live in
//! [`reims_vgpu_wire::page_table`], which owns them — [`Geometry`] and the two
//! pathway constants are re-exports, so there is no second declaration to
//! drift. What this module adds is the device's side of the walk: task-root
//! reads, the typed refusal statuses the failure channel reports, and the run
//! form the span readers use.

use reims_vgpu_wire::mem as wire_mem;
use reims_vgpu_wire::page_table as wire_page_table;

pub use wire_page_table::{Geometry, ARM64E as ARM64E_GEOMETRY, X86_64 as X86_64_GEOMETRY};

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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Translation {
    pub status: ResolveStatus,
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

pub fn read_task_root(
    reader: &dyn PhysReader,
    task: &Task,
    geometry: Geometry,
) -> Result<TaskRoot, ResolveStatus> {
    if geometry.validate().is_err() {
        return Err(ResolveStatus::ErrUnsupportedGeometry);
    }
    if !task.active {
        return Err(ResolveStatus::ErrInactiveTask);
    }
    if task.directory_pfn == 0 {
        return Err(ResolveStatus::ErrNoDirectory);
    }
    let mem = PhysAsGuestMemory(reader);
    // Geometry and the zero directory are checked above with this module's own
    // statuses, so the only failure left for the wire read is the read itself.
    let (root_pfn, depth) =
        wire_page_table::read_directory(&mem, geometry, task.directory_pfn)
            .map_err(|_| ResolveStatus::ErrDirectoryRead)?;
    Ok(TaskRoot {
        directory_pfn: task.directory_pfn,
        root_pfn,
        depth,
    })
}

pub fn translate_root(
    reader: &dyn PhysReader,
    geometry: Geometry,
    root_pfn: u32,
    depth: u32,
    gva: u64,
) -> Translation {
    let mut out = Translation {
        gva,
        root_pfn,
        depth,
        ..Default::default()
    };
    if geometry.validate().is_err() {
        out.status = ResolveStatus::ErrUnsupportedGeometry;
        return out;
    }
    out.gva_page_index = gva >> geometry.page_shift;

    // The descent itself lives in `reims_vgpu_wire::page_table`, which owns the
    // format. Keeping it there means there is one declaration rather than two
    // that could drift apart silently, and the tree walk gets exercised by that
    // crate's tests as well as by this module's.
    let mem = PhysAsGuestMemory(reader);
    match wire_page_table::walk(&mem, geometry, root_pfn, depth, gva) {
        Ok(w) => {
            // On success the walker reports the deepest level it read, which is
            // where the loop this replaced left these fields.
            out.level = depth - 1;
            out.entry_index = (w.page_index & geometry.index_mask()) as u32;
            out.raw_pte = w.raw_pte;
            out.status = ResolveStatus::Ok;
            out.leaf_pfn = w.leaf_pfn;
            out.gpa_page = w.addr_page;
            out.gpa = w.addr;
        }
        Err(f) => {
            out.level = f.level;
            out.entry_index = f.entry_index;
            out.raw_pte = f.raw_pte;
            out.status = resolve_status_of(f.error);
        }
    }
    out
}

/// Translate a run of consecutive pages under one root, calling `visit` with
/// each page's GPA or the walk's refusal for a page the table cannot
/// translate.
///
/// The run form of [`translate_root`], and it exists for one reason: the
/// per-page form re-reads every upper level of the tree for every page, and a
/// run visits each page index exactly once. A 1080p surface's licence check
/// walks 2 025 consecutive pages; the upper levels of all of them are the same
/// three or four entries, and the descent —
/// [`wire_page_table::walk_run`] — reads each shared entry once.
///
/// The visitor stops the run by answering `false`. It is not called at all when
/// the root or geometry is unusable, so a caller must compare what it saw
/// against what it expected rather than reading a quiet return as agreement —
/// the same contract [`translate_root`] has by returning a status.
pub fn translate_root_run(
    reader: &dyn PhysReader,
    geometry: Geometry,
    root_pfn: u32,
    depth: u32,
    first_gva: u64,
    pages: u64,
    visit: &mut dyn FnMut(u64, Result<u64, ResolveStatus>) -> bool,
) {
    if geometry.validate().is_err() || root_pfn == 0 || depth == 0 {
        return;
    }
    let mem = PhysAsGuestMemory(reader);
    wire_page_table::walk_run(
        &mem,
        geometry,
        root_pfn,
        depth,
        first_gva,
        pages,
        &mut |index, walked| {
            visit(
                index,
                walked
                    .map(|w| w.addr)
                    .map_err(|f| resolve_status_of(f.error)),
            )
        },
    );
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::contract::gva::{pfn_to_gpa, DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
    use crate::model::PAGE_SHIFT_ARM64E;
    use reims_vgpu_wire::page_table::{PTE_FLAG_MASK, PTE_SIZE};
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
        assert!(ARM64E_GEOMETRY.validate().is_ok());
        assert!(X86_64_GEOMETRY.validate().is_ok());
        let mut bad = ARM64E_GEOMETRY;
        bad.page_shift = 13;
        assert!(bad.validate().is_err());
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
        let root = read_task_root(&r, &task, ARM64E_GEOMETRY).unwrap();
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
        let t = translate_root(&r, ARM64E_GEOMETRY, 1, 1, 0x100);
        assert_eq!(t.status, ResolveStatus::Ok);
        assert_eq!(t.leaf_pfn, 5);
        assert_eq!(t.gpa, ((5u64) << PAGE_SHIFT_ARM64E) + 0x100);
    }

    /// The run form answers page for page what the single form answers, and
    /// carries the same refusal for a page that does not translate.
    #[test]
    fn a_run_answers_as_the_single_form_does_and_keeps_the_refusal() {
        let mut r = MapReader::new();
        let table_gpa = (1u64) << PAGE_SHIFT_ARM64E;
        // Pages 0 and 2 map; page 1 is absent (entry reads zero, which the
        // reader models as an unreadable word — TableRead rather than
        // NotPresent, and either way a refusal the visitor must see).
        r.put_u32(table_gpa, 5);
        r.put_u32(table_gpa + 2 * PTE_SIZE as u64, 7);
        let mut seen = Vec::new();
        translate_root_run(&r, ARM64E_GEOMETRY, 1, 1, 0x100, 3, &mut |i, res| {
            seen.push((i, res));
            true
        });
        assert_eq!(seen.len(), 3);
        for (i, res) in &seen {
            let single = translate_root(
                &r,
                ARM64E_GEOMETRY,
                1,
                1,
                ((*i) << PAGE_SHIFT_ARM64E) + 0x100,
            );
            match res {
                Ok(gpa) => {
                    assert_eq!(single.status, ResolveStatus::Ok);
                    assert_eq!(*gpa, single.gpa);
                }
                Err(status) => assert_eq!(*status, single.status),
            }
        }
        assert!(seen[1].1.is_err(), "the unmapped page carries its refusal");
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
        let t = translate_root(&r, g, TABLE_PFN[0], 4, gva);

        assert_eq!(t.status, ResolveStatus::Ok);
        assert_eq!(t.leaf_pfn, LEAF_PFN);
        assert_eq!(t.gpa, ((LEAF_PFN as u64) << g.page_shift) + PAGE_OFF);
        assert_eq!(t.gpa_page, (LEAF_PFN as u64) << g.page_shift);
        assert_eq!(t.level, 3, "the last level walked");
        assert_eq!(t.entry_index as u64, IDX[3]);
        assert_eq!(t.raw_pte, PTE_FLAG_MASK | LEAF_PFN);
    }

    /// No address a walk can form overflows, at any geometry the wire crate's
    /// `validate` accepts.
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
            assert!(g.validate().is_ok());
            // The widest table or leaf base a `u32` PFN can name.
            let max_base = pfn_to_gpa(u32::MAX, g.page_shift);
            let max_entry_off = g.index_mask() * (PTE_SIZE as u64);
            let max_page_off = g.page_offset_mask();
            assert!(max_base.checked_add(max_entry_off).is_some());
            assert!(max_base.checked_add(max_page_off).is_some());
        }

        // And nothing wider is accepted. The shift alone decides this, so the
        // depth bound is left at the shared maximum on purpose.
        for shift in 0..64u32 {
            let g = Geometry {
                page_shift: shift,
                max_depth: ARM64E_GEOMETRY.max_depth,
            };
            if g.validate().is_ok() {
                assert!(
                    shift == PAGE_SHIFT_ARM64E || shift == X86_64_GEOMETRY.page_shift,
                    "unexpected page shift accepted"
                );
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
            read_task_root(&r, &task, ARM64E_GEOMETRY).unwrap_err(),
            ResolveStatus::ErrInactiveTask
        );
        let t = translate_root(&r, ARM64E_GEOMETRY, 0, 1, 0);
        assert_eq!(t.status, ResolveStatus::ErrZeroRootPfn);
    }
}
