//! The guest GPU page table, and the walk that resolves a GVA through it.
//!
//! # Provenance
//!
//! Unlike [`crate::ops`], this layout is not derived by perturbing a serializer,
//! and it cannot be: no serializer record carries a page table, so there is no
//! fixture that could pin one. It comes from the device contract, checked field
//! by field, and it is what the device's GVA resolution runs on — a wrong
//! constant anywhere in it would fail to resolve anything at all.
//!
//! The format:
//!
//! - A node holds an array of four-byte entries indexed directly by the index,
//!   so an entry is [`PTE_SIZE`] bytes.
//! - Bit 31 is a flag, [`PTE_FLAG_MASK`]. The frame number is the remaining
//!   [`PTE_PFN_MASK`], used raw with no further shift.
//! - **Zero is the sole not-present encoding**: an absent entry reads zero, and
//!   a removed one is cleared back to zero.
//! - A frame number never has bit 31 set, and physical page zero is never
//!   mapped.
//!
//! Those last two points are why [`WalkError::MalformedPte`] is a distinct error
//! rather than a pedantic split of [`WalkError::NotPresent`]. A working guest
//! cannot produce a nonzero entry whose frame-number field is zero. Reading one
//! means the page holding the table was corrupted, and collapsing the two arms
//! would discard that signal to save a branch.
//!
//! The fan-out is 1024 entries per node and byte lengths convert to pages with a
//! 12-bit shift. Both are the x86 pathway's values and both agree with
//! [`X86_64`] below. See [`Geometry::index_bits`] for why those two numbers are
//! not independent, and so why deriving one pathway's does not leave the other
//! one guessed.
//!
//! # Relationship to `reims_vgpu::contract::gva_resolve`
//!
//! That module walks the same tree and reached the same constants
//! independently, which is why the agreement is worth something — including on
//! the subtle part, the two-arm split on a zero PFN.
//!
//! What stays there is the part that is not byte interpretation: the translation
//! cache, task lookup, and the device's typed refusal channel. This module owns
//! the tree and nothing else.

use crate::mem::GuestMemory;

/// Byte width of one page-table entry.
///
/// The node's entry array is indexed with a four-byte scale.
pub const PTE_SIZE: u32 = 4;

/// Bit 31, a flag carried alongside the frame number.
///
/// This module never interprets it; it is preserved in [`Walk::raw_pte`] so a
/// caller that learns its meaning does not have to re-read the entry.
pub const PTE_FLAG_MASK: u32 = 0x8000_0000;

/// Bits `[30:0]`, the page frame number, used raw and unshifted.
pub const PTE_PFN_MASK: u32 = 0x7fff_ffff;

/// Offset of the root page number within a task's directory page.
pub const DIRECTORY_ROOT_PFN: u64 = 0x00;

/// Offset of the tree depth within a task's directory page.
///
/// Read per task rather than assumed. The x86 guest has only ever been observed
/// to say 3, but the field exists and a hardcoded depth would be a guess.
pub const DIRECTORY_DEPTH: u64 = 0x04;

/// Upper bound on tree depth, as a sanity bound rather than the depth itself.
pub const MAX_DEPTH: u32 = 4;

/// Page-table shape for one guest pathway.
///
/// Only two numbers are stored. Everything else about the shape — entries per
/// table, index mask, page size — is derived, because they are not independent:
/// a node is exactly one page of four-byte entries, so the fan-out is fixed by
/// the page size. Storing them separately invites a struct whose fields
/// disagree, which is what [`Geometry::validate`] would then have to catch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Geometry {
    /// Guest page shift: 12 on x86_64, 14 on arm64e. Never defaulted.
    pub page_shift: u32,
    /// Bound on the depth read from the task directory.
    pub max_depth: u32,
}

/// x86_64 macOS guest: 4 KiB pages, 1024 entries per node, ten index bits.
pub const X86_64: Geometry = Geometry {
    page_shift: 12,
    max_depth: MAX_DEPTH,
};

/// arm64e macOS guest: 16 KiB pages, 4096 entries per node, twelve index bits.
pub const ARM64E: Geometry = Geometry {
    page_shift: 14,
    max_depth: MAX_DEPTH,
};

impl Geometry {
    /// Bytes per guest page.
    #[inline]
    pub const fn page_size(self) -> u64 {
        1u64 << self.page_shift
    }

    /// Mask selecting the byte offset within a page.
    #[inline]
    pub const fn page_offset_mask(self) -> u64 {
        self.page_size() - 1
    }

    /// Index bits per level.
    ///
    /// A node is one page of four-byte entries, so it holds
    /// `page_size / 4 == 2^(page_shift - 2)` of them. The `- 2` is
    /// `log2(PTE_SIZE)` and is the whole reason this is derived rather than
    /// stored: x86's ten bits and arm64e's twelve are both this expression, and
    /// the walk masks each index to this width.
    #[inline]
    pub const fn index_bits(self) -> u32 {
        self.page_shift - 2
    }

    /// Entries in one node.
    #[inline]
    pub const fn entries_per_table(self) -> u64 {
        1u64 << self.index_bits()
    }

    /// Mask selecting one level's index out of a page index.
    #[inline]
    pub const fn index_mask(self) -> u64 {
        self.entries_per_table() - 1
    }

    /// Guest page frame number to guest address.
    #[inline]
    pub const fn pfn_to_addr(self, pfn: u32) -> u64 {
        (pfn as u64) << self.page_shift
    }

    /// Reject a shape this walk cannot execute.
    ///
    /// The page shift is checked against the two pathways rather than against a
    /// range, because a third value would not be an untested configuration — it
    /// would mean the geometry was inferred from something other than the
    /// pathway, and every constant derived from it would be suspect.
    pub const fn validate(self) -> Result<(), WalkError> {
        if self.page_shift != 12 && self.page_shift != 14 {
            return Err(WalkError::UnsupportedGeometry);
        }
        if self.max_depth == 0 || self.max_depth > MAX_DEPTH {
            return Err(WalkError::UnsupportedGeometry);
        }
        Ok(())
    }
}

/// Why a walk stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WalkError {
    /// The [`Geometry`] is not one this walk can execute.
    UnsupportedGeometry,
    /// The task named no root page.
    ZeroRootPfn,
    /// The task's directory reported depth zero.
    ZeroDepth,
    /// The task's directory reported a depth past [`Geometry::max_depth`].
    DepthTooDeep,
    /// A page holding part of the tree could not be read.
    TableRead,
    /// The entry was zero: the guest has not mapped this address.
    ///
    /// Expected control flow, not a device defect — see the module docs.
    NotPresent,
    /// The entry was nonzero but named PFN zero.
    ///
    /// A working guest cannot produce this: a frame number never carries bit 31,
    /// and physical page zero is never mapped. Reading one means the table's own
    /// page is corrupt.
    MalformedPte,
}

/// A failed walk, with the position it failed at.
///
/// The position is carried because the device reports it: which level, which
/// entry, and the raw word read there is the difference between "this address
/// is not mapped" and a diagnosable corruption.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WalkFailure {
    pub error: WalkError,
    /// Level the walk stopped at, zero-based from the root.
    pub level: u32,
    /// Index within that level's node.
    pub entry_index: u32,
    /// The entry as read, before masking. Zero if the read itself failed.
    pub raw_pte: u32,
}

/// A resolved address.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Walk {
    /// Page frame the walk arrived at.
    pub leaf_pfn: u32,
    /// Address of that page.
    pub addr_page: u64,
    /// Address of the requested byte within it.
    pub addr: u64,
    /// Page index of the input address.
    pub page_index: u64,
    /// The leaf entry as read, before masking, so a caller can read
    /// [`PTE_FLAG_MASK`] without walking again.
    pub raw_pte: u32,
}

/// Read a task's root page number and tree depth from its directory page.
///
/// Both come off the directory rather than from constants: the depth is a field
/// the guest writes, and treating it as known is how a device ends up walking
/// the wrong number of levels when a guest changes.
pub fn read_directory<M: GuestMemory>(
    mem: &M,
    geometry: Geometry,
    directory_pfn: u32,
) -> Result<(u32, u32), WalkError> {
    geometry.validate()?;
    if directory_pfn == 0 {
        return Err(WalkError::ZeroRootPfn);
    }
    let base = geometry.pfn_to_addr(directory_pfn);
    let root_pfn = mem
        .u32_at(base + DIRECTORY_ROOT_PFN)
        .ok_or(WalkError::TableRead)?;
    let depth = mem
        .u32_at(base + DIRECTORY_DEPTH)
        .ok_or(WalkError::TableRead)?;
    Ok((root_pfn, depth))
}

/// Walk `gva` from `root_pfn` down `depth` levels.
///
/// Every level reads one entry and descends into the page frame it names; the
/// frame the last level names is the leaf. `depth` is validated to be at least
/// one, so the loop always runs and the returned `leaf_pfn` always came from an
/// entry rather than from `root_pfn`.
pub fn walk<M: GuestMemory>(
    mem: &M,
    geometry: Geometry,
    root_pfn: u32,
    depth: u32,
    gva: u64,
) -> Result<Walk, WalkFailure> {
    let fail = |error| WalkFailure {
        error,
        level: 0,
        entry_index: 0,
        raw_pte: 0,
    };
    geometry.validate().map_err(fail)?;
    if root_pfn == 0 {
        return Err(fail(WalkError::ZeroRootPfn));
    }
    if depth == 0 {
        return Err(fail(WalkError::ZeroDepth));
    }
    if depth > geometry.max_depth {
        return Err(fail(WalkError::DepthTooDeep));
    }

    let page_index = gva >> geometry.page_shift;
    let page_off = gva & geometry.page_offset_mask();
    let mut current_pfn = root_pfn;
    let mut raw_pte = 0;

    for level in 0..depth {
        // The root indexes by the most significant slice of the page index, so
        // the shift shrinks as the walk descends.
        let shift = (depth - 1 - level) * geometry.index_bits();
        let entry_index = ((page_index >> shift) & geometry.index_mask()) as u32;
        let entry_addr = geometry.pfn_to_addr(current_pfn) + (entry_index as u64) * PTE_SIZE as u64;

        let at = |error, raw_pte| WalkFailure {
            error,
            level,
            entry_index,
            raw_pte,
        };
        let pte = mem.u32_at(entry_addr).ok_or(at(WalkError::TableRead, 0))?;
        raw_pte = pte;

        let next_pfn = pte & PTE_PFN_MASK;
        if next_pfn == 0 {
            // An absent entry is written as zero, and a frame number never
            // carries bit 31, so these two cases have different causes.
            return Err(at(
                if pte == 0 {
                    WalkError::NotPresent
                } else {
                    WalkError::MalformedPte
                },
                pte,
            ));
        }
        current_pfn = next_pfn;
    }

    let addr_page = geometry.pfn_to_addr(current_pfn);
    Ok(Walk {
        leaf_pfn: current_pfn,
        addr_page,
        addr: addr_page + page_off,
        page_index,
        raw_pte,
    })
}

/// Walk a run of consecutive pages, re-reading only the levels whose entry
/// index changed.
///
/// A run of `pages` pages starting at `first_gva` shares every level of the tree
/// except the deepest for `1 << index_bits` pages at a time, so walking each one
/// with [`walk`] re-reads the same upper entries `depth - 1` times per page. On
/// a guest with a four-level tree that is four guest-memory reads per page where
/// one is needed, and the caller that motivates this — a licence check over a
/// 1080p surface's page list — pays it 2 025 times a flush.
///
/// The visitor is called once per page in ascending order with the page's index
/// within the run, and stops the walk by answering `false`. A failure is
/// reported for the page it happened on and does not stop the run: a caller
/// checking a cached list against the live table needs to know *which* pages
/// disagree, and a walk that stopped at the first would report a shorter
/// disagreement than there is.
///
/// # What the reuse assumes
///
/// That the tree does not change under the walk. It is the same assumption
/// [`walk`] makes within one descent, extended to the run — a guest that
/// rewrites an upper entry midway through is a guest editing a page table this
/// device is reading, and neither form of walk can be atomic against that.
/// A caller needing a coherent snapshot needs one from the hypervisor, not from
/// a re-read here.
pub fn walk_run<M: GuestMemory>(
    mem: &M,
    geometry: Geometry,
    root_pfn: u32,
    depth: u32,
    first_gva: u64,
    pages: u64,
    visit: &mut dyn FnMut(u64, Result<Walk, WalkFailure>) -> bool,
) {
    let fail = |error| WalkFailure {
        error,
        level: 0,
        entry_index: 0,
        raw_pte: 0,
    };
    if let Err(f) = geometry.validate().map_err(fail) {
        visit(0, Err(f));
        return;
    }
    if root_pfn == 0 {
        visit(0, Err(fail(WalkError::ZeroRootPfn)));
        return;
    }
    if depth == 0 {
        visit(0, Err(fail(WalkError::ZeroDepth)));
        return;
    }
    if depth > geometry.max_depth {
        visit(0, Err(fail(WalkError::DepthTooDeep)));
        return;
    }

    // The entry index taken at each level of the previous page's descent, and
    // the frame that entry named. `held` is how many *leading* levels of that
    // record are still true: a level whose index differs invalidates itself and
    // everything under it, which is why one prefix length is enough and a
    // per-level valid bit is not.
    let mut seen_index = [0u32; MAX_DEPTH as usize];
    let mut seen_next = [0u32; MAX_DEPTH as usize];
    let mut held = 0usize;

    let first_page = first_gva >> geometry.page_shift;
    for i in 0..pages {
        let page_index = first_page + i;
        let gva = (page_index << geometry.page_shift) | (first_gva & geometry.page_offset_mask());
        let page_off = gva & geometry.page_offset_mask();
        let mut current_pfn = root_pfn;
        let mut raw_pte = 0u32;
        let mut failure = None;
        for level in 0..depth {
            let shift = (depth - 1 - level) * geometry.index_bits();
            let entry_index = ((page_index >> shift) & geometry.index_mask()) as u32;
            let lv = level as usize;
            if lv < held && seen_index[lv] == entry_index {
                current_pfn = seen_next[lv];
                continue;
            }
            let entry_addr =
                geometry.pfn_to_addr(current_pfn) + (entry_index as u64) * PTE_SIZE as u64;
            let at = |error, raw_pte| WalkFailure {
                error,
                level,
                entry_index,
                raw_pte,
            };
            let Some(pte) = mem.u32_at(entry_addr) else {
                failure = Some(at(WalkError::TableRead, 0));
                break;
            };
            raw_pte = pte;
            let next_pfn = pte & PTE_PFN_MASK;
            if next_pfn == 0 {
                failure = Some(at(
                    if pte == 0 {
                        WalkError::NotPresent
                    } else {
                        WalkError::MalformedPte
                    },
                    pte,
                ));
                break;
            }
            seen_index[lv] = entry_index;
            seen_next[lv] = next_pfn;
            held = lv + 1;
            current_pfn = next_pfn;
        }
        let result = match failure {
            // A failed descent leaves the record describing a tree the walk did
            // not finish reading, so the next page starts from the root.
            Some(f) => {
                held = 0;
                Err(f)
            }
            None => {
                let addr_page = geometry.pfn_to_addr(current_pfn);
                Ok(Walk {
                    leaf_pfn: current_pfn,
                    addr_page,
                    addr: addr_page + page_off,
                    page_index,
                    raw_pte,
                })
            }
        };
        if !visit(i, result) {
            return;
        }
    }
}

/// Builds page tables the way the guest does.
///
/// This exists so tests walk a tree assembled by the format's own rules rather
/// than one hand-written to satisfy the walker — the two agree only if the
/// walker is right. [`Builder::set_entry`] enforces both of the format's write
/// guards, so a test that tries to build a malformed entry fails at the build
/// rather than producing a tree the walker then quietly accepts.
///
/// It carves pages out of a caller-provided buffer rather than allocating, so
/// the crate's no-allocation invariant holds in tests as well as in the decode
/// path. `reims-vgpu`'s tests use it too, which is why it is not `#[cfg(test)]`.
pub struct Builder<'a> {
    geometry: Geometry,
    pages: &'a mut [u8],
    next_pfn: u32,
}

impl<'a> Builder<'a> {
    /// Take a buffer as the guest-physical image.
    ///
    /// Page zero is reserved immediately, so PFN 0 always means "no page" — the
    /// same reservation that makes zero a usable not-present encoding.
    ///
    /// Panics if the buffer is not a whole number of pages, which would let a
    /// frame at the tail be silently short.
    pub fn new(geometry: Geometry, pages: &'a mut [u8]) -> Self {
        let page = geometry.page_size() as usize;
        assert!(
            pages.len() >= page && pages.len().is_multiple_of(page),
            "the image must be a nonzero whole number of {page}-byte pages"
        );
        pages.fill(0);
        Self {
            geometry,
            pages,
            next_pfn: 1,
        }
    }

    /// Number of pages an image must hold for `frames` frames plus the reserved
    /// page zero. Use it to size the buffer passed to [`Builder::new`].
    pub const fn image_len(geometry: Geometry, frames: usize) -> usize {
        (frames + 1) * geometry.page_size() as usize
    }

    /// Claim the next zeroed page and return its frame number.
    pub fn alloc_page(&mut self) -> u32 {
        let pfn = self.next_pfn;
        let end = (pfn as usize + 1) * self.geometry.page_size() as usize;
        assert!(end <= self.pages.len(), "image too small for another frame");
        self.next_pfn += 1;
        pfn
    }

    fn slot(&mut self, node_pfn: u32, index: u32) -> &mut [u8] {
        let at =
            (self.geometry.pfn_to_addr(node_pfn) as usize) + (index as usize) * PTE_SIZE as usize;
        &mut self.pages[at..at + PTE_SIZE as usize]
    }

    /// Write one entry, enforcing the format's two write guards.
    ///
    /// Panics if the slot is already occupied or if `pfn` has bit 31 set —
    /// exactly the two entries a guest cannot produce.
    pub fn set_entry(&mut self, node_pfn: u32, index: u32, pfn: u32, flag: bool) {
        assert_eq!(pfn & PTE_FLAG_MASK, 0, "a PFN never has bit 31 already set");
        let entry = pfn | if flag { PTE_FLAG_MASK } else { 0 };
        let slot = self.slot(node_pfn, index);
        assert_eq!(
            u32::from_le_bytes(slot.try_into().unwrap()),
            0,
            "an entry is only written into an empty slot"
        );
        slot.copy_from_slice(&entry.to_le_bytes());
    }

    /// Write a raw word, bypassing the guards, to synthesize corruption.
    ///
    /// The guest cannot produce these; the point is to prove the walker reports
    /// them rather than descending into them.
    pub fn poke_entry(&mut self, node_pfn: u32, index: u32, raw: u32) {
        self.slot(node_pfn, index)
            .copy_from_slice(&raw.to_le_bytes());
    }

    /// Read an entry's frame number back, for node reuse.
    fn child_of(&mut self, node_pfn: u32, index: u32) -> u32 {
        u32::from_le_bytes(self.slot(node_pfn, index).try_into().unwrap()) & PTE_PFN_MASK
    }

    /// Build a `depth`-level tree mapping `page_index` to `leaf_pfn`.
    ///
    /// Returns the root frame number.
    pub fn map(&mut self, depth: u32, page_index: u64, leaf_pfn: u32) -> u32 {
        let root = self.alloc_page();
        self.map_into(root, depth, page_index, leaf_pfn);
        root
    }

    /// Add a mapping to an existing tree, reusing nodes already present.
    pub fn map_into(&mut self, root: u32, depth: u32, page_index: u64, leaf_pfn: u32) {
        let mut node = root;
        for level in 0..depth {
            let shift = (depth - 1 - level) * self.geometry.index_bits();
            let index = ((page_index >> shift) & self.geometry.index_mask()) as u32;
            if level == depth - 1 {
                self.set_entry(node, index, leaf_pfn, false);
                return;
            }
            node = match self.child_of(node, index) {
                0 => {
                    let child = self.alloc_page();
                    self.set_entry(node, index, child, false);
                    child
                }
                existing => existing,
            };
        }
    }

    /// The assembled guest-physical image.
    pub fn bytes(&self) -> &[u8] {
        self.pages
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mem::SliceMemory;

    #[test]
    fn the_derived_shape_matches_the_contracts_constants() {
        // The width each index is masked to, and four-byte entries filling
        // exactly one page.
        assert_eq!(X86_64.index_bits(), 10);
        assert_eq!(X86_64.index_mask(), 0x3ff);
        assert_eq!(X86_64.entries_per_table(), 1024);
        assert_eq!(
            X86_64.entries_per_table() * PTE_SIZE as u64,
            X86_64.page_size()
        );

        assert_eq!(ARM64E.index_bits(), 12);
        assert_eq!(ARM64E.index_mask(), 0xfff);
        assert_eq!(ARM64E.entries_per_table(), 4096);
        assert_eq!(
            ARM64E.entries_per_table() * PTE_SIZE as u64,
            ARM64E.page_size()
        );

        assert_eq!(PTE_PFN_MASK | PTE_FLAG_MASK, u32::MAX);
        assert_eq!(PTE_PFN_MASK & PTE_FLAG_MASK, 0);
    }

    #[test]
    fn a_geometry_off_either_pathway_is_refused_rather_than_walked() {
        for shift in [0, 11, 13, 15, 16] {
            let g = Geometry {
                page_shift: shift,
                max_depth: MAX_DEPTH,
            };
            assert_eq!(g.validate(), Err(WalkError::UnsupportedGeometry));
        }
        assert_eq!(X86_64.validate(), Ok(()));
        assert_eq!(ARM64E.validate(), Ok(()));
    }

    /// Enough frames for a full-depth tree at either page size. Sized for the
    /// larger page so one buffer type serves both geometries; it stays a whole
    /// number of pages at 4 KiB too, which [`Builder::new`] requires.
    const IMAGE: usize = Builder::image_len(ARM64E, 8);

    #[test]
    fn a_walk_over_a_validly_built_tree_finds_the_leaf() {
        for geometry in [X86_64, ARM64E] {
            for depth in 1..=MAX_DEPTH {
                let mut buf = [0u8; IMAGE];
                let mut b = Builder::new(geometry, &mut buf);
                let leaf = 0x2bc;
                // A page index with a distinct value in every level's slice, so
                // a walk that mixes two levels up cannot land by accident.
                let page_index = (0..depth)
                    .map(|l| ((l as u64) + 1) << (l * geometry.index_bits()))
                    .fold(0, |a, b| a | b);
                let root = b.map(depth, page_index, leaf);
                let mem = SliceMemory::new(b.bytes());

                let off = 0x21;
                let gva = (page_index << geometry.page_shift) | off;
                let w = walk(&mem, geometry, root, depth, gva).expect("mapped");
                assert_eq!(w.leaf_pfn, leaf);
                assert_eq!(w.page_index, page_index);
                assert_eq!(w.addr, geometry.pfn_to_addr(leaf) + off);
            }
        }
    }

    /// `walk_run` and `walk` must answer identically for every page of a run,
    /// including the pages that do not resolve.
    ///
    /// This is the whole of `walk_run`'s contract. It exists only to avoid
    /// re-reading upper levels, so the moment it answers differently from the
    /// walk it optimises it is not an optimisation but a second, weaker walker —
    /// and the way it would fail is by carrying a stale upper level across an
    /// index boundary, which the run below crosses deliberately.
    #[test]
    fn a_run_walk_agrees_with_the_single_walk_on_every_page() {
        for geometry in [X86_64, ARM64E] {
            for depth in 1..=MAX_DEPTH {
                let mut buf = [0u8; IMAGE];
                let mut b = Builder::new(geometry, &mut buf);
                // Two pages whose indices differ above the deepest level, so the
                // run has to notice the upper entry changed, plus their
                // neighbours, which must reuse it.
                let stride = 1u64 << geometry.index_bits();
                let root = b.map(depth, 0, 0x11);
                b.map_into(root, depth, 1, 0x12);
                if depth > 1 {
                    b.map_into(root, depth, stride, 0x21);
                    b.map_into(root, depth, stride + 1, 0x22);
                }
                let mem = SliceMemory::new(b.bytes());

                // Covers both mapped clusters, the hole between them, and the
                // unmapped tail past the second.
                let pages = if depth > 1 { stride + 3 } else { 4 };
                let mut seen = 0u64;
                walk_run(&mem, geometry, root, depth, 0, pages, &mut |i, got| {
                    let gva = i << geometry.page_shift;
                    let want = walk(&mem, geometry, root, depth, gva);
                    match (&got, &want) {
                        (Ok(a), Ok(e)) => assert_eq!(a, e, "page {i} depth {depth}"),
                        (Err(a), Err(e)) => assert_eq!(a, e, "page {i} depth {depth}"),
                        _ => panic!("page {i} depth {depth}: {got:?} vs {want:?}"),
                    }
                    seen += 1;
                    true
                });
                assert_eq!(seen, pages, "every page of the run is visited, in order");
            }
        }
    }

    /// The visitor stops the run by answering `false`, and no page past it is
    /// walked. A caller checking a page list against the live table stops at the
    /// first disagreement it cares about, and a run that kept reading would cost
    /// the guest-memory reads the stop exists to avoid.
    #[test]
    fn a_run_walk_stops_when_the_visitor_says_so() {
        let geometry = X86_64;
        let mut buf = [0u8; IMAGE];
        let mut b = Builder::new(geometry, &mut buf);
        let root = b.map(2, 0, 0x11);
        b.map_into(root, 2, 1, 0x12);
        let mem = SliceMemory::new(b.bytes());

        let mut visited = 0;
        walk_run(&mem, geometry, root, 2, 0, 64, &mut |_, _| {
            visited += 1;
            visited < 2
        });
        assert_eq!(visited, 2);
    }

    #[test]
    fn an_unmapped_address_reports_not_present_rather_than_corruption() {
        let geometry = X86_64;
        let mut buf = [0u8; IMAGE];
        let mut b = Builder::new(geometry, &mut buf);
        let root = b.map(2, 0, 9);
        let mem = SliceMemory::new(b.bytes());

        // Sibling of the mapped entry at the deepest level.
        let gva = 1u64 << geometry.page_shift;
        let f = walk(&mem, geometry, root, 2, gva).unwrap_err();
        assert_eq!(f.error, WalkError::NotPresent);
        assert_eq!(f.level, 1);
        assert_eq!(f.entry_index, 1);
        assert_eq!(f.raw_pte, 0);
    }

    #[test]
    fn a_nonzero_entry_naming_no_page_is_corruption_not_absence() {
        // A frame number never carries bit 31, so the guest cannot write this
        // and the walker must not treat it as a hole.
        let geometry = X86_64;
        let mut buf = [0u8; IMAGE];
        let mut b = Builder::new(geometry, &mut buf);
        let root = b.map(1, 0, 9);
        b.poke_entry(root, 0, PTE_FLAG_MASK);
        let mem = SliceMemory::new(b.bytes());

        let f = walk(&mem, geometry, root, 1, 0).unwrap_err();
        assert_eq!(f.error, WalkError::MalformedPte);
        assert_eq!(f.raw_pte, PTE_FLAG_MASK);
    }

    #[test]
    #[should_panic(expected = "a PFN never has bit 31 already set")]
    fn the_builder_refuses_a_pfn_that_already_has_bit_31_set() {
        // Guards the guard: if `set_entry` stopped enforcing this, the test
        // above would be synthesizing corruption the guest could also write,
        // and `MalformedPte` would stop meaning corruption.
        let mut buf = [0u8; IMAGE];
        let mut b = Builder::new(X86_64, &mut buf);
        let root = b.alloc_page();
        b.set_entry(root, 0, PTE_FLAG_MASK | 1, false);
    }

    #[test]
    #[should_panic(expected = "an entry is only written into an empty slot")]
    fn the_builder_refuses_to_overwrite_a_live_entry() {
        let mut buf = [0u8; IMAGE];
        let mut b = Builder::new(X86_64, &mut buf);
        let root = b.alloc_page();
        b.set_entry(root, 0, 1, false);
        b.set_entry(root, 0, 2, false);
    }

    #[test]
    fn the_flag_bit_survives_to_the_caller_and_never_reaches_the_frame_number() {
        let geometry = X86_64;
        let mut buf = [0u8; IMAGE];
        let mut b = Builder::new(geometry, &mut buf);
        let root = b.alloc_page();
        b.set_entry(root, 0, 0x1234, true);
        let mem = SliceMemory::new(b.bytes());

        let w = walk(&mem, geometry, root, 1, 0).expect("mapped");
        assert_eq!(w.leaf_pfn, 0x1234);
        assert_eq!(w.raw_pte, PTE_FLAG_MASK | 0x1234);
        assert_eq!(w.addr_page, geometry.pfn_to_addr(0x1234));
    }

    #[test]
    fn a_depth_outside_the_bound_is_refused_before_any_page_is_read() {
        let geometry = X86_64;
        let mut buf = [0u8; IMAGE];
        let b = Builder::new(geometry, &mut buf);
        let mem = SliceMemory::new(b.bytes());
        assert_eq!(
            walk(&mem, geometry, 1, 0, 0).unwrap_err().error,
            WalkError::ZeroDepth
        );
        assert_eq!(
            walk(&mem, geometry, 1, MAX_DEPTH + 1, 0).unwrap_err().error,
            WalkError::DepthTooDeep
        );
        assert_eq!(
            walk(&mem, geometry, 0, 1, 0).unwrap_err().error,
            WalkError::ZeroRootPfn
        );
    }

    #[test]
    fn a_table_page_outside_the_image_reports_a_read_failure() {
        let geometry = X86_64;
        let mut buf = [0u8; IMAGE];
        let mut b = Builder::new(geometry, &mut buf);
        let root = b.alloc_page();
        // Point at a frame the image does not contain.
        b.set_entry(root, 0, 0x7fff, false);
        let mem = SliceMemory::new(b.bytes());

        let f = walk(&mem, geometry, root, 2, 0).unwrap_err();
        assert_eq!(f.error, WalkError::TableRead);
        assert_eq!(f.level, 1);
    }

    #[test]
    fn the_directory_supplies_root_and_depth_rather_than_a_constant() {
        let geometry = X86_64;
        let mut buf = [0u8; IMAGE];
        let mut b = Builder::new(geometry, &mut buf);
        let dir = b.alloc_page();
        b.poke_entry(dir, 0, 7); // root pfn
        b.poke_entry(dir, 1, 3); // depth
        let mem = SliceMemory::new(b.bytes());

        assert_eq!(read_directory(&mem, geometry, dir), Ok((7, 3)));
        assert_eq!(
            read_directory(&mem, geometry, 0),
            Err(WalkError::ZeroRootPfn)
        );
    }
}
