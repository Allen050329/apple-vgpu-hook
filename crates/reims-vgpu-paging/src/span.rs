//! Resolving a byte span to the guest pages under it.

use crate::resolve::{read_task_root, translate_root_run, Geometry, ResolveStatus, Task};
use reims_vgpu_wire::mem::GuestMemory;

/// How many guest pages `[gva, gva+span)` touches, given `page_size`.
///
/// The `gva % page_size` term is the whole content: a span that starts
/// mid-page reaches one page further than its length alone implies. Callers
/// compare a walk's result against this to decide whether the *whole* span
/// resolved, and getting it wrong reads as "fully covered" for exactly the
/// windows that straddle a page boundary — which is most of them.
pub fn pages_spanned(gva: u64, span: u64, page_size: u64) -> u64 {
    ((gva % page_size) + span).div_ceil(page_size)
}

/// Every guest page of `[gva, gva + span)` under `task`'s page table, in
/// ascending order, as its **page-aligned** GPA.
///
/// The one spelling of "walk this span". Four rails in the device need it and
/// each used to open with the same five steps — pick the geometry, build the
/// task, read the root, refuse a root or depth of zero, count the pages — with
/// the refusals written out by hand at each one. That is the shape where a
/// missing term hides: three of the four carried the zero-root guard and the
/// fourth did not, which is correct for that one and was impossible to see
/// without reading all four together.
///
/// # Two kinds of failure, and why they are returned differently
///
/// A **setup** failure — an unusable geometry, an inactive task, an unreadable
/// directory, a zero root or a zero depth — means no page of the span can
/// resolve, and it comes back as `Err`. The visitor is not called at all, so a
/// caller cannot mistake "nothing resolved" for "the span was empty".
///
/// A **per-page** failure reaches the visitor as its own `Err`, because which
/// page failed is the finding: a caller checking a cached page list against the
/// live table needs the position, and one that is merely reading needs to stop
/// there. Walking on past it is the visitor's choice, exactly as with a
/// resolved page.
///
/// The zero-root and zero-depth refusals are the reason this returns a
/// `Result` at all. [`translate_root_run`] answers both by visiting nothing,
/// which is indistinguishable at the call site from a span that resolved
/// cleanly and had no pages — so every caller that reads bytes has to turn them
/// into a refusal, and now does it here once.
pub fn walk_span(
    mem: &dyn GuestMemory,
    geometry: Geometry,
    task: &Task,
    gva: u64,
    span: u64,
    visit: &mut dyn FnMut(u64, Result<u64, ResolveStatus>) -> bool,
) -> Result<(), ResolveStatus> {
    if geometry.validate().is_err() {
        return Err(ResolveStatus::ErrUnsupportedGeometry);
    }
    let root = read_task_root(mem, task, geometry)?;
    if root.root_pfn == 0 {
        return Err(ResolveStatus::ErrZeroRootPfn);
    }
    if root.depth == 0 {
        return Err(ResolveStatus::ErrZeroDepth);
    }
    if span == 0 {
        return Ok(());
    }
    let page = geometry.page_size();
    let mask = geometry.page_offset_mask();
    // Walk from the span's first page rather than from `gva`, so every page's
    // answer is its own base. The run walker carries the starting offset onto
    // every page it reports, which a caller reading bytes has to undo.
    translate_root_run(
        mem,
        geometry,
        root.root_pfn,
        root.depth,
        gva & !mask,
        pages_spanned(gva, span, page),
        &mut |index, r| visit(index, r.map(|gpa| gpa & !mask)),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    use reims_vgpu_wire::mem::SliceMemory;
    use reims_vgpu_wire::page_table::{
        Builder, DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN, PTE_SIZE, X86_64,
    };

    const IMAGE: usize = Builder::image_len(X86_64, 16);

    /// Assemble a task whose directory names `root_pfn` and `depth`, in a page
    /// the builder carves for it.
    ///
    /// The directory's two fields are byte offsets and `poke_entry` indexes by
    /// word, so the two are divided rather than restated — a directory laid out
    /// by hand here could disagree with the one `read_task_root` reads.
    fn directory(b: &mut Builder<'_>, root_pfn: u32, depth: u32) -> Task {
        let dir = b.alloc_page();
        b.poke_entry(dir, (DIRECTORY_ROOT_PFN / PTE_SIZE as u64) as u32, root_pfn);
        b.poke_entry(dir, (DIRECTORY_DEPTH / PTE_SIZE as u64) as u32, depth);
        Task {
            active: true,
            directory_pfn: dir,
        }
    }

    /// The span walk reports one page-aligned GPA per page, in order, and the
    /// offset the span starts at decides how many pages that is.
    #[test]
    fn a_span_walk_reports_each_of_its_pages_once_and_page_aligned() {
        let g = X86_64;
        let page = g.page_size();
        let mut buf = [0u8; IMAGE];
        let mut b = Builder::new(g, &mut buf);
        let root = b.map(1, 0, 0x40);
        b.map_into(root, 1, 1, 0x41);
        b.map_into(root, 1, 2, 0x42);
        let task = directory(&mut b, root, 1);
        let mem = SliceMemory::new(b.bytes());

        // Starting mid-page, a two-page span reaches three pages — the term
        // `pages_spanned` exists for, driven through the walk rather than
        // against the arithmetic alone.
        let mut seen = Vec::new();
        walk_span(&mem, g, &task, page / 2, 2 * page, &mut |i, r| {
            seen.push((i, r));
            true
        })
        .unwrap();
        let got: Vec<u64> = seen.iter().map(|(_, r)| r.unwrap()).collect();
        assert_eq!(
            got,
            [0x40u64 * page, 0x41 * page, 0x42 * page],
            "each page's own base, not the first page's offset"
        );
        assert_eq!(
            seen.iter().map(|(i, _)| *i).collect::<Vec<_>>(),
            [0, 1, 2],
            "indexed from the span's first page, in order"
        );
    }

    /// A setup refusal is returned and visits nothing, so a caller cannot read
    /// it as a span that resolved and had no pages.
    ///
    /// The zero-root and zero-depth arms are the ones this function exists for:
    /// the run walker answers both by visiting nothing, which is exactly what a
    /// clean empty span looks like.
    #[test]
    fn a_setup_refusal_is_returned_rather_than_visited() {
        let g = X86_64;
        let mut buf = [0u8; IMAGE];
        let mut b = Builder::new(g, &mut buf);
        let root = b.map(1, 0, 0x40);
        let live = directory(&mut b, root, 1);
        let zero_root = directory(&mut b, 0, 1);
        let zero_depth = directory(&mut b, root, 0);
        let mem = SliceMemory::new(b.bytes());

        for (task, want) in [
            (zero_root, ResolveStatus::ErrZeroRootPfn),
            (zero_depth, ResolveStatus::ErrZeroDepth),
            (
                Task {
                    active: false,
                    directory_pfn: live.directory_pfn,
                },
                ResolveStatus::ErrInactiveTask,
            ),
            (
                Task {
                    active: true,
                    directory_pfn: 0,
                },
                ResolveStatus::ErrNoDirectory,
            ),
        ] {
            let mut visited = 0;
            let r = walk_span(&mem, g, &task, 0, g.page_size(), &mut |_, _| {
                visited += 1;
                true
            });
            assert_eq!(r, Err(want));
            assert_eq!(visited, 0, "{want:?} must visit nothing");
        }

        // And a geometry off both pathways is refused before the task is read
        // at all, so a bad page shift cannot walk a tree at the wrong stride.
        let bad = Geometry {
            page_shift: 13,
            max_depth: g.max_depth,
        };
        assert_eq!(
            walk_span(&mem, bad, &live, 0, 4096, &mut |_, _| true),
            Err(ResolveStatus::ErrUnsupportedGeometry)
        );
    }

    /// A page that does not resolve reaches the visitor as its own refusal,
    /// carrying its position, and the walk is the visitor's to stop.
    #[test]
    fn an_unresolved_page_reaches_the_visitor_with_its_position() {
        let g = X86_64;
        let page = g.page_size();
        let mut buf = [0u8; IMAGE];
        let mut b = Builder::new(g, &mut buf);
        // Page 1 of the span is left unmapped between two that resolve.
        let root = b.map(1, 0, 0x40);
        b.map_into(root, 1, 2, 0x42);
        let task = directory(&mut b, root, 1);
        let mem = SliceMemory::new(b.bytes());

        let mut seen = Vec::new();
        walk_span(&mem, g, &task, 0, 3 * page, &mut |i, r| {
            seen.push((i, r.is_ok()));
            true
        })
        .unwrap();
        assert_eq!(seen, [(0, true), (1, false), (2, true)]);

        // The visitor stops the walk at the hole when it wants to.
        let mut count = 0;
        walk_span(&mem, g, &task, 0, 3 * page, &mut |_, r| {
            count += 1;
            r.is_ok()
        })
        .unwrap();
        assert_eq!(count, 2, "stopped at the page that refused");
    }

    /// A zero-length span resolves to no pages, rather than to the one its
    /// start address sits in.
    #[test]
    fn a_zero_length_span_covers_no_pages() {
        let g = X86_64;
        let mut buf = [0u8; IMAGE];
        let mut b = Builder::new(g, &mut buf);
        let root = b.map(1, 0, 0x40);
        let task = directory(&mut b, root, 1);
        let mem = SliceMemory::new(b.bytes());

        let mut visited = 0;
        walk_span(&mem, g, &task, 0x800, 0, &mut |_, _| {
            visited += 1;
            true
        })
        .unwrap();
        assert_eq!(visited, 0);
    }

    /// A span's page count is decided by where it *starts*, not only by how
    /// long it is.
    ///
    /// The device's rails compare a walk's page count against this to decide
    /// whether the whole span resolved. Drop the offset term and a window that
    /// straddles a page boundary — which is most of them, since a texture row
    /// rarely starts page-aligned — reports fully covered while missing its
    /// last page. The gather then hands the GPU a short buffer, which is a
    /// wrong frame.
    #[test]
    fn pages_spanned_counts_the_page_the_offset_pushes_a_span_into() {
        const PAGE: u64 = 4096;
        // Page-aligned: exactly what the length implies.
        assert_eq!(pages_spanned(0, PAGE, PAGE), 1);
        assert_eq!(pages_spanned(PAGE * 7, PAGE * 3, PAGE), 3);
        // Offset by one byte: the same length now reaches one page further.
        assert_eq!(pages_spanned(1, PAGE, PAGE), 2);
        assert_eq!(pages_spanned(PAGE * 7 + 1, PAGE * 3, PAGE), 4);
        // A span wholly inside one page stays at one, wherever it starts.
        assert_eq!(pages_spanned(PAGE - 1, 1, PAGE), 1);
        // …and one byte longer crosses.
        assert_eq!(pages_spanned(PAGE - 1, 2, PAGE), 2);
        // The arm64 pathway's 16 KiB pages take the same rule.
        assert_eq!(pages_spanned(16384 * 3 + 5, 16384, 16384), 2);
        // A zero span touches nothing.
        assert_eq!(pages_spanned(0, 0, PAGE), 0);
    }
}
