//! Host-pointer import-window geometry for `VK_EXT_external_memory_host`.
//!
//! Importing registers (pins) host pages for GPU DMA; importing a whole QEMU
//! RAMBlock VMA would pin all guest RAM for the VM lifetime. These helpers bucket
//! a span into a `HOST_IMPORT_WINDOW_CAP`-bounded, VMA-relative aligned window so
//! the pinning stays bounded while a handful of windows amortize steady state.
//! Pure geometry plus the per-host "which mapping holds this pointer?" query
//! (`/proc/self/maps` on Linux, `mach_vm_region` on macOS) — no `ResourcePools`
//! state — so it lives apart from the pool bookkeeping in the parent module.

/// Importing registers (pins) host pages for GPU DMA — importing the whole
/// QEMU RAMBlock VMA (16 GiB+) pins the guest's entire RAM on the host for
/// the VM lifetime. The driver accepts it, but the host pays gigabytes of
/// locked memory. 1 GiB windows bound the pinning while keeping the
/// amortization: spans bucket into VMA-relative aligned windows, so steady
/// state reuses a handful of windows instead of importing per span.
/// Standing rule (AGENTS.md): never raise this back to whole-VMA.
pub(super) const HOST_IMPORT_WINDOW_CAP: u64 = 1 << 30;

/// Compute the capped import window inside `[vma_base, vma_base+vma_len)`
/// covering `[ptr, end)`: the `HOST_IMPORT_WINDOW_CAP`-aligned bucket
/// (relative to the VMA base) containing `ptr`, extended when the span
/// crosses the bucket edge, clamped to the VMA. A VMA at or under the cap
/// imports whole. `align` (the driver's min imported-host-pointer
/// alignment) divides the result because VMA bounds are page-aligned and
/// the cap is a page multiple; the caller re-checks before importing.
fn capped_import_window(
    vma_base: usize,
    vma_len: u64,
    ptr: usize,
    end: u64,
    align: u64,
) -> (usize, u64) {
    debug_assert!(
        ptr >= vma_base && (ptr as u64) < vma_base as u64 + vma_len,
        "ptr {ptr:#x} must lie inside the VMA {vma_base:#x}+{vma_len:#x}; both \
         callers establish this by finding the mapping that contains it, and \
         without it the bucket subtraction below underflows"
    );
    if vma_len <= HOST_IMPORT_WINDOW_CAP {
        return (vma_base, vma_len);
    }
    let vma_end = vma_base as u64 + vma_len;
    let bucket = (ptr as u64 - vma_base as u64) / HOST_IMPORT_WINDOW_CAP;
    let base = vma_base as u64 + bucket * HOST_IMPORT_WINDOW_CAP;
    let span_end = end.div_ceil(align) * align;
    let win_end = (base + HOST_IMPORT_WINDOW_CAP).max(span_end).min(vma_end);
    (base as usize, win_end - base)
}

/// The import regions that could carry `[ptr, end)`, best first, or the
/// budget's refusal.
///
/// The budget is asked at `align` — the smallest a candidate can be, since both
/// are align-rounded covers of a non-empty span — **before** either candidate is
/// built, and that ordering is the whole point of this function. [`vma_bounds`]
/// reads and parses all of `/proc/self/maps`; on the x86 rail that file is 3572
/// lines and 292 µs per call. Once the pool sits at its cap, every resolve of an
/// unresident span paid that walk only to be declined by a check that never
/// looked at the answer. One boot spent **228 s of 484 s** of total drain time
/// there, across 379 639 misses, every one of them declined for `TotalBytes`.
///
/// Short-circuiting is sound because [`super::host_import_budget`] only gets
/// harder as the candidate grows, so a refusal at `align` is the same refusal at
/// every size a candidate can take. The caller re-asks it per candidate, since
/// passing at `align` says nothing about passing at a gigabyte.
pub(super) fn host_import_candidates(
    region_count: usize,
    imported_bytes: u64,
    ptr: usize,
    end: u64,
    align: u64,
) -> Result<Vec<(usize, u64)>, super::HostImportDecline> {
    super::host_import_budget(region_count, imported_bytes, align)?;
    let mut candidates = Vec::new();
    if let Some((vma_base, vma_len)) = vma_bounds(ptr) {
        candidates.push(capped_import_window(
            vma_base,
            vma_len as u64,
            ptr,
            end,
            align,
        ));
    }
    let span_base = ((ptr as u64) / align * align) as usize;
    let span_len = (end - span_base as u64).div_ceil(align) * align;
    candidates.push((span_base, span_len));
    Ok(candidates)
}

/// Megabytes per [`WindowOccupancy`] bit. A resolve asks for one page run, so
/// anything finer measures page geometry rather than window occupancy.
pub(super) const OCCUPANCY_GRANULE: u64 = 1 << 20;

/// Bits per map: one [`HOST_IMPORT_WINDOW_CAP`] window at [`OCCUPANCY_GRANULE`].
const OCCUPANCY_BITS: usize = (HOST_IMPORT_WINDOW_CAP / OCCUPANCY_GRANULE) as usize;

/// Which megabytes of an import window a resolve has ever asked for.
///
/// The window is 1 GiB but a resolve asks for one page run at a time, so the
/// share of a window that ever carries DMA decides whether a finer window would
/// cover the same working set for fewer pinned bytes. That is the question
/// [`HOST_IMPORT_WINDOW_CAP`]'s doc defers — "a smaller cap **only if** the hot
/// pages prove sparse within the spread" — and the create/evict census cannot
/// answer it, because it counts windows and never their occupancy.
///
/// Kept per *bucket base*, not per live region, so a re-import continues the
/// same map: occupancy is a property of what the guest touches, and zeroing it
/// on eviction would report the thrash rate instead of the working set.
#[derive(Default)]
pub(super) struct WindowOccupancy {
    bits: [u64; OCCUPANCY_BITS / 64],
}

impl WindowOccupancy {
    /// Record that `[offset, offset+len)` of the window carries DMA. Offsets at
    /// or past the map's reach are dropped: a span may extend a window past the
    /// cap (`capped_import_window` widens for a bucket-crossing span), and the
    /// tail beyond the cap belongs to the next bucket's map.
    pub(super) fn mark(&mut self, offset: u64, len: u64) {
        let first = offset / OCCUPANCY_GRANULE;
        let last = offset.saturating_add(len.max(1) - 1) / OCCUPANCY_GRANULE;
        for granule in first..=last.min(OCCUPANCY_BITS as u64 - 1) {
            let g = granule as usize;
            self.bits[g / 64] |= 1u64 << (g % 64);
        }
    }

    /// How many `granule_mib`-sized chunks of the window hold any marked
    /// megabyte — i.e. how many windows of that size a finer resolver would
    /// have had to import to cover the same traffic.
    pub(super) fn chunks_touched(&self, granule_mib: usize) -> usize {
        (0..OCCUPANCY_BITS)
            .step_by(granule_mib)
            .filter(|&start| {
                (start..(start + granule_mib).min(OCCUPANCY_BITS))
                    .any(|g| self.bits[g / 64] & (1u64 << (g % 64)) != 0)
            })
            .count()
    }
}

/// Bounds of the host mapping containing `ptr`, or `None` if it cannot be
/// determined.
///
/// `None` is not benign: without VMA bounds `capped_import_window` has nothing to
/// bucket against, so `host_import_resolve` falls through to its bare
/// aligned-span candidate and imports **one region per span**. On the arm64 guest
/// that is one 16 KiB region per page touched, which walks the 512-region cap in
/// ~128 MiB — two orders of magnitude below the 1 GiB byte cap that is supposed
/// to be the binding constraint — and every import after that declines, leaving
/// fresh surfaces at zeros. See [[failure-logging]] for the boot that showed it.
#[cfg(not(target_os = "macos"))]
fn vma_bounds(ptr: usize) -> Option<(usize, usize)> {
    let maps = std::fs::read_to_string("/proc/self/maps").ok()?;
    vma_bounds_in(&maps, ptr)
}

/// macOS has no `/proc`, so the bounds come from the Mach VM map instead.
///
/// This is a *query about a pointer we already hold*, not a scan: one
/// `mach_vm_region` call asking which region contains `ptr`. Mach coalesces
/// adjacent mappings that share protection, inheritance and behaviour, so the
/// reported region can be **wider** than QEMU's RAMBlock. That is safe here and
/// only here, because the caller never imports the region — it imports the
/// `HOST_IMPORT_WINDOW_CAP` bucket of it containing `ptr`, so a too-wide answer
/// shifts bucket boundaries without ever growing what gets pinned.
#[cfg(target_os = "macos")]
fn vma_bounds(ptr: usize) -> Option<(usize, usize)> {
    // `mach_vm_region` lives in libSystem; `libc` exposes the task port and the
    // Mach integer types but not this call.
    extern "C" {
        // `mach_task_self()` is a C macro over this global; `libc`'s function
        // wrapper for it is deprecated in favour of the `mach2` crate, and
        // taking a dependency to read one `int` is not worth it.
        static mach_task_self_: libc::c_uint;
        fn mach_vm_region(
            target_task: libc::c_uint,
            address: *mut u64,
            size: *mut u64,
            flavor: libc::c_int,
            info: *mut libc::c_int,
            info_count: *mut libc::c_uint,
            object_name: *mut libc::c_uint,
        ) -> libc::c_int;
    }
    // `VM_REGION_BASIC_INFO_64` and the `int`-count of
    // `vm_region_basic_info_data_64_t` (40 bytes / 4), from `mach/vm_region.h`.
    const VM_REGION_BASIC_INFO_64: libc::c_int = 9;
    const VM_REGION_BASIC_INFO_COUNT_64: libc::c_uint = 10;
    const KERN_SUCCESS: libc::c_int = 0;
    const VM_PROT_READ: libc::c_int = 0x01;

    let mut address = ptr as u64;
    let mut size = 0u64;
    let mut info = [0i32; VM_REGION_BASIC_INFO_COUNT_64 as usize];
    let mut count = VM_REGION_BASIC_INFO_COUNT_64;
    let mut object_name: libc::c_uint = 0;
    // SAFETY: all out-params are owned locals of the sizes the flavor declares,
    // and `count` says how many `int`s `info` can hold.
    let kr = unsafe {
        mach_vm_region(
            mach_task_self_,
            &mut address,
            &mut size,
            VM_REGION_BASIC_INFO_64,
            info.as_mut_ptr(),
            &mut count,
            &mut object_name,
        )
    };
    if kr != KERN_SUCCESS || size == 0 {
        return None;
    }
    // `mach_vm_region` returns the region at or *after* the address it is given,
    // so a hit has to be re-confirmed: an unmapped `ptr` yields the next region up.
    let base = usize::try_from(address).ok()?;
    let len = usize::try_from(size).ok()?;
    if ptr < base || ptr >= base.checked_add(len)? {
        return None;
    }
    // Same check as the Linux parser: the GPU must be able to read the pages, so
    // a non-readable region can never be a guest-RAM block. `protection` is the
    // first `int` of `vm_region_basic_info_data_64_t`.
    if info[0] & VM_PROT_READ == 0 {
        return None;
    }
    Some((base, len))
}

/// Pure parser over `/proc/self/maps` content: find the readable mapping
/// containing `ptr`. Line shape: `start-end perms offset dev inode [path]`.
///
/// Compiled on every host so its tests run everywhere, not only on the Linux
/// rows — the parser is where the Linux answer's correctness lives.
#[cfg_attr(
    target_os = "macos",
    allow(dead_code, reason = "compiled for its tests; the caller here is Mach")
)]
fn vma_bounds_in(maps: &str, ptr: usize) -> Option<(usize, usize)> {
    for line in maps.lines() {
        let mut fields = line.split_whitespace();
        let range = fields.next()?;
        let perms = fields.next().unwrap_or("");
        let (start_hex, end_hex) = range.split_once('-')?;
        let start = usize::from_str_radix(start_hex, 16).ok()?;
        let end = usize::from_str_radix(end_hex, 16).ok()?;
        if ptr >= start && ptr < end {
            // The GPU must be able to read the pages; a non-readable VMA
            // (guard region) can never be a guest-RAM block.
            if !perms.starts_with('r') {
                return None;
            }
            return Some((start, end - start));
        }
    }
    None
}

#[cfg(test)]
mod window_occupancy_tests {
    use super::{WindowOccupancy, OCCUPANCY_GRANULE};

    /// The measurement the density line exists to make: a working set of a few
    /// scattered page runs occupies a handful of megabytes of a 1 GiB window,
    /// and the coarser counts say how many windows of each size would carry it.
    #[test]
    fn scattered_runs_report_their_own_spread_not_the_window() {
        let mut occ = WindowOccupancy::default();
        for mib in [0u64, 3, 200, 900] {
            occ.mark(mib * OCCUPANCY_GRANULE, 16 << 10);
        }
        assert_eq!(occ.chunks_touched(1), 4, "four distinct megabytes");
        assert_eq!(
            occ.chunks_touched(4),
            3,
            "MiB 0 and 3 share one 4 MiB chunk"
        );
        assert_eq!(
            occ.chunks_touched(256),
            2,
            "MiB 0/3/200 share one, 900 alone"
        );
    }

    /// A run spanning granules marks all of them, and one that would run past
    /// the window's reach is clipped instead of wrapping into low bits.
    #[test]
    fn spans_mark_every_granule_and_clip_at_the_window() {
        let mut occ = WindowOccupancy::default();
        occ.mark(0, 4 * OCCUPANCY_GRANULE + 1);
        assert_eq!(occ.chunks_touched(1), 5);
        let mut past = WindowOccupancy::default();
        past.mark(1 << 30, 1 << 20);
        assert_eq!(
            past.chunks_touched(1),
            0,
            "past the cap belongs to the next bucket"
        );
    }

    /// A zero-length resolve still names one granule; the caller refuses it
    /// upstream, but the map must not underflow computing `offset+len-1`.
    #[test]
    fn zero_length_marks_one_granule() {
        let mut occ = WindowOccupancy::default();
        occ.mark(0, 0);
        assert_eq!(occ.chunks_touched(1), 1);
    }
}

#[cfg(test)]
mod host_import_window_tests {
    use super::{capped_import_window, HOST_IMPORT_WINDOW_CAP};

    /// The bug this closes: `vma_bounds` was `/proc/self/maps`-only, so on macOS
    /// it answered `None` for *every* pointer. With no VMA to bucket against,
    /// `host_import_resolve` fell through to its bare aligned-span candidate and
    /// created one region per span — on a live arm64 boot, 310 of 513 regions were
    /// exactly one 16 KiB guest page. The 512-region cap then fired at 128 MiB,
    /// against a 1 GiB byte cap that never came close, and every later import
    /// declined.
    #[cfg(target_os = "macos")]
    #[test]
    fn mach_answers_bounds_that_contain_a_live_mapping() {
        let buf = vec![0u8; 1 << 20];
        let ptr = buf.as_ptr() as usize;
        let (base, len) = super::vma_bounds(ptr).expect(
            "a live readable mapping must have bounds — None here is what \
             collapses the window resolver to one region per span",
        );
        assert!(
            base <= ptr && ptr < base + len,
            "bounds {base:#x}+{len:#x} must contain {ptr:#x}"
        );
        assert!(
            len >= buf.len(),
            "the region must cover the whole allocation, got {len:#x}"
        );
    }

    /// A pointer past the end of its region must not inherit that region's bounds.
    /// `mach_vm_region` answers with the region at or *after* the address it is
    /// given, so without the containment re-check an unmapped address silently
    /// returns the next mapping up — and the caller would bucket a window that
    /// does not hold the span at all.
    #[cfg(target_os = "macos")]
    #[test]
    fn mach_rejects_an_address_outside_every_readable_region() {
        // `__PAGEZERO` covers the low address space with `VM_PROT_NONE`, so this
        // exercises the readability check rather than the containment one.
        assert_eq!(super::vma_bounds(0x1000), None);
    }

    /// With real bounds a single guest page buckets into a 1 GiB window instead of
    /// its own 16 KiB region — the property whose absence walked the region cap.
    /// 16 KiB is the arm64 guest page (`PAGE_SIZE_ARM64E`).
    #[test]
    fn one_guest_page_buckets_into_a_window_rather_than_its_own_region() {
        const GUEST_PAGE: u64 = 0x4000;
        let ptr = VMA_BASE + 0x3_0000_0000;
        let (base, len) =
            capped_import_window(VMA_BASE, VMA_LEN, ptr, ptr as u64 + GUEST_PAGE, ALIGN);
        assert_eq!(
            len, HOST_IMPORT_WINDOW_CAP,
            "a page must pull a whole window"
        );
        assert!(base <= ptr && (ptr as u64) < base as u64 + len);
        // 512 such spans scattered across a 16 GiB VMA land in at most 16 windows,
        // where the per-span fallback made 512 regions.
        let windows: std::collections::BTreeSet<usize> = (0..512)
            .map(|i| {
                let p = VMA_BASE + i * 0x0200_0000;
                capped_import_window(VMA_BASE, VMA_LEN, p, p as u64 + GUEST_PAGE, ALIGN).0
            })
            .collect();
        assert!(
            windows.len() <= 16,
            "512 scattered pages must not need 512 regions, got {}",
            windows.len()
        );
    }

    const ALIGN: u64 = 0x1000;
    const VMA_BASE: usize = 0x7f3a_0000_0000;
    const VMA_LEN: u64 = 0x4_0000_0000; // 16 GiB QEMU RAMBlock shape

    /// A VMA at or under the cap imports whole — small blocks (ROMs, option
    /// RAM) keep the one-import-serves-all behavior.
    #[test]
    fn small_vma_imports_whole() {
        let (base, len) = capped_import_window(
            VMA_BASE,
            HOST_IMPORT_WINDOW_CAP,
            VMA_BASE + 0x5000,
            VMA_BASE as u64 + 0x9000,
            ALIGN,
        );
        assert_eq!((base, len), (VMA_BASE, HOST_IMPORT_WINDOW_CAP));
    }

    /// A span inside a big VMA gets its 1 GiB bucket, never the whole VMA.
    #[test]
    fn big_vma_is_capped_to_bucket() {
        let ptr = VMA_BASE + 0x1_2345_6000; // inside bucket 4 (16 GiB VMA)
        let (base, len) =
            capped_import_window(VMA_BASE, VMA_LEN, ptr, ptr as u64 + 0x80_0000, ALIGN);
        assert_eq!(base as u64, VMA_BASE as u64 + 4 * HOST_IMPORT_WINDOW_CAP);
        assert_eq!(len, HOST_IMPORT_WINDOW_CAP);
        assert!(base <= ptr);
        assert!(ptr as u64 + 0x80_0000 <= base as u64 + len);
    }

    /// A span crossing the bucket edge extends the window to cover it — the
    /// caller requires full coverage of [ptr, end).
    #[test]
    fn span_crossing_bucket_edge_extends() {
        let ptr = VMA_BASE + (HOST_IMPORT_WINDOW_CAP as usize) - 0x2000;
        let end = ptr as u64 + 0x8000; // 8 KiB before the edge into 24 KiB after
        let (base, len) = capped_import_window(VMA_BASE, VMA_LEN, ptr, end, ALIGN);
        assert_eq!(base, VMA_BASE);
        assert!(end <= base as u64 + len);
        assert_eq!(len % ALIGN, 0);
    }

    /// The last bucket clamps to the VMA end — never import past the block.
    #[test]
    fn last_bucket_clamps_to_vma_end() {
        let vma_len = 15 * HOST_IMPORT_WINDOW_CAP + 0x10_0000; // non-multiple
        let ptr = VMA_BASE + (15 * HOST_IMPORT_WINDOW_CAP) as usize + 0x8000;
        let (base, len) = capped_import_window(VMA_BASE, vma_len, ptr, ptr as u64 + 0x1000, ALIGN);
        assert_eq!(base as u64, VMA_BASE as u64 + 15 * HOST_IMPORT_WINDOW_CAP);
        assert_eq!(base as u64 + len, VMA_BASE as u64 + vma_len);
    }

    /// Window base and length stay aligned for page-aligned VMAs.
    #[test]
    fn window_stays_aligned() {
        for off in [0usize, 0x1000, 0x3fff_f000, 0x2_0000_0000] {
            let ptr = VMA_BASE + off;
            let (base, len) =
                capped_import_window(VMA_BASE, VMA_LEN, ptr, ptr as u64 + 0x4000, ALIGN);
            assert_eq!(base as u64 % ALIGN, 0);
            assert_eq!(len % ALIGN, 0);
            assert!(len <= HOST_IMPORT_WINDOW_CAP + 0x4000);
        }
    }
}

#[cfg(test)]
mod vma_bounds_tests {
    use super::vma_bounds_in;

    const MAPS: &str = "\
00400000-00452000 r-xp 00000000 08:02 173521 /usr/bin/dbus-daemon
7f3a00000000-7f3c00000000 rw-p 00000000 00:00 0
7f3c10000000-7f3c10021000 ---p 00000000 00:00 0
7ffc04b1c000-7ffc04b3d000 rw-p 00000000 00:00 0 [stack]";

    /// A pointer inside the big anonymous rw mapping (the QEMU RAMBlock
    /// shape) must resolve to that mapping's exact bounds.
    #[test]
    fn pointer_inside_ram_block_resolves() {
        let ptr = 0x7f3a_8000_0000usize;
        assert_eq!(
            vma_bounds_in(MAPS, ptr),
            Some((0x7f3a_0000_0000, 0x2_0000_0000))
        );
    }

    /// A pointer in an unmapped hole must not resolve — importing unmapped
    /// VA would be undefined behavior at allocate time.
    #[test]
    fn pointer_in_hole_is_none() {
        assert_eq!(vma_bounds_in(MAPS, 0x7f3c_0800_0000), None);
    }

    /// A non-readable guard mapping must refuse (the GPU cannot DMA from it).
    #[test]
    fn non_readable_vma_refused() {
        assert_eq!(vma_bounds_in(MAPS, 0x7f3c_1000_0000), None);
    }

    /// Boundary conditions: first byte in, last byte in, first byte past.
    #[test]
    fn bounds_are_half_open() {
        assert!(vma_bounds_in(MAPS, 0x7f3a_0000_0000).is_some());
        assert!(vma_bounds_in(MAPS, 0x7f3b_ffff_ffff).is_some());
        assert!(vma_bounds_in(MAPS, 0x7f3c_0000_0000).is_none());
    }
}

/// The saturated resolve must not walk the VMA table.
///
/// `host_import_candidates` exists to put the budget ahead of [`vma_bounds`],
/// and on Linux that walk is a `read` of `/proc/self/maps` — so the kernel's own
/// read-syscall counter for this process measures the ordering directly, with no
/// product-side probe. One boot spent 228 s of 484 s of drain time on walks whose
/// answer a saturated budget then discarded.
///
/// The admissible arm is not decoration: it is the converse check `AGENTS.md`
/// requires. A counter that reads zero for the saturated case proves nothing
/// unless it is also shown to move when the walk *does* happen.
#[cfg(all(test, target_os = "linux"))]
mod candidate_walk_tests {
    use super::super::{HOST_IMPORT_REGION_CAP, HOST_IMPORT_TOTAL_BYTE_CAP};
    use super::host_import_candidates;

    /// Read syscalls this process has completed, from `/proc/self/io`.
    fn read_syscalls() -> u64 {
        let io = std::fs::read_to_string("/proc/self/io").expect("procfs io accounting");
        io.lines()
            .find_map(|l| l.strip_prefix("syscr:"))
            .and_then(|v| v.trim().parse().ok())
            .expect("syscr line")
    }

    /// Reads charged to `body`, over a before/after pair of [`read_syscalls`].
    ///
    /// The pair is not free — each sample reads `/proc/self/io` to EOF, and the
    /// reads that finish the `before` sample land after the value it returns. So
    /// a delta carries a fixed instrument cost, which [`baseline`] measures
    /// rather than assumes; comparing raw deltas against zero is what a first cut
    /// of this test did, and the instrument failed it.
    fn reads_during(body: impl FnOnce()) -> u64 {
        let before = read_syscalls();
        body();
        read_syscalls() - before
    }

    /// The instrument's own cost: the same measurement around no work at all.
    fn baseline() -> u64 {
        reads_during(|| {})
    }

    fn walk_cost(region_count: usize, imported_bytes: u64) -> (bool, u64) {
        let probe = 0u64;
        let ptr = &probe as *const u64 as usize;
        let mut admitted = false;
        let reads = reads_during(|| {
            admitted =
                host_import_candidates(region_count, imported_bytes, ptr, ptr as u64 + 8, 0x1000)
                    .is_ok();
        });
        (admitted, reads)
    }

    #[test]
    fn a_refused_budget_costs_no_vma_walk_and_an_admitted_one_does() {
        let base = baseline();

        // Both ways a budget can refuse: the region count and the byte cap.
        let (admitted, reads) = walk_cost(HOST_IMPORT_REGION_CAP, 0);
        assert!(!admitted, "the region cap must refuse");
        assert_eq!(
            reads, base,
            "a refused budget must not read /proc/self/maps"
        );

        let (admitted, reads) = walk_cost(0, HOST_IMPORT_TOTAL_BYTE_CAP);
        assert!(!admitted, "the byte cap must refuse");
        assert_eq!(
            reads, base,
            "a refused budget must not read /proc/self/maps"
        );

        // The converse: the counter does move when the walk happens, so the two
        // readings above are an ordering result and not a dead probe.
        let (admitted, reads) = walk_cost(0, 0);
        assert!(admitted, "an empty pool must admit");
        assert!(
            reads > base,
            "the walk this test measures must be visible to it: {reads} vs baseline {base}"
        );
    }
}
