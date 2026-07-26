//! Task-GVA HostOps views — MapMemory2 / UnmapMemory lifecycle.
//!
//! Apple's host path maps guest pages into a task VA window (`mapMemory`) and
//! tears that mapping down on unmap (`unmapMemory` / guest `CmdUnmapMemory`).
//! Our analogue is a registry of contiguous host-VA views obtained via
//! [`HostOps::map_pages`] after walking the guest task page table for a GVA
//! span. Views are **created on demand** ([`ensure_gva_view`]) and **must** be
//! retired when the guest unmaps or remaps that range so we never write through
//! a host pointer after the GPU page-table mapping is gone.
//!
//! Distinct from:
//! - [`MappingEntry::contig_ptr`] — iosfc `mapping_id` page list (MAP/UNMAP ring)
//! - [`DeviceState::host_gva_surfaces`] — discrete encode cache (retained on Unmap)
//!
//! See [[map-memory2]] GPU-import model and HostOps `map_pages` / `unmap_pages`.

use crate::contract::gva_resolve::{read_task_root, translate_root, Cache, ResolveStatus, Task};
use crate::model::{DeviceState, GvaHostView, TaskEntry};
use crate::runtime::gva_mem::geometry_for_page_shift;
use crate::runtime::host::{HostMemory, HostOps};

/// True if half-open ranges `[a, a+la)` and `[b, b+lb)` overlap.
#[inline]
pub fn ranges_overlap(a: u64, la: u64, b: u64, lb: u64) -> bool {
    if la == 0 || lb == 0 {
        return false;
    }
    let a_end = a.saturating_add(la);
    let b_end = b.saturating_add(lb);
    a < b_end && b < a_end
}

/// Task id used when the view was built matches wire `task_id` (or define-task raw id).
#[inline]
pub(crate) fn task_matches(view_task: u32, wire_task: u32) -> bool {
    view_task == wire_task || view_task == (wire_task >> 1) || (view_task << 1) == wire_task
}

/// Retire every registered GVA view that overlaps `[gva, gva+length)` under `task_id`.
///
/// Pushes `(ptr, ptr_len)` into `retired_views` for [`mapper::flush_retired_views`].
/// Does **not** touch `host_gva_surfaces` (encode content retained across Unmap).
///
/// Returns the number of views retired. Always-on proxy when `n > 0` is logged by caller.
pub fn retire_gva_views_overlapping(
    state: &mut DeviceState,
    task_id: u32,
    gva: u64,
    length: u64,
) -> u32 {
    if gva == 0 || length == 0 {
        return 0;
    }
    let mut n = 0u32;
    let mut i = 0;
    while i < state.gva_host_views.len() {
        let v = &state.gva_host_views[i];
        if task_matches(v.task_id, task_id) && ranges_overlap(v.gva, v.length, gva, length) {
            let v = state.gva_host_views.swap_remove(i);
            if v.ptr != 0 && v.ptr_len != 0 {
                state.retired_views.push((v.ptr, v.ptr_len));
            }
            n = n.saturating_add(1);
        } else {
            i += 1;
        }
    }
    // The draw-time guest-run memo aliases the same task PT — same lifecycle.
    state.guest_run_memo.retain(|e| {
        !(task_matches(e.task_id, task_id) && ranges_overlap(e.gva, e.length, gva, length))
    });
    // The flush no-intersection memo is keyed by the same (task, gva, span) and
    // caches a PT-dependent walk result — drop entries whose gva range this
    // remap invalidates (else a bind could skip the flush after its pages moved
    // under a live deferred window). The 1-in-64 sampled walk is only a backstop.
    state
        .flush_nohit_memo
        .retain(|&(t, g, s), _| !(task_matches(t, task_id) && ranges_overlap(g, s, gva, length)));
    n
}

/// Retire all GVA views for a task (delete_task / task redefine).
pub fn retire_gva_views_for_task(state: &mut DeviceState, task_id: u32) -> u32 {
    let mut n = 0u32;
    let mut i = 0;
    while i < state.gva_host_views.len() {
        if task_matches(state.gva_host_views[i].task_id, task_id) {
            let v = state.gva_host_views.swap_remove(i);
            if v.ptr != 0 && v.ptr_len != 0 {
                state.retired_views.push((v.ptr, v.ptr_len));
            }
            n = n.saturating_add(1);
        } else {
            i += 1;
        }
    }
    state
        .guest_run_memo
        .retain(|e| !task_matches(e.task_id, task_id));
    state
        .flush_nohit_memo
        .retain(|&(t, _, _), _| !task_matches(t, task_id));
    n
}

/// Find a covering view for `task_id` + `[gva, gva+length)` if one is registered.
pub fn find_covering_view(
    state: &DeviceState,
    task_id: u32,
    gva: u64,
    length: u64,
) -> Option<&GvaHostView> {
    if length == 0 {
        return None;
    }
    state.gva_host_views.iter().find(|v| {
        task_matches(v.task_id, task_id)
            && v.gva <= gva
            && gva.saturating_add(length) <= v.gva.saturating_add(v.length)
            && v.ptr != 0
    })
}

/// Resolve which task slot to walk (wire id, then define-task `>> 1`).
fn resolve_task_for_walk(tasks: &[TaskEntry], task_id: u32) -> Option<(u32, &TaskEntry)> {
    if (task_id as usize) < tasks.len() {
        let t = &tasks[task_id as usize];
        if t.active && t.directory_pfn != 0 {
            return Some((task_id, t));
        }
    }
    let shifted = task_id >> 1;
    if shifted != task_id && (shifted as usize) < tasks.len() {
        let t = &tasks[shifted as usize];
        if t.active && t.directory_pfn != 0 {
            return Some((shifted, t));
        }
    }
    None
}

/// Collect one GPA per guest page covering `[gva, gva+length)` under the task PT.
///
/// Returns page-aligned GPAs in GVA order. Fails closed on any unmapped page.
fn collect_span_gpas<M: HostMemory>(
    host: &M,
    task: &TaskEntry,
    gva: u64,
    length: u64,
    page_shift: u32,
) -> Option<Vec<u64>> {
    if length == 0 || !task.active || task.directory_pfn == 0 {
        return None;
    }
    let geom = geometry_for_page_shift(page_shift)?;
    let page_size = geom.page_size as u64;
    struct HostPhys<'a, M: HostMemory>(&'a M);
    impl<M: HostMemory> crate::contract::gva_resolve::PhysReader for HostPhys<'_, M> {
        fn read_phys(&self, gpa: u64, dst: &mut [u8]) -> bool {
            self.0.read_gpa(gpa, dst).is_ok()
        }
    }
    let reader = HostPhys(host);
    let gr_task = Task {
        active: true,
        directory_pfn: task.directory_pfn,
    };
    let root = read_task_root(&reader, &gr_task, geom).ok()?;
    let mut cache = Cache::default();
    let end = gva.saturating_add(length);
    let mut page_gva = gva & !(page_size - 1);
    let mut gpas = Vec::new();
    while page_gva < end {
        let t = translate_root(
            &reader,
            geom,
            root.root_pfn,
            root.depth,
            page_gva,
            Some(&mut cache),
        );
        if t.status != ResolveStatus::Ok {
            return None;
        }
        // HostOps map_pages expects page-aligned GPAs (page base, not +offset).
        let gpa_base = t.gpa & !(page_size - 1);
        gpas.push(gpa_base);
        page_gva = page_gva.saturating_add(page_size);
    }
    if gpas.is_empty() {
        return None;
    }
    Some(gpas)
}

/// Maximal packed-contig runs in a page-GPA list (product Linux `map_pages`).
///
/// Each run is a half-open index range `[start, end)` into `gpas` where
/// `gpas[i+1] == gpas[i] + page_size`. Callers multi-import one run at a time.
pub fn contig_page_runs(gpas: &[u64], page_size: u64) -> Vec<std::ops::Range<usize>> {
    if gpas.is_empty() || page_size == 0 {
        return Vec::new();
    }
    let mut runs = Vec::new();
    let mut start = 0usize;
    for i in 1..gpas.len() {
        if gpas[i] != gpas[i - 1].wrapping_add(page_size) {
            runs.push(start..i);
            start = i;
        }
    }
    runs.push(start..gpas.len());
    runs
}

/// Build or reuse a contiguous host-VA view of guest pages for `[gva, gva+length)`.
///
/// Walks the task page table (PPNs already installed by the guest before MapMemory2),
/// then [`HostOps::map_pages`]. Returns `(ptr, host_len)`. On Linux, map_pages requires
/// a **packed** sequential host-VA run — fragmented GVA spans return `None` here;
/// use [`write_span`] / [`read_span`] which multi-import maximal runs. Does not invent
/// PTEs.
pub fn ensure_gva_view<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    task_id: u32,
    gva: u64,
    length: u64,
) -> Option<(usize, usize)> {
    if gva == 0 || length == 0 {
        return None;
    }
    if let Some(v) = find_covering_view(state, task_id, gva, length) {
        let (vptr, vlen) = (v.ptr, v.ptr_len);
        let view = v.clone();
        // Sampled staleness verify (1-in-32 reuses): re-translate the view's
        // first/last leaf and compare GPAs. A guest PT rewire the Unmap/Map2
        // notifies missed (or that raced ahead of the FIFO) makes the cached
        // view alias pages the guest already recycled — reads through it see
        // freshly zeroed memory (the black-tile class). A mismatch retires
        // the view fail-visibly and rebuilds fresh below.
        state.view_verify_ctr = state.view_verify_ctr.wrapping_add(1);
        if !state.view_verify_ctr.is_multiple_of(32) || view_gpas_current(host, state, &view) {
            return Some((vptr, vlen));
        }
        state.view_stale_reads = state.view_stale_reads.saturating_add(1);
        let n = state.view_stale_reads;
        if n == 1 || n.is_multiple_of(256) {
            crate::observe::fail(format!(
                "gva_view_stale task={} gva={:#x} len={:#x} count={n}",
                view.task_id, view.gva, view.length
            ));
        }
        let mut i = 0;
        while i < state.gva_host_views.len() {
            let w = &state.gva_host_views[i];
            if w.ptr == view.ptr && w.gva == view.gva && w.task_id == view.task_id {
                let w = state.gva_host_views.swap_remove(i);
                if w.ptr != 0 && w.ptr_len != 0 {
                    state.retired_views.push((w.ptr, w.ptr_len));
                }
            } else {
                i += 1;
            }
        }
    }
    // Flush any pending unmaps before allocating a new view (Darwin private VA).
    crate::runtime::mapper::flush_retired_views(state, host);

    let page_shift = state.page_shift;
    let (resolved_tid, gpas) = {
        let (tid, task) = resolve_task_for_walk(&state.tasks, task_id)?;
        let gpas = collect_span_gpas(host, task, gva, length, page_shift)?;
        (tid, gpas)
    };
    let page_sz = state.page_size() as usize;
    // Reject non-RAM leaf GPAs (mapper / wild-PFN class) before map_pages.
    if gpas.iter().any(|&g| !host.is_ram_gpa(g)) {
        return None;
    }
    // Full-span packed view only (single map_pages). Fragmented → None.
    let ptr = host.map_pages(&gpas, page_sz)?;
    let page_sz = (1usize) << page_shift;
    let ptr_len = gpas.len().saturating_mul(page_sz);
    state.gva_host_views.push(GvaHostView {
        task_id: resolved_tid,
        gva,
        length,
        ptr,
        ptr_len,
        first_gpa: gpas.first().copied().unwrap_or(0),
        last_gpa: gpas.last().copied().unwrap_or(0),
    });
    Some((ptr, ptr_len))
}

/// True when the view's first/last leaf still translate to the GPAs recorded
/// at build time (`first_gpa == 0` = unverifiable fixture view, passes).
///
/// First/last-page coverage is exact for the 1–2 page views tile content
/// uses and a cheap canary for larger spans (2 leaf walks, ~1 µs each).
fn view_gpas_current<H: HostMemory>(
    host: &H,
    state: &DeviceState,
    v: &crate::model::GvaHostView,
) -> bool {
    if v.first_gpa == 0 {
        return true;
    }
    let Some((_tid, task)) = resolve_task_for_walk(&state.tasks, v.task_id) else {
        return false;
    };
    let page_shift = state.page_shift;
    let page = 1u64 << page_shift;
    let first_page = v.gva & !(page - 1);
    match collect_span_gpas(host, task, first_page, 1, page_shift) {
        Some(g) if g.first() == Some(&v.first_gpa) => {}
        _ => return false,
    }
    if v.last_gpa != 0 {
        let last_page = v.gva.saturating_add(v.length).saturating_sub(1) & !(page - 1);
        if last_page != first_page {
            match collect_span_gpas(host, task, last_page, 1, page_shift) {
                Some(g) if g.first() == Some(&v.last_gpa) => {}
                _ => return false,
            }
        }
    }
    true
}

/// Always-on line when views are retired (proxy for Unmap/Map teardown).
///
/// `op` names the guest operation that retired them, and the key is `op=` rather
/// than `reason=` deliberately: retiring a view on Unmap/Map is *correct*
/// behaviour, not a refusal, so it has no registered decline slug and must not
/// claim one. It was `reason=UnmapMemory` — CamelCase, so
/// `grep 'reason=[a-z_]*'` silently missed it while a reader scanning for
/// refusals found a line that was not one.
pub fn log_retire(op: &str, task_id: u32, gva: u64, length: u64, n: u32) {
    if n == 0 {
        return;
    }
    crate::observe::off(format!(
        "gva_view_drop op={op} task={task_id} gva={gva:#x} len={length:#x} n={n}"
    ));
}

/// Host pointer to the first byte of guest `gva` for a span of `length` bytes.
///
/// Builds/reuses a contig HostOps view over the task page table. Returns
/// `(host_ptr, available_bytes_from_ptr)` covering at least `length`, or `None`
/// if any page is unmapped / non-contiguous on the host.
pub fn host_ptr_for_span<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    task_id: u32,
    gva: u64,
    length: u64,
) -> Option<(*mut u8, usize)> {
    if gva == 0 || length == 0 {
        return None;
    }
    let (ptr, ptr_len) = ensure_gva_view(state, host, task_id, gva, length)?;
    let page_size = state.page_size();
    let page_mask = page_size - 1;
    // ensure_gva_view maps from the page base of the registered span base.
    // Prefer the covering view's registered gva for offset math.
    let view_gva = find_covering_view(state, task_id, gva, length)
        .map(|v| v.gva)
        .unwrap_or(gva);
    let view_page_base = view_gva & !page_mask;
    let off = (gva.saturating_sub(view_page_base)) as usize;
    if off >= ptr_len {
        return None;
    }
    let avail = ptr_len - off;
    if (avail as u64) < length {
        return None;
    }
    // SAFETY: ensure_gva_view returns ptr for ptr_len host-mapped bytes.
    let p = unsafe { (ptr as *mut u8).add(off) };
    Some((p, avail))
}

/// Write `buf` into guest `[gva, gva+buf.len())` via HostOps map_pages.
///
/// **Writes never reuse a cached view.** A registered `gva_host_views` entry
/// goes stale the moment the guest rewires its task PT (tile/page recycle)
/// and is only retired when the Unmap/Map2 notify drains — a write through
/// it lands in whatever now owns those host pages (guest heap corruption:
/// the 2026-07-19 WindowServer SIGSEGV class). Every write walks the PT at
/// write time: packed spans map once, fragmented spans multi-import per run.
pub fn write_span<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    task_id: u32,
    gva: u64,
    buf: &[u8],
) -> bool {
    if buf.is_empty() {
        return true;
    }
    write_span_multi(state, host, task_id, gva, buf)
}

/// Ephemeral fresh-walk host mapping of `[gva, gva+length)` for guest writes.
///
/// Same write-freshness rule as [`write_span`]: walks the task PT at call
/// time and maps the packed span without consulting or registering
/// `gva_host_views`. The caller must release it with [`unmap_fresh_span`]
/// (product Linux unmap is a no-op alias; Darwin unmaps a real region).
/// Fragmented spans return `None` — callers fall back to their per-row
/// multi-import path, which is also fresh.
pub struct FreshSpan {
    /// First byte of `gva` inside the mapped span.
    pub ptr: *mut u8,
    /// Writable bytes available at `ptr` (>= the requested length).
    pub avail: usize,
    map_base: usize,
    map_len: usize,
}

/// Build a [`FreshSpan`] over `[gva, gva+length)` — fresh PT walk, packed map.
pub fn map_fresh_span<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    task_id: u32,
    gva: u64,
    length: u64,
) -> Option<FreshSpan> {
    if gva == 0 || length == 0 {
        return None;
    }
    let page_shift = state.page_shift;
    let gpas = {
        let (_tid, task) = resolve_task_for_walk(&state.tasks, task_id)?;
        collect_span_gpas(host, task, gva, length, page_shift)?
    };
    if gpas.iter().any(|&g| !host.is_ram_gpa(g)) {
        return None;
    }
    crate::runtime::mapper::flush_retired_views(state, host);
    let page_size = state.page_size();
    let page_sz = page_size as usize;
    let ptr_base = host.map_pages(&gpas, page_sz)?;
    let map_len = gpas.len().saturating_mul(page_sz);
    let off = (gva & (page_size - 1)) as usize;
    if off >= map_len || ((map_len - off) as u64) < length {
        host.unmap_pages(ptr_base, map_len);
        return None;
    }
    Some(FreshSpan {
        // SAFETY: map_pages returned `map_len` mapped bytes at `ptr_base`.
        ptr: unsafe { (ptr_base as *mut u8).add(off) },
        avail: map_len - off,
        map_base: ptr_base,
        map_len,
    })
}

/// Release a [`map_fresh_span`] mapping.
pub fn unmap_fresh_span<H: HostOps>(host: &mut H, span: FreshSpan) {
    host.unmap_pages(span.map_base, span.map_len);
}

/// Read `buf.len()` bytes from guest `gva` via HostOps map_pages (multi-import).
pub fn read_span<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    task_id: u32,
    gva: u64,
    buf: &mut [u8],
) -> bool {
    if buf.is_empty() {
        return true;
    }
    if let Some((ptr, avail)) = host_ptr_for_span(state, host, task_id, gva, buf.len() as u64) {
        if avail >= buf.len() {
            // SAFETY: host_ptr_for_span guarantees `avail` readable bytes.
            unsafe {
                std::ptr::copy_nonoverlapping(ptr as *const u8, buf.as_mut_ptr(), buf.len());
            }
            return true;
        }
    }
    read_span_multi(state, host, task_id, gva, buf)
}

/// Multi-import write: map each packed GPA run, copy, unmap. No write_gpa.
///
/// Ephemeral per-run maps (do not register partial views — Darwin unmap needs
/// the full map_pages base; product Linux alias is a no-op unmap).
fn write_span_multi<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    task_id: u32,
    gva: u64,
    buf: &[u8],
) -> bool {
    let length = buf.len() as u64;
    let page_shift = state.page_shift;
    let page_size = state.page_size();
    let page_sz = page_size as usize;
    let gpas = {
        let Some((_tid, task)) = resolve_task_for_walk(&state.tasks, task_id) else {
            return false;
        };
        let Some(gpas) = collect_span_gpas(host, task, gva, length, page_shift) else {
            return false;
        };
        gpas
    };
    if gpas.iter().any(|&g| !host.is_ram_gpa(g)) {
        return false;
    }
    let runs = contig_page_runs(&gpas, page_size);
    if runs.is_empty() {
        return false;
    }
    crate::runtime::mapper::flush_retired_views(state, host);
    let span_page_base = gva & !(page_size - 1);
    let end = gva.saturating_add(length);
    for run in &runs {
        let run_gpas = &gpas[run.clone()];
        let Some(ptr) = host.map_pages(run_gpas, page_sz) else {
            return false;
        };
        let total = run_gpas.len().saturating_mul(page_sz);
        let run_gva = span_page_base.saturating_add((run.start as u64).saturating_mul(page_size));
        let run_end = run_gva.saturating_add(total as u64);
        let copy_lo = gva.max(run_gva);
        let copy_hi = end.min(run_end);
        if copy_lo >= copy_hi {
            host.unmap_pages(ptr, total);
            continue;
        }
        let buf_off = (copy_lo - gva) as usize;
        let host_off = (copy_lo - run_gva) as usize;
        let n = (copy_hi - copy_lo) as usize;
        if host_off + n > total || buf_off + n > buf.len() {
            host.unmap_pages(ptr, total);
            return false;
        }
        // SAFETY: map_pages packed `total` bytes; host_off+n in range.
        unsafe {
            std::ptr::copy_nonoverlapping(
                buf.as_ptr().add(buf_off),
                (ptr as *mut u8).add(host_off),
                n,
            );
        }
        host.unmap_pages(ptr, total);
    }
    true
}

/// Multi-import read: map each packed GPA run, copy out, unmap.
fn read_span_multi<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    task_id: u32,
    gva: u64,
    buf: &mut [u8],
) -> bool {
    let length = buf.len() as u64;
    let page_shift = state.page_shift;
    let page_size = state.page_size();
    let page_sz = page_size as usize;
    let gpas = {
        let Some((_tid, task)) = resolve_task_for_walk(&state.tasks, task_id) else {
            return false;
        };
        let Some(gpas) = collect_span_gpas(host, task, gva, length, page_shift) else {
            return false;
        };
        gpas
    };
    if gpas.iter().any(|&g| !host.is_ram_gpa(g)) {
        return false;
    }
    let runs = contig_page_runs(&gpas, page_size);
    if runs.is_empty() {
        return false;
    }
    crate::runtime::mapper::flush_retired_views(state, host);
    let span_page_base = gva & !(page_size - 1);
    let end = gva.saturating_add(length);
    for run in &runs {
        let run_gpas = &gpas[run.clone()];
        let Some(ptr) = host.map_pages(run_gpas, page_sz) else {
            return false;
        };
        let total = run_gpas.len().saturating_mul(page_sz);
        let run_gva = span_page_base.saturating_add((run.start as u64).saturating_mul(page_size));
        let run_end = run_gva.saturating_add(total as u64);
        let copy_lo = gva.max(run_gva);
        let copy_hi = end.min(run_end);
        if copy_lo >= copy_hi {
            host.unmap_pages(ptr, total);
            continue;
        }
        let buf_off = (copy_lo - gva) as usize;
        let host_off = (copy_lo - run_gva) as usize;
        let n = (copy_hi - copy_lo) as usize;
        if host_off + n > total || buf_off + n > buf.len() {
            host.unmap_pages(ptr, total);
            return false;
        }
        // SAFETY: map covers total bytes.
        unsafe {
            std::ptr::copy_nonoverlapping(
                (ptr as *const u8).add(host_off),
                buf.as_mut_ptr().add(buf_off),
                n,
            );
        }
        host.unmap_pages(ptr, total);
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::endian::st32;
    use crate::contract::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
    use crate::model::{DeviceId, PAGE_SHIFT_X86};
    use crate::runtime::host::FakeHost;

    fn state_x86() -> DeviceState {
        DeviceState::new(DeviceId(1), 12)
    }

    #[test]
    fn contig_page_runs_splits_gaps() {
        let page = 0x1000u64;
        let gpas = [0x1000u64, 0x2000, 0x4000, 0x5000, 0x8000];
        let runs = contig_page_runs(&gpas, page);
        assert_eq!(runs, vec![0..2, 2..4, 4..5]);
        assert_eq!(contig_page_runs(&[0x1000], page), vec![0..1]);
        assert!(contig_page_runs(&[], page).is_empty());
    }

    #[test]
    fn ranges_overlap_basic() {
        assert!(ranges_overlap(0x1000, 0x1000, 0x1800, 0x1000));
        assert!(!ranges_overlap(0x1000, 0x1000, 0x2000, 0x1000));
        assert!(!ranges_overlap(0x1000, 0, 0x1000, 0x1000));
    }

    /// Fragmented GVA (non-adjacent leaf PFNs): product write multi-imports runs.
    ///
    /// Linux FakeHost `strict_linux_map` matches product QEMU (no bounce) so the
    /// full-span map_pages fails and the multi-run path is load-bearing.
    #[test]
    fn multi_import_fragmented_gva_write() {
        let page_shift = PAGE_SHIFT_X86;
        let page = 1u64 << page_shift;
        let mut host = FakeHost::new();
        host.strict_linux_map = true;
        // dir pfn=2, root pfn=3, data page0 pfn=4, data page1 pfn=10 (gap).
        let dir_gpa = 2u64 << page_shift;
        let root_gpa = 3u64 << page_shift;
        let data0 = 4u64 << page_shift;
        let data1 = 10u64 << page_shift;
        host.map_range(dir_gpa, page as usize, 0);
        host.map_range(root_gpa, page as usize, 0);
        host.map_range(data0, page as usize, 0);
        host.map_range(data1, page as usize, 0);
        let mut d = [0u8; 8];
        st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 3);
        st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
        host.write_gpa(dir_gpa, &d).unwrap();
        let mut pte = [0u8; 4];
        st32(&mut pte, 4);
        host.write_gpa(root_gpa, &pte).unwrap();
        st32(&mut pte, 10);
        host.write_gpa(root_gpa + 4, &pte).unwrap();

        #[cfg(not(target_os = "macos"))]
        {
            // Full span map of [data0, data1] must fail under strict Linux semantics.
            assert!(
                host.map_pages(&[data0, data1], page as usize).is_none(),
                "strict map must reject non-packed GPA list"
            );
        }

        let mut state = state_x86();
        assert!(state.define_task(1, page, 2));
        let payload = [0x11u8, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88];
        // Write 4 bytes at end of page0 + 4 at start of page1 (crosses gap).
        let gva = page - 4;
        assert!(
            crate::runtime::gva_mem::write_task_gva_product(
                &mut state, &mut host, 1, gva, &payload
            )
            .is_ok(),
            "multi-import product write must succeed across fragmented PFNs"
        );
        let mut back = [0u8; 8];
        assert!(host.read_gpa(data0 + page - 4, &mut back[..4]).is_ok());
        assert!(host.read_gpa(data1, &mut back[4..]).is_ok());
        assert_eq!(back, payload);
    }

    /// PT fixture: dir pfn 2 (root pfn 3, depth 1), PTE[0] → data0 (pfn 4).
    /// data1 (pfn 10) is mapped but initially unreferenced by the PT.
    fn pt_fixture(page_shift: u32) -> (FakeHost, u64, u64, u64, u64) {
        let page = 1u64 << page_shift;
        let mut host = FakeHost::new();
        host.strict_linux_map = true;
        let dir_gpa = 2u64 << page_shift;
        let root_gpa = 3u64 << page_shift;
        let data0 = 4u64 << page_shift;
        let data1 = 10u64 << page_shift;
        host.map_range(dir_gpa, page as usize, 0);
        host.map_range(root_gpa, page as usize, 0);
        host.map_range(data0, page as usize, 0);
        host.map_range(data1, page as usize, 0);
        let mut d = [0u8; 8];
        st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 3);
        st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
        host.write_gpa(dir_gpa, &d).unwrap();
        let mut pte = [0u8; 4];
        st32(&mut pte, 4);
        host.write_gpa(root_gpa, &pte).unwrap();
        (host, root_gpa, data0, data1, page)
    }

    /// Guest writes must land where the PT points **now**, not where a
    /// registered view pointed when it was built (stale-view heap-corruption
    /// class — the guest recycles pages before the Unmap notify drains).
    #[test]
    fn write_span_ignores_stale_registered_view() {
        let page_shift = PAGE_SHIFT_X86;
        let (mut host, root_gpa, data0, data1, page) = pt_fixture(page_shift);
        let mut state = state_x86();
        assert!(state.define_task(1, page, 2));
        let gva = 8u64;
        assert!(write_span(&mut state, &mut host, 1, gva, &[1, 2, 3, 4]));
        let mut back = [0u8; 4];
        host.read_gpa(data0 + gva, &mut back).unwrap();
        assert_eq!(back, [1, 2, 3, 4]);

        // Register a view over the span (as a read would), then rewire the
        // PTE to data1 WITHOUT any Unmap notify — the view is now stale.
        let (vptr, vlen) = ensure_gva_view(&mut state, &mut host, 1, gva, 4).unwrap();
        assert!(vptr != 0 && vlen != 0);
        let mut pte = [0u8; 4];
        st32(&mut pte, 10);
        host.write_gpa(root_gpa, &pte).unwrap();

        assert!(write_span(&mut state, &mut host, 1, gva, &[5, 6, 7, 8]));
        host.read_gpa(data1 + gva, &mut back).unwrap();
        assert_eq!(back, [5, 6, 7, 8], "write must follow the live PT");
        host.read_gpa(data0 + gva, &mut back).unwrap();
        assert_eq!(back, [1, 2, 3, 4], "stale page must not be touched");
    }

    /// map_fresh_span re-walks per call: after a PT rewire, writes through
    /// the returned pointer land in the newly wired page.
    #[test]
    fn map_fresh_span_follows_pt_rewire() {
        let page_shift = PAGE_SHIFT_X86;
        let (mut host, root_gpa, data0, data1, page) = pt_fixture(page_shift);
        let mut state = state_x86();
        assert!(state.define_task(1, page, 2));
        let gva = 8u64;
        let s = map_fresh_span(&mut state, &mut host, 1, gva, 16).unwrap();
        assert!(s.avail >= 16);
        // SAFETY: map_fresh_span guarantees ≥16 writable bytes at ptr.
        unsafe { std::ptr::copy_nonoverlapping([0xaau8; 4].as_ptr(), s.ptr, 4) };
        unmap_fresh_span(&mut host, s);
        let mut back = [0u8; 4];
        host.read_gpa(data0 + gva, &mut back).unwrap();
        assert_eq!(back, [0xaa; 4]);

        let mut pte = [0u8; 4];
        st32(&mut pte, 10);
        host.write_gpa(root_gpa, &pte).unwrap();
        let s = map_fresh_span(&mut state, &mut host, 1, gva, 16).unwrap();
        // SAFETY: as above.
        unsafe { std::ptr::copy_nonoverlapping([0xbbu8; 4].as_ptr(), s.ptr, 4) };
        unmap_fresh_span(&mut host, s);
        host.read_gpa(data1 + gva, &mut back).unwrap();
        assert_eq!(back, [0xbb; 4], "fresh span must follow the rewired PT");
        host.read_gpa(data0 + gva, &mut back).unwrap();
        assert_eq!(back, [0xaa; 4], "old page must not see the second write");
    }

    /// The 1-in-32 sampled reuse verify catches a PT rewire under a cached
    /// view, retires it, and rebuilds fresh; unsampled reuses stay cheap.
    #[test]
    fn stale_covering_view_detected_and_rebuilt() {
        let page_shift = PAGE_SHIFT_X86;
        let (mut host, root_gpa, data0, data1, page) = pt_fixture(page_shift);
        let mut state = state_x86();
        assert!(state.define_task(1, page, 2));
        let gva = 8u64;
        let (p0, _) = ensure_gva_view(&mut state, &mut host, 1, gva, 16).unwrap();
        assert_eq!(state.gva_host_views.len(), 1);
        assert_eq!(state.gva_host_views[0].first_gpa, data0);

        // Rewire the PTE (no Unmap notify) — the registered view is stale.
        let mut pte = [0u8; 4];
        st32(&mut pte, 10);
        host.write_gpa(root_gpa, &pte).unwrap();

        // Unsampled reuse still returns the cached view (sampling contract).
        state.view_verify_ctr = 0;
        let (p1, _) = ensure_gva_view(&mut state, &mut host, 1, gva, 16).unwrap();
        assert_eq!(p1, p0);
        assert_eq!(state.view_stale_reads, 0);

        // Sampled reuse (ctr hits a multiple of 32) detects, retires, rebuilds.
        state.view_verify_ctr = 31;
        let (p2, _) = ensure_gva_view(&mut state, &mut host, 1, gva, 16).unwrap();
        assert_eq!(state.view_stale_reads, 1);
        assert_eq!(state.gva_host_views.len(), 1);
        assert_eq!(state.gva_host_views[0].first_gpa, data1);
        // Writes through the rebuilt view land in the newly wired page.
        // SAFETY: ensure_gva_view mapped the page; gva page offset is 8.
        unsafe { *((p2 as *mut u8).add(gva as usize)) = 0xcc };
        let mut back = [0u8; 1];
        host.read_gpa(data1 + gva, &mut back).unwrap();
        assert_eq!(back[0], 0xcc);
    }

    #[test]
    fn unmap_retires_overlapping_view_only() {
        let mut state = state_x86();
        state.gva_host_views.push(GvaHostView {
            task_id: 2,
            gva: 0x1000,
            length: 0x2000,
            ptr: 0xaaaa,
            ptr_len: 0x2000,
            ..Default::default()
        });
        state.gva_host_views.push(GvaHostView {
            task_id: 2,
            gva: 0x10000,
            length: 0x1000,
            ptr: 0xbbbb,
            ptr_len: 0x1000,
            ..Default::default()
        });
        state.gva_host_views.push(GvaHostView {
            task_id: 3,
            gva: 0x1000,
            length: 0x2000,
            ptr: 0xcccc,
            ptr_len: 0x2000,
            ..Default::default()
        });

        let n = retire_gva_views_overlapping(&mut state, 2, 0x1500, 0x100);
        assert_eq!(n, 1);
        assert_eq!(state.gva_host_views.len(), 2);
        assert!(state
            .gva_host_views
            .iter()
            .any(|v| v.ptr == 0xbbbb && v.task_id == 2));
        assert!(state
            .gva_host_views
            .iter()
            .any(|v| v.ptr == 0xcccc && v.task_id == 3));
        assert_eq!(state.retired_views, vec![(0xaaaa, 0x2000)]);
    }

    #[test]
    fn unmap_matches_define_task_id_shift() {
        let mut state = state_x86();
        // View stored under resolved task slot 1; wire Unmap may carry raw id 2.
        state.gva_host_views.push(GvaHostView {
            task_id: 1,
            gva: 0x2000,
            length: 0x1000,
            ptr: 0x1111,
            ptr_len: 0x1000,
            ..Default::default()
        });
        let n = retire_gva_views_overlapping(&mut state, 2, 0x2000, 0x1000);
        assert_eq!(n, 1);
        assert!(state.gva_host_views.is_empty());
        assert_eq!(state.retired_views, vec![(0x1111, 0x1000)]);
    }

    #[test]
    fn delete_task_retires_views() {
        let mut state = state_x86();
        assert!(state.define_task(1, 0x1_0000, 0x100));
        state.gva_host_views.push(GvaHostView {
            task_id: 1,
            gva: 0x3000,
            length: 0x1000,
            ptr: 0xdddd,
            ptr_len: 0x1000,
            ..Default::default()
        });
        state.gva_host_views.push(GvaHostView {
            task_id: 2,
            gva: 0x3000,
            length: 0x1000,
            ptr: 0xeeee,
            ptr_len: 0x1000,
            ..Default::default()
        });
        assert!(state.delete_task(1));
        assert_eq!(state.gva_host_views.len(), 1);
        assert_eq!(state.gva_host_views[0].ptr, 0xeeee);
        assert_eq!(state.retired_views, vec![(0xdddd, 0x1000)]);
    }

    #[test]
    fn ensure_gva_view_none_without_task() {
        let mut state = state_x86();
        let mut host = FakeHost::new();
        assert!(ensure_gva_view(&mut state, &mut host, 1, 0x1000, 0x1000).is_none());
    }

    #[test]
    fn covering_view_reuse() {
        let mut state = state_x86();
        state.gva_host_views.push(GvaHostView {
            task_id: 1,
            gva: 0x1000,
            length: 0x4000,
            ptr: 0x9000,
            ptr_len: 0x4000,
            ..Default::default()
        });
        let v = find_covering_view(&state, 1, 0x1800, 0x100).unwrap();
        assert_eq!(v.ptr, 0x9000);
        assert!(find_covering_view(&state, 1, 0x1000, 0x5000).is_none());
    }
}
