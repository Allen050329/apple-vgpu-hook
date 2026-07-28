//! Read task GPU-virtual addresses via the task page directory.
//!
//! Thin wrapper over [`crate::contract::gva_resolve`] + [`HostMemory`].
//! Geometry always requires an explicit create-time page_shift (12 = x86_64,
//! 14 = arm64e). There is no arm-default overload — callers must choose.

use crate::contract::gva_resolve::{
    read_task_root, resolve_status_name, translate_root, Cache, Geometry, PhysReader,
    ResolveStatus, Task, ARM64E_GEOMETRY, X86_64_GEOMETRY,
};
use crate::model::TaskEntry;
use crate::runtime::host::{HostMemory, MemError};

struct HostPhys<'a, M: HostMemory>(&'a M);

impl<M: HostMemory> PhysReader for HostPhys<'_, M> {
    fn read_phys(&self, gpa: u64, dst: &mut [u8]) -> bool {
        self.0.read_gpa(gpa, dst).is_ok()
    }
}

/// Select page-table geometry for a known guest page size.
///
/// Only 12 (x86_64) and 14 (arm64e) are valid. Unknown shifts return `None`
/// (no silent arm fallback).
#[inline]
pub fn geometry_for_page_shift(page_shift: u32) -> Option<&'static Geometry> {
    if page_shift == X86_64_GEOMETRY.page_shift {
        Some(&X86_64_GEOMETRY)
    } else if page_shift == ARM64E_GEOMETRY.page_shift {
        Some(&ARM64E_GEOMETRY)
    } else {
        None
    }
}

/// Translate `gva` under `task` and copy `buf.len()` bytes into `buf`.
///
/// `page_shift` must be the device create-time guest page shift (12 or 14).
pub fn read_task_gva<M: HostMemory>(
    host: &M,
    task: &TaskEntry,
    gva: u64,
    buf: &mut [u8],
    page_shift: u32,
) -> Result<(), MemError> {
    if buf.is_empty() {
        return Ok(());
    }
    if !task.active || task.directory_pfn == 0 {
        return Err(MemError::NoTaskDirectory);
    }
    let geom = geometry_for_page_shift(page_shift).ok_or(MemError::UnsupportedPageShift)?;
    let reader = HostPhys(host);
    let gr_task = Task {
        active: true,
        directory_pfn: task.directory_pfn,
    };
    let root = read_task_root(&reader, &gr_task, geom).map_err(|_| MemError::TaskRootRead)?;
    let mut cache = Cache::default();
    let mut filled = 0usize;
    while filled < buf.len() {
        let cur = gva.saturating_add(filled as u64);
        let t = translate_root(
            &reader,
            geom,
            root.root_pfn,
            root.depth,
            cur,
            Some(&mut cache),
        );
        if t.status != ResolveStatus::Ok {
            return Err(MemError::Unresolved(t.status));
        }
        let page_left = geom.page_size as u64 - (cur & geom.page_offset_mask as u64);
        let n = (buf.len() - filled).min(page_left as usize);
        host.read_gpa(t.gpa, &mut buf[filled..filled + n])?;
        filled += n;
    }
    Ok(())
}

/// Same as [`read_task_gva`] but also tries `task_id >> 1` when the wire task
/// word is a raw define-task id (archive cursor path).
pub fn read_task_gva_fallback<M: HostMemory>(
    host: &M,
    tasks: &[TaskEntry],
    task_id: u32,
    gva: u64,
    buf: &mut [u8],
    page_shift: u32,
) -> Result<(), MemError> {
    if (task_id as usize) < tasks.len()
        && read_task_gva(host, &tasks[task_id as usize], gva, buf, page_shift).is_ok()
    {
        return Ok(());
    }
    let shifted = task_id >> 1;
    if shifted != task_id && (shifted as usize) < tasks.len() {
        return read_task_gva(host, &tasks[shifted as usize], gva, buf, page_shift);
    }
    Err(MemError::NoSuchTask)
}

/// Translate `gva` under `task` and write `buf` into guest RAM via `write_gpa`.
///
/// **Tests / fixtures only.** Product paths must use [`write_task_gva_product`]
/// (contig HostOps view). Do not call from product encode/blit/compute.
pub fn write_task_gva<M: HostMemory>(
    host: &mut M,
    task: &TaskEntry,
    gva: u64,
    buf: &[u8],
    page_shift: u32,
) -> Result<(), MemError> {
    if buf.is_empty() {
        return Ok(());
    }
    if !task.active || task.directory_pfn == 0 {
        return Err(MemError::NoTaskDirectory);
    }
    let geom = geometry_for_page_shift(page_shift).ok_or(MemError::UnsupportedPageShift)?;
    let gr_task = Task {
        active: true,
        directory_pfn: task.directory_pfn,
    };
    let root = {
        let reader = HostPhys(&*host);
        read_task_root(&reader, &gr_task, geom).map_err(|_| MemError::TaskRootRead)?
    };
    let mut cache = Cache::default();
    let mut written = 0usize;
    while written < buf.len() {
        let cur = gva.saturating_add(written as u64);
        let t = {
            let reader = HostPhys(&*host);
            translate_root(
                &reader,
                geom,
                root.root_pfn,
                root.depth,
                cur,
                Some(&mut cache),
            )
        };
        if t.status != ResolveStatus::Ok {
            return Err(MemError::Unresolved(t.status));
        }
        let page_left = geom.page_size as u64 - (cur & geom.page_offset_mask as u64);
        let n = (buf.len() - written).min(page_left as usize);
        host.write_gpa(t.gpa, &buf[written..written + n])?;
        written += n;
    }
    Ok(())
}

/// Same as [`write_task_gva`] with define-task id fallback (`task_id >> 1`).
///
/// **Tests / fixtures only** — not product (see [`write_task_gva_product`]).
pub fn write_task_gva_fallback<M: HostMemory>(
    host: &mut M,
    tasks: &[TaskEntry],
    task_id: u32,
    gva: u64,
    buf: &[u8],
    page_shift: u32,
) -> Result<(), MemError> {
    if (task_id as usize) < tasks.len()
        && write_task_gva(host, &tasks[task_id as usize], gva, buf, page_shift).is_ok()
    {
        return Ok(());
    }
    let shifted = task_id >> 1;
    if shifted != task_id && (shifted as usize) < tasks.len() {
        return write_task_gva(host, &tasks[shifted as usize], gva, buf, page_shift);
    }
    Err(MemError::Unmapped)
}

/// Product GVA write: HostOps `map_pages` only (no `write_gpa` walk).
///
/// Full-span packed view when possible; otherwise **multi-import** maximal
/// packed GPA runs ([`crate::runtime::gva_view::write_span`]). Fails closed when
/// any page is unmapped or a run cannot be mapped. When the task has recorded
/// MapMemory2 spans, the write must lie inside one span (`outside_map`).
/// Always-on: `gva_write fail reason=…`, carrying the check `write_span`
/// actually refused on rather than a reason chosen here.
pub fn write_task_gva_product<H: HostMemory + crate::runtime::host::HostOps>(
    state: &mut crate::model::DeviceState,
    host: &mut H,
    task_id: u32,
    gva: u64,
    buf: &[u8],
) -> Result<(), MemError> {
    if buf.is_empty() {
        return Ok(());
    }
    // The gate's own answer, not a label chosen here. It has three distinct ways
    // to permit a write and nothing has ever read which one applied: a covered
    // span, an *aliased* task's span, or no span registry at all. Only the last
    // of those is a bounds check that did not happen, and `delete_task` clears
    // the registry, so a write arriving after a teardown takes it.
    let gate = state.gva_write_gate(task_id, gva, buf.len() as u64);
    if gate == crate::model::WriteGate::Outside {
        crate::observe::Emit::decline("gva_write", &gate)
            .field("task", task_id)
            .field("gva", format!("{gva:#x}"))
            .field("len", format!("{:#x}", buf.len()))
            .fail();
        return Err(MemError::OutsideMap);
    }
    // Census the permissive arms, latched per (arm, task). Exact is the arm the
    // gate is supposed to take; the other two are the reading this exists for.
    if gate != crate::model::WriteGate::Exact {
        use crate::observe::Decline;
        let by = match gate {
            crate::model::WriteGate::Aliased { by } => by,
            _ => 0,
        };
        if crate::observe::first_sight(gate.slug(), (u64::from(task_id) << 32) | u64::from(by)) {
            // `by` is the gate's own answer and it is also the only answer the
            // gate's alias predicate can give — it searches `task >> 1` alone,
            // so reading `by == task >> 1` back as a finding measures the
            // predicate. `owners` is the unfiltered version of the same
            // question: every task whose span actually covers this range, with
            // no relation assumed. If it ever holds an id that is not
            // `task >> 1`, the halving is not the relationship.
            //
            // `spans` is here for the other permissive arm: `no_spans` means "no
            // span for this task or its alias", not "the registry is empty", and
            // only the second of those is a bounds check that did not run.
            let owners = state.tasks_covering(gva, buf.len() as u64);
            crate::observe::Emit::decline("gva_write_gate", &gate)
                .field("task", task_id)
                .field("gva", format!("{gva:#x}"))
                .field("len", format!("{:#x}", buf.len()))
                .field("owners", format!("{owners:?}"))
                .field("spans", state.task_map_span_count())
                .fail();
        }
    }
    let Err(err) = crate::runtime::gva_view::write_span(state, host, task_id, gva, buf) else {
        return Ok(());
    };
    crate::observe::Emit::decline("gva_write", &err)
        .field("task", task_id)
        .field("gva", format!("{gva:#x}"))
        .field("len", format!("{:#x}", buf.len()))
        .fail();
    Err(err)
}

/// Resolve pages of `[gva, gva + span)` under the same task selection as
/// [`read_task_gva_fallback`] (wire id first, then `id >> 1`) and call `visit`
/// with each page-aligned GPA. Stops early when `visit` returns `false`.
/// `stride_pages` visits every Nth page plus always the last (1 = every page);
/// callers trade probe density against walk cost.
///
/// This is a lookup, not a validator: pages that fail to translate are
/// skipped silently — the content read that follows fails (and fail-logs) on
/// its own terms. One page-walk cache and one root read span the whole range.
#[allow(
    clippy::too_many_arguments,
    reason = "the visitor API exposes task, span, page geometry, and callback state explicitly"
)]
pub fn visit_task_gva_page_gpas<M: HostMemory>(
    host: &M,
    tasks: &[TaskEntry],
    task_id: u32,
    gva: u64,
    span: u64,
    page_shift: u32,
    stride_pages: u64,
    visit: &mut dyn FnMut(u64) -> bool,
) {
    if span == 0 || stride_pages == 0 {
        return;
    }
    let Some(geom) = geometry_for_page_shift(page_shift) else {
        return;
    };
    let reader = HostPhys(host);
    let mut root = None;
    for id in [task_id, task_id >> 1] {
        if root.is_some() {
            break;
        }
        let Some(task) = tasks.get(id as usize) else {
            continue;
        };
        if !task.active || task.directory_pfn == 0 {
            continue;
        }
        let gr_task = Task {
            active: true,
            directory_pfn: task.directory_pfn,
        };
        if let Ok(r) = read_task_root(&reader, &gr_task, geom) {
            root = Some(r);
        }
    }
    let Some(root) = root else {
        return;
    };
    let page = geom.page_size as u64;
    let first = gva & !(page - 1);
    let last = gva.saturating_add(span - 1) & !(page - 1);
    let step = page.saturating_mul(stride_pages);
    let mut cache = Cache::default();
    let mut cur = first;
    loop {
        let t = translate_root(
            &reader,
            geom,
            root.root_pfn,
            root.depth,
            cur,
            Some(&mut cache),
        );
        if t.status == ResolveStatus::Ok && !visit(t.gpa & !(page - 1)) {
            return;
        }
        if cur == last {
            return;
        }
        // Always end on the exact last page so span tails are covered.
        cur = cur.saturating_add(step).min(last);
    }
}

/// Translate one GVA to a GPA under the task directory (single page).
pub fn translate_task_gva<M: HostMemory>(
    host: &M,
    task: &TaskEntry,
    gva: u64,
    page_shift: u32,
) -> Option<u64> {
    if !task.active || task.directory_pfn == 0 {
        return None;
    }
    let geom = geometry_for_page_shift(page_shift)?;
    let reader = HostPhys(host);
    let gr_task = Task {
        active: true,
        directory_pfn: task.directory_pfn,
    };
    let root = read_task_root(&reader, &gr_task, geom).ok()?;
    let t = translate_root(&reader, geom, root.root_pfn, root.depth, gva, None);
    if t.status != ResolveStatus::Ok {
        return None;
    }
    Some(t.gpa)
}

/// One-line walk diagnosis for a single task slot (measure-only; no product gates).
///
/// Example: `tid=2 act=1 dir=0xabc root=0xdef depth=2 st=zero-pfn pte=0 lvl=1 idx=4`
pub fn diagnose_task_slot<M: HostMemory>(
    host: &M,
    task: &TaskEntry,
    task_id: u32,
    gva: u64,
    page_shift: u32,
) -> String {
    if !task.active {
        return format!(
            "tid={task_id} act=0 dir={:#x} st=inactive",
            task.directory_pfn
        );
    }
    if task.directory_pfn == 0 {
        return format!("tid={task_id} act=1 dir=0 st=no-directory");
    }
    let Some(geom) = geometry_for_page_shift(page_shift) else {
        return format!(
            "tid={task_id} act=1 dir={:#x} st=bad-page-shift({page_shift})",
            task.directory_pfn
        );
    };
    let reader = HostPhys(host);
    let gr_task = Task {
        active: true,
        directory_pfn: task.directory_pfn,
    };
    let root = match read_task_root(&reader, &gr_task, geom) {
        Ok(r) => r,
        Err(st) => {
            return format!(
                "tid={task_id} act=1 dir={:#x} st=root({})",
                task.directory_pfn,
                resolve_status_name(st)
            );
        }
    };
    let t = translate_root(&reader, geom, root.root_pfn, root.depth, gva, None);
    if t.status == ResolveStatus::Ok {
        format!(
            "tid={task_id} act=1 dir={:#x} root={:#x} depth={} st=ok gpa={:#x} leaf_pfn={:#x}",
            task.directory_pfn, root.root_pfn, root.depth, t.gpa, t.leaf_pfn
        )
    } else {
        format!(
            "tid={task_id} act=1 dir={:#x} root={:#x} depth={} st={} pte={:#x} lvl={} idx={}",
            task.directory_pfn,
            root.root_pfn,
            root.depth,
            resolve_status_name(t.status),
            t.raw_pte,
            t.level,
            t.entry_index
        )
    }
}

/// Diagnose walk under wire `task_id`, `task_id>>1`, and a few active peers.
///
/// Compact multi-clause string for one fail-log line (MapMemory2 / stage Unmapped).
pub fn diagnose_gva_walk<M: HostMemory>(
    host: &M,
    tasks: &[TaskEntry],
    task_id: u32,
    gva: u64,
    page_shift: u32,
) -> String {
    let mut parts: Vec<String> = Vec::with_capacity(6);
    let mut tried = std::collections::BTreeSet::new();
    let try_id = |id: u32, parts: &mut Vec<String>, tried: &mut std::collections::BTreeSet<u32>| {
        if !tried.insert(id) {
            return;
        }
        if (id as usize) >= tasks.len() {
            parts.push(format!("tid={id} st=oob"));
            return;
        }
        parts.push(diagnose_task_slot(
            host,
            &tasks[id as usize],
            id,
            gva,
            page_shift,
        ));
    };
    try_id(task_id, &mut parts, &mut tried);
    try_id(task_id >> 1, &mut parts, &mut tried);
    // Peer scan: active tasks with a directory (cap 4 extras) — catches wrong-task walks.
    let mut peers = 0u32;
    for (i, t) in tasks.iter().enumerate() {
        if peers >= 4 {
            break;
        }
        let id = i as u32;
        if tried.contains(&id) || !t.active || t.directory_pfn == 0 {
            continue;
        }
        try_id(id, &mut parts, &mut tried);
        peers += 1;
    }
    format!(
        "gva={gva:#x} page_shift={page_shift} | {}",
        parts.join(" || ")
    )
}

/// Snapshot of active task directories (for periodic map census).
pub fn format_active_tasks(tasks: &[TaskEntry]) -> String {
    let mut bits = Vec::new();
    for (i, t) in tasks.iter().enumerate() {
        if !t.active {
            continue;
        }
        bits.push(format!(
            "t{i}:dir={:#x},len={:#x},ol_pfn={:#x},ol_n={}",
            t.directory_pfn, t.length, t.object_list_pfn, t.object_list_count
        ));
    }
    if bits.is_empty() {
        "tasks=none".into()
    } else {
        format!("tasks[{}]={}", bits.len(), bits.join(";"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::endian::st32;
    use crate::observe::Decline;

    /// The collapse this migration ended: nineteen distinct checks — the walk's
    /// fifteen plus four of its own — all answered `MemError::Unmapped`, and
    /// `MemError` reached the always-on log at no site in the crate. So a
    /// malformed PTE, a zero root PFN, an inactive task and a genuinely unmapped
    /// GPA were one value, invisibly, on the guest-memory hot path.
    ///
    /// Asserted as "no two of these share a slug" rather than by naming each
    /// one, because the property that matters is the absence of aliasing.
    #[test]
    fn no_two_guest_memory_checks_answer_with_the_same_reason() {
        use crate::contract::gva_resolve::ResolveStatus as R;
        const WALK: &[R] = &[
            R::ErrArgs,
            R::ErrInactiveTask,
            R::ErrNoDirectory,
            R::ErrDirectoryRead,
            R::ErrZeroRootPfn,
            R::ErrZeroDepth,
            R::ErrDepthTooDeep,
            R::ErrAddressOutOfRange,
            R::ErrPageTableRead,
            R::ErrZeroPfn,
            R::ErrMalformedPte,
            R::ErrSpanOverflow,
            R::ErrVisitorStopped,
            R::ErrUnsupportedGeometry,
            R::ErrSpanTooLarge,
        ];
        let mut slugs: Vec<&str> = WALK
            .iter()
            .map(|r| MemError::Unresolved(*r).slug())
            .chain(
                [
                    MemError::Unmapped,
                    MemError::NoCpu,
                    MemError::Overflow,
                    MemError::BadArgs,
                    MemError::QemuReadGpaCallbackMissing,
                    MemError::QemuReadGpaCallbackFailed(-1),
                    MemError::QemuWriteGpaCallbackMissing,
                    MemError::QemuWriteGpaCallbackFailed(-1),
                    MemError::QemuReadKvaCallbackMissing,
                    MemError::QemuReadKvaCallbackFailed(-1),
                    MemError::XregUnavailable,
                    MemError::QemuReadXregCallbackMissing,
                    MemError::QemuReadXregCallbackFailed(-1),
                    MemError::NoTaskDirectory,
                    MemError::UnsupportedPageShift,
                    MemError::TaskRootRead,
                    MemError::NoSuchTask,
                    MemError::OutsideMap,
                    MemError::NotRam,
                    MemError::MapPagesRefused,
                    MemError::RunOutOfRange,
                ]
                .iter()
                .map(|e| e.slug()),
            )
            .collect();
        let total = slugs.len();
        assert_eq!(total, 36, "15 walk reasons + 21 memory reasons");
        slugs.sort_unstable();
        slugs.dedup();
        assert_eq!(
            slugs.len(),
            total,
            "two guest-memory checks share a reason slug"
        );

        // `Unresolved` must forward, not invent: the walk already named the
        // check, and a second name here would make two log lines disagree about
        // one event.
        assert_eq!(
            MemError::Unresolved(R::ErrMalformedPte).slug(),
            "gva_malformed_pte"
        );
        // And `Ok` inside `Unresolved` is a construction bug, named as one
        // rather than reported as a plausible walk reason.
        assert_eq!(MemError::Unresolved(R::Ok).slug(), "mem_unresolved_ok");
    }

    use crate::contract::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
    use crate::model::{DeviceId, DeviceState, PAGE_SHIFT_ARM64E, PAGE_SHIFT_X86};
    use crate::runtime::decode::resource::RESOURCE_PAGE_SHIFT;
    use crate::runtime::host::FakeHost;

    #[test]
    fn diagnose_reports_ok_and_zero_pfn() {
        let mut host = FakeHost::new();
        let dir_gpa = 2u64 << PAGE_SHIFT_X86;
        let root_gpa = 3u64 << PAGE_SHIFT_X86;
        let data_gpa = 4u64 << PAGE_SHIFT_X86;
        host.map_range(dir_gpa, 0x20, 0);
        host.map_range(root_gpa, 0x1000, 0);
        host.map_range(data_gpa, 0x100, 0xab);
        let mut d = [0u8; 8];
        st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 3);
        st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
        host.write_gpa(dir_gpa, &d).unwrap();
        // leaf PTE for page index 0 → pfn 4
        let mut pte = [0u8; 4];
        st32(&mut pte, 4);
        host.write_gpa(root_gpa, &pte).unwrap();
        let mut tasks: [TaskEntry; 4] = std::array::from_fn(|_| TaskEntry::default());
        tasks[1] = TaskEntry::define(0x1000, 2);
        let ok = diagnose_gva_walk(&host, &tasks, 1, 0, PAGE_SHIFT_X86);
        assert!(ok.contains("st=ok"), "{ok}");
        assert!(
            ok.contains("gpa=0x4000") || ok.contains("leaf_pfn=0x4"),
            "{ok}"
        );
        // unmapped page index 1
        let miss = diagnose_gva_walk(&host, &tasks, 1, 0x1000, PAGE_SHIFT_X86);
        assert!(
            miss.contains("zero-pfn") || miss.contains("st=zero"),
            "{miss}"
        );
    }

    #[test]
    fn one_level_gva_read() {
        let mut host = FakeHost::new();
        // directory at pfn 2, root table at pfn 3, leaf data at pfn 4
        let dir_gpa = 2u64 << RESOURCE_PAGE_SHIFT;
        let root_gpa = 3u64 << RESOURCE_PAGE_SHIFT;
        let data_gpa = 4u64 << RESOURCE_PAGE_SHIFT;
        host.map_range(dir_gpa, 0x20, 0);
        host.map_range(root_gpa, 0x4000, 0);
        host.map_range(data_gpa, 0x100, 0xab);
        // directory: root_pfn=3, depth=1
        let mut d = [0u8; 8];
        st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 3);
        st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
        let _ = host.write_gpa(dir_gpa, &d);
        // PTE for gva page 0: pfn 4
        st32(&mut d[..4], 4);
        let _ = host.write_gpa(root_gpa, &d[..4]);

        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        assert!(state.define_task(1, 0x1000, 2));
        let mut buf = [0u8; 4];
        assert!(read_task_gva(&host, &state.tasks[1], 0, &mut buf, PAGE_SHIFT_ARM64E).is_ok());
        assert_eq!(buf, [0xab; 4]);
        // Round-trip write.
        let out = [1u8, 2, 3, 4];
        assert!(write_task_gva(&mut host, &state.tasks[1], 0, &out, PAGE_SHIFT_ARM64E).is_ok());
        let mut back = [0u8; 4];
        assert!(read_task_gva(&host, &state.tasks[1], 0, &mut back, PAGE_SHIFT_ARM64E).is_ok());
        assert_eq!(back, out);
    }

    #[test]
    fn x86_4k_geometry_read() {
        let mut host = FakeHost::new();
        let page_shift = PAGE_SHIFT_X86;
        let dir_gpa = 2u64 << page_shift;
        let root_gpa = 3u64 << page_shift;
        let data_gpa = 4u64 << page_shift;
        host.map_range(dir_gpa, 0x20, 0);
        host.map_range(root_gpa, 0x1000, 0);
        host.map_range(data_gpa, 0x100, 0xcd);
        let mut d = [0u8; 8];
        st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 3);
        st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
        let _ = host.write_gpa(dir_gpa, &d);
        st32(&mut d[..4], 4);
        let _ = host.write_gpa(root_gpa, &d[..4]);

        let mut state = DeviceState::new(DeviceId(1), page_shift);
        assert!(state.define_task(1, 0x1000, 2));
        let mut buf = [0u8; 4];
        assert!(read_task_gva(&host, &state.tasks[1], 0, &mut buf, page_shift).is_ok());
        assert_eq!(buf, [0xcd; 4]);
    }

    #[test]
    fn unknown_page_shift_rejected() {
        assert!(geometry_for_page_shift(13).is_none());
        assert!(geometry_for_page_shift(0).is_none());
        assert!(geometry_for_page_shift(PAGE_SHIFT_X86).is_some());
        assert!(geometry_for_page_shift(PAGE_SHIFT_ARM64E).is_some());
    }

    #[test]
    fn product_write_outside_map_fails_when_spans_recorded() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        assert!(state.define_task(1, 0x1_0000, 2));
        // No MapMemory2 yet → allow (fixtures / pre-map).
        assert!(state.gva_write_allowed(1, 0x1000, 0x100));
        state.note_task_map(1, 0x2000, 0x1000);
        assert!(state.gva_write_allowed(1, 0x2000, 0x100));
        assert!(state.gva_write_allowed(1, 0x2f00, 0x100));
        assert!(!state.gva_write_allowed(1, 0x1000, 0x100)); // outside
        assert!(!state.gva_write_allowed(1, 0x2f00, 0x200)); // straddles end
                                                             // Product path fails closed with outside_map.
        let err = write_task_gva_product(&mut state, &mut host, 1, 0x1000, &[1, 2, 3, 4]);
        assert!(err.is_err());
        state.note_task_unmap(1, 0x2000, 0x1000);
        // Registry empty again → allow.
        assert!(state.gva_write_allowed(1, 0x1000, 0x100));
    }

    /// The gate's four answers must be distinguishable, because three of them
    /// are "allowed" and only one of those is a bounds check that happened.
    ///
    /// `gva_write_allowed` collapsed all of this to a `bool` and the caller
    /// supplied the word `mem_outside_map` on the refusal — the shape
    /// `AGENTS.md` calls out. A census built on the `bool` could not tell a
    /// covered write from one permitted because the registry was empty, which
    /// is exactly the state `delete_task` leaves behind.
    #[test]
    fn the_write_gate_names_which_arm_permitted_the_write() {
        use crate::model::WriteGate;
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        assert!(state.define_task(6, 0x1_0000, 2));

        // Nothing recorded for task 6 or its aliases: allowed by default, and
        // that is not a bounds check.
        assert_eq!(
            state.gva_write_gate(6, 0x2000, 0x100),
            WriteGate::NoSpans,
            "an empty registry is a gate that did not run, not a gate that passed"
        );

        // An exact span covers it.
        state.note_task_map(6, 0x2000, 0x1000);
        assert_eq!(state.gva_write_gate(6, 0x2000, 0x100), WriteGate::Exact);

        // Spans exist for this task but none covers the range.
        assert_eq!(state.gva_write_gate(6, 0x9000, 0x100), WriteGate::Outside);
        assert!(!state.gva_write_allowed(6, 0x9000, 0x100));

        // Only task 3's span covers it (6 >> 1 == 3), so the write is permitted
        // on another task's authorisation. Exact must still win when both apply.
        let mut aliased = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        assert!(aliased.define_task(3, 0x1_0000, 2));
        aliased.note_task_map(3, 0x4000, 0x1000);
        assert_eq!(
            aliased.gva_write_gate(6, 0x4000, 0x100),
            WriteGate::Aliased { by: 3 },
            "a span recorded by task 3 must not be reported as task 6's own"
        );
        aliased.note_task_map(6, 0x4000, 0x1000);
        assert_eq!(aliased.gva_write_gate(6, 0x4000, 0x100), WriteGate::Exact);
    }

    /// The alias predicate has exactly **one** candidate, and `by` therefore
    /// cannot report anything except `task >> 1`.
    ///
    /// It used to read `t == (task_id >> 1) || (t << 1) == task_id`, which reads
    /// as a search over two relations. Over every id the task table can hold
    /// those clauses accept exactly the same set — so `by == task >> 1` was
    /// forced by the predicate, not measured from the guest, and two rounds of
    /// notes read it as evidence of a factor of two on the write path.
    #[test]
    fn the_alias_predicate_never_had_a_second_candidate() {
        use crate::model::MAX_TASKS;
        for task_id in 0..MAX_TASKS as u32 {
            for t in 0..MAX_TASKS as u32 {
                assert_eq!(
                    t == (task_id >> 1),
                    t == (task_id >> 1) || (t << 1) == task_id,
                    "the deleted clause changed the answer for t={t} task={task_id}"
                );
            }
        }
    }

    /// The unfiltered owner scan sees what the gate's one-candidate search
    /// cannot, and it does not change what the gate decides.
    ///
    /// Task 9's span covers the range and task 5's does not, so a write by task
    /// 11 (`11 >> 1 == 5`) is refused — and the log line must be able to say
    /// that a *different* task, unrelated by halving, held the covering span.
    /// With `owners` reporting only what `by` could, "the covering owner is
    /// always `task >> 1`" would be true by construction on every boot.
    #[test]
    fn the_owner_readout_sees_covering_tasks_the_gate_never_considered() {
        use crate::model::WriteGate;
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        assert!(state.define_task(11, 0x1_0000, 2));
        state.note_task_map(5, 0x8000, 0x1000);
        state.note_task_map(9, 0x4000, 0x1000);
        state.note_task_map(4, 0x4000, 0x1000);

        assert_eq!(
            state.gva_write_gate(11, 0x4000, 0x100),
            WriteGate::Outside,
            "the gate's decision must not move because a readout was added"
        );
        assert_eq!(
            state.tasks_covering(0x4000, 0x100),
            vec![4, 9],
            "both covering owners, ascending, and neither is 11 >> 1"
        );
        assert_eq!(state.task_map_span_count(), 3);
        // A zero-length write covers nothing, matching the gate's own early out.
        assert!(state.tasks_covering(0x4000, 0).is_empty());
    }

    /// `no_spans` names what the gate searched, not what the registry holds, so
    /// the line has to carry the total or it reads as "the registry is empty".
    #[test]
    fn a_populated_registry_can_still_answer_no_spans() {
        use crate::model::WriteGate;
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        assert!(state.define_task(11, 0x1_0000, 2));
        state.note_task_map(7, 0x4000, 0x1000);
        assert_eq!(
            state.gva_write_gate(11, 0x4000, 0x100),
            WriteGate::NoSpans,
            "nothing filed by 11 or 5, so the bounds check did not run"
        );
        assert_eq!(state.task_map_span_count(), 1, "yet the registry is not empty");
        assert_eq!(state.tasks_covering(0x4000, 0x100), vec![7]);
    }

    /// The only inputs the deleted clause ever added were `u32` wraparound, and
    /// on those it failed **open**.
    ///
    /// `0x8000_0001 << 1` wraps to `2`, so a span filed by task `0x8000_0001`
    /// used to authorise a write by task 2 — arithmetic overflow presented as an
    /// ownership relation. Task 2 has spans of its own here and none covers the
    /// range, so the correct answer is the refusal `Outside`; the old predicate
    /// returned `Aliased` and let the write through.
    #[test]
    fn a_span_owned_by_a_wrapping_id_no_longer_authorises_another_tasks_write() {
        use crate::model::WriteGate;
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        assert!(state.define_task(2, 0x1_0000, 2));
        state.note_task_map(2, 0x8000, 0x1000);
        state.note_task_map(0x8000_0001, 0x4000, 0x1000);
        assert_eq!(0x8000_0001u32 << 1, 2, "the wrap this test is about");
        assert_eq!(
            state.gva_write_gate(2, 0x4000, 0x100),
            WriteGate::Outside,
            "a wrapped id is not an alias, and task 2's own spans do not cover this"
        );
        assert!(!state.gva_write_allowed(2, 0x4000, 0x100));
    }
}
