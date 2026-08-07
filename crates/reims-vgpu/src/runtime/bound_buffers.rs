//! Draw-time buffer binds, resolved once per reference and held.
//!
//! # The shape this follows
//!
//! Apple's host resolves a guest object reference to a host buffer **once**,
//! when the object is created, and stores it on the task under that reference.
//! Its render decoder then reads a `{u32 reference, u64 offset}` record per
//! bound slot, asks the task for the buffer by reference, and hands Metal the
//! buffer and the offset. No address translation happens on the draw path at
//! all — the page-run computation on that side is reachable only from the
//! map/unmap handlers, never from a decoder.
//!
//! This device resolved the bind instead: every bound buffer of every draw
//! walked the task page table over the bound span, coalesced the GPA-contiguous
//! stretches, and asked the host to alias each one. That is the same answer
//! every time until the guest changes a mapping, and the guest changes mappings
//! about four orders of magnitude less often than it draws.
//!
//! So this holds the resolution and the draw path looks it up.
//!
//! # What a held resolution is, and is not
//!
//! It is an **address** resolution: which host spans back this reference's
//! bytes right now. It is not the bytes. The runs point into this process's
//! import of guest RAM and the GPU reads them when the command buffer executes,
//! so a guest CPU write to those pages is picked up with nothing invalidated —
//! the same property the walking rail had, and the reason `CmdInvalidateResources`
//! and the exec resource table's validity quad do not appear anywhere here.
//! Content invalidation is not this module's business.
//!
//! Only an **address** change matters, and the guest announces every one of
//! them:
//!
//! * `CmdMapMemory2` / `CmdUnmapMemory` — the guest mutates the task page table
//!   and then notifies, carrying the exact `(task, gva, length)` that moved.
//!   Retired by range.
//! * `CmdReplacePhysical` — a GPA behind a GVA changed.
//! * `CmdSetObjectList` / `CmdDeleteObject` — a reference now names something
//!   else, or nothing.
//! * `CmdDefineTask2` / `CmdDeleteTask` — the page table root changed or the
//!   task is gone.
//!
//! The last four retire the whole task rather than a range. They are rare —
//! a driven boot sees single-digit `replace_physical` events against thousands
//! of draws a second — and a narrower rule would have to map an object id back
//! to the references that resolved through it, which is machinery bought with
//! nothing.
//!
//! # Why the key carries the offset
//!
//! Apple keys purely by reference, because their buffer covers the whole
//! allocation and the offset rides to Metal beside it. A resolution here covers
//! `[gva + offset, gva + size)` — the span the bind actually asked for — so two
//! binds of one reference at different offsets are two resolutions.
//!
//! Resolving the whole allocation once and slicing would collapse those, and it
//! is what Apple does, but it would also refuse a bind whose allocation has an
//! unmapped tail page even though the bind itself resolves. Apple can afford
//! that because `visitUnmappedRanges` tells them which sub-ranges are live;
//! this device has no such record, so it keeps the narrower key and the wider
//! admission. In practice a reference is bound at one offset and the two agree.
//!
//! # No capacity
//!
//! There is no cap and no eviction. The population is one entry per live
//! `(task, reference, offset)` a draw has actually bound, which the guest bounds
//! by its own working set, and every entry leaves through one of the retirement
//! rules above or through [`BoundBuffers::clear`] at device reset. A capacity
//! here would be a second, invisible reason for a resolution to disappear, and
//! the miss it caused would read as a mapping change that never happened.

use std::collections::HashMap;
use std::sync::Arc;

use crate::backend::vulkan::engine::GuestRun;
use crate::runtime::guest_ram_map::GuestWindowRun;

/// A resolved bind: where this reference's bytes live, as the engine binds them.
///
/// Both lists are `Arc`ed by the producer already, so a lookup hands the draw
/// path the same allocation the walk built rather than a copy of it.
#[derive(Clone, Debug)]
pub struct BoundBuffer {
    /// Guest VA the resolution starts at (the backing's `gva + offset`).
    pub gva: u64,
    /// Byte length the runs cover, and the bind's `total_len`.
    pub span: u64,
    /// Host-pointer spans the CPU gather walks.
    pub runs: Arc<Vec<GuestRun>>,
    /// The same bytes as bounded references into this process's import, when
    /// the host can import guest RAM at all. `None` keeps the caller on the
    /// gathering arm exactly as a fresh resolution would.
    pub pages: Option<Arc<Vec<GuestWindowRun>>>,
}

impl BoundBuffer {
    /// Whether this resolution's bytes overlap `[gva, gva + len)`.
    ///
    /// Half-open on both sides. A zero-length range overlaps nothing, which is
    /// what a map notify carrying no length means.
    fn overlaps(&self, gva: u64, len: u64) -> bool {
        if len == 0 || self.span == 0 {
            return false;
        }
        let a_end = self.gva.saturating_add(self.span);
        let b_end = gva.saturating_add(len);
        self.gva < b_end && gva < a_end
    }
}

/// `(task, reference, offset)` — see the module doc on why the offset is here.
type Key = (u32, u32, u64);

/// Every held bind resolution on this device.
#[derive(Default, Debug)]
pub struct BoundBuffers {
    held: HashMap<Key, BoundBuffer>,
}

impl BoundBuffers {
    /// The resolution for this bind, if one is held.
    pub fn get(&self, task_id: u32, buffer_ref: u32, offset: u64) -> Option<&BoundBuffer> {
        self.held.get(&(task_id, buffer_ref, offset))
    }

    /// Hold a freshly walked resolution.
    pub fn insert(&mut self, task_id: u32, buffer_ref: u32, offset: u64, bound: BoundBuffer) {
        self.held.insert((task_id, buffer_ref, offset), bound);
    }

    /// Drop everything held for one task.
    ///
    /// The answer for a page-table root change, a new object list, a deleted
    /// object and a deleted task: in every one of them a reference may now name
    /// different bytes, and which references is not knowable from the packet.
    pub fn retire_task(&mut self, task_id: u32) -> usize {
        let before = self.held.len();
        self.held.retain(|(t, _, _), _| *t != task_id);
        before - self.held.len()
    }

    /// Drop everything held for `task_id` whose bytes overlap `[gva, gva+len)`.
    ///
    /// The map/unmap answer, which carries the exact range that moved.
    pub fn retire_range(&mut self, task_id: u32, gva: u64, len: u64) -> usize {
        let before = self.held.len();
        self.held
            .retain(|(t, _, _), b| *t != task_id || !b.overlaps(gva, len));
        before - self.held.len()
    }

    /// Drop everything. Device reset, where no guest state survives.
    pub fn clear(&mut self) {
        self.held.clear();
    }

    /// How many resolutions are held, for the census.
    pub fn len(&self) -> usize {
        self.held.len()
    }

    /// Whether nothing is held.
    pub fn is_empty(&self) -> bool {
        self.held.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bound(gva: u64, span: u64) -> BoundBuffer {
        BoundBuffer {
            gva,
            span,
            runs: Arc::new(Vec::new()),
            pages: None,
        }
    }

    /// The lookup is keyed by all three of task, reference and offset, so no
    /// two binds can collide onto one resolution.
    #[test]
    fn a_resolution_is_found_only_by_its_own_key() {
        let mut b = BoundBuffers::default();
        b.insert(7, 3, 0, bound(0x1000, 0x2000));
        assert!(b.get(7, 3, 0).is_some());
        assert!(b.get(7, 3, 0x100).is_none(), "a different offset");
        assert!(b.get(7, 4, 0).is_none(), "a different reference");
        assert!(b.get(8, 3, 0).is_none(), "a different task");
    }

    /// A map/unmap notify retires exactly the resolutions whose bytes moved,
    /// and leaves the neighbours that did not.
    #[test]
    fn a_range_retire_takes_the_overlapping_resolutions_only() {
        let mut b = BoundBuffers::default();
        b.insert(1, 1, 0, bound(0x1000, 0x1000)); // [0x1000,0x2000)
        b.insert(1, 2, 0, bound(0x2000, 0x1000)); // [0x2000,0x3000)
        b.insert(1, 3, 0, bound(0x9000, 0x1000)); // far away
        assert_eq!(b.retire_range(1, 0x1800, 0x1000), 2, "spans the first two");
        assert!(b.get(1, 3, 0).is_some(), "the far one survives");
        assert_eq!(b.len(), 1);
    }

    /// A range retire is scoped to its task: the same GVA under another task is
    /// a different address space and must not be touched.
    #[test]
    fn a_range_retire_does_not_cross_tasks() {
        let mut b = BoundBuffers::default();
        b.insert(1, 1, 0, bound(0x1000, 0x1000));
        b.insert(2, 1, 0, bound(0x1000, 0x1000));
        assert_eq!(b.retire_range(1, 0x1000, 0x1000), 1);
        assert!(b.get(2, 1, 0).is_some());
    }

    /// A zero-length notify names no bytes and must retire nothing — otherwise
    /// a malformed packet would silently drop every resolution it touched.
    #[test]
    fn a_zero_length_range_retires_nothing() {
        let mut b = BoundBuffers::default();
        b.insert(1, 1, 0, bound(0x1000, 0x1000));
        assert_eq!(b.retire_range(1, 0x1000, 0), 0);
        assert_eq!(b.len(), 1);
    }

    /// Ranges that merely touch at an endpoint do not overlap, so an unmap of
    /// the page after a resolution does not retire it.
    #[test]
    fn abutting_ranges_do_not_overlap() {
        let mut b = BoundBuffers::default();
        b.insert(1, 1, 0, bound(0x1000, 0x1000)); // [0x1000,0x2000)
        assert_eq!(b.retire_range(1, 0x2000, 0x1000), 0, "starts where it ends");
        assert_eq!(b.retire_range(1, 0x0000, 0x1000), 0, "ends where it starts");
        assert_eq!(b.len(), 1);
    }

    /// A task retire takes that task's resolutions whatever their addresses,
    /// and leaves every other task alone.
    #[test]
    fn a_task_retire_takes_the_whole_task() {
        let mut b = BoundBuffers::default();
        b.insert(1, 1, 0, bound(0x1000, 0x1000));
        b.insert(1, 2, 0x40, bound(0x8000, 0x1000));
        b.insert(2, 1, 0, bound(0x1000, 0x1000));
        assert_eq!(b.retire_task(1), 2);
        assert_eq!(b.len(), 1);
        assert!(b.get(2, 1, 0).is_some());
    }
}
