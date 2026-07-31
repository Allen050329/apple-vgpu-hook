//! Apply the guest's per-resource validity quad, from either producer.
//!
//! # Two producers, one record layout
//!
//! The guest states who owns a resource's authoritative bytes with four u8
//! fields — `clear_host_valid | set_host_valid | clear_guest_valid |
//! set_guest_valid` — and emits them from two places:
//!
//! - `pageBacking` → `CmdInvalidateResources` (`0x34`), 8-byte records, one
//!   hardcoded quad (`clear_host + set_guest`).
//! - `AppleParavirtCommandQueue::writeInvalidates` → the resource table inside
//!   every `EXEC_INDIRECT2` payload, 24-byte records, a quad computed per
//!   resource.
//!
//! The record *lengths* differ; the quad does not. Both decode through
//! [`InvalidateValidityOps`] and both land here, so the two paths cannot drift
//! into two different meanings for the same four bytes.
//!
//! # Why `clear_host_valid` has to do more than bump a generation
//!
//! `AppleParavirtResource::shouldInvalidateHost()` is a `lock btr` test-and-clear
//! of the resource's dirty bit plus a sticky flag it also clears, and
//! `writeInvalidates` is its only caller. So "the guest CPU-wrote this resource"
//! is delivered exactly once, in one submission's table, and is never resent.
//!
//! A pending deferred window for that resource holds pixels the device rendered
//! *before* that guest write. Landing it afterwards replaces bytes the guest
//! authored with bytes the guest has just declared stale — a full-extent clobber
//! of the guest's own work. `flush_all_windows_before_fence` cannot see this: it
//! decides *when* a window lands, and the answer here is that it must not land
//! at all. So a `clear_host_valid` drops the window rather than resequencing it.
//!
//! # Order within one quad
//!
//! Clear before set, in wire field order. `0x00000101` — both host bits in one
//! record — occurs in live traffic, and clear-then-set is the only reading under
//! which it is not self-contradictory: the guest wrote the resource, and this
//! submission then rewrites it.

use crate::model::{DeviceState, ResourceValidity};
use crate::runtime::decode::fifo::InvalidateValidityOps;

/// Which producer delivered a quad. Only used to name the counters, so an arm
/// can tell an exec-table statement from an invalidate-command one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValiditySite {
    ExecTable,
    InvalidateResources,
}

impl ValiditySite {
    fn slug(self) -> &'static str {
        match self {
            Self::ExecTable => "exec",
            Self::InvalidateResources => "inv",
        }
    }

    fn clear_host_route(self) -> &'static str {
        match self {
            Self::ExecTable => "validity_clr_host_exec",
            Self::InvalidateResources => "validity_clr_host_inv",
        }
    }
}

/// What one record changed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ValidityOutcome {
    /// Mappings whose `content_generation` this record advanced.
    pub bumped: u32,
    /// Deferred windows dropped because the guest's copy supersedes ours.
    pub windows_dropped: u32,
    /// The record named no mapping this device holds.
    pub missed: bool,
}

/// Apply one record's quad to whatever mapping state the object id names.
///
/// `task_id` is needed because a table id may be a texture ref rather than a
/// mapping id, and `texture_to_mapping` is per-task. Both are applied when both
/// resolve — the crate carries two registries for one guest object and a
/// statement about that object is a statement about both.
pub fn apply(
    state: &mut DeviceState,
    task_id: u32,
    object_id: u32,
    ops: InvalidateValidityOps,
    site: ValiditySite,
) -> ValidityOutcome {
    let mut out = ValidityOutcome::default();
    if object_id == 0 {
        // `writeInvalidates` skips null resources and id 0; `pageBacking` never
        // emits one. A zero id names nothing to apply to.
        return out;
    }
    let mut targets = vec![object_id];
    if let Some(&mid) = state.texture_to_mapping.get(&(task_id, object_id)) {
        if mid != object_id {
            targets.push(mid);
        }
    }
    let mut hit = false;
    for id in targets {
        if !state.mappings.contains_key(&id) {
            continue;
        }
        hit = true;
        if ops.clear_host_valid != 0 {
            // The guest wrote these pages after our last render into them. Our
            // copy is stale by the guest's own statement, so a pending window
            // must not land, and the next read must re-take the guest pages.
            out.windows_dropped = out
                .windows_dropped
                .saturating_add(drop_stale_windows(state, id, site));
            if let Some(m) = state.mappings.get_mut(&id) {
                m.content_generation = m.content_generation.saturating_add(1);
                out.bumped = out.bumped.saturating_add(1);
            }
        }
        let Some(m) = state.mappings.get_mut(&id) else {
            continue;
        };
        m.validity = next_validity(m.validity, ops);
    }
    out.missed = !hit;
    if ops.clear_host_valid != 0 {
        crate::runtime::drain::note_store_route(site.clear_host_route());
    }
    out
}

/// The quad applied to one validity pair, clear before set.
///
/// Split out from [`apply`] so the transition table is testable without a
/// device: it is the part that has to match the host framework's
/// `setIsHostValid:` / `setIsGuestValid:` semantics, and the part a second
/// producer could silently disagree with.
pub fn next_validity(prev: ResourceValidity, ops: InvalidateValidityOps) -> ResourceValidity {
    let mut next = prev;
    if ops.clear_host_valid != 0 {
        next.host_valid = false;
        next.host_stated = true;
    }
    if ops.set_host_valid != 0 {
        next.host_valid = true;
        next.host_stated = true;
    }
    if ops.clear_guest_valid != 0 {
        next.guest_valid = false;
        next.guest_stated = true;
    }
    if ops.set_guest_valid != 0 {
        next.guest_valid = true;
        next.guest_stated = true;
    }
    next
}

/// What the guest's last statement says about landing a deferred writeback into
/// a mapping's pages.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WritebackLicence {
    /// The guest last said the host copy is authoritative. The writeback is owed.
    Licensed,
    /// The guest has since said its own pages hold the newer bytes. Landing our
    /// frame would replace the guest's work with a copy it declared stale.
    Superseded,
    /// The guest has never named this resource in a validity quad, so there is
    /// no statement to obey either way.
    Unstated,
}

/// Read the licence for one mapping, counting the population as it goes.
///
/// Every landing is counted by outcome whether or not the caller enforces the
/// verdict. `Unstated` is the reason this is a census and not just a gate: the
/// safe reading of "the guest never said" is to allow the writeback, since
/// refusing it withholds the device's frame and turns a compositing layer black
/// — a failure this project has already paid a boot to discover once. The
/// counter is what makes tightening that direction provable rather than a guess:
/// an `unstated` rate that reaches zero is a rate that can be required.
pub fn writeback_licence(state: &DeviceState, mapping_id: u32) -> WritebackLicence {
    let validity = state
        .mappings
        .get(&mapping_id)
        .map(|m| m.validity)
        .unwrap_or_default();
    let licence = if !validity.host_stated {
        WritebackLicence::Unstated
    } else if validity.host_valid {
        WritebackLicence::Licensed
    } else {
        WritebackLicence::Superseded
    };
    crate::runtime::drain::note_store_route(match licence {
        WritebackLicence::Licensed => "validity_wb_licensed",
        WritebackLicence::Superseded => "validity_wb_superseded",
        WritebackLicence::Unstated => "validity_wb_unstated",
    });
    licence
}

/// Whether a landing writeback must be refused on the guest's own statement.
///
/// Only `Superseded` refuses, and only with the `writeback` rail armed-in — the
/// knob's default is to enforce, and `REIMS_VGPU_RESOURCE_VALIDITY_OFF=writeback`
/// is the control that lets the write through while still counting it.
pub fn writeback_refused(state: &DeviceState, mapping_id: u32) -> bool {
    writeback_licence(state, mapping_id) == WritebackLicence::Superseded
        && !crate::observe::resource_validity_disabled("writeback")
}

/// Take the mapping's pending windows, or count what would have been taken.
///
/// The count is reported either way, which is what makes the knob a measurement
/// rather than a hiding place: a control boot reports the same
/// `validity_windows_dropped` number the armed boot acts on, so the two differ
/// only in whether those windows landed.
fn drop_stale_windows(state: &mut DeviceState, mapping_id: u32, site: ValiditySite) -> u32 {
    let pending = state.deferred_flush_window_count(mapping_id);
    if pending == 0 {
        return 0;
    }
    crate::runtime::drain::note_store_route_n("validity_windows_dropped", pending as u64);
    if crate::observe::resource_validity_disabled(site.slug()) {
        return 0;
    }
    crate::runtime::storage_flush::drop_windows(
        state,
        mapping_id,
        match site {
            ValiditySite::ExecTable => "guest_wrote_resource_exec",
            ValiditySite::InvalidateResources => "guest_wrote_resource_inv",
        },
    );
    pending
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{DeviceId, PAGE_SHIFT_X86};

    fn quad(clr_h: u8, set_h: u8, clr_g: u8, set_g: u8) -> InvalidateValidityOps {
        InvalidateValidityOps {
            clear_host_valid: clr_h,
            set_host_valid: set_h,
            clear_guest_valid: clr_g,
            set_guest_valid: set_g,
        }
    }

    /// `0x00000101` — both host bits in one record — is live traffic. Clear
    /// before set is the only reading under which it is not self-contradictory.
    #[test]
    fn a_record_carrying_both_host_bits_ends_host_valid() {
        let after = next_validity(ResourceValidity::default(), quad(1, 1, 0, 0));
        assert!(after.host_valid);
        assert!(after.host_stated);
    }

    /// An op the record does not carry must leave its bit alone, including the
    /// "never stated" flag — otherwise every quad would look like a statement
    /// about all four bits.
    #[test]
    fn an_absent_op_states_nothing() {
        let after = next_validity(ResourceValidity::default(), quad(1, 0, 0, 0));
        assert!(after.host_stated);
        assert!(!after.guest_stated, "guest side was never mentioned");
        assert!(!after.guest_valid);
    }

    /// Pageon's hardcoded quad: the host copy goes stale, the guest pages become
    /// authoritative.
    #[test]
    fn the_pageon_quad_hands_ownership_to_the_guest() {
        let after = next_validity(ResourceValidity::default(), InvalidateValidityOps::PAGEON);
        assert!(!after.host_valid);
        assert!(after.guest_valid);
        assert!(after.host_stated && after.guest_stated);
    }

    /// The whole point of consuming `clear_host_valid`: a mapping the guest says
    /// it wrote must not keep a deferred window that would replay stale pixels
    /// over the guest's bytes.
    #[test]
    fn clearing_host_valid_drops_the_mappings_pending_windows() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let m = state.mappings.entry(9).or_default();
        m.mapped = true;
        m.width = 64;
        m.height = 64;
        arm_one_window(&mut state, 9);
        assert_eq!(state.deferred_flush_window_count(9), 1);

        let before_gen = state.mappings[&9].content_generation;
        let out = apply(&mut state, 0, 9, quad(1, 0, 0, 0), ValiditySite::ExecTable);
        assert_eq!(out.windows_dropped, 1);
        assert_eq!(out.bumped, 1);
        assert!(!out.missed);
        assert_eq!(state.deferred_flush_window_count(9), 0);
        assert_eq!(state.mappings[&9].content_generation, before_gen + 1);
        assert!(state.mappings[&9].validity.host_stated);
        assert!(!state.mappings[&9].validity.host_valid);
    }

    /// A licence with no `clear_host_valid` must leave the pending window in
    /// place: the submission is about to render into this resource, and dropping
    /// its window would throw away the frame the device already produced.
    #[test]
    fn a_set_host_record_keeps_the_pending_window() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        state.mappings.entry(9).or_default().mapped = true;
        arm_one_window(&mut state, 9);
        let out = apply(&mut state, 0, 9, quad(0, 1, 0, 0), ValiditySite::ExecTable);
        assert_eq!(out.windows_dropped, 0);
        assert_eq!(state.deferred_flush_window_count(9), 1);
        assert!(state.mappings[&9].validity.host_valid);
    }

    /// A texture ref and the mapping it resolves to are one guest resource. A
    /// statement about the ref that stopped at the ref would leave the mapping's
    /// window armed, which is the case this whole path exists to prevent.
    #[test]
    fn a_statement_about_a_texture_ref_reaches_its_mapping() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        state.mappings.entry(77).or_default().mapped = true;
        state.texture_to_mapping.insert((4, 12), 77);
        arm_one_window(&mut state, 77);
        let out = apply(&mut state, 4, 12, quad(1, 0, 0, 0), ValiditySite::ExecTable);
        assert_eq!(out.windows_dropped, 1);
        assert_eq!(state.deferred_flush_window_count(77), 0);
    }

    /// An id no registry answers for is reported, not silently skipped.
    #[test]
    fn an_unknown_object_is_reported_as_a_miss() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let out = apply(&mut state, 0, 4242, quad(1, 0, 0, 0), ValiditySite::ExecTable);
        assert!(out.missed);
        assert_eq!(out.bumped, 0);
    }

    #[test]
    fn object_id_zero_applies_to_nothing() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        state.mappings.entry(0).or_default().mapped = true;
        let out = apply(&mut state, 0, 0, quad(1, 0, 0, 0), ValiditySite::ExecTable);
        assert_eq!(out, ValidityOutcome::default());
    }

    /// A mapping the guest has never named must not have its writeback refused.
    /// Refusing withholds the device's frame, which is a compositing layer going
    /// black — a strictly worse failure than landing a frame nobody vouched for.
    #[test]
    fn a_never_stated_mapping_is_not_refused() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        state.mappings.entry(5).or_default().mapped = true;
        assert_eq!(writeback_licence(&state, 5), WritebackLicence::Unstated);
        assert!(!writeback_refused(&state, 5));
    }

    #[test]
    fn a_licensed_mapping_is_owed_its_writeback() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        state.mappings.entry(5).or_default().mapped = true;
        apply(&mut state, 0, 5, quad(0, 1, 0, 0), ValiditySite::ExecTable);
        assert_eq!(writeback_licence(&state, 5), WritebackLicence::Licensed);
        assert!(!writeback_refused(&state, 5));
    }

    /// The gate: once the guest says its own pages are authoritative, a window
    /// armed before that statement must not land over them.
    #[test]
    fn a_superseded_mapping_refuses_its_writeback() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        state.mappings.entry(5).or_default().mapped = true;
        apply(&mut state, 0, 5, quad(0, 1, 0, 0), ValiditySite::ExecTable);
        apply(&mut state, 0, 5, quad(1, 0, 0, 0), ValiditySite::ExecTable);
        assert_eq!(writeback_licence(&state, 5), WritebackLicence::Superseded);
        assert!(writeback_refused(&state, 5));
    }

    /// A mapping this device does not hold has no statement either way, and the
    /// flush rails' own `map_generation` guard is what refuses those.
    #[test]
    fn an_absent_mapping_reads_as_unstated() {
        let state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        assert_eq!(writeback_licence(&state, 999), WritebackLicence::Unstated);
    }

    fn arm_one_window(state: &mut DeviceState, mapping_id: u32) {
        let key = crate::model::ComputeStorageResidencyKey {
            mapping_id,
            map_generation: 0,
            surface_offset: 0,
            surface_bpr: 64 * 4,
            span_end: 64 * 64 * 4,
            width: 64,
            height: 64,
            pixel_format: 0x50,
            texture_ref: 0,
        };
        state.compute_deferred_flush.insert(
            key,
            crate::model::DeferredOwner::Render {
                armed_seq: 0,
                armed_stamp_seq: 0,
                source: crate::model::RenderWindowSource::Owned(std::sync::Arc::new(vec![
                    0u8;
                    64 * 64 * 4
                ])),
            },
        );
    }
}
