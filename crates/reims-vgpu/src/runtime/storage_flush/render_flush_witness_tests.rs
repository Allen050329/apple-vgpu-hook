use super::report::{
    note_render_flush_cache_read, note_render_flush_landed, note_render_flush_pages_read,
};
use crate::model::{DeviceId, DeviceState, MappingEntry, PAGE_SHIFT_X86};

fn state_with_mapping(mid: u32) -> DeviceState {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    state.mappings.insert(
        mid,
        MappingEntry {
            mapped: true,
            ..Default::default()
        },
    );
    state
}

/// Nothing to score on the first landing: a surface that has only just
/// arrived has no previous flush whose copies anything could have read.
#[test]
fn the_first_landing_of_a_mapping_scores_nothing() {
    let mut state = state_with_mapping(7);
    assert_eq!(note_render_flush_landed(&mut state, 7, true), None);
    let w = state.mappings[&7].render_flush;
    assert_eq!(
        (w.landed, w.cache_unread, w.pages_unread),
        (true, true, true)
    );
}

/// The age is stamped from the landing, not left at the `Default` zero: a
/// zero stamp would score every second landing as a frame-plus survivor and
/// hide exactly the burst case the bucket exists to find.
#[test]
fn a_landing_stamps_the_time_it_landed() {
    let mut state = state_with_mapping(7);
    note_render_flush_landed(&mut state, 7, true);
    let first = state.mappings[&7].render_flush.landed_us;
    assert!(first > 0, "landing must stamp a live clock reading");
    note_render_flush_landed(&mut state, 7, true);
    assert!(
        state.mappings[&7].render_flush.landed_us >= first,
        "each landing re-stamps"
    );
}

/// The whole point of the witness: a flush neither leg was read from is
/// reported as unread, which is what says the readback bought nothing.
#[test]
fn a_landing_nothing_read_scores_both_legs_unread() {
    let mut state = state_with_mapping(7);
    note_render_flush_landed(&mut state, 7, true);
    let scored = note_render_flush_landed(&mut state, 7, true).expect("second landing scores");
    assert!(scored.cache_unread && scored.pages_unread);
}

/// Each reader clears only the copy it took. A cache hit must not excuse
/// the guest-page write, or a flush whose pages nothing reads would be
/// scored as consumed and the write leg would look owed when it is not.
#[test]
fn each_leg_is_cleared_only_by_its_own_reader() {
    let mut state = state_with_mapping(7);
    note_render_flush_landed(&mut state, 7, true);
    note_render_flush_cache_read(&mut state, 7);
    let scored = note_render_flush_landed(&mut state, 7, true).expect("second landing scores");
    assert!(!scored.cache_unread, "cache read must clear the cache leg");
    assert!(
        scored.pages_unread,
        "cache read must not clear the pages leg"
    );

    note_render_flush_pages_read(&mut state, 7);
    let scored = note_render_flush_landed(&mut state, 7, true).expect("third landing scores");
    assert!(!scored.pages_unread, "pages read must clear the pages leg");
    assert!(
        scored.cache_unread,
        "pages read must not clear the cache leg"
    );
}

/// A flush that stored no cache copy has no cache leg to score.
///
/// `render_flush_cache_unread` is the number a future reader would use to
/// decide whether the cache leg is worth keeping, and a borrowed-frame flush
/// stores nothing — it drops the entry, because the memory holding its frame
/// goes straight back to the readback pool. Arming the leg anyway would
/// report a copy that was never made, once per flush, at the rate the guest
/// paints: a counter that looks like a measurement and is an artefact.
///
/// The pages leg is asserted alongside it, because it is the one that stays
/// meaningful and the two must not be conflated.
#[test]
fn a_flush_that_stored_no_cache_copy_arms_no_cache_leg() {
    let mut state = state_with_mapping(7);
    note_render_flush_landed(&mut state, 7, false);
    let w = state.mappings[&7].render_flush;
    assert!(!w.cache_stored, "no copy was stored");
    assert!(
        !w.cache_unread,
        "an absent copy must not be armed as an unread one"
    );
    assert!(w.pages_unread, "the guest pages were still written");

    // And the scoring of the previous landing skips the leg rather than
    // reporting it either way.
    let scored = note_render_flush_landed(&mut state, 7, true).expect("second landing scores");
    assert!(!scored.cache_stored);

    // A stored copy still scores normally, so the gate narrows the count
    // rather than silencing it.
    note_render_flush_cache_read(&mut state, 7);
    let scored = note_render_flush_landed(&mut state, 7, true).expect("third landing scores");
    assert!(scored.cache_stored && !scored.cache_unread);
}

/// A read attributed to a mapping the mapper no longer holds is dropped
/// rather than resurrecting an entry: the cache outlives its mapping, so a
/// late read of a stale entry must not create mapping state.
#[test]
fn a_read_of_an_unknown_mapping_creates_nothing() {
    let mut state = state_with_mapping(7);
    note_render_flush_cache_read(&mut state, 9);
    note_render_flush_pages_read(&mut state, 9);
    assert!(!state.mappings.contains_key(&9));
    assert_eq!(note_render_flush_landed(&mut state, 9, true), None);
}
