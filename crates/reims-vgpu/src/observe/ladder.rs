//! One vocabulary for the four ways an object-list lookup fails.
//!
//! Roughly twenty rails resolve a guest object reference the same way:
//!
//! ```text
//! lookup_list_entry(ref)   -> the guest never put anything under this ref
//! entry.object_type == ?   -> something is there, but not the kind asked for
//! read_descriptor(entry)   -> the entry names bytes that cannot be read
//! decode_*_descriptor(..)  -> the bytes are there and are not that descriptor
//! ```
//!
//! Each rail names its own failures, and by the time this module was written
//! those four conditions had **ten spellings** between them. The first rung
//! alone was written as `_no_list_entry` (10 sites), `_no_entry` (8) and
//! `_entry_missing` (7), split so evenly that no spelling was the one to grep
//! for; `TextureViewDecline` added an eleventh with `_hop_descriptor_missing`.
//!
//! That is not a tidiness problem. `reason=` is how the fail log is counted, so
//! ten spellings mean the question "how often did the guest name an object that
//! is not in the list" had no answer — every grep returned a subset and looked
//! complete. `AGENTS.md` says to filter the channel before ranking `reason=`;
//! it cannot help if the ranking is over three names for one thing.
//!
//! # Why a macro and not four constants
//!
//! The slug has to stay `&'static str`: `Decline::slug` returns one, and
//! `blit_exec` stashes one in a `Cell<&'static str>` that a later emitter reads.
//! And the **role** must survive — `blit_exec`'s reason travels to an emitter
//! that has lost the call site, so `buf_` versus `tex_` is the only thing left
//! saying which resource failed. A runtime `format!` cannot produce a `'static`
//! slug and a plain constant cannot carry the role, so the composition happens
//! at compile time.
//!
//! The point of doing it here rather than by convention is that a fifth
//! spelling stops being possible. `ladder_slug!("depth_stencil", entry_missing)`
//! is not a compile error message about style — it does not compile at all,
//! because no arm matches it.
//!
//! # What does not belong here
//!
//! Only the four rungs above. A rail's *semantic* refusals — the descriptor
//! decoded and the resource is unusable — stay in the rail's own vocabulary,
//! because "the wire format is corrupt" and "the resource exists but has no
//! backing" are different findings and collapsing them would lose the one that
//! matters. `blit_exec`'s `buf_no_backing` is the example: it follows a
//! successful decode and is deliberately not a rung.

/// Compose a ladder slug from a rail's role and one of the four rungs.
///
/// The role is whatever the rail already used to say which of its resources
/// failed (`"buf"`, `"tex"`, `"icb_type1"`, `"compute_stage_buf"`), and is kept
/// exactly as it was. Only the condition half is fixed, which is the half that
/// had ten spellings.
///
/// A rail whose event name already carries the domain passes `""` — see
/// `compute_exec`'s `compute_load_mtlb fail reason=…`, where the reason is
/// bare on purpose because the event says which load it was.
///
/// ```ignore
/// ladder_slug!("buf", no_list_entry)  // "buf_no_list_entry"
/// ladder_slug!("", wrong_type)        // "wrong_type"
/// ```
macro_rules! ladder_slug {
    ("", no_list_entry) => {
        "no_list_entry"
    };
    ("", wrong_type) => {
        "wrong_type"
    };
    ("", desc_read) => {
        "desc_read"
    };
    ("", desc_decode) => {
        "desc_decode"
    };
    // The guest put nothing under this ref. Expected while the guest is still
    // populating a task's list, which is why several rails resolve it quietly.
    ($role:literal, no_list_entry) => {
        concat!($role, "_no_list_entry")
    };
    // Something is under the ref and it is not the kind this rail asked for.
    ($role:literal, wrong_type) => {
        concat!($role, "_wrong_type")
    };
    // The entry names descriptor bytes that could not be read — the ref is live
    // but its descriptor GVA is not mapped right now.
    ($role:literal, desc_read) => {
        concat!($role, "_desc_read")
    };
    // The bytes were read and are not that descriptor.
    ($role:literal, desc_decode) => {
        concat!($role, "_desc_decode")
    };
}

pub(crate) use ladder_slug;

#[cfg(test)]
mod tests {
    /// The composition, stated once so a later edit to an arm is visible.
    #[test]
    fn a_role_and_a_rung_compose_into_the_slug_the_rail_emits() {
        assert_eq!(ladder_slug!("buf", no_list_entry), "buf_no_list_entry");
        assert_eq!(ladder_slug!("tex", wrong_type), "tex_wrong_type");
        assert_eq!(ladder_slug!("icb_type1", desc_read), "icb_type1_desc_read");
        assert_eq!(
            ladder_slug!("compute_stage_buf", desc_decode),
            "compute_stage_buf_desc_decode"
        );
    }

    /// A rail whose event name already says which load this was passes no role,
    /// and gets the bare condition rather than a leading underscore.
    #[test]
    fn an_empty_role_yields_the_bare_condition() {
        assert_eq!(ladder_slug!("", no_list_entry), "no_list_entry");
        assert_eq!(ladder_slug!("", wrong_type), "wrong_type");
        assert_eq!(ladder_slug!("", desc_read), "desc_read");
        assert_eq!(ladder_slug!("", desc_decode), "desc_decode");
    }

    /// Every slug is a `&'static str` usable where a `const` is required, which
    /// is what `Decline::slug` and `blit_exec`'s `Cell<&'static str>` need and
    /// what a runtime `format!` could not have given.
    #[test]
    fn a_composed_slug_is_a_compile_time_constant() {
        const BUF_MISS: &str = ladder_slug!("buf", no_list_entry);
        assert_eq!(BUF_MISS, "buf_no_list_entry");
    }
}
