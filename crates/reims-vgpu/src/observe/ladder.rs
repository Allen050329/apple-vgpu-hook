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
//! With one qualification, learned the hard way: that guarantee covers the
//! *condition* half, not the composition. `ladder_slugs!` forwarded its role as
//! a `literal` fragment, which is opaque to the token match that selects the
//! bare-role arms, so the empty role silently composed into `_wrong_type` and
//! two more like it — three fresh spellings, produced by the very macro meant to
//! stop them, all with a leading underscore that matched no documented grep. A
//! macro rules out the spellings a *call site* can write; what it expands to
//! still has to be asserted. `both_macros_spell_a_rung_the_same_way` is that
//! assertion, and it is why every wrapper around these macros needs one.
//!
//! # What does not belong here
//!
//! Only the four rungs above. A rail's *semantic* refusals — the descriptor
//! decoded and the resource is unusable — stay in the rail's own vocabulary,
//! because "the wire format is corrupt" and "the resource exists but has no
//! backing" are different findings and collapsing them would lose the one that
//! matters. `blit_exec`'s `buf_no_backing` is the example: it follows a
//! successful decode and is deliberately not a rung.
//!
//! # A healthy boot fires no rung at all
//!
//! Measured, so the next reader does not have to guess whether these paths are
//! exercised: a driven x86/Vulkan boot (Safari window drag, 2 752 posted events,
//! real motion, ~35 Hz median present) produces **zero** records matching any
//! of the four conditions. Every object reference the guest named resolved.
//!
//! So a green boot says the rails still work and says nothing whatever about the
//! rungs — they are held by tests, not by booting. A rung appearing in the fail
//! log is a real event worth reading, not background noise, and the fail
//! channel's whole reason set on that boot was three unrelated slugs.

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

/// The slug one role gives each rung of
/// [`crate::runtime::objects::resolve_descriptor`]'s refusal.
///
/// Expands to a closure, so a call site that has just been handed a
/// [`LadderRung`](crate::runtime::objects::LadderRung) turns it into that
/// rail's own `&'static str` in one expression:
///
/// ```ignore
/// objects::resolve_descriptor(state, host, task_id, buffer_ref, &[OBJECT_TYPE_BUFFER])
///     .map_err(|rung| br(BlitStatus::MissingResource, ladder_slugs!("buf")(rung)))?
/// ```
///
/// The match is exhaustive over the rungs, so adding one to the enum breaks
/// every rail here rather than letting a rail fall through to a catch-all —
/// which is the whole reason the rung is a value and not a string.
///
/// # Why `$role:tt` and not `$role:literal`
///
/// Because `ladder_slug!` selects its bare-role arms by matching the *token*
/// `""`, and a fragment captured as `literal` is an opaque AST node that no
/// longer matches a token pattern. Written `$role:literal`, this macro's
/// `ladder_slugs!("")` therefore fell past those arms into the composing ones
/// and emitted `_wrong_type`, `_no_list_entry`, `_desc_read` — leading
/// underscore and all. That is precisely the failure this module exists to
/// prevent: three more spellings of the four conditions, matching none of the
/// documented ones, so a `reason=wrong_type` grep of the fail log returned
/// every rail *except* the ones routing through here.
///
/// `tt` is one of the three fragment kinds that stay transparent to later token
/// matching, so the bare arms are reachable again. The composition is pinned
/// below for both macros; a change back to `literal` turns those tests red.
macro_rules! ladder_slugs {
    ($role:tt) => {
        |rung: $crate::runtime::objects::LadderRung| -> &'static str {
            match rung {
                $crate::runtime::objects::LadderRung::NoListEntry => {
                    $crate::observe::ladder_slug!($role, no_list_entry)
                }
                $crate::runtime::objects::LadderRung::WrongType { .. } => {
                    $crate::observe::ladder_slug!($role, wrong_type)
                }
                $crate::runtime::objects::LadderRung::DescRead { .. } => {
                    $crate::observe::ladder_slug!($role, desc_read)
                }
            }
        }
    };
}

pub(crate) use {ladder_slug, ladder_slugs};

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

    /// The plural macro must agree with the singular one for every role it can
    /// be given — including the empty one.
    ///
    /// It did not. `ladder_slugs!` captured its role as a `literal` fragment and
    /// handed it to `ladder_slug!`, whose bare-role arms select on the *token*
    /// `""`; a captured fragment is opaque to token matching, so the empty role
    /// fell through to the composing arms and every rail using `ladder_slugs!("")`
    /// emitted `_wrong_type` rather than `wrong_type`. The test above passed
    /// throughout, because it only ever exercised the singular macro — the arm
    /// that was already right.
    ///
    /// So this asserts the *composition* both spellings produce, which is the
    /// thing the fail log actually receives.
    #[test]
    fn both_macros_spell_a_rung_the_same_way() {
        use crate::runtime::objects::LadderRung;
        let rungs = [
            LadderRung::NoListEntry,
            LadderRung::WrongType { got: 7 },
            LadderRung::DescRead { declared_len: 32 },
        ];

        let bare = ladder_slugs!("");
        assert_eq!(
            rungs.map(bare),
            [
                ladder_slug!("", no_list_entry),
                ladder_slug!("", wrong_type),
                ladder_slug!("", desc_read),
            ],
            "a rail whose event name carries the domain gets the bare condition"
        );
        for slug in rungs.map(bare) {
            assert!(
                !slug.starts_with('_'),
                "`{slug}` is a spelling of a rung that no documented grep finds"
            );
        }

        let roled = ladder_slugs!("buf");
        assert_eq!(
            rungs.map(roled),
            [
                ladder_slug!("buf", no_list_entry),
                ladder_slug!("buf", wrong_type),
                ladder_slug!("buf", desc_read),
            ],
        );
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
