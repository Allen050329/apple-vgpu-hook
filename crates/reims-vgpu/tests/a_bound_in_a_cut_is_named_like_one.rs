//! A constant that cuts a walk short must be named so the bound scans see it.
//!
//! The three bound directions — `an_eviction_says_what_it_costs`,
//! `a_bounded_insert_says_what_it_drops`, `a_bounded_walk_says_what_it_skips` —
//! all filter by the constant's *name*, through the vocabulary
//! [`source_scan::is_bound`] holds. That is a deliberate choice with a
//! deliberate cost: a bound named for **what it limits** rather than for **the
//! fact that it limits** is invisible to all three, and its site is not reported
//! as unadjudicated, it is not reported at all.
//!
//! This test closes that hole for the one position where it can be closed
//! exactly. It reads the same lines through the same [`source_scan::cut_arguments`]
//! the walk scan reads, keeps the arguments that walk scan's filter *rejects*,
//! and fails on any that is spelled like a constant. The rule it states is one
//! sentence: **if a `SCREAMING_SNAKE_CASE` constant is the whole argument of a
//! `.take()` or a `.min()`, its name must say it is a bound.**
//!
//! # Why a gate, and not a wider vocabulary
//!
//! Widening [`source_scan::BOUND_WORDS`] answers the two constants that exist
//! today and nothing about the next one. `REPORTED` and `WORDS` were the two
//! spellings this tree happened to grow; `_SHOWN`, `_ECHOED`, `_SAMPLED`,
//! `_HEAD` and `_PREVIEW` are all equally natural and none of them is in any
//! vocabulary. A gate does not have to predict the spelling — it requires the
//! author to use one that is already understood, which is the same bargain
//! `decline_slugs_are_unique` and `a_reims_vgpu_prefix_means_the_c_boundary`
//! take.
//!
//! # Why this was worth writing rather than noting
//!
//! The walk scan's own module doc used to name `DOORBELL_OFFSETS_REPORTED` as
//! the single bound its vocabulary missed, and argue — correctly — that
//! widening three vocabularies for a log-line truncation was not worth it. The
//! argument held. **The count did not.** `UNKNOWN_OPCODE_ECHO_WORDS` sat in the
//! identical position in `model::state` and appeared in no sentence anywhere,
//! because a documented exception records the misses somebody happened to find
//! and then reads as a complete list. Both were harmless, which is exactly why
//! neither was ever revisited — the same reason the insert scan found two sites
//! a careful hand sweep had walked past.
//!
//! # What it does not cover, stated so this file is not read as exhaustive
//!
//! Only the `.take()`/`.min()` position, and only a bound spelled as a named
//! constant.
//!
//! - **A literal in a cut** — `.take(8)` — has no name to check. Swept by hand
//!   at the time this was written, through this file's own `cut_arguments`, so
//!   the population is the same one: **8** in the two guest-facing crates. Six
//!   narrow a diagnostic — two hex-dump previews, two bind-table summaries, a
//!   report tail, and `gva_mem`'s four-peer scan that builds a fail line
//!   explaining a failed walk. The other two are not walks at all: a channel
//!   value clamped to 255 and a shift width clamped to 64, both scalars that
//!   `.min` happens to spell the same way. None is over a decoded record. A gate
//!   here would have to judge the *receiver*, which is the discriminator the
//!   insert scan has and this position does not.
//! - **The eviction and insert positions.** Their bound sits in a comparison,
//!   and a comparison against a shouty constant is overwhelmingly a payload
//!   *length* precondition in this crate — measured: 58 distinct names, of which
//!   all but a handful end `_LEN`, `_OFF` or `_WORDS` and state how long a
//!   record must be before it is read, not how much of it will be. A gate over
//!   that position would be noise, and the two scans' own receiver test is what
//!   keeps them precise instead.
//! - **A bound that is not a constant at all**: a field, a capability reading, a
//!   value computed at runtime. Those are not bounds this device chose, which is
//!   what `Skip::Narrowing` and `Skip::NotAWalkBound` already say about the ones
//!   in [`SKIPS`].
//!
//! An integration test rather than a `#[cfg(test)]` module because it reads
//! source text and must run on every arm, including `backend-metal`, which this
//! development host can compile but cannot execute.

mod source_scan;
use source_scan::{cut_arguments, guest_facing_sources, is_bound, is_shouty};

/// A shouty cut argument that [`is_bound`] does not accept, and why it is not a
/// bound after all.
///
/// Empty, and it should stay that way — the fix for a real bound is to rename
/// it, which costs one line and buys adjudication in all three directions. An
/// entry here is for a constant that genuinely bounds nothing: a stride, an
/// index, a width that happens to sit in a cut. Write what it *is*, not that it
/// is safe.
const NOT_BOUNDS: &[(&str, &str)] = &[];

/// Every cut argument in the two guest-facing crates, as `(file, line, token)`.
fn cut_sites() -> Vec<(String, usize, String)> {
    let mut sites = Vec::new();
    for (file, text) in guest_facing_sources() {
        for (i, line) in text.lines().enumerate() {
            for arg in cut_arguments(line) {
                sites.push((file.clone(), i + 1, arg));
            }
        }
    }
    sites
}

#[test]
fn every_constant_cutting_a_walk_is_named_like_a_bound() {
    let sites = cut_sites();

    // The self-check every source scan in this directory carries: prove the scan
    // can see before believing what it cannot. Both `CUT` spellings and both
    // backends, since the Vulkan and Metal sites are `#[cfg]`-ed out of each
    // other and only source text sees them together. These are named as
    // *accepted* bounds, so a change that broke `cut_arguments` outright would
    // empty this list and fail here rather than passing vacuously.
    for (file, token, why) in [
        (
            "reims-vgpu/src/backend/metal/render.rs",
            "REIMS_VGPU_METAL_MAX_ATTRS",
            "the `.take(` spelling, on the Metal arm",
        ),
        (
            "reims-vgpu/src/backend/metal/stage_input.rs",
            "REIMS_VGPU_COMPUTE_STAGE_INPUT_MAX_LAYOUTS",
            "the `.min(` spelling, on the Metal arm",
        ),
        (
            "reims-vgpu/src/runtime/drain/census.rs",
            "DOORBELL_OFFSETS_REPORTED_MAX",
            "the constant this test was written for, on the always-built arm",
        ),
    ] {
        assert!(
            sites
                .iter()
                .any(|(f, _, t)| f == file && t == token && is_bound(t)),
            "the scan cannot see {token} in {file} ({why}), so nothing it \
             reports about any other cut can be believed"
        );
    }

    let unnamed: Vec<String> = sites
        .iter()
        .filter(|(_, _, token)| is_shouty(token) && !is_bound(token))
        .filter(|(_, _, token)| !NOT_BOUNDS.iter().any(|(name, _)| name == token))
        .map(|(file, line, token)| format!("  {file}:{line}  {token}"))
        .collect();

    assert!(
        unnamed.is_empty(),
        "a constant cuts a walk short and is not named like a bound, so all \
         three bound scans pass over its site without reporting it as \
         unadjudicated — they report nothing at all:\n{}\n\n\
         Rename it to carry one of {:?} — `FOO_MAX` and `MAX_FOO` are both in \
         use here — and add its verdict to `a_bounded_walk_says_what_it_skips`. \
         If it truly bounds nothing, put it in this file's `NOT_BOUNDS` with \
         what it is instead.",
        unnamed.join("\n"),
        source_scan::BOUND_WORDS,
    );
}

/// The gate is only as good as the vocabulary being single.
///
/// [`is_bound`] rejecting a path-qualified token is what keeps `u64::MAX` out of
/// every direction, and it is the rule the three scans had already parted on
/// while each carried a comment saying they had not. Asserted here rather than
/// left to a reading, because a vocabulary that is wrong in one scan and right
/// in another is invisible from inside either.
#[test]
fn the_shared_vocabulary_does_not_mistake_a_types_extreme_for_a_policy() {
    assert!(is_bound("MAX_MAPPINGS"));
    assert!(is_bound("DOORBELL_OFFSETS_REPORTED_MAX"));
    assert!(is_bound("cap"));
    assert!(!is_bound("u64::MAX"));
    assert!(!is_bound("usize::MAX"));
    assert!(!is_bound("HEADER_LEN"));
    assert!(!is_bound(""));
    // Bare `max` is the name of two std methods, and admitting it would put
    // `is_bound` beside almost every arithmetic clamp in the crate.
    assert!(!is_bound("max"));
}
