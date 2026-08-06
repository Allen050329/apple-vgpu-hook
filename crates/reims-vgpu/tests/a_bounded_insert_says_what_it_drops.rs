//! A collection this device refuses to *grow* must say what the refusal drops.
//!
//! [`an_eviction_says_what_it_costs`] closed one direction of the standing
//! goal: nothing already admitted may be quietly forgotten to stay under a
//! number. This is the other direction, and no removal scan can see it —
//! there is nothing to remove. The site simply does not record what the guest
//! sent:
//!
//! ```ignore
//! if out.layouts.len() < MAX_COMPUTE_STAGE_INPUT_LAYOUTS {
//!     out.layouts.push(decoded);
//! }                                    // <- and if it does not, then what?
//! ```
//!
//! A skipped insert is worse than an eviction in one specific way. An evicted
//! entry was, at some point, present and correct; something downstream may have
//! consumed it, and the collection's *shape* still records that a bound bit. A
//! skipped insert leaves a collection that looks complete. A reader cannot tell
//! a list of 31 layouts the guest sent from a list of 31 the guest sent 40 of,
//! unless the site said so at the moment it decided.
//!
//! # What this asserts
//!
//! The same thing its sibling asserts, in the same shape: not that a particular
//! answer is right, but that the question was **asked**. Every insertion this
//! workspace gates on a capacity appears in [`DROPS`] with a verdict, and a new one
//! fails until its author writes the line.
//!
//! [`Drop_::LosesGuestWork`] exists for the same reason it does there — so an
//! author who answers honestly gets a failing build naming the architecture
//! rather than being tempted into one of the gentler words.
//!
//! # Why the receiver has to match
//!
//! The obvious scan — "a `len()` comparison against a bound, with a `push` near
//! it" — was tried and is useless here. This crate compares a *payload's* length
//! against a *needed* length before almost every decode, and those sit directly
//! above the loop that pushes what they validated. That is a short-record check,
//! not a capacity policy, and it floods.
//!
//! So the rule is tighter: the same collection expression must appear on both
//! sides — `x.len()` compared against a bound, and `x.push`/`x.insert` inside
//! the window. `bytes.len() < need` above `out.layouts.push(..)` does not match,
//! because `bytes` is not `out.layouts`. That one condition is what takes this
//! scan from unusable to four hits across both guest-facing crates.
//!
//! # It found two that a hand sweep did not
//!
//! This class was swept by hand first, carefully, and the sweep reported two
//! sites. The scan reports four. The two it added are both harmless — a census
//! dedup set and a ring that evicts rather than skipping — which is exactly why
//! a reader looking for danger walked past them, and exactly why the answer
//! belongs in a file that re-derives it on every run instead of in a note
//! saying somebody once looked.
//!
//! [`an_eviction_says_what_it_costs`]: ../an_eviction_says_what_it_costs/index.html

mod source_scan;
use source_scan::{guest_facing_sources, is_bound};

/// What is lost when this site declines to record something.
///
/// Three of these are unused today. They stay because they are the vocabulary
/// the failure message offers an author, and a verdict list that only contains
/// the answers already given invites the next person to bend one of them.
#[allow(
    dead_code,
    reason = "the vocabulary is the point; LosesGuestWork is kept unused by the \
              assertion below, and the other two by there being no such site yet"
)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Drop_ {
    /// The bound is the wire encoding's own, so the guest cannot express a
    /// request that exceeds it. Nothing is dropped because nothing can arrive:
    /// the skip is unreachable from any stream the serializer can write. The
    /// strongest verdict available, and the only one that needs no downstream
    /// behaviour to be true — but it is a claim about the *wire*, so it must
    /// name the field width or mask that makes it so.
    Unencodable,
    /// The skip is counted, and something downstream refuses the whole request
    /// on that count. The guest gets an error rather than a silently truncated
    /// result, which is what a GPU does when it cannot honour a call.
    RefusesTheRequest,
    /// The collection is a cache of things that can be made again, so a skipped
    /// insert costs the work of remaking one and nothing else.
    Recomputable,
    /// The collection is a witness, a census, or a log-dedup set. A skipped
    /// insert costs a reading or a log line, never guest work — and the skip
    /// says so, which is what keeps the reading it weakens honest.
    Observability,
    /// The scan matched a comparison that does not gate the insert: the bound is
    /// checked and then an *eviction* is made to fit the new entry, which always
    /// lands. Classified rather than filtered out, for the reason its sibling
    /// classifies `NotABound` — "the scan should not have flagged this" is worth
    /// writing down once instead of re-deriving at every reading. The cost of
    /// the eviction it triggers belongs in `an_eviction_says_what_it_costs`, and
    /// the entry here must say so, so neither test can be the only one holding
    /// the site.
    NotAGate,
    /// Guest work with no other holder, silently not recorded. **Forbidden**,
    /// and asserted absent below.
    LosesGuestWork,
}

/// Every capacity-gated insertion in the two guest-facing crates, and what a
/// skip drops.
///
/// Keyed by `(file, receiver)` with the number of sites, so a second insert
/// added beside an existing one moves the count and fails rather than
/// inheriting a verdict written about a different line.
const DROPS: &[(&str, &str, usize, Drop_, &str)] = &[
    (
        "reims-vgpu/src/runtime/decode/resource/mod.rs",
        "out.layouts",
        1,
        Drop_::Unencodable,
        "MAX_COMPUTE_STAGE_INPUT_LAYOUTS is COMPUTE_STAGE_INPUT_HEADER0_COUNT_MASK, \
         the wire field's own width, pinned by a `const` assertion at the \
         declaration — a guest cannot encode a 32nd layout. Belt and braces \
         anyway: the skip increments `dropped_layouts`, and compute_exec refuses \
         the whole pipeline as `stage_input_over_cap` if it is non-zero",
    ),
    (
        "reims-vgpu/src/runtime/decode/resource/mod.rs",
        "out.attributes",
        1,
        Drop_::Unencodable,
        "MAX_COMPUTE_STAGE_INPUT_ATTRS, same mask and same `const` assertion; \
         `dropped_attributes` and the same pipeline refusal",
    ),
    (
        "reims-vgpu/src/runtime/objects/mod.rs",
        "seen",
        1,
        Drop_::Observability,
        "note_type4_surface_shape's distinct-shape census; MAX_SHAPES bounds how \
         many descriptor shapes get reported per boot and gates nothing but \
         emission. The 25th shape is not recorded — and the cap firing is itself \
         reported as `type4_desc_shape outcome=cap_reached`, which the emitter's \
         own doc explains: a silent truncation would read as 'we saw everything', \
         the exact error this probe exists to rule out",
    ),
    (
        "reims-vgpu/src/backend/vulkan/engine/pools/images_and_registry.rs",
        "self.reclaimed_recent",
        1,
        Drop_::NotAGate,
        "note_resident_reclaimed pops the front when the ring is full and then \
         pushes unconditionally, so no insert is ever skipped. The `pop_front` is \
         the real event and `an_eviction_says_what_it_costs` carries it, as \
         Observability: RECLAIM_HISTORY bounds how far back a census can look and \
         nothing reads it to decide device behaviour",
    ),
];

/// Ways a collection grows.
const INSERT: &[&str] = &["push", "push_back", "push_front", "insert", "extend"];

/// How far above an insertion to look for the comparison that gates it.
///
/// Much tighter than the eviction scan's twenty. A capacity gate on an insert is
/// an `if` whose body *is* the insert, so it is within a line or two; widening
/// this only picks up unrelated `if`s in the same function and does not find a
/// single extra real site. Measured: at 20 the scan reports 30 candidates and
/// still exactly 2 real ones.
const WINDOW_LINES: usize = 6;

#[derive(Debug)]
struct Site {
    file: String,
    line: usize,
    receiver: String,
    bound: String,
}

/// The receiver of `.len()` in `line`, if the comparison looks like a capacity
/// gate: `<recv>.len() < BOUND` or `<recv>.len() >= BOUND`.
///
/// Returns the receiver text and the bound token. Both polarities count — the
/// `<` form guards the insert directly and the `>=` form early-returns above it.
fn capacity_gate(line: &str) -> Option<(String, String)> {
    let at = line.find(".len()")?;
    let rest = line[at + ".len()".len()..].trim_start();
    let rest = rest
        .strip_prefix("<=")
        .or_else(|| rest.strip_prefix(">="))
        .or_else(|| rest.strip_prefix('<'))
        .or_else(|| rest.strip_prefix('>'))?
        .trim_start();
    let token: String = rest
        .chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_')
        .collect();
    if !is_bound(&token) {
        return None;
    }
    // Walk left from `.len()` over the receiver expression.
    let head = &line[..at];
    let recv: String = head
        .chars()
        .rev()
        .take_while(|c| c.is_alphanumeric() || *c == '_' || *c == '.')
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    if recv.is_empty() {
        return None;
    }
    Some((recv, token))
}

fn find_sites(sources: &[(String, String)]) -> Vec<Site> {
    let mut sites = Vec::new();
    for (file, text) in sources {
        let lines: Vec<&str> = text.lines().collect();
        for (i, line) in lines.iter().enumerate() {
            let Some((recv, bound)) = capacity_gate(line) else {
                continue;
            };
            // The insertion the gate governs, into that same collection.
            let to = (i + WINDOW_LINES).min(lines.len() - 1);
            let governs = lines[i..=to]
                .iter()
                .any(|l| INSERT.iter().any(|m| l.contains(&format!("{recv}.{m}("))));
            if !governs {
                continue;
            }
            sites.push(Site {
                file: file.clone(),
                line: i + 1,
                receiver: recv,
                bound,
            });
        }
    }
    sites
}

#[test]
fn every_capacity_gated_insert_says_what_a_skip_drops() {
    let sources = guest_facing_sources();

    let sites = find_sites(&sources);

    // Self-check, the rule every source scan in this directory carries: refuse
    // to report a clean tree until the scan has proved it can see the sites it
    // was written for. This one has three ways to go blind — the insert list,
    // the bound vocabulary, and the receiver match — and an empty result is
    // indistinguishable from a healthy crate without it.
    for (file, receiver) in [
        (
            "reims-vgpu/src/runtime/decode/resource/mod.rs",
            "out.layouts",
        ),
        (
            "reims-vgpu/src/runtime/decode/resource/mod.rs",
            "out.attributes",
        ),
    ] {
        assert!(
            sites
                .iter()
                .any(|s| s.file == file && s.receiver == receiver),
            "the scan cannot see {file}'s `{receiver}` gate, so its silence about \
             everything else means nothing.\n\nFound:\n{}",
            summarize(&sites)
        );
    }

    let mut unclassified = Vec::new();
    for site in &sites {
        let known = DROPS
            .iter()
            .any(|(file, recv, _, _, _)| *file == site.file && *recv == site.receiver);
        if !known {
            unclassified.push(format!(
                "{}:{} — `{}` gated on `{}`",
                site.file, site.line, site.receiver, site.bound
            ));
        }
    }
    assert!(
        unclassified.is_empty(),
        "these decline to record something because a collection is already at a \
         bound, and say nothing about what that drops.\n\nFor each, answer in \
         this order. Can the guest even express a request that exceeds the \
         bound? If the bound is the wire field's own width or mask, it is \
         `Unencodable` — name the mask. Otherwise: is the skip counted, and does \
         something downstream refuse the whole request on that count? Then it is \
         `RefusesTheRequest`, which is what a GPU does when it cannot honour a \
         call. Is the collection a cache whose entries can be made again? \
         `Recomputable`. If none of those hold, the guest sent something this \
         device silently did not write down, and `LosesGuestWork` will say so by \
         failing. Add a line to DROPS in {}.\n\n{}",
        file!(),
        unclassified.join("\n")
    );

    let mut wrong = Vec::new();
    for (file, recv, expected, _, _) in DROPS {
        let live = sites
            .iter()
            .filter(|s| s.file == *file && s.receiver == *recv)
            .count();
        if live != *expected {
            wrong.push(format!(
                "{file} `{recv}`: DROPS says {expected} site(s), the scan finds {live}"
            ));
        }
    }
    assert!(
        wrong.is_empty(),
        "DROPS no longer describes this crate. A count that grew is a new gate \
         inheriting a verdict written about a different line; one that shrank is \
         a claim about code that is gone.\n\n{}\n\nCurrent sites:\n{}",
        wrong.join("\n"),
        summarize(&sites)
    );
}

/// No gate may be classified as silently losing guest work.
///
/// Separate from the classification test for the reason its sibling gives: that
/// one fails when nobody has answered, this one when somebody has and the answer
/// is that a number stands between the guest and its own data. The two need
/// different messages because they need different fixes — a line of prose
/// against a redesign.
#[test]
fn no_bounded_insert_is_allowed_to_lose_guest_work() {
    let losing: Vec<&str> = DROPS
        .iter()
        .filter(|(_, _, _, drop, _)| *drop == Drop_::LosesGuestWork)
        .map(|(file, ..)| *file)
        .collect();
    assert!(
        losing.is_empty(),
        "these leave a collection that looks complete while the guest sent more \
         than it holds, and nothing downstream is told. A GPU refuses a call it \
         cannot honour; it does not honour part of it and report success. The \
         fix is to hold the entry, refuse the request, or prove the bound is the \
         wire's own.\n\n{}",
        losing.join("\n")
    );
}

fn summarize(sites: &[Site]) -> String {
    sites
        .iter()
        .map(|s| format!("  {}:{} `{}` bound={}", s.file, s.line, s.receiver, s.bound))
        .collect::<Vec<_>>()
        .join("\n")
}
