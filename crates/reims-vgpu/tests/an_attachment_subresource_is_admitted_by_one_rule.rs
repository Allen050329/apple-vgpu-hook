//! An attachment's subresource is admitted by the whole rule or by none of it.
//!
//! A render pass's colour, depth and stencil slots share one 28-byte wire prefix
//! (`reims_vgpu_wire::ops::render_pass::AttachmentPrefix`), and this device can
//! bind exactly one shape out of it: the whole texture at level 0, slice 0,
//! plane 0, with no multisample resolve target. Four terms.
//!
//! # Every hand-written copy of those four terms has been missing one
//!
//! That is the measurement this gate exists for, not a worry about one.
//!
//! - The Metal rail's copy tested `level` and `resolve_texture_ref` and not
//!   `slice` or `depth_plane`, so a depth buffer bound at slice 5 was read as
//!   slice 0 and silently accepted.
//! - The colour arm's copy tested `level`, `slice` and `depth_plane` and not
//!   `resolve_texture_ref`. A colour attachment naming a resolve target *is* a
//!   multisample colour pass — the attachment is multisampled, the store action
//!   is `MultisampleResolve`, and the resolve texture is where the
//!   single-sampled result goes. The device admitted it, rendered at one sample
//!   into the attachment, and never wrote the resolve target. The guest reads
//!   the resolve target. Wrong pixels, no counter, no fail line.
//!
//! Both were found by reading two arms against each other, which is a thing that
//! happens when someone thinks to do it. `runtime::decode::render::AttachSubresource`
//! and `attachment_subresource_is_bindable` are the structural repair — a caller
//! converts the whole prefix and asks one question — but nothing *forces* a
//! fifth arm through them. It can still write the terms out, and if it writes
//! three of the four it is today's bug again with a different field missing.
//!
//! # What is refused
//!
//! A site that compares **more than one** of the four coordinate names against
//! zero, and does not compare all four. One is fine: a texture view's own
//! `level` has nothing to do with an attachment, and a lone comparison cannot be
//! a partial copy of a four-term rule. Two or three is the shape that has been
//! wrong twice.
//!
//! An integration test rather than a `#[cfg(test)]` module because it reads
//! source text and must hold on every arm, including `backend-metal`, which this
//! development host compiles but cannot execute.

mod source_scan;
use source_scan::guest_facing_sources;

/// The four terms, as they are spelled on every attachment type this crate
/// decodes and on [`AttachSubresource`] itself.
///
/// Not derived from the struct, deliberately: the point is to catch a site that
/// writes the terms out *instead of* naming the type, so the names are what the
/// scan has to know. A fifth coordinate added to the prefix belongs here, and
/// the wire crate's own layout tests are what would say it exists.
const TERMS: [&str; 4] = ["level", "slice", "depth_plane", "resolve_texture_ref"];

/// A site that spells some of the rule, and the verdict on why that is not a
/// partial copy of it.
///
/// `(file relative to `crates/`, the substring identifying the line, why)`.
/// Empty is the healthy state: the rule has one home and every arm reaches it.
const SPELLED_ELSEWHERE: &[(&str, &str, &str)] = &[];

/// Lines that compare at least `min` of [`TERMS`] against zero.
fn lines_testing_terms(text: &str, min: usize) -> Vec<(usize, String)> {
    text.lines()
        .enumerate()
        .filter_map(|(i, line)| {
            let hit = TERMS
                .iter()
                .filter(|t| {
                    // `<something>.term` compared against a literal zero, in
                    // either polarity. The field access is what separates a
                    // coordinate from a local of the same name.
                    let pat = format!(".{t}");
                    let Some(at) = line.find(&pat) else {
                        return false;
                    };
                    let rest = &line[at + pat.len()..];
                    let rest = rest.trim_start();
                    (rest.starts_with("== 0") || rest.starts_with("!= 0"))
                        && !rest.starts_with("== 0x")
                        && !rest.starts_with("!= 0x")
                })
                .count();
            (hit >= min).then(|| (i + 1, line.trim().to_string()))
        })
        .collect()
}

/// The scan can see the rule it is about.
///
/// The self-check every source scan in this directory carries, and it is not a
/// formality here: a scan that matched nothing would report an empty population
/// as a clean tree, which is the exact failure mode the gate exists to prevent.
/// The predicate's own body tests all four terms on one line, so it is the one
/// site that must always be found.
#[test]
fn the_scan_finds_the_rule_it_is_about() {
    let sources = guest_facing_sources();
    let predicate = sources
        .iter()
        .find(|(path, _)| path == "reims-vgpu/src/runtime/decode/render/mod.rs")
        .map(|(_, text)| text)
        .expect("the module that declares the rule is a product source");
    let all_four = lines_testing_terms(predicate, TERMS.len());
    assert_eq!(
        all_four.len(),
        1,
        "the scan should find exactly the predicate's own body testing all four \
         terms on one line; it found {}: {all_four:?}",
        all_four.len()
    );
    assert!(
        all_four[0].1.contains("resolve_texture_ref == 0"),
        "the line found is not the predicate's body: {}",
        all_four[0].1
    );
}

/// No site spells part of the rule.
#[test]
fn no_site_tests_some_of_the_attachment_coordinates_and_not_the_rest() {
    let mut partial: Vec<String> = Vec::new();
    for (path, text) in guest_facing_sources() {
        for (line_no, line) in lines_testing_terms(&text, 2) {
            // All four is the rule itself, wherever it is written.
            if lines_testing_terms(&line, TERMS.len()).len() == 1 {
                continue;
            }
            if SPELLED_ELSEWHERE
                .iter()
                .any(|(f, needle, _)| *f == path && line.contains(needle))
            {
                continue;
            }
            partial.push(format!("{path}:{line_no}  {line}"));
        }
    }
    assert!(
        partial.is_empty(),
        "a site tests some of an attachment's subresource coordinates and not \
         the rest, which is how this device shipped a depth buffer bound at the \
         wrong slice and a multisample colour pass whose resolve target was \
         never written. Convert the attachment into \
         `runtime::decode::render::AttachSubresource` and ask \
         `attachment_subresource_is_bindable`, or — if these coordinates are not \
         an attachment's — add a row to SPELLED_ELSEWHERE saying whose they \
         are:\n{}",
        partial.join("\n")
    );
}
