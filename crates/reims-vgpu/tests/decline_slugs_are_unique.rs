//! No two checks in this crate share a `reason=` slug.
//!
//! This is the one property of the decline vocabulary that is genuinely
//! crate-wide and invisible from any single `impl`, and it is load-bearing
//! rather than tidy. `Emit::fail_once` latches on `(slug, discriminant)` in one
//! process-global set, so two declines spelling the same slug share a latch:
//! whichever fires first silences the other for the life of the boot, and the
//! log still looks healthy. That is not hypothetical — `mapping_gpa_span` had
//! exactly this shape between its two emitters, and the silence it produced was
//! written up as a finding about the device before the collision was found.
//!
//! A `gate` module used to check this by scanning source text; it was removed
//! wholesale in `db80389` along with the 2 700-line `REGISTRY` it compared
//! against, because the registry could only ever agree or disagree with the
//! arms it copied. This test keeps the one check that survived that argument
//! and none of the restatement: it reads the `slug()` and `refusal()` bodies
//! themselves, so it cannot drift from them.
//!
//! # Why not a grep
//!
//! The obvious `grep -rn 'impl Decline for'` misses two impls in this tree —
//! `GatherWitnessFault` and `SurfaceWriteRefusal` both spell the trait as
//! `crate::observe::decline::Decline` — and picks up one that does not exist,
//! the `impl<T: Decline> Display for T` written inside a doc comment in
//! `observe/decline.rs`. Both classes are handled below, and the scanner
//! asserts it saw the real two before believing anything it reports.
//!
//! Reading source rather than compiled code is also what makes this answer for
//! the whole crate on either arm: a `#[cfg(feature = "backend-metal")]` decline
//! is text on a Vulkan host, so its slugs are compared here even though nothing
//! on this pathway can construct one.

use std::collections::BTreeMap;
use std::path::Path;

mod source_scan;
use source_scan::{blank_comments, close_brace, rust_sources, strip_attributes, workspace_root};

/// One `impl Decline`/`impl Refusal` block, identified by where it is written.
///
/// The type name alone is not an identity here: five distinct `DecodeStatus`
/// types live in five modules, and `Status` names two more. The file is what
/// separates them.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Owner {
    file: String,
    ty: String,
}

/// Rewrite every `ladder_slug!("role", rung)` into the slug it expands to.
///
/// This scan reads source text, so a macro-composed slug is invisible to it —
/// `ladder_slug!("draw_index", desc_read)` offers one string literal, `"draw_index"`,
/// and four arms using four rungs all looked like four arms returning the same
/// slug. That reported a collision where there was none, and the dangerous half
/// is the other direction: had the role happened to be unique per arm, the scan
/// would have passed while comparing roles instead of slugs, and a genuine
/// collision between two composed slugs would have been invisible.
///
/// Expanding here rather than special-casing the shape in [`literals`] keeps
/// this file honest about what it is comparing: after this pass, every literal
/// the scan sees is a slug that reaches the log.
fn expand_ladder_slugs(body: &str) -> String {
    const MACRO: &str = "ladder_slug!(";
    let mut out = String::with_capacity(body.len());
    let mut rest = body;
    while let Some(at) = rest.find(MACRO) {
        let (before, tail) = rest.split_at(at);
        out.push_str(before);
        let args = &tail[MACRO.len()..];
        let Some(close) = args.find(')') else {
            // Not a call this scan understands; leave it as text so the slug it
            // would have produced is missing rather than silently wrong.
            out.push_str(tail);
            return out;
        };
        let (inner, after) = args.split_at(close);
        let mut parts = inner.splitn(2, ',');
        let role = parts.next().unwrap_or("").trim().trim_matches('"');
        let rung = parts.next().unwrap_or("").trim();
        if role.is_empty() {
            out.push_str(&format!("\"{rung}\""));
        } else {
            out.push_str(&format!("\"{role}_{rung}\""));
        }
        rest = &after[1..];
    }
    out.push_str(rest);
    out
}

/// Every string literal in `body`.
fn literals(body: &str) -> Vec<String> {
    let chars: Vec<char> = body.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '"' {
            let mut lit = String::new();
            i += 1;
            while i < chars.len() && chars[i] != '"' {
                if chars[i] == '\\' {
                    i += 1;
                }
                if i < chars.len() {
                    lit.push(chars[i]);
                }
                i += 1;
            }
            out.push(lit);
        }
        i += 1;
    }
    out
}

/// The identifier immediately before `end` in `chars`, if any.
fn ident_ending_at(chars: &[char], end: usize) -> String {
    let mut start = end;
    while start > 0 && (chars[start - 1].is_alphanumeric() || chars[start - 1] == '_') {
        start -= 1;
    }
    chars[start..end].iter().collect()
}

/// Every slug literal in the crate, mapped to the impl blocks that spell it.
fn slug_owners(root: &Path) -> BTreeMap<String, Vec<Owner>> {
    let sources = rust_sources(&root.join("crates/reims-vgpu/src"));
    assert!(
        sources.len() > 50,
        "walked {} files, which is not this crate",
        sources.len()
    );

    let mut out: BTreeMap<String, Vec<Owner>> = BTreeMap::new();
    for path in sources {
        let raw = std::fs::read_to_string(&path).expect("crate source must be readable");
        let text = blank_comments(&raw);
        let chars: Vec<char> = text.chars().collect();
        let rel_path = path
            .strip_prefix(root)
            .unwrap_or(&path)
            .to_string_lossy()
            .into_owned();

        let mut at = 0usize;
        while let Some(found) = text[at..].find("impl") {
            let start = at + found;
            at = start + "impl".len();
            // A bare `impl` token, not the tail of an identifier.
            let before_ok = start == 0 || !(text.as_bytes()[start - 1] as char).is_alphanumeric();
            if !before_ok {
                continue;
            }
            let Some(open_rel) = text[at..].find('{') else {
                break;
            };
            let open = at + open_rel;
            let header = &text[at..open];
            // `impl<…> Trait for Type`: the trait is the identifier just before
            // the last ` for `, so a fully-qualified path matches and a generic
            // bound (`impl<T: Decline> Display for T`) does not.
            let Some(for_rel) = header.rfind(" for ") else {
                continue;
            };
            let head_chars: Vec<char> = header.chars().collect();
            let for_idx = header[..for_rel].chars().count();
            let trait_name = ident_ending_at(&head_chars, for_idx);
            if trait_name != "Decline" && trait_name != "Refusal" {
                continue;
            }
            let ty = header[for_rel + " for ".len()..].trim().to_string();

            let open_chars = text[..open].chars().count();
            let end = close_brace(&chars, open_chars);
            let body: String = chars[open_chars..end].iter().collect();

            for name in ["fn slug", "fn refusal"] {
                let mut fat = 0usize;
                while let Some(rel) = body[fat..].find(name) {
                    let fn_start = fat + rel;
                    fat = fn_start + name.len();
                    let Some(brace_rel) = body[fat..].find('{') else {
                        break;
                    };
                    let brace = fat + brace_rel;
                    let body_chars: Vec<char> = body.chars().collect();
                    let brace_ci = body[..brace].chars().count();
                    let fn_end = close_brace(&body_chars, brace_ci);
                    let fn_body =
                        expand_ladder_slugs(&strip_attributes(&body_chars[brace_ci..fn_end]));
                    for slug in literals(&fn_body) {
                        out.entry(slug).or_default().push(Owner {
                            file: rel_path.clone(),
                            ty: ty.clone(),
                        });
                    }
                    fat = fn_end;
                }
            }
            at = end;
        }
    }
    out
}

#[test]
fn no_two_checks_share_a_reason_slug() {
    let root = workspace_root();
    let owners = slug_owners(&root);

    // Refuse a verdict until the scan has proved it can see the two impls that
    // spell the trait through its full path, which is the case the grep this
    // replaces misses. An empty or half-blind scan reports zero collisions and
    // reads exactly like a clean tree.
    for (slug, why) in [
        (
            "gather_witness_vouched_bytes_moved",
            "runtime/gather_witness.rs, `impl crate::observe::decline::Decline`",
        ),
        (
            "surface_write_mapping_absent",
            "runtime/mapping_write.rs, `impl crate::observe::decline::Decline`",
        ),
    ] {
        assert!(
            owners.contains_key(slug),
            "the scan did not find `{slug}` ({why}), so its notion of \
             `no collisions` is a blind spot and not a measurement"
        );
    }
    assert!(
        owners.len() > 500,
        "found {} slugs, which is not this crate's vocabulary",
        owners.len()
    );

    let mut collisions: Vec<String> = Vec::new();
    for (slug, sites) in &owners {
        if sites.len() == 1 {
            continue;
        }
        let mut distinct = sites.clone();
        distinct.sort();
        distinct.dedup();
        if distinct.len() > 1 {
            let named: Vec<String> = distinct
                .iter()
                .map(|o| format!("{} ({})", o.ty, o.file))
                .collect();
            collisions.push(format!(
                "`{slug}` is claimed by {} impls: {}",
                distinct.len(),
                named.join(", ")
            ));
        } else {
            // One impl spelling the same slug from two arms: the log cannot
            // tell the two checks apart, which is the same defect at a smaller
            // radius. `#[cfg]` twins of one type are not this — they are one
            // check, and they land here only if the arms differ.
            collisions.push(format!(
                "`{slug}` is returned by {} arms of {} ({})",
                sites.len(),
                distinct[0].ty,
                distinct[0].file
            ));
        }
    }

    assert!(
        collisions.is_empty(),
        "declines sharing a slug share `Emit::fail_once`'s latch, so the first \
         to fire silences the others for the boot. Give each check its own \
         slug:\n  {}",
        collisions.join("\n  ")
    );
}
