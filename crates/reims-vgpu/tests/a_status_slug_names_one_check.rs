//! No two Metal-arm checks construct a `Status` with the same slug.
//!
//! `backend::metal::error::Status` is the Metal encode path's refusal
//! vocabulary, and it is a *second* vocabulary: it does not implement `Decline`
//! or `Refusal`, so [`decline_slugs_are_unique`] — which reads `slug()` and
//! `refusal()` bodies — cannot see one of its 160 slugs. The collision hazard is
//! the same one, and load-bearing for the same reason. A `Status` reaches the
//! log through `observe`'s emitters, whose `fail_once` latches on the slug, so
//! two checks spelling one slug share a latch: whichever fires first silences
//! the other for the life of the boot and the log still reads healthy.
//!
//! It is also the vocabulary this host cannot execute. Every constructor below
//! is inside `#[cfg(feature = "backend-metal")]` code, so no Linux test run
//! constructs one — but the slugs are text, and text is readable on every arm.
//! That is the whole reason this is a source scan.
//!
//! # What a duplicate meant, the three times this found one
//!
//! Two checks in `render.rs` both said `metal_render_vertex_step_function_
//! unsupported`, one reporting `location` and one `buffer`, for an
//! unconvertible ordinal and for a tessellation step function this pipeline has
//! no stage for. Two in `stage_input.rs` both said `metal_stage_input_step_
//! function_unsupported`, where the second firing means the guard and the
//! conversion table *disagree* — a different event entirely. The third was a
//! test constructing a product slug to assert on it, which is why test code is
//! blanked before anything is counted.
//!
//! [`decline_slugs_are_unique`]: ../decline_slugs_are_unique/index.html

use std::collections::BTreeMap;

mod source_scan;
use source_scan::{blank_comments, blank_test_items, rust_sources, workspace_root};

/// The `Status` constructors that take a slug.
///
/// `Status::OK` carries none — it is the absence of a refusal — and `field` and
/// `code` are not constructors. A constructor added without being named here is
/// invisible to this test, which is why the floor below asserts the scan found
/// the population it expects rather than trusting an empty result.
const CONSTRUCTORS: &[&str] = &["Status::args(", "Status::execute("];

/// Every `(slug, file, line)` a slug-taking constructor is written at.
fn slug_sites() -> Vec<(String, String, usize)> {
    let root = workspace_root();
    let src = root.join("crates/reims-vgpu/src");
    let files = rust_sources(&src);
    assert!(
        files.len() > 50,
        "walked {} files, which is not this crate",
        files.len()
    );

    let mut out = Vec::new();
    for path in files {
        let rel = path
            .strip_prefix(&root)
            .expect("every walked file is under the workspace")
            .to_string_lossy()
            .into_owned();
        // A whole `tests.rs` sibling is test code even though no `#[cfg(test)]`
        // marker appears in it; an inline `#[cfg(test)] mod` is blanked by the
        // shared helper. Both spell the same thing and both construct product
        // slugs to assert on them.
        if path.file_name().is_some_and(|n| n == "tests.rs") {
            continue;
        }
        let text = std::fs::read_to_string(&path).expect("crate source must be readable");
        let code = blank_test_items(&blank_comments(&text));
        for (n, line) in code.lines().enumerate() {
            for ctor in CONSTRUCTORS {
                let mut from = 0;
                while let Some(at) = line[from..].find(ctor) {
                    let open = from + at + ctor.len();
                    from = open;
                    let Some(slug) = literal_after(&line[open..]) else {
                        continue;
                    };
                    out.push((slug, rel.clone(), n + 1));
                }
            }
        }
    }
    out
}

/// The string literal a constructor opens with, if it opens with one.
///
/// A constructor whose slug is a variable or a macro yields `None` rather than a
/// guess. None exist today — the floor test below pins the count, so one
/// appearing changes a number somebody has to look at.
fn literal_after(rest: &str) -> Option<String> {
    let mut chars = rest.chars();
    if chars.next()? != '"' {
        return None;
    }
    let mut slug = String::new();
    for c in chars {
        if c == '"' {
            return Some(slug);
        }
        slug.push(c);
    }
    None
}

#[test]
fn no_two_checks_construct_the_same_status_slug() {
    let mut by_slug: BTreeMap<String, Vec<(String, usize)>> = BTreeMap::new();
    for (slug, file, line) in slug_sites() {
        by_slug.entry(slug).or_default().push((file, line));
    }

    let mut collisions = Vec::new();
    for (slug, sites) in &by_slug {
        if sites.len() < 2 {
            continue;
        }
        let where_ = sites
            .iter()
            .map(|(f, l)| format!("{f}:{l}"))
            .collect::<Vec<_>>()
            .join(", ");
        collisions.push(format!("  {slug}  at {where_}"));
    }

    assert!(
        collisions.is_empty(),
        "a slug is one check, and these are written at two sites — they share \
         `fail_once`'s latch, so whichever fires first silences the other for \
         the life of the boot:\n{}\nGive the second question its own slug; do \
         not widen this test.",
        collisions.join("\n")
    );
}

/// The scan reaches the vocabulary it claims to check.
///
/// A scanner that matched nothing would pass the test above forever, and every
/// way of breaking it is quiet: a constructor renamed, `blank_test_items`
/// blanking more than it should, the walk rooted at the wrong directory. The
/// floor is deliberately loose about the exact number and strict about the
/// order of magnitude, and it names both constructors so a class that stops
/// being seen fails here rather than reading as a clean tree.
#[test]
fn the_scan_sees_the_metal_refusal_vocabulary() {
    let sites = slug_sites();
    assert!(
        sites.len() > 100,
        "found only {} Status slug sites; the scan is not reading the crate",
        sites.len()
    );
    let distinct: std::collections::BTreeSet<&str> =
        sites.iter().map(|(s, _, _)| s.as_str()).collect();
    assert_eq!(
        distinct.len(),
        sites.len(),
        "every site is its own slug, so the two counts agree"
    );
    for known in [
        "metal_render_vertex_step_rate_zero",
        "metal_render_command_buffer_failed",
    ] {
        assert!(
            distinct.contains(known),
            "{known} is constructed in this crate and the scan did not see it"
        );
    }
}
