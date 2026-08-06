//! A guest object's type tag is compared against the constant that names it.
//!
//! `decode/resource` declares the tag values Apple's object list uses —
//! `OBJECT_TYPE_BUFFER = 1`, `OBJECT_TYPE_TEXTURE = 2`, and so on — and roughly
//! forty sites across `runtime/` ask "is this entry the type I want" before
//! reading its descriptor. Written as `entry.object_type != OBJECT_TYPE_BUFFER`
//! the question answers itself; written as `entry.object_type != 1` it does not,
//! and this crate had one of the latter (`runtime::draw`'s index-buffer load) in a
//! file that already imported the constant and already used it four times.
//!
//! Two reasons that is worth a gate rather than a fix.
//!
//! **The number is not ours.** The tags are decoded from a guest structure, so
//! they are contract fidelity — a value drawn from Apple's list rather than one
//! we chose. `AGENTS.md`'s no-magic-numbers rule is at its sharpest here: a bare
//! `1` records neither where the value came from nor what it means, and a reader
//! checking it against the header has nothing to search for.
//!
//! **A literal cannot be renamed, so it cannot be found.** If a tag's meaning is
//! ever revised, the compiler rewrites every named site and silently leaves
//! every numeric one comparing the old value under the new contract. That is the
//! failure mode this project keeps meeting: two copies of one rule that agree
//! today.
//!
//! # The set is derived
//!
//! Nothing here lists the tags. The scan reads whatever `OBJECT_TYPE_*`
//! constants `decode/resource/mod.rs` declares, so a new tag closes the door
//! behind itself, and a failure can name the constant the literal should have
//! been. A list would be a second copy of the very thing being pinned.
//!
//! An integration test rather than a `#[cfg(test)]` module because it reads
//! source text and must hold on every arm, including `backend-metal`, which this
//! development host compiles but cannot execute.

use std::collections::BTreeMap;

mod source_scan;
use source_scan::{blank_comments, blank_test_items, rust_sources, workspace_root};

fn crate_src() -> std::path::PathBuf {
    workspace_root().join("crates/reims-vgpu/src")
}

/// `OBJECT_TYPE_* = <n>` as declared, keyed by value.
///
/// Two constants sharing a value would make the "did you mean" half ambiguous,
/// so that is reported rather than silently resolved to whichever parsed last.
fn declared_tags(src: &std::path::Path) -> BTreeMap<u64, Vec<String>> {
    let decl = src.join("runtime/decode/resource/mod.rs");
    let text = std::fs::read_to_string(&decl)
        .expect("the module declaring the object tags must be readable");
    let code = blank_comments(&text);

    let mut out: BTreeMap<u64, Vec<String>> = BTreeMap::new();
    for line in code.lines() {
        let line = line.trim();
        let Some(rest) = line.strip_prefix("pub const OBJECT_TYPE_") else {
            continue;
        };
        let Some((name_tail, value)) = rest.split_once('=') else {
            continue;
        };
        let name = format!(
            "OBJECT_TYPE_{}",
            name_tail
                .split(|c: char| c == ':' || c.is_whitespace())
                .next()
                .unwrap_or("")
        );
        let digits: String = value
            .trim()
            .trim_end_matches(';')
            .trim()
            .chars()
            .take_while(|c| c.is_ascii_digit())
            .collect();
        if let Ok(v) = digits.parse::<u64>() {
            out.entry(v).or_default().push(name);
        }
    }
    out
}

/// The decimal literal on the right of an `==`/`!=` whose left side mentions
/// `object_type`, if there is one.
///
/// Deliberately narrow. It answers only the question this file is about, so a
/// `match object_type { 1 => .. }`, a range, or a value passed to a formatter
/// is not reported — none of those is a comparison spelled with a bare number,
/// and widening it would turn the gate into a source of noise nobody reads.
fn numeric_tag_comparison(code_line: &str) -> Option<u64> {
    for op in ["==", "!="] {
        for (at, _) in code_line.match_indices(op) {
            let (lhs, rhs) = code_line.split_at(at);
            if !lhs.contains("object_type") {
                continue;
            }
            let rhs = rhs[op.len()..].trim_start();
            let digits: String = rhs.chars().take_while(|c| c.is_ascii_digit()).collect();
            if digits.is_empty() {
                continue;
            }
            // `!= 1u8` and `!= 1 {` are the tag; `!= 1.0` and `!= 12abc` are
            // not this shape at all, so only a clean boundary counts.
            let after = rhs[digits.len()..].chars().next().unwrap_or(' ');
            if after == '.' || after.is_alphanumeric() && after != 'u' {
                continue;
            }
            if let Ok(v) = digits.parse::<u64>() {
                return Some(v);
            }
        }
    }
    None
}

#[test]
fn the_scanner_can_see_a_numeric_tag_comparison() {
    // The gate below reports "none", and a scanner that reports none because it
    // matches nothing looks exactly like a clean tree. These are the shapes it
    // must catch and the shapes it must not, stated before its silence is
    // believed.
    assert_eq!(
        numeric_tag_comparison("    if entry.object_type != 1 {"),
        Some(1),
        "the exact line this file was written out of"
    );
    assert_eq!(
        numeric_tag_comparison("if e.object_type == 11 && x {"),
        Some(11),
        "a two-digit tag in a compound condition"
    );
    assert_eq!(
        numeric_tag_comparison("if entry.object_type != 8u8 {"),
        Some(8),
        "a suffixed literal is still a literal"
    );
    assert_eq!(
        numeric_tag_comparison("if entry.object_type != OBJECT_TYPE_BUFFER {"),
        None,
        "the named form is the whole point and must not be reported"
    );
    assert_eq!(
        numeric_tag_comparison("if other_field == 1 {"),
        None,
        "a comparison that is not about a type tag"
    );
    assert_eq!(
        numeric_tag_comparison("            .field(\"obj_type\", object_type)"),
        None,
        "reporting a tag is not comparing one"
    );
}

#[test]
fn every_object_tag_is_compared_by_name() {
    let src = crate_src();
    let tags = declared_tags(&src);

    // An empty table means the declaring module moved or was renamed, and the
    // scan below would then pass by measuring nothing.
    assert!(
        tags.len() >= 5,
        "found {} OBJECT_TYPE_* declarations in runtime/decode/resource/mod.rs, \
         which is not that module — this gate is checking nothing",
        tags.len()
    );
    for (value, names) in &tags {
        assert!(
            names.len() == 1,
            "tag {value} is declared under {names:?}; two names for one value \
             makes 'did you mean' ambiguous — say which is canonical here"
        );
    }

    let files = rust_sources(&src);
    assert!(
        files.len() > 50,
        "walked {} files, which is not this crate",
        files.len()
    );

    let mut failures = Vec::new();
    for path in files {
        let rel = path
            .strip_prefix(&src)
            .expect("every walked file is under src")
            .to_string_lossy()
            .into_owned();
        if rel.ends_with("tests.rs") {
            continue;
        }
        let text = std::fs::read_to_string(&path).expect("crate source must be readable");
        let code = blank_test_items(&blank_comments(&text));
        for (n, line) in code.lines().enumerate() {
            let Some(value) = numeric_tag_comparison(line) else {
                continue;
            };
            let named = match tags.get(&value) {
                Some(names) => format!("`{}`", names[0]),
                None => format!(
                    "no declared tag — {value} is not a value \
                     runtime/decode/resource declares"
                ),
            };
            failures.push(format!(
                "  {rel}:{}  {}\n      (compare against {named})",
                n + 1,
                line.trim()
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "a guest object tag is compared by name, and these use the number:\n{}",
        failures.join("\n")
    );
}
