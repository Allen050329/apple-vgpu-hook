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
use std::path::{Path, PathBuf};

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("the crate lives two levels below the workspace root")
        .to_path_buf()
}

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

/// Blank out comments so neither the impl scan nor the literal scan reads one.
///
/// String literals are preserved verbatim, including raw strings, because the
/// slugs *are* string literals. Everything a comment contained becomes spaces,
/// which keeps every byte offset stable.
fn blank_comments(text: &str) -> String {
    let bytes: Vec<char> = text.chars().collect();
    let mut out: Vec<char> = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        // Raw string: `r"…"` or `r#"…"#` with any number of hashes.
        if c == 'r' && i + 1 < bytes.len() && (bytes[i + 1] == '"' || bytes[i + 1] == '#') {
            let mut hashes = 0;
            let mut j = i + 1;
            while j < bytes.len() && bytes[j] == '#' {
                hashes += 1;
                j += 1;
            }
            if j < bytes.len() && bytes[j] == '"' {
                out.push('r');
                out.extend(std::iter::repeat_n('#', hashes));
                out.push('"');
                j += 1;
                // Closing is `"` followed by exactly `hashes` hashes.
                while j < bytes.len() {
                    if bytes[j] == '"' && bytes[j + 1..].iter().take(hashes).all(|c| *c == '#') {
                        out.push('"');
                        out.extend(std::iter::repeat_n('#', hashes));
                        j += 1 + hashes;
                        break;
                    }
                    out.push(bytes[j]);
                    j += 1;
                }
                i = j;
                continue;
            }
        }
        match c {
            '"' => {
                out.push('"');
                i += 1;
                while i < bytes.len() {
                    if bytes[i] == '\\' && i + 1 < bytes.len() {
                        out.push(bytes[i]);
                        out.push(bytes[i + 1]);
                        i += 2;
                        continue;
                    }
                    out.push(bytes[i]);
                    i += 1;
                    if bytes[i - 1] == '"' {
                        break;
                    }
                }
            }
            // Char literal, so an apostrophe in `'"'` cannot open a string.
            // Lifetimes (`'a`) fall through harmlessly: they contain no quote.
            '\'' if i + 2 < bytes.len() && bytes[i + 1] == '"' && bytes[i + 2] == '\'' => {
                out.extend_from_slice(&bytes[i..i + 3]);
                i += 3;
            }
            '/' if bytes.get(i + 1) == Some(&'/') => {
                while i < bytes.len() && bytes[i] != '\n' {
                    out.push(' ');
                    i += 1;
                }
            }
            '/' if bytes.get(i + 1) == Some(&'*') => {
                let mut depth = 0usize;
                while i < bytes.len() {
                    if bytes[i] == '/' && bytes.get(i + 1) == Some(&'*') {
                        depth += 1;
                        out.push(' ');
                        out.push(' ');
                        i += 2;
                    } else if bytes[i] == '*' && bytes.get(i + 1) == Some(&'/') {
                        depth -= 1;
                        out.push(' ');
                        out.push(' ');
                        i += 2;
                        if depth == 0 {
                            break;
                        }
                    } else {
                        out.push(if bytes[i] == '\n' { '\n' } else { ' ' });
                        i += 1;
                    }
                }
            }
            _ => {
                out.push(c);
                i += 1;
            }
        }
    }
    out.into_iter().collect()
}

/// Index of the `}` closing the `{` at `open`.
fn close_brace(chars: &[char], open: usize) -> usize {
    let mut depth = 0usize;
    let mut i = open;
    while i < chars.len() {
        match chars[i] {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return i;
                }
            }
            _ => {}
        }
        i += 1;
    }
    chars.len()
}

/// Remove `#[…]` attribute spans, whose own string literals (`target_os =
/// "macos"`) are not slugs. Without this the scan reports `macos` as a
/// three-way collision between `ComputeStatus`, `EncodeStatus` and
/// `MipmapStatus`.
fn strip_attributes(chars: &[char]) -> String {
    let mut out = String::new();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '#' && chars.get(i + 1) == Some(&'[') {
            let mut depth = 0usize;
            i += 1;
            while i < chars.len() {
                match chars[i] {
                    '[' => depth += 1,
                    ']' => {
                        depth -= 1;
                        if depth == 0 {
                            i += 1;
                            break;
                        }
                    }
                    _ => {}
                }
                i += 1;
            }
            continue;
        }
        out.push(chars[i]);
        i += 1;
    }
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
    let mut sources = Vec::new();
    collect_rs(&root.join("crates/reims-vgpu/src"), &mut sources);
    sources.sort();
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
                    let fn_body = strip_attributes(&body_chars[brace_ci..fn_end]);
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

fn collect_rs(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).expect("crate src must be readable") {
        let path = entry.expect("a readable dir entry").path();
        if path.is_dir() {
            collect_rs(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
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
