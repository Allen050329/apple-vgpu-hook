//! A hand-written `const ALL: &[Enum]` lists every variant of that enum.
//!
//! Several tests in this crate check a property of *every* member of an enum —
//! that each refusal names its own slug, that no two share one, that each is
//! `vk_`-prefixed and log-safe — by iterating a hand-written list. The list is
//! the test's input, so a variant added to the enum and not to the list is
//! silently uncovered: the test keeps passing, over a smaller set, and reports
//! green about the variant it never saw.
//!
//! That is the shape the refactor plan calls meta-debt, and it is not
//! hypothetical. When this test was written it found **six** variants missing
//! from three lists:
//!
//! - `VkOp`'s four `Window*Staging*` operations, so four of 117 Vulkan-call
//!   slugs had never been checked for their rail prefix, their character set, or
//!   uniqueness against the other 113;
//! - `DrawReason::DualSourceBlendUnsupported`;
//! - `IndexLoadReason::BaseVertexOutOfRange`.
//!
//! All six passed once added, so these were gaps in coverage rather than latent
//! slug defects — which is exactly why nothing had noticed them.
//!
//! # Why a source scan rather than a macro
//!
//! The obvious cure is a declarative macro that emits the enum and its `ALL`
//! from one source, and it was considered and rejected: it restructures every
//! enum declaration it touches, and it only helps the enums that adopt it. An
//! exhaustive `match` beside the list is not a cure at all — satisfying the
//! compiler does not force the new variant into `ALL`. This reads the two
//! declarations and compares them, so it cannot drift from either, it covers
//! every list in the tree at once, and it costs the enums nothing.
//!
//! # What "covered" means, and why the scope is the enclosing item
//!
//! A variant counts as covered when the test that owns the list mentions it
//! anywhere — not only inside the list literal. Some variants cannot go in the
//! list and are checked beside it instead: `MipmapStatus::Metal` carries a Metal
//! error type and is constructed in a `#[cfg(all(feature = "backend-metal", …))]`
//! block below its list, and `DrawReason::VertexFormat` is built from a
//! translation reason in its own statement. Scoping to the enclosing `fn` or
//! `mod` counts both, and scoping to the literal alone reported both as gaps.
//! Getting this wrong in the loose direction is worse: the whole *file* would
//! count the `slug()` match arms, which name every variant, and the test would
//! pass over any list at all.

mod source_scan;

use source_scan::{blank_comments, close_brace, rust_sources, workspace_root};
use std::collections::BTreeMap;
use std::path::Path;

/// One `const ALL…: &[Ty] = &[…]` found in the tree.
#[derive(Debug)]
struct AllList {
    file: String,
    konst: String,
    ty: String,
    /// Variant names mentioned anywhere in the item that owns the list.
    mentioned: Vec<String>,
}

/// One `enum Ty { … }` declaration and the variants it names.
#[derive(Debug)]
struct EnumDecl {
    file: String,
    variants: Vec<String>,
}

fn is_ident(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// The identifier starting at `at`, if one does.
fn ident_at(chars: &[char], at: usize) -> Option<String> {
    if !chars
        .get(at)
        .copied()
        .is_some_and(|c| c.is_alphabetic() || c == '_')
    {
        return None;
    }
    let mut end = at;
    while end < chars.len() && is_ident(chars[end]) {
        end += 1;
    }
    Some(chars[at..end].iter().collect())
}

/// Whether `chars[at..]` starts with `word` as a whole token.
fn word_at(chars: &[char], at: usize, word: &str) -> bool {
    let w: Vec<char> = word.chars().collect();
    if at + w.len() > chars.len() || chars[at..at + w.len()] != w[..] {
        return false;
    }
    if at > 0 && is_ident(chars[at - 1]) {
        return false;
    }
    !chars.get(at + w.len()).copied().is_some_and(is_ident)
}

/// The next non-whitespace index at or after `at`.
fn skip_ws(chars: &[char], mut at: usize) -> usize {
    while at < chars.len() && chars[at].is_whitespace() {
        at += 1;
    }
    at
}

/// Top-level variant names inside an `enum` body, ignoring payloads.
///
/// Splits on commas at brace/paren depth zero so a struct-shaped variant's own
/// fields are not read as variants, and drops `#[…]` so an attribute's
/// identifiers are not either.
fn enum_variants(chars: &[char], open: usize) -> Vec<String> {
    let end = close_brace(chars, open);
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut want_name = true;
    let mut i = open + 1;
    while i < end {
        match chars[i] {
            '{' | '(' | '[' => depth += 1,
            '}' | ')' | ']' => depth -= 1,
            ',' if depth == 0 => want_name = true,
            '#' if depth == 0 => {
                // Skip the whole `#[…]`, whose brackets we must not count.
                let mut d = 0i32;
                while i < end {
                    match chars[i] {
                        '[' => d += 1,
                        ']' => {
                            d -= 1;
                            if d == 0 {
                                break;
                            }
                        }
                        _ => {}
                    }
                    i += 1;
                }
            }
            c if depth == 0 && want_name && !c.is_whitespace() => {
                if c.is_uppercase() {
                    if let Some(name) = ident_at(chars, i) {
                        i += name.chars().count();
                        out.push(name);
                        want_name = false;
                        continue;
                    }
                }
                want_name = false;
            }
            _ => {}
        }
        i += 1;
    }
    out
}

/// The `{ … }` span of the innermost `fn` or `mod` item containing `pos`.
///
/// Both, because a list can sit directly in a `#[cfg(test)] mod tests` body
/// (`VkOp`, `DrawReason`, `TranslateReason`) or inside the `#[test]` fn that
/// consumes it (`MipmapStatus`, `IndexLoadReason`). Innermost wins, so a list in
/// a fn is not scored against its whole module.
fn enclosing_item(chars: &[char], pos: usize) -> Option<(usize, usize)> {
    let mut best: Option<(usize, usize)> = None;
    let mut i = 0;
    while i < pos {
        if word_at(chars, i, "fn") || word_at(chars, i, "mod") {
            let mut j = i;
            while j < chars.len() && chars[j] != '{' && chars[j] != ';' {
                j += 1;
            }
            if chars.get(j) == Some(&'{') && j < pos {
                let end = close_brace(chars, j);
                if end > pos && best.is_none_or(|(o, _)| j > o) {
                    best = Some((j, end));
                }
            }
        }
        i += 1;
    }
    best
}

/// Every `Ty::Variant` / `Self::Variant` name in `chars[lo..hi]`.
fn mentioned_variants(chars: &[char], lo: usize, hi: usize, ty: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut i = lo;
    while i + 2 < hi {
        if word_at(chars, i, ty) || word_at(chars, i, "Self") {
            if let Some(name) = ident_at(chars, i) {
                let after = i + name.chars().count();
                if chars.get(after) == Some(&':') && chars.get(after + 1) == Some(&':') {
                    if let Some(v) = ident_at(chars, after + 2) {
                        i = after + 2 + v.chars().count();
                        out.push(v);
                        continue;
                    }
                }
            }
        }
        i += 1;
    }
    out
}

/// `const ALL…: &[Ty] = &[` or `const ALL…: [Ty; N] = [`, returning `(konst, ty)`.
///
/// Only `ALL`-prefixed constants: this is about lists that claim to be the whole
/// set. A `const SOME_FORMATS` making no such claim is not this test's business.
fn parse_all_header(chars: &[char], at: usize) -> Option<(String, String, usize)> {
    if !word_at(chars, at, "const") {
        return None;
    }
    let name_at = skip_ws(chars, at + 5);
    let konst = ident_at(chars, name_at)?;
    if !konst.starts_with("ALL") {
        return None;
    }
    let mut i = skip_ws(chars, name_at + konst.chars().count());
    if chars.get(i) != Some(&':') {
        return None;
    }
    i = skip_ws(chars, i + 1);
    if chars.get(i) == Some(&'&') {
        i = skip_ws(chars, i + 1);
    }
    if chars.get(i) != Some(&'[') {
        return None;
    }
    let ty = ident_at(chars, skip_ws(chars, i + 1))?;
    // Find the `= &[` / `= [` that opens the literal, on the same item.
    let mut j = i;
    while j < chars.len() && chars[j] != '=' && chars[j] != ';' {
        j += 1;
    }
    if chars.get(j) != Some(&'=') {
        return None;
    }
    Some((konst, ty, j))
}

fn scan(root: &Path) -> (Vec<AllList>, BTreeMap<String, Vec<EnumDecl>>) {
    let mut lists = Vec::new();
    let mut enums: BTreeMap<String, Vec<EnumDecl>> = BTreeMap::new();
    let mut files = rust_sources(&root.join("crates/reims-vgpu/src"));
    files.extend(rust_sources(&root.join("crates/reims-vgpu/tests")));
    for path in files {
        let text = std::fs::read_to_string(&path).unwrap_or_default();
        let chars: Vec<char> = blank_comments(&text).chars().collect();
        let rel = path
            .strip_prefix(root)
            .unwrap_or(&path)
            .to_string_lossy()
            .into_owned();
        for i in 0..chars.len() {
            if word_at(&chars, i, "enum") {
                let name_at = skip_ws(&chars, i + 4);
                if let Some(name) = ident_at(&chars, name_at) {
                    let mut j = skip_ws(&chars, name_at + name.chars().count());
                    if chars.get(j) == Some(&'{') {
                        enums.entry(name).or_default().push(EnumDecl {
                            file: rel.clone(),
                            variants: enum_variants(&chars, j),
                        });
                        continue;
                    }
                    // `enum Foo<T> {` — step over the generics.
                    if chars.get(j) == Some(&'<') {
                        let mut d = 0i32;
                        while j < chars.len() {
                            match chars[j] {
                                '<' => d += 1,
                                '>' => {
                                    d -= 1;
                                    if d == 0 {
                                        break;
                                    }
                                }
                                _ => {}
                            }
                            j += 1;
                        }
                        j = skip_ws(&chars, j + 1);
                        if chars.get(j) == Some(&'{') {
                            enums.entry(name).or_default().push(EnumDecl {
                                file: rel.clone(),
                                variants: enum_variants(&chars, j),
                            });
                        }
                    }
                }
                continue;
            }
            let Some((konst, mut ty, eq)) = parse_all_header(&chars, i) else {
                continue;
            };
            // `[Self; 2]` names the type of the enclosing `impl`.
            if ty == "Self" {
                let mut k = i;
                let mut found = None;
                while k > 0 {
                    k -= 1;
                    if word_at(&chars, k, "impl") {
                        let mut m = skip_ws(&chars, k + 4);
                        if let Some(first) = ident_at(&chars, m) {
                            m = skip_ws(&chars, m + first.chars().count());
                            found = Some(if word_at(&chars, m, "for") {
                                ident_at(&chars, skip_ws(&chars, m + 3)).unwrap_or(first)
                            } else {
                                first
                            });
                        }
                        break;
                    }
                }
                let Some(f) = found else { continue };
                ty = f;
            }
            let (lo, hi) = enclosing_item(&chars, eq).unwrap_or((0, chars.len()));
            lists.push(AllList {
                file: rel.clone(),
                konst,
                mentioned: mentioned_variants(&chars, lo, hi, &ty),
                ty,
            });
        }
    }
    (lists, enums)
}

/// The scanner can see an `ALL` list, an `enum`, and the difference between a
/// covered variant and an uncovered one.
///
/// A structural test is guilty until it has failed. This one's failure mode is
/// silence — a regex that matches nothing reports a clean tree — so it refuses a
/// verdict until it has proved it parsed real input on both sides.
#[test]
fn the_scanner_sees_lists_and_enums_before_it_reports_anything() {
    let root = workspace_root();
    let (lists, enums) = scan(&root);
    assert!(
        lists.len() >= 10,
        "the scanner found only {} `const ALL` lists; it is not reading the tree",
        lists.len()
    );
    assert!(
        enums.len() >= 100,
        "the scanner found only {} enums; it is not reading the tree",
        enums.len()
    );
    let vk = enums
        .get("VkOp")
        .and_then(|d| d.first())
        .expect("VkOp is an enum in this crate");
    assert!(
        vk.variants.len() > 100,
        "VkOp parsed to {} variants, which is not its shape",
        vk.variants.len()
    );
    assert!(
        vk.variants.iter().any(|v| v == "WindowMapStagingMemory"),
        "the enum parser lost a variant it must see"
    );
    let vk_list = lists
        .iter()
        .find(|l| l.ty == "VkOp")
        .expect("VkOp has an ALL list");
    assert!(
        vk_list.mentioned.len() > 100,
        "the scope for VkOp's list resolved to {} mentions, which is too narrow to mean anything",
        vk_list.mentioned.len()
    );
}

#[test]
fn every_all_list_names_every_variant_of_its_enum() {
    let root = workspace_root();
    let (lists, enums) = scan(&root);
    let mut gaps = Vec::new();
    let mut unresolved = Vec::new();
    let mut checked = 0usize;

    for list in &lists {
        let candidates = enums.get(&list.ty);
        // Same file first: five distinct `DecodeStatus` types live in five
        // modules, so the name alone is not an identity here.
        let decl = candidates.and_then(|c| {
            c.iter().find(|d| d.file == list.file).or_else(|| {
                if c.len() == 1 {
                    c.first()
                } else {
                    None
                }
            })
        });
        let Some(decl) = decl else {
            unresolved.push(format!(
                "{}: `{}: &[{}]` — {}",
                list.file,
                list.konst,
                list.ty,
                match candidates {
                    None => "no enum of that name in the crate (a type alias, or not an enum)"
                        .to_string(),
                    Some(c) =>
                        format!("{} enums share that name and none is in this file", c.len()),
                }
            ));
            continue;
        };
        checked += 1;
        let missing: Vec<&String> = decl
            .variants
            .iter()
            .filter(|v| !list.mentioned.contains(v))
            .collect();
        if !missing.is_empty() {
            gaps.push(format!(
                "  {}: `{}` covers {} of {}'s {} variants; never mentions {:?}",
                list.file,
                list.konst,
                decl.variants.len() - missing.len(),
                list.ty,
                decl.variants.len(),
                missing
            ));
        }
    }

    // Reported, never silent: a list this scan cannot resolve is a list nothing
    // checks, and a count of them that only lived in the scanner would make this
    // test's green mean less than it looks.
    assert!(
        checked >= 5,
        "only {checked} lists resolved to an enum; unresolved: {unresolved:#?}"
    );
    assert!(
        unresolved.len() <= 1,
        "more `const ALL` lists became unresolvable than the one known alias \
         (`translate::vertex`'s `ALL_FORMATS: &[F]`). Resolve or exempt each: {unresolved:#?}"
    );
    assert!(
        gaps.is_empty(),
        "a variant missing from an `ALL` list is silently uncovered — the test \
         over that list keeps passing, over a smaller set:\n{}",
        gaps.join("\n")
    );
}
