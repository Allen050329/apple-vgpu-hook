//! Reading this crate's own source, for the structural tests that hold rules
//! the compiler cannot.
//!
//! Three of those tests exist because the property they check is crate-wide and
//! invisible from any one file: no two declines share a slug, no Vulkan state
//! enum is spelled outside `translate`, nothing reaches past the API floor. All
//! three answer by scanning text, and all three hit the same two hazards — a
//! doc comment that contains code-shaped text, and a `#[cfg(test)]` module whose
//! contents are fixtures rather than product code. Getting either wrong is how a
//! scanner reports a clean tree while measuring the wrong half of it, so the
//! handling lives here once rather than in each caller.
//!
//! Not every helper has every caller, which is what the module-level `dead_code`
//! allow is for: this is a shared toolbox compiled separately into each test
//! binary, so anything one binary does not use is unused *there*.
#![allow(dead_code)]

use std::path::{Path, PathBuf};

/// The repository root, from the crate this test belongs to.
pub fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("the crate lives two levels below the workspace root")
        .to_path_buf()
}

/// Every product source file in the workspace's two guest-facing crates, as
/// `(path relative to `crates/`, text with comments and test modules blanked)`.
///
/// The set the bound scans mean by "this crate". It is **two** crates, not one:
/// `reims-vgpu-wire` decodes guest bytes just as `reims-vgpu` does, so a
/// capacity that cuts a decoded record is exactly as dangerous there, and a scan
/// rooted at one crate's `src/` reports a clean tree for the other by
/// construction — the directory-level version of the failure every one of those
/// scans carries a self-check against.
///
/// Keys are relative to `crates/`, so they read `reims-vgpu/src/…` and
/// `reims-vgpu-wire/src/…` and a verdict says which crate it is about.
///
/// `*/tests.rs` is this workspace's spelling for a unit-test module in its own
/// file; its fixtures shrink and cap collections constantly and none of it is
/// device behaviour, so it is dropped here rather than in each caller.
pub fn guest_facing_sources() -> Vec<(String, String)> {
    let root = workspace_root();
    let crates = root.join("crates");
    ["reims-vgpu", "reims-vgpu-wire"]
        .into_iter()
        .flat_map(|name| rust_sources(&crates.join(name).join("src")))
        .filter(|p| p.file_name().is_some_and(|n| n != "tests.rs"))
        .map(|p| {
            let raw = std::fs::read_to_string(&p).expect("read source");
            let text = blank_test_modules(&blank_comments(&raw));
            let rel = p
                .strip_prefix(&crates)
                .unwrap_or(&p)
                .to_string_lossy()
                .to_string();
            (rel, text)
        })
        .collect()
}

/// Words that make an identifier a statement about how many of something is
/// allowed, rather than about any particular one.
///
/// The vocabulary the three bound scans share. It lived in three copies, one per
/// scan, each carrying a comment claiming the three were "deliberately
/// identical" — and nothing compared them, which is the exact failure
/// `a_bound_is_compared_where_it_is_declared` exists to name. They had already
/// parted: two spelled [`is_bound`] without the path-qualified test that keeps
/// `u64::MAX` from reading as a policy, so a `.min(u64::MAX)` would have been a
/// bound to two scans and not to the third.
///
/// One definition, so a constant renamed out of the vocabulary disappears from
/// all three at once rather than from one — which is what makes a site that
/// moves between the three directions impossible to lose.
pub const BOUND_WORDS: &[&str] = &[
    "CAP", "MAX", "LIMIT", "BUDGET", "RING", "HISTORY", "KEYS", "PER_", "WINDOWS",
];

/// Whether `token` names how many of something is allowed.
///
/// A path-qualified token is never one: `u64::MAX` names a type's extreme, not
/// this device's policy. Callers that tokenize bare identifiers out of a window
/// must decide qualification themselves — that is an extraction question, and
/// only the vocabulary belongs here — but a token that still carries its `::`
/// is rejected here so a caller cannot forget.
///
/// The bare lowercase words are in the vocabulary because a real bound is very
/// often a parameter called exactly `cap`; leaving them out was measured, by
/// injecting a `while self.recent.len() > cap { … }` into `host_writes` and
/// watching the eviction scan stay green. Bare `max` is deliberately absent in
/// any position: it is the name of two std methods and sits beside almost every
/// arithmetic clamp in this crate, and a `max_` *prefix* was tried and reverted
/// in the same hour — its only catch was `delete_task`'s `retain`, next to the
/// `max_task_id_seen` high-water counter, which is a lifetime removal and not a
/// bound at all. `MAX_` in a shouty constant is the spelling a real bound uses
/// here, and the shouty rule already has it.
pub fn is_bound(token: &str) -> bool {
    if token.is_empty() || token.contains("::") {
        return false;
    }
    if is_shouty(token) && BOUND_WORDS.iter().any(|w| token.contains(w)) {
        return true;
    }
    let lower = token.to_ascii_lowercase();
    matches!(lower.as_str(), "cap" | "capacity" | "limit" | "budget")
        || lower.contains("_cap")
        || lower.contains("cap_")
        || lower.contains("_limit")
        || lower.contains("_budget")
        || lower.contains("_max")
}

/// Ways a walk is cut short by a value rather than by the data running out.
pub const CUT: &[&str] = &[".take(", ".min("];

/// Every bare identifier standing as the whole argument of a [`CUT`] on `line`.
///
/// The extraction half of the walk direction, published apart from the
/// vocabulary half so two tests can ask different questions of one population.
/// `a_bounded_walk_says_what_it_skips` keeps the arguments [`is_bound`] accepts
/// and demands a verdict for each; `a_bound_in_a_cut_is_named_like_one` keeps
/// the shouty ones it *rejects* and demands they be renamed. Sharing this
/// function is what makes the second a gate on the first rather than a second
/// opinion about it — if this drifted, the gate would certify a population the
/// scan does not read.
///
/// Only a bare whole argument counts. `.min(a.len())` and `.take(over_cap + 1)`
/// are arithmetic over a runtime value rather than a policy, and a `::`-carrying
/// path is dropped here so `u32::MAX` never reaches either question.
pub fn cut_arguments(line: &str) -> Vec<String> {
    let mut found = Vec::new();
    for cut in CUT {
        let mut from = 0;
        while let Some(at) = line[from..].find(cut) {
            let open = from + at + cut.len();
            from = open;
            let arg: String = line[open..]
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_' || *c == ':')
                .collect();
            if arg.contains("::") {
                continue;
            }
            // The argument must be the whole argument: `MAX_FOO)` not `MAX_FOO -
            // 1)` or `MAX_FOO as usize)`, both of which are arithmetic.
            if !line[open + arg.len()..].starts_with(')') {
                continue;
            }
            found.push(arg);
        }
    }
    found
}

/// Whether `token` is spelled like a constant: `SCREAMING_SNAKE_CASE`.
///
/// Published beside [`is_bound`] because the naming gate needs the same answer
/// for the opposite purpose — [`is_bound`] asks whether a constant says it is a
/// bound, and the gate asks which constants had to.
pub fn is_shouty(token: &str) -> bool {
    token.len() >= 3
        && token.chars().any(|c| c.is_ascii_uppercase())
        && token
            .chars()
            .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
}

/// Every `.rs` file under `dir`, recursively, in a stable order.
pub fn rust_sources(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    collect(dir, &mut out);
    out.sort();
    out
}

fn collect(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).expect("source directory must be readable") {
        let path = entry.expect("a readable dir entry").path();
        if path.is_dir() {
            collect(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

/// Blank out comments, leaving string literals — including raw strings — intact.
///
/// Every comment byte becomes a space and every newline stays a newline, so line
/// numbers and offsets survive. Doc comments are the reason: `observe/decline.rs`
/// writes `impl<T: Decline> Display for T` inside one, and `engine/types.rs`
/// names a dozen `vk::Format` variants in field docs. A scanner that reads those
/// is reporting prose.
pub fn blank_comments(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out: Vec<char> = Vec::with_capacity(chars.len());
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        // Raw string: `r"…"` or `r#"…"#` with any number of hashes.
        if c == 'r' && i + 1 < chars.len() && (chars[i + 1] == '"' || chars[i + 1] == '#') {
            let mut hashes = 0;
            let mut j = i + 1;
            while j < chars.len() && chars[j] == '#' {
                hashes += 1;
                j += 1;
            }
            if j < chars.len() && chars[j] == '"' {
                out.push('r');
                out.extend(std::iter::repeat_n('#', hashes));
                out.push('"');
                j += 1;
                while j < chars.len() {
                    if chars[j] == '"' && chars[j + 1..].iter().take(hashes).all(|c| *c == '#') {
                        out.push('"');
                        out.extend(std::iter::repeat_n('#', hashes));
                        j += 1 + hashes;
                        break;
                    }
                    out.push(chars[j]);
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
                while i < chars.len() {
                    if chars[i] == '\\' && i + 1 < chars.len() {
                        out.push(chars[i]);
                        out.push(chars[i + 1]);
                        i += 2;
                        continue;
                    }
                    out.push(chars[i]);
                    i += 1;
                    if chars[i - 1] == '"' {
                        break;
                    }
                }
            }
            // A char literal, so the quote in `'"'` cannot open a string.
            // Lifetimes (`'a`) fall through harmlessly: they contain no quote.
            '\'' if i + 2 < chars.len() && chars[i + 1] == '"' && chars[i + 2] == '\'' => {
                out.extend_from_slice(&chars[i..i + 3]);
                i += 3;
            }
            '/' if chars.get(i + 1) == Some(&'/') => {
                while i < chars.len() && chars[i] != '\n' {
                    out.push(' ');
                    i += 1;
                }
            }
            '/' if chars.get(i + 1) == Some(&'*') => {
                let mut depth = 0usize;
                while i < chars.len() {
                    if chars[i] == '/' && chars.get(i + 1) == Some(&'*') {
                        depth += 1;
                        out.push(' ');
                        out.push(' ');
                        i += 2;
                    } else if chars[i] == '*' && chars.get(i + 1) == Some(&'/') {
                        depth -= 1;
                        out.push(' ');
                        out.push(' ');
                        i += 2;
                        if depth == 0 {
                            break;
                        }
                    } else {
                        out.push(if chars[i] == '\n' { '\n' } else { ' ' });
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
pub fn close_brace(chars: &[char], open: usize) -> usize {
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

/// Remove `#[…]` attribute spans, whose own string literals are not the code's.
///
/// `#[cfg(all(feature = "backend-metal", target_os = "macos"))]` is the case
/// that matters: reading its literals as slugs reports `macos` as a three-way
/// collision between three unrelated status enums.
pub fn strip_attributes(chars: &[char]) -> String {
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

/// Blank every test module's body, keeping offsets stable.
///
/// Both spellings of the attribute are read: `#[cfg(test)]` and the
/// `#[cfg(all(test, feature = "…"))]` form that `draw/vulkan.rs` uses for
/// `vulkan_split_tests`. Brace-matching each body rather than cutting at the
/// first marker is what keeps production code *after* a test module visible — a
/// first-marker cutoff hides it, and a scanner that hides code reports clean.
///
/// A test module's contents are fixtures: `pools/mod.rs` builds a
/// `SampledImageResource` with a literal `vk::Format::R8G8B8A8_UNORM` and
/// asserts it survives a key round-trip. That is not the product spelling a
/// format, and a scanner that counts it will report erosion where there is a
/// test doing its job.
///
/// Run this **after** [`blank_comments`], so an attribute written inside a doc
/// comment cannot open a region.
pub fn blank_test_modules(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = chars.clone();
    let mut at = 0usize;
    while let Some((rel, marker_len)) = ["#[cfg(test)]", "#[cfg(all(test,"]
        .iter()
        .filter_map(|m| text[at..].find(m).map(|i| (i, m.len())))
        .min()
    {
        let marker = at + rel;
        at = marker + marker_len;
        // Only a module has a body worth blanking; a `#[cfg(test)]` on a `use`
        // or a `const` is a declaration the product side never sees anyway.
        let after: String = text[at..].chars().take(200).collect();
        let Some(mod_rel) = after.find("mod ") else {
            continue;
        };
        // Nothing but whitespace and further attributes may sit between the
        // marker and its `mod`, or this is a different item that merely has one
        // nearby.
        if after[..mod_rel]
            .chars()
            .any(|c| !c.is_whitespace() && c != '#' && c != '[' && c != ']' && c != '(' && c != ')')
        {
            continue;
        }
        let Some(brace_rel) = text[at..].find('{') else {
            break;
        };
        let open = text[..at + brace_rel].chars().count();
        let end = close_brace(&chars, open);
        for slot in out.iter_mut().take(end).skip(open) {
            if *slot != '\n' {
                *slot = ' ';
            }
        }
        at = text
            .char_indices()
            .nth(end)
            .map(|(byte, _)| byte)
            .unwrap_or(text.len());
    }
    out.into_iter().collect()
}
