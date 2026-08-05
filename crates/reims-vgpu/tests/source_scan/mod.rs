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

/// Blank every `#[cfg(test)] mod … { … }` body, keeping offsets stable.
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
    while let Some(rel) = text[at..].find("#[cfg(test)]") {
        let marker = at + rel;
        at = marker + "#[cfg(test)]".len();
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
