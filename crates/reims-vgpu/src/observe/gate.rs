//! Static gates over the decline vocabulary.
//!
//! These generalize three tests that already existed and each covered one
//! island: `every_reason_has_its_own_slug` and `slugs_are_log_safe` in
//! `translate/reason.rs` and `engine/reason.rs` (each checking only its own
//! enum), and `declined_slugs_are_actually_emitted` in `translate/coverage.rs`
//! (checking only render-descriptor fields). Written in the same style as
//! `translate/gate.rs` and `caps/gate.rs`: read the source, assert a property,
//! name the defect in the failure message.

use super::decline::{Emission, REGISTRY};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

fn crate_src() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src")
}

fn rust_files(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return out;
    };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            out.extend(rust_files(&p));
        } else if p.extension().is_some_and(|x| x == "rs") {
            out.push(p);
        }
    }
    out
}

/// Byte offsets of `Result<Success, String>` spellings in Rust code.
///
/// This is intentionally a small lexical scan rather than a same-line grep:
/// rustfmt can wrap either generic argument, and nested generic `>` tokens must
/// not be mistaken for the outer result. Comments and ordinary string/char
/// literals are masked so the gate can explain the forbidden shape and test
/// itself without creating a false source hit.
fn result_string_error_offsets(src: &str) -> Vec<usize> {
    #[derive(Clone, Copy)]
    enum State {
        Code,
        LineComment,
        BlockComment(usize),
        String,
        RawString(usize),
        Char,
    }

    let bytes = src.as_bytes();
    let mut masked = bytes.to_vec();
    let mut state = State::Code;
    let mut i = 0usize;
    while i < bytes.len() {
        state = match state {
            State::Code if bytes[i..].starts_with(b"//") => {
                masked[i] = b' ';
                if i + 1 < masked.len() {
                    masked[i + 1] = b' ';
                }
                i += 2;
                State::LineComment
            }
            State::Code if bytes[i..].starts_with(b"/*") => {
                masked[i] = b' ';
                if i + 1 < masked.len() {
                    masked[i + 1] = b' ';
                }
                i += 2;
                State::BlockComment(1)
            }
            State::Code if bytes[i] == b'r' => {
                let mut quote = i + 1;
                while bytes.get(quote) == Some(&b'#') {
                    quote += 1;
                }
                if bytes.get(quote) == Some(&b'"') {
                    let hashes = quote - (i + 1);
                    for byte in masked.iter_mut().take(quote + 1).skip(i) {
                        *byte = b' ';
                    }
                    i = quote + 1;
                    State::RawString(hashes)
                } else {
                    i += 1;
                    State::Code
                }
            }
            State::Code if bytes[i] == b'"' => {
                masked[i] = b' ';
                i += 1;
                State::String
            }
            State::Code
                if bytes[i] == b'\''
                    && (bytes.get(i + 1) == Some(&b'\\') || bytes.get(i + 2) == Some(&b'\'')) =>
            {
                masked[i] = b' ';
                i += 1;
                State::Char
            }
            State::Code => {
                i += 1;
                State::Code
            }
            State::LineComment if bytes[i] == b'\n' => {
                i += 1;
                State::Code
            }
            State::LineComment => {
                masked[i] = b' ';
                i += 1;
                State::LineComment
            }
            State::BlockComment(depth) if bytes[i..].starts_with(b"/*") => {
                masked[i] = b' ';
                if i + 1 < masked.len() {
                    masked[i + 1] = b' ';
                }
                i += 2;
                State::BlockComment(depth + 1)
            }
            State::BlockComment(depth) if bytes[i..].starts_with(b"*/") => {
                masked[i] = b' ';
                if i + 1 < masked.len() {
                    masked[i + 1] = b' ';
                }
                i += 2;
                if depth == 1 {
                    State::Code
                } else {
                    State::BlockComment(depth - 1)
                }
            }
            State::BlockComment(depth) => {
                if bytes[i] != b'\n' {
                    masked[i] = b' ';
                }
                i += 1;
                State::BlockComment(depth)
            }
            State::String if bytes[i] == b'\\' && i + 1 < bytes.len() => {
                masked[i] = b' ';
                masked[i + 1] = b' ';
                i += 2;
                State::String
            }
            State::String if bytes[i] == b'"' => {
                masked[i] = b' ';
                i += 1;
                State::Code
            }
            State::String => {
                if bytes[i] != b'\n' {
                    masked[i] = b' ';
                }
                i += 1;
                State::String
            }
            State::RawString(hashes)
                if bytes[i] == b'"' && (0..hashes).all(|n| bytes.get(i + 1 + n) == Some(&b'#')) =>
            {
                for byte in masked.iter_mut().take(i + 1 + hashes).skip(i) {
                    *byte = b' ';
                }
                i += 1 + hashes;
                State::Code
            }
            State::RawString(hashes) => {
                if bytes[i] != b'\n' {
                    masked[i] = b' ';
                }
                i += 1;
                State::RawString(hashes)
            }
            State::Char if bytes[i] == b'\\' && i + 1 < bytes.len() => {
                masked[i] = b' ';
                masked[i + 1] = b' ';
                i += 2;
                State::Char
            }
            State::Char if bytes[i] == b'\'' => {
                masked[i] = b' ';
                i += 1;
                State::Code
            }
            State::Char => {
                if bytes[i] != b'\n' {
                    masked[i] = b' ';
                }
                i += 1;
                State::Char
            }
        };
    }

    let mut hits = Vec::new();
    let mut from = 0usize;
    while let Some(rel) = masked[from..]
        .windows(6)
        .position(|window| window == b"Result")
    {
        let start = from + rel;
        let mut cursor = start + 6;
        while masked.get(cursor).is_some_and(u8::is_ascii_whitespace) {
            cursor += 1;
        }
        if masked.get(cursor) != Some(&b'<') {
            from = start + 6;
            continue;
        }
        cursor += 1;
        let mut depth = 1usize;
        let mut error_start = None;
        let mut end = None;
        while cursor < masked.len() {
            match masked[cursor] {
                b'<' => depth += 1,
                b'>' => {
                    depth -= 1;
                    if depth == 0 {
                        end = Some(cursor);
                        break;
                    }
                }
                b',' if depth == 1 && error_start.is_none() => error_start = Some(cursor + 1),
                _ => {}
            }
            cursor += 1;
        }
        if let (Some(error_start), Some(end)) = (error_start, end) {
            let error: String = masked[error_start..end]
                .iter()
                .copied()
                .filter(|byte| !byte.is_ascii_whitespace())
                .map(char::from)
                .collect();
            if matches!(
                error.trim_end_matches(','),
                "String" | "std::string::String" | "alloc::string::String"
            ) {
                hits.push(start);
            }
            from = end + 1;
        } else {
            from = start + 6;
        }
    }
    hits
}

#[test]
fn the_result_string_scanner_reads_wrapped_types_and_only_the_error_slot() {
    let source = r#"
        fn a() -> Result<Vec<String>, Typed> { todo!() }
        fn b() -> Result<
            Vec<Result<u32, Typed>>,
            String,
        > { todo!() }
        type C = Result<u32, std::string::String>;
        // Result<u32, String> in a comment is not code.
        const NOTE: &str = "Result<u32, String> in a string is not code";
    "#;
    let hits = result_string_error_offsets(source);
    assert_eq!(hits.len(), 2, "wrapped/direct string errors: {hits:?}");
}

/// Free-text `Result` errors cannot return. A typed decline may preserve an
/// external driver's prose as a field, but the error carrier itself must remain
/// exhaustively matchable and registered.
#[test]
fn no_result_uses_string_as_its_error_type() {
    let root = crate_src();
    let mut hits = Vec::new();
    for path in rust_files(&root) {
        let src = std::fs::read_to_string(&path).expect("read Rust source");
        for offset in result_string_error_offsets(&src) {
            let line = src[..offset].bytes().filter(|byte| *byte == b'\n').count() + 1;
            hits.push(format!(
                "{}:{line}",
                path.strip_prefix(&root).unwrap_or(&path).to_string_lossy()
            ));
        }
    }
    assert!(
        hits.is_empty(),
        "free-text Result errors returned; use a registered typed decline:\n  {}",
        hits.join("\n  ")
    );
}

/// Two checks sharing a slug is the exact failure `AGENTS.md` names: you grep
/// the fail log, see the slug fire, and still cannot tell which of the two
/// refused. Uniqueness is now crate-wide, not per-enum — the old per-enum tests
/// could not have caught `translate` and `engine` colliding.
#[test]
fn every_registered_slug_is_unique_crate_wide() {
    let mut owner: BTreeMap<&str, &str> = BTreeMap::new();
    let mut clashes = Vec::new();
    for class in REGISTRY {
        for slug in class.slugs {
            if let Some(prev) = owner.insert(slug, class.type_name) {
                clashes.push(format!(
                    "`{slug}` claimed by both {prev} and {}",
                    class.type_name
                ));
            }
        }
    }
    assert!(
        clashes.is_empty(),
        "decline slugs must be unique crate-wide:\n  {}",
        clashes.join("\n  ")
    );
}

/// Slugs are grepped out of a space-separated log line, so they may not carry
/// whitespace or an `=`, and they stay snake_case for consistency with the
/// existing `caps`, `translate` and census slugs.
#[test]
fn every_registered_slug_is_log_safe() {
    for class in REGISTRY {
        for slug in class.slugs {
            assert!(!slug.is_empty(), "{}: empty slug", class.type_name);
            assert!(
                slug.bytes()
                    .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_'),
                "{}: slug {slug:?} must be lowercase snake_case",
                class.type_name
            );
        }
    }
}

/// Every registered type must actually exist where the registry says it does.
/// A row pointing at a moved or deleted file documents a vocabulary the crate
/// no longer has.
#[test]
fn every_registered_type_is_defined_where_the_registry_says() {
    for class in REGISTRY {
        let path = crate_src().join(class.defined_in);
        let src = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("{}: {}: {e}", class.type_name, class.defined_in));
        assert!(
            src.contains(class.type_name),
            "{} is not defined in {}",
            class.type_name,
            class.defined_in
        );
    }
}

/// Does `src` open an `Emit::decline`/`Emit::refusal` whose first argument is the
/// string `event`?
///
/// Textual, because the alternative is resolving types, and deliberately anchored
/// on the *first* argument rather than "the event appears somewhere in the file":
/// an event token also shows up in tests and in prose, and a row that passed on
/// those would be back to claiming a file rather than a line. rustfmt breaks long
/// calls after the open paren, so the argument may be on the following line —
/// hence the skip over whitespace rather than a `contains` of the joined text.
fn emits_event(src: &str, event: &str) -> bool {
    let needle = format!("\"{event}\"");
    for opener in ["Emit::decline(", "Emit::refusal("] {
        let mut from = 0usize;
        while let Some(rel) = src[from..].find(opener) {
            let after = from + rel + opener.len();
            if src[after..].trim_start().starts_with(&needle) {
                return true;
            }
            from = after;
        }
    }
    false
}

#[test]
fn the_emission_check_reads_the_first_argument_not_the_whole_file() {
    assert!(emits_event(
        "crate::observe::Emit::refusal(\"stream_frame_fail\", &status)",
        "stream_frame_fail"
    ));
    // rustfmt's wrapped form, as in `metal_draw.rs`.
    assert!(emits_event(
        "Emit::decline(\n    \"shader_state_degraded\",\n    &reason,\n)",
        "shader_state_degraded"
    ));
    // The event named as a *field value* or in prose is not an emission. This is
    // the case that makes the check a claim about a line: without it, a comment
    // mentioning the slug would keep a deleted emission green.
    assert!(!emits_event(
        "// we used to Emit::decline here for blit_fail\ne.field(\"why\", \"blit_fail\")",
        "blit_fail"
    ));
    // A different event in the same file must not satisfy this row.
    assert!(!emits_event(
        "Emit::refusal(\"blit_decode\", &status)",
        "render_decode"
    ));
}

/// **The gate that closes the unchecked handoff.**
///
/// `translate/` and `caps/` are pure: they return typed declines and log
/// nothing, which is correct. But nothing checked that any *caller* logged
/// them, and that is the mechanism by which a decline can be typed, returned,
/// and still never reach `/tmp/reims-vgpu-fail.log`. A registered type must name a
/// `(file, event)` pair where it meets an `observe::` emitter, and that call
/// must really be there.
#[test]
fn every_registered_type_reaches_the_sink() {
    let mut unlogged = Vec::new();
    for class in REGISTRY {
        let sites = match class.emission {
            // An unreachable type is exempt from *emission*, not from scrutiny:
            // the claim is checked below by
            // `unreachable_declines_really_have_no_caller`.
            Emission::Unreachable(why) => {
                assert!(
                    !why.is_empty(),
                    "{}: Unreachable needs an argument",
                    class.type_name
                );
                continue;
            }
            Emission::At(sites) => sites,
        };
        if sites.is_empty() {
            unlogged.push(format!(
                "{} names no emission site — a typed decline nobody logs is \
                 still a silent failure",
                class.type_name
            ));
            continue;
        }
        for (site, event) in sites {
            let path = crate_src().join(site);
            let Ok(src) = std::fs::read_to_string(&path) else {
                unlogged.push(format!(
                    "{}: emission site {site} does not exist",
                    class.type_name
                ));
                continue;
            };
            // The row must name the *line*, not the file. Either builder:
            // `decline` for an always-refusal type, `refusal` for a status enum
            // whose `Ok` must not produce a line. Matching the event token means
            // deleting the emission fails the gate, which naming only the file
            // did not — `contains(type_name)` was satisfied by an unused import.
            if !emits_event(&src, event) {
                unlogged.push(format!(
                    "{}: {site} has no Emit::decline/refusal(\"{event}\") — the \
                     row names an emission that is not there",
                    class.type_name
                ));
            }
        }
    }
    assert!(
        unlogged.is_empty(),
        "these decline types never reach the always-on sink:\n  {}",
        unlogged.join("\n  ")
    );
}

/// `Emission::Unreachable` is a strong claim — "nothing can log this because
/// nothing calls it" — and a wrong one would excuse a genuinely silent path.
/// It is therefore checked, not trusted: if a caller appears, the exemption
/// must be replaced by a real emission site.
#[test]
fn unreachable_declines_really_have_no_caller() {
    for class in REGISTRY {
        let Emission::Unreachable(_) = class.emission else {
            continue;
        };
        let mut callers = Vec::new();
        for path in rust_files(&crate_src()) {
            let rel = path
                .strip_prefix(crate_src())
                .unwrap_or(&path)
                .to_string_lossy()
                .replace('\\', "/");
            // The defining file, the registry and the gates name the type
            // without consuming it.
            if rel == class.defined_in || rel.starts_with("observe/") {
                continue;
            }
            let Ok(src) = std::fs::read_to_string(&path) else {
                continue;
            };
            // A consumer *handles* the error: matches a variant or propagates
            // it. Merely implementing the trait that returns it does not count.
            if src.contains(&format!("{}::", class.type_name))
                && !src.contains(&format!("impl Backend for"))
            {
                callers.push(rel);
            }
        }
        assert!(
            callers.is_empty(),
            "{} is registered Unreachable but is consumed in {callers:?} — \
             give it a real emission site",
            class.type_name
        );
    }
}

/// The source of the block `anchor` opens, delimited by its braces.
///
/// A source scan rather than a value-level enumeration because Rust cannot
/// iterate an enum's variants: the exhaustive `match` inside `slug()` is the one
/// place the compiler *does* force completeness, so scanning that match is how a
/// new variant becomes visible to the gate. Add a variant and the compiler makes
/// you write an arm; write an arm and this scan makes you register its slug.
fn block_after(src: &str, anchor: &str) -> Option<String> {
    let start = src.find(anchor)?;
    let body = &src[start..];
    let bytes = body.as_bytes();
    let mut depth = 0i32;
    let mut in_str = false;
    for (i, b) in bytes.iter().enumerate() {
        match b {
            b'"' if !in_str => in_str = true,
            b'"' if in_str && bytes[i - 1] != b'\\' => in_str = false,
            b'{' if !in_str => depth += 1,
            b'}' if !in_str => {
                depth -= 1;
                if depth == 0 {
                    return Some(body[..i].to_string());
                }
            }
            _ => {}
        }
    }
    None
}

/// Every slug a block returns: the literal on the right of a `match` arm, of a
/// `Some(..)`, or of a `return` — including the form rustfmt wraps a long arm
/// into.
///
/// Deliberately *not* every literal in the block. `fields()` lives in the same
/// impl and its keys (`"backend"`, `"what"`) are lowercase snake_case too, so a
/// shape test alone would count field keys as vocabulary and the census would
/// drift by exactly the number of fields anyone added.
///
/// The wrapped case is load-bearing, not cosmetic: rustfmt turns an arm whose
/// single-literal body exceeds the line width — `X => "a_long_slug",` — into
/// `X => {\n    "a_long_slug"\n}`, at which point the literal is no longer
/// adjacent to the arrow and the direct patterns miss it. A slug silently
/// dropped from the census merely because it is long is exactly the "uncounted
/// refusal" this gate exists to prevent, so `=> {` blocks are read too.
fn slugs_returned_by(block: &str) -> Vec<String> {
    const RESULT_OF: &[&str] = &["=> \"", "=> Some(\"", "return \"", "return Some(\""];
    let mut out = Vec::new();
    let push_if_slug = |out: &mut Vec<String>, lit: &str| {
        if !lit.is_empty()
            && lit
                .bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')
        {
            out.push(lit.to_string());
        }
    };
    for pat in RESULT_OF {
        let mut rest = block;
        while let Some(at) = rest.find(pat) {
            rest = &rest[at + pat.len()..];
            let Some(close) = rest.find('"') else { break };
            push_if_slug(&mut out, &rest[..close]);
            rest = &rest[close + 1..];
        }
    }
    // The rustfmt-wrapped arm: read the first string literal inside each `=> {`
    // block, bounded by that block's own braces so a following arm is not
    // misattributed. A delegating (`=> { reason.slug() }`) or `None` arm has no
    // literal and contributes nothing. Duplicates a literal the direct patterns
    // could never reach, and the caller dedups, so being liberal is safe.
    let bytes = block.as_bytes();
    let mut from = 0usize;
    while let Some(rel) = block[from..].find("=> {") {
        let open = from + rel + "=> {".len() - 1; // index of the '{'
        let mut depth = 0i32;
        let mut end = block.len();
        for (i, b) in bytes.iter().enumerate().skip(open) {
            match b {
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        end = i;
                        break;
                    }
                }
                _ => {}
            }
        }
        let inner = &block[open..end];
        if let Some(q) = inner.find('"') {
            if let Some(close) = inner[q + 1..].find('"') {
                push_if_slug(&mut out, &inner[q + 1..q + 1 + close]);
            }
        }
        from = end;
    }
    out
}

/// The production half of a file: comment lines dropped and everything from the
/// test module onward cut off.
///
/// Both exclusions are load-bearing for [`slugs_passed_to`], which reads
/// *construction sites* rather than a trait impl and so has no braces to bound
/// it. A doc comment showing an example call would otherwise register a slug the
/// crate never writes, and a test constructing a refusal with a made-up reason
/// would register a slug that only exists under `cargo test`.
///
/// Every `#[cfg(test)] mod` block is removed, **not** everything after the first
/// one and not every `#[cfg(test)]` item. Several files carry either a test-only
/// helper before production (`blit_exec.rs`) or a complete test module in the
/// middle of the file (`backend/metal/render.rs`). Cutting at either marker
/// silently hides real vocabulary below it. A `#[cfg(test)]` helper outside a
/// test module is therefore still read — it sits in the production namespace,
/// so a slug it constructs is one the crate can write.
fn production_source(src: &str) -> String {
    let mut body = String::with_capacity(src.len());
    let mut copy_from = 0usize;
    let mut search_from = 0usize;
    while let Some(rel) = src[search_from..].find("#[cfg(test)]") {
        let at = search_from + rel;
        let after = &src[at + "#[cfg(test)]".len()..];
        if after
            .lines()
            .find(|l| !l.trim().is_empty())
            .is_some_and(|l| l.trim_start().starts_with("mod "))
        {
            let tail = &src[at..];
            let Some(mod_at) = tail.find("mod ") else {
                break;
            };
            let Some(block) = block_after(tail, "mod ") else {
                break;
            };
            let end = at + mod_at + block.len() + 1;
            body.push_str(&src[copy_from..at]);
            // Keep line structure stable for diagnostics while hiding every
            // literal inside the test module from the vocabulary extractor.
            body.extend(
                src[at..end]
                    .bytes()
                    .map(|byte| if byte == b'\n' { '\n' } else { ' ' }),
            );
            copy_from = end;
            search_from = end;
            continue;
        }
        search_from = at + "#[cfg(test)]".len();
    }
    body.push_str(&src[copy_from..]);
    body.lines()
        .filter(|l| !l.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Every slug written as a literal at a *call* site: the first string literal
/// inside the parentheses opened by each occurrence of `call`.
///
/// This is the shape a vocabulary takes when the reason lives in the value or in
/// a side channel rather than in a `match` — `FenceStatus::Unsupported(
/// "fence_domain_unknown")`, or the blit rail's `br(BlitStatus::Bounds,
/// "fill_out_of_range")`. Nobody writes a 178-arm `slug()`, so for those types
/// the construction site *is* the vocabulary, and a census that could not read it
/// there would have to exempt them — which is how a rail stays uncounted.
///
/// A site with no literal is skipped rather than flagged: it is forwarding a slug
/// it received (`br(BlitStatus::Unsupported, e.slug())`, `FenceStatus::
/// Unsupported(why)`), and the delegate that owns that slug is registered on its
/// own row. That is also why `fn br(status: BlitStatus, reason: &'static str)` —
/// the definition — contributes nothing.
fn slugs_passed_to(src: &str, call: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = src;
    while let Some(at) = rest.find(call) {
        rest = &rest[at + call.len()..];
        let bytes = rest.as_bytes();
        let mut depth = 1i32;
        let mut lit: Option<&str> = None;
        let mut i = 0usize;
        while i < bytes.len() {
            match bytes[i] {
                b'(' => depth += 1,
                b')' => {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                }
                b'"' if lit.is_none() => {
                    let start = i + 1;
                    let mut j = start;
                    while j < bytes.len() && !(bytes[j] == b'"' && bytes[j - 1] != b'\\') {
                        j += 1;
                    }
                    lit = Some(&rest[start..j.min(rest.len())]);
                    i = j;
                }
                _ => {}
            }
            i += 1;
        }
        if let Some(l) = lit {
            if !l.is_empty()
                && l.bytes()
                    .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')
            {
                out.push(l.to_string());
            }
        }
    }
    out
}

/// Every block the gate reads for one registered type: its own trait impl, plus
/// any delegate the row names.
fn vocabulary_blocks(class: &super::decline::DeclineClass) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for trait_name in ["Decline for ", "Refusal for "] {
        out.push((
            class.defined_in.to_string(),
            format!("{trait_name}{}", class.type_name),
        ));
    }
    for (file, anchor) in class.slug_blocks {
        out.push((file.to_string(), anchor.to_string()));
    }
    out
}

/// **The gate that keeps the census honest.**
///
/// A registry row that lists a slug its type cannot produce documents a refusal
/// the crate cannot make; a type that produces a slug no row lists is a refusal
/// nobody can count. Both are the census lying, and both are cheap to catch by
/// reading the one place the vocabulary is written down.
#[test]
fn every_row_lists_exactly_the_slugs_its_type_writes() {
    let mut wrong = Vec::new();
    for class in REGISTRY {
        let mut found: Vec<String> = Vec::new();
        // Types whose reason is carried in the value or side-channelled write
        // their vocabulary at construction sites, not in a match arm.
        for (rel, call) in class.slug_calls {
            let path = crate_src().join(rel);
            match std::fs::read_to_string(&path) {
                Ok(src) => {
                    let sites = slugs_passed_to(&production_source(&src), call);
                    if sites.is_empty() {
                        wrong.push(format!(
                            "{}: no `{call}` site in {rel} passes a slug literal \
                             — the row points at a vocabulary that is not there",
                            class.type_name
                        ));
                    }
                    found.extend(sites);
                }
                Err(e) => wrong.push(format!("{}: {rel}: {e}", class.type_name)),
            }
        }
        for (rel, anchor) in vocabulary_blocks(class) {
            let path = crate_src().join(&rel);
            let Ok(src) = std::fs::read_to_string(&path) else {
                wrong.push(format!("{}: {rel} does not exist", class.type_name));
                continue;
            };
            let Some(block) = block_after(&src, &anchor) else {
                continue;
            };
            // Only the vocabulary methods, not the whole impl: `fields()` sits
            // in the same block and its arms return field *values*
            // (`MmioWindow::Gfx => "gfx"`), which are not slugs. Reading the
            // impl wholesale counted those, and this gate caught it on its own
            // first registration — which is the argument for the gate.
            let mut any = false;
            for method in ["fn slug", "fn refusal"] {
                if let Some(body) = block_after(&block, method) {
                    found.extend(slugs_returned_by(&body));
                    any = true;
                }
            }
            if !any {
                wrong.push(format!(
                    "{}: block `{anchor}` in {rel} has neither `fn slug` nor \
                     `fn refusal` — the gate cannot read its vocabulary",
                    class.type_name
                ));
            }
        }
        if found.is_empty() {
            wrong.push(format!(
                "{}: no slug literals found — does it implement Decline or \
                 Refusal in {}?",
                class.type_name, class.defined_in
            ));
            continue;
        }
        found.sort_unstable();
        found.dedup();
        let mut listed: Vec<String> = class.slugs.iter().map(|s| s.to_string()).collect();
        listed.sort_unstable();
        listed.dedup();
        for slug in found.iter().filter(|s| !listed.contains(s)) {
            wrong.push(format!(
                "{}: writes `{slug}` but no registry row lists it — an \
                 uncounted refusal",
                class.type_name
            ));
        }
        for slug in listed.iter().filter(|s| !found.contains(s)) {
            wrong.push(format!(
                "{}: registry lists `{slug}` but the type never writes it — a \
                 refusal the crate cannot make",
                class.type_name
            ));
        }
    }
    assert!(
        wrong.is_empty(),
        "the decline registry does not match the vocabulary the code writes:\n  {}",
        wrong.join("\n  ")
    );
}

/// The extractor has to be right or every row above passes vacuously, so it is
/// checked against hand-written blocks with the exact shapes the crate uses.
#[test]
fn the_slug_extractor_reads_arms_and_not_field_keys() {
    let block = r#"impl Decline for X {
        fn slug(&self) -> &'static str {
            match self {
                Self::A => "a_slug",
                Self::B(op, _) => op.slug(),
                Self::C => "c_slug",
            }
        }
        fn fields(&self) -> Vec<(&'static str, String)> {
            vec![(
                "window",
                match self {
                    Self::A => "gfx",
                    _ => "iosfc",
                }
                .to_string(),
            )]
        }
    }"#;
    let slug_body = block_after(block, "fn slug").expect("slug body");
    assert_eq!(slugs_returned_by(&slug_body), vec!["a_slug", "c_slug"]);
    // The load-bearing exclusion: a `fields()` arm returning a *value* is not
    // vocabulary. Reading the whole impl counted `"gfx"` as a slug.
    let fields_body = block_after(block, "fn fields").expect("fields body");
    assert!(slugs_returned_by(&fields_body).contains(&"gfx".to_string()));
    assert!(!slugs_returned_by(&slug_body).contains(&"gfx".to_string()));

    // rustfmt wraps an arm whose single-literal body exceeds the line width, and
    // a delegating arm wrapped the same way carries no literal. The wrapped slug
    // must still be read; the wrapped delegation must still contribute nothing —
    // otherwise a slug vanishes from the census purely for being long, which is
    // how DrawReason's two `no_device_local_memory_for_*` slugs first slipped.
    let wrapped = r#"impl Decline for W {
        fn slug(&self) -> &'static str {
            match self {
                Self::Short => "short",
                Self::LongOne { .. } => {
                    "a_very_long_wrapped_slug_name"
                }
                Self::Delegates(inner) => {
                    inner.slug()
                }
            }
        }
    }"#;
    let wrapped_body = block_after(wrapped, "fn slug").expect("wrapped slug body");
    let got = slugs_returned_by(&wrapped_body);
    assert!(got.contains(&"short".to_string()));
    assert!(
        got.contains(&"a_very_long_wrapped_slug_name".to_string()),
        "a wrapped arm's slug must still be counted: {got:?}"
    );
    assert_eq!(got.len(), 2, "a wrapped delegation is not a slug: {got:?}");

    let refusal = r#"impl Refusal for S {
        fn refusal(&self) -> Option<&'static str> {
            match self {
                Self::Ok | Self::Done => None,
                Self::ErrShort => Some("s_short"),
            }
        }
    }"#;
    let body = block_after(refusal, "fn refusal").expect("refusal body");
    assert_eq!(slugs_returned_by(&body), vec!["s_short"]);

    // Brace walking must stop at the end of the anchored impl, not run on into
    // the next one — otherwise a neighbouring type's slugs are attributed here.
    let two = "impl Decline for A {\n  fn slug(&self) { \"a\" }\n}\nimpl Decline for B {\n  fn slug(&self) { \"b\" }\n}\n";
    let a = block_after(two, "Decline for A").expect("A's block");
    assert!(!a.contains("\"b\""), "block ran past its own impl: {a}");
}

/// The call-site extractor is the only thing standing between a carried-reason
/// vocabulary and being exempt from the census, so it is checked against the
/// exact shapes the crate writes — including the three it must *not* count.
#[test]
fn the_call_site_extractor_reads_constructed_reasons_and_skips_forwarded_ones() {
    let src = r#"
fn br(status: BlitStatus, reason: &'static str) -> BlitStatus { status }

fn a() -> FenceStatus {
    return FenceStatus::Unsupported("fence_domain_unknown");
}
fn b() -> FenceStatus {
    refused(
        FenceStatus::Unsupported("event_plan_invalid"),
        r,
        |e| e.field("task", t),
    )
}
fn c(why: &'static str) -> FenceStatus { FenceStatus::Unsupported(why) }
"#;
    let got = slugs_passed_to(&production_source(src), "FenceStatus::Unsupported(");
    assert_eq!(
        got,
        vec!["fence_domain_unknown", "event_plan_invalid"],
        "a forwarded `why` is delegation, not vocabulary"
    );

    // The side-channel shape: the slug is the *second* argument, and the `fn br`
    // signature above must contribute nothing despite matching `br(`.
    let channel = r#"
fn br(status: BlitStatus, reason: &'static str) -> BlitStatus { status }
fn d() -> BlitStatus {
    if x { return br(BlitStatus::Bounds, "fill_out_of_range"); }
    y.ok_or_else(|| br(BlitStatus::Capacity, "copy_region_src_row_overflow"))?;
    z.map_err(|e| br(BlitStatus::Unsupported, e.slug()))?
}
"#;
    assert_eq!(
        slugs_passed_to(&production_source(channel), "br("),
        vec!["fill_out_of_range", "copy_region_src_row_overflow"],
        "the definition and the delegating site must both contribute nothing"
    );

    // A doc comment demonstrating the call, and a test constructing a made-up
    // reason, would each register a slug the crate never writes. The
    // `#[cfg(test)]` *helper* partway down is the trap: cutting at the first
    // `#[cfg(test)]` rather than at the test module hid 2450 of `blit_exec.rs`'s
    // lines, and with them every slug the file writes.
    let noise = r#"
/// Refuse with br(BlitStatus::Bounds, "doc_example_only").
fn e() -> BlitStatus { br(BlitStatus::Bounds, "real_reason") }

#[cfg(test)]
fn reset_dedup_for_test() {}

fn g() -> BlitStatus { br(BlitStatus::Bounds, "reason_below_the_helper") }

#[cfg(test)]
mod tests {
    fn f() -> BlitStatus { br(BlitStatus::Bounds, "test_only_reason") }
}

fn h() -> BlitStatus { br(BlitStatus::Bounds, "reason_below_the_test_module") }
"#;
    assert_eq!(
        slugs_passed_to(&production_source(noise), "br("),
        vec![
            "real_reason",
            "reason_below_the_helper",
            "reason_below_the_test_module"
        ],
        "neither a cfg(test) helper nor a mid-file test module may hide \
         later production vocabulary"
    );
}

/// The census: how much of the crate's refusal surface is typed and registered.
///
/// This is a **baseline, not a target**. The audit that opened this phase
/// counted 21 error/decline types of which 5 carried a `slug()`; the rest could
/// not be grepped by name, counted, or exhaustively tested. Each phase that
/// migrates a type moves this number up and updates it in the same commit, so
/// the remaining silent surface stays a written figure rather than a vibe.
#[test]
fn the_registry_is_what_the_last_migration_recorded() {
    let types = REGISTRY.len();
    let slugs: usize = REGISTRY.iter().map(|c| c.slugs.len()).sum();
    assert_eq!(
        (types, slugs),
        (74, 1755),
        "the decline registry moved; update this baseline in the same commit \
         that moves it, and say which way in the journal"
    );
}

/// The staged list is the counted remainder of the migration. It may shrink,
/// never grow: a new payload-free decline must be typed, not appended here.
#[test]
fn the_staged_decline_backlog_only_shrinks() {
    assert!(
        STAGED.is_empty(),
        "STAGED grew to {} — a new decline type must name its reason, in the \
         variant or at its call sites, rather than join the backlog",
        STAGED.len()
    );
    for (file, name, why) in STAGED {
        assert!(
            !why.is_empty(),
            "{file}: {name} must say how many checks its Unsupported hides"
        );
    }
}

/// Pin both halves of the typed draw-error surface: an unused compatibility
/// variant is an invitation to reopen the backlog, while a constructor spelled
/// elsewhere is a live untyped refusal even if the enum declaration changes.
#[test]
fn draw_error_has_no_untyped_carrier_or_constructors() {
    let root = crate_src();
    let types = std::fs::read_to_string(root.join("backend/vulkan/engine/types.rs"))
        .expect("read Vulkan engine DrawError definition");
    assert!(
        !types.contains("Invalid(String),"),
        "the free-text DrawError carrier was reintroduced; add a typed decline instead"
    );

    let constructor = ["DrawError::", "Invalid("].concat();
    for path in rust_files(&root) {
        let src = std::fs::read_to_string(&path).expect("read Rust source");
        for (line_no, line) in src.lines().enumerate() {
            if !line.trim_start().starts_with("//") && line.contains(&constructor) {
                panic!(
                    "{}:{} constructs the deleted untyped draw error",
                    path.strip_prefix(&root).unwrap_or(&path).to_string_lossy(),
                    line_no + 1
                );
            }
        }
    }
}

/// A registry row that lists a slug the type cannot produce, or omits one it
/// can, is a census that lies. Checked against the real `slug()` for every
/// constructible variant.
#[test]
fn the_registry_lists_exactly_the_slugs_backend_error_produces() {
    use crate::backend::{BackendError, BackendKind, BackendOp};
    use crate::observe::Decline;

    const OPS: &[BackendOp] = &[
        BackendOp::WriteTexture,
        BackendOp::ReadTexture,
        BackendOp::SetPipelineLibrary,
        BackendOp::ExecuteBlit,
        BackendOp::ExecuteCompute,
        BackendOp::ExecuteRender,
        BackendOp::RenderDraw,
        BackendOp::Present,
        BackendOp::EncodeSimpleDraw,
    ];
    let mut produced: Vec<&'static str> = OPS
        .iter()
        .map(|op| BackendError::Unsupported(*op, BackendKind::Vulkan).slug())
        .collect();
    for e in [
        BackendError::InvalidArgument,
        BackendError::ResourceMissing,
        BackendError::ShaderError,
        BackendError::DeviceLost,
        BackendError::Other("x"),
    ] {
        produced.push(e.slug());
    }
    produced.sort_unstable();

    let row = REGISTRY
        .iter()
        .find(|c| c.type_name == "BackendError")
        .expect("BackendError is registered");
    let mut listed = row.slugs.to_vec();
    listed.sort_unstable();

    assert_eq!(
        produced, listed,
        "BackendError's registry row is out of date"
    );
}

/// Not every payload-free `Unsupported` is a defect, and the difference is
/// worth stating rather than assuming.
///
/// A **decline** answers "I refused your command"; it must say which check
/// refused. A **classification** answers "this is what I found"; `Unsupported`
/// is a legitimate terminal value and a payload would be meaningless. The two
/// permanent exceptions below are the latter. The staged rows are the former —
/// genuine declines awaiting migration, listed so the scan stays green while
/// the work is staged, and countable so the debt cannot quietly grow.
const PERMANENT: &[(&str, &str, &str)] = &[
    (
        "runtime/spirv_bind.rs",
        "ReflectedSampledKind",
        "a classification of what reflection reported, not a refusal — the \
         sibling variants are Kind(..) and Absent",
    ),
    (
        "backend/vulkan/caps/device_features.rs",
        "MirrorClampToEdge",
        "a capability rung: the device does not offer the feature. caps/gate \
         governs it, and the decline is named at the sampler binding site",
    ),
];

/// Genuine declines whose `Unsupported` still collapses several checks.
/// **This list may only shrink.** Each row names how many distinct checks it
/// currently hides, which is the size of the diagnostic gap.
///
/// Empty: every genuine decline now names its check, either in the variant's
/// payload or through a registered call-site vocabulary. Kept rather than deleted
/// because `the_staged_decline_backlog_only_shrinks` pins it at zero, so the
/// backlog cannot reopen without someone editing that pin.
const STAGED: &[(&str, &str, &str)] = &[];

/// A payload-free `Unsupported`-shaped variant is the defect the ground rules
/// name by example. Catching it by scan rather than by memory is what stops it
/// being reintroduced the next time an enum grows a catch-all.
#[test]
fn no_error_enum_carries_a_payload_free_unsupported() {
    // A third exemption, and unlike the two lists above it is *derived* rather
    // than asserted: a type whose registry row names `slug_calls` writes its
    // reason at the construction sites, and `every_row_lists_exactly_the_slugs_
    // its_type_writes` checks that the row and those sites agree. So the crate
    // can already say which check refused — provably, for all 177 of
    // `BlitStatus`'s — and a payload would only duplicate the channel.
    //
    // Deriving it matters: a hand-written exemption would also excuse a type that
    // *stopped* naming its reasons, whereas this one evaporates the moment the
    // row loses its `slug_calls`.
    let names_its_reasons_at_call_sites = |name: &str| {
        REGISTRY
            .iter()
            .any(|c| c.type_name == name && !c.slug_calls.is_empty())
    };
    let allowed = |rel: &str, name: &str| {
        PERMANENT.iter().any(|(f, e, _)| *f == rel && *e == name)
            || STAGED.iter().any(|(f, e, _)| *f == rel && *e == name)
            || names_its_reasons_at_call_sites(name)
    };
    let mut bare: Vec<String> = Vec::new();
    for path in rust_files(&crate_src()) {
        let rel = path
            .strip_prefix(crate_src())
            .unwrap_or(&path)
            .to_string_lossy()
            .replace('\\', "/");
        let Ok(src) = std::fs::read_to_string(&path) else {
            continue;
        };
        let mut current_enum: Option<String> = None;
        for line in src.lines() {
            let t = line.trim();
            if let Some(rest) = t
                .strip_prefix("pub enum ")
                .or_else(|| t.strip_prefix("enum "))
            {
                current_enum = Some(
                    rest.split(|c: char| !(c.is_alphanumeric() || c == '_'))
                        .next()
                        .unwrap_or("")
                        .to_string(),
                );
                continue;
            }
            if t == "}" {
                current_enum = None;
                continue;
            }
            let Some(name) = current_enum.as_deref() else {
                continue;
            };
            // A bare `Unsupported,` inside an enum body: no payload to say which
            // check refused.
            if t == "Unsupported," && !allowed(&rel, name) {
                bare.push(format!("{rel}: {name}::Unsupported carries no reason"));
            }
        }
    }
    assert!(
        bare.is_empty(),
        "a payload-free Unsupported cannot say which check refused — give it a \
         reason type, as BackendError and DrawError have:\n  {}",
        bare.join("\n  ")
    );
}

/// The scanner must actually walk the crate; a broken walk would make every
/// scan above vacuously green.
#[test]
fn the_scanner_walks_the_whole_crate() {
    let files = rust_files(&crate_src());
    assert!(
        files.len() > 100,
        "expected the crate's full file list, found {}",
        files.len()
    );
    for expect in [
        "observe/decline.rs",
        "backend/mod.rs",
        "runtime/drain/mod.rs",
        "backend/vulkan/engine/pools/images_and_registry.rs",
    ] {
        assert!(
            files.iter().any(|p| p.ends_with(expect)),
            "scanner never reached {expect}"
        );
    }
}
