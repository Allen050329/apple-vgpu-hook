//! Static gates over the decline vocabulary and the support-matrix boundary.
//!
//! Each one reads the crate's own source, asserts a property, and names the
//! defect in the failure message. Scanning source is crude, but it fails at
//! `cargo test` time on the machine that made the change — which is the only
//! place these fixes are cheap. The alternative for every property here is
//! noticing at runtime, on a specific host, in a specific frame.
//!
//! # What used to be here, and why it is not
//!
//! Eleven of these tests checked a 2 700-line `#[cfg(test)]` `REGISTRY` in
//! `super::decline` — a hand-maintained table restating, for each of 67 decline
//! types, its defining file, its emission site and all 1 425 of its slugs. The
//! table was a copy of the `slug()` arms, so it could only ever agree or
//! disagree with them; agreeing added no invariant the arms did not already
//! carry, and disagreeing was reported as "the registry drifted" rather than as
//! a defect in the code. Meanwhile every deletion cost a second edit plus a
//! hand-bumped `(types, slugs)` baseline carrying forty lines of changelog prose,
//! which is what made shrinking this crate expensive.
//!
//! The one property that is genuinely crate-wide — **no two checks share a
//! slug**, which no single impl can see — is now read straight off the
//! `Decline`/`Refusal` impls by
//! [`no_two_declines_share_a_slug`]. It is a scan of the code rather than of a
//! copy of the code, so it cannot drift from it, and it needs no baseline.

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

/// [`rust_files`] minus the files that are **entirely** test code.
///
/// [`production_source`] strips `mod tests { … }` *blocks*, which is the whole
/// story for a file that mixes the two. It is no story at all for
/// `runtime/*/tests.rs`, four of which exist here: each is declared
/// `#[cfg(test)] mod tests;` by its parent, so nothing inside the file itself
/// marks it, and every scan that called `production_source` on one was reading
/// a test fixture as shipped code.
///
/// Found by the unbounded-raw-GVA-write gate, which flagged
/// `runtime/compute_exec/tests.rs` as an unjustified writer on its first run.
/// The test module is *declared* by the parent, so that is where this looks —
/// a filename rule would be a guess, and would also miss a test-only module
/// named anything else.
fn production_files(root: &Path) -> Vec<PathBuf> {
    rust_files(root)
        .into_iter()
        .filter(|p| !declared_cfg_test(p))
        .collect()
}

/// Whether this file's own module declaration in its parent is `#[cfg(test)]`.
fn declared_cfg_test(path: &Path) -> bool {
    let Some(stem) = path.file_stem().map(|s| s.to_string_lossy().into_owned()) else {
        return false;
    };
    // `dir/name.rs` is declared by `dir/mod.rs`; `dir/mod.rs` is declared by the
    // grandparent as `mod dir;`, which is not a case this needs.
    let Some(parent) = path.parent() else {
        return false;
    };
    let Ok(src) = std::fs::read_to_string(parent.join("mod.rs")) else {
        return false;
    };
    let decl = format!("mod {stem};");
    let mut cfg_test_pending = false;
    for line in src.lines() {
        let t = line.trim();
        if t == decl {
            return cfg_test_pending;
        }
        // Attributes stack, and anything else between resets the run.
        if t.starts_with("#[") {
            cfg_test_pending |= t == "#[cfg(test)]";
        } else if !t.is_empty() && !t.starts_with("//") {
            cfg_test_pending = false;
        }
    }
    false
}

/// Repo-relative path with forward slashes, for stable messages.
fn rel(path: &Path, root: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

/// `src` with every comment and every ordinary string/char literal replaced by
/// spaces, byte-for-byte, so offsets and line numbers survive the mask.
///
/// Shared by every source gate in this file, because a gate that greps raw
/// source cannot tell code from prose: a doc comment quoting the shape it
/// forbids, or a fail-log format string naming the very symbol under test, both
/// read as source hits. Raw strings with any hash count, escaped string
/// literals, `'x'`/`'\''` char literals and nested block comments are all
/// handled here rather than in each caller — hand-rolling a second scrubber is
/// how a sweep silently starts reporting live code as dead.
fn mask_comments_and_literals(src: &str) -> Vec<u8> {
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
    masked
}

/// Byte offsets of `Result<Success, String>` spellings in Rust code.
///
/// This is intentionally a small lexical scan rather than a same-line grep:
/// rustfmt can wrap either generic argument, and nested generic `>` tokens must
/// not be mistaken for the outer result. Comments and ordinary string/char
/// literals are masked so the gate can explain the forbidden shape and test
/// itself without creating a false source hit.
fn result_string_error_offsets(src: &str) -> Vec<usize> {
    let masked = mask_comments_and_literals(src);
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

/// `VK_EXT_external_memory_host` must never be asked for.
///
/// Importing a host pointer over guest RAM gives the host GPU write access to
/// the guest VM's memory. That is a property of the mechanism, not of how much
/// of it is used, so the bound is "never requested" rather than any budget.
///
/// This is a **source** assertion and not a behavioural one, and it is here
/// because the behavioural form cannot be built: what the flip changes is what
/// `DeviceContext::create` asks the driver for, and every fixture that reads
/// that answer needs `instance.create_device` to succeed — which on a driverless
/// host degenerates into a skip, i.e. a green summary that is produced whether
/// or not the code is right. A source gate has no such arm.
///
/// The needles are the whole API surface of the mechanism, not just the request.
/// The name constant is how a device *asks* for it
/// (`has_device_extension(…NAME)`, `enabled_device_extensions.push(…NAME…)`);
/// the loader type, the two import structs, the properties entry point and the
/// `HOST_ALLOCATION_EXT` handle type are how it would then be *used*. A
/// reintroduction has to name at least one of them, and matching all six means
/// the gate does not depend on which end someone starts from.
///
/// Earlier this gate matched only the two name constants, on the reasoning that
/// the loader type and the `ext_external_memory_host` field should stay in the
/// tree as hard-`None` decline sites. That subsystem is deleted now — the
/// resolver, its window budget, the scatter pool and both present entry points
/// went with it — so nothing legitimate names any of these, and the narrower
/// gate would no longer notice a whole rail coming back.
///
/// Comments and string literals are masked, so the paragraph above is not a hit.
/// Whitespace is folded so a rustfmt wrap inside a path cannot hide a request.
#[test]
fn the_host_pointer_import_extension_is_never_requested() {
    let root = crate_src();
    let mut hits = Vec::new();
    for path in rust_files(&root) {
        let src = std::fs::read_to_string(&path).expect("read Rust source");
        let masked = mask_comments_and_literals(&src);
        let folded: String = masked
            .iter()
            .copied()
            .filter(|byte| !byte.is_ascii_whitespace())
            .map(char::from)
            .collect();
        for needle in [
            "external_memory_host::NAME",
            "EXT_EXTERNAL_MEMORY_HOST_NAME",
            "ash::ext::external_memory_host",
            "ImportMemoryHostPointerInfoEXT",
            "get_memory_host_pointer_properties_ext",
            "ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT",
        ] {
            if folded.contains(needle) {
                hits.push(format!(
                    "{} names {needle}",
                    path.strip_prefix(&root).unwrap_or(&path).to_string_lossy()
                ));
            }
        }
    }
    assert!(
        hits.is_empty(),
        "VK_EXT_external_memory_host is requested; the GPU must not be able to \
         write guest RAM:\n  {}",
        hits.join("\n  ")
    );
}

/// Metal has exactly one no-copy buffer constructor, and it takes host bytes.
///
/// `newBufferWithBytesNoCopy` is the Metal half of the same hazard the Vulkan
/// gate above covers: it hands the GPU a pointer, and if that pointer is a
/// `mach_vm_remap` view of the guest's pages then the host GPU can read and
/// write guest RAM. It is not banned outright, because the crate legitimately
/// aliases its **own** allocations with it — the CPU-staged vertex, fragment
/// and compute byte vectors in `new_buffer_from_host`, which are `Vec<u8>`s
/// this process owns.
///
/// So the invariant is not "never call it", it is "call it in exactly one
/// place, whose argument is provably host-owned". A second call site is the
/// thing to look at, whatever it claims to be aliasing. The one that used to
/// exist took `MappingEntry::contig_ptr` and became a linear texture the guest
/// surface was rendered into.
///
/// This is a **source** assertion for the same reason as the Vulkan gate, and
/// one more: `backend-metal` is Apple-only and does not compile on a Linux
/// host at all, so nothing else in this tree — not the compiler, not clippy,
/// not the feature matrix — reads that code on the machine most of this work
/// happens on. Comments and string literals are masked, so this paragraph is
/// not a hit.
#[test]
fn metal_no_copy_buffers_alias_host_memory_and_nothing_else() {
    const OWNER: &str = "backend/metal/runtime.rs";
    let root = crate_src();
    let mut sites = Vec::new();
    for path in rust_files(&root) {
        let src = std::fs::read_to_string(&path).expect("read Rust source");
        let masked = mask_comments_and_literals(&src);
        let folded: String = masked
            .iter()
            .copied()
            .filter(|byte| !byte.is_ascii_whitespace())
            .map(char::from)
            .collect();
        let count = folded.matches("new_buffer_with_bytes_no_copy").count();
        if count > 0 {
            sites.push((
                path.strip_prefix(&root)
                    .unwrap_or(&path)
                    .to_string_lossy()
                    .replace('\\', "/"),
                count,
            ));
        }
    }
    assert_eq!(
        sites,
        vec![(OWNER.to_string(), 1)],
        "newBufferWithBytesNoCopy must appear exactly once, in {OWNER}'s \
         new_buffer_from_host over host-owned bytes; a second site is a \
         candidate guest-RAM alias"
    );
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
    // Every literal in a result position, with no charset filter: anchoring on the
    // `slug()`/`refusal()` body is what makes a literal here vocabulary, and a slug
    // that is *not* log-safe is a defect `every_declared_slug_is_log_safe` must be
    // able to see rather than one this extractor should quietly drop.
    let push_if_slug = |out: &mut Vec<String>, lit: &str| {
        if !lit.is_empty() {
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

/// Every `(slug, owning type)` pair the crate's `Decline`/`Refusal` impls write.
///
/// Read off the impls themselves rather than off a table naming them, so a new
/// variant becomes visible here the moment its `slug()` arm is written. Rust
/// cannot iterate an enum's variants, but the exhaustive `match` inside `slug()`
/// is the one place the compiler *does* force completeness: add a variant and
/// the compiler makes you write an arm, write an arm and this scan reads it.
///
/// Anchored on the trait impl and then on the `fn slug` / `fn refusal` body
/// inside it, because `fields()` lives in the same impl and its keys are
/// lowercase snake_case too — reading the whole impl would count field keys as
/// vocabulary.
fn declared_slugs() -> Vec<(String, String)> {
    let root = crate_src();
    let mut out = Vec::new();
    for path in production_files(&root) {
        let Ok(raw) = std::fs::read_to_string(&path) else {
            continue;
        };
        // Comments are dropped so a doc comment demonstrating an impl cannot
        // register vocabulary; `production_source` also hides `#[cfg(test)]`
        // modules, whose fixture impls are not slugs the crate can write.
        let src = production_source(&raw);
        let rel = rel(&path, &root);
        for trait_name in ["Decline for ", "Refusal for "] {
            let mut from = 0usize;
            while let Some(at) = src[from..].find(trait_name) {
                let start = from + at;
                let Some(block) = block_after(&src[start..], "{") else {
                    from = start + trait_name.len();
                    continue;
                };
                let ty = src[start + trait_name.len()..]
                    .split(|c: char| !(c.is_alphanumeric() || c == '_'))
                    .next()
                    .unwrap_or("")
                    .to_string();
                for anchor in ["fn slug", "fn refusal"] {
                    if let Some(body) = block_after(&block, anchor) {
                        for slug in slugs_returned_by(&body) {
                            out.push((slug, format!("{rel}: {ty}")));
                        }
                    }
                }
                from = start + trait_name.len() + block.len();
            }
        }
    }
    out
}

/// Two checks sharing a slug is the exact failure `AGENTS.md` names: you grep the
/// fail log, watch the slug fire, and still cannot tell which of the two refused.
///
/// This is the property that needs a crate-wide scan. Uniqueness *within* an enum
/// is visible in the `match` and gets caught by reading it; a `translate` slug
/// colliding with an `engine` slug is visible from neither impl, and the
/// per-enum tests this replaced could not have caught it.
///
/// **What this covers, exactly.** The slugs written as literals in a
/// `slug()`/`refusal()` result position. It does *not* cover the vocabularies
/// written at *construction* sites — `br(BlitStatus::Bounds, "fill_out_of_range")`
/// and `FenceStatus::Unsupported("fence_domain_unknown")` — where the reason rides
/// in the value rather than in a `match` arm. The registry this replaced listed
/// those too, at the cost of a second hand-maintained table of `(file, call)` pairs
/// per type; a collision inside one of those rails is not caught here. Stated so
/// the number is not read as the crate's whole refusal surface.
#[test]
fn no_two_declines_share_a_slug() {
    let declared = declared_slugs();
    // The scan must actually reach the vocabulary, or the assertion below is
    // vacuously green — the failure mode every sweep in this repo has had.
    assert!(
        declared.len() > 400,
        "the slug scan found only {} slugs; it is not reading the impls",
        declared.len()
    );

    let mut owner: BTreeMap<&str, &str> = BTreeMap::new();
    let mut clashes = Vec::new();
    for (slug, who) in &declared {
        if let Some(prev) = owner.insert(slug, who) {
            if prev != who {
                clashes.push(format!("`{slug}` claimed by both {prev} and {who}"));
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
/// `caps`, `translate` and census slugs.
#[test]
fn every_declared_slug_is_log_safe() {
    let mut bad = Vec::new();
    for (slug, who) in declared_slugs() {
        if slug.is_empty()
            || !slug
                .bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')
        {
            bad.push(format!("{who}: {slug:?}"));
        }
    }
    assert!(
        bad.is_empty(),
        "a slug must be lowercase snake_case to survive a grep of the log:\n  {}",
        bad.join("\n  ")
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

/// `production_source` is what keeps a doc comment's example and a test module's
/// fixture out of the vocabulary. Both exclusions are load-bearing and neither is
/// visible from a passing scan, so they are checked directly.
///
/// The `#[cfg(test)]` *helper* partway down is the trap: cutting at the first
/// `#[cfg(test)]` rather than at the test module hid 2 450 of `blit_exec.rs`'s
/// lines, and with them every slug the file writes.
#[test]
fn production_source_hides_prose_and_test_modules_but_not_what_follows_them() {
    let noise = r#"
/// Refuses with `Self::A => "doc_example_only"`.
impl Decline for A {
    fn slug(&self) -> &'static str {
        match self {
            Self::A => "real_reason",
        }
    }
}

#[cfg(test)]
fn reset_dedup_for_test() {}

impl Decline for B {
    fn slug(&self) -> &'static str {
        match self {
            Self::B => "reason_below_the_helper",
        }
    }
}

#[cfg(test)]
mod tests {
    impl Decline for T {
        fn slug(&self) -> &'static str {
            match self {
                Self::T => "test_only_reason",
            }
        }
    }
}

impl Decline for C {
    fn slug(&self) -> &'static str {
        match self {
            Self::C => "reason_below_the_test_module",
        }
    }
}
"#;
    let kept = production_source(noise);
    assert_eq!(
        slugs_returned_by(&kept),
        vec![
            "real_reason",
            "reason_below_the_helper",
            "reason_below_the_test_module"
        ],
        "neither a doc comment, a cfg(test) helper nor a mid-file test module may \
         hide or contribute vocabulary"
    );
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
    (
        "runtime/blit_exec.rs",
        "BlitStatus",
        "the reason lives at the construction site, not in the variant: all 177 \
         of this rail's refusals are written `br(BlitStatus::Unsupported, \
         \"slug\")`, so a payload would only duplicate that channel",
    ),
];

/// A payload-free `Unsupported`-shaped variant is the defect the ground rules
/// name by example. Catching it by scan rather than by memory is what stops it
/// being reintroduced the next time an enum grows a catch-all.
#[test]
fn no_error_enum_carries_a_payload_free_unsupported() {
    let allowed = |rel: &str, name: &str| {
        PERMANENT
            .iter()
            .any(|(file, enum_name, _)| *file == rel && *enum_name == name)
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
         reason type, as DrawError does:\n  {}",
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

/// The production filter must actually drop the test-only files, and must not
/// drop anything else.
///
/// Both directions matter and they fail in opposite ways. If
/// [`declared_cfg_test`] silently returned `false` — a renamed `mod.rs`, an
/// attribute written differently — the filter would be vacuous and every gate
/// would go back to reading `runtime/*/tests.rs` as shipped code, which is the
/// bug it was added for. If it returned `true` too readily it would hide real
/// production files from every scan above, which is the direction that reads as
/// a pass.
#[test]
fn the_production_filter_drops_test_only_files_and_nothing_else() {
    let root = crate_src();
    let all: Vec<String> = rust_files(&root).iter().map(|p| rel(p, &root)).collect();
    let production: Vec<String> = production_files(&root)
        .iter()
        .map(|p| rel(p, &root))
        .collect();

    // Eight files in this crate are entirely test code, declared `#[cfg(test)]
    // mod x;` by their parent. Every one was being read as shipped source by
    // the three scans above — including this file, which is itself
    // `#[cfg(test)] mod gate;` and whose tables name the very symbols those
    // scans forbid.
    let mut dropped: Vec<&String> = all.iter().filter(|p| !production.contains(p)).collect();
    dropped.sort();
    assert_eq!(
        dropped,
        [
            "backend/vulkan/caps/gate.rs",
            "backend/vulkan/translate/coverage.rs",
            "backend/vulkan/translate/gate.rs",
            "observe/gate.rs",
            "runtime/compute_exec/tests.rs",
            "runtime/drain/tests.rs",
            "runtime/icb/tests.rs",
            "runtime/metal_draw/tests.rs",
        ]
        .iter()
        .collect::<Vec<_>>(),
        "the set of test-only files changed; a new one must be dropped from the \
         production scans and a removed one must not be listed here"
    );
    // And the production files those live beside are still scanned.
    for expect in ["runtime/drain/mod.rs", "runtime/mipmap.rs", "observe/mod.rs"] {
        assert!(
            production.iter().any(|p| p == expect),
            "the filter dropped production file {expect}"
        );
    }
}

/// A row exempting a type must name a file that exists and say why. An exemption
/// pointing at a moved or deleted file excuses nothing and hides the next one.
#[test]
fn every_permanent_exemption_names_a_live_file_and_a_reason() {
    for (file, enum_name, why) in PERMANENT {
        assert!(
            !why.is_empty(),
            "{file}: {enum_name} must say why it is exempt"
        );
        let src = std::fs::read_to_string(crate_src().join(file))
            .unwrap_or_else(|e| panic!("{file}: {e}"));
        assert!(
            src.contains(enum_name),
            "{file} no longer defines {enum_name}; drop the exemption"
        );
    }
}

/// Every way production code gets a host pointer that aliases guest RAM, so the
/// scan below cannot be fooled by a file that writes guest memory without ever
/// naming `map_pages`.
///
/// The first cut of this gate listed only `map_pages` callers and **had exactly
/// that hole**. `runtime/mapping_write.rs` takes its pointer from
/// `mapper::ensure_contig_view` through two local wrappers and pokes BGRA rows
/// straight into it — the largest guest-write rail in the device — and the gate
/// scored the file as having no `map_pages` call at all, which is true and
/// irrelevant. It passed, and the footprint was missing that whole rail.
///
/// So the needle is the *pointer*, not one of its sources. Anything that hands
/// back a writable alias belongs here.
const GUEST_RAM_POINTER_SOURCES: &[&str] = &[
    ".map_pages(",
    "ensure_contig_view(",
    "map_fresh_span(",
    "map_fresh_span_within(",
    "contig_for_span(",
    "contig_for_write(",
];

/// Every production site that obtains a host pointer over guest RAM, and whether
/// it writes through it.
///
/// A host pointer over guest pages is one of the two ways this device can put
/// bytes in the guest — the other being `HostMemory::write_gpa`, which has a
/// single production implementation and is marked there. Every site here that
/// writes must also mark `observe::footprint`, or that write's frames are
/// missing from the set a guest panic is scored against. **The missing mark is
/// the dangerous direction**: it produces a "this device never wrote that page"
/// that is false, which is an exoneration nobody can tell from a real one.
///
/// So the classification is the point of the row, and the count is what stops a
/// new call appearing inside an already-listed file without anyone deciding
/// which kind it is.
/// How a site's writes reach `observe::footprint`, or why they need not.
#[derive(PartialEq, Eq, Debug, Clone, Copy)]
enum Marks {
    /// Writes guest RAM and marks the footprint in this file.
    Here,
    /// Writes guest RAM, but the pointer source it calls marks on its behalf.
    /// Correct and preferable — one marking site serves every caller — so this
    /// file must *not* mark again or the frames are counted twice.
    BySource,
    /// Takes a guest-RAM pointer only to copy out of it.
    ReadOnly,
}

const MAP_PAGES_SITES: &[(&str, usize, Marks, &str)] = &[
    (
        "runtime/gpa_map.rs",
        1,
        Marks::Here,
        "control-plane writes (stamp, DeviceInfo, display shared, child HEAD) \
         map the covering pages and poke bytes; marked on the exact byte range \
         after the copy",
    ),
    (
        "runtime/gva_view.rs",
        5,
        Marks::Here,
        "the raw-GVA rails, and the pointer source for the two files below. \
         `write_span_multi` marks each packed run's exact destination; \
         `map_fresh_span_within` marks the span it is about to hand to a caller \
         that writes through the pointer. The read paths (`read_span_multi`, \
         `host_ptr_for_span`) reach guest RAM only to copy out of it",
    ),
    (
        "runtime/mapper.rs",
        5,
        Marks::Here,
        "the mapping-keyed rails. `write_mapping_bytes` marks through the \
         mapping's own page list — a scatter, so never over the span's hull — on \
         both the contiguous-view fast path and the per-run slow path",
    ),
    (
        "runtime/mapping_write.rs",
        8,
        Marks::Here,
        "the BGRA row writers, and the largest guest-write rail in the device. \
         They take a contig view through `contig_for_write` and poke rows \
         straight into it, reaching `mapper::write_mapping_bytes` not at all — \
         which is exactly how the first cut of this gate missed them. \
         `contig_for_write` marks for all of them",
    ),
    (
        "runtime/metal_draw/mod.rs",
        4,
        Marks::BySource,
        "`write_gva_rgba8_within` and its peer write rows through a `FreshSpan`; \
         `gva_view::map_fresh_span_within` marks the span when it resolves it",
    ),
    (
        "runtime/compute_exec/mod.rs",
        2,
        Marks::BySource,
        "`write_linear_texture_bulk` writes rows through a `FreshSpan`; marked by \
         `gva_view::map_fresh_span_within` as above",
    ),
    (
        "runtime/metal_draw/vulkan.rs",
        3,
        Marks::ReadOnly,
        "`task_gva_guest_runs`, `try_type11_sample_zero_copy` and \
         `try_type5_sample_zero_copy` build `engine::GuestRun` spans the engine \
         reads *out of* — vertex, storage and sampled sources uploaded to the \
         GPU. Nothing writes back through them, and the GPU cannot: \
         `the_host_pointer_import_extension_is_never_requested` holds that the \
         one extension which would let it is never requested",
    ),
    (
        "runtime/scanout.rs",
        1,
        Marks::ReadOnly,
        "screen capture reads the scanout surface out of its contig view; it \
         puts nothing back",
    ),
];

/// A `map_pages` caller that writes guest RAM must record where.
///
/// See [`MAP_PAGES_SITES`]. This is the mechanism behind the completeness claim
/// `observe::footprint` makes: the footprint can only be trusted as evidence
/// about a guest panic if the set of rails feeding it is closed, and "we
/// remembered to hook the new one" is not a mechanism.
#[test]
fn every_map_pages_caller_is_classified_and_the_writers_mark_the_footprint() {
    let root = crate_src();
    let mut found: std::collections::BTreeMap<String, usize> = Default::default();
    let mut marks: std::collections::BTreeSet<String> = Default::default();
    for path in production_files(&root) {
        let src = std::fs::read_to_string(&path).expect("read Rust source");
        let production = production_source(&src);
        let masked = mask_comments_and_literals(&production);
        let text: String = masked.iter().copied().map(char::from).collect();
        let rel_path = rel(&path, &root);
        for line in text.lines() {
            // A definition is not a caller: `fn ensure_contig_view(` would
            // otherwise score the function that *is* the pointer source as a
            // site that consumes one.
            let is_definition = line.trim_start().starts_with("fn ")
                || line.trim_start().starts_with("pub fn ")
                || line.trim_start().starts_with("pub(crate) fn ");
            if !is_definition
                && GUEST_RAM_POINTER_SOURCES
                    .iter()
                    .any(|needle| line.contains(needle))
            {
                *found.entry(rel_path.clone()).or_default() += 1;
            }
            // `note_mapping_write_footprint` is the mapper's helper that resolves
            // a write's frames through a mapping's scatter page list and marks
            // them. It is a mark, so a file calling it is a marking file; leaving
            // it off this list would fail a rail that does record its writes.
            if !is_definition
                && (line.contains("footprint::note_written_range(")
                    || line.contains("footprint::note_written_pages(")
                    || line.contains("note_mapping_write_footprint("))
            {
                marks.insert(rel_path.clone());
            }
        }
    }

    let expected: std::collections::BTreeMap<String, usize> = MAP_PAGES_SITES
        .iter()
        .map(|(file, n, _, _)| ((*file).to_string(), *n))
        .collect();
    assert_eq!(
        found, expected,
        "the set of `map_pages` callers changed. Each one aliases guest RAM \
         writably, so a new entry has to be classified in MAP_PAGES_SITES as a \
         writer (and then mark observe::footprint) or as a reader (and say why \
         nothing writes back through it)."
    );

    for (file, _, how, why) in MAP_PAGES_SITES {
        assert!(!why.is_empty(), "{file}: classify it, do not just list it");
        assert!(
            root.join(file).exists(),
            "{file} no longer exists; drop the row"
        );
        assert_eq!(
            marks.contains(*file),
            *how == Marks::Here,
            "{file}: classified {how:?}, but the file {} mark observe::footprint. \
             A writing rail that marks nowhere leaves its frames out of the set, \
             and the resulting `pn` miss reads exactly like a real exoneration; a \
             `BySource` rail that also marks here counts its frames twice.",
            if marks.contains(*file) {
                "does"
            } else {
                "does not"
            }
        );
    }
}

/// The other funnel: `HostMemory::write_gpa` reaches guest RAM without
/// `map_pages` at all, so the footprint has to be marked in the one production
/// implementation of it.
///
/// `FakeHost`'s implementation deliberately does not mark, for the reason its
/// [`MAP_PAGES_SITES`] row gives, so this asserts the QEMU side specifically
/// rather than counting implementations.
#[test]
fn the_real_write_gpa_marks_the_footprint() {
    let src = std::fs::read_to_string(crate_src().join("qemu/host_ops.rs"))
        .expect("read the QEMU host shim");
    let production = production_source(&src);
    let masked = mask_comments_and_literals(&production);
    let text: String = masked.iter().copied().map(char::from).collect();
    let body = text
        .split("fn write_gpa(")
        .nth(1)
        .expect("QemuHost still implements write_gpa");
    // Bounded to the function: the next `fn ` starts the following method, and
    // a mark that had drifted out of `write_gpa` into a neighbour would
    // otherwise satisfy a whole-file search while recording nothing.
    let body = body.split("\n    fn ").next().unwrap_or(body);
    assert!(
        body.contains("footprint::note_written_range("),
        "QemuHost::write_gpa must record the frames it writes. It is one of the \
         two ways this device reaches guest RAM, and the whole of the \
         control-plane traffic goes through it."
    );
}

/// Product raw-GVA writes that are deliberately **not** bounded to an armed
/// page set, each with the authorisation that makes it sound.
///
/// `write_task_gva_product_within(.., allowed)` restricts a write to the guest
/// pages a deferred window was armed on. The bare `write_task_gva_product` has
/// no such bound, so every use of it needs a different reason to be writing
/// where it is — and the reason is always the same shape here: the write is
/// **synchronous with the command that named the address**, so the guest cannot
/// have freed it in between. A deferred rail cannot say that, which is exactly
/// why it carries an armed set.
///
/// The count is part of the row so this debt cannot grow silently: a new
/// unbounded write in one of these files fails the test even though the file is
/// already listed.
const UNBOUNDED_RAW_GVA_WRITES: &[(&str, usize, &str)] = &[
    (
        "runtime/drain/mod.rs",
        3,
        "GetComputeInfo and the texture-requirement query write their reply into \
         the `reply_gva` the request packet itself carries, while that packet is \
         being executed. There is no interval in which the guest could have \
         handed the address to something else",
    ),
    (
        "runtime/mipmap.rs",
        1,
        "mipmap generation writes rows into the destination texture's own GVA \
         during the command that asked for them; the texture is live for the \
         duration of its own command",
    ),
    (
        "runtime/blit_exec.rs",
        1,
        "the type-2 staging blit writes destination rows at the GVA the blit \
         command names, synchronously inside that blit. Every other write on \
         this rail IS bounded (`_within(.., allowed)`), which is what makes this \
         one worth naming rather than assuming",
    ),
];

/// A raw-GVA write with no armed page set has to be a deliberate, named choice.
///
/// This is the write-after-free class's blast radius, so the question "which
/// writes can reach guest RAM without a page set bounding them?" must have an
/// answer that survives review. It has been re-derived by hand in at least
/// three sessions — each time reaching the same two or three sites, each time
/// from scratch, and one handoff ends with "don't re-chase these", which is not
/// a mechanism. This is the mechanism.
///
/// Bounded writes go through `write_task_gva_product_within` with an `allowed`
/// set and are not counted here; the bare call is. Adding one fails this test
/// until it is listed with a reason, and removing one fails it too, so the
/// table cannot drift out of date in either direction.
#[test]
fn every_unbounded_raw_gva_write_is_named_and_justified() {
    let root = crate_src();
    let mut found: std::collections::BTreeMap<String, usize> = Default::default();
    for path in production_files(&root) {
        let src = std::fs::read_to_string(&path).expect("read Rust source");
        // Test modules are stripped: a fixture writing through the product
        // helper is exercising it, not shipping an unbounded write.
        let production = production_source(&src);
        let masked = mask_comments_and_literals(&production);
        let text: String = masked.iter().copied().map(char::from).collect();
        let rel_path = rel(&path, &root);
        for line in text.lines() {
            // The open paren immediately after the name is what separates the
            // unbounded call from `write_task_gva_product_within(`, which is a
            // strict prefix of it and would otherwise be counted as unbounded —
            // scoring the bounded rail as the hazard.
            if !line.contains("write_task_gva_product(") {
                continue;
            }
            // Its own definition and the one-line forwarder that supplies
            // `None` are the implementation, not a caller.
            if rel_path == "runtime/gva_mem.rs" {
                continue;
            }
            *found.entry(rel_path.clone()).or_default() += 1;
        }
    }

    let expected: std::collections::BTreeMap<String, usize> = UNBOUNDED_RAW_GVA_WRITES
        .iter()
        .map(|(file, n, _)| ((*file).to_string(), *n))
        .collect();
    assert_eq!(
        found, expected,
        "unbounded raw-GVA writes changed. Every one of these can write guest \
         RAM with no armed page set bounding it, so a new entry needs a stated \
         authorisation in UNBOUNDED_RAW_GVA_WRITES (and a removed one needs the \
         row dropped). Bounded writes use write_task_gva_product_within(.., \
         allowed) and are not counted."
    );
    for (file, _, why) in UNBOUNDED_RAW_GVA_WRITES {
        assert!(!why.is_empty(), "{file}: an unbounded write must say why");
        assert!(
            root.join(file).exists(),
            "{file} no longer exists; drop the row"
        );
    }
}

/// The page-drift witness has exactly one production caller, so the policy and
/// its control knob cannot be bypassed by adding a rail.
///
/// `mapper::type4_pages_witness` answers *whether* a mapping's cached page
/// list still names the guest memory it was walked from — and, since it reports
/// `Unwitnessed` apart from `Verified`, whether it was in a position to know.
/// `mapper::mapping_pages_verdict` decides what the device does about the
/// answer: count it, consult `REIMS_VGPU_MAPPING_PAGE_GUARD_OFF`, and — when it
/// refuses — invalidate the list rather than skip one write. Those are separable
/// and a caller that reaches past the second to the first gets the question
/// without any of the answer.
///
/// That is not hypothetical. The witness shipped in `1b6e423` with one caller,
/// the deferred render flush, while the four direct writers in `mapping_write`
/// wrote through the same `page_entries` unchecked; the crash class this crate
/// is chasing lived in that gap for a release. Reviewing "did the new rail
/// remember to call the check?" is exactly the question nobody asked, so it is
/// asked here instead, once, by the compiler's test runner.
///
/// A new rail that genuinely needs the raw witness should call
/// `mapping_pages_verdict` and match on the outcome. If it truly cannot, this
/// test is the place to record why.
#[test]
fn the_page_drift_witness_is_only_consulted_through_the_policy() {
    let root = crate_src();
    let mut callers = Vec::new();
    for path in production_files(&root) {
        let src = std::fs::read_to_string(&path).expect("read Rust source");
        let production = production_source(&src);
        let masked = mask_comments_and_literals(&production);
        let text: String = masked.iter().copied().map(char::from).collect();
        for (index, line) in text.lines().enumerate() {
            if !line.contains("type4_pages_witness") {
                continue;
            }
            // Its own definition, and the one function allowed to ask it.
            if line.contains("pub fn type4_pages_witness") {
                continue;
            }
            callers.push(format!("{}:{}", rel(&path, &root), index + 1));
        }
    }
    assert_eq!(
        callers.len(),
        1,
        "the page-drift witness must be reached only through \
         mapper::mapping_pages_verdict, which is what counts the outcome, \
         consults REIMS_VGPU_MAPPING_PAGE_GUARD_OFF and invalidates a \
         contradicted list. Callers found:\n  {}",
        callers.join("\n  ")
    );
}
