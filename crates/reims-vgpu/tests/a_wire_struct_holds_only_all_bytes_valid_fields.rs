//! The one wire invariant whose violation is undefined behaviour is the one
//! nothing checked.
//!
//! `reims-vgpu-wire`'s `Wire` trait is `unsafe` and asks implementors for two
//! things. The second — align-1 — is enforced by the compiler:
//! `Wire::ASSERT_ALIGN_1` is an associated `const` every constructor forces, so
//! an over-aligned field fails the build. The first is not enforced by
//! anything:
//!
//! > **Every** byte pattern of `size_of::<Self>()` bytes is a valid `Self`.
//! > Integers and arrays of them qualify. `bool`, `char`, references, and
//! > `#[repr(u32)]` enums do **not** — an out-of-range guest value would be an
//! > invalid value, which is undefined behaviour rather than a decode error.
//!
//! That crate's `AGENTS.md` states it as invariant 4 and adds `NonZero*` and
//! raw pointers to the list. It was prose, held by review, across **135**
//! `unsafe impl Wire` lines — and the payoff for getting one wrong is not a
//! wrong pixel or a dropped record. `view` casts the guest's bytes straight to
//! `&T`; a `bool` field whose byte is `2`, or a `#[repr(u32)]` enum whose word
//! is not a declared discriminant, is instant UB from a buffer the guest wrote.
//! It is the same rule `backend::metal::mtl_enum` exists to keep on the Metal
//! side, where it *is* mechanised.
//!
//! So this reads the source instead. For every `unsafe impl Wire for T`, `T`'s
//! declaration is found in the same file and every field type is checked
//! against what may legally appear:
//!
//! - a `crate::le` scalar — and the list of those is **read from the macro
//!   invocations that generate them**, not written out here, so an `I32le`
//!   added upstream is admissible the moment it exists and a scalar deleted
//!   stops being admissible without anyone editing this test;
//! - `u8` or `i8`, the only primitives that are align-1 and total;
//! - an array of anything admissible;
//! - another `Wire` type, which this test audits by the same rule.
//!
//! Anything else fails, and the message says so rather than guessing what the
//! author meant. That default is the point: a type this test has never heard of
//! is exactly the case where "probably fine" is how a `bool` gets in.
//!
//! Same file, deliberately. Four names — `Ref`, `SamplerLodBind`,
//! `MemoryBarrierResources`, `MemoryBarrierScope` — are each declared twice in
//! different op families, so a crate-wide name lookup would audit one
//! declaration twice and the other never.

use std::collections::{BTreeMap, BTreeSet};

mod source_scan;

/// Primitives that are align-1 and for which every byte pattern is valid.
///
/// `u8` and `i8` and nothing else. A bare `u16` upward is align-2 upward, which
/// `ASSERT_ALIGN_1` already rejects; they are absent here so this test's message
/// is the one a reader gets, and it names the `le` scalar to use instead.
const TOTAL_PRIMITIVES: [&str; 2] = ["u8", "i8"];

/// The `le` scalar names, read from the macro invocations that define them.
///
/// `le.rs` generates each of these with `le_scalar!(Name, prim, width)` or
/// `le_float!(..)`, so the invocation list *is* the set. Writing the names out
/// here instead would be a second copy that can only ever agree or disagree
/// with the first — and the direction it would fail in is the dangerous one: a
/// scalar this test did not know about would be reported as an unrecognised
/// field type, and the obvious way to silence that is to add it to a list,
/// which is also how a genuinely unsafe type would get added.
fn le_scalar_names() -> BTreeSet<String> {
    let path = source_scan::workspace_root().join("crates/reims-vgpu-wire/src/le.rs");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{} must be readable: {e}", path.display()));
    let mut out = BTreeSet::new();
    for line in text.lines() {
        let line = line.trim();
        for macro_name in ["le_scalar!(", "le_float!("] {
            if let Some(rest) = line.strip_prefix(macro_name) {
                let name: String = rest
                    .chars()
                    .take_while(|c| c.is_alphanumeric() || *c == '_')
                    .collect();
                if !name.is_empty() {
                    out.insert(name);
                }
            }
        }
    }
    assert!(
        !out.is_empty(),
        "no `le_scalar!`/`le_float!` invocation was found in le.rs; this test can no longer see \
         the scalar set and would reject every wire field"
    );
    out
}

/// The leaf name of a possibly path-qualified type: `crate::le::U64le` → `U64le`.
fn leaf(ty: &str) -> &str {
    ty.rsplit("::").next().unwrap_or(ty).trim()
}

/// Whether `ty` may appear as a field of a `Wire` struct.
fn admissible(ty: &str, scalars: &BTreeSet<String>, wire_types: &BTreeSet<String>) -> bool {
    let ty = ty.trim();
    // `[T; N]` — an array is admissible exactly when its element is. The length
    // is not this test's business: it changes the size, never the validity.
    if let Some(inner) = ty.strip_prefix('[').and_then(|t| t.strip_suffix(']')) {
        let elem = inner.split(';').next().unwrap_or(inner);
        return admissible(elem, scalars, wire_types);
    }
    let name = leaf(ty);
    TOTAL_PRIMITIVES.contains(&name) || scalars.contains(name) || wire_types.contains(name)
}

/// `struct Name { .. }` bodies in one file, keyed by name.
///
/// The brace group is taken by matching depth rather than by finding the next
/// `\n}`, so a struct holding an array with a `const` expression in it, or any
/// future nesting, is read whole instead of truncated at the first line that
/// happens to start with a brace.
fn struct_bodies(text: &str) -> BTreeMap<String, String> {
    let chars: Vec<char> = text.chars().collect();
    let mut out = BTreeMap::new();
    let mut at = 0usize;
    while let Some(rel) = text[at..].find("struct ") {
        let start = at + rel + "struct ".len();
        at = start;
        let name: String = text[start..]
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect();
        if name.is_empty() {
            continue;
        }
        let after = start + name.len();
        // Whichever of these comes first decides the struct's form. Only a
        // braced one has named fields to audit: `;` is a unit struct and `(` a
        // tuple struct, and this crate declares neither as a wire type.
        let Some(delim) = text[after..].find(['{', ';', '(']) else {
            continue;
        };
        if !text[after + delim..].starts_with('{') {
            continue;
        }
        // `close_brace` indexes chars, so the byte offset becomes a char offset
        // once, here, rather than the two being mixed downstream.
        let open = text[..after + delim].chars().count();
        let close = source_scan::close_brace(&chars, open);
        if close > open {
            let body: String = chars[open + 1..close].iter().collect();
            out.insert(name, body);
        }
    }
    out
}

/// `name: Type` pairs from a struct body, ignoring attributes and nesting.
fn fields(body: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for raw in body.split(',') {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        // Split at the *separating* colon: the first one that is not part of a
        // `::` path. Splitting at the last colon instead reads
        // `pub nz: core::num::NonZeroU8` as a field named `core::num:` — the
        // type still comes out right and the check still fires, but the message
        // then names a field that does not exist, which is how a reader is sent
        // to the wrong line.
        let bytes = line.as_bytes();
        let mut split = None;
        for (i, b) in bytes.iter().enumerate() {
            if *b != b':' {
                continue;
            }
            let prev_is_colon = i > 0 && bytes[i - 1] == b':';
            let next_is_colon = bytes.get(i + 1) == Some(&b':');
            if !prev_is_colon && !next_is_colon {
                split = Some(i);
                break;
            }
        }
        let Some(at) = split else {
            continue;
        };
        let (lhs, rhs) = (&line[..at], &line[at + 1..]);
        let name = lhs.split_whitespace().last().unwrap_or(lhs).to_string();
        let ty = rhs.trim().to_string();
        if !ty.is_empty() && !name.is_empty() {
            out.push((name, ty));
        }
    }
    out
}

#[test]
fn every_wire_struct_field_is_valid_for_every_byte_pattern() {
    let scalars = le_scalar_names();
    let sources: Vec<(String, String)> = source_scan::guest_facing_sources()
        .into_iter()
        .filter(|(rel, _)| rel.starts_with("reims-vgpu-wire/"))
        .collect();

    assert!(
        !sources.is_empty(),
        "no `reims-vgpu-wire` source was scanned; the path prefix has moved and this test is \
         auditing nothing"
    );

    // Every `Wire` type in the crate, so a field naming one is admissible on the
    // strength of that type's own audit below.
    let wire_types: BTreeSet<String> = sources
        .iter()
        .flat_map(|(_, text)| impl_names(text))
        .collect();

    let mut audited = 0usize;
    let mut problems: Vec<String> = Vec::new();

    for (rel, text) in &sources {
        let bodies = struct_bodies(text);
        for name in impl_names(text) {
            let Some(body) = bodies.get(&name) else {
                problems.push(format!(
                    "{rel}: `unsafe impl Wire for {name}` but no `struct {name} {{ .. }}` in that \
                     file, so its fields cannot be audited"
                ));
                continue;
            };
            audited += 1;
            for (field, ty) in fields(body) {
                if !admissible(&ty, &scalars, &wire_types) {
                    problems.push(format!(
                        "{rel}: `{name}.{field}: {ty}` is not known to be valid for every byte \
                         pattern. A wire struct may hold only a `crate::le` scalar, `u8`/`i8`, an \
                         array of those, or another `Wire` type — `view` casts guest bytes \
                         straight to `&{name}`, so anything else is undefined behaviour rather \
                         than a decode error. Keep the raw scalar and expose a fallible accessor."
                    ));
                }
            }
        }
    }

    assert!(problems.is_empty(), "{}", problems.join("\n  "));
    // A scan that stops matching reports green while looking at nothing, and
    // this one has three separate ways to stop matching: the path prefix, the
    // `unsafe impl` spelling, and the struct-body parse.
    assert!(
        audited > 0,
        "no `unsafe impl Wire` was audited at all; the scan is reading the wrong shape"
    );
}

/// The `T` of every `unsafe impl Wire for T` in one file.
fn impl_names(text: &str) -> Vec<String> {
    let needle = "unsafe impl Wire for ";
    let mut out = Vec::new();
    let mut at = 0usize;
    while let Some(rel) = text[at..].find(needle) {
        let start = at + rel + needle.len();
        at = start;
        let name: String = text[start..]
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect();
        if !name.is_empty() {
            out.push(name);
        }
    }
    out
}
