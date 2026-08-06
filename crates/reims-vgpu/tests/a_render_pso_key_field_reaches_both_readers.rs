//! Every `RenderPsoKey` field must reach both the hash that buckets it and the
//! comparison that matches it.
//!
//! `ContentCache::find` buckets on `RenderPsoKey::key_hash` and then linear-scans
//! the bucket with `matches`, which is `RenderPsoKey::equal`. So a field that
//! distinguishes two pipelines and reaches **neither** reader puts them in one
//! bucket *and* makes them compare equal — the second pipeline is never built,
//! and every draw that asked for it gets the first one. Wrong blend factors,
//! wrong vertex stride, wrong colour write mask, on the success path, with no
//! line in any log, because a cache hit is not an event.
//!
//! That is the worst thing a cache in this device can do. An eviction costs
//! rework and the three bound scans cover it; this costs correctness and no
//! scan could see it, because nothing is dropped and nothing is bounded — the
//! key is simply not the key.
//!
//! # Why both, when either alone is sufficient today
//!
//! Correctness needs only one: two keys differing in a hashed field land in
//! different buckets and are never compared, and two in one bucket are
//! separated by `equal`. Requiring both is deliberate, because "sufficient
//! today" is the shape of a latent bug. A field carried only by the hash is
//! covered by `equal` solely because `equal` compares `key_hash`; delete or
//! narrow the hash later and that field silently stops distinguishing anything.
//! Requiring both means neither reader can be weakened without a failure.
//!
//! `key_hash` is the one exemption in the hash direction: it is that fold's
//! *output*, assigned at the end, and folding it into itself is not a thing.
//! `equal` reads it like any other field and this test requires that.
//!
//! # Why only this key, of the six on `ContentCache`
//!
//! Because it is the only one whose comparison is transcribed by hand. The six
//! `CacheEntry` impls in `backend::metal::cache` were read against each other:
//!
//! * `FnEntry`, `ReflectEntry`, `SamplerCacheEntry` and `ComputePsoEntry` all
//!   compare with `self.key == *key` — derived `PartialEq`, which the compiler
//!   generates over every field. A field added to one of those keys is covered
//!   the moment it is declared, and no test can improve on that.
//! * `DepthStencilEntry` compares `depth_stencil_eq`, which is a byte
//!   comparison of the whole `ReimsVgpuDepthStencilState`. Also complete by
//!   construction, and its failure direction is the safe one: any disagreement
//!   the bytes carry produces a *miss*, so at worst a state is rebuilt.
//! * `RenderPsoEntry` compares `RenderPsoKey::equal`, thirty-three fields
//!   written out by hand, against a fold written out by hand in another file.
//!
//! So the scan is narrow on purpose rather than by omission. `SamplerCacheEntry`
//! still has `metal_sampler_key_covers_the_descriptor`, which asks the different
//! question this one cannot: whether the key names what the *descriptor* reads.
//! Derived equality guarantees a key is compared completely; it says nothing
//! about whether the key holds the right fields in the first place.
//!
//! # Why source text
//!
//! `backend::metal` does not compile on a host without an Apple linker, so a
//! test that called either reader could not run where it is most needed — the
//! same reason `metal_sampler_key_covers_the_descriptor` reads source. The two
//! files always exist. Structural rather than a hardcoded field list, so a field
//! added to the struct is covered without editing this test, which is the whole
//! point: the failure has to arrive with the change that causes it.

mod source_scan;
use source_scan::{blank_comments, workspace_root};

use std::collections::BTreeSet;

fn source(rel: &str) -> String {
    let path = workspace_root().join("crates/reims-vgpu").join(rel);
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{} must be readable: {e}", path.display()));
    // Doc comments on these fields quote sibling field names — the struct's own
    // `color_formats[i]` is one — and a scan that counts prose is measuring the
    // documentation.
    blank_comments(&raw)
}

/// The text between `start` and the first `end` after it, panicking if either
/// marker is missing.
///
/// A missing marker is a rename, not a pass: every caller here would otherwise
/// scan an empty string and find every field trivially absent, or scan the whole
/// file and find them all trivially present. Both are silent wrong answers, so
/// the marker is load-bearing and says so.
fn between<'a>(text: &'a str, start: &str, end: &str, what: &str) -> &'a str {
    let from = text
        .find(start)
        .unwrap_or_else(|| panic!("cannot find the start of {what} (`{start}`); it was renamed"));
    let rest = &text[from + start.len()..];
    let to = rest
        .find(end)
        .unwrap_or_else(|| panic!("cannot find the end of {what} (`{end}`); it was renamed"));
    &rest[..to]
}

/// Field names read off `recv` in `body`, e.g. every `self.foo` for `self`.
fn fields_read(body: &str, recv: &str) -> BTreeSet<String> {
    let needle = format!("{recv}.");
    let mut out = BTreeSet::new();
    let mut from = 0;
    while let Some(at) = body[from..].find(&needle) {
        let start = from + at;
        // `other.self.x` cannot happen, but `foo_self.x` could: require the
        // receiver to begin a token.
        let boundary = start == 0 || {
            let prev = body[..start].chars().next_back().unwrap();
            !(prev.is_alphanumeric() || prev == '_' || prev == '.')
        };
        from = start + needle.len();
        if !boundary {
            continue;
        }
        let name: String = body[from..]
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect();
        if !name.is_empty() {
            out.insert(name);
        }
    }
    out
}

/// The declared fields of `RenderPsoKey`.
fn key_fields(cache: &str) -> BTreeSet<String> {
    let body = between(
        cache,
        "pub struct RenderPsoKey {",
        "\n}",
        "the RenderPsoKey struct",
    );
    body.lines()
        .filter_map(|l| l.trim().strip_prefix("pub "))
        .filter_map(|l| l.split(':').next())
        .map(|n| n.trim().to_string())
        .filter(|n| !n.is_empty())
        .collect()
}

#[test]
fn every_render_pso_key_field_reaches_both_readers() {
    let cache = source("src/backend/metal/cache.rs");
    let render = source("src/backend/metal/render.rs");

    let fields = key_fields(&cache);
    // Self-check before believing anything: a scan that parsed zero fields would
    // report a perfectly covered key. The number is a floor, not a pin — adding
    // a field must not fail *here*, it must fail on the coverage assertions
    // below, which is where the message explains what to do.
    assert!(
        fields.len() >= 25,
        "the struct scan found only {} RenderPsoKey fields, so it is not reading \
         the declaration and its verdict means nothing: {fields:?}",
        fields.len()
    );

    let equal_body = between(
        &cache,
        "pub fn equal(&self, other: &Self) -> bool {",
        "\n    }",
        "RenderPsoKey::equal",
    );
    let read_by_equal = fields_read(equal_body, "self");

    // The fold runs inside `fill_render_pso_key` and ends by assigning its
    // result, so the assignment is the terminator rather than a closing brace.
    let hash_body = between(
        &render,
        "let mut h = FNV_OFFSET_BASIS;",
        "key.key_hash = h;",
        "the RenderPsoKey hash fold",
    );
    let read_by_hash = fields_read(hash_body, "key");

    let missing_from_equal: Vec<&String> = fields.difference(&read_by_equal).collect();
    assert!(
        missing_from_equal.is_empty(),
        "these RenderPsoKey fields are never compared by `equal`, so two \
         pipelines differing only in one of them match in the cache. Add them to \
         `equal` in src/backend/metal/cache.rs.\n\n{missing_from_equal:?}"
    );

    // `key_hash` is the fold's output; everything else is its input.
    let mut hashable = fields.clone();
    hashable.remove("key_hash");
    let missing_from_hash: Vec<&String> = hashable.difference(&read_by_hash).collect();
    assert!(
        missing_from_hash.is_empty(),
        "these RenderPsoKey fields are never folded into `key_hash`, so two \
         pipelines differing only in one of them share a bucket and rely \
         entirely on `equal` to tell them apart. That is true today and is not \
         the invariant: a field covered by one reader stops being covered the \
         moment the other is narrowed. Fold them in `fill_render_pso_key` in \
         src/backend/metal/render.rs.\n\n{missing_from_hash:?}"
    );

    // The other direction, cheap and worth having: a reader naming something the
    // struct does not declare is a rename half-done, and would otherwise sit
    // there reading as coverage.
    let stray_equal: Vec<&String> = read_by_equal.difference(&fields).collect();
    let stray_hash: Vec<&String> = read_by_hash.difference(&fields).collect();
    assert!(
        stray_equal.is_empty() && stray_hash.is_empty(),
        "these names are read as RenderPsoKey fields but are not declared on it \
         — a half-finished rename.\n  equal: {stray_equal:?}\n  hash: {stray_hash:?}"
    );
}
