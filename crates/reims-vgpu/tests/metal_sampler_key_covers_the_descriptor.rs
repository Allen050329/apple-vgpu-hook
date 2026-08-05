//! The Metal sampler cache key must name every `ReimsVgpuSampler` word the
//! sampler descriptor is built from, and no other.
//!
//! `make_explicit_sampler` turns a bound sampler into an `MTLSamplerState` and
//! memoises it; `SamplerDescriptorKey` decides when a later bind may reuse one.
//! A word the descriptor reads and the key omits is a cache **hit** on a state
//! built from different words — the wrong filter, wrap mode or compare function
//! on a texture, with nothing in the log to say so, because a hit is the
//! success path. A word the key carries and the descriptor never reads is the
//! opposite and cheaper: a miss where a hit was correct.
//!
//! That set used to be transcribed four times over — into a `SamplerCacheEntry`
//! that repeated all fourteen fields, into its `matches`, into a standalone
//! `sampler_key_hash`, and into the entry's construction. Three of those are
//! gone: the key is one array and the compiler keeps them in step. This is the
//! fourth edge, and no compiler can see it, because the descriptor is built by
//! Objective-C setter calls in another file.
//!
//! Structural rather than a hardcoded list, so a property added to the
//! descriptor is covered without editing this test. Reading source is the only
//! way to check it from here at all: `backend::metal` is Apple-only and does
//! not compile on this host, while these files always exist.
//!
//! Deliberately *not* covered, and the reason belongs with the rule: three
//! `ReimsVgpuSampler` words are absent from both sides. `has_lod_clamp` and the
//! two `clamp_lod_*` words are the encoder call
//! `setSamplerState:lodMinClamp:lodMaxClamp:atIndex:`, applied per bind and
//! never baked into the state, and `binding` is per bind too. They are read at
//! the bind sites in `backend::metal::{render, compute}`, not here, so a set
//! comparison between these two files is exactly the right shape — adding one
//! of them to the key would make two binds that must share a state miss.

use std::collections::BTreeSet;
use std::path::PathBuf;

fn source(rel: &str) -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(rel);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{} must be readable: {e}", path.display()))
}

/// The delimited body `header` opens, by matching `open` against `close`.
fn item_body<'a>(src: &'a str, header: &str, open: u8, close: u8) -> &'a str {
    let start = src
        .find(header)
        .unwrap_or_else(|| panic!("`{header}` must still exist"));
    let at = start
        + src[start..]
            .bytes()
            .position(|b| b == open)
            .unwrap_or_else(|| panic!("`{header}` must open a body"));
    let mut depth = 0usize;
    for (i, b) in src[at..].bytes().enumerate() {
        if b == open {
            depth += 1;
        } else if b == close {
            depth -= 1;
            if depth == 0 {
                return &src[at..at + i];
            }
        }
    }
    panic!("`{header}` has no closing delimiter");
}

/// Every `<binding>.<field>` read in `code`.
fn fields_read(code: &str, binding: &str) -> BTreeSet<String> {
    let needle = format!("{binding}.");
    let mut out = BTreeSet::new();
    let mut rest = code;
    while let Some(at) = rest.find(&needle) {
        // A preceding identifier character means this is some other binding
        // whose name merely ends in `binding` — `explicit_sampler.foo`.
        let before = rest[..at].chars().next_back();
        let boundary = before.is_none_or(|c| !c.is_alphanumeric() && c != '_');
        let tail = &rest[at + needle.len()..];
        let end = tail
            .find(|c: char| !c.is_alphanumeric() && c != '_')
            .unwrap_or(tail.len());
        if boundary && end > 0 {
            out.insert(tail[..end].to_string());
        }
        rest = &tail[end..];
    }
    out
}

#[test]
fn the_sampler_cache_key_names_exactly_the_descriptor_words() {
    let samplers = source("src/backend/metal/samplers.rs");
    let cache = source("src/backend/metal/cache.rs");

    let built_from = fields_read(
        item_body(&samplers, "fn make_explicit_sampler", b'{', b'}'),
        "sampler",
    );
    let keyed_on = fields_read(item_body(&cache, "let words = [", b'[', b']'), "s");

    assert!(
        !built_from.is_empty() && !keyed_on.is_empty(),
        "neither side may come back empty — that would pass this test for the \
         wrong reason (built_from={built_from:?} keyed_on={keyed_on:?})"
    );

    let missed: Vec<_> = built_from.difference(&keyed_on).collect();
    assert!(
        missed.is_empty(),
        "the sampler descriptor is built from {missed:?}, which the cache key \
         does not carry: two samplers differing only in those share one \
         MTLSamplerState, and the second bind silently gets the first's"
    );
    let unread: Vec<_> = keyed_on.difference(&built_from).collect();
    assert!(
        unread.is_empty(),
        "the cache key carries {unread:?}, which the descriptor never reads: \
         either the key splits states that are identical, or the property was \
         dropped from the descriptor and the key was not told"
    );
}
