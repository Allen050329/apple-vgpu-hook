//! A Vulkan state enum is spelled in `translate/`, or it is spelled twice.
//!
//! `translate/mod.rs` states the rule — "put it here, not at the call site; a
//! bare `vk::Format` return type is fine, spelling a specific format is not" —
//! and credits an `observe/gate.rs` with enforcing it. That module was deleted
//! in `db80389` and the rule has been unenforced since, which is the shape this
//! project treats as worse than no rule: three docs described an enforcement
//! that did not run.
//!
//! Why it matters is not tidiness. Every one of these tables is a Metal ordinal
//! meeting a Vulkan value, the two specs agree on the *ordering* of several of
//! them by coincidence, and a second copy of one arm somewhere else in the
//! engine is invisible to the compiler and to every boot that does not happen to
//! take it. The recurring cure in this codebase is one table both consumers
//! read; the recurring bug is two that agree today.
//!
//! # The property is derived, not listed
//!
//! There is no hand-written list of "state enums" here. The set is whatever
//! `translate/` itself spells variants of, so declaring a new table
//! automatically closes the door behind it, and retiring one opens it. That is
//! the difference between pinning a property and pinning a list — a list is a
//! second copy of the rule, which is the defect this file exists to catch.
//!
//! `caps/` is the other legitimate speller, per the same doc: a capability query
//! names formats to ask the driver about them, which is not a translation of
//! anything the guest said.

use std::collections::{BTreeMap, BTreeSet};

mod source_scan;
use source_scan::{blank_comments, blank_test_modules, rust_sources, workspace_root};

/// Production sites outside `translate/` and `caps/` that spell a variant of a
/// type `translate/` owns, and why each one is not a second copy of a table.
///
/// Keyed by (path suffix, type). Adding an entry is a claim; each carries the
/// argument for it, and the test fails if an entry stops matching anything.
const ALLOWED: &[(&str, &str, &str)] = &[
    (
        "backend/vulkan/engine/context.rs",
        "Format",
        "the combined depth-stencil probe. `D32_SFLOAT_S8_UINT` and \
         `D24_UNORM_S8_UINT` are the pair the Vulkan required-format table \
         guarantees one of — which one is a property of the driver, so the \
         device asks. No Metal value reaches this: the depth attachment format \
         is chosen from what the host supports, not from anything the guest \
         said, which makes it the `caps` half of the rule wearing the wrong \
         directory. Moving it needs the device-creation split, not a table.",
    ),
    (
        "backend/vulkan/engine/context.rs",
        "FormatFeatureFlags",
        "`DEPTH_STENCIL_ATTACHMENT`, the feature bit the probe above asks \
         about. It is in the derived set only because `translate::support` \
         also queries feature flags; neither is a Metal-to-Vulkan mapping.",
    ),
];

/// Every `vk::<Type>::<VARIANT>` spelling in `text`, as (type, variant).
fn variant_spellings(text: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut at = 0usize;
    while let Some(rel) = text[at..].find("vk::") {
        let start = at + rel;
        at = start + "vk::".len();
        let after: Vec<char> = text[at..].chars().collect();
        let ty: String = after
            .iter()
            .take_while(|c| c.is_alphanumeric() || **c == '_')
            .collect();
        // A type name, not a module (`vk::khr::…`) or a bare constant.
        if ty.is_empty() || !ty.starts_with(|c: char| c.is_uppercase()) {
            continue;
        }
        let rest: String = after.iter().skip(ty.chars().count()).collect();
        let Some(tail) = rest.strip_prefix("::") else {
            continue;
        };
        let variant: String = tail
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect();
        // `SCREAMING_SNAKE` is how ash spells an enum variant; `default()` and
        // friends are calls on the type, not values of it.
        if variant.is_empty()
            || !variant
                .chars()
                .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
        {
            continue;
        }
        out.push((ty, variant));
    }
    out
}

/// Production text of one file: comments and `#[cfg(test)]` module bodies gone.
fn production_text(path: &std::path::Path) -> String {
    let raw = std::fs::read_to_string(path).expect("crate source must be readable");
    blank_test_modules(&blank_comments(&raw))
}

#[test]
fn no_vulkan_state_enum_variant_is_spelled_outside_translate() {
    let root = workspace_root();
    let src = root.join("crates/reims-vgpu/src");
    let vulkan = src.join("backend/vulkan");

    // The types translate owns, derived from what it spells.
    let mut owned: BTreeSet<String> = BTreeSet::new();
    for path in rust_sources(&vulkan.join("translate")) {
        for (ty, _) in variant_spellings(&production_text(&path)) {
            owned.insert(ty);
        }
    }
    // Refuse a verdict unless the derivation found the tables that are the
    // reason this rule exists. An empty or half-blind `owned` set makes every
    // site elsewhere legal and the test green while measuring nothing.
    for expected in ["Format", "BlendFactor", "SamplerAddressMode", "CompareOp"] {
        assert!(
            owned.contains(expected),
            "translate/ does not appear to spell `vk::{expected}`, so the \
             derived owner set is wrong and nothing below is a measurement: \
             {owned:?}"
        );
    }

    let mut found: BTreeMap<(String, String), Vec<String>> = BTreeMap::new();
    for path in rust_sources(&src) {
        let rel = path
            .strip_prefix(&root)
            .unwrap_or(&path)
            .to_string_lossy()
            .into_owned();
        if rel.contains("backend/vulkan/translate/") || rel.contains("backend/vulkan/caps/") {
            continue;
        }
        for (ty, variant) in variant_spellings(&production_text(&path)) {
            if owned.contains(&ty) {
                found
                    .entry((rel.clone(), ty))
                    .or_default()
                    .push(variant.clone());
            }
        }
    }

    // The other half of the self-check: the scan must see the one site that is
    // known to be there. If it sees nothing at all, "no offenders" would be
    // indistinguishable from "read no production code".
    assert!(
        found.contains_key(&(
            "crates/reims-vgpu/src/backend/vulkan/engine/context.rs".into(),
            "Format".into()
        )),
        "the scan did not find the depth-stencil probe in engine/context.rs, \
         so it is not reading production code: {:?}",
        found.keys().collect::<Vec<_>>()
    );

    let allowed: BTreeSet<(String, String)> = ALLOWED
        .iter()
        .map(|(file, ty, _)| (file.to_string(), ty.to_string()))
        .collect();

    let mut offenders: Vec<String> = Vec::new();
    for ((file, ty), variants) in &found {
        if allowed.iter().any(|(f, t)| file.ends_with(f) && t == ty) {
            continue;
        }
        let mut names: Vec<&String> = variants.iter().collect();
        names.sort();
        names.dedup();
        offenders.push(format!("{file}: vk::{ty}::{{{}}}", {
            let joined: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
            joined.join(", ")
        }));
    }
    assert!(
        offenders.is_empty(),
        "these sites spell a Vulkan value whose table lives in \
         `backend/vulkan/translate/`. Move the mapping there beside the decoder \
         that produces its input, or add an entry to ALLOWED saying why this one \
         is not a translation:\n  {}",
        offenders.join("\n  ")
    );

    let stale: Vec<String> = ALLOWED
        .iter()
        .filter(|(file, ty, _)| {
            !found
                .keys()
                .any(|(f, t)| f.ends_with(file) && t == ty.to_string().as_str())
        })
        .map(|(file, ty, _)| format!("{file}: vk::{ty}"))
        .collect();
    assert!(
        stale.is_empty(),
        "ALLOWED names sites that no longer spell these types; delete the \
         entries so the list keeps meaning something: {stale:?}"
    );
}
