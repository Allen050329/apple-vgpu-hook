//! A deferred window is armed only where the Vulkan engine exists.
//!
//! `runtime/storage_flush` defers a guest writeback because a *pinned engine
//! resident* holds the authoritative content, and the engine is
//! `backend::vulkan::engine`. On a build without `backend-vulkan` — every
//! `backend-metal` build, since `lib.rs` requires exactly one — the rail's seven
//! entry points are stubs, and **three of them are silently empty**:
//! `flush_gva_windows_before_fence`, `flush_linear_windows_before_fence` and
//! `flush_mapping_windows_before_fence` simply return.
//!
//! That is correct only because nothing can arm a window on that build. Every
//! production arm site sits inside `backend-vulkan`-gated code today. If one
//! ever lands outside a gate, those three stubs stop being an empty pass over an
//! empty set and become a silent drop of a real writeback obligation — the
//! failure this project's first rule exists to prevent, on the one arm this host
//! cannot boot to catch it.
//!
//! So the premise is pinned here rather than assumed. The check is a whitelist
//! of arm sites, not a gate-parser: there are four, each names the `#[cfg]` it
//! relies on, and a fifth appearing anywhere fails until someone decides which
//! side of the line it is on.

use std::collections::BTreeSet;

mod source_scan;
use source_scan::{blank_comments, blank_test_items, rust_sources, workspace_root};

/// The calls that put a window into `DeviceState`'s deferred maps.
///
/// Arming is spelled two ways: the two `arm_*_deferred_window` methods, and a
/// direct `compute_deferred_flush.insert` — the mapping-keyed map is a `pub`
/// field with no arming method, so the insert *is* the arm.
///
/// Each pattern starts at the receiver's dot. That is load-bearing twice: it
/// skips the `pub fn arm_…` definitions in `model/state.rs`, and it stops
/// `arm_linear_deferred_window(` from matching inside
/// `disarm_linear_deferred_window(`, which storage_flush calls on the way *out*
/// of a window.
const ARM_CALLS: &[&str] = &[
    ".arm_gva_deferred_window(",
    ".arm_linear_deferred_window(",
    ".compute_deferred_flush.insert(",
];

/// Every production arm site, and the `#[cfg]` that keeps it off a Metal build.
///
/// `(file, call, gate)`. The gate is prose, checked by a reader; the point of
/// writing it down is that adding a row makes you say which one you are
/// relying on.
const ARM_SITES: &[(&str, &str, &str)] = &[
    (
        "runtime/draw/vulkan.rs",
        ".compute_deferred_flush.insert(",
        "the whole module: `draw/mod.rs` declares it \
         `#[cfg(feature = \"backend-vulkan\")] mod vulkan;`",
    ),
    (
        "runtime/draw/vulkan.rs",
        ".arm_gva_deferred_window(",
        "the same module gate.",
    ),
    (
        "runtime/compute_exec/mod.rs",
        ".arm_linear_deferred_window(",
        "`execute_dispatch_linux`, which carries \
         `#[cfg(feature = \"backend-vulkan\")]`.",
    ),
    (
        "runtime/compute_exec/mod.rs",
        ".compute_deferred_flush.insert(",
        "the same function.",
    ),
];

#[test]
fn no_arm_site_lives_outside_a_backend_vulkan_gate() {
    let root = workspace_root();
    let src = root.join("crates/reims-vgpu/src");

    let mut found: BTreeSet<(String, String)> = BTreeSet::new();
    for path in rust_sources(&src) {
        let rel = path
            .strip_prefix(&src)
            .unwrap_or(&path)
            .to_string_lossy()
            .into_owned();
        if rel.ends_with("tests.rs") || rel.contains("tests/") {
            continue;
        }
        // Comments first, then test-module bodies: a fixture arming a window
        // is not a production arm site, and `storage_flush` alone arms one in
        // about thirty tests. Brace-matched rather than cut at the first marker,
        // because production code lives after test modules in this tree and a
        // cutoff would hide it — measured: appending an arm site to the end of
        // `mapper.rs` was invisible to the cutoff version of this scan.
        let production = blank_test_items(&blank_comments(
            &std::fs::read_to_string(&path).expect("readable"),
        ));
        let production = production.as_str();
        for call in ARM_CALLS {
            if production.contains(call) {
                found.insert((rel.clone(), (*call).to_string()));
            }
        }
    }

    // Refuse a verdict unless the scan sees the sites that are known to be
    // there. An over-eager test boundary, a wrong path or a renamed method would
    // all report an empty set, which reads exactly like a clean tree.
    let known: BTreeSet<(String, String)> = ARM_SITES
        .iter()
        .map(|(f, c, _)| (f.to_string(), c.to_string()))
        .collect();
    let missing: Vec<&(String, String)> = known.difference(&found).collect();
    assert!(
        missing.is_empty(),
        "the scan did not find these known arm sites, so its notion of `no \
         ungated armers` is a blind spot and not a measurement: {missing:?}\n\
         saw: {found:?}"
    );

    let unlisted: Vec<&(String, String)> = found.difference(&known).collect();
    assert!(
        unlisted.is_empty(),
        "these arm a deferred window and are not in ARM_SITES. A window armed \
         outside a `backend-vulkan` gate is a writeback the Metal arm's three \
         empty `*_before_fence` stubs drop in silence — put the site behind the \
         gate, or add a row saying which `#[cfg]` covers it:\n  {unlisted:?}"
    );
}
