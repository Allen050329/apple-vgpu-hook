//! The two QEMU device shims must reach a wrapped ABI entry point only through
//! `reims-vgpu-shim.c`.
//!
//! Nothing else in the toolchain compares the shims: they are C, they do not
//! read Rust, and each is built for a different host, so a rule that exists in
//! both is two copies with no diff between them. That has cost twice already,
//! both times on who owns the host console. `reims_vgpu_qemu_scanout_may_paint`
//! was assembled shim-side from two other queries, and the x86 shim gated on it
//! while the arm64 shim painted every present it was handed. Then
//! `reims_vgpu_qemu_console_feed` was called raw by the arm64 shim, which read a
//! failed call as "not early" and fell through to its post-boundary re-push —
//! the shim inventing a policy for "no answer".
//!
//! So the rule is structural rather than a list: once a `reims_vgpu_qemu_*`
//! entry point is wrapped in the shared shim, that wrapper is the only caller.
//! A wrapper added later is covered without editing this test, and re-inlining
//! one fails here rather than on whichever pathway is not being booted.

use std::collections::BTreeSet;
use std::path::PathBuf;

/// `vendor/qemu/hw/display/<name>`, with C comments blanked.
///
/// The comments are stripped because they are where these symbols get *named*:
/// a wrapper's doc says which entry point it forwards, and matching that would
/// report the documentation as the violation it warns about.
fn shim_code(name: &str) -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../vendor/qemu/hw/display")
        .join(name);
    let src = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{} must be readable: {e}", path.display()));

    let mut out = String::with_capacity(src.len());
    let mut rest = src.as_str();
    while let Some(cut) = rest.find("/*").into_iter().chain(rest.find("//")).min() {
        out.push_str(&rest[..cut]);
        let (open, close) = if rest[cut..].starts_with("/*") {
            ("/*", "*/")
        } else {
            ("//", "\n")
        };
        let tail = &rest[cut + open.len()..];
        rest = match tail.find(close) {
            Some(end) => &tail[end + close.len()..],
            None => "",
        };
    }
    out.push_str(rest);
    out
}

/// Every `reims_vgpu_qemu_*` staticlib entry point *called* in `code`.
///
/// Lowercase only, because the `REIMS_VGPU_QEMU_*` status and kind macros share
/// the prefix and are constants both shims are meant to read directly. And a
/// call rather than a mention, because `#include "reims_vgpu_qemu_abi.h"` shares
/// it too and every one of these files needs that include.
fn entry_points(code: &str) -> BTreeSet<String> {
    const PREFIX: &str = "reims_vgpu_qemu_";
    let bytes = code.as_bytes();
    let mut found = BTreeSet::new();
    for (start, _) in code.match_indices(PREFIX) {
        // A match inside a longer identifier is not this entry point.
        if start > 0 && (bytes[start - 1].is_ascii_alphanumeric() || bytes[start - 1] == b'_') {
            continue;
        }
        let end = code[start..]
            .find(|c: char| !c.is_ascii_alphanumeric() && c != '_')
            .map_or(code.len(), |n| start + n);
        if code[end..].trim_start().starts_with('(') {
            found.insert(code[start..end].to_string());
        }
    }
    found
}

#[test]
fn a_wrapped_abi_entry_point_has_exactly_one_caller() {
    let wrapped = entry_points(&shim_code("reims-vgpu-shim.c"));
    assert!(
        wrapped.contains("reims_vgpu_qemu_scanout_may_paint")
            && wrapped.contains("reims_vgpu_qemu_console_feed"),
        "the shared shim must still wrap the two console-ownership entry points, \
         else this test passes by finding nothing to check: {wrapped:?}"
    );

    for device in ["reims-vgpu-pci.c", "reims-vgpu-mmio.c"] {
        let direct: Vec<_> = entry_points(&shim_code(device))
            .intersection(&wrapped)
            .cloned()
            .collect();
        assert!(
            direct.is_empty(),
            "{device} calls {direct:?} directly; reims-vgpu-shim.c already wraps \
             each of those, and the wrapper exists because the two shims had \
             drifted on the rule it holds. Call the reims_vgpu_shim_* wrapper."
        );
    }
}
