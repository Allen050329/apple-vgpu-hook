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

/// The body of the `if`/`else if` chain each shim branches on a console-feed
/// kind with, one entry per arm, in source order.
///
/// Brace-matched rather than line-scanned, because an arm's body contains
/// nested blocks and the closing brace that ends it is not the first one.
fn console_feed_arm_bodies(code: &str) -> Vec<String> {
    const KIND: &str = "REIMS_VGPU_CONSOLE_FEED_";
    let mut bodies = Vec::new();
    for (at, _) in code.match_indices(KIND) {
        // The kind names also appear where the feed's own macros are compared
        // against nothing; only a comparison opening a block is an arm.
        let Some(open) = code[at..].find('{').map(|n| at + n) else {
            continue;
        };
        if code[at..open].contains(';') || code[at..open].contains('}') {
            continue;
        }
        let mut depth = 0usize;
        let mut end = None;
        for (i, c) in code[open..].char_indices() {
            match c {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        end = Some(open + i);
                        break;
                    }
                }
                _ => {}
            }
        }
        if let Some(end) = end {
            bodies.push(code[open + 1..end].to_string());
        }
    }
    bodies
}

/// The last statement of an arm body, or `None` if it has none.
///
/// Everything up to the final `}` is skipped first, because a `return` nested
/// inside the arm's own `if` is exactly the shape being ruled out: the x86 shim
/// returned from inside its paint check and fell out of the arm when the paint
/// failed. "Contains a return" passes on that; "ends in a return" does not.
fn last_statement(body: &str) -> Option<&str> {
    let tail = match body.rfind('}') {
        Some(close) => &body[close + 1..],
        None => body,
    };
    tail.split(';').map(str::trim).rfind(|s| !s.is_empty())
}

/// A shim that has observed a console-feed kind must not reach the re-push
/// below it, whatever the paint it attempted did.
///
/// Every named kind means Rust has said who owns the host console, and the
/// re-push at the bottom of `fb_update` pushes the last *product* frame. Pushing
/// that while Rust names the firmware or early console is the pre-boundary steal
/// the feed exists to prevent, so each arm terminates rather than falling
/// through — which is a policy, and the reason it is checked here is that it is
/// a policy held in C twice with nothing comparing the copies.
///
/// This is the third time this exact class has cost something. The module doc
/// records the first two, both on console ownership. The third was the x86
/// shim's `_EARLY` arm returning only when its paint *succeeded* and otherwise
/// falling through to the re-push, while the arm64 shim returned either way —
/// invisible on any single-pathway boot, because each host builds one shim.
#[test]
fn a_console_feed_arm_never_falls_through_to_the_product_re_push() {
    for device in ["reims-vgpu-pci.c", "reims-vgpu-mmio.c"] {
        let bodies = console_feed_arm_bodies(&shim_code(device));
        assert!(
            !bodies.is_empty(),
            "{device} must still branch on a console-feed kind, else this test \
             passes by finding nothing to check"
        );
        for (n, body) in bodies.iter().enumerate() {
            let last = last_statement(body);
            assert!(
                last.is_some_and(|s| s.starts_with("return")),
                "{device} console-feed arm {n} can fall through to the product \
                 re-push. Rust has named who owns the console; the arm must \
                 terminate whether or not its paint landed, so its last \
                 statement is a return rather than a return nested in the paint \
                 check. Ends in: {last:?}"
            );
        }
    }
}
