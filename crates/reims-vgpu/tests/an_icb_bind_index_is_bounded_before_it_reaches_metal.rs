//! An ICB command fill bounds every bind index against the create descriptor.
//!
//! `MTLIndirectCommandBufferDescriptor` takes one maximum bind count per stage,
//! and `runtime::icb::materialize_icb` hands Metal all five the type-7 create
//! body carries — vertex, fragment, object, mesh, kernel. A later fill that sets
//! a buffer at an index past the maximum its stage declared is out of range for
//! the setter, and Metal answers an out-of-range index with an exception that
//! aborts the process rather than a status this device can decline. That is the
//! same hazard `backend::metal::constants` documents for the direct bind paths.
//!
//! `fill_compute_command` has bounded its binds since it was written. The render
//! twin, `fill_render_command`, decoded all four of its sibling maxima, passed
//! every one to Metal at create, and then bound at whatever index the fill
//! record carried. The two are the same job over one wire form, which is the
//! pairing `AGENTS.md` says must be diffed rather than read alone.
//!
//! # Why this reads source instead of calling the functions
//!
//! Both fills need a live `metal::Device`; on a Vulkan build they return
//! `IcbStatus::NoMetal` before reaching any bind, and the Linux host this crate
//! is developed on can cross-compile the Metal arm but never run it. So no test
//! that *executes* either function can observe the check on the pathway that has
//! it. The arithmetic is covered by unit tests on the pure predicate
//! (`runtime::icb::tests`); what cannot be covered that way is that the fill
//! bodies still *call* it, and that is what this scans for.
//!
//! Deleting the call from either body is what this test is written to catch.

mod source_scan;

use source_scan::{blank_comments, close_brace, workspace_root};

/// `(enclosing fn, the call its body must contain, what the call bounds)`.
const BOUNDED_FILLS: &[(&str, &str, &str)] = &[
    (
        "fn fill_render_command",
        "refuse_render_bind_past_declared_max",
        "vertex/fragment/object/mesh binds against their own stage's maximum",
    ),
    (
        "fn fill_compute_command",
        "max_kernel_buffer_bind_count",
        "kernel binds against maxKernelBufferBindCount",
    ),
];

#[test]
fn every_icb_fill_bounds_its_bind_indices_against_the_create_descriptor() {
    let path = workspace_root().join("crates/reims-vgpu/src/runtime/icb/mod.rs");
    let text = std::fs::read_to_string(&path).expect("the icb module must be readable");
    // Comments are blanked first: this file's module doc and several function
    // docs name the very slugs and fields below, so a scan over raw text would
    // find every needle in prose and pass whatever the bodies do.
    let source = blank_comments(&text);
    let chars: Vec<char> = source.chars().collect();

    for (function, needle, bounds) in BOUNDED_FILLS {
        let start = source.find(function).unwrap_or_else(|| {
            panic!("{function} is gone from icb/mod.rs — rename or retarget it")
        });
        let open = source[start..]
            .find('{')
            .map(|o| start + o)
            .unwrap_or_else(|| panic!("{function} has no body"));
        let end = close_brace(&chars, open);
        let body = &source[open..end];

        assert!(
            body.contains(needle),
            "{function} no longer bounds its {bounds}: `{needle}` is not in its body. \
             An index past the count the create descriptor declared reaches \
             Metal's set*Buffer: and aborts the process."
        );
    }
}

/// The scan can distinguish a body that has the call from one that does not.
///
/// Without this, a `close_brace` that returned the wrong offset — or a
/// `blank_comments` that blanked too much — would leave every assertion above
/// searching an empty string and reporting a clean tree. Both halves are
/// asserted, so the scanner has to have proved it can see a needle and proved it
/// can miss one before its verdict on the real bodies means anything.
#[test]
fn the_scan_can_tell_a_bounded_body_from_an_unbounded_one() {
    let source = blank_comments(
        "fn fill_bounded() { for b in &binds { refuse_render_bind_past_declared_max(b)?; } }\n\
         fn fill_unbounded() { for b in &binds { cmd.set_vertex_buffer(b.index); } }\n",
    );
    let chars: Vec<char> = source.chars().collect();

    let mut seen = Vec::new();
    for name in ["fn fill_bounded", "fn fill_unbounded"] {
        let start = source.find(name).expect("fixture must contain the fn");
        let open = start + source[start..].find('{').expect("fixture fn has a body");
        let end = close_brace(&chars, open);
        seen.push(source[open..end].contains("refuse_render_bind_past_declared_max"));
    }

    assert_eq!(
        seen,
        vec![true, false],
        "the body scan cannot distinguish a bounded fill from an unbounded one, \
         so its verdict on the real bodies is meaningless"
    );
}
