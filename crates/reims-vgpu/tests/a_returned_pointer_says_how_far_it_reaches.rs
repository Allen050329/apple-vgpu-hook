//! A function that hands back a raw pointer must say how far it reaches.
//!
//! Three bugs in this crate had one shape, and it is the shape this gate
//! exists for. A pointer into mapped memory was returned, and the number of
//! bytes it was good for was supplied by somebody else — the caller's own
//! request, a field two frames up, a `SAFETY` comment asserting the span "is
//! exactly `bytes.len()`". Each was sound at the time it was written and none
//! was checked, so each was one call away from a host-memory overrun that
//! nothing in the toolchain would have reported.
//!
//! All three were the *same* regression, too: the persistent-mapping
//! optimisation. `vkMapMemory` cannot map past its memory object, so while
//! mapping-per-write was the only arm the bound was the driver's and no code
//! here had to state it. Caching the mapping deleted the check along with the
//! call, silently, in a commit that was about latency.
//!
//! So the rule: **a pointer and its extent leave together, or the function that
//! yields the pointer has already compared the two.** Anything else makes the
//! extent an input the callee never sees.
//!
//! # What this reads
//!
//! Return positions only — a `->` carrying a `*mut` or a `*const`. That is
//! narrow on purpose and the narrowness is the point: it is a population of
//! [`ROWS`]`.len()` in two crates, small enough that every entry is read rather
//! than sampled, and it is exactly where the three bugs were.
//!
//! It is **not** the whole surface, and the largest hole is worth naming
//! precisely, because it means this gate does not cover the third of the three
//! bugs. **A host address in this crate is very often a `usize`, not a
//! pointer** — `BufferSlot::mapped`, `ReadbackLease::ptr` and
//! `GuestRun::host_ptr` all are, deliberately, so the engine's state stays
//! `Send`. A `usize` has no syntax this scan can recognise, so
//! `lease_readback`, which returns exactly such an address, is invisible here
//! and its verdict lives in `slot_span_fits`'s caller instead.
//!
//! The other two uncovered positions are the struct field and the function
//! parameter. Both are stated rather than covered for the same reason: there is
//! no syntactic pairing rule to check. Two fields of a struct are adjacent
//! whether or not one means the other, and a rule that guessed would report
//! every `usize` field in the crate.
//!
//! So what this holds is the position that *can* be checked mechanically, and
//! what it buys is that the seventh signature cannot be added without a verdict.
//! A fourth bug in the `usize` or field position needs a different instrument,
//! and the honest thing is that this file does not have one.
//!
//! An integration test rather than a `#[cfg(test)]` module because it reads
//! source text and must run on every arm, including `backend-metal`, which this
//! development host can compile but cannot execute.

mod source_scan;
use source_scan::guest_facing_sources;

/// How the extent of a returned pointer is established.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Verdict {
    /// The length leaves with the pointer, in the same value. The caller cannot
    /// hold one without the other.
    PairedWithLength,
    /// The function takes the extent as an argument and compares it against the
    /// object's own size before forming the pointer. The caller supplies the
    /// number; the callee refuses it.
    CheckedBeforeReturn,
    /// Not a span. A C string, an opaque handle, an Objective-C object — a
    /// pointer with no byte count to have.
    NotASpan,
    /// The extent comes from somewhere the callee never sees. **Must not
    /// appear.** This is the shape of all three bugs.
    ExtentFromElsewhere,
}

struct Row {
    at: &'static str,
    verdict: Verdict,
    why: &'static str,
}

/// Every returned raw pointer, with a written verdict.
const ROWS: &[Row] = &[
    Row {
        at: "reims-vgpu/src/backend/metal/compute.rs:84",
        verdict: Verdict::PairedWithLength,
        why: "Returns (*mut u8, len, offset) for a compute buffer's backing; the \
              three checks above it relate offset and len to backing_len before \
              any of them is returned.",
    },
    Row {
        at: "reims-vgpu/src/backend/metal/raw_metal.rs:631",
        verdict: Verdict::NotASpan,
        why: "An Objective-C `*mut Object` beside a pipeline state — a retained \
              object handle, not a byte range, so there is no extent to carry.",
    },
    Row {
        at: "reims-vgpu/src/backend/vulkan/caps/device_features.rs:269",
        verdict: Verdict::NotASpan,
        why: "A Vec of `*const c_char` extension names for vkCreateDevice; NUL \
              terminates each one and Vulkan reads them that way.",
    },
    Row {
        at: "reims-vgpu/src/backend/vulkan/caps/external_memory.rs:165",
        verdict: Verdict::NotASpan,
        why: "The same list from the external-memory rung — `*const c_char` \
              extension names, terminated rather than measured.",
    },
    Row {
        at: "reims-vgpu/src/backend/vulkan/engine/pools/mod.rs:2246",
        verdict: Verdict::CheckedBeforeReturn,
        why: "`staging_write_ptr` takes the write size and asks `slot_span_fits` \
              against the slot's own size before either arm forms a pointer. The \
              persistent-mapping arm is why: it inherits no bound from \
              vkMapMemory.",
    },
    Row {
        at: "reims-vgpu/src/runtime/gva_view.rs:470",
        verdict: Verdict::PairedWithLength,
        why: "Returns (*mut u8, usize) for a contiguous guest page run, and the \
              usize is the run's own packed length rather than the caller's \
              request.",
    },
];

/// A `->` carrying a raw pointer, as `file:line`.
fn returned_pointers() -> Vec<String> {
    let mut found = Vec::new();
    for (path, text) in guest_facing_sources() {
        for (i, line) in text.lines().enumerate() {
            let trimmed = line.trim_start();
            if trimmed.starts_with("//") || trimmed.starts_with('*') {
                continue;
            }
            let Some(ret) = line.split("->").nth(1) else {
                continue;
            };
            if !ret.contains("*mut ") && !ret.contains("*const ") {
                continue;
            }
            found.push(format!("{path}:{}", i + 1));
        }
    }
    found.sort();
    found.dedup();
    found
}

/// The population is adjudicated, exactly.
#[test]
fn every_returned_pointer_says_how_far_it_reaches() {
    let found = returned_pointers();
    let missing: Vec<&String> = found
        .iter()
        .filter(|at| !ROWS.iter().any(|r| &r.at == at))
        .collect();
    assert!(
        missing.is_empty(),
        "a function returns a raw pointer and nothing says how far it reaches. \
         Return the length with it, compare it inside, or add a row saying it is \
         not a span:\n{}",
        missing
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>()
            .join("\n")
    );

    let stale: Vec<&str> = ROWS
        .iter()
        .filter(|r| !found.iter().any(|at| at == r.at))
        .map(|r| r.at)
        .collect();
    assert!(
        stale.is_empty(),
        "a verdict names a signature the scan no longer finds — the line moved \
         or the return type changed. Re-read it rather than re-pointing it:\n{}",
        stale.join("\n")
    );
}

/// No returned pointer takes its extent from somewhere its own function cannot
/// see.
#[test]
fn no_returned_pointer_borrows_its_extent() {
    let guilty: Vec<&str> = ROWS
        .iter()
        .filter(|r| r.verdict == Verdict::ExtentFromElsewhere)
        .map(|r| r.at)
        .collect();
    assert!(
        guilty.is_empty(),
        "a pointer leaves without its extent, and the caller supplies one the \
         callee never checked. That is the shape of every bug this gate was \
         written for:\n{}",
        guilty.join("\n")
    );
}

/// Every verdict carries a reason.
#[test]
fn every_verdict_says_why() {
    for row in ROWS {
        assert!(
            row.why.len() >= 40,
            "{}: a verdict needs a reason long enough to be checked",
            row.at
        );
    }
}

/// The scan can see the signatures it is about.
///
/// The three bugs all lived behind one of these two, and the scan's single
/// filter — a `*mut`/`*const` after a `->` — fails silently if a signature is
/// reformatted onto lines it does not read. Naming both means the file's
/// silence about a seventh signature is worth something.
#[test]
fn the_scan_can_see_the_signatures_it_is_about() {
    let found = returned_pointers();
    for file in [
        "reims-vgpu/src/backend/vulkan/engine/pools/mod.rs",
        "reims-vgpu/src/runtime/gva_view.rs",
    ] {
        assert!(
            found.iter().any(|at| at.starts_with(file)),
            "the scan no longer finds a returned pointer in {file}; its silence \
             proves nothing until it does"
        );
    }
}
