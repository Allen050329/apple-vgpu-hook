//! The host GPU must not be able to address the guest VM's memory.
//!
//! Two source assertions, one per backend. They are the surviving security
//! invariants of the deleted `observe/gate.rs`, which held them alongside
//! twenty-one style rules and was removed whole (`db80389`) on the stated
//! ground that its properties "are review concerns; none of them can make the
//! device mis-execute a guest command". That is true of the style rules and it
//! is not true of these two: what they bound is not how the device executes a
//! command but whether the host GPU can read and write the guest's RAM. They
//! are restored here, apart from the style scanner and without its lexer, so
//! that distinction is visible in the file's name.
//!
//! # Why these are source assertions and not behavioural ones
//!
//! The behavioural form cannot be built. What a reintroduction changes is what
//! `DeviceContext::create` asks the driver for, and every fixture that reads
//! that answer needs `instance.create_device` to succeed — which on a
//! driverless host degenerates into a skip, i.e. a green summary produced
//! whether or not the code is right. A source assertion has no such arm.
//!
//! The Metal invariant has a second reason: `backend-metal` is Apple-only and
//! does not compile on a Linux host at all, so nothing else in this tree — not
//! the compiler, not clippy, not the feature matrix — reads that code on the
//! machine most of this work happens on.
//!
//! # Why no comment/string masking
//!
//! The deleted gate ran a 180-line lexer to mask comments and literals, because
//! its own prose spelled the needles it searched for. The needles here live in
//! one array in this file, and this file excludes itself from the walk, so the
//! lexer buys nothing: no other file in the crate contains any of them in any
//! position, comment or code. The tree's several prose mentions of the
//! extension name are not hits, because the plain name is deliberately not one
//! of the needles — only the API surfaces are.

use std::path::{Path, PathBuf};

fn crate_src() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src")
}

/// Every `.rs` file under `src/`.
fn rust_files(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).expect("read crate source directory") {
            let path = entry.expect("read directory entry").path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "rs") {
                out.push(path);
            }
        }
    }
    out
}

/// `VK_EXT_external_memory_host` must never be asked for.
///
/// Importing a host pointer over guest RAM gives the host GPU write access to
/// the guest VM's memory. That is a property of the mechanism and not of how
/// much of it is used, so the bound is "never requested" rather than any
/// budget. The subsystem that did it — the resolver, its window budget, the
/// scatter pool and both present entry points — was deleted in `018499e`, and
/// nothing legitimate names any of these symbols now.
///
/// The needles are the whole API surface, not just the request. The name
/// constant is how a device *asks* for it; the loader type, the import struct,
/// the properties entry point and the `HOST_ALLOCATION_EXT` handle type are how
/// it would then be *used*. A reintroduction has to name at least one, so
/// matching all six means the gate does not depend on which end someone starts
/// from.
#[test]
fn the_host_pointer_import_extension_is_never_requested() {
    const NEEDLES: [&str; 6] = [
        "external_memory_host::NAME",
        "EXT_EXTERNAL_MEMORY_HOST_NAME",
        "ash::ext::external_memory_host",
        "ImportMemoryHostPointerInfoEXT",
        "get_memory_host_pointer_properties_ext",
        "ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT",
    ];
    let root = crate_src();
    let mut hits = Vec::new();
    for path in rust_files(&root) {
        let src = std::fs::read_to_string(&path).expect("read Rust source");
        // Folded, so a rustfmt wrap inside a path cannot hide a request.
        let folded: String = src.chars().filter(|c| !c.is_whitespace()).collect();
        for needle in NEEDLES {
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
        "VK_EXT_external_memory_host is requested; the host GPU must not be \
         able to write guest RAM:\n  {}",
        hits.join("\n  ")
    );
}

/// Metal has exactly one no-copy buffer constructor, and its bytes are ours.
///
/// `newBufferWithBytesNoCopy` is the Metal half of the same hazard: it hands
/// the GPU a pointer, and if that pointer is a `mach_vm_remap` view of the
/// guest's pages then the host GPU can read and write guest RAM. It is not
/// banned outright, because the crate legitimately aliases its **own**
/// allocations with it — the CPU-staged vertex, fragment and compute byte
/// vectors in `new_buffer_from_host`, which are `Vec<u8>`s this process owns.
///
/// So the invariant is not "never call it", it is "call it in exactly one
/// place, whose argument is provably host-owned". A second call site is the
/// thing to look at, whatever it claims to be aliasing. The one that used to
/// exist took `MappingEntry::contig_ptr` and became a linear texture the guest
/// surface was rendered into (`4fd4695`).
///
/// The needle is the Rust binding's snake-case spelling, so the tree's prose
/// mentions of the Objective-C selector are not hits.
#[test]
fn metal_no_copy_buffers_alias_host_memory_and_nothing_else() {
    const OWNER: &str = "backend/metal/runtime.rs";
    const NEEDLE: &str = "new_buffer_with_bytes_no_copy(";
    let root = crate_src();
    let mut sites = Vec::new();
    for path in rust_files(&root) {
        let src = std::fs::read_to_string(&path).expect("read Rust source");
        let folded: String = src.chars().filter(|c| !c.is_whitespace()).collect();
        if folded.contains(NEEDLE) {
            sites.push(
                path.strip_prefix(&root)
                    .unwrap_or(&path)
                    .to_string_lossy()
                    .replace('\\', "/"),
            );
        }
    }
    assert_eq!(
        sites,
        vec![OWNER.to_string()],
        "the only no-copy Metal buffer may be the one over this process's own \
         bytes; a second site hands the GPU a pointer nothing here has vouched \
         for"
    );
}
