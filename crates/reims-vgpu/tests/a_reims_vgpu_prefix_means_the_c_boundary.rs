//! The `REIMS_VGPU_` prefix means one thing, and one module wears it meaning
//! another.
//!
//! `AGENTS.md` states the rule this guards:
//!
//! > Anything crossing the boundary lives twice, once in Rust and once in
//! > `crates/reims-vgpu/include/reims_vgpu_qemu_abi.h`, and nothing in the
//! > toolchain compares the two … Every constant that crosses gets a test,
//! > using `qemu::abi::header_define`.
//!
//! `scripts/abi-pins` checks that rule in one direction: for every define in
//! the header that a C shim reads, is there a Rust assertion pinning it. It has
//! nothing to say about the other direction, and the other direction is where
//! this tree is surprising.
//!
//! **`backend/metal/abi.rs` declares 48 constants with the `REIMS_VGPU_`
//! prefix and not one of them appears in the header.** They mirror an
//! `reims_vgpu_backend.h` that is archived and not in this repository, which
//! that module's own doc says — but the prefix does not, and the prefix is what
//! a reader meets first. An agent applying the rule above to
//! `REIMS_VGPU_MTL_LOAD_ACTION_CLEAR` goes looking for a header entry to pin it
//! against, finds none, and has to decide whether that is a gap or the design.
//!
//! So this test writes the answer down as an assertion rather than as prose:
//! the two name sets are **disjoint**, and they are expected to stay disjoint.
//! It fails in both directions, and each failure is a real question:
//!
//! - a name in both places means a constant genuinely started crossing the live
//!   boundary, and it now needs a `header_define` pin per the rule above — the
//!   test says so in its message;
//! - an empty scan on either side means the test measured nothing, which is how
//!   a structural check reports green while looking at the wrong file.

use std::collections::BTreeSet;
use std::path::Path;

mod source_scan;

/// Names declared as `pub const REIMS_VGPU_…` in the Metal backend's ABI
/// mirror.
fn metal_abi_names() -> BTreeSet<String> {
    let path = source_scan::workspace_root().join("crates/reims-vgpu/src/backend/metal/abi.rs");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{} must be readable: {e}", path.display()));
    text.lines()
        .filter_map(|l| l.trim().strip_prefix("pub const "))
        .filter_map(|rest| rest.split(':').next())
        .map(str::trim)
        .filter(|n| n.starts_with("REIMS_VGPU_"))
        .map(str::to_owned)
        .collect()
}

/// Every `REIMS_VGPU_…` token the shared header mentions, however it mentions
/// it — a `#define`, an enum member, a comment. Deliberately wider than
/// "defines": the claim being made is that the Metal mirror's names do not
/// appear in the header *at all*, and a narrower scan could miss a name that
/// arrives as an enumerator.
fn header_names() -> BTreeSet<String> {
    let path =
        source_scan::workspace_root().join("crates/reims-vgpu/include/reims_vgpu_qemu_abi.h");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{} must be readable: {e}", path.display()));
    let mut out = BTreeSet::new();
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i].is_ascii_alphabetic() || bytes[i] == b'_' {
            let start = i;
            while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                i += 1;
            }
            let word = &text[start..i];
            if word.starts_with("REIMS_VGPU_") {
                out.insert(word.to_owned());
            }
        } else {
            i += 1;
        }
    }
    out
}

/// Neither scan may come back empty, or the disjointness below is vacuous.
///
/// This is the assertion the `wire_families_have_a_consumer` pattern asks for:
/// prove the scanner can see one before believing its diff. Both floors are
/// well under today's counts (48 and 50) and exist to catch a moved file or a
/// changed declaration style, not to pin a number.
#[test]
fn both_scans_find_constants_before_anything_is_concluded() {
    let metal = metal_abi_names();
    let header = header_names();
    assert!(
        metal.len() >= 20,
        "only {} REIMS_VGPU_ constants found in backend/metal/abi.rs — the file \
         moved or its declaration style changed, and the disjointness test below \
         would have passed by measuring nothing",
        metal.len()
    );
    assert!(
        header.len() >= 20,
        "only {} REIMS_VGPU_ names found in reims_vgpu_qemu_abi.h — same hazard",
        header.len()
    );
}

/// The Metal backend's ABI mirror shares no name with the live QEMU header.
#[test]
fn the_metal_abi_mirror_names_nothing_the_shared_header_names() {
    let metal = metal_abi_names();
    let header = header_names();
    let both: Vec<&String> = metal.intersection(&header).collect();
    assert!(
        both.is_empty(),
        "these names are declared in backend/metal/abi.rs AND appear in \
         reims_vgpu_qemu_abi.h: {both:?}\n\
         If one really crosses the C boundary now, it needs an assertion beside \
         its Rust value using `qemu::abi::header_define`, per AGENTS.md — and it \
         should move out of the Metal mirror, whose whole point is that it \
         mirrors an archived header instead. If it does not cross, rename it so \
         the prefix stops claiming that it does."
    );
}

/// The header this crate ships is the only one, so "the C boundary" is not
/// ambiguous.
///
/// The Metal mirror's doc points at `reims_vgpu_backend.h`, which is archived
/// and outside this repository. If a second header ever appears here, the
/// disjointness above stops being the whole question and this test says so
/// before someone reads a green run as covering it.
#[test]
fn the_crate_ships_exactly_one_c_header() {
    let dir = source_scan::workspace_root().join("crates/reims-vgpu/include");
    let mut headers: Vec<String> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("{} must be readable: {e}", dir.display()))
        .map(|e| e.expect("a readable dir entry").file_name())
        .map(|n| n.to_string_lossy().into_owned())
        .filter(|n| Path::new(n).extension().is_some_and(|e| e == "h"))
        .collect();
    headers.sort();
    assert_eq!(
        headers,
        ["reims_vgpu_qemu_abi.h"],
        "a second shipped header changes what `REIMS_VGPU_` can mean"
    );
}
