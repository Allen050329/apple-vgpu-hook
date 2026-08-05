//! Which record families `reims-vgpu-wire` describes and this crate never reads.
//!
//! The wire crate's stated goal is that all wire-format interpretation lives
//! there, so a family with no consumer here is not automatically wrong — it is
//! either a **gap** (a real serializer record this device does not act on yet)
//! or a **family still declared twice**, which is the defect. Either way the set
//! must be small, deliberate and written down, because it only grows quietly.
//!
//! AGENTS.md asked this question with two greps:
//!
//! ```sh
//! ls crates/reims-vgpu-wire/src/ops/*.rs | xargs -n1 basename | sed 's/.rs$//'
//! grep -rh 'use reims_vgpu_wire' --include='*.rs' crates/reims-vgpu/src
//! ```
//!
//! Comparing those two lists by eye gets the answer **wrong**, and not
//! marginally: run against this tree it reports five families with no importer,
//! and two of the five — `depth_stencil` and `texture_view` — are imported on
//! the very next line of the same file. Both come in through
//!
//! ```ignore
//! use reims_vgpu_wire::ops::{
//!     backed_texture as w_backed, depth_stencil as w_ds, heap_texture as w_heap,
//!     icb as w_icb, sampler as w_smp, texture_view as w_view,
//! };
//! ```
//!
//! where the family name never follows the token `ops::` at all — it sits inside
//! a brace group, spanning lines, behind an alias. A 40 % false-positive rate on
//! a question whose answer is "delete this module" is worse than no instrument,
//! so this test replaces the grep pair: it parses the brace group, and it fails
//! when the unconsumed set stops matching the list below.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// Families this crate does not read, each with why it is a gap and not a
/// duplicate. Adding a name here is a claim, so each one carries the guest
/// action that would need a consumer.
const UNCONSUMED: &[(&str, &str)] = &[
    (
        "destroy",
        "serializer delete records (0x3e8..0x3f7). This device frees objects \
         from the FIFO packet opcode CHILD_OP_DELETE_OBJECT (0x25) instead, \
         which is a different opcode space — so these are decoded by nothing \
         and duplicated by nothing.",
    ),
    (
        "fence",
        "newFence (opcode 13). The blit encoder's fence update/wait (0x13c / \
         0x13d) is a separate record family that runtime::blit_exec does \
         execute; creating the fence object is what has no consumer.",
    ),
    (
        "rate_map",
        "newRasterizationRateMap (opcode 0x32). Variable rasterization rate is \
         a real Metal feature no rail here binds.",
    ),
];

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("the crate lives two levels below the workspace root")
        .to_path_buf()
}

/// Every `ops/*.rs` family the wire crate declares.
fn wire_families(root: &Path) -> BTreeSet<String> {
    let dir = root.join("crates/reims-vgpu-wire/src/ops");
    let mut out = BTreeSet::new();
    for entry in std::fs::read_dir(&dir).expect("the wire crate must have an ops directory") {
        let path = entry.expect("a readable dir entry").path();
        if path.extension().is_some_and(|e| e == "rs") {
            let stem = path
                .file_stem()
                .and_then(|s| s.to_str())
                .expect("a utf-8 file name")
                .to_string();
            if stem != "mod" {
                out.insert(stem);
            }
        }
    }
    assert!(
        out.len() > 5,
        "found {} families, which is not the wire crate",
        out.len()
    );
    out
}

/// Every family named anywhere under this crate's `src/`.
///
/// Both spellings are read: a direct path (`reims_vgpu_wire::ops::render`) and a
/// brace group, which may span lines and may alias every member
/// (`ops::{texture_view as w_view, ..}`). The brace group is what the grep this
/// replaces could not see, so it is taken from the `{` to the matching `}` and
/// every bare identifier in it counts.
fn consumed_families(root: &Path, families: &BTreeSet<String>) -> BTreeSet<String> {
    let mut sources = Vec::new();
    collect_rs(&root.join("crates/reims-vgpu/src"), &mut sources);
    assert!(
        sources.len() > 50,
        "walked {} files, which is not this crate",
        sources.len()
    );

    let mut seen = BTreeSet::new();
    for path in sources {
        let text = std::fs::read_to_string(&path).expect("crate source must be readable");
        let mut rest = text.as_str();
        while let Some(at) = rest.find("ops::") {
            let after = &rest[at + "ops::".len()..];
            match after.strip_prefix('{') {
                // Brace group: scan to the closing brace, take every identifier.
                Some(group) => {
                    let end = group.find('}').unwrap_or(group.len());
                    for word in group[..end].split(|c: char| !c.is_alphanumeric() && c != '_') {
                        if families.contains(word) {
                            seen.insert(word.to_string());
                        }
                    }
                }
                // Direct path: the family is the identifier that follows.
                None => {
                    let word: String = after
                        .chars()
                        .take_while(|c| c.is_alphanumeric() || *c == '_')
                        .collect();
                    if families.contains(&word) {
                        seen.insert(word);
                    }
                }
            }
            rest = &rest[at + "ops::".len()..];
        }
    }
    seen
}

fn collect_rs(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).expect("crate src must be readable") {
        let path = entry.expect("a readable dir entry").path();
        if path.is_dir() {
            collect_rs(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

#[test]
fn every_wire_family_is_consumed_or_listed_as_a_gap() {
    let root = workspace_root();
    let families = wire_families(&root);
    let consumed = consumed_families(&root, &families);

    // The scan is only trustworthy if it can see an aliased brace-group import,
    // which is the exact case the grep pair missed. Assert it does before
    // believing anything the diff below says.
    assert!(
        consumed.contains("texture_view") && consumed.contains("depth_stencil"),
        "the scan cannot see `ops::{{ .. as w_view, .. as w_ds }}` in \
         runtime/decode/resource/mod.rs, so its notion of `unconsumed` is the \
         grep's and not worth reading: {consumed:?}"
    );

    let listed: BTreeSet<String> = UNCONSUMED.iter().map(|(n, _)| n.to_string()).collect();
    let unconsumed: BTreeSet<String> = families.difference(&consumed).cloned().collect();

    let unlisted: Vec<&String> = unconsumed.difference(&listed).collect();
    assert!(
        unlisted.is_empty(),
        "these wire families have no consumer in this crate and no entry in \
         UNCONSUMED. Each is either a gap worth naming or a family declared \
         twice — decide which and say so there: {unlisted:?}"
    );

    let stale: Vec<&String> = listed.difference(&unconsumed).collect();
    assert!(
        stale.is_empty(),
        "UNCONSUMED still names families this crate now reads; delete the \
         entries so the list keeps meaning something: {stale:?}"
    );
}
