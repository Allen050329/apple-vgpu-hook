//! What a serialized `instanceCount:0` means is decided at exactly one site.
//!
//! `decode::render::wire_instance_count` is that site, and its doc says so.
//! Eight decode arms used to write `(…get() as u32).max(1)` with no reason
//! beside any of them; folding them into one named function is what made the
//! question answerable at all — and it is a live question, because the three
//! arms of this device that see a zero instance count still disagree. The decode
//! site floors it to one, `backend::metal::render` refuses it by name, and
//! `runtime::icb` hands the zero straight to Metal.
//!
//! The danger is not the clamp. It is a *copy* of the clamp downstream: a
//! restatement changes nothing until the rule it copies changes, and then it
//! changes one arm and not the others. A sweep that retired the downstream
//! copies missed the one in `runtime::draw::vulkan`, and the decode site's doc
//! went on claiming to be the only one for as long as that copy existed —
//! precisely the silence this test exists to break.
//!
//! # Why source text and not a behavioural test
//!
//! The property is "this rule is written in one place", which no run can
//! observe: every copy agrees with the original by construction, so a device
//! carrying four of them behaves exactly like one carrying none. It becomes
//! observable only after somebody edits the original, which is the moment the
//! test needs to have already fired.
//!
//! Reading source is also what makes this answer for both arms at once. The
//! Vulkan draw path is `#[cfg]`-ed out of the Metal build and the Metal one out
//! of the Vulkan build, so no single compilation sees both — but both are text.

mod source_scan;
use source_scan::{blank_comments, blank_test_modules, rust_sources, workspace_root};

/// The one function allowed to decide it, by the file it is written in.
const DECIDING_FILE: &str = "src/runtime/decode/render/mod.rs";

/// Every way the crate spells "floor this count at one" for a field whose name
/// ends in `instance_count`.
///
/// Only the receiver matters: `x.instance_count.max(1)`, `req.instance_count
/// .max(1)` and a bare `instance_count.max(1)` are the same restatement. The
/// scan therefore looks for the field name followed by the clamp, with the
/// intervening whitespace collapsed, rather than for a fixed expression.
fn clamps_instance_count(text: &str) -> Vec<String> {
    let flat: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut hits = Vec::new();
    let needle = "instance_count";
    let mut from = 0usize;
    while let Some(at) = flat[from..].find(needle) {
        let start = from + at;
        let after = &flat[start + needle.len()..];
        // `.max(` with any argument, and `.max( 1 )` in particular. A `.max(` on
        // something other than 1 is still a floor applied here rather than at
        // the deciding site, so it is reported too.
        let trimmed = after.trim_start_matches([' ', '\n']);
        if trimmed.starts_with(". max (") || trimmed.starts_with(".max(") {
            let end = (start + needle.len() + 24).min(flat.len());
            hits.push(flat[start..end].to_string());
        }
        from = start + needle.len();
    }
    hits
}

#[test]
fn no_file_outside_the_decoder_floors_an_instance_count() {
    let root = workspace_root();
    let src = root.join("crates/reims-vgpu/src");
    let files = rust_sources(&src);
    assert!(
        files.len() > 50,
        "the scanner found only {} sources; it is not seeing the crate and would \
         report nothing whatever the crate contained",
        files.len()
    );

    // Prove the scanner can see the shape it hunts before believing an empty
    // report. The deciding site does not spell it this way — it is an `if count
    // == 0` — so a synthetic string stands in.
    assert_eq!(
        clamps_instance_count("let n = req.instance_count.max(1);").len(),
        1,
        "the scanner cannot see a clamp it is meant to find"
    );
    assert!(
        clamps_instance_count("let n = req.instance_count;").is_empty(),
        "the scanner reports a pass-through as a clamp"
    );

    let mut offenders: Vec<String> = Vec::new();
    for path in &files {
        let rel = path
            .strip_prefix(&root)
            .unwrap_or(path)
            .to_string_lossy()
            .replace('\\', "/");
        if rel.ends_with(DECIDING_FILE) {
            continue;
        }
        let text = std::fs::read_to_string(path).expect("source is readable");
        // Comments name the rule on purpose — the pass-through in
        // `runtime::draw::vulkan` explains itself by quoting it — and a test
        // module may build one deliberately.
        let body = blank_test_modules(&blank_comments(&text));
        for hit in clamps_instance_count(&body) {
            offenders.push(format!("{rel}: {hit}"));
        }
    }

    assert!(
        offenders.is_empty(),
        "an instance count is floored outside `{DECIDING_FILE}`, which owns that \
         decision:\n  {}\n\
         Delete the copy and let the decoder's value through. If the decoder's \
         answer is wrong, change it there — that is the point of it being one \
         site.",
        offenders.join("\n  ")
    );
}

/// The deciding site still exists and still decides.
///
/// Without this, deleting `wire_instance_count` outright would leave the test
/// above passing over a crate that floors nothing anywhere — a green run for
/// the opposite of what is wanted.
#[test]
fn the_deciding_site_is_still_there() {
    let path = workspace_root().join("crates/reims-vgpu").join(
        DECIDING_FILE
            .strip_prefix("src/")
            .map(|rest| format!("src/{rest}"))
            .expect("the deciding file is under src/"),
    );
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{} is readable: {e}", path.display()));
    let body = blank_comments(&text);
    assert!(
        body.contains("fn wire_instance_count"),
        "the site that owns the zero-instance-count decision is gone; the sweep \
         test beside this one would then pass over a crate that decides nothing"
    );
    assert!(
        body.contains("draw_instance_count_zero"),
        "the deciding site no longer counts the zero it acts on, so a firing \
         would be invisible and the reading that settles the three arms cannot \
         be taken"
    );
}
