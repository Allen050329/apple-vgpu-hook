//! A texture level's byte extent is one rule, so it is computed in one place.
//!
//! Bounding a mip level against the guest's declared `allocation_size` needs the
//! number of bytes the device actually touches from the level's offset. The
//! obvious `row_stride * height` is wrong by one row of trailing padding, and
//! `TextureLevelLayout::read_span` is the rule — its doc carries the measured
//! case, a 27x27 `RG8Unorm` window-corner mask that scores 12 496 against a
//! 12 288-byte allocation under the loose form and 12 166 under the true one.
//! The refusal that produced dropped the WindowServer's whole composite draw, so
//! window corners rendered square.
//!
//! That was fixed once, in `draw::texture_view` and `runtime::mipmap`. It
//! was not fixed in `draw::load_linear_texture_rgba_at_level`, which is the
//! same row-by-row reader on the other backend arm and kept `bpr * h` for months
//! afterwards. Nothing could see it: the two are twin functions behind different
//! `cfg`s, so no build compiles both, and the arm carrying the stale copy is
//! `backend-metal`, which the Linux development host can cross-compile but never
//! run. A boot on either pathway looks identical.
//!
//! So this test reads source text rather than running anything. It is the only
//! instrument in the tree that can see a divergence between two `cfg` arms, and
//! it runs on every arm because it never compiles the code it inspects.
//!
//! # What is checked, and what is deliberately not
//!
//! Every site that compares a level extent against `allocation_size` must derive
//! that extent from `read_span` or `slice_read_span` — *except* the sites in
//! [`WIDE_SPAN_EXEMPTIONS`], which genuinely touch the whole strided span. There
//! are two, for two different reasons, and neither is "a reader that forgot":
//! one bounds a render-target *write*, and one is a memo that reads each row's
//! padding on purpose so a guest write into it cannot go unnoticed.
//!
//! Writing this test is what found the second one. It was not on the list of
//! sites I meant to check.

mod source_scan;

use std::path::Path;

/// Sites whose extent really is `row_stride * height`, and why.
///
/// `(file, enclosing fn, why)`. Both entries reached this list by being caught
/// and then argued, which is the only way one should.
///
/// **Keyed by function, not by file, and that is load-bearing.** The first draft
/// of this list keyed on the file alone. `runtime/draw/mod.rs` then held
/// both an exempt writer *and* the reader this whole test was written for, so
/// the file-wide exemption covered the bug: restoring `bpr * h` at the reader
/// left the test green. It was caught only by putting the bug back and watching
/// for red, which is the one step that separates a gate from a decoration.
///
/// The writer has since moved to `render_target.rs`, so today the two are not
/// even in the same file — which is exactly why the key must not be one. A move
/// is not an argument, and the next pair to share a file will not announce
/// itself.
const WIDE_SPAN_EXEMPTIONS: &[(&str, &str, &str)] = &[
    (
        "runtime/draw/render_target.rs",
        "resolve_render_target",
        "its mip>0 arm bounds a render-target WRITE; whether the store writes \
         the last row's padding is unmeasured, and the wider span is the \
         fail-closed direction for a write",
    ),
    (
        "runtime/draw/vulkan.rs",
        "load_linear_guest_memoized",
        "deliberately READS the padding: its memo compares the full native span \
         byte for byte, so a guest write into a row's trailing bytes has to be \
         covered or the memo would serve stale texels. It touches what it \
         charges for",
    ),
];

/// The name of the `fn` a line sits inside, by scanning backwards for the
/// nearest declaration.
fn enclosing_fn(lines: &[&str], at: usize) -> String {
    for line in lines[..=at].iter().rev() {
        let t = line.trim_start();
        let rest = t
            .strip_prefix("pub(crate) fn ")
            .or_else(|| t.strip_prefix("pub(super) fn "))
            .or_else(|| t.strip_prefix("pub fn "))
            .or_else(|| t.strip_prefix("fn "));
        if let Some(rest) = rest {
            return rest
                .split(|c: char| !c.is_alphanumeric() && c != '_')
                .next()
                .unwrap_or_default()
                .to_string();
        }
    }
    String::new()
}

/// The loose form, as it can be spelled.
fn is_loose_span(line: &str) -> bool {
    let l = line.replace(' ', "");
    // `row_stride`/`bpr` times a height, in either order, by any of the
    // multiply spellings this crate uses.
    const STRIDES: &[&str] = &["row_stride", "bpr", "stride"];
    const HEIGHTS: &[&str] = &["height", "h)", "hasu64", "heightasu64"];
    let mul = l.contains("checked_mul") || l.contains("saturating_mul") || l.contains('*');
    if !mul {
        return false;
    }
    STRIDES.iter().any(|s| l.contains(s)) && HEIGHTS.iter().any(|h| l.contains(h))
}

#[test]
fn every_allocation_bound_derives_its_span_from_read_span() {
    let root = source_scan::workspace_root();
    let src = root.join("crates/reims-vgpu/src");
    let files = source_scan::rust_sources(&src);
    assert!(
        files.len() > 50,
        "the scanner found {} files, which is not this crate — a path change would \
         otherwise make this test pass by inspecting nothing",
        files.len()
    );

    // Proof the scanner can see the shape it is looking for, before its silence
    // is read as a clean tree.
    assert!(
        is_loose_span("let span = layout.row_stride.checked_mul(layout.height as u64)?;"),
        "the matcher must recognise the loose form"
    );
    assert!(
        is_loose_span("let span = bpr.checked_mul(h as u64)?;"),
        "the matcher must recognise the loose form's short spelling"
    );
    assert!(
        !is_loose_span("let span = layout.read_span(tight)?;"),
        "the matcher must not flag the rule itself"
    );

    let mut checked_sites = 0usize;
    let mut offenders: Vec<String> = Vec::new();

    for file in &files {
        let rel = file
            .strip_prefix(&src)
            .expect("every scanned file is under src")
            .to_string_lossy()
            .replace('\\', "/");
        // The type that declares the rule states it; its own tests exercise both
        // forms on purpose.
        if rel == "runtime/decode/resource/mod.rs" || rel.ends_with("/tests.rs") {
            continue;
        }
        let text = std::fs::read_to_string(file).expect("a readable source file");
        let text = source_scan::blank_test_modules(&source_scan::blank_comments(&text));
        let lines: Vec<&str> = text.lines().collect();

        for (i, line) in lines.iter().enumerate() {
            if !line.contains("allocation_size") {
                continue;
            }
            // A comparison against the declared allocation. The span it uses was
            // computed in the few lines above it.
            if !line.contains('>') && !line.contains('<') {
                continue;
            }
            checked_sites += 1;
            let window_start = i.saturating_sub(8);
            let window = &lines[window_start..i];
            let loose = window.iter().any(|l| is_loose_span(l));
            let owner = enclosing_fn(&lines, i);
            let exempt = WIDE_SPAN_EXEMPTIONS
                .iter()
                .any(|(f, func, _)| *f == rel && *func == owner);
            if loose && !exempt {
                offenders.push(format!(
                    "{rel}:{} (in `{owner}`) bounds allocation_size with a \
                     row_stride*height span; use TextureLevelLayout::read_span",
                    i + 1
                ));
            }
        }
    }

    assert!(
        checked_sites >= 4,
        "only {checked_sites} allocation_size comparisons found; the scan is not \
         reaching the sites it was written for"
    );
    assert!(
        offenders.is_empty(),
        "a level extent must come from read_span:\n  {}",
        offenders.join("\n  ")
    );
}

/// Every exemption must name a file that exists, so a rename cannot quietly
/// widen the list into a blanket pass.
#[test]
fn each_wide_span_exemption_still_points_at_a_file() {
    let src = source_scan::workspace_root().join("crates/reims-vgpu/src");
    assert!(
        !WIDE_SPAN_EXEMPTIONS.is_empty(),
        "the list is data, not a stub"
    );
    for (file, func, why) in WIDE_SPAN_EXEMPTIONS {
        let path = Path::new(&src).join(file);
        assert!(path.exists(), "exempted file {file} no longer exists");
        let text = std::fs::read_to_string(&path).expect("a readable source file");
        assert!(
            text.contains(&format!("fn {func}")),
            "exempted fn `{func}` no longer exists in {file}; a rename would \
             otherwise leave a live exemption pointing at nothing and quietly \
             re-cover whatever takes its place"
        );
        assert!(
            why.len() > 40,
            "exemption for {file}::{func} must state its reason, not assert one"
        );
    }
}
