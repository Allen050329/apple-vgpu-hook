//! The Metal arm's unit tests do not run on a host that cannot link for macOS,
//! and nothing in a green run says so.
//!
//! This is worse than the wire-fixture gap, which at least reports `ignored`:
//! everything under `src/backend/metal/` is `cfg`-ed out of the arm a Linux host
//! can build, so those tests are not skipped, not ignored, and not counted —
//! they are simply absent, and the run reads exactly like a clean tree. The
//! cross-compiled clippy and `cargo check` arms *compile* them, which is why a
//! code warning there is still caught, but `cargo test --target
//! aarch64-apple-darwin … --no-run` fails at the **link** step (no Apple linker,
//! no macOS SDK), so no binary is ever produced and nobody on a Linux host has
//! executed one.
//!
//! # What this test does about it
//!
//! It cannot make them run. It makes the size of the hole a checked fact, in the
//! only place that can see it from either arm — the source text.
//!
//! Two things follow from pinning the count rather than merely reporting it.
//! Deleting a Metal-arm test on a Linux host stops being invisible: the number
//! moves and the Vulkan gate fails, which is the case AGENTS.md's "do not commit
//! a dropped test count without calling it out" exists for and the one arm where
//! nothing else could enforce it. And *adding* one fails too, on purpose: the
//! author is told, at the moment they write it, that the test they just wrote
//! will not run where they are.
//!
//! The friction is one line and it lands on exactly the person who should know.
//!
//! # Why this file is not `#[ignore]`d
//!
//! An ignored test moves the ignored count and prints its own name, which was
//! the first design. It reports the hole once and then reports it identically
//! forever — it cannot notice the hole changing size, which is the part a run
//! can act on.

mod source_scan;
use source_scan::{blank_comments, rust_sources, workspace_root};

/// `#[test]` functions under `src/backend/metal/`.
///
/// Counted from source, so this is the population on **both** arms rather than
/// whatever the current build happened to compile.
///
/// Update it in the same commit that changes the population, and say in that
/// commit's body that the tests concerned do not run on a non-Apple host — which
/// is the sentence this constant exists to make somebody write.
///
/// # It is 35 and not 39
///
/// AGENTS.md said 39 from the day the gap was written down. A bare
/// `grep -rh '#\[test\]' | wc -l` over that directory answers 39, and four of
/// those are inside comments — every one of them a sentence explaining why some
/// pin is a `const` assertion **rather than** a `#[test]`. So the number
/// describing the untested arm was itself produced by reading prose as code, and
/// blanking comments before counting is why this one is not.
///
/// It went 35 → 33 the moment this test existed, and that is the intended use:
/// `backend::hash` moved up out of the gated tree because it names nothing from
/// the `metal` crate, so its two tests now run on every arm instead of none.
/// Every further reduction should have that shape — a test that *runs* somewhere
/// — and not a deletion.
///
/// # It went 33 → 28
///
/// `backend::metal::mipmap` was six tests, and five of them never reached
/// `system_device()`: they walked an argument ladder — zero width, zero height,
/// a level count of one, an integer format, a short level 0 — and checked which
/// refusal came back. That is arithmetic over guest numbers, so it moved to
/// `contract::mipmap` along with `MetalMipmapError` itself, and it now runs on
/// every arm. The sixth stayed, because asking Metal to filter real pixels and
/// checking the colour survived is the one question no other host can answer.
///
/// That is the shape to copy, and the second time it has paid: the reduction
/// came from a *file* whose portable half was larger than its Metal half, not
/// from picking off individual tests. Look for those.
const METAL_ARM_TEST_FUNCTIONS: usize = 28;

#[test]
fn the_metal_arm_test_functions_are_counted_because_this_build_may_not_run_them() {
    let root = workspace_root();
    let metal = root.join("crates/reims-vgpu/src/backend/metal");
    assert!(
        metal.is_dir(),
        "src/backend/metal is gone, so this count is about nothing. Delete this \
         test in the same commit as the directory, or fix the path."
    );

    let mut per_file: Vec<(String, usize)> = Vec::new();
    let mut total = 0usize;
    for path in rust_sources(&metal) {
        let raw = std::fs::read_to_string(&path).expect("read source");
        // Comments are blanked because these files quote `#[test]` in prose —
        // and a scan that counts a doc comment is measuring the documentation.
        // Test modules are deliberately *not* blanked here: they are the whole
        // subject.
        let text = blank_comments(&raw);
        let n = text.matches("#[test]").count();
        if n == 0 {
            continue;
        }
        total += n;
        per_file.push((
            path.strip_prefix(&metal)
                .unwrap_or(&path)
                .to_string_lossy()
                .to_string(),
            n,
        ));
    }

    // Self-check before reporting, the same rule every source scan in this
    // directory carries: a count of zero here would "pass" the day the scan
    // broke, if the constant were ever edited to match it.
    assert!(
        per_file.len() >= 5,
        "the scan found test functions in only {} of the Metal arm's files, so it \
         is not seeing them and its number means nothing: {per_file:?}",
        per_file.len()
    );

    let breakdown = per_file
        .iter()
        .map(|(f, n)| format!("  {f}: {n}"))
        .collect::<Vec<_>>()
        .join("\n");
    assert_eq!(
        total, METAL_ARM_TEST_FUNCTIONS,
        "the Metal arm's test population changed, from {METAL_ARM_TEST_FUNCTIONS} \
         to {total}.\n\nOn a non-Apple host **none of these run** — they are \
         `cfg`-ed out of the arm that builds here, and the cross-compiled \
         `--no-run` fails at the link step — so neither their addition nor their \
         removal shows up in any test count on this machine. That is what this \
         constant is standing in for.\n\nIf you added one: it will not run where \
         you are; say so in the commit body and update \
         METAL_ARM_TEST_FUNCTIONS. If you removed one: say which, and why the \
         coverage it held is no longer needed.\n\nPer file:\n{breakdown}"
    );
}
