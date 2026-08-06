//! A product cast to a wider type must be computed in the wider one.
//!
//! `(w * h) as usize` multiplies in whatever `w` and `h` already are — `u32`, for
//! every extent this device decodes — and widens the answer afterwards. The cast
//! is doing nothing: the overflow, if there is one, happened before it. The
//! spelling that works is `(w as usize) * (h as usize)`, or a `checked_mul` on a
//! `u64`, which is what [`crate::contract::extent::tight_image_bytes`] does.
//!
//! Both halves of the failure are bad and neither is loud:
//!
//! - **Debug**: an arithmetic overflow panics, which aborts the process. This
//!   device *is* the guest's GPU, so a panic reached from a decoded command is
//!   the guest losing its display, and every one of these is reached from a
//!   decoded command.
//! - **Release**: it wraps. `65536 * 65536` becomes `0`, a loop over it runs
//!   zero times, and a buffer allocated at full length goes out part-filled or
//!   unfilled. Nothing refuses, nothing counts it, and the guest reads back an
//!   image the device said it had drawn.
//!
//! # What this scan looks for, and why the shape is the filter
//!
//! A parenthesized expression, immediately cast, whose top level contains a
//! binary `*`. "Top level" is doing real work: `[a[i * 2], a[i * 2 + 1]]` is an
//! index inside brackets and not a product being widened, and requiring depth
//! zero drops it without an exemption. So does requiring spaces around the `*`,
//! which is how rustfmt writes a binary multiply and never how it writes a
//! dereference — `(*cursor + LEN) as u64` is a `+`, and a scan that matched a
//! bare `*` would call it a product.
//!
//! An inner ` as ` means the widening already happened at the operands, which is
//! the correct spelling, so those are not reported at all rather than being
//! exempted one by one.
//!
//! # What it cannot see
//!
//! A product that is never cast. `let n: usize = w * h;` on two `u32`s does not
//! compile, so the interesting version of that is `w * h` assigned to a `u32` and
//! used as a count, which is the same overflow with no cast to key on. Nothing
//! here finds it. The cast form is worth gating on its own because it is the one
//! that *looks* safe — a reader sees `as usize` and reads a widening.
//!
//! An integration test rather than a `#[cfg(test)]` module because it reads
//! source text and must run on every arm, including `backend-metal`, which this
//! development host can compile but cannot execute.

mod source_scan;
use source_scan::guest_facing_sources;

/// Why a product computed in the narrow type cannot overflow it.
#[allow(
    dead_code,
    reason = "Overflows is kept unused by the assertion below; the vocabulary is \
              offered to an author by the failure message"
)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Width {
    /// Both operands are bounded upstream, tightly enough that the product fits.
    /// The verdict must **name the bound**, because the safety is a runtime
    /// property of another check and nothing at the multiplication says so.
    RefusedUpstream,
    /// Neither operand is the guest's. A device counter, a host capability, a
    /// value this device chose — so the range is one this code controls.
    DeviceValues,
    /// The scan matched a product that is not an integer one. Floating-point
    /// multiplication does not overflow — it reaches an infinity — and Rust's
    /// float-to-integer `as` saturates rather than wrapping or trapping, so
    /// neither half of this test's failure mode exists. Classified rather than
    /// filtered out, for the reason all three bound scans carry a `NotA…`
    /// verdict: "the scan should not have flagged this" is worth writing down
    /// once instead of re-deriving it at every reading.
    NotAnIntegerProduct,
    /// The product can exceed the type it is computed in, and something the
    /// guest asked for gets lost or the process dies. **Forbidden**, and
    /// asserted absent below.
    Overflows,
}

/// Every narrow product widened after the fact, and why it fits.
///
/// Keyed by `(file, expression)`, so a second one on the same line moves the
/// count and fails rather than inheriting a verdict about the first.
const PRODUCTS: &[(&str, &str, Width, &str)] = &[
    (
        "reims-vgpu/src/contract/pixel_format.rs",
        "value * f64::from(UNORM8_MAX) + 0.5",
        Width::NotAnIntegerProduct,
        "`f64_to_unorm8`'s rounding step, guarded above by arms that return \
         UNORM8_MIN at or below 0.0 and UNORM8_MAX at or above 1.0, so the \
         multiplicand is in (0, 1) by the time it reaches here",
    ),
    (
        "reims-vgpu/src/contract/pixel_format.rs",
        "f * f32::from(UNORM8_MAX) + 0.5",
        Width::NotAnIntegerProduct,
        "the f32 twin of the line above, with the same clamped domain",
    ),
    (
        "reims-vgpu/src/runtime/drain/census.rs",
        "since_n * 1000",
        Width::DeviceValues,
        "the VBL delivery count in one census window, scaled to a rate. `since_n` \
         is a delta of this device's own `AtomicU64` counter against the previous \
         window and 1000 is the ms/s factor, so nothing the guest sends changes \
         it and a u64 has no reachable ceiling here",
    ),
    (
        "reims-vgpu/src/runtime/drain/mod.rs",
        "width * height",
        Width::RefusedUpstream,
        "the cursor glyph's texel count. Both axes are `ld16` reads, so they are \
         u16-ranged before anything else, and the geometry check twenty lines \
         above refuses `width > CURSOR_MAX_DIM || height > CURSOR_MAX_DIM` with \
         `cursor_glyph_fail reason=cursor_glyph_geom`. Either bound alone is \
         enough: 65535 * 65535 is still under u32::MAX, barely, and CURSOR_MAX_DIM \
         is far below that",
    ),
];

/// `(expr) as` on `line`, for every `expr` whose top level multiplies.
fn widened_products(line: &str) -> Vec<String> {
    let mut found = Vec::new();
    let mut from = 0;
    while let Some(at) = line[from..].find(") as ") {
        let close = from + at;
        from = close + 1;
        let before: Vec<char> = line[..close].chars().collect();
        let mut depth = 1i32;
        let mut open = before.len();
        while open > 0 {
            open -= 1;
            match before[open] {
                ')' => depth += 1,
                '(' => {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                }
                _ => {}
            }
        }
        if depth != 0 {
            continue;
        }
        let inner: String = before[open + 1..].iter().collect();
        // An operand that was already widened is the correct spelling.
        if inner.contains(" as ") {
            continue;
        }
        if has_top_level_product(&inner) {
            found.push(inner.trim().to_string());
        }
    }
    found
}

/// Whether `expr` multiplies outside any bracket or parenthesis of its own.
///
/// A binary `*` is spaced on both sides; a dereference is not, and an index's
/// arithmetic sits at a depth this refuses to look at.
fn has_top_level_product(expr: &str) -> bool {
    let chars: Vec<char> = expr.chars().collect();
    let mut depth = 0i32;
    for (i, &c) in chars.iter().enumerate() {
        match c {
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth -= 1,
            '*' if depth == 0 => {
                let spaced_before = i > 0 && chars[i - 1] == ' ';
                let spaced_after = chars.get(i + 1) == Some(&' ');
                if spaced_before && spaced_after {
                    return true;
                }
            }
            _ => {}
        }
    }
    false
}

fn sites() -> Vec<(String, usize, String)> {
    let mut out = Vec::new();
    for (file, text) in guest_facing_sources() {
        for (i, line) in text.lines().enumerate() {
            for expr in widened_products(line) {
                out.push((file.clone(), i + 1, expr));
            }
        }
    }
    out
}

#[test]
fn every_product_widened_after_the_fact_says_why_it_fits() {
    let found = sites();

    // The self-check every source scan in this directory carries: prove the scan
    // can see before believing what it cannot. Injected forms rather than tree
    // ones, so this half keeps working when the last real site is fixed.
    assert!(
        widened_products("let n = (w * h) as usize;") == vec!["w * h"],
        "the scan does not find the plainest form of what it is for"
    );
    assert!(
        widened_products("let n = (w as usize * h as usize) as u64;").is_empty(),
        "an already-widened operand is the correct spelling and must not be reported"
    );
    assert!(
        widened_products("let a = ([v[i * 2], v[i * 2 + 1]]) as u32;").is_empty(),
        "an index inside brackets is not a product being widened"
    );
    assert!(
        widened_products("let n = (*cursor + LEN) as u64;").is_empty(),
        "a dereference is not a binary multiply"
    );

    let unlisted: Vec<String> = found
        .iter()
        .filter(|(file, _, expr)| {
            !PRODUCTS
                .iter()
                .any(|(f, e, _, _)| f == file && *e == expr.as_str())
        })
        .map(|(file, line, expr)| format!("  {file}:{line}  ({expr}) as …"))
        .collect();
    assert!(
        unlisted.is_empty(),
        "a product is computed in the narrow type and widened afterwards, so the \
         cast cannot save it — it panics in a debug build and wraps in a release \
         one:\n{}\n\nWiden the operands — `(a as usize) * (b as usize)` — or reach \
         for `contract::extent`, which does it in u64 with a `checked_mul`. If it \
         genuinely cannot overflow, add a row to `PRODUCTS` naming the bound that \
         says so.",
        unlisted.join("\n")
    );

    let stale: Vec<&str> = PRODUCTS
        .iter()
        .filter(|(f, e, _, _)| !found.iter().any(|(file, _, expr)| file == f && expr == e))
        .map(|(_, e, _, _)| *e)
        .collect();
    assert!(
        stale.is_empty(),
        "these rows describe a product that is gone — it was widened, \
         consolidated, or deleted. Drop the row so the list keeps meaning \
         something: {stale:?}"
    );
}

/// No product may be one the guest can overflow.
///
/// Separate from the classification test on purpose, as the three bound scans
/// separate theirs: that one fails when nobody has answered, this one when
/// somebody has and the answer is that a decoded geometry can wrap an
/// arithmetic this device then trusts. The two need different messages because
/// they need different fixes — one is a line of prose, the other is a cast.
#[test]
fn no_widened_product_can_be_overflowed_by_a_guest() {
    let bad: Vec<&str> = PRODUCTS
        .iter()
        .filter(|(_, _, w, _)| *w == Width::Overflows)
        .map(|(_, e, _, _)| *e)
        .collect();
    assert!(
        bad.is_empty(),
        "a guest-reachable product overflows the type it is computed in: {bad:?}"
    );
}
