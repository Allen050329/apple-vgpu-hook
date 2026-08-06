//! A bound that refuses a guest request must say whose limit the number is.
//!
//! Three scans already adjudicate three ways a cap can cost guest work:
//! `an_eviction_says_what_it_costs` for a cap that drops an entry the device had
//! already admitted, `a_bounded_insert_says_what_it_drops` for one that stops an
//! entry being recorded, `a_bounded_walk_says_what_it_skips` for one that stops
//! a walk before the guest's data runs out. All three ask the same question —
//! *what does the guest lose* — and all three are about a bound applied
//! **quietly**.
//!
//! This is the fourth position and the loudest one: `if guest_count > BOUND {
//! return Err(…) }`. Nothing is silently dropped, the refusal is named and
//! fail-visible, and that is exactly why it was never scanned: a loud refusal
//! reads as correct behaviour. It is correct only if the number is not this
//! device's own. A real GPU refuses a well-formed request for one reason, that
//! its memory is exhausted; every other refusal it makes is the API's own rule,
//! published in a limits table the guest can read. A bound this implementation
//! picked because it had to pick something is a request the guest is entitled to
//! make, will not have served, and has no way to anticipate.
//!
//! So the rule is not "do not refuse". It is: **name the limit's owner**, and
//! put the evidence at the constant's declaration, where the next reader of the
//! constant meets it — not at the comparison, where "this array is eight wide"
//! is true of whatever number somebody wrote.
//!
//! # The verdict that must not appear
//!
//! [`Verdict::DeviceChoice`] is declared and asserted absent. It exists so an
//! honest answer fails the build: a site that genuinely has no owner but this
//! device cannot be labelled without turning the suite red, which is the only
//! thing that keeps a verdict table a measurement instead of a signature page.
//! The sibling scans carry `LosesGuestWork` for the same reason.
//!
//! # What the first sweep found
//!
//! Two phantoms and one loose number, which is the argument for the scan.
//!
//! `contract::gva`'s `MAX_SPAN_PAGES` said in its own doc that a longer span
//! was declined; it was a constant compared against itself through a `Geometry`
//! field that `wire_geometry` dropped before the walk. It is gone.
//! `REIMS_VGPU_METAL_MAX_SAMPLERS` was the one argument-table size with no
//! reachable derivation, and now pins to the serializer capture that measured
//! it. And the two viewport bounds turned out to guard a list this device builds
//! with at most one entry — see [`Verdict::NeverReached`], which is where the
//! interesting half of this population is.
//!
//! # What this reads, and what it therefore cannot see
//!
//! A comparison against a constant [`source_scan::is_bound`] accepts, inside the
//! condition of an `if` whose body refuses. "Refuses" is a vocabulary
//! ([`REFUSALS`]), and a body that declines through a spelling not in it drops
//! out of the population silently — the failure mode every scan in this
//! directory has, and the reason [`the_scan_can_see_the_sites_it_is_about`]
//! asserts four known sites are found before this file's silence about anything
//! else means a thing.
//!
//! Three things are deliberately out of scope, and the first is a line worth
//! stating precisely. **Only a `SCREAMING_SNAKE_CASE` name is adjudicated**,
//! which is narrower than [`source_scan::is_bound`] on its own — that
//! vocabulary also accepts a lowercase `cap`, `capacity` or `_max`, because in
//! the walk position a bound really is often a parameter spelled `cap`. In
//! *this* position it is not: a lowercase one is a runtime value, and a runtime
//! value is by definition not a number this device chose. Measured rather than
//! assumed — dropping the shouty requirement adds exactly three sites, and all
//! three are that: `exec.rs`'s and `exec_compute.rs`'s `lod_min > lod_max`,
//! which compares two guest fields against each other, and `drain`'s
//! `total_size > ring_capacity`, which measures a guest packet against the ring
//! the guest itself sized. None is a limit with an owner to name.
//!
//! Second, a refusal driven by a host capability reading is the host refusing
//! rather than this device. Third, a `while i < BOUND` loop is the walk
//! position, which has its own scan.
//!
//! An integration test rather than a `#[cfg(test)]` module because it reads
//! source text and must run on every arm, including `backend-metal`, which this
//! development host can compile but cannot execute — and which is where half of
//! this population lives.

mod source_scan;
use source_scan::{close_brace, guest_facing_sources, is_bound, is_shouty};

/// Ways an `if` body says the request is not being served.
///
/// Read against the body of the `if` the comparison guards, after
/// [`source_scan::guest_facing_sources`] has blanked comments — so a doc comment
/// using one of these words cannot enrol a site, and a refusal slug written as a
/// string literal still can.
///
/// `::Err` is here for the status enums that are not `Result`. `gva_resolve`
/// answers `ResolveStatus::ErrUnsupportedGeometry`, and without this entry both
/// of its geometry bounds fell out of the population entirely — which is how
/// this vocabulary was measured too short the first time it was written.
const REFUSALS: &[&str] = &[
    "return Err(",
    "Err(Status",
    "Status::",
    "::Err",
    "refus",
    "Refus",
    "decline",
    "Decline",
];

/// Whose limit the number is.
///
/// Not a severity scale. Every variant but the last says the refusal is the one
/// the hardware, the protocol or the host would have made anyway.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Verdict {
    /// The external API states this limit. Metal's argument-table widths, its
    /// eight colour attachments, the depth at which a page-table walk has
    /// covered the whole address space either guest can form. Passing the value
    /// through instead of refusing is a process-aborting exception in Metal's
    /// case, not a status anything could decline.
    ContractLimit,
    /// The number is the width of the field or array in Apple's serialized
    /// record that carries the count, so no record Apple emits can exceed it.
    /// The refusal is the decoder distrusting bytes it has not yet checked.
    WireField,
    /// A bound on host memory. The one refusal the standing goal permits without
    /// qualification: a GPU saying its memory is full. The reason must say what
    /// the caller does instead, because "refuses the fast rail and takes the
    /// slow one" and "refuses the draw" are different answers.
    Allocation,
    /// The host GPU has failed, and no value of this bound would serve the
    /// request. The reason must say why raising it changes nothing.
    HostFailure,
    /// The comparison is against a constant the vocabulary accepts but which
    /// bounds no capacity — an enum ordinal, a header length, the minimum size a
    /// record must reach before it can be read at all. Refusing a short record
    /// is decode validity.
    NotABound,
    /// The comparison cannot fire, because the value it tests is produced by
    /// this device rather than by the guest and is narrower by construction.
    ///
    /// **The reason must name what makes it unreachable**, because that is
    /// usually where the guest work actually goes. Both viewport bounds are
    /// here, and what makes them unreachable is that the device models a single
    /// viewport and counts the rest of a plural record as a loss — a bound that
    /// cannot fire in front of a truncation that already did.
    NeverReached,
    /// A limit this device picked. **Must not appear.** A guest request refused
    /// for a number nobody outside this repository knows about is guest work
    /// lost, however loudly it is logged.
    DeviceChoice,
}

/// One adjudicated site.
struct Row {
    at: &'static str,
    bound: &'static str,
    verdict: Verdict,
    why: &'static str,
}

/// Every site the scan finds, with a written verdict for each.
///
/// Keyed by `file:line`, which is what the scan reports, so a line moving fails
/// this test rather than silently re-pointing a verdict at a different site.
/// That is the intended cost: re-reading a bound when its neighbourhood changes
/// is the whole point of writing the verdict down.
const ROWS: &[Row] = &[
    Row {
        at: "reims-vgpu/src/backend/metal/compute.rs:514",
        bound: "REIMS_VGPU_METAL_MAX_SAMPLERS",
        verdict: Verdict::ContractLimit,
        why: "Metal's sampler argument table, pinned at the declaration to \
              wire::ops::bind_limit::SAMPLER, which measured Apple's serializer \
              truncating a plural sampler bind at 16 per stage.",
    },
    Row {
        at: "reims-vgpu/src/backend/metal/compute.rs:1041",
        bound: "REIMS_VGPU_METAL_MAX_TEXTURES",
        verdict: Verdict::ContractLimit,
        why: "Metal's texture argument table, pinned to draw::MAX_TEXTURE_BIND_SLOTS, \
              which is the same 128 the serializer truncates a plural texture bind at.",
    },
    Row {
        at: "reims-vgpu/src/backend/metal/render.rs:370",
        bound: "REIMS_VGPU_METAL_MAX_BUFFERS",
        verdict: Verdict::ContractLimit,
        why: "Metal's buffer argument table, pinned to draw::MAX_BUFFER_BIND_SLOTS; \
              setVertexBuffer: at an index past it throws rather than returning.",
    },
    Row {
        at: "reims-vgpu/src/backend/metal/render.rs:406",
        bound: "REIMS_VGPU_METAL_MAX_ATTRS",
        verdict: Verdict::ContractLimit,
        why: "MTLVertexDescriptor.attributes is a 31-slot array; pinned to the \
              decoder's MAX_VERTEX_ATTRS, which is what keeps the key arrays and \
              the refusal the same width.",
    },
    Row {
        at: "reims-vgpu/src/backend/metal/render.rs:419",
        bound: "REIMS_VGPU_METAL_MAX_ATTRS",
        verdict: Verdict::ContractLimit,
        why: "The same 31-slot array, indexed by an attribute's own location \
              rather than by the count — a subscript MTLVertexDescriptor does \
              not have.",
    },
    Row {
        at: "reims-vgpu/src/backend/metal/render.rs:480",
        bound: "MTL_VERTEX_STEP_FUNCTION_PER_INSTANCE",
        verdict: Verdict::NotABound,
        why: "An MTLVertexStepFunction ordinal, not a capacity. It is in this \
              population only because the shared vocabulary's PER_ entry matches \
              inside PER_INSTANCE.",
    },
    Row {
        at: "reims-vgpu/src/backend/metal/render.rs:1653",
        bound: "REIMS_VGPU_METAL_MAX_COLOR_RTS",
        verdict: Verdict::WireField,
        why: "Metal's eight colour attachments, held equal by a const assertion to \
              PASS_MAX_COLOR_ATTACHMENTS, the width of the colour-slot array in \
              Apple's serialized render-pass record.",
    },
    Row {
        at: "reims-vgpu/src/backend/metal/render.rs:1663",
        bound: "REIMS_VGPU_METAL_MAX_COLOR_RTS",
        verdict: Verdict::WireField,
        why: "The same eight, tested against a slot number rather than a count: a \
              slot at or above it names an attachment the wire record has no room \
              to have carried.",
    },
    Row {
        at: "reims-vgpu/src/backend/metal/render.rs:1748",
        bound: "REIMS_VGPU_BACKEND_MAX_VIEWPORTS",
        verdict: Verdict::NeverReached,
        why: "The list is built in draw::mod from DrawEncodeRequest::viewport, an \
              Option, so it holds nought or one. The extra viewports were dropped at \
              decode and counted as render_extra_viewports_dropped. Raising the \
              carriage to an array would render the same pixels: selecting a \
              viewport past 0 needs a shader writing the view index, which arrives \
              as setVertexAmplificationCount: and is itself dropped \
              (render_vertex_amplification_dropped). The gap is multi-view \
              rendering, not this bound.",
    },
    Row {
        at: "reims-vgpu/src/backend/metal/render.rs:1754",
        bound: "REIMS_VGPU_BACKEND_MAX_SCISSORS",
        verdict: Verdict::NeverReached,
        why: "The scissor twin of the viewport row above, unreachable for the same \
              reason and blocked behind the same feature: Metal's setScissorRects: \
              is one rect per viewport, so rects past the first are dropped and \
              counted as render_extra_scissors_dropped for want of multi-view, not \
              for want of room.",
    },
    Row {
        at: "reims-vgpu/src/backend/metal/stage_input.rs:125",
        bound: "REIMS_VGPU_COMPUTE_STAGE_INPUT_MAX_ATTRIBUTES",
        verdict: Verdict::ContractLimit,
        why: "MTLStageInputOutputDescriptor.attributes is a 31-slot array, and it \
              is also the width of the mirror array this descriptor is decoded \
              into, so the refusal is what keeps the loop below in bounds.",
    },
    Row {
        at: "reims-vgpu/src/backend/metal/stage_input.rs:131",
        bound: "REIMS_VGPU_COMPUTE_STAGE_INPUT_MAX_LAYOUTS",
        verdict: Verdict::ContractLimit,
        why: "MTLStageInputOutputDescriptor.layouts is the matching 31-slot array; \
              same derivation and same mirror-array width as the attribute row.",
    },
    Row {
        at: "reims-vgpu/src/backend/metal/stage_input.rs:255",
        bound: "REIMS_VGPU_METAL_MAX_ATTRS",
        verdict: Verdict::ContractLimit,
        why: "An attribute location indexing the 31-wide seen-set and, past it, \
              MTLVertexDescriptor's own attributes array.",
    },
    Row {
        at: "reims-vgpu/src/backend/vulkan/engine/context.rs:939",
        bound: "MAX_DEVICE_RECREATES",
        verdict: Verdict::HostFailure,
        why: "Consecutive device rebuilds with no draw completing between them. \
              note_work_completed resets it, so reaching the bound means the host \
              GPU has been rebuilt three times and executed nothing on any of \
              them — a larger number buys another spin, not a served request.",
    },
    Row {
        at: "reims-vgpu/src/backend/vulkan/engine/context.rs:988",
        bound: "MAX_DEVICE_RECREATES",
        verdict: Verdict::HostFailure,
        why: "The same budget checked on the recreate path rather than the ensure \
              path, so a caller that reaches try_recreate directly gets the same \
              answer as one that goes through ensure.",
    },
    Row {
        at: "reims-vgpu/src/backend/vulkan/engine/dmabuf.rs:475",
        bound: "MAX_IMPORTED_BYTES",
        verdict: Verdict::Allocation,
        why: "Guest pages held pinned through dma-bufs, which stop being swappable \
              for as long as they are held. Exceeding it refuses the import, not \
              the draw: the caller falls back to its CPU gather, which is slower \
              and correct.",
    },
    Row {
        at: "reims-vgpu/src/backend/vulkan/engine/exec.rs:1230",
        bound: "MAX_SECONDARY_ATTACH",
        verdict: Verdict::WireField,
        why: "Colour attachments past the primary, pinned by a const assertion to \
              PASS_MAX_COLOR_ATTACHMENTS minus one — so it is Apple's render-pass \
              array width counted from the other end.",
    },
    Row {
        at: "reims-vgpu/src/contract/gva_resolve.rs:274",
        bound: "MAX_DEPTH",
        verdict: Verdict::ContractLimit,
        why: "Four levels is the first depth covering a 48-bit virtual address on \
              both page geometries, which is the whole address space either guest \
              can form. A deeper table describes addresses no guest can name.",
    },
    Row {
        at: "reims-vgpu/src/contract/iosurface_pages.rs:631",
        bound: "MAPPER_REQUEST_ENTRY_LEN",
        verdict: Verdict::NotABound,
        why: "The size one mapper request entry occupies, tested as a minimum \
              length before the entry is read. In this population only because \
              the vocabulary's PER_ entry matches inside MAPPER_.",
    },
    Row {
        at: "reims-vgpu/src/runtime/decode/resource/mod.rs:1848",
        bound: "MAX_VERTEX_LAYOUTS",
        verdict: Verdict::ContractLimit,
        why: "MTLVertexDescriptor.layouts is a 31-slot array, so a layout naming a \
              buffer index at or above it cannot be built by any Metal caller.",
    },
    Row {
        at: "reims-vgpu/src/runtime/decode/resource/mod.rs:1876",
        bound: "MAX_VERTEX_ATTRS",
        verdict: Verdict::ContractLimit,
        why: "MTLVertexDescriptor.attributes is a 31-slot array, so a descriptor \
              declaring more is malformed rather than something this device chose \
              not to read. It is also what stops the backend's key arrays being \
              indexed off the end.",
    },
    Row {
        at: "reims-vgpu/src/runtime/compute_exec/mod.rs:318",
        bound: "MAX_COMPUTE_BUFFER_SLOTS",
        verdict: Verdict::ContractLimit,
        why: "Metal's compute buffer argument table, 31, which is also exactly \
              Apple's serializer's own buffer bound. The dispatch is refused \
              rather than run with the binding absent — see \
              `a_bind_past_the_argument_table_refuses_the_dispatch`.",
    },
    Row {
        at: "reims-vgpu/src/runtime/compute_exec/mod.rs:365",
        bound: "MAX_COMPUTE_TEXTURE_SLOTS",
        verdict: Verdict::ContractLimit,
        why: "Metal's compute texture argument table, 128, matching Apple's \
              serializer. The narrower 31 this rail once refused at was the \
              descriptor binding band, which `spirv_bind::widen_sampled_bands` \
              widened to `[32,160)`; the constant's own doc records that.",
    },
    Row {
        at: "reims-vgpu/src/runtime/compute_exec/mod.rs:397",
        bound: "MAX_COMPUTE_SAMPLER_SLOTS",
        verdict: Verdict::ContractLimit,
        why: "Metal's sampler argument table, which is genuinely 16 — measured \
              by asking Apple's serializer for 200 and reading back 16, the \
              same basis `wire::ops::bind_limit::SAMPLER` carries.",
    },
    Row {
        at: "reims-vgpu/src/runtime/decode/resource/mod.rs:2471",
        bound: "MAX_COLOR_ATTACHMENTS",
        verdict: Verdict::WireField,
        why: "The declared colour-attachment count against the width of the slot \
              array in Apple's serialized record, which is the same eight Metal \
              itself stops at.",
    },
    Row {
        at: "reims-vgpu/src/runtime/mipmap.rs:222",
        bound: "TEXTURE_MAX_MIP_LEVELS",
        verdict: Verdict::ContractLimit,
        why: "Sixteen levels put 2^15 = 32768 pixels in the base's largest \
              dimension, which is above every Metal texture-size limit — so a \
              deeper pyramid describes a texture no Metal device would create.",
    },
    Row {
        at: "reims-vgpu/src/runtime/mtlb.rs:239",
        bound: "WRAPPER_HEADER_LEN",
        verdict: Verdict::NotABound,
        why: "The length of the wrapper header, tested as a minimum before the \
              header is read. In this population only because the vocabulary's \
              PER_ entry matches inside WRAPPER_.",
    },
    Row {
        at: "reims-vgpu-wire/src/page_table.rs:191",
        bound: "MAX_DEPTH",
        verdict: Verdict::ContractLimit,
        why: "The wire crate's own declaration of the same four levels the device \
              re-exports; this is the walker refusing a geometry before walking it.",
    },
];

/// The population, as `(file:line, bound name)`.
fn refusal_bounds() -> Vec<(String, String)> {
    let mut found = Vec::new();
    for (path, text) in guest_facing_sources() {
        let chars: Vec<char> = text.chars().collect();
        let lines: Vec<&str> = text.lines().collect();
        let mut starts = Vec::with_capacity(lines.len());
        let mut at = 0usize;
        for line in &lines {
            starts.push(at);
            at += line.chars().count() + 1;
        }
        for (i, line) in lines.iter().enumerate() {
            let trimmed = line.trim_start();
            if trimmed.starts_with("//") || trimmed.starts_with('*') {
                continue;
            }
            let mut names: Vec<String> = compared_constants(line)
                .into_iter()
                .filter(|n| is_shouty(n) && is_bound(n))
                .collect();
            names.sort();
            names.dedup();
            if names.is_empty() {
                continue;
            }
            let Some(anchor) = enclosing_if(&lines, i) else {
                continue;
            };
            let Some(body) = if_body(&chars, &starts, &lines, anchor) else {
                continue;
            };
            if !REFUSALS.iter().any(|r| body.contains(r)) {
                continue;
            }
            for name in names {
                found.push((format!("{path}:{}", i + 1), name));
            }
        }
    }
    found.sort();
    found.dedup();
    found
}

/// Every bare identifier standing on either side of a `<`, `>`, `<=` or `>=` on
/// `line`.
///
/// Shifts are not comparisons — `u32::MAX >> BITS` would otherwise enrol
/// `u32::MAX` — and `->` and `=>` are not operators at all. A `::`-carrying path
/// is dropped for the reason [`is_bound`] gives: a type's extreme is not this
/// device's policy. So is anything after a `.`, because `self.max_depth` is the
/// value being bounded rather than the bound.
fn compared_constants(line: &str) -> Vec<String> {
    let chars: Vec<char> = line.chars().collect();
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < chars.len() {
        let c = chars[i];
        if c != '<' && c != '>' {
            i += 1;
            continue;
        }
        if chars.get(i + 1) == Some(&c) {
            i += 2;
            continue;
        }
        if i > 0 && matches!(chars[i - 1], '<' | '>' | '-' | '=') {
            i += 1;
            continue;
        }
        let mut after = i + 1;
        if chars.get(after) == Some(&'=') {
            after += 1;
        }
        let mut k = after;
        while chars.get(k) == Some(&' ') {
            k += 1;
        }
        let start = k;
        while k < chars.len() && (chars[k].is_alphanumeric() || chars[k] == '_') {
            k += 1;
        }
        if k > start && !(chars.get(k) == Some(&':') && chars.get(k + 1) == Some(&':')) {
            out.push(chars[start..k].iter().collect());
        }
        let mut k = i;
        while k > 0 && chars[k - 1] == ' ' {
            k -= 1;
        }
        let end = k;
        while k > 0 && (chars[k - 1].is_alphanumeric() || chars[k - 1] == '_') {
            k -= 1;
        }
        if end > k && !(k > 0 && matches!(chars[k - 1], ':' | '.')) {
            out.push(chars[k..end].iter().collect());
        }
        i = after;
    }
    out
}

/// How far above a comparison the `if` that owns it may sit.
///
/// `gva_resolve::validate_geometry` is why this is not one: it refuses on an
/// eleven-clause `||` chain, and its bound is past the ninth line of it.
const IF_ANCHOR_REACH: usize = 16;

/// Line index of the `if` whose condition contains line `at`.
fn enclosing_if(lines: &[&str], at: usize) -> Option<usize> {
    let floor = at.saturating_sub(IF_ANCHOR_REACH);
    (floor..=at).rev().find(|&j| {
        let t = lines[j].trim_start();
        t.starts_with("if ") || t.starts_with("} else if ") || t.starts_with("else if ")
    })
}

/// How far past the `if` its opening brace may sit — the same chain, read from
/// the other end.
const IF_BRACE_REACH: usize = 20;

/// The braced body of the `if` anchored at line `anchor`.
fn if_body(chars: &[char], starts: &[usize], lines: &[&str], anchor: usize) -> Option<String> {
    let last = (anchor + IF_BRACE_REACH).min(lines.len());
    for (j, line) in lines.iter().enumerate().take(last).skip(anchor) {
        let Some(col) = line.chars().position(|c| c == '{') else {
            continue;
        };
        let open = starts[j] + col;
        let end = close_brace(chars, open);
        return Some(chars[open..end].iter().collect());
    }
    None
}

/// The population is adjudicated, exactly.
#[test]
fn every_refusal_bound_has_a_verdict() {
    let found = refusal_bounds();
    let missing: Vec<String> = found
        .iter()
        .filter(|(at, bound)| !ROWS.iter().any(|r| r.at == at && r.bound == bound))
        .map(|(at, bound)| format!("{at}  {bound}"))
        .collect();
    assert!(
        missing.is_empty(),
        "a bound refuses a guest request and nothing says whose limit the number \
         is. Add a row to ROWS in this file, with the evidence at the constant's \
         own declaration:\n{}",
        missing.join("\n")
    );

    let stale: Vec<String> = ROWS
        .iter()
        .filter(|r| !found.iter().any(|(at, b)| at == r.at && b == r.bound))
        .map(|r| format!("{}  {}", r.at, r.bound))
        .collect();
    assert!(
        stale.is_empty(),
        "a verdict names a site the scan no longer finds — the line moved, the \
         refusal changed shape, or the bound is gone. Re-read it rather than \
         re-pointing it:\n{}",
        stale.join("\n")
    );
}

/// No site is adjudicated as this device's own number.
#[test]
fn no_refusal_is_this_devices_own_number() {
    let guilty: Vec<&str> = ROWS
        .iter()
        .filter(|r| r.verdict == Verdict::DeviceChoice)
        .map(|r| r.at)
        .collect();
    assert!(
        guilty.is_empty(),
        "a guest request is refused for a number this device invented. Derive it \
         from the API contract and say so at the declaration, or stop \
         refusing:\n{}",
        guilty.join("\n")
    );
}

/// Every verdict carries a reason, and a `NeverReached` one names what makes it
/// unreachable.
///
/// The second half is the load-bearing one. `NeverReached` is the verdict a site
/// gets when nothing can reach it, and a site nothing can reach is usually
/// standing behind something that already took the guest's work — so a bare
/// "cannot happen" would file the interesting half of the finding under the
/// boring half.
#[test]
fn every_verdict_says_why() {
    for row in ROWS {
        assert!(
            row.why.len() >= 40,
            "{} {}: a verdict needs a reason long enough to be checked",
            row.at,
            row.bound
        );
        if row.verdict == Verdict::NeverReached {
            assert!(
                row.why.contains("drop")
                    || row.why.contains("Option")
                    || row.why.contains("construction")
                    || row.why.contains("truncat"),
                "{} {}: a NeverReached verdict must name what makes it unreachable",
                row.at,
                row.bound
            );
        }
    }
}

/// The scan can see the sites it is about.
///
/// Every filter here — the operator walk, the `if` anchor, the brace reach, the
/// refusal vocabulary — can go wrong in the direction that reports a clean tree.
/// These four were read by hand and each exercises a different one: a plain
/// one-line `if`, a cast on the left of the operator, a `>=` against an index
/// rather than a count, and the multi-line `||` chain that answers through a
/// status enum instead of a `Result`. If the scan stops finding any of them, its
/// silence about everything else means nothing.
#[test]
fn the_scan_can_see_the_sites_it_is_about() {
    let found = refusal_bounds();
    for (file, bound) in [
        (
            "reims-vgpu/src/backend/metal/render.rs",
            "REIMS_VGPU_BACKEND_MAX_VIEWPORTS",
        ),
        (
            "reims-vgpu/src/backend/metal/stage_input.rs",
            "REIMS_VGPU_COMPUTE_STAGE_INPUT_MAX_ATTRIBUTES",
        ),
        (
            "reims-vgpu/src/runtime/decode/resource/mod.rs",
            "MAX_VERTEX_LAYOUTS",
        ),
        ("reims-vgpu/src/contract/gva_resolve.rs", "MAX_DEPTH"),
    ] {
        assert!(
            found.iter().any(|(at, b)| at.starts_with(file) && b == bound),
            "the scan no longer finds {bound} in {file}; its silence proves nothing until it does"
        );
    }
}
