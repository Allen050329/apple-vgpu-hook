//! A walk this device stops early must say what it did not reach.
//!
//! The third direction of the standing goal, and the last one. Its two siblings
//! ask what a bound does to a collection this device *holds*:
//! `an_eviction_says_what_it_costs` for an entry removed to stay under a number,
//! `a_bounded_insert_says_what_it_drops` for one never recorded. This one asks
//! what a bound does to a run of data the device is *reading*:
//!
//! ```ignore
//! for attr in attrs.iter().take(REIMS_VGPU_METAL_MAX_ATTRS) { … }
//! let n = (stage_input.layout_count as usize).min(MAX_LAYOUTS);
//! ```
//!
//! Neither sibling can see this. Nothing is stored, so nothing is evicted, and
//! nothing is skipped on the way in — the collection is complete and correct and
//! the device reads part of it. That is the quietest of the three: a truncated
//! *read* leaves no trace anywhere, not in a counter, not in the shape of a
//! structure, not in a later lookup that misses.
//!
//! # What this asserts
//!
//! What both siblings assert: not that an answer is right, but that the question
//! was asked. Every walk in `src/` bounded by a capacity-shaped constant appears
//! in [`SKIPS`] with a verdict, and a new one fails until somebody writes the
//! line. [`Skip::LosesGuestWork`] is in the vocabulary so that answering
//! honestly produces a failing build.
//!
//! # Why the bound's *name* is the whole filter
//!
//! There is no receiver trick available here — the thing being walked and the
//! thing bounding it are unrelated expressions, unlike an insert whose gate
//! names the same collection. So the discriminator is the bound token alone:
//! `.take(n)` where `n` is a computed length is arithmetic, and `.take(MAX_FOO)`
//! is a policy. Sharing [`is_bound`]'s vocabulary with both siblings is what
//! makes that hold — the three cannot disagree about what a bound is called,
//! and a constant renamed out of the vocabulary disappears from all three at
//! once rather than from one.
//!
//! The cost of having only one filter is that this scan reports the *safe*
//! majority too: a walk whose bound the decoder already refused past is listed
//! here with the refusal named. That is deliberate. Those are exactly the
//! entries where the safety is a cross-module runtime invariant rather than a
//! local fact, and writing down which refusal holds it is the only thing that
//! notices when the refusal moves.
//!
//! # What this scan cannot see, measured rather than guessed
//!
//! Writing the verdicts first and then running the scan found one bound the
//! shared vocabulary misses: `drain::census`'s `DOORBELL_OFFSETS_REPORTED`,
//! which caps how many distinct doorbell offsets a census line names. It is a
//! bound by every meaning of the word and contains none of [`BOUND_WORDS`], so
//! nothing here matches it.
//!
//! It is left missed on purpose. Adding `REPORTED` to the vocabulary would
//! either fork it from the two sibling scans — and their agreement about what a
//! bound is called is what keeps a renamed constant from vanishing out of one
//! scan while staying in another — or widen all three for a class that by
//! construction bounds only a log line. The honest fix is to name the miss here
//! so the next reader does not conclude the scan is exhaustive.
//!
//! The general shape of the blind spot: a bound named for **what it limits**
//! rather than for **the fact that it limits**. If you add one, give it a `MAX`
//! or `CAP` in the name and all three scans pick it up for free.

mod source_scan;
use source_scan::guest_facing_sources;

/// What a walk that stops at its bound fails to reach.
#[allow(
    dead_code,
    reason = "LosesGuestWork is kept unused by the assertion below; the \
              vocabulary is offered to an author by the failure message"
)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Skip {
    /// Something upstream already refused any input longer than this bound, so
    /// the walk cannot stop early — it is a second expression of a refusal made
    /// elsewhere. The verdict must **name that refusal**, because the safety is
    /// a cross-module runtime property and nothing local says so.
    RefusedUpstream,
    /// The bound is the wire encoding's own width or mask, so no conforming
    /// stream can present more than it. Stronger than `RefusedUpstream`: no
    /// runtime check has to hold for it to be true.
    Unencodable,
    /// The walk is over a collection this device filled itself, to a size it
    /// chose, so the bound cannot cut guest data.
    DeviceFilled,
    /// The bound narrows a *report*: a hex dump, a census sample, a diagnostic
    /// tail. Truncating it costs a reading, never guest work — and where it
    /// matters the truncation says it happened.
    Observability,
    /// The value is being narrowed toward what the device can actually do — a
    /// negotiated version, a capability rung. Narrowing is the correct direction
    /// and the guest is told the result.
    Narrowing,
    /// The scan matched a cut that is not a capacity policy: the value is a
    /// count computed at runtime, so the walk reads exactly as far as it meant
    /// to. Classified rather than filtered out, for the reason both siblings
    /// carry `NotABound` and `NotAGate` — "the scan should not have flagged
    /// this" is worth writing down once instead of re-deriving at each reading.
    ///
    /// These exist because [`is_bound`] accepts a lowercase `cap`-shaped local,
    /// which both siblings measured as necessary: a real bound is very often a
    /// parameter called exactly `cap`. Tightening the vocabulary to silence
    /// these would silence real bounds in all three scans at once.
    NotAWalkBound,
    /// Guest work the device silently never read. **Forbidden**, and asserted
    /// absent below.
    LosesGuestWork,
}

/// Every capacity-bounded walk in the two guest-facing crates, and what
/// stopping there skips.
///
/// Keyed by `(file, bound)` with the number of sites that pair covers.
const SKIPS: &[(&str, &str, usize, Skip, &str)] = &[
    (
        "reims-vgpu/src/backend/metal/render.rs",
        "REIMS_VGPU_METAL_MAX_ATTRS",
        1,
        Skip::RefusedUpstream,
        "fill_render_pso_key's attribute walk. `parse_vertex_block` answers \
         DecodeStatus::ErrUnsupported(\"res_vertex_attr_count_over\") above \
         MAX_VERTEX_ATTRS rather than truncating, and a `const` assertion in \
         backend::metal::constants pins the two numbers equal — so `attrs` is \
         never longer than the take. That pin is load-bearing beyond this walk: \
         the key's hash and its PartialEq both index `0..attr_count` over arrays \
         this wide, so a longer `attrs` would abort rather than truncate. 31 is \
         MTLVertexDescriptor.attributes' own slot count",
    ),
    (
        "reims-vgpu/src/backend/metal/render.rs",
        "REIMS_VGPU_METAL_MAX_COLOR_RTS",
        2,
        Skip::RefusedUpstream,
        "the colour-attachment walk and the `color_count` that indexes with it. \
         Unlike attributes this one is `.min`ed at the same site, deliberately, \
         because it indexes the same arrays; the render entry point refuses a \
         longer colour list before reaching here",
    ),
    (
        "reims-vgpu/src/backend/metal/stage_input.rs",
        "REIMS_VGPU_COMPUTE_STAGE_INPUT_MAX_LAYOUTS",
        2,
        Skip::Unencodable,
        "pinned by a `const` assertion to MAX_COMPUTE_STAGE_INPUT_LAYOUTS, which \
         is COMPUTE_STAGE_INPUT_HEADER0_COUNT_MASK — the wire field's own width. \
         A guest cannot encode a 32nd layout. Same bound the insert-side scan \
         classifies Unencodable, reached from the Metal arm",
    ),
    (
        "reims-vgpu/src/backend/metal/stage_input.rs",
        "REIMS_VGPU_COMPUTE_STAGE_INPUT_MAX_ATTRIBUTES",
        1,
        Skip::Unencodable,
        "the attribute half of the same mask and the same `const` assertion",
    ),
    (
        "reims-vgpu/src/runtime/compute_exec/mod.rs",
        "REIMS_VGPU_COMPUTE_STAGE_INPUT_MAX_ATTRIBUTES",
        1,
        Skip::DeviceFilled,
        "walks the fixed-size array the decoder filled, not the guest's record; \
         the record's own over-cap count is what refuses the pipeline",
    ),
    (
        "reims-vgpu/src/runtime/compute_exec/mod.rs",
        "REIMS_VGPU_COMPUTE_STAGE_INPUT_MAX_LAYOUTS",
        1,
        Skip::DeviceFilled,
        "the layout half of the same array",
    ),
    (
        "reims-vgpu/src/backend/vulkan/engine/exec.rs",
        "MAX_SECONDARY_ATTACH",
        2,
        Skip::RefusedUpstream,
        "MRT secondary attachments; the draw entry point refuses a request naming \
         more before these walks run, and the bound's basis is a `const` \
         assertion at its own declaration rather than at the comparison — read \
         the declaration, not these sites",
    ),
    (
        "reims-vgpu/src/runtime/objects/mod.rs",
        "TYPE4_PLANE_CAP",
        2,
        Skip::RefusedUpstream,
        "type-4 plane walks. The header's plane_count is compared against \
         TYPE4_PLANE_CAP and the descriptor refused above it, so these two \
         `.min`s are that refusal restated where the array is indexed",
    ),
    (
        "reims-vgpu/src/runtime/objects/mod.rs",
        "HEX_MAX",
        1,
        Skip::Observability,
        "the undecoded-tail hex dump in the type4_desc_shape census line; bounds \
         a log line's length and nothing else",
    ),
    (
        "reims-vgpu/src/runtime/drain/mod.rs",
        "TAIL_DUMP_MAX",
        1,
        Skip::Observability,
        "how many trailing bytes a malformed-packet dump prints",
    ),
    (
        "reims-vgpu/src/backend/vulkan/caps/api_floor.rs",
        "MAX_USEFUL_API",
        1,
        Skip::Narrowing,
        "the loader's advertised Vulkan version narrowed to the highest this \
         device has a use for. An override may only narrow, never widen — this \
         is that rule applied to the API version itself",
    ),
    (
        "reims-vgpu/src/model/regs.rs",
        "PROTOCOL_VERSION_MAX",
        1,
        Skip::Narrowing,
        "protocol version negotiation: the guest asks, the device answers with \
         the lower of the two and the guest reads the answer back. Narrowing is \
         what negotiation *is*",
    ),
    (
        "reims-vgpu-wire/src/device_desc.rs",
        "TYPE4_PLANE_CAP",
        1,
        Skip::DeviceFilled,
        "the type-4 descriptor *builder*'s `Vec` reserve. It bounds the bytes the \
         builder sets aside in its own fixed array and nothing else — its doc is \
         explicit that `plane_count` is written through **unclamped**, including \
         values over the cap, because a corrupt descriptor is a thing the device \
         has to be tested against. So this caps the fixture's storage, never a \
         decoded record",
    ),
    (
        "reims-vgpu-wire/src/device_desc.rs",
        "TYPE4_BUILDER_CAP",
        1,
        Skip::DeviceFilled,
        "`with_len` clamping to the builder's own `[u8; TYPE4_BUILDER_CAP]`. The \
         array is the bound, so the clamp is a bounds check on this device's \
         storage rather than a policy about guest data",
    ),
    (
        "reims-vgpu-wire/src/device_desc.rs",
        "TYPE5_BUILDER_CAP",
        1,
        Skip::DeviceFilled,
        "the type-5 builder's `with_len`, same array and same argument",
    ),
    (
        "reims-vgpu/src/backend/vulkan/engine/window_present.rs",
        "caps_max",
        1,
        Skip::Narrowing,
        "swapchain_image_count clamping to the surface's reported \
         maxImageCount. Not a number this device chose — a surface that caps at \
         two cannot be argued with, and Vulkan requires the request to sit \
         inside the reported range. Nothing of the guest's is being read here",
    ),
    (
        "reims-vgpu/src/runtime/compute_exec/mod.rs",
        "over_cap",
        1,
        Skip::NotAWalkBound,
        "`(siblings.len() + 1).saturating_sub(STORAGE_RESIDENCY_WINDOWS_PER_MAPPING)` \
         — the number of entries to evict, not a bound on how far to read, so \
         the walk visits exactly the victims it computed. The eviction it \
         performs is the real event and `an_eviction_says_what_it_costs` carries \
         it, as Recomputable: dropping a residency-mirror entry sends the next \
         read back to the guest pages the writeback had just written",
    ),
];

/// Words that make an identifier a statement about how many of something is
/// allowed.
///
/// Deliberately identical to both siblings' vocabulary, so the three directions
/// cannot disagree about what a bound is called.
const BOUND_WORDS: &[&str] = &[
    "CAP", "MAX", "LIMIT", "BUDGET", "RING", "HISTORY", "KEYS", "PER_", "WINDOWS",
];

/// Ways a walk is cut short by a value rather than by the data running out.
const CUT: &[&str] = &[".take(", ".min("];

/// Whether `token` names how many of something is allowed.
fn is_bound(token: &str) -> bool {
    if token.is_empty() {
        return false;
    }
    let shouty = token.len() >= 3
        && token.chars().any(|c| c.is_ascii_uppercase())
        && token
            .chars()
            .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_');
    if shouty && BOUND_WORDS.iter().any(|w| token.contains(w)) {
        return true;
    }
    let lower = token.to_ascii_lowercase();
    matches!(lower.as_str(), "cap" | "capacity" | "limit" | "budget")
        || lower.contains("_cap")
        || lower.contains("cap_")
        || lower.contains("_limit")
        || lower.contains("_budget")
        || lower.contains("_max")
}

#[derive(Debug)]
struct Site {
    file: String,
    line: usize,
    bound: String,
}

/// The capacity constants a line cuts a walk with.
///
/// Only a bare path argument counts — `.min(a.len())` and `.take(over_cap)` are
/// arithmetic over a runtime value, not a policy, and the type's own `MAX` is
/// excluded by the `::`-qualified test so `u32::MAX` never reads as one.
fn cuts(line: &str) -> Vec<String> {
    let mut found = Vec::new();
    for cut in CUT {
        let mut from = 0;
        while let Some(at) = line[from..].find(cut) {
            let open = from + at + cut.len();
            from = open;
            let arg: String = line[open..]
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_' || *c == ':')
                .collect();
            // `u32::MAX` names a type's extreme, not this device's policy. Take
            // the last path segment and require the path to be a bare name.
            if arg.contains("::") {
                continue;
            }
            // The argument must be the whole argument: `MAX_FOO)` not `MAX_FOO -
            // 1)` or `MAX_FOO as usize)`, both of which are arithmetic.
            if !line[open + arg.len()..].starts_with(')') {
                continue;
            }
            if is_bound(&arg) {
                found.push(arg);
            }
        }
    }
    found
}

fn find_sites(sources: &[(String, String)]) -> Vec<Site> {
    let mut sites = Vec::new();
    for (file, text) in sources {
        for (i, line) in text.lines().enumerate() {
            for bound in cuts(line) {
                sites.push(Site {
                    file: file.clone(),
                    line: i + 1,
                    bound,
                });
            }
        }
    }
    sites
}

#[test]
fn every_capacity_bounded_walk_says_what_it_skips() {
    let sources = guest_facing_sources();

    let sites = find_sites(&sources);

    // The self-check every source scan in this directory carries: prove the scan
    // can see before believing what it cannot. Both `CUT` spellings are named,
    // and both backends, because the Vulkan and Metal sites are `#[cfg]`-ed out
    // of each other and only source text sees them together.
    for (file, bound, why) in [
        (
            "reims-vgpu/src/backend/metal/render.rs",
            "REIMS_VGPU_METAL_MAX_ATTRS",
            "a `.take(` on the Metal arm",
        ),
        (
            "reims-vgpu/src/backend/metal/stage_input.rs",
            "REIMS_VGPU_COMPUTE_STAGE_INPUT_MAX_LAYOUTS",
            "a `.min(` on the Metal arm",
        ),
        (
            "reims-vgpu/src/backend/vulkan/engine/exec.rs",
            "MAX_SECONDARY_ATTACH",
            "a `.take(` on the Vulkan arm",
        ),
        (
            "reims-vgpu-wire/src/device_desc.rs",
            "TYPE4_BUILDER_CAP",
            "the whole of `reims-vgpu-wire` — a second crate that decodes guest \
             bytes, and one a scan rooted at `reims-vgpu/src` reports as clean by \
             construction",
        ),
    ] {
        assert!(
            sites.iter().any(|s| s.file == file && s.bound == bound),
            "the scan cannot see {file}'s `{bound}`, which is its only cover for \
             {why}. Its silence about everything else therefore means \
             nothing.\n\nFound:\n{}",
            summarize(&sites)
        );
    }

    let mut unclassified = Vec::new();
    for site in &sites {
        let known = SKIPS
            .iter()
            .any(|(file, bound, _, _, _)| *file == site.file && *bound == site.bound);
        if !known {
            unclassified.push(format!(
                "{}:{} — cut at `{}`",
                site.file, site.line, site.bound
            ));
        }
    }
    assert!(
        unclassified.is_empty(),
        "these stop reading at a number and say nothing about what they did not \
         reach. A truncated read leaves no trace — no counter moves, no structure \
         changes shape, no later lookup misses — so this is the one of the three \
         bound directions that a boot can never surface.\n\nFor each: is the \
         bound the wire's own field width? `Unencodable`. Did something upstream \
         refuse a longer input — and *which* refusal? `RefusedUpstream`, and name \
         it, because that safety is a cross-module runtime property and this site \
         does not state it. Is the walk over an array the device filled? \
         `DeviceFilled`. Does it bound a log line or a census sample? \
         `Observability`. Is it narrowing a version or capability toward what the \
         device can do? `Narrowing`. If none of those hold, the guest sent bytes \
         this device never looked at, and `LosesGuestWork` will say so by \
         failing. Add a line to SKIPS in {}.\n\n{}",
        file!(),
        unclassified.join("\n")
    );

    let mut wrong = Vec::new();
    for (file, bound, expected, _, _) in SKIPS {
        let live = sites
            .iter()
            .filter(|s| s.file == *file && s.bound == *bound)
            .count();
        if live != *expected {
            wrong.push(format!(
                "{file} `{bound}`: SKIPS says {expected} site(s), the scan finds {live}"
            ));
        }
    }
    assert!(
        wrong.is_empty(),
        "SKIPS no longer describes this crate. A count that grew is a new walk \
         inheriting a verdict written about a different line; one that shrank is \
         a claim about code that is gone.\n\n{}\n\nCurrent sites:\n{}",
        wrong.join("\n"),
        summarize(&sites)
    );
}

/// No walk may be classified as silently skipping guest work.
#[test]
fn no_bounded_walk_is_allowed_to_lose_guest_work() {
    let losing: Vec<&str> = SKIPS
        .iter()
        .filter(|(_, _, _, skip, _)| *skip == Skip::LosesGuestWork)
        .map(|(file, ..)| *file)
        .collect();
    assert!(
        losing.is_empty(),
        "these read part of what the guest sent and execute as though that were \
         all of it. A GPU refuses a call it cannot honour; it does not honour the \
         first N of it. The fix is to read the whole run, or to refuse the \
         request and say which bound refused it.\n\n{}",
        losing.join("\n")
    );
}

fn summarize(sites: &[Site]) -> String {
    sites
        .iter()
        .map(|s| format!("  {}:{} cut={}", s.file, s.line, s.bound))
        .collect::<Vec<_>>()
        .join("\n")
}
