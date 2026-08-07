//! A loss reported only as a counter still has to say what it costs.
//!
//! `a_decline_says_whether_the_guest_lost_work` is the census of what this
//! device does to guest work, and its population is **types**: everything
//! implementing `observe::Decline`. That is most of the surface and it is not
//! all of it. A loss can also be reported as a bare
//! `drain::note_store_route("…_dropped")` — a name and a count, no type, and so
//! no row in that census and no line on the fail channel either.
//!
//! Thirty slugs in this crate are spelled that way, and twenty-five of them
//! name a decoded guest command or a decoded piece of render state that this
//! device did not apply. They are exactly the `Loss::ExecutedModified`
//! class, wearing a spelling the census cannot see — and there are more than
//! three times as many of them as the eight that census counts.
//!
//! # What this does not do
//!
//! It does not merge them into that census, and `EXECUTED_MODIFIED_CEILING`
//! does not count them. The ceiling is a ratchet over rows that were each read
//! against their emitter, and **four of the original sixteen were wrong when
//! written** — the verdict held every time, the stated reason and the stated
//! retirement path did not. Folding seventeen more in on the strength of one
//! reading would put the tree's most load-bearing number on the weakest
//! evidence in it.
//!
//! Most of them are spelled inside a `match` arm rather than as a literal
//! argument, which is why [`counter_arguments`] balances parentheses instead of
//! reading a line. A line-based grep finds about half and reports a clean
//! population; that is how this file's first draft undercounted by half.
//!
//! So this is the second half of one question asked in its own file, and the
//! honest reading of the standing goal's size is the two populations together.
//! The vocabularies are deliberately different: this one classifies by *what the
//! count is for*, because that is what the sites here differ in.
//!
//! # Why a counter and not a line, at all
//!
//! Because these fire per record on a per-draw path. `note_unimplemented_render_opcode`
//! measured ~2620 fail lines from six app launches before it was deduped, and
//! these sit in the same arms. A count is the right instrument for "how much of
//! this is happening"; what it cannot do is tell a reader what it cost, which is
//! what the rows below supply.
//!
//! # The trap this is built around
//!
//! Every one of them is argued at its own site, most with a boot reading beside
//! it. That is not the same as the population being adjudicated: one added
//! tomorrow inherits nothing, appears in no census, and
//! reads to a log reader as one more counter among the hundreds this device
//! publishes. The five bound scans exist for the same reason, and this file
//! borrows their shape — a verdict per site, a forbidden word in the vocabulary,
//! and a self-check that refuses to report a clean population it cannot see.

mod source_scan;
use source_scan::guest_facing_sources;

/// What a counted-only loss costs.
#[allow(
    dead_code,
    reason = "Unadjudicated is kept unused by the assertion below; the vocabulary \
              is offered to an author by the failure message"
)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Counted {
    /// Decoded render state this device does not apply, so the draw runs with
    /// the API default where the guest asked for something else. The counter
    /// fires only when the guest asked for a *non-default*, which is what makes
    /// a zero meaningful rather than merely absent. This is the
    /// `ExecutedModified` class in counter form, and the `why` must name the
    /// default it falls back to.
    StateNotApplied,
    /// Decoded guest *work* — a draw, a dispatch — that this device does not
    /// execute. Worse than state left at a default, and each of these keeps a
    /// deduped fail line beside the count for that reason; the count is here
    /// because the line is latched per opcode and cannot say how much.
    WorkNotExecuted,
    /// The count is the denominator of a measurement rather than an alarm. Some
    /// other instrument decides whether anything was lost, and the `why` must
    /// name it.
    Denominator,
    /// The count is raised beside a typed decline that the loss census already
    /// adjudicates, so the row there is the verdict and this is its volume.
    BesideATypedDecline,
    /// Nothing was lost: the count is of a housekeeping step, or of guest state
    /// that reaches the guest by another route.
    NotALoss,
    /// Nobody has read the emitter. **Forbidden**, and asserted absent below.
    Unadjudicated,
}

/// Every loss-named slug that reaches a counter and nothing else, and what the
/// count costs the guest.
///
/// Keyed by the slug, which is the one name that cannot move when a line does.
const ROWS: &[(&str, Counted, &str)] = &[
    (
        "render_store_action_options_dropped",
        Counted::StateNotApplied,
        "the options sibling of the same record, and now the only half of it \
         still dropped: the action is applied. MTLStoreActionOptions carries \
         CustomSamplePositions, which asks that a multisample resolve use the \
         pass's programmable sample positions — state this device also does not \
         set (render_pass_sample_positions_dropped), and which means nothing at \
         one sample per pixel. So the loss is real but is the same loss as its \
         sibling row rather than a second one",
    ),
    (
        "render_vertex_amplification_dropped",
        Counted::StateNotApplied,
        "amplification makes one vertex invocation produce several views, so a \
         dropped record renders one view where the guest asked for many. Both \
         wire forms have an API default meaning no amplification — count 1, \
         mode 0 — and the counter fires only past it",
    ),
    (
        "render_visibility_result_mode_dropped",
        Counted::StateNotApplied,
        "MTLVisibilityResultModeDisabled is 0, so a zero mode is a guest \
         disarming a query this rail never armed and is not counted. A non-zero \
         one asked for an occlusion result that no draw will write",
    ),
    (
        "render_pass_visibility_buffer_dropped",
        Counted::StateNotApplied,
        "the pass names a visibility result buffer and no draw in it writes an \
         occlusion count, so the buffer keeps whatever it held. Zero on every \
         driven boot recorded, beside a pass-extent count of 1575 from the same \
         record — which is what makes this zero measured rather than unreached",
    ),
    (
        "render_pass_array_length_dropped",
        Counted::StateNotApplied,
        "layered rendering: the pass declares a render-target array length above \
         1 and this device renders one layer. Decoded from the same record as \
         the extent count beside it, so it shares that record's proof of being \
         reached",
    ),
    (
        "render_pass_rate_map_dropped",
        Counted::StateNotApplied,
        "a rasterization rate map changes where fragments land, and this device \
         rasterizes at the attachment's own rate. One of six RenderPassProperty \
         records, each counted under its own name because they are not equally \
         costly to drop; all six sit behind a serializer capability that \
         defaults off",
    ),
    (
        "render_tessellation_factor_buffer_dropped",
        Counted::StateNotApplied,
        "the state half of a tessellated draw. Unapplied like the patch draws \
         themselves, so this should track `render_draw_patches_dropped`: the two \
         diverging would mean one of the two arms is wrong rather than that the \
         guest changed behaviour",
    ),
    (
        "render_line_width_dropped",
        Counted::StateNotApplied,
        "a line width other than 1.0, compared exactly rather than with a \
         tolerance because the question is whether the guest wrote *the* \
         literal, not whether it wrote something close to it",
    ),
    (
        "render_tile_dispatch_dropped",
        Counted::WorkNotExecuted,
        "a tile shader the guest asked to run, so work rather than state, and it \
         keeps the deduped fail line as well. The one healthy zero is a \
         genuinely empty grid: Metal dispatches nothing when any dimension of \
         threadsPerTile is 0, so those are excluded rather than counted, which \
         keeps this a loss estimate and not a record count",
    ),
    (
        "render_pass_target_extent_unapplied",
        Counted::Denominator,
        "the denominator of `note_pass_extent_coverage`'s bands, not an alarm. \
         Those bands answer whether ignoring the extent lost anything, and they \
         have: `pass_extent_full` takes 11826 of 11827 scored passes on \
         arm64/Vulkan and every scored pass on x86. The extent is the attachment \
         restated. This is the only one of the seventeen that fires in volume — \
         1615 in a window — and it is the one that costs nothing",
    ),
    (
        "render_color_subresource_unsupported",
        Counted::BesideATypedDecline,
        "raised by `note_color_subresource_unsupported`, which returns the \
         `StreamDrawDrop::ColorSubresourceUnsupported` its caller records as \
         `StreamRefusal::Pass` — so the pass is refused and the loss census \
         holds the verdict. The count is its volume",
    ),
    (
        "validity_windows_dropped",
        Counted::NotALoss,
        "`drop_stale_windows` takes a mapping's pending deferred-flush windows \
         when the licence says a later writeback has superseded them. The bytes \
         those windows described are written by the landing that superseded \
         them, so nothing is unwritten; the count is how many were folded. \
         Reported with `note_store_route_n` rather than per window, which is why \
         it is a volume and not an event",
    ),
    (
        "render_tessellation_scale_dropped",
        Counted::StateNotApplied,
        "a tessellation factor scale other than 1.0, the SetFloatState sibling \
         of the line width above and compared exactly for the same reason. It \
         scales geometry this device does not tessellate at all, so it tracks \
         the patch draws rather than standing alone",
    ),
    (
        "render_draw_patches_dropped",
        Counted::WorkNotExecuted,
        "a tessellated draw: geometry the guest asked for and did not get. \
         Keeps `note_unimplemented_render_opcode`'s deduped line beside the \
         count, and should track `render_tessellation_factor_buffer_dropped` — \
         the two diverging means one of the arms is wrong",
    ),
    (
        "render_draw_patches_indirect_dropped",
        Counted::WorkNotExecuted,
        "the indirect form of the same patch draw, counted apart because it \
         carries the indirect buffer's problem on top of the tessellation one \
         and so is not retired by the same work",
    ),
    (
        "render_pass_imageblock_dropped",
        Counted::StateNotApplied,
        "one of the six RenderPassProperty records. Imageblock sample length is \
         tile-shader pass geometry, and this device has no tile executor at all, \
         so it is dropped with the tile binds rather than with the raster state",
    ),
    (
        "render_pass_raster_sample_count_dropped",
        Counted::StateNotApplied,
        "the default raster sample count: how many fragments the rasterizer \
         produces per pixel. This device rasterizes at one sample, so a guest \
         asking for more gets an unmultisampled pass",
    ),
    (
        "render_pass_sample_positions_dropped",
        Counted::StateNotApplied,
        "programmable sample positions change *where* fragments land inside a \
         pixel. Counted apart from the sample count beside it because the two \
         are separately costly to drop and separately implementable",
    ),
    (
        "render_pass_threadgroup_memory_dropped",
        Counted::StateNotApplied,
        "tile-shader threadgroup memory length, the third of the tile-shaped \
         pass properties. Nothing allocates it because nothing dispatches a \
         tile shader to use it",
    ),
    (
        "render_pass_tile_size_dropped",
        Counted::StateNotApplied,
        "the pass's tile width and height, which sizes the tile grid this device \
         does not dispatch over. Behind the same serializer capability as its \
         five siblings, all of which default off",
    ),
    (
        "render_tile_buffer_bind_dropped",
        Counted::StateNotApplied,
        "a bind against the tile buffer argument table, which this device does \
         not have. Counted unconditionally because a bind has no default to sit \
         at. Not routed into the vertex or fragment table on purpose: that would \
         not be a partial implementation, it would be a wrong one",
    ),
    (
        "render_tile_texture_bind_dropped",
        Counted::StateNotApplied,
        "the texture half of the same table, counted apart because a tile \
         texture bind is a sampled attachment and a tile buffer bind is \
         imageblock storage — they are not interchangeable when the work is \
         costed",
    ),
    (
        "render_tile_sampler_bind_dropped",
        Counted::StateNotApplied,
        "the sampler half of the same table, split from the other two for the \
         same costing reason",
    ),
    (
        "render_tile_threadgroup_memory_dropped",
        Counted::StateNotApplied,
        "imageblock memory rather than an argument-table slot: the tile shader's \
         scratch storage, priced on its own rather than with the binds it sits \
         beside",
    ),
    (
        "mrt_secondary_dropped",
        Counted::BesideATypedDecline,
        "the `dropped` half of the MRT census, raised when an MRT draw arrived \
         and every secondary attachment was refused. `MrtDrop` carries the \
         verdict in the loss census; this and `mrt_secondary_built` beside it \
         are what separate 'we render every attachment' from 'we render none', \
         which the `mrt_drop_*` reasons cannot say because the whole feature is \
         silent when no MRT draw arrives",
    ),
    (
        "replace_physical_dropped",
        Counted::NotALoss,
        "a ReplacePhysical packet invalidated a mapping's page list, and `had` \
         says there were pages to invalidate. Dropping them is the guest's own \
         request — it has just handed the device a new plan for those pages — so \
         the count is of the request being carried out. The deferred windows \
         riding the old plan are taken separately just above, by \
         `drop_windows`, so that loss has its own name rather than disappearing \
         into a generation mismatch",
    ),
];

/// Index of the `)` closing the `(` at `open`.
///
/// `source_scan::close_brace` is the sibling of this and is brace-specific; a
/// counter's argument can be a whole `match` expression, so the parenthesis is
/// what has to be balanced and the braces inside it come along for the ride.
fn close_paren(chars: &[char], open: usize) -> usize {
    let mut depth = 0usize;
    for (i, c) in chars.iter().enumerate().skip(open) {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    return i;
                }
            }
            _ => {}
        }
    }
    chars.len() - 1
}

/// The text of every `note_store_route`/`note_store_route_n` argument list.
fn counter_arguments(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    for call in ["note_store_route(", "note_store_route_n("] {
        let needle: Vec<char> = call.chars().collect();
        let mut i = 0;
        while i + needle.len() <= chars.len() {
            if chars[i..i + needle.len()] == needle[..] {
                let open = i + needle.len() - 1;
                let end = close_paren(&chars, open);
                out.push(chars[open..=end.min(chars.len() - 1)].iter().collect());
                i = open;
            }
            i += 1;
        }
    }
    out
}

/// String literals in `text` whose name asserts that something was lost.
fn loss_named_literals(text: &str) -> Vec<String> {
    const SUFFIXES: [&str; 7] = [
        "_dropped",
        "_unapplied",
        "_unsupported",
        "_ignored",
        "_skipped",
        "_degraded",
        "_truncated",
    ];
    let mut out = Vec::new();
    for piece in text.split('"').skip(1).step_by(2) {
        if piece.len() < 4
            || !piece
                .chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
        {
            continue;
        }
        if SUFFIXES.iter().any(|s| piece.ends_with(s)) {
            out.push(piece.to_string());
        }
    }
    out
}

/// Slugs that name a loss and appear **only** inside a counter call.
///
/// The "only" is the whole discriminator. A slug that also appears in a
/// `Decline::slug`, a `Status::args`, or a `format!` reaches a reader some other
/// way and is somebody else's population; one that appears nowhere but here is
/// a name, a number, and no account of what it cost.
fn counter_only_slugs() -> Vec<String> {
    let sources = guest_facing_sources();
    let mut everywhere: Vec<String> = Vec::new();
    let mut in_counters: Vec<String> = Vec::new();
    for (_, text) in &sources {
        everywhere.extend(loss_named_literals(text));
        for arg in counter_arguments(text) {
            in_counters.extend(loss_named_literals(&arg));
        }
    }
    let mut out: Vec<String> = in_counters
        .iter()
        .filter(|s| {
            let total = everywhere.iter().filter(|e| e == s).count();
            let counted = in_counters.iter().filter(|e| e == s).count();
            total == counted
        })
        .cloned()
        .collect();
    out.sort();
    out.dedup();
    out
}

#[test]
fn every_counted_loss_says_what_it_costs() {
    let found = counter_only_slugs();

    // Self-check before believing anything, the rule every source scan in this
    // directory carries: a scan that matched no counter argument would report
    // an empty population as fully adjudicated.
    assert!(
        found.len() >= 26,
        "the scan found only {} counter-only loss slugs, so it is not parsing \
         `note_store_route` arguments and its verdict means nothing: {found:?}",
        found.len()
    );

    let missing: Vec<&String> = found
        .iter()
        .filter(|s| !ROWS.iter().any(|(rs, _, _)| *rs == s.as_str()))
        .collect();
    assert!(
        missing.is_empty(),
        "a slug names a loss, reaches a counter, and reaches nothing else — no \
         `Decline` type, so no row in `a_decline_says_whether_the_guest_lost_work`, \
         and no line on the fail channel. Read the arm and write down what the \
         guest is left with, then add a row to ROWS in this file:\n{missing:?}"
    );

    let stale: Vec<&str> = ROWS
        .iter()
        .filter(|(s, _, _)| !found.iter().any(|f| f == s))
        .map(|(s, _, _)| *s)
        .collect();
    assert!(
        stale.is_empty(),
        "a verdict names a slug the scan no longer finds. Either it was retired \
         — delete the row and say so — or it grew a second spelling and belongs \
         to another population now:\n{}",
        stale.join("\n")
    );
}

/// The forbidden verdict is absent.
#[test]
fn no_counted_loss_is_unread() {
    let unread: Vec<&str> = ROWS
        .iter()
        .filter(|(_, c, _)| *c == Counted::Unadjudicated)
        .map(|(s, _, _)| *s)
        .collect();
    assert!(
        unread.is_empty(),
        "these counters report a loss nobody has priced:\n{}",
        unread.join("\n")
    );
}

/// The two classes that cost the guest something are the ones worth counting,
/// and the count is pinned so that adding one is a decision rather than a diff.
///
/// **Lower this when you retire one**, exactly as
/// `EXECUTED_MODIFIED_CEILING` asks. Retiring means applying the state or
/// executing the work — not renaming the slug, which would move it out of this
/// population without changing anything the guest sees.
#[test]
fn the_counted_loss_census_only_shrinks() {
    const COUNTED_LOSS_CEILING: usize = 21;
    let losses = ROWS
        .iter()
        .filter(|(_, c, _)| matches!(c, Counted::StateNotApplied | Counted::WorkNotExecuted))
        .count();
    assert!(
        losses <= COUNTED_LOSS_CEILING,
        "{losses} counters now report guest work this device did not do, above \
         the {COUNTED_LOSS_CEILING} this file was written against. Adding one is \
         allowed and raising the ceiling is the way to do it, in a commit that \
         says which state or which command stopped being applied."
    );
}

/// Every verdict says why, in more than a word.
#[test]
fn every_verdict_says_why() {
    for (slug, _, why) in ROWS {
        assert!(
            why.len() > 60,
            "{slug}'s verdict does not explain itself: {why:?}"
        );
    }
}
