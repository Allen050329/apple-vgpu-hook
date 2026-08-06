//! A first-sighting latch may silence a line. It may not change what the
//! device does.
//!
//! A dedup latch is a process-global set of everything already reported this
//! boot, and the ones here hand the caller a `bool`. That return value is the
//! hazard: it is the only latch state a caller can branch on, so it is the only
//! one that can accidentally end up deciding something other than whether to
//! write a line.
//!
//! The failure it produces is unlike anything the bound scans model. Put a
//! refusal inside the latch and the **first** malformed record is refused and
//! every later one is executed — so the device is faithful exactly once, the
//! fail log contains a line saying the check works, and the counter that would
//! have shown the rest reads one. A test that drives the path once passes. It is
//! a correctness bug wearing the costume of a fixed one.
//!
//! # Why this exists
//!
//! Two commits on this branch converted a silent drop into a refusal, and both
//! had an emission already sited under a latch. In both, the obvious edit —
//! returning the error from inside the block that was already there — would have
//! produced exactly the bug above, and in both the fix was to move the refusal
//! out and leave the line in. Each carries a comment saying so
//! (`res_color_field_unread`, `res_color_entry_fields_short`). Twice in one
//! session is a shape, and a shape that has to be re-derived at every site by
//! whoever happens to be looking is what an instrument is for.
//!
//! `Emit::fail_once` is deliberately **not** in this population. It consumes
//! `self` and returns `()`, so it takes the latch and sends the line in one move
//! and there is no boolean for a caller to branch on. That is the shape to
//! prefer. A site needing the bare latch should say why; the two reasons in this
//! tree are a hot path that must not render the line eagerly, and a block that
//! reads extra state purely to enrich it.
//!
//! # Why the population is discovered rather than listed
//!
//! `crate::observe::first_sight` is not the only latch, and it is not most of
//! them. Six more are declared privately over their own sets —
//! `draw::degrade_log_first`, four `note_*` helpers in `blit_exec`, and
//! `census::view_swizzle_census`'s own module-local `first_sight` — while one
//! file inlines the insert and calls nothing at all
//! (`census::srgb_census::note_downgrade`).
//!
//! Scanning for the name `first_sight` alone finds 48 of the 81 sites. It misses
//! `blit_exec` and `draw/mod.rs` **entirely**, and `degrade_log_first`'s callers
//! sit in `draw/depth_stencil.rs`, a third file again — a helper declared in one
//! module and called from another is exactly the case a name filter cannot see,
//! and it is the same blind spot `a_bound_in_a_cut_is_named_like_one` was built
//! to close for bounds.
//!
//! So [`latch_names`] finds the declarations first and [`sites`] then hunts
//! calls to what it found. That ordering is load-bearing rather than tidy: three
//! of the seven names — `cursor_glyph_fail`, `note_copy_region_io`,
//! `note_repack_storage_assumed` — were missed by a hand search of this crate
//! for this exact class, and the scan turned them up on its first run. The
//! self-check below refuses to report anything until it has found more than one
//! declaration, because a scan that found only the seeded name would report the
//! whole population as adjudicated.
//!
//! # What this asserts
//!
//! What the bound scans assert: not that an answer is right, but that the
//! question was asked. Every call site appears in [`ROWS`] with a verdict, and a
//! new one fails until somebody writes the line. [`Guarded::Behaviour`] is in
//! the vocabulary so that answering honestly produces a failing build.
//!
//! An integration test rather than a `#[cfg(test)]` module because it reads
//! source text and must cover `backend-metal`, which this development host can
//! compile but cannot execute.

mod source_scan;
use source_scan::guest_facing_sources;

/// What the sites in a file decide.
#[allow(
    dead_code,
    reason = "Behaviour is kept unused by the assertion below; the vocabulary is \
              offered to an author by the failure message"
)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Guarded {
    /// Every site in the file guards a line and nothing else. Covers the two
    /// benign shapes: `if latch(..) { emit }` with the action outside it, and
    /// `if !latch(..) { return; }` at the head of a reporter whose entire body
    /// is the report and whose result no caller reads.
    EmitsOnly,
    /// As `EmitsOnly`, and at least one site additionally reads device state
    /// inside the block to enrich its line. The read is a pure query and its
    /// value reaches the line and nothing else; it sits under the latch so the
    /// walk costs once per distinct subject rather than once per sample.
    ReadsToReport,
    /// The site is the latch's own mechanism — its declaration, or a wrapper
    /// that takes it and sends in one move — rather than a user of it.
    TheLatchItself,
    /// The device does something different on the second occurrence than on the
    /// first. **Forbidden**, and asserted absent below.
    Behaviour,
}

/// Every file holding a dedup-latch call, the number it holds, and what they
/// decide.
///
/// Keyed by `(file, count)` rather than by line, like
/// `a_bounded_walk_says_what_it_skips` and unlike the two scans that key by
/// `file:line` — a line number moves whenever anything above it does, and
/// re-pointing a row teaches nobody anything. A *new* site in a file moves the
/// count and fails, which is the event worth catching.
const ROWS: &[(&str, usize, Guarded, &str)] = &[
    (
        "reims-vgpu/src/backend/vulkan/engine/context.rs",
        1,
        Guarded::ReadsToReport,
        "memory_type_for. `picked` is computed before the latch and returned \
         after it whatever the latch says; the block reads memory_types[i] and \
         its heap only to name the flags the pick landed on, which is not \
         derivable from source",
    ),
    (
        "reims-vgpu/src/backend/vulkan/engine/pools/submission_and_buffers.rs",
        1,
        Guarded::EmitsOnly,
        "note_readback_memory, an `if !latch { return; }` reporter whose whole \
         body is the line and whose result no caller reads",
    ),
    (
        "reims-vgpu/src/model/regs.rs",
        1,
        Guarded::EmitsOnly,
        "the out-of-range child channel line; the `false` this function answers \
         with sits after the block and is unconditional",
    ),
    (
        "reims-vgpu/src/observe/emit.rs",
        1,
        Guarded::TheLatchItself,
        "Emit::fail_once, which takes the latch and sends in one move. This is \
         the shape every other site should prefer, and the reason it is the only \
         one here that cannot be got wrong",
    ),
    (
        "reims-vgpu/src/runtime/blit_exec/mod.rs",
        9,
        Guarded::EmitsOnly,
        "five `note_*` reporters over their own sets, and their calls. Two answer \
         `bool` and both callers discard it — `let _ =` at the tex_wrong_type \
         site, and the t2t_overlap site returns BlitStatus::Overlap \
         unconditionally *after* the call rather than from inside it. \
         note_copy_region_io and note_repack_storage_assumed are both statement \
         position with the Err return and the bpp fallback after them",
    ),
    (
        "reims-vgpu/src/runtime/census/srgb_census.rs",
        1,
        Guarded::EmitsOnly,
        "note_downgrade inlines the insert into a local named `first_sight` and \
         branches on it to emit. The only site in the tree that spells the latch \
         as a variable, which is why the scan looks for the insert and not only \
         for a call",
    ),
    (
        "reims-vgpu/src/runtime/census/view_swizzle_census.rs",
        2,
        Guarded::EmitsOnly,
        "a module-local `first_sight` over a module-local set, so the two sites \
         dedup per (reason, texture_ref) independently of the global latch. Both \
         guard an Emit and nothing else",
    ),
    (
        "reims-vgpu/src/runtime/compute_exec/mod.rs",
        1,
        Guarded::EmitsOnly,
        "the unknown-dispatch-type line; the MTL_DISPATCH_TYPE_SERIAL this \
         function falls back to is returned after the block and unconditionally",
    ),
    (
        "reims-vgpu/src/runtime/decode/resource/mod.rs",
        10,
        Guarded::EmitsOnly,
        "the type-7 colour-attachment and vertex-descriptor reporters. Four sites \
         sit directly above a `return Err(..)` — res_color_slot_over, \
         res_color_write_mask_over, res_color_field_unread, \
         res_color_entry_fields_short — and in all four the return is outside \
         the block at the enclosing indent. These are the worked examples the \
         module doc names",
    ),
    (
        "reims-vgpu/src/runtime/drain/mod.rs",
        10,
        Guarded::EmitsOnly,
        "packet and doorbell reporters plus cursor_glyph_fail and its six \
         callers. cursor_glyph_fail is the best-shaped site in the tree and \
         worth copying: it consumes the latch internally and **always returns \
         `false`**, so `return cursor_glyph_fail(..)` reads as a latch deciding \
         a result and cannot be one. Its doc says so at the declaration",
    ),
    (
        "reims-vgpu/src/runtime/draw/metal_icb.rs",
        1,
        Guarded::EmitsOnly,
        "the icb_color_not_bgra8 line. The EncodeStatus::BadArgs below it belongs \
         to the geometry check that follows, not to the latch",
    ),
    (
        "reims-vgpu/src/runtime/draw/mod.rs",
        4,
        Guarded::EmitsOnly,
        "degrade_log_first's declaration and one caller. The declaration is the \
         latch helper this scan discovers and then hunts callers of, including \
         the two in draw/depth_stencil.rs that no name filter over this file \
         would have found",
    ),
    (
        "reims-vgpu/src/runtime/draw/depth_stencil.rs",
        2,
        Guarded::EmitsOnly,
        "two degrade_log_first calls guarding shader_state_degraded lines. The \
         attachment drop and the clear-fill they report both happen outside the \
         block, which is the point: the degradation is unconditional and only \
         its line is deduped",
    ),
    (
        "reims-vgpu/src/runtime/draw/render_target.rs",
        2,
        Guarded::EmitsOnly,
        "the base-format decline and the type-5 view divergence lines, both \
         emission only",
    ),
    (
        "reims-vgpu/src/runtime/draw/vulkan.rs",
        16,
        Guarded::ReadsToReport,
        "the largest population. Eleven are emission or reporter early returns; \
         the lin_rung_blank site calls surface_cache::gva_backing_state under \
         the latch, which takes &DeviceState and &H and only reads, so the walk \
         is once per distinct blank span rather than per sample and its answer \
         reaches the line alone",
    ),
    (
        "reims-vgpu/src/runtime/exec/mod.rs",
        1,
        Guarded::EmitsOnly,
        "render_set_pipeline_zero, latched as the second half of an `&&`. The \
         `acc.pipeline_ref = cmd.pipeline_ref` assignment it reports on is after \
         the block and unconditional",
    ),
    (
        "reims-vgpu/src/runtime/exec/report.rs",
        4,
        Guarded::EmitsOnly,
        "stream and ICB reporters, three global and one over a local set; this \
         file is reporting by definition and holds no device behaviour to gate",
    ),
    (
        "reims-vgpu/src/runtime/gva_mem.rs",
        1,
        Guarded::EmitsOnly,
        "note_read_refusal, an `if !latch { return; }` reporter; the refusal it \
         names was decided by its caller",
    ),
    (
        "reims-vgpu/src/runtime/gva_view.rs",
        2,
        Guarded::EmitsOnly,
        "the two fragmented-span lines. Both sit inside a `runs > 1` branch that \
         ends `return None`, and that return is outside the latch — so a \
         fragmented span is refused a contiguous view every time and says so \
         once",
    ),
    (
        "reims-vgpu/src/runtime/mapper/mod.rs",
        1,
        Guarded::EmitsOnly,
        "the mapper span-seen line, namespaced apart from the type-4 adoption \
         site's so the two paths do not share one latch",
    ),
    (
        "reims-vgpu/src/runtime/objects/mod.rs",
        5,
        Guarded::EmitsOnly,
        "object-list and type-4 claimant reporters; four are `note_*` early \
         returns and the fifth reads an object type into its own line",
    ),
    (
        "reims-vgpu/src/runtime/storage_flush/access.rs",
        1,
        Guarded::EmitsOnly,
        "the deferred-flush-held line, keyed per mapping id; the hold it reports \
         is decided by the caller and this only names the first mapping to take \
         it",
    ),
    (
        "reims-vgpu/src/runtime/storage_flush/report.rs",
        3,
        Guarded::EmitsOnly,
        "three reporters in a file that exists to report; no rail branches on \
         any of them",
    ),
    (
        "reims-vgpu/src/runtime/task_slot.rs",
        1,
        Guarded::EmitsOnly,
        "the task-slot decode line, keyed on the decline's own discriminant",
    ),
];

/// Names of functions that hand a caller a dedup latch's answer.
///
/// A `-> bool` whose body inserts into a set named for having been seen. That
/// is the shape of every one in this tree, and it is what makes the population
/// discoverable rather than transcribed — the two `blit_exec` helpers and
/// `degrade_log_first` are private, spelled nothing like `first_sight`, and
/// called from files that declare no latch at all.
fn latch_names(sources: &[(String, String)]) -> Vec<String> {
    let mut out = vec!["first_sight".to_string()];
    for (_, text) in sources {
        let lines: Vec<&str> = text.lines().collect();
        for (i, line) in lines.iter().enumerate() {
            let Some(rest) = line.trim_start().strip_prefix("fn ") else {
                continue;
            };
            let name: String = rest
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            if name.is_empty() {
                continue;
            }
            // The signature can wrap, so look at the declaration and the few
            // lines after it for the return type, then at the body for the set.
            let head = lines[i..lines.len().min(i + 12)].join("\n");
            let body = lines[i..lines.len().min(i + 40)].join("\n");
            if head.contains("-> bool") && body.contains(".insert(") && body.contains("SEEN") {
                out.push(name);
            }
        }
    }
    out.sort();
    out.dedup();
    out
}

/// Every call to one of `names`, plus every inline insert into a `*SEEN*` set
/// that is not inside one of those functions.
fn sites(text: &str, names: &[String]) -> usize {
    let lines: Vec<&str> = text.lines().collect();
    let mut n = 0;
    for (i, line) in lines.iter().enumerate() {
        for name in names {
            // A call, not a declaration and not a mention.
            let call = format!("{name}(");
            let mut from = 0;
            while let Some(at) = line[from..].find(&call) {
                let start = from + at;
                from = start + call.len();
                let before = &line[..start];
                if before.ends_with("fn ") {
                    continue;
                }
                if before
                    .chars()
                    .next_back()
                    .is_some_and(|c| c.is_alphanumeric() || c == '_')
                {
                    continue; // span_first_sight_key, not first_sight
                }
                n += 1;
            }
        }
        // The inline spelling: a `*SEEN*` set inserted into within a few lines,
        // where those lines are not the body of a latch function already
        // counted at its call sites.
        if line.contains("SEEN") && !line.trim_start().starts_with("static ") {
            let window = lines[i..lines.len().min(i + 4)].join("\n");
            let back = lines[i.saturating_sub(8)..i].join("\n");
            let in_latch = names.iter().any(|nm| back.contains(&format!("fn {nm}(")));
            if window.contains(".insert(") && !in_latch {
                n += 1;
            }
        }
    }
    n
}

#[test]
fn every_dedup_latch_says_what_it_guards() {
    let sources = guest_facing_sources();
    let names = latch_names(&sources);

    // Self-check before believing anything, the rule every source scan in this
    // directory carries. Two separate ways this scan can silently answer
    // "everything is adjudicated": finding no latch declarations beyond the one
    // it was seeded with, and finding no call sites.
    assert!(
        names.len() >= 3,
        "the scan found only these latch declarations, so it is not seeing the \
         private ones and cannot have found their callers: {names:?}"
    );

    let mut found: Vec<(String, usize)> = sources
        .iter()
        .map(|(path, text)| (path.clone(), sites(text, &names)))
        .filter(|(_, n)| *n > 0)
        .collect();
    found.sort();

    let total: usize = found.iter().map(|(_, n)| n).sum();
    assert!(
        total >= 40 && found.len() >= 15,
        "the scan found {total} latch sites in {} files, which is too few to be \
         reading them: {found:?}",
        found.len()
    );

    let missing: Vec<String> = found
        .iter()
        .filter(|(f, n)| !ROWS.iter().any(|(rf, rn, _, _)| rf == f && rn == n))
        .map(|(f, n)| format!("  {f}  ({n} site(s))"))
        .collect();
    assert!(
        missing.is_empty(),
        "a dedup latch is not adjudicated, or a file's site count changed. A \
         latch may decide whether a line is written and nothing else: if the \
         device behaves differently on the second occurrence than on the first, \
         it is faithful exactly once and the log says the check works. Move the \
         action out of the block and leave the line in — `res_color_field_unread` \
         in decode/resource is the worked example — then add a row to ROWS.\n\n\
         Latch functions the scan is hunting calls to: {names:?}\n\n{}",
        missing.join("\n")
    );

    let stale: Vec<&str> = ROWS
        .iter()
        .filter(|(f, n, _, _)| !found.iter().any(|(ff, nn)| ff == f && nn == n))
        .map(|(f, _, _, _)| *f)
        .collect();
    assert!(
        stale.is_empty(),
        "a verdict names a file the scan no longer finds, or whose site count \
         moved. Re-read it rather than re-pointing it:\n{}",
        stale.join("\n")
    );
}

/// The forbidden verdict is absent. This is why the vocabulary carries it:
/// classifying a site honestly fails the build here rather than tempting the
/// author into one of the safe words.
#[test]
fn no_latch_decides_what_the_device_does() {
    let gating: Vec<&str> = ROWS
        .iter()
        .filter(|(_, _, g, _)| *g == Guarded::Behaviour)
        .map(|(f, _, _, _)| *f)
        .collect();
    assert!(
        gating.is_empty(),
        "these files make the device behave differently the second time a \
         condition occurs. The first record is handled correctly and every later \
         one is not, which is worse than never having checked:\n{}",
        gating.join("\n")
    );
}

/// Every verdict says why, in more than a word.
#[test]
fn every_verdict_says_why() {
    for (file, _, _, why) in ROWS {
        assert!(
            why.len() > 60,
            "{file}'s verdict does not explain itself: {why:?}"
        );
    }
}
