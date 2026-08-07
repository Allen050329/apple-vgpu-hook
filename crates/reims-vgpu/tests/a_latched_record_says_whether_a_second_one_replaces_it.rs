//! A slot that holds one decoded record must say what a second one does to it.
//!
//! The five bound scans in this directory all find their population by looking
//! for a **number**: a `MAX`, a `CAP`, a `.take(n)`, a shift width. There is a
//! bound with no number at all, and none of them can see it —
//!
//! ```ignore
//! acc.execute_icb = Some(RenderIcbExecute { .. });   // capacity: one
//! ```
//!
//! `executeCommandsInBuffer:` is *work*. Metal's ordinary indirect-command
//! shape is one buffer per object batch, so several in one render encoder is
//! the expected case — and that field held the last one. The first ICB's
//! commands never ran, and because there was no constant to name, no counter
//! to read and no line to grep, nothing anywhere said so. It was found by
//! reading the struct, which is exactly the way `AGENTS.md` says not to hold a
//! bound class.
//!
//! # The distinction this file is entirely about
//!
//! Most of these fields *should* hold one. `cull_mode`, `viewport`,
//! `blend_color`, `pipeline_ref` are encoder **state**, and a second
//! `setCullMode:` genuinely replaces the first — a list there would be a bug in
//! the other direction. So this cannot be a scan for "an `Option` in an
//! accumulator"; the answer differs per field and only a human knows it.
//!
//! The question that separates them: **if the guest sends this record twice,
//! does Metal do the thing twice?** If yes, the field is work and one slot
//! loses a call. If no, the field is state and one slot is the contract.
//!
//! So this file asks for the answer to be written down, once per field, and
//! fails when a field is added without one. [`Second::LosesTheFirst`] exists
//! for the same reason the sibling scans keep their forbidden verdict: an
//! author who answers honestly gets a failing build naming the architecture
//! rather than being nudged into a gentler word.
//!
//! # How the population is found, and the hole that was closed first
//!
//! Two seed routes, unioned, then closed under composition:
//!
//! - a `struct` whose name ends in `Accum`;
//! - the type of a `&mut`/`&` function parameter named `acc`;
//! - **and then**, to a fixpoint, any struct holding a field whose type is
//!   already in the set.
//!
//! The two seeds alone repeat the naming hole `a_bound_in_a_cut_is_named_like_one`
//! exists to hold shut: a struct renamed `RenderBatch` vanishes from the first,
//! and one only ever constructed inline vanishes from the second. The closure is
//! what actually caught `ComputeSegment` — it is a per-segment accumulator, it
//! is named for neither route, and it reaches this population only because it
//! holds a `ComputeAccum`. Each route must find something or the test refuses to
//! report, because an empty scan and a clean crate look identical.

mod source_scan;
use source_scan::{blank_comments, guest_facing_sources};

/// What a second record of the same kind does to a field that holds one.
#[allow(
    dead_code,
    reason = "LosesTheFirst is kept unused by the assertion below; the whole \
              vocabulary is what the failure message offers an author"
)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Second {
    /// Encoder **state**. A second record replaces the first because that is
    /// what the Metal selector means, so one slot is the contract rather than a
    /// bound. The `why` must name the selector.
    ReplacesIt,
    /// A keyed table. A second record at the same index replaces that index and
    /// one at another index lands beside it — the Metal `atIndex:` binds. The
    /// `why` must name what bounds the index space, because *that* number is a
    /// bound and belongs to the scans that hunt for one.
    ReplacesItsSlot,
    /// A collection. Every record is kept, so there is no capacity to lose one
    /// to. The `why` must say what bounds the collection, and "the stream's own
    /// byte count" is an answer.
    AddsToIt,
    /// A counter, a monotone flag, or an encoder opened on demand. A second
    /// record folds into what is already there rather than replacing it, so
    /// nothing is held that could be lost.
    FoldsIntoIt,
    /// The field is itself one of the structs in this population, adjudicated
    /// field by field under its own name. Nothing is latched here.
    HoldsAnAccumulator,
    /// Staging for the record being decoded right now: written and read on the
    /// same record, so a second record never meets the first's value. The `why`
    /// must name the consumer that empties it.
    ConsumedByItsOwnRecord,
    /// A remembered refusal or an intentional first-wins latch, deliberately not
    /// replaced. The `why` must name the retirement path, or say why none is
    /// needed — `a_remembered_refusal_says_whether_it_can_go_stale` holds the
    /// other half of that question.
    StickyOnPurpose,
    /// The field holds guest **work** and holds one, so a second record drops
    /// the first. **Forbidden**, and asserted absent below.
    LosesTheFirst,
}

/// Every field of every struct that accumulates decoded guest records, and what
/// a second record of that kind does to it.
///
/// Keyed by `(struct, field)`. A field renamed is a new row rather than an
/// inherited verdict, which is the point — the verdict is about what the guest
/// record means, and a rename is usually where that changed.
const VERDICTS: &[(&str, &str, Second, &str)] = &[
    // --- StreamAccum: one serialized Metal render pass -------------------
    (
        "StreamAccum",
        "pipeline_ref",
        Second::ReplacesIt,
        "setRenderPipelineState:. One PSO is bound at a time and the last one \
         bound is what a draw uses",
    ),
    (
        "StreamAccum",
        "clears",
        Second::AddsToIt,
        "one entry per colour attachment the pass descriptor declared as Clear. \
         Bounded by the attachment count the same descriptor carries, not by \
         anything this crate chose",
    ),
    (
        "StreamAccum",
        "color_slots",
        Second::AddsToIt,
        "the pass's colour attachments, slot index beside each. Same bound as \
         `clears`: the descriptor says how many there are",
    ),
    (
        "StreamAccum",
        "color_targets",
        Second::AddsToIt,
        "the texture refs of those attachments, kept apart so `dirty_color_targets` \
         can walk them without the attachment bodies. Bounded by `color_slots`",
    ),
    (
        "StreamAccum",
        "draws",
        Second::AddsToIt,
        "every draw in the pass, in stream order, because a Metal render encoder \
         executes all of them. A `MAX_DRAWS_PER_STREAM = 64` truncation used to \
         stand here and is gone; what bounds it now is the stream, since a draw \
         record has a minimum encoded length and the bytes are already in memory",
    ),
    (
        "StreamAccum",
        "saw_draw",
        Second::FoldsIntoIt,
        "a monotone flag: the pass either has draws in it or does not, and a \
         second draw cannot make it more true",
    ),
    (
        "StreamAccum",
        "execute_icb",
        Second::AddsToIt,
        "`executeCommandsInBuffer:` is work, and one buffer per object batch is \
         Metal's ordinary shape, so a pass may ask for several. This is the \
         field the whole file is named after — it was an `Option` assigned with \
         `=`, and the first of two executes was dropped in silence. Bounded by \
         the stream, like `draws`",
    ),
    (
        "StreamAccum",
        "vertex_buffers",
        Second::ReplacesItsSlot,
        "setVertexBuffer:offset:atIndex:. The index space is bounded by the \
         `BindTable` walk, whose refusal is recorded in `unrepresentable` and \
         adjudicated by the bounded-walk scan",
    ),
    (
        "StreamAccum",
        "fragment_buffers",
        Second::ReplacesItsSlot,
        "setFragmentBuffer:offset:atIndex:, the fragment half of the table above \
         and bounded the same way",
    ),
    (
        "StreamAccum",
        "vertex_textures",
        Second::ReplacesItsSlot,
        "setVertexTexture:atIndex:, same table shape and same bound",
    ),
    (
        "StreamAccum",
        "fragment_textures",
        Second::ReplacesItsSlot,
        "setFragmentTexture:atIndex:, same table shape and same bound",
    ),
    (
        "StreamAccum",
        "vertex_samplers",
        Second::ReplacesItsSlot,
        "setVertexSamplerState:atIndex:, same table shape and same bound",
    ),
    (
        "StreamAccum",
        "fragment_samplers",
        Second::ReplacesItsSlot,
        "setFragmentSamplerState:atIndex:, same table shape and same bound",
    ),
    (
        "StreamAccum",
        "viewports",
        Second::ReplacesIt,
        "setViewports:count:. A second record moves the viewports; it does not \
         append to them, so this is assigned rather than extended and a record \
         of two after a record of five leaves two. It is a collection because \
         one *record* carries several, which is a different question from what \
         a second record does — the singular `setViewport:` is this same field \
         at length one",
    ),
    (
        "StreamAccum",
        "scissors",
        Second::ReplacesIt,
        "setScissorRects:count:, on the same reading as the viewports above",
    ),
    (
        "StreamAccum",
        "indexed",
        Second::ConsumedByItsOwnRecord,
        "Metal carries the index buffer as an argument to \
         `drawIndexedPrimitives:` rather than as encoder state, so this is \
         written and read inside one draw arm: the `PendingDraw` pushed on the \
         same record takes it through `bind_snapshot`. The `else` branch clears \
         it for exactly that reason",
    ),
    (
        "StreamAccum",
        "blend_color",
        Second::ReplacesIt,
        "setBlendColor:red:green:blue:alpha:, one constant colour for the encoder",
    ),
    (
        "StreamAccum",
        "cull_mode",
        Second::ReplacesIt,
        "setCullMode:. One face-culling mode is in force at a time and the last \
         one set is what every later draw in the pass uses",
    ),
    (
        "StreamAccum",
        "front_facing",
        Second::ReplacesIt,
        "setFrontFacingWinding:, which decides only which face `cull_mode` \
         above culls and is one value like it",
    ),
    (
        "StreamAccum",
        "fill_mode",
        Second::ReplacesIt,
        "setTriangleFillMode:. Latched whatever the value, including Metal's \
         default, because a pass that sets Lines and then sets Fill again is \
         asking for Fill",
    ),
    (
        "StreamAccum",
        "depth_clip_mode",
        Second::ReplacesIt,
        "setDepthClipMode:, the sibling of the fill mode above and latched on \
         the same terms",
    ),
    (
        "StreamAccum",
        "depth_bias",
        Second::ReplacesIt,
        "setDepthBias:slopeScale:clamp:. Three floats set by one selector, so \
         they replace as a group and cannot be half-updated",
    ),
    (
        "StreamAccum",
        "depth_stencil_ref",
        Second::ReplacesIt,
        "setDepthStencilState:, an object ref and one bound state",
    ),
    (
        "StreamAccum",
        "stencil_ref",
        Second::ReplacesIt,
        "setStencilReferenceValue: and its front/back pair, which set the same \
         encoder state through two records",
    ),
    (
        "StreamAccum",
        "depth_attach",
        Second::ReplacesIt,
        "the pass descriptor's depth attachment. A render pass has one, unlike \
         its colour attachments, which is why this is a field and `color_slots` \
         is a list",
    ),
    (
        "StreamAccum",
        "stencil_attach",
        Second::ReplacesIt,
        "the pass descriptor's stencil attachment, one per pass like the depth \
         one above",
    ),
    (
        "StreamAccum",
        "dropped_unbound",
        Second::FoldsIntoIt,
        "a count of draw records that reached no `PendingDraw`, reported once \
         per stream by `note_stream_draw_drops`",
    ),
    (
        "StreamAccum",
        "unrepresentable",
        Second::StickyOnPurpose,
        "the stream's own refusal, read by `bind_snapshot` so both consumers \
         refuse on the same terms. No retirement path and none needed: a \
         `StreamAccum` is built fresh per stream and dropped at `finish_stream`, \
         so it cannot outlive what it describes — its own doc argues this \
         against the compute rail's, which does need one",
    ),
    // --- ComputeAccum: one compute encoder, many dispatches --------------
    (
        "ComputeAccum",
        "pipeline_ref",
        Second::ReplacesIt,
        "setComputePipelineState:, and `set_pipeline` additionally ignores a \
         zero so an unbind does not read as a rebind",
    ),
    (
        "ComputeAccum",
        "buffers",
        Second::ReplacesItsSlot,
        "setBuffer:offset:atIndex:. Keyed by index — a nil entry `retain`s the \
         slot out, a live one replaces it — and the index space is bounded by \
         `MAX_COMPUTE_BUFFER_SLOTS`, whose overflow lands in `refused_bind` \
         below rather than being skipped",
    ),
    (
        "ComputeAccum",
        "textures",
        Second::ReplacesItsSlot,
        "setTexture:atIndex:, same keyed shape and its own slot bound",
    ),
    (
        "ComputeAccum",
        "samplers",
        Second::ReplacesItsSlot,
        "setSamplerState:atIndex:, same keyed shape and its own slot bound",
    ),
    (
        "ComputeAccum",
        "threadgroup_memory",
        Second::ReplacesItsSlot,
        "setThreadgroupMemoryLength:atIndex:, same keyed shape",
    ),
    (
        "ComputeAccum",
        "stage_in_region",
        Second::ReplacesIt,
        "the direct `0xd1` stage-in region. One region describes the next \
         dispatch, and `0xd2` clears it — the two records are alternatives, not \
         a sequence",
    ),
    (
        "ComputeAccum",
        "stage_in_region_indirect",
        Second::ReplacesIt,
        "the indirect `0xd2` form of the same one region, which is why setting \
         it clears the direct one",
    ),
    (
        "ComputeAccum",
        "imageblock",
        Second::ReplacesIt,
        "setImageblockWidth:height:, one imageblock geometry per encoder",
    ),
    (
        "ComputeAccum",
        "dispatch_type",
        Second::ReplacesIt,
        "the `0xdb` Metal serial/concurrent dispatch type, encoder state with 0 \
         as its default",
    ),
    (
        "ComputeAccum",
        "refused_bind",
        Second::StickyOnPurpose,
        "a bind this accumulator could not hold, kept so \
         `resolve_dispatch_dims_reported` can refuse the dispatch instead of \
         running it without the binding. First-wins via `get_or_insert`, and \
         its retirement path is `clear_refusal_at`: a nil bind at the slot that \
         overflowed says the guest no longer wants anything there",
    ),
    // --- ComputeSegment: one SEGMENT_TYPE_COMPUTE segment ----------------
    (
        "ComputeSegment",
        "acc",
        Second::HoldsAnAccumulator,
        "the segment's `ComputeAccum`, whose own ten fields are adjudicated \
         above. Nothing is latched at this level",
    ),
    (
        "ComputeSegment",
        "session",
        Second::FoldsIntoIt,
        "one multi-record encoder per segment, opened on demand by the first \
         control-flow or ICB record through `ensure_session` and committed at \
         segment end. A later record joins the open session rather than \
         replacing it, which is why the opener is get-or-create and not an \
         assignment",
    ),
    (
        "ComputeSegment",
        "block",
        Second::StickyOnPurpose,
        "a latched sequencing failure: once set, `apply_sequencing` refuses \
         every later dispatch in the segment rather than running one whose \
         control flow this device could not build. No retirement path and none \
         needed — the segment's records are the ones it describes, and the \
         value dies with the segment",
    ),
];

/// A `struct` declaration, as `(field name, the type names it mentions)`.
#[derive(Debug)]
struct Decl {
    name: String,
    fields: Vec<(String, Vec<String>)>,
}

/// Every brace-bodied `struct` in the two guest-facing crates, with its named
/// fields and the type identifiers each field mentions.
///
/// Type identifiers rather than a parsed type, because the only question asked
/// of them is "does this field mention an accumulator" — `Option<ComputeAccum>`
/// and `ComputeAccum` are the same answer, and a parser that told them apart
/// would only be a parser to keep correct.
fn declarations(sources: &[(String, String)]) -> Vec<Decl> {
    let mut out = Vec::new();
    for (_, text) in sources {
        let text = blank_comments(text);
        let bytes: Vec<&str> = text.lines().collect();
        for (i, line) in bytes.iter().enumerate() {
            let trimmed = line.trim_start();
            let Some(rest) = trimmed
                .strip_prefix("pub struct ")
                .or_else(|| trimmed.strip_prefix("pub(crate) struct "))
                .or_else(|| trimmed.strip_prefix("struct "))
            else {
                continue;
            };
            if !rest.ends_with('{') {
                // A tuple struct, a unit struct, or a generic wrapping onto the
                // next line. None of them is an accumulator of decoded records:
                // those are all plain brace structs with named fields.
                continue;
            }
            let name = rest
                .split(|c: char| !c.is_alphanumeric() && c != '_')
                .next()
                .unwrap_or_default()
                .to_owned();
            if name.is_empty() {
                continue;
            }
            let indent = line.len() - trimmed.len();
            let close = format!("{}}}", " ".repeat(indent));
            let mut fields = Vec::new();
            for body in bytes.iter().skip(i + 1) {
                if *body == close {
                    break;
                }
                let t = body.trim();
                if t.is_empty() || t.starts_with('#') {
                    continue;
                }
                let Some((lhs, rhs)) = t.split_once(':') else {
                    continue;
                };
                let ident = lhs.trim().rsplit(' ').next().unwrap_or_default().to_owned();
                if ident.is_empty()
                    || !ident
                        .chars()
                        .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
                {
                    continue;
                }
                let mentions = rhs
                    .split(|c: char| !c.is_alphanumeric() && c != '_')
                    .filter(|t| t.chars().next().is_some_and(char::is_uppercase))
                    .map(str::to_owned)
                    .collect();
                fields.push((ident, mentions));
            }
            if !fields.is_empty() {
                out.push(Decl { name, fields });
            }
        }
    }
    out
}

/// Struct names reached by each discovery route, kept apart so the self-check
/// can prove no route went blind.
#[derive(Default, Debug)]
struct Discovered {
    by_name: Vec<String>,
    by_parameter: Vec<String>,
    by_composition: Vec<String>,
}

impl Discovered {
    fn all(&self) -> Vec<String> {
        let mut names = self.by_name.clone();
        names.extend(self.by_parameter.iter().cloned());
        names.extend(self.by_composition.iter().cloned());
        names.sort();
        names.dedup();
        names
    }
}

fn discover(sources: &[(String, String)], decls: &[Decl]) -> Discovered {
    let mut out = Discovered::default();
    for decl in decls {
        if decl.name.ends_with("Accum") {
            out.by_name.push(decl.name.clone());
        }
    }
    for (_, text) in sources {
        for line in blank_comments(text).lines() {
            // `acc: &mut T`, `acc: &T`, `acc: T`. The type is whatever follows
            // the borrow markers, up to the first non-path character.
            let Some(rest) = line.split("acc: ").nth(1) else {
                continue;
            };
            let ty = rest
                .trim_start_matches('&')
                .trim_start()
                .trim_start_matches("mut ")
                .split(|c: char| !c.is_alphanumeric() && c != '_')
                .next()
                .unwrap_or_default();
            if ty.chars().next().is_some_and(char::is_uppercase) {
                out.by_parameter.push(ty.to_owned());
            }
        }
    }
    out.by_name.sort();
    out.by_name.dedup();
    out.by_parameter.sort();
    out.by_parameter.dedup();
    // Close under composition: a struct holding an accumulator accumulates the
    // same records through it, so it is one too. To a fixpoint, because a
    // wrapper may itself be wrapped.
    loop {
        let known = out.all();
        let added: Vec<String> = decls
            .iter()
            .filter(|d| !known.contains(&d.name))
            .filter(|d| {
                d.fields
                    .iter()
                    .any(|(_, mentions)| mentions.iter().any(|m| known.contains(m)))
            })
            .map(|d| d.name.clone())
            .collect();
        if added.is_empty() {
            break;
        }
        out.by_composition.extend(added);
        out.by_composition.sort();
        out.by_composition.dedup();
    }
    out
}

#[test]
fn every_accumulated_field_says_what_a_second_record_does() {
    let sources = guest_facing_sources();
    let decls = declarations(&sources);
    let found = discover(&sources, &decls);

    // The self-check every source scan here carries. This one has three ways to
    // go blind and they fail differently, so each is asserted on its own: a
    // renamed struct empties the first list, an inline-only one empties the
    // second, and a field-type reader that stops matching empties the third —
    // and a clean crate is indistinguishable from any of them.
    assert!(
        found.by_name.contains(&"StreamAccum".to_owned()),
        "the `*Accum` route cannot see `StreamAccum`, so its silence about \
         every other struct means nothing. Found by name: {:?}",
        found.by_name
    );
    assert!(
        found.by_parameter.contains(&"ComputeAccum".to_owned()),
        "the `acc:` parameter route cannot see `ComputeAccum`, so a struct that \
         stops being named `*Accum` would drop out of the population without \
         failing anything. Found by parameter: {:?}",
        found.by_parameter
    );
    assert!(
        found.by_composition.contains(&"ComputeSegment".to_owned()),
        "the composition closure cannot see `ComputeSegment`, which holds a \
         `ComputeAccum` and is named for neither seed route. Without it this \
         scan is two name-matches wearing a closure. Found by composition: {:?}",
        found.by_composition
    );

    let names = found.all();
    let accums: Vec<&Decl> = decls.iter().filter(|d| names.contains(&d.name)).collect();
    assert!(
        accums.len() >= names.len(),
        "the declaration reader found {} struct(s) with fields out of {} \
         discovered name(s); it cannot report a clean tree it did not read. \
         Names: {names:?}",
        accums.len(),
        names.len()
    );

    let mut unadjudicated = Vec::new();
    for accum in &accums {
        for (field, _) in &accum.fields {
            let known = VERDICTS
                .iter()
                .any(|(s, f, _, _)| *s == accum.name && f == field);
            if !known {
                unadjudicated.push(format!("{}::{field}", accum.name));
            }
        }
    }
    assert!(
        unadjudicated.is_empty(),
        "these hold a decoded guest record and nothing says what a second one \
         of the same kind does to them.\n\nAsk one question: if the guest sends \
         that record twice, does Metal do the thing twice? If it does not — a \
         cull mode, a viewport, a bound pipeline — the field is encoder state, \
         one slot is the contract, and the verdict is `ReplacesIt` naming the \
         selector. If it does, the field is work and one slot silently drops a \
         call; make it a collection and answer `AddsToIt`, or answer \
         `LosesTheFirst` and get a failing build that says so. Add a row to \
         VERDICTS in {}.\n\n{}",
        file!(),
        unadjudicated.join("\n")
    );

    let mut stale = Vec::new();
    for (s, f, _, _) in VERDICTS {
        let live = accums
            .iter()
            .any(|a| a.name == *s && a.fields.iter().any(|(x, _)| x == f));
        if !live {
            stale.push(format!("{s}::{f}"));
        }
    }
    assert!(
        stale.is_empty(),
        "a verdict names a field this scan no longer finds — the field was \
         renamed, moved, or the struct changed shape. Re-read what it means now \
         rather than re-pointing the row; a rename is usually where the answer \
         changed.\n\n{}",
        stale.join("\n")
    );
}

/// No field may be classified as dropping the guest's first record.
///
/// Separate from the adjudication test for the reason the sibling scans give:
/// that one fails when nobody has answered, this one when somebody has and the
/// answer is that a call the guest made does not happen. The two need different
/// messages because they need different fixes — a line of prose against a type
/// change.
#[test]
fn no_accumulated_field_is_allowed_to_lose_the_first_record() {
    let losing: Vec<String> = VERDICTS
        .iter()
        .filter(|(_, _, second, _)| *second == Second::LosesTheFirst)
        .map(|(s, f, _, _)| format!("{s}::{f}"))
        .collect();
    assert!(
        losing.is_empty(),
        "these hold guest work in a slot that holds one, so a second record of \
         the same kind drops the first. A GPU refuses a call it cannot honour; \
         it does not accept two and perform one. The fix is a collection, and \
         then the question of what bounds it — which the other bound scans in \
         this directory will ask.\n\n{}",
        losing.join("\n")
    );
}

/// Every verdict says why, in more than a word.
#[test]
fn every_verdict_says_why() {
    for (s, f, _, why) in VERDICTS {
        assert!(
            why.len() > 30,
            "{s}::{f}'s verdict is asserted and not argued: {why:?}"
        );
    }
}
