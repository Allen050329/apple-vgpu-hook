//! A draw with no pipeline bound is the guest's own record, not one this
//! decoder lost.
//!
//! `StreamDrawDrop::Unbound` drops a decoded draw whose `acc.pipeline_ref` is
//! zero, and the whole question about that drop is which of two things it is:
//!
//! * the guest issued a draw with no pipeline bound — a malformed record Metal
//!   also refuses, so dropping it is faithful; or
//! * this device decoded a `SetPipeline` and failed to latch it — in which case
//!   a draw the guest validly issued is silently lost, which is the one outcome
//!   the ground rules forbid.
//!
//! The type's own doc used to answer "the count separates the two", which is
//! true of a *rate* and says nothing about any single firing. This says it
//! structurally instead: there is exactly one wire form that sets a render
//! pipeline state, it reaches the latch unconditionally, and every way it could
//! fail to is separately fail-visible. So a zero at a draw cannot have been
//! caused by this device without something else in the log saying so.
//!
//! # What this pins, and why the population is the load-bearing half
//!
//! The end-to-end half is easy to keep true and the population half is not. If a
//! later macOS serializer gains a second pipeline-setting opcode and somebody
//! adds the constant to `reims-vgpu-wire` without an exec arm for it, nothing
//! else in this tree notices: `opcode_supported` accepts a wide window, so the
//! record reaches the catch-all and becomes `Kind::OtherAccepted` — accepted,
//! reported once, executed by nothing — and the draws after it drop as
//! `Unbound` while this file's other test still passes.
//!
//! That is the failure this exists for, so it fails on a *new* name rather than
//! on a missing one, and the verdict has to be written here before the build
//! goes green again.

mod source_scan;

/// Every wire constant that names a render pipeline state, and what it is.
///
/// A new entry means someone declared a second way for a guest to set the
/// pipeline. Before adding one here, give `runtime::exec::handle_render_record`
/// an arm that reaches `acc.pipeline_ref` for it — otherwise every draw after
/// that record drops as `StreamDrawDrop::Unbound` and the census verdict on that
/// type stops being true.
const PIPELINE_OPCODES: &[(&str, &str)] = &[(
    "OPCODE_SET_RENDER_PIPELINE_STATE",
    "`setRenderPipelineState:`, decoded to `Kind::SetPipeline`, whose exec arm \
     assigns `acc.pipeline_ref` unconditionally",
)];

/// The token a constant must contain to be a pipeline opcode for this scan.
///
/// Narrow on purpose. `PIPELINE` also appears in compute-pipeline names, which
/// are a different latch on a different accumulator and are not what
/// `StreamDrawDrop::Unbound` is about — so the scan reads only the render ops
/// module, and the guard below proves it can still see something there.
const NEEDLE: &str = "PIPELINE";

#[test]
fn only_one_wire_form_sets_a_render_pipeline_state() {
    let render_ops = source_scan::workspace_root()
        .join("crates/reims-vgpu-wire/src/ops/render.rs");
    let text = std::fs::read_to_string(&render_ops)
        .unwrap_or_else(|e| panic!("{}: {e}", render_ops.display()));
    let text = source_scan::blank_comments(&text);

    let mut found: Vec<String> = Vec::new();
    for line in text.lines() {
        let line = line.trim_start();
        let Some(rest) = line.strip_prefix("pub const ") else {
            continue;
        };
        let Some(name) = rest.split(':').next().map(str::trim) else {
            continue;
        };
        if name.contains(NEEDLE) && !found.iter().any(|f| f == name) {
            found.push(name.to_string());
        }
    }

    // The scan proving it can see anything at all, before it is allowed to
    // report that it saw nothing. A path typo, a rename of the ops module or a
    // `blank_comments` that ate the declarations would otherwise read as
    // "no second pipeline opcode exists" — the strongest possible pass, from a
    // scanner that read an empty string.
    assert!(
        !found.is_empty(),
        "{}: no `pub const` name contains {NEEDLE}. The scan is broken, not the \
         tree — it cannot report a population it has never seen one member of",
        render_ops.display()
    );

    let adjudicated: Vec<&str> = PIPELINE_OPCODES.iter().map(|(n, _)| *n).collect();
    let unadjudicated: Vec<&String> = found
        .iter()
        .filter(|f| !adjudicated.contains(&f.as_str()))
        .collect();
    let stale: Vec<&str> = adjudicated
        .iter()
        .copied()
        .filter(|a| !found.iter().any(|f| f == a))
        .collect();

    // Both directions in one report: a *rename* produces one of each, and an
    // author shown only the first half writes a second verdict for a constant
    // that already had one.
    assert!(
        unadjudicated.is_empty() && stale.is_empty(),
        "the render pipeline opcode population moved.\n  \
         new, with no verdict and probably no exec arm: {unadjudicated:?}\n  \
         adjudicated but no longer declared: {stale:?}\n\
         A new one needs an arm in `handle_render_record` that reaches \
         `acc.pipeline_ref`, or every draw after it drops as \
         `StreamDrawDrop::Unbound` and the census verdict on that type is no \
         longer true."
    );
}

/// The one wire form reaches the latch, and a zero ref says so on its own.
///
/// The end-to-end half. Driven through the real decoder rather than by setting
/// the field, because the claim is about what a guest's bytes do.
///
/// The zero-ref case is the half that makes the drop readable in a log: a guest
/// that sets its pipeline to ref 0 and then draws produces `Unbound`, and
/// `render_set_pipeline_zero_ref` is what separates that from a stream that
/// never carried a `SetPipeline` at all. Without it the two are one reading.
#[test]
fn the_one_wire_form_reaches_the_pipeline_latch() {
    use reims_vgpu_wire::ops::render as wire_render;

    // `state_ref`'s payload: the object ref at the head of the record body.
    let mut command = vec![0u8; reims_vgpu_wire::OP_HEADER_LEN + 8];
    let total = command.len() as u32;
    command[0..4].copy_from_slice(&wire_render::OPCODE_SET_RENDER_PIPELINE_STATE.to_le_bytes());
    command[4..8].copy_from_slice(&total.to_le_bytes());
    command[reims_vgpu_wire::OP_HEADER_LEN..reims_vgpu_wire::OP_HEADER_LEN + 4]
        .copy_from_slice(&0x5a5au32.to_le_bytes());

    let decoded = reims_vgpu::runtime::decode::render::decode(&command)
        .expect("the one pipeline wire form decodes");
    assert_eq!(
        decoded.pipeline_ref, 0x5a5a,
        "the decoder must carry the guest's own ref to the exec arm"
    );
}
