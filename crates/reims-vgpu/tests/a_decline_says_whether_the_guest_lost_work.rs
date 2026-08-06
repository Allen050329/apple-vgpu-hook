//! Every typed decline says whether the guest lost work, and how.
//!
//! The decline vocabulary answers *what* refused — one slug per check, never
//! shared — and that is the property `decline_slugs_are_unique` holds. It does
//! not answer the question the standing goal is actually about: **did the guest
//! lose work, and did this device execute something other than what the guest
//! asked for?**
//!
//! `AGENTS.md` states the gap as advice to the reader:
//!
//! > A named reason on the fail channel is not automatically lost work. Some
//! > report a repair that *succeeded*, fail-visible so the reliance stays
//! > measurable. **Read the emitter.**
//!
//! Ninety-nine `impl Decline`/`impl Refusal` blocks is ninety-nine emitters to
//! read, once per question, with no way to check the reading afterwards. So a
//! boot log cannot be ranked by loss, and "which failure modes are left" —
//! the thing this project is trying to drive to zero — is not a measurement
//! anybody can take. This test makes it one: the population comes from a scan
//! of the source, and each entry carries a written verdict that had to be
//! reached by reading the caller.
//!
//! # The classes, worst first
//!
//! A type's verdict is the **worst** consequence any of its arms can carry, so
//! a loss can never hide inside a type whose other arms are benign. That is a
//! deliberate over-statement: this table is an upper bound on the harm each
//! type can do, not a claim about the arm that fires most.
//!
//! # What a red here means
//!
//! Either a new decline type was added without saying what it costs the guest,
//! or an existing one moved. Both want the same fix: read the emitter, write
//! the verdict. Do not answer [`Loss::ExecutedModified`] to make a new type
//! compile — that variant is a defect list, and
//! [`the_executed_modified_census_only_shrinks`] is what stops it growing.

mod source_scan;
use source_scan::decline_impls;

/// What the guest is left with when this decline fires.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Loss {
    /// The device executed something **other than what the guest asked for**: a
    /// record dropped out of a plural command while the rest ran, a table
    /// truncated, a flag or an attribute ignored, a count clamped. No error
    /// reaches the guest, and the frame or the compute result is wrong.
    ///
    /// Also covers a command that ran *nothing* and reported completion, which
    /// does the guest the same harm from the only side it can see: it frees the
    /// targets and reads the results either way. Nothing in this device has an
    /// error channel back to the guest, so "the guest was not told" cannot be
    /// what separates this class from [`Loss::Refused`] — what separates them is
    /// whether the guest's own view of what happened is wrong.
    ///
    /// A real GPU does not do this. This is the class the standing goal exists
    /// to eliminate, and every entry must name what would retire it.
    ExecutedModified,
    /// The whole command was declined because this device does not implement
    /// the feature. Honest — the guest is not lied to — but the work is lost,
    /// and the fix is to implement it rather than to reclassify it.
    Unimplemented,
    /// The whole command was declined for a reason a real GPU also has: its
    /// memory is full, the guest record is malformed, the request is outside
    /// what the API can express. Refusing is the faithful answer.
    Refused,
    /// The host GPU, its driver, or the OS failed. No value of any constant in
    /// this repository would have served the request.
    HostFault,
    /// Full fidelity kept. A cheaper rail declined and a correct slower one ran
    /// instead — the copying path where an import was refused, the CPU gather
    /// where the GPU could not reach guest pages. The reason must say which
    /// rail was taken, because "refuses the fast rail" and "refuses the draw"
    /// are different answers wearing similar words.
    SlowPath,
    /// The condition was detected and corrected. The guest's command executed
    /// as asked; the line exists so the reliance on the repair stays visible.
    Repaired,
    /// Not a failure. Not-ready-yet, an intentionally unbound `ref == 0`,
    /// ordering, or a census that borrows this vocabulary to name itself.
    /// `AGENTS.md`: "Expected control flow should stay quiet."
    Ordering,
}

/// One adjudicated decline type.
struct Row {
    /// Path relative to the workspace root, as the scan reports it.
    file: &'static str,
    /// The type the trait is implemented for.
    ty: &'static str,
    /// The worst consequence any arm of this type can carry.
    loss: Loss,
    /// Why, naming the arm that justifies the verdict and what the caller does
    /// after it. For [`Loss::ExecutedModified`], also what would retire it.
    why: &'static str,
}

/// How many types may answer [`Loss::ExecutedModified`].
///
/// Not a budget. A ratchet: see [`the_executed_modified_census_only_shrinks`].
const EXECUTED_MODIFIED_CEILING: usize = 8;

/// Every `impl Decline`/`impl Refusal` in the crate, and what its worst arm
/// costs the guest.
///
/// Keyed by `(file, type)` rather than by type alone, because five distinct
/// `DecodeStatus` types live in five modules and `Status` names two more.
const ROWS: &[Row] = &[
    Row {
        file: "crates/reims-vgpu/src/backend/metal/error.rs",
        ty: "Status",
        loss: Loss::Refused,
        why: "the Metal backend's own status; every non-ok class names an \
              argument or execute check that stops the operation before it \
              reaches the driver, and the caller returns rather than encoding",
    },
    Row {
        file: "crates/reims-vgpu/src/backend/metal/raw_metal.rs",
        ty: "MetalPipelineDecline",
        loss: Loss::Unimplemented,
        why: "a compute pipeline Metal would not build with reflection; the \
              dispatch is declined whole",
    },
    Row {
        file: "crates/reims-vgpu/src/backend/metal/raw_metal.rs",
        ty: "MetalSamplerMaskOverflow",
        loss: Loss::ExecutedModified,
        why: "the emitter `continue`s past the sampler slot and finishes \
              building the mask, so the shader samples a slot that never \
              receives its default sampler. A healthy zero — the table this \
              exceeds is Metal's own, so a firing means this backend's idea of \
              it has parted from the driver's. Retired by deriving the bound \
              from the driver rather than from a constant",
    },
    Row {
        file: "crates/reims-vgpu/src/backend/vulkan/engine/caches.rs",
        ty: "VertexFormatWidenDecline",
        loss: Loss::ExecutedModified,
        why: "a three-component vertex format the host does not offer is \
              widened to its mandatory four-component sibling and the pipeline \
              is built anyway, with `resolve` checking only that the wider read \
              stays inside the stride. **Not retirable by proving the read \
              identical**, which is the obvious attempt: a shader input \
              declared `vec4` over a three-component attribute takes \
              `(x,y,z,1.0)` from the format the guest asked for and \
              `(x,y,z,<whatever those four bytes hold>)` from the substitute, \
              because the fourth component is now supplied rather than \
              defaulted. Identical only where every consumer reads three \
              components or fewer, which nothing here checks. Retired by \
              refusing the pipeline, or by reading the shader's declared input \
              width and widening only when it is three or under",
    },
    Row {
        file: "crates/reims-vgpu/src/backend/vulkan/engine/compute_execution.rs",
        ty: "ComputeExecutionDecline",
        loss: Loss::Refused,
        why: "execute-time checks that resident state still matches what \
              validation admitted; the dispatch does not run",
    },
    Row {
        file: "crates/reims-vgpu/src/backend/vulkan/engine/compute_validation.rs",
        ty: "ComputeValidationDecline",
        loss: Loss::Refused,
        why: "structural checks on the dispatch request before any GPU work; \
              the dispatch is declined whole",
    },
    Row {
        file: "crates/reims-vgpu/src/backend/vulkan/engine/context.rs",
        ty: "PipelineCacheDecline",
        loss: Loss::SlowPath,
        why: "the pipeline cache failed to load, warm or persist; every draw \
              still compiles and executes, from cold",
    },
    Row {
        file: "crates/reims-vgpu/src/backend/vulkan/engine/desc_arena.rs",
        ty: "SetExceedsBlock",
        loss: Loss::Refused,
        why: "a descriptor set wants more of one type than an empty block \
              holds, so growing the arena changes nothing; the draw is refused",
    },
    Row {
        file: "crates/reims-vgpu/src/backend/vulkan/engine/device_lost.rs",
        ty: "DeviceLostDecline",
        loss: Loss::HostFault,
        why: "the driver returned ERROR_DEVICE_LOST or recreation failed",
    },
    Row {
        file: "crates/reims-vgpu/src/backend/vulkan/engine/dmabuf.rs",
        ty: "DmaBufDecline",
        loss: Loss::SlowPath,
        why: "the guest-pages import was refused; the caller gathers the same \
              bytes on the CPU into staging, which is the only rail on a host \
              without the extension",
    },
    Row {
        file: "crates/reims-vgpu/src/backend/vulkan/engine/dmabuf.rs",
        ty: "GuestWriteDecline",
        loss: Loss::SlowPath,
        why: "the GPU could not write the guest's pages directly; the frame \
              lands through the copying writeback instead",
    },
    Row {
        file: "crates/reims-vgpu/src/backend/vulkan/engine/draw_execution.rs",
        ty: "DrawExecutionDecline",
        loss: Loss::Refused,
        why: "execute-time disagreements between resident state and what \
              validation admitted; the draw does not execute",
    },
    Row {
        file: "crates/reims-vgpu/src/backend/vulkan/engine/draw_preparation.rs",
        ty: "DrawPreparationDecline",
        loss: Loss::Refused,
        why: "the draw could not be prepared — a missing pipeline, an MTLB \
              that would not extract, a translation that failed; nothing runs",
    },
    Row {
        file: "crates/reims-vgpu/src/backend/vulkan/engine/draw_validation.rs",
        ty: "DrawValidationDecline",
        loss: Loss::Refused,
        why: "structural checks on the draw request before any GPU work",
    },
    Row {
        file: "crates/reims-vgpu/src/backend/vulkan/engine/exec.rs",
        ty: "BufferImportDecline",
        loss: Loss::SlowPath,
        why: "the vertex/index buffer could not be imported from guest pages; \
              the bytes reach the GPU through the CPU gather instead",
    },
    Row {
        file: "crates/reims-vgpu/src/backend/vulkan/engine/exec.rs",
        ty: "SampledImportDecline",
        loss: Loss::SlowPath,
        why: "the sampled texture could not be imported from guest pages; the \
              texels are copied into staging instead",
    },
    Row {
        file: "crates/reims-vgpu/src/backend/vulkan/engine/facade_decline.rs",
        ty: "EngineFacadeDecline",
        loss: Loss::Refused,
        why: "the façade's own state disagreed with the request — a presenter \
              not attached, a resident that disappeared before it was pinned; \
              the named operation does not proceed",
    },
    Row {
        file: "crates/reims-vgpu/src/backend/vulkan/engine/init_decline.rs",
        ty: "InitDecline",
        loss: Loss::Unimplemented,
        why: "bring-up failed and the error is latched, so every later draw is \
              refused with it. Not a per-command loss but the whole boot's; the \
              retryable out-of-memory class is deliberately not latched",
    },
    Row {
        file: "crates/reims-vgpu/src/backend/vulkan/engine/pools/submission_and_buffers.rs",
        ty: "ReadbackMemoryDegrade",
        loss: Loss::SlowPath,
        why: "the readback slot landed in uncached host memory; the bytes are \
              correct and the copy out of it is roughly an order slower",
    },
    Row {
        file: "crates/reims-vgpu/src/backend/vulkan/engine/pools/teardown.rs",
        ty: "ReadbackLeaseQuiesceExpired",
        loss: Loss::HostFault,
        why: "teardown waited out its quiesce window for lease holders that \
              never returned, which is a broken invariant in the host layer \
              rather than anything a guest asked for",
    },
    Row {
        file: "crates/reims-vgpu/src/backend/vulkan/engine/reason.rs",
        ty: "DrawReason",
        loss: Loss::Unimplemented,
        why: "the draw names a feature this backend does not offer — a device \
              feature bit not advertised, a Metal state with no Vulkan spelling \
              here; the draw is refused whole",
    },
    Row {
        file: "crates/reims-vgpu/src/backend/vulkan/engine/reason.rs",
        ty: "TargetReadDecline",
        loss: Loss::Refused,
        why: "the readback's identity is not in the registry, or its content \
              is not ready; the read is refused and the caller is told which",
    },
    Row {
        file: "crates/reims-vgpu/src/backend/vulkan/engine/slab.rs",
        ty: "SlabDecline",
        loss: Loss::Refused,
        why: "the suballocator refused a request an allocator may refuse — a \
              zero size, an out-of-bounds or double release. `FreeListInvariant` \
              poisons the block rather than risk aliasing two images",
    },
    Row {
        file: "crates/reims-vgpu/src/backend/vulkan/engine/types.rs",
        ty: "DrawError",
        loss: Loss::Unimplemented,
        why: "the engine's outer error; it wraps the types adjudicated above \
              and its worst arm is `Init`, which latches for the boot",
    },
    Row {
        file: "crates/reims-vgpu/src/backend/vulkan/engine/vk_call.rs",
        ty: "VkCall",
        loss: Loss::HostFault,
        why: "a Vulkan entry point returned a failure; the op and the \
              `vk::Result` are what the line carries",
    },
    Row {
        file: "crates/reims-vgpu/src/backend/vulkan/engine/window_present.rs",
        ty: "SlateReason",
        loss: Loss::Ordering,
        why: "why the host window painted slate this frame — no source \
              published, content not landed yet. The present succeeds; this \
              names what there was to show, not a refusal",
    },
    Row {
        file: "crates/reims-vgpu/src/backend/vulkan/engine/window_present.rs",
        ty: "StagingError",
        loss: Loss::Refused,
        why: "the CPU present fallback could not get a staging buffer; the \
              frame degrades to slate on the host window. The host window is a \
              development view, not the guest's own scanout",
    },
    Row {
        file: "crates/reims-vgpu/src/backend/vulkan/engine/window_present.rs",
        ty: "WindowPresentDecline",
        loss: Loss::Ordering,
        why: "the swapchain has read suboptimal for many consecutive presents \
              without converging; the present itself succeeds",
    },
    Row {
        file: "crates/reims-vgpu/src/backend/vulkan/translate/reason.rs",
        ty: "TranslateReason",
        loss: Loss::Unimplemented,
        why: "a decoded Metal value has no Vulkan spelling this backend will \
              emit; the command that carried it is refused rather than \
              translated to something near it",
    },
    Row {
        file: "crates/reims-vgpu/src/contract/gva_resolve.rs",
        ty: "ResolveStatus",
        loss: Loss::Refused,
        why: "the guest page-table walk faulted for a reason an MMU also has — \
              an inactive task, a zero root, a malformed PTE",
    },
    Row {
        file: "crates/reims-vgpu/src/contract/iosurface_pages.rs",
        ty: "Status",
        loss: Loss::Refused,
        why: "the IOSurface page list did not decode: a short descriptor, an \
              address that is not a kernel VA, an internal field that \
              contradicts the rest",
    },
    Row {
        file: "crates/reims-vgpu/src/contract/mipmap.rs",
        ty: "MetalMipmapError",
        loss: Loss::Refused,
        why: "generation is refused for reasons Metal itself refuses — a zero \
              dimension, a level count the API rejects, a buffer too short",
    },
    Row {
        file: "crates/reims-vgpu/src/host_window/present.rs",
        ty: "WindowError",
        loss: Loss::Unimplemented,
        why: "the development host window could not come up. It is not the \
              guest's scanout, so no guest command is lost with it",
    },
    Row {
        file: "crates/reims-vgpu/src/model/state.rs",
        ty: "FailEvent",
        loss: Loss::ExecutedModified,
        why: "an unrecognised opcode runs nothing and the packet's completion \
              stamp retires anyway — `write_stamp`'s own doc says that write is \
              where \"the guest is told anything finished\", after which it may \
              free the render targets. So the guest frees and reads on a \
              completion nothing earned. **Withholding the stamp is not the \
              retirement**: the guest waits on it, so a device that stopped \
              stamping would hang rather than refuse. The retirement is \
              identifying the opcode",
    },
    Row {
        file: "crates/reims-vgpu/src/model/state.rs",
        ty: "PresentBacking",
        loss: Loss::Ordering,
        why: "a census of whether the presented resident was ever stored; the \
              present executes either way",
    },
    Row {
        file: "crates/reims-vgpu/src/model/state.rs",
        ty: "StateMutationDecline",
        loss: Loss::Refused,
        why: "a device-state mutation carrying a value outside its declared \
              range; the mutator returns false and the state is not moved",
    },
    Row {
        file: "crates/reims-vgpu/src/observe/emit.rs",
        ty: "Fake",
        loss: Loss::Ordering,
        why: "the fixture `observe::emit`'s own tests drive the line builder \
              with; it names no check in the device",
    },
    Row {
        file: "crates/reims-vgpu/src/observe/panic.rs",
        ty: "AbiPanic",
        loss: Loss::HostFault,
        why: "this device panicked inside a C ABI entry point; the catch turns \
              it into an error code instead of unwinding into QEMU",
    },
    Row {
        file: "crates/reims-vgpu/src/qemu/host_ops.rs",
        ty: "QemuHostDecline",
        loss: Loss::HostFault,
        why: "a host callback QEMU was to install is missing or failed",
    },
    Row {
        file: "crates/reims-vgpu/src/runtime/blit_exec/mod.rs",
        ty: "BlitStatus",
        loss: Loss::Unimplemented,
        why: "`Unsupported` covers whole blit families this device does not \
              implement — swizzled type-8 views, multisample, PVRTC rows; the \
              blit is declined whole",
    },
    Row {
        file: "crates/reims-vgpu/src/runtime/census/present_proxy.rs",
        ty: "MrtDrop",
        loss: Loss::ExecutedModified,
        why: "a multi-render-target draw is degraded to a single target and \
              executed, so a later sample of the dropped attachment reads what \
              was there before. Retired by carrying every attachment through \
              the render pass",
    },
    Row {
        file: "crates/reims-vgpu/src/runtime/census/present_proxy.rs",
        ty: "WindowPublishDrop",
        loss: Loss::Ordering,
        why: "a census of frames the window publisher skipped because the \
              resident was not ready; the guest's own scanout is unaffected",
    },
    Row {
        file: "crates/reims-vgpu/src/runtime/census/view_swizzle_census.rs",
        ty: "SwizzleDecline",
        loss: Loss::Repaired,
        why: "a non-identity swizzle the host could not bind directly was \
              applied by rewriting every texel on the CPU. The output is what \
              the guest asked for; the zero-copy property is what was lost",
    },
    Row {
        file: "crates/reims-vgpu/src/runtime/compute_exec/mod.rs",
        ty: "ComputeBindOverflow",
        loss: Loss::Refused,
        why: "the bind loop still `continue`s — there is no slot past the table \
              to record into — but it now records the refusal, and \
              `resolve_dispatch_dims_reported`, the gate both executors pass \
              through, refuses the dispatch with \
              `compute_dispatch_bind_past_table`. It used to dispatch with the \
              index unbound and nothing downstream refused on its absence",
    },
    Row {
        file: "crates/reims-vgpu/src/runtime/compute_exec/mod.rs",
        ty: "ComputeSpirvDecline",
        loss: Loss::Refused,
        why: "the kernel module's header is short or misaligned; it cannot be \
              reflected and the dispatch is refused",
    },
    Row {
        file: "crates/reims-vgpu/src/runtime/compute_exec/mod.rs",
        ty: "ComputeStatus",
        loss: Loss::Refused,
        why: "the dispatch names a pipeline, buffer or texture that is not \
              there, or a grid that is not describable; it does not run",
    },
    Row {
        file: "crates/reims-vgpu/src/runtime/decode/blit.rs",
        ty: "BlitOptionError",
        loss: Loss::Refused,
        why: "the blit option word carries bits this decoder does not know, or \
              aspect bits that contradict each other; the record is refused",
    },
    Row {
        file: "crates/reims-vgpu/src/runtime/decode/blit.rs",
        ty: "DecodeStatus",
        loss: Loss::Refused,
        why: "a short or unknown blit record; nothing is executed from it",
    },
    Row {
        file: "crates/reims-vgpu/src/runtime/decode/compute.rs",
        ty: "DecodeStatus",
        loss: Loss::Refused,
        why: "a short, unknown or unsupported compute record; nothing is \
              executed from it",
    },
    Row {
        file: "crates/reims-vgpu/src/runtime/decode/event.rs",
        ty: "DecodeStatus",
        loss: Loss::Refused,
        why: "a short, mis-lengthed or unknown event record; nothing is \
              executed from it",
    },
    Row {
        file: "crates/reims-vgpu/src/runtime/decode/fifo.rs",
        ty: "ResourceListDecodeError",
        loss: Loss::Refused,
        why: "the resource list's header or body does not match its declared \
              structure; the whole invalidate/synchronize is abandoned rather \
              than applied to the entries that did parse",
    },
    Row {
        file: "crates/reims-vgpu/src/runtime/decode/render/mod.rs",
        ty: "DecodeStatus",
        loss: Loss::Refused,
        why: "a short, unknown, mis-lengthed or out-of-range render record. \
              Note `OtherAccepted` is not a decode — it is the catch-all for \
              'no arm claimed this', and reading it as success hides a family \
              of lost records behind a green run",
    },
    Row {
        file: "crates/reims-vgpu/src/runtime/decode/resource/mod.rs",
        ty: "ColorAttachEntryShort",
        loss: Loss::Refused,
        why: "a colour-attachment entry whose `field_count` promises more fields \
              than the descriptor holds. The walk used to `break` and continue, \
              so every tag past the cut was *absent* rather than unreadable and \
              `entry_tag_u32` supplied its default — opaque `ONE`/`ZERO` \
              blending, no pixel format, a write mask of `all`. Now \
              `res_color_entry_fields_short`, which is the entry level of the \
              fault `note_color_table_truncated` already refuses three ways at \
              the section level. Line latched, refusal not",
    },
    Row {
        file: "crates/reims-vgpu/src/runtime/decode/resource/mod.rs",
        ty: "ColorAttachDropped",
        loss: Loss::Refused,
        why: "a colour-attachment TLV tag outside `0x00..=0x09`. Those ten are \
              the entry's index plus every property of \
              `MTLRenderPipelineColorAttachmentDescriptor`, so an eleventh is a \
              property this decoder has no name for, and the pipeline is refused \
              `res_color_field_unread` rather than built with Metal's default \
              where the guest set its own. It used to report and continue. What \
              licensed the change is the sibling that fires: \
              `type7_color_attach_shape` appears 4-13 times on every driven boot \
              in the record, each `unconsumed=0`, so this zero is measured and \
              not unreached. The line stays `first_sight`-latched and the \
              refusal does not",
    },
    Row {
        file: "crates/reims-vgpu/src/runtime/decode/resource/mod.rs",
        ty: "ColorAttachIndexOutOfRange",
        loss: Loss::Refused,
        why: "an attachment index past the table; the pipeline is refused \
              whole with `ErrUnsupported`",
    },
    Row {
        file: "crates/reims-vgpu/src/runtime/decode/resource/mod.rs",
        ty: "ColorAttachTableTruncated",
        loss: Loss::Refused,
        why: "the attachment section is incomplete; every path out of it \
              refuses the pipeline rather than building the entries that fit",
    },
    Row {
        file: "crates/reims-vgpu/src/runtime/decode/resource/mod.rs",
        ty: "ColorWriteMaskOutOfRange",
        loss: Loss::Refused,
        why: "a write mask wider than the four bits `MTLColorWriteMask` holds \
              refuses the pipeline with `res_color_write_mask_over`, as the \
              attachment-index check beside it does. It used to keep the \
              default and return Ok — and the default is `all`, so a guest that \
              masked a channel off got it written",
    },
    Row {
        file: "crates/reims-vgpu/src/runtime/decode/resource/mod.rs",
        ty: "DecodeStatus",
        loss: Loss::Refused,
        why: "a short, unknown-typed or unsupported resource descriptor; the \
              object is not created",
    },
    Row {
        file: "crates/reims-vgpu/src/runtime/decode/resource/mod.rs",
        ty: "VertexDescriptorTruncated",
        loss: Loss::Refused,
        why: "an attribute or layout count past this device's table; the \
              caller returns `Err` immediately after the line, so no partial \
              descriptor is built",
    },
    Row {
        file: "crates/reims-vgpu/src/runtime/decode/stream.rs",
        ty: "DecodeStatus",
        loss: Loss::Refused,
        why: "the stream's framing does not hold; the walk stops rather than \
              guessing where the next record begins",
    },
    Row {
        file: "crates/reims-vgpu/src/runtime/decode/stream.rs",
        ty: "SegmentDisposition",
        loss: Loss::Ordering,
        why: "which of walk/envelope/unknown a segment is. Only `Unknown` is a \
              refusal, and it stops that segment alone",
    },
    Row {
        file: "crates/reims-vgpu/src/runtime/drain/mod.rs",
        ty: "WrapperUpperHalf",
        loss: Loss::ExecutedModified,
        why: "the wrapper word's upper half is non-zero and the dispatch uses \
              the lower half regardless, so a record is executed as an opcode \
              the guest did not name. Retired by learning what the upper half \
              selects, or by refusing the packet until it is known",
    },
    Row {
        file: "crates/reims-vgpu/src/runtime/draw/metal_icb.rs",
        ty: "MetalIcbInheritanceDecline",
        loss: Loss::Refused,
        why: "a parent encoder state an ICB cannot inherit; the ICB execute is \
              refused rather than run under the wrong state",
    },
    Row {
        file: "crates/reims-vgpu/src/runtime/draw/mod.rs",
        ty: "EncodeStatus",
        loss: Loss::Refused,
        why: "the render encode could not start or complete — a missing \
              pipeline or MTLB, a Metal failure, bad arguments",
    },
    Row {
        file: "crates/reims-vgpu/src/runtime/draw/mod.rs",
        ty: "IndexLoadReason",
        loss: Loss::Refused,
        why: "the indexed draw's index bytes could not be loaded or validated; \
              the draw is refused rather than issued over a partial buffer",
    },
    Row {
        file: "crates/reims-vgpu/src/runtime/draw/mod.rs",
        ty: "MetalStateDecline",
        loss: Loss::Refused,
        why: "a sampler or depth-stencil state would not resolve or decode; \
              the draw is refused",
    },
    Row {
        file: "crates/reims-vgpu/src/runtime/draw/render_target.rs",
        ty: "RenderTargetRefusal",
        loss: Loss::Refused,
        why: "no rung resolved the attachment to something renderable. \
              `LinearPastAllocation` is the sharp one: the rows would end past \
              the allocation, and writing them is guest memory this device does \
              not own",
    },
    Row {
        file: "crates/reims-vgpu/src/runtime/draw/texture_view.rs",
        ty: "LinearLoadRefusal",
        loss: Loss::Refused,
        why: "a linear texture could not be loaded for sampling or seeding; \
              the operation is refused",
    },
    Row {
        file: "crates/reims-vgpu/src/runtime/draw/texture_view.rs",
        ty: "TextureViewDecline",
        loss: Loss::Refused,
        why: "the view chain would not resolve — a missing entry, an \
              undecodable descriptor, a chain that cycles or overruns",
    },
    Row {
        file: "crates/reims-vgpu/src/runtime/draw/vulkan.rs",
        ty: "SurfaceResidentArmDecline",
        loss: Loss::SlowPath,
        why: "the deferred resident window could not be armed; the caller \
              takes the synchronous readback instead and the frame is correct",
    },
    Row {
        file: "crates/reims-vgpu/src/runtime/draw/vulkan.rs",
        ty: "Type11SeedDecline",
        loss: Loss::Ordering,
        why: "which rung a type-11 seed came from, or that there was no entry \
              to seed from; a census of the lookup, not a refusal",
    },
    Row {
        file: "crates/reims-vgpu/src/runtime/draw/vulkan.rs",
        ty: "Type5ViewDecline",
        loss: Loss::Refused,
        why: "all twelve arms return through one `fail` closure that answers \
              `None`, and the sole caller propagates it with `?`, so a view \
              this loader could not build is never bound — the `Read` arm drops \
              the partly-filled `native` buffer with the early return rather \
              than binding what it gathered",
    },
    Row {
        file: "crates/reims-vgpu/src/runtime/exec/mod.rs",
        ty: "BindSlotPastTable",
        loss: Loss::Refused,
        why: "the bind loop still breaks at the first slot past the table — \
              there is no slot to put it in — but it now records the bind on \
              `StreamAccum::refused_bind`, and `bind_snapshot` refuses both \
              consumers of the stream's bind state: a decoded draw and an \
              end-of-stream ICB execute. Draws recorded before it snapshotted \
              tables that were still complete and still stand. All three tables \
              meet or exceed Apple's own, so a firing is a record Apple's \
              serializer cannot emit",
    },
    Row {
        file: "crates/reims-vgpu/src/runtime/exec/mod.rs",
        ty: "BufferOffsetSlotPastTable",
        loss: Loss::Refused,
        why: "a `SetBufferOffset` naming a slot past the buffer table has \
              nowhere to land, so the record returns without applying anything \
              and sets `StreamAccum::unrepresentable`, which refuses the \
              stream's later draws through `bind_snapshot`. The bound is Apple's \
              own buffer table exactly, so a firing is a record Apple's \
              serializer cannot emit; in a conforming stream the bind at that \
              slot refused first, and this does not rely on that",
    },
    Row {
        file: "crates/reims-vgpu/src/runtime/exec/mod.rs",
        ty: "ChainAbandonDecline",
        loss: Loss::ExecutedModified,
        why: "the chain `break`s at the offending draw after landing the \
              earlier image, so every later draw in the guest's list is \
              dropped while the ones before it stand. Retired by refusing the \
              whole chain, which is what makes the frame consistent",
    },
    Row {
        file: "crates/reims-vgpu/src/runtime/exec/mod.rs",
        ty: "IcbRecordDropped",
        loss: Loss::Refused,
        why: "the reset or copy record is abandoned outright — the blit arm \
              logs and returns, applying no part of it. Its cost lands on a \
              *later* command: an ICB the guest asked to clear or replace still \
              holds what it held, and the next execute of it runs those. That \
              is a refusal whose consequence outlives it, not a modified \
              execution of this record, and it goes away by implementing both",
    },
    Row {
        file: "crates/reims-vgpu/src/runtime/exec/mod.rs",
        ty: "ResourceTableDecline",
        loss: Loss::Ordering,
        why: "`TailPopulated` counts resource-table fields this device does \
              not read; it names what is unrecovered, and nothing is refused",
    },
    Row {
        file: "crates/reims-vgpu/src/runtime/exec/mod.rs",
        ty: "StreamDrawDrop",
        loss: Loss::Refused,
        why: "the two arms that used to continue no longer do: a dropped \
              depth/stencil attachment and an unbindable colour subresource \
              both set `StreamAccum::unrepresentable`, and `bind_snapshot` \
              refuses the stream's draws rather than running the pass without \
              depth or into the base level. The `Unbound` arm keeps the draw \
              out of `acc.draws` entirely, and its ambiguity is now closed \
              structurally rather than by a rate: \
              `a_pipeline_reaches_the_latch_by_one_wire_form` pins that exactly \
              one wire constant sets a render pipeline state, and its exec arm \
              assigns `acc.pipeline_ref` unconditionally. Every way this device \
              could have caused the zero is separately fail-visible — a `0x74` \
              that fails to decode is an `ErrShort` on that opcode, an opcode \
              nobody enumerated is reported by `note_unimplemented_render_opcode`, \
              and a guest that sets ref 0 itself emits \
              `render_set_pipeline_zero_ref`. So a bare `Unbound` is a draw with \
              no pipeline bound, which Metal refuses too",
    },
    Row {
        file: "crates/reims-vgpu/src/runtime/exec/mod.rs",
        ty: "TextureFillDropped",
        loss: Loss::Refused,
        why: "the fill arm logs and returns without executing any of it, so \
              nothing partial lands; the region keeps what it held. A read-back \
              afterwards sees stale content, which is what a refused write \
              always leaves. Retired by implementing the fill",
    },
    Row {
        file: "crates/reims-vgpu/src/runtime/fence_exec.rs",
        ty: "FenceStatus",
        loss: Loss::Refused,
        why: "the fence or event operation names something this device's \
              synchronisation model cannot express; it is declined whole",
    },
    Row {
        file: "crates/reims-vgpu/src/runtime/gather_witness.rs",
        ty: "GatherWitnessFault",
        loss: Loss::Repaired,
        why: "the witness caught its own vouch breaking under audit and \
              dropped the stale generation, so the next bind re-gathers. The \
              line is what keeps the reliance on the elision measurable",
    },
    Row {
        file: "crates/reims-vgpu/src/runtime/heap_query.rs",
        ty: "QueryError",
        loss: Loss::Refused,
        why: "the heap-size query could not be decoded or answered; the reply \
              carries a zero requirement rather than a guess",
    },
    Row {
        file: "crates/reims-vgpu/src/runtime/host.rs",
        ty: "DmaBufExportError",
        loss: Loss::SlowPath,
        why: "guest pages could not be exported as a dma-buf — most often \
              because guest RAM is not fd-backed; every rail behind it copies \
              instead, which is the only arm on most hosts",
    },
    Row {
        file: "crates/reims-vgpu/src/runtime/host.rs",
        ty: "MemError",
        loss: Loss::Refused,
        why: "a guest memory read or write did not resolve; the caller is told \
              rather than handed zeroes",
    },
    Row {
        file: "crates/reims-vgpu/src/runtime/icb/mod.rs",
        ty: "DroppedVertexAttribute",
        loss: Loss::ExecutedModified,
        why: "**verdict confirmed against the emitter, description was not.** \
              This is not about a location the pipeline has no input for. Both \
              arms fire when a word off the guest's type-7 descriptor is not a \
              declared `MTLVertexFormat` or `MTLVertexStepFunction` variant: the \
              attribute is skipped, the remaining ones are encoded, and \
              `Some(vd)` is returned as long as one survived. So the pipeline is \
              built with a `[[stage_in]]` struct missing a field and the shader \
              reads whatever occupies it — wrong geometry, not an error, which \
              is what the emitter's own doc says. Retired by returning `None` \
              when any attribute was dropped rather than when all were, which \
              needs the caller's `None` path checked first: it means `no vertex \
              descriptor`, and building the pipeline without one is not a \
              refusal either",
    },
    Row {
        file: "crates/reims-vgpu/src/runtime/icb/mod.rs",
        ty: "IcbFlagDropped",
        loss: Loss::ExecutedModified,
        why: "eight decoded create-descriptor flags reach no host setter, so \
              the ICB is built with Metal's default where the guest set its \
              own. Six of the eight default on at both ends, which is why each \
              counter is a healthy zero and a non-zero reading names the flag \
              to build a setter for",
    },
    Row {
        file: "crates/reims-vgpu/src/runtime/icb/mod.rs",
        ty: "IcbStatus",
        loss: Loss::Refused,
        why: "the ICB operation names something missing or a Metal call that \
              failed; the command slot does not execute",
    },
    Row {
        file: "crates/reims-vgpu/src/runtime/m2v_cache.rs",
        ty: "M2vCacheDecline",
        loss: Loss::Unimplemented,
        why: "the shader could not be translated to SPIR-V; the draw is \
              abandoned rather than issued against a different program",
    },
    Row {
        file: "crates/reims-vgpu/src/runtime/mapper/mod.rs",
        ty: "MapperDecline",
        loss: Loss::Refused,
        why: "the capture's own fields contradict each other or name a kernel \
              address that is not one; the mapping never attaches, so every \
              later use of it fails visibly rather than reading elsewhere",
    },
    Row {
        file: "crates/reims-vgpu/src/runtime/mapping_write/mod.rs",
        ty: "GpuWritebackDecline",
        loss: Loss::SlowPath,
        why: "the GPU could not write the guest's pages for this window; the \
              copying writeback lands the same frame",
    },
    Row {
        file: "crates/reims-vgpu/src/runtime/mapping_write/mod.rs",
        ty: "SurfaceWriteRefusal",
        loss: Loss::Refused,
        why: "the writeback has no mapping, or the geometry or format it \
              would write does not match the one recorded; nothing is written",
    },
    Row {
        file: "crates/reims-vgpu/src/runtime/mipmap.rs",
        ty: "MipmapStatus",
        loss: Loss::Unimplemented,
        why: "mip generation names a format or level shape this device does \
              not build; the upper levels keep whatever they held",
    },
    Row {
        file: "crates/reims-vgpu/src/runtime/mtlb.rs",
        ty: "MtlbDecline",
        loss: Loss::Refused,
        why: "the shader container did not parse — a missing wrapper, a \
              truncated header, a blob that runs past its extent",
    },
    Row {
        file: "crates/reims-vgpu/src/runtime/scanout/mod.rs",
        ty: "CaptureDecline",
        loss: Loss::Refused,
        why: "the scanout source could not be captured; the console paints \
              nothing rather than stale or wrong pixels",
    },
    Row {
        file: "crates/reims-vgpu/src/runtime/scanout/mod.rs",
        ty: "ConsoleEfiRowRefused",
        loss: Loss::HostFault,
        why: "the page walk said the row was readable and the read then \
              refused, which is two host answers contradicting each other",
    },
    Row {
        file: "crates/reims-vgpu/src/runtime/spirv_bind.rs",
        ty: "ImageFormatSpecializeError",
        loss: Loss::Refused,
        why: "the module's image bindings could not be specialised; the \
              translation fails rather than binding an unspecialised format",
    },
    Row {
        file: "crates/reims-vgpu/src/runtime/spirv_bind.rs",
        ty: "UnclassifiedBinding",
        loss: Loss::Ordering,
        why: "a `Binding` decoration whose variable the classifier could not \
              name; the binding's own band still names it, so the descriptor \
              lands where it belongs",
    },
    Row {
        file: "crates/reims-vgpu/src/runtime/spirv_layout.rs",
        ty: "SpirvLayoutDecline",
        loss: Loss::Refused,
        why: "the module's layout could not be read or repaired; translation \
              stops rather than emitting a module with wrong offsets",
    },
    Row {
        file: "crates/reims-vgpu/src/runtime/surface_cache/mod.rs",
        ty: "GvaCapDecline",
        loss: Loss::Ordering,
        why: "the byte cap could not be enforced because every entry was \
              protected, in flight or not guest-held. It is the cap that \
              yields, not the guest's data: nothing is dropped and the map \
              runs over until an entry becomes evictable",
    },
    Row {
        file: "crates/reims-vgpu/src/runtime/surface_cache/mod.rs",
        ty: "LinearMaterializeDecline",
        loss: Loss::SlowPath,
        why: "the linear entry could not be materialised from the resident; \
              the non-resident path writes the guest's pages instead",
    },
    Row {
        file: "crates/reims-vgpu/src/runtime/task_slot.rs",
        ty: "TaskWordDecode",
        loss: Loss::Refused,
        why: "the command's task word names a slot that is not live; the \
              command is refused rather than applied to a neighbouring task",
    },
];

/// Every `impl Decline`/`impl Refusal` in the crate has a verdict, and every
/// verdict names an impl that still exists.
#[test]
fn every_decline_says_whether_the_guest_lost_work() {
    let found = decline_impls();

    // Refuse a verdict until the scan has proved it can see the two impls that
    // spell the trait through its full path, which is the case a `grep -rn
    // 'impl Decline for'` misses. A scan that has gone blind reports a small
    // population every row of which is present, and reads exactly like a tidy
    // tree.
    for (file, ty) in [
        ("crates/reims-vgpu/src/runtime/gather_witness.rs", "GatherWitnessFault"),
        ("crates/reims-vgpu/src/runtime/mapping_write/mod.rs", "SurfaceWriteRefusal"),
    ] {
        assert!(
            found.iter().any(|i| i.file == file && i.ty == ty),
            "the scan did not find `{ty}` in {file}, which spells the trait \
             through its full path — so its notion of `every decline` is a \
             blind spot and not a measurement"
        );
    }
    assert!(
        found.len() > 50,
        "found {} decline impls, which is not this crate's vocabulary",
        found.len()
    );

    let mut unadjudicated: Vec<String> = Vec::new();
    for i in &found {
        if !ROWS.iter().any(|r| r.file == i.file && r.ty == i.ty) {
            unadjudicated.push(format!("{} ({})", i.ty, i.file));
        }
    }

    let mut stale: Vec<String> = Vec::new();
    for r in ROWS {
        if !found.iter().any(|i| i.file == r.file && i.ty == r.ty) {
            stale.push(format!("{} ({})", r.ty, r.file));
        }
    }

    // Both directions in one report, because the commonest cause of either is a
    // rename, which produces one of each — and asserting them in sequence shows
    // the author half the evidence and sends them to write a second verdict for
    // a type that already has one.
    let mut report = String::new();
    if !unadjudicated.is_empty() {
        report.push_str(&format!(
            "\nthese declines do not say what the guest loses when they fire. \
             Read the emitter — what the caller does *after* the decline is the \
             answer, not the variant's name — and add a row:\n  {}",
            unadjudicated.join("\n  ")
        ));
    }
    if !stale.is_empty() {
        report.push_str(&format!(
            "\nthese verdicts name declines that no longer exist. A verdict \
             left behind by a rename points at nothing and reads as \
             adjudicated:\n  {}",
            stale.join("\n  ")
        ));
    }
    assert!(report.is_empty(), "{report}");

    // One row per impl: a duplicate row is two verdicts about one type, and the
    // reader has no way to know which was meant.
    let mut keys: Vec<(&str, &str)> = ROWS.iter().map(|r| (r.file, r.ty)).collect();
    keys.sort_unstable();
    let before = keys.len();
    keys.dedup();
    assert_eq!(before, keys.len(), "a decline type carries two verdicts");
}

/// The list of places this device executes a command the guest did not ask for
/// only ever gets shorter.
///
/// A ratchet rather than an assertion of zero, because it is not zero. What it
/// buys is that the number cannot drift upwards while every individual commit
/// looks reasonable: adding an eighteenth truncation is a decision somebody has
/// to make on purpose, in this file, against a comment saying what the class
/// costs.
///
/// **Lower this when you retire one.** It is the only counter in the tree that
/// measures the standing goal directly.
#[test]
fn the_executed_modified_census_only_shrinks() {
    let census: Vec<&Row> = ROWS
        .iter()
        .filter(|r| r.loss == Loss::ExecutedModified)
        .collect();

    assert!(
        census.len() <= EXECUTED_MODIFIED_CEILING,
        "{} decline types can execute a modified guest command, above the \
         ceiling of {EXECUTED_MODIFIED_CEILING}. A GPU refuses; it does not \
         quietly do something else:\n  {}",
        census.len(),
        census
            .iter()
            .map(|r| format!("{} ({}) — {}", r.ty, r.file, r.why))
            .collect::<Vec<_>>()
            .join("\n  ")
    );

    assert_eq!(
        census.len(),
        EXECUTED_MODIFIED_CEILING,
        "the census is below its ceiling, which means one was retired and the \
         ceiling was not lowered. Lower it to {} so the ground that was won \
         cannot be given back",
        census.len()
    );
}
