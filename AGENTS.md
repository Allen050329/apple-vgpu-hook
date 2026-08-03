# AGENTS.md

Operating guide for AI agents working in this repository.

## What This Project Is

This research project emulates Apple's paravirtualized GPU on the host. An unmodified macOS guest
uses Apple's own GPU drivers; our QEMU device and Rust backend decode the command stream and
execute it through Metal or Vulkan. We ship no guest driver.

`crates/reims-vgpu` supports three first-class pathways:

| Pathway | Host | Guest | Attach | Page shift | Backend | Boot |
|---|---|---|---|---|---|---|
| x86 macOS / Linux Vulkan | Linux x86_64 (KVM) | x86_64 macOS Metal guest | PCI (`reims-vgpu-pci`) | 12 | Vulkan | `vm/boot-x86.sh` |
| arm64 macOS / macOS Metal | Apple Silicon macOS (HVF) | arm64 macOS Metal guest | sysbus MMIO (`reims-vgpu-mmio`) | 14 | Metal-direct | `vm/boot-arm64.sh` |
| arm64 macOS / macOS Vulkan | Apple Silicon macOS (HVF) | arm64 macOS Metal guest | sysbus MMIO (`reims-vgpu-mmio`) | 14 | Vulkan through MoltenVK | `vm/boot-arm64.sh` |

Pathway-specific facts must be verified on the pathway being changed. Do not generalize from arm64
to x86, from Metal to Vulkan, or from one host GPU class to another.

## Main Components

- `vendor/qemu` - QEMU fork with the thin device shim: QOM, MMIO/BAR, IRQ/MSI, console/display
  integration, and HostOps plumbing.
- `crates/reims-vgpu` - Rust staticlib that owns protocol decode, device model, memory mapping,
  command planning/execution, scheduling, and Metal/Vulkan backend behavior.
- `crates/reims-vgpu/src/observe/` - crate-wide observability: fail logs, typed decline reasons,
  emission helpers, and gates.
- `vm/` - snapshot-revert boot scripts for arm64 and x86 guests.

Start with the owning source modules and nearby tests when changing device, decode, present, or
backend behavior. Keep durable design facts in tracked docs or code comments close to the behavior
they explain.

## Operating Principles

### C Is A Thin Shim

C and Objective-C in the QEMU path exist to connect QEMU to Rust. Keep product logic in
`crates/reims-vgpu`: protocol interpretation, resource state, scheduling, GPU encode, present model,
backend policy, and performance behavior belong in Rust.

A shim that calls two queries and branches on the pair has reconstructed a rule, which is the same
violation as writing one. Export the answer, not the inputs — and delete the inputs, because a shim
that can still assemble its own answer eventually will.

Anything crossing the boundary lives twice, once in Rust and once in
`crates/reims-vgpu/include/reims_vgpu_qemu_abi.h`, and nothing in the toolchain compares the two:
Rust does not include the header and the shims do not read Rust. Every constant that crosses gets a
test, using `qemu::abi::header_define` — see `the_abi_header_agrees_on_the_version`,
`..._on_the_scanout_bound` and `..._on_the_console_feed_kinds`. Add one with any new shared
constant; a drift here is a bug on exactly one pathway.

Verifying a shim change needs the pathway that runs it. The default
`vm/boot-x86.sh` config sets `REIMS_VGPU_WINDOW=1`, so QEMU takes `-display none` and **never calls
`gfx_update`** — a boot like that does not exercise `fb_update` or `apply_scanout` at all. Use
`REIMS_VGPU_WINDOW=0` for those, and A/B against a stashed baseline: that console renders black on
both arms, so a screenshot only means something next to the baseline's.

### Never Fail Silently

If a decoded guest command is rejected, dropped, degraded, unsupported, or mis-executed, make the
reason visible. Use typed decline/refusal reasons and emit them through the always-on failure path
so `/tmp/reims-vgpu-fail.log` explains what happened.

Expected control flow should stay quiet. A resolver saying "not ready yet" or an intentionally
unbound `ref == 0` is not a failure. A real loss of guest work is.

### Measure Before Fixing

You cannot fix what you cannot measure. If we do not know what class of failure we are fixing, we
are operating blind and guessing.

Before landing a visual, protocol, performance, or translation fix, add or identify a log-level or
test-level proxy for the bug class. Screenshots are useful evidence, but they are not a regression
gate by themselves.

### Tests Define Done

If there is no test for it, it is not done. No test means a future agent can regress the changeset
without noticing.

Behavior changes need tests that fail without the change. Bug fixes need a focused synthetic case
or proxy test for the bug class. Run Rust tests serially with `-- --test-threads=1`; GPU-touching
tests are not safe to run in parallel.

### Do Not Overfit Fixes

Never special-case behavior for a screenshot, boot stage, pixel dimension, resource size, object id,
function name, pipeline ref, or observed content pattern. Implement the decoded API contract.

Temporary probes are fine when they collect evidence. Remove probe-only behavior before claiming the
fix. Do not turn observations into product heuristics.

For metal2vulkan, do not make translation pass by matching corpus names. Handle the structural AIR
or LLVM semantics, or leave the gap visible.

### No Magic Numbers

Do not guess numbers because they fit one observation. Derive constants from the contract: SDK
headers, `sizeof`/`offsetof`, decoded guest fields, documented serializer output, or controlled
empirical measurement. Record the basis in the code, tracked docs, or the commit body when the
value is not obvious.

Guest page geometry is always explicit. Portable code takes `page_shift` or `page_size`; arch-fixed
helpers must say so in their names.

### Write Comments For The Code

Inline comments should explain the code as it exists now. Do not mention the prompt, a temporary
plan, a phase number, a section of an implementation plan, or the reason a plan asked for the
change. Plans are consumed and deleted; code comments outlive them and must not point at things
that no longer exist.

### Keep Claims Narrow

State exactly what you verified. A single green boot does not prove an entire class is fixed. Broad
claims such as "zero-copy everywhere" or "no fallback remains" require an audit of every place that
could falsify them.

## Where Code Reduction Stands

Read this before opening a broad "what can be deleted" sweep. Each line below is a search that has
already been run to exhaustion; re-running one costs hours and has so far returned nothing.

- **Compile-dead items: swept.** `scripts/dead-state/dead-state.sh` is the instrument, and its
  intersection report is empty. Its one remaining lead is the ~45 items dead on the Metal arm
  (`--keep`, `hits.metal`) — those are **not** deletions. A shared census that Metal never calls is
  more likely an observability gap on a first-class pathway than dead code, and a `cfg` would cement
  it invisibly against Never Fail Silently. Deciding needs an Apple host.
- **Heuristics and fallbacks: none of substance left.** What reads like a fallback is a device
  capability query (depth/vertex format, memory topology), a real two-path strategy, or a comment
  recording a heuristic that was already measured and removed. Do not "discover" the latter and
  delete the explanation.
- **Censuses are not bloat.** The phase censuses (`chain_phase`, `bind_phase`, `draw_phase`,
  `stage_phase`) are ~55 lines of code each under their docs, they reconcile against each other by
  construction, and they are how "slow" gets diagnosed. A never-firing *decline* is a quiet failure
  path, which is the healthy state — not a dead branch.
- **Structural unification has been costed and rejected twice.** The three sampled rails (linear,
  type-11, type-5) differ in wire source, stride handling and bounds, not in naming; the two
  memoized loaders differ in read, conversion, census and return type. In `backend/vulkan`, merging
  the keyed caches loses per-key bucketing and negative caching, and a shared barrier helper puts a
  stage/access-mask bug in both the graphics and compute paths at once. The measured savings were
  tens of lines each. Do not re-derive this.
- **A zero hit rate on one pathway is not a dead cache.** `gva_view`'s `view_reuse` reads 0 on
  x86/Vulkan because a 12-bit page shift fragments nearly every span. A 14-bit shift covers the same
  span in a quarter of the pages. The module documents this at its own reuse site.
- **A census field that is zero on every sample can be zero because of where it samples.**
  `host_cache_levels surfaces/surface_bytes/surface_largest` read 0 across 4 896 samples while the
  two tiers beside them hold 170 MB. The surface tier is not dead: the census runs at the drain
  tail, every armed render window lands inside that drain, and every landing is leased and forgets
  its entry — so the tier is guaranteed empty at the instant it is read, and a **non-zero** reading
  is the leak alarm. Documented at `cache_levels` in `runtime/surface_cache.rs`. Before cutting a
  constant field, find its sampling point; `scripts/constant-fields/README.md` lists five ways a
  constant can be legitimate and this is a sixth.

- **Four more large modules have now been swept, and three returned nothing.** `storage_flush.rs`,
  `mapper.rs` + `mapping_write.rs`, `scanout.rs` + `host_window/present.rs` + `surface_cache.rs` +
  `window_present.rs`, and `decode/resource.rs` + `model/state.rs` + `objects.rs`. **No oracles were
  found in any of them** — no PFN plausibility heuristic, no poison-pattern check, no stride or
  geometry guessing, no multi-interpretation resolve ladder, no retry loop. What the sweep did find
  was two *speculative* rails with no producer (both deleted) and two decoders overriding a decoded
  field (both fixed). Re-running the oracle hunt over these files is not worth the hours.
- **A never-firing dispatch arm is almost never a deletion.** An audit of all 85 `store_routes`
  names against two driven boots found 39 that never fire. Every one resolved into either contract
  fidelity — a real Apple opcode this workload happens not to issue (`icb_exec_seen`,
  `compute_ctrl_seen`, the five `compute_noop_*` fence/barrier/residency arms) — or a healthy-zero
  alarm, where a firing *is* the bug (23 of them, all drift/unbounded/unwitnessed detectors). None
  were reducible. Deleting a decoded-but-untaken arm loses guest work silently the first time a
  guest takes it.
- **The decode fidelity vein is `resource.rs`, not `decode/` generally.** Four real decoder bugs
  landed recently — a field overridden from a command mask, a format chosen by magnitude, a value
  read from the wrong record when the preferred one was unreachable, an unconditional `has_` flag —
  and every one was in `decode/resource.rs`. The other six decoders (`render`, `compute`, `blit`,
  `fifo`, `stream`, `event`, ~3 750 lines) were then audited for those same four shapes plus silent
  drops, and came back clean: `has_*` flags are conditional on the read that sets them, draw and
  copy forms are distinguished by exact length rather than magnitude, variable-length records
  bounds-check before access, and every overflow path names its own `ErrBadLength` slug. Spend the
  next fidelity hour on `resource.rs` and the descriptor formats, not on re-reading these six.
- **The consumer side has now been swept too, and the vein there is *divergence between two arms
  that consume one wire form*.** `blit_exec`, `drain`, `exec`, `compute_exec`, `compute_session` and
  `metal_draw` (~30 000 lines, the clusters the earlier oracle sweep did not reach) were audited for
  the same shapes. Almost no classic oracles came back — no poison/sentinel checks, no retry loops,
  no PFN plausibility test, no magnitude-based format selection, and the one thing that looks like a
  resolve ladder (the four-rung type-11 sampled path) is four distinct sources with a traffic census,
  not one source reinterpreted. What *did* come back was five real bugs, and four of them are the
  same shape: **two consumers of one decoded record, one of which contradicts a rule the other one
  states in a comment.** A nil compute bind entry did not unbind while `exec::apply_binds` did and
  `ExecResult::buffer_unbinds` said it must; a Metal colour slot borrowed slot 0's blend state while
  the Vulkan arm's comment named that exact line as inventing state; seven root FIFO arms dropped a
  short packet silently while the child arm of one of the same opcodes already reported it; two row
  loops skipped the `dest_window` bound six sibling call sites take. When auditing here, diff the two
  arms against each other rather than reading either alone — and grep the *other* arm's comments,
  because in three of these four the correct rule was already written down next to the wrong code.
- **The one remaining oracle-shaped thing is in the mapper, and only an Apple host can settle it.**
  `contract/iosurface_pages::build_table_plan` reaches the IOSurface page table through *two*
  candidate chases — `MappingInternal` field `+0x48` then `+0xb8`, and field `+0x50` then `+0x28` —
  builds a candidate from each field that holds a kernel VA, and returns the entries of whichever
  candidate `read_table_entries` parses first. Read cold that is the classic "try both, keep the one
  that works" ladder, and `MapperMem::read` beside it picks `read_kva` vs `read_gpa` by testing the
  address against the kernel-VA ranges rather than by knowing which space the field is in.
  **Do not change either from an x86 host.** This whole rail is arm64-only: it is entered from
  `mapper::capture_at_producer`, which needs `HostOps::read_xreg`, and the x86 PCI shim returns -1
  for that unconditionally (`reims_vgpu_pci_read_xreg`, "no arm xreg handoff"). Nothing here
  executes on the x86/Vulkan pathway, so no boot on this host can measure it and no x86 evidence can
  justify touching it.
  It also may not be a ladder at all. The two-candidate branch only *chooses* when both fields hold
  kernel VAs at once; if exactly one is ever populated, this is two layouts handled side by side and
  is fine as written. Settling it needs one arm64 boot and two counts: how often both fields are
  populated, and — when they are — whether candidate `+0x48` ever fails where `+0x50` then succeeds.
  If that pair never happens, delete the fallback and keep the field test. If it does, the
  discriminator is a real contract gap and the fix is to find the field that selects between them,
  not to keep parsing both.
- **The x86 GVA page-table walk is exactly conformant; do not re-audit it.** `contract/gva.rs` and
  `contract/gva_resolve.rs` were checked field by field against the contract and every constant
  holds: a PTE is 4 bytes, the PFN is bits `[30:0]` raw (`PTE_PFN_MASK`), bit 31 is a flag, the
  fan-out is 1024 (`X86_64_INDEX_BITS = 10`), the page shift is 12, and the index split
  `(page_index >> ((depth-1-level) * index_bits)) & index_mask` is the right one. The one subtlety
  is already right and is worth not "simplifying": **`pte == 0` is the guest's sole not-present
  encoding**, so the walk reports `ErrZeroPfn` for it and `ErrMalformedPte` for a zero PFN with bit
  31 set. That second case cannot occur legitimately — the guest refuses to store an entry whose
  bit 31 is already set and never maps physical page 0 — so collapsing the two arms would discard a
  real corruption signal to save a branch. `gva_zero_pfn` in the fail log is therefore the guest
  saying "not mapped here", not a device defect.
  `MAX_DEPTH = 4` is a bound, not the depth: the depth is read per task from the task descriptor
  (`DIRECTORY_DEPTH`), which is correct and must stay — do not hardcode it even though the x86 guest
  currently always says 3.
- **The type-4 task search is not an oracle, and replacing it with "the task the guest named"
  regresses the boot.** It reads exactly like one: `resolve_type4_surface_ex` takes a bare surface id
  and probes up to 256 task object lists — task 0 first as the "historical home", then a hint its own
  doc calls "allowed to be wrong" — accepting the first list that yields a translatable backing,
  while the sibling call on the line above (`resolve_type11_ref`) is already being passed `task_id`.
  Threading that same `task_id` in and deleting the search took a driven x86 boot from **0
  `rt_resolve FAIL` and 6 `present_unbacked` to 21 816 and 2 774**, with the desktop unbacked.
  The reason is the thing the id spaces hide: **an IOSurface is cross-process.** It is created in one
  task and referenced from another, so the task whose command stream names a surface id is routinely
  *not* the task whose object list holds it — the naming task holds the type-5 *view*, and the
  descriptor inside it points at a surface owned elsewhere. The search is how the owner is located,
  and there is no task word on the wire that answers it. `blit_exec`'s "never the task object-list
  ref — those id spaces collide" is about the id *spaces*, not about which list to read, and reading
  it as the latter is what makes this look like a settled bug.
  The narrower worry that used to sit here — the search takes the first list that *translates*, and
  two tasks could both hold an `OBJECT_TYPE_SURFACE` at the same slot — has now been measured, and it
  does not occur. There is indeed nothing to verify a candidate against: the object-list entry is
  twelve bytes, `[type | desc_len]` plus `desc_gva`, with no identity field, and the type-4
  descriptor is fully consumed — its only undecoded span is three bytes at `0x11` that read
  `undecoded_nz=0` on every distinct shape a driven boot produces. So `type4_claimants` counts
  instead. On a driven x86 boot: **87 distinct surface ids, `claims=1` on every one, `winner=0` on
  every one**, across 16 defined tasks. Every type-4 IOSurface lives in task 0's object list and only
  task 0's, and every successful resolve stops on the first probe.
  That sharpens why the task threading was reverted: surfaces live in task 0 while the command
  streams naming them belong to other tasks, so threading the naming task looks in the one list that
  structurally cannot hold them. It also means the search's other 255 probes contribute to no
  successful resolve on this workload — but do not delete them on that reading alone. It is one
  workload on one pathway, and `claims > 1` is a live failure-channel alarm that will say so if the
  assumption ever breaks.

- **Two arms of one guest-memory write must be diffed against each other, not read alone.** This is
  the `mapping_write.rs` instance of the consumer-side finding above, and it has now produced three
  bugs in a row: a `span_end` bound present on two arms of three, two entries draining deferred
  windows only on the scattered arm, and the staged arm zeroing inter-row padding the contig arm
  preserves. The trap is that **the arm that is easiest to read is not the arm that runs**: on a
  driven x86 boot `write_split` reads `contig=0` on essentially every window, so the fragmented
  staging arm is the live one and the pointer arm is nearly dead. Check which arm the boot takes
  before concluding a divergence is theoretical.
  Note also what is *not* a divergence there: `write_rect_raw_at_impl`'s `full_tight_direct` fast
  path returns early and skips the shared `invalidate_storage_residency_window`, but its own guard
  requires `frame_len == span_end - base_off` and `write_mapping_bytes` invalidates `[off, off+len)`
  internally — the same window. That one has been checked; it is correct.
- **The per-dispatch compute stall watchdog is priced, and it is not worth rewriting.**
  `spawn_compute_engine_stall_watchdog` spawns a thread and clones the SPIR-V on *every* compute
  dispatch, then sleeps 2 s. At the measured peak of 124 computes/s that is ~250 live sleeping
  threads and 124 spawns/s for a probe that has never fired (0 hits across a driven boot, and no
  `/tmp/reims-vgpu-compute-stall-*` dump has ever been written). It reads like obvious bloat. It is
  ~0.25% of one core and a few MB of RSS — three orders of magnitude below the flush rail above —
  and it is a healthy-zero hang alarm for backend calls a Vulkan fence timeout cannot bound. Collapsing
  it to one long-lived thread plus an in-flight registry is correct but buys almost nothing; do the
  flush rail first.
- **The two C shims are not 2 200 duplicated lines.** `reims-vgpu-pci.c` and `reims-vgpu-mmio.c` look
  like near-copies and are read that way by every fresh sweep. What is actually shared is already in
  `reims-vgpu-shim.c`, and the rest is blocked by that header's own stated rule — bus-specific trace
  events and the per-device dirty tracker stay in their shim. `gfx_read`/`gfx_write` and the
  `MemoryRegionOps` (~77 lines) touch both and cannot move. The eight dirty-tracking wrappers (~60
  lines) *are* byte-identical apart from the `ctx` cast, but `HostOps.ctx` is genuinely per-device —
  `schedule_bh`, `map_pages`, `read_xreg` and `notify_actions` all need the bus object — so
  collapsing them needs a second ctx field in the ABI, which buys 30 net lines for a new drift
  surface. What the shims *did* hold was a real rule, and that is now exported (`scanout_may_paint`);
  look for reconstructed rules there, not for duplicate bodies.

The remaining reduction lever is runtime-dead code — paths that compile and are reachable but that
the protocol never takes — and `dead-state` cannot see that class. `scripts/runtime-dead` is the
instrument: it builds the staticlib with `-C instrument-coverage`, boots x86, drives the guest, and
reports every function whose counter stayed at zero. One boot, whole crate, function granularity.
It replaces reading `/tmp/reims-vgpu-fail.log` for this, which was never good at it — every decoder
in `decode/` is silent on success, so absence of a line proves nothing.

**Its output is a map, not a kill list.** First run: 1 066 of 2 826 functions never ran on a driven
boot. `runtime/icb/mod.rs` reads 0.00%, which is the correct reading and is already priced as *not*
a deletion — the instrument agreeing with a conclusion reached the expensive way is why to trust it,
not a reason to act. Six files read 0.00% and none were deleted on the strength of it. The reasons a
zero is legitimate — a decline that healthily never fired, a real Apple opcode this workload does
not issue, a path the one driven workload never asked for, an error path, the other pathway's
geometry — are in `scripts/runtime-dead/README.md`, with the test to apply: **name the guest action
that would take this path.** If you can name it, it is contract fidelity and it stays.

### Reading the fail log

Two things about its shape, so a reader does not mis-size what they are looking at:

- **Volume is not alarm.** Across a driven boot only ~1.7 % of records are on the `fail` channel;
  the rest are `OFF`. The seven highest-volume tags (`window_publish`, `host_cache_levels`,
  `guest_write_footprint`, `engine_lock`, `engine_delta`, `drain_duty`, `host_window_loop`) are
  ~59 % of all records and are **one 1 Hz heartbeat** — inter-record deltas of 1000-1003 ms. That is
  cadence working, not an over-eager emitter.
- **Absence of a decode line proves nothing.** Every decoder in `decode/` is silent on success and
  emits only on `Err*`. So "opcode X never appears in the log" is not evidence that arm never ran.
  The only usable never-fired signal is the `store_routes` counter set.

### Units

`draw_stall`'s headline `us=` field once carried nanoseconds while its thirteen siblings carried
microseconds — a 1000× error on the device's own first-look number for a slow draw. The rest of the
tree was then audited for the same class: every other `*_us` field traces to a variable already in
microseconds. Fixed and checked; do not re-run this sweep.

## Support Matrix

arm64 and x86 are both first-class. Metal and Vulkan are both first-class where the host supports
them.

The Vulkan backend must support all four memory/DMA cells:

| | DMA available | No DMA available |
|---|---|---|
| Unified memory | Apple M-series / MoltenVK, Intel/AMD iGPU on Mesa | Unified-memory hosts with no sharing mechanism |
| Discrete memory | Discrete GPUs with a working sharing path | Discrete GPUs that require copy crossings |

Vulkan 1.2 is the baseline. Anything above Vulkan 1.2 must have a fallback or a capability-gated
path. Gate on capabilities, not vendor names, driver names, or API-version assumptions.

Host-pointer imports must stay windowed. Do not import the whole guest RAM VMA for GPU DMA; it pins
host RAM. Use the existing capped window resolver.

## Verification

Pick the pathway your change affects.

- Arm64: `vm/boot-arm64.sh --device reims-vgpu-mmio --testing`, then
  `scripts/screenshot-when-macos-host/screenshot-when-macos-host.sh /tmp/screen.png`
- x86: `vm/boot-x86.sh --device reims-vgpu-pci --testing`, then
  `scripts/screenshot-when-kde-plasma-host/screenshot-when-kde-plasma-host.sh -o /tmp/screen.png`

### An undriven boot measures an idle device

A `--testing` boot reaches the desktop and then sits there. Its counters are the *idle* device's,
and reading them as this device's behaviour is how a rail gets called dead when the workload simply
never asked for it. If a change is about throughput, caching, writeback or present cadence, the boot
has to be driven.

Run the boot in the background and drive the guest **while it is up** — the `--testing` boot exposes
SSH on `localhost:2222` (`macos-vm` in `~/.ssh/config`) for its whole life, so a probe does not need
its own boot:

```sh
vm/boot-x86.sh --device reims-vgpu-pci --testing &     # ~7 min before its own hard kill
ssh macos-vm true                                      # wait for the guest to answer
scripts/window-drag-probe/window-drag-probe.sh --seconds 25 --app Safari
```

That produces real window-server compositing — a measured 499 draws/s at drain duty 0.97, against
0 draws/s idle. The probe refuses a verdict if the window never moved, so a run that produced no
motion cannot be mistaken for a slow device.

### The probe's Hz verdict is upstream of the present path, not in it

`seconds below 100 Hz: 23/23` reads like the host window is dropping frames. It is not. Across a
driven boot's 36 `host_window_cadence` windows, **35 have `presents == offered` with `busy=0`**, and
the one exception still presented all 19 frames it was offered (`busy_fence=2`). `direct_frac` is
1.00 throughout. The window presents every frame it is handed, immediately, and stalls on
essentially nothing.

`present_hz` therefore tracks `offered_hz` exactly, and both measure what the guest and the drain
*produced* — 17-35 fresh frames a second while the compositor ran at 496 draws/s, of which
`draws_fresh` is 11-17 and the rest are the loop correctly declining to re-present unchanged
content. So a slow verdict from this probe points **upstream**: drain, decode, draw. It does not
implicate `host_window/present.rs`, `backend/vulkan/engine/window_present.rs`, or the surface-cache
present path, and measuring those again to explain a low Hz number is measuring the wrong end.

### Upstream is the flush rail, and it is bytes

The three candidates that verdict leaves — drain, decode, draw — are not equal, and the driven log
already settles it. In the busiest window `drain_duty` reads `flush_us=732687` of `drain_us=995063`
against `draw_us=225287`: **73% of the device's entire time budget is writeback, 3.2× draw.**
`compute_us` is 0. `flush_rails` puts essentially all of it on the render rail (`render_us=717130`
over 304 flushes; gva, linear and storage rails are ~0).

Do not then read `readback_split`'s `fence_us` as latency. `gpu_us`/`bar_us` are GPU timestamps
taken *inside* that fence: `fence_us=410022` with `gpu_us=324787` and `bar_us=729` means 79% of the
fence is the readback command buffer's own copy and the barrier waiting on the draw batch is one
microsecond per fence. Adding `write_us=290863` and `map_us`, the rail is **86% bytes, 13% latency**.
720 fences that second each copied a whole surface to produce 11-17 fresh frames.

So the lever is bounding *what* a flush copies, not scheduling it differently, and it is not blocked
on a host that can address guest memory the way the zero-copy endgame is. `flush_render_one`'s doc
carries the full reading, including the separate measurement that 99.3% of what the rail writes is
never read by anything in the device before the next flush replaces it. Note also that a
tile-difference delivery path was built and deleted whole in `6df980c` for being reached by nothing
— read that commit before rebuilding it.

For Rust changes, run the relevant native tests serially from the repo root:

```sh
cargo test -p reims-vgpu --no-default-features --features backend-vulkan,host-window -- --test-threads=1
cargo test -p reims-vgpu --no-default-features --features backend-metal -- --test-threads=1
```

`backend-metal` is Apple-only; run that arm only on Apple hosts. Run the feature matrix from the
repo root when cfgs, features, backend boundaries, or shared Rust code change:

```sh
scripts/feature-matrix/feature-matrix.sh
```

Before and after long Rust test runs, sweep orphaned test binaries:

```sh
pkill -9 -f 'target/debug/deps/reims_vgp[u]-'
```

## Commit Guidelines

Commit only work you wrote. Never commit third-party code or intellectual property, including Apple
software, firmware, disk images, `.mtlb`, AIR, or SPIR-V. Keep those artifacts ignored and local.
Reports may include original analysis, metadata, hashes, and reproduction steps, but no third-party
bytes or excerpts.

Each commit should have a detailed message body that states:

- Which component or pathway it touches.
- What behavior changed and why.
- What tests, clippy runs, feature-matrix checks, or live-VM verification were performed.
- What was not verified, if anything.

Rust commits should be warning-free under clippy with `-D warnings` for every affected matrix arm.
Use the appropriate subset for the host and change; the Metal command is Apple-only:

```sh
cargo clippy -p reims-vgpu --all-targets --features backend-metal -- -D warnings
cargo clippy -p reims-vgpu --all-targets --no-default-features --features backend-vulkan,host-window -- -D warnings
cargo clippy -p reims-vgpu --target x86_64-unknown-linux-gnu --all-targets --no-default-features --features backend-vulkan,host-window -- -D warnings
```

Do not hide warnings, skip an affected arm, or commit a dropped test count without calling it out.
