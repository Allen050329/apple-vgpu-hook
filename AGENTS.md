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

The remaining reduction lever is runtime-dead code — paths that compile and are reachable but that
the protocol never takes — and `dead-state` cannot see that class. `/tmp/reims-vgpu-fail.log` is the
only ground truth for it, and it is untracked and disappears.

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
