# AGENTS.md

Operating guide for AI agents working in this repository.

## What Belongs In This File

Durable rules that change how an agent works: the principles below, the support matrix, the commands
that verify a change, and what a commit must say.

**Findings do not belong here.** A measurement, a counter reading, a sweep that came back empty, or
an account of how a past session was misled is not an instruction — and this file has repeatedly
grown to several times its useful length by collecting them. Put them where they will be read:

- **next to the code they explain**, as a module or function doc — that is where the next reader
  meets them, and it is the only place that stays true when the code moves;
- **in the commit body**, for what one change measured and did not verify;
- **in `kb/` and `journal/`** (both gitignored) for investigation notes, working hypotheses, and
  session logs. `kb/` entries carry frontmatter and `[[links]]`; follow the existing shape.

Before adding anything to this file, ask whether it changes what someone *does*. If it only records
what was once true, it goes in one of the three places above.

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
to x86, from Metal to Vulkan, or from one host GPU class to another. Some rails run on exactly one
pathway — the arm64-only mapper rail is the standing example — and no boot on the other host can
measure them.

## Main Components

- `vendor/qemu` - QEMU fork with the thin device shim: QOM, MMIO/BAR, IRQ/MSI, console/display
  integration, and HostOps plumbing.
- `crates/reims-vgpu` - Rust staticlib that owns protocol decode, device model, memory mapping,
  command planning/execution, scheduling, and Metal/Vulkan backend behavior.
- `crates/reims-vgpu/src/observe/` - crate-wide observability: fail logs, typed decline reasons,
  emission helpers, and gates.
- `crates/reims-vgpu-wire` - derived wire-format views, with their own `AGENTS.md`. Where that file
  is stricter than this one, it wins.
- `vm/` - snapshot-revert boot scripts for arm64 and x86 guests.

Start with the owning source modules and nearby tests when changing device, decode, present, or
backend behavior. Keep durable design facts in code comments close to the behavior they explain.

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

### No Magic Numbers

Do not guess numbers because they fit one observation. Derive constants from the contract: SDK
headers, `sizeof`/`offsetof`, decoded guest fields, documented serializer output, or controlled
empirical measurement. Record the basis in the code or the commit body when the value is not obvious.

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
could falsify them. One workload on one pathway proves one workload on one pathway.

### A Subagent Shares Your Working Tree

A delegated agent runs in this same checkout, so anything it does to git happens to you. Brief every
one of them read-only, by name: no `checkout`, `switch`, `stash`, `reset`, `restore` or `commit`.
The failure is quiet — an agent that runs `git checkout HEAD~1` to get a "clean build" and does not
return leaves HEAD detached, and the next commit lands off the branch where nothing but the reflog
can find it. Check `git status` after any delegated run before committing.

## Before A Broad Sweep

Deletion and audit sweeps over this crate have been run many times. What each concluded lives next
to the code it concluded it about — read the module doc before deciding a rail is dead, and do not
"discover" a comment recording a heuristic that was already measured and removed.

Four rules survive every sweep:

- **A never-firing branch is almost never a deletion.** A decoded-but-untaken arm is usually
  contract fidelity — a real Apple opcode this workload does not issue — or a healthy-zero alarm,
  where a firing *is* the bug. The test to apply: **name the guest action that would take this
  path.** If you can name it, it stays. Deleting one loses guest work silently the first time a
  guest takes it.
- **A zero can be an artifact of where it is sampled.** Find a census field's sampling point before
  cutting it; `scripts/constant-fields/README.md` lists the ways a constant reading is legitimate.
  A zero hit rate on one pathway is not a dead cache either — page shift alone changes it.
- **A drop counter reading zero is not a measurement.** A record stopping at slot 4 and one stopping
  at slot 30 both read zero, and only one says the bound has headroom. Band the *requested* reach
  before widening or narrowing any table.
- **Two arms that consume one wire form must be diffed against each other, not read alone.** The
  comment that settles a divergence is more often on the callee than on the call site, and the arm
  that is easiest to read is often not the arm the boot takes. Check which one runs before calling a
  divergence theoretical. When you find one, check whether its failure line is shared or copied — a
  copied one is the next divergence.

Prefer an instrument over a reading. Reading an audit against itself cannot see an opcode that is
simply the wrong number, a length four bytes off, or a field two bytes too wide:

| Question | Instrument |
|---|---|
| What compiles but is never referenced? | `scripts/dead-state/dead-state.sh` |
| What is reachable but never runs? | `scripts/runtime-dead` — coverage-instrumented driven boot |
| Does a decoder refuse or drop a record Apple emits? | `crates/reims-vgpu/tests/wire_fixtures_reach_the_decoders.rs` |
| Is a wire family declared twice, or read by nobody? | `crates/reims-vgpu/tests/wire_families_have_a_consumer.rs` |
| Does a doc comment name a symbol that no longer exists? | `cargo doc`'s intra-doc link pass |
| Does a value travel as loose parameters when a type for it exists? | `scripts/scattered-struct` |
| Is a validity rule written out at every site instead of beside its constant? | `scripts/scattered-bound` |
| Does a decoded record fail a guard and vanish into a no-op catch-all? | `scripts/silent-arms` |
| Does a constant crossing the C boundary have the test this file asks for? | `scripts/abi-pins` |
| Do two checks share a `reason=` slug, and so share `fail_once`'s latch? | `crates/reims-vgpu/tests/decline_slugs_are_unique.rs` |
| Is a Vulkan value spelled outside the `translate/` table that owns it? | `crates/reims-vgpu/tests/vulkan_state_enums_live_in_translate.rs` |
| Does anything reach a Vulkan 1.3 core name the 1.2 floor forbids? | `crates/reims-vgpu/tests/nothing_reaches_past_the_vulkan_api_floor.rs` |

Do **not** answer that one by diffing `ls src/ops/` against a `grep` for
`use reims_vgpu_wire` — that pair used to live here and it is wrong by 40 % on
this tree, because a family imported as `ops::{texture_view as w_view, ..}`
never puts its own name after the token `ops::`. The test above parses the brace
group, and refuses to report anything until it has proved it can see one.

```sh
RUSTDOCFLAGS="-A rustdoc::private_intra_doc_links" cargo doc -p reims-vgpu \
  --no-deps --document-private-items \
  --no-default-features --features backend-vulkan,host-window
```

Triage its output before editing anything, because most of it is not rot. Three classes, and only
the first is:

- **The symbol exists nowhere.** Real rot, and the only class worth a commit. Confirm with a grep
  for the leaf name; run the doc build on the Metal arm too (`--target aarch64-apple-darwin
  --features backend-metal`) and take the intersection, or a `backend-metal`-gated target will read
  as missing on the Vulkan arm.
- **A bare name inside a `//!` module doc.** These never resolve here whatever they name — a
  `pub fn` in that same module fails exactly as a deleted one does, and `self::` does not help.
  Only a fully-qualified `crate::…` path resolves from a `//!` doc. Cosmetic; the reference is
  correct, it just does not become a hyperlink.
- **An accurate path to a private item**, or to anything under `engine::pools` (a private `mod`).
  `--document-private-items` does not make these linkable across modules. Correct as prose; do not
  "fix" one by deleting a true reference.

A wire module with no importer is either a real gap or a family still declared twice. Where a device
offset names a field a wire struct already declares, reach for `offset_of!` rather than a re-exported
number, so a rename fails the build.

Their output is a map, not a kill list. And one trap they teach: **an `Ok` from `render::decode` is
not a decode** — `Kind::OtherAccepted` is the catch-all for "no arm claimed this", and reading it as
success hides a whole family of lost records behind a green run.

### Reading the fail log

- **Volume is not alarm.** Most records are on the `OFF` channel, and the highest-volume tags are a
  1 Hz heartbeat. That is cadence working, not an over-eager emitter.
- **Absence of a decode line proves nothing.** Every decoder in `decode/` is silent on success and
  emits only on `Err*`. "Opcode X never appears in the log" is not evidence that arm never ran; the
  `store_routes` counter set is the only usable never-fired signal.
- **Filter the channel before ranking `reason=`.** `OFF` records carry `reason=` too, for ordering
  and control-flow events that are not losses, so the obvious
  `grep -o 'reason=[a-z_0-9]*' | sort | uniq -c | sort -rn` inverts the queue. A fail-channel record
  begins with its own event name and an off-channel one begins with the literal `OFF `, so
  `grep -v '^OFF '` first.
- **A named reason on the fail channel is not automatically lost work.** Some report a repair that
  *succeeded*, fail-visible so the reliance stays measurable. Read the emitter.
- **A counter and a fail line count different things.** Census counters are per-window and
  cumulative; emitters dedupe. Do not quote one as the other.

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

**Host-pointer imports over guest RAM are forbidden.** `VK_EXT_external_memory_host` must never be
asked for, and Metal's `newBufferWithBytesNoCopy` may alias only this process's own bytes. Importing
a host pointer over guest RAM gives the host GPU read *and write* access to the guest VM's memory,
and that is a property of the mechanism rather than of how much of it is used — so the bound is
"never requested", not a budget. Both invariants are enforced by
`crates/reims-vgpu/tests/guest_ram_isolation.rs`.

**A dma-buf is the one mechanism that is allowed, and it is not that one wearing a new name.** The
GPU reaches guest pages through `VK_EXT_external_memory_dma_buf`, gated on
`caps::external_memory::DmaBufImport`. Three properties are why, and a proposal that loses any of
them is the banned mechanism again:

- **Bounded by construction.** The fd names an explicit list of page ranges chosen when it was
  created. A page not named then is not reachable through it — there is no pointer to stray from
  and no surrounding mapping to reach.
- **Revocable.** Closing the fd and freeing the `VkDeviceMemory` ends the access. A host-pointer
  import has no such handle, which is why "how much of it is used" was never the question.
- **Kernel-mediated.** The importing driver takes a reference on pages the kernel is tracking,
  rather than being handed a raw address it must trust.

So the deferred-flush rail — the device's largest cost — is retired by writing into guest pages
directly, which is what `storage_flush.rs` always said would retire it. Read that module's own
qualifier and the routes that are *not* blocked before assuming a window is safe to skip. Note that
`runtime/gva_view.rs::ensure_gva_view` hands back a host pointer but is not a window resolver — it
requires the span to be one contiguous page run and returns `None` otherwise.

Two consequences that are easy to miss. Guest RAM must be **fd-backed** for any of this to work
(`memory-backend-memfd,share=on` in the boot scripts); a plain `-m` allocation refuses every export
with `not_memfd` and the device silently falls back to copying, which is why that refusal is
fail-visible. And a dma-buf **pins** the pages it names — they stop being swappable or migratable —
so every cache holding one is bounded by *pinned bytes*, not by entry count. `runtime/guest_dmabuf.rs`
carries that bound and its basis.

**Neither extension is ever required.** A host that lacks them creates a device that asks for
neither and runs every guest-memory rail through the copying path, so the copying rails are not a
legacy arm — they are the only arm on most hosts and must keep working. Both halves are gated:
the import site on `HostGpuCaps::dma_buf_import`, and the export in `runtime/guest_dmabuf.rs` on the
rung `caps::external_memory` publishes, because exporting first and declining after pins guest pages
for nothing.

### Environment overrides

Every variable the crate reads is named in `crates/reims-vgpu/src/env.rs`, which also owns the parse.
Read a variable through it or the second spelling of "off" is a divergence nothing can find.

**An override may only narrow what the device does; it may never widen it.** A switch can turn off a
rail the host could have run. It can never turn on one the host reported it cannot: capability is
measured from the device, and binding an extension a host does not advertise fails `vkCreateDevice`
while importing a handle type it declines is undefined behavior inside the driver. Add a switch as a
new refusal reason, never as a new permission.

`REIMS_VGPU_DMABUF=off` is the one that matters for verification: it takes a capable host down to the
`disabled_by_env` rung, which is how the copying rails get exercised without hunting for hardware
that lacks the extension.

## Verification

Pick the pathway your change affects.

- Arm64: `vm/boot-arm64.sh --device reims-vgpu-mmio --testing`, then
  `scripts/screenshot-when-macos-host/screenshot-when-macos-host.sh /tmp/screen.png`
- x86: `vm/boot-x86.sh --device reims-vgpu-pci --testing`, then
  `scripts/screenshot-when-kde-plasma-host/screenshot-when-kde-plasma-host.sh -o /tmp/screen.png`

### A boot on a capable host does not exercise the copying rails

Where the import works, every guest window takes it, and the copying rails run zero times — so a
green boot says nothing about them, and they are the only rails on a host without the extension. A
change touching guest-memory upload, writeback or bind needs the boot a second time with
`REIMS_VGPU_DMABUF=off`. Confirm it took: `vk_caps` reports `dma_buf_import=disabled_by_env`, and
`guest_dmabuf_export off reason=backend_cannot_import` appears once. `guest_dmabuf_*` counters
should then read zero — a non-zero one means an export ran past a closed gate.

### An undriven boot measures an idle device

A `--testing` boot reaches the desktop and then sits there. Reading its counters as this device's
behavior is how a rail gets called dead when the workload simply never asked for it. If a change is
about throughput, caching, writeback or present cadence, the boot has to be driven.

Run the boot in the background and drive the guest **while it is up** — the `--testing` boot exposes
SSH on `localhost:2222` (`macos-vm` in `~/.ssh/config`) for its whole life, so a probe does not need
its own boot:

```sh
vm/boot-x86.sh --device reims-vgpu-pci --testing &     # ~7 min before its own hard kill
ssh macos-vm true                                      # wait for the guest to answer
scripts/window-drag-probe/window-drag-probe.sh --seconds 25 --app Safari
```

That produces real window-server compositing, against 0 draws/s idle. The probe refuses a verdict if
the window never moved, so a run that produced no motion cannot be mistaken for a slow device.

### A boot measured next to your own subagents measures the contention

Every `us=` number this device reports is wall clock on a shared machine, so a driven boot taken
while a subagent greps, a `cargo` build runs, or a second VM lives is measuring your harness as much
as the device. This does not look like an error — the log is well-formed and the counters are
self-consistent — and it has been measured to halve throughput, triple per-draw cost, and invert the
ranking between the device's two largest costs.

**Run the boot with nothing else running, and check `uptime` before believing a timing.** Counts are
far more robust than timings: `store_routes`, refusal counters and the gate do not measure time and
survive contention. When a machine cannot be quiesced, reason from counts and treat every `_us`
field as an upper bound.

### Rust tests

Run the relevant native tests serially from the repo root:

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

**The only layout-truth tests do not run on a checkout without Apple's captured fixtures**, which is
every non-Apple checkout — `crates/reims-vgpu-wire/fixtures/` is gitignored. They report `ignored`
rather than `ok`, so the run says so, and the ignored count is the one to read. Nothing else in
either suite covers what they cover, so a green run is not evidence about a wire layout. Regenerate
with `scripts/wire-oracle/wire-oracle.sh` on an Apple host, and set `REIMS_WIRE_FIXTURES_REQUIRED=1`
there so their absence fails the build.

The `backend-metal` `--lib` arm has six pre-existing failures, all in
`runtime::storage_flush::tests` and all exercising `flush_render_one`, which is a fail-visible stub
on a build without `backend-vulkan`. They are Vulkan-rail tests compiled unconditionally rather than
a Metal defect. Check the failing *module and count*, not the pass count — the pass count moves
whenever anyone adds a test. Do not "fix" them by weakening what they assert.

## Commit Guidelines

Commit only work you wrote. Never commit third-party code or intellectual property, including Apple
software, firmware, disk images, `.mtlb`, AIR, or SPIR-V, or a disassembly listing of any of them.
Keep those artifacts ignored and local. Reports may include original analysis, metadata, hashes, and
reproduction steps, but no third-party bytes or excerpts.

Each commit should have a detailed message body that states:

- Which component or pathway it touches.
- What behavior changed and why.
- What tests, clippy runs, feature-matrix checks, or live-VM verification were performed.
- What was not verified, if anything.

Rust commits should be warning-free under clippy with `-D warnings` for every affected matrix arm.
**All three run on a Linux host** — the Metal arm needs its `--target`, and with it clippy analyses
the `backend-metal` code without an Apple machine:

```sh
cargo clippy -p reims-vgpu --target aarch64-apple-darwin --all-targets --no-default-features --features backend-metal -- -D warnings
cargo clippy -p reims-vgpu --all-targets --no-default-features --features backend-vulkan,host-window -- -D warnings
cargo clippy -p reims-vgpu --target x86_64-unknown-linux-gnu --all-targets --no-default-features --features backend-vulkan,host-window -- -D warnings
```

Expect zero from all three. **`scripts/feature-matrix` does not cover this**: it runs `cargo check`,
so its `warnings=0` is a rustc count and it cannot see a clippy lint on any arm. That gap plus a
"the Metal command is Apple-only" line that used to sit here is how a `clippy::question_mark` in
`runtime/metal_draw/mod.rs` survived several commits that each said "clippy clean" — every one of
them was clean on the arms it ran, and nobody on a Linux host ran the Metal one.

Do not hide warnings, skip an affected arm, or commit a dropped test
count without calling it out — and **do not read "clippy clean" in a commit body as covering every
arm**; it means the arms that commit ran.

Two standing exceptions, both carried by `#[allow]`s at the module declarations that state the
reason. `backend::metal::error::Status` is large by design — the payload is what makes each refusal
name the check that refused, and it is `Copy` and compared by value at hundreds of sites — so
`result_large_err` and `large_enum_variant` are exempted there. **A new error type that is large for
no such reason should still be boxed**, not added to the exemption. Separately, the
`transmute::<u64, MTL*>` sites turn a guest-decoded ordinal into a `#[repr(u64)]` Metal enum, where
an out-of-range value is undefined behavior rather than a decode error; they are greppable and
nothing range-checks them yet. Do not add more.
