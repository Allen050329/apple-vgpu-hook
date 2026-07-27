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

**Identify comes before add.** The always-on lines already carry more than they look like they do,
and slicing an existing one per case is faster than writing a probe. A defect here was traced to a
missing multi-plane sampling conversion, the gap was real, and the fix site had been located — and
slicing the existing `type4 pages … planes= multi=` line per case stopped that fix cold before a
line of it was written. That was the right move and it cost minutes.

**Then say exactly what you ruled out.** The claim drawn from that slice was "not YUV", and it was
too broad twice over. First, the verbose draw log later showed both cases working in per-plane `R8`
and `RG8` views — chroma subsampling that a surface-level plane count cannot see. Second, and worse,
even the narrow reading did not survive: see "An event count is not a state" below, which is the same
probe, re-measured, coming out the other way. Name the path a probe covers, not the idea it seemed
to kill — and expect the narrow claim to be the only part that lasts, if any of it does.

So when a mechanism looks obvious, spend the cheap step first: find a line already emitted on both
sides and check that it separates them. Then check the converse — that it would have *shown* the
mechanism had it been present. A probe that cannot distinguish the cases is not evidence in either
direction.

**An event count is not a state.** Separating the cases is necessary and it is not sufficient: the
line also has to measure the quantity your claim is about. An always-on log is a record of
*transitions*, so a count of zero means "this did not happen during the window", never "this does not
exist". Any claim of the form "the failing case never has X" needs a probe that reads X, not one that
fires when X is created.

This has cost two iterations here. A surface-attach line was sliced per case, showed a two-plane YUV
surface in the working case and none in the failing one, and that became a standing exclusion. The
line's own comment says it fires on first attach and re-fires per recycle — so a surface attached
during the previous case and still live is used by the next one in complete silence. On re-measure
the failing case showed the same count as the working one, and the exclusion had to be struck. The
same absence-is-not-evidence trap applies to any "we never saw a decline for it" argument.

**A reason the caller writes is not a reading.** The most convincing thing in a log is a typed
`reason=`, because it looks like the code telling you what happened. Check that it is. If the callee
returns a bare `bool` and the *caller* supplies the word, the field carries the caller's assumption at
full confidence, and it gets quoted back later as though it had been measured.

An iteration here read `gva_write fail reason=not_contig` on a dropped writeback and reasoned about
which fragmented spans could have produced it. The callee had six distinct refusals and returned
`bool`; the caller printed `not_contig` for all six. The label was also impossible on its face — that
writer goes one row at a time, the row was 512 bytes, and a write inside a single guest page cannot
be non-contiguous. Typed and re-run, every instance was `gva_zero_pfn`: the guest's own page table
had no mapping at flush time. The narrow conclusion drawn at the time happened to survive; the
mechanism under it was unrelated, and the per-case correlation built from the same lines came out the
other way round on re-measure.

So before quoting a `reason=`, read its emission site and confirm the value came *from* the check
that refused. If it did not, the fix is to make the callee carry its reason, not to reason harder
about the label. The "one status for N checks" collapse the typed-decline work already ended regrows
anywhere a `-> bool` crosses a module boundary.

**Exclusions decay, and nobody re-tests them.** The anti-pattern list already forbids claiming a
class is fixed from one clean boot. Negative results are exactly as fragile and strictly more
dangerous, because a wrong fix gets found the next time someone looks at the screen while a wrong
exclusion just sits in a table telling future readers not to look there. Before a measurement becomes
a standing exclusion, say which boot it came from and what it would take to overturn it — and when a
later run happens to re-measure it, actually check.

**A once-per-boot probe still attributes per case.** Deduplicated dumps look like they throw
attribution away — one line per pipeline for the whole boot, so how would you know which case ran
it? You know because *first appearance is the signal*: a dedup'd probe fires exactly when something
ran that had never run before, and the always-on line it emits is timestamped. A repro that drives
several cases in one boot therefore gets a free set difference between them, and the reasoning is
one-sided in a useful way — "first seen in the failing case" proves it did not run in any earlier
case, which is the direction you usually want. Do not add a second, per-case probe for this.

Deriving the case boundaries needs the same suspicion as any other measurement, because the repro's
step markers usually go to the console and not into the log. Anchor them on something the run
already recorded — capture mtimes, a known cadence, the last line's timestamp — and then **print a
consistency check the boundaries have to pass**. Here that check is the lead time from each window's
first dump to its own capture: it must come out at the scripted sleep for every window, and one
window disagreeing is a mis-set boundary rather than a finding. Four windows agreeing to 0.3 s is
what makes the set difference trustworthy.

### Interleave The Arms Of A Live A/B

A live before/after on the VM rig compares two arms separated by wall-clock time, and neither the
guest nor the host is constant in time. Snapshot-revert resets the guest disk, **not** the guest
clock, and macOS changes its own rendering with time of day.

The largest known instance: the guest's default desktop picture is a **dynamic** desktop — a
Display-P3 HEIC carrying five variants that macOS selects and crossfades between on a daily
schedule. Its rendered colour is therefore a function of when the boot happened. Measured on one
boot by moving the guest clock and restarting the Dock, whole-screen mean RGB:

| guest clock | mean R / G / B |
|---|---|
| 10:30 | 0.9827 / 0.3178 / 0.2382 |
| 14:00 | 0.9827 / 0.3178 / 0.2382 |
| 19:30 | 0.9780 / 0.1254 / 0.2347 |
| 23:00 | 0.9718 / 0.0991 / 0.2270 |

**So pin the wallpaper to a static image before any colour-sensitive comparison.** A static image
renders identically regardless of the clock. The Dock only re-reads its picture on restart, so a
clock change with no `killall Dock` proves nothing — an earlier probe concluded time of day was not
involved for exactly that reason.

Even with the wallpaper pinned, do not run N boots of the parent and then N boots of the child.
**Alternate them** — parent, child, parent, child — so anything else drifting with time lands on
both arms. A sequential A/B once produced 3-of-3 versus 2-of-2 agreement and was still entirely
confounded: the "regression" survived a full revert of the change that supposedly caused it.
Replicating the treatment says nothing about a confound that moves with time.

Related: a boot whose captures differ from the known-good constant for that sequence must be
**discarded**, not interpreted. A brightness floor does not catch wrong content of the right
brightness.

### Measure Against Known Input, Not Against Another Unknown

Comparing a rendered frame to a *different* rendered frame can show that two states differ. It
cannot say what the transform between them is, and it inherits every confound both frames carry.
Two separate investigations here stalled for exactly that reason: one compared a boot's desktop
against another boot's desktop, the other compared it against a wallpaper file whose on-screen crop,
scale and variant selection were all unknown.

When the question is "is this path faithful", put **known values** through it and read them back.
Displaying a generated patch pattern and measuring the patches settled in one boot what
boot-to-boot capture comparison had not settled in eleven: the present path reproduces the neutral
ramp with zero deviation on both the host window and the guest's own screencapture, so a
wrong-looking desktop is not the present path.

The same discipline bounds a claim. Content selection can only ever produce a *convex combination*
of the variants it selects among, so rendering each variant on its own establishes the range the
output must fall in. A result outside that range is a defect and not a selection, and that argument
holds without knowing which variant was selected.

### Fit The Wrong Output Before Naming A Wrong Mechanism

Once known values have been through the path, you hold measured/nominal pairs. **Fit them to a
closed form before nominating any mechanism.** A transform that reproduces the measurements with no
free parameters tells you the *value* that is wrong, and that converts every later probe from "does
this look off" into "does this read X" — a question a probe can answer wrong-way-round, which is the
only kind worth landing.

It also settles arguments that qualitative reading cannot. A table of destroyed patches here sat in
the notes for two iterations described as "YUV-shaped", which was true and useless. Fitted, it is
full-range **BT.601** luma of the correct image with the chroma pair pinned at (0, 255) — fifteen
numbers, worst error 1.6, zero fitted parameters, and BT.709 refuted because no constant chroma
satisfies its green and blue together. "YUV-shaped" cannot be acted on; "the chroma pair is exactly
(0, 255)" can, because it turns the next probe into a yes/no about a specific value.

Stop the fit there. A closed form tells you *what* the wrong value is and says nothing about where it
came from, and the temptation to append a mechanism to it is strong precisely because the fit is
convincing. The same result above was written up with "and (0, 255) are the format-fill constants of
a sampled image's z,w lanes, which nothing else produces" — a day later, reading the kernel showed it
writes a literal 1.0 into lanes itself, so a 1.0 is not evidence of a fill at all. The fit survived;
the mechanism sentence bolted onto it did not, and it would have aimed the next iteration at the
wrong site.

Two rules make the fit trustworthy:

- **Hold points out of sample.** Fit on one subset, predict the rest. The fit above was built on the
  neutral ramp alone and then predicted five saturated primaries to within 2/255. Without that step
  a six-parameter matrix fitted to six greys proves nothing.
- **Cross your readout grid against every spatial defect before reading its values as a transfer
  function.** A patch readout inherits the geometry of anything else wrong in the frame. Three
  patches in that same table read exactly (0,0,0) and were written up as the frame's most specific
  clue — a *wraparound*, which pointed away from a matrix. They were simply the three patches in the
  rightmost column, sitting inside a separately-noted black band. The refutation was one line of
  arithmetic on `i % COLS` and it was never run.

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
