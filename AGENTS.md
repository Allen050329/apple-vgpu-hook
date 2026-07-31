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

### A live A/B needs one binary per arm

`vm/boot-x86.sh` rebuilds QEMU on every boot, so two boots of "the same" tree are two binaries
unless the tree is committed and untouched between them. Comparing a boot against a result recorded
in an earlier session compares the change to the rig's drift as well, and this project has spent
whole sessions on the difference: five consecutive boots were read as a regression from the change
under test, and a sixth — the same binary with only that change switched off — showed the same
defect. It was three commits older, unbooted, and nobody had measured a control.

Measure the control on the binary you are testing. Where a behaviour cannot be switched off at
runtime, give it a gate first (`REIMS_VGPU_SAMPLED_RESIDENT_GATE_OFF`, `REIMS_VGPU_STORE_DEFER_OFF`,
`REIMS_VGPU_CONTENT_REUSE_OFF` are all this pattern) — that boot is cheaper than the session lost to
reading a stale baseline. To attribute a defect to a *commit*, build each arm from its own source
(`git checkout <ref> -- crates/ vendor/qemu && git submodule update --init vendor/qemu`) and run the
same harness on each.

### The rig is untracked, so its bugs are invisible to review

`.agents/` and `kb/` are gitignored. Nothing in them appears in a diff and `git add -A` silently
skips them, so a commit body that describes a repro fix describes something a later reader cannot
find. Put durable rig rules here instead, and treat a harness like product code: it needs its own
controls, because a broken harness fails in the direction that looks like success.

Worked example, found by reading a null result twice. `crash-hunt.sh` snapshotted the guest's crash
reports with `ls -1` over two directories, which also prints their headers and any *subdirectory* —
and `/Library/Logs/DiagnosticReports` contains one called `Retired`. Every run therefore reported
exactly one "new crash report" named `Retired` whether or not anything had crashed, so both arms of
an A/B scored one hit when the true count in both was zero. **A harness whose null result looks like
a hit cannot score an arm.** Before believing a repro, confirm you have seen it print the negative.

The mirror of that failure is the more dangerous one, because it reads as a pass. Every census A/B
here scores a *count of bad events* scraped from `/tmp/reims-vgpu-fail.log` — windows that outlived
a fence, declines, lost flushes. A boot that produced no GPU work at all, or one whose fail log was
unlinked out from under the device's append fd, scores **zero of everything**, which is exactly what
a perfect arm scores. So a count-based arm needs a validity gate that is independent of the count:
confirm the log carries a `store_routes` census *and* that the census reports real store traffic,
and report anything else as `UNMEASURED` rather than folding it in as clean.
`.agents/repros/fence-census-ab.sh` is the worked example.

Name the boot's serial log by the path `vm/boot-x86.sh` prints (`serial → …` on its first line), never
by taking the newest match in `vm/disks/run/`. Picking by mtime attributes whichever boot wrote last
— possibly another arm's, or a concurrent run's — to the boot being scored. Measured instance: in the
six-round icon A/B of 2026-07-31 the round-2 *arm* boot guest-kernel-panicked
(`serial-20260731-144329.log`, `IOAccelSegmentResourceList::prepare` with `RAX = 0xffffffffffffffff`)
and the harness recorded it as `NO_ROUNDS` — indistinguishable in the verdict table from a boot that
simply scored nothing. **A panicked arm is unmeasured, and it is also a result; a harness that cannot
tell those apart loses both.**

Bracket a character in every `pgrep -f` / `pkill -f` pattern, not just the ones that kill. The pattern
you search for is itself in the command line of the shell doing the searching, so `pgrep -f
icon-boot-ab` matches that shell. Both failure modes bite, and neither announces itself: a wait loop
written `until ! pgrep -f icon-boot-ab; do sleep 10; done` never exits, because it always finds
itself, and a run that had actually finished 20 minutes earlier still looks live; and `pkill -f
qemu-system-x86_64` kills the shell that ran it, so the sweep reports failure while the thing it
meant to kill was never running. The `sweep()` in the repro scripts already writes
`[q]emu-system-x86_64 -enable-kvm` for exactly this reason — the convention is right, it just has to
reach the *waiting* code too, which is where a false positive costs an hour instead of a signal.

### A scorer written from one report is blind to the rest of the class

`scrollpatch.py` — the Goal 3 gate — flagged a pixel only when it was *bright and unsaturated*, on
the stated grounds that "the observed defect paints white". That is a rule about one user report, not
about the bug class, and it made the instrument blind to the two likelier shapes of the same loss: a
tile whose content never landed is **transparent**, so the capture shows the page background
(`#101040`, dark); and a freshly allocated surface that was never written is **zero-filled**, so it
shows black. Both are the defect. Neither is white.

The harness ran three times and scored `none` three times. Measured by injecting each fill into a
synthetic control and re-scoring: the old rule called **three of four injected patches CLEAN** —
background, black, and a missing cell layer — and caught only literal white. So those three null
results were never evidence about the device.

Scoring is now against the page's *palette*: a pixel is a defect when it is not one of the colours
the page is made of, whatever it is instead. The palette is **measured from the control capture**,
not read from the generator's CSS, because Safari renders through the display colour profile and
`#c02020` in the stylesheet is not `(0xc0,0x20,0x20)` in the framebuffer — comparing against the
declared value needs a tolerance wide enough to swallow real defects.

That moves the whole verdict onto the control, so the control needs its own gates, and they are the
usual asymmetry: the page must be reachable in few flat colours and those must cover ≥ 98 % of the
non-chrome control (a photograph or a desktop capture fails this), **and** neither the page
background nor the band fill may appear in it. The cells cover the viewport, so a control showing
either is *itself* patched — and admitting that colour to the palette would score every later capture
carrying the same patch as CLEAN. That is the one failure direction that reads as a pass, so it
aborts.

An abort prints no `VERDICT=` line, and `case "$line" in *VERDICT=PATCHED*)` reads an empty line as
not-patched. `shoot()` therefore synthesises `VERDICT=UNSCOREABLE` and the report lists those
separately: **an unscoreable capture is neither clean nor patched**, and any of them voids the
`none` on the line above it.

### Drive the guest with arrow keys, never the mouse wheel

The wheel wedges, and it wedges silently. Measured on the x86 rig against a page 42 bands (~17 000 px)
tall, with the top verified by the generator's colour swatch being on screen:

```text
wheel up 40   moved_frac 0.98   0.39   0.0000  0.0000  0.0000  0.0000  0.0000  0.0000
key down x20  moved_frac 0.61   0.83   0.77    0.81    0.97    0.99
```

After roughly 80 synthetic ticks the page stops at a position **byte-identical to what cmd-Down
reaches**, so the obvious reading — "it hit the bottom" — is what the evidence looks like and is
wrong. Ruled out separately: a modal dialog over the page (reproduced with the screen clear), page
zoom (reproduced after cmd-0), and the document being short (42 `class=band` divs in the file the
guest received).

The wheel carries a second trap on top of that one. macOS ships **natural scrolling**, which inverts
it: `wheel down` asks the content to move *up*, which at the top of a document is a no-op. Three runs
of `scroll-patch.sh` sent 480 `wheel down` ticks into the top of the page and captured the same frame
twelve times.

Arrow keys have neither problem and are what the guest should be driven with. Note that `cmd-up` is
not a QMP key name — `send-key` answers `Parameter 'data' does not accept value 'cmd-up'` — the
spelling is `meta_l+up`, and a calibration that used the wrong one failed into its fallback silently.

### The harness never scrolled, and three `none` results say nothing

Worse than the blind scorer, found by re-scoring the archived captures of those same three runs. The
premise of `scroll-patch.sh` is that it visits twelve scroll offsets twice. It did not.

Every `down-*.png` in a run carries a colour histogram **identical to the control's, count for
count**, and consecutive down captures differ in `moved_frac = 0.0000` of sampled pixels. The down
pass never moved. The up pass sat at one single other offset for all twelve of its captures. The
files have distinct md5s — a clock ticks in the corner — which is why nothing noticed.

**A harness that cannot reach the state cannot find the defect, and it reports that as a clean
page.** `shoot()` now takes `--prev` and refuses any capture whose frame did not change, as
`VERDICT=UNSCOREABLE`. The two populations are nowhere near the 0.10 threshold — an unmoved frame
measures 0.0000 and a real screenful measures 0.9941 — so the number is not fitted between them.

The first capture of the up pass is exempt and must be: that pass shoots before it scrolls, so it is
deliberately the same offset as the last down capture.

A third defect in the same instrument, from the same re-scoring. The palette was measured from a
control that is **one screenful**, so it could not contain a hue that first appears six screenfuls
down, and every scrolled capture showing a later band scored 0.92 off-palette — all of it correct
page. The generator now emits a swatch of every colour it can ever use at the top of the page, so
the control contains the whole palette by construction. Keeping it as page content rather than as a
list in the scorer is what stops the two drifting apart when a hue is added.

The same three-way split applies inside the scorer. `PATCHED` needs a *blob* of at least
`--min-blob`; scattered off-palette pixels over the floor with no blob are `NOISY`. Collapsing those
into `PATCHED`, which the scorer did at first, scores noise as a finding — injecting mild Gaussian
noise into an otherwise clean capture produced `bad_frac=0.048 blobs=0` and a `PATCHED` verdict.
Collapsing them into `CLEAN` instead would score an untrustworthy capture as a pass. Neither is
available: a wash and a rectangle are different claims about the device, so they get different words.

Two counters in that census were themselves blind until `2327a79`. `*_stamp_outlived` compares a
window's `armed_stamp_seq` against `DeviceState::completion_stamp_seq`, and only `write_stamp`
advanced that counter — the root completion stamp, written inline by `drain_main_fifo`, did not. A
window that sat through hundreds of root completions therefore scored `stamp_same`. **A measurement
that shares a code path with the thing it measures will agree with it.** When a counter says a rail
is punctual, check what advances the clock before believing the rail.

### A boot the firmware aborted is not a device wedge, and it looks exactly like one

From outside, a `boot.efi` abort and a hung device are the same picture: QEMU alive, the guest
answering nothing, **zero guest GPU commands**, and a fail log of nothing but
`host_window_cadence`. One session read that as a regression from the change under test and spent 53
minutes on it. The serial log settles it in one line:

```text
AAPL: #[EB.MM.AKMR|!] Err(0xE) <- EB.M.BAPr2 2 2 50271 0x700000
AAPL: #[EB.B.MN|!]    Err(0xE) <- EB.MM.AKMR
AAPL: #[EB|STOP] 0x15
OC: Boot failed - Aborted
```

`boot.efi` draws a KASLR slide and asks for its 50271-page (~196 MiB) kernel region at
`0x100000 + slide * 0x200000` with `AllocatePages(AllocateAddress, …)`. This firmware's map has ACPI
NVS at `0x800000` and ~14 MiB of BootServicesData behind it, so every base below `0x1780000`
collides — slides 0–11 of 256, **4.69 %**. Measured over the 513 serial logs then on disk: 24
aborts, **4.68 %**, and their failing bases were exactly `0x100000, 0x300000 … 0x1700000`.

Fixed at the source: OpenCore's `Booter → Quirks → ProvideCustomSlide` was `false` in the snapshot's
`config.plist`, so nothing filtered the slide list. With it `true`, OpenCore computes the valid
slides from the memory map and passes one (`slide=247` on the first boot after; the kernel reports
`KASLR slide: 0x1ee00000 dynamic`). The change lives in a new immutable snapshot — snapshots are
never edited in place — made by converting `OpenCore.qcow2` to raw, `mcopy`-ing the edited plist into
the ESP at partition offset `0x100000`, converting back, and reflinking `macos.img`/`OVMF_VARS.fd`
from the previous snapshot.

After: **30 boots, 0 aborts**. Thirty boots at the old rate would be expected to show 1.4, so the
count alone is only suggestive (p ≈ 0.24 under the old rate); what carries it is the mechanism and
the arithmetic — 12 bad slides of 256 predicts 4.69 %, and 24 of 513 measured 4.68 %.

`vm/boot-x86.sh` now watches the serial log for `#[EB|STOP]` / `Boot failed - Aborted` and exits
**125** within ~30 s instead of sitting out `TESTING_TIMEOUT`. Treat 125 as "retry the boot"; no
measurement from such a boot is about this device. `.agents/repros/boot-abort-rate.sh` turns it into
a rate.

### Every guest kernel panic this project ever caused is on disk, in `vm/disks/run/`

`vm/disks/run/serial-*.log` is never cleaned. 547 of them were swept and **11 carried a guest
kernel panic (2.0 % of boots)**. Nothing had ever reported one: `-action reboot=shutdown` turns the
panic reboot into a QEMU exit, so the boot script said "qemu exited", the drive script said "VM IS
GONE", and the word panic appeared nowhere.

The reasons are a corruption census, not a grab bag:

```text
[kalloc.type.var6.6144]: element modified after free (off:0, val:0xffffffffffffffff, sz:6144)
[kalloc.type.var3.256]:  element modified after free (off:0, val:0xffffffffffffffff, sz:256)
pmap_page_protect() pmap=… pn=… vaddr=…                                              (x2)
Kernel trap at 0xffffffffffffffff   <- indirect call through a clobbered ifnet fn ptr
"hitting assertion" @AppleParavirtPageTable.cpp:200  <- in WindowServer's own submit
Kernel trap at <various>                                                             (x4)
```

The two `element modified after free` reports are the strongest evidence this project has for the
write-after-fence class. The kernel's poison check found a **whole freed element filled with 0xFF
from offset 0** — 256 bytes in one, 6144 in the other. That is not a stray pointer; it is a bulk
write of opaque white pixels into memory the guest kernel had already freed. Both predate
`311cb11`, which is the repair for exactly that.

`vm/boot-x86.sh` now detects a panic in the serial log, copies it to `/tmp/reims-vgpu-panics/`, and
exits **126** — distinct from 125 (firmware abort) and 124 (wedge). Sweep the historical logs with:

```sh
for f in vm/disks/run/serial-*.log; do grep -l 'Debugger called: <panic>' "$f"; done
```

#### The panicking process is the finding, and it is almost never WindowServer

Re-swept at 552 boots: **12 panics, 2.2 %**. Print the panicking task with each, because the spread
is the diagnosis:

| panicking task | site |
|---|---|
| WindowServer | `IOAccelSegmentResourceList::prepare` ← `AppleParavirtSegmentResourceList::prepare` ← `IOAccelCommandQueue::processSegment`, page fault at CR2 `0x16f` |
| WindowServer | `IOSurfaceClient::~IOSurfaceClient` ← `IOSurfaceRootUserClient::release_surface` |
| WindowServer | `"hitting assertion" @AppleParavirtPageTable.cpp:200` |
| `airportd` | `Kernel trap at 0xffffffffffffffff` — an indirect call through a pointer overwritten with 0xFF |
| `tccd` | apfs `obj_get` ← `btree_node_get_internal` |
| `followupd`, `com.apple.AppleUserHIDDrivers` | `kalloc` poison: element modified after free, `val:0xffffffffffffffff` |
| `Safari`, `ReportCrash` (×2) | `pmap_page_protect`, non-sleepable RW lock |

A bug in one device path panics in that path. **These land in an apfs btree node, a network
interface's function pointer, a HID driver's heap element and a security daemon** — subsystems this
device never touches. That is not a logic error with a backtrace worth reading; it is a device
writing where it no longer holds title, hitting whatever the guest allocator happened to put there.
Read the *distribution*, not the individual trace.

The `0xFF` recurs across all of them and is the same write seen from different victims: opaque white
BGRA pixels. It is almost certainly a legitimate white frame landing at the wrong address — the
defect is *where*, not *what*, so do not go looking for a source of white.

Panics still fire on binaries carrying the fence repairs (three on 2026-07-31 alone, one of them
14:43 on a tree with all three raw-address rails bound). A fence binding orders a write against the
guest's completion stamp; it does not make the destination address correct. The address is
`mapper::type4_pages_still_ours`'s question.

At 2.2 % an A/B needs ~150 boots per arm and is not affordable. Let panics accumulate as a side
effect of every boot and re-sweep the census instead.

### A panicked boot must not be scored, and "no rounds" does not say it was not

`.agents/repros/icon-boot-ab.sh` retries a boot that panics *before ssh* (exit 126) and does not
score it. A boot that panics **during the drive** reaches the scoring branch instead, and that
branch decides between `PANIC` and `NO_ROUNDS` by taking the newest `vm/disks/run/serial-*.log` and
grepping it. On 2026-07-31 that arm scored `NO_ROUNDS` for a boot whose serial log plainly carried
`Debugger called: <panic>` — the mtime ordering did not name the boot that had just run.

Do not identify a boot's evidence by mtime. `vm/boot-x86.sh` names the serial log with the run
stamp and prints the path; read the path out of the boot log. An arm that panicked is **unmeasured**
— not clean, not corrupt, and not a row to average — and a harness that cannot tell those apart is
the failure mode the `Retired` directory already cost this project a session to.

### A `store_routes` counter is a per-interval count, and `grep -c` returns the census cadence

Every counter in a `store_routes` line is the count **for that census interval**, not a running
total: the boot's figure is the sum of the key across every census line. Counting those lines with
`grep -c mapping_pages_ours` therefore measures *how often the census fired*, not the quantity — and
it returns a plausible number, which is why it survives review.

An earlier revision of this section called the counters running totals while still prescribing the
sum, which is the remedy for the opposite convention. Taking the last line instead — the reading a
running total would demand — under-reports by the census cadence, about **100×** on a 2 900 s boot.
The values settle it directly: `rendw_stamp_outlived` on one control boot reads
`10 3 4 9 2 23 32 53 107 95 99 89 …` and ends at `83`. A running total cannot decrease.

Measured: the first run of `.agents/repros/mapping-guard-census.sh` printed `mapping_pages_ours 310`
and `mapping_pages_drifted 20` for a boot whose true totals were **25 646 and 22**. Both numbers were
wrong, both looked like results, and the ratio between them was wrong by two orders of magnitude —
6.1 % refusal instead of 0.086 %. A conclusion drawn from that pair would have been backwards about
whether the guard was hot or rare.

Sum the key across census lines (`route_sum` in the repro scripts). Use `grep -c` only for **event**
lines — one line per occurrence, like `mapping_page_drift ` or `deferred_flush_lost` — and note that
`mapping_page_drift` without the trailing space also matches `reason=mapping_page_drift` on the
`deferred_flush_lost` line that follows it, so the unanchored count double-counts. Print which
convention each number uses (`(sum)` / `(lines)`) so the next reader does not have to re-derive it.

### Measured: the fence bindings close the Finder icon class

Two experiments, both one-binary-apart from `fbf7bd9` with nothing in `crates/` touched between
arms, `icon-composite.sh` at 4 rounds per boot. The second was **interleaved** — arm and control
alternating, and which one leads flipping each pair — to remove the first's blocked-in-time
confound.

```text
                                       boots            rounds
first A/B   arm  (default)          0 / 8            0 / 32
   (blocked) ctl  (FENCE_FLUSH_OFF)  4 / 8            5 / 32     p = 0.077 / 0.053
replicate   arm                     0 / 6            0 / 24
(interleaved) ctl                   2 / 6            2 / 24     p = 0.455 / 0.489
────────────────────────────────────────────────────────────────────────────────
combined    arm                     0 / 14           0 / 56
            ctl                     6 / 14           7 / 56     p = 0.016 / 0.013
```

Read it as three statements, not one.

**The replicate alone does not reach significance.** 2 of 6 against 0 of 6 is p = 0.46; on its own
it is not evidence. What it does do is reproduce the *direction* under a design that cannot be
explained by the hour the arm ran in, which is the specific objection the first experiment could not
answer.

**The first experiment's effect size was inflated.** 4 of 8 control boots corrupt did not survive:
the interleaved rate is 2 of 6. Pooled, the control corrupts about 43 % of boots, and the pooled
figure is the one to quote — a later run predicted from 50 % will look like a regression when it
lands at 30 %.

**The arm has not corrupted once in 14 boots and 56 rounds.** That is the practically load-bearing
fact and it does not depend on the p-value: the shipped configuration shows no icon corruption on
this rig, and the only change that brings it back is turning the fence bindings off.

The control's validity gate is independent of its icon score and it passed hard. Stated properly —
summed per boot, per rail, over the four interleaved pairs of the replicate:

```text
                  gvaw          linw          rendw            storw
arm  p1..p4    0 / 3 823     0 /    9      0 / 21 300       0 / 43
               0 / 5 547     0 /    9      0 / 22 266       0 / 43
               0 / 4 987     0 /  104      0 / 23 661       0 / 91
               0 / 5 558     0 /   84      0 / 20 760       0 / 91
ctl  p1..p4    5 / 5         6 /    6  9 236 / 9 236       22 / 22
               3 / 3         6 /    6  7 454 / 7 454       22 / 22
               1 / 1        47 /   47  9 869 / 9 869       25 / 25
               2 / 2        36 /   36  8 918 / 8 918       25 / 25
```

That is a completeness statement about **Goal 1**, not just a knob check, and it is stronger than
"the knob took". On the default binary **every one of ~88 000 render-window stores landed inside its
fence, on all four rails, in every boot**; with `REIMS_VGPU_FENCE_FLUSH_OFF=all` the `same` column is
**0 everywhere** — not one store lands inside its fence. The bindings are not a partial improvement
on some rails; they are the entire difference between "always" and "never", and no rail is missing
one.

(The earlier figure here, "1 000–1 700 per round", was the census cadence times the interval count —
the arithmetic the section above this one now warns about. The per-boot totals are the row above.)

Two cautions that belong with the number:

- **Nothing in the census sees this class.** 71 counters, compared per round across 5 corrupt and 27
  clean rounds on identical binaries: **not one separates them**, and the knob's own counter shows no
  dose-response (corrupt rounds average *fewer* outlived windows, 125.6 vs 131.1 per 1000 draws; boot
  6 corrupted with 20x fewer outlived windows than any other boot). So do not go looking for the
  mechanism in `store_routes` — it is not in there. The fence changes *when* a write lands, and
  nothing counts lateness per surface.

**Killed by the replicate: the round-position effect.** The first experiment's 5 corrupt rounds were
all round 2 or 3 of 4, never 1 and never 4 — p ≈ 0.03 under uniform, and it was recorded here so a
bigger sample could confirm or kill it. The replicate's two corrupt rounds were round **2** and round
**4**, so "never round 4" is gone on the first new observation that could have contradicted it. This
is what a p ≈ 0.03 pattern found by looking at five points, after the fact, in a table of many
possible patterns, is worth. Leave the note as a worked example rather than deleting it.

**What this does not say.** It does not identify *which* fence commit closed the class.
`REIMS_VGPU_FENCE_FLUSH_OFF=all` reverts the whole group. The prime suspect is `2327a79` (the root
completion stamp was a fence nothing was bound to), because it landed at 15:26 *during* the
2026-07-31 `icon-ab2` run that scored 3 corrupt of 11 — which is also why that run's 27 % is not a
usable baseline: `vm/boot-x86.sh` rebuilds from the working tree, so its rounds were built from
several different binaries as the session edited `crates/`. To attribute it, build each arm from its
own source (`git checkout 7763f2f -- crates/ vendor/qemu`, the commit before `2327a79`) and run the
same harness.

### `grep -c` prints a count and exits 1, so `|| echo 0` emits two lines

`n=$(grep -c foo file || echo 0)` is the natural way to write "count, defaulting to zero", and it is
wrong: on zero matches `grep -c` **prints `0` and exits 1**, so the `||` fires and `n` becomes the
two-line string `0\n0`. Any later `$(( ))` on it dies with `arithmetic syntax error`, and any
`printf` of it silently emits a stray line.

It has bitten twice here in one session. First as a cosmetic stray `0` under `clobber` and
`identity_split` in the census output, which was read past. Then, unfixed, it killed an 8-boot icon
run at the first scoring step — the loop wrote no rows at all and still printed its completion
marker, so the output was an empty table that looked like "no boots corrupted".

`grep -c` always prints a number when the file exists. Write `n=$(grep -c foo file 2>/dev/null);
n=${n:-0}` and keep the default for the *missing file* case only.

And check that the file exists, separately, before believing the count — because the missing-file case
is the one that reads as a pass. Hit while writing this session's boot-validity gate: the serial log
was named by a relative path from the wrong directory, `grep` found nothing because there was nothing
to find, and the gate printed `panics=0 aborts=0`, which is exactly what a healthy boot prints. The
`2>/dev/null` that keeps the output tidy is also what hides the `No such file` that would have said
so. A gate over a path that might not exist must fail on the *path*, not on the count.

### A test double more generous than the host cannot fail the way production does

`FakeHost` armed a `track_guest_writes` set at generation 1 the instant it was tracked, and returned
`Some(0)` for an unarmed one. The product shim does neither: `reims_vgpu_dirty_gen` holds a new
set at 0 for a deliberate two-harvest window, and `qemu::host_ops` maps that 0 to `None` because 0 is
also "unknown token".

So every test saw a readable baseline that the real host would never give, and a rail that could not
work in production passed its whole suite. It shipped that way, and its dead counter
(`gvac_gw_clean` = 0 of 201 331) was then written up in a doc comment as a discovery about the guest.

When a fixture stands in for a host contract, model the contract's *refusals* — startup windows,
sentinel values, rate limits — not just its happy answer. `guest_write_startup_window` and
`finish_guest_write_arming` are that, kept opt-in so turning them on is a deliberate statement by the
rail under test rather than a silent change to a hundred unrelated assertions.

### The corruption lands in the page the surface left, not the one it moved to

Worth knowing before writing any repro for the write-after-free class, because it decides which
address the assertion reads.

When the guest silently re-points a type-4 surface, the device keeps writing to the **old** physical
pages: `ensure_contig_view` caches a `mach_vm_remap` of the PFNs it walked and returns it again on
every later call, so the stale view resolves to where the surface *was*. Those are exactly the pages
the guest freed and handed to something else — which is why the crash census lands in an apfs btree
node, an ifnet function pointer and a malloc small-zone free list rather than anywhere near a
surface.

A first draft of `a_repointed_surface_refuses_the_write_and_leaves_the_new_owner_alone` asserted the
**new** backing page was untouched. It passed against a deliberately unguarded build, because
nothing ever writes there. A test for this class must seed the abandoned page with a new owner's
bytes and assert those survive.

### Never delete the live fail log

The device holds an append fd on `/tmp/reims-vgpu-fail.log` from a background writer thread. `rm`
unlinks the name and every later line goes to an inode nothing can open, so the boot produces empty
logs and no census — discovered after a 30-minute 14-round run. Move it instead, before the boot:

```sh
mv /tmp/reims-vgpu-fail.log /tmp/<arm>/fail-prev.log
```

### Vulkan validation layers without root

Sync validation (`SYNC-HAZARD-*`) names a read/write hazard at the command that causes
it. Five such hazards were found and fixed by hand on this rail before anyone tried it;
every one would have been a single log line. The device creates its instance with no
layers, so the loader's environment is the whole mechanism — and the layer package does
not have to be installed system-wide, which is what previously blocked this behind a
`sudo` password:

```sh
mkdir -p /tmp/vklayers && cd /tmp/vklayers
curl -sO https://geo.mirror.pkgbuild.com/extra/os/x86_64/vulkan-validation-layers-<ver>-x86_64.pkg.tar.zst
tar --use-compress-program=unzstd -xf vulkan-validation-layers-*.pkg.tar.zst
```

The layer manifest names its library relatively, so the extracted `usr/lib` has to be on
the loader's search path:

```sh
export VK_LAYER_PATH=/tmp/vklayers/usr/share/vulkan/explicit_layer.d
export LD_LIBRARY_PATH=/tmp/vklayers/usr/lib
export VK_LOADER_LAYERS_ENABLE='*validation*'
export VK_LAYER_SETTINGS_PATH=/tmp/vklayers/vk_layer_settings.txt
```

`vk_layer_settings.txt` is where sync validation is turned on and where the output goes
somewhere a repro can slice:

```text
khronos_validation.validate_sync = true
khronos_validation.report_flags = error,warn
khronos_validation.log_filename = /tmp/reims-vgpu-vkvalidation.log
khronos_validation.duplicate_message_limit = 3
```

Confirm the layer actually loaded before believing a clean run — `vulkaninfo --summary`
must list `VK_LAYER_KHRONOS_validation` under `Instance Layers`. A boot with sync
validation on is not a frame-rate or timing measurement.

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
