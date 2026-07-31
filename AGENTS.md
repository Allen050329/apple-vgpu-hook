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
lines — one line per occurrence, like `mapping_page_drift` or `deferred_flush_lost`. Print which
convention each number uses (`(sum)` / `(lines)`) so the next reader does not have to re-derive it.

**Anchor event counts at the line start, not with a trailing space.** An earlier revision of this
section said `mapping_page_drift` "without the trailing space" also matches `reason=mapping_page_drift`
on the `deferred_flush_lost` line, and prescribed the trailing space as the fix. That remedy does not
work, and it was still wrong when a scorer written *from this paragraph* used it: the lost line ends
`… reason=mapping_page_drift t=27804`, so a trailing space matches it too. Every drift is followed by
its own lost line, so the count comes back **exactly double** and looks entirely plausible. Measured
on the pre-guard boot: `grep -c 'mapping_page_drift '` returns 18 where `grep -c '^mapping_page_drift '`
returns 9. Use `^`.

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

#### Landing inside the fence does not empty the clobber window, and the code says it does

`storage_flush.rs` states, at `render_flush_guest_written_ranges`, that once
`flush_mapping_windows_before_fence` binds every armed window "the interval in which a guest store
can be both after the Store and before the writeback does not exist", so that "a
`render_flush_over_guest_write` after the binding names a window that landed outside the fence
anyway, which is a defect and not a cost."

One boot of the four falsifies the second half of that, and the other three are the reason this
section is worth reading. All four are 600 s driven x86/Vulkan boots from the same A/B, same
workload, counters summed per the per-interval rule:

```text
                    surface_flush   rendw_stamp_outlived   render_flush_over_guest_write
arm-1                      35 115                      0                             679
arm-2                      41 014                      0                               0
ctl-1                      41 105                      0                               0
ctl-2                      37 551                      0                               0
```

**`rendw_stamp_outlived` is 0 in every boot and `render_flush_over_guest_write` is 679 in one of
them, so the two cannot be the same event.** That is the whole refutation and it does not need a
rate. The joining inference is what fails: the guest does not need to be told the render is finished
before it may write to an IOSurface it owns, so CoreGraphics blits and inter-buffer damage
forward-copies land in the Store→fence interval whether or not the window is punctual. The interval
is short, not absent. Read the doc comment as intent, not as an invariant the census confirms.

**Do not quote 679 as a rate.** An earlier revision of this section did — it called the clobber "1.9 %
of windows" from that single boot, and three further boots on the same binary and workload scored
**zero**. The instrument was not asleep in them: `t11_gw_armed` is 69 532–70 779 across all four and
`t11rung_resident_gw_clean` 124 910–143 569, so the witness was armed and answering in every boot and
answered `Wrote` in only one.

Nor is it a burst. Binned by 30 s, arm-1's clobbers run 29–54 per bin continuously from t ≈ 110 s to
the end of the boot, so that boot entered a state at two minutes in and stayed there while three
others never entered it at all. 541 of the 679 are on five 1920×1080 compositor mappings, which is
what a guest doing sustained CPU compositing into the surfaces we also render into would look like.
The most likely reading is a **workload difference between sessions**, not a device state — but
nobody has varied it deliberately, so that is a hypothesis.

Two reasons the instrument itself is not the suspect. The verdict comes from
`host.guest_write_gen(token)` — QEMU's own dirty generation for the tracked page set — so it does not
share a code path with the fence machinery it is being compared against, which is the failure mode
this document warns about elsewhere. And it is a three-way verdict: `NoStamp` and `Unreadable` are
counted separately, so `Wrote` requires a real baseline generation and a real differing one.

Harm is **not** established. The concentration on surfaces redrawn every frame is exactly where a
replaced guest write self-heals; the interesting tail (1240×702 Safari content, 15×622 scrollbars) is
small, and nobody has tied a single clobber to a visible artefact. What the four-boot spread does say
is that any A/B scoring this counter needs **many boots per arm**: an arm of three boots could score
0 and mean nothing.

Do not "fix" it by preserving the guest's pages without a control. The same function records a
bisect where a preserving variant scored **0 of 14 rounds clean with the screen black at 19 Hz**,
against 2–3 of 4 for the non-preserving arms. The machinery to do it properly
(`write_bgra8_skipping`, `HostOps::guest_written_pages`) is still present and unused on this rail;
`render_flush_guest_written_ranges` computes nothing and returns an empty `Vec`, so today it is
purely a detector.

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

### When the page walk fails the device guesses GPA == GVA, and that guess is 2.8 % of pages

`apply_type4_backing` translates each of a surface's backing pages through the owning task's page
tables. When `translate_task_gva` returns `None` it does **not** refuse — `objects.rs:780-789` uses
the guest *virtual* address as if it were a guest *physical* address, gated only on
`host.read_gpa(candidate, …).is_ok()`:

```rust
None => {
    let candidate = gva;
    let mut probe = [0u8; 1];
    if host.read_gpa(candidate, &mut probe).is_ok() { id_hits += 1; Some(candidate) } else { None }
}
```

That probe asks "is this address backed by RAM", not "is this address this surface's". Essentially
all of low guest RAM answers yes, so the gate admits nearly every failed walk. The fabricated PFN
then lands in `m.page_entries`, which is the address list **every** later reader and writer resolves
through — `mapping_write`, `blit_exec`, the deferred flush, and `ensure_contig_view`'s `map_pages`
DMA window.

Measured on one ~42 min Safari session (boot `20260731-181529`), summing the `type4 pages … gva_hits=
… id_hits=…` line:

```text
attaches reporting              11 561
pages resolved by a real walk    1 174 789
pages resolved by the guess         34 407   (2.845 % of pages, ~134 MiB of write targets)
attaches with >= 1 guessed page        106   (0.92 %)
  of those, guessed EVERY page         105
```

Read the last row first: this is not scattered pages that happened not to be faulted in. **105 of
106 failures are whole-surface**, on `task=0` — the same task id that resolves 11 455 attaches
cleanly. So the walk fails for a surface entirely, for a bounded stretch of time, and the device
fills the whole page list with virtual addresses.

The guess does not stay contained, and the drift guard is what exposes it. Of the 9
`mapping_page_drift` events that boot, **7 had `cached` numerically equal to `gva`** — they were
guesses, not translations:

```text
mid= 59 gva=0xbb07000 cached=0xbb07000 live=0x301fc1000   <- identity
mid= 16 gva=0xa69e000 cached=0xa69e000 live=0x4347fd000   <- identity
mid= 34 gva=0x59d4000 cached=0x281b77000 live=0x267995000 <- a real translation that moved
```

So the log line those events print — `reason=translation_moved (the guest re-pointed this surface
and no packet said so)` — **is wrong for most of them.** The guest re-pointed nothing. The device
never had a translation; it had a guess, and the live walk later produced the real answer. That the
live column holds a plausible high address is the proof the pages *are* walkable — just not at the
instant the device gave up and guessed.

Two consequences, and keep them separate because only the first is measured here.

**Measured — it loses guest work (Goal 3).** A guessed entry is contradicted on the next drift check,
`mapping_pages_verdict` returns `Drifted`, `flush_render_one` refuses, and the pixels are dropped as
`deferred_flush_lost … reason=mapping_page_drift`. Six of the nine losses that boot were at
`1225x512` — a WebKit tile strip, not a display-sized surface. A compositor surface that is redrawn
every frame survives a dropped writeback; **a WebKit tile is painted once and never repainted**, so
the drop is permanent and pinned to that scroll offset. That is the Goal 3 symptom exactly.

**Hypothesis, not yet demonstrated — it is a candidate for the crash class (Goal 1).** The guessed
GPAs are low RAM (0x8101000, 0xa69e000, 0xbb07000 — 135-196 MiB), which is where the guest kernel's
own allocations live, and the payload is opaque white BGRA. The panic census in this document lands
in an apfs btree node, an ifnet function pointer and a HID heap element with `val:0xffffffffffffffff`
— unrelated subsystems, filled with 0xFF. That is what writing pixels to a virtual-address-as-
physical would produce. **This is a coincidence of shape, not an attribution**: nobody has shown a
specific panic arising from a specific guessed entry, and the drift guard catches guesses on the
paths that consult it. Do not write it up as the cause until an arm measures it.

Note the census undercounts. The `type4 pages` line is emitted only when `page_entries` is empty
(`first_attach`, `objects.rs:820-825`), so an attach over a still-populated list guesses silently.
34 407 is a **lower bound**.

#### Measured: the guess was standing in for an answer 20 ms away

The guard landed in `d455c3e`; `REIMS_VGPU_TYPE4_IDENTITY_GUARD_OFF=1` restores the substitution.
The per-surface evidence is far stronger than the aggregate, so read it first. On boot
`20260731-192622` (Safari, `scroll-patch.sh`, 932 attaches) the guard refused twice, and **both
surfaces resolved completely, on the same task, one or two frames later**:

```text
sid=210  refused t=53237  ->  type4 pages task=0 n=34 gva_hits=34 id_hits=0  t=53257   (+20 ms)
sid=201  refused t=55081  ->  type4 pages task=0 n=34 gva_hits=34 id_hits=0  t=55092   (+11 ms)
```

That is the mechanism, and it is simpler than the one predicted. The guest had not finished mapping
the surface's backing when the device first asked. Every caller is per-frame, so refusing costs a
frame and returns the real translation; the old path instead cached a fabricated address **for the
life of the surface**. So the defect was never "the walk cannot succeed" — it was "the device
answered before the guest was ready, and kept the answer".

**The predicted mechanism was not observed.** The reasoning that a guess ends the task search in
`resolve_type4_surface_ex` (task 0 is probed first, and returning `true` stops the loop) is a real
property of the code and has a unit test, but the rig does not exercise it: **all 980 successful
attaches on the guarded boot resolved on `task=0`**, exactly as all 11 561 did before. Do not repeat
the "the owning task was never tried" story as a rig finding — on this workload task 0 *is* the
owner, and it is merely late.

**The aggregate does not yet carry a claim.** Zero drift and zero lost flushes on the guarded boot
looks decisive and is not: at the pre-guard rate (9 losses / 11 561 attaches) a 932-attach boot
expects **0.73**, so observing 0 is p ≈ 0.48. It is consistent with the fix and equally consistent
with a short boot. The Goal 3 scorer likewise returned 24/24 CLEAN — the same result the page gave
*before* the change, so it discriminates nothing here and the page needs the harder provocation
already noted above.

##### 20 ms was two points on one page; on Wikipedia it is 20 seconds, or never

The heading above is kept because the observation under it is real, but **do not generalise it**. It
rests on two refusals on the flat-colour page, and the `gva0` field added in `e1632f6` makes the
question answerable properly — a refusal can now be matched to a later resolve by the **backing
address** it names rather than by a surface id, which recycles across geometries within a boot and
cannot identify a surface on its own.

Re-asked that way on a control boot whose predrive browses Wikipedia, for each fabricating attach,
the delay until the *same backing* next resolved with `id_hits=0`:

```text
+247 ms   +1 054 ms   +20 161 ms   +20 320 ms   +20 895 ms   +60 624 ms   +96 087 ms
and 4 of 11 never resolved at that backing at all
```

So "the answer was one frame away" is a property of that page, not of the class. Here the backing is
untranslatable for tens of seconds, and more than a third of the time for the rest of the boot.

Two consequences, and they pull in opposite directions, which is why both belong here.

**It strengthens the guard.** A fabricated address cached for 20–96 s is 20–96 s of writes aimed at
memory chosen by address rather than ownership, not one frame of it. The window the old path left
open was far larger than the flat-page measurement suggested.

**It weakens "refusing costs a frame".** It costs whatever that surface's content was worth for tens
of seconds, or permanently. Re-asking per frame does not recover it, because the answer is not
arriving next frame. If a Goal 3 patch turns out to be a surface whose backing never resolved, the
identity guard does **not** fix it — the pixels go nowhere instead of going somewhere wrong, and the
tile is blank either way. That is the single most important open question about the guard.

Caveat on the method, and it only cuts one way. Matching by `gva0` across 60–96 s cannot distinguish
"the same surface finally got mapped" from "the guest reused that virtual address for a new
allocation", so the long delays are an upper bound on how much was recovered. The **4 that never
resolve** carry no such ambiguity: those fabrications were never corrected.

#### A refusal has three outcomes, and only one of them is a cost

`.agents/repros/type4guess.py` scores a boot's refusals, and collapsing these would lose the claim:

- **corrected** — the same surface resolves cleanly within a frame or two. The guess had been
  standing in for an answer about to arrive.
- **terminal** — the surface is never resolved again for the rest of the boot. This is what a
  **teardown** looks like, and it is not a cost. Measured on `20260731-192622`, sids 14, 16 and 17
  each resolved cleanly many times over the first 40 s and then refused **at the same millisecond**
  (t=375745) and never came back: the owning address space went away. There the old path would have
  fabricated an address and written pixels into memory the guest had just freed — which is the
  write-after-free shape the Goal 1 panic census keeps landing in. Refusing at teardown is the
  device declining to write into a dead process.
- **stranded** — the surface does come back, but later than the window, so it was missing for more
  than a frame. This is the only outcome that is a cost of refusing.

On that boot: **corrected 3, terminal 6, stranded 0**, against `fabricated_attaches` 0 where the
pre-guard session had 106 (34 407 pages). No surface went missing for longer than a frame.

Two counting traps in the same scorer. `note_type4_fail` **latches on (sid, reason)**, so a surface
that refuses every frame contributes exactly one log line: the boot logged **9** refusals while the
census summed **201**. Neither number is wrong and they answer different questions — per-surface
outcomes key to the logged one, occurrence counts to the census — so print which is which. And the
`terminal`/`stranded` split cannot be made from the refusal lines alone; it needs the *later*
attaches for that sid, which is why the scorer reads the whole log rather than grepping.

One instrument correction, because the counter is asymmetric by construction.
`type4_identity_pages` **cannot be compared across arms**. With the guard off the loop runs to the
end and it counts every untranslatable page; with the guard on the loop returns at the first one, so
it counts at most one per refused attach and merely tracks `type4_translate_refused` (3 and 3 on
this boot). The pre-guard boot's guessing attaches averaged 325 pages each, so the guarded number
understates the fabrication it prevents by about that factor.

#### Measured: two interleaved pairs, and the guard fabricates nothing

`.agents/repros/type4-guess-ab.sh /tmp/type4ab 2`, `DRIVE_SECS=600`, arms one binary apart
(`REIMS_VGPU_TYPE4_IDENTITY_GUARD_OFF=1` is the only difference), interleaved with the leading arm
flipping each pair. All four logs re-scored **uniformly with one scorer afterwards** — the row the
harness writes live for the first arm was produced by an older scorer and reported `drift=2` where a
uniform re-score gives `1`, which is the double-count the `^`-anchoring rule elsewhere in this
document describes.

```text
            attaches  refusals  corrected  terminal  stranded  fab_attaches  fab_pages
arm-1         1 815         5          2         3         0             0          0
arm-2         2 412         6          4         2         0             0          0
ctl-1         2 448         0          –         –         –             8      4 519
ctl-2         1 885         0          –         –         –             9      3 284
────────────────────────────────────────────────────────────────────────────────────────
arm           4 227        11          6         5         0             0          0
ctl           4 333         0          –         –         –            17      7 803
```

Read it as three statements.

**The guard fabricates nothing, on 4 227 attaches.** Not "less" — none. Flip the knob and the same
binary invents a whole backing for 17 surfaces covering 7 803 pages, about **30.5 MiB of write
targets** aimed by address rather than by ownership.

**Every fabrication is whole-surface, and the geometry is the interesting part.** All 17 have
`gva_hits=0`: the walk failed for *every* page, not for a few that happened not to be faulted in.
Seven of ctl-1's eight are **1225×512** — the WebKit tile strip — and the rest are the 15×622
scrollbar. So the class lands on exactly the surface kind whose loss the Goal 3 report describes.

**No surface went missing for longer than a frame, in either arm.** `stranded` is 0 across 11
refusals; the split is 6 corrected and 5 terminal. Refusing costs a frame on the surfaces it
corrects and costs nothing on the ones that were being torn down.

Two things this does **not** say. It is not a crash-rate measurement — four boots cannot move a
2.2 % panic rate, and no panic or `.ips` from these boots has been tied to a fabricated address. And
`drift`/`lost` do not separate the arms here (1/1 and 0/0 against 0/0 and 0/0): at the pre-guard rate
a 2 000-attach boot expects well under one, so these counts are consistent with the fix and equally
consistent with short boots. **Quote the fabrication columns, not the drift columns.**

#### Exactly two things populate `page_entries`, and both are now sound

A completeness claim, of the kind this document says needs an audit rather than an assertion.
`page_entries` is the address list every mapping-keyed reader and writer resolves through, so "the
device cannot aim a write at an address it did not earn" is a statement about its writers. Grepping
every assignment, clear, push and extend of the field across the crate and excluding test modules
leaves **two** product sites:

- `runtime/objects.rs`, in `apply_type4_backing` — per-page walks of the owning task's page table.
  A failed walk used to be answered with the GVA; since `d455c3e` it refuses, and since the
  `walk=[...]` field it names which of the walk's fifteen checks refused.
- `runtime/mapper.rs:633`, `m.page_entries = plan.entries` — the guest's **own** IOSurface page
  table, reached through `contract::iosurface_pages::build_table_plan`.

The second one had never been audited and it holds up. It takes the table pointer from
`MappingInternal` fields 0x48/0x50, requires each to be a kernel VA *before* dereferencing and the
value read back to be a kernel VA *again*, tries both candidates, and returns `Err(Status::…)` with
a distinct slug for every failure — `iosurface_page_table_pointer_48_invalid`,
`…_50_read`, `iosurface_page_table_candidate_missing`, `…_failure_unattributed`. There is no branch
that invents an entry, substitutes one address space for another, or admits a value because it
happens to be backed by RAM. The addresses are the guest's own declaration, not this device's guess.

Everything else that touches the field is in a `mod tests`, and `model/state.rs` only ever `clear()`s
it. So the class the identity guard closed had exactly one other place it could have been hiding, and
it is not hiding there.

**Scope this claim precisely.** It is about `page_entries` and therefore about the mapping-keyed
rails. The raw-address rails (`flush_gva_one`, `flush_linear_one`) name guest addresses with no
mapping incarnation and are guarded by `deferred_pages_still_ours` and the fence ordering instead;
this audit says nothing about them.

#### A fabricated backing is contiguous, and a real one is not

Worth knowing because it makes every archived log re-readable without new instrumentation. The
fabricated GPA is the GVA, and the GVAs are contiguous by construction — page `i` is
`(backing_pfn + i) << page_shift` — so a fabricated span is exactly `page_count` pages end to end. A
real walk returns whatever the guest allocator gave. The same surface, one incarnation apart on
ctl-1:

```text
mid=165 gen=1 pages=640 lo=0x2a0138000 hi=0x2eb598000   contig_view_fragmented runs=161   real
mid=165 gen=2 pages=640 lo=0x1aeda000  hi=0x1b15a000    hi-lo = 0x280000 = 640 pages       fabricated
```

So `mapping_gpa_span` alone identifies the defect: a `src=type4` span whose `hi - lo` equals
`pages << page_shift` is a fabrication, and one with a `contig_view_fragmented runs=` in the hundreds
is a translation. Note also where the fabricated spans land — 0x18bda000–0x20dda000, **415–549 MiB**
— which is low guest RAM, where the kernel's own allocations live.

#### Fabrication tracks the page, not the clock — the flat-colour page barely provokes it

This section first read the data below as "fabrication is a late-session phenomenon" and concluded a
harness must drive for eight minutes to see it. The next boot falsified that in twenty seconds, and
the corrected reading is more useful than the original.

Fabrication timestamps, in ms since the device's first log line. The first two boots drive
`scroll-patch.sh`'s **generated flat-colour page**; the third is a Wikipedia browse:

```text
flat page   ctl-1   419 686  421 800  434 641  494 616  527 811  554 685  614 648  633 659
flat page   ctl-2   402 295  459 571  489 797  508 285  508 320  568 190  589 265  628 321  629 455
wikipedia           20 714  102 541  143 594  190 270  224 730  269 402  285 290  …
```

**Nothing before 402 s on the flat page; the first fabrication on Wikipedia is at 20.7 s.** So the
variable is the workload, not elapsed session time, and the earlier "accumulated churn" guess was
guessing at a pattern that was not there. Two boots of one page cannot tell a property of the class
from a property of the page — and this document already says a single workload cannot, in the
sentence about a scorer written from one report.

The corrected reading matters for what to point a pixel scorer at. The generated page is six flat
colours in promoted layers, which is what makes it *scoreable* by palette — and it is a weak
provocation, producing a handful of long-lived tiles. Wikipedia produces a stream of short-lived
surfaces at many geometries: the seven above are 62×52, 1225×512, 193×864, 1225×512, 64×64, 16×16
and 1225×512, so the class is **not** confined to the WebKit tile strip that the first A/B's control
made it look like. Wikipedia is also where the user reported the defect.

That leaves the harness with a real tension rather than a fix: the page that provokes the class is
the page whose palette cannot be scored, and the page that can be scored barely provokes it.

`scroll-patch.sh`'s three recorded `none` results are still unexplained by anything measured. "It
never reached the state" is now only one candidate and no longer the leading one.

#### The first attempt to join them measured its own gap — read the row as inconclusive

`.agents/repros/g3-fabrication-ab.sh` tried to split the difference: browse Wikipedia for
`PREDRIVE_SECS=480`, then score the flat page, so one boot carries both the provocation and a
scoreable frame. Its control row looks like a result and is not one:

```text
ctl-1 OK attaches=2346 stores=18839 fabricated_attaches=20 persistent=[none] unscoreable=[none]
```

Twenty fabrications and a clean screen at all 24 scored offsets — which reads as "fabrication does
not produce the visible patch". Then read the timestamps. The 20 fabrications land at
t = 21, 103, 144, 190, 225, 269, 285, 306, 327, 347, 367, 408, 450, 468, 469, 489, 493, 494, 494,
494 s, and the scored scroll phase ran from ~500 s to 660 s. **Not one fabrication happened during
the phase that was scored, and none of them was on the page that was scored.** The provocation and
the measurement were disjoint in both time and document.

So the row is `INCONCLUSIVE`, not evidence in either direction, and the design is what produced it:
predriving on one page and scoring another separates the two by construction. Do not cite that
`persistent=[none]`.

#### Scoring a patch without a palette: force the offset to paint twice

`.agents/repros/reload-diff.py` is the instrument for the provoking page, and it drops the palette
requirement entirely. The reported defect is *content that never arrived* — "that patch is all
white/blank … that glitched patch lives in that particular place in the scroll buffer" — which is a
claim about the difference between what was shown and what should have been. So make the page paint
the same offset a second time (`cmd-R`, then the same counted key presses) and compare the pair. A
patch is then **a localized disagreement between two renders of one offset**, which needs no palette
and no page generator, and works whether the loss is white, black, page-background or stale content.

The gate is the same quantity as the score, which is what makes it hard to fool. Two captures of the
"same" offset reached by counted key presses may not be the same offset, and a harness that could
not tell would score the mismatch as an enormous defect — the direction that reads as a finding.
`same_frac` below `--min-same` is `UNSCOREABLE`; only inside a comparable pair does a localized
disagreement count. Direction is reported too, and asymmetrically: `blank_first` (uniform in A,
detailed in B) is the loss, and *both sides detailed* is `CHURN` — page animation, a clock, a
lazily-loaded image — counted apart so a live site's moving parts cannot be folded into the count.

**It was blind when first written, and the injected controls are what caught it.** Six synthetic
pairs against a detailed frame:

```text
identical                       CLEAN
white rectangle in A            PATCHED blank_first   <- all three were CHURN before the fix
black rectangle in A            PATCHED blank_first
page-background rectangle in A  PATCHED blank_first
region shuffled in A            CHURN                 (animation, correctly not a finding)
a different frame               UNSCOREABLE offset_mismatch
```

The bug was that "is this region uniform" took the **max** spread over the blob's cells, and a cell
straddling the edge of a solid patch holds patch on one side and page on the other. One such
boundary cell reported spread=255 for a perfectly solid white rectangle, so every injected fill
classified as page animation. Eroding the blob by one cell before measuring is the fix. This is the
same shape of blindness `scrollpatch.py` had — a rule that could not see the defect it was built for
— found the same way, by injecting the defect into a control and checking the scorer notices.

`.agents/repros/wiki-reload-diff.sh` drives it: pass A over N screenfuls, reload, pass B over the
same offsets, score each pair. It carries three gates, and the first two are the failures this
document has already paid for — consecutive captures inside a pass must **disagree** (the page
actually scrolled), and after the reload the first capture must **agree** with pass A's first (the
reload happened and re-registered at the top). A run failing either states so instead of reporting a
clean page.

### The user's crash report is a corrupt malloc free list under a backdrop blur

`~/Downloads/crash-report-from-user.txt`, WindowServer on macOS 13.7.8, 76 s after boot. Read the
stack from the bottom up, because the top of it is not the bug:

```text
 5  small_free_list_remove_ptr_no_clear + 1017     <- malloc finds its free list corrupt
 4  malloc_zone_error                              <- and aborts
 9  _objc_rootAllocWithZone
10  AppleParavirtGPUMetal                          <- the allocation that discovered it
11  CA::OGL::MetalContext::start_render_encoder
19  CA::OGL::Context::blur_surface
20  CA::OGL::GaussianBlurFilter::render
22  CA::OGL::filter_backdrop
23  CA::OGL::capture_backdrop
```

`EXC_CRASH (SIGABRT)`, `abort() called`. This is **not** a fault in the paravirt driver: malloc's
small-zone free list was *already* corrupt, and the driver's next allocation is merely what walked
into it. The guest kernel panic census in this document already contains "a malloc small-zone free
list" and a `kalloc` poison report reading `val:0xffffffffffffffff`; this is the userspace member of
that same family, and it is the shape a stray write of opaque white BGRA leaves.

The frame it happened under is the actionable part, and an earlier revision of this section named it
too broadly. It is not "blur on screen": the crashing thread is **`ws_main_thread`**, not the render
server, and the blur chain is nested inside a **window capture to an IOSurface** —
`_XHWCaptureWindowListToIOSurface` → `WSCaptureCreateIOSurfaceMachPortForWindowList` →
`CaptureSurfaceMetal::Populate`, with `capture_backdrop` / `filter_backdrop` under that. Capture
allocates a fresh destination IOSurface per request and blur allocates a chain of short-lived
intermediates inside it, so the provocation is *capture of a blurred window*, which churns far harder
than either alone. That churn is precisely the condition that makes the device resolve a type-4
surface before the guest has finished mapping its backing, and therefore the condition under which
the pre-guard device fabricated a GPA from a GVA.

**State this as a consistency, not an attribution.** Three things line up — a corrupt heap free list,
a workload that maximises transient surface churn, and a device path that answered that churn with an
invented address — and none of them is a measurement of this crash. Nobody has tied a specific
guessed entry to this abort, and the report predates the guard.

It does dictate the next repro, though, and that is the point of writing it down. `scroll-patch.sh`
uses flat colour bands, which produce a handful of long-lived tiles and barely exercise the race;
`.agents/repros/blur-provoke.sh` is the opposite by construction — `backdrop-filter: blur()` plus
promoted layers — and scores the same `type4guess.py` counters so the two pages can be compared.
Note that page **cannot** be scored by `scrollpatch.py`: blur produces intermediate colours by
construction, so the flat-palette rule that makes that scorer sound does not hold, and pointing it
there would score correct rendering as a defect.

#### This rig has already reproduced the user's crash, and a second report carries the payload

Two guest crash reports were captured by earlier repro runs and left unread in their output
directories. Both are `iMac19,1` / macOS 13.7.8 (22H730) — the user's exact environment, so the
comparison is not confounded by build. Neither had ever been opened.

`/tmp/ch1/ips/WindowServer-2026-07-30-221616.ips` is **the user's failure, on our hardware**:
`EXC_CRASH (SIGABRT)`, `abort() called`, 57 s uptime, the same detector at the same offset
(`small_free_list_remove_ptr_no_clear + 1017` → `malloc_zone_error + 183` → `abort + 123`). Two
differences carry information rather than noise:

- It is found from the opposite end. The user's report walks into the corrupt free list while
  **allocating** (`small_malloc_from_free_list`); this one while **freeing**
  (`free_small` ← `CA::Render::Image::~Image` ← `CA::Render::Context::delete_object`).
- **There is no paravirt frame on the stack at all**, and it is the render-server thread, not
  `ws_main_thread`. The block being freed is a plain `CA::Render::Image` pixel buffer.

That second point is the one to keep. The user's report alone is consistent with a bug *inside* the
driver's own allocations, because the driver is on the stack. This one hits the same free list from a
destructor that never touches the GPU, which means the corrupting write is not on the path that
discovers it.

`/tmp/icon-ab2/r5-ctl/drive/ips/Safari-2026-07-31-153215.ips` is a different signature with the same
cause, and it is the first time this project has recovered the **payload value** in userspace.
`EXC_BAD_ACCESS (SIGSEGV)`, `KERN_INVALID_ADDRESS at 0x18`, on the main thread inside
`objc_msgSend` ← `-[__NSDictionaryM objectForKey:]` ← CFPreferences ←
`+[AppController shouldPersistPrivateWindows]`. **`rdi = 0xffffffffffffffff`** — at `objc_msgSend`
entry RDI is the receiver, and `rsi` holds a plausible shared-cache selector pointer, so the frame is
at ABI entry state and the receiver really was an all-ones word. The `0x18` fault address is derived,
not the corrupt value: an all-ones pointer is read as a tagged pointer, resolves to a nil class, and
the class-cache load at `class+0x18` faults. `cr2=0x18 trap=0xe err=0x4` all agree.

So the corruption is a **bulk 0xFF overwrite selected by address, not by ownership**, and the victim
is whatever the allocator happened to place there — here a preferences dictionary in a *non-graphics*
subsystem of a *user* process. That is the same `val:0xffffffffffffffff` the kernel panic census
records and the same shape as the `airportd` indirect call through a 0xFF pointer, and it says the
class is not confined to WindowServer. No thread in either report is executing GPU or IOSurface code.

**Cautions.** The Safari report was captured during the control arm of an unrelated A/B, so a guard
was disabled — but it is `n = 1` and unpaired, and nothing here is an A/B result. The 0xFF value is a
shape match to the stray-write hypothesis, not an attribution to any specific fabricated address.

The operational lesson is separate from the diagnosis: **repro runs collect `.ips` files and nobody
reads them.** Two reports sat on disk for a day, one of them a reproduction of the exact crash the
project is chartered to fix. Check `*/ips/` in every run directory before concluding a run found
nothing — and note that `/tmp/ch2/ips/` contains a zero-byte file named `Retired`, which is the
`ls -1` subdirectory trap this document describes elsewhere, frozen in place.

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
