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

~~The `0xFF` recurs across all of them and is the same write seen from different victims: opaque
white BGRA pixels.~~ **Struck: tabulated, it recurs in 5 of the 12, and in 7 it does not appear at
all.** See the section below before building anything on it.

Panics still fire on binaries carrying the fence repairs (three on 2026-07-31 alone, one of them
14:43 on a tree with all three raw-address rails bound). A fence binding orders a write against the
guest's completion stamp; it does not make the destination address correct. The address is
`mapper::type4_pages_still_ours`'s question.

At 2.2 % an A/B needs ~150 boots per arm and is not affordable. Let panics accumulate as a side
effect of every boot and re-sweep the census instead.

##### Tabulated: the `0xFF` is in 5 of the 12, and one grep makes it look like all 635

Nobody had counted. The claim struck above — "the `0xFF` recurs across all of them" — has shaped
every Goal 1 decision since it was written, including which instruments got built. Tabulating the
evidence *per panic*, rather than reading the pile:

```text
                                         bulk 0xff fill        all-ones register
20260728-154602  Safari                        –                      –
20260729-152409  WindowServer                  –                      –
20260729-155116  followupd            sz:6144  off:0                  –
20260729-155610  (nested panic)                –                      –
20260729-174129  ReportCrash                   –                      –
20260729-175456  com.apple.AppleU     sz:256   off:0                  –
20260729-183422  ReportCrash                   –                      –
20260729-195245  airportd                      –                      –
20260730-210813  WindowServer                  –                      –
20260731-135242  tccd                          –              RAX, R14   CR2=0x7
20260731-140555  airportd                      –              RIP, CR2   CR2=0xffffffffffffffff
20260731-144329  WindowServer                  –              RAX        CR2=0x16f
─────────────────────────────────────────────────────────────────────────────────
                                            2 of 12               3 of 12
```

**Seven of the twelve carry no `0xff` evidence at all.** Two carry the real thing — a `kalloc` poison
check reporting a *whole freed element* filled from `off:0`, at `sz:6144` and `sz:256`. Three carry a
single 8-byte field.

**The three register cases are one shape, and the arithmetic says so.** Both faulting addresses are
an all-ones pointer plus a small structure offset:

```text
tccd          CR2 = 0x7   = (-1) + 0x8      a load 8 bytes into an all-ones pointer
WindowServer  CR2 = 0x16f = (-1) + 0x170    a load 0x170 bytes into an all-ones pointer
airportd      RIP = CR2   = -1              an indirect CALL through an all-ones pointer
```

and the Safari `.ips` in `/tmp/icon-ab2/` is the userspace member (`rdi = -1`, faulting at `0x18`,
which is where objc looks up the class cache after resolving an all-ones receiver to a nil class).

**Do not read that as refuting the bulk-fill reading — it does not distinguish.** A bulk `0xff` fill
sets every pointer inside the region it covers to `-1`, so a single dereferenced field holding `-1`
is exactly what a victim *of a bulk fill* looks like from the register dump; the dump only shows the
one field the code touched. What the table does establish is narrower and still worth having: the
direct evidence for a bulk fill is **2 observations, not 12**, and any instrument justified by "the
whole class is 0xff" is justified by two data points.

**The harness trap, and it is the `Retired` shape again.** Every one of these logs contains
`libignition: 1: log fd : 0xffffffffffffffff` — a boot-time line meaning "no log fd", i.e. `-1`. It
is in **630 of the 635 serial logs on disk**, panicking or not:

```text
serial logs                                    635
containing 0xffffffffffffffff anywhere         630   (99.2 %)
of which actually panicked                      12
```

So `grep -l 0xffffffffffffffff vm/disks/run/serial-*.log` returns essentially every boot ever run,
and a scorer built on that pattern reports a hit for a clean boot. **Anchor on the evidence, not on
the value**: `element modified after free (off:…, val:0x…, sz:…)` for a bulk fill, and an anchored
register name (`\b(RAX|RIP|CR2|R14): 0xffffffffffffffff`) for the single-word form. Confirm you have
seen the negative before believing either.

##### Re-swept at 608 boots: 12 panics, and every one predates the identity guard

The sweep is the prescription above, carried out. 608 serial logs, **12 panics (1.97 %)** — the same
12. The newest is `20260731-144329`; `d455c3e` (the type-4 identity guard) landed at 19:17 that day
and `fbf7bd9` (the fence bindings) at 17:02.

```text
boots after fbf7bd9 (fence bindings)   43    panics 0    expected 0.85 at the historical rate
boots after d455c3e (identity guard)   14    panics 0    expected 0.28
```

**Do not read this as the repair being confirmed.** Under the historical rate a run of 43 clean boots
has p = 0.42, and 14 has p = 0.76 — both are exactly what a short run looks like whether or not
anything was fixed. It is consistent with the fix and equally consistent with having not yet drawn a
panic. The direction is right and the n is not there; say so in that order.

What it does establish is the denominator, so the next agent does not re-derive it: **the post-guard
arm needs roughly 150 boots before a zero means anything**, and it has 14. Keep booting for other
reasons and re-sweep — that is cheaper than a dedicated soak and it is the same data.

Sweep with the loop above; count boots with `ls -1 vm/disks/run/serial-*.log | wc -l` and split by
timestamp against a commit date (`git log -1 --format=%cd --date=format:'%Y%m%d-%H%M%S' <ref>`), since
the log name is the run stamp and sorts lexically.

###### Re-swept at 633 boots: still the same 12, and the post-guard arm is 39 of the ~156 it needs

```text
633 boots        12 panics (1.90 %)      newest panic 20260731-144329
post-d455c3e     39 boots, 0 panics      expected 0.74      p(0 | rate unchanged) = 0.47
```

A coin that comes up heads is not evidence the coin is bent. **p = 0.47 means this run is exactly
what an unfixed device looks like half the time**, and the arithmetic gives the target precisely:
`log(0.05) / log(1 - 0.0190)` = **156 boots** before a zero clears the 5 % bar. The arm has 39. Quote
the 39, not the zero.

**A dedicated soak is the wrong way to get the remaining 117, and this section already said so two
paragraphs up.** One was run anyway — `panic-rate.sh` for 40 boots on a pinned QEMU — and it was
stopped at 10 in favour of driven boots on HEAD, because the driven boots accumulate *the same panic
data* (every boot writes `vm/disks/run/serial-*.log`, and the census re-sweeps all of them) while
also exercising the instruments the session was building. A soak buys one number; a driven boot buys
that number and a measurement. The 10 boots it did produce are banked and counted above — stopping it
lost nothing but its future boots.

The pinned-binary trick is still right for anything that must not see a changing tree; it is the
*dedication* that was wasteful, not the pinning.

####### Re-swept at 635 boots: 41 of the ~158 the arm needs

```text
635 boots        12 panics (1.89 %)      newest panic still 20260731-144329
post-d455c3e     41 boots, 0 panics      expected 0.77      p(0 | rate unchanged) = 0.46
```

Two driven boots' worth of progress, which is what an accumulating census looks like and is the
point of not dedicating the rig to it. The target moves with the rate:
`log(0.05) / log(1 - 0.0189)` = **158**. Quote the 41.

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

#### The same trap without `-c`: one census key is a suffix of another

`grep -o KEY=[0-9]*` over a scorer's own output row is the natural way to pull a field back out of
it, and it silently returns **two** values whenever one key ends with another. The census rows here
carry both `attaches=` and `fabricated_attaches=`, so

```sh
attaches=$(printf '%s' "$row" | grep -o 'attaches=[0-9]*' | cut -d= -f2)   # -> "2346\n20"
```

produces a two-line string, exactly like the `|| echo 0` case above, and it then flows into a numeric
comparison and a `printf`. It ran for two boots of `g3-fabrication-ab.sh` before anyone looked at the
row closely enough to notice it had wrapped:

```text
ctl-1 OK attaches=2346
20 stores=18839 fabricated_attaches=20 …
```

The row still says `OK`, still carries every other field, and the stray line reads as a formatting
quirk rather than as a variable holding two numbers. Anchor the key at a word start —
`grep -oE '(^| )attaches=[0-9]+'` — and check a pulled field by printing it in brackets
(`attaches=[2346]`) the first time the extractor is written. `stores=` is a suffix of `min_stores=`
for the same reason, and any new counter can create the collision later without touching the
extractor.

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

### A source gate that strips `mod tests {}` still reads eight whole test files as shipped code

`observe/gate.rs` scans the crate's own source to answer questions no runtime test can — "which
writes can reach guest RAM with no page set bounding them?", "does every caller of the witness go
through the policy?". Every one of those scans first calls `production_source`, which strips
`#[cfg(test)] mod tests { … }` **blocks** so a fixture exercising a product helper is not scored as a
shipping use of it.

That is the wrong half of the problem. A test module can also be a **file**, declared
`#[cfg(test)] mod tests;` and living in `tests.rs` next door, and `production_source` sees that file
as an ordinary module with no `#[cfg(test)]` anywhere in it. Eight files in this crate are declared
that way today, and they are not all obviously test files by name:

```text
runtime/drain/tests.rs   runtime/compute_exec/tests.rs   runtime/metal_draw/tests.rs
runtime/icb/tests.rs     backend/vulkan/caps/gate.rs     backend/vulkan/translate/gate.rs
backend/vulkan/translate/coverage.rs                     observe/gate.rs
```

The last one is the gate file itself, so every source scan was reading its own scanning code as
product. All three scans were affected and none of them was wrong *yet* — which is the point: the
gate would have started lying the first time a fixture in one of those files called the pattern being
counted, and it would have lied in the direction of a larger, more alarming count that a reader would
then chase.

`production_files` is the fix — enumerate the files a `#[cfg(test)] mod x;` declaration names and drop
them before scanning — and it has a test asserting the **exact** dropped set in both directions, so
neither adding a test file nor promoting one to product can move the denominator silently. A source
gate needs the same treatment as a harness: it fails in the direction that looks like a finding.

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

#### Answered: the walk fails because the guest's level-2 directory entry is zero

The `walk=[...]` field added in `e1632f6` answered this on its first boot, and the answer is the same
for every refusal on it. Eleven refusals over 1 112 attaches, x86/Vulkan, Safari on Wikipedia:

```text
walk=[tid=0 act=1 dir=0x3cc9e8 root=0x3cc527 depth=3 st=zero-pfn pte=0x0 lvl=2 idx=973]
walk=[tid=0 act=1 dir=0x3cc9e8 root=0x3cc527 depth=3 st=zero-pfn pte=0x0 lvl=2 idx=667]
…  11 of 11 identical but for idx
```

Every field except the index is constant, and each rules something out:

- `tid=0 act=1` with the same `dir`/`root` as every *successful* walk on the boot — so it is not the
  wrong task, and not a task whose page table had gone away. The task-search story stays dead.
- `root` read cleanly and `depth=3` decoded — so it is not a malformed or unreadable directory.
- `pte=0x0` exactly, not garbage — so it is not corruption, a torn read, or a stale mapping.
- **`lvl=2`, not the leaf.** The walk stops at the page *directory*. A zero entry there means the
  guest has not mapped **the whole region**, not one page of a surface.

So the device asked for a translation of an address range the guest had not mapped at all, and the
old path answered that by using the virtual address as a physical one. "The device answered before
the guest was ready" is now measured rather than inferred, and it is coarser than it looked: the
absence is a directory entry, so the surface's entire backing is missing at once — which is exactly
why every fabricating attach in the A/B had `gva_hits=0` rather than a few unfaulted pages.

This is what the single word `translate` was hiding. Fifteen distinct refusal slugs existed the whole
time and the one that fires is `zero-pfn` at level 2, every time.

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

##### The capture is a whole desktop, and four fifths of it cannot scroll

The third gate was the broken one, and it was broken twice over. Both defects come from the same
unexamined assumption: that the capture is *the page*. It is not — it is the entire macOS desktop,
with wallpaper, Dock, menu bar, Safari's own chrome, the Wikipedia sidebar and the reader-settings
panel all in frame. Measured on a real control capture, the article column is `view=384,136
464x424` on a 1280x719 frame: **21 % of the picture**. Everything else is furniture that no scroll
can move.

**The gate was the same quantity as the score.** `reload-diff.py` called a pair comparable when
`same_frac >= --min-same` (0.90) and unscoreable below it. But a defect *lowers* agreement, so a
large enough defect trips the mismatch gate and is reported as `UNSCOREABLE` — which reads as "not a
finding". Injecting a solid white rectangle into a real capture pair from `/tmp/wikictl` and
sweeping its size:

```text
area_frac 0.008  CHURN                 (below --min-blob; correctly not a finding)
area_frac 0.018  PATCHED           <-- sensitivity window opens
area_frac 0.100  PATCHED           <-- and closes
area_frac 0.130  UNSCOREABLE offset_mismatch same_frac=0.8692
area_frac 0.399  UNSCOREABLE offset_mismatch same_frac=0.6849
```

Those fractions are of the *whole capture*, so the ceiling was reached by a loss covering roughly
**half the visible article column**. And no threshold would have fixed it: real anchor-mismatched
pairs measure `same_frac` 0.75–0.80 while injected patches measure 0.87 down to 0.68. **The two
populations overlap**, so the number cannot separate them at any setting.

(State the tile size carefully. A 1225×512 WebKit tile is 627 200 px, but that is the guest
*surface*; its footprint on screen is bounded by the article column. A first draft of this section
called it "0.68 of the frame" by confusing the two.)

**The replacement gate asks about alignment, not agreement**, and it must be measured *inside the
scrolling column* — a first cut that voted over the whole frame returned `shift=0` for **every**
mismatched pair, because the wallpaper, Dock and chrome genuinely are at zero displacement and they
outvote the content. One such pair then scored `PATCHED`: a false finding on two captures that were
simply at different places in the document.

So the scorer now takes `--viewport-from`, a capture from the same pass at a different offset, and
scores only what moved between them. That definition needs no knowledge of the page, the browser or
the window layout. Inside it, rows whose sampled signature is *rare* vote for their displacement in
the other capture. The mechanism is the point: **a blank patch does not vote wrong, it does not vote
at all** — uniform rows share a signature with hundreds of others, so they are discarded as
non-distinctive, and the patch removes its band from the gate instead of corrupting it.

Validated in both directions on real captures, which is the discipline this rig keeps re-learning —
a gate is not trustworthy until it has printed both the negative and the positive:

```text
8 of 8  same-anchor pairs       scoreable, shift=0, margin 15-300x over runner-up (needs 2x)
7 of 7  different-anchor pairs  UNSCOREABLE — none scored as a finding (all 7 scored CHURN or
                                PATCHED under the old whole-frame gate)
injected fill, share of viewport:
  0.09 .. 0.94                  PATCHED, for white, black AND page-background
  1.00                          UNSCOREABLE no_alignment — honest: with the whole column blank
                                there is nothing left to place the offset with
```

Restricting to the viewport also made `same_frac` informative as a *reported* quantity: same-anchor
pairs measure 0.97–1.00 and different-anchor pairs 0.14–0.34, where whole-frame they were 0.97 and
0.75–0.80. It is still not the gate, and must not become one.

##### Measured: fabrication did not produce a visible patch in either arm

One pair of boots, x86/Vulkan, Safari on the same Wikipedia article, 8 anchors scored twice each,
re-scored **uniformly with the fixed instrument afterwards** (the rows the harness wrote live came
from the blind scorer and are void):

```text
                 attaches  fabricating attaches  fabricated pages  PATCHED offsets
ctl (guard off)     541             10                 3 535            0 of 8
arm (guard on)      543              0 (41 refusals)       0            0 of 8
```

The control's provocation was **inside** the scored window, which is the check `g3-fabrication-ab.sh`
failed: six of its ten fabrications land at t = 48.3–104.6 s against a pass-A capture window of
t ≈ 49.5–105.5 s, four of them on 1225×512 WebKit tile strips and one on the 1240×622 Safari content
surface. So this is not the disjoint-provocation trap — the fabrication hit the right surface class
at the right time, and no patch appeared.

**Read this as a result about fabrication, not as a clean bill for Goal 3.** It says the identity
guard's class is not sufficient to produce the reported patch on this workload. It does not say the
patch does not exist, and one boot per arm cannot carry a rate.

One caution for whoever reads the two reports side by side. The per-offset `same_frac` and
`diff_cells` come out **identical between the two boots** (0.9727, 0.9457, 0.9772, 0.9772, 0.9831,
0.9707, 0.9814, 1.0000), which looks exactly like a harness scoring one run twice. It is not: the
captures differ by 123 058 pixels at offset 0 and the scorer's own vote count differs (280 vs 270),
so it is demonstrably reading different files. The agreement is itself the finding — **the A-to-B
difference on this page is page-deterministic**, driven by first-load-versus-reload layout rather
than by anything the device did. That is also why it is weak evidence: an instrument whose output
does not move between arms is measuring the page, and a workload that provoked the device would show
boot-to-boot spread.

#### Traced end to end in one boot: fabrication, drift, and a WebKit tile that never lands

The chain from a fabricated address to a permanently lost tile had been assembled from separate
sessions and separate surfaces. It is now on one surface, in one control boot (`/tmp/wikictl`,
guard off), with every step carrying the same backing address `gva0=0xb9d9000`:

```text
t= 41150  type4 pages sid=62 n=640 gva_hits=640 id_hits=0   gpa0=0x2cd26c000   real walk, empty
t=104620  type4 pages sid=62 n=640 gva_hits=0   id_hits=640 gpa0=0xb9d9000     FABRICATED (= gva)
t=120824  mapping_page_drift mid=62 cached=0xb9d9000 live=0x2cd26c000          refused
t=120824  deferred_flush_lost kind=render 1225x512 reason=mapping_page_drift   tile lost
t=120931  type4 pages sid=62 n=640 gva_hits=640 id_hits=0   gpa0=0x2cd26c000   real walk, content
```

Read the third line against the first. **`live` at the drift is byte for byte the address the walk
returned 80 s earlier**, and the walk returns it again 107 ms later. The surface never moved. What
drifted was the device's own guess, and the log called it `translation_moved (the guest re-pointed
this surface and no packet said so)` — a sentence about a guest bug that does not exist. Pre-guard
sessions logged 7 of 9 drifts this way.

So the mechanism is: a transient walk failure makes the device fabricate GPA = GVA and **cache it**;
the render targets the fabricated pages; the drift guard later catches the mismatch and correctly
refuses to write; and `flush_intersecting` has already **taken** the window out of
`compute_deferred_flush`, so the obligation is gone and nothing re-arms it. The tile's pixels land
nowhere. On a WebKit tile, painted once and never repainted, that is permanent and pinned to one
scroll offset — the reported Goal 3 symptom exactly.

The guard closes this path: the arm boot had 0 fabrications, 0 drifts and **0 lost flushes**, against
the control's 10 fabrications, 2 drifts and 2 losses (one at 1225×512, one at 1877×24).

**Two cautions, and the first one matters more than the trace.**

The pixel scorer saw none of this, and the reason is timing, not invisibility: the loss is at
t=120824, pass A's last capture is at t≈105500 and pass B's first at t≈128500, so the tile was lost
**during the reload** and repainted before pass B looked. A scored window has to cover the loss, and
nothing in the harness currently arranges that.

And the fabrication at t=104620 landed a frame before pass A's last capture, which scored
`same_frac=1.0000 diff_cells=0` — pixel-perfect. That is worth sitting with: **a fabricated write
does not necessarily show on screen**, because the compositor can serve the frame from our own
resident image while the guest's pages are the ones left wrong. It bounds what any screenshot-based
Goal 3 scorer can ever prove, and it is a reason to weight the fail-log rails over the pixels.

`identity_entry_corrected` is now a distinct decline reason from `translation_moved`, so the two are
countable apart in any later census.

#### The Goal 3 loss is now countable, and both controls have been run live

`flush_render_one` refuses on five grounds. The three resident-mismatch ones always carried census
routes; the two **drift** ones did not, and they are the two that lose a painted tile. They are now
`rendflush_gen_drift` and `rendflush_page_drift`, so a boot can be scored on the Goal 3 event from
the per-second census instead of by grepping the fail log — which matters because the fail log is one
line per occurrence and the census is per-interval, the convention mismatch this document records as
having produced a 100× error.

Validated in production, one binary apart, both arms driven by `wiki-reload-diff.sh` on the same
article:

```text
                  attaches  fabricating  fab pages  rendflush_page_drift  deferred_flush_lost
arm (guard on)       520          0            0             0                    0
ctl (guard off)      513          5        1 292             1                    1
```

The control is the positive control the counter needed: **a counter that has only ever read zero has
not been shown to work.** It fired once, on the same event the fail log recorded, and the arm read
zero on a boot with 3 514 `surface_flush`es — so the zero is a measurement and not a dead wire.

The same boot is also the live check on the reattribution above. Its one drift reads

```text
mapping_page_drift mid=14 page=0/1 gva=0x4130000 cached=Some(68354048) live=Some(14086090752)
  reason=identity_entry_corrected
```

and `68354048` is `0x4130000` — the GVA exactly. `translation_moved` did not fire at all on that
boot. Before this change that line would have read "the guest re-pointed this surface and no packet
said so".

**Do not read the control's pixel verdicts.** Its reload gate failed (`same_frac=0.7819`, needs 0.90)
and the harness says so; offset 0 came back `UNSCOREABLE shift=33`, which is the new gate correctly
measuring a 33-row displacement rather than scoring it as a defect. The census counters come from the
fail log and are independent of the pixel scoring, so the counter validation stands while the pixel
arm is void. Keep the two apart.

#### Four boots, two arms, byte-identical pixel scores — this page cannot separate arms

Every offset of every `wiki-reload-diff` run so far scores the same, to four decimals, across four
boots and two arms:

```text
same_frac   0.9727  0.9457  0.9772  0.9772  0.9831  0.9707  0.9814  1.0000
diff_cells      84       -       -       -      52      90      73       0
votes          280     322     214     233     342     236     297     312
```

It is not a harness scoring one run twice — the captures differ by 456–2 331 pixels between boots,
and at offset 0 by 123 058. The 8 px cells and the tolerance quantise that boot-to-boot noise away,
and what survives is the page's own first-load-versus-reload difference, which is deterministic.

So the honest reading is that **this instrument has no measurable device sensitivity on this page**,
and running more boots of it will not acquire any. It is a working detector — injected fills from
0.09 to 0.94 of the viewport are caught — pointed at a workload that does not produce the defect.
Goal 3 needs a different provocation, not a larger n. `blur-provoke.sh` (window capture with a
backdrop blur, the frame from the user's own crash report) is the untried candidate, and the crash
report says that is the churn that provokes.

##### The two instruments now meet, and the price is the animation

`.agents/repros/blur-reload-diff.sh` is that combination: a backdrop-blur document scored by
`reload-diff.py`. It works because that scorer needs **no palette** — a patch is a localized
disagreement between two renders of one offset, which is the user's description restated — so the
intermediate colours that make blur unscoreable by `scrollpatch.py` are irrelevant to it.

`wiki-reload-diff.sh` is reused unchanged as the scoring half, via a new `ANCHORS_SRC` env var that
makes `discover_anchors` read a locally generated page instead of `curl`ing one. `curl` cannot fetch
a `file://` URL that exists only in the guest, and hard-coding the anchor names in the caller would
let a generator change silently reduce a run to fewer offsets than it reports — the failure the
network path already avoids by reading the live article rather than guessing its section names.

**State the price when quoting any result from it.** `blur-provoke.sh`'s page carries a `.spin`
element animating forever, which is what makes it churn hardest at rest. That animation makes the
page unscoreable here by construction: pass A and pass B would differ everywhere and every offset
would return `CHURN`, which is not a finding. The anchored page drops it, so **it is a weaker
provocation and must not be described as the same workload.** What remains is real —
`backdrop-filter` still routes through `filter_backdrop`/`capture_backdrop`, and each anchor
navigation re-blurs the newly exposed bands — but at capture time the page is at rest, which is
exactly why it can be scored at all.

So a `none` from this harness means "the strongest provocation that is also scoreable did not produce
it", not "Goal 3 is absent". The run prints `type4guess.py` and footprint counters beside the pixel
verdicts for that reason: **a run whose type-4 counters are all zero provoked nothing and has no
opinion about the page.** Check those before reading the verdict line.

Two page-design traps that are not obvious and are already handled, so a future edit does not
reintroduce them. The per-band filler text must **differ per band**: `reload-diff.py`'s alignment
gate votes with rows whose sampled signature is *rare*, and a page of interchangeable rows starves
that vote into `no_alignment` — unscoreable, on a page that rendered perfectly. And the bands need
fine detail at all: a blurred flat colour is still flat, and the scorer calls a uniform region
"blank", so a page of uniform bands would make every cell look like a lost tile to the very rule that
detects one.

#### Measured on the provoking workload: 65 141 render flushes, none refused

The Wikipedia page cannot separate arms, so the Goal 3 loss rail was re-measured on
`blur-provoke.sh` — the backdrop-blur-under-window-capture page built from the frame the user's own
crash report names, which is the heaviest surface churn this rig can produce. Shipped configuration
(guard on), 600 s driven, x86/Vulkan, 579 census intervals:

```text
attaches 1 571      stores 65 172      guest_panics 0
type-4 refusals 6:  corrected 2   terminal 4   stranded 0
fabricated_attaches 0             fabricated_pages 0

surface_flush              65 141     <- attempts
  rendflush_gen_drift           0
  rendflush_page_drift          0
  rendflush_resident_absent     0
  rendflush_epoch_cleared       0
  rendflush_epoch_drift         0
deferred_flush_lost               0     mapping_page_drift 0
```

`surface_flush` is incremented at the top of `flush_render_one`, **before** every refusal branch, so
it is the denominator and the five `rendflush_*` counters are the numerator. The claim this supports
is therefore a completeness one, not a rate: **every one of 65 141 render-window flush attempts
landed, and not one was refused on any of the five grounds.** Zero of the type-4 refusals stranded a
surface — 2 were corrected within a frame and 4 were teardown.

Scope it precisely. This is one boot, one workload, the mapping-keyed render rail, and it says
nothing about the raw-address rails or about a rate for anything that did not happen. What it does
retire is the idea that the Goal 3 loss path is still open on the shipped binary under the strongest
provocation available: it is not firing at all, on the workload most likely to fire it.

#### Audited: `flush_intersecting` is the only place a live surface's obligation is dropped

A delegated enumeration of every product mutation of `compute_deferred_flush`, `gva_deferred_flush`,
`linear_deferred_flush` and `deferred_alias_pages` (16 sites; 11 test-only sites excluded) proposed
two further loss sites. **Both are wrong, and the way they are wrong is worth keeping.**

`retire_linear_residents` (`storage_flush.rs:1371`) drops a linear window without flushing it. That
is task teardown: the GPU VA maps are gone, and writing guest pages from there is precisely the
write-after-free class this project exists to fix. It is also not silent — it emits
`linear_deferred_dropped reason=retired`. This is the `terminal` outcome, which the type-4 section
above already argues is a cost of nothing.

The claimed contrast — "GVA windows *are* flushed at task teardown, so the linear rail is
inconsistent" — is a misreading of one argument. `retire_gva_windows` calls
`flush_gva_one(state, host, *gva, entry, false, "task_retired")`, and that fifth parameter is
`guest_write`. With it `false` the guest write never happens; the call materialises into the cache
only. **Both rails decline to write guest pages at teardown.** The audit read the call site and not
its argument, and it overrode an explicit comment stating the rule.

Net completeness result, which is the useful part: **`flush_intersecting` remains the only site where
a still-live surface's obligation is taken and its pixels lost.** Everything else is teardown
(deliberate and reported) or supersede by a newer window covering the same bytes.

The general lesson, since it will recur: an enumeration is cheap to delegate and its *classifications*
are not trustworthy without reading the arguments at each call site. Take the site list; re-derive the
verdicts.

#### The page-drift guard is armed on 100 % of writes, and until now that was unknowable

`mapw_pages_refused = 0` had two readings and the census printed them identically: a guard that
examined every write and found no drift, or a guard that was never armed for any of them. Those are
opposite claims about the write-after-free class, and `mapping_write`'s own doc hedged in prose —
"the guard here is currently inert; say so rather than counting it as the repair" — because nobody
could tell from the numbers.

The reason is structural. The witness had **five exits and only one of them checked a page**: no
mapping, no latched `type4_walk`, a walk latched at a superseded `map_generation`, and an empty page
list all returned "true" without walking anything. Its own doc said so — "a caller must not read
`true` as 'these pages were verified'" — and both callers then folded that into the same
`PagesVerdict::Ours` a full clean re-walk produces.

Split (`Type4Witness::{Verified, Unwitnessed(why), Drifted}`), the answer is unambiguous. One
420 s driven boot of `blur-provoke.sh` — the backdrop-blur-under-window-capture page from the user's
own crash report, the heaviest surface churn this rig produces — 412 census intervals, counters
summed per the per-interval rule:

```text
                    verified   unwitnessed   refused
direct writers        52 459             0         0     100.00 % verified
deferred flush        52 445             0         0     100.00 % verified

mapw_unwit_no_walk 0   _superseded 0   _no_pages 0   _no_mapping 0
```

So the zero was the good reading. **Every one of ~105 000 mapping-keyed writes across both rails was
re-walked page by page against the guest's own page table before it landed, and not one list had
drifted.** That is a completeness statement about the mapping rails on the provoking workload, and
it is the first time the `refused = 0` figure has meant anything.

Boot health: 1 286 attaches, 0 guest panics, 0 fabricated attaches, 0 drift, 0 lost flushes; 4 type-4
refusals (3 corrected, 1 terminal, **0 stranded**).

Two limits, and the first is the one to respect. **`Unwitnessed` has no live positive control** — it
read 0 across 105 000 writes, and this document's own rule is that a counter which has only ever read
zero has not been shown to work. What stands in for it is a unit test that drives every state and
asserts each slug, so it is not a dead wire; it is not the same as having seen one fire on the rig.
Second, one boot, one workload.

##### The raw-address rail too, and its blind exit was an empty iterator

The paragraph above originally excluded the raw-address rails — `flush_gva_one`, `flush_linear_one`,
guarded by `deferred_pages_still_ours` rather than by the mapping witness. Same lens, and that guard
has the same blind spot in **three** places rather than one:

```text
span == 0 || armed.is_empty()            -> nothing was recorded to compare against
live.iter().all(|p| armed.contains(p))   -> `all` over an EMPTY `live` is TRUE
```

The second is the one worth knowing, because it is the same conflation arrived at by a completely
different route: not an early return, but a **vacuously true iterator**. A window whose walk resolves
no page at all returns "still ours" through the identical branch a fully verified window returns
through. It is harmless — `write_gva_rgba8` resolves its destination per row from that same walk, so
no row lands either — and it is indistinguishable in the census from the guard having compared every
page and agreed.

Split (`defw_pages_verified` / `defw_unwit_no_armed` / `defw_unwit_no_live` / `defw_pages_drifted`)
and measured on a second 420 s `blur-provoke.sh` boot, 412 intervals:

```text
                       verified   unwitnessed   refused
direct writers           53 721             0         0
deferred flush           49 720             0         0
raw-address windows      70 311             0         0     <- the busiest of the three
```

**173 752 guarded writes across all three rails, every one of them re-walked and agreed, none
unwitnessed and none drifted.** Boot health: 1 241 attaches, 0 guest panics, 0 fabricated attaches,
0 lost flushes, 1 type-4 refusal (corrected, 0 stranded).

That is the completeness statement for the device's guest-write guards on the provoking workload,
and it is worth stating what it does and does not close. It closes "is the guard armed?" — which was
genuinely open, and which the `refused = 0` figure could not answer for either rail. It does **not**
close the crash class: a guard that verifies the destination is still the surface's says nothing
about the panics that predate it, and the post-guard panic denominator is still ~150 boots against
the 20 it has.

One fixture note, since it will recur. The first draft of the `defw_unwit_no_live` test picked a GVA
far past the root page's extent, and the walk **resolved** it — reading whatever GPA follows the root
page — so the test failed on the negative it was asserting. Use an index inside the root page whose
PTE was never written. "Far away" is not "unmapped".

Note also what this retires: the recurring suspicion that `ensure_contig_view` hands back a stale
`mach_vm_remap` after a silent re-point. It does not. `resolve_mapping_backing` retires the view and
bumps `map_generation` whenever the resolved plan differs, `ensure_contig_view` revalidates before
returning a cached pointer, and every writer additionally holds a `PagesVouched` token checked after
the flush that could invalidate it. A delegated audit reached the same conclusion but by the wrong
route — it claimed the vouch "re-walks type-4 pages before each write", which is exactly the
overstatement the witness's own doc warns against. The route matters, because the measurement above
is what makes the claim true rather than the argument.

#### "Did we write there?" is now a set lookup, and only 2 of 12 panics can ask it

Every guard above answers *whether a write was allowed*. None of them answers the question the panic
census actually poses, which is **where this device's writes went**. That question was never hard —
it was unasked. XNU's `pmap_page_protect` panic prints a guest **physical page number**
(`pn=0x46b53b`), and this device knew its own destinations only as transient locals, so the link
between the census and this device stayed what this document calls it: a coincidence of shape.

`observe::footprint` is the set that closes it — one bit per guest 4 KiB frame, set by every rail
that can put bytes in guest RAM, accumulated for the whole boot, emitted on the existing per-second
census:

```text
guest_write_footprint pages=34407 kib=137628 dropped=0 frame_shift=12
guest_write_footprint_runs seq=7 part=1/12 runs=561 0x2cd26c-0x2cd50b 0xb9d9-0xb9e8 …
```

Frames are fixed at 4 KiB rather than the device's `page_shift`: it keeps guest page geometry out of
hooks at layers that have no business knowing it, and it is at least as fine as any page this project
supports, so an arm64 16 KiB page marks exactly four frames and nothing is rounded up into a frame no
byte reached.

**Read a hit and a miss differently — they are not symmetric.** A miss is strong: these rails
demonstrably never wrote that frame, which exonerates them. A hit is evidence proportional to
density, because the device is *supposed* to write those frames. A boot covering 0.8 % of a 16 GiB
guest puts an unrelated victim inside about one time in 125; `pages` is on every summary line so a
reader computes that ratio rather than assuming it.

**The coverage limit is the part to internalise, and it was measured rather than assumed.** Of the 12
panics on disk, **only 2 carry a `pn=`** — both `pmap_page_protect`. The other ten give a kernel VA
(the kalloc poison element), a faulting VA (`CR2`), or a backtrace, and none of those can be turned
into a guest physical frame after the fact. So the join covers about a sixth of the class.
`.agents/repros/footprint-attribute.py` reports the rest as `UNSCOREABLE` and says which evidence it
saw and declined to score. **Do not let a scorer fold those into `MISS`** — that would manufacture
ten exonerations with no basis, which is this rig's standing failure direction.

The scorer's own controls are the usual both-directions discipline (`--selftest`, and it passes): a
`pn` inside a run, one on each run boundary, one just outside, one far outside, a VA-only panic, and
a **truncated final dump**. That last one matters because a panic cuts the log mid-dump, and a
partial set has frames missing — missing frames produce false `MISS`es. The scorer falls back to the
newest *complete* dump and flags that the miss is correspondingly weaker. Pointed at a boot that
predates the instrument it prints `UNSCOREABLE`, not `MISS`.

**Completeness is a gate, not a promise.** There are exactly two ways this device reaches guest RAM:
`HostMemory::write_gpa` (one production implementation, marked in `QemuHost`) and a host pointer from
`HostOps::map_pages`. `every_map_pages_caller_is_classified_and_the_writers_mark_the_footprint` pins
all eight production `map_pages` callers with a stated verdict and asserts that every file classified
as a writer marks and every reader does not, so a new writing rail that skips the hook fails the
build. GPU-direct writes are not a gap: `VK_EXT_external_memory_host` is never requested and a
separate gate holds that, so the three `metal_draw/vulkan.rs` sites are read-only upload sources.
`FakeHost` deliberately does **not** mark — fixture addresses in a set whose only use is comparison
against a live guest would be noise that reads as signal.

Two implementation traps worth keeping, both caught by tests rather than by a boot. The run extractor
used `(!word >> lo).trailing_zeros()`, which is 64 when a word is set through bit 63 — a length
measured from bit 0, not from `lo` — so frames 60..=63 dumped as 60..=123, sixty frames never
written. And the scatter form marks **per page, never over a page list's hull**: a surface's pages
are wherever the guest allocator put them, and a hull would claim the guest's other allocations
between them, every one of which then reads as a hit for the rest of the boot.

##### First live reading, mid-drive on `blur-provoke.sh`: density 1.05 %, and no white at all

Read off `/tmp/reims-vgpu-fail.log` while the boot was still driving, so the numbers grow; the
*shapes* are what matter.

```text
footprint  runs=8760  frames=44120 (172.3 MiB)  dropped=0     density 1.052 % of 16 GiB
payload    sampled=25529  all_ff=0 (0.00 %)  all_zero=15557 (60.94 %)  ff_bytes=0/837949604
retire     retired=0  write_after_retire=0  retire_scans=0
```

**The instrument works end to end.** `dropped=0`, 8 760 runs reassembled from 35-part dumps, and
`footprint-attribute.py` parsed the live log without modification. Density **1.05 %** is the number
every later footprint hit is weighed against: an unrelated victim lands inside about **1 time in 95**.

**`all_ff` is 0 in 25 529 samples, and the counter is not dead** — `all_zero` is 15 557 on the same
samples, which is the positive control the census was built with. So on this workload this device
wrote **no all-`0xff` payload at all**, over 838 MB of sampled bytes.

**Do not read that as refuting the white-frame hypothesis.** The blur page is saturated colour bands
under a 12 %-alpha glass; it has no white content, so a device faithfully rendering it *should* never
produce a white buffer. What the run establishes is narrower and still useful: this workload cannot
be the one that produces a `0xff` victim, and the census is live and trustworthy. The hypothesis
needs a workload that actually paints white — a white web page, or Finder, whose icon class is where
Goal 2 lived. Run that before concluding anything in either direction.

**`retire_scans=0`, so `write_after_retire=0` is UNMEASURED, exactly as this section warned.** The
retire path never ran: `unmap_surface` was not reached on this workload. That is the "confirm
`retired=` is non-zero before believing the zero" gate failing on its first outing, and it is a
finding about the detector rather than about the device — the next session must find out whether the
guest never Unmaps here, or whether `unmap_surface` is the wrong retire point, before that detector
means anything.

###### Completed, and both readings above needed correcting

The boot finished at 600 s. `retire_scans=0` held to the end, and the two other numbers moved enough
that the mid-drive figures must not be quoted:

```text
footprint  runs=28960  frames=173034 (675.9 MiB)  dropped=0     density 4.125 % of 16 GiB
payload    writes=11643167  sampled=181925  all_ff=0  all_zero=114064 (62.70 %)
attaches 1567   stores 66182   panics 0   fabricated 0   stranded 0
```

**Density is 4.125 %, not 1.05 %** — an unrelated victim lands inside about **1 time in 24**. A
mid-drive density is a floor, and the set only grows, so read one as a lower bound on how weak a
later hit will be.

**The `retire_scans=0` was structural, not a property of the workload.** `note_mapping_pages_retired`
bailed on an empty `page_entries`, and every route into `unmap_surface` from a guest teardown arrives
with that list *already empty*, because the step before moved it into `condemned_entries`:
`DeleteIOSurfaceBacking2` calls `condemn_surface_backing` first (which does exactly that move), the
second delete then reaches `unmap_surface` through the `mapping_backing_condemned` branch, and the
fall-through branch is reached only *because* `condemn_surface_backing` returned false — which it
does precisely when the list was already empty. `map_surface` moves it the same way. So the detector
could never fire on the delete path in any boot. Fixed in `80f385e` by retiring the condemned list
too, which is sound *there* and would not be at condemn time: at condemn the reprieve can still hand
the list back, and `resolve_mapping_backing` un-retires through `note_pages_authorized` when it does.

Writing the alias control for that fix found a second defect in the same function: a **surviving**
mapping's condemned list was not counted as still-held, so tearing down a mapping that aliased it
retired pages the survivor may yet be reprieved onto. That is a false positive out of the device's
own bookkeeping — the thing that gets a detector switched off before it reports a real finding.

**`all_ff=0` cannot carry the claim it was written up with, and the largest rail was not in the
census.** Both are recorded in the section below; do not cite the paragraph above it.

##### `all_ff` is blind to a white page, and the census was missing the rail that paints

Two defects, either of which alone voids the reading "this device produced no all-`0xff` payload".

**The predicate does not describe the question.** `all_ff` requires the **whole** buffer uniform, and
these rails hand over whole frames and whole source images. A white browser page has a menu bar, a
scrollbar and text on it, so a device faithfully rendering megabytes of white still scores
`all_ff=0`. The panic census implies something narrower and answerable: its two `kalloc` poison
reports found a whole freed element filled with `0xff` **from offset 0**, at `sz:256` and `sz:6144`,
so a write that could have produced the smaller of them put at least **256 consecutive `0xff` bytes**
into guest RAM. That element size is the basis for `FF_RUN_MIN`; it is not a threshold fitted to an
observation. `ff_run` / `ff_run_max` count it and are the columns to read. `all_ff` stays so the
already-recorded boots remain comparable.

The scan probes every 256th byte and expands only on a hit — any run of 256 consecutive bytes
contains a multiple of 256, so nothing long enough to matter hides between probes, and a photograph
costs `len/256` loads.

**`runtime/mapping_write.rs` fed the census nothing at all.** It is the largest guest-write rail in
the device and it reaches `mapper::write_mapping_bytes` not at all — it takes a pointer from
`contig_for_write` and pokes rows into it. This is the **identical shape** to the footprint gap
`2df845f` fixed, one instrument later: that gate closed the *footprint* over these rails and says
nothing about the *payload*. `metal_draw/mod.rs` and `compute_exec/mod.rs` were missing for a related
reason — they write rows through a `FreshSpan`, and `map_fresh_span_within` resolves a span without
ever seeing a buffer, so it can mark on their behalf but cannot sample on their behalf.

The general rule, since this is twice: **a completeness gate covers one instrument, not one rail.**
Adding a second instrument over rails a gate already closed does not inherit that closure.
`every_guest_ram_writer_is_classified_for_the_payload_census` derives its writer set from
`MAP_PAGES_SITES` so the two tables cannot drift, and demands a verdict with a reason for each.
`gpa_map.rs` is the one deliberate `Skipped`: tens of bytes per control-plane write, far more
frequent than a frame, so sampling it would spend the 1-in-64 budget on writes that cannot carry a
256-byte run.

###### Measured: this device does write long `0xff` runs, and only when the guest paints white

The question the payload census was built for, asked directly and answered. `.agents/repros/
white-payload.sh` drives **two pages in one boot** on the same binary — 300 s of a dark page with no
white anywhere, then 300 s of an overwhelmingly white page with chrome and text on it. The counters
are levels, so the difference between the phases is content-attributable, and a census that reported
the same number for both would be measuring something other than what the guest painted.

```text
                sampled     ff_run    rate       ff_run_max     all_ff
dark phase       15 928          2    0.013 %          285           0
white-only       20 690        256    1.238 %        4 961           0
                                        ~99x
```

Three statements, and the third is the one that closes an old question.

**`all_ff` stayed 0 through all 36 618 samples while `ff_run` reached 258.** That is the blindness
argued for in the section above, demonstrated rather than reasoned: the strict predicate reported
"this device produced no all-`0xff` payload" on a boot where it produced 258 of them. Any earlier
`all_ff=0` reading is void, not merely weak.

**The counter tracks the content, by a factor of ~99.** It is not firing on everything, and it is not
dead: the dark page produces 2 in 15 928 and the white page 256 in 20 690, on one binary, minutes
apart. `ff_run_max=4961` is comfortably longer than either `kalloc` element in the panic census, so
runs of the size that could have produced those reports are demonstrably written.

**So "do not go looking for a source of white" is right, and now measured rather than assumed.** This
device writes white when the guest paints white, which is exactly what a faithful device should do.
The `0xff` in the panic census is consistent with this device's own payload, and the defect is
*where* those bytes land, not *what* they are. Note the scale, though: at 0.70 % of sampled writes
over the whole boot, a `0xff` victim sharpens a footprint hit by ~142x on top of a 3.7 % density —
roughly 3 790:1 against coincidence, **if** the two are independent, which nobody has shown.

Boot health: 0 guest panics, 0 firmware aborts, `dropped=0`, density 3.744 %.

##### `write_after_retire` asks the question the drift guard structurally cannot

Every guard in the sections above asks the **guest's page table** whether a mapping's cached list
still resolves the same way. That is the right question and it has a blind spot shaped exactly like
the crash class: a surface the guest has destroyed keeps its translations for as long as the address
space lives, so the walk agrees, `mapping_pages_verdict` returns `Ours`, and the write lands in
memory the guest has already handed to something else. Nothing in the page table changed, so nothing
in that family of checks can see it.

`observe::footprint`'s retired set asks a different question, out of this device's own bookkeeping:
the guest *told* us, in a packet, that those pages stopped being a surface's. A write into one
afterwards is write-after-teardown. It is reported as `retired=` and `write_after_retire=` on the
census line and as a per-frame-latched `write_after_retire` fail line.

**The reason to care about this one specifically is the denominator.** The panic rate is ~2 %, so a
defect that only manifests as a panic needs ~150 boots per arm before a zero means anything — the
soak this document keeps prescribing. This detector fires on the boot the write happens in. It turns
a 150-boot question into a one-boot question, for the subset of the class it can see.

Two ways it could have been useless, both closed, and both are the shape of thing to check before
trusting any new detector here:

- **Aliases.** Mappings genuinely name the same guest pages, so retiring on the dying mapping's list
  alone would mark pages a live surface writes every frame, and the counter would read in the
  thousands on a healthy boot. Frames any other live mapping still names are excluded at retire time.
- **A set that only grows.** The guest recycles physical pages between surfaces constantly. Without
  un-retiring at adoption, every one of those ordinary reuses reads as a defect. Both adoption points
  (`mapper::resolve_mapping_backing`, `objects::apply_type4_backing`) clear the frames they take.

**Only the guest's Unmap retires.** The device's own invalidations — a failed resolve, a condemned
list awaiting its fingerprint compare — are this device deciding it no longer trusts a list, not the
guest saying the memory is no longer a surface's, and the reprieve path can hand the same list
straight back. A detector whose first finding is its own bookkeeping gets switched off before it ever
reports a real one.

**It has now fired on the rig, and the same asymmetry applies to the detection side.** First live
outing after `80f385e` made the retire path reachable, on a `white-payload.sh` boot:

```text
retired=3278   write_after_retire=12432   retire_scans=10   scan_pages=747512
                                          …and exactly ONE fail line
```

`retire_scans=10` clears the gate the paragraph above demanded — the path runs. But **12 432 hits
against one distinct frame is not the shape of a mis-aimed render**; it is a page the guest recycled
being written over and over. The paragraph below argues the raw rails cannot be *retired* because
they have no un-retire event. That argument applies unchanged to a raw rail **writing** into a frame
some mapping retired: nothing announces the page's return to service, so ordinary recycling scores a
hit by construction, for the rest of the boot. The detection side had inherited the blind spot the
retire side was carefully kept out of.

`9edd6fd` splits the counter by `Rail::{Mapping,RawGva,Gpa}` and gives only the mapping rail a fail
line. **Read `war_mapping` and nothing else as a finding**; `war_rawgva` is the blind spot's own
number and is expected to be large. A boot before that split reports `UNATTRIBUTED`, which is what
the 12 432 above is — do not quote it as evidence in either direction.

**Do not extend it to task teardown, and here is the wall you will hit.** The obvious next step is to
retire a dying task's GVA-window pages in `retire_task_gva_windows` / `define_task`, which would put
the raw-address rails under the same detector. It is not sound, for a structural reason rather than
an implementation one: the detector needs an **un-retire event**, and the raw-address rails have
none. A mapping announces adoption, so a recycled page is put back in service by a packet the device
sees. A raw-GVA write resolves its destination through the page table at write time and announces
nothing, so once a task's GPAs are retired there is no event that could clear them — and the guest
reuses those physical pages for other tasks' surfaces within seconds. Every such ordinary reuse would
then score as write-after-teardown, accumulating for the rest of the boot. Using the write's own
resolution as the un-retire event is circular: it clears the bit it is being tested against.

So the mapping rail is the only sound scope for this today. Extending it needs an adoption signal for
the raw rails first, not a wider retire.

##### The completeness gate keyed on one source of the pointer, and missed the largest rail

Worth reading before writing any gate of this shape, because it was green and wrong for one commit
and the thing it was wrong about is the thing it existed to guarantee.

The first cut enumerated every production caller of `HostOps::map_pages`, on the correct reasoning
that a host pointer over guest pages is one of only two ways this device writes guest RAM.
`runtime/mapping_write.rs` calls `map_pages` **zero times**. It takes its pointer from
`mapper::ensure_contig_view` through two local wrappers and pokes BGRA rows straight into the view,
never reaching `mapper::write_mapping_bytes` either. So the gate scored the file as reaching guest
RAM by no mechanism at all — true of the needle, false of the file — and the six raw-pointer row
writes that carry nearly every pixel this device produces were absent from the footprint.

An empty-ish footprint answers "we never wrote there" to every panic it is asked about. The gate
being green is what would have made that believed.

**Key the gate on the capability, not on one source of it.** `GUEST_RAM_POINTER_SOURCES` now lists
every way to obtain a writable alias — `map_pages`, `ensure_contig_view`, `map_fresh_span{,_within}`,
`contig_for_{span,write}` — and the site set went from 4 files to **8**: `mapping_write.rs`,
`metal_draw/mod.rs`, `compute_exec/mod.rs` and `scanout.rs` were all invisible.

Two of the newly-visible files forced a third classification, and it is the interesting one. They
write guest RAM but are marked *by their pointer source* — `gva_view::map_fresh_span_within` marks
the span it resolves, once, for every caller. That is better than marking at each call site, so the
gate cannot simply demand "every writer marks here": it has to distinguish `Here` from `BySource`
from `ReadOnly` and assert all three, because a `BySource` file that also marks locally counts its
frames twice. A two-valued gate would have forced the worse design to satisfy the check.

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

### Goal 4: the staging row-copy is bounded by its caller, not by the copy

Recorded as a **negative** result, because it was written down as a lead and reads like a live bug.

There are two unsafe row-by-row copies into the present staging image, and neither has a bounds check
at the copy site:

```text
backend/vulkan/engine/window_present.rs:1113   dst = mapped + offset + y * row_pitch
host_window/present.rs:1804                    dst = mapped + offset + y * row_pitch
```

Both are nonetheless bounded, by two upstream conditions that have to be read together:

- **The source length is validated.** The engine rail filters on `cpu_frame_complete` at
  `window_present.rs:865` — a real production call site, not only the five unit tests around it. The
  host-window rail does the same check inline in `prepare_frame`
  (`if f.bgra.len() < f.width * f.height * 4` → `BlitSource::Slate`), with the reasoning stated: a
  short buffer is a torn frame, and a slate beats blitting uninitialised memory.
- **The destination is forced to the frame's geometry.** Both rails call `ensure_staging(width,
  height)` first, and it destroys and recreates the image whenever `s.width`/`s.height` differ. So by
  the time the copy runs, `y < height == staging.height`, and `row_pitch >= width * 4` holds by the
  Vulkan linear-image contract.

So a resize cannot make the copy write past the mapped region, and a stale short buffer cannot make
it read past the source. "`y * staging.row_pitch` has no bounds check against the mapped region" is
true of the line and false of the path.

This does not clear Goal 4 — the reported symptoms are a wallpaper that stops working at 4K and a
crash after a couple of resizes, and neither has been reproduced or measured yet. It closes one
candidate mechanism so the next session does not re-chase it. The other standing lead is unexamined:
`host_surfaces` has `remove()` at `surface_cache.rs:257` and `:1156`, so an earlier claim of "no
eviction path" is overstated, but there is **no size cap**, and a 4K surface is ~4x the bytes of a
1080p one.

#### The resize direction is guest-to-host, which is why there is no repro yet

Worth knowing before anyone tries to build one. **The guest drives the geometry and the host window
follows**, not the other way round: the guest presents at a new size, `guest_resize_request`
(`host_window/present.rs:151`) notices the incoming geometry differs from both the last observed one
and the window's, and asks for a matching native resize, held as a `PendingGuestResize` until a
`Resized` event confirms it (bounded by `GUEST_RESIZE_WARN_AFTER`, after which it drops
fail-visibly and presents letterboxed). The native-resize half is `#[cfg(target_os = "macos")]`.

So resizing the *host* window is not a way to drive this, and there is no QMP or device-side knob
either — the resolution has to be changed **inside the guest**. That is the whole reason no Goal 4
repro exists: macOS ships no built-in CLI for display mode, `osascript` is TCC-rejected on this rig
(recorded above), and the Displays pane needs GUI automation. `displayplacer` (Homebrew) is the
obvious candidate and the guest does have network, but nothing has tried it.

Do not start Goal 4 by reading present code. Start by getting the guest to change resolution at all,
and confirm the rig can observe the change — a repro that cannot reach the state reports a clean
device, which is the failure this document already paid three `none` results for.

#### The guest can now be driven to 4K, and x86/Linux does not reproduce Goal 4

`.agents/repros/guest-display-mode.m` is ~60 lines of CoreGraphics compiled **in the guest**
(`/usr/bin/clang` and Command Line Tools are present, and so is network). No third-party binary and
nothing to install:

```sh
scp .agents/repros/guest-display-mode.m macos-vm:/tmp/
ssh macos-vm 'cd /tmp && clang -framework CoreGraphics -framework Foundation -o guest-display-mode guest-display-mode.m'
ssh macos-vm '/tmp/guest-display-mode list'        # 4 distinct geometries, incl. 3840x2160
ssh macos-vm '/tmp/guest-display-mode set 3840 2160'
```

It prints the mode that *actually took* rather than the one requested, because a mode change that
silently does not apply is the failure direction that reads as a pass — the repro would drive a
resize that never happened and score the device clean.

**Measured, and it is a negative:** 22 mode changes on one x86/Vulkan boot — 12 on an idle desktop,
then 10 more with Safari on a Wikipedia article — alternating 1920×1080 and 3840×2160. QEMU survived
all 22, the 4K desktop screenshots correctly with the wallpaper intact, and the fail log carries no
`deferred_flush_lost`, no `mapping_page_drift`, and no staging decline (`no_source`,
`upload_no_staging`, `staging_failed`).

**The structural reason matters more than the count.** `request_guest_geometry`
(`host_window/present.rs:1040`) is `#[cfg(target_os = "macos")]`, and it emitted **0**
`host_window_guest_resize` lines across all 22 changes. So on a Linux host the window never follows
the guest: the guest renders 3840×2160 and the device scales it into the 1921×1079 window it already
had. The swapchain recreation, the `PendingGuestResize` hold, and the
`GUEST_RESIZE_WARN_AFTER` drop-to-letterbox path — the machinery a resize crash would most likely
live in — **never execute on x86/Linux at all**.

That scopes Goal 4 sharply, and it is the kind of pathway-specific fact this document's support
matrix exists to force. Either the defect is **arm64/macOS-only**, in which case it cannot be
reproduced or fixed on this host and needs an Apple host; or it is in guest-side 4K rendering, which
x86 does exercise and which did not fail here. Do not read 22 clean resizes as "Goal 4 is fixed" —
read them as "the x86 arm does not carry it", and note that the user's own crash report is macOS
13.7.8 on `iMac19,1`, which is the x86 guest.

##### The macOS gate was incidental, and lifting it made the rail testable here

Superseded in part by `a6fff65`: nothing in that machinery is AppKit-specific. `request_inner_size`
is cross-platform winit, `Resized` was already handled on both rails, and the two tests covering this
code (`engine_present_gate_*`, `guest_geometry_change_*`) were themselves `#[cfg(target_os =
"macos")]` — so they had never run on the rig the feature matrix actually exercises. Un-gated, the
window follows the guest on Linux and **the swapchain recreation, the `PendingGuestResize` hold and
the `GUEST_RESIZE_WARN_AFTER` path all execute on x86 for the first time.**

The gate had been hiding a live defect on the *other* side of the same invariant. `viewport.rs` says
presentation and pointer translation must move as one unit, and records the rolled-back experiment
that broke it. The engine presenter's `aspect_fit`
(`backend/vulkan/engine/window_present.rs:940`) carries **no `cfg` at all** and its own comment
asserts "the window input path maps pointer positions through this same transform" — while that path
was macOS-only. So on Linux the presenter letterboxed and the pointer reported raw window
coordinates against the window extent, and the consumer scales `x` against the reported `width`
(`runtime/input.rs`: "min_in = 0, max_in = dim"), so both the offset and the ratio were wrong.

Not hypothetical: **the guest offers 1440x1080 and 1280x1024**, so a 4:3 mode in a 16:9 window
pillarboxes 240 px either side and the viewport's left edge reported as `x=240 of 1920` where the
truth is `x=0 of 1440`. Anyone reaching for the mouse on this rig was hitting it.

The window's own `VkState` fallback **stretches** to fill rather than aspect-fitting, and is not
driven by `draw_engine_window`, so it leaves `guest_extent` `None` and keeps full-window mapping —
which is the transform its blit actually implies. That agreement was accidental; it is now stated at
`request_guest_geometry`. Do not "fix" that rail by calling `request_guest_geometry` from it without
also making it aspect-fit, or you pair a viewport-mapped pointer with a stretched picture — the same
split, mirrored.

##### An adjusted resize is an answer, and 75 % of them read as refusals

The first Linux boot of that rail immediately found the next one (`94e44d3`).
`note_guest_resize_applied` cleared the pending request only on a **byte-exact** match, and this
host does not answer exactly:

```text
requested 1920x1080 -> 1921x1079      requested 3840x2160 -> 3840x2160
requested 1440x1080 -> 1440x1079      requested 1280x1024 -> 1281x1024
```

Never more than a pixel, never in a fixed direction. Over 53 resizes: 13 exact, **40 logged
`native_resize_not_applied`** — a fail-visible line saying the resize had not happened, about a
window that had resized. And because `draw_engine_window` returns early while a request is
outstanding, each of those 40 **held all presentation for the full second** the alarm takes. Three
quarters of guest mode changes froze the display for a second and then claimed refusal. The hold's
comment ("normally single-digit milliseconds") was a macOS observation generalised to a rail that had
never run anywhere else.

Any `Resized` now settles the hold (`status=applied` exact, `status=adjusted` otherwise). The alarm
keeps the case it was written for: a window system that ignores the request emits no `Resized` at
all, so nothing settles it. **A compositor is free to adjust a request; only silence is a refusal.**

Verified live, two boots one binary apart, same 12-cycle drive:

```text
                             requested  applied  adjusted  native_resize_not_applied
ctl (exact match only)          53         13        —              40
arm (any Resized settles)       49         12       37               0
```

Every false failure is gone and reappears as `adjusted` at the same ~75 % rate, so the counter did
not merely stop counting — the outcome was reclassified, and each of those no longer stalls a second
of presentation. The request totals differ (49 vs 53) only because the control boot had four
exploratory mode sets before its cycles; they are not matched arms and the ratio is the comparable
quantity, not the count.

##### Measured: 53 resizes with the machinery live, and Goal 4 still does not reproduce on x86

One boot, x86/Vulkan, guest macOS 13.7.8, 12 full cycles across 3840x2160 / 1440x1080 / 1920x1080 /
1280x1024 — with the window genuinely resizing and the swapchain genuinely being recreated, which the
22-resize run above could not do:

```text
resizes 53      qemu alive      guest panics 0      firmware aborts 0
deferred_flush_lost 0   mapping_page_drift 0   rendflush_page_drift 0
rendflush_gen_drift 0   staging_failed 0       no_source 0
```

The 4K desktop screenshots correctly, wallpaper intact, full-frame with no letterbox. So **neither
Goal 4 symptom reproduces on x86/Linux even now that the resize machinery executes on it.** That
strengthens the arm64/macOS-only reading rather than settling it, and it is still one boot: a crash
"after a couple resizes" that survives 53 is either not on this pathway or needs a provocation nobody
has found. Drive the guest with `.agents/repros/guest-display-mode.m` (compiled in the guest; it
prints the mode that actually took, because a mode change that silently does not apply is the failure
direction that reads as a pass).

### "No qemu process" means "not started yet" for the first few minutes of a boot

`vm/boot-x86.sh` **rebuilds QEMU before it launches it**, so there is a multi-minute window at the
head of every boot in which no `qemu-system-x86_64` exists and the boot is perfectly healthy. A wait
loop written the obvious way:

```sh
until ssh …; do pgrep -f '[q]emu-system-x86_64' >/dev/null || { echo "QEMU GONE"; exit 1; }; sleep 10; done
```

fires its bail-out on the first poll, every time, and reports a dead VM for a boot that goes on to
run fine. Measured: it declared `QEMU GONE` while the guest was in OpenCore and the serial log was
still growing.

It is the same shape as the `Retired` trap — **a check whose negative is indistinguishable from its
positive** — with the sign flipped: here the harness reports failure where there is none, which is
the cheaper direction but still costs a diagnosis. Either poll for the *success* condition alone with
a bounded attempt count, or latch "qemu was seen alive" before treating its absence as death.

### An abandoned boot keeps port 2222, and the next run scores the guest it left behind

The mirror of the section above, and far more expensive, because this one reads as a *pass*.

Every repro here reaches the guest through `hostfwd=tcp::2222-:22` and waits for ssh to answer.
Interrupting a driver script in the foreground does not take its VM with it: `boot-x86.sh` is
`nohup`'d, so QEMU survives, keeps the guest running, and keeps listening on 2222. The next run then
gets ssh on its **first** poll — it prints `guest is up after 0s`, which is not a plausible boot time
and is exactly what a healthy fast start would look like at a glance — and proceeds to drive and
score **the previous run's guest**, on the previous run's binary, with the new arm's name on the row.

That is worse than a wedge. A wedge produces no data; this produces a full, well-formed row that
attributes one binary's behaviour to another. Everything downstream of it is arm-swapped, and nothing
in the output says so.

Before launching any boot, sweep and check the port:

```sh
pkill -9 -f '[q]emu-system-x86_64 -enable-kvm'; pkill -9 -f '[b]oot-x86.sh'
ss -ltn | grep :2222 && echo "STALE GUEST — do not launch"
```

Bracket a character in those patterns for the reason the `pgrep` section above gives, and treat
`guest is up after 0s` as a harness fault rather than good luck: the boot script rebuilds QEMU first,
so the floor on an honest boot is minutes, not seconds.

### `host_cache_levels` is a level, and `store_routes` is a rate

The two census lines in this device now use **opposite conventions**, so read the line's own text
before doing arithmetic on it. `store_routes` fields are counts for one interval and must be summed
across lines; `host_cache_levels` fields are the size *right now* and must not be. Summing a steady
cache across a 2 900 s boot multiplies it by the census cadence and reports a leak that is not there
— the same shape of error, in the other direction, as the one that produced `mapping_pages_ours 310`
for a true 25 646.

The line labels itself `(levels, not per-interval)` for exactly this reason. It carries per-cache
`entries` / `bytes` / `largest`, plus a device-global `peak_bytes`:

- **`largest`** separates "many small surfaces" from "a few 4K ones". A 4K entry is ~4x a 1080p one,
  so entry *count* alone cannot tell a benign map from an expensive one.
- **`peak_bytes`** is there because the last line cannot show a transient spike, and a spike is what
  a resolution change produces — every geometry change orphans the previous geometry's entries until
  something replaces or evicts them.

`bytes` sums `Arc<Vec<u8>>` lengths and a deferred render window can share an entry's allocation, so
it is the size of the pixels *reachable through the cache*, not memory additional to the windows.
Do not add it to a window figure and call the total resident.

Do not measure this class from QEMU's RSS. The first attempt did and read **9.15 GB** — a number
dominated by however much of the 16 GB guest RAM the guest had touched, which moves for reasons that
have nothing to do with these maps.

#### Measured on the first boot that read it: `host_gva_surfaces` does not stop growing

x86/Vulkan, 60 guest-driven resolution changes cycling 3840x2160 / 1440x1080 / 1280x1024 /
1920x1080. Sampled every twelfth census line:

```text
   t_ms  total_MB   surf    gva   lin
  33551      83.2     14     26     6      <- idle 1080p desktop
  70084     151.7     11     63    39
  94485     293.4     11    152    52
 131151     191.2     13    223    51
 168164     228.3     13    269    51
 205210     358.0     16    312    50
 242023     284.0     15    330    50
 266627     291.0     11    354    51      <- drive ends
 278645     291.0     11    354    51      <- steady after
```

Read the three caches separately, because only one of them is the finding.

**`surf` and `lin` are bounded.** `host_surfaces` sits at 11-16 for the whole boot and
`host_linear_textures` rises once to ~51 and stays. Both churn and neither accumulates.

**`gva` is strictly monotonic — 26 to 354, and it never decreases once, in any of the 27 census
lines.** That is not churn with a high water mark; it is a map with no eviction on this workload,
growing about 5.5 entries per resolution change and holding them. `total_bytes` is noisier than the
count because entry sizes differ (`surface_largest` reads exactly 8 294 400 = 1920x1080x4), but it
went from 75 MB to 291 MB with a `peak_bytes` of 440 MB, and it flatlines the moment the drive stops
— so the growth tracks the resizes, not the clock.

The mechanism follows from the key. `host_gva_surfaces` is a `BTreeMap<u64, HostSurface>` keyed by
guest **virtual** address, and a store does `.entry(gva).or_default()` — so a *new geometry at the
same GVA replaces* an entry and costs nothing. Growth is therefore entirely from **new GVAs**: each
mode change has the guest allocate its surfaces somewhere new, and every address it abandons keeps
its entry forever.

The retention is deliberate and is the thing to be careful of. `GvaHostView`'s doc records that this
cache is "retained across Unmap (wallpaper class)" — the guest unmaps the wallpaper surface and
samples it again later, so an eviction on Unmap would wipe content that is still wanted. **Do not
"fix" this by dropping entries on Unmap.** This document already records a preserving variant of a
neighbouring rail that scored 0 of 14 rounds clean with the screen black at 19 Hz; the same class of
regression is available here.

What makes an eviction rule defensible is that it must only drop entries a lookup could never serve.
`get_gva_with_gen` serves on **(gva, exact width, exact height)**, and `GvaBacking` records the
`task_id` whose page table produced the pixels — an entry whose task is gone can never be matched
again, because the GVA is only a name in that page table. That is the candidate, and it is
**unverified**: nobody has shown that the 328 accumulated entries belong to dead tasks, and on this
workload WindowServer plausibly survives every resize, in which case task-death eviction reclaims
nothing and the rule has to key on something else.

So: the leak is measured, the mechanism is understood, and **the fix is not chosen yet.** Whatever it
is, gate it and A/B it on this counter — `host_cache_levels` is now the instrument for exactly that,
and the 60-resize drive above is a repro that moves it by 4x in four minutes.

##### Killed by measurement: the entries are not dead-task entries

The task-death candidate above was the obvious rule and it is **wrong**, which is why
`gva_cache_staleness` was written before the patch rather than after. A second boot, same 60-resize
drive, reading the new columns:

```text
round  0 (idle)   gva= 28   gva_dead_task=0   gva_no_backing=0
round  5          gva=227   gva_dead_task=0   gva_no_backing=0
round 10          gva=293   gva_dead_task=0   gva_no_backing=0
round 15          gva=331   gva_dead_task=0   gva_no_backing=0
```

**Every one of the 331 accumulated entries is backed by a task that is still active, and not one is
unbacked.** Task-death eviction would have reclaimed exactly nothing while reading like a fix. The
compositor survives every resolution change and simply keeps allocating new virtual addresses; the
abandoned ones are live-task entries that nothing will ever ask for again.

Two facts from the same boot worth carrying:

- `gva_largest` is **33 423 360 = 3840 x 2176 x 4** — a 4K entry with its height padded to a multiple
  of 64, so it costs **4x** the 8 294 400 of a 1080p one. Entry *count* alone understates what a
  4K-heavy session holds, which is why the gauge reports `largest`.
- `gva_backing_bytes` reached 305 952 — the page lists are ~0.2 % of the pixel bytes, so they are
  real but not where the memory is. Do not spend a rule on them.

There is also **no memory-reclaim eviction path at all**: `evict_gva` has exactly one product caller
(`metal_draw/vulkan.rs`, on the deferred-Store re-arm), and it evicts *the same key* being replaced,
for correctness — stale encodes must not serve. Nothing anywhere walks the map to drop an abandoned
address. Note the neighbouring deferred-window map *is* bounded, by `GVA_DEFERRED_WINDOW_CAP`; the
encode cache never got the equivalent.

##### Sized: the backing rule reclaims about half, and `unmapped` is the wallpaper

`gva_backing_moved` (`cec389b`) walks the first recorded GPA of each entry and compares it against a
fresh walk of the key. Third boot, same 60-resize drive:

```text
                gva   moved   unmapped   valid
idle             27       2         14      11
round  5        218      68        135      15
round 10        272     107        151      14
round 15        305     149        143      13
after drive     305     101        194      10
```

Three things, and the third is the one that decides the implementation.

**The working set is ~13 entries.** It is 11 at idle and still 13 after 60 resolution changes, while
the map goes to 305. Roughly 95 % of this cache is dead weight, which is the leak restated per-entry
rather than in bytes.

**`unmapped` must never authorise an eviction, and the idle row proves it.** At idle, before any
resize, **14 of 27 entries are already unmapped** — this cache is *deliberately* retained across
Unmap for the wallpaper class, so "the guest unmapped this VA" is the normal state of exactly the
content the cache exists to keep. A rule that collapsed `moved` and `unmapped` would evict 16 of 27
entries on an idle desktop and wipe the wallpaper. That is the black-screen regression this document
already records, reachable in one line of code, and separating the two columns is the only reason it
is visible here instead of after a boot.

**`moved` is not monotonic, so it is a statement about an instant, not a verdict.** It fell from 149
to 101 between the last two rows while `unmapped` rose — the guest re-points these addresses
continuously, and an entry can move away and come back. So an eviction on `moved` can still lose a
churn-and-return case, which is precisely the case `GvaBacking` was added to serve
(`gva_backing_separates_a_churned_mapping_from_a_reassigned_address`: a mapping that churned and came
back reads `Confirmed`). **The `moved` rule is well-sized at ~half the entries and it is NOT
provably free** — it needs a gated A/B against the wallpaper, not an assumption of safety.

The honest state: the leak is measured, two candidate rules have been sized, one is dead
(`gva_dead_task=0`) and one is half a fix with a real risk attached. Nothing has been evicted yet, on
purpose — every step here was cheaper than the boot that would have found the regression.

Not a crash, on this boot: 60 resizes, QEMU alive, 0 guest panics, 0 firmware aborts, 0 losses on
every rail. 291 MB is not an out-of-memory. The reason this is written up under Goal 4 anyway is that
"after a couple resizes the whole thing just crashing" wants an unbounded resource, and this is the
first one anyone has found — but a session long enough to make 291 MB into an OOM has not been run,
so **the link to the reported crash is a hypothesis and not a measurement.**

##### Fixed by recency, because neither staleness rule was available

Both rules the sections above sized are dead, and the third option was never a staleness rule at
all. `host_gva_surfaces` is now bounded by `GVA_ENCODE_CACHE_BYTE_CAP` (128 MiB, the same value and
the same basis as `LINEAR_SAMPLED_MEMO_BYTE_CAP`, which bounds the sibling cache holding the same
content) with least-recently-**used** eviction.

The reason recency is admissible where staleness is not is the one property both other rules lack,
and `LruBytesMemo`'s own header already named it: an entry read every frame but never rewritten — a
wallpaper plane — is touched on every hit, so it is the **hottest** thing in the map and can never
be the victim. Eviction reaches only entries nothing has looked at. That makes this a resource
bound rather than a guess about what the guest still wants, which is the bar this project holds
itself to.

The touch is the whole mechanism, so it is wired at the three product read paths
(`seed_color_load`, the sampled read and the Load seed in `metal_draw`), charged on a **confirmed
serve** and not on an attempted one.

Measured, one interleaved pair, one binary apart (`REIMS_VGPU_GVA_CACHE_CAP_OFF=1` is the only
difference), 60 guest-driven mode changes each, `.agents/repros/gva-cache-cap-ab.sh`:

```text
              resizes  census  gva_max  max gva_bytes   evicted  wanted  forgotten  drift
arm (capped)     60      349      209    114 809 056      257       0        0        0
ctl (uncapped)   60      351      277    153 905 896        0       –        –        0

gva trajectory, every 40th census line
  arm    0 113 176 140 173 131 167 136 133      <- oscillates under the bound
  ctl    0 110 156 182 197 234 255 261 273      <- strictly climbing, never once down
```

Four statements, and the third is the one that makes this a fix rather than a trade.

**The bound holds.** The arm never exceeded 109.5 MiB against a 112 MiB low-water mark, at any of
349 samples. The control ended at 146.8 MiB — past the 128 MiB cap — and its entry count was still
strictly monotonic at the last sample, exactly as the uncapped 60-resize boot before it was.

**The cap engaged, so the arm is measured.** 257 evictions. An arm reporting `evicted=0` would be a
cap that never fired, which is *not* the same claim as a cap that fired safely, and the harness
reports it as `UNMEASURED` for the cost question for that reason.

**It cost nothing measurable.** `gva_cap_wanted` — lookups that missed on an identity the cap had
evicted, which is precisely the harm and nothing else — is **0 against those 257 evictions**, and
`forgotten` is 0, so that zero is exact rather than a lower bound. The wallpaper is intact in both
arms by eye and by colour: mean RGB 0.681/0.246/0.109 against 0.691/0.248/0.107.

**The running total is exact.** `gva_cap_drift` (the total the cap tests against, minus the real sum
the census computes anyway for `gva_bytes`) was **0 on all 700 census lines across both boots**. The
cap reads a running total rather than re-summing the map on every store, because enforcement runs on
the store path; a second source of truth is how a bound silently stops bounding, so it reports its
own divergence.

**Scope it.** One pair, one workload — resize churn on an otherwise idle desktop. It is not a rate,
it does not say no workload can be harmed, and it does not touch the reported crash: 291 MB was
never shown to be an OOM and this makes it not happen rather than proving it was the cause.

Two traps worth keeping. `DESKTOP_MEAN_FLOOR` (0.70) is **mis-calibrated for this sequence** — both
arms scored `DARK` at ~0.665 on captures whose wallpaper is plainly correct, because the capture is
a 1280x719 letterboxed host window and the floor was set for a different one. The arm-versus-control
comparison is what carries the wallpaper claim; the absolute floor would have read as a regression
in both arms. And enforcement runs *after* the insert, so it must be told which address the
triggering store just wrote or a single entry over the low-water mark evicts itself and is never
cached at all — reachable in production, since `MAX_SCANOUT_DIM` is 8192 and admits a 256 MiB entry
against a 112 MiB mark.

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
