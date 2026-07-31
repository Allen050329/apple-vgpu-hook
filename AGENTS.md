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
