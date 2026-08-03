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

## What The Guest Driver Puts Out Of Reach

Some capabilities cannot be delivered from the host at all, because they are decided by constants
inside Apple's guest-side driver and userland plugin. Record them here rather than rediscovering
them: each of these has already cost a session.

**Chrome's GPU blocker is one capability: Tier 2 argument buffers.** ANGLE's Metal backend gates
display creation on `[MTLCreateSystemDefaultDevice() supportsFamily:MTLGPUFamilyMac2]`, and the
guest answers NO. The check runs *before* the EGL display object exists, so no ANGLE feature
override, EGL attribute or Chrome flag can reach it, and it returns `EGL_NO_DISPLAY` with
`EGL_SUCCESS` and no message — which is why it reads as an unexplained failure. Chrome flag
combinations are therefore not worth another session; the gate is a device capability.

Which capability is now pinned down. Measured on the live guest: `supportsFeatureSet` is **true**
for `macOS_GPUFamily1_v1/v2/v3/v4` and false only for `macOS_GPUFamily2_v1`;
`argumentBuffersSupport` is **Tier 1**; `supportsFamily` is Mac1 yes, Mac2 no, Common1/2 yes,
Common3 no.

**This is settled, and it is settled against us on the x86 pathway.** Two constants decide it,
both of them compile-time immediates with no inputs:

- Metal.framework's `-[_MTLDevice indirectArgumentBufferCapabilities]` returns a **literal 0**, and
  `-[_MTLDevice argumentBuffersSupport]` reports Tier 2 only when three low bits of that value are
  all set. Apple's paravirt plugin overrides neither, nor `supportsFamily:`, nor
  `supportsFeatureSet:`, so a base-class literal answers. **Tier 1 is decided inside Metal itself**,
  not by the plugin, not by the kext, and not by anything the host sends.
- `-[AppleParavirtDevice featureProfile]` returns a **literal 10000**, and Metal's
  `initGPUFamilySupport` switches on exactly that value to build the family vector
  `supportsFamily:` linearly searches: 10000 pushes Common1/Common2/**Mac1**, and only 10001 pushes
  **Mac2**.

**Two earlier claims in this file were wrong in opposite directions. Do not restore either.**

The first said `featureProfile = 10000` is hardcoded and host-independent. That was **right**.
The second "refuted" it on the grounds that a device pinned at 10000 could not report
`macOS_GPUFamily1_v4` true. That refutation is **unsound**: feature sets 10000, 10001, 10003 and
10004 — `macOS_GPUFamily1_v1` through `v4` — all map onto the *same* `MTLGPUFamilyMac1`, and only
`macOS_GPUFamily2_v1` (10005) needs Mac2. A device at profile 10000 reports v1–v4 true and v2_v1
false, which is precisely what was measured. The measurement never contradicted the constant.

**The rung ladder is not a route to Tier 2.** Rungs 43 and 60 move `metalHeaps` and
`bufferFromIOSurface` in the guest's feature struct, which the plugin reads for `supportsHeaps` and
`supportsBufferWithIOSurface`. Neither `featureProfile` nor `indirectArgumentBufferCapabilities`
ever reads that struct. Raising the rung cannot raise the tier; drop that lead.

**The live lead is the pathway, not the protocol.** The x86/PCI personality and the arm64 one load
*different* Metal plugin bundles, named by `MetalPluginName` in each kext personality, and the two
bundles carry different `featureProfile` immediates: the PCI one **10000** (Mac1), the arm64
IOGPUFamily one **10001** (Mac2). If that holds, `supportsFamily:MTLGPUFamilyMac2` is **true** on
the arm64 pathway and ANGLE's gate passes, so **Chrome's GPU acceleration is an x86-pathway defect,
not a paravirtualization one**. That is a concrete falsifiable prediction and it is untested — no
one has run Chrome on an arm64 guest here. Test it before building anything on it, and note that
the arm64 plugin subclasses a different device base class, so "the bytes say 10001" is not yet
"Chrome works".

Raising the rung remains mechanically available for its own sake: the guest writes a fixed 4 to
`GFX_REG_VERSION` and then switches on the value it **reads back**, so the host decides the
effective rung. Apple's host only ever clamps down, so doing so is out of contract — and on the
x86 plugin heaps are hard-disabled at source regardless (`newHeapWithDescriptor:` returns nil).

**Hardware OpenGL does not exist in the guest on the x86 pathway.** The PCI personality's Metal
plugin returns `supportsOpenGL = 0` from an unconditional literal, and the kext's `Info.plist`
declares no `IOGLBundleName`, so the guest's only CGL renderer is the software one. Anything whose
acceleration route is OpenGL — Firefox's compositor among them — has no hardware path there without
shipping a guest driver, which this project does not do.

**This one is now confirmed live, from Firefox's own words, and it settles three goals at once.**
Firefox 153 on the guest (macOS 13.7.8, x86/PCI pathway), launched with `MOZ_LOG='WebGL:5,gfx:5'`,
prints exactly three graphics failures and nothing else:

```text
[GFX1-]: Failed GL context creation for WebRender: 0x0
[GFX1-]: Failed to connect WebRenderBridgeChild. isParent=true
[GFX1-]: Fallback WR to SW-WR
```

Firefox's compositor on macOS is WebRender over a **CGL** context; there is no Metal WebRender
backend on that platform. No context means SW-WR, and SW-WR means the same probe that reports
Safari at `WebGL 1.0` + `WebGL 2.0` on `Apple GPU` reports Firefox at `webgl1: NOT_AVAILABLE`,
`webgl2: NOT_AVAILABLE`. So on x86:

- **Firefox GPU acceleration** is the `supportsOpenGL = 0` literal, not a device defect.
- **Firefox WebGL** is the same literal — WebGL needs the GL context that failed to create.
- **Jumpy video in Firefox** is downstream of it: the whole compositor is on the CPU.

Do not spend a session on Firefox flags, `gfx.webrender.*` prefs, or `MOZ_*` overrides. None of
them can supply a renderer the guest does not have. Firefox's rAF still reads ~123 fps because
SW-WR paces fine on an idle page; frame rate is not the symptom to measure here, and a good rAF
number from Firefox says nothing about whether the GPU is involved.

**Safari WebGPU is not reachable on this guest either, and it is not our doing.** Safari 16.6 on
macOS 13.7.8 exposes no `navigator.gpu`, and it stays absent after setting every spelling of the
experimental flag (`WebGPUEnabled`, `ExperimentalWebGPUEnabled`, `WebKitWebGPUEnabled`,
`WebKitExperimentalWebGPUEnabled`). WebGPU is not compiled into that release. Any WebGPU goal on
this guest image is blocked on the guest's browser versions, not on the device.

Narrow this claim to x86 deliberately: in the arm64 IOGPUFamily bundle the same selector is **not**
a literal — it forwards to the serializer's own `supportsOpenGL`. Whether that ever answers yes is
unknown and untested. Same shape as the `featureProfile` split above, and the same warning: this
was read out of the binaries statically and has not been confirmed on a running arm64 guest.

Safari is unaffected: it uses Metal directly rather than through ANGLE, and Mac1 is sufficient for
it. Treat Safari as the browser where browser-facing GPU goals are actually measurable.

**There is no hardware video decode in the guest, for any codec, and this device is not the reason
— it is not a codec device at all.** Measured on the live x86/PCI guest (macOS 13.7.8):

```text
VTIsHardwareDecodeSupported(H264) False   (HEVC, VP9, AV1 all False)
kextstat  | grep -iE 'avd|videotoolbox|codec|h264'   -> nothing
ioreg -l  | grep -iE 'AppleAVD|AppleVXD|IOVideoDecode|AppleH264'  -> nothing
```

`system_profiler` reports the GPU normally beside it (`Apple Paravirtualized Graphics Device`,
`Metal 2`, 1920x1080 @ 120 Hz), so this is not a device that failed to attach — VideoToolbox has no
codec hardware to bind to, because nothing in this stack exposes one. Our QEMU device is a GPU: no
codec BAR, no codec protocol, no VideoToolbox forwarding. There is nothing on the host side that
could answer even if the guest asked.

Two things follow. **Goal 4 (ffmpeg GPU acceleration) is blocked on a device that does not exist**,
not on a defect in this one — and note ffmpeg is not installed in the guest image either, so the
symptom was never reproducible there in the first place. And **every browser's video decode runs on
the CPU**, which is a second and independent cause of jumpy playback alongside Firefox's
`supportsOpenGL = 0` compositor fallback. Unlike that one it applies to Safari too, so "Safari is
the browser where GPU goals are measurable" does not extend to video.

**That paragraph used to end by saying hardware video here would need a new device *and a guest
driver for it, which this project does not ship*. The second half is wrong, and it is the half that
made goal 4 look impossible.** Apple ships the guest driver, and it is installed on this image right
now. The check the paragraph asked for has since been run, and Apple's paravirt stack does have a
codec path:

- `AppleVideoToolboxParavirtualization.kext` is present in the guest under
  `/System/Library/Extensions/`. It is a *separate* kext from the GPU one.
- Its driver personality is `AppleVideoToolboxParavirtualizationDriver` over provider
  `AppleVirtIOTransport`, with `IOUserClientClass` `AppleVideoToolboxParavirtualizationUserClient`,
  matching `IOVirtIOPrimaryMatch = 0x1a03106b`. A second personality binds `AppleVirtIOPCITransport`
  to `IOPCIDevice` on `IOPCIPrimaryMatch = 0x1a03106b`, so the whole path hangs off **one PCI ID:
  vendor 0x106b, device 0x1a03**.
- It declares dependencies on `AppleParavirtIOSurface` and `IOSurface`, so decoded frames are meant
  to land in IOSurfaces shared with the graphics stack rather than being copied out.
- It is **not** attached to the GPU's PCI ID or to either GPU personality, and it is not part of the
  x86-vs-arm64 plugin split that decides `featureProfile`. It is its own VirtIO device on both
  pathways, so unlike Chrome's Tier 2 gate this one is not an x86-only defect.

Confirmed on the live x86 guest, and this is the part that matters: the kext is **installed and not
loaded** — `kmutil showloaded` matches it zero times. It never matches because nothing on our bus
presents `106b:1a03`. So the reason `VTIsHardwareDecodeSupported` answers False for every codec is
not that the guest lacks a driver. It is that **the host does not expose the device that driver
binds to**, which is a host-side gap of exactly the kind this project already fills for the GPU.

Goal 4 is therefore **not** blocked on "a device that does not exist". It is blocked on a device
nobody has written. Keep the scope of that honest, because it is a long way from here to working
decode:

- **Established**: the attach contract above — the ID, the transport, the user client, the IOSurface
  dependency, and that the driver is present and idle on a running guest.
- **Not established**: the wire protocol. Nothing has decoded the VirtIO queue layout, the command
  set, codec negotiation, or the surface handoff, and none of that can be guessed. Nor has it been
  shown that a host-side decoder (VideoToolbox on macOS hosts, VAAPI or Vulkan Video on Linux) can
  satisfy whatever that protocol asks for.
- Presenting `106b:1a03` and then answering nothing would make the guest *load* a driver onto a
  device that does not work, which is worse than the current state. Do not present the ID until
  there is something behind it.

The measurements above still stand as a description of the symptom: with no such device, every
browser's video decode is on the CPU, and that remains a second, independent cause of jumpy playback
which also applies to Safari.

Verify claims of this kind against the binaries before adding one, and say which constant decides
it. "The guest cannot do X" is exactly the sort of broad claim `Keep Claims Narrow` is about.

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

**Host-pointer imports are not windowed any more — they are forbidden.** `VK_EXT_external_memory_host`
must never be asked for. Importing a host pointer over guest RAM gives the host GPU write access to
the guest VM's memory, and that is a property of the mechanism rather than of how much of it is
used, so the bound is "never requested" and not any budget. The resolver, its window budget, the
scatter pool and both present entry points are all deleted, and
`observe::gate::the_host_pointer_import_extension_is_never_requested` fails the build for source
naming any of the six API surfaces back into existence.

This paragraph used to read "keep imports windowed; use the existing capped window resolver", which
sent a reader after a mechanism that no longer exists to solve a problem the gate forbids solving
that way. It is worth knowing why the temptation recurs: the render deferred-flush rail moves a
gigabyte a second into guest pages, and that is the single largest cost in the device. It is still
not the route. Reducing the bytes, or making the guest's own reads observable so the writeback
becomes demand-driven, are.

**There are two GB/s-scale CPU rails in this device, not one.** The other is the
zero-copy sampled gather: `SampledSource::GuestRuns` copies its whole window out of
guest RAM into a staging buffer on every bind, with no content cache, measured at
360 gathers and **842 MB a second** on a driven x86/PCI boot and repeating to the
digit across consecutive windows — the shape of unchanged content re-read every
frame. Both the task-GVA linear rail and the mapping-backed type-11 rail
contribute; the type-5 video rail is idle unless something is playing.

Skipping those gathers needs a witness for "these bytes did not change", and the
measured answer is that **it takes two**, because they cover disjoint writers:

- The hypervisor dirty bitmap (`HostOps::guest_write_gen`) witnesses guest CPU
  stores. Measured sound: bytes moved with no writer of any kind seen is **0**
  across four driven boots.
- `DeviceState::host_writes` witnesses this device's own writes, which the bitmap
  is defined not to see. It has to be **page-exact**: a per-mapping count was
  tried and read 15 stale binds a minute, because guest pages are reachable under
  more than one mapping id, which is what `deferred_alias_pages` exists for.

With both halves in place the rail caches: measured live at **5852 gathers skipped
against 4167 taken and 75.8 % of its bytes never read**, screen correct.

**A skip that still folds the window is not a skip, and that is how the first
version shipped.** The content fold that scored the rules above ran on every bind,
including the ones the cache served, so the bytes were still read — the copy was
gone and the cold read was not. The fold is now an audit on one bind in
`gather_witness::AUDIT_STRIDE`, and the decision is the two witness halves alone.
Its counterexample cell `gw_audit_unsound` is a standing alarm that also drops the
generation it refutes, so a witness that goes unsound self-heals within a stride
instead of serving a stale image forever. The counters that scored the two losing
rules (`gw_clean_*`, `gw_hit_global`, `gw_hit_scoped`) are gone with them; a doc
citing those names is describing a measurement, not a log you can grep.

The general shape is worth carrying: **when a measurement licenses a mechanism,
check whether the measurement is still on the hot path afterwards.** Here it was
the entire remaining cost of the thing being optimised.

Getting the second one complete took two passes, and the lesson generalises. A
hand-picked list of writer call sites missed `gva_view::map_fresh_span_within`,
whose callers write through a raw alias — the same hole `observe::gate`'s
`MAP_PAGES_SITES` was built after the footprint rail fell into it. **That table is
the authority on which code writes guest RAM; a new rail that needs the set should
read it rather than grep.**

That sentence used to say **two** CPU passes. There is one. The pass that copied the mapped readback
buffer into a `Vec` before scattering it is gone — the scatter reads the staging buffer in place
through a lease (`engine::LeasedFrame`), measured at `readback_split map_us=0 map=120 map_max_us=0`
on a driven boot against ~0.82 ms per flush before. What remains is `write_split land_us`, ~0.87 ms
per 8 MB frame of cache-cold scattered writes into guest RAM, and there is no second pass left to
remove: the only way past it is not to write the bytes at all. The rail's own doc
(`flush_mapping_windows_before_fence`) carries the full ledger, including the four levers that are
closed and why.

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

*Running* `backend-metal` is Apple-only, but *checking* it is not. `src/lib.rs` gates the arm on
`target_os`, not on the host, so any host can type-check it with no Apple SDK:

```sh
cargo check -p reims-vgpu --target aarch64-apple-darwin --all-targets --features backend-metal
```

Do this whenever you touch shared Rust code. Treating the arm as unavailable off Apple is what let
it rot to 11 errors once; "I am not on a Mac" is not a reason to leave it unchecked. Say
compile-checked rather than tested when that is what you did. Run the feature matrix from the repo
root when cfgs, features, backend boundaries, or shared Rust code change — it now includes that
cell:

```sh
scripts/feature-matrix/feature-matrix.sh
```

Before and after long Rust test runs, sweep orphaned test binaries:

```sh
pkill -9 -f 'target/debug/deps/reims_vgp[u]-'
```

The bracket is not decoration, and it is needed for QEMU too:

```sh
pkill -f 'qemu-system-x86_6[4]'    # not 'qemu-system-x86_64'
```

`pkill -f` matches against whole command lines, **including the command line of
the shell running the `pkill`**. Spelled without the bracket, the pattern occurs
in that shell's own arguments, so the sweep kills its own shell and everything
sequenced after it — the boot never starts. The bracketed class matches the real
process (whose argv contains `...x86_64`) but not the literal text
`...x86_6[4]` in the invoking command line.

This is what produced the long-standing "backgrounding a boot intermittently
exits 144 and silently does not start". It was never intermittent and never
about backgrounding; it fired whenever the surrounding command mentioned the
process name. Check the log's mtime, not just its contents — a stale log from a
previous boot reads exactly like a fresh failed one.

The bracket only protects the pattern from itself. **Any other literal mention
of the target name in the same command line is still matched**, so

```sh
pkill -f 'boot-x8[6]'; vm/boot-x86.sh ...     # still suicide
```

dies too: `boot-x8[6]` matches the `vm/boot-x86.sh` later on the same line.
Sweep in one command and start the thing you swept for in the next.

A boot must also outlive the shell that starts it, or a driver with its own
timeout takes the VM down mid-measurement — `boot-x86.sh` traps the signal and
kills QEMU, so the run ends looking like a guest failure:

```sh
setsid nohup vm/boot-x86.sh --device reims-vgpu-pci --testing >/tmp/boot.log 2>&1 </dev/null &
```

Then poll `ssh macos-vm true` until it answers, and abort the wait if
`pgrep -f 'qemu-system-x86_6[4]'` stops matching — otherwise a boot that died
early is indistinguishable from one still coming up. Give that poll a grace
period before it concludes the boot is dead: `boot-x86.sh` rebuilds the
staticlib, relinks QEMU and reverts the snapshot before QEMU exists at all, so a
`pgrep` in the first ~30 seconds reports "QEMU GONE" for a boot that has not
started yet.

**A `--testing` boot kills itself after `TESTING_TIMEOUT` (default 420 s) and
that is a budget you can spend before you measure.** The kill is the wedge
verdict — the script does a QMP register capture and reverts — so it reads in
`/tmp/boot.log` as a `Killed`/`terminating on signal 15` and in the guest as SSH
suddenly refusing, which looks exactly like the workload crashing the VM. Seven
minutes is not much: settling the guest can take four, and two 20-second probes
with their launch overhead take another two. Plan the run to fit, or raise it:

```sh
TESTING_TIMEOUT=1200 setsid nohup vm/boot-x86.sh --device reims-vgpu-pci --testing ...
```

Check `drain_duty` at the tail of the fail log before concluding a workload
killed the guest: `duty=0.001 draws=0` for the seconds leading up to the exit is
an idle, healthy device being torn down on schedule.

**SSH answering is not the guest being ready, and the difference is not
subtle.** sshd comes up well before the desktop settles, and a frame-pacing
probe started in that window scores the guest's own startup work. One measured
run reported **12.2 fps with a 35.8-second frame**, while `drain_duty` sat at
`duty=0.001` with `max_tranche_us` in the tens of microseconds for the whole
probe — the device had nothing to do, because the guest was not compositing
yet. The same build, re-probed on the same boot once `uptime` reported a load
average under 2, measured **119.2 fps**.

So gate the probe on the guest being quiet, not on the port being open:

```sh
ssh macos-vm "uptime"      # poll until the 1-minute load average settles
```

A pacing number taken from an unsettled guest is not a slow result, it is a
result about something else, and it will read as a catastrophic regression in
whatever change happens to be in the tree.

**Load settling is necessary and it is not sufficient. Probe twice.** Safari rAF
here is bimodal at ~59 fps and ~118 fps with nothing between them, and both
states occur on one unchanged binary within one boot. Four probes of the same
build over six minutes read **59.5, 117.3, 119.0, 120.0** — the low one being the
first probe after login — while `system_profiler` reported the display at
`1920 x 1080 @ 120.00Hz` throughout. So 59 is half-rate pacing, not a mode
change. The long-frame share falls with it (1.10%, 1.15%, 0.63%, 0.04%), so the
guest keeps improving well after the load average stops moving. A `load < 1.0`
gate does not separate the two: two boots that passed it
(1 user, 2-3 minutes up, load 0.93 and 0.99) still read 58.9 and 59.1 on their
first probe.

A *single* rAF figure therefore cannot support a claim about a code change in
either direction. This is not hypothetical — it produced a 118.9-vs-59.1 split
across two builds that read exactly like a 2x regression, and the "regressing"
change re-probed on the same boot at **117.3**. Four boots were spent before
re-probing rather than rebooting settled it.

**Three probes are not enough either, and the low one is not always first.** A
later session read **119.7, 121.3, 122.1** on one boot and **59.6, 60.8, 64.2**
on the boot before it — three probes each, consistent within each boot, and it
looked exactly like a 2x improvement from the one change between them. A third
boot on the *same* binary then read **119.3, 59.6, 119.3**. The low mode arrived
second, so neither "use the later reading" nor "three agreeing probes" survives;
what the guest latches is stable for minutes at a time and independent of the
build. Take the device-side counters as the result and rAF as decoration, and if
you must cite rAF, cite every probe you took rather than a summary of them.

```sh
# Probe at least twice on one boot; use the later reading, and say which.
for r in 1 2 3; do PROBE_SECONDS=20 scripts/browser-probe/web-gpu-probe.sh safari; sleep 20; done
```

Report rAF beside the device-side counters, which do not share this bimodality:
`drain_duty` (`duty`, `draw_us`, `flush_us`), `draw_phase`, `flush_rails`,
`readback_split`. Those reproduce across boots and are what a performance claim
should rest on; rAF is the corroborating number, not the evidence.

**For anything about frames reaching the screen, `present_hz` alone is not a
result — read the chain.** Four families divide it and each one has been the
suspect at some point:

| line | what it answers |
|---|---|
| `window_publish fresh` / `same_key` | how many distinct frames the device offered. `fresh + same_key == drain_duty tranches` always, because `publish_window_frame` runs once per tranche |
| `host_window_loop` | the window thread's own ticks, the redraws it asked for, the redraws the platform delivered, and each draw's disposition |
| `engine_lock` | wait and hold on the one `ENGINE` mutex, split into the drain worker and the window thread |
| `host_window_cadence` | what actually reached the swapchain, with `busy_fence` / `busy_acquire` |

The counterfactual switch belongs with them. Setting
`REIMS_VGPU_PROBE_NO_RENDER_WRITEBACK=1` on a boot drops every mapping-keyed
render window at the fence, which prices the writeback rail by removing it.
**Such a boot is incorrect and frame rate is the only number it can support**;
it announces itself once as `PROBE render_writeback_counterfactual=on`, so
check for that line before trusting anything else in a log.

**Record the host GPU's clock and power state beside any GPU-timing number, or
you are measuring the governor.** This is the same class of error as probing an
unsettled guest, and it is larger. Measured on one boot, one build, one driven
probe, with the only difference a synthetic load holding the host GPU at its top
clock:

| | host GPU at its own clock | held at top clock |
|---|---|---|
| `readback_split` `fence_us`/`fence` | 2.55 - 2.83 ms | **0.40 ms** |
| total fence time per second | 265 - 341 ms | **35 ms** |
| `drain_duty` `flush_us`/`flushes` | 4.0 ms | **0.83 - 1.75 ms** |
| Safari rAF long frames / worst frame | 7 (0.39 %) / 42 ms | **0 / 21 ms** |

On the measured host the GPU sat at P5, 800-1450 MHz of a 3090 MHz part, at
33-37 % reported utilisation, and dropped to P8/180 MHz the moment the guest went
quiet. So six sevenths of the fence wait was clock, not work, and the device's
real GPU cost per composited frame is about **0.40 ms**. The governor is behaving
correctly for what it sees: this workload submits ~0.4 ms of work per frame and
then blocks, which is a few per cent occupancy.

```sh
nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,utilization.gpu,pstate --format=csv
```

Two things follow, and the second decides what is worth building:

- A performance comparison between two builds is void unless both were taken in
  the same power state. Nothing in the boot scripts pins it.
- The second bullet used to read "**this device is latency-bound on a
  usually-downclocked GPU, not throughput-bound** — removing a whole GPU round
  trip is worth about six times what the flat GPU cost suggests; removing bytes
  is worth what it always was." **Do not restore it. It does not follow from the
  table above and it is now measured false.** The premise is that the wait
  shrinks with clock, and a copy that moves 8 MB shrinks with clock exactly as
  much as a latency does. The table could not tell them apart; the bullet picked
  one, and it picked wrong.

**What the wait actually is, measured.** `readback_split` now carries `bar_us`
and `gpu_us`, written by the device's own timestamp queries either side of the
copy inside the readback command buffer — GPU-timeline deltas, so no clock
correlation is involved. Driven one-second windows, x86/PCI, host GPU at P5:

```text
fence 2.549 ms   copy 2.286 ms (89.7%)   draw-wait 0.0010 ms   ask 0.262 ms
fence 1.906 ms   copy 1.710 ms (89.7%)   draw-wait 0.0010 ms   ask 0.195 ms
fence 1.474 ms   copy 1.296 ms (87.9%)   draw-wait 0.0010 ms   ask 0.177 ms
```

**87-91 % of the fence wait is the copy executing**, moving 8.29 MB at
3.6-6.4 GB/s in that power state. The draw batch it waits on is **0.05 %** — so
the composite render is effectively free and the readback is this device's entire
GPU cost. The remaining ~0.19 ms is the cost of asking. So:

- **Removing bytes is worth ~1:1 against 90 % of the largest cost in the
  device.** The deferred-flush ledger's four levers are priced in bytes and that
  is the right currency; they were being weighed against a wrong number.
- **Removing the second submission is worth the other ~11 %** — 0.18-0.26 ms per
  readback, stable, and no more. Do not spend a session merging the readback into
  the draw batch's submission expecting a frame back.

It also reframes the reports behind goal 11 (poor performance on iGPUs and older
GPUs). Those parts have the same governors and less headroom, and a device whose
frame cost is dominated by dragging a whole framebuffer back across the bus every
frame degrades on them faster than its own work would predict.

### Finding State Nothing Reads

Do not grep for this. `reims-vgpu` is a staticlib whose types are almost all
`pub`, and a `pub` item in a library is reachable by definition, so rustc's
dead-code pass never fires on it. Every hand-rolled sweep here has been either
wrong or exhausting: a grep for `.mapping_id` matches a dozen unrelated types,
and a grep restricted to `src/` misses the engine hooks only `tests/` calls.

```sh
scripts/dead-state/dead-state.sh --fields-only
```

This compiles a `pub`-downgraded copy of the crate in a scratch tree, which
turns rustc's own reachability analysis back on — it understands trait impls,
macros, cfg arms and generics, and it reads `--all-targets`, so a helper used
only by `tests/` counts as used. The working tree is untouched.

It is a report, not a gate. Three classes are legitimately unread: contract
tables (register maps, SDK enum mirrors, wire field offsets), error variants
only a future decode path constructs, and the `qemu/abi.rs` C surface QEMU
calls. A field written at five sites and read at none is not one of them.

Counting a test as a use is deliberate — it is what stops the report calling
`tests/`-only engine hooks dead — but it hides the opposite mistake: a product
mechanism whose only caller is the test written to prove it works.

```sh
scripts/dead-state/dead-state.sh --test-only
```

This compiles each arm a second time as a plain `--lib`, with `cfg(test)` off,
and reports what is dead there and live with tests. Test *infrastructure*
(`FakeHost`, the log redirectors) lands in this report and belongs there — the
integration tests are separate crates, so it cannot be `#[cfg(test)]`. For
everything else, deleting a hit means deleting its test; name the test in the
commit so a dropped count is never silent.

Neither mode can see a **local** that is computed and then thrown away, because
`let _ = x;` counts as a use — to rustc and to both scripts alike. That is not a
hypothetical: it hid four of them, one holding an unconditional `color_meta[0]`
index that could have panicked.

```sh
grep -rn '^\s*let _ = [a-z_][a-z0-9_]*;' --include='*.rs' crates/reims-vgpu/src
```

Triage each by what the name binds. Suppressing an unused **parameter** is
legitimate and usually means one cfg arm does not need it. Discarding a **local**
means the computation above it is dead. Two shapes are neither: a `let _ = buf;`
at the end of a scope can be a deliberate keep-alive for a retained backing
buffer, and the binding may carry a load-bearing `?` — `let bpp =
render_target_bpp(fmt)?;` refuses an unknown format, so the binding is dead but
the call is not. Read the whole enclosing function before cutting; a local used
only under one `#[cfg]` looks identical to a dead one on the other arm.

### Finding Measurements That Cannot Measure

`dead-state.sh` answers "does anything read this?". It cannot answer "does this
ever say anything?" — a counter with a live reader still earns nothing if the
branch that would move it is never taken. That failure mode is invisible in the
source: the field looks like a live counter until you count it.

```sh
scripts/constant-fields/constant-fields.sh          # defaults to the always-on log
```

This reports `key=value` fields in `/tmp/reims-vgpu-fail.log` that only ever take
**one** value, bucketed by emitting line family. A field that never varies is
either structurally impossible or vestigial. Its control is that it re-finds
`gva_write gpa_match=0`, the probe `d128fc1` deleted for exactly this reason.

Prefer the whole accumulated log over one boot, and drive the guest before
believing a zero — the type-11 ladder measured 12/5/8/0 undriven against
31 916/1 694/705/150 driven, quiet enough to talk someone into deleting a live
rung. Triage as a report, not a gate: a standing alarm reading zero is working,
and so is a host capability constant. The shape that convicts is a bucket
downstream of another bucket that is *itself* always zero, or a counterfactual
for code already deleted. See the script's README.

One caveat the script's own header does not give you: run it on a **single
driven boot** when you mean to act on a result. The accumulated log spans
builds, so a field that is constant there may only be code that has since
changed.

### Finding Comments That Describe Deleted Code

A doc link to an item that no longer exists reads exactly like one that does, so
a reader follows it and concludes the mechanism is still there. This is not a
formatting nit — it is the comment asserting something false about the code,
which is what "Write Comments For The Code" forbids. Two separate dead ends in
one session came from believing such a comment.

rustdoc already finds them; nothing was reading its output.

```sh
cargo doc -p reims-vgpu --no-deps --no-default-features \
  --features backend-vulkan,host-window 2>&1 |
  grep -o 'unresolved link to `[^`]*`'
```

Most hits are **not** stale: an item cfg'd out on the arm being documented
(`backend::metal` from a vulkan build) is unresolved and perfectly correct, and
that is the majority. Keep only targets that match no `fn`/`struct`/`enum`/
`const` anywhere in `src/` — those name something deleted. At the 628b37e sweep
that filter cut 64 hits to 12, all twelve real.

Do not `deny(rustdoc::broken_intra_doc_links)`; the cross-arm hits are load-bearing
and would have to be silenced individually.

**A third class is neither stale nor cross-arm, and it looks exactly like the
first.** A bare `` [`Item`] `` inside a module's own `//!` block does not
resolve here even when the item is `pub` in that very module — `[`Walk`]`,
`[`AUDIT_STRIDE`]` and `[`FINE_TILE`]` were all reported against the modules
that define them. The same shorthand in a `///` doc on an item resolves fine, so
a file can have both spellings and only one of them warns. rustdoc reports these
with **no `-->` file:line at all**, which is the tell; a cross-arm hit has one.

The fix is the full path — `` [`crate::runtime::land_redundancy::Walk`] `` —
not deleting the link, and it is worth doing rather than triaging past: three of
these were sitting in the report as permanent noise that a reader has to
re-triage every sweep. Verify with `comm` against a before-list rather than by
eye; the report is 60-odd lines and a new hit does not stand out in it.

Deleting a function is when this gets created. Its doc comment does not go with
it if the deletion is done by hand — and a doc block with no item under it does
not error, it silently concatenates onto the **next** item's doc. Twice in one
session that left a real contract explanation attached to an unrelated function
while the function it described had none. A detector for the blank-line-separated
form finds zero crate-wide; the adjacent form is not mechanically detectable, so
check by eye when you delete an item.

### Counting A Deduped Family Does Not Give You A Rate

Fourteen log families are emitted behind `observe::first_sight`, which fires
once per distinct key for the life of the boot. Grepping one of those counts
**distinct instances, not occurrences**, and the two can differ by more than an
order of magnitude: `lin_rung_blank_with_host_entry` read 15 lines and 360
occurrences on the same boot. A class can triple in rate with a flat line count.

That matters because the flat line count is what usually gets recorded. Before
carrying any "family X was N" forward — from a doc, a commit body, or a handoff
— check whether X is deduped, and say which quantity you mean. Two sections of
`note_guest_rung_blank`'s doc record a `0` that is a line count, next to prose
reading as though the class had stopped happening.

`lin_rung_blank_with_host_entry` is currently the only family carrying both a
deduped line and an unconditional `note_store_route` counter, so it is the only
one where a rate is available at all. For the other thirteen the occurrence rate
is not measured, and "the counts are stable across boots" is a statement about
distinct instances only.

### A Performance Commit Is Not Verified Until Its Pixels Are

**A performance commit does not land until `scripts/visual-gate/visual-gate.sh`
passes on a live boot of the pathway it touches, and the commit body says which
pathway and quotes the verdict line.**

This is a rule rather than a suggestion because the alternative has already been
paid for. A branch of 59 commits — 21 of them code-changing, +7687/−820 across 38
files, about seven hours — was reset off wholesale because it had introduced
graphical glitches. Every one of those commits was verified: clippy, unit tests,
the feature matrix, device-side performance counters. The work was performance
work, so the instruments it was checked with were performance instruments, and a
rendering regression was invisible **by construction**. Nothing lied. The right
question was never asked.

The three probes below already answer that question and had done so for weeks.
None of them gated anything, which is the entire defect. `visual-gate` runs all
three on a settled guest, applies the six silent-loss counters over its own byte
window of the fail log, and exits non-zero on any of it. Its README has the
counter table and the two things it is not.

If a change cannot be gated — because it only manifests under a workload no
probe drives — say so explicitly in the commit body under "Not verified". The
suspect commit on that branch did exactly that, honestly, in its own body. The
failure was that nobody treated the admission as blocking.

### A Screenshot Cannot Say Who Dropped The Pixels

Several open goals are reported as screenshots: web content whose background
disappears, a logout window missing its buttons, a wallpaper shifted with a black
band. A screenshot shows that something is not on screen. It cannot show whether
the guest declined to draw it or whether this device lost it on the way, and
those are different bugs with different owners. Staring at the image does not
separate them on the tenth look either.

Two independent observations of the same frame do. The guest's accessibility API
reports what it believes it drew and at what rectangles — that reads the guest's
own view hierarchy, upstream of everything here — and the host capture is then
measured at exactly those rectangles.

```sh
scripts/modal-button-probe/modal-button-probe.sh -n 20 --appearance alternate --keep /tmp/mbp
```

Exits 1 on any button the guest declared and the frame does not show. It
currently finds none: 20 trials alternating dark and light, 40 button checks, all
drawn. So it is the instrument, not a result — and note it summons the **log-out**
modal, which is scriptable and dismissable, not the Control-Power one the bug
report names. Read its README before quoting a result from it.

The shape generalises past this one script. When a visual bug report arrives,
ask what the guest *intended* before asking what the device did; if there is no
way to read the intent, building one is the fix-enabling work, not a detour.

`scripts/web-content-probe/` is the same shape for goal 8: the page declares a
palette and its screen rectangles, the host classifies measured means to the
nearest palette entry, and a lost fill reports as `WHITE` or `BLACK` by name.
Under layer churn — a subtree of 24 rotated, half-promoted, scrolling children
rebuilt every 250 ms behind `position: fixed` patches — **goal 8 has not
reproduced**: Safari 8 captures and Firefox 12 captures, every region correct, on
a settled x86/PCI guest. That is a small sample against a bug reported as
occasional, so it bounds nothing; what it establishes is that the ordinary
compositing path is sound under churn.

**Both of that probe's earlier results were worthless, for opposite reasons, and
both failure modes are general.** The first churning run reported clean on a page
whose repaint timer reached the screen with nothing — the churn used viewport
coordinates inside a `contain: strict` container that clipped them away, so the
"stressor" was a static page. The next run reported seven of eleven regions
corrupt in nineteen consecutive captures, and the cause was a macOS sheet dimming
the window: every colour measured exactly half, and the regions that *passed*
passed only because halved `BG` and halved `GREEN` are equidistant from their own
palette entry and from `BLACK`.

`scripts/wallpaper-probe/` does the same for goal 10, and it gets a stronger
declaration than either of the others because it **supplies the wallpaper**: 64
vertical bars in two colours in a fixed aperiodic pattern, decoded out of the
host capture at three vertical bands. Three bands rather than one because the
distinction that names the owner is invisible in a screenshot —

| what the bands say | what it means |
|---|---|
| same shift in all three | uniform origin offset |
| shift growing down the screen | row stride mismatch, and the difference gives the error |
| no shift, bars lost at an edge | clipped, not moved |
| every bar lost in every band | desktop covered — not a result |

Verified against synthetic frames: a 192 px left slide reads `-6/-6/-6`, and a
shear from 0 to 240 px reads `-2/-4/-6`. Those two are the same screenshot to the
eye. **Goal 10 did not reproduce in 6 live trials** driven by its own reported
trigger (appearance flipped, desktop passed through a system dynamic picture and
back, guest asked each time which picture it believes is set); every band read
`shift=0 lost=0`, and the kept frame was read to confirm the barcode really was
the full-screen wallpaper. Six trials bound nothing about an occasional bug.

`scripts/window-drag-probe/` is goal 6's first instrument, and its first result
is the largest open number in this file. Moving a 1000x640 Safari window at
~115 Hz for fifteen seconds, twice, on a settled x86/PCI guest:

```text
present_hz med 10.7    duty med 0.98    max_tranche_us med ~130 000
flush_us 641 ms of one 1095 ms second, across 523 flushes
draws 2351, flushes 523, presents 11 (offered 11) in that second
```

The idle control on the same guest — Safari open, no motion — is `duty=0.001`,
`max_tranche_us=8`, **zero** draws and flushes, so all of the above is the
motion. The device is busy for essentially the whole second and produces eleven
frames, `offered` equals `presents` so it is not dropping frames it made, and
**about two thirds of the worker's second is the deferred writeback rail** — the
cost `flush_mapping_windows_before_fence` documents, now seen dominating a
window-move workload rather than a WebGL one. A single tranche blocks the worker
for ~130 ms, fifteen frames at 120 Hz.

The device keeps up with 212 fences a second and presents 11, so **roughly 200
full-frame composites a second are written back to guest RAM and never
displayed**. Do not read that as a superseding opportunity: the bucket that
means collapsible is `render_flush_age_sub_ms` (a burst rewriting one surface
inside one drain tranche, no fence between) and it is **still 0** here. What
grew is `_sub_frame`, landings 1-8.33 ms apart, each its own composite behind
its own fence — and every fence entitles the guest to the bytes, so collapsing
them is the undeclared-read question and not a separate lever. This paragraph
exists because the first write-up of this measurement claimed the lever had
reopened, and it had not.

That paragraph used to end "nothing in these numbers separates *the guest asked
for this much work* from *the device does more than it was asked*". **Reading
the rest of the same log separates them, and the answer is the guest.** Three
families nobody had put beside each other:

| what it reads | what it removes |
|---|---|
| `display_vbl window_hz=120.0` throughout the drag | the guest is not pacing slowly for want of a time base |
| `THRASH present_action_starvation` — **zero** in the whole accumulated log | `MAX_UNPAINTED_PRESENTS` is not holding the guest at the FIFO head |
| `window_publish fresh + same_key == drain_duty tranches`, every window (19+28=47, 23+14=37, 22+8=30) | the publish rate is the tranche rate *by construction* |

That identity is not a coincidence: `publish_window_frame` runs once per drain
tranche, after `device.drain` returns, so the device samples its own finished
frame at the tranche boundary and nowhere else. `fresh` is therefore the count
of distinct frames offered to the window, and it reads **fewer than
`tranches`** — so it is not an undersample of a faster guest.

So the guest completes ~20 frames a second on that boot, and 233/20 is **~12
full-screen composites per frame** — its layer tree, not work this device
invented. Execution is sync-per-packet, which makes the guest's frame time our
per-packet cost: 12 render flushes at 3.03 ms (`flush_rails render_us=499575
render=165`) plus ~120 draws at 136 µs is **~53 ms, and 1000/53 is 19**, which
matches `fresh`. **Goal 6 is not a presentation-cadence bug. The guest is
serialized behind this device's writeback rail**, and the ~200 composites a
second that never reach the screen are the other 11 of each 12. The lever is
the one `flush_mapping_windows_before_fence` names.

**One link in that chain is separately broken, and the first guess at it was
wrong.** Two 15-second runs on one settled x86/PCI guest, host GPU at P8:

```text
window_publish fresh   med 34/s      (fresh+same_key == tranches, every second)
host_window_cadence    present_hz med 16.8 / 17.0,  busy_fence=0 busy_acquire=0
engine_lock  window=17  window_blocked med 0-1   window_wait_us med 0-440
             worker=12 796  worker_hold_us med 488 000  worker_hold_max_us 4 600
```

Half the frames this device offers never reach the screen. The obvious
candidate was the shared `ENGINE` mutex — `window_present_frame` takes the same
lock the drain worker holds through its readback fences — and `engine_lock` was
built to test it. **It is refuted.** The worker acquires that lock ~12 800
times a second and holds it for half of every second, but the window thread
asks for it only 17 times and is blocked on it for **0.04 % of a second**. The
window is not waiting for the engine.

**The loop is not it either, and that finishes the presentation path.**
`host_window_loop` counts the window thread's own ticks, the redraws it asked
for, the redraws the platform delivered, and each draw's disposition. Driven,
on the same workload:

```text
host_window_loop ticks=1014 redraws_asked=498 draws=498 draws_fresh=19 draws_stale=479
window_publish   fresh=36    host_window_cadence presents=19
```

The loop wakes ~1000 times a second under load, exactly as it does idle
(`ticks=998 draws=499 draws_stale=499` with the desktop still), and **every
redraw it asks for is delivered** — `redraws_asked == draws` to the unit, so
nothing is coalescing or delaying them. It polls 500 times a second and finds a
new `Frame::seq` only 17-19 times against 34-36 publishes.

A 500 Hz sampler can only miss half of 34 events if they arrive in pairs closer
together than 2 ms. So **the publishes are bursty, and the burst is the drain
worker's tranche shape**: `max_tranche_us` med 87 ms with `slow_tranches` 17 of
38, one long tranche producing nothing visible and then several short ones
publishing back to back. Latest-wins collapses each pair, correctly — the first
of a pair was superseded before any display could have shown it.

So there is no frame to recover here. **Every stage of the presentation path is
now measured and healthy** — VBL at 120 Hz, backpressure never engaged, publish
keeping up with the tranche rate, the event loop at 500 Hz with no dropped
redraws, the engine lock uncontended, `busy_fence` and `busy_acquire` zero.
Goal 6 is entirely the ~30 ms this device charges the guest per frame, and
inside that the eight full-screen writebacks (2.26 GB/s, `write_split frag=272
bytes=2256076800` in one second against 34 published frames) are ~20 ms of it.
Do not spend another session on cadence.

### The guest's own validity quad will not tell you which writebacks to skip

The recurring idea is that the protocol already states resource ownership, so
the guest must be telling us which of those ~200 undisplayed composites a second
it actually wants. `resource_validity` consumes exactly that statement — a
four-field quad from `CmdInvalidateResources` (`0x34`) and from the resource
table inside every `EXEC_INDIRECT2` — and `writeback_refused` already gates both
flush rails on it. **The measurement that decides the idea needs no new code:
`validity_wb_licensed` / `_unstated` / `_superseded` are emitted per landing and
sum exactly to `mapw_fence_flush`.**

Two full visual-gate windows on a settled x86/PCI guest, Safari compositing:

```text
run   licensed   unstated   superseded   mapw_fence_flush
 A       2442       5203            0         7645
 B       1941       5668            0         7609
```

**The sets do diverge, and hugely — 68-74 % of every window landed at a fence is
one the guest never made a statement about.** That is the divergence the idea
predicts. It is also useless, and the direction is the whole point:

- The quad says who wrote a resource *last*, not who is about to *read* it.
  `Unstated` means the guest has never claimed a CPU write, so there is nothing
  to order our publish against — it is the absence of a claim, not a statement
  that the bytes are unwanted.
- **Refusing an `Unstated` landing has been tried and it turns a compositing
  layer black.** `writeback_refused`'s own doc records the boot that cost. The
  safe reading of "the guest never claimed a write" is to deliver the frame.
- `superseded` reading **0** in both windows is the other half: the exec table's
  `clear_host_valid` already drops a mapping's pending windows when it arrives,
  so the case where the guest genuinely does own the pages is handled upstream
  and never reaches a flush.

So the quad is fully consumed, its refusal path is sound, and there is no
unclaimed saving in it. A demand-driven writeback needs a witness for *reads*,
which nothing in this protocol supplies. Do not spend a session re-deriving this
from the decode side.

### What the writeback is worth, measured by not doing it

The ~20 ms above is an *attribution* — the parts sum to the whole. Whether
removing the rail returns its share is a different claim, and it is the one the
read-witness route is priced on, so
`REIMS_VGPU_PROBE_NO_RENDER_WRITEBACK=1` exists to ask. It drops every
mapping-keyed render window at the fence instead of landing it. **A boot with
it set is incorrect** — the guest is told its render completed while its pages
hold the previous frame — and frame rate is the only number such a boot can
support. The screen stays correct, which is why the experiment is possible at
all: the host window presents from the engine resident (`direct_frac=1.00`), so
it never reads the pages being withheld.

One representative second from each, same guest image, same stressor, host GPU
at P8/180 MHz throughout:

| | control | counterfactual #1 | counterfactual #2 |
|---|---|---|---|
| `window_publish fresh` — guest frames/s | 34 | **98** | 34 |
| `host_window_cadence present_hz` | 17.4 | **68.6** | 16.4 |
| `drain_duty duty` | 0.97 | **0.77** | 0.97 |
| `tranches` | 43 | **237** | 39 |
| `mapw_fence_flush` — composites | 296 | 800 | 280 |
| composites per guest frame | 8.7 | 8.2 | 8.2 |
| `draws` | 2270 | 6102 | 2150 |
| draws per guest frame | 67 | 62 | 63 |
| `flush_us` | 760 ms | **122 ms** | **70 ms** |
| `draw_us`/draw | 103 µs | 97 µs | **421 µs** |

Read the per-frame rows first: **the guest asks for the same work in all three
— 8.2 composites and ~63 draws per frame.** So this is the same workload run at
different speeds, not three different workloads.

**Counterfactual #1 is the reading that matters. Removing the writeback ran the
identical workload at 2.9x the guest frame rate and 3.9x the displayed one,
with the drain worker no longer saturated** (`duty` 0.77, so it had capacity
left over) and the tranche count up 5.5x, which is also what un-bunched the
publishes and let the window loop see 68 of them instead of 17. Per-draw cost
was unchanged, so nothing was traded away for it.

**Counterfactual #2, on the same boot minutes later, gave the whole thing
back**, and how it did is the finding to carry. `flush_us` stayed collapsed —
70 ms — but `draw_us`/draw went from 97 µs to **421 µs**, and the store routes
say why: 54 `t11rung_resident_refused` with `gw_rail_t11_kb=437400`, i.e. 54
type-11 binds refused their resident and gathered **8 MB each** out of guest
RAM. The control has `gw_rail_t11=114` at 0.9 MB each and **zero** refusals;
counterfactual #1 has no type-11 gather at all.

That refusal is `guest_replaced` — the guest CPU-wrote the surface's pages —
and it is an artefact of the experiment being wrong rather than a property of a
correct demand-driven rail: with our frame never landed, the pages drift into
holding neither our pixels nor a full guest frame, and the drift compounds over
a boot.

Six runs across three boots put that beyond a single-run story, and they also
supply the control the comparison needs:

| run | boot | writeback | `t11rung_resident_refused` | `fresh` |
|---|---|---|---|---|
| 1 | A, first drag | on | — | 34 |
| 2 | A, second drag | on | — | 36 |
| 3 | B, first drag | on | — | 37 |
| 4 | C, **first drag** | **off** | — | **99** |
| 5 | C, second drag | off | 54 | 35 |
| 6 | C, third drag | off | 49 | 38 |

**The control does not degrade across runs** — 34, 36, 37, no refusal in any of
them — so "the first drag after a boot is fast" is not the explanation; run 3
is a first drag and reads 37. The refusals appear only with the writeback off,
only after its first run, and exactly where the gain disappears.

The chain from the missing writeback to the 8 MB gather is named by counters
rather than inferred. Three that appear in the degraded run and in **neither**
the control nor the fast counterfactual:

```text
gw_refused_guest_store=121     the gather witness refused to vouch: a store it
                               could not attribute to this device
type11_seed_guest_wrote=86     the seed path concluded the guest wrote the surface
t11rung_resident_refused=54    the resident was refused, and 8 MB gathered instead
```

and one that appears only in the control: `gw_vouched=40
gw_vouched_kb=9920`. **Neither counterfactual run vouches a single window** —
that is the direct consequence of never landing, since the device's own
page-exact write record is what the witness subtracts to tell its stores from
the guest's. Once real guest CPU stores accumulate with no device write to
re-baseline against, the witness has to assume the worst, and the type-11 rung
above it does the same.

The hazard is what a demand-driven rail would hit **if it got the witness
wrong**, and the exchange rate is ruinous: a 2.26 GB/s writeback traded for an
8 MB-per-bind gather. **So the design requirement, stated before anyone builds
it: skipping a writeback must keep the guest-write witness and the type-11
resident rung sound. A rail that only stops writing has been measured, and it
is a wash.**

Two things this does not establish. The causal story for runs 5 and 6 is a
correlation across six runs, not a proven mechanism — nothing was changed to
test it. And 98 fps is not 120: even with the largest cost in the device
removed outright, this workload does not reach goal 6's number, so the
writeback is necessary but not obviously sufficient.

### 86% of what the writeback writes is already in the page

The ledger says removing bytes from that rail is worth ~1:1. It never asked how
many of the bytes were needed. `scripts/window-drag-probe` now reports
`land_redundancy`, which on one write in 64 compares the bytes about to be
stored against the bytes already at the destination. Settled x86/PCI, Safari
drag, five landings a second audited over fourteen consecutive seconds:

| granularity | median already identical | range |
|---|---|---|
| 256-byte tile | **86.1 %** | 83.9 – 90.0 |
| 4 KiB guest page | 43.1 % | 35.9 – 49.3 |

The idle desktop reads 2025/2025 pages identical, so the idle case is total.

**⚠ Both rows are biased samples and the bias is measured, so do not rest a
design on the exact figure.** The audit takes one write in 64 from a stream of
~8.5 composites per guest frame whose redundancy differs by tens of points —
static layer surfaces are near-totally redundant, the final composite is not —
so *which* of the 8.5 the stride lands on is most of the answer. This was found
the hard way: adding a second hook, on a writer firing 37 000 times a second
against the mapping leg's ~290, moved the **mapping** leg's own reading from
`same_fine` 90.75 % / `same_pages` 51.78 % to **99.60 % / 97.85 %** — same
stressor, same guest, comparable motion, `duty` 0.97 both, the same number of
landings sampled. A fixed stride over a stream one source dominates 130:1
aliases onto a different phase of the other's cycle.

The stride is now per leg, which stops one hook perturbing another's sampling
but does **not** make the number a population estimate. Six driven runs at tile
granularity, in the order taken:

```text
86.1   89.9   91.8   90.75   99.60   78.20      (medians, per run)
```

The last is the first run with the corrected per-leg stride, and it is the
*lowest* of the six — so the fix did not converge the answer, it just removed
one known bias. The spread is run-to-run over a heterogeneous population that a
1-in-64 stride samples at ~4 landings a second out of ~288.

**What this supports is a range, not a figure: roughly 70 – 99 % of the
writeback's bytes are already at the destination, worst observed second 68 %.**
That is enough to justify building the GPU pass — even the worst second declines
~68 % of a 300 ms/s copy and a 209 ms/s scatter — and not enough to predict its
saving to better than a factor.

### The census that closed that range: 92 %, counted rather than sampled

That paragraph used to end by asking for a per-surface split, and saying nobody
had built one. It is built and run. `REIMS_VGPU_PROBE_TILE_DIFF_CENSUS=1` runs
the GPU difference pass (`backend::vulkan::engine::diff_pass`) over **every**
resident-target readback, keyed per `TargetIdentity`, and counts the 256-byte
tiles that differ from the previous frame of that same surface. Two driven
Safari window drags (`--motion reposition`, the default and the weaker one) on
one settled x86/PCI guest:

```text
run 1   3011 frames    97 214 840 tiles    8 002 871 changed   8.23 %
run 2   2940 frames    94 716 416 tiles    7 499 751 changed   7.92 %
        0 declines in either run
```

**About 92 % of a composited frame is already what the frame before it was**,
and that is a complete count over the drag rather than one landing in 64. It
reproduces across runs to 0.3 points and within a run per second: twelve of
nineteen seconds of run 1 fall between 8.2 % and 9.2 % changed, one at 1.4 %,
two at 25.5 %. The undriven desktop before the drag is near-total — 108 counted
frames moved `tdc_tiles_changed` by 44 tiles.

So the sampled range 78 – 99 % and this 92 % agree, and the sample was not
biased so much as *noisy*. **Use 92 % where the older text says "a factor".**

Four things this does not say, and the last is load-bearing for anyone sizing
the real rail:

- **It is not a timing result.** The census runs its own submission per
  readback — a second whole-frame device-local copy, a dispatch and a fence —
  so `fence_us`, `duty` and `present_hz` from such a boot are not comparable to
  anything. It announces itself once as `PROBE tile_diff_census=on`; check for
  that line before trusting a number from a log. That said `window_publish
  fresh` read med 34/s and `land_us/land` med 763 µs here, both in line with
  the control runs above, so the probe did not visibly deform what it measured.
- **It measures frame-to-frame change, not change against the guest's pages.**
  `prev` is the previous frame of the same identity. The two coincide only
  while the eager writeback keeps landing every frame — which it does on this
  boot, and which is exactly why the measurement is taken before the rail is
  changed rather than after.
- **It covers the mapping-keyed leg only** (`read_target_leased`). The raw
  task-GVA leg reads `gvaw_fence_flush=432` against `mapw_fence_flush=288` and
  is not censused, so this is not a statement about all of the writeback.
- `tdc_overflow` fired 5 times in run 1 against a bound of **8** targets, and
  every overflow drops the whole map and re-seeds — which is where all 39 of
  that run's seeds came from. So the live working set is above 8 surfaces. At
  two frames of device-local memory per surface, the real rail's equivalent
  bound is a VRAM decision, not a bookkeeping one, and 8 was already too low.

  The 8 came from reading "~8 composites per guest frame" as a surface count.
  It is not one: that figure counts **flushes**, several of which are the same
  surface flushed again, and the population a scratch pool bounds is *distinct
  identities*. Nothing had measured that quantity. The bound is now 16 — one
  doubling, not a derivation — and `tdc_targets_sum` reports the live map size
  accumulated per censused readback, so the working set is
  `tdc_targets_sum / (tdc_frames + tdc_seed)` and the next run states it
  instead of leaving it inferred from whether an overflow fired. **That number
  has not been read yet**; the runs above predate the counter.

**The redundancy is spread across every landing, not concentrated in a few, and
that decides which build.** A mean cannot tell *seven wholly-unchanged surfaces
beside one wholly-changed one* from *eight surfaces each 87 % unchanged*; both
read 87 %, and only the first is collected by hashing each landing and declining
it entire — no tile bitmap, no per-target shadow of the previous frame, no
compaction. `land_redundancy` now buckets each audited landing by its **own**
`same_fine/fine` and charges its bytes to the bucket. Two 15-second drags on one
settled x86/PCI guest, 114 audited landings on the mapping leg:

```text
        landings  whole  over_90  over_50  under_50   whole bytes   same_fine
drag1      61       10      49       2        0          16.4 %       97.60 %
drag2      53        2      32      19        0           3.8 %       92.61 %
```

**Declining whole landings collects 4 – 16 % of the bytes; tile compaction
collects 93 – 98 %.** A factor of six to twenty-four, so the cheap build is not
a first step toward the expensive one — it is a different and much smaller
thing. Do not spend a session on a landing-granular hash-and-skip.

`under_50` reading **zero** across all 114 is the second finding: every landing
is at least half already-correct and most are over nine tenths, so a tile pass
collects near-uniformly and there is no worst-case landing to design around.
Both are shapes, not point estimates — the window mean still moved 97.6 → 92.6
between two runs minutes apart, while the shape did not.

**Probed twice on that boot, and the second run is higher**: fine 86.2 – 95.6 %
(median 89.9) over eleven seconds, pages 54.0 – 81.8 % (median 61.1). So the
first run's 86 % is the conservative reading, and the two rows do not move
together — the page fraction swung 43 → 61 between runs while the tile fraction
moved 86 → 90. **The tile number is the one a design should rest on**; the page
number depends on where in the screen the motion happens to be and is much the
noisier of the two.

**Do not confuse this with the damage rect, which is separately measured and
separately dead.** `note_store_damage_coverage` reads the guest's declared
`store_damage_texels / store_attach_texels` at **99.34 %** — the Store that ends
a full-screen composite declares the full screen, and the whole declared rect is
worth 0.66 %. The guest is not lying: it really does re-composite the whole
desktop. It just produces, for most of it, the bytes that were already there,
because the wallpaper under a moving window does not change because the window
moved. Declared damage and content change are different quantities and only one
of them is worth anything.

The gap between the two rows decides which unit a rail should work in. A
page-granular one leaves half the redundancy on the table and a row-granular one
would find almost none, so when a skip is being considered the granularity is
not an implementation detail, it is the size of the prize.

**The CPU rail this licensed was built, and it is refuted. Do not build it
again.** Compare each tile in the scatter, store only the runs that differ.
Measured on one settled x86/PCI guest with the same stressor:

| run | tile rail | bytes declined | `land_us` per landing |
|---|---|---|---|
| drag1 | off | — | med **744** (732 – 760) |
| drag2 | off | — | med **769** (737 – 788) |
| drag4 | on | 91.6 – 91.8 % | med **802** (791 – 956) |

**Declining 92 % of the stores made the scatter slower**, and the ranges do not
overlap. The cause is worth carrying past this device: **a full-cache-line store
does not read its destination.** The hardware elides the read-for-ownership, so
a store that gets declined never cost a read in the first place, and the compare
adds a whole 8 MB read of guest RAM the eager path never paid. What it saves is
DRAM write bandwidth, and `land_us` is not bound by that.

The run did confirm the audit to the decimal — `same_fine` 91.6/91.8/91.8 %
against bytes actually declined 91.6/91.8/91.8 % — so the measurement and the
mechanism agree and it is the mechanism's *economics* that were wrong. That is
the general trap: a redundancy fraction is not a saving, because it says nothing
about what the work it would skip actually costs.

**What survives is the number the GPU pass is priced on.** The failure is the
compare, not the skipping, and the same run separates them. A full landing is
744 µs = read `src` + write `dst`; a scatter handed its changed-tile set from
outside reads and writes only 8 % of that:

```text
0.082 * 744 us = ~61 us per landing, saving ~683 us
683 us * 272 landings/s = ~186 ms/s of a ~990 ms busy second
```

So a **GPU-side pass is the only route, and now for two reasons rather than
one.** It is the only thing that can decline the copy across the bus — 78 % of
the readback fence — *and* the only way to get the scatter's own 186 ms/s,
because the CPU cannot derive the tile set for less than the saving is worth.
Both halves need the same bitmap and neither is reachable without it.

The skip is an **identity, not a heuristic** — not writing a byte that already
holds the value being written leaves memory in the same state — so it needs no
rect, no witness and no guess. It also satisfies the requirement the
counterfactual left behind: pages always hold the frame the landing would have
put there, so the type-11 resident rung and `gather_witness`'s subtraction of
this device's own page-exact write record both see exactly what an eager landing
would have left. That is what `REIMS_VGPU_PROBE_NO_RENDER_WRITEBACK` could not
say, and why it was a wash.

Two limits on the number. It covers the **mapping-keyed** leg that took the
fragmented path — `write_split contig=0 frag=272` on that boot, so all of it
there, but a host with contiguous mappings would take `mapping_write`'s row
writers instead and the line would go silent rather than wrong. And
`store_routes` reads `gvaw_fence_flush=432` beside `mapw_fence_flush=288`, so
the raw task-GVA leg is not small and is not covered; it is also not the leg the
counterfactual dropped to measure 2.9x.

### The tier behind it, so it does not have to be re-derived

Run 4 left the worker at `duty` 0.77 with 62 draws a frame at ~97 µs, so the
draw path becomes the ceiling next. It costs the **same per draw with the
writeback on or off** (control 103 µs, counterfactual 97 µs), so the control's
own numbers price it. Per draw, from `chain_phase` over one control second of
2270 draws:

```text
binds    23.5 us        engine   50.4 us  ->  stage           25.2 us
sampled   7.7 us                              sampled_upload  11.1 us
store    11.0 us                              record           3.8 us
pipeline  4.2 us                              pipeline         3.3 us
prep      1.9 us                              prep             1.7 us
```

`binds_us` and `draw_phase`'s `stage_us` are the two big ones and are within a
microsecond of each other.

**`binds_us` is divided, but the division that was published is not evidence and
has to be re-taken.** `bind_phase` read, on a driven drag second of 2546 draws:

```text
vertex_us 13.32   fragment_us 8.26   attrs_us 0.00   -> 21.6 of binds_us 23.15
```

That was written up as "93 % accounted, and the attribute walk is **zero**".
**Both halves rest on a truncating accumulator and neither survives it.** Every
phase census here charged `Duration::as_micros`, which rounds a 700 ns span down
to nothing — see `observe::phase_clock` — so a part whose per-item cost is
sub-microsecond reads zero however often it runs, and a remainder computed
against it reads as though the expensive parts were the whole cost. `attrs_us`
reading `0`, `3`, `23`, `10`, `0`, `0` across consecutive windows is that floor,
not a free attribute walk. The censuses now accumulate nanoseconds and divide
once at the window boundary; the numbers above predate the fix.

**Re-taken on the fixed accumulator, the conclusion holds and is now evidence.**
Five driven one-second windows over 500 binds each, x86/PCI, host GPU P8, with
`bind_phase` and `chain_phase` joined per window:

```text
binds   vertex_us   fragment_us   attrs_us   binds_us   accounted
  735       15860         12394         98      29493       96.1 %
  599       12041          7864         67      20635       96.8 %
  871       13452          8323        116      22933       95.5 %
  768       11031          6486         86      18259       96.4 %
  514        9959          7071         73      17749       96.4 %
```

**95.5-96.8 % accounted, and the attribute walk is 0.11-0.14 µs a bind** —
0.33-0.51 % of the column. So it is not zero, it never was, and it is still
negligible: the truncating reading was right by accident and is now right on
purpose. The column is `load_buffer_content` over vertex and fragment buffers,
and the remaining ~3.5 % is the two shader `Arc` clones and the `BTreeSet`.
Whatever is done here is done inside that one function. `draw_phase`'s
`stage_us` is still undivided.

**`descriptors_us` has a number now, and it is not a lever.** It had never been
quoted anywhere, so the descriptor-side ceiling was unknown rather than small.
Six driven windows of 400-514 draws on the same boot, per draw:

```text
prep 2.1-7.3   pipeline 4.7-11.6   stage 53.6-66.9   acquire_sampled 0.4 / 20.3
sampled_upload 6.8-9.6   descriptors 0.74-0.92   record 5.3-6.7   submit 1.4-2.7
```

**0.74-0.92 µs a draw**, about 1.5 % of the draw. Writing the descriptor set is
not where the time goes and pool pressure is not a story here.

Two other things that reading says. `acquire_sampled_us` is **bimodal to two
decimal places** — 0.40-0.46 on three windows and 20.30-20.48 on the other two,
with nothing between — so something in that phase is switched, not scaled, and
whatever it is costs 20 µs a draw when it is on. And `stage_us` at **54-67 µs a
draw is two thirds of the whole draw on this workload**, four to five times what
it costs under a window drag, which makes it the largest undivided column in the
device by a wide margin.

Read those against the window-drag table above with care: this is Safari
compositing the web-content probe's churn page, not a window drag, and the two
workloads apportion the draw differently. What transfers is the ranking, not the
microseconds.

The arithmetic that says whether the tier matters: 120 fps is 8.33 ms, run 4
achieved 10.2 ms, and draws are 6.0 ms of that. So goal 6 needs about **2 ms a
frame out of the draw path** on top of the writeback — roughly a third of it,
or two thirds of either big column. Not obviously reachable, and not obviously
out of reach.

One dead end already checked: `BufferContent::Bytes` is an `Arc<Vec<u8>>`, so
the per-attribute `content.clone()` inside the binds phase is a refcount bump,
not a copy.

Two cautions before building on any of it. The stressor moves the window through
the accessibility API, not a pointer drag: `CGEventPost` is silently discarded
here because the posting process is not trusted for Accessibility and TCC.db
cannot be written (no passwordless sudo, SIP Filesystem Protections on) —
measured as 1800 events posted at exactly 120.0 Hz with the window not moving
one pixel, which is why the harness refuses a verdict unless the window moved.
And the 53 ms arithmetic above is an attribution, not a controlled experiment:
it says the parts sum to the whole, not that removing one part returns its share.

Ask the guest for the desktop size with `system_profiler SPDisplaysDataType`, not
with Finder: `tell application "Finder" to get bounds of window of desktop`
answers `AppleEvent timed out (-1712)` here, which reads like a wedged guest
rather than like one unavailable scripting target.

So a probe has to witness two things besides its verdict, and this one now does.
**That its stressor ran**: the page publishes a beat counter, the host refuses to
start unless it advances, and one churn child (`CHURN_WITNESS`) is declared and
checked like a patch. **That the frame it measured was the frame it meant to**: a
least-squares fit of a single scale across all regions, where a tight fit below
0.9 is a global dim and the capture is discarded, because a real loss is local —
measured 0.4996 at worst residual 3/255 for the sheet, against residual 187 when
one region is lost to black. Neither guard is specific to this probe, and a
"clean" run from an instrument carrying neither is not evidence.

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
Use the appropriate subset for the host and change. The first command needs an Apple host; off
Apple, substitute the `cargo check --target aarch64-apple-darwin` above, which is weaker than
clippy but far better than skipping the arm:

```sh
cargo clippy -p reims-vgpu --all-targets --features backend-metal -- -D warnings
cargo clippy -p reims-vgpu --all-targets --no-default-features --features backend-vulkan,host-window -- -D warnings
cargo clippy -p reims-vgpu --target x86_64-unknown-linux-gnu --all-targets --no-default-features --features backend-vulkan,host-window -- -D warnings
```

Do not hide warnings, skip an affected arm, or commit a dropped test count without calling it out.
