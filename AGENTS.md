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

**A pixel count is not a visual defect — read the magnitude.** `magick compare -metric AE` counts a
pixel as differing if any channel differs *at all*, so a pixel off by 1/255 scores exactly like one
off by 255/255. Every screen-difference number this project recorded for the residue class came from
that metric, and the metric was manufacturing the defect.

Two boots each reported ~52 000 "residue pixels" after a rubber-band drag, in a region that turned
out to be precisely the front window's rect plus its drop shadow — a convincing, reproducible,
replicated result. The largest channel deviation anywhere in either frame was **3/255**, and zero
pixels differed by more than 4. It was a re-encode rounding difference. The class had never been
reproduced by the rig at all, and a mechanism had already been written up for it from a census
correlation on one of those boots.

So a difference count needs its magnitude distribution next to it or it means nothing. Report the
max deviation and counts at several magnitudes, and treat a run whose max is a handful of LSB as
**not having reproduced** anything. `.agents/repros/imgdiff.py` does this; prefer it to a bare `AE`.
The same trap applies to any "N pixels changed" claim, including brightness means and region diffs.

Note what saved it and what did not: the sub-perceptual reading came from looking at the *pixels*,
after two boots of counts, a 2x2 design, an interleaved A/B and a null arm had all agreed with each
other. Agreement among measurements that share a metric does not test the metric.

**A channel mean cannot see speckle, and every colour reading here has been a mean.** Per-pixel noise
is roughly symmetric about the value it corrupts, so a wallpaper rendering as dense speckle scores an
utterly ordinary mean. The whole-screen means recorded for the desktop class — R 0.97-0.98 across
four guest clock settings — are all equally consistent with a smooth image, and say nothing about
whether neighbouring pixels agree with each other.

The cheap separator is the residual against a local median, `mean |px - 3x3 median|`, per channel. It
needs no reference frame and does not care about crop, downscale, variant selection or guest clock —
the four confounds that have wrecked every whole-screen comparison this project has run. Measured on
one boot, in the same relative patch of bare wallpaper, at 1280x719:

| desktop picture | dR | dG | dB |
|---|---|---|---|
| PNG extracted from the HEIC, variant 0 | 0.14 | 0.18 | 0.31 |
| PNG extracted from the HEIC, variant 4 | 0.14 | 0.18 | 0.33 |
| stock HEIC, our host window | **0.00** | 7.64 | 18.23 |
| stock HEIC, the guest's `screencapture` | **0.00** | 8.76 | 20.80 |

Sixty times the blue residual, and a red channel that is not merely "pinned at 255" as a range but
*exactly* flat — zero deviation from its own local median. A min/max range cannot state that and a
mean cannot see it. Report the residual next to the mean whenever the complaint is about colour.

**Count what your A/B actually changed.** Swapping the guest's desktop picture from the stock HEIC to
a PNG extracted from that same HEIC makes the corruption vanish, which reads as a clean isolation of
one mechanism. It changes *four*: subsampled YUV becomes packed RGB, Display-P3 becomes sRGB, a
dynamic multi-variant picture becomes a static one, and — the one nobody listed — a 6016x6016 source
becomes one no larger than the display. Each alone explains the result. The one that gets written
down is whichever the reader already suspected; here the YUV reading was about to aim an iteration at
the biplanar sampling path on the strength of a comparison that never isolated it.

Enumerating them and building the grid took one boot and settled it. Every arm below is the same base
pixels at one geometry per row, pinned through the same `desktoppicture.db` path, Dock hidden, scored
by the local-median residual with the stock picture re-run twice as a positive control:

| desktop picture | dB, our window | dB, guest capture |
|---|---|---|
| PNG / JPEG 4:2:0, sRGB or P3, 1920x1080 | 0.19 - 0.31 | 0.25 - 0.68 |
| HEIC, static, sRGB or P3, 1920x1080 or 1920x1920 | 0.08 - 0.11 | 0.13 - 0.17 |
| **PNG, P3, 3840x2160** | **0.31** | **0.68** |
| **JPEG 4:2:0, P3, 3840x2160** | **0.31** | **0.68** |
| **HEIC, static, P3, 3840x2160** | **6.80** | **13.65** |
| **HEIC, static, P3, 6016x6016** | **6.14** | **10.46** |
| **HEIC, stock dynamic, P3, 6016x6016** | **18.30** | **31.94** |

Subsampled YUV: refuted, a 4:2:0 JPEG is clean at both sizes. Display-P3: refuted, P3 is clean in
every row it appears in that does not also speckle for another reason. Dynamic multi-variant:
refuted, a single-image HEIC `sips` wrote in the guest reproduces. Aspect ratio: refuted, 1920x1920
is clean and 3840x2160 is not. Resolution alone: refuted, PNG and JPEG at 3840x2160 are clean.

What is left is an **interaction** — HEIC *and* a source larger than the display — which no main
effect would have found and which every single-knob A/B in this investigation had to miss. When a
swap changes N things, the answer can be a pair of them. Build the grid.

The always-on sink refutes the YUV lead a second time, independently of the pixels. Slicing this
boot per arm, the `type4 pages … multi=1` biplanar `'420f'` lines appear in arms F, G and H — four
each — and in no other arm. **F and G are clean arms.** Biplanar YUV traffic is therefore present
while the screen is correct and absent from every arm that reproduces, which is the converse check:
the line can see multi-plane surfaces, and it puts them on the wrong side.

Nothing else in the census separates the arms either. A first pass found `compute_linear_flush` and
`linear_deferred_flush` at 36 each in the reproducing set and zero in the clean set — until the
per-arm breakdown showed all 72 belonged to one arm and predated every arm window. A set difference
computed over a pooled class is worth exactly as much as the pooling; print the per-arm column before
believing the totals.

So the defect is invisible to every line this device already emits. That is the actionable part:
the next step has to be a probe, and the census is not where to put it.

That is where the measurement stops. Which property of an oversized HEIC decode does it is
**unmeasured**, and no mechanism is named here.

**A transient defect is invisible to a before/after pair.** A repro that captures once before a
gesture and once after cannot see anything that repairs itself in between, and it will report clean
runs indefinitely while a human watching the same screen sees the defect plainly. That happened
here: a scripted gesture sweep scored byte-identical frames at every step while an observer watching
the live window during the same run reported black corruption that had self-repaired by the next
capture.

So when the class involves corruption that comes and goes, capture **continuously** through and
after the gesture and score the burst, rather than sampling its endpoints. Before concluding a
sequence does not reproduce, confirm the sampling rate could have caught it — and prefer the
always-on log, which is continuous by construction and did record the failure the captures missed.

**A live rig is not a live workload — validate the specific thing you drove.** "The log grew and
presents happened" proves the rig is alive. It does not prove the workload ran, and the two get
confused because a healthy-looking log is exactly what a validity check is supposed to produce.

Two escalating drives here — 8 heavy pages with gestures, then 12 pages in separate windows plus 60 s
of sustained animation — each returned zero for the counter under test. A probe-validity check was
run and passed: 1.9 MB and 10 678 lines appended, 797 presents, 383 exports. A per-boot mechanism was
inferred from the zeros and written into the KB and a commit body. **The guest was at the login
window the whole time.** The browser never ran, not one page loaded, and every one of those presents
was the login screen — which renders a full-screen wallpaper and a cursor and drives the present path
perfectly well.

Nothing objected, because nothing was asked: `open -a` is silent with no session, ssh answers, QMP
answers, and the failure channel is genuinely clean when there is no work to fail at. It was caught
only because the *next* experiment needed the browser to be running and said so out loud.

So a validity check must be specific to the workload: for a browser drive that is "the browser is
running", not "pixels moved". This is the dead-VM-scores-`0 px` failure one level up — the guard
written for that proves QEMU exists, and nothing proved a session did. Assert the precondition of the
thing you are driving, before the drive, and abort rather than score it.

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

**Bound that combination on the extremum, not the mean.** A convex combination is taken per pixel, so
if every variant's value at every pixel of a patch is at most V, no blend of them can exceed V
anywhere in that patch. The max is therefore a far tighter test than the average, and it survives
crop and scale. In a fixed bare-wallpaper patch the five desktop-picture variants each render with a
blue channel reaching at most 93/255, while the stock picture reaches 250-255 in the same patch —
refuted by 2.7x, pixel-wise, with no free parameters and no need to know the blend weights. The same
argument on whole-screen means had a margin of a few percent and still had to reason about which
variant was selected.

**Capture the guest's screen next to ours — it is the nearest thing to known input this rig has.**
The host window is our present path's output. The guest's own `screencapture` reads the window
server's composite, which is what the guest believes it is showing. Two captures of the same instant
split the only question that matters for a whole class of defects: *is the guest still publishing
this, or are we still presenting it?*

That split is what finally localized the residue class, after every prior attempt had compared one of
our frames against another of our frames. Eight windows were opened, the app was killed, its process
count was polled to zero, and both screens were captured. The guest's screen was **byte-identical**
to the same round's bare desktop, menu bar reading Finder. Ours held a fully drawn, dead application
window — half a million pixels above 64/255. Three times. The guest had moved on; we had not. No
host-only measurement can reach that conclusion, because a host-only measurement cannot tell a stale
present from a guest that has not repainted.

Two things make the guest arm trustworthy. It is **self-consistent by construction** — the reference
and the measurement are taken with the same instrument in the same round, so whatever `screencapture`
includes or omits (it renders without the Dock here) it does so identically in both. And it is
**independent of our code**, which is the entire point: it is the only frame in the comparison that
our present path did not produce.

Assert the transition you are measuring across. "Killed the app" must mean a process count read as
zero, not a keystroke that was sent. A first hand-run of that sequence quit with `meta_l+q`, the app
did not quit, and the post-close capture scored 391 083 differing pixels — which is "a window is
present", not residue. It was caught by looking at the image.

The same two captures also **rule things out**, in one shot, which is the cheaper direction and the
one to reach for first. A user-reported colour corruption — the desktop picture rendering as dense
per-pixel speckle in saturated red while window content on top of it stayed clean — was localized
this way before any code was read. A pure-wallpaper patch measured, on two boots:

| capture | R | G | B |
|---|---|---|---|
| our host window | 255..255 | 0..27 | 0..255 |
| the guest's `screencapture` | 255..255 | 0..55 | 0..255 |

Two channels pinned to constants and one carrying all the variation, and **the guest's own composite
has it too**. Since `screencapture` re-executes that composite, the wallpaper was already wrong
before anything of ours presented it: the present path, the export path and the host window are all
out, in a single measurement, without a probe. Report the per-channel range rather than "it looks
noisy" — the pinning is the whole finding, and a zoom confirmed no stride banding, so it is a channel
defect and not a tiling or dither one.

Stop the write-up there. Two constant channels and one live one is the same shape as the BT.601 fit
recorded below, and the desktop picture is a HEIC — subsampled YUV, which this rail does carry as
2-plane `'420f'` surfaces. That is a *lead*, not a mechanism, and the fit-then-mechanism trap below
has already cost one iteration on this exact class.

**The guest's screen is an oracle. The guest's memory is not.** The distinction is that
`screencapture` makes the guest *re-execute* the composite; guest memory for a surface we render
into is our own output one step removed, and on a rail that defers the writeback it is not even
that. It is tempting to skip the external capture and read the pages from inside the device, because
that gives a continuous always-on line instead of a scripted round. It does not give an independent
one.

Measured, on x86/Vulkan: a present-boundary probe compared the frame being shown against the
presented surface's own guest pages. Every window read ~2 062 000 of 2 073 600 pixels differing at
full swing, on every present, with a deferred render window armed over the span every time. That is
the deferred-writeback contract stating itself — the pinned resident is authoritative and the guest
window holds pre-dispatch bytes until something reads them — and because the compositor's front
buffer is deferred on every present, the "nothing owed" case that would have meant something never
occurred. Fifty-two windows, zero.

That is the **third** way a probe fails, after "cannot distinguish the cases" and "counts events, not
state": the discriminating condition is unreachable by construction. It is the most flattering of the
three, because the probe fires, produces large confident numbers, and every one of them is the
mechanism you already knew about. Guard against it by splitting the line so the discriminating
subset has its own counter — then "it never fired" and "it could never fire" are different readings,
and a zero denominator says which. A single fused count cannot.

**Instrument the branch, not the arm.** The cheapest version of that split costs nothing and is
almost always skipped: a probe placed *inside* a conditional cannot tell "the condition was false"
from "the outcome never occurred", so its zero is unreadable no matter how carefully the rest of it
was designed. The first line on any suspect path should therefore report **which way the path went**,
before anything reports what happened along it.

That was paid for here, twice, in opposite directions. A six-way resolver on the present path had
three terminal log lines, all silent for many boots, and "absence of a log is weak evidence" was the
correct reading — a branch can be entered and leave through a path that logs nothing. The line that
settled it was `present_route`: two lines per process, naming the branch itself, and 104 boots of
`route=named` with not one `route=clear_only` proved ~300 lines of resolver had never executed. Then
the lesson was immediately un-learned: a careful census was built *inside* that same branch, with a
split denominator and a test pinning its identity rule, and it reported nothing — because its branch
does not run. Grepping the always-on sink for the lines already emitted on that path would have cost
one command and saved the commit.

So before adding a probe, grep the sink for what the path already emits, and if nothing there names
the branch, make that the probe. Prefer a line whose *dedup key is the branch taken* — it is bounded
at one line per outcome per process, which is what makes it safe to leave on forever, and it keeps
answering for every boot after the question that prompted it is closed.

**Establish what each arm of a comparison renders before trusting it.** An arm that omits a region
does not report "no difference" there — it reports nothing, and the other arm's difference is then
unopposed. The guest-screen comparison above has exactly this hole: `screencapture` renders without
the Dock and our host window shows it, so any Dock change scores as a host-only difference with a
byte-clean guest arm. That is the same reading as "the guest moved on and we did not", produced for
free, in a region the design cannot see.

It is not hypothetical. Three "residue reproductions" across two boots were a 61-65 px strip whose
top edge sat within 4 px of y=616; cropped, the strip is the Dock, and the guest capture at the same
coordinates is bare wallpaper. Launching and quitting apps changes the Dock — its running-apps
section grows and shrinks and every icon shifts — so the artifact fires on the exact transition the
repro is built around. It passed the magnitude rule (max 255, tens of thousands of pixels above 64),
the two-arm rule, and the process-count assertion. It failed only when the pixels were looked at.

Fix by removing the region from **both** arms, not by masking it out of one: the repro now hides the
Dock before its first round and aborts if it cannot. Masking needs a hard-coded rect and leaves the
comparison partial; hiding keeps it whole-frame.

Note which way this cuts. The two-arm design is still the strongest tool here — it is what localized
the class at all. But "the guest's arm is independent of our code" is a statement about *causation*,
not about *coverage*, and it says nothing about regions the guest's instrument does not draw.

**A reproduction rate pools whatever you called the same defect.** "Reproduced 3 of 6" is a number
about a name, and the name was chosen before anything was measured. Print the per-instance shape next
to the rate and look at whether it is one population.

Measured here on one six-round run: three rounds reproduced the residue class, and their bounding
boxes fall into two groups that share no dimension. One lost a whole window (1251 px wide, 534 tall).
Two lost a strip 61-65 px tall whose top edge sat at y=614-618 — and a third instance of that strip,
from a different boot, sat at y=618. A position that repeatable across boots is not the same event as
losing a window; pooling them into a single rate had been hiding a second sub-class for several
iterations, and a fix validated against the pooled rate could move one population and be scored by
the other.

So report the distribution, not just the count, and be suspicious when a "class" has instances that
differ by an order of magnitude in size. The corollary for a fix: state which sub-class it addresses
and score that one.

**Deleting a consumer makes its producer's waste audible — listen on the next boot.** When a mechanism
comes out, the state that fed it usually does not, and it starts reporting. Here, removing the
resolver left the present↔store pairing queue with nothing draining it: `present_store_fifo_drop` had
zero occurrences across 141 boots and fired 1750 times on the first boot after, saying in its own
always-on text that entries were ageing out unpaired. That is not a regression to fix, it is the next
deletion announcing itself. Boot once after any deletion and diff the always-on line census against
the boot before — the new lines are the work list, and the ones that vanish are the confirmation.

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
