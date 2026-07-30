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

**A census is not a decline, and the difference is whether anything else reports the loss.** This
codebase grew a large family of "always-on proxies" that summarised how often each path ran, and the
rule above is what they cited for existing. Most of them were not reporting losses at all. The test
that separates them, applied to twelve such modules, is one question: *if this census were deleted,
would a dropped guest command become invisible?*

- If the refusal already emits a typed decline at the point it refuses, the census only tracked its
  *rate*. `present_import::note(false, true)` sat on the line directly above
  `Emit::decline("import_present", &d).fail()`. Delete the census; the decline is the report.
- If the census is the only record, it must stay — or better, become a typed decline.
  `window_publish::note(false)` is the sole evidence that a captured frame never reached the window,
  and its own comment says the interesting signal is a sustained run. Deleting that manufactures the
  silent failure this section forbids.
- A tally of *successful* work — residents released by the idle sweep, teardowns counted, which
  source served a capture, microseconds per sub-step — never had a claim under this rule.

Two tells make this cheap to check. A census that carries no typed decline slug is almost always a
tally: the crate's registered-slug count did not move across four separate commits that removed nine
such modules — nine censuses deleted, zero refusals lost with them. (That count was read off a
`#[cfg(test)] REGISTRY` table in `observe/decline.rs`, since deleted; the equivalent reading today is
`observe::gate::declared_slugs().len()`, which scans the `Decline`/`Refusal` impls directly.) And a
line whose text says `ok`, `stage_ok`, `resident_ready=1` or `retain` is narrating the success path,
whatever sink it was written to.

**Cost hides behind "measure-only".** The proxies here were not free, and the comments admitted it
in situ: "2-3 ms per display-sized storage image on the stamp path", "the full-frame stats scan
exists only for the verbose line". Removed in this family: a GPU compute reduction dispatched every
present, a SIMD census fused into every readback, an O(w·h) scan of every bound texture on every
draw, and a full read-back of the guest window on every Store. Before writing `// Measure-only` on
something, price it at the rate it will actually run.

### A Table That Restates The Code Adds No Invariant

The census family above is one shape of safeguard-that-isn't. The other is a hand-maintained table
that mirrors what the code already says, plus a source scanner whose job is to check that the mirror
still matches. It reads as rigour, and it can only ever agree or disagree with its source: agreeing
adds nothing the source did not already carry, and disagreeing gets reported as "the table drifted"
rather than as a defect in the code.

The one that was here was 2 746 lines. `observe/decline.rs` carried a `#[cfg(test)] REGISTRY` naming,
for each of 67 decline types, its defining file, its `Emit` call sites, its delegate impl blocks and
all 1 425 of its slugs as literals; eleven of `observe/gate.rs`'s eighteen tests existed to lex the
crate and confirm the table. The properties it enforced were real, and *all but one of them were
already enforced by the `slug()` arms it was copying*.

The exception is the test worth keeping, and it is the tell for which part of a mirror to save: **the
property no single site can see.** Slug uniqueness is crate-wide, so nothing in `translate`'s impl can
notice a collision with `engine`'s. That one now reads the impls directly —
`gate::declared_slugs()` anchors on each `impl Decline for` / `impl Refusal for` and then on the
`fn slug` body inside it (not the whole impl: `fields()` keys are lowercase snake_case too), returning
644 pairs. A scan of the code cannot drift from the code, and it needs no baseline.

Two costs are worth naming because they are what make this class expensive rather than merely large:

- **A hand-bumped baseline taxes every deletion.** `the_registry_is_what_the_last_migration_recorded`
  pinned `(66, 1425)` and had accumulated forty lines of changelog prose inside the test body
  explaining the last three bumps. Any commit that removed a decline had to edit the table, re-count,
  and justify the direction. That is a toll on exactly the work this file asks for.
- **A mirror hides what it does not list.** `every_registered_type_reaches_the_sink` checked that each
  registered type named a real `Emit::decline(` site — a claim about the table's accuracy, not about
  the code's coverage. A decline type with *no row at all* was invisible to it, and one was:
  `BlitOptionError`, which the registry's own trailing comment admits it declined to certify.

So when a table and a scanner appear together, ask which properties survive deleting the table. Keep
the ones no single site can see; delete the rest, and say plainly which coverage went with them. Note
what does *not* qualify: `translate/coverage.rs` looks like the same shape and is not. It enforces
that every `pub` field under `runtime/decode/` has a *disposition* — completeness over a set the code
cannot enumerate about itself — and its 23 `DroppedSilently` rows are a defect list under a
shrink-only ceiling, not a restatement. Audited and kept.

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

**A measurement that a branch reads has stopped being a measurement.** `runtime/census/README.md`
states the rule outright — "Nothing in the device or backend may read one of these back to decide what
to present, decode or execute. A proxy that changes behaviour has stopped being a proxy and become a
content heuristic, which the ground rules forbid outright." `metal_draw/vulkan.rs`'s sampled-source
resolve broke it, and the shape is worth keeping because it is how this class hides:

```rust
if let Some(bgra) = crate::runtime::surface_cache::get(state, mid, w, h) {
    let rgba = swap_rb_channels(bgra);
    let (nz, _, _) = crate::observe::rgba_rgb_stats(&rgba);
    if nz > 0 || !resident_ready {
```

`rgba_rgb_stats` is one of the sink's whole-frame pixel scanners and `nz` is a count of non-black
pixels, so *which image gets bound to a sampled texture* depended on whether the host cache's copy
happened to contain one — an O(w·h) scan per bind whose result is a branch, on content, which is
exactly the two things forbidden. "All black" is also a legal frame, so it mistook a correct black
surface for an empty one.

**Deleted 2026-07-30, and it needed no boot to retire — because it could not fire.** `resident_ready`
is false at that line on both backends: under `backend-vulkan` an `if resident_ready { return
Target(...) }` twelve lines above returns unconditionally, and under `backend-metal` `resident_ready`
is bound to a literal `false`. So `!resident_ready` was always true, the disjunction was always taken,
and `nz` was computed and thrown away. The identical gate on the *guest-pages* branch below had
already been reasoned out and removed, with a comment stating the argument; this copy sat three lines
above it and kept paying for the scan.

Three transferable points, in order of how much they cost:

- **"Cannot be scored without a boot" was wrong, and it is what kept this alive for several
  sessions.** The entry here previously said "not changed, because it selects what gets bound and this
  rig cannot boot to score the change". A heuristic that cannot execute needs no experiment — reading
  twelve lines up settles it. Before deferring a removal to a boot, check whether the branch is
  reachable at all.
- **A scan whose result is consumed by an `if` is not findable by grepping for census modules.** It is
  spelled as an ordinary function call on the hot path. The "measure-only" audits walked past this
  line several times.
- **A dead disjunct hides its own cost.** `nz > 0 || …` reads as a cheap short-circuit; the expensive
  part is the statement above it, which runs whatever the condition does.

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

Bisecting the size on static HEICs puts the step at the display width, not at any texture or codec
limit, and shows it is a step rather than a ramp — the magnitude barely moves across a 3x range of
scale factor after it fires:

| static HEIC source | 1920x1080 | 1920x1920 | 2048x1152 | 2560x1440 | 3072x1728 | 3840x2160 | 6016x6016 |
|---|---|---|---|---|---|---|---|
| dB, our window | 0.11 | 0.11 | **4.40** | **5.88** | **7.12** | **6.80** | **6.14** |

41x between 1920 and 2048 wide, then flat. The obvious reading — "the clean arm is the one needing no
downscale, so any downscale does it" — is wrong, and bisecting further is what caught it:

| static HEIC source | 960x540 | 1920x1080 | 1920x1920 | 1921x1081 | 1984x1116 | 2047x1152 | 2048x1152 |
|---|---|---|---|---|---|---|---|
| dB, our window | 0.13 | 0.11 | 0.11 | 0.11 | **3.70** | **4.15** | **4.40** |

1921x1081 needs a downscale and is clean, so it is not "any downscale". An *upscale* from 960x540 is
clean too, so it is not scaling in general. The step sits between 1921 and 1984 wide — not at 2048,
so not a power of two, and nowhere near a 4096 texture limit. It is also not pixel count: 1920x1920
is 3.7 MP and clean while 2048x1152 is 2.4 MP and speckles, which makes the governing dimension the
width and not the area.

Grid tiling is not the discriminator either: every one of these files is a tiled HEIC, including the
clean 1920x1080 one, so "it got tiled above some size" does not survive reading the files.

The useful product of the bisection is not the threshold — that is a guest-side decode policy, not
our contract, and pinning it to the exact pixel would be overfitting. It is the **pair**: 1921x1081
and 1984x1116 differ by 3% in every dimension and land on opposite sides. Anything that differs in
the traffic between those two arms is the defect; almost everything else is held constant by
construction. That is the tightest differential this class has, and it is where a probe goes.

**Then read the corrupt pixels at native resolution, and the shape of the defect changes.** Every
capture up to this point was our host window, downscaled to 1280x719 or 640x360, and at that scale
the defect reads as dense speckle over the whole picture. The guest's `screencapture` is native
1920x1080. On it, the same frame is *mostly correct*, with **63 816 outlier pixels — 3.08%** —
isolated against clean surroundings. "Dense speckle" was the downscale accumulating sparse errors.

Two measurements on that native frame close out the family of explanations this investigation spent
all its time in:

- **No grid alignment.** Outlier positions are uniform in `x%2`, `y%2`, `x%4` and `y%4` — 25.0 /
  25.0 / 25.0 / 24.8 % in the four `(x%2, y%2)` cells, and 24.9-25.3 % across the mod-4 bins. Chroma
  subsampling, block artifacts and tiling at those scales all predict a strong bias and there is
  none.
- **No plateau at 2x2.** Residual against a block mean grows smoothly with block size — 0, 5.18,
  12.1, 19.0, 24.7 for 1x1 through 16x16. If the error lived on a 4:2:0 chroma block the 2x2 figure
  would be near zero; it is a third of the way to the 16x16 value.

So the corruption is sparse, per-pixel, uniformly scattered, and each affected pixel is wrong by a
large margin rather than slightly off. That is not the signature of a format conversion, a
subsampling ratio or a tile boundary — the three things every hypothesis in this class has assumed.
Measure the geometry of the wrong pixels before assuming a transform produced them.

It is also **fully deterministic**. Three `screencapture` frames taken four seconds apart, with
nothing driving the guest, have the identical 63 816 outliers — the masks differ in zero pixels and
the frames are byte-identical, max channel delta 0. So it is not a race, not a read-during-write and
not a timing window; the wrong value is computed once and then presented consistently.

And the error rate is a **strong function of the correct value**, not of position. Binning the
outliers by what the pixel should have been:

| correct B | 0-15 | 16-31 | 64-79 | 112-127 | 128-143 | 176-191 | 224-239 | 240-255 |
|---|---|---|---|---|---|---|---|---|
| outlier rate | 1.6% | 38.0% | 55.1% | 59.7% | 54.8% | 42.2% | 23.0% | 6.1% |

Near-immune at both extremes, 20-60% wrong through the middle. The whole-frame rate is only 3%
because 93% of this picture sits in the bottom bin. A defect that spares 0 and 255 and peaks in the
mid-tones is a different animal from one that corrupts uniformly, and it is the first result in this
class that points at *what* is being computed rather than what it is being computed from.

The channels are hit at different rates — in the same 512x512 crop, B 2.70% and G 0.98%, with G's
value dependence weak and non-monotone where B's is strong. **Do not read R's zero residual as "R is
correct".** R is pinned at 255 across this whole picture, so an error in R has nowhere to go and
would clip back to 255 regardless; a saturated channel cannot report. The honest statement is that B
is corrupted about three times as often as G and that R is untestable on this content. Testing it
needs an image whose red channel is not already at the top of its range.

**A per-pixel join is a join — check the lengths.** The first run of that table came out flat, 0.2-1.4%
in every bin, which reads as "no value dependence" and would have closed the question the wrong way.
The two pixel dumps were `paste`d together and one of them had exactly twice as many lines, because
the extraction matched both `(0)` and `gray(0)` on each row. `paste` does not complain; it silently
pairs each value with an unrelated pixel, and randomly-paired data produces a flat rate *by
construction*. The null result was manufactured by the join, and the corrected table is not
marginally different from it — it inverts it. Print both input lengths next to any per-pixel
correlation before reading the correlation.

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

**One candidate was found by reading code rather than pixels, and it is now REFUTED. Do not
re-nominate it.** `compute_exec`'s `try_recover_sentinel_grid` fired on a `DispatchThreadgroups` whose
wire read `grid = [ceil(ow/tg), u64::MAX, 1]`, `tg = [32, 0, 1]`, and **substituted dispatch dimensions
the guest never sent**: both grid axes from "the largest write-capable bound texture by pixel area",
with `tg.y` set to `tg.x` on the assumption the tile is square. Its doc comment named the driving case
as the Live Core Image wallpaper, and it fit the measured speckle signature better than anything else
— errors that are sparse, per-pixel, uniformly scattered with no grid alignment, deterministic, and a
function of the value being computed — because a wrong grid makes every thread compute its coordinates
wrongly.

It was also invisible: its only report was `observe::line`, the `REIMS_VGPU_DRAW_LOG=1` tier, so every
boot that took the path said nothing on the channel all the census work above was read from. Moved to
`observe::off` (always-on, deduped per invented geometry) so the question could be asked.

**Asked and answered on one x86/Vulkan boot, 2026-07-30. It fires zero times.**

| | |
|---|---|
| desktop picture | `/System/Library/CoreServices/DefaultDesktop.heic` — the stock dynamic HEIC, i.e. the reproducing arm |
| workload | desktop revealed via cmd-H, Safari on apple.com and testufo.com, Calendar, System Settings, Mission Control |
| slice | 12 841 lines, 2 153 `present_content`, 848 `compute_linux` (302 `storage_access`), 746 `drain_duty` |
| `compute_sentinel_recover` | **0** |
| `compute_grid_dim_range` / `BadGrid` | **0** |

The zero is readable, which is the part that took care to arrange. `resolve_dispatch_dims` called
`try_recover_sentinel_grid` on **every** `DispatchThreadgroups` before `u32_dim`, so entry was
unconditional — this is "instrument the branch, not the arm" satisfied by construction, and the zero
therefore means "no dispatch carried that shape", not "the branch was never reached". Zero `BadGrid`
beside it says no dispatch came near the range check either. Two screenshots confirm the guest rendered
correctly throughout, so it is a healthy boot rather than a rig that failed to drive the case.

So the heuristic is **deleted**, along with `largest_bound_texture_dims`, `ceil_div_u64` and its three
constants, which had no other caller. The gap it covered is now the typed
`BadGrid("compute_grid_dim_range")`, and its own claim — "without this every wallpaper VTMTS/CI
dispatch hits `BadGrid` and the desktop stays black" — is refuted twice: no dispatch reaches that
check, and a second boot at the deleting commit brings up wallpaper, Safari on apple.com, Dock and menu
bar with `BadGrid` still at 0. The speckle class is back to having **no named mechanism**.

Two things this cost that are worth carrying forward:

- **The test that covered the heuristic asserted it does not fire.**
  `sentinel_grid_recovers_from_largest_texture` was named for recovery and its own comment admits the
  object-list setup was never written, so its single assertion was `.is_none()`. A test named for the
  behaviour it does not exercise is worse than no test: the grep for coverage finds it. Replaced by
  `the_zero_threadgroup_wire_shape_is_refused_by_name`, which drives the exact wire shape past a bound
  1440x1080 write target and asserts the typed refusal.
- **A "fits the signature" argument is worth one probe, not one iteration.** This one fit better than
  every hypothesis the format/subsampling/tiling family produced, and it was still wrong. The probe
  that killed it was a one-line change of output channel plus a grep.

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

**A frame is not a present: `present_content` counts mappings.** Frame rate is the headline number
for this device and the obvious line to derive it from is wrong by construction. `present_content`
is emitted once per *mapping*, and the guest composites several layers per frame, so a gap histogram
over it reports a fast tail that no frame ever had.

The tell is one command. Split the gaps by whether consecutive lines carry the same `mid=`:

| gap | same `mid` | different `mid` |
|---|---|---|
| < 20 ms | **0** | 328 |
| ≥ 20 ms | 4 | 2114 |

Every single sub-20 ms gap was between two different mappings — the layers of one composite going
out back to back — and no mapping ever re-presented that fast. Read ungrouped, that boot is 2.9 Hz
with a floor of 2 ms, which invites "the present path can do 500 Hz". Grouped into frames on a 20 ms
burst boundary it is **2.06 Hz**, and the fastest any single layer ever repeated is 148 ms.

So group bursts before quoting a rate, and print the distinct `mid` count next to it: if that drops
to 1 the grouping is a no-op and the two numbers must agree. `.agents/repros/soak.sh` does this.

**The ~2 Hz frame rate is FOUND and FIXED — see "The ~2 Hz Frame Rate Was One Bit In A Memory-Type
Query" below.** It was `MemoryClass::Readback` requiring `HOST_COHERENT`, which discarded the
`HOST_CACHED` preference on every driver that does not carry both, so every full-frame readback was an
uncached memcpy at 460 MB/s. 1.49 Hz → 18.76 Hz. Everything in the rest of this section was measured
correctly and its *attribution* is what to read carefully: the bytes were real, the duty cycle was
real, and "it is the draws" was true — but the draws were slow for a reason none of it names, because
none of it divided `readback_bytes` by `readback_us`. The remaining paragraphs stay because the
deferred-rail analysis and the refutations below are still live.

**Three things are already refuted as the cause of the ~2 Hz frame rate, each from the always-on
channel.** Do not re-derive them:

- *The present path.* A whole five-layer composite goes out in ~20 ms and per-layer gaps bottom out
  at 2 ms. It is not the limit.
- *VBL delivery.* This was unmeasured until `display_vbl` was added, and the guess was that we were
  starving the guest's display link. We are not: it reads `window_hz=125.0` against a 125 Hz grid,
  steady, for the whole boot. The guest is being ticked 125 times a second and produces two frames.
- *Shader translation stalls.* `exec_translation_deferred reason=air_loading` fires 115 times across
  680 s — 0.17/s against 2 frames/s. Occasional compiles, not a per-frame cost.

Resist the two counters that look like they answer it: `contig_view_fragmented` and
`type4_pages_refreshed` both land near one event per frame in aggregate, but they fire on
*re-derivation*, not per access — `mid=2` accounts for 59 events in 1124 s — so neither can price
the path it names. That is the "an event count is not a state" rule applied to a cost rather than to
a state.

**Where the time does go is now measured, and it is one number: ~360 MB/s of CPU↔GPU copying.**
`drain_duty` reads the drain worker's duty cycle, which is the right frame to ask in — that worker is
the device's only executor, holds the device lock for a whole tranche, and therefore caps the guest's
composite rate at the rate it finishes tranches. Under Safari load it sits at **duty 0.93–0.99**, so
the rate is ours and not the guest's own pacing. Three readings follow from the same line and each
kills a hypothesis that had looked strong:

- *The host-window export is not the cost.* `publish_us` is 2–10 ms against a `drain_us` of
  800–1500 ms — about **0.5%** — even though `export_present_dmabuf` quiesces the whole GPU twice per
  present (`begin_entry_sync` then `retire_all`). Removing those two quiesces is a real cleanup and
  worth roughly nothing; do not spend a session on it expecting frames.
- *The machinery around the work is not the cost.* Idle windows read `duty=0.001` across 210+
  tranches per second.
- *It is the draws.* `draw_us` is **96–99%** of `drain_us`, at 150–840 draws/s and **1.5–7 ms each**.
  Compute and flush are noise (`compute_us=0`, `flush_us` a few percent). The guest issues hundreds of
  draws per composite, so ~2 s of work arrives per second of wall clock.

`engine_delta` then names the per-draw cost, and it needed **no new instrumentation** —
`engine::counter_snapshot` maintained seventy-odd counters for the life of every boot and had no
product caller, so nothing had ever read one. In one second:

| | per second |
|---|---|
| `readbacks` / `readback_bytes` | 20 / **165 888 000** |
| `seed_uploads` / `seed_upload_bytes` | 30 / **190 918 800** |
| `creates` | 30–340 |
| `pipeline_misses`, `shader_misses`, `target_evicts` | ~0 |

166 MB/s back plus 191 MB/s out. And `readback_bytes / readbacks` is exactly **8 294 400 =
1920·1080·4** — every readback is a whole framebuffer, not a tile or a dirty rect. The seed side is
the mirror image: `exec.rs`'s staging block says in situ that it is "Target seed staging (CPU import
only — **not LoadFromTarget**)". So each render pass uploads its target's prior contents from host
memory, draws, and reads the whole target back.

Note which way the cache counters cut. `pipeline_misses`, `shader_misses` and `target_evicts` are all
~0, so this is **not** churn or thrash — the caches are working and the round trip is what the code
does on the *hit* path. A hypothesis of the form "something is missing its cache" is refuted by the
same line that shows the bytes.

**The route that pays it is `cpu_portability`, and it is the one route with no deferred rail.**
`store_routes` counts the guest-Store routing decision per second, next to `engine_delta` in the same
window. Under load, and with `readbacks` alongside:

| `cpu_portability` | `gva_deferred` | `readbacks` | readback MB | seed MB |
|---|---|---|---|---|
| 60 | 30 | 80 | 196 | 188 |
| 67 | 29 | 86 | 191 | 192 |
| 111 | 48 | 155 | 249 | 236 |
| 28 | 38 | 32 | 180 | 180 |
| 43 | 47 | 66 | 242 | 241 |

Two things to read off it. `readbacks` tracks `cpu_portability` almost one-for-one while
`gva_deferred` adds few — that is the deferred rail *working*, taking no readback at Store time and
paying one only if the window is later flushed. And `seed MB` equals `readback MB` in every row, so
the traffic is symmetric: each pass uploads the target's prior contents, draws, and reads the whole
target back.

The asymmetry is structural, not incidental. `gva_store_defer_eligible` opens with

```rust
if c0.mapping_id != 0 || c0.target_gva == 0 || c0.row_stride == 0 {
    return false;
}
```

so a **type-11 composite Store can never defer** — it always reads back and CPU-copies into the
mapping's guest pages. That is precisely the hole the host-pointer import used to cover, and
`metal_draw/vulkan.rs` says so where it sets the flag: "the import is gone, so the only way a Store's
pixels reach the guest is the CPU writeback, and that needs them read back."

So the actionable shape of the ~2 Hz problem is to **give the type-11 render Store a deferred rail**.
This section used to say how: keep the composite on the registry resident with `skip_readback`, arm a
flush-on-access window, and let the matching Load take `LoadOp::LoadFromTarget` instead of a CPU seed
— "so a type-11 window would suppress the seed for free".

**That was built, booted, and is refuted. Do not rebuild it.** The screen came up black with orange
fragments, and one boot logged **2374** `chain_resident_land_fail` and **816** `deferred_flush_lost
kind=render`, every one `reason=read_target_no_ready_content`, with `target_evicts` at 134-281/s and
the cadence reader scoring "too few frames to read".

The tree already said why, in a comment sitting on the exact line the change re-enables (the "Type-11
Load used to have a GPU rail here" note in `metal_draw/vulkan.rs`): a type-11 `LoadFromTarget` has to
resolve *which resident holds the frame the guest's compositor computes its damage against* — the
presented front's own resident, this target's, or the guest pages. The guest ping-pongs its front
buffer between mappings (that boot: mid 1 and mid 4, both 1920x1080), so a LOAD on one surface
routinely wants the other's content. That resolve was ~170 lines of front-frame retention policy,
deleted with the import rail, and making the resident authoritative silently requires it back. Nothing
about it is free.

Two facts that attempt established, and which still hold:

- `engine::read_resident_bgra` returns `None` unless `slot.bgra`. A resident-authoritative type-11
  rail therefore *needs* `req.output_bgra` — that is why it is listed below as part of the fix, and it
  is a hard requirement rather than an optimization. (This bullet used to cite
  `export_present_from_resident_fd_policy` refusing with `PresentExportResidentNotBgra` for the same
  reason; both were deleted with the dmabuf rail on 2026-07-30 and `read_resident_bgra` is now the
  only gate of this shape.)
- The mapping-keyed flush-trigger backbone described below is real and does work. It is what the
  shipped rail is built on.

**What shipped instead: defer the writeback, not the readback.** Keep the readback and `surface_cache`
exactly as they were, and defer only `mapping_write::write_rgba8_image_changed` — the per-row
RGBA→native conversion and the fragmented scatter into guest pages, ~8 MB of CPU work per Store at the
28-111 Stores/s `store_routes` measures. The Load seed, the present capture and the chain
intermediates are then untouched, which is precisely why the front-buffer problem above does not
arise. The flush reads the frame back out of `surface_cache` and writes it with `write_bgra8`; no pin,
no `content_ready` to hold across frames, no identity to resolve. That boot: screen correct, CLEAN,
`chain_resident_land_fail` **0**, `deferred_flush_lost kind=render` **10**, `target_evicts` **0**, and
`cpu_portability` gone from `store_routes`.

Two ordering defects in that rail were found by re-reading the diff rather than from a boot, and both
are worth knowing because they are the generic hazards of any deferred rail:

- **Superseding must drop, not flush.** The arm originally flushed everything intersecting its own
  guest range before arming. A compositor painting one surface re-Stores the *identical* range every
  frame, so the previous window always intersects — landing it there performs exactly the write the
  rail exists to skip, once per Store, and the rail becomes a rescheduling with extra steps. Use
  `supersede_gva_window`'s rule: a window the new Store *fully covers* is dropped. Sound for the same
  reason as on the GVA rail — those bytes were never observable without a flush, since any reader
  would have taken the window first.
- **Refresh the cache after the flush that reads it.** The flush sources its pixels from
  `surface_cache`, so storing the new frame first makes an older window at a different geometry miss
  the cache and report a loss for a window that was perfectly landable.

**That second one was a symptom, and fixing the ordering only narrowed it. The window must own its
pixels.** `surface_cache` holds exactly **one entry per mapping**, and a window is armed against a
*geometry*. So any later Store at a different size replaces the entry the older window was pointing
at, and no ordering rule inside the arm can help — the two events are frames apart. Measured on one
boot, after the ordering fix: **15** `deferred_flush_lost … reason=cache_miss`, every one
`kind=render`, and the geometries name what was lost — a 1920x1080 desktop surface, a 1920x24 menu
bar, a 1225x70 toolbar, several window-sized rects. Two mappings appear in that list *twice at two
different geometries*, which is the mechanism stating itself.

On screen that is a whole compositing layer rendering **solid black**. The user-reported screenshot
for this class is the tell and it is worth reading the shape rather than the pixels: every black
region is a sharp **axis-aligned rectangle at a layer boundary** — the desktop wallpaper, a Safari
tab-bar strip, a window content sublayer, while the window chrome around it renders perfectly. That
is whole layers failing to land, and it rules out the entire per-pixel family (format conversion,
subsampling, tiling) before any code is read.

`DeferredOwner::Render` now carries an `Arc<Vec<u8>>` of the frame it deferred, shared with the
cache entry the same readback stored — a refcount at arm time, no copy. `cache_miss` is not a
narrower failure now, it is unreachable. It also *dissolves* the ordering rule above rather than
restating it: the arm no longer cares whether it refreshes the cache before or after landing
intersecting windows, because those windows are not reading the cache.

Generalise it. **A deferred obligation that names its data indirectly is only as durable as the
indirection**, and every cache in this tree is keyed loosely enough to be replaced under a live
window. Prefer owning the bytes; with `Arc` it is free.

**The symptom was then caught live and correlated, and the correlating line needed no new probe.**
A user watching a boot reported the desktop background *oscillating* between half-black and correct.
That boot was still on the pre-fix binary, so it is a reproduction rather than a survival, and the
whole correlation came out of `present_content` — which carries `rgb_nz`, a count of pixels with
max(B,G,R) > 0, i.e. a black-fraction proxy for the frame actually presented. Sliced per `mid`:

| mid | presents | mean non-black | lost a Store? |
|---|---|---|---|
| 4 | 253 | 95.2% | no |
| 6 | 268 | 79.8% | yes — `1920x1080 reason=cache_miss` |
| 37 | 175 | 80.6% | yes — `1920x1080 reason=cache_miss` |

mid 4 is under 10% black in 238 of its 253 presents. Mids 6 and 37 spend 84 and 54 presents in the
20-70% band. The guest ping-pongs its front buffer `4 → 37 → 6 → 4`, so the screen alternates between
a clean mapping and two that lost pixels — which is exactly "oscillating between half black and not",
and is why a single screenshot of this class can look fine.

The timing closes it. mid 6 lost its Store at `t=68750` and its first 61%-black present is `t=68838`,
88 ms later; mid 37 lost at `t=68786` and presents 61% black at `t=68840`, 54 ms later. Each within
one frame of its own loss.

Two process points. First, `present_content`'s `rgb_nz` was already being emitted on every present
and nobody had ever sliced it per `mid` — the "identify comes before add" rule paying for itself, and
the answer cost one `awk`. Second, note what this is and is not: a per-mid correlation plus a
one-frame lead is strong, but it is still a correlation, and the fix is not scored by it. The fix is
scored by `cache_miss` being **unreachable by construction** once the window owns its pixels.

Do not read the per-mid means as a standing exclusion of mid 4 either. It did not lose a Store *in
that boot*; other boots lost full-screen Stores on other mids, and which mapping the compositor picks
is not ours to predict.

**Scored on a boot at the fix, with the same workload: `cache_miss` 9 → 0, and the oscillation is
gone.** `.agents/repros/blacklayer-score.sh` reads both boots the same way. The fix boot did **6.4x
more deferred work** (`surface_deferred` 33744 against 5239), which matters — a loss count that falls
while the denominator rises is not a quieter workload.

| | pre-fix (`73648aa`) | at the fix |
|---|---|---|
| `deferred_flush_lost` total | 14 | 7 |
| … `reason=cache_miss` | **9** | **0** |
| … `reason=map_generation_drift` | 5 | 7 |
| `surface_deferred` | 5239 | 33744 |
| presents 20-90% black, per mid | **82 / 54 / 3** | **3 / 3 / 3** |
| mean black fraction, per mid | 20.2% / 19.4% / 4.8% | 4.0% / 4.0% / 3.4% |

The per-mid row is the one that maps to what a person sees. Before, the two mids that had lost a
full-screen Store presented partly-black frames 82 and 54 times while the clean mid did so 3 times;
after, **every presented mid behaves like the clean one did**. The host window at the end of that
boot renders wallpaper, Calendar, Finder, Dock and menu bar with no black rectangle anywhere.

The 7 remaining losses are all `map_generation_drift`, which is the guard doing its job — the pages
moved under the window and writing them would land a framebuffer in whatever owns that memory now,
which is the corruption class. That number going *up* while `cache_miss` goes to zero is expected:
those windows previously died of `cache_miss` first.

**One boot does not close the class**, per the standing rule here. What the boot establishes is that
the mechanism is gone from the path it was measured on; what actually retires `cache_miss` is that it
is unreachable once the window owns its frame, which is a construction argument and does not depend
on this boot.

Two things this boot does **not** show, stated so they are not quoted later as though it did:

- **No frame-rate claim.** Cadence read 5.85 / 8.35 / 11.86 Hz per round against a pre-fix band of
  5.3-14.4 across eight boots, with the third round always fastest in both. That is inside the noise,
  which is exactly what `us_per_draw`'s 1.8x boot-to-boot spread predicts. Scoring the CPU-copy
  removals needs interleaving or a counter, not this.
- **`read_overrun` fired 0 times.** The guard added for the unbounded contig read is therefore
  untested by this boot in the only way that would matter — it has no positive evidence that the
  overrun was ever being taken. Treat it as a bound that is now enforced, not as a defect that was
  observed.

The counter that separates those two worlds is `surface_flush` on the `store_routes` line: an arm
count cannot tell "the writeback was skipped" from "the writeback happened a millisecond later", and
`surface_flush / surface_deferred` is the ratio that can.

**That ratio has now been read, and the rail is a real deferral: 0.138.** One 4-round soak boot
(Finder + Calendar + Safari on wikipedia/apple, desktop drags, session asserted) summed
`surface_flush=2555` against `surface_deferred=18503`. So 86% of type-11 guest-page writebacks are
never performed at all, rather than performed a moment later. Note this needs no interleaving and no
A/B: it is a ratio of two counters inside one boot, which is why it survives the `us_per_draw` drift
that makes every cross-boot comparison on this rig worthless.

Do not do this by restoring `VK_EXT_external_memory_host`. The deleted `import_present` rail had an
"ack-fast deferred rung" that looks like the same idea, but what it needed the host pointer for was
the eventual *DMA*; the deferral itself did not. The flush here is the CPU writeback that already
exists (`mapping_write::write_rgba8_image_changed`) — the win is doing it once on demand instead of
~70 times a second unconditionally.

**The hard part of that is already built, and it is the flush-trigger backbone.** The reason a
deferred rail is dangerous is that every reader of the deferred pages must be made to flush first, and
a missed reader means the guest silently reads stale pixels. For type-11 mappings that set is already
closed and already wired, because the *compute* rail keeps mapping-keyed deferred windows
(`ComputeStorageResidencyKey`'s "Surface window (`mapping_id != 0`)" kind) and every guest-page reader
already drains them through `storage_flush::flush_intersecting(mapping_id, lo, hi)`:

| reader | flushes first? |
|---|---|
| `mapper::write_mapping_bytes` | yes |
| `mapper::read_mapping_bytes` | yes |
| `flush_mapping_for_guest_read` (guest `SynchronizeResources`) | yes |
| `scanout::capture_display_frame` | yes |
| `flush_intersecting_task_gva` (raw task-GVA aliasing the mapping) | yes |
| `mapping_write::read_rect_raw_at` | yes — **but only after 2026-07-29; see below** |
| mapping teardown / unmap / delete-backing / replace-physical | drops the window (pages unreachable) |

**That table was wrong, and the way it was wrong is the reusable part: a function is not a choke
point if its own branches disagree.** `read_rect_raw_at` has two paths. The fragmented one ends in
`mapper::read_mapping_bytes`, which flushes and is on the list above. The contiguous one is a raw
`copy_nonoverlapping` out of the mapped span and flushed nothing. So whether a type-11 surface read
observed a deferred Store depended on **whether its guest pages happened to be contiguous** — which
is not a property anybody was reasoning about, and is exactly the kind of condition that makes a
defect intermittent and unreproducible.

Three callers read guest pages through it with no flush of their own: the type-5 view loader
(`metal_draw/vulkan.rs`), a blit reading a type-11 texture backing (`blit_exec.rs`), and the compute
sample stage (`compute_exec`). The fix is one `flush_intersecting` at the top of `read_rect_raw_at`,
matching `read_mapping_bytes` — at the choke point, so all three are covered and so is the next
caller. It is cheap: `flush_intersecting` returns immediately when nothing is armed.

So when auditing a flush-on-access contract, **enumerate the paths inside each reader, not the
readers**. "This function flushes" is a claim about a function; the guest reads through a *branch*.
Note also that this one was found by an adversarial subagent sweep and then confirmed by hand at the
two call sites — the confirmation is not optional, since AGENTS.md already records a subagent audit
on this same rail that was flatly wrong about `map_generation`.

And the decisive one: **the host-window present path never reads the mapping's guest pages at all.**
It reads `surface_cache` and the engine resident. (Until 2026-07-30 the chain here was
`publish_window_frame` → `export_present_dmabuf` → `export_present_from_resident_fd_policy`; those two
are deleted, and the conclusion is unchanged because neither ever touched a guest page either.)
`note_front_buffer_writeback`, `note_dense_frame_published` and
`note_surface_composite` only record metadata and enqueue a `HostAction` — none of them touches a guest
page. So the ~70 Stores a second are writing bytes that, on the present path, nothing reads. That is
what makes the deferral a win rather than a rescheduling.

What is genuinely missing, and is the work:

- **Pinning.** The render rail pins by `TargetIdentity` (`pin_resident_target`); the compute rail pins
  by `ComputeStorageResidencyKey` (`pin_resident_storage`). A type-11 render window needs a resident
  pinned under `TargetIdentity::Surface` — which `render_chain_identity` already produces for
  `mapping_id != 0` — while being *indexed* for flushing in the mapping-keyed range map. Those two
  are different keys for the same image and joining them is the substance of the change.
- **Load-seed suppression.** The GVA Load skips its CPU seed when a deferred window exists at matching
  geometry. The type-11 Load has no such check and always tries `surface_cache::get`, so without it
  only half the traffic goes away.
- **Supersede.** `supersede_gva_window`'s equivalent for re-arm, geometry change and clear-Store at the
  same mapping range.
- **The write gate.** The GVA flush checks `gva_write_allowed` before touching guest pages; the type-11
  flush needs the mapping-side equivalent — and the drift guard, since this is a deferred window and
  `deferred_pages_still_ours` exists for exactly this hazard.

Do not assume `flush_one` can be called as-is: the compute flush reads a *storage* resident by
`ComputeStorageResidencyKey`, and a render Store's pixels live in a *target* resident by
`TargetIdentity`. The index and the triggers are reusable; the read is not.

**`req.output_bgra` was built, tested, and unreachable from product code. As of `e2c2dee` the BGRA
resident path is live, but not through that flag** — do not read the paragraphs below as still
describing the tree. `req.output_bgra` is *still* set nowhere in `src/`; what changed is that
`exec.rs` now reads `output_bgra` as `identity.is_bgra() || req.output_bgra`, so every
`TargetIdentity::Surface` resident is BGRA whether or not a caller asks. The flag survives as an
explicit opt-in for namespaces whose identity does not imply an order, and the reasoning below about
what the dead path cost and what it already had test coverage for is the argument that motivated
turning it on. See "Its prerequisite shipped in `e2c2dee`" further down for the shipped form.

The original reading, kept because the *cost* accounting in it is still correct: `grep -rn
output_bgra crates/reims-vgpu` found it read in five places in `engine/exec.rs`, and assigned `true`
in exactly six places, *all of them in `tests/vk_engine_parity.rs`*. No `src/` file ever set it, so
`let output_bgra = req.output_bgra && req.target_identity.is_some()` was permanently false and the
engine's "BGRA output, so a raw image→buffer copy lands guest scanout order with **no CPU swizzle**"
path never ran.

That mattered twice over. It *was* why the runtime paid `swap_rb_channels` on every type-11 Load seed
— see the seed-order work below, which removed that half without touching the flag. And it means the
swizzle-free resident path already exists with six GPU-executing parity tests behind it
(`partial_draw_preserves_rgba_seed_on_bgra_target`,
`sampled_rgba_upload_to_bgra_target_preserves_semantic_channels`,
`a_view_swizzle_is_performed_by_the_image_view_not_the_cpu` and three more, all `grep -c SKIP` of 0
when filtered). Do not delete it as dead; it is the other half of the deferred type-11 rail.

One piece of that waste *was* independently removable and is gone: `execute_draw_inner` cloned the
whole seed frame twice in fifteen lines — once into `resolved_load` (structural, `req` is a shared
reference) and again into `seed_bytes`, purely to own a buffer the `output_bgra` arm could mutate in
place. The second is now a move. Nothing else read `resolved_load` after that point.

**Both are now gone, and the honest accounting is five full-frame CPU passes per composite pass, not
one.** `resolved_load` was deleted outright: its `Clear(c)` arm was already dead (the primary clear
value handed to `cmd_begin_render_pass` is hardcoded `[0,0,0,0]` and `c` was never read), so the
enum existed only to own a buffer for the swizzle. `seed_bytes` is now `Option<&[u8]>` borrowed from
`req`, and the `output_bgra` swizzle folds into the copy that has to happen anyway
(`write_staging_rgba_as_bgra`, one pass into the mapped span — the same transformation
`write_staging_from_runs` already documents for buffer binds).

Count the rest before claiming the round trip is cheap. Under a browser workload `engine_delta` reads
**readback 448 MB/s ≈ seed 461 MB/s** with `drain_duty` at duty 0.948 and ~1142 chains/s, and each
type-11 composite pass touches its frame five times:

| side | pass | status |
|---|---|---|
| seed | `swap_rb_channels(bgra)` in `metal_draw/vulkan.rs` — alloc + copy + swizzle | **gone** — the seed states its order |
| seed | `write_staging` into the mapped span | no — this is the upload |
| readback | mapped buffer → `vec![0u8; rb_size]` | no — this is the download |
| readback | `swap_rb_channels(&rgba)` back to BGRA for `surface_cache` | still there; needs `output_bgra` |
| readback | `surface_cache::store` (+ a `.clone()` when `texture_ref != 0`) | **gone** — the cache holds an `Arc` |
| return | `rgba.clone()` at each exit of the Store block | **gone** — the block takes rather than borrows |

That was ~450 MB/s × 5 ≈ **2.2 GB/s of CPU memory traffic**, which explains a drain worker pinned at
0.95 far better than the 900 MB/s headline does. Three of the six are now deleted.

**How the seed one came out, because the trap is real and the way round it is not obvious.** Simply
setting `output_bgra` makes the *engine* swizzle instead of the runtime and wins nothing — the
runtime keeps converting, because `LoadSeed` was *defined* as semantic RGBA8 and the cache holds
BGRA. The fix is to stop defining it: `DrawRequest.target_seed_order` (`SeedOrder::{Rgba8, Bgra8}`)
names what is actually in the bytes, the attachment's order is `output_bgra`, and the R/B exchange
happens exactly when they disagree — inside the copy into the mapped staging span that has to happen
regardless. `write_staging_rgba_as_bgra` is now `write_staging_swap_rb`, which is what it always was:
the exchange is an involution, so one routine serves both directions.

`target_rgba8` is an `Arc<Vec<u8>>` and `surface_cache::get_shared` hands the cache's own buffer
over, so a type-11 Load seeds a draw with a refcount. `get_shared` requires the stored length to be
*exactly* the geometry and misses otherwise — a handle cannot be truncated the way `get`'s slice is,
and the engine rejects a seed of the wrong length, so serving a buffer with slop would turn a working
draw into a declined one.

**`output_bgra` is still not set anywhere in `src/`** — the above changes no attachment format. But
the remaining readback swizzle is now the *only* thing it buys, and turning it on makes that
conversion disappear rather than move, because the seed condition simply goes false. Before flipping
it, note the blast radius — **which halved on 2026-07-30 and is no longer what this paragraph used
to say.** It named two gates; `export_present_from_resident_fd_policy` and its
`PresentExportResidentNotBgra` refusal were deleted with the dmabuf rail, so do not grep for them.
What remains is one: `read_resident_bgra` returns `None` unless `slot.bgra`, and its only caller is
`scanout.rs`'s resident capture. Making type-11 residents BGRA therefore enables **one** present path
that currently never fires, not two. Still a display-path change, and still wants a live boot rather
than a test run.

Also note what the same line says about batching: `batch_opens == batch_flushes` **exactly**, in
every window measured (912/912, 2036/2036, 504/504, 733/733, 517/517, 1000/1000), at ~1.7
`batch_flush_draws` per batch. Draw batching is barely engaging, and the reason is the seed:
`joins` requires `req.target_rgba8.is_none()`, and a CPU-seeded type-11 Load always sets it.

**Do not read that as "fix the seed and batching follows" — it does not, and the same line says so.**
`joins` also requires `req.skip_readback`, and a type-11 composite Store reads back by construction,
so those draws could never batch whatever the seed did. The batching that is being lost belongs to
the *chain intermediates*, which is a different population from the Stores the seed cost was measured
on. Two problems that share a predicate are still two problems.

**`us_per_draw` has a 1.8x boot-to-boot spread and drifts upward within a session, so it cannot score
a change sequentially.** This is the interleaving rule above, restated for the perf metric, and it is
worth stating separately because `drain_duty` looks precise — it is a within-boot duty cycle over 140
one-second windows, and the temptation is to treat a 35% move as signal. Eight boots, mean
`draw_us/draws` over busy windows (`duty > 0.8`), in the order they were run:

| boot | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| us/draw | 2310 | 2660 | 3266 | 3561 | 3766 | 4202 | 3582 | 3928 |

The first **six** are all *unchanged* code and span 2310-4202. The last two are the seed-clone
removal, and they land in the middle of that band. The pre-change series is also close to monotone in
run order, which is a time drift and not a code effect — so a sequential A/B on this metric will
report whatever direction the session happens to be drifting.

Concretely: reading boots 2 and 7 as a before/after gives "35% regression" for a commit that deletes
a `memcpy` and can only be neutral or better. That is the same shape as the 3-of-3 versus 2-of-2
result recorded above which survived a full revert of its supposed cause. Interleave, or score the
change with a counter that measures the thing directly rather than the frame rate downstream of it.

**The Linux host window is a second `VkDevice`, and that — not a capability — is why every present is
a CPU round trip.** Mapped 2026-07-30. `host_window/present.rs` creates its own `ash::Entry`,
`Instance` and `Device` (`VkState::new`), so a frame reaches the window as CPU bytes by construction:
the drain reads the resident back (`read_target`, 8.15 MB), `publish_window_frame` copies it again
(`p.frame_bgra[..need].to_vec()`), and the window thread memcpys it a third time into a persistently
mapped LINEAR staging image before blitting that to the swapchain. **Three full-frame CPU passes per
presented layer**, ~450 MB/s each at the measured rate. The module's own doc comment states the fix:
"Presenting the engine's resident image directly on a shared `VkDevice` would remove that copy and is
not implemented."

macOS already does it. `engine/window_present.rs` blits the engine resident straight into the acquired
swapchain image, and the file is **platform-agnostic** — it reaches the surface through
`ash_window::create_surface`, which dispatches on the raw handle type, and nothing in it calls a macOS
API. It was merely `#[cfg(target_os = "macos")]`. Ungating it compiles clean on Linux with zero
warnings, and brings **nine unit tests that had never run on this host** (lib 969 → 978).

Its capability gate is already written and already fail-closed: `WindowPresenter::create` refuses
`SwapchainUnavailable` when the device extension is absent and `QueueCannotPresent { queue_family }`
when `get_physical_device_surface_support` says no. That is the "gate on capabilities, not vendor
names" rule already satisfied — do not write a second one.

**The prerequisite this file recorded as blocking is dissolved, and the resident rail dissolved it.**
The entry below says the export path asks the registry for the mapping the guest named
(`outcome=orphan`, mid 2 while the registry held mid 1) and warns against substituting a peer's
resident. That was measured before type-11 composite Stores became registry-resident. Re-measured on
one x86/Vulkan boot at `fc61203`: **`target_reads` 58/s against ~57 presented layers/s**, with
`export_present_miss`, `orphan`, `no_resident_content` and
`vk_engine_export_present_unknown_identity` all **0**, across the three mids the guest actually
presents (2, 5, 6). Essentially every presented layer is already served by a resident read today, so
the identity question the retraction was waiting on is answered by construction: the Store that
produces the frame pins and stamps the resident under the same mapping the present names.

Both halves also select the **same physical device** — `host_window_vk_caps role=consumer` and
`vk_caps` both name the RTX 5080 on this host — so sharing the engine device does not change which
GPU renders.

What is built: the engine instance now enables `VK_KHR_surface` plus whichever of
xlib/xcb/wayland the loader advertises (the window does not exist when the context is created, so the
platform cannot be known then, and an enabled surface extension with no surface is inert), the device
enables `VK_KHR_swapchain` on Linux, and the presenter and its facade are ungated. What remains is the
wiring: `host_window/present.rs`'s `resumed()` and `draw()` still take the `VkState` arm on Linux, and
`publish_window_frame` still builds a resident source only under `#[cfg(target_os = "macos")]`.

The one design question left is **what to do when no resident is ready**. macOS drops the frame and
counts `window_publish::note(false)`; Linux has always had CPU bytes to fall back on. Publishing a
resident when one is ready and CPU bytes otherwise keeps the window from freezing, but the capture
that produces those bytes is the cost being removed — so the choice has to be made in
`capture_present_frame`, before the readback, not at publish time.

**Open lead: the window handoff is CPU staging on every present, and the two halves disagree about
which mechanism is in use.** Every boot ends its throttled census the same way:

```
present skip-ratio uploads=1024 presents=1024 (elided 0 redundant full-frame uploads)
    source dmabuf=0 staging=1024 fresh_imports=0 redundant_fds=0
direct_present_source presents=1024 uploads=1024 dmabuf_blits=0 staging_blits=1024 fresh_imports=0
```

So every present into the host window is a full-frame CPU upload and the dmabuf import never once
took.

**The pair of capability lines used to be cited here as corroboration, and that citation is
withdrawn — the lines were fiction.** They read:

- producer: `vk_caps … handoff=dmabuf_fd handoff_declined=[]`
- consumer: `host_window_vk_caps role=consumer … dmabuf_import=true handoff=engine_swapchain`

which looked like the two halves of the device disagreeing about a live mechanism. Neither `handoff=`
was a reading. Both came from `caps::HandoffLadder::resolve`, a pure function whose *only* consumers
were those two format strings — nothing in the present path ever asked it anything — and the consumer
side hardcoded `dmabuf_export: false, engine_swapchain: true` at the call site so the word
`engine_swapchain` would come out. The whole classification layer (`frame_interop.rs`, `matrix.rs`,
`zero_copy.rs`: `FrameHandoff`, `SupportCell`, `DmaSupport`, `GuestRead`, `GuestWrite`,
`ZeroCopyProfile`) was deleted 2026-07-30 for that reason; `HostGpuCaps` now carries one measured bit,
`dmabuf`, and both lines report it and agree. `memory_topology` was untouched and stays live, because
every allocation names a `MemoryClass` and something reads the answer.

That is "a reason the caller writes is not a reading" in its most expensive form: not one `reason=`
mislabelled by a caller, but a whole taxonomy of them, with a `declined()` list that fired
`vk_caps_zero_copy_declined reason=no_host_pointer_import` twice on every device create on every host
because `resolve` built it as a constant. A decline that cannot *not* fire reports a design decision,
not a loss.

The lead itself survives, on the evidence below, which is about the branch and not about the labels.

The decisive detail is what is **absent**. `present.rs` emits typed
`direct_present_decline`/`direct_present_degrade` lines for `import_failed` and `fd_missing`, and a
boot has **zero** of either. Those live *inside* the `if let Some(dm)` arm, so their silence does not
mean "the import was tried and worked" — it means the arm is never entered and the frame carries no
dmabuf handle at all. That is the "instrument the branch, not the arm" trap in its most flattering
form: `fresh_imports=0` with a clean failure channel reads as health.

**Why the frame carries no handle is now read, and it is not a capability gate.** The export decline
is `fail_once`-latched, so it says this once per boot and then goes quiet:

```
OFF export_present_miss outcome=orphan want=surface geom=1920x1080 want_gen=1 surfaces=[1:1:1] gva=0 reg_len=1
export_present reason=vk_engine_export_present_unknown_identity identity_kind=surface identity_id=2 …
```

The present path asks the registry for **mid 2**'s resident; the registry holds only **mid 1**. On x86
the guest presents ClearOnly mids 2/3 while content renders into mids 1/4/5, so the mapping named by
`CmdDisplaySwap` is not the mapping anything drew into. The chain is short and every link is a single
line: `publish_window_frame` passes `p.frame_mapping`; `scanout.rs` sets `frame_mapping` to whatever
`capture_present_frame` was handed; `drain` hands it the mapping the guest named; and
`window_present_source` builds **one** candidate from it and nothing else. `outcome=orphan` is
`classify_export_present_miss` stating exactly this — by its own definition it means "a resident
exists at this geometry under a different key" — and `surfaces=[…]` is `id:generation:content_ready`.

`state.present.early_front_mapping` already tracks the Composite peer for precisely this dual-mid
shape. Its only readers are `early_scanout_target` and two diagnostic lines; **the export path never
consults it.** The `WindowPresentSource` field is literally `candidates: Vec<…>` while
`export_present_dmabuf` reads only `.first()?` — the plural was built for a multi-candidate resolve
and has never carried more than one entry.

**Do not just add `early_front_mapping` as a second candidate.** The CPU path that works today reads
`surface_cache::get(state, frame_mapping, …)` — the ClearOnly mid's *own* cached bytes, which
something did fill. Substituting the Composite mid's resident presents a different surface, and
nothing has established the two hold the same pixels. The next step is a measurement: capture mid 2's
cache and mid 1's resident at the same instant and compare. If they differ, this is the same
front-frame retention policy question that the retracted type-11 `LoadFromTarget` design ran into
above, and it deserves the same suspicion.

Note also that this is on the host-window present thread, not the drain worker, so `drain_duty`'s
`publish_us` of 2-10 ms does **not** bound it and it is invisible to the duty measurement. Any claim
that it does or does not cost frames needs its own measurement.

**The type-11 sample window is resolved from the guest's descriptor, not invented — measured, and it
is the reason not to delete the invent rung.** Type-11 is the one case with **no wire plane index**:
nothing on the wire names which plane a texture wants, so `sample_window_prefer_device` matches width,
height and bytes-per-element and takes the plane only when *exactly one* matches. Zero matches and
two-or-more matches both fall through to an invented packed window at offset 0, which on a multi-plane
surface is a bind of the wrong plane — and the geometry scan cannot detect that by construction. The
fallback therefore reports itself: `type11_window_invent`, always-on, deduped per (mapping, geometry,
format), carrying the plane count so "no descriptor yet" (`planes=-1`) reads differently from "the scan
could not pick a plane" (`planes>1`).

One x86/Vulkan boot, 2026-07-30, guest at a real session (Dock asserted, Safari on three pages,
Calendar, System Settings, host window verified rendering all of it): **0 lines**, while the type-11
Store rail was demonstrably live (`store_routes surface_deferred`/`surface_flush`, `type11_store_route`
×3). So every type-11 window on that boot came from the guest's `sIOSurfaceDeviceDescriptor`.

Do **not** read that as licence to delete the invent rung. Two reasons, and the second is the one that
matters: the denominator is small, per the standing rule that an event count over one window is not a
state; and the rung covers the *headerless* case, where the mapping has geometry but no descriptor has
been cached yet, so deleting it turns a legitimate early bind into a refusal. What the zero does buy
is that the wrong-plane hazard is not currently being taken on this pathway.

Two process points are worth as much as the result. First, the decisive probe was a *duty cycle* —
a state — where the pre-existing `sync_exec_lock_hold` was an event count above a 250 ms threshold,
and the measured frame period sat at 252–665 ms, i.e. just under that threshold for entire runs. It
fired **once** in a four-round boot while the worker was pinned at 100%. Second, `counter_snapshot`
is the second instrument here found fully built and never called; before adding a probe, grep for a
snapshot/census function and check it has a caller.

**Do not edit Rust while a multi-boot harness is running.** `boot-x86.sh` rebuilds QEMU every boot,
so a tree edit lands in the middle of a run. One `stability-n.sh` run was scored `DISCARD` for exactly
this. The numbers taken before the edit are still good; the verdict is not. `interleave-ab.sh` is
worse than that: it `git switch --detach`es per boot, so an edit mid-run either fails the checkout or
is silently measured on the wrong arm. It also leaves the repo on a detached HEAD if it is killed
outside its trap — check `git rev-parse --abbrev-ref HEAD` before believing a later `git status`.

**`interleave-ab.sh` fed garbage arguments to every parameterized repro, silently.** Its
`boot_and_measure` read `local arm="$1" ref="$2" round="$3"` and then forwarded `"$@"` — which still
held all three, because `local x="$1"` reads a positional without consuming it. The repro therefore
got `(out, arm, ref, round, …its own args)`, so `idle-then-damage.sh` read `IDLE="${2:-90}"` and got
the literal string `"parent"`. Nothing errors: the run just produces no verdicts, which reads exactly
like "the arms did not differ". Fixed with `shift 3`. **Any earlier interleaved run that passed repro
arguments is suspect** — a run with no extra arguments was unaffected, because then `$@` was empty at
the call site and the stray three were the only ones forwarded.

That is the same shape as the `grep -c` and `qmp.sock` traps already recorded here: the rig failing
quietly and its silence reading as a clean measurement. Before trusting a harness result, print the
argv the repro actually received.

**A watchdog on `hostfwd=tcp::2222` can answer for the previous boot.** The port is claimed by
whichever QEMU holds it, so a multi-boot harness starts the watchdog while the last guest is still
alive on it. Measured: run 2's watchdog latched `seen_session=1` off run 1's healthy desktop, then
scored run 2's normal login window as `SESSION_LOST` — a false WindowServer-crash verdict on a
completely clean boot, in both runs of one harness, with no `.ips` harvested in either. `watchdog.sh`
now scopes every latch to the guest's `kern.boottime` epoch and announces a change as
`GUEST_REBOOTED`. Same class as the `qmp.sock` symlink that outlives its QEMU: **before believing any
guest reading, establish which guest answered.**

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

**A reference-count sweep is a measurement, and its failure mode points at deletion.** The three
mechanical sweeps this repo runs — unreferenced `pub fn`, write-only `pub` fields, per-file
reachability — all reduce to "count the uses". When the counter is wrong it is almost always wrong
*low*, and a low count reads as "safe to delete". Every other measurement here fails toward "no
finding"; this one fails toward a commit.

The concrete trap, hit once: the field sweep blanks string literals before counting, and the obvious
way to do that is `re.sub(r'"..."', '""')`. Rust has raw strings (`r#"..."#`) and `'"'` char
literals, so that regex mis-pairs quotes and silently blanks whole regions of real code. It reported
nine live `PresentState` fields — `window_active`, `painted_mapping`, the `backpressure_hold_*` trio
— as never read, and each of them is read three lines from where it is written. Rerun with a
character-walking scrubber that understands `r#"`, `"`, `'x'` and both comment forms, the same sweep
returns 36 hits, all of them the `repr(C)` and `runtime/decode/**` rows that are *supposed* to be
there.

So validate the sweep before believing it, on a field you have already confirmed by hand is read.
`grep -c` for the bare name next to the sweep's count is enough: raw occurrences far above counted
reads means the counter is broken, not that the field is dead.

The same trap has a second door. Stripping `#[cfg(test)]` blocks by brace matching runs off the end
whenever the item has no braces — `#[cfg(test)] mod tests;` and `#[cfg(test)] let x = …;` both occur
here — and swallows the next real block, again reporting live code as unreferenced. An item ends at
its matching `}` *or* at the first `;`, whichever comes first.

Do not write a fourth one of these. `observe/gate.rs` already carries a correct Rust lexer —
`result_string_error_offsets` masks line and block comments, raw strings with any hash count, escaped
string literals and char literals, in one pass — and it exists precisely because the gate it backs
could not be written with a grep. Reuse it before hand-rolling a scrubber.

**Always print the sweep's denominator, and gate it on a known-live case.** The rule above says this
family fails low; the cheap defence is to make the sweep state how much it found *before* it states
what is dead, and to check that count against something already known. Both halves earned their
place on 2026-07-30:

- The write-only-field sweep was gated on `content_generation`, `map_generation`, `mapped`, `width`,
  `height` — all known-read — and required them to score high before its zero-read list was believed.
  They came back 65 / 60 / 132 / 615 / 633, the gate passed, and the list was 10 names of 923.
- The decline-variant sweep skipped its gate and produced a **clean null**: "0 variants referenced
  ≤1 time", which reads as "the taxonomy is fully live". It had found **15** `Decline`/`Refusal`
  impls. There are 72 impl lines. The regex was `impl\s+(?:\w+::)?(?:Decline|Refusal)`, one `::`
  segment, and 36 of the impls are written `impl crate::observe::Decline for X` — two segments. With
  `(?:\w+::)*` it finds 58 distinct types and 559 variants, and *then* reports 0 dead, which is the
  same answer from four times the population and is the one worth quoting.

A null result from a sweep is only as good as its denominator, and a sweep that silently examined a
quarter of the population announces nothing at all. `grep -c` for the raw construct next to the
sweep's own count takes one command.

**Sweep results as of 2026-07-30, so they are not re-run blind.** Each is a measurement with a date,
not a standing guarantee; re-run after any large deletion, since that is exactly when new orphans
appear.

| sweep | population | flagged | disposition |
|---|---|---|---|
| unreferenced `fn` | 1651 | 6 | all false positives — 4 winit trait impls, a `Deref`, an example `main` |
| unreferenced `pub` item | 1281 | 33 | all `repr(C)` / protocol constant tables (`abi.rs`, `regs.rs`, `decode/`) — wire documentation, keep |
| write-only field (non-`repr(C)`, non-`decode/`) | 923 names | 10 | 4 derive-`Hash`/`Eq` key fields, 1 regex artefact, 5 wire-layout fields in `contract/iosurface_pages.rs`; the one real hit, `model::ObjectEntry`, is gone |
| never-constructed decline variant | 559 across 58 types | 0 | taxonomy fully live |

The productive one was the field sweep, and the reason is worth carrying: it is the only one of the
four that looks at *state* rather than at *code*. Dead code gets noticed because someone reads it;
dead state gets written on a hot path forever and reads as load-bearing, because the write is right
there. `presented_geoms`, `last_store_seq`, `store_seq` and `ObjectEntry`'s three fields were all
found this way within one session. Point the next sweep at fields, not functions.

Two hits from that sweep needed a hand-check that the sweep could not do, and both went the other
way from the machine's answer:

- **`.field` cannot see a destructuring read**, so a zero is a question, not an answer.
- **`.field` cannot see that *membership* is the payload.** `DeviceState::objects` scored as an
  unread map — `objects.get/contains/iter/values/len` returns only tests — and `delete_object`
  gates the whole host-side resource teardown on `objects.remove(..).is_some()`. The map is
  load-bearing; only its `ObjectEntry` value was dead. A sweep that had been trusted there would
  have deleted the teardown gate.

**A doc comment keeps a dead function's reference count above one.** The unreferenced-`pub fn` sweep
counts `grep -rw` hits, and `[`execute_draw`]` in a doc comment is a hit. Five functions sat at two
or three references made entirely of prose: the two engine facade owning-forms, an event
decode-then-execute wrapper, a batch ref resolver, and a cache `invalidate`. Run the same sweep over
comment-stripped text and they drop to one.

Which does not by itself mean delete. `Cache::invalidate` having no caller could equally have meant a
translation cache was going stale — the failure the idle-drain regression already cost this project
once. It is safe here only because every `Cache` in the tree is built with `Cache::default()` at the
top of one walk and dropped at the end of it, so it cannot outlive its own validity. Check that
before reading "nothing calls the invalidate" as good news.

**Ask whether a `cfg` can be satisfied at all — the decline gates cannot.** The gates read the source
for a slug literal, so a `Emit`/`Status` inside a block that *no configuration compiles* still counts
as a writer. `compute_dispatch_no_backend` was registered and unwritable for exactly that reason: its
arm was gated `all(not(feature = "backend-vulkan"), feature = "backend-vulkan")`, a contradiction.

The crate's four hard constraints make more configurations impossible than they look, and each is a
cheap grep:

- exactly one of `backend-metal` / `backend-vulkan` (`lib.rs`), so any `all(…metal…, …vulkan…)` gate
  outside the `compile_error!` itself is dead;
- `backend-metal` implies `target_os = "macos"` (`lib.rs`), so `backend/metal/**` has no non-Apple
  arm — `host_stub.rs` was 107 lines gated on precisely the refused conjunction, and had been
  broken long enough for a function it called to move modules unnoticed;
- `backend-vulkan` implies macOS or Linux;
- `host-window` implies `backend-vulkan` (Cargo feature dep).

The tell for this class is a build that *never runs*: `--features backend-metal` is Apple-only, so
nothing on a Linux CI or a Linux agent ever compiles those blocks, and `-D warnings` cannot report
what it does not build. Grep for the contradiction directly rather than waiting for a compiler that
is never invoked.

**Round-trip an extraction the host cannot compile.** `backend-metal` is Apple-only and does not
build on Linux at all, so a refactor of Metal-gated tests gets no compiler check whatsoever. Textual
extractions are still verifiable: strip the new helper, mechanically inline every call back to the
lines it replaced, and diff against `git show HEAD:<file>`. Identical means the substitution is
information-preserving, which is the property actually at issue. Two extractions of 29 and 24 sites
were landed on that evidence.

When the round trip cannot be exact — a helper taking `&mut T` where the inline block wrote
`&mut state` — audit the diff content instead: every removed line should appear exactly N times and
once in the helper, and every varying piece (a message tag, an id) should survive verbatim as an
argument. Print the deduplicated `-` and `+` line sets and read them.

For a hoisted `use`, the check is the *set of importable names* before and after. That catches the
specific way this goes wrong: two import blocks that normalise to the same text under whitespace
folding but differ by one name, so hoisting either one silently drops the other's.

**`#[cfg]` can sit below `#[test]`, and an attribute scan that looks above will miss it.** Twenty-nine
tests here are gated that way. A scan reporting "no cfg" for every one of them is the tell; the real
check is whether the Vulkan arm warns `dead_code` on a helper only those tests call.

**An extraction can make the file longer — measure after rustfmt, not before.** A helper whose call
exceeds 100 columns gets wrapped at every call site. Collapsing 173 two-call pairs into one seven-
argument call took `icb/tests.rs` from 7824 lines to 8499; dropping one redundant argument to get
under the limit took it to 7314. Same extraction, opposite sign. Run `rustfmt` on the file — by path,
never bare `cargo fmt`, which reformats pre-existing drift crate-wide — and re-count.

**But `rustfmt <path>` is not the safe half of that advice, and both of its failure modes are
silent.** The rule above is right that `cargo fmt` is worse; it is wrong that by-path is contained.

- **`rustfmt <path>` recurses into every `mod` the file declares.** Formatting
  `backend/vulkan/engine/mod.rs` rewrote `types.rs`, `caches.rs`, `context.rs`, `exec.rs`,
  `slab.rs` and `pools/mod.rs` — none of them named on the command line, all of them carrying
  pre-existing drift that then landed in the commit as unrelated hunks.
- **`rustfmt <path>` defaults to style edition 2015; this tree is formatted to 2024.** Nothing
  errors. `rustfmt.toml` says `edition = "2021"`, so even `cargo fmt` disagrees with the tree, and
  `cargo fmt -- --check` reports diffs in files nobody has touched. The visible symptom is an
  `assert!(Struct { … }.method())` reflowing one way and back, and `use` lists reordering by case.

Use `rustfmt --edition 2021 --style-edition 2024 <path>`, and then **read `git status` and the hunk
list**, not just the files you meant to change. The check that catches it is one command:

```sh
git diff <baseline> -- <file> | grep '^@@'
```

One hunk per edit you made, or you are committing somebody else's reformatting. That check is what
found this; a clean `cargo clippy`, a green 986-test suite and a PASS from `feature-matrix.sh` all
passed with six unrelated files in the commit.

And note the trap that hid it for two rounds: verifying the baseline by copying a file to `/tmp` and
running `rustfmt --check` on the copy reports **zero diffs for every file**, because the copy has no
sibling modules and rustfmt bails instead of formatting. A null result from an instrument that could
not run is not a baseline — compare in a real checkout or a `git worktree`.

### A Random Victim Is A Memory-Corruption Signature, Not Seven Unrelated Bugs

The guest panics, and every panic looks like somebody else's bug until you line them up. Across 351
retained `vm/disks/run/serial-*.log`, **7 boots ended in a guest kernel panic and the panicking
process is a different, unrelated one every time**:

| boot | process | first panic |
|---|---|---|
| 0728-154602 | Safari | `pmap_page_protect() pn=0x46b53b vaddr=0x7ff84ab74000` |
| 0729-174129 | ReportCrash | `pmap_page_protect() pn=0x2a8882 vaddr=0x124007000` |
| 0729-152409 | WindowServer | page fault, `CR2=0x0` |
| 0729-155116 | followupd | trap `0xd` (GP), `CR2=0x0` |
| 0729-175456 | com.apple.AppleU | trap `0xd` (GP), `CR2=0x0` |
| 0729-183422 | ReportCrash | non-sleepable RW lock with preemption enabled |
| 0729-195245 | airportd | kernel **NX fault**, `RIP == CR2`, inside `_sysctl` |

Read one at a time each invites a subsystem theory — a WiFi bug, a Safari bug, a WindowServer bug.
Read together they cannot be seven bugs in seven subsystems. **A defect in one subsystem kills that
subsystem every time; a defect that kills a uniformly random victim is corrupted memory**, and the
only component here that can write guest physical memory behind the guest kernel's back is this
device.

Both `pmap_page_protect` panics are `@pmap_x86_common.c:1788`, which XNU raises when the pv list says
a physical page is mapped at some `vaddr` but walking that pmap for `vaddr` yields **no PTE**. The pv
list and the page tables disagree. Overwriting a guest page-table page with framebuffer bytes
produces exactly that, and also produces the NULL derefs, the GP faults and the NX jump — one
mechanism covering all seven rather than seven mechanisms covering one each.

**That is a signature, not an attribution.** No probe has yet caught this device writing outside a
surface, and the mechanism is *unmeasured*. What has been audited, and is sound, is written here so
the next iteration does not re-read it:

- `mapper::write_mapping_bytes` — re-resolves `mapping_page_gpas` at write time, refuses on
  `need_end > span_end`, bounds every run copy. Sound.
- `mapping_write::write_bgra8` / `write_rgba8_image_changed` — validate against the latched geom, and
  `contig_for_span` enforces `len >= span_end` before the raw poke. Sound.
- `mapper::plan_adoption_decision` — the check that retires a cached `contig_ptr` is a **full
  element-wise PFN compare** (`current != plan`), not a length or a first-page test, and it is tested.
  A subagent audit reported the opposite ("map_generation is not bumped on a successful rewire"); it
  is wrong, mapper.rs bumps on `pages_changed` three lines below the compare. Read the code, not the
  summary.
- `gva_view::write_span` / `map_fresh_span` — walk the task page table **at write time** and never
  reuse a cached view. Their doc comment names the class this already fixed once (the 2026-07-19
  WindowServer SIGSEGV).

Two things are known-weak and both were *refuted as the cause by measurement*, which is the only
reason they are not being worked on:

- `gva_write_gate` returns `NoSpans` when a task declared no `MapMemory2` spans, and
  `gva_write_allowed` treats that as **allowed** — a fail-open authorization arm on a path that writes
  megabytes. It is instrumented (`write_gate_no_spans`, latched per task+caller) and it fired **0
  times** in the panic boot, while `write_gate_outside` fired 13 times and refused. The hole is real
  in principle and is not being taken in practice.
- `ensure_gva_view` validates a cached view only when `view_verify_ctr % 32 == 0`, and
  `view_gpas_current` then checks **only the first and last leaf page**. So 31 of 32 cache hits are
  unvalidated and the 32nd is a 2-of-N canary. This governs **reads only** — every writer takes the
  fresh-walk path above — so its failure mode is stale pixels, not corruption.

The hazard those guards exist for is real and frequent, which is what makes the class worth chasing:
`deferred_window_page_drift` fires ~8 times a boot with `armed_pages=164 live_pages=164 moved=156` —
**the page count is unchanged while 95% of the PFNs moved**. Any future check that compares a count,
a length, or one page will pass straight through that. Only a full PFN compare or a generation bumped
by one will not.

**A guard that re-resolves through the path it is guarding cannot see a substitution — it reproduces
it.** `deferred_pages_still_ours` is the drift check above, and it decided whether a deferred window
may still be written to guest RAM by comparing the armed page set against a *fresh* walk. Both walks
went through `gva_mem::visit_task_gva_page_gpas`, under the same `entry.task_id`
(`storage_flush.rs:438` ← `vulkan.rs:4934`) — and until 2026-07-30 that function resolved
`for id in [task_id, task_id >> 1]`. So a window indexed under the neighbour's page table was
re-indexed under the neighbour's page table, `live == *armed` held, and the guard reported the pages
"still ours". The freshness of the second walk was real; its *task selection* was the thing in
question, and it was copied rather than checked.

That was the **fourth** `task_id >> 1` arm in this crate and the last to go. The others are worth
listing together, because the pattern is that each was justified by "only one of the two slots is
ever live" and that premise is simply false: `task_slot::resolve_task_word` decides raw-only
(`raw_live.then_some(raw)`) and merely censuses the shifted case; `read_task_gva_by_id` refuses, after
measuring 9-11 wrong substitutions per boot; `gva_view::resolve_task_for_walk` returns `None`. One
healthy x86/Vulkan boot measures the premise directly — `task_walk_ambiguous` fires with
`named=1 other=0`, `named=2 other=1`, `named=5 other=2`, `named=7 other=3`, i.e. **every** time, the
`>> 1` slot was live *simultaneously* with the named one. `task_id >> 1` is not a spare spelling of
the same task; it is a different, running task.

Two process notes, both of which cost something here:

- **The subagent audit that found it was wrong about the consequence**, and this is the second such
  audit on this same rail (the `map_generation` one above is the first). It concluded "heap corruption
  if GPAs resolve under a different task". That does not survive reading the writer:
  `write_gva_rgba8` resolves through `gva_view::resolve_task_for_walk`, which is named-task-only and
  fails closed, so a misdirected *write* was already impossible. The real cost was a deferred window
  indexed against another task's pages, a drift guard agreeing with itself, and a flush choke point
  that misses — stale pixels and lost writebacks. Treat the nomination as a pointer to a file; derive
  the consequence by reading the code that actually performs the operation.
- **A stale doc comment marked the spot.** `visit_task_gva_page_gpas` documented itself as resolving
  "the same task selection as `read_task_gva_by_id`" while that function had already dropped its
  shifted arm. When a doc claims parity with another function, the cheap check is whether the other
  function still does what the doc says.

**Do not pool a boot failure with a clean boot.** The same 351 logs contain **15 boots (4.3%) that
never started macOS at all** — OpenCore `Boot failed - Aborted` — spread across many days and many
commits, so it is rig flakiness. A harness that scores "did not panic" over all boots counts those 15
as successes, which means a change that breaks booting reads as a change that fixes panics.
`.agents/repros/panic-rate.sh` keeps `PANIC` / `NOBOOT` / `NOWORK` / `OK` apart and reports
`PANIC / (PANIC + OK)`.

### A Seedless LOAD Is A Wipe, And That Was The Black-Rectangle Class

**Fixed by `c47efca`, measured before and after. Do not re-derive the mechanism.**

`engine/exec.rs` resolves a pass load action as "explicit `load_op` > `target_rgba8` > **Clear**". So
a `LOAD` that resolves no seed does not degrade — it begins the render pass with `LoadOp::CLEAR`
against the hardcoded `[0,0,0,0]` primary clear, and the matching Store reads that wipe back and
publishes it. **One whole compositing layer goes solid black.** The type-11 arm of the seed ladder
(`runtime/metal_draw/vulkan.rs`, `PASS_LOAD_ACTION_LOAD`) had exactly one source,
`surface_cache::get_shared(mid, w, h)`, and left `target_rgba8` unset on a miss — with **no report of
any kind**.

The always-on probe is `type11_load_seed`, latched per (mapping, requested geometry, outcome), and it
reports **every** outcome (`outcome=cache_hit`, `outcome=guest_pages`, or a typed decline) because a
zero on the miss arm has to be readable. Both arms carry `mapgeom` and `mapgen`: `want == mapgeom` is
the condition under which the guest-pages rung can serve, so the pair says whether a miss was
recoverable.

Two sequential x86/Vulkan boots, `.agents/repros/seed-loss.sh` (3 phases sliced on log byte offsets,
Dock hidden, native captures), scored as the fraction of near-black pixels (`max(r,g,b) < 0.03`) in
the host window:

| | phase 1 idle 30 s | phase 2 desktop selection drag 60 s | phase 3 testufo + 8 Safari window drags |
|---|---|---|---|
| seed misses, at `7502518` (probe only) | **0** | **5** | **116** |
| seed misses, at `c47efca` (fix) | 0 | **0** | **0** |
| black fraction, pre-fix | 0.0013 % | 0.6 / **62** / **90** / 0.6 / **62** / **62** % | 11.6 - **45.4** % |
| black fraction, at the fix | 0.0013 % | **0.0013 %** ×6 | 11.1 - 11.5 % |
| failure-channel lines | 0 → 2 | 9 → 4 | 205 → **76** |

Read three things off it. **Phase 2 is the reported wallpaper/drag class and it is gone**: six of six
captures during the drag now sit at `1.30261e-05`, which is the *same* six-figure constant the idle
frame produces in both boots — a within-boot positive control that also replicates across boots, so
this is "measure against known input" rather than a cross-boot brightness comparison. **Phase 3's
settled frame went 45.4 % → 11.1 %**, and the residual ~11.3 % is stable across every phase-3 capture
in both arms (it is the Safari window's own content, not a defect). **The phase-3 window-drag captures
barely moved** (11.6-11.9 % → 11.1-11.5 %), so do not claim that gesture as scored by this.

Every one of the 121 pre-fix misses was `reason=type11_seed_cache_absent` with `hostgen=0`, and every
one had `want == mapgeom`. `type11_seed_cache_geom` fired **0** times, so this was *not* the
geometry-replacement hole the earlier `deferred_flush_lost cache_miss` work closed — that fix covered
the deferred *flush*, and the Load seed read the same cache under the same rule and was never covered.
Four of phase 2's five were at the full **1920x1080** composite extent (mids 1, 4, 5, 21): a
whole-screen surface wiped. Several carried `mapgen` 2-5 — a mapping whose backing is replaced has its
cache entry evicted by `unmap_surface`, and nothing re-established the content, so its next LOAD wiped
it.

The fix is one rung, and it is a contract statement rather than a heuristic: **the host cache is an
accelerator, not the surface.** What a type-11 attachment *contains* is its guest IOSurface pages, so
a cache miss is a reason to read them (`load_type11_rgba_static`, whose `paint_mapping` lands every
intersecting deferred window first). The hit path is byte-for-byte unchanged, so the change can only
replace a wipe with the surface's actual bytes. **The sibling Metal path already did this** — type-11
`seed_color_load` reaches the same reader through `load_sampled_rgba_static`; only the Vulkan arm
stopped at the cache, which is the tell worth generalising: when two backends share a decode and only
one has a rung, the missing rung is a defect and not a design.

The ladder is now `resolve_type11_load_seed`, extracted so it is unit-testable —
`a_type11_load_seed_falls_back_to_the_surfaces_own_guest_pages` was verified to fail with the rung
stubbed to `None`.

What this does **not** close: the guest-pages rung's cost is unpriced. It runs only on a miss and the
following Store repopulates the cache, so it should be once per surface incarnation — but the probe is
latched, so 138 `outcome=guest_pages` lines are distinct first sightings, not occurrences. If a
mapping's Store keeps failing, its LOAD would gather guest pages every frame; `outcome=guest_pages`
count against `drain_duty` is how that gets read.

### Where The Drain Worker's Time Actually Goes, Measured Rather Than Decomposed

**`draw_phase`'s nine buckets sum to 72 % of `draw_us`.** The residual was ~245 ms per second on
2026-07-30, stable at 25-30 % across 200 windows — larger than `stage_us` and `readback_us`, second
only to `wait_us`. `draw_phase` brackets the *engine's* internals; this is the runtime work either
side of them, and no instrument named it while several sessions used that table to choose what to
fix.

`store_routes` now carries it (`note_store_route_us`, same per-second map as the route counts, so it
divides into the same window's `draw_us` with no join). Measured, `t11_*` spans on the type-11 arm:

| | per second | share of the arm |
|---|---|---|
| `t11_store_us` (whole type-11 arm) | 180 ms | — |
| `t11_convert_us` (the RGBA→BGRA frame) | **152 ms** | **84.5 %** |
| `t11_publish_us` (`publish_surface_store`) | 0.008 ms | nil |
| rest of the arm | 28 ms | 15 % |

So one `rgba.to_vec()` plus an in-place R/B swap was 62 % of everything the phase table could not
see. `publish_surface_store` being 8 *microseconds* per second is worth noting on its own: it had
been described as a publish step and it is three metadata writes.

**Split it before removing it.** A microbenchmark, per 8.29 MB frame: `to_vec()` 297 us, `to_vec()` +
swap 782 us, swap in place 448 us. The clone is ~43 % and the swizzle ~57 %, so they are separate
changes with separate risk.

**The clone is gone (`fb89924`), scored at −19 % per Store.** `arm_surface_deferred_store_with` now
takes `Vec<u8>` by value. It could not before because the caller returned the buffer as the draw's
chain value — and that return is **dead on this route**: the arm is reached only under
`writeback_guest`, `multi_draw_store_plan` grants that solely to the last record of a packet, `exec.rs`
feeds `chain_rgba` into record N+1's seed (exec.rs:1609) and there is no N+1, and every other reader
is an abandon arm in a loop that has just ended.

The signature is `Result<u32, Vec<u8>>` rather than `bool` because a moved buffer cannot be un-moved:
the type makes "refused" and "you still have the pixels" the same statement, and the three refusal
gates hand the frame back to the synchronous route.
`a_refused_deferred_arm_returns_the_frame_it_was_given` pins it — an `Err` built with an empty or
wrong buffer still compiles and would write a blank frame into guest pages on every refusal.

| | before | at `fb89924` |
|---|---|---|
| `t11_convert_us` **per Store** | 776 us | **627 us** |
| unaccounted share of `draw_us` | 27.6 % | **22.2 %** |
| `type11_seed_elided` / `uploaded` | 4588 / 278 | 5984 / 221 (96.4 %) |
| `FAIL` lines | 0 | 0 |

**Read the per-Store number, not the frame rate, and this boot is why.** Host presents went 21.9 Hz →
17.2 Hz across the pair, which reads as a regression from a commit that deletes a `memcpy` and can
only be neutral or better. The after-boot did **10 % more readback bytes/s (967 → 1065 MB), 6 % more
draws/s and 34 % more presented layers** — a heavier workload, on a rig AGENTS.md already records with
a 1.8x cross-boot spread. The Store rate was near-identical (196.5 vs 192.9/s), which is what makes
`t11_convert_us / surface_deferred` comparable when nothing else is.

Note also what the guest said on that boot: **testufo read 60.000 Hz / 60 fps**, against 41.000 Hz two
boots earlier. The guest's own opinion is not our present rate and does not score this change — but it
is the first time this rig has produced the goal's target number on the guest side.

**Next, and it is the large one.** `wait_us` (287-311 ms/s) + `readback_us` (133-156 ms/s) ≈ **half of
`draw_us`**, both charged per byte, and `store_routes surface_deferred` ~200/s against `readbacks`
~208/s says essentially every readback is a type-11 composite Store. `skip_readback` returns from
`execute_draw_inner` *before* `Phase::Wait`, so dropping that readback drops both together.

**Its prerequisite shipped in `e2c2dee`: a `Surface` resident is BGRA, and the order is now a property
of the identity rather than of the draw.** `TargetIdentity::is_bgra()` is the single rule, read by
`exec.rs`'s `output_bgra` derivation, and it deletes the ~152 ms/s (776 us/Store, 84.5 % of
`t11_store_us`) whole-frame swizzle outright instead of moving it. `req.output_bgra` survives as an
explicit opt-in for namespaces that do not imply an order, and is still set nowhere in `src/`.

Two design points worth carrying, because both were nearly got wrong:

- **Derive the order from the identity, not from the rail.** `registry_ensure` destroys and recreates
  the image whenever a draw disagrees with the slot, and a composite Store, a chain intermediate and an
  MRT primary all reach one surface through `render_chain_identity`. Keying on what they already agree
  on makes them agree here for free; a per-path predicate that one path spells differently is a full
  reallocation per composite — `target_evicts` climbing, not a wrong colour.
- **Then make the engine *report* the order, because the runtime cannot re-derive it.** Whether a
  record got a resident or a pooled target depends on whether an identity resolved, so
  `DrawOutput::pixels_bgra` and `TargetReadback { pixels, bgra }` carry it (the latter read from the
  registry slot under the same lock as the copy). This is "a reason the caller writes is not a reading"
  applied to byte order, and what it forecloses is a silent R/B exchange on a whole frame — a defect
  class **no assertion in this crate was watching for**, which is why the parity suite could not have
  caught it. Those cases now normalize through the reported order (`semantic_rgba`, `into_rgba8`) where
  they assert colour and keep raw `.pixels` where they assert layout, so they no longer silently follow
  whatever the engine picks.

The three order hazards the plan named were all real and are handled: `read_resident_chain` normalizes
at the one place its three callers share; the `cpu_portability` route now calls `write_bgra8` (same
tail — residency invalidation, `mark_mapping_written`, cache republish — so a substitution, and its
changed-span rung was never used from that site); the sampled reader off `surface_cache` is unaffected
and is still an unpriced fourth full-frame pass.

One thing fell out that the plan did not list: a type-11 Store was returning its whole frame as the
packet's chain value, and that binding **has no reader**. `writeback_guest` is granted only to
`di == last_i`, so there is no record N+1 to seed and every other reader in `exec.rs` is inside the
record loop that just ended. Now `None`.

**Scored on one x86/Vulkan boot at `e62bb9e`** (Safari fullscreen on testufo.com/refreshrate, 45 s
settle, `.agents/repros/testufo-fps.sh`; 8543 sliced lines, 80 `store_routes` / `draw_phase` windows,
85 `engine_delta` windows):

| | at `fb89924` | at `e2c2dee` |
|---|---|---|
| `t11_convert_us` **per Store** | 627 us | **0.00 us** (7 us total over 20 079 Stores) |
| `t11_store_us` per Store | ~920 us | **152 us** |
| `t11_store_us` per second | ~180 ms | **40 ms** |
| unaccounted share of `draw_us` | 22-28 % | **11 %** |
| `target_evicts` | 0 | **0**, in all 85 windows |
| `deferred_flush_lost reason=cache_miss` | 0 | **0** |
| failure-channel lines | — | no new kind; 1 `map_generation_drift` |

The numbers that establish it are within-boot: 7 us of convert across 20 079 Stores is the pass being
*gone* rather than cheaper, and the phase table going from covering 72 % of `draw_us` to **89 %** says
the time came out of the bucket `b872e43` instrumented rather than moving somewhere unmeasured.
`target_evicts` at 0 across every window is the identity-derived rule working — that counter is what a
per-path predicate would have moved. Host presents read 25.03 Hz and the guest read 48.000 Hz / 89 fps;
neither scores this, per the 1.8x spread.

**Correctness is the screenshot, because a wrong byte order emits no line at all.** The captured frame
has the SYNC-FAILURE banner **red** (an R/B exchange makes it blue), hyperlinks **blue**, the Blur
Busters banner **purple** and the mascot **green**. That is the check to repeat for any future change
to attachment format; the failure channel cannot see this class.

Two counts on that slice were mis-read first time, and both traps are cheap to repeat:

- **`grep -c cache_miss` returned 85 and meant nothing.** It matches `sampled_cache_misses`, a *field
  name* printed on every `engine_delta` line whatever its value. The black-layer class is
  `deferred_flush_lost.*reason=cache_miss`, which is 0. Grep the whole slug, not a fragment of it.
- **`grep -c FAIL` returned 0 while a `deferred_flush_lost` was present.** `observe::fail` does not
  prefix its lines with `FAIL`; the failure channel is the lines *without* the `OFF ` prefix. Counting
  `FAIL` measures a different, smaller set — and it reads as a clean boot.

`read_resident_bgra`'s present rung is now *reachable* for the first time (`slot.bgra` can be true) and
is still never taken, because the present capture prefers `surface_cache` and the Store keeps
refreshing it. That rung is the gate for the work below and this boot does not exercise it.

**The bottleneck after this is sharper than the entry below predicted.** Same boot, `draw_phase` p50
per second of wall clock against a busy `draw_us` of 832 ms and `duty` 0.983:

| phase | us/s | share of `draw_us` |
|---|---|---|
| `wait_us` | **371 000** | **44.6 %** |
| `readback_us` | **194 000** | **23.4 %** |
| `stage_us` | 51 600 | 6.2 % |
| `prep_us` | 45 900 | 5.5 % |
| `acquire_us` | 36 100 | 4.3 % |
| `submit_us` | 33 300 | 4.0 % |
| `record_us` / `pipeline_us` / `descriptors_us` | ≤ 8 500 | ~1.4 % |

`wait_us + readback_us` is **565 ms/s, 68 % of `draw_us`** — not "half", as recorded below — and
`surface_deferred` runs 251/s against `readbacks` ~250/s, so essentially every readback is still a
type-11 composite Store. `skip_readback` drops both together.

**And the readback is almost entirely waste, which is now measured rather than argued.** Same slice:

| | |
|---|---|
| `surface_deferred` / `surface_flush` | 19 995 / **450** — **97.75 % of Stores are never read by the guest** |
| `readbacks` per Store | **1.02** — every one of them reads a whole frame anyway |
| readback volume | **97.4 GB in 85 s, 1145 MB/s** |
| `present_content` per Store | **0.169** |

So the pixels are produced 251 times a second and wanted about 0.19 times per Store. A rail that read
the resident only when a consumer asked would take roughly **5x fewer** readbacks, and `wait_us` is
charged per byte (~270 us/MB, controlled instrument) so the two phases fall together.

**Two of the three consumers turn out to need nothing, and that is the part that makes this tractable.**
Verified by reading, at `d1b5c9d`:

- **The sampled bind already prefers the resident.** `metal_draw/vulkan.rs`'s type-11 sample resolve
  does `if resident_ready { return SampledSourceRequest::Target(resident_id) }` *before* it looks at
  `surface_cache`, so the cache read below it is reached only when no resident is authoritative. Under
  `skip_readback` the resident stays ready, so this consumer moves to the GPU path rather than needing
  a lazy readback. `sampled_cache_hits` 7377 / misses 288 on that boot.
- **The Load seed is already 96 % elided** by the `4c82c4d` epoch witness (`type11_seed_elided` 6284,
  `uploaded` 285). The 4 % residual falls to `load_type11_rgba_static`, which reaches guest pages
  through `scanout::read_mapping_bgra8` and therefore lands intersecting deferred windows first — so it
  stays *correct* under `skip_readback`, just expensive. Materializing from the resident instead is an
  optimization of a 4 % path, not a prerequisite.

The linchpin for all of it: **`registry_mark_ready(identity)` is called unconditionally after submit,
before the readback branch**, so `content_ready` does not depend on `skip_readback`. That is what keeps
the resident authoritative for the sampled bind and the LOAD.

That leaves exactly two consumers to re-source, and their combined rate is ~0.19 per Store:

- **The present capture.** `try_capture_from_resident` already reads the resident via
  `read_resident_bgra`, and that rung is live for the first time now that `slot.bgra` is true. The one
  change it needs is ordering: `capture_present_frame` checks `surface_cache` *first*
  (`if !from_host_cache && !try_capture_from_resident(..)`), so a Store that stops refreshing the cache
  must also stop the cache from winning. Marking the entry resident-authoritative — `get` returns
  `None` — does that without touching the present path at all.
- **The flush to guest pages**, 2.25 % of Stores, which must read the resident instead of the window's
  owned `Arc`.

**The shape to copy for the cache is already shipped, for linear textures.**
`HostLinearTexture::resident_gen` means "the pinned resident at this generation is authoritative and
`bytes` is empty", with `note_linear_texture_resident` / `linear_texture_resident_gen` /
`materialize_linear_resident` around it, and `get_linear_texture` returning `None` while it is set.
`HostSurface` wants the same three, keyed on the `surface_content_epoch` witness that already exists.

**Durability is the risk, and the eviction audit came back clean.** AGENTS.md records the rule this
deliberately trades against — "a deferred obligation that names its data indirectly is only as durable
as the indirection … prefer owning the bytes; with `Arc` it is free." Here owning the bytes *is* the
readback, so the trade is the point and the mitigation has to be explicit. Audited by hand at
`f5e9418`, every path that can drop a `TargetIdentity::Surface` resident:

| path | honours a pin? |
|---|---|
| `evict_registry_to_cap` (LRU / `REGISTRY_CAP`) | **yes** — `pin_count > 0` rotates to the back instead of evicting |
| the same function's cap arithmetic | **yes** — `non_pinned_registry_len()` excludes pinned slots, so a pinned burst cannot force the active set out (thrash) |
| the idle target drain (`submission_and_buffers.rs:110`) | **yes** — victims require `pin_count == 0` |
| `registry_ensure` destroy-and-recreate | **no** — and it cannot fire for a `Surface`; see below |
| mapping teardown / unmap / device reset / `test_reset_engine` | drops the window through `take_deferred_flush_window*`, which is the same choke point `release_window_pin` guards |

**A subagent sweep nominated that one row as the "critical hazard" of the whole design — "a live
deferred window's resident can be yanked by a re-resolve with different generation". It cannot, and the
refutation is three lines of reading.** `registry_ensure`'s reuse test is
`slot.width == width && slot.height == height && slot.generation == generation && slot.bgra == bgra`,
and for a `Surface` every one of those is fixed by the key:

- `generation` is not an independent argument. `exec.rs` does `let gen = identity.generation();` and
  passes *that*, so `slot.generation == generation` can only fail if a slot were created under a
  different generation for the same key — and the generation is *in* the key.
- `width`/`height` likewise: `render_chain_identity` builds the identity from `req.width`/`req.height`
  when both are nonzero and returns `None` when either is zero, so there is no identity to ensure in
  the case where they could disagree.
- `bgra` became structurally impossible at `e2c2dee`, which derives the order from the key.

`target_evicts` reading **0 across all 85 windows** of the `e62bb9e` boot is consistent with that, but
it is not what establishes it — one boot's zero is not a state, per the rule above. The construction
argument is, and it is the third time an audit of this rail has had to be checked by reading rather
than believed (see the `map_generation` and `task_id >> 1` notes).

Note the consequence would also have been mild rather than critical, which is worth stating so the
next reader prices it correctly: a destroyed resident makes `read_target` return `UnknownIdentity`, the
flush emits a typed `deferred_flush_lost`, and the guest keeps its pre-Store bytes — stale but coherent,
not corrupt. And `registry_ensure` runs at the *start of a draw into that same identity*, so a new Store
is about to overwrite the surface anyway and the supersede rule already drops fully-covered windows.

`pin_resident_target` is **counted, not boolean** (`slot.pin_count`), so several windows on one surface
each hold a count and the slot survives until the last unpins — which is exactly the sibling-geometry
case that produced the `cache_miss` black-layer class from the other side. It also refuses when
`!content_ready`, so the caller has a fail-closed fallback to the synchronous Store.

**The insertion point for the unpin already exists and is already documented as mandatory.**
`storage_flush::release_window_pin` says "every site that takes a window and does not flush it must go
through this rather than calling `unpin_resident_storage` directly", and its `Render` arm is an empty
`{}` *because a render window owns nothing on the GPU today*. That arm becoming
`unpin_resident_target(&identity)` covers every drop-without-flush site at once — teardown, supersede,
the population cap — which is the difference between auditing one function and auditing a dozen callers.

**One leak hazard is already visible in the code and must be fixed in the same change.**
`flush_render_one` returns `false` early on `reason=map_generation_drift` (and on `write_refused`)
*before* doing anything else, because today it holds no pin to release. Add a pin without touching
those exits and each drifted mapping strands a full framebuffer pinned for the guest lifetime — the
"~260 stale residents (~516 MiB)" shape this file already records. The GVA rail is the template and
it gets this right: `flush_gva_one` calls `unpin_resident_target` on **both** the read-failure path and
the success path. That drift is not rare — the `e62bb9e` boot logged one in 85 s, and
`deferred_window_page_drift` fires ~8 times a boot elsewhere.

So the shape is: window carries `(identity, epoch)` instead of `Arc<Vec<u8>>`; arm pins and marks the
cache entry resident-authoritative; flush reads the resident, compares the epoch, unpins on every exit,
and emits a typed `deferred_flush_lost` rather than writing a frame it cannot vouch for. A loss on this
rail is a whole compositing layer, so the epoch comparison is not optional and neither is
`pin_resident_target`'s `false` return being honoured.

Consumers `skip_readback` must then re-source: `surface_cache` (present capture at `scanout.rs:248`,
the ~4 % of LOADs that still seed, and the sampled bind), the deferred window's owned frame, and the
guest writeback at flush (`storage_flush.rs:876`, which today reads the window's `Arc` and never
touches the GPU). The pin machinery to keep the resident alive across that already exists and the GVA
rail uses it (`pin_resident_target`, armed at `vulkan.rs:5282`, unpinned at `storage_flush.rs:527/535`
and on the eviction path).

**Built (`e5d13d2`, `4e5d03d`) and now scored. The rail works, and the readback it removed did not go
away — it moved to the present capture, which is not in `draw_us`.** That distinction was invisible
until `ce3f095` split the counter, and the way it hid is the transferable part.

`readbacks` pooled two populations: the copy a draw takes as its own tail, and the full-frame
`read_target` a consumer asks for later. A deferred Store *moves* a copy between them, so the one
number meant to score the rail reads the same whether the deferral worked or merely rescheduled the
work. The first boot of the rail came back with `readbacks / surface_deferred` at **1.39**, up from
1.02, which reads as an outright regression and is not one. `render_post_wait_skips` — which would
have said so immediately — was already incremented on every `skip_readback` draw and **had no reader
anywhere**, the third instrument in this tree found fully built and never called.

Split, on one x86/Vulkan boot (testufo fullscreen, 45 s settle, per-second medians over 7 819 lines):

| | per second |
|---|---|
| `surface_deferred` / `surface_resident` | 124 / **116 — 93.5 % of composite Stores defer** |
| `surface_flush` | **3** — 2.4 % of arms are ever asked for |
| `render_post_wait_skips` | **258** of 384 draws, i.e. 67 % skip their fence wait |
| `readbacks` (draw rail) | 124 at **383 MB/s** |
| `target_reads` (resident reads) | **58 at 473 MB/s**, mean **8.15 MB** = a whole 1920x1080 frame |
| `target_evicts` | **0**, every window |

`58 - 3 = 55` of those full-frame reads per second are the **present capture**, and that is where the
cost went. The Store no longer reads its frame back; the present boundary reads it instead, once per
presented layer, because arming the window cedes the host cache and `capture_present_frame` then falls
through to `try_capture_from_resident`.

**Which is why `draw_us` fell and the frame rate did not.** `publish_present_boundary` runs inside
`device_drain`'s tranche but outside every `DrainPhase`, so the moved cost landed in a bucket no
instrument named:

| | ms/s |
|---|---|
| `drain_us` | 738 |
| `draw_us` + `flush_us` + `compute_us` | 440 + 6 + 0 |
| **unattributed residual** | **292 — 40 % of the tranche** |
| predicted from `target_read_bytes` alone | 128 fence wait (at the measured 270 us/MB) + 77 memcpy (at 6137 MB/s) = **205** |

So the present capture accounts for roughly 70 % of a residual that had no name, and the arithmetic
closes without a free parameter. Read `drain_us - draw_us` on any future rail that claims to remove a
cost from the draw phase: a cost moved out of `draw_us` into the tranche's unattributed remainder
looks exactly like a cost deleted.

**The next step is therefore not another deferral — it is the present round trip itself.** Each
presented layer currently pays GPU→CPU 8.15 MB *and then* CPU→GPU 8.15 MB, because
`direct_present_source` reports `staging_blits` on every present and `dmabuf_blits=0`. That is a
full-frame round trip through host memory for an image that is already on the device the window
presents from. Do not read the frame rates across these boots as evidence either way: 14.41 Hz then
12.09 Hz, with the guest's own testufo readout at 119.959 Hz / 98 fps and then 61.000 Hz / 60 fps, on
a rig with a documented 1.8x spread and an instrumentation-only commit between them.

### The Present Round Trip Is Gone: The Window Blits The Engine's Own Resident

**Built by `3af5832`, scored on two x86/Vulkan boots either side. `target_read_bytes`
621 MB/s → 0. Do not rebuild the CPU path.**

`host_window/present.rs` created its own `ash::Entry`/`Instance`/`Device`, so a frame
reached the window as CPU bytes by construction — three full-frame passes per presented
layer: the drain's `read_target`, `publish_window_frame`'s `frame_bgra[..need].to_vec()`,
and the window thread's memcpy into a mapped LINEAR staging image. `c3ae935` made the
engine able to own a Linux swapchain; `3af5832` wired it.

Same workload both boots (Safari fullscreen on testufo.com/refreshrate, 45 s settle,
`.agents/repros/testufo-fps.sh`), per-second medians over the busy windows:

| | at `c3ae935` | at `3af5832` |
|---|---|---|
| `target_reads` / `target_read_bytes` | 76 / **621 MB/s** | **0 / 0** |
| `capture_sampling` | `full=4096 light=0` | **`full=1 light=8191`** |
| `publish_us` per second | 50 000 us | **65 us** |
| `drain_us − draw_us − flush_us` (unattributed) | **341 ms — 39 %** | **25 ms — 5 %** |
| `drain_duty` | 0.88 - 0.92 | **0.48 - 0.51** |
| `tranches` per second | 194 - 253 | **597 - 628** |
| draws per second | 504 - 555 | 789 - 855 |
| **present boundaries per second** | **74.3** | **109.0** |
| guest's own opinion (testufo) | 104 fps | 119 fps |

**Read `capture_sampling`, not the frame rate.** `full=1 light=8191` says one full CPU
capture happened in the entire boot and every present after it elided the readback — a
within-boot statement of the mechanism that no cross-boot drift can manufacture.
`target_read_bytes` going to a literal **0** is the other one: this is a cost *deleted*,
not moved, which is the distinction `ce3f095` split the counter to make visible and which
the deferred-Store rail before it got wrong.

The 1.47x on present boundaries is a cross-boot comparison and is therefore subject to the
documented 1.8x `us_per_draw` spread. It is corroboration, not proof.

Three things worth carrying:

- **The two halves must ask the same question, and they did not.** The publish path decides
  a frame AHEAD of the presenter whether to read the frame back at all. `resident_content_ready`
  checked `content_ready`; the presenter's selection checked `content_ready && bgra &&
  width == want && height == want`. A looser predicate on the publish side elides the readback
  for a frame the presenter then refuses — a blank window with no CPU pixels behind it, and a
  disagreement **neither call site can see on its own**. `pools::slot_presentable` is now the
  single rule and both call it. This is the same shape as the flush-choke-point audit above:
  a contract split across two sites is only as good as the predicate they share.
- **The CPU fallback is not a second rail and must not be deleted.** Presents with no resident
  at all are normal — the firmware framebuffer, a mapping the compositor cleared but never
  rendered into, the frames after a device reset. The presenter stages those into a
  host-visible LINEAR image. One boot logged exactly **one** `host_window_cpu_fallback` run
  (boot) closed by one `host_window_slate_end`, and zero `host_window_slate`. That split is
  deliberate: a covered run shows the guest's frame and only costs the host copy (census,
  `off()`), an uncovered one is a blank window (failure, `fail()`). Reporting both as blank
  would have cried wolf on every boot.
- **A LINEAR image written through a persistent map must stay in `GENERAL`.** `BlitSource`
  carries that: a resident moves to `TRANSFER_SRC_OPTIMAL`, the staging image takes a
  `HOST_WRITE → TRANSFER_READ` barrier and is read from `GENERAL`, and the `PREINITIALIZED →
  GENERAL` first-use transition is recorded only **after** the submit succeeds — declaring
  `GENERAL` as the old layout of an image still in `PREINITIALIZED` discards the frame it
  holds.

**The instrument this removed, and what replaced it.** `present_content` carries `rgb_nz`,
a whole-frame pixel scan of the CPU capture — so deleting the capture deleted the line, and
`testufo-fps.sh` came back `HOST: too few presents to read (n=0)` on a boot that was 1.5x
faster. That reads exactly like a catastrophic regression. Two readings replace it and both
are now in the repro: `host_window_cadence`'s `present_hz`/`direct_frac`, which counts actual
`vkQueuePresentKHR` calls, and `capture_sampling`, which counts present boundaries and is the
one rate emitted on **both** rails. Expect this whenever a fix removes a computation an
instrument was reading — check what the probe was made of before reading its silence.

**A latent defect this retired.** `0760635` dropped a parameter from `window_write_frame` and
updated the Linux call site only, so the macOS arm passed six arguments to a five-parameter
function for eight commits. It could not have compiled, and nothing said so because
`backend-metal`/macOS never builds on this host — the "a build that never runs" class this
file already records. Unifying the two publish arms deleted the cfg that hid it.

**What is left, measured on the fix boot.** `drain_duty` now has headroom (0.48) and the cost
is entirely in `draw_us` (460 ms/s), where the readback/seed pair is **symmetric to the byte**:

| | per second |
|---|---|
| `readbacks` / `readback_bytes` | 256 / **760 MB/s** |
| `seed_uploads` / `seed_upload_bytes` | 256 / **760 MB/s** |
| mean readback | **2.97 MB** — not a 1920x1080 frame (8.29 MB) |
| `render_post_wait_skips` | 599 of 855 draws |
| `wait_us` / `readback_us` | 263 ms / 58 ms |
| `batch_opens` / `batch_joins` / `batch_flush_draws` | **479 / 0 / 479** |
| `surface_deferred` / `surface_resident` | 256 / **240 — 94 % take the resident rail** |

So the type-11 composites are deferring correctly (mean readback is a third of a full frame,
and `surface_resident` is 94 %), and what remains is **non-composite Store records reading
their result back to host memory and feeding it straight back as the next record's seed** —
760 MB/s each way, at ~3 MB per layer. That round trip is the next target, and it is the
same shape the type-11 rail already solved one namespace over.

`batch_joins` is **0**, not merely low. `joins` has eight conditions and `begin_entry` flushes
the open batch, so a single non-joining draw between two joinable ones closes the batch — 479
opens for 479 draws is every batch a singleton. Do not assume the cause: AGENTS.md's older note
blames the `target_rgba8.is_none() && skip_readback` pair, but 599 draws now skip readback and
605 carry no seed, so ~599 should satisfy that pair and none joins. `matches!(req.load_op,
Some(LoadOp::LoadFromTarget))` is the untested suspect. Instrument which of the eight refuses
before changing any of them.

### The Presenter Is Not The Cap, And One Early Return Keeps Every Remaining Readback

**Both halves of this heading are now overturned — see "The Composite Round Trip Is
Gone, And The Presenter Is Now The Cap" below. The early return is fixed (`bd70e4a`),
and the presenter *is* the cap once the device stops being one. Kept because the
reasoning about why `presents` alone could not answer, and the 3.9x spread it
records, are both still live.**

**Measured on one x86/Vulkan boot at `08110da`** (Safari fullscreen on
testufo.com/refreshrate, 45 s settle, 2885 sliced lines, 47 cadence windows). Two
questions the previous boot could not answer, both answered unanimously.

**1. `offered == presents`, in every window. The window presents every frame it is
offered, immediately.**

```
host_window_cadence presents=60 direct=60 busy=0 busy_fence=0 busy_acquire=0 offered=60
    present_hz=59.8 offered_hz=59.8 direct_frac=1.00
```

`busy` was 0 in most windows and never above 4. So the host-window present path is
**not** the frame-rate limit and must not be worked on for frame rate: the device's
publish rate *is* the frame rate. This retires the reading that `presents=20
busy=420` on the previous boot implied a 5x drop — a `Busy` return leaves the
window's seq gate unchanged, so `busy` counted retries of the same 20 frames.
`offered` is the denominator that says so, and `presents` alone never could.

**2. `t11_keep_chain_from_resident` equals `surface_deferred` exactly, in all twelve
windows. One condition, 100 % of the population.**

| | per second |
|---|---|
| `surface_deferred` (the route that reads back) | 112 - 132 |
| `t11_keep_chain_from_resident` | **112 - 132 — identical, every window** |
| every other `t11_keep_*` | **0** |
| `readbacks` / `readback_bytes` | 116 - 118 / **366 - 372 MB/s** (3.16 MB each) |
| `surface_resident` (the route that does not) | 111 - 132 |

`type11_store_identity` opens with `if req.chain_from_resident || !writeback_guest {
return None }`, so a chained Store can never reach the resident rail however eligible
its mapping is — `surface_resident_store` stays false, the record falls through to
`M2vDrawSpan::Pixels`, reads its surface back, and lands on
`arm_surface_deferred_store_with` (which defers the *writeback* and therefore needs
the bytes). That early return is the entire remaining readback population.

**The fix is small and the identity is already correct.** For these records the block
above already ran `if req.chain_from_resident || (store_is_store && !writeback_guest)`
and set `resources.target_identity = render_chain_identity(state, req)` — which is the
**same function** `type11_store_identity` would have called. So the surface identity is
already on the request; what is missing is `skip_readback = true` and the arm. The
guard to keep is `writeback_guest`, which `multi_draw_store_plan` grants solely to the
last record of a packet, plus `store_action == PASS_STORE_ACTION_STORE` and
`mapping_id != 0` — i.e. this record *is* the guest-visible composite Store, not a
chain intermediate. Note the ordered probe cannot separate "`chain_from_resident`" from
"`identity_taken_by_another_rail`" here because it tests the former first and both are
true; they are the same fact, not two.

Do not confuse this with the retracted type-11 `LoadFromTarget` rail ("That was built,
booted, and is refuted"). That was about a **Load** trusting a resident to hold the
mapping's current pixels, which needs the front-buffer identity resolve. This is a
**Store** skipping its own readback on the rail `4e5d03d` already ships and which 50 %
of composite Stores already take successfully in the same boot.

**3. The boot-to-boot frame-rate spread is at least 3.9x, on identical behaviour.**
`3af5832` and `08110da` differ only by instrumentation, and `host_window_cadence`
`present_hz` reads **p50 20.00 (106 windows)** on the first and **p50 78.20, 53-60
sustained (47 windows)** on the second. Present boundaries read 109/s and 83/s; the
guest's testufo read 20.000 Hz / 119 fps and 59.000 Hz / 61 fps. The layers-per-frame
ratio moved from 5.5 to 1.5, so the guest settled into a different composite shape —
which is a *workload* difference the harness does not control.

That is far worse than the 1.8x `us_per_draw` spread this file already records, and it
means **no single-boot frame-rate number scores anything on this rig**, in either
direction. Score with within-boot ratios (`offered`/`presents`, `t11_keep_*`/`surface_deferred`,
`readback_bytes`/`readback_us`) and treat Hz as colour. The 53-60 Hz window is
encouraging and is not evidence that the 60 fps goal is met.

### The Composite Round Trip Is Gone, And The Presenter Is Now The Cap

**Built by `bd70e4a` + `04dc2f4`, scored on two x86/Vulkan boots.
`readbacks` 0, `wait_us` 0, `readback_us` 0, `target_read_bytes` 621 MB/s → 2.5 MB/s,
`drain_duty` 0.88-0.99 → 0.135. Do not rebuild any of it.** Same workload both
boots (Safari fullscreen on testufo.com/refreshrate, 45 s settle,
`.agents/repros/testufo-fps.sh`), ~105 one-second windows at the fix.

`type11_store_identity` opened with `if req.chain_from_resident || !writeback_guest
{ return None }`. `08110da`'s ordered probe had already measured the cost —
`t11_keep_chain_from_resident` equal to `surface_deferred` in all twelve windows,
112-132 Stores/s at 366-372 MB/s, every other keep-reason at zero — and the refusal
had no mechanism under it. `retarget_render_pass_draw` builds every record of a
packet from one attachment template, so records N-1 and N carry the same
`mapping_id` and geometry and therefore the same `render_chain_identity`; the
intermediates already render into that resident under `skip_readback` with
`LoadOp::LoadFromTarget`. The last record differs only in what happens *after* the
draw. The gate below it tested `resources.target_identity.is_none()` and so read
that agreement as a conflict; it now compares the two identities.

**The first boot of that put 100 % of Stores on the resident rail and the readback
did not go away — it moved to `flush_render_one`, and the shape of how is the
transferable part.** `surface_flush / surface_resident` came out **1369/1373**: one
flush per arm, a deferral degraded into a rescheduling with a GPU round trip added.
`type11_load_seed` named it with no new probe — `outcome=guest_pages` 110 against
17 `cache_hit`, `hostgen=0` on every one. The resident rail cedes the host cache, so
**any LOAD that misses the epoch check reads the mapping's guest pages, and reading
them lands the window the rail just armed.** Two independent gaps fed that loop and
either alone sustains it:

- **The witness did not survive its own flush.** `write_bgra8` ends in
  `mark_mapping_written`, which advances `surface_content_epoch` — correctly, the
  guest pages did change. But the *pixels* did not: they are the resident's, copied
  out of it one statement earlier. AGENTS.md recorded this asymmetry as designed-for
  and conservative, "at 95 % elision the residual is not worth chasing" — true while
  half the Stores kept the cache populated, false at 100 %. `flush_render_one` now
  hands the epoch back to the image the frame came out of, on the `Resident` path
  only (an `Owned` window's bytes came from an `Arc`, and nothing there establishes
  the slot still holds them).
- **An intermediate record could not ask the question.** The currency check was keyed
  on `type11_store_identity`, which requires `writeback_guest` — granted solely to a
  packet's last record. Record 1 of a chain therefore never asked, took a CPU seed,
  and read guest pages. It renders into the same slot.

Split into `type11_render_identity` (the slot this record renders into) and
`type11_store_identity` (that slot, restricted to the record that also stores it for
the guest) — a superset and its restriction, not two derivations. **The LOAD now goes
through `type11_load_currency_query`, whose signature takes no `writeback_guest`: a
resolver that cannot see the record's role cannot be keyed on it.** That is
deliberate in place of a test, and the reason is worth carrying: a unit test on the
resolver was written first, and with the call site stubbed back to the Store identity
**it still passed**. It was pinning the function, not the behaviour.

| | at `bd70e4a` | at `04dc2f4` |
|---|---|---|
| `surface_flush / surface_resident` | **0.997** (1369/1373) | **0.0034** (146/43912) |
| type-11 seed elision | 4-15 per window | **99.72 %** (42831 / 120) |
| `readbacks` / `readback_bytes` | 0 / 0 | **0 / 0** |
| `target_reads` / bytes, per window | 1051 / 3.24 GB | **p50 1 / 32 KB** |
| `seed_uploads` / bytes, per window | 1066 / 3.24 GB | **p50 1 / 32 KB** |
| `wait_us` / `readback_us` per second | — | **0 / 0** |
| `drain_duty` | — | **0.135** |
| `surface_deferred` (byte-owning route) | absent | **absent, 0 of 105 windows** |
| `target_evicts` | 0 | **0**, every window |
| `batch_joins` / `batch_opens` | 1070 / 1190 | 34396 / 37181 = **0.93** |
| `deferred_flush_lost` | 0 | **0** |
| guest's own opinion (testufo) | 88.000 Hz / 88 fps | 78.000 Hz / 78 fps |

Read the ratios, not the Hz. `surface_flush / surface_resident` and the elision
percentage are two counters from one window of one boot, which is the only kind that
survives this rig's spread. `wait_us` and `readback_us` at a literal **0** is the
cost *deleted* — the distinction `ce3f095` split the counter to make visible.
`target_evicts` at 0 across all 105 windows is the identity-derived BGRA rule still
holding under a doubled pin rate.

**A pin leak was shipped with the resident rail and is fixed here.**
`prepare_surface_deferred_window`'s supersede loop took each covered window with a
bare `take_deferred_flush_window_exact` and discarded it — the one site in the crate
that took a window and dropped it without `release_window_pin`, whose own doc comment
declares that mandatory. Pins are counted, and a repainting compositor re-Stores the
identical range every frame, so the superseded key *equals* the new one: the same
slot's `pin_count` climbed once per Store without bound. `evict_registry_to_cap`
rotates pinned slots instead of evicting and the idle drain requires
`pin_count == 0`, so a slot that got there could never be reclaimed — the "~260 stale
residents (~516 MiB)" shape, one frame at a time. Fixed at the choke point
(`storage_flush::supersede_covered_render_windows`, now the only product caller of
`take_deferred_flush_window_exact`), and `release_window_pin` returns the identity it
unpinned **because `unpin_resident_target` is a silent no-op on an absent slot and
the engine logs nothing** — "the pin was released" was a claim no test and no boot
could read, which is how it went several commits unnoticed. Two existing supersede
tests modelled the drop with `take_deferred_flush_window_exact` instead of calling
the real function; as written neither could have caught it.

**The bottleneck is now the swapchain acquire, and this retires the section above.**
"The Presenter Is Not The Cap" was measured when the device offered ~83 frames/s and
the window presented ~83. It now offers far more than the window will take:

```
host_window_cadence presents=20 direct=20 busy=418 busy_fence=0 busy_acquire=418
    offered=99 present_hz=20.0 offered_hz=99.0 direct_frac=1.00
```

`presents` p50 **20**, `busy_acquire` p50 **409**, `busy_fence` **0** — so ~438
`present()` calls a second, 95 % of which fail the non-blocking
`acquire_next_image(swapchain, 0, …)` at `window_present.rs:793`. The device behind it
is at `duty=0.135` with 625 tranches and 662 draws a second. Two facts bound the next
step and neither depends on a cross-boot comparison: **the host panel is 120 Hz**
(`modetest -c`: eDP-2 3840x2400 @ 120.00, preferred), and **`present_hz` reads exactly
20.0 in 90 of 106 windows** — a lock, not contention, which would scatter.

**All three follow-ups below were answered, and the answer was not in this repo — see
"The 20 Hz Present Cap Is The Host Panel Being Asleep, Not The Swapchain". The host
panel's `dpms` was `Off`; waking it moves a `vkcube` FIFO present from 19.5 Hz to
80.85 Hz at the same geometry on the same GPU. The `busy_acquire=409` reading is
correct and the refusal is real; the cause is the compositor throttling a blanked
output, so no present mode can move it. Read the numbered items below only for the
two facts they settled, both of which still hold.**

Do not read the 20 Hz as a regression, and do not chase it on one boot. AGENTS.md already
records `present_hz` p50 **20.00** at `3af5832` against **78.20** at `08110da` on
code differing only by instrumentation, and 20.00 is one of those two arms — **those two
arms are asleep and awake.** What is *new* and drift-free is the within-boot ratio: the
presenter refuses 95 % of its own attempts while the device idles.

- **What `offered` counts — settled, and it is a real drop ratio.** `cadence_offered`
  increments only when `present()` is handed `Some(cpu)` *and* the frame's `seq` differs
  from the last offered one (`window_present.rs:779`). `draw_engine_window` builds that
  `cpu` from the `FrameSlot` entry unconditionally, so it is present on the resident rail
  even when the byte capture is elided — `capture_sampling full=1 light=…` does not
  suppress it. So `offered` is the count of **distinct device publishes that reached the
  window**, and `offered_hz=99` against `present_hz=20` is 80 % of produced frames not
  reaching the screen.
- **Whether 20.0 is ours — settled, it is not.** Nothing in the publish path paces at
  20 Hz. `ENGINE_WINDOW_REDRAW_POLL` is 2 ms and `about_to_wait` re-requests a redraw on
  that grid, which is the ~438 `present()` calls a second; the seq gate holds
  `engine_redraw_required` set across a `Busy` return, which is why every failed acquire
  is retried rather than dropping the frame. The 20 Hz is the compositor's release rate
  for a blanked output, reproduced by `vkcube`.
- **`PresentModeKHR::FIFO` with `min_image_count + 1`** (`window_present.rs:661`,
  `:709`). MAILBOX would decouple the acquire from the display, and is a capability
  question (`get_physical_device_surface_present_modes`), not a vendor one — but
  changing it changes pacing, so it needs the within-boot `offered/presents` ratio to
  score it, not Hz. **Awake, FIFO on this host tops out near 80 Hz against a device
  offering 99, so this is worth doing on latency and on the wasted acquires — but it is
  worth far less than the 5x the asleep reading implied. Score it with the panel held
  awake or not at all.**

**Goals 1-3 re-verified green by slug on the fix boot**, 1775 sliced lines:
`linear_load_span_exceeds_alloc`, `deferred_flush_lost`, `type11_seed_cache_absent`,
`draw_vk_nothing_stored`, `resident_chain_abandoned_cpu_recovery`, `frame_bgra_short`,
`chain_resident_land_fail`, all three `surface_resident_*` arm declines,
`type11_window_invent` and `BadGrid` — **all 0**. The host window rendered testufo
correctly and the byte-order check passed on pixels: the banner rendered **blue**
("SYNCING", up from red "SYNC FAILURE" on the previous boot) and the Blur Busters
header purple, which no failure line can see.

### The ~2 Hz Frame Rate Was One Bit In A Memory-Type Query

**Fixed by `b19074e`. 1.49 Hz → 18.76 Hz on the same workload, and the mechanism is settled by a
within-boot ratio rather than by the frame counter. Do not re-derive.**

`MemoryClass::Readback` asked for `required: HOST_VISIBLE | HOST_COHERENT` with `HOST_CACHED`
*preferred*. `select_memory_type` only ever tries a preference **together with** the requirement, so
on a device whose cached type is not coherent both preferences match nothing and it falls through to
the bare requirement. Vulkan guarantees a `HOST_VISIBLE|HOST_COHERENT` type exists, so this never
fails and never logs; it just puts every readback buffer in uncached memory.

Intel ANV is that device. `vulkaninfo` on the x86 dev host, five types over one 70 GiB DEVICE_LOCAL
heap: `0x07` = `DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT` and `0x0b` =
`DEVICE_LOCAL|HOST_VISIBLE|HOST_CACHED`, each twice, with a PROTECTED type between them. **Nothing
carries both bits.** The module's own doc comment had warned that an uncached CPU read is
"catastrophically slow" — and had assumed that could only happen on the discrete row.

The measurement, two boots either side, same workload (Safari fullscreen on
testufo.com/refreshrate, `.agents/repros/testufo-fps.sh`):

| | before | after |
|---|---|---|
| **`readback_bytes / readback_us`** | **460 MB/s** | **6137 MB/s** |
| `readback_us`, per second of wall clock | 495-790 ms (**70-86 %** of all phase time) | 125-139 ms |
| `readbacks` per second | 72 | **167** |
| `max_tranche_us` | 504 592 - 744 671 | 54 176 - 70 481 |
| `drain_duty tranches` per second | 6-9 | 53-62 |
| us per draw | 7 134 | ~3 200 |
| host presents, frame-grouped | **1.49 Hz** | **18.76 Hz** |
| guest's own opinion (testufo) | 35.000 Hz | 38.000 Hz |

Four things are worth carrying out of it.

**The drift-free number is the ratio, not the frame rate.** AGENTS.md already records that
`us_per_draw` has a 1.8x boot-to-boot spread and drifts upward within a session, so a sequential
before/after cannot score frames on this rig. `readback_bytes / readback_us` is two counters from the
*same* window of the *same* boot, and 460 → 6137 MB/s cannot be a session drift. The 12.6x frame
number is consistent with it and is not what establishes it. Note also that the after arm did **2.3x
more readbacks** in less than a fifth of the time, so the throughput gain is larger than the frame
count shows.

**The guest was never the limit, and its own instrument said so before the fix.** The guest reported
35 Hz while we presented 1.49, and after the fix it reports 38 while we present 18.76. It was
producing frames the whole time; `display_vbl not_claimed=28752 / delivered=29696` was the same fact
from our side. A 23x guest/host gap is a *statement about the host*, and it took one screenshot of a
page we did not write to get it — the arm AGENTS.md had listed as missing.

**`draw_phase` already carried the answer and had never been sliced for it.** `Phase::Readback`
brackets exactly `map_memory` + allocate + `copy_nonoverlapping` + `unmap_memory` — the fence wait is
`wait_us` and read 53-75 ms/s throughout. So "70-86 % of all draw time is one memcpy" was on the
always-on channel of every boot in this project's history, next to `readback_bytes`, and dividing the
two takes one command. Identify before add, again.

**A fixture more capable than the hardware cannot fail the way the hardware does.** `intel_igpu()`
gave type 1 `HOST_COHERENT|HOST_CACHED` together — invented, while the Apple fixture beside it was
transcribed from a live `vulkaninfo`. Every `MemoryClass::Readback` selection test passed against a
device that does not exist. Worse, the test named for this property,
`readback_prefers_cached_on_every_topology`, asserted that `preferred` mentioned `HOST_CACHED` and
that `required` contained `HOST_COHERENT` — both true, and *jointly the cause*. A test that reads the
query cannot see a preference that never matches; assert the **selected type**.

Two consequences of dropping the coherence requirement, both load-bearing:

- **Readers owe `vkInvalidateMappedMemoryRanges`, and a missed one returns the previous frame.** There
  is now exactly one reader, `pools::read_back_slot`, and the four rails (draw, `read_target_inner` /
  storage, compute storage-buffer, compute storage-image) all go through it. Patching only the draw
  rail took `vk_engine_parity` to 38/41 with the two informative failures both reporting the *prior*
  render — `a_view_swizzle_…` read `[203,91,17,255]` where `[17,91,203,255]` was due, which is exactly
  the identity result coming out of the swizzled draw. A missed invalidate does not fault; it silently
  hands back whatever the reader last read through that pooled buffer.
- **`vk_engine_compute` was 6/14 at HEAD and nothing had said so.** Every SSBO case failed
  `reason=vk_compute_exec_map_storage_readback vk_result=Mapping_of_a_memory_object_has_failed`: the
  compute rail reads back out of slots it takes from `acquire_staging`, which maps for the slot's
  lifetime, and `vkMapMemory` on an already-mapped object is `VK_ERROR_MEMORY_MAP_FAILED`. The same
  choke point fixed it by honouring an existing mapping. **Run both GPU suites, not just
  `vk_engine_parity`** — that regression survived because the last several sessions ran one of them.

`readback_memory` is the standing report, always-on and latched per memory type. A healthy boot reads
`readback_memory type=1 cached=1 coherent=0 topology=unified`; the degraded case is the typed decline
`readback_memory_not_cached` carrying the type and the `memoryTypeBits` it had to choose from. It
fires once per boot, before any slice mark a repro sets, so grep the whole log rather than a slice.

**Where the time goes now, from the same line — the bottleneck moved and this is the next work.** Per
second of wall clock, at ~280 draws/s and `duty` still 0.96-0.98:

| phase | us per second | note |
|---|---|---|
| `wait_us` | **246 000 - 265 000** | fence wait; the new largest single phase |
| `stage_us` | **148 000 - 163 000** | the 851 MB/s seed upload, mirror of the readback |
| `readback_us` | 125 000 - 139 000 | now cached |
| `acquire_us` | 31 000 - 61 000 | |
| `prep_us` | 26 000 - 33 000 | |
| `submit_us` | 24 000 - 26 000 | |
| `record_us`, `pipeline_us`, `descriptors_us` | ≤ 6 400 | noise |

`batch_opens=105 batch_joins=2` is unchanged: batching still gets ~1.02 draws per batch, and
`engine_delta` says why — `readbacks=167` against `draws≈280`, so Stores and intermediates alternate
and `begin_entry` flushes the open batch every time. That is a *scheduling* problem, not the join
predicate AGENTS.md's older note points at.

**`wait_us` scales with the bytes read back, at ~270 us/MB. It is the bytes.**

**This reverses a standing exclusion, and the way it was wrong is worth more than the result.** The
entry here used to read "`wait_us` is a fixed ~1.5 ms per readback submission and does not scale with
the bytes copied", drawn from 89 one-second windows of a live boot split into outer quartiles by
bytes-per-readback: 24 % more bytes bought 3.5 % more time, with `us/MB` moving inversely and
`us/readback` showing a tighter CV than `us/MB`. It concluded "cutting it means fewer submit-and-wait
round trips, not fewer bytes", which is the opposite of the truth and would have aimed a session at
batching — the one lever already refuted.

Re-run on 214 windows of the 2026-07-30 boot, the same split reproduces exactly and **both models now
fit equally**: `us/readback` CV **0.089** against `us/MB` CV **0.090**. Indistinguishable. That is the
tell, and it is the general point:

> **A ratio test has no power when its denominator barely varies.** `MB per readback` spans 4.49 to
> 5.20 across a whole live boot — a 16 % range against ~9 % residual noise. Neither normalisation can
> win, so whichever CV comes out lower is noise, and the first run's 0.042-vs-0.103 gap was read as a
> result. Before splitting by a quantity, print its range: if the split variable moves less than the
> scatter, the test cannot answer and the honest output is "no power", not a conclusion.

The experiment with power is `measure_draw_cost_against_pass_size` (`#[ignore]`d in
`vk_engine_parity`), which varies geometry over a **256x** range with everything else held: 16x16 costs
278 us and 1920x1080 costs 2386 us. Same submission, same fence, same one draw — **8.6x the time for
8000x the bytes.** No per-submission model survives that single comparison, and no observational slice
of a live boot can overturn it. Floor ~270 us, slope ~270 us/MB.

The two agree in absolute terms, which is the corroboration: the live boot reads **279 us/MB** against
the controlled **270 us/MB**. So of the measured 1356 us mean wait, the ~4.7 MB average readback
accounts for ~1270 us and the fixed floor for ~270 — call it 80/20. Batching therefore caps out at the
floor, 230/s × 270 us ≈ **62 ms/s of 286**, which is the same conclusion the instrument's own doc
comment reached by a different route.

**So the actionable form is: move fewer bytes.** At 1.08 GB/s of readback, the bytes cost ~292 ms/s of
fence wait *plus* the 135 ms/s `readback_us` memcpy that follows it — **~427 ms/s, 49 % of `draw_us`**.
And `store_routes` says where they come from: `surface_deferred` runs ~218/s against `readbacks` 230/s,
so **essentially every readback is a type-11 composite Store**.

`skip_readback` returns from `execute_draw_inner` *before* `Phase::Wait` (`render_post_wait_skips`), so
a composite Store that stopped reading back would drop both costs together, not one of them. That is
the next step, and the seed elision above was its prerequisite — what it still needs is a source for
the three consumers the readback currently feeds: `surface_cache` (present capture + the ~5 % of LOADs
that still seed), the deferred window's owned frame, and the guest writeback at flush.

**The hazard that retraction actually named, restated so the next attempt starts from it.** A
`LoadFromTarget` on a type-11 surface trusts the resident to hold the mapping's current pixels, and
**every non-draw writer to that mapping breaks it**: a blit into the surface, a compute storage
writeback, a guest CPU write through `mapper::write_mapping_bytes` all land in guest pages and/or
`surface_cache` and leave the resident untouched. With a CPU seed the LOAD picks those up; with
`LoadFromTarget` it silently renders on top of a stale frame, which is the black/torn-layer class. The
front-buffer ping-pong that boot reported is one instance of it, not the whole of it.

So the rail needs a **witness, not an argument**: a monotonic per-mapping content epoch that every
writer bumps, recorded on the resident when a draw stores into it, and `LoadFromTarget` taken only
when the two agree. That is maintainable only if the writer set is closed — which is the same
completeness question the flush-trigger backbone had to answer, and which "enumerate the paths inside
each reader, not the readers" says how to get wrong.

**That witness already exists, and the compute rail already implements the whole pattern. The render
rail's version is a port, not a design.** Hand-verified 2026-07-30 (a subagent produced the writer
survey; every claim below was then read at the cited line, because AGENTS.md records two subagent
audits on this rail that were flatly wrong):

- `MappingEntry.content_generation: u32` — `model/state.rs:567`. Bumped by
  `DeviceState::mark_mapping_written` (`state.rs:2618`), wrapping and skipping 0 so 0 means "no host
  write since attach". Reset to 0 at `state.rs:2466`, `2541`, `2608` (attach / re-attach / the path
  whose comment says the guest pages stay authoritative).
- **Every guest-page writer bumps it.** Callers: `mapping_write.rs:283` (`write_bgra8`), `442`
  (`write_rgba8_image_changed`), `549` (`write_raw_rows`), `1065` and `1100` (the `write_rect_raw_at`
  family, which is what `blit_exec` and `compute_exec`'s type-11 writeback reach guest pages through);
  `compute_exec/mod.rs:3705`; `exec.rs:1901` and `1903`.
- **No render path reads it.** Its only product reader is the compute rail's `seed_generation`
  (`compute_exec/mod.rs:1665`).
- `TargetIdentity::Surface` carries `generation` = **`map_generation`**, not the content epoch, and
  `ResidentTargetSlot` (`pools/mod.rs:486`) has `generation: u64` + `content_ready: bool` and **no
  content epoch**. So a render resident cannot currently answer "are my pixels current".
- `ResidentStorageImageSlot.generation: u32` **is** a content epoch. It is compared at
  `pools/images_and_registry.rs:181` (`generation_match: resident.generation == seed_generation`) and
  consumed at `exec_compute.rs:540`:

  ```rust
  let st = if generation_match { None } else { /* acquire_staging + write_staging */ };
  ```

  which is exactly the seed elision this section is asking for, shipped and tested on the compute rail.
- It is **fail-closed**. `exec_compute.rs:527` refuses with the typed
  `ComputeExecutionDecline::ResidentSeedGenerationLost` when the caller skipped the guest read on the
  strength of a generation that has since moved, rather than seeding a placeholder — "the named failure
  instead", in its own comment.
- `state.compute_storage_residency` is the mapping-keyed mirror of that currency, maintained by
  `note_storage_residency_writeback` (`compute_exec/mod.rs:955`). Read its comment before copying: the
  mirror stores `next(seed_generation)`, **not** the mapping-level content generation, precisely so
  that two disjoint sibling windows on one mapping (ping-pong canvases) cannot desync the pair — and it
  bounds the sibling count per mapping.

**Built and scored by `4c82c4d`. `stage_us` roughly halved, 95 % of type-11 LOAD seeds are gone, and
the screen is clean. Do not rebuild.**

The port is what the section above describes, with **one addition that the design as written would have
got wrong**. `content_generation` is not a sufficient witness on its own. `surface_cache` holds exactly
one entry per mapping, so a sibling Store at a *different geometry* replaces the entry an older
geometry's resident is being compared against — and it does that without writing a guest page, so
`content_generation` never moves and the stale resident reads as current. That is the same
one-entry-per-mapping hazard that produced the `deferred_flush_lost reason=cache_miss` black-layer
class, arriving from the other side.

So the witness is a separate `MappingEntry::surface_content_epoch`, strictly coarser than
`content_generation`: `mark_mapping_written` advances it (which closes the eight guest-page writers for
free, by construction rather than by enumeration), and `note_surface_content_published` advances it for
the deferred type-11 publish — the one writer that changes the pixels without touching guest pages. It
stays a separate field so the compute rail's `seed_generation` semantics are untouched.

`ResidentTargetSlot::content_epoch` is `Option<u32>`, `None` on create, recycle and **both**
`registry_mark_ready*` arms. The `Option` is load-bearing: epoch 0 is a legal mapping value ("nothing
published since attach"), so a bare `u32` sentinel would let a never-stamped slot match it. `None ==
None` is `true` in Rust, so the comparison guards `is_some()` on the mapping side — without that,
"no mapping entry" matches "never stamped" and the pass loads undefined memory as the guest's prior
frame. That test (`an_unstamped_resident_never_matches_a_mapping_with_no_epoch`) was verified to fail
with the guard dropped.

Note also what the Store had to gain first: a type-11 composite Store previously set **no
`target_identity` at all** — the first block's predicate is `chain_from_resident || (store_is_store &&
!writeback_guest)` and the second requires `mapping_id == 0` — so there was no resident to load from.
It now renders into the registry resident *and still reads back*, which is a configuration no test in
`vk_engine_parity` covered (every other `LoadFromTarget` case sets `skip_readback = true`).

Measured on one x86/Vulkan boot, Safari fullscreen on testufo.com/refreshrate, 45 s settle:

| | reading |
|---|---|
| `type11_seed_elided` / `type11_seed_uploaded` | **4872 / 249 — 95.1 %** (1734 / 37, 97.9 %, over a drag window) |
| `stage_us` per second | **p10 74.5 / p50 81.0 / p90 92.1 ms** (was 148-163) |
| `seed_upload_bytes` / `readback_bytes` | **0.39-0.60** (was **exactly ~1.0** — "seed MB equals readback MB in every row") |
| `target_evicts` | **0**, in all 81 windows |
| `FAIL` lines | **0** |
| `wait_us` per second | p50 **286 ms** — now unambiguously the largest single cost |
| host presents, frame-grouped | 22.77 Hz |
| guest's own opinion (testufo) | 41.000 Hz |

**The number that establishes it is the elision ratio and the broken symmetry, not the frame rate.**
Both are within-boot, so neither can be the `us_per_draw` drift. The seed/readback symmetry is the
better of the two because AGENTS.md had already recorded it as *exact* across five independent windows
on the old code; a ratio that was 1.03 and is now 0.44 is not a sampling artefact. 22.77 Hz against a
18.76 Hz predecessor is one sequential pair on a rig with a documented 1.8x spread — consistent, and
not evidence.

Three things this boot does **not** show:

- **Batching did not move.** `batch_opens=157 batch_joins=2` at ~1.02 draws per batch, unchanged. The
  `joins` predicate needs `target_rgba8.is_none()` *and* `skip_readback`, and a composite Store reads
  back by construction, so satisfying half of it buys nothing. The older note above predicting
  "`joins` gains its half" is correct and irrelevant — two conditions, one still false.
- **`target_evicts=0` is one boot's answer** to whether making composite Stores registry-resident
  causes churn. Three or four surface identities were live; a workload with many more mappings has not
  been run.
- **The remaining `wait_us` is untouched.** It is now 286 of ~865 ms/s of `draw_us` and is the next
  target. Per the measurement above it is per-submission, not per-byte — so it is fewer submit-and-wait
  round trips, and `skip_readback` for the type-11 Store is the change that would deliver them.

**One asymmetry that was designed for rather than discovered, and it held.** The deferred writeback rail
arms a window and does *not* write guest pages, so it does not bump `content_generation` at arm time —
the bump comes later at flush. A resident stamped at arm time therefore goes stale on its own flush even
though the pixels never changed. That is conservative (it falls back to the CPU seed, which is correct)
and it shipped that way; at 95 % elision the residual is not worth chasing.

The deferred route also **captures its epoch at the publish rather than re-reading it at the end of the
function**, which matters for a reason worth generalising: `evict_render_windows_to_cap` runs between
those two points and can land a sibling window, which writes guest pages and replaces the
one-per-mapping cache entry. Re-reading would stamp the resident with an epoch that already reflects
somebody else's write. A deferred obligation must record the epoch *of its own act*, not the epoch it
finds later.

### A Bounds Check That Charges A Stride For The Last Row Squares Window Corners

**Fixed by `e181e0f`, reproduced and scored either side. Do not re-derive.** This was the
dark-mode square-corner report, and the whole chain is four lines of always-on log.

`load_linear_texture_impl` bounded a linear texture level with `offset + row_stride * height`
against `TextureDescriptor::allocation_size`. What it *reads* is `(height - 1) * row_stride +
tight_row` — every path in this crate walks `gva + y * row_stride` for one tight row, and never
touches the padding after the last one. So the bound demanded a stride's worth of trailing bytes
that no row occupies, and refused allocations the guest had sized exactly right.

The measured instance, on an x86/Vulkan boot with the guest in Dark appearance:

    linear_sample_miss reason=linear_load_span_exceeds_alloc end=12496 alloc=12288
        task=1 ref=368 objtype=3 gva=0xca3850 fmt=0x1e geom=27x27 bpr=384
    linux_m2v_draw reason=draw_prepare_texture_resolve_missing stage=fragment index=3
        pipe=65 task=1 geom=1920x1080
    draw_encode_fail reason=draw_vk_nothing_stored class=no_metal pipe=65
    writeback_chain_rgba reason=resident_chain_abandoned_cpu_recovery mid=1 1920x1080

A 27x27 `RG8Unorm` mask at offset 0x850 in a 12 288-byte allocation: `2128 + 384*27 = 12496`
refuses, `2128 + 26*384 + 54 = 12166` fits with 122 bytes to spare. Refusing it drops the fragment
draw that samples it, and that draw is the WindowServer's full-screen composite — so the layer is
abandoned to CPU recovery and **the window renders with hard rectangular corners and no drop
shadow**. Rounded corners and window shadows are the same alpha, which is why both vanish together;
that pairing is the tell for this class rather than a second symptom.

Scored on two boots, same repro (`.agents/repros/darkmode-corners.sh`), light and dark captures of a
Finder window over bare wallpaper with the corners cropped and magnified 8x:

| | before `e181e0f` | at `e181e0f` |
|---|---|---|
| dark-mode top-left corner | hard 90°, no shadow | rounded, antialiased, shadowed — same as light |
| `linear_load_span_exceeds_alloc` | 2 | **0** |
| `draw_prepare_texture_resolve_missing` | 1 | **0** |
| `draw_vk_nothing_stored` | 1 | **0** |
| `resident_chain_abandoned_cpu_recovery` | 2-4 | **0** |
| light/dark diff bbox | `1695x611+54+0` (whole strip incl. menu bar) | `929x450+496+157` (the window) |

The bbox row is worth reading: with the composite draw failing, the difference between the two
appearances spanned the whole screen; with it fixed, it is confined to the window that actually
changed.

The rule now lives once, as `TextureLevelLayout::read_span`, because **three readers had
independently written the loose form**: the sampling loader and both mipmap paths, the write side
included — where the same over-strict bound refuses a level the guest can legally hold. A fourth
would have written it too.

Two transferable points:

- **A bounds check has to bound what is touched.** `stride * height` reads as conservative and is
  not: it converts correct guest allocations into refusals, and this one was routed through a path
  that drops an entire draw rather than degrading one texture.
- **The typed refusal is what found it, in one boot.** The caller printed
  `linear_sample_miss reason=guest_load` for all fifteen of the loader's refusals — object-list
  miss, undecodable descriptor, missing row conversion and unmapped guest page all under one word,
  with four different fixes. `3320dcb` made the callee carry its reason (`LinearLoadRefusal`,
  fifteen variants, each printing what it saw); the next boot named the check and the arithmetic
  was then a two-line check. This is the third time the "a reason the caller writes is not a
  reading" rule has paid here, and the first where it converted an open class into a fix directly.

### `MTLColorWriteMask` Is Tag `0x09`, And It Is Not The Corner Mechanism

Read off a live guest by `17e916c`'s census, honoured by `7c669d1` (Vulkan) and `0d91e23` (Metal).
Recorded because it is a real closed gap and because it was the leading hypothesis for the corner
class and **is refuted as that** — the corners still squared with it honoured, and `e181e0f` fixed
them without touching it.

`translate/coverage.rs` had recorded the field as `absent` with "where it sits in the type-7
colour-attachment block is unknown — an RE task, not a guess". The entry is self-describing, so
`note_color_entry_fields` now reports every tag it walks past. One x86/Vulkan boot, nine distinct
entry shapes:

    type7_color_attach_shape slot=0 nfields=5 tags=[00:4*,01:4,02:4,06:4,09:4*] unconsumed=2
    type7_color_attach reason=color_attachment_field_dropped tag=0x09 len=4 value=1

Tag `0x09` is `writeMask` by the argument that already names `0x01..0x08`:
`MTLRenderPipelineColorAttachmentDescriptor` has exactly nine properties and `MTLRenderPipeline.h`
declares them in the order those eight tags follow, so the tag is the property's one-based header
index. `value=1` is `MTLColorWriteMaskAlpha`. The identification is *checked*, not asserted:
`ColorWriteMask::new` refuses anything above `0xf` and reports
`color_write_mask_out_of_range` with the value, so a wrong identification says so by name.

Three things worth carrying:

- **Metal's mask bits are alpha-first and Vulkan's are red-first** — bit-reversed over four bits,
  not equal. A cast turns alpha-only into red-only. `metal-0.33`'s own `MTLColorWriteMask` bitflags
  (`Red = 0x1 << 3` … `Alpha = 0x1 << 0`) independently confirm the constants.
- **The mask is independent of `blendingEnabled`**, so it cannot ride inside
  `Option<BlendStateResource>` and is applied on both arms of `attachment_blend`.
- **Tag `0x00` rides every entry, `len=4`, `value=0` in every workload measured, and is still
  unconsumed.** Position and constancy suggest the attachment index, which is currently derived
  from entry order instead. Not honoured, deliberately: no boot has produced a nonzero value, so
  honouring it would be a guess with no evidence. It keeps reporting.

### The Guest Answering `:2222` Is Not Necessarily The One You Booted

`boot-x86.sh` fails with `Could not set up host forwarding rule 'tcp::2222-:22'` and exits
immediately when a previous guest still holds the port — and then `wait_ssh` succeeds against that
*previous* guest. A full repro ran to completion this way, drove the old guest (which already had
the appearance flipped by the run before it), and produced a complete set of scored arms with no
error anywhere. AGENTS.md already recorded this shape for `watchdog.sh`; it applies to every repro.

Two guards, both cheap. Wait for the *new* boot rather than for any QEMU — `until grep -q "first
frame presented" <bootlog>` beats `until pgrep -f qemu-system`, which matches the one still running.
And assert the guest's age before driving it: `.agents/repros/darkmode-corners.sh` reads
`kern.boottime` and aborts above `MAX_GUEST_AGE` (900 s).

Parse that field carefully. `sysctl -n kern.boottime` prints
`{ sec = 1785377446, usec = 560316 } Thu Jul 30 ...`, and the obvious
`sed -n 's/.*sec = \([0-9]*\).*/\1/p'` matches **`usec = `** because `.*` is greedy — it captures
the microseconds. Anchor on the brace. The wrong parse produced an age of 1 784 817 149 s, which at
least failed loudly; a value that merely looked plausible would not have.

### Appearance Flips Need To Be Asserted On Pixels, Not On A Preference

`defaults read -g AppleInterfaceStyle` returning `Dark` says a preference moved. One run scored a
"dark" capture that was **byte-identical** to its light one — `imgdiff` gave 0 differing pixels
across the whole frame — because a modal quit dialog had frozen the session and nothing repainted.
The preference had genuinely changed; the screen had not.

So gate the comparison on the repaint: `.agents/repros/darkmode-corners.sh` requires >100 000 pixels
differing by more than 64 before it will score anything, and aborts otherwise. Same rule as "validate
the specific thing you drove", applied to a state change rather than a workload — and note the
failure mode was a *dialog*, which in a downscaled thumbnail looks exactly like a healthy screen.

### 60 fps Is Reached, The Presenter Drops Nothing, And The Remaining Cap Is The Guest's Compositor

**The "remaining cap" half of this heading is RETRACTED — see "60+ fps Confirmed On A Second
Panel-Awake Boot" above. A later boot on the same code reaches `offered` p50 110 / max 120 with the
guest reporting 115 fps, so the 60 measured here is a per-boot state and not a ceiling. The four
refuted levers below still stand as refutations; what does not stand is the conclusion that 60 was
the guest's fixed choice.**

**Measured on one x86/Vulkan boot at `bbfb567` with the host panel held awake and the hold
verified (`PANEL: On 14/14 samples`).** This is the first frame-rate reading in this file taken with
its confounder controlled, and it settles the present-mode question in the negative: there is
nothing to gain there.

`.agents/repros/testufo-fps.sh /tmp/ufo-awake 45`, Safari fullscreen on testufo.com/refreshrate,
sliced to this boot by the `t=` reset (the log appends across boots and an unsliced read mixed 1314
windows from many boots into this measurement on the first attempt — slice first):

| | whole boot (121 windows) | the 46-window settle |
|---|---|---|
| `presents` | p50 60 | p50 **61**, min 58, max 62 |
| `offered` | p50 60 | p50 **61**, min 58, max 62 |
| `busy_acquire` | p50 0, max 51 | p50 **0**, max **1** |
| `busy_fence` | p50 0, max 2 | **0** |
| `present_hz` | p50 59.6 | p50 **60.00**, min 56.10, max 61.60 |
| `direct_frac` | — | **1.00** |

The guest's own opinion, read off the capture: **`61.000 Hz`, `Frame Rate 60 fps`, `Refresh Rate
60 Hz`.** Device side on the same boot: `drain_duty duty=0.121` at 481 tranches and 445 draws a
second, `store_routes surface_resident=252 type11_seed_elided=252` (100 % resident rail, 100 % seed
elision, `surface_deferred` absent), `capture_sampling full=1 light=7167` — one CPU capture in the
whole boot.

**`offered == presents` in every window, and `busy_acquire` is 0.** That is the within-boot ratio the
previous entry asked for, and it retires the entire present-mode work list: FIFO is not dropping a
single frame, so MAILBOX, a larger `min_image_count` and a blocking acquire would each buy exactly
nothing. Do not spend a session on `PresentModeKHR`. The 95 %-refusal reading that motivated it was
the blanked panel, per the section below.

**Goals 1-3 re-verified on the same slice**, and two of the three apparent hits are the documented
grep traps:

| sentinel | count |
|---|---|
| `deferred_flush_lost … reason=cache_miss` (the black-layer class) | **0** |
| `linear_load_span_exceeds_alloc`, `type11_seed_cache_absent`, `draw_vk_nothing_stored` | **0** |
| `resident_chain_abandoned_cpu_recovery`, `frame_bgra_short`, `chain_resident_land_fail` | **0** |
| `surface_resident_*`, `type11_window_invent`, `BadGrid`, `read_overrun`, `write_gate_no_spans` | **0** |
| bare `cache_miss` | 175 — **all `engine_delta`'s `sampled_cache_misses` field name** |
| `deferred_flush_lost` | 1, `reason=map_generation_drift` — the guard doing its job |
| `host_window_slate_end` | 1, `frames=567 covered=1` — the firmware-boot run, guest frame on screen throughout; **no uncovered `host_window_slate` line at all** |
| `present_unbacked` | 1, `mid=4 gen=0 reason=never_stored` — the documented boot-time class (4 over the full 524 s run; **all four had a resident carrying them — see below**) |

Byte order was checked on pixels, since no failure line can see it: the testufo stutter banner
renders **orange**, hyperlinks **blue**, the "Problems?" text **red**.

**The remaining cap is the guest's compositor, and every cheap explanation for it is already
refuted.** Do not re-derive these:

- *We advertise 60.* No — `DISPLAY_REFRESH_HZ` is **120**, and the guest latched it:
  `system_profiler SPDisplaysDataType` reads `UI Looks like: 1920 x 1080 @ 120.00Hz`.
- *Our VBL delivery is starving or jittering the display link.* No — `display_vbl` reads
  `window_hz=125.0` against `grid_hz=125.0` in **40 consecutive census windows** from t=38 s to
  t=349 s, with one 124.7 outlier. Rock steady, and 4 % *above* the advertised rate rather than
  below it. `not_claimed` tracks `delivered` at 1.04:1, which is the 8 ms limiter rejecting exactly
  the every-other 4 ms poll it is meant to.
- *It is Safari's rAF policy.* No — Mission Control, driven six times through `ctrl+up`/`ctrl+down`
  with no browser in the path, measures `offered` p50 **61**, min 59, max 62. Core Animation paces
  the same as the browser did.
- *The device has no headroom.* No — `duty=0.121`.

So the guest knows the display is 120 Hz, receives a steady 125 VBL/s, has an 88 %-idle device
underneath it, and composites 60. Across 343 windows only 2 exceed 65 offered frames/s (70 and 76,
both in a multi-layer burst phase). Whether macOS's paravirtualized display can be made to pace its
compositor above 60 is **unmeasured and has no named mechanism**; the four obvious levers above are
spent. It is also not required: the goal is 60+ fps stable and the guest reports 61.000 Hz while we
present every frame it produces.

One genuine contract mismatch is worth recording even though it is refuted as the cause: we advertise
120 Hz (8.333 ms) and deliver VBL on an **8 ms integer grid** (125 Hz), because
`DISPLAY_VBL_MIN_INTERVAL_MS` is whole milliseconds and the poll heartbeat is 4 ms — a grid that
cannot express 120 Hz at all. It errs fast, so it cannot starve the guest, which is why the steady
125.0 reading exonerates it. Deriving the grid from `DISPLAY_REFRESH_HZ` in microseconds would be the
principled form, and it needs a finer poll than 4 ms to be worth anything.

### A MapMemory2 Notification Is Not An Authorization

**FOUND AND FIXED by `76eaf66`, scored on a live boot: all five loss classes 0, and 14 of 14
undeclared writes confirmed by a later notification. Do not re-add the refusal.**

The guest's order is **allocate → install PTEs → use → notify**. `drain`'s own comment already
recorded the first half ("Map notify: PTEs already live"); what nobody had checked is that the FIFO
carrying the notification is ordered against *nothing that uses the memory*. So a `MapMemory2` span
cannot authorise anything, and six rails were treating "not in the span registry" as "not writable".

The decisive probe was `write_gate_late_map` (`5961a70`), which joins the write to the notification
that follows it — a reading the refusal site cannot take, because its evidence arrives *after* the
event. That is "an event count is not a state" pointing **forwards**: when a claim is "this never
gets declared", the probe has to outlive the declaration, not fire at the write.

    gva_write reason=write_gate_outside task=1 gva=0x1ada000 len=0x10000 … gpa_match=0
    write_gate_late_map task=1 gva=0x1ada000 len=0x30000 span_gva=0x1ada000 span_len=0x30000 late_ms=37

5 of 5 on the probe boot, at the **exact base address**, 0–29 ms later. The three refusals at
`0x1ada000`/`0x1aea000`/`0x1afa000` are 3×0x10000 of one `OP_COPY_BUFFER_TO_TEXTURE`, and the span
that follows is `0x30000` — the upload's total length, exactly.

What was being lost per boot, all one cause: a 192 KiB texture upload (`copy_region` aborted three
times), a whole deferred window (`guest=skip_uncovered`, 240x135 / 320x512 / 256x256), a linear
writeback up to 1.3 MiB, six glyph-atlas compute writebacks (`reason=linear_unmapped`, 79x52, 90x20,
8x8 …), and three flush obligations never armed (`guest_flush=0`).

**The tree had already reached this answer once, on one rail, and it was not generalised.**
`an_rgba8_store_outside_the_tasks_declared_span_still_reaches_guest_ram` records `exact=1155
no_spans=0 outside=893` over 2048 render Stores — **44 %** — and hand-exempted `write_gva_rgba8`
because refusing blanks the screen. Six rails were not exempted. When one rail carries a measured
exemption from a shared check, ask what the other callers of that check are doing.

**Two refuted hypotheses, both from this file, both dead:**

- *The owner declared it, so authorise on shared GPAs.* `gpa_match=0` on every refusal — tasks 0 and
  1 resolve those GVAs to **different physical pages**. `owners=[0]` was a virtual-address
  coincidence inside task 0's 64 MiB span. `tasks_covering` compares GVAs across address spaces and
  therefore cannot mean ownership.
- *Do not make the gate consult the page tables.* Still correct, and this is its conclusion rather
  than its violation: the fix is not a second check, it is to stop refusing on a notification. What
  bounds a host→guest write is the task's page table, which every writer already re-walks at write
  time and fails closed on.

`WriteGate::Outside` is now `WriteGate::Undeclared`, reported everywhere and refused nowhere;
`gva_write_allowed` is deleted. All six rails report through one helper
(`gva_mem::report_undeclared_write`) so they cannot drift apart, with `via=` naming the rail. Two
joins make the two possible worlds legible: `write_gate_late_map … late_ms=` (the guest did notify,
we were early) and `write_gate_never_declared … age_ms=` (evicted from the ring uncovered — the
shape the open memory-corruption signature would take). Ranges **coalesce** when they overlap or
touch, because `copy_region` and the linear fallback write a region one row at a time and an
exact-range dedup would file 135 entries for a 135-row texture.

Scored on one x86/Vulkan boot at `76eaf66` (`PANEL: On 15/15`, `present_hz` p50 **60.00**, guest
testufo **60.000 Hz / 58 fps**, 2146 sliced lines):

| | at `5961a70` | at `76eaf66` |
|---|---|---|
| `write_gate_outside` / `copy_region_write_io` | 5 / 5 | **0 / 0** |
| `skip_uncovered` | 1 | **0** |
| `linear_unmapped` | 8 | **0** |
| `guest_flush=0` | 3 | **0** |
| `write_gate_undeclared` (permitted, reported) | — | 14 |
| **`write_gate_late_map`** | — | **14 — every one confirmed** |
| **`write_gate_never_declared`** | — | **0** |

`late_ms` was 0 ms ×5, 1 ms ×3, 2 ms ×4, 8 ms, 37 ms. The 14/14 ratio is **within-boot**, so it is
not subject to this rig's documented cross-boot spread. Twelve goal-1-to-3 sentinels all 0, and the
captured frame is correct on the checks no failure line can see: banner **orange**, hyperlinks
**blue**, Blur Busters header **purple**.

**Replicated on a second boot at `98c8725`** (`PANEL: On 14/14`, `present_hz` p50 **111.10**, max
119.00, guest testufo **119.968 Hz / 120 fps**, 2316 sliced lines): `write_gate_undeclared` **17**,
`write_gate_late_map` **17** — again every undeclared write confirmed by a later notification — and
`write_gate_never_declared` **0**. The same twelve goal-1-to-3 sentinels are 0, `deferred_flush_lost
… reason=cache_miss` is 0, and the failure channel is 59 lines led by the documented-benign
`cmd_task_ambiguous` (11), `gva_zero_pfn` (10) and `task_walk_ambiguous` (4). A longer
`darkmode-corners.sh` window on the same boot read **71 of 76**, the five outstanding being windows
still inside the ring rather than losses.

So the ratio holds at two boots and 31 of 31 joined writes. `write_gate_never_declared` — the arm
that would carry the open memory-corruption signature — is **still untested live**, in the same
sense that `present_unbacked`'s `carried=nothing` is: it has never once fired, so its report path
has no positive evidence behind it.

**A graceful-degradation path was hiding behind the gate, and removing the gate exposed it.** The
compute rail's early return also happened to catch genuinely *unmapped* GVAs, and that degradation
is correct. It is kept and now keyed on the condition itself: `write_linear_guest` returns
`LinearWrite::{Written, Unmapped, Failed}` instead of `bool`. This is "one status for N checks"
being load-bearing rather than merely untidy — the only caller able to degrade was doing so off a
proxy that also caught healthy writes. When deleting a check, look for what else was riding on it.

`MemError::OutsideMap` became unconstructible and is deleted (33 → 32 slugs).

**Two fixture traps, both recorded in the tests.** A one-level page walk masks its index to the
entry count, so `4096 * page` aliases index 0 and an "unmapped" assertion built on it passes for the
wrong reason. And `note_task_map` ignores a zero base as a sentinel, so a confirmation at GVA 0 can
never fire.

**Diff hygiene, since this cost a round.** `rustfmt --edition 2021 --style-edition 2024` normalises
pre-existing drift across the whole file *and* recurses into `mod` children: a first pass produced
111 hunks across 7 files including one this change never touched. The fix is not to hand-revert
hunks — format a **pristine checkout the same way** to get "HEAD + drift", then `git merge-file` only
the difference. 111 → 41 hunks, and the semantic hunk count at `-U3` then matches the merged
`git diff` exactly, which is the check that it worked.

#### Superseded: the original reading of this class

**Measured on one x86/Vulkan boot at `37365a0`. `named_mapped == pages` in every refusal. These are
FALSE REFUSALS and they drop real guest work.** The mechanism of the fix is *not* settled; what is
settled is which of the two candidate readings applies.

`gva_write_gate` refuses a write when the writing task has filed at least one `MapMemory2` span and
none covers the range. Five such refusals a boot each abort a whole guest upload
(`blit_exec.rs:1168` → `MemError::OutsideMap` → `BlitStatus::GuestIo`, no partial work, no recovery).
`37365a0` added the state read that separates "the gate is over-strict" from "the address was never
reachable anyway". Every line came back the same way:

```
gva_write reason=write_gate_outside task=1 gva=0x1ada000 len=0x10000 owners=[0] own=4
    pages=16 named_mapped=16 owner_mapped=16 via=runtime/blit_exec.rs:1168
```

| refusal | pages | named_mapped | owner_mapped | own |
|---|---|---|---|---|
| `0x1ada000 +0x10000` | 16 | **16** | 16 | 4 |
| `0x1aea000 +0x10000` | 16 | **16** | 16 | 4 |
| `0x1afa000 +0x10000` | 16 | **16** | 16 | 4 |
| `0x3d23000 +0xf00` | 1 | **1** | 1 | 38 |
| `0x38ab000 +0xf00` | 1 | **1** | 1 | 155 |

**The writing task's own page tables resolve every page of every refused range.** So the write would
have landed exactly where the guest asked, in the guest's own mapped memory, and `task_map_spans` is
**not a complete authorization set** — a premise `drain/mod.rs`'s map site asserts in a comment and
nothing verified. `own=155` on the last row rules out "task 1 declared nothing"; it had filed 155
spans and none covered.

What is lost is concrete: the first three are `opcode=0x12c` (`OP_COPY_BUFFER_TO_TEXTURE`) at
offsets 0 / 65536 / 131072 — **three consecutive 64 KiB rows of one 192 KiB texture upload** — and
the last two are `opcode=0x12e` (texture→buffer) at `size=240x135x1`, thumbnail-shaped.

Four facts from the same slice fix the decode, so do not re-litigate them:

- `map_memory2_key word=0x0 dec=0 gva=0x101000 len=0x4000000` and `word=0x1 dec=1 gva=0x9f9000
  len=0x20000`. `0x1ada000` is inside task 0's 64 MiB declaration and outside task 1's 128 KiB one.
  These are the exact values `only_the_writing_tasks_own_spans_authorise_a_write` was written from.
- `define_task root raw=0x1 task=0 dir=0x4389e2` against `raw=0x2 task=1 dir=0x45789e` — **different
  directory roots**, so tasks 0 and 1 are genuinely separate address spaces.
- `define_task` raw words are `0x1` then strictly even; `map_memory2_key` words include `0x5,0x7,0x9`.
  So `MapMemory2` names the slot directly and `exec_indirect2`'s word is in that same slot-id space:
  `task=1` is the **correct** resolution of `raw=0x1`. **This does not reopen `task_id >> 1`** — all
  four removed arms stay removed.
- `owner_mapped == pages` too, so *both* tasks resolve the range. Consistent with a shared buffer
  pool: task 0 declares it, task 1 also maps it and uses it.

**RESOLVED — the paragraph below is the superseded plan, kept only because its warning is still
right. The lead it proposes (compare resolved GPAs) is refuted: `gpa_match=0` on every refusal.**

**Do not "fix" this by making the gate consult the page tables.** That defeats its purpose exactly.
The gate is meaningful *because* page tables resolve more than `MapMemory2` declares — a task's own
heap is mapped in its address space, and writing there through a mis-decoded GVA is the corruption
class this guard exists for. `write_span` already fails closed on an unmapped page, so a page-table
check adds nothing the writer does not already do.

The lead worth pursuing instead: **a `MapMemory2` span plausibly authorizes physical pages, not one
task's GVAs.** If task 0's declared range and task 1's usage resolve to the *same GPAs*, the honest
gate compares resolved GPAs against the declared spans' GPAs — which authorizes these five writes
while still refusing a write into an undeclared heap. That is **unmeasured**: `owner_mapped=16` says
task 0 resolves 16 pages there, not that they are the same 16 physical pages. Extending
`probe_write_gate_pages` to compare the two GPA sets is the next measurement and it is cheap — both
walks already happen in that function.

### Goal 2 Has A Cheap Repro At Last: A Finder Icon Composited Under Load

**Found 2026-07-30 on one x86/Vulkan boot at `98c8725`. The subject is a 64x64 icon rather than a
1920x1080 layer, the trigger is one ssh command, and nothing in the always-on channel reports it.**

Every previous handle on the black-framebuffer class was a whole compositing layer, caught by
dragging a Safari window and hoping the capture landed on the bad swap buffer. This one is small,
cheap and repeatable, and it reproduces the same family of artefacts.

The recipe, on a guest already at a real session:

```sh
ssh macos-vm 'open -a Safari "https://testufo.com/refreshrate"; sleep 35;
              killall Finder; sleep 4; open ~; sleep 6'
```

then capture the host window and magnify the icon row of the Finder window. The load is what
matters; `killall Finder; open ~` is only a way to force every icon in the window to be composited
*while* it is running.

Measured, in one such capture at 1280x719, cropped `640x110+340+140` and magnified 180 % with
`-filter point`:

| icon | rendered |
|---|---|
| Desktop | a small solid **red** square |
| Documents, Downloads, Movies | correct |
| Music | a tiny dark glyph fragment, upper-right of its cell |
| Pictures | a narrow solid **black** vertical bar |
| Public | very nearly absent |

An earlier capture in the same boot, at a different moment, had a *different* set wrong and a
different set right — Pictures correct while Public was a full-size black rectangle, Desktop a black
rect with a white glyph, and four others shrunken. **Which icons come out wrong varies between
composites; that they come out wrong does not.** That is the same "one of the two swap framebuffers"
intermittency the goal statement describes, in a form small enough to magnify.

Read the *shapes*, because they are the finding and they rule out a whole family at once. The wrong
icons are **solid single colours** (a red square, a black bar) and **shrunken fragments of the
icon's inner glyph with the blue folder body absent** — Downloads' download-arrow ring at a fraction
of its size, Movies' film glyph, Desktop's monitor screen. A correct icon has both layers. Stale
content would render as a *previous icon*; it does not. Format conversion, subsampling and tiling
all predict per-pixel error and these are whole-cell.

**Three hypotheses are already refuted, each by a controlled arm on this boot. Do not re-derive.**

- *The idle resident sweep ages something out.* A Finder window left completely untouched was
  captured at t = 4, 20, 40, 70 and 110 s. **All six icons correct in all five frames.** So
  `maintain_idle_residents` and every other decay mechanism is out: a static window does not rot.
- *The appearance flip does it.* `defaults write -g AppleInterfaceStyle Dark; killall Dock`, then a
  capture 14 s later: **all six icons correct.** The flip forces a full icon re-composition and is
  harmless on its own. This matters because the defect was *first* seen in a dark-mode run and the
  obvious reading was that dark mode caused it.
- *It is dark mode at all.* The `darkmode-corners.sh` **light** capture was already corrupt —
  Desktop and Documents ghost-glyphs, Pictures and Public narrow orange bars, orange being exactly
  the wallpaper colour behind them. Both appearances, same defect.

What separates the clean arms from the reproducing one is that the reproducing arm had a **heavy GPU
workload running while the icons were composited**. In the clean arms the window was composited on
an idle device.

**Nothing reports it.** The failure channel over the reproducing slice (1804 lines) is 2
`cmd_task_ambiguous`, 1 `task_walk_ambiguous`, 1 `gva_zero_pfn` and 1 `deferred_flush_lost
kind=compute mapping=47 16x16 … reason=map_generation_drift` — that last being the guard working, at
a geometry no icon has. `linear_deferred_dropped reason=retired` fires once. Three or four icons are
wrong and the device says nothing about any of them, which makes this a silent loss under the ground
rules and means the next step is a probe rather than another slice.

Where to point it: icons reach the screen on the **compute** rail, not the render rail. The slice
carries 233 `compute_linux`, 73 `compute_stage_resident_sample` and 32 `compute_stage_resident_skip`,
and the icon windows are visible in them by geometry — `mapping=7 32x32 fmt=0x73 bytes=8192` and
`mapping=7 64x64 … bytes=32768`, i.e. 8 bytes per pixel, `Rgba16Float`.

Two things were checked there and are **not** the answer, so they do not need re-reading:

- **`ComputeStorageResidencyKey` includes `width`, `height` and `pixel_format`**, so the same mapping
  at two geometries is two keys. The one-entry-per-mapping hazard that produced the black-layer
  `cache_miss` class does not apply to this map.
- **`invalidate_storage_residency_window` tests intersection, not exact match**
  (`key.span_end <= lo || key.surface_offset >= hi`), so an overlapping guest write does drop the
  mirror. The comment above the skip gate in `compute_exec/mod.rs` calls it "exact-window
  invalidation" and is stale — the code is right and the comment is wrong.

**It is persistent, and that is what makes it tractable.** Two captures 25 s apart of an untouched
screen are pixel-identical in the icon row — the same red square, the same black bar, the same tiny
fragment, the same three correct folders. So this is not a frame caught mid-composition, and the
whole "we presented a partial composite" family is out. The wrong pixels are **computed once and
then reused**: the guest asks us to composite an icon, we return the wrong image, the guest caches
it, and it renders from that cache until something forces a recomposite. `killall Finder; open ~`
on an idle device repairs every icon, which is the same statement from the other side.

That narrows the subject to **one compute dispatch's output**, which is a far smaller thing than a
present path or a swap chain. It also explains the intermittency without any timing argument: which
icons are wrong is decided once, when they happen to be composited under load.

Note what is *not* affected in the same window: the sidebar's small SF-Symbol glyphs (Recents,
Applications, Desktop, Documents, Downloads) all render correctly in the same frame as the broken
64x64 folder icons. Whatever this is, it does not touch every compute-composited image.

**Two mechanisms were nominated and eliminated by reading, so they need no boot.** Both are the
shape a load-dependent defect invites, which is why they are worth recording as dead:

- *Staging buffers recycled under a dispatch still reading them.* `recycle_staging` does drain every
  live slot back to the free pool with no fence check — but all four of its call sites are unit
  tests or the pre-submit `force_loss` arm, where nothing has been submitted. The real path moves
  `staging_live` into the submission entry (`std::mem::take`), which is retired on its fence. Pool
  wrap-around under load is therefore not reachable here.
- *The reinterpret-sibling sample source.* `compute_stage_tex` will serve a sampled view from a
  *different* resident of the same byte window when the two agree on height and row bytes — matching
  on neither width nor format, which reads exactly like the kind of guess that produces a wrong
  image. It is sound in principle (same window, same height, same row stride means the same bytes,
  and the guest asked for two views over them), and in any case it **fires 0 times in both corrupt
  arms**, so it is not this defect. Its line is `compute_stage_resident_reinterpret`.
- *The fail-closed arm catching an eviction between the gate and the acquire.* The runtime checks
  `compute_resident_storage_generation` under one `lock_engine()` and the engine acquires under
  another, so an LRU eviction can land in between; the design covers that with
  `ResidentSeedGenerationLost`. That decline fires **0 times** in both corrupt arms, so the window is
  not being taken either.
- *A fresh resident matching a stale mirror at generation 0.* `ensure_resident_storage_image`
  creates evicted-and-recreated residents at `generation: 0` with `layout: UNDEFINED`, and the seed
  skip gate is `mirror == engine_generation` — so a mirror holding 0 would skip the seed into an
  uninitialized image, which fits "solid colour" exactly. It cannot: every mirror insert goes
  through `next_mapping_content_generation`, which skips 0 the way `mark_mapping_written` does, or
  through `flush_one`, whose value came from an armed window. The newly-created arm also hardcodes
  `generation_match: false`.

**Do not chase the `storage_format_specialize` bytes-per-pixel disagreement — it is on a different
population from the icons.** The line reports three specialization strategies in the reproducing
slice: `specialized=Rgba16Float` (9x, `guest_bpp=8 shader_bpp=4`), `specialized=Unknown` for
`Bgra8Unorm` (15x, 4 and 4) and `specialized=Rgba8Unorm` (6x, 4 and 4). Only the `Rgba16Float` one
disagrees about bpp, and a ratio of 2 next to half-width bar artefacts reads as a strong lead.

It is not one, and the arithmetic says so. The icon dispatches are identifiable by geometry and
byte count: `66x66` at 17 424 bytes, `64x65` at 16 640, `46x28` at 5 152, `44x26` at 4 576 — all
exactly `w * h * 4`, i.e. **4 bytes per pixel**, which puts them in the `Rgba8Unorm` group where
`guest_bpp == shader_bpp`. Six such dispatches for six icons. The 8-bpp `Rgba16Float` windows
(`mapping=7 32x32 bytes=8192`, `64x64 bytes=32768`) are a separate population that happens to share
the slice. This is the "count what your A/B actually changed" trap in miniature: two anomalies in
one log are not therefore the same anomaly, and the byte count separates them in one command.

**The dispatch is not being dropped: `compute_record` has fired 0 times in the entire accumulated
log.** That is 17 MB spanning every boot ever run on this machine, so it is not a slice artefact and
it eliminates the whole "the icon was never composited, so the guest cached whatever was in the
buffer" family — which is otherwise the most natural reading of a solid-colour output.

Getting there needed one correction worth carrying, because it nearly produced a false elimination
from the *other* direction. `note_compute_refusal` reports through `.fail_once(pipeline_ref)`, so a
pipeline that refuses once is **silent for every later refusal in that process**. A slice taken well
after boot therefore cannot see a refusal that started before it, and "0 in my slice" would have
meant nothing at all. A **zero over the whole log** does survive the dedup, which is why the check
has to be run against the file rather than the slice. The same reasoning applies to every
`fail_once` line in this crate.

Two sites on that path look like silent drops and are **not** — worth stating because both were
written up as violations here before the propagation was traced. `resolve_dispatch_dims` failing
abandons the dispatch with only an `observe::line` (the `REIMS_VGPU_DRAW_LOG=1` tier), and the
`tg_x == 0 || …` guard below it returns `BadGrid("compute_vk_zero_dims")` with no line at all. Both
statuses propagate out of `execute_dispatch_linux` to `handle_compute_record`, which calls
`note_compute_refusal` on every non-`Ok` record, so the always-on channel does name them. What the
verbose line adds is the detail — the actual grid and threadgroup values — and what the arrangement
costs is that the always-on side is deduped per pipeline, so it reports *that* a pipeline refused and
never *how often*. That is the designed behaviour of a rail-boundary refusal, not a hole.

So the specialization mismatch is real, unexplained, and **not** the icon defect. What is left is
that the icon dispatches look entirely ordinary on every line this device emits — `3` samples and
`2` skips per geometry, `access=write_only seed=1`, no decline — which is the finding: the next step
is a probe on the dispatch that produces a guest-visible compute output, reporting its identity,
whether the seed was skipped and whether its resident was freshly created, so a corrupt icon's
geometry can be joined to its own dispatch. Nothing today permits that join.

**The standing shape of the defect, after all of the above.** Every icon dispatch executes and
returns `Ok`; three of six produce correct pixels and three do not, in the same window, from the
same shader, with every logged parameter identical. That combination rules out the shader itself —
a translation defect would not spare half the icons — and leaves two places for the error: **what
the dispatch sampled**, or **where its output landed and was read back from**. The sample side is
the one with machinery that can substitute an image (the resident-sample skip and the reinterpret
sibling); both read 0 in the corrupt arms, so if it is the sample side it is a path neither of those
lines covers.

One weaker correlation is worth recording *as* weak, because it will look stronger than it is on
re-reading. `type4_pages_stale` ("task PT translation moved; rebuilding") fires 3 and 4 times in the
two corrupt arms and **0** times in both clean arms — a 2-vs-2 split. Against it: the surfaces whose
pages moved are 500-page ones (`sid=14`, `sid=90`), not the icon surfaces; load is an obvious
confounder, since the corrupt arms are also the busy ones; and the line fires on *re-derivation*, so
it is an event count and not a state. Treat it as an observation about the environment the defect
occurs in, not as its mechanism.

**RETRACTED — the compute rail is not where this defect lives, and the "standing shape" above is
the shape of a wrong subject.** Measured on one x86/Vulkan boot at `2b32fd1` with
`REIMS_VGPU_CONTENT_PROBE=1`, two icon composites in the same boot. Do not resume the sample-side /
output-side fork above; it was a fork inside the wrong rail.

The whole attribution to compute rested on a *geometry* correlation: icon-shaped `WxH` values with
`bytes == w*h*4` appearing in `compute_linux` lines during a corrupt arm. Nothing had ever read what
those dispatches actually produced. `observe::content_summary` reads it, and every compute output in
the boot is a real image:

| round | on screen | compute outputs landing in guest pages |
|---|---|---|
| 1 | **Movies icon absent**, five correct | 4 icon-geometry mappings (44x26, 46x28, 66x66, 64x65), each `distinct=64+`, `nz` 50-65 % — folder icons with transparent margins |
| 2 | **Desktop absent, Documents a shrunken dark fragment**, four correct | **two 16x16/28x28 glyphs and nothing else** |
| whole boot | — | 25 outputs total, none degenerate |

Round 2 is the one that settles it: the folder icons **did not touch the compute rail at all** in the
arm that corrupted two of them. Round 1 is the converse check — compute *did* run there, and what it
produced was correct. So compute is exonerated from both directions, which is the pair of readings
the older entry never had.

Three further facts from the same boot, all of which survive the retraction:

- **The victim moves, so it is ours.** Two `killall Finder; open ~` rounds in one boot corrupted
  disjoint sets: Movies in the first, Desktop and Documents in the second, with Movies *correct* the
  second time. A defect that picks a different victim from the same content on the same guest is not
  content-specific and not a guest-side icon cache holding a bad image. (`~/Movies` exists and is the
  only home folder without an ACL — a red herring; it rendered perfectly in round 2.)
- **It is persistent within a round.** Two captures seconds apart are identical, so the frame is not
  caught mid-composite. Combined with the moving victim: the wrong result is decided once per
  composite and then held.
- **The failure channel is silent.** Both slices carry only the documented-benign
  `gva_zero_pfn` / `cmd_task_ambiguous` / `task_walk_ambiguous` / `type4_pages_stale` /
  `present_named_pages` / `present_order_hold`. No `linear_sample_miss`, no
  `draw_prepare_texture_resolve_missing`, no `draw_vk_nothing_stored`. This is a silent loss under
  the ground rules.

**The shrunken fragment is the informative symptom and it names a geometry, not a colour.** Documents
rendered as a small dark angular piece in the top-left of its cell with the rest transparent. That is
what a sampler produces when a texture's *allocated extent is larger than the region actually filled*
— the quad maps [0,1]² over the whole extent, so content confined to a corner arrives shrunken into
that corner. An absent icon is the same failure with the filled region empty. Both are one class:
**the sampled texture bound for the icon did not hold the icon over its full extent.** That is a
statement about the draw rail's sampled-texture load, which is where the next probe goes.

`content_summary`'s `quad=nw/ne/sw/se` field exists for exactly this and was added because a scalar
`nz` cannot see it: a 64x64 texture filled only in its top-left sixteenth reports `nz=256` — an
entirely ordinary count — and `quad=256/0/0/0`. The same 256 texels spread evenly report
`quad=64/64/64/64`. Pinned by `a_shrunken_top_left_image_is_visible_only_in_the_quadrants`.

Two process points, both of which this cost:

- **A geometry correlation is not an attribution.** "Icon-shaped dimensions appeared in the compute
  log during a corrupt arm" survived four commits of investigation and a written-up "standing shape"
  without anyone reading the pixels those dispatches produced. The refutation cost one boot once a
  probe reported *content* instead of *shape*.
- **Reproduce twice in one boot before believing a victim list.** The single most useful thing this
  boot did was run the repro a second time. One round gives a victim; two rounds give the fact that
  the victim moves, which is worth more than either round alone and is immune to the cross-boot drift
  this rig is documented to have.

### The Icon Class Is A Sampled Bind Falling Off Its Resident Onto Guest Pages That Hold A Fragment

Measured on two x86/Vulkan boots at `757e56c` and `ed8e2c2` with `REIMS_VGPU_CONTENT_PROBE=1`, panel
held awake, workload asserted by process count. This is where the class stands; **no fix yet**, and
the last step is not established.

`sampled_content` names which of `resolve_sampled_source`'s thirteen rungs served each bind, plus the
content fingerprint on the CPU-bytes rung. Keyed as a *transition* per (texture, geometry) rather
than a first sighting, because the event is the switch between rungs and a first-sighting latch goes
quiet after one.

**Sort the transitions by order before reading them, or the normal case reads as the defect.** Of 94
textures that saw more than one rung, the ones worth looking at split exactly in half and the two
halves mean opposite things:

| order | count | what it is |
|---|---|---|
| empty `bytes` **then** `resident` | 17 | **normal.** A surface being created. `type4 pages sid=63 … sample0_nz=0/16` then `mapping_gpa_span … changed=1` then the empty sample then `type11_load_seed outcome=guest_pages`: the guest allocated a 96x96 IOSurface and sampled it before drawing into it. Zeros are the correct answer. |
| `resident` **then** sparse/empty `bytes` | **17** | **the defect.** A texture whose content the GPU holds is later bound from guest memory that holds a fragment of it. |

Six of the seventeen regressions are `64x64 mid=0` — linear textures, icon-shaped — and they land at
`t=71619..71986`, bracketing the folder-icon composite at `t=71627..71772`. Their CPU bytes carry
`nz=0`, `59`, `290`, `618` and `784` of 4096, every one of them confined to the north-west quadrant.
On screen that is an absent icon and a small dark fragment in the top-left of the cell, which is
exactly what five rounds of the repro show.

The rung they fall off is `try_sample_deferred_gva`, and its two exit conditions say why it is
fragile: it needs `state.gva_deferred_flush.get(&gva)` to still hold a window **and**
`resident_content_ready`. So it serves the resident only while the deferred window is armed. Once the
window goes, the bind reads guest VA — which is correct only if the window's content was *flushed*
there. If the window was dropped rather than flushed, guest pages keep whatever the CPU last wrote,
and for an icon that is a small rasterised fragment.

`deferred_flush_lost` is **0** in that slice, so nothing reported a loss — which is consistent with a
window being *dropped* (supersede, eviction) rather than failing to flush. AGENTS.md's own supersede
rule rests on the argument that "those bytes were never observable without a flush, since any reader
would have taken the window first". **That argument has a hole exactly here**: a reader that requires
`resident_content_ready` does *not* take the window when the resident is not ready, and falls through
to guest pages instead. The soundness argument assumes a total reader; the reader is conditional.

One confirmed loss of this shape is in the same window and is *not* silent:

```
deferred_window_page_drift gva=0x33d000 task=7 140x130 trigger=linear_flush ref=5
    armed_pages=36 live_pages=36 moved=36 guest=refused @71840
```

All 36 pages moved under a 140x130 linear compute window and the flush refused — the guard working,
and the content never reached guest memory. That is the documented page-count-cannot-see-it hazard
(`armed_pages == live_pages` while every PFN moved) firing during the icon composite.

**Three readings that were nearly written up and are wrong. Do not repeat them.**

- **A `quad` with two zero quadrants is not a defect.** Text labels and glyph atlases legitimately
  carry content only in their top rows: `185x28 quad=479/469/0/0`, `2048x32 quad=1529/1510/0/0`,
  `256x28 quad=406/0/0/0` are all ordinary. What survives is `nz` far below the extent *on a texture
  that also binds from a resident*, not "content sits at the top".
- **An empty first bind is not a defect** — see the creation-order half of the table above. This one
  was half-written before the ordering was computed.
- **A texture ref is not a stable identity**; the guest recycles object refs, so "the same ref was
  served two ways" could be two textures. What rules that out here is the *span*: several regressions
  flip within 0-1 ms and one flips five times in 58 ms. Quote the span whenever a per-ref claim is
  made.

What is **not** established: why the window is gone by the second bind, and whether the six 64x64
regressions are the six folder icons rather than merely coincident with them. The join from a screen
cell to a texture ref still does not exist; the argument is geometry plus timestamp plus the shape of
the fragment, which is circumstantial. The next measurement is the fate of a deferred GVA window
between the two binds — armed, flushed, superseded or evicted — which nothing currently reports.

**REFUTED by a control arm on the very next boot (`41c2423`). The `resident` → sparse-`bytes`
transition is normal traffic, not the defect. Do not re-derive it, and do not spend another boot on
`try_sample_deferred_gva`.**

That boot rendered **all six icons correctly** and still produced **18 regressions**, three of them
`64x64 mid=0`, carrying the *identical* fingerprints to the corrupting boot:

| | corrupting boot (`ed8e2c2`) | clean boot (`41c2423`) |
|---|---|---|
| regressions | 17 | **18** |
| of which `64x64 mid=0` | 6 | **3** |
| their fingerprints | `nz=0`, `290 quad=290/0/0/0`, `784 quad=784/0/0/0`, … | **`nz=0`, `290 quad=290/0/0/0`, `784 quad=784/0/0/0`** |

Byte-identical `quad` signatures on a screen with no defect. So a 28x28 block of content in the
top-left of a 64x64 texture is simply what these textures contain — the guest rasterises art into a
sub-rect and the draw picks it out with texture coordinates — and the "shrunken into the top-left"
reading of the *screen* symptom does not transfer to the *texture*. Two different things were being
called the same shape.

And the typed fall-through says the proposed mechanism cannot fire at all. Across the whole slice:
`no_window` **191**, `window_geometry` **1**, `owner_object_type` **0**, and
**`resident_not_ready` 0**. The arm named as "the one that matters" — a window armed while the
resident holds nothing — never happened. `no_window` dominating is the benign case by construction:
no deferred window means guest pages are authoritative and reading them is correct.

Three transferable points, in order of what they cost:

- **Run the control arm before writing the mechanism up.** The regression population was computed on
  a boot that reproduced, and every number in it looked like a defect. The same computation on a
  clean boot returns the same numbers. One boot with a known-good screen would have caught it at any
  point, and it was available the whole time — this repro corrupts on some rounds and not others, so
  a clean arm costs nothing but re-running it.
- **Instrumenting the branch paid for itself in the negative direction.** `41c2423` typed a bare
  `None` into four reasons purely to find out which one the icons took. The answer was "none of
  them", which retired the whole lead in one boot instead of a session of narrowing.
- **A shape argument that crosses domains needs re-checking in the second domain.** "Content confined
  to the top-left" was read off the *screen* and then matched against `quad` on a *texture*. They are
  not the same coordinate space — the screen cell is the quad the shader draws, the texture is the
  atlas it samples from — and the match was coincidence.

So the icon class is back to: **draw rail, silent, victim moves between composites, no named
mechanism.** What survives from this section is the compute exoneration above it, the moving victim,
the persistence within a round, and the tooling (`content_summary`, `state_changed`,
`sampled_content`, `gva_sample_rung`). What does not survive is every sentence about the sampled-bind
rung transition.

### A Per-Entry Generation Cannot Name Content, Because The Entry Is Destroyed On The Hot Path

**Found and fixed. The mechanism is settled by a unit test that fails without the fix, not by a
boot.** This is the first named mechanism the icon class has had.

The engine's sampled cache has two hit paths and only one of them looks at content.
`find_cached_sampled`'s fallback hashes the incoming bytes and matches a 128-bit digest, so it
cannot serve the wrong image. The **identity** fast path above it matches `SampledContentIdentity`
(`key` + `generation`) and binds the retained `VkImage` **without hashing or comparing anything** —
that is the whole saving. So the identity is a *claim*, and until now nothing on either side ever
re-read it: the retained image has no CPU mirror, so a false claim is invisible from both directions.

`audit_sampled_identity` (`REIMS_VGPU_CONTENT_PROBE=1`) hashes the incoming bytes on every identity
hit and compares them against the digest the entry was admitted with. **It fired on the first boot:
6 mismatches in 299 008 checked claims.**

```
sampled_identity_audit reason=sampled_identity_stale identity_key=0xa4c000 generation=1
    retained=1680017cc77f6dc5bf1b2e2773463371 incoming=35c5df2df42ee4a49ad7a7cfd846b71c
    geom=64x64 checked=262755 mismatched=1
```

Two fields attribute it with no further work. `geom=64x64` is the icon geometry — the subject of the
repro, not a display-sized layer. `generation=1` was below the old `GUEST_LINEAR_GEN_BASE` (`1 << 32`),
so the producer is the **GVA host cache** and not either guest memo.

**The mechanism, read from the code.** `store_gva_owned` did
`host_gva_surfaces.entry(gva).or_default()` then `host_gen.wrapping_add(1)` — a **per-entry** counter,
so a freshly created entry is *always* generation 1. Entries are removed by `evict_gva`, and
`evict_gva` is not rare: it is called on **every deferred GVA render Store arm**
(`metal_draw/vulkan.rs`), which is the routine compositor path. So:

1. gva X is stored — entry created, `host_gen = 1`, content A. The engine retains image A under
   identity (X, 1).
2. A deferred GVA render Store arms at X and calls `evict_gva` — the entry is destroyed.
3. The next store re-creates it: `or_default()`, **`host_gen = 1` again**, content B.
4. The next bind resolves B, claims (X, 1), and the engine's identity fast path binds **A**.

`(gva, 1)` therefore names an unbounded number of distinct contents over a boot. That accounts for
every property this class has: silent (nothing compared anything), load-dependent (deferred Store
arms need compositor load), victim moves (which texture sits on a re-created entry varies), and
decided once per composite then held.

**The fix is uniqueness by construction, not a second check.** Every producer of a sampled-content
identity now draws its generation from one device-global monotonic counter,
`DeviceState::next_sampled_content_generation`. A value is issued once and never again, so identity
uniqueness no longer depends on any producer's entry lifetime, key space or eviction policy. The
guest-linear and type-5 memos were already sharing a counter and were sound; the GVA host cache was
the one producer that was not, and the mid-keyed and ref-keyed caches had the same latent shape.

That also **deletes `GUEST_LINEAR_GEN_BASE`**. The `1 << 32` namespace split existed only because two
producers kept independent counters that could collide; one counter removes the constant and the
failure mode it was guarding. `HostSurface.host_gen` and `LinearSampledMemo.host_gen` widen to `u64`
(three diagnostic readers print it; `HostLinearTexture.host_gen` is a different quantity — the
engine's u32 compute-resident generation — and is untouched).

Three things worth carrying:

- **The old tests asserted the namespace, which is a restatement of the constant, not the property.**
  Three of them checked `generation > u32::MAX`, i.e. "the two counters are in different ranges". That
  is true of the broken code as well: it says nothing about whether a generation ever repeats. The
  test that matters is `a_gva_reused_after_eviction_never_repeats_a_generation` — store, `evict_gva`,
  store — which fails on the parent commit with `left: 1`, the exact value the boot measured. **When a
  test asserts a namespace, ask what it would say if the counter inside one namespace restarted.**
- **A claim that crosses a module boundary needs an audit before it needs a fix.** This is the
  `-> bool` collapse rule ("a reason the caller writes is not a reading") applied to an *identity*
  rather than to a decline reason: the producer asserted "these bytes are unchanged" and the consumer
  spent a whole cache on believing it. The audit cost ~40 lines and one boot, and it converted a class
  that had survived a dozen commits of correlation into a two-line arithmetic bug.
- **The audit is a lower bound on incidence, not a census of the defect.** It sees only identity
  fast-path hits whose bytes actually differ, so a round can corrupt with zero lines — and two did on
  that boot. Do not read a future zero as "the class is closed".

**Repro tooling, in `.agents/repros/` (untracked, like the rest).** `icon-composite.sh` drives N
rounds in one boot, asserts the load arm by process count, holds the panel, slices the log per round,
and reports **SKIPPED** for a round byte-identical to its predecessor — the second `killall Finder`
does not always recomposite, and a round that did not happen must not count as evidence in either
direction. `iconscore.py` scores folder icons as connected blue blobs, so it needs no golden
reference and no fixed crop: window position, capture scale and appearance all move between runs and
blob geometry does not. Its `--across` presence table is the instrument this class was missing —

| cell | rounds 1-6 | |
|---|---|---|
| `y 9 x 28` | `.#####` | Desktop, absent in round 1 |
| `y 9 x 32` | `####.#` | Documents, absent in round 5 |
| `y 9 x 37` | `###...` | Downloads, absent in rounds 4-6 |
| `y 9 x 42` / `46` / `51`, `y 14 x 28` | `######` | always present |

4 of 6 rounds corrupt and 2 clean, so **the clean control arm this file demands is produced by the
same run at no extra cost** — which is exactly what the previous mechanism in this class died for the
want of. Confirmed on pixels: round 5 renders Documents as a narrow dark vertical bar and Downloads
as a shrunken top-left fragment; round 3 renders all seven correctly.

`REIMS_VGPU_SAMPLED_CACHE_OFF=1` forces every sampled bind to miss and re-upload. It is the bisection
that separates "the wrong bytes were resolved" from "the right bytes were resolved and a retained
image was bound in their place". Unused so far — the audit answered first — and it costs one upload
per bind, so a boot that sets it must not be read for frame rate.

**Scored on a boot at the fix, and the honest reading is: the stale bind is gone and THE ICON CLASS
IS NOT FIXED.** Same repro, same workload, `PANEL: On 27/27`:

| | at `1d8b718` (probe only) | at `81e2163` (fix) |
|---|---|---|
| `sampled_identity_stale` | **6** | **0** |
| identity claims checked | 299 008 | 294 912 |
| rounds corrupt / driven | 4 / 6 | **5 / 6** |
| cells that move between rounds | 3 of 7 | **7 of 8** |
| failure-channel lines | 13 | 12 |

The first two rows are a within-boot ratio at a comparable denominator, so 6 → 0 is the fix working
and is not subject to this rig's cross-boot spread. **The bottom two rows are the point of this
entry.** The corruption did not improve, and if anything more cells moved. A user watching the same
screen confirmed it independently.

So a proven invariant violation, on the exact geometry, in the exact repro, with a mechanism that
predicts every property of the class, was **not the cause of the class** — or not the only one. The
transferable lesson is the one this file keeps paying for from a new direction each time: *a defect
that fits the signature and is real is still not thereby the defect you are chasing*. The audit
measured a claim, and the claim was false; nothing in it ever measured that this claim was what put a
black rectangle on the screen. The scoring boot is what says that, and it had to be run.

What the fix is still worth, stated so it is not reverted by someone reading the table above: it
removes a silent wrong-image bind on the routine compositor path, it is proven by a test that fails
without it, and it deletes a namespace constant. Keep it. Do not credit it with the icons.

**The audit stays as the standing instrument.** It is the only thing in this crate that can see a
wrong *image* rather than a wrong *rate*, it costs nothing unless the probe is on, and the next
producer to invent a second generation source will be reported by name rather than by a screenshot.

### The Desktop Wallpaper Renders Solid Black On Whole Boots, Silently, And It Is Not A Regression

Found 2026-07-30. This is the cheapest handle the black-framebuffer class has had: **whole-screen,
stable for an entire boot, and it needs no gesture, no browser and no drag**. Every other layer --
Dock, menu bar, window chrome, Finder icons -- renders perfectly in the same frame, so it is a
whole-compositing-layer loss. `.agents/repros/black-desktop.sh` scores it across N boots.

Three things are settled, and the last one is the one that will otherwise be re-derived every session.

**It is in the draw rail, not the present path.** The guest's own native `screencapture` is black in
the same instant our host window is (desktop-region mean 0.001 against 0.37 on a good boot). That
capture re-executes the guest's composite through us, so the wallpaper layer is black in the image
the *guest* computes. `killall Dock` does not repair it, and neither does `killall WindowServer` --
so it is not a one-shot loss at boot that a repaint fixes, and the whole "we presented a stale or
partial composite" family is out.

**It is completely silent.** On a black boot every sentinel this class has reads 0:
`type11_seed_cache_absent`, `type11_seed_cache_geom`, `type11_seed_cache_ceded`,
`deferred_flush_lost`, `draw_vk_nothing_stored`, `resident_chain_abandoned_cpu_recovery`,
`chain_resident_land_fail`, `present_unbacked`, `linear_load_span_exceeds_alloc`,
`draw_prepare_chain_resident_not_ready`, `draw_prepare_chain_resident_identity_missing`,
`type11_window_invent`, `BadGrid`. The failure channel carries only the documented-benign
`cmd_task_ambiguous` / `gva_zero_pfn` / `task_walk_ambiguous`. A whole layer of guest work is lost
and nothing reports it, which is a ground-rules violation independent of the cause. A slug-set
difference over 8 black and 3 good boots returns **nothing present in every black boot and absent
from every good one**, in either direction -- so the next step for this class is a probe, and the
census is not where to put it.

**It is NOT a regression, and a bisect said it was.** This is the expensive part and it is the
transferable one. The defect presented as perfectly deterministic -- 7 boots at `f818b11`, all black,
`desk_mean` bit-identical to six significant figures at `0.000496384` -- and last night's `1c34f7c`
rendered correctly, so `git bisect run` over the day's 100 commits was the obvious move. It ran to a
confident answer. **The answer was garbage**: it bracketed the first bad commit between `98c8725`
(scored good) and `45336a2` (scored bad), and

```sh
git diff 98c8725 45336a2 -- crates vendor vm scripts   # empty
```

Those two trees are **byte-identical outside AGENTS.md**, as are the two commits between them. Then a
fresh boot at `f818b11` -- the same commit that had just gone black seven times -- rendered the
wallpaper. The same binary produces both outcomes.

So: **before believing a bisect, diff the code between the reported good and bad.** It costs one
command. A flaky predicate does not make `git bisect` fail; it makes it succeed, and hand back a
commit with a plausible-looking message. Three of the four candidates here were `:mag:` documentation
commits, which is the other cheap tell -- if the culprit cannot have changed behaviour, the run is
invalid, not surprising.

**An empty `desktoppicture.db` does not mean "no wallpaper", and reading it that way cost a wrong
"this is rig state" conclusion.** The guest's db has `pictures: 6, data: 0, preferences: 0`, which
reads exactly like a wallpaper that was configured and then cleared -- and AGENTS.md even documents
the trigger that GCs the orphaned `data` row. It is a red herring: **macOS falls back to the stock
`Ventura Graphic.heic` when no preference row exists**, and boots with that same empty db render the
wallpaper correctly. Nothing was stripped from the 2026-07-22 snapshot. Score the pixels, not the
configuration.

**Scoring, and the one-character trap in it.** Read the *desktop region only* -- the whole-screen mean
cannot separate the populations well because the Dock and menu bar are bright in both. And pass
`-alpha off`: these captures carry an opaque alpha channel and `%[fx:mean]` averages it in, which maps
a pure-black desktop to 0.5 and a wallpaper to 0.715 -- two populations 0.2 apart, both clearing any
sane floor, i.e. a scorer that silently cannot see the defect. With alpha dropped they are 0.0005 and
0.43, three orders apart.

**Open, and the next measurement.** The black boots and the good boots so far fall into two clean
blocks that differ by *two* things at once -- whether the host panel was held awake, and an hour of
wall clock:

| arm | boots | black |
|---|---|---|
| panel unheld, 20:00-20:35 | 6 | **6** |
| panel unheld (`--interactive`, same window) | 1 | **1** |
| panel held awake, 21:50-22:30 | 8 | **0** |

8/8 against 6/6 is a large split and it is **not attributed**: this is the "count what your A/B
actually changed" trap in its standard form, and the panel arm is the more suspicious of the two only
because it is the variable that was deliberately introduced. Note the host panel is already documented
here as a 4x confound on `present_hz`; whether it can reach the *guest's own composite* is exactly what
is unmeasured. Separating them needs the two arms **interleaved**, scored on the guest capture (which
cannot be affected by whether the host panel is lit), with the `dpms` value sampled throughout and
printed next to the verdict.

### 60+ fps Confirmed On A Second Panel-Awake Boot, And The 60 Hz Ceiling Was Not Real

Same boot as above, `.agents/repros/testufo-fps.sh /tmp/ufo-probe 45`, `PANEL: On 15/15 samples`,
149 cadence windows:

| | |
|---|---|
| `present_hz` | p50 **107.20**, min 5.40 (boot), **max 118.90** |
| `offered_hz` | p50 109.20, max **120.00** |
| `presents` / `offered` | p50 108 / 110 |
| `busy_acquire` | p50 38, max 309 |
| `drain_duty` | **0.118** at 718 tranches and 648 draws/s |
| `store_routes` | `surface_resident=428 type11_seed_elided=428` — 100 % resident rail |
| `display_vbl` | `window_hz=125.0 grid_hz=125.0` |
| guest testufo | **`Frame Rate 115 fps`** |

**This retires the "the guest composites 60 and four levers are spent" reading in the section below.**
The previous panel-awake boot measured `offered` p50 61 with Mission Control agreeing at 61, and that
was written up as a guest-side pacing decision. It is not a ceiling: the same code, same workload and
same guest reach 110 offered and 120 max. Whatever produced the 60 Hz arm is a per-boot state, not a
cap — so treat *both* numbers as boot samples and quote the within-boot ratios instead.

Read the testufo capture carefully, because its headline number is a trap: it shows **`20.000 Hz`**
in the big readout with a red **SYNC FAILURE: Browser unable to VSYNC** banner and
`Refresh Rate - Hz`. That big number is testufo's *failed refresh estimator*, not a frame rate. The
frame rate field reads **115 fps**, and our own `present_hz` reads 107. Do not quote the 20.

`busy_acquire` p50 38 is new and is the device now out-producing FIFO on this host (110 offered
against the ~80-120 the awake panel takes), losing ~2 frames/s. That is the first reading that would
make `PresentModeKHR::MAILBOX` worth anything — and it is worth ~2 %, not the 5x the asleep boot
implied.

Byte order checked on pixels: banner **red**, hyperlinks **blue**, `Problems?` **red**.

**Goals 1-3 re-verified green on this slice** (2131 lines): `linear_load_span_exceeds_alloc`,
`type11_seed_cache_absent`, `draw_vk_nothing_stored`, `resident_chain_abandoned_cpu_recovery`,
`frame_bgra_short`, `chain_resident_land_fail`, `surface_resident_*`, `type11_window_invent`,
`BadGrid`, `read_overrun`, `write_gate_no_spans`, uncovered `host_window_slate`, and
`deferred_flush_lost … reason=cache_miss` — **all 0**. Failure channel 91 lines, led by the
documented-benign `cmd_task_ambiguous` (13), `gva_zero_pfn` (11), `task_walk_ambiguous` (5).

### An Unbacked Present Is Only Black When Nothing Carries It

**The goal-2 detector was measuring a witness the resident rail stopped maintaining, and saying so in
a sentence it could not support. Fixed by `459a0c4`.** The line's slug and shape changed: greps for
`present_unbacked … reason=never_stored` no longer match, and the reasons are now
`present_backing_never_stored` / `present_backing_restaled` with a new `carried=` field.

`present_unbacked` is the always-on gate for the black-framebuffer class. Its witness is
`dense_frame_seq`, and the only site that advances it is `publish_surface_store`, whose own doc says
it runs for "a type-11 Store whose pixels have landed in the mapping's **guest pages**". The resident
rail renders into the registry and skips that write. So "no full frame was published for this mid"
stopped implying "nothing can show one" — while the line went on saying *"the surface is
uninitialized, so this shows black"*.

The 524 s boot above measured four `never_stored` lines making that claim. Against them, in the same
slice:

| | |
|---|---|
| `host_window_slate*` lines in the whole run | **1** — `slate_end … frames=567 covered=1` at t=22 s, the firmware-boot run |
| uncovered `host_window_slate` (blank window) | **0** |
| cadence windows bracketing all four events | `presents == offered`, `direct_frac=1.00`, no dip |

A resident carried every one of them. `note_present_backing`'s own doc already said it "never [reads]
the resident"; the *message* asserted a visual consequence anyway. That is "a reason the caller writes
is not a reading", applied to an outcome rather than to a cause — and it is the more expensive
direction, because the benign case and the real one were emitted identically. A genuine black-screen
boot is exactly what this gate exists to rank, and this one could not.

The fix asks the presenter's own question through the rule it already shares
(`pools::slot_presentable`, via `engine::resident_presentable`) and splits on the answer the same way
`host_window_slate` / `host_window_slate_end` already split: **a present nothing can carry is a black
frame and goes to the failure channel; one a resident carries cost no guest work and is a census.**
`PresentBacking` now carries its own reason (`impl Decline`), so the caller stops supplying the word.

Three details worth keeping:

- **Fail-closed on "cannot answer".** `carried` is `Option<bool>`, and the `None` arm — a build with
  no target registry — stays on the failure channel. `carried != Some(true)` and
  `carried == Some(false)` differ *only* there, which is precisely where a demoted black frame would
  go unnoticed, so that one character has its own test
  (`an_unbacked_present_fails_unless_a_resident_positively_carries_it`, verified to fail when it is
  flipped).
- **`carried=unknown` is a third word on purpose.** "Nothing carried it" and "we did not look" are a
  defect and an unmeasured build; one field that collapsed them would be the same fusion this
  vocabulary exists to prevent.
- **Priced where it runs.** One registry lookup under the engine lock, *inside* the refusal arm — four
  times in that boot, not 60 times a second. A carrier read on the healthy path would have been an
  engine-lock acquisition per present on the drain worker.

`resident_presentable` lost its `host-window` gate to do this: the question is about the target
registry, not about a window.

**Verified live on the `37365a0` boot.** Both occurrences read

```
OFF present_unbacked reason=present_backing_never_stored mid=5 geom=1920x1080 gen=0 carried=resident
```

— the `OFF ` prefix, i.e. the census channel, with `carried=resident`. Under the old code these were
failure-channel lines asserting "the surface is uninitialized, so this shows black" while a resident
carried them. `carried=nothing` has not been observed, so the failure arm is still untested by a live
boot; that is the arm that matters and it needs a boot that actually loses a frame.

### The 20 Hz Present Cap Is The Host Panel Being Asleep, Not The Swapchain

**This is a property of the x86 rig, not of the product, and it retires the entry above it. Do not
work on present modes on the strength of a `present_hz` reading taken with the screen off.** One
`kscreen-doctor --dpms on` moves the number 4x.

The section above ("The Composite Round Trip Is Gone, And The Presenter Is Now The Cap") measured
`presents` p50 **20**, `busy_acquire` p50 **409**, `busy_fence` **0** — 95 % of `present()` calls
failing the non-blocking `acquire_next_image(swapchain, 0, …)` while the device idled at
`duty=0.135` — and named the swapchain acquire as the next target, with `PresentModeKHR::FIFO` and
`min_image_count + 1` as the suspects. It also recorded, and could not explain, `present_hz` reading
p50 **20.00** at `3af5832` against **78.20** at `08110da` on code differing only by instrumentation.

Both are one variable, and it is outside the repo. **`/sys/class/drm/card0-eDP-2/dpms` was `Off`.**
A compositor with a blanked output stops pacing its clients at the refresh rate and releases
swapchain images at a slow fixed cadence instead. Measured with `vkcube` — which shares nothing with
this project but the compositor — 200 frames per run, `VK_PRESENT_MODE_FIFO_KHR`, KWin Wayland:

| panel | GPU | 320x240 | 1920x1080 | 3840x2400 |
|---|---|---|---|---|
| asleep | RTX 5080 (dGPU) | 19.42 | 19.50 | 19.36 |
| asleep | Intel (iGPU) | — | 19.42 | — |
| **awake** | RTX 5080 (dGPU) | **98.82** | **80.85** | — |
| **awake** | Intel (iGPU) | — | **109.63** | — |

Four readings that each kill something:

- **Flat across a 300x range of pixel count** (320x240 to 3840x2400, 19.36-19.50) — so it is a
  frame-callback throttle and not a composite cost, a bandwidth limit or a swapchain-image count.
- **Identical on both GPUs while asleep** — so it is not the cross-GPU present. (The panel is on
  `card0`/i915 and the engine renders on the NVIDIA `card1`, whose connectors are all disconnected,
  so every present *is* a PRIME crossing. Awake, that crossing costs about 25 %: 80.85 against the
  iGPU's 109.63 at the same size. Real, separate, and still a rig property — we correctly pick the
  fastest render device.)
- **Awake is 4x higher, from one command** — `kscreen-doctor --dpms on`, nothing else changed, same
  binary, same session, one minute apart.
- **78.20 is the awake dGPU figure and 20.00 is the asleep one.** The >=3.9x boot-to-boot
  `present_hz` spread this file recorded as unexplained noise on identical code was the screen
  blanking during one boot and not the other. That is not noise; it is a controlled variable, and
  controlling it is one line.

VRR is not involved (`Vrr: Never`), the mode is 3840x2400@120 throughout, and the session was
neither locked nor idle-hinted (`IdleHint=no`, `LockedHint=no`) — only the *output* was off. So none
of the obvious session-state checks would have caught it; the sysfs `dpms` attribute is the one that
answers.

**What this costs if unguarded is a whole session.** `busy_acquire=409` is a true reading of a real
refusal, `busy_fence=0` correctly exonerates the engine queue, and the conclusion drawn from them —
"the presenter is the cap" — is *also* true. It is just not the presenter's fault, and the three
fixes it invites (MAILBOX, more images, a blocking acquire) would each have been scored against a
throttle that no present mode can move. This is "instrument the branch, not the arm" one level out
from the process: the branch was outside our address space.

So `.agents/repros/lib.sh` now carries `host_panel_hold` / `host_panel_report`, and
`testufo-fps.sh` calls them. It wakes the panel, re-asserts every 20 s because the idle timer keeps
running, **samples `dpms` every 5 s for the whole run**, and prints `PANEL: On n/n` — or declares the
run's host Hz numbers `UNSCOREABLE` if any sample came back `Off`. The sampling is the part that
matters: a one-shot check at the start cannot see the panel blanking in the middle of a 45 s settle,
which is exactly how long the idle timer takes to matter.

**Do not read this as "the frame rate is fine".** What it establishes is that no host `present_hz`
number recorded in this file before 2026-07-30 has a known panel state behind it, so none of them
bound anything, in either direction. `offered_hz=99` from the `04dc2f4` boot is unaffected — it
counts distinct device publish seqs handed to `present()` and never touches the swapchain — and 99
frames/s of device production against a 60 fps goal is the reading to carry forward. Re-measure the
host side with `PANEL: On n/n` in the output before quoting any of it.

### The ~950 ms Idle Stall Is The Host GPU Suspending, Not This Device

**This is a property of the x86 rig, not of the product. Do not fix it, and do not measure across
it.** It has already driven one branch's worth of investigation and it is invisible unless you look
outside the repo.

The signature is unmistakable once named: a `sync_exec_lock_hold` of **916-979 ms** carrying two to
eight draws, on a device otherwise reading `duty=0.001`, recurring on a roughly 60-second period
while the guest sits at a quiet desktop. Seven in one boot, five in another, five in a third. A
cluster that tight is a fixed cost, not work.

`draw_phase` puts every one of them in the staging span, and `SlowStagingWrite` names the call: a
**single `acquire_staging` of one 256 KiB bucket**, `kind=acquire us=907112..948954 bytes=262144`.
Thirteen such events across one boot, median 937 ms. The same draw's *image* allocations in the
`acquire` phase, microseconds later, cost 3-4 ms. So it is not that allocation is expensive — 12 to
45 `vkAllocateMemory` calls happen per wake-up and exactly **one** of them stalls. That is a
first-touch signature.

What it is first-touching is the host GPU. This box is an **RTX 5080 Laptop** with NVIDIA
fine-grained runtime power management:

```
/proc/driver/nvidia/params:  DynamicPowerManagement: 2
/sys/bus/pci/devices/0000:01:00.0/power/control: auto     pstate: P8
```

Sampled once a second through a 221-second guest-idle window — reading only sysfs, which does not
touch the GPU and so cannot perturb what it measures — `runtime_suspended_time` advanced **150 203 ms
of 221 000**, through **seven** `suspended → resuming → active` cycles. Fine-grained RTD3 suspends
the device even though our Vulkan context holds it open, and the next access pays the resume. Four
`sync_exec_lock_hold` and three `staging_write_slow` landed in that same window against those seven
resumes — the counts do not match one-for-one because the host's own compositor resumes it too.

Three consequences, and the second is the expensive one:

- **No product change is warranted.** Keeping the staging pool warm would only move the stall to
  whichever call touches the device first. Keeping the GPU busy to defeat RTD3 is exactly the
  overfitted heuristic the rules here forbid, and it would be a battery regression on the only class
  of host that has this behaviour.
- **Every idle-boundary measurement on this rig is contaminated**, and the contamination is ~1 s of
  guest-visible stall that a desktop host will not have. `idle-then-damage.sh` and anything else
  that measures across a quiet stretch is measuring RTD3. The busy population is clean — a soak
  window at `duty > 0.8` never lets the GPU idle long enough to suspend — so the per-draw numbers
  below are unaffected.
- **It rewrites the black-screen lead.** The `gen=0` uninitialized present was preceded by exactly
  these stalls, with the guest awake throughout (the user confirmed the guest was awake; an earlier
  reading blamed display sleep and is retracted). WindowServer re-creating its scanout surfaces after
  a second of no progress is a plausible response to the *host* going to sleep under it. The gap in
  our code is still real and `present_unbacked` still reports it, but the *trigger* may not exist off
  this laptop. **Its slug and shape changed on 2026-07-30** — see "An Unbacked Present Is Only Black
  When Nothing Carries It" below; a grep for `reason=never_stored` no longer matches.

Note what did not work and why, because the same shortcut will look attractive again. The
`staging_pool` census reports per-bucket **mean** microseconds per miss; the 262144 bucket showed
365 us against 1 867 misses, and a handful of 940 ms outliers move that mean by half a millisecond.
Reasoning "the census shows no slow acquires, therefore it is one of the other three calls in the
span" was the available shortcut, it was taken, and it was wrong. A mean cannot see an outlier — the
same rule the speckle work already paid for, in a different disguise.

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

Two things make the guest arm attractive. It is **self-consistent by construction** — the reference
and the measurement are taken with the same instrument in the same round. And it is **independent of
our code**, which is the entire point: it is the only frame in the comparison that our present path
did not produce.

**But on this rig `screencapture` does not render application windows at all, and it says nothing
when it declines to.** Measured directly: TextEdit was opened on a file, its process count read 1,
and the guest's own capture showed the menu bar correctly switched to **TextEdit** — so the
instrument is live and tracking the frontmost app — with **no window anywhere in the frame**. Our
host window at the same instant showed that window, titled, with its text. The same test on Safari
and on a Finder window gives the same answer. Nine captures across one run were bare desktop, eight
of them byte-identical.

The cause is TCC: since macOS 10.15 a process without the **Screen Recording** grant gets a
screenshot containing the desktop picture and the menu bar and nothing else, with **exit code 0 and
no diagnostic**. This guest's system TCC database holds three rows and not one of them is a screen
grant, and SIP is enabled, so it cannot be inserted headlessly. An SSH-launched `screencapture` will
therefore never see a window here.

So the guest arm is valid for exactly what it draws — **wallpaper, desktop, menu bar** — and the
colour findings above, which are all wallpaper measurements, stand. It is **worthless for anything
window-shaped**, and any comparison of the form "the guest's screen is bare and ours holds a window"
is that hole reporting itself. **The residue localization this section used to cite was exactly that
shape and is retracted**: "eight windows opened, app killed, guest byte-identical to bare desktop,
ours holding a dead window, three times" is the guaranteed output of an instrument that omits every
window, whether or not any residue existed.

This is the "establish what each arm renders" rule below, one level worse than the Dock case that
motivated it: there the arm omitted a strip, here it omits the entire subject. Note how it survived —
the arm was checked for *liveness* (it tracked the frontmost app, its bytes changed when the scene
changed) and liveness was read as *coverage*. Before trusting any arm, drive the specific thing you
intend to measure into it and confirm it appears.

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
unopposed. The guest-screen comparison above has this hole twice over: `screencapture` renders
without the Dock, and — far worse, see the retraction there — without **any application window**. So
both a Dock change and an entire window score as a host-only difference against a byte-clean guest
arm. That is the same reading as "the guest moved on and we did not", produced for free, in regions
the design cannot see.

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

**A product call nested inside a census argument dies with the census.** The censuses this codebase
grew are mostly `note(x)` one-liners, and the idiom that grew with them is to compute `x` inline. When
`x` is itself the call that does the work, deleting the census statement deletes the work, and every
check that would catch it passes: it compiles, the tests pass, clippy is clean, and the function it
called is `pub` so nothing warns that it now has no callers.

That happened here. `idle_drain::note(maintain_idle_residents(display, now) as u64)` appeared at three
sites — two present-publish paths and the poll heartbeat. Removing the `idle_drain` census removed all
three, and `maintain_idle_residents` sat uncalled for four commits. It is not a small function to
lose: past the registry reclaim it trims the recycle pools, ages out the sampled-content cache and the
compute-storage residents, releases empty slab blocks, and sweeps cold host imports. Its own comment
says a publish-clocked drain once froze with "~260 stale residents (~516 MiB) pinned for the guest
lifetime", which is what the device now does with no drain at all.

So when removing an observability call, read its **arguments** before deleting the statement, and
restate what each one does. If an argument calls anything, hoist the call to its own statement first
and delete only the `note`. The mechanical check that finds an already-orphaned one: list every
`pub fn` in the crate whose name occurs exactly once across `src/` and `tests/` — a definition with no
reference. `pub` items in a staticlib are invisible to the dead-code lint, which is why this class
survives a clean `-D warnings` build.

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

### The Cheapest Real Reduction Left Is A Helper The File Already Has

Product code is close to its floor — the four sweeps above return almost nothing. The test modules
are not, and the highest-yield shape there is **not** "extract a helper". It is **"adopt the helper
that already exists"**, which is strictly safer because it introduces no abstraction and the target
signature is already fixed by 40-odd existing callers.

`icb/tests.rs` had `put_function_object(host, state, ref_, desc_gva, blob_page, blob)` with 42
callers, and **fourteen** other sites open-coding its exact body, 14 lines each. Adopting it removed
**182 lines** and added none. The tell that this was adoption rather than extraction only appeared
after a `git checkout` and a re-grep: an early attempt wrote a *new* helper with a different argument
order, and the file already had one. **Before extracting anything from a test module, grep the file
for a function whose body is the block you are about to hoist.**

The generic method, both halves needed:

1. `clones.py <MIN>` for candidates, and read the line ranges — the huge groups
   (`abi.rs` 337x, `vk_call.rs` 87x) are sliding-window artefacts on constant tables, visible because
   their ranges overlap (`6-23`, `7-24`, `8-26`). Real groups have disjoint ranges.
2. Before substituting, check **every** matched block for uses of its locals *after* the block, in
   the enclosing fn. That is what makes a substitution total rather than partial, and it is one
   regex. All 14 `put_function_object` sites were clean; the page-table fixture below is not.

Verify with an **exact round trip**, and build it as a canonical form rather than an inline-back:
rewrite both `HEAD` and the working tree so every occurrence — hand-rolled *or* helper call —
collapses to the same token, then compare. A naive inline-back fails when a newly-generated call is
textually identical to a pre-existing one, and the first attempt here did exactly that and reported a
false difference. Print the token count on both sides; equal counts plus identical text is the proof.

This matters most where there is no compiler. `backend-metal` never builds on a Linux host, so a
refactor of Metal-gated test code gets **no** check at all — the round trip is the whole verification.
One mistake in this class *is* catchable and was caught: an ungated helper over Metal-gated constants
(`TYPE7_FIRST_TLVS` and friends live behind `#[cfg(all(feature = "backend-metal", target_os =
"macos"))]`) fails the Vulkan-arm clippy with `cannot find value`. Gate the helper like its callers.

**The yield curve is steep, and this vein is now exhausted.** Measured on `icb/tests.rs`, in the
order taken: **−182** (adopt `put_function_object`, 14 sites), **−44**
(`make_render_pipeline_desc`, 6 sites), **−33** (`make_compute_pipeline_desc`, 8 sites). Then it
stops, and the reason is worth knowing because it is a property of test code rather than of this
file.

A whole-file scan for any 6-to-16-line block recurring 4+ times, with integer literals normalised so
near-identical blocks group, returns nothing else worth taking. Every remaining group is a **struct
literal carrying that test's own values** — `IcbRenderFill { command_index, pipeline_ref, … }`,
`IcbRenderDraw::MeshThreadgroups { threadgroups_x, …, mesh_tg_z }`, buffer-bind descriptors — at 13
to 22 sites each. Those are not duplication, they are the specifications: the literal *is* what the
case asserts about, a builder would need one parameter per field so the call would be no shorter, and
hiding the values behind it would make the suite harder to read for no line saving. Leave them.

The two remaining non-literal groups were checked and rejected on size:

- `metal_draw/tests.rs` page-table directory fixture — the clone report says 5 sites, exact-match is
  **3**, and each uses `dir_pfn` and `root_gpa` afterwards, so the helper must return a tuple. ~17
  lines.
- `icb/tests.rs` around 4043/4250/4358 — already helper-based after the three changes above; what
  repeats now is a prologue of *distinct* helper calls with per-test refs and GVAs.

So the rule that falls out: **a repeated block is reducible when it is repeated *code*, and not when
it is repeated *shape around different data*.** The clone detector cannot tell those apart — it
normalises literals precisely so it can group them — so the last step is always to read one instance
and ask which of the two it is.

A test helper is only worth extracting when it removes more than it adds *after* rustfmt, and the
call has to fit in 100 columns or every site wraps to eight lines. The `put_function_object` calls
land at 73 columns; the same extraction pre-wrapped by hand scored −64 instead of −182.

### Four Deletion Candidates That Do Not Survive Reading The Code

Audited 2026-07-30 by hand after a subagent sweep nominated each. All four **rejected**, with the
mechanism recorded so the next sweep — which will nominate them again, because they look identical
from a distance — costs a read rather than a session. Per "exclusions decay", each names what would
overturn it.

**`census/srgb_census.rs` — keep. The nomination inverted the rule.** It was proposed for deletion on
the grounds that it "carries a typed decline slug ✓", read as evidence the refusal is reported
elsewhere. It is not: `note_downgrade` **is** the emitter — it calls `observe::fail` with
`SRGB_DOWNGRADED_SLUG` itself, and its module doc says the class it exists for is that the fold "used
to be **silent** … at twelve independent sites, with nothing in the fail log". Deleting it
manufactures exactly the silent failure the ground rules forbid. The rule reads the other way round:
a census carrying **no** slug is the tally suspect; one that *is* the slug's only writer is the
report. Cheap, too — deduped per (site, format), bounded by 6 sites times the sRGB formats. It has
fired **0 times in the whole accumulated log**, which its own doc names as the healthy reading ("no
lines means the guest never asked for sRGB"), not as evidence of uselessness.

**`census/t11_decline.rs` — keep, and the reason is not the census.** This one *is* a true tally: no
slug, `observe::off` only, cumulative atomics, throttled one line per 256 declines — and it has
produced **0 lines in the entire recorded history**, so on this rig the "hundreds of MB/session"
lever it was built to size does not exist. That is a real argument for deleting it, and it loses to a
better one: `t11_decline::Reason` is the `Err` type of `try_type11_sample_zero_copy`, used at 19
sites, and the census is its **only** consumer. Delete the census and the enum is dead; delete both
and nine typed early-outs collapse into an `Option` — which is the "one status for N checks" regress
that the typed-decline work already ended, re-introduced on a rail whose failure mode is a silent CPU
copy. Overturn it by giving `Reason` a second reader (a typed decline at the call site), then the
tally goes.

**`blit_exec.rs` `read_texture_row` / `write_texture_row` (lines ~854–1008) — reject, same mechanism
as the b2t/t2b pair above.** 148 lines that are genuinely near-identical; the merge still loses. The
two carry **16 distinct slug literals** between them (`rd_row_*` / `wr_row_*`), and slugs cannot be
built by concatenation — `observe/gate.rs` finds them by lexing the source for literals, so a
runtime-assembled slug is invisible to the gate. A merged form therefore needs eight `&'static str`
parameters passed at each of two call sites, which costs more than it saves. Independently, the
buffer is `&mut [u8]` one way and `&[u8]` the other, so it cannot be one parameter without an enum or
closure. This is the third time this shape has been nominated and rejected; the discriminator is
always "count the slug literals first".

**`mapping_write.rs` `read_raw_rows` / `write_raw_rows` (lines ~486–622) — reject on auditability,
not on line count.** These return `bool`, so the slug argument does not apply and a merge would
genuinely save ~55 lines. It is still the wrong trade: both bodies end in an
`unsafe { copy_nonoverlapping }` over a raw `contig_for_span` pointer, and merging makes the copy
*direction* a runtime flag inside that unsafe block. This crate has an open, unexplained
guest-memory-corruption signature — seven kernel panics with a uniformly random victim process — and
these two writers are named in that section as hand-audited **Sound**. Trading the readability of a
raw guest-memory copy for 55 lines, while a corruption class is open and unattributed, is a bad deal.
Overturn it if the corruption class is ever closed.

The transferable point: three of the four were rejected by reading **one specific thing** — who calls
`observe::fail`, who else reads the enum, how many slug literals there are, whether the block is
`unsafe`. A size-and-similarity sweep cannot see any of them, so treat every such nomination as a
pointer to a file, never as a finding.

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

**A healthy x86/Vulkan boot, as a baseline to diff against.** Three boots on 2026-07-30, same
workload each time (guest to a real session with `ps -Ao comm | grep -c "MacOS/Dock$"` asserted 1,
then Safari on apple.com asserted running by process count, Calendar, System Settings; host window
verified by screenshot showing wallpaper, Dock, menu bar and all three apps):

| | boot 1 | boot 2 | boot 3 |
|---|---|---|---|
| sliced lines | 2586 | 2081 | 1987 |
| `FAIL` lines | **0** | **0** | **0** |
| `reason=air_loading` | 165 | 168 | 173 |
| `reason=generation_match` | 31 | 31 | 31 |
| `reason=cmd_task_ambiguous` | 11 | 11 | 11 |
| `reason=write_gate_outside` | 4 | 4 | 4 |

The `reason=` histogram is the comparison worth making, and it is remarkably stable — several slugs
land on the identical count across all three. **Do not diff the set of line *kinds* instead.** Those
slices differ in length by 30%, so roughly ten kinds appear in one and not another purely from
throttling and run duration (`direct_present_source` fires every 1024th present; `staging_write_slow`
is the RTD3 host artifact documented above). That set difference looks like a finding and is noise;
the `reason=` counts are what hold still.

Slice a boot with `stat -c%s /tmp/reims-vgpu-fail.log` before it and `tail -c +$((M+1))` after — the
log appends across boots, so an unsliced `grep -c` answers for every boot ever run on the machine.

**Two slugs in those three boots have since been retired, so a newer slice will not match them.**
`read_refused_neither` and `read_refused_shifted_would_serve` were the two arms of a counterfactual
that walked `task_id >> 1`'s page table purely to label the refusal; both are gone, and the same
events now report the reason the walk itself returned. A fourth healthy boot on 2026-07-30 (2535
lines, same workload, host window verified rendering wallpaper, Dock, menu bar and Safari on
Wikipedia) reads `air_loading` 182, `generation_match` 32, `cmd_task_ambiguous` 14,
`write_gate_outside` 4, and in place of the two retired slugs: `gva_zero_pfn` 13, every one of them
`gva_read_refused reason=gva_zero_pfn … via=runtime/objects.rs:607`. That is `lookup_list_entry`,
which is exactly where AGENTS.md had already attributed the substitutions — so the typed reason
confirms the attribution *and* names the site, which the two-arm caller label could not do. The
uniformity is the same result the `gva_write` work got when it typed its reason: one real cause behind
a label that had implied two.

That boot's one `FAIL` line is `present_capture … gen=0 reason=no_resident_content`, the documented
`gen=0` uninitialized-present class tied to the RTD3 host artifact above — pre-existing and on the
present path, not a page-table result. Do not read `FAIL 0` in the three-boot baseline as a promise;
it is three samples of a class that fires on an idle boundary.

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

**A green Vulkan suite does not mean the GPU ran it.** `tests/vk_engine_parity.rs` guards every
GPU-executing case with a `skip_if_no_gpu` arm that prints `SKIP …` to stdout and **returns `ok`**.
Cargo's summary cannot distinguish that from a pass. Measured in an agent environment with an
NVIDIA GPU physically present, `/dev/dri/renderD128` readable and a `nvidia_icd.json` installed:

| suite | reported | actually executed on the GPU |
|---|---|---|
| `vk_engine_parity` | `41 passed; 0 failed` | 21 — the other **20** skipped on `vk_init_create_instance vk_result=Unable_to_find_a_Vulkan_driver` |

So `cargo test` alone verifies decode, planning and the CPU-side rails, and verifies **nothing**
about barriers, layout transitions, pipeline state, descriptor writes or readback. Before landing a
change to any of those, run the suite with `--nocapture` and read the SKIP count:

```sh
cargo test -p reims-vgpu --no-default-features --features backend-vulkan,host-window \
  --test vk_engine_parity -- --test-threads=1 --nocapture 2>&1 | grep -c SKIP
```

A nonzero count means the arm you are changing was not executed, and the honest options are a live
boot, an exact **round trip** (as for `backend-metal` below), or not making the change. This is the
"validate the specific thing you drove" rule applied to the test suite itself: the green summary is
the healthy-looking log, and it is produced whether or not a driver exists.

**Read the count; do not assume it. On the x86 dev host it is now 0.** Measured 2026-07-30, serial,
three separate full-suite runs: `vk_engine_parity` **40 passed, `grep -c SKIP` = 0**, and
`vk_engine_compute` 14 of 14 with 0 SKIP in the same session. So on that host `cargo test` *does*
execute barriers, layout transitions, descriptor writes and readback, and a change to them is
test-covered rather than unverified.

**And the pass count in the line above was wrong when it was written — this is the failure mode the
paragraph below warns about, committed in the same breath as the warning.** At `93f8a26` the suite is
**44 passed / 0 failed / 1 ignored, 0 SKIP**. It reached 0 failed only at that commit: since
`e2c2dee` introduced it, `a_surface_resident_reads_back_in_guest_scanout_order` had failed on every
run on this host, asserting green `== 128` where `triangle_spirv` writes `0.5 * 255 = 127.5` — an
exact float→unorm8 tie that Vulkan does not pin, and that Intel ANV rounds down. The file already
carried `near()` with the doc comment "allow ±1 LSB" and the sibling case already used it.

Two things to take from it. A suite quoted as "40 passed" while it was really 39-and-a-known-red is
indistinguishable from a healthy suite *unless the failures are named*, so quote the failing test
names or quote nothing. And a test whose assertion sits exactly on a rounding tie will be red on some
drivers and green on others forever — when a constant comes from `f * 255`, check whether it lands on
a `.5` before asserting it exactly.

This does not retire the rule or the ceiling described below — it says the ceiling is a property of
the environment, and the environment the ceiling was measured in was an agent sandbox, not this
machine. The command above is one line and answers for the box you are actually on. Run it; quoting
either number from memory is how a "20 skipped" exclusion or a "it all runs" reassurance outlives the
host it was true for.

**The `vk_result` in that line is not the diagnosis — re-measured, the driver was there.** On a host
with a working NVIDIA RTX 5080 (`vulkaninfo --summary` names it) the same 20 cases still skip, while
`vk_engine_compute` runs **15 of 15 with zero SKIP** in the same environment and the same session.
`Unable_to_find_a_Vulkan_driver` is what the loader *says*, not what is true, and the exclusion
"this box has no GPU" that it invites is wrong.

The failure is **positional, not per-test**. In the serial run, parity cases 1-19 execute and cases
20-41 skip, every one of them, on a clean cliff. Any skipped case passes when run alone or in a
small `--test` filter. What separates the arms is *how many times the suite has already torn the
engine down*: `test_reset_engine` calls `ctx.destroy()`, which calls `vkDestroyInstance`, and the
parity suite calls it 43 times. Sampling `/proc/PID/maps` through a run shows the `nvidia` mappings
dropping to zero and reloading each cycle, and around the twentieth the reload stops working.

Confirmed by intervention, not correlation: making `test_reset_engine` retain its `DeviceContext`
instead of destroying it takes the SKIP count **from 20 to 0** in one run. That patch is not a fix
and was reverted — a retained context keeps a warm pipeline cache, which is exactly what
`warm_identical_draw_zero_creates_and_allocs` exists to measure, and that case fails under it.

Two consequences worth knowing before touching this:

- **The skips hid a real failure for the entire life of the tree.** Three cases —
  `sampled_and_sampler_still_renders`, `sampled_identity_fast_path_skips_content_compare` and
  `sampled_rgba_upload_to_bgra_target_preserves_semantic_channels` — failed a `sampled_cache_hits`
  assertion, and they were right: the sampled content cache could not hit until the submission ring
  wrapped, so an unchanging texture re-uploaded for `RING_DEPTH - 1` draws. Fixed by reaping the
  oldest signaled run in `begin_entry`.

  What let it hide for so long is worth more than the bug. **Every assertion in the suite that the
  cache *hits* was inside one of those three masked cases. The assertions that passed were all
  `sampled_cache_hits == 0`** — and a `== 0` assertion passes whether the cache works or is
  completely broken. So the suite had no passing evidence the cache had ever hit even once, and it
  read as green. When a subsystem's positive tests are all skipped, its negative tests do not hold
  the line; they go quiet in exactly the same shape as success.

  Re-measured after the fix: each of the 42 cases run one at a time, `--exact`, all 42 pass with
  `grep -c SKIP` of **0** each. There is no remaining masked failure. Note this does not test
  *ordering* — a full run still cannot execute every case, so state leaking between cases is
  untested by construction.

  The "4 failed" this section used to claim is retracted. The fourth was
  `warm_identical_draw_zero_creates_and_allocs`, and it failed only under the retained-`DeviceContext`
  patch used to un-skip the suite — that patch keeps a warm pipeline cache, which is precisely what
  that case measures. The instrument manufactured the fourth failure, which is the same trap as the
  `magick compare -metric AE` residue count above: a number produced by the measurement, not by the
  code.
- **Hoisting the loader handle does nothing.** `ash::Entry::load()` dlopens libvulkan per
  `DeviceContext` and dropping the `Entry` dlcloses it, which looks like the mechanism and is not:
  making the `Entry` a process-wide `OnceLock` leaves the count at exactly 20. The ICD is unloaded
  by `vkDestroyInstance`, not by releasing the loader. Predicted 0, measured 20, hypothesis dead.

**Where the next one of these is hiding: `CounterSnapshot` fields no test ever proves nonzero.**
The sampled-cache defect was findable in advance by asking which counters the suite only ever
asserts `== 0`, because that is the signature of a path with no positive coverage. Running that
question over all 71 fields — folding whitespace so multi-line asserts match, and gated on four
fields whose bucket is known by hand (`shader_hits`, `dispatches`, `device_lost`,
`sampled_cache_hits` must all come out "positive"; the first pass failed that gate and had to be
rewritten) — gives 35 positive, 9 asserted-but-never-nonzero, 25 never named.

The nine: `seed_uploads`, `buffer_zerocopy_binds`, `ring_retire_blocks`,
`compute_direct_writeback_bytes`, `compute_direct_writeback_fallbacks`,
`compute_sampled_resident_copy_bytes`, `compute_sampled_reinterpret_copy_bytes`,
`compute_deferred_writeback_bytes`, `compute_deferred_flush_bytes`.

The sharpest of them is the **guest-gather ("zero-copy") bind pair**,
`buffer_zerocopy_binds` (`engine/exec.rs:118`) and `sampled_zerocopy_binds` (`engine/exec.rs:1465`).
Both fire only for `SampledSource::GuestRuns` / buffer guest runs, and **every test in the suite
builds `SampledSource::Bytes`**, so neither has ever been executed by a test. That is the path that
copies straight out of imported guest pages instead of staging through the host — the one that
matters most for a real guest — and it is the same shape the sampled cache was in.

**This is a coverage statement, not a defect claim.** A zero here is equally consistent with the
path working fine and simply never being driven, and both sites do emit typed declines
(`BufferGuestRunImportMissing` and its sampled twin) when an import is missing. Do not write "the
zero-copy path is broken" anywhere on the strength of this table. What it licenses is the next
*measurement*: drive a `GuestRuns` request and read the counter.

So to exercise a parity case on the GPU, **run it in a filter small enough to stay under the cycle
limit** and confirm `grep -c SKIP` is 0 for that run. A full-suite count of 20 on a machine with a
working driver is this ceiling, and says nothing about whether your change works.

**A serial suite runs in alphabetical order, so renaming a test is a scheduling change.** This is
the sharpest instance of "instrument the branch, not the arm" that this repo has produced, because
the control experiment was run, was careful, and did not vary the variable.

`0760635` deleted the dmabuf present rail and replaced one parity case with a differently-named one.
`vk_engine_parity` went 40/40 → 39/40, failing
`framebuffer_fetch_reads_destination_via_input_attachment` with
`vk_device_lost_recreate_cap_exhausted cap=3`. It was written up as a product regression caused by
that commit, with the three deleted `VK_KHR_external_memory*` device-extension pushes as prime
suspect, and the branch was handed over red.

**All of it was wrong, and the same failure reproduces at the parent commit.** Two tests, run as a
filtered pair at `b26c7c8` — the commit *before* the deletion — fail identically:

```sh
cargo test … --test vk_engine_parity -- --test-threads=1 \
  device_loss_named_and_recreate_bounded framebuffer_fetch_reads_destination_via_input_attachment
```

The mechanism is entirely in the suite. `device_loss_named_and_recreate_bounded` exists to drive the
device-recreate budget to `MAX_DEVICE_RECREATES` and prove it stops there, so it leaves an engine
that refuses every subsequent draw — correct product behaviour, since the cap is a permanent
give-up. Every other case opened with `test_reset_engine()`; `framebuffer_fetch` was **the only
engine-touching case in the suite that did not**, and alphabetically `e…` used to sit between `d…`
and `f…`. The renamed replacement moved to `a_bgra_…`, the shield went away, and a latent bug that
had been there for the life of the file surfaced attached to an unrelated commit.

Three transferable points:

- **The control preserved the condition it meant to remove.** "Deleting the replacement test
  entirely still fails, so it is not a position shift" — but deleting it leaves `framebuffer_fetch`
  *still* immediately after `device_loss`, which is the whole mechanism. A control that holds the
  suspect variable fixed reads exactly like a refutation. State what the control changed, not what
  it removed.
- **"It passes alone" is evidence *for* order-dependence, not against it.** It was recorded as
  evidence the test was innocent. A test that passes alone and fails in the suite is the definition
  of contaminated state; the next command after observing it should be a two-test filter, which
  costs three seconds and settles it.
- **The log said so and was not read.** The per-process sink
  (`/tmp/reims-vgpu-fail-test-<pid>.log`, not the product log — `redirect_logs_for_tests`) showed
  four `vk_device_select` lines and then the cap-exhausted refusal, with **no device create for the
  failing test at all**. A device that is never created cannot have been lost by a missing
  extension. One `cat` of the right file discriminates "the engine lost the device" from "the engine
  refused before touching it".

Fixed structurally rather than by adding the missing line: `engine_test_lock()` is now
`engine_test_session()`, which takes the lock *and* resets, and hands back the guard. The omission
is no longer writable — there is no way to obtain the lock without having reset — and it deleted 53
lines of two-line preamble across `vk_engine_parity.rs` and `vk_engine_compute.rs`. The product-side
property it now rests on (a reset clears an exhausted budget) is pinned by an assertion at the tail
of `device_loss_named_and_recreate_bounded`, verified to fail without it.

Note `tests/vk_engine_batch.rs` still has the raw-lock idiom and **no `test_reset_engine()` in any
of its six cases**. It passes because nothing in that binary manufactures a poisoned engine — the
trap is unarmed there, not absent.

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
