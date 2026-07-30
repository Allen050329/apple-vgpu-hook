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

**`req.output_bgra` is built, tested, and unreachable from product code — turning it on is part of
this fix.** `grep -rn output_bgra crates/reims-vgpu` finds it read in five places in
`engine/exec.rs`, and assigned `true` in exactly six places, *all of them in
`tests/vk_engine_parity.rs`*. No `src/` file ever sets it, so `let output_bgra = req.output_bgra &&
req.target_identity.is_some()` is permanently false and the engine's "BGRA output, so a raw
image→buffer copy lands guest scanout order with **no CPU swizzle**" path never runs.

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

**Do not pool a boot failure with a clean boot.** The same 351 logs contain **15 boots (4.3%) that
never started macOS at all** — OpenCore `Boot failed - Aborted` — spread across many days and many
commits, so it is rig flakiness. A harness that scores "did not panic" over all boots counts those 15
as successes, which means a change that breaks booting reads as a change that fixes panics.
`.agents/repros/panic-rate.sh` keeps `PANIC` / `NOBOOT` / `NOWORK` / `OK` apart and reports
`PANIC / (PANIC + OK)`.

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
  our code is still real and `present_unbacked reason=never_stored` still reports it, but the
  *trigger* may not exist off this laptop.

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
