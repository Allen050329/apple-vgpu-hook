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
tally: the decline registry baseline in `observe/gate.rs` did not move across four separate commits
that removed nine such modules. And a line whose text says `ok`, `stage_ok`, `resident_ready=1` or
`retain` is narrating the success path, whatever sink it was written to.

**Cost hides behind "measure-only".** The proxies here were not free, and the comments admitted it
in situ: "2-3 ms per display-sized storage image on the stamp path", "the full-frame stats scan
exists only for the verbose line". Removed in this family: a GPU compute reduction dispatched every
present, a SIMD census fused into every readback, an O(w·h) scan of every bound texture on every
draw, and a full read-back of the guest window on every Store. Before writing `// Measure-only` on
something, price it at the rate it will actually run.

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

So the actionable shape of the ~2 Hz problem is: **give the type-11 render Store the deferred rail the
GVA render Store already has** — keep the composite on the registry resident with `skip_readback`,
arm a flush-on-access window, and let the matching Load take `LoadOp::LoadFromTarget` instead of a CPU
seed. That kills both halves, because the Load-side seed is chosen by asking whether a deferred window
exists (`metal_draw/mod.rs`, the `PASS_LOAD_ACTION_LOAD && mapping_id == 0` arm), so a type-11 window
would suppress the seed for free.

Do not do this by restoring `VK_EXT_external_memory_host`. The deleted `import_present` rail had an
"ack-fast deferred rung" that looks like the same idea, but what it needed the host pointer for was
the eventual *DMA*; the deferral itself did not. The flush here is the CPU writeback that already
exists (`mapping_write::write_rgba8_image_changed`) — the win is doing it once on demand instead of
~70 times a second unconditionally.

Worth checking before building: the model already carries mapping-keyed deferred windows for the
*compute* rail (`ComputeStorageResidencyKey`'s "Surface window (`mapping_id != 0`)" kind, landed by
`storage_flush`'s `flush_one` and its `compute_deferred_flush mapping=` line). The render Store may be
able to arm one of those rather than growing a fourth window kind.

Two process points are worth as much as the result. First, the decisive probe was a *duty cycle* —
a state — where the pre-existing `sync_exec_lock_hold` was an event count above a 250 ms threshold,
and the measured frame period sat at 252–665 ms, i.e. just under that threshold for entire runs. It
fired **once** in a four-round boot while the worker was pinned at 100%. Second, `counter_snapshot`
is the second instrument here found fully built and never called; before adding a probe, grep for a
snapshot/census function and check it has a caller.

**Do not edit Rust while a multi-boot harness is running.** `boot-x86.sh` rebuilds QEMU every boot,
so a tree edit lands in the middle of a run. One `stability-n.sh` run was scored `DISCARD` for exactly
this. The numbers taken before the edit are still good; the verdict is not.

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
