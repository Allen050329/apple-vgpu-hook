# web-content-probe

Checks whether the guest's web content reaches the screen intact, against a
declaration of what it was supposed to look like.

## Why it exists

Goal 8 is "web content is occasionally corrupted in Firefox and Safari —
background disappearing, subtle bugs, not all-white". A screenshot of a real
page cannot be checked against anything, because nothing says what it should
have looked like, and "occasional" means the interesting frame is one in
hundreds. Both problems are the same problem: there is no oracle.

So the page is built to be one. `content-probe.html` fills the viewport with a
fixed palette of widely separated, fully opaque colours in known regions, and
POSTs back each region's **screen-space** rectangle and the colour it is
supposed to be. The host then samples its own capture at exactly those
rectangles and classifies each measured mean to the nearest palette entry.

Same intent-versus-result shape as `scripts/modal-button-probe`, and for the
same reason: only the guest's own declaration can say whether a wrong pixel is
this device's fault.

## Usage

```sh
scripts/web-content-probe/web-content-probe.sh [-n CAPTURES] [--browser safari|chrome|firefox] [--keep DIR]
```

Needs a running guest reachable as `macos-vm` and a host QEMU window the KDE
screenshot script can capture. Exit `0` when every declared region measured its
declared colour in every capture, `1` on any mismatch, `2` on setup failure.
`--keep DIR` retains every frame and the layout records — do that whenever you
expect a failure, because the frame is the evidence.

## How the verdict is reached

`WHITE` and `BLACK` are palette members on purpose: a region that loses its fill
classifies *by name* rather than as an unexplained threshold miss, so
"the background turned white" and "the background turned black" are different
verdicts. Nearest-entry classification also means there is no tolerance to tune
— the colours are far apart by construction, so "which of these is it" is
answerable without naming a distance that counts as close.

Each rectangle is inset by a sixth before measuring, so a one-pixel rounding
error in the 720p downscale cannot pull a neighbouring colour into the mean.

## The mistake this probe made first, kept because it will recur

The first run reported all six patches and two backgrounds wrong, identically,
in all five captures — and the retained frame showed the page rendering
*perfectly*. The declaration was wrong, not the pixels.

`window.innerWidth/innerHeight` and `window.outerWidth/outerHeight` do not update
atomically. Read during Safari's fullscreen transition, `innerWidth` was already
1920 while `outerWidth` was still 1854, so the computed viewport origin came out
at `(-33, -184)` and shifted every rectangle a whole grid row off the content.

Two things follow, and both are in the code now. The page **refuses** a
declaration whose origin is negative or whose viewport does not fit the screen,
rather than publishing it; and it republishes every second, with the host
re-reading the newest record before every capture. That also makes the probe
robust to a window being moved or resized mid-run, which would otherwise report
as ten simultaneous corruptions.

The general lesson is worth more than the fix: **when the oracle and the result
disagree everywhere at once, suspect the oracle.** A real compositing loss is
local and intermittent. A uniform, perfectly reproducible disagreement across
every region is a coordinate bug.

## What it does not cover

It detects a region whose *colour* is wrong at its declared rectangle. It does
not detect content shifted by a small amount (the inset absorbs a few pixels), a
wrong gradient, wrong text, or a transient wrong frame that falls between two
captures. It uses plain opaque divs deliberately — the aim is to catch a
compositing loss, not to exercise one particular path into one.
