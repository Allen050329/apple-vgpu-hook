# screenshot-when-kde-plasma-host

Host-side PNG capture of a **live QEMU display window** (Plasma Wayland, via a
KWin script + `spectacle`).

When one QEMU process owns multiple windows (for example a serial console plus
the integrated GPU display), the helper captures the largest matching client
area so the GPU display wins.

This is **not** guest QMP display capture. It grabs the compositor window, so
you see the same pixels a human sees on the host desktop.

The host-owned window's caption is a compile-time constant set in
`crates/reims-vgpu` (`device_window_start` builds the `WindowConfig`), so the
normal case needs **no selector at all** — just run the script.

## Behavior

- **Selectors are strict.** `--pid` / `--window` / `--match` matching nothing is
  a hard error that dumps every candidate window (pid, id, class, caption, size). The fallback to
  the largest QEMU-class window is opt-in via `--any`.
- **`--match SUBSTR`** selects on caption/class, immune to the PID drift above.
  It is an escape hatch for non-default windows.
- **Activation is verified.** After setting `workspace.activeWindow`, the script
  re-reads it and aborts with `ACTIVATEFAIL` unless it is the selected window, so
  `spectacle -a` can no longer be pointed at something else.
- **Black frames fail the capture.** A uniformly black result (max channel ≤ 8)
  exits non-zero with an explicit "this is a FAILED CAPTURE, not evidence the
  guest rendered black" message. `--allow-black` accepts black captures.
- **`WAYLAND_DISPLAY` is asserted.** Without it `spectacle` SIGABRTs (exit 134,
  core dumped, no output file); that now fails with the reason.

**Standing rule regardless:** treat a black capture as "capture failed" until
proven otherwise, and corroborate against `/tmp/reims-vgpu-thrash.log` before concluding
anything about present correctness.

## Default selector

With no selector the script now selects the window whose caption contains
**`Apple vGPU Hook`** — the title `device_window_start` gives the winit window in
`crates/reims-vgpu`. It is a constant in our own tree, so making the caller
supply it is unnecessary.

Two concurrent VMs produce windows with the identical caption, so
**`--window <internalId>` is the only selector that separates concurrent VMs.**
`--match` remains for reaching a
non-default window, e.g. QEMU's own GTK window when comparing its
`DisplaySurface` against the host-owned one.

## Requirements

- Plasma Wayland session for the same user (KWin + session bus)
- `spectacle`, `qdbus6` (or `qdbus`), `journalctl`
- A running `qemu-system-*` with a real window on the compositor — either QEMU's
  own (`-display gtk` / similar) or the host-owned Rust window under
  `-display none` (same process, so PID matching should still resolve it)

## Usage

```sh
# Temp PNG under /tmp; path printed on stdout
scripts/screenshot-when-kde-plasma-host/screenshot-when-kde-plasma-host.sh

# Explicit path
scripts/screenshot-when-kde-plasma-host/screenshot-when-kde-plasma-host.sh -o /tmp/qemu.png

# PREFERRED when several VMs run: their captions are identical, so only the
# KWin internalId separates them. Ids come from the candidate list an error prints.
scripts/screenshot-when-kde-plasma-host/screenshot-when-kde-plasma-host.sh --window '{e1b5ae37-...}' -o /tmp/qemu.png

# Escape hatch for a non-default window, e.g. QEMU's own GTK window
scripts/screenshot-when-kde-plasma-host/screenshot-when-kde-plasma-host.sh --match qemu -o /tmp/qemu-surface.png

# A specific qemu PID — strict: errors out if no window matches it
scripts/screenshot-when-kde-plasma-host/screenshot-when-kde-plasma-host.sh --pid 2001800 -o /tmp/qemu.png

# Include window decorations
scripts/screenshot-when-kde-plasma-host/screenshot-when-kde-plasma-host.sh --decorations -o /tmp/qemu.png
```

Safe over **SSH** / non-graphical ttys: the script imports `DISPLAY`,
`WAYLAND_DISPLAY`, `XDG_RUNTIME_DIR`, and `DBUS_SESSION_BUS_ADDRESS` from the
user's `plasmashell` / `kwin_wayland` process.

## Exit behaviour

| Situation | Result |
|-----------|--------|
| No qemu-system process | Error, non-zero |
| Process up but no KWin window at all | Error, non-zero |
| No Plasma/KWin session for this user | Error, non-zero |
| `WAYLAND_DISPLAY` unresolvable | Error, non-zero (spectacle would SIGABRT) |
| Selector matched no window | Error, non-zero + candidate dump (`--any` to override) |
| Selected window would not activate | Error, non-zero (`ACTIVATEFAIL`) |
| Capture is uniformly black | Error, non-zero (`--allow-black` to override) |
| Capture OK | `0`, PNG path on stdout |

Diagnostics go to stderr; the selected window and the capture's
`max=/mean=/colors=` stats are always printed there, so every run leaves a record
of what was grabbed and whether it had content.

The black-frame guard needs `magick` or `convert`. Without either it is **skipped
with a warning** — a black capture then exits `0` again, so keep ImageMagick
installed on any host where this gates verification.

## Output size

Every capture is shrunk to fit a **1280x720** box, aspect ratio preserved
(ImageMagick `-resize '1280x720>'`; the trailing `>` only ever shrinks, so a
window already below 720p is untouched). Measured: 3840x2160 → 1280x720 is 9x
fewer pixels and ~12% of the PNG bytes on realistic content.

The reason is token cost: these captures are read by agents, and a 4K PNG is a
large multiple of a 720p one for no added signal at that review size.

**There is deliberately no flag.** A lever would mean every caller has to
remember to pass it, which is how the default ends up being the wasteful one. If
a full-resolution capture is genuinely needed, run `spectacle` directly.

It runs **after** the black-frame guard, so the `max=/mean=/colors=` stats
describe the pixels actually captured rather than resampled ones, and a failed
(black) capture is left at full resolution for inspection. Like the guard, it
needs ImageMagick and is skipped with a warning without it; the capture then
keeps its native size.
