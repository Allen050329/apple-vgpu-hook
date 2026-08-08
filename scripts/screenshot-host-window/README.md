# screenshot-host-window

One command for "screenshot the guest window", on whatever the host runs.

## Why it exists

Verification is "boot, then screenshot", and the two helpers that existed named a
compositor each: `screenshot-when-kde-plasma-host` and
`screenshot-when-macos-host`. On GNOME, or a wlroots session, the step had no
command at all. This picks a backend, delegates to those helpers where they
apply, and reports which one it chose.

```sh
scripts/screenshot-host-window/screenshot-host-window.sh -o /tmp/screen.png
scripts/screenshot-host-window/screenshot-host-window.sh --print-backend   # names it, captures nothing
scripts/screenshot-host-window/screenshot-host-window.sh -o /tmp/s.png -- --pid 1234
```

Arguments after `--` reach the delegate, so the KDE helper's `--pid`, `--window`,
`--match` and `--decorations` still work. The PNG path goes to stdout either way.

## Backends

Ordered by what each one can *see*, not by how portable it is.

| Backend | Selected when | Targets |
|---|---|---|
| `macos` | host is Darwin | the window |
| `kde` | `spectacle` present and a Plasma session is running | the window |
| `x11` | `XDG_SESSION_TYPE=x11` and ImageMagick `import` | the window |
| `portal` | `gdbus` and `python3` | **the whole screen** |

A Plasma session is detected by a running `kwin_wayland` / `kwin_x11` /
`plasmashell`, not by `XDG_CURRENT_DESKTOP` alone. An SSH tty sets no such
variable and the KDE helper is built to run from one, so gating on the variable
would send every SSH-driven run to the portal and lose window targeting for no
reason.

Output is capped at 1280x720 whichever backend ran, matching what the KDE and
macOS helpers already did.

## The portal prompts, so it hangs unattended

`xdg-desktop-portal` is the only capture interface KDE, GNOME and the wlroots
compositors all implement, which is what makes it the fallback. Two things about
it are worth knowing before relying on it.

It has **no window handle**. The Screenshot API captures the screen, so a portal
capture contains whatever else was on the display, and the guest window has to be
cropped out afterwards. It is a fallback, not an equivalent, and it says so on
stderr when it runs.

It also **asks the user**. `interactive=false` suppresses the picker but not the
permission prompt, and with nobody at the keyboard the request never returns.
Measured on Plasma 6: no response, and `portal-screenshot.py` gives up at its
30-second deadline with exit 3 rather than hanging to whatever outer timeout the
caller had. An unattended run on a portal-only host therefore needs the
permission granted ahead of time, through the desktop's own privacy settings or
the portal permission store — the script cannot arrange it.

Exit codes from `portal-screenshot.py`: `0` wrote the file, `2` refused or
dismissed, `3` no response before the deadline, `4` portal unreachable.
