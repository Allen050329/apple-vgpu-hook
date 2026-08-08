#!/usr/bin/env bash
# screenshot-host-window.sh — one screenshot entry point, whatever the host runs.
#
# Verification says "boot, then screenshot", and until now that meant naming a
# compositor-specific helper. There are two, KDE and macOS, so on GNOME or a
# wlroots session the step had no command at all. This picks a backend and says
# which one it picked.
#
# Usage:
#   scripts/screenshot-host-window/screenshot-host-window.sh [-o PATH] [--print-backend] [-- ARGS...]
#
# Prints the PNG path on stdout, like the helpers it delegates to. `ARGS` after
# `--` go to the chosen delegate, so `--pid` and `--decorations` still reach the
# KDE one. `--print-backend` names the backend and exits without capturing,
# which is how a host reports what it would do without spending a capture.
#
# Backends, best first. The order is by what the backend can *see*, not by
# preference: only the first three can single out the guest window, and the
# portal cannot, so it comes last however portable it is.
#
#   macos    scripts/screenshot-when-macos-host    window-targeted
#   kde      scripts/screenshot-when-kde-plasma-host  window-targeted
#   x11      ImageMagick `import -window`          window-targeted
#   portal   xdg-desktop-portal Screenshot         WHOLE SCREEN
#
# A portal capture therefore contains whatever else is on the display. It is a
# fallback, not an equivalent, and it is labelled as one on stderr when used.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

OUT=""
PRINT_BACKEND=0
DELEGATE_ARGS=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    -o) shift; OUT="${1:-}"; shift ;;
    -o=*|--out=*) OUT="${1#*=}"; shift ;;
    --print-backend) PRINT_BACKEND=1; shift ;;
    --) shift; DELEGATE_ARGS=("$@"); break ;;
    -h|--help) sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "screenshot-host-window: unknown arg: $1" >&2; exit 64 ;;
  esac
done
OUT="${OUT:-/tmp/apple-vgpu-hook-screen.png}"

have() { command -v "$1" >/dev/null 2>&1; }

# The guest window is a `winit` client, so an X11-only grabber can reach it only
# when the session itself is X11. Under Wayland the same tool would silently
# capture nothing useful, which is why the session type gates this and not the
# presence of the binary.
choose_backend() {
  if [ "$(uname -s)" = "Darwin" ]; then
    echo macos; return
  fi
  # A Plasma session is detected by a running compositor, not by
  # XDG_CURRENT_DESKTOP alone: an SSH tty sets no such variable, and the KDE
  # helper is built to work from one — it imports the session env out of
  # plasmashell itself. Gating on the variable would send every SSH-driven run
  # to the portal and lose window targeting for no reason.
  case "${XDG_CURRENT_DESKTOP:-}" in
    *KDE*|*plasma*|*Plasma*) if have spectacle; then echo kde; return; fi ;;
  esac
  if have spectacle && pgrep -x 'kwin_wayland|kwin_x11|plasmashel[l]' >/dev/null 2>&1; then
    echo kde; return
  fi
  if [ "${XDG_SESSION_TYPE:-}" = "x11" ] && have import; then
    echo x11; return
  fi
  if have gdbus && have python3; then
    echo portal; return
  fi
  echo none
}

BACKEND="$(choose_backend)"
if [ "$PRINT_BACKEND" = "1" ]; then
  echo "$BACKEND"
  exit 0
fi

# Every delegate below except the KDE and macOS ones hands back a full-resolution
# PNG. Those two already cap their output so an agent reading the file does not
# pay for pixels it cannot use; match that here rather than leaving the cap a
# property of which backend happened to run.
normalise() {
  local path="$1" tool=""
  have magick && tool="magick"
  [ -z "$tool" ] && have convert && tool="convert"
  [ -z "$tool" ] && return 0
  "$tool" "$path" -resize '1280x720>' "$path" 2>/dev/null || true
}

case "$BACKEND" in
  macos)
    exec "$SCRIPTS_DIR/screenshot-when-macos-host/screenshot-when-macos-host.sh" \
      "$OUT" ${DELEGATE_ARGS[@]+"${DELEGATE_ARGS[@]}"}
    ;;
  kde)
    exec "$SCRIPTS_DIR/screenshot-when-kde-plasma-host/screenshot-when-kde-plasma-host.sh" \
      -o "$OUT" ${DELEGATE_ARGS[@]+"${DELEGATE_ARGS[@]}"}
    ;;
  x11)
    # `-name` matches the caption the device sets; see crates/reims-vgpu
    # device_window_start. Falls back to the root window when no match, which is
    # reported rather than passed off as a window capture.
    if ! import -name "Apple vGPU Hook" "$OUT" 2>/dev/null; then
      echo "screenshot-host-window: no window titled 'Apple vGPU Hook'; capturing the root window" >&2
      import -window root "$OUT"
    fi
    normalise "$OUT"
    echo "$OUT"
    ;;
  portal)
    echo "screenshot-host-window: portal backend captures the WHOLE SCREEN, not the guest window" >&2
    "$SCRIPT_DIR/portal-screenshot.py" "$OUT" >/dev/null
    normalise "$OUT"
    echo "$OUT"
    ;;
  none)
    # Each line names the thing that was missing, so the reader can act on it
    # without re-deriving the rule from the source.
    echo "screenshot-host-window: no usable backend. Host is $(uname -s)," \
         "desktop '${XDG_CURRENT_DESKTOP:-unset}', session '${XDG_SESSION_TYPE:-unset}'." >&2
    echo "  macOS   needs a Darwin host" >&2
    echo "  KDE     needs spectacle$(have spectacle || echo ' — MISSING')" >&2
    echo "  X11     needs an x11 session and ImageMagick import$(have import || echo ' — import MISSING')" >&2
    echo "  portal  needs gdbus$(have gdbus || echo ' — MISSING') and python3$(have python3 || echo ' — MISSING')" >&2
    exit 1
    ;;
esac
