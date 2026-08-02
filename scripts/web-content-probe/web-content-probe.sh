#!/usr/bin/env bash
# web-content-probe.sh — does the guest's web content reach the screen intact?
#
# Goal 8 is "web content is occasionally corrupted in Firefox and Safari —
# background disappearing, subtle bugs, not all-white". A screenshot of a real
# page cannot be checked against anything, because nothing declares what it
# should have looked like. So this serves a page that does declare it
# (`content-probe.html`: a fixed palette of widely separated opaque colours,
# posting back each region's screen rectangle and expected colour), then samples
# the host capture at exactly those rectangles, repeatedly.
#
# Same intent-versus-result shape as `scripts/modal-button-probe`, and for the
# same reason: the guest's own declaration is the only thing that can say
# whether a wrong pixel is this device's fault.
#
# Usage:
#   scripts/web-content-probe/web-content-probe.sh [-n CAPTURES] [--browser safari|chrome|firefox] [--keep DIR]
#
# Exits 0 when every declared region measured its declared colour in every
# capture, 1 on any mismatch, 2 on a setup failure.
set -euo pipefail
# ImageMagick prints statistics with a '.', and awk must read them the same way.
export LC_ALL=C

CAPTURES=20
BROWSER=safari
KEEP=""
GUEST="${GUEST:-macos-vm}"
PORT="${PROBE_PORT:-8997}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SHOT="$REPO_ROOT/scripts/screenshot-when-kde-plasma-host/screenshot-when-kde-plasma-host.sh"
SERVER="$REPO_ROOT/scripts/browser-probe/probe_server.py"

while [ $# -gt 0 ]; do
  case "$1" in
    -n|--captures) CAPTURES="$2"; shift 2 ;;
    --browser) BROWSER="$2"; shift 2 ;;
    --keep) KEEP="$2"; shift 2 ;;
    -h|--help) sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
    *) echo "web-content-probe: unknown argument $1" >&2; exit 2 ;;
  esac
done

case "$BROWSER" in
  safari)  APP="Safari" ;;
  chrome)  APP="Google Chrome" ;;
  firefox) APP="Firefox" ;;
  *) echo "web-content-probe: unknown browser $BROWSER" >&2; exit 2 ;;
esac

WORK="${KEEP:-$(mktemp -d)}"
mkdir -p "$WORK"
[ -n "$KEEP" ] || trap 'rm -rf "$WORK"' EXIT
say() { echo "web-content-probe: $*"; }

ssh -o ConnectTimeout=8 -o BatchMode=yes "$GUEST" true 2>/dev/null || {
  say "no guest at $GUEST" >&2; exit 2; }

scp -q "$SERVER" "$SCRIPT_DIR/content-probe.html" "$GUEST:/tmp/"
ssh -o BatchMode=yes "$GUEST" "pkill -f probe_server.py >/dev/null 2>&1 || true
pkill -f '$APP' >/dev/null 2>&1 || true
sleep 2
nohup python3 /tmp/probe_server.py $PORT /tmp/content-probe.html /tmp/content-probe.json \
  >/tmp/content-probe-server.log 2>&1 &
sleep 2
open -a '$APP' 'http://127.0.0.1:$PORT/'
sleep 6
# Fullscreen so the viewport is the display and the page's screen-space
# rectangles need no chrome model. Done after load: entering fullscreen fires a
# resize, and the page re-declares its layout on resize.
osascript -e 'tell application \"System Events\" to key code 3 using {command down, control down}' >/dev/null 2>&1 || true
sleep 5" >/dev/null

LAYOUT="$WORK/layout.json"

# The page republishes its layout every second, and the newest record is the one
# that describes the window the next capture will contain. Re-read per capture
# rather than once: a window that is moved, resized or fullscreened mid-run would
# otherwise be measured against rectangles for a window that no longer exists,
# and every region would report a mismatch that is the probe's fault.
refresh_layout() {
  ssh -o BatchMode=yes "$GUEST" "cat /tmp/content-probe.json 2>/dev/null" >"$LAYOUT" || true
  grep -q '"kind":' "$LAYOUT" || return 1
  python3 - "$LAYOUT" "$WORK/regions.txt" <<'PY'
import json, sys
last = None
for line in open(sys.argv[1]):
    line = line.strip()
    if not line:
        continue
    try:
        d = json.loads(line)
    except ValueError:
        continue
    if d.get("kind") == "layout":
        last = d
if last is None:
    sys.exit("no layout record")
with open(sys.argv[2], "w") as f:
    f.write(f"SCREEN {last['screen']['w']} {last['screen']['h']}\n")
    for r in last["regions"]:
        e = r["expect"]
        f.write(f"R {r['name']} {r['x']} {r['y']} {r['w']} {r['h']} {e[0]} {e[1]} {e[2]}\n")
PY
}

refresh_layout || {
  say "the page never declared a layout — see /tmp/content-probe-server.log in the guest" >&2
  ssh -o BatchMode=yes "$GUEST" "pkill -f probe_server.py; pkill -f '$APP'" >/dev/null 2>&1 || true
  exit 2; }

read -r _ SCR_W SCR_H < <(grep -m1 '^SCREEN ' "$WORK/regions.txt")
say "guest screen ${SCR_W}x${SCR_H}, $(grep -c '^R ' "$WORK/regions.txt") declared regions"

fails=0
for i in $(seq 1 "$CAPTURES"); do
  refresh_layout || { say "capture $i: no fresh layout" >&2; continue; }
  png="$WORK/cap-$i.png"
  "$SHOT" -o "$png" >/dev/null 2>&1 || { say "capture $i failed" >&2; continue; }
  IMG_W=$(identify -format '%w' "$png")
  IMG_H=$(identify -format '%h' "$png")

  bad=$(python3 - "$WORK/regions.txt" "$png" "$IMG_W" "$IMG_H" "$SCR_W" "$SCR_H" <<'PY'
import subprocess, sys
regions_path, png, iw, ih, sw, sh = sys.argv[1:7]
sx, sy = int(iw) / int(sw), int(ih) / int(sh)
# The palette every measured mean is classified against. Nearest-entry rather
# than a tolerance: the colours are far apart by construction, so "which of
# these is it" is answerable without naming a distance that counts as close,
# and a region that lost its fill reports as WHITE or BLACK by name instead of
# as an unexplained miss.
PALETTE = {
    "BG": (0x20, 0x20, 0x80), "RED": (0xe0, 0x10, 0x10), "GREEN": (0x10, 0xc0, 0x30),
    "YELLOW": (0xf0, 0xe0, 0x10), "MAGENTA": (0xd0, 0x10, 0xd0), "CYAN": (0x10, 0xd0, 0xe0),
    "ORANGE": (0xf0, 0x80, 0x10), "WHITE": (0xff, 0xff, 0xff), "BLACK": (0x00, 0x00, 0x00),
}
specs = []
for line in open(regions_path):
    p = line.split()
    if p and p[0] == "R":
        specs.append((p[1], *(int(v) for v in p[2:9])))
# Inset each rectangle before measuring so a one-pixel rounding error in the
# downscale cannot pull a neighbouring colour into the mean.
args = []
for name, x, y, w, h, *_ in specs:
    px, py = int(x * sx), int(y * sy)
    pw, ph = max(1, int(w * sx)), max(1, int(h * sy))
    ix, iy = px + max(1, pw // 6), py + max(1, ph // 6)
    iw2, ih2 = max(1, pw - 2 * max(1, pw // 6)), max(1, ph - 2 * max(1, ph // 6))
    args.append((name, ix, iy, iw2, ih2))
bad = []
for (name, x, y, w, h), spec in zip(args, specs):
    r = subprocess.run(["magick", png, "-crop", f"{w}x{h}+{x}+{y}", "+repage",
                        "-format", "%[fx:mean.r*255] %[fx:mean.g*255] %[fx:mean.b*255]",
                        "info:"], capture_output=True, text=True)
    try:
        mr, mg, mb = (float(v) for v in r.stdout.split())
    except ValueError:
        bad.append(f"{name}=UNREADABLE")
        continue
    got = min(PALETTE, key=lambda k: sum((a - b) ** 2 for a, b in zip(PALETTE[k], (mr, mg, mb))))
    want = min(PALETTE, key=lambda k: sum((a - b) ** 2 for a, b in zip(PALETTE[k], spec[5:8])))
    if got != want:
        bad.append(f"{name}: declared {want} measured {got} rgb=({mr:.0f},{mg:.0f},{mb:.0f})")
print("; ".join(bad))
PY
)
  if [ -n "$bad" ]; then
    fails=$((fails + 1))
    say "capture $i: $bad"
    [ -n "$KEEP" ] && say "  frame kept at $png"
  fi
  sleep 1
done

ssh -o BatchMode=yes "$GUEST" "pkill -f probe_server.py; pkill -f '$APP'" >/dev/null 2>&1 || true
say "$CAPTURES captures, $fails with a region that did not measure its declared colour"
[ "$fails" -eq 0 ] || exit 1
exit 0
