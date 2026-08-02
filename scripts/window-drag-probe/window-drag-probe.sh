#!/usr/bin/env bash
# window-drag-probe.sh — what does this device do while a window is being dragged?
#
# Goal 6 is "window-dragging performance of Safari should be a stable 120 fps".
# Nothing measured it. `scripts/browser-probe` measures rAF inside a page, which
# is the wrong instrument twice over: a drag is window-server compositing rather
# than page script, and AGENTS.md records that Safari's rAF here is bimodal at
# ~59 and ~118 with nothing between, so a single rAF figure cannot support a
# claim about a code change in either direction.
#
# The device-side counters do not share that bimodality, so they are the result
# and this script is the harness that makes them mean something:
#
#   host_window_cadence  present_hz, offered_hz  — frames this device put out
#   drain_duty           duty, draw_us, flush_us — what the drain worker spent
#   readback_split       fence_us, bar_us, gpu_us
#
# The drag is a real Quartz event stream (`drag.py`), not an accessibility
# reposition, because a pointer held across a title bar and a window teleported
# by the AX API do not take the same path through the window server.
#
# `drag.py` reports what it actually posted. A drag posted at 40 Hz cannot show
# a 120 Hz device, so a run whose posted rate fell short says so instead of
# reporting the device as slow — the standing lesson from
# `scripts/web-content-probe`, whose first clean verdict came from a stressor
# that produced nothing.
#
# Usage:
#   scripts/window-drag-probe/window-drag-probe.sh [--seconds N] [--hz N]
#                                                  [--app "Safari"] [--keep DIR]
#
# Exits 0 when the drag was posted at the requested rate and the counters were
# collected, 2 on a setup failure. It does not fail on a slow device: this is an
# instrument, and the number it prints is the result.
set -euo pipefail
export LC_ALL=C

SECONDS_RUN=15
HZ=120
APP="Safari"
KEEP=""
GUEST="${GUEST:-macos-vm}"
FAILLOG="${REIMS_FAIL_LOG:-/tmp/reims-vgpu-fail.log}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

while [ $# -gt 0 ]; do
  case "$1" in
    --seconds) SECONDS_RUN="$2"; shift 2 ;;
    --hz) HZ="$2"; shift 2 ;;
    --app) APP="$2"; shift 2 ;;
    --keep) KEEP="$2"; shift 2 ;;
    -h|--help) sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
    *) echo "window-drag-probe: unknown argument $1" >&2; exit 2 ;;
  esac
done

WORK="${KEEP:-$(mktemp -d)}"
mkdir -p "$WORK"
[ -n "$KEEP" ] || trap 'rm -rf "$WORK"' EXIT
say() { echo "window-drag-probe: $*"; }
osa() { ssh -o BatchMode=yes "$GUEST" "osascript -e '$1'" 2>/dev/null; }

ssh -o ConnectTimeout=8 -o BatchMode=yes "$GUEST" true 2>/dev/null || {
  say "no guest at $GUEST" >&2; exit 2; }
[ -f "$FAILLOG" ] || { say "no fail log at $FAILLOG — is a boot running?" >&2; exit 2; }

scp -q "$SCRIPT_DIR/drag.py" "$GUEST:/tmp/reims-drag.py"

# A window that fills the screen leaves the compositor almost nothing to
# recomposite behind it, and one that is tiny damages almost nothing. Half the
# screen, placed off-centre so the drag path stays on-screen throughout.
ssh -o BatchMode=yes "$GUEST" "open -a '$APP' about:blank" >/dev/null 2>&1 || true
sleep 5
osa "tell application \"System Events\" to tell process \"$APP\" to set position of window 1 to {320, 180}" >/dev/null || true
osa "tell application \"System Events\" to tell process \"$APP\" to set size of window 1 to {1000, 640}" >/dev/null || true
sleep 2

# Grab the title bar. Read the window's real frame rather than assuming the
# reposition took: if the app refused it, dragging at the assumed point grabs
# the desktop and the run measures nothing.
POS=$(osa "tell application \"System Events\" to tell process \"$APP\" to get position of window 1" || true)
SIZ=$(osa "tell application \"System Events\" to tell process \"$APP\" to get size of window 1" || true)
WX=$(echo "$POS" | awk -F', *' '{print $1}')
WY=$(echo "$POS" | awk -F', *' '{print $2}')
WW=$(echo "$SIZ" | awk -F', *' '{print $1}')
case "${WX:-}${WY:-}${WW:-}" in
  ''|*[!0-9-]*) say "could not read $APP's window frame (pos '$POS' size '$SIZ')" >&2; exit 2 ;;
esac
# Middle of the title bar. 14 px down is inside the bar for every macOS window
# style and clear of the traffic lights, which sit at the left.
GX=$((WX + WW / 2))
GY=$((WY + 14))
say "dragging $APP's title bar at ($GX,$GY) for ${SECONDS_RUN}s at ${HZ} Hz"

# Mark the fail log so only lines the drag produced are read. Byte offset rather
# than a timestamp: the log's `t=` is device time and this shell's clock is not.
OFF=$(stat -c %s "$FAILLOG")

DRAG=$(ssh -o BatchMode=yes "$GUEST" \
  "/usr/bin/python3 /tmp/reims-drag.py $GX $GY $SECONDS_RUN $HZ 180 90" 2>"$WORK/drag.err") || {
  say "the drag did not run — see $WORK/drag.err:" >&2; sed 's/^/  /' "$WORK/drag.err" >&2; exit 2; }

tail -c "+$((OFF + 1))" "$FAILLOG" >"$WORK/window.log"
say "drag: $DRAG"

posted_hz=$(echo "$DRAG" | python3 -c 'import json,sys; print(json.load(sys.stdin)["posted_hz"])')
# Short of the ask by more than a fifth and the drag, not the device, is the
# slow thing. Say so rather than reporting the device's rate as its ceiling.
if awk -v p="$posted_hz" -v h="$HZ" 'BEGIN{exit !(p < 0.8 * h)}'; then
  say "the drag was posted at ${posted_hz} Hz against a requested ${HZ} Hz — the \
counters below are bounded by the drag, not by this device" >&2
fi

python3 - "$WORK/window.log" <<'PY'
import re, statistics, sys

text = open(sys.argv[1], errors="replace").read()


def rows(family, keys):
    out = []
    for line in text.splitlines():
        if f" {family} " not in f" {line} ":
            continue
        got = {}
        for k in keys:
            m = re.search(rf"\b{k}=([0-9.]+)", line)
            if m:
                got[k] = float(m.group(1))
        if len(got) == len(keys):
            out.append(got)
    return out


def show(label, vals, unit=""):
    if not vals:
        print(f"  {label:<22} (no samples)")
        return
    vals = sorted(vals)
    med = statistics.median(vals)
    print(f"  {label:<22} n={len(vals):<4} min={vals[0]:.2f} med={med:.2f} "
          f"max={vals[-1]:.2f}{unit}")


cad = rows("host_window_cadence", ["present_hz", "offered_hz", "window_ms"])
duty = rows("drain_duty", ["duty", "draw_us", "draws", "flush_us", "flushes",
                           "max_tranche_us"])
rb = rows("readback_split", ["fence_us", "fence"])

print("host_window_cadence — frames this device put out")
show("present_hz", [r["present_hz"] for r in cad], " Hz")
show("offered_hz", [r["offered_hz"] for r in cad], " Hz")
print("drain_duty — what the drain worker spent")
show("duty", [r["duty"] for r in duty])
show("max_tranche_us", [r["max_tranche_us"] for r in duty], " us")
show("draw_us/draw", [r["draw_us"] / r["draws"] for r in duty if r["draws"]], " us")
show("flush_us/flush", [r["flush_us"] / r["flushes"] for r in duty if r["flushes"]], " us")
print("readback_split — the device's GPU cost")
show("fence_us/fence", [r["fence_us"] / r["fence"] for r in rb if r["fence"]], " us")

# The pacing claim goal 6 is about, stated as the counters see it rather than as
# a mean. A "stable 120" that spends a second at 60 is not stable, and a median
# hides exactly that.
hz = [r["present_hz"] for r in cad]
if hz:
    low = sum(1 for v in hz if v < 100)
    print(f"\nseconds below 100 Hz: {low}/{len(hz)}"
          f"   worst second: {min(hz):.1f} Hz")
PY

[ -n "$KEEP" ] && say "counters kept in $WORK/window.log"
exit 0
