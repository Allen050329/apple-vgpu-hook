#!/usr/bin/env bash
#
# scripts/browser-probe/web-gpu-probe.sh — ask each browser, in its own words,
# whether WebGL/WebGPU came up on hardware and how many frames it sustained.
#
#   scripts/browser-probe/web-gpu-probe.sh safari|chrome|firefox [extra browser args…]
#
# The page is served over http from inside the guest and posts its result back
# to that same server (see probe_server.py for why). One browser per run: they
# fight over the foreground and a background window's requestAnimationFrame is
# throttled, which would be scored as a frame-rate result.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GUEST="${GUEST:-macos-vm}"
PORT="${PROBE_PORT:-8998}"
BROWSER="${1:?usage: web-gpu-probe.sh safari|chrome|firefox [args…]}"
shift || true
SETTLE="${PROBE_SETTLE:-30}"

scp -q "$SCRIPT_DIR/probe_server.py" "$SCRIPT_DIR/gpu-probe.html" "$GUEST:/tmp/"

case "$BROWSER" in
  safari)  APP="Safari" ;;
  chrome)  APP="Google Chrome" ;;
  firefox) APP="Firefox" ;;
  *) echo "unknown browser: $BROWSER" >&2; exit 64 ;;
esac

ssh "$GUEST" "pkill -f probe_server.py >/dev/null 2>&1 || true
pkill -f '$APP' >/dev/null 2>&1 || true
sleep 2
nohup python3 /tmp/probe_server.py $PORT /tmp/gpu-probe.html /tmp/probe-result.json \
  >/tmp/probe-server.log 2>&1 &
sleep 2
open -a '$APP' ${*:+--args $*} 'http://127.0.0.1:$PORT/'
sleep $SETTLE
cat /tmp/probe-result.json 2>/dev/null || echo 'NO RESULT POSTED'"
