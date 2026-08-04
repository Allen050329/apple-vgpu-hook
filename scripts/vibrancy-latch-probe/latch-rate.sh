#!/usr/bin/env bash
# latch-rate.sh — how often does a boot come up with the vibrancy rail broken?
#
# `metal_draw/vulkan.rs`'s `note_gva_resident_aliasing` records the property that
# makes every single-boot reading of this class useless: pooled over five
# 14-round boots on one binary the corruption was 0, 0, 0, 14, 0 — **all-or-
# nothing per boot, never mixed**. Something latches once per boot and then holds
# for every round after it. So a run that comes back clean has measured that
# boot's latch, not the change under test, and five clean runs in a row are
# exactly what a one-in-five rate looks like.
#
# Scoring this class needs boots. That was impractical while the verdict was a
# person looking at two PNGs; `pane-frost-gate.sh` makes it a number, and this
# script is the loop around it.
#
# Each iteration is a full boot from the same immutable snapshot, so the boots
# are independent by construction — nothing carries over but the binary.
#
#   settle   boot, wait for ssh
#   probe    vibrancy-latch-probe: pane / load / same pane again
#   gate     pane-frost-gate on the two panes
#   record   verdict + the fail log, then kill the VM
#
# Reports the rate and every per-boot RMSE, because the distribution is the
# finding: a class that latches per boot shows up as a bimodal column (a cluster
# at the noise floor and a cluster far above it), and a mean would hide exactly
# that.
#
# Usage:
#   latch-rate.sh [--boots N] [--load-seconds S] [--census-seconds C] [--out DIR]
#
# Exits 0 whenever the loop ran. It does not fail on a degraded boot — that is
# the measurement, not an error.
set -euo pipefail
export LC_ALL=C

BOOTS=6
LOAD_SECONDS=150
CENSUS_SECONDS=12
OUT=""
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROBE="$REPO_ROOT/scripts/vibrancy-latch-probe/vibrancy-latch-probe.sh"
GATE="$REPO_ROOT/scripts/vibrancy-latch-probe/pane-frost-gate.sh"
FAILLOG="${REIMS_FAIL_LOG:-/tmp/reims-vgpu-fail.log}"

while [ $# -gt 0 ]; do
  case "$1" in
    --boots) BOOTS="$2"; shift 2 ;;
    --load-seconds) LOAD_SECONDS="$2"; shift 2 ;;
    --census-seconds) CENSUS_SECONDS="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    -h|--help) sed -n '2,36p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
    *) echo "latch-rate: unknown argument $1" >&2; exit 2 ;;
  esac
done

WORK="${OUT:-$(mktemp -d -t latch-rate-XXXXXX)}"
mkdir -p "$WORK"
say() { echo "latch-rate: $*"; }

# `pgrep -x` cannot see this process: the name is longer than 15 characters.
kill_vm() {
  pkill -9 -f 'qemu-system-x86_64' 2>/dev/null || true
  # Give the port forward time to come back, or the next boot dies on
  # "Could not set up host forwarding rule 'tcp::2222-:22'" — and the ssh wait
  # then succeeds against nothing.
  sleep 6
}

# The probe's phases plus the boot, with headroom; the boot's own hard kill is
# the backstop that keeps a wedged guest from stalling the sweep.
budget=$(( LOAD_SECONDS + 4 * CENSUS_SECONDS + 420 ))

say "work dir $WORK — $BOOTS boots, ${LOAD_SECONDS}s load each"
verdicts="$WORK/verdicts.tsv"
: >"$verdicts"

for i in $(seq 1 "$BOOTS"); do
  boot_dir="$WORK/boot-$i"
  mkdir -p "$boot_dir"
  kill_vm
  rm -f "$FAILLOG"
  say "boot $i/$BOOTS"
  TESTING_TIMEOUT="$budget" "$REPO_ROOT/vm/boot-x86.sh" --device reims-vgpu-pci --testing \
    >"$boot_dir/boot.log" 2>&1 &
  boot_pid=$!

  waited=0
  until ssh -o ConnectTimeout=5 -o BatchMode=yes macos-vm true 2>/dev/null; do
    sleep 5
    waited=$((waited + 5))
    if [ "$waited" -ge 300 ]; then break; fi
    # A boot that died takes its guest with it; do not wait out the whole budget
    # against a process that is gone.
    kill -0 "$boot_pid" 2>/dev/null || break
  done
  if ! ssh -o ConnectTimeout=5 -o BatchMode=yes macos-vm true 2>/dev/null; then
    say "boot $i never came up — see $boot_dir/boot.log" >&2
    printf '%s\t%s\t%s\n' "$i" "no-boot" "-" >>"$verdicts"
    continue
  fi

  if ! "$PROBE" --load-seconds "$LOAD_SECONDS" --census-seconds "$CENSUS_SECONDS" \
      --out "$boot_dir" >"$boot_dir/probe.log" 2>&1; then
    say "boot $i: the probe refused a verdict — see $boot_dir/probe.log" >&2
    printf '%s\t%s\t%s\n' "$i" "probe-refused" "-" >>"$verdicts"
    cp "$FAILLOG" "$boot_dir/full-boot.log" 2>/dev/null || true
    continue
  fi

  gate_out=$("$GATE" --before "$boot_dir/before.png" --after "$boot_dir/after.png" 2>&1) \
    && verdict="clean" || verdict="degraded"
  echo "$gate_out" >"$boot_dir/gate.log"
  rmse=$(echo "$gate_out" | sed -n 's/.*pane=[^ ]* rmse=\([0-9.e-]*\).*/\1/p')
  # The gate's own refusal (control moved) is neither clean nor degraded.
  echo "$gate_out" | grep -q "not of the same scene" && verdict="gate-refused"
  printf '%s\t%s\t%s\n' "$i" "$verdict" "${rmse:--}" >>"$verdicts"
  say "boot $i: $verdict (pane rmse ${rmse:--})"
  cp "$FAILLOG" "$boot_dir/full-boot.log" 2>/dev/null || true
done

kill_vm

say ""
say "boot	verdict	pane_rmse"
sed 's/^/latch-rate:   /' "$verdicts"
clean=$(awk -F'\t' '$2 == "clean"' "$verdicts" | wc -l)
degraded=$(awk -F'\t' '$2 == "degraded"' "$verdicts" | wc -l)
scored=$((clean + degraded))
say ""
if [ "$scored" -gt 0 ]; then
  say "degraded $degraded of $scored scored boots"
else
  say "no boot produced a scoreable pair"
fi
say "per-boot evidence in $WORK/boot-*/ (screenshots, census logs, full-boot.log)"
exit 0
