#!/usr/bin/env bash
# counter-budget.sh — read the six silent-loss classes out of one window of the
# fail log.
#
# Split out of `visual-gate.sh` so it can be tested against synthetic log text
# without a live boot. A parser that silently matches nothing reads exactly like
# a clean run, which is the failure mode this whole gate exists to remove.
#
#   scripts/visual-gate/counter-budget.sh WINDOW_LOG
#
# Prints one `name<TAB>count` line per class, in a fixed order. Exits 0 when
# every class read zero, 1 when any did not, 2 when the file cannot be read.
#
# Three classes are line families, keyed on the prefix their emitter writes at
# the start of a line. Three are `note_store_route` counts, which arrive as
# `key=value` fields on a `store_routes` line — that line is emitted once per
# per-second window, so a class has to be summed across every such line in the
# window rather than read off the last one.
set -euo pipefail
export LC_ALL=C

[ $# -eq 1 ] || { echo "counter-budget: usage: counter-budget.sh WINDOW_LOG" >&2; exit 2; }
WINDOW="$1"
[ -r "$WINDOW" ] || { echo "counter-budget: cannot read $WINDOW" >&2; exit 2; }

# `deferred_flush_lost` — a guest render this device dropped. Always a real loss.
# `mapping_page_drift`  — the page list changed under an armed window.
# `THRASH present_action_starvation` — one class spelled as two words; zero
#   across the whole accumulated log to date, so a first occurrence is a result.
LINE_CLASSES=(
  'deferred_flush_lost|deferred_flush_lost '
  'mapping_page_drift|mapping_page_drift '
  'present_action_starvation|THRASH present_action_starvation '
)

# `gw_audit_unsound` — the gather witness refuted itself; a stale image is being
#   served.
# `render_flush_over_guest_write` — documented as expected-never; if it fires the
#   writeback ordering repair has broken.
# `tdc_overflow` — the census target map overflowed and re-seeded. Only
#   meaningful when the census probe is on, and it cannot fire when it is off.
ROUTE_CLASSES=(gw_audit_unsound render_flush_over_guest_write tdc_overflow)

hot=0

for entry in "${LINE_CLASSES[@]}"; do
  name=${entry%%|*}
  prefix=${entry#*|}
  n=$(grep -c -- "^$prefix" "$WINDOW" || true)
  printf '%s\t%s\n' "$name" "$n"
  [ "$n" -eq 0 ] || hot=1
done

for name in "${ROUTE_CLASSES[@]}"; do
  n=$(awk -v key="$name" '
    /^store_routes /{
      for (i = 2; i <= NF; i++) {
        split($i, kv, "=")
        if (kv[1] == key) total += kv[2]
      }
    }
    END { print total + 0 }' "$WINDOW")
  printf '%s\t%s\n' "$name" "$n"
  [ "$n" -eq 0 ] || hot=1
done

exit "$hot"
