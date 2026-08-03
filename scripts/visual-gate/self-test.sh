#!/usr/bin/env bash
# self-test.sh — does the counter budget actually read the log?
#
# The gate's whole value is that a silent loss stops being silent. A parser that
# matches nothing prints the same six zeros a clean boot does, so "the gate
# passed" would mean nothing and there would be no way to tell from the output.
# These cases fail without the parser and pass with it.
#
# The log text below is quoted from the emitters, not invented:
# `runtime::storage_flush` writes `deferred_flush_lost kind=...`,
# `runtime::mapper` writes `mapping_page_drift mid=...`, `runtime::drain` writes
# `THRASH present_action_starvation reason=...`, and `note_store_route` counts
# arrive as `key=value` fields on the `store_routes` line that same module
# formats once per per-second window.
#
#   scripts/visual-gate/self-test.sh
#
# Exits 0 when every case holds, 1 on the first that does not. No guest, no
# QEMU, no GPU.
set -uo pipefail
export LC_ALL=C

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUDGET="$SCRIPT_DIR/counter-budget.sh"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

fails=0
check() { # check NAME EXPECTED_EXIT EXPECTED_GREP <<< log text
  local name="$1" want_rc="$2" want="$3" log="$WORK/case.log"
  cat >"$log"
  local out rc
  out=$("$BUDGET" "$log")
  rc=$?
  if [ "$rc" != "$want_rc" ]; then
    echo "self-test: FAIL $name — exit $rc, wanted $want_rc" >&2
    echo "$out" | sed 's/^/self-test:   /' >&2
    fails=$((fails + 1))
    return
  fi
  if ! echo "$out" | grep -q -- "$want"; then
    echo "self-test: FAIL $name — output has no '$want'" >&2
    echo "$out" | sed 's/^/self-test:   /' >&2
    fails=$((fails + 1))
    return
  fi
  echo "self-test: ok $name"
}

# A window with nothing wrong in it is the working state, and it has to be
# distinguishable from a window the parser could not read.
check 'a clean window passes' 0 'deferred_flush_lost	0' <<'EOF'
drain_duty duty=0.001 draws=0 flushes=0
store_routes mapw_fence_flush=288 gvaw_fence_flush=432 gw_vouched=40
window_publish fresh=34 same_key=9
EOF

check 'a dropped render fails' 1 'deferred_flush_lost	1' <<'EOF'
drain_duty duty=0.97 draws=2270 flushes=523
deferred_flush_lost kind=gva reason=no_backend gva=0x7f0000 1920x1080 trigger=fence
EOF

check 'page drift under an armed window fails' 1 'mapping_page_drift	2' <<'EOF'
mapping_page_drift mid=11 task=3 reason=task_inactive pages=2048
mapping_page_drift mid=12 task=3 page=7/2048 gva=0x1000 reason=moved
EOF

check 'present starvation fails' 1 'present_action_starvation	1' <<'EOF'
THRASH present_action_starvation reason=pending_frames_cap ch=0 head=4 tail=9 unpainted=8 episode=1
EOF

# The route classes are the ones a naive parser gets wrong: they are fields on a
# shared line, not lines of their own.
check 'an unsound witness fails' 1 'gw_audit_unsound	3' <<'EOF'
store_routes gw_audit_unsound=3 gw_vouched=40 mapw_fence_flush=288
EOF

check 'route counts sum across windows' 1 'tdc_overflow	5' <<'EOF'
store_routes tdc_overflow=2 tdc_frames=1200
store_routes tdc_overflow=3 tdc_frames=1811
EOF

check 'the writeback ordering repair breaking fails' 1 'render_flush_over_guest_write	1' <<'EOF'
store_routes mapw_fence_flush=288 render_flush_over_guest_write=1
EOF

# A route name that is a prefix of another must not be counted for it, or a
# class could read hot because an unrelated counter shares its opening letters.
check 'a longer field name is not this class' 0 'tdc_overflow	0' <<'EOF'
store_routes tdc_overflow_reseeds=7 tdc_frames=1200
EOF

# The same shape on the line side: `deferred_flush_lost_probe` is not
# `deferred_flush_lost`, and a substring match anywhere in a line is not either.
check 'a line family matches its own prefix only' 0 'deferred_flush_lost	0' <<'EOF'
deferred_flush_lost_probe kind=gva gva=0x1000
readback_split note=deferred_flush_lost was considered
EOF

# Every class is reported on every run, so a reader can tell "this class read
# zero" from "this class is no longer parsed".
n=$("$BUDGET" /dev/null | wc -l)
if [ "$n" = 6 ]; then
  echo "self-test: ok all six classes are reported"
else
  echo "self-test: FAIL only $n classes reported, wanted 6" >&2
  fails=$((fails + 1))
fi

if [ "$fails" = 0 ]; then
  echo "self-test: PASS"
  exit 0
fi
echo "self-test: FAIL — $fails case(s)" >&2
exit 1
