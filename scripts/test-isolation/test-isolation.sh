#!/usr/bin/env bash
# test-isolation — find tests whose result depends on what ran before them.
#
#   scripts/test-isolation/test-isolation.sh [--features <list>]
#
# The suite runs in one process, and several things this crate deliberately
# keeps for a whole boot outlive the test that touched them: the `first_sight`
# and `state_changed` dedup registries, `observe::sink`'s process-monotonic
# clock, the `ACC` phase accumulators, every `OnceLock`. A test can therefore
# pass because a predecessor left the process in a state it needs, or fail
# because one spent a latch it wanted — and which happens depends on libtest's
# name ordering, so a rename can flip it.
#
# This runs every `--lib` test in its own process and reports the ones whose
# isolated result differs from the suite's. Both directions are findings:
#
#   passes in-suite, fails alone   the test needs a predecessor; its assertion
#                                  is not proving what it says
#   fails in-suite, passes alone   the test was poisoned by a predecessor
#
# Exits non-zero when either set is non-empty. See README.md for triage.
set -euo pipefail

features="backend-vulkan,host-window"
while [ $# -gt 0 ]; do
  case "$1" in
    --features) features="$2"; shift 2 ;;
    *) echo "[test-isolation] unknown argument: $1" >&2; exit 2 ;;
  esac
done

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"

echo "[test-isolation] building --lib with --features $features"
bin="$(cargo test -p reims-vgpu --no-default-features --features "$features" \
        --lib --no-run --message-format=json 2>/dev/null \
       | grep -o '"executable":"[^"]*reims_vgpu-[^"]*"' | tail -1 | cut -d'"' -f4)"

if [ -z "${bin:-}" ] || [ ! -x "$bin" ]; then
  echo "[test-isolation] could not locate the built --lib binary; refusing a verdict" >&2
  exit 2
fi

names="$(mktemp)"; trap 'rm -f "$names"' EXIT
"$bin" --list --format terse 2>/dev/null | sed -n 's/: test$//p' > "$names"
total="$(wc -l < "$names")"

# Refuse a verdict rather than report a clean sweep over nothing. A `--list`
# that stops working — a libtest format change, a build that produced no tests —
# otherwise reads exactly like a suite with no order-dependence in it.
if [ "$total" -lt 100 ]; then
  echo "[test-isolation] only $total tests listed; the scanner cannot see the" \
       "suite, so it has nothing to report on. Refusing a verdict." >&2
  exit 2
fi

echo "[test-isolation] $total tests; running the suite once, then each alone"

suite_out="$(mktemp)"; trap 'rm -f "$names" "$suite_out"' EXIT
"$bin" --test-threads=1 > "$suite_out" 2>&1 || true
# libtest prints `test <name> ... FAILED` for each failure in the run.
suite_failed="$(sed -n 's/^test \(.*\) \.\.\. FAILED$/\1/p' "$suite_out" | sort)"

alone_failed=""
while IFS= read -r t; do
  if ! "$bin" --exact "$t" --test-threads=1 >/dev/null 2>&1; then
    alone_failed="$alone_failed$t"$'\n'
  fi
done < "$names"
alone_failed="$(printf '%s' "$alone_failed" | sed '/^$/d' | sort)"

needs_predecessor="$(comm -13 <(printf '%s\n' "$suite_failed" | sed '/^$/d') \
                              <(printf '%s\n' "$alone_failed" | sed '/^$/d'))"
poisoned="$(comm -23 <(printf '%s\n' "$suite_failed" | sed '/^$/d') \
                     <(printf '%s\n' "$alone_failed" | sed '/^$/d'))"

status=0
if [ -n "$needs_predecessor" ]; then
  echo
  echo "[test-isolation] pass in-suite, FAIL alone — these need a predecessor:"
  printf '  %s\n' $needs_predecessor
  status=1
fi
if [ -n "$poisoned" ]; then
  echo
  echo "[test-isolation] fail in-suite, PASS alone — these were poisoned by one:"
  printf '  %s\n' $poisoned
  status=1
fi
[ "$status" -eq 0 ] && echo "[test-isolation] $total tests, no order-dependence"
exit "$status"
