#!/usr/bin/env bash
#
# Report always-on log fields that only ever take ONE value.
#
# A `key=value` field that never varies across a whole log history is not a
# measurement. It is either structurally impossible (the branch that would move
# it returns before the line is emitted), or vestigial (it measures a mechanism
# that has since been deleted). Both are code to remove, and neither is visible
# from the source: the field looks like a live counter until you count it.
#
# See README.md for the triage rules and for what this deliberately does NOT
# report.

set -euo pipefail

LOG="${1:-/tmp/reims-vgpu-fail.log}"
MIN="${2:-200}"

if [[ ! -r "$LOG" ]]; then
    echo "constant-fields: cannot read $LOG" >&2
    echo "usage: $0 [logfile] [min-samples]" >&2
    exit 1
fi

echo "[constant-fields] log=$LOG  min-samples=$MIN"
echo

python3 - "$LOG" "$MIN" <<'PY'
import collections, re, sys

log, min_samples = sys.argv[1], int(sys.argv[2])

# Field name must start with a letter and be >=3 chars, so hex dumps and
# `t=<ms>` timestamps do not swamp the report. The lookbehind keeps `a=b` out of
# `foo.a=b` and of `x=y=z` tails.
field = re.compile(r'(?<![\w.])([a-z][a-z0-9_]{2,})=(\S+)')

values = collections.defaultdict(set)
counts = collections.Counter()

with open(log, errors='ignore') as fh:
    for line in fh:
        parts = line.split()
        # Field names repeat across unrelated line families, so bucket by the
        # emitting family (field 2, after the OFF/verbose marker). Without this
        # a `reason=` that is constant on one line and varied on another cancels
        # out and the constant one is never reported.
        family = parts[1] if len(parts) > 1 else '?'
        for key, value in field.findall(line):
            slot = (family, key)
            if len(values[slot]) < 6:
                values[slot].add(value)
            counts[slot] += 1

const = [
    (n, fam, key, next(iter(values[(fam, key)])))
    for (fam, key), n in counts.items()
    if len(values[(fam, key)]) == 1 and n >= min_samples
]
const.sort(reverse=True)

if not const:
    print("  none")
else:
    print(f"{'samples':>8}  {'family':<34} field=value")
    for n, fam, key, val in const:
        print(f"{n:8d}  {fam:<34} {key}={val}")

print()
print(f"[constant-fields] {len(const)} single-valued fields at >={min_samples} samples")
PY

cat <<'NOTE'

[constant-fields] Triage before deleting. A field can be legitimately constant:
[constant-fields]   - a standing alarm reading zero (it is doing its job)
[constant-fields]   - a capability/geometry constant for this host (page shift,
[constant-fields]     a cap's configured size, a device feature flag)
[constant-fields]   - an arm the workload never reached (drive the guest harder
[constant-fields]     before believing it, per the type-11 ladder's own note)
[constant-fields] What it is NOT: a bucket downstream of another bucket that is
[constant-fields] itself always zero, or a counterfactual for deleted code.
[constant-fields] Read the emitting code before cutting; the log only points.
NOTE
