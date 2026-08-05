#!/usr/bin/env bash
# silent-arms — find `match` arms guarded by an `if` inside a `match` whose
# catch-all is a no-op, where a record that fails the guard is dropped without
# a word.
#
#   scripts/silent-arms/silent-arms.sh [crate-src-dir ...]
#
# Defaults to both workspace crates' `src` trees, skipping test modules. Most
# hits are not findings — see README.md for the triage rule before editing
# anything.
set -euo pipefail

roots=("$@")
if [ ${#roots[@]} -eq 0 ]; then
  here="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
  roots=("$here/crates/reims-vgpu/src" "$here/crates/reims-vgpu-wire/src")
fi

python3 - "${roots[@]}" <<'PY'
import re, sys, pathlib

roots = [pathlib.Path(r) for r in sys.argv[1:]]
files = sorted(p for r in roots for p in r.rglob('*.rs') if p.name != 'tests.rs')

MATCH_OPEN = re.compile(r'\bmatch\b.*\{\s*$')
NOOP_CATCHALL = re.compile(r'^\s*_\s*=>\s*(\{\s*\}|\(\))\s*,?\s*$', re.M)
# A guard that only asks whether the decoder saw the field at all, or whether
# the record is long enough to hold it. Both are the decoder reporting absence
# rather than the guest having asked for something, so the arm not running is
# not a loss. Note `word_count >= N`: a SPIR-V instruction too short for the
# field it would name is a truncated instruction, not a dropped request.
DECODER_PRESENCE = re.compile(
    r'\b(has_[a-z_]+|is_some|is_none|\.is_empty|word_count\s*>=|\.len\(\)\s*[=<>!]=)')


def guard_of(arm):
    """The guard expression of a match arm, or None.

    The arm separator is the FIRST `=>`: `==`, `>=` and `!=` do not contain one,
    and a `=>` inside the arm's body comes later. Taking the head this way is
    what makes `x if x == FOO =>` match, which a regex excluding `=` silently
    does not — and that failure looks exactly like a clean report.
    """
    head, sep, _ = arm.partition('=>')
    if not sep or not re.search(r'\bif\b', head):
        return None
    return head.strip()

hits = presence = 0
for path in files:
    lines = path.read_text().split('\n')
    for i, line in enumerate(lines):
        if not MATCH_OPEN.search(line):
            continue
        # Brace-walk the match block.
        depth, j, body, started = 0, i, [], False
        while j < len(lines):
            for ch in lines[j]:
                if ch == '{':
                    depth += 1
                    started = True
                elif ch == '}':
                    depth -= 1
            body.append((j, lines[j]))
            if started and depth == 0:
                break
            j += 1
        if not NOOP_CATCHALL.search('\n'.join(t for _, t in body)):
            continue
        # Arms sit at depth 1 of this match.
        d = 0
        for ln, text in body:
            stripped = text.strip()
            guard = guard_of(stripped) if d == 1 else None
            if guard is not None:
                if DECODER_PRESENCE.search(guard):
                    presence += 1
                else:
                    hits += 1
                    print(f'  {path}:{ln + 1}\n      {stripped[:140]}')
            for ch in text:
                if ch == '{':
                    d += 1
                elif ch == '}':
                    d -= 1

print()
print(f'[silent-arms] {hits} arms guarded on a decoded value, '
      f'{presence} on decoder presence (suppressed).')
print('[silent-arms] Triage before editing: a guard on a decoded value that')
print('[silent-arms] falls into `_ => {}` loses guest work in silence, but a')
print('[silent-arms] catch-all that is a real no-op state does not. README.md')
print('[silent-arms] lists both, with the ones already adjudicated.')
PY
