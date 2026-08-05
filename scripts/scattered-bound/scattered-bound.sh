#!/usr/bin/env bash
# scattered-bound — find a validity rule that is written out at every site that
# needs it, instead of once where the constant it bounds is declared.
#
#   scripts/scattered-bound/scattered-bound.sh [crate-src-dir ...]
#
# Defaults to both workspace crates' `src` trees. See README.md for what each
# report class means and which ones are not findings.
set -euo pipefail

roots=("$@")
if [ ${#roots[@]} -eq 0 ]; then
  here="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
  roots=("$here/crates/reims-vgpu/src" "$here/crates/reims-vgpu-wire/src")
fi

python3 - "${roots[@]}" <<'PY'
import re, sys, pathlib
from collections import defaultdict

roots = [pathlib.Path(r) for r in sys.argv[1:]]
files = sorted(p for r in roots for p in r.rglob('*.rs'))

CONST = re.compile(r'\b([A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+)\b')
# `a >= C`, `a > C`, `C > a`, `C >= a` and the `<` mirrors.
#
# Four neighbouring shapes are deliberately not comparisons, and each one was a
# whole page of false report before it was excluded:
#
#   `pfn << PAGE_ENTRY_PFN_SHIFT`   a shift — the single largest source of noise,
#                                   because every page-table fixture writes one
#   `Vec<MAX_THING>`                a generic argument
#   `1..MAX_CHANNELS`               a range, which restates no rule
#   `[T; MAX_CHANNELS]`             an array length, likewise
#   `Foo => MAX_THING,`             a match arm — the fat arrow is not a `>`
#
# The shift, the generic and the fat arrow are refused by the lookarounds below
# — the operator must not be doubled and must not follow an `=`. A range and an
# array length never produce a bare `<`/`>` token at all.


def relations(code, name):
    out = []
    for m in re.finditer(r'(?<![<>=])(<=|>=|<|>)(?![<>=])\s*' + re.escape(name) + r'\b', code):
        out.append(m.group(1))
    for m in re.finditer(r'\b' + re.escape(name) + r'\s*(?<![<>=])(<=|>=|<|>)(?![<>=])', code):
        out.append(m.group(1))
    return out

# Which spelling a site uses. A rule written as `id == 0 || id >= MAX` and its
# exact negation `id >= 1 && id < MAX` are the same rule; they do not grep as
# the same string, and that is what makes an inverted copy the expensive kind.
def polarity(ops):
    up = any(o in ('>', '>=') for o in ops)
    down = any(o in ('<', '<=') for o in ops)
    if up and down:
        return 'both'
    return 'refuses-above' if up else 'admits-below'

sites = defaultdict(list)
declared_in = {}
for path in files:
    # A fixture restating a bound is not a second rule — nothing ships it, and
    # the sites that write page-table entries by hand would otherwise drown the
    # report. Both spellings of "this file is tests" are skipped: a whole
    # `tests.rs` module and an inline `#[cfg(test)] mod`.
    if path.name == 'tests.rs' or path.parent.name == 'tests':
        continue
    try:
        text = path.read_text(errors='replace')
    except OSError:
        continue
    in_test, depth = False, 0
    for n, line in enumerate(text.split('\n'), 1):
        stripped = line.strip()
        if in_test:
            depth += line.count('{') - line.count('}')
            if depth <= 0:
                in_test = False
            continue
        if stripped.startswith('#[cfg(test)]'):
            in_test, depth = True, 0
            continue
        code = line.split('//')[0]
        for m in re.finditer(r'^\s*(?:pub(?:\([^)]*\))?\s+)?const\s+([A-Z][A-Z0-9_]+)\s*:', code):
            declared_in[m.group(1)] = path
        for name in set(CONST.findall(code)):
            ops = relations(code, name)
            if ops:
                sites[name].append((path, n, stripped, polarity(ops)))

rows = []
for name, occ in sites.items():
    owner = declared_in.get(name)
    files_hit = {p for p, _, _, _ in occ}
    away = {p for p in files_hit if p != owner}
    # One comparison away from the declaring file is the ordinary case — a cap
    # declared in a constants module and enforced at the single place it
    # applies. Two is the smallest number that can disagree, and 28 of the 52
    # rows this threshold removes are that ordinary shape. The cost is stated in
    # README.md: this script cannot see a *reintroduced* copy of a bound that
    # has already been consolidated to one predicate, because that copy is the
    # only comparison left. The per-bound test in `model::regs::tests` is what
    # covers that case, and it was measured to.
    if not away or len(occ) < 2:
        continue
    forms = {pol for _, _, _, pol in occ}
    rows.append((len(forms) > 1, len(away), len(occ), name, owner, occ))

rows.sort(key=lambda r: (not r[0], -r[1], -r[2]))

if not rows:
    print('[scattered-bound] every bound is compared only where it is declared.')
    raise SystemExit(0)

print('== a bound compared away from the constant that declares it ==')
print('   (the rule is the comparison; a second copy of it is a second rule)')
print()
for inverted, nfiles, ncmp, name, owner, occ in rows:
    flag = '  <-- two spellings, one of them inverted' if inverted else ''
    where = owner.name if owner else 'declared outside these trees'
    print(f'  {name}  ({ncmp} comparisons in {nfiles} files away from {where}){flag}')
    for path, n, text, pol in occ[:6]:
        print(f'      {path}:{n}  [{pol}]  {text[:96]}')
    if len(occ) > 6:
        print(f'      ... and {len(occ) - 6} more')
    print()

inv = sum(1 for r in rows if r[0])
print(f'[scattered-bound] {len(rows)} scattered bounds, {inv} written in more than one polarity')
PY
