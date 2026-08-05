#!/usr/bin/env bash
# scattered-struct — find values that travel as loose parameters when a type
# for them already exists, or obviously wants to.
#
#   scripts/scattered-struct/scattered-struct.sh [crate-src-dir ...]
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

roots = [pathlib.Path(r) for r in sys.argv[1:]]
files = sorted(p for r in roots for p in r.rglob('*.rs'))

FN = re.compile(
    r'^(?P<indent>\s*)(?:pub(?:\([^)]*\))?\s+)?(?:const\s+)?(?:async\s+)?'
    r'(?:unsafe\s+)?(?:extern\s+"[^"]*"\s+)?fn\s+(?P<name>\w+)')
# A struct literal field written in shorthand: `foo,` on its own line.
SHORTHAND = re.compile(r'^\s*(\w+),\s*$')
STRUCT_OPEN = re.compile(r'(?:^|[\s=(\[])(?P<ty>[A-Z]\w+)\s*\{\s*$')
# `) -> Status {` opens a function body, not a struct literal, and reading it
# as one made the scan collect every shorthand field in the function. Any line
# carrying a return arrow, or a block keyword before the brace, is not a
# literal.
NOT_LITERAL = re.compile(r'->|\b(?:fn|impl|match|if|else|for|while|loop|unsafe|mod)\b')
# `X => (lit, lit, ..)` — an enum arm flattened into loose primitives.
ARM_TUPLE = re.compile(r'^\s*(?P<pat>[\w:]+(?:\s*\|\s*[\w:]+)*)\s*=>\s*\((?P<t>[^()]*)\)\s*,?\s*$')
LITERAL = re.compile(r'^(true|false|-?\d[\w.]*|None)$')

def params_of(lines, i):
    """(name, type) of each parameter of the fn whose signature starts at i."""
    depth, buf, j = 0, [], i
    while j < len(lines) and j < i + 60:
        buf.append(lines[j])
        depth += lines[j].count('(') - lines[j].count(')')
        if depth <= 0 and '(' in ''.join(buf):
            break
        j += 1
    sig = ' '.join(buf)
    inner = sig[sig.find('(') + 1:]
    typed = []
    for part in re.split(r',(?![^<>()\[\]]*[>)\]])', inner):
        nm, sep, ty = part.partition(':')
        nm = nm.strip().lstrip('&').strip()
        ty = ty.split(')')[0].strip().rstrip(',')
        if sep and re.fullmatch(r'\w+', nm) and nm not in ('self',):
            typed.append((nm, ty))
    return typed, j

def body_range(lines, j):
    """Line range of the body whose opening brace is at or after line j."""
    k = j
    while k < len(lines) and '{' not in lines[k]:
        k += 1
    if k >= len(lines):
        return None
    depth, started, e = 0, False, k
    while e < len(lines):
        depth += lines[e].count('{') - lines[e].count('}')
        if depth > 0:
            started = True
        if started and depth <= 0:
            return (k, e)
        e += 1
    return None

NUMERIC = {'u16', 'u32', 'u64', 'usize', 'i32', 'f32'}

def longest_same_type_run(typed):
    """Longest run of adjacent parameters sharing one primitive type."""
    best, run = [], []
    for nm, ty in list(typed) + [(None, None)]:
        if ty in NUMERIC or ty == 'bool':
            if run and run[-1][1] == ty:
                run.append((nm, ty))
                continue
            if len(run) > len(best):
                best = run
            run = [(nm, ty)]
        else:
            if len(run) > len(best):
                best = run
            run = []
    return best

scatter, flattened, runs = [], [], []

for path in files:
    lines = path.read_text().splitlines()
    for i, line in enumerate(lines):
        m = FN.match(line)
        if not m:
            continue
        typed, j = params_of(lines, i)
        names = [n for n, _ in typed]
        if len(names) < 4:
            continue
        # Class 3: a run of adjacent parameters of one primitive type. Every
        # permutation of the run compiles; nothing at any call site can object.
        run = longest_same_type_run(typed)
        if len(run) >= 4:
            runs.append((path, i + 1, m.group('name'),
                         run[0][1], [n for n, _ in run]))
        span = body_range(lines, j)
        if not span:
            continue
        b, e = span
        pset = set(names)
        # Class 1: the body's own words say these belong together — a struct
        # literal built out of the function's parameters in shorthand.
        for k in range(b, min(e, b + 400)):
            sm = STRUCT_OPEN.search(lines[k])
            if not sm or NOT_LITERAL.search(lines[k]):
                continue
            # Walk exactly this literal's braces: depth goes 1 on the opening
            # line and the literal ends the moment it returns to 0. Without
            # that bound the scan ran to the end of the function and collected
            # shorthand fields from every later literal too, which is how a
            # 6-parameter function reported 11 matching fields.
            fields, depth, q = [], 0, k
            while q <= e:
                depth += lines[q].count('{') - lines[q].count('}')
                if q > k:
                    fm = SHORTHAND.match(lines[q])
                    if fm:
                        fields.append(fm.group(1))
                if depth <= 0 and q > k:
                    break
                q += 1
            hit = sorted({f for f in fields if f in pset})
            if len(hit) >= 4:
                scatter.append((path, i + 1, m.group('name'),
                                sm.group('ty'), len(hit), len(names)))
                break

    # Class 2: an enum flattened into a tuple of literals — the type existed
    # and the match is where it stopped existing.
    for i, line in enumerate(lines):
        am = ARM_TUPLE.match(line)
        if not am:
            continue
        parts = [p.strip() for p in am.group('t').split(',')]
        if len(parts) < 2 or not all(LITERAL.fullmatch(p) for p in parts):
            continue
        flattened.append((path, i + 1, am.group('pat'), len(parts)))

print("== a type is built out of the parameters that were passed to build it ==")
print("   (the struct already exists; the call sites spell its fields)\n")
if not scatter:
    print("   none\n")
for path, ln, fn, ty, hit, total in sorted(scatter, key=lambda r: -r[4]):
    print(f"  {path}:{ln}\n      fn {fn}  builds {ty} from {hit} of its {total} params")

print("\n== an enum arm flattened into loose primitives ==")
print("   (grouped by file; a run of arms over one enum is the finding)\n")
# A lone arm is not a finding — one `X => (0, 1)` is as likely to be a pair of
# unrelated numbers as a flattened enum — so a file contributes only when it
# holds two. Count what is reported, not what was matched: the two used to
# differ, and a footer reading "1 flattened arms" over a section that printed
# nothing sent a reader looking for a finding this script had already decided
# was below its own bar.
byfile = {}
for path, ln, pat, n in flattened:
    byfile.setdefault(path, []).append((ln, pat, n))
reported = {path: rows for path, rows in byfile.items() if len(rows) >= 2}
flattened_reported = sum(len(rows) for rows in reported.values())
if not reported:
    print("   none")
for path, rows in sorted(reported.items()):
    print(f"  {path}")
    for ln, pat, n in rows:
        print(f"      :{ln}  {pat} => {n} loose values")

print("\n== a run of adjacent parameters of one primitive type ==")
print("   (every permutation of the run compiles; sorted by run length)\n")
if not runs:
    print("   none")
for path, ln, fn, ty, names in sorted(runs, key=lambda r: -len(r[4])):
    print(f"  {path}:{ln}\n      fn {fn}  {len(names)} x {ty}: {', '.join(names)}")

print(f"\n[scattered-struct] {len(scatter)} scattered constructions, "
      f"{flattened_reported} flattened arms, {len(runs)} same-typed runs")
PY
