#!/usr/bin/env bash
# Report reims-vgpu items and struct fields that nothing reads.
#
# Rust's dead-code pass is the right tool for this, and on this crate it is
# switched off: `reims-vgpu` is a staticlib whose types are almost all `pub`,
# and a `pub` item in a library is reachable from outside the crate by
# definition, so rustc never calls it dead. That is why every previous sweep for
# unread state here was a grep, and why greps got it wrong in both directions —
# `.mapping_id` matches a dozen unrelated types, and a scan restricted to `src/`
# calls seven engine hooks dead that `tests/vk_engine_*.rs` uses.
#
# This script switches the pass back on, in a scratch copy; the working tree is
# never touched. Three rewrites make it correct:
#
#   1. Every leading `pub ` becomes `pub(crate) `, so nothing is externally
#      reachable and rustc's own analysis applies. That analysis understands
#      trait impls, macros, cfg arms and generics; a grep understands none.
#   2. `crates/reims-vgpu/tests/` is moved aside and its files are pulled into
#      the crate as `#[cfg(test)] #[path = ...] mod` items, with their
#      `reims_vgpu::` paths rewritten to `crate::`. Without this the
#      integration tests compile as separate crates that can no longer see a
#      `pub(crate)` item, every one of their uses vanishes from the analysis,
#      and the report is full of live code. Moving the directory is what stops
#      cargo from also building them the old way.
#   3. Both backend arms are compiled and only the INTERSECTION is reported. An
#      item can be live on one arm and cfg'd out of the other — every argument
#      struct `execute_dispatch_metal` builds reads as dead from the Vulkan arm.
#
# If either arm fails to compile, this exits non-zero and reports nothing. A
# null result from an instrument that could not run is not a clean bill.
#
# LIMIT: a derived impl that is itself used counts as a read of every field it
# touches. `#[derive(Debug)]` is neutralised below because `DeviceState` derives
# it and the fail log formats it, which alone was enough to make the field
# report permanently empty. `Clone`, `PartialEq` and `Hash` are NOT neutralised
# — removing them does not compile — so a field on a struct whose `clone()` is
# actually called will not be reported. An empty field report is therefore weak
# evidence, and a hit is strong evidence. `PresentState::mapping_id`, written at
# five sites and read at none, was found by hand and never by this script, for
# exactly that reason; `observe::gate::no_present_state_field_is_write_only`
# covers that one struct directly and does not have this blind spot.
#
# Item results — functions, methods, constants, variants, type aliases — carry
# no such caveat.
#
# It cannot tell you whether a finding should be deleted. Three classes are
# legitimately unread from Rust and recur here:
#
#   * Contract tables — register maps (`model/regs.rs`), SDK enum mirrors
#     (`backend/metal/raw_metal.rs`), wire field offsets (`runtime/decode/`).
#     A hole in a documented register map costs more than the line saves.
#   * Error and decline variants only a future decode path constructs.
#   * The C ABI surface in `qemu/abi.rs`, which QEMU calls and Rust cannot see.
#
# Everything else — a struct field written at five sites and read at none, a
# subsystem whose entry point only tests call — is what this is for.
#
# `--test-only` reports that last class, which the default cannot see. Rule 2
# above deliberately makes a test count as a use, because a helper only
# `tests/vk_engine_*.rs` calls is live code. The cost is that a product function
# whose *sole* caller is the unit test written to prove it works also reads as
# live — and that is not live code. It is a mechanism the product does not use,
# plus a test keeping it compiling. To separate the two, each arm is compiled a
# second time with cfg(test) off; an item dead in that build and live in the
# other is reached only from test code.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRATCH="${TMPDIR:-/tmp}/reims-dead-state.$$"
CRATE="$SCRATCH/crates/reims-vgpu"
FIELDS_ONLY=0
TEST_ONLY=0
KEEP=0

usage() {
  cat <<'EOF'
usage: scripts/dead-state/dead-state.sh [--fields-only|--test-only] [--keep]

Reports reims-vgpu items and struct fields that no code path reads, by
compiling a `pub`-downgraded copy of the crate so rustc's dead-code pass can
see them. Both backend arms are compiled; only findings dead on both are
reported. The working tree is never modified.

  --fields-only  Report only never-read struct fields. Highest signal: an
                 unread field is state some other rail may be expected to keep
                 consistent, which costs every author who reads the struct.
  --test-only    Report items the product never reaches and only test code
                 uses. These do not appear in the default report, which counts
                 a test as a use on purpose. Costs two extra compiles.
  --keep         Leave the scratch tree in place for inspection.

Exits 0 with a report, or non-zero if an arm failed to compile. It is a report,
not a gate — triage every hit against the three legitimately-unread classes in
the file header before deleting anything.
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --fields-only) FIELDS_ONLY=1 ;;
    --test-only) TEST_ONLY=1 ;;
    --keep) KEEP=1 ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "dead-state: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

if [ "$FIELDS_ONLY" -eq 1 ] && [ "$TEST_ONLY" -eq 1 ]; then
  echo "dead-state: --fields-only and --test-only select different reports" >&2
  exit 2
fi

cleanup() { [ "$KEEP" -eq 1 ] || rm -rf "$SCRATCH"; }
trap cleanup EXIT

echo "[dead-state] scratch copy: $SCRATCH"
mkdir -p "$SCRATCH/crates"
cp "$REPO/Cargo.toml" "$SCRATCH/"
[ -f "$REPO/Cargo.lock" ] && cp "$REPO/Cargo.lock" "$SCRATCH/"
# Both workspace members are copied. `reims-vgpu-wire` is a path dependency of
# `reims-vgpu` and a member of the root workspace, so omitting it makes cargo
# refuse the whole manifest before it compiles anything. Only `reims-vgpu` is
# rewritten below: the wire crate is a real dependency here, its `pub` surface
# is what `reims-vgpu` consumes, and downgrading it would break the build
# rather than measure anything.
(cd "$REPO/crates" && tar --exclude=target -cf - reims-vgpu reims-vgpu-wire) |
  (cd "$SCRATCH/crates" && tar -xf -)

# 1. Downgrade every `pub` that opens an item. Leaves the restricted forms
#    alone; an item declaration always starts its line in this crate.
find "$CRATE/src" -name '*.rs' -print0 |
  xargs -0 sed -i -E 's/^([[:space:]]*)pub (\(crate\)|\(super\)|\(in )?/\1pub\2 /; s/^([[:space:]]*)pub ([^(])/\1pub(crate) \2/'

# 1b. Neutralise `#[derive(Debug)]` on top-level structs.
#
#     A derived `Debug` reads every field, and once anything reachable formats
#     the outer struct, rustc counts every field of every struct nested inside
#     it as read. `DeviceState` derives `Debug` and the fail log formats it, so
#     without this every field in the device model reads as live and the report
#     is empty — which is exactly what it said before this pass existed.
#
#     Replaced with a stub impl that names the type and reads nothing, so code
#     that formats still compiles. Only column-0 structs: a stub for a struct
#     declared inside a `mod` cannot be appended at file scope. Enums keep their
#     derive; variant liveness is a different lint and is not masked this way.
python3 - "$CRATE/src" <<'PY'
import os, re, sys
root = sys.argv[1]
# The `#[cfg(...)]` guard is required. A stub is appended at file scope and
# carries no cfg of its own, so stubbing a cfg-gated struct emits an impl for a
# type that does not exist on the other arm. `metal_draw/vulkan.rs` is full of
# these; it is `include!`d rather than a module, so every item in it is
# individually gated. `#[cfg(test)]` matters just as much: `--test-only`
# compiles each arm a second time as a plain lib, where a stub for a
# test-only type names a type that does not exist.
#
# The whole leading attribute-and-doc block is therefore captured, not just the
# line directly above the `derive`. A doc comment between the two — which is how
# `runtime/host.rs` writes `RealRange`, `FakeHost`, `GuestWriteSet` and
# `Rewire` — otherwise hides the `#[cfg(test)]` from the guard below.
ATTR_OR_DOC = r'(?:^(?:#!?\[[^\n]*\]|///[^\n]*|//![^\n]*)\n)*'
pat = re.compile(
    r'(' + ATTR_OR_DOC + r'^#\[derive\(([^)]*)\)\]\n' + ATTR_OR_DOC +
    r'^(?:pub(?:\(crate\))?\s+)?struct\s+(\w+)\s*(?=[{(;]))',
    re.M,
)
for dirpath, _, files in os.walk(root):
    for f in files:
        if not f.endswith(".rs"):
            continue
        fp = os.path.join(dirpath, f)
        src = open(fp).read()
        stubs = []

        def fix(m):
            whole, derives, name = m.group(1), m.group(2), m.group(3)
            parts = [p.strip() for p in derives.split(",")]
            if "Debug" not in parts or "#[cfg" in whole:
                return whole
            rest = [p for p in parts if p != "Debug"]
            stubs.append(name)
            new = f"#[derive({', '.join(rest)})]" if rest else ""
            return whole.replace(f"#[derive({derives})]", new, 1)

        out = pat.sub(fix, src)
        if stubs:
            out += "\n" + "\n".join(
                "impl std::fmt::Debug for %s { fn fmt(&self, f: &mut std::fmt::Formatter<'_>)"
                ' -> std::fmt::Result { f.write_str("%s") } }' % (n, n)
                for n in stubs
            ) + "\n"
            open(fp, "w").write(out)
PY

# 2. Pull the integration tests inside the crate.
if [ -d "$CRATE/tests" ]; then
  mv "$CRATE/tests" "$CRATE/dead-state-tests"
  sed -i 's/\breims_vgpu::/crate::/g' "$CRATE"/dead-state-tests/*.rs
  python3 - "$CRATE" <<'PY'
import glob, os, sys
crate = sys.argv[1]
lib = os.path.join(crate, "src", "lib.rs")
lines = open(lib).read().split("\n")
# Inner attributes and the crate doc comment must stay first.
i = 0
while i < len(lines) and (
    lines[i].startswith("//!") or lines[i].startswith("#![") or not lines[i].strip()
):
    i += 1
mods = ["#[allow(unused_extern_crates)]", "extern crate self as reims_vgpu;"]
for t in sorted(glob.glob(os.path.join(crate, "dead-state-tests", "*.rs"))):
    name = os.path.basename(t)[:-3]
    mods += ['#[cfg(test)]', f'#[path = "../dead-state-tests/{name}.rs"]', f"mod __ds_{name};"]
lines.insert(i, "\n".join(mods) + "\n")
open(lib, "w").write("\n".join(lines))
PY
fi

# 3. Compile both arms. `--tests` is what turns on cfg(test), which is what
#    makes the pulled-in integration tests count. `--test-only` also compiles
#    each arm as a plain lib, where cfg(test) is off and no test — unit or
#    pulled-in integration — exists to count as a use.
run_arm() {
  local label="$1" out="$2" targets="$3"
  shift 3
  echo "[dead-state] compiling arm: $label"
  local log="$SCRATCH/log.$(echo "$label" | tr -c 'a-z0-9' '_')"
  # shellcheck disable=SC2086 # $targets is a deliberate word-split flag list.
  if ! (cd "$SCRATCH" && cargo check -p reims-vgpu $targets "$@" --message-format short) >"$log" 2>&1; then
    echo "[dead-state] FAILED to compile arm: $label" >&2
    grep -E ': error' "$log" | head -20 >&2
    echo "[dead-state] Reporting nothing: a null result from an instrument that" >&2
    echo "[dead-state] could not run is not a clean bill." >&2
    exit 1
  fi
  grep -E 'never read|never used|never constructed' "$log" |
    sed -E 's/^crates/  crates/; s/: warning: /  ->  /' |
    sort -u >"$out"
}

VULKAN_ARM=(--no-default-features --features backend-vulkan,host-window)
METAL_ARM=(--target aarch64-apple-darwin --features backend-metal)

run_arm 'vulkan,host-window' "$SCRATCH/hits.vulkan" --tests "${VULKAN_ARM[@]}"
run_arm 'metal / aarch64-apple-darwin' "$SCRATCH/hits.metal" --tests "${METAL_ARM[@]}"

comm -12 "$SCRATCH/hits.vulkan" "$SCRATCH/hits.metal" >"$SCRATCH/hits.both"

if [ "$TEST_ONLY" -eq 1 ]; then
  run_arm 'vulkan,host-window (lib only)' "$SCRATCH/lib.vulkan" --lib "${VULKAN_ARM[@]}"
  run_arm 'metal / aarch64-apple-darwin (lib only)' "$SCRATCH/lib.metal" --lib "${METAL_ARM[@]}"
  comm -12 "$SCRATCH/lib.vulkan" "$SCRATCH/lib.metal" >"$SCRATCH/lib.both"
  # Dead without tests, live with them: the only thing reaching it is a test.
  comm -23 "$SCRATCH/lib.both" "$SCRATCH/hits.both" >"$SCRATCH/testonly.both"
fi

echo
if [ "$FIELDS_ONLY" -eq 1 ]; then
  echo "[dead-state] never-read struct fields, dead on BOTH arms:"
  grep -E 'field' "$SCRATCH/hits.both" || echo "  none"
elif [ "$TEST_ONLY" -eq 1 ]; then
  echo "[dead-state] reached only from test code, on BOTH arms:"
  grep . "$SCRATCH/testonly.both" || echo "  none"
  echo
  echo "[dead-state] Each of these is a product mechanism no product caller"
  echo "[dead-state] uses, plus the test that keeps it compiling. Deleting one"
  echo "[dead-state] means deleting its test too — say so in the commit."
else
  echo "[dead-state] never read / never used / never constructed, on BOTH arms:"
  grep . "$SCRATCH/hits.both" || echo "  none"
fi

ONLY_V="$(comm -23 "$SCRATCH/hits.vulkan" "$SCRATCH/hits.metal" | wc -l | tr -d ' ')"
ONLY_M="$(comm -13 "$SCRATCH/hits.vulkan" "$SCRATCH/hits.metal" | wc -l | tr -d ' ')"
echo
echo "[dead-state] suppressed as arm-specific: ${ONLY_V} dead only on Vulkan, ${ONLY_M} only on Metal."
echo "[dead-state] Those are live on the other arm. Use --keep and read"
echo "[dead-state] hits.vulkan / hits.metal for the per-arm lists."
echo
echo "[dead-state] Triage before deleting. Contract tables (register maps, SDK"
echo "[dead-state] enum mirrors, wire field offsets) and the qemu/abi.rs C"
echo "[dead-state] surface are legitimately unread from Rust. See the header."
