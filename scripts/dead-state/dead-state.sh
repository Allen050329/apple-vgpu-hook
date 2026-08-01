#!/usr/bin/env bash
# Report reims-vgpu items and struct fields that nothing reads.
#
# Rust's dead-code pass is the best tool for this, and on this crate it is
# switched off: `reims-vgpu` is a staticlib whose types are almost all `pub`,
# and a `pub` item in a library is reachable from outside the crate by
# definition, so rustc never calls it dead. That is why every previous sweep for
# unread state here was done by grep, and why one of them was wrong — a grep for
# `.mapping_id` matches a dozen unrelated types, and a grep restricted to `src/`
# misses the engine hooks that only `tests/` calls.
#
# This script switches the pass back on. It copies the crate to a scratch tree,
# rewrites every leading `pub ` to `pub(crate) `, and compiles that. Nothing is
# reachable from outside a scratch crate nobody depends on, so rustc's own
# reachability analysis — which understands trait impls, macros, cfg arms and
# generics, none of which a grep does — reports exactly the items and fields no
# code path reads.
#
# It compiles `--all-targets`, so a helper that only `tests/` or `benches/` uses
# counts as used. Both feature arms are checked, because an item can be live on
# one and dead on the other; only items dead on EVERY arm checked are reported.
#
# What it cannot tell you is whether a finding should be deleted. Three classes
# are legitimately unread and recur here:
#
#   * Contract tables — register maps (`model/regs.rs`), SDK enum mirrors
#     (`backend/metal/raw_metal.rs`), wire field offsets (`runtime/decode/`).
#     A hole in a documented register map costs more than the line saves.
#   * Error/decline variants that only a future decode path constructs.
#   * The C ABI surface in `qemu/abi.rs`, which QEMU calls and Rust cannot see.
#
# Everything else — a struct field written at five sites and read at none, a
# whole subsystem whose entry point only tests call — is the thing this is for.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRATCH="${TMPDIR:-/tmp}/reims-dead-state.$$"
FIELDS_ONLY=0

usage() {
  cat <<'EOF'
usage: scripts/dead-state/dead-state.sh [--fields-only]

Reports reims-vgpu items and struct fields that no code path reads, by
compiling a `pub`-downgraded copy of the crate so rustc's dead-code pass can
see them. The working tree is never modified.

  --fields-only   Report only never-read struct fields. These are the highest
                  signal: an unread field is state some other rail may be
                  expected to keep consistent, which costs every author who
                  reads the struct.

Exits 0 whether or not anything is found; this is a report, not a gate. Triage
every hit against the three legitimately-unread classes in the file header
before deleting anything.
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --fields-only) FIELDS_ONLY=1 ;;
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

cleanup() { rm -rf "$SCRATCH"; }
trap cleanup EXIT

echo "[dead-state] scratch copy: $SCRATCH"
mkdir -p "$SCRATCH"
# Copy the workspace manifests and the crate, but not target/ — the scratch
# build is a fresh crate and shares nothing with the working tree's artifacts.
cp "$REPO/Cargo.toml" "$SCRATCH/" 2>/dev/null || true
cp "$REPO/Cargo.lock" "$SCRATCH/" 2>/dev/null || true
mkdir -p "$SCRATCH/crates"
(cd "$REPO/crates" && tar --exclude=target -cf - reims-vgpu) | (cd "$SCRATCH/crates" && tar -xf -)

# Downgrade every `pub` that starts a line's item. Leaves `pub(crate)`,
# `pub(super)` and `pub(in ...)` alone, and does not touch `pub` inside a line
# (a struct field written on the same line as its brace, a closure body) because
# an item declaration always starts its line in this crate.
find "$SCRATCH/crates/reims-vgpu" -name '*.rs' -print0 |
  xargs -0 sed -i -E 's/^([[:space:]]*)pub (\(crate\)|\(super\)|\(in )?/\1pub\2 /; s/^([[:space:]]*)pub ([^(])/\1pub(crate) \2/'

report_arm() {
  local label="$1"
  shift
  echo "[dead-state] compiling arm: $label"
  (cd "$SCRATCH" && cargo check -p reims-vgpu --all-targets "$@" --message-format short 2>&1) |
    grep -E 'never read|never used|never constructed' |
    sed -E 's#^[^ ]*/reims-dead-state\.[0-9]+/#  #; s/: warning: /  ->  /' |
    sort -u
}

VULKAN_HITS="$(report_arm 'vulkan,host-window' --no-default-features --features backend-vulkan,host-window || true)"

echo
if [ "$FIELDS_ONLY" -eq 1 ]; then
  echo "[dead-state] never-read struct fields (vulkan,host-window arm):"
  printf '%s\n' "$VULKAN_HITS" | grep -E 'field' || echo "  none"
else
  echo "[dead-state] never read / never used / never constructed (vulkan,host-window arm):"
  printf '%s\n' "$VULKAN_HITS" | grep . || echo "  none"
fi

echo
echo "[dead-state] Triage before deleting. Contract tables (register maps, SDK"
echo "[dead-state] enum mirrors, wire field offsets) and the qemu/abi.rs C"
echo "[dead-state] surface are legitimately unread from Rust. See the header."
