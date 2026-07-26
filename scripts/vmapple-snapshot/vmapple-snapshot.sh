#!/usr/bin/env bash
#
# scripts/vmapple-snapshot/vmapple-snapshot.sh
#
# Manage the vmapple guest's IMMUTABLE snapshot history under
# vm/guest/snapshots/<label>/{disk.img,aux.img.trimmed} + a `current` symlink.
# Snapshots are APFS clones (instant, COW) and read-only; they are NEVER
# overwritten. vm/boot-arm64.sh reverts to `current` on every boot and captures new
# snapshots via `--snapshot`; this tool covers the rest.
#
#   list                 list all snapshots (marks current)
#   current              print the current snapshot label
#   rollback <label>     repoint `current` to an existing snapshot (no data touched)
#   create [label]       clone the at-rest guest bundle into a NEW snapshot and
#                        make it current (guest must be shut down)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
GUEST_DIR="${GUEST_DIR:-$REPO_ROOT/vm/guest}"
SNAPSHOTS_DIR="${SNAPSHOTS_DIR:-$GUEST_DIR/snapshots}"
CURRENT="$SNAPSHOTS_DIR/current"

die() { echo "vmapple-snapshot: $*" >&2; exit 1; }
cur_label() { readlink "$CURRENT" 2>/dev/null || echo ""; }

cmd_list() {
  [ -d "$SNAPSHOTS_DIR" ] || die "no snapshots dir at $SNAPSHOTS_DIR"
  local c; c="$(cur_label)"
  for d in "$SNAPSHOTS_DIR"/*/; do
    [ -d "$d" ] || continue
    local name; name="$(basename "$d")"
    [ "$name" = "current" ] && continue          # skip the `current` symlink
    local mark="  "; [ "$name" = "$c" ] && mark="* "
    printf '%s%s\n' "$mark" "$name"
  done
}

cmd_current() { local c; c="$(cur_label)"; [ -n "$c" ] && echo "$c" || die "no current snapshot"; }

cmd_rollback() {
  local label="${1:-}"; [ -n "$label" ] || die "usage: rollback <label>"
  [ -d "$SNAPSHOTS_DIR/$label" ] || die "no such snapshot: $label (see: list)"
  ln -sfn "$label" "$CURRENT"
  echo "vmapple-snapshot: current -> $label"
}

cmd_create() {
  [ -f "$GUEST_DIR/disk.img" ] && [ -f "$GUEST_DIR/aux.img.trimmed" ] \
    || die "no at-rest bundle at $GUEST_DIR (disk.img + aux.img.trimmed)"
  if pgrep -f 'qemu-system-aarch64.*vmapple' >/dev/null 2>&1; then
    die "guest is running — shut it down first (scripts/vmapple-shutdown) for a clean snapshot"
  fi
  local label="${1:-$(date +%Y-%m-%d-%H%M%S)-manual}"
  local dir="$SNAPSHOTS_DIR/$label"
  [ -e "$dir" ] && die "snapshot already exists: $label"
  mkdir -p "$dir"
  cp -c "$GUEST_DIR/disk.img" "$dir/disk.img" 2>/dev/null || cp "$GUEST_DIR/disk.img" "$dir/disk.img"
  cp -c "$GUEST_DIR/aux.img.trimmed" "$dir/aux.img.trimmed" 2>/dev/null || cp "$GUEST_DIR/aux.img.trimmed" "$dir/aux.img.trimmed"
  chmod 444 "$dir/disk.img" "$dir/aux.img.trimmed"
  ln -sfn "$label" "$CURRENT"
  echo "vmapple-snapshot: created + current -> $label"
}

case "${1:-list}" in
  list)     cmd_list ;;
  current)  cmd_current ;;
  rollback) shift; cmd_rollback "$@" ;;
  create)   shift; cmd_create "$@" ;;
  -h|--help) sed -n '2,20p' "$0" ;;
  *) die "unknown command: ${1:-} (list | current | rollback <label> | create [label])" ;;
esac
