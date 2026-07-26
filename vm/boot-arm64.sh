#!/usr/bin/env bash
#
# vm/boot-arm64.sh — boot an arm64 macOS guest under QEMU's vmapple machine (HVF) on the Mac.
#
# Display is selected by --device (machine property gfx-device):
#   apple-gfx-mmio     Apple ParavirtualizedGraphics.framework (reference, default)
#   reims-vgpu-mmio    product thin C → crates/reims-vgpu Rust staticlib
# The vmapple machine creates exactly one device at the fixed Reims vGPU GFX/IOSFC
# addresses — do not add a second display via -device.
#
# Snapshot revert: snapshots form an
# IMMUTABLE HISTORY under `vm/guest/snapshots/<label>/{disk.img,aux.img.trimmed}`
# (each read-only, never overwritten). `vm/guest/snapshots/current` is a symlink
# naming the active one. EVERY boot starts from a byte-identical APFS clone of
# `current` (clonefile: instant, COW) and discards that clone on exit, so a harsh
# kill or a wedge costs nothing and poisons nothing. A snapshot is never booted
# directly.
#
# Boot classes:
#   --testing      agent-driven measurement (default): GUI + serial-to-file,
#                  SSH-driven, 7-minute hard kill + capture-then-revert. Reverts.
#   --interactive  human/GUI boot, no time limit. Reverts (nothing persists).
#   --snapshot     boot writable to CAPTURE A NEW snapshot: on a clean guest
#                  shutdown the modified disk/aux are saved as a NEW immutable
#                  snapshot and `current` is repointed to it. Existing snapshots
#                  (incl. the base) are never touched. Roll back by repointing
#                  `current` (see scripts/vmapple-snapshot).
#
# Launch configuration is CLI flags / env here, not device/backend code.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# --- Configuration (override via env or flags) ----------------------------------
# Guest bundle provisioned by scripts/vmapple-provision (large + private, gitignored).
GUEST_DIR="${GUEST_DIR:-$REPO_ROOT/vm/guest}"
# Immutable snapshot history; `current` symlinks the active snapshot to revert to.
SNAPSHOTS_DIR="${SNAPSHOTS_DIR:-$GUEST_DIR/snapshots}"
# Per-boot scratch (clones + logs). Same APFS volume as GUEST_DIR for clonefile.
RUN_DIR="${RUN_DIR:-$GUEST_DIR/run}"

QEMU_BIN_DEFAULT="$REPO_ROOT/vendor/qemu/build/qemu-system-aarch64"
QEMU_BIN="${QEMU_BIN:-$QEMU_BIN_DEFAULT}"
REIMS_VGPU_EFI_ROM_SCRIPT="$REPO_ROOT/crates/reims-vgpu-efi/scripts/reims-vgpu-efi-rom/reims-vgpu-efi-rom.sh"
AVPBOOTER="${AVPBOOTER:-/System/Library/Frameworks/Virtualization.framework/Resources/AVPBooter.vmapple2.bin}"

RAM="${RAM:-8G}"
CPUS="${CPUS:-4}"
SSH_PORT="${SSH_PORT:-2222}"
TESTING_TIMEOUT="${TESTING_TIMEOUT:-420}" # 7-minute hard kill for testing boots
# PIN the guest NIC MAC. Without a fixed MAC, QEMU assigns a random one each boot,
# so a reverted snapshot shows macOS a brand-new unconfigured interface → no DHCP
# lease → broken networking + unreachable sshd. A stable MAC keeps the guest's
# saved network service valid across reverts.
GUEST_MAC="${GUEST_MAC:-52:54:00:76:61:70}"

BOOT_CLASS="testing"     # testing | interactive | snapshot
GFX_DEVICE="apple-gfx-mmio"  # apple-gfx-mmio | reims-vgpu-mmio

usage() {
  cat <<EOF
usage: vm/boot-arm64.sh [--device apple-gfx-mmio|reims-vgpu-mmio] [--testing|--interactive|--snapshot]

  --device NAME          Reims vGPU slot backend (default: apple-gfx-mmio)
                         apple-gfx-mmio  Apple PVG framework (reference)
                         reims-vgpu-mmio    product (reims-vgpu Rust path)
  --testing              agent boot (default): GUI, ${TESTING_TIMEOUT}s hard kill, reverts
  --interactive          human/GUI boot, no time limit, reverts
  --snapshot             boot writable; a clean guest shutdown CAPTURES a new snapshot
                         (also bootstraps the first snapshot on a fresh guest)

Every boot reverts to the current snapshot:
  $SNAPSHOTS_DIR/current -> <label>/{disk.img,aux.img.trimmed}
Always builds reims-vgpu-efi and reims-vgpu before boot. In-tree QEMU is rebuilt
unless QEMU_BIN is set to something other than the default path.
Env: GUEST_DIR SNAPSHOTS_DIR RUN_DIR QEMU_BIN AVPBOOTER RAM CPUS SSH_PORT REIMS_VGPU_BACKEND
     (vulkan default for reims-vgpu-mmio; metal default for apple-gfx-mmio)
     TESTING_TIMEOUT QMP_DUMP_TIMEOUT GUEST_MAC
     NET=user (SLIRP, default) | NET=none (no NIC — one-time offline Setup Assistant bootstrap)
     TRACE=1 — control-plane trace rail: the display device's QEMU trace events
     (MMIO order, ring records, map/unmap, IRQs, frames) → \$RUN_DIR/trace-<stamp>.log
     TRACE_PATTERN=glob — override the default display-device trace glob
     TRACE_EVENTS_FILE=path — QEMU trace event list file; overrides TRACE_PATTERN
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --device)
      shift
      GFX_DEVICE="${1:-}"
      case "$GFX_DEVICE" in
        apple-gfx-mmio|reims-vgpu-mmio) ;;
        *)
          echo "boot-arm64.sh: invalid --device '$GFX_DEVICE' (apple-gfx-mmio | reims-vgpu-mmio)" >&2
          exit 64
          ;;
      esac
      shift
      ;;
    --device=*)
      GFX_DEVICE="${1#--device=}"
      case "$GFX_DEVICE" in
        apple-gfx-mmio|reims-vgpu-mmio) ;;
        *)
          echo "boot-arm64.sh: invalid --device '$GFX_DEVICE' (apple-gfx-mmio | reims-vgpu-mmio)" >&2
          exit 64
          ;;
      esac
      shift
      ;;
    --testing) BOOT_CLASS="testing"; shift ;;
    --interactive) BOOT_CLASS="interactive"; shift ;;
    --snapshot) BOOT_CLASS="snapshot"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "boot-arm64.sh: unknown arg: $1" >&2; usage >&2; exit 64 ;;
  esac
done

# --- Preflight ------------------------------------------------------------------
die() { echo "boot-arm64.sh: $*" >&2; exit 1; }

ensure_rust_tools() {
  if ! command -v cargo >/dev/null 2>&1 && [ -x "$HOME/.cargo/bin/cargo" ]; then
    export PATH="$HOME/.cargo/bin:$PATH"
  fi
  command -v cargo >/dev/null 2>&1 || die "cargo not found (needed to build reims-vgpu)"
}

build_reims_vgpu_efi() {
  [ -x "$REIMS_VGPU_EFI_ROM_SCRIPT" ] || die "EFI ROM builder not executable: $REIMS_VGPU_EFI_ROM_SCRIPT"
  echo "boot-arm64.sh: building reims-vgpu-efi option ROM ..."
  "$REIMS_VGPU_EFI_ROM_SCRIPT" || die "reims-vgpu-efi build failed"
}

build_reims_vgpu_standalone() {
  local backend="$1"
  case "$backend" in
    metal)
      echo "boot-arm64.sh: building reims-vgpu crate (backend-metal) ..."
      (cd "" && cargo build --release -p reims-vgpu --features backend-metal) \
        || die "reims-vgpu build failed"
      ;;
    vulkan)
      echo "boot-arm64.sh: building reims-vgpu crate (backend-vulkan,host-window) ..."
      (cd "" && cargo build --release -p reims-vgpu \
        --no-default-features --features backend-vulkan,host-window) \
        || die "reims-vgpu build failed"
      ;;
    *) die "unknown REIMS_VGPU_BACKEND: $backend (metal | vulkan)" ;;
  esac
}

ensure_rust_tools
build_reims_vgpu_efi
if [ -z "${REIMS_VGPU_BACKEND:-}" ]; then
  case "$GFX_DEVICE" in
    reims-vgpu-mmio) REIMS_VGPU_BACKEND=vulkan ;;
    *) REIMS_VGPU_BACKEND=metal ;;
  esac
fi
if [ "$QEMU_BIN" = "$QEMU_BIN_DEFAULT" ]; then
  echo "boot-arm64.sh: building in-tree QEMU (scripts/qemu-build --target aarch64 --backend $REIMS_VGPU_BACKEND) ..."
  "$REPO_ROOT/scripts/qemu-build/qemu-build.sh" --target aarch64 --backend "$REIMS_VGPU_BACKEND" \
    || die "qemu-build failed"
else
  build_reims_vgpu_standalone "$REIMS_VGPU_BACKEND"
fi

[ -x "$QEMU_BIN" ] || die "QEMU not available: $QEMU_BIN"
[ -f "$AVPBOOTER" ] || die "AVPBooter ROM not found: $AVPBOOTER"
[ -f "$GUEST_DIR/vm.json" ] || die "guest vm.json not found: $GUEST_DIR/vm.json (provision first)"

# Snapshot state. When none exists yet, only --snapshot can bootstrap it: it
# boots the freshly provisioned disk WRITE-THROUGH so you can finish Setup
# Assistant + config, and a clean guest shutdown captures the first immutable
# snapshot. --testing/--interactive need a snapshot to revert to.
CURRENT="$SNAPSHOTS_DIR/current"
HAVE_SNAPSHOT=0
if [ -e "$CURRENT" ] && [ -f "$CURRENT/disk.img" ] && [ -f "$CURRENT/aux.img.trimmed" ]; then
  HAVE_SNAPSHOT=1
fi
if [ "$HAVE_SNAPSHOT" -eq 0 ]; then
  [ "$BOOT_CLASS" = "snapshot" ] || die \
    "no snapshot yet — bootstrap it with:  vm/boot-arm64.sh --snapshot
(boots the provisioned disk writable for Setup Assistant + config; a clean guest shutdown then
captures the first immutable snapshot. --testing/--interactive need a snapshot to revert to.)"
  [ -f "$GUEST_DIR/disk.img" ] && [ -f "$GUEST_DIR/aux.img.trimmed" ] \
    || die "no provisioned bundle at $GUEST_DIR (run scripts/vmapple-provision first)"
fi

# ECID/UUID: vmapple's uuid= is the ECID from the bundle's machineId (== macosvm
# contrib/vmapple/uuid.sh). Extract it from vm.json.
UUID="$(plutil -extract machineId raw "$GUEST_DIR/vm.json" | base64 -d | plutil -extract ECID raw -)"
[ -n "$UUID" ] || die "could not extract ECID/UUID from $GUEST_DIR/vm.json"

# --- Choose the boot disk: revert-clone, or bootstrap write-through -------------
mkdir -p "$RUN_DIR"
STAMP="$(date +%Y%m%d-%H%M%S)"
SERIAL_LOG="$RUN_DIR/serial-$STAMP.log"
QMP_SOCK="$RUN_DIR/qmp-$STAMP.sock"
# Stable alias to the live boot's QMP socket (scripts/qmp): QMP_SOCK=vm/guest/run/qmp.sock
ln -sfn "qmp-$STAMP.sock" "$RUN_DIR/qmp.sock"

# --- Control-plane trace rail ---------------------------------------------------
# TRACE=1 enables the display device's QEMU trace events — protocol records ONLY
# (MMIO access order, ring/FIFO records, map/unmap requests, IRQ raises, frames,
# mode changes), never referenced guest-memory content. This is the sanctioned
# boot-script switch for tracing.
TRACE="${TRACE:-0}"
TRACE_LOG=""
TRACE_SPEC=""
if [ "$TRACE" = "1" ]; then
  TRACE_LOG="$RUN_DIR/trace-$STAMP.log"
  if [ -n "${TRACE_EVENTS_FILE:-}" ]; then
    [ -f "$TRACE_EVENTS_FILE" ] || die "TRACE_EVENTS_FILE not found: $TRACE_EVENTS_FILE"
    TRACE_SPEC="events=$TRACE_EVENTS_FILE"
  else
    # TRACE_PATTERN may be overridden in env; default matches the selected device.
    if [ -z "${TRACE_PATTERN:-}" ]; then
      case "$GFX_DEVICE" in
        reims-vgpu-mmio) TRACE_PATTERN="reims_vgpu_mmio_*" ;;
        *)            TRACE_PATTERN="apple_gfx_*" ;;
      esac
    fi
    TRACE_SPEC="$TRACE_PATTERN"
  fi
fi

if [ "$HAVE_SNAPSHOT" -eq 0 ]; then
  # Bootstrap (--snapshot only): boot the provisioned master write-through so
  # Setup Assistant + config persist; a clean shutdown captures snapshot #1.
  DISK="$GUEST_DIR/disk.img"; AUX="$GUEST_DIR/aux.img.trimmed"; IS_CLONE=0
  echo "boot-arm64.sh: bootstrap — booting provisioned disk write-through (no snapshot yet) ..."
else
  # Revert: clone the current snapshot into a throwaway working copy.
  DISK="$RUN_DIR/disk-$STAMP.img"; AUX="$RUN_DIR/aux-$STAMP.img"; IS_CLONE=1
  echo "boot-arm64.sh: reverting to snapshot '$(readlink "$CURRENT" 2>/dev/null || echo current)' ..."
  cp -c "$CURRENT/disk.img" "$DISK" 2>/dev/null || cp "$CURRENT/disk.img" "$DISK"
  cp -c "$CURRENT/aux.img.trimmed" "$AUX" 2>/dev/null || cp "$CURRENT/aux.img.trimmed" "$AUX"
  chmod u+w "$DISK" "$AUX"   # snapshots are read-only; the working clone must be writable
fi

# --- Network -------------------------------------------------------------------
# NET=user (default): QEMU SLIRP user-mode NAT — no privileges, real outbound
#   TCP/UDP+DNS (verified reaching Apple over HTTPS), SSH via hostfwd. ipv6=off
#   per QEMU's own vmapple.rst reference invocation: SLIRP's fec0::/64 RA
#   otherwise gives the guest a phantom IPv6 default that macOS prefers and that
#   goes nowhere, so outbound traffic (DNS first) stalls before falling back to v4.
# NET=none: no NIC at all (passes -nic none). Note QEMU adds a DEFAULT
#   user-mode NIC when no -netdev/-nic is given, so genuinely disabling the
#   network requires an explicit -nic none — omitting the netdev is not enough.
#   Rarely needed: Setup Assistant completes fine online (its Device-Enrollment
#   pane stall is non-deterministic and clears on its own).
NET="${NET:-user}"
case "$NET" in
  user) NETDEV="user,id=net0,ipv6=off,hostfwd=tcp::${SSH_PORT}-:22" ;;
  none) NETDEV="" ;;
  *) die "unknown NET: $NET (user | none)" ;;
esac

# --- Build the QEMU command line ------------------------------------------------
# Per docs/system/arm/vmapple.rst: aux + disk each as pflash (pre-boot env) AND as
# virtio-blk (runtime), plus the AVPBooter ROM as -bios and -M vmapple,uuid=ECID.
QEMU_ARGS=(
  -m "$RAM"
  -accel hvf
  -smp "$CPUS"
  -M "vmapple,uuid=$UUID,gfx-device=$GFX_DEVICE"
  -bios "$AVPBOOTER"
  -drive "file=$AUX,if=pflash,format=raw"
  -drive "file=$DISK,if=pflash,format=raw"
  -drive "file=$AUX,if=none,id=aux,format=raw"
  -drive "file=$DISK,if=none,id=root,format=raw"
  -device vmapple-virtio-blk-pci,variant=aux,drive=aux
  -device vmapple-virtio-blk-pci,variant=root,drive=root
  -qmp "unix:$QMP_SOCK,server=on,wait=off"
)
if [ -n "$TRACE_LOG" ]; then
  QEMU_ARGS+=(-trace "$TRACE_SPEC" -D "$TRACE_LOG")
fi
if [ -n "$NETDEV" ]; then
  QEMU_ARGS+=(-netdev "$NETDEV" -device "virtio-net-pci,netdev=net0,mac=$GUEST_MAC")
else
  QEMU_ARGS+=(-nic none)   # suppress QEMU's implicit default user-mode NIC
fi

# The Vulkan product build owns its AppKit window in Rust and therefore disables
# QEMU's Cocoa display. The Apple reference and Metal-direct builds retain Cocoa.
# The build stamp is authoritative configure-time state, not an env-gated device
# path; fail closed rather than accidentally run two competing display windows.
DISPLAY_KIND="cocoa"
if [ "$GFX_DEVICE" = "reims-vgpu-mmio" ]; then
  BACKEND_STAMP="$(dirname "$QEMU_BIN")/reims-vgpu-backend.stamp"
  [ -f "$BACKEND_STAMP" ] || die \
    "missing backend stamp: $BACKEND_STAMP (rebuild with scripts/qemu-build/qemu-build.sh)"
  case "$(cat "$BACKEND_STAMP")" in
    vulkan)
      DISPLAY_KIND="reims-host-window"
      VULKAN_LOADER_DIR="/opt/homebrew/opt/vulkan-loader/lib"
      MOLTENVK_ICD="/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json"
      [ -d "$VULKAN_LOADER_DIR" ] || die \
        "Vulkan loader not found: $VULKAN_LOADER_DIR (install Homebrew vulkan-loader)"
      [ -f "$MOLTENVK_ICD" ] || die \
        "MoltenVK ICD not found: $MOLTENVK_ICD (install Homebrew molten-vk)"
      export DYLD_FALLBACK_LIBRARY_PATH="$VULKAN_LOADER_DIR${DYLD_FALLBACK_LIBRARY_PATH:+:$DYLD_FALLBACK_LIBRARY_PATH}"
      export VK_ICD_FILENAMES="$MOLTENVK_ICD"
      ;;
    metal) DISPLAY_KIND="cocoa" ;;
    *) die "invalid backend stamp: $BACKEND_STAMP" ;;
  esac
fi

echo "boot-arm64.sh: device=$GFX_DEVICE class=$BOOT_CLASS uuid=$UUID"
echo "boot-arm64.sh: display=$DISPLAY_KIND"
echo "boot-arm64.sh: ssh → localhost:$SSH_PORT   serial → $SERIAL_LOG   qmp → $QMP_SOCK"
[ -n "$TRACE_LOG" ] && echo "boot-arm64.sh: trace → $TRACE_LOG ($TRACE_SPEC)"

# Discard the per-boot working clone. Never deletes the provisioned master (used
# write-through during bootstrap), only a RUN_DIR clone.
discard_clone() { [ "${IS_CLONE:-1}" -eq 1 ] && rm -f "$DISK" "$AUX"; rm -f "$QMP_SOCK" "$RUN_DIR/qmp.sock"; }

promote_to_snapshot() {
  # Save this boot's (modified) disk/aux as a NEW immutable snapshot and repoint
  # `current` to it. Existing snapshots (incl. the base) are never overwritten.
  # Called only after a clean guest shutdown in --snapshot mode.
  local label new_dir
  if [ "$HAVE_SNAPSHOT" -eq 0 ]; then label="$(date +%Y-%m-%d-%H%M%S)-base"; else label="$(date +%Y-%m-%d-%H%M%S)-snap"; fi
  new_dir="$SNAPSHOTS_DIR/$label"
  echo "boot-arm64.sh: capturing new immutable snapshot '$label' ..."
  mkdir -p "$new_dir"
  cp -c "$DISK" "$new_dir/disk.img" 2>/dev/null || cp "$DISK" "$new_dir/disk.img"
  cp -c "$AUX" "$new_dir/aux.img.trimmed" 2>/dev/null || cp "$AUX" "$new_dir/aux.img.trimmed"
  chmod 444 "$new_dir/disk.img" "$new_dir/aux.img.trimmed"
  ln -sfn "$label" "$CURRENT"
  discard_clone
  echo "boot-arm64.sh: snapshot '$label' captured; current -> $label"
}

# --- Interactive / snapshot: foreground GUI, no time limit ----------------------
if [ "$BOOT_CLASS" = "interactive" ] || [ "$BOOT_CLASS" = "snapshot" ]; then
  if [ "$DISPLAY_KIND" = "reims-host-window" ]; then
    QEMU_ARGS+=(-display none -serial mon:stdio)
  else
    QEMU_ARGS+=(-display cocoa -serial mon:stdio)
  fi
  rc=0
  "$QEMU_BIN" "${QEMU_ARGS[@]}" || rc=$?
  if [ "$BOOT_CLASS" = "snapshot" ] && [ "$rc" -eq 0 ]; then
    promote_to_snapshot
  else
    [ "$BOOT_CLASS" = "snapshot" ] && echo "boot-arm64.sh: qemu exited rc=$rc (not clean) — snapshot NOT updated"
    discard_clone
  fi
  exit "$rc"
fi

# --- Testing: background GUI + hard kill + capture-then-revert -------------------
if [ "$DISPLAY_KIND" = "reims-host-window" ]; then
  QEMU_ARGS+=(-display none -serial "file:$SERIAL_LOG")
else
  QEMU_ARGS+=(-display cocoa -serial "file:$SERIAL_LOG")
fi

# Best-effort QMP register dump. Must never block hard-kill more than
# QMP_DUMP_TIMEOUT seconds (default 3). Unbounded `nc -U` can hang testing
# boots after the timer (kill was unreachable behind a wedged QMP).
QMP_DUMP_TIMEOUT="${QMP_DUMP_TIMEOUT:-3}"

qmp_dump_registers() {
  local out="$RUN_DIR/registers-$STAMP.txt"
  local watchdog_pid="" nc_pid=""
  if [ ! -S "$QMP_SOCK" ] || ! command -v nc >/dev/null 2>&1; then
    return 0
  fi
  if command -v timeout >/dev/null 2>&1; then
    timeout --signal=KILL "${QMP_DUMP_TIMEOUT}s" sh -c "
      {
        printf '%s\\n' '{\"execute\":\"qmp_capabilities\"}'
        printf '%s\\n' '{\"execute\":\"human-monitor-command\",\"arguments\":{\"command-line\":\"info registers -a\"}}'
        sleep 0.3
      } | nc -U \"\$1\"
    " sh "$QMP_SOCK" >"$out" 2>/dev/null || true
    return 0
  fi
  # Portable fallback (macOS / no timeout): background nc + watchdog kill.
  {
    printf '{"execute":"qmp_capabilities"}\n'
    printf '{"execute":"human-monitor-command","arguments":{"command-line":"info registers -a"}}\n'
    sleep 0.3
  } | nc -U "$QMP_SOCK" >"$out" 2>/dev/null &
  nc_pid=$!
  (
    sleep "$QMP_DUMP_TIMEOUT"
    kill -9 "$nc_pid" 2>/dev/null || true
  ) &
  watchdog_pid=$!
  wait "$nc_pid" 2>/dev/null || true
  kill "$watchdog_pid" 2>/dev/null || true
  wait "$watchdog_pid" 2>/dev/null || true
}

kill_qemu() {
  if [ -z "${QEMU_PID:-}" ]; then
    return 0
  fi
  if ! kill -0 "$QEMU_PID" 2>/dev/null; then
    return 0
  fi
  echo "boot-arm64.sh: killing qemu pid=$QEMU_PID"
  kill -TERM "$QEMU_PID" 2>/dev/null || true
  sleep 2
  if kill -0 "$QEMU_PID" 2>/dev/null; then
    kill -KILL "$QEMU_PID" 2>/dev/null || true
  fi
  wait "$QEMU_PID" 2>/dev/null || true
}

capture_then_revert() {
  local reason="$1"
  echo "boot-arm64.sh: capture-then-revert ($reason)"
  # Dump first (bounded), then always kill — never gate kill on QMP success.
  qmp_dump_registers
  kill_qemu
  discard_clone
  echo "boot-arm64.sh: reverted (clone discarded); evidence in $RUN_DIR (serial-$STAMP.log)"
}

"$QEMU_BIN" "${QEMU_ARGS[@]}" &
QEMU_PID=$!
trap 'capture_then_revert signal; exit 130' INT TERM

elapsed=0
while kill -0 "$QEMU_PID" 2>/dev/null; do
  if [ "$elapsed" -ge "$TESTING_TIMEOUT" ]; then
    capture_then_revert "timeout ${TESTING_TIMEOUT}s — wedge verdict"
    exit 124
  fi
  sleep 5
  elapsed=$((elapsed + 5))
done

wait "$QEMU_PID" 2>/dev/null || true
capture_then_revert "qemu exited"
