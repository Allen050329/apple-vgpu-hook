#!/usr/bin/env bash
#
# runtime-dead.sh — which reims-vgpu functions never executed on a real boot.
#
# `scripts/dead-state` answers "what does nothing reference". This answers the
# other question: what compiles, links, is reachable, and the guest protocol
# still never takes. Those are different sets, and only the second one needs a
# guest to measure.
#
# Method: build the staticlib with -C instrument-coverage, link the LLVM profile
# runtime into QEMU (see hw/display/meson.build), boot the x86 guest, drive it
# so the measurement is of a busy device rather than an idle one, then stop QEMU
# with SIGTERM so the atexit writer runs. SIGKILL loses everything — continuous
# mode (%c) would survive it but needs runtime counter relocation, which this
# toolchain does not build.
#
# READ THE README BEFORE DELETING ANYTHING. A zero here is not a verdict.
#
# Usage: scripts/runtime-dead/runtime-dead.sh [--seconds N] [--app NAME]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUT_DIR="${OUT_DIR:-/tmp/reims-vgpu-runtime-dead}"
DRIVE_SECONDS=25
DRIVE_APP=Safari

while [ $# -gt 0 ]; do
    case "$1" in
        --seconds) DRIVE_SECONDS="$2"; shift 2 ;;
        --app) DRIVE_APP="$2"; shift 2 ;;
        *) echo "runtime-dead: unknown argument '$1'" >&2; exit 2 ;;
    esac
done

# The profile runtime is compiler-rt's, not rustup's: rustup ships
# profiler_builtins as an rlib for linking into Rust artifacts, and what QEMU
# needs is the plain archive. Its LLVM major must match rustc's or the .profraw
# it writes will not parse.
rustc_llvm="$(rustc --version --verbose | sed -n 's/^LLVM version: \([0-9]*\).*/\1/p')"
PROFILE_RT=""
for cand in /usr/lib/clang/"$rustc_llvm"/lib/linux/libclang_rt.profile-x86_64.a \
            /usr/lib/clang/*/lib/linux/libclang_rt.profile-x86_64.a; do
    [ -f "$cand" ] || continue
    PROFILE_RT="$cand"
    break
done
if [ -z "$PROFILE_RT" ]; then
    echo "runtime-dead: no libclang_rt.profile-x86_64.a found (want LLVM $rustc_llvm)" >&2
    echo "runtime-dead: install compiler-rt for a clang whose major matches rustc's" >&2
    exit 1
fi
rt_llvm="$(printf %s "$PROFILE_RT" | sed -n 's|.*/clang/\([0-9]*\)/.*|\1|p')"
if [ -n "$rt_llvm" ] && [ "$rt_llvm" != "$rustc_llvm" ]; then
    echo "runtime-dead: WARNING profile runtime is LLVM $rt_llvm, rustc is $rustc_llvm" >&2
fi

command -v llvm-profdata >/dev/null || { echo "runtime-dead: need llvm-profdata" >&2; exit 1; }
command -v llvm-cov >/dev/null || { echo "runtime-dead: need llvm-cov" >&2; exit 1; }

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"
QEMU_BIN="$REPO_ROOT/vendor/qemu/build/qemu-system-x86_64"

# Scoped to the host triple on purpose. A bare RUSTFLAGS also reaches the
# x86_64-unknown-uefi option ROM, which has no profiler_builtins for its target
# and fails the whole boot before QEMU starts.
export CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUSTFLAGS="-C instrument-coverage"
export REIMS_VGPU_COVERAGE="$PROFILE_RT"
export LLVM_PROFILE_FILE="$OUT_DIR/reims.profraw"

echo "runtime-dead: profile runtime $PROFILE_RT"
echo "runtime-dead: booting (instrumented) ..."
"$REPO_ROOT/vm/boot-x86.sh" --device reims-vgpu-pci --testing > "$OUT_DIR/boot.log" 2>&1 &

qemu_pid=""
for _ in $(seq 1 180); do
    qemu_pid="$(ps -eo pid,args | grep '[q]emu-system-x86_64' | awk '{print $1}' | head -1)"
    [ -n "$qemu_pid" ] && break
    sleep 2
done
if [ -z "$qemu_pid" ]; then
    echo "runtime-dead: QEMU never started; see $OUT_DIR/boot.log" >&2
    exit 1
fi

echo "runtime-dead: waiting for the guest ..."
guest_up=0
for _ in $(seq 1 120); do
    if ssh -o ConnectTimeout=4 -o BatchMode=yes macos-vm true 2>/dev/null; then
        guest_up=1
        break
    fi
done
if [ "$guest_up" -eq 0 ]; then
    echo "runtime-dead: guest never answered on macos-vm; see $OUT_DIR/boot.log" >&2
    kill -TERM "$qemu_pid" 2>/dev/null || true
    exit 1
fi

# An undriven boot reaches the desktop and sits there, and its zeros are the
# idle device's. The probe refuses a verdict if the window never moved, so a run
# that produced no compositing cannot be mistaken for one that did.
echo "runtime-dead: driving the guest (${DRIVE_SECONDS}s, $DRIVE_APP) ..."
"$REPO_ROOT/scripts/window-drag-probe/window-drag-probe.sh" \
    --seconds "$DRIVE_SECONDS" --app "$DRIVE_APP" > "$OUT_DIR/drive.log" 2>&1 || true
tail -1 "$OUT_DIR/drive.log"

echo "runtime-dead: stopping QEMU (SIGTERM — the profile is written at exit) ..."
kill -TERM "$qemu_pid" 2>/dev/null || true
for _ in $(seq 1 60); do
    ps -p "$qemu_pid" >/dev/null 2>&1 || break
    sleep 2
done

if [ ! -s "$LLVM_PROFILE_FILE" ]; then
    echo "runtime-dead: no profile data — QEMU did not exit cleanly" >&2
    exit 1
fi

echo "runtime-dead: merging ..."
llvm-profdata merge -sparse "$LLVM_PROFILE_FILE" -o "$OUT_DIR/merged.profdata"

# The coverage mapping lives in the linked QEMU binary, not the archive.
mapfile -t sources < <(find "$REPO_ROOT/crates/reims-vgpu/src" -name '*.rs')
llvm-cov report --instr-profile="$OUT_DIR/merged.profdata" "$QEMU_BIN" \
    "${sources[@]}" > "$OUT_DIR/by-file.txt" 2>/dev/null
llvm-cov export --instr-profile="$OUT_DIR/merged.profdata" "$QEMU_BIN" \
    --format=text "${sources[@]}" > "$OUT_DIR/export.json" 2>/dev/null

# Every function whose counter stayed at zero, with the file it lives in.
python3 - "$OUT_DIR/export.json" "$OUT_DIR/never-ran.txt" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
rows = []
for export in data["data"]:
    for fn in export.get("functions", []):
        if fn["count"]:
            continue
        files = ", ".join(sorted({f.split("/crates/reims-vgpu/src/")[-1]
                                  for f in fn["filenames"]}))
        rows.append((files, fn["name"]))
rows.sort()
with open(sys.argv[2], "w") as out:
    for path, name in rows:
        out.write(f"{path}\t{name}\n")
print(f"runtime-dead: {len(rows)} functions never ran")
PY

echo
echo "runtime-dead: per-file coverage  $OUT_DIR/by-file.txt"
echo "runtime-dead: never-ran list     $OUT_DIR/never-ran.txt"
echo "runtime-dead: driven-boot log    $OUT_DIR/drive.log"
echo
echo "A zero is a question, not a verdict. See $SCRIPT_DIR/README.md."
