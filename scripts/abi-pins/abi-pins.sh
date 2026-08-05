#!/usr/bin/env bash
# abi-pins — find shared-ABI constants a C shim reads that no Rust test pins.
#
#   scripts/abi-pins/abi-pins.sh
#
# AGENTS.md: "Anything crossing the boundary lives twice ... and nothing in the
# toolchain compares the two: Rust does not include the header and the shims do
# not read Rust. Every constant that crosses gets a test, using
# `qemu::abi::header_define`." This is what checks that rule is still kept.
#
# Exits non-zero when something a shim reads is unpinned, so it can gate a
# commit. See README.md for the two ways it can mislead.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

python3 - "$here" <<'PY'
import re, sys, glob, pathlib

root = pathlib.Path(sys.argv[1])
header = root / "crates/reims-vgpu/include/reims_vgpu_qemu_abi.h"
if not header.exists():
    sys.exit(f"[abi-pins] missing shared header: {header}")

defs = set(re.findall(r'^#define (REIMS_VGPU_[A-Z0-9_]+)', header.read_text(), re.M))
# The include guard is a name, not a value that can drift.
defs.discard("REIMS_VGPU_QEMU_ABI_H")

# A define is "pinned" when some Rust test names it as a string literal. Match
# the literal rather than `header_define(` — several tests alias the import
# (`use ...::header_define as define;`) and keying on the call under-reports.
pinned = set()
rust = glob.glob(str(root / "crates/reims-vgpu/src/**/*.rs"), recursive=True)
rust += glob.glob(str(root / "crates/reims-vgpu/tests/*.rs"))
for f in rust:
    pinned |= set(re.findall(r'"(REIMS_VGPU_[A-Z0-9_]+)"',
                             pathlib.Path(f).read_text(errors="ignore")))

shims = glob.glob(str(root / "vendor/qemu/hw/display/reims-vgpu*.c"))
shims += glob.glob(str(root / "vendor/qemu/hw/display/reims-vgpu*.h"))
if not shims:
    sys.exit("[abi-pins] no shim sources found — is the qemu submodule checked out?")

read_by_c = set()
for f in shims:
    txt = pathlib.Path(f).read_text(errors="ignore")
    for d in defs:
        if re.search(r'\b' + d + r'\b', txt):
            read_by_c.add(d)

gap = sorted(read_by_c - pinned)
unused = sorted(defs - read_by_c - pinned)

print(f"[abi-pins] {len(defs)} defines, {len(read_by_c)} read by a shim, "
      f"{len(pinned & defs)} pinned by a test")

if unused:
    print("\n[abi-pins] defined, no shim reads it, no test pins it "
          "(not a failure — a constant the C side has not needed yet):")
    for d in unused:
        print(f"  {d}")

if not gap:
    print("\n[abi-pins] every constant a shim reads is pinned.")
    sys.exit(0)

print(f"\n[abi-pins] READ BY A SHIM, PINNED BY NOTHING ({len(gap)}):")
for d in gap:
    print(f"  {d}")
print("\n[abi-pins] Each of these exists twice with nothing comparing the two.")
print("[abi-pins] Add an assertion using `qemu::abi::header_define` (or")
print("[abi-pins] `header_define_i32` for a signed one) beside the Rust value.")
sys.exit(1)
PY
