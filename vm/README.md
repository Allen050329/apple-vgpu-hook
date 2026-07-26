# vm/ — macOS guests for the reims-vgpu pathways

`crates/reims-vgpu` runs three backend pathways over two guest rails. Pick the boot script for the
guest you are on and the QEMU/reims-vgpu backend for the host GPU path.

| Pathway | Script | Host accel | Backend | Typical device |
|---|---|---|---|---|
| x86 macOS / Linux Vulkan | `vm/boot-x86.sh` | KVM / OpenCore+OVMF | Vulkan via `metal2vulkan` | `reims-vgpu-pci` |
| arm64 macOS / macOS Metal | `vm/boot-arm64.sh` | HVF / `vmapple` | Metal-direct | `reims-vgpu-mmio` (product) or `apple-gfx-mmio` (Apple reference A/B) |
| arm64 macOS / macOS Vulkan | `vm/boot-arm64.sh` | HVF / `vmapple` | Vulkan via `metal2vulkan` through MoltenVK | `reims-vgpu-mmio` |

```bash
# arm64 macOS guest on Mac host (Metal or Vulkan/MoltenVK backend)
vm/boot-arm64.sh --testing                 # agent boot: GUI + serial-to-file, hard kill, reverts
vm/boot-arm64.sh --interactive             # human boot: GUI, no time limit, reverts
vm/boot-arm64.sh --device reims-vgpu-mmio --testing
vm/boot-arm64.sh --device apple-gfx-mmio --testing   # Apple-paravirt reference (arm only)

# x86 macOS guest on Linux host
vm/boot-x86.sh --testing
vm/boot-x86.sh --device reims-vgpu-pci --testing
```

Both scripts use a snapshot-revert lifecycle (testing vs interactive classes; testing hard kill).
QMP is a per-boot unix socket under the run dir for that pathway (`scripts/qmp/qmp.py`).

**Networking** is typically QEMU SLIRP with SSH hostfwd. Guest disks, IPSWs, OpenCore blobs, and
runtime clones are **gitignored** - private and large; never commit them.

## Layout (gitignored runtime)

Pathway-specific trees hold provisioned disks, snapshots, and per-boot clones. See
the boot and provisioning scripts for where golden images and SSH setup live.

Stop a wedged guest with the pathway's shutdown helper when available
(`scripts/vmapple-shutdown` on arm); otherwise use QMP quit plus a process kill.
