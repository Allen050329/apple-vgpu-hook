# vmapple-snapshot.sh

Manage the vmapple guest's **immutable snapshot history** under
`vm/guest/snapshots/<label>/{disk.img,aux.img.trimmed}`, with a `current` symlink naming the active
one. Snapshots are APFS clones (instant, COW) and read-only — they are **never overwritten**.
`vm/boot-arm64.sh` reverts to `current` on every boot and captures new snapshots via `--snapshot`; this
tool covers the rest.

```sh
scripts/vmapple-snapshot/vmapple-snapshot.sh list             # list snapshots (* = current)
scripts/vmapple-snapshot/vmapple-snapshot.sh current          # print current label
scripts/vmapple-snapshot/vmapple-snapshot.sh rollback <label> # repoint current (no data touched)
scripts/vmapple-snapshot/vmapple-snapshot.sh create [label]   # clone the at-rest bundle → new snapshot, make current
```

`create` requires the guest to be shut down (clean snapshot). `rollback` just moves the `current`
pointer, so you can jump between snapshots freely — the whole history stays on disk.

Env: `GUEST_DIR`, `SNAPSHOTS_DIR`.
