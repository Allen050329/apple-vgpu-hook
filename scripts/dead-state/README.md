# dead-state

Reports `reims-vgpu` items and struct fields that no code path reads.

```sh
scripts/dead-state/dead-state.sh              # items and fields
scripts/dead-state/dead-state.sh --fields-only
```

## Why a script and not a grep

Rust already has the right tool and this crate has it switched off.
`reims-vgpu` is a staticlib whose types are almost all `pub`, and a `pub` item
in a library is reachable from outside the crate by definition, so rustc's
dead-code pass never calls one dead.

So every sweep for unread state here used to be a grep, and greps get this
wrong in both directions:

- **False negatives.** Field names are ordinary words. `.mapping_id` matches
  `MappingEntry`, `ComputeStorageResidencyKey` and a dozen other types, so a
  genuinely-unread field looks read. This exact miss let
  `PresentState::mapping_id` sit unread through more than one sweep.
- **False positives.** A grep over `src/` alone calls seven engine hooks dead
  that `tests/vk_engine_*.rs` uses.

The script copies the crate to a scratch tree, rewrites every leading `pub ` to
`pub(crate) `, and compiles that with `--all-targets`. Nothing outside can
reach a scratch crate nobody depends on, so rustc reports exactly what no code
path reads — and it does so with full knowledge of trait impls, macros, cfg
arms and generics. The working tree is never modified.

## Triage

It is a report, not a gate. Three classes are legitimately unread from Rust and
should stay:

- **Contract tables** — register maps (`model/regs.rs`), SDK enum mirrors
  (`backend/metal/raw_metal.rs`), wire field offsets (`runtime/decode/`). A
  hole in a documented register map costs more than the line saves.
- **Error and decline variants** only a future decode path constructs.
- **The C ABI surface** in `qemu/abi.rs`. QEMU calls it; Rust cannot see that.

Everything else is what this is for. A struct field written at five sites and
read at none is not documentation — it is state that every author who reads the
struct has to work out whether some other rail is expected to keep consistent.

## Known limitation

Only the `backend-vulkan,host-window` arm is compiled, because that is the arm
this host can build. An item live only on `backend-metal` will be reported as
dead here. Check the Metal arm before deleting anything under
`backend/metal/`:

```sh
cargo check -p reims-vgpu --target aarch64-apple-darwin --all-targets --features backend-metal
```
