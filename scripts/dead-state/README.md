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

## What it is strong and weak at

**Items — functions, methods, constants, enum variants, type aliases — are
reliable.** Nothing masks them.

**Fields are one-directional evidence.** A derived impl that is itself used
counts as a read of every field it touches. `#[derive(Debug)]` is neutralised
in the scratch tree (stub impls replace it) because `DeviceState` derives it
and the fail log formats it, which alone kept the field report permanently
empty. `Clone`, `PartialEq` and `Hash` cannot be neutralised the same way —
removing them does not compile — so a field on a struct whose `clone()` is
actually called will not be reported.

So a field hit is strong evidence; an empty field report is not proof. The
worked example: `PresentState::mapping_id`, written at five sites and read at
none, was found by hand and never by this script, because `PresentState`
derives `Clone` and `DeviceState` is cloned.
`observe::gate::no_present_state_field_is_write_only` covers that one struct
directly and has no such blind spot; extend that pattern to any other struct
where this matters.

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

## Both arms, intersected

An item can be live on one backend and cfg'd out of the other. Every argument
struct `execute_dispatch_metal` builds reads as dead from the Vulkan arm, and
there are seventeen of those — enough to bury the real findings.

So the script compiles both `backend-vulkan,host-window` and `backend-metal`
(the latter cross-checked at `aarch64-apple-darwin`, which needs no Apple SDK
and no Apple host) and reports only the intersection. It prints how many hits
each arm suppressed, so a large asymmetry is visible rather than silent.
