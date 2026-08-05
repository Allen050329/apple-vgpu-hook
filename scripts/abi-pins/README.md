# abi-pins

Which shared-ABI constants a C shim reads that no Rust test pins.

```sh
scripts/abi-pins/abi-pins.sh     # exits non-zero if anything is unpinned
```

## Why a script

`AGENTS.md` states the rule this checks:

> Anything crossing the boundary lives twice, once in Rust and once in
> `crates/reims-vgpu/include/reims_vgpu_qemu_abi.h`, and nothing in the toolchain
> compares the two: Rust does not include the header and the shims do not read
> Rust. Every constant that crosses gets a test.

The rule was written and then not kept. When this script was added, 22 of the 41
constants a shim reads were pinned by nothing — including
`REIMS_VGPU_HOST_ACTION_*`, the discriminant both shims switch on to pick an
action handler, and `REIMS_VGPU_QEMU_OK`, the value both drain loops compare
against. Nothing noticed, because nothing was looking: a rule with no instrument
degrades silently as the header grows.

**Tables are the worst case and the reason this counts entries rather than
files.** A lone constant that drifts usually breaks something loudly. One entry
of an eleven-entry action table drifts past the ten around it that still agree,
and the symptom is one action kind running the wrong handler.

## Why not a test

It would have to read the shim sources, which live in the `vendor/qemu`
submodule. A `#[test]` that fails when a submodule is not checked out reports a
missing checkout as an ABI defect, so the check lives here and the *assertions it
asks for* are the tests.

## Reading the output

Three sections, and only the third is a failure:

- **the counts** — defines, how many a shim reads, how many are pinned.
- **defined, no shim reads it, no test pins it** — not a failure. A constant the
  C side has not needed yet cannot drift against a shim that never names it. It
  becomes a finding the moment a shim uses it, which is when this script starts
  reporting it in the third section.
- **READ BY A SHIM, PINNED BY NOTHING** — the finding. Add an assertion beside
  the Rust value using `qemu::abi::header_define`, or `header_define_i32` for a
  signed one (the dma-buf refusal codes and the entry-point return codes are
  negative or `c_int`).

Put the assertion next to the Rust definition, not in one central test: the
existing pins live beside `ConsoleFeed`, `HostActionKind`, `ReimsVgpuButton` and
the register map, which is where someone changing a value will see them.

## Two ways it can mislead

**It matches string literals, not calls.** A define counts as pinned when any
Rust file under `src/` or `tests/` contains its name as a `"..."` literal. That
is deliberate — several tests alias the import (`use ...::header_define as
define;`) and keying on `header_define(` under-reported by 13 — but it means a
constant merely *mentioned* in a string counts as pinned. If you add its name to
a log message rather than an assertion, this script will believe you.

**It matches whole words in the shim sources, including comments.** A define
named only in a shim comment reads as "used by C". That errs toward asking for a
pin that may not be needed, which is the safe direction.

Neither can produce a false *pass* for a constant that is genuinely used and
genuinely unchecked, which is the failure this exists to prevent.
