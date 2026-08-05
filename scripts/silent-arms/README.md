# silent-arms

Reports `match` arms carrying an `if` guard, inside a `match` whose catch-all is
`_ => {}` or `_ => ()`.

```sh
scripts/silent-arms/silent-arms.sh              # both crates' src trees
scripts/silent-arms/silent-arms.sh <dir> ...
```

## What it is looking for

An arm written as

```rust
RenderKind::ExecuteCommands if cmd.indirect_command_buffer_ref != 0 => { … }
…
_ => {}
```

decodes a guest record, tests something the guest *said*, and — when the test
fails — hands it to a catch-all that does nothing and says nothing. The record
was understood well enough to be classified and is then dropped with no line in
`/tmp/reims-vgpu-fail.log`, which is the failure mode `AGENTS.md`'s "Never Fail
Silently" exists to prevent. Nothing else finds this shape: the arm compiles,
the match is exhaustive, no counter moves, and a driven boot looks identical.

It is also invisible to a reader, because the guard and the catch-all are
usually hundreds of lines apart.

## The triage rule

**Most hits are not findings.** The question to ask is what the guard is
testing:

- **A guard on decoder presence is not a finding.** `if cmd.has_cull_mode`,
  `if x.is_some()`, `if word_count >= 3` — these ask whether the decoder saw the
  field at all, or whether the record was long enough to hold it. The arm not
  running means the guest did not send the thing, so there is nothing to lose.
  The script counts these and suppresses them.
- **A guard on a decoded value is a candidate.** `if cmd.scissor_w > 0`,
  `if cmd.indirect_command_buffer_ref != 0` — the guest sent a record and said
  something specific in it. Falling through means the device read the guest's
  statement and discarded it.
- **A candidate is still not automatically a finding.** The catch-all may be a
  real no-op state. Apply the test `AGENTS.md` gives for a never-firing branch,
  inverted: *name what the guest is asking for, and say what happens to it.* If
  the answer is "nothing should happen", the arm is right and wants a comment,
  not a counter.

## Already adjudicated — do not re-report these

Nine candidates on the current tree, all read and all legitimate:

- `runtime/exec.rs` `SetPipeline if cmd.pipeline_ref != 0` — a zero ref is an
  intentional unbind, which `AGENTS.md` names as expected control flow.
- `runtime/metal_draw/mod.rs` ×4 and `metal_draw/vulkan.rs` ×3 — the
  `PASS_LOAD_ACTION_*` ladders. The catch-all is `DONT_CARE`, whose whole
  meaning is "do nothing to this attachment", and the value is bounded upstream
  by `load_action <= PASS_LOAD_ACTION_CLEAR` so it cannot be an unknown action.
- `runtime/spirv_bind.rs:531` — the taint scan's `_ if …any(is_derived)`. The
  catch-all is an instruction that touches no derived value, which is the
  scan's normal case rather than a drop.

Re-check them when the surrounding code changes; do not spend a second session
rediscovering that they are fine.

## Validation

Run against the tree before `f18608a`, which fixed two bugs of exactly this
shape, and the script reports both:

```sh
mkdir -p /tmp/pre/runtime
git show f18608a^:crates/reims-vgpu/src/runtime/exec.rs > /tmp/pre/runtime/exec.rs
scripts/silent-arms/silent-arms.sh /tmp/pre
```

Three candidates, five presence-guards suppressed. Two of the three were the
real losses: an empty `SetScissor` that left the *previous* rectangle clipping
later draws, and an `ExecuteCommands` with `ref == 0` that dropped a whole ICB
batch. The third is the `SetPipeline` arm above.

## Known weakness

It matches an arm head by taking the text before the arm's first `=>`. That is
right because `==`, `>=` and `!=` contain no `=>`, but a guard containing a
closure with its own `=>` before the arm separator would be misread. There is no
such arm in this crate today. An earlier version excluded `=` from the head
entirely and so matched no `x if x == FOO =>` arm at all — it reported one hit
and looked clean, which is the failure worth knowing about: **a quiet report
from this script should be checked against the validation above before it is
believed.**
