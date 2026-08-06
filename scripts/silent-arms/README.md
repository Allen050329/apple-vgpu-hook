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

- **A guard on decoder presence is not a finding.** `if x.is_some()`,
  `if word_count >= 3` — these ask whether the decoder saw the field at all, or
  whether the record was long enough to hold it. The arm not running means the
  guest did not send the thing, so there is nothing to lose. The script counts
  these and suppresses them.
- **A presence guard that cannot fail is worth deleting, not suppressing.** Check
  where the flag is set. If the decoder assigns it in the *same block* that
  assigns the kind the arm matches, and that kind has one producer, the guard is
  true whenever the arm is reached. It still costs something: the reader — and
  this script — sees a shape that says the device may discard a state the guest
  set, so it has to be re-triaged every run. The five `has_blend_color` /
  `has_cull_mode` / `has_front_facing` / `has_depth_bias` / `has_stencil_ref`
  guards in `runtime/exec.rs` were this, along with the five `Command` fields
  behind them; the suppressed count went 30 to 25 when they went.
- **A guard on a decoded value is a candidate.** `if cmd.scissor_w > 0`,
  `if cmd.indirect_command_buffer_ref != 0` — the guest sent a record and said
  something specific in it. Falling through means the device read the guest's
  statement and discarded it.
- **A candidate is still not automatically a finding.** The catch-all may be a
  real no-op state. Apply the test `AGENTS.md` gives for a never-firing branch,
  inverted: *name what the guest is asking for, and say what happens to it.* If
  the answer is "nothing should happen", the arm is right and wants a comment,
  not a counter.

## An adjudication that was wrong, and how

The `SetPipeline if cmd.pipeline_ref != 0` arm was on this list as legitimate,
reasoned as "a zero ref is an intentional unbind, which `AGENTS.md` names as
expected control flow". The premise was right and the conclusion did not follow.
A zero ref *is* an intentional unbind — but the arm did not perform one. Falling
through to `_ => {}` left the **previous** pipeline latched, so the following
draws were encoded against a pipeline the guest had stopped asking for, and
`dropped_unbound` read zero because they were executed rather than dropped.

The trap is that "expected control flow" answers a question about the *guest*
("is the guest allowed to send this?") and the arm is a statement about the
*device* ("what does this device then do?"). Reading the first as settling the
second is what put it on the list.

So the rule for this list: an entry is only adjudicated once it names **what the
device does instead**, not just why the guest's record is unremarkable. Every
entry below does. Fixed in the commit that added this section.

## Already adjudicated — do not re-report these

**Two** candidates on the current tree, both read and both legitimate. The
`PASS_LOAD_ACTION_*` ladders below no longer reach the script — the seven arms
this list used to carry were rewritten into shapes it does not match — so the
count moved from eight to two without anything being adjudicated away. Line
numbers drift; match on the arm text.

- `runtime/spirv_bind.rs` — the taint scan's
  `_ if …any(is_derived) => unknown = true`. The catch-all is an instruction that
  touches no derived value, which is the scan's normal case rather than a drop.
  Note the arm itself is the *fail-closed* direction: anything unrecognised that
  touches a derived image forces `StorageImageAccess::Unknown`, so the catch-all
  is reached only by instructions that provably touch nothing tracked.
- `runtime/draw/vulkan.rs` — `MTL_LOAD_ACTION_LOAD if chain_load_from_target`.
  What the device does instead: nothing, deliberately, and that *is* the LOAD.
  The guard means the resident render target already holds the previous frame's
  contents, so the attachment loads from itself and no CPU seed has to be staged
  — the `type11_seed_elided` counter above it is the same decision, counted. The
  un-guarded `MTL_LOAD_ACTION_LOAD` arm directly below performs the seed for
  every other case, so falling past this arm is served rather than dropped.
  Unknown actions never reach the ladder at all: the lines above coerce anything
  failing `load_action_in_contract` to `MTL_LOAD_ACTION_DONT_CARE` first, which
  is the fix for an earlier bug of exactly this shape.

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

Three candidates, five presence-guards suppressed. All three were real losses.
Two were fixed by `f18608a`: an empty `SetScissor` that left the *previous*
rectangle clipping later draws, and an `ExecuteCommands` with `ref == 0` that
dropped a whole ICB batch. The third was the `SetPipeline` arm, which is the same
shape as the `SetScissor` one — a guard that reads as a drop and is really a
carry-forward of the last value — and which this list initially cleared. It is
fixed now.

## Known weakness

It matches an arm head by taking the text before the arm's first `=>`. That is
right because `==`, `>=` and `!=` contain no `=>`, but a guard containing a
closure with its own `=>` before the arm separator would be misread. There is no
such arm in this crate today. An earlier version excluded `=` from the head
entirely and so matched no `x if x == FOO =>` arm at all — it reported one hit
and looked clean, which is the failure worth knowing about: **a quiet report
from this script should be checked against the validation above before it is
believed.**
