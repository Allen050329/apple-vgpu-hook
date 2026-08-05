# test-isolation

Runs every `--lib` test in its own process and diffs the result against the
in-suite one. A difference in either direction means the test's verdict is
partly a property of what ran before it.

```sh
scripts/test-isolation/test-isolation.sh
scripts/test-isolation/test-isolation.sh --features backend-metal   # Apple hosts only
```

Takes a few minutes: one process per test, and this crate's `--lib` binary is
large. Nothing about it is timing-sensitive, so it is safe to run beside other
work — unlike a boot.

## Why this is not covered by running the suite

The suite is one process, and several things this crate keeps for a whole boot
outlive the test that touched them:

- `observe::emit`'s `first_sight` and `state_changed` registries,
- `observe::sink`'s process-monotonic clock (`elapsed_us` reads 0 for the first
  microsecond of a run and never again),
- the `ACC` phase accumulators in `backend/vulkan/engine/draw_phase.rs`,
- every `OnceLock` in the crate, including the resolved log paths.

Whether two tests interact through one of these depends on which runs first,
and libtest orders by name — so the answer changes when a module is renamed.
It changed for real: renaming `runtime::metal_draw` to `runtime::draw` moved
several hundred tests across `runtime::mapper` and turned a green suite red.

## Triage

**Pass in-suite, fail alone.** The test needs a predecessor. This is the
common class and it is always a finding, because the assertion is not proving
what it claims — it is proving something about the process the suite happened
to hand it. Both hits on the first run of this script were assertions that a
*correct* implementation can violate:

- `landed_us > 0`, where the value is a reading of a clock that starts at zero.
  The cure is to establish the precondition the assertion needs (wait for the
  clock to leave zero) and then assert the real property (the stamp lies
  between readings taken either side of the call), not to relax the bound.
- `acquire_us == 0`, where the slot legitimately accumulates the cost of one
  statement — sub-microsecond in a warm process, 8 µs in a cold one. The cure
  is a ceiling *derived from the constant the test already declares*, chosen so
  it cannot be satisfied by the failure the assertion guards.

Neither cure is "make the assertion looser until it passes". Both made the
test strictly stronger; check yours does too, by restoring the bug it guards
and watching it go red.

**Fail in-suite, pass alone.** The test was poisoned by a predecessor. For a
capturing test this should now be impossible by construction —
`FailCapture::start` drops both dedup registries — so a hit here means either a
test that reaches an emitter without capturing the sink, or a fifth piece of
process state not on the list above. Find which, and prefer removing the
mechanism over renaming a fixture value around it.

## How it can mislead

- **It only sees `--lib`.** The 21 integration binaries each run in their own
  process already, but a multi-test integration binary can carry the same
  coupling internally and this script will not look.
- **A test that is order-dependent but whose neighbours never trigger it reads
  clean.** This measures the ordering that exists today, not every ordering. A
  clean run is evidence about this tree, not a proof about the next rename.
- **It refuses a verdict below 100 listed tests** rather than reporting a clean
  sweep over nothing, since a `--list` that stopped working looks exactly like
  a suite with no order-dependence in it.
