# `runtime/census/` — the always-on proxies

Every bug class in this project earns a **log- or test-level proxy** before or alongside its fix: a
signal that says "this class is happening" without anyone staring at a screenshot. Each module here
is one such class. There are ten because there have been ten classes, and that is the discipline
working, not sprawl.

## Why they are a directory now

Each proxy arrived as a new file next to `metal_draw.rs`, so `runtime/` came to read as though
measurement modules were peers of the execution path. They are not: the execution path calls them,
never the reverse. Grouping them makes that direction visible at a glance and leaves `runtime/`
listing the things that actually execute guest work.

## What is deliberately *not* here

`runtime/draw_log.rs` — the **sink**. `draw_log::fail` and `draw_log::off` are the always-on
outputs that everything in this directory writes to, and the execution path calls them directly for
its own declines. Filing the sink under `census/` would suggest that a dropped guest command is a
measurement. It is not: a decline **must** be logged, a measurement **must not** change behaviour.
That distinction is the whole point of the split.

`draw_log::line` is the third tier — gated behind `REIMS_VGPU_DRAW_LOG=1`, so a failure logged only through
it is invisible on a normal boot and does not satisfy the fail-visible rule.

## The rule every module here obeys

**Measuring is allowed; branching on the measurement is not.**

These modules may count nonzero pixels, sparsity, format volume, cache churn and geometry, and write
those counts to the always-on log. Nothing in the device or backend may read one back to decide what
to present, decode or execute. A proxy that changes behaviour has stopped being a proxy and become a
content heuristic, which the ground rules forbid outright.

## Reading them

They write to `/tmp/reims-vgpu-fail.log` (`OFF` and `THRASH` lines) and, for the present path,
`/tmp/reims-vgpu-thrash.log`. Count-based fields (`misses`, `hits`, `hitches`, `readback_mb`) are always
trustworthy. Wall-clock fields (`us_*`, `*_us_avg`) are SCHED_IDLE-contaminated whenever the host is
busy, because the testing boot runs QEMU at SCHED_IDLE — trust them only on a quiet host.

## Adding one

A new proxy belongs here when it measures a **named** bug class. Give it the class name, a
regression test that fires it on a synthetic case, and an entry in the table in `mod.rs`. Then verify
on a healthy boot that it fires **zero** times before calling the work done — a proxy that floods is
worse than none, because it trains the next reader to ignore the log.
