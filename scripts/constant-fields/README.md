# constant-fields

Reports always-on log fields that only ever take **one** value.

```sh
scripts/constant-fields/constant-fields.sh [logfile] [min-samples]
# defaults: /tmp/reims-vgpu-fail.log, 200
```

## Why this finds things reading the source cannot

`observe/` is large and almost every counter looks live in the source. What the
source does not show is whether a branch that would move a field is ever taken.
A field printed 12 000 times with the same value is not a measurement — it is
either **structurally impossible** or **vestigial**, and both are removable.

The sweep's own control is that it re-finds `gva_write gpa_match=0`, the write-gate
probe deleted in `d128fc1` for exactly this reason. If a run stops reporting that
on a pre-`d128fc1` log, the parser has drifted.

Two towers it found on first use:

- `type4 id_hits=0`, 12 172 samples. The page loop returns on the first
  untranslatable page, so any page that advances the counter also exits before
  the census line. Its sibling route `type4_identity_pages` turned out to have a
  distribution identical to `type4_translate_refused` — a duplicate counter — and
  the whole thing measured the counterfactual of an identity guess deleted in
  `6a3d878`.
- `guest_write_footprint war_licensed/superseded/unstated/unattributed=0`, 6 655
  samples. The licence tally only increments on the mapping rail, and
  `war_mapping` is *also* always zero, so the four buckets split a population
  that has never had a member. The `WriteLicence` parameter was threaded through
  the per-write, per-frame `mark()` hot path to fill them.

That second one is the shape worth internalising: **a bucket downstream of
another bucket that is itself always zero.** Four zeros looked like four quiet
measurements until the fifth zero showed they were unreachable.

## Bucketing by line family is load-bearing

Field names repeat across unrelated families. A `reason=` that is constant on one
line family and varied on another cancels out if you group by field name alone,
and the constant one is never reported. The script keys on `(family, field)`,
taking the family from the second whitespace field — the token after the
`OFF`/verbose marker.

## What a constant field can legitimately be

Do not cut on the report alone. Read the emitting code. These are all fine:

- **A standing alarm reading zero.** It is doing its job. `host_cache_levels
  gva_cap_wanted=0` is the GVA cap's harm reading, deliberately kept by `d16dd43`
  after the opposite claim turned out to be wrong.
- **A capability or geometry constant** for this host: `frame_shift=12`,
  `gva_cap_bytes=134217728`, a probed device feature.
- **An arm the workload never reached.** The type-11 ladder's own doc records
  that an undriven boot reported 12/5/8/0 where a driven one reported
  31 916/1 694/705/150 — "quiet enough to talk someone into deleting it". Drive
  the guest before believing a zero, and prefer the whole accumulated log over
  one boot.
- **A support-matrix cell** selected by a host *capability* rather than by the
  workload, per `AGENTS.md`.
- **One cell of a partition, where the constant value is the measurement.** A
  bucket reading zero on every driven second can be the finding rather than a
  dead counter: if the buckets partition the whole range, the empty one says the
  population never reaches it, and deleting the field deletes the result. This
  class is the hardest of the five to spot, because the field looks exactly like
  a vestigial counter and the value it holds is the one a vestigial counter
  holds. Weigh it against the opposite reading — once such a measurement has an
  answer and the answer is recorded where it is acted on, the instrument is a
  probe that outlived its investigation and should go.

The report points at code; it does not convict it.
