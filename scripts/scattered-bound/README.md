# scattered-bound

Finds a validity rule that is written out at every site that needs it, instead
of once beside the constant it bounds.

```sh
scripts/scattered-bound/scattered-bound.sh [crate-src-dir ...]
# defaults: crates/reims-vgpu/src and crates/reims-vgpu-wire/src
```

## Why the constant being shared is not enough

Sharing `MAX_SCANOUT_DIM` across fourteen call sites shares the *number* and
leaves the *rule* copied fourteen times. The number is what a header-drift test
already pins; the rule is `width == 0 || height == 0 || width > MAX ||
height > MAX`, and nothing in the toolchain compares one copy of it to another.

That is not hypothetical arithmetic. Every consolidation this script was written
out of found copies that had already parted:

| bound | copies | what had parted |
|---|---|---|
| `MAX_SCANOUT_DIM` | 14 in 4 files | seven spelled the ceiling `>` and nothing checked the other seven did |
| `MAX_CHANNELS` | 7 in 4 files | three wrote `id == 0 \|\| id >= MAX`, four wrote the exact negation |
| `MAX_MAPPINGS` | 10 in 4 files | **six omitted the zero test entirely**, and zero is the "no mapping" sentinel |

The third one was a live defect: the mapper decodes `mapping_id` out of the
guest's iosfc ring, and a MAP naming mapping 0 would have opened a slot no
sentinel-aware reader ever consults.

## Reading the report

Entries are ranked by whether the rule appears in more than one **polarity**,
then by how many files hold a copy away from the declaring one.

`<-- two spellings, one of them inverted` is the flag worth opening first. A
rule written once as `id >= MAX` and once as `id < MAX` does not grep as the
same string, so the reader who goes looking for copies finds a subset and stops.
Both `MAX_CHANNELS` and `MAX_MAPPINGS` were found this way.

## Three classes that are not findings

- **A predicate and its else-branch.** `if len < N { refuse } … else if len >= N
  { proceed }` is one rule stated for both outcomes, and the script cannot tell
  it from two rules. `DEVICE_DESC_LEN` and `PACKET_HEADER_LEN` are both this.
  Check whether the two sites decide the *same question* before touching either.
- **A bound whose sites are all in one function.** Two `off + LEN > size` tests
  inside one decoder are bounds-checking two different offsets, not restating
  one rule. The `ICB_*_LEN` entries are all this shape.
- **A cap read by its own owner's helpers.** `SAMPLED_CACHE_BYTE_CAP` is
  compared three times inside `submission_and_buffers.rs` because the eviction
  walk asks it at three points in one loop. Moving it would buy a name and lose
  the loop.

The rule of thumb: **name the second question.** If both sites are asking one
question, one of them is a copy. If they are asking two, they are two.

## Two shapes the scan deliberately does not see

A relational operator is required, so a range (`1..MAX_CHANNELS`) and an array
length (`[T; MAX_CHANNELS]`) never appear — neither restates a rule. Shifts
(`pfn << PAGE_ENTRY_PFN_SHIFT`), generic arguments (`Vec<MAX_THING>`) and match
arms (`Foo => MAX_THING,`) are excluded by lookaround, because each produced a
whole page of report before it was. Test modules are skipped in both spellings —
a whole `tests.rs` and an inline `#[cfg(test)] mod` — since a fixture restating
a bound ships nothing.

## Fixing one

Declare the predicate beside the constant, as a `const fn`, and route every site
through it. Where exactly one caller has to name *which* half of the rule broke
— because it emits a typed decline — return a named fault enum rather than a
bool and let that one caller match on it; `regs::scanout_extent_fault` is the
template and `regs::is_child_channel` is the plain-bool one.

Then add the gate, or the next copy lands unnoticed:

```rust
#[test]
fn the_thing_bound_is_compared_in_exactly_one_place() {
    assert_compared_only_in_regs("MAX_THING", "regs::is_thing");
}
```

`assert_compared_only_in_regs` lives in `model::regs::tests` and walks the
crate's `src/` for the same shape this script looks for. Three bounds use it.

**The test is not optional, and this script is not a substitute for it.** A bound
is only reported here when at least two sites compare it, because one
enforcement site away from a constants module is the ordinary case and reporting
it buries the report in 28 rows of nothing. The consequence was measured: after
`MAX_MAPPINGS` and `MAX_CHANNELS` were consolidated to one predicate each,
reintroducing a single copy of either — in the original inverted spelling —
produced **no output from this script at all**, because that copy was then the
only comparison left. `the_mapping_id_bound_is_compared_in_exactly_one_place`
caught the same reintroduction by name. The script finds the copies once; the
test is the only thing that keeps them from coming back.
