# scattered-struct

Finds values that travel as loose parameters when a type for them already
exists, or obviously wants to.

```sh
scripts/scattered-struct/scattered-struct.sh [crate-src-dir ...]
# defaults: crates/reims-vgpu/src and crates/reims-vgpu-wire/src
```

## Why this finds things clippy cannot

`clippy::too_many_arguments` fires on arity alone, so it is silent on a
four-parameter function that should take one struct and loud on a nine-parameter
one that legitimately takes nine. It also cannot see the thing that makes these
dangerous, which is not the *count* but the **run of same-typed neighbours**: a
signature ending `volume, cube, arrayed, one_dim` accepts all 24 permutations of
those four `bool`s and the compiler objects to none of them. Every commit this
script was written out of found the same shape, and in each the type that should
have travelled was already declared somewhere in the crate.

Two report classes.

### A type built out of the parameters passed to build it

A function whose body constructs a struct literal using four or more of its own
parameters in shorthand. The body's own words say the values belong together;
the signature took them apart. The fix is to move the construction to the call
site and pass the type.

The reason it is worth doing beyond tidiness: a caller that needs one field to
differ then says so by name — `SampledKey { format: resident_color(bgra),
..SampledKey::of(r) }` — where a positional call could only express it by
putting a different expression in the eighth slot, which nothing reading the
call would notice.

### An enum arm flattened into loose primitives

A `match` whose arms are tuples of bare literals. The enum is the type; the
tuple is where it stopped being one. A run of these over one enum in one file is
the finding — a lone arm is usually something else.

This class is the more serious of the two, because flattening an *n*-state enum
into *k* booleans manufactures `2^k − n` states that the type could not express.
Every consumer downstream then has to decide what the extra states mean, and
they will not all decide the same way: five functions inheriting one such
flattening had four different readings of the state that could not occur.

### A run of adjacent parameters of one primitive type

Four or more neighbouring parameters sharing one primitive type. Every
permutation of the run compiles and no call site can object, so the run is the
hazard whether or not a type is waiting for it.

**The finding is a run repeated across a family**, not a long run on its own.
`mapping_write`'s five rect functions all take `origin_x, origin_y, width,
height` and four of them add `bpp`; that repetition says the rectangle is a
thing the module keeps re-spelling. By contrast `compute_core`'s `grid_x,
grid_y, grid_z, tg_x, tg_y, tg_z` is one function mirroring the shape of
`dispatchThreadgroups:threadsPerThreadgroup:`, which is what the Metal API
takes — the run is the SDK's, not ours.

Sort the report by how many *signatures* share a run before sorting by run
length.

## What a hit can legitimately be

The report points at code; it does not convict it.

- **The parameters are transformed on the way in.** `batch_append` puts seven of
  its eight parameters into an `OpenBatch`, but `dset` becomes a collection and
  `sampled_retains` is moved, and its other arm — the joiner — uses three of the
  seven and discards the rest. Passing a built `OpenBatch` would hand the joiner
  a value it exists to not need.
- **The struct is the function's output, not its input.** `registry_ensure` and
  `fill_render_pso_key` are named for producing the thing they build. A
  constructor assembling its own fields from parameters is what a constructor
  is; the smell is only present when *callers* already hold the assembled value.
- **The enum arm carries genuinely unrelated values.** Two literals that happen
  to share an arm are not a flattened type. Look for the same arity in every arm
  and a struct built immediately after the `match`.

- **The run mirrors an API this code does not own.** See the `compute_core`
  case above. A run that reproduces an SDK call's parameter list is contract
  fidelity; grouping it would put a translation step between the decoded values
  and the call that consumes them.

The class that is always worth fixing is the last one: **callers that all hold
the same source value and spell its fields out**. Grep the call sites before
deciding — if every one of them reads `f(x.a, x.b, x.c, x.d)`, the type is
already there and the signature is the only thing that disagrees.

## Standing findings

Recorded so the next reader starts from the triage rather than the raw report.

- **`mapping_write`'s rectangle** — *done*, see the commit that introduced
  `SurfaceWindow` and `Rect`. Worth keeping the method: the 29 call sites were
  rewritten by a script that split each argument list at top-level commas and
  regrouped by position, rather than by hand. Hand-editing 29 sites where four
  of the arguments are same-typed `u32`s is how a crossing gets introduced by
  the very commit that exists to prevent one.
- **`translate::blend::state`** — `src_rgb, dst_rgb, op_rgb, src_alpha,
  dst_alpha, op_alpha` is two triples, and swapping the halves is silent. One
  signature only, so by the rule above it is weaker evidence than the rect
  family, but the two-triples shape makes the swap plausible rather than
  theoretical. Not yet examined.

## Known limits

Regex over lines, not a parser. It requires a struct literal's fields in
shorthand on their own lines, so `Foo { a: a, b: b }` and single-line literals
are invisible; and it reads only the first qualifying literal per function. A
clean report is weak evidence, a hit is strong evidence. Two filters exist
because their absence produced nonsense rather than noise: `) -> Status {` reads
as a struct literal to any regex this simple, and the field scan must stop at
the literal's own closing brace or it collects every shorthand field in the
function — that one reported eleven matching fields for a six-parameter
function.
