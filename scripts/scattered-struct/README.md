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

Three report classes.

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

**Two shapes this class reports that the cure does not fit**, both adjudicated
on this tree and both still in the report:

- *The function creates most of what it packs.* `registry_ensure` builds a
  `NewResident` out of four parameters — and out of the image, memory, view and
  framebuffer it just created, which no caller has. Moving the construction up
  is not available; what the type buys here is that the *other* four cannot be
  transposed on the way in, and that a second creation arm cannot invent
  different values for the fields neither of them is given.
- *Whether the struct is built at all is decided inside.* `batch_append` opens a
  batch or extends the open one, and only the opening arm constructs an
  `OpenBatch`. A caller passing a finished one would be building a value that is
  discarded on the path it cannot predict. Its five parameters also have five
  distinct types, so the permutation hazard this script exists for is absent.

The tell for both: ask what the call site would have to know to build the type.
If the answer includes something the function has not created or decided yet,
the report is describing the signature accurately and the cure still does not
apply.

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
`mapping_write`'s five rect functions all took `origin_x, origin_y, width,
height` and four of them added `bpp`; that repetition says the rectangle is a
thing the module keeps re-spelling.

Sort the report by how many *signatures* share a run before sorting by run
length — but do not stop there. A run that looks like an SDK's parameter list
still needs its *callers* checked: `compute_core`'s six dispatch dimensions
read as a mirror of `dispatchThreadgroups:threadsPerThreadgroup:` and were
triaged away here on that basis, until the producer turned out to be a function
returning an anonymous seven-tuple over a type that already existed. See the
standing findings below. The SDK-mirror exemption is about where a value is
*going*, and this class is about where it has been.

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
- **The run mirrors an API this code does not own, *and arrives loose*.** A run
  reproducing an SDK call's parameter list is contract fidelity, and grouping it
  would put a translation step between the decoded values and the call that
  consumes them. Both halves are required: `compute_core` satisfies the first
  and fails the second, which is why it is a standing finding below rather than
  an example here.

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
- **`translate::blend::state`** — *done*. Two triples where a half-swap returns
  `Ok` with the wrong channels blended, so there is nothing for the fail
  channel to say. Now takes the decoded `&PipelineColorAttachment`.
- **The compute dispatch dimensions** — open, and a correction to this file's
  own earlier triage. `compute_core` and `compute_encode_on_encoder` were
  called SDK mirrors here because `grid_x, grid_y, grid_z, tg_x, tg_y, tg_z`
  reproduces `dispatchThreadgroups:threadsPerThreadgroup:`. That is true of the
  *Metal call at the bottom*, but not of how the values arrive:
  `compute_exec::resolve_dispatch_dims_reported` returns
  `(u32, u32, u32, u32, u32, u32, bool)` — an anonymous seven-tuple destructured
  by two callers — and `decode::compute::Size3` already exists as the type for
  each half. So this is the `Point` shape after all, with an SDK-shaped call
  only at the far end.

  **The producer half is done.** `DispatchDims` was a type alias over
  `(u32, u32, u32, u32, u32, u32, bool)` — it had a name but no type — and is
  now a struct of two `Extent3`s. `Size3` itself could not be reused: it is
  `u64` on the wire and the resolve narrows each component through `u32_dim`.

  **The consumer half is done too**, at `cf4d41c`: `compute_core` and
  `compute_encode_on_encoder` take `grid: Extent3, threadgroup: Extent3`. This
  entry said otherwise for eight hours after that landed — it was written before
  the fix and not revisited when the file was next edited — and the next reader
  to trust it spent a detour confirming code that was already correct. If you
  are about to record a finding as open here, check the tree first; that is the
  whole reason this file exists.

  **What the detour did find**, and the reason this entry is worth keeping: with
  the six extents gone, `dispatch_kind` and `dispatch_type` are the *only* run
  left at that call — two adjacent `u32`s, one call site, and the script no
  longer reports them because its threshold is four. Both are `{0, 1}`:
  `THREADGROUPS`/`THREADS` and `SERIAL`/`CONCURRENT`. So a transposition
  compiles, passes *both* validators (every value of one is a valid value of the
  other), and silently changes two things at once — `dispatchThreads` versus
  `dispatchThreadgroups`, and whether Metal may overlap the segment.

  The lesson for this script's triage: **fixing part of a long run can hide the
  rest of it.** A run of eight shortened to two is not a run that got safer, it
  is a run that dropped below the report threshold — and a two-run whose members
  share a value domain is more dangerous than an eight-run of distinct ones. When
  closing one of these, re-read what is left rather than re-running the script.

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
