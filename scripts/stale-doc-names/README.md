# stale-doc-names

Finds a doc comment that names a symbol this workspace deleted.

```sh
scripts/stale-doc-names/stale-doc-names.py
```

## The gap it covers

`AGENTS.md` already answers "does a doc comment name a symbol that no longer
exists?" with `cargo doc`'s intra-doc link pass. That pass only sees the
`[`link`]` form. Most prose in this tree writes a bare `` `name` ``, and
**nothing in the toolchain checks one** — a deleted function keeps being cited
in present tense, by a sentence that reads exactly like a true one.

That is not a cosmetic failure here. This repo's comments are where a measured
fact lives, so a comment crediting a rule to a function that no longer runs is a
reader believing a rule that is not enforced anywhere.

## Why the filter is git and not spelling

A backtick census alone is useless: 513 distinct backticked identifiers in this
tree are absent from the code, and almost all are log tags, census fields, wire
selector names, Vulkan enums, kernel symbols and guest-side names. Ranking them
does not help either — the rot is rare and unremarkable-looking.

The discriminator that works is two conditions at once:

- the name appears **only** inside comments today, and
- **a commit once removed a definition of it** (`git log -p` over `crates/*.rs`,
  collecting every `-` line that defined an item).

A log tag was never a definition, so it fails the second. A live function fails
the first. What survives both is a name this repo itself defined and then
deleted. That took 513 candidates to 45.

## Reading the output

**Most survivors are deliberate history, and they should stay.** This codebase
records what a bound used to be and why it was retired, and those records name
the retired thing on purpose — `MAX_TASKS`, `REGISTRY_CAP`, `MAX_BIND_SLOTS`,
`CHILD_RESOURCE_LIST_MAX_COUNT`, `opcode_is_apple_rejected` all read like rot and
are the opposite.

**Read the tense.** A sentence saying a name *used to* do something is the record
this repo wants kept. A sentence saying it *does* something is the rot. That is
the whole triage, and it is not mechanizable — the script's job is to get the
list down to something a person can read in one sitting.

Three shapes that are never rot, so do not "fix" them:

- **A deliberate do-not-reintroduce list.** `contract/gva.rs` names `PAGE_SHIFT`,
  `INDEX_BITS`, `INDEX_MASK` and `ENTRIES_PER_TABLE` to say those spellings must
  never come back. The names being absent is the point.
- **A non-Rust word that happens to match.** `EXT` is Vulkan's extension suffix
  in prose, and it once matched a deleted item name.
- **A test named as an argument, where the argument is that it was deleted.**
  `decline_slugs_are_unique`'s own module doc explains the `REGISTRY` it replaced.

## What the first run found

45 candidates, of which **13 were real** — every one a present-tense claim:

| name | what the comment claimed |
|---|---|
| `evict_registry_to_cap` (5 sites) | that a retired cap sweep is what skips pinned slots. The reclaim that does is `recoverable_residents`. |
| `cap_eviction_victim`, `compute_storage_eviction_victim` | that a cap sweep consults `last_touch_ms`. Both registries are bounded by the allocator now. |
| `ensure_host_imports` | a caller obligation to verify a host-pointer import — the one mechanism `AGENTS.md` forbids outright. |
| `OP_ACCEPTED_LAST` | that the render decode window ends at `0x98`. It ends at `0xa6`. |
| `execute_compute` | the shipped compute entry point's name. |
| `note_compute_bind_overflow` | which emitter puts an over-cap compute bind on the fail channel. |
| `render_target_class` | which contract function answers "may a colour attachment be this format". |
| `DISPLAY_VBL_MIN_INTERVAL_MS` | the VBL grid interval's name and unit (it is `_US`). |
| `write_span`, `write_linear_guest`, `map_fresh_span`, `is_single_packed_run` | four renamed guest-write helpers, each cited as the gate that refuses. |
| `note_store_damage_coverage`, `write_staging_rgba_as_bgra`, `present_into_host_runs`, `begin_entry_sync` | measurements and paths credited to names that no longer run. |
| `rgba_not_import`, `type11_cpu_store_fallback_allowed` | two store routes described as "kept as call sites" that are not call sites. |
| two test names | cited as locking a product choice that nothing locks. |

The `ensure_host_imports` and `OP_ACCEPTED_LAST` rows are the ones that justify
the script: both are load-bearing claims about what this device accepts, and both
had been wrong for long enough that a reader would have taken them as measured.
