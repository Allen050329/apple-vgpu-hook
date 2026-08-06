//! A validity bound is one rule, so it is compared in one place.
//!
//! Sharing a constant shares the *number*. The **rule** is the comparison, and
//! nothing in the toolchain compares one copy of a comparison to another — so a
//! bound enforced at every site that needs it is a rule with as many
//! definitions as sites, and they drift. Every entry in [`BOUNDS`] below was
//! found already drifted:
//!
//! | bound | copies | what had parted |
//! |---|---|---|
//! | `MAX_SCANOUT_DIM` | 14 in 4 files | seven spelled the ceiling `>` and nothing checked the other seven did |
//! | `MAX_CHANNELS` | 7 in 4 files | three wrote `id == 0 \|\| id >= MAX`, four wrote the exact negation |
//! | `MAX_MAPPINGS` | 10 in 4 files | six omitted the zero test, and zero is the "no mapping" sentinel |
//! | `REIMS_VGPU_METAL_MAX_BUFFERS` | 8 in 4 files | a helper stating the rule existed, and five sites did not call it |
//! | `TEXTURE_VIEW_MIN_SIMPLE` | 2 in 2 files | both guarded an 8-byte header peek with one type-8 variant's *total* length |
//!
//! Two of those rows name constants that no longer exist, and the rows stay
//! because consolidating each is what made its removal readable. Gathering
//! `MAX_MAPPINGS`'s ten copies into `is_mapping_id` is what exposed that the
//! ceiling half bounded nothing — `DeviceState::mappings` is a `BTreeMap` keyed
//! by the full `u32` — so the rule kept its zero test and lost its ceiling.
//! `MAX_TASKS` went the same way one step further: its storage really was an
//! array, so removing the bound meant replacing the array, and `DeviceState::tasks`
//! is now a `TaskTable` over a map. A scattered bound is worth consolidating
//! even when the answer turns out to be that it should not exist.
//!
//! # Two tests, because there are two questions
//!
//! [`every_bound_is_compared_only_where_it_is_declared`] answers the strong one
//! for the five bounds above: *no* comparison outside the owning file, with each
//! survivor exempted by name and reason. It is the only shape that catches a
//! **single** reintroduced copy — once a bound is consolidated, a lone new
//! comparison is the only one left, and no census of "compared at two or more
//! sites" can see it. Measured, not assumed.
//!
//! [`no_bound_becomes_scattered_without_being_looked_at`] answers the weak one
//! for every constant in both crates: which bounds are compared away from where
//! they are declared *at all*. That is the census `scripts/scattered-bound`
//! prints, and it was wired to nothing — a discovery instrument nobody runs
//! reports a clean tree by never being asked. [`RECORDED`] freezes what it says
//! today, so a bound that becomes scattered tomorrow fails the build.
//!
//! The script stays: it ranks by polarity, which is the signal for *which* of
//! these to look at first, and a report is easier to read than an assertion.
//! The scan here was written independently and agrees with it exactly — same 24
//! names — which is the only reason to believe either.
//!
//! An integration test rather than a `#[cfg(test)]` module because it reads
//! source text and must run on every arm, including `backend-metal`, which this
//! development host can compile but cannot execute.

use std::path::{Path, PathBuf};

mod source_scan;

/// A bound, the file allowed to compare it, and the sites that legitimately
/// compare it anyway.
struct Bound {
    /// The constant's name as it appears in source.
    name: &'static str,
    /// The file that declares the rule. Comparisons here are the definition.
    owner: &'static str,
    /// `(file, needle, why)` for a comparison that asks a *different* question
    /// about the same constant. The needle must match the line's trimmed text.
    exempt: &'static [(&'static str, &'static str, &'static str)],
}

const BOUNDS: &[Bound] = &[
    Bound {
        name: "MAX_SCANOUT_DIM",
        owner: "model/regs.rs",
        exempt: &[],
    },
    Bound {
        name: "MAX_CHANNELS",
        owner: "model/regs.rs",
        exempt: &[],
    },
    Bound {
        name: "REIMS_VGPU_METAL_MAX_BUFFERS",
        owner: "backend/metal/util.rs",
        exempt: &[
            (
                "backend/metal/render.rs",
                "if slots.len() >= REIMS_VGPU_METAL_MAX_BUFFERS {",
                "a capacity check on the vertex-buffer slot list, not a test of \
                 one binding's index — `valid_buffer_binding` cannot answer \
                 \"is this list full\". Name the second question before adding \
                 an exemption.",
            ),
            (
                "backend/metal/constants.rs",
                "const _: () = assert!(REIMS_VGPU_METAL_MAX_BUFFERS as u32 <= REIMS_VGPU_BINDING_TEXTURE_BASE);",
                "relates two constants to each other, not a decoded index to a \
                 bound: it pins that the buffer band ends before the texture \
                 band begins, so no binding can name a slot in both. A predicate \
                 over one index cannot express that.",
            ),
        ],
    },
    Bound {
        name: "TEXTURE_VIEW_MIN_SIMPLE",
        owner: "runtime/decode/resource/mod.rs",
        exempt: &[],
    },
];

fn crate_src() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src")
}

fn collect_rs(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).expect("crate src must be readable") {
        let path = entry.expect("a readable dir entry").path();
        if path.is_dir() {
            collect_rs(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

/// Lines that relate `name` to something with `<`, `>`, `<=` or `>=`.
///
/// Comment text is stripped first, so a doc comment may name the constant
/// freely. A range (`1..MAX`), an array length (`[T; MAX]`) and a
/// `.field("limit", MAX)` report produce no relational token and are not
/// comparisons — none of them states the rule.
fn comparisons(root: &Path, name: &str) -> Vec<(String, usize, String)> {
    let mut files = Vec::new();
    collect_rs(root, &mut files);
    assert!(
        files.len() > 50,
        "walked {} files, which is not this crate",
        files.len()
    );
    files.sort();

    let mut out = Vec::new();
    for path in files {
        let rel = path
            .strip_prefix(root)
            .expect("every walked file is under src")
            .to_string_lossy()
            .into_owned();
        let text = std::fs::read_to_string(&path).expect("crate source must be readable");
        for (n, line) in text.lines().enumerate() {
            let code = line.split("//").next().unwrap_or("");
            if code.contains(name) && (code.contains('>') || code.contains('<')) {
                out.push((rel.clone(), n + 1, line.trim().to_string()));
            }
        }
    }
    out
}

#[test]
fn every_bound_is_compared_only_where_it_is_declared() {
    let root = crate_src();
    let mut failures = Vec::new();

    for bound in BOUNDS {
        let found = comparisons(&root, bound.name);

        // A bound nobody compares is a bound whose name changed out from under
        // this table, and an empty result would pass silently forever.
        assert!(
            found.iter().any(|(f, _, _)| f == bound.owner),
            "{} is compared nowhere in its declaring file {} — either the \
             constant was renamed or the rule moved, and this table is now \
             checking nothing",
            bound.name,
            bound.owner
        );

        for (file, line, text) in &found {
            if file == bound.owner {
                continue;
            }
            let exempted = bound
                .exempt
                .iter()
                .any(|(f, needle, _)| f == file && text == *needle);
            if !exempted {
                failures.push(format!(
                    "  {file}:{line}  {text}\n      ({} is owned by {}; route this through the \
                     predicate there, or add an exemption saying which *other* question it asks)",
                    bound.name, bound.owner
                ));
            }
        }

        // An exemption that no longer matches anything is a claim about code
        // that has moved on, and it would quietly widen the gate.
        for (file, needle, _) in bound.exempt {
            assert!(
                found.iter().any(|(f, _, text)| f == file && text == needle),
                "the exemption for {} at {file} matches no line any more; \
                 delete it so the gate keeps meaning something.\n  wanted: {needle}",
                bound.name
            );
        }
    }

    assert!(
        failures.is_empty(),
        "a bound is one rule and these restate it:\n{}",
        failures.join("\n")
    );
}

// ---------------------------------------------------------------------------
// The property, not the list.
// ---------------------------------------------------------------------------

/// A bound compared away from the file that declares it, and what was found
/// when it was looked at.
///
/// This is a **ratchet, not a certificate**. Each row records that the shape was
/// measured and left alone, with the reason it is not (or not yet) a finding;
/// none of them is a claim that the copies agree — that claim is what [`BOUNDS`]
/// above makes, for the five bounds where it was established site by site. What
/// the row buys is that a bound becoming scattered *after* this was written
/// fails the build instead of waiting for somebody to run a script.
struct Recorded {
    name: &'static str,
    why: &'static str,
}

const RECORDED: &[Recorded] = &[
    // The bind-slot admission, one constant per argument-table class. These were
    // a single `MAX_BIND_SLOTS`, which the polarity ranking put first and which
    // was Metal's *buffer* table applied to all three classes — so the texture
    // and sampler bounds were buffer facts and could not move independently.
    // Still not consolidated behind one predicate, for the reason the shared
    // constant had: the sites do not agree on what happens after a refusal —
    // `draw::vulkan` skips the bind (`continue`), `draw::metal_icb` refuses the
    // whole record — so one predicate would have to return which.
    Recorded {
        name: "MAX_BUFFER_BIND_SLOTS",
        why: "8 comparisons across the three render bind paths. Metal's buffer argument \
              table, pinned equal to it by a `const` assertion beside \
              `REIMS_VGPU_METAL_MAX_BUFFERS`, and independently equal to Apple's own \
              `bind_limit::BUFFER` — so the sites cannot disagree about the number, only \
              about what they do when it is exceeded.",
    },
    Recorded {
        name: "MAX_TEXTURE_BIND_SLOTS",
        why: "11 comparisons across the three render bind paths. Not a table size: it is \
              the width of the descriptor binding band between `TEXTURE_BINDING_BASE` and \
              `SAMPLER_BINDING_BASE`, held there by a `const` assertion, because a flat \
              binding number cannot say which class wrote it.",
    },
    Recorded {
        name: "MAX_SAMPLER_BIND_SLOTS",
        why: "9 comparisons across the three render bind paths. The same band-width \
              basis as the texture bound, applied to `SAMPLER_BINDING_BASE`..\
              `COLOR_INPUT_BINDING_BASE`. Metal's own 16-entry sampler table is tighter \
              and is enforced in the backend that owns it, fail-visibly, rather than \
              here — a Metal table size applied during stream accumulation would take \
              the slot from the Vulkan arm too.",
    },
    // Slice-bound checks. The constant is a record's length and each site bounds
    // a different read at a different offset, so the rule is `offset + LEN
    // <= buf.len()` and not a predicate over one value. Consolidating these
    // means checked accessors in `reims-vgpu-wire`, which is where that work
    // belongs and how the wire crate already states it.
    Recorded {
        name: "OP_HEADER_LEN",
        why: "record-header floor before a header read, at each decode entry point",
    },
    Recorded {
        name: "SEGMENT_HEADER_LEN",
        why: "segment-header floor, four reads in `decode::stream` plus the wire crate's own",
    },
    Recorded {
        name: "PACKET_HEADER_LEN",
        why: "FIFO packet-header floor in `drain`; the two polarities are \
              'too short to read' and 'long enough to snapshot'",
    },
    Recorded {
        name: "HEADER_WORDS",
        why: "SPIR-V header floor before indexing `words`, at four `spirv_bind` entries",
    },
    Recorded {
        name: "CHILD_EXEC_INDIRECT_HEADER_LEN",
        why: "one header floor asked in `decode::fifo` and again in `exec` on the payload it \
              hands on",
    },
    Recorded {
        name: "ICB_BUFFER_BIND_STRIDE",
        why: "`off + STRIDE > slot.len()` before two different ICB body reads",
    },
    Recorded {
        name: "ICB_CONCURRENT_DISPATCH_ARGS_LEN",
        why: "`args + LEN > len` before two different ICB body reads",
    },
    Recorded {
        name: "ICB_TESSELLATION_FACTOR_LEN",
        why: "`off + LEN > slot.len()` twice in one ICB decoder, at two offsets",
    },
    Recorded {
        name: "TYPE4_MIN_LEN",
        why: "descriptor-length floor before two different type-4 field reads",
    },
    // Capacity versus index: one site asks whether a decoded index is in the
    // band, another whether a list is full. `backend/metal/util.rs` holds the
    // index predicates; the same distinction is already written out as a named
    // exemption for `REIMS_VGPU_METAL_MAX_BUFFERS` in `BOUNDS` above.
    Recorded {
        name: "REIMS_VGPU_METAL_MAX_BUFFERS",
        why: "index band in `util`, list capacity in `render` — see the BOUNDS exemptions",
    },
    Recorded {
        name: "REIMS_VGPU_METAL_MAX_TEXTURES",
        why: "index band in `util`, list capacity and a usage cap in `compute`",
    },
    Recorded {
        name: "REIMS_VGPU_METAL_MAX_SAMPLERS",
        why: "index band in `util`, list capacity in `compute`",
    },
    Recorded {
        name: "REIMS_VGPU_METAL_MAX_ATTRS",
        why: "attribute-location band in `render` and in `stage_input`, plus a list capacity; \
              the two files narrow the same field for two descriptors",
    },
    Recorded {
        name: "REIMS_VGPU_METAL_MAX_COLOR_RTS",
        why: "list capacity and slot index, two lines apart in one function",
    },
    // A band's own edges, related to each other rather than to a decoded value.
    Recorded {
        name: "REIMS_VGPU_BINDING_TEXTURE_BASE",
        why: "`util`'s band predicate, and a `const` assertion in `constants` pinning the \
              buffer band to end before this one begins — the exemption `BOUNDS` already names",
    },
    Recorded {
        name: "REIMS_VGPU_BINDING_SAMPLER_BASE",
        why: "same pair as the texture base, one band up",
    },
    // The wire crate validates its own record and the device re-asks over the
    // type it built from it. Two crates, so no predicate is shared without one
    // depending on the other.
    Recorded {
        name: "MAX_DEPTH",
        why: "`wire::page_table` validates the declared depth; `contract::gva_resolve` re-asks \
              over the geometry it assembled",
    },
    Recorded {
        name: "TYPE4_PLANE_CAP",
        why: "`wire::device_desc` bounds a plane index, `objects` bounds a decoded plane count",
    },
    // Single-file pairs.
    Recorded {
        name: "CURSOR_MAX_DIM",
        why: "width and height on adjacent lines of one guard — one rule over two fields",
    },
    Recorded {
        name: "REG_BASE",
        why: "the MMIO read path and the write path each ask whether an offset is below the \
              register window",
    },
    Recorded {
        name: "SAMPLED_CACHE_CAP",
        why: "admission and the eviction loop, in the pool that owns both",
    },
    Recorded {
        name: "SAMPLED_CACHE_BYTE_CAP",
        why: "one entry's size, the running total, and the eviction loop, all in that pool",
    },
];

/// Where each constant is declared, by name.
type DeclaredIn = std::collections::BTreeMap<String, String>;
/// Every `(file, line)` a constant is compared at, by name.
type ComparedAt = std::collections::BTreeMap<String, Vec<(String, usize)>>;

/// Every `(name, file, line)` where a `SCREAMING_CASE` constant is related to
/// something by `<`, `>`, `<=` or `>=`.
fn comparison_census() -> (DeclaredIn, ComparedAt) {
    let root = source_scan::workspace_root();
    let mut declared: DeclaredIn = Default::default();
    let mut sites: ComparedAt = Default::default();

    for tree in ["crates/reims-vgpu/src", "crates/reims-vgpu-wire/src"] {
        for path in source_scan::rust_sources(&root.join(tree)) {
            // Three spellings of "this file is tests", all of which restate a
            // bound as a fixture: a whole `tests.rs` sibling, a `*_tests.rs`
            // one (`cap_tests.rs`, `revalidate_tests.rs`,
            // `render_flush_witness_tests.rs` — declared `#[cfg(test)] mod` by
            // their parent, so the marker is not in the file), and a `tests/`
            // directory. An inline `#[cfg(test)] mod` is blanked below.
            let name = path.file_name().unwrap_or_default().to_string_lossy();
            if name == "tests.rs"
                || name.ends_with("_tests.rs")
                || path.parent().is_some_and(|p| p.ends_with("tests"))
            {
                continue;
            }
            let rel = path
                .strip_prefix(&root)
                .expect("under the workspace")
                .to_string_lossy()
                .into_owned();
            let text = std::fs::read_to_string(&path).expect("source must be readable");
            let code = source_scan::blank_test_modules(&source_scan::blank_comments(&text));
            for (n, line) in code.lines().enumerate() {
                if let Some(name) = declares_const(line) {
                    declared.insert(name, rel.clone());
                }
                for name in compared_names(line) {
                    sites.entry(name).or_default().push((rel.clone(), n + 1));
                }
            }
        }
    }
    (declared, sites)
}

/// The constant a line declares, if it declares one.
fn declares_const(line: &str) -> Option<String> {
    let rest = line.trim_start();
    let rest = rest.strip_prefix("pub").map_or(rest, |r| {
        let r = r.trim_start();
        r.strip_prefix('(')
            .and_then(|r| r.split_once(')'))
            .map_or(r, |(_, after)| after.trim_start())
    });
    let rest = rest.strip_prefix("const ")?.trim_start();
    let name: String = rest.chars().take_while(|c| is_ident_char(*c)).collect();
    let after = rest[name.len()..].trim_start();
    (after.starts_with(':') && is_screaming(&name)).then_some(name)
}

fn is_ident_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

/// `MAX_THING` yes, `Max`, `MAX`, `max_thing` no — the shape a bound is spelled
/// in, which is also what keeps a type parameter or a field name out.
fn is_screaming(name: &str) -> bool {
    name.starts_with(|c: char| c.is_ascii_uppercase())
        && name.contains('_')
        && name
            .chars()
            .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
}

/// The constants this line *relates* to something, as opposed to naming.
///
/// Four neighbouring shapes are not comparisons and each is a page of false
/// report: `pfn << PAGE_ENTRY_PFN_SHIFT` (a shift — by far the largest source,
/// every page-table fixture writes one), `Foo => MAX_THING` (a match arm; the
/// fat arrow is not a `>`), `1..MAX_CHANNELS` and `[T; MAX_CHANNELS]` (a range
/// and an array length, neither of which states a rule). The first two are
/// refused by requiring the operator to be undoubled and not preceded by `=`;
/// the last two never produce a bare `<`/`>` beside the name at all.
fn compared_names(line: &str) -> Vec<String> {
    let b: Vec<char> = line.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        if !is_ident_char(b[i]) || (i > 0 && is_ident_char(b[i - 1])) {
            i += 1;
            continue;
        }
        let start = i;
        while i < b.len() && is_ident_char(b[i]) {
            i += 1;
        }
        let name: String = b[start..i].iter().collect();
        if is_screaming(&name)
            && (relation_before(&b, path_start(&b, start)) || relation_after(&b, i))
        {
            out.push(name);
        }
    }
    out.sort();
    out.dedup();
    out
}

/// Where the path leading to the name at `at` begins.
///
/// `x > wire::OPCODE_DRAW_PATCHES` is a comparison, and looking only at the
/// character before the name finds a colon. Both this scan and
/// `scripts/scattered-bound` missed every qualified comparison until this
/// existed; `decode::render` has two, on the opcode band that decides whether a
/// record is a draw at all.
fn path_start(b: &[char], name_start: usize) -> usize {
    let mut at = name_start;
    while at >= 2 && b[at - 1] == ':' && b[at - 2] == ':' {
        let mut seg = at - 2;
        while seg > 0 && is_ident_char(b[seg - 1]) {
            seg -= 1;
        }
        if seg == at - 2 {
            break;
        }
        at = seg;
    }
    at
}

/// A relational operator immediately to the left of `at`, ignoring spaces.
fn relation_before(b: &[char], at: usize) -> bool {
    let mut j = at;
    while j > 0 && b[j - 1] == ' ' {
        j -= 1;
    }
    if j == 0 {
        return false;
    }
    // `<=` / `>=`, then the bare `<` / `>`.
    let (op_start, ok) = if b[j - 1] == '=' && j >= 2 && (b[j - 2] == '<' || b[j - 2] == '>') {
        (j - 2, true)
    } else if b[j - 1] == '<' || b[j - 1] == '>' {
        (j - 1, true)
    } else {
        (j, false)
    };
    ok && (op_start == 0 || !matches!(b[op_start - 1], '<' | '>' | '='))
}

/// A relational operator immediately to the right of `at`, ignoring spaces.
fn relation_after(b: &[char], at: usize) -> bool {
    let mut j = at;
    while j < b.len() && b[j] == ' ' {
        j += 1;
    }
    if j >= b.len() || !matches!(b[j], '<' | '>') {
        return false;
    }
    // `<<`, `>>` and `<=`-that-is-really-`<==` cannot occur; a doubled operator
    // is a shift and `=` after is the two-character form, which is a relation.
    !matches!(b.get(j + 1), Some('<') | Some('>'))
}

#[test]
fn no_bound_becomes_scattered_without_being_looked_at() {
    let (declared, sites) = comparison_census();
    assert!(
        declared.len() > 200,
        "found {} declared constants, which is not these two crates",
        declared.len()
    );

    let mut found: Vec<(&String, &Vec<(String, usize)>)> = Vec::new();
    for (name, occ) in &sites {
        let owner = declared.get(name);
        let away = occ.iter().any(|(f, _)| Some(f) != owner);
        if away && occ.len() >= 2 {
            found.push((name, occ));
        }
    }

    let recorded: std::collections::BTreeSet<&str> = RECORDED.iter().map(|r| r.name).collect();
    let new: Vec<String> = found
        .iter()
        .filter(|(n, _)| !recorded.contains(n.as_str()))
        .map(|(n, occ)| {
            let where_ = occ
                .iter()
                .take(4)
                .map(|(f, l)| format!("{f}:{l}"))
                .collect::<Vec<_>>()
                .join(", ");
            format!("  {n}  ({} comparisons) {where_}", occ.len())
        })
        .collect();
    assert!(
        new.is_empty(),
        "these bounds are compared away from the file that declares them, and \
         nobody has looked:\n{}\nConsolidate the rule beside its constant, or \
         add a `Recorded` row saying what you found.",
        new.join("\n")
    );

    let names: std::collections::BTreeSet<&str> = found.iter().map(|(n, _)| n.as_str()).collect();
    let stale: Vec<String> = RECORDED
        .iter()
        .filter(|r| !names.contains(r.name))
        .map(|r| format!("  {}  (recorded as: {})", r.name, r.why))
        .collect();
    assert!(
        stale.is_empty(),
        "these rows describe a shape that is gone — the bound was consolidated, \
         renamed, or its second comparison deleted. Delete the row so the list \
         keeps meaning something:\n{}",
        stale.join("\n")
    );
}
