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
//! `scripts/scattered-bound` finds this shape across the crate. It is a
//! discovery instrument and cannot replace this test: it only reports a bound
//! compared at two or more sites, so once a bound is consolidated a *single*
//! reintroduced copy is invisible to it — that copy is then the only comparison
//! left. Measured, not assumed. This test names the file and line instead.
//!
//! An integration test rather than a `#[cfg(test)]` module because it reads
//! source text and must run on every arm, including `backend-metal`, which this
//! development host can compile but cannot execute.

use std::path::{Path, PathBuf};

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
        name: "MAX_MAPPINGS",
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
                "assert!(REIMS_VGPU_METAL_MAX_BUFFERS as u32 <= REIMS_VGPU_BINDING_TEXTURE_BASE);",
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
