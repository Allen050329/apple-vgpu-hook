//! A refusal this device stores past the call that produced it must say whether
//! it can go stale.
//!
//! A GPU refuses while its memory is full and stops refusing once it is not.
//! This device has repeatedly done the first half and not the second: it stored
//! the refusal, consulted the store before re-attempting, and so never asked
//! again. Three of those shipped, in three unrelated modules, and each was
//! invisible for the same reason — the store looks like a cache, caches are
//! supposed to be free, and nothing about the storing line says the value in it
//! has a shelf life.
//!
//! * `engine::caches::ObjectCache::negative` remembered `vkCreate*` failures
//!   including `ERROR_OUT_OF_DEVICE_MEMORY`. The lookup consults `negative`
//!   before the create, so the create that would displace the entry never ran:
//!   a guest that freed a texture atlas and re-bound the same pipeline got a
//!   replayed error and the driver was never asked.
//! * `engine::context::ContextOwner::init_error` latched whatever bring-up
//!   refused with. `vkCreateInstance` and `vkCreateDevice` both refuse with
//!   `ERROR_OUT_OF_HOST_MEMORY`, so a host briefly short of RAM at the first
//!   draw lost the whole Vulkan engine for the life of the process.
//! * `m2v_cache::Entry::Failed` remembered a scratch-file write that failed for
//!   want of disk, turning one full `/tmp` into a shader that never renders
//!   again.
//!
//! # What this test asserts
//!
//! Not that any particular answer is right — that is a judgement about the
//! failure, and it belongs next to the code. It asserts that the question was
//! *asked*: every place an error type is stored in a non-error type appears
//! below with a verdict. A new one fails this test until its author writes a
//! line here, which is the moment the question is cheapest to answer.
//!
//! # Why source text
//!
//! The property is "somebody decided", which no run observes: a store that
//! never checks staleness behaves exactly like one that checked and concluded
//! the refusal is permanent, right up until the day the refusal isn't. Source
//! also answers for both backends at once — the Vulkan sites are `#[cfg]`-ed
//! out of the Metal build and vice versa, so no single compilation sees them
//! all, but both are text.

mod source_scan;
use source_scan::{blank_comments, blank_test_modules, rust_sources, workspace_root};

/// How a stored refusal answers the question.
///
/// `Permanent` is currently unused, and the `#[allow]` is there to say that
/// rather than to hide it: as of this test, every store in the crate either
/// drops its stale entries or is not a store at all. A store that keeps
/// everything can still be right — one holding nothing but capability refusals
/// would be — so the vocabulary keeps the option. Choosing it is a claim that no
/// refusal reaching that store can ever describe a single instant, which is the
/// claim all three fixed sites got wrong.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Staleness {
    /// A second identical attempt meets the same answer, so remembering it
    /// costs nothing and saves the re-attempt.
    Permanent,
    /// The refusal describes the host at one instant, so the store drops it and
    /// a later ask reaches the real attempt again.
    Guarded,
    /// The error is a payload of something that is not a store at all — it
    /// travels with a value rather than outliving one.
    NotAStore,
}

/// Every field or variant in this crate whose type carries one of its error
/// types, and what was decided about it.
///
/// Keyed by `(file, declaring type, member)`. The member is the field name, or
/// the variant name for an enum payload.
const VERDICTS: &[(&str, &str, &str, Staleness, &str)] = &[
    (
        "src/backend/vulkan/engine/caches.rs",
        "ObjectCache",
        "negative",
        Staleness::Guarded,
        "insert_negative drops anything DrawError::out_of_memory() answers for; \
         the rest is malformed input or a capability this host lacks",
    ),
    (
        "src/backend/vulkan/engine/context.rs",
        "ContextOwner",
        "init_error",
        Staleness::Guarded,
        "note_init_failure latches only bring-up verdicts about the machine — no \
         loader, no device, none above the API floor, no graphics queue",
    ),
    (
        "src/runtime/m2v_cache.rs",
        "Entry",
        "Failed",
        Staleness::Guarded,
        "forget_if_transient drops the scratch-write declines when the lookup \
         hands one back; the translate declines are about the AIR bytes the \
         cache is keyed by",
    ),
    // The rest are one decline quoting the inner one that caused it. They are
    // the shape this scan cannot tell from a store by type alone — an error
    // field in an error — and each is a value travelling with its own refusal to
    // the log, not one outliving anything.
    (
        "src/backend/vulkan/engine/draw_preparation.rs",
        "DrawPreparationDecline",
        "reason",
        Staleness::NotAStore,
        "the m2v decline that refused the vertex or fragment module",
    ),
    (
        "src/runtime/draw/texture_view.rs",
        "TextureViewDecline",
        "reason",
        Staleness::NotAStore,
        "the decode status that refused the serialized view descriptor",
    ),
    (
        "src/runtime/m2v_cache.rs",
        "M2vCacheDecline",
        "reason",
        Staleness::NotAStore,
        "the SPIR-V layout decline that refused the post-emit repair",
    ),
    (
        "src/backend/vulkan/engine/device_lost.rs",
        "DeviceLostDecline",
        "cause",
        Staleness::NotAStore,
        "RecreateFailed keeps the typed cause instead of flattening it through Display",
    ),
    (
        "src/backend/vulkan/engine/dmabuf.rs",
        "GuestWriteDecline",
        "inner",
        Staleness::NotAStore,
        "Import names the step of the import that declined",
    ),
    (
        "src/backend/vulkan/engine/window_present.rs",
        "StagingError",
        "Call",
        Staleness::NotAStore,
        "names which of the presenter's five staging allocations refused",
    ),
    (
        "src/runtime/draw/mod.rs",
        "MetalStateDecline",
        "reason",
        Staleness::NotAStore,
        "the decode status that refused a sampler or depth-stencil descriptor",
    ),
    (
        "src/runtime/mapping_write/mod.rs",
        "GpuWritebackDecline",
        "inner",
        Staleness::NotAStore,
        "Engine names which engine decline or copy failure the writeback met",
    ),
    // `Translation` is a per-call result, not a store, and the distinction is
    // load-bearing here rather than cosmetic: this rail does have a cache, and a
    // cached "this GVA does not translate" would survive the guest mapping the
    // page. It cannot happen — `translate_root`'s walk-failure arm returns
    // before `cache_insert`, so only a successful `gpa_page` is ever inserted,
    // and `ResolveStatus` rides out with the answer rather than into the map.
    (
        "src/contract/gva_resolve.rs",
        "Translation",
        "status",
        Staleness::NotAStore,
        "the outcome of one page-table walk, returned to the caller; only \
         successful walks reach cache_insert",
    ),
    (
        "src/contract/gva_resolve.rs",
        "Translation",
        "cache_status",
        Staleness::NotAStore,
        "whether that one walk hit the cache — the cache reporting on itself",
    ),
];

/// The error and status types whose storage this test tracks.
///
/// Gathered from the source rather than listed, so a new decline type is
/// covered the day it is written. Three families: everything with a `Decline`
/// impl, the `*Status` enums the decode and blit rails return, and `DrawError`,
/// which is the engine's own sum over most of the first family.
fn error_type_names(sources: &[(std::path::PathBuf, String)]) -> Vec<String> {
    let mut names: Vec<String> = vec!["DrawError".to_string()];
    for (_, text) in sources {
        for line in text.lines() {
            let line = line.trim();
            if let Some(rest) = line.strip_prefix("impl Decline for ") {
                let name = rest.trim_end_matches(" {").trim();
                if !name.is_empty() && name.chars().all(|c| c.is_alphanumeric() || c == '_') {
                    names.push(name.to_string());
                }
            }
            for prefix in ["pub enum ", "enum ", "pub(crate) enum "] {
                if let Some(rest) = line.strip_prefix(prefix) {
                    let name = rest.trim_end_matches(" {").trim();
                    if name.ends_with("Status") || name.ends_with("Decline") {
                        names.push(name.to_string());
                    }
                }
            }
        }
    }
    names.sort();
    names.dedup();
    // `Fake` is a test double for the `Decline` trait itself and never leaves
    // its own module.
    names.retain(|n| n != "Fake");
    names
}

/// Every `struct`/`enum` body in `text`, as `(name, body span in chars)`.
///
/// Brace-matched rather than read as "the nearest declaration above this line".
/// Nearest-above cannot tell a field from a match arm, and this crate is full of
/// arms that look exactly like one — `objects::LadderRung::NoListEntry =>
/// TextureViewDecline::HopEntryMissing { texture_ref },` parses as a member
/// named `objects` holding an error type. Scoping to the body is what makes the
/// scan report declarations only.
fn type_bodies(text: &str) -> Vec<(String, usize, usize)> {
    let chars: Vec<char> = text.chars().collect();
    let mut bodies = Vec::new();
    for keyword in ["struct ", "enum "] {
        let mut from = 0usize;
        while let Some(at) = text[from..].find(keyword) {
            let start = from + at;
            from = start + keyword.len();
            // A keyword must start a declaration, not sit inside an identifier
            // or a path (`enum` in `MyEnum `, `struct ` after `::`).
            let before = text[..start].chars().next_back();
            if before.is_some_and(|c| c.is_alphanumeric() || c == '_' || c == ':') {
                continue;
            }
            let rest = &text[start + keyword.len()..];
            let name: String = rest
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            if name.is_empty() {
                continue;
            }
            // Generics and where-clauses sit between the name and the body.
            let Some(open_rel) = rest.find('{') else {
                continue;
            };
            // A `;` first means a unit or tuple struct with no braced body.
            if rest[..open_rel].contains(';') {
                continue;
            }
            let open =
                text[..start + keyword.len()].chars().count() + rest[..open_rel].chars().count();
            let close = source_scan::close_brace(&chars, open);
            bodies.push((name, open, close));
        }
    }
    bodies
}

/// The innermost `struct`/`enum` body containing character `at`.
fn declaring_type(bodies: &[(String, usize, usize)], at: usize) -> Option<String> {
    bodies
        .iter()
        .filter(|(_, open, close)| *open < at && at < *close)
        .min_by_key(|(_, open, close)| close - open)
        .map(|(name, _, _)| name.clone())
}

/// One place an error type is written into a member of some type.
#[derive(Debug)]
struct Site {
    file: String,
    declaring: String,
    member: String,
    line: usize,
}

fn find_sites(sources: &[(std::path::PathBuf, String)], errors: &[String]) -> Vec<Site> {
    let mut sites = Vec::new();
    for (path, text) in sources {
        let file = path.to_string_lossy().to_string();
        let bodies = type_bodies(text);
        let mut at = 0usize;
        for (i, line) in text.lines().enumerate() {
            let line_start = at;
            at += line.chars().count() + 1;
            let trimmed = line.trim();
            // A member declaration: `name: Type,` for a struct field or a named
            // enum-variant field, or `Variant(Type)` for a tuple payload.
            let (member, ty) = if let Some((lhs, rhs)) = trimmed.split_once(':') {
                let name: String = lhs
                    .rsplit(char::is_whitespace)
                    .next()
                    .unwrap_or(lhs)
                    .to_string();
                if !name
                    .chars()
                    .all(|c| c.is_lowercase() || c.is_numeric() || c == '_')
                    || name.is_empty()
                {
                    continue;
                }
                (name, rhs.to_string())
            } else if let Some((lhs, rhs)) = trimmed.split_once('(') {
                let name = lhs.trim().to_string();
                if name.is_empty() || !name.chars().next().is_some_and(char::is_uppercase) {
                    continue;
                }
                if !name.chars().all(|c| c.is_alphanumeric() || c == '_') {
                    continue;
                }
                (name, rhs.to_string())
            } else {
                continue;
            };
            // The line has to end a declaration, not open an expression.
            if !ty.trim_end().ends_with(',') && !ty.trim_end().ends_with("),") {
                continue;
            }
            let Some(error) = errors.iter().find(|e| mentions_type(&ty, e)) else {
                continue;
            };
            let _ = error;
            let Some(declaring) = declaring_type(&bodies, line_start) else {
                continue;
            };
            sites.push(Site {
                file: file.clone(),
                declaring,
                member,
                line: i + 1,
            });
        }
    }
    sites
}

/// Whether `ty` names `name` as a whole path segment.
///
/// Segment-wise so `SpirvLayoutDecline` inside
/// `crate::runtime::spirv_layout::SpirvLayoutDecline` is found, while
/// `DrawErrorKind` would not be mistaken for `DrawError`.
fn mentions_type(ty: &str, name: &str) -> bool {
    ty.split(|c: char| !(c.is_alphanumeric() || c == '_'))
        .any(|seg| seg == name)
}

#[test]
fn every_stored_refusal_carries_a_verdict() {
    let root = workspace_root();
    let src = root.join("crates/reims-vgpu/src");
    let sources: Vec<(std::path::PathBuf, String)> = rust_sources(&src)
        .into_iter()
        .map(|p| {
            let raw = std::fs::read_to_string(&p).expect("read source");
            let text = blank_test_modules(&blank_comments(&raw));
            let rel = p
                .strip_prefix(root.join("crates/reims-vgpu"))
                .unwrap_or(&p)
                .to_string_lossy()
                .to_string();
            (std::path::PathBuf::from(rel), text)
        })
        .collect();

    let errors = error_type_names(&sources);
    assert!(
        errors.len() >= 10,
        "the scan found only {} error types, so it is not seeing this crate: {errors:?}",
        errors.len()
    );

    let sites = find_sites(&sources, &errors);

    // Self-check, in the shape `wire_families_have_a_consumer` uses: refuse to
    // report anything until the scan has proved it can see the three sites this
    // test was written for. A regex that silently matches nothing reports a
    // clean tree, which is the failure mode of every source scan.
    for known in [
        ("src/backend/vulkan/engine/caches.rs", "negative"),
        ("src/backend/vulkan/engine/context.rs", "init_error"),
        ("src/runtime/m2v_cache.rs", "Failed"),
    ] {
        assert!(
            sites
                .iter()
                .any(|s| s.file == known.0 && s.member == known.1),
            "the scan cannot see {}::{}, so its silence means nothing. Found: {:#?}",
            known.0,
            known.1,
            sites
        );
    }

    let mut unclassified = Vec::new();
    for site in &sites {
        let known = VERDICTS.iter().any(|(file, declaring, member, _, _)| {
            *file == site.file && *declaring == site.declaring && *member == site.member
        });
        if !known {
            unclassified.push(format!(
                "{}:{} — {}::{} stores an error type",
                site.file, site.line, site.declaring, site.member
            ));
        }
    }
    assert!(
        unclassified.is_empty(),
        "these hold a refusal past the call that produced it and say nothing about \
         whether it can go stale.\n\nFor each: can a second identical attempt \
         answer differently? If yes, drop the entry when it is handed back — see \
         `ObjectCache::insert_negative`. If no, or if the error is only a payload \
         travelling inside another error, add a line to VERDICTS in {} saying so.\
         \n\n{}",
        file!(),
        unclassified.join("\n")
    );

    // The converse: a verdict for a site that no longer exists is a stale claim
    // about this crate, and reads as coverage it does not have.
    let mut orphaned = Vec::new();
    for (file, declaring, member, _, _) in VERDICTS {
        let live = sites
            .iter()
            .any(|s| s.file == *file && s.declaring == *declaring && s.member == *member);
        if !live {
            orphaned.push(format!("{file} — {declaring}::{member}"));
        }
    }
    assert!(
        orphaned.is_empty(),
        "VERDICTS claims these are classified, but the scan no longer finds them. \
         Delete the entries if the code went away.\n\n{}",
        orphaned.join("\n")
    );
}

/// The three stores that motivated this test are all guarded, and each names the
/// function that does it. If one is ever downgraded to `Permanent`, that is a
/// decision to stop asking the driver again — and it should be a visible one.
#[test]
fn the_three_known_stores_are_still_guarded() {
    for (file, member) in [
        ("src/backend/vulkan/engine/caches.rs", "negative"),
        ("src/backend/vulkan/engine/context.rs", "init_error"),
        ("src/runtime/m2v_cache.rs", "Failed"),
    ] {
        let verdict = VERDICTS
            .iter()
            .find(|(f, _, m, _, _)| *f == file && *m == member)
            .unwrap_or_else(|| panic!("{file}::{member} lost its verdict"));
        assert_eq!(
            verdict.3,
            Staleness::Guarded,
            "{file}::{member} stopped dropping refusals that describe one instant"
        );
    }
}
