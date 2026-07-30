//! Source-level gate that keeps the translation boundary from eroding.
//!
//! Everything this refactor fixed is cheap to undo by accident. One
//! `vk::Format::B8G8R8A8_UNORM` typed at a new call site is a second opinion
//! about what a Metal format means, and it will agree with the first one right
//! up until someone changes one of them. The cost of *noticing* is what makes
//! that class expensive: the two answers only disagree at runtime, on a
//! specific host, in a specific frame.
//!
//! Scanning source is crude, but it fails at `cargo test` time on the machine
//! that made the change — which is the only place the fix is cheap.

use std::fs;
use std::path::{Path, PathBuf};

fn crate_src() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src")
}

fn rust_files(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(entries) = fs::read_dir(&d) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
                out.push(path);
            }
        }
    }
    out.sort();
    out
}

fn rel(path: &Path, root: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

/// The Vulkan state enums whose *variants* encode a translated decision.
///
/// Each entry is the `vk::Thing::` prefix, i.e. the spelling of a variant. The
/// bare type name is deliberately absent: a function that *returns*
/// `vk::Format` is fine, it is naming a specific format that is the problem.
const TRANSLATED_STATE_SPELLINGS: &[&str] = &[
    "vk::Format::",
    "vk::BlendFactor::",
    "vk::BlendOp::",
    "vk::CompareOp::",
    "vk::PrimitiveTopology::",
    "vk::StencilOp::",
    "vk::IndexType::",
    "vk::Filter::",
    "vk::SamplerMipmapMode::",
    "vk::SamplerAddressMode::",
    "vk::BorderColor::",
    "ComponentSwizzle",
];

/// Directories that own translation and capability decisions and may therefore
/// spell these freely.
const OWNING_DIRS: &[&str] = &["backend/vulkan/translate/", "backend/vulkan/caps/"];

/// Files allowed to name a translated state spelling outside the owning
/// directories, each with the reason it is not a translation.
///
/// This is deliberately tiny. A file appearing here is claiming that its use of
/// a Vulkan state enum is **not** a decision about what a Metal value means —
/// which is true for exactly one thing today: negotiating a combined
/// depth-stencil format against the driver, where the question is "which of
/// these does this device support", not "what did the guest ask for".
///
/// Note what is NOT here. Swapchain images and resident targets
/// all use one format because the guest's scanout order is BGRA8 — that IS a
/// translated fact, so they name `translate::pixel::SCANOUT_FORMAT` instead of
/// spelling it, and a single wrong spelling among them would have shown up as
/// red-and-blue-swapped output rather than a failure.
const ALLOWLIST: &[(&str, &str)] = &[(
    "backend/vulkan/engine/context.rs",
    "depth-stencil format negotiation: asks the device which combined format it \
     supports, which is a capability question and not a Metal value's meaning",
)];

/// The engine state enums whose variants are the *output* of a translation.
///
/// The scan above is Vulkan-facing: it catches a site that spells
/// `vk::CompareOp::LESS`. This list is the other half of the same crossing —
/// the Metal-facing side, where a raw wire number becomes an engine enum. A
/// duplicate written in that direction spells no `vk::` variant at all, so the
/// Vulkan-facing scan never saw it, and three exact copies of
/// `translate::raster`'s tables lived in `runtime/metal_draw/mod.rs` through an
/// entire refactor written to drain that file.
///
/// A translation has two halves and a gate that watches one of them is a gate
/// that watches neither.
const ENGINE_STATE_ENUMS: &[&str] = &[
    "CullMode",
    "StencilOp",
    "PrimitiveTopology",
    "IndexType",
    "SamplerCompareFunction",
    "SamplerFilter",
    "SamplerMipFilter",
    "SamplerAddressMode",
    "SamplerBorderColor",
    "BlendFactor",
    "BlendOp",
    "StorageImageFormat",
    "VertexAttributeFormat",
    "VertexStepFunction",
];

/// Files outside `translate/` allowed to build an engine state enum from a
/// numeric pattern, each with the reason it is not a Metal translation.
///
/// Empty, and meant to stay that way. An entry here is a claim that a raw
/// number becoming pipeline state is *not* a decision about what a Metal value
/// means — which is a hard claim to make honestly, because that is what those
/// enums are for.
const METAL_FACING_ALLOWLIST: &[(&str, &str)] = &[];

/// Source split into function items.
///
/// Relies on the one thing rustfmt guarantees and this crate enforces: a
/// function's closing brace sits at exactly the same indent as its `fn`
/// keyword. Brace counting was tried first and bled adjacent functions
/// together — braces inside string literals and multi-line signatures both
/// break it — which produced offender reports naming a function that did not
/// contain the offending arm.
#[cfg(test)]
fn items(src: &str) -> Vec<(usize, String)> {
    let lines: Vec<&str> = src.lines().collect();
    let mut out = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        let indent = line.len() - line.trim_start().len();
        let t = line.trim_start();
        let is_fn = ["fn ", "pub fn ", "pub(crate) fn ", "pub(super) fn "]
            .iter()
            .any(|p| t.starts_with(p));
        if !is_fn {
            continue;
        }
        let closing = format!("{}}}", " ".repeat(indent));
        let end = lines[i + 1..]
            .iter()
            .position(|l| l.trim_end() == closing)
            .map(|p| i + 1 + p)
            .unwrap_or(lines.len() - 1);
        out.push((i + 1, lines[i..=end].join("\n")));
    }
    out
}

/// Does `body` name `word` as a whole identifier?
///
/// Substring matching is wrong here in a way that costs a whole afternoon:
/// `IndexType` occurs inside `MTLIndexTypeUInt16`, which is Apple's own enum in
/// a comment, so a plain `contains` reported three functions that translate
/// nothing. An engine enum is named only when the identifier stands alone.
#[cfg(test)]
fn names_word(body: &str, word: &str) -> bool {
    let bytes = body.as_bytes();
    let ident = |c: u8| c.is_ascii_alphanumeric() || c == b'_';
    body.match_indices(word).any(|(i, _)| {
        let before_ok = i == 0 || !ident(bytes[i - 1]);
        let after = i + word.len();
        let after_ok = after >= bytes.len() || !ident(bytes[after]);
        before_ok && after_ok
    })
}

/// Does this line open a match arm whose pattern is a raw wire value?
///
/// Three spellings, all seen in the duplicates this gate was written for:
/// a bare integer (`3 => StencilOp::Replace`), a hex literal, and the
/// `x if x == SOME_CONST as u32 =>` guard form that a `const`-valued pattern
/// forces you into.
#[cfg(test)]
fn is_wire_value_arm(line: &str) -> bool {
    let t = line.trim();
    if t.starts_with("//") || !t.contains("=>") {
        return false;
    }
    let pattern = t.split("=>").next().unwrap_or("").trim();
    if pattern.is_empty() {
        return false;
    }
    // `x if x == FOO =>` / `s if s == Sel::Bar as u32 =>`
    if pattern.contains(" if ") && pattern.contains("==") {
        return true;
    }
    // `3 =>`, `0x41 =>`, `0 | 1 =>`, `3..=7 =>`
    pattern
        .split(['|', '.'])
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .all(|p| {
            p.strip_prefix("0x")
                .map(|h| h.chars().all(|c| c.is_ascii_hexdigit()))
                .unwrap_or_else(|| p.chars().all(|c| c.is_ascii_digit()))
        })
}

/// Each wire-value arm in `body`, paired with the arm's own body.
///
/// The arm body runs from the arm line to the next line indented no deeper than
/// the arm itself, which is where the arm's block closes. That bound is what
/// makes "this arm produces the enum" answerable: a translation writes the
/// engine variant inside the arm that decoded the wire value, so the enum and
/// the arm are the same statement rather than the same function.
#[cfg(test)]
fn arm_bodies(body: &str) -> Vec<(&str, String)> {
    let lines: Vec<&str> = body.lines().collect();
    let mut out = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        if !is_wire_value_arm(line) {
            continue;
        }
        let indent = line.len() - line.trim_start().len();
        let end = lines[i + 1..]
            .iter()
            .position(|l| !l.trim().is_empty() && (l.len() - l.trim_start().len()) <= indent)
            .map(|p| i + 1 + p)
            .unwrap_or(lines.len() - 1);
        out.push((line.trim(), lines[i..=end].join("\n")));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A Vulkan state-enum variant outside the owning directories is a second
    /// opinion about what a Metal value means.
    #[test]
    fn translated_state_is_spelled_only_where_it_is_owned() {
        let root = crate_src();
        let mut offenders = Vec::new();
        for path in rust_files(&root) {
            let name = rel(&path, &root);
            if OWNING_DIRS.iter().any(|d| name.starts_with(d))
                || ALLOWLIST.iter().any(|(f, _)| *f == name)
            {
                continue;
            }
            let Ok(src) = fs::read_to_string(&path) else {
                continue;
            };
            for (i, line) in src.lines().enumerate() {
                for spelling in TRANSLATED_STATE_SPELLINGS {
                    if line.contains(spelling) {
                        offenders.push(format!("{name}:{}: {}", i + 1, line.trim()));
                    }
                }
            }
        }
        assert!(
            offenders.is_empty(),
            "these sites decide a Metal→Vulkan mapping outside \
             backend/vulkan/translate — route them through it (or name the \
             constant it already exports) so the same decision cannot be made \
             twice with two answers:\n{}",
            offenders.join("\n")
        );
    }

    /// The allowlist must stay minimal, point at files that exist, and point at
    /// files that still do the thing they were excused for. A stale entry
    /// silently widens the gate to a whole file.
    #[test]
    fn the_allowlist_is_exact() {
        let root = crate_src();
        assert_eq!(
            ALLOWLIST.len(),
            1,
            "one excused site: depth-stencil format negotiation. Adding a second \
             means something is being decided outside translate — check whether \
             it is really a capability question first"
        );
        for (site, reason) in ALLOWLIST {
            let path = root.join(site);
            assert!(path.exists(), "allowlisted site {site} no longer exists");
            assert!(!reason.is_empty(), "{site} has no stated reason");
            let src = fs::read_to_string(&path).expect("read allowlisted site");
            assert!(
                src.contains("depth_stencil_format"),
                "{site} is allowlisted for depth-stencil negotiation but no \
                 longer does it — drop the entry"
            );
            // The excuse covers depth-stencil formats and nothing else: if the
            // file starts spelling colour formats again, the exemption is being
            // used for something it was not granted for.
            let strays: Vec<&str> = src
                .lines()
                .filter(|l| l.contains("vk::Format::"))
                .filter(|l| !l.contains("_S8_UINT"))
                .collect();
            assert!(
                strays.is_empty(),
                "{site} is excused only for combined depth-stencil formats:\n{}",
                strays.join("\n")
            );
        }
    }

    /// Every spelling in the forbidden list must actually be one this crate can
    /// produce, and the list must cover the families `translate` owns — a
    /// missing entry is a hole that never reports itself.
    #[test]
    fn the_forbidden_list_covers_what_translate_owns() {
        let root = crate_src();
        let translate = root.join("backend/vulkan/translate");
        let mut owned = String::new();
        for path in rust_files(&translate) {
            owned.push_str(&fs::read_to_string(&path).unwrap_or_default());
        }
        for spelling in TRANSLATED_STATE_SPELLINGS {
            assert!(
                owned.contains(spelling),
                "{spelling} is forbidden elsewhere but translate never produces \
                 it — either the list is stale or a family lost its home"
            );
        }
        // Bare type names must not be in the list: returning `vk::Format` is
        // not a decision, spelling a variant is.
        for spelling in TRANSLATED_STATE_SPELLINGS {
            assert!(
                spelling.ends_with("::") || *spelling == "ComponentSwizzle",
                "{spelling} would ban naming the type, not deciding a value"
            );
        }
    }

    /// A gate that silently inspects nothing always passes.
    #[test]
    fn the_scanner_walks_the_whole_crate() {
        let root = crate_src();
        let files = rust_files(&root);
        assert!(
            files.len() > 50,
            "expected the full crate, saw {}",
            files.len()
        );
        let names: Vec<_> = files.iter().map(|p| rel(p, &root)).collect();
        assert!(names.contains(&"runtime/metal_draw/mod.rs".to_string()));
        assert!(names.contains(&"backend/vulkan/engine/types.rs".to_string()));
        assert!(names.contains(&"backend/vulkan/translate/pixel.rs".to_string()));
    }

    /// The other half of the crossing: no function outside `translate/` may
    /// turn a raw wire number into an engine state enum.
    ///
    /// This is what would have caught `engine_cull_mode`, `engine_depth_compare`
    /// and `engine_stencil_op` — three arm-for-arm copies of `translate::raster`
    /// tables, each returning an unnamed `Option` decline where the canonical
    /// version returns a named `TranslateReason`, all three invisible to a gate
    /// that only looked for `vk::` spellings.
    #[test]
    fn no_metal_facing_translation_lives_outside_translate() {
        let root = crate_src();
        let mut offenders = Vec::new();
        for path in rust_files(&root) {
            let name = rel(&path, &root);
            if OWNING_DIRS.iter().any(|d| name.starts_with(d))
                || METAL_FACING_ALLOWLIST.iter().any(|(f, _)| *f == name)
            {
                continue;
            }
            let Ok(src) = fs::read_to_string(&path) else {
                continue;
            };
            for (start, body) in items(&src) {
                // Tests build fixtures out of these enums constantly; the gate
                // is about the product path.
                if body.contains("#[cfg(test)]") {
                    continue;
                }
                // Comments describe Apple's enums constantly; only code counts.
                let code: String = body
                    .lines()
                    .filter(|l| !l.trim_start().starts_with("//"))
                    .collect::<Vec<_>>()
                    .join("\n");
                let Some(named) = ENGINE_STATE_ENUMS.iter().find(|e| names_word(&code, e)) else {
                    continue;
                };
                // The enum must be produced BY a wire-value arm, not merely
                // mentioned somewhere in the same item. A hand-rolled Metal→
                // Vulkan state translation names the engine enum inside the arm
                // that decodes the wire value — that is the shape this gate is
                // for. Pairing any mention with any arm across a whole item is
                // unsound once an item is large: `try_metal2vulkan_draw` matches
                // pass load actions at one end and passes `CullMode::None` as the
                // documented fallback to `translate::raster::cull_mode` at the
                // other, which is precisely the routing the gate wants, and the
                // loose pairing reported it as a violation.
                let arms: Vec<&str> = arm_bodies(&body)
                    .into_iter()
                    .filter(|(_, arm_body)| names_word(arm_body, named))
                    .map(|(arm, _)| arm)
                    .collect();
                if !arms.is_empty() {
                    offenders.push(format!(
                        "{name}:{start}: names `{named}` and matches raw wire values:\n    {}",
                        arms.join("\n    ")
                    ));
                }
            }
        }
        assert!(
            offenders.is_empty(),
            "these sites decide what a Metal wire value means outside \
             backend/vulkan/translate — a translation has two halves, and a \
             copy written in this direction spells no `vk::` variant, so the \
             other scan cannot see it. Route them through `translate::` and let \
             the decline be a named TranslateReason:\n{}",
            offenders.join("\n")
        );
    }

    /// Both allowlists must stay honest: entries point at files that exist, and
    /// the Metal-facing one stays empty until something genuinely earns a spot.
    #[test]
    fn the_metal_facing_allowlist_is_exact() {
        let root = crate_src();
        assert!(
            METAL_FACING_ALLOWLIST.is_empty(),
            "no site has yet made an honest case for building pipeline state \
             from a raw number outside translate/ — if one has, state the reason \
             here and say why it is not a Metal decision"
        );
        for (site, reason) in METAL_FACING_ALLOWLIST {
            assert!(
                root.join(site).exists(),
                "allowlisted site {site} no longer exists"
            );
            assert!(!reason.is_empty(), "{site} has no stated reason");
        }
    }

    /// Every enum in the Metal-facing list must be one `translate/` actually
    /// produces — otherwise the list is banning a spelling nothing owns, which
    /// reads as coverage while providing none.
    #[test]
    fn the_engine_state_list_is_owned_by_translate() {
        let root = crate_src();
        let mut owned = String::new();
        for path in rust_files(&root.join("backend/vulkan/translate")) {
            owned.push_str(&fs::read_to_string(&path).unwrap_or_default());
        }
        for name in ENGINE_STATE_ENUMS {
            assert!(
                owned.contains(name),
                "{name} is forbidden outside translate but translate never \
                 produces it — either the list is stale or a family lost its home"
            );
        }
    }

    /// The wire-value detector itself, since a scanner that recognises nothing
    /// passes everything.
    #[test]
    fn wire_value_arms_are_recognised() {
        assert!(is_wire_value_arm("        3 => StencilOp::Replace,"));
        assert!(is_wire_value_arm("    0x41 => Rg16Float,"));
        assert!(is_wire_value_arm(
            "  x if x == S::Rgba8Uint as u32 => Some(V::Rgba8Uint),"
        ));
        assert!(is_wire_value_arm("            0 | 1 => CullMode::None,"));
        // Named patterns are how the *canonical* tables and ordinary matches
        // read; only raw wire values are the tell.
        assert!(!is_wire_value_arm(
            "        CullMode::Front => vk::CullModeFlags::FRONT,"
        ));
        assert!(!is_wire_value_arm("        other => return Err(reason),"));
        assert!(!is_wire_value_arm("        // 3 => Replace, per the SDK"));
        assert!(!is_wire_value_arm("        let x = 3;"));
    }

    /// The arm-scoped pairing, which is what separates a hand-rolled
    /// translation from a large function that happens to mention an engine enum
    /// somewhere else entirely.
    ///
    /// The loose "same item" pairing this replaced reported
    /// `try_metal2vulkan_draw` as an offender: it matches pass load actions in
    /// one statement and passes `CullMode::None` as the documented fallback to
    /// `translate::raster::cull_mode` in another, ~700 lines apart. That is the
    /// routing the gate exists to require, scored as a violation of it.
    #[test]
    fn an_arm_must_itself_produce_the_engine_enum() {
        // A real duplicate: the arm decodes the wire value and names the enum.
        let offender = "\
fn hand_rolled(x: u32) -> CullMode {
    match x {
        1 => CullMode::Front,
        _ => CullMode::None,
    }
}";
        let paired: Vec<&str> = arm_bodies(offender)
            .into_iter()
            .filter(|(_, b)| names_word(b, "CullMode"))
            .map(|(a, _)| a)
            .collect();
        assert_eq!(paired, vec!["1 => CullMode::Front,"]);

        // Incidental co-location: the wire arm decodes something unrelated and
        // the enum is produced by a `translate::` call far away.
        let innocent = "\
fn big(load_action: u16, req: &Req) -> R {
    match load_action {
        x if x == PASS_LOAD_ACTION_CLEAR => {
            seed = clear(req);
        }
        _ => {}
    }
    R {
        cull_mode: raster_or_default(req.cull, translate::raster::cull_mode, CullMode::None),
    }
}";
        assert!(
            arm_bodies(innocent)
                .into_iter()
                .all(|(_, b)| !names_word(&b, "CullMode")),
            "an arm that does not produce the enum must not pair with it"
        );
        // The arm is still recognised as a raw wire-value arm; only the pairing
        // changed, so the gate has not stopped seeing this shape.
        assert_eq!(arm_bodies(innocent).len(), 1);
    }

    /// The word-boundary rule, which is the difference between this gate
    /// reporting three real duplicates and reporting three comments.
    #[test]
    fn engine_enum_names_match_only_whole_identifiers() {
        assert!(names_word("let x: IndexType = a;", "IndexType"));
        assert!(names_word("IndexType", "IndexType"));
        assert!(names_word("engine::IndexType::U16", "IndexType"));
        // Apple's own enums embed ours as a substring.
        assert!(!names_word(
            "0 => Some(2), // MTLIndexTypeUInt16",
            "IndexType"
        ));
        assert!(!names_word("metal::MTLCullMode::Front", "CullMode"));
        assert!(!names_word("let my_cull_mode_x = 1;", "CullMode"));
    }

    /// The item splitter must actually find items, or every scan above is a
    /// loop over nothing.
    #[test]
    fn the_item_splitter_finds_functions() {
        let src = "\
fn a() {
    match v {
        0 => X::One,
    }
}

fn b() -> u32 {
    7
}
";
        let found = items(src);
        assert_eq!(found.len(), 2, "expected two items, got {found:?}");
        assert!(found[0].1.contains("0 => X::One"));
        assert!(!found[1].1.contains("0 => X::One"), "items must not bleed");
    }

    /// `runtime/` decodes the guest's command stream; it must not also decide
    /// what those values are called in Vulkan. Checked separately from the
    /// blanket scan so the message names the actual boundary being crossed.
    #[test]
    fn the_runtime_names_no_vulkan_state_at_all() {
        let root = crate_src();
        let mut offenders = Vec::new();
        for path in rust_files(&root.join("runtime")) {
            let name = rel(&path, &root);
            let Ok(src) = fs::read_to_string(&path) else {
                continue;
            };
            for (i, line) in src.lines().enumerate() {
                if line.contains("vk::") {
                    offenders.push(format!("{name}:{}: {}", i + 1, line.trim()));
                }
            }
        }
        assert!(
            offenders.is_empty(),
            "runtime/ decodes the guest command stream; naming Vulkan types \
             there puts protocol decode and GPU spelling in one place:\n{}",
            offenders.join("\n")
        );
    }
}
