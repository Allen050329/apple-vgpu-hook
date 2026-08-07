//! A Metal object that failed to allocate must be caught as a pointer.
//!
//! `metal-0.33`'s allocators are `msg_send!` with the return typed as the owned
//! wrapper — `Device::new_texture` is `msg_send![self, newTextureWithDescriptor:
//! d] -> metal::Texture` — and `foreign_types` 0.5 declares those wrappers as
//! `struct Texture(NonNull<MTLTexture>)`, built through
//! `NonNull::new_unchecked`. Every one of these Objective-C methods returns
//! **nil when it cannot allocate**, which for a texture or a buffer is what a
//! Metal device does when its VRAM is full.
//!
//! So the failure writes a null pointer into a `NonNull` field. That is an
//! invalid value — undefined behaviour, not a `Texture` the caller can test —
//! and no check placed after the call can recover it. It is the same class as
//! `backend::metal::mtl_enum`'s, where a guest ordinal that names no variant
//! must be rejected *before* it becomes an `MTL*` enum, and it has the same
//! answer: take the pointer raw, test it, wrap it only if it is real.
//!
//! `backend::metal::raw_metal` is where that dance lives. It had always been
//! there for `new_texture_view_swizzled` — the one such API metal-0.33 does not
//! expose, so it had to be hand-written, and writing it out made the nil
//! obvious. Nothing about a swizzled view made it special; the allocators
//! metal-0.33 *does* expose simply hid the pointer, and so skipped the check.
//!
//! # What this asserts
//!
//! No file under `src/backend/metal/` calls a nil-returning metal-0.33
//! allocator, except `raw_metal.rs` itself and the sites named in [`UNCONVERTED`]
//! with a reason.
//!
//! The allocators that return `Result` — `new_library_with_data`,
//! `new_compute_pipeline_state`, `new_render_pipeline_state` — are not in scope:
//! metal-0.33 wraps those in `try_objc!`, so the failure already has a
//! representation and never reaches a `NonNull`.
//!
//! # Why the unconverted list exists rather than being converted away
//!
//! The four entries are fixed-cost objects — a sampler state, a depth-stencil
//! state, a command queue. A nil from those is a malformed descriptor or a
//! device-level failure, not the out-of-memory case this device has to answer
//! faithfully, and they are reached through caches whose conversion is a wider
//! change than the allocation sites. They are listed rather than filtered out so
//! that "these are known and unconverted" is a claim in the tree instead of an
//! absence, which is the difference between a bounded scope and a silent one.

use std::path::{Path, PathBuf};

/// metal-0.33 allocators that return the owned wrapper and answer nil on
/// failure.
///
/// Spelled with the leading `.` so a definition inside `raw_metal` — `pub fn
/// new_texture(` — is not mistaken for a call to metal-0.33's.
const NIL_RETURNING: &[&str] = &[
    ".new_texture(",
    ".new_buffer(",
    ".new_buffer_with_data(",
    ".new_buffer_with_bytes_no_copy(",
    ".new_sampler(",
    ".new_depth_stencil_state(",
    ".new_command_queue(",
    // The encoder and command-buffer vendors. These are worse than the
    // allocators above rather than better: metal-0.33 types them as `&XxxRef`,
    // so a nil becomes a **null reference** — always undefined behaviour, and
    // dereferenced by the very next method call. `commandBuffer` answers nil
    // when the queue will not issue another, which is a pressure refusal, and
    // `renderCommandEncoderWithDescriptor:` answers nil for a pass descriptor
    // Metal rejects.
    ".new_command_buffer(",
    ".new_render_command_encoder(",
    ".new_blit_command_encoder(",
    ".new_compute_command_encoder(",
    ".compute_command_encoder_with_dispatch_type(",
];

/// Files outside `src/backend/metal/` that also reach metal-0.33 directly.
///
/// The encoder vendors are called from `runtime/` too — the Metal ICB rail and
/// the compute session own their own command buffers — so scanning only the
/// backend would report a clean tree while two of the ten sites sat elsewhere.
const EXTRA_SCANNED: &[&str] = &[
    "src/runtime/draw/metal_icb.rs",
    "src/runtime/compute_session.rs",
];

/// A call left unconverted, and why.
struct Unconverted {
    /// Path relative to `src/backend/metal/`.
    file: &'static str,
    method: &'static str,
    why: &'static str,
}

const UNCONVERTED: &[Unconverted] = &[
    Unconverted {
        file: "render.rs",
        method: ".new_depth_stencil_state(",
        why: "a depth-stencil state is a fixed-cost descriptor object, not an \
              allocation sized by guest data; a nil is a malformed descriptor \
              rather than the device being out of memory",
    },
    Unconverted {
        file: "runtime.rs",
        method: ".new_command_queue(",
        why: "one queue per thread, created once and cached; a nil here is a \
              device-level failure that the very next Metal call also reports",
    },
    Unconverted {
        file: "runtime.rs",
        method: ".new_sampler(",
        why: "the cached default sampler: one fixed-cost descriptor object for \
              the whole process, so a nil is a rejected descriptor rather than \
              a device that has run out of memory",
    },
    Unconverted {
        file: "samplers.rs",
        method: ".new_sampler(",
        why: "a sampler state is a fixed-cost descriptor object; its nil means \
              the descriptor was rejected, not that VRAM ran out",
    },
];

fn metal_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src/backend/metal")
}

fn rust_sources(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for entry in std::fs::read_dir(&d).expect("the metal backend must be readable") {
            let path = entry.expect("a readable dir entry").path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "rs") {
                out.push(path);
            }
        }
    }
    out.sort();
    out
}

/// Comments blanked so a doc comment naming `.new_texture(` is not a call.
fn strip_comments(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for line in text.lines() {
        let code = match line.find("//") {
            Some(at) => &line[..at],
            None => line,
        };
        out.push_str(code);
        out.push('\n');
    }
    out
}

#[test]
fn every_nil_returning_metal_allocation_is_checked_as_a_pointer() {
    let dir = metal_dir();
    let crate_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut sources = rust_sources(&dir);
    for extra in EXTRA_SCANNED {
        let p = crate_root.join(extra);
        assert!(
            p.is_file(),
            "EXTRA_SCANNED names {extra}, which is not a file — it moved, and \
             this scan would silently stop reading it"
        );
        sources.push(p);
    }

    // Self-check: a scan that read nothing would report a clean tree. `raw_metal`
    // defines the checked replacements, so its own file must be present and the
    // population must be non-trivial.
    assert!(
        sources.len() > 5,
        "the metal backend scan found {} files, so it is not reading the tree",
        sources.len()
    );
    assert!(
        sources.iter().any(|p| p.ends_with("raw_metal.rs")),
        "raw_metal.rs must be in the scanned set — it is where the checked \
         allocators live, and a scan that cannot see it cannot see their callers"
    );

    let mut unchecked: Vec<String> = Vec::new();
    for path in &sources {
        let rel = path
            .strip_prefix(&dir)
            .or_else(|_| path.strip_prefix(crate_root))
            .expect("a scanned file is under the metal backend or the crate root")
            .to_string_lossy()
            .to_string();
        // `raw_metal` is the one place allowed to reach metal-0.33's allocators,
        // and it does not: it sends the selectors itself. Skipped so its own
        // `pub fn new_buffer(` definitions cannot read as calls.
        if rel == "raw_metal.rs" {
            continue;
        }
        let text = strip_comments(&std::fs::read_to_string(path).expect("a readable source"));
        for (n, line) in text.lines().enumerate() {
            for method in NIL_RETURNING {
                if !line.contains(method) {
                    continue;
                }
                if UNCONVERTED
                    .iter()
                    .any(|a| a.file == rel && a.method == *method)
                {
                    continue;
                }
                unchecked.push(format!("{rel}:{} {}", n + 1, line.trim()));
            }
        }
    }

    assert!(
        unchecked.is_empty(),
        "a metal-0.33 allocator that answers nil is called outside `raw_metal`. \
         Its return type is a `NonNull` wrapper, so the failing allocation \
         becomes an invalid value before anything can test it — an out-of-VRAM \
         device would be undefined behaviour rather than a refusal. Use the \
         checked `raw_metal::new_*` and turn `None` into a typed refusal, or add \
         a row to UNCONVERTED saying why this one cannot run out of memory:\n  {}",
        unchecked.join("\n  ")
    );
}

/// Every row still names a call that is there.
///
/// Without this, converting a site would leave its excuse behind, and the next
/// unchecked allocation added to that file under the same method would inherit
/// a verdict written about a line that no longer exists.
#[test]
fn no_unconverted_metal_allocation_row_is_stale() {
    let dir = metal_dir();
    let mut stale = Vec::new();
    for a in UNCONVERTED {
        let path = dir.join(a.file);
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("UNCONVERTED names {}, which is unreadable: {e}", a.file));
        if !strip_comments(&text).contains(a.method) {
            stale.push(format!("{} no longer calls {}", a.file, a.method));
        }
    }
    assert!(
        stale.is_empty(),
        "an unconverted list row names a call the tree no longer makes. Delete the \
         row:\n  {}",
        stale.join("\n  ")
    );
}

/// Every row explains itself in more than a word.
#[test]
fn every_unconverted_metal_allocation_says_why() {
    for a in UNCONVERTED {
        assert!(
            a.why.len() > 60,
            "{}'s {} does not explain itself: {:?}",
            a.file,
            a.method,
            a.why
        );
    }
}
