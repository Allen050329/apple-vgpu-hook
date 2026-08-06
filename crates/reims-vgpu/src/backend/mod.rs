//! Backend selection seam.
//!
//! - [`metal`] / [`vulkan`] = concrete backends (feature-selected), each
//!   **self-contained** in this crate (Metal via `metal`; Vulkan via `ash` +
//!   [`vulkan::engine`]).
//! - Draws, compute and blits do **not** come through this module. The live
//!   seams are `runtime/draw::try_metal2vulkan_draw` → [`vulkan::engine`]
//!   on the Vulkan rail and `metal::render::render_core_mrt` /
//!   `metal::compute::compute_core` on the Metal rail; the runtime drives them
//!   directly.
//!
//! Metal indices/semantics are canonical (guest wire is serialized Metal).
//! Vulkan-only binding rewrites live only in [`vulkan`].

/// Content hashes for the Metal backend's compiled-object caches.
///
/// Declared here rather than inside `metal` — not a doc link, because that
/// module does not exist on the Vulkan arm, which is the point — and ungated,
/// for one reason: it
/// names nothing from the `metal` crate, and gating it made two test functions
/// that any host can execute run on none of them. Everything under
/// `backend/metal/` is `cfg`-ed out of the arm a Linux host builds, and
/// `cargo test --target aarch64-apple-darwin --no-run` fails at the link step —
/// so while this was a `mod hash` in that tree, the only way to run its tests
/// was to copy the file to `/tmp` and invoke `rustc --test` by hand. That is not
/// a gate, and AGENTS.md recorded it as a workaround rather than as the bug it
/// was.
///
/// The cost is a Vulkan build compiling twenty lines of arithmetic it never
/// calls, which is not a reason to hide a test from the only machine that can
/// run it. `scripts/dead-state` takes the intersection of both arms, so it does
/// not read as dead.
///
/// It stays out of `contract::fnv` for the reason its own doc gives: the fold
/// here is not the shared one, and a caller reaching for the wrong one would
/// produce keys in a different keyspace without anything failing.
pub mod hash;

#[cfg(feature = "backend-metal")]
// `Status` is 264 bytes — six inline `(key, value)` fields, no allocation — and
// it is the `Err` of most of this module's functions, so `result_large_err` and
// `large_enum_variant` fire across it. Boxing is the lint's remedy and it is
// the wrong trade here: the payload is what makes every refusal name the check
// that refused (see `backend::metal::error::Status`), the type is `Copy` and
// compared by value at hundreds of sites, and the cost being complained about
// is stack traffic on a **failure** path. A new error type that is large for no
// such reason should still be boxed rather than added to this exemption.
#[allow(clippy::result_large_err, clippy::large_enum_variant)]
pub mod metal;
#[cfg(feature = "backend-vulkan")]
pub mod vulkan;

/// Guest-lifetime teardown, the one thing a backend owns that the runtime
/// cannot do for it.
///
/// The trait is this small on purpose. It once declared the whole
/// Metal-semantic operation set — texture create/write/read, blit, compute,
/// render, present — and nothing ever called any of it: the runtime drives the
/// backends directly through their own seams, so every one of those methods
/// returned a refusal or a bare `Ok` without touching a GPU.
pub trait Backend {
    /// Drop all state derived from the current guest lifetime.
    ///
    /// Immutable, content-keyed shader/pipeline caches may survive. Guest object
    /// identities, resident images, and aliases of guest memory must not.
    fn reset(&mut self) {}
}

/// Null backend for protocol/device tests without a GPU.
#[derive(Default)]
pub struct NullBackend;

impl Backend for NullBackend {}
