//! Crate-wide observability: the always-on log sink, and the decline
//! vocabulary every subsystem reports failures through.
//!
//! # Why this is not under `runtime/`
//!
//! Fail-visibility is not a runtime concern — `backend/`, `contract/`,
//! `model/` and `host_window/` all reject guest work and all owe the reader a
//! reason. It lived under `runtime/` only because that is where the first
//! caller happened to be, and the result was the lapse this module exists to
//! close: 451 fail sites in `runtime/` against 0 in `backend/metal/`,
//! `contract/` and `qemu/`.
//!
//! `translate/` and `caps/` are the other half of the argument. They are pure —
//! they return typed declines and log nothing, which is correct — so the sink
//! must sit somewhere they can name their reason type without depending on
//! `runtime/`. This module is that place.
//!
//! # The parts
//!
//! - [`sink`] — the always-on writer behind `/tmp/reims-vgpu-fail.log`, its background
//!   thread, flood self-detector and test isolation. Moved here verbatim from
//!   `runtime/draw_log.rs`; the machinery was never the problem, the vocabulary
//!   on top of it was.
//!
//! - [`decline`] — the [`Decline`] trait and the crate-wide slug registry.
//! - [`emit`] — the one builder that renders `reason=<slug> k=v …`, and cannot
//!   produce a line without a reason.
//! - `gate` — static scans that keep the above true (test-only).
//!
//! # The obligation
//!
//! Per `AGENTS.md` I2: every path that rejects, drops, degrades or mis-executes
//! a decoded guest command returns a **registered** typed decline whose slug is
//! unique crate-wide and reaches the sink at some call site. A typed decline
//! nobody logs is still a silent failure — that unchecked handoff is what
//! `gate::every_registered_type_reaches_the_sink` closes.
//!
//! The judgement no gate can make stays with the author: do **not** log
//! speculative returns (a resolver legitimately answering "not ready yet" every
//! poll, a genuinely-unbound `ref==0`). Those flood the log.

pub mod decline;
pub mod emit;
#[cfg(test)]
mod gate;
pub mod sink;

pub use decline::{Decline, DeclineClass, Emission, Refusal, REGISTRY};
/// Re-exported so call sites write `crate::observe::decline_display!(..)`
/// next to the trait it implements, rather than reaching into the submodule.
pub(crate) use decline::decline_display;
pub use emit::{first_sight, Emit};

// The sink's surface is re-exported flat so call sites read `observe::fail(…)`
// rather than `observe::sink::fail(…)`. `sink` stays public for the gate and
// for readers who want the machinery.
pub use sink::{
    bgra_present_stats, bgra_present_stats_scalar, bgra_rgb_stats, fail, line, nonzero_stats, off,
    redirect_logs_for_tests, rgba_rgb_a0_stats, rgba_rgb_stats,
};
pub(crate) use sink::{draw_log_enabled, elapsed_ms};

// Path accessors and the line matcher exist so tests can assert against the
// real sink rather than a mock; production never reads them back.
#[cfg(test)]
pub(crate) use sink::{fail_log_path, line_is, FailCapture};
