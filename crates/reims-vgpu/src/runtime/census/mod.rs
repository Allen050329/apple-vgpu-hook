//! Always-on declines whose reason needs state the raising site does not hold.
//!
//! # What is here, and what is deliberately not
//!
//! [`crate::observe`] is the **sink**: `observe::fail`, `observe::off` and
//! `observe::Emit` are where every always-on line lands, and the execution path
//! calls them directly for its own declines. Filing the sink under `census/`
//! would suggest a draw decline is a measurement, which is the distinction the
//! ground rules turn on.
//!
//! These four modules are the cases where the *reason* needs state the raising
//! site does not have — a dedup set spanning draws, or a slug vocabulary shared
//! by several call sites — so the line is written here instead. They are still
//! declines. The execution path calls them, never the reverse.
//!
//! # The rule these all obey
//!
//! **Measuring is allowed; branching on the measurement is not.** Nothing in
//! the device or backend may read one of these back to decide what to present,
//! decode or execute. A proxy that changes behaviour has become a content
//! heuristic, which the ground rules forbid outright.
//!
//! # What each one reports
//!
//! | Module | Class it reports |
//! |---|---|
//! | [`present_proxy`] | `secondary_mrt_drop` / `mrt_mask_bind_miss` — a multi-RT draw degraded to single-RT, or a rendered mask that failed to bind at sample time — plus `stale_online_pending` and [`present_proxy::window_publish`], the sole record that a captured frame never reached the host window |
//! | [`srgb_census`] | which rails drop the sRGB transfer function |
//! | [`view_swizzle_census`] | type-8 view swizzles dropped, or served by rewriting texels on the CPU |
//! | [`t11_decline`] | why the type-11 sampled rail declined its zero-copy gather, by reason |
//! | [`exec_resource_table`] | what the guest declares about each resource an `EXEC_INDIRECT2` submission touches |
//!
//! [`exec_resource_table`] is the one entry here that reports guest *input*
//! rather than a device decline, and it qualifies on the same test: the loss is
//! otherwise invisible. The guest's statement that it CPU-wrote a resource is
//! delivered once, inside a table this device stepped over unread, so no counter
//! could separate "the guest never said" from "we discarded what it said".
//!
//! # Adding one
//!
//! A module belongs here only when the loss it names is otherwise invisible. If
//! the refusal already emits a typed decline at the point it refuses, a second
//! count of its *rate* has no claim under the fail-visible rule — and a tally of
//! successful work never had one. Modules and rate-halves have been deleted on
//! exactly that test more often than they have been added; run it before writing
//! the next one.

pub mod exec_resource_table;
pub mod present_proxy;
pub mod srgb_census;
pub mod t11_decline;
pub mod view_swizzle_census;
