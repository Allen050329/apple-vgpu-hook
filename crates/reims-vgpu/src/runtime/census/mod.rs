//! Always-on proxies and censuses — the measurement half of the ground rules.
//!
//! # Why these are together, and why the sink is not
//!
//! "You cannot fix what you cannot measure" means every bug class earns a
//! log- or test-level proxy that says *this class is happening* without anyone
//! staring at a screenshot. Each module here is one such class, added with the
//! fix it made possible. That is why there are ten of them, and it is correct.
//!
//! What was not correct is where they lived: each arrived as a new sibling of
//! `metal_draw.rs`, so `runtime/` read as though measurement modules were peers
//! of the execution path. They are not — the execution path calls them, never
//! the reverse.
//!
//! [`crate::observe`] deliberately stays outside this directory. It is the
//! **sink**, not a census: `observe::fail` and `observe::off` are the
//! always-on outputs that everything here writes to, and the execution path
//! calls them directly for its own declines. Filing the sink under `census/`
//! would suggest a draw decline is a measurement, which is the distinction the
//! ground rules turn on — a decline must be logged, a measurement must not gate
//! behaviour.
//!
//! # The rule these all obey
//!
//! **Measuring is allowed; branching on the measurement is not.** These modules
//! may count nonzero pixels, sparsity, format volume, cache churn and geometry,
//! and they may write those counts to the always-on log. Nothing in the device
//! or backend may read one back to decide what to present, decode or execute.
//! A proxy that changes behaviour has become a content heuristic.
//!
//! # What each one measures
//!
//! | Module | Class it measures |
//! |---|---|
//! | [`present_proxy`] | present-path thrash: `nz_swing`, `sparse_present`, `mid_switch`, `geom_mismatch`, `capture_fail`, plus the secondary-MRT drop/blend census |
//! | [`srgb_census`] | which rails drop the sRGB transfer function, and how often |
//! | [`sampled_census`] | sampled-source resolution: which rail served a texture bind and which missed |
//! | [`setup_tex_census`] | setup-phase texture staging, before the compositor converges |
//! | [`sample_seed_relation`] | exact relations between a sampled texel and the seed row it came from |
//! | [`sampled_gva_churn`] | how often a sampled guest-VA texture's backing moves |
//! | [`ensure_surface_census`] | surface-ensure outcomes on the present path |
//! | [`view_swizzle_census`] | type-8 view swizzle plans actually bound |
//! | [`writeback_census`] | compute storage-image writeback volume and shape |
//! | [`t11_decline`] | type-11 IOSurface resolution declines, by reason |

pub mod ensure_surface_census;
pub mod present_proxy;
pub(crate) mod sample_seed_relation;
pub mod sampled_census;
pub mod sampled_gva_churn;
pub mod setup_tex_census;
pub mod srgb_census;
pub mod t11_decline;
pub mod view_swizzle_census;
pub mod writeback_census;
