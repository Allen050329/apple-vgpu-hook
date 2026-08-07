//! Scattered guest windows → image-copy rectangles.
//!
//! The planner and its findings live in `reims_vgpu_paging::regions` — pure
//! arithmetic with no backend names, which is what lets its tests run on every
//! arm. This module re-exports it under the device's established path.

pub use reims_vgpu_paging::regions::{plan_regions, WindowGeometry, WindowRegion};
