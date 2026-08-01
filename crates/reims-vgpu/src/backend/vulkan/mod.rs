//! Self-contained Vulkan execution backend (build-time alternate to Metal).
//!
//! Ownership mirrors [`crate::backend::metal`]: all host GPU work for this rail
//! lives under `backend/vulkan/`, driven by `ash`. Product draw encode uses the
//! internal [`engine`] (persistent ash context + content-keyed caches). This
//! crate has no external graphics-executor dependency; AIR translation comes
//! from the pinned public `metal2vulkan` crate.
//!
//! The [`Backend`] trait carries only guest-lifetime reset; the live draw seam
//! is `runtime/metal_draw::try_metal2vulkan_draw` → [`engine::execute_draw_request`].
//!
//! [`caps`] classifies the bound host GPU into the four-cell support matrix
//! (unified/discrete memory × has/has-no DMA) that every path here must keep
//! working. Capability decisions belong there, not at call sites.
//!
//! [`translate`] is the matching seam for *state*: decoded Metal formats and
//! pipeline enums become Vulkan ones there and nowhere else, so the same
//! decision cannot be made twice with two different answers.

pub mod caps;
pub mod engine;
pub mod translate;

use crate::backend::Backend;

/// Vulkan-rail backend handle. Holds the ash-facing context when encode lands.
#[derive(Debug, Default)]
pub struct VulkanBackend {
    ready: bool,
}

impl VulkanBackend {
    pub fn new() -> Self {
        // Device/instance spin-up stays lazy until the first real encode path
        // needs it, so off-VM protocol tests can construct the shell without a
        // Vulkan ICD.
        Self { ready: true }
    }

    pub fn name(&self) -> &'static str {
        "vulkan"
    }

    pub fn ready(&self) -> bool {
        self.ready
    }
}

impl Backend for VulkanBackend {
    fn reset(&mut self) {
        engine::reset_guest_state();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn name_is_vulkan() {
        assert_eq!(VulkanBackend::new().name(), "vulkan");
    }
}
