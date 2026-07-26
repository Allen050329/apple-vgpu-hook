//! Host-owned presentation window ([[host-window]]) — a Rust-owned `winit`
//! window with its own `VkSurfaceKHR`/swapchain that replaces QEMU's built-in UI
//! and presents the engine frame directly, keeping the C/QEMU side thin.
//!
//! Gated behind the `host-window` cargo feature (off by default, so the QEMU
//! staticlib link is unchanged until the window is proven).
//!
//! Layers, built incrementally (see `.agents/host-window-plan.md`):
//! - [`input_map`] — pure winit-event → neutral input mapping (this file's
//!   sibling). No window needed; unit-tested off-VM. **Landed.**
//! - present thread — winit event loop on a dedicated thread, `VkSurfaceKHR` on
//!   the engine `VkInstance`, swapchain acquire → blit latest frame → present.
//!   *Next.*
//! - producer glue — feed [`input_map`] output onto the prompt action queue via
//!   the thread-safe `notify_actions` path.

pub mod input_map;
pub mod present;
pub mod viewport;
