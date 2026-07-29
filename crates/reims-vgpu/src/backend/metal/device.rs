//! MTLDevice probe helpers + Backend trait for device lifecycle.

use crate::backend::metal::runtime::{system_device, system_device_name as runtime_device_name};
use crate::backend::Backend;
use metal::Device;

/// Runtime handle for pure-Rust Metal probes.
pub struct MetalRuntime;

impl MetalRuntime {
    pub fn device() -> Option<&'static Device> {
        system_device()
    }
}

/// Device lifecycle handle; product encode is the C ABI in `ffi`.
#[derive(Debug, Default)]
pub struct MetalBackend {
    ready: bool,
}

impl MetalBackend {
    pub fn new() -> Self {
        Self {
            ready: system_device().is_some(),
        }
    }

    pub fn ready(&self) -> bool {
        self.ready
    }

    pub fn name(&self) -> &'static str {
        "metal"
    }
}

impl Backend for MetalBackend {
    fn reset(&mut self) {
        crate::runtime::icb::clear_icb_cache();
    }
}

pub fn system_device_name() -> Option<String> {
    runtime_device_name()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn system_device() {
        assert!(MetalRuntime::device().is_some());
        assert!(system_device_name().is_some());
        assert!(MetalBackend::new().ready());
    }
}
