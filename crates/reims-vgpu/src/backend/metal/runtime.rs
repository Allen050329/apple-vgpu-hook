//! Global MTLDevice, per-thread command queues, native color format TLS, host buffers.
//!
//! `new_buffer_from_host` is the only `newBufferWithBytesNoCopy` left, and it
//! aliases **host** allocations — the CPU-staged vertex/fragment/compute byte
//! vectors this crate owns. It is not a guest-RAM alias and must not become
//! one: the type-11 attachment cache that did alias guest pages
//! (`mach_vm_remap` view → no-copy MTLBuffer → linear texture view) is deleted,
//! because a page the host GPU can read is one it can write.

use metal::{Buffer, CommandQueue, Device, MTLResourceOptions};
use once_cell::sync::OnceCell;
use parking_lot::Mutex;
use std::cell::RefCell;

use crate::backend::metal::util::Status;

static DEVICE: OnceCell<Device> = OnceCell::new();
static DEFAULT_SAMPLER: Mutex<Option<metal::SamplerState>> = Mutex::new(None);
thread_local! {
    static QUEUE: RefCell<Option<CommandQueue>> = const { RefCell::new(None) };
}

pub fn system_device() -> Option<&'static Device> {
    DEVICE
        .get_or_try_init(|| Device::system_default().ok_or(()))
        .ok()
}

pub fn system_device_name() -> Option<String> {
    system_device().map(|d| d.name().to_string())
}

/// Per-worker-thread command queue (never one shared process-global queue).
pub fn thread_queue(device: &Device) -> CommandQueue {
    QUEUE.with(|q| {
        if let Some(existing) = q.borrow().as_ref() {
            return existing.clone();
        }
        let queue = device.new_command_queue();
        *q.borrow_mut() = Some(queue.clone());
        queue
    })
}

fn no_copy_buffer_length_status(requested_len: usize, actual_len: u64) -> Status {
    if actual_len == requested_len as u64 {
        Status::OK
    } else {
        Status::execute("metal_buffer_no_copy_length_mismatch")
            .field("requested_len", requested_len)
            .field("actual_len", actual_len)
    }
}

/// Prefer no-copy when host pointer+length are page-aligned (Metal contract).
/// Fall back to a copy otherwise. Caller owns host bytes for command-buffer lifetime.
pub fn new_buffer_from_host(device: &Device, data: *const u8, len: usize) -> Option<Buffer> {
    if data.is_null() || len == 0 {
        return None;
    }
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize };
    let addr = data as usize;
    if page != 0 && addr % page == 0 && len % page == 0 {
        let buf = device.new_buffer_with_bytes_no_copy(
            data as *const _,
            len as u64,
            MTLResourceOptions::StorageModeShared,
            None,
        );
        let actual_len = buf.length();
        let status = no_copy_buffer_length_status(len, actual_len);
        if status.is_ok() {
            return Some(buf);
        }
        // A null Metal result behaves as a zero-length Objective-C receiver.
        // Keep the correct copy fallback, but make the performance degradation
        // visible once for each rejected requested/actual length pair.
        if let Some(emit) = crate::observe::Emit::refusal("metal_buffer_copy_fallback", &status) {
            emit.fail_once((len as u64) ^ actual_len.rotate_left(32));
        }
    }
    Some(device.new_buffer_with_data(
        data as *const _,
        len as u64,
        MTLResourceOptions::StorageModeShared,
    ))
}

pub fn cached_default_sampler(device: &Device) -> metal::SamplerState {
    let mut guard = DEFAULT_SAMPLER.lock();
    if let Some(s) = guard.as_ref() {
        return s.clone();
    }
    let desc = metal::SamplerDescriptor::new();
    desc.set_min_filter(metal::MTLSamplerMinMagFilter::Linear);
    desc.set_mag_filter(metal::MTLSamplerMinMagFilter::Linear);
    desc.set_address_mode_s(metal::MTLSamplerAddressMode::ClampToEdge);
    desc.set_address_mode_t(metal::MTLSamplerAddressMode::ClampToEdge);
    desc.set_normalized_coordinates(true);
    let sampler = device.new_sampler(&desc);
    *guard = Some(sampler.clone());
    sampler
}

#[cfg(test)]
mod no_copy_buffer_tests {
    use super::*;
    use crate::observe::{Emit, Refusal as _};

    #[test]
    fn rejected_no_copy_buffer_names_the_copy_fallback() {
        let status = no_copy_buffer_length_status(0x4000, 0);
        assert_eq!(
            status.refusal(),
            Some("metal_buffer_no_copy_length_mismatch")
        );
        assert_eq!(
            Emit::refusal("metal_buffer_copy_fallback", &status)
                .expect("length mismatch must be a refusal")
                .render(),
            "metal_buffer_copy_fallback reason=metal_buffer_no_copy_length_mismatch \
             class=execute requested_len=16384 actual_len=0"
        );
    }
}
