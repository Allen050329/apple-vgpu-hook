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

/// The selected `MTLDevice`'s name.
///
/// Gated with the test that reads it, because **no product path on this arm
/// reports which GPU was selected.** The Vulkan arm emits it in its `vk_caps`
/// line (`caps::Snapshot::selection_line`); this arm has the string available
/// and never says it, so a Metal boot's log cannot answer "which device did we
/// pick" at all. That is an observability gap on a pathway no Linux host can
/// boot — recorded here rather than filled, since adding an emission is a
/// behaviour change that would land unverified.
#[cfg(test)]
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
    if page != 0 && addr.is_multiple_of(page) && len.is_multiple_of(page) {
        // A nil here is the device refusing the allocation, and it arrives as
        // `None` rather than as a `Buffer`. This used to read
        // `device.new_buffer_with_bytes_no_copy(..)`, whose return type is
        // `metal::Buffer` — a `NonNull` — so the nil became an invalid value
        // before anything could test it, and the comment that stood here said a
        // null Metal result "behaves as a zero-length Objective-C receiver". The
        // *messaging* does; the Rust wrapper does not, and `.length()` returning
        // zero was reading a value that already had no legal representation.
        let allocated = unsafe {
            crate::backend::metal::raw_metal::new_buffer_no_copy(
                device,
                data as *mut _,
                len as u64,
                MTLResourceOptions::StorageModeShared,
            )
        };
        if let Some(buf) = allocated {
            let actual_len = buf.length();
            let status = no_copy_buffer_length_status(len, actual_len);
            if status.is_ok() {
                return Some(buf);
            }
            // A length Metal did not honour is not an allocation failure, so the
            // copy fallback below is still correct. Make the performance
            // degradation visible once per rejected requested/actual pair.
            if let Some(emit) = crate::observe::Emit::refusal("metal_buffer_copy_fallback", &status)
            {
                emit.fail_once((len as u64) ^ actual_len.rotate_left(32));
            }
        }
    }
    // The copying constructor can refuse too, and for the reason that matters
    // most: it is the one that has to find `len` bytes. `None` reaches the
    // caller as a refusal instead of a buffer that is not one.
    unsafe {
        crate::backend::metal::raw_metal::new_buffer_with_data(
            device,
            data as *const _,
            len as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }
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
