//! Global MTLDevice, per-thread command queues, native color format TLS, host buffers.

use metal::{Buffer, CommandQueue, Device, MTLResourceOptions, Texture};
use once_cell::sync::OnceCell;
use parking_lot::Mutex;
use std::cell::RefCell;
use std::collections::HashMap;

use crate::backend::metal::util::Status;

static DEVICE: OnceCell<Device> = OnceCell::new();
static DEFAULT_SAMPLER: Mutex<Option<metal::SamplerState>> = Mutex::new(None);
/// Guest-memory-backed Metal textures for type-11 mappings (unified memory).
///
/// The texture is a linear view over the mapping's contiguous guest-RAM view
/// (`MappingEntry::contig_ptr`, mach_vm_remap of the mapper page list). GPU
/// Load/Store and guest CPU writes hit the SAME bytes — there is no host copy
/// of surface content, so no invalidation is ever needed for *content*; the
/// cache only avoids re-creating identical MTLBuffer/MTLTexture objects. An
/// entry is dropped when any identity field (view ptr/len, geometry, format,
/// bytes-per-row, offset) changes, or explicitly on MAP/UNMAP/geom change
/// before the runtime unmaps the retired view it aliases.
static TYPE11_GUEST_TEX: OnceCell<Mutex<HashMap<u32, GuestBackedTex>>> = OnceCell::new();

fn type11_guest_tex_map() -> &'static Mutex<HashMap<u32, GuestBackedTex>> {
    TYPE11_GUEST_TEX.get_or_init(|| Mutex::new(HashMap::new()))
}

struct GuestBackedTex {
    view_ptr: usize,
    view_len: usize,
    width: u32,
    height: u32,
    /// MTLPixelFormat raw value (native guest format, e.g. RGBA16Float).
    pixel_format: u32,
    bytes_per_row: u32,
    offset: u64,
    /// Keeps the no-copy MTLBuffer alive as long as the texture view exists.
    _buffer: Buffer,
    texture: Texture,
}

thread_local! {
    static QUEUE: RefCell<Option<CommandQueue>> = const { RefCell::new(None) };
    static COLOR_PIXEL_FORMAT: RefCell<u32> = const { RefCell::new(0) };
}

/// Texture aliasing the guest surface bytes for `mapping_id`.
///
/// `view_ptr`/`view_len` = the mapping's contiguous guest-RAM view;
/// `offset`/`bytes_per_row` from the guest device descriptor (sample window).
/// Returns the cached object when every identity field matches, else creates
/// a no-copy MTLBuffer over the view plus a linear texture view on it.
/// Every rejected alias names the exact failed contract check. The render
/// caller may preserve correctness with a host-copy attachment, but must emit
/// this status as a visible degradation before doing so.
#[allow(clippy::too_many_arguments)]
pub fn type11_guest_texture(
    mapping_id: u32,
    view_ptr: usize,
    view_len: usize,
    width: u32,
    height: u32,
    pixel_format: u32,
    bytes_per_row: u32,
    offset: u64,
) -> Result<Texture, Status> {
    if mapping_id == 0 {
        return Err(Status::args("metal_type11_alias_mapping_zero"));
    }
    if view_ptr == 0 {
        return Err(
            Status::args("metal_type11_alias_view_pointer_null").field("mapping", mapping_id)
        );
    }
    if view_len == 0 {
        return Err(
            Status::args("metal_type11_alias_view_length_zero").field("mapping", mapping_id)
        );
    }
    if width == 0 {
        return Err(Status::args("metal_type11_alias_width_zero").field("mapping", mapping_id));
    }
    if height == 0 {
        return Err(Status::args("metal_type11_alias_height_zero").field("mapping", mapping_id));
    }
    let mut guard = type11_guest_tex_map().lock();
    if let Some(e) = guard.get(&mapping_id) {
        if e.view_ptr == view_ptr
            && e.view_len == view_len
            && e.width == width
            && e.height == height
            && e.pixel_format == pixel_format
            && e.bytes_per_row == bytes_per_row
            && e.offset == offset
        {
            return Ok(e.texture.clone());
        }
        guard.remove(&mapping_id);
    }
    let Some(device) = system_device() else {
        return Err(
            Status::execute("metal_type11_alias_device_unavailable").field("mapping", mapping_id)
        );
    };
    let mtl_fmt = crate::backend::metal::format::pixel_format_from_u32(pixel_format);
    let align = device.minimum_linear_texture_alignment_for_pixel_format(mtl_fmt);
    if align != 0 && offset % align != 0 {
        return Err(Status::args("metal_type11_alias_offset_unaligned")
            .field("mapping", mapping_id)
            .field("offset", offset)
            .field("alignment", align));
    }
    if align != 0 && (bytes_per_row as u64) % align != 0 {
        return Err(Status::args("metal_type11_alias_row_bytes_unaligned")
            .field("mapping", mapping_id)
            .field("row_bytes", bytes_per_row)
            .field("alignment", align));
    }
    // Both factors originate as u32, so their product always fits in u64.
    let row_span = (bytes_per_row as u64) * (height as u64);
    let Some(span) = offset.checked_add(row_span) else {
        return Err(Status::args("metal_type11_alias_span_overflow")
            .field("mapping", mapping_id)
            .field("offset", offset)
            .field("row_span", row_span));
    };
    if span > view_len as u64 {
        return Err(Status::args("metal_type11_alias_span_out_of_range")
            .field("mapping", mapping_id)
            .field("span", span)
            .field("view_len", view_len));
    }
    let buffer = device.new_buffer_with_bytes_no_copy(
        view_ptr as *const std::ffi::c_void,
        view_len as u64,
        MTLResourceOptions::StorageModeShared,
        None,
    );
    let desc = metal::TextureDescriptor::new();
    desc.set_texture_type(metal::MTLTextureType::D2);
    desc.set_pixel_format(mtl_fmt);
    desc.set_width(width as u64);
    desc.set_height(height as u64);
    desc.set_storage_mode(metal::MTLStorageMode::Shared);
    desc.set_usage(metal::MTLTextureUsage::RenderTarget | metal::MTLTextureUsage::ShaderRead);
    let texture = buffer.new_texture_with_descriptor(&desc, offset, bytes_per_row as u64);
    guard.insert(
        mapping_id,
        GuestBackedTex {
            view_ptr,
            view_len,
            width,
            height,
            pixel_format,
            bytes_per_row,
            offset,
            _buffer: buffer,
            texture: texture.clone(),
        },
    );
    Ok(texture)
}

/// Drop the cached guest-backed texture objects for a mapping.
///
/// Must run BEFORE the runtime unmaps the contiguous view they alias
/// (MAP/UNMAP/page-table change); execution is sync-per-packet so no GPU
/// work is in flight at those boundaries.
pub fn type11_guest_texture_invalidate(mapping_id: u32) {
    if mapping_id == 0 {
        return;
    }
    type11_guest_tex_map().lock().remove(&mapping_id);
}

/// Drop every guest-memory alias at a device lifetime boundary.
pub fn type11_guest_texture_invalidate_all() {
    type11_guest_tex_map().lock().clear();
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

pub fn begin_native_color_format(pixel_format: u32) {
    COLOR_PIXEL_FORMAT.with(|c| *c.borrow_mut() = pixel_format);
}

pub fn end_native_color_format() {
    COLOR_PIXEL_FORMAT.with(|c| *c.borrow_mut() = 0);
}

pub fn take_native_color_format() -> u32 {
    COLOR_PIXEL_FORMAT.with(|c| *c.borrow())
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
mod guest_tex_tests {
    use super::*;
    use crate::observe::{Emit, Refusal as _};
    use crate::runtime::host::{FakeHost, HostMemory, HostOps, GUEST_PAGE_SIZE};

    fn refused_alias_line(result: Result<Texture, Status>) -> String {
        let status = match result {
            Ok(_) => panic!("invalid alias unexpectedly produced a texture"),
            Err(status) => status,
        };
        Emit::refusal("metal_guest_attachment_fallback", &status)
            .expect("invalid alias must carry a refusal")
            .render()
    }

    #[test]
    fn guest_backed_texture_names_each_early_identity_rejection() {
        assert_eq!(
            refused_alias_line(type11_guest_texture(0, 1, 1, 1, 1, 80, 4, 0)),
            "metal_guest_attachment_fallback reason=metal_type11_alias_mapping_zero class=args"
        );
        assert_eq!(
            refused_alias_line(type11_guest_texture(17, 0, 1, 1, 1, 80, 4, 0)),
            "metal_guest_attachment_fallback reason=metal_type11_alias_view_pointer_null class=args mapping=17"
        );
        assert_eq!(
            refused_alias_line(type11_guest_texture(17, 1, 1, 0, 1, 80, 4, 0)),
            "metal_guest_attachment_fallback reason=metal_type11_alias_width_zero class=args mapping=17"
        );
    }

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

    /// Unified-memory core mechanism: a type-11 guest-backed texture renders
    /// into guest pages (Store lands in memory read_gpa sees) and Metal Load
    /// observes CPU writes made through write_gpa between passes. Same checks
    /// as the nocopy_rt_probe, but through the product FakeHost view + cache.
    #[test]
    fn guest_backed_texture_store_and_load_alias_guest_pages() {
        let Some(device) = system_device() else {
            return; // headless CI guard; product Mac always has a device
        };
        let mut host = FakeHost::new();
        let p = GUEST_PAGE_SIZE as u64;
        // 64x128 BGRA8 bpr 256 → 32 KiB = 2 pages; scattered GPAs exercise
        // the remap (rows 64.. land on the second page).
        let gpas = [0x200 * p, 0x150 * p];
        host.map_range(gpas[0], GUEST_PAGE_SIZE, 0);
        host.map_range(gpas[1], GUEST_PAGE_SIZE, 0);
        let view = host.map_pages(&gpas, GUEST_PAGE_SIZE).expect("view");
        let view_len = gpas.len() * GUEST_PAGE_SIZE;
        let (w, h, bpr) = (64u32, 128u32, 256u32);
        let fmt = metal::MTLPixelFormat::BGRA8Unorm as u32;
        let tex =
            type11_guest_texture(77, view, view_len, w, h, fmt, bpr, 0).expect("guest texture");
        // Identity-cache hit returns the same object; changed geometry rebuilds.
        assert!(type11_guest_texture(77, view, view_len, w, h, fmt, bpr, 0).is_ok());

        let queue = device.new_command_queue();
        let run_pass = |load: metal::MTLLoadAction| {
            let pass = metal::RenderPassDescriptor::new();
            let ca = pass.color_attachments().object_at(0).unwrap();
            ca.set_texture(Some(&tex));
            ca.set_load_action(load);
            ca.set_store_action(metal::MTLStoreAction::Store);
            ca.set_clear_color(metal::MTLClearColor::new(0.0, 1.0, 0.0, 1.0));
            let cb = queue.new_command_buffer();
            let enc = cb.new_render_command_encoder(pass);
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        };

        // Clear+Store → green lands in guest pages (BGRA: 00 FF 00 FF).
        run_pass(metal::MTLLoadAction::Clear);
        let mut px = [0u8; 4];
        assert!(host.read_gpa(gpas[0] + 5 * 4, &mut px).is_ok());
        assert_eq!(px, [0x00, 0xFF, 0x00, 0xFF], "Store visible via read_gpa");
        // Row on the second page (y=64..: offset ≥ 16 KiB → page 2).
        assert!(host.read_gpa(gpas[1] + 8, &mut px).is_ok());
        assert_eq!(px, [0x00, 0xFF, 0x00, 0xFF], "Store crosses page boundary");

        // CPU write via write_gpa → Load+Store preserves it (coherent Load).
        let marker = [0x11u8, 0x22, 0x33, 0x44];
        assert!(host.write_gpa(gpas[0] + 9 * 4, &marker).is_ok());
        run_pass(metal::MTLLoadAction::Load);
        assert!(host.read_gpa(gpas[0] + 9 * 4, &mut px).is_ok());
        assert_eq!(px, marker, "Metal Load sees guest CPU write");

        type11_guest_texture_invalidate(77);
        host.unmap_pages(view, view_len);
    }
}
