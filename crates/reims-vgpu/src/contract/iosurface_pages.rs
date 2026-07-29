//! IOSurface mapper/page-table planning (port of `host/utils/reims-vgpu-iosurface-pages`).

use crate::contract::endian::{ld16, ld32, ld64};
use crate::contract::pixel_format;
use crate::contract::{align_up_u64, checked_add_u64, checked_mul_u64};

pub const U32_SIZE: usize = 4;
pub const U64_SIZE: usize = 8;

/// Minimum typed type-11 object-list descriptor length (geometry prefix).
/// Live blobs are often longer (0x38/0x58) with an unused/constant tail.
/// There is no multi-mip level-record layout on type-11: Metal rejects
/// mipmapped IOSurface textures (`mipmapLevelCount > 1`).
pub const TEXTURE_DESC_MIN_LEN: usize = 0x20;
pub const TEXTURE_DESC_MAPPING_ID: usize = 0x00;
pub const TEXTURE_DESC_OBJECT_REF: usize = 0x10;
pub const TEXTURE_DESC_PIXEL_FORMAT: usize = 0x16;
pub const TEXTURE_DESC_WIDTH: usize = 0x18;
pub const TEXTURE_DESC_HEIGHT: usize = 0x1c;

pub const MAPPER_REQUEST_ENTRY_LEN: usize = 16;
pub const MAPPER_REQUEST_TYPE: usize = 0x00;
pub const MAPPER_REQUEST_MAPPING_ID: usize = 0x04;
pub const MAPPER_REQUEST_RESERVED: usize = 0x08;
pub const MAPPER_REQUEST_MAP: u32 = 1;
pub const MAPPER_REQUEST_UNMAP: u32 = 2;

/*
 * Directed register handoff at do_host_mapping_gated → iosfc producer write:
 * guest leaves mapper device / request type / MappingInternal* in these
 * arm64e xregs (kb + archived reims-vgpu-iosurface-pages format header).
 */
pub const MAPPER_CAPTURE_REG_MAPPER_DEVICE: u32 = 19;
pub const MAPPER_CAPTURE_REG_REQUEST_TYPE: u32 = 21;
pub const MAPPER_CAPTURE_REG_MAPPING_INTERNAL: u32 = 22;

pub const ROW_BYTES_ALIGN: u64 = 128;
pub const DEVICE_PLANE_DESC_LEN: usize = 0x40;
pub const DEVICE_PLANE_OFFSET: usize = 0x08;
pub const DEVICE_PLANE_BASE: usize = 0x0c;
pub const DEVICE_PLANE_SIZE: usize = 0x10;
pub const DEVICE_PLANE_DIMS: usize = 0x14;
pub const DEVICE_PLANE_BPR: usize = 0x1c;
pub const DEVICE_PLANE_BPE: usize = 0x20;

pub const DEVICE_DESC_LEN: usize = 0x200;
pub const DEVICE_DESC_PIXEL_FORMAT: usize = 0x04;
pub const DEVICE_DESC_BASE_OFFSET: usize = 0x08;
pub const DEVICE_DESC_ALLOC_SIZE: usize = 0x10;
pub const DEVICE_DESC_DIMS: usize = 0x14;
pub const DEVICE_DESC_BPR: usize = 0x1c;
pub const DEVICE_DESC_BPE: usize = 0x20;
pub const DEVICE_DESC_PLANE_COUNT: usize = 0x24;
pub const DEVICE_DESC_PLANES: usize = 0x40;

/// Arm64e page shift/size — fixtures and C ABI defaults that still assume arm.
/// Product paths must use `page_size_of(state.page_shift)` / `*_shift` APIs.
pub const PAGE_SHIFT_ARM64E: u32 = crate::contract::gva::PAGE_SHIFT_ARM64E;
pub const PAGE_SIZE_ARM64E: u64 = 1u64 << PAGE_SHIFT_ARM64E;
/// x86 page shift/size.
pub const PAGE_SHIFT_X86: u32 = crate::contract::gva::PAGE_SHIFT_X86;
pub const PAGE_SIZE_X86: u64 = 1u64 << PAGE_SHIFT_X86;

#[inline]
pub fn page_size_of(page_shift: u32) -> u64 {
    1u64 << page_shift
}

#[inline]
pub fn page_offset_mask(page_shift: u32) -> u64 {
    page_size_of(page_shift) - 1
}

pub const PAGE_ENTRY_VALID: u32 = 0x1;
pub const PAGE_ENTRY_PFN_SHIFT: u32 = 2;

pub const MAPPING_INTERNAL_BACKPTR: u64 = 0x18;
pub const MAPPING_INTERNAL_ID: u64 = 0x30;
pub const MAPPING_INTERNAL_DESC_PTR: u64 = 0x38;
pub const MAPPING_INTERNAL_SIZE: u64 = 0x40;
pub const MAPPING_INTERNAL_EXPECTED_SIZE: u32 = 0x200;
pub const MAPPING_INTERNAL_PAGE_FIELD_48: u64 = 0x48;
pub const MAPPING_INTERNAL_PAGE_FIELD_50: u64 = 0x50;
pub const MAPPING_INTERNAL_PAGE_COUNT: u64 = 0x70;
pub const MAPPING_PAGE_TABLE_FROM_F48: u64 = 0xb8;
pub const MAPPING_PAGE_TABLE_FROM_F50: u64 = 0x28;

pub const ARM_KERNEL_VA_MASK: u64 = 0xffffff00_00000000;
pub const ARM_KERNEL_VA_BASE: u64 = 0xfffffe00_00000000;
/// x86_64 Darwin canonical kernel half (bits 63:47 all ones in 48-bit VA).
pub const X86_KERNEL_VA_MIN: u64 = 0xffff8000_00000000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Ok,
    ErrArgs(&'static str),
    ErrShortDescriptor(&'static str),
    ErrOverflow(&'static str),
    ErrNotKernelVa(&'static str),
    ErrInternalRead(&'static str),
    ErrInternalOwner(&'static str),
    ErrInternalMappingId(&'static str),
    ErrInternalSize(&'static str),
    ErrInternalFields(&'static str),
    ErrPageCount(&'static str),
    ErrPageTableRead(&'static str),
    ErrPageEntry(&'static str),
    ErrNoPageTable(&'static str),
    ErrSpanRange(&'static str),
}

impl crate::observe::Refusal for Status {
    fn refusal(&self) -> Option<&'static str> {
        match self {
            Self::Ok => None,
            Self::ErrArgs(reason)
            | Self::ErrShortDescriptor(reason)
            | Self::ErrOverflow(reason)
            | Self::ErrNotKernelVa(reason)
            | Self::ErrInternalRead(reason)
            | Self::ErrInternalOwner(reason)
            | Self::ErrInternalMappingId(reason)
            | Self::ErrInternalSize(reason)
            | Self::ErrInternalFields(reason)
            | Self::ErrPageCount(reason)
            | Self::ErrPageTableRead(reason)
            | Self::ErrPageEntry(reason)
            | Self::ErrNoPageTable(reason)
            | Self::ErrSpanRange(reason) => Some(reason),
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        let class = match self {
            Self::Ok => return Vec::new(),
            Self::ErrArgs(_) => "args",
            Self::ErrShortDescriptor(_) => "short_descriptor",
            Self::ErrOverflow(_) => "overflow",
            Self::ErrNotKernelVa(_) => "not_kernel_va",
            Self::ErrInternalRead(_) => "internal_read",
            Self::ErrInternalOwner(_) => "internal_owner",
            Self::ErrInternalMappingId(_) => "internal_mapping_id",
            Self::ErrInternalSize(_) => "internal_size",
            Self::ErrInternalFields(_) => "internal_fields",
            Self::ErrPageCount(_) => "page_count",
            Self::ErrPageTableRead(_) => "page_table_read",
            Self::ErrPageEntry(_) => "page_entry",
            Self::ErrNoPageTable(_) => "no_page_table",
            Self::ErrSpanRange(_) => "span_range",
        };
        vec![("class", class.to_string())]
    }
}

/// Memory access callbacks for mapper/page-table reads.
pub trait PagesMemory {
    fn read(&self, address: u64, dst: &mut [u8]) -> bool;
    fn is_kernel_va(&self, address: u64) -> bool {
        guest_kernel_va(address)
    }
    fn is_ram_gpa(&self, _address: u64) -> bool {
        true
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TextureDescriptor {
    pub mapping_id64: u64,
    pub mapping_id: u32,
    pub object_ref: u32,
    pub has_plane_index: bool,
    pub plane_index: u32,
    pub pixel_format: u16,
    pub width: u32,
    pub height: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MapperRequestEntry {
    pub request_type: u32,
    pub mapping_id: u32,
    pub reserved: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MapperInternalFields {
    pub internal_kva: u64,
    pub has_mapper_device: bool,
    pub mapper_device_kva: u64,
    pub owner_kva: u64,
    pub mapping_id: u32,
    pub internal_size: u32,
    pub page_field_48: u64,
    pub page_field_50: u64,
    pub raw_page_count: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PageTablePlan {
    pub entries: Vec<u32>,
    pub page_table_kva: u64,
    pub min_size: u64,
    pub required_pages: u64,
    pub cached: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PageFragment {
    pub surface_offset: u64,
    pub gpa: u64,
    pub page_index: u32,
    pub page_offset: u32,
    pub length: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeviceSurfaceRecord {
    pub pixel_format: u32,
    pub base_offset: u32,
    pub alloc_size: u32,
    pub width: u32,
    pub height: u32,
    pub bytes_per_row: u32,
    pub bytes_per_element: u16,
    pub plane_count: u8,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DevicePlaneRecord {
    pub plane_offset: u32,
    pub plane_base: u32,
    pub plane_size: u32,
    pub width: u32,
    pub height: u32,
    pub bytes_per_row: u32,
    pub bytes_per_element: u16,
}

pub fn arm_kernel_va(address: u64) -> bool {
    (address & ARM_KERNEL_VA_MASK) == ARM_KERNEL_VA_BASE
}

pub fn x86_kernel_va(address: u64) -> bool {
    address >= X86_KERNEL_VA_MIN
}

/// Guest kernel VA: arm64e TTBR1 window **or** x86_64 Darwin high half.
pub fn guest_kernel_va(address: u64) -> bool {
    arm_kernel_va(address) || x86_kernel_va(address)
}

pub fn span_page_count_shift(min_size: u64, page_shift: u32) -> u64 {
    if min_size == 0 {
        1
    } else {
        ((min_size - 1) >> page_shift) + 1
    }
}

pub fn format_bytes_per_pixel(pixel_format: u16) -> Option<u32> {
    // Match pixel_format storage set / resource-resolve iosurface_bpp.
    pixel_format::bytes_per_pixel(pixel_format)
}

pub fn sample_window(
    _plane_index: u32,
    pixel_format: u16,
    width: u32,
    height: u32,
) -> Option<(u64, u32, u64)> {
    if width == 0 || height == 0 {
        return None;
    }
    let bpp = format_bytes_per_pixel(pixel_format)?;
    let tight = checked_mul_u64(width as u64, bpp as u64)?;
    let bpr = align_up_u64(tight, ROW_BYTES_ALIGN)?;
    if bpr > u32::MAX as u64 {
        return None;
    }
    let span_end = checked_mul_u64(bpr, height as u64)?;
    Some((0, bpr as u32, span_end))
}

pub fn decode_device_surface(bytes: &[u8]) -> Option<DeviceSurfaceRecord> {
    if bytes.len() < DEVICE_DESC_LEN {
        return None;
    }
    let dims = ld64(&bytes[DEVICE_DESC_DIMS..]);
    Some(DeviceSurfaceRecord {
        pixel_format: ld32(&bytes[DEVICE_DESC_PIXEL_FORMAT..]),
        base_offset: ld32(&bytes[DEVICE_DESC_BASE_OFFSET..]),
        alloc_size: ld32(&bytes[DEVICE_DESC_ALLOC_SIZE..]),
        width: ((dims >> 8) & 0xffffff) as u32,
        height: ((dims >> 40) & 0xffffff) as u32,
        bytes_per_row: ld32(&bytes[DEVICE_DESC_BPR..]),
        bytes_per_element: ld16(&bytes[DEVICE_DESC_BPE..]),
        plane_count: bytes[DEVICE_DESC_PLANE_COUNT],
    })
}

pub fn decode_device_plane(bytes: &[u8]) -> Option<DevicePlaneRecord> {
    if bytes.len() < DEVICE_PLANE_DESC_LEN {
        return None;
    }
    let dims = ld64(&bytes[DEVICE_PLANE_DIMS..]);
    Some(DevicePlaneRecord {
        plane_offset: ld32(&bytes[DEVICE_PLANE_OFFSET..]),
        plane_base: ld32(&bytes[DEVICE_PLANE_BASE..]),
        plane_size: ld32(&bytes[DEVICE_PLANE_SIZE..]),
        width: ((dims >> 8) & 0xffffff) as u32,
        height: ((dims >> 40) & 0xffffff) as u32,
        bytes_per_row: ld32(&bytes[DEVICE_PLANE_BPR..]),
        bytes_per_element: ld16(&bytes[DEVICE_PLANE_BPE..]),
    })
}

pub fn device_desc_plane(desc: &[u8], plane_index: u32) -> Option<(DevicePlaneRecord, u32)> {
    if desc.len() < DEVICE_DESC_LEN {
        return None;
    }
    let plane_count = desc[DEVICE_DESC_PLANE_COUNT] as u32;
    if plane_count == 0 || plane_count > 8 || plane_index >= plane_count {
        return None;
    }
    let plane_off = DEVICE_DESC_PLANES + (plane_index as usize) * DEVICE_PLANE_DESC_LEN;
    if plane_off + DEVICE_PLANE_DESC_LEN > desc.len() {
        return None;
    }
    let plane = decode_device_plane(&desc[plane_off..plane_off + DEVICE_PLANE_DESC_LEN])?;
    Some((plane, plane_count))
}

pub fn sample_window_from_device_plane(
    plane: &DevicePlaneRecord,
    pixel_format: u16,
    width: u32,
    height: u32,
) -> Option<(u64, u32, u64)> {
    if width == 0 || height == 0 {
        return None;
    }
    let bpp = format_bytes_per_pixel(pixel_format)?;
    let tight = checked_mul_u64(width as u64, bpp as u64)?;
    if plane.bytes_per_row == 0 || (plane.bytes_per_row as u64) < tight {
        return None;
    }
    let mut surface_off = plane.plane_offset as u64;
    if surface_off == 0 {
        surface_off = plane.plane_base as u64;
    }
    let plane_bytes = checked_mul_u64(plane.bytes_per_row as u64, (height - 1) as u64)?;
    let plane_bytes = checked_add_u64(plane_bytes, tight)?;
    let span_end = checked_add_u64(surface_off, plane_bytes)?;
    if plane.plane_size != 0 && (plane.plane_size as u64) < plane_bytes {
        return None;
    }
    if (plane.width != 0 && plane.width < width) || (plane.height != 0 && plane.height < height) {
        return None;
    }
    Some((surface_off, plane.bytes_per_row, span_end))
}

pub fn sample_window_from_device_surface(
    surf: &DeviceSurfaceRecord,
    pixel_format: u16,
    width: u32,
    height: u32,
) -> Option<(u64, u32, u64)> {
    if width == 0 || height == 0 {
        return None;
    }
    let bpp = format_bytes_per_pixel(pixel_format)?;
    let tight = checked_mul_u64(width as u64, bpp as u64)?;
    if surf.bytes_per_row == 0 || (surf.bytes_per_row as u64) < tight {
        return None;
    }
    if (surf.width != 0 && surf.width < width) || (surf.height != 0 && surf.height < height) {
        return None;
    }
    let rows_span = checked_mul_u64(surf.bytes_per_row as u64, (height - 1) as u64)?;
    let rows_span = checked_add_u64(rows_span, tight)?;
    let span_end = checked_add_u64(surf.base_offset as u64, rows_span)?;
    if surf.alloc_size != 0 && span_end > surf.alloc_size as u64 {
        return None;
    }
    Some((surf.base_offset as u64, surf.bytes_per_row, span_end))
}

pub fn sample_window_prefer_device(
    desc: Option<&[u8]>,
    plane_index: Option<u32>,
    pixel_format: u16,
    width: u32,
    height: u32,
) -> Option<(u64, u32, u64, bool)> {
    if let Some(desc) = desc {
        if desc.len() >= DEVICE_DESC_LEN {
            if let Some(surf) = decode_device_surface(desc) {
                // A wire-carried plane index (type-5 record `+0x20`) names its
                // record directly — the only key that separates same-geometry
                // planes (v0a8 Y plane 0 vs alpha plane 2). Geometry scan
                // below stays for callers whose wire has no index (type-11).
                if let Some(p) = plane_index {
                    if surf.plane_count > 0 {
                        if let Some((cand, _)) = device_desc_plane(desc, p) {
                            if let Some((off, bpr, end)) =
                                sample_window_from_device_plane(&cand, pixel_format, width, height)
                            {
                                return Some((off, bpr, end, true));
                            }
                        }
                    }
                }
                if surf.plane_count == 0 {
                    if let Some((off, bpr, end)) =
                        sample_window_from_device_surface(&surf, pixel_format, width, height)
                    {
                        return Some((off, bpr, end, true));
                    }
                } else if let Some(bpp) = format_bytes_per_pixel(pixel_format) {
                    let mut matches = 0u32;
                    let mut plane = DevicePlaneRecord::default();
                    for p in 0..surf.plane_count.min(8) {
                        if let Some((cand, _)) = device_desc_plane(desc, p as u32) {
                            if cand.width == width
                                && cand.height == height
                                && (cand.bytes_per_element == 0
                                    || cand.bytes_per_element as u32 == bpp)
                            {
                                matches += 1;
                                plane = cand;
                            }
                        }
                    }
                    if matches == 1 {
                        if let Some((off, bpr, end)) =
                            sample_window_from_device_plane(&plane, pixel_format, width, height)
                        {
                            return Some((off, bpr, end, true));
                        }
                    }
                }
            }
        }
    }
    // Invent fallback. RE (`allocateBackingHandle`): type-4 `length` is the
    // page-aligned allocation written at desc+0 (independent of plane w/h/bpr
    // filled from per-plane getters). We stash that as device_desc.alloc_size.
    // The device-surface path already rejects span_end > alloc_size; invent must
    // not invent a larger span past that wire allocation (old invent ignored it
    // → host claimed a plane-sized window over fewer mapped pages).
    let (off, bpr, end) = sample_window(plane_index.unwrap_or(0), pixel_format, width, height)?;
    if let Some(desc) = desc {
        if desc.len() >= DEVICE_DESC_LEN {
            if let Some(surf) = decode_device_surface(desc) {
                if surf.alloc_size != 0 && end > surf.alloc_size as u64 {
                    return None;
                }
            }
        }
    }
    Some((off, bpr, end, false))
}

pub fn entry_gpa_shift(entry: u32, page_shift: u32) -> Option<u64> {
    if (entry & PAGE_ENTRY_VALID) == 0 {
        return None;
    }
    Some(((entry >> PAGE_ENTRY_PFN_SHIFT) as u64) << page_shift)
}

pub fn mapper_request_entry_offset(index: u32) -> u64 {
    (index as u64) * MAPPER_REQUEST_ENTRY_LEN as u64
}

pub fn mapper_request_published_entry_offset(producer: u32) -> Option<u64> {
    if producer == 0 {
        None
    } else {
        Some(mapper_request_entry_offset(producer - 1))
    }
}

pub fn required_entry_count(
    fields: &MapperInternalFields,
    min_size: u64,
    page_shift: u32,
) -> Result<u32, Status> {
    let pages64 = fields.raw_page_count;
    let required_pages = span_page_count_shift(min_size, page_shift);
    // Guest page count is authoritative — no product 4096-page ceiling.
    // Fail only on zero, span coverage, or host-unaddressable entry vectors
    // (process addressability for `Vec<u32>` of entries — not a MiB budget).
    if pages64 == 0 || pages64 < required_pages || pages64 > u32::MAX as u64 {
        return Err(Status::ErrPageCount("iosurface_page_count_invalid"));
    }
    let entry_bytes = pages64.saturating_mul(4);
    if usize::try_from(entry_bytes)
        .ok()
        .filter(|&n| n <= isize::MAX as usize)
        .is_none()
    {
        return Err(Status::ErrPageCount(
            "iosurface_page_count_host_addressability",
        ));
    }
    Ok(pages64 as u32)
}

pub fn decode_texture_descriptor(bytes: &[u8]) -> Result<TextureDescriptor, Status> {
    if bytes.len() < TEXTURE_DESC_MIN_LEN {
        return Err(Status::ErrShortDescriptor(
            "iosurface_texture_descriptor_short",
        ));
    }
    Ok(TextureDescriptor {
        mapping_id64: ld64(&bytes[TEXTURE_DESC_MAPPING_ID..]),
        mapping_id: ld32(&bytes[TEXTURE_DESC_MAPPING_ID..]),
        object_ref: ld32(&bytes[TEXTURE_DESC_OBJECT_REF..]),
        has_plane_index: false,
        plane_index: 0,
        pixel_format: ld16(&bytes[TEXTURE_DESC_PIXEL_FORMAT..]),
        width: ld32(&bytes[TEXTURE_DESC_WIDTH..]),
        height: ld32(&bytes[TEXTURE_DESC_HEIGHT..]),
    })
}

pub fn decode_mapper_request_entry(bytes: &[u8]) -> Result<MapperRequestEntry, Status> {
    if bytes.len() < MAPPER_REQUEST_ENTRY_LEN {
        return Err(Status::ErrShortDescriptor("iosurface_mapper_request_short"));
    }
    Ok(MapperRequestEntry {
        request_type: ld32(&bytes[MAPPER_REQUEST_TYPE..]),
        mapping_id: ld32(&bytes[MAPPER_REQUEST_MAPPING_ID..]),
        reserved: ld64(&bytes[MAPPER_REQUEST_RESERVED..]),
    })
}

fn read_u32(mem: &dyn PagesMemory, address: u64) -> Option<u32> {
    let mut bytes = [0u8; 4];
    if address > u64::MAX - 3 {
        return None;
    }
    if !mem.read(address, &mut bytes) {
        return None;
    }
    Some(ld32(&bytes))
}

fn read_u64(mem: &dyn PagesMemory, address: u64) -> Option<u64> {
    let mut bytes = [0u8; 8];
    if address > u64::MAX - 7 {
        return None;
    }
    if !mem.read(address, &mut bytes) {
        return None;
    }
    Some(ld64(&bytes))
}

fn read_u32_at(mem: &dyn PagesMemory, base: u64, offset: u64) -> Option<u32> {
    let address = checked_add_u64(base, offset)?;
    read_u32(mem, address)
}

fn read_u64_at(mem: &dyn PagesMemory, base: u64, offset: u64) -> Option<u64> {
    let address = checked_add_u64(base, offset)?;
    read_u64(mem, address)
}

pub fn read_mapper_identity(
    mem: &dyn PagesMemory,
    internal_kva: u64,
    has_mapper_device: bool,
    mapper_device_kva: u64,
) -> Result<MapperInternalFields, Status> {
    if !mem.is_kernel_va(internal_kva) {
        return Err(Status::ErrNotKernelVa(
            "iosurface_mapper_internal_kva_invalid",
        ));
    }
    if has_mapper_device && !mem.is_kernel_va(mapper_device_kva) {
        return Err(Status::ErrNotKernelVa(
            "iosurface_mapper_device_kva_invalid",
        ));
    }
    let owner_kva = read_u64_at(mem, internal_kva, MAPPING_INTERNAL_BACKPTR).ok_or(
        Status::ErrInternalRead("iosurface_mapper_internal_owner_read"),
    )?;
    let mapping_id = read_u32_at(mem, internal_kva, MAPPING_INTERNAL_ID).ok_or(
        Status::ErrInternalRead("iosurface_mapper_internal_mapping_id_read"),
    )?;
    let internal_size = read_u32_at(mem, internal_kva, MAPPING_INTERNAL_SIZE).ok_or(
        Status::ErrInternalRead("iosurface_mapper_internal_size_read"),
    )?;
    Ok(MapperInternalFields {
        internal_kva,
        has_mapper_device,
        mapper_device_kva,
        owner_kva,
        mapping_id,
        internal_size,
        page_field_48: 0,
        page_field_50: 0,
        raw_page_count: 0,
    })
}

pub fn read_mapper_internal(
    mem: &dyn PagesMemory,
    internal_kva: u64,
    has_mapper_device: bool,
    mapper_device_kva: u64,
) -> Result<MapperInternalFields, Status> {
    let mut fields = read_mapper_identity(mem, internal_kva, has_mapper_device, mapper_device_kva)?;
    fields.page_field_48 = read_u64_at(mem, internal_kva, MAPPING_INTERNAL_PAGE_FIELD_48).ok_or(
        Status::ErrInternalRead("iosurface_mapper_page_field_48_read"),
    )?;
    fields.page_field_50 = read_u64_at(mem, internal_kva, MAPPING_INTERNAL_PAGE_FIELD_50).ok_or(
        Status::ErrInternalRead("iosurface_mapper_page_field_50_read"),
    )?;
    fields.raw_page_count = read_u64_at(mem, internal_kva, MAPPING_INTERNAL_PAGE_COUNT)
        .ok_or(Status::ErrInternalRead("iosurface_mapper_page_count_read"))?;
    Ok(fields)
}

pub fn read_internal_desc_ptr(mem: &dyn PagesMemory, internal_kva: u64) -> Result<u64, Status> {
    let desc_kva = read_u64_at(mem, internal_kva, MAPPING_INTERNAL_DESC_PTR).ok_or(
        Status::ErrInternalRead("iosurface_mapper_device_desc_pointer_read"),
    )?;
    if desc_kva == 0 {
        return Err(Status::ErrInternalFields(
            "iosurface_mapper_device_desc_pointer_zero",
        ));
    }
    if !mem.is_kernel_va(desc_kva) {
        return Err(Status::ErrInternalFields(
            "iosurface_mapper_device_desc_pointer_invalid",
        ));
    }
    Ok(desc_kva)
}

pub fn validate_mapper_internal(
    mem: &dyn PagesMemory,
    expected_mapping_id: u32,
    fields: &MapperInternalFields,
) -> Status {
    if !mem.is_kernel_va(fields.internal_kva) {
        return Status::ErrNotKernelVa("iosurface_validate_internal_kva_invalid");
    }
    if fields.mapping_id != expected_mapping_id {
        return Status::ErrInternalMappingId("iosurface_validate_mapping_id_mismatch");
    }
    if fields.internal_size != MAPPING_INTERNAL_EXPECTED_SIZE {
        return Status::ErrInternalSize("iosurface_validate_internal_size_mismatch");
    }
    if fields.has_mapper_device {
        if !mem.is_kernel_va(fields.mapper_device_kva) {
            return Status::ErrNotKernelVa("iosurface_validate_mapper_device_kva_invalid");
        }
        if fields.owner_kva != fields.mapper_device_kva {
            return Status::ErrInternalOwner("iosurface_validate_internal_owner_mismatch");
        }
    }
    Status::Ok
}

fn read_table_entries(
    mem: &dyn PagesMemory,
    table_kva: u64,
    pages: u32,
    page_shift: u32,
) -> Result<Vec<u32>, Status> {
    let mut entries = Vec::with_capacity(pages as usize);
    for i in 0..pages {
        let entry = read_u32_at(mem, table_kva, (i as u64) * U32_SIZE as u64)
            .ok_or(Status::ErrPageTableRead("iosurface_page_table_entry_read"))?;
        let gpa = entry_gpa_shift(entry, page_shift)
            .ok_or(Status::ErrPageEntry("iosurface_page_table_entry_invalid"))?;
        if !mem.is_ram_gpa(gpa) {
            return Err(Status::ErrPageEntry("iosurface_page_table_gpa_not_ram"));
        }
        entries.push(entry);
    }
    Ok(entries)
}

pub fn build_table_plan(
    mem: &dyn PagesMemory,
    expected_mapping_id: u32,
    fields: &MapperInternalFields,
    min_size: u64,
    page_shift: u32,
) -> Result<PageTablePlan, Status> {
    let st = validate_mapper_internal(mem, expected_mapping_id, fields);
    if st != Status::Ok {
        return Err(st);
    }
    if !mem.is_kernel_va(fields.page_field_48) && !mem.is_kernel_va(fields.page_field_50) {
        return Err(Status::ErrInternalFields(
            "iosurface_page_table_fields_invalid",
        ));
    }
    let required_pages = span_page_count_shift(min_size, page_shift);
    let pages = required_entry_count(fields, min_size, page_shift)?;

    let mut candidates = [0u64; 2];
    let mut candidate_valid = [false; 2];
    let mut first_candidate_failure = None;
    if mem.is_kernel_va(fields.page_field_48) {
        match read_u64_at(mem, fields.page_field_48, MAPPING_PAGE_TABLE_FROM_F48) {
            Some(v) if mem.is_kernel_va(v) => {
                candidates[0] = v;
                candidate_valid[0] = true;
            }
            Some(_) => {
                first_candidate_failure = Some(Status::ErrNoPageTable(
                    "iosurface_page_table_pointer_48_invalid",
                ));
            }
            None => {
                first_candidate_failure = Some(Status::ErrPageTableRead(
                    "iosurface_page_table_pointer_48_read",
                ));
            }
        }
    }
    if mem.is_kernel_va(fields.page_field_50) {
        match read_u64_at(mem, fields.page_field_50, MAPPING_PAGE_TABLE_FROM_F50) {
            Some(v) if mem.is_kernel_va(v) => {
                candidates[1] = v;
                candidate_valid[1] = true;
            }
            Some(_) if first_candidate_failure.is_none() => {
                first_candidate_failure = Some(Status::ErrNoPageTable(
                    "iosurface_page_table_pointer_50_invalid",
                ));
            }
            None if first_candidate_failure.is_none() => {
                first_candidate_failure = Some(Status::ErrPageTableRead(
                    "iosurface_page_table_pointer_50_read",
                ));
            }
            _ => {}
        }
    }

    let mut saw_kernel = false;
    let mut first_table_failure = None;
    for i in 0..2 {
        let table_kva = candidates[i];
        if !candidate_valid[i] {
            continue;
        }
        saw_kernel = true;
        match read_table_entries(mem, table_kva, pages, page_shift) {
            Ok(entries) => {
                return Ok(PageTablePlan {
                    entries,
                    page_table_kva: table_kva,
                    min_size,
                    required_pages,
                    cached: false,
                });
            }
            Err(e) => {
                if first_table_failure.is_none() {
                    first_table_failure = Some(e);
                }
            }
        }
    }
    if !saw_kernel {
        return Err(first_candidate_failure.unwrap_or(Status::ErrNoPageTable(
            "iosurface_page_table_candidate_missing",
        )));
    }
    Err(first_table_failure
        .or(first_candidate_failure)
        .unwrap_or(Status::ErrNoPageTable(
            "iosurface_page_table_failure_unattributed",
        )))
}

/// **Arm64e only.** Prefer [`plan_span_shift`] with device page_shift.
pub fn plan_span_arm64e(
    mem: &dyn PagesMemory,
    table: &PageTablePlan,
    offset: u64,
    length: u32,
) -> Result<Vec<PageFragment>, Status> {
    plan_span_shift(mem, table, offset, length, PAGE_SHIFT_ARM64E)
}

/// Explicit page_shift (12 = x86, 14 = arm64e). Product paths use this.
pub fn plan_span_shift(
    mem: &dyn PagesMemory,
    table: &PageTablePlan,
    offset: u64,
    length: u32,
    page_shift: u32,
) -> Result<Vec<PageFragment>, Status> {
    if page_shift == 0 || page_shift > 30 {
        return Err(Status::ErrArgs("iosurface_span_page_shift_invalid"));
    }
    if length == 0 {
        return Ok(Vec::new());
    }
    let page_size = page_size_of(page_shift);
    let off_mask = page_offset_mask(page_shift);
    let end = checked_add_u64(offset, length as u64)
        .ok_or(Status::ErrOverflow("iosurface_span_end_overflow"))?;
    let table_bytes = (table.entries.len() as u64) * page_size;
    if end > table_bytes {
        return Err(Status::ErrSpanRange("iosurface_span_out_of_range"));
    }
    let mut fragments = Vec::new();
    let mut pos = offset;
    while pos < end {
        let page = (pos >> page_shift) as u32;
        let page_off = (pos & off_mask) as u32;
        if page as usize >= table.entries.len() {
            return Err(Status::ErrPageEntry(
                "iosurface_span_page_index_out_of_range",
            ));
        }
        let page_gpa = entry_gpa_shift(table.entries[page as usize], page_shift)
            .ok_or(Status::ErrPageEntry("iosurface_span_page_entry_invalid"))?;
        if !mem.is_ram_gpa(page_gpa) {
            return Err(Status::ErrPageEntry("iosurface_span_gpa_not_ram"));
        }
        let mut chunk = page_size - page_off as u64;
        if chunk > end - pos {
            chunk = end - pos;
        }
        if chunk > u32::MAX as u64 {
            return Err(Status::ErrOverflow("iosurface_span_chunk_length_overflow"));
        }
        let gpa = checked_add_u64(page_gpa, page_off as u64)
            .ok_or(Status::ErrOverflow("iosurface_span_gpa_overflow"))?;
        fragments.push(PageFragment {
            surface_offset: pos,
            gpa,
            page_index: page,
            page_offset: page_off,
            length: chunk as u32,
        });
        pos += chunk;
    }
    Ok(fragments)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::iosurface_pages::{PAGE_SHIFT_ARM64E, PAGE_SIZE_ARM64E};
    use crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    use crate::observe::Refusal;

    use std::collections::HashMap;

    struct MapMem {
        map: HashMap<u64, u8>,
    }
    impl MapMem {
        fn new() -> Self {
            Self {
                map: HashMap::new(),
            }
        }
        fn put_u32(&mut self, a: u64, v: u32) {
            for (i, b) in v.to_le_bytes().iter().enumerate() {
                self.map.insert(a + i as u64, *b);
            }
        }
        fn put_u64(&mut self, a: u64, v: u64) {
            for (i, b) in v.to_le_bytes().iter().enumerate() {
                self.map.insert(a + i as u64, *b);
            }
        }
    }
    impl PagesMemory for MapMem {
        fn read(&self, address: u64, dst: &mut [u8]) -> bool {
            for (i, s) in dst.iter_mut().enumerate() {
                match self.map.get(&(address + i as u64)) {
                    Some(b) => *s = *b,
                    None => return false,
                }
            }
            true
        }
        fn is_kernel_va(&self, address: u64) -> bool {
            arm_kernel_va(address)
        }
    }

    #[test]
    fn status_refusal_separates_control_flow_from_exact_failures() {
        assert_eq!(Status::Ok.refusal(), None);
        assert!(
            crate::observe::Emit::refusal("mapper_resolve_fail", &Status::Ok).is_none(),
            "success must not be representable as a failure line"
        );

        let texture = decode_texture_descriptor(&[]).unwrap_err();
        let request = decode_mapper_request_entry(&[]).unwrap_err();
        assert_eq!(
            texture.refusal(),
            Some("iosurface_texture_descriptor_short")
        );
        assert_eq!(request.refusal(), Some("iosurface_mapper_request_short"));
        assert_ne!(
            texture.refusal(),
            request.refusal(),
            "two distinct short-record checks must not collapse to one reason"
        );
        assert_eq!(
            crate::observe::Emit::refusal("mapper_resolve_fail", &texture)
                .unwrap()
                .field("mapping", 7)
                .render(),
            "mapper_resolve_fail reason=iosurface_texture_descriptor_short \
             class=short_descriptor mapping=7"
        );
    }

    #[test]
    fn table_entry_failure_outranks_an_unreadable_alternative_pointer() {
        let internal = ARM_KERNEL_VA_BASE + 0x10_000;
        let field_48 = ARM_KERNEL_VA_BASE + 0x20_000;
        let field_50 = ARM_KERNEL_VA_BASE + 0x30_000;
        let table = ARM_KERNEL_VA_BASE + 0x40_000;
        let mut mem = MapMem::new();

        // The 0x48 candidate is unreadable. The 0x50 candidate resolves to a
        // real table, whose first entry is invalid. The result must describe
        // the candidate that was actually walked, not the irrelevant earlier
        // pointer read.
        mem.put_u64(field_50 + MAPPING_PAGE_TABLE_FROM_F50, table);
        mem.put_u32(table, 0);
        let fields = MapperInternalFields {
            internal_kva: internal,
            mapping_id: 3,
            internal_size: MAPPING_INTERNAL_EXPECTED_SIZE,
            page_field_48: field_48,
            page_field_50: field_50,
            raw_page_count: 1,
            ..MapperInternalFields::default()
        };

        let error =
            build_table_plan(&mem, 3, &fields, PAGE_SIZE_ARM64E, PAGE_SHIFT_ARM64E).unwrap_err();
        assert_eq!(error.refusal(), Some("iosurface_page_table_entry_invalid"));
    }

    #[test]
    fn sample_window_packed() {
        let (off, bpr, end) = sample_window(0, MTL_FORMAT_BGRA8_UNORM, 200, 100).unwrap();
        assert_eq!(off, 0);
        assert_eq!(bpr, 896);
        assert_eq!(end, 896 * 100);
    }

    #[test]
    fn texture_desc_and_geometry() {
        let mut bytes = [0u8; 0x20];
        bytes[0] = 3; // mapping id
                      // format BGRA at 0x16
        bytes[0x16] = 0x50;
        // width 64 height 32
        bytes[0x18] = 64;
        bytes[0x1c] = 32;
        let d = decode_texture_descriptor(&bytes).unwrap();
        assert_eq!(d.mapping_id, 3);
        assert_eq!(d.pixel_format, MTL_FORMAT_BGRA8_UNORM);
        assert_eq!(d.width, 64);
        assert_eq!(d.height, 32);
    }

    #[test]
    fn entry_gpa_and_span() {
        assert!(entry_gpa_shift(0, PAGE_SHIFT_ARM64E).is_none());
        let e = (5 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
        assert_eq!(
            entry_gpa_shift(e, PAGE_SHIFT_ARM64E).unwrap(),
            (5u64) << PAGE_SHIFT_ARM64E
        );
        assert_eq!(span_page_count_shift(0, PAGE_SHIFT_ARM64E), 1);
        assert_eq!(span_page_count_shift(1, PAGE_SHIFT_ARM64E), 1);
        assert_eq!(
            span_page_count_shift(PAGE_SIZE_ARM64E + 1, PAGE_SHIFT_ARM64E),
            2
        );
    }

    #[test]
    fn plan_span_fragments() {
        let e = (9 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
        let table = PageTablePlan {
            entries: vec![e, e],
            page_table_kva: 0,
            min_size: 0,
            required_pages: 2,
            cached: false,
        };
        let mem = MapMem::new();
        let frags = plan_span_arm64e(&mem, &table, PAGE_SIZE_ARM64E - 16, 32).unwrap();
        assert_eq!(frags.len(), 2);
        assert_eq!(frags[0].length, 16);
        assert_eq!(frags[1].length, 16);
    }

    #[test]
    fn kernel_va_and_identity() {
        assert!(arm_kernel_va(ARM_KERNEL_VA_BASE + 0x1000));
        assert!(!arm_kernel_va(0x1000));
        assert!(x86_kernel_va(X86_KERNEL_VA_MIN + 0x1000));
        assert!(!x86_kernel_va(0x1000));
        assert!(guest_kernel_va(ARM_KERNEL_VA_BASE + 1));
        assert!(guest_kernel_va(X86_KERNEL_VA_MIN + 1));
        let mut m = MapMem::new();
        let kva = ARM_KERNEL_VA_BASE + 0x10000;
        m.put_u64(kva + MAPPING_INTERNAL_BACKPTR, kva);
        m.put_u32(kva + MAPPING_INTERNAL_ID, 1);
        m.put_u32(kva + MAPPING_INTERNAL_SIZE, MAPPING_INTERNAL_EXPECTED_SIZE);
        let f = read_mapper_identity(&m, kva, false, 0).unwrap();
        assert_eq!(f.mapping_id, 1);
        assert_eq!(validate_mapper_internal(&m, 1, &f), Status::Ok);
    }

    #[test]
    fn property_fuzz_sample_window() {
        for w in [1u32, 2, 64, 200, 1920] {
            for h in [1u32, 2, 100] {
                let r = sample_window(0, MTL_FORMAT_BGRA8_UNORM, w, h);
                if let Some((off, bpr, end)) = r {
                    assert_eq!(off, 0);
                    assert_eq!(bpr % 128, 0);
                    assert!(end >= w as u64 * 4);
                }
            }
        }
    }

    /// Pack device-plane dims word: elemW@0, width u24@1, elemH@4, height u24@5.
    fn pack_plane_dims(width: u32, height: u32) -> u64 {
        ((width as u64 & 0xffffff) << 8) | ((height as u64 & 0xffffff) << 40)
    }

    /// Invent must not invent past wire alloc_size (type-4 `length`).
    /// RE: allocateBackingHandle writes length@0 independently of plane dims.
    #[test]
    fn sample_window_invent_rejects_span_past_alloc_size() {
        use crate::contract::endian::{st32, st64};
        use crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;

        // Device desc: 1024×1024 dims, alloc only 384*4096 = 0x180000.
        let mut desc = vec![0u8; DEVICE_DESC_LEN];
        st32(&mut desc[DEVICE_DESC_ALLOC_SIZE..], 0x18_0000);
        st32(
            &mut desc[DEVICE_DESC_PIXEL_FORMAT..],
            MTL_FORMAT_BGRA8_UNORM as u32,
        );
        let dims = ((1024u64) << 8) | ((1024u64) << 40);
        st64(&mut desc[DEVICE_DESC_DIMS..], dims);
        // bpr too small for 1024 BGRA → device-surface path rejects → invent.
        st32(&mut desc[DEVICE_DESC_BPR..], 64);
        desc[DEVICE_DESC_PLANE_COUNT] = 0;

        // Invent would need 1024*4096 > alloc → None (fail closed, no height lie).
        assert!(
            sample_window_prefer_device(Some(&desc), None, MTL_FORMAT_BGRA8_UNORM, 1024, 1024)
                .is_none()
        );
        // Within alloc: invent ok.
        let (off, bpr, end, from_dev) =
            sample_window_prefer_device(Some(&desc), None, MTL_FORMAT_BGRA8_UNORM, 1024, 384)
                .expect("within alloc");
        assert!(!from_dev);
        assert_eq!(off, 0);
        assert_eq!(bpr, 4096);
        assert_eq!(end, 384 * 4096);
    }

    #[test]
    fn sample_window_prefer_device_biplanar_geometry() {
        use crate::contract::endian::{st16, st32, st64};
        use crate::contract::pixel_format::{MTL_FORMAT_R8_UNORM, MTL_FORMAT_RG8_UNORM};

        let mut desc = vec![0u8; DEVICE_DESC_LEN];
        st32(&mut desc[DEVICE_DESC_ALLOC_SIZE..], 0x20000);
        desc[DEVICE_DESC_PLANE_COUNT] = 2;
        // Plane 0: Y 16×8 R8 bpr=64 offset=512 size=512
        let p0 = DEVICE_DESC_PLANES;
        st32(&mut desc[p0 + DEVICE_PLANE_OFFSET..], 512);
        st32(&mut desc[p0 + DEVICE_PLANE_SIZE..], 512);
        st64(&mut desc[p0 + DEVICE_PLANE_DIMS..], pack_plane_dims(16, 8));
        st32(&mut desc[p0 + DEVICE_PLANE_BPR..], 64);
        st16(&mut desc[p0 + DEVICE_PLANE_BPE..], 1);
        // Plane 1: UV 8×4 RG8 bpr=64 offset=1024 size=256
        let p1 = DEVICE_DESC_PLANES + DEVICE_PLANE_DESC_LEN;
        st32(&mut desc[p1 + DEVICE_PLANE_OFFSET..], 1024);
        st32(&mut desc[p1 + DEVICE_PLANE_SIZE..], 256);
        st64(&mut desc[p1 + DEVICE_PLANE_DIMS..], pack_plane_dims(8, 4));
        st32(&mut desc[p1 + DEVICE_PLANE_BPR..], 64);
        st16(&mut desc[p1 + DEVICE_PLANE_BPE..], 2);

        let (off_y, bpr_y, end_y, from_dev) =
            sample_window_prefer_device(Some(&desc), None, MTL_FORMAT_R8_UNORM, 16, 8).unwrap();
        assert!(from_dev);
        assert_eq!(off_y, 512);
        assert_eq!(bpr_y, 64);
        // exclusive last-row end: 512 + 7*64 + 16
        assert_eq!(end_y, 512 + 7 * 64 + 16);

        let (off_uv, bpr_uv, end_uv, from_dev_uv) =
            sample_window_prefer_device(Some(&desc), None, MTL_FORMAT_RG8_UNORM, 8, 4).unwrap();
        assert!(from_dev_uv);
        assert_eq!(off_uv, 1024);
        assert_eq!(bpr_uv, 64);
        assert_eq!(end_uv, 1024 + 3 * 64 + 16);

        // Ambiguous dims → invent (zero matches if dims don't hit a plane).
        let (off_inv, bpr_inv, _, from_inv) =
            sample_window_prefer_device(Some(&desc), None, MTL_FORMAT_R8_UNORM, 4, 4).unwrap();
        assert!(!from_inv);
        assert_eq!(off_inv, 0);
        assert_eq!(bpr_inv % 128, 0);
    }

    /// v0a8 (biplanar video + alpha) shape from the live apple.com hero: the
    /// Y and alpha planes share geometry and bpe, so the geometry scan is
    /// ambiguous by construction — only an explicit wire plane index (type-5
    /// record `+0x20`) separates them.
    #[test]
    fn sample_window_plane_index_selects_among_same_geometry_planes() {
        use crate::contract::endian::{st16, st32, st64};
        use crate::contract::pixel_format::{MTL_FORMAT_R8_UNORM, MTL_FORMAT_RG8_UNORM};

        // Live shape (scaled): Y 946×350 @32 bpr 960; UV 473×175 @336032
        // bpr 960 bpe 2; alpha 946×350 @504992 bpr 960 bpe 1.
        let mut desc = vec![0u8; DEVICE_DESC_LEN];
        st32(&mut desc[DEVICE_DESC_ALLOC_SIZE..], 843_776);
        desc[DEVICE_DESC_PLANE_COUNT] = 3;
        let planes = [
            (32u32, 336_000u32, 946u32, 350u32, 960u32, 1u16),
            (336_032, 168_000, 473, 175, 960, 2),
            (504_992, 336_000, 946, 350, 960, 1),
        ];
        for (i, (off, size, w, h, bpr, bpe)) in planes.iter().enumerate() {
            let base = DEVICE_DESC_PLANES + i * DEVICE_PLANE_DESC_LEN;
            st32(&mut desc[base + DEVICE_PLANE_OFFSET..], *off);
            st32(&mut desc[base + DEVICE_PLANE_SIZE..], *size);
            st64(
                &mut desc[base + DEVICE_PLANE_DIMS..],
                pack_plane_dims(*w, *h),
            );
            st32(&mut desc[base + DEVICE_PLANE_BPR..], *bpr);
            st16(&mut desc[base + DEVICE_PLANE_BPE..], *bpe);
        }

        // Indexed selection: each plane record by its wire index.
        let y = sample_window_prefer_device(Some(&desc), Some(0), MTL_FORMAT_R8_UNORM, 946, 350)
            .unwrap();
        assert_eq!((y.0, y.1, y.3), (32, 960, true));
        let uv = sample_window_prefer_device(Some(&desc), Some(1), MTL_FORMAT_RG8_UNORM, 473, 175)
            .unwrap();
        assert_eq!((uv.0, uv.1, uv.3), (336_032, 960, true));
        let a = sample_window_prefer_device(Some(&desc), Some(2), MTL_FORMAT_R8_UNORM, 946, 350)
            .unwrap();
        assert_eq!((a.0, a.1, a.3), (504_992, 960, true));

        // No index: Y geometry matches plane 0 AND plane 2 → ambiguity rule
        // falls back to invent (never a silent wrong-plane bind).
        let (off_inv, bpr_inv, _, from_inv) =
            sample_window_prefer_device(Some(&desc), None, MTL_FORMAT_R8_UNORM, 946, 350).unwrap();
        assert!(!from_inv);
        assert_eq!(off_inv, 0);
        assert_eq!(bpr_inv, 1024);

        // Out-of-range index falls back the same way, never a wrong record.
        let (_, _, _, from_bad) =
            sample_window_prefer_device(Some(&desc), Some(7), MTL_FORMAT_R8_UNORM, 946, 350)
                .unwrap();
        assert!(!from_bad);
    }
}
