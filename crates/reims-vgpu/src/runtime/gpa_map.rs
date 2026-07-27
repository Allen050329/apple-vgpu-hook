//! Guest-physical control-plane writes via HostOps `map_pages` (no raw write_gpa).
//!
//! Product stamp / DeviceInfo / display shared / child HEAD writes map the
//! covering guest page(s), poke host bytes, then unmap. Fail closed when
//! `map_pages` refuses the packed page run (`gpa_write reason=mem_map_pages_refused`).

use crate::runtime::host::{HostMemory, HostOps, MemError};

/// Write `buf` at guest physical `gpa` through a HostOps map_pages view.
pub fn write_bytes<H: HostMemory + HostOps>(
    host: &mut H,
    gpa: u64,
    buf: &[u8],
    page_size: usize,
) -> Result<(), MemError> {
    if buf.is_empty() {
        return Ok(());
    }
    if page_size == 0 || !page_size.is_power_of_two() {
        return Err(MemError::BadArgs);
    }
    let page_size_u = page_size as u64;
    let page_mask = page_size_u - 1;
    let start = gpa & !page_mask;
    let end = gpa
        .checked_add(buf.len() as u64)
        .ok_or(MemError::Overflow)?;
    let mut gpas = Vec::new();
    let mut p = start;
    while p < end {
        gpas.push(p);
        p = p.checked_add(page_size_u).ok_or(MemError::Overflow)?;
    }
    // The GPA list here is packed by construction (`p += page_size`), so a
    // refusal is `map_pages` declining a run it cannot alias — a RAMBlock or
    // MemoryRegion edge — never a gap in the list. Naming it that way keeps the
    // reason a fact about the refusal rather than a guess about the input.
    let Some(ptr) = host.map_pages(&gpas, page_size) else {
        let err = MemError::MapPagesRefused;
        crate::observe::Emit::decline("gpa_write", &err)
            .field("gpa", format!("{gpa:#x}"))
            .field("len", format!("{:#x}", buf.len()))
            .field("pages", gpas.len())
            .fail();
        return Err(err);
    };
    let total = gpas.len() * page_size;
    let off = (gpa - start) as usize;
    if ptr == 0 || off + buf.len() > total {
        host.unmap_pages(ptr, total);
        return Err(MemError::Unmapped);
    }
    // SAFETY: map_pages returned `total` host bytes at ptr; off+len in range.
    unsafe {
        std::ptr::copy_nonoverlapping(buf.as_ptr(), (ptr as *mut u8).add(off), buf.len());
    }
    host.unmap_pages(ptr, total);
    Ok(())
}

/// Write a little-endian u32 at `gpa`.
pub fn write_u32<H: HostMemory + HostOps>(
    host: &mut H,
    gpa: u64,
    v: u32,
    page_size: usize,
) -> Result<(), MemError> {
    write_bytes(host, gpa, &v.to_le_bytes(), page_size)
}

/// Write a little-endian u16 at `gpa`.
pub fn write_u16<H: HostMemory + HostOps>(
    host: &mut H,
    gpa: u64,
    v: u16,
    page_size: usize,
) -> Result<(), MemError> {
    write_bytes(host, gpa, &v.to_le_bytes(), page_size)
}

/// Read `buf.len()` bytes from guest physical `gpa` through map_pages.
pub fn read_bytes<H: HostMemory + HostOps>(
    host: &mut H,
    gpa: u64,
    buf: &mut [u8],
    page_size: usize,
) -> Result<(), MemError> {
    if buf.is_empty() {
        return Ok(());
    }
    if page_size == 0 || !page_size.is_power_of_two() {
        return Err(MemError::BadArgs);
    }
    let page_size_u = page_size as u64;
    let page_mask = page_size_u - 1;
    let start = gpa & !page_mask;
    let end = gpa
        .checked_add(buf.len() as u64)
        .ok_or(MemError::Overflow)?;
    let mut gpas = Vec::new();
    let mut p = start;
    while p < end {
        gpas.push(p);
        p = p.checked_add(page_size_u).ok_or(MemError::Overflow)?;
    }
    let Some(ptr) = host.map_pages(&gpas, page_size) else {
        return Err(MemError::Unmapped);
    };
    let total = gpas.len() * page_size;
    let off = (gpa - start) as usize;
    if ptr == 0 || off + buf.len() > total {
        host.unmap_pages(ptr, total);
        return Err(MemError::Unmapped);
    }
    // SAFETY: map covers total bytes.
    unsafe {
        std::ptr::copy_nonoverlapping((ptr as *const u8).add(off), buf.as_mut_ptr(), buf.len());
    }
    host.unmap_pages(ptr, total);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::host::FakeHost;

    #[test]
    fn write_u32_via_map_pages_aliases_guest() {
        let mut host = FakeHost::new();
        let page = 4096usize;
        let gpa = 0x4000u64;
        host.map_range(gpa, page, 0);
        assert!(write_u32(&mut host, gpa + 8, 0xdead_beef, page).is_ok());
        assert_eq!(host.get_u32(gpa + 8), 0xdead_beef);
    }

    #[test]
    fn write_bytes_crosses_page_boundary() {
        let mut host = FakeHost::new();
        let page = 4096usize;
        host.map_range(0x1000, page * 2, 0);
        let buf = [1u8, 2, 3, 4, 5, 6, 7, 8];
        // Start 4 bytes before page boundary so span covers two pages.
        assert!(write_bytes(&mut host, 0x1ffc, &buf, page).is_ok());
        let mut back = [0u8; 8];
        assert!(host.read_gpa(0x1ffc, &mut back).is_ok());
        assert_eq!(back, buf);
    }
}
