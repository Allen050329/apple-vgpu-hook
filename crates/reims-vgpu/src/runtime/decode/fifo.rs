//! Root/child FIFO wire constants, and the two child commands decoded here.
//!
//! This module holds the parts of the FIFO contract the live drain path reads:
//! the child opcode table, the resource-list / invalidate / synchronize record
//! layout, the display-descriptor timing entries, and the `EXEC_INDIRECT2`
//! header offsets. Packet framing itself — reading a ring, walking headers and
//! stamps, writing back head — lives in `runtime/drain/mod.rs`, which does it
//! against live guest memory and reports each failure as a `PacketFault`.

use crate::contract::endian::{ld32, st16, st32};

// --- child opcodes and record layout (static RE of the PVG command table) ---

/// PVG table: CmdUnmapMemory (not map — MapMemory2 is `0x39`).
pub const CHILD_OP_UNMAP_MEMORY: u16 = 0x22;
/// PVG: CmdInvalidateResources.
pub const CHILD_OP_INVALIDATE_RESOURCES: u16 = 0x34;
/// PVG: CmdSynchronizeResources.
pub const CHILD_OP_SYNCHRONIZE_RESOURCES: u16 = 0x35;
/// PVG: CmdMapMemory2 (task GPU-VA map).
pub const CHILD_OP_MAP_MEMORY2: u16 = 0x39;
pub const CHILD_OP_CONFIG_40: u16 = 0x40;

/// CmdInvalidateResources / CmdSynchronizeResources shared header (RE pageBacking).
pub const CHILD_RESOURCE_LIST_TASK_ID: u32 = 0x00;
pub const CHILD_RESOURCE_LIST_COUNT: u32 = 0x04;
pub const CHILD_RESOURCE_LIST_HEADER_LEN: u32 = 8;
/// Per-object record on Invalidate: `{object_id u32}` + 4 validity-op bytes.
pub const CHILD_INVALIDATE_RECORD_LEN: u32 = 8;
/// Per-object record on Synchronize: `{object_id u32}` only (no validity ops).
pub const CHILD_SYNCHRONIZE_RECORD_LEN: u32 = 4;
/// Hardcoded pageon second dword from `pageBacking` (LE bytes `01 00 00 01`).
///
/// Not a free-form bitfield. PVG host `invalidateResources:` treats the four
/// bytes after `ref` as:
/// `clear_host_valid | set_host_valid | clear_guest_valid | set_guest_valid`.
/// Pageon = clear hostValid + set guestValid (host cache stale; guest pages live).
pub const CHILD_INVALIDATE_PAGEON_FLAGS: u32 = 0x0100_0001;
/// Cap decoded list entries (guest pageBacking hardcodes count=1; generic parse).
pub const CHILD_RESOURCE_LIST_MAX_COUNT: u32 = 256;

pub const DISPLAY_DESC_TIMING_BASE: u64 = 0x210;
pub const DISPLAY_DESC_TIMING_STRIDE: u32 = 0x10;
pub const DISPLAY_TIMING_WIDTH: u32 = 0x00;
pub const DISPLAY_TIMING_HEIGHT: u32 = 0x02;
pub const DISPLAY_TIMING_REFRESH: u32 = 0x04;
pub const DISPLAY_TIMING_TAIL0: u32 = 0x08;
pub const DISPLAY_TIMING_TAIL1: u32 = 0x0c;
pub const DISPLAY_TIMING_REFRESH_FRAC_BITS: u32 = 16;

pub const CHILD_EXEC_INDIRECT_TASK_ID: u32 = 0x00;
pub const CHILD_EXEC_INDIRECT_RESOURCE_COUNT: u32 = 0x04;
pub const CHILD_EXEC_INDIRECT_CMDBUF_COUNT: u32 = 0x08;
pub const CHILD_EXEC_INDIRECT_HEADER_LEN: u32 = 12;
pub const CHILD_EXEC_INDIRECT_RESOURCE_DESC_LEN: u32 = 24;
pub const CHILD_EXEC_INDIRECT_CMDBUF_DESC_LEN: u32 = 16;
pub const CHILD_EXEC_INDIRECT_CMDBUF_GVA: u32 = 0x00;
pub const CHILD_EXEC_INDIRECT_CMDBUF_LENGTH: u32 = 0x08;

// --- display-descriptor timing entries ---

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DisplayTimingEntry {
    pub width: u16,
    pub height: u16,
    pub refresh_1616: u32,
    pub tail0: u32,
    pub tail1: u32,
}

fn bounded_entry_offset_from_base(
    index: u32,
    base: u64,
    byte_capacity: u64,
    entry_len: u32,
) -> Option<u64> {
    if entry_len == 0 {
        return None;
    }
    let entry_offset = index as u64 * entry_len as u64;
    let offset = base.checked_add(entry_offset)?;
    if offset > byte_capacity || byte_capacity - offset < entry_len as u64 {
        None
    } else {
        Some(offset)
    }
}

pub fn display_refresh_hz_1616(refresh_hz: u32) -> Option<u32> {
    if refresh_hz > (u32::MAX >> DISPLAY_TIMING_REFRESH_FRAC_BITS) {
        return None;
    }
    Some(refresh_hz << DISPLAY_TIMING_REFRESH_FRAC_BITS)
}

pub fn encode_display_timing_entry(entry: &DisplayTimingEntry, dst: &mut [u8]) -> bool {
    if dst.len() < DISPLAY_DESC_TIMING_STRIDE as usize {
        return false;
    }
    st16(&mut dst[DISPLAY_TIMING_WIDTH as usize..], entry.width);
    st16(&mut dst[DISPLAY_TIMING_HEIGHT as usize..], entry.height);
    st32(
        &mut dst[DISPLAY_TIMING_REFRESH as usize..],
        entry.refresh_1616,
    );
    st32(&mut dst[DISPLAY_TIMING_TAIL0 as usize..], entry.tail0);
    st32(&mut dst[DISPLAY_TIMING_TAIL1 as usize..], entry.tail1);
    true
}

pub fn display_timing_entry_offset(index: u32, byte_capacity: u64) -> Option<u64> {
    bounded_entry_offset_from_base(
        index,
        DISPLAY_DESC_TIMING_BASE,
        byte_capacity,
        DISPLAY_DESC_TIMING_STRIDE,
    )
}

/// Validity ops packed after object_id in a CmdInvalidateResources record.
///
/// Wire layout (PVG host + guest pageon hardcode): four **u8** fields, not a bit mask.
/// Non-zero means apply that op to the resource's hostValid/guestValid state.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct InvalidateValidityOps {
    pub clear_host_valid: u8,
    pub set_host_valid: u8,
    pub clear_guest_valid: u8,
    pub set_guest_valid: u8,
}

impl InvalidateValidityOps {
    /// Decode LE dword as four validity-op bytes (`0x01000001` → clr host + set guest).
    pub fn from_le_dword(flags: u32) -> Self {
        let b = flags.to_le_bytes();
        Self {
            clear_host_valid: b[0],
            set_host_valid: b[1],
            clear_guest_valid: b[2],
            set_guest_valid: b[3],
        }
    }

    pub fn to_le_dword(self) -> u32 {
        u32::from_le_bytes([
            self.clear_host_valid,
            self.set_host_valid,
            self.clear_guest_valid,
            self.set_guest_valid,
        ])
    }

    /// Pageon hardcode: clr hostValid + set guestValid.
    pub const PAGEON: Self = Self {
        clear_host_valid: 1,
        set_host_valid: 0,
        clear_guest_valid: 0,
        set_guest_valid: 1,
    };
}

/// One CmdInvalidateResources object record (RE: `pageBacking` second `getCommandBytes(8)`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct InvalidateResourceRecord {
    pub object_id: u32,
    /// LE dword form of the four validity-op bytes (see [`InvalidateValidityOps`]).
    pub flags: u32,
    pub ops: InvalidateValidityOps,
}

/// FIFO CmdInvalidateResources (0x34) payload (RE pageBacking + live plen=16).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct InvalidateResourcesCommand {
    pub task_id: u32,
    pub count: u32,
    pub records: Vec<InvalidateResourceRecord>,
}

/// FIFO CmdSynchronizeResources (0x35) payload (RE synchronizeForUnwire + live plen=12).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SynchronizeResourcesCommand {
    pub task_id: u32,
    pub count: u32,
    pub object_ids: Vec<u32>,
}

/// Decode CmdInvalidateResources: header `{task_id, count}` + `count × {object_id, flags}`.
///
/// Guest `pageBacking` always writes count=1 and flags=`0x1000001` on this kext; decoder
/// still accepts count>1 when the payload is long enough (forward-compatible).
pub fn decode_invalidate_resources(payload: &[u8]) -> Option<InvalidateResourcesCommand> {
    if payload.len() < CHILD_RESOURCE_LIST_HEADER_LEN as usize {
        return None;
    }
    let task_id = ld32(&payload[CHILD_RESOURCE_LIST_TASK_ID as usize..]);
    let count = ld32(&payload[CHILD_RESOURCE_LIST_COUNT as usize..]);
    if count > CHILD_RESOURCE_LIST_MAX_COUNT {
        return None;
    }
    let need = (CHILD_RESOURCE_LIST_HEADER_LEN as u64)
        .checked_add((count as u64).checked_mul(CHILD_INVALIDATE_RECORD_LEN as u64)?)?
        as usize;
    if payload.len() < need {
        return None;
    }
    let mut records = Vec::with_capacity(count as usize);
    let mut off = CHILD_RESOURCE_LIST_HEADER_LEN as usize;
    for _ in 0..count {
        let object_id = ld32(&payload[off..]);
        let flags = ld32(&payload[off + 4..]);
        records.push(InvalidateResourceRecord {
            object_id,
            flags,
            ops: InvalidateValidityOps::from_le_dword(flags),
        });
        off += CHILD_INVALIDATE_RECORD_LEN as usize;
    }
    Some(InvalidateResourcesCommand {
        task_id,
        count,
        records,
    })
}

/// Decode CmdSynchronizeResources: header `{task_id, count}` + `count × {object_id}`.
///
/// Guest `synchronizeForUnwire` uses `getCommandBytes(4)` for the object cell (no flags).
pub fn decode_synchronize_resources(payload: &[u8]) -> Option<SynchronizeResourcesCommand> {
    if payload.len() < CHILD_RESOURCE_LIST_HEADER_LEN as usize {
        return None;
    }
    let task_id = ld32(&payload[CHILD_RESOURCE_LIST_TASK_ID as usize..]);
    let count = ld32(&payload[CHILD_RESOURCE_LIST_COUNT as usize..]);
    if count > CHILD_RESOURCE_LIST_MAX_COUNT {
        return None;
    }
    let need = (CHILD_RESOURCE_LIST_HEADER_LEN as u64)
        .checked_add((count as u64).checked_mul(CHILD_SYNCHRONIZE_RECORD_LEN as u64)?)?
        as usize;
    if payload.len() < need {
        return None;
    }
    let mut object_ids = Vec::with_capacity(count as usize);
    let mut off = CHILD_RESOURCE_LIST_HEADER_LEN as usize;
    for _ in 0..count {
        object_ids.push(ld32(&payload[off..]));
        off += CHILD_SYNCHRONIZE_RECORD_LEN as usize;
    }
    Some(SynchronizeResourcesCommand {
        task_id,
        count,
        object_ids,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::endian::st32;

    /// RE pageBacking: plen=16 = header + one 8-byte record; LE `01 00 00 01`
    /// = clear_host_valid + set_guest_valid (PVG validity-op bytes).
    #[test]
    fn decode_invalidate_pageon_shape() {
        let mut p = [0u8; 16];
        st32(&mut p[0..], 0); // task
        st32(&mut p[4..], 1); // count
        st32(&mut p[8..], 0x2a); // object_id
        st32(&mut p[12..], CHILD_INVALIDATE_PAGEON_FLAGS);
        let c = decode_invalidate_resources(&p).expect("decode");
        assert_eq!(c.task_id, 0);
        assert_eq!(c.count, 1);
        assert_eq!(c.records.len(), 1);
        assert_eq!(c.records[0].object_id, 0x2a);
        assert_eq!(c.records[0].flags, CHILD_INVALIDATE_PAGEON_FLAGS);
        assert_eq!(c.records[0].ops, InvalidateValidityOps::PAGEON);
        assert_eq!(c.records[0].ops.clear_host_valid, 1);
        assert_eq!(c.records[0].ops.set_host_valid, 0);
        assert_eq!(c.records[0].ops.clear_guest_valid, 0);
        assert_eq!(c.records[0].ops.set_guest_valid, 1);
        // LE memory: not bit0|bit24 as independent product bits.
        assert_eq!(CHILD_INVALIDATE_PAGEON_FLAGS.to_le_bytes(), [1, 0, 0, 1]);
    }

    #[test]
    fn invalidate_validity_ops_roundtrip() {
        let ops = InvalidateValidityOps {
            clear_host_valid: 1,
            set_host_valid: 0,
            clear_guest_valid: 1,
            set_guest_valid: 0,
        };
        assert_eq!(InvalidateValidityOps::from_le_dword(ops.to_le_dword()), ops);
    }

    /// RE synchronizeForUnwire: plen=12 = header + one u32 object_id.
    #[test]
    fn decode_synchronize_unwire_shape() {
        let mut p = [0u8; 12];
        st32(&mut p[0..], 4); // task
        st32(&mut p[4..], 1);
        st32(&mut p[8..], 99);
        let c = decode_synchronize_resources(&p).expect("decode");
        assert_eq!(c.task_id, 4);
        assert_eq!(c.count, 1);
        assert_eq!(c.object_ids, vec![99]);
    }

    #[test]
    fn decode_invalidate_multi_object_when_payload_long() {
        let mut p = [0u8; 8 + 16];
        st32(&mut p[0..], 1);
        st32(&mut p[4..], 2);
        st32(&mut p[8..], 10);
        st32(&mut p[12..], 0x1000001);
        st32(&mut p[16..], 11);
        st32(&mut p[20..], 0x1000001);
        let c = decode_invalidate_resources(&p).expect("decode");
        assert_eq!(c.count, 2);
        assert_eq!(c.records[0].object_id, 10);
        assert_eq!(c.records[1].object_id, 11);
    }

    #[test]
    fn decode_invalidate_rejects_short_for_count() {
        let mut p = [0u8; 12]; // header claims count=2 but only 4 payload bytes
        st32(&mut p[0..], 0);
        st32(&mut p[4..], 2);
        st32(&mut p[8..], 1);
        assert!(decode_invalidate_resources(&p).is_none());
    }
}
