//! Root/child FIFO drains, stamp writeback, and fail-visible command dispatch.
//!
//! Prefer structure correctness over full exec.c coverage: known root/child
//! control-plane ops update device state; unknown opcodes are recorded visibly.

use crate::contract::endian::{ld16, ld32, st16, st32};
use crate::model::*;
use crate::model::{DeviceState, ExecFault, FailEvent, PacketFault, StampSlot};
use crate::observe::Emit;
use crate::runtime::decode::fifo::{
    display_refresh_hz_1616, display_timing_entry_offset, encode_display_timing_entry,
    DisplayTimingEntry, DISPLAY_DESC_TIMING_STRIDE,
};
use crate::runtime::gpa_map;
use crate::runtime::heap_query::QueryError;
use crate::runtime::host::{HostAction, HostMemory, HostOps, MemError};
use crate::runtime::task_slot::{resolve_task_word, TaskWordSite};

/// apple-gfx `pending_frames >= 2`: hold further guest presents at FIFO head
/// until host paint consumes +0x188. Entry-side waitForPendingFrames — not
/// stamp-after-paint (that inverted PGDisplay completion and stacked tooltips).
pub const MAX_UNPAINTED_PRESENTS: u32 = 2;

/// Bit 0 in `translation_order_hold_mask` names the root FIFO. Child FIFOs use
/// their channel bit, matching `translation_deferred_mask`.
const TRANSLATION_ROOT_FIFO_BIT: u32 = 1;

fn note_translation_order_hold(state: &mut DeviceState, held_mask: u32) {
    let new_mask = held_mask & !state.translation_order_hold_mask;
    if new_mask == 0 {
        return;
    }
    let starts_episode = state.translation_order_hold_mask == 0;
    state.translation_order_hold_mask |= new_mask;
    if starts_episode {
        state.translation_order_holds = state.translation_order_holds.saturating_add(1);
    }
    // Census, not a failure: this is a resolver saying "not ready yet". The FIFO
    // is parked until the AIR module loads and `release_translation_order_holds`
    // takes the mask back down — and its release line was already `off`, so
    // logging the wait half as a failure made one control-flow pair straddle both
    // channels. Boot 87: 34 episodes started, 35 released, i.e. every one. A hold
    // that never releases is caught at `DeviceState::reset` instead, where the
    // guest's own teardown is the deadline and no age or depth has to be invented.
    crate::observe::off(format!(
        "translation_order_hold reason=air_loading held_mask={:#x} new_mask={new_mask:#x} producer_mask={:#x} count={}",
        state.translation_order_hold_mask,
        state.translation_deferred_mask,
        state.translation_order_holds
    ));
}

fn release_translation_order_holds(state: &mut DeviceState) {
    if state.translation_deferred_mask != 0 || state.translation_order_hold_mask == 0 {
        return;
    }
    let held_mask = std::mem::take(&mut state.translation_order_hold_mask);
    crate::observe::off(format!(
        "translation_order_release held_mask={held_mask:#x} producer_mask=0x0"
    ));
}

/// Samples logged per distinct display-transaction `(opcode, payload_len)` shape.
///
/// One sample proves the shape exists; a few more let the reader compare which
/// words move between frames (surface id, task id) and which are constant
/// (pipe index) without re-booting.
const DISPLAY_TXN_PAYLOAD_SAMPLES: u32 = 4;

/// Payload bytes hex-dumped per sample. A transaction payload that carried an
/// inline plane list would be tens of bytes, not kilobytes; this bounds a
/// pathological length without truncating the interesting case.
const DISPLAY_TXN_PAYLOAD_DUMP_MAX: usize = 128;

/// Trailer size `submitTransaction` appends after serializing the transaction's
/// resource list: `[pipe][task][surface][gamma…]` for the gamma command,
/// `[pipe][surface][task]` otherwise.
fn display_txn_trailer_len(opcode: u16) -> usize {
    if opcode == CHILD_OP_PRESENT_GAMMA_X86 {
        0x24
    } else {
        0x0c
    }
}

/// Word slots of the surface id and the task field within the trailer. The
/// gamma command swaps the two relative to the plain one, which is why this is
/// keyed on the opcode rather than assumed.
fn display_txn_trailer_slots(opcode: u16) -> (usize, usize) {
    if opcode == CHILD_OP_PRESENT_GAMMA_X86 {
        // [pipe][task][surface][gamma…]
        (2, 1)
    } else {
        // [pipe][surface][task]
        (1, 2)
    }
}

/// Measurement for the display-transaction wire shape (`display_txn_payload`).
///
/// We decode opcode 6/7 as a fixed 12/0x24-byte record read from payload offset
/// zero, which yields a single surface id — plane 0 of what is really an
/// `IOAccelDisplayPipeTransaction2`: a per-frame list of planes with source,
/// destination and dirty rects. Whether the rest of that list rides inline in
/// this payload decides where a real decode reads it from, and nothing in the
/// guest driver settles it statically because the serializer it calls lives in
/// IOAcceleratorFamily2.
///
/// So record the shape from a live boot. `head*` is the trailer under the
/// current offset-zero reading; `tail*` is the same trailer read from the end of
/// the payload, which is where it lands if a plane list precedes it. When the
/// two agree the payload is trailer-only and the list travels elsewhere; when
/// they diverge, `hex` shows the list and `tail*` is the correct reading.
///
/// A live x86 session answered the framing half: every payload was trailer-only,
/// so `tail*` is the authoritative reading and the named `pipe`/`surface`/`task`
/// fields below are decoded from it. What remains open is the task field, which
/// was zero for every sample taken during bring-up. It is `task->+0x268` in
/// `submitTransaction`, and whether it identifies the GPU task that produced the
/// surface decides how the host learns that a present's content is ready — so
/// the sample budget re-arms when it first becomes non-zero.
fn note_display_txn_payload(state: &mut DeviceState, channel_id: u32, packet: &Packet) {
    let plen = packet.payload.len();
    let trailer = display_txn_trailer_len(packet.opcode);
    let tail_base = plen.checked_sub(trailer);
    let word =
        |off: usize| -> Option<u32> { (off + 4 <= plen).then(|| ld32(&packet.payload[off..])) };
    let show =
        |v: Option<u32>| -> String { v.map_or_else(|| "-".to_string(), |w| format!("{w:#010x}")) };
    let tail = |slot: usize| -> Option<u32> { tail_base.and_then(|base| word(base + slot * 4)) };

    let (surface_slot, task_slot) = display_txn_trailer_slots(packet.opcode);
    let pipe = tail(0);
    let surface = tail(surface_slot);
    let task = tail(task_slot);

    let seen = state
        .display
        .txn_payload_samples
        .entry((
            packet.opcode,
            plen,
            pipe.unwrap_or(u32::MAX),
            task.is_some_and(|t| t != 0),
        ))
        .or_insert(0);
    if *seen >= DISPLAY_TXN_PAYLOAD_SAMPLES {
        return;
    }
    *seen += 1;
    let sample = *seen;

    let dumped = plen.min(DISPLAY_TXN_PAYLOAD_DUMP_MAX);
    let mut hex = String::with_capacity(dumped * 2);
    for b in &packet.payload[..dumped] {
        hex.push_str(&format!("{b:02x}"));
    }

    crate::observe::fail(format!(
        "display_txn_payload op={:#x} ch={channel_id} plen={plen} total_size={} stamp={:#x} \
         sample={sample}/{DISPLAY_TXN_PAYLOAD_SAMPLES} trailer={trailer} \
         pipe={} surface={} task={} \
         head0={} head1={} head2={} tail0={} tail1={} tail2={} dumped={dumped} hex={hex}",
        packet.opcode,
        packet.total_size,
        packet.completion_stamp,
        show(pipe),
        show(surface),
        show(task),
        show(word(0)),
        show(word(4)),
        show(word(8)),
        show(tail(0)),
        show(tail(1)),
        show(tail(2)),
    ));
}

/// What the CPU-side capture can say about a present's content.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PresentContentVerdict {
    /// No CPU pixels exist for this present, so nothing can be claimed.
    Unsampled,
    /// Sampled, and every pixel's RGB is zero.
    Black,
    /// Sampled, and something is visible.
    Content,
}

/// Judge a present's captured frame.
///
/// An empty `frame_bgra` is **not** a black frame. When a dmabuf carries the
/// present (route B), `capture_present_frame` deliberately skips the full-frame
/// GPU→CPU readback and leaves the buffer empty, so a plain `max_rgb == 0` test
/// reports black on every such present — 1338 `present_black_retain` records
/// against 1312 presents on a live boot. That buries the always-on log under a
/// wolf-cry and hides the genuinely black frame the record exists to catch,
/// which is the opposite of what an always-on failure sink is for. With no
/// pixels there is no evidence either way, so the absence has its own verdict.
pub(crate) fn present_content_verdict(frame_bgra: &[u8], max_rgb: u8) -> PresentContentVerdict {
    if frame_bgra.is_empty() {
        PresentContentVerdict::Unsampled
    } else if max_rgb == 0 {
        PresentContentVerdict::Black
    } else {
        PresentContentVerdict::Content
    }
}

/// Parsed FIFO packet (main + child share framing).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Packet {
    pub opcode: u16,
    pub stamp_count: u16,
    pub total_size: u32,
    pub completion_stamp: u32,
    pub payload: Vec<u8>,
    pub next_head: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PacketError {
    ShortHeader,
    BadSize,
    Incomplete,
    Desynced,
}

impl PacketError {
    /// The registered fault this error reports, or `None` when it is ring
    /// control flow rather than a refusal.
    ///
    /// This is I2's carve-out made mechanical instead of conventional. A partial
    /// packet in the ring is the *normal* state of a producer mid-write: the
    /// drain loop breaks and comes back. Logging it would flood the always-on
    /// sink on every healthy boot, which is why `ShortHeader` and `Incomplete`
    /// answer `None` here — and why a future variant cannot be added without its
    /// author deciding which side of that line it falls on.
    pub fn fault(self) -> Option<PacketFault> {
        match self {
            Self::ShortHeader | Self::Incomplete => None,
            Self::BadSize => Some(PacketFault::BadSize),
            Self::Desynced => Some(PacketFault::Desynced),
        }
    }
}

fn decode_packet(bytes: &[u8], head: u32, available: u32) -> Result<Packet, PacketError> {
    if available < PACKET_HEADER_LEN {
        return Err(PacketError::ShortHeader);
    }
    if bytes.len() < PACKET_HEADER_LEN as usize {
        return Err(PacketError::ShortHeader);
    }
    let opcode = ld16(&bytes[PACKET_OPCODE..]);
    let stamp_count = ld16(&bytes[PACKET_STAMP_COUNT..]);
    let total_size = ld32(&bytes[PACKET_TOTAL_SIZE..]);
    let completion_stamp = ld32(&bytes[PACKET_COMPLETION_STAMP..]);

    if total_size < PACKET_HEADER_LEN || total_size as usize > bytes.len() {
        return Err(PacketError::BadSize);
    }
    if available < total_size {
        return Err(PacketError::Incomplete);
    }
    let stamps_bytes = stamp_count as u32 * PACKET_STAMP_LEN;
    let min_payload_off = PACKET_HEADER_LEN + stamps_bytes;
    if total_size < min_payload_off {
        return Err(PacketError::BadSize);
    }
    let payload = bytes[min_payload_off as usize..total_size as usize].to_vec();
    Ok(Packet {
        opcode,
        stamp_count,
        total_size,
        completion_stamp,
        payload,
        next_head: head.wrapping_add(total_size),
    })
}

fn read_ring_bytes<M: HostMemory>(
    mem: &M,
    base_gpa: u64,
    ring_size: u32,
    absolute: u32,
    len: u32,
) -> Result<Vec<u8>, MemError> {
    let mut out = vec![0u8; len as usize];
    if ring_size == 0 || len == 0 {
        return Ok(out);
    }
    let mut copied = 0u32;
    while copied < len {
        let off = absolute.wrapping_add(copied) % ring_size;
        let chunk = (ring_size - off).min(len - copied);
        mem.read_gpa(
            base_gpa + off as u64,
            &mut out[copied as usize..(copied + chunk) as usize],
        )?;
        copied += chunk;
    }
    Ok(out)
}

/// Write stamp value to FIFO base page slot and set status bit.
pub fn write_stamp<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    stamp_index: u32,
    stamp_value: u32,
) {
    let index = stamp_slot_index(stamp_index);
    if state.gfx.fifo_base_page == 0 {
        return;
    }
    // Before the guest is told anything finished, everything this device still
    // owes guest RAM has to be in guest RAM. After this write the guest may free
    // the render targets and its allocator may hand those pages to anything, and
    // no later check can tell that memory apart from the target it used to be —
    // which is why the page-set guard passed on 810 of 810 landings and the heap
    // corruption continued. See `storage_flush::flush_all_windows_before_fence`,
    // which the root completion stamp in `drain_main_fifo` shares.
    crate::runtime::storage_flush::flush_all_windows_before_fence(state, host);
    let Some(off) = stamp_slot_offset(index, state.page_size()) else {
        return;
    };
    let gpa = state.pfn_gpa(state.gfx.fifo_base_page) + off;
    let page_size = state.page_size() as usize;
    if gpa_map::write_u32(host, gpa, stamp_value, page_size).is_ok() {
        // The guest's fence has moved. Everything it allocated for the work this
        // stamp completes may be freed from here on, so any deferred window
        // still holding bytes for guest RAM is now writing behind the guest's
        // back — see `GvaDeferredEntry::armed_stamp_seq`.
        state.completion_stamp_seq = state.completion_stamp_seq.wrapping_add(1);
        state
            .gfx
            .interrupt_status_gpu
            .fetch_or(1u32 << (index & 0x1f), std::sync::atomic::Ordering::AcqRel);
        host.enqueue(HostAction::irq_gfx());
    }
}

fn reply_device_info<H: HostMemory + HostOps>(
    host: &mut H,
    count: u32,
    reply_pfn: u32,
    page_shift: u32,
) -> Result<(), MemError> {
    if reply_pfn == 0 {
        return Ok(());
    }
    let page_size = 1usize << page_shift;
    let gpa = pfn_to_gpa(reply_pfn, page_shift);
    // Contract: guest reply buffer is one page. Cap pairs so we never write past it.
    let max_pairs = (page_size / DEVICE_INFO_REPLY_PAIR_LEN) as u32;
    if max_pairs == 0 {
        crate::observe::fail(format!(
            "device_info fail reason=page_too_small page={page_size:#x}"
        ));
        return Err(MemError::BadArgs);
    }
    let n = (DEVICE_INFO_CAPS.len() as u32).min(count).min(max_pairs);
    // When guest asks for more than one page of pairs, still write at most a
    // page of caps + optional sentinel only if room remains.
    let write_sentinel = n < count && n.saturating_add(1) <= max_pairs;
    if count > max_pairs {
        crate::observe::fail(format!(
            "device_info cap reason=reply_page count={count} max_pairs={max_pairs} page={page_size:#x}"
        ));
    }
    for i in 0..n {
        let (key, value) = DEVICE_INFO_CAPS[i as usize];
        let mut pair = [0u8; DEVICE_INFO_REPLY_PAIR_LEN];
        st32(&mut pair[0..4], key);
        st32(&mut pair[4..8], value);
        gpa_map::write_bytes(
            host,
            gpa + (i as u64) * DEVICE_INFO_REPLY_PAIR_LEN as u64,
            &pair,
            page_size,
        )?;
    }
    if write_sentinel {
        let sentinel = [0u8; DEVICE_INFO_REPLY_PAIR_LEN];
        gpa_map::write_bytes(
            host,
            gpa + (n as u64) * DEVICE_INFO_REPLY_PAIR_LEN as u64,
            &sentinel,
            page_size,
        )?;
    }
    Ok(())
}

/// Conservative compute-pipeline info pairs (kb tahoe-x86 + texture-ref 29-06-26).
/// key1 maxTotalThreadsPerThreadgroup, key3 threadExecutionWidth,
/// key4 staticThreadgroupMemoryLength. Real values should come from pipeline
/// reflection once metal2vulkan encode lands; zeros blocked guest MPS/compute.
const COMPUTE_INFO_CAPS: &[(u32, u32)] = &[(1, 1024), (3, 32), (4, 0)];

/// Child `CmdGetComputeInfo` (0x3b): 24B payload
/// `[task_id@0][pipeline_ref@4][max_key@8][count@12][reply_gva@16]`.
/// Host writes key/value pairs at reply_gva before stamp (Apple host contract).
fn reply_compute_info<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    payload: &[u8],
) -> bool {
    if payload.len() < 24 {
        return false;
    }
    let raw_task = ld32(&payload[0..]);
    let pipeline_ref = ld32(&payload[4..]);
    let max_key = ld32(&payload[8..]);
    let count = ld32(&payload[12..]);
    let reply_gva = u64::from_le_bytes(payload[16..24].try_into().unwrap_or([0; 8]));
    if reply_gva == 0 || count == 0 {
        crate::observe::fail(format!(
            "get_compute_info empty task={raw_task} pipe={pipeline_ref} max_key={max_key} count={count} gva={reply_gva:#x}"
        ));
        return false;
    }
    // A live slot or nothing; `bad_task` now names the word the guest sent
    // rather than the halved id this used to have resolved to by then.
    let Some(task_id) = resolve_task_word(&state.tasks, TaskWordSite::ComputeInfo, raw_task) else {
        crate::observe::fail(format!(
            "get_compute_info bad_task task={raw_task} pipe={pipeline_ref}"
        ));
        return false;
    };
    let mut wrote = 0u32;
    for &(key, value) in COMPUTE_INFO_CAPS {
        if key > max_key {
            continue;
        }
        if wrote >= count {
            break;
        }
        let mut pair = [0u8; DEVICE_INFO_REPLY_PAIR_LEN];
        st32(&mut pair[0..4], key);
        st32(&mut pair[4..8], value);
        let off = (wrote as u64) * DEVICE_INFO_REPLY_PAIR_LEN as u64;
        if crate::runtime::gva_mem::write_task_gva_product(
            state,
            host,
            task_id,
            reply_gva + off,
            &pair,
        )
        .is_err()
        {
            crate::observe::fail(format!(
                "get_compute_info write_fail task={task_id} gva={reply_gva:#x} wrote={wrote}"
            ));
            return false;
        }
        wrote += 1;
    }
    if wrote < count {
        let sentinel = [0u8; DEVICE_INFO_REPLY_PAIR_LEN];
        let off = (wrote as u64) * DEVICE_INFO_REPLY_PAIR_LEN as u64;
        let _ = crate::runtime::gva_mem::write_task_gva_product(
            state,
            host,
            task_id,
            reply_gva + off,
            &sentinel,
        );
    }
    // Success census — the reply landed. Route to `off()` so it stays always-on
    // in the log but leaves the curated real-error view clean; the genuine
    // failures (`empty`/`bad_task`/`write_fail`/`short`) above stay `fail()`.
    crate::observe::off(format!(
        "get_compute_info ok task={task_id} pipe={pipeline_ref} max_key={max_key} count={count} wrote={wrote} gva={reply_gva:#x}"
    ));
    true
}

fn reply_heap_texture_size_and_align<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    payload: &[u8],
) -> bool {
    let request = match crate::runtime::heap_query::decode_request(payload) {
        Ok(request) => request,
        Err(error) => {
            Emit::decline("heap_texture_query", &error)
                .field("plen", payload.len())
                .fail();
            return false;
        }
    };
    // A live slot or nothing. `resolved_task` is gone with the arm that made the
    // two differ: the only slot this can act on is the one the guest named.
    let Some(task_id) = resolve_task_word(
        &state.tasks,
        TaskWordSite::HeapTextureQuery,
        request.task_id,
    ) else {
        Emit::decline("heap_texture_query", &QueryError::BadTask)
            .field("task", request.task_id)
            .field("gva", format!("{:#x}", request.reply_gva))
            .fail();
        return false;
    };
    let requirement = match crate::runtime::heap_query::query_size_and_align(&request.descriptor) {
        Ok(requirement) => requirement,
        Err(error) => {
            let desc = request.descriptor;
            Emit::decline("heap_texture_query", &error)
                .field("task", task_id)
                .field("type", desc.texture_type)
                .field("fmt", format!("{:#x}", desc.pixel_format))
                .field(
                    "dims",
                    format!("{}x{}x{}", desc.width, desc.height, desc.depth),
                )
                .field("mips", desc.mipmap_level_count)
                .field("samples", desc.sample_count)
                .field("array", desc.array_length)
                .field("usage", format!("{:#x}", desc.usage))
                .field("options", format!("{:#x}", desc.resource_options))
                .fail();
            return false;
        }
    };
    let reply = requirement.encode();
    if crate::runtime::gva_mem::write_task_gva_product(
        state,
        host,
        task_id,
        request.reply_gva,
        &reply,
    )
    .is_err()
    {
        crate::observe::fail(format!(
            "heap_texture_query fail reason=write_fail task={task_id} gva={:#x} reply_len={} size={:#x} align={:#x}",
            request.reply_gva,
            request.reply_len,
            requirement.size,
            requirement.align
        ));
        return false;
    }
    let desc = request.descriptor;
    crate::observe::off(format!(
        "heap_texture_query ok task={task_id} gva={:#x} type={} fmt={:#x} {}x{}x{} mips={} samples={} array={} usage={:#x} options={:#x} size={:#x} align={:#x}",
        request.reply_gva,
        desc.texture_type,
        desc.pixel_format,
        desc.width,
        desc.height,
        desc.depth,
        desc.mipmap_level_count,
        desc.sample_count,
        desc.array_length,
        desc.usage,
        desc.resource_options,
        requirement.size,
        requirement.align
    ));
    true
}

fn process_root_packet<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    packet: &Packet,
) {
    let op = packet.opcode;
    let effective = if op == ROOT_OP_WRAPPER {
        if packet.payload.len() >= 4 {
            ld32(&packet.payload[0..]) as u16
        } else {
            op
        }
    } else {
        op
    };

    match effective {
        ROOT_OP_DEVICE_INFO_TAHOE => {
            if packet.payload.len() >= DEVICE_INFO_TAHOE_REPLY_PFN + 4 {
                let count = ld32(&packet.payload[DEVICE_INFO_TAHOE_COUNT..]);
                let pfn = ld32(&packet.payload[DEVICE_INFO_TAHOE_REPLY_PFN..]);
                let _ = reply_device_info(host, count, pfn, state.page_shift);
            }
        }
        ROOT_OP_DEVICE_INFO_MONTEREY => {
            if packet.payload.len() >= DEVICE_INFO_MONTEREY_REPLY_PFN + 4 {
                let count = ld32(&packet.payload[DEVICE_INFO_MONTEREY_COUNT..]);
                let pfn = ld32(&packet.payload[DEVICE_INFO_MONTEREY_REPLY_PFN..]);
                let _ = reply_device_info(host, count, pfn, state.page_shift);
            }
        }
        ROOT_OP_DEFINE_FIFO => {
            if packet.payload.len() >= 4 {
                let ch = ld32(&packet.payload[0..]);
                if ch >= 1 && (ch as usize) < MAX_CHANNELS {
                    let bit = 1u32 << ch;
                    state.active_child_mask |= bit;
                    state.translation_deferred_mask &= !bit;
                    state.translation_order_hold_mask &= !bit;
                    state.present_translation_hold_mask &= !bit;
                    // Invalidate ring cache for this channel.
                    state.child_rings[ch as usize] = Default::default();
                }
            }
        }
        ROOT_OP_FREE_FIFO => {
            if packet.payload.len() >= 4 {
                let ch = ld32(&packet.payload[0..]);
                if ch >= 1 && (ch as usize) < MAX_CHANNELS {
                    let bit = 1u32 << ch;
                    state.active_child_mask &= !bit;
                    state.pending.child_mask &= !bit;
                    state.translation_deferred_mask &= !bit;
                    state.translation_order_hold_mask &= !bit;
                    state.present_translation_hold_mask &= !bit;
                    state.child_rings[ch as usize] = Default::default();
                    state.child_stamps[ch as usize].reset();
                }
            }
        }
        ROOT_OP_DEFINE_TASK2 => {
            if packet.payload.len() >= DEFINE_TASK_LEN {
                let raw_id = ld32(&packet.payload[DEFINE_TASK_RAW_ID..]);
                let length = ld32(&packet.payload[DEFINE_TASK_LENGTH..]) as u64;
                // length field is only low 32 in compact layout; full u64 at +4 if present.
                let length = if packet.payload.len() >= 12 {
                    u64::from(ld32(&packet.payload[DEFINE_TASK_LENGTH..]))
                        | (u64::from(ld32(&packet.payload[DEFINE_TASK_LENGTH + 4..])) << 32)
                } else {
                    length
                };
                let dir = ld32(&packet.payload[DEFINE_TASK_DIRECTORY_PFN..]);
                let task_id = raw_id >> DEFINE_TASK_ID_SHIFT;
                let ok = state.define_task(task_id, length, dir);
                // Measure: capture directory + root/depth so one boot shows PT identity.
                if ok && (task_id as usize) < state.tasks.len() {
                    let slot = &state.tasks[task_id as usize];
                    let walk = crate::runtime::gva_mem::diagnose_task_slot(
                        host,
                        slot,
                        task_id,
                        0,
                        state.page_shift,
                    );
                    crate::observe::off(format!(
                        "define_task root raw={raw_id:#x} task={task_id} len={length:#x} dir={dir:#x} page_shift={} {walk}",
                        state.page_shift
                    ));
                }
            }
        }
        ROOT_OP_SET_OBJECT_LIST => {
            if packet.payload.len() >= SET_OBJECT_LIST_LEN {
                let task_id = ld32(&packet.payload[SET_OBJECT_LIST_TASK_ID..]);
                let pfn = ld32(&packet.payload[SET_OBJECT_LIST_PFN..]);
                let count = ld32(&packet.payload[SET_OBJECT_LIST_COUNT..]);
                let _ = state.set_object_list(task_id, pfn, count);
            }
        }
        // PVG CmdDeleteTask (0x20). Live: top UnknownRootOpcode was op 32 total_size=16
        // (12-byte header + task_id u32). Guest reuses task ids — must clear.
        ROOT_OP_DELETE_TASK => {
            let task_id = if packet.payload.len() >= 4 {
                ld32(&packet.payload[0..])
            } else {
                0
            };
            let ok = state.delete_task(task_id);
            crate::observe::off(format!(
                "delete_task root task={task_id} ok={} plen={}",
                ok as u8,
                packet.payload.len()
            ));
        }
        _ => {
            state.record_fail(FailEvent::UnknownRootOpcode {
                opcode: effective,
                total_size: packet.total_size,
            });
        }
    }
}

/// Drain the main (root) FIFO while producer != consumer.
pub fn drain_main_fifo<H: HostMemory + HostOps>(state: &mut DeviceState, host: &mut H) {
    let ring_size = main_ring_data_size(state.gfx.fifo_length, state.gfx.fifo_start);
    if ring_size == 0 || state.gfx.fifo_base_page == 0 {
        state.pending.main_drain = false;
        return;
    }
    let base = state.pfn_gpa(state.gfx.fifo_base_page) + state.gfx.fifo_start as u64;
    let mut completed = false;

    while state
        .gfx
        .fifo_read
        .load(std::sync::atomic::Ordering::Acquire)
        != state.gfx.fifo_written
    {
        let Some(available) = published_byte_count(
            state
                .gfx
                .fifo_read
                .load(std::sync::atomic::Ordering::Acquire),
            state.gfx.fifo_written,
            ring_size,
        ) else {
            state.record_fail(FailEvent::MalformedRootPacket {
                fault: PacketFault::DesyncedHeadTail,
                head: state
                    .gfx
                    .fifo_read
                    .load(std::sync::atomic::Ordering::Acquire),
            });
            break;
        };
        if available < PACKET_HEADER_LEN {
            break;
        }
        // Snapshot up to min(available, ring_size) — header first then full packet.
        let header = match read_ring_bytes(
            host,
            base,
            ring_size,
            state
                .gfx
                .fifo_read
                .load(std::sync::atomic::Ordering::Acquire),
            PACKET_HEADER_LEN,
        ) {
            Ok(h) => h,
            Err(_) => {
                state.record_fail(FailEvent::MalformedRootPacket {
                    fault: PacketFault::RootHeaderRead,
                    head: state
                        .gfx
                        .fifo_read
                        .load(std::sync::atomic::Ordering::Acquire),
                });
                break;
            }
        };
        let total_size = ld32(&header[PACKET_TOTAL_SIZE..]);
        let snap_len = if total_size >= PACKET_HEADER_LEN
            && total_size <= ring_size
            && available >= total_size
        {
            total_size
        } else if available >= PACKET_HEADER_LEN {
            // incomplete or bad — try decode to classify
            PACKET_HEADER_LEN
        } else {
            break;
        };
        let snap = match read_ring_bytes(
            host,
            base,
            ring_size,
            state
                .gfx
                .fifo_read
                .load(std::sync::atomic::Ordering::Acquire),
            snap_len,
        ) {
            Ok(s) => s,
            Err(_) => {
                state.record_fail(FailEvent::MalformedRootPacket {
                    fault: PacketFault::RootSnapRead,
                    head: state
                        .gfx
                        .fifo_read
                        .load(std::sync::atomic::Ordering::Acquire),
                });
                break;
            }
        };
        match decode_packet(
            &snap,
            state
                .gfx
                .fifo_read
                .load(std::sync::atomic::Ordering::Acquire),
            available,
        ) {
            Ok(packet) => {
                process_root_packet(state, host, &packet);
                state
                    .gfx
                    .fifo_read
                    .store(packet.next_head, std::sync::atomic::Ordering::Release);
                // Root stamp = slot 0.
                if state.gfx.fifo_base_page != 0 {
                    if let Some(off) = stamp_slot_offset(0, state.page_size()) {
                        // The root stamp is a completion the guest waits on, so
                        // every deferred rail owes guest RAM its bytes here, not
                        // only at `write_stamp`'s child slots.
                        crate::runtime::storage_flush::flush_all_windows_before_fence(state, host);
                        let gpa = state.pfn_gpa(state.gfx.fifo_base_page) + off;
                        if gpa_map::write_u32(
                            host,
                            gpa,
                            packet.completion_stamp,
                            state.page_size() as usize,
                        )
                        .is_ok()
                        {
                            // A window armed after this point has outlived a fence
                            // the moment it is still armed at the next one. The
                            // counter is what `armed_stamp_seq` is compared
                            // against, so a rail that does not move it reads as
                            // punctual however long it actually waited.
                            state.completion_stamp_seq =
                                state.completion_stamp_seq.wrapping_add(1);
                            completed = true;
                        } else {
                            // The guest waits on this root completion stamp; a
                            // silent writeback failure hangs it forever with no
                            // trace (drain.rs Rank-2 audit).
                            state.record_fail(FailEvent::MalformedRootPacket {
                                fault: PacketFault::RootStampWriteback,
                                head: state
                                    .gfx
                                    .fifo_read
                                    .load(std::sync::atomic::Ordering::Acquire),
                            });
                        }
                    }
                }
            }
            Err(PacketError::Incomplete) | Err(PacketError::ShortHeader) => break,
            Err(PacketError::BadSize) => {
                state.record_fail(FailEvent::MalformedRootPacket {
                    fault: PacketFault::BadSize,
                    head: state
                        .gfx
                        .fifo_read
                        .load(std::sync::atomic::Ordering::Acquire),
                });
                break;
            }
            Err(PacketError::Desynced) => {
                state.record_fail(FailEvent::MalformedRootPacket {
                    fault: PacketFault::Desynced,
                    head: state
                        .gfx
                        .fifo_read
                        .load(std::sync::atomic::Ordering::Acquire),
                });
                break;
            }
        }
    }

    if completed {
        state
            .gfx
            .interrupt_status_gpu
            .fetch_or(1, std::sync::atomic::Ordering::AcqRel);
        host.enqueue(HostAction::irq_gfx());
    }
    state.pending.main_drain = false;
}

fn ensure_child_ring<M: HostMemory>(
    state: &mut DeviceState,
    mem: &M,
    channel_id: u32,
    base_pfn: u32,
) -> u32 {
    if channel_id == 0 || channel_id as usize >= MAX_CHANNELS || base_pfn == 0 {
        return 0;
    }
    let page_shift = state.page_shift;
    let page_size = state.page_size();
    let ring = &mut state.child_rings[channel_id as usize];
    if ring.valid && ring.base_pfn == base_pfn {
        return ring.length;
    }
    // Count leading non-zero PFNs in the page list (one page of u32 PFNs).
    let list_gpa = pfn_to_gpa(base_pfn, page_shift);
    let max_entries = (page_size / CHILD_RING_PFN_ENTRY_LEN) as u32;
    let mut page_gpas = Vec::new();
    for i in 0..max_entries {
        let mut b = [0u8; 4];
        if mem
            .read_gpa(list_gpa + i as u64 * CHILD_RING_PFN_ENTRY_LEN, &mut b)
            .is_err()
        {
            break;
        }
        let pfn = u32::from_le_bytes(b);
        if pfn == 0 {
            break;
        }
        page_gpas.push(pfn_to_gpa(pfn, page_shift));
    }
    let length = (page_gpas.len() as u32).saturating_mul(page_size as u32);
    *ring = crate::model::ChannelRing {
        valid: length != 0,
        base_pfn,
        length,
        page_gpas,
    };
    length
}

fn read_child_ring_bytes<M: HostMemory>(
    mem: &M,
    page_gpas: &[u64],
    ring_length: u32,
    absolute: u32,
    len: u32,
    page_shift: u32,
) -> Result<Vec<u8>, MemError> {
    let page_size = 1u64 << page_shift;
    let mut out = vec![0u8; len as usize];
    if ring_length == 0 || page_gpas.is_empty() {
        return Ok(out);
    }
    for i in 0..len {
        let off = absolute.wrapping_add(i) % ring_length;
        let page = (off as u64) >> page_shift;
        let page_off = (off as u64) & (page_size - 1);
        if page as usize >= page_gpas.len() {
            out[i as usize] = 0;
            continue;
        }
        let mut b = [0u8; 1];
        mem.read_gpa(page_gpas[page as usize] + page_off, &mut b)?;
        out[i as usize] = b[0];
    }
    Ok(out)
}

fn shared_w16<H: HostMemory + HostOps>(host: &mut H, gpa: u64, off: u64, v: u16, page_size: usize) {
    let mut b = [0u8; 2];
    st16(&mut b, v);
    let _ = gpa_map::write_bytes(host, gpa + off, &b, page_size);
}

fn shared_w32<H: HostMemory + HostOps>(host: &mut H, gpa: u64, off: u64, v: u32, page_size: usize) {
    let mut b = [0u8; 4];
    st32(&mut b, v);
    let _ = gpa_map::write_bytes(host, gpa + off, &b, page_size);
}

/// Fill the guest display descriptor page (archive `apple_pv_gpu_display_setup`).
///
///: `+0x208` is the timing-element **count**, not a
/// pixel width. Modes are 1920×1080, 1440×1080, 1280×1024 (apple-gfx A/B
/// reference geometry) plus 3840×2160 (4K UHD), each advertised at
/// `DISPLAY_REFRESH_HZ` (120 Hz), so the guest always latches the 120 Hz mode.
/// Element 0 (1920×1080) stays the native/preferred format (+0x210/+0x212 double
/// as NativeFormat*Pixels), so boot resolution is unchanged and 4K is an
/// additional selectable mode; the dynamic scanout/present/host-window geometry
/// follows the surface the guest actually presents at the selected mode.
fn fill_display_descriptor<H: HostMemory + HostOps>(
    host: &mut H,
    gpa: u64,
    index: u32,
    page_size: u64,
) {
    if gpa == 0 {
        return;
    }
    let Some(refresh) = display_refresh_hz_1616(DISPLAY_REFRESH_HZ) else {
        return;
    };
    let psz = page_size as usize;

    shared_w32(host, gpa, DISPLAY_DESC_SERIAL, DISPLAY_SERIAL_NUMBER, psz);
    let _ = gpa_map::write_bytes(
        host,
        gpa + DISPLAY_DESC_PRODUCT_NAME,
        DISPLAY_PRODUCT_NAME,
        psz,
    );
    shared_w16(host, gpa, DISPLAY_DESC_INDEX, index as u16, psz);
    shared_w16(host, gpa, DISPLAY_DESC_WIDTH_MM, DISPLAY_WIDTH_MM, psz);
    shared_w16(host, gpa, DISPLAY_DESC_HEIGHT_MM, DISPLAY_HEIGHT_MM, psz);
    shared_w32(host, gpa, DISPLAY_DESC_FEATURES, 0, psz);

    // HW cursor capability so the guest doorbells glyph/show/move.
    let max_wh = (CURSOR_MAX_DIM & 0xffff) | ((CURSOR_MAX_DIM & 0xffff) << 16);
    shared_w32(host, gpa, DISPLAY_SHARED_CURSOR_MAX_WH, max_wh, psz);
    shared_w32(
        host,
        gpa,
        DISPLAY_SHARED_CURSOR_FEATURES,
        DISPLAY_CURSOR_FEATURE_HW,
        psz,
    );

    const MODES: &[(u16, u16)] = &[
        (DISPLAY_MODE_EFI_W, DISPLAY_MODE_EFI_H),
        (DISPLAY_MODE1_W, DISPLAY_MODE1_H),
        (DISPLAY_MODE2_W, DISPLAY_MODE2_H),
        (DISPLAY_MODE3_W, DISPLAY_MODE3_H),
    ];
    shared_w16(
        host,
        gpa,
        DISPLAY_DESC_TIMING_COUNT,
        MODES.len() as u16,
        psz,
    );

    let mut encoded = [0u8; DISPLAY_DESC_TIMING_STRIDE as usize];
    for (i, &(width, height)) in MODES.iter().enumerate() {
        let Some(off) = display_timing_entry_offset(i as u32, page_size) else {
            return;
        };
        let entry = DisplayTimingEntry {
            width,
            height,
            refresh_1616: refresh,
            tail0: 0,
            tail1: 0,
        };
        encoded.fill(0);
        if !encode_display_timing_entry(&entry, &mut encoded) {
            return;
        }
        let _ = gpa_map::write_bytes(host, gpa + off, &encoded, psz);
    }
}

/// Sample cursor x/y/show from the display shared-state page (GPA +0xe00).
fn sample_cursor_position<M: HostMemory>(state: &mut DeviceState, mem: &M) {
    if state.display.shared_gpa == 0 {
        return;
    }
    let mut pos = [0u8; 4];
    if mem
        .read_gpa(
            state.display.shared_gpa + DISPLAY_SHARED_CURSOR_POS,
            &mut pos,
        )
        .is_err()
    {
        return;
    }
    let packed = ld32(&pos);
    if packed == 0xffff_ffff {
        state.cursor.show = false;
        return;
    }
    state.cursor.x = (packed & 0xffff) as u16;
    state.cursor.y = ((packed >> 16) & 0xffff) as u16;
    let mut show = [0u8; 4];
    if mem
        .read_gpa(
            state.display.shared_gpa + DISPLAY_SHARED_CURSOR_SHOW,
            &mut show,
        )
        .is_ok()
    {
        // Guest may only write a byte; treat non-zero low byte as show.
        state.cursor.show = show[0] != 0 || ld32(&show) != 0;
    }
}

/// Load CmdDisplayCursorGlyph pixels (BGRA guest → ARGB QEMUCursor).
/// Fail-visible, once per reason per boot, for the silent `load_cursor_glyph`
/// drop sites: a malformed cursor-glyph packet leaves the cursor stale/wrong
/// with no log. Cursor glyphs are infrequent (sent when the pointer *image*
/// changes, not per move) but a persistently-bad glyph could repeat, so latch
/// each reason once. Always returns `false` so callers stay `return cg_fail(..)`.
fn cursor_glyph_fail(reason: &'static str, detail: String) -> bool {
    use std::collections::HashSet;
    use std::sync::Mutex;
    static SEEN: Mutex<Option<HashSet<&'static str>>> = Mutex::new(None);
    let mut guard = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if guard.get_or_insert_with(HashSet::new).insert(reason) {
        crate::observe::fail(detail);
    }
    false
}

fn load_cursor_glyph<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &H,
    packet: &Packet,
) -> bool {
    if packet.payload.len() < CURSOR_GLYPH_PAYLOAD_LEN {
        return cursor_glyph_fail(
            "cursor_glyph_short",
            format!(
                "cursor_glyph_fail reason=cursor_glyph_short plen={} need={CURSOR_GLYPH_PAYLOAD_LEN}",
                packet.payload.len()
            ),
        );
    }
    let task_id = ld32(&packet.payload[0x04..]);
    let virtual_offset = u64::from_le_bytes(packet.payload[0x08..0x10].try_into().unwrap());
    let mapped_length = u64::from_le_bytes(packet.payload[0x10..0x18].try_into().unwrap());
    let stride = u64::from_le_bytes(packet.payload[0x18..0x20].try_into().unwrap()) as u32;
    let width = ld16(&packet.payload[0x20..]) as u32;
    let height = ld16(&packet.payload[0x22..]) as u32;
    let hot_x = ld16(&packet.payload[0x24..]) as u32;
    let hot_y = ld16(&packet.payload[0x26..]) as u32;

    if width == 0
        || height == 0
        || width > CURSOR_MAX_DIM
        || height > CURSOR_MAX_DIM
        || stride < width.saturating_mul(CURSOR_GLYPH_BPP)
        || hot_x >= width
        || hot_y >= height
    {
        return cursor_glyph_fail(
            "cursor_glyph_geom",
            format!("cursor_glyph_fail reason=cursor_glyph_geom {width}x{height} stride={stride} hot=({hot_x},{hot_y}) max={CURSOR_MAX_DIM}"),
        );
    }
    let need = (height as u64 - 1)
        .saturating_mul(stride as u64)
        .saturating_add(width as u64 * CURSOR_GLYPH_BPP as u64);
    if mapped_length < need {
        return cursor_glyph_fail(
            "cursor_glyph_mapped_len",
            format!("cursor_glyph_fail reason=cursor_glyph_mapped_len mapped_length={mapped_length} need={need} {width}x{height}"),
        );
    }
    let Some(need_host) = crate::runtime::metal_draw::host_alloc_len(need) else {
        return cursor_glyph_fail(
            "cursor_glyph_alloc",
            format!("cursor_glyph_fail reason=cursor_glyph_alloc need={need}"),
        );
    };

    let mut src = vec![0u8; need_host];
    if crate::runtime::gva_mem::read_task_gva_by_id(
        host,
        &state.tasks,
        task_id,
        virtual_offset,
        &mut src,
        state.page_shift,
    )
    .is_err()
    {
        return cursor_glyph_fail(
            "cursor_glyph_read",
            format!("cursor_glyph_fail reason=cursor_glyph_read task={task_id} voff={virtual_offset:#x} need_host={need_host}"),
        );
    }

    let mut pixels = Vec::with_capacity((width * height) as usize);
    for y in 0..height {
        let row = (y as usize).saturating_mul(stride as usize);
        for x in 0..width {
            let px = row + (x as usize) * CURSOR_GLYPH_BPP as usize;
            if px + 4 > src.len() {
                return cursor_glyph_fail(
                    "cursor_glyph_bounds",
                    format!("cursor_glyph_fail reason=cursor_glyph_bounds px={px} src_len={} {width}x{height} stride={stride}", src.len()),
                );
            }
            let b = src[px];
            let g = src[px + 1];
            let r = src[px + 2];
            let a = src[px + 3];
            // QEMUCursor 0xAARRGGBB
            pixels.push(((a as u32) << 24) | ((r as u32) << 16) | ((g as u32) << 8) | (b as u32));
        }
    }

    state.cursor.width = width as u16;
    state.cursor.height = height as u16;
    state.cursor.hot_x = hot_x as u16;
    state.cursor.hot_y = hot_y as u16;
    state.cursor.pixels = pixels;
    state.cursor.glyph_ready = true;
    sample_cursor_position(state, host);
    true
}

/// Account one accepted present and request a worker→host action boundary.
///
/// Yielding here bounds how far the drain runs ahead of the display consumer.
/// Continuing to consume guest work can fill `pending_frames`, then hold
/// Display0 forever while its frame remains unconsumed.
fn enqueue_present_scanout<H: HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    width: u32,
    height: u32,
) {
    // Two presentation paths, selected by the live window link:
    //
    // - Window active (x86 default): no CPU `ScanoutUpdate`. QEMU runs
    //   `-display none`, no DisplayChangeListener ticks `gfx_update`, and the
    //   surface would be painted for nobody. The window is fed by
    //   `publish_window_frame` from this drain, and the present-completion ack
    //   is re-homed onto the drain tail (see `device_drain`).
    //
    // - No window (arm64 MMIO `-display cocoa`, or `REIMS_VGPU_WINDOW=0`): the QEMU
    //   console IS the display, so every present enqueues the CPU
    //   `ScanoutUpdate` (coalesced latest-wins in the action queue) and the
    //   ack comes from the console paint (`device_scanout_copy`), releasing
    //   `unpainted_presents` + `present_action_pending` there. Skipping the
    //   action here freezes the console at the last pre-boundary early-FB
    //   paint while the guest keeps presenting (live class: arm64 boot
    //   serial-20260723-221445, console stuck on the 15% progress bar while
    //   gen 38 presented the login wallpaper).
    if !state.present.window_active {
        host.enqueue(HostAction::scanout_gen(
            state.present.frame_mapping,
            width,
            height,
            state.present.frame_generation,
        ));
    }
    state.present.unpainted_presents = state.present.unpainted_presents.saturating_add(1);
    state.pending.host_action_yield = true;
}

fn present_page_identity_line(state: &DeviceState, mapping: u32, w: u32, h: u32) -> Option<String> {
    use std::collections::HashSet;
    let named = state.mappings.get(&mapping)?;
    let named_pfns: HashSet<u32> = named
        .page_entries
        .iter()
        .filter(|&&e| e & crate::contract::iosurface_pages::PAGE_ENTRY_VALID != 0)
        .map(|&e| e >> crate::contract::iosurface_pages::PAGE_ENTRY_PFN_SHIFT)
        .collect();
    let mut peers = String::new();
    for (&mid, m) in state.mappings.iter() {
        if mid == mapping
            || !m.has_geom
            || m.width != w
            || m.height != h
            || m.page_entries.is_empty()
        {
            continue;
        }
        let identical = m.page_entries == named.page_entries;
        let overlap = if identical {
            named_pfns.len()
        } else {
            m.page_entries
                .iter()
                .filter(|&&e| e & crate::contract::iosurface_pages::PAGE_ENTRY_VALID != 0)
                .filter(|&&e| {
                    named_pfns
                        .contains(&(e >> crate::contract::iosurface_pages::PAGE_ENTRY_PFN_SHIFT))
                })
                .count()
        };
        if !peers.is_empty() {
            peers.push(',');
        }
        peers.push_str(&format!(
            "mid{mid}:pages={}:overlap={overlap}:ident={}:kind={:?}",
            m.page_entries.len(),
            identical as u8,
            state.surface_write_kind(mid)
        ));
    }
    Some(format!(
        "present_page_identity mid={mapping} {w}x{h} pages={} valid={} map_gen={} kind={:?} peers=[{peers}]",
        named.page_entries.len(),
        named_pfns.len(),
        named.map_generation,
        state.surface_write_kind(mapping)
    ))
}

/// Which of the two present routes a present took, once per distinct route per
/// process.
///
/// Every present captures the surface the transaction names. This line splits
/// them on the named surface's write history anyway: `route=clear_only` is a
/// present whose named mid's most recent write was a `display_clear`/CLEAR
/// Store rather than a draw — the guest asking us to show a surface it has only
/// ever cleared. `route=named` is everything else. The split is the standing
/// measurement of whether that case occurs at all on a given rail; two lines per
/// process at most, which is what makes it safe to leave on.
///
/// **Measured: only `route=named write_kind=Composite`, on 104 x86/Vulkan boots
/// — every boot in the failure log since this line landed.** Not one
/// `route=clear_only`, including a 1766 s session driven through the
/// heavy-Safari residue repro. The dedup is per process, so one line per boot is
/// the whole reading for that boot.
///
/// That is an x86 statement only. `note_surface_clear` marks a mid ClearOnly
/// from a decoded `display_clear`/CLEAR Store, which is not rail-specific — what
/// the measurement shows is that on x86 the guest never *presents* a mid whose
/// most recent write was a Clear. An arm64 reading of this same line is what
/// would say whether that holds everywhere.
fn note_present_route(write_kind: crate::model::SurfaceWriteKind, is_clear_only: bool) {
    use std::sync::Mutex;
    static SEEN: Mutex<Option<std::collections::BTreeSet<bool>>> = Mutex::new(None);
    {
        let mut guard = SEEN.lock().unwrap_or_else(|p| p.into_inner());
        if !guard
            .get_or_insert_with(Default::default)
            .insert(is_clear_only)
        {
            return;
        }
    }
    crate::observe::fail(format!(
        "present_route route={} write_kind={write_kind:?}",
        if is_clear_only { "clear_only" } else { "named" },
    ));
}

fn log_present_page_identity(state: &DeviceState, mapping: u32, w: u32, h: u32) {
    use std::collections::HashSet;
    use std::sync::Mutex;
    static SEEN: Mutex<Option<HashSet<(u32, u32)>>> = Mutex::new(None);
    let Some(named) = state.mappings.get(&mapping) else {
        return;
    };
    let key = (mapping, named.map_generation);
    {
        let mut guard = SEEN.lock().unwrap_or_else(|p| p.into_inner());
        let seen = guard.get_or_insert_with(HashSet::new);
        if seen.len() > 1024 {
            seen.clear();
        }
        if !seen.insert(key) {
            return;
        }
    }
    if let Some(line) = present_page_identity_line(state, mapping, w, h) {
        crate::observe::fail(line);
    }
}

/// Always-on diagnostic: sample spread guest pages of the present-named
/// surface and log BGRA-interpreted content stats.
///
/// Decides whether the guest itself writes frame content into the presented
/// surface's own pages (the pages are pixel storage, so page bytes ARE
/// decoded surface content). Bounded: at most 16 pages per present, logged
/// only when the sampled stats change, capped per mapping. Never selects
/// behavior.
fn log_present_named_page_content<H: HostMemory + HostOps>(
    state: &DeviceState,
    host: &mut H,
    mapping: u32,
) {
    use std::collections::HashMap;
    use std::sync::Mutex;
    type PageContentStats = (usize, u8, u32);
    type PageContentByMapping = HashMap<u32, PageContentStats>;
    static LAST: Mutex<Option<PageContentByMapping>> = Mutex::new(None);
    const SAMPLE_PAGES: usize = 16;
    const LOG_CAP_PER_MID: u32 = 32;
    let Some(m) = state.mappings.get(&mapping) else {
        return;
    };
    if m.page_entries.is_empty() {
        return;
    }
    let n = m.page_entries.len();
    let step = (n / SAMPLE_PAGES).max(1);
    let page_size = state.page_size() as usize;
    let mut buf = vec![0u8; page_size];
    let mut rgb_nz = 0usize;
    let mut max_rgb = 0u8;
    let mut pages_read = 0usize;
    for i in (0..n).step_by(step).take(SAMPLE_PAGES) {
        let Some(gpa) =
            crate::contract::iosurface_pages::entry_gpa_shift(m.page_entries[i], state.page_shift)
        else {
            continue;
        };
        if host.read_gpa(gpa, &mut buf).is_err() {
            continue;
        }
        pages_read += 1;
        let (nz, mx, _) = crate::observe::bgra_rgb_stats(&buf);
        rgb_nz += nz;
        max_rgb = max_rgb.max(mx);
    }
    let mut guard = LAST.lock().unwrap_or_else(|p| p.into_inner());
    let last = guard.get_or_insert_with(HashMap::new);
    let entry = last.entry(mapping).or_insert((usize::MAX, 0, 0));
    if entry.2 >= LOG_CAP_PER_MID || (entry.0 == rgb_nz && entry.1 == max_rgb) {
        return;
    }
    *entry = (rgb_nz, max_rgb, entry.2 + 1);
    crate::observe::fail(format!(
        "present_named_pages mid={mapping} sampled={pages_read}/{SAMPLE_PAGES} step={step} rgb_nz={rgb_nz} max_rgb={max_rgb} map_gen={}",
        m.map_generation
    ));
}

/// Present a named mapping to the host console (DisplaySwap / x86 present op6/7).
fn present_named_mapping<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    channel_id: u32,
    mapping: u32,
) -> ChildPacketDisposition {
    if mapping == 0 {
        return ChildPacketDisposition::Complete;
    }
    // Archive apple_pv_gpu_display_swap:
    //   render_wait_surface(s, false, swap->mapping_id);
    //   scanout_present_boundary(...);
    //
    // Plus archive poll_tick Dekker rescue (apple_pv_gpu_poll_tick):
    // guest may publish child work without a doorbell while a drain
    // was in flight. Product has no separate host timer during the
    // DisplaySwap packet; drain **other** child FIFOs (skip mid-
    // packet channel) before and after wait_surface so body-layer
    // draws that land during the wait are frozen into the retain.
    // Never re-enter skip/draining_mask channels (boot wedge).
    // Not gen-stable multi-round; not surface_inflight invent.
    let skip = if state.draining_channel != 0 {
        state.draining_channel
    } else {
        channel_id
    };
    drain_other_child_fifos(state, host, skip);
    // Archive render_wait_surface: already-submitted async jobs only.
    let _ = wait_surface_mapping(state, host, mapping);
    drain_other_child_fifos(state, host, skip);
    // Main-ring Dekker only (not full drain_stranded): guest may
    // publish root control work while child drains ran. Full
    // drain_stranded re-enters this child channel and wedged iBoot
    // (6720ce170). Main drain never re-enters a child mid-packet.
    if state.gfx.control_fifo != 0
        && state
            .gfx
            .fifo_read
            .load(std::sync::atomic::Ordering::Acquire)
            != state.gfx.fifo_written
    {
        drain_main_fifo(state, host);
        // Body-layer child work may be doorbell'd from main packets.
        drain_other_child_fifos(state, host, skip);
    }

    // Preflight translation keeps an EXEC packet at its channel head. If one
    // is still held after all rescue drains, accepting this display packet
    // would publish the prior +0x188 retain before the earlier render packet
    // executes. Leave the display head and stamp untouched. Poll-tick re-drives
    // all active channels; once translation is ready, the EXEC runs and this
    // packet is retried in order without blocking a vCPU or the QEMU main loop.
    let current_bit = 1u32.checked_shl(channel_id).unwrap_or(0);
    let deferred_other = state.translation_deferred_mask & !current_bit;
    if deferred_other != 0 {
        if state.present_translation_hold_mask & current_bit == 0 {
            state.present_translation_holds = state.present_translation_holds.saturating_add(1);
            state.present_translation_hold_mask |= current_bit;
            crate::observe::fail(format!(
                "present_order_hold reason=translation_deferred ch={channel_id} mid={mapping} pending_mask={deferred_other:#x} frame_mapping={} early_front={} count={}",
                state.present.frame_mapping,
                state.present.early_front_mapping,
                state.present_translation_holds
            ));
        }
        return ChildPacketDisposition::Deferred;
    }
    if state.present_translation_hold_mask & current_bit != 0 {
        state.present_translation_hold_mask &= !current_bit;
        crate::observe::off(format!(
            "present_order_release ch={channel_id} mid={mapping} pending_mask={:#x}",
            state.translation_deferred_mask
        ));
    }

    state.present.present_mapping = mapping;
    state.present.host_mapping = mapping;
    state.present.mapping_id = mapping;
    state.present.valid = true;
    // x86: present surface_id → type-4 object-list slot (heap index =
    // IOSurface getSurfaceID). Arm: MappingInternal page-table resolve.
    // Always attempt type-4 when pages empty; then iosfc/mapper path.
    let _ = crate::runtime::objects::ensure_surface_for_present(state, host, mapping);
    let force = state
        .mappings
        .get(&mapping)
        .map(|m| m.mapping_internal != 0)
        .unwrap_or(false);
    if force {
        let _ = crate::runtime::mapper::resolve_mapping_backing(state, host, mapping);
    }
    // Paint only from the presented surface's own geom — never the
    // previous console size fallback (that freezes mode switches).
    // Re-read gen after wait_surface (writebacks may have landed).
    let paint = state.mappings.get(&mapping).and_then(|m| {
        if m.has_geom && m.width > 0 && m.height > 0 {
            Some((m.width, m.height, m.content_generation))
        } else {
            None
        }
    });
    if let Some((w, h, gen)) = paint {
        state.present.width = w;
        state.present.height = h;
        state.present.generation = gen;
        log_present_page_identity(state, mapping, w, h);
        log_present_named_page_content(state, host, mapping);
        // Every present takes one route: capture the surface the transaction
        // named. A ClearOnly present — one whose named mid's most recent write
        // was a `display_clear`/CLEAR Store rather than a draw — used to take a
        // six-way resolver instead, choosing some *other* same-geometry surface
        // on the theory that the named one held nothing. `note_present_route`
        // still names which route each present takes, and its reading is why
        // there is only one left.
        let write_kind = state.surface_write_kind(mapping);
        let is_clear_only = matches!(write_kind, crate::model::SurfaceWriteKind::ClearOnly);
        note_present_route(write_kind, is_clear_only);

        // presentFrame names the front surface (leave-BAR1 boundary) once we
        // have a non-init present. Geom/capture may still fail after this.
        state.present.frame_flush_seen = true;
        // PGDisplay presentFrame **retains** the named surface into
        // +0x188 at present time; encodeCurrentFrame later re-shows
        // that retained surface (hostPresentCount). Guest may recycle
        // the mapping as soon as this packet's stamp completes — so
        // freeze guest pages **now**, after wait_surface drains, not
        // at BH after the stamp (that freezes mid-recycle partials:
        // toolbar-only dual-mid under app load). Mid-writeback Stores
        // must not recapture here — present boundary only.
        //
        // Always-on backing gate: a member presented twice with no full-frame
        // Store naming it in between is being displayed with content the guest
        // never sent for it. That is a real loss of guest work and belongs in
        // the log; nothing here papers over it.
        //
        // The line says "naming this mid" rather than "received", because that
        // is the whole of what `note_present_backing` read: decoded Store
        // bookkeeping, never the resident.
        //
        // WHICH IS WHY IT ALSO HAS TO READ THE CARRIER. The gate's witness is
        // `dense_frame_seq`, advanced only by `publish_surface_store` — i.e. when
        // a Store's pixels reached the mapping's GUEST PAGES. The resident rail
        // renders into the registry and skips that write, so "no full frame was
        // published for this mid" no longer implies "nothing can show one". A
        // 524 s boot measured four `reason=…never_stored` lines, each claiming
        // the surface was uninitialized and therefore black, against exactly one
        // `host_window_slate*` line in the whole run — a `covered=1` boot run at
        // t=22 s — with `presents == offered` and `direct_frac=1.00` in every
        // cadence window bracketing all four. A resident carried every one of
        // them. The message asserted a visual consequence the check cannot see,
        // which is "a reason the caller writes is not a reading" applied to an
        // outcome instead of a cause.
        //
        // So ask the presenter's own question, through the rule it shares
        // (`pools::slot_presentable`), and split on the answer the same way
        // `host_window_slate` / `host_window_slate_end` already split: a present
        // nothing can carry is a black frame and belongs on the failure channel;
        // one a resident carries cost no guest work and is a census. Reporting
        // both as black cries wolf every boot and — worse — leaves the real case
        // indistinguishable from the benign one, which is how a genuine
        // black-screen boot once produced zero lines here.
        //
        // Priced where it runs: one registry lookup under the engine lock, inside
        // the arm, so only on a present the structural gate has already refused —
        // four times in that boot, not 60 times a second.
        if let Some(backing) = state.note_present_backing(mapping) {
            let carried = present_resident_carries(state, mapping, w, h);
            let emit = crate::observe::Emit::decline("present_unbacked", &backing)
                .field("mid", mapping)
                .field("geom", format!("{w}x{h}"))
                .field("gen", gen)
                .field("carried", carrier_word(carried));
            if unbacked_present_is_a_loss(carried) {
                emit.fail();
            } else {
                emit.off();
            }
        }
        // The transaction payload carries exactly one field: plane 0's surface
        // id. So the capture source is the surface the guest named, and no
        // comparison between our own full-frame sequences may override it.
        // Presenting a "denser" same-geometry peer instead shows a buffer one
        // rotation step behind the one the guest asked for — residue when a
        // window closed in between, a stale region when one moved, thrash as
        // the choice oscillates.
        let encoded = crate::runtime::scanout::capture_present_frame(state, mapping, w, h, gen);
        if !encoded {
            // Retry encode at first host paint. Do **not** clear
            // frame_valid: PGDisplay keeps the prior presentFrame
            // (+0x188) for hostPresentCount until a new capture
            // succeeds. Invalidating the retain forced a black /
            // empty console when dual-mid page resolve raced.
            state.present.frame_encode_pending = true;
            let (pages, mapped, fmt) = state
                .mappings
                .get(&mapping)
                .map(|m| (m.page_entries.len(), m.mapped as u8, m.format))
                .unwrap_or((0, 0, 0));
            crate::observe::fail(format!(
                "present capture fail mid={mapping} {w}x{h} gen={gen} \
                 keep_prior={} pages={pages} mapped={mapped} fmt={fmt:#x}",
                state.present.frame_valid as u8
            ));
        } else {
            // One pass. `bgra_rgb_stats` already maxes the same
            // `px[0].max(px[1]).max(px[2])` per pixel, so a separate scan for
            // `max_rgb` was a second full 8 MiB walk of the frame, under the
            // device lock, for a value this call already returns.
            let (rgb_nz, max_rgb, px0) = crate::observe::bgra_rgb_stats(&state.present.frame_bgra);
            let verdict = present_content_verdict(&state.present.frame_bgra, max_rgb);
            if verdict == PresentContentVerdict::Unsampled {
                // Not a decline: the dmabuf rail carried the frame, so there are
                // no CPU pixels to judge and no guest work was lost.
                // `present_black` below is the alarm. On that rail this is the
                // normal outcome of every present.
                crate::observe::line(format!(
                    "present_content_unsampled mid={mapping} {w}x{h} gen={gen} \
                     (dmabuf carried the frame; no CPU pixels to judge)"
                ));
            } else if verdict == PresentContentVerdict::Black {
                // Measure dual-mid: other same-geom host_caches with visible RGB
                // while the named mid freezes black (console stays black).
                let mut peers = String::new();
                for (&mid, e) in state.host_surfaces.iter() {
                    if mid == mapping || e.width != w || e.height != h || e.bgra.is_empty() {
                        continue;
                    }
                    let (pnz, pmax, _) = crate::observe::bgra_rgb_stats(&e.bgra);
                    if pmax > 0 && pnz > 10_000 {
                        if !peers.is_empty() {
                            peers.push(',');
                        }
                        peers.push_str(&format!(
                            "mid{mid}:rgb_nz={pnz}:max_rgb={pmax}:hgen={}",
                            e.host_gen
                        ));
                    }
                }
                crate::observe::off(format!(
                    "present_black mid={mapping} {w}x{h} gen={gen} rgb_nz={rgb_nz} px0=[{},{},{},{}] (QMP will be black) peers=[{peers}]",
                    px0[0], px0[1], px0[2], px0[3]
                ));
                crate::observe::fail(format!(
                    "present_black_retain mid={mapping} {w}x{h} gen={gen} (alpha-only/black +0x188) peers=[{peers}]"
                ));
            } else {
                crate::observe::off(format!(
                    "present_content mid={mapping} {w}x{h} gen={gen} rgb_nz={rgb_nz} max_rgb={max_rgb} px0=[{},{},{},{}] encoded={}",
                    px0[0], px0[1], px0[2], px0[3], encoded as u8
                ));
            }
        }
        // No guest-page comparison here. The presented surface's guest window is
        // stale by construction on the Vulkan rail — `import_present` defers the
        // compositor front buffer's writeback on every present, so the pinned
        // resident is authoritative and those pages hold pre-dispatch bytes
        // until a host path reads them. Measured: ~99.5% of the frame differs at
        // full swing on every present, with a deferred window armed every time.
        // The guest's `screencapture` is an oracle because it makes the guest
        // re-execute the composite; its memory for a surface we render into is
        // not.
        // One line per accepted present, verbose-only. `present_enqueue` carried
        // the same fields through the always-on sink alongside it.
        crate::observe::line(format!(
            "present paint mid={mapping} {w}x{h} gen={gen} encoded={} retain={} unpainted={}",
            encoded as u8,
            state.present.frame_valid as u8,
            state.present.unpainted_presents.saturating_add(1)
        ));
        // Account the accepted present. The retain-vs-DisplaySwap (mapping,
        // generation) choice that used to be computed here addressed
        // `copy_to_bgra8`'s Unchanged/expected_generation checks on the QEMU
        // paint; with no paint action produced, the window resolves the frame
        // from `state.present` directly and the distinction has no consumer.
        enqueue_present_scanout(state, host, w, h);
        // Entry-side waitForPendingFrames / apple-gfx pending_frames:
        // count accepted presents until host paint. Stamp still
        // fires with this packet (below) — PGDisplay completion
        // after +0x188 retain, not after host encode.
    } else {
        // Named present without geom: still a product present attempt — leave
        // BAR1 (not a ClearOnly-init handoff defer, which requires geom).
        // Keep early_front peer tracker for dual-mid ClearOnly presents.
        state.present.frame_flush_seen = true;
    }
    // else: hold last painted console (no HostAction / no resize).

    // PGDisplay completion block runs for every present after the
    // +0x188 retain (also when geometry held the paint): display
    // shared-page present bit + conditional display IRQ.
    signal_display_present_complete(state, host);
    ChildPacketDisposition::Complete
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ChildPacketDisposition {
    Complete,
    Deferred,
}

fn exec_summary(channel_id: u32, result: &crate::runtime::exec::ExecResult, plen: usize) -> String {
    format!(
        "exec_indirect2 ch={channel_id} task={} streams={} saw_draw={} clears={} draws_ok={} draws_fail={} rt_resolves={} guest_stores={} icb_ok={} icb_fail={} compute_ctrl_fail={} compute_icb_fail={} render_unbinds={}/{}/{} total_us={} plen={plen}",
        result.task_id,
        result.streams_loaded,
        result.saw_draw as u8,
        result.clears_applied,
        result.metal_draws_ok,
        result.metal_draws_fail,
        result.render_attachment_resolves,
        result.render_guest_stores,
        result.render_icb_ok,
        result.render_icb_fail,
        result.compute_control_fail,
        result.compute_icb_fail,
        result.buffer_unbinds,
        result.texture_unbinds,
        result.sampler_unbinds,
        result.total_us,
    )
}

/// A synchronous ExecIndirect2 holding `DeviceInner` for this long starves the
/// guest's read-to-clear completion/status registers. This is a diagnostic
/// proxy only; it never changes packet ordering or completion behavior.
const SYNC_EXEC_STALL_US: u64 = 250_000;

#[inline]
fn sync_exec_stalled(total_us: u64) -> bool {
    total_us >= SYNC_EXEC_STALL_US
}

fn process_child_packet<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    channel_id: u32,
    packet: &Packet,
) -> ChildPacketDisposition {
    match packet.opcode {
        CHILD_OP_DEFINE_TASK2 => {
            if packet.payload.len() >= DEFINE_TASK_LEN {
                let raw_id = ld32(&packet.payload[DEFINE_TASK_RAW_ID..]);
                let length = u64::from(ld32(&packet.payload[DEFINE_TASK_LENGTH..]));
                let dir = ld32(&packet.payload[DEFINE_TASK_DIRECTORY_PFN..]);
                let task_id = raw_id >> DEFINE_TASK_ID_SHIFT;
                let ok = state.define_task(task_id, length, dir);
                if ok && (task_id as usize) < state.tasks.len() {
                    let slot = &state.tasks[task_id as usize];
                    let walk = crate::runtime::gva_mem::diagnose_task_slot(
                        host,
                        slot,
                        task_id,
                        0,
                        state.page_shift,
                    );
                    crate::observe::off(format!(
                        "define_task child ch={channel_id} raw={raw_id:#x} task={task_id} len={length:#x} dir={dir:#x} page_shift={} {walk}",
                        state.page_shift
                    ));
                }
            } else {
                // A short DEFINE_TASK2 silently drops the task definition — every
                // subsequent draw/resolve on that task then fails downstream with
                // no root cause. Never happens on a well-formed boot.
                crate::observe::fail(format!(
                    "child_packet_short reason=define_task2_short ch={channel_id} plen={} need={DEFINE_TASK_LEN}",
                    packet.payload.len()
                ));
            }
        }
        CHILD_OP_SET_OBJECT_LIST => {
            if packet.payload.len() >= SET_OBJECT_LIST_LEN {
                let task_id = ld32(&packet.payload[SET_OBJECT_LIST_TASK_ID..]);
                let pfn = ld32(&packet.payload[SET_OBJECT_LIST_PFN..]);
                let count = ld32(&packet.payload[SET_OBJECT_LIST_COUNT..]);
                let _ = state.set_object_list(task_id, pfn, count);
            } else {
                // A short SET_OBJECT_LIST silently leaves the task's object list
                // unbound — every type-11 texture/object resolve on it then
                // fails (object_list_count==0). Never on a well-formed boot.
                crate::observe::fail(format!(
                    "child_packet_short reason=set_object_list_short ch={channel_id} plen={} need={SET_OBJECT_LIST_LEN}",
                    packet.payload.len()
                ));
            }
        }
        CHILD_OP_DELETE_OBJECT => {
            if packet.payload.len() >= 8 {
                let task_id = ld32(&packet.payload[0..]);
                let id = ld32(&packet.payload[4..]);
                let _ = state.delete_object(task_id, id);
            }
        }
        // PVG CmdDeleteTask (0x20) on child channels too (was SMALL_ID alias only in decode).
        CHILD_OP_DELETE_TASK => {
            let task_id = if packet.payload.len() >= 4 {
                ld32(&packet.payload[0..])
            } else {
                0
            };
            let ok = state.delete_task(task_id);
            crate::observe::off(format!(
                "delete_task ch={channel_id} task={task_id} ok={} plen={}",
                ok as u8,
                packet.payload.len()
            ));
        }
        CHILD_OP_SETUP_SHARED_STATE => {
            if packet.payload.len() >= CHILD_SHARED_STATE_LEN {
                let index = ld32(&packet.payload[CHILD_SHARED_STATE_INDEX..]);
                let pfn = ld32(&packet.payload[CHILD_SHARED_STATE_PFN..]);
                // reinit=1 means the guest tears down + re-registers the display
                // shared page while it was already ONLINE — the AppleParavirtDisplayPipe
                // setupSharedState/teardownSharedState re-init that makes WindowServer
                // rebuild display attributes (signalDisplay bit2 → process_online).
                // A reinit AFTER present_converge is the smoking gun for the intermittent
                // post-converge boot-progress overlay. Rare
                // event → always-on so a bad boot leaves a display-lifecycle timeline.
                let reinit = state.display.online_acked as u8;
                state.display.display_index = index;
                state.display.shared_gpa = state.pfn_gpa(pfn);
                state.display.online_acked = false;
                state.display.online_tries = 0;
                state.display.poll_ctr = 0;
                crate::observe::fail(format!(
                    "display_shared_state_setup index={index} gpa={:#x} reinit={reinit}",
                    state.display.shared_gpa
                ));
                // Archive apple_pv_gpu_display_setup: fill descriptor + modes
                // before completion so createDisplayAttributes sees TimingElements.
                // Do **not** pulse ONLINE here — enable() has not set +0x104 yet
                // (archive poll waits for mask bit 2, then pending+IRQ).
                fill_display_descriptor(host, state.display.shared_gpa, index, state.page_size());
            } else {
                // A short SETUP_SHARED_STATE silently drops display registration:
                // shared_gpa/index never latch, so the display NEVER onlines and
                // the boot wedges on a blank/console frame with no root cause.
                // The single loudest silent-drop in the pipeline. Never on a
                // well-formed boot.
                crate::observe::fail(format!(
                    "child_packet_short reason=setup_shared_state_short ch={channel_id} plen={} need={CHILD_SHARED_STATE_LEN}",
                    packet.payload.len()
                ));
            }
        }
        CHILD_OP_ONLINE_ACK => {
            state.display.online_acked = true;
            // The connectionChange-ack (process_online opcode 2) is believed to
            // echo the shared-descriptor `+0x200` token back to the host in its
            // payload. We consume the ack (online_acked)
            // but never inspect that token — capture it here (raw first words +
            // len, rare/once-per-online so no flood) so a bad boot records what
            // value the guest round-tripped. Measure-only.
            let w0 = if packet.payload.len() >= 4 {
                ld32(&packet.payload[0..])
            } else {
                0
            };
            let w1 = if packet.payload.len() >= 8 {
                ld32(&packet.payload[4..])
            } else {
                0
            };
            crate::observe::fail(format!(
                "display_online_ack index={} plen={} w0={:#x} w1={:#x}",
                state.display.display_index,
                packet.payload.len(),
                w0,
                w1
            ));
        }
        /*
         * Scanout policy:
         * - Early boot: front type-11 writebacks paint while !frame_flush_seen
         *   and job W×H matches established console (no mid-switch thrash).
         * - After first boundary: display presents paint (op8 DisplaySwap on
         *   arm ch4, **or** op6/7 on x86 Ventura/Tahoe display ch5).
         * - ch2 PRESENT_FRAME 0x28 / FLUSH 0x3b: bookkeeping only (mid-composite).
         */
        CHILD_OP_DISPLAY_SWAP => {
            note_display_txn_payload(state, channel_id, packet);
            let mapping = if packet.payload.len() >= DISPLAY_SWAP_MIN_LEN {
                ld32(&packet.payload[DISPLAY_SWAP_MAPPING..])
            } else {
                0
            };
            // Offline: display index @0 vs mapping @8 (arm op8).
            if packet.payload.len() >= 12 {
                let disp = ld32(&packet.payload[DISPLAY_SWAP_DISPLAY..]);
                // Per-present decode census (~30k/session under animation);
                // the present rate lives in the present_proxy summary, so gate
                // the per-packet line behind REIMS_VGPU_DRAW_LOG.
                crate::observe::line(format!(
                    "present_op8 ch={channel_id} disp={disp} mid={mapping} plen={} unpainted={}",
                    packet.payload.len(),
                    state.present.unpainted_presents
                ));
            }
            if present_named_mapping(state, host, channel_id, mapping)
                == ChildPacketDisposition::Deferred
            {
                return ChildPacketDisposition::Deferred;
            }
        }
        // x86 Ventura/Tahoe display pipe: present is opcode 6/7 (not 0x28).
        // Live fail log: UnknownChildOpcode ch5 op6 — no paint until handled.
        CHILD_OP_PRESENT_X86 => {
            note_display_txn_payload(state, channel_id, packet);
            let mapping = if packet.payload.len() >= PRESENT_X86_MIN_LEN {
                ld32(&packet.payload[PRESENT_X86_SURFACE_ID..])
            } else {
                0
            };
            // op6 trailer: [pipe_index@0][surface_id@4][task@8]. The word at +8
            // is the submitting task's field, not a completion stamp — the
            // packet's own stamp lives in the FIFO header.
            if packet.payload.len() >= 12 {
                let disp = ld32(&packet.payload[0..]);
                let task = ld32(&packet.payload[8..]);
                crate::observe::line(format!(
                    "present_op6 ch={channel_id} pipe={disp} sid={mapping} task={task:#x} plen={} unpainted={} prior_present_mapping={}",
                    packet.payload.len(),
                    state.present.unpainted_presents,
                    state.present.present_mapping
                ));
            }
            if present_named_mapping(state, host, channel_id, mapping)
                == ChildPacketDisposition::Deferred
            {
                return ChildPacketDisposition::Deferred;
            }
        }
        CHILD_OP_PRESENT_GAMMA_X86 => {
            note_display_txn_payload(state, channel_id, packet);
            let mapping = if packet.payload.len() >= PRESENT_GAMMA_X86_SURFACE_ID + 4 {
                ld32(&packet.payload[PRESENT_GAMMA_X86_SURFACE_ID..])
            } else if packet.payload.len() >= PRESENT_X86_MIN_LEN {
                ld32(&packet.payload[PRESENT_X86_SURFACE_ID..])
            } else {
                0
            };
            if packet.payload.len() >= 12 {
                crate::observe::line(format!(
                    "present_op7 ch={channel_id} sid={mapping} plen={} unpainted={}",
                    packet.payload.len(),
                    state.present.unpainted_presents
                ));
            }
            if present_named_mapping(state, host, channel_id, mapping)
                == ChildPacketDisposition::Deferred
            {
                return ChildPacketDisposition::Deferred;
            }
        }
        CHILD_OP_PRESENT_FRAME => {
            // PVG: CmdDeleteObject on some maps; arm misread as present. Never
            // paint. x86 present is op6/7 on display channel.
            let _ = packet;
        }
        // PVG / Monterey: 0x3b = CmdGetComputeInfo (query). Must write reply
        // before stamp or createComputePipeline stalls (texture-ref 29-06-26).
        // `CHILD_OP_PRESENT_FRAME_FLUSH` is the recovered legacy name for the
        // same wire opcode.
        CHILD_OP_GET_COMPUTE_INFO => {
            if packet.payload.len() >= 24 {
                let _ = reply_compute_info(state, host, &packet.payload);
            } else {
                crate::observe::fail(format!(
                    "get_compute_info short ch={channel_id} len={}",
                    packet.payload.len()
                ));
            }
        }
        CHILD_OP_CURSOR_SHOW => {
            if packet.payload.len() >= 8 {
                let show = ld32(&packet.payload[4..]) != 0;
                state.cursor.show = show;
                sample_cursor_position(state, host);
                host.enqueue(HostAction::cursor(state.cursor.x, state.cursor.y, show));
            }
        }
        CHILD_OP_CURSOR_GLYPH => {
            if load_cursor_glyph(state, host, packet) {
                host.enqueue(HostAction::cursor_glyph());
                host.enqueue(HostAction::cursor(
                    state.cursor.x,
                    state.cursor.y,
                    state.cursor.show,
                ));
            }
        }
        CHILD_OP_EXEC_INDIRECT2 => {
            if packet.payload.len() < 12 {
                state.record_fail(FailEvent::UnsupportedExec {
                    channel: channel_id,
                    fault: ExecFault::Indirect2Short,
                });
            } else {
                // Process this channel's exec packet. Archive does not drain
                // other child FIFOs here; surface RAW is render_wait_surface on
                // the specific type-11/GVA key at sample/Load/swap sites.
                let result =
                    crate::runtime::exec::process_exec_indirect2(state, host, &packet.payload);
                let channel_bit = 1u32.checked_shl(channel_id).unwrap_or(0);
                if result.deferred {
                    if channel_bit != 0 && state.translation_deferred_mask & channel_bit == 0 {
                        state.translation_deferred_mask |= channel_bit;
                        // Census for the same reason as `translation_order_hold`:
                        // the packet is NOT consumed (`Deferred` leaves it at the
                        // FIFO head to be retried), and the matching
                        // `exec_translation_ready` below is already `off`. Boot 87:
                        // 55 deferrals, 56 readies.
                        crate::observe::off(format!(
                            "exec_translation_deferred reason=air_loading ch={channel_id} task={} pending_mask={:#x}",
                            result.task_id, state.translation_deferred_mask
                        ));
                    }
                    return ChildPacketDisposition::Deferred;
                }
                if channel_bit != 0 && state.translation_deferred_mask & channel_bit != 0 {
                    state.translation_deferred_mask &= !channel_bit;
                    crate::observe::off(format!(
                        "exec_translation_ready ch={channel_id} task={} pending_mask={:#x}",
                        result.task_id, state.translation_deferred_mask
                    ));
                }
                // Failure-carrying packets keep the full per-packet line on the
                // always-on sink (context for the per-site reason=<slug> lines).
                // Healthy packets are expected control flow and stay quiet
                // unless the draw log is on — the per-packet form ran ~1k
                // lines/s under Safari scroll.
                let packet_failed = result.metal_draws_fail > 0
                    || result.render_icb_fail > 0
                    || result.compute_control_fail > 0
                    || result.compute_icb_fail > 0;
                if packet_failed {
                    crate::observe::fail(exec_summary(channel_id, &result, packet.payload.len()));
                } else if crate::observe::draw_log_enabled() {
                    crate::observe::line(exec_summary(channel_id, &result, packet.payload.len()));
                }
                if sync_exec_stalled(result.total_us) {
                    crate::observe::fail(format!(
                        "TRANSPORT reason=sync_exec_lock_hold ch={channel_id} task={} total_us={} draws={} rt_resolves={} guest_stores={} threshold_us={SYNC_EXEC_STALL_US}",
                        result.task_id,
                        result.total_us,
                        result.metal_draws_ok.saturating_add(result.metal_draws_fail),
                        result.render_attachment_resolves,
                        result.render_guest_stores
                    ));
                }
            }
        }
        crate::runtime::decode::fifo::CHILD_OP_CONFIG_40 => {
            let _ = reply_heap_texture_size_and_align(state, host, &packet.payload);
        }
        // PVG bookkeeping family: accept + stamp (already below). Full PT/map
        // semantics land with metal2vulkan encode; until then fail-visible
        // UnknownChildOpcode flooded /tmp/reims-vgpu-fail and hid draw telemetry.
        CHILD_OP_UNMAP_MEMORY
        | CHILD_OP_MAP_MEMORY2
        | CHILD_OP_INVALIDATE_RESOURCES
        | CHILD_OP_SYNCHRONIZE_RESOURCES
        | CHILD_OP_DELETE_IOSURFACE_BACKING2
        | CHILD_OP_REPLACE_PHYSICAL => {
            // Stamp-complete for PT wire (no invent). Unmap/Map retire
            // gva_host_views; verbose-gated map_probe census for stage Unmapped.
            //
            // Live MapMemory2 plen=20 layout lead (not yet contract-final):
            //   task_id@0 u32, gva@4 u64, length@12 u64  (matches fifo MapMemoryCommand).
            let plen = packet.payload.len();
            let name = match packet.opcode {
                CHILD_OP_MAP_MEMORY2 => "MapMemory2",
                CHILD_OP_REPLACE_PHYSICAL => "ReplacePhysical",
                CHILD_OP_UNMAP_MEMORY => "UnmapMemory",
                CHILD_OP_INVALIDATE_RESOURCES => "InvalidateResources",
                CHILD_OP_SYNCHRONIZE_RESOURCES => "SynchronizeResources",
                CHILD_OP_DELETE_IOSURFACE_BACKING2 => "DeleteIOSurfaceBacking2",
                _ => "map_family",
            };
            if matches!(packet.opcode, CHILD_OP_MAP_MEMORY2 | CHILD_OP_UNMAP_MEMORY) && plen >= 20 {
                let task_id = crate::contract::endian::ld32(&packet.payload[0..]);
                let gva = crate::contract::endian::ld64(&packet.payload[4..]);
                let length = crate::contract::endian::ld64(&packet.payload[12..]);
                // Verbose-gated walk probe at map/unmap time. This runs a full
                // guest page-table walk (`diagnose_gva_walk`) purely to build the
                // log string, and fired ~9k times/boot on the drain path — a flood
                // and a real per-map cost. Gate it (and the periodic census) behind
                // `REIMS_VGPU_DRAW_LOG=1` so a normal boot pays neither; the functional
                // view-retire below stays always-on. Wire has no PPNs — the probe
                // asks whether the guest PT is already walkable under wire task_id.
                if crate::observe::draw_log_enabled() {
                    let walk = crate::runtime::gva_mem::diagnose_gva_walk(
                        host,
                        &state.tasks,
                        task_id,
                        gva,
                        state.page_shift,
                    );
                    crate::observe::line(format!(
                        "map_probe op={name} ch={channel_id} task={task_id} gva={gva:#x} len={length:#x} page_shift={} {walk}",
                        state.page_shift
                    ));
                    // Periodic active-task census (every 32 map/unmap) for boot overview.
                    state.map_family_events = state.map_family_events.saturating_add(1);
                    if state.map_family_events == 1 || state.map_family_events.is_multiple_of(32) {
                        let census = crate::runtime::gva_mem::format_active_tasks(&state.tasks);
                        crate::observe::line(format!(
                            "map_census n={} last_op={name} task={task_id} {census}",
                            state.map_family_events
                        ));
                    }
                }
                // RE (AppleParavirtMemoryMap): Unmap/Map only mutate the **task
                // page table** then notify — wire has no PPNs. Guest order is
                // deallocate/allocate **then** FIFO, so:
                // - Unmap notify: PTEs already gone → cannot GVA-write; retain
                //   host_gva_surfaces for sample (wallpaper wipe class).
                // - Map notify: PTEs already live → flush host_gva encode into
                //   **new** PFNs (not invent PTEs; not invent geom). Discrete
                //   type-2/3 content may live only in host_cache until this.
                // Samples still prefer host_cache GVA key on Load.
                //
                // HostOps **views** (gva_host_views) are the opposite of encode
                // cache: they alias the pages that were in the GPU PT. On Unmap
                // those pages are no longer mapped for the GPU — drop any host
                // view covering the range (Apple unmapMemory analogue). On Map
                // the PFNs may have changed under the same GVA — drop stale
                // views so the next ensure_gva_view re-walks. Does not invent
                // PTEs and does not destroy host_gva_surfaces content.
                if gva != 0 && length != 0 {
                    let n = crate::runtime::gva_view::retire_gva_views_overlapping(
                        state, task_id, gva, length,
                    );
                    let op = if packet.opcode == CHILD_OP_UNMAP_MEMORY {
                        "unmap_memory"
                    } else {
                        "map_memory2"
                    };
                    crate::runtime::gva_view::log_retire(op, task_id, gva, length, n);
                    // Registry for product GVA write bounds (notify ranges only).
                    if packet.opcode == CHILD_OP_UNMAP_MEMORY {
                        state.note_task_unmap(task_id, gva, length);
                    } else {
                        // Always-on, once per distinct key: the payload word this
                        // opcode files the span under, unfiltered.
                        //
                        // `task_id` here IS the raw word — this opcode reads it
                        // unshifted while `DefineTask2` halves its own, and the
                        // write gate is observed permitting writes for task `n`
                        // against spans filed under `n >> 1`. Deciding whether
                        // that is the two decodes disagreeing or a real
                        // parent/child ownership needs the *set* of keys this
                        // registry holds, compared against the set of
                        // `define_task root raw=…` words. The neighbouring
                        // `map_memory2` retire line cannot answer it: that one
                        // only prints when views were actually retired, so it
                        // shows a filtered subset of the keys.
                        if crate::observe::first_sight("map_memory2_key", u64::from(task_id)) {
                            crate::observe::off(format!(
                                "map_memory2_key word={task_id:#x} dec={task_id} \
                                 gva={gva:#x} len={length:#x}"
                            ));
                        }
                        state.note_task_map(task_id, gva, length);
                    }
                }
                // Deferred GVA render-Store windows overlapping the notified
                // VA range land **cache-only**: on Unmap the PTEs are already
                // gone; on Map the PFNs are fresh and the map-notify guest
                // flush is forbidden (PTE-corruption class). The encode cache
                // preserves the content for samples (wallpaper-retain).
                if gva != 0 && length != 0 && !state.gva_deferred_flush.is_empty() {
                    let hi = gva.saturating_add(length);
                    let overlapped: Vec<u64> = state
                        .gva_deferred_flush
                        .iter()
                        .filter(|(&wgva, e)| {
                            // This task's windows only. A GVA means nothing
                            // outside the address space that named it, so the
                            // overlap test is an overlap only once the task
                            // matches — and both sides are slot ids
                            // (`task_slot::resolve_task_word` on one, the
                            // unshifted `MapMemory2`/`UnmapMemory` word on the
                            // other). The `>> 1` arms this replaces also matched
                            // slots `task_id / 2`, `2 * task_id` and
                            // `2 * task_id + 1`: live, unrelated tasks whose
                            // pending frames were then landed cache-only and so
                            // never reached guest RAM.
                            e.task_id == task_id
                                && wgva < hi
                                && gva < wgva.saturating_add(e.span())
                        })
                        .map(|(&wgva, _)| wgva)
                        .collect();
                    for wgva in overlapped {
                        let trigger = if packet.opcode == CHILD_OP_UNMAP_MEMORY {
                            "unmap"
                        } else {
                            "remap"
                        };
                        crate::runtime::storage_flush::flush_gva_exact(
                            state, host, wgva, false, trigger,
                        );
                    }
                }
                // There is deliberately no host_cache→guest GVA flush on
                // MapMemory2. One existed and was disabled after
                // serial-20260714-035023: PTE Corruption (freelist-shaped
                // 0xff100000ff000000) ~135s into boot while it was writing —
                // one Map of len=0x1c3e000 alone drove 13 GVA rewrites. Samples
                // use the `host_gva_surfaces` retain on Unmap instead. Any
                // re-introduction has to be a *narrower* policy than that one
                // (exact-base only, no multi-key heap maps) and RE-justified, so
                // the broad implementation is not kept around to be switched
                // back on. See kb map-memory2 / xnu-pte-corruption-windowserver.
            } else if packet.opcode == CHILD_OP_DELETE_IOSURFACE_BACKING2 && plen >= 8 {
                // Live Ventura payload + current-kext symbol agree with the
                // resource contract: `{objectID, taskID}`. This is the lifetime
                // boundary for the host IOSurface backing, not stamp-only
                // bookkeeping. Keeping page_entries after it lets later id
                // reuse/clear write pixels into pages the guest has recycled.
                let object_id = crate::contract::endian::ld32(&packet.payload[0..]);
                let task_id = crate::contract::endian::ld32(&packet.payload[4..]);
                let _ = wait_surface_mapping(state, host, object_id);
                // Never write guest pages here — the delete trails the guest's
                // CPU-side release asynchronously and the pages may already be
                // recycled (boot-16 PTE-corruption panic: a 14.7 MB delete-time
                // flush landed pixel bytes in a PTE page). But the id itself
                // may ALSO already be re-used by a live surface whose paint is
                // still deferred (~20 ms recycle under scroll — black-band
                // class), so content state must survive until the next page
                // resolve proves which incarnation this delete was for
                // (fingerprint compare in mapper::resolve). A second delete
                // with no resolve between is genuinely dead: tear down fully.
                let mode = if state.mapping_backing_condemned(object_id) {
                    crate::runtime::storage_flush::drop_windows(state, object_id, "delete_backing");
                    let _ = state.unmap_surface(object_id);
                    "dead"
                } else if state.condemn_surface_backing(object_id) {
                    "condemn"
                } else {
                    // No resolved pages ⇒ nothing a stale delete could hurt.
                    crate::runtime::storage_flush::drop_windows(state, object_id, "delete_backing");
                    let _ = state.unmap_surface(object_id);
                    "unmapped"
                };
                crate::runtime::mapper::flush_retired_views(state, host);
                if crate::observe::draw_log_enabled() {
                    crate::observe::line(format!(
                        "map_family op=DeleteIOSurfaceBacking2 ch={channel_id} object={object_id} task={task_id} plen={plen} mode={mode}"
                    ));
                }
            } else if packet.opcode == CHILD_OP_REPLACE_PHYSICAL && plen >= 8 {
                // Archived lead: {taskID, objectID}; live total_size=20 ⇒ header+payload.
                // Guest may rebind physical pages under the object — drop cached
                // page_entries / contig so the next Store re-resolves (safe
                // zero-copy / freelist-prevention). Object id is typically a
                // texture ref; also try as mapping_id when texture map misses.
                let task_id = crate::contract::endian::ld32(&packet.payload[0..]);
                let object_id = crate::contract::endian::ld32(&packet.payload[4..]);
                // Guest MAY rebind physical pages under the object — retire the
                // cached bindings so the next Store re-resolves (freelist
                // prevention). Like the trailing delete, this must not destroy
                // a live incarnation's deferred paint (a tile's only copy sits
                // in the GPU resident until writeback): condemn with a page
                // fingerprint and let the next resolve decide. A genuine rebind
                // resolves to different pages → bump + windows dropped there;
                // a revalidation/no-op resolves identical → content survives.
                let mut n_inv = 0u32;
                let mut n_cond = 0u32;
                let mut targets = vec![object_id];
                if let Some(&mid) = state.texture_to_mapping.get(&(task_id, object_id)) {
                    if mid != object_id {
                        targets.push(mid);
                    }
                }
                for id in targets {
                    if state.mapping_backing_condemned(id) {
                        // Decision already pending; the next resolve settles it.
                        continue;
                    }
                    if state.condemn_surface_backing(id) {
                        n_cond = n_cond.saturating_add(1);
                    } else {
                        // No resolved pages ⇒ nothing a stale replace could
                        // hurt; keep the old teardown semantics.
                        crate::runtime::storage_flush::drop_windows(state, id, "replace_physical");
                        if state.invalidate_mapping_pages(id) {
                            n_inv = n_inv.saturating_add(1);
                        }
                    }
                }
                // Per-op echo of a routine lifecycle op. Keep the per-op detail
                // (inv/condemn split) gated so it does not flood the always-on
                // sink; the `draw_log_enabled()` guard also skips the format
                // alloc on a healthy boot (mirrors the DeleteIOSurfaceBacking2
                // site above).
                if crate::observe::draw_log_enabled() {
                    crate::observe::line(format!(
                        "map_family op=ReplacePhysical ch={channel_id} task={task_id} object={object_id} plen={plen} inv_pages={n_inv} condemned={n_cond}"
                    ));
                }
            } else if packet.opcode == CHILD_OP_INVALIDATE_RESOURCES {
                // RE: {task_id, count} + count×{object_id, 4×u8 validity ops}.
                // Ops (PVG host layout): clr_host, set_host, clr_guest, set_guest.
                // Pageon hardcodes LE 01 00 00 01 = clr hostValid + set guestValid.
                //
                // The same four bytes the EXEC_INDIRECT2 resource table carries,
                // through the same consumer: this producer's records are 8 bytes
                // and that one's are 24, but the quad is one contract and must
                // not acquire two meanings.
                use crate::runtime::decode::fifo::{
                    decode_invalidate_resources, CHILD_INVALIDATE_PAGEON_FLAGS,
                };
                use crate::runtime::resource_validity::{apply, ValiditySite};
                match decode_invalidate_resources(&packet.payload) {
                    Some(cmd) => {
                        let mut bumped = 0u32;
                        let mut miss = 0u32;
                        let mut windows_dropped = 0u32;
                        for rec in &cmd.records {
                            let outcome = apply(
                                state,
                                cmd.task_id,
                                rec.object_id,
                                rec.ops,
                                ValiditySite::InvalidateResources,
                            );
                            bumped = bumped.saturating_add(outcome.bumped);
                            windows_dropped =
                                windows_dropped.saturating_add(outcome.windows_dropped);
                            if outcome.missed {
                                miss = miss.saturating_add(1);
                            }
                        }
                        // One counter here, two on the exec side: `pageBacking`
                        // names mapping ids, so a record this device holds no
                        // mapping for is already the surprising case. The exec
                        // table names task object refs, most of which have no
                        // surface state by construction.
                        note_store_route_n("validity_miss_inv", miss as u64);
                        let rec0 = cmd.records.first();
                        let oid = rec0.map(|r| r.object_id).unwrap_or(0);
                        let flags = rec0.map(|r| r.flags).unwrap_or(0);
                        let ops = rec0.map(|r| r.ops).unwrap_or_default();
                        let pageon = flags == CHILD_INVALIDATE_PAGEON_FLAGS;
                        // ~11k/boot of routine guest cache-coherence ops. The
                        // always-on rate is the `validity_*` family in the
                        // per-second `store_routes` line; gate the per-op decode
                        // detail so it does not bury the curated fail view. The
                        // `decode_fail` and `inv_multi` paths below stay
                        // fail-visible, and the guard also skips the format alloc
                        // on a healthy boot.
                        if crate::observe::draw_log_enabled() {
                            crate::observe::line(format!(
                            "map_family op=InvalidateResources opcode={:#x} ch={channel_id} plen={plen} task={} count={} oid={oid:#x} flags={flags:#x} clr_h={} set_h={} clr_g={} set_g={} pageon={pageon} bumped={bumped} miss={miss} windows_dropped={windows_dropped}",
                            packet.opcode,
                            cmd.task_id,
                            cmd.count,
                            ops.clear_host_valid,
                            ops.set_host_valid,
                            ops.clear_guest_valid,
                            ops.set_guest_valid
                        ));
                        }
                        if cmd.count > 1 {
                            let ids: Vec<String> = cmd
                                .records
                                .iter()
                                .map(|r| {
                                    format!(
                                        "{:#x}:clr_h={}/set_g={}",
                                        r.object_id, r.ops.clear_host_valid, r.ops.set_guest_valid
                                    )
                                })
                                .collect();
                            crate::observe::fail(format!(
                                "inv_multi ch={channel_id} task={} n={} recs=[{}]",
                                cmd.task_id,
                                cmd.count,
                                ids.join(",")
                            ));
                        }
                    }
                    None => {
                        let w0 = if plen >= 4 {
                            crate::contract::endian::ld32(&packet.payload[0..])
                        } else {
                            0
                        };
                        let w1 = if plen >= 8 {
                            crate::contract::endian::ld32(&packet.payload[4..])
                        } else {
                            0
                        };
                        crate::observe::fail(format!(
                            "map_family op=InvalidateResources opcode={:#x} ch={channel_id} plen={plen} decode_fail w0={w0:#x} w1={w1:#x}",
                            packet.opcode
                        ));
                    }
                }
            } else if packet.opcode == CHILD_OP_SYNCHRONIZE_RESOURCES {
                // RE synchronizeForUnwire → FIFO 0x35: {task,count}+{oid} only.
                // Guest contract is finish host GPU use before pageoff — not
                // "host invents pixels into guest pages." Discrete host_cache→
                // guest write was product invent (pre-change successful boots
                // were stamp-only). Keep decode + wait_surface; no guest write.
                use crate::runtime::decode::fifo::decode_synchronize_resources;
                match decode_synchronize_resources(&packet.payload) {
                    Some(cmd) => {
                        // The guest is about to CPU-read these resources
                        // (pageoff/unwire): land every deferred writeback
                        // (render/compute/linear-alias) into guest pages first
                        // — the only host-visible choke point for guest CPU
                        // reads (boot-25 black-wallpaper class).
                        let mut flushed = 0u32;
                        let mut flush_ok = true;
                        for &oid in &cmd.object_ids {
                            let _gen = wait_surface_mapping(state, host, oid);
                            let (ok, n) =
                                crate::runtime::storage_flush::flush_mapping_for_guest_read(
                                    state, host, oid,
                                );
                            flush_ok &= ok;
                            flushed = flushed.saturating_add(n);
                        }
                        let oid = cmd.object_ids.first().copied().unwrap_or(0);
                        // Count into the always-on teardown-churn proxy; the
                        // per-event census floods to ~49k/session under a
                        // continuously-animating app, so it moves behind
                        // REIMS_VGPU_DRAW_LOG below.
                        // A deferred guest-read flush that did NOT land right
                        // before the guest CPU-reads these pages is a genuine
                        // black/stale-content drop — previously buried in the
                        // off() census (invisible in the curated fail view).
                        // Promote it to a reason-slugged fail line.
                        if !flush_ok {
                            crate::observe::fail(format!(
                                "map_family op=SynchronizeResources reason=guest_read_flush_incomplete ch={channel_id} task={} oid={oid:#x} deferred_flushed={flushed}",
                                cmd.task_id
                            ));
                        }
                        if crate::observe::draw_log_enabled() {
                            crate::observe::line(format!(
                                "map_family op=SynchronizeResources opcode={:#x} ch={channel_id} plen={plen} task={} count={} oid={oid:#x} deferred_flushed={flushed} flush_ok={flush_ok}",
                                packet.opcode, cmd.task_id, cmd.count
                            ));
                        }
                        if cmd.count > 1 {
                            let ids: Vec<String> =
                                cmd.object_ids.iter().map(|id| format!("{id:#x}")).collect();
                            crate::observe::fail(format!(
                                "sync_multi ch={channel_id} task={} n={} oids=[{}]",
                                cmd.task_id,
                                cmd.count,
                                ids.join(",")
                            ));
                        }
                    }
                    None => {
                        let w0 = if plen >= 4 {
                            crate::contract::endian::ld32(&packet.payload[0..])
                        } else {
                            0
                        };
                        let w1 = if plen >= 8 {
                            crate::contract::endian::ld32(&packet.payload[4..])
                        } else {
                            0
                        };
                        crate::observe::fail(format!(
                            "map_family op=SynchronizeResources opcode={:#x} ch={channel_id} plen={plen} decode_fail w0={w0:#x} w1={w1:#x}",
                            packet.opcode
                        ));
                    }
                }
            } else {
                let w0 = if plen >= 4 {
                    crate::contract::endian::ld32(&packet.payload[0..])
                } else {
                    0
                };
                let w1 = if plen >= 8 {
                    crate::contract::endian::ld32(&packet.payload[4..])
                } else {
                    0
                };
                crate::observe::off(format!(
                    "map_family op={name} opcode={:#x} ch={channel_id} plen={plen} w0={w0:#x} w1={w1:#x}",
                    packet.opcode
                ));
            }
        }
        _ => {
            state.record_fail(FailEvent::UnknownChildOpcode {
                channel: channel_id,
                opcode: packet.opcode,
                total_size: packet.total_size,
            });
        }
    }
    ChildPacketDisposition::Complete
}

/// Drain one child channel.
pub fn drain_child_fifo<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    channel_id: u32,
) {
    if state.gfx.root_page == 0 || channel_id == 0 || channel_id as usize >= MAX_CHANNELS {
        return;
    }
    let Some(regs_off) = child_reg_block_offset(channel_id) else {
        return;
    };
    let regs_gpa = state.pfn_gpa(state.gfx.root_page) + regs_off;

    let mut head = match crate::runtime::host::read_u32(host, regs_gpa + CHILD_REG_HEAD) {
        Ok(v) => v,
        Err(_) => {
            state.record_fail(FailEvent::MalformedChildPacket {
                channel: channel_id,
                fault: PacketFault::ChildRegsHeadRead,
                head: 0,
            });
            return;
        }
    };
    let stamp_index = match crate::runtime::host::read_u32(host, regs_gpa + CHILD_REG_STAMP_INDEX) {
        Ok(v) => v,
        Err(_) => {
            state.record_fail(FailEvent::MalformedChildPacket {
                channel: channel_id,
                fault: PacketFault::ChildRegsStampRead,
                head,
            });
            return;
        }
    };
    let base_pfn = match crate::runtime::host::read_u32(host, regs_gpa + CHILD_REG_BASE_PFN) {
        Ok(v) => v,
        Err(_) => {
            state.record_fail(FailEvent::MalformedChildPacket {
                channel: channel_id,
                fault: PacketFault::ChildRegsBaseRead,
                head,
            });
            return;
        }
    };

    let ring_length = ensure_child_ring(state, host, channel_id, base_pfn);
    if ring_length == 0 {
        return;
    }
    let page_gpas = state.child_rings[channel_id as usize].page_gpas.clone();

    // Nested drain_other must skip this channel (no re-enter head).
    // Use a bit mask so nested drains skip the full stack, not only the leaf.
    let prev_channel = state.draining_channel;
    let bit = 1u32 << channel_id;
    state.draining_channel = channel_id;
    state.draining_mask |= bit;

    loop {
        let tail = match crate::runtime::host::read_u32(host, regs_gpa + CHILD_REG_TAIL) {
            Ok(v) => v,
            Err(_) => {
                state.record_fail(FailEvent::MalformedChildPacket {
                    channel: channel_id,
                    fault: PacketFault::ChildTailRead,
                    head,
                });
                break;
            }
        };
        if head == tail {
            break;
        }
        let Some(available) = published_byte_count(head, tail, ring_length) else {
            state.record_fail(FailEvent::MalformedChildPacket {
                channel: channel_id,
                fault: PacketFault::DesyncedHeadTail,
                head,
            });
            break;
        };
        if available < PACKET_HEADER_LEN {
            break;
        }
        let header = match read_child_ring_bytes(
            host,
            &page_gpas,
            ring_length,
            head,
            PACKET_HEADER_LEN,
            state.page_shift,
        ) {
            Ok(h) => h,
            Err(_) => {
                state.record_fail(FailEvent::MalformedChildPacket {
                    channel: channel_id,
                    fault: PacketFault::ChildHeaderRead,
                    head,
                });
                break;
            }
        };
        let total_size = ld32(&header[PACKET_TOTAL_SIZE..]);
        let snap_len = if total_size >= PACKET_HEADER_LEN
            && total_size <= ring_length
            && available >= total_size
        {
            total_size
        } else {
            PACKET_HEADER_LEN
        };
        let snap = match read_child_ring_bytes(
            host,
            &page_gpas,
            ring_length,
            head,
            snap_len,
            state.page_shift,
        ) {
            Ok(s) => s,
            Err(_) => {
                state.record_fail(FailEvent::MalformedChildPacket {
                    channel: channel_id,
                    fault: PacketFault::ChildSnapRead,
                    head,
                });
                break;
            }
        };
        // Entry gate before decode of full payload: hold CmdDisplaySwap when
        // host paint is already two presents behind (apple-gfx pending_frames
        // >= 2). Leave head unmoved so body draws on other channels can still
        // land via drain_other; re-enter after note_present_paint_consumed.
        let peek_opcode = ld16(&header[PACKET_OPCODE..]);
        if matches!(
            peek_opcode,
            CHILD_OP_DISPLAY_SWAP | CHILD_OP_PRESENT_X86 | CHILD_OP_PRESENT_GAMMA_X86
        ) && state.present.unpainted_presents >= MAX_UNPAINTED_PRESENTS
        {
            note_present_backpressure_hold(state, channel_id, head, tail);
            // Paint will schedule the next worker slice. Preserve this channel
            // without self-waking the worker ahead of QEMU's action BH.
            state.pending.child_mask |= bit;
            break;
        }
        match decode_packet(&snap, head, available) {
            Ok(packet) => {
                if process_child_packet(state, host, channel_id, &packet)
                    == ChildPacketDisposition::Deferred
                {
                    // Translation owns only immutable AIR bytes. Keep head and
                    // stamp untouched so retry cannot duplicate any packet
                    // side effect; continue with sibling channels in the
                    // outer pending-drain loop.
                    break;
                }
                head = packet.next_head;
                if gpa_map::write_u32(
                    host,
                    regs_gpa + CHILD_REG_HEAD,
                    head,
                    state.page_size() as usize,
                )
                .is_err()
                {
                    // The packet was processed + stamped, but the consumer
                    // pointer never advanced: the next drain re-reads the stale
                    // head and RE-EXECUTES the same packets. Fail-visible so
                    // that silent replay is diagnosable (drain.rs Rank-1 audit).
                    state.record_fail(FailEvent::MalformedChildPacket {
                        channel: channel_id,
                        fault: PacketFault::ChildHeadWriteback,
                        head,
                    });
                }

                // Ordered stamp: sync packets ready immediately (DisplaySwap
                // included — PGDisplay present completion after +0x188 retain,
                // not after host encode/paint). target_mapping=0: no async
                // surface hazard for wait_surface.
                let slot = StampSlot {
                    stamp_index,
                    stamp_value: packet.completion_stamp,
                    ready: true,
                    job_id: None,
                    target_mapping: 0,
                };
                state.child_stamps[channel_id as usize].push(slot);
                let ready = state.child_stamps[channel_id as usize].drain_ready();
                for s in ready {
                    write_stamp(state, host, s.stamp_index, s.stamp_value);
                }
                if state.pending.host_action_yield {
                    if head != tail {
                        state.pending.child_mask |= bit;
                    }
                    break;
                }
            }
            Err(PacketError::Incomplete) | Err(PacketError::ShortHeader) => break,
            Err(PacketError::BadSize) => {
                state.record_fail(FailEvent::MalformedChildPacket {
                    channel: channel_id,
                    fault: PacketFault::BadSize,
                    head,
                });
                break;
            }
            Err(PacketError::Desynced) => {
                state.record_fail(FailEvent::MalformedChildPacket {
                    channel: channel_id,
                    fault: PacketFault::Desynced,
                    head,
                });
                break;
            }
        }
    }

    state.draining_mask &= !bit;
    state.draining_channel = prev_channel;
}

/// Drain iosfc mapper producer→consumer handshake.
///
/// Prefer calling this on the **iosfc producer MMIO path** (publishing vCPU)
/// so `resolve_mapping_backing` KVA walks use `current_cpu`. BH-only resolve
/// with `cpu_memory_rw_debug(first_cpu)` deadlocks against MMIO holding
/// `DEVICES` (see reims-vgpu-mmio.c `read_kva`).
pub fn drain_iosfc<H: HostMemory + HostOps>(state: &mut DeviceState, host: &mut H) {
    let producer = state.iosfc.producer;
    let mut consumer = state.iosfc.consumer;
    if producer == consumer {
        state.pending.iosfc = false;
        return;
    }

    // Process requests between consumer and producer when ring is programmed.
    if state.iosfc.ring_base != 0 && producer > consumer {
        let start = consumer;
        let end = producer;
        for idx in start..end {
            let entry_off = (idx as u64) * MAPPER_REQUEST_ENTRY_LEN as u64;
            let mut e = [0u8; MAPPER_REQUEST_ENTRY_LEN];
            if host
                .read_gpa(state.iosfc.ring_base + entry_off, &mut e)
                .is_err()
            {
                break;
            }
            let rtype = ld32(&e[0..]);
            let mapping_id = ld32(&e[4..]);
            // Capture was taken at producer write for published entry (idx+1).
            let cap = match state.mapper_capture {
                Some(c) if c.producer == idx + 1 => state.mapper_capture.take(),
                _ => None,
            };
            match rtype {
                MAPPER_REQUEST_MAP => {
                    let _ = state.map_surface(mapping_id);
                    if let Some(c) = cap {
                        if c.request_type == MAPPER_REQUEST_MAP {
                            let _ = crate::runtime::mapper::apply_capture(state, &c, mapping_id);
                            // Eager page-table + device-desc geometry when KVA works.
                            let _ = crate::runtime::mapper::resolve_mapping_backing(
                                state, host, mapping_id,
                            );
                        } else {
                            // Mismatched capture — put back for a later entry.
                            state.mapper_capture = Some(c);
                        }
                    }
                }
                MAPPER_REQUEST_UNMAP => {
                    // Deferred-writeback: DROP, never write — same recycled-page
                    // hazard as DeleteIOSurfaceBacking2 (the unmap request
                    // trails the guest release; writing risks PTE corruption).
                    crate::runtime::storage_flush::drop_windows(state, mapping_id, "mapper_unmap");
                    if let Some(c) = cap {
                        if c.request_type == MAPPER_REQUEST_UNMAP {
                            let _ = crate::runtime::mapper::apply_capture(state, &c, mapping_id);
                        } else {
                            state.mapper_capture = Some(c);
                            let _ = state.unmap_surface(mapping_id);
                        }
                    } else {
                        let _ = state.unmap_surface(mapping_id);
                    }
                }
                _ => {
                    if let Some(c) = cap {
                        state.mapper_capture = Some(c);
                    }
                    // Unknown mapper request: fail-visible, still advance.
                    state.record_fail(FailEvent::UnknownChildOpcode {
                        channel: 0,
                        opcode: rtype as u16,
                        total_size: MAPPER_REQUEST_ENTRY_LEN as u32,
                    });
                }
            }
            consumer = idx.wrapping_add(1);
        }
    } else {
        // No ring base: still catch consumer up (boot handshake).
        consumer = producer;
    }

    state.iosfc.consumer = consumer;
    if state.iosfc.consumer == state.iosfc.producer {
        host.enqueue(HostAction::irq_iosfc());
    }
    state.pending.iosfc = false;
}

/// Display-side present completion (PGDisplay `_presentMappedSurface`
/// completion block, live PVG binary RE in): after
/// `presentFrame` retains the surface into `+0x188`, the block sets pending
/// bit 1 on the display shared page, reads the enable mask, and pokes the
/// display IRQ when the guest asked for present notifications. This is the
/// guest's frame-done pacing edge — separate from the packet header stamp
/// (the swap fence). Without it the guest keeps swapping (fence releases)
/// but never receives the per-present display event.
pub fn signal_display_present_complete<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
) {
    let gpa = state.display.shared_gpa;
    if gpa == 0 {
        return;
    }
    let mut mask_le = [0u8; 4];
    if host
        .read_gpa(gpa + DISPLAY_SHARED_ENABLE_MASK, &mut mask_le)
        .is_err()
    {
        return;
    }
    let mask = ld32(&mask_le);
    // Pending word is atomic read-and-clear (ldclral) on the guest side; OR
    // the present bit so a not-yet-consumed ONLINE event is preserved.
    let mut pending_le = [0u8; 4];
    let pending = if host
        .read_gpa(gpa + DISPLAY_SHARED_PENDING, &mut pending_le)
        .is_ok()
    {
        ld32(&pending_le)
    } else {
        0
    };
    // A bit2 (ONLINE) still pending *after* online was acked is stale: the guest
    // already consumed that online event (`online_acked`), so re-delivering it
    // makes `signalDisplay` re-run process_online → connectionChange → a
    // boot-progress overlay rebuild (x86 RE 2026-07-17: the host-driven strobe).
    // Preserving bit2 via the `pending |` write is only correct *pre-ack*; drop
    // it once acked so we don't hand the guest a redundant online. `stale` is 0
    // on healthy boots (bit2 clears at ack), so this is a no-op there — it only
    // suppresses the intermittent try_display_online/ack race leftover. A fresh
    // legitimate online (after a reinit) clears `online_acked` first, so it is
    // never masked here. Still logged (measure + fix together).
    let stale = state.display.online_acked && pending & DISPLAY_ONLINE_EVENT_MASK != 0;
    let base = if stale {
        pending & !DISPLAY_ONLINE_EVENT_MASK
    } else {
        pending
    };
    shared_w32(
        host,
        gpa,
        DISPLAY_SHARED_PENDING,
        base | DISPLAY_PRESENT_EVENT_MASK,
        state.page_size() as usize,
    );
    if stale {
        crate::runtime::census::present_proxy::note_stale_online_pending("present", pending);
    }
    if mask & DISPLAY_PRESENT_EVENT_MASK != 0 {
        let bit = 1u32 << (state.display.display_index & 0x1f);
        state
            .gfx
            .interrupt_status_disp
            .fetch_or(bit, std::sync::atomic::Ordering::AcqRel);
        host.enqueue(HostAction::irq_gfx());
    }
}

/// Minimum wall-clock interval shared by both display VBL signal paths.
///
/// The x86 QEMU heartbeat oversamples this interval every `REIMS_VGPU_PCI_HEARTBEAT_MS`
/// (4 ms). The shared limiter caps heartbeat and active-console polls at
/// approximately 120 Hz (8 ms) to match the advertised `DISPLAY_REFRESH_HZ`,
/// without aliasing a heartbeat-only workload down to half rate.
pub(crate) const DISPLAY_VBL_MIN_INTERVAL_MS: u64 = 8;

/// Atomically claim the next display VBL for either the locked or lock-free
/// poll path. A single shared timestamp makes the cadence independent of device
/// lock contention and prevents both paths from signaling the same interval.
///
/// The claimed timestamp advances on a **fixed interval grid** (`last +
/// INTERVAL`), not to `now_ms`. Resetting to `now` lets poll jitter shift the
/// cadence phase permanently: a poll that lands slightly late pushes the *next*
/// deadline out another full interval, so the delivered VBL rate aliases down —
/// when the effective poll spacing sits in the danger zone (just under the
/// interval) it needs two polls per delivery and halves toward ~60 Hz. That is
/// the boot-to-boot 60-vs-120 split the user reports: on a boot where the poll
/// heartbeat jitters into that zone the guest latches 60 Hz. Advancing by exactly
/// one interval keeps delivery phase-locked to the grid and lets a late poll
/// "catch up" (each subsequent poll delivers until the grid is caught, then a
/// poll naturally skips and resyncs) so the *steady* rate converges to the grid
/// (~120 Hz) regardless of poll jitter, erring toward the ceiling the guest caps
/// at rather than latching 60. A long stall (≥2 intervals, e.g. the drain worker
/// held the lock) resyncs the phase to `now_ms` so we never unleash a burst of
/// back-dated VBLs.
pub(crate) fn claim_display_vbl(last_ms: &std::sync::atomic::AtomicU64, now_ms: u64) -> bool {
    let last = last_ms.load(std::sync::atomic::Ordering::Acquire);
    let gap = now_ms.saturating_sub(last);
    if gap < DISPLAY_VBL_MIN_INTERVAL_MS {
        return false;
    }
    let next = if gap >= 2 * DISPLAY_VBL_MIN_INTERVAL_MS {
        now_ms
    } else {
        last + DISPLAY_VBL_MIN_INTERVAL_MS
    };
    last_ms
        .compare_exchange(
            last,
            next,
            std::sync::atomic::Ordering::AcqRel,
            std::sync::atomic::Ordering::Acquire,
        )
        .is_ok()
}

/// Pulse VBL at the phase-locked ~120 Hz cadence (grid interval
/// `DISPLAY_VBL_MIN_INTERVAL_MS`; see `claim_display_vbl`).
///
/// Writes pending bit 0, sets 0x1014 display bit, and raises MSI after ONLINE
/// has been acked. The limiter is owned outside `DeviceState` so this locked
/// path and `vbl_contended_pulse` use one time base. Without VBL the guest
/// compositor can stick on clear-only DisplaySwap of empty flip buffers.
pub fn signal_display_vbl<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    last_ms: &std::sync::atomic::AtomicU64,
) {
    signal_display_vbl_at(state, host, last_ms, crate::observe::elapsed_ms() as u64);
}

/// Delivered-VBL rate, reported from the branch that decides it.
///
/// VBL is what paces the guest's compositor: WindowServer produces a frame off
/// its display-link callback, so whatever rate we deliver here is a ceiling on
/// guest frame rate no matter how fast the present path runs. Nothing measured
/// it. A driven boot emitted **zero** lines matching `vbl` anywhere in the
/// always-on channel, so "are we starving the display link" could not be
/// answered from a log, only guessed at from the constants.
///
/// The three arms are counted separately because a single "delivered" tally
/// cannot tell the two silences apart, and they have opposite meanings:
/// `not_online` is the display never having come up (no VBL is owed at all),
/// while `not_claimed` is the 8 ms limiter doing its job at a healthy 125 Hz.
/// Reading a low delivered count without them would license both conclusions.
///
/// One line per 1024 deliveries — about 8 s at the grid rate, and it costs three
/// relaxed increments per poll otherwise.
/// Which way the VBL path went. Indices into [`VblCensus`].
pub(crate) const VBL_NOT_ONLINE: usize = 0;
pub(crate) const VBL_NOT_CLAIMED: usize = 1;
pub(crate) const VBL_DELIVERED: usize = 2;

/// One report per this many deliveries — about 8 s at the grid rate.
const VBL_REPORT_EVERY: u64 = 1024;

#[derive(Default)]
pub(crate) struct VblCensus {
    arms: [std::sync::atomic::AtomicU64; 3],
    last_report_ms: std::sync::atomic::AtomicU64,
    last_report_n: std::sync::atomic::AtomicU64,
}

impl VblCensus {
    /// Count one traversal and return the line to emit when a report is due.
    ///
    /// Returns the line rather than emitting it so the reporting rule is
    /// testable without a log sink: the interesting properties are "only
    /// deliveries report", "the rate is measured over the window and not the
    /// process lifetime", and "the two silent arms stay separable", and all
    /// three are assertions about this return value.
    pub(crate) fn note(&self, arm: usize, now_ms: u64) -> Option<String> {
        use std::sync::atomic::Ordering::Relaxed;
        let n = self.arms[arm].fetch_add(1, Relaxed) + 1;
        if arm != VBL_DELIVERED || !n.is_multiple_of(VBL_REPORT_EVERY) {
            return None;
        }
        let since_ms = now_ms.saturating_sub(self.last_report_ms.swap(now_ms, Relaxed));
        let since_n = n.saturating_sub(self.last_report_n.swap(n, Relaxed));
        // Window rate, not a lifetime average: the lifetime figure carries the
        // pre-online stretch forever and would read low long after the display
        // came up.
        let hz = if since_ms > 0 {
            (since_n * 1000) as f64 / since_ms as f64
        } else {
            0.0
        };
        Some(format!(
            "display_vbl delivered={n} not_claimed={} not_online={} window_hz={hz:.1} \
             grid_hz={:.1}",
            self.arms[VBL_NOT_CLAIMED].load(Relaxed),
            self.arms[VBL_NOT_ONLINE].load(Relaxed),
            1000.0 / DISPLAY_VBL_MIN_INTERVAL_MS as f64,
        ))
    }
}

pub(crate) fn note_vbl(arm: usize, now_ms: u64) {
    static VBL: std::sync::LazyLock<VblCensus> = std::sync::LazyLock::new(VblCensus::default);
    if let Some(line) = VBL.note(arm, now_ms) {
        crate::observe::off(line);
    }
}

/// Report at most this often. One line per second is bounded enough to leave on
/// for the life of the device and dense enough to see a stall move.
const DRAIN_DUTY_REPORT_MS: u64 = 1000;

/// Where the drain worker's wall clock goes.
///
/// The worker is the device's only executor: `device_drain` holds the device
/// lock for a whole tranche, so every guest FIFO packet, every GPU encode and
/// the host-window export are serialised behind it, and the guest's composite
/// rate cannot exceed the rate at which this thread finishes tranches.
///
/// Nothing else measures that. `sync_exec_lock_hold` is a per-packet threshold
/// line that only fires above `SYNC_EXEC_STALL_US`, so a worker pinned at 100%
/// by a steady stream of 200 ms tranches is completely silent — which is the
/// "an event count is not a state" trap, applied to a cost. This reads the
/// state: what fraction of wall clock the worker spends holding the lock, split
/// by the two phases that can own it.
///
/// The split is the point. `drain_us` is guest work (FIFO decode, draws, compute,
/// guest writeback); `publish_us` is our host-window export, which quiesces the
/// whole GPU twice per present. A duty near 1 says the ~2 Hz composite rate is
/// ours and names which half to attack; a duty near 0 says the worker is idle
/// and the guest is blocked on something upstream of us. No other line separates
/// those two readings.
///
/// `skipped` counts tranches that returned before taking the lock at all
/// (`present_action_pending`): a worker that keeps bailing looks identical to an
/// idle one in the duty figure alone, and it is not the same fault.
/// Which phase of guest work a slice of `drain_us` belongs to.
///
/// These are attributions inside `drain_us`, not a partition of it: a flush
/// reached from inside a draw is counted by both. That is deliberate and it is
/// self-checking — if the three sum to more than `drain_us` the phases nest, and
/// if they sum to much less the time is somewhere none of them names. Either
/// reading is useful and a single fused figure gives neither.
#[derive(Clone, Copy)]
pub enum DrainPhase {
    /// `encode_draw_chain`: metal2vulkan translate, encode, submit, readback.
    Draw,
    /// One compute record applied: bind bookkeeping for most kinds, encode +
    /// execute for a dispatch. Timed as a whole because "the binds are the cost"
    /// is exactly as interesting an answer as "the dispatch is".
    Compute,
    /// Deferred window flush: resident readback + guest writeback.
    Flush,
}

#[derive(Default)]
pub(crate) struct DrainDutyCensus {
    tranches: std::sync::atomic::AtomicU64,
    skipped: std::sync::atomic::AtomicU64,
    drain_us: std::sync::atomic::AtomicU64,
    publish_us: std::sync::atomic::AtomicU64,
    draw_us: std::sync::atomic::AtomicU64,
    draws: std::sync::atomic::AtomicU64,
    compute_us: std::sync::atomic::AtomicU64,
    computes: std::sync::atomic::AtomicU64,
    flush_us: std::sync::atomic::AtomicU64,
    flushes: std::sync::atomic::AtomicU64,
    max_tranche_us: std::sync::atomic::AtomicU64,
    last_report_ms: std::sync::atomic::AtomicU64,
}

impl DrainDutyCensus {
    /// Count one skipped tranche (lock never taken).
    pub(crate) fn note_skipped(&self) {
        self.skipped
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Attribute `us` of the current tranche's `drain_us` to one phase.
    pub(crate) fn note_phase(&self, phase: DrainPhase, us: u64) {
        use std::sync::atomic::Ordering::Relaxed;
        let (total, count) = match phase {
            DrainPhase::Draw => (&self.draw_us, &self.draws),
            DrainPhase::Compute => (&self.compute_us, &self.computes),
            DrainPhase::Flush => (&self.flush_us, &self.flushes),
        };
        total.fetch_add(us, Relaxed);
        count.fetch_add(1, Relaxed);
    }

    /// Accumulate one completed tranche and return the line when a report is
    /// due. Returns the line rather than emitting it so the reporting rule is
    /// testable without a log sink: that the window resets on report (so the
    /// figure is a rate over the window, not a lifetime average), and that duty
    /// is busy time over elapsed time.
    pub(crate) fn note(&self, drain_us: u64, publish_us: u64, now_ms: u64) -> Option<String> {
        use std::sync::atomic::Ordering::Relaxed;
        self.tranches.fetch_add(1, Relaxed);
        self.drain_us.fetch_add(drain_us, Relaxed);
        self.publish_us.fetch_add(publish_us, Relaxed);
        self.max_tranche_us
            .fetch_max(drain_us.saturating_add(publish_us), Relaxed);
        let last = self.last_report_ms.load(Relaxed);
        // First call arms the window; it does not report a duty against a zero
        // origin, which would divide the whole boot's idle time into one tranche.
        if last == 0 {
            self.last_report_ms.store(now_ms, Relaxed);
            return None;
        }
        let win_ms = now_ms.saturating_sub(last);
        if win_ms < DRAIN_DUTY_REPORT_MS {
            return None;
        }
        self.last_report_ms.store(now_ms, Relaxed);
        let tranches = self.tranches.swap(0, Relaxed);
        let skipped = self.skipped.swap(0, Relaxed);
        let drain = self.drain_us.swap(0, Relaxed);
        let publish = self.publish_us.swap(0, Relaxed);
        let max = self.max_tranche_us.swap(0, Relaxed);
        let draw = self.draw_us.swap(0, Relaxed);
        let draws = self.draws.swap(0, Relaxed);
        let compute = self.compute_us.swap(0, Relaxed);
        let computes = self.computes.swap(0, Relaxed);
        let flush = self.flush_us.swap(0, Relaxed);
        let flushes = self.flushes.swap(0, Relaxed);
        let busy = drain.saturating_add(publish);
        let duty = busy as f64 / (win_ms as f64 * 1000.0);
        Some(format!(
            "drain_duty win_ms={win_ms} tranches={tranches} skipped={skipped} busy_us={busy} \
             duty={duty:.3} drain_us={drain} publish_us={publish} max_tranche_us={max} \
             draw_us={draw} draws={draws} compute_us={compute} computes={computes} \
             flush_us={flush} flushes={flushes}"
        ))
    }
}

static DRAIN_DUTY: std::sync::LazyLock<DrainDutyCensus> =
    std::sync::LazyLock::new(DrainDutyCensus::default);

/// Accumulate one completed drain tranche; emits at most once per second.
pub fn note_drain_tranche(drain_us: u64, publish_us: u64) {
    if let Some(line) = DRAIN_DUTY.note(drain_us, publish_us, crate::observe::elapsed_ms() as u64) {
        crate::observe::off(line);
        if let Some(routes) = take_store_routes() {
            crate::observe::off(routes);
        }
        // Same cadence, same reason: `EXEC_INDIRECT2` is the hottest opcode in
        // the device, so its resource table is reported as one window line
        // rather than one line per submission.
        if let Some(line) = crate::runtime::census::exec_resource_table::take_window() {
            crate::observe::off(line);
        }
        // Onto the census cadence rather than a timer of its own, so a reader
        // pairing the footprint against `store_routes` is reading one clock.
        // The run dump rate-limits itself; this is the only caller.
        for line in crate::observe::footprint::census_lines(crate::observe::elapsed_ms() as u64) {
            crate::observe::off(line);
        }
        emit_engine_delta();
    }
}

/// The engine's own counters, over the window `drain_duty` just reported.
///
/// `drain_duty` established that 96-99% of the saturated drain second is
/// `draw_us`, at 1.5-7 ms per draw — orders of magnitude more than a draw's CPU
/// encode should cost. Which of the engine's per-draw costs that is was already
/// being counted and never reported: `engine::counter_snapshot` had no product
/// caller, so every one of these numbers existed and no boot had read one.
///
/// So this adds no instrumentation, only a window delta of what the engine
/// already tallies, chosen to separate the candidates that could each explain
/// milliseconds per draw:
///
/// - `batch_*` — whether draws coalesce into one submission or each takes its
///   own. Per-draw submission is a full CPU-GPU round trip.
/// - `readbacks` / `readback_bytes` — whether every draw drags its target back
///   to host memory, which is a fence wait plus a copy.
/// - `render_post_wait_skips` / `target_reads` — the two halves of the deferred
///   composite Store. The first counts draws that returned without a fence wait
///   because they kept their pixels on the GPU; the second counts the reads a
///   consumer later asked for. A rail that only *moves* the copy raises the
///   second by as much as it raises the first, and `readbacks` alone — which
///   pooled both until it was split — reported no change at all in that case.
/// - `creates` / `*_misses` — pipeline, shader and descriptor churn, where a
///   miss is a driver compile rather than a lookup.
/// - `sampled_reuploads` — re-staging texture content a cache hit should have
///   kept.
/// - `ring_retire_blocks` / `target_evicts` — the engine waiting on itself.
///
/// One line per second, one atomic load per field. Emitted from the same window
/// as `drain_duty` so the two divide against each other; a delta on its own
/// clock would not.
/// Would a resident carry the present this mapping names, at this geometry?
///
/// `Some(true)` a presentable resident exists, `Some(false)` none does — so a
/// present with no guest-page frame behind it shows black — and `None` on a
/// backend with no target registry to ask, where the honest answer is that this
/// build cannot tell.
///
/// It asks through [`crate::backend::vulkan::engine::resident_presentable`],
/// which shares `pools::slot_presentable` with the window presenter's own
/// selection. Sharing the rule is the point rather than tidiness: a looser
/// predicate here would report a frame as carried that the presenter then
/// refuses, which is a disagreement neither call site can see on its own — the
/// same shape as the publish/present split that once blanked the window.
#[cfg(feature = "backend-vulkan")]
fn present_resident_carries(
    state: &crate::model::DeviceState,
    mapping: u32,
    width: u32,
    height: u32,
) -> Option<bool> {
    let identity =
        crate::runtime::present_identity::surface_identity(state, mapping, width, height);
    Some(crate::backend::vulkan::engine::resident_presentable(
        &identity, width, height,
    ))
}

#[cfg(not(feature = "backend-vulkan"))]
fn present_resident_carries(
    _state: &crate::model::DeviceState,
    _mapping: u32,
    _width: u32,
    _height: u32,
) -> Option<bool> {
    None
}

/// Which channel an unbacked present belongs on: `true` is the failure channel.
///
/// A separate function because the `None` arm is the whole content of the rule
/// and it is one character away from being wrong. `carried != Some(true)` and
/// `carried == Some(false)` differ only when the build cannot answer, and that is
/// exactly the case where a possible black frame would be downgraded to a census
/// with nothing to notice it. Fail-closed: only a resident that positively
/// carries the frame demotes the line.
fn unbacked_present_is_a_loss(carried: Option<bool>) -> bool {
    carried != Some(true)
}

/// The `carried=` field: what answered for this present, or that nothing could.
fn carrier_word(carried: Option<bool>) -> &'static str {
    match carried {
        Some(true) => "resident",
        Some(false) => "nothing",
        None => "unknown",
    }
}

#[cfg(feature = "backend-vulkan")]
fn emit_engine_delta() {
    use crate::backend::vulkan::engine::CounterSnapshot;
    static PREV: std::sync::Mutex<Option<CounterSnapshot>> = std::sync::Mutex::new(None);
    let now = crate::backend::vulkan::engine::counter_snapshot();
    let Ok(mut prev) = PREV.lock() else {
        return;
    };
    let d = now.delta_since(&prev.unwrap_or_default());
    *prev = Some(now);
    crate::observe::off(format!(
        "engine_delta creates={} allocs={} batch_opens={} batch_joins={} batch_flushes={} \
         batch_flush_draws={} readbacks={} readback_bytes={} render_post_wait_skips={} \
         target_reads={} target_read_bytes={} pipeline_misses={} \
         shader_misses={} pass_misses={} layout_misses={} sampler_misses={} \
         sampled_cache_hits={} sampled_cache_misses={} sampled_reuploads={} \
         sampled_reupload_bytes={} seed_uploads={} seed_upload_bytes={} \
         ring_retire_blocks={} target_evicts={} desc_pool_grow={} gen_mismatch={}",
        d.creates,
        d.allocs,
        d.batch_opens,
        d.batch_joins,
        d.batch_flushes,
        d.batch_flush_draws,
        d.readbacks,
        d.readback_bytes,
        d.render_post_wait_skips,
        d.target_reads,
        d.target_read_bytes,
        d.pipeline_misses,
        d.shader_misses,
        d.pass_misses,
        d.layout_misses,
        d.sampler_misses,
        d.sampled_cache_hits,
        d.sampled_cache_misses,
        d.sampled_reuploads,
        d.sampled_reupload_bytes,
        d.seed_uploads,
        d.seed_upload_bytes,
        d.ring_retire_blocks,
        d.target_evicts,
        d.desc_pool_grow,
        d.gen_mismatch,
    ));
    emit_draw_phase();
}

/// The split of `drain_duty`'s `draw_us`, over the same window.
///
/// `drain_duty` says a saturated second is 93-99% `draw_us` and `engine_delta`
/// says ~450 MB/s crosses the bus each way. Those two are consistent with
/// opposite fixes — moving fewer bytes, or stopping the per-draw GPU round trip
/// — and neither line can tell them apart. This one can: `readback_us` and the
/// staging half of `setup_us` scale with bytes, `wait_us` does not.
///
/// Silent when no draw ran, so an idle desktop costs nothing.
#[cfg(feature = "backend-vulkan")]
fn emit_draw_phase() {
    let Some(w) = crate::backend::vulkan::engine::draw_phase_window() else {
        return;
    };
    crate::observe::off(format!(
        "draw_phase draws={} prep_us={} pipeline_us={} stage_us={} stage_pass_us={} \
         acquire_us={} descriptors_us={} record_us={} submit_us={} wait_us={} \
         readback_us={} max_us={} stalls={}",
        w.draws,
        w.prep_us,
        w.pipeline_us,
        w.stage_us,
        w.stage_pass_us,
        w.acquire_us,
        w.descriptors_us,
        w.record_us,
        w.submit_us,
        w.wait_us,
        w.readback_us,
        w.max_us,
        w.stalls,
    ));
}

#[cfg(not(feature = "backend-vulkan"))]
fn emit_draw_phase() {}

#[cfg(not(feature = "backend-vulkan"))]
fn emit_engine_delta() {}

/// Count a drain wake-up that returned before taking the device lock.
pub fn note_drain_skipped() {
    DRAIN_DUTY.note_skipped();
}

/// Attribute elapsed time since `started` to one phase of the current tranche.
pub fn note_drain_phase(phase: DrainPhase, started: std::time::Instant) {
    DRAIN_DUTY.note_phase(phase, started.elapsed().as_micros() as u64);
}

/// Count one guest-Store routing decision, by route name.
///
/// The routes are the attribution for `engine_delta`'s readback bytes: only
/// `cpu_portability` reads a full frame back and CPU-copies it into the guest's
/// pages, and only it is forced to — `gva_store_defer_eligible` refuses any
/// target with a nonzero `mapping_id`, so a type-11 composite Store has no
/// deferred rail to take. Whether that is 2 Stores a second or 20 decides
/// whether building one is worth it, and the route's own first-appearance line
/// is deduplicated per process and cannot say.
static STORE_ROUTES: std::sync::Mutex<Option<std::collections::BTreeMap<&'static str, u64>>> =
    std::sync::Mutex::new(None);

/// # This census cannot find the Finder icon defect, and that is now measured
///
/// Several sessions have concluded "no counter separates a corrupt icon round
/// from a clean one" by printing six or eight hand-picked columns. That is a
/// statement about the columns someone thought to print, not about the census.
/// It has now been asked of the whole census at once.
///
/// Three 14-round `icon-composite.sh` boots, x86 / Vulkan, pooled: **42 scored
/// rounds, 9 corrupt, 33 clean**. Every counter in this map present in at least
/// 80% of rounds was normalised per 1000 `draw_scissor_full` — round length
/// varies ~40% on this rig and almost every draw-path counter is proportional
/// to it — and ranked by AUC, the probability that a random corrupt round
/// scores above a random clean one. The best column in the entire census:
///
/// ```text
/// AUC 0.75  surface_flush             permutation p = 0.021 raw
/// AUC 0.73  load_seed_ok                            p = 0.914 Bonferroni
/// AUC 0.72  type11_seed_uploaded      (43 columns tested)
/// AUC 0.72  type11_seed_guest_wrote
/// AUC 0.71  t11_gw_ref_moved
/// ```
///
/// Corrected for having looked at 43 columns, nothing is distinguishable from
/// noise. The leaders are also largely one quantity wearing different names — a
/// type-11 seed upload is a `load_seed_ok_mapping` — so they are one weak
/// signal, not five.
///
/// The reason is structural rather than a gap to be filled by adding counters.
/// A round runs ~11 000 draws and the defect is **one** icon: a single
/// operation going wrong is a ~1e-4 perturbation of any population, which no
/// aggregate can resolve. Adding a counter to this map cannot change that, and
/// a session that adds one and reads it per round is repeating a measurement
/// that has now been shown to have no power.
///
/// What would have power is a *screen-to-resource join*: name the 64x64 target
/// backing the cell that is blank in the capture, then dump that one target's
/// history. [`crate::observe::content_summary`] is the existing half of it — a
/// correct icon carries hundreds of distinct texels and a blank one collapses
/// to one — and the missing half is the mapping from a screen rectangle to a
/// target identity.
///
/// Settled by the same three boots, so nobody re-runs it: the Vulkan
/// synchronization repairs are not the producer either. Corruption rates were
/// 3/14 before them, 4/14 after the first, 2/14 after all five. The hazards
/// they closed were real undefined behaviour and those fixes stand on that
/// ground alone — see
/// [`crate::backend::vulkan::engine::exec::resident_read_source_scope`] — but
/// they do not move this class.
///
/// # A scoring flaw that inverts verdicts, recorded here because the harness is not tracked
///
/// The repro scripts live under `.agents/`, which is gitignored, so a fix made
/// there does not survive to the next session and this warning would vanish
/// with it.
///
/// `iconscore.py` scores a capture by counting blue blobs in a horizontal band
/// and comparing the count to `--expect`. Its own description defines the
/// population as "blue blobs of near-identical area", but it only ever policed
/// the *small* side of that (a `shrunk` class). On 2026-07-31 an unrelated blue
/// object of area 3247, against an icon median of 1235, entered the band and
/// was counted toward `expect`. That **inverted the verdict of all fourteen
/// rounds of a probe boot**: a round showing all seven icons counted 8 and read
/// CORRUPT, and a round genuinely missing one counted 7 and read CLEAN.
///
/// It was caught only by re-deriving each round's verdict from the *positions*
/// of the blobs rather than their number. Any conclusion of the form "n corrupt
/// rounds out of m" is worth exactly as much as the assumption that nothing
/// else blue and icon-sized was on screen, and that assumption is not
/// self-checking. A symmetric `outsized` exclusion, reported on the output line
/// rather than applied silently, is the fix; if the harness in front of you
/// does not print `outsized=` when something is excluded, it predates this and
/// its verdicts should be re-derived positionally before they are believed.
pub fn note_store_route(route: &'static str) {
    note_store_route_n(route, 1);
}

/// Add `n` to a named count in the same per-second window as [`note_store_route`].
///
/// For events that arrive in batches — one notify marking many cache entries —
/// where the number that matters is the entries, not the notifies, and taking
/// the lock once per entry would cost more than the census is worth.
pub fn note_store_route_n(route: &'static str, n: u64) {
    if n == 0 {
        return;
    }
    if let Ok(mut g) = STORE_ROUTES.lock() {
        *g.get_or_insert_with(Default::default)
            .entry(route)
            .or_default() += n;
    }
}

/// Accumulate microseconds against a named cost, into the same per-second window
/// as the route counts above.
///
/// The same map on purpose. `store_routes` is already drained once a second
/// beside `drain_duty`, so a cost reported here divides into that window's
/// `draw_us` with no join and no cross-boot comparison. `draw_phase` cannot
/// carry these: it brackets the *engine's* internals, and this is the runtime
/// work on either side of them — which is where **28 % of `draw_us`** was
/// going unattributed (~245 ms per second, stable across 200 windows of the
/// 2026-07-30 boot, larger than `stage_us` and `readback_us` and second only to
/// `wait_us`). A phase table that sums to 72 % of the thing it decomposes
/// cannot be used to choose what to fix.
pub fn note_store_route_us(name: &'static str, us: u64) {
    if let Ok(mut g) = STORE_ROUTES.lock() {
        *g.get_or_insert_with(Default::default)
            .entry(name)
            .or_default() += us;
    }
}

/// Read one route's count out of the live window, for tests that assert a
/// census fired rather than trusting that it was wired up.
///
/// A counter nobody reads back is a counter that can be deleted, mistyped, or
/// placed on the wrong side of an early return without any test noticing — and
/// several of this crate's readings have turned on exactly which side of a
/// branch a `note_store_route` sat on.
#[cfg(test)]
pub(crate) fn store_route_count(route: &str) -> u64 {
    STORE_ROUTES
        .lock()
        .ok()
        .and_then(|g| g.as_ref().and_then(|m| m.get(route).copied()))
        .unwrap_or(0)
}

/// Drain and format the window's route counts, or `None` if none were taken.
fn take_store_routes() -> Option<String> {
    let mut g = STORE_ROUTES.lock().ok()?;
    let routes = g.as_mut()?;
    if routes.is_empty() {
        return None;
    }
    let mut out = String::from("store_routes");
    for (route, n) in routes.iter() {
        out.push_str(&format!(" {route}={n}"));
    }
    routes.clear();
    Some(out)
}

fn signal_display_vbl_at<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    last_ms: &std::sync::atomic::AtomicU64,
    now_ms: u64,
) {
    if state.display.shared_gpa == 0 || !state.display.online_acked {
        note_vbl(VBL_NOT_ONLINE, now_ms);
        return;
    }
    if !claim_display_vbl(last_ms, now_ms) {
        note_vbl(VBL_NOT_CLAIMED, now_ms);
        return;
    }
    note_vbl(VBL_DELIVERED, now_ms);
    let gpa = state.display.shared_gpa;
    let mut pending_le = [0u8; 4];
    let pending = if host
        .read_gpa(gpa + DISPLAY_SHARED_PENDING, &mut pending_le)
        .is_ok()
    {
        ld32(&pending_le)
    } else {
        0
    };
    // Drop a stale (already-acked) ONLINE bit so we don't re-deliver it and make
    // the guest re-run process_online → connectionChange → overlay rebuild (see
    // signal_display_present_complete). signal_display_vbl only runs post-ack, so
    // online_acked is already true here; `stale` is 0 on healthy boots (no-op).
    let stale = state.display.online_acked && pending & DISPLAY_ONLINE_EVENT_MASK != 0;
    let base = if stale {
        pending & !DISPLAY_ONLINE_EVENT_MASK
    } else {
        pending
    };
    shared_w32(
        host,
        gpa,
        DISPLAY_SHARED_PENDING,
        base | DISPLAY_VBL_EVENT_MASK,
        state.page_size() as usize,
    );
    if stale {
        crate::runtime::census::present_proxy::note_stale_online_pending("vbl", pending);
    }
    let bit = 1u32 << (state.display.display_index & 0x1f);
    state
        .gfx
        .interrupt_status_disp
        .fetch_or(bit, std::sync::atomic::Ordering::AcqRel);
    host.enqueue(HostAction::irq_gfx());
}

/// Assert display ONLINE once the guest has published the enable mask.
///
/// Archive `apple_pv_gpu_display_signal_online` + poll_tick gate:
/// write shared `+0x100` pending bit 2, then pulse display IRQ. Only after
/// `enable()` sets `+0x104` bit 2 — earlier IRQs wedge an unregistered display.
/// createDisplayAttributes then consumes TimingElements (incl. 1440 mode).
pub fn try_display_online<H: HostMemory + HostOps>(state: &mut DeviceState, host: &mut H) {
    if state.display.shared_gpa == 0 || state.display.online_acked {
        return;
    }
    if state.display.online_tries >= DISPLAY_ONLINE_MAX_TRIES {
        return;
    }
    // Cadence: skip most ticks (archive divisor); still run often enough via
    // gfx_update / drain that enable() is observed within seconds.
    let ctr = state.display.poll_ctr.wrapping_add(1);
    state.display.poll_ctr = ctr;
    if !ctr.is_multiple_of(DISPLAY_ONLINE_POLL_DIVISOR) {
        return;
    }
    let gpa = state.display.shared_gpa;
    let mut mask_le = [0u8; 4];
    if host
        .read_gpa(gpa + DISPLAY_SHARED_ENABLE_MASK, &mut mask_le)
        .is_err()
    {
        return;
    }
    let mask = ld32(&mask_le);
    if mask & DISPLAY_ONLINE_EVENT_MASK == 0 {
        return;
    }
    // pending word is atomic read-and-clear on the guest side.
    shared_w32(
        host,
        gpa,
        DISPLAY_SHARED_PENDING,
        DISPLAY_ONLINE_EVENT_MASK,
        state.page_size() as usize,
    );
    let bit = 1u32 << (state.display.display_index & 0x1f);
    state
        .gfx
        .interrupt_status_disp
        .fetch_or(bit, std::sync::atomic::Ordering::AcqRel);
    host.enqueue(HostAction::irq_gfx());
    // Always-on on the first ONLINE pulse per shared-state generation (rare, not a
    // flood): the display-lifecycle timeline entry point. A second pass through here
    // after a reinit setup pairs with display_shared_state_setup reinit=1 to show a
    // post-converge display rebuild.
    if state.display.online_tries == 0 {
        crate::observe::fail(format!(
            "display_online_signal index={}",
            state.display.display_index
        ));
    }
    state.display.online_tries = state.display.online_tries.saturating_add(1);
}

/// Drain active/pending child FIFOs other than channels mid-drain.
///
/// Used by DisplaySwap Dekker rescue and stranded paths — **not** by
/// `render_wait_surface` (archive wait is surface-keyed async completion only).
///
/// Mask matches archive `poll_tick`: `active_child_mask | pending_child_mask`
/// so work doorbell'd while a drain was in flight is not skipped. Skips
/// `skip_channel` and every bit in `state.draining_mask` so nested drains
/// cannot re-enter a mid-packet channel (same head re-process).
pub fn drain_other_child_fifos<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    skip_channel: u32,
) {
    let mask = state.active_child_mask | state.pending.child_mask;
    let nested = state.draining_mask;

    // A cold-translation EXEC is already the oldest accepted item in the host
    // scheduler timeline. Retry its channel(s) first. Sibling FIFO packets may
    // tear down tasks, mappings, objects, or surfaces referenced by that EXEC,
    // so none may pass the artificial translation boundary. Translation owns
    // immutable AIR and completes independently of all FIFO drains.
    let deferred = state.translation_deferred_mask;
    if deferred != 0 {
        for ch in 1..MAX_CHANNELS as u32 {
            let bit = 1u32 << ch;
            if deferred & bit == 0 || ch == skip_channel || nested & bit != 0 {
                continue;
            }
            state.pending.child_mask &= !bit;
            drain_child_fifo(state, host, ch);
        }
        if state.translation_deferred_mask != 0 {
            let held = mask & !deferred & !nested & !(1u32 << skip_channel);
            state.pending.child_mask |= held | state.translation_deferred_mask;
            note_translation_order_hold(state, held);
            return;
        }
        release_translation_order_holds(state);
    }

    let mut remaining = mask;
    for ch in 1..MAX_CHANNELS as u32 {
        if state.pending.host_action_yield {
            break;
        }
        if ch == skip_channel {
            continue;
        }
        if nested & (1u32 << ch) != 0 {
            continue;
        }
        if mask & (1u32 << ch) == 0 {
            continue;
        }
        remaining &= !(1u32 << ch);
        // Clear pending bit for channels we actually drain (archive poll_tick
        // consumes pending when it drains). Leave skip/nested bits alone.
        state.pending.child_mask &= !(1u32 << ch);
        drain_child_fifo(state, host, ch);
        if state.translation_deferred_mask != 0 {
            let held = remaining & !nested & !(1u32 << skip_channel);
            state.pending.child_mask |= held | state.translation_deferred_mask;
            note_translation_order_hold(state, held);
            break;
        }
    }
}

/// Archive `apple_pv_gpu_surface_inflight`: true iff a not-ready stamp slot
/// names this type-11 mapping as its async write target.
fn surface_inflight(state: &DeviceState, mapping: u32) -> bool {
    if mapping == 0 {
        return false;
    }
    for ch in 0..MAX_CHANNELS {
        for slot in &state.child_stamps[ch].queue {
            if !slot.ready && slot.target_mapping == mapping {
                return true;
            }
        }
    }
    false
}

/// Archive `apple_pv_gpu_render_wait_surface` for a type-11 mapping.
///
/// Archive (apple-pv-gpu-render.c):
/// ```c
/// if (draw_jobs_inflight == 0 || !surface_inflight(s, is_gva, key)) return;
/// while (draw_jobs_inflight > 0 && surface_inflight(s, is_gva, key))
///     aio_poll(...);
/// ```
/// Waits only for **already-submitted** async jobs that still write this
/// surface. Does **not** drain other child FIFOs, does **not** loop on
/// `content_generation`, does **not** require multiple "quiet rounds".
///
/// Product currently completes draws before the packet stamp (sync-per-packet),
/// so this is a no-op unless an async job was enqueued with `target_mapping`.
/// Returns the mapping's content generation after the wait (0 if unmapped).
pub fn wait_surface_other_channels<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    _skip_channel: u32,
    mapping: u32,
) -> u32 {
    let _ = host;
    if mapping == 0 {
        return 0;
    }
    // Product has no host-side aio_poll worker loop yet. When async jobs with
    // target_mapping are introduced, completions must mark the slot ready
    // (complete_async_job) before or during this wait — same as archive BH.
    // Never invent FIFO drains here as a substitute for surface_inflight.
    debug_assert!(
        !surface_inflight(state, mapping),
        "wait_surface: async job still targets mapping {mapping} (no product aio_poll)"
    );
    mapping_content_gen(state, mapping)
}

/// RAW barrier before snapshotting a type-11 surface (DisplaySwap present,
/// sample Load, color Load seed). Same archive `render_wait_surface` path.
pub fn wait_surface_mapping<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    mapping: u32,
) -> u32 {
    wait_surface_other_channels(state, host, 0, mapping)
}

/// Host paint consumed the current +0x188 retain (Painted or Unchanged).
///
/// Clears the entry-side present backpressure counter so a DisplaySwap held
/// at channel head can run on the next drain (schedule_bh from scanout path).
pub fn note_present_paint_consumed(state: &mut DeviceState) {
    state.present.unpainted_presents = 0;
    state.present.backpressure_hold_active = false;
    state.pending.host_action_yield = false;
}

fn note_present_backpressure_hold(state: &mut DeviceState, channel: u32, head: u32, tail: u32) {
    if state.present.backpressure_hold_active
        && state.present.backpressure_hold_channel == channel
        && state.present.backpressure_hold_head == head
    {
        return;
    }
    state.present.backpressure_hold_active = true;
    state.present.backpressure_hold_channel = channel;
    state.present.backpressure_hold_head = head;
    state.present.backpressure_hold_count = state.present.backpressure_hold_count.saturating_add(1);
    crate::observe::fail(format!(
        "THRASH present_action_starvation reason=pending_frames_cap ch={channel} head={head} tail={tail} unpainted={} episode={}",
        state.present.unpainted_presents, state.present.backpressure_hold_count
    ));
}

#[inline]
fn mapping_content_gen(state: &DeviceState, mapping: u32) -> u32 {
    state
        .mappings
        .get(&mapping)
        .map(|m| m.content_generation)
        .unwrap_or(0)
}

/// Publish the poll-tick/Dekker rescue to the asynchronous drain owner.
///
/// This performs no guest-memory reads and no command execution. QEMU may call
/// it from its display/main-loop context without accidentally translating or
/// submitting GPU work under the BQL. Active child channels are intentionally
/// coalesced into one mask; the worker's normal ring checks make idle channels
/// cheap no-ops.
pub fn publish_stranded_fifos<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
) -> bool {
    let mut published = false;
    if state.gfx.control_fifo != 0 {
        if state
            .gfx
            .fifo_read
            .load(std::sync::atomic::Ordering::Acquire)
            != state.gfx.fifo_written
        {
            state.pending.main_drain = true;
            published = true;
        }
        if state.active_child_mask != 0 {
            state.pending.child_mask |= state.active_child_mask;
            published = true;
        }
    }
    if state.iosfc.consumer != state.iosfc.producer {
        state.pending.iosfc = true;
        published = true;
    }
    if published {
        host.schedule_bh();
    }
    published
}

/// Run all pending drains (BH body).
pub fn drain_pending<H: HostMemory + HostOps>(state: &mut DeviceState, host: &mut H) {
    // A queued present action is part of the ordered device timeline. QEMU
    // cannot paint it while this worker owns the device lock, so later worker
    // wakeups must leave guest work queued until scanout consumes the action.
    if state.pending.host_action_yield {
        return;
    }
    release_translation_order_holds(state);
    // Retry an already translation-held EXEC before allowing either the root
    // FIFO or a sibling child FIFO to overtake it. The guest is free to queue
    // Unmap/Delete immediately after submission; without this boundary the
    // retried EXEC can stage successfully and then write back after its task
    // mapping has been destroyed.
    let deferred = state.translation_deferred_mask;
    if deferred != 0 {
        if state.pending.main_drain {
            note_translation_order_hold(state, TRANSLATION_ROOT_FIFO_BIT);
        }
        let sibling_pending = state.pending.child_mask & !deferred;
        note_translation_order_hold(state, sibling_pending);
        for ch in 1..MAX_CHANNELS as u32 {
            let bit = 1u32 << ch;
            if deferred & bit == 0 {
                continue;
            }
            state.pending.child_mask &= !bit;
            drain_child_fifo(state, host, ch);
        }
        if state.translation_deferred_mask != 0 {
            state.pending.child_mask |= state.translation_deferred_mask;
            return;
        }
        release_translation_order_holds(state);
    }
    if state.pending.main_drain {
        drain_main_fifo(state, host);
    }
    let mask = state.pending.child_mask;
    state.pending.child_mask = 0;
    let mut remaining = mask;
    for ch in 1..MAX_CHANNELS as u32 {
        let bit = 1u32 << ch;
        if mask & bit != 0 {
            remaining &= !bit;
            drain_child_fifo(state, host, ch);
            if state.translation_deferred_mask != 0 {
                state.pending.child_mask |= remaining | state.translation_deferred_mask;
                note_translation_order_hold(state, remaining);
                return;
            }
            if state.pending.host_action_yield {
                state.pending.child_mask |= remaining;
                return;
            }
        }
    }
    if state.pending.iosfc {
        drain_iosfc(state, host);
    }
    try_display_online(state, host);
    // Unmap contiguous views retired by MAP/UNMAP/page-table changes (their
    // Metal objects were dropped at retire time; execution is sync-per-packet
    // so nothing aliases them anymore).
    crate::runtime::mapper::flush_retired_views(state, host);
    // Unpin engine residents of linear cache entries dropped by task/object
    // deletes this drain (they become LRU-evictable instead of leaking).
    crate::runtime::storage_flush::retire_linear_residents(state);
    // Land GVA render-Store windows whose task died this drain (cache-only —
    // the GVA walk went with the task) and unpin their residents.
    crate::runtime::storage_flush::retire_gva_windows(state, host);
}

/// Enqueue an async stamp that holds the channel order until marked ready.
///
/// `target_mapping` is the type-11 surface this job writes (0 = none) — archive
/// `DrawJob.mapping_id` for `surface_inflight` / `render_wait_surface`.
pub fn enqueue_async_stamp_surface(
    state: &mut DeviceState,
    channel_id: u32,
    stamp_index: u32,
    stamp_value: u32,
    target_mapping: u32,
) -> Option<u64> {
    if channel_id as usize >= MAX_CHANNELS {
        return None;
    }
    let job = state.alloc_job_id();
    state.child_stamps[channel_id as usize].push(StampSlot {
        stamp_index,
        stamp_value,
        ready: false,
        job_id: Some(job),
        target_mapping,
    });
    Some(job)
}

/// Complete an async job and fire any leading ready stamps.
pub fn complete_async_job<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    channel_id: u32,
    job_id: u64,
) {
    if channel_id as usize >= MAX_CHANNELS {
        return;
    }
    if !state.child_stamps[channel_id as usize].mark_ready(job_id) {
        return;
    }
    let ready = state.child_stamps[channel_id as usize].drain_ready();
    for s in ready {
        write_stamp(state, host, s.stamp_index, s.stamp_value);
    }
}

#[cfg(test)]
mod tests;
