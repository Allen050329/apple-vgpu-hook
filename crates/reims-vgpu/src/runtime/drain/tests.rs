use super::*;
use crate::model::{PAGE_SHIFT_ARM64E, PAGE_SHIFT_X86, PAGE_SIZE_ARM64E};

/// I2's carve-out, asserted rather than trusted: a partial packet is the
/// normal state of a ring whose producer is mid-write, so it must not reach
/// the always-on log. A bad size or a desync must.
///
/// Without this, the flood is silent — a healthy boot would write one line
/// per drain iteration and the sink's own detector would be the only thing
/// that noticed.
#[test]
fn a_partial_packet_is_control_flow_and_never_a_logged_fault() {
    assert_eq!(PacketError::ShortHeader.fault(), None);
    assert_eq!(PacketError::Incomplete.fault(), None);
    assert_eq!(PacketError::BadSize.fault(), Some(PacketFault::BadSize));
    assert_eq!(PacketError::Desynced.fault(), Some(PacketFault::Desynced));
}

#[test]
fn present_scanout_action_follows_window_active() {
    use crate::runtime::host::{FakeHost, HostActionKind};

    let mut state = DeviceState::new(crate::model::DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    state.present.frame_mapping = 7;
    state.present.frame_generation = 42;

    // No host window (arm64 MMIO / REIMS_VGPU_WINDOW=0): the QEMU console is the
    // display — the present MUST enqueue the CPU ScanoutUpdate and request
    // the action boundary, or the console freezes at its last paint.
    state.present.window_active = false;
    enqueue_present_scanout(&mut state, &mut host, 1440, 1080);
    let scan: Vec<_> = host
        .actions
        .iter()
        .filter(|a| a.kind == HostActionKind::ScanoutUpdate)
        .collect();
    assert_eq!(scan.len(), 1, "windowless present must paint the console");
    assert_eq!(scan[0].a0, 7);
    assert_eq!(scan[0].a1, 1440);
    assert_eq!(scan[0].a2, 1080);
    assert_eq!(scan[0].a3, 42);
    assert_eq!(state.present.unpainted_presents, 1);
    assert!(state.pending.host_action_yield);

    // Live host window: the drain publishes + self-acks; no CPU paint
    // action (QEMU runs -display none, the surface is painted for nobody).
    host.actions.clear();
    state.present.window_active = true;
    enqueue_present_scanout(&mut state, &mut host, 1440, 1080);
    assert!(
        !host
            .actions
            .iter()
            .any(|a| a.kind == HostActionKind::ScanoutUpdate),
        "window path must not produce a QEMU paint action"
    );
}

/// The mapping the host window will show for the present just accepted.
///
/// These tests used to observe the present through the `ScanoutUpdate`
/// action's `a0`. No CPU paint action is produced per present any more, so
/// the observable moves to the retain in `state.present` — which is what the
/// window reads. The gate mirrors the `paint_mid` selection that used to pick
/// the action's mapping, so "the mapping that would have been painted" and
/// "the mapping the window shows" stay the same assertion.
fn presented_mapping(state: &DeviceState) -> Option<u32> {
    let p = &state.present;
    (p.frame_valid && p.frame_mapping != 0 && !p.frame_bgra.is_empty()).then_some(p.frame_mapping)
}

/// These selection tests run the windowless (QEMU-console) configuration,
/// where every accepted present enqueues the CPU `ScanoutUpdate` —
/// coalesced latest-wins, so however many presents a test drives, at most
/// ONE may be pending. Presence/absence per presentation path is locked by
/// `present_scanout_action_follows_window_active`; this tripwire catches a
/// re-introduced per-present backlog (the dual-mid half-frame thrash
/// class).
fn assert_coalesced_paint_action(host: &crate::runtime::host::FakeHost, ctx: &str) {
    assert!(
        host.action_count(HostActionKind::ScanoutUpdate) <= 1,
        "{ctx}: pending CPU ScanoutUpdate paints must coalesce to at most one"
    );
}

#[test]
fn exec_summary_names_each_synchronous_timing_bucket() {
    let result = crate::runtime::exec::ExecResult {
        task_id: 3,
        streams_loaded: 1,
        buffer_unbinds: 2,
        texture_unbinds: 3,
        sampler_unbinds: 4,
        render_attachment_resolves: 1,
        render_guest_stores: 1,
        load_us: 11,
        render_us: 12,
        blit_us: 13,
        compute_us: 14,
        event_us: 15,
        info_us: 16,
        finish_us: 17,
        total_us: 98,
        ..Default::default()
    };
    let line = exec_summary(1, &result, 52);
    for field in [
        "load_us=11",
        "render_us=12",
        "blit_us=13",
        "compute_us=14",
        "rt_resolves=1",
        "guest_stores=1",
        "render_unbinds=2/3/4",
        "event_us=15",
        "info_us=16",
        "finish_us=17",
        "total_us=98",
    ] {
        assert!(line.contains(field), "missing {field}: {line}");
    }
}

#[test]
fn sync_exec_stall_proxy_fires_at_watchdog_scale_only() {
    assert!(!sync_exec_stalled(SYNC_EXEC_STALL_US - 1));
    assert!(sync_exec_stalled(SYNC_EXEC_STALL_US));
    assert!(sync_exec_stalled(3_406_929));
}
use crate::runtime::host::{FakeHost, HostActionKind};

fn packet_bytes(opcode: u16, stamp_value: u32, payload: &[u8]) -> Vec<u8> {
    let total = PACKET_HEADER_LEN as usize + payload.len();
    let mut v = vec![0u8; total];
    v[0..2].copy_from_slice(&opcode.to_le_bytes());
    v[2..4].copy_from_slice(&0u16.to_le_bytes());
    v[4..8].copy_from_slice(&(total as u32).to_le_bytes());
    v[8..12].copy_from_slice(&stamp_value.to_le_bytes());
    v[12..].copy_from_slice(payload);
    v
}

#[test]
fn decode_basic_packet() {
    let p = packet_bytes(ROOT_OP_DEFINE_FIFO, 7, &1u32.to_le_bytes());
    let dec = decode_packet(&p, 0, p.len() as u32).unwrap();
    assert_eq!(dec.opcode, ROOT_OP_DEFINE_FIFO);
    assert_eq!(dec.completion_stamp, 7);
    assert_eq!(dec.next_head, p.len() as u32);
}

#[test]
fn display_descriptor_advertises_four_modes_incl_4k() {
    let mut host = FakeHost::new();
    let gpa = 0x7a000000u64;
    host.map_range(gpa, PAGE_SIZE_ARM64E as usize, 0);
    fill_display_descriptor(&mut host, gpa, 0, PAGE_SIZE_ARM64E);
    let mut count = [0u8; 2];
    host.read_gpa(gpa + DISPLAY_DESC_TIMING_COUNT, &mut count)
        .unwrap();
    assert_eq!(u16::from_le_bytes(count), 4);
    let read16 = |host: &mut FakeHost, off: u64| {
        let mut b = [0u8; 2];
        host.read_gpa(gpa + off, &mut b).unwrap();
        u16::from_le_bytes(b)
    };
    // Element 0 (native/preferred) stays 1920×1080; 4K is appended last so
    // boot resolution is unchanged. Stride 0x10 from base 0x210.
    assert_eq!(read16(&mut host, 0x210), 1920);
    assert_eq!(read16(&mut host, 0x212), 1080);
    assert_eq!(read16(&mut host, 0x220), 1440);
    assert_eq!(read16(&mut host, 0x230), 1280);
    assert_eq!(read16(&mut host, 0x240), 3840);
    assert_eq!(read16(&mut host, 0x242), 2160);
    // Every element carries the same 120 Hz refresh (16.16 fixed-point).
    let mut refresh = [0u8; 4];
    host.read_gpa(gpa + 0x244, &mut refresh).unwrap();
    assert_eq!(u32::from_le_bytes(refresh), DISPLAY_REFRESH_HZ << 16);
}

#[test]
fn present_page_identity_reports_alias_and_disjoint_peers() {
    use crate::contract::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let entry = |pfn: u32| (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
    // Present-named surface (surface-id namespace): pages A.
    assert!(state.map_surface(4));
    {
        let m = state.mappings.get_mut(&4).unwrap();
        m.mapped = true;
        m.has_geom = true;
        m.width = 4;
        m.height = 4;
        m.page_entries = vec![entry(0x100), entry(0x101)];
    }
    // Composite peer aliasing the SAME pages (mapping namespace).
    assert!(state.map_surface(1));
    {
        let m = state.mappings.get_mut(&1).unwrap();
        m.mapped = true;
        m.has_geom = true;
        m.width = 4;
        m.height = 4;
        m.page_entries = vec![entry(0x100), entry(0x101)];
    }
    state
        .surface_write_kind
        .insert(1, crate::model::SurfaceWriteKind::Composite);
    // Same-geometry peer with disjoint pages.
    assert!(state.map_surface(2));
    {
        let m = state.mappings.get_mut(&2).unwrap();
        m.mapped = true;
        m.has_geom = true;
        m.width = 4;
        m.height = 4;
        m.page_entries = vec![entry(0x200), entry(0x201)];
    }
    // Different geometry: excluded entirely.
    assert!(state.map_surface(9));
    {
        let m = state.mappings.get_mut(&9).unwrap();
        m.mapped = true;
        m.has_geom = true;
        m.width = 8;
        m.height = 8;
        m.page_entries = vec![entry(0x100)];
    }
    let line = present_page_identity_line(&state, 4, 4, 4).expect("line");
    assert!(line.contains("present_page_identity mid=4 4x4 pages=2 valid=2"));
    assert!(
        line.contains("mid1:pages=2:overlap=2:ident=1:kind=Composite"),
        "alias peer must report identical pages: {line}"
    );
    assert!(
        line.contains("mid2:pages=2:overlap=0:ident=0"),
        "disjoint peer must report zero overlap: {line}"
    );
    assert!(
        !line.contains("mid9"),
        "geometry-mismatched mapping excluded: {line}"
    );
}

#[test]
fn display_swap_paints_mapping_geom_not_console_fallback() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    // Established boot console 1920×1080.
    state.present.valid = true;
    state.present.width = 1920;
    state.present.height = 1080;
    assert!(state.map_surface(3));
    {
        let m = state.mappings.get_mut(&3).unwrap();
        m.mapped = true;
        m.has_geom = true;
        m.width = 1440;
        m.height = 1080;
        m.content_generation = 5;
        m.page_entries = vec![1];
    }
    let mut payload = vec![0u8; DISPLAY_SWAP_MIN_LEN];
    payload[DISPLAY_SWAP_MAPPING..DISPLAY_SWAP_MAPPING + 4].copy_from_slice(&3u32.to_le_bytes());
    let pkt = Packet {
        opcode: CHILD_OP_DISPLAY_SWAP,
        stamp_count: 0,
        total_size: PACKET_HEADER_LEN + DISPLAY_SWAP_MIN_LEN as u32,
        completion_stamp: 0,
        payload,
        next_head: 0,
    };
    process_child_packet(&mut state, &mut host, 4, &pkt);
    assert!(state.present.frame_flush_seen);
    assert_eq!(state.present.width, 1440);
    assert_eq!(state.present.height, 1080);
    // Geometry is asserted above off `state.present`. The mapping identity
    // moves from the (now absent) action's a0 to `present_mapping` — the
    // accepted present. NOT the retain: the capture fails here (no guest
    // pages), which the old action tolerated via its `paint_mid` fallback.
    assert_eq!(state.present.present_mapping, 3);
    assert_coalesced_paint_action(&host, "mapping geom, not console fallback");
}

/// After bootstrap, ClearOnly must **not** hop to a different early_front
/// Composite mid (serial-223416 / tip 145ee5ff: mode=follow oscillated
/// peer=5↔peer=1 → mid_sw thrash). mode=keep re-shows frozen retain;
/// mode=refresh recaptures only when early_front is the retain mid.
#[test]
fn clear_only_present_sticky_refresh_ignores_other_early_front() {
    use crate::contract::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
    use crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    use crate::runtime::mapping_write::write_bgra8;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let w = 1920u32;
    let h = 1080u32;
    let stride = w * 4;
    let need = (stride as usize) * (h as usize);
    let page_shift = PAGE_SHIFT_X86;
    let page_size = 1u64 << page_shift;
    let pages = (need as u64).div_ceil(page_size) as usize;
    for mid in [1u32, 2u32, 5u32] {
        let base_pfn = 0x100u32 + mid * 0x1000;
        let mut entries = Vec::with_capacity(pages);
        for i in 0..pages {
            let pfn = base_pfn + i as u32;
            let gpa = (pfn as u64) << page_shift;
            host.map_range(gpa, page_size as usize, 0);
            entries
                .push((((pfn as u64) << PAGE_ENTRY_PFN_SHIFT) | (PAGE_ENTRY_VALID as u64)) as u32);
        }
        assert!(state.map_surface(mid));
        {
            let m = state.mappings.get_mut(&mid).unwrap();
            m.mapped = true;
            m.mapping_internal = mid as u64;
            m.page_entries = entries;
        }
        assert!(state.set_mapping_geom(mid, w, h, MTL_FORMAT_BGRA8_UNORM));
    }
    // Bootstrap: mid 1 logo, ClearOnly present mid 2.
    let logo = vec![0x11u8; need];
    assert!(write_bgra8(&mut state, &mut host, 1, &logo, stride, w, h));
    state.note_surface_composite(1);
    state.present.early_front_mapping = 1;
    state.present.early_front_generation = 2;
    state.present.valid = true;
    state.present.width = w;
    state.present.height = h;
    let mut clear = vec![0u8; need];
    for px in clear.chunks_exact_mut(4) {
        px[3] = 255;
    }
    assert!(write_bgra8(&mut state, &mut host, 2, &clear, stride, w, h));
    state.note_surface_clear(2);
    {
        let mut payload = vec![0u8; PRESENT_X86_MIN_LEN];
        payload[PRESENT_X86_SURFACE_ID..PRESENT_X86_SURFACE_ID + 4]
            .copy_from_slice(&2u32.to_le_bytes());
        process_child_packet(
            &mut state,
            &mut host,
            5,
            &Packet {
                opcode: CHILD_OP_PRESENT_X86,
                stamp_count: 0,
                total_size: PACKET_HEADER_LEN + PRESENT_X86_MIN_LEN as u32,
                completion_stamp: 0,
                payload,
                next_head: 0,
            },
        );
    }
    assert_eq!(state.present.frame_mapping, 1);
    assert_eq!(state.present.frame_bgra[0], 0x11);
    host.actions.clear();

    // Mid 5 is latest early_front (other Composite). mode=keep must stay on 1
    // frozen bootstrap pixels (no hop, no recapture of mid1 updates).
    let gray = vec![0x99u8; need];
    assert!(write_bgra8(&mut state, &mut host, 5, &gray, stride, w, h));
    state.note_surface_composite(5);
    let logo2 = vec![0x22u8; need];
    assert!(write_bgra8(&mut state, &mut host, 1, &logo2, stride, w, h));
    state.note_surface_composite(1);
    for _ in 0..3 {
        state.present.early_front_mapping = 5;
        state.present.early_front_generation += 1;
        host.actions.clear();
        let mut payload = vec![0u8; PRESENT_X86_MIN_LEN];
        payload[PRESENT_X86_SURFACE_ID..PRESENT_X86_SURFACE_ID + 4]
            .copy_from_slice(&2u32.to_le_bytes());
        process_child_packet(
            &mut state,
            &mut host,
            5,
            &Packet {
                opcode: CHILD_OP_PRESENT_X86,
                stamp_count: 0,
                total_size: PACKET_HEADER_LEN + PRESENT_X86_MIN_LEN as u32,
                completion_stamp: 0,
                payload,
                next_head: 0,
            },
        );
        assert_eq!(
            state.present.frame_mapping, 1,
            "sticky retain must not follow early_front=5"
        );
        assert_eq!(
            state.present.frame_bgra[0], 0x11,
            "mode=keep re-shows frozen +0x188, does not recapture mid1"
        );
        assert_eq!(
            presented_mapping(&state),
            Some(1),
            "window shows retain mid 1"
        );
        assert_coalesced_paint_action(&host, "sticky refresh");
    }

    // early_front returns to retain mid → mode=refresh recaptures logo2.
    state.present.early_front_mapping = 1;
    state.present.early_front_generation += 1;
    host.actions.clear();
    {
        let mut payload = vec![0u8; PRESENT_X86_MIN_LEN];
        payload[PRESENT_X86_SURFACE_ID..PRESENT_X86_SURFACE_ID + 4]
            .copy_from_slice(&2u32.to_le_bytes());
        process_child_packet(
            &mut state,
            &mut host,
            5,
            &Packet {
                opcode: CHILD_OP_PRESENT_X86,
                stamp_count: 0,
                total_size: PACKET_HEADER_LEN + PRESENT_X86_MIN_LEN as u32,
                completion_stamp: 0,
                payload,
                next_head: 0,
            },
        );
    }
    assert_eq!(state.present.frame_mapping, 1);
    assert_eq!(
        state.present.frame_bgra[0], 0x22,
        "mode=refresh recaptures retain mid when early_front matches"
    );

    // A decoded same-geometry linear texture dependency is categorically
    // different from an unrelated writer: the full output was produced
    // from a type-2/3 input, so the next ClearOnly present may hand
    // ownership to the downstream RT.
    assert!(crate::runtime::scanout::note_linear_compositor_output(
        &mut state, 5, w, h, 77
    ));
    state.present.early_front_mapping = 1;
    host.actions.clear();
    {
        let mut payload = vec![0u8; PRESENT_X86_MIN_LEN];
        payload[PRESENT_X86_SURFACE_ID..PRESENT_X86_SURFACE_ID + 4]
            .copy_from_slice(&2u32.to_le_bytes());
        process_child_packet(
            &mut state,
            &mut host,
            5,
            &Packet {
                opcode: CHILD_OP_PRESENT_X86,
                stamp_count: 0,
                total_size: PACKET_HEADER_LEN + PRESENT_X86_MIN_LEN as u32,
                completion_stamp: 0,
                payload,
                next_head: 0,
            },
        );
    }
    assert_eq!(
        state.present.frame_mapping, 5,
        "full-geometry linear compositor edge must hand off to downstream output"
    );
    assert_eq!(state.present.frame_bgra[0], 0x99);
    let body = std::fs::read_to_string(crate::observe::fail_log_path())
        .expect("reims-vgpu-fail.log readable");
    assert!(
        body.lines().any(|line| {
            line.starts_with("OFF present_owner_graph source=linear output_mid=5 ")
                && line.contains("retain_mid=1")
                && line.contains("mode=graph_handoff cap=1")
        }),
        "linear graph ownership handoff proxy must be always-on"
    );

    // An old base writer becoming early_front again is not a reverse edge;
    // keep the graph output instead of recreating mode=follow oscillation.
    state.present.early_front_mapping = 1;
    host.actions.clear();
    {
        let mut payload = vec![0u8; PRESENT_X86_MIN_LEN];
        payload[PRESENT_X86_SURFACE_ID..PRESENT_X86_SURFACE_ID + 4]
            .copy_from_slice(&2u32.to_le_bytes());
        process_child_packet(
            &mut state,
            &mut host,
            5,
            &Packet {
                opcode: CHILD_OP_PRESENT_X86,
                stamp_count: 0,
                total_size: PACKET_HEADER_LEN + PRESENT_X86_MIN_LEN as u32,
                completion_stamp: 0,
                payload,
                next_head: 0,
            },
        );
    }
    assert_eq!(state.present.frame_mapping, 5);
    assert_eq!(state.present.frame_bgra[0], 0x99);
}

/// Peer capture fail must keep prior Composite +0x188 — never fall through
/// to capture the named ClearOnly mid (live mid_sw 1→3 alpha-only thrash).
#[test]
fn clear_only_peer_capture_fail_keeps_prior_not_clear_mid() {
    use crate::contract::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
    use crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    use crate::runtime::mapping_write::write_bgra8;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let w = 1920u32;
    let h = 1080u32;
    let stride = w * 4;
    let need = (stride as usize) * (h as usize);
    let page_shift = PAGE_SHIFT_X86;
    let page_size = 1u64 << page_shift;
    let pages = (need as u64).div_ceil(page_size) as usize;
    for mid in [1u32, 3u32] {
        let base_pfn = 0x100u32 + mid * 0x1000;
        let mut entries = Vec::with_capacity(pages);
        for i in 0..pages {
            let pfn = base_pfn + i as u32;
            let gpa = (pfn as u64) << page_shift;
            host.map_range(gpa, page_size as usize, 0);
            entries
                .push((((pfn as u64) << PAGE_ENTRY_PFN_SHIFT) | (PAGE_ENTRY_VALID as u64)) as u32);
        }
        assert!(state.map_surface(mid));
        {
            let m = state.mappings.get_mut(&mid).unwrap();
            m.mapped = true;
            m.mapping_internal = mid as u64;
            m.page_entries = entries;
        }
        assert!(state.set_mapping_geom(mid, w, h, MTL_FORMAT_BGRA8_UNORM));
    }
    let logo = vec![0x44u8; need];
    assert!(write_bgra8(&mut state, &mut host, 1, &logo, stride, w, h));
    state.note_surface_composite(1);
    state.present.early_front_mapping = 1;
    state.present.early_front_generation = 2;
    state.present.valid = true;
    state.present.width = w;
    state.present.height = h;
    let mut clear = vec![0u8; need];
    for px in clear.chunks_exact_mut(4) {
        px[3] = 255;
    }
    assert!(write_bgra8(&mut state, &mut host, 3, &clear, stride, w, h));
    state.note_surface_clear(3);
    // Bootstrap retain mid=1 via ClearOnly present mid=3.
    {
        let mut payload = vec![0u8; PRESENT_X86_MIN_LEN];
        payload[PRESENT_X86_SURFACE_ID..PRESENT_X86_SURFACE_ID + 4]
            .copy_from_slice(&3u32.to_le_bytes());
        process_child_packet(
            &mut state,
            &mut host,
            5,
            &Packet {
                opcode: CHILD_OP_PRESENT_X86,
                stamp_count: 0,
                total_size: PACKET_HEADER_LEN + PRESENT_X86_MIN_LEN as u32,
                completion_stamp: 0,
                payload,
                next_head: 0,
            },
        );
    }
    assert_eq!(state.present.frame_mapping, 1);
    assert_eq!(state.present.frame_bgra[0], 0x44);
    // Make retain mid unreadable (pages gone) so refresh would fail; ClearOnly
    // mid 3 pages remain solid black — old code fallthrough would capture them.
    {
        let m = state.mappings.get_mut(&1).unwrap();
        m.page_entries.clear();
        m.mapped = false;
    }
    state.present.early_front_mapping = 1;
    host.actions.clear();
    {
        let mut payload = vec![0u8; PRESENT_X86_MIN_LEN];
        payload[PRESENT_X86_SURFACE_ID..PRESENT_X86_SURFACE_ID + 4]
            .copy_from_slice(&3u32.to_le_bytes());
        process_child_packet(
            &mut state,
            &mut host,
            5,
            &Packet {
                opcode: CHILD_OP_PRESENT_X86,
                stamp_count: 0,
                total_size: PACKET_HEADER_LEN + PRESENT_X86_MIN_LEN as u32,
                completion_stamp: 0,
                payload,
                next_head: 0,
            },
        );
    }
    assert_eq!(
        state.present.frame_mapping, 1,
        "must not install ClearOnly mid=3 into +0x188 after peer fail"
    );
    assert_eq!(
        state.present.frame_bgra[0], 0x44,
        "prior Composite retain pixels must survive peer capture fail"
    );
    assert_eq!(
        presented_mapping(&state),
        Some(1),
        "window re-shows retain mid 1"
    );
    assert_coalesced_paint_action(&host, "peer capture fail");
}

/// Flipping early_front every ClearOnly present must keep established retain
/// (serial-222258 mid_sw thrash class when follow had no hysteresis).
#[test]
fn clear_only_present_keeps_retain_while_early_front_bounces() {
    use crate::contract::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
    use crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    use crate::runtime::mapping_write::write_bgra8;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let w = 1920u32;
    let h = 1080u32;
    let stride = w * 4;
    let need = (stride as usize) * (h as usize);
    let page_shift = PAGE_SHIFT_X86;
    let page_size = 1u64 << page_shift;
    let pages = (need as u64).div_ceil(page_size) as usize;
    for mid in [1u32, 2u32, 5u32] {
        let base_pfn = 0x100u32 + mid * 0x1000;
        let mut entries = Vec::with_capacity(pages);
        for i in 0..pages {
            let pfn = base_pfn + i as u32;
            let gpa = (pfn as u64) << page_shift;
            host.map_range(gpa, page_size as usize, 0);
            entries
                .push((((pfn as u64) << PAGE_ENTRY_PFN_SHIFT) | (PAGE_ENTRY_VALID as u64)) as u32);
        }
        assert!(state.map_surface(mid));
        {
            let m = state.mappings.get_mut(&mid).unwrap();
            m.mapped = true;
            m.mapping_internal = mid as u64;
            m.page_entries = entries;
        }
        assert!(state.set_mapping_geom(mid, w, h, MTL_FORMAT_BGRA8_UNORM));
    }
    let logo = vec![0x11u8; need];
    assert!(write_bgra8(&mut state, &mut host, 1, &logo, stride, w, h));
    state.note_surface_composite(1);
    state.present.early_front_mapping = 1;
    state.present.early_front_generation = 2;
    state.present.valid = true;
    state.present.width = w;
    state.present.height = h;
    let mut clear = vec![0u8; need];
    for px in clear.chunks_exact_mut(4) {
        px[3] = 255;
    }
    assert!(write_bgra8(&mut state, &mut host, 2, &clear, stride, w, h));
    state.note_surface_clear(2);
    // Bootstrap retain mid=1.
    {
        let mut payload = vec![0u8; PRESENT_X86_MIN_LEN];
        payload[PRESENT_X86_SURFACE_ID..PRESENT_X86_SURFACE_ID + 4]
            .copy_from_slice(&2u32.to_le_bytes());
        process_child_packet(
            &mut state,
            &mut host,
            5,
            &Packet {
                opcode: CHILD_OP_PRESENT_X86,
                stamp_count: 0,
                total_size: PACKET_HEADER_LEN + PRESENT_X86_MIN_LEN as u32,
                completion_stamp: 0,
                payload,
                next_head: 0,
            },
        );
    }
    assert_eq!(state.present.frame_mapping, 1);
    let gray = vec![0x99u8; need];
    assert!(write_bgra8(&mut state, &mut host, 5, &gray, stride, w, h));
    state.note_surface_composite(5);

    // Bounce early_front 5 → 1 → 5 across ClearOnly presents; retain stays 1.
    for &ef in &[5u32, 1u32, 5u32, 1u32] {
        state.present.early_front_mapping = ef;
        state.present.early_front_generation += 1;
        host.actions.clear();
        let mut payload = vec![0u8; PRESENT_X86_MIN_LEN];
        payload[PRESENT_X86_SURFACE_ID..PRESENT_X86_SURFACE_ID + 4]
            .copy_from_slice(&2u32.to_le_bytes());
        process_child_packet(
            &mut state,
            &mut host,
            5,
            &Packet {
                opcode: CHILD_OP_PRESENT_X86,
                stamp_count: 0,
                total_size: PACKET_HEADER_LEN + PRESENT_X86_MIN_LEN as u32,
                completion_stamp: 0,
                payload,
                next_head: 0,
            },
        );
        assert_eq!(
            state.present.frame_mapping, 1,
            "bounce early_front={ef} must keep retain mid=1"
        );
        assert_eq!(state.present.frame_bgra[0], 0x11);
    }
}

/// CmdDeleteTask (root 0x20) must clear the task — not flood UnknownRootOpcode.
#[test]
fn delete_task_root_clears_active_task() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    assert!(state.define_task(3, 0x1000, 2));
    assert!(state.tasks[3].active);
    process_root_packet(
        &mut state,
        &mut host,
        &Packet {
            opcode: ROOT_OP_DELETE_TASK,
            stamp_count: 0,
            total_size: PACKET_HEADER_LEN + 4,
            completion_stamp: 0,
            payload: 3u32.to_le_bytes().to_vec(),
            next_head: 0,
        },
    );
    assert!(!state.tasks[3].active, "DeleteTask must deactivate task 3");
    assert!(
        !state
            .fails
            .iter()
            .any(|e| matches!(e, FailEvent::UnknownRootOpcode { opcode: 0x20, .. })),
        "0x20 must not be UnknownRootOpcode"
    );
}

/// CmdReplacePhysical (0x3c) is stamp-complete bookkeeping — not UnknownChildOpcode.
#[test]
fn replace_physical_is_accepted_not_unknown() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let pkt = Packet {
        opcode: CHILD_OP_REPLACE_PHYSICAL,
        stamp_count: 0,
        total_size: PACKET_HEADER_LEN + 8,
        completion_stamp: 0,
        payload: vec![0u8; 8], // {taskID, objectID} placeholder
        next_head: 0,
    };
    process_child_packet(&mut state, &mut host, 2, &pkt);
    assert!(
        !state
            .fails
            .iter()
            .any(|e| matches!(e, FailEvent::UnknownChildOpcode { opcode: 0x3c, .. })),
        "0x3c must not flood UnknownChildOpcode"
    );
}

#[test]
fn delete_iosurface_backing_condemns_then_second_delete_tears_down() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    assert!(state.map_surface(3));
    {
        let m = state.mappings.get_mut(&3).unwrap();
        m.page_entries = vec![0x101];
        m.mapping_internal = 0x1234;
    }
    let delete = |state: &mut DeviceState, host: &mut FakeHost| {
        let mut payload = Vec::new();
        payload.extend_from_slice(&3u32.to_le_bytes()); // objectID
        payload.extend_from_slice(&1u32.to_le_bytes()); // taskID
        process_child_packet(
            state,
            host,
            2,
            &Packet {
                opcode: CHILD_OP_DELETE_IOSURFACE_BACKING2,
                stamp_count: 0,
                total_size: PACKET_HEADER_LEN + payload.len() as u32,
                completion_stamp: 0,
                payload,
                next_head: 0,
            },
        );
    };
    // First delete: the id may already carry a live re-used incarnation
    // (the delete trails the guest release) — condemn: retire bindings,
    // keep content state for the resolve-time fingerprint decision.
    delete(&mut state, &mut host);
    let m = state.mappings.get(&3).unwrap();
    assert!(m.mapped, "condemn keeps the slot live");
    assert!(m.page_entries.is_empty(), "bindings must be retired");
    assert_eq!(m.condemned_entries.as_deref(), Some(&[0x101u32][..]));
    // Second delete with no resolve between: genuinely dead — full
    // teardown.
    delete(&mut state, &mut host);
    let m = state.mappings.get(&3).unwrap();
    assert!(!m.mapped);
    assert!(m.page_entries.is_empty());
    assert!(m.condemned_entries.is_none());
    assert_eq!(m.mapping_internal, 0);
}

/// ClearOnly present with sticky Composite early_front bootstraps leave-BAR1
/// from the peer mid (live x86 dual-mid: present 2/3 clear, content 1/4).
#[test]
fn clear_only_present_captures_early_front_composite_peer() {
    use crate::contract::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
    use crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    use crate::runtime::mapping_write::write_bgra8;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let w = 1920u32;
    let h = 1080u32;
    let stride = w * 4;
    let need = (stride as usize) * (h as usize);
    let page_shift = PAGE_SHIFT_X86;
    let page_size = 1u64 << page_shift;
    let pages = (need as u64).div_ceil(page_size) as usize;
    for mid in [1u32, 2u32] {
        let base_pfn = 0x100u32 + mid * 0x1000;
        let mut entries = Vec::with_capacity(pages);
        for i in 0..pages {
            let pfn = base_pfn + i as u32;
            let gpa = (pfn as u64) << page_shift;
            host.map_range(gpa, page_size as usize, 0);
            entries
                .push((((pfn as u64) << PAGE_ENTRY_PFN_SHIFT) | (PAGE_ENTRY_VALID as u64)) as u32);
        }
        assert!(state.map_surface(mid));
        {
            let m = state.mappings.get_mut(&mid).unwrap();
            m.mapped = true;
            m.mapping_internal = mid as u64;
            m.page_entries = entries;
        }
        assert!(state.set_mapping_geom(mid, w, h, MTL_FORMAT_BGRA8_UNORM));
    }
    // Mid 1: composite content (logo/desktop) — never presented by guest.
    let gray = vec![0xAAu8; need];
    assert!(write_bgra8(&mut state, &mut host, 1, &gray, stride, w, h));
    state.note_surface_composite(1);
    state.present.early_front_mapping = 1;
    state.present.early_front_generation = 3;
    state.present.valid = true;
    state.present.width = w;
    state.present.height = h;
    assert!(!state.present.frame_flush_seen);
    assert!(!state.present.frame_valid);

    // Mid 2: clear-only present (live present_op6 sid=2).
    let mut clear = vec![0u8; need];
    for px in clear.chunks_exact_mut(4) {
        px[3] = 255;
    }
    assert!(write_bgra8(&mut state, &mut host, 2, &clear, stride, w, h));
    state.note_surface_clear(2);

    let mut payload = vec![0u8; PRESENT_X86_MIN_LEN];
    payload[PRESENT_X86_SURFACE_ID..PRESENT_X86_SURFACE_ID + 4]
        .copy_from_slice(&2u32.to_le_bytes());
    process_child_packet(
        &mut state,
        &mut host,
        5,
        &Packet {
            opcode: CHILD_OP_PRESENT_X86,
            stamp_count: 0,
            total_size: PACKET_HEADER_LEN + PRESENT_X86_MIN_LEN as u32,
            completion_stamp: 0,
            payload,
            next_head: 0,
        },
    );

    assert_eq!(
        state.present.present_mapping, 2,
        "guest still names ClearOnly mid"
    );
    assert!(
        state.present.frame_flush_seen,
        "leave-BAR1 once Composite peer is capturable"
    );
    assert_eq!(state.present.frame_mapping, 1, "+0x188 from Composite peer");
    assert!(state.present.frame_valid);
    assert_eq!(state.present.frame_bgra[0], 0xAA);
    assert_eq!(
        presented_mapping(&state),
        Some(1),
        "window shows peer mid 1, not clear mid 2"
    );
    assert_coalesced_paint_action(&host, "composite peer");
}

/// A retained-frame re-show is still a present of that retained member.
/// `capture_present_frame` records presented geometry on recapture, but
/// mode=keep/keep_retain deliberately does not recapture. If that path does
/// not refresh `presented_geoms`, direct-present export can ask for a stale
/// per-surface identity even though the shared OutputGroup resident is the
/// live logical framebuffer
/// (`export_present_miss outcome=orphan want=surface group=ready`).
/// Vulkan-arm only: asserts on `TargetIdentity`, which is the Vulkan
/// engine's resident-identity type, and on `import_present`, a
/// `backend-vulkan` module.
#[cfg(feature = "backend-vulkan")]
#[test]
fn clear_only_keep_retain_marks_retained_member_presented_for_group_identity() {
    use crate::backend::vulkan::engine::TargetIdentity;
    use crate::contract::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
    use crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let w = 1920u32;
    let h = 1080u32;
    let stride = w * 4;
    let need = (stride as usize) * (h as usize);
    for mid in [1u32, 2u32, 5u32] {
        assert!(state.map_surface(mid));
        {
            let m = state.mappings.get_mut(&mid).unwrap();
            m.mapped = true;
            m.mapping_internal = mid as u64;
            m.has_geom = true;
            m.width = w;
            m.height = h;
            m.format = MTL_FORMAT_BGRA8_UNORM;
            m.page_entries = vec![
                (((0x1000 + mid) as u64) << PAGE_ENTRY_PFN_SHIFT | PAGE_ENTRY_VALID as u64) as u32,
            ];
        }
    }
    // Both logical framebuffer members are known, but only peer mid 5 has
    // prior present evidence. Mid 1 is a valid retained frame about to be
    // re-shown through keep_retain, not recaptured.
    state.note_compositor_member_published(1, w, h);
    state.note_compositor_member_published(5, w, h);
    state.note_presented_geom(5, w, h);
    assert!(matches!(
        crate::runtime::import_present::surface_identity(&state, 1, w, h),
        TargetIdentity::Surface { id: 1, .. }
    ));

    state.note_surface_clear(2);
    state.present.valid = true;
    state.present.width = w;
    state.present.height = h;
    state.present.frame_flush_seen = true;
    state.present.frame_valid = true;
    state.present.frame_mapping = 1;
    state.present.frame_generation = 9;
    state.present.frame_width = w;
    state.present.frame_height = h;
    state.present.frame_bgra = vec![0x66u8; need];

    let mut payload = vec![0u8; PRESENT_X86_MIN_LEN];
    payload[PRESENT_X86_SURFACE_ID..PRESENT_X86_SURFACE_ID + 4]
        .copy_from_slice(&2u32.to_le_bytes());
    process_child_packet(
        &mut state,
        &mut host,
        5,
        &Packet {
            opcode: CHILD_OP_PRESENT_X86,
            stamp_count: 0,
            total_size: PACKET_HEADER_LEN + PRESENT_X86_MIN_LEN as u32,
            completion_stamp: 0,
            payload,
            next_head: 0,
        },
    );

    assert_eq!(
        state.present.frame_mapping, 1,
        "keep_retain re-shows the retained member"
    );
    assert!(
        state.presented_at(1, w, h),
        "the retained member was displayed and must refresh present evidence"
    );
    assert!(matches!(
        crate::runtime::import_present::surface_identity(&state, 1, w, h),
        TargetIdentity::OutputGroup { .. }
    ));
}

/// A ClearOnly present whose page table is identical to a Composite
/// mapping's names THAT frame (two views of one IOSurface). page_alias
/// must beat the sticky retain — this is the guest's double-buffer
/// alternation, not thrash.
#[test]
fn clear_only_present_follows_page_identical_composite_peer() {
    use crate::contract::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
    use crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    use crate::runtime::mapping_write::write_bgra8;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let w = 1920u32;
    let h = 1080u32;
    let stride = w * 4;
    let need = (stride as usize) * (h as usize);
    let page_shift = PAGE_SHIFT_X86;
    let page_size = 1u64 << page_shift;
    let pages = (need as u64).div_ceil(page_size) as usize;
    let build = |host: &mut FakeHost, base_pfn: u32| -> Vec<u32> {
        let mut entries = Vec::with_capacity(pages);
        for i in 0..pages {
            let pfn = base_pfn + i as u32;
            let gpa = (pfn as u64) << page_shift;
            host.map_range(gpa, page_size as usize, 0);
            entries
                .push((((pfn as u64) << PAGE_ENTRY_PFN_SHIFT) | (PAGE_ENTRY_VALID as u64)) as u32);
        }
        entries
    };
    let shared = build(&mut host, 0x10_000);
    let retain_pages = build(&mut host, 0x80_000);
    // Composite frame buffer (mapping namespace) on the shared pages.
    for (mid, entries) in [(1u32, shared.clone()), (2u32, shared), (3u32, retain_pages)] {
        assert!(state.map_surface(mid));
        {
            let m = state.mappings.get_mut(&mid).unwrap();
            m.mapped = true;
            m.mapping_internal = mid as u64;
            m.page_entries = entries;
        }
        assert!(state.set_mapping_geom(mid, w, h, MTL_FORMAT_BGRA8_UNORM));
    }
    // Mid 1 composited this frame's content into the shared IOSurface.
    let gray = vec![0xAAu8; need];
    assert!(write_bgra8(&mut state, &mut host, 1, &gray, stride, w, h));
    state.note_surface_composite(1);
    // Mid 3 is the previously retained Composite (sticky retain target).
    let blue = vec![0x55u8; need];
    assert!(write_bgra8(&mut state, &mut host, 3, &blue, stride, w, h));
    state.note_surface_composite(3);
    state.present.frame_valid = true;
    state.present.frame_mapping = 3;
    state.present.frame_generation = 7;
    state.present.frame_width = w;
    state.present.frame_height = h;
    state.present.frame_bgra = blue;
    state.present.frame_flush_seen = true;
    state.present.valid = true;
    state.present.width = w;
    state.present.height = h;
    // Guest presents the surface-id view (ClearOnly) of the shared pages.
    state.note_surface_clear(2);

    let mut payload = vec![0u8; PRESENT_X86_MIN_LEN];
    payload[PRESENT_X86_SURFACE_ID..PRESENT_X86_SURFACE_ID + 4]
        .copy_from_slice(&2u32.to_le_bytes());
    process_child_packet(
        &mut state,
        &mut host,
        5,
        &Packet {
            opcode: CHILD_OP_PRESENT_X86,
            stamp_count: 0,
            total_size: PACKET_HEADER_LEN + PRESENT_X86_MIN_LEN as u32,
            completion_stamp: 0,
            payload,
            next_head: 0,
        },
    );

    assert_eq!(state.present.present_mapping, 2, "guest names the sid view");
    assert_eq!(
        state.present.frame_mapping, 1,
        "page_alias follows the page-identical Composite peer over the sticky retain"
    );
    assert!(state.present.frame_valid);
    assert_eq!(
        state.present.frame_bgra[0], 0xAA,
        "captured the named frame"
    );
    assert_eq!(
        presented_mapping(&state),
        Some(1),
        "window shows the aliased Composite mid"
    );
    assert_coalesced_paint_action(&host, "page-identical peer");
}

/// Live x86 census (2026-07-16 boot): WindowServer double-buffers full
/// frames across mids 1↔5 with damage passes (partial quads, never a
/// full-coverage edge after boot), while presents name disjoint ClearOnly
/// sids. The graph pin froze on the last boot-time full paint and every
/// present captured the same mid — half the frames never shown. Composite
/// writebacks into *proven* compositor-output members must re-pin the
/// graph so ClearOnly presents follow the alternation.
#[test]
fn clear_only_present_follows_member_store_alternation() {
    use crate::contract::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
    use crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    use crate::runtime::mapping_write::write_bgra8;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let w = 1920u32;
    let h = 1080u32;
    let stride = w * 4;
    let need = (stride as usize) * (h as usize);
    let page_shift = PAGE_SHIFT_X86;
    let page_size = 1u64 << page_shift;
    let pages = (need as u64).div_ceil(page_size) as usize;
    for mid in [1u32, 2u32, 5u32] {
        let base_pfn = 0x100u32 + mid * 0x1000;
        let mut entries = Vec::with_capacity(pages);
        for i in 0..pages {
            let pfn = base_pfn + i as u32;
            host.map_range((pfn as u64) << page_shift, page_size as usize, 0);
            entries
                .push((((pfn as u64) << PAGE_ENTRY_PFN_SHIFT) | (PAGE_ENTRY_VALID as u64)) as u32);
        }
        assert!(state.map_surface(mid));
        {
            let m = state.mappings.get_mut(&mid).unwrap();
            m.mapped = true;
            m.mapping_internal = mid as u64;
            m.page_entries = entries;
        }
        assert!(state.set_mapping_geom(mid, w, h, MTL_FORMAT_BGRA8_UNORM));
    }
    // Boot-time full-coverage passes prove both double-buffer halves.
    let gray = vec![0xAAu8; need];
    assert!(write_bgra8(&mut state, &mut host, 1, &gray, stride, w, h));
    state.note_surface_composite(1);
    assert!(crate::runtime::scanout::note_linear_compositor_output(
        &mut state, 1, w, h, 11
    ));
    let blue = vec![0x55u8; need];
    assert!(write_bgra8(&mut state, &mut host, 5, &blue, stride, w, h));
    state.note_surface_composite(5);
    assert!(crate::runtime::scanout::note_linear_compositor_output(
        &mut state, 5, w, h, 12
    ));
    state.note_surface_clear(2);
    state.present.valid = true;
    state.present.width = w;
    state.present.height = h;

    let present_sid2 = |state: &mut DeviceState, host: &mut FakeHost| {
        let mut payload = vec![0u8; PRESENT_X86_MIN_LEN];
        payload[PRESENT_X86_SURFACE_ID..PRESENT_X86_SURFACE_ID + 4]
            .copy_from_slice(&2u32.to_le_bytes());
        process_child_packet(
            state,
            host,
            5,
            &Packet {
                opcode: CHILD_OP_PRESENT_X86,
                stamp_count: 0,
                total_size: PACKET_HEADER_LEN + PRESENT_X86_MIN_LEN as u32,
                completion_stamp: 0,
                payload,
                next_head: 0,
            },
        );
    };

    // First present pins on the last proven edge (mid 5).
    present_sid2(&mut state, &mut host);
    assert_eq!(state.present.frame_mapping, 5);
    assert_eq!(state.present.frame_bgra[0], 0x55);

    // Steady state: damage-pass Stores alternate the members; each present
    // must capture the buffer the guest just finished writing.
    for (mid, byte) in [(1u32, 0x11u8), (5u32, 0x22u8), (1u32, 0x33u8)] {
        let fill = vec![byte; need];
        assert!(write_bgra8(&mut state, &mut host, mid, &fill, stride, w, h));
        state.note_surface_composite(mid);
        crate::runtime::scanout::note_front_buffer_writeback(&mut state, &mut host, mid, w, h, 0);
        host.actions.clear();
        present_sid2(&mut state, &mut host);
        assert_eq!(
            state.present.frame_mapping, mid,
            "present follows the member the guest last stored"
        );
        assert_eq!(
            state.present.frame_bgra[0], byte,
            "captured the fresh frame"
        );
        assert_eq!(
            presented_mapping(&state),
            Some(mid),
            "window shows the alternated member"
        );
        assert_coalesced_paint_action(&host, "member alternation");
    }
}

/// The guest pipelines its double buffer a frame ahead: ring order is
/// store B, store A, present, present (live x86 drag census 2026-07-18,
/// serial-20260718-105304: both member stores land before both ClearOnly
/// presents every ~78 ms cycle). Pin/latest-following captured the newest
/// member for BOTH presents — B's frame never displayed (halved display
/// rate) and the on-screen frame paired with the wrong present slot
/// (dual-mid residue class). The present↔store FIFO must pair present #1
/// with B and present #2 with A.
#[test]
fn clear_only_present_pairs_with_store_fifo_order() {
    use crate::contract::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
    use crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    use crate::runtime::mapping_write::write_bgra8;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let w = 1920u32;
    let h = 1080u32;
    let stride = w * 4;
    let need = (stride as usize) * (h as usize);
    let page_shift = PAGE_SHIFT_X86;
    let page_size = 1u64 << page_shift;
    let pages = (need as u64).div_ceil(page_size) as usize;
    for mid in [1u32, 2u32, 5u32] {
        let base_pfn = 0x100u32 + mid * 0x1000;
        let mut entries = Vec::with_capacity(pages);
        for i in 0..pages {
            let pfn = base_pfn + i as u32;
            host.map_range((pfn as u64) << page_shift, page_size as usize, 0);
            entries
                .push((((pfn as u64) << PAGE_ENTRY_PFN_SHIFT) | (PAGE_ENTRY_VALID as u64)) as u32);
        }
        assert!(state.map_surface(mid));
        {
            let m = state.mappings.get_mut(&mid).unwrap();
            m.mapped = true;
            m.mapping_internal = mid as u64;
            m.page_entries = entries;
        }
        assert!(state.set_mapping_geom(mid, w, h, MTL_FORMAT_BGRA8_UNORM));
    }
    // Boot-time full-coverage passes prove both double-buffer halves.
    let gray = vec![0xAAu8; need];
    assert!(write_bgra8(&mut state, &mut host, 1, &gray, stride, w, h));
    state.note_surface_composite(1);
    assert!(crate::runtime::scanout::note_linear_compositor_output(
        &mut state, 1, w, h, 11
    ));
    let blue = vec![0x55u8; need];
    assert!(write_bgra8(&mut state, &mut host, 5, &blue, stride, w, h));
    state.note_surface_composite(5);
    assert!(crate::runtime::scanout::note_linear_compositor_output(
        &mut state, 5, w, h, 12
    ));
    state.note_surface_clear(2);
    state.present.valid = true;
    state.present.width = w;
    state.present.height = h;

    let present_sid2 = |state: &mut DeviceState, host: &mut FakeHost| {
        let mut payload = vec![0u8; PRESENT_X86_MIN_LEN];
        payload[PRESENT_X86_SURFACE_ID..PRESENT_X86_SURFACE_ID + 4]
            .copy_from_slice(&2u32.to_le_bytes());
        process_child_packet(
            state,
            host,
            5,
            &Packet {
                opcode: CHILD_OP_PRESENT_X86,
                stamp_count: 0,
                total_size: PACKET_HEADER_LEN + PRESENT_X86_MIN_LEN as u32,
                completion_stamp: 0,
                payload,
                next_head: 0,
            },
        );
    };

    // Establish the retain (leave-BAR1) before the pipelined cycle.
    present_sid2(&mut state, &mut host);
    assert_eq!(state.present.frame_mapping, 5);

    // Pipelined cycle: BOTH members store before EITHER present drains.
    let fill5 = vec![0x44u8; need];
    assert!(write_bgra8(&mut state, &mut host, 5, &fill5, stride, w, h));
    state.note_surface_composite(5);
    crate::runtime::scanout::note_front_buffer_writeback(&mut state, &mut host, 5, w, h, 0);
    let fill1 = vec![0x66u8; need];
    assert!(write_bgra8(&mut state, &mut host, 1, &fill1, stride, w, h));
    state.note_surface_composite(1);
    crate::runtime::scanout::note_front_buffer_writeback(&mut state, &mut host, 1, w, h, 0);
    assert_eq!(
        state.present_store_fifo.len(),
        2,
        "both member stores queue for pairing"
    );

    present_sid2(&mut state, &mut host);
    assert_eq!(
        state.present.frame_mapping, 5,
        "present #1 pairs with the OLDER store (mid 5), not the newest pin"
    );
    assert_eq!(
        state.present.frame_bgra[0], 0x44,
        "present #1 shows mid 5's frame"
    );

    present_sid2(&mut state, &mut host);
    assert_eq!(
        state.present.frame_mapping, 1,
        "present #2 pairs with the newer store (mid 1)"
    );
    assert_eq!(
        state.present.frame_bgra[0], 0x66,
        "present #2 shows mid 1's frame"
    );
    assert!(
        state.present_store_fifo.is_empty(),
        "both entries consumed by their presents"
    );

    // Present without a new store (cursor-only frame): FIFO empty, the
    // retain refresh fallback re-captures the current member — no panic,
    // no stale pop.
    present_sid2(&mut state, &mut host);
    assert_eq!(state.present.frame_mapping, 1);
}

/// Fullscreen-transition torn-capture guard: a store_fifo entry pairs a
/// present with a member that missed a RUN of full frames (its
/// `dense_frame_seq` lags a same-geometry peer by >= RETENTION_GAP_MARGIN — a
/// scaling snapshot enqueued on one early full-frame Store, then abandoned).
/// Capturing it would show stale / partially-unwritten pages (the
/// vertical-strip + checkerboard torn frame). The present drain must
/// substitute the full-frame-freshest same-geometry peer as the capture
/// source. Keyed on the decoded full-frame-Store sequence, not pixel content;
/// the healthy-alternation case (both within 1 → no substitution) is locked by
/// `clear_only_present_pairs_with_store_fifo_order`.
#[test]
fn clear_only_present_substitutes_fresh_peer_for_starved_fifo_member() {
    use crate::contract::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
    use crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    use crate::runtime::census::present_proxy::RETENTION_GAP_MARGIN;
    use crate::runtime::mapping_write::write_bgra8;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let w = 1920u32;
    let h = 1080u32;
    let stride = w * 4;
    let need = (stride as usize) * (h as usize);
    let page_shift = PAGE_SHIFT_X86;
    let page_size = 1u64 << page_shift;
    let pages = (need as u64).div_ceil(page_size) as usize;
    for mid in [1u32, 5u32, 2u32] {
        let base_pfn = 0x100u32 + mid * 0x1000;
        let mut entries = Vec::with_capacity(pages);
        for i in 0..pages {
            let pfn = base_pfn + i as u32;
            host.map_range((pfn as u64) << page_shift, page_size as usize, 0);
            entries
                .push((((pfn as u64) << PAGE_ENTRY_PFN_SHIFT) | (PAGE_ENTRY_VALID as u64)) as u32);
        }
        assert!(state.map_surface(mid));
        {
            let m = state.mappings.get_mut(&mid).unwrap();
            m.mapped = true;
            m.mapping_internal = mid as u64;
            m.page_entries = entries;
        }
        assert!(state.set_mapping_geom(mid, w, h, MTL_FORMAT_BGRA8_UNORM));
    }
    // mid 1 = the fresh peer (0x11); mid 5 = the starved FIFO member (0x55).
    let fresh = vec![0x11u8; need];
    assert!(write_bgra8(&mut state, &mut host, 1, &fresh, stride, w, h));
    state.note_surface_composite(1);
    assert!(crate::runtime::scanout::note_linear_compositor_output(
        &mut state, 1, w, h, 11
    ));
    // mid 1 is a genuine swapchain sibling displayed as it alternates as front;
    // mark it presented so the presented-peer gate keeps it eligible as the fresh
    // substitute (a never-displayed publisher is excluded — the residue guard).
    state.note_presented_geom(1, w, h);
    let starved = vec![0x55u8; need];
    assert!(write_bgra8(
        &mut state, &mut host, 5, &starved, stride, w, h
    ));
    state.note_surface_composite(5);
    assert!(crate::runtime::scanout::note_linear_compositor_output(
        &mut state, 5, w, h, 12
    ));
    // Build the dense gap: mid 5 gets ONE full-frame publish, then mid 1 gets
    // a RUN, so mid 5 lags mid 1 by >= RETENTION_GAP_MARGIN.
    state.note_compositor_member_published(5, w, h);
    for _ in 0..(RETENTION_GAP_MARGIN + 2) {
        state.note_compositor_member_published(1, w, h);
    }
    let (peer, mine_seq, peer_seq) = state
        .dense_retention_gap(5, w, h)
        .expect("mid 5 lags a fresher peer");
    assert_eq!(peer, 1, "mid 1 is the full-frame-freshest peer");
    assert!(
        peer_seq >= mine_seq + RETENTION_GAP_MARGIN,
        "the gap crosses the substitution margin"
    );
    // The FIFO pairs the next present with the STARVED member (mid 5).
    assert!(state.note_member_store(5, w, h, 12));
    state.note_surface_clear(2);
    state.present.valid = true;
    state.present.width = w;
    state.present.height = h;

    let mut payload = vec![0u8; PRESENT_X86_MIN_LEN];
    payload[PRESENT_X86_SURFACE_ID..PRESENT_X86_SURFACE_ID + 4]
        .copy_from_slice(&2u32.to_le_bytes());
    process_child_packet(
        &mut state,
        &mut host,
        5,
        &Packet {
            opcode: CHILD_OP_PRESENT_X86,
            stamp_count: 0,
            total_size: PACKET_HEADER_LEN + PRESENT_X86_MIN_LEN as u32,
            completion_stamp: 0,
            payload,
            next_head: 0,
        },
    );

    assert_eq!(
        state.present.frame_mapping, 1,
        "starved FIFO member (mid 5) substituted by the full-frame-freshest peer (mid 1)"
    );
    assert_eq!(
        state.present.frame_bgra[0], 0x11,
        "captured the fresh peer's content, not the starved member's stale pages"
    );
}

/// Direct Composite-named present (no ClearOnly pairing): the transaction
/// payload carries exactly one thing — plane 0's surface id — so the only
/// correct capture source is the surface the guest named. No comparison
/// between our own full-frame sequences may override it, however far the
/// named member's sequence lags a same-geometry peer's. Substituting the
/// "denser" peer is what shows a buffer one rotation step behind the one the
/// guest asked for: residue when a window closed in between, a stale region
/// when one moved, and visible thrash as the choice oscillates.
#[test]
fn composite_named_present_captures_the_named_member_however_far_it_lags() {
    use crate::contract::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
    use crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    use crate::runtime::census::present_proxy::RETENTION_GAP_MARGIN;
    use crate::runtime::mapping_write::write_bgra8;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let w = 1920u32;
    let h = 1080u32;
    let stride = w * 4;
    let need = (stride as usize) * (h as usize);
    let page_shift = PAGE_SHIFT_X86;
    let page_size = 1u64 << page_shift;
    let pages = (need as u64).div_ceil(page_size) as usize;
    for mid in [1u32, 5u32] {
        let base_pfn = 0x100u32 + mid * 0x1000;
        let mut entries = Vec::with_capacity(pages);
        for i in 0..pages {
            let pfn = base_pfn + i as u32;
            host.map_range((pfn as u64) << page_shift, page_size as usize, 0);
            entries
                .push((((pfn as u64) << PAGE_ENTRY_PFN_SHIFT) | (PAGE_ENTRY_VALID as u64)) as u32);
        }
        assert!(state.map_surface(mid));
        {
            let m = state.mappings.get_mut(&mid).unwrap();
            m.mapped = true;
            m.mapping_internal = mid as u64;
            m.page_entries = entries;
        }
        assert!(state.set_mapping_geom(mid, w, h, MTL_FORMAT_BGRA8_UNORM));
    }
    let fresh = vec![0x11u8; need];
    assert!(write_bgra8(&mut state, &mut host, 1, &fresh, stride, w, h));
    state.note_surface_composite(1);
    assert!(crate::runtime::scanout::note_linear_compositor_output(
        &mut state, 1, w, h, 11
    ));
    let stale = vec![0x55u8; need];
    assert!(write_bgra8(&mut state, &mut host, 5, &stale, stride, w, h));
    state.note_surface_composite(5);
    assert!(crate::runtime::scanout::note_linear_compositor_output(
        &mut state, 5, w, h, 12
    ));
    // Both members are genuine swapchain buffers that alternate as the presented
    // front. Mark mid 1 displayed once at this geometry so the presented-peer gate
    // in `dense_retention_gap` keeps it eligible as the fresh substitute — a buffer
    // the guest never displays (a WebKit content tile / offscreen publisher) is NOT
    // a valid substitute, which is the intermittent-residue guard this gate adds.
    state.note_presented_geom(1, w, h);
    state.present.valid = true;
    state.present.width = w;
    state.present.height = h;

    let present_named = |state: &mut DeviceState, host: &mut FakeHost, mid: u32| {
        let mut payload = vec![0u8; PRESENT_X86_MIN_LEN];
        payload[PRESENT_X86_SURFACE_ID..PRESENT_X86_SURFACE_ID + 4]
            .copy_from_slice(&mid.to_le_bytes());
        process_child_packet(
            state,
            host,
            5,
            &Packet {
                opcode: CHILD_OP_PRESENT_X86,
                stamp_count: 0,
                total_size: PACKET_HEADER_LEN + PRESENT_X86_MIN_LEN as u32,
                completion_stamp: 0,
                payload,
                next_head: 0,
            },
        );
    };

    // Healthy alternation: both members publish, the named member is captured.
    state.note_compositor_member_published(5, w, h);
    state.note_compositor_member_published(1, w, h);
    present_named(&mut state, &mut host, 5);
    assert_eq!(
        state.present.frame_mapping, 5,
        "alternation captures the named member"
    );
    assert_eq!(state.present.frame_bgra[0], 0x55);

    // Drive the named member's full-frame sequence arbitrarily far behind its
    // peer's: mid 1 publishes a long run while mid 5 receives none. The guest
    // still names mid 5, so mid 5 is still what goes on screen.
    for _ in 0..(RETENTION_GAP_MARGIN * 8 + 2) {
        state.note_compositor_member_published(1, w, h);
    }
    let (peer, named_seq, peer_seq) = state
        .dense_retention_gap(5, w, h)
        .expect("mid 1 is a same-geometry presented peer of mid 5");
    assert_eq!(peer, 1);
    assert!(
        peer_seq - named_seq > RETENTION_GAP_MARGIN,
        "the lag this test needs is present: {peer_seq} - {named_seq}"
    );
    present_named(&mut state, &mut host, 5);
    assert_eq!(
        state.present.frame_mapping, 5,
        "the guest named mid 5; no sequence comparison may substitute a peer"
    );
    assert_eq!(
        state.present.frame_bgra[0], 0x55,
        "captured the named member's content, not the peer's"
    );
}

/// ClearOnly buffer-init present must not leave BAR1 (post-kdp handoff).
#[test]
fn clear_only_init_present_defers_frame_flush_boundary() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    assert!(state.map_surface(2));
    {
        let m = state.mappings.get_mut(&2).unwrap();
        m.mapped = true;
        m.has_geom = true;
        m.width = 1920;
        m.height = 1080;
        m.content_generation = 1;
        m.page_entries = vec![1];
    }
    state.note_surface_clear(2);
    assert!(!state.present.frame_valid);
    assert!(!state.present.frame_flush_seen);

    let mut payload = vec![0u8; PRESENT_X86_MIN_LEN.max(12)];
    payload[PRESENT_X86_SURFACE_ID..PRESENT_X86_SURFACE_ID + 4]
        .copy_from_slice(&2u32.to_le_bytes());
    let pkt = Packet {
        opcode: CHILD_OP_PRESENT_X86,
        stamp_count: 0,
        total_size: PACKET_HEADER_LEN + payload.len() as u32,
        completion_stamp: 0,
        payload,
        next_head: 0,
    };
    process_child_packet(&mut state, &mut host, 5, &pkt);

    // Guest present bookkeeping yes; leave-BAR1 boundary no.
    assert_eq!(state.present.present_mapping, 2);
    assert!(!state.present.frame_flush_seen);
    assert!(!state.present.frame_valid);
    // Was "must not enqueue a ScanoutUpdate over the early console". That
    // assertion is vacuous now that nothing enqueues one, so it is re-pointed
    // at the invariant it existed to protect: a ClearOnly init must not hand
    // the display a frame at all, leaving the early console owning the screen.
    assert_eq!(
        presented_mapping(&state),
        None,
        "ClearOnly init must not hand the display a frame over the early console"
    );
    assert_coalesced_paint_action(&host, "clear-only init");
    // host_console decision stays on early FB.
    assert!(crate::host_console_uses_bar1(false, false));
    assert!(crate::host_console_uses_bar1(
        state.present.frame_flush_seen,
        false
    ));
}

/// A display transaction cannot overtake an EXEC packet held on another
/// child channel while immutable AIR translation is loading. Repeated
/// polls hold the same packet without side effects or proxy-log flooding;
/// once ready, the packet completes normally.
#[test]
fn present_holds_for_translation_deferred_on_other_channel() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    state.translation_deferred_mask = 1 << 1;

    assert_eq!(
        present_named_mapping(&mut state, &mut host, 5, 2),
        ChildPacketDisposition::Deferred
    );
    assert_eq!(
        present_named_mapping(&mut state, &mut host, 5, 2),
        ChildPacketDisposition::Deferred
    );

    assert_eq!(state.present_translation_holds, 1);
    assert_eq!(state.present_translation_hold_mask, 1 << 5);
    assert_eq!(state.present.present_mapping, 0);
    assert!(!state.present.frame_flush_seen);

    state.translation_deferred_mask = 0;
    assert_eq!(
        present_named_mapping(&mut state, &mut host, 5, 2),
        ChildPacketDisposition::Complete
    );
    assert_eq!(state.present_translation_hold_mask, 0);
    assert_eq!(state.present.present_mapping, 2);
    assert!(state.present.frame_flush_seen);
}

/// The currently executing display channel cannot be an overtaken sibling
/// and is excluded from the proxy mask.
#[test]
fn present_does_not_hold_for_current_channel_translation_bit() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    state.translation_deferred_mask = 1 << 5;

    assert_eq!(
        present_named_mapping(&mut state, &mut host, 5, 2),
        ChildPacketDisposition::Complete
    );

    assert_eq!(state.present_translation_holds, 0);
}

/// A cold-translation EXEC owns the scheduler timeline even though its AIR
/// worker is asynchronous. A sibling Unmap must remain at FIFO head with
/// its stamp and task-map state untouched until that boundary is ready.
#[test]
fn translation_deferred_holds_sibling_unmap_head_and_stamp() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
    let page_size = state.page_size() as usize;
    let channel = 2u32;
    let producer_bit = 1u32 << 1;
    let sibling_bit = 1u32 << channel;
    let root_pfn = 0x10u32;
    let list_pfn = 0x20u32;
    let ring_pfn = 0x30u32;
    let stamp_pfn = 0x40u32;
    let root_gpa = state.pfn_gpa(root_pfn);
    let list_gpa = state.pfn_gpa(list_pfn);
    let ring_gpa = state.pfn_gpa(ring_pfn);
    let stamp_gpa = state.pfn_gpa(stamp_pfn);
    for gpa in [root_gpa, list_gpa, ring_gpa, stamp_gpa] {
        host.map_range(gpa, page_size, 0);
    }

    let task_id = 6u32;
    let gva = 0x101000u64;
    let length = 0x4000u64;
    state.note_task_map(task_id, gva, length);
    let mut payload = vec![0u8; 20];
    payload[0..4].copy_from_slice(&task_id.to_le_bytes());
    payload[4..12].copy_from_slice(&gva.to_le_bytes());
    payload[12..20].copy_from_slice(&length.to_le_bytes());
    let packet = packet_bytes(CHILD_OP_UNMAP_MEMORY, 0x55, &payload);
    host.write_gpa(ring_gpa, &packet).unwrap();
    host.put_u32(list_gpa, ring_pfn);

    let regs_gpa = root_gpa + child_reg_block_offset(channel).unwrap();
    host.put_u32(regs_gpa + CHILD_REG_TAIL, packet.len() as u32);
    host.put_u32(regs_gpa + CHILD_REG_HEAD, 0);
    host.put_u32(regs_gpa + CHILD_REG_STAMP_INDEX, 1);
    host.put_u32(regs_gpa + CHILD_REG_BASE_PFN, list_pfn);
    state.gfx.root_page = root_pfn;
    state.gfx.fifo_base_page = stamp_pfn;
    state.active_child_mask = producer_bit | sibling_bit;
    state.pending.child_mask = sibling_bit;
    state.translation_deferred_mask = producer_bit;

    drain_pending(&mut state, &mut host);
    drain_pending(&mut state, &mut host);
    assert_eq!(host.get_u32(regs_gpa + CHILD_REG_HEAD), 0);
    assert_eq!(host.get_u32(stamp_gpa + 4), 0);
    assert_eq!(state.task_map_spans.len(), 1);
    assert_eq!(state.translation_order_hold_mask, sibling_bit);
    assert_eq!(state.translation_order_holds, 1, "poll retries coalesce");

    note_translation_order_hold(&mut state, TRANSLATION_ROOT_FIFO_BIT);
    assert_eq!(
        state.translation_order_holds, 1,
        "new timeline bits in one ownership interval remain one episode"
    );

    // Simulate the immutable AIR worker becoming ready. The real producer
    // retry clears this bit in process_child_packet before siblings resume.
    state.translation_deferred_mask = 0;
    drain_pending(&mut state, &mut host);
    assert_eq!(host.get_u32(regs_gpa + CHILD_REG_HEAD), packet.len() as u32);
    assert_eq!(host.get_u32(stamp_gpa + 4), 0x55);
    assert!(state.task_map_spans.is_empty());
    assert_eq!(state.translation_order_hold_mask, 0);
}

/// FIFO redefine/free retires scheduler ownership so a removed producer
/// cannot strand later display transactions behind a stale bit.
#[test]
fn free_fifo_clears_translation_scheduler_state() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let bit = 1 << 1;
    state.active_child_mask = bit;
    state.pending.child_mask = bit;
    state.translation_deferred_mask = bit;
    state.translation_order_hold_mask = bit;
    state.present_translation_hold_mask = bit;

    process_root_packet(
        &mut state,
        &mut host,
        &Packet {
            opcode: ROOT_OP_FREE_FIFO,
            stamp_count: 0,
            total_size: PACKET_HEADER_LEN + 4,
            completion_stamp: 0,
            payload: 1u32.to_le_bytes().to_vec(),
            next_head: 0,
        },
    );

    assert_eq!(state.active_child_mask & bit, 0);
    assert_eq!(state.pending.child_mask & bit, 0);
    assert_eq!(state.translation_deferred_mask & bit, 0);
    assert_eq!(state.translation_order_hold_mask & bit, 0);
    assert_eq!(state.present_translation_hold_mask & bit, 0);
}

/// First Composite present takes the leave-BAR1 boundary.
#[test]
fn composite_present_sets_frame_flush_boundary() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    assert!(state.map_surface(4));
    {
        let m = state.mappings.get_mut(&4).unwrap();
        m.mapped = true;
        m.has_geom = true;
        m.width = 1920;
        m.height = 1080;
        m.content_generation = 2;
        m.page_entries = vec![1];
    }
    state.note_surface_composite(4);

    let mut payload = vec![0u8; PRESENT_X86_MIN_LEN.max(12)];
    payload[PRESENT_X86_SURFACE_ID..PRESENT_X86_SURFACE_ID + 4]
        .copy_from_slice(&4u32.to_le_bytes());
    let pkt = Packet {
        opcode: CHILD_OP_PRESENT_X86,
        stamp_count: 0,
        total_size: PACKET_HEADER_LEN + payload.len() as u32,
        completion_stamp: 0,
        payload,
        next_head: 0,
    };
    process_child_packet(&mut state, &mut host, 5, &pkt);
    assert!(state.present.frame_flush_seen);
    assert_coalesced_paint_action(&host, "composite sets flush boundary");
}

#[test]
fn display_swap_without_geom_holds_last_frame() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    state.present.valid = true;
    state.present.width = 1920;
    state.present.height = 1080;
    assert!(state.map_surface(9));
    // Mapped but no has_geom — do not resize/paint.
    let mut payload = vec![0u8; DISPLAY_SWAP_MIN_LEN];
    payload[DISPLAY_SWAP_MAPPING..DISPLAY_SWAP_MAPPING + 4].copy_from_slice(&9u32.to_le_bytes());
    let pkt = Packet {
        opcode: CHILD_OP_DISPLAY_SWAP,
        stamp_count: 0,
        total_size: PACKET_HEADER_LEN + DISPLAY_SWAP_MIN_LEN as u32,
        completion_stamp: 0,
        payload,
        next_head: 0,
    };
    process_child_packet(&mut state, &mut host, 4, &pkt);
    assert!(state.present.frame_flush_seen);
    assert_eq!(state.present.present_mapping, 9);
    // Console size unchanged; no scanout HostAction.
    assert_eq!(state.present.width, 1920);
    assert_eq!(state.present.height, 1080);
    assert!(host.actions.is_empty());
}

#[test]
fn map_surface_clears_stale_geom() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    assert!(state.map_surface(1));
    assert!(state.set_mapping_geom(1, 1920, 1080, 0x73));
    assert!(state.mappings[&1].has_geom);
    assert!(state.map_surface(1));
    let m = &state.mappings[&1];
    assert!(!m.has_geom);
    assert_eq!(m.width, 0);
    assert_eq!(m.height, 0);
}

/// qemu-shim: DisplaySwap present completion is with the packet stamp
/// after +0x188 retain (PGDisplay presentFrame completion block), not
/// deferred until host paint. waitForPendingFrames gates *entry*.
#[test]
fn display_swap_stamp_ready_after_present_retain_not_paint() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let host = FakeHost::new();
    // Sync stamp path: ready immediately, drains to write_stamp.
    let slot = StampSlot {
        stamp_index: 0,
        stamp_value: 42,
        ready: true,
        job_id: None,
        target_mapping: 0,
    };
    state.child_stamps[4].push(slot);
    let ready = state.child_stamps[4].drain_ready();
    assert_eq!(ready.len(), 1);
    assert_eq!(ready[0].stamp_value, 42);
    assert!(
        state.child_stamps[4].queue.is_empty(),
        "ready present stamp flushes without waiting for paint"
    );
    let _ = host;
}

/// x86 Ventura/Tahoe display pipe: present opcode 6 paints like DisplaySwap.
#[test]
fn present_x86_op6_paints_surface_id_mapping() {
    use crate::contract::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
    use crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    use crate::runtime::mapping_write::write_bgra8;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let pfn = 0x71u32;
    let gpa = (pfn as u64) << PAGE_SHIFT_ARM64E;
    host.map_range(gpa, 0x4000, 0);
    assert!(state.map_surface(5));
    {
        let m = state.mappings.get_mut(&5).unwrap();
        m.mapped = true;
        m.mapping_internal = 1;
        m.page_entries = vec![(pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
    }
    assert!(state.set_mapping_geom(5, 2, 2, MTL_FORMAT_BGRA8_UNORM));
    let px = [0x22u8; 16];
    assert!(write_bgra8(&mut state, &mut host, 5, &px, 8, 2, 2));

    let mut payload = vec![0u8; PRESENT_X86_MIN_LEN];
    payload[PRESENT_X86_SURFACE_ID..PRESENT_X86_SURFACE_ID + 4]
        .copy_from_slice(&5u32.to_le_bytes());
    process_child_packet(
        &mut state,
        &mut host,
        5,
        &Packet {
            opcode: CHILD_OP_PRESENT_X86,
            stamp_count: 0,
            total_size: PACKET_HEADER_LEN + PRESENT_X86_MIN_LEN as u32,
            completion_stamp: 0,
            payload,
            next_head: 0,
        },
    );
    assert_eq!(state.present.present_mapping, 5);
    assert!(state.present.frame_flush_seen);
    assert!(state.present.frame_valid || state.present.frame_encode_pending);
    assert!(
        state.present.frame_valid || state.present.frame_encode_pending,
        "op6 present hands the window a frame (or defers to encode)"
    );
    assert_coalesced_paint_action(&host, "x86 op6 present");
}

/// qemu-shim: each accepted DisplaySwap with geom increments unpainted
/// presents; host paint clears the counter (entry-side backpressure).
#[test]
fn display_swap_unpainted_presents_counts_until_paint() {
    use crate::contract::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
    use crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    use crate::runtime::mapping_write::write_bgra8;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let pfn = 0x70u32;
    let gpa = (pfn as u64) << PAGE_SHIFT_ARM64E;
    host.map_range(gpa, 0x4000, 0);
    assert!(state.map_surface(3));
    {
        let m = state.mappings.get_mut(&3).unwrap();
        m.mapped = true;
        m.mapping_internal = 1;
        m.page_entries = vec![(pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
    }
    assert!(state.set_mapping_geom(3, 2, 2, MTL_FORMAT_BGRA8_UNORM));
    let px = [0x11u8; 16];
    assert!(write_bgra8(&mut state, &mut host, 3, &px, 8, 2, 2));

    let swap = |state: &mut DeviceState, host: &mut FakeHost| {
        let mut payload = vec![0u8; DISPLAY_SWAP_MIN_LEN];
        payload[DISPLAY_SWAP_MAPPING..DISPLAY_SWAP_MAPPING + 4]
            .copy_from_slice(&3u32.to_le_bytes());
        process_child_packet(
            state,
            host,
            4,
            &Packet {
                opcode: CHILD_OP_DISPLAY_SWAP,
                stamp_count: 0,
                total_size: PACKET_HEADER_LEN + DISPLAY_SWAP_MIN_LEN as u32,
                completion_stamp: 0,
                payload,
                next_head: 0,
            },
        );
    };

    assert_eq!(state.present.unpainted_presents, 0);
    swap(&mut state, &mut host);
    assert_eq!(
        state.present.unpainted_presents, 1,
        "first accepted DisplaySwap counts as unpainted"
    );
    swap(&mut state, &mut host);
    assert_eq!(
        state.present.unpainted_presents, 2,
        "second accepted DisplaySwap reaches apple-gfx pending_frames cap"
    );
    // process_child_packet itself does not gate — drain_child_fifo does.
    // Counter keeps climbing if tests call process directly (stamp still fires).
    swap(&mut state, &mut host);
    assert_eq!(state.present.unpainted_presents, 3);
    note_present_paint_consumed(&mut state);
    assert_eq!(
        state.present.unpainted_presents, 0,
        "host paint clears entry-side present backpressure"
    );
    // Gate predicate used by drain_child_fifo.
    assert!(
        state.present.unpainted_presents < MAX_UNPAINTED_PRESENTS,
        "after paint, DisplaySwap entry is open"
    );
}

/// PVG present completion: every accepted DisplaySwap sets pending bit 1
/// on the display shared page and pokes the display IRQ when the guest
/// enable mask asks for present notifications (completion block after
/// +0x188 retain). ONLINE pending (bit 2) must be preserved (guest
/// read-clears the word).
#[test]
fn display_swap_signals_present_complete_on_shared_page() {
    use crate::contract::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
    use crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let pfn = 0x70u32;
    let gpa = (pfn as u64) << PAGE_SHIFT_ARM64E;
    host.map_range(gpa, 0x4000, 0);
    assert!(state.map_surface(3));
    {
        let m = state.mappings.get_mut(&3).unwrap();
        m.mapped = true;
        m.mapping_internal = 1;
        m.page_entries = vec![(pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
    }
    assert!(state.set_mapping_geom(3, 2, 2, MTL_FORMAT_BGRA8_UNORM));

    // Display shared page with enable mask asking for present events;
    // a stale ONLINE pending bit must survive the present OR.
    let shared = 0x9000_0000u64;
    host.map_range(shared, 0x1000, 0);
    state.display.shared_gpa = shared;
    state.display.display_index = 0;
    host.put_u32(
        shared + DISPLAY_SHARED_ENABLE_MASK,
        DISPLAY_PRESENT_EVENT_MASK | DISPLAY_ONLINE_EVENT_MASK,
    );
    host.put_u32(shared + DISPLAY_SHARED_PENDING, DISPLAY_ONLINE_EVENT_MASK);

    let mut payload = vec![0u8; DISPLAY_SWAP_MIN_LEN];
    payload[DISPLAY_SWAP_MAPPING..DISPLAY_SWAP_MAPPING + 4].copy_from_slice(&3u32.to_le_bytes());
    process_child_packet(
        &mut state,
        &mut host,
        4,
        &Packet {
            opcode: CHILD_OP_DISPLAY_SWAP,
            stamp_count: 0,
            total_size: PACKET_HEADER_LEN + DISPLAY_SWAP_MIN_LEN as u32,
            completion_stamp: 0,
            payload,
            next_head: 0,
        },
    );

    let mut le = [0u8; 4];
    assert!(host
        .read_gpa(shared + DISPLAY_SHARED_PENDING, &mut le)
        .is_ok());
    let pending = u32::from_le_bytes(le);
    assert_ne!(
        pending & DISPLAY_PRESENT_EVENT_MASK,
        0,
        "present completion must set pending bit 1"
    );
    assert_ne!(
        pending & DISPLAY_ONLINE_EVENT_MASK,
        0,
        "present completion must not clobber other pending events"
    );
    assert_ne!(
        state
            .gfx
            .interrupt_status_disp
            .load(std::sync::atomic::Ordering::Acquire)
            & 1,
        0,
        "display IRQ status must name display 0"
    );

    // Enable mask without the present bit: pending still set, no IRQ.
    state
        .gfx
        .interrupt_status_disp
        .store(0, std::sync::atomic::Ordering::Release);
    host.put_u32(shared + DISPLAY_SHARED_ENABLE_MASK, 0);
    host.put_u32(shared + DISPLAY_SHARED_PENDING, 0);
    signal_display_present_complete(&mut state, &mut host);
    assert!(host
        .read_gpa(shared + DISPLAY_SHARED_PENDING, &mut le)
        .is_ok());
    assert_ne!(u32::from_le_bytes(le) & DISPLAY_PRESENT_EVENT_MASK, 0);
    assert_eq!(
        state
            .gfx
            .interrupt_status_disp
            .load(std::sync::atomic::Ordering::Acquire),
        0,
        "no display IRQ when the guest did not ask for present events"
    );
}

/// qemu-shim: entry gate holds when unpainted_presents >= MAX (apple-gfx
/// pending_frames >= 2). Stamp of accepted presents remains at retain.
#[test]
fn display_swap_entry_gated_when_unpainted_at_cap() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    state.present.unpainted_presents = MAX_UNPAINTED_PRESENTS;
    assert!(
        state.present.unpainted_presents >= MAX_UNPAINTED_PRESENTS,
        "drain_child_fifo must hold DisplaySwap when unpainted at cap"
    );
    note_present_paint_consumed(&mut state);
    assert!(
        state.present.unpainted_presents < MAX_UNPAINTED_PRESENTS,
        "paint re-opens DisplaySwap entry"
    );
}

#[test]
fn present_action_starvation_proxy_is_once_per_held_head() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    state.present.unpainted_presents = MAX_UNPAINTED_PRESENTS;

    note_present_backpressure_hold(&mut state, 5, 464, 592);
    note_present_backpressure_hold(&mut state, 5, 464, 592);
    assert_eq!(state.present.backpressure_hold_count, 1);

    note_present_paint_consumed(&mut state);
    state.present.unpainted_presents = MAX_UNPAINTED_PRESENTS;
    note_present_backpressure_hold(&mut state, 5, 464, 592);
    assert_eq!(
        state.present.backpressure_hold_count, 2,
        "a later hold after paint is a distinct starvation episode"
    );
}

#[test]
fn child_drain_yields_after_present_for_display_consumer() {
    use crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
    let page_size = state.page_size() as usize;
    let channel = 5u32;
    let root_pfn = 0x10u32;
    let list_pfn = 0x20u32;
    let ring_pfn = 0x30u32;
    let stamp_pfn = 0x40u32;
    let root_gpa = state.pfn_gpa(root_pfn);
    let list_gpa = state.pfn_gpa(list_pfn);
    let ring_gpa = state.pfn_gpa(ring_pfn);
    let stamp_gpa = state.pfn_gpa(stamp_pfn);
    for gpa in [root_gpa, list_gpa, ring_gpa, stamp_gpa] {
        host.map_range(gpa, page_size, 0);
    }

    assert!(state.map_surface(4));
    assert!(state.set_mapping_geom(4, 2, 2, MTL_FORMAT_BGRA8_UNORM));

    let mut payload = vec![0u8; PRESENT_X86_MIN_LEN];
    payload[PRESENT_X86_SURFACE_ID..PRESENT_X86_SURFACE_ID + 4]
        .copy_from_slice(&4u32.to_le_bytes());
    let first = packet_bytes(CHILD_OP_PRESENT_X86, 21, &payload);
    let second = packet_bytes(CHILD_OP_PRESENT_X86, 22, &payload);
    let mut ring = first.clone();
    ring.extend_from_slice(&second);
    host.write_gpa(ring_gpa, &ring).unwrap();
    host.put_u32(list_gpa, ring_pfn);

    let regs_gpa = root_gpa + child_reg_block_offset(channel).unwrap();
    host.put_u32(regs_gpa + CHILD_REG_TAIL, ring.len() as u32);
    host.put_u32(regs_gpa + CHILD_REG_HEAD, 0);
    host.put_u32(regs_gpa + CHILD_REG_STAMP_INDEX, 1);
    host.put_u32(regs_gpa + CHILD_REG_BASE_PFN, list_pfn);
    state.gfx.root_page = root_pfn;
    state.gfx.fifo_base_page = stamp_pfn;
    state.active_child_mask = 1u32 << channel;
    state.pending.child_mask = 1u32 << channel;

    drain_pending(&mut state, &mut host);
    assert_eq!(
        host.get_u32(regs_gpa + CHILD_REG_HEAD),
        first.len() as u32,
        "the first drain slice must stop after accepting one present"
    );
    assert_ne!(state.pending.child_mask & (1u32 << channel), 0);
    assert_eq!(state.present.unpainted_presents, 1);
    assert!(
        state.pending.host_action_yield,
        "an accepted present must end the drain slice"
    );
    assert_coalesced_paint_action(&host, "first present");

    // The ack. `device_drain` calls this itself after publishing the frame to
    // the host window; it used to arrive from QEMU's DisplaySurface paint
    // after the lock was released. Either way it reopens the queued channel
    // for its next ordered packet — that contract is what this test locks.
    note_present_paint_consumed(&mut state);
    host.actions.clear();
    drain_pending(&mut state, &mut host);
    assert_eq!(host.get_u32(regs_gpa + CHILD_REG_HEAD), ring.len() as u32);
    assert_eq!(state.present.unpainted_presents, 1);
    assert_coalesced_paint_action(&host, "second present after ack");
    assert_eq!(host.get_u32(stamp_gpa + 4), 22);
}

/// Mode switch (1920→1440) is a new surface identity: reset
/// content_generation (Load/scanout semantics restart).
#[test]
fn set_mapping_geom_size_change_resets_content_generation() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    assert!(state.map_surface(4));
    assert!(state.set_mapping_geom(4, 1920, 1080, 0x73));
    {
        let m = state.mappings.get_mut(&4).unwrap();
        m.content_generation = 42;
    }
    assert!(state.set_mapping_geom(4, 1440, 1080, 0x73));
    let m = &state.mappings[&4];
    assert_eq!(m.width, 1440);
    assert_eq!(m.height, 1080);
    assert_eq!(
        m.content_generation, 0,
        "new size must not keep prior gen (new surface identity)"
    );
    // Same size again: preserve gen (no identity change).
    {
        let m = state.mappings.get_mut(&4).unwrap();
        m.content_generation = 3;
    }
    assert!(state.set_mapping_geom(4, 1440, 1080, 0x50));
    assert_eq!(
        state.mappings[&4].content_generation, 3,
        "same size preserves generation"
    );
}

/// Archive poll_tick / render_wait_surface helpers: no rings → no-op, no panic.
#[test]
fn drain_other_and_stranded_are_safe_noop_without_rings() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    state.active_child_mask = (1 << 1) | (1 << 4);
    state.pending.child_mask = 1 << 1;
    state.gfx.control_fifo = 1;
    // No root_page / rings: drains return immediately.
    drain_other_child_fifos(&mut state, &mut host, 4);
    drain_stranded_fifos(&mut state, &mut host);
    assert_eq!(
        state.pending.child_mask, 0,
        "stranded drain clears pending mask"
    );
}

#[test]
fn poll_rescue_only_publishes_work_for_async_drain() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    state.gfx.control_fifo = 0x1000;
    state
        .gfx
        .fifo_read
        .store(3, std::sync::atomic::Ordering::Release);
    state.gfx.fifo_written = 4;
    state.active_child_mask = (1 << 2) | (1 << 5);

    assert!(publish_stranded_fifos(&mut state, &mut host));
    assert!(state.pending.main_drain);
    assert_eq!(state.pending.child_mask, (1 << 2) | (1 << 5));
    assert!(host.bh_scheduled);
    assert_eq!(
        state
            .gfx
            .fifo_read
            .load(std::sync::atomic::Ordering::Acquire),
        3,
        "poll context must not drain"
    );
}

/// Archive render_wait_surface: no inflight async job for mapping ⇒ no-op,
/// returns current content_generation. Does not drain other FIFOs.
#[test]
fn wait_surface_noop_when_no_async_job() {
    use crate::contract::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
    use crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    use crate::runtime::mapping_write::write_bgra8;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let pfn = 0x22u32;
    host.map_range((pfn as u64) << PAGE_SHIFT_ARM64E, 0x4000, 0);
    assert!(state.map_surface(7));
    {
        let m = state.mappings.get_mut(&7).unwrap();
        m.mapped = true;
        m.mapping_internal = 1;
        m.page_entries = vec![(pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
    }
    assert!(state.set_mapping_geom(7, 2, 2, MTL_FORMAT_BGRA8_UNORM));
    assert!(write_bgra8(
        &mut state,
        &mut host,
        7,
        &[0x55u8; 16],
        8,
        2,
        2
    ));
    let gen = state.mappings.get(&7).unwrap().content_generation;
    state.active_child_mask = (1 << 1) | (1 << 4);
    let out = wait_surface_other_channels(&mut state, &mut host, 4, 7);
    assert_eq!(out, gen, "no async job ⇒ return current gen");
    assert_eq!(wait_surface_mapping(&mut state, &mut host, 0), 0);
}

/// qemu-shim e2e: surface_inflight sees async job target_mapping until
/// complete_async_job; wait_surface after complete is quiet.
#[test]
fn wait_surface_surface_inflight_tracks_async_target_mapping() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let job = enqueue_async_stamp_surface(&mut state, 1, 0, 99, 42).expect("job");
    assert!(
        surface_inflight(&state, 42),
        "not-ready job with target_mapping must be inflight"
    );
    assert!(
        !surface_inflight(&state, 7),
        "other mapping is not inflight"
    );
    // After complete, slot is ready and drained — no longer inflight.
    complete_async_job(&mut state, &mut host, 1, job);
    assert!(!surface_inflight(&state, 42));
    assert_eq!(wait_surface_mapping(&mut state, &mut host, 42), 0);
}

/// Sample/Load path shares the same wait (archive one function).
#[test]
fn wait_surface_snapshot_once_matches_mapping_wait() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    state.draining_channel = 4;
    state.draining_mask = 1 << 4;
    wait_surface_snapshot_once(&mut state, &mut host, 3);
    assert_eq!(state.draining_channel, 4);
    assert_eq!(state.draining_mask, 1 << 4);
}

/// qemu-shim dual-mid: incomplete last_store on one mid (logo/partial)
/// must fire thrash `nz_swing` when DisplaySwap alternates full vs sparse.
/// Regression gate for P1 dual-mid flicker (measure before fix).
#[test]
/// Contig write_bgra8 + thrash proxy: full→sparse mid hop fires nz_swing.
///
/// DisplaySwap sticky dual-mid policy may retain one mid (mode=refresh) so
/// this locks the thrash class via present_proxy samples after real
/// type-11 contig Stores (not invent nz).
fn contig_store_dual_mid_incomplete_fires_nz_swing_proxy() {
    use crate::contract::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
    use crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    use crate::runtime::census::present_proxy::{self, counters, PresentCaptureSample};
    use crate::runtime::mapping_write::write_bgra8;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let w = 400u32;
    let h = 300u32;
    let stride = w * 4;
    let need = (stride * h) as usize;
    let pfn_a = 0x50u32;
    let pfn_b = 0x90u32; // 32 pages from 0x50 → 0x70; 0x90 is clear
    let pages_needed = 32usize;
    let page = 1usize << PAGE_SHIFT_ARM64E as usize;
    for (mid, base_pfn) in [(3u32, pfn_a), (4u32, pfn_b)] {
        let base_gpa = (base_pfn as u64) << PAGE_SHIFT_ARM64E;
        host.map_range(
            base_gpa,
            pages_needed * page,
            if mid == 3 { 0xEE } else { 0x00 },
        );
        let mut entries = Vec::new();
        for i in 0..pages_needed {
            let pfn = base_pfn + i as u32;
            entries.push((pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID);
        }
        assert!(state.map_surface(mid));
        {
            let m = state.mappings.get_mut(&mid).unwrap();
            m.mapped = true;
            m.mapping_internal = mid as u64;
            m.page_entries = entries;
        }
        assert!(state.set_mapping_geom(mid, w, h, MTL_FORMAT_BGRA8_UNORM));
    }
    let full = vec![0xCCu8; need];
    assert!(write_bgra8(&mut state, &mut host, 3, &full, stride, w, h));
    let mut sparse = vec![0u8; need];
    for b in sparse.iter_mut().take(need / 20) {
        *b = 0x80;
    }
    assert!(write_bgra8(&mut state, &mut host, 4, &sparse, stride, w, h));

    // Guest pages hold full vs sparse (contig alias); sample like capture.
    let mut px0 = [0u8; 4];
    assert!(host
        .read_gpa((pfn_a as u64) << PAGE_SHIFT_ARM64E, &mut px0)
        .is_ok());
    assert_ne!(
        px0,
        [0, 0, 0, 0],
        "contig write_bgra8 must land in guest RAM"
    );

    let _proxy = present_proxy::test_exclusive();
    present_proxy::reset_for_test();
    let full_rgb = (w as usize) * (h as usize); // every px non-zero
    let sparse_rgb = need / 20 / 4; // approx non-zero pixels in sparse fill
    present_proxy::note_capture_ok(PresentCaptureSample {
        mapping_id: 3,
        generation: 1,
        width: w,
        height: h,
        nz: need,
        rgb_nz: full_rgb,
        max_byte: 0xCC,
        max_rgb: 0xCC,
        from_last_store: true,
        edge_energy: 1000,
        named_peer: false,
    });
    present_proxy::note_capture_ok(PresentCaptureSample {
        mapping_id: 4,
        generation: 1,
        width: w,
        height: h,
        nz: need / 20,
        rgb_nz: sparse_rgb.max(1),
        max_byte: 0x80,
        max_rgb: 0x80,
        from_last_store: true,
        edge_energy: 50,
        named_peer: false,
    });
    let c = counters();
    assert!(
        c.nz_swings >= 1,
        "full→sparse dual-mid must count nz_swing (got {c:?})"
    );
    assert!(c.mid_switches >= 1);
}

/// qemu-shim DisplaySwap retains surface at present (after wait_surface
/// drains); HostAction paints the frozen snapshot (hostPresentCount).
#[test]
fn display_swap_encodes_at_present_after_wait_surface() {
    use crate::contract::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
    use crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    use crate::runtime::mapping_write::write_bgra8;
    use crate::runtime::scanout::{copy_to_bgra8, ScanoutCopyResult};

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let pfn = 0x21u32;
    let gpa = (pfn as u64) << PAGE_SHIFT_ARM64E;
    host.map_range(gpa, 0x4000, 0);
    let entry = (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
    assert!(state.map_surface(3));
    {
        let m = state.mappings.get_mut(&3).unwrap();
        m.mapped = true;
        m.mapping_internal = 1;
        m.page_entries = vec![entry];
    }
    assert!(state.set_mapping_geom(3, 2, 2, MTL_FORMAT_BGRA8_UNORM));
    let px = [
        0x11u8, 0x22, 0x33, 0xFF, 0x11, 0x22, 0x33, 0xFF, 0x11, 0x22, 0x33, 0xFF, 0x11, 0x22, 0x33,
        0xFF,
    ];
    assert!(write_bgra8(&mut state, &mut host, 3, &px, 8, 2, 2));
    let gen = state.mappings.get(&3).unwrap().content_generation;
    let mut payload = vec![0u8; DISPLAY_SWAP_MIN_LEN];
    payload[DISPLAY_SWAP_MAPPING..DISPLAY_SWAP_MAPPING + 4].copy_from_slice(&3u32.to_le_bytes());
    let pkt = Packet {
        opcode: CHILD_OP_DISPLAY_SWAP,
        stamp_count: 0,
        total_size: PACKET_HEADER_LEN + DISPLAY_SWAP_MIN_LEN as u32,
        completion_stamp: 0,
        payload,
        next_head: 0,
    };
    process_child_packet(&mut state, &mut host, 4, &pkt);
    assert!(state.present.frame_flush_seen);
    assert!(
        state.present.frame_valid,
        "DisplaySwap freezes surface at present after wait_surface"
    );
    // Capture forces one host blit of +0x188 (encode_pending) so early
    // painted mid/gen cannot Unchanged-skip logo/pill onto frozen EFI.
    assert!(
        state.present.frame_encode_pending,
        "successful capture must force first paint of retain"
    );
    assert_eq!(state.present.frame_mapping, 3);
    assert_eq!(presented_mapping(&state), Some(3));
    assert_coalesced_paint_action(&host, "encode at present");
    // Host paint re-shows frozen snapshot.
    let mut dst = vec![0u8; 16];
    assert_eq!(
        copy_to_bgra8(&mut state, &mut host, 3, &mut dst, 8, 2, 2, gen),
        ScanoutCopyResult::Painted
    );
    assert_eq!(&dst[..], &px[..]);
    assert!(!state.present.frame_encode_pending);

    // Guest mutates mapping after stamp (recycle) — re-show still frozen.
    let mut_px = [0xAAu8; 16];
    assert!(write_bgra8(&mut state, &mut host, 3, &mut_px, 8, 2, 2));
    state.present.painted_generation = 0;
    assert_eq!(
        copy_to_bgra8(&mut state, &mut host, 3, &mut dst, 8, 2, 2, gen),
        ScanoutCopyResult::Painted
    );
    assert_eq!(
        &dst[..],
        &px[..],
        "post-stamp guest writes must not change retained present frame"
    );
}

/// qemu-shim: DisplaySwap capture fail must not drop PGDisplay +0x188 retain.
/// hostPresentCount re-shows the last successful present until capture works.
#[test]
fn display_swap_capture_fail_keeps_prior_retain() {
    use crate::contract::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
    use crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    use crate::runtime::mapping_write::write_bgra8;
    use crate::runtime::scanout::{copy_to_bgra8, ScanoutCopyResult};

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let pfn = 0x40u32;
    let gpa = (pfn as u64) << PAGE_SHIFT_ARM64E;
    host.map_range(gpa, 0x4000, 0);
    assert!(state.map_surface(5));
    {
        let m = state.mappings.get_mut(&5).unwrap();
        m.mapped = true;
        m.mapping_internal = 1;
        m.page_entries = vec![(pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
    }
    assert!(state.set_mapping_geom(5, 2, 2, MTL_FORMAT_BGRA8_UNORM));
    let full = [
        0x11u8, 0x22, 0x33, 0xFF, 0x44, 0x55, 0x66, 0xFF, 0xAA, 0, 0, 0xFF, 0xAA, 0, 0, 0xFF,
    ];
    assert!(write_bgra8(&mut state, &mut host, 5, &full, 8, 2, 2));

    let swap = |state: &mut DeviceState, host: &mut FakeHost, mid: u32| {
        host.actions.clear();
        let mut payload = vec![0u8; DISPLAY_SWAP_MIN_LEN];
        payload[DISPLAY_SWAP_MAPPING..DISPLAY_SWAP_MAPPING + 4].copy_from_slice(&mid.to_le_bytes());
        process_child_packet(
            state,
            host,
            4,
            &Packet {
                opcode: CHILD_OP_DISPLAY_SWAP,
                stamp_count: 0,
                total_size: PACKET_HEADER_LEN + DISPLAY_SWAP_MIN_LEN as u32,
                completion_stamp: 0,
                payload,
                next_head: 0,
            },
        );
    };

    // First swap: full dock composite retained.
    swap(&mut state, &mut host, 5);
    assert!(state.present.frame_valid);
    assert_eq!(state.present.frame_mapping, 5);
    let gen_ok = state.present.frame_generation;
    let mut dst = vec![0u8; 16];
    assert_eq!(
        copy_to_bgra8(&mut state, &mut host, 5, &mut dst, 8, 2, 2, gen_ok),
        ScanoutCopyResult::Painted
    );
    assert_eq!(&dst[..], &full[..]);

    // Second swap: pages unreadable + host-cache gone → capture fails.
    {
        let m = state.mappings.get_mut(&5).unwrap();
        m.page_entries.clear();
        // Bump gen so HostAction is distinct; guest would still name mid 5.
        m.content_generation = gen_ok + 1;
    }
    crate::runtime::surface_cache::evict(&mut state, 5);
    swap(&mut state, &mut host, 5);
    assert!(
        state.present.frame_encode_pending,
        "capture fail must set pending retry"
    );
    assert!(
        state.present.frame_valid,
        "PGDisplay +0x188 prior retain must survive capture fail"
    );
    assert_eq!(
        state.present.frame_mapping, 5,
        "prior retain mapping unchanged"
    );
    assert_eq!(
        presented_mapping(&state),
        Some(5),
        "window still shows the prior retain after a capture fail"
    );
    assert_coalesced_paint_action(&host, "capture fail keeps prior retain");
    // hostPresentCount / HostAction still shows the last good full composite.
    state.present.painted_generation = 0;
    assert_eq!(
        copy_to_bgra8(&mut state, &mut host, 5, &mut dst, 8, 2, 2, gen_ok + 1),
        ScanoutCopyResult::Painted
    );
    assert_eq!(
        &dst[..],
        &full[..],
        "capture-fail DisplaySwap must re-show prior full retain, not black/empty"
    );
}

/// qemu-shim dual-mid: each CmdDisplaySwap freezes that mid's **full** guest
/// composite (dock strip pattern); hostPresentCount re-shows the latest
/// retain only — never mixes mid A partial with mid B full.
#[test]
fn display_swap_dual_mid_full_composites_both_retain() {
    use crate::contract::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
    use crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    use crate::runtime::mapping_write::write_bgra8;
    use crate::runtime::scanout::{copy_to_bgra8, ScanoutCopyResult};

    // 4×2 BGRA: row0 = "dock" strip (distinct L/R icons), row1 = wallpaper.
    fn frame(left: [u8; 4], right: [u8; 4], wall: [u8; 4]) -> Vec<u8> {
        let mut v = Vec::with_capacity(32);
        v.extend_from_slice(&left);
        v.extend_from_slice(&right);
        v.extend_from_slice(&wall);
        v.extend_from_slice(&wall);
        v
    }
    let full_a = frame(
        [0x11, 0x22, 0x33, 0xFF],
        [0x44, 0x55, 0x66, 0xFF],
        [0xAA, 0x00, 0x00, 0xFF],
    );
    let full_b = frame(
        [0x77, 0x88, 0x99, 0xFF],
        [0xBB, 0xCC, 0xDD, 0xFF],
        [0x00, 0xAA, 0x00, 0xFF],
    );
    // Partial dock: left icons only, right = wallpaper (residual as-t4 shape).
    let partial_b = frame(
        [0x77, 0x88, 0x99, 0xFF],
        [0x00, 0xAA, 0x00, 0xFF],
        [0x00, 0xAA, 0x00, 0xFF],
    );

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    for (mid, pfn) in [(3u32, 0x30u32), (4u32, 0x31u32)] {
        let gpa = (pfn as u64) << PAGE_SHIFT_ARM64E;
        host.map_range(gpa, 0x4000, 0);
        assert!(state.map_surface(mid));
        {
            let m = state.mappings.get_mut(&mid).unwrap();
            m.mapped = true;
            m.mapping_internal = 1;
            m.page_entries = vec![(pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
        }
        assert!(state.set_mapping_geom(mid, 2, 2, MTL_FORMAT_BGRA8_UNORM));
    }
    assert!(write_bgra8(&mut state, &mut host, 3, &full_a, 8, 2, 2));
    assert!(write_bgra8(&mut state, &mut host, 4, &partial_b, 8, 2, 2));

    let swap = |state: &mut DeviceState, host: &mut FakeHost, mid: u32| {
        host.actions.clear();
        let mut payload = vec![0u8; DISPLAY_SWAP_MIN_LEN];
        payload[DISPLAY_SWAP_MAPPING..DISPLAY_SWAP_MAPPING + 4].copy_from_slice(&mid.to_le_bytes());
        process_child_packet(
            state,
            host,
            4,
            &Packet {
                opcode: CHILD_OP_DISPLAY_SWAP,
                stamp_count: 0,
                total_size: PACKET_HEADER_LEN + DISPLAY_SWAP_MIN_LEN as u32,
                completion_stamp: 0,
                payload,
                next_head: 0,
            },
        );
    };

    // Present mid3 full dock → +0x188.
    swap(&mut state, &mut host, 3);
    assert!(state.present.frame_valid);
    assert_eq!(state.present.frame_mapping, 3);
    let mut dst = vec![0u8; 16];
    let gen3 = state.present.frame_generation;
    assert_eq!(
        copy_to_bgra8(&mut state, &mut host, 3, &mut dst, 8, 2, 2, gen3),
        ScanoutCopyResult::Painted
    );
    assert_eq!(
        &dst[..],
        &full_a[..],
        "mid3 present freezes full dock into +0x188"
    );

    // Present mid4 while guest still has partial dock on mid4 (overwrites +0x188).
    swap(&mut state, &mut host, 4);
    assert_eq!(
        state.present.frame_mapping, 4,
        "DisplaySwap must re-retain mid4"
    );
    let gen4p = state.present.frame_generation;
    state.present.painted_generation = 0; // force paint (same gen as mid3 possible)
    assert_eq!(
        copy_to_bgra8(&mut state, &mut host, 4, &mut dst, 8, 2, 2, gen4p),
        ScanoutCopyResult::Painted
    );
    assert_eq!(
        &dst[..],
        &partial_b[..],
        "mid4 present shows guest partial until full composite lands"
    );
    // Late HostAction for mid3: encodeCurrentFrame shows current +0x188
    // (mid4 partial), not a mid3 backlog and not live mid3 if recycled.
    state.present.painted_generation = 0;
    assert_eq!(
        copy_to_bgra8(&mut state, &mut host, 3, &mut dst, 8, 2, 2, gen3),
        ScanoutCopyResult::Painted
    );
    assert_eq!(
        &dst[..],
        &partial_b[..],
        "late mid3 HostAction paints current +0x188 (mid4)"
    );

    // Guest finishes full dock on mid4; DisplaySwap freezes full composite.
    assert!(write_bgra8(&mut state, &mut host, 4, &full_b, 8, 2, 2));
    swap(&mut state, &mut host, 4);
    let gen4 = state.present.frame_generation;
    assert_eq!(
        copy_to_bgra8(&mut state, &mut host, 4, &mut dst, 8, 2, 2, gen4),
        ScanoutCopyResult::Painted
    );
    assert_eq!(
        &dst[..],
        &full_b[..],
        "mid4 after full composite: both dock L/R present"
    );
    // Live mid3 rewrite must not affect +0x188 hostPresentCount re-show.
    assert!(write_bgra8(&mut state, &mut host, 3, &full_a, 8, 2, 2));
    state.present.painted_generation = 0;
    assert_eq!(
        copy_to_bgra8(&mut state, &mut host, 3, &mut dst, 8, 2, 2, gen3),
        ScanoutCopyResult::Painted
    );
    assert_eq!(
        &dst[..],
        &full_b[..],
        "+0x188 still mid4 full after live mid3 page writes"
    );
}

/// Double-buffer present: alternating DisplaySwap mapping ids each paint
/// the named surface (guest mid3/mid4). Both composites land independently.
#[test]
fn display_swap_alternating_mappings_both_paint() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    for mid in [3u32, 4u32] {
        assert!(state.map_surface(mid));
        let m = state.mappings.get_mut(&mid).unwrap();
        m.mapped = true;
        m.has_geom = true;
        m.width = 1440;
        m.height = 1080;
        m.content_generation = mid * 10;
        m.page_entries = vec![1];
    }
    for mid in [3u32, 4u32, 3u32] {
        host.actions.clear();
        let mut payload = vec![0u8; DISPLAY_SWAP_MIN_LEN];
        payload[DISPLAY_SWAP_MAPPING..DISPLAY_SWAP_MAPPING + 4].copy_from_slice(&mid.to_le_bytes());
        let pkt = Packet {
            opcode: CHILD_OP_DISPLAY_SWAP,
            stamp_count: 0,
            total_size: PACKET_HEADER_LEN + DISPLAY_SWAP_MIN_LEN as u32,
            completion_stamp: 0,
            payload,
            next_head: 0,
        };
        process_child_packet(&mut state, &mut host, 4, &pkt);
        assert!(state.present.frame_flush_seen);
        assert_eq!(state.present.present_mapping, mid);
        assert_eq!(state.present.width, 1440);
        assert_eq!(state.present.height, 1080);
        assert_coalesced_paint_action(&host, "alternating mappings");
    }
}

/// qemu-shim present contract: only CmdDisplaySwap (ch4 op8) paints after
/// the first frame boundary — writebacks and ch2 present-into-mid must not.
#[test]
fn only_display_swap_paints_after_frame_flush_seen() {
    use crate::contract::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
    use crate::contract::pixel_format::MTL_FORMAT_RGBA16_FLOAT;
    use crate::runtime::scanout::note_front_buffer_writeback;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    state.present.frame_flush_seen = true;
    state.present.valid = true;
    state.present.width = 1440;
    state.present.height = 1080;
    state.present.present_mapping = 3;
    state.present.host_mapping = 3;
    // Back buffer mid=4 writeback (compositor composite into non-front).
    assert!(state.map_surface(4));
    {
        let m = state.mappings.get_mut(&4).unwrap();
        m.mapped = true;
        m.has_geom = true;
        m.width = 1440;
        m.height = 1080;
        m.format = MTL_FORMAT_RGBA16_FLOAT;
        m.content_generation = 9;
        m.page_entries = vec![(1u32 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
    }
    note_front_buffer_writeback(
        &mut state,
        &mut host,
        4,
        1440,
        1080,
        MTL_FORMAT_RGBA16_FLOAT,
    );
    assert!(
        host.actions.is_empty(),
        "post-boundary writeback must not paint"
    );
    assert_eq!(
        state.present.present_mapping, 3,
        "writeback must not rename presented mid after DisplaySwap"
    );

    // DisplaySwap with geom → paint named mapping.
    assert!(state.map_surface(5));
    {
        let m = state.mappings.get_mut(&5).unwrap();
        m.mapped = true;
        m.has_geom = true;
        m.width = 1440;
        m.height = 1080;
        m.content_generation = 10;
        m.page_entries = vec![(2u32 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
    }
    let mut payload = vec![0u8; DISPLAY_SWAP_MIN_LEN];
    payload[DISPLAY_SWAP_MAPPING..DISPLAY_SWAP_MAPPING + 4].copy_from_slice(&5u32.to_le_bytes());
    let pkt = Packet {
        opcode: CHILD_OP_DISPLAY_SWAP,
        stamp_count: 0,
        total_size: PACKET_HEADER_LEN + DISPLAY_SWAP_MIN_LEN as u32,
        completion_stamp: 0,
        payload,
        next_head: 0,
    };
    process_child_packet(&mut state, &mut host, 4, &pkt);
    assert_coalesced_paint_action(&host, "post-flush display swap");
    assert_eq!(state.present.present_mapping, 5);
}

#[test]
fn display_online_waits_for_enable_mask_then_signals() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let gpa = 0x7b000000u64;
    host.map_range(gpa, PAGE_SIZE_ARM64E as usize, 0);
    state.display.shared_gpa = gpa;
    state.display.display_index = 0;
    // No enable mask yet — even after divisor ticks, no IRQ.
    state.display.poll_ctr = DISPLAY_ONLINE_POLL_DIVISOR - 1;
    try_display_online(&mut state, &mut host);
    assert!(host.actions.is_empty());
    assert_eq!(state.display.online_tries, 0);
    // Guest enable() published bit 2.
    let mut m = [0u8; 4];
    st32(&mut m, DISPLAY_ONLINE_EVENT_MASK);
    host.write_gpa(gpa + DISPLAY_SHARED_ENABLE_MASK, &m)
        .unwrap();
    state.display.poll_ctr = DISPLAY_ONLINE_POLL_DIVISOR - 1;
    try_display_online(&mut state, &mut host);
    assert_eq!(host.actions.len(), 1);
    assert_eq!(host.actions[0].kind, HostActionKind::IrqGfxPulse);
    assert_eq!(state.display.online_tries, 1);
    let mut pending = [0u8; 4];
    host.read_gpa(gpa + DISPLAY_SHARED_PENDING, &mut pending)
        .unwrap();
    assert_eq!(ld32(&pending), DISPLAY_ONLINE_EVENT_MASK);
    // After ack, no more asserts.
    state.display.online_acked = true;
    host.actions.clear();
    state.display.poll_ctr = DISPLAY_ONLINE_POLL_DIVISOR - 1;
    try_display_online(&mut state, &mut host);
    assert!(host.actions.is_empty());
}

/// Display-lifecycle instrumentation: SETUP_SHARED_STATE, ONLINE ack, and the
/// first ONLINE signal each leave an always-on line so a bad boot has a
/// display-lifecycle timeline to correlate with post_converge_regress. A
/// SETUP_SHARED_STATE while already ONLINE logs reinit=1 — the post-converge
/// display rebuild that is the standing overlay lead.
#[test]
fn display_lifecycle_events_are_always_logged() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let index = 0u32;
    let pfn = 0x7bu32;
    let gpa = state.pfn_gpa(pfn);
    host.map_range(gpa, PAGE_SIZE_ARM64E as usize, 0);

    let mut payload = vec![0u8; CHILD_SHARED_STATE_LEN];
    payload[CHILD_SHARED_STATE_INDEX..CHILD_SHARED_STATE_INDEX + 4]
        .copy_from_slice(&index.to_le_bytes());
    payload[CHILD_SHARED_STATE_PFN..CHILD_SHARED_STATE_PFN + 4].copy_from_slice(&pfn.to_le_bytes());
    let setup = Packet {
        opcode: CHILD_OP_SETUP_SHARED_STATE,
        stamp_count: 0,
        total_size: PACKET_HEADER_LEN + CHILD_SHARED_STATE_LEN as u32,
        completion_stamp: 0,
        payload,
        next_head: 0,
    };

    // First setup: reinit=0 (initial display registration).
    process_child_packet(&mut state, &mut host, 4, &setup);
    // Guest ack.
    let ack = Packet {
        opcode: CHILD_OP_ONLINE_ACK,
        stamp_count: 0,
        total_size: PACKET_HEADER_LEN,
        completion_stamp: 0,
        payload: vec![],
        next_head: 0,
    };
    process_child_packet(&mut state, &mut host, 4, &ack);
    assert!(state.display.online_acked);
    // First ONLINE signal (guest published enable bit 2).
    let mut m = [0u8; 4];
    st32(&mut m, DISPLAY_ONLINE_EVENT_MASK);
    host.write_gpa(gpa + DISPLAY_SHARED_ENABLE_MASK, &m)
        .unwrap();
    state.display.online_acked = false;
    state.display.poll_ctr = DISPLAY_ONLINE_POLL_DIVISOR - 1;
    try_display_online(&mut state, &mut host);

    // Second setup while previously ONLINE: reinit=1 (the post-converge rebuild).
    state.display.online_acked = true;
    process_child_packet(&mut state, &mut host, 4, &setup);

    let log = std::fs::read_to_string(crate::observe::fail_log_path()).expect("fail log");
    assert!(
        log.contains(&format!(
            "display_shared_state_setup index={index} gpa={gpa:#x} reinit=0"
        )),
        "initial setup must log reinit=0"
    );
    assert!(
        log.contains(&format!(
            "display_shared_state_setup index={index} gpa={gpa:#x} reinit=1"
        )),
        "re-setup while ONLINE must log reinit=1"
    );
    assert!(
        log.contains(&format!("display_online_ack index={index}")),
        "ONLINE ack must be logged"
    );
    assert!(
        log.contains(&format!("display_online_signal index={index}")),
        "first ONLINE signal must be logged"
    );
}

/// A guest display reinit (SETUP_SHARED_STATE while already ONLINE) that
/// arrives *after* boot-convergence self-labels with one correlated
/// `post_converge_display_reinit` line — the smoking gun for the intermittent
/// post-converge boot-progress overlay. Before
/// convergence the same reinit must NOT emit the correlated line (a display
/// re-register during normal boot bring-up is expected, not the overlay).
#[test]
fn post_converge_display_reinit_self_labels_only_after_converge() {
    use crate::runtime::census::present_proxy::{self, PresentCaptureSample};
    let _proxy = present_proxy::test_exclusive();

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let index = 0u32;
    let pfn = 0x91u32;
    let gpa = state.pfn_gpa(pfn);
    host.map_range(gpa, PAGE_SIZE_ARM64E as usize, 0);

    let mut payload = vec![0u8; CHILD_SHARED_STATE_LEN];
    payload[CHILD_SHARED_STATE_INDEX..CHILD_SHARED_STATE_INDEX + 4]
        .copy_from_slice(&index.to_le_bytes());
    payload[CHILD_SHARED_STATE_PFN..CHILD_SHARED_STATE_PFN + 4].copy_from_slice(&pfn.to_le_bytes());
    let setup = Packet {
        opcode: CHILD_OP_SETUP_SHARED_STATE,
        stamp_count: 0,
        total_size: PACKET_HEADER_LEN + CHILD_SHARED_STATE_LEN as u32,
        completion_stamp: 0,
        payload,
        next_head: 0,
    };

    // Reinit BEFORE convergence: expected boot bring-up, no correlated line.
    present_proxy::reset_for_test();
    assert!(!present_proxy::has_converged());
    state.display.online_acked = true;
    process_child_packet(&mut state, &mut host, 4, &setup);
    assert!(
        !present_proxy::has_converged(),
        "no dense present fed → must not be converged"
    );

    // Drive convergence: one dense full-size present (present_converge fires).
    present_proxy::note_capture_ok(PresentCaptureSample {
        mapping_id: 2,
        generation: 1,
        width: 1920,
        height: 1080,
        nz: 1920 * 1080 * 4,
        rgb_nz: 1920 * 1080,
        max_byte: 0xCC,
        max_rgb: 0xCC,
        from_last_store: true,
        edge_energy: 1000,
        named_peer: false,
    });
    assert!(
        present_proxy::has_converged(),
        "dense 1920x1080 present must converge the proxy"
    );

    // Reinit AFTER convergence: the smoking-gun correlated line must fire.
    state.display.online_acked = true;
    process_child_packet(&mut state, &mut host, 4, &setup);

    let log = std::fs::read_to_string(crate::observe::fail_log_path()).expect("fail log");
    assert!(
        log.contains(&format!(
            "post_converge_display_reinit index={index} gpa={gpa:#x}"
        )),
        "post-converge reinit must self-label with the correlated proxy line"
    );
}

/// Clear-only DisplaySwap of mid 2 must keep a finished composite retain on mid 1
/// (command-class history — not rgb_nz). Live dual-mid black class.
#[test]
fn present_clear_only_keeps_prior_composite_retain() {
    use crate::contract::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
    use crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    use crate::runtime::mapping_write::write_bgra8;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let w = 1920u32;
    let h = 1080u32;
    let stride = w * 4;
    let need = (stride as usize) * (h as usize);
    let page_shift = PAGE_SHIFT_X86;
    let page_size = 1u64 << page_shift;
    let pages = (need as u64).div_ceil(page_size) as usize;
    for mid in [1u32, 2u32] {
        let base_pfn = 0x100u32 + mid * 0x1000;
        let mut entries = Vec::with_capacity(pages);
        for i in 0..pages {
            let pfn = base_pfn + i as u32;
            let gpa = (pfn as u64) << page_shift;
            host.map_range(gpa, page_size as usize, 0);
            entries
                .push((((pfn as u64) << PAGE_ENTRY_PFN_SHIFT) | (PAGE_ENTRY_VALID as u64)) as u32);
        }
        assert!(state.map_surface(mid));
        {
            let m = state.mappings.get_mut(&mid).unwrap();
            m.mapped = true;
            m.mapping_internal = mid as u64;
            m.page_entries = entries;
        }
        assert!(state.set_mapping_geom(mid, w, h, MTL_FORMAT_BGRA8_UNORM));
    }
    // Mid 1: composite gray (Draw Store).
    let gray = vec![0xCCu8; need];
    assert!(write_bgra8(&mut state, &mut host, 1, &gray, stride, w, h));
    state.note_surface_composite(1);
    // Present mid 1 → +0x188 holds gray.
    {
        let mut payload = vec![0u8; PRESENT_X86_MIN_LEN];
        payload[PRESENT_X86_SURFACE_ID..PRESENT_X86_SURFACE_ID + 4]
            .copy_from_slice(&1u32.to_le_bytes());
        process_child_packet(
            &mut state,
            &mut host,
            5,
            &Packet {
                opcode: CHILD_OP_PRESENT_X86,
                stamp_count: 0,
                total_size: PACKET_HEADER_LEN + PRESENT_X86_MIN_LEN as u32,
                completion_stamp: 0,
                payload,
                next_head: 0,
            },
        );
    }
    assert_eq!(state.present.frame_mapping, 1);
    assert!(state.present.frame_valid);
    let prior_px = state.present.frame_bgra[0];
    assert_eq!(prior_px, 0xCC);
    host.actions.clear();
    // Mid 2: clear-only (no composite Store).
    let black = vec![0u8; need];
    // solid alpha-black clear
    let mut clear = black;
    for px in clear.chunks_exact_mut(4) {
        px[3] = 255;
    }
    assert!(write_bgra8(&mut state, &mut host, 2, &clear, stride, w, h));
    state.note_surface_clear(2);
    // Present mid 2 ClearOnly → keep mid 1 gray retain.
    {
        let mut payload = vec![0u8; PRESENT_X86_MIN_LEN];
        payload[PRESENT_X86_SURFACE_ID..PRESENT_X86_SURFACE_ID + 4]
            .copy_from_slice(&2u32.to_le_bytes());
        process_child_packet(
            &mut state,
            &mut host,
            5,
            &Packet {
                opcode: CHILD_OP_PRESENT_X86,
                stamp_count: 0,
                total_size: PACKET_HEADER_LEN + PRESENT_X86_MIN_LEN as u32,
                completion_stamp: 0,
                payload,
                next_head: 0,
            },
        );
    }
    assert_eq!(
        state.present.present_mapping, 2,
        "guest DisplaySwap mid still named present"
    );
    assert_eq!(
        state.present.frame_mapping, 1,
        "+0x188 still prior composite mid"
    );
    assert_eq!(state.present.frame_bgra[0], 0xCC);
    assert_eq!(
        presented_mapping(&state),
        Some(1),
        "window must re-show prior composite mid 1"
    );
    assert_coalesced_paint_action(&host, "clear-only keeps composite retain");
}

#[test]
fn signal_display_vbl_after_online_uses_shared_time_limiter() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let last_ms = std::sync::atomic::AtomicU64::new(0);
    let gpa = 0x7c000000u64;
    host.map_range(gpa, PAGE_SIZE_ARM64E as usize, 0);
    state.display.shared_gpa = gpa;
    state.display.display_index = 0;
    state.display.online_acked = true;

    let base = 5_000;
    signal_display_vbl_at(&mut state, &mut host, &last_ms, base);
    assert_eq!(host.actions.len(), 1);
    assert!(
        !claim_display_vbl(&last_ms, base),
        "the contended path cannot claim the locked path's interval"
    );
    signal_display_vbl_at(
        &mut state,
        &mut host,
        &last_ms,
        base + DISPLAY_VBL_MIN_INTERVAL_MS - 1,
    );
    assert_eq!(
        host.actions.len(),
        1,
        "polls inside the interval must not over-signal"
    );
    signal_display_vbl_at(
        &mut state,
        &mut host,
        &last_ms,
        base + DISPLAY_VBL_MIN_INTERVAL_MS,
    );
    assert_eq!(
        host.actions.len(),
        2,
        "the exact interval boundary must signal"
    );
    assert_eq!(host.actions[0].kind, HostActionKind::IrqGfxPulse);
    let mut pending = [0u8; 4];
    host.read_gpa(gpa + DISPLAY_SHARED_PENDING, &mut pending)
        .unwrap();
    assert_ne!(ld32(&pending) & DISPLAY_VBL_EVENT_MASK, 0);
    assert_ne!(
        state
            .gfx
            .interrupt_status_disp
            .load(std::sync::atomic::Ordering::Acquire)
            & 1,
        0
    );
}

/// The VBL limiter is phase-locked to a fixed interval grid so poll jitter
/// cannot alias the delivered rate down to ~60 Hz (the boot-to-boot fps split).
/// Polls spaced just under the interval — the worst aliasing case — must still
/// converge to roughly the grid rate, NOT halve.
#[test]
fn claim_display_vbl_phase_locks_grid_under_jittery_polls() {
    use std::sync::atomic::AtomicU64;
    let interval = DISPLAY_VBL_MIN_INTERVAL_MS;
    // Legacy "reset to now" behaviour would need two of these ~(interval-1)ms
    // polls per claim -> half rate. Phase-locking must claim on (nearly) every
    // poll once warmed up, because a late poll advances the grid by exactly one
    // interval and the next poll is already past the new deadline.
    let last = AtomicU64::new(0);
    let step = interval - 1; // poll spacing in the aliasing danger zone
    let polls = 64u64;
    let mut claims = 0u64;
    for i in 1..=polls {
        if claim_display_vbl(&last, i * step) {
            claims += 1;
        }
    }
    // Wall time covered is polls*step; a phase-locked grid delivers about one
    // VBL per interval, i.e. ~polls*step/interval claims — far above the
    // half-rate (~polls/2) the "reset to now" limiter produced.
    let grid_expected = polls * step / interval;
    assert!(
        claims >= grid_expected - 1,
        "phase-locked claims {claims} should track the grid rate ~{grid_expected}, not halve"
    );
    assert!(
        claims > polls * 2 / 3,
        "claims {claims} aliased below the grid — the 60-Hz-latch regression"
    );
}

/// A stall longer than two intervals (drain worker held the lock) resyncs the
/// phase to `now` rather than firing a back-dated burst of catch-up VBLs.
#[test]
fn claim_display_vbl_long_stall_resyncs_without_burst() {
    use std::sync::atomic::{AtomicU64, Ordering};
    let interval = DISPLAY_VBL_MIN_INTERVAL_MS;
    let last = AtomicU64::new(1_000);
    // A single poll after a 10*interval stall claims exactly once and lands the
    // grid at `now` (no accumulated catch-up credit).
    let now = 1_000 + 10 * interval;
    assert!(claim_display_vbl(&last, now));
    assert_eq!(
        last.load(Ordering::Acquire),
        now,
        "long stall resyncs to now"
    );
    // The immediately following poll one interval later claims once more — a
    // steady single-VBL cadence, not a burst.
    assert!(claim_display_vbl(&last, now + interval));
    assert!(!claim_display_vbl(&last, now + interval)); // same instant: no double
}

/// After online is acked, a stale ONLINE bit (bit2) left in pending is
/// suppressed by the present/VBL signalers instead of re-delivered — else the
/// guest re-runs process_online → connectionChange → boot-progress overlay
/// (x86 RE 2026-07-17). The signaler still records `stale_online_pending`
/// (measure + fix together). Pre-ack the bit is preserved (see the present
/// completion test) — the suppression is gated strictly on `online_acked`.
#[test]
fn acked_stale_online_bit_is_suppressed_not_redelivered() {
    let _proxy = crate::runtime::census::present_proxy::test_exclusive();
    crate::runtime::census::present_proxy::reset_for_test();

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let gpa = 0x7d00_0000u64;
    host.map_range(gpa, PAGE_SIZE_ARM64E as usize, 0);
    state.display.shared_gpa = gpa;
    state.display.display_index = 0;
    state.display.online_acked = true;
    host.put_u32(
        gpa + DISPLAY_SHARED_ENABLE_MASK,
        DISPLAY_PRESENT_EVENT_MASK | DISPLAY_ONLINE_EVENT_MASK,
    );
    // Stale ONLINE bit left in pending (the try_display_online/ack race).
    host.put_u32(gpa + DISPLAY_SHARED_PENDING, DISPLAY_ONLINE_EVENT_MASK);

    signal_display_present_complete(&mut state, &mut host);

    let mut pending = [0u8; 4];
    host.read_gpa(gpa + DISPLAY_SHARED_PENDING, &mut pending)
        .unwrap();
    let p = ld32(&pending);
    assert_ne!(
        p & DISPLAY_PRESENT_EVENT_MASK,
        0,
        "present completion still sets the present bit"
    );
    assert_eq!(
        p & DISPLAY_ONLINE_EVENT_MASK,
        0,
        "a stale acked ONLINE bit must be suppressed, not re-delivered"
    );
    assert_eq!(
        crate::runtime::census::present_proxy::counters().stale_online_pending,
        1,
        "the suppressed stale online must still be measured"
    );
}

/// RE: Unmap is PT-only. Discrete encode must stay in host_cache so sample
/// hits GVA key after remount even when guest pages are new zeros.
/// MapMemory2 stays notify-only (no invent write).
/// HostOps GVA views covering the range **must** be retired (Apple unmapMemory).
#[test]
fn unmap_memory_retains_gva_host_cache_for_sample() {
    use crate::contract::endian::{st32, st64};
    use crate::model::GvaHostView;
    use crate::runtime::decode::fifo::CHILD_OP_UNMAP_MEMORY;
    use crate::runtime::surface_cache;

    let page_shift = PAGE_SHIFT_X86;
    let mut host = FakeHost::new();
    let mut state = DeviceState::new(DeviceId(1), page_shift);
    let gva = 0x2c22000u64;
    let w = 32u32;
    let h = 24u32;
    let need = (w * h * 4) as usize;
    let mut bgra = vec![0u8; need];
    for px in bgra.chunks_exact_mut(4) {
        px[0] = 185;
        px[1] = 126;
        px[2] = 81;
        px[3] = 255;
    }
    surface_cache::store_gva(&mut state, gva, w, h, bgra);
    // Simulated HostOps view of the same GVA (zero-copy import substrate).
    state.gva_host_views.push(GvaHostView {
        task_id: 1,
        gva,
        length: 0x10000,
        ptr: 0xfeed_0000,
        ptr_len: 0x10000,
        ..Default::default()
    });
    // Unrelated range must survive.
    state.gva_host_views.push(GvaHostView {
        task_id: 1,
        gva: 0x4000_0000,
        length: 0x1000,
        ptr: 0xcafe_0000,
        ptr_len: 0x1000,
        ..Default::default()
    });

    let mut unmap_pl = vec![0u8; 20];
    st32(&mut unmap_pl[0..4], 1);
    st64(&mut unmap_pl[4..12], gva);
    st64(&mut unmap_pl[12..20], 0x10000);
    let unmap = Packet {
        opcode: CHILD_OP_UNMAP_MEMORY,
        stamp_count: 0,
        total_size: PACKET_HEADER_LEN + 20,
        completion_stamp: 0,
        payload: unmap_pl,
        next_head: 0,
    };
    process_child_packet(&mut state, &mut host, 2, &unmap);

    // Still sampleable from host_cache (no size gate; no Map rehydrate write).
    let got = surface_cache::get_gva(&state, gva, w, h).expect("retain after Unmap");
    assert_eq!(&got[0..4], &[185, 126, 81, 255]);
    // HostOps view of the unmapped range is gone; other GVA view kept.
    assert_eq!(state.gva_host_views.len(), 1);
    assert_eq!(state.gva_host_views[0].ptr, 0xcafe_0000);
    assert_eq!(state.retired_views, vec![(0xfeed_0000, 0x10000)]);
}

/// RE pageBacking Invalidate: clr hostValid → bump content_generation.
#[test]
fn invalidate_resources_bumps_mapping_content_generation() {
    use crate::contract::endian::st32;
    use crate::runtime::decode::fifo::{
        CHILD_INVALIDATE_PAGEON_FLAGS, CHILD_OP_INVALIDATE_RESOURCES,
    };

    let mut host = FakeHost::new();
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    assert!(state.map_surface(0x2a));
    {
        let m = state.mappings.get_mut(&0x2a).unwrap();
        m.content_generation = 7;
    }
    let mut pl = vec![0u8; 16];
    st32(&mut pl[0..], 0);
    st32(&mut pl[4..], 1);
    st32(&mut pl[8..], 0x2a);
    st32(&mut pl[12..], CHILD_INVALIDATE_PAGEON_FLAGS);
    process_child_packet(
        &mut state,
        &mut host,
        4,
        &Packet {
            opcode: CHILD_OP_INVALIDATE_RESOURCES,
            stamp_count: 0,
            total_size: PACKET_HEADER_LEN + 16,
            completion_stamp: 0,
            payload: pl,
            next_head: 0,
        },
    );
    assert_eq!(state.mappings[&0x2a].content_generation, 8);
}

/// MapMemory2 product path must **not** write guest GVA (flush disabled after
/// freelist PTE panic correlation). Helper still unit-tested in surface_cache.
#[test]
fn map_memory2_does_not_flush_gva_host_cache_on_wire() {
    use crate::contract::endian::{st32, st64};
    use crate::contract::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
    use crate::runtime::decode::fifo::CHILD_OP_MAP_MEMORY2;
    use crate::runtime::surface_cache;

    let page_shift = PAGE_SHIFT_X86;
    let mut host = FakeHost::new();
    let dir_gpa = 2u64 << page_shift;
    let root_gpa = 3u64 << page_shift;
    let data_gpa = 5u64 << page_shift;
    host.map_range(dir_gpa, 0x20, 0);
    host.map_range(root_gpa, 0x1000, 0);
    host.map_range(data_gpa, 0x1000, 0);
    let mut d = [0u8; 8];
    st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 3);
    st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
    let _ = host.write_gpa(dir_gpa, &d);
    st32(&mut d[..4], 5);
    let _ = host.write_gpa(root_gpa + 4, &d[..4]);

    let mut state = DeviceState::new(DeviceId(1), page_shift);
    assert!(state.define_task(1, 0x1000, 2));
    let gva = 1u64 << page_shift;
    let mut bgra = vec![0u8; 16];
    bgra[0] = 185;
    bgra[1] = 126;
    bgra[2] = 81;
    bgra[3] = 255;
    surface_cache::store_gva(&mut state, gva, 2, 2, bgra);

    let mut pl = vec![0u8; 20];
    st32(&mut pl[0..], 1);
    st64(&mut pl[4..], gva);
    st64(&mut pl[12..], 0x1000);
    process_child_packet(
        &mut state,
        &mut host,
        2,
        &Packet {
            opcode: CHILD_OP_MAP_MEMORY2,
            stamp_count: 0,
            total_size: PACKET_HEADER_LEN + 20,
            completion_stamp: 0,
            payload: pl,
            next_head: 0,
        },
    );
    let mut probe = [0u8; 4];
    host.read_gpa(data_gpa, &mut probe).unwrap();
    assert_eq!(
        probe,
        [0, 0, 0, 0],
        "product MapMemory2 must stay notify-only for GVA (no auto flush)"
    );
    // Helper path still works when called explicitly.
    let fl = surface_cache::flush_gva_host_cache_on_map(&mut state, &mut host, 1, gva, 0x1000);
    assert_eq!(fl.wrote, 1);
    host.read_gpa(data_gpa, &mut probe).unwrap();
    assert_eq!(&probe[..4], &[185, 126, 81, 255]);
}

/// Synchronize 0x35 is stamp + wait only — no host_cache→guest write (RE audit).
#[test]
fn synchronize_resources_does_not_write_guest_pages() {
    use crate::contract::endian::st32;
    use crate::contract::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
    use crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    use crate::runtime::decode::fifo::CHILD_OP_SYNCHRONIZE_RESOURCES;
    use crate::runtime::surface_cache;

    let page_shift = PAGE_SHIFT_X86;
    let page_size = 1u64 << page_shift;
    let mut host = FakeHost::new();
    let mut state = DeviceState::new(DeviceId(1), page_shift);
    let mid = 0x2au32;
    let w = 2u32;
    let h = 2u32;
    let pfn = 0x7000u32;
    let gpa = (pfn as u64) << page_shift;
    host.map_range(gpa, page_size as usize, 0);
    assert!(state.map_surface(mid));
    {
        let m = state.mappings.get_mut(&mid).unwrap();
        m.mapped = true;
        m.page_entries =
            vec![(((pfn as u64) << PAGE_ENTRY_PFN_SHIFT) | (PAGE_ENTRY_VALID as u64)) as u32];
    }
    assert!(state.set_mapping_geom(mid, w, h, MTL_FORMAT_BGRA8_UNORM));
    let mut bgra = vec![0u8; (w * h * 4) as usize];
    bgra[0] = 0x10;
    bgra[1] = 0x20;
    bgra[2] = 0x30;
    bgra[3] = 0xff;
    surface_cache::store(&mut state, mid, w, h, bgra);

    let mut pl = vec![0u8; 12];
    st32(&mut pl[0..], 1);
    st32(&mut pl[4..], 1);
    st32(&mut pl[8..], mid);
    process_child_packet(
        &mut state,
        &mut host,
        4,
        &Packet {
            opcode: CHILD_OP_SYNCHRONIZE_RESOURCES,
            stamp_count: 0,
            total_size: PACKET_HEADER_LEN + 12,
            completion_stamp: 0,
            payload: pl,
            next_head: 0,
        },
    );
    let mut probe = [0u8; 4];
    host.read_gpa(gpa, &mut probe).unwrap();
    assert_eq!(
        probe,
        [0, 0, 0, 0],
        "Synchronize must not write host_cache into guest pages"
    );
    // Helper remains for explicit/unit use only.
    assert!(matches!(
        surface_cache::flush_surface_id_to_guest_pages(&mut state, &mut host, mid),
        surface_cache::SyncFlushResult::WroteGuest { .. }
    ));
}

/// set guestValid alone must not bump host content generation.
#[test]
fn invalidate_without_clr_host_does_not_bump_generation() {
    use crate::contract::endian::st32;
    use crate::runtime::decode::fifo::CHILD_OP_INVALIDATE_RESOURCES;

    let mut host = FakeHost::new();
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    assert!(state.map_surface(0x2a));
    {
        let m = state.mappings.get_mut(&0x2a).unwrap();
        m.content_generation = 7;
    }
    // LE bytes 00 00 00 01 = only set_guest_valid
    let mut pl = vec![0u8; 16];
    st32(&mut pl[0..], 0);
    st32(&mut pl[4..], 1);
    st32(&mut pl[8..], 0x2a);
    st32(&mut pl[12..], 0x0100_0000);
    process_child_packet(
        &mut state,
        &mut host,
        4,
        &Packet {
            opcode: CHILD_OP_INVALIDATE_RESOURCES,
            stamp_count: 0,
            total_size: PACKET_HEADER_LEN + 16,
            completion_stamp: 0,
            payload: pl,
            next_head: 0,
        },
    );
    assert_eq!(state.mappings[&0x2a].content_generation, 7);
}

/// `present_unbacked` gate: a member presented twice with no full-frame Store
/// **naming it** in between is being shown content the guest never sent for it.
/// `note_compositor_member_published` is the only site that advances
/// `dense_frame_seq`, so an unchanged seq across a member's own two presents is
/// the exact structural witness.
///
/// The gate used to be described as covering "a full-frame Store or an
/// inter-buffer seed". `62587b1` deleted the peer front seed, so only the first
/// half survives.
///
/// Healthy alternation must stay quiet: each buffer advances on its own turn.
#[test]
fn present_backing_gate_fires_only_when_a_member_gained_nothing() {
    let w = 1920u32;
    let h = 1080u32;
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    for mid in [1u32, 5u32] {
        state.map_surface(mid);
        state.note_compositor_member_published(mid, w, h);
    }

    // First present of each member has no prior witness — never a report.
    assert_eq!(state.note_present_backing(1), None);
    assert_eq!(state.note_present_backing(5), None);

    // Healthy a/b alternation: each member gets its own full frame before its
    // next present, so the seq advances and the gate stays silent.
    for _ in 0..4 {
        state.note_compositor_member_published(1, w, h);
        assert_eq!(state.note_present_backing(1), None);
        state.note_compositor_member_published(5, w, h);
        assert_eq!(state.note_present_backing(5), None);
    }

    // Mid 5 now goes dark: every full frame lands on mid 1, but the guest keeps
    // naming mid 5 at present. Each of those presents shows content mid 5 never
    // received, and each is reported (once per present, not once per lifetime).
    for _ in 0..3 {
        state.note_compositor_member_published(1, w, h);
        assert_eq!(state.note_present_backing(1), None);
        assert!(
            state.note_present_backing(5).is_some(),
            "no full-frame store named mid 5"
        );
    }

    // Backing is the seq itself, whatever advanced it: a member that reaches the
    // source's seq is quiet again on its next present.
    state.present.dense_frame_seq.insert(
        5,
        state.present.dense_frame_seq.get(&1).copied().unwrap_or(0),
    );
    assert_eq!(state.note_present_backing(5), None);

    // A recycled mapping id must not compare against its predecessor's witness.
    state.unmap_surface(5);
    state.map_surface(5);
    state.note_compositor_member_published(5, w, h);
    assert_eq!(state.note_present_backing(5), None);
}

/// What the backing gate CANNOT see, pinned so the limitation is executable
/// rather than a comment someone stops believing.
///
/// The gate's answer is a function of Store bookkeeping alone. Two states with
/// identical publish/present histories give identical answers even when one
/// resolves the mid to the shared `OutputGroup` resident and the other to a
/// private per-mid `Surface` — which is precisely the disagreement that blacks
/// out the desktop: the guest's full frame is stored, so the seq advances and
/// this gate reports backed, while the present reads a resident that never got
/// those pixels. `present_identity_flip` is the gate for that, and it is the one
/// that moved 9 -> 0 across the fix.
#[test]
fn the_backing_gate_answers_the_same_whichever_resident_the_mid_resolves_to() {
    let (w, h) = (1920u32, 1080u32);

    // Grouped: two members presented at this geometry, so the swapchain latches
    // and both resolve to the shared resident.
    let mut grouped = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    for mid in [1u32, 6u32] {
        grouped.map_surface(mid);
        grouped.note_presented_geom(mid, w, h);
    }
    assert!(grouped.output_group_for(6, w, h).is_some());

    // Private: same publish/present history, but nothing was ever presented
    // here, so mid 6 keeps a per-mid resident.
    let mut private = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    for mid in [1u32, 6u32] {
        private.map_surface(mid);
    }
    assert!(private.output_group_for(6, w, h).is_none());

    for state in [&mut grouped, &mut private] {
        // A full frame naming mid 6 was stored. The gate goes quiet on the next
        // present in BOTH states — including the one where mid 6 owns a private
        // resident. That silence is about the Store, not about where its pixels
        // landed, and it is the false negative the black-desktop class hides in.
        state.note_compositor_member_published(6, w, h);
        assert_eq!(state.note_present_backing(6), None);
        // Frames now go to mid 1 only; mid 6 gains nothing.
        state.note_compositor_member_published(1, w, h);
    }
    let (g, p) = (grouped.note_present_backing(6), private.note_present_backing(6));
    assert!(
        g.is_some() && p.is_some(),
        "both arms must actually reach the firing state, or the equality below \
         proves nothing: grouped={g:?} private={p:?}"
    );
    assert_eq!(
        g.is_some(),
        p.is_some(),
        "the gate must not be read as evidence about the resident: it gives the \
         same answer for a mid on the shared resident and a mid on its own"
    );
}

/// The display-transaction probe must key on the payload *shape*, not fire per
/// present.
///
/// A steady-state x86 boot pushes tens of thousands of opcode-6 packets. An
/// unbounded `display_txn_payload` line would bury every other always-on record
/// in the fail log, which is the one place a bad boot explains itself — the
/// probe would destroy the evidence it exists to collect. A repeat of a shape we
/// have already sampled carries no new information about where the plane list
/// lives, so the budget is per
/// `(opcode, payload_len, pipe_index, task_field_is_set)`.
#[test]
fn display_txn_payload_probe_is_bounded_per_wire_shape() {
    let mut state = DeviceState::new(crate::model::DeviceId(1), PAGE_SHIFT_X86);
    let packet = |opcode: u16, plen: usize| Packet {
        opcode,
        stamp_count: 0,
        total_size: PACKET_HEADER_LEN + plen as u32,
        completion_stamp: 0,
        payload: vec![0u8; plen],
        next_head: 0,
    };

    for _ in 0..(DISPLAY_TXN_PAYLOAD_SAMPLES * 8) {
        note_display_txn_payload(&mut state, 5, &packet(CHILD_OP_PRESENT_X86, 12));
    }
    assert_eq!(
        state
            .display
            .txn_payload_samples
            .get(&(CHILD_OP_PRESENT_X86, 12, 0, false)),
        Some(&DISPLAY_TXN_PAYLOAD_SAMPLES),
        "a known shape must stop logging once its sample budget is spent"
    );

    // A *new* length is the whole point of the measurement: if the guest ever
    // appends an inline plane list the payload grows, and that packet must be
    // sampled even though the opcode is one we have already seen many times.
    note_display_txn_payload(&mut state, 5, &packet(CHILD_OP_PRESENT_X86, 64));
    assert_eq!(
        state
            .display
            .txn_payload_samples
            .get(&(CHILD_OP_PRESENT_X86, 64, 0, false)),
        Some(&1),
        "a new payload length must get its own sample budget"
    );

    // Opcodes are budgeted independently: 6 and 7 are different commands with
    // different trailers, and 8 is the arm64 pathway entirely.
    note_display_txn_payload(&mut state, 5, &packet(CHILD_OP_PRESENT_GAMMA_X86, 0x24));
    note_display_txn_payload(&mut state, 4, &packet(CHILD_OP_DISPLAY_SWAP, 12));
    assert_eq!(
        state
            .display
            .txn_payload_samples
            .get(&(CHILD_OP_PRESENT_GAMMA_X86, 0x24, 0, false)),
        Some(&1)
    );
    assert_eq!(
        state
            .display
            .txn_payload_samples
            .get(&(CHILD_OP_DISPLAY_SWAP, 12, 0, false)),
        Some(&1)
    );
}

/// The plane-0 surface id changes every frame by design; the pipe index and the
/// task field do not.
///
/// Keying the budget on length alone spent it inside the first 400ms of a live
/// boot and left the probe silent for the rest of the session, because the
/// length never varies. The trailer's other two words are what still carry news,
/// so a second display pipe or the task field's first non-zero value must each
/// re-arm the budget — while a fresh surface id every frame must not, or the
/// probe becomes unbounded again.
#[test]
fn display_txn_payload_probe_rearms_on_trailer_change_but_not_on_surface_id() {
    let mut state = DeviceState::new(crate::model::DeviceId(1), PAGE_SHIFT_X86);
    let trailer = |pipe: u32, surface: u32, task: u32| {
        let mut payload = Vec::new();
        payload.extend_from_slice(&pipe.to_le_bytes());
        payload.extend_from_slice(&surface.to_le_bytes());
        payload.extend_from_slice(&task.to_le_bytes());
        Packet {
            opcode: CHILD_OP_PRESENT_X86,
            stamp_count: 0,
            total_size: PACKET_HEADER_LEN + payload.len() as u32,
            completion_stamp: 0,
            payload,
            next_head: 0,
        }
    };

    // Bring-up: pipe 0, task still zero, a different surface id every frame.
    for surface in 0..(DISPLAY_TXN_PAYLOAD_SAMPLES * 8) {
        note_display_txn_payload(&mut state, 5, &trailer(0, surface + 1, 0));
    }
    assert_eq!(
        state
            .display
            .txn_payload_samples
            .get(&(CHILD_OP_PRESENT_X86, 12, 0, false)),
        Some(&DISPLAY_TXN_PAYLOAD_SAMPLES),
        "a per-frame surface id must not re-arm the budget"
    );
    assert_eq!(
        state.display.txn_payload_samples.len(),
        1,
        "surface ids must not each open their own bucket"
    );

    // Steady state: the task field goes non-zero. That transition is the open
    // question, so it gets a fresh budget exactly once.
    note_display_txn_payload(&mut state, 5, &trailer(0, 0x2a, 7));
    note_display_txn_payload(&mut state, 5, &trailer(0, 0x2b, 9));
    assert_eq!(
        state
            .display
            .txn_payload_samples
            .get(&(CHILD_OP_PRESENT_X86, 12, 0, true)),
        Some(&2),
        "the task field's first non-zero value must re-arm the budget"
    );

    // A second display pipe is a different wire shape, not a repeat.
    note_display_txn_payload(&mut state, 6, &trailer(1, 0x2a, 7));
    assert_eq!(
        state
            .display
            .txn_payload_samples
            .get(&(CHILD_OP_PRESENT_X86, 12, 1, true)),
        Some(&1),
        "a new pipe index must get its own sample budget"
    );
}

/// The gamma command swaps the surface id and the task field relative to the
/// plain one.
///
/// Both words are u32s in adjacent slots, so reading them at the wrong offsets
/// still yields plausible-looking values — the probe would key its budget on the
/// surface id and re-arm every frame, and the emitted `task=` would be a surface
/// id. Nothing downstream would report an error.
#[test]
fn display_txn_trailer_slots_follow_the_emitting_command() {
    // command 6: [pipe][surface][task] — surface in slot 1, task in slot 2.
    assert_eq!(display_txn_trailer_slots(CHILD_OP_PRESENT_X86), (1, 2));
    // command 7: [pipe][task][surface][gamma…] — the two are swapped.
    assert_eq!(display_txn_trailer_slots(CHILD_OP_PRESENT_GAMMA_X86), (2, 1));
    assert_eq!(display_txn_trailer_slots(CHILD_OP_DISPLAY_SWAP), (1, 2));

    // The swap has to survive the budget key, not just the log line: a gamma
    // packet whose *task* is zero and whose surface id is non-zero must land in
    // the task-is-zero bucket.
    let mut state = DeviceState::new(crate::model::DeviceId(1), PAGE_SHIFT_X86);
    let mut payload = Vec::new();
    payload.extend_from_slice(&0u32.to_le_bytes()); // pipe
    payload.extend_from_slice(&0u32.to_le_bytes()); // task
    payload.extend_from_slice(&0x2au32.to_le_bytes()); // surface
    payload.resize(0x24, 0);
    note_display_txn_payload(
        &mut state,
        5,
        &Packet {
            opcode: CHILD_OP_PRESENT_GAMMA_X86,
            stamp_count: 0,
            total_size: PACKET_HEADER_LEN + payload.len() as u32,
            completion_stamp: 0,
            payload,
            next_head: 0,
        },
    );
    assert_eq!(
        state
            .display
            .txn_payload_samples
            .get(&(CHILD_OP_PRESENT_GAMMA_X86, 0x24, 0, false)),
        Some(&1),
        "gamma's task field is slot 1; reading slot 2 would mistake a surface id for it"
    );
}

/// The trailer the guest appends after serializing the transaction's resource
/// list is 0x24 bytes for the gamma command and 0x0c for the plain one.
///
/// The probe reports the trailer read from *both* ends of the payload, and the
/// tail reading is only meaningful at the right width — get this wrong and a
/// payload that does carry an inline plane list would still look trailer-only.
#[test]
fn display_txn_trailer_width_matches_the_emitting_command() {
    assert_eq!(display_txn_trailer_len(CHILD_OP_PRESENT_X86), 0x0c);
    assert_eq!(display_txn_trailer_len(CHILD_OP_PRESENT_GAMMA_X86), 0x24);
    assert_eq!(display_txn_trailer_len(CHILD_OP_DISPLAY_SWAP), 0x0c);
}

/// Head and tail readings coincide exactly when the payload is trailer-only.
///
/// That coincidence is the measurement's verdict: agreement means the plane list
/// is not inline and a real decode has to reach it another way; divergence means
/// the list precedes the trailer and our fixed offset-zero read has been parsing
/// the list header all along.
#[test]
fn display_txn_probe_distinguishes_trailer_only_from_prefixed_payload() {
    let mut state = DeviceState::new(crate::model::DeviceId(1), PAGE_SHIFT_X86);

    // Trailer-only: [pipe=0][surface=0x2a][task=7].
    let mut trailer_only = Vec::new();
    trailer_only.extend_from_slice(&0u32.to_le_bytes());
    trailer_only.extend_from_slice(&0x2au32.to_le_bytes());
    trailer_only.extend_from_slice(&7u32.to_le_bytes());
    assert_eq!(trailer_only.len(), display_txn_trailer_len(CHILD_OP_PRESENT_X86));

    // Same trailer behind an 8-byte prefix: offset-zero now reads the prefix.
    let mut prefixed = vec![0xEEu8; 8];
    prefixed.extend_from_slice(&trailer_only);

    for payload in [trailer_only, prefixed] {
        let plen = payload.len();
        note_display_txn_payload(
            &mut state,
            5,
            &Packet {
                opcode: CHILD_OP_PRESENT_X86,
                stamp_count: 0,
                total_size: PACKET_HEADER_LEN + plen as u32,
                completion_stamp: 0,
                payload,
                next_head: 0,
            },
        );
        assert_eq!(
            state
                .display
                .txn_payload_samples
                .get(&(CHILD_OP_PRESENT_X86, plen, 0, true)),
            Some(&1),
            "each distinct shape must be sampled once"
        );
    }
}

/// A present the dmabuf carried is not a black present.
///
/// Route B skips the full-frame GPU→CPU readback on purpose, so `frame_bgra` is
/// empty by design and any `max_rgb == 0` test reports black on every present —
/// a live boot logged 1338 `present_black_retain` records against 1312 presents.
/// An always-on failure sink that fires on every healthy frame cannot surface the
/// unhealthy one, so "no pixels" must be its own verdict rather than folded into
/// "black".
#[test]
fn a_dmabuf_carried_present_is_unsampled_not_black() {
    assert_eq!(
        present_content_verdict(&[], 0),
        PresentContentVerdict::Unsampled,
        "no CPU pixels means no evidence, not evidence of black"
    );
    // A genuinely black sampled frame must still be caught — that is the record's
    // whole purpose, and the fix must not trade one blind spot for another.
    assert_eq!(
        present_content_verdict(&[0, 0, 0, 255], 0),
        PresentContentVerdict::Black,
        "an opaque all-zero-RGB frame is still black"
    );
    assert_eq!(
        present_content_verdict(&[0, 0, 0x40, 255], 0x40),
        PresentContentVerdict::Content
    );
}
