//! Tests for the deferred-flush rail.
//!
//! Nineteen of these carry `#[cfg(feature = "backend-vulkan")]` and the rest
//! do not, and the line between them is not a matter of taste: on a build
//! without the engine, the rail's seven entry points are stubs, so a test that
//! asserts a window's bytes reached guest memory is asserting something the
//! build cannot do. Those tests are Vulkan-rail tests and say so.
//!
//! The other twenty-one exercise decisions the rail makes before it reaches an
//! entry point — key identity, window bookkeeping, the refusals, the bands —
//! and hold on both arms.
//!
//! The split was measured rather than guessed. Compiling the seven entry
//! points to their non-Vulkan stubs on a `backend-vulkan` build and removing
//! the already-gated tests reproduces exactly what the Metal arm runs: it
//! reported 27 passed and 6 failed, and those six are the ones the gate was
//! added to. Re-running the same probe after gating them reports 27 passed and
//! none failed.

use crate::model::{ComputeStorageResidencyKey, DeviceId, DeviceState, PAGE_SHIFT_X86};

/// The fence-batch bands cover every count, start at one, and separate the
/// two readings that decide the round-trip build.
///
/// A batch of 1 and a batch of many are the whole question — one says the
/// per-window submit is all there is to save, the other says the waits
/// collapse — so they must never share a band. Zero is excluded rather than
/// banded: [`flush_gva_windows_before_fence`] returns before the call when
/// the map is empty, so a zero band could only ever count calls with nothing
/// to do, and one appearing would mean that early return had moved.
#[cfg(feature = "backend-vulkan")]
#[test]
fn the_fence_batch_bands_separate_one_from_many() {
    use super::witness::fence_batch_band;
    assert_eq!(fence_batch_band(0), None, "an empty batch was banded");
    assert_eq!(fence_batch_band(1), Some("gvaw_fence_batch_1"));
    assert_eq!(fence_batch_band(2), Some("gvaw_fence_batch_2"));
    // Every band is reachable and the boundaries do not overlap or leave a
    // gap: each count lands in exactly one, and the edges land where the
    // arms say they do.
    for (landed, want) in [
        (3u64, "gvaw_fence_batch_3_4"),
        (4, "gvaw_fence_batch_3_4"),
        (5, "gvaw_fence_batch_5_8"),
        (8, "gvaw_fence_batch_5_8"),
        (9, "gvaw_fence_batch_9_16"),
        (16, "gvaw_fence_batch_9_16"),
        (17, "gvaw_fence_batch_17_64"),
        (64, "gvaw_fence_batch_17_64"),
        (65, "gvaw_fence_batch_over_64"),
        (u64::MAX, "gvaw_fence_batch_over_64"),
    ] {
        assert_eq!(fence_batch_band(landed), Some(want), "landed={landed}");
    }
}

/// The alias probe charges the quiet arm when no rail is armed.
///
/// Every zero-copy bind makes this call, so which arm it takes is what
/// decides whether the probe is on the draw path's bill at all. The two
/// counters are read against each other on the `store_routes` line and a
/// counter on the wrong side of the early return would read as "the walk
/// never runs" — which is the answer that would retire a cost that is
/// actually being paid.
#[test]
fn an_unarmed_alias_probe_takes_the_quiet_arm() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut host = crate::runtime::host::FakeHost::default();
    let before = crate::runtime::drain::store_route_count("gva_alias_probe_quiet");
    let walked_before = crate::runtime::drain::store_route_count("gva_alias_probe_walked");
    super::flush_intersecting_task_gva(&mut state, &mut host, 1, 0x4000, 0x1000);
    assert_eq!(
        crate::runtime::drain::store_route_count("gva_alias_probe_quiet"),
        before + 1,
        "an unarmed probe must charge the quiet arm"
    );
    assert_eq!(
        crate::runtime::drain::store_route_count("gva_alias_probe_walked"),
        walked_before,
        "an unarmed probe must not reach the walking arm"
    );
}

/// A zero span is quiet too, and is counted — it shares the early return
/// with the unarmed case, so a reader dividing `gva_alias_probe_us` by the
/// walked count is not silently also dividing by these.
#[test]
fn a_zero_span_alias_probe_never_walks() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut host = crate::runtime::host::FakeHost::default();
    let walked_before = crate::runtime::drain::store_route_count("gva_alias_probe_walked");
    super::flush_intersecting_task_gva(&mut state, &mut host, 1, 0x4000, 0);
    assert_eq!(
        crate::runtime::drain::store_route_count("gva_alias_probe_walked"),
        walked_before
    );
}

fn key(mapping_id: u32, lo: u64, hi: u64) -> ComputeStorageResidencyKey {
    ComputeStorageResidencyKey {
        mapping_id,
        map_generation: 1,
        surface_offset: lo,
        surface_bpr: 64,
        span_end: hi,
        width: 4,
        height: 4,
        pixel_format: 0x46,
        texture_ref: 0,
    }
}

/// A render window carrying its own 4x4 BGRA frame — the geometry [`key`]
/// names, since the flush writes `key.width x key.height` from these bytes.
fn render_owner(armed_seq: u64) -> crate::model::DeferredOwner {
    crate::model::DeferredOwner::Render {
        armed_seq,
        armed_stamp_seq: 0,
        source: crate::model::RenderWindowSource::Owned(std::sync::Arc::new(vec![0u8; 4 * 4 * 4])),
    }
}

#[test]
fn condemn_keeps_content_state_and_lifecycle_clears_it() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    let m = state.mappings.entry(7).or_default();
    m.mapped = true;
    m.has_geom = true;
    m.width = 100;
    m.height = 50;
    m.format = 0x46;
    m.map_generation = 4;
    m.page_entries = vec![5, 9, 13];
    assert!(state.condemn_surface_backing(7));
    let e = state.mappings.get(&7).unwrap();
    assert!(e.mapped, "condemn must not unmap");
    assert!(e.has_geom, "condemn must keep geometry");
    assert_eq!(e.map_generation, 4, "condemn must not bump the generation");
    assert!(e.page_entries.is_empty(), "live bindings must be retired");
    assert_eq!(e.condemned_entries.as_deref(), Some(&[5u32, 9, 13][..]));
    assert!(state.mapping_backing_condemned(7));
    // Second condemn with no resolve between: nothing left to stash — the
    // caller falls back to full teardown (genuinely dead).
    // (mapping_backing_condemned gates that in the drain handler.)
    // A fresh MAP notify does NOT settle the pending decision (the notify
    // may trail our eager resolve of the same surface): the fingerprint
    // survives; only a resolve (or unmap/new-internal) settles it.
    assert!(state.map_surface(7));
    assert!(state.mapping_backing_condemned(7));
    assert!(state.unmap_surface(7));
    assert!(!state.mapping_backing_condemned(7));
    // Pageless mapping: condemn declines (caller tears down).
    let m = state.mappings.entry(8).or_default();
    m.mapped = true;
    assert!(!state.condemn_surface_backing(8));
}

#[test]
fn map_notify_stashes_fingerprint_instead_of_bumping() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    let m = state.mappings.entry(5).or_default();
    m.mapped = true;
    m.map_generation = 7;
    m.page_entries = vec![1, 2, 3];
    // The MAP notify often trails the eager resolve that established the
    // same surface: it must not bump (the resolve-time fingerprint compare
    // decides), so a deferred paint's resident/window stay live.
    assert!(state.map_surface(5));
    let e = state.mappings.get(&5).unwrap();
    assert_eq!(e.map_generation, 7, "late MAP notify must not bump");
    assert_eq!(e.condemned_entries.as_deref(), Some(&[1u32, 2, 3][..]));
    assert!(!e.has_geom, "geometry must re-resolve after MAP");
    // Same MappingInternal re-statement: full no-op for content state.
    let m = state.mappings.entry(6).or_default();
    m.mapped = true;
    m.map_generation = 9;
    m.mapping_internal = 0xabc;
    m.page_entries = vec![4, 5];
    m.has_geom = true;
    assert!(state.attach_mapping_internal(6, 0xabc));
    let e = state.mappings.get(&6).unwrap();
    assert_eq!(e.map_generation, 9);
    assert_eq!(e.page_entries, vec![4, 5]);
    assert!(e.has_geom, "same-internal re-statement keeps geometry");
    // Different MappingInternal: genuine new surface — full reset + bump.
    assert!(state.attach_mapping_internal(6, 0xdef));
    let e = state.mappings.get(&6).unwrap();
    assert_eq!(e.map_generation, 10);
    assert!(e.page_entries.is_empty());
}

/// The compute flush's drift line must print the generation its guard
/// compared, not the other one in scope.
///
/// `flush_one` holds two unrelated `u32`s: `key.map_generation` (the
/// mapping lifetime it compares) and the pinned resident's *content*
/// generation. The line printed the content generation in a field named
/// `gen`, adjacent to `reason=map_generation_drift current=…`, and a boot
/// was read as showing a mapping lifetime running backwards (`gen=3
/// current=Some(2)`) when the two numbers were never comparable.
#[cfg(feature = "backend-vulkan")]
#[test]
fn the_compute_drift_line_names_the_generation_it_compared() {
    use crate::runtime::host::FakeHost;
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
    let m = state.mappings.entry(9).or_default();
    m.mapped = true;
    // Distinct on purpose: the window's map_generation is 1 (from `key`),
    // the mapping is at 5, and the content generation is 3. Only one pair
    // of those is what the guard compares.
    m.map_generation = 5;
    state.compute_deferred_flush.insert(
        key(9, 0, 256),
        crate::model::DeferredOwner::Storage {
            generation: 3,
            armed_stamp_seq: 0,
        },
    );
    let cap = crate::observe::FailCapture::start();
    assert!(!super::flush_intersecting(
        &mut state,
        &mut host,
        9,
        0,
        u64::MAX
    ));
    let line = cap.one("deferred_flush_lost");
    assert!(
        line.contains("reason=map_generation_drift"),
        "wrong refusal: {line}"
    );
    assert!(
        line.contains(" gen=1 ") && line.contains("current=Some(5)"),
        "`gen=` must be the compared window generation: {line}"
    );
    assert!(
        line.contains("content_gen=3"),
        "the resident's content generation must say so in its name: {line}"
    );
    assert!(
        line.contains("kind=compute"),
        "every deferred_flush_lost names its path: {line}"
    );
}

/// A type-11 render window is found and landed by the *same* mapping-keyed
/// trigger the compute rail uses, and is read as a render window.
///
/// This is the property the whole deferred type-11 rail rests on. Its
/// pixels live in a target resident that `ComputeStorageResidencyKey`
/// cannot name, so the flush has to dispatch on the owner; if it did not,
/// `flush_intersecting` would hand a render window to the storage read and
/// report a compute loss for a window the compute rail never armed. Driving
/// it through `flush_intersecting` — rather than calling the flush directly
/// — is deliberate: that call is the choke point every guest-page reader
/// goes through, so this also pins the trigger wiring.
///
/// The map-generation drift is the cheap way to make the flush take a
/// decisive branch with no engine present. It doubles as coverage of the
/// recycled-pages guard: a mapping rebound since arm time must never have a
/// stale framebuffer written through its new pages.
#[cfg(feature = "backend-vulkan")]
#[test]
fn a_render_window_flushes_through_the_shared_trigger_and_names_its_rail() {
    use crate::runtime::host::FakeHost;
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
    let m = state.mappings.entry(9).or_default();
    m.mapped = true;
    // The window latched map_generation 1 (from `key`); the mapping has
    // since moved to 5, so its pages are not the ones the Store rendered
    // for.
    m.map_generation = 5;
    state
        .compute_deferred_flush
        .insert(key(9, 0, 256), render_owner(1));
    // Route counts are process-global and this suite runs serially, so take
    // a baseline rather than assuming this is the first window to drift.
    let before_gen_drift = crate::runtime::drain::store_route_count("rendflush_gen_drift");
    let cap = crate::observe::FailCapture::start();
    assert!(
        !super::flush_intersecting(&mut state, &mut host, 9, 0, u64::MAX),
        "a window that cannot be written must report the loss"
    );
    let line = cap.one("deferred_flush_lost");
    assert!(
        line.contains("kind=render"),
        "a render window must not be reported as a compute one: {line}"
    );
    assert!(
        line.contains("reason=map_generation_drift") && line.contains("current=Some(5)"),
        "the rebound mapping must be the stated refusal: {line}"
    );
    assert!(
        state.compute_deferred_flush.is_empty(),
        "the trigger must consume the window it took"
    );
    // A lost tile has to be countable, not just loggable. The window is gone
    // from `compute_deferred_flush` (asserted just above) and nothing
    // re-arms it, so this is a permanent loss of painted pixels — the Goal 3
    // event — and a census that cannot count it cannot score an arm against
    // it.
    //
    // `mapping_pages_drifted` is not a substitute: it is incremented inside
    // `mapping_pages_still_ours`, which more than one caller reaches, so it
    // counts refusals rather than lost tiles.
    assert_eq!(
        crate::runtime::drain::store_route_count("rendflush_gen_drift"),
        before_gen_drift + 1,
        "the generation-drift loss must be counted on the store-route census"
    );
}

/// The other drift refusal, and the one a live boot actually takes.
///
/// `map_generation` drift is the guest's *declared* rebind; page drift is a
/// type-4 surface re-pointed with nothing declared, which is the shape
/// traced end to end on a control boot — a 1225x512 WebKit tile whose
/// backing was fabricated at its own GVA, then refused when the live walk
/// disagreed. Both refusals are correct and both lose the tile, so both have
/// to be countable apart; testing only the sibling would leave the branch a
/// live boot exercises uncovered.
#[cfg(feature = "backend-vulkan")]
#[test]
fn a_render_window_over_repointed_pages_is_refused_and_counted() {
    use crate::contract::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
    use crate::contract::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
    use crate::model::Type4Walk;
    use crate::runtime::host::{FakeHost, HostMemory};

    const BACKING_PFN: u32 = 9;
    let page = 1u64 << PAGE_SHIFT_X86;
    let mut host = FakeHost::new();
    let dir_gpa = 2u64 << PAGE_SHIFT_X86;
    let root_gpa = 3u64 << PAGE_SHIFT_X86;
    let data0 = 4u64 << PAGE_SHIFT_X86;
    for gpa in [dir_gpa, root_gpa, data0] {
        host.map_range(gpa, page as usize, 0);
    }
    let st32 = |b: &mut [u8], v: u32| b[..4].copy_from_slice(&v.to_le_bytes());
    let mut d = [0u8; 8];
    st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 3);
    st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
    host.write_gpa(dir_gpa, &d).unwrap();
    // Depth-1 table: the live translation of GVA page 9 is `data0`.
    let mut pte = [0u8; 4];
    st32(&mut pte, (data0 >> PAGE_SHIFT_X86) as u32);
    host.write_gpa(root_gpa + u64::from(BACKING_PFN) * 4, &pte)
        .unwrap();

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    assert!(state.define_task(1, page, 2));
    {
        let m = state.mappings.entry(9).or_default();
        m.mapped = true;
        // The generation still matches the window's, so this test cannot
        // pass through the sibling's branch: the only thing wrong is where
        // the cached entry points.
        m.map_generation = 1;
        m.page_entries = vec![(0x77u32 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
        m.type4_walk = Some(Type4Walk {
            task_id: 1,
            backing_pfn: BACKING_PFN,
            map_generation: 1,
        });
    }
    state
        .compute_deferred_flush
        .insert(key(9, 0, 256), render_owner(1));
    let before = crate::runtime::drain::store_route_count("rendflush_page_drift");
    let cap = crate::observe::FailCapture::start();
    assert!(
        !super::flush_intersecting(&mut state, &mut host, 9, 0, u64::MAX),
        "a window whose pages moved must report the loss"
    );
    let line = cap.one("deferred_flush_lost");
    assert!(
        line.contains("kind=render") && line.contains("reason=mapping_page_drift"),
        "the re-pointed pages must be the stated refusal, not the generation: {line}"
    );
    assert!(
        state.compute_deferred_flush.is_empty(),
        "the trigger consumes the window it took, so the obligation is gone"
    );
    assert_eq!(
        crate::runtime::drain::store_route_count("rendflush_page_drift"),
        before + 1,
        "the page-drift loss must be counted on the store-route census"
    );
}

/// A window landing over pages the guest wrote preserves nothing and says
/// so; one landing over untouched pages preserves nothing and stays quiet.
///
/// Both halves are the test. The report has to be keyed on the guest write
/// and not on the landing — the writeback runs on every landing and the
/// interesting population is the subset the guest also wrote — so the
/// untouched arm is what makes the reporting arm mean anything.
///
/// This test asserted the opposite of its first half until the rail was
/// bisected on live boots: the preserving behaviour turned the screen black
/// (0 of 14 rounds, against 3 of 4 and 2 of 4 clean on the two commits
/// before it), because `page_gen` is stamped at the harvest and not at the
/// write, so a store the device's own render superseded can still be named
/// "written since the Store". See
/// [`super::witness::note_render_flush_over_guest_write`], which returns nothing at
/// all now — "preserves nothing" is in its signature and no longer only in
/// this assertion.
#[cfg(feature = "backend-vulkan")]
#[test]
fn a_render_window_landing_over_guest_writes_reports_them_and_preserves_nothing() {
    use crate::runtime::host::{FakeHost, HostOps};
    let page = 1u64 << PAGE_SHIFT_X86;
    for guest_wrote in [false, true] {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        let token = host
            .track_guest_writes(&[page], 1usize << PAGE_SHIFT_X86)
            .unwrap();
        let stamped = host.guest_write_gen(token).unwrap();
        let m = state.mappings.entry(9).or_default();
        m.mapped = true;
        m.map_generation = 1;
        // The one tracked page IS this surface's page 0, so the report the
        // host gives back has somewhere in the mapping to land.
        let pfn = (page >> PAGE_SHIFT_X86) as u32;
        m.page_entries = vec![
            (pfn << crate::contract::iosurface_pages::PAGE_ENTRY_PFN_SHIFT)
                | crate::contract::iosurface_pages::PAGE_ENTRY_VALID,
        ];
        m.guest_write_token = token;
        m.guest_write_token_gen = 1;
        m.guest_write_gen_at_store = stamped;
        if guest_wrote {
            host.guest_wrote_page(page);
        }
        let cap = crate::observe::FailCapture::start();
        super::witness::note_render_flush_over_guest_write(&state, &host, &key(9, 0, 256));
        let clobbers: Vec<String> = cap
            .lines()
            .into_iter()
            .filter(|l| l.split_whitespace().next() == Some("deferred_flush_clobber"))
            .collect();
        assert_eq!(
            clobbers.len(),
            usize::from(guest_wrote),
            "guest_wrote={guest_wrote} must decide whether the loss is reported: {clobbers:?}"
        );
    }
}

/// All the report knows is that the generation moved since the stamp. It
/// carries no ordering against the Store, so the line must not claim one.
///
/// Why that matters is at the shim, not here. `reims_vgpu_dirty_gen` answers
/// with the generation as of the last harvest and only marks a read as owed;
/// `reims_vgpu_dirty_harvest` returns early unless one is owed, and runs at
/// the drain tail. A guest store in a tranche whose harvest has not yet run
/// is therefore stamped into a generation that moves *after* the Store, and
/// arrives here indistinguishable from a store that genuinely followed it —
/// the same unsoundness that made preserving the pages black the screen.
///
/// `FakeHost` cannot stage that: `guest_wrote_page` is both the write and
/// the observation, so the harvest lag has no double here and this fixture
/// does not reproduce it. What it does pin is the consequence — the verdict
/// is a bare generation comparison — and that the emitted line stops short of
/// an ordering claim. `note_render_flush_over_guest_write`'s doc used to make
/// that claim two paragraphs after `render_flush_guest_written_ranges` stated
/// the opposite rule.
#[cfg(feature = "backend-vulkan")]
#[test]
fn the_clobber_report_claims_no_ordering_against_the_store() {
    use crate::runtime::host::{FakeHost, HostOps};
    let page = 1u64 << PAGE_SHIFT_X86;
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
    let token = host
        .track_guest_writes(&[page], 1usize << PAGE_SHIFT_X86)
        .unwrap();

    let stamped = host.guest_write_gen(token).unwrap();
    // The only input the verdict has: the generation is no longer `stamped`.
    // Whether the store behind it preceded or followed the Store is exactly
    // what neither this fixture nor the product witness can say.
    host.guest_wrote_page(page);

    let m = state.mappings.entry(9).or_default();
    m.mapped = true;
    m.map_generation = 1;
    m.page_entries = vec![
        (((page >> PAGE_SHIFT_X86) as u32)
            << crate::contract::iosurface_pages::PAGE_ENTRY_PFN_SHIFT)
            | crate::contract::iosurface_pages::PAGE_ENTRY_VALID,
    ];
    m.guest_write_token = token;
    m.guest_write_token_gen = 1;
    m.guest_write_gen_at_store = stamped;

    let cap = crate::observe::FailCapture::start();
    super::witness::note_render_flush_over_guest_write(&state, &host, &key(9, 0, 256));
    let clobbers: Vec<String> = cap
        .lines()
        .into_iter()
        .filter(|l| l.split_whitespace().next() == Some("deferred_flush_clobber"))
        .collect();
    assert_eq!(
        clobbers.len(),
        1,
        "the witness cannot order the write against the Store, so it reports \
         either way — the line is an upper bound, not a defect count"
    );
    assert!(
        !clobbers[0].contains("wrote pages of this surface after"),
        "the line must not claim an ordering the witness cannot establish: {:?}",
        clobbers[0]
    );
}

/// A `Resident` window whose resident no longer vouches for the frame
/// declines, and leaves the guest's pages exactly as it found them.
///
/// This is the whole safety argument for the `skip_readback` rail. An `Owned`
/// window carries its pixels and cannot be wrong about them; a `Resident`
/// window carries only a *claim* that a GPU image still holds them, and the
/// epoch is what tests the claim. `registry_mark_ready` clears a slot's
/// `content_epoch` on every draw into it, so a mismatch means another layer
/// rendered over this surface after the Store that armed the window — and
/// writing then lands that other layer's pixels in these pages, which is the
/// black/torn-layer class rather than a merely stale frame.
///
/// No engine is initialized here, so the registry has no slot at the
/// reconstructed identity at all, and the refusal this asserts is therefore
/// `resident_absent` rather than `resident_epoch_cleared`. Those two used to
/// be one `reason=resident_epoch_drift` with `live=None`, and separating
/// them is what `engine::ResidentContent` exists for: an un-stamped slot is
/// expected traffic, a missing one cannot happen to a pinned identity and
/// means the arm and the flush name the target differently.
///
/// The assertion that matters either way is the *guest memory*: a decline
/// that still wrote would pass a log-only check.
#[cfg(feature = "backend-vulkan")]
#[test]
fn a_resident_window_that_cannot_be_vouched_for_declines_without_writing() {
    use crate::contract::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
    use crate::runtime::host::{FakeHost, HostMemory};
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
    let page = 1u64 << PAGE_SHIFT_X86;
    let gpa = 0x4500_0000u64;
    host.map_range(gpa, page as usize, 0);
    // A recognizable pre-Store pattern, so "did not write" is checkable
    // rather than indistinguishable from a zeroed page.
    let pre = [0x5Cu8; 256];
    host.write_gpa(gpa, &pre).unwrap();
    state.map_surface(9);
    {
        let m = state.mappings.get_mut(&9).unwrap();
        m.mapped = true;
        m.map_generation = 1;
        m.has_geom = true;
        m.width = 4;
        m.height = 4;
        m.format = crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;
        m.page_entries =
            vec![(((gpa >> PAGE_SHIFT_X86) as u32) << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
    }
    state.compute_deferred_flush.insert(
        key(9, 0, 256),
        crate::model::DeferredOwner::Render {
            armed_seq: 1,
            armed_stamp_seq: 0,
            source: crate::model::RenderWindowSource::Resident { epoch: 7 },
        },
    );
    let cap = crate::observe::FailCapture::start();
    assert!(
        !super::flush_intersecting(&mut state, &mut host, 9, 0, u64::MAX),
        "a window whose resident cannot be vouched for must report the loss"
    );
    let line = cap.one("deferred_flush_lost");
    assert!(
        line.contains("kind=render")
            && line.contains("reason=resident_absent")
            && line.contains("want=7"),
        "the epoch witness must be the stated refusal, naming which kind of \
         absence it was and the value it wanted: {line}"
    );
    let mut after = [0u8; 256];
    host.read_gpa(gpa, &mut after).unwrap();
    assert_eq!(
        &after[..],
        &pre[..],
        "a declined resident window must leave the guest's own bytes untouched"
    );
    assert!(
        state.compute_deferred_flush.is_empty(),
        "the trigger must consume the window it took"
    );
}

/// The identity a `Resident` window's flush rebuilds from its key is the one
/// the draw rendered into, pinned and stamped.
///
/// Four separate places name this slot — the draw's `target_identity`, the
/// arm's `pin_resident_target`, the arm's `stamp_resident_content_epoch`, and
/// the flush's `read_target` — and all four resolve through
/// `present_identity::surface_identity` except the last, which has only the
/// key. If those two spellings ever disagree the pin protects one image while
/// the flush reads another: the frame is silently the wrong one, and no
/// assertion in the crate is watching for it because both lookups *succeed*.
#[cfg(feature = "backend-vulkan")]
#[test]
fn a_render_windows_key_rebuilds_the_identity_the_draw_rendered_into() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    for generation in [1u32, 5, u32::MAX] {
        let m = state.mappings.entry(9).or_default();
        m.map_generation = generation;
        let mut k = key(9, 0, 256);
        k.map_generation = generation;
        assert_eq!(
            super::render_window_identity(&k),
            crate::runtime::present_identity::surface_identity(&state, 9, k.width, k.height),
            "the flush's rebuilt identity must equal the one the draw and the pin used"
        );
    }
}

/// A render window lands its own pixels even when `surface_cache` has moved
/// on to another geometry for the same mapping.
///
/// The flush used to source its bytes from
/// `surface_cache::get(mapping_id, key.width, key.height)`, and that cache
/// holds exactly one entry per mapping. A guest that re-Stores the surface at
/// a new size therefore orphaned every window still armed at the old one:
/// the flush missed, emitted `deferred_flush_lost reason=cache_miss` and the
/// guest kept its stale pixels. One boot lost 15 whole layers that way —
/// including a 1920x1080 desktop surface and a 1920x24 menu bar — which on
/// screen is a compositing layer rendering solid black.
#[cfg(feature = "backend-vulkan")]
#[test]
fn a_render_window_lands_its_own_pixels_after_the_cache_moved_geometry() {
    use crate::contract::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
    use crate::runtime::host::{FakeHost, HostMemory};
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
    let page = 1u64 << PAGE_SHIFT_X86;
    let gpa = 0x4400_0000u64;
    host.map_range(gpa, page as usize, 0);
    state.map_surface(9);
    {
        let m = state.mappings.get_mut(&9).unwrap();
        m.mapped = true;
        m.map_generation = 1;
        m.has_geom = true;
        m.width = 4;
        m.height = 4;
        m.format = crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;
        m.page_entries =
            vec![(((gpa >> PAGE_SHIFT_X86) as u32) << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
    }
    // The window's own frame — every byte 0xA7.
    let frame = vec![0xA7u8; 4 * 4 * 4];
    state.compute_deferred_flush.insert(
        key(9, 0, 256),
        crate::model::DeferredOwner::Render {
            armed_seq: 1,
            armed_stamp_seq: 0,
            source: crate::model::RenderWindowSource::Owned(std::sync::Arc::new(frame.clone())),
        },
    );
    // A later Store re-Stored this mapping at 8x8, replacing the one cache
    // entry it has. The 4x4 window above is now unreachable through it.
    crate::runtime::surface_cache::store(&mut state, 9, 8, 8, vec![0x11u8; 8 * 8 * 4]);

    let cap = crate::observe::FailCapture::start();
    assert!(
        super::flush_intersecting(&mut state, &mut host, 9, 0, u64::MAX),
        "a window carrying its own pixels is always landable"
    );
    assert!(
        cap.lines()
            .iter()
            .all(|l| !l.contains("deferred_flush_lost")),
        "nothing may be lost: {:?}",
        cap.lines()
    );
    // The guest side is row-strided at the mapping's own bytes-per-row, so
    // read it the way the writeback wrote it.
    let (base_off, bpr, _) = {
        let m = state.mappings.get(&9).unwrap();
        crate::runtime::mapping_write::type11_sample_window(
            m,
            4,
            4,
            crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM,
        )
        .expect("the mapping has a type-11 sample window")
    };
    for y in 0..4u64 {
        let mut row = [0u8; 4 * 4];
        host.read_gpa(gpa + base_off + y * bpr as u64, &mut row)
            .unwrap();
        assert_eq!(
            &row[..],
            &frame[(y as usize) * 16..(y as usize) * 16 + 16],
            "row {y} of the guest pages must hold the window's frame, not the cache's"
        );
    }
}

/// A render window fully covered by a later writer is *dropped*, not
/// flushed, and dropping it takes its alias-index refs with it.
///
/// This is the difference between a deferral and a rescheduling. A guest
/// compositing into one surface re-Stores the identical guest range every
/// frame, so the previous window always intersects the new one; landing it
/// there performs exactly the guest write the rail exists to skip, once per
/// Store, and `surface_flush` would track `surface_deferred` at a ratio of 1.
///
/// The alias-index half is the part that is easy to get wrong: taking the
/// entry with a bare `remove` leaves `deferred_alias_pages` holding page
/// refs for a mapping with no windows left, and the raw-GVA sampling guard
/// then walks pages nothing defers on.
#[cfg(feature = "backend-vulkan")]
#[test]
fn a_superseded_render_window_is_dropped_and_releases_its_alias_pages() {
    use crate::contract::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    state.map_surface(9);
    {
        let m = state.mappings.get_mut(&9).unwrap();
        m.page_entries = vec![(0x300 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
    }
    let k = key(9, 0, 256);
    state.compute_deferred_flush.insert(k, render_owner(1));
    state.index_deferred_alias_pages(9);
    assert!(
        state.deferred_alias_pages.contains_key(&9),
        "arming indexes the mapping's pages for the raw-GVA guard"
    );

    let released = super::supersede_covered_render_windows(&mut state, &k);
    assert_eq!(
        released.iter().map(|(k, _)| *k).collect::<Vec<_>>(),
        vec![k],
        "the exact key is the one taken"
    );
    assert!(state.compute_deferred_flush.is_empty());
    assert!(
        !state.deferred_alias_pages.contains_key(&9),
        "the last window leaving must drop the mapping's alias-page refs"
    );
}

/// The other half of dropping a superseded window: a `Resident` one holds a
/// counted registry pin, and the supersede is one of the exits
/// `release_window_pin` names.
///
/// The arm site got this wrong. It took each covered window with a bare
/// `take_deferred_flush_window_exact` and discarded it, so every composite
/// Store on a repainted surface leaked one pin — and since the re-Store
/// carries the same key, it is the same slot's `pin_count` climbing without
/// bound until nothing can ever reclaim it. `unpin_resident_target` is a
/// silent no-op with no engine here, so the assertion is on the *identity*
/// the release named: it has to be rebuilt from the superseded window's own
/// key, since a covered sibling may carry a different geometry over the same
/// guest range.
#[cfg(feature = "backend-vulkan")]
#[test]
fn superseding_a_resident_window_releases_the_pin_it_held() {
    use crate::backend::vulkan::engine::TargetIdentity;
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut k = key(9, 0, 256);
    k.map_generation = 3;
    state.compute_deferred_flush.insert(
        k,
        crate::model::DeferredOwner::Render {
            armed_seq: 1,
            armed_stamp_seq: 0,
            source: crate::model::RenderWindowSource::Resident { epoch: 11 },
        },
    );

    let released = super::supersede_covered_render_windows(&mut state, &k);
    assert_eq!(
        released,
        vec![(
            k,
            Some(TargetIdentity::Surface {
                id: 9,
                width: k.width,
                height: k.height,
                generation: 3,
            })
        )],
        "a resident window's pin must be released, under the identity its own key names"
    );

    // An `Owned` window holds nothing on the GPU, so `None` is the answer and
    // not a missed release — unpinning for one would name a slot the arm never
    // pinned and succeed silently.
    state.compute_deferred_flush.insert(k, render_owner(2));
    assert_eq!(
        super::supersede_covered_render_windows(&mut state, &k),
        vec![(k, None)],
        "an owned window releases nothing"
    );
}

/// Superseding one window must not disturb a sibling covering a different
/// guest range on the same mapping — that one holds bytes the new Store does
/// not write, and dropping it would lose them.
#[cfg(feature = "backend-vulkan")]
#[test]
fn superseding_one_window_leaves_a_disjoint_sibling_armed() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    let covered = key(9, 0, 256);
    let sibling = key(9, 256, 512);
    state
        .compute_deferred_flush
        .insert(covered, render_owner(1));
    state
        .compute_deferred_flush
        .insert(sibling, render_owner(2));

    assert_eq!(
        super::supersede_covered_render_windows(&mut state, &covered).len(),
        1
    );
    assert!(
        state.compute_deferred_flush.contains_key(&sibling),
        "a different range is a different obligation"
    );
    assert_eq!(state.compute_deferred_flush.len(), 1);
}

/// Teardown must name the render rail, because the two rails pin different
/// registries and the drop is where the pin is released.
///
/// Unpinning storage for a render window succeeds silently and leaves the
/// target resident pinned for the life of the boot — a display-sized image
/// per window, which is the "~260 stale residents (~516 MiB)" shape. The
/// slug on this line is the only always-on evidence that the right registry
/// was chosen.
#[test]
fn dropping_a_render_window_reports_the_render_rail() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    state
        .compute_deferred_flush
        .insert(key(9, 0, 256), render_owner(7));
    state.compute_deferred_flush.insert(
        key(9, 256, 512),
        crate::model::DeferredOwner::Storage {
            generation: 3,
            armed_stamp_seq: 0,
        },
    );
    let cap = crate::observe::FailCapture::start();
    super::drop_windows(&mut state, 9, "unit");
    let lines: Vec<String> = cap
        .lines()
        .into_iter()
        .filter(|l| l.split_whitespace().next() == Some("deferred_flush_dropped"))
        .collect();
    assert_eq!(lines.len(), 2, "both windows drop: {lines:?}");
    assert!(
        lines.iter().any(|l| l.contains("owner=render")),
        "the render window must say so: {lines:?}"
    );
    assert!(
        lines.iter().any(|l| l.contains("owner=compute")),
        "the compute window must say so: {lines:?}"
    );
    assert!(state.compute_deferred_flush.is_empty());
}

/// `condemn_surface_backing` keeps a mapping's deferred windows on purpose:
/// `DeleteIOSurfaceBacking2` may name a prior incarnation of a recycled id,
/// and `mapper::resolve` settles it later by fingerprint compare. A flush
/// trigger arriving inside that undecided window must therefore leave the
/// obligation armed — consuming it destroys the very thing the fingerprint
/// decision exists to reprieve, and reports a loss the flush did not cause.
#[test]
fn flush_holds_windows_while_the_backing_is_condemned() {
    use crate::contract::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
    use crate::runtime::host::FakeHost;
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
    let k = key(9, 0, 4096);
    state.map_surface(9);
    {
        let m = state.mappings.get_mut(&9).unwrap();
        m.map_generation = 2;
        m.page_entries = vec![(0x300 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
    }
    state.compute_deferred_flush.insert(
        k,
        crate::model::DeferredOwner::Storage {
            generation: 3,
            armed_stamp_seq: 0,
        },
    );
    // The guest deletes the backing; the window is kept for the fingerprint
    // decision and the page list moves to `condemned_entries`.
    assert!(state.condemn_surface_backing(9));
    assert!(state.mapping_backing_condemned(9));
    let ok = super::flush_intersecting(&mut state, &mut host, 9, 0, u64::MAX);
    assert!(ok, "an undecided window is not a loss");
    assert!(
        state.compute_deferred_flush.contains_key(&k),
        "the window must survive for mapper::resolve to reprieve or drop"
    );
}

/// A window whose mapping the guest has since declared it owns must not
/// land, and the refusal must name itself.
///
/// The window's frame is what the device rendered *before* the guest's CPU
/// write. Landing it replaces the guest's own bytes over the full attachment
/// extent with a copy the guest has already said is stale. Every other guard
/// on this path asks where the bytes would land; this one asks whether they
/// are owed at all, which is why it runs first.
#[cfg(feature = "backend-vulkan")]
#[test]
fn a_window_the_guest_superseded_is_refused_by_name() {
    use crate::runtime::host::FakeHost;
    use crate::runtime::resource_validity::{apply, ValiditySite};
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
    let k = key(9, 0, 4096);
    let m = state.mappings.entry(9).or_default();
    m.mapped = true;
    m.map_generation = k.map_generation;
    state.compute_deferred_flush.insert(
        k,
        crate::model::DeferredOwner::Render {
            armed_seq: 0,
            armed_stamp_seq: 0,
            source: crate::model::RenderWindowSource::Owned(std::sync::Arc::new(vec![0u8; 4096])),
        },
    );
    // The device published this surface's pixels, and the guest then claimed
    // a CPU write to it — so the guest's bytes are the newer ones.
    state.note_surface_content_published(9);
    apply(
        &mut state,
        0,
        9,
        crate::runtime::decode::fifo::InvalidateValidityOps {
            clear_host_valid: 1,
            set_host_valid: 0,
            clear_guest_valid: 0,
            set_guest_valid: 0,
        },
        ValiditySite::ExecTable,
    );
    // The claim also drops the window, which is the repair upstream of this
    // gate; re-arm it so the gate itself is what this exercises.
    state.compute_deferred_flush.insert(
        k,
        crate::model::DeferredOwner::Render {
            armed_seq: 0,
            armed_stamp_seq: 0,
            source: crate::model::RenderWindowSource::Owned(std::sync::Arc::new(vec![0u8; 4096])),
        },
    );

    let cap = crate::observe::FailCapture::start();
    let ok = super::flush_intersecting(&mut state, &mut host, 9, 0, u64::MAX);
    assert!(!ok, "a refused writeback is a reported loss, not a success");
    let line = cap.one("deferred_flush_lost");
    assert!(
        line.contains("reason=host_copy_superseded"),
        "the refusal must name itself: {line}"
    );
    assert!(state.compute_deferred_flush.is_empty());
}

#[test]
fn flush_intersecting_takes_windows_and_reports_loss() {
    use crate::runtime::host::FakeHost;
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
    // Window over an unmapped mapping: the flush must fail closed
    // (fail-visible loss), remove the window, and return false.
    state.compute_deferred_flush.insert(
        key(9, 0, 4096),
        crate::model::DeferredOwner::Storage {
            generation: 3,
            armed_stamp_seq: 0,
        },
    );
    let ok = super::flush_intersecting(&mut state, &mut host, 9, 0, u64::MAX);
    assert!(!ok, "lost window must report failure");
    assert!(
        state.compute_deferred_flush.is_empty(),
        "taken windows never return to the map"
    );
    // Disjoint mapping id: untouched.
    state.compute_deferred_flush.insert(
        key(10, 0, 4096),
        crate::model::DeferredOwner::Storage {
            generation: 3,
            armed_stamp_seq: 0,
        },
    );
    assert!(super::flush_intersecting(
        &mut state,
        &mut host,
        11,
        0,
        u64::MAX
    ));
    assert_eq!(state.compute_deferred_flush.len(), 1);
}

/// A raw task-GVA span whose physical pages alias a deferred window's
/// mapping pages must take (and attempt to flush) that window; a window
/// on non-aliased pages stays. Locks the boot-18 linear_sample poisoning
/// channel: GVA reads bypassing the mapping-keyed hooks.
#[test]
fn gva_alias_takes_only_aliased_windows() {
    use crate::contract::endian::st32;
    use crate::contract::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
    use crate::runtime::host::{FakeHost, HostMemory};
    let page_shift = PAGE_SHIFT_X86;
    let mut host = FakeHost::new();
    // Task 1 directory at pfn 2 → root table pfn 3 → gva page 0 =
    // pfn 0x2000. Data pfns sit past the default task object list
    // (pfn 1 + 0x100000 slots = 4096 pages), which the mapping
    // control-page collision check treats as reserved.
    let dir_gpa = 2u64 << page_shift;
    let root_gpa = 3u64 << page_shift;
    let data_gpa = 0x2000u64 << page_shift;
    host.map_range(dir_gpa, 0x20, 0);
    host.map_range(root_gpa, 0x1000, 0);
    host.map_range(data_gpa, 0x1000, 0xab);
    let mut d = [0u8; 8];
    st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 3);
    st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
    host.write_gpa(dir_gpa, &d).unwrap();
    let mut pte = [0u8; 4];
    st32(&mut pte, 0x2000);
    host.write_gpa(root_gpa, &pte).unwrap();

    let mut state = DeviceState::new(DeviceId(1), page_shift);
    assert!(state.define_task(1, 0x1000, 2));
    // Mapping 9 is backed by pfn 0x2000 (the page the GVA span resolves
    // to); mapping 10 is backed by pfn 0x2001 (disjoint).
    let page_entry = |pfn: u32| (pfn << 2) | 1;
    for (mid, pfn) in [(9u32, 0x2000u32), (10, 0x2001)] {
        let m = state.mappings.entry(mid).or_default();
        m.mapped = true;
        m.page_entries = vec![page_entry(pfn)];
    }
    let ckey = |mapping_id: u32| key(mapping_id, 0, 0x1000);
    state.compute_deferred_flush.insert(
        ckey(9),
        crate::model::DeferredOwner::Storage {
            generation: 3,
            armed_stamp_seq: 0,
        },
    );
    state.compute_deferred_flush.insert(
        ckey(10),
        crate::model::DeferredOwner::Storage {
            generation: 3,
            armed_stamp_seq: 0,
        },
    );
    // Product defer sites index pages at defer time.
    state.index_deferred_alias_pages(9);
    state.index_deferred_alias_pages(10);
    assert_eq!(state.deferred_alias_pages.len(), 2);

    super::flush_intersecting_task_gva(&mut state, &mut host, 1, 0, 0x100);
    assert!(
        !state.compute_deferred_flush.contains_key(&ckey(9)),
        "aliased window must be taken for flush"
    );
    assert!(
        state.compute_deferred_flush.contains_key(&ckey(10)),
        "non-aliased window must stay deferred"
    );
    assert!(
        !state.deferred_alias_pages.contains_key(&9),
        "alias index must drop with the mapping's last window"
    );
    assert!(
        state.deferred_alias_pages.contains_key(&10),
        "alias index for the untouched mapping must stay"
    );
}

/// SynchronizeResources choke point: the guest names a mapping it is
/// about to CPU-read; every deferred window on it — mapping-keyed
/// (compute) and linear windows whose defer-time page index aliases the
/// mapping's physical pages — must be taken for flush.
/// Windows on disjoint mappings/pages stay deferred. Locks the
/// boot-25 black-wallpaper class (guest-CPU composite of stale pages).
#[test]
fn guest_read_flush_takes_keyed_and_linear_alias_windows() {
    use crate::runtime::host::FakeHost;
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
    let page_entry = |pfn: u32| (pfn << 2) | 1;
    for (mid, pfn) in [(9u32, 0x2000u32), (10, 0x2001)] {
        let m = state.mappings.entry(mid).or_default();
        m.mapped = true;
        m.page_entries = vec![page_entry(pfn)];
    }
    state.compute_deferred_flush.insert(
        key(9, 0, 256),
        crate::model::DeferredOwner::Storage {
            generation: 3,
            armed_stamp_seq: 0,
        },
    );
    let disjoint = key(10, 0, 0x1000);
    state.compute_deferred_flush.insert(
        disjoint,
        crate::model::DeferredOwner::Storage {
            generation: 3,
            armed_stamp_seq: 0,
        },
    );
    // Linear windows never name the mapping: one aliases mapping 9's
    // physical page, one sits on a disjoint page.
    let mut lin_aliased = key(0, 0, 0x1000);
    lin_aliased.texture_ref = 42;
    let mut lin_disjoint = key(0, 0, 0x1000);
    lin_disjoint.texture_ref = 43;
    let aliased_pages: std::collections::HashSet<u64> =
        [(0x2000u64) << PAGE_SHIFT_X86].into_iter().collect();
    let disjoint_pages: std::collections::HashSet<u64> =
        [(0x3000u64) << PAGE_SHIFT_X86].into_iter().collect();
    state.arm_linear_deferred_window(lin_aliased, 1, aliased_pages);
    state.arm_linear_deferred_window(lin_disjoint, 1, disjoint_pages);

    // No windows on mapping 11: clean no-op.
    assert_eq!(
        super::flush_mapping_for_guest_read(&mut state, &mut host, 11),
        (true, 0)
    );

    let (ok, flushed) = super::flush_mapping_for_guest_read(&mut state, &mut host, 9);
    // Nothing is engine-pinned / host-mapped in this fixture, so every
    // flush reports a fail-visible loss — but every aliased window must
    // still be taken (obligations never return to the maps).
    assert!(!ok, "losses must be reported");
    assert_eq!(flushed, 2, "compute@9 + linear alias");
    assert!(!state.compute_deferred_flush.contains_key(&key(9, 0, 256)));
    assert!(
        state.compute_deferred_flush.contains_key(&disjoint),
        "disjoint mapping's window must stay deferred"
    );
    assert!(
        !state.linear_deferred_flush.contains_key(&lin_aliased),
        "page-aliased linear window must be taken"
    );
    assert!(
        state.linear_deferred_flush.contains_key(&lin_disjoint),
        "disjoint-page linear window must stay deferred"
    );
}

fn gva_entry(task_id: u32, w: u32, h: u32, pages: &[u64]) -> crate::model::GvaDeferredEntry {
    crate::model::GvaDeferredEntry {
        task_id,
        texture_ref: 5,
        producer_object_type: 2,
        width: w,
        height: h,
        row_stride: w * 4,
        format: 0x46,
        armed_seq: 0,
        armed_stamp_seq: 0,
        pages: pages.iter().copied().collect(),
        alloc_gen: 0,
    }
}

/// A linear compute-storage window records the fence it was armed under, and
/// a re-arm records the fence it was re-armed under.
///
/// This rail writes a raw task GVA with no mapping incarnation to name, so
/// the only thing that can say a landing is late is the stamp counter at arm
/// time. Without it every linear landing is unscoreable, which is the state
/// `6bc2220` left it in while clearing the two rails that *do* carry an
/// allocation identity.
#[test]
fn a_linear_window_records_the_fence_it_was_armed_under() {
    use crate::model::{ComputeStorageResidencyKey, DeviceState, PAGE_SHIFT_X86};
    let mut state = DeviceState::new(crate::model::DeviceId(1), PAGE_SHIFT_X86);
    let p = |pfn: u64| pfn << PAGE_SHIFT_X86;
    let key = ComputeStorageResidencyKey::linear(1, 7, 0x4000, 256, 0x1000, 64, 64, 0x46);

    state.completion_stamp_seq = 41;
    state.arm_linear_deferred_window(key, 1, [p(0xA)].into_iter().collect());
    assert_eq!(
        state
            .linear_deferred_flush
            .get(&key)
            .unwrap()
            .armed_stamp_seq,
        41,
        "the window must carry the fence it was armed under"
    );

    // The guest is fenced twice, then the same key re-arms: the window is a
    // NEW obligation and must be scored against the new fence, not the one
    // its predecessor was armed under.
    state.completion_stamp_seq = 43;
    state.arm_linear_deferred_window(key, 2, [p(0xB)].into_iter().collect());
    let window = state.disarm_linear_deferred_window(&key).unwrap();
    assert_eq!(window.armed_stamp_seq, 43, "a re-arm re-stamps the window");
    assert_eq!(window.generation, 2);
    assert_eq!(window.pages, [p(0xB)].into_iter().collect());
}

/// A raw task-GVA span aliasing a deferred GVA render-Store window's
/// pages (or naming its base GVA exactly) must take the window; windows
/// on disjoint pages stay armed. Same channel as the linear windows —
/// GVA reads that bypass every mapping-keyed hook.
#[test]
fn task_gva_alias_takes_gva_store_windows() {
    use crate::contract::endian::st32;
    use crate::contract::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
    use crate::runtime::host::{FakeHost, HostMemory};
    let page_shift = PAGE_SHIFT_X86;
    let mut host = FakeHost::new();
    let dir_gpa = 2u64 << page_shift;
    let root_gpa = 3u64 << page_shift;
    let data_gpa = 0x2000u64 << page_shift;
    host.map_range(dir_gpa, 0x20, 0);
    host.map_range(root_gpa, 0x1000, 0);
    host.map_range(data_gpa, 0x1000, 0xab);
    let mut d = [0u8; 8];
    st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 3);
    st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
    host.write_gpa(dir_gpa, &d).unwrap();
    let mut pte = [0u8; 4];
    st32(&mut pte, 0x2000);
    host.write_gpa(root_gpa, &pte).unwrap();

    let mut state = DeviceState::new(DeviceId(1), page_shift);
    assert!(state.define_task(1, 0x1000, 2));
    // Window A aliases the page the span resolves to; window B does not.
    state.arm_gva_deferred_window(0x9000_0000, gva_entry(1, 4, 4, &[0x2000u64 << page_shift]));
    state.arm_gva_deferred_window(0x9100_0000, gva_entry(1, 4, 4, &[0x3000u64 << page_shift]));

    super::flush_intersecting_task_gva(&mut state, &mut host, 1, 0, 0x100);
    // No engine in this fixture: the flush reports a fail-visible loss,
    // but the aliased window must be taken (obligations never return).
    assert!(
        !state.gva_deferred_flush.contains_key(&0x9000_0000),
        "page-aliased GVA window must be taken"
    );
    assert!(
        state.gva_deferred_flush.contains_key(&0x9100_0000),
        "disjoint GVA window must stay armed"
    );

    // Exact-base fast path: a read naming the window's own GVA takes it
    // without any page walk.
    super::flush_intersecting_task_gva(&mut state, &mut host, 1, 0x9100_0000, 0x10);
    assert!(
        !state.gva_deferred_flush.contains_key(&0x9100_0000),
        "exact-base read must take the window"
    );
}

/// PT builder shared by the alias-walk tests: task 1's GVA `0..0x1000`
/// resolves to data page `0x2000<<shift`, and page `0x3000<<shift` is mapped
/// but unreferenced so a test can point a PTE at it. Returns the root PTE
/// GPA so the caller can remap and simulate a task page-table change.
fn alias_pt_fixture() -> (crate::runtime::host::FakeHost, DeviceState, u64, u32) {
    use crate::contract::endian::st32;
    use crate::contract::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
    use crate::runtime::host::{FakeHost, HostMemory};
    let page_shift = PAGE_SHIFT_X86;
    let mut host = FakeHost::new();
    let dir_gpa = 2u64 << page_shift;
    let root_gpa = 3u64 << page_shift;
    host.map_range(dir_gpa, 0x20, 0);
    host.map_range(root_gpa, 0x1000, 0);
    host.map_range(0x2000u64 << page_shift, 0x1000, 0xab);
    host.map_range(0x3000u64 << page_shift, 0x1000, 0xcd);
    let mut d = [0u8; 8];
    st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 3);
    st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
    host.write_gpa(dir_gpa, &d).unwrap();
    let mut pte = [0u8; 4];
    st32(&mut pte, 0x2000);
    host.write_gpa(root_gpa, &pte).unwrap();
    let mut state = DeviceState::new(DeviceId(1), page_shift);
    assert!(state.define_task(1, 0x1000, 2));
    (host, state, root_gpa, page_shift)
}

/// A large bind's alias must be found wherever it sits, not only where a
/// sample point happens to land.
///
/// This walk used to sample every 16th page once a span passed 64 pages, on
/// the stated grounds that "real aliases are same-surface, so the first page
/// hits". Measured on the rail, no alias hit page 0 — the three observed
/// landed at 16, 32 and 48 of 127- and 256-page spans, i.e. partial overlaps
/// somewhere below each sample point. So the miss window was live, and this
/// is what falls through it: a 65-page bind overlapping a window on page 1
/// alone, which a stride of 16 steps straight over.
#[test]
fn a_large_bind_alias_is_found_off_the_sample_points() {
    use crate::contract::endian::st32;
    use crate::runtime::host::HostMemory;
    let (mut host, mut state, root_gpa, page_shift) = alias_pt_fixture();
    // 65 pages, so the old rule ran a strided walk. Page i -> pfn 0x4000+i.
    const N: u64 = 65;
    for i in 0..N {
        let pfn = 0x4000 + i;
        host.map_range(pfn << page_shift, 0x1000, 0);
        let mut pte = [0u8; 4];
        st32(&mut pte, pfn as u32);
        host.write_gpa(root_gpa + 4 * i, &pte).unwrap();
    }
    // The deferred window covers page 1 and nothing else. A stride-16 walk
    // visits 0, 16, 32, 48, 64 — never 1.
    state.arm_gva_deferred_window(0x9100_0000, gva_entry(1, 4, 4, &[0x4001u64 << page_shift]));
    super::flush_intersecting_task_gva(&mut state, &mut host, 1, 0, N << page_shift);
    assert!(
        !state.gva_deferred_flush.contains_key(&0x9100_0000),
        "a window aliasing page 1 of a 65-page bind must be found and flushed"
    );
}

/// The alias walk is never skipped, so a bind that has already been walked
/// still finds a window armed onto its pages afterwards.
///
/// This used to be answered by the no-intersection memo's cheap page
/// recheck. With the memo gone the same bind simply walks again, and the
/// repeat is what this pins: identical `(task, gva, span)`, walked once with
/// nothing to find, then walked again with a window on its resolved page.
#[test]
fn a_repeat_bind_walks_again_and_takes_a_newly_armed_window() {
    let (mut host, mut state, _root, page_shift) = alias_pt_fixture();
    // Disjoint window on page 0x3000 keeps the deferred set non-empty (so
    // the early-out does not answer) but never aliases the [0,0x100) bind,
    // which resolves to page 0x2000.
    state.arm_gva_deferred_window(0x9100_0000, gva_entry(1, 4, 4, &[0x3000u64 << page_shift]));
    super::flush_intersecting_task_gva(&mut state, &mut host, 1, 0, 0x100);
    assert!(
        state.gva_deferred_flush.contains_key(&0x9100_0000),
        "a disjoint window must stay armed"
    );

    // Arm a window ON the bind's resolved page and repeat the same bind.
    state.arm_gva_deferred_window(0x9300_0000, gva_entry(1, 4, 4, &[0x2000u64 << page_shift]));
    super::flush_intersecting_task_gva(&mut state, &mut host, 1, 0, 0x100);
    assert!(
        !state.gva_deferred_flush.contains_key(&0x9300_0000),
        "the repeat bind must walk again and take the newly armed window"
    );
    assert!(
        state.gva_deferred_flush.contains_key(&0x9100_0000),
        "the disjoint window must still stay armed"
    );
}

/// A task page-table remap that nothing told the device about is seen by the
/// very next bind.
///
/// The deferred set does not change here and no invalidation hook fires —
/// only the guest's PTE moves, so the bind's pages land under an
/// already-armed window. The memo that used to cache this bind's resolved
/// pages could not see that; it closed the hole with a 1-in-64 sampled walk,
/// which left up to 63 binds reading stale bytes. An unconditional walk has
/// no such hole, and this is the test that would fail if one came back.
#[test]
fn a_task_pt_remap_is_seen_by_the_very_next_bind() {
    use crate::contract::endian::st32;
    use crate::runtime::host::HostMemory;
    let (mut host, mut state, root_gpa, page_shift) = alias_pt_fixture();
    state.arm_gva_deferred_window(0x9100_0000, gva_entry(1, 4, 4, &[0x3000u64 << page_shift]));
    // Disjoint at first: the bind resolves to page 0x2000.
    super::flush_intersecting_task_gva(&mut state, &mut host, 1, 0, 0x100);
    assert!(state.gva_deferred_flush.contains_key(&0x9100_0000));

    // Remap gva page 0 -> 0x3000 directly in guest RAM. No retire, no
    // deferred-set change: the bind now aliases the armed window.
    let mut pte = [0u8; 4];
    st32(&mut pte, 0x3000);
    host.write_gpa(root_gpa, &pte).unwrap();

    super::flush_intersecting_task_gva(&mut state, &mut host, 1, 0, 0x100);
    assert!(
        !state.gva_deferred_flush.contains_key(&0x9100_0000),
        "the bind after the remap must flush the window it now aliases"
    );
}

/// SynchronizeResources choke point: GVA windows whose defer-time pages
/// alias the named mapping's physical pages must be taken for flush.
#[test]
fn guest_read_flush_takes_gva_store_alias_windows() {
    use crate::runtime::host::FakeHost;
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
    let page_entry = |pfn: u32| (pfn << 2) | 1;
    let m = state.mappings.entry(9).or_default();
    m.mapped = true;
    m.page_entries = vec![page_entry(0x2000)];
    state.arm_gva_deferred_window(
        0x9000_0000,
        gva_entry(1, 4, 4, &[0x2000u64 << PAGE_SHIFT_X86]),
    );
    state.arm_gva_deferred_window(
        0x9100_0000,
        gva_entry(1, 4, 4, &[0x3000u64 << PAGE_SHIFT_X86]),
    );

    let declared = crate::runtime::drain::store_route_count("guest_read_declared");
    let landed = crate::runtime::drain::store_route_count("guest_read_landed");
    let dry = crate::runtime::drain::store_route_count("guest_read_dry");

    let (ok, flushed) = super::flush_mapping_for_guest_read(&mut state, &mut host, 9);
    assert!(!ok, "engine-less flush reports the loss");
    assert_eq!(flushed, 1, "exactly the aliased GVA window");
    assert!(!state.gva_deferred_flush.contains_key(&0x9000_0000));
    assert!(state.gva_deferred_flush.contains_key(&0x9100_0000));

    // The declaration rate is the number the demand-driven writeback design
    // turns on, and until these counters existed nothing measured it. Assert
    // them here rather than trusting the wiring: `guest_read_landed` must
    // agree with the returned count, and the dry counter must stay put on a
    // call that landed something — a route on the wrong side of that branch
    // would read as "the guest never declares" and close the question the
    // wrong way. Deltas, not absolutes: the census window is process-wide
    // and other tests share it.
    assert_eq!(
        crate::runtime::drain::store_route_count("guest_read_declared") - declared,
        1,
        "every call must count as a declaration"
    );
    assert_eq!(
        crate::runtime::drain::store_route_count("guest_read_landed") - landed,
        u64::from(flushed),
        "landed must count windows, not calls"
    );
    assert_eq!(
        crate::runtime::drain::store_route_count("guest_read_dry") - dry,
        0,
        "a call that landed a window is not dry"
    );
}

/// A declaration that finds nothing armed counts as dry and lands nothing.
///
/// The complement of the case above, and the one that will dominate live:
/// the fence-bound writeback runs first and empties the windows, so most
/// declarations arrive to an empty set. That reading is only interpretable
/// if "dry" is known to mean *nothing was armed* rather than *the counter
/// never fired*, which is what this pins.
#[test]
fn guest_read_flush_with_nothing_armed_counts_as_dry() {
    use crate::runtime::host::FakeHost;
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
    let m = state.mappings.entry(9).or_default();
    m.mapped = true;
    m.page_entries = vec![(0x2000u32 << 2) | 1];

    let declared = crate::runtime::drain::store_route_count("guest_read_declared");
    let landed = crate::runtime::drain::store_route_count("guest_read_landed");
    let dry = crate::runtime::drain::store_route_count("guest_read_dry");

    let (ok, flushed) = super::flush_mapping_for_guest_read(&mut state, &mut host, 9);
    assert!(ok, "nothing armed is not a failure");
    assert_eq!(flushed, 0, "nothing armed lands nothing");
    assert_eq!(
        crate::runtime::drain::store_route_count("guest_read_declared") - declared,
        1,
        "a dry call is still a declaration"
    );
    assert_eq!(
        crate::runtime::drain::store_route_count("guest_read_dry") - dry,
        1,
        "nothing armed must count as dry"
    );
    assert_eq!(
        crate::runtime::drain::store_route_count("guest_read_landed") - landed,
        0,
        "nothing armed lands nothing"
    );
}

/// A declaration is attributed by whether the fence rail has ever written
/// back that mapping, and the two arms must be exclusive.
///
/// This is the split that decides whether the eager writeback could become
/// demand-driven, and it is the one `guest_read_dry` cannot make — the fence
/// empties the windows before any declaration arrives, so a declaration on a
/// surface the fence just wrote and one on an unrelated surface look
/// identical from the dry count. A mis-wired split here would read as "the
/// guest declares on surfaces we never write back" and close a month of work
/// the wrong way, so both arms are driven through the same fixture.
#[test]
fn a_declaration_is_attributed_to_whether_the_fence_writes_that_mapping() {
    use crate::runtime::host::FakeHost;
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
    for mid in [7u32, 9u32] {
        let m = state.mappings.entry(mid).or_default();
        m.mapped = true;
        m.page_entries = vec![(0x2000u32 << 2) | 1];
    }

    let other0 = crate::runtime::drain::store_route_count("guest_read_on_other_mid");
    let flushed0 = crate::runtime::drain::store_route_count("guest_read_on_flushed_mid");

    // Nothing has been written back yet: both mappings are "other".
    super::flush_mapping_for_guest_read(&mut state, &mut host, 9);
    assert_eq!(
        crate::runtime::drain::store_route_count("guest_read_on_other_mid") - other0,
        1,
        "a mapping the fence never wrote is not a flushed mid"
    );
    assert_eq!(
        crate::runtime::drain::store_route_count("guest_read_on_flushed_mid") - flushed0,
        0,
        "nothing has been fence-flushed yet"
    );

    // Once the fence has landed a window on 9, a declaration on 9 counts as
    // covered and one on 7 still does not.
    state.fence_flushed_mappings.insert(9);
    super::flush_mapping_for_guest_read(&mut state, &mut host, 9);
    super::flush_mapping_for_guest_read(&mut state, &mut host, 7);
    assert_eq!(
        crate::runtime::drain::store_route_count("guest_read_on_flushed_mid") - flushed0,
        1,
        "a declaration on a fence-written mapping must count as covered"
    );
    assert_eq!(
        crate::runtime::drain::store_route_count("guest_read_on_other_mid") - other0,
        2,
        "the arms must be exclusive — every call lands in exactly one"
    );
}

/// Page drift must distinguish the cases it exists to separate, and now
/// **decide** them.
///
/// A probe that reports nothing is indistinguishable from a probe that
/// cannot fire, and this codebase has already paid for three of those. So
/// drive both controls through the same fixture: a window whose GVA still
/// resolves to its armed pages must stay silent and stay writable, and one
/// whose pages moved under it must produce the line and be refused — same
/// task, same geometry, only the armed set differs.
///
/// The decision is asserted alongside the line because they are two separate
/// claims. Logging drift while still writing is exactly what this used to
/// do, and the guest heap corruption that allowed — WindowServer aborting in
/// `small_free_list_remove_ptr_no_clear` — is why it decides now.
/// The mapping-keyed rails get the same reading as the GVA rail, per rail,
/// so a boot can say whether `map_generation` in the key is already enough
/// to make deferral here safe — rather than the two being assumed alike.
#[test]
fn each_mapping_rail_is_scored_against_the_fence_under_its_own_name() {
    use crate::runtime::drain::store_route_count;
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    let k = key(7, 0, 256);

    let render = render_owner(1);
    let storage = crate::model::DeferredOwner::Storage {
        generation: 3,
        armed_stamp_seq: 0,
    };

    // Inside the fence: neither rail may be reported.
    let before = [
        store_route_count("rendw_stamp_outlived"),
        store_route_count("storw_stamp_outlived"),
    ];
    super::witness::note_mapping_window_against_fence(&state, &k, &render);
    super::witness::note_mapping_window_against_fence(&state, &k, &storage);
    assert_eq!(
        [
            store_route_count("rendw_stamp_outlived"),
            store_route_count("storw_stamp_outlived")
        ],
        before,
        "a window landed inside its own fence is the safe case on both rails"
    );

    // Past the fence: each rail reports under its own counter, so a boot can
    // tell a render-Store window from a compute-storage one.
    state.completion_stamp_seq = 5;
    super::witness::note_mapping_window_against_fence(&state, &k, &render);
    assert_eq!(
        [
            store_route_count("rendw_stamp_outlived"),
            store_route_count("storw_stamp_outlived")
        ],
        [before[0] + 1, before[1]],
        "the render rail must not be counted under the storage rail's name"
    );
    super::witness::note_mapping_window_against_fence(&state, &k, &storage);
    assert_eq!(
        [
            store_route_count("rendw_stamp_outlived"),
            store_route_count("storw_stamp_outlived")
        ],
        [before[0] + 1, before[1] + 1],
        "and the storage rail must not be counted under the render rail's"
    );
}

/// The window and the resident it pinned must be the same slot, and the two
/// spellings that name it must agree by construction.
///
/// `arm_surface_resident_store` pins `render_chain_identity`;
/// `flush_render_one` rebuilds `render_window_identity` from
/// `key.width`/`key.height`. Both now read color0's declared geometry —
/// the draw request has only that one — so the geometry axis is closed by
/// construction rather than by this check. It was not: the arm's spelling
/// preferred a whole-request pass extent, and a record whose extent
/// differed from its attachment produced two different
/// `TargetIdentity::Surface` values. The arm pinned one slot, the flush
/// looked up another, `registry_get` missed, and the frame was lost —
/// reported as `live=Absent` — while the pin leaked, because eviction skips
/// pinned slots by design. One measured boot lost ~135 frames at 1920x1080
/// with `live=None`, the whole desktop compositing layer keeping pre-Store
/// bytes in guest memory.
///
/// The first two assertions hold that closure: with one geometry the two
/// spellings are the same value, and a different extent is a different slot
/// that must never be pinned on a window's behalf. The third is the axis
/// still live at runtime, and the reason the equality check stays.
#[cfg(feature = "backend-vulkan")]
#[test]
fn a_window_and_the_resident_it_pins_cannot_be_named_at_two_geometries() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    let k = key(7, 0, 256);
    state.map_surface(k.mapping_id);
    state
        .mappings
        .get_mut(&k.mapping_id)
        .unwrap()
        .map_generation = k.map_generation;
    let from_key = super::render_window_identity(&k);

    // The spelling the arm uses, when the pass extent equals the attachment
    // and the mapping still carries the incarnation the key was built at.
    assert_eq!(
        from_key,
        crate::runtime::present_identity::surface_identity(&state, k.mapping_id, k.width, k.height),
        "with one geometry the two spellings must be the same value, or the \
         rail is broken for every window rather than only the split ones"
    );

    // And when it does not. This is the value the arm would have pinned for
    // a record whose pass extent is larger than its color0 attachment; the
    // flush cannot find it from the key, which is why the arm refuses.
    assert_ne!(
        from_key,
        crate::runtime::present_identity::surface_identity(
            &state,
            k.mapping_id,
            k.width + 1,
            k.height
        ),
        "geometry is part of the resident's shape, so a pass-extent identity \
         is a different slot and must not be pinned on a window's behalf"
    );

    // Generation is the second axis, and it is the one that can move
    // *inside* the arm. `arm_surface_resident_store` takes the identity from
    // the live mapping before it builds the key, and the step between them —
    // `prepare_surface_deferred_window` — lands intersecting windows, whose
    // writeback re-resolves the mapping and can bump `map_generation`. The
    // arm would then pin the pre-bump slot and hand the window a post-bump
    // key. Same miss, same lost frame, same leaked pin.
    crate::model::DeviceState::bump_map_generation(state.mappings.get_mut(&k.mapping_id).unwrap());
    assert_ne!(
        from_key,
        crate::runtime::present_identity::surface_identity(&state, k.mapping_id, k.width, k.height),
        "a generation that moved during the arm names a different slot, and \
         the equality check is what stops the window being armed across it"
    );
}

/// A completion stamp is the guest's licence to free everything it allocated
/// for the work being completed, so the stamp must leave nothing owed to
/// guest RAM. Asserted through [`crate::runtime::drain::write_stamp`] itself
/// rather than against the helper, because the claim that matters is the
/// wiring: a helper nothing calls at the fence is the bug this fixes.
#[cfg(feature = "backend-vulkan")]
#[test]
fn a_completion_stamp_leaves_no_window_still_owing_guest_ram() {
    use crate::runtime::host::FakeHost;
    let page = 1u64 << PAGE_SHIFT_X86;
    let mut host = FakeHost::new();
    let stamp_pfn = 9u32;
    host.map_range(u64::from(stamp_pfn) << PAGE_SHIFT_X86, page as usize, 0);
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    state.gfx.fifo_base_page = stamp_pfn;

    state.arm_gva_deferred_window(0x1000, gva_entry(1, 4, 4, &[]));
    state.arm_gva_deferred_window(0x2000, gva_entry(1, 4, 4, &[]));
    assert_eq!(state.gva_deferred_flush.len(), 2, "two windows armed");

    crate::runtime::drain::write_stamp(&mut state, &mut host, 1, 0x55);

    assert!(
        state.gva_deferred_flush.is_empty(),
        "the guest may free every one of these targets the instant it reads \
         the stamp, so none of them may still be waiting to be written"
    );
    assert_eq!(
        state.completion_stamp_seq, 1,
        "the fence the windows are measured against must have moved"
    );
}

/// The guest's fence is the only thing that separates a deferred write from
/// a write into somebody else's allocation, and the page-set guard cannot
/// see it: free-then-reuse inside one process leaves the translation
/// identical, so `deferred_pages_still_ours` says yes to exactly the window
/// that corrupts the guest heap.
///
/// Both directions are asserted. A census that fires on every landing is as
/// useless as one that never fires — the whole point is that it separates
/// the windows landed inside their own fence from the ones that outlived it.
#[cfg(feature = "backend-vulkan")]
#[test]
fn a_window_landed_after_its_fence_is_counted_apart_from_one_landed_inside_it() {
    use crate::runtime::drain::store_route_count;
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);

    // Negative control: armed and landed under the same stamp. The guest
    // has not been told this render finished, so it cannot have freed the
    // target, and the write is the one the Store promised.
    let mut inside = gva_entry(1, 4, 4, &[]);
    inside.armed_stamp_seq = state.completion_stamp_seq;
    let same_before = store_route_count("gvaw_stamp_same");
    let outlived_before = store_route_count("gvaw_stamp_outlived");
    super::witness::note_window_outlived_its_stamp(&state, 0x1000, &inside, "rearm");
    assert_eq!(
        store_route_count("gvaw_stamp_same"),
        same_before + 1,
        "a window landed inside its own fence is the safe case and must be counted as one"
    );
    assert_eq!(
        store_route_count("gvaw_stamp_outlived"),
        outlived_before,
        "a guard that fires on every landing cannot price the repair"
    );

    // Positive control: the same window, landed after the guest was fenced.
    state.completion_stamp_seq = state.completion_stamp_seq.wrapping_add(3);
    let same_before = store_route_count("gvaw_stamp_same");
    super::witness::note_window_outlived_its_stamp(&state, 0x1000, &inside, "gva_alias");
    assert_eq!(
        store_route_count("gvaw_stamp_outlived"),
        outlived_before + 1,
        "a window whose stamp moved before it landed writes memory the guest was \
         told it could reclaim, and that is the class the page-set guard is blind to"
    );
    assert_eq!(
        store_route_count("gvaw_stamp_same"),
        same_before,
        "the two outcomes must not both be counted for one landing"
    );
}

#[cfg(feature = "backend-vulkan")]
#[test]
fn window_page_drift_refuses_the_guest_write_and_is_silent_without_it() {
    use crate::contract::endian::st32;
    use crate::contract::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
    use crate::runtime::host::{FakeHost, HostMemory};
    let page = 1u64 << PAGE_SHIFT_X86;
    let mut host = FakeHost::new();
    let (dir_gpa, root_gpa, data0) = (2 * page, 3 * page, 4 * page);
    for gpa in [dir_gpa, root_gpa, data0] {
        host.map_range(gpa, page as usize, 0);
    }
    let mut d = [0u8; 8];
    st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 3);
    st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
    host.write_gpa(dir_gpa, &d).unwrap();
    let mut pte = [0u8; 4];
    st32(&mut pte, 4);
    host.write_gpa(root_gpa, &pte).unwrap();
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    assert!(state.define_task(1, page, 2));

    crate::observe::redirect_logs_for_tests();
    let drift_lines = |from: usize| -> usize {
        std::fs::read_to_string(crate::observe::fail_log_path())
            .unwrap_or_default()
            .get(from..)
            .unwrap_or_default()
            .lines()
            .filter(|l| l.starts_with("deferred_window_page_drift "))
            .count()
    };
    let mark = || {
        std::fs::read_to_string(crate::observe::fail_log_path())
            .unwrap_or_default()
            .len()
    };

    // Negative control: armed on the page the GVA resolves to right now.
    let at = mark();
    assert!(
        super::guards::window_pages_still_ours(
            &state,
            &host,
            0,
            &gva_entry(1, 4, 4, &[data0]),
            "gva_alias",
            "guest=refused",
        ),
        "an unmoved window must stay writable — a guard that refuses every \
         flush means the guest never sees a Store"
    );
    assert_eq!(
        drift_lines(at),
        0,
        "a window that did not move must be quiet"
    );

    // Positive control: same window, armed on a page it no longer maps to.
    let at = mark();
    assert!(
        !super::guards::window_pages_still_ours(
            &state,
            &host,
            0,
            &gva_entry(1, 4, 4, &[9 * page]),
            "gva_alias",
            "guest=refused",
        ),
        "a window whose pages moved must be refused, not merely reported"
    );
    assert_eq!(drift_lines(at), 1, "a window whose pages moved must report");

    // A window armed on TWO pages whose range now resolves ONE page, and
    // that page is not one of the two. This is the arrangement a guest
    // produces by releasing a GPU allocation and letting part of the virtual
    // range be re-pointed: fewer pages come back, and what does come back
    // belongs to somebody else.
    //
    // A guard keyed on page COUNT reads this as "shorter walk, therefore
    // teardown, therefore nothing to protect" and permits it, and the writer
    // then lands rows in `data0` — which this window never owned. Keyed on
    // membership it is refused, which is what the guest's own crash reports
    // say has to happen.
    let at = mark();
    assert!(
        !super::guards::window_pages_still_ours(
            &state,
            &host,
            0,
            &gva_entry(1, 4, 4, &[7 * page, 8 * page]),
            "clear_store",
            "guest=refused",
        ),
        "a short walk that resolves a page the window was never armed on is \
         not a teardown — it is a write into another owner's pages"
    );
    assert_eq!(
        drift_lines(at),
        1,
        "the refusal must be visible; a silent one cannot be scored"
    );

    // The benign half of the same shape: fewer pages come back, and every
    // one of them is still ours. Refusing this would drop live Stores whose
    // destination never moved, so the guard must not simply require equal
    // sets.
    let at = mark();
    assert!(
        super::guards::window_pages_still_ours(
            &state,
            &host,
            0,
            &gva_entry(1, 4, 4, &[data0, 8 * page]),
            "clear_store",
            "guest=refused",
        ),
        "a walk that came back short but entirely inside the armed pages is \
         the teardown case, and its rows land in this window's own memory"
    );
    assert_eq!(drift_lines(at), 0, "a subset walk must stay quiet");
}

/// The same window, asked by the reader instead of the writer.
///
/// The cross-pass resident Load in `encode_draw_chain` trusts a GVA resident
/// as a draw's *prior content*, gated on a deferred window existing at the
/// address with matching geometry — conditions a different allocation
/// reusing the address satisfies exactly. The flush path had refused that
/// drift since it was written; the read path did not ask, which left the two
/// sides of one window disagreeing about whether it still belonged to its
/// name.
///
/// What this pins is that the reader gets the same verdict *and its own
/// outcome word*. A drift line is the only record either consumer leaves,
/// and `guest=refused` on a line emitted by a Load would say guest memory
/// was protected when what was actually refused was a stale picture.
#[cfg(feature = "backend-vulkan")]
#[test]
fn the_resident_load_reader_gets_the_same_drift_verdict_under_its_own_name() {
    use crate::contract::endian::st32;
    use crate::contract::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
    use crate::runtime::host::{FakeHost, HostMemory};
    let page = 1u64 << PAGE_SHIFT_X86;
    let mut host = FakeHost::new();
    let (dir_gpa, root_gpa, data0) = (2 * page, 3 * page, 4 * page);
    for gpa in [dir_gpa, root_gpa, data0] {
        host.map_range(gpa, page as usize, 0);
    }
    let mut d = [0u8; 8];
    st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 3);
    st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
    host.write_gpa(dir_gpa, &d).unwrap();
    let mut pte = [0u8; 4];
    st32(&mut pte, 4);
    host.write_gpa(root_gpa, &pte).unwrap();
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    assert!(state.define_task(1, page, 2));

    crate::observe::redirect_logs_for_tests();
    let tail = |from: usize| -> String {
        std::fs::read_to_string(crate::observe::fail_log_path())
            .unwrap_or_default()
            .get(from..)
            .unwrap_or_default()
            .lines()
            .filter(|l| l.starts_with("deferred_window_page_drift "))
            .collect::<Vec<_>>()
            .join("\n")
    };
    let mark = || {
        std::fs::read_to_string(crate::observe::fail_log_path())
            .unwrap_or_default()
            .len()
    };

    // The address still names the pages the window was armed on, so the
    // resident behind it is this allocation's own prior frame and the Load
    // may take it. Refusing here would cost a seed on every chained pass.
    let at = mark();
    assert!(
        super::guards::window_pages_still_ours(
            &state,
            &host,
            0,
            &gva_entry(1, 4, 4, &[data0]),
            "xpass_load",
            "resident=refused",
        ),
        "an unmoved window's resident is the draw's own prior content"
    );
    assert_eq!(
        tail(at),
        "",
        "an unmoved window must be quiet on both sides"
    );

    // The guest handed this address to a different allocation. The resident
    // still exists, still has the geometry, and still reports content_ready
    // — every gate the Load had before this check. It holds the previous
    // allocation's pixels.
    let at = mark();
    assert!(
        !super::guards::window_pages_still_ours(
            &state,
            &host,
            0,
            &gva_entry(1, 4, 4, &[9 * page]),
            "xpass_load",
            "resident=refused",
        ),
        "a reallocated address must not load the previous owner's pixels as \
         this draw's prior content"
    );
    let line = tail(at);
    assert!(
        line.contains("trigger=xpass_load"),
        "the line must name the reader that asked: {line}"
    );
    assert!(
        line.contains("resident=refused"),
        "the reader refuses a resident, not a guest write: {line}"
    );
    assert!(
        !line.contains("guest=refused"),
        "a Load must not claim it protected guest memory: {line}"
    );
}

/// The linear compute-storage rail gets the same drift decision as the GVA
/// rail, and takes its span the way the arm site does.
///
/// `flush_linear_one` needs a live Vulkan engine to reach its guest write, so
/// this exercises the decision itself with a linear key's geometry. The span
/// argument is the subtle part and the positive control is what pins it: a
/// linear key's `span_end` is a *length* (`row_stride * height`), not an end
/// address, and the arm site walks `(surface_offset, span_end)` with exactly
/// those two values. Reading `span_end` as an end address here would make the
/// span `page - page == 0`, the walk would come back empty, the short-walk arm
/// would permit, and the positive control below would fail — which is the
/// point of siting it at a nonzero offset rather than at GVA 0, where both
/// readings coincide.
#[cfg(feature = "backend-vulkan")]
#[test]
fn a_linear_window_whose_pages_moved_is_refused_and_reads_its_span_as_a_length() {
    use crate::contract::endian::st32;
    use crate::contract::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
    use crate::runtime::host::{FakeHost, HostMemory};
    let page = 1u64 << PAGE_SHIFT_X86;
    let mut host = FakeHost::new();
    let (dir_gpa, root_gpa, data0, data1) = (2 * page, 3 * page, 4 * page, 5 * page);
    for gpa in [dir_gpa, root_gpa, data0, data1] {
        host.map_range(gpa, page as usize, 0);
    }
    let mut d = [0u8; 8];
    st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 3);
    st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
    host.write_gpa(dir_gpa, &d).unwrap();
    // Two PTEs: GVA page 0 → data0, GVA page 1 → data1.
    let mut ptes = [0u8; 8];
    st32(&mut ptes[0..], 4);
    st32(&mut ptes[4..], 5);
    host.write_gpa(root_gpa, &ptes).unwrap();
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    assert!(state.define_task(1, 8 * page, 2));

    crate::observe::redirect_logs_for_tests();
    let drift_lines = |from: usize| -> usize {
        std::fs::read_to_string(crate::observe::fail_log_path())
            .unwrap_or_default()
            .get(from..)
            .unwrap_or_default()
            .lines()
            .filter(|l| l.starts_with("deferred_window_page_drift "))
            .count()
    };
    let mark = || {
        std::fs::read_to_string(crate::observe::fail_log_path())
            .unwrap_or_default()
            .len()
    };
    // One page long, sited at GVA `page` so a length/end confusion is visible.
    let (offset, span) = (page, page);

    // Negative control: armed on the page GVA `page` resolves to right now.
    let at = mark();
    assert!(
        super::guards::deferred_pages_still_ours(
            &state,
            &host,
            1,
            offset,
            span,
            &[data1].into_iter().collect(),
            "8x8 trigger=linear_flush ref=5",
            "guest=refused",
        ),
        "an unmoved linear window must stay writable — a guard that refuses \
         every flush means the guest never sees a compute Store"
    );
    assert_eq!(
        drift_lines(at),
        0,
        "a linear window that did not move must be quiet"
    );

    // Positive control: same window, armed on a page it no longer maps to.
    // This is also the assertion that the span is read as a length.
    let at = mark();
    assert!(
        !super::guards::deferred_pages_still_ours(
            &state,
            &host,
            1,
            offset,
            span,
            &[9 * page].into_iter().collect(),
            "8x8 trigger=linear_flush ref=5",
            "guest=refused",
        ),
        "a linear window whose pages moved must be refused — and a zero-length \
         walk from misreading span_end as an end address would permit it"
    );
    assert_eq!(
        drift_lines(at),
        1,
        "a linear window whose pages moved must report"
    );

    // A walk that resolves NOTHING also returns "still ours", because
    // `all` over an empty set is true — and that is not the guard agreeing,
    // it is the guard having nothing to compare. Counted apart, or this
    // rail's "no drift" reads as a verification it never made.
    use crate::runtime::drain::store_route_count;
    let verified_before = store_route_count("defw_pages_verified");
    let unwit_before = store_route_count("defw_unwit_no_live");
    let at = mark();
    assert!(
        super::guards::deferred_pages_still_ours(
            &state,
            &host,
            1,
            // Page index 3: inside the root page, but its PTE was never
            // written, so it is zero and translates to nothing. An index
            // past the root page's own extent would instead read whatever
            // GPA follows it, which is a different (and resolvable) thing.
            3 * page,
            span,
            &[data1].into_iter().collect(),
            "8x8 trigger=linear_flush ref=5",
            "guest=refused",
        ),
        "an unresolvable window is not drift — no row can land through it"
    );
    assert_eq!(drift_lines(at), 0, "and it is not reported as drift");
    assert_eq!(
        store_route_count("defw_unwit_no_live"),
        unwit_before + 1,
        "it must be counted as unwitnessed"
    );
    assert_eq!(
        store_route_count("defw_pages_verified"),
        verified_before,
        "and must NOT be counted as a verification"
    );

    // An empty armed set is the other unchecked exit, and it is its own slug
    // because it is a different gap.
    let no_armed_before = store_route_count("defw_unwit_no_armed");
    assert!(super::guards::deferred_pages_still_ours(
        &state,
        &host,
        1,
        offset,
        span,
        &std::collections::HashSet::new(),
        "8x8 trigger=linear_flush ref=5",
        "guest=refused",
    ));
    assert_eq!(
        store_route_count("defw_unwit_no_armed"),
        no_armed_before + 1
    );
}

/// Task teardown moves the task's GVA windows to the retired list (model)
/// and the runtime lands them cache-only — obligations never write guest
/// pages from teardown and never linger.
#[test]
fn task_delete_retires_gva_windows_cache_only() {
    use crate::runtime::host::FakeHost;
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
    assert!(state.define_task(6, 0x1000, 2));
    state.arm_gva_deferred_window(0x9000_0000, gva_entry(6, 4, 4, &[]));
    state.arm_gva_deferred_window(0x9100_0000, gva_entry(7, 4, 4, &[]));
    assert!(state.delete_task(6));
    assert!(
        !state.gva_deferred_flush.contains_key(&0x9000_0000),
        "dead task's window must leave the armed map"
    );
    assert!(
        state.gva_deferred_flush.contains_key(&0x9100_0000),
        "other task's window must stay armed"
    );
    assert_eq!(state.retired_gva_windows.len(), 1);
    super::retire_gva_windows(&mut state, &mut host);
    assert!(state.retired_gva_windows.is_empty());
}

/// The window cap lands the oldest-armed window first.
#[test]
fn oldest_gva_window_is_taken_first() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut newer = gva_entry(1, 4, 4, &[]);
    newer.armed_seq = 9;
    let mut older = gva_entry(1, 4, 4, &[]);
    older.armed_seq = 3;
    state.arm_gva_deferred_window(0x1000, newer);
    state.arm_gva_deferred_window(0x2000, older);
    let (gva, entry) = state.take_oldest_gva_deferred_window().unwrap();
    assert_eq!(gva, 0x2000);
    assert_eq!(entry.armed_seq, 3);
    assert_eq!(state.gva_deferred_flush.len(), 1);
}

#[test]
fn take_deferred_windows_is_exact_intersection() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    state.compute_deferred_flush.insert(
        key(7, 0, 256),
        crate::model::DeferredOwner::Storage {
            generation: 3,
            armed_stamp_seq: 0,
        },
    );
    state.compute_deferred_flush.insert(
        key(7, 256, 512),
        crate::model::DeferredOwner::Storage {
            generation: 4,
            armed_stamp_seq: 0,
        },
    );
    state.compute_deferred_flush.insert(
        key(8, 0, 256),
        crate::model::DeferredOwner::Storage {
            generation: 5,
            armed_stamp_seq: 0,
        },
    );

    // Disjoint range takes nothing.
    assert!(state.take_deferred_flush_windows(7, 512, 1024).is_empty());
    assert_eq!(state.compute_deferred_flush.len(), 3);

    // Intersecting range takes only the touching window on that mapping.
    let taken = state.take_deferred_flush_windows(7, 200, 257);
    assert_eq!(taken.len(), 2, "both mapping-7 windows intersect [200,257)");
    assert_eq!(state.compute_deferred_flush.len(), 1);
    assert!(state.compute_deferred_flush.contains_key(&key(8, 0, 256)));
}
