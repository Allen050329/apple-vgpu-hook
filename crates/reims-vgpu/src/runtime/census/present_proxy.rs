//! Always-on **flicker / dual-mid thrash proxies** for the present path.
//!
//! These do **not** change product behavior. They record compact, fail-visible
//! signals so a log census can tell "dual-mid incomplete thrash is happening"
//! without opening screenshots.
//!
//! ## Proxies (log: `/tmp/reims-vgpu-thrash.log`, also `observe::fail` on events)
//!
//! | Proxy | Meaning | Why it tracks flicker |
//! | --- | --- | --- |
//! | `mid_switch` | Consecutive CmdDisplaySwap captures name different mapping ids | Dual-buffer A/B present |
//! | `nz_swing` | On mid_switch, min(nz)/max(nz) < [`NZ_SWING_RATIO`] | One mid holds far less content (logo/partial vs full) |
//! | `sparse_present` | Captured frame nonzero fraction < [`SPARSE_NZ_FRAC`] at full desktop size | Logo-like / mostly-empty retain |
//! | `geom_mismatch` | Capture W×H differs from previous present geom | Letterbox / mode thrash |
//! | `capture_fail` | DisplaySwap capture returned false | Retain hole / keep_prior path |
//! | `selected_peer_divergence` | The protocol-selected retain is sparse while a same-geometry Store peer is dense | Hidden complete ping/pong sibling / incomplete retained Load base |
//!
//! Summary counters: `THRASH summary presents=… present_hz=… mid_sw=… nz_sw=… struct_sw=…
//! sparse=… geom=… fail=… peer_divergence=… converged=… post_converge_regress=…
//! stale_online=… t11_fb=…` (emitted every [`SUMMARY_EVERY`] present captures).

use std::collections::HashMap;
use std::sync::Mutex;

use crate::observe;

/// min(nz)/max(nz) below this on a mid switch ⇒ nz_swing thrash event.
/// Dual-mid logo residual: full desktop ~6.2e6 nz vs logo mid ~2e6 (ratio ~0.33).
/// Live incomplete dual-mid (black+layers ~3.4e6 vs full ~6.2e6) sits ~0.54 —
/// still incomplete visual thrash (structure_swing also fires). Measure-only.
const NZ_SWING_RATIO: f64 = 0.60;

/// Nonzero-byte fraction of a full frame below this ⇒ sparse_present.
/// Logo mid is sparse relative to wallpaper; empty dock glass is also low-variance
/// but still mid-nz — use a low bar so only near-empty/logo frames fire.
const SPARSE_NZ_FRAC: f64 = 0.08;

/// RGB-occupancy fraction at or above which a full-size present counts as a
/// **converged** desktop frame (wallpaper composited, not chrome-only-over-black).
/// The dual-mid boot strobe alternates a chrome-only sparse buffer (rgb_frac
/// ≈0.003) with the full desktop (rgb_frac ≈1.0); half-occupancy cleanly
/// separates the two and is reached only once the wallpaper underlay lands.
/// Measure-only — the `present_converge` timestamp it emits never gates present
/// selection (that would be the forbidden content-heuristic).
const CONVERGE_DENSE_FRAC: f64 = 0.50;

/// Consecutive non-dense (rgb_frac < [`CONVERGE_DENSE_FRAC`]) full-size presents
/// *after* the desktop first converged that count as a **post-converge regression**
/// — the desktop composited, then the guest reverted to the login/boot overlay
/// (rgb_frac ≈0.13, Apple-logo-over-black) or a sparse strobe. Because the retain
/// captured into +0x188 is the *full composited frame*, a normal desktop present
/// stays dense (~1.0) even for a tiny damage update; rgb_frac only falls below the
/// dense bar when the guest genuinely composites a mostly-empty/overlay frame. A
/// sustained run is therefore the intermittent "desktop rendered then broke" class.
/// Conservative so a brief
/// legitimately-dark fullscreen transition does not fire; measure-only.
const POST_CONVERGE_REGRESS_RUN: u64 = 24;

/// Minimum pixel count before sparse/swing proxies apply (ignore cursors / 2×2 tests).
const MIN_PROXY_PIXELS: u64 = 320 * 200;

const SUMMARY_EVERY: u64 = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PresentCaptureSample {
    pub mapping_id: u32,
    pub generation: u32,
    pub width: u32,
    pub height: u32,
    /// Byte nonzero (includes alpha-only black clears). Prefer [`Self::rgb_nz`]
    /// for content thrash.
    pub nz: usize,
    pub max_byte: u8,
    /// Pixels with any RGB channel nonzero — alpha-only clear is 0.
    pub rgb_nz: usize,
    pub max_rgb: u8,
    pub from_last_store: bool,
    /// Subsampled mean absolute horizontal edge (luma) — structure thrash proxy.
    pub edge_energy: u32,
}

struct ThrashState {
    last: Option<PresentCaptureSample>,
    presents: u64,
    mid_switches: u64,
    nz_swings: u64,
    structure_swings: u64,
    sparse: u64,
    geom_mismatch: u64,
    capture_fail: u64,
    selected_peer_divergence: u64,
    /// Dedup for `secondary_mrt_drop`: (reason_code, width, height) already
    /// reported this boot, so a per-draw MRT-secondary drop fires once per
    /// distinct combo, never per frame. Names which build path silently degraded
    /// a multi-RT draw to single-RT — the vibrancy coverage-mask drop that leaves
    /// a later material sample reading zero alpha (transparent tooltip / frosted
    /// pass-through class). Bounded by the small set of
    /// (reason, geometry) combinations a boot produces.
    secondary_mrt_drop_seen: std::collections::BTreeSet<(u8, u32, u32)>,
    secondary_mrt_blend_seen: std::collections::BTreeSet<(u32, u32, u32)>,
    /// Boot-convergence proxy: whether a full-size present has reached
    /// [`CONVERGE_DENSE_FRAC`] RGB occupancy yet (the wallpaper/desktop first
    /// fully composited). Latched once — the `present_converge` line fires
    /// exactly at that first dense present so every boot leaves one greppable
    /// convergence timestamp, and a boot that strobes forever leaves
    /// `converged=0` in the summary instead of silence.
    first_dense_seen: bool,
    /// Consecutive non-dense full-size presents since the last dense one, counted
    /// only after `first_dense_seen`. A dense present resets it to 0; crossing
    /// [`POST_CONVERGE_REGRESS_RUN`] fires `post_converge_regress`.
    post_converge_nondense_run: u64,
    /// Distinct post-converge regression episodes (desktop → overlay/strobe).
    post_converge_regress: u64,
    /// Post-**ack** display IRQs raised while the shared-page ONLINE bit (bit2)
    /// was still pending — the guest re-reads it and re-runs `process_online` →
    /// `connectionChange` → boot-progress overlay rebuild (x86 RE 2026-07-17).
    /// The host-driven strobe source the RE named ("re-signals bit2 every frame").
    stale_online_pending: u64,
    /// Latch so the (per-VBL, ~60 Hz) stale-online line fires once per boot.
    stale_online_logged: bool,
    /// Large-mapping type-11 zero-copy fallbacks (composite sampled guest
    /// pages because no current-generation resident was ready). Always-on
    /// black-band discriminator: `t11_fb=0` in the summary means the composite
    /// never fell back — any zeros came through the resident/cache path
    /// (guest-painted); a nonzero total means residents were missing at sample
    /// time for the mids named by the `t11_large_fallback` lines.
    t11_fb_total: u64,
    /// `(mid, map_generation)` combinations already named by a
    /// `t11_large_fallback` line, so a sustained episode logs once per surface
    /// instead of once per frame. Bounded by the live large-mapping set.
    t11_fb_seen: std::collections::BTreeSet<(u32, u32)>,
    /// Process-monotonic ms and present count at the previous summary, so each
    /// summary can report `present_hz` — the guest composite/render throughput.
    /// Under a real app this is the render-bound fps (the guest is render-, not
    /// VBL-bound: composites ~31–34/s at both 60 and 120 Hz). Making it a
    /// first-class always-on number turns a render-throughput regression (a
    /// pipeline stall dragging composites down) into a visible drop instead of a
    /// hand-diff of two `presents=` counters. `0` on the first summary (no prior
    /// window). Runs on the drain worker (off the QEMU main core).
    last_summary_ms: u64,
    last_summary_presents: u64,
}

/// Which multi-RT build check bailed, degrading the draw to single-RT.
///
/// The driving case is a vibrancy tile whose slot-1 RG16Float coverage mask is
/// dropped: a later material draw samples that mask GVA, finds no rendered
/// resident, and reads zero alpha — the see-through frosted-material class.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MrtDrop {
    /// The requested slots are not a contiguous run from 0.
    NonContiguousSlot,
    /// A secondary attachment's geometry differs from the primary's.
    GeometryMismatch,
    /// The secondary's pixel format has no known engine mapping.
    UnknownFormat,
    /// The secondary target has no resident identity to render into.
    NoIdentity,
    /// The secondary resolves to the primary's own resident.
    AliasesPrimary,
}

impl crate::observe::Decline for MrtDrop {
    fn slug(&self) -> &'static str {
        match self {
            Self::NonContiguousSlot => "mrt_drop_non_contiguous_slot",
            Self::GeometryMismatch => "mrt_drop_geometry_mismatch",
            Self::UnknownFormat => "mrt_drop_unknown_format",
            Self::NoIdentity => "mrt_drop_no_identity",
            Self::AliasesPrimary => "mrt_drop_aliases_primary",
        }
    }
}

impl MrtDrop {
    /// Compact stable code for the dedup key. Disjoint from [`MaskBindMiss`]'s
    /// because both share one dedup set.
    fn code(self) -> u8 {
        match self {
            Self::NonContiguousSlot => 1,
            Self::GeometryMismatch => 2,
            Self::UnknownFormat => 3,
            Self::NoIdentity => 4,
            Self::AliasesPrimary => 5,
        }
    }
}

/// Why a sample that matched a rendered MRT mask failed to bind it.
///
/// The sample side of [`MrtDrop`]. Both used to answer `geometry_mismatch` —
/// one string for two different checks in two different proxies, so a grep for
/// it could not tell a dropped render from a failed sample. They are now
/// distinct slugs; that collision is what registering the census vocabulary
/// found.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaskBindMiss {
    /// The mask resident's geometry differs from the sample's.
    GeometryMismatch,
    /// The mask resident is not content-ready.
    ResidentNotReady,
}

impl crate::observe::Decline for MaskBindMiss {
    fn slug(&self) -> &'static str {
        match self {
            Self::GeometryMismatch => "mask_bind_geometry_mismatch",
            Self::ResidentNotReady => "mask_bind_resident_not_ready",
        }
    }
}

impl MaskBindMiss {
    /// Dedup code, disjoint from [`MrtDrop::code`] — the two share one set.
    fn code(self) -> u8 {
        match self {
            Self::GeometryMismatch => 10,
            Self::ResidentNotReady => 11,
        }
    }
}

impl ThrashState {
    const fn new() -> Self {
        Self {
            last: None,
            presents: 0,
            mid_switches: 0,
            nz_swings: 0,
            structure_swings: 0,
            sparse: 0,
            geom_mismatch: 0,
            capture_fail: 0,
            selected_peer_divergence: 0,
            secondary_mrt_drop_seen: std::collections::BTreeSet::new(),
            secondary_mrt_blend_seen: std::collections::BTreeSet::new(),
            first_dense_seen: false,
            post_converge_nondense_run: 0,
            post_converge_regress: 0,
            stale_online_pending: 0,
            stale_online_logged: false,
            t11_fb_total: 0,
            t11_fb_seen: std::collections::BTreeSet::new(),
            last_summary_ms: 0,
            last_summary_presents: 0,
        }
    }
}

/// Record a large-mapping type-11 zero-copy fallback (always-on; called from
/// the sample rail when a ≥250k-px mapping lacks a current-generation ready
/// resident). Rare on a healthy boot — the black-band discriminator.
///
/// The running total feeds `t11_fb=` in the summary; the identity of the mids
/// that fell back is emitted once per `(mid, map_generation)` so a sustained
/// episode names its surfaces without a per-frame line. `probe` is the newest
/// any-generation registry entry for that surface at sample time
/// (`(generation, content_ready)`), which is what separates a genuinely absent
/// resident from one orphaned under another generation.
pub fn note_t11_large_fallback(mid: u32, map_gen: u32, probe: Option<(u64, bool)>) {
    let mut st = STATE.lock().unwrap_or_else(|e| e.into_inner());
    st.t11_fb_total = st.t11_fb_total.saturating_add(1);
    let first = st.t11_fb_seen.insert((mid, map_gen));
    let total = st.t11_fb_total;
    drop(st);
    if first {
        observe::off(format!(
            "t11_large_fallback mid={mid} map_gen={map_gen} probe={probe:?} total={total}"
        ));
    }
}

/// Dual-mid structure thrash: similar nz (wallpaper-in-holes still nonzero) but
/// edge energy diverges — live Safari shattered vs full-chrome alternating frames.
const STRUCTURE_SWING_RATIO: f64 = 0.75;

/// Single mutex so unit tests and concurrent presents cannot interleave counters.
static STATE: Mutex<ThrashState> = Mutex::new(ThrashState::new());

/// Per-mapping peak display Store nz (measure-only poison detector).
fn display_store_peak_map() -> &'static Mutex<HashMap<u32, usize>> {
    static MAP: std::sync::OnceLock<Mutex<HashMap<u32, usize>>> = std::sync::OnceLock::new();
    MAP.get_or_init(|| Mutex::new(HashMap::new()))
}

/// A conservative selected-vs-peer Store comparison. This is deliberately
/// separate from present selection: RGB occupancy can diagnose divergence but
/// must never choose the surface shown to the guest.
const PEER_DENSE_MIN_FRAC: f64 = 0.90;
const SELECTED_PEER_MAX_RATIO: f64 = 0.45;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DisplayStoreSample {
    width: u32,
    height: u32,
    rgb_nz: usize,
    seq: u64,
}

#[derive(Default)]
struct PeerDivergenceState {
    stores: HashMap<u32, DisplayStoreSample>,
    active: Option<(u32, u32, u32, u32)>,
    seq: u64,
}

fn peer_divergence_state() -> &'static Mutex<PeerDivergenceState> {
    static STATE: std::sync::OnceLock<Mutex<PeerDivergenceState>> = std::sync::OnceLock::new();
    STATE.get_or_init(|| Mutex::new(PeerDivergenceState::default()))
}

/// Record a display Store and detect a sparse protocol-selected retain with a
/// dense same-geometry sibling. Returns true only on entry into a new episode.
/// Measurement-only: the result never gates ownership, Store, or presentation.
pub fn note_selected_peer_divergence(
    mapping_id: u32,
    selected_mapping: u32,
    width: u32,
    height: u32,
    rgb_nz: usize,
) -> bool {
    let total = (width as u64).saturating_mul(height as u64);
    if mapping_id == 0 || selected_mapping == 0 || total < MIN_PROXY_PIXELS || rgb_nz == 0 {
        return false;
    }

    let event = {
        let mut peers = peer_divergence_state()
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        peers.seq = peers.seq.saturating_add(1);
        let seq = peers.seq;
        peers.stores.insert(
            mapping_id,
            DisplayStoreSample {
                width,
                height,
                rgb_nz,
                seq,
            },
        );
        let Some(selected) = peers.stores.get(&selected_mapping).copied() else {
            peers.active = None;
            return false;
        };
        let dense_min = (total as f64 * PEER_DENSE_MIN_FRAC).ceil() as usize;
        let dense = peers
            .stores
            .iter()
            .filter(|(mid, sample)| {
                **mid != selected_mapping
                    && sample.width == selected.width
                    && sample.height == selected.height
                    && sample.rgb_nz >= dense_min
            })
            .max_by_key(|(_, sample)| sample.rgb_nz)
            .map(|(mid, sample)| (*mid, *sample));
        let Some((dense_mid, dense)) = dense else {
            peers.active = None;
            return false;
        };
        let ratio = selected.rgb_nz as f64 / dense.rgb_nz.max(1) as f64;
        if ratio >= SELECTED_PEER_MAX_RATIO {
            peers.active = None;
            return false;
        }
        let key = (selected_mapping, dense_mid, selected.width, selected.height);
        if peers.active == Some(key) {
            return false;
        }
        peers.active = Some(key);
        Some((selected, dense_mid, dense, ratio))
    };

    let Some((selected, dense_mid, dense, ratio)) = event else {
        return false;
    };
    let mut st = STATE.lock().unwrap_or_else(|e| e.into_inner());
    st.selected_peer_divergence = st.selected_peer_divergence.saturating_add(1);
    drop(st);
    let selected_newer = selected.seq >= dense.seq;
    let seq_delta = if selected_newer {
        selected.seq - dense.seq
    } else {
        dense.seq - selected.seq
    };
    thrash_line(&format!(
        "selected_peer_divergence incoming_mid={mapping_id} selected_mid={selected_mapping} selected_rgb_nz={} selected_seq={} dense_mid={dense_mid} dense_rgb_nz={} dense_seq={} selected_newer={} seq_delta={} ratio={ratio:.4} {}x{}",
        selected.rgb_nz,
        selected.seq,
        dense.rgb_nz,
        dense.seq,
        u8::from(selected_newer),
        seq_delta,
        selected.width,
        selected.height
    ));
    true
}

/// Mapping-lifetime hook for selected-vs-peer diagnostics. A recycled mapping id
/// must not keep an old Store occupancy sample, or the measure-only
/// `selected_peer_divergence` proxy can compare the new surface against a dead
/// peer. This prunes only diagnostic state; present selection and residency are
/// unaffected.
pub fn forget_display_store_sample(mapping_id: u32) {
    if mapping_id == 0 {
        return;
    }
    let mut peers = peer_divergence_state()
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    peers.stores.remove(&mapping_id);
    if peers
        .active
        .is_some_and(|(selected, dense, _, _)| selected == mapping_id || dense == mapping_id)
    {
        peers.active = None;
    }
}

/// Always-on visibility for a **silently-degraded MRT draw**: a draw whose color
/// list has >1 attachment (the guest asked for multiple render targets) but whose
/// secondary attachments could not be built, so `build_secondary_targets`
/// returned empty and the draw fell back to the single-RT path. The driving case
/// is a vibrancy tile whose slot-1 RG16Float coverage mask is dropped: a later
/// material draw then samples that mask GVA, finds no rendered resident, and reads
/// zero alpha — the see-through frosted-material class.
/// This path used to be silent (every early-return was a bare `Vec::new()`), so a
/// dropped MRT looked identical to a legitimate single-RT draw.
///
/// `reason` is a stable slug for WHICH build check bailed; deduped on
/// `(reason, w, h)` so it fires once per distinct combination per boot (never per
/// frame). Runs on the render/drain worker (metal_draw), never the QEMU main loop.
/// Measure-only — it does NOT change the fallback behavior, only reports it.
pub fn note_secondary_mrt_drop(reason: MrtDrop, width: u32, height: u32) {
    let mut st = STATE.lock().unwrap_or_else(|e| e.into_inner());
    if !st
        .secondary_mrt_drop_seen
        .insert((reason.code(), width, height))
    {
        return;
    }
    drop(st);
    // The parenthetical prose this line used to carry ("multi-RT draw degraded
    // to single-RT…") moved to the doc comment above: `Emit` field values cannot
    // hold whitespace, because the log is parsed by splitting on spaces, and a
    // machine-readable line is worth more here than a sentence the reader can
    // get from the slug.
    observe::Emit::decline("secondary_mrt_drop", &reason)
        .field("geom", format!("{width}x{height}"))
        .fail();
}

/// A secondary MRT attachment carries its OWN blend state and the pipeline now
/// honours it.
///
/// Until 2026-07-25 every secondary attachment was forced unblended, on the
/// strength of a comment claiming the decode side did not carry per-attachment
/// blend — it did, and the Metal arm had been reading it per slot all along.
/// This is the proxy for the fixed class: it fires when a guest MRT pipeline
/// actually asks to blend a secondary slot, which is the only case whose
/// rendering changed. **Zero lines means the fix is inert on this workload**,
/// not that it is wrong; a nonzero count is the population that used to get a
/// raw store where Metal composites.
///
/// Deduped on `(slot, w, h)` and measure-only — nothing branches on it.
pub fn note_secondary_mrt_blend(slot: u32, width: u32, height: u32) {
    let mut st = STATE.lock().unwrap_or_else(|e| e.into_inner());
    if !st.secondary_mrt_blend_seen.insert((slot, width, height)) {
        return;
    }
    drop(st);
    observe::off(format!(
        "secondary_mrt_blend slot={slot} {width}x{height} \
         (per-slot blend honored; this slot used to write unblended)"
    ));
}

/// Sibling of [`note_secondary_mrt_drop`] for the SAMPLE side: a draw sampled a
/// texture whose GVA matches a mask this frame rendered as an MRT secondary (so it
/// IS the vibrancy coverage-mask sample), but the bind failed — geometry mismatch
/// or the secondary resident was not content-ready — so the material falls through
/// to the host-cache / guest-pages path and its alpha modulation may read a stale
/// or zero mask (the see-through frosted-material class).
/// Fires only when the sampled GVA is a KNOWN rendered mask (never on ordinary
/// texture samples), deduped on `(reason, w, h)`. Runs on the render/drain worker.
/// Measure-only. Shares the `secondary_mrt_drop_seen` dedup set (disjoint codes).
pub fn note_mrt_mask_bind_miss(reason: MaskBindMiss, width: u32, height: u32) {
    let mut st = STATE.lock().unwrap_or_else(|e| e.into_inner());
    if !st
        .secondary_mrt_drop_seen
        .insert((reason.code(), width, height))
    {
        return;
    }
    drop(st);
    observe::Emit::decline("mrt_mask_bind_miss", &reason)
        .field("geom", format!("{width}x{height}"))
        .fail();
}

/// Record a display-sized type-11 Store nz. Returns `(peak_before_or_after, poisoned)`.
///
/// `poisoned` when this Store falls below 70% of the mid's prior peak — the mid
/// lost content (Clear invent / incomplete Load base / dual-mid lag class).
/// Measure-only: never gates encode or present.
pub fn note_display_store_nz(mapping_id: u32, nz: usize) -> (usize, bool) {
    if mapping_id == 0 || nz == 0 {
        return (0, false);
    }
    let mut map = display_store_peak_map()
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let peak = map.get(&mapping_id).copied().unwrap_or(0);
    let poison = peak > 0 && (nz as u64).saturating_mul(10) < (peak as u64).saturating_mul(7);
    if nz > peak {
        map.insert(mapping_id, nz);
        (nz, false)
    } else {
        (peak, poison)
    }
}

/// Test-only isolation: proxy state is process-global, so parallel tests that
/// reset a device (`lib.rs device_reset` → [`reset_for_device`]) or drive
/// product note paths mutate counters and anchors out from under a multi-call
/// sequence assertion. Sequence tests hold the write side for their whole
/// body; product entry points take short scoped read guards in test builds.
/// Compiled out of product builds.
#[cfg(test)]
pub(crate) static TEST_STATE_ISOLATION: std::sync::RwLock<()> = std::sync::RwLock::new(());

/// Exclusive proxy-state guard for tests that assert multi-call sequences or
/// exact counter values (also used by other modules' proxy-asserting tests).
/// While holding it, call proxy `note_*`/`reset_for_test` directly only —
/// product paths take [`test_shared`] and would self-deadlock.
#[cfg(test)]
pub(crate) fn test_exclusive() -> std::sync::RwLockWriteGuard<'static, ()> {
    TEST_STATE_ISOLATION
        .write()
        .unwrap_or_else(|e| e.into_inner())
}

/// Shared-side guard for product paths that feed the proxy (capture, present,
/// draw notes). Scoped and never nested — recursive reads on `std` RwLock can
/// deadlock against a queued writer.
#[cfg(test)]
pub(crate) fn test_shared() -> std::sync::RwLockReadGuard<'static, ()> {
    TEST_STATE_ISOLATION
        .read()
        .unwrap_or_else(|e| e.into_inner())
}

/// Clear diagnostic state at a device lifetime boundary.
pub fn reset_for_device() {
    #[cfg(test)]
    let _shared = TEST_STATE_ISOLATION
        .read()
        .unwrap_or_else(|e| e.into_inner());
    reset_state_inner();
}

fn reset_state_inner() {
    let mut st = STATE.lock().unwrap_or_else(|e| e.into_inner());
    *st = ThrashState::new();
    drop(st);
    display_store_peak_map()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clear();
    *peer_divergence_state()
        .lock()
        .unwrap_or_else(|e| e.into_inner()) = PeerDivergenceState::default();
}

/// Test/helper: clear display Store peaks (unit tests isolation).
#[cfg(test)]
pub fn reset_display_store_peaks_for_test() {
    reset_state_inner();
}

fn append_thrash_file(msg: &str) {
    // Dedicated always-on file for `grep THRASH /tmp/reims-vgpu-thrash.log`.
    let _ = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("/tmp/reims-vgpu-thrash.log")
        .and_then(|mut f| {
            use std::io::Write;
            writeln!(f, "{msg}")
        });
}

fn thrash_line(msg: &str) {
    observe::fail(format!("THRASH {msg}"));
    append_thrash_file(msg);
}

/// Always-on **display-signal cadence** — the guest's real frame pacing.
///
/// WindowServer paces on the VBL / present-complete IRQs this device raises, and
/// VBL is wall-clock-limited across both the locked and contended device-poll
/// paths. This census reports source-poll / VBL / present-complete Hz over a
/// ~1 s window so the log alone reveals the pacing WindowServer actually sees
/// and whether the limiter remains stable under lock contention. Measure-only:
/// it raises no IRQ and changes no present decision. Emits on the poll path at
/// most once per window.
pub mod cadence {
    use crate::observe;
    use std::sync::atomic::{AtomicU64, Ordering};

    static POLLS: AtomicU64 = AtomicU64::new(0);
    static VBLS: AtomicU64 = AtomicU64::new(0);
    static PRESENT_COMPLETES: AtomicU64 = AtomicU64::new(0);
    /// 0 = not yet seeded. Holds the ms timestamp the current window opened.
    static WINDOW_START_MS: AtomicU64 = AtomicU64::new(0);

    /// Report cadence roughly once per second — dense enough to see a stall
    /// begin/end, sparse enough that the line is never a flood (1/s).
    const WINDOW_MS: u64 = 1000;

    /// A VBL IRQ was actually raised (post shared time limiter).
    pub fn note_vbl() {
        VBLS.fetch_add(1, Ordering::Relaxed);
        // Same signal feeds the per-frame hitch proxy: the averaged cadence line
        // above hides an isolated stalled frame, so measure the inter-VBL gap
        // tail here where every VBL raise passes through, both raise paths.
        super::hitch::record_vbl_live();
    }

    /// A present-complete IRQ was actually raised (guest present-event mask on).
    pub fn note_present_complete() {
        PRESENT_COMPLETES.fetch_add(1, Ordering::Relaxed);
    }

    /// Window-flip decision, split out for a deterministic unit test (the live
    /// `note_poll` feeds it the process-monotonic clock). Returns the census
    /// line to emit when the window closes, `None` while it is still open.
    fn maybe_emit(now: u64) -> Option<String> {
        let start = WINDOW_START_MS.load(Ordering::Relaxed);
        if start == 0 {
            // First ever poll: seed the window (CAS so a racing poll seeds once).
            let _ = WINDOW_START_MS.compare_exchange(0, now, Ordering::Relaxed, Ordering::Relaxed);
            return None;
        }
        let dt = now.saturating_sub(start);
        if dt < WINDOW_MS {
            return None;
        }
        // Exactly one poll wins the flip; the losers keep counting into the
        // next window rather than double-emitting.
        if WINDOW_START_MS
            .compare_exchange(start, now, Ordering::Relaxed, Ordering::Relaxed)
            .is_err()
        {
            return None;
        }
        let polls = POLLS.swap(0, Ordering::Relaxed);
        let vbls = VBLS.swap(0, Ordering::Relaxed);
        let pcs = PRESENT_COMPLETES.swap(0, Ordering::Relaxed);
        let hz = |n: u64| n.saturating_mul(1000) as f64 / dt.max(1) as f64;
        Some(format!(
            "display_cadence poll_hz={:.1} vbl_hz={:.1} present_complete_hz={:.1} \
             polls={polls} vbls={vbls} present_completes={pcs} window_ms={dt}",
            hz(polls),
            hz(vbls),
            hz(pcs),
        ))
    }

    /// Called once per `gfx_update` poll; owns the periodic census emit.
    pub fn note_poll_at(now: u64) -> Option<String> {
        POLLS.fetch_add(1, Ordering::Relaxed);
        let line = maybe_emit(now);
        if let Some(l) = &line {
            observe::off(l);
        }
        line
    }

    /// Live entry: stamp the poll with the same clock that trails every log line.
    pub fn note_poll() {
        note_poll_at(observe::elapsed_ms() as u64);
    }

    #[cfg(test)]
    pub(crate) fn reset() {
        POLLS.store(0, Ordering::Relaxed);
        VBLS.store(0, Ordering::Relaxed);
        PRESENT_COMPLETES.store(0, Ordering::Relaxed);
        WINDOW_START_MS.store(0, Ordering::Relaxed);
    }

    #[cfg(test)]
    pub(crate) fn window_ms() -> u64 {
        WINDOW_MS
    }
}

/// Always-on **per-frame present hitch** proxy — the tail the averaged
/// [`cadence`] line cannot show.
///
/// P3 wants *consistent* pacing: no stutter every N seconds. A 1 s-averaged
/// `vbl_hz` stays ~120 through an isolated 60 ms stalled frame (one bad gap in
/// 120 good ones barely moves the mean), so the averaged census is blind to the
/// exact bug class P3 targets. This proxy measures the **inter-VBL gap** — the
/// interval between consecutive VBL raises, which is what WindowServer paces on
/// and what the user perceives as smoothness — and reports the per-window *worst
/// gap* plus a count of gaps that exceeded the local cadence.
///
/// Three properties make it honest rather than a false-alarm generator:
/// - **Cadence-relative, not absolute.** A 33 ms gap is a normal frame at 30 fps
///   but a 4-frame stall at 120 fps. The hitch threshold tracks an EWMA of the
///   recent gap (the live nominal interval) and flags a gap only when it exceeds
///   `max(ABS_FLOOR_MS, K × ewma)`. So steady 30 fps video never trips it, while
///   a single dropped frame at 120 fps does.
/// - **Idle resume is not a hitch.** When the guest stops presenting (static
///   page), VBLs stop; the next VBL after the page wakes shows a multi-second
///   gap that is *resumption*, not stutter. Gaps above `RESUME_CEIL_MS` are
///   counted as `resumes`, excluded from the EWMA, the max, and the hitch count.
/// - **Silent when smooth.** A healthy boot holds a steady cadence, so
///   `hitches=0` and no per-event line ever fires — the fail-visible contract
///   (zero on a healthy boot, loud on the bug). A genuine periodic stutter shows
///   as a periodic spike in `max_gap_ms` with `hitches>0`.
///
/// Measure-only: raises no IRQ, changes no present/decode/execute decision. Fed
/// from [`cadence::note_vbl`] so both VBL raise paths are covered by one hook.
pub mod hitch {
    use crate::observe;
    use std::sync::Mutex;

    /// EWMA smoothing: `ewma += (gap - ewma) / 8` — alpha 1/8, ~8-gap memory.
    /// Fast enough to track a 120→30 fps content change within a few frames,
    /// slow enough that one outlier gap does not pull the baseline up to hide
    /// the next hitch.
    const EWMA_SHIFT: u64 = 3;
    /// A gap counts as a hitch when it exceeds `K × ewma`. K = 5/2 = 2.5: two
    /// and a half nominal frame intervals — comfortably past scheduler jitter,
    /// short of only catching catastrophic stalls.
    const HITCH_K_NUM: u64 = 5;
    const HITCH_K_DEN: u64 = 2;
    /// Absolute floor so the relative threshold never drops below a perceptible
    /// stutter at high refresh: at 120 fps (ewma≈8 ms) `K×ewma≈20 ms`, but a
    /// 20 ms gap is barely two frames — hold the bar at 24 ms so the census only
    /// names gaps a user could actually see.
    const ABS_FLOOR_MS: u64 = 24;
    /// A gap larger than this is the guest resuming after an idle/static stretch
    /// (VBLs had stopped), not a mid-stream stutter. Excluded from every hitch
    /// statistic. 400 ms ≫ any real frame interval (2.5 fps) yet well below a
    /// human-noticeable "the page was paused" stretch.
    const RESUME_CEIL_MS: u64 = 400;
    /// Emit an immediate, greppable per-event line when a single gap reaches this
    /// — the exact wall-clock moment of a clear stutter. Kept high enough that a
    /// healthy boot never fires it (the windowed summary carries the smaller
    /// hitches). 50 ms = a dropped frame even at 30 fps.
    const EVENT_MS: u64 = 50;
    /// Summary window. Matches the cadence window so the two lines interleave at
    /// the same rate and a reader can line them up.
    const WINDOW_MS: u64 = 1000;

    struct HitchState {
        /// Timestamp of the previous VBL raise; 0 = unseeded.
        last_vbl_ms: u64,
        /// EWMA of the recent inter-VBL gap, ms; 0 = unseeded.
        ewma_gap_ms: u64,
        /// Open timestamp of the current summary window; 0 = unseeded.
        window_start_ms: u64,
        vbls: u64,
        hitches: u64,
        max_gap_ms: u64,
        resumes: u64,
    }

    impl HitchState {
        const fn new() -> Self {
            Self {
                last_vbl_ms: 0,
                ewma_gap_ms: 0,
                window_start_ms: 0,
                vbls: 0,
                hitches: 0,
                max_gap_ms: 0,
                resumes: 0,
            }
        }
    }

    static STATE: Mutex<HitchState> = Mutex::new(HitchState::new());

    /// Adaptive hitch threshold in ms for the current EWMA. Split out so the
    /// decision is unit-testable without the clock or the mutex.
    fn threshold_ms(ewma_gap_ms: u64) -> u64 {
        (ewma_gap_ms.saturating_mul(HITCH_K_NUM) / HITCH_K_DEN).max(ABS_FLOOR_MS)
    }

    /// Record one VBL raise at `now` (ms). Returns the summary line when this
    /// raise closes a window, plus an optional immediate per-event line for a
    /// clear stutter — both `None` on a smooth, mid-window raise. Pure w.r.t. the
    /// clock (the live entry passes `observe::elapsed_ms`) so the windowing and
    /// the adaptive threshold are deterministically testable.
    fn record_at(now: u64) -> (Option<String>, Option<String>) {
        let mut s = STATE.lock().unwrap_or_else(|e| e.into_inner());
        s.vbls = s.vbls.saturating_add(1);
        let mut event = None;
        if s.last_vbl_ms == 0 {
            // First ever VBL: seed the gap baseline + window, no gap to measure.
            s.last_vbl_ms = now;
            s.window_start_ms = now;
            return (None, None);
        }
        let gap = now.saturating_sub(s.last_vbl_ms);
        s.last_vbl_ms = now;
        if gap > RESUME_CEIL_MS {
            // Stream resumed after idle — not a stutter. Do not poison the EWMA
            // or the max; a fresh baseline re-seeds on the next steady gaps.
            s.resumes = s.resumes.saturating_add(1);
        } else {
            // The very first real gap only *seeds* the baseline — there is no
            // prior cadence to judge it against, so a stream that opens at a low
            // frame rate (100 ms gaps) must not read its first gap as a stutter.
            let was_seeded = s.ewma_gap_ms != 0;
            let thresh = threshold_ms(s.ewma_gap_ms);
            s.ewma_gap_ms = if !was_seeded {
                gap
            } else {
                s.ewma_gap_ms - (s.ewma_gap_ms >> EWMA_SHIFT) + (gap >> EWMA_SHIFT)
            };
            if was_seeded && gap > thresh && gap >= ABS_FLOOR_MS {
                s.hitches = s.hitches.saturating_add(1);
                if gap >= EVENT_MS {
                    event = Some(format!(
                        "present_hitch_event gap_ms={gap} thresh_ms={thresh} ewma_ms={}",
                        s.ewma_gap_ms
                    ));
                }
            }
            if gap > s.max_gap_ms {
                s.max_gap_ms = gap;
            }
        }
        let dt = now.saturating_sub(s.window_start_ms);
        if dt < WINDOW_MS {
            return (None, event);
        }
        let summary = format!(
            "present_hitch vbls={} max_gap_ms={} hitches={} resumes={} ewma_ms={} window_ms={dt}",
            s.vbls, s.max_gap_ms, s.hitches, s.resumes, s.ewma_gap_ms
        );
        // Reset window counters; keep last_vbl_ms + ewma so cadence carries over.
        s.window_start_ms = now;
        s.vbls = 0;
        s.hitches = 0;
        s.max_gap_ms = 0;
        s.resumes = 0;
        (Some(summary), event)
    }

    /// Live entry from [`super::cadence::note_vbl`]: stamp with the same
    /// process-monotonic clock that trails every log line, and sink any lines.
    pub fn record_vbl_live() {
        let (summary, event) = record_at(observe::elapsed_ms() as u64);
        if let Some(e) = event {
            observe::off(&e);
        }
        if let Some(l) = summary {
            observe::off(&l);
        }
    }

    #[cfg(test)]
    pub(crate) fn reset() {
        let mut s = STATE.lock().unwrap_or_else(|e| e.into_inner());
        *s = HitchState::new();
    }

    #[cfg(test)]
    pub(crate) fn record_for_test(now: u64) -> (Option<String>, Option<String>) {
        record_at(now)
    }

    #[cfg(test)]
    pub(crate) fn threshold_for_test(ewma_gap_ms: u64) -> u64 {
        threshold_ms(ewma_gap_ms)
    }

    #[cfg(test)]
    pub(crate) fn window_ms() -> u64 {
        WINDOW_MS
    }
}

/// Always-on **resource-teardown churn** rate — the guest IOSurface/paging
/// lifecycle pump, folded into a ~1 s summary instead of a per-event flood.
///
/// Under sustained compositor activity (a continuously-animating app targeting
/// 60+ fps) the guest recycles layer-backing IOSurfaces every frame: each
/// recycled surface emits a `DeleteIOSurfaceBacking2` (0x36) and, on unwire, a
/// `SynchronizeResources` (0x35). One always-on census line per event floods
/// the sink to ~10^5 lines/session (measured 48k delete + 49k sync under a
/// bouncing-ball rAF page — a ~1M-line boot). This proxy folds those into one
/// window summary carrying the rate + load-bearing aggregates; the per-event
/// lines move behind `REIMS_VGPU_DRAW_LOG`.
///
/// It is also the **host-visible forensic proxy for the guest-kext
/// orphaned-memory-pool UAF panic** (kb reims-vgpu-resource-paging): the crash fires in
/// the guest's `IOAccelOrphanedMemoryPool` collector walking a freed
/// `AppleParavirtResource*`, and this teardown rate is the churn that feeds that
/// pool. A crash post-mortem reads the last `teardown_churn` line before the
/// guest died.
///
/// Self-clocked on the drain worker (off the QEMU main loop / vCPU): every
/// counted event checks the window and the event that wins the CAS emits. Emits
/// nothing while all counters are zero, so a truly idle desktop never adds a
/// line. Measure-only — raises no IRQ, changes no decode/present decision.
pub mod lifecycle_churn {
    use crate::observe;
    use std::sync::atomic::{AtomicU64, Ordering};

    static DELETE_DEAD: AtomicU64 = AtomicU64::new(0);
    static DELETE_CONDEMN: AtomicU64 = AtomicU64::new(0);
    static DELETE_UNMAPPED: AtomicU64 = AtomicU64::new(0);
    static SYNCHRONIZE: AtomicU64 = AtomicU64::new(0);
    static SYNC_FLUSHED: AtomicU64 = AtomicU64::new(0);
    static REPLACE: AtomicU64 = AtomicU64::new(0);
    static INVALIDATE: AtomicU64 = AtomicU64::new(0);
    /// 0 = not yet seeded. Holds the ms timestamp the current window opened.
    static WINDOW_START_MS: AtomicU64 = AtomicU64::new(0);

    const WINDOW_MS: u64 = 1000;

    /// Outcome of a `DeleteIOSurfaceBacking2` teardown (mirrors the drain.rs
    /// `mode` string so the summary keeps the dead/condemn/unmapped split).
    #[derive(Clone, Copy)]
    pub enum DeleteMode {
        Dead,
        Condemn,
        Unmapped,
    }

    pub fn note_delete(mode: DeleteMode) {
        match mode {
            DeleteMode::Dead => &DELETE_DEAD,
            DeleteMode::Condemn => &DELETE_CONDEMN,
            DeleteMode::Unmapped => &DELETE_UNMAPPED,
        }
        .fetch_add(1, Ordering::Relaxed);
        emit_if_window_closed();
    }

    pub fn note_synchronize(deferred_flushed: u32) {
        SYNCHRONIZE.fetch_add(1, Ordering::Relaxed);
        SYNC_FLUSHED.fetch_add(deferred_flushed as u64, Ordering::Relaxed);
        emit_if_window_closed();
    }

    pub fn note_replace() {
        REPLACE.fetch_add(1, Ordering::Relaxed);
        emit_if_window_closed();
    }

    pub fn note_invalidate() {
        INVALIDATE.fetch_add(1, Ordering::Relaxed);
        emit_if_window_closed();
    }

    fn emit_if_window_closed() {
        if let Some(line) = maybe_line_at(observe::elapsed_ms() as u64) {
            observe::off(line);
        }
    }

    /// Window-flip + summary format, split out for a deterministic unit test.
    /// Returns the summary line when this call wins the ~1 s window flip **and**
    /// at least one event landed in the window; `None` while the window is still
    /// open, on the seeding poll, when another thread wins the CAS, or when the
    /// closed window counted nothing (idle → no flood).
    fn maybe_line_at(now: u64) -> Option<String> {
        let start = WINDOW_START_MS.load(Ordering::Relaxed);
        if start == 0 {
            let _ = WINDOW_START_MS.compare_exchange(0, now, Ordering::Relaxed, Ordering::Relaxed);
            return None;
        }
        let dt = now.saturating_sub(start);
        if dt < WINDOW_MS {
            return None;
        }
        if WINDOW_START_MS
            .compare_exchange(start, now, Ordering::Relaxed, Ordering::Relaxed)
            .is_err()
        {
            return None;
        }
        let dead = DELETE_DEAD.swap(0, Ordering::Relaxed);
        let condemn = DELETE_CONDEMN.swap(0, Ordering::Relaxed);
        let unmapped = DELETE_UNMAPPED.swap(0, Ordering::Relaxed);
        let sync = SYNCHRONIZE.swap(0, Ordering::Relaxed);
        let flushed = SYNC_FLUSHED.swap(0, Ordering::Relaxed);
        let replace = REPLACE.swap(0, Ordering::Relaxed);
        let invalidate = INVALIDATE.swap(0, Ordering::Relaxed);
        let deletes = dead.saturating_add(condemn).saturating_add(unmapped);
        if deletes == 0 && sync == 0 && replace == 0 && invalidate == 0 {
            return None;
        }
        let rate = |n: u64| n.saturating_mul(1000) as f64 / dt.max(1) as f64;
        Some(format!(
            "teardown_churn delete_hz={:.1} sync_hz={:.1} deletes={deletes} dead={dead} \
             condemn={condemn} unmapped={unmapped} sync={sync} deferred_flushed={flushed} \
             replace={replace} invalidate={invalidate} window_ms={dt}",
            rate(deletes),
            rate(sync),
        ))
    }

    #[cfg(test)]
    pub(crate) fn reset() {
        for a in [
            &DELETE_DEAD,
            &DELETE_CONDEMN,
            &DELETE_UNMAPPED,
            &SYNCHRONIZE,
            &SYNC_FLUSHED,
            &REPLACE,
            &INVALIDATE,
            &WINDOW_START_MS,
        ] {
            a.store(0, Ordering::Relaxed);
        }
    }

    /// Bump a delete counter without the live clock/emit — for the unit test to
    /// stage counts, then drive the flip via `test_maybe_line`.
    #[cfg(test)]
    pub(crate) fn test_note_delete(mode: DeleteMode) {
        match mode {
            DeleteMode::Dead => &DELETE_DEAD,
            DeleteMode::Condemn => &DELETE_CONDEMN,
            DeleteMode::Unmapped => &DELETE_UNMAPPED,
        }
        .fetch_add(1, Ordering::Relaxed);
    }

    #[cfg(test)]
    pub(crate) fn test_note_synchronize(deferred_flushed: u32) {
        SYNCHRONIZE.fetch_add(1, Ordering::Relaxed);
        SYNC_FLUSHED.fetch_add(deferred_flushed as u64, Ordering::Relaxed);
    }

    #[cfg(test)]
    pub(crate) fn test_maybe_line(now: u64) -> Option<String> {
        maybe_line_at(now)
    }
}

/// Always-on **type-11 import-present outcome** rate, folded into a ~1 s summary.
///
/// `log_result` emits one `import_present used=1` line per successful Store import
/// — under a continuously-animating app that is ~1/present (~77k/session), a raw
/// flood on BOTH schedulers (unlike the wall-clock-gated `import_store_timing`,
/// which only floods under SCHED_IDLE preemption). The success census is redundant
/// with this windowed count, so `used=1` moves behind `REIMS_VGPU_DRAW_LOG`; the aggregate
/// used/skipped/failed rate lives here. Skip/Fail (`used=0`) keep their fail-visible
/// per-event line with the reason. Self-clocked on the drain worker (off-main-core);
/// silent when a window counted nothing.
pub mod present_import {
    use crate::observe;
    use std::sync::atomic::{AtomicU64, Ordering};

    static USED: AtomicU64 = AtomicU64::new(0);
    static SKIPPED: AtomicU64 = AtomicU64::new(0);
    static FAILED: AtomicU64 = AtomicU64::new(0);
    static WINDOW_START_MS: AtomicU64 = AtomicU64::new(0);

    const WINDOW_MS: u64 = 1000;

    /// Count one import outcome (`used`=Ok, `is_fail`=Fail vs Skip) and emit the
    /// window summary if this call closes a ~1 s window.
    pub fn note(used: bool, is_fail: bool) {
        if used {
            USED.fetch_add(1, Ordering::Relaxed);
        } else if is_fail {
            FAILED.fetch_add(1, Ordering::Relaxed);
        } else {
            SKIPPED.fetch_add(1, Ordering::Relaxed);
        }
        if let Some(line) = maybe_line_at(observe::elapsed_ms() as u64) {
            observe::off(line);
        }
    }

    fn maybe_line_at(now: u64) -> Option<String> {
        let start = WINDOW_START_MS.load(Ordering::Relaxed);
        if start == 0 {
            let _ = WINDOW_START_MS.compare_exchange(0, now, Ordering::Relaxed, Ordering::Relaxed);
            return None;
        }
        let dt = now.saturating_sub(start);
        if dt < WINDOW_MS {
            return None;
        }
        if WINDOW_START_MS
            .compare_exchange(start, now, Ordering::Relaxed, Ordering::Relaxed)
            .is_err()
        {
            return None;
        }
        let used = USED.swap(0, Ordering::Relaxed);
        let skipped = SKIPPED.swap(0, Ordering::Relaxed);
        let failed = FAILED.swap(0, Ordering::Relaxed);
        if used == 0 && skipped == 0 && failed == 0 {
            return None;
        }
        let rate = used.saturating_mul(1000) as f64 / dt.max(1) as f64;
        Some(format!(
            "present_import used_hz={rate:.1} used={used} skipped={skipped} failed={failed} window_ms={dt}"
        ))
    }

    #[cfg(test)]
    pub(crate) fn reset() {
        for a in [&USED, &SKIPPED, &FAILED, &WINDOW_START_MS] {
            a.store(0, Ordering::Relaxed);
        }
    }

    #[cfg(test)]
    pub(crate) fn test_note(used: bool, is_fail: bool) {
        if used {
            USED.fetch_add(1, Ordering::Relaxed);
        } else if is_fail {
            FAILED.fetch_add(1, Ordering::Relaxed);
        } else {
            SKIPPED.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[cfg(test)]
    pub(crate) fn test_maybe_line(now: u64) -> Option<String> {
        maybe_line_at(now)
    }
}

/// Always-on windowed census of the GPU stats oracle (the zero-copy proxy rail).
///
/// The proxies are fed by a compute reduction over the resident, not by a CPU
/// frame copy. This is the health signal for that rail: `armed` = dispatches
/// submitted, `arm_fail` = the resident was not reducible (not content-ready,
/// not BGRA, or the pool saturated), `superseded` = a pending reduction was
/// cancelled because the next present arrived before the GPU finished it.
///
/// Read it as: a healthy desktop shows `armed` tracking the present rate with
/// `arm_fail=0`. A climbing `arm_fail` means the proxies are going dark — the
/// resident registry is missing the presented mid — and a climbing `superseded`
/// means the reduction is not keeping up with the present rate, so proxy
/// sampling is thinning out. Neither ever costs correctness; both cost coverage,
/// which is exactly what a measurement rail must report about itself.
pub mod stats_oracle {
    use crate::observe;
    use std::sync::atomic::{AtomicU64, Ordering};

    static ARMED: AtomicU64 = AtomicU64::new(0);
    static ARM_FAIL: AtomicU64 = AtomicU64::new(0);
    static SUPERSEDED: AtomicU64 = AtomicU64::new(0);
    static WINDOW_START_MS: AtomicU64 = AtomicU64::new(0);

    const WINDOW_MS: u64 = 1000;

    pub fn note_armed(ok: bool) {
        if ok {
            ARMED.fetch_add(1, Ordering::Relaxed);
        } else {
            ARM_FAIL.fetch_add(1, Ordering::Relaxed);
        }
        if let Some(line) = maybe_line_at(observe::elapsed_ms() as u64) {
            observe::off(line);
        }
    }

    pub fn note_superseded() {
        SUPERSEDED.fetch_add(1, Ordering::Relaxed);
    }

    fn maybe_line_at(now: u64) -> Option<String> {
        let start = WINDOW_START_MS.load(Ordering::Relaxed);
        if start == 0 {
            let _ = WINDOW_START_MS.compare_exchange(0, now, Ordering::Relaxed, Ordering::Relaxed);
            return None;
        }
        let dt = now.saturating_sub(start);
        if dt < WINDOW_MS {
            return None;
        }
        if WINDOW_START_MS
            .compare_exchange(start, now, Ordering::Relaxed, Ordering::Relaxed)
            .is_err()
        {
            return None;
        }
        let armed = ARMED.swap(0, Ordering::Relaxed);
        let arm_fail = ARM_FAIL.swap(0, Ordering::Relaxed);
        let superseded = SUPERSEDED.swap(0, Ordering::Relaxed);
        if armed.saturating_add(arm_fail) == 0 {
            return None;
        }
        Some(format_line(dt, armed, arm_fail, superseded))
    }

    fn format_line(dt: u64, armed: u64, arm_fail: u64, superseded: u64) -> String {
        let hz = armed.saturating_mul(1000) as f64 / dt.max(1) as f64;
        format!(
            "stats_oracle window_ms={dt} armed={armed} arm_fail={arm_fail} \
             superseded={superseded} armed_hz={hz:.1}"
        )
    }

    #[cfg(test)]
    pub(crate) fn reset() {
        for a in [&ARMED, &ARM_FAIL, &SUPERSEDED, &WINDOW_START_MS] {
            a.store(0, Ordering::Relaxed);
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn window_reports_arm_health() {
            reset();
            assert_eq!(maybe_line_at(1), None, "first call opens the window");
            ARMED.fetch_add(30, Ordering::Relaxed);
            ARM_FAIL.fetch_add(2, Ordering::Relaxed);
            SUPERSEDED.fetch_add(5, Ordering::Relaxed);
            let line = maybe_line_at(1 + WINDOW_MS).expect("line");
            assert!(line.contains("armed=30"), "{line}");
            assert!(line.contains("arm_fail=2"), "{line}");
            assert!(line.contains("superseded=5"), "{line}");
            assert!(
                maybe_line_at(1 + 2 * WINDOW_MS).is_none(),
                "idle window emits nothing"
            );
        }
    }
}

/// Always-on windowed census of where a FULL present capture sourced its frame.
///
/// `capture_present_frame` reads the GPU resident (`read_resident_bgra`: the
/// readback alone, no guest-page scatter). `resident` counts successful reads;
/// `guest` is retained in the log schema as a legacy counter and remains zero
/// because there is deliberately no guest-page fallback. Both are COUNT-based,
/// so they are trustworthy under the SCHED_IDLE agent boot where the `*_us`
/// buckets are not.
///
/// Read it as the CPU-fallback capture source: any full capture that succeeds
/// should report `resident_frac=1.0`. A missing resident fails visibly instead
/// of silently switching to guest-page scatter. Measure-only; never gates.
/// Window-publish outcome: did the captured guest frame actually reach the host
/// window, or was it dropped before the window ever saw it?
///
/// The macOS/MoltenVK publish path drops a captured frame outright when no
/// candidate resident has landed content, so the window keeps showing its
/// previous (or slate) contents. That drop used to be completely silent — the
/// only trace was `dmabuf_active` flipping false. A sustained drop run is the
/// "desktop frozen but the device is alive" class, so it needs a name and a
/// count.
pub mod window_publish {
    use crate::observe;
    use std::sync::atomic::{AtomicU64, Ordering};

    static PUBLISHED: AtomicU64 = AtomicU64::new(0);
    static DROPPED: AtomicU64 = AtomicU64::new(0);
    static WINDOW_START_MS: AtomicU64 = AtomicU64::new(0);

    const WINDOW_MS: u64 = 1000;

    /// Count one publish decision: `published`=the frame was handed to the
    /// window, else it was dropped because no resident carried its content.
    pub fn note(published: bool) {
        if published {
            PUBLISHED.fetch_add(1, Ordering::Relaxed);
        } else {
            DROPPED.fetch_add(1, Ordering::Relaxed);
        }
        if let Some(line) = maybe_line_at(observe::elapsed_ms() as u64) {
            observe::off(line);
        }
    }

    fn maybe_line_at(now: u64) -> Option<String> {
        let start = WINDOW_START_MS.load(Ordering::Relaxed);
        if start == 0 {
            let _ = WINDOW_START_MS.compare_exchange(0, now, Ordering::Relaxed, Ordering::Relaxed);
            return None;
        }
        let dt = now.saturating_sub(start);
        if dt < WINDOW_MS {
            return None;
        }
        if WINDOW_START_MS
            .compare_exchange(start, now, Ordering::Relaxed, Ordering::Relaxed)
            .is_err()
        {
            return None;
        }
        let published = PUBLISHED.swap(0, Ordering::Relaxed);
        let dropped = DROPPED.swap(0, Ordering::Relaxed);
        if published.saturating_add(dropped) == 0 {
            return None;
        }
        Some(format_line(dt, published, dropped))
    }

    /// Why the host window published nothing for a frame it was asked to show.
    ///
    /// One variant today, and a type rather than a literal because bare
    /// `resident_not_ready` is one grep away from the MRT mask rail's
    /// `mask_bind_resident_not_ready` — a different subsystem answering a
    /// same-sounding question. A second drop cause gets its own variant here.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum WindowPublishDrop {
        /// The engine had no content-ready resident to hand the window.
        ResidentNotReady,
    }

    impl crate::observe::Decline for WindowPublishDrop {
        fn slug(&self) -> &'static str {
            match self {
                Self::ResidentNotReady => "window_publish_resident_not_ready",
            }
        }
    }

    /// One line per active second. `reason=` is present only when frames were
    /// actually dropped, so a grep for the reason slug finds exactly the
    /// windows where the host window went stale.
    fn format_line(dt: u64, published: u64, dropped: u64) -> String {
        if dropped == 0 {
            return format!(
                "window_publish window_ms={dt} published={published} dropped={dropped}"
            );
        }
        observe::Emit::decline("window_publish", &WindowPublishDrop::ResidentNotReady)
            .field("window_ms", dt)
            .field("published", published)
            .field("dropped", dropped)
            .render()
    }

    #[cfg(test)]
    pub(crate) fn reset() {
        for a in [&PUBLISHED, &DROPPED, &WINDOW_START_MS] {
            a.store(0, Ordering::Relaxed);
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// A clean window names no reason — a reader greps the slug to find
        /// only the windows where the window actually went stale.
        #[test]
        fn healthy_window_carries_no_reason() {
            let line = format_line(1000, 120, 0);
            assert!(line.contains("published=120 dropped=0"), "{line}");
            assert!(!line.contains("reason="), "{line}");
        }

        /// Any drop names the class, so the silent-freeze case is greppable.
        #[test]
        fn dropped_frames_name_the_reason() {
            let line = format_line(1000, 0, 118);
            assert!(line.contains("dropped=118"), "{line}");
            assert!(
                line.contains("reason=window_publish_resident_not_ready"),
                "{line}"
            );
        }

        /// An idle second emits nothing at all — the proxy must not flood a log
        /// with empty windows while the guest is not presenting.
        #[test]
        fn idle_window_emits_nothing() {
            reset();
            assert_eq!(maybe_line_at(0), None, "first call only arms the window");
            assert_eq!(maybe_line_at(WINDOW_MS + 1), None, "no samples, no line");
        }

        /// The window only closes once WINDOW_MS has elapsed.
        #[test]
        fn line_waits_for_the_full_window() {
            reset();
            assert_eq!(maybe_line_at(10), None);
            PUBLISHED.store(5, Ordering::Relaxed);
            assert_eq!(maybe_line_at(10 + WINDOW_MS - 1), None);
            let line = maybe_line_at(10 + WINDOW_MS).expect("window closed");
            assert!(line.contains("published=5"), "{line}");
        }
    }
}

pub mod capture_source {
    use crate::observe;
    use std::sync::atomic::{AtomicU64, Ordering};

    static RESIDENT: AtomicU64 = AtomicU64::new(0);
    static GUEST: AtomicU64 = AtomicU64::new(0);
    static WINDOW_START_MS: AtomicU64 = AtomicU64::new(0);

    const WINDOW_MS: u64 = 1000;

    /// Count one full-capture source decision: `resident`=the GPU resident
    /// supplied the frame, else the guest-page fallback ran.
    pub fn note(resident: bool) {
        if resident {
            RESIDENT.fetch_add(1, Ordering::Relaxed);
        } else {
            GUEST.fetch_add(1, Ordering::Relaxed);
        }
        if let Some(line) = maybe_line_at(observe::elapsed_ms() as u64) {
            observe::off(line);
        }
    }

    fn maybe_line_at(now: u64) -> Option<String> {
        let start = WINDOW_START_MS.load(Ordering::Relaxed);
        if start == 0 {
            let _ = WINDOW_START_MS.compare_exchange(0, now, Ordering::Relaxed, Ordering::Relaxed);
            return None;
        }
        let dt = now.saturating_sub(start);
        if dt < WINDOW_MS {
            return None;
        }
        if WINDOW_START_MS
            .compare_exchange(start, now, Ordering::Relaxed, Ordering::Relaxed)
            .is_err()
        {
            return None;
        }
        let resident = RESIDENT.swap(0, Ordering::Relaxed);
        let guest = GUEST.swap(0, Ordering::Relaxed);
        if resident.saturating_add(guest) == 0 {
            return None;
        }
        Some(format_line(dt, resident, guest))
    }

    fn format_line(dt: u64, resident: u64, guest: u64) -> String {
        let n = resident.saturating_add(guest).max(1);
        let frac = resident as f64 / n as f64;
        format!(
            "capture_source window_ms={dt} resident={resident} guest={guest} \
             resident_frac={frac:.2}"
        )
    }

    #[cfg(test)]
    pub(crate) fn reset() {
        for a in [&RESIDENT, &GUEST, &WINDOW_START_MS] {
            a.store(0, Ordering::Relaxed);
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn format_line_reports_split_and_fraction() {
            let line = format_line(1000, 3, 1);
            assert!(line.contains("resident=3"), "{line}");
            assert!(line.contains("guest=1"), "{line}");
            assert!(line.contains("resident_frac=0.75"), "{line}");
        }

        /// Counters are bumped directly (not via `note`, which stamps the window
        /// from the real clock) so the window boundaries stay deterministic.
        #[test]
        fn seeds_then_emits_once_past_window_and_stays_silent_when_idle() {
            reset();
            assert_eq!(maybe_line_at(1), None, "first call only opens the window");
            RESIDENT.fetch_add(3, Ordering::Relaxed);
            GUEST.fetch_add(1, Ordering::Relaxed);
            assert!(
                maybe_line_at(WINDOW_MS).is_none(),
                "inside the window: no line"
            );
            let line = maybe_line_at(1 + WINDOW_MS + 5).expect("window closed with samples");
            assert!(line.contains("resident=3"), "{line}");
            assert!(line.contains("guest=1"), "{line}");
            assert!(
                maybe_line_at(1 + 2 * WINDOW_MS + 10).is_none(),
                "an idle window emits nothing (counters drained)"
            );
        }
    }
}

/// Always-on windowed census of the per-present dmabuf EXPORT cost (route B).
///
/// Under a dmabuf-carried display, `publish_window_frame` calls
/// `export_present_dmabuf` every present, which submits a synchronous GPU blit
/// (OPTIMAL resident → LINEAR export image) and WAITS on its fence on the drain
/// worker before returning the fd
/// (`export_present_from_resident_fd_policy` → `retire_all`).
/// This is the biggest remaining per-present serialization once the CPU readback
/// and writeback prefetch are elided, but its cost is otherwise lumped into
/// `retire_wait_us`. This census isolates it: `hits` (a dmabuf fd was produced) /
/// `misses` (None → CPU fallback) are COUNT-based → trustworthy under the
/// SCHED_IDLE agent boot; `us_avg` / `us_max` are wall-clock so they are
/// **SCHED_IDLE-contaminated for the agent** — only the USER at SCHED_OTHER reads
/// a real per-present blit cost from them. It exists to answer, evidence-first,
/// whether the export blit is the fps bottleneck before the (hard, cross-device-
/// sync) async-export work is attempted. Measure-only — never gates.
pub mod export_present {
    use crate::observe;
    use std::sync::atomic::{AtomicU64, Ordering};

    static HITS: AtomicU64 = AtomicU64::new(0);
    static MISSES: AtomicU64 = AtomicU64::new(0);
    static TOTAL_US: AtomicU64 = AtomicU64::new(0);
    static MAX_US: AtomicU64 = AtomicU64::new(0);
    // Split of the export-present wall-clock into its two synchronous phases, so
    // we can tell inherent guest-composite-drain from our own blit cost BEFORE
    // committing to the hard cross-device-sync async-export work. `drain` is the
    // pre-blit `begin_entry_sync` (waits for the guest's in-flight compositing
    // draws into the resident — inherent, not ours to remove); `blit` is the
    // post-submit `retire_all` fence wait for our OPTIMAL→LINEAR export copy
    // (bandwidth-bound, scales ~4x from 1080p to 4K — the optimizable part).
    static DRAIN_US: AtomicU64 = AtomicU64::new(0);
    static BLIT_US: AtomicU64 = AtomicU64::new(0);
    static PHASE_N: AtomicU64 = AtomicU64::new(0);
    static WINDOW_START_MS: AtomicU64 = AtomicU64::new(0);

    const WINDOW_MS: u64 = 1000;

    /// Count one export attempt: `hit`=a dmabuf fd was produced, `us`=wall-clock
    /// of the whole `export_present_dmabuf` call (blit submit + fence wait).
    pub fn note(hit: bool, us: u64) {
        if hit {
            HITS.fetch_add(1, Ordering::Relaxed);
        } else {
            MISSES.fetch_add(1, Ordering::Relaxed);
        }
        TOTAL_US.fetch_add(us, Ordering::Relaxed);
        MAX_US.fetch_max(us, Ordering::Relaxed);
        if let Some(line) = maybe_line_at(observe::elapsed_ms() as u64) {
            observe::off(line);
        }
    }

    /// Record the per-present phase split (measure-only). `drain_us` =
    /// `begin_entry_sync` (guest-composite drain), `blit_us` = `retire_all`
    /// (our export blit fence wait). Emitted as `drain_us_avg`/`blit_us_avg` in
    /// the window line. Called once per successful export from the engine.
    pub fn note_phases(drain_us: u64, blit_us: u64) {
        DRAIN_US.fetch_add(drain_us, Ordering::Relaxed);
        BLIT_US.fetch_add(blit_us, Ordering::Relaxed);
        PHASE_N.fetch_add(1, Ordering::Relaxed);
    }

    fn maybe_line_at(now: u64) -> Option<String> {
        let start = WINDOW_START_MS.load(Ordering::Relaxed);
        if start == 0 {
            let _ = WINDOW_START_MS.compare_exchange(0, now, Ordering::Relaxed, Ordering::Relaxed);
            return None;
        }
        let dt = now.saturating_sub(start);
        if dt < WINDOW_MS {
            return None;
        }
        if WINDOW_START_MS
            .compare_exchange(start, now, Ordering::Relaxed, Ordering::Relaxed)
            .is_err()
        {
            return None;
        }
        let hits = HITS.swap(0, Ordering::Relaxed);
        let misses = MISSES.swap(0, Ordering::Relaxed);
        let total_us = TOTAL_US.swap(0, Ordering::Relaxed);
        let max_us = MAX_US.swap(0, Ordering::Relaxed);
        let drain_us = DRAIN_US.swap(0, Ordering::Relaxed);
        let blit_us = BLIT_US.swap(0, Ordering::Relaxed);
        let phase_n = PHASE_N.swap(0, Ordering::Relaxed);
        let n = hits.saturating_add(misses);
        if n == 0 {
            return None;
        }
        Some(format_line(
            dt, hits, misses, total_us, max_us, drain_us, blit_us, phase_n,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn format_line(
        dt: u64,
        hits: u64,
        misses: u64,
        total_us: u64,
        max_us: u64,
        drain_us: u64,
        blit_us: u64,
        phase_n: u64,
    ) -> String {
        let n = hits.saturating_add(misses).max(1);
        let us_avg = total_us / n;
        let pn = phase_n.max(1);
        let drain_avg = drain_us / pn;
        let blit_avg = blit_us / pn;
        let hz = hits.saturating_mul(1000) as f64 / dt.max(1) as f64;
        // `us_avg`/`us_max` are SCHED_IDLE-contaminated under the agent; the USER
        // reads the real per-present blit cost at SCHED_OTHER. hits/misses trusted.
        // drain_us_avg = inherent guest-composite drain; blit_us_avg = our export
        // blit fence wait (the optimizable part). See export_present module docs.
        format!(
            "export_present window_ms={dt} hits={hits} misses={misses} hit_hz={hz:.1} \
             us_avg={us_avg} us_max={max_us} drain_us_avg={drain_avg} blit_us_avg={blit_avg}"
        )
    }

    #[cfg(test)]
    pub(crate) fn reset() {
        for a in [
            &HITS,
            &MISSES,
            &TOTAL_US,
            &MAX_US,
            &DRAIN_US,
            &BLIT_US,
            &PHASE_N,
            &WINDOW_START_MS,
        ] {
            a.store(0, Ordering::Relaxed);
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn window_reports_hits_misses_and_avg() {
            // 90 phase samples: drain 90*1200, blit 90*800.
            let line = format_line(
                1000,
                90,
                10,
                90 * 2000 + 10 * 500,
                5000,
                90 * 1200,
                90 * 800,
                90,
            );
            assert!(line.contains("hits=90 misses=10"));
            assert!(line.contains("us_max=5000"));
            // avg over all 100 attempts = (90*2000+10*500)/100 = 1850.
            assert!(line.contains("us_avg=1850"), "{line}");
            // phase split averages over the 90 successful exports.
            assert!(line.contains("drain_us_avg=1200"), "{line}");
            assert!(line.contains("blit_us_avg=800"), "{line}");
        }

        #[test]
        fn seeds_then_emits_once_past_window() {
            reset();
            assert_eq!(maybe_line_at(1), None);
            HITS.fetch_add(3, Ordering::Relaxed);
            TOTAL_US.fetch_add(6000, Ordering::Relaxed);
            assert!(maybe_line_at(WINDOW_MS).is_none());
            let line = maybe_line_at(1 + WINDOW_MS + 5).expect("line");
            assert!(line.contains("hits=3 misses=0"));
            assert!(maybe_line_at(1 + 2 * WINDOW_MS + 10).is_none());
        }
    }
}

/// Always-on census of the render-deferred **window-cap force-flush**
/// (`import_present::try_defer_present_store` flushing the oldest window when the
/// live population exceeds `RENDER_DEFERRED_WINDOW_CAP`). The cap bounds the
/// pinned registry, but a force-flush is a GPU->guest writeback landed early —
/// if a burst force-flushes fast enough this is itself a stall source. This
/// makes that cost a visible line instead of an unexplained hitch: silent unless
/// a force-flush actually happened in the window.
pub mod cap_flush {
    use crate::observe;
    use std::sync::atomic::{AtomicU64, Ordering};

    static FLUSHES: AtomicU64 = AtomicU64::new(0);
    static TOTAL_US: AtomicU64 = AtomicU64::new(0);
    static MAX_US: AtomicU64 = AtomicU64::new(0);
    static PEAK_BATCH: AtomicU64 = AtomicU64::new(0);
    static WINDOW_START_MS: AtomicU64 = AtomicU64::new(0);

    const WINDOW_MS: u64 = 1000;

    /// Record one cap-driven force-flush *batch*: `n` windows landed in this
    /// `try_defer_present_store` call, taking `us` wall-clock total.
    pub fn note(n: u64, us: u64) {
        if n == 0 {
            return;
        }
        FLUSHES.fetch_add(n, Ordering::Relaxed);
        TOTAL_US.fetch_add(us, Ordering::Relaxed);
        MAX_US.fetch_max(us, Ordering::Relaxed);
        PEAK_BATCH.fetch_max(n, Ordering::Relaxed);
        if let Some(line) = maybe_line_at(observe::elapsed_ms() as u64) {
            observe::off(line);
        }
    }

    fn maybe_line_at(now: u64) -> Option<String> {
        let start = WINDOW_START_MS.load(Ordering::Relaxed);
        if start == 0 {
            let _ = WINDOW_START_MS.compare_exchange(0, now, Ordering::Relaxed, Ordering::Relaxed);
            return None;
        }
        let dt = now.saturating_sub(start);
        if dt < WINDOW_MS {
            return None;
        }
        if WINDOW_START_MS
            .compare_exchange(start, now, Ordering::Relaxed, Ordering::Relaxed)
            .is_err()
        {
            return None;
        }
        let flushes = FLUSHES.swap(0, Ordering::Relaxed);
        let total_us = TOTAL_US.swap(0, Ordering::Relaxed);
        let max_us = MAX_US.swap(0, Ordering::Relaxed);
        let peak_batch = PEAK_BATCH.swap(0, Ordering::Relaxed);
        if flushes == 0 {
            return None;
        }
        Some(format_line(dt, flushes, total_us, max_us, peak_batch))
    }

    fn format_line(dt: u64, flushes: u64, total_us: u64, max_us: u64, peak_batch: u64) -> String {
        let us_avg = total_us / flushes.max(1);
        // `us_*` are SCHED_IDLE-contaminated under the agent; `flushes`/`peak_batch`
        // are count-trustworthy. A high, sustained `flushes` means the cap is set
        // below the workload's live working set (raise it or the drain lags).
        format!(
            "cap_flush window_ms={dt} flushes={flushes} peak_batch={peak_batch} \
             us_avg={us_avg} us_max={max_us}"
        )
    }

    #[cfg(test)]
    pub(crate) fn reset() {
        for a in [&FLUSHES, &TOTAL_US, &MAX_US, &PEAK_BATCH, &WINDOW_START_MS] {
            a.store(0, Ordering::Relaxed);
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn format_reports_flushes_batch_and_avg() {
            // 8 flushes, 5500 us total → avg 687; peak batch 5, max 4000.
            let line = format_line(1000, 8, 5500, 4000, 5);
            assert!(line.contains("flushes=8"), "{line}");
            assert!(line.contains("peak_batch=5"), "{line}");
            assert!(line.contains("us_avg=687"), "{line}");
            assert!(line.contains("us_max=4000"), "{line}");
        }

        #[test]
        fn silent_until_a_flush_then_reports_batch_peak() {
            // Seed the window with synthetic times; poke the atomics directly
            // rather than via note() (which stamps the real clock — the full
            // serial suite advances elapsed_ms past any synthetic dt).
            reset();
            assert_eq!(maybe_line_at(1), None);
            // A window with no flushes stays silent.
            assert!(maybe_line_at(1 + WINDOW_MS).is_none());
            FLUSHES.fetch_add(8, Ordering::Relaxed);
            TOTAL_US.fetch_add(5500, Ordering::Relaxed);
            PEAK_BATCH.fetch_max(5, Ordering::Relaxed);
            let line = maybe_line_at(1 + 2 * WINDOW_MS + 5).expect("line");
            assert!(line.contains("flushes=8"), "{line}");
            assert!(line.contains("peak_batch=5"), "{line}");
        }
    }
}

/// Always-on census of the resident-target **idle drain** — non-pinned residents
/// reclaimed because they went untouched for `IDLE_TARGET_AGE_MS` of wall-clock.
/// This is how VRAM returns to the working-set baseline after a compositing
/// burst — including on a static page where publishes have stopped (the drain is
/// clocked off the poll heartbeat, not publish count). The line makes that
/// reclamation visible (and its absence, if the drain silently stalled,
/// diagnosable). Silent unless a drain actually happened in the window.
pub mod idle_drain {
    use crate::observe;
    use std::sync::atomic::{AtomicU64, Ordering};

    static DRAINED: AtomicU64 = AtomicU64::new(0);
    static WINDOW_START_MS: AtomicU64 = AtomicU64::new(0);

    const WINDOW_MS: u64 = 1000;

    /// Record `n` residents reclaimed by one drain pass (0 is ignored).
    pub fn note(n: u64) {
        if n == 0 {
            return;
        }
        DRAINED.fetch_add(n, Ordering::Relaxed);
        if let Some(line) = maybe_line_at(observe::elapsed_ms() as u64) {
            observe::off(line);
        }
    }

    fn maybe_line_at(now: u64) -> Option<String> {
        let start = WINDOW_START_MS.load(Ordering::Relaxed);
        if start == 0 {
            let _ = WINDOW_START_MS.compare_exchange(0, now, Ordering::Relaxed, Ordering::Relaxed);
            return None;
        }
        let dt = now.saturating_sub(start);
        if dt < WINDOW_MS {
            return None;
        }
        if WINDOW_START_MS
            .compare_exchange(start, now, Ordering::Relaxed, Ordering::Relaxed)
            .is_err()
        {
            return None;
        }
        let drained = DRAINED.swap(0, Ordering::Relaxed);
        if drained == 0 {
            return None;
        }
        Some(format!(
            "idle_target_drain window_ms={dt} drained={drained}"
        ))
    }

    #[cfg(test)]
    pub(crate) fn reset() {
        DRAINED.store(0, Ordering::Relaxed);
        WINDOW_START_MS.store(0, Ordering::Relaxed);
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn silent_until_a_drain_then_sums_window() {
            reset();
            assert_eq!(maybe_line_at(1), None);
            assert!(maybe_line_at(1 + WINDOW_MS).is_none());
            DRAINED.fetch_add(7, Ordering::Relaxed);
            let line = maybe_line_at(1 + 2 * WINDOW_MS + 5).expect("line");
            assert!(line.contains("drained=7"), "{line}");
        }
    }
}

/// Record a successful present retain (after `capture_present_frame` fills +0x188).
/// Composite/render throughput over one summary window: presents delta / dt.
/// `prev_ms == 0` marks the first summary (no prior window) → `0.0`; a
/// zero/backwards `dt` also yields `0.0` (the monotonic clock cannot regress,
/// but be defensive). Saturating so a counter reset never underflows.
fn present_hz_over_window(prev_ms: u64, prev_presents: u64, now_ms: u64, presents: u64) -> f64 {
    if prev_ms == 0 {
        return 0.0;
    }
    let dt = now_ms.saturating_sub(prev_ms);
    if dt == 0 {
        return 0.0;
    }
    let dp = presents.saturating_sub(prev_presents);
    dp as f64 * 1000.0 / dt as f64
}

pub fn note_capture_ok(sample: PresentCaptureSample) {
    let mut st = STATE.lock().unwrap_or_else(|e| e.into_inner());
    st.presents = st.presents.saturating_add(1);
    let n = st.presents;
    let pixels = (sample.width as u64).saturating_mul(sample.height as u64);
    let total_bytes = pixels.saturating_mul(4);

    if pixels >= MIN_PROXY_PIXELS && total_bytes > 0 {
        // RGB occupancy: alpha-only DisplaySwap clears (byte_nz full, rgb_nz 0)
        // are empty presents — the dual-mid black class from live OFF logs.
        let rgb_frac = sample.rgb_nz as f64 / pixels as f64;
        if sample.rgb_nz == 0 || rgb_frac < SPARSE_NZ_FRAC {
            st.sparse = st.sparse.saturating_add(1);
            thrash_line(&format!(
                "sparse_present mid={} gen={} {}x{} nz={} rgb_nz={} max={} max_rgb={} frac={:.4} rgb_frac={:.4} last_store={}",
                sample.mapping_id,
                sample.generation,
                sample.width,
                sample.height,
                sample.nz,
                sample.rgb_nz,
                sample.max_byte,
                sample.max_rgb,
                sample.nz as f64 / total_bytes as f64,
                rgb_frac,
                sample.from_last_store as u8
            ));
            // Named alias for greps (`sparse_front`); same condition as sparse_present.
            // Measure-only — never gates present/decode/execute. No size heuristic.
            thrash_line(&format!(
                "sparse_front mid={} gen={} {}x{} rgb_nz={} rgb_frac={:.4}",
                sample.mapping_id,
                sample.generation,
                sample.width,
                sample.height,
                sample.rgb_nz,
                rgb_frac
            ));
        } else if rgb_frac >= CONVERGE_DENSE_FRAC && !st.first_dense_seen {
            // Boot-convergence: the first full-size present that composites the
            // wallpaper/desktop (not chrome-only-over-black). The trailing
            // `t=<ms>` field on the fail-log copy of this THRASH line is the
            // wall-clock time to a full desktop — the user-visible "console →
            // logo → desktop" latency. `sparse_before` / `presents_before`
            // quantify how long the dual-mid strobe ran first. Fires once per
            // boot; a boot that never converges leaves `converged=0` in the
            // periodic summary rather than no signal at all (the pathological
            // slow boot otherwise emits *less* than a healthy one).
            st.first_dense_seen = true;
            thrash_line(&format!(
                "present_converge mid={} gen={} {}x{} rgb_nz={} rgb_frac={:.4} presents_before={} sparse_before={}",
                sample.mapping_id,
                sample.generation,
                sample.width,
                sample.height,
                sample.rgb_nz,
                rgb_frac,
                n.saturating_sub(1),
                st.sparse
            ));
        }

        // Post-converge regression: once the desktop has fully composited, a
        // sustained run of non-dense full-size presents means it reverted to the
        // login/boot overlay or a sparse strobe. A dense present
        // clears the run; a normal desktop present captures the full composited
        // +0x188 so it stays dense even for a tiny damage update. Measure-only.
        if st.first_dense_seen {
            if rgb_frac >= CONVERGE_DENSE_FRAC {
                st.post_converge_nondense_run = 0;
            } else {
                st.post_converge_nondense_run = st.post_converge_nondense_run.saturating_add(1);
                if st
                    .post_converge_nondense_run
                    .is_multiple_of(POST_CONVERGE_REGRESS_RUN)
                {
                    st.post_converge_regress = st.post_converge_regress.saturating_add(1);
                    // Two structural discriminators (measure-only; the fire
                    // condition stays occupancy-based so a real overlay/strobe is
                    // never missed): `gen_adv` is whether a FRESH full-frame Store
                    // landed since the previous present (generation advanced) vs.
                    // re-presenting a STUCK retained frame, and `edge` is the
                    // subsampled luma-edge structure. A sustained legitimately-dark
                    // fullscreen app (a dark video/game/site — the false-positive
                    // class) reads `gen_adv=1 edge>0` (live, structured content just
                    // below the 0.50 dense threshold); a genuine reversion to the
                    // login/boot overlay or a strobe reads `gen_adv=0` (stuck) and/or
                    // near-zero `edge` (flat) with a much lower `rgb_frac` (~0.13).
                    // So an auditor can tell benign-dark-fullscreen from a real
                    // regression straight from the always-on line.
                    let gen_adv = st.last.is_none_or(|p| p.generation != sample.generation);
                    thrash_line(&format!(
                        "post_converge_regress mid={} gen={} gen_adv={} edge={} {}x{} rgb_nz={} rgb_frac={:.4} nondense_run={} episodes={}",
                        sample.mapping_id,
                        sample.generation,
                        gen_adv as u8,
                        sample.edge_energy,
                        sample.width,
                        sample.height,
                        sample.rgb_nz,
                        rgb_frac,
                        st.post_converge_nondense_run,
                        st.post_converge_regress
                    ));
                }
            }
        }
    }

    if let Some(p) = st.last {
        if p.width != sample.width || p.height != sample.height {
            st.geom_mismatch = st.geom_mismatch.saturating_add(1);
            thrash_line(&format!(
                "geom_mismatch prev={}x{} mid={} → {}x{} mid={} gen={}",
                p.width,
                p.height,
                p.mapping_id,
                sample.width,
                sample.height,
                sample.mapping_id,
                sample.generation
            ));
        }
        if p.mapping_id != sample.mapping_id && p.mapping_id != 0 && sample.mapping_id != 0 {
            st.mid_switches = st.mid_switches.saturating_add(1);
            if pixels >= MIN_PROXY_PIXELS {
                // Prefer RGB occupancy so alpha-only black mid2↔mid3 (byte_nz full)
                // does not hide the dual-mid empty present class.
                let lo = p.rgb_nz.min(sample.rgb_nz) as f64;
                let hi = p.rgb_nz.max(sample.rgb_nz) as f64;
                // When both are empty (rgb 0), still not an nz_swing — both black.
                if hi > 0.0 && (lo / hi) < NZ_SWING_RATIO {
                    st.nz_swings = st.nz_swings.saturating_add(1);
                    thrash_line(&format!(
                        "nz_swing mid {}→{} gen {}→{} nz {}→{} rgb_nz {}→{} ratio={:.3} {}x{} last_store {}→{}",
                        p.mapping_id,
                        sample.mapping_id,
                        p.generation,
                        sample.generation,
                        p.nz,
                        sample.nz,
                        p.rgb_nz,
                        sample.rgb_nz,
                        lo / hi,
                        sample.width,
                        sample.height,
                        p.from_last_store as u8,
                        sample.from_last_store as u8
                    ));
                } else {
                    thrash_line(&format!(
                        "mid_switch {}→{} gen {}→{} nz {}→{} rgb_nz {}→{} {}x{}",
                        p.mapping_id,
                        sample.mapping_id,
                        p.generation,
                        sample.generation,
                        p.nz,
                        sample.nz,
                        p.rgb_nz,
                        sample.rgb_nz,
                        sample.width,
                        sample.height
                    ));
                }
                // Structure thrash: equal-ish nz but edge energy diverges
                // (wallpaper-filled holes in shattered Safari still count as nz).
                let elo = p.edge_energy.min(sample.edge_energy) as f64;
                let ehi = p.edge_energy.max(sample.edge_energy) as f64;
                if ehi > 0.0 && (elo / ehi) < STRUCTURE_SWING_RATIO {
                    st.structure_swings = st.structure_swings.saturating_add(1);
                    thrash_line(&format!(
                        "structure_swing mid {}→{} gen {}→{} edge {}→{} ratio={:.3} nz {}→{} {}x{}",
                        p.mapping_id,
                        sample.mapping_id,
                        p.generation,
                        sample.generation,
                        p.edge_energy,
                        sample.edge_energy,
                        elo / ehi,
                        p.nz,
                        sample.nz,
                        sample.width,
                        sample.height
                    ));
                }
            }
        }
    }
    st.last = Some(sample);

    if n.is_multiple_of(SUMMARY_EVERY) {
        // Composite/render throughput over this summary window (drain worker,
        // off main core). First summary has no prior window → 0.0.
        let now_ms = crate::observe::elapsed_ms() as u64;
        let present_hz = present_hz_over_window(
            st.last_summary_ms,
            st.last_summary_presents,
            now_ms,
            st.presents,
        );
        st.last_summary_ms = now_ms;
        st.last_summary_presents = st.presents;
        thrash_line(&format!(
            "summary presents={} present_hz={:.1} mid_sw={} nz_sw={} struct_sw={} sparse={} geom={} fail={} peer_divergence={} converged={} post_converge_regress={} stale_online={} t11_fb={}",
            st.presents,
            present_hz,
            st.mid_switches,
            st.nz_swings,
            st.structure_swings,
            st.sparse,
            st.geom_mismatch,
            st.capture_fail,
            st.selected_peer_divergence,
            st.first_dense_seen as u8,
            st.post_converge_regress,
            st.stale_online_pending,
            st.t11_fb_total
        ));
    }
}

/// Subsampled horizontal edge energy (BGRA luma). Measure-only structure proxy.
///
/// Returns **total** edge energy scaled by `>> 8` (not the per-sample average).
/// Live 1440×1080 wallpaper averages ~0–1 abs-luma/sample so a mean-based
/// score collapsed every frame to 0/1 and structure_swing never fired on
/// equal-nz dual-mid thrash (full dock vs partial glass, boxy Safari). The
/// integrated total still separates chrome-rich frames from smooth wallpaper.
pub fn edge_energy_bgra(frame_bgra: &[u8], width: u32, height: u32) -> u32 {
    if width < 8 || height < 8 {
        return 0;
    }
    let stride = (width as usize).saturating_mul(4);
    let need = stride.saturating_mul(height as usize);
    if frame_bgra.len() < need {
        return 0;
    }
    let mut sum: u64 = 0;
    let mut y = 0u32;
    while y < height {
        let row = (y as usize) * stride;
        let mut x = 0u32;
        while x + 1 < width {
            let o0 = row + (x as usize) * 4;
            let o1 = o0 + 4;
            // Approximate luma from BGRA: 0.25R+0.5G+0.25B
            let l0 = (frame_bgra[o0 + 2] as u32)
                .saturating_add((frame_bgra[o0 + 1] as u32) << 1)
                .saturating_add(frame_bgra[o0] as u32)
                >> 2;
            let l1 = (frame_bgra[o1 + 2] as u32)
                .saturating_add((frame_bgra[o1 + 1] as u32) << 1)
                .saturating_add(frame_bgra[o1] as u32)
                >> 2;
            sum += l0.abs_diff(l1) as u64;
            x = x.saturating_add(4);
        }
        y = y.saturating_add(4);
    }
    // Scale so a full 1440×1080 desktop fits in u32 without saturating tests.
    (sum >> 8).min(u32::MAX as u64) as u32
}

/// Measure-only: fragment/vertex sample resolved but all-zero payload.
///
/// Proxies empty Favourites tiles / zero icon RTs without gating encode.
/// Logs:
/// - mid-size textures (32…512) — icon / strip tiles
/// - **display-sized** (≥1280×720) — wallpaper / multi-bind layer class
///   (previously filtered out, so empty desktop layers were invisible)
///
/// Filters default 1×1 clears and other tiny nulls.
pub fn note_empty_sample_if(texture_ref: u32, w: u32, h: u32, rgba: &[u8], stage: &str) {
    let mid_tile = w >= 32 && h >= 32 && w <= 512 && h <= 512;
    let display = w >= 1280 && h >= 720;
    if !mid_tile && !display {
        return;
    }
    // RGB-only: covered black (0,0,0,A=255) is empty content for wallpaper
    // layers — byte nonzero_stats would miss it (alpha alone).
    let (rgb_nz, max_rgb, _) = observe::rgba_rgb_stats(rgba);
    if rgb_nz != 0 {
        return;
    }
    let kind = if display { "display" } else { "tile" };
    thrash_line(&format!(
        "empty_sample stage={stage} kind={kind} ref={texture_ref} {w}x{h} max_rgb={max_rgb}"
    ));
}

/// Record a failed DisplaySwap capture (retain not updated).
pub fn note_capture_fail(mapping_id: u32, width: u32, height: u32, generation: u32) {
    let mut st = STATE.lock().unwrap_or_else(|e| e.into_inner());
    st.capture_fail = st.capture_fail.saturating_add(1);
    thrash_line(&format!(
        "capture_fail mid={mapping_id} {width}x{height} gen={generation}"
    ));
}

/// Snapshot counters (tests / external poll).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ThrashCounters {
    pub presents: u64,
    pub mid_switches: u64,
    pub nz_swings: u64,
    pub structure_swings: u64,
    pub sparse: u64,
    pub geom_mismatch: u64,
    pub capture_fail: u64,
    pub selected_peer_divergence: u64,
    /// True once a full-size present reached [`CONVERGE_DENSE_FRAC`] occupancy
    /// (the desktop first fully composited this boot).
    pub converged: bool,
    /// Distinct post-converge regression episodes (desktop → overlay/strobe).
    pub post_converge_regress: u64,
    /// Post-converge display IRQs raised with the ONLINE bit still pending.
    pub stale_online_pending: u64,
}

/// Whether this boot has reached boot-convergence (a full-size present crossed
/// [`CONVERGE_DENSE_FRAC`] and `present_converge` fired). Latched for the boot.
/// Read by the display-lifecycle path to correlate a guest display reinit that
/// arrives *after* convergence — the smoking gun for the post-converge
/// boot-progress overlay. Measure-only.
pub fn has_converged() -> bool {
    STATE
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .first_dense_seen
}

/// Record that a post-ack display IRQ (`src` = vbl|present) was raised while the
/// shared-page ONLINE bit (bit2) was still pending — meaning the guest will
/// re-dispatch `process_online` → `connectionChange` and re-composite the
/// boot-progress overlay (x86 RE 2026-07-17: the host-driven strobe source).
/// Counts every occurrence; emits an always-on `stale_online_pending` line only
/// the **first** time per boot (VBL runs ~60 Hz — a per-call line would flood).
/// Measure-only; never gates the IRQ.
pub fn note_stale_online_pending(src: &str, pending: u32) {
    let mut st = STATE.lock().unwrap_or_else(|e| e.into_inner());
    st.stale_online_pending = st.stale_online_pending.saturating_add(1);
    if !st.stale_online_logged {
        st.stale_online_logged = true;
        let n = st.stale_online_pending;
        drop(st);
        observe::fail(format!(
            "stale_online_pending src={src} pending={pending:#x} count={n}"
        ));
    }
}

pub fn counters() -> ThrashCounters {
    let st = STATE.lock().unwrap_or_else(|e| e.into_inner());
    ThrashCounters {
        presents: st.presents,
        mid_switches: st.mid_switches,
        nz_swings: st.nz_swings,
        structure_swings: st.structure_swings,
        sparse: st.sparse,
        geom_mismatch: st.geom_mismatch,
        capture_fail: st.capture_fail,
        selected_peer_divergence: st.selected_peer_divergence,
        converged: st.first_dense_seen,
        post_converge_regress: st.post_converge_regress,
        stale_online_pending: st.stale_online_pending,
    }
}

/// Test-only reset (unit tests). Safe while holding [`test_exclusive`]
/// (unlike [`reset_for_device`], which takes the shared side of the guard).
#[cfg(test)]
pub fn reset_for_test() {
    reset_state_inner();
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Serialize all present_proxy tests — they share global thrash STATE —
    /// and exclude parallel device resets ([`test_exclusive`]).
    fn test_lock() -> std::sync::RwLockWriteGuard<'static, ()> {
        test_exclusive()
    }

    #[test]
    fn present_hz_window_math_and_edges() {
        // First summary: no prior window → 0.0 (prev_ms == 0), regardless of
        // the present count, so a boot never reports a bogus spike.
        assert_eq!(present_hz_over_window(0, 0, 1_000, 32), 0.0);
        assert_eq!(present_hz_over_window(0, 100, 1_000, 132), 0.0);
        // 32 presents over exactly 1 s → 32 Hz (the observed idle/scroll band).
        assert_eq!(present_hz_over_window(1_000, 100, 2_000, 132), 32.0);
        // 60 presents over 500 ms → 120 Hz (headroom case).
        assert_eq!(present_hz_over_window(1_000, 0, 1_500, 60), 120.0);
        // Zero elapsed (two summaries same ms) → 0.0, no divide-by-zero.
        assert_eq!(present_hz_over_window(5_000, 100, 5_000, 132), 0.0);
        // Defensive: a backwards clock or counter reset saturates to 0, never
        // underflows into a huge bogus rate.
        assert_eq!(present_hz_over_window(9_000, 100, 8_000, 132), 0.0);
        assert_eq!(present_hz_over_window(1_000, 200, 2_000, 100), 0.0);
    }

    /// The display-cadence census opens on the first poll, then emits once per
    /// window with the poll/VBL/present-complete counts accumulated since. It
    /// must stay silent until a full window elapses and then report the exact
    /// counts — so a throttled `poll_hz` (VBL starvation) reads straight off the
    /// log. Uses the injectable clock so the window boundary is deterministic.
    #[test]
    fn display_cadence_emits_once_per_window_with_exact_counts() {
        let _g = test_lock();
        cadence::reset();
        let w = cadence::window_ms();
        // Nonzero base: 0 is the "unseeded" sentinel, and the live clock (ms
        // since first log line) is never 0 by the time the device polls.
        let base = 5000u64;

        // First poll seeds the window and emits nothing yet.
        assert_eq!(cadence::note_poll_at(base), None, "seed poll must not emit");
        // Two VBLs + one present-complete land inside the open window.
        cadence::note_vbl();
        cadence::note_vbl();
        cadence::note_present_complete();
        // A poll still inside the window is silent.
        assert_eq!(
            cadence::note_poll_at(base + w - 1),
            None,
            "poll before the window closes must not emit"
        );

        // The poll that crosses the window boundary emits the census. Counts
        // include every poll/VBL/present-complete since the window opened: 3
        // polls (seed + in-window + boundary), 2 VBLs, 1 present-complete.
        let line = cadence::note_poll_at(base + w).expect("boundary poll must emit");
        assert!(line.contains("polls=3"), "line: {line}");
        assert!(line.contains("vbls=2"), "line: {line}");
        assert!(line.contains("present_completes=1"), "line: {line}");
        assert!(
            line.contains(&format!("window_ms={w}")),
            "window duration reported: {line}"
        );

        // Counters reset with the new window: the next boundary reports only the
        // fresh polls, no carry-over from the emitted window.
        assert_eq!(
            cadence::note_poll_at(base + w + 10),
            None,
            "new window just opened"
        );
        let line2 = cadence::note_poll_at(base + w + w).expect("second boundary emits");
        assert!(
            line2.contains("polls=2"),
            "counters reset per window: {line2}"
        );
        assert!(line2.contains("vbls=0"), "no VBLs this window: {line2}");
        cadence::reset();
    }

    /// The hitch threshold must scale with the live cadence, never a fixed ms:
    /// steady 30 fps (33 ms gaps) must not read as a stutter while a dropped
    /// frame at 120 fps (8 ms cadence) must. Below the floor the absolute 24 ms
    /// bar dominates so a barely-two-frame gap at high refresh is not named.
    #[test]
    fn hitch_threshold_is_cadence_relative_with_floor() {
        // 120 fps: ewma≈8 ms, K×ewma=20 ms → clamped up to the 24 ms floor.
        assert_eq!(hitch::threshold_for_test(8), 24);
        // 60 fps: ewma≈16 ms, K×ewma=40 ms → above the floor, relative wins.
        assert_eq!(hitch::threshold_for_test(16), 40);
        // 30 fps: ewma≈33 ms, K×ewma=82 ms — a normal 33 ms frame is far under.
        assert_eq!(hitch::threshold_for_test(33), 82);
        // Unseeded / zero ewma still yields the floor, never 0.
        assert_eq!(hitch::threshold_for_test(0), 24);
    }

    /// A smooth 120 fps stream (steady 8 ms gaps) fires no hitch and no event —
    /// the fail-visible contract's "silent on a healthy boot". Then a single
    /// 60 ms stalled frame is flagged: an immediate per-event line at the exact
    /// moment plus a `hitches`≥1 with `max_gap_ms`≥60 in the window summary.
    #[test]
    fn hitch_flags_a_dropped_frame_and_stays_silent_when_smooth() {
        let _g = test_lock();
        hitch::reset();
        let mut last = 5000u64;
        // Seed + a run of steady 8 ms gaps: every raise is smooth.
        hitch::record_for_test(last);
        for _ in 0..20 {
            last += 8;
            let (summary, event) = hitch::record_for_test(last);
            assert!(event.is_none(), "smooth cadence must not fire an event");
            assert!(summary.is_none(), "no window has closed yet at t={last}");
        }
        // One 60 ms stall — a dropped frame at 120 fps. Fires the per-event line.
        last += 60;
        let (_summary, event) = hitch::record_for_test(last);
        let ev = event.expect("a 60 ms gap at 8 ms cadence must fire an event");
        assert!(ev.contains("gap_ms=60"), "event names the gap: {ev}");
        // Roll to the window boundary with smooth gaps; the summary carries the
        // stall in max_gap_ms and hitches, which the averaged cadence line hides.
        let mut summary_line = None;
        for _ in 0..200 {
            last += 8;
            let (summary, _e) = hitch::record_for_test(last);
            if let Some(l) = summary {
                summary_line = Some(l);
                break;
            }
        }
        let line = summary_line.expect("a window must close within 200 raises");
        assert!(line.contains("max_gap_ms=60"), "worst gap surfaced: {line}");
        assert!(
            !line.contains("hitches=0"),
            "the stall must be counted: {line}"
        );
        hitch::reset();
    }

    /// A multi-second gap is the guest resuming after a static page (VBLs had
    /// stopped), not a stutter: it must land in `resumes`, never `hitches`, and
    /// must not poison `max_gap_ms` — otherwise every wake from idle would read
    /// as a giant hitch and the proxy would cry wolf.
    #[test]
    fn hitch_excludes_idle_resume_gap() {
        let _g = test_lock();
        hitch::reset();
        let mut last = 5000u64;
        hitch::record_for_test(last);
        // A short run of smooth 8 ms gaps establishes an 8 ms max, well within
        // the same window (80 ms ≪ the 1 s window).
        for _ in 0..9 {
            last += 8;
            let _ = hitch::record_for_test(last);
        }
        // 2 s gap: the page was static, VBLs paused, now one fires. This raise
        // also crosses the window boundary (dt ≫ 1 s), so it emits the summary.
        last += 2000;
        let (summary, event) = hitch::record_for_test(last);
        assert!(event.is_none(), "an idle-resume gap is not a stutter event");
        let l = summary.expect("the resume raise closes the window and emits");
        assert!(l.contains("resumes=1"), "resume counted apart: {l}");
        assert!(l.contains("hitches=0"), "resume is not a hitch: {l}");
        assert!(
            l.contains("max_gap_ms=8"),
            "resume gap must not poison the max: {l}"
        );
        hitch::reset();
    }

    /// Steady 10 fps (100 ms gaps, well under the 400 ms resume ceiling and far
    /// under its own 250 ms relative threshold) must produce a clean window with
    /// zero hitches, then reset — proving the summary is periodic and its
    /// counters do not carry across windows.
    #[test]
    fn hitch_summary_emits_once_per_window_and_resets() {
        let _g = test_lock();
        hitch::reset();
        let w = hitch::window_ms();
        let base = 5000u64;
        // Seed.
        assert_eq!(hitch::record_for_test(base), (None, None), "seed is silent");
        // Ten 100 ms gaps land exactly on the window boundary at base+w.
        let mut line = None;
        for i in 1..=(w / 100) {
            let (summary, event) = hitch::record_for_test(base + i * 100);
            assert!(event.is_none(), "100 ms steady is not a stutter");
            if i * 100 >= w {
                line = summary;
            } else {
                assert!(summary.is_none(), "window still open at +{}", i * 100);
            }
        }
        let l = line.expect("boundary raise emits the summary");
        assert!(l.contains("hitches=0"), "steady 10 fps has no hitch: {l}");
        assert!(l.contains("resumes=0"), "100 ms < resume ceiling: {l}");
        assert!(l.contains("max_gap_ms=100"), "worst gap is 100 ms: {l}");
        hitch::reset();
    }

    /// The teardown-churn proxy folds per-event delete/sync census into one ~1 s
    /// window summary: silent until a full window elapses, then reports the exact
    /// per-mode counts + summed deferred flushes, and — critically — a window
    /// that counted **nothing** emits no line at all (idle desktop must never add
    /// a churn line). Uses the injectable clock for a deterministic boundary.
    #[test]
    fn teardown_churn_summarizes_window_and_stays_silent_when_idle() {
        use lifecycle_churn::DeleteMode;
        let _g = test_lock();
        lifecycle_churn::reset();
        let w = 1000u64;
        let base = 5000u64;

        // Seed the window (nonzero base; 0 is the unseeded sentinel).
        assert_eq!(
            lifecycle_churn::test_maybe_line(base),
            None,
            "seed must not emit"
        );
        // A recycled-surface frame's worth of teardown lands in the window.
        lifecycle_churn::test_note_delete(DeleteMode::Condemn);
        lifecycle_churn::test_note_delete(DeleteMode::Condemn);
        lifecycle_churn::test_note_delete(DeleteMode::Dead);
        lifecycle_churn::test_note_synchronize(2);
        lifecycle_churn::test_note_synchronize(0);
        // Still inside the window → silent.
        assert_eq!(
            lifecycle_churn::test_maybe_line(base + w - 1),
            None,
            "before window close must not emit"
        );
        // Boundary poll emits the aggregated summary.
        let line = lifecycle_churn::test_maybe_line(base + w).expect("boundary emits");
        assert!(line.contains("deletes=3"), "line: {line}");
        assert!(line.contains("condemn=2"), "line: {line}");
        assert!(line.contains("dead=1"), "line: {line}");
        assert!(line.contains("unmapped=0"), "line: {line}");
        assert!(line.contains("sync=2"), "line: {line}");
        assert!(line.contains("deferred_flushed=2"), "line: {line}");
        assert!(line.contains(&format!("window_ms={w}")), "line: {line}");

        // Counters reset with the new window. An empty window (no events) must
        // return None — this is the "idle desktop never floods" invariant.
        assert_eq!(
            lifecycle_churn::test_maybe_line(base + w + w),
            None,
            "empty window emits nothing"
        );
        lifecycle_churn::reset();
    }

    /// The import-present proxy folds per-import `used=1` census into one ~1 s
    /// summary with used/skipped/failed counts, and stays silent on an empty
    /// window (idle desktop must never add a line).
    #[test]
    fn present_import_summarizes_window_and_stays_silent_when_idle() {
        let _g = test_lock();
        present_import::reset();
        let w = 1000u64;
        let base = 5000u64;

        assert_eq!(present_import::test_maybe_line(base), None, "seed no emit");
        present_import::test_note(true, false); // used
        present_import::test_note(true, false); // used
        present_import::test_note(false, true); // fail
        present_import::test_note(false, false); // skip
        assert_eq!(
            present_import::test_maybe_line(base + w - 1),
            None,
            "before close no emit"
        );
        let line = present_import::test_maybe_line(base + w).expect("boundary emits");
        assert!(line.contains("used=2"), "line: {line}");
        assert!(line.contains("failed=1"), "line: {line}");
        assert!(line.contains("skipped=1"), "line: {line}");
        assert!(line.contains(&format!("window_ms={w}")), "line: {line}");
        assert_eq!(
            present_import::test_maybe_line(base + w + w),
            None,
            "empty window emits nothing"
        );
        present_import::reset();
    }

    /// `rgb_px` = RGB-nonzero **pixel** count; byte `nz` ≈ 4× for alpha-full frames.
    fn sample(mid: u32, rgb_px: usize, w: u32, h: u32) -> PresentCaptureSample {
        PresentCaptureSample {
            mapping_id: mid,
            generation: 1,
            width: w,
            height: h,
            nz: rgb_px.saturating_mul(4),
            max_byte: 255,
            rgb_nz: rgb_px,
            max_rgb: 255,
            from_last_store: true,
            edge_energy: 10,
        }
    }

    fn sample_edge(mid: u32, rgb_px: usize, w: u32, h: u32, edge: u32) -> PresentCaptureSample {
        PresentCaptureSample {
            mapping_id: mid,
            generation: 1,
            width: w,
            height: h,
            nz: rgb_px.saturating_mul(4),
            max_byte: 255,
            rgb_nz: rgb_px,
            max_rgb: 255,
            from_last_store: true,
            edge_energy: edge,
        }
    }

    /// `note_stale_online_pending` counts every post-converge stale-ONLINE IRQ but
    /// logs its always-on line only once per boot (VBL is ~60 Hz — a per-call line
    /// would flood). The count keeps climbing after the latched line.
    #[test]
    fn stale_online_pending_counts_all_logs_once() {
        let _g = test_lock();
        reset_for_test();
        use crate::model::DISPLAY_ONLINE_EVENT_MASK;
        note_stale_online_pending("vbl", DISPLAY_ONLINE_EVENT_MASK);
        note_stale_online_pending("vbl", DISPLAY_ONLINE_EVENT_MASK);
        note_stale_online_pending("present", DISPLAY_ONLINE_EVENT_MASK);
        assert_eq!(
            counters().stale_online_pending,
            3,
            "every stale-online IRQ must be counted"
        );
        let log = std::fs::read_to_string(observe::fail_log_path()).expect("fail log");
        assert!(
            log.contains("stale_online_pending src=vbl pending=0x4 count=1"),
            "first stale-online must log the always-on line"
        );
    }

    /// The desktop converges, then the guest reverts to a sustained overlay
    /// (rgb_frac ≈0.13 — not sparse, not dense): `post_converge_regress` fires
    /// exactly at the run threshold, a dense present clears the run, and it does
    /// not re-fire until the run rebuilds.
    #[test]
    fn post_converge_regress_fires_on_sustained_overlay_after_desktop() {
        let _g = test_lock();
        reset_for_test();
        let (w, h) = (1920u32, 1080u32);
        let pixels = (w as usize) * (h as usize);
        let dense = pixels * 60 / 100; // 0.60 → converged desktop
        let overlay = pixels * 13 / 100; // 0.13 → logo overlay (partial: not sparse/dense)

        note_capture_ok(sample(1, dense, w, h));
        assert!(counters().converged);
        assert_eq!(counters().post_converge_regress, 0);

        for _ in 0..(POST_CONVERGE_REGRESS_RUN - 1) {
            note_capture_ok(sample(4, overlay, w, h));
        }
        assert_eq!(
            counters().post_converge_regress,
            0,
            "below threshold stays quiet"
        );
        note_capture_ok(sample(4, overlay, w, h)); // run reaches threshold
        assert_eq!(counters().post_converge_regress, 1);
        // The fired line carries the structural discriminators: a stuck overlay
        // re-presents the same retained frame, so `generation` never advances
        // (`sample()` pins gen=1) → `gen_adv=0`. That is the real-regression
        // signature (vs. a live dark app, next test).
        let log = std::fs::read_to_string(observe::fail_log_path()).expect("fail log");
        assert!(
            log.contains("post_converge_regress")
                && log
                    .lines()
                    .any(|l| l.contains("post_converge_regress") && l.contains("gen_adv=0")),
            "stuck-overlay regress line must report gen_adv=0"
        );

        // A dense present clears the run; a lone overlay after does not re-fire.
        note_capture_ok(sample(1, dense, w, h));
        note_capture_ok(sample(4, overlay, w, h));
        assert_eq!(counters().post_converge_regress, 1);
    }

    /// A sustained legitimately-dark FULLSCREEN app (a dark video/game/site) sits
    /// just below the 0.50 dense threshold, so it trips the same non-dense run —
    /// but it is live, structured content, not a broken desktop. The fired line
    /// must distinguish it: `gen_adv=1` (a FRESH full-frame Store lands every
    /// present, generation advancing) and `edge` well above zero (real structure),
    /// vs the stuck overlay's `gen_adv=0`. This is the false-positive class
    /// observed live under testufo fullscreen (rgb_frac≈0.49, gen advancing);
    /// the discriminator lets an audit tell it from a real regression without
    /// gating the measure-only proxy on content.
    #[test]
    fn post_converge_regress_line_flags_live_dark_fullscreen_as_fresh() {
        let _g = test_lock();
        reset_for_test();
        let (w, h) = (1920u32, 1080u32);
        let pixels = (w as usize) * (h as usize);
        let dense = pixels * 60 / 100; // 0.60 → converged desktop
        let dark = pixels * 49 / 100; // 0.49 → legitimately-dark fullscreen app

        note_capture_ok(sample(1, dense, w, h));
        assert!(counters().converged);

        // A live app: same fullscreen member, moderate structure, but a fresh
        // Store (advancing generation) each present.
        for g in 0..POST_CONVERGE_REGRESS_RUN {
            let mut s = sample_edge(1, dark, w, h, 4000);
            s.generation = 100 + g as u32; // fresh full-frame Store every present
            note_capture_ok(s);
        }
        assert_eq!(
            counters().post_converge_regress,
            1,
            "run still trips (measure-only)"
        );
        let log = std::fs::read_to_string(observe::fail_log_path()).expect("fail log");
        assert!(
            log.lines().any(|l| l.contains("post_converge_regress")
                && l.contains("gen_adv=1")
                && l.contains("edge=4000")),
            "live-dark-fullscreen regress line must report gen_adv=1 and nonzero edge"
        );
    }

    /// The pre-convergence boot strobe (overlay before the desktop ever
    /// composites) must NOT fire the post-converge regression.
    #[test]
    fn post_converge_regress_quiet_before_convergence() {
        let _g = test_lock();
        reset_for_test();
        let (w, h) = (1920u32, 1080u32);
        let overlay = (w as usize) * (h as usize) * 13 / 100;
        for _ in 0..(POST_CONVERGE_REGRESS_RUN * 2) {
            note_capture_ok(sample(4, overlay, w, h));
        }
        assert!(!counters().converged);
        assert_eq!(counters().post_converge_regress, 0);
    }

    #[test]
    fn nz_swing_fires_on_dual_mid_incomplete() {
        let _g = test_lock();
        reset_for_test();
        // Full desktop-ish RGB occupancy vs sparse logo-ish on mid switch.
        let full = (1440u64 * 1080 * 40 / 100) as usize; // 40% of pixels
        let logo = (1440u64 * 1080 * 5 / 100) as usize; // 5% of pixels
        note_capture_ok(sample(3, full, 1440, 1080));
        note_capture_ok(sample(4, logo, 1440, 1080));
        let c = counters();
        assert_eq!(c.mid_switches, 1);
        assert_eq!(
            c.nz_swings, 1,
            "logo vs full dual-mid must count as nz_swing"
        );
        assert!(c.sparse >= 1, "5% RGB occupancy is sparse_present");
    }

    #[test]
    fn present_converge_latches_once_after_sparse_boot_strobe() {
        let _g = test_lock();
        reset_for_test();
        // Dual-mid boot strobe: chrome-only sparse frames (rgb_frac ≈0.003)
        // alternating before the wallpaper lands, then the first dense desktop.
        let chrome = (1920u64 * 1080 * 3 / 1000) as usize; // 0.3% RGB occupancy
        let dense = (1920u64 * 1080) as usize; // full wallpaper (100%)
        for _ in 0..4 {
            note_capture_ok(sample(1, chrome, 1920, 1080));
            note_capture_ok(sample(5, chrome, 1920, 1080));
        }
        let before = counters();
        assert!(
            !before.converged,
            "still strobing chrome-only, not converged"
        );
        assert_eq!(before.sparse, 8, "all eight chrome frames are sparse");

        // First full-desktop present: converge latches.
        note_capture_ok(sample(1, dense, 1920, 1080));
        let at = counters();
        assert!(at.converged, "dense wallpaper present converges the boot");

        // A later sparse relapse must NOT re-arm (latched once per boot).
        note_capture_ok(sample(5, chrome, 1920, 1080));
        note_capture_ok(sample(1, dense, 1920, 1080));
        let after = counters();
        assert!(after.converged, "convergence stays latched across relapse");
        // sparse keeps counting the relapse, but the convergence latch is stable.
        assert!(after.sparse >= at.sparse, "relapse still counted as sparse");
    }

    #[test]
    fn mid_switch_without_swing_is_not_nz_swing() {
        let _g = test_lock();
        reset_for_test();
        let a = (1440u64 * 1080 * 50 / 100) as usize;
        let b = (1440u64 * 1080 * 48 / 100) as usize;
        note_capture_ok(sample(3, a, 1440, 1080));
        note_capture_ok(sample(4, b, 1440, 1080));
        let c = counters();
        assert_eq!(c.mid_switches, 1);
        assert_eq!(
            c.nz_swings, 0,
            "similar nz dual-buffer must not false-positive"
        );
        assert_eq!(
            c.structure_swings, 0,
            "equal default edge_energy must not structure_swing"
        );
    }

    /// qemu-shim: dual-mid with similar nz but divergent edge energy (live
    /// Safari shattered vs full-chrome) must fire structure_swing.
    #[test]
    fn structure_swing_fires_on_equal_nz_boxy_mid() {
        let _g = test_lock();
        reset_for_test();
        let nz = (1440u64 * 1080 * 4 * 50 / 100) as usize;
        note_capture_ok(sample_edge(3, nz, 1440, 1080, 8)); // smooth
        note_capture_ok(sample_edge(4, nz, 1440, 1080, 40)); // shattered
        let c = counters();
        assert_eq!(c.mid_switches, 1);
        assert_eq!(c.nz_swings, 0, "equal nz must not nz_swing");
        assert_eq!(
            c.structure_swings, 1,
            "edge 8 vs 40 must count structure_swing"
        );
    }

    #[test]
    fn geom_mismatch_counted() {
        let _g = test_lock();
        reset_for_test();
        note_capture_ok(sample(3, 1_000_000, 1920, 1080));
        note_capture_ok(sample(3, 1_000_000, 1440, 1080));
        assert_eq!(counters().geom_mismatch, 1);
    }

    #[test]
    fn capture_fail_counted() {
        let _g = test_lock();
        reset_for_test();
        note_capture_fail(5, 1440, 1080, 2);
        assert_eq!(counters().capture_fail, 1);
    }

    /// qemu-shim: display Store peak tracks dual-mid poison (nz drop vs mid peak).
    #[test]
    fn display_store_nz_poison_on_drop_from_peak() {
        let _g = test_lock();
        reset_display_store_peaks_for_test();
        let (p1, poison1) = note_display_store_nz(3, 6_000_000);
        assert_eq!(p1, 6_000_000);
        assert!(!poison1, "first Store establishes peak, not poison");
        let (p2, poison2) = note_display_store_nz(3, 5_500_000);
        assert_eq!(p2, 6_000_000);
        assert!(!poison2, "small drop stays under 30% threshold");
        let (p3, poison3) = note_display_store_nz(3, 3_000_000);
        assert_eq!(p3, 6_000_000);
        assert!(poison3, "drop below 70% of peak is poison");
        let (p4, poison4) = note_display_store_nz(4, 2_000_000);
        assert_eq!(p4, 2_000_000);
        assert!(!poison4, "other mid has independent peak");
    }

    #[test]
    fn selected_peer_divergence_fires_once_per_sparse_episode() {
        let _g = test_lock();
        reset_for_test();
        let total = 1920 * 1080;

        assert!(!note_selected_peer_divergence(1, 2, 1920, 1080, total));
        assert!(note_selected_peer_divergence(2, 2, 1920, 1080, 300_000));
        assert_eq!(counters().selected_peer_divergence, 1);
        assert!(!note_selected_peer_divergence(2, 2, 1920, 1080, 400_000));
        assert!(!note_selected_peer_divergence(11, 2, 1920, 1080, 6_018));
        assert_eq!(counters().selected_peer_divergence, 1);

        assert!(!note_selected_peer_divergence(2, 2, 1920, 1080, total));
        assert!(note_selected_peer_divergence(2, 2, 1920, 1080, 300_000));
        assert_eq!(counters().selected_peer_divergence, 2);
    }

    #[test]
    fn selected_peer_divergence_forgets_recycled_mapping_samples() {
        let _g = test_lock();
        reset_for_test();
        let total = 1920 * 1080;

        assert!(!note_selected_peer_divergence(1, 2, 1920, 1080, total));
        assert!(note_selected_peer_divergence(2, 2, 1920, 1080, 300_000));
        assert_eq!(counters().selected_peer_divergence, 1);

        forget_display_store_sample(1);
        assert!(
            !note_selected_peer_divergence(2, 2, 1920, 1080, 300_000),
            "forgotten dense peer must not keep the episode alive"
        );
        assert_eq!(counters().selected_peer_divergence, 1);

        assert!(
            note_selected_peer_divergence(3, 2, 1920, 1080, total),
            "a new dense peer re-arms a new diagnosed episode"
        );
        assert_eq!(counters().selected_peer_divergence, 2);

        forget_display_store_sample(2);
        assert!(
            !note_selected_peer_divergence(3, 2, 1920, 1080, total),
            "forgotten selected mapping leaves no selected sample"
        );
        assert!(note_selected_peer_divergence(2, 2, 1920, 1080, 300_000));
        assert_eq!(counters().selected_peer_divergence, 3);
    }

    /// The sample side and the render side of the MRT-mask class had one name
    /// between them.
    ///
    /// Both `note_secondary_mrt_drop` and `note_mrt_mask_bind_miss` were called
    /// with the bare string `"geometry_mismatch"`, so a `grep reason=` for it
    /// could not tell a dropped multi-RT render from a mask sample that failed to
    /// bind — two different checks in two different proxies. Crate-wide
    /// uniqueness is the gate; this names the pair the gate is protecting.
    #[test]
    fn the_render_and_sample_sides_of_the_mask_class_have_different_names() {
        use crate::observe::Decline as _;
        assert_ne!(
            MrtDrop::GeometryMismatch.slug(),
            MaskBindMiss::GeometryMismatch.slug()
        );
        assert!(MrtDrop::GeometryMismatch.slug().starts_with("mrt_drop_"));
        assert!(MaskBindMiss::GeometryMismatch
            .slug()
            .starts_with("mask_bind_"));
        // The dedup codes must stay disjoint too: both proxies share one set, so
        // a collision there would silence one of them per geometry.
        assert_ne!(
            MrtDrop::GeometryMismatch.code(),
            MaskBindMiss::GeometryMismatch.code()
        );
        let mrt: Vec<u8> = [
            MrtDrop::NonContiguousSlot,
            MrtDrop::GeometryMismatch,
            MrtDrop::UnknownFormat,
            MrtDrop::NoIdentity,
            MrtDrop::AliasesPrimary,
        ]
        .iter()
        .map(|r| r.code())
        .collect();
        let mask: Vec<u8> = [
            MaskBindMiss::GeometryMismatch,
            MaskBindMiss::ResidentNotReady,
        ]
        .iter()
        .map(|r| r.code())
        .collect();
        assert!(
            mrt.iter().all(|c| !mask.contains(c)),
            "dedup codes overlap: {mrt:?} vs {mask:?}"
        );
    }

    /// secondary_mrt_drop: a silently-degraded multi-RT draw fires an always-on
    /// line naming the reason, deduped on (reason, geometry) so a per-frame drop
    /// reports once per distinct combo (never floods). Distinct reasons and
    /// distinct geometries are independent episodes.
    #[test]
    fn secondary_mrt_drop_dedups_per_reason_and_geometry() {
        let _g = test_lock();
        reset_for_test();
        let path = observe::fail_log_path();
        let count = |needle: &str| {
            std::fs::read_to_string(path)
                .unwrap_or_default()
                .matches(needle)
                .count()
        };
        let l0 = count("secondary_mrt_drop reason=mrt_drop_geometry_mismatch geom=214x54");

        // First drop at a geometry fires once.
        note_secondary_mrt_drop(MrtDrop::GeometryMismatch, 214, 54);
        assert_eq!(
            count("secondary_mrt_drop reason=mrt_drop_geometry_mismatch geom=214x54"),
            l0 + 1
        );
        // Same reason+geometry → deduped (no per-frame re-fire).
        note_secondary_mrt_drop(MrtDrop::GeometryMismatch, 214, 54);
        note_secondary_mrt_drop(MrtDrop::GeometryMismatch, 214, 54);
        assert_eq!(
            count("secondary_mrt_drop reason=mrt_drop_geometry_mismatch geom=214x54"),
            l0 + 1,
            "same reason+geometry must dedup"
        );
        // A different reason at the same geometry is its own episode.
        let n_unknown = count("secondary_mrt_drop reason=mrt_drop_unknown_format geom=214x54");
        note_secondary_mrt_drop(MrtDrop::UnknownFormat, 214, 54);
        assert_eq!(
            count("secondary_mrt_drop reason=mrt_drop_unknown_format geom=214x54"),
            n_unknown + 1
        );
        // A different geometry, same reason, is its own episode.
        let n_other = count("secondary_mrt_drop reason=mrt_drop_geometry_mismatch geom=100x40");
        note_secondary_mrt_drop(MrtDrop::GeometryMismatch, 100, 40);
        assert_eq!(
            count("secondary_mrt_drop reason=mrt_drop_geometry_mismatch geom=100x40"),
            n_other + 1
        );
    }

    /// Regression guard for [`edge_energy_bgra`], the horizontal-luma-gradient
    /// proxy behind `PresentCaptureSample::edge_energy` and the
    /// `structure_swing` event. Its contract is load-bearing for that event and
    /// for the GPU twin in `stats_reduce`, which must agree with it: a uniform
    /// frame must read **zero** energy so a smooth desktop never registers as
    /// structure, structured content must read **nonzero** and grow with
    /// contrast so a shattered/chrome-rich mid separates from a smooth one, and
    /// degenerate/short input must read zero rather than panic or emit garbage.
    #[test]
    fn edge_energy_flat_is_zero_and_grows_with_contrast() {
        // Tight BGRA8 frame filled with a single column pattern (repeats each
        // row) so the subsampled horizontal scan sees the same gradient on
        // every sampled row.
        fn frame(width: u32, height: u32, col: impl Fn(u32) -> u8) -> Vec<u8> {
            let mut v = vec![0u8; (width * height * 4) as usize];
            for y in 0..height {
                for x in 0..width {
                    let o = ((y * width + x) * 4) as usize;
                    let g = col(x);
                    v[o] = g; // B
                    v[o + 1] = g; // G
                    v[o + 2] = g; // R
                    v[o + 3] = 0xFF; // A
                }
            }
            v
        }

        let (w, h) = (16u32, 16u32);

        // Degenerate geometry and short buffers read zero, never panic.
        assert_eq!(edge_energy_bgra(&frame(7, 7, |_| 200), 7, 7), 0);
        assert_eq!(edge_energy_bgra(&[0u8; 16], w, h), 0, "short buffer -> 0");

        // A perfectly uniform frame has no horizontal gradient -> zero energy.
        let flat = frame(w, h, |_| 128);
        assert_eq!(
            edge_energy_bgra(&flat, w, h),
            0,
            "a clean uniform desktop must read zero edge energy",
        );

        // Alternating dark/bright columns produce energy that scales with the
        // per-pixel contrast (the residue gate needs monotonicity to threshold).
        let low = edge_energy_bgra(&frame(w, h, |x| if x % 2 == 0 { 100 } else { 130 }), w, h);
        let high = edge_energy_bgra(&frame(w, h, |x| if x % 2 == 0 { 0 } else { 255 }), w, h);
        assert!(low > 0, "structured content must register nonzero energy");
        assert!(
            high > low,
            "higher per-pixel contrast must read higher energy ({high} !> {low})",
        );
    }
}

/// Which scatter the guest-store path executed.
///
/// The GPU-direct scatter writes the resident into imported guest pages with no
/// frame bytes on the CPU; the fallback pays two full-frame copies. Both produce
/// identical guest content, so nothing visual names a permanent regression to
/// the fallback — this census does. `gpu_frac` below 1.0 means some stores could
/// not import their runs (`unresolved`) or failed to submit (`submit_fail`).
pub mod store_scatter {
    use crate::observe;
    use std::sync::atomic::{AtomicU64, Ordering};

    static GPU: AtomicU64 = AtomicU64::new(0);
    static CPU: AtomicU64 = AtomicU64::new(0);
    static WINDOW_START_MS: AtomicU64 = AtomicU64::new(0);
    /// Cumulative engine-side totals, snapshotted to report deltas per window.
    static LAST_UNRESOLVED: AtomicU64 = AtomicU64::new(0);
    static LAST_SUBMIT_FAIL: AtomicU64 = AtomicU64::new(0);

    const WINDOW_MS: u64 = 1000;

    pub fn note(gpu: bool, _gpu_stores: u64, unresolved: u64, submit_fail: u64) {
        if gpu {
            GPU.fetch_add(1, Ordering::Relaxed);
        } else {
            CPU.fetch_add(1, Ordering::Relaxed);
        }
        if let Some(line) = maybe_line_at(observe::elapsed_ms() as u64, unresolved, submit_fail) {
            observe::off(line);
        }
    }

    fn maybe_line_at(now: u64, unresolved: u64, submit_fail: u64) -> Option<String> {
        let start = WINDOW_START_MS.load(Ordering::Relaxed);
        if start == 0 {
            let _ = WINDOW_START_MS.compare_exchange(0, now, Ordering::Relaxed, Ordering::Relaxed);
            LAST_UNRESOLVED.store(unresolved, Ordering::Relaxed);
            LAST_SUBMIT_FAIL.store(submit_fail, Ordering::Relaxed);
            return None;
        }
        let dt = now.saturating_sub(start);
        if dt < WINDOW_MS {
            return None;
        }
        if WINDOW_START_MS
            .compare_exchange(start, now, Ordering::Relaxed, Ordering::Relaxed)
            .is_err()
        {
            return None;
        }
        let gpu = GPU.swap(0, Ordering::Relaxed);
        let cpu = CPU.swap(0, Ordering::Relaxed);
        if gpu.saturating_add(cpu) == 0 {
            return None;
        }
        let d_unres =
            unresolved.saturating_sub(LAST_UNRESOLVED.swap(unresolved, Ordering::Relaxed));
        let d_submit =
            submit_fail.saturating_sub(LAST_SUBMIT_FAIL.swap(submit_fail, Ordering::Relaxed));
        Some(format_line(dt, gpu, cpu, d_unres, d_submit))
    }

    fn format_line(dt: u64, gpu: u64, cpu: u64, unresolved: u64, submit_fail: u64) -> String {
        let total = gpu.saturating_add(cpu);
        let frac = gpu as f64 / total.max(1) as f64;
        // Bytes of frame the CPU path copied that the GPU path would not have:
        // readback copy-out + scatter, i.e. two per fallback store.
        format!(
            "store_scatter window_ms={dt} gpu={gpu} cpu={cpu} gpu_frac={frac:.3} \
             unresolved={unresolved} submit_fail={submit_fail}"
        )
    }

    #[cfg(test)]
    pub(crate) fn reset() {
        for a in [
            &GPU,
            &CPU,
            &WINDOW_START_MS,
            &LAST_UNRESOLVED,
            &LAST_SUBMIT_FAIL,
        ] {
            a.store(0, Ordering::Relaxed);
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// Bump the atomics directly rather than through `note()`: `note()`
        /// stamps the window from the real clock, which makes the assertion
        /// depend on wall time.
        #[test]
        fn window_reports_scatter_split_and_stays_silent_when_idle() {
            reset();
            assert_eq!(maybe_line_at(1, 0, 0), None, "first call opens the window");
            GPU.fetch_add(48, Ordering::Relaxed);
            CPU.fetch_add(2, Ordering::Relaxed);
            let line = maybe_line_at(1 + WINDOW_MS, 2, 0).expect("line");
            assert!(line.contains("gpu=48"), "{line}");
            assert!(line.contains("cpu=2"), "{line}");
            assert!(line.contains("gpu_frac=0.960"), "{line}");
            assert!(line.contains("unresolved=2"), "{line}");

            // Idle window: no stores, no line (the log must not tick on idle).
            assert_eq!(maybe_line_at(1 + 3 * WINDOW_MS, 2, 0), None);
        }

        /// The engine's counters are cumulative; the line reports per-window
        /// deltas so a single early failure does not brand every later window.
        #[test]
        fn cumulative_counters_report_as_deltas() {
            reset();
            assert_eq!(maybe_line_at(1, 5, 1), None);
            GPU.fetch_add(1, Ordering::Relaxed);
            let line = maybe_line_at(1 + WINDOW_MS, 5, 1).expect("line");
            assert!(line.contains("unresolved=0"), "no new failures: {line}");
            assert!(line.contains("submit_fail=0"), "{line}");
        }
    }
}

/// Cap-pressure census: the always-on signal for the "a cap blew and render fell
/// off a cliff" class (registry / sampled-cache / graveyard evictions). The
/// caps (`REGISTRY_CAP`, `SAMPLED_CACHE_BYTE_CAP`, …) bound host VRAM, but when a
/// workload's live working set exceeds one, the LRU sweep evicts a target that
/// is needed again next frame — a re-allocate / re-upload storm that turns
/// 120 fps into single digits. The engine already counts `target_evicts` /
/// `sampled_reuploads`; nothing surfaced them live, so a cap-blow was invisible
/// until someone stared at a slow screen. This windowed line names it: per-second
/// eviction + reupload deltas alongside peak occupancy vs cap, so the exact cap
/// that blew (and how close the others are) is legible without a profiler.
///
/// Fires ~1/s ONLY when there was real pressure in the window (an eviction or a
/// reupload). A healthy boot whose working set fits under every cap stays
/// silent — verify zero lines on an idle desktop before trusting a spike.
pub mod cap_pressure {
    use crate::observe;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// One per-present occupancy + cumulative-counter reading. Plain primitives
    /// so this census stays backend-feature-agnostic (the caller unpacks the
    /// engine snapshot).
    pub struct Sample {
        pub registry_len: usize,
        pub registry_cap: usize,
        /// Resident targets held against LRU eviction by a live deferred write
        /// window. When this nears `registry_len`, the registry has soft-
        /// exceeded its slot cap and cannot shrink — the cliff root.
        pub registry_pinned: usize,
        /// Live `render_deferred_flush` windows (one pins one resident). The
        /// state-side count of what drives `registry_pinned` upward.
        pub render_windows: usize,
        pub sampled_len: usize,
        pub sampled_cap: usize,
        pub sampled_bytes: usize,
        pub sampled_byte_cap: usize,
        pub graveyard_len: usize,
        pub target_evicts: u64,
        pub gen_mismatch: u64,
        pub sampled_reuploads: u64,
        pub sampled_reupload_bytes: u64,
    }

    static WINDOW_START_MS: AtomicU64 = AtomicU64::new(0);
    static LAST_EVICTS: AtomicU64 = AtomicU64::new(0);
    static LAST_GEN_MISMATCH: AtomicU64 = AtomicU64::new(0);
    static LAST_REUPLOADS: AtomicU64 = AtomicU64::new(0);
    static LAST_REUPLOAD_BYTES: AtomicU64 = AtomicU64::new(0);
    /// Peak occupancy seen in the window, so a spike between two emits is not
    /// missed by sampling only at emit time.
    static PEAK_REG: AtomicU64 = AtomicU64::new(0);
    static PEAK_PINNED: AtomicU64 = AtomicU64::new(0);
    static PEAK_RENDER_WIN: AtomicU64 = AtomicU64::new(0);
    static PEAK_SAMPLED: AtomicU64 = AtomicU64::new(0);
    static PEAK_SAMPLED_BYTES: AtomicU64 = AtomicU64::new(0);
    static PEAK_GRAVEYARD: AtomicU64 = AtomicU64::new(0);

    const WINDOW_MS: u64 = 1000;

    pub fn note(s: Sample) {
        if let Some(line) = maybe_line_at(observe::elapsed_ms() as u64, &s) {
            observe::off(line);
        }
    }

    fn maybe_line_at(now: u64, s: &Sample) -> Option<String> {
        // Fold this reading into the window's running peaks first, so a spike
        // between two emits is captured even though the emit-time sample missed
        // it. Every `note()` call lands here (most return early below).
        PEAK_REG.fetch_max(s.registry_len as u64, Ordering::Relaxed);
        PEAK_PINNED.fetch_max(s.registry_pinned as u64, Ordering::Relaxed);
        PEAK_RENDER_WIN.fetch_max(s.render_windows as u64, Ordering::Relaxed);
        PEAK_SAMPLED.fetch_max(s.sampled_len as u64, Ordering::Relaxed);
        PEAK_SAMPLED_BYTES.fetch_max(s.sampled_bytes as u64, Ordering::Relaxed);
        PEAK_GRAVEYARD.fetch_max(s.graveyard_len as u64, Ordering::Relaxed);
        let start = WINDOW_START_MS.load(Ordering::Relaxed);
        if start == 0 {
            let _ = WINDOW_START_MS.compare_exchange(0, now, Ordering::Relaxed, Ordering::Relaxed);
            LAST_EVICTS.store(s.target_evicts, Ordering::Relaxed);
            LAST_GEN_MISMATCH.store(s.gen_mismatch, Ordering::Relaxed);
            LAST_REUPLOADS.store(s.sampled_reuploads, Ordering::Relaxed);
            LAST_REUPLOAD_BYTES.store(s.sampled_reupload_bytes, Ordering::Relaxed);
            return None;
        }
        let dt = now.saturating_sub(start);
        if dt < WINDOW_MS {
            return None;
        }
        if WINDOW_START_MS
            .compare_exchange(start, now, Ordering::Relaxed, Ordering::Relaxed)
            .is_err()
        {
            return None;
        }
        let d_evicts = s
            .target_evicts
            .saturating_sub(LAST_EVICTS.swap(s.target_evicts, Ordering::Relaxed));
        let d_gen = s
            .gen_mismatch
            .saturating_sub(LAST_GEN_MISMATCH.swap(s.gen_mismatch, Ordering::Relaxed));
        let d_reup = s
            .sampled_reuploads
            .saturating_sub(LAST_REUPLOADS.swap(s.sampled_reuploads, Ordering::Relaxed));
        let d_reup_bytes = s
            .sampled_reupload_bytes
            .saturating_sub(LAST_REUPLOAD_BYTES.swap(s.sampled_reupload_bytes, Ordering::Relaxed));
        let peak_reg = PEAK_REG.swap(0, Ordering::Relaxed);
        let peak_pinned = PEAK_PINNED.swap(0, Ordering::Relaxed);
        let peak_render_win = PEAK_RENDER_WIN.swap(0, Ordering::Relaxed);
        let peak_sampled = PEAK_SAMPLED.swap(0, Ordering::Relaxed);
        let peak_sampled_bytes = PEAK_SAMPLED_BYTES.swap(0, Ordering::Relaxed);
        let peak_graveyard = PEAK_GRAVEYARD.swap(0, Ordering::Relaxed);
        // Only speak on genuine pressure — an eviction or a reupload happened.
        // Occupancy alone (near a cap but not evicting) is not yet a cliff.
        if d_evicts == 0 && d_reup == 0 {
            return None;
        }
        Some(format_line(
            dt,
            d_evicts,
            d_gen,
            d_reup,
            d_reup_bytes,
            peak_reg,
            s.registry_cap as u64,
            peak_pinned,
            peak_render_win,
            peak_sampled,
            s.sampled_cap as u64,
            peak_sampled_bytes,
            s.sampled_byte_cap as u64,
            peak_graveyard,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn format_line(
        dt: u64,
        evicts: u64,
        gen_mismatch: u64,
        reuploads: u64,
        reupload_bytes: u64,
        peak_reg: u64,
        reg_cap: u64,
        peak_pinned: u64,
        peak_render_win: u64,
        peak_sampled: u64,
        sampled_cap: u64,
        peak_sampled_bytes: u64,
        sampled_byte_cap: u64,
        peak_graveyard: u64,
    ) -> String {
        let reup_mb = reupload_bytes as f64 / (1024.0 * 1024.0);
        let sampled_mb = peak_sampled_bytes as f64 / (1024.0 * 1024.0);
        let sampled_cap_mb = sampled_byte_cap as f64 / (1024.0 * 1024.0);
        format!(
            "cap_pressure window_ms={dt} evicts={evicts} gen_mismatch={gen_mismatch} \
             reuploads={reuploads} reup_mb={reup_mb:.1} reg={peak_reg}/{reg_cap} \
             pinned={peak_pinned} render_win={peak_render_win} \
             sampled={peak_sampled}/{sampled_cap} sampled_mb={sampled_mb:.1}/{sampled_cap_mb:.0} \
             graveyard={peak_graveyard}"
        )
    }

    #[cfg(test)]
    pub(crate) fn reset() {
        for a in [
            &WINDOW_START_MS,
            &LAST_EVICTS,
            &LAST_GEN_MISMATCH,
            &LAST_REUPLOADS,
            &LAST_REUPLOAD_BYTES,
            &PEAK_REG,
            &PEAK_PINNED,
            &PEAK_RENDER_WIN,
            &PEAK_SAMPLED,
            &PEAK_SAMPLED_BYTES,
            &PEAK_GRAVEYARD,
        ] {
            a.store(0, Ordering::Relaxed);
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn sample(evicts: u64, reuploads: u64, reg: usize, sampled_bytes: usize) -> Sample {
            Sample {
                registry_len: reg,
                registry_cap: 64,
                registry_pinned: 0,
                render_windows: 0,
                sampled_len: 10,
                sampled_cap: 64,
                sampled_bytes,
                sampled_byte_cap: 128 * 1024 * 1024,
                graveyard_len: 0,
                target_evicts: evicts,
                gen_mismatch: 0,
                sampled_reuploads: reuploads,
                sampled_reupload_bytes: reuploads * 1024 * 1024,
            }
        }

        #[test]
        fn fires_on_eviction_pressure_reports_peak_occupancy() {
            reset();
            // First call opens the window and snapshots the cumulative baseline.
            assert_eq!(maybe_line_at(1, &sample(100, 5, 40, 0)), None);
            // A later window with 20 more evictions + 3 more reuploads speaks.
            let line = maybe_line_at(1 + WINDOW_MS, &sample(120, 8, 64, 100 * 1024 * 1024))
                .expect("pressure line");
            assert!(line.contains("evicts=20"), "{line}");
            assert!(line.contains("reuploads=3"), "{line}");
            assert!(line.contains("reg=64/64"), "{line}");
            assert!(line.contains("sampled_mb=100.0/128"), "{line}");
        }

        #[test]
        fn stays_silent_without_evictions_or_reuploads() {
            reset();
            assert_eq!(maybe_line_at(1, &sample(0, 0, 60, 0)), None);
            // Occupancy climbed to 63/64 but nothing evicted — not yet a cliff,
            // so no line (occupancy alone must not tick the log).
            assert_eq!(
                maybe_line_at(1 + WINDOW_MS, &sample(0, 0, 63, 120 * 1024 * 1024)),
                None
            );
        }

        #[test]
        fn cumulative_counters_report_as_per_window_deltas() {
            reset();
            assert_eq!(maybe_line_at(1, &sample(1000, 500, 32, 0)), None);
            let line = maybe_line_at(1 + WINDOW_MS, &sample(1001, 500, 32, 0)).expect("line");
            // One new eviction, zero new reuploads since the baseline.
            assert!(line.contains("evicts=1"), "{line}");
            assert!(line.contains("reuploads=0"), "{line}");
        }
    }
}

/// Always-on VRAM-footprint census. Unlike `cap_pressure` (gated on eviction/
/// reupload pressure, silent at idle) this emits every window regardless, so the
/// physical slab footprint is visible even on a static page — the direct signal
/// for the "least VRAM possible" goal. Driven from the poll heartbeat so it keeps
/// reporting when the guest stops publishing.
///
/// Reading it: `resident` is the VkDeviceMemory the slab holds from the driver;
/// `free` is how much of that is unbound. `resident` staying high after a burst
/// drains (registry back to baseline) while `free` is large and `frees` is flat
/// is the **fragmentation** signature — scattered survivors pin every block so
/// none can be returned. `tfree`/`sfree` are the recycle-pool image counts (each
/// pins a slab sub-allocation). Emitted only when the footprint or a pool count
/// changed since the last window, so a fully-idle steady state is quiet.
pub mod vram {
    use crate::observe;
    use std::sync::atomic::{AtomicU64, Ordering};

    pub struct Sample {
        pub slab_blocks: u64,
        pub slab_empty_blocks: u64,
        pub slab_resident_bytes: u64,
        pub slab_free_bytes: u64,
        pub slab_live_subs: u64,
        pub block_allocs: u64,
        pub block_frees: u64,
        /// Free bytes of the most-empty shared block (fragmentation stranding).
        pub max_block_free_bytes: u64,
        /// Live sub-allocations bucketed by size (see slab `SIZE_BUCKET_EDGES`).
        pub size_buckets: [u64; 8],
        pub registry_len: usize,
        pub registry_pinned: usize,
        /// Live descriptor-arena pool blocks (`desc_blocks`); 1 = never grew.
        pub desc_blocks: usize,
        pub sampled_len: usize,
        pub target_free_imgs: usize,
        pub sampled_free_imgs: usize,
        /// Transient compute-storage recycle-pool image count. Each is a standalone
        /// (non-slab) VkDeviceMemory, invisible to the slab `resident`/`live_subs`
        /// fields, so it carries its own emit trigger below.
        pub storage_free_imgs: usize,
        /// Cumulative compute-storage recycle admits / cap-drops (`st_admit` /
        /// `st_drop`). A rising `st_drop` is the "cap is bounding a leak" signal.
        pub storage_recycle_admits: u64,
        pub storage_recycle_cap_drops: u64,
        /// Live compute-storage residents (`st_res`) and pinned subset (`st_pin`) —
        /// non-slab, so a stale-resident idle hold is invisible to the slab fields;
        /// `st_res` carries its own emit trigger below (measure-only for now).
        pub storage_resident: usize,
        pub storage_resident_pinned: usize,
        /// HOST_VISIBLE staging/readback recycle-pool footprint (bytes) — system
        /// RAM on a discrete GPU, shared guest RAM on an iGPU.
        pub staging_free_bytes: u64,
        pub readback_free_bytes: u64,
    }

    static WINDOW_START_MS: AtomicU64 = AtomicU64::new(0);
    static LAST_RESIDENT: AtomicU64 = AtomicU64::new(u64::MAX);
    static LAST_LIVE_SUBS: AtomicU64 = AtomicU64::new(u64::MAX);
    static LAST_HOSTVIS: AtomicU64 = AtomicU64::new(u64::MAX);
    static LAST_STORAGE_FREE: AtomicU64 = AtomicU64::new(u64::MAX);
    static LAST_STORAGE_RES: AtomicU64 = AtomicU64::new(u64::MAX);
    const WINDOW_MS: u64 = 2000;
    /// Also speak when the live-sub-allocation count moved by at least this much
    /// since the last emit, even if `resident_bytes` held flat. That divergence —
    /// the working set draining while the physical footprint does *not* return —
    /// is exactly the fragmentation the census exists to make visible; keying
    /// only on `resident_bytes` hides it (a drain frees images into holes without
    /// emptying a block). 16 is coarse enough to stay quiet at idle.
    const LIVE_SUBS_DELTA: u64 = 16;
    /// Also speak when the HOST_VISIBLE staging+readback pool footprint moved by
    /// at least this many bytes since the last emit, even if the slab footprint
    /// held flat. Those pools are separate `VkDeviceMemory` from the slab, so a
    /// staging drain after a video session (or a staging LEAK) is otherwise
    /// invisible at idle — the slab `resident_bytes`/`live_subs` do not move when
    /// only the HOST_VISIBLE buffers alloc/free. This is the direct signal that
    /// the settled-idle buffer trim actually returns the upload working set.
    /// 8 MiB is one large staging buffer — coarse enough to stay quiet at idle,
    /// fine enough to see the pool drain buffer-by-buffer.
    const HOSTVIS_DELTA: u64 = 8 * 1024 * 1024;
    /// Also speak when the compute-storage recycle-pool image count moved by at
    /// least this much. Those images are standalone (non-slab) VkDeviceMemory, so
    /// — like the HOST_VISIBLE buffers — the slab `resident`/`live_subs` fields do
    /// not move when the pool grows or drains; without this trigger a
    /// compute-storage recycle leak would be invisible on a static page. 4 is one
    /// small burst's worth of storage images — coarse enough to stay quiet at idle.
    const STORAGE_FREE_DELTA: u64 = 4;

    pub fn note(s: Sample) {
        if let Some(line) = maybe_line_at(observe::elapsed_ms() as u64, &s) {
            observe::off(line);
        }
    }

    fn maybe_line_at(now: u64, s: &Sample) -> Option<String> {
        let start = WINDOW_START_MS.load(Ordering::Relaxed);
        if start == 0 {
            let _ = WINDOW_START_MS.compare_exchange(0, now, Ordering::Relaxed, Ordering::Relaxed);
            return None;
        }
        if now.saturating_sub(start) < WINDOW_MS {
            return None;
        }
        if WINDOW_START_MS
            .compare_exchange(start, now, Ordering::Relaxed, Ordering::Relaxed)
            .is_err()
        {
            return None;
        }
        // Quiet a fully-idle steady state, but speak when EITHER the physical
        // footprint moved (a burst grew it / a drain returned a block) OR the
        // working set drained materially under a flat footprint (fragmentation).
        let prev = LAST_RESIDENT.swap(s.slab_resident_bytes, Ordering::Relaxed);
        let prev_subs = LAST_LIVE_SUBS.swap(s.slab_live_subs, Ordering::Relaxed);
        let hostvis = s.staging_free_bytes + s.readback_free_bytes;
        let prev_hostvis = LAST_HOSTVIS.swap(hostvis, Ordering::Relaxed);
        let storage_free = s.storage_free_imgs as u64;
        let prev_storage = LAST_STORAGE_FREE.swap(storage_free, Ordering::Relaxed);
        let storage_res = s.storage_resident as u64;
        let prev_storage_res = LAST_STORAGE_RES.swap(storage_res, Ordering::Relaxed);
        let subs_moved =
            prev_subs == u64::MAX || s.slab_live_subs.abs_diff(prev_subs) >= LIVE_SUBS_DELTA;
        let hostvis_moved =
            prev_hostvis == u64::MAX || hostvis.abs_diff(prev_hostvis) >= HOSTVIS_DELTA;
        let storage_moved = prev_storage == u64::MAX
            || storage_free.abs_diff(prev_storage) >= STORAGE_FREE_DELTA
            || prev_storage_res == u64::MAX
            || storage_res.abs_diff(prev_storage_res) >= STORAGE_FREE_DELTA;
        if prev == s.slab_resident_bytes && !subs_moved && !hostvis_moved && !storage_moved {
            return None;
        }
        let mb = |b: u64| b as f64 / (1024.0 * 1024.0);
        let bkt = s
            .size_buckets
            .iter()
            .map(|n| n.to_string())
            .collect::<Vec<_>>()
            .join(",");
        Some(format!(
            "vram resident_mb={:.0} free_mb={:.0} max_block_free_mb={:.0} blocks={} empty={} \
             live_subs={} block_allocs={} block_frees={} reg={} pinned={} desc_blocks={} \
             sampled={} tfree={} \
             sfree={} stfree={} st_admit={} st_drop={} st_res={} st_pin={} sizes=[{bkt}] \
             staging_mb={:.0} readback_mb={:.0}",
            mb(s.slab_resident_bytes),
            mb(s.slab_free_bytes),
            mb(s.max_block_free_bytes),
            s.slab_blocks,
            s.slab_empty_blocks,
            s.slab_live_subs,
            s.block_allocs,
            s.block_frees,
            s.registry_len,
            s.registry_pinned,
            s.desc_blocks,
            s.sampled_len,
            s.target_free_imgs,
            s.sampled_free_imgs,
            s.storage_free_imgs,
            s.storage_recycle_admits,
            s.storage_recycle_cap_drops,
            s.storage_resident,
            s.storage_resident_pinned,
            mb(s.staging_free_bytes),
            mb(s.readback_free_bytes),
        ))
    }

    #[cfg(test)]
    pub(crate) fn reset() {
        WINDOW_START_MS.store(0, Ordering::Relaxed);
        LAST_RESIDENT.store(u64::MAX, Ordering::Relaxed);
        LAST_LIVE_SUBS.store(u64::MAX, Ordering::Relaxed);
        LAST_HOSTVIS.store(u64::MAX, Ordering::Relaxed);
        LAST_STORAGE_FREE.store(u64::MAX, Ordering::Relaxed);
        LAST_STORAGE_RES.store(u64::MAX, Ordering::Relaxed);
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn sample(resident: u64) -> Sample {
            sample_subs(resident, 10)
        }

        fn sample_subs(resident: u64, live_subs: u64) -> Sample {
            Sample {
                slab_blocks: 4,
                slab_empty_blocks: 1,
                slab_resident_bytes: resident,
                slab_free_bytes: resident / 2,
                slab_live_subs: live_subs,
                block_allocs: 4,
                block_frees: 0,
                max_block_free_bytes: resident / 8,
                size_buckets: [1, 2, 3, 4, 0, 0, 0, 0],
                registry_len: 20,
                registry_pinned: 0,
                desc_blocks: 1,
                sampled_len: 64,
                target_free_imgs: 3,
                sampled_free_imgs: 2,
                storage_free_imgs: 0,
                storage_recycle_admits: 0,
                storage_recycle_cap_drops: 0,
                storage_resident: 0,
                storage_resident_pinned: 0,
                staging_free_bytes: 0,
                readback_free_bytes: 0,
            }
        }

        /// First call opens the window; a later window with a changed footprint
        /// speaks; an unchanged footprint stays silent (idle steady state).
        #[test]
        fn speaks_on_footprint_change_quiet_when_flat() {
            reset();
            assert_eq!(maybe_line_at(1, &sample(256 << 20)), None);
            let line = maybe_line_at(1 + WINDOW_MS, &sample(320 << 20)).expect("changed");
            assert!(line.contains("resident_mb=320"), "{line}");
            assert!(line.contains("sizes=[1,2,3,4,0,0,0,0]"), "buckets: {line}");
            assert!(line.contains("desc_blocks=1"), "desc arena census: {line}");
            // Same footprint AND same working set next window → quiet.
            assert!(maybe_line_at(1 + 3 * WINDOW_MS, &sample(320 << 20)).is_none());
            // Dropped footprint (a drain returned blocks) → speaks again.
            let line = maybe_line_at(1 + 5 * WINDOW_MS, &sample(192 << 20)).expect("dropped");
            assert!(line.contains("resident_mb=192"), "{line}");
        }

        /// The fragmentation case: the working set drains (live_subs falls) while
        /// the physical footprint does not budge (blocks pinned by survivors).
        /// Keying only on `resident_bytes` would go silent through exactly this;
        /// the `LIVE_SUBS_DELTA` clause must make it speak so the drain-without-
        /// return is visible in the log.
        #[test]
        fn speaks_on_working_set_drain_under_flat_footprint() {
            reset();
            assert_eq!(maybe_line_at(1, &sample_subs(320 << 20, 200)), None);
            // Seed the first emit (subs baseline).
            let _ = maybe_line_at(1 + WINDOW_MS, &sample_subs(320 << 20, 200)).expect("first emit");
            // Same footprint, working set drained 200 → 150: must speak.
            let line = maybe_line_at(1 + 2 * WINDOW_MS, &sample_subs(320 << 20, 150))
                .expect("working-set drain under flat footprint must speak");
            assert!(line.contains("live_subs=150"), "{line}");
            // Same footprint, only a tiny sub move (150 → 149): back to quiet.
            assert!(maybe_line_at(1 + 3 * WINDOW_MS, &sample_subs(320 << 20, 149)).is_none());
        }

        /// A sample with an explicit HOST_VISIBLE staging/readback footprint.
        fn sample_hostvis(resident: u64, staging: u64, readback: u64) -> Sample {
            let mut s = sample_subs(resident, 10);
            s.staging_free_bytes = staging;
            s.readback_free_bytes = readback;
            s
        }

        /// The observability case that motivated `HOSTVIS_DELTA`: after a video
        /// session the slab footprint (`resident_bytes`/`live_subs`) is flat but
        /// the settled-idle trim drains the HOST_VISIBLE staging pool. Those pools
        /// are separate `VkDeviceMemory`, so keying only on the slab would hide the
        /// drain (or a leak). The staging-delta clause must make it speak.
        #[test]
        fn speaks_on_hostvis_drain_under_flat_slab() {
            reset();
            let staged = || sample_hostvis(320 << 20, 177 << 20, 61 << 20);
            assert_eq!(maybe_line_at(1, &staged()), None, "opens window");
            let _ = maybe_line_at(1 + WINDOW_MS, &staged()).expect("first emit seeds baselines");
            // Slab identical, staging drained 177 → 20 MiB: must speak.
            let line = maybe_line_at(
                1 + 2 * WINDOW_MS,
                &sample_hostvis(320 << 20, 20 << 20, 61 << 20),
            )
            .expect("staging drain under flat slab must speak");
            assert!(line.contains("staging_mb=20"), "{line}");
            // A sub-threshold staging wobble (20 → 22 MiB, <8 MiB delta) stays quiet.
            assert!(
                maybe_line_at(
                    1 + 3 * WINDOW_MS,
                    &sample_hostvis(320 << 20, 22 << 20, 61 << 20)
                )
                .is_none(),
                "sub-HOSTVIS_DELTA wobble stays quiet"
            );
            // Readback moving materially also speaks (61 → 3 MiB).
            let line = maybe_line_at(
                1 + 4 * WINDOW_MS,
                &sample_hostvis(320 << 20, 22 << 20, 3 << 20),
            )
            .expect("readback drain must speak");
            assert!(line.contains("readback_mb=3"), "{line}");
        }
    }
}
