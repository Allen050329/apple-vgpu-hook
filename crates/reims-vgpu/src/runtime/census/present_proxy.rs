//! Always-on **present-path census and draw-side drop proxies**.
//!
//! These do **not** change product behavior. They record compact, fail-visible
//! signals so a log census can name a class without opening screenshots.
//!
//! ## Proxies (log: `/tmp/reims-vgpu-thrash.log`, also `observe::fail` on events)
//!
//! | Proxy | Meaning |
//! | --- | --- |
//! | `capture_fail` | DisplaySwap capture returned false — retain hole / keep_prior path |
//! | `stale_online_pending` | A post-ack display IRQ raised with the shared-page ONLINE bit still pending |
//! | `secondary_mrt_drop` / `mrt_mask_bind_miss` | A multi-RT draw degraded to single-RT, or a rendered mask failed to bind at sample time |
//! | `t11_large_fallback` | A large type-11 composite sampled guest pages because no current-generation resident was ready |
//! | `empty_sample` | A resolved fragment/vertex sample whose payload was all-zero |
//!
//! The windowed submodules below (`cadence`, `hitch`, `present_import`,
//! `window_publish`, `capture_source`, `export_present`, `cap_flush`,
//! `idle_drain`, `store_scatter`, `cap_pressure`, `vram`, `lifecycle_churn`)
//! each emit one line per window and stay silent while their counters are zero.

use std::sync::Mutex;

use crate::observe;

struct ThrashState {
    capture_fail: u64,
    /// Dedup for `secondary_mrt_drop`: (reason_code, width, height) already
    /// reported this boot, so a per-draw MRT-secondary drop fires once per
    /// distinct combo, never per frame. Names which build path silently degraded
    /// a multi-RT draw to single-RT — the vibrancy coverage-mask drop that leaves
    /// a later material sample reading zero alpha (transparent tooltip / frosted
    /// pass-through class). Bounded by the small set of
    /// (reason, geometry) combinations a boot produces.
    secondary_mrt_drop_seen: std::collections::BTreeSet<(u8, u32, u32)>,
    secondary_mrt_blend_seen: std::collections::BTreeSet<(u32, u32, u32)>,
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
            capture_fail: 0,
            secondary_mrt_drop_seen: std::collections::BTreeSet::new(),
            secondary_mrt_blend_seen: std::collections::BTreeSet::new(),
            stale_online_pending: 0,
            stale_online_logged: false,
            t11_fb_total: 0,
            t11_fb_seen: std::collections::BTreeSet::new(),
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

/// Single mutex so unit tests and concurrent presents cannot interleave counters.
static STATE: Mutex<ThrashState> = Mutex::new(ThrashState::new());

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
    static PEAK_BATCH: AtomicU64 = AtomicU64::new(0);
    static WINDOW_START_MS: AtomicU64 = AtomicU64::new(0);

    const WINDOW_MS: u64 = 1000;

    /// Record one cap-driven force-flush *batch*: `n` windows landed in this
    /// `try_defer_present_store` call.
    pub fn note(n: u64) {
        if n == 0 {
            return;
        }
        FLUSHES.fetch_add(n, Ordering::Relaxed);
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
        let peak_batch = PEAK_BATCH.swap(0, Ordering::Relaxed);
        if flushes == 0 {
            return None;
        }
        Some(format_line(dt, flushes, peak_batch))
    }

    fn format_line(dt: u64, flushes: u64, peak_batch: u64) -> String {
        // A high, sustained `flushes` means the cap is set below the workload's
        // live working set (raise it, or the drain is lagging).
        format!("cap_flush window_ms={dt} flushes={flushes} peak_batch={peak_batch}")
    }

    #[cfg(test)]
    pub(crate) fn reset() {
        for a in [&FLUSHES, &PEAK_BATCH, &WINDOW_START_MS] {
            a.store(0, Ordering::Relaxed);
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn format_reports_flushes_and_peak_batch() {
            let line = format_line(1000, 8, 5);
            assert!(line.contains("flushes=8"), "{line}");
            assert!(line.contains("peak_batch=5"), "{line}");
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
            PEAK_BATCH.fetch_max(5, Ordering::Relaxed);
            let line = maybe_line_at(1 + 2 * WINDOW_MS + 5).expect("line");
            assert!(line.contains("flushes=8"), "{line}");
            assert!(line.contains("peak_batch=5"), "{line}");
        }
    }
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
    pub capture_fail: u64,
    /// Post-converge display IRQs raised with the ONLINE bit still pending.
    pub stale_online_pending: u64,
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
        capture_fail: st.capture_fail,
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

    #[test]
    fn capture_fail_counted() {
        let _g = test_lock();
        reset_for_test();
        note_capture_fail(5, 1440, 1080, 2);
        assert_eq!(counters().capture_fail, 1);
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
