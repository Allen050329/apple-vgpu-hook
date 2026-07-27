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
//! | `sparse_dock` | Bottom dock strip has low structure vs band above (see [`note_dock_strip`]) | Partial/empty dock glass under multi-window |
//! | `rainbow_menu` | Top menu band is high-chroma **and** spatially incoherent (see [`note_menu_strip`]) | Garbled menu chrome (wrong format/channel/stride) — chroma alone false-fires on a translucent menu over a vibrant wallpaper |
//! | `rect_void` | A populated frame contains a large axis-aligned block of near-black coarse tiles | Stable black quadrants / missing retained damage base |
//! | `damage_hole` | A large connected frame transition encloses a large unchanged rectangle | Old wallpaper retained inside newly painted window bounds |
//! | `selected_peer_divergence` | The protocol-selected retain is sparse while a same-geometry Store peer is dense | Hidden complete ping/pong sibling / incomplete retained Load base |
//! | `tile_composite` | A detected tile-divergence episode was route-B composited or skipped with a named reason | Separates corrected residue from a missing peer-copy precondition |
//!
//! Summary counters: `reims_vgpu_thrash_summary mid_sw=… nz_sw=… sparse=… geom=… fail=… dock=… rainbow=… void=… damage_hole=… peer_divergence=… tile_comp=… tile_comp_skip=…`
//! (emitted every [`SUMMARY_EVERY`] present captures).
//!
//! Always-on census (not a thrash event): `OFF present_strip …` on every display-sized
//! capture — top-band `rgb_nz` / `chroma_hi` / `gray_frac` for menu residual greps.

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

/// Bottom strip height (guest pixels) for dock glass residual measure.
const DOCK_STRIP_H: u32 = 72;

/// Minimum desktop geom for dock strip proxy.
const DOCK_MIN_W: u32 = 800;
const DOCK_MIN_H: u32 = 600;

/// Fraction of dock columns with high dock-vs-above chroma that a full icon row
/// typically shows (live desktop ~0.30). Below this ⇒ sparse_dock.
const DOCK_ACTIVE_MIN: f64 = 0.12;

/// Top menu-bar band height (guest pixels) — live residual is 1920×24-class.
const MENU_STRIP_H: u32 = 24;

/// Display-sized present before top-band menu stats apply.
const MENU_STRIP_MIN_W: u32 = 1280;
const MENU_STRIP_MIN_H: u32 = 720;

/// max(B,G,R)−min(B,G,R) above this ⇒ high-chroma pixel (rainbow static class).
const MENU_CHROMA_HI: u8 = 40;

/// max−min ≤ this and any RGB nonzero ⇒ gray chrome pixel (healthy menu labels).
const MENU_CHROMA_GRAY: u8 = 10;

/// Fraction of top-band pixels with high chroma that fires `rainbow_menu`.
/// Clean grayscale menu: ≈0. Clean with sparse icons: still low. Live rainbow
/// boots: chroma_hi/total ≳0.25–0.50 on the 24-row band (measured p06/rainbow).
const RAINBOW_MENU_CHROMA_FRAC: f64 = 0.12;

/// Minimum top-band RGB occupancy before rainbow fires (ignore pure black bar).
const RAINBOW_MENU_MIN_RGB_FRAC: f64 = 0.02;

/// Adjacent horizontal max-abs channel delta above which a pixel is
/// **spatially incoherent** vs its right neighbor — the noise signature that
/// separates garbled chrome (wrong format/channel/stride) from a legitimately
/// colorful translucent menu bar over a smooth vibrant wallpaper. Chroma alone
/// cannot: a Ventura translucent menu over the orange wallpaper measures
/// chroma_frac≈0.88 — *higher* than the 0.25–0.50 real rainbow boots hit — so
/// chroma_frac fired on ~93% of healthy presents. Live measurement of the
/// dumped present band: real menu incoherent_frac≈0.037 (sparse text edges only)
/// vs synthetic/real rainbow≈0.99 (every neighbor differs).
const MENU_INCOHERENT_DELTA: u8 = 24;

/// Fraction of top-band pixels that must be spatially incoherent (above
/// [`MENU_INCOHERENT_DELTA`]) — in addition to high chroma — before
/// `rainbow_menu` fires. Real vibrant-wallpaper menu ≈0.037; garble ≈0.99;
/// 0.20 sits in the wide gap with margin on both sides.
const RAINBOW_MENU_INCOHERENT_FRAC: f64 = 0.20;

/// Coarse frame grid for stable rectangular-void measurement.
const VOID_GRID_W: usize = 16;
const VOID_GRID_H: usize = 9;

/// A tile is void when at least this fraction of its pixels have zero RGB.
const VOID_TILE_ZERO_FRAC: f64 = 0.98;

/// Do not classify globally sparse boot frames; `sparse_present` owns those.
const VOID_FRAME_MIN_RGB_FRAC: f64 = 0.25;

/// Largest all-void coarse rectangle must cover this fraction of the frame grid.
const VOID_REGION_MIN_FRAC: f64 = 0.15;

/// Coarse transition grid and per-tile RGB sample lattice. The sampled frame is
/// bounded to 32*18*8*8*3 = 110,592 bytes instead of retaining a full scanout.
const DAMAGE_GRID_W: usize = 32;
const DAMAGE_GRID_H: usize = 18;
const DAMAGE_SAMPLES_AXIS: usize = 8;

/// A tile changed when at least 10% of its fixed RGB samples differ.
const DAMAGE_TILE_CHANGED_FRAC: f64 = 0.10;

/// Ignore cursor/menu specks: the connected changed component must cover 10%
/// of the complete grid and its bounds must cover 20%.
const DAMAGE_COMPONENT_MIN_FRAC: f64 = 0.10;
const DAMAGE_BOUNDS_MIN_FRAC: f64 = 0.20;

/// The largest unchanged rectangle enclosed by that component's bounds must
/// cover 3% of the complete frame grid.
const DAMAGE_HOLE_MIN_FRAC: f64 = 0.03;

/// Keep a pre-transition anchor across a bounded run of large, complete
/// intermediate captures. WindowServer may publish chrome/content in stages;
/// rebasing on every adjacent capture loses the old-frame evidence before a
/// retained rectangle becomes visible. This is diagnostic state only.
const DAMAGE_REBASE_CAPTURES: u8 = 8;

/// Diagnostic-only classification threshold (never gates presentation): the
/// enclosed hole is called a background match when its mean RGB is within this
/// per-channel distance of the surrounding desktop (tiles outside the changed
/// component bounds). A match means the "window interior" is showing the same
/// content as the wallpaper around it — genuine stale-background — whereas a
/// distinct interior color is a legitimately static window body that simply did
/// not change frame-to-frame. This only enriches the log line.
const DAMAGE_BG_MATCH_TOL: f64 = 24.0;

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
    /// Capture target is a proven compositor-output member (decoded graph
    /// edge). Member↔member mid switches are guest double-buffer alternation:
    /// counted separately (`named_sw`) and logged via `mid_switch_named`
    /// instead of the THRASH `mid_switch` class. nz/structure swings still
    /// fire on named switches (a sparse member frame is real signal).
    pub named_peer: bool,
}

struct ThrashState {
    last: Option<PresentCaptureSample>,
    presents: u64,
    mid_switches: u64,
    named_switches: u64,
    nz_swings: u64,
    structure_swings: u64,
    sparse: u64,
    geom_mismatch: u64,
    capture_fail: u64,
    sparse_dock: u64,
    rainbow_menu: u64,
    rect_void: u64,
    damage_hole: u64,
    /// Subset of `damage_hole` where the enclosed rectangle matched the
    /// surrounding desktop (`bg_match=1`) — a genuine stale-background suspect,
    /// as opposed to a legitimately-static window body. Surfaced in the summary
    /// so a boot's fail-log shows at a glance whether any real hole occurred.
    damage_hole_bg: u64,
    selected_peer_divergence: u64,
    /// Distinct a/b inter-buffer retention-gap episodes: a presented compositor
    /// member whose last full frame lags a same-geometry peer by
    /// [`RETENTION_GAP_MARGIN`]+ full frames (it received neither a full-frame
    /// Store nor a seed while the peer advanced). Protocol-structural sibling of
    /// `selected_peer_divergence` — keyed on the full-frame-Store sequence, not
    /// nz occupancy, so it cannot be fooled by legitimately-dark content and
    /// names the structural cause. Measure-only.
    dense_retention_gap: u64,
    /// Dedup for `dense_retention_gap`: presented_mid → the peer_seq last
    /// reported, so a sustained gap fires once per newly-widened episode, never
    /// per present. Pruned lazily (bounded by the live member set).
    dense_gap_active: std::collections::BTreeMap<u32, u64>,
    /// Distinct per-tile damage-coverage divergence episodes. A presented mid
    /// holds stale tiles a same-geometry
    /// peer erased — the residue class (stuck menu dropdown, rubber-band trail)
    /// that whole-frame `dense_retention_gap` cannot see because the seqs match.
    /// Measure-only in this increment; the cross-mid tile composite is later.
    tile_divergence: u64,
    /// Dedup for `tile_divergence`: presented_mid → the divergent tile COUNT last
    /// reported, so a sustained residue fires once per changed count, never per
    /// present (the anti-flood the reverted prototype lacked).
    tile_divergence_active: std::collections::BTreeMap<u32, u32>,
    /// Distinct route-B tile-composite outcomes for tile-divergence episodes:
    /// `applied` when the peer-copy preconditions held and at least one Vulkan
    /// region was recorded, otherwise `skipped reason=<slug>` naming the exact
    /// missing precondition. This is a result census for the already-detected
    /// divergence; it never influences which frame is presented.
    tile_composite_applied: u64,
    tile_composite_skipped: u64,
    /// Dedup the latest composite result shape. Cleared when `tile_divergence`
    /// clears, so a later episode with the same shape re-fires without logging
    /// every present while a residue is stable.
    tile_composite_active: Option<TileCompositeLast>,
    /// Distinct torn-capture substitutions: a ClearOnly present's selection (via
    /// store_fifo ring order or a graph fallback) picked a compositor member
    /// whose full-frame sequence (`dense_frame_seq`) lagged a same-geometry peer
    /// by [`RETENTION_GAP_MARGIN`]+ full frames — a member that missed a RUN of
    /// full frames (the fullscreen-transition vertical-strip + checkerboard torn
    /// frame). The present drain substituted the full-frame-
    /// freshest peer as the capture source; this counts each save. Steady-state
    /// alternation stays below the margin and never substitutes.
    stale_present_substitute: u64,
    /// Dedup for `stale_present_substitute`: selected_mid → the denser peer's
    /// `dense_frame_seq` last reported, so a sustained lag fires once per
    /// newly-widened episode, never per present.
    stale_subst_active: std::collections::BTreeMap<u32, u64>,
    /// Dedup for `secondary_mrt_drop`: (reason_code, width, height) already
    /// reported this boot, so a per-draw MRT-secondary drop fires once per
    /// distinct combo, never per frame. Names which build path silently degraded
    /// a multi-RT draw to single-RT — the vibrancy coverage-mask drop that leaves
    /// a later material sample reading zero alpha (transparent tooltip / frosted
    /// pass-through class). Bounded by the small set of
    /// (reason, geometry) combinations a boot produces.
    secondary_mrt_drop_seen: std::collections::BTreeSet<(u8, u32, u32)>,
    secondary_mrt_blend_seen: std::collections::BTreeSet<(u32, u32, u32)>,
    /// Dedup for [`note_tile_composite_unpresented_peer`], keyed on the
    /// `(presented_mid, peer_mid)` pair so a persistent leak logs once per pair
    /// per boot rather than once per present.
    tile_comp_unpresented_seen: std::collections::BTreeSet<(u32, u32)>,
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
    /// Dual-mid peer dense-retention-seed fires this boot. The seed is a
    /// **relaxed one-shot** (`DeviceState::peer_needs_front_seed`): it re-arms
    /// re-arms only on a strictly-newer full frame, so each fire is a genuine
    /// ≥`RETENTION_GAP_MARGIN` retention-gap fix. Pure telemetry (`peer_seeds=` in
    /// the summary): under active multi-video compositing it legitimately climbs
    /// into the thousands (many real dense-frame changes), so there is no runtime
    /// flood alarm — the margin gate + its unit test are the regression guard (see
    /// `note_peer_front_seed`).
    peer_seeds: u64,
    /// Large-mapping type-11 zero-copy fallbacks (composite sampled guest
    /// pages because no current-generation resident was ready). Always-on
    /// black-band discriminator: `rect_void` firing while the recent ring is
    /// EMPTY means the composite never fell back — the zeros came through the
    /// resident/cache path (guest-painted); a populated ring names the mids
    /// whose residents were missing at sample time.
    t11_fb_total: u64,
    t11_fb_ring: [Option<T11FallbackEvent>; T11_FB_RING],
    t11_fb_next: usize,
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
    damage: std::collections::BTreeMap<u32, DamageAnchor>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TileCompositeLast {
    outcome: TileComposite,
    width: u32,
    height: u32,
    rects: usize,
    regions: usize,
}

/// How a detected tile-divergence episode ended.
///
/// Every skip is a **requirement** that did not hold (ready + BGRA + exact
/// geometry), so a miss exports the presented mid's own frame unchanged and can
/// never darken a tile. It is still a correction that did not run, which is why
/// each one is named rather than counted as "skipped".
///
/// A `Refusal` and not a `Decline`: `Applied` and `NoPeerRequested` are the two
/// values that are not skips at all, and `Emit::refusal` is what makes them
/// unable to write a `reason=` by accident — the exact defect the census survey
/// found on the sibling `import_content` line.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TileComposite {
    /// The peer copy ran: preconditions held and `regions>0` were recorded.
    Applied,
    /// No peer was offered for this present — not a tile-composite situation.
    NoPeerRequested,
    /// A peer was offered with an empty damage-rect list.
    EmptyRects,
    /// The peer resolved but produced no copy regions.
    EmptyRegions,
    /// The peer identity is the presented target itself.
    SameIdentity,
    /// The peer identity is not in the resident registry.
    PeerMissing,
    /// The peer resident has never had content written.
    PeerNotReady,
    /// The peer resident is not BGRA, so a copy would mis-order channels.
    PeerNotBgra,
    /// The peer resident's geometry differs from the presented target's.
    PeerGeomMismatch,
}

impl crate::observe::Refusal for TileComposite {
    fn refusal(&self) -> Option<&'static str> {
        match self {
            Self::Applied | Self::NoPeerRequested => None,
            Self::EmptyRects => Some("tile_peer_empty_rects"),
            Self::EmptyRegions => Some("tile_peer_empty_regions"),
            Self::SameIdentity => Some("tile_peer_same_identity"),
            Self::PeerMissing => Some("tile_peer_missing"),
            Self::PeerNotReady => Some("tile_peer_not_ready"),
            Self::PeerNotBgra => Some("tile_peer_not_bgra"),
            Self::PeerGeomMismatch => Some("tile_peer_geom_mismatch"),
        }
    }
}

impl TileComposite {
    /// `status=` on the census line: whether the correction ran.
    pub fn status(&self) -> &'static str {
        match self {
            Self::Applied => "applied",
            _ => "skipped",
        }
    }
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

/// One large-mapping t11 zero-copy fallback (see `t11_fb_ring`).
#[derive(Clone, Copy)]
struct T11FallbackEvent {
    mid: u32,
    map_gen: u32,
    /// Newest any-generation registry entry for the surface at sample time:
    /// `(generation, content_ready)`; `None` when absent entirely.
    probe: Option<(u64, bool)>,
    at: std::time::Instant,
}

/// Ring depth for recent fallback events (a band episode composites a handful
/// of tiles per frame; 8 covers the working set of one episode frame).
const T11_FB_RING: usize = 8;

/// Ring entries older than this are omitted from `rect_void_ctx` (stale
/// fallbacks from long before the episode carry no signal for it).
const T11_FB_RECENT_MS: u128 = 5_000;

impl ThrashState {
    const fn new() -> Self {
        Self {
            last: None,
            presents: 0,
            mid_switches: 0,
            named_switches: 0,
            nz_swings: 0,
            structure_swings: 0,
            sparse: 0,
            geom_mismatch: 0,
            capture_fail: 0,
            sparse_dock: 0,
            rainbow_menu: 0,
            rect_void: 0,
            damage_hole: 0,
            damage_hole_bg: 0,
            selected_peer_divergence: 0,
            dense_retention_gap: 0,
            dense_gap_active: std::collections::BTreeMap::new(),
            tile_divergence: 0,
            tile_divergence_active: std::collections::BTreeMap::new(),
            tile_composite_applied: 0,
            tile_composite_skipped: 0,
            tile_composite_active: None,
            stale_present_substitute: 0,
            stale_subst_active: std::collections::BTreeMap::new(),
            secondary_mrt_drop_seen: std::collections::BTreeSet::new(),
            tile_comp_unpresented_seen: std::collections::BTreeSet::new(),
            secondary_mrt_blend_seen: std::collections::BTreeSet::new(),
            first_dense_seen: false,
            post_converge_nondense_run: 0,
            post_converge_regress: 0,
            stale_online_pending: 0,
            stale_online_logged: false,
            peer_seeds: 0,
            t11_fb_total: 0,
            t11_fb_ring: [None; T11_FB_RING],
            t11_fb_next: 0,
            last_summary_ms: 0,
            last_summary_presents: 0,
            damage: std::collections::BTreeMap::new(),
        }
    }
}

/// Record a large-mapping type-11 zero-copy fallback (always-on; called from
/// the sample rail when a ≥250k-px mapping lacks a current-generation ready
/// resident). Rare on a healthy boot — the black-band discriminator.
pub fn note_t11_large_fallback(mid: u32, map_gen: u32, probe: Option<(u64, bool)>) {
    let mut st = STATE.lock().unwrap_or_else(|e| e.into_inner());
    st.t11_fb_total = st.t11_fb_total.saturating_add(1);
    let slot = st.t11_fb_next % T11_FB_RING;
    st.t11_fb_ring[slot] = Some(T11FallbackEvent {
        mid,
        map_gen,
        probe,
        at: std::time::Instant::now(),
    });
    st.t11_fb_next = st.t11_fb_next.wrapping_add(1);
}

/// Per-mapping damage baseline. Anchors are keyed by mapping id because
/// compositor output double-buffers: presents alternate mids, and a single
/// global anchor would re-base on every switch — the damage_hole proxy could
/// never accumulate old-frame evidence on an alternating stream.
struct DamageAnchor {
    frame: DamageFrame,
    rebase_count: u8,
}

/// Bounded anchors (each is a coarse 32×18 sample grid). Presents concentrate
/// on a handful of live mids; evict the lowest id when full.
const DAMAGE_ANCHOR_CAP: usize = 8;

#[derive(Clone, Debug, PartialEq, Eq)]
struct DamageFrame {
    width: u32,
    height: u32,
    samples: Vec<[u8; 3]>,
}

/// Measure-only description of an unchanged rectangular hole enclosed by one
/// large connected frame transition.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DamageHoleStats {
    pub changed_tiles: usize,
    pub component_tiles: usize,
    pub bounds_x: usize,
    pub bounds_y: usize,
    pub bounds_w: usize,
    pub bounds_h: usize,
    pub hole_tiles: usize,
    pub hole_x: usize,
    pub hole_y: usize,
    pub hole_w: usize,
    pub hole_h: usize,
}

fn sample_damage_frame(frame_bgra: &[u8], width: u32, height: u32) -> Option<DamageFrame> {
    let width_usize = width as usize;
    let height_usize = height as usize;
    if width_usize < DAMAGE_GRID_W || height_usize < DAMAGE_GRID_H {
        return None;
    }
    let stride = width_usize.checked_mul(4)?;
    let need = stride.checked_mul(height_usize)?;
    if frame_bgra.len() < need {
        return None;
    }
    let mut samples = Vec::with_capacity(
        DAMAGE_GRID_W * DAMAGE_GRID_H * DAMAGE_SAMPLES_AXIS * DAMAGE_SAMPLES_AXIS,
    );
    for gy in 0..DAMAGE_GRID_H {
        let y0 = gy * height_usize / DAMAGE_GRID_H;
        let y1 = (gy + 1) * height_usize / DAMAGE_GRID_H;
        for gx in 0..DAMAGE_GRID_W {
            let x0 = gx * width_usize / DAMAGE_GRID_W;
            let x1 = (gx + 1) * width_usize / DAMAGE_GRID_W;
            for sy in 0..DAMAGE_SAMPLES_AXIS {
                let y = y0 + (2 * sy + 1) * (y1 - y0) / (2 * DAMAGE_SAMPLES_AXIS);
                for sx in 0..DAMAGE_SAMPLES_AXIS {
                    let x = x0 + (2 * sx + 1) * (x1 - x0) / (2 * DAMAGE_SAMPLES_AXIS);
                    let o = y * stride + x * 4;
                    samples.push([frame_bgra[o], frame_bgra[o + 1], frame_bgra[o + 2]]);
                }
            }
        }
    }
    Some(DamageFrame {
        width,
        height,
        samples,
    })
}

fn damage_hole_stats(previous: &DamageFrame, current: &DamageFrame) -> Option<DamageHoleStats> {
    if previous.width != current.width
        || previous.height != current.height
        || previous.samples.len() != current.samples.len()
    {
        return None;
    }
    let samples_per_tile = DAMAGE_SAMPLES_AXIS * DAMAGE_SAMPLES_AXIS;
    let grid_tiles = DAMAGE_GRID_W * DAMAGE_GRID_H;
    let mut changed = vec![false; grid_tiles];
    let mut changed_tiles = 0usize;
    for (tile, cell) in changed.iter_mut().enumerate() {
        let start = tile * samples_per_tile;
        let end = start + samples_per_tile;
        let different = previous.samples[start..end]
            .iter()
            .zip(&current.samples[start..end])
            .filter(|(a, b)| a != b)
            .count();
        *cell = different as f64 / samples_per_tile as f64 >= DAMAGE_TILE_CHANGED_FRAC;
        changed_tiles += *cell as usize;
    }

    let min_component = (grid_tiles as f64 * DAMAGE_COMPONENT_MIN_FRAC).ceil() as usize;
    let min_bounds = (grid_tiles as f64 * DAMAGE_BOUNDS_MIN_FRAC).ceil() as usize;
    let min_hole = (grid_tiles as f64 * DAMAGE_HOLE_MIN_FRAC).ceil() as usize;
    let mut visited = vec![false; grid_tiles];
    let mut best = DamageHoleStats {
        changed_tiles,
        ..DamageHoleStats::default()
    };
    for start in 0..grid_tiles {
        if !changed[start] || visited[start] {
            continue;
        }
        let mut stack = vec![start];
        visited[start] = true;
        let mut component = Vec::new();
        while let Some(tile) = stack.pop() {
            component.push(tile);
            let x = tile % DAMAGE_GRID_W;
            let y = tile / DAMAGE_GRID_W;
            let neighbors = [
                x.checked_sub(1).map(|nx| y * DAMAGE_GRID_W + nx),
                (x + 1 < DAMAGE_GRID_W).then_some(y * DAMAGE_GRID_W + x + 1),
                y.checked_sub(1).map(|ny| ny * DAMAGE_GRID_W + x),
                (y + 1 < DAMAGE_GRID_H).then_some((y + 1) * DAMAGE_GRID_W + x),
            ];
            for neighbor in neighbors.into_iter().flatten() {
                if changed[neighbor] && !visited[neighbor] {
                    visited[neighbor] = true;
                    stack.push(neighbor);
                }
            }
        }
        if component.len() < min_component {
            continue;
        }
        let min_x = component.iter().map(|tile| tile % DAMAGE_GRID_W).min()?;
        let max_x = component.iter().map(|tile| tile % DAMAGE_GRID_W).max()?;
        let min_y = component.iter().map(|tile| tile / DAMAGE_GRID_W).min()?;
        let max_y = component.iter().map(|tile| tile / DAMAGE_GRID_W).max()?;
        let bounds_w = max_x - min_x + 1;
        let bounds_h = max_y - min_y + 1;
        if bounds_w * bounds_h < min_bounds {
            continue;
        }

        let mut local_best = DamageHoleStats {
            changed_tiles,
            component_tiles: component.len(),
            bounds_x: min_x,
            bounds_y: min_y,
            bounds_w,
            bounds_h,
            ..DamageHoleStats::default()
        };
        for y0 in min_y..=max_y {
            for y1 in y0 + 1..=max_y + 1 {
                for x0 in min_x..=max_x {
                    for x1 in x0 + 1..=max_x + 1 {
                        let area = (x1 - x0) * (y1 - y0);
                        if area <= local_best.hole_tiles {
                            continue;
                        }
                        // A genuine "background didn't clear" hole is ENCLOSED by
                        // the changed window on all four sides — the window
                        // painted around a stale interior rectangle. An unchanged
                        // rectangle that touches the changed component's
                        // bounding-box boundary is not enclosed: it is the
                        // window failing to fill its bbox edge (e.g. a scrolling
                        // window whose bbox bottom row is the correctly-static
                        // wallpaper below it). Requiring the hole strictly inside
                        // the bbox keeps every window-interior hole and drops the
                        // edge-strip false positive that made `damage_hole_bg`
                        // cry wolf on ordinary Safari scrolling. Because the hole
                        // is the maximal unchanged rectangle, a strict-interior
                        // position also implies a changed tile on each side (else
                        // the rectangle would extend), i.e. real enclosure.
                        let enclosed = x0 > min_x && x1 <= max_x && y0 > min_y && y1 <= max_y;
                        let all_unchanged = enclosed
                            && (y0..y1).all(|y| (x0..x1).all(|x| !changed[y * DAMAGE_GRID_W + x]));
                        if all_unchanged {
                            local_best.hole_tiles = area;
                            local_best.hole_x = x0;
                            local_best.hole_y = y0;
                            local_best.hole_w = x1 - x0;
                            local_best.hole_h = y1 - y0;
                        }
                    }
                }
            }
        }
        if local_best.hole_tiles >= min_hole && local_best.hole_tiles > best.hole_tiles {
            best = local_best;
        }
    }
    (best.hole_tiles >= min_hole).then_some(best)
}

/// Measure-only classification of an already-detected damage hole: does the
/// enclosed unchanged rectangle show the same content as the surrounding desktop
/// (genuine stale-background — the "background did not clear" bug), or a distinct
/// window body (a legitimately static interior)? Computed from the current
/// frame's coarse grid samples only; never influences presentation.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct DamageHoleClass {
    hole_rgb: [u8; 3],
    periph_rgb: [u8; 3],
    /// Max per-channel |hole_mean - periph_mean|.
    dist: f64,
    /// Max per-channel (max_sample - min_sample) inside the hole: near-zero for
    /// a flat window body, large for a wallpaper gradient.
    hole_spread: u8,
    /// Number of grid tiles outside the component bounds that fed periph_rgb.
    periph_tiles: usize,
    bg_match: bool,
}

fn tile_sample_range(frame: &DamageFrame, gx: usize, gy: usize) -> impl Iterator<Item = &[u8; 3]> {
    let samples_per_tile = DAMAGE_SAMPLES_AXIS * DAMAGE_SAMPLES_AXIS;
    let tile = gy * DAMAGE_GRID_W + gx;
    let start = tile * samples_per_tile;
    frame.samples[start..start + samples_per_tile].iter()
}

fn classify_damage_hole(current: &DamageFrame, s: &DamageHoleStats) -> Option<DamageHoleClass> {
    let samples_per_tile = DAMAGE_SAMPLES_AXIS * DAMAGE_SAMPLES_AXIS;
    if current.samples.len() < DAMAGE_GRID_W * DAMAGE_GRID_H * samples_per_tile {
        return None;
    }
    // Hole mean + per-channel spread over its grid tiles.
    let mut hole_sum = [0u64; 3];
    let mut hole_n = 0u64;
    let mut hole_min = [u8::MAX; 3];
    let mut hole_max = [0u8; 3];
    for gy in s.hole_y..(s.hole_y + s.hole_h) {
        for gx in s.hole_x..(s.hole_x + s.hole_w) {
            for px in tile_sample_range(current, gx, gy) {
                for c in 0..3 {
                    hole_sum[c] += px[c] as u64;
                    hole_min[c] = hole_min[c].min(px[c]);
                    hole_max[c] = hole_max[c].max(px[c]);
                }
                hole_n += 1;
            }
        }
    }
    if hole_n == 0 {
        return None;
    }
    // Peripheral desktop mean: grid tiles entirely outside the component bounds.
    let (bx0, by0) = (s.bounds_x, s.bounds_y);
    let (bx1, by1) = (s.bounds_x + s.bounds_w, s.bounds_y + s.bounds_h);
    let mut periph_sum = [0u64; 3];
    let mut periph_n = 0u64;
    let mut periph_tiles = 0usize;
    for gy in 0..DAMAGE_GRID_H {
        for gx in 0..DAMAGE_GRID_W {
            let inside = gx >= bx0 && gx < bx1 && gy >= by0 && gy < by1;
            if inside {
                continue;
            }
            periph_tiles += 1;
            for px in tile_sample_range(current, gx, gy) {
                for c in 0..3 {
                    periph_sum[c] += px[c] as u64;
                }
                periph_n += 1;
            }
        }
    }
    if periph_n == 0 {
        return None;
    }
    let hole_rgb = [
        (hole_sum[0] / hole_n) as u8,
        (hole_sum[1] / hole_n) as u8,
        (hole_sum[2] / hole_n) as u8,
    ];
    let periph_rgb = [
        (periph_sum[0] / periph_n) as u8,
        (periph_sum[1] / periph_n) as u8,
        (periph_sum[2] / periph_n) as u8,
    ];
    let dist = (0..3)
        .map(|c| (hole_rgb[c] as f64 - periph_rgb[c] as f64).abs())
        .fold(0.0, f64::max);
    let hole_spread = (0..3)
        .map(|c| hole_max[c].saturating_sub(hole_min[c]))
        .max()
        .unwrap_or(0);
    Some(DamageHoleClass {
        hole_rgb,
        periph_rgb,
        dist,
        hole_spread,
        periph_tiles,
        bg_match: dist <= DAMAGE_BG_MATCH_TOL,
    })
}

fn damage_changed_tiles(previous: &DamageFrame, current: &DamageFrame) -> Option<usize> {
    if previous.width != current.width
        || previous.height != current.height
        || previous.samples.len() != current.samples.len()
    {
        return None;
    }
    let samples_per_tile = DAMAGE_SAMPLES_AXIS * DAMAGE_SAMPLES_AXIS;
    Some(
        previous
            .samples
            .chunks_exact(samples_per_tile)
            .zip(current.samples.chunks_exact(samples_per_tile))
            .filter(|(a, b)| {
                let different = a.iter().zip(*b).filter(|(x, y)| x != y).count();
                different as f64 / samples_per_tile as f64 >= DAMAGE_TILE_CHANGED_FRAC
            })
            .count(),
    )
}

/// Record old-frame rectangles retained inside a large connected transition.
/// The bounded anchor survives staged intermediate captures; sampled pixels are
/// diagnostics only and never influence presentation.
pub fn note_damage_hole(
    mapping_id: u32,
    generation: u32,
    width: u32,
    height: u32,
    frame_bgra: &[u8],
) {
    let Some(current) = sample_damage_frame(frame_bgra, width, height) else {
        return;
    };
    let mut st = STATE.lock().unwrap_or_else(|e| e.into_inner());
    let Some(anchor) = st.damage.get_mut(&mapping_id) else {
        if st.damage.len() >= DAMAGE_ANCHOR_CAP {
            let evict = *st.damage.keys().next().expect("non-empty at cap");
            st.damage.remove(&evict);
        }
        st.damage.insert(
            mapping_id,
            DamageAnchor {
                frame: current,
                rebase_count: 0,
            },
        );
        return;
    };
    if anchor.frame.width != current.width || anchor.frame.height != current.height {
        anchor.frame = current;
        anchor.rebase_count = 0;
        return;
    }
    let stats = damage_hole_stats(&anchor.frame, &current);
    let Some(s) = stats else {
        let changed = damage_changed_tiles(&anchor.frame, &current).unwrap_or(0);
        let min_component =
            ((DAMAGE_GRID_W * DAMAGE_GRID_H) as f64 * DAMAGE_COMPONENT_MIN_FRAC).ceil() as usize;
        if changed >= min_component {
            anchor.rebase_count = anchor.rebase_count.saturating_add(1);
            if anchor.rebase_count >= DAMAGE_REBASE_CAPTURES {
                anchor.frame = current;
                anchor.rebase_count = 0;
            }
        } else {
            anchor.rebase_count = 0;
        }
        return;
    };
    // Classify before `current` is moved into the anchor: does the enclosed hole
    // show the surrounding desktop (`bg_match=1`, loud) or a distinct,
    // legitimately-static window body (`bg_match=0`, quiet)?
    //
    // A missing classification means there is no periphery to sample because the
    // changed component fills the whole grid — a full-screen recomposite whose
    // "hole" is a coincidentally-static strip (menu bar / dock edge), not the
    // window-over-wallpaper bug, so route it quiet. Any other missing
    // classification (grid too small — never for a real display) stays loud.
    let full_frame = s.bounds_w >= DAMAGE_GRID_W && s.bounds_h >= DAMAGE_GRID_H;
    let class = classify_damage_hole(&current, &s);
    let bg_match = match class {
        Some(c) => c.bg_match,
        None => !full_frame,
    };
    anchor.frame = current;
    anchor.rebase_count = 0;
    st.damage_hole = st.damage_hole.saturating_add(1);
    if bg_match {
        st.damage_hole_bg = st.damage_hole_bg.saturating_add(1);
    }
    drop(st);
    let grid_tiles = DAMAGE_GRID_W * DAMAGE_GRID_H;
    let class_str = match class {
        Some(c) => format!(
            " bg_match={} dist={:.0} hole_rgb=[{},{},{}] periph_rgb=[{},{},{}] hole_spread={} periph_tiles={}",
            c.bg_match as u8,
            c.dist,
            c.hole_rgb[0],
            c.hole_rgb[1],
            c.hole_rgb[2],
            c.periph_rgb[0],
            c.periph_rgb[1],
            c.periph_rgb[2],
            c.hole_spread,
            c.periph_tiles,
        ),
        None if full_frame => " bg_match=0 class=full_frame_recomposite".to_string(),
        None => String::new(),
    };
    let line = format!(
        "damage_hole mid={mapping_id} gen={generation} {width}x{height} changed_tiles={} component_tiles={} bounds={}x{}+{}+{} hole_tiles={} hole={}x{}+{}+{} frac={:.4}{class_str}",
        s.changed_tiles,
        s.component_tiles,
        s.bounds_w,
        s.bounds_h,
        s.bounds_x,
        s.bounds_y,
        s.hole_tiles,
        s.hole_w,
        s.hole_h,
        s.hole_x,
        s.hole_y,
        s.hole_tiles as f64 / grid_tiles as f64
    );
    // A genuine stale-background suspect (`bg_match=1`) is fail-visible; a
    // legitimately-static window body (`bg_match=0`, distinct from the
    // surrounding desktop) is recorded to the always-on thrash file only so it
    // does not dilute the curated fail-log. The summary line (fail-visible)
    // still carries the full `damage_hole` total and the `damage_hole_bg`
    // subset, so no event is hidden from a boot audit.
    if bg_match {
        thrash_line(&line);
    } else {
        thrash_line_quiet(&line);
    }
}

/// Coarse, measure-only description of a large rectangular RGB void.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RectVoidStats {
    pub rgb_nz: usize,
    pub void_tiles: usize,
    pub largest_tiles: usize,
    pub grid_x: usize,
    pub grid_y: usize,
    pub grid_w: usize,
    pub grid_h: usize,
}

/// Find the largest axis-aligned rectangle of near-black coarse tiles.
///
/// This is deliberately a content census only. It is never consulted by decode,
/// execute, ownership, or present selection.
pub fn rect_void_stats_bgra(
    frame_bgra: &[u8],
    width: u32,
    height: u32,
    rgb_nz: usize,
) -> Option<RectVoidStats> {
    let width = width as usize;
    let height = height as usize;
    if width < VOID_GRID_W || height < VOID_GRID_H {
        return None;
    }
    let stride = width.checked_mul(4)?;
    let need = stride.checked_mul(height)?;
    if frame_bgra.len() < need {
        return None;
    }

    // `rgb_nz` is the caller's already-computed BGRA nonzero-pixel count (the
    // fused `bgra_present_stats` pass), not a second full scan of the frame.
    let pixels = width.checked_mul(height)?;
    if pixels == 0 || rgb_nz as f64 / (pixels as f64) < VOID_FRAME_MIN_RGB_FRAC {
        return None;
    }

    let mut void = [[false; VOID_GRID_W]; VOID_GRID_H];
    let mut void_tiles = 0usize;
    for (gy, row) in void.iter_mut().enumerate() {
        let y0 = gy * height / VOID_GRID_H;
        let y1 = (gy + 1) * height / VOID_GRID_H;
        for (gx, cell) in row.iter_mut().enumerate() {
            let x0 = gx * width / VOID_GRID_W;
            let x1 = (gx + 1) * width / VOID_GRID_W;
            let total = (x1 - x0).saturating_mul(y1 - y0);
            let mut zero = 0usize;
            for y in y0..y1 {
                let row_off = y * stride;
                for x in x0..x1 {
                    let o = row_off + x * 4;
                    zero += (frame_bgra[o] == 0 && frame_bgra[o + 1] == 0 && frame_bgra[o + 2] == 0)
                        as usize;
                }
            }
            *cell = total > 0 && zero as f64 / total as f64 >= VOID_TILE_ZERO_FRAC;
            void_tiles += *cell as usize;
        }
    }

    // The grid is tiny (16×9), so exhaustive rectangles keep the definition
    // obvious and deterministic without a stateful image-analysis dependency.
    let mut best = RectVoidStats {
        rgb_nz,
        void_tiles,
        ..RectVoidStats::default()
    };
    for y0 in 0..VOID_GRID_H {
        for y1 in y0 + 1..=VOID_GRID_H {
            for x0 in 0..VOID_GRID_W {
                for x1 in x0 + 1..=VOID_GRID_W {
                    let area = (x1 - x0) * (y1 - y0);
                    if area <= best.largest_tiles {
                        continue;
                    }
                    let all_void = void[y0..y1]
                        .iter()
                        .all(|row| row[x0..x1].iter().all(|&v| v));
                    if all_void {
                        best.largest_tiles = area;
                        best.grid_x = x0;
                        best.grid_y = y0;
                        best.grid_w = x1 - x0;
                        best.grid_h = y1 - y0;
                    }
                }
            }
        }
    }
    Some(best)
}

/// Record stable black-quadrant / missing-retained-base evidence.
///
/// Returns `Some(stats)` with the firing void rectangle when a qualifying void
/// fired (so the caller can run the cheap `note_rect_void_origin` band compare
/// only on a real firing), `None` on the common no-void path.
pub fn note_rect_void(
    mapping_id: u32,
    generation: u32,
    width: u32,
    height: u32,
    frame_bgra: &[u8],
    rgb_nz: usize,
) -> Option<RectVoidStats> {
    // Exact early-out (no signal loss): a qualifying void is a rectangle of at
    // least `VOID_REGION_MIN_FRAC` of the grid tiles, each ≥ `VOID_TILE_ZERO_FRAC`
    // black, so a firing frame needs ≥ `VOID_REGION_MIN_FRAC * VOID_TILE_ZERO_FRAC`
    // of all pixels black. If fewer are black than that floor, no void can fire —
    // skip both full-frame scans. On a healthy dense desktop this drops rect_void
    // to zero cost on the present drain (was two 8 MiB passes / present).
    let pixels = (width as usize).saturating_mul(height as usize);
    if pixels == 0 {
        return None;
    }
    let black = pixels.saturating_sub(rgb_nz);
    let min_black = (VOID_REGION_MIN_FRAC * VOID_TILE_ZERO_FRAC * pixels as f64) as usize;
    if black < min_black {
        return None;
    }
    let s = rect_void_stats_bgra(frame_bgra, width, height, rgb_nz)?;
    let grid_tiles = VOID_GRID_W * VOID_GRID_H;
    if s.largest_tiles as f64 / (grid_tiles as f64) < VOID_REGION_MIN_FRAC {
        return None;
    }
    let mut st = STATE.lock().unwrap_or_else(|e| e.into_inner());
    st.rect_void = st.rect_void.saturating_add(1);
    thrash_line(&format!(
        "rect_void mid={mapping_id} gen={generation} {width}x{height} rgb_nz={} void_tiles={} largest_tiles={} grid={}x{}+{}+{} frac={:.4}",
        s.rgb_nz,
        s.void_tiles,
        s.largest_tiles,
        s.grid_w,
        s.grid_h,
        s.grid_x,
        s.grid_y,
        s.largest_tiles as f64 / grid_tiles as f64
    ));
    // Discriminator context: did the composite fall back to guest pages for
    // any large mapping in the last few seconds? `recent=[]` = the zeros came
    // through resident/cache (guest-painted); entries name the fallback mids
    // (probe = newest any-generation registry entry at sample time).
    let now = std::time::Instant::now();
    let mut recent = String::new();
    for e in st.t11_fb_ring.iter().flatten() {
        let age = now.duration_since(e.at).as_millis();
        if age > T11_FB_RECENT_MS {
            continue;
        }
        if !recent.is_empty() {
            recent.push(' ');
        }
        match e.probe {
            Some((g, r)) => recent.push_str(&format!(
                "mid={}:gen={}:probe={}:{}:{}ms",
                e.mid, e.map_gen, g, r as u8, age
            )),
            None => recent.push_str(&format!(
                "mid={}:gen={}:probe=none:{}ms",
                e.mid, e.map_gen, age
            )),
        }
    }
    thrash_line(&format!(
        "rect_void_ctx t11_fb_total={} recent=[{recent}]",
        st.t11_fb_total
    ));
    Some(s)
}

/// Convert a firing void rectangle's grid cells to inclusive-exclusive pixel
/// bounds, matching the exact grid math `rect_void_stats_bgra` uses.
fn void_pixel_bounds(s: &RectVoidStats, width: u32, height: u32) -> (usize, usize, usize, usize) {
    let width = width as usize;
    let height = height as usize;
    let x0 = s.grid_x * width / VOID_GRID_W;
    let x1 = (s.grid_x + s.grid_w) * width / VOID_GRID_W;
    let y0 = s.grid_y * height / VOID_GRID_H;
    let y1 = (s.grid_y + s.grid_h) * height / VOID_GRID_H;
    (x0, x1, y0, y1)
}

/// Count RGB-nonzero pixels inside a pixel rectangle of a tight BGRA8 frame.
fn band_rgb_nz(frame_bgra: &[u8], width: u32, bounds: (usize, usize, usize, usize)) -> usize {
    let (x0, x1, y0, y1) = bounds;
    let stride = (width as usize).saturating_mul(4);
    if stride == 0 || frame_bgra.len() < stride.saturating_mul(y1) {
        return 0;
    }
    let mut nz = 0usize;
    for y in y0..y1 {
        let row = y * stride;
        for x in x0..x1 {
            let o = row + x * 4;
            nz += (frame_bgra[o] != 0 || frame_bgra[o + 1] != 0 || frame_bgra[o + 2] != 0) as usize;
        }
    }
    nz
}

/// Dedup set for `rect_void_origin`: (grid signature, origin-code, src-code) so a
/// sustained void fires the origin discriminator once per distinct combo instead
/// of ~10k lines/boot. Bounded by the tiny grid × {retention_loss, persistent}
/// × {src kinds} space.
static VOID_ORIGIN_SEEN: std::sync::Mutex<Option<std::collections::BTreeSet<(u32, u8, u8)>>> =
    std::sync::Mutex::new(None);

/// Discriminate WHERE a firing `rect_void`'s black band came from: content that
/// existed in the previously-retained frame and was **lost** this present
/// (`origin=retention_loss` — a host retention failure we can fix) vs a band that
/// was **never** populated (`origin=persistent_black` — guest never composited it,
/// or a persistent gap the seed can't source). Measure-only, runs on the present
/// drain (off the QEMU main core), deduped so it never floods. `prev_frame` is the
/// still-live retained `frame_bgra` (capture replaces it AFTER the proxy block),
/// compared only when its geometry matches this capture.
#[allow(
    clippy::too_many_arguments,
    reason = "the proxy records the full present identity and damage rectangle"
)]
pub fn note_rect_void_origin(
    s: &RectVoidStats,
    new_frame: &[u8],
    prev_frame: &[u8],
    prev_w: u32,
    prev_h: u32,
    width: u32,
    height: u32,
    src: &str,
) {
    if prev_w != width || prev_h != height {
        return;
    }
    let bounds = void_pixel_bounds(s, width, height);
    let (x0, x1, y0, y1) = bounds;
    let band_px = (x1.saturating_sub(x0)).saturating_mul(y1.saturating_sub(y0));
    if band_px == 0 {
        return;
    }
    let new_nz = band_rgb_nz(new_frame, width, bounds);
    let prev_nz = band_rgb_nz(prev_frame, width, bounds);
    // The band is a firing void, so `new_nz` is ~0 by construction. The signal is
    // whether the SAME band held content one present ago: > 10% populated in the
    // retained frame means we had it and dropped it (retention loss).
    let prev_frac = prev_nz as f64 / band_px as f64;
    let (origin, ocode) = if prev_frac >= 0.10 {
        ("retention_loss", 0u8)
    } else {
        ("persistent_black", 1u8)
    };
    let scode = match src {
        "reuse_store" => 0u8,
        "host_cache" => 1,
        "guest_pages_contig" => 2,
        "guest_pages_frag" => 3,
        _ => 4,
    };
    let sig = ((s.grid_x as u32) << 24)
        | ((s.grid_y as u32) << 16)
        | ((s.grid_w as u32) << 8)
        | (s.grid_h as u32);
    let mut guard = VOID_ORIGIN_SEEN.lock().unwrap_or_else(|e| e.into_inner());
    let seen = guard.get_or_insert_with(std::collections::BTreeSet::new);
    if !seen.insert((sig, ocode, scode)) {
        return;
    }
    observe::fail(format!(
        "rect_void_origin origin={origin} src={src} grid={}x{}+{}+{} band_px={band_px} new_nz={new_nz} prev_nz={prev_nz} prev_frac={prev_frac:.3}",
        s.grid_w, s.grid_h, s.grid_x, s.grid_y
    ));
}

/// Thin horizontal chrome strip (menu bar / label layers), not a full desktop.
///
/// Live residual class: 1920×24 BGRA strips + partials (1877×24, 43×24). Used by
/// import Store enrich so thin surfaces are not under-logged (resident stats used
/// to skip `h < 720`). Measure-only — never gates Store/present.
#[inline]
pub fn is_menu_strip_geom(width: u32, height: u32) -> bool {
    height > 0 && height <= 64 && width >= 640
}

/// Top-band occupancy / chroma for a tight BGRA8 present frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MenuStripStats {
    pub strip_h: u32,
    pub total: usize,
    /// Pixels with any of B/G/R nonzero.
    pub rgb_nz: usize,
    /// Pixels with channel max−min > [`MENU_CHROMA_HI`].
    pub chroma_hi: usize,
    /// Nonzero RGB and max−min ≤ [`MENU_CHROMA_GRAY`] (healthy gray labels).
    pub gray: usize,
    /// Pixels whose right horizontal neighbor differs by more than
    /// [`MENU_INCOHERENT_DELTA`] on any channel (spatial-noise signature; the
    /// last column of each row is excluded — no right neighbor).
    pub incoherent: usize,
    pub px0: [u8; 4],
}

/// Compute top-band menu stats. `None` when geom is not display-sized or buffer short.
pub fn menu_strip_stats_bgra(frame_bgra: &[u8], width: u32, height: u32) -> Option<MenuStripStats> {
    if width < MENU_STRIP_MIN_W || height < MENU_STRIP_MIN_H {
        return None;
    }
    let strip_h = MENU_STRIP_H.min(height);
    if strip_h == 0 {
        return None;
    }
    let stride = (width as usize).saturating_mul(4);
    let need = stride.saturating_mul(strip_h as usize);
    if frame_bgra.len() < need {
        return None;
    }
    let mut rgb_nz = 0usize;
    let mut chroma_hi = 0usize;
    let mut gray = 0usize;
    let mut incoherent = 0usize;
    let total = (width as usize).saturating_mul(strip_h as usize);
    let px0 = if need >= 4 {
        [frame_bgra[0], frame_bgra[1], frame_bgra[2], frame_bgra[3]]
    } else {
        [0, 0, 0, 0]
    };
    let w = width as usize;
    for y in 0..strip_h as usize {
        let row = y * stride;
        for x in 0..w {
            let o = row + x * 4;
            let b = frame_bgra[o];
            let g = frame_bgra[o + 1];
            let r = frame_bgra[o + 2];
            let m = b.max(g).max(r);
            // Spatial-incoherence signal (garble vs smooth translucent chrome):
            // max abs channel delta to the right neighbor. Independent of the
            // black-bar early-continue below so a garbled band with black gaps
            // still registers as noisy.
            if x + 1 < w {
                let n = o + 4;
                let d = b.abs_diff(frame_bgra[n]).max(
                    g.abs_diff(frame_bgra[n + 1])
                        .max(r.abs_diff(frame_bgra[n + 2])),
                );
                if d > MENU_INCOHERENT_DELTA {
                    incoherent += 1;
                }
            }
            if m == 0 {
                continue;
            }
            rgb_nz += 1;
            let ch = m - b.min(g).min(r);
            if ch > MENU_CHROMA_HI {
                chroma_hi += 1;
            } else if ch <= MENU_CHROMA_GRAY {
                gray += 1;
            }
        }
    }
    Some(MenuStripStats {
        strip_h,
        total,
        rgb_nz,
        chroma_hi,
        gray,
        incoherent,
        px0,
    })
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

/// A presented compositor member can legitimately lag a same-geometry peer by
/// ~1 full frame during healthy a/b alternation (the peer got the newest full
/// frame; this buffer inherits it via the next seed). Only a lag of this many
/// full frames — the buffer missed BOTH a full-frame Store and the inter-buffer
/// seed, repeatedly — is the retention-gap regression, not normal alternation.
pub(crate) const RETENTION_GAP_MARGIN: u64 = 4;

/// Protocol-structural a/b inter-buffer retention-gap guard. `presented_mid` (at
/// `w`×`h`) holds a full frame `presented_seq` old while a same-geometry
/// `peer_mid` holds `peer_seq`; fires once per newly-widened episode when the
/// lag ≥ [`RETENTION_GAP_MARGIN`]. Unlike `selected_peer_divergence` (nz
/// occupancy) this keys on the full-frame-Store SEQUENCE
/// ([`crate::model::DeviceState::dense_retention_gap`]), so it names the
/// structural cause and cannot be fooled by legitimately-dark content.
/// Measure-only; the caller runs it on the present drain (off the main core).
pub fn note_dense_retention_gap(
    presented_mid: u32,
    peer_mid: u32,
    presented_seq: u64,
    peer_seq: u64,
    width: u32,
    height: u32,
) -> bool {
    if peer_seq < presented_seq.saturating_add(RETENTION_GAP_MARGIN) {
        return false;
    }
    let mut st = STATE.lock().unwrap_or_else(|e| e.into_inner());
    // Dedup: fire once per newly-widened gap for this presented buffer (keyed on
    // the peer_seq that widened it), never per present while the gap persists.
    if st.dense_gap_active.get(&presented_mid) == Some(&peer_seq) {
        return false;
    }
    st.dense_gap_active.insert(presented_mid, peer_seq);
    st.dense_retention_gap = st.dense_retention_gap.saturating_add(1);
    drop(st);
    thrash_line(&format!(
        "dense_retention_gap presented_mid={presented_mid} presented_seq={presented_seq} peer_mid={peer_mid} peer_seq={peer_seq} lag={} {width}x{height}",
        peer_seq - presented_seq
    ));
    true
}

/// Per-tile damage-coverage divergence proxy, measure-only. `tiles` divergent
/// cells (peer fresher by the retention
/// margin) with the divergent-tile grid bbox `[gx0,gy0,gx1,gy1]` on a
/// `TILE_GEN_GRID_W×TILE_GEN_GRID_H` grid. Emits ONCE per changed divergent count
/// for a presented mid (deduped, never per present), so a sustained residue is
/// one line and a healthy/steady frame is silent. `tiles==0` clears the dedup so
/// the next episode re-fires. Cheap: the caller already did the generation
/// compare; this only logs. Returns true when a new/widened episode was logged.
pub fn note_tile_divergence(
    presented_mid: u32,
    peer_mid: u32,
    tiles: u32,
    bbox: [u32; 4],
    width: u32,
    height: u32,
) -> bool {
    let mut st = STATE.lock().unwrap_or_else(|e| e.into_inner());
    if tiles == 0 {
        // Residue cleared: drop the dedup so a fresh episode re-fires.
        st.tile_divergence_active.remove(&presented_mid);
        st.tile_composite_active = None;
        return false;
    }
    if st.tile_divergence_active.get(&presented_mid) == Some(&tiles) {
        return false;
    }
    st.tile_divergence_active.insert(presented_mid, tiles);
    st.tile_divergence = st.tile_divergence.saturating_add(1);
    drop(st);
    thrash_line(&format!(
        "tile_divergence presented_mid={presented_mid} peer_mid={peer_mid} tiles={tiles} bbox=[{},{},{},{}] {width}x{height}",
        bbox[0], bbox[1], bbox[2], bbox[3]
    ));
    true
}

/// Route-B tile-composite result census for an already-detected
/// [`note_tile_divergence`] episode. `reason="applied"` means the peer-copy
/// preconditions held and `regions>0` copy regions were recorded; every other
/// reason is a named skip precondition (`peer_missing`, `peer_not_ready`,
/// `peer_not_bgra`, `peer_geom_mismatch`, `empty_regions`, ...). Always-on and
/// fail-log visible, but behavior-neutral: this only names whether the existing
/// divergent-tile correction actually ran. Returns true when a distinct result
/// was logged.
pub fn note_tile_composite_result(
    outcome: TileComposite,
    rects: usize,
    regions: usize,
    width: u32,
    height: u32,
) -> bool {
    if rects == 0 || width == 0 || height == 0 {
        return false;
    }
    let event = TileCompositeLast {
        outcome,
        width,
        height,
        rects,
        regions,
    };
    let mut st = STATE.lock().unwrap_or_else(|e| e.into_inner());
    if st.tile_composite_active == Some(event) {
        return false;
    }
    st.tile_composite_active = Some(event);
    if outcome == TileComposite::Applied {
        st.tile_composite_applied = st.tile_composite_applied.saturating_add(1);
    } else {
        st.tile_composite_skipped = st.tile_composite_skipped.saturating_add(1);
    }
    drop(st);
    // `status=` says whether the correction ran; `reason=` appears **only** when
    // it did not, because a success has no reason to name. The line used to
    // carry `reason=applied`, which is the shape that teaches a reader to ignore
    // `reason=`. Field order is identical in both branches so one grep reads
    // either.
    let tail = format!(
        "status={} rects={rects} regions={regions}",
        outcome.status()
    );
    match observe::Emit::refusal("tile_composite", &outcome) {
        Some(e) => e
            .field("status", outcome.status())
            .field("rects", rects)
            .field("regions", regions)
            .field("geom", format!("{width}x{height}"))
            .off(),
        None => observe::off(format!("tile_composite {tail} geom={width}x{height}")),
    }
    true
}

/// Measure-only: the tile-composite path selected a `peer_mid` that has NEVER
/// been displayed at this geometry ([`crate::model::DeviceState::presented_at`])
/// as the source of `rects` divergent tiles. `compositor_geometry_peer` is not
/// presented-gated (its distinct-resident divergence patches the black-band
/// class), so a never-displayed full-frame publisher can bleed its tiles onto a
/// real output — the residue/tiles symptom, and the tile-level analogue of the
/// `peer_presented=0` whole-frame substitution that
/// [`crate::model::DeviceState::dense_retention_gap`] now excludes. This has been
/// dormant on every traced x86 boot; a nonzero count is the reproduction a real
/// gate here would need. Deduped on `(presented_mid, peer_mid)` — one line per
/// pair per boot, not per present. Returns whether a new pair was logged.
pub fn note_tile_composite_unpresented_peer(
    presented_mid: u32,
    peer_mid: u32,
    rects: usize,
    width: u32,
    height: u32,
) -> bool {
    let mut st = STATE.lock().unwrap_or_else(|e| e.into_inner());
    if !st.tile_comp_unpresented_seen.insert((presented_mid, peer_mid)) {
        return false;
    }
    drop(st);
    observe::fail(format!(
        "tile_composite_unpresented_peer presented_mid={presented_mid} peer_mid={peer_mid} \
         rects={rects} {width}x{height} (compositing tiles from a never-displayed peer)"
    ));
    true
}

/// Fullscreen-transition torn-capture guard + regression proxy. The present
/// drain substituted the full-frame-freshest same-geometry peer (`denser_mid`)
/// for a selected member (`selected_mid`, via `mode`) whose full-frame sequence
/// (`dense_frame_seq`) lagged by [`RETENTION_GAP_MARGIN`]+ — a member that missed
/// a RUN of full frames, whose stale / partially-unwritten pages are the
/// vertical-strip + checkerboard torn frame. Keyed purely on the decoded
/// full-frame-Store sequence (protocol state), never pixel content. Fires once
/// per newly-widened episode (deduped on the denser peer's seq), never per
/// present. The caller runs it on the present drain (off the QEMU main core).
pub fn note_stale_present_substitute(
    selected_mid: u32,
    selected_seq: u64,
    denser_mid: u32,
    denser_seq: u64,
    width: u32,
    height: u32,
    mode: &str,
    peer_presented: bool,
) -> bool {
    if denser_seq <= selected_seq {
        return false;
    }
    let mut st = STATE.lock().unwrap_or_else(|e| e.into_inner());
    if st.stale_subst_active.get(&selected_mid) == Some(&denser_seq) {
        return false;
    }
    st.stale_subst_active.insert(selected_mid, denser_seq);
    st.stale_present_substitute = st.stale_present_substitute.saturating_add(1);
    drop(st);
    // `peer_presented` separates the two populations this substitution serves:
    // a genuine swapchain sibling has itself been presented at this geometry
    // (`presented_at`), whereas a never-presented full-frame publisher (a WebKit
    // content tile / offscreen scratch surface) has not — substituting the
    // latter hands one logical output's frame to another (the residue class).
    thrash_line(&format!(
        "stale_present_substitute selected_mid={selected_mid} selected_seq={selected_seq} denser_mid={denser_mid} denser_seq={denser_seq} lag={} mode={mode} peer_presented={} {width}x{height}",
        denser_seq - selected_seq,
        peer_presented as u8
    ));
    true
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

/// Thrash record that stays out of the curated fail-log: it lands only in the
/// always-on `/tmp/reims-vgpu-thrash.log` (still greppable) plus the `REIMS_VGPU_DRAW_LOG`
/// verbose sink. For high-cadence, benign-by-classification proxy lines whose
/// count is already surfaced in the fail-visible summary.
fn thrash_line_quiet(msg: &str) {
    observe::line(format!("THRASH {msg}"));
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
/// (`export_present_from_resident_composited_fd_policy` → `retire_all`).
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
            // Both captures on proven compositor-output members: guest
            // double-buffer alternation (decoded graph provenance), not the
            // dual-mid thrash class. Counted apart so mid_sw stays a clean
            // regression signal; content swings below still apply.
            let named_switch = p.named_peer && sample.named_peer;
            if named_switch {
                st.named_switches = st.named_switches.saturating_add(1);
            } else {
                st.mid_switches = st.mid_switches.saturating_add(1);
            }
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
                } else if !named_switch {
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
            "summary presents={} present_hz={:.1} mid_sw={} named_sw={} nz_sw={} struct_sw={} sparse={} geom={} fail={} dock={} rainbow={} void={} damage_hole={} damage_hole_bg={} peer_divergence={} dense_gap={} tile_comp={} tile_comp_skip={} stale_subst={} peer_seeds={} converged={} post_converge_regress={} stale_online={} t11_fb={}",
            st.presents,
            present_hz,
            st.mid_switches,
            st.named_switches,
            st.nz_swings,
            st.structure_swings,
            st.sparse,
            st.geom_mismatch,
            st.capture_fail,
            st.sparse_dock,
            st.rainbow_menu,
            st.rect_void,
            st.damage_hole,
            st.damage_hole_bg,
            st.selected_peer_divergence,
            st.dense_retention_gap,
            st.tile_composite_applied,
            st.tile_composite_skipped,
            st.stale_present_substitute,
            st.peer_seeds,
            st.first_dense_seen as u8,
            st.post_converge_regress,
            st.stale_online_pending,
            st.t11_fb_total
        ));
    }
}

/// Measure-only: top menu-bar band on a full-desktop present retain.
///
/// Always logs `OFF present_strip` (census). Fires `THRASH rainbow_menu` when the
/// top band has meaningful RGB occupancy, high chroma fraction, **and** high
/// spatial incoherence — the visual "rainbow chrome" class (wrong
/// format/channel/stride), not missing present. The incoherence gate is what
/// separates garble (noisy neighbors) from a legitimately colorful translucent
/// menu bar over a smooth vibrant wallpaper (coherent neighbors), which chroma
/// fraction alone could not.
///
/// Does **not** gate present/decode/execute.
pub fn note_menu_strip(
    mapping_id: u32,
    generation: u32,
    width: u32,
    height: u32,
    frame_bgra: &[u8],
) {
    let Some(s) = menu_strip_stats_bgra(frame_bgra, width, height) else {
        return;
    };
    let total = s.total.max(1) as f64;
    let rgb_frac = s.rgb_nz as f64 / total;
    let chroma_frac = s.chroma_hi as f64 / total;
    let gray_frac = s.gray as f64 / total;
    let incoherent_frac = s.incoherent as f64 / total;
    // Per-present menu-band census (~1.4k/25s under a continuously-animating
    // app). Diagnostic detail used offline to tune the rainbow_menu gate + the
    // once-per-process PPM dump; the always-on menu anomaly signal is the
    // `rainbow_menu` THRASH line below + the `summary rainbow=` counter, so gate
    // the raw census behind REIMS_VGPU_DRAW_LOG (available for A/B tuning under
    // REIMS_VGPU_DRAW_LOG=1 without flooding a normal boot).
    observe::line(format!(
        "present_strip mid={mapping_id} gen={generation} {width}x{height} strip_h={} rgb_nz={} chroma_hi={} gray={} incoherent={} total={} rgb_frac={rgb_frac:.4} chroma_frac={chroma_frac:.4} gray_frac={gray_frac:.4} incoherent_frac={incoherent_frac:.4} px0=[{},{},{},{}]",
        s.strip_h,
        s.rgb_nz,
        s.chroma_hi,
        s.gray,
        s.incoherent,
        s.total,
        s.px0[0],
        s.px0[1],
        s.px0[2],
        s.px0[3]
    ));
    // Garble requires BOTH high chroma AND spatial incoherence. A translucent
    // menu over a smooth vibrant wallpaper is high-chroma but coherent
    // (incoherent_frac≈0.04); genuine rainbow/format-stride garble is noisy
    // (≈0.99). Chroma alone false-fired on ~93% of healthy presents.
    if rgb_frac < RAINBOW_MENU_MIN_RGB_FRAC
        || chroma_frac < RAINBOW_MENU_CHROMA_FRAC
        || incoherent_frac < RAINBOW_MENU_INCOHERENT_FRAC
    {
        return;
    }
    let mut st = STATE.lock().unwrap_or_else(|e| e.into_inner());
    st.rainbow_menu = st.rainbow_menu.saturating_add(1);
    thrash_line(&format!(
        "rainbow_menu mid={mapping_id} gen={generation} {width}x{height} strip_h={} rgb_nz={} chroma_hi={} gray={} chroma_frac={chroma_frac:.4} gray_frac={gray_frac:.4} incoherent_frac={incoherent_frac:.4}",
        s.strip_h, s.rgb_nz, s.chroma_hi, s.gray
    ));
    // Once-per-process: dump the top strip as PPM for A/B row inspection.
    static DUMPED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !DUMPED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        let path = format!(
            "/tmp/reims-vgpu-menu-strip-present-mid{mapping_id}-{width}x{}.ppm",
            s.strip_h
        );
        if let Ok(mut f) = std::fs::File::create(&path) {
            use std::io::Write;
            let _ = writeln!(f, "P6\n{width} {}\n255", s.strip_h);
            let stride = (width as usize) * 4;
            let band = &frame_bgra[..stride
                .saturating_mul(s.strip_h as usize)
                .min(frame_bgra.len())];
            for px in band.chunks_exact(4) {
                let _ = f.write_all(&[px[2], px[1], px[0]]);
            }
            observe::fail(format!(
                "menu_strip dump path={path} mid={mapping_id} chroma_frac={chroma_frac:.4}"
            ));
        }
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

/// Measure-only: bottom dock strip structure vs the band above it.
///
/// Full desktop dock icons create column-wise difference between the dock
/// strip and wallpaper immediately above. Glass-only / missing-icon frames
/// show low `active_frac` (live residual under multi-window Calendar).
///
/// Skips early-boot logo/clear frames (`frame_nz` fraction below 0.25) so the
/// proxy does not fire on every pre-desktop present (live boot dock=63 noise).
///
/// `frame_bgra` is tight BGRA8 row-major (`width * height * 4`).
pub fn note_dock_strip(
    mapping_id: u32,
    generation: u32,
    width: u32,
    height: u32,
    frame_bgra: &[u8],
    frame_nz: usize,
) {
    if width < DOCK_MIN_W || height < DOCK_MIN_H {
        return;
    }
    let total = (width as usize)
        .saturating_mul(height as usize)
        .saturating_mul(4);
    // Settled full desktop is ~0.95+ nonzero; dual-mid incomplete (~0.32) and
    // logo frames must not flood sparse_dock (live dock=122 on incomplete mids).
    if total == 0 || (frame_nz as f64 / total as f64) < 0.55 {
        return;
    }
    let strip_h = DOCK_STRIP_H.min(height / 8);
    if strip_h < 16 {
        return;
    }
    let stride = (width as usize).saturating_mul(4);
    let need = stride.saturating_mul(height as usize);
    if frame_bgra.len() < need {
        return;
    }
    // Dock: last strip_h rows. Above: strip_h rows ending strip_h*2 from bottom.
    let dock_y0 = (height - strip_h) as usize;
    let above_y0 = (height - strip_h * 2) as usize;
    // Sample every 4th column; collect activity flags for whole / left / right.
    let mut col_active: Vec<bool> = Vec::with_capacity((width as usize) / 4 + 1);
    let mut x = 0u32;
    while x < width {
        let mut dock_sum = [0u64; 3];
        let mut above_sum = [0u64; 3];
        let mut n = 0u64;
        for dy in 0..strip_h as usize {
            let d_off = (dock_y0 + dy) * stride + (x as usize) * 4;
            let a_off = (above_y0 + dy) * stride + (x as usize) * 4;
            // BGRA
            dock_sum[0] += frame_bgra[d_off + 2] as u64;
            dock_sum[1] += frame_bgra[d_off + 1] as u64;
            dock_sum[2] += frame_bgra[d_off] as u64;
            above_sum[0] += frame_bgra[a_off + 2] as u64;
            above_sum[1] += frame_bgra[a_off + 1] as u64;
            above_sum[2] += frame_bgra[a_off] as u64;
            n += 1;
        }
        let Some(divisor) = std::num::NonZeroU64::new(n) else {
            x = x.saturating_add(4);
            continue;
        };
        let n = divisor.get();
        let dr = (dock_sum[0] / n) as i32;
        let dg = (dock_sum[1] / n) as i32;
        let db = (dock_sum[2] / n) as i32;
        let ar = (above_sum[0] / n) as i32;
        let ag = (above_sum[1] / n) as i32;
        let ab = (above_sum[2] / n) as i32;
        let diff = (dr - ar).unsigned_abs() + (dg - ag).unsigned_abs() + (db - ab).unsigned_abs();
        // Full dock icons: live desktop column mean abs ~60; thr=25 filters noise.
        col_active.push(diff > 25);
        x = x.saturating_add(4);
    }
    let cols = col_active.len() as u32;
    if cols == 0 {
        return;
    }
    let active = col_active.iter().filter(|&&a| a).count() as u32;
    let active_frac = active as f64 / cols as f64;
    // Partial dock (live Safari residual): left half has icons, right half is
    // glass/wallpaper. Overall active_frac can sit ~0.30 (left-only icons).
    let half = (cols / 2).max(1) as usize;
    let left_active = col_active[..half.min(col_active.len())]
        .iter()
        .filter(|&&a| a)
        .count() as u32;
    let right_slice = &col_active[half.min(col_active.len())..];
    let right_active = right_slice.iter().filter(|&&a| a).count() as u32;
    let left_frac = left_active as f64 / half as f64;
    let right_n = right_slice.len().max(1) as f64;
    let right_frac = right_active as f64 / right_n;
    let glass_only = active_frac < DOCK_ACTIVE_MIN;
    let partial_left = left_frac >= DOCK_ACTIVE_MIN
        && right_frac < DOCK_ACTIVE_MIN * 0.5
        && (left_frac - right_frac) > 0.15;
    if !glass_only && !partial_left {
        return;
    }
    let mut st = STATE.lock().unwrap_or_else(|e| e.into_inner());
    st.sparse_dock = st.sparse_dock.saturating_add(1);
    thrash_line(&format!(
        "sparse_dock mid={mapping_id} gen={generation} {width}x{height} active_frac={active_frac:.3} left={left_frac:.3} right={right_frac:.3} cols={cols}"
    ));
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
    pub named_switches: u64,
    pub nz_swings: u64,
    pub structure_swings: u64,
    pub sparse: u64,
    pub geom_mismatch: u64,
    pub capture_fail: u64,
    pub sparse_dock: u64,
    pub rainbow_menu: u64,
    pub rect_void: u64,
    pub damage_hole: u64,
    pub damage_hole_bg: u64,
    pub selected_peer_divergence: u64,
    /// Distinct a/b inter-buffer retention-gap episodes (structural sibling of
    /// `selected_peer_divergence`; see [`note_dense_retention_gap`]).
    pub dense_retention_gap: u64,
    /// Distinct route-B peer-copy results for tile-divergence episodes.
    pub tile_composite_applied: u64,
    pub tile_composite_skipped: u64,
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

/// Count each dual-mid peer dense-retention-seed fire this boot. Surfaced as
/// `peer_seeds=` in the periodic summary — pure telemetry, no alarm.
///
/// There is deliberately **no** runtime flood alarm here. An earlier version
/// tripped a one-shot `peer_seed_flood` line once a cumulative-count threshold
/// (64) was crossed, on the assumption that legitimate fires are "a handful per
/// boot" bounded by real dense-frame changes. That assumption only holds at
/// idle: the seed fires once per *strictly-newer* full-frame Store on a peer
/// lagging by ≥`RETENTION_GAP_MARGIN` (`DeviceState::peer_needs_front_seed`), and
/// under active multi-video compositing there are thousands of such dense-frame
/// changes per boot — measured live `peer_seeds` climbing past 2000 (and >18000
/// under a quad-4K grid), every fire a genuine ≥margin retention-gap fix. Worse,
/// fires are counted per *draw* while presents are per *present*, so no
/// cumulative count, rate, or seeds-vs-presents ratio separates legitimate heavy
/// seeding from the every-flip regression the alarm targeted (both scale with
/// draw activity). The **only** discriminator is the per-seed lag, and that is
/// enforced structurally by the margin gate in `peer_needs_front_seed` and locked
/// by the `peer_front_seed_gate_is_structural_and_once_per_lifetime` unit test
/// (1-lag → no seed; ≥margin → seed; one-shot until the lag re-widens). So a
/// removed/bypassed margin is caught at test time; the runtime alarm added no
/// coverage and cried wolf on every active-workload boot. Runs on the
/// render/drain worker, never the QEMU main loop. Returns the running count.
pub fn note_peer_front_seed() -> u64 {
    let mut st = STATE.lock().unwrap_or_else(|e| e.into_inner());
    st.peer_seeds = st.peer_seeds.saturating_add(1);
    st.peer_seeds
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
        named_switches: st.named_switches,
        nz_swings: st.nz_swings,
        structure_swings: st.structure_swings,
        sparse: st.sparse,
        geom_mismatch: st.geom_mismatch,
        capture_fail: st.capture_fail,
        sparse_dock: st.sparse_dock,
        rainbow_menu: st.rainbow_menu,
        rect_void: st.rect_void,
        damage_hole: st.damage_hole,
        damage_hole_bg: st.damage_hole_bg,
        selected_peer_divergence: st.selected_peer_divergence,
        dense_retention_gap: st.dense_retention_gap,
        tile_composite_applied: st.tile_composite_applied,
        tile_composite_skipped: st.tile_composite_skipped,
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
    if let Ok(mut g) = VOID_ORIGIN_SEEN.lock() {
        *g = None;
    }
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
            named_peer: false,
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
            named_peer: false,
        }
    }

    #[test]
    fn named_member_switch_is_not_mid_switch_thrash() {
        let _g = test_lock();
        reset_for_test();
        let a = (1440u64 * 1080 * 50 / 100) as usize;
        let b = (1440u64 * 1080 * 48 / 100) as usize;
        let named = |mid: u32, px: usize| PresentCaptureSample {
            named_peer: true,
            ..sample(mid, px, 1440, 1080)
        };
        note_capture_ok(named(1, a));
        note_capture_ok(named(5, b));
        note_capture_ok(named(1, a));
        let c = counters();
        assert_eq!(
            c.mid_switches, 0,
            "member↔member alternation must not count as mid_switch thrash"
        );
        assert_eq!(c.named_switches, 2, "named switches counted separately");
        // A switch to a non-member capture is still the thrash class.
        note_capture_ok(sample(4, b, 1440, 1080));
        assert_eq!(counters().mid_switches, 1);
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

    /// The peer front-seed counter is pure telemetry with NO runtime flood alarm:
    /// under active multi-video compositing it legitimately climbs into the
    /// thousands (every fire a genuine ≥margin retention-gap fix — measured live
    /// >2000, >18000 under quad-4K), and no cumulative count/rate/ratio separates
    /// > that from the every-flip regression (fires are per-draw, presents
    /// > per-present). The margin gate + `peer_front_seed_gate_is_structural_and_
    /// once_per_lifetime` are the regression guard. This locks that a high count
    /// > never emits a `peer_seed_flood` (or any) fail line, so a future edit cannot
    /// > silently reintroduce the cry-wolf alarm.
    #[test]
    fn peer_front_seed_counts_without_flood_alarm() {
        let _g = test_lock();
        reset_for_test();
        let mut last = 0;
        for _ in 0..500 {
            last = note_peer_front_seed();
        }
        assert_eq!(last, 500, "every fire is counted (telemetry)");
        let log = std::fs::read_to_string(observe::fail_log_path()).unwrap_or_default();
        assert!(
            !log.contains("peer_seed_flood"),
            "no runtime flood alarm — legitimate heavy seeding must never fail-log"
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
    fn named_member_switch_still_fires_nz_swing() {
        let _g = test_lock();
        reset_for_test();
        let full = (1440u64 * 1080 * 40 / 100) as usize;
        let logo = (1440u64 * 1080 * 5 / 100) as usize;
        let named = |mid: u32, px: usize| PresentCaptureSample {
            named_peer: true,
            ..sample(mid, px, 1440, 1080)
        };
        note_capture_ok(named(1, full));
        note_capture_ok(named(5, logo));
        let c = counters();
        assert_eq!(c.mid_switches, 0);
        assert_eq!(
            c.nz_swings, 1,
            "sparse member frame on alternation is real signal"
        );
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

    #[test]
    fn rectangular_void_fires_inside_populated_frame_only() {
        let _g = test_lock();
        let w = 320u32;
        let h = 180u32;
        let stride = w as usize * 4;
        let mut frame = vec![0u8; stride * h as usize];
        for px in frame.chunks_exact_mut(4) {
            px.copy_from_slice(&[24, 48, 96, 255]);
        }
        // Caller-supplied BGRA nonzero-pixel count (the fused capture scan).
        let rgb_nz_of = |f: &[u8]| -> usize {
            f.chunks_exact(4)
                .filter(|p| p[0] != 0 || p[1] != 0 || p[2] != 0)
                .count()
        };

        reset_for_test();
        note_rect_void(6, 19, w, h, &frame, rgb_nz_of(&frame));
        assert_eq!(counters().rect_void, 0, "populated frame must stay quiet");

        // Synthetic stable black right half: globally 50% populated, but a
        // large rectangular retained-base hole occupies 8x9 coarse tiles.
        for y in 0..h as usize {
            for x in (w as usize / 2)..w as usize {
                let o = y * stride + x * 4;
                frame[o..o + 4].copy_from_slice(&[0, 0, 0, 255]);
            }
        }
        let s =
            rect_void_stats_bgra(&frame, w, h, rgb_nz_of(&frame)).expect("populated frame stats");
        assert_eq!(s.grid_x, 8);
        assert_eq!(s.grid_y, 0);
        assert_eq!(s.grid_w, 8);
        assert_eq!(s.grid_h, 9);
        assert_eq!(s.largest_tiles, 72);

        note_rect_void(6, 20, w, h, &frame, rgb_nz_of(&frame));
        assert_eq!(counters().rect_void, 1, "large rectangular void must fire");
        let body = std::fs::read_to_string(crate::observe::fail_log_path())
            .expect("reims-vgpu-fail.log readable");
        assert!(body.lines().any(|line| {
            line.starts_with("THRASH rect_void mid=6 gen=20 320x180 ")
                && line.contains("largest_tiles=72 grid=8x9+8+0")
        }));

        // Globally sparse/black boot frames belong to sparse_present, not this
        // dense-frame rectangular-hole class.
        reset_for_test();
        let black = vec![0u8; stride * h as usize];
        assert_eq!(rect_void_stats_bgra(&black, w, h, 0), None);
        note_rect_void(6, 21, w, h, &black, 0);
        assert_eq!(counters().rect_void, 0);
    }

    #[test]
    fn rect_void_origin_discriminates_retention_loss_from_persistent_black() {
        let _g = test_lock();
        let w = 320u32;
        let h = 180u32;
        let stride = w as usize * 4;
        let rgb_nz_of = |f: &[u8]| -> usize {
            f.chunks_exact(4)
                .filter(|p| p[0] != 0 || p[1] != 0 || p[2] != 0)
                .count()
        };
        // A dense frame whose right half went black — a firing void of 8x9 tiles
        // at grid +8+0 (same construction as the rect_void test).
        let mut frame = vec![0u8; stride * h as usize];
        for px in frame.chunks_exact_mut(4) {
            px.copy_from_slice(&[24, 48, 96, 255]);
        }
        for y in 0..h as usize {
            for x in (w as usize / 2)..w as usize {
                let o = y * stride + x * 4;
                frame[o..o + 4].copy_from_slice(&[0, 0, 0, 255]);
            }
        }

        // Case 1: the retained frame HELD content in the now-black band ->
        // retention_loss.
        reset_for_test();
        let full = {
            let mut f = vec![0u8; stride * h as usize];
            for px in f.chunks_exact_mut(4) {
                px.copy_from_slice(&[24, 48, 96, 255]);
            }
            f
        };
        let s = note_rect_void(1, 1, w, h, &frame, rgb_nz_of(&frame)).expect("void must fire");
        note_rect_void_origin(&s, &frame, &full, w, h, w, h, "reuse_store");
        let body = std::fs::read_to_string(crate::observe::fail_log_path())
            .expect("reims-vgpu-fail.log readable");
        assert!(
            body.lines()
                .any(|l| l.contains("rect_void_origin origin=retention_loss")
                    && l.contains("src=reuse_store")
                    && l.contains("grid=8x9+8+0")),
            "retained content in the band must read as retention_loss"
        );

        // Case 2: the retained frame was ALSO black in the band -> persistent.
        reset_for_test();
        let s = note_rect_void(1, 2, w, h, &frame, rgb_nz_of(&frame)).expect("void must fire");
        note_rect_void_origin(&s, &frame, &frame, w, h, w, h, "guest_pages_frag");
        let body = std::fs::read_to_string(crate::observe::fail_log_path())
            .expect("reims-vgpu-fail.log readable");
        assert!(
            body.lines()
                .any(|l| l.contains("rect_void_origin origin=persistent_black")
                    && l.contains("src=guest_pages_frag")),
            "black retained band must read as persistent_black"
        );

        // Dedup: a second identical (grid, origin, src) call emits nothing new.
        let before = body.matches("rect_void_origin").count();
        note_rect_void_origin(&s, &frame, &frame, w, h, w, h, "guest_pages_frag");
        let after = std::fs::read_to_string(crate::observe::fail_log_path())
            .expect("reims-vgpu-fail.log readable")
            .matches("rect_void_origin")
            .count();
        assert_eq!(before, after, "identical origin combo must dedup");

        // Geometry mismatch (retain differs) is skipped, not misattributed.
        reset_for_test();
        let s = note_rect_void(1, 3, w, h, &frame, rgb_nz_of(&frame)).expect("void must fire");
        let before = std::fs::read_to_string(crate::observe::fail_log_path())
            .map(|b| b.matches("rect_void_origin").count())
            .unwrap_or(0);
        note_rect_void_origin(&s, &frame, &full, w + 4, h, w, h, "reuse_store");
        let after = std::fs::read_to_string(crate::observe::fail_log_path())
            .map(|b| b.matches("rect_void_origin").count())
            .unwrap_or(0);
        assert_eq!(before, after, "geometry-mismatched retain must be skipped");
    }

    #[test]
    fn damage_hole_fires_for_retained_rectangle_inside_connected_transition() {
        let _g = test_lock();
        let w = 320u32;
        let h = 180u32;
        let stride = w as usize * 4;
        let previous = vec![32u8; stride * h as usize];
        let mut complete = previous.clone();
        for y in 20..160usize {
            for x in 40..280usize {
                let o = y * stride + x * 4;
                complete[o..o + 4].copy_from_slice(&[96, 144, 224, 255]);
            }
        }

        reset_for_test();
        note_damage_hole(0x4a11, 1, w, h, &previous);
        note_damage_hole(0x4a11, 2, w, h, &complete);
        assert_eq!(
            counters().damage_hole,
            0,
            "a completely painted rectangle has no retained interior hole"
        );

        let mut incomplete = complete.clone();
        // Restore a large old-frame rectangle inside the otherwise connected
        // new-window transition. At 32x18 this is a 6x8-class coarse hole.
        for y in 50..130usize {
            for x in 100..160usize {
                let o = y * stride + x * 4;
                incomplete[o..o + 4].copy_from_slice(&previous[o..o + 4]);
            }
        }
        reset_for_test();
        note_damage_hole(0x4a11, 3, w, h, &previous);
        // A complete intermediate capture must not erase the pre-transition
        // baseline. Relative only to `complete`, the later old-color hole is a
        // small isolated change and the adjacent-frame implementation missed it.
        note_damage_hole(0x4a11, 4, w, h, &complete);
        note_damage_hole(0x4a11, 5, w, h, &incomplete);
        assert_eq!(counters().damage_hole, 1);
        // The restored interior shows the same color as the surrounding
        // background (`previous`), so it classifies as a genuine
        // stale-background suspect and stays fail-visible.
        assert_eq!(
            counters().damage_hole_bg,
            1,
            "a hole matching the surrounding desktop is a bg_match suspect"
        );
        let body = std::fs::read_to_string(crate::observe::fail_log_path())
            .expect("reims-vgpu-fail.log readable");
        assert!(body.lines().any(|line| {
            line.starts_with("THRASH damage_hole mid=18961 gen=5 320x180 ")
                && line.contains("hole_tiles=")
                && line.contains("bg_match=1")
        }));

        // A small cursor-like transition is below the connected-component gate.
        let mut cursor = previous.clone();
        for y in 80..90usize {
            for x in 150..160usize {
                let o = y * stride + x * 4;
                cursor[o..o + 4].copy_from_slice(&[255, 255, 255, 255]);
            }
        }
        reset_for_test();
        note_damage_hole(0x4a11, 6, w, h, &previous);
        note_damage_hole(0x4a11, 7, w, h, &cursor);
        assert_eq!(counters().damage_hole, 0);
    }

    /// Double-buffer alternation: presents interleave two mids; each keeps its
    /// own damage anchor, so a retained-rectangle hole on one buffer still
    /// fires (a single global anchor re-based on every mid switch and went
    /// blind on exactly this stream shape).
    #[test]
    fn damage_hole_fires_across_double_buffer_mid_alternation() {
        let _g = test_lock();
        let w = 320u32;
        let h = 180u32;
        let stride = w as usize * 4;
        let mid_a = 0x4a21u32;
        let mid_b = 0x4a22u32;
        let previous = vec![32u8; stride * h as usize];
        let mut complete = previous.clone();
        for y in 20..160usize {
            for x in 40..280usize {
                let o = y * stride + x * 4;
                complete[o..o + 4].copy_from_slice(&[96, 144, 224, 255]);
            }
        }
        let mut incomplete = complete.clone();
        for y in 50..130usize {
            for x in 100..160usize {
                let o = y * stride + x * 4;
                incomplete[o..o + 4].copy_from_slice(&previous[o..o + 4]);
            }
        }

        reset_for_test();
        // A-buffer anchors, B-buffer frames interleave every present.
        note_damage_hole(mid_a, 1, w, h, &previous);
        note_damage_hole(mid_b, 1, w, h, &previous);
        note_damage_hole(mid_a, 2, w, h, &complete);
        note_damage_hole(mid_b, 2, w, h, &complete);
        assert_eq!(counters().damage_hole, 0);
        note_damage_hole(mid_b, 3, w, h, &incomplete);
        assert_eq!(
            counters().damage_hole,
            1,
            "the B-buffer hole must fire despite interleaved A presents"
        );
    }

    /// A legitimately-static window body (distinct from the surrounding desktop)
    /// surrounded by a changed frame produces a damage hole, but it classifies
    /// as `bg_match=0` and must stay OUT of the curated fail-log while still
    /// counting toward the fail-visible `damage_hole` total.
    #[test]
    fn damage_hole_static_window_body_is_quiet_not_bg_match() {
        let _g = test_lock();
        let w = 320u32;
        let h = 180u32;
        let stride = w as usize * 4;
        let mid = 0x4a31u32;
        let paint = |buf: &mut [u8], x0: usize, x1: usize, y0: usize, y1: usize, c: [u8; 4]| {
            for y in y0..y1 {
                for x in x0..x1 {
                    let o = y * stride + x * 4;
                    buf[o..o + 4].copy_from_slice(&c);
                }
            }
        };
        // Anchor: gray desktop with an already-painted white window body.
        let mut prev = vec![32u8; stride * h as usize];
        paint(&mut prev, 80, 240, 40, 140, [250, 250, 250, 255]);
        // Current: the white body is UNCHANGED, but a thick blue frame around it
        // changes (window chrome / neighboring content animating).
        let mut cur = vec![32u8; stride * h as usize];
        paint(&mut cur, 20, 300, 10, 170, [96, 144, 224, 255]);
        paint(&mut cur, 80, 240, 40, 140, [250, 250, 250, 255]);

        reset_for_test();
        let mark = std::fs::read_to_string(crate::observe::fail_log_path())
            .map(|b| b.len())
            .unwrap_or(0);
        note_damage_hole(mid, 1, w, h, &prev);
        note_damage_hole(mid, 2, w, h, &cur);
        assert_eq!(
            counters().damage_hole,
            1,
            "the static white interior forms a hole"
        );
        assert_eq!(
            counters().damage_hole_bg,
            0,
            "a white body over gray desktop is not a stale-background match"
        );
        let body = std::fs::read_to_string(crate::observe::fail_log_path())
            .expect("reims-vgpu-fail.log readable");
        let appended = &body[mark.min(body.len())..];
        assert!(
            !appended.contains("THRASH damage_hole"),
            "a benign bg_match=0 hole must not reach the curated fail-log"
        );
    }

    /// A full-screen recomposite (the changed component fills the whole grid, so
    /// there is no periphery to sample) whose only hole is an ENCLOSED interior
    /// static rectangle is not the window-over-wallpaper bug: it must count
    /// toward `damage_hole` but classify quiet (`class=full_frame_recomposite`,
    /// not `damage_hole_bg`) and stay out of the curated fail-log.
    #[test]
    fn damage_hole_full_frame_recomposite_is_quiet() {
        let _g = test_lock();
        let w = 320u32;
        let h = 180u32;
        let stride = w as usize * 4;
        let mid = 0x4a41u32;
        // Anchor: uniform gray. Current: everything changes EXCEPT an interior
        // static box (grid x[14,20) y[8,12) = pixels x[140,200) y[80,120)),
        // enclosed on all four sides by changed tiles. The changed region still
        // reaches every grid edge, so the component bounds fill the whole 32x18
        // grid (no periphery to classify) while enclosing that hole.
        let prev = vec![32u8; stride * h as usize];
        let mut cur = vec![32u8; stride * h as usize];
        for y in 0..h as usize {
            for x in 0..w as usize {
                let in_hole = (80..120).contains(&y) && (140..200).contains(&x);
                if in_hole {
                    continue; // stays gray = unchanged
                }
                let o = y * stride + x * 4;
                // Position-dependent value so the changed region is one connected
                // full-frame component.
                let v = ((x + y) % 200 + 40) as u8;
                cur[o..o + 4].copy_from_slice(&[v, v, v, 255]);
            }
        }

        reset_for_test();
        let mark = std::fs::read_to_string(crate::observe::fail_log_path())
            .map(|b| b.len())
            .unwrap_or(0);
        note_damage_hole(mid, 1, w, h, &prev);
        note_damage_hole(mid, 2, w, h, &cur);
        // The interior static box is an enclosed hole inside a full-grid component.
        assert_eq!(counters().damage_hole, 1, "the interior box forms a hole");
        assert_eq!(
            counters().damage_hole_bg,
            0,
            "a full-frame recomposite is not a stale-background suspect"
        );
        let body = std::fs::read_to_string(crate::observe::fail_log_path())
            .expect("reims-vgpu-fail.log readable");
        let appended = &body[mark.min(body.len())..];
        assert!(
            !appended.contains("THRASH damage_hole"),
            "a full-frame recomposite hole must not reach the curated fail-log"
        );
    }

    /// Live false-positive regression: a scrolling window whose changed
    /// bounding box includes the correctly-static wallpaper strip just below the
    /// window (the bottom bbox tile-row) must NOT register a damage hole. That
    /// strip touches the component's bounding-box edge — it is the window not
    /// filling its bbox edge, not an enclosed stale-background hole — yet before
    /// the enclosure requirement it fired `damage_hole_bg=1` (loud) on ordinary
    /// Safari scrolling because the strip color matched the surrounding wallpaper.
    #[test]
    fn damage_hole_bbox_edge_wallpaper_strip_is_ignored() {
        let _g = test_lock();
        let w = 320u32;
        let h = 180u32;
        let stride = w as usize * 4;
        let mid = 0x4a51u32;
        // Wallpaper everywhere (a distinctive blue). The window occupies the
        // upper-left region; below it the wallpaper stays static.
        let wallpaper = [187u8, 77, 54, 255]; // BGRA
        let prev: Vec<u8> = wallpaper
            .iter()
            .copied()
            .cycle()
            .take(stride * h as usize)
            .collect();
        // Window bbox: pixels x[60,260) y[0,130) — a non-full-grid component that
        // changes every frame. y[130,180) stays wallpaper (the static strip below
        // the window). The bbox bottom tile-row (y grid 12) is thus unchanged
        // wallpaper touching the component's bottom bbox edge.
        let mut cur = prev.clone();
        for y in 0..130usize {
            for x in 60..260usize {
                let o = y * stride + x * 4;
                let v = ((x + y) % 180 + 50) as u8;
                cur[o..o + 4].copy_from_slice(&[v, v, v, 255]);
            }
        }

        reset_for_test();
        note_damage_hole(mid, 1, w, h, &prev);
        note_damage_hole(mid, 2, w, h, &cur);
        assert_eq!(
            counters().damage_hole,
            0,
            "a static wallpaper strip at the bbox edge is not an enclosed hole"
        );
        assert_eq!(counters().damage_hole_bg, 0, "and never a bg_match suspect");
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

    #[test]
    fn dense_retention_gap_fires_above_margin_and_dedups() {
        let _g = test_lock();
        reset_for_test();
        let (w, h) = (1920u32, 1080u32);

        // A 1-frame lag is healthy a/b alternation (below the margin) → quiet.
        assert!(!note_dense_retention_gap(4, 1, 6, 7, w, h));
        assert_eq!(counters().dense_retention_gap, 0);

        // Lag == RETENTION_GAP_MARGIN (missed 4 full frames + the seed): fires once.
        assert!(note_dense_retention_gap(4, 1, 3, 7, w, h));
        assert_eq!(counters().dense_retention_gap, 1);
        // Same widened gap (same peer_seq) → deduped, no per-present re-fire.
        assert!(!note_dense_retention_gap(4, 1, 3, 7, w, h));
        assert_eq!(counters().dense_retention_gap, 1);

        // A further-widened gap (a newer peer full frame the buffer still missed)
        // → a new episode fires.
        assert!(note_dense_retention_gap(4, 1, 3, 9, w, h));
        assert_eq!(counters().dense_retention_gap, 2);

        // A different presented buffer lagging is its own independent episode.
        assert!(note_dense_retention_gap(5, 1, 2, 9, w, h));
        assert_eq!(counters().dense_retention_gap, 3);
    }

    #[test]
    fn tile_composite_result_dedups_and_rearms_on_divergence_clear() {
        let _g = test_lock();
        reset_for_test();
        let (w, h) = (1920u32, 1080u32);

        assert!(note_tile_divergence(4, 1, 6, [1, 0, 3, 2], w, h));
        assert!(note_tile_composite_result(
            TileComposite::Applied,
            2,
            2,
            w,
            h
        ));
        assert_eq!(counters().tile_composite_applied, 1);
        assert_eq!(counters().tile_composite_skipped, 0);
        assert!(
            !note_tile_composite_result(TileComposite::Applied, 2, 2, w, h),
            "stable applied result should not re-log every present"
        );
        assert_eq!(counters().tile_composite_applied, 1);

        assert!(note_tile_composite_result(
            TileComposite::PeerNotReady,
            2,
            0,
            w,
            h
        ));
        assert_eq!(counters().tile_composite_applied, 1);
        assert_eq!(counters().tile_composite_skipped, 1);
        assert!(
            !note_tile_composite_result(TileComposite::PeerNotReady, 2, 0, w, h),
            "stable skip reason should be deduped"
        );
        assert_eq!(counters().tile_composite_skipped, 1);

        assert!(
            !note_tile_divergence(4, 1, 0, [0; 4], w, h),
            "clearing divergence only rearms the result dedup"
        );
        assert!(note_tile_composite_result(
            TileComposite::PeerNotReady,
            2,
            0,
            w,
            h
        ));
        assert_eq!(counters().tile_composite_skipped, 2);
    }

    /// The tile-composite never-displayed-peer proxy logs once per
    /// `(presented_mid, peer_mid)` pair (a persistent leak must not spam per
    /// present), and a distinct pair re-fires.
    #[test]
    fn tile_composite_unpresented_peer_dedups_per_pair() {
        let _g = test_lock();
        reset_for_test();
        let (w, h) = (1920u32, 1080u32);

        assert!(
            note_tile_composite_unpresented_peer(6, 1, 40, w, h),
            "first sighting of a never-displayed peer logs"
        );
        assert!(
            !note_tile_composite_unpresented_peer(6, 1, 40, w, h),
            "same pair on a later present is deduped"
        );
        assert!(
            !note_tile_composite_unpresented_peer(6, 1, 12, w, h),
            "dedup is keyed on the pair, not the rect count"
        );
        assert!(
            note_tile_composite_unpresented_peer(6, 9, 5, w, h),
            "a distinct never-displayed peer re-fires"
        );
    }

    /// `reason=` names a refusal, so a *success* must not carry one.
    ///
    /// This line used to render `reason=applied`, which is the shape that teaches
    /// a reader to ignore the field — and the census survey that opened this unit
    /// found 190 lines of the same shape on one boot. `Emit::refusal` returning
    /// `Option` is what makes it unrepresentable now rather than merely
    /// discouraged.
    #[test]
    fn only_a_skipped_tile_composite_carries_a_reason() {
        let _g = test_lock();
        reset_for_test();
        let path = observe::fail_log_path();
        let read = || std::fs::read_to_string(path).unwrap_or_default();
        let (w, h) = (1913u32, 1077u32);

        let before = read().lines().count();
        let lines_since = |before: usize| -> Vec<String> {
            read()
                .lines()
                .skip(before)
                .filter(|l| l.contains("tile_composite "))
                .map(str::to_string)
                .collect()
        };

        assert!(note_tile_composite_result(
            TileComposite::Applied,
            2,
            2,
            w,
            h
        ));
        let applied = lines_since(before);
        assert_eq!(applied.len(), 1, "expected one line, got {applied:?}");
        assert!(
            applied[0].contains("status=applied") && !applied[0].contains("reason="),
            "a success must not name a reason: {}",
            applied[0]
        );

        let before = read().lines().count();
        assert!(note_tile_composite_result(
            TileComposite::PeerGeomMismatch,
            2,
            0,
            w,
            h
        ));
        let skipped = lines_since(before);
        assert_eq!(skipped.len(), 1, "expected one line, got {skipped:?}");
        assert!(
            skipped[0].contains("reason=tile_peer_geom_mismatch")
                && skipped[0].contains("status=skipped"),
            "a skip must name which precondition failed: {}",
            skipped[0]
        );
        reset_for_test();
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

    /// sparse_dock: glass-only strip fires; icon-structured strip stays quiet.
    /// Also left-only partial dock (live Safari residual) and early-boot gate.
    #[test]
    fn sparse_dock_glass_vs_icons() {
        let _g = test_lock();
        let w = 800u32;
        let h = 600u32;
        let stride = (w as usize) * 4;
        let mut orange = vec![0u8; stride * h as usize];
        for px in orange.chunks_exact_mut(4) {
            px[0] = 0;
            px[1] = 80;
            px[2] = 250;
            px[3] = 255;
        }
        // Full-frame nz high enough to pass the settled-desktop gate.
        let orange_nz = orange.iter().filter(|&&b| b != 0).count();
        reset_for_test();
        note_dock_strip(3, 1, w, h, &orange, orange_nz);
        assert_eq!(
            counters().sparse_dock,
            1,
            "uniform dock strip must count as sparse_dock"
        );

        // Icon row: high-contrast dock columns vs wallpaper above.
        let mut icons = orange.clone();
        let strip_h = DOCK_STRIP_H.min(h / 8);
        let dock_y0 = (h - strip_h) as usize;
        for y in dock_y0..h as usize {
            for x in 0..w as usize {
                if (x / 32) % 2 == 0 {
                    let o = y * stride + x * 4;
                    icons[o] = 255;
                    icons[o + 1] = 128;
                    icons[o + 2] = 0;
                    icons[o + 3] = 255;
                }
            }
        }
        let icons_nz = icons.iter().filter(|&&b| b != 0).count();
        reset_for_test();
        note_dock_strip(3, 2, w, h, &icons, icons_nz);
        assert_eq!(
            counters().sparse_dock,
            0,
            "icon-structured dock must not fire sparse_dock"
        );

        // Left-half icons only (live Safari residual right glass).
        let mut partial = orange.clone();
        for y in dock_y0..h as usize {
            for x in 0..(w as usize / 2) {
                if (x / 32) % 2 == 0 {
                    let o = y * stride + x * 4;
                    partial[o] = 255;
                    partial[o + 1] = 128;
                    partial[o + 2] = 0;
                    partial[o + 3] = 255;
                }
            }
        }
        let partial_nz = partial.iter().filter(|&&b| b != 0).count();
        reset_for_test();
        note_dock_strip(3, 3, w, h, &partial, partial_nz);
        assert_eq!(
            counters().sparse_dock,
            1,
            "left-only dock icons must fire sparse_dock (partial)"
        );

        // Incomplete dual-mid (~0.32 nz) must not fire dock proxy.
        reset_for_test();
        let incomplete_nz = total_bytes_frac_nz(w, h, 0.32);
        note_dock_strip(3, 4, w, h, &orange, incomplete_nz);
        assert_eq!(
            counters().sparse_dock,
            0,
            "incomplete dual-mid frame must skip sparse_dock"
        );
    }

    #[test]
    fn menu_strip_geom_classifies_thin_chrome() {
        assert!(is_menu_strip_geom(1920, 24));
        assert!(is_menu_strip_geom(1877, 24));
        assert!(is_menu_strip_geom(1280, 48));
        assert!(!is_menu_strip_geom(1920, 1080));
        assert!(!is_menu_strip_geom(43, 24)); // too narrow (partial corner)
        assert!(!is_menu_strip_geom(1920, 0));
        assert!(!is_menu_strip_geom(100, 24));
    }

    #[test]
    fn menu_strip_stats_gray_vs_rainbow() {
        let _g = test_lock();
        let w = 1280u32;
        let h = 720u32;
        let stride = (w as usize) * 4;
        // Healthy gray menu labels in top 24 rows.
        let mut gray = vec![0u8; stride * h as usize];
        for y in 0..24usize {
            for x in 0..w as usize {
                // Sparse gray glyphs every 8th column.
                if x % 8 == 0 {
                    let o = y * stride + x * 4;
                    gray[o] = 180;
                    gray[o + 1] = 180;
                    gray[o + 2] = 180;
                    gray[o + 3] = 255;
                }
            }
        }
        let gs = menu_strip_stats_bgra(&gray, w, h).expect("gray stats");
        assert!(gs.rgb_nz > 0);
        assert_eq!(gs.chroma_hi, 0, "gray labels must not count as high chroma");
        assert!(gs.gray > 0);

        // Rainbow: high-chroma static across the band.
        let mut rain = vec![0u8; stride * h as usize];
        for y in 0..24usize {
            for x in 0..w as usize {
                let o = y * stride + x * 4;
                rain[o] = ((x * 17 + y * 3) % 256) as u8;
                rain[o + 1] = ((x * 31 + y * 5) % 256) as u8;
                rain[o + 2] = ((x * 7 + y * 11) % 256) as u8;
                rain[o + 3] = 255;
            }
        }
        let rs = menu_strip_stats_bgra(&rain, w, h).expect("rain stats");
        assert!(rs.rgb_nz > rs.total / 2);
        assert!(
            rs.chroma_hi as f64 / rs.total as f64 > RAINBOW_MENU_CHROMA_FRAC,
            "synthetic rainbow must exceed chroma threshold"
        );
        assert!(
            rs.incoherent as f64 / rs.total as f64 > RAINBOW_MENU_INCOHERENT_FRAC,
            "synthetic rainbow noise must be spatially incoherent"
        );

        // Smooth vibrant wallpaper seen through a translucent menu bar: HIGH
        // chroma (colorful) but spatially COHERENT (a gradient, not noise). This
        // is the healthy live class that chroma-only falsely flagged on ~93% of
        // presents (measured chroma_frac≈0.88, incoherent_frac≈0.04). It must
        // NOT fire rainbow_menu.
        let mut wallpaper = vec![0u8; stride * h as usize];
        for y in 0..24usize {
            for x in 0..w as usize {
                let o = y * stride + x * 4;
                let t = x as f64 / w as f64;
                // BGRA vibrant orange gradient (dark red → bright orange).
                wallpaper[o] = (10.0 + 30.0 * t) as u8; // B
                wallpaper[o + 1] = (20.0 + 120.0 * t) as u8; // G
                wallpaper[o + 2] = (40.0 + 200.0 * t) as u8; // R
                wallpaper[o + 3] = 255;
            }
        }
        let ws = menu_strip_stats_bgra(&wallpaper, w, h).expect("wallpaper stats");
        assert!(
            ws.chroma_hi as f64 / ws.total as f64 > RAINBOW_MENU_CHROMA_FRAC,
            "vibrant wallpaper is genuinely high-chroma"
        );
        assert!(
            (ws.incoherent as f64 / ws.total as f64) < RAINBOW_MENU_INCOHERENT_FRAC,
            "smooth wallpaper gradient must be spatially coherent"
        );

        reset_for_test();
        note_menu_strip(1, 10, w, h, &gray);
        assert_eq!(
            counters().rainbow_menu,
            0,
            "gray menu must not fire rainbow_menu"
        );
        note_menu_strip(1, 12, w, h, &wallpaper);
        assert_eq!(
            counters().rainbow_menu,
            0,
            "colorful-but-coherent translucent menu must not fire rainbow_menu"
        );
        note_menu_strip(1, 11, w, h, &rain);
        assert_eq!(
            counters().rainbow_menu,
            1,
            "high-chroma AND incoherent top band must fire rainbow_menu"
        );
    }

    fn total_bytes_frac_nz(w: u32, h: u32, frac: f64) -> usize {
        ((w as f64) * (h as f64) * 4.0 * frac) as usize
    }

    /// Regression guard for `edge_energy_bgra`, the horizontal-luma-gradient
    /// proxy the residue / dock-strip gates read. Its contract is load-bearing
    /// for those correctness gates: a uniform (clean) frame must read **zero**
    /// energy so the residue proxy stays quiet on a cleared desktop, structured
    /// content must read **nonzero** and grow with contrast so residue/streaks
    /// actually register, and degenerate/short input must read zero rather than
    /// panic or emit garbage. A silent break here would blind the residue gate.
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

    /// Regression guard for `damage_changed_tiles`, the tile-diff counter the
    /// damage-hole residue proxy reads. Its contract is load-bearing: a
    /// geometry/sample-count mismatch must bail (None) instead of comparing
    /// misaligned grids, identical frames must count zero changed tiles (so the
    /// proxy stays quiet on a static desktop), and a tile counts as changed
    /// only once its differing-sample fraction crosses DAMAGE_TILE_CHANGED_FRAC
    /// — the exact threshold that separates real damage from sub-tile noise.
    #[test]
    fn damage_changed_tiles_threshold_and_geometry_guard() {
        let per_tile = DAMAGE_SAMPLES_AXIS * DAMAGE_SAMPLES_AXIS; // 64
                                                                  // 0.10 * 64 = 6.4 -> a tile needs >= 7 differing samples to count.
        let threshold = (DAMAGE_TILE_CHANGED_FRAC * per_tile as f64).ceil() as usize;
        assert_eq!(threshold, 7, "guard assumes 7/64 changed samples to count");

        // Two-tile frame: 128 samples, all black.
        let frame = |samples: Vec<[u8; 3]>| DamageFrame {
            width: 16,
            height: 8,
            samples,
        };
        let base: Vec<[u8; 3]> = vec![[0, 0, 0]; 2 * per_tile];
        let prev = frame(base.clone());

        // Geometry / sample-count mismatch -> None (never a misaligned compare).
        assert_eq!(
            damage_changed_tiles(
                &prev,
                &DamageFrame {
                    width: 17,
                    ..frame(base.clone())
                }
            ),
            None,
        );
        assert_eq!(
            damage_changed_tiles(&prev, &frame(vec![[0, 0, 0]; per_tile])),
            None,
            "differing sample count must bail",
        );

        // Identical frames -> zero changed tiles.
        assert_eq!(damage_changed_tiles(&prev, &frame(base.clone())), Some(0));

        // Helper: flip the first `n` samples of tile `t` to a distinct value.
        let with_changes = |t: usize, n: usize| {
            let mut s = base.clone();
            for i in 0..n {
                s[t * per_tile + i] = [255, 255, 255];
            }
            frame(s)
        };

        // Just below threshold in a single tile -> not counted.
        assert_eq!(
            damage_changed_tiles(&prev, &with_changes(0, threshold - 1)),
            Some(0),
            "6/64 changed is sub-threshold noise",
        );
        // Exactly at threshold -> that tile counts.
        assert_eq!(
            damage_changed_tiles(&prev, &with_changes(0, threshold)),
            Some(1),
        );
        // Threshold crossed in tile 1 only -> counted independently of tile 0.
        let mut both = base.clone();
        for pixel in both.iter_mut().skip(per_tile).take(threshold) {
            *pixel = [255, 255, 255]; // tile 1
        }
        assert_eq!(damage_changed_tiles(&prev, &frame(both.clone())), Some(1));
        // Both tiles over threshold -> both counted.
        for pixel in both.iter_mut().take(threshold) {
            *pixel = [255, 255, 255]; // tile 0
        }
        assert_eq!(damage_changed_tiles(&prev, &frame(both)), Some(2));
    }

    #[test]
    fn classify_damage_hole_bg_match_vs_window_body() {
        let per_tile = DAMAGE_SAMPLES_AXIS * DAMAGE_SAMPLES_AXIS;
        let grid_tiles = DAMAGE_GRID_W * DAMAGE_GRID_H;
        let wallpaper = [200u8, 150, 100];
        // Component bounds and enclosed hole (grid coords), mirroring a real
        // window-open event: bounds 25x14+1+0, hole 20x11+6+2.
        let s = DamageHoleStats {
            changed_tiles: 0,
            component_tiles: 0,
            bounds_x: 1,
            bounds_y: 0,
            bounds_w: 25,
            bounds_h: 14,
            hole_tiles: 20 * 11,
            hole_x: 6,
            hole_y: 2,
            hole_w: 20,
            hole_h: 11,
        };
        let in_hole = |gx: usize, gy: usize| {
            gx >= s.hole_x && gx < s.hole_x + s.hole_w && gy >= s.hole_y && gy < s.hole_y + s.hole_h
        };
        let build = |hole_color: [u8; 3]| {
            let mut samples = vec![[0u8; 3]; grid_tiles * per_tile];
            for gy in 0..DAMAGE_GRID_H {
                for gx in 0..DAMAGE_GRID_W {
                    // Everything defaults to wallpaper; the hole gets its own
                    // color. Interior-of-bounds-but-not-hole tiles are window
                    // chrome — irrelevant to the classifier, keep wallpaper so
                    // periph (strictly-outside-bounds) is the only desktop feed.
                    let color = if in_hole(gx, gy) {
                        hole_color
                    } else {
                        wallpaper
                    };
                    let tile = gy * DAMAGE_GRID_W + gx;
                    for i in 0..per_tile {
                        samples[tile * per_tile + i] = color;
                    }
                }
            }
            DamageFrame {
                width: 1920,
                height: 1080,
                samples,
            }
        };

        // Genuine stale-background: the hole shows the same wallpaper as the
        // surrounding desktop -> bg_match, dist 0, flat.
        let stale = build(wallpaper);
        let c = classify_damage_hole(&stale, &s).expect("classifies");
        assert!(c.bg_match, "wallpaper-filled hole must match background");
        assert_eq!(c.dist, 0.0);
        assert_eq!(c.hole_spread, 0);
        assert_eq!(c.hole_rgb, wallpaper);
        assert_eq!(c.periph_rgb, wallpaper);
        assert!(c.periph_tiles > 0);

        // Benign static window body: a distinct interior color far from the
        // wallpaper -> not a background match.
        let body = build([255, 255, 255]);
        let c = classify_damage_hole(&body, &s).expect("classifies");
        assert!(
            !c.bg_match,
            "white window body must not match orange wallpaper"
        );
        assert!(c.dist > DAMAGE_BG_MATCH_TOL);
        assert_eq!(c.hole_rgb, [255, 255, 255]);

        // A near-threshold interior (within tolerance) still reads as a match:
        // the classifier is a distance test, not an exact compare.
        let near = build([
            wallpaper[0] + 10,
            wallpaper[1].wrapping_sub(10),
            wallpaper[2] + 5,
        ]);
        let c = classify_damage_hole(&near, &s).expect("classifies");
        assert!(c.bg_match, "within-tolerance interior counts as background");
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
