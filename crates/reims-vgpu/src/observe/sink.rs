//! Temporary bring-up log for metal/scanout (research). Append-only `/tmp/reims-vgpu-draw.log`.
//!
//! Verbose lines: `REIMS_VGPU_DRAW_LOG=1` only — full-frame logging otherwise stalls the
//! guest compositor. Failures always append (lightweight, fail-visible).
//!
//! ## Offline offline-analysis prefixes (`/tmp/reims-vgpu-fail.log`, always-on)
//!
//! | Prefix | Meaning |
//! | --- | --- |
//! | `OFF present_op6/7/8` | Display present packet (channel, surface/mapping id, stamp) |
//! | `OFF present_black` | max_rgb==0 after capture (console will stay black) |
//! | `OFF present_paint` | HostAction paint / Unchanged |
//! | `OFF host_cache_store` | Discrete-GPU host surface cache write |
//! | `OFF host_cache_evict` | Cache drop (unmap/delete) |
//! | `OFF m2v_store` | metal2vulkan Store to type-11/type-4 mid (incl. is_front) |
//! | `OFF m2v_store_gva` | metal2vulkan Store to type-2/3 GVA |
//! | `OFF m2v_load_seed` | Load seed path (host_cache vs missing) |
//! | `OFF load_seed_black` | Deduplicated zero-RGB Load seed preserved by protocol provenance |
//! | `OFF linear_sample` | Display-sized type-2/3 sample provenance + content census |
//! | `OFF sampled_branch_census` | Cumulative per-branch sampled-resolution counts:bytes, every 256 |
//! | `OFF sample_alpha_mask` | Deduplicated zero-RGB/nonzero-alpha sample census; alpha is preserved |
//! | `linear_sample_miss` | Display-sized type-2/3 sample failed, with descriptor identity |
//! | `OFF linear_coverage_gap` | Typed stage-in/shader-evaluated coverage check rejected full-display ownership |
//! | `import_content` | Resident-to-guest Store census; display rows include exact changed/R↔B-swapped pixel counts |
//! | `linux_m2v_resources` | Per-draw resource census; `fixed_gap=[...]` names decoded fixed state absent from the Vulkan request |
//! | `linux_m2v_timing` | always-on stage µs: load/m2v/setup/engine/composite + total |
//! | `OFF display_clear` | Clear-only stream Store into a display-sized mid |
//! | `OFF rt_resolve` | Color RT lookup (type-4/5/11 → mapping_id) display-sized |
//! | `OFF front_wb` | note_front_buffer_writeback latch / post-boundary skip |
//! | `OFF blit` | Product blit path enter/result (buffer↔texture) |
//!
//! **rgb_nz / max_rgb** on OFF lines count **pixels with max(B,G,R) > 0** (BGRA).
//! Byte-wise `nonzero_stats` still counts alpha=255 as nonzero — do not use that
//! alone to claim the screen is not black.

use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(test)]
use std::{
    fs::{File, OpenOptions},
    io::Write,
    sync::Mutex,
};

static ENABLED: AtomicBool = AtomicBool::new(false);
static INIT: AtomicBool = AtomicBool::new(false);
#[cfg(test)]
static FAIL_FILE: Mutex<Option<File>> = Mutex::new(None);
#[cfg(test)]
static DRAW_FILE: Mutex<Option<File>> = Mutex::new(None);

/// Which always-on sink a line targets.
#[derive(Clone, Copy)]
enum Sink {
    Fail,
    Draw,
}

pub(crate) fn enabled() -> bool {
    if !INIT.swap(true, Ordering::Relaxed) {
        let on = std::env::var_os("REIMS_VGPU_DRAW_LOG")
            .map(|v| v == "1")
            .unwrap_or(false);
        ENABLED.store(on, Ordering::Relaxed);
    }
    ENABLED.load(Ordering::Relaxed)
}

/// Whether verbose draw logging (`REIMS_VGPU_DRAW_LOG=1`) is active. Lets always-on
/// paths skip building expensive *diagnostic-only* detail (e.g. per-peer
/// full-frame rescans for a log field) on a normal boot without losing the
/// always-on line itself.
pub(crate) fn draw_log_enabled() -> bool {
    enabled()
}

/// Milliseconds since the first log line of this process. Appended as a
/// trailing `t=<ms>` field so cross-boot phase timing (first present, desktop
/// settle, tranche bursts) is measurable from the logs alone. Trailing — not a
/// prefix — so `awk '{print $1}'` line-class censuses keep working.
///
/// `pub(crate)` so always-on rate proxies (e.g. display-signal cadence) can
/// window their counters on the same process-monotonic clock that stamps every
/// line — no second time base to reconcile against `t=`.
pub(crate) fn elapsed_ms() -> u128 {
    static T0: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    T0.get_or_init(std::time::Instant::now)
        .elapsed()
        .as_millis()
}

/// Sink paths. Test runs write per-process files instead of the product
/// `/tmp/reims-vgpu-fail.log`: `cargo test` runs on the same machine as live product
/// boots, and both appending to one shared file interleaves test fixture
/// lines (synthetic device resets, malformed-packet fail_events,
/// deferred_flush_lost probes) into live A/B evidence — indistinguishable
/// from real device failures when reading the log offline. Unit-test builds
/// isolate via `cfg(test)`; integration-test binaries (no `cfg(test)` on the
/// lib) must call [`redirect_logs_for_tests`] before the first log line.
static FAIL_PATH: std::sync::OnceLock<String> = std::sync::OnceLock::new();
static DRAW_PATH: std::sync::OnceLock<String> = std::sync::OnceLock::new();

fn test_path(kind: &str) -> String {
    format!("/tmp/reims-vgpu-{kind}-test-{}.log", std::process::id())
}

pub(crate) fn fail_log_path() -> &'static str {
    #[cfg(test)]
    return FAIL_PATH.get_or_init(|| test_path("fail"));
    #[cfg(not(test))]
    FAIL_PATH.get_or_init(|| "/tmp/reims-vgpu-fail.log".to_string())
}

pub(crate) fn draw_log_path() -> &'static str {
    #[cfg(test)]
    return DRAW_PATH.get_or_init(|| test_path("draw"));
    #[cfg(not(test))]
    DRAW_PATH.get_or_init(|| "/tmp/reims-vgpu-draw.log".to_string())
}

/// Test-harness support: point the always-on sinks at per-process files so a
/// test run never contaminates a concurrent live boot's logs. For integration
/// test binaries, where `cfg(test)` does not apply to the lib; call once
/// before anything logs. No effect on a sink that already resolved its path.
pub fn redirect_logs_for_tests() {
    let _ = FAIL_PATH.set(test_path("fail"));
    let _ = DRAW_PATH.set(test_path("draw"));
}

/// Synchronous single-line append (unit-test builds only). Worker + MMIO proxy
/// lines may arrive concurrently; keep each record on one physical line so
/// failure evidence never merges into another event.
#[cfg(test)]
fn append_sync(file: &Mutex<Option<File>>, path: &str, msg: &str, t: u128) {
    let mut file = file.lock().unwrap_or_else(|e| e.into_inner());
    if file.is_none() {
        *file = OpenOptions::new().create(true).append(true).open(path).ok();
    }
    let Some(f) = file.as_mut() else {
        return;
    };
    if writeln!(f, "{msg} t={t}").is_err() {
        // A later record gets one fresh open attempt after a failed write.
        *file = None;
    }
}

/// Emit one always-on line to `sink`, timestamped with the process-monotonic
/// clock at the call site (so `t=` reflects when the event happened, not when
/// the background writer drains it).
///
/// Product builds hand the formatted line to a background writer thread
/// ([`writer`]) so the doorbell / worker vCPU never pays a `write(2)` syscall
/// or contends the file lock — full-frame GPU boots emit ~200k lines/boot and
/// the synchronous path serialized them on the guest's critical path. Unit-test
/// builds stay synchronous: many tests write a line then immediately
/// `read_to_string` the sink and assert on it.
fn emit(sink: Sink, msg: &str) {
    let t = elapsed_ms();
    #[cfg(test)]
    {
        if matches!(sink, Sink::Fail) {
            if let Some(buf) = CAPTURED.lock().unwrap_or_else(|p| p.into_inner()).as_mut() {
                buf.push(msg.to_string());
            }
        }
        let (file, path) = match sink {
            Sink::Fail => (&FAIL_FILE, fail_log_path()),
            Sink::Draw => (&DRAW_FILE, draw_log_path()),
        };
        append_sync(file, path, msg, t);
    }
    #[cfg(not(test))]
    writer::enqueue(sink, format!("{msg} t={t}"));
}

/// Flood self-detector (regression guard). A per-event line wrongly routed to
/// the always-on sink (the `type5_view_zc`/`InvalidateResources` class that
/// buried the curated fail view under ~130k lines/boot) fires per bind/op — far
/// above any legitimate always-on rate. This watches the **always-on** stream in
/// the background writer thread (zero producer cost) and emits ONE
/// `log_flood_detected` line per window per runaway prefix, so a regression that
/// reintroduces a flood is named on the very boot it lands instead of silently
/// drowning real failures. Legitimate always-on lines are self-clocked windowed
/// summaries (`teardown_churn`, `present_import`) well under the threshold.
const FLOOD_WINDOW_MS: u128 = 1000;
const FLOOD_THRESHOLD_PER_WINDOW: u64 = 1000;

/// The flood-accounting key for an always-on line: its slug — the first
/// whitespace token, skipping a leading `OFF ` marker. Groups a runaway line by
/// kind (`type5_view_zc`, `map_family`, …) so the warning names the culprit.
fn flood_key(line: &str) -> &str {
    let slug = line.strip_prefix("OFF ").unwrap_or(line);
    slug.split(' ').next().unwrap_or(slug)
}

/// Windowed per-prefix counter for the always-on stream. Pure + always compiled
/// so the threshold/keying is unit-tested without a background thread.
struct FloodWindow {
    counts: std::collections::HashMap<String, u64>,
    window_start_ms: u128,
}

impl FloodWindow {
    fn new(now: u128) -> Self {
        Self {
            counts: std::collections::HashMap::new(),
            window_start_ms: now,
        }
    }

    /// Record one always-on line. When the ~1 s window closes, returns the
    /// prefixes that exceeded the flood threshold (sorted desc by count for a
    /// stable warning order) and opens a fresh window; otherwise returns empty.
    fn note(&mut self, line: &str, now: u128) -> Vec<(String, u64)> {
        *self.counts.entry(flood_key(line).to_string()).or_insert(0) += 1;
        if now.saturating_sub(self.window_start_ms) < FLOOD_WINDOW_MS {
            return Vec::new();
        }
        let mut flooders: Vec<(String, u64)> = self
            .counts
            .drain()
            .filter(|(_, c)| *c >= FLOOD_THRESHOLD_PER_WINDOW)
            .collect();
        flooders.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        self.window_start_ms = now;
        flooders
    }
}

/// Background log writer (product builds). A single thread owns both sink files
/// behind buffered writers; producers only push a formatted line onto an mpsc
/// channel. The thread batch-drains (block on one, then greedily take the rest)
/// and flushes after each batch, so failure visibility trails real time by at
/// most one drain cycle while the hot path stays syscall-free.
#[cfg(not(test))]
mod writer {
    use super::{draw_log_path, fail_log_path, Sink};
    use std::io::{BufWriter, Write};
    use std::sync::mpsc::{Receiver, Sender};
    use std::sync::OnceLock;

    enum Msg {
        Fail(String),
        Draw(String),
    }

    // `Sender<T>: Sync` (std, since 1.72), so producers share one sender with no
    // lock — the hot path is a lock-free channel send.
    static SENDER: OnceLock<Sender<Msg>> = OnceLock::new();

    fn sender() -> &'static Sender<Msg> {
        SENDER.get_or_init(|| {
            let (tx, rx) = std::sync::mpsc::channel::<Msg>();
            // Resolve sink paths on the spawning thread (honors any prior
            // `redirect_logs_for_tests`); the writer owns the file handles.
            let fail_path = fail_log_path().to_string();
            let draw_path = draw_log_path().to_string();
            let _ = std::thread::Builder::new()
                .name("reims-vgpu-drawlog".to_string())
                .spawn(move || writer_loop(rx, fail_path, draw_path));
            tx
        })
    }

    fn open(path: &str) -> Option<BufWriter<std::fs::File>> {
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .ok()
            .map(BufWriter::new)
    }

    fn writer_loop(rx: Receiver<Msg>, fail_path: String, draw_path: String) {
        let mut fail = open(&fail_path);
        let mut draw = open(&draw_path);
        let mut flood = super::FloodWindow::new(super::elapsed_ms());
        // Block for the next line, then greedily drain everything already
        // queued before a single flush — one syscall amortizes a whole burst.
        while let Ok(first) = rx.recv() {
            write_watched(&mut fail, &mut draw, &mut flood, first);
            while let Ok(m) = rx.try_recv() {
                write_watched(&mut fail, &mut draw, &mut flood, m);
            }
            if let Some(w) = fail.as_mut() {
                let _ = w.flush();
            }
            if let Some(w) = draw.as_mut() {
                let _ = w.flush();
            }
        }
    }

    /// Write one line, and for the always-on (Fail) sink feed the flood
    /// self-detector — a runaway prefix gets one named `log_flood_detected`
    /// warning per window, written straight to the fail file (not re-queued, so
    /// it never self-counts). All in the writer thread: no producer-side cost.
    fn write_watched(
        fail: &mut Option<BufWriter<std::fs::File>>,
        draw: &mut Option<BufWriter<std::fs::File>>,
        flood: &mut super::FloodWindow,
        m: Msg,
    ) {
        if let Msg::Fail(s) = &m {
            let flooders = flood.note(s, super::elapsed_ms());
            if let Some(w) = fail.as_mut() {
                for (prefix, count) in flooders {
                    let _ = writeln!(
                        w,
                        "log_flood_detected prefix={prefix} count={count} window_ms={} threshold={} t={}",
                        super::FLOOD_WINDOW_MS,
                        super::FLOOD_THRESHOLD_PER_WINDOW,
                        super::elapsed_ms()
                    );
                }
            }
        }
        write_msg(fail, draw, m);
    }

    fn write_msg(
        fail: &mut Option<BufWriter<std::fs::File>>,
        draw: &mut Option<BufWriter<std::fs::File>>,
        m: Msg,
    ) {
        let (w, line) = match m {
            Msg::Fail(s) => (fail.as_mut(), s),
            Msg::Draw(s) => (draw.as_mut(), s),
        };
        if let Some(w) = w {
            let _ = w.write_all(line.as_bytes());
            let _ = w.write_all(b"\n");
        }
    }

    /// Push one already-timestamped line to the background writer. Lock-free and
    /// never blocks on I/O — a bare channel send.
    pub(super) fn enqueue(sink: Sink, line: String) {
        let msg = match sink {
            Sink::Fail => Msg::Fail(line),
            Sink::Draw => Msg::Draw(line),
        };
        let _ = sender().send(msg);
    }
}

/// Test-only in-memory copy of the always-on stream, armed by [`FailCapture`].
#[cfg(test)]
static CAPTURED: std::sync::Mutex<Option<Vec<String>>> = std::sync::Mutex::new(None);

/// Records every always-on ([`fail`] / [`off`]) line emitted while it is alive.
///
/// The always-on stream is the project's primary evidence, and until this
/// existed nothing could assert on it: a probe could name a field after the
/// quantity its check compared while printing a different one, and no test
/// could tell. That is not hypothetical — the compute flush's
/// `map_generation_drift` printed a *content* generation in a field called
/// `gen`, next to the `map_generation` it had actually compared, and the
/// mismatch was read off a live boot as a generation that had gone backwards.
///
/// Relies on the crate's serial test convention (`--test-threads=1`); a second
/// capture armed concurrently would see the other test's lines.
#[cfg(test)]
pub(crate) struct FailCapture;

#[cfg(test)]
impl FailCapture {
    pub(crate) fn start() -> Self {
        *CAPTURED.lock().unwrap_or_else(|p| p.into_inner()) = Some(Vec::new());
        Self
    }

    /// Every always-on line emitted since `start`, in order.
    pub(crate) fn lines(&self) -> Vec<String> {
        CAPTURED
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
            .unwrap_or_default()
    }

    /// The one line whose first whitespace token is `slug`. Panics unless
    /// exactly one matched — "no line" and "several lines" are both reasons a
    /// downstream assertion would otherwise pass or fail for the wrong reason.
    pub(crate) fn one(&self, slug: &str) -> String {
        let hits: Vec<String> = self
            .lines()
            .into_iter()
            .filter(|l| l.split_whitespace().next() == Some(slug))
            .collect();
        assert_eq!(
            hits.len(),
            1,
            "expected exactly one `{slug}` line, got {hits:?} (all: {:?})",
            self.lines()
        );
        hits.into_iter().next().unwrap_or_default()
    }
}

#[cfg(test)]
impl Drop for FailCapture {
    fn drop(&mut self) {
        *CAPTURED.lock().unwrap_or_else(|p| p.into_inner()) = None;
    }
}

/// Test predicate: `line` is exactly `marker` plus the always-appended
/// trailing ` t=<ms>` field (see [`elapsed_ms`]).
#[cfg(test)]
pub(crate) fn line_is(line: &str, marker: &str) -> bool {
    line.strip_prefix(marker)
        .is_some_and(|rest| rest.is_empty() || rest.starts_with(" t="))
}

pub fn line(msg: impl AsRef<str>) {
    if !enabled() {
        return;
    }
    emit(Sink::Draw, msg.as_ref());
}

/// Always-on fail-visible line (writeback / Metal / missing resource / offline OFF).
pub fn fail(msg: impl AsRef<str>) {
    emit(Sink::Fail, msg.as_ref());
    if enabled() {
        emit(Sink::Draw, msg.as_ref());
    }
}

/// Always-on offline analysis line (prefix `OFF `). Same sink as [`fail`].
#[inline]
pub fn off(msg: impl AsRef<str>) {
    fail(format!("OFF {}", msg.as_ref()));
}

/// Whether the temporary guest-visible-content probe (`REIMS_VGPU_CONTENT_PROBE=1`)
/// is active.
///
/// The probe walks a whole frame, so it is off by default and must be read
/// *before* building any summary — see [`content_summary`].
pub fn content_probe_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var_os("REIMS_VGPU_CONTENT_PROBE")
            .map(|v| v == "1")
            .unwrap_or(false)
    })
}

/// Bisection knob (`REIMS_VGPU_SAMPLED_CACHE_OFF=1`): force every sampled bind
/// to miss the engine's retained-image cache and re-upload the producer's bytes.
///
/// The cache is the one place a draw binds pixels that were resolved for some
/// *earlier* draw, so it is the seam that separates "the wrong bytes were
/// chosen" from "the right bytes were chosen and the wrong image was bound".
/// Nothing else can bisect there: the retained `VkImage` has no CPU mirror to
/// compare against, and a boot with the cache off answers the question by
/// construction rather than by a correlation.
///
/// This costs one upload per bind — it is a diagnostic arm, never a product
/// configuration, and a boot that sets it must not be read for frame rate.
///
/// **Run against the Finder icon class, and it clears the cache.** Ten
/// recomposites with the cache off produced two corrupt rounds (5 and 6 of 7
/// icons), against two corrupt rounds on the matching cache-on boot. The rate
/// comparison at n=10 is weak, but the knob does not rely on it: with every
/// bind re-uploading the producer's bytes, a defect that lives in the retained
/// image cannot survive the arm at all, and corrupt rounds did. So for that
/// class the wrong pixels are already wrong when the runtime hands them over,
/// or the draw that consumes them covers the wrong region — the cache is not
/// binding a different image than the one it was given.
pub fn sampled_cache_disabled() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| {
        std::env::var_os("REIMS_VGPU_SAMPLED_CACHE_OFF")
            .map(|v| v == "1")
            .unwrap_or(false)
    })
}

/// Bisection knob (`REIMS_VGPU_SAMPLED_RESIDENT_GATE_OFF=1`): let the sampled
/// type-4 ladder's resident rung bind a ready resident with no currency test,
/// which is what it did before the guest-write gate and the merge behind it.
///
/// The rung serves a GPU image in place of a surface's guest pages. Whether that
/// image is still the surface is a question about the *guest*, and the only
/// arm that can answer it is one where the device stops asking: a boot with this
/// set reproduces the ungated behaviour exactly, on the same binary, so the
/// difference between the two boots is the gate and nothing else.
///
/// It exists because the alternative was comparing against a remembered
/// baseline from an earlier session and an earlier build. A device that can only
/// be A/B'd across rebuilds cannot separate its own change from the rig's drift.
///
/// A diagnostic arm, never a product configuration: with it set the device
/// knowingly binds content the hypervisor has watched the guest replace.
pub fn sampled_resident_gate_disabled() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| {
        std::env::var_os("REIMS_VGPU_SAMPLED_RESIDENT_GATE_OFF")
            .map(|v| v == "1")
            .unwrap_or(false)
    })
}

/// Which content-reuse rail a call site belongs to, so
/// [`content_reuse_disabled`] can be armed one rail at a time.
///
/// The names are the accepted values of `REIMS_VGPU_CONTENT_REUSE_OFF`, which
/// also takes `1` or `all` for every rail and a comma-separated list for any
/// subset. Splitting the family was earned rather than designed in: the bundled
/// arm returned "not the cause, but the latch", and separating a latch from two
/// innocents is three boots that only became worth spending once that was known.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContentReuseRail {
    /// The type-11 `LOAD` seed elision. The composite rail, and the only one of
    /// the three that can hold a whole *frame*: a resident that is taken to be
    /// current is loaded from by every later damage draw, so a bad one is
    /// preserved rather than replaced for as long as the stamp keeps matching.
    Type11Seed,
    /// `linear_sampled_memo_reuse` — a swizzled `Arc` reused when the
    /// authoritative entry's `(gva, generation, geometry)` is unchanged.
    LinearSampledMemo,
    /// The `guest_linear_memo` hit — a swizzled `Arc` reused when a re-read of
    /// the guest's own native rows byte-compares equal. The most conservative of
    /// the three: it re-reads guest memory every call and only skips the
    /// conversion, so it can only be wrong if the guest bytes are wrong.
    GuestLinearMemo,
}

impl ContentReuseRail {
    fn token(self) -> &'static str {
        match self {
            Self::Type11Seed => "t11seed",
            Self::LinearSampledMemo => "linmemo",
            Self::GuestLinearMemo => "guestmemo",
        }
    }
}

/// Bisection knob (`REIMS_VGPU_CONTENT_REUSE_OFF`): make a rail that *skips*
/// producing content because it believes the content is already there produce it
/// anyway. `1`/`all` arms every rail; a comma-separated list of
/// [`ContentReuseRail`] tokens arms a subset.
///
/// [`sampled_cache_disabled`] bisects one seam — which `VkImage` a bind gets.
/// This one bisects the seam above it: whether the bytes were recomputed at all.
/// The two are different questions and the first cannot answer the second, which
/// is why a boot with the sampled cache off still produced corrupt rounds.
///
/// The rails it covers all share one shape — a witness says "nothing has changed
/// since we last produced this", so the production is skipped:
///
/// - the type-11 `LOAD` seed elision, where a resident stamped with the
///   mapping's current `surface_content_epoch` is taken to already hold the
///   bytes the seed would upload (`type11_seed_elided`, ~8400 per recomposite
///   round — by far the largest of them),
/// - the linear sampled memo, which reuses a swizzled `Arc` when the
///   authoritative entry's `(gva, generation, geometry)` is unchanged,
/// - the guest linear memo, which reuses one when a re-read of the guest's own
///   native rows byte-compares equal.
///
/// Any of those serving a stale answer produces exactly the Goal-2 signature: a
/// wrong picture that is *held*, because the same witness keeps saying nothing
/// changed, and no counter anomaly, because the elision counts as success on
/// every rail it takes. A census cannot separate "the witness was right" from
/// "the witness was wrong" — only re-deriving the content can.
///
/// This re-uploads a frame per seed and re-swizzles every sampled bind. It is a
/// diagnostic arm, never a product configuration, and a boot that sets it must
/// not be read for frame rate.
///
/// # What it measured: the class is two defects, not one
///
/// Two 14-round Finder recomposite boots, x86 / Vulkan, same HEAD:
///
/// ```text
/// reuse ON   rounds 3,4,5,6 corrupt — held, no round recovered
/// reuse OFF  round 4 corrupt (5 of 7 icons), rounds 5-14 all clean
/// ```
///
/// Corruption **survives** the arm. That was read as "none of the three rails
/// causes it: the wrong pixels are wrong before any witness is consulted, and
/// re-deriving the content produces the wrong content again".
///
/// **That reading is confounded for the `t11seed` rail, and only for it.** With
/// the elision off, the LOAD falls back to `resolve_type11_load_seed`. Its first
/// rung is the host cache, which the skip-readback Store rail *cedes*; its
/// second reads the mapping's guest pages through `paint_mapping`, which lands
/// every intersecting deferred window before reading. The window it lands is the
/// one that surface's last Store armed, and that window's frame is the same
/// resident the elision would have loaded from. So "re-derived" here means the
/// resident's bytes copied into guest memory and read back: the arm changes the
/// transport and not the pixels, for the ~86 % of composite Stores that arm a
/// window at all. A rail that hands over the same bytes either way cannot be
/// cleared by an arm that only changes which way.
///
/// [`store_defer_disabled`] is the switch that separates them, because with the
/// writeback synchronous no window exists for the seed's read to land.
///
/// The other two rails (`linmemo`, `guestmemo`) are not affected by this: their
/// sources are guest linear memory, not a deferred surface window.
///
/// What it did not predict is the second half. With the rails off the defect
/// stopped *holding*. Every prior measurement of this class — six rounds of
/// `icon-recovery.sh` sampled 14 times over 65 s, byte-identical at t=1 s and
/// t=65 s — established a state that never recovers on its own; here it lasted
/// one round and the next redraw was clean.
///
/// So the user-visible defect is a composition of two:
///
/// 1. an intermittent producer of a wrong composite, which this arm does not
///    touch and which is the thing to find, and
/// 2. a reuse rail that latches whatever it was handed and never re-derives it,
///    which is what turns a one-frame glitch into a permanent one.
///
/// (2) is worth fixing on its own terms even though (1) is the root cause. A
/// device whose witnesses can only ever say "unchanged" has no way back from a
/// bad frame, and "renders correctly for a few frames then stays corrupted" is
/// exactly what that looks like from the guest.
///
/// The family gate has now earned its split: the three rails were bundled to
/// spend one boot on one bit, and that bit came back as "not the cause, but the
/// latch". Which of the three latches is the next question, and it is now worth
/// three boots.
///
/// n is small — one corrupt round against four — so the rate comparison is not
/// the claim. The claim is the qualitative one: with the rails off a corrupt
/// round was followed by a clean one, which no reuse-on boot has ever produced.
pub fn content_reuse_disabled(rail: ContentReuseRail) -> bool {
    static SPEC: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    let spec = SPEC.get_or_init(|| {
        std::env::var_os("REIMS_VGPU_CONTENT_REUSE_OFF")
            .map(|v| v.to_string_lossy().into_owned())
    });
    content_reuse_spec_arms(spec.as_deref(), rail)
}

/// The parse, split out so it is testable without an environment.
///
/// An unrecognised token arms nothing rather than everything: a typo in a
/// diagnostic flag must not silently turn a bisection into an all-rails boot
/// that then gets read as a per-rail result.
fn content_reuse_spec_arms(spec: Option<&str>, rail: ContentReuseRail) -> bool {
    let Some(spec) = spec else {
        return false;
    };
    spec.split(',')
        .map(str::trim)
        .any(|t| t == "1" || t == "all" || t == rail.token())
}

/// Which deferred-writeback rail a call site belongs to, so
/// [`store_defer_disabled`] can be armed one rail at a time.
///
/// The names are the accepted values of `REIMS_VGPU_STORE_DEFER_OFF`, which also
/// takes `1` or `all` for every rail and a comma-separated list for any subset.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StoreDeferRail {
    /// The type-11 composite Store: both the readback-then-defer window and the
    /// skip-readback resident window.
    Type11Surface,
    /// The type-2/3 GVA render Store's deferred window.
    Gva,
}

impl StoreDeferRail {
    fn token(self) -> &'static str {
        match self {
            Self::Type11Surface => "t11surface",
            Self::Gva => "gva",
        }
    }
}

/// Bisection knob (`REIMS_VGPU_STORE_DEFER_OFF`): make a Store write the guest's
/// pages before it returns instead of arming a window that writes them on
/// demand. `1`/`all` arms every rail; a comma-separated list of
/// [`StoreDeferRail`] tokens arms a subset.
///
/// # Why this is a different seam from [`content_reuse_disabled`]
///
/// [`content_reuse_disabled`] bisects "were the bytes recomputed at all". It
/// cannot bisect the deferred Store, and the reason is worth stating because its
/// own negative result depends on it: with the type-11 seed elision off, the
/// LOAD falls back to `resolve_type11_load_seed`, whose first rung is the host
/// cache — which the skip-readback rail has *ceded* — and whose second rung
/// reads the mapping's own guest pages through `paint_mapping`, which lands
/// every intersecting deferred window first. The window it lands is the one this
/// surface's last Store armed, and its frame is that Store's resident. So the
/// "re-derived" seed is the same resident's bytes, laundered through guest
/// memory. Turning the elision off changes the transport and not the pixels
/// whenever a window is armed, which is ~86 % of composite Stores.
///
/// This knob is the switch that actually separates them. With it on, the Store's
/// pixels are in the mapping's guest pages when the Store returns, no window
/// exists for a later reader to land, and a seed read of those pages is
/// independent of the resident.
///
/// It is a diagnostic arm, never a product configuration: it restores a whole
/// framebuffer GPU→CPU readback plus a per-row scatter into guest pages on every
/// composite Store, priced at 565 ms per second of wall clock on the x86/Vulkan
/// rail. A boot that sets it must not be read for frame rate.
///
/// # What it measured: the deferral is not the icon producer
///
/// Two 14-round Finder recomposite boots under load, x86 / Vulkan, same HEAD:
///
/// ```text
/// deferral on    3/14 corrupt   surface_flush 99, render_flush_over_guest_write 68
/// deferral off   7/14 corrupt   no windows armed at all (t11_keep_gpu_only_denied 293/round)
/// ```
///
/// With the rail off there is no window for a seed's guest-page read to land, so
/// the seed really is independent of the resident — the confound above is gone —
/// and the class still reproduces, at no lower a rate. The deferred type-11
/// writeback does not produce the wrong composite. (The off arm ran on a session
/// that had already survived an aborted round, so its *higher* rate is not a
/// claim; the claim is only that corruption survives the arm.)
///
/// What the same pair did establish is the size of a different defect. Two of
/// every three window landings were replacing guest bytes the guest itself had
/// written since the Store — ~18 300 in one session, silently. That is what
/// `render_flush_guest_written_ranges` now preserves, and it is a correctness
/// fix on its own terms rather than a fix for this class.
///
/// For whoever takes the next step: the corrupt cell is *not* blank and never
/// was. Read the crops. One control round put the Finder toolbar's sidebar
/// toggle and a chevron where the Desktop folder icon belongs; an off-arm round
/// put a narrow dark strip where Downloads belongs. Another UI element's pixels
/// arrive at the icon's destination, which makes this a question about which
/// source a sampled bind resolves to, not about whether a draw or a writeback
/// was lost.
pub fn store_defer_disabled(rail: StoreDeferRail) -> bool {
    static SPEC: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    let spec = SPEC.get_or_init(|| {
        std::env::var_os("REIMS_VGPU_STORE_DEFER_OFF").map(|v| v.to_string_lossy().into_owned())
    });
    store_defer_spec_arms(spec.as_deref(), rail)
}

/// Which deferred rail's fence binding [`fence_flush_disabled`] restores.
///
/// The names are the accepted values of `REIMS_VGPU_FENCE_FLUSH_OFF`, which also
/// takes `1` or `all` for every rail and a comma-separated list for any subset —
/// same grammar as `REIMS_VGPU_STORE_DEFER_OFF`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FenceFlushRail {
    /// `storage_flush::flush_gva_windows_before_fence` — the type-2/3 raw-GVA
    /// render Store rail.
    Gva,
    /// `storage_flush::flush_linear_windows_before_fence` — the linear
    /// compute-storage rail, keyed by a raw task GVA.
    Linear,
    /// `storage_flush::flush_mapping_windows_before_fence` — the mapping-keyed
    /// rails: type-11 render Stores and compute storage.
    Mapping,
}

impl FenceFlushRail {
    fn token(self) -> &'static str {
        match self {
            Self::Gva => "gva",
            Self::Linear => "linear",
            Self::Mapping => "mapping",
        }
    }
}

/// Bisection knob (`REIMS_VGPU_FENCE_FLUSH_OFF`): let one rail's deferred
/// windows outlive the completion stamp again, the way they all did before
/// `storage_flush::flush_gva_windows_before_fence`.
///
/// A control for this cannot be recorded on an earlier binary: `vm/boot-x86.sh`
/// rebuilds QEMU every boot, and a whole session on this branch went to reading
/// a baseline taken three commits back as if it were the arm's control. The
/// knob is what makes the arm and its control one binary apart.
///
/// It is per-rail because the rails no longer land together. The GVA and linear
/// bindings are measured and in place; the mapping binding is the one still
/// being priced, and an `=1` control that reverted all three would score the
/// mapping rail's cost against a control that had also given back two repairs
/// the corruption verdict depends on. `REIMS_VGPU_FENCE_FLUSH_OFF=mapping` is
/// the control for that measurement; `=1` and `=all` still revert everything,
/// which is what a from-scratch bisection wants.
pub fn fence_flush_disabled(rail: FenceFlushRail) -> bool {
    static SPEC: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    let spec = SPEC.get_or_init(|| {
        std::env::var_os("REIMS_VGPU_FENCE_FLUSH_OFF").map(|v| v.to_string_lossy().into_owned())
    });
    fence_flush_spec_arms(spec.as_deref(), rail)
}

/// Bisection knob (`REIMS_VGPU_MAPPING_PAGE_GUARD_OFF=1`): let a deferred
/// mapping-keyed flush write through a page list a fresh walk says has moved,
/// the way it did before `mapper::type4_pages_still_ours`.
///
/// The counters `mapping_pages_ours` / `mapping_pages_drifted` are emitted
/// whichever way this is set, on purpose: with the guard off a boot still
/// reports how many writes it would have refused, so the knob measures the
/// guard's cost in lost frames rather than hiding its rate. An arm and its
/// control are one binary apart.
pub fn mapping_page_guard_disabled() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| {
        std::env::var_os("REIMS_VGPU_MAPPING_PAGE_GUARD_OFF").is_some_and(|v| v == "1")
    })
}

/// The parse, split out so it is testable without an environment. Same
/// unrecognised-token rule as [`store_defer_spec_arms`]: a typo arms nothing.
fn fence_flush_spec_arms(spec: Option<&str>, rail: FenceFlushRail) -> bool {
    let Some(spec) = spec else {
        return false;
    };
    spec.split(',')
        .map(str::trim)
        .any(|t| t == "1" || t == "all" || t == rail.token())
}

/// Bisection knob (`REIMS_VGPU_GVA_IDENTITY_GEN_OFF=1`): key a GVA render
/// target's engine resident on `(gva, width, height)` alone again, the way it
/// was keyed before `TargetIdentity::Gva::generation` carried the hash of the
/// guest physical pages behind the span.
///
/// One producer reads this — `metal_draw::vulkan::gva_alloc_generation`, which
/// resolves `DrawEncodeRequest::gva_alloc_gen` once per draw. Every other site
/// copies that value (the pinned identity, the deferred window's `alloc_gen`,
/// the MRT secondary map), so a set knob makes every generation 0 and the rail
/// byte-identical to the shared-image behaviour.
///
/// A control for this cannot be recorded on an earlier binary: `vm/boot-x86.sh`
/// rebuilds QEMU every boot, so a baseline from another tree measures the
/// rebuild as well as the change. The knob is what makes the arm and its
/// control one binary apart.
///
/// # The generation does not stop the guest booting
///
/// The generation's first boot looked like it did. That arm sat 53 minutes
/// with only `host_window_cadence` in the fail log — not one guest GPU command
/// — and showed a macOS restart screen, while its control answered ssh in 33 s
/// with GPU traffic flowing. It was one boot per arm, and the evidence argued
/// against the obvious reading even then: with *zero* draws, nothing had called
/// `gva_alloc_generation`, so whatever stalled preceded this code entirely.
///
/// Re-measured over four boots, alternating so neither arm always follows a
/// cold rig (`.agents/repros/idgen-boot-ab.sh`, seconds from QEMU launch to the
/// guest answering ssh; `gpucmds` counts `linux_m2v_async`/`compute_linux`/
/// `type4` lines in that boot's fail-log slice):
///
/// ```text
/// round  arm  secs_to_ssh  gpucmds     round  arm  secs_to_ssh  gpucmds
///   1    arm      35         123         1    ctl      23         122
///   2    arm      24         181         2    ctl      23         171
/// ```
///
/// Every boot reached a running guest with GPU traffic. The hang did not
/// reproduce in either arm, so it was an early-boot event of its own and not
/// the generation, and the default stays on.
///
/// The residual claim is narrow: four boots say the arm boots. They say nothing
/// about the icon rate, which latches per boot and needs boots to score.
///
/// # The icon rate was then scored, and this knob does not move it
///
/// Six pairs, order alternating, one binary, identical 300 s pre-drive before
/// either arm was scored (`.agents/repros/icon-boot-ab.sh`, 2026-07-31):
///
/// ```text
/// arm  clean 4  corrupt 1  unmeasured 1 (guest kernel panic)
/// ctl  clean 4  corrupt 2  unmeasured 0
/// ```
///
/// One corrupt of five measured against two of six is no difference (Fisher
/// exact, two-tailed, p ≈ 1.0), and the class **still fires with the generation
/// on**. So this is not an underpowered look at a real effect: the arm
/// reproduces the bug.
///
/// The mechanism counter agrees, which is what makes this more than a null
/// verdict. If aliasing drove the latch, corrupt boots should carry more of it.
/// `gvares_aliased` was 1306, 1127 and 1599 on the three corrupt boots against a
/// clean range of 993–1718 — means 1344 corrupt against 1414 clean, no
/// separation, and not even on the high side.
///
/// Two allocations sharing one GPU image at a recycled GVA is therefore not what
/// latches Finder icon corruption per boot. Keying on the page hash is still the
/// right contract and stays on by default; it is simply not this bug. Both arms
/// carried the GVA rail's fence binding and neither carried the mapping-keyed
/// rail's, so the untested candidates are there rather than here.
pub fn gva_identity_gen_disabled() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| {
        std::env::var_os("REIMS_VGPU_GVA_IDENTITY_GEN_OFF").is_some_and(|v| v == "1")
    })
}

/// The parse, split out so it is testable without an environment. Same
/// unrecognised-token rule as [`content_reuse_spec_arms`]: a typo arms nothing.
fn store_defer_spec_arms(spec: Option<&str>, rail: StoreDeferRail) -> bool {
    let Some(spec) = spec else {
        return false;
    };
    spec.split(',')
        .map(str::trim)
        .any(|t| t == "1" || t == "all" || t == rail.token())
}

/// Summarise a tightly packed `texel`-byte-per-pixel image so a *wrong* image
/// identifies itself in the log without a screen-to-mapping join.
///
/// The discriminating field is `distinct`: a correct icon carries hundreds of
/// distinct texels, and every corruption shape this was written for — a solid
/// square, a solid bar, a cleared region — collapses to one. `px0` then names
/// which colour it collapsed to, and `hash` lets the same image be recognised
/// at two stages. Counting stops at [`DISTINCT_CAP`] so the cost is bounded on
/// a display-sized frame rather than proportional to its palette.
///
/// `quad` reports nonzero texels per screen quadrant (`nw/ne/sw/se`), because a
/// scalar count cannot see *where* the content is. The shape this exists to
/// catch is an icon rendering shrunken into the top-left of its quad with the
/// rest transparent — a whole-image `nz` reads that as "some content, looks
/// plausible", while `quad` reads it as everything in `nw` and nothing
/// elsewhere.
///
/// Caller must gate on [`content_probe_enabled`].
///
/// Cost is bounded at [`SAMPLE_CAP`] texels by a uniform 2-D subsample, not
/// proportional to the buffer: this probe runs on the same path that flushes
/// 1920x1080 composites ~200 times a second, and a whole-frame walk there would
/// slow the device enough to stop reproducing a load-dependent defect. The
/// subsample is 2-D rather than linear precisely so `quad` keeps its meaning.
/// `stride` is reported so `nz` reads as the sample it is.
///
/// `buf` must be tightly packed at `width * texel` bytes per row.
pub fn content_summary(buf: &[u8], texel: u32, width: u32, height: u32) -> String {
    /// Enough to separate "one colour" from "an image" — the question this
    /// probe asks — without building a palette of a 1920x1080 frame.
    const DISTINCT_CAP: usize = 64;
    /// ~16k texels is a full-precision walk of any icon-scale surface and a
    /// 1/128 sample of a display-sized one.
    const SAMPLE_CAP: usize = 16384;
    let texel = texel.max(1) as usize;
    let texels = buf.len() / texel;
    let (w, h) = (width as usize, height as usize);
    if w == 0 || h == 0 || w * h * texel > buf.len() {
        return format!("texels={texels} geom_mismatch=1 buf={}", buf.len());
    }
    // Smallest uniform 2-D step that brings the sample under the cap. Integer
    // search rather than a sqrt so the bound is exact for every shape.
    let mut stride = 1usize;
    while w.div_ceil(stride) * h.div_ceil(stride) > SAMPLE_CAP {
        stride += 1;
    }
    let mut distinct: Vec<&[u8]> = Vec::with_capacity(DISTINCT_CAP);
    let mut nz = 0usize;
    let mut sampled = 0usize;
    let mut quad = [0usize; 4];
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for y in (0..h).step_by(stride) {
        for x in (0..w).step_by(stride) {
            let off = (y * w + x) * texel;
            let px = &buf[off..off + texel];
            sampled += 1;
            for &b in px {
                hash ^= b as u64;
                hash = hash.wrapping_mul(0x1000_0000_01b3);
            }
            if px.iter().any(|&b| b != 0) {
                nz += 1;
                quad[usize::from(x >= w / 2) + 2 * usize::from(y >= h / 2)] += 1;
            }
            if distinct.len() < DISTINCT_CAP && !distinct.contains(&px) {
                distinct.push(px);
            }
        }
    }
    let px0: String = buf
        .iter()
        .take(texel)
        .map(|b| format!("{b:02x}"))
        .collect::<Vec<_>>()
        .join("");
    let capped = if distinct.len() >= DISTINCT_CAP {
        "+"
    } else {
        ""
    };
    format!(
        "texels={texels} stride={stride} sampled={sampled} distinct={}{capped} nz={nz} quad={}/{}/{}/{} px0={px0} hash={hash:016x}",
        distinct.len(),
        quad[0],
        quad[1],
        quad[2],
        quad[3],
    )
}

/// Count nonzero **bytes** and max sample in a tightly packed image buffer.
///
/// Note: solid black with alpha=255 has nz == byte_len (every A channel). Prefer
/// [`bgra_rgb_stats`] when diagnosing visible content vs QMP black.
pub fn nonzero_stats(buf: &[u8]) -> (usize, u8) {
    let mut nz = 0usize;
    let mut max = 0u8;
    for &b in buf {
        if b != 0 {
            nz += 1;
        }
        if b > max {
            max = b;
        }
    }
    (nz, max)
}

/// Visible-content stats for tight BGRA8: rgb_nz = pixels with max(B,G,R)>0,
/// max_rgb = max of B/G/R, px0 = first pixel BGRA.
pub fn bgra_rgb_stats(bgra: &[u8]) -> (usize, u8, [u8; 4]) {
    let mut rgb_nz = 0usize;
    let mut max_rgb = 0u8;
    let px0 = if bgra.len() >= 4 {
        [bgra[0], bgra[1], bgra[2], bgra[3]]
    } else {
        [0, 0, 0, 0]
    };
    for px in bgra.chunks_exact(4) {
        let m = px[0].max(px[1]).max(px[2]);
        if m != 0 {
            rgb_nz += 1;
        }
        if m > max_rgb {
            max_rgb = m;
        }
    }
    (rgb_nz, max_rgb, px0)
}

/// Fused present-capture stats for tight BGRA8: one pass yielding what
/// [`nonzero_stats`] and [`bgra_rgb_stats`] compute separately —
/// `(byte_nz, byte_max, rgb_nz, max_rgb, px0)`. The present drain path scans
/// the full 8 MiB frame on every present while holding the device lock; folding
/// the two per-pixel passes into one halves that measure-only lock-hold (a
/// direct win on present cadence / boot convergence). Byte-exact with the two
/// separate functions: `byte_nz`/`byte_max` count all four channels,
/// `rgb_nz`/`max_rgb` the low three.
///
/// On x86_64 this dispatches to an SSE2 vectorized kernel (16 bytes/iteration,
/// ~11× the scalar loop measured at opt-level 2); SSE2 is baseline for the
/// arch so no runtime feature detection is needed. The scalar body remains the
/// reference on other targets (arm backend-metal build) and the oracle the
/// `bgra_present_stats_byte_exact_with_sse2` unit asserts against.
#[inline]
pub fn bgra_present_stats(bgra: &[u8]) -> (usize, u8, usize, u8, [u8; 4]) {
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: SSE2 is part of the x86_64 baseline ABI, always available.
        unsafe { bgra_present_stats_sse2(bgra) }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        bgra_present_stats_scalar(bgra)
    }
}

/// Scalar reference for [`bgra_present_stats`] — the byte-exact definition the
/// SSE2 kernel matches. Kept out of the x86_64 hot path but used verbatim on
/// other arches and as the unit-test oracle.
pub fn bgra_present_stats_scalar(bgra: &[u8]) -> (usize, u8, usize, u8, [u8; 4]) {
    let mut byte_nz = 0usize;
    let mut byte_max = 0u8;
    let mut rgb_nz = 0usize;
    let mut max_rgb = 0u8;
    let px0 = if bgra.len() >= 4 {
        [bgra[0], bgra[1], bgra[2], bgra[3]]
    } else {
        [0, 0, 0, 0]
    };
    for px in bgra.chunks_exact(4) {
        // Per-byte nonzero/max across all four channels (== nonzero_stats over
        // a length that is a multiple of 4; the present frame always is).
        for &b in px {
            if b != 0 {
                byte_nz += 1;
            }
            if b > byte_max {
                byte_max = b;
            }
        }
        let m = px[0].max(px[1]).max(px[2]);
        if m != 0 {
            rgb_nz += 1;
        }
        if m > max_rgb {
            max_rgb = m;
        }
    }
    (byte_nz, byte_max, rgb_nz, max_rgb, px0)
}

/// SSE2 kernel for [`bgra_present_stats`], byte-exact with
/// [`bgra_present_stats_scalar`]. Processes 16 bytes (4 BGRA pixels) per
/// iteration: `pmaxub` for the running byte/rgb maxima, `pcmpeqb` + `psadbw`
/// to count zero bytes (`byte_nz = len − zeros`), and a per-u32-lane
/// `pcmpeqd` on alpha-masked pixels to count fully-black-rgb pixels
/// (`rgb_nz = pixels − rgb_zeros`). The 8-bit zero accumulator is flushed via
/// `psadbw` every 255 iterations so it cannot overflow.
///
/// # Safety
/// Requires SSE2, which is guaranteed on every x86_64 target.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn bgra_present_stats_sse2(bgra: &[u8]) -> (usize, u8, usize, u8, [u8; 4]) {
    use std::arch::x86_64::*;
    // SAFETY: SSE2 is guaranteed on x86_64; all loads/stores below are bounded
    // by `n = len & !15` (full 16-byte blocks) with a scalar tail for the rest.
    unsafe {
        let px0 = if bgra.len() >= 4 {
            [bgra[0], bgra[1], bgra[2], bgra[3]]
        } else {
            [0, 0, 0, 0]
        };
        let n = bgra.len() & !15;
        let mut byte_max = 0u8;
        let mut max_rgb = 0u8;
        let zero = _mm_setzero_si128();
        // Keep the low three bytes (B,G,R) of each little-endian pixel, drop alpha.
        let rgb_mask = _mm_set1_epi32(0x00FF_FFFFu32 as i32);
        let mut vmax = zero; // running max over all four channels
        let mut vmax_rgb = zero; // running max over B/G/R only
        let mut vzero_bytes = zero; // 64-bit lanes: total zero-byte count
        let mut vzero_rgb = zero; // 32-bit lanes: fully-black-rgb pixel count
        let mut ptr = bgra.as_ptr();
        let mut rem = n;
        while rem > 0 {
            // Bound the 8-bit zero-byte accumulator to ≤255 before flushing.
            let block = rem.min(255 * 16);
            let mut inner_zero = zero;
            let mut b = 0usize;
            while b < block {
                let v = _mm_loadu_si128(ptr as *const __m128i);
                vmax = _mm_max_epu8(vmax, v);
                let zmask = _mm_cmpeq_epi8(v, zero); // 0xFF per zero byte
                inner_zero = _mm_sub_epi8(inner_zero, zmask); // +1 per zero byte
                let vr = _mm_and_si128(v, rgb_mask);
                vmax_rgb = _mm_max_epu8(vmax_rgb, vr);
                let rgb_eq = _mm_cmpeq_epi32(vr, zero); // 0xFFFFFFFF per black-rgb px
                vzero_rgb = _mm_sub_epi32(vzero_rgb, rgb_eq);
                ptr = ptr.add(16);
                b += 16;
            }
            vzero_bytes = _mm_add_epi64(vzero_bytes, _mm_sad_epu8(inner_zero, zero));
            rem -= block;
        }
        let mut lanes = [0u8; 16];
        _mm_storeu_si128(lanes.as_mut_ptr() as *mut __m128i, vmax);
        for &x in &lanes {
            byte_max = byte_max.max(x);
        }
        _mm_storeu_si128(lanes.as_mut_ptr() as *mut __m128i, vmax_rgb);
        for &x in &lanes {
            max_rgb = max_rgb.max(x);
        }
        let mut z64 = [0u64; 2];
        _mm_storeu_si128(z64.as_mut_ptr() as *mut __m128i, vzero_bytes);
        let mut byte_nz = n - (z64[0] + z64[1]) as usize;
        let mut z32 = [0u32; 4];
        _mm_storeu_si128(z32.as_mut_ptr() as *mut __m128i, vzero_rgb);
        let rgb_zeros = (z32[0] + z32[1] + z32[2] + z32[3]) as usize;
        let mut rgb_nz = n / 4 - rgb_zeros;
        // Scalar tail (frame length is a multiple of 4 but not always of 16).
        for px in bgra[n..].chunks_exact(4) {
            for &b in px {
                if b != 0 {
                    byte_nz += 1;
                }
                byte_max = byte_max.max(b);
            }
            let m = px[0].max(px[1]).max(px[2]);
            if m != 0 {
                rgb_nz += 1;
            }
            max_rgb = max_rgb.max(m);
        }
        (byte_nz, byte_max, rgb_nz, max_rgb, px0)
    }
}

/// Same for tight RGBA8 (m2v encode output).
pub fn rgba_rgb_stats(rgba: &[u8]) -> (usize, u8, [u8; 4]) {
    let mut rgb_nz = 0usize;
    let mut max_rgb = 0u8;
    let px0 = if rgba.len() >= 4 {
        [rgba[0], rgba[1], rgba[2], rgba[3]]
    } else {
        [0, 0, 0, 0]
    };
    for px in rgba.chunks_exact(4) {
        let m = px[0].max(px[1]).max(px[2]);
        if m != 0 {
            rgb_nz += 1;
        }
        if m > max_rgb {
            max_rgb = m;
        }
    }
    (rgb_nz, max_rgb, px0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    /// The bisection spec arms exactly the rails it names.
    ///
    /// A per-rail boot is read as "this rail is the latch" or "this rail is
    /// innocent", and both readings are wrong if the spec silently armed a
    /// different set than the one on the command line. In particular a typo must
    /// arm *nothing*: arming everything would turn a bisection boot into the
    /// all-rails boot whose result is already known, and it would agree with the
    /// hypothesis under test for the wrong reason.
    #[test]
    fn the_content_reuse_spec_arms_exactly_the_rails_it_names() {
        use ContentReuseRail::*;
        let all = [Type11Seed, LinearSampledMemo, GuestLinearMemo];

        // Absent: nothing armed. This is every normal boot.
        for r in all {
            assert!(!content_reuse_spec_arms(None, r));
        }
        // The two whole-family spellings.
        for spec in ["1", "all"] {
            for r in all {
                assert!(content_reuse_spec_arms(Some(spec), r), "{spec} must arm {r:?}");
            }
        }
        // One rail names one rail.
        assert!(content_reuse_spec_arms(Some("t11seed"), Type11Seed));
        assert!(!content_reuse_spec_arms(Some("t11seed"), LinearSampledMemo));
        assert!(!content_reuse_spec_arms(Some("t11seed"), GuestLinearMemo));
        // Subsets, and whitespace around list entries.
        assert!(content_reuse_spec_arms(
            Some("linmemo, guestmemo"),
            GuestLinearMemo
        ));
        assert!(!content_reuse_spec_arms(
            Some("linmemo, guestmemo"),
            Type11Seed
        ));
        // A typo arms nothing — it must not fall back to the whole family.
        for spec in ["", "0", "t11", "seed", "yes"] {
            for r in all {
                assert!(
                    !content_reuse_spec_arms(Some(spec), r),
                    "{spec:?} must not arm {r:?}"
                );
            }
        }
    }

    /// Same contract as the reuse spec, on the rail set that bisects the
    /// deferred writeback. Kept as its own test rather than folded into the one
    /// above because the two knobs answer different questions and a shared
    /// helper would let one family's tokens arm the other's rails.
    #[test]
    fn the_store_defer_spec_arms_exactly_the_rails_it_names() {
        use StoreDeferRail::*;
        let all = [Type11Surface, Gva];

        for r in all {
            assert!(!store_defer_spec_arms(None, r));
        }
        for spec in ["1", "all"] {
            for r in all {
                assert!(store_defer_spec_arms(Some(spec), r), "{spec} must arm {r:?}");
            }
        }
        assert!(store_defer_spec_arms(Some("t11surface"), Type11Surface));
        assert!(!store_defer_spec_arms(Some("t11surface"), Gva));
        assert!(store_defer_spec_arms(Some("gva, t11surface"), Gva));
        // The other knob's tokens name no rail here.
        for spec in ["", "0", "t11seed", "linmemo", "surface", "defer"] {
            for r in all {
                assert!(
                    !store_defer_spec_arms(Some(spec), r),
                    "{spec:?} must not arm {r:?}"
                );
            }
        }
    }

    /// The fence knob names one rail at a time, and that is the whole point of
    /// it: the GVA and linear bindings are already measured and in place, so a
    /// control that reverted all three would price the mapping rail against a
    /// control that had also given back two repairs the corruption verdict
    /// depends on. `=1` and `=all` still revert everything for a from-scratch
    /// bisection.
    #[test]
    fn the_fence_flush_spec_arms_exactly_the_rails_it_names() {
        use FenceFlushRail::*;
        let all = [Gva, Linear, Mapping];

        for r in all {
            assert!(!fence_flush_spec_arms(None, r));
        }
        for spec in ["1", "all"] {
            for r in all {
                assert!(fence_flush_spec_arms(Some(spec), r), "{spec} must arm {r:?}");
            }
        }
        assert!(fence_flush_spec_arms(Some("mapping"), Mapping));
        assert!(
            !fence_flush_spec_arms(Some("mapping"), Gva),
            "the mapping rail's control must leave the GVA repair in place"
        );
        assert!(
            !fence_flush_spec_arms(Some("mapping"), Linear),
            "and must leave the linear repair in place"
        );
        assert!(fence_flush_spec_arms(Some("gva, linear"), Linear));
        // `gva` is a token in both knobs' grammars; nothing else crosses over,
        // and an unrecognised token arms nothing rather than everything.
        for spec in ["", "0", "t11surface", "render", "surface", "fence"] {
            for r in all {
                assert!(
                    !fence_flush_spec_arms(Some(spec), r),
                    "{spec:?} must not arm {r:?}"
                );
            }
        }
    }

    #[test]
    fn flood_key_is_the_slug_skipping_the_off_marker() {
        assert_eq!(
            flood_key("OFF type5_view_zc ref=355 sid=62 view=1920x1080 t=1"),
            "type5_view_zc"
        );
        assert_eq!(
            flood_key("map_family op=InvalidateResources ch=3 t=1"),
            "map_family"
        );
        // A bare slug with no fields still keys on itself.
        assert_eq!(flood_key("present_converge"), "present_converge");
    }

    #[test]
    fn flood_window_names_only_over_threshold_prefixes_once_per_window() {
        let mut fw = FloodWindow::new(0);
        // A runaway prefix (over threshold) alongside a quiet one, all inside the
        // window → nothing reported until the window closes.
        for _ in 0..FLOOD_THRESHOLD_PER_WINDOW {
            assert!(fw.note("OFF hot_line a=1 t=1", 10).is_empty());
        }
        assert!(fw.note("OFF quiet_line b=2 t=1", 10).is_empty());

        // A note past the window boundary closes it: only the over-threshold
        // prefix is named (the +1 in this closing window still counts).
        let flooders = fw.note("OFF hot_line a=1 t=1", FLOOD_WINDOW_MS);
        assert_eq!(flooders.len(), 1, "only the runaway prefix is reported");
        assert_eq!(flooders[0].0, "hot_line");
        assert!(flooders[0].1 >= FLOOD_THRESHOLD_PER_WINDOW);

        // Window reset: the quiet prefix never trips it across a fresh window.
        assert!(fw
            .note("OFF quiet_line b=2 t=1", FLOOD_WINDOW_MS + 10)
            .is_empty());
        let none = fw.note("OFF quiet_line b=2 t=1", 2 * FLOOD_WINDOW_MS + 20);
        assert!(none.is_empty(), "a quiet prefix never floods");
    }

    #[test]
    fn bgra_present_stats_is_byte_exact_with_separate_scans() {
        // Mixed content: black, alpha-only, colored, saturated — the classes
        // the present proxies distinguish.
        let frame: Vec<u8> = vec![
            0, 0, 0, 0, // fully black
            0, 0, 0, 255, // alpha-only (rgb-empty, byte-nonzero)
            10, 20, 30, 40, // colored
            255, 255, 255, 255, // saturated
            0, 200, 0, 128, // green only
        ];
        let (nz, maxb) = nonzero_stats(&frame);
        let (rgb_nz, max_rgb, px0) = bgra_rgb_stats(&frame);
        let fused = bgra_present_stats(&frame);
        assert_eq!(
            fused,
            (nz, maxb, rgb_nz, max_rgb, px0),
            "fused present stats must equal the two separate scans byte-for-byte"
        );
        // Sanity: alpha-only pixel counts as byte-nonzero but not rgb-nonzero.
        assert_eq!(rgb_nz, 3, "black + alpha-only are rgb-empty");
        assert_eq!(nz, 2 + 3 + 4 + 2, "nonzero bytes across all four channels");
        // The dispatched entry must equal the scalar reference on this arch too.
        assert_eq!(fused, bgra_present_stats_scalar(&frame));
    }

    #[test]
    fn bgra_present_stats_byte_exact_with_sse2() {
        // Exercise the SSE2 kernel against the scalar reference over sizes that
        // hit the full-block path, the 255-iteration accumulator flush, the
        // sub-16-byte scalar tail, and the short/empty guards — with content
        // covering every class (black, alpha-only, single-channel, saturated).
        for &pixels in &[0usize, 1, 3, 4, 16, 17, 255 * 4, 255 * 4 + 5, 1920 * 1080] {
            let mut frame = vec![0u8; pixels * 4];
            for (i, b) in frame.iter_mut().enumerate() {
                // Deterministic pseudo-content with black runs and saturation.
                let v = (i.wrapping_mul(2_654_435_761) >> 11) & 0xff;
                *b = if i % 7 == 0 { 0 } else { v as u8 };
            }
            let want = bgra_present_stats_scalar(&frame);
            let got = bgra_present_stats(&frame);
            assert_eq!(got, want, "SSE2 kernel diverged at pixels={pixels}");
        }
        // All-black and all-saturated corner cases.
        assert_eq!(
            bgra_present_stats(&[0u8; 64]),
            bgra_present_stats_scalar(&[0u8; 64])
        );
        assert_eq!(
            bgra_present_stats(&[255u8; 64]),
            bgra_present_stats_scalar(&[255u8; 64])
        );
    }

    #[test]
    fn append_reuses_the_open_file_handle() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock after epoch")
            .as_nanos();
        let path = format!("/tmp/reims-vgpu-draw-log-{nonce}.log");
        let moved = format!("{path}.moved");
        let file = Mutex::new(None);

        append_sync(&file, &path, "first", 0);
        fs::rename(&path, &moved).expect("rename open log");
        append_sync(&file, &path, "second", 0);

        assert!(!std::path::Path::new(&path).exists());
        let body = fs::read_to_string(&moved).unwrap();
        let lines: Vec<&str> = body.lines().collect();
        assert_eq!(lines.len(), 2);
        assert!(lines[0].starts_with("first t="));
        assert!(lines[1].starts_with("second t="));
        fs::remove_file(moved).unwrap();
    }

    #[test]
    fn fail_writes_apv_fail_log() {
        let path = fail_log_path();
        let marker = format!("draw_log_selftest_{}", std::process::id());
        fail(&marker);
        let body = fs::read_to_string(path).expect("fail log readable");
        assert!(
            body.lines().any(|l| l.contains(&marker)),
            "fail() must append to the fail log"
        );
        assert_ne!(
            path, "/tmp/reims-vgpu-fail.log",
            "test builds must not share the product fail log — a cargo test \
             run concurrent with a live boot corrupts A/B evidence"
        );
    }

    #[test]
    fn bgra_rgb_stats_black_alpha_full_is_zero_rgb_nz() {
        // Solid black opaque: every A byte is 255 → byte nz is full, rgb_nz must be 0.
        let mut bgra = vec![0u8; 16];
        for px in bgra.chunks_exact_mut(4) {
            px[3] = 255;
        }
        let (byte_nz, _) = nonzero_stats(&bgra);
        assert_eq!(byte_nz, 4); // four alpha bytes
        let (rgb_nz, max_rgb, px0) = bgra_rgb_stats(&bgra);
        assert_eq!(rgb_nz, 0);
        assert_eq!(max_rgb, 0);
        assert_eq!(px0, [0, 0, 0, 255]);
    }

    #[test]
    fn bgra_rgb_stats_gray_counts_pixels() {
        let mut bgra = vec![0u8; 8];
        bgra[0] = 100;
        bgra[1] = 100;
        bgra[2] = 100;
        bgra[3] = 255;
        bgra[7] = 255;
        let (rgb_nz, max_rgb, _) = bgra_rgb_stats(&bgra);
        assert_eq!(rgb_nz, 1);
        assert_eq!(max_rgb, 100);
    }

    /// The one property the content probe exists for: a solid fill and a real
    /// image must not read alike, at icon scale, with no sampling in the way.
    #[test]
    fn a_solid_fill_and_an_image_differ_in_the_distinct_count() {
        let solid = [0xffu8, 0x00, 0x00, 0xff].repeat(66 * 66);
        let s = content_summary(&solid, 4, 66, 66);
        assert!(s.contains(" stride=1 "), "icon scale must not sample: {s}");
        assert!(s.contains(" distinct=1 "), "solid fill is one colour: {s}");
        assert!(s.contains(" px0=ff0000ff "), "px0 names the colour: {s}");

        let image: Vec<u8> = (0..66u32 * 66)
            .flat_map(|i| [(i % 251) as u8, (i % 241) as u8, (i % 239) as u8, 0xff])
            .collect();
        let g = content_summary(&image, 4, 66, 66);
        assert!(
            g.contains(" distinct=64+ "),
            "an image is many colours: {g}"
        );
        assert_ne!(
            s.split(" hash=").nth(1),
            g.split(" hash=").nth(1),
            "distinct content must not share a hash"
        );
    }

    /// The shape `quad` exists for: an image shrunken into the top-left of its
    /// allocated extent has an unremarkable whole-image `nz` and is only
    /// separable by where the content sits.
    #[test]
    fn a_shrunken_top_left_image_is_visible_only_in_the_quadrants() {
        let (w, h) = (64usize, 64usize);
        let mut buf = vec![0u8; w * h * 4];
        for y in 0..h / 4 {
            for x in 0..w / 4 {
                buf[(y * w + x) * 4..][..4].copy_from_slice(&[0x40, 0x80, 0xc0, 0xff]);
            }
        }
        let s = content_summary(&buf, 4, w as u32, h as u32);
        assert!(s.contains(" nz=256 "), "a scalar count looks ordinary: {s}");
        assert!(
            s.contains(" quad=256/0/0/0 "),
            "all content must land in nw: {s}"
        );

        // The same texel count spread over the whole extent is the healthy
        // shape, and must not read alike.
        let mut spread = vec![0u8; w * h * 4];
        for i in (0..w * h).step_by(16) {
            spread[i * 4..][..4].copy_from_slice(&[0x40, 0x80, 0xc0, 0xff]);
        }
        let g = content_summary(&spread, 4, w as u32, h as u32);
        assert!(g.contains(" nz=256 "), "same scalar count: {g}");
        assert!(g.contains(" quad=64/64/64/64 "), "evenly spread: {g}");
    }

    /// A geometry that does not describe the buffer must say so rather than
    /// index past the end or report a quadrant split it cannot support.
    #[test]
    fn a_geometry_that_overruns_the_buffer_is_named_not_indexed() {
        let buf = vec![0u8; 64 * 4];
        let s = content_summary(&buf, 4, 64, 64);
        assert!(s.contains("geom_mismatch=1"), "{s}");
    }

    /// The probe runs on the 1920x1080 flush path, so its cost must be capped
    /// by striding rather than growing with the frame.
    #[test]
    fn a_display_sized_frame_is_sampled_not_walked() {
        let frame = vec![0u8; 1920 * 1080 * 4];
        let s = content_summary(&frame, 4, 1920, 1080);
        assert!(s.starts_with("texels=2073600 "), "{s}");
        let sampled: usize = s
            .split(" sampled=")
            .nth(1)
            .and_then(|r| r.split(' ').next())
            .and_then(|v| v.parse().ok())
            .expect("sampled field");
        assert!(sampled <= 16384, "cost must stay bounded, got {sampled}");
    }
}
