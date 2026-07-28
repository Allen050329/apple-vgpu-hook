//! Always-on probe: does the frame we are about to show still match the
//! guest's own copy of the surface it named?
//!
//! [`crate::runtime::scanout::capture_present_frame`] reads the GPU resident
//! and the host surface cache — it never touches guest memory. So if a guest
//! Store fails to reach the resident, no later present can recover: there is
//! no path by which the guest's own pages correct our copy, and the surface
//! keeps showing whatever the resident last held. That is a structural claim
//! about what happens *if* the resident is stale; it says nothing about
//! whether it ever is. This module reads that.
//!
//! Two properties make the guest arm worth reading here and nowhere else:
//!
//! - It is the only copy in the comparison our present path did not produce.
//!   Comparing our resident against another of our residents (a same-geometry
//!   peer, a previous frame) can show two states differ and cannot say which
//!   one the guest asked for.
//! - The guest pages are the surface. Under unified memory the guest's own
//!   `screencapture` composites from them, so a divergence here is exactly
//!   the split measured by capturing both screens at once: guest moved on,
//!   we did not.
//!
//! ## Why this does not read through `mapper::read_mapping_bytes`
//!
//! That reader calls [`crate::runtime::storage_flush::flush_intersecting`]
//! first, which copies our pinned resident *into* the guest window before
//! returning the bytes. A comparison built on it would agree with itself
//! whenever a deferred window was pending — the probe would be structurally
//! unable to report the case it exists to find. This reads the guest's bytes
//! as they stand, page by page, and reports the pending-window count
//! separately so a legitimate not-yet-flushed window is never read as staleness.
//!
//! ## Why magnitude, not a count
//!
//! A pixel differing by 1/255 and one differing by 255/255 are not the same
//! finding, and this project has already manufactured a defect class out of a
//! metric that scored them alike. Every line carries the deviation histogram
//! and the maximum; a window whose `max` is a handful of LSB has found a
//! re-encode rounding difference, not a stale frame.

use crate::model::DeviceState;
use crate::runtime::host::{HostMemory, HostOps};

/// Rows sampled per present, as fractions of the frame height. Three rows
/// spread over the frame, so a divergence confined to one band (a window, a
/// menu strip) is still crossed by at least one of them; the probe reports
/// which rows it actually read.
const SAMPLE_ROW_NUMERATORS: [u32; 3] = [1, 2, 3];
const SAMPLE_ROW_DENOMINATOR: u32 = 4;

/// Per-channel deviation between the frame we captured and the guest's pages.
///
/// `px` is the number of pixels compared, so every count below has a
/// denominator on the same line. `gt*` are pixel counts whose largest
/// colour-channel deviation exceeds the named threshold; `max` is the largest
/// deviation anywhere in the sample.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RowDivergence {
    pub px: usize,
    pub gt1: usize,
    pub gt4: usize,
    pub gt16: usize,
    pub gt64: usize,
    pub max: u8,
    /// Largest deviation after swapping the guest row's first and third
    /// channels. A channel-order mismatch reads as a large `max` with a small
    /// `max_swapped`; a stale resident is large in both. Without this the two
    /// are indistinguishable, and this project has confused them before.
    pub max_swapped: u8,
}

impl RowDivergence {
    /// Accumulate `other` into `self`, keeping worst-case magnitudes.
    pub fn merge(&mut self, other: RowDivergence) {
        self.px += other.px;
        self.gt1 += other.gt1;
        self.gt4 += other.gt4;
        self.gt16 += other.gt16;
        self.gt64 += other.gt64;
        self.max = self.max.max(other.max);
        self.max_swapped = self.max_swapped.max(other.max_swapped);
    }

    /// Whether anything above sub-perceptual rounding was seen. 4/255 is the
    /// ceiling this project measured for a pure re-encode difference over a
    /// whole frame; below it there is nothing a human could see.
    pub fn is_visible(&self) -> bool {
        self.gt4 > 0
    }
}

/// Compare two tight BGRA8 rows of the same pixel count.
///
/// Both sides are wire order (BGRA); alpha is excluded because the present
/// path and the guest's own compositor do not agree on it by contract and a
/// difference there is not a visible defect.
pub fn compare_bgra_rows(ours: &[u8], guest: &[u8]) -> RowDivergence {
    let mut d = RowDivergence::default();
    for (a, b) in ours.chunks_exact(4).zip(guest.chunks_exact(4)) {
        let dev = (0..3).map(|i| a[i].abs_diff(b[i])).max().unwrap_or(0);
        let dev_swapped = a[0]
            .abs_diff(b[2])
            .max(a[1].abs_diff(b[1]))
            .max(a[2].abs_diff(b[0]));
        d.px += 1;
        d.gt1 += usize::from(dev > 1);
        d.gt4 += usize::from(dev > 4);
        d.gt16 += usize::from(dev > 16);
        d.gt64 += usize::from(dev > 64);
        d.max = d.max.max(dev);
        d.max_swapped = d.max_swapped.max(dev_swapped);
    }
    d
}

/// Read `[off, off + buf.len())` of a mapping's guest pages without flushing.
///
/// Walks `page_entries` and reads each page's slice directly, so no deferred
/// window is written back as a side effect of measuring. Returns `false` on
/// the first page that is absent, invalid, or unreadable — a partial read is
/// not comparable, and reporting it as a divergence would invent one.
fn read_guest_span_unflushed<H: HostMemory + HostOps>(
    state: &DeviceState,
    host: &mut H,
    mapping_id: u32,
    off: u64,
    buf: &mut [u8],
) -> bool {
    if buf.is_empty() {
        return true;
    }
    let Some(m) = state.mappings.get(&mapping_id) else {
        return false;
    };
    let page_size = state.page_size();
    if page_size == 0 {
        return false;
    }
    let mut done = 0u64;
    let total = buf.len() as u64;
    while done < total {
        let cur = off + done;
        let page_index = (cur / page_size) as usize;
        let page_off = cur % page_size;
        let Some(&entry) = m.page_entries.get(page_index) else {
            return false;
        };
        let Some(gpa) = crate::contract::iosurface_pages::entry_gpa_shift(entry, state.page_shift)
        else {
            return false;
        };
        let take = (page_size - page_off).min(total - done);
        let dst = &mut buf[done as usize..(done + take) as usize];
        if host.read_gpa(gpa + page_off, dst).is_err() {
            return false;
        }
        done += take;
    }
    true
}

/// Deferred windows on this mapping that intersect `[lo, hi)`.
///
/// A pending window means the guest's pages are *legitimately* behind our
/// resident — that is the deferred-writeback contract, not a lost render.
/// The probe reports the count rather than flushing, so a reader can tell the
/// two apart without the measurement changing what it measures.
fn pending_deferred_windows(state: &DeviceState, mapping_id: u32, lo: u64, hi: u64) -> usize {
    state
        .render_deferred_flush
        .keys()
        .filter(|k| k.mapping_id == mapping_id && k.surface_offset < hi && k.span_end > lo)
        .count()
}

/// Why a present produced no comparison. Named so the log never has to say
/// "no divergence" when what happened is "nothing was compared".
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DivergenceSkip {
    /// Route B: the dmabuf carried the frame, so there are no CPU pixels.
    NoCpuFrame,
    /// The captured frame is not `w * h * 4` — geometry moved under us.
    FrameGeometry,
    /// No sample window: the mapping carries neither a device descriptor
    /// plane nor an inventable packed layout for this geometry.
    NoSampleWindow,
    /// The mapping's bytes-per-row is below one tight row of pixels.
    BprBelowTight,
    /// A sampled row's guest pages were absent or unreadable.
    GuestRowUnreadable,
}

impl DivergenceSkip {
    pub fn as_str(self) -> &'static str {
        match self {
            DivergenceSkip::NoCpuFrame => "no_cpu_frame",
            DivergenceSkip::FrameGeometry => "frame_geometry",
            DivergenceSkip::NoSampleWindow => "no_sample_window",
            DivergenceSkip::BprBelowTight => "bpr_below_tight",
            DivergenceSkip::GuestRowUnreadable => "guest_row_unreadable",
        }
    }
}

/// What one present's comparison found.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PresentDivergence {
    pub div: RowDivergence,
    /// Rows actually compared (0..=`SAMPLE_ROW_NUMERATORS.len()`).
    pub rows: u32,
    /// Deferred render windows intersecting the sampled rows.
    pub pending: usize,
    /// The sample window came from the guest's own device descriptor
    /// (`true`) rather than an invented packed layout (`false`). An invented
    /// window can name the wrong bytes, so a divergence measured over one is
    /// a weaker reading — and the flag has to come from the resolver, not
    /// from the caller's belief about it.
    pub from_device: bool,
}

/// Compare the captured present frame against the presented mapping's guest
/// pages. `None` with the reason when no comparison was possible.
pub fn measure<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    mapping_id: u32,
    width: u32,
    height: u32,
) -> Result<PresentDivergence, DivergenceSkip> {
    if state.present.frame_bgra.is_empty() {
        return Err(DivergenceSkip::NoCpuFrame);
    }
    let tight = (width as usize).saturating_mul(4);
    let need = tight.saturating_mul(height as usize);
    if need == 0 || state.present.frame_bgra.len() != need {
        return Err(DivergenceSkip::FrameGeometry);
    }
    let fmt = state
        .mappings
        .get(&mapping_id)
        .map(|m| {
            if m.format != 0 {
                m.format
            } else {
                crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM
            }
        })
        .unwrap_or(crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM);
    let Some((base_off, bpr, _span_end, from_device)) =
        state.mappings.get(&mapping_id).and_then(|m| {
            crate::runtime::mapping_write::type11_sample_window_ex(m, width, height, fmt)
        })
    else {
        return Err(DivergenceSkip::NoSampleWindow);
    };
    if (bpr as usize) < tight {
        return Err(DivergenceSkip::BprBelowTight);
    }

    let mut out = PresentDivergence {
        div: RowDivergence::default(),
        rows: 0,
        pending: 0,
        from_device,
    };
    let mut guest_row = vec![0u8; tight];
    for num in SAMPLE_ROW_NUMERATORS {
        let y = height.saturating_mul(num) / SAMPLE_ROW_DENOMINATOR;
        if y >= height {
            continue;
        }
        let off = base_off.saturating_add((y as u64).saturating_mul(bpr as u64));
        out.pending += pending_deferred_windows(state, mapping_id, off, off + tight as u64);
        if !read_guest_span_unflushed(state, host, mapping_id, off, &mut guest_row) {
            return Err(DivergenceSkip::GuestRowUnreadable);
        }
        let ours = &state.present.frame_bgra[(y as usize) * tight..][..tight];
        out.div.merge(compare_bgra_rows(ours, &guest_row));
        out.rows += 1;
    }
    if out.rows == 0 {
        return Err(DivergenceSkip::FrameGeometry);
    }
    Ok(out)
}

/// Emission cadence. This paces the log only — nothing on the present path
/// reads it, and no decision changes with it. Per-present emission would put
/// a line under every frame at the guest's refresh rate and trip the sink's
/// own flood detector; per-window keeps the worst case and the rate.
const WINDOW_MS: u128 = 2000;

#[derive(Default)]
struct Window {
    presents: u32,
    compared: u32,
    visible: u32,
    worst: RowDivergence,
    worst_mid: u32,
    worst_pending: usize,
    invented: u32,
    pending_any: u32,
    skips: Vec<(DivergenceSkip, u32)>,
}

impl Window {
    fn note_skip(&mut self, why: DivergenceSkip) {
        if let Some(slot) = self.skips.iter_mut().find(|(w, _)| *w == why) {
            slot.1 += 1;
        } else {
            self.skips.push((why, 1));
        }
    }

    fn skips_str(&self) -> String {
        self.skips
            .iter()
            .map(|(w, n)| format!("{}:{n}", w.as_str()))
            .collect::<Vec<_>>()
            .join(",")
    }

    /// Fold one present's outcome into the window.
    fn note(&mut self, mapping_id: u32, measured: Result<PresentDivergence, DivergenceSkip>) {
        self.presents += 1;
        let Ok(d) = measured else {
            self.note_skip(measured.unwrap_err());
            return;
        };
        self.compared += 1;
        self.invented += u32::from(!d.from_device);
        self.pending_any += u32::from(d.pending > 0);
        self.visible += u32::from(d.div.is_visible());
        // Worst by the magnitude that matters, not by pixel count: a frame with
        // a million pixels off by 2 has not lost guest work and one with a
        // thousand off by 200 has.
        if d.div.max > self.worst.max
            || (d.div.max == self.worst.max && d.div.gt64 > self.worst.gt64)
        {
            self.worst = d.div;
            self.worst_mid = mapping_id;
            self.worst_pending = d.pending;
        }
    }

    /// The window's line, and whether it is a failure.
    ///
    /// A window that saw a visible divergence is reporting the frame on screen
    /// disagreeing with the frame the guest holds for the surface it named.
    /// Nothing on the present path can correct that — the resident is the only
    /// copy we read — so it is a loss of guest work reaching the display, not a
    /// transient, and it belongs on the always-on failure channel.
    fn line(&self, dt: u128) -> (String, bool) {
        (
            format!(
                "present_vs_guest window_ms={dt} presents={} compared={} visible={} \
                 worst_mid={} px={} gt1={} gt4={} gt16={} gt64={} max={} max_swapped={} \
                 pending={} pending_presents={} invented={} skips=[{}]",
                self.presents,
                self.compared,
                self.visible,
                self.worst_mid,
                self.worst.px,
                self.worst.gt1,
                self.worst.gt4,
                self.worst.gt16,
                self.worst.gt64,
                self.worst.max,
                self.worst.max_swapped,
                self.worst_pending,
                self.pending_any,
                self.invented,
                self.skips_str()
            ),
            self.visible > 0,
        )
    }
}

/// Measure this present and emit the window summary when the window closes.
///
/// Call at the present boundary, after the capture that produced the frame we
/// are about to show. Measure-only: the return value is discarded, nothing
/// downstream reads any field, and the guest's pages are read but never
/// written (see the module docs on why this does not go through
/// `mapper::read_mapping_bytes`).
pub fn note_present<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    mapping_id: u32,
    width: u32,
    height: u32,
) {
    use std::sync::Mutex;
    use std::time::Instant;
    static WINDOW: Mutex<Option<(Instant, Window)>> = Mutex::new(None);

    let measured = measure(state, host, mapping_id, width, height);
    let mut guard = WINDOW.lock().unwrap_or_else(|p| p.into_inner());
    let (started, w) = guard.get_or_insert_with(|| (Instant::now(), Window::default()));
    w.note(mapping_id, measured);
    if started.elapsed().as_millis() < WINDOW_MS {
        return;
    }
    let Some((started, w)) = guard.take() else {
        return;
    };
    let (line, is_fail) = w.line(started.elapsed().as_millis());
    if is_fail {
        crate::observe::fail(line);
    } else {
        crate::observe::off(line);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
    use crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    use crate::model::{DeviceId, PAGE_SHIFT_X86};
    use crate::runtime::host::FakeHost;

    const W: u32 = 64;
    const H: u32 = 8;

    /// A mapping whose guest pages are real host memory, filled with `fill`,
    /// plus a captured present frame of the same geometry filled with `ours`.
    fn rig(guest_fill: u8, ours_fill: u8) -> (DeviceState, FakeHost) {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        let tight = (W as usize) * 4;
        let need = tight * (H as usize);
        let page_size = 1u64 << PAGE_SHIFT_X86;
        let pages = (need as u64).div_ceil(page_size) as usize;
        let mut entries = Vec::with_capacity(pages);
        for i in 0..pages {
            let pfn = 0x200u32 + i as u32;
            let gpa = (pfn as u64) << PAGE_SHIFT_X86;
            host.map_range(gpa, page_size as usize, guest_fill);
            entries.push(((pfn as u64) << PAGE_ENTRY_PFN_SHIFT) as u32 | PAGE_ENTRY_VALID);
        }
        assert!(state.map_surface(1));
        {
            let m = state.mappings.get_mut(&1).unwrap();
            m.mapped = true;
            m.page_entries = entries;
        }
        assert!(state.set_mapping_geom(1, W, H, MTL_FORMAT_BGRA8_UNORM));
        state.present.frame_bgra = vec![ours_fill; need];
        (state, host)
    }

    /// The whole point of the probe: the frame we are about to show differs
    /// from the guest's own copy of the surface the guest named, and the probe
    /// says so with a magnitude.
    #[test]
    fn a_constructed_disagreement_is_reported_with_its_magnitude() {
        let (mut state, mut host) = rig(0x10, 0xf0);
        let d = super::measure(&mut state, &mut host, 1, W, H).expect("comparison must run");
        assert_eq!(d.rows, 3, "three spread rows must be read");
        assert_eq!(d.div.px, (W as usize) * 3);
        assert_eq!(d.div.max, 0xe0);
        assert_eq!(d.div.gt64, (W as usize) * 3);
        assert!(d.div.is_visible());
        assert_eq!(d.pending, 0, "no deferred window was armed");
    }

    /// The converse, which is what makes the disagreement above mean anything:
    /// with the same rig and matching bytes the probe reads clean. Without
    /// this, a probe that always fired would pass the test above.
    #[test]
    fn matching_content_reads_clean_on_the_same_rig() {
        let (mut state, mut host) = rig(0x77, 0x77);
        let d = super::measure(&mut state, &mut host, 1, W, H).expect("comparison must run");
        assert_eq!(d.rows, 3);
        assert_eq!(d.div.max, 0, "identical bytes must not report a deviation");
        assert!(!d.div.is_visible());
    }

    /// A divergence confined to one band must still be found, because that is
    /// the shape the class takes: a dead window's rect, not a whole frame.
    #[test]
    fn a_band_that_crosses_one_sampled_row_is_found() {
        let (mut state, mut host) = rig(0x20, 0x20);
        // Row H/2 only — the middle of the three sampled rows.
        let tight = (W as usize) * 4;
        let y = (H / 2) as usize;
        for b in &mut state.present.frame_bgra[y * tight..(y + 1) * tight] {
            *b = 0xd0;
        }
        let d = super::measure(&mut state, &mut host, 1, W, H).expect("comparison must run");
        assert_eq!(d.div.max, 0xb0);
        assert_eq!(
            d.div.gt64, W as usize,
            "exactly the one sampled row that crosses the band"
        );
    }

    /// Route B leaves no CPU pixels. The probe must say "nothing was compared"
    /// rather than "no divergence" — the two are not the same reading, and a
    /// silent zero here would report a dmabuf present as agreement.
    #[test]
    fn a_dmabuf_present_is_a_skip_and_not_a_clean_reading() {
        let (mut state, mut host) = rig(0x10, 0xf0);
        state.present.frame_bgra.clear();
        assert_eq!(
            super::measure(&mut state, &mut host, 1, W, H),
            Err(DivergenceSkip::NoCpuFrame)
        );
    }

    /// The window that a visible divergence produces must reach the failure
    /// channel and carry its magnitude. Without this the measurement above can
    /// be perfect and still never be seen, which is the shape of a probe that
    /// looks healthy and reports nothing.
    #[test]
    fn a_visible_divergence_closes_the_window_on_the_failure_channel() {
        let (mut state, mut host) = rig(0x10, 0xf0);
        let measured = super::measure(&mut state, &mut host, 1, W, H);
        let mut w = Window::default();
        w.note(1, measured);
        let (line, is_fail) = w.line(2001);
        assert!(is_fail, "a visible divergence must be a failure, not census");
        assert!(line.contains("presents=1 compared=1 visible=1"), "{line}");
        assert!(line.contains("worst_mid=1"), "{line}");
        assert!(line.contains(&format!("gt64={}", (W as usize) * 3)), "{line}");
        assert!(line.contains("max=224"), "{line}");
        assert!(line.contains("skips=[]"), "{line}");
    }

    /// The converse for the emission half: agreement is a census line, and a
    /// skip is neither. A window of nothing-but-skips must not read as a clean
    /// present path — the counts have to say nothing was compared.
    #[test]
    fn agreement_is_census_and_a_skip_is_neither_reading() {
        let (mut state, mut host) = rig(0x77, 0x77);
        let measured = super::measure(&mut state, &mut host, 1, W, H);
        let mut w = Window::default();
        w.note(1, measured);
        let (line, is_fail) = w.line(2000);
        assert!(!is_fail, "agreement is not a failure");
        assert!(line.contains("visible=0"), "{line}");
        assert!(line.contains("max=0"), "{line}");

        let mut w = Window::default();
        w.note(1, Err(DivergenceSkip::NoCpuFrame));
        w.note(1, Err(DivergenceSkip::NoCpuFrame));
        let (line, is_fail) = w.line(2000);
        assert!(!is_fail);
        assert!(
            line.contains("presents=2 compared=0"),
            "a skipped window must not read as two clean presents: {line}"
        );
        assert!(line.contains("skips=[no_cpu_frame:2]"), "{line}");
    }

    /// Unreadable guest pages are a skip for the same reason: a partial read
    /// compared against a full frame would invent a divergence.
    #[test]
    fn missing_guest_pages_are_a_skip() {
        let (mut state, mut host) = rig(0x10, 0xf0);
        state.mappings.get_mut(&1).unwrap().page_entries.clear();
        assert_eq!(
            super::measure(&mut state, &mut host, 1, W, H),
            Err(DivergenceSkip::GuestRowUnreadable)
        );
    }

    #[test]
    fn identical_rows_diverge_nowhere() {
        let row = vec![0x40u8; 64];
        let d = compare_bgra_rows(&row, &row);
        assert_eq!(d.px, 16);
        assert_eq!(d.max, 0);
        assert_eq!(d.gt1, 0);
        assert!(!d.is_visible());
    }

    #[test]
    fn a_one_lsb_difference_is_not_visible_but_a_full_swing_is() {
        // The whole reason the histogram exists: these two must not read alike.
        let ours = vec![0x80u8; 64];
        let mut rounding = ours.clone();
        for px in rounding.chunks_exact_mut(4) {
            px[1] = 0x81;
        }
        let d = compare_bgra_rows(&ours, &rounding);
        assert_eq!(d.max, 1);
        assert_eq!(d.gt1, 0);
        assert!(!d.is_visible());

        let stale = vec![0x00u8; 64];
        let d = compare_bgra_rows(&ours, &stale);
        assert_eq!(d.max, 0x80);
        assert_eq!(d.gt64, 16);
        assert!(d.is_visible());
    }

    #[test]
    fn a_channel_swap_reads_large_direct_and_zero_swapped() {
        let mut ours = Vec::new();
        let mut guest = Vec::new();
        for i in 0..16u8 {
            let (b, g, r) = (i * 8, 0x20, 0xf0 - i * 4);
            ours.extend_from_slice(&[b, g, r, 0xff]);
            guest.extend_from_slice(&[r, g, b, 0xff]);
        }
        let d = compare_bgra_rows(&ours, &guest);
        assert!(d.max > 64, "a channel swap must not read as agreement");
        assert_eq!(
            d.max_swapped, 0,
            "swapped comparison must recover the match, or the probe cannot \
             tell a channel-order bug from a stale resident"
        );
    }

    #[test]
    fn alpha_is_excluded() {
        let ours = vec![0x10, 0x20, 0x30, 0x00];
        let guest = vec![0x10, 0x20, 0x30, 0xff];
        assert_eq!(compare_bgra_rows(&ours, &guest).max, 0);
    }

    #[test]
    fn merge_keeps_worst_case_and_sums_counts() {
        let mut a = RowDivergence {
            px: 10,
            gt1: 3,
            gt4: 2,
            gt16: 1,
            gt64: 0,
            max: 20,
            max_swapped: 5,
        };
        a.merge(RowDivergence {
            px: 10,
            gt1: 1,
            gt4: 1,
            gt16: 1,
            gt64: 1,
            max: 200,
            max_swapped: 2,
        });
        assert_eq!(a.px, 20);
        assert_eq!(a.gt64, 1);
        assert_eq!(a.max, 200);
        assert_eq!(a.max_swapped, 5);
    }
}
