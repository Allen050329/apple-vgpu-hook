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

/// Per-channel deviation between the frame we are showing and the guest's
/// pages, over the **whole frame**.
///
/// It compares every pixel rather than a row grid, and that is not thoroughness
/// for its own sake. A first version of this probe sampled three rows at h/4,
/// h/2 and 3h/4; the residue it was built to find landed in a 61-pixel band at
/// y=618..679, which all three rows miss. It would have reported agreement
/// across the exact capture that scored 32 403 differing pixels on the host
/// window, and the reading would have looked like a result. A readout grid
/// inherits the geometry of whatever else is wrong in the frame, so there is
/// no grid.
///
/// `px` is the number of pixels compared, so every count below has a
/// denominator on the same line. `gt*` are pixel counts whose largest
/// colour-channel deviation exceeds the named threshold; `max` is the largest
/// deviation anywhere.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FrameDivergence {
    pub px: usize,
    pub gt1: usize,
    pub gt4: usize,
    pub gt16: usize,
    pub gt64: usize,
    pub max: u8,
    /// Largest deviation after swapping the guest pixel's first and third
    /// channels. A channel-order mismatch reads as a large `max` with a small
    /// `max_swapped`; a stale resident is large in both. Without this the two
    /// are indistinguishable, and this project has confused them before.
    pub max_swapped: u8,
    /// Bounding box of the pixels over the visibility threshold, as
    /// `(x0, y0, x1, y1)` half-open. Directly comparable to the `bbox` the
    /// screen-capture differ prints, which is what makes a log line and a
    /// screenshot the same claim.
    pub bbox: Option<(u32, u32, u32, u32)>,
}

impl FrameDivergence {
    /// Whether anything above sub-perceptual rounding was seen. 4/255 is the
    /// ceiling this project measured for a pure re-encode difference over a
    /// whole frame; below it there is nothing a human could see.
    pub fn is_visible(&self) -> bool {
        self.gt4 > 0
    }

    pub fn bbox_str(&self) -> String {
        match self.bbox {
            Some((x0, y0, x1, y1)) => format!("{}x{}+{x0}+{y0}", x1 - x0, y1 - y0),
            None => "-".to_string(),
        }
    }
}

/// Compare two tight BGRA8 frames of `width` x `height`.
///
/// Both sides are wire order (BGRA); alpha is excluded because the present
/// path and the guest's own compositor do not agree on it by contract and a
/// difference there is not a visible defect.
pub fn compare_bgra_frames(ours: &[u8], guest: &[u8], width: u32) -> FrameDivergence {
    let mut d = FrameDivergence::default();
    let (mut x0, mut y0, mut x1, mut y1) = (u32::MAX, u32::MAX, 0u32, 0u32);
    for (i, (a, b)) in ours.chunks_exact(4).zip(guest.chunks_exact(4)).enumerate() {
        let dev = a[0]
            .abs_diff(b[0])
            .max(a[1].abs_diff(b[1]))
            .max(a[2].abs_diff(b[2]));
        let dev_swapped = a[0]
            .abs_diff(b[2])
            .max(a[1].abs_diff(b[1]))
            .max(a[2].abs_diff(b[0]));
        d.px += 1;
        d.gt1 += usize::from(dev > 1);
        d.gt16 += usize::from(dev > 16);
        d.gt64 += usize::from(dev > 64);
        d.max = d.max.max(dev);
        d.max_swapped = d.max_swapped.max(dev_swapped);
        if dev > 4 {
            d.gt4 += 1;
            let (x, y) = ((i as u32) % width, (i as u32) / width);
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x + 1);
            y1 = y1.max(y + 1);
        }
    }
    if d.gt4 > 0 {
        d.bbox = Some((x0, y0, x1, y1));
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
    /// Route B: the dmabuf carried the frame, so there are no CPU pixels, and
    /// this present was not the one that pays for a resident readback.
    NoCpuFrame,
    /// Route B, readback attempted, and the resident the present path would
    /// have read is not there to read: unknown identity, no ready content, or
    /// a non-BGRA target.
    NoResident,
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
            DivergenceSkip::NoResident => "no_resident",
            DivergenceSkip::FrameGeometry => "frame_geometry",
            DivergenceSkip::NoSampleWindow => "no_sample_window",
            DivergenceSkip::BprBelowTight => "bpr_below_tight",
            DivergenceSkip::GuestRowUnreadable => "guest_pages_unreadable",
        }
    }
}

/// Which copy of the surface stood in for "what we are showing".
///
/// Route A fills `frame_bgra` and the window blits it, so that buffer *is* the
/// frame. Route B publishes the GPU resident as a dmabuf and
/// `capture_present_frame` deliberately skips the CPU readback — there, the
/// resident is the frame and `frame_bgra` holds nothing. A probe that only
/// read `frame_bgra` would therefore skip every present on the live
/// x86/Vulkan host-window rail, which is exactly what the first boot of this
/// line measured: 82 of 82 presents in one window, `skips=[no_cpu_frame:82]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OurSide {
    /// `state.present.frame_bgra`, already in hand — free to compare.
    Captured,
    /// A GPU→host readback of the resident, resolved through the same
    /// identity the capture path uses. Costs a full-frame readback, so it is
    /// taken once per window rather than once per present.
    Resident,
}

impl OurSide {
    pub fn as_str(self) -> &'static str {
        match self {
            OurSide::Captured => "captured",
            OurSide::Resident => "resident",
        }
    }
}

/// What one present's comparison found.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PresentDivergence {
    pub div: FrameDivergence,
    /// Which copy of ours the guest's pages were compared against.
    pub src: OurSide,
    /// Deferred render windows intersecting the compared surface span.
    pub pending: usize,
    /// The sample window came from the guest's own device descriptor
    /// (`true`) rather than an invented packed layout (`false`). An invented
    /// window can name the wrong bytes, so a divergence measured over one is
    /// a weaker reading — and the flag has to come from the resolver, not
    /// from the caller's belief about it.
    pub from_device: bool,
}

/// Read the resident the present path would have read for this surface.
///
/// Resolves the identity exactly as `scanout::try_capture_from_resident` does,
/// primary then member, because the comparison is only meaningful against the
/// same target the present would have shown.
#[cfg(feature = "backend-vulkan")]
fn read_resident_frame(
    state: &mut DeviceState,
    mapping_id: u32,
    width: u32,
    height: u32,
    need: usize,
) -> Option<Vec<u8>> {
    use crate::backend::vulkan::engine::read_resident_bgra;
    let identity =
        crate::runtime::import_present::surface_identity(state, mapping_id, width, height);
    let member =
        crate::runtime::import_present::member_surface_identity(state, mapping_id, width, height);
    read_resident_bgra(&identity, need).or_else(|| {
        (member != identity)
            .then(|| read_resident_bgra(&member, need))
            .flatten()
    })
}

#[cfg(not(feature = "backend-vulkan"))]
fn read_resident_frame(
    _state: &mut DeviceState,
    _mapping_id: u32,
    _width: u32,
    _height: u32,
    _need: usize,
) -> Option<Vec<u8>> {
    None
}

/// Compare the frame we are showing against the presented mapping's guest
/// pages. `Err` with the typed reason when no comparison was possible.
///
/// `allow_resident` pays for a full-frame GPU→host readback when route B left
/// no CPU pixels. The caller takes it once per emission window, not once per
/// present: on the live rail every present is route B, so without it this
/// probe reads nothing at all, and with it on every present it would put an
/// 8 MiB readback under each frame on the path route B exists to keep clear.
pub fn measure<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    mapping_id: u32,
    width: u32,
    height: u32,
    allow_resident: bool,
) -> Result<PresentDivergence, DivergenceSkip> {
    let tight = (width as usize).saturating_mul(4);
    let need = tight.saturating_mul(height as usize);
    if need == 0 {
        return Err(DivergenceSkip::FrameGeometry);
    }
    let (ours, src) = if state.present.frame_bgra.len() == need {
        (None, OurSide::Captured)
    } else if !state.present.frame_bgra.is_empty() {
        return Err(DivergenceSkip::FrameGeometry);
    } else if !allow_resident {
        return Err(DivergenceSkip::NoCpuFrame);
    } else {
        let Some(px) = read_resident_frame(state, mapping_id, width, height, need) else {
            return Err(DivergenceSkip::NoResident);
        };
        if px.len() != need {
            return Err(DivergenceSkip::FrameGeometry);
        }
        (Some(px), OurSide::Resident)
    };
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

    // The guest's rows are `bpr` apart and ours are tight, so the guest side is
    // gathered row by row into a tight frame before the compare.
    let mut guest = vec![0u8; need];
    for y in 0..height {
        let off = base_off.saturating_add((y as u64).saturating_mul(bpr as u64));
        let at = (y as usize) * tight;
        if !read_guest_span_unflushed(state, host, mapping_id, off, &mut guest[at..at + tight]) {
            return Err(DivergenceSkip::GuestRowUnreadable);
        }
    }
    let span_lo = base_off;
    let span_hi = base_off.saturating_add((height as u64).saturating_mul(bpr as u64));
    let mut out = PresentDivergence {
        div: FrameDivergence::default(),
        src,
        pending: pending_deferred_windows(state, mapping_id, span_lo, span_hi),
        from_device,
    };
    let ours = match &ours {
        Some(px) => px.as_slice(),
        None => state.present.frame_bgra.as_slice(),
    };
    out.div = compare_bgra_frames(ours, &guest, width);
    Ok(out)
}

/// Emission cadence. This paces the log only — nothing on the present path
/// reads it, and no decision changes with it. Per-present emission would put
/// a line under every frame at the guest's refresh rate and trip the sink's
/// own flood detector; per-window keeps the worst case and the rate.
const WINDOW_MS: u128 = 2000;

/// Worst case over a set of comparisons, with the count they came from.
///
/// `compared` is the denominator every other number here needs; without it a
/// window that compared nothing and a window that compared ten clean frames
/// both print zeros.
#[derive(Default)]
struct Agg {
    compared: u32,
    visible: u32,
    worst: FrameDivergence,
    worst_mid: u32,
}

impl Agg {
    fn note(&mut self, mapping_id: u32, d: &PresentDivergence) {
        self.compared += 1;
        self.visible += u32::from(d.div.is_visible());
        // Worst by the magnitude that matters, not by pixel count: a frame with
        // a million pixels off by 2 has not lost guest work and one with a
        // thousand off by 200 has. The first comparison always takes the slot,
        // so a clean window still reports the pixel count it read.
        if self.compared == 1
            || d.div.max > self.worst.max
            || (d.div.max == self.worst.max && d.div.gt64 > self.worst.gt64)
        {
            self.worst = d.div;
            self.worst_mid = mapping_id;
        }
    }

    fn fields(&self, prefix: &str) -> String {
        format!(
            "{prefix}cmp={} {prefix}vis={} {prefix}mid={} {prefix}px={} {prefix}gt1={} \
             {prefix}gt4={} {prefix}gt16={} {prefix}gt64={} {prefix}max={} {prefix}swap={} \
             {prefix}bbox={}",
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
            self.worst.bbox_str()
        )
    }
}

#[derive(Default)]
struct Window {
    presents: u32,
    /// Every comparison, including those with a deferred window armed.
    all: Agg,
    /// Only comparisons with **no** deferred render window over the sampled
    /// rows. This is the reading that means something: with an obligation
    /// outstanding the guest's pages are supposed to be behind us, so a
    /// divergence there is the deferred-writeback contract working. With none
    /// outstanding, the two copies of the surface have no licence to differ.
    settled: Agg,
    invented: u32,
    /// Comparisons that read our GPU resident rather than a captured frame.
    resident_reads: u32,
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
        self.resident_reads += u32::from(d.src == OurSide::Resident);
        self.invented += u32::from(!d.from_device);
        self.all.note(mapping_id, &d);
        if d.pending == 0 {
            self.settled.note(mapping_id, &d);
        }
    }

    /// The window's line, and whether it is a failure.
    ///
    /// Only the **settled** subset can fail. A window whose divergences all
    /// carried an armed deferred window is reporting the deferred-writeback
    /// rail doing exactly what it says: the pinned resident is authoritative
    /// and the guest window is stale until something reads it. Failing on that
    /// would put a line under most frames of a healthy boot and bury the case
    /// that matters — a settled divergence, where our copy and the guest's
    /// disagree with no obligation outstanding to explain it.
    fn line(&self, dt: u128) -> (String, bool) {
        (
            format!(
                "present_vs_guest window_ms={dt} presents={} resident={} invented={} \
                 {} {} skips=[{}]",
                self.presents,
                self.resident_reads,
                self.invented,
                self.settled.fields("settled_"),
                self.all.fields("all_"),
                self.skips_str()
            ),
            self.settled.visible > 0,
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

    let mut guard = WINDOW.lock().unwrap_or_else(|p| p.into_inner());
    // The window's last present is the one that pays for a resident readback,
    // so route B gets exactly one comparison per window instead of none.
    let closing = guard
        .get_or_insert_with(|| (Instant::now(), Window::default()))
        .0
        .elapsed()
        .as_millis()
        >= WINDOW_MS;
    let measured = measure(state, host, mapping_id, width, height, closing);
    if let Some((_, w)) = guard.as_mut() {
        w.note(mapping_id, measured);
    }
    if !closing {
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
    const H: u32 = 16;

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
        let d = super::measure(&mut state, &mut host, 1, W, H, false).expect("comparison must run");
        assert_eq!(d.div.px, (W as usize) * (H as usize), "every pixel compared");
        assert_eq!(d.div.max, 0xe0);
        assert_eq!(d.div.gt64, (W as usize) * (H as usize));
        assert_eq!(d.div.bbox, Some((0, 0, W, H)));
        assert!(d.div.is_visible());
        assert_eq!(d.pending, 0, "no deferred window was armed");
    }

    /// The converse, which is what makes the disagreement above mean anything:
    /// with the same rig and matching bytes the probe reads clean. Without
    /// this, a probe that always fired would pass the test above.
    #[test]
    fn matching_content_reads_clean_on_the_same_rig() {
        let (mut state, mut host) = rig(0x77, 0x77);
        let d = super::measure(&mut state, &mut host, 1, W, H, false).expect("comparison must run");
        assert_eq!(d.div.px, (W as usize) * (H as usize));
        assert_eq!(d.div.max, 0, "identical bytes must not report a deviation");
        assert_eq!(d.div.bbox, None);
        assert!(!d.div.is_visible());
    }

    /// A band anywhere in the frame must be found and located, because that is
    /// the shape the class takes: a dead window's rect, not a whole frame.
    ///
    /// This test exists because the first version of this probe would have
    /// failed it. It sampled y = H/4, H/2, 3H/4; the band that reproduced on
    /// the rig sat at y = 618..679 of 1080 and none of those rows crosses it.
    /// The band here is deliberately placed off every such fraction.
    #[test]
    fn a_band_that_no_row_grid_would_cross_is_found_and_located() {
        let (mut state, mut host) = rig(0x20, 0x20);
        let tight = (W as usize) * 4;
        let (y_lo, y_hi) = (5usize, 7usize);
        for y in y_lo..y_hi {
            for b in &mut state.present.frame_bgra[y * tight..(y + 1) * tight] {
                *b = 0xd0;
            }
        }
        assert!(
            ![1, 2, 3].iter().any(|n| (H as usize) * n / 4 >= y_lo
                && (H as usize) * n / 4 < y_hi),
            "the band must miss every quarter row, or this test proves nothing"
        );
        let d = super::measure(&mut state, &mut host, 1, W, H, false).expect("comparison must run");
        assert_eq!(d.div.max, 0xb0);
        assert_eq!(d.div.gt64, (W as usize) * (y_hi - y_lo));
        assert_eq!(
            d.div.bbox,
            Some((0, y_lo as u32, W, y_hi as u32)),
            "the bbox must name where the divergence is, not just that there is one"
        );
        assert_eq!(d.div.bbox_str(), format!("{W}x2+0+{y_lo}"));
    }

    /// Route B leaves no CPU pixels. The probe must say "nothing was compared"
    /// rather than "no divergence" — the two are not the same reading, and a
    /// silent zero here would report a dmabuf present as agreement.
    ///
    /// This is not hypothetical: the first live boot of this line read
    /// `presents=82 compared=0 skips=[no_cpu_frame:82]`, because every present
    /// on the x86/Vulkan host-window rail is route B. The `skips=` field is
    /// the only reason that read as "the probe cannot fire" instead of "the
    /// present path is clean".
    #[test]
    fn a_dmabuf_present_without_a_readback_budget_is_a_skip_not_a_clean_reading() {
        let (mut state, mut host) = rig(0x10, 0xf0);
        state.present.frame_bgra.clear();
        assert_eq!(
            super::measure(&mut state, &mut host, 1, W, H, false),
            Err(DivergenceSkip::NoCpuFrame)
        );
    }

    /// With the readback budget the probe reaches for the resident instead —
    /// and when there is no resident to read (no GPU engine in a unit test),
    /// that is its own typed skip. `no_resident` and `no_cpu_frame` must not
    /// collapse: the first says the readback was tried and found nothing, the
    /// second says it was never tried.
    #[test]
    fn a_dmabuf_present_with_a_readback_budget_reaches_for_the_resident() {
        let (mut state, mut host) = rig(0x10, 0xf0);
        state.present.frame_bgra.clear();
        assert_eq!(
            super::measure(&mut state, &mut host, 1, W, H, true),
            Err(DivergenceSkip::NoResident)
        );
    }

    /// A captured frame is used whether or not the readback budget is granted,
    /// so route A never pays for a readback it does not need — and the line
    /// says which copy it read.
    #[test]
    fn a_captured_frame_is_preferred_over_a_readback() {
        let (mut state, mut host) = rig(0x10, 0xf0);
        let d = super::measure(&mut state, &mut host, 1, W, H, true).expect("comparison must run");
        assert_eq!(d.src, OurSide::Captured);
        assert_eq!(d.div.max, 0xe0);
    }

    /// A settled divergence — the two copies disagree with no deferred window
    /// outstanding to explain it — must reach the failure channel and carry its
    /// magnitude. Without this the measurement can be perfect and still never
    /// be seen, which is the shape of a probe that looks healthy and reports
    /// nothing.
    #[test]
    fn a_settled_divergence_closes_the_window_on_the_failure_channel() {
        let (mut state, mut host) = rig(0x10, 0xf0);
        let measured = super::measure(&mut state, &mut host, 1, W, H, false);
        let mut w = Window::default();
        w.note(1, measured);
        let (line, is_fail) = w.line(2001);
        assert!(is_fail, "a settled divergence must be a failure, not census");
        // `invented=1`: the fixture mapping carries no device descriptor, so
        // the sample window is the packed fallback. The flag comes from the
        // resolver that decided, which is the point of carrying it.
        assert!(line.contains("presents=1 resident=0 invented=1"), "{line}");
        assert!(line.contains("settled_cmp=1 settled_vis=1"), "{line}");
        assert!(line.contains("settled_mid=1"), "{line}");
        assert!(
            line.contains(&format!("settled_gt64={}", (W as usize) * (H as usize))),
            "{line}"
        );
        assert!(line.contains("settled_max=224"), "{line}");
        assert!(line.contains("all_cmp=1 all_vis=1"), "{line}");
        assert!(line.contains("skips=[]"), "{line}");
    }

    /// A divergence with a deferred window armed over the surface span is the
    /// deferred-writeback contract, not a lost render: the pinned resident is
    /// authoritative and the guest's pages are stale until something reads
    /// them. It must be counted and must NOT fail — on the live rail nearly
    /// every present has one, and failing on it would bury the settled case.
    #[test]
    fn a_divergence_with_a_deferred_window_armed_is_counted_but_does_not_fail() {
        let d = PresentDivergence {
            div: FrameDivergence {
                px: 100,
                gt1: 100,
                gt4: 100,
                gt16: 100,
                gt64: 100,
                max: 255,
                max_swapped: 255,
                bbox: Some((0, 0, 10, 10)),
            },
            src: OurSide::Resident,
            pending: 3,
            from_device: true,
        };
        let mut w = Window::default();
        w.note(6, Ok(d));
        let (line, is_fail) = w.line(2000);
        assert!(!is_fail, "the deferred contract is not a failure: {line}");
        assert!(line.contains("all_cmp=1 all_vis=1"), "{line}");
        assert!(line.contains("all_max=255"), "{line}");
        assert!(
            line.contains("settled_cmp=0 settled_vis=0"),
            "an armed window must not enter the settled reading: {line}"
        );
    }

    /// The converse for the emission half: agreement is a census line, and a
    /// skip is neither. A window of nothing-but-skips must not read as a clean
    /// present path — the counts have to say nothing was compared.
    #[test]
    fn agreement_is_census_and_a_skip_is_neither_reading() {
        let (mut state, mut host) = rig(0x77, 0x77);
        let measured = super::measure(&mut state, &mut host, 1, W, H, false);
        let mut w = Window::default();
        w.note(1, measured);
        let (line, is_fail) = w.line(2000);
        assert!(!is_fail, "agreement is not a failure");
        assert!(line.contains("settled_cmp=1 settled_vis=0"), "{line}");
        assert!(line.contains("settled_max=0"), "{line}");
        assert!(
            line.contains(&format!("settled_px={}", (W as usize) * (H as usize))),
            "a clean comparison must still report what it read: {line}"
        );

        let mut w = Window::default();
        w.note(1, Err(DivergenceSkip::NoCpuFrame));
        w.note(1, Err(DivergenceSkip::NoCpuFrame));
        let (line, is_fail) = w.line(2000);
        assert!(!is_fail);
        assert!(
            line.contains("presents=2 ") && line.contains("all_cmp=0"),
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
            super::measure(&mut state, &mut host, 1, W, H, false),
            Err(DivergenceSkip::GuestRowUnreadable)
        );
    }

    #[test]
    fn identical_frames_diverge_nowhere() {
        let row = vec![0x40u8; 64];
        let d = compare_bgra_frames(&row, &row, 16);
        assert_eq!(d.px, 16);
        assert_eq!(d.bbox, None);
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
        let d = compare_bgra_frames(&ours, &rounding, 16);
        assert_eq!(d.max, 1);
        assert_eq!(d.gt1, 0);
        assert!(!d.is_visible());

        let stale = vec![0x00u8; 64];
        let d = compare_bgra_frames(&ours, &stale, 16);
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
        let d = compare_bgra_frames(&ours, &guest, 16);
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
        assert_eq!(compare_bgra_frames(&ours, &guest, 1).max, 0);
    }
}
