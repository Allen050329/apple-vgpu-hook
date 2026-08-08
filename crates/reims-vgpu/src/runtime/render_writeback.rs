//! Land a render Store's frame in the guest's pages, at the Store.
//!
//! A type-11 render Store names a mapping and a resident image the draw just
//! rendered into. This module copies the one into the other and returns. There
//! is no window, no pin held across the call, and nothing to land later.
//!
//! # Why the frame is written here rather than deferred
//!
//! It used to be deferred. A Store armed a window naming the pinned resident,
//! and the window was landed either by the next completion stamp or by a host
//! path that touched the mapping's bytes first. The argument for that shape was
//! coalescing: several passes fully covering one surface inside one submission
//! would land once instead of once each.
//!
//! That coalescing was measured and it never occurred. Arms and lands were
//! equal on every census line of an accumulated driven x86/Vulkan log — 193 458
//! each across 1 780 lines, not one line differing — because no later Store
//! ever fully covered a live window. A ratio pinned at 1.0 by the workload is
//! not coalescing; it is the statement that none was available.
//!
//! What the rail did buy is real and is kept: the Store does not read the frame
//! back off the GPU. [`crate::runtime::mapping_write::write_bgra8_from_resident_gpu`]
//! makes the guest's own pages the destination of the copy the GPU was going to
//! make anyway, so nothing crosses host memory on the arm that runs. Landing at
//! the Store keeps that and drops the window.
//!
//! # What the window cost that this cannot
//!
//! Every hazard the deferred rail had to answer came from the window outliving
//! the Store, and none of them can arise here:
//!
//! * **Resident drift.** A window promised pixels from a slot a later draw
//!   could render over, so the land compared a content epoch and refused on a
//!   mismatch — losing the frame. Here the resident is the one the draw just
//!   produced and nothing runs in between.
//! * **Pin leaks.** A window held a registry pin that the reclaim paths skip by
//!   design, so a pin dropped on any early return stranded a framebuffer for the
//!   guest's lifetime. Nothing is pinned here.
//! * **Page recycling.** The guest could hand a window's pages to a different
//!   allocation before it landed, which is the PTE-corruption class the window
//!   guards existed for. The pages cannot move inside this call.
//! * **Write ordering against the guest's own claim.** A window could hold
//!   pixels rendered *before* a guest CPU write to the same resource, and
//!   landing it afterwards clobbered the guest's bytes with stale ones. The
//!   Store and the write are now ordered by when they happen.
//!
//! # Ordering against the guest
//!
//! The copy is recorded into the engine's command stream, not waited on. It is
//! ordered before the guest can observe it by the completion stamp: the stamp
//! word is written behind an `ALL_COMMANDS -> TRANSFER` barrier and every
//! submitted guest-page write settles before the stamp moves. See
//! `backend::vulkan::engine::write_stamp_after_guest_writes`.

use crate::model::DeviceState;
#[cfg(feature = "backend-vulkan")]
use crate::runtime::host::{HostMemory, HostOps};

/// Declare the settle sites once, and derive both census route names from one
/// slug each: `concat!` builds `<slug>_us` from `<slug>`, so the count route
/// and the cost route cannot drift into naming different sites.
macro_rules! settle_sites {
    ($($(#[$doc:meta])* $variant:ident => $slug:literal,)*) => {
        /// Which host-side toucher of guest bytes is settling.
        ///
        /// # Why the settle names its caller
        ///
        /// The settle blocks this thread until every submitted guest-page
        /// writeback has executed on the GPU, and on a driven boot that block is
        /// the largest single item in the drain worker's wall clock — a
        /// Safari-drag boot spent 15.6 of the worker's 24.7 busy seconds inside
        /// it. It has sixteen call sites and, until this enum, one flag and one
        /// `fence_us` total served all of them, so no boot could say which site
        /// paid it. A fix aimed at that number was aimed by guess.
        ///
        /// Counting *calls* would rank the sites by how often they ask, which is
        /// the wrong ranking: the flag is clear on most calls and those return
        /// without touching a queue. [`settle_guest_writes`] therefore counts
        /// only the calls that actually waited, and charges the microseconds to
        /// the same site, because a site that settles rarely and expensively and
        /// a site that settles constantly and cheaply read identically in a bare
        /// count and want opposite fixes.
        ///
        /// The per-site microseconds and the `readback_split` `fence_us` total
        /// are the same wait attributed twice, and that is the identity worth
        /// checking: every settle in the device comes through here, so
        /// `sum(settle_*_us)` and `fence_us` agree to within the sampling
        /// window. Their diverging means a new caller reached
        /// `engine::quiesce_guest_writes` directly.
        ///
        /// **The identity holds only where the host-pointer import works**, and
        /// a boot that forgets that will read a huge unattributed remainder as
        /// a missing caller. On the copying arm — a host without the extension,
        /// or `REIMS_VGPU_GUEST_IMPORT=off` — no writeback is ever submitted
        /// without waiting, so every site here reads zero while `fence_us` is
        /// the copying rail's own blocking readback, reported as the same
        /// `ReadbackPhase::Fence`. One measured import-off boot: `fence_us`
        /// 8.38 s against zero settles at every site.
        #[derive(Clone, Copy, PartialEq, Eq, Debug)]
        pub enum SettleSite {
            $($(#[$doc])* $variant,)*
        }

        impl SettleSite {
            /// Every site, for the tests that must be exhaustive over them.
            pub const ALL: &'static [SettleSite] = &[$(SettleSite::$variant,)*];

            /// Census route counting the settles at this site that waited.
            pub fn route(self) -> &'static str {
                match self { $(Self::$variant => $slug,)* }
            }

            /// Census cost charged to this site, in microseconds blocked.
            pub fn route_us(self) -> &'static str {
                match self { $(Self::$variant => concat!($slug, "_us"),)* }
            }

            /// Waits [`settle_guest_writes_unless_disjoint`] skipped here
            /// because nothing outstanding lands in what this site reads.
            pub fn route_disjoint(self) -> &'static str {
                match self { $(Self::$variant => concat!($slug, "_disjoint"),)* }
            }

            /// Waits genuinely owed: the outstanding writeback lands in a page
            /// this site is about to read.
            pub fn route_overlap(self) -> &'static str {
                match self { $(Self::$variant => concat!($slug, "_overlap"),)* }
            }

            /// Waits taken because nothing could be ruled out — this site could
            /// not name its pages, or more than one writeback was outstanding.
            pub fn route_unnamed(self) -> &'static str {
                match self { $(Self::$variant => concat!($slug, "_unnamed"),)* }
            }
        }
    };
}

settle_sites! {
    /// `draw::texture_view::load_linear_texture_impl` — CPU read of a linear
    /// texture's guest pages for the host cache.
    LinearTextureLoad => "settle_linear_texture_load",
    /// `draw::vulkan::load_linear_guest_memoized` — the memoized full-span CPU
    /// re-read behind every linear sampled bind the gather rail declines.
    LinearMemoRead => "settle_linear_memo_read",
    /// `draw::read_buffer_bytes_resolved` — the one CPU read of a buffer's
    /// guest bytes, reached by buffer-backed sampled textures, the indirect
    /// command buffer decode and the CPU buffer fallback.
    BufferGuestRead => "settle_buffer_guest_read",
    /// `compute_exec::stage_texture_raw` — staging a compute texture's guest
    /// bytes.
    ComputeStageTexture => "settle_compute_stage_texture",
    /// `scanout::paint_mapping` — the host console reading a mapping to paint.
    ScanoutPaint => "settle_scanout_paint",
    /// `drain::write_stamp` — the completion stamp's blocking fallback, taken
    /// when the GPU-ordered stamp path declined.
    CompletionStamp => "settle_completion_stamp",
    /// `drain::drain_main_fifo` — the root packet's completion stamp.
    RootStamp => "settle_root_stamp",
    /// `drain::process_child_packet` — a child packet's completion stamp.
    ChildStamp => "settle_child_stamp",
    /// `mapping_write::write_bgra8_inner` — the copying type-11 Store.
    MappingBgra8Write => "settle_mapping_bgra8_write",
    /// `mapping_write::write_rgba8_image_changed`.
    MappingRgba8Write => "settle_mapping_rgba8_write",
    /// `mapping_write::write_raw_rows`.
    MappingRawRowsWrite => "settle_mapping_raw_rows_write",
    /// `mapping_write::read_raw_rows`.
    MappingRawRowsRead => "settle_mapping_raw_rows_read",
    /// `mapping_write::read_rect_raw_at`.
    MappingRectRead => "settle_mapping_rect_read",
    /// `mapping_write::write_rect_raw_at_impl`.
    MappingRectWrite => "settle_mapping_rect_write",
    /// `mapper::write_mapping_bytes_only`.
    MappingBytesWrite => "settle_mapping_bytes_write",
    /// `mapper::read_mapping_bytes`.
    MappingBytesRead => "settle_mapping_bytes_read",
}

/// Block until every guest-page write this device has submitted has executed.
///
/// The writes above are recorded into the engine's command stream and not
/// waited on, which is what makes a Store cheap. A **host-side** reader of the
/// same guest bytes — a mapping read, a CPU seed, a present capture — is not
/// ordered against them by anything the GPU knows about, so it has to settle
/// them first or it reads the pre-Store bytes.
///
/// The guest is ordered separately and does not come through here: its
/// completion stamp is written behind a barrier that already subsumes these
/// copies (`engine::write_stamp_after_guest_writes`).
///
/// Free when nothing is outstanding — the engine keeps a debt flag and this
/// returns without touching a queue when it is clear. `site` is what a boot
/// reads to find which caller pays for the waits that are not free; see
/// [`SettleSite`].
pub fn settle_guest_writes(site: SettleSite) {
    #[cfg(feature = "backend-vulkan")]
    {
        // The flag read is one relaxed-acquire load and clear is the common
        // answer, so the census below runs only on the calls that cost
        // something. It can race a writeback armed on another thread between
        // this load and the wait, which makes the site's count a lower bound by
        // at most one per race — the engine re-reads the flag under its own
        // lock and the ordering is unaffected.
        if !crate::backend::vulkan::engine::guest_writes_outstanding() {
            return;
        }
        let started = std::time::Instant::now();
        crate::backend::vulkan::engine::quiesce_guest_writes();
        crate::runtime::drain::note_store_route(site.route());
        crate::runtime::drain::note_store_route_us(
            site.route_us(),
            started.elapsed().as_micros() as u64,
        );
    }
    #[cfg(not(feature = "backend-vulkan"))]
    let _ = site;
}

/// [`settle_guest_writes`], skipped when the outstanding writeback lands nowhere
/// near what this caller is about to read.
///
/// A writeback lands in one surface's pages. Most readers that block on it are
/// reading somewhere else entirely — a glyph atlas, a small linear texture — and
/// the wait they take is for a write that will never touch a byte they read. A
/// driven Safari-drag boot spent 11.5 s in one such reader.
///
/// `pages` is resolved by the closure and the closure runs **only** when
/// something is outstanding, so a caller may put a page-table walk in it: the
/// common answer is the debt flag being clear, and that costs one atomic load
/// exactly as [`settle_guest_writes`] does. It must return every page the caller
/// is about to read, and `None` for "cannot say" — a short list would license a
/// read of pages it had omitted, which is a stale frame.
///
/// Three outcomes, counted apart because they want different fixes:
/// `<site>_disjoint` is the wait this saved, `<site>_overlap` is a wait that was
/// genuinely owed, and `<site>_unnamed` is one taken because nothing could be
/// ruled out — a caller whose walk failed, or a second outstanding writeback
/// (`gwdebt_unnamed`).
pub fn settle_guest_writes_unless_disjoint(
    site: SettleSite,
    pages: impl FnOnce() -> Option<Vec<u64>>,
) {
    #[cfg(feature = "backend-vulkan")]
    {
        if !crate::backend::vulkan::engine::guest_writes_outstanding() {
            return;
        }
        use crate::backend::vulkan::engine::GuestWriteReach as Reach;
        let reach = match pages() {
            Some(p) => crate::backend::vulkan::engine::guest_writes_reaching(&p),
            // The caller could not name its own window, which is the same
            // undecidable as the ledger failing to name the writeback's.
            None => Reach::Unnamed,
        };
        crate::runtime::drain::note_store_route(match reach {
            Reach::Disjoint => site.route_disjoint(),
            Reach::Overlap => site.route_overlap(),
            Reach::Unnamed => site.route_unnamed(),
        });
        if reach == Reach::Disjoint {
            return;
        }
        settle_guest_writes(site);
    }
    #[cfg(not(feature = "backend-vulkan"))]
    {
        let _ = (site, pages);
    }
}

/// Release the engine residents of linear cache entries whose task or object
/// the guest deleted this drain.
///
/// Two releases, and dropping either one is a leak in the opposite direction: an
/// unpin alone leaves the image holding the only copy of content nothing may
/// reclaim, and retiring the content alone leaves a pinned slot no reclaim path
/// may take. Together they make the image ordinarily evictable.
///
/// Task teardown means the GPU VA maps are gone, so nothing here writes guest
/// pages — the deleted object's bytes are not guest work any more.
pub fn retire_linear_residents(state: &mut DeviceState) {
    if state.retired_linear_residents.is_empty() {
        return;
    }
    let retired = std::mem::take(&mut state.retired_linear_residents);
    // The engine that holds these pins is the Vulkan one; a `backend-metal`
    // build arms nothing that could have pinned them, so taking the list is the
    // whole of the work there.
    #[cfg(feature = "backend-vulkan")]
    for key in &retired {
        crate::backend::vulkan::engine::unpin_resident_storage(key);
        crate::backend::vulkan::engine::retire_resident_storage_content(key);
        crate::observe::off(format!(
            "linear_resident_retired task={} ref={} gva={:#x} {}x{} fmt={:#x}",
            key.map_generation,
            key.texture_ref,
            key.surface_offset,
            key.width,
            key.height,
            key.pixel_format
        ));
    }
    #[cfg(not(feature = "backend-vulkan"))]
    drop(retired);
}

/// Copy `identity`'s pixels into `mapping_id`'s guest pages.
///
/// `true` when the guest's pages hold the frame. `false` is a real loss and is
/// reported on the failure channel by the arm that refused — the caller has no
/// second copy to fall back to, because this rail never made one.
#[cfg(feature = "backend-vulkan")]
pub fn store_render_frame<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mapping_id: u32,
    identity: &crate::backend::vulkan::engine::TargetIdentity,
    width: u32,
    height: u32,
) -> bool {
    let started = std::time::Instant::now();
    crate::runtime::drain::note_store_route("surface_flush");
    // The GPU writes the guest's pages directly. Tried first because when it
    // works there is nothing left to do: no staging buffer is mapped and no
    // host pass over the frame happens at all.
    match crate::runtime::mapping_write::write_bgra8_from_resident_gpu(
        state, host, mapping_id, identity, width, height,
    ) {
        Ok(bytes) => {
            crate::runtime::drain::note_store_route("render_flush_gpu_direct");
            finish(state, mapping_id, identity, bytes as usize, started);
            return true;
        }
        Err(decline) => {
            // Latched per mapping as well as per reason: a host without
            // `VK_EXT_external_memory_host` declines every Store of every
            // surface, and a line each would drown the channel.
            crate::observe::Emit::decline("render_flush_gpu_declined", &decline)
                .field("mapping", mapping_id)
                .field("geom", format!("{width}x{height}"))
                .fail_once(u64::from(mapping_id));
            crate::runtime::drain::note_store_route("render_flush_gpu_declined");
        }
    }
    // The copying arms. These are the only arms on a host that cannot import
    // guest RAM, and the arm a discrete GPU takes regardless.
    //
    // Borrow the readback where it needs no transformation. The writer below is
    // declared in guest scanout order, so a resident reporting semantic RGBA8
    // owes an R/B exchange first — a whole-frame pass, and `into_bgra8` on an
    // owned copy is its home, so a non-BGRA resident takes the copy rather than
    // teaching the lease to rewrite memory it does not own.
    let bpr = width.saturating_mul(4);
    let write_started = std::time::Instant::now();
    let (ok, frame_len) = match crate::backend::vulkan::engine::read_target_leased(identity) {
        Ok(Some(leased)) if leased.bgra => {
            crate::runtime::drain::note_store_route("render_flush_leased");
            let len = leased.bytes().len();
            let ok = crate::runtime::mapping_write::write_bgra8_uncached(
                state,
                host,
                mapping_id,
                leased.bytes(),
                bpr,
                width,
                height,
            );
            // End the lease before anything below reaches the engine again: the
            // re-stamp in `finish` does, and a holder blocking on the engine
            // lock while a teardown waits for this lease is the deadlock
            // `LeasedFrame` forbids.
            drop(leased);
            (ok, len)
        }
        // Either the pool declined the lease (uncached readback memory, where
        // reading the mapping in place is the slower shape) or the resident is
        // not in scanout order. Drop any leased frame first so its slot is back
        // in the pool before the second readback asks for one.
        Ok(leased) => {
            drop(leased);
            crate::runtime::drain::note_store_route("render_flush_copied");
            match crate::backend::vulkan::engine::read_target(identity) {
                Ok(rb) => {
                    // Shared rather than owned outright: the write's tail
                    // publishes this frame to the surface cache, and a cache
                    // entry holds its frame behind an `Arc` precisely so the two
                    // can name one allocation instead of copying it.
                    let bytes = std::sync::Arc::new(rb.into_bgra8());
                    let len = bytes.len();
                    let ok = crate::runtime::mapping_write::write_bgra8_owned(
                        state, host, mapping_id, &bytes, bpr, width, height,
                    );
                    (ok, len)
                }
                Err(e) => {
                    crate::observe::fail(format!(
                        "render_store_lost mapping={mapping_id} {width}x{height} \
                         reason=resident_read err={e}"
                    ));
                    return false;
                }
            }
        }
        Err(e) => {
            crate::observe::fail(format!(
                "render_store_lost mapping={mapping_id} {width}x{height} \
                 reason=resident_read err={e}"
            ));
            return false;
        }
    };
    crate::runtime::drain::note_readback_phase(
        crate::runtime::drain::ReadbackPhase::Write,
        write_started.elapsed().as_micros() as u64,
    );
    if !ok {
        crate::observe::fail(format!(
            "render_store_lost mapping={mapping_id} {width}x{height} reason=write_refused"
        ));
        return false;
    }
    finish(state, mapping_id, identity, frame_len, started);
    true
}

/// Hand the currency witness back to the image the frame came out of, and score
/// the write.
///
/// `write_bgra8_*` ends in `mark_mapping_written`, which advances the mapping's
/// `surface_content_epoch` — correctly, because its guest pages did change. But
/// the *pixels* did not: they are this resident's, copied out of it one
/// statement ago. Leaving the stamp behind invalidates a resident that holds
/// exactly the mapping's content, which costs the next Load its elision and
/// sends it to a CPU seed for bytes it already has.
#[cfg(feature = "backend-vulkan")]
fn finish(
    state: &mut DeviceState,
    mapping_id: u32,
    identity: &crate::backend::vulkan::engine::TargetIdentity,
    frame_len: usize,
    started: std::time::Instant,
) {
    if let Some(epoch) = state
        .mappings
        .get(&mapping_id)
        .map(|m| m.surface_content_epoch)
    {
        crate::backend::vulkan::engine::stamp_resident_content_epoch(identity, epoch);
    }
    // The copy above means this image has stopped being the only place these
    // pixels exist, so the reclaim paths may take it.
    crate::backend::vulkan::engine::note_resident_content_copied_out(identity);
    crate::runtime::drain::note_drain_phase(
        crate::runtime::drain::DrainPhase::Flush(crate::runtime::drain::FlushRail::Render),
        started,
    );
    crate::observe::line(format!(
        "render_store mapping={mapping_id} bytes={frame_len} us={}",
        started.elapsed().as_micros()
    ));
}

/// Why a GVA render Store could not hand its resident straight to the guest's
/// pages, so it fell back to reading the frame back and converting it row by
/// row.
///
/// Every one of these is a routing answer and not a loss — the copying rail
/// still lands the frame — but each costs a blocking GPU→host readback of a
/// whole framebuffer plus a host pass over it, which is the largest single cost
/// this device pays. They are named individually so a boot says which check is
/// holding the volume.
#[cfg(feature = "backend-vulkan")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GvaWritebackDecline {
    /// The guest declared a destination format whose texel is not four bytes of
    /// colour, so no image→buffer copy can produce it. A `RGBA16_FLOAT` render
    /// target lands here and always will; `convert_rgba8_to_row` is its only
    /// route.
    FormatNeedsConversion { format: u16 },
    /// The resident's channel order is not the order the destination stores.
    /// Distinct from the engine's own check of the same pair: this one is asked
    /// before the walk, so a mismatch costs no page-table work.
    OrderMismatch { resident_bgra: bool, want_bgra: bool },
    /// The guest's row pitch is not a whole number of texels, or is narrower
    /// than the frame, so there is no `bufferRowLength` that describes it.
    PitchNotTexels { row_stride: u32 },
    /// The frame's first texel does not start on a texel boundary within its
    /// page. `VkBufferImageCopy::bufferOffset` must be a multiple of the texel
    /// block size, and a copy that ignored this is undefined rather than
    /// misaligned.
    OffsetNotTexelAligned { in_page: u64 },
    /// The command resolved no destination pages before it was submitted, so
    /// there is nothing this rail is authorised to write. The copying rail
    /// treats the same answer as "unbounded" and writes anyway through its own
    /// re-walk; this rail has no second walk to fail closed on, so it declines.
    Unlicensed,
    /// The pre-submit walk did not resolve every page of the destination span,
    /// so its page list cannot be read positionally.
    /// [`crate::runtime::draw::StoreTargetPages::ordered_complete`] states what
    /// a short list would land.
    SpanIncomplete,
    /// The destination span did not become a guest-RAM reference; the inner
    /// refusal names the check, restated here for the reason
    /// `GpuWritebackDecline::GuestRefRefused` restates its own.
    GuestRefRefused {
        refusal: crate::runtime::guest_ram_map::MapRefusal,
    },
    /// The engine declined or the copy failed; the inner error names which.
    Engine {
        inner: crate::backend::vulkan::engine::DrawError,
    },
}

#[cfg(feature = "backend-vulkan")]
impl crate::observe::Decline for GvaWritebackDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::FormatNeedsConversion { .. } => "gvawb_format_needs_conversion",
            Self::OrderMismatch { .. } => "gvawb_order_mismatch",
            Self::PitchNotTexels { .. } => "gvawb_pitch_not_texels",
            Self::OffsetNotTexelAligned { .. } => "gvawb_offset_not_texel_aligned",
            Self::Unlicensed => "gvawb_unlicensed",
            Self::SpanIncomplete => "gvawb_span_incomplete",
            Self::GuestRefRefused { .. } => "gvawb_guest_ref_refused",
            Self::Engine { inner } => crate::observe::Decline::slug(inner),
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::Unlicensed | Self::SpanIncomplete => Vec::new(),
            Self::FormatNeedsConversion { format } => vec![("fmt", format!("{format:#x}"))],
            Self::OrderMismatch {
                resident_bgra,
                want_bgra,
            } => vec![
                ("resident", if *resident_bgra { "bgra" } else { "rgba" }.to_string()),
                ("want", if *want_bgra { "bgra" } else { "rgba" }.to_string()),
            ],
            Self::PitchNotTexels { row_stride } => vec![("bpr", row_stride.to_string())],
            Self::OffsetNotTexelAligned { in_page } => vec![("in_page", in_page.to_string())],
            Self::GuestRefRefused { refusal } => {
                let mut f = vec![("via", crate::observe::Decline::slug(refusal).to_string())];
                f.extend(crate::observe::Decline::fields(refusal));
                f
            }
            Self::Engine { inner } => crate::observe::Decline::fields(inner),
        }
    }
}

#[cfg(feature = "backend-vulkan")]
crate::observe::decline::decline_display!(GvaWritebackDecline);

/// Copy `identity`'s pixels into the guest pages behind a type-2/3 render
/// target's `target_gva`, with no host copy of the frame at any point.
///
/// The GVA twin of [`store_render_frame`]'s first arm, and worth diffing
/// against it: both end in `copy_target_to_guest_pages` and they differ only in
/// how the destination pages are named. A mapping carries its own page list and
/// a page-table vouch licenses it; a GVA carries neither, so the licence is the
/// walk the command took **before it was submitted** — `pages`, which the
/// caller resolved at that point and which this rail may not widen.
///
/// # Why this is the whole cost of a GVA Store
///
/// The rail this stands in front of reads the resident back to the host
/// (`read_resident_chain`, a blocking fence) and then writes it out again a row
/// at a time through `convert_rgba8_to_row`. On a driven desktop-compositing
/// boot that is 59 % of render Stores and most of the time the device spends
/// waiting on a fence. Everything this call declines on pays that instead, so a
/// decline is a cost and never a lost frame.
///
/// # What it does not do
///
/// It publishes no host-side copy of the frame. The two GVA pixel caches are
/// *dropped* instead, for the same reason `store_render_frame` forgets the
/// mapping's: after this call the guest's own pages are the only place the
/// frame exists, and an entry left behind would serve a later sample the
/// previous Store's bytes. The sampled rail re-reads them from guest memory,
/// which is what `store_gva_owned`'s `guest_holds_bytes` already promised.
#[cfg(feature = "backend-vulkan")]
pub(crate) fn store_gva_frame<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    identity: &crate::backend::vulkan::engine::TargetIdentity,
    c0: &crate::runtime::draw::ColorRtRequest,
    texture_ref: u32,
    pages: Option<&crate::runtime::draw::StoreTargetPages>,
) -> Result<u64, GvaWritebackDecline> {
    use crate::contract::pixel_format::{store_texel_order, TexelLayout};
    // The destination's channel order, and the whole reason this rail can exist
    // at all: a copy converts nothing, so the guest must already read these
    // bytes in the order the resident holds them.
    let Some(order) = store_texel_order(c0.format) else {
        return Err(GvaWritebackDecline::FormatNeedsConversion { format: c0.format });
    };
    let want_bgra = order == TexelLayout::Bgra8;
    let resident_bgra = identity.is_bgra();
    // A healthy zero on the rail as it stands: `gva_chain_identity` builds the
    // key from this same `c0.format`, so the two agree by construction and this
    // arm is the alarm for an identity that came from somewhere else. Kept
    // rather than asserted because the answer it protects — whether the bytes
    // about to be copied are the bytes the guest reads — is not one to take on
    // trust from a caller.
    if resident_bgra != want_bgra {
        return Err(GvaWritebackDecline::OrderMismatch {
            resident_bgra,
            want_bgra,
        });
    }
    let bpt = u64::from(order.bytes_per_texel());
    let row_stride = u64::from(c0.row_stride);
    if row_stride == 0 || !row_stride.is_multiple_of(bpt) || row_stride < u64::from(c0.width) * bpt {
        return Err(GvaWritebackDecline::PitchNotTexels {
            row_stride: c0.row_stride,
        });
    }
    let Some(pages) = pages else {
        return Err(GvaWritebackDecline::Unlicensed);
    };
    let page_size = state.page_size();
    let Some(gpas) = pages.ordered_complete(c0.target_gva, page_size) else {
        return Err(GvaWritebackDecline::SpanIncomplete);
    };
    let in_page = c0.target_gva % page_size;
    if !in_page.is_multiple_of(bpt) {
        return Err(GvaWritebackDecline::OffsetNotTexelAligned { in_page });
    }
    // The extent the copy names and nothing past it: padding after the final
    // row belongs to the allocation but is not texels this Store was given, and
    // the copying rail leaves it alone too. The two rails must land identical
    // guest memory or a fallback would be visible.
    let extent = u64::from(c0.height.saturating_sub(1)) * row_stride + u64::from(c0.width) * bpt;
    let runs = crate::runtime::guest_ram_map::references_for_runs(
        host, gpas, page_size, in_page, extent,
    )
    .map_err(|refusal| GvaWritebackDecline::GuestRefRefused { refusal })?;
    let target = crate::backend::vulkan::engine::GuestPageTarget {
        runs,
        // Checked above to divide exactly, so this is the guest's pitch and not
        // a rounding of it.
        row_length_texels: (row_stride / bpt) as u32,
        width: c0.width,
        height: c0.height,
        bgra: want_bgra,
    };
    // This device is about to write these guest pages, and the hypervisor's
    // dirty bitmap is defined not to see it. Without this record a reader
    // holding a gathered image over the same pages
    // (`crate::runtime::gather_witness`) cannot tell "nobody wrote them" from
    // "we wrote them ourselves", and vouches a retained image that no longer
    // matches the pages — a wrong frame that then persists.
    //
    // The copying rail this stands in front of records the identical fact from
    // inside `gva_view`, so the two rails leave the same witness behind and a
    // decline is invisible to every reader. Before the submit and not after it,
    // and over the whole walked span rather than the copy's extent: a spurious
    // bump costs a re-read of bytes that did not change, and the opposite error
    // hands out a stale copy.
    state.note_host_wrote_pages(gpas.to_vec());
    crate::backend::vulkan::engine::copy_target_to_guest_pages(identity, &target, gpas)
        .map_err(|inner| GvaWritebackDecline::Engine { inner })?;
    // Nothing here leaves a host copy of the frame, so neither GVA-keyed cache
    // may go on naming one.
    crate::runtime::surface_cache::evict_gva(state, c0.target_gva);
    if texture_ref != 0 {
        crate::runtime::surface_cache::evict_texture(state, texture_ref);
    }
    // The copy means this image has stopped being the only place these pixels
    // exist, so the reclaim paths may take it — the same handover
    // `store_render_frame` performs in `finish`.
    crate::backend::vulkan::engine::note_resident_content_copied_out(identity);
    Ok(extent)
}

#[cfg(test)]
mod tests {
    use super::SettleSite;

    /// Two sites sharing a slug would silently sum their waits into one census
    /// line, and the reading would name the wrong caller as the device's largest
    /// cost — which is the exact mistake [`SettleSite`] exists to stop. Walks
    /// [`SettleSite::ALL`], so a variant added without a slug of its own fails
    /// here rather than at the next boot's ranking.
    #[test]
    fn every_settle_site_carries_its_own_census_route() {
        let mut seen = std::collections::BTreeSet::new();
        for site in SettleSite::ALL {
            assert!(
                seen.insert(site.route()),
                "{:?} reuses the route {}",
                site,
                site.route()
            );
            assert_eq!(
                site.route_us(),
                format!("{}_us", site.route()),
                "{site:?}"
            );
        }
        assert_eq!(seen.len(), SettleSite::ALL.len());
    }
}
