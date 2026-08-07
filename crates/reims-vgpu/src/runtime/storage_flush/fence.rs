//! The before-fence passes: everything owed to guest RAM at a completion stamp.
//!
//! A completion stamp is this device's statement that the work is finished, so
//! the guest is entitled to read the bytes through its own mapping — a path no
//! host-side choke point in [`super::access`] can intercept. Each of the four
//! deferred rails therefore lands every window it still holds before the fence
//! is signalled. What each pass measured is recorded by [`super::report`].

#[cfg(feature = "backend-vulkan")]
use super::access::flush_intersecting;
#[cfg(feature = "backend-vulkan")]
use super::land::{flush_gva_one, flush_linear_one};
#[cfg(feature = "backend-vulkan")]
use super::report::note_fence_batch_band;
use crate::model::DeviceState;
use crate::runtime::host::{HostMemory, HostOps};

/// Land every armed GVA render-Store window, because the guest is about to be
/// told the work is finished.
///
/// This is the deferral rail's contract with the guest, and it is the one thing
/// `guards::deferred_pages_still_ours` cannot substitute for. A completion stamp is
/// this device's only statement that a render is done; from the instant it lands
/// the guest may free the target, and its own allocator may hand those pages to
/// anything at all without touching a page table — so no later walk, page-set
/// comparison or content test can tell the memory apart from the target it used
/// to be. The only sound moment to write a render's bytes into guest RAM is
/// before the fence that claims they are already there.
///
/// Apple's device needs no equivalent because it has no equivalent window: the
/// render target *is* the guest allocation, so completion and "the bytes are in
/// guest memory" are the same event. This is that invariant restated for a rail
/// that has to copy.
///
/// What the deferral still buys is everything inside one fence: a chain of
/// passes rendering into the same target reuses the registry resident, and
/// `supersede_gva_window` still drops a window the same submission re-renders.
/// What it stops buying is survival across the fence, which was never the
/// device's to sell.
#[cfg(feature = "backend-vulkan")]
pub fn flush_gva_windows_before_fence<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
) {
    if state.gva_deferred_flush.is_empty() {
        return;
    }
    // Oldest-first, so windows land in the order they were rendered: a later
    // Store at an address the guest recycled within one submission must not be
    // overwritten by the earlier one.
    let mut landed = 0u64;
    while let Some((gva, entry)) = state.take_oldest_gva_deferred_window() {
        crate::runtime::drain::note_store_route("gvaw_fence_flush");
        let _ = flush_gva_one(state, host, gva, &entry, true, "fence");
        landed += 1;
    }
    note_fence_batch_band(landed);
}

/// Metal-direct builds never arm GVA windows — nothing to land at the fence.
#[cfg(not(feature = "backend-vulkan"))]
pub fn flush_gva_windows_before_fence<M: HostMemory + HostOps>(
    _state: &mut DeviceState,
    _host: &mut M,
) {
}

/// Land every armed linear compute-storage window, for the same reason and under
/// the same contract as [`flush_gva_windows_before_fence`].
///
/// This rail writes a raw task GVA. `ComputeStorageResidencyKey::linear` sets
/// `mapping_id` to 0 and stores the *task id* in `map_generation`, so there is no
/// mapping incarnation to compare and no lifecycle notify anywhere in the wire
/// format — exactly the position the GVA render rail is in, and exactly why
/// `6bc2220` could clear `flush_render_one` and `flush_storage_one` on
/// `map_generation` drift and could not clear this one.
///
/// # Measured before it was repaired
///
/// One x86/Vulkan boot on the crash-hunt workload (Safari on three compositing
/// pages, Finder windows, then 600 s of Mission Control ×71, Spotlight ×71,
/// window drags ×142):
///
/// ```text
/// linw_stamp_same       0
/// linw_stamp_outlived   1     task=5 ref=52 gva=0x39f000 128x135 stamps=1019
/// ```
///
/// Both halves matter. The rail is late whenever it lands at all — the one
/// landing in ten minutes came 1 019 fences after the guest was told the work was
/// done. And it lands almost never, which is what makes the repair free: the
/// objection that stopped the fence repair from being applied to the render rail
/// was the cost of writing back full-screen frames ~98 % of which nothing reads,
/// and there is no such cost here. One window per ten minutes is not a writeback
/// budget.
///
/// A rate this low cannot on its own convict this rail of any guest crash, and no
/// such claim is made. What it does mean is that the correct behaviour is also
/// the cheap one, so there is nothing to trade.
#[cfg(feature = "backend-vulkan")]
pub fn flush_linear_windows_before_fence<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
) {
    if state.linear_deferred_flush.is_empty() {
        return;
    }
    // Snapshot the keys first: `flush_linear_one` disarms its own window and may
    // flush others through the cache paths below it, so iterating the live map
    // would borrow it across a mutation. A key whose window is gone by the time
    // it comes up disarms to `None` and the flush is a no-op on the guest.
    let armed: Vec<(crate::model::ComputeStorageResidencyKey, u32)> = state
        .linear_deferred_flush
        .iter()
        .map(|(key, entry)| (*key, entry.generation))
        .collect();
    for (key, generation) in armed {
        if !state.linear_deferred_flush.contains_key(&key) {
            continue;
        }
        crate::runtime::drain::note_store_route("linw_fence_flush");
        let _ = flush_linear_one(state, host, &key, generation);
    }
}

/// Metal-direct builds never arm linear windows — nothing to land at the fence.
#[cfg(not(feature = "backend-vulkan"))]
pub fn flush_linear_windows_before_fence<M: HostMemory + HostOps>(
    _state: &mut DeviceState,
    _host: &mut M,
) {
}

/// Land every armed mapping-keyed window — type-11 render Stores and compute
/// storage alike — because the guest is about to be told the work is finished.
///
/// This is the last of the four deferred rails to be bound to the fence, and it
/// is bound for a *different* reason from the other three, which is why it was
/// measured first rather than assumed.
///
/// # The other three rails were bound because they could not name their memory
///
/// A `GvaDeferredEntry` and a `ComputeStorageResidencyKey::linear` name a raw
/// address, so a guest that frees the allocation and reuses the pages leaves the
/// window pointing at somebody else's memory with nothing to refuse on. That is
/// not this rail's position: `land::flush_render_one` and `land::flush_storage_one`
/// compare the mapping's live `map_generation` against `key.map_generation` and
/// refuse before reading, and `map_generation` moves on exactly the events that
/// let a guest reuse an IOSurface's storage. [`note_mapping_window_against_fence`](crate::runtime::storage_flush::report::note_mapping_window_against_fence)
/// records that argument in full and still holds.
///
/// # This rail is bound because the guest is entitled to the bytes at the fence
///
/// A completion stamp is this device's statement that the render is finished. A
/// guest that has been told so may map the IOSurface and read it — CoreGraphics
/// reading back a layer, a damage forward-copy from the previous buffer, any
/// CPU-side compositing step — and it reads *guest RAM*, through its own
/// mapping, without crossing a single host path this device can intercept.
/// `flush_intersecting` covers every reader that goes through us and there is no
/// mechanism that covers the ones that do not.
///
/// So a deferred window is a bet that nothing reads those pages before we land
/// them, and when the bet loses the guest composites the *pre-Store* bytes: a
/// region of the surface holding whatever was there one frame ago, or nothing at
/// all. That is a stale rectangle in an otherwise correct frame, and it is
/// indistinguishable from the corruption classes this device is chasing.
///
/// Apple's device does not take that bet and does not need to. Its render target
/// *is* the guest allocation, so "the render is complete" and "the bytes are in
/// guest memory" are one event. This is that invariant restated for a rail that
/// has to copy: the copy happens before the statement, not after it.
///
/// # And because it clobbers writes the guest itself made
///
/// A deferred window promises to replay a Store later, and that is only a replay
/// while nothing else writes those pages in between. The guest *is* something
/// else: it maps the same IOSurface and does inter-buffer damage forward-copies
/// and CoreGraphics blits into it. The writeback covers the full attachment
/// extent, so every such guest store inside the deferral interval is gone when
/// the window lands. One x86/Vulkan boot on the icon workload (Safari + Finder,
/// 300 s of Mission Control ×41 / Spotlight ×41 / window drags ×82, then four
/// Finder recomposite rounds) measured that directly:
///
/// ```text
/// surface_resident               49 706
/// surface_flush                  12 343    windows that landed
/// render_flush_over_guest_write   8 968    of those, 73 % clobbered guest bytes
/// rendw_stamp_outlived           12 343    every one landed after the fence
/// storw_stamp_outlived              101
/// ```
///
/// `deferred_flush_clobber` is 8 975 lines of that boot's fail log — the largest
/// self-declared loss of guest work anywhere in it.
///
/// **Those are the numbers that motivated the fence, not current ones.** A
/// driven x86/Vulkan boot on today's binary — 30 s Safari window drag plus two
/// web-content probe runs — reads:
///
/// ```text
/// surface_resident               23 196
/// surface_flush                  23 196
/// render_flush_over_guest_write     152    0.66 % of windows, was 73 %
/// rendw_stamp_outlived                0    was 12 343
/// ```
///
/// `rendw_stamp_outlived` going to zero is the structural half: no window is
/// landing after `write_stamp` any more, which is the ordering statement this
/// rail can actually make, and the doc below names its non-zero as the defect
/// to watch. The clobber rate falling with it is what that predicts.
///
/// Do not read the two tables as one experiment. The 73 % boot was a heavier
/// and much more guest-CPU-composited workload (Mission Control, Spotlight,
/// Finder recomposite rounds) over 300 s, so workload and binary are
/// confounded and the ratio between them is not a measured improvement. What
/// the second table does establish is that **the clobber class is no longer the
/// largest loss in the log on the workload this project drives**, and that
/// bounding what a flush copies is now motivated by its byte cost rather than
/// by this correctness hazard.
///
/// [`note_render_flush_over_guest_write`](crate::runtime::storage_flush::report::note_render_flush_over_guest_write) states why the obvious repair —
/// preserve the pages the guest wrote — is not available: `page_gen[p]` is
/// stamped at the *harvest* that saw page `p` dirty, not at the write, so the
/// witness cannot say whether a store happened before or after the Store this
/// window defers. Preserving on it withheld the device's own frames and turned
/// the screen black (`13ae46d`, 0 of 14 rounds).
///
/// The fence deletes the question rather than answering it. A window that lands
/// before [`crate::runtime::drain::write_stamp`] covers only the interval a
/// synchronous Store would itself have covered, so there is no interval left in
/// which a guest write can be both after the Store and before the writeback.
/// Nothing has to be preserved because nothing is clobbered.
///
/// # What the deferral still buys, and what it stops buying
///
/// Everything inside one fence survives: a chain of passes into the same surface
/// still reuses one resident, and `supersede_covered_render_windows` still drops
/// a window a later Store in the same submission fully covers. What it stops
/// buying is survival *across* the fence, and that is where this rail's cost is,
/// because unlike the linear rail it is not free.
///
/// `arm_surface_resident_store` exists to skip the whole-framebuffer GPU→host
/// readback entirely on the ~86 % of windows nothing ever flushes — `draw_phase`
/// prices that skip at 565 ms per second of wall clock. Landing every window at
/// its fence pays a readback for each: `surface_resident 49 706` against
/// `surface_flush 12 343` bounds it at 4× the current landings. That is the trade
/// this binding makes, and it is a trade rather than a regression only if the
/// measurement says so, and `present_hz` and `draw_us` were read both ways to
/// settle it.
///
/// The GVA rail's binding was expected to cost frame rate and paid back instead
/// (5.9 → 9.5 Hz, `draw_us` 524 ms → 156 ms), because the unbounded rail spent
/// its time in oldest-first `window_cap` eviction storms holding residents pinned
/// across hundreds of frames. `evict_render_windows_to_cap` is the same shape and
/// may go the same way, but that is a prediction and not a reading.
///
/// The endgame removes the trade rather than choosing a side of it: a resident
/// whose image memory *is* the guest pages has nothing to write back, which is
/// why Apple's device has neither this rail nor this cost. That is a backend
/// allocation change, not a scheduling one.
///
/// # What this costs, measured
///
/// A flush is **~1.0 ms**, and the two host passes over the frame are gone.
/// [`crate::runtime::mapping_write::write_bgra8_from_resident_gpu`] makes the
/// guest's own pages the destination of the copy the GPU was making anyway, so
/// nothing reads the frame back to the host and nothing scatters it out again.
/// On a driven x86 window-drag boot the whole log sums to `render=13573
/// render_us=13636753` — **1005 µs a flush** — and `readback_split` reports
/// `write_us=0 write=0`, which is the phase recording nothing because nothing
/// runs. The rail sustains 528 flushes in its busiest second at ~830 µs each.
///
/// **This rail is the device's largest cost again, and the reason is that the
/// other one was removed rather than that this one grew.** The ranking has now
/// inverted twice, so read it as a ranking and not as a fact about this rail:
///
/// ```text
/// driven Safari window-drag, busiest second   draw_us   flush_us
///   the copying rail (21/69 below)               21%        69%
///   after the GPU writeback landed               61%        34%
///   after the read direction went zero-copy      35%        59%
/// ```
///
/// The third line is what a draw costs once it stops copying guest RAM on the
/// CPU: the guest's pages reach the GPU as host-pointer imports for sampled
/// textures and for vertex/storage binds, so the gather phase
/// (`stage_phase runs_us`) all but vanishes from a worker second and a draw
/// gets cheaper with it. Nothing about the flush changed between the second and
/// third lines. It is 59% of the worker because it is the last rail that still
/// moves a whole frame per fence.
///
/// ## What is left inside a flush, and which lever moves it
///
/// `readback_split` and the GPU's own timestamps divide it, and the answer is
/// bytes rather than latency — which decides between the two candidate levers,
/// and the naive reading decides it wrong. Over one boot, both rails together:
///
/// ```text
/// fence/op=235us  gpu/op=146us  bar/op=1.2us     gpu is 62% of the fence
/// vouch/op=143us  resolve/op=30us  write/op=0us
/// ```
///
/// Netting out the GVA rail's smaller copies puts a render flush's own fence
/// near 434 µs with ~354 µs of it GPU execution — about 23 GB/s for an 8 MB
/// surface, which is a PCIe write to system memory running at roughly the width
/// of the link. So:
///
/// - **Batching the fences** (one command buffer for the N windows landing at
///   one drain fence, one wait) recovers only the ~80 µs of submit-to-signal per
///   flush. Real, and about a tenth of a flush.
/// - **Bounding what each flush copies** attacks the 354 µs, and was the lever
///   the "moving whole frames that nobody asked for" paragraph below argued for
///   across three sessions. **It has been built and it saves nothing.** Read the
///   next section before building it a fourth time.
///
/// ### The damage rect was built, measured at zero, and removed
///
/// The design was the obvious one and it was not wrong: a damage enum on the
/// engine's `ResidentTargetSlot`, initialised `All`, unioned per draw through
/// `registry_mark_ready` (which every path leaving new pixels in a resident
/// already goes through, the same property that makes the `content_epoch`
/// invalidation total), taken and reset by the flush, and used to narrow the
/// copy, the page list and the imported range to a band of rows. Fail-closed on
/// a CLEAR load action, on either seed form, and on a recycled image.
///
/// It narrowed **nothing**. On a driven x86 Safari window-drag boot the census
/// pair `flush_rows` / `flush_surface_rows` read *exactly equal on every line of
/// the boot* — every flushed surface was fully damaged, every time.
///
/// The cause is the guest's own draws, not the rail. In the busiest second:
///
/// ```text
/// draws that could write anywhere    clears 840   loaded/full-scissor 1637
/// draws bounded by a scissor         1718  (41% of 4195)
/// flushes                            560
/// ```
///
/// 2477 whole-surface writes against 560 flushes is **4.4 per flush interval**,
/// and one is enough: the union of a bounded draw and an unbounded one is
/// unbounded. The 41% of draws that *are* scissored never get an interval to
/// themselves. `creates=312 target_evicts=0 gen_mismatch=0` in the same second
/// rules out the other explanation — the identities are stable, so the rail was
/// unioning over the right window and not restarting from `All` each frame.
///
/// What survives is the instrument, because the verdict is a property of the
/// workload and will change with it: `draw_cover_full`,
/// `draw_cover_loaded_full_scissor` and `draw_cover_loaded_partial_scissor` on
/// the `engine_delta` line. Build the rect when the first two, summed, fall well
/// below the flush rate. `EngineCounters::note_draw_coverage` carries the
/// arithmetic and this reading.
///
/// `vouch` is the other measurable half and it is now near its floor: one guest
/// page-table entry read per page is what checking every page costs, and
/// `bcbc859` already removed the repeated descents around it (221 µs → 143 µs).
/// Going below that means not checking every page, which is a different and much
/// more careful argument.
///
/// The paragraphs below are the reading from **before** that rail existed, kept
/// because what they establish about the *shape* of the remaining cost still
/// decides what to build next. Read them as history: the numbers in them are the
/// copying rail's, and the copying rail now runs only where the GPU one declines
/// (`render_flush_gpu_declined`, which the boot above never once emitted).
///
/// ## The copying rail, as it was
///
/// It was the single largest cost in the device. On a driven x86 boot (Safari
/// WebGL, 120 Hz) `flush_rails` read `render_us=688003 render=100` with the
/// gva, linear and storage rails all at zero: **69% of the drain worker's
/// entire second**, against 21% for draws. `readback_split` divided each 6.9 ms
/// flush into submit 7 µs, fence 3.04 ms, staging memcpy 0.83 ms and guest-page
/// write 2.68 ms. The two that the GPU rail deletes outright are the last two,
/// and the 6.8× it measures is larger than their 3.5 ms share alone, because a
/// fence that no longer waits on a full-surface readback into host-visible
/// memory is a shorter fence.
///
/// `fence_us` owning that line read like latency, and it was not. The GPU
/// timestamp pair taken inside the fence divides it. On a driven Safari
/// window-drag boot, one 1063 ms window reads `render_us=717130` split
/// `fence_us=410022 write_us=290863 submit_us=6163 map_us=430`, and inside the
/// fence `gpu_us=324787 bar_us=729`. So **79% of `fence_us` is the readback
/// command buffer's own execution** — the copy — and the barrier waiting on the
/// draw batch ahead of it is 729 µs across 720 fences, one microsecond each.
/// Summing what scales with bytes (`gpu_us` + `write_us` + `map_us`) against
/// what does not (`bar_us` + `submit_us` + the fence's non-GPU remainder) puts
/// the rail at **86% bytes and 13% latency**.
///
/// That matters because it decides which of the two endgames below is worth
/// building first, and the naive reading decides it wrong. 720 fences that
/// second each copied a full surface, to produce the 11-17 fresh frames the
/// window presented. The rail is not waiting on the GPU; it is moving whole
/// frames that nobody asked for, so bounding *what* each flush copies attacks
/// six sevenths of it and is not blocked on the host being able to address
/// guest memory.
///
/// All of it is speculative. In the same second `mapw_fence_flush` equals
/// `surface_flush` exactly (104 = 104), so **every flush is this fence and none
/// is a guest demand** — [`flush_mapping_for_guest_read`](crate::runtime::storage_flush::access::flush_mapping_for_guest_read), the
/// `SynchronizeResources` path that fires when the guest actually declares a CPU
/// read, contributes nothing while driving.
///
/// Nothing reads what it produces, either. `RenderFlushWitness` marks each of
/// the two copies a flush lands — the mapping's guest pages and its host surface
/// cache entry — and clears the mark when a host reader takes that copy, so the
/// next flush of the same mapping reports what became of the previous one. The
/// GPU rail lands only the first of the two, so its cache leg is `cache_stored =
/// false` on every flush and the `render_flush_cache_*` pair now scores the
/// copying rail alone. A 30 s driven Safari probe scored 3766 landings:
///
/// ```text
/// render_flush_cache_used      15    render_flush_cache_unread   3751
/// render_flush_pages_used      26    render_flush_pages_unread   3740
/// ```
///
/// **0.4% of the cache copies and 0.7% of the guest-page writes are read by
/// anything in the device before the next flush replaces them.** That is not
/// surprising once stated: every device-side reader of these bytes sits below a
/// rung that prefers the GPU resident (`t11rung_resident`, the LOAD elision, the
/// window's resident-carried present), and the resident is exactly what the
/// flush is a copy of. The readers only fall through to a copy when there is no
/// resident to read — and then there was nothing to write back either.
///
/// That is still not licence to drop the writeback, and the witness says so
/// itself: it can only see readers *inside* the device. The guest CPU loads
/// these pages with no device operation at all and has been observed doing it
/// without declaring it (the black-wallpaper fade snapshot named in
/// [`flush_mapping_for_guest_read`](crate::runtime::storage_flush::access::flush_mapping_for_guest_read)), which is why this fence exists, and after
/// the completion stamp the pages may belong to something else entirely. So the
/// 99% is a bound on what a *cheaper* rail could save, not a licence to delete
/// this one: "write now or never write" is the real choice and this side of it
/// is the safe one.
///
/// What the pair of numbers argues for is not flushing less often but not
/// needing to flush at all. Three routes were named; one has been built.
///
/// - **Built.** The host copies are gone: the GPU writes the frame into the
///   guest's pages through a host-pointer import, so the flush is one copy
///   instead of three passes over the frame. This is the near half of the
///   "zero-copy endgame" — the *destination* is guest memory, though the
///   resident's own image memory still is not, and cannot be while the resident
///   is tiled.
/// - Making the undeclared guest read observable, so the writeback becomes
///   demand-driven everywhere rather than only on `SynchronizeResources`. That
///   is what would make the rail's cost proportional to its 0.7% of consumed
///   work on a discrete host too. Still the route that pays on every host, and
///   still unbuilt; the section below is why it starts as a counter.
/// - **Tried and removed.** Bounding *what* each flush copies. The arithmetic
///   below is still right — essentially all of the ~1.0 ms is the GPU copying a
///   whole surface — and it is still not what the guest gives this device a
///   choice about. See "The damage rect was built, measured at zero, and
///   removed" above.
/// - **Not** "flush only the mappings whose copies get read": which flushes were
///   wasted is knowable only in hindsight, and a mapping whose pages are read
///   while stale has already served wrong pixels.
///
/// The async-readback split (release the device lock across the fence wait) is
/// the step that does not require any of them, and the fence is now most of what
/// a flush is.
///
/// # What witnessing the undeclared read would take
///
/// The second route is the one that pays on every host, so it is worth being
/// exact about why it is not simply the write witness turned around.
///
/// [`crate::runtime::gather_witness`] skips a gather when two halves agree that
/// nothing wrote a page set: [`HostOps::guest_write_gen`] over the hypervisor
/// dirty bitmap for guest CPU stores, and [`crate::runtime::host_writes`] for
/// this device's own writes, which the bitmap is defined not to see. Neither
/// half has a reading counterpart and no third one can be added, because **a
/// read leaves no trace anywhere**. A dirty bitmap is a record of stores; a page
/// the guest loaded and a page it never touched are the same bits in it. That is
/// a property of the hardware, not a gap in the shim.
///
/// The only way a load becomes observable is to make it fault, which means the
/// page must not be present in the guest's mapping when the load happens. On
/// Linux that is `userfaultfd`: register the pages, punch them out, and the
/// access traps to a handler that supplies the bytes before the vCPU resumes.
/// QEMU already runs this combination for post-copy migration, so KVM taking a
/// uffd fault on guest RAM is settled behaviour rather than a research question
/// — but note what is being borrowed. Post-copy fills a page once and is done;
/// this rail would re-arm the same pages every frame, so the per-fault cost and
/// the arm/disarm cost are on the hot path in a way they never are there.
///
/// **The first build of this should be a counter, not a rail, and the reason is
/// in the numbers above rather than in caution.** The measured case for
/// demand-driving is an upper bound on waste, not a measurement of demand: 0.7%
/// of guest-page writes are read by a device reader and 4.2% of landings are
/// declared by `SynchronizeResources`, and neither can see an undeclared guest
/// load. The undeclared load is known to exist — the black-wallpaper fade
/// snapshot named in [`flush_mapping_for_guest_read`](crate::runtime::storage_flush::access::flush_mapping_for_guest_read) is one — but its *rate*
/// has never been measured, and the entire value of the route depends on it. So
/// arm a sample of windows (the [`crate::runtime::gather_witness::AUDIT_STRIDE`]
/// shape is the precedent), still write the bytes, still fill correct content on
/// fault, and count faults against landings. That answers "how often does the
/// guest read what nobody declared" for a fraction of one rail's cost, and it is
/// the number that decides whether the fast path is worth building at all.
///
/// Three hazards, recorded because each one turns the result into a wrong one
/// rather than a noisy one:
///
/// - A fault is not a *read*. `UFFDIO_REGISTER_MODE_MISSING` traps the first
///   access of either kind, so a fault count is an upper bound that includes the
///   guest's own stores — which happen on 73% of these windows, per
///   `render_flush_over_guest_write` above. Separating them needs the fault's
///   `UFFD_PAGEFAULT_FLAG_WRITE`, and a rail that ignores it will conclude the
///   guest reads everything.
/// - Arming a page this device is about to write itself is a fault this rail
///   caused, so the arming site has to know every rail that writes guest RAM —
///   and a grep for that has already missed `gva_view::map_fresh_span_within`
///   once.
/// - Punching a page out loses whatever the guest had put there, so the content
///   to fill with has to be captured before the punch, not after. This one is
///   `MISSING`-mode only; guest RAM here is a shared memfd, where minor-fault
///   mode unmaps without evicting the page cache and there is nothing to
///   capture. See "Which pages to register" below.
///
/// ## What the rail is worth, and the second-order cost of not doing it
///
/// The ledger prices the rail by attribution — the parts sum to the whole — but
/// "removing it returns 20 ms" is a different claim, and one a probe that
/// dropped every mapping-keyed render window at the fence answered directly.
/// That probe is gone; its result is here. Measured on one settled x86/PCI
/// guest, same workload, host GPU at P8 throughout, one representative second
/// each — the guest asks for 8.2 composites and ~63 draws per frame in **all
/// three**, so these are one workload at three speeds:
///
/// ```text
///                     control     no-writeback #1   no-writeback #2
/// guest frames/s         34            98                34
/// present_hz             17.4          68.6              16.4
/// duty                    0.97          0.77              0.97
/// flush_us              760 ms        122 ms             70 ms
/// draw_us per draw      103 us         97 us            421 us
/// ```
///
/// **#1 is the price: 2.9x the guest frame rate and 3.9x the displayed one, at
/// unchanged per-draw cost, with the worker no longer saturated.** So the rail
/// is the cap, and the read-witness route is worth its cost.
///
/// **#2 gave it all back, and that is the warning.** `flush_us` stayed
/// collapsed while `draw_us` per draw quadrupled: 54 `t11rung_resident_refused`
/// with `gw_rail_t11_kb=437400` — binds that refused their resident on
/// `guest_replaced` and gathered 8 MB each out of guest RAM. The control has
/// zero such refusals and gathers 0.9 MB a bind.
///
/// The counterfactual provokes that by being wrong — pages left holding neither
/// our frame nor a whole guest one — so it is not what a correct rail would do.
/// It is what a correct rail would do *if it got the witness wrong*, and the
/// exchange rate is terrible: a 2.26 GB/s writeback for an 8 MB-per-bind
/// gather. **Skipping a writeback has to keep the guest-write witness and the
/// type-11 resident rung sound. Only stopping the write is a wash.**
///
/// Six runs over three boots, `fresh` against `t11rung_resident_refused`:
/// control 34 / 36 / 37 with no refusal in any of them, counterfactual **99**
/// on its first drag with none, then 35 and 38 with 54 and 49. The control does
/// not decay across runs and its own first drag reads 37, so "the first drag
/// after a boot is fast" is not the explanation.
///
/// The chain is named by counters, not inferred. `gw_refused_guest_store=121`
/// and `type11_seed_guest_wrote=86` appear in the degraded run and in neither
/// the control nor the fast counterfactual, and `gw_vouched` — 40 windows in
/// the control — is **absent from both counterfactual runs**. That last one is
/// the mechanism: [`crate::runtime::gather_witness`] subtracts this device's own
/// page-exact write record to tell its stores from the guest's, and a rail that
/// never lands never writes that record. Once real guest stores accumulate with
/// nothing to re-baseline against, the witness assumes the worst and the
/// type-11 rung above it follows.
///
/// ## Which pages to register, and where the route stops
///
/// A `userfaultfd` registration only traps accesses made through the VMA it was
/// registered on, so "register the pages" has to mean the VMA KVM's memslot
/// `userspace_addr` points at, and not some second alias of the same physical
/// memory. [`crate::runtime::host::HostOps::map_pages`] is the crate's only
/// handle on a host VA for guest pages, and the two shims answer differently:
///
/// - **x86/PCI is the pathway this works on.** Its shim never allocates: it
///   translates and hands back `memory_region_get_ram_ptr(mr) + xlat`, which is
///   QEMU's own RAMBlock pointer, and answers `map_pages_stable = 1` for exactly
///   that reason. A registration on that range does trap the vCPU.
///
///   Guest RAM is **shmem, not anonymous**, and that changes which uffd mode
///   applies. Nothing about how the GPU reaches guest pages asks for that — a
///   host-pointer import is taken over an ordinary mapping, so a plain `-m`
///   allocation would serve it. `vm/boot-x86.sh` passes
///   `memory-backend-memfd,share=on` for this section's own reason, and the
///   consequence is favourable rather than not.
///
///   `MISSING` mode over shmem needs the page punched out of the *file*, which
///   is the third hazard below; but a shared memfd keeps the content in the
///   page cache, so the applicable primitive is minor-fault mode
///   (`UFFD_FEATURE_MINOR_SHMEM`, Linux 5.19+; this host runs 7.1.3). A page
///   unmapped from the VMA but still in the page cache raises a *minor* fault,
///   and `UFFDIO_CONTINUE` maps the cached page back with no copy and no
///   content to have captured first. That is the mode post-copy uses for shared
///   memory, and it is the one a rail re-arming the same pages every frame
///   wants: arming costs an unmap, not a punch.
///
///   None of which makes uffd available here — see the privilege section below,
///   which is unchanged and is what actually decides the mechanism.
/// - **arm64/MMIO cannot take this route at all**, for two independent reasons.
///   Its shim answers `map_pages_stable = 0` because a page list that is not
///   host-contiguous gets a packed `mach_vm_remap` view — a second alias, which
///   a fault registration on the RAMBlock would not cover and which would not
///   itself trap the vCPU. And its host is macOS, which has no `userfaultfd`.
///
/// So the read witness is a **Linux-host mechanism**, and shipping it would
/// leave the arm64/macOS pathway on the eager rail with no equivalent. That is
/// not a reason to skip it — x86 is where the cost was measured — but it is a
/// reason not to write it as though it were the general answer, and a reason
/// the eager rail cannot be deleted behind it. The dirty bitmap does not have
/// this problem: KVM indexes it by physical address, so a write through any
/// alias is seen.
///
/// ## `userfaultfd` needs a privilege QEMU does not have
///
/// Measured on the development host (Linux 7.1.3), as the user QEMU runs as:
///
/// ```text
/// userfaultfd(0)                    -> EPERM   /proc/sys/vm/unprivileged_userfaultfd = 0
/// userfaultfd(UFFD_USER_MODE_ONLY)  -> ok
/// ```
///
/// The mode that is available is the one that cannot do the job.
/// `UFFD_USER_MODE_ONLY` exists to stop an unprivileged process trapping
/// **kernel-mode** faults, and a vCPU touching a missing guest page is exactly
/// that: KVM takes the EPT violation and resolves the HVA through
/// `get_user_pages`, in kernel context. (Creatability is measured; that
/// user-mode-only misses the vCPU is the flag's documented purpose and has not
/// been tested here.) A full `userfaultfd` needs `CAP_SYS_PTRACE` on the QEMU
/// binary or `vm.unprivileged_userfaultfd=1`, both of them root changes on the
/// host running the VM.
///
/// That is not fatal, but it decides the shape: the witness cannot be a rail
/// this device silently enables. It is opt-in and it fails visibly when the
/// host will not grant it.
///
/// **The privilege-free alternative is KVM's own, and it is worth costing
/// before assuming uffd.** Deleting the memslot that covers a surface's pages
/// makes guest accesses to them exit to userspace as MMIO — a supported KVM
/// path, no capability, and it reports direction, which uffd MISSING mode does
/// not without reading the fault flags. It is far too slow for a rail (an exit
/// per access against 2 M accesses in a full-frame read) but that does not
/// matter for a counter: un-protect on the first fault, which is all the
/// question "did anything read this landing" needs. What it costs instead is
/// memslot churn — splitting the RAM slot around an 8 MB surface is three
/// `KVM_SET_USER_MEMORY_REGION` calls and a VM-wide EPT flush per arm — which
/// at an [`crate::runtime::gather_witness::AUDIT_STRIDE`]-shaped sample rate is
/// a handful a second. Neither mechanism has been built.
///
/// # Ordering
///
/// Render windows first in arm order, then whatever remains, and both through
/// [`flush_intersecting`](crate::runtime::storage_flush::access::flush_intersecting) rather than by taking entries directly. That choke
/// point runs the fixpoint that drags in every sibling overlapping the same guest
/// bytes, so windows that overlap land together in one pass whatever order this
/// loop reaches them in — the ordering here decides only which *disjoint* window
/// goes first, and disjoint windows cannot overwrite each other.
///
/// A window may legitimately survive: `flush_intersecting` holds every window on
/// a condemned backing so `mapper::resolve` can settle whether the delete named
/// this incarnation. That hold is the existing contract and the fence does not
/// override it — such a window is not owed to guest RAM until the resolve says
/// the memory is still ours.
#[cfg(feature = "backend-vulkan")]
pub fn flush_mapping_windows_before_fence<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
) {
    if state.compute_deferred_flush.is_empty() {
        return;
    }
    // Once per fence that has anything to land, against `mapw_fence_flush`'s
    // once per window. The ratio is how many windows are armed at the same
    // time, which `surface_resident` and `surface_flush` cannot say — both are
    // per-window rates and read 104/s whether that is 104 fences of one window
    // or 52 of two.
    //
    // **Measured, and the answer is one.** A driven x86/PCI second reads
    // `mapw_fence_pass=116 mapw_fence_flush=116 surface_resident=116
    // surface_flush=116` — all four equal, on three consecutive windows. So the
    // concurrency is exactly 1: every armed window is landed by the very next
    // fence, and no fence ever lands two.
    //
    // Two things follow, and both are negative results worth keeping. Batching
    // the readbacks — one submit and one fence for N windows — cannot pay,
    // because N is 1. And the deferral saves no readback at all: arming and
    // flushing are the same rate, so it only moves the readback from the Store
    // to the fence a moment later. That is consistent with `ResidentArmCensus`
    // measuring a mean arm-to-flush age of 351 us against a 2.6 ms fence, which
    // is the same finding from the other side and is why submitting at arm time
    // was refuted: there is no interval to hide the wait in.
    //
    // A narrower version of that idea is *also* refuted, and it is worth naming
    // separately because it is not obviously the same one. At the moment
    // `arm_surface_resident_store` runs, the engine's draw batch is still open
    // and recording, and the render target is already in `TRANSFER_SRC_OPTIMAL`
    // — so the copy could be appended to the render's OWN command buffer rather
    // than submitted as a second one, which the "submit a separate CB earlier"
    // refutation above does not cover. It still does not pay. The second submit
    // costs ~10 us of the ~1.5 ms `fence_us`; `begin_entry` already flushes the
    // open batch without waiting on it, so the GPU runs render and copy
    // back-to-back and one wait covers both. There is no second pipeline drain
    // to save. Worse, arming is not flushing: an icon workload measured 49 706
    // arms against 12 343 flushes, so recording the copy at arm time would pay a
    // full-frame DMA for four windows in five that nothing ever reads.
    //
    // What is left is volume. These 116 flushes are 116 whole 1920x1080 frames,
    // 962 MB/s, read back for ~62 presented frames; every phase in
    // `ReadbackPhase` is proportional to it.
    //
    // **Reading back less than the whole attachment is measured, and it is not a
    // lever.** The guest supplies a damage rect and this device carries it
    // verbatim (`OPCODE_SET_SCISSOR` -> `req.scissor`), so a damage-limited
    // writeback would be the decoded contract rather than a guess — but
    // a driven probe measured the damage rect covering **99.34%** of the
    // attachment's texels, with half the Stores
    // carrying no scissor at all and the other half one that spans the
    // attachment. Partial scissors belong to the small draws *inside* a pass;
    // the Store that ends a full-screen composite declares the full screen. The
    // whole rect is worth 0.66%.
    //
    // Moving the bytes without a CPU pass is closed too, and by policy rather
    // than by measurement: it needs `VK_EXT_external_memory_host`, which this
    // pathway does not request, because importing a host pointer over guest RAM
    // gives the host GPU write access to guest memory.
    //
    // Flushing *fewer times* is closed as well, and by the same witness.
    // [`crate::model::RenderFlushWitness::landed_us`] buckets how long each
    // landing survived before the next replaced it, and across two driven
    // probes `render_flush_age_sub_ms` is **0** against 3079 and 3090 at
    // `_frame_plus`. Nothing is ever rewritten inside a millisecond, so the 99%
    // nobody reads is not one surface written repeatedly inside a burst — it is
    // one full-screen composite per displayed frame, landed once each, at
    // exactly the rate the guest paints. Superseding windows across fence
    // boundaries would have nothing to collapse.
    //
    // **A window-move workload does not reopen that, and it is worth recording
    // why not, because the numbers look at first as though it does.** Moving a
    // 1000x640 Safari window at ~115 Hz (`scripts/window-drag-probe`), per
    // second, against an idle control on the same guest of zero draws and zero
    // flushes at `duty` 0.001:
    //
    // ```text
    // surface_flush 212   render_flush_age_sub_frame 139   _frame_plus 73
    // write_split   frag=212  bytes=1758412800  (8 294 400 each: 1920x1080x4)
    // host_window_cadence presents=11 offered=11   drain_duty duty=0.98
    // render_flush_pages_used 0-5    render_flush_pages_unread 212
    // guest_read_declared 3
    // ```
    //
    // Two thirds of landings are replaced inside a frame here, where the WebGL
    // probe had 97% surviving one. But the bucket that means "collapsible" is
    // `_sub_ms` — a burst rewriting one surface inside a single drain tranche,
    // which no fence boundary separated and nothing could have observed between
    // — and it is **still absent**. What moved is `_sub_frame`: landings 1 to
    // 8.33 ms apart, each its own composite behind its own fence. Every one of
    // those fences entitles the guest to the bytes, so collapsing them is the
    // undeclared-read question again and not a separate lever.
    //
    // What the workload *does* establish is the size of the problem, which is
    // larger than the WebGL figure suggested: **212 full-frame writebacks and
    // 1.76 GB/s to put eleven frames on the screen**, with the worker at 0.98
    // duty. The device keeps up with 212 fences a second and presents 11, so
    // roughly 200 composites per second are written back to guest RAM and never
    // displayed. Whether that ratio is the guest asking or this device
    // presenting too little is not answered by any counter here, and it is the
    // question to take next — `validity_wb_unstated=180` against
    // `validity_wb_licensed=32` in the same second is where to start.
    //
    // What is left is not doing it. Every landing is speculative
    // (`mapw_fence_flush == surface_flush`) and 99% of what it lands is read by
    // nothing (`RenderFlushWitness`), so the writeback survives on exactly one
    // case: a guest CPU read that was never declared. Making *that* observable
    // is the remaining route, and it is a hypervisor-side change rather than a
    // device-side one.
    //
    // ## How much of it lands for nobody, and why declarations cannot replace it
    //
    // That paragraph rested on one witness, which sees only *device* readers.
    // The guest's own declared reads are now counted too
    // ([`flush_mapping_for_guest_read`]), so both consumers can be bounded at
    // once. One driven x86/PCI boot, three Safari probes, summed over its
    // `store_routes` windows:
    //
    // ```text
    // mapw_fence_flush           8051   windows landed
    // render_flush_pages_used     187   landings a device reader consumed  (2.3%)
    // render_flush_pages_unread  7774
    // guest_read_declared         778   guest declarations of a CPU read
    // guest_read_on_flushed_mid   339   of those, on a mapping this rail writes (4.2% of landings)
    // guest_read_on_other_mid     439
    // ```
    //
    // So **at most ~6.5 % of the writeback has a witnessed consumer**, and the
    // two sets may overlap, so that is a ceiling rather than a total. Ninety-odd
    // per cent of the largest cost in this device lands for a consumer nobody
    // has ever observed. That is the case for building the read witness.
    //
    // It is also the case against the cheap version of it. Declarations cover
    // 4.2 % of landings, so **dropping the eager rail and relying on op 0x35
    // would lose the other 95 %** — the tripwire is required, not an
    // optimisation on top of the declarations. And the two rates do not move
    // together: an earlier, lighter boot read 1035 declarations against 3379
    // landings (0.31 each) where this one reads 778 against 8051 (0.10), so the
    // flush rate scales with rendering and the declaration rate does not. The
    // gap widens exactly when the cost matters most.
    //
    // Note also `guest_read_dry` is 778 of 778 on both boots. That is expected
    // and is not evidence of anything: this fence empties every window before
    // any declaration can arrive, so a declaration can never land one.
    //
    // # One of the two CPU passes over the result is gone; the other is the floor
    //
    // The four closed levers above are all about the readback. The *number of
    // CPU passes over its result* was a separate cost, and it was reducible.
    //
    // A flush used to make two passes over ~8 MB: `readback_split map_us` copied
    // the mapped staging buffer into a `Vec<u8>`, then `write_split land_us`
    // scattered that Vec into guest pages — about 0.82 ms and 1.06 ms per frame,
    // together ~250 ms of a loaded second, as large as the fence. The first
    // existed only so the host surface cache could hold an `Arc<Vec<u8>>`, and
    // `render_flush_cache_used` prices that entry at 0.4 %.
    //
    // It is deleted. `read_target_leased` lends the staging buffer through
    // [`crate::backend::vulkan::engine::LeasedFrame` ] and the scatter reads it
    // in place. Measured on a driven x86/PCI boot, three consecutive one-second
    // windows at 120 flushes each:
    //
    // ```text
    // readback_split  map_us=0 map=120 map_max_us=0
    // write_split     stage_us=0 stage=0 land_us=104832 land=120 cache_us=0
    // store_routes    render_flush_leased=120 surface_flush=120
    // ```
    //
    // `map_max_us=0` is the part worth keeping: not an average that rounded
    // down, but no single flush in 360 spending a microsecond there. What the
    // phase still times on that arm is the `vkInvalidateMappedMemoryRanges` a
    // non-coherent readback owes, and this host's readback memory is coherent.
    //
    // Two things make the borrow sound and both are load-bearing. The slot is
    // taken out of every list that could hand it to a GPU copy — including the
    // ring entry's pending cleanup, which is why the lease is claimed before
    // `seal_entry` — and the holder takes no engine lock, which is why
    // `flush_render_one` runs `flush_windows_under_bgra8_write` *before* it
    // acquires the frame rather than letting the writeback's own
    // `flush_intersecting` read another resident from inside the borrow. Do
    // *not* simply hold the engine lock across the scatter instead: the host
    // window's present path takes it, and adding a millisecond per flush at this
    // rate would move the stall onto the window.
    //
    // The cache entry is **invalidated**, not left behind. A reader that hits a
    // stale entry is served an old frame with no witness saying so, which is the
    // corruption shape the fence binding exists to close. Falling through to the
    // guest pages this flush just wrote is correct by construction, and the boot
    // above logged zero `present_capture FAIL` and zero `deferred_flush_lost` on
    // the leased rail.
    //
    // What is left is the floor. `land_us` is 0.87 ms per frame at ~1 GB/s of
    // cache-cold scattered writes into guest RAM, and there is no second pass to
    // remove — the only way past it is not to write the bytes at all, which is
    // the demand-driven route named above.
    crate::runtime::drain::note_store_route("mapw_fence_pass");
    // Snapshot first: landing one window consumes its overlapping siblings
    // through the fixpoint, so iterating the live map would borrow it across a
    // mutation. A key already consumed by an earlier pass is skipped rather than
    // re-flushed.
    for key in mapping_windows_fence_order(state) {
        if !state.compute_deferred_flush.contains_key(&key) {
            continue;
        }
        crate::runtime::drain::note_store_route("mapw_fence_flush");
        state.fence_flushed_mappings.insert(key.mapping_id);
        flush_intersecting(
            state,
            host,
            key.mapping_id,
            key.surface_offset,
            key.span_end,
        );
    }
}

/// Metal-direct builds never arm mapping-keyed windows — nothing to land.
#[cfg(not(feature = "backend-vulkan"))]
pub fn flush_mapping_windows_before_fence<M: HostMemory + HostOps>(
    _state: &mut DeviceState,
    _host: &mut M,
) {
}

/// Land every deferred rail. Call this immediately before any word that tells the
/// guest work has finished.
///
/// There is more than one such word. The child stamp slots go through
/// [`crate::runtime::drain::write_stamp`], but the *root* completion stamp is
/// written straight into slot 0 by the main FIFO drain, and it is the one the
/// guest's root packets wait on. A rail bound only to the child path is not bound:
/// the guest may free a render target the moment the root stamp moves, and its
/// allocator may hand those pages to anything — a kalloc element, another
/// process's heap — which no later check can tell from the target they used to be.
///
/// So the binding belongs to "the guest is about to be told", not to one of the
/// two writers that tell it. Every caller of this function is such a site, and a
/// new completion word is a new caller.
///
/// Each rail early-returns when nothing is armed, so the common case — a root
/// packet completing with no deferred window outstanding — costs three map
/// emptiness checks.
pub fn flush_all_windows_before_fence<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
) {
    // The two address-named rails carry the free-then-reuse hazard: they name raw
    // guest addresses with no mapping incarnation to refuse on, so nothing but
    // this ordering keeps them off memory the guest has reclaimed.
    flush_gva_windows_before_fence(state, host);
    flush_linear_windows_before_fence(state, host);
    // The mapping-keyed rails can refuse a replaced incarnation, so they are here
    // for the other hazard: a deferred writeback covers the whole attachment
    // extent while the guest writes the same IOSurface, and landing inside the
    // fence leaves no interval for that to happen in.
    flush_mapping_windows_before_fence(state, host);
}

/// The order [`flush_mapping_windows_before_fence`] lands windows in: render
/// windows oldest-first by `armed_seq`, then every other window.
///
/// Only the render rail carries an arm sequence, and only the render rail can
/// hold several live windows on one mapping at once (different planes, different
/// geometries at the same offset). Compute storage windows are keyed by the
/// dispatch span that produced them and are appended in key order, which is the
/// order every other flush trigger has always used.
#[cfg(feature = "backend-vulkan")]
fn mapping_windows_fence_order(
    state: &DeviceState,
) -> Vec<crate::model::ComputeStorageResidencyKey> {
    let mut render: Vec<(u64, crate::model::ComputeStorageResidencyKey)> = Vec::new();
    let mut rest: Vec<crate::model::ComputeStorageResidencyKey> = Vec::new();
    for (key, owner) in &state.compute_deferred_flush {
        match owner {
            crate::model::DeferredOwner::Render { armed_seq, .. } => {
                render.push((*armed_seq, *key))
            }
            crate::model::DeferredOwner::Storage { .. } => rest.push(*key),
        }
    }
    render.sort_unstable_by_key(|(seq, _)| *seq);
    render.into_iter().map(|(_, key)| key).chain(rest).collect()
}
