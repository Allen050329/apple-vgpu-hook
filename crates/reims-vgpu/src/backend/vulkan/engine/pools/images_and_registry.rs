//! The resident-image registry and the storage images beside it — the
//! [`ResourcePools`] methods whose unit is a *guest identity* rather than a
//! slot.
//!
//! A registry entry outlives any one submission. The guest names the same
//! surface across frames, so an entry is keyed by `TargetIdentity`, held alive
//! by a pin count, and reclaimed by an LRU cap walk rather than by a fence —
//! which is why none of this sits with the ring in
//! [`super::submission_and_buffers`].
//!
//! The two meet in one place: the idle-drain planner over there picks its
//! victims out of the registry order maintained here.
//!
//! `use super::*` is the seam. This is an `impl` chapter of the module that
//! declares `ResourcePools` and owns its fields, not a layer beneath it.

use super::*;

/// Band how long a resident had gone untouched before something read it,
/// against the cutoff that would have destroyed it.
///
/// The idle drain terminally destroys any non-pinned resident untouched for
/// `IDLE_TARGET_AGE_MS`, and nothing recreates a resident's content — so a draw
/// that samples one afterwards refuses permanently. What stops that today is
/// that a resident being read is touched by the read
/// ([`ResourcePools::registry_note_sampled_use`]), which only helps while the
/// gaps between reads stay under the cutoff. A guest that renders a layer, has
/// it occluded for longer than that, and then reveals it is the shape that loses
/// it.
///
/// `sampled_resident_missing` reading zero cannot say whether that is far away
/// or one slow frame away — it is a drop counter, and this project's own rule is
/// that a drop counter reading zero is not a measurement: a gap that peaks at
/// 50 ms and one that peaks at 1900 ms both report zero, and only one of them
/// says the cutoff has headroom. This is the reach that separates them, in time
/// rather than in slots, and it is the same instrument the bind tables already
/// have as their `reach_route` bands.
///
/// Read `resident_resample_past_cutoff` as the alarm: a resident that was read
/// after sitting longer than its own drain cutoff survived only because the
/// drain is throttled to `IDLE_TARGET_DRAIN_MAX_PER_CALL` per pass and had not
/// reached it yet. A non-zero reading there is the argument for changing the
/// drain — it does not mean content was lost, it means the margin is gone.
///
/// Bands are fractions of `IDLE_TARGET_AGE_MS` rather than absolute
/// milliseconds, so retuning the cutoff moves them with it and no reading is
/// ever quoted against a bound it did not come from. Division rather than
/// multiplication so a large gap cannot overflow the comparison.
///
/// # First reading, and it is not the comfortable one
///
/// Driven x86/PCI boot, `web-content-probe --churn 1`, whole run, against
/// `IDLE_TARGET_AGE_MS` = 2000 ms:
///
/// ```text
///   lt_eighth_cutoff   (<250 ms)      24643
///   lt_quarter_cutoff  (250-499 ms)     413
///   lt_half_cutoff     (500-999 ms)       5
///   under_cutoff       (1000-1999 ms)     1
///   past_cutoff        (>=2000 ms)        0
/// ```
///
/// The distribution is overwhelmingly under an eighth of the cutoff, which is
/// the answer that would have been assumed. The reading that matters is the
/// **1**: one resample arrived after its resident had sat between 1000 and
/// 1999 ms, so somewhere between half the budget and none of it was left. The
/// drain destroys at `last_touch_ms <= now - cutoff` and is throttled to
/// `IDLE_TARGET_DRAIN_MAX_PER_CALL` per pass, so that resident was not yet a
/// victim — but it is not the case that this workload stays an order of
/// magnitude clear of the cutoff, which is what `sampled_resident_missing=0`
/// on its own invites you to conclude.
///
/// Finer bands past the half mark are what would say whether that one sample sat
/// at 1.0 s or at 1.9 s, and those are very different margins. This does not
/// resolve it; it establishes that the question is live.
///
/// # Reading the count against `resident_samples`
///
/// The band total is about **twice** `sampled_gpu_binds` (25062 against 12531 on
/// the run above, exactly 2x). That is not a discrepancy:
/// [`ResourcePools::registry_note_sampled_use`] is called from two sites per
/// draw — the pre-pass loop that marks every sampled target before the render
/// target is ensured, and the resolve loop that binds them — while
/// `sampled_gpu_binds` increments once per bind. So these bands count *touches*
/// and `resident_samples` counts *binds*. Do not divide one by the other and
/// call the result a rate.
fn resident_resample_band(idle_ms: u64) -> &'static str {
    let cutoff = IDLE_TARGET_AGE_MS;
    if idle_ms < cutoff / 8 {
        "resident_resample_lt_eighth_cutoff"
    } else if idle_ms < cutoff / 4 {
        "resident_resample_lt_quarter_cutoff"
    } else if idle_ms < cutoff / 2 {
        "resident_resample_lt_half_cutoff"
    } else if idle_ms < cutoff {
        "resident_resample_under_cutoff"
    } else {
        "resident_resample_past_cutoff"
    }
}

/// Everything a creation site knows about a resident it has just built.
///
/// The stored [`ResidentTargetSlot`] is this plus what the registry owns and a
/// creation site does not: the birth state and the two LRU clocks. Handing over
/// this rather than a finished slot is what stops an arm getting either wrong,
/// and it is why [`ResourcePools::register_resident`] takes no `&mut` slot to
/// patch afterwards.
struct NewResident {
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    /// `vk::Framebuffer::null()` for a resident that is never bound as a
    /// standalone single-RT target — see
    /// [`ResidentTargetSlot::owed_framebuffer`], which is what every destroy
    /// path asks instead of testing this field itself.
    framebuffer: vk::Framebuffer,
    /// Null exactly when `framebuffer` is: the pass a per-slot framebuffer was
    /// built against, and the thing `registry_ensure` compares to decide
    /// whether a reused image needs its framebuffer rebuilt.
    render_pass: vk::RenderPass,
    width: u32,
    height: u32,
    generation: u64,
    color_format: vk::Format,
}

impl ResourcePools {
    pub(crate) unsafe fn acquire_storage_image(
        &mut self,
        ctx: &DeviceContext,
        key: StorageImageKey,
        counters: &EngineCounters,
    ) -> Result<StorageImageSlot, DrawError> {
        if let Some(slot) = self.storage_image_free.take(&key) {
            self.storage_image_live.push(StorageImageSlot {
                image: slot.image,
                memory: slot.memory,
                view: slot.view,
                key: slot.key,
            });
            return Ok(slot);
        }
        let format = key.format.vk_format();
        let image = ctx
            .device
            .create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(format)
                    .extent(vk::Extent3D {
                        width: key.width.max(1),
                        height: key.height.max(1),
                        depth: 1,
                    })
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(if key.sampled_only {
                        vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST
                    } else {
                        vk::ImageUsageFlags::STORAGE
                            | vk::ImageUsageFlags::TRANSFER_DST
                            | vk::ImageUsageFlags::TRANSFER_SRC
                    })
                    .initial_layout(vk::ImageLayout::UNDEFINED),
                None,
            )
            .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::PoolsCreateStorageImage, e)))?;
        counters.note_create();
        let req = ctx.device.get_image_memory_requirements(image);
        let mt = ctx
            .memory_type_for(req.memory_type_bits, MemoryClass::DeviceLocal)
            .ok_or({
                DrawError::Unsupported(reason::DrawReason::NoDeviceLocalMemoryForStorageImage {
                    memory_type_bits: req.memory_type_bits,
                })
            })?;
        let memory = allocate_memory_timed(
            ctx,
            &vk::MemoryAllocateInfo::default()
                .allocation_size(req.size)
                .memory_type_index(mt),
            AllocSite::StorageImage,
        )
        .map_err(|e| {
            ctx.device.destroy_image(image, None);
            DrawError::VkCall(VkCall::new(VkOp::PoolsAllocStorageImage, e))
        })?;
        counters.note_alloc();
        ctx.device
            .bind_image_memory(image, memory, 0)
            .map_err(|e| {
                ctx.device.free_memory(memory, None);
                ctx.device.destroy_image(image, None);
                DrawError::VkCall(VkCall::new(VkOp::PoolsBindStorageImage, e))
            })?;
        let view = ctx
            .device
            .create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format)
                    .subresource_range(color_subresource_range()),
                None,
            )
            .map_err(|e| {
                ctx.device.free_memory(memory, None);
                ctx.device.destroy_image(image, None);
                DrawError::VkCall(VkCall::new(VkOp::PoolsCreateStorageImageView, e))
            })?;
        counters.note_create();
        let slot = StorageImageSlot {
            image,
            memory,
            view,
            key,
        };
        self.storage_image_live.push(slot);
        Ok(slot)
    }

    pub(crate) fn recycle_storage_images(&mut self) {
        for slot in self.storage_image_live.drain(..) {
            self.storage_image_free.push_uncapped(slot.key, slot);
        }
    }

    pub(crate) unsafe fn acquire_resident_storage_image(
        &mut self,
        ctx: &DeviceContext,
        identity: ComputeStorageResidencyKey,
        key: StorageImageKey,
        seed_generation: u32,
        counters: &EngineCounters,
    ) -> Result<ResidentStorageImageUse, DrawError> {
        // A shape change re-keys the identity, and one identity holds one slot,
        // so the old image is destroyed. Every other removal in this registry
        // skips a pinned resident — the cap sweep and the age drain both do —
        // because a pin means the content owes a deferred writeback and exists
        // nowhere but that image. This path did not, so a re-shape between a
        // Store and its flush discarded accepted guest output with nothing said,
        // surfacing later and elsewhere as `StorageReadResidentAbsent`. Refuse
        // instead: the pin clears when the writeback lands and the next dispatch
        // re-keys normally, so this holds the request rather than ending it.
        if let Some(decline) = self.compute_rekey_refusal(&identity, key) {
            return Err(DrawError::ComputeExecution(decline));
        }
        if self
            .compute_storage_registry
            .get(&identity)
            .is_some_and(|resident| resident.slot.key != key)
        {
            if let Some(old) = self.compute_storage_registry.remove(&identity) {
                self.dispose(
                    &ctx.device,
                    DeferredHandle::Image {
                        image: old.slot.image,
                        view: old.slot.view,
                        memory: old.slot.memory,
                    },
                );
            }
            self.compute_storage_order
                .retain(|entry| entry != &identity);
        }
        let now = self.idle_clock_ms;
        if let Some(resident) = self.compute_storage_registry.get_mut(&identity) {
            resident.last_touch_ms = now;
            return Ok(ResidentStorageImageUse {
                slot: resident.slot,
                layout: resident.layout,
                generation_match: resident.generation == seed_generation,
            });
        }

        // Least-recently-used, skipping pinned residents (deferred-writeback
        // content whose only copy is on the GPU). When everything left is
        // pinned there is no victim and the registry soft-exceeds the cap
        // rather than lose unflushed content — the same trade the sibling
        // target registry's walk makes.
        while self.compute_storage_registry.len() >= COMPUTE_STORAGE_REGISTRY_CAP {
            let Some(victim) = self.compute_storage_eviction_victim() else {
                break;
            };
            self.compute_storage_order.retain(|entry| entry != &victim);
            if let Some(old) = self.compute_storage_registry.remove(&victim) {
                self.dispose(
                    &ctx.device,
                    DeferredHandle::Image {
                        image: old.slot.image,
                        view: old.slot.view,
                        memory: old.slot.memory,
                    },
                );
            }
            self.compute_storage_cap_evictions += 1;
        }

        // Reuse the common allocator, then detach its bookkeeping copy from
        // the transient live list: the registry now owns this allocation.
        let slot = self.acquire_storage_image(ctx, key, counters)?;
        let live = self.storage_image_live.pop().ok_or({
            DrawError::ComputeExecution(ComputeExecutionDecline::ResidentAllocatorLiveSlotMissing {
                identity,
                width: key.width,
                height: key.height,
                format: key.format,
            })
        })?;
        debug_assert_eq!(live.image, slot.image);
        self.compute_storage_registry.insert(
            identity,
            ResidentStorageImageSlot {
                slot,
                generation: 0,
                layout: vk::ImageLayout::UNDEFINED,
                pinned: false,
                last_touch_ms: now,
            },
        );
        self.compute_storage_order.push_back(identity);
        Ok(ResidentStorageImageUse {
            slot,
            layout: vk::ImageLayout::UNDEFINED,
            generation_match: false,
        })
    }

    /// The refusal owed when `identity` is already held at a different image
    /// shape by a resident that still owes a deferred writeback, or `None` when
    /// the re-key is safe to perform.
    ///
    /// Split out from [`ResourcePools::acquire_resident_storage_image`] so the
    /// pin check is unit-testable without a device, the same reason
    /// `take_aged_storage_residents` is split from its disposal half.
    pub(crate) fn compute_rekey_refusal(
        &self,
        identity: &ComputeStorageResidencyKey,
        key: StorageImageKey,
    ) -> Option<ComputeExecutionDecline> {
        let held = self
            .compute_storage_registry
            .get(identity)
            .filter(|resident| resident.pinned && resident.slot.key != key)?;
        Some(ComputeExecutionDecline::ResidentRekeyWouldDropPinned {
            identity: *identity,
            held_width: held.slot.key.width,
            held_height: held.slot.key.height,
            held_format: held.slot.key.format,
            wanted_width: key.width,
            wanted_height: key.height,
            wanted_format: key.format,
        })
    }

    /// Least-recently-used evictable compute-storage resident, or `None` when
    /// every remaining entry is pinned.
    ///
    /// Selected by minimum `last_touch_ms` rather than by the front of
    /// [`ResourcePools::compute_storage_order`], for the reason the sibling
    /// target registry's `cap_eviction_victim` states: insertion order makes a
    /// long-lived resident the permanent front and so the first victim of every
    /// burst, however hard the current chain is reading it. Iterating the order
    /// rather than the map keeps the choice deterministic — `min_by_key` returns
    /// the first minimum, so ties fall to the oldest-created entry, which is
    /// what this walk did for every entry before.
    ///
    /// O(n) per eviction rather than O(1) amortised. That is the same trade the
    /// target registry made and for the same reason: evictions are rare next to
    /// the reads this avoids promoting on.
    pub(crate) fn compute_storage_eviction_victim(&self) -> Option<ComputeStorageResidencyKey> {
        self.compute_storage_order
            .iter()
            .filter_map(|identity| {
                self.compute_storage_registry
                    .get(identity)
                    .filter(|resident| !resident.pinned)
                    .map(|resident| (identity, resident.last_touch_ms))
            })
            .min_by_key(|(_, touch)| *touch)
            .map(|(identity, _)| *identity)
    }

    /// Record that something read this resident, refreshing its reclaim stamp.
    ///
    /// Reading a resident is using it. The three read-only accessors below all
    /// mean "a guest chain is about to consume this image" — the stage-time
    /// guest-read skip, the copy-on-sample gate, and the flush/sample snapshot —
    /// so a produce-once/sample-many resident that is never dispatched into
    /// again is in continuous use while looking stone-cold to both reclaim
    /// rules, which read `last_touch_ms`: the cap sweep takes the minimum and
    /// the age drain compares it against its cutoff.
    ///
    /// The sibling target registry had exactly this defect and names it at its
    /// own call site: "aging it out between two attempts is how a recoverable
    /// not-ready became a permanent missing." Here the loss is a refused
    /// dispatch — `ResidentSampleAbsent` or `ResidentSeedGenerationLost` — so
    /// the stamp is written by the accessors themselves rather than by their
    /// callers, because a caller that forgets is indistinguishable from this
    /// bug.
    fn note_compute_resident_use(&mut self, identity: &ComputeStorageResidencyKey) {
        let touch = self.idle_clock_ms;
        if let Some(resident) = self.compute_storage_registry.get_mut(identity) {
            resident.last_touch_ms = touch;
        }
    }

    pub(crate) fn mark_resident_storage_image(
        &mut self,
        identity: &ComputeStorageResidencyKey,
        generation: u32,
        layout: vk::ImageLayout,
    ) {
        if let Some(resident) = self.compute_storage_registry.get_mut(identity) {
            resident.generation = generation;
            resident.layout = layout;
        }
    }

    /// Pin/unpin a resident against LRU eviction (deferred-writeback content
    /// whose only copy is the GPU image). No-op for an absent identity.
    pub(crate) fn pin_resident_storage(
        &mut self,
        identity: &ComputeStorageResidencyKey,
        pinned: bool,
    ) {
        if let Some(resident) = self.compute_storage_registry.get_mut(identity) {
            resident.pinned = pinned;
        }
    }

    /// Record the post-flush layout of a resident (the flush read transitions
    /// it to TRANSFER_SRC_OPTIMAL).
    pub(crate) fn set_resident_storage_layout(
        &mut self,
        identity: &ComputeStorageResidencyKey,
        layout: vk::ImageLayout,
    ) {
        if let Some(resident) = self.compute_storage_registry.get_mut(identity) {
            resident.layout = layout;
        }
    }

    /// Generation of a resident compute storage image, if one is registered.
    /// Used by the runtime to decide a stage-time guest-read skip.
    ///
    /// Takes `&mut self` to record the read — see
    /// [`ResourcePools::note_compute_resident_use`]. A skip taken against this
    /// answer means the dispatch is about to consume the resident, which is the
    /// definition of using it.
    pub(crate) fn compute_resident_generation(
        &mut self,
        identity: &ComputeStorageResidencyKey,
    ) -> Option<u32> {
        self.note_compute_resident_use(identity);
        self.compute_storage_registry
            .get(identity)
            .map(|resident| resident.generation)
    }

    /// Generation + engine format of a resident compute storage image, if one
    /// is registered. Read-only — used by the runtime to decide a stage-time
    /// copy-on-sample skip (the format must match what the sampled view will
    /// bind, or the engine's resident-bind shape guard would fail every run).
    pub(crate) fn compute_resident_sample_source(
        &mut self,
        identity: &ComputeStorageResidencyKey,
    ) -> Option<(u32, StorageImageFormat)> {
        self.note_compute_resident_use(identity);
        self.compute_storage_registry
            .get(identity)
            .map(|resident| (resident.generation, resident.slot.key.format))
    }

    /// Snapshot of a resident storage image for a copy-on-sample source:
    /// `(image, key, generation, current layout)`.
    pub(crate) fn compute_resident_snapshot(
        &mut self,
        identity: &ComputeStorageResidencyKey,
    ) -> Option<(vk::Image, StorageImageKey, u32, vk::ImageLayout)> {
        self.note_compute_resident_use(identity);
        self.compute_storage_registry.get(identity).map(|resident| {
            (
                resident.slot.image,
                resident.slot.key,
                resident.generation,
                resident.layout,
            )
        })
    }

    // --- Target registry (workstream D) ------------------------------------

    pub(crate) fn registry_get(&self, identity: &TargetIdentity) -> Option<&ResidentTargetSlot> {
        self.registry.get(identity)
    }

    /// Forget the resident registered under `identity`, recording `why`, and
    /// hand back the slot that was removed.
    ///
    /// Split out from [`Self::retire_resident`] for the same reason
    /// [`Self::cap_eviction_victim`] is split out of
    /// [`Self::evict_registry_to_cap`]: retiring needs a live `DeviceContext` to
    /// dispose what it removes, and the bookkeeping — which is the part that was
    /// diverging — is worth testing without a GPU.
    ///
    /// Every path that removes a live entry comes through here, including the
    /// idle drain in [`super::submission_and_buffers`], which disposes on its
    /// own terms and so is not a [`Self::retire_resident`] caller. It is the
    /// death counterpart of [`Self::register_resident`], and the pair is why
    /// `registry` and `registry_order` cannot fall out of step.
    ///
    /// `registry_order` is pruned whether or not the map held the entry, which
    /// is what every caller did around its own copy. Nothing is recorded for an
    /// identity that held no resident: [`Self::prior_reclaim`] deliberately does
    /// not guess between "never held one" and "reclaimed too long ago", and a
    /// record for a removal that did not happen would make it guess wrong.
    pub(super) fn unregister_resident(
        &mut self,
        identity: &TargetIdentity,
        why: ResidentReclaim,
    ) -> Option<ResidentTargetSlot> {
        let old = self.registry.remove(identity);
        self.registry_order.retain(|k| k != identity);
        let old = old?;
        if old.pin_count == 0 {
            self.registry_non_pinned_adjust(Self::slot_attachment_bytes(&old), false);
        }
        self.note_resident_reclaimed(identity, why);
        Some(old)
    }

    /// Hand a resident's framebuffer to the deferred-destroy path, if it has
    /// one. [`ResidentTargetSlot::owed_framebuffer`] is where "if" is decided,
    /// and why.
    ///
    /// # Safety
    /// The caller must already have taken this framebuffer out of the registry,
    /// or be about to overwrite the field it came from. The graveyard frees it
    /// once the ring says no command buffer still references it.
    pub(super) unsafe fn dispose_owed_framebuffer(
        &mut self,
        device: &ash::Device,
        owed: Option<vk::Framebuffer>,
    ) {
        if let Some(fb) = owed {
            self.dispose(device, DeferredHandle::Framebuffer(fb));
        }
    }

    /// Store a newly created resident under `identity` and put it at the back
    /// of the LRU order.
    ///
    /// One home for a resident's *birth*, as [`Self::unregister_resident`] is
    /// one home for its death. Both `registry_ensure*` arms wrote all of this
    /// out, and it is three rules rather than one:
    ///
    /// - **The birth state.** Nothing has drawn into an image created a line
    ///   ago, so it carries no content stamp and no epoch; nothing has
    ///   transitioned it, so its layout is `UNDEFINED`; and no window holds it,
    ///   so it is unpinned. These are not defaults a creation site may pick —
    ///   `registry_mark_ready`, the type-11 LOAD gate and the idle drain each
    ///   read one of them, and an arm that guessed differently would be
    ///   answering a question the others think they already asked.
    /// - **The LRU clocks belong to the registry.** `use_seq` comes from
    ///   `use_clock`, which has to advance exactly once per registration or the
    ///   cap walk's ordering ties break arbitrarily. A creation site does not
    ///   own that counter and cannot see the other arm's registrations.
    /// - **`registry` and `registry_order` are written together.** They are one
    ///   structure split for lookup and for order. An entry in the map but not
    ///   the order is a resident no sweep can ever choose; one in the order but
    ///   not the map is a victim that frees nothing.
    fn register_resident(&mut self, identity: &TargetIdentity, new: NewResident) {
        self.use_clock += 1;
        let (last_touch_ms, use_seq) = (self.idle_clock_ms, self.use_clock);
        self.registry.insert(
            identity.clone(),
            ResidentTargetSlot {
                image: new.image,
                memory: new.memory,
                view: new.view,
                framebuffer: new.framebuffer,
                render_pass: new.render_pass,
                width: new.width,
                height: new.height,
                generation: new.generation,
                content_ready: false,
                content_epoch: None,
                layout: vk::ImageLayout::UNDEFINED,
                color_format: new.color_format,
                pin_count: 0,
                last_touch_ms,
                use_seq,
            },
        );
        self.registry_order.push_back(identity.clone());
        // Born unpinned (see the birth-state rule above), so it joins the
        // non-pinned totals unconditionally.
        let bytes = self
            .registry
            .get(identity)
            .map(Self::slot_attachment_bytes)
            .unwrap_or(0);
        self.registry_non_pinned_adjust(bytes, true);
    }

    /// Drop the resident registered under `identity`, recording `why`, returning
    /// its image/memory/view to `target_free` and its framebuffer to the
    /// graveyard. Returns the slot that was removed, or `None` when nothing was
    /// registered.
    ///
    /// The recycling exit for a live registry entry: the two `registry_ensure*`
    /// recreate arms and [`Self::evict_registry_to_cap`] all take it, and were
    /// copies of one another before they did. The MRT-secondary path recorded
    /// no reclaim reason at all, so a later draw whose sampled source that path
    /// had recreated could not be told "taken from under you" from "never
    /// existed", which is the whole point of
    /// [`Self::note_resident_reclaimed`]. The primary path was the one that
    /// disposed `old.framebuffer` without asking whether the slot had one.
    ///
    /// It is not the *only* exit, and reading it as one is how the fourth stayed
    /// out of step. The idle drain destroys rather than recycles and does not
    /// count a `target_evict`, both deliberately, so it cannot come through
    /// here — what it shares is [`Self::unregister_resident`] and
    /// [`ResidentTargetSlot::owed_framebuffer`], which is why the bookkeeping
    /// and the null question are each their own function rather than lines in
    /// this body.
    unsafe fn retire_resident(
        &mut self,
        ctx: &DeviceContext,
        identity: &TargetIdentity,
        why: ResidentReclaim,
        counters: &EngineCounters,
    ) -> Option<ResidentTargetSlot> {
        let old = self.unregister_resident(identity, why)?;
        self.dispose_owed_framebuffer(&ctx.device, old.owed_framebuffer());
        self.dispose(
            &ctx.device,
            DeferredHandle::RecycleTarget(FreeTargetImage {
                image: old.image,
                memory: old.memory,
                view: old.view,
                width: old.width,
                height: old.height,
                format: old.color_format,
            }),
        );
        counters
            .target_evicts
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Some(old)
    }

    /// Ensure a resident target exists for `identity` with the given geometry + pass.
    /// Image/memory persist across Load vs Clear render-pass changes; only the
    /// framebuffer is rebuilt when the pass handle differs.
    /// `protect` shields one additional identity (a same-draw GPU seed
    /// source) from the capacity sweep, exactly like a pinned slot.
    #[allow(
        clippy::too_many_arguments,
        reason = "resident creation mirrors the target identity, format, seed, and protection set"
    )]
    pub(crate) unsafe fn registry_ensure(
        &mut self,
        ctx: &DeviceContext,
        identity: TargetIdentity,
        width: u32,
        height: u32,
        render_pass: vk::RenderPass,
        generation: u64,
        bgra: bool,
        protect: Option<&TargetIdentity>,
        counters: &EngineCounters,
    ) -> Result<&ResidentTargetSlot, DrawError> {
        // Compatible geometry + gen + format: reuse image; rebuild FB if pass
        // changed. A format change must recreate the image, not just the
        // framebuffer — an RGBA image under a BGRA pass is invalid.
        let format = translate::pixel::resident_color(bgra);
        if let Some(slot) = self.registry.get(&identity) {
            if slot.reusable_for(width, height, generation, format) {
                if slot.render_pass == render_pass {
                    counters
                        .gpu_load_hits
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let touch = self.idle_clock_ms;
                    self.use_clock += 1;
                    let seq = self.use_clock;
                    let slot = self.registry.get_mut(&identity).unwrap();
                    slot.last_touch_ms = touch;
                    slot.use_seq = seq;
                    return Ok(slot);
                }
                // Same image, new pass → recreate framebuffer only.
                let view = slot.view;
                let old_fb = slot.owed_framebuffer();
                let attachments = [view];
                let framebuffer = ctx
                    .device
                    .create_framebuffer(
                        &vk::FramebufferCreateInfo::default()
                            .render_pass(render_pass)
                            .attachments(&attachments)
                            .width(width)
                            .height(height)
                            .layers(1),
                        None,
                    )
                    .map_err(|e| {
                        DrawError::VkCall(VkCall::new(VkOp::PoolsCreateRegistryFramebuffer, e))
                    })?;
                counters.note_create();
                self.dispose_owed_framebuffer(&ctx.device, old_fb);
                let slot = self.registry.get_mut(&identity).unwrap();
                slot.framebuffer = framebuffer;
                slot.render_pass = render_pass;
                counters
                    .gpu_load_hits
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Ok(self.registry.get(&identity).unwrap());
            }
            // Geometry/gen mismatch → destroy and recreate.
            if let Some(old) =
                self.retire_resident(ctx, &identity, ResidentReclaim::Recreated, counters)
            {
                if old.generation != generation {
                    counters
                        .gen_mismatch
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
            }
        }
        // Cap the *non-pinned* (evictable) population at REGISTRY_CAP, shielding
        // the just-resolved `protect` identity from its own eviction.
        self.evict_registry_to_cap(ctx, counters, protect);
        let usage = vk::ImageUsageFlags::COLOR_ATTACHMENT
            | vk::ImageUsageFlags::INPUT_ATTACHMENT
            | vk::ImageUsageFlags::TRANSFER_SRC
            | vk::ImageUsageFlags::TRANSFER_DST
            | vk::ImageUsageFlags::SAMPLED;
        // Reuse a recycled image+memory+view of identical (geometry, format)
        // before allocating a fresh one — the usage set is identical across all
        // registry targets, so a recycled image of the same geometry/format is
        // bind-compatible. This is what collapses the per-frame realloc storm a
        // per-generation target (video) would otherwise pay: skips vkCreateImage
        // + vkAllocateMemory + bind + view (and their note_create/note_alloc).
        // The recycled contents are stale — the slot is inserted with
        // layout=UNDEFINED / content_ready=false, and a fresh framebuffer is
        // always built below (it binds this specific render_pass).
        let (image, memory, view) = if let Some(free) = self.take_free_target(width, height, format)
        {
            (free.image, free.memory, free.view)
        } else {
            let image = ctx
                .device
                .create_image(
                    &vk::ImageCreateInfo::default()
                        .image_type(vk::ImageType::TYPE_2D)
                        .format(format)
                        .extent(vk::Extent3D {
                            width,
                            height,
                            depth: 1,
                        })
                        .mip_levels(1)
                        .array_layers(1)
                        .samples(vk::SampleCountFlags::TYPE_1)
                        .tiling(vk::ImageTiling::OPTIMAL)
                        .usage(usage)
                        .initial_layout(vk::ImageLayout::UNDEFINED),
                    None,
                )
                .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::PoolsCreateRegistryTarget, e)))?;
            counters.note_create();
            let ireq = ctx.device.get_image_memory_requirements(image);
            let memory = match self.bind_image_slab(
                ctx,
                image,
                &ireq,
                VkOp::PoolsBindRegistryTarget,
                counters,
            ) {
                Ok(m) => m,
                Err(error) => {
                    ctx.device.destroy_image(image, None);
                    return Err(error);
                }
            };
            let view = match ctx.device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format)
                    .subresource_range(color_subresource_range()),
                None,
            ) {
                Ok(v) => v,
                Err(e) => {
                    self.free_image_slab(&ctx.device, image);
                    ctx.device.destroy_image(image, None);
                    return Err(DrawError::VkCall(VkCall::new(
                        VkOp::PoolsCreateRegistryView,
                        e,
                    )));
                }
            };
            counters.note_create();
            (image, memory, view)
        };
        let attachments = [view];
        let framebuffer = match ctx.device.create_framebuffer(
            &vk::FramebufferCreateInfo::default()
                .render_pass(render_pass)
                .attachments(&attachments)
                .width(width)
                .height(height)
                .layers(1),
            None,
        ) {
            Ok(fb) => fb,
            Err(e) => {
                ctx.device.destroy_image_view(view, None);
                self.free_image_slab(&ctx.device, image);
                ctx.device.destroy_image(image, None);
                return Err(DrawError::VkCall(VkCall::new(
                    VkOp::PoolsCreateRegistryFramebuffer,
                    e,
                )));
            }
        };
        counters.note_create();
        self.register_resident(
            &identity,
            NewResident {
                image,
                memory,
                view,
                framebuffer,
                render_pass,
                width,
                height,
                generation,
                color_format: format,
            },
        );
        Ok(self.registry.get(&identity).unwrap())
    }

    /// Ensure a resident color attachment of an arbitrary Vulkan format (MRT
    /// secondary path — the primary single-RT `registry_ensure` only speaks
    /// `bgra`). No per-slot framebuffer is built: a secondary attachment is
    /// only ever bound as attachment N of an ad-hoc MRT framebuffer or sampled
    /// via its view, never as a standalone single-RT target. Reuse requires an
    /// exact (geometry, generation, format) match. Returns (image, view).
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn registry_ensure_color(
        &mut self,
        ctx: &DeviceContext,
        identity: TargetIdentity,
        width: u32,
        height: u32,
        generation: u64,
        format: vk::Format,
        counters: &EngineCounters,
    ) -> Result<(vk::Image, vk::ImageView), DrawError> {
        if let Some(slot) = self.registry.get(&identity) {
            if slot.reusable_for(width, height, generation, format) {
                counters
                    .gpu_load_hits
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Ok((slot.image, slot.view));
            }
            // Geometry / gen / format mismatch → destroy and recreate.
            self.retire_resident(ctx, &identity, ResidentReclaim::Recreated, counters);
        }
        // Cap the *non-pinned* population (skip pinned slots), same LRU
        // discipline as the primary `registry_ensure` — pinned deferred windows
        // are bounded separately and excluded from the cap count. No `protect`
        // here: this color path has no just-resolved identity to shield.
        self.evict_registry_to_cap(ctx, counters, None);
        let usage = vk::ImageUsageFlags::COLOR_ATTACHMENT
            | vk::ImageUsageFlags::INPUT_ATTACHMENT
            | vk::ImageUsageFlags::TRANSFER_SRC
            | vk::ImageUsageFlags::TRANSFER_DST
            | vk::ImageUsageFlags::SAMPLED;
        // Reuse a recycled image+memory+view of identical (geometry, format)
        // before allocating — same recycle discipline as the primary
        // `registry_ensure` (the usage set is identical, so images cross-flow
        // between the two paths by geometry+format). Skips the create/alloc/bind/
        // view + their note_create/note_alloc; recycled contents are stale, so
        // the slot below is inserted layout=UNDEFINED / content_ready=false.
        let (image, memory, view) = if let Some(free) = self.take_free_target(width, height, format)
        {
            (free.image, free.memory, free.view)
        } else {
            let image = ctx
                .device
                .create_image(
                    &vk::ImageCreateInfo::default()
                        .image_type(vk::ImageType::TYPE_2D)
                        .format(format)
                        .extent(vk::Extent3D {
                            width,
                            height,
                            depth: 1,
                        })
                        .mip_levels(1)
                        .array_layers(1)
                        .samples(vk::SampleCountFlags::TYPE_1)
                        .tiling(vk::ImageTiling::OPTIMAL)
                        .usage(usage)
                        .initial_layout(vk::ImageLayout::UNDEFINED),
                    None,
                )
                .map_err(|e| {
                    DrawError::VkCall(VkCall::new(VkOp::PoolsCreateMrtSecondaryTarget, e))
                })?;
            counters.note_create();
            let ireq = ctx.device.get_image_memory_requirements(image);
            let imt = ctx
                .memory_type_for(ireq.memory_type_bits, MemoryClass::DeviceLocal)
                .ok_or_else(|| {
                    ctx.device.destroy_image(image, None);
                    DrawError::Unsupported(reason::DrawReason::NoDeviceLocalMemoryForMrtSecondary {
                        memory_type_bits: ireq.memory_type_bits,
                    })
                })?;
            let memory = allocate_memory_timed(
                ctx,
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(ireq.size)
                    .memory_type_index(imt),
                AllocSite::MrtSecondary,
            )
            .map_err(|e| {
                ctx.device.destroy_image(image, None);
                DrawError::VkCall(VkCall::new(VkOp::PoolsAllocMrtSecondary, e))
            })?;
            counters.note_alloc();
            ctx.device
                .bind_image_memory(image, memory, 0)
                .map_err(|e| {
                    ctx.device.free_memory(memory, None);
                    ctx.device.destroy_image(image, None);
                    DrawError::VkCall(VkCall::new(VkOp::PoolsBindMrtSecondary, e))
                })?;
            let view = ctx
                .device
                .create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(image)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(format)
                        .subresource_range(color_subresource_range()),
                    None,
                )
                .map_err(|e| {
                    ctx.device.free_memory(memory, None);
                    ctx.device.destroy_image(image, None);
                    DrawError::VkCall(VkCall::new(VkOp::PoolsCreateMrtSecondaryView, e))
                })?;
            counters.note_create();
            (image, memory, view)
        };
        self.register_resident(
            &identity,
            NewResident {
                image,
                memory,
                view,
                // No per-slot framebuffer and so no pass it was built against:
                // this arm's residents are bound as attachment N of an ad-hoc
                // MRT framebuffer, or sampled through the view.
                framebuffer: vk::Framebuffer::null(),
                render_pass: vk::RenderPass::null(),
                width,
                height,
                generation,
                color_format: format,
            },
        );
        Ok((image, view))
    }

    /// Allocate a transient D32_SFLOAT depth attachment (image + memory + view)
    /// sized to `width`x`height`. The caller owns it for exactly one draw and
    /// must dispose it deferred (`DeferredHandle::Image`) after submit — the CB
    /// still references it until its fence signals. Depth is never read back, so
    /// no TRANSFER_SRC usage.
    ///
    /// # It does not recycle, and on this pathway it never runs
    ///
    /// Every sibling allocator here reuses before allocating; this one does not.
    /// `exec` creates one per draw that carries depth state and disposes it at
    /// the end of that same draw. That was once by far the largest allocator in
    /// the engine: driven boots read `vk_alloc_sites transient_depth=5374:21225`
    /// and later `4623:18257` — thousands of `vkAllocateMemory` calls totalling
    /// ~18-21 GiB, against `slab_block=41:2568` for every resident colour target
    /// in the same boot. A depth [`FreePool`] was built against exactly that,
    /// measured at a 4 % improvement, and reverted as more code for no benefit.
    ///
    /// **A driven boot on the current build allocates zero.** x86/Vulkan,
    /// measured across all 126 `vk_alloc_sites` census windows spanning the boot
    /// (Safari page loads and page-downs, a title-bar drag, a wallpaper drag,
    /// Chess, the WebGL aquarium, then `killall` teardown): every window read
    /// `transient_depth=0:0`. The zero is not a broken probe — the counter is the
    /// `allocate_memory` wrapper's, keyed [`AllocSite::TransientDepth`], and
    /// `slab_block` and `staging` moved by thousands in the same lines.
    ///
    /// Nothing was refused on the way, either: zero `shader_state_degraded`, zero
    /// `depth_compare_unmapped`, zero `depth_load_unsupported_transient` in the
    /// whole log. So the guest is not asking for depth and being turned away — it
    /// is not asking. `resources.depth` is set only where the guest's
    /// depth-stencil descriptor decodes with a mapped compare, and no draw we
    /// executed carried one. The likely reading, NOT measured, is that a 3D
    /// application's depth work happens before its surface reaches our stream and
    /// what we execute is the compositor's 2D layer work.
    ///
    /// So do not build the depth pool. The premise it was queued against — a
    /// per-draw allocation storm — does not reproduce, and a fourth recycle pool
    /// would be more mechanism guarding nothing. `vk_alloc_sites transient_depth`
    /// is the number that would say a future workload changed this; until it is
    /// nonzero there is nothing here to recycle.
    ///
    pub(crate) unsafe fn create_transient_depth(
        &mut self,
        ctx: &DeviceContext,
        width: u32,
        height: u32,
        with_stencil: bool,
        counters: &EngineCounters,
    ) -> Result<(vk::Image, vk::DeviceMemory, vk::ImageView), DrawError> {
        // Device-queried combined depth-stencil format when the bound state runs
        // the stencil test (D32_S8 preferred, D24_S8 fallback — see
        // DeviceContext::depth_stencil_format); plain D32_SFLOAT (no stencil
        // aspect) otherwise, which is spec-mandatory. Depth is 32-bit float in
        // the preferred case; the D24_S8 fallback is 24-bit UNORM depth, which
        // the stencil-test path tolerates (it asserts stencil, not depth bits).
        let (format, aspect) = if with_stencil {
            (
                ctx.depth_stencil_format,
                vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL,
            )
        } else {
            (
                translate::pixel::TRANSIENT_DEPTH_FORMAT,
                vk::ImageAspectFlags::DEPTH,
            )
        };
        let image = ctx
            .device
            .create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(format)
                    .extent(vk::Extent3D {
                        width,
                        height,
                        depth: 1,
                    })
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
                    .initial_layout(vk::ImageLayout::UNDEFINED),
                None,
            )
            .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::PoolsCreateDepthImage, e)))?;
        counters.note_create();
        let ireq = ctx.device.get_image_memory_requirements(image);
        let imt = ctx
            .memory_type_for(ireq.memory_type_bits, MemoryClass::DeviceLocal)
            .ok_or_else(|| {
                ctx.device.destroy_image(image, None);
                DrawError::Unsupported(reason::DrawReason::NoDeviceLocalMemoryForDepth {
                    memory_type_bits: ireq.memory_type_bits,
                })
            })?;
        let memory = allocate_memory_timed(
            ctx,
            &vk::MemoryAllocateInfo::default()
                .allocation_size(ireq.size)
                .memory_type_index(imt),
            AllocSite::TransientDepth,
        )
        .map_err(|e| {
            ctx.device.destroy_image(image, None);
            DrawError::VkCall(VkCall::new(VkOp::PoolsAllocDepth, e))
        })?;
        counters.note_alloc();
        ctx.device
            .bind_image_memory(image, memory, 0)
            .map_err(|e| {
                ctx.device.free_memory(memory, None);
                ctx.device.destroy_image(image, None);
                DrawError::VkCall(VkCall::new(VkOp::PoolsBindDepth, e))
            })?;
        let view = ctx
            .device
            .create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: aspect,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    }),
                None,
            )
            .map_err(|e| {
                ctx.device.free_memory(memory, None);
                ctx.device.destroy_image(image, None);
                DrawError::VkCall(VkCall::new(VkOp::PoolsCreateDepthView, e))
            })?;
        counters.note_create();
        Ok((image, memory, view))
    }

    /// Build an ad-hoc MRT framebuffer over `views` (primary slot 0 + secondary
    /// slots 1..) under `render_pass`. Not cached — the caller disposes it via
    /// `dispose(Framebuffer)` after the draw is sealed onto the ring slot.
    pub(crate) unsafe fn create_mrt_framebuffer(
        &mut self,
        ctx: &DeviceContext,
        render_pass: vk::RenderPass,
        views: &[vk::ImageView],
        width: u32,
        height: u32,
        counters: &EngineCounters,
    ) -> Result<vk::Framebuffer, DrawError> {
        let fb = ctx
            .device
            .create_framebuffer(
                &vk::FramebufferCreateInfo::default()
                    .render_pass(render_pass)
                    .attachments(views)
                    .width(width)
                    .height(height)
                    .layers(1),
                None,
            )
            .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::PoolsCreateMrtFramebuffer, e)))?;
        counters.note_create();
        Ok(fb)
    }

    /// Mark a resident ready with an explicit post-pass layout (MRT secondary
    /// resolves to SHADER_READ_ONLY_OPTIMAL; the primary uses
    /// `registry_mark_ready`'s TRANSFER_SRC_OPTIMAL).
    pub(crate) fn registry_mark_ready_at(
        &mut self,
        identity: &TargetIdentity,
        layout: vk::ImageLayout,
    ) {
        if let Some(slot) = self.registry.get_mut(identity) {
            slot.content_ready = true;
            slot.content_epoch = None;
            slot.layout = layout;
        }
    }

    /// Pin/unpin a resident render target against LRU eviction (deferred
    /// render Stores). Pins are counted, not boolean: a surface can have several
    /// deferred windows armed at once and each holds one count, so the slot
    /// stays protected until every holder unpins. Returns false when the identity is absent or (for pinning) its
    /// content is not ready — callers must fall back to the synchronous
    /// Store. Unpin saturates at zero (a spurious unpin never underflows).
    pub(crate) fn pin_resident_target(&mut self, identity: &TargetIdentity, pinned: bool) -> bool {
        let Some(slot) = self.registry.get_mut(identity) else {
            return false;
        };
        if pinned && !slot.content_ready {
            return false;
        }
        // Counted pins, so only the 0 <-> 1 crossings change whether this slot is
        // in the non-pinned totals. A second pin, or an unpin that leaves one
        // holder, moves nothing — and a saturating unpin at zero must not add a
        // slot that was already counted.
        let before_non_pinned = slot.pin_count == 0;
        if pinned {
            slot.pin_count += 1;
        } else {
            slot.pin_count = slot.pin_count.saturating_sub(1);
        }
        let after_non_pinned = slot.pin_count == 0;
        if before_non_pinned != after_non_pinned {
            let bytes = Self::slot_attachment_bytes(slot);
            self.registry_non_pinned_adjust(bytes, after_non_pinned);
        }
        true
    }

    /// Mark a resident ready after a draw stored into it.
    ///
    /// Clears `content_epoch`: this image's pixels just changed, and until
    /// something publishes them as the mapping's content and stamps the slot,
    /// nothing may claim they match a mapping epoch. Every path that ends in a
    /// resident holding new pixels comes through here or
    /// [`Self::registry_mark_ready_at`], which is what keeps the reset total
    /// rather than a list of the writers somebody remembered.
    pub(crate) fn registry_mark_ready(&mut self, identity: &TargetIdentity) {
        if let Some(slot) = self.registry.get_mut(identity) {
            slot.content_ready = true;
            slot.content_epoch = None;
            // Draw pass final_layout is TRANSFER_SRC_OPTIMAL.
            slot.layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
        }
    }

    /// Record that this resident's pixels are the mapping's content as of
    /// `epoch`. Refuses (returns false) unless the slot exists and is
    /// content_ready — stamping an image no draw has stored into would vouch
    /// for undefined memory.
    pub(crate) fn registry_stamp_content_epoch(
        &mut self,
        identity: &TargetIdentity,
        epoch: u32,
    ) -> bool {
        match self.registry.get_mut(identity) {
            Some(slot) if slot.content_ready => {
                slot.content_epoch = Some(epoch);
                true
            }
            _ => false,
        }
    }

    pub(crate) fn registry_set_layout(
        &mut self,
        identity: &TargetIdentity,
        layout: vk::ImageLayout,
    ) {
        if let Some(slot) = self.registry.get_mut(identity) {
            slot.layout = layout;
        }
    }

    /// Count of registry residents NOT held by a deferred-write pin — the
    /// LRU-evictable (active) working set the `REGISTRY_CAP` bounds. Pinned slots
    /// are bounded separately (by the arming rail's own window cap) and excluded
    /// so a pinned burst cannot force the active set into eviction thrash.
    ///
    /// O(1). This and [`Self::non_pinned_registry_bytes`] each walked the whole
    /// registry, on every admit — free only for as long as `REGISTRY_CAP` holds
    /// the population near 320, which is the bound the byte measurement exists to
    /// remove. [`Self::registry_non_pinned`] is maintained instead at the sites
    /// that can change either total, and
    /// `non_pinned_registry_totals_by_walk` is the walk kept as the thing to
    /// check it against (test-only, so not linkable from here).
    fn non_pinned_registry_len(&self) -> usize {
        self.registry_non_pinned.count
    }

    /// Attachment bytes the same non-pinned set occupies: `w x h x texel` summed
    /// over every slot [`Self::non_pinned_registry_len`] counts.
    ///
    /// The number [`REGISTRY_CAP`]'s doc argues from — it says "slots are cheap;
    /// the real VRAM guard is per-image bytes", and then bounds the slots. It
    /// also quoted ~516 MiB for a burst and a ~1005 MiB idle baseline, both from
    /// a `vram` census line that no longer exists anywhere in this crate, so
    /// until this counter there was nothing in the device that could say what a
    /// count of 320 costs. 320 slots is 5 MiB of 16x16 scratch or 10 GiB of 4K.
    fn non_pinned_registry_bytes(&self) -> u64 {
        self.registry_non_pinned.bytes
    }

    /// One slot's contribution to [`Self::non_pinned_registry_bytes`].
    ///
    /// Attachment footprint, not allocation footprint: it does not know tiling
    /// padding or the slab's rounding, and a format
    /// [`crate::backend::vulkan::translate::pixel::bytes_per_texel`] declines
    /// (block-compressed, multi-planar — neither of which a colour attachment
    /// uses) contributes nothing rather than a guessed size. So the total is a
    /// lower bound on VRAM, which is the safe direction for a figure that exists
    /// to decide whether a bound is too loose.
    fn slot_attachment_bytes(slot: &ResidentTargetSlot) -> u64 {
        crate::backend::vulkan::translate::pixel::bytes_per_texel(slot.color_format)
            .map(|texel| u64::from(slot.width) * u64::from(slot.height) * u64::from(texel))
            .unwrap_or(0)
    }

    /// The same two totals recomputed from the registry, for the test that says
    /// the maintained pair still agrees with it.
    ///
    /// Kept because the maintained pair has three writers and a fourth mutation
    /// site would desync it in silence — a resident that stopped being counted
    /// makes the population read smaller than it is, which is the direction that
    /// lets the cap sit above its own bound. This is what a desync is diffed
    /// against.
    #[cfg(test)]
    fn non_pinned_registry_totals_by_walk(&self) -> NonPinnedTotals {
        let non_pinned = || self.registry.values().filter(|slot| slot.pin_count == 0);
        NonPinnedTotals {
            count: non_pinned().count(),
            bytes: non_pinned().map(Self::slot_attachment_bytes).sum(),
        }
    }

    /// Fold one slot into or out of the maintained non-pinned totals.
    ///
    /// Every change of "is this slot non-pinned" goes through here, so the count
    /// and the bytes cannot move apart from each other, or be updated at two
    /// sites and forgotten at a third.
    fn registry_non_pinned_adjust(&mut self, slot_bytes: u64, joined: bool) {
        let totals = &mut self.registry_non_pinned;
        if joined {
            totals.count += 1;
            totals.bytes += slot_bytes;
        } else {
            totals.count = totals.count.saturating_sub(1);
            totals.bytes = totals.bytes.saturating_sub(slot_bytes);
        }
    }

    /// Fold the current non-pinned population into the high-water band and
    /// return it.
    ///
    /// Split from [`Self::evict_registry_to_cap`] for the same reason
    /// [`Self::cap_eviction_victim`] is: that function is `unsafe` and needs a
    /// live `DeviceContext` to dispose what it evicts, so nothing about it can
    /// be exercised without a GPU — and an instrument that is never tested is
    /// one that can silently read zero forever, which is the failure it exists
    /// to prevent in the first place.
    ///
    /// Called at the top of the capacity walk, before any eviction: that is the
    /// one point every admission passes through, and it is where the population
    /// is at its highest. Sampling after the walk would record the cap back
    /// rather than the demand that crossed it.
    /// Both bands are folded here, from the same sample, so `peak` and
    /// `peak_bytes` describe one population rather than two moments — the two
    /// together are what say whether a slot count is a sane proxy for VRAM.
    fn note_registry_reach(&mut self) -> usize {
        let non_pinned = self.non_pinned_registry_len();
        self.registry_non_pinned_peak = self.registry_non_pinned_peak.max(non_pinned as u64);
        self.registry_non_pinned_peak_bytes = self
            .registry_non_pinned_peak_bytes
            .max(self.non_pinned_registry_bytes());
        non_pinned
    }

    /// Evict non-pinned resident targets (LRU, oldest at the front of
    /// `registry_order`) until the non-pinned population is at or below
    /// [`REGISTRY_CAP`]. Pinned slots (deferred render Stores whose only copy is
    /// on the GPU) and an optional `protect`ed identity rotate to the back
    /// instead of evicting — they are bounded separately and must not count
    /// toward the cap, or a pinned burst would force the active set out (thrash).
    /// One full rotation is the budget. `registry_order` is a `VecDeque`, so each
    /// front pop / rotate-to-back is O(1) and the whole sweep is O(n) — not the
    /// O(n²) a `Vec` front-`remove(0)` per rotation would cost under a large
    /// pinned population (`reg=512` measured under multi-4K load).
    ///
    /// Shared by both admit paths (`registry_ensure` passes the just-resolved
    /// identity as `protect`; `registry_ensure_color` passes `None`).
    unsafe fn evict_registry_to_cap(
        &mut self,
        ctx: &DeviceContext,
        counters: &EngineCounters,
        protect: Option<&TargetIdentity>,
    ) {
        let mut non_pinned = self.note_registry_reach();
        while non_pinned > REGISTRY_CAP {
            // Least-recently-used, for real. `registry_order` is insertion
            // order — nothing promotes an entry when a draw reuses it — so
            // popping its front evicted the oldest-*created* resident. A
            // compositor backdrop is created early and lives for the whole
            // session, which made it the permanent front and the first victim of
            // every burst, however hard the current frame was reading it.
            //
            // Selecting the minimum `last_touch_ms` protects an in-use resident
            // without giving up the hard bound: skipping recent entries instead
            // would let a burst that touches more than `REGISTRY_CAP` targets
            // inside `IDLE_TARGET_AGE_MS` evict nothing at all and grow the
            // registry without limit. Here the cap always finds a victim while
            // any evictable entry exists, and the victim is never more recently
            // used than an alternative.
            //
            // Iterating `registry_order` rather than the map keeps the choice
            // deterministic: `min_by_key` returns the first minimum, so ties
            // fall to the oldest-created entry, which is what this walk did for
            // every entry before. O(n) per eviction rather than O(1) amortised,
            // and evictions are rare next to binds — the cost this avoids
            // paying is a promotion on every sampled bind of every draw.
            let victim = self.cap_eviction_victim(protect);
            // Everything left is pinned or protected. Pinned residents are
            // bounded separately by the rail that armed them, so the registry
            // soft-exceeds the cap rather than dropping content whose only copy
            // is on the GPU — the trade this walk has always made.
            let Some(victim) = victim else {
                break;
            };
            self.retire_resident(ctx, &victim, ResidentReclaim::CapEvicted, counters);
            self.registry_cap_evictions += 1;
            non_pinned = non_pinned.saturating_sub(1);
        }
    }

    /// Refresh a resident's idle-drain timestamp to at least `now_ms`. Used by
    /// host-window direct present before the export attempt so offscreen
    /// compositor peers needed for route-B tile compositing do not age out while
    /// the displayed member remains active.
    pub(crate) fn registry_touch_at(&mut self, identity: &TargetIdentity, now_ms: u64) {
        if now_ms > self.idle_clock_ms {
            self.idle_clock_ms = now_ms;
        }
        let touch = self.idle_clock_ms;
        self.use_clock += 1;
        let seq = self.use_clock;
        if let Some(slot) = self.registry.get_mut(identity) {
            slot.last_touch_ms = touch;
            slot.use_seq = seq;
        }
    }

    /// Remember that `identity`'s resident was reclaimed, and by which path.
    ///
    /// Called from every site that removes a live registry entry, so a later
    /// draw sampling it can distinguish "taken from under you" from "never
    /// existed". Bounded FIFO; the oldest record is dropped rather than letting
    /// a diagnostic grow without limit.
    pub(crate) fn note_resident_reclaimed(
        &mut self,
        identity: &TargetIdentity,
        why: ResidentReclaim,
    ) {
        if self.reclaimed_recent.len() >= RECLAIM_HISTORY {
            self.reclaimed_recent.pop_front();
        }
        self.reclaimed_recent.push_back((identity.clone(), why));
    }

    /// The most recent thing this device did with `identity`'s resident, if it
    /// is still inside the history window. `None` means no record — which covers
    /// both "never held one" and "reclaimed longer ago than the window reaches",
    /// two cases this deliberately does not guess between.
    pub(crate) fn prior_reclaim(&self, identity: &TargetIdentity) -> Option<ResidentReclaim> {
        self.reclaimed_recent
            .iter()
            .rev()
            .find(|(k, _)| k == identity)
            .map(|(_, why)| *why)
    }

    /// The resident the capacity walk should evict next, or `None` when every
    /// entry left is pinned or protected.
    ///
    /// Least-recently-used, chosen by `use_seq` — **not** by `last_touch_ms`,
    /// which is the other reclaim path's clock. The two are deliberately
    /// different: `last_touch_ms` is wall clock and answers "has this gone
    /// untouched for `IDLE_TARGET_AGE_MS`", which is what the idle drain needs;
    /// `use_seq` is a monotonic use counter and answers "which of these is
    /// least recently used", which is the only question with an answer when
    /// every entry was touched inside the same millisecond. A capacity walk on
    /// the wall clock cannot order a burst that arrives faster than the clock
    /// ticks, and a burst is exactly when it runs.
    ///
    /// Split out from [`Self::evict_registry_to_cap`] because that function
    /// needs a live `DeviceContext` to dispose what it evicts, and the choice —
    /// the part with the policy in it — is worth testing without a GPU.
    ///
    /// Iterates `registry_order` rather than the map so the result is
    /// deterministic: `min_by_key` returns the first minimum, so equal stamps
    /// fall to the oldest-created entry.
    fn cap_eviction_victim(&self, protect: Option<&TargetIdentity>) -> Option<TargetIdentity> {
        self.registry_order
            .iter()
            .filter_map(|k| self.registry.get(k).map(|slot| (k, slot)))
            .filter(|(k, slot)| slot.pin_count == 0 && protect != Some(k))
            .min_by_key(|(_, slot)| slot.use_seq)
            .map(|(k, _)| k.clone())
    }

    /// Record that a draw is reading this resident as a **sampled source**, so
    /// both reclaim paths count it as in use.
    ///
    /// See [`resident_resample_band`] for why this also bands how long the
    /// resident had been sitting untouched before the read.
    ///
    /// Reading a resident was not a use. `last_touch_ms` was refreshed by
    /// `registry_ensure` (a draw rendering *into* the target), by the present
    /// touch, and by nothing else — while the sampled-source resolve in
    /// `execute_draw_inner` goes through `registry_get`, which takes `&self` and
    /// therefore cannot mark anything. A resident that every frame samples but
    /// no frame draws into consequently aged as if it were abandoned.
    ///
    /// That is the shape of a compositor backdrop: the desktop behind a
    /// translucent panel is rendered once and then read by every vibrancy draw
    /// over it. After `IDLE_TARGET_AGE_MS` the idle drain took it, and the drain
    /// is a terminal destroy rather than a recycle, so the pixels were gone. The
    /// next draw to sample it refuses with
    /// `vk_draw_exec_sampled_resident_missing`, and because the exec loop
    /// abandons the remaining records of a packet once a record cannot encode,
    /// one missing backdrop drops a whole packet of draws.
    ///
    /// Nothing recreates a resident except a draw rendering into that identity,
    /// so a backdrop the guest considers still valid is never rebuilt: the
    /// refusal repeats for the life of the boot. That is why this class survives
    /// closing the application that caused the pressure and why only a reboot
    /// clears it.
    ///
    /// The stamp this writes is the one both reclaim paths read — the idle drain
    /// compares it against `IDLE_TARGET_AGE_MS`, and the capacity walk evicts
    /// the smallest — so recording a read here is what protects a resident from
    /// each of them.
    pub(crate) fn registry_note_sampled_use(&mut self, identity: &TargetIdentity) {
        let touch = self.idle_clock_ms;
        self.use_clock += 1;
        let seq = self.use_clock;
        if let Some(slot) = self.registry.get_mut(identity) {
            let idle_ms = touch.saturating_sub(slot.last_touch_ms);
            slot.last_touch_ms = touch;
            slot.use_seq = seq;
            crate::runtime::drain::note_store_route(resident_resample_band(idle_ms));
            // The bands give the distribution; this gives the margin. They
            // answer different questions, and the bands alone could not say
            // whether their one sample above the half mark sat at 1.0 s or at
            // 1.9 s against a 2 s cutoff — which is the difference between
            // comfortable and one slow frame from a permanent loss.
            self.resident_resample_peak_ms = self.resident_resample_peak_ms.max(idle_ms);
        }
    }
}

#[cfg(test)]
mod pin_count_tests {
    use super::*;

    fn dummy_slot(content_ready: bool) -> ResidentTargetSlot {
        ResidentTargetSlot {
            image: vk::Image::null(),
            memory: vk::DeviceMemory::null(),
            view: vk::ImageView::null(),
            framebuffer: vk::Framebuffer::null(),
            render_pass: vk::RenderPass::null(),
            width: 16,
            height: 16,
            generation: 1,
            content_ready,
            content_epoch: None,
            layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            color_format: translate::pixel::SCANOUT_FORMAT,
            pin_count: 0,
            last_touch_ms: 0,
            use_seq: 0,
        }
    }

    fn pinned_identity() -> TargetIdentity {
        TargetIdentity::Surface {
            id: 1,
            width: 16,
            height: 16,
            generation: 0,
        }
    }

    /// The window presenter blits a resident with no format conversion and no
    /// source scaling, so every one of these four conditions is load-bearing.
    ///
    /// It matters that this is ONE function: the device's publish path asks it
    /// a frame ahead of the presenter to decide whether to read the frame back
    /// into host memory. A looser predicate there elides the readback for a
    /// frame the presenter then refuses, and the window goes blank with no CPU
    /// pixels behind it — a disagreement neither call site can see on its own.
    #[test]
    fn only_a_ready_bgra_resident_at_the_exact_geometry_is_presentable() {
        let ready = dummy_slot(true);
        assert!(slot_presentable(&ready, 16, 16));

        assert!(
            !slot_presentable(&dummy_slot(false), 16, 16),
            "content that has not landed would present the previous frame"
        );

        let mut rgba = dummy_slot(true);
        rgba.color_format = translate::pixel::RESIDENT_RGBA_FORMAT;
        assert!(
            !slot_presentable(&rgba, 16, 16),
            "the blit does no channel swap; RGBA would present with red and blue exchanged"
        );

        assert!(
            !slot_presentable(&ready, 32, 16),
            "a wider present than the resident holds would blit a stretched frame"
        );
        assert!(
            !slot_presentable(&ready, 16, 32),
            "a taller present than the resident holds would blit a stretched frame"
        );
    }

    /// A draw into this identity invalidates any stamp on it. The image's
    /// pixels just changed, and until something publishes them as the mapping's
    /// content the type-11 LOAD gate must not treat them as current — otherwise
    /// an intermediate record's output is loaded as though it were the guest's
    /// prior frame.
    ///
    /// Placed on `registry_mark_ready` rather than on the individual writers on
    /// purpose: every path that leaves a resident holding new pixels goes
    /// through here or `registry_mark_ready_at`, so the invalidation is total
    /// rather than a list of the writers somebody remembered.
    #[test]
    fn a_draw_into_a_resident_clears_its_content_stamp() {
        let mut pools = ResourcePools::new();
        let id = pinned_identity();
        pools.registry.insert(id.clone(), dummy_slot(true));

        assert!(pools.registry_stamp_content_epoch(&id, 9));
        assert_eq!(pools.registry.get(&id).unwrap().content_epoch, Some(9));

        pools.registry_mark_ready(&id);
        assert_eq!(
            pools.registry.get(&id).unwrap().content_epoch,
            None,
            "a draw stored new pixels; the old stamp cannot vouch for them"
        );

        assert!(pools.registry_stamp_content_epoch(&id, 10));
        pools.registry_mark_ready_at(&id, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        assert_eq!(
            pools.registry.get(&id).unwrap().content_epoch,
            None,
            "the MRT-secondary ready arm must invalidate identically"
        );
    }

    /// Stamping an image no draw has stored into would vouch for undefined
    /// memory, and an absent identity has no image at all. Both refuse, and the
    /// caller reads the `false` as "the elision is off for this surface".
    #[test]
    fn a_stamp_refuses_an_image_no_draw_has_written() {
        let mut pools = ResourcePools::new();
        let id = pinned_identity();

        assert!(
            !pools.registry_stamp_content_epoch(&id, 1),
            "an absent identity cannot be stamped"
        );

        pools.registry.insert(id.clone(), dummy_slot(false));
        assert!(
            !pools.registry_stamp_content_epoch(&id, 1),
            "a resident that is not content_ready holds undefined pixels"
        );
        assert_eq!(pools.registry.get(&id).unwrap().content_epoch, None);
    }

    /// Two deferred windows on one surface pin the SAME identity; the first
    /// window's flush-unpin must NOT expose the image to the LRU sweep while the
    /// second is still armed. This is the eviction window a boolean pin had.
    #[test]
    fn shared_identity_pin_is_counted_not_boolean() {
        let mut pools = ResourcePools::new();
        let id = pinned_identity();
        pools.registry.insert(id.clone(), dummy_slot(true));

        assert!(pools.pin_resident_target(&id, true), "window A pin");
        assert!(pools.pin_resident_target(&id, true), "window B pin");
        assert_eq!(pools.registry.get(&id).unwrap().pin_count, 2);

        // Window A flushes: one unpin — the slot must stay sweep-protected.
        assert!(pools.pin_resident_target(&id, false));
        assert_eq!(
            pools.registry.get(&id).unwrap().pin_count,
            1,
            "the second window is still armed: slot must remain pinned"
        );

        // Window B flushes: fully released.
        assert!(pools.pin_resident_target(&id, false));
        assert_eq!(pools.registry.get(&id).unwrap().pin_count, 0);
    }

    /// A spurious unpin (double-release) saturates at zero instead of
    /// underflowing into a forever-pin.
    #[test]
    fn unpin_saturates_at_zero() {
        let mut pools = ResourcePools::new();
        let id = pinned_identity();
        pools.registry.insert(id.clone(), dummy_slot(true));
        assert!(pools.pin_resident_target(&id, false));
        assert_eq!(pools.registry.get(&id).unwrap().pin_count, 0);
        assert!(pools.pin_resident_target(&id, true));
        assert_eq!(pools.registry.get(&id).unwrap().pin_count, 1);
    }

    /// Pin still refuses a not-ready slot (callers fall back to the sync
    /// Store) and an absent identity.
    #[test]
    fn pin_refuses_not_ready_and_absent() {
        let mut pools = ResourcePools::new();
        let id = pinned_identity();
        assert!(!pools.pin_resident_target(&id, true), "absent identity");
        pools.registry.insert(id.clone(), dummy_slot(false));
        assert!(!pools.pin_resident_target(&id, true), "not-ready slot");
        assert_eq!(pools.registry.get(&id).unwrap().pin_count, 0);
    }

    fn surf(id: u32) -> TargetIdentity {
        TargetIdentity::Surface {
            id,
            width: 16,
            height: 16,
            generation: 1,
        }
    }

    /// A resident shaped like the one an arm builds, for the registration tests.
    ///
    /// The two arms differ in exactly these two handles, so they are the
    /// parameters: `registry_ensure` passes a real framebuffer and the pass it
    /// was built against, `registry_ensure_color` passes neither.
    fn new_resident(framebuffer: vk::Framebuffer, render_pass: vk::RenderPass) -> NewResident {
        NewResident {
            image: vk::Image::null(),
            memory: vk::DeviceMemory::null(),
            view: vk::ImageView::null(),
            framebuffer,
            render_pass,
            width: 16,
            height: 16,
            generation: 1,
            color_format: translate::pixel::SCANOUT_FORMAT,
        }
    }

    /// A framebuffer handle that is merely non-null. Nothing dereferences it —
    /// every test here asks only whether the slot has one.
    fn some_framebuffer() -> vk::Framebuffer {
        vk::Framebuffer::from_raw(1)
    }

    /// Admit a resident with an explicit last-touch stamp and pin count.
    ///
    /// Registers through the product path rather than writing the map and the
    /// order itself, so this helper cannot be the copy that keeps them in step
    /// by accident while the product one stops.
    fn admit(pools: &mut ResourcePools, id: TargetIdentity, last_touch_ms: u64, pin: u32) {
        pools.register_resident(
            &id,
            new_resident(some_framebuffer(), vk::RenderPass::null()),
        );
        let slot = pools.registry.get_mut(&id).expect("just registered");
        slot.content_ready = true;
        slot.last_touch_ms = last_touch_ms;
        // Through the product path, not `slot.pin_count = pin`: pinning is what
        // takes a resident out of the maintained non-pinned totals, and a helper
        // that wrote the field itself would be the one mutation site the totals
        // cannot see — which is the desync
        // `the_maintained_non_pinned_totals_track_the_walk` exists to catch.
        for _ in 0..pin {
            assert!(pools.pin_resident_target(&id, true), "content is ready");
        }
    }

    /// [`admit`] at an explicit geometry, so a test can build populations the
    /// slot count cannot tell apart. Geometry is fixed at registration because
    /// nothing in the product mutates a live slot's — a geometry change goes
    /// through unregister + register — and the byte total relies on that.
    fn admit_sized(
        pools: &mut ResourcePools,
        id: TargetIdentity,
        last_touch_ms: u64,
        pin: u32,
        (width, height): (u32, u32),
    ) {
        let mut resident = new_resident(some_framebuffer(), vk::RenderPass::null());
        resident.width = width;
        resident.height = height;
        pools.register_resident(&id, resident);
        let slot = pools.registry.get_mut(&id).expect("just registered");
        slot.content_ready = true;
        slot.last_touch_ms = last_touch_ms;
        for _ in 0..pin {
            assert!(pools.pin_resident_target(&id, true), "content is ready");
        }
    }

    /// The MRT-secondary arm builds no per-slot framebuffer, so the residents it
    /// creates owe the graveyard nothing — and the deferred-handle ring is
    /// bounded, so a destroy path that enqueued their null handle would be
    /// spending a slot every real destroy has to wait behind.
    ///
    /// `vkDestroyFramebuffer` accepts `VK_NULL_HANDLE` and does nothing with it.
    /// That is why the two paths which asked this question wrong produced no
    /// crash, no validation error and no log line, and why the answer is worth
    /// a test rather than an assertion at each site.
    #[test]
    fn a_resident_built_without_a_framebuffer_owes_the_graveyard_none() {
        let mut pools = ResourcePools::new();
        pools.register_resident(
            &surf(1),
            new_resident(vk::Framebuffer::null(), vk::RenderPass::null()),
        );
        pools.register_resident(
            &surf(2),
            new_resident(some_framebuffer(), vk::RenderPass::null()),
        );

        assert_eq!(
            pools.registry.get(&surf(1)).unwrap().owed_framebuffer(),
            None,
            "an MRT-secondary resident has no framebuffer to destroy"
        );
        assert_eq!(
            pools.registry.get(&surf(2)).unwrap().owed_framebuffer(),
            Some(some_framebuffer()),
            "a single-RT resident owes the one it was built with"
        );
    }

    /// A resident is born with nothing drawn into it, nothing vouching for its
    /// pixels, no layout transition behind it and no window holding it.
    ///
    /// Each of these four is read by a different rail — `registry_mark_ready`,
    /// the type-11 LOAD gate's epoch check, the barrier tracker and the idle
    /// drain — so an arm that registered a slot with any of them set differently
    /// would be answering a question the other rails believe they already asked.
    #[test]
    fn a_registered_resident_is_born_undrawn_unvouched_untransitioned_and_unpinned() {
        let mut pools = ResourcePools::new();
        pools.register_resident(
            &surf(1),
            new_resident(some_framebuffer(), vk::RenderPass::null()),
        );

        let slot = pools.registry.get(&surf(1)).expect("registered");
        assert!(!slot.content_ready, "nothing has drawn into it yet");
        assert_eq!(
            slot.content_epoch, None,
            "nothing has vouched for its pixels"
        );
        assert_eq!(
            slot.layout,
            vk::ImageLayout::UNDEFINED,
            "nothing has transitioned it yet"
        );
        assert_eq!(slot.pin_count, 0, "no deferred window holds it yet");
    }

    /// Registration writes the map and the order together, and stamps a use
    /// sequence that strictly advances.
    ///
    /// `registry` and `registry_order` are one structure split for lookup and
    /// for order: an entry in the map alone is a resident no sweep can choose,
    /// and one in the order alone is a victim that frees nothing. The sequence
    /// is what `cap_eviction_victim` breaks ties on, so two registrations inside
    /// one clock tick must still be ordered.
    #[test]
    fn registration_writes_both_halves_and_advances_the_use_sequence() {
        let mut pools = ResourcePools::new();
        pools.register_resident(
            &surf(1),
            new_resident(some_framebuffer(), vk::RenderPass::null()),
        );
        pools.register_resident(
            &surf(2),
            new_resident(some_framebuffer(), vk::RenderPass::null()),
        );

        assert_eq!(
            pools.registry_order.iter().cloned().collect::<Vec<_>>(),
            vec![surf(1), surf(2)],
            "the order holds both, in registration order"
        );
        assert!(
            pools.registry.contains_key(&surf(1)) && pools.registry.contains_key(&surf(2)),
            "the map holds both"
        );
        assert!(
            pools.registry.get(&surf(1)).unwrap().use_seq
                < pools.registry.get(&surf(2)).unwrap().use_seq,
            "the second registration is later even inside one clock tick"
        );
    }

    /// A non-pinned resident untouched for `IDLE_TARGET_AGE_MS` is selected; a
    /// freshly-touched peer and a pinned peer are not. The wall clock advances to
    /// the passed `now_ms` (not a per-call increment), so a static guest that
    /// keeps ticking the poll heartbeat still reclaims stale VRAM.
    #[test]
    fn plan_idle_drain_selects_only_aged_non_pinned() {
        let mut pools = ResourcePools::new();
        admit(&mut pools, surf(1), 10, 0); // aged, non-pinned  -> victim
        admit(&mut pools, surf(2), 10, 1); // aged but PINNED   -> kept
                                           // now = 10 + AGE + 1 so slot 1's cutoff is crossed; a fresh slot is not.
        let now = 10 + IDLE_TARGET_AGE_MS + 1;
        admit(&mut pools, surf(3), now, 0); // fresh            -> kept
        let victims = pools.plan_idle_drain(now, None).expect("pass due");
        assert_eq!(victims, vec![surf(1)], "only the aged non-pinned resident");
        assert_eq!(pools.idle_clock_ms, now, "clock advanced to wall time");
    }

    /// The resample bands are fractions of the drain cutoff, so retuning the
    /// cutoff moves them with it and no reading is ever quoted against a bound
    /// it did not come from.
    ///
    /// The boundary that matters is the last one. `past_cutoff` is exactly the
    /// case where a resident was read after sitting longer than the age at which
    /// the drain would have destroyed it, so it survived only because the drain
    /// is throttled to `IDLE_TARGET_DRAIN_MAX_PER_CALL` per pass and had not
    /// reached it. `IDLE_TARGET_AGE_MS` itself must therefore land in that band
    /// and not under it: the drain's own comparison is `last_touch_ms <= cutoff`,
    /// so a resident exactly at the cutoff is already a victim.
    #[test]
    fn the_resident_resample_bands_are_fractions_of_the_drain_cutoff() {
        let c = IDLE_TARGET_AGE_MS;
        for (idle, expected) in [
            (0, "resident_resample_lt_eighth_cutoff"),
            (c / 8 - 1, "resident_resample_lt_eighth_cutoff"),
            (c / 8, "resident_resample_lt_quarter_cutoff"),
            (c / 4, "resident_resample_lt_half_cutoff"),
            (c / 2, "resident_resample_under_cutoff"),
            (c - 1, "resident_resample_under_cutoff"),
            (c, "resident_resample_past_cutoff"),
            (u64::MAX, "resident_resample_past_cutoff"),
        ] {
            assert_eq!(
                resident_resample_band(idle),
                expected,
                "idle_ms={idle} against cutoff={c}"
            );
        }
    }

    /// "This device destroyed a resident for this identity" and "this device
    /// never held one" must be distinguishable, and a re-created resident must
    /// report neither.
    ///
    /// These are the three states behind `resident_absent_after_reclaim`, whose
    /// body is `if present { None } else { prior_reclaim(..) }`. The composition
    /// is what matters: `prior_reclaim` alone keeps answering for the life of
    /// `RECLAIM_HISTORY`, so consulting it without the presence check would make
    /// a resident that was reclaimed and then re-created keep reporting itself
    /// as destroyed — and the caller uses that answer to decide whether falling
    /// through to the guest's pages is sound.
    ///
    /// The facade itself needs the engine lock and so cannot be unit-tested; the
    /// logic it composes is here.
    #[test]
    fn a_recreated_resident_no_longer_reports_the_reclaim_that_took_it() {
        let mut pools = ResourcePools::new();
        admit(&mut pools, surf(1), 0, 0);

        // Never held: no record, and nothing to mistake for one.
        assert_eq!(pools.prior_reclaim(&surf(2)), None);
        assert!(!pools.registry.contains_key(&surf(2)));

        // Held: present, so the facade short-circuits before prior_reclaim.
        assert!(pools.registry.contains_key(&surf(1)));

        // Destroyed: absent, and the cause survives.
        pools.unregister_resident(&surf(1), ResidentReclaim::IdleDrained);
        assert!(!pools.registry.contains_key(&surf(1)));
        assert_eq!(
            pools.prior_reclaim(&surf(1)),
            Some(ResidentReclaim::IdleDrained)
        );

        // Re-created: the record is still in history, so only the presence
        // check keeps this from reading as destroyed.
        admit(&mut pools, surf(1), 0, 0);
        assert!(
            pools.registry.contains_key(&surf(1)),
            "the presence check is the only thing separating this from the case above"
        );
        assert_eq!(
            pools.prior_reclaim(&surf(1)),
            Some(ResidentReclaim::IdleDrained),
            "history is deliberately not cleared on re-admit, which is why the \
             presence check cannot be dropped"
        );
    }

    /// The resample peak is the worst gap the boot ever saw, not the last one.
    ///
    /// A high-water, so a large gap early is not erased by a run of small ones
    /// after it — which is the whole reason it is not a windowed reading. The
    /// margin question is "how close did this boot ever come to
    /// `IDLE_TARGET_AGE_MS`", and a gap that peaks between two census samples is
    /// exactly what an instantaneous value misses.
    ///
    /// Fails without the fix: nothing records the gap at all.
    #[test]
    fn the_resample_peak_holds_the_worst_gap_not_the_latest() {
        let mut pools = ResourcePools::new();
        admit(&mut pools, surf(1), 0, 0);
        assert_eq!(pools.resident_resample_peak_ms(), 0);

        // A 900 ms gap, then a 100 ms one. The peak must keep the 900.
        pools.idle_clock_ms = 900;
        pools.registry_note_sampled_use(&surf(1));
        assert_eq!(pools.resident_resample_peak_ms(), 900);
        pools.idle_clock_ms = 1_000;
        pools.registry_note_sampled_use(&surf(1));
        assert_eq!(
            pools.resident_resample_peak_ms(),
            900,
            "a smaller later gap must not lower the high-water"
        );

        // And a larger one does raise it.
        pools.idle_clock_ms = 1_000 + IDLE_TARGET_AGE_MS;
        pools.registry_note_sampled_use(&surf(1));
        assert_eq!(pools.resident_resample_peak_ms(), IDLE_TARGET_AGE_MS);

        // A read of an identity the registry does not hold records nothing —
        // there is no gap to measure, and counting it as one would report a
        // margin that no resident ever spent.
        pools.idle_clock_ms = u64::MAX;
        pools.registry_note_sampled_use(&surf(99));
        assert_eq!(pools.resident_resample_peak_ms(), IDLE_TARGET_AGE_MS);
    }

    /// A resident that a draw only ever *samples* survives the idle drain.
    ///
    /// This is the compositor-backdrop case: the desktop behind a translucent
    /// panel is rendered once and then read by every vibrancy draw over it.
    /// Before [`ResourcePools::registry_note_sampled_use`] existed, reading a
    /// resident refreshed nothing — `registry_ensure` (render *into*) and the
    /// present touch were the only writers of `last_touch_ms` — so the backdrop
    /// aged exactly as if it had been abandoned and the drain terminally
    /// destroyed it. Every later draw sampling it then refused with
    /// `vk_draw_exec_sampled_resident_missing`, and nothing recreates a resident
    /// except a draw rendering into that identity, so the refusal held for the
    /// rest of the boot.
    ///
    /// Fails without the fix: with the body of `registry_note_sampled_use`
    /// removed, `surf(1)` is selected and the assertion reports it as a victim.
    #[test]
    fn a_sampled_only_resident_is_not_aged_out() {
        let mut pools = ResourcePools::new();
        // Aged past the cutoff by every measure the drain has, and never drawn
        // into again — only read.
        admit(&mut pools, surf(1), 10, 0);
        admit(&mut pools, surf(2), 10, 0);
        let now = 10 + IDLE_TARGET_AGE_MS + 1;
        // The drain's clock has to be current before a use can be recorded
        // against it; the real caller advances it from the poll heartbeat.
        pools.plan_idle_drain(now, None);
        pools.registry_note_sampled_use(&surf(1));
        // A second pass, far enough after the first to clear the throttle.
        let later = now + IDLE_DRAIN_INTERVAL_MS + 1;
        let victims = pools.plan_idle_drain(later, None).expect("pass due");
        assert!(
            !victims.contains(&surf(1)),
            "a resident being sampled is in use and must not be destroyed"
        );
        assert!(
            victims.contains(&surf(2)),
            "its untouched peer is still reclaimed — the fix must not disable the drain"
        );
    }

    /// The capacity walk evicts the least-recently-*used* resident, so a
    /// backdrop a draw is reading is not taken merely for being old.
    ///
    /// `registry_order` is insertion order — nothing promotes an entry when a
    /// draw reuses it — so popping its front evicted the oldest-*created*
    /// resident, which for a session-long backdrop is permanently the front.
    /// Choosing by `last_touch_ms` is what separates "created first" from "not
    /// being used", and it keeps the cap hard: skipping recent entries instead
    /// would let a burst touching more than `REGISTRY_CAP` targets inside the
    /// age window evict nothing at all.
    #[test]
    fn the_cap_walk_evicts_the_least_recently_used_not_the_oldest_created() {
        let mut pools = ResourcePools::new();
        // surf(1) is created first and is therefore the front of insertion
        // order — the old victim — but it is the one being read.
        admit(&mut pools, surf(1), 0, 0);
        admit(&mut pools, surf(2), 0, 0);
        admit(&mut pools, surf(3), 0, 0);
        pools.idle_clock_ms = 5_000;
        pools.registry_note_sampled_use(&surf(1));
        assert_eq!(
            pools.cap_eviction_victim(None),
            Some(surf(2)),
            "the least-recently-used resident is the victim, not the first created"
        );
        // Protection still applies, and the walk still finds someone else.
        assert_eq!(
            pools.cap_eviction_victim(Some(&surf(2))),
            Some(surf(3)),
            "a protected identity is passed over for the next-oldest use"
        );
    }

    /// Uses inside one poll tick are still ordered against each other.
    ///
    /// `last_touch_ms` comes from the ~244 Hz poll heartbeat, so every use
    /// between two ticks carries the same millisecond. Choosing the victim on
    /// that alone leaves a tie, and the tie falls to the oldest-created entry —
    /// which is precisely the session-long backdrop this walk must stop taking.
    /// `use_seq` gives the total order, so a resident read this tick outranks
    /// one last read several ticks ago even though the clock never moved
    /// between them.
    #[test]
    fn uses_within_one_clock_tick_are_still_ordered() {
        let mut pools = ResourcePools::new();
        // Same wall-clock stamp on every slot: one poll tick.
        admit(&mut pools, surf(1), 500, 0);
        admit(&mut pools, surf(2), 500, 0);
        pools.idle_clock_ms = 500;
        // Read the oldest-created one last. The clock cannot express that.
        pools.registry_note_sampled_use(&surf(2));
        pools.registry_note_sampled_use(&surf(1));
        assert_eq!(
            pools.registry.get(&surf(1)).unwrap().last_touch_ms,
            pools.registry.get(&surf(2)).unwrap().last_touch_ms,
            "precondition: the wall clock cannot separate these two uses"
        );
        assert_eq!(
            pools.cap_eviction_victim(None),
            Some(surf(2)),
            "the earlier use in the tick is the victim, not the earlier creation"
        );
    }

    /// A pinned resident is never the victim, and a registry with nothing else
    /// left reports no victim rather than dropping content whose only copy is on
    /// the GPU — the soft-exceed the walk has always traded for.
    #[test]
    fn the_cap_walk_never_evicts_a_pinned_resident() {
        let mut pools = ResourcePools::new();
        admit(&mut pools, surf(1), 0, 1);
        admit(&mut pools, surf(2), 0, 2);
        assert_eq!(
            pools.cap_eviction_victim(None),
            None,
            "every entry is pinned, so the cap soft-exceeds rather than evicting"
        );
        admit(&mut pools, surf(3), 10, 0);
        assert_eq!(
            pools.cap_eviction_victim(None),
            Some(surf(3)),
            "the one unpinned resident is the only candidate"
        );
    }

    /// Every removal of a live registry entry leaves a record naming the path.
    ///
    /// This is the invariant `note_resident_reclaimed` claims ("called from
    /// every site that removes a live registry entry") and that the MRT
    /// secondary recreate arm broke while it was a copy: it removed the entry
    /// and recorded nothing, so `prior_reclaim` answered `None` — which
    /// `exec` reports as "never existed" for a resident this device had just
    /// taken. Routing all three sites through `unregister_resident` is what
    /// makes the record unconditional.
    #[test]
    fn unregistering_a_resident_always_names_why_and_leaves_the_order_clean() {
        let mut pools = ResourcePools::new();
        admit(&mut pools, surf(1), 0, 0);
        admit(&mut pools, surf(2), 0, 0);

        assert!(pools
            .unregister_resident(&surf(1), ResidentReclaim::Recreated)
            .is_some());
        assert_eq!(
            pools.prior_reclaim(&surf(1)),
            Some(ResidentReclaim::Recreated),
            "a removed resident must say which path took it"
        );
        assert!(!pools.registry.contains_key(&surf(1)));
        assert!(
            !pools.registry_order.contains(&surf(1)),
            "the order list must not keep a key the map no longer holds"
        );
        assert!(
            pools.registry_order.contains(&surf(2)),
            "an untouched resident keeps its place"
        );

        // An identity that held nothing is not a removal, so it gets no record.
        // Writing one would make `prior_reclaim` claim this device took a
        // resident that never existed — the exact confusion it exists to avoid.
        assert!(pools
            .unregister_resident(&surf(3), ResidentReclaim::CapEvicted)
            .is_none());
        assert_eq!(pools.prior_reclaim(&surf(3)), None);
    }

    /// The reclaim history answers which path took a resident, and says "no
    /// record" rather than guessing when it cannot.
    ///
    /// This is what lets `vk_draw_exec_sampled_resident_missing` distinguish a
    /// resident reclaimed out from under an active reader from one the guest
    /// never rendered into. Both present as an absent registry entry, and the
    /// two have different repairs.
    #[test]
    fn the_reclaim_history_names_the_path_and_is_bounded() {
        let mut pools = ResourcePools::new();
        pools.note_resident_reclaimed(&surf(1), ResidentReclaim::IdleDrained);
        pools.note_resident_reclaimed(&surf(2), ResidentReclaim::CapEvicted);
        assert_eq!(
            pools.prior_reclaim(&surf(1)),
            Some(ResidentReclaim::IdleDrained)
        );
        assert_eq!(
            pools.prior_reclaim(&surf(2)),
            Some(ResidentReclaim::CapEvicted)
        );
        assert_eq!(
            pools.prior_reclaim(&surf(3)),
            None,
            "an identity never reclaimed has no record, and is not guessed at"
        );
        // The most recent verdict wins: an identity recreated after being
        // evicted is not still reported as evicted.
        pools.note_resident_reclaimed(&surf(1), ResidentReclaim::Recreated);
        assert_eq!(
            pools.prior_reclaim(&surf(1)),
            Some(ResidentReclaim::Recreated)
        );
        // Bounded: the oldest record falls out rather than the history growing
        // without limit, and falling out reads as no record.
        for i in 0..RECLAIM_HISTORY as u32 {
            pools.note_resident_reclaimed(&surf(1000 + i), ResidentReclaim::CapEvicted);
        }
        assert!(pools.reclaimed_recent.len() <= RECLAIM_HISTORY);
        assert_eq!(
            pools.prior_reclaim(&surf(2)),
            None,
            "aged out of the window"
        );
    }

    /// The reclaim pass is throttled to `IDLE_DRAIN_INTERVAL_MS`: a second call
    /// inside the interval selects nothing even though a resident is aged, so the
    /// ~244 Hz poll cadence cannot empty the registry at once. The clock still
    /// advances (admits stay fresh).
    #[test]
    fn plan_idle_drain_throttles_between_passes() {
        let mut pools = ResourcePools::new();
        admit(&mut pools, surf(1), 0, 0);
        let t0 = IDLE_TARGET_AGE_MS + 1;
        assert_eq!(pools.plan_idle_drain(t0, None), Some(vec![surf(1)]));
        // Simulate the dispose the real caller (advance_registry_touch_and_drain)
        // performs for each selected victim.
        pools.registry.remove(&surf(1));
        pools.registry_order.retain(|k| k != &surf(1));
        admit(&mut pools, surf(2), 0, 0);
        // A call one ms later is inside the interval → no pass (None), despite
        // surf(2) being aged.
        assert_eq!(
            pools.plan_idle_drain(t0 + 1, None),
            None,
            "throttled: no pass"
        );
        assert_eq!(
            pools.idle_clock_ms,
            t0 + 1,
            "clock still advances when throttled"
        );
        // Past the interval → the next aged resident is selected.
        assert_eq!(
            pools.plan_idle_drain(t0 + IDLE_DRAIN_INTERVAL_MS, None),
            Some(vec![surf(2)])
        );
    }

    /// Each pass selects at most `IDLE_TARGET_DRAIN_MAX_PER_CALL` so a huge stale
    /// set drains gradually (no dispose storm that would be a P3 hitch itself).
    #[test]
    fn plan_idle_drain_bounds_batch_per_pass() {
        let mut pools = ResourcePools::new();
        for i in 0..(IDLE_TARGET_DRAIN_MAX_PER_CALL as u32 + 5) {
            admit(&mut pools, surf(100 + i), 0, 0);
        }
        let victims = pools
            .plan_idle_drain(IDLE_TARGET_AGE_MS + 1, None)
            .expect("pass due");
        assert_eq!(victims.len(), IDLE_TARGET_DRAIN_MAX_PER_CALL);
    }

    /// A pass with no registry victim but live staging traffic is NOT settled.
    ///
    /// This is the case the victim count alone cannot see and the one that
    /// actually happens: a steady animation re-uses the same render targets, so
    /// nothing ages out and every pass reads as quiet, while the upload path runs
    /// flat out. Measured under testufo the trim fired about once a second
    /// throughout the load and cost 607 re-allocations of the 8 MiB full-frame
    /// staging bucket at 12.6 ms each.
    #[test]
    fn a_pass_with_no_victims_but_live_uploads_is_not_settled() {
        let mut pools = ResourcePools::new();
        // Quiet the gate first, so the assertion below is about uploads and not
        // about the counter still warming up.
        for _ in 0..SETTLED_PASSES_FOR_BUFFER_TRIM {
            pools.note_drain_settled(0);
        }
        assert!(
            pools.note_drain_settled(0),
            "no victims, no uploads → settled"
        );

        // One staging acquire between passes — no victim, still not settled.
        pools.staging_hits += 1;
        assert!(
            !pools.note_drain_settled(0),
            "uploads ran between passes; the buffer pools must not be trimmed"
        );
        // …and the gate stays shut while uploads keep flowing, however many
        // zero-victim passes go by.
        for _ in 0..(SETTLED_PASSES_FOR_BUFFER_TRIM * 3) {
            pools.staging_misses += 1;
            assert!(!pools.note_drain_settled(0), "still uploading");
        }
        // Uploads stop: the gate reopens after the usual consecutive passes.
        for _ in 0..(SETTLED_PASSES_FOR_BUFFER_TRIM - 1) {
            assert!(!pools.note_drain_settled(0), "counter restarted from zero");
        }
        assert!(pools.note_drain_settled(0), "settled once uploads stopped");
    }

    /// The HOST_VISIBLE buffer trim gate: only permitted after
    /// `SETTLED_PASSES_FOR_BUFFER_TRIM` consecutive zero-victim passes, and any
    /// pass that drains ≥1 victim (active churn) resets the counter — so a
    /// staging buffer cannot be freed and re-alloc'd mid-video.
    #[test]
    fn note_drain_settled_gates_buffer_trim_on_consecutive_idle() {
        let mut pools = ResourcePools::new();
        // Fewer than the threshold of quiet passes: no buffer trim yet.
        for _ in 0..(SETTLED_PASSES_FOR_BUFFER_TRIM - 1) {
            assert!(!pools.note_drain_settled(0), "not settled enough yet");
        }
        // The Nth consecutive zero-victim pass crosses the threshold.
        assert!(
            pools.note_drain_settled(0),
            "N consecutive settled passes → trim allowed"
        );
        // A subsequent quiet pass stays allowed.
        assert!(pools.note_drain_settled(0), "stays settled");
        // A pass that drains a victim (active churn) resets the counter…
        assert!(
            !pools.note_drain_settled(1),
            "any drained victim resets settled state"
        );
        // …and the gate stays closed until the run rebuilds.
        for _ in 0..(SETTLED_PASSES_FOR_BUFFER_TRIM - 1) {
            assert!(!pools.note_drain_settled(0), "counter restarted from zero");
        }
        assert!(pools.note_drain_settled(0), "settled again after rebuild");
    }

    /// The presented target passed as `display` is stamped to the current clock
    /// every call, so even though it is only resolved via `registry_get` (never
    /// re-drawn on a static page) it never ages out from under the display.
    #[test]
    fn plan_idle_drain_keeps_display_target_alive() {
        let mut pools = ResourcePools::new();
        admit(&mut pools, surf(1), 0, 0); // would be aged...
        let now = IDLE_TARGET_AGE_MS + 500;
        // ...but it is the presented target this frame.
        let victims = pools
            .plan_idle_drain(now, Some(&surf(1)))
            .expect("pass due");
        assert!(victims.is_empty(), "display target must not be reclaimed");
        assert_eq!(
            pools.registry.get(&surf(1)).unwrap().last_touch_ms,
            now,
            "display target stamped fresh"
        );
    }

    /// `registry_touch_at` refreshes a target against the idle-drain cutoff
    /// without going through the draw path, so a target that is registered but
    /// not being drawn survives a static desktop interval when a caller still
    /// needs it.
    #[test]
    fn registry_touch_at_defers_the_idle_drain_for_an_untouched_target() {
        let mut pools = ResourcePools::new();
        admit(&mut pools, surf(1), 0, 0); // displayed target
        admit(&mut pools, surf(4), 0, 0); // registered but undrawn, otherwise aged
        let now = IDLE_TARGET_AGE_MS + 500;

        pools.registry_touch_at(&surf(4), now);
        let victims = pools
            .plan_idle_drain(now, Some(&surf(1)))
            .expect("pass due");
        assert_eq!(
            victims,
            Vec::<TargetIdentity>::new(),
            "the display target and the touched target both survive"
        );
        assert_eq!(
            pools.registry.get(&surf(4)).unwrap().last_touch_ms,
            now,
            "the touched target is stamped at the touch time"
        );
    }

    /// The byte band is not the slot band scaled, and a boot needs both.
    ///
    /// `REGISTRY_CAP` bounds slots while saying the resource it protects is
    /// bytes. This drives the difference directly: two populations of the same
    /// size, one of 16x16 scratch and one of 4K attachments, are indistinguishable
    /// to the slot band and four orders of magnitude apart in VRAM. A cap that
    /// cannot see that gap is the reason this counter exists.
    #[test]
    fn the_registry_byte_band_separates_populations_the_slot_band_cannot() {
        const TEXEL: u64 = 4; // SCANOUT_FORMAT, the shape `new_resident` builds
        const SMALL: (u32, u32) = (16, 16);
        const UHD: (u32, u32) = (3840, 2160);
        let mut pools = ResourcePools::new();
        for i in 1..=3u32 {
            admit_sized(&mut pools, surf(i), 10, 0, SMALL);
        }
        pools.note_registry_reach();
        let (slots_small, _, bytes_small) = pools.registry_pressure_stats();
        assert_eq!(slots_small, 3);
        assert_eq!(bytes_small, 3 * 16 * 16 * TEXEL);

        // The same slot count at 4K geometry. Pinned peers stay out of both
        // bands, or the byte reading would count VRAM the cap never bounds.
        let mut big = ResourcePools::new();
        for i in 1..=3u32 {
            admit_sized(&mut big, surf(i), 10, 0, UHD);
        }
        admit_sized(&mut big, surf(9), 10, 1, UHD);
        big.note_registry_reach();
        let (slots_big, _, bytes_big) = big.registry_pressure_stats();
        assert_eq!(
            slots_big, slots_small,
            "the slot band cannot tell the two populations apart"
        );
        assert_eq!(
            bytes_big,
            3 * 3840 * 2160 * TEXEL,
            "the byte band can, and the pinned 4K peer is not in it"
        );

        // A high-water mark, like its sibling: the population falling does not
        // lower it, or a burst that drains between two census samples reads as
        // if it never happened.
        for i in 1..=3u32 {
            big.unregister_resident(&surf(i), ResidentReclaim::CapEvicted);
        }
        big.note_registry_reach();
        assert_eq!(
            big.registry_pressure_stats().2,
            bytes_big,
            "the byte band holds its peak"
        );
    }

    /// The maintained non-pinned totals still say what a full walk would.
    ///
    /// They stopped being a walk so the population can grow past `REGISTRY_CAP`
    /// without a per-admit O(n) scan, and the cost of that is three writers that
    /// can fall out of step with the registry in silence. A slot that stopped
    /// being counted makes the population read smaller than it is, which is the
    /// direction that lets a bound sit above itself — so every transition is
    /// driven here and diffed against the walk after each one.
    ///
    /// Counted pins are the part worth driving twice: only the 0 <-> 1 crossings
    /// may move the totals, a second pin must not remove the slot again, and an
    /// unpin that saturates at zero must not add a slot that is already there.
    #[test]
    fn the_maintained_non_pinned_totals_track_the_walk() {
        let mut pools = ResourcePools::new();
        let check = |pools: &ResourcePools, what: &str| {
            assert_eq!(
                pools.registry_non_pinned,
                pools.non_pinned_registry_totals_by_walk(),
                "maintained totals disagree with the walk after {what}"
            );
        };
        check(&pools, "construction");

        admit_sized(&mut pools, surf(1), 0, 0, (16, 16));
        admit_sized(&mut pools, surf(2), 0, 0, (64, 32));
        check(&pools, "two admits");
        assert_eq!(pools.registry_non_pinned.count, 2);

        // First pin removes it; the second must not remove it twice.
        assert!(pools.pin_resident_target(&surf(1), true));
        check(&pools, "first pin");
        assert_eq!(pools.registry_non_pinned.count, 1);
        assert!(pools.pin_resident_target(&surf(1), true));
        check(&pools, "second pin");
        assert_eq!(pools.registry_non_pinned.count, 1, "a second pin moves nothing");

        // First unpin leaves a holder, so it stays out; the second returns it.
        assert!(pools.pin_resident_target(&surf(1), false));
        check(&pools, "first unpin");
        assert_eq!(pools.registry_non_pinned.count, 1);
        assert!(pools.pin_resident_target(&surf(1), false));
        check(&pools, "second unpin");
        assert_eq!(pools.registry_non_pinned.count, 2, "the last unpin returns it");

        // A spurious unpin saturates at zero and must not add it again.
        assert!(pools.pin_resident_target(&surf(1), false));
        check(&pools, "unpin below zero");
        assert_eq!(pools.registry_non_pinned.count, 2);

        // Death, of a pinned slot and an unpinned one — only the unpinned one
        // was ever in the totals.
        assert!(pools.pin_resident_target(&surf(2), true));
        check(&pools, "pinning the second");
        pools.unregister_resident(&surf(2), ResidentReclaim::CapEvicted);
        check(&pools, "unregistering a pinned resident");
        pools.unregister_resident(&surf(1), ResidentReclaim::CapEvicted);
        check(&pools, "unregistering an unpinned resident");
        assert_eq!(pools.registry_non_pinned, NonPinnedTotals::default());

        // And an unregister of something that was never there.
        pools.unregister_resident(&surf(7), ResidentReclaim::CapEvicted);
        check(&pools, "unregistering an absent identity");
    }

    /// The registry reach band records the highest population, and does not
    /// fall back when residents go away.
    ///
    /// Without this the cap's own counter is uninterpretable. A boot reporting
    /// `evicts=0` has said only that `REGISTRY_CAP` did not bind on the workload
    /// that ran, and a peak of 40 and a peak of one-below-the-cap both satisfy
    /// that — opposite answers to whether the bound has headroom. AGENTS.md
    /// states the rule this implements: band the requested reach before widening
    /// or narrowing any table.
    ///
    /// Two properties, because only the pair is a high-water mark. It has to
    /// rise with the population, and it has to *stay* when the population drops
    /// — an instrument that tracked the current value would report whatever the
    /// registry happened to hold at census time and miss every burst, which is
    /// the only thing the cap exists for.
    ///
    /// Pinned residents are excluded, matching what `REGISTRY_CAP` bounds, so a
    /// pinned peer must not inflate the reading.
    #[test]
    fn the_registry_reach_band_holds_the_peak_and_ignores_pinned_residents() {
        let mut pools = ResourcePools::new();
        assert_eq!(
            pools.registry_pressure_stats(),
            (0, 0, 0),
            "a fresh pools has neither reach, loss, nor footprint"
        );

        admit(&mut pools, surf(1), 10, 0);
        admit(&mut pools, surf(2), 10, 0);
        admit(&mut pools, surf(3), 10, 1); // pinned -- not what the cap bounds
        assert_eq!(pools.note_registry_reach(), 2, "the pinned peer is excluded");
        assert_eq!(
            pools.registry_pressure_stats().0,
            2,
            "the band took the non-pinned population"
        );

        admit(&mut pools, surf(4), 10, 0);
        assert_eq!(pools.note_registry_reach(), 3);
        assert_eq!(pools.registry_pressure_stats().0, 3, "the band rose");

        // Every non-pinned resident goes away, leaving only the pinned peer, so
        // the non-pinned population returns to zero. A current-value reading
        // would now report nothing at all and the burst above would be
        // invisible — which is exactly the failure this band prevents.
        // Through `unregister_resident`, not a hand-written map+order removal:
        // that pair is what the maintained non-pinned totals hang off, and a
        // test that wrote both itself would leave them counting residents that
        // are gone.
        for id in [surf(1), surf(2), surf(4)] {
            pools.unregister_resident(&id, ResidentReclaim::CapEvicted);
        }
        assert_eq!(
            pools.note_registry_reach(),
            0,
            "the population really fell, and the pinned peer never counted"
        );
        assert_eq!(
            pools.registry_pressure_stats().0,
            3,
            "the peak is a high-water mark, not the current population"
        );
        assert_eq!(
            pools.registry_pressure_stats().1,
            0,
            "nothing was evicted, so the loss half stays zero"
        );
    }

}
