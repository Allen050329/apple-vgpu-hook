impl ResourcePools {
    /// Resolve a guest-RAM host span to a cached VK_EXT_external_memory_host
    /// import, creating the import on first sight. Returns the TRANSFER_SRC
    /// buffer bound over the region and the span's byte offset within it.
    ///
    /// Import candidates, in order: a capped window of the containing VMA
    /// (see [`host_import::HOST_IMPORT_WINDOW_CAP`] — never the whole QEMU RAMBlock;
    /// importing pins host pages for DMA, so a whole-block import would pin
    /// all guest RAM for the VM lifetime) then the aligned span itself
    /// (fallback when the driver rejects the window import). `Err` names the
    /// exact refusal; callers fall back to the CPU byte path.
    ///
    /// The error is **returned**, not just logged, because the logging here is
    /// flood-latched per cause: after the first sighting of each, a caller that
    /// keeps losing guest work has no record of *which* precondition refused.
    /// A scatter store that dies on the byte cap and one that dies on a driver
    /// rejection need opposite fixes, and both used to read `host_import_resolve`
    /// at the loss site because the eight typed variants collapsed into one
    /// `None` at this boundary.
    /// One flood latch per distinct cause.
    ///
    /// Keyed on the variant rather than on "have we logged a host-import failure",
    /// because a shared latch would report whichever cause happened to fire first
    /// and silence the rest — the precise defect that made 190 declines
    /// undiagnosable. The `match` is exhaustive, so a new cause cannot be added
    /// without deciding where its latch lives.
    fn host_import_first_time(&mut self, reason: HostImportDecline) -> bool {
        let seen = match reason {
            HostImportDecline::RegionCount => &mut self.host_import_count_cap_logged,
            HostImportDecline::TotalBytes => &mut self.host_import_byte_cap_logged,
            HostImportDecline::ZeroLength => &mut self.host_import_zero_len_logged,
            HostImportDecline::ExtensionAbsent => &mut self.host_import_no_ext_logged,
            HostImportDecline::PointerMisaligned { .. }
            | HostImportDecline::SizeMisaligned { .. }
            | HostImportDecline::RangeOverflow { .. }
            | HostImportDecline::NoValidWindow { .. } => {
                unreachable!("non-latched import declines are emitted at their attempted site")
            }
        };
        let first = !*seen;
        *seen = true;
        first
    }

    /// Name a budget refusal on the always-on sink, latched per cause.
    ///
    /// Both refusal sites share this: the pre-walk ask at the minimum candidate
    /// size and the per-candidate ask at its real size. `candidate_bytes` is what
    /// was asked for, so the two read apart in the log.
    fn note_host_import_budget_decline(&mut self, reason: HostImportDecline, candidate_bytes: u64) {
        if self.host_import_first_time(reason) {
            crate::observe::Emit::decline("host_import_fail", &reason)
                .field("regions", self.host_imports.len())
                .field("imported_bytes", format!("{:#x}", self.host_import_bytes()))
                .field("candidate_bytes", format!("{candidate_bytes:#x}"))
                .field("cap", format!("{HOST_IMPORT_TOTAL_BYTE_CAP:#x}"))
                .fail();
        }
    }

    /// Record that a resolve asked for `[offset, offset+len)` of the window
    /// based at `base`. Keyed on the base so the map outlives the region and
    /// describes the guest's working set rather than the eviction rate.
    fn mark_host_import_occupancy(&mut self, base: usize, offset: u64, len: u64) {
        self.host_import_occupancy
            .entry(base)
            .or_default()
            .mark(offset, len);
    }

    /// Emit how much of each imported window is actually asked for, at the
    /// window sizes a finer resolver could use.
    ///
    /// This is the reading [`HOST_IMPORT_WINDOW_CAP`]'s doc requires before
    /// either constant moves: `mib` against `windows*1024` is the density, and
    /// `g<N>` is how many N MiB windows would have had to be resident to carry
    /// the same traffic — i.e. `g<N> * N` MiB is what that granule would pin.
    ///
    /// Emitted once per *newly seen bucket*, which is bounded by guest RAM over
    /// the window size and does not track the create rate. Keying it on creates
    /// instead would have gone silent exactly when the pool stopped thrashing:
    /// a healthy boot now makes about a dozen imports in total, and a
    /// per-64-creates line would never fire on one.
    fn report_host_import_occupancy(&self, buckets_before: usize) {
        if self.host_import_occupancy.len() == buckets_before {
            return;
        }
        let chunks = |granule_mib: usize| -> usize {
            self.host_import_occupancy
                .values()
                .map(|o| o.chunks_touched(granule_mib))
                .sum()
        };
        crate::observe::off(format!(
            "host_import_density windows={} mib={} g4={} g16={} g64={} g256={} creates={} evictions={}",
            self.host_import_occupancy.len(),
            chunks(1),
            chunks(4),
            chunks(16),
            chunks(64),
            chunks(256),
            self.host_import_creates,
            self.host_import_evictions,
        ));
    }

    pub(crate) unsafe fn host_import_resolve(
        &mut self,
        ctx: &DeviceContext,
        ptr: usize,
        len: u64,
    ) -> Result<(vk::Buffer, u64), DrawError> {
        // Both of these used to be one unlogged `return None`. They are latched,
        // not per-call: a zero-length span recurs per present and an absent
        // extension refuses every span for the device's lifetime, so an unlatched
        // line would flood. The latch is per cause, so one does not mask the other.
        if len == 0 {
            if self.host_import_first_time(HostImportDecline::ZeroLength) {
                crate::observe::Emit::decline("host_import_fail", &HostImportDecline::ZeroLength)
                    .field("ptr", format!("{ptr:#x}"))
                    .fail();
            }
            return Err(DrawError::HostImport(HostImportDecline::ZeroLength));
        }
        if ctx.ext_external_memory_host.is_none() {
            if self.host_import_first_time(HostImportDecline::ExtensionAbsent) {
                crate::observe::Emit::decline(
                    "host_import_fail",
                    &HostImportDecline::ExtensionAbsent,
                )
                .field("len", format!("{len:#x}"))
                .fail();
            }
            return Err(DrawError::HostImport(HostImportDecline::ExtensionAbsent));
        }
        let Some(end) = (ptr as u64).checked_add(len) else {
            let reason = HostImportDecline::RangeOverflow { host_ptr: ptr, len };
            crate::observe::Emit::decline("host_import_fail", &reason).fail_once(0);
            return Err(DrawError::HostImport(reason));
        };
        if let Some(i) = self
            .host_imports
            .iter()
            .position(|r| r.base as u64 <= ptr as u64 && end <= r.base as u64 + r.len)
        {
            let (touch, epoch, now_ms) =
                (self.host_import_touch + 1, self.host_import_epoch, self.idle_clock_ms);
            self.host_import_touch = touch;
            let r = &mut self.host_imports[i];
            r.last_touch = touch;
            r.last_epoch = epoch;
            r.last_touch_ms = now_ms;
            let (base, buffer) = (r.base, r.buffer);
            let offset = ptr as u64 - base as u64;
            self.mark_host_import_occupancy(base, offset, len);
            return Ok((buffer, offset));
        }
        let align = ctx.min_imported_host_pointer_alignment.max(1);
        // Admit only when admission is free. A miss here is not a failure — the
        // caller writes the span through the CPU byte path — so releasing a
        // resident window to make room trades a few megabytes of `memcpy` for a
        // whole 1 GiB re-import, and then pays it again next frame because the
        // window it released is in the same working set.
        //
        // That is not a hypothetical LRU argument, it is what the census
        // recorded: 2247 creates against 2246 evictions over one browsing
        // session, victims logged `age_ms=0`, 14 one-GiB buckets touched against
        // a budget of 8. An import measured 19.3 ms — longer than a 60 Hz frame
        // — and one drain tranche spent 1342 ms of its life inside
        // `zc_import_us`. The guest's spread is real and no window size closes it
        // (`host_import_density` put 6.4 GiB of touched pages across those
        // buckets at 46 % occupancy), so the overflow has to be *served*, not
        // shuffled.
        //
        // Serving it has to be cheap, which is why the budget is asked inside
        // `host_import_candidates` before the VMA walk rather than after it.
        let candidates = match host_import_candidates(
            self.host_imports.len(),
            self.host_import_bytes(),
            ptr,
            end,
            align,
        ) {
            Ok(candidates) => candidates,
            Err(reason) => {
                self.note_host_import_budget_decline(reason, align);
                return Err(DrawError::HostImport(reason));
            }
        };
        let mut last_error = None;
        for (base, region_len) in candidates {
            if !(base as u64).is_multiple_of(align)
                || region_len % align != 0
                || base > ptr
                || end > base as u64 + region_len
            {
                continue;
            }
            if let Err(reason) =
                host_import_budget(self.host_imports.len(), self.host_import_bytes(), region_len)
            {
                self.note_host_import_budget_decline(reason, region_len);
                return Err(DrawError::HostImport(reason));
            }
            let memory = match ctx.import_host_ptr(base as *mut std::ffi::c_void, region_len) {
                Ok(memory) => memory,
                Err(error) => {
                    last_error = Some(error);
                    continue;
                }
            };
            let info = vk::BufferCreateInfo::default()
                .size(region_len)
                .usage(vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            let buffer = match ctx.device.create_buffer(&info, None) {
                Ok(b) => b,
                Err(result) => {
                    ctx.device.free_memory(memory, None);
                    last_error = Some(DrawError::VkCall(VkCall::new(
                        VkOp::PoolsHostImportCreateBuffer,
                        result,
                    )));
                    continue;
                }
            };
            if let Err(result) = ctx.device.bind_buffer_memory(buffer, memory, 0) {
                ctx.device.destroy_buffer(buffer, None);
                ctx.device.free_memory(memory, None);
                last_error = Some(DrawError::VkCall(VkCall::new(
                    VkOp::PoolsHostImportBindBuffer,
                    result,
                )));
                continue;
            }
            self.host_import_creates = self.host_import_creates.saturating_add(1);
            crate::observe::off(format!(
                "host_import_region base={base:#x} len={region_len:#x} regions={} \
                 creates={} evictions={}",
                self.host_imports.len() + 1,
                self.host_import_creates,
                self.host_import_evictions
            ));
            self.host_import_touch = self.host_import_touch.saturating_add(1);
            self.host_imports.push(HostImportRegion {
                base,
                len: region_len,
                memory,
                buffer,
                last_touch: self.host_import_touch,
                last_epoch: self.host_import_epoch,
                last_touch_ms: self.idle_clock_ms,
            });
            let buckets_before = self.host_import_occupancy.len();
            self.mark_host_import_occupancy(base, ptr as u64 - base as u64, len);
            self.report_host_import_occupancy(buckets_before);
            let r = self.host_imports.last().unwrap();
            return Ok((r.buffer, ptr as u64 - r.base as u64));
        }
        let error = terminal_host_import_error(last_error, ptr, len, align);
        crate::observe::Emit::decline("host_import_fail", &error)
            .field("ptr", format!("{ptr:#x}"))
            .field("len", format!("{len:#x}"))
            .fail_once(0);
        Err(error)
    }

    /// Release `victims` (indices into `host_imports`, as returned by
    /// [`Self::plan_host_import_eviction`]) through the in-flight-safe
    /// deferral. Each is named on the always-on census next to the
    /// `host_import_region` line that created it, so create-vs-evict rates read
    /// off one log: equal rates mean the working set does not fit the budget and
    /// the pool is re-importing what it just released.
    pub(crate) unsafe fn evict_host_imports(
        &mut self,
        ctx: &DeviceContext,
        victims: Vec<usize>,
        reason: &str,
    ) {
        if victims.is_empty() {
            return;
        }
        // Descending so each `swap_remove` leaves the lower indices valid.
        let mut victims = victims;
        victims.sort_unstable_by(|a, b| b.cmp(a));
        victims.dedup();
        for i in victims {
            if i >= self.host_imports.len() {
                continue;
            }
            let region = self.host_imports.swap_remove(i);
            self.host_import_evictions = self.host_import_evictions.saturating_add(1);
            crate::observe::off(format!(
                "host_import_evict base={:#x} len={:#x} reason={reason} age_ms={} regions={} \
                 imported_bytes={:#x} creates={} evictions={}",
                region.base,
                region.len,
                self.idle_clock_ms.saturating_sub(region.last_touch_ms),
                self.host_imports.len(),
                self.host_import_bytes(),
                self.host_import_creates,
                self.host_import_evictions
            ));
            self.dispose(
                &ctx.device,
                DeferredHandle::HostImport {
                    buffer: region.buffer,
                    memory: region.memory,
                },
            );
        }
    }

    pub(crate) unsafe fn destroy_all(&mut self, device: &ash::Device) {
        // An open (never-submitted) batch dies with the pool: its CB belongs
        // to cmd_pool (destroyed below) and its dsets to desc_pool; the
        // accumulated transients are already in the live lists.
        self.open_batch = None;
        // Best-effort quiesce: wait every in-flight fence so no CB references
        // what we are about to destroy. On device loss the waits fail — the
        // teardown proceeds regardless, matching the recreate path.
        for slot in &self.slots {
            if slot.pending.is_some() {
                if let Err(result) = device.wait_for_fences(&[slot.fence], true, FENCE_TIMEOUT_NS) {
                    let decline =
                        DrawError::VkCall(VkCall::new(VkOp::PoolsWaitFencesDestroy, result));
                    crate::observe::Emit::decline("vk_pools_destroy", &decline).fail_once(0);
                }
            }
        }
        for slot in &mut self.slots {
            if let Some(pending) = slot.pending.take() {
                // The descriptor pool is destroyed below (frees every set);
                // move the owed transients into the live lists so the drains
                // below destroy them.
                self.staging_live.extend(pending.staging);
                self.readback_multi_live.extend(pending.readback);
                self.sampled_live.extend(pending.sampled);
                self.storage_image_live.extend(pending.storage_images);
            }
        }
        self.in_flight = 0;
        self.drain_graveyard(device);
        for list in self.staging_free.values_mut() {
            for s in list.drain(..) {
                device.destroy_buffer(s.buffer, None);
                device.free_memory(s.memory, None);
            }
        }
        for s in self.staging_live.drain(..) {
            device.destroy_buffer(s.buffer, None);
            device.free_memory(s.memory, None);
        }
        for list in self.readback_free.values_mut() {
            for s in list.drain(..) {
                device.destroy_buffer(s.buffer, None);
                device.free_memory(s.memory, None);
            }
        }
        if let Some(s) = self.readback_live.take() {
            device.destroy_buffer(s.buffer, None);
            device.free_memory(s.memory, None);
        }
        for s in self.readback_multi_live.drain(..) {
            device.destroy_buffer(s.buffer, None);
            device.free_memory(s.memory, None);
        }
        // Sampled / target / registry images are slab-backed: destroy the image
        // + view handles here, but their memory belongs to shared blocks freed
        // once by `self.slab.destroy_all(device)` at the end — never a per-image
        // `vkFreeMemory` (that would double-free a block many images share).
        for list in self.sampled_free.values_mut() {
            for s in list.drain(..) {
                device.destroy_image_view(s.view, None);
                device.destroy_image(s.image, None);
            }
        }
        for list in self.target_free.values_mut() {
            for img in list.drain(..) {
                device.destroy_image_view(img.view, None);
                device.destroy_image(img.image, None);
            }
        }
        for s in self.sampled_live.drain(..) {
            device.destroy_image_view(s.view, None);
            device.destroy_image(s.image, None);
        }
        for s in self.sampled_cache.drain(..) {
            device.destroy_image_view(s.slot.view, None);
            device.destroy_image(s.slot.image, None);
        }
        self.sampled_cache_bytes = 0;
        for list in self.storage_image_free.values_mut() {
            for s in list.drain(..) {
                device.destroy_image_view(s.view, None);
                device.destroy_image(s.image, None);
                device.free_memory(s.memory, None);
            }
        }
        for s in self.storage_image_live.drain(..) {
            device.destroy_image_view(s.view, None);
            device.destroy_image(s.image, None);
            device.free_memory(s.memory, None);
        }
        for (_, resident) in self.compute_storage_registry.drain() {
            device.destroy_image_view(resident.slot.view, None);
            device.destroy_image(resident.slot.image, None);
            device.free_memory(resident.slot.memory, None);
        }
        self.compute_storage_order.clear();
        for (_, t) in self.targets.drain() {
            device.destroy_framebuffer(t.framebuffer, None);
            device.destroy_image_view(t.view, None);
            device.destroy_image(t.image, None);
        }
        self.target_order.clear();
        for (_, t) in self.registry.drain() {
            device.destroy_framebuffer(t.framebuffer, None);
            device.destroy_image_view(t.view, None);
            device.destroy_image(t.image, None);
        }
        self.registry_order.clear();
        // Free every slab block now that all slab-backed images are destroyed.
        self.slab.destroy_all(device);
        // Host-import regions: guest RAM outlives the device; only the Vulkan
        // handles are ours to free (in-flight fences were waited above).
        for r in self.host_imports.drain(..) {
            device.destroy_buffer(r.buffer, None);
            device.free_memory(r.memory, None);
        }
        // Prefetch slots own dedicated fences + host-coherent buffers; their CBs
        // are freed with `cmd_pool` below. Destroy them before the pool so the
        // best-effort in-flight fence waits happen while the device is live.
        // Same rule for the stats-reduction pool: dedicated fences, a
        // persistently-mapped buffer, a private descriptor pool and a
        // sampler. Its CBs come from `cmd_pool`, so hand it over here and
        // destroy before the pool itself goes.
        self.stats_reduce.destroy_all(device, self.cmd_pool);
        self.host_scatter.destroy_all(device, self.cmd_pool);
        for slot in self.slots.drain(..) {
            device.destroy_fence(slot.fence, None);
        }
        self.cur = 0;
        self.desc_arena.destroy(device);
        if self.cmd_pool != vk::CommandPool::null() {
            device.destroy_command_pool(self.cmd_pool, None);
            self.cmd_pool = vk::CommandPool::null();
        }
        self.initialized = false;
    }
}


#[cfg(test)]
mod host_import_split_tests {
    use super::host_import_budget;

    #[test]
    fn small_arm_aliases_fit_under_shared_byte_ceiling() {
        assert_eq!(host_import_budget(128, 64 << 20, 16 << 10), Ok(()));
    }
}
