impl ResourcePools {
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
        // Every fence above was waited (or failed on a lost device, where the
        // handles die with the device anyway), so no slot can still be reading:
        // release the whole graveyard regardless of what each handle waits on.
        self.release_graveyard(device, SlotMask::MAX);
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
        for s in self.sampled_free.drain() {
            device.destroy_image_view(s.view, None);
            device.destroy_image(s.image, None);
        }
        for img in self.target_free.drain() {
            device.destroy_image_view(img.view, None);
            device.destroy_image(img.image, None);
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
        for s in self.storage_image_free.drain() {
            device.destroy_image_view(s.view, None);
            device.destroy_image(s.image, None);
            device.free_memory(s.memory, None);
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
