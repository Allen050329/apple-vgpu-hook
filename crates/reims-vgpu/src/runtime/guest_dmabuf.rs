//! The dma-buf over a guest page window, made once and kept.
//!
//! # Why this is cached rather than made per draw
//!
//! `UDMABUF_CREATE_LIST` is not a lookup. It walks the run list, takes a
//! reference on every page, and builds a scatter-gather table — and the pages it
//! references become unswappable and unmigratable for as long as the fd lives.
//! A guest surface is bound several times per draw and its pages do not move
//! between binds, so making one per bind would pay that walk thousands of times
//! a second to get the same answer.
//!
//! # What bounds it
//!
//! Pinning is the cost, so pinned bytes are the bound — not entry count, which
//! says nothing about how much of the guest's RAM this device has made
//! unswappable. [`MAX_PINNED_BYTES`] caps it and the least-recently-used window
//! is dropped to stay under.
//!
//! Dropping an entry here does **not** by itself end the GPU's access: an
//! importer holds its own reference through the `VkDeviceMemory` it made, and
//! that reference is what its own bounded cache releases. The two caches are
//! deliberately independent — each bounds the resource it actually holds, and
//! neither can be made to hold the other's — so an entry evicted here is only
//! ever a repeated ioctl, never a use-after-free.
//!
//! # Staleness
//!
//! The key *is* the page list, so a guest that rewires its page table produces a
//! different key and gets a different dma-buf. There is no stale-content case to
//! detect: an entry describes exactly the pages it was made from, and a window
//! naming other pages cannot reach it.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;

use crate::backend::vulkan::engine::GuestDmaBuf;
use crate::runtime::host::{DmaBufExportError, HostOps};

/// Most guest memory this device will hold pinned through dma-bufs at once.
///
/// The basis is the largest single surface the device accepts. Every host pixel
/// buffer here is tightly packed BGRA8, so [`crate::model::MAX_SCANOUT_DIM`]
/// squared at 4 bytes per texel — 256 MiB — is the largest window that can ever
/// be asked for. Two of those means a full-size frame's windows plus a second
/// set arriving, which is what keeps a resize or a display change from evicting
/// the frame it is in the middle of drawing; and it bounds the share of the
/// guest's RAM this device can make unswappable at 512 MiB regardless of how
/// much the guest has.
pub const MAX_PINNED_BYTES: u64 =
    2 * (crate::model::MAX_SCANOUT_DIM as u64) * (crate::model::MAX_SCANOUT_DIM as u64) * 4;

/// Source of the ids in [`GuestDmaBuf::id`]. Monotonic and never reused, which
/// is the property an importer's cache key needs and an fd number lacks.
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

struct Entry {
    gpas: Vec<u64>,
    bytes: u64,
    dmabuf: Arc<GuestDmaBuf>,
    /// Bumped on every hit; the smallest is evicted first.
    used: u64,
}

#[derive(Default)]
struct Cache {
    /// Windows grouped by [`digest`] of their page list.
    ///
    /// A bucket holds more than one entry only on a digest collision, so the
    /// full page-list comparison that decides a hit runs once per lookup
    /// instead of once per table entry. The digest is not stored on the
    /// [`Entry`] as well: the map key already is it, and a second copy is a
    /// field that can drift from the bucket it sits in.
    buckets: std::collections::HashMap<u64, Vec<Entry>>,
    /// Windows across all buckets. Kept alongside rather than summed on demand
    /// because eviction tests it every pass and the census reads it per miss.
    entry_count: usize,
    pinned_bytes: u64,
    clock: u64,
    /// Set once the host has refused for a reason that cannot change while the
    /// process runs, so a host with no `/dev/udmabuf` stops asking per bind.
    /// A per-window refusal — a scattered run list, a page that is not RAM —
    /// must NOT land here: it says nothing about the next window.
    host_refusal: Option<DmaBufExportError>,
}

static CACHE: Mutex<Option<Cache>> = Mutex::new(None);

/// Whether a refusal is a property of the host rather than of this window.
///
/// The distinction decides whether asking again can ever produce a different
/// answer. A missing `/dev/udmabuf` and a guest RAM with no backing fd are
/// settled for the life of the process; a run list past the bound, a page that
/// is not RAM, and a failed create are all properties of the window in hand and
/// say nothing about the next one.
fn is_permanent(error: DmaBufExportError) -> bool {
    matches!(
        error,
        DmaBufExportError::CallbackMissing
            | DmaBufExportError::Unsupported
            | DmaBufExportError::NotMemfd
    )
}

/// A 64-bit digest of the page list, used as the cache key.
///
/// FNV-1a over the GPAs and the page size. A collision would bind one window's
/// dma-buf for another's pages, so the full list is compared on hit rather than
/// trusting the digest — the digest only narrows the search.
fn digest(gpas: &[u64], page_size: u32) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    let mut mix = |value: u64| {
        for byte in value.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    mix(u64::from(page_size));
    mix(gpas.len() as u64);
    for gpa in gpas {
        mix(*gpa);
    }
    hash
}

/// Find the window for `gpas`, marking it used, and report what the search
/// compared.
///
/// # Why the step count is still taken
///
/// It is the proof that the bucket is doing its job. The comparison this counts
/// is the full page list — the digest only narrows the search and cannot decide
/// a hit — so a lookup that compares one list has resolved the table in one
/// step, and one that compares many has found a digest collision. Charged on
/// hit *and* miss, so `guest_dmabuf_scan_steps / guest_dmabuf_lookups` read
/// against `guest_dmabuf_windows_sum / guest_dmabuf_misses` says whether the
/// search still scales with the table.
///
/// A count rather than a timer: this rail is read on a shared development host
/// where every `_us` field is an upper bound, and counts survive contention.
fn lookup(
    cache: &mut Cache,
    key: u64,
    gpas: &[u64],
    clock: u64,
) -> (Option<Arc<GuestDmaBuf>>, u64) {
    let mut steps: u64 = 0;
    let Some(bucket) = cache.buckets.get_mut(&key) else {
        return (None, steps);
    };
    let found = bucket.iter_mut().find(|e| {
        steps += 1;
        e.gpas == gpas
    });
    match found {
        Some(entry) => {
            entry.used = clock;
            (Some(Arc::clone(&entry.dmabuf)), steps)
        }
        None => (None, steps),
    }
}

/// The dma-buf over `gpas`, made if this is the first time these pages have been
/// asked for and reused otherwise.
///
/// `None` when this host cannot export them. That is a routing answer and not a
/// failure — the caller gathers on the CPU instead — but it is emitted through
/// the fail-visible path the first time each distinct reason appears, because a
/// silent fall back to the copy is exactly the "works, but slowly, and nobody
/// knows why" case the zero-copy rail exists to remove.
pub fn dmabuf_for<M: HostOps>(
    host: &mut M,
    gpas: &[u64],
    page_size: u32,
) -> Option<Arc<GuestDmaBuf>> {
    if gpas.is_empty() || page_size == 0 {
        return None;
    }
    let key = digest(gpas, page_size);
    let mut guard = CACHE.lock();
    let cache = guard.get_or_insert_with(Cache::default);
    if let Some(refusal) = cache.host_refusal {
        // Settled for the life of the process. Asking again would ioctl on
        // every bind to be told the same thing.
        let _ = refusal;
        return None;
    }
    cache.clock += 1;
    let clock = cache.clock;
    let (found, steps) = lookup(cache, key, gpas, clock);
    crate::runtime::drain::note_store_route("guest_dmabuf_lookups");
    crate::runtime::drain::note_store_route_n("guest_dmabuf_scan_steps", steps);
    if let Some(dmabuf) = found {
        crate::runtime::drain::note_store_route("guest_dmabuf_hits");
        return Some(dmabuf);
    }

    let fd = match host.dmabuf_for_pages(gpas, page_size as usize) {
        Ok(fd) => fd,
        Err(error) => {
            if is_permanent(error) {
                cache.host_refusal = Some(error);
            }
            drop(guard);
            // Fail-visible, and latched per reason: a host that cannot export at
            // all says so once rather than once per bind, and a window that is
            // too scattered says so with the numbers that made it so.
            crate::observe::Emit::decline("guest_dmabuf_export", &error)
                .field("pages", gpas.len())
                .field("page_size", page_size)
                .fail_once(0);
            return None;
        }
    };

    let bytes = gpas.len() as u64 * u64::from(page_size);
    let dmabuf = Arc::new(GuestDmaBuf {
        id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
        fd,
    });
    cache.buckets.entry(key).or_default().push(Entry {
        gpas: gpas.to_vec(),
        bytes,
        dmabuf: Arc::clone(&dmabuf),
        used: clock,
    });
    cache.entry_count += 1;
    cache.pinned_bytes += bytes;
    evict_to_bound(cache);
    // The table's own size, sampled once per miss. Both fields are needed
    // because the entry count and the pin come apart by orders of magnitude:
    // 512 MiB of 8 KiB vertex windows is 65 536 entries and 512 MiB of one
    // framebuffer is one.
    //
    // A driven x86/Vulkan boot reads this table at ~956 windows holding ~490 MiB
    // — right against [`MAX_PINNED_BYTES`], so it evicts continuously, and far
    // longer than a table a linear search can afford. That reading is what put
    // the lookup in a bucket; it stays sampled because the bound it establishes
    // is the one [`evict_to_bound`]'s cost is priced against.
    //
    // These two are **sums, not gauges** — the census map adds. Divide each by
    // `guest_dmabuf_misses` for the mean over the window; a reader quoting the
    // raw field as "the cache holds N windows" is off by the miss count.
    crate::runtime::drain::note_store_route("guest_dmabuf_misses");
    crate::runtime::drain::note_store_route_n("guest_dmabuf_windows_sum", cache.entry_count as u64);
    crate::runtime::drain::note_store_route_n(
        "guest_dmabuf_pinned_kb_sum",
        cache.pinned_bytes >> 10,
    );
    Some(dmabuf)
}

/// Drop least-recently-used windows until the pinned total is back under
/// [`MAX_PINNED_BYTES`].
///
/// The entry just inserted is the most recently used, so it is never the one
/// evicted — a single window larger than the whole bound would otherwise be
/// created and dropped on every bind, which is worse than not caching at all.
///
/// Still a walk of every window, and deliberately so: it runs only on a miss
/// that pushed the pin over the bound, which the lookup no longer does, and an
/// LRU order maintained on the side would have to be updated on every one of
/// the millions of hits to save work on the thousands of evictions.
fn evict_to_bound(cache: &mut Cache) {
    while cache.pinned_bytes > MAX_PINNED_BYTES && cache.entry_count > 1 {
        // Ties in `used` break in whatever order the map iterates. They cannot
        // arise between two live windows — the clock advances once per lookup
        // and only the window that lookup touched takes the new value — and
        // between equally-stale windows either choice is the same choice.
        let Some((key, idx)) = cache
            .buckets
            .iter()
            .flat_map(|(k, bucket)| bucket.iter().enumerate().map(move |(i, e)| (*k, i, e.used)))
            .min_by_key(|&(_, _, used)| used)
            .map(|(k, i, _)| (k, i))
        else {
            return;
        };
        let Some(bucket) = cache.buckets.get_mut(&key) else {
            return;
        };
        cache.pinned_bytes -= bucket[idx].bytes;
        bucket.swap_remove(idx);
        if bucket.is_empty() {
            cache.buckets.remove(&key);
        }
        cache.entry_count -= 1;
    }
}

/// Guest memory currently held pinned through cached dma-bufs. Census only.
pub fn pinned_bytes() -> u64 {
    CACHE.lock().as_ref().map_or(0, |c| c.pinned_bytes)
}

/// Windows currently cached. Census only.
pub fn cached_windows() -> usize {
    CACHE.lock().as_ref().map_or(0, |c| c.entry_count)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two different page lists must not share a dma-buf, and the same list
    /// must not build a second one.
    ///
    /// The digest alone cannot carry this: it narrows the search, and the full
    /// list decides. A cache that trusted the digest would hand one window's
    /// pages to another on a collision, which is silent corruption rather than
    /// a slow frame.
    #[test]
    fn the_key_is_the_page_list_and_not_just_its_digest() {
        let a = [0x1000u64, 0x2000, 0x3000];
        let b = [0x1000u64, 0x2000, 0x4000];
        assert_ne!(digest(&a, 4096), digest(&b, 4096));
        // Same pages under a different page size are a different window: the
        // dma-buf's ranges are sized from it.
        assert_ne!(digest(&a, 4096), digest(&a, 16384));
        assert_eq!(digest(&a, 4096), digest(&a, 4096));
    }

    /// A host that cannot export at all is asked once. The alternative is an
    /// ioctl per bind — several thousand a second under compositing — to be
    /// told the same thing every time.
    #[test]
    fn a_permanent_refusal_is_settled_and_a_local_one_is_not() {
        for permanent in [
            DmaBufExportError::CallbackMissing,
            DmaBufExportError::Unsupported,
            DmaBufExportError::NotMemfd,
        ] {
            assert!(is_permanent(permanent), "{permanent:?}");
        }
        // These describe the window in hand. Latching any of them would turn
        // one scattered surface into a process-wide loss of the rail.
        for local in [
            DmaBufExportError::TooFragmented,
            DmaBufExportError::NotRam,
            DmaBufExportError::Create,
            DmaBufExportError::Alignment,
            DmaBufExportError::PageSize,
            DmaBufExportError::Args,
            DmaBufExportError::UnknownCode(-99),
        ] {
            assert!(!is_permanent(local), "{local:?}");
        }
    }

    /// The bound is on pinned bytes, and eviction never drops the entry that
    /// was just made.
    ///
    /// A window larger than the whole bound would otherwise be created, counted,
    /// and immediately evicted on every single bind — paying the page walk every
    /// time to cache nothing, which is strictly worse than not caching.
    #[test]
    fn eviction_bounds_pinned_bytes_and_keeps_the_newest() {
        let mut cache = Cache::default();
        let big = MAX_PINNED_BYTES / 2 + 1;
        push(&mut cache, 1, big, 1);
        push(&mut cache, 2, big, 2);
        push(&mut cache, 3, big, 3);
        evict_to_bound(&mut cache);
        assert!(cache.pinned_bytes <= MAX_PINNED_BYTES, "bound not restored");
        let ids = ids(&cache);
        assert!(ids.contains(&3), "the newest window was evicted: {ids:?}");
        assert!(!ids.contains(&1), "the oldest window survived: {ids:?}");
        assert_eq!(cache.entry_count, ids.len(), "entry_count drifted");
    }

    /// A hit resolves the table in one page-list comparison however many windows
    /// it holds.
    ///
    /// This is the bound the bucket exists for. The lookup used to compare every
    /// entry's digest in turn, and a driven x86/Vulkan boot read 3.6 M lookups
    /// against 1.61 G comparisons — 444 per lookup, over a table averaging 956
    /// windows, which is the half-table walk a linear scan of a hit costs. The
    /// assertion is on the *step count* rather than a duration for the reason
    /// [`lookup`] gives.
    ///
    /// A thousand windows rather than a handful: the failure mode being excluded
    /// is one that only shows up as the table grows, so a table small enough for
    /// a scan to look fine cannot witness it.
    #[test]
    fn a_hit_does_not_scale_with_the_number_of_cached_windows() {
        let mut cache = Cache::default();
        for id in 1..=1000u64 {
            push(&mut cache, id, 4096, id);
        }
        assert_eq!(cache.entry_count, 1000);
        // The window inserted last and the one inserted first cost the same,
        // which a scan cannot manage for both at once.
        for probe in [1u64, 500, 1000] {
            let gpas = [probe];
            let (found, steps) = lookup(&mut cache, digest(&gpas, 4096), &gpas, 10_000);
            assert!(found.is_some(), "window {probe} was not found");
            assert_eq!(steps, 1, "window {probe} took {steps} comparisons");
        }
        // A window that was never cached compares nothing at all: its digest
        // names no bucket.
        let absent = [0xdead_beefu64];
        let (found, steps) = lookup(&mut cache, digest(&absent, 4096), &absent, 10_001);
        assert!(found.is_none());
        assert_eq!(steps, 0, "a miss walked {steps} entries");
    }

    /// Two windows whose page lists collide under [`digest`] share a bucket, and
    /// each still gets its own dma-buf.
    ///
    /// The digest narrows and the full list decides, so the bucket has to hold
    /// both and the comparison inside it has to separate them. Constructed by
    /// filing both under one key rather than by finding a real FNV-1a collision:
    /// what is under test is the bucket's behaviour when it holds two, not the
    /// hash's spread.
    #[test]
    fn a_digest_collision_keeps_the_windows_apart() {
        let mut cache = Cache::default();
        let shared = 0x5eed_u64;
        for id in [11u64, 22] {
            cache.buckets.entry(shared).or_default().push(Entry {
                gpas: vec![id],
                bytes: 4096,
                dmabuf: Arc::new(GuestDmaBuf { id, fd: pipe_fd() }),
                used: id,
            });
            cache.entry_count += 1;
            cache.pinned_bytes += 4096;
        }
        for (probe, want) in [(11u64, 11u64), (22, 22)] {
            let (found, steps) = lookup(&mut cache, shared, &[probe], 99);
            assert_eq!(found.expect("collided window").id, want);
            assert!((1..=2).contains(&steps), "unexpected step count {steps}");
        }
        // A third list under the same digest matches neither.
        let (found, _) = lookup(&mut cache, shared, &[33], 100);
        assert!(found.is_none(), "a non-member matched inside the bucket");
    }

    /// Push a window with a one-page list of `id` under its real digest.
    fn push(cache: &mut Cache, id: u64, bytes: u64, used: u64) {
        let gpas = vec![id];
        cache
            .buckets
            .entry(digest(&gpas, 4096))
            .or_default()
            .push(Entry {
                gpas,
                bytes,
                dmabuf: Arc::new(GuestDmaBuf {
                    id,
                    // A pipe read end is a cheap real fd; the cache never looks
                    // at what the fd *is*, only that it owns one.
                    fd: pipe_fd(),
                }),
                used,
            });
        cache.entry_count += 1;
        cache.pinned_bytes += bytes;
    }

    fn ids(cache: &Cache) -> Vec<u64> {
        cache
            .buckets
            .values()
            .flatten()
            .map(|e| e.dmabuf.id)
            .collect()
    }

    /// A single window larger than the whole bound is kept rather than
    /// thrashed, and the accounting stays honest about being over.
    #[test]
    fn one_oversized_window_is_kept_rather_than_rebuilt_every_bind() {
        let mut cache = Cache::default();
        push(&mut cache, 1, MAX_PINNED_BYTES * 2, 1);
        evict_to_bound(&mut cache);
        assert_eq!(cache.entry_count, 1, "the only entry was evicted");
    }

    fn pipe_fd() -> std::os::fd::OwnedFd {
        let (read, _write) = std::io::pipe().expect("a pipe for a placeholder fd");
        std::os::fd::OwnedFd::from(read)
    }
}
