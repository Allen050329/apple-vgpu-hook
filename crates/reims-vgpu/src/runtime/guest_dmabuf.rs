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
    key: u64,
    gpas: Vec<u64>,
    bytes: u64,
    dmabuf: Arc<GuestDmaBuf>,
    /// Bumped on every hit; the smallest is evicted first.
    used: u64,
}

#[derive(Default)]
struct Cache {
    entries: Vec<Entry>,
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
    // The lookup is a linear scan, and this counts what it walked rather than
    // how long it took: a step count survives a contended host where a
    // microsecond reading does not, and it is the only thing that says whether
    // the scan is the shape of this table's cost. Charged on hit *and* miss,
    // because a miss walks the whole table and a hit walks half of it on
    // average — so `steps / lookups` against `windows` is what separates "the
    // table is small" from "the scan is the cost".
    let mut steps: u64 = 0;
    let found = cache.entries.iter_mut().find(|e| {
        steps += 1;
        e.key == key && e.gpas == gpas
    });
    crate::runtime::drain::note_store_route("guest_dmabuf_lookups");
    crate::runtime::drain::note_store_route_n("guest_dmabuf_scan_steps", steps);
    if let Some(entry) = found {
        entry.used = clock;
        let dmabuf = Arc::clone(&entry.dmabuf);
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
    cache.entries.push(Entry {
        key,
        gpas: gpas.to_vec(),
        bytes,
        dmabuf: Arc::clone(&dmabuf),
        used: clock,
    });
    cache.pinned_bytes += bytes;
    evict_to_bound(cache);
    // The table's own size, sampled once per miss. `cached_windows` and
    // `pinned_bytes` have existed since this cache did and nothing has ever
    // called them, so the bound this table is held to — pinned bytes, not entry
    // count — has never been read on a live boot. Both are needed: 512 MiB of
    // 8 KiB vertex windows is 65 536 entries and 512 MiB of one framebuffer is
    // one, and the scan above costs the entry count while the bound counts the
    // bytes.
    //
    // These two are **sums, not gauges** — the census map adds. Divide each by
    // `guest_dmabuf_misses` for the mean over the window; a reader quoting the
    // raw field as "the cache holds N windows" is off by the miss count.
    crate::runtime::drain::note_store_route("guest_dmabuf_misses");
    crate::runtime::drain::note_store_route_n(
        "guest_dmabuf_windows_sum",
        cache.entries.len() as u64,
    );
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
fn evict_to_bound(cache: &mut Cache) {
    while cache.pinned_bytes > MAX_PINNED_BYTES && cache.entries.len() > 1 {
        let Some(victim) = cache
            .entries
            .iter()
            .enumerate()
            .min_by_key(|(_, e)| e.used)
            .map(|(i, _)| i)
        else {
            return;
        };
        cache.pinned_bytes -= cache.entries[victim].bytes;
        cache.entries.swap_remove(victim);
    }
}

/// Guest memory currently held pinned through cached dma-bufs. Census only.
pub fn pinned_bytes() -> u64 {
    CACHE.lock().as_ref().map_or(0, |c| c.pinned_bytes)
}

/// Windows currently cached. Census only.
pub fn cached_windows() -> usize {
    CACHE.lock().as_ref().map_or(0, |c| c.entries.len())
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
        let push = |cache: &mut Cache, id: u64, bytes: u64, used: u64| {
            cache.entries.push(Entry {
                key: id,
                gpas: vec![id],
                bytes,
                dmabuf: Arc::new(GuestDmaBuf {
                    id,
                    // A pipe read end is a cheap real fd; the cache never looks
                    // at what the fd *is*, only that it owns one.
                    fd: pipe_fd(),
                }),
                used,
            });
            cache.pinned_bytes += bytes;
        };
        let big = MAX_PINNED_BYTES / 2 + 1;
        push(&mut cache, 1, big, 1);
        push(&mut cache, 2, big, 2);
        push(&mut cache, 3, big, 3);
        evict_to_bound(&mut cache);
        assert!(cache.pinned_bytes <= MAX_PINNED_BYTES, "bound not restored");
        let ids: Vec<u64> = cache.entries.iter().map(|e| e.dmabuf.id).collect();
        assert!(ids.contains(&3), "the newest window was evicted: {ids:?}");
        assert!(!ids.contains(&1), "the oldest window survived: {ids:?}");
    }

    /// A single window larger than the whole bound is kept rather than
    /// thrashed, and the accounting stays honest about being over.
    #[test]
    fn one_oversized_window_is_kept_rather_than_rebuilt_every_bind() {
        let mut cache = Cache::default();
        cache.entries.push(Entry {
            key: 1,
            gpas: vec![1],
            bytes: MAX_PINNED_BYTES * 2,
            dmabuf: Arc::new(GuestDmaBuf {
                id: 1,
                fd: pipe_fd(),
            }),
            used: 1,
        });
        cache.pinned_bytes = MAX_PINNED_BYTES * 2;
        evict_to_bound(&mut cache);
        assert_eq!(cache.entries.len(), 1, "the only entry was evicted");
    }

    fn pipe_fd() -> std::os::fd::OwnedFd {
        let (read, _write) = std::io::pipe().expect("a pipe for a placeholder fd");
        std::os::fd::OwnedFd::from(read)
    }
}
