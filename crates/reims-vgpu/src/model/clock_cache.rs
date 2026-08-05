//! Entry-count-bounded cache with clock replacement.
//!
//! The sibling of [`super::lru_memo`], and here beside it for the same reason:
//! this is a container, not a backend fact. Nothing in it names a GPU type. It
//! lived inside `backend::metal::cache`, which is behind
//! `feature = "backend-metal"`, so its three tests — all of which drive the
//! table with an entry that holds no Metal object at all — were compiled out on
//! every non-Apple host and had never run anywhere a Linux checkout could see.
//!
//! The two bounds differ and the difference is the reason both exist.
//! `LruBytesMemo` bounds *bytes* and evicts the least-recently-touched, because
//! its entries are decoded surfaces whose sizes differ by orders of magnitude
//! and whose hot set must survive a cap crossing. This one bounds *entries* and
//! evicts a rotating slot, because its entries are compiled pipeline objects:
//! the cost is in building one, not in choosing which to drop, and there is no
//! recency signal to consult.
//!
//! Only `backend::metal` builds one today, so on any other arm the type is
//! unconstructed and the allow below is what keeps that from being a warning —
//! the same `cfg_attr` shape `runtime::icb` uses for its Metal-only helpers.
//! The tests still run on every arm, which is the whole point of the move: the
//! container is generic, and gating it to match its one caller would put its
//! only executable coverage back on the host that cannot run it.

#![cfg_attr(not(feature = "backend-metal"), allow(dead_code))]

/// What makes two entries of one cache the same entry.
///
/// Stated once, beside the entry type. Each cache used to state it twice —
/// once in its `_lookup` scan and once in the re-scan its `_insert` does under
/// the lock — six rules in twelve places, with nothing comparing any pair. One
/// of the twelve was already missing: the reflection cache's insert did not
/// re-scan at all, so two callers that missed the same blob both pushed and the
/// cache carried a duplicate.
pub(crate) trait CacheEntry {
    type Key;
    /// The key this entry was filed under. An insert asks the entry for it
    /// rather than taking it a second time from the caller, so the two cannot
    /// disagree.
    fn key(&self) -> &Self::Key;
    fn matches(&self, key: &Self::Key) -> bool;
}

/// A process-global cache with clock replacement.
///
/// Fill to `cap`, then overwrite a rotating slot. There is no recency signal —
/// a clock hand is what these caches have always used, and the entries are
/// compiled GPU objects whose cost is in building them, not in choosing which
/// to drop.
pub(crate) struct ClockCache<E: CacheEntry> {
    entries: Vec<Option<E>>,
    /// Next slot the hand will overwrite once `entries` is at capacity.
    clock: usize,
    cap: usize,
}

impl<E: CacheEntry> ClockCache<E> {
    pub(crate) const fn new(cap: usize) -> Self {
        Self {
            entries: Vec::new(),
            clock: 0,
            cap,
        }
    }

    pub(crate) fn find(&self, key: &E::Key) -> Option<&E> {
        self.entries.iter().flatten().find(|e| e.matches(key))
    }

    /// Insert `entry`, unless one with its key arrived between the caller's
    /// [`find`](Self::find) and this call — the lock is released in between
    /// while the caller builds the GPU object, so it can.
    pub(crate) fn insert_unique(&mut self, entry: E) -> &E {
        let raced = self
            .entries
            .iter()
            .position(|e| e.as_ref().is_some_and(|e| e.matches(entry.key())));
        let slot = match raced {
            Some(raced) => raced,
            None if self.entries.len() < self.cap => {
                self.entries.push(Some(entry));
                self.entries.len() - 1
            }
            None => {
                let slot = self.clock % self.cap;
                self.clock = self.clock.wrapping_add(1);
                if self.entries.len() <= slot {
                    self.entries.resize_with(slot + 1, || None);
                }
                self.entries[slot] = Some(entry);
                slot
            }
        };
        self.entries[slot]
            .as_ref()
            .expect("the slot was just matched or just written")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Stands in for the caches' real keys, which are all a content hash beside
    /// the byte length it was taken over.
    #[derive(Clone, Copy, PartialEq, Eq)]
    struct ProbeKey {
        hash: u64,
        len: usize,
    }

    /// An entry with no GPU object in it, so the table itself can be driven
    /// without a device — which is what makes these three tests runnable on a
    /// host that has no Metal.
    struct Probe {
        key: ProbeKey,
        tag: u32,
    }

    impl CacheEntry for Probe {
        type Key = ProbeKey;
        fn key(&self) -> &ProbeKey {
            &self.key
        }
        fn matches(&self, key: &ProbeKey) -> bool {
            self.key == *key
        }
    }

    fn probe(hash: u64, tag: u32) -> Probe {
        Probe {
            key: ProbeKey { hash, len: 8 },
            tag,
        }
    }

    /// An insert that races another caller's insert must not add a second copy.
    ///
    /// The lock is released between a caller's `find` and its `insert_unique`,
    /// while it builds the GPU object, so two callers can miss the same key and
    /// both arrive here. Five of the six caches re-scanned for that; the
    /// reflection cache did not, and carried the duplicate.
    #[test]
    fn a_raced_insert_returns_the_entry_already_there() {
        let mut cache: ClockCache<Probe> = ClockCache::new(4);
        assert_eq!(cache.insert_unique(probe(1, 10)).tag, 10);
        assert_eq!(
            cache.insert_unique(probe(1, 20)).tag,
            10,
            "the loser of the race gets the winner's entry, not its own"
        );
        assert_eq!(
            cache.entries.iter().flatten().count(),
            1,
            "and the cache holds one copy of the key, not two"
        );
    }

    /// At capacity the hand overwrites a rotating slot and the table stops
    /// growing — the bound is on entries held, so a cache that kept pushing
    /// would hold every GPU object the guest ever compiled.
    #[test]
    fn the_clock_hand_bounds_the_table_at_its_capacity() {
        let mut cache: ClockCache<Probe> = ClockCache::new(3);
        for i in 0..3 {
            cache.insert_unique(probe(i, i as u32));
        }
        assert_eq!(cache.entries.len(), 3);
        assert!(cache.find(&ProbeKey { hash: 0, len: 8 }).is_some());

        // Three more evict the three that were there, one slot at a time.
        for i in 3..6 {
            cache.insert_unique(probe(i, i as u32));
        }
        assert_eq!(cache.entries.len(), 3, "the table never grows past its cap");
        for i in 0..3 {
            assert!(
                cache.find(&ProbeKey { hash: i, len: 8 }).is_none(),
                "entry {i} should have been overwritten"
            );
        }
        for i in 3..6 {
            assert!(cache.find(&ProbeKey { hash: i, len: 8 }).is_some());
        }
    }

    /// The length is part of the key, not decoration beside it: two blobs whose
    /// hashes collide must not share one compiled object.
    #[test]
    fn a_hash_collision_across_lengths_is_not_a_hit() {
        let mut cache: ClockCache<Probe> = ClockCache::new(4);
        cache.insert_unique(probe(7, 70));
        assert!(cache.find(&ProbeKey { hash: 7, len: 8 }).is_some());
        assert!(cache.find(&ProbeKey { hash: 7, len: 9 }).is_none());
    }
}
