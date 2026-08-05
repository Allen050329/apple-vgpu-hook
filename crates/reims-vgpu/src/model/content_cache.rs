//! Unbounded content-keyed cache of compiled GPU objects.
//!
//! The sibling of [`super::lru_memo`], and here beside it for the same reason:
//! this is a container, not a backend fact. Nothing in it names a GPU type. It
//! lived inside `backend::metal::cache`, which is behind
//! `feature = "backend-metal"`, so its tests — all of which drive the table with
//! an entry that holds no Metal object at all — were compiled out on every
//! non-Apple host and had never run anywhere a Linux checkout could see. They
//! run on every arm from here, which is the whole point of the location.
//!
//! # Why there is no capacity
//!
//! This held `cap` entries and, once full, overwrote a rotating slot — a clock
//! hand with no reference bit, so the victim was whichever slot came next
//! regardless of whether the guest was still drawing with it.
//!
//! The caps were 96 functions, 64 render pipeline states, 64 compute pipeline
//! states, 64 reflections, 32 samplers and 16 depth-stencil states.
//!
//! **The render-pipeline cap was below what this guest asks for.** The Vulkan
//! arm decodes the same command stream into the same object identities, so its
//! `object_cache_levels` census is a direct reading of the guest's distinct
//! object set. A driven x86 boot, window-drag probe against Safari, settles at:
//!
//! ```text
//!   m2v=75  shaders=75  layouts=33  passes=4  pipelines=92  samplers=14
//!   compute_pipelines=16
//! ```
//!
//! 92 distinct render pipelines against a 64-slot table is not headroom, it is
//! sustained thrash: 28 pipelines more than the table holds, every one of them
//! live, with a rotating hand choosing the victim. On a compositing desktop the
//! object it picks is more often than not one that will be bound again on the
//! next frame, and rebuilding it is `newRenderPipelineStateWithDescriptor:` —
//! a shader compile.
//!
//! So the bound was not protecting the host from the guest; it was capping the
//! guest below what it had already been observed to need. The live entry count
//! here is the number of *distinct* objects the guest has compiled, which is a
//! property of its own program and state set rather than of how long the device
//! has run — the same bound a real driver's pipeline cache has. When a guest
//! genuinely asks for more than the host can build, `newRenderPipelineState…`
//! returns nil and the caller declines with a reason. That is a GPU refusing
//! because its memory is full, which is the behaviour being emulated; silently
//! forgetting an object the guest still has bound is not.
//!
//! Removing the bound removes the linear scan with it. A `find` used to walk
//! every live slot, which was affordable only because the table was small. The
//! entries are indexed by the `u64` prefilter every one of these keys already
//! carries, so a lookup descends an ordered map to one bucket and walks that,
//! and [`CacheEntry::matches`] — the full identity compare — still decides every
//! hit exactly as before.

#![cfg_attr(not(feature = "backend-metal"), allow(dead_code))]

use std::collections::BTreeMap;

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
    /// The full identity compare. This alone decides a hit; [`Self::bucket`]
    /// only narrows which entries are asked.
    fn matches(&self, key: &Self::Key) -> bool;
    /// A cheap `u64` that must be equal whenever [`Self::matches`] is true.
    ///
    /// Every key in this crate already carries one — the prefilter hash its
    /// `matches` consults before the byte compare — so this is a projection of
    /// the existing identity rather than a second one. Two keys sharing a bucket
    /// is merely a longer walk; two keys that match but bucket differently is a
    /// lookup that misses, which is why the invariant is stated here rather than
    /// left to each implementor to rediscover.
    fn bucket(key: &Self::Key) -> u64;
}

/// A process-global content-keyed cache, retained for the life of the process.
///
/// See the module header for why there is no capacity and no replacement rule.
pub(crate) struct ContentCache<E: CacheEntry> {
    /// Bucket (`CacheEntry::bucket`) → the entries filed under it. A bucket
    /// holds more than one entry only on a prefilter-hash collision.
    ///
    /// A `BTreeMap` rather than a `HashMap` for two reasons: its `new` is
    /// `const`, which is what lets the whole table live in the `const fn` the
    /// Metal caches are built from, and the key is already a well-mixed `u64`,
    /// so a handful of integer comparisons beats re-hashing it through
    /// `SipHash`.
    buckets: BTreeMap<u64, Vec<E>>,
}

impl<E: CacheEntry> ContentCache<E> {
    pub(crate) const fn new() -> Self {
        Self {
            buckets: BTreeMap::new(),
        }
    }

    pub(crate) fn find(&self, key: &E::Key) -> Option<&E> {
        self.buckets
            .get(&E::bucket(key))?
            .iter()
            .find(|e| e.matches(key))
    }

    /// Insert `entry`, unless one with its key arrived between the caller's
    /// [`find`](Self::find) and this call — the lock is released in between
    /// while the caller builds the GPU object, so it can.
    pub(crate) fn insert_unique(&mut self, entry: E) -> &E {
        let bucket = self.buckets.entry(E::bucket(entry.key())).or_default();
        let slot = match bucket.iter().position(|e| e.matches(entry.key())) {
            Some(raced) => raced,
            None => {
                bucket.push(entry);
                bucket.len() - 1
            }
        };
        &bucket[slot]
    }

    /// Live entries across every bucket.
    ///
    /// This is the level the `object_cache_levels` census publishes, and the
    /// reading that can falsify the module header's argument: a count still
    /// climbing minutes into a boot means some key is carrying per-frame state
    /// rather than guest state.
    pub(crate) fn len(&self) -> usize {
        self.buckets.values().map(Vec::len).sum()
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
    /// without a device — which is what makes these tests runnable on a host
    /// that has no Metal.
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
        /// Only the hash, exactly as the real keys do — the length is left to
        /// `matches`, so the collision test below shares one bucket.
        fn bucket(key: &ProbeKey) -> u64 {
            key.hash
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
        let mut cache: ContentCache<Probe> = ContentCache::new();
        assert_eq!(cache.insert_unique(probe(1, 10)).tag, 10);
        assert_eq!(
            cache.insert_unique(probe(1, 20)).tag,
            10,
            "the loser of the race gets the winner's entry, not its own"
        );
        assert_eq!(
            cache.len(),
            1,
            "and the cache holds one copy of the key, not two"
        );
    }

    /// Nothing is ever displaced. The retired table held `cap` entries and then
    /// overwrote a rotating slot, so on the Metal arm a guest with more distinct
    /// pipeline states than the cap — which the Vulkan arm measures this guest to
    /// have — recompiled objects it was still drawing with. Drive well past every
    /// cap this container used to be built with (96, 64, 32, 16) and assert the
    /// first entry is still served.
    #[test]
    fn every_distinct_key_is_retained_past_every_retired_capacity() {
        let mut cache: ContentCache<Probe> = ContentCache::new();
        for i in 0..1024 {
            cache.insert_unique(probe(i, i as u32));
        }
        assert_eq!(cache.len(), 1024, "the table never displaces an entry");
        assert_eq!(
            cache.find(&ProbeKey { hash: 0, len: 8 }).map(|e| e.tag),
            Some(0),
            "the first object compiled is still there after 1023 later ones"
        );
        assert_eq!(
            cache.find(&ProbeKey { hash: 1023, len: 8 }).map(|e| e.tag),
            Some(1023)
        );
    }

    /// The length is part of the key, not decoration beside it: two blobs whose
    /// hashes collide must not share one compiled object. They share a bucket —
    /// `bucket` is the hash alone — so this also pins that a bucket walk applies
    /// the full compare rather than trusting the prefilter.
    #[test]
    fn a_hash_collision_across_lengths_is_not_a_hit() {
        let mut cache: ContentCache<Probe> = ContentCache::new();
        cache.insert_unique(probe(7, 70));
        assert!(cache.find(&ProbeKey { hash: 7, len: 8 }).is_some());
        assert!(cache.find(&ProbeKey { hash: 7, len: 9 }).is_none());

        // Both live in bucket 7, and each still resolves to its own entry.
        cache.insert_unique(Probe {
            key: ProbeKey { hash: 7, len: 9 },
            tag: 90,
        });
        assert_eq!(
            cache.find(&ProbeKey { hash: 7, len: 8 }).map(|e| e.tag),
            Some(70)
        );
        assert_eq!(
            cache.find(&ProbeKey { hash: 7, len: 9 }).map(|e| e.tag),
            Some(90)
        );
        assert_eq!(cache.len(), 2);
    }
}
