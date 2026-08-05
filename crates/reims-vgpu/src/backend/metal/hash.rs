//! FNV-1a style content hash, used to key the pipeline and shader caches.
//!
//! # What constrains these numbers
//!
//! This file's doc used to say the hash matched an ObjC `reims_vgpu_hash_bytes`.
//! **No such symbol exists anywhere in this repository, in any language, at any
//! commit** — `git log -S` finds only the commit that added that sentence. Read
//! it as provenance for where the algorithm came from, not as a live
//! cross-check, because there is nothing here to cross-check against.
//!
//! So nothing outside this crate pins these values. The only requirement the
//! tree actually imposes is self-consistency within one process: every producer
//! and consumer of a cache key must fold the same way. That is a weaker
//! obligation than an ABI, and worth stating plainly rather than leaving a
//! reader to infer an external contract that is not there.

/// FNV-1a's 64-bit offset basis, and the value every fold here starts from.
///
/// Named because it was written at three sites in **two different bases** —
/// twice in decimal in this file and once as `0xcbf29ce484222325` seeding the
/// render pipeline key in [`super::render`] — and no grep finds the two
/// spellings together. They are equal today; this name is what keeps them
/// equal, since a hash whose two seeds diverge does not fail loudly, it just
/// stops sharing cache entries.
pub const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;

/// FNV-1a's 64-bit prime.
pub const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Fold `data` into a 64-bit content hash.
///
/// The trailing length mix is not stock FNV-1a: it makes a run of zero bytes
/// hash differently from a shorter one, which matters because these keys are
/// compared against shader blobs that can share a prefix.
pub fn hash_bytes(data: &[u8]) -> u64 {
    let mut h: u64 = FNV_OFFSET_BASIS;
    for &b in data {
        h ^= u64::from(b);
        h = h.wrapping_mul(FNV_PRIME);
    }
    h ^= data.len() as u64;
    h = h.wrapping_mul(FNV_PRIME);
    h
}

pub fn hash_u64(mut h: u64, v: u64) -> u64 {
    h ^= v
        .wrapping_add(0x9e3779b97f4a7c15)
        .wrapping_add(h << 6)
        .wrapping_add(h >> 2);
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pinned against a literal, not against a recomputation.
    ///
    /// This test used to re-spell both constants and redo the function's own
    /// final step, so it could only catch an edit to `hash_bytes` — an edit to
    /// a *constant* changed both sides at once and the assertion followed it.
    /// The literal is derivable rather than observed: with no bytes to fold,
    /// `hash_bytes` reduces to FNV-1a over the single zero byte contributed by
    /// the length mix, and FNV-1a of one NUL byte is the published
    /// `0xaf63bd4c8601b7df`. So this pins the basis, the prime, and the length
    /// mix at once, against a value from outside this file.
    #[test]
    fn the_empty_input_hashes_to_fnv1a_of_one_zero_byte() {
        assert_eq!(hash_bytes(b""), 0xaf63_bd4c_8601_b7df);
    }

    /// The two constants are the published FNV-1a ones, whichever base a site
    /// happens to spell them in. Cheap, and it is what makes the literal above
    /// a pin on the algorithm rather than on this implementation of it.
    #[test]
    fn the_constants_are_the_published_fnv1a_pair() {
        assert_eq!(FNV_OFFSET_BASIS, 14695981039346656037);
        assert_eq!(FNV_PRIME, 1099511628211);
    }

    #[test]
    fn distinguishes_content() {
        assert_ne!(hash_bytes(b"a"), hash_bytes(b"b"));
    }
}
