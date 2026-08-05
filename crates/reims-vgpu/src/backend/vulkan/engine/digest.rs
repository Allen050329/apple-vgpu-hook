//! ≥128-bit content digests for cache keys (never bare DefaultHasher u64 alone).
//!
//! # This rule is this backend's, and the other backend does not follow it
//!
//! Stated here so the asymmetry is visible from the side that has the rule.
//! `backend::metal::cache::RenderPsoKey` identifies a shader by a 64-bit
//! non-keyed FNV-1a fingerprint plus its length and never compares the bytes —
//! the same "matches by fingerprint alone" shape that
//! `engine::pools::sampled_content_hash` argues needs 128 bits. That struct's
//! doc carries the full account, including why it was recorded rather than
//! changed. Do not read the rule above as crate-wide: it holds on the Vulkan arm
//! only, and a reader who assumes otherwise will mis-price a Metal cache.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Two independently-seeded 64-bit hashes + byte length (≥128-bit effective).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Default)]
pub struct Digest128 {
    pub a: u64,
    pub b: u64,
    pub len: u64,
}

impl Digest128 {
    pub fn of_bytes(bytes: &[u8]) -> Self {
        let mut ha = DefaultHasher::new();
        0x9e37_79b9_7f4a_7c15u64.hash(&mut ha);
        bytes.hash(&mut ha);
        let a = ha.finish();

        let mut hb = DefaultHasher::new();
        0xc2b2_ae3d_27d4_eb4fu64.hash(&mut hb);
        // reverse-mix seed so collisions in a alone do not imply collisions in b
        for chunk in bytes.chunks(8).rev() {
            chunk.hash(&mut hb);
        }
        bytes.len().hash(&mut hb);
        let b = hb.finish();

        Self {
            a,
            b,
            len: bytes.len() as u64,
        }
    }

    pub fn of_u32_words(words: &[u32]) -> Self {
        #[cfg(target_endian = "little")]
        {
            // LE hosts: the words already are the byte sequence — hash in place
            // (this runs per draw; the copy was a full-module alloc each call).
            let bytes =
                unsafe { std::slice::from_raw_parts(words.as_ptr().cast::<u8>(), words.len() * 4) };
            Self::of_bytes(bytes)
        }
        #[cfg(not(target_endian = "little"))]
        {
            let mut bytes = Vec::with_capacity(words.len() * 4);
            for w in words {
                bytes.extend_from_slice(&w.to_le_bytes());
            }
            Self::of_bytes(&bytes)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digest_differs_on_content() {
        let a = Digest128::of_bytes(b"hello");
        let b = Digest128::of_bytes(b"world");
        assert_ne!(a, b);
        assert_eq!(a, Digest128::of_bytes(b"hello"));
    }

    #[test]
    fn digest_length_distinguishes_prefix() {
        let a = Digest128::of_bytes(b"ab");
        let b = Digest128::of_bytes(b"abc");
        assert_ne!(a, b);
    }
}
