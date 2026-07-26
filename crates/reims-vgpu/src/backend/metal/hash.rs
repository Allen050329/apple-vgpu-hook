//! FNV-1a style content hash matching ObjC `reims_vgpu_hash_bytes`.

pub fn hash_bytes(data: &[u8]) -> u64 {
    let mut h: u64 = 14695981039346656037;
    for &b in data {
        h ^= u64::from(b);
        h = h.wrapping_mul(1099511628211);
    }
    h ^= data.len() as u64;
    h = h.wrapping_mul(1099511628211);
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
    #[test]
    fn empty_hash_matches_seed_mix() {
        let mut h: u64 = 14695981039346656037;
        h ^= 0;
        h = h.wrapping_mul(1099511628211);
        assert_eq!(hash_bytes(b""), h);
    }
    #[test]
    fn distinguishes_content() {
        assert_ne!(hash_bytes(b"a"), hash_bytes(b"b"));
    }
}
