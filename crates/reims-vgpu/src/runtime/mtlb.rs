//! MTLB → wrapped-AIR extraction (product DRAW path).
//!
//! Guest type-6 function objects hold per-function MTLB containers; metal2vulkan
//! consumes the LLVM BitcodeWrapper (`0x0b17c0de`) record inside. Port of
//! archive `reims-vgpu-backend-vulkan` `mtlb.rs` (structural carve only — no guest scan).

/// LLVM BitcodeWrapperHeader magic `0x0b17c0de` LE.
const AIR_WRAP_MAGIC: [u8; 4] = [0xde, 0xc0, 0x17, 0x0b];
const WRAPPER_HEADER_LEN: usize = 0x14;

/// A structural refusal while locating the LLVM BitcodeWrapper inside an MTLB.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MtlbDecline {
    WrappedAirMissing {
        data_len: usize,
    },
    WrapperHeaderTruncated {
        offset: usize,
        data_len: usize,
    },
    BlobOutOfBounds {
        offset: usize,
        blob_len: u64,
        data_len: usize,
    },
}

impl crate::observe::Decline for MtlbDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::WrappedAirMissing { .. } => "mtlb_wrapped_air_missing",
            Self::WrapperHeaderTruncated { .. } => "mtlb_wrapper_header_truncated",
            Self::BlobOutOfBounds { .. } => "mtlb_blob_out_of_bounds",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::WrappedAirMissing { data_len } => vec![("data_len", data_len.to_string())],
            Self::WrapperHeaderTruncated { offset, data_len } => vec![
                ("offset", offset.to_string()),
                ("data_len", data_len.to_string()),
            ],
            Self::BlobOutOfBounds {
                offset,
                blob_len,
                data_len,
            } => vec![
                ("offset", offset.to_string()),
                ("blob_len", blob_len.to_string()),
                ("data_len", data_len.to_string()),
            ],
        }
    }
}

impl std::fmt::Display for MtlbDecline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use crate::observe::Decline as _;
        write!(f, "reason={}", self.slug())?;
        for (key, value) in self.fields() {
            write!(f, " {key}={value}")?;
        }
        Ok(())
    }
}

impl std::error::Error for MtlbDecline {}

/// Extract the wrapped-AIR blob from an MTLB container or bare wrapper.
pub fn extract_air(data: &[u8]) -> Result<&[u8], MtlbDecline> {
    let start = find_wrap_magic(data, 0).ok_or(MtlbDecline::WrappedAirMissing {
        data_len: data.len(),
    })?;
    blob_at(data, start)
}

fn find_wrap_magic(data: &[u8], from: usize) -> Option<usize> {
    if data.len() < WRAPPER_HEADER_LEN {
        return None;
    }
    (from..=data.len() - AIR_WRAP_MAGIC.len()).find(|&i| data[i..i + 4] == AIR_WRAP_MAGIC)
}

fn blob_at(data: &[u8], off: usize) -> Result<&[u8], MtlbDecline> {
    let header_end = off.saturating_add(WRAPPER_HEADER_LEN);
    if header_end > data.len() {
        return Err(MtlbDecline::WrapperHeaderTruncated {
            offset: off,
            data_len: data.len(),
        });
    }
    let bc_off = u32::from_le_bytes(data[off + 8..off + 12].try_into().unwrap());
    let bc_size = u32::from_le_bytes(data[off + 12..off + 16].try_into().unwrap());
    let blob_len = u64::from(bc_off) + u64::from(bc_size);
    let blob_end = usize::try_from(blob_len)
        .ok()
        .and_then(|len| off.checked_add(len));
    // Guest/header sizes are authoritative — no product MiB ceiling. Only require
    // the declared blob fits inside the MTLB buffer we already loaded.
    if blob_len < WRAPPER_HEADER_LEN as u64 || blob_end.is_none_or(|end| end > data.len()) {
        return Err(MtlbDecline::BlobOutOfBounds {
            offset: off,
            blob_len,
            data_len: data.len(),
        });
    }
    Ok(&data[off..blob_end.expect("bounds checked above")])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_bare_wrapper() {
        // Minimal synthetic: magic + version + offset 0x14 + size 4 + cpu + 4 body bytes.
        let mut data = vec![0u8; 0x18];
        data[0..4].copy_from_slice(&AIR_WRAP_MAGIC);
        data[4..8].copy_from_slice(&0u32.to_le_bytes()); // version
        data[8..12].copy_from_slice(&0x14u32.to_le_bytes()); // BitcodeOffset
        data[12..16].copy_from_slice(&4u32.to_le_bytes()); // BitcodeSize
        data[0x14..0x18].copy_from_slice(&[1, 2, 3, 4]);
        let air = extract_air(&data).expect("air");
        assert_eq!(air.len(), 0x18);
    }

    #[test]
    fn malformed_wrappers_fire_typed_declines() {
        assert_eq!(
            extract_air(&[]).unwrap_err(),
            MtlbDecline::WrappedAirMissing { data_len: 0 }
        );
        assert_eq!(
            blob_at(&[0; 8], 0).unwrap_err(),
            MtlbDecline::WrapperHeaderTruncated {
                offset: 0,
                data_len: 8
            }
        );

        let mut data = vec![0u8; WRAPPER_HEADER_LEN];
        data[8..12].copy_from_slice(&u32::MAX.to_le_bytes());
        data[12..16].copy_from_slice(&u32::MAX.to_le_bytes());
        let expected = MtlbDecline::BlobOutOfBounds {
            offset: 0,
            blob_len: u64::from(u32::MAX) * 2,
            data_len: WRAPPER_HEADER_LEN,
        };
        assert_eq!(blob_at(&data, 0).unwrap_err(), expected);
    }

    #[test]
    fn mtlb_declines_have_distinct_log_safe_reasons() {
        use crate::observe::Decline as _;
        let cases = [
            MtlbDecline::WrappedAirMissing { data_len: 1 },
            MtlbDecline::WrapperHeaderTruncated {
                offset: 1,
                data_len: 2,
            },
            MtlbDecline::BlobOutOfBounds {
                offset: 1,
                blob_len: 2,
                data_len: 3,
            },
        ];
        let mut slugs = std::collections::HashSet::new();
        for decline in cases {
            assert!(slugs.insert(decline.slug()));
            for (_, value) in decline.fields() {
                assert!(!value.contains(char::is_whitespace));
            }
        }
    }
}
