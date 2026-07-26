//! Blit planning/normalization (port of `host/utils/reims-vgpu-blit-plan`).

use crate::runtime::decode::blit::{
    self, Command as BlitCommand, CopyKind, Kind, OP_GENERATE_MIPMAPS,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Ok,
    ErrArgs,
    ErrUnsupported,
    ErrOverflow,
    ErrZeroSize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlannedCopy {
    pub copy_kind: CopyKind,
    pub source: u32,
    pub destination: u32,
    pub source_offset: u64,
    pub destination_offset: u64,
    pub size: u64,
    pub source_bytes_per_row: u64,
    pub destination_bytes_per_row: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlannedFill {
    pub buffer: u32,
    pub range_location: u64,
    pub range_length: u64,
    pub fill_value: u8,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlannedBlit {
    Copy(PlannedCopy),
    Fill(PlannedFill),
    /// Blit `0x133 generateMipmaps` — filter L0 into lower mips on a texture ref.
    GenerateMipmaps {
        texture: u32,
    },
    /// Blit `0x13e` whole-surface multi-slice / multi-level texture copy.
    ///
    /// Zero `slice_count` or `level_count` is a no-op plan (Metal contract).
    CopySliceLevel {
        source: u32,
        destination: u32,
        source_slice: u16,
        source_level: u16,
        destination_slice: u16,
        destination_level: u16,
        slice_count: u16,
        level_count: u16,
    },
    Resource {
        resource: u32,
    },
    Image {
        texture: u32,
        slice: u16,
        level: u16,
    },
    Fence {
        fence: u32,
    },
    Nop,
}

pub fn plan_blit(cmd: &BlitCommand) -> Result<PlannedBlit, Status> {
    match cmd.kind {
        Kind::Copy if cmd.copy_kind == CopyKind::TextureToTextureSliceLevel => {
            if cmd.source == 0 || cmd.destination == 0 {
                return Err(Status::ErrArgs);
            }
            // Zero counts are a typed no-op (Metal); keep the plan shape for exec.
            Ok(PlannedBlit::CopySliceLevel {
                source: cmd.source,
                destination: cmd.destination,
                source_slice: cmd.source_slice,
                source_level: cmd.source_level,
                destination_slice: cmd.destination_slice,
                destination_level: cmd.destination_level,
                slice_count: cmd.slice_count,
                level_count: cmd.level_count,
            })
        }
        Kind::Copy => {
            let size = if cmd.copy_kind == CopyKind::BufferToBuffer {
                cmd.size
            } else {
                // Texture copies: product of extents when non-zero.
                let w = cmd.source_size.width.max(1);
                let h = cmd.source_size.height.max(1);
                let d = cmd.source_size.depth.max(1);
                w.checked_mul(h)
                    .and_then(|x| x.checked_mul(d))
                    .ok_or(Status::ErrOverflow)?
            };
            if cmd.copy_kind == CopyKind::BufferToBuffer && size == 0 {
                return Err(Status::ErrZeroSize);
            }
            Ok(PlannedBlit::Copy(PlannedCopy {
                copy_kind: cmd.copy_kind,
                source: cmd.source,
                destination: cmd.destination,
                source_offset: cmd.source_offset,
                destination_offset: cmd.destination_offset,
                size,
                source_bytes_per_row: cmd.source_bytes_per_row,
                destination_bytes_per_row: cmd.destination_bytes_per_row,
            }))
        }
        Kind::FillBuffer => {
            if cmd.range_length == 0 {
                return Err(Status::ErrZeroSize);
            }
            Ok(PlannedBlit::Fill(PlannedFill {
                buffer: cmd.buffer,
                range_location: cmd.range_location,
                range_length: cmd.range_length,
                fill_value: cmd.fill_value,
            }))
        }
        Kind::Resource if cmd.opcode == OP_GENERATE_MIPMAPS => {
            if cmd.resource == 0 {
                return Err(Status::ErrArgs);
            }
            Ok(PlannedBlit::GenerateMipmaps {
                texture: cmd.resource,
            })
        }
        Kind::Resource => Ok(PlannedBlit::Resource {
            resource: cmd.resource,
        }),
        Kind::Image => Ok(PlannedBlit::Image {
            texture: cmd.texture,
            slice: cmd.slice,
            level: cmd.level,
        }),
        Kind::Fence => Ok(PlannedBlit::Fence { fence: cmd.fence }),
        Kind::Unknown => Err(Status::ErrUnsupported),
    }
}

pub fn plan_from_bytes(bytes: &[u8]) -> Result<PlannedBlit, Status> {
    let cmd = blit::decode(bytes).map_err(|_| Status::ErrArgs)?;
    plan_blit(&cmd)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::endian::{st32, st64};
    use crate::runtime::decode::blit::OP_COPY_TEXTURE_TO_TEXTURE_SLICE_LEVEL;

    #[test]
    fn plan_b2b() {
        let mut v = vec![0u8; 0x28];
        st32(&mut v[0..], 0x12d);
        st32(&mut v[4..], 0x28);
        st32(&mut v[8..], 1);
        st32(&mut v[12..], 2);
        st64(&mut v[0x10..], 0);
        st64(&mut v[0x18..], 0);
        st64(&mut v[0x20..], 64);
        match plan_from_bytes(&v).unwrap() {
            PlannedBlit::Copy(c) => {
                assert_eq!(c.size, 64);
                assert_eq!(c.source, 1);
            }
            _ => panic!("expected copy"),
        }
    }

    #[test]
    fn plan_generate_mipmaps() {
        let mut v = vec![0u8; 0x0c];
        st32(&mut v[0..], 0x133);
        st32(&mut v[4..], 0x0c);
        st32(&mut v[8..], 7);
        match plan_from_bytes(&v).unwrap() {
            PlannedBlit::GenerateMipmaps { texture } => assert_eq!(texture, 7),
            _ => panic!("expected generate mipmaps"),
        }
        // Zero ref is ErrArgs.
        st32(&mut v[8..], 0);
        assert_eq!(plan_from_bytes(&v), Err(Status::ErrArgs));
    }

    #[test]
    fn plan_copy_slice_level() {
        use crate::contract::endian::st16;
        let mut v = vec![0u8; 0x1c];
        st32(&mut v[0..], OP_COPY_TEXTURE_TO_TEXTURE_SLICE_LEVEL);
        st32(&mut v[4..], 0x1c);
        st32(&mut v[8..], 2); // source
        st32(&mut v[12..], 3); // dest
        st16(&mut v[0x10..], 1); // src slice
        st16(&mut v[0x12..], 0); // src level
        st16(&mut v[0x14..], 0); // dst slice
        st16(&mut v[0x16..], 0); // dst level
        st16(&mut v[0x18..], 2); // slice count
        st16(&mut v[0x1a..], 1); // level count
        match plan_from_bytes(&v).unwrap() {
            PlannedBlit::CopySliceLevel {
                source,
                destination,
                source_slice,
                slice_count,
                level_count,
                ..
            } => {
                assert_eq!((source, destination), (2, 3));
                assert_eq!(source_slice, 1);
                assert_eq!((slice_count, level_count), (2, 1));
            }
            _ => panic!("expected copy slice/level"),
        }
        // Zero counts still plan (no-op at exec).
        st16(&mut v[0x18..], 0);
        match plan_from_bytes(&v).unwrap() {
            PlannedBlit::CopySliceLevel { slice_count, .. } => assert_eq!(slice_count, 0),
            _ => panic!("expected plan"),
        }
    }
}
