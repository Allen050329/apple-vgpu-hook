//! Render planning (port of `host/utils/reims-vgpu-render-plan`).

use crate::runtime::decode::render::{self, Command as RenderCommand, Kind, Stage};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Ok,
    ErrArgs,
    ErrUnsupported,
    ErrMissingPipeline,
    ErrZeroDraw,
}

#[derive(Clone, Debug, PartialEq)]
pub enum PlannedRender {
    SetPipeline {
        pipeline_ref: u32,
    },
    SetBuffer {
        stage: Stage,
        index: u32,
        buffer_ref: u32,
        offset: u64,
    },
    SetTexture {
        stage: Stage,
        index: u32,
        texture_ref: u32,
    },
    Draw {
        indexed: bool,
        vertex_or_index_count: u32,
        instance_count: u32,
        primitive_type: u32,
    },
    SetViewport {
        values: [f64; 6],
    },
    SetScissor {
        x: u32,
        y: u32,
        w: u32,
        h: u32,
    },
    Fence {
        wait: bool,
        fence_ref: u32,
    },
    Other {
        kind: Kind,
        opcode: u32,
    },
}

pub fn plan_render(cmd: &RenderCommand) -> Result<PlannedRender, Status> {
    match cmd.kind {
        Kind::SetPipeline => {
            if cmd.pipeline_ref == 0 {
                return Err(Status::ErrMissingPipeline);
            }
            Ok(PlannedRender::SetPipeline {
                pipeline_ref: cmd.pipeline_ref,
            })
        }
        Kind::SetBuffer => Ok(PlannedRender::SetBuffer {
            stage: cmd.stage,
            index: cmd.first,
            buffer_ref: cmd.buffer_ref,
            offset: cmd.buffer_offset,
        }),
        Kind::SetTexture => Ok(PlannedRender::SetTexture {
            stage: cmd.stage,
            index: cmd.first,
            texture_ref: cmd.texture_ref,
        }),
        Kind::Draw => {
            let count = if cmd.index_count != 0 {
                cmd.index_count
            } else {
                cmd.vertex_count
            };
            if count == 0 {
                return Err(Status::ErrZeroDraw);
            }
            Ok(PlannedRender::Draw {
                indexed: cmd.index_count != 0,
                vertex_or_index_count: count,
                instance_count: cmd.instance_count.max(1),
                primitive_type: cmd.primitive_type,
            })
        }
        Kind::SetViewport => Ok(PlannedRender::SetViewport {
            values: cmd.viewport,
        }),
        Kind::SetScissor => Ok(PlannedRender::SetScissor {
            x: cmd.scissor_x,
            y: cmd.scissor_y,
            w: cmd.scissor_w,
            h: cmd.scissor_h,
        }),
        Kind::Fence => Ok(PlannedRender::Fence {
            wait: cmd.opcode == render::OP_WAIT_FENCE,
            fence_ref: cmd.fence_ref,
        }),
        Kind::RenderPass => Ok(PlannedRender::Other {
            kind: Kind::RenderPass,
            opcode: cmd.opcode,
        }),
        Kind::Unknown => Err(Status::ErrUnsupported),
        other => Ok(PlannedRender::Other {
            kind: other,
            opcode: cmd.opcode,
        }),
    }
}

pub fn plan_from_bytes(bytes: &[u8]) -> Result<PlannedRender, Status> {
    let cmd = render::decode(bytes).map_err(|_| Status::ErrArgs)?;
    plan_render(&cmd)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::endian::{st16, st32};

    /// Opcode `0x1` is the compact `drawPrimitives:vertexStart:vertexCount:` —
    /// `alloc(1, 8)`, wire sz `0x10`. This fixture was a synthetic 24-byte
    /// record with four u32s, matching a decoder that read neither the compact
    /// nor the wide layout; the bytes below are the contract's.
    #[test]
    fn plan_draw() {
        let mut v = vec![0u8; 16];
        st32(&mut v[0..], 0x01);
        st32(&mut v[4..], 16);
        st32(&mut v[8..], 3); // primitiveType = triangle list
        st16(&mut v[12..], 0); // vertexStart
        st16(&mut v[14..], 3); // vertexCount
        match plan_from_bytes(&v).unwrap() {
            PlannedRender::Draw {
                vertex_or_index_count,
                ..
            } => assert_eq!(vertex_or_index_count, 3),
            _ => panic!("expected draw"),
        }
    }
}
