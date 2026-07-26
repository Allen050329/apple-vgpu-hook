//! Compute planning (port of `host/utils/reims-vgpu-compute-plan`).

use crate::runtime::decode::compute::{self, Command as ComputeCommand, Kind};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Ok,
    ErrArgs,
    ErrUnsupported,
    ErrMissingPipeline,
    ErrZeroGrid,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlannedCompute {
    SetPipeline {
        pipeline_ref: u32,
    },
    BindBuffers {
        first: u32,
        refs: Vec<u32>,
    },
    BindTextures {
        first: u32,
        refs: Vec<u32>,
    },
    Dispatch {
        threads: bool,
        grid_x: u64,
        grid_y: u64,
        grid_z: u64,
        tpt_x: u64,
        tpt_y: u64,
        tpt_z: u64,
    },
    DispatchIndirect {
        buffer_ref: u32,
        offset: u64,
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

pub fn plan_compute(cmd: &ComputeCommand) -> Result<PlannedCompute, Status> {
    match cmd.kind {
        Kind::Pipeline => {
            if cmd.pipeline_ref == 0 {
                return Err(Status::ErrMissingPipeline);
            }
            Ok(PlannedCompute::SetPipeline {
                pipeline_ref: cmd.pipeline_ref,
            })
        }
        Kind::BufferBind | Kind::BufferBindAttributeStride => Ok(PlannedCompute::BindBuffers {
            first: cmd.first,
            refs: cmd.buffers.iter().map(|b| b.ref_).collect(),
        }),
        Kind::TextureBind => Ok(PlannedCompute::BindTextures {
            first: cmd.first,
            refs: cmd.textures.iter().map(|t| t.ref_).collect(),
        }),
        Kind::DispatchThreadgroups | Kind::DispatchThreads => {
            if cmd.grid.x == 0 || cmd.grid.y == 0 || cmd.grid.z == 0 {
                return Err(Status::ErrZeroGrid);
            }
            Ok(PlannedCompute::Dispatch {
                threads: matches!(cmd.kind, Kind::DispatchThreads),
                grid_x: cmd.grid.x,
                grid_y: cmd.grid.y,
                grid_z: cmd.grid.z,
                tpt_x: cmd.threads_per_threadgroup.x,
                tpt_y: cmd.threads_per_threadgroup.y,
                tpt_z: cmd.threads_per_threadgroup.z,
            })
        }
        Kind::DispatchThreadgroupsIndirect | Kind::DispatchThreadsIndirect => {
            Ok(PlannedCompute::DispatchIndirect {
                buffer_ref: cmd.indirect_buffer_ref,
                offset: cmd.indirect_buffer_offset,
            })
        }
        Kind::UpdateFence => Ok(PlannedCompute::Fence {
            wait: false,
            fence_ref: cmd.fence_ref,
        }),
        Kind::WaitFence => Ok(PlannedCompute::Fence {
            wait: true,
            fence_ref: cmd.fence_ref,
        }),
        Kind::Unknown => Err(Status::ErrUnsupported),
        other => Ok(PlannedCompute::Other {
            kind: other,
            opcode: cmd.opcode,
        }),
    }
}

pub fn plan_from_bytes(bytes: &[u8]) -> Result<PlannedCompute, Status> {
    let cmd = compute::decode(bytes).map_err(|_| Status::ErrArgs)?;
    plan_compute(&cmd)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::endian::st32;

    #[test]
    fn plan_pipeline() {
        let mut v = vec![0u8; 12];
        st32(&mut v[0..], 0xd0);
        st32(&mut v[4..], 12);
        st32(&mut v[8..], 7);
        match plan_from_bytes(&v).unwrap() {
            PlannedCompute::SetPipeline { pipeline_ref } => assert_eq!(pipeline_ref, 7),
            _ => panic!("expected pipeline"),
        }
    }
}
