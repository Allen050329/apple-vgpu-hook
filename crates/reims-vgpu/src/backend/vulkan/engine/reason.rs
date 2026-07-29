//! Typed reasons the engine declines a request it understood.
//!
//! # Why this is not a `String`
//!
//! `AGENTS.md` requires every rejected guest command to name a `reason=<slug>`
//! at the failing site, and calls out this exact shape: *"When N distinct checks
//! share one status (`Unsupported`, `MissingBuffer`, …), each needs its own slug
//! so you can tell which fired."* With free text that rule cannot be satisfied
//! mechanically or under test — you can only read the sites and hope. Twenty-odd
//! unrelated causes were sharing `DrawError::Unsupported(String)`, several of
//! them prose that differed only in wording between neighbouring branches.
//!
//! An enum makes the rule checkable: a unit test asserts no two variants share a
//! slug, and a new decline that forgets one does not compile.
//!
//! # `Unsupported` means *this device or this engine cannot*, not *the guest is
//! wrong*
//!
//! A malformed request is a validation or preparation decline; a failed Vulkan
//! call is `DrawError::VkCall`. These are the cases where the request made sense
//! and the answer is still no — which is why several of them name a capability
//! the host GPU lacks. Those are the ones that matter on the matrix rows nobody
//! here owns.

use crate::backend::vulkan::translate::TranslateReason;
use crate::observe::Decline;

/// A request the engine understood and declined.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DrawReason {
    /// More than one viewport/scissor in a draw. Metal's multi-viewport
    /// rasterization is not modelled.
    MultiViewportArray { count: usize },
    /// A resident target bound as a sampled image must be a plain 2D image;
    /// arrayed and volume residents have no bind path.
    ResidentSampledNot2d { binding: u32 },
    /// Same for a zero-copy guest-run sampled bind.
    GuestRunSampledNot2d { binding: u32 },
    /// More MRT secondary attachments than the render pass can carry.
    SecondaryAttachmentCap { requested: usize, cap: usize },
    /// A depth test combined with MRT secondaries — the depth attachment is
    /// appended after the secondaries and the two paths have not been proven
    /// together.
    DepthWithSecondaryAttachments,
    /// The device does not advertise `samplerAnisotropy` and the guest sampler
    /// asked for it.
    SamplerAnisotropyUnsupported,
    /// The guest sampler uses `MTLSamplerAddressModeMirrorClampToEdge` and this
    /// device offers neither the Vulkan 1.2 `samplerMirrorClampToEdge` feature
    /// nor `VK_KHR_sampler_mirror_clamp_to_edge`.
    ///
    /// Binding it anyway is what this crate used to do — the translation table
    /// emitted `MIRROR_CLAMP_TO_EDGE` and nothing ever requested the feature,
    /// so the sampler was created with a mode the device had not been asked
    /// for. That is undefined behaviour a validation layer catches on someone
    /// else's GPU; declining by name is the honest answer.
    SamplerMirrorClampToEdgeUnsupported,
    /// The device declines this vertex attribute format and no portable
    /// substitute fits. Carries the translation-layer reason so the two log
    /// lines agree on why.
    VertexFormat(TranslateReason),
    /// A constant-rate vertex attribute (`divisor == 0`) on a device without
    /// `vertexAttributeInstanceRateZeroDivisor`.
    ConstantVertexAttribute,
    /// A per-instance step rate above 1 on a device without
    /// `vertexAttributeInstanceRateDivisor`.
    InstanceRateDivisorUnsupported { step_rate: u32 },
    /// A per-instance step rate above the device's `maxVertexAttribDivisor`.
    InstanceRateDivisorOverLimit { step_rate: u32, limit: u32 },
    /// No queue family supports graphics and compute together, which the
    /// engine's single-queue submit model requires.
    NoCombinedGraphicsComputeQueue,
    /// `VK_EXT_external_memory_host` is absent, so guest pages cannot become
    /// device memory. The guest-read and guest-write zero-copy rails both
    /// depend on it.
    HostPointerImportUnavailable,
    /// The extension is present but no memory type is both host-visible and
    /// importable for this pointer.
    NoImportableHostMemoryType { memory_type_bits: u32 },
    // The memory-type lookups. Each is a `memory_type_for(bits, class)` that
    // found nothing: the device advertises no memory type satisfying the buffer
    // or image's requirement bits under the class this allocation needs. That is
    // a device *capability* refusal, not a failed Vulkan call — it matters on the
    // matrix rows nobody here owns, where a class an NVIDIA host offers may be
    // absent. Named per purpose because "which allocation had nowhere to live" is
    // the diagnostic; each carries the requirement bits that matched no type.
    /// No host-visible memory type for a staging (upload) buffer.
    NoHostVisibleMemoryForStaging { memory_type_bits: u32 },
    /// No host-visible memory type for a readback buffer.
    NoHostVisibleMemoryForReadback { memory_type_bits: u32 },
    /// No host-visible memory type for the stats-reduction readback buffer.
    NoHostVisibleMemoryForStats { memory_type_bits: u32 },
    /// No device-local memory type for a storage image.
    NoDeviceLocalMemoryForStorageImage { memory_type_bits: u32 },
    /// No device-local memory type for a shared optimal-image slab.
    NoDeviceLocalMemoryForSlab { memory_type_bits: u32 },
    /// No device-local memory type for an MRT secondary attachment image.
    NoDeviceLocalMemoryForMrtSecondary { memory_type_bits: u32 },
    /// No device-local memory type for a depth attachment image.
    NoDeviceLocalMemoryForDepth { memory_type_bits: u32 },
    /// No memory type for the exportable (dmabuf) scanout image.
    NoMemoryTypeForScanoutExport { memory_type_bits: u32 },
    /// No memory type in the intersection of the image's requirements and the
    /// kernel's allowed set for an imported dmabuf. Carries that intersection.
    NoMemoryTypeForDmabufImport { memory_type_bits: u32 },
    /// dmabuf export extensions absent — the display zero-copy rail.
    DmabufExportUnavailable,
    /// Per-present export specifically.
    PresentExportUnavailable,
    /// A resident asked to be exported for present is not in guest scanout
    /// order, so exporting it would hand the display swapped channels.
    PresentExportResidentNotBgra,
    /// Present-into-guest-pages needs the host-pointer import that this device
    /// does not have.
    PresentHostPtrImportUnavailable,
    /// The import exists but the cached window for this span could not be
    /// resolved (span outside every window, or the cap would be exceeded).
    PresentHostImportResolve,
    /// The host's `map_pages` views are not stable guest-RAM aliases, so a
    /// scatter into them cannot be trusted to reach the guest.
    PresentRunsUnstable,
    /// The resident selected for GPU-direct scatter is not in guest BGRA order.
    PresentScatterResidentNotBgra,
    /// `VK_KHR_swapchain` is not enabled on the engine device.
    SwapchainUnavailable,
    /// The engine's queue family cannot present to the host window's surface.
    QueueCannotPresent { queue_family: u32 },
    /// The surface's swapchain images cannot be a transfer destination, which
    /// the present blit requires.
    SwapchainLacksTransferDst,
    /// The surface advertises no formats at all.
    SwapchainNoSurfaceFormat,
    /// The surface advertises no composite-alpha mode.
    SwapchainNoCompositeAlpha,
}

impl crate::observe::Decline for DrawReason {
    /// Stable slug for `reason=` in the always-on fail log. One per distinct
    /// check, never shared.
    fn slug(&self) -> &'static str {
        match self {
            Self::MultiViewportArray { .. } => "multi_viewport_array",
            Self::ResidentSampledNot2d { .. } => "resident_sampled_not_2d",
            Self::GuestRunSampledNot2d { .. } => "guest_run_sampled_not_2d",
            Self::SecondaryAttachmentCap { .. } => "secondary_attachment_cap",
            Self::DepthWithSecondaryAttachments => "depth_with_secondary_attachments",
            Self::SamplerAnisotropyUnsupported => "sampler_anisotropy_unsupported",
            Self::SamplerMirrorClampToEdgeUnsupported => "sampler_mirror_clamp_to_edge_unsupported",
            // Deliberately delegates: the translation layer already named the
            // exact format problem, and inventing a second slug here would make
            // the two log lines disagree about one event.
            Self::VertexFormat(reason) => reason.slug(),
            Self::ConstantVertexAttribute => "constant_vertex_attribute",
            Self::InstanceRateDivisorUnsupported { .. } => "instance_rate_divisor_unsupported",
            Self::InstanceRateDivisorOverLimit { .. } => "instance_rate_divisor_over_limit",
            Self::NoCombinedGraphicsComputeQueue => "no_combined_graphics_compute_queue",
            Self::HostPointerImportUnavailable => "host_pointer_import_unavailable",
            Self::NoImportableHostMemoryType { .. } => "no_importable_host_memory_type",
            Self::NoHostVisibleMemoryForStaging { .. } => "no_host_visible_memory_for_staging",
            Self::NoHostVisibleMemoryForReadback { .. } => "no_host_visible_memory_for_readback",
            Self::NoHostVisibleMemoryForStats { .. } => "no_host_visible_memory_for_stats",
            Self::NoDeviceLocalMemoryForStorageImage { .. } => {
                "no_device_local_memory_for_storage_image"
            }
            Self::NoDeviceLocalMemoryForSlab { .. } => "no_device_local_memory_for_slab",
            Self::NoDeviceLocalMemoryForMrtSecondary { .. } => {
                "no_device_local_memory_for_mrt_secondary"
            }
            Self::NoDeviceLocalMemoryForDepth { .. } => "no_device_local_memory_for_depth",
            Self::NoMemoryTypeForScanoutExport { .. } => "no_memory_type_for_scanout_export",
            Self::NoMemoryTypeForDmabufImport { .. } => "no_memory_type_for_dmabuf_import",
            Self::DmabufExportUnavailable => "dmabuf_export_unavailable",
            Self::PresentExportUnavailable => "present_export_unavailable",
            Self::PresentExportResidentNotBgra => "present_export_resident_not_bgra",
            Self::PresentHostPtrImportUnavailable => "present_host_ptr_import_unavailable",
            Self::PresentHostImportResolve => "host_import_resolve",
            Self::PresentRunsUnstable => "runs_unstable",
            Self::PresentScatterResidentNotBgra => "present_scatter_resident_not_bgra",
            Self::SwapchainUnavailable => "swapchain_unavailable",
            Self::QueueCannotPresent { .. } => "queue_cannot_present",
            Self::SwapchainLacksTransferDst => "swapchain_lacks_transfer_dst",
            Self::SwapchainNoSurfaceFormat => "swapchain_no_surface_format",
            Self::SwapchainNoCompositeAlpha => "swapchain_no_composite_alpha",
        }
    }
}

impl std::fmt::Display for DrawReason {
    /// `reason=<slug>` plus the fields that make the line actionable. A decline
    /// naming only its class leaves the reader without the number that caused
    /// it — which binding, which step rate, which limit.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "reason={}", self.slug())?;
        match self {
            Self::MultiViewportArray { count } => write!(f, " count={count}"),
            Self::ResidentSampledNot2d { binding } | Self::GuestRunSampledNot2d { binding } => {
                write!(f, " binding={binding}")
            }
            Self::SecondaryAttachmentCap { requested, cap } => {
                write!(f, " requested={requested} cap={cap}")
            }
            Self::VertexFormat(reason) => write!(f, " value={}", reason.value()),
            Self::InstanceRateDivisorUnsupported { step_rate } => write!(f, " rate={step_rate}"),
            Self::InstanceRateDivisorOverLimit { step_rate, limit } => {
                write!(f, " rate={step_rate} limit={limit}")
            }
            Self::NoImportableHostMemoryType { memory_type_bits }
            | Self::NoHostVisibleMemoryForStaging { memory_type_bits }
            | Self::NoHostVisibleMemoryForReadback { memory_type_bits }
            | Self::NoHostVisibleMemoryForStats { memory_type_bits }
            | Self::NoDeviceLocalMemoryForStorageImage { memory_type_bits }
            | Self::NoDeviceLocalMemoryForSlab { memory_type_bits }
            | Self::NoDeviceLocalMemoryForMrtSecondary { memory_type_bits }
            | Self::NoDeviceLocalMemoryForDepth { memory_type_bits }
            | Self::NoMemoryTypeForScanoutExport { memory_type_bits }
            | Self::NoMemoryTypeForDmabufImport { memory_type_bits } => {
                write!(f, " memory_type_bits={memory_type_bits:#x}")
            }
            Self::QueueCannotPresent { queue_family } => write!(f, " queue_family={queue_family}"),
            _ => Ok(()),
        }
    }
}

/// Why a **host-present** request violated the layout contract between the
/// guest's mapping and the frame the engine would DMA into it.
///
/// # Why this one is typed even though `Invalid` is still free text
///
/// These are the checks a *caller in `runtime/` classifies on*.
/// `runtime/import_present.rs` decided whether a failed present was a missing
/// extension, a short buffer, an absent resident or a driver error by calling
/// `e.to_string().contains(…)` on this prose — so the payload wording was load-
/// bearing behaviour that no test covered and no gate could see. One of those
/// branches was already dead: `contains("external_memory_host")` never matched,
/// because the extension check had been typed into
/// [`DrawReason::PresentHostPtrImportUnavailable`], whose slug does not contain
/// that substring. Every host-pointer present failure on a host without
/// `VK_EXT_external_memory_host` was therefore reported as a driver error.
///
/// Typing them makes the classification a `match` on a variant, which the
/// compiler forces open when a check is added, and makes the emitted
/// `reason=<slug>` the name of the check that actually fired rather than one of
/// four coarse buckets.
///
/// # Why the slugs carry their entry point
///
/// `present_into_host_ptr_strided` and `present_into_host_runs` apply the *same*
/// row-stride predicate, so a shared `bad_row_bytes` would leave a grep unable
/// to say which path refused. The `host_ptr_` / `host_runs_` prefixes are the
/// same choice the render rail made with `draw_mtl_*` / `draw_vk_*`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HostPresentDecline {
    /// The CPU-readback path's identity is not in the resident registry.
    ReadTargetUnknownIdentity,
    /// The CPU-readback path's resident has never had content written.
    ReadTargetNoReadyContent,
    /// The identity is not in the resident registry.
    HostPtrUnknownIdentity,
    /// The resident exists but has never had content written.
    HostPtrNoReadyContent,
    /// The guest's row stride is narrower than a tight BGRA row, or not a
    /// multiple of 4.
    HostPtrBadRowBytes { row_bytes: u32, tight: u32 },
    /// The guest buffer is smaller than the alignment-rounded frame the import
    /// requires.
    HostPtrShort { ptr_len: u64, import_size: u64 },
    /// No runs were offered.
    RunsEmpty,
    /// `width * 4` overflowed.
    RunsTightRowOverflow,
    /// As [`Self::HostPtrBadRowBytes`], on the fragmented path.
    RunsBadRowBytes { row_bytes: u32, tight: u32 },
    /// A run has a null pointer or a zero length.
    RunsNullOrEmpty { index: usize },
    /// A run claims more linear bytes than its host mapping holds.
    RunsLenExceedsPtr { linear_len: u64, ptr_len: u64 },
    /// `linear_base + linear_len` overflowed.
    RunsEndOverflow,
    /// The runs are not a strictly ascending, non-overlapping sequence, so the
    /// row walk below cannot rely on visiting them in order.
    RunsOutOfOrder { index: usize },
    /// `y * row_bytes` overflowed.
    RunsRowOffsetOverflow,
    /// `sample_base_off + row offset` overflowed.
    RunsSampleOffsetOverflow,
    /// `row_start + tight` overflowed.
    RunsRowEndOverflow,
    /// A computed copy span reaches past the end of its run's host mapping.
    /// Writing it would scribble on memory the guest did not offer.
    RunsScatterOob { dst_offset: u64, len: u64, cap: u64 },
    /// A scatter span names a run that was not resolved.
    RunsScatterRunIndexOutOfBounds {
        span_index: usize,
        run_index: usize,
        run_count: usize,
    },
    /// A scatter span's run exists, but no imported buffer was retained for it.
    RunsScatterBufferIndexOutOfBounds {
        span_index: usize,
        run_index: usize,
        buffer_count: usize,
    },
    /// A zero-width Vulkan copy region is invalid and cannot update a row.
    RunsScatterZeroTexels { span_index: usize },
    /// A scatter span reaches outside the resident image.
    RunsScatterSourceOutOfBounds {
        span_index: usize,
        x: u32,
        y: u32,
        texels: u32,
        width: u32,
        height: u32,
    },
    /// `dst_offset + texel_bytes` overflowed before the run-bounds check.
    RunsScatterSpanEndOverflow {
        span_index: usize,
        dst_offset: u64,
        len: u64,
    },
    /// Adding the run-relative destination to its imported-window offset
    /// overflowed.
    RunsScatterBufferOffsetOverflow {
        span_index: usize,
        window_offset: u64,
        dst_offset: u64,
    },
    /// The runs leave a hole in this row, so part of the frame has nowhere to
    /// go. Presenting the rest would publish a partially-updated frame.
    RunsUncoveredRow { row: u32 },
}

impl Decline for HostPresentDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::ReadTargetUnknownIdentity => "read_target_unknown_identity",
            Self::ReadTargetNoReadyContent => "read_target_no_ready_content",
            Self::HostPtrUnknownIdentity => "host_ptr_unknown_identity",
            Self::HostPtrNoReadyContent => "host_ptr_no_ready_content",
            Self::HostPtrBadRowBytes { .. } => "host_ptr_bad_row_bytes",
            Self::HostPtrShort { .. } => "host_ptr_short",
            Self::RunsEmpty => "host_runs_empty",
            Self::RunsTightRowOverflow => "host_runs_tight_row_overflow",
            Self::RunsBadRowBytes { .. } => "host_runs_bad_row_bytes",
            Self::RunsNullOrEmpty { .. } => "host_runs_null_or_empty",
            Self::RunsLenExceedsPtr { .. } => "host_runs_len_exceeds_ptr",
            Self::RunsEndOverflow => "host_runs_end_overflow",
            Self::RunsOutOfOrder { .. } => "host_runs_out_of_order",
            Self::RunsRowOffsetOverflow => "host_runs_row_offset_overflow",
            Self::RunsSampleOffsetOverflow => "host_runs_sample_offset_overflow",
            Self::RunsRowEndOverflow => "host_runs_row_end_overflow",
            Self::RunsScatterOob { .. } => "host_runs_scatter_oob",
            Self::RunsScatterRunIndexOutOfBounds { .. } => "host_runs_scatter_run_index_oob",
            Self::RunsScatterBufferIndexOutOfBounds { .. } => "host_runs_scatter_buffer_index_oob",
            Self::RunsScatterZeroTexels { .. } => "host_runs_scatter_zero_texels",
            Self::RunsScatterSourceOutOfBounds { .. } => "host_runs_scatter_source_oob",
            Self::RunsScatterSpanEndOverflow { .. } => "host_runs_scatter_span_end_overflow",
            Self::RunsScatterBufferOffsetOverflow { .. } => {
                "host_runs_scatter_buffer_offset_overflow"
            }
            Self::RunsUncoveredRow { .. } => "host_runs_uncovered_row",
        }
    }

    /// The same numbers [`Display`](std::fmt::Display) renders, as `k=v` pairs
    /// for [`crate::observe::Emit`]. Both exist because the emitter builds a line
    /// field-by-field while `Display` is what a `{e}` in someone else's
    /// `format!` produces; they must not disagree, which is what
    /// `the_fields_and_the_display_agree` checks.
    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::HostPtrBadRowBytes { row_bytes, tight }
            | Self::RunsBadRowBytes { row_bytes, tight } => vec![
                ("row_bytes", row_bytes.to_string()),
                ("tight", tight.to_string()),
            ],
            Self::HostPtrShort {
                ptr_len,
                import_size,
            } => vec![
                ("ptr_len", ptr_len.to_string()),
                ("import_size", import_size.to_string()),
            ],
            Self::RunsNullOrEmpty { index } | Self::RunsOutOfOrder { index } => {
                vec![("index", index.to_string())]
            }
            Self::RunsLenExceedsPtr {
                linear_len,
                ptr_len,
            } => vec![
                ("linear_len", linear_len.to_string()),
                ("ptr_len", ptr_len.to_string()),
            ],
            Self::RunsScatterOob {
                dst_offset,
                len,
                cap,
            } => vec![
                ("dst_off", dst_offset.to_string()),
                ("len", len.to_string()),
                ("cap", cap.to_string()),
            ],
            Self::RunsScatterRunIndexOutOfBounds {
                span_index,
                run_index,
                run_count,
            } => vec![
                ("span_index", span_index.to_string()),
                ("run_index", run_index.to_string()),
                ("run_count", run_count.to_string()),
            ],
            Self::RunsScatterBufferIndexOutOfBounds {
                span_index,
                run_index,
                buffer_count,
            } => vec![
                ("span_index", span_index.to_string()),
                ("run_index", run_index.to_string()),
                ("buffer_count", buffer_count.to_string()),
            ],
            Self::RunsScatterZeroTexels { span_index } => {
                vec![("span_index", span_index.to_string())]
            }
            Self::RunsScatterSourceOutOfBounds {
                span_index,
                x,
                y,
                texels,
                width,
                height,
            } => vec![
                ("span_index", span_index.to_string()),
                ("x", x.to_string()),
                ("y", y.to_string()),
                ("texels", texels.to_string()),
                ("width", width.to_string()),
                ("height", height.to_string()),
            ],
            Self::RunsScatterSpanEndOverflow {
                span_index,
                dst_offset,
                len,
            } => vec![
                ("span_index", span_index.to_string()),
                ("dst_off", dst_offset.to_string()),
                ("len", len.to_string()),
            ],
            Self::RunsScatterBufferOffsetOverflow {
                span_index,
                window_offset,
                dst_offset,
            } => vec![
                ("span_index", span_index.to_string()),
                ("window_offset", window_offset.to_string()),
                ("dst_off", dst_offset.to_string()),
            ],
            Self::RunsUncoveredRow { row } => vec![("row", row.to_string())],
            _ => Vec::new(),
        }
    }
}

impl std::fmt::Display for HostPresentDecline {
    /// `reason=<slug>` plus the numbers that made the check fail — the prose
    /// these replaced carried them, and a bounds refusal without its bound is
    /// not actionable.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "reason={}", self.slug())?;
        match self {
            Self::HostPtrBadRowBytes { row_bytes, tight }
            | Self::RunsBadRowBytes { row_bytes, tight } => {
                write!(f, " row_bytes={row_bytes} tight={tight}")
            }
            Self::HostPtrShort {
                ptr_len,
                import_size,
            } => write!(f, " ptr_len={ptr_len} import_size={import_size}"),
            Self::RunsNullOrEmpty { index } | Self::RunsOutOfOrder { index } => {
                write!(f, " index={index}")
            }
            Self::RunsLenExceedsPtr {
                linear_len,
                ptr_len,
            } => {
                write!(f, " linear_len={linear_len} ptr_len={ptr_len}")
            }
            Self::RunsScatterOob {
                dst_offset,
                len,
                cap,
            } => write!(f, " dst_off={dst_offset} len={len} cap={cap}"),
            Self::RunsScatterRunIndexOutOfBounds {
                span_index,
                run_index,
                run_count,
            } => write!(
                f,
                " span_index={span_index} run_index={run_index} run_count={run_count}"
            ),
            Self::RunsScatterBufferIndexOutOfBounds {
                span_index,
                run_index,
                buffer_count,
            } => write!(
                f,
                " span_index={span_index} run_index={run_index} buffer_count={buffer_count}"
            ),
            Self::RunsScatterZeroTexels { span_index } => {
                write!(f, " span_index={span_index}")
            }
            Self::RunsScatterSourceOutOfBounds {
                span_index,
                x,
                y,
                texels,
                width,
                height,
            } => write!(
                f,
                " span_index={span_index} x={x} y={y} texels={texels} width={width} height={height}"
            ),
            Self::RunsScatterSpanEndOverflow {
                span_index,
                dst_offset,
                len,
            } => write!(f, " span_index={span_index} dst_off={dst_offset} len={len}"),
            Self::RunsScatterBufferOffsetOverflow {
                span_index,
                window_offset,
                dst_offset,
            } => write!(
                f,
                " span_index={span_index} window_offset={window_offset} dst_off={dst_offset}"
            ),
            Self::RunsUncoveredRow { row } => write!(f, " row={row}"),
            _ => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: &[DrawReason] = &[
        DrawReason::MultiViewportArray { count: 0 },
        DrawReason::ResidentSampledNot2d { binding: 0 },
        DrawReason::GuestRunSampledNot2d { binding: 0 },
        DrawReason::SecondaryAttachmentCap {
            requested: 0,
            cap: 0,
        },
        DrawReason::DepthWithSecondaryAttachments,
        DrawReason::SamplerAnisotropyUnsupported,
        DrawReason::SamplerMirrorClampToEdgeUnsupported,
        DrawReason::ConstantVertexAttribute,
        DrawReason::InstanceRateDivisorUnsupported { step_rate: 0 },
        DrawReason::InstanceRateDivisorOverLimit {
            step_rate: 0,
            limit: 0,
        },
        DrawReason::NoCombinedGraphicsComputeQueue,
        DrawReason::HostPointerImportUnavailable,
        DrawReason::NoImportableHostMemoryType {
            memory_type_bits: 0,
        },
        DrawReason::NoHostVisibleMemoryForStaging {
            memory_type_bits: 0,
        },
        DrawReason::NoHostVisibleMemoryForReadback {
            memory_type_bits: 0,
        },
        DrawReason::NoHostVisibleMemoryForStats {
            memory_type_bits: 0,
        },
        DrawReason::NoDeviceLocalMemoryForStorageImage {
            memory_type_bits: 0,
        },
        DrawReason::NoDeviceLocalMemoryForSlab {
            memory_type_bits: 0,
        },
        DrawReason::NoDeviceLocalMemoryForMrtSecondary {
            memory_type_bits: 0,
        },
        DrawReason::NoDeviceLocalMemoryForDepth {
            memory_type_bits: 0,
        },
        DrawReason::NoMemoryTypeForScanoutExport {
            memory_type_bits: 0,
        },
        DrawReason::NoMemoryTypeForDmabufImport {
            memory_type_bits: 0,
        },
        DrawReason::DmabufExportUnavailable,
        DrawReason::PresentExportUnavailable,
        DrawReason::PresentExportResidentNotBgra,
        DrawReason::PresentHostPtrImportUnavailable,
        DrawReason::PresentHostImportResolve,
        DrawReason::PresentRunsUnstable,
        DrawReason::PresentScatterResidentNotBgra,
        DrawReason::SwapchainUnavailable,
        DrawReason::QueueCannotPresent { queue_family: 0 },
        DrawReason::SwapchainLacksTransferDst,
        DrawReason::SwapchainNoSurfaceFormat,
        DrawReason::SwapchainNoCompositeAlpha,
    ];

    /// The rule this enum exists to enforce: two checks sharing a slug means a
    /// grep of the fail log cannot tell you which one fired.
    #[test]
    fn every_reason_has_its_own_slug() {
        let mut slugs: Vec<&str> = ALL.iter().map(|r| r.slug()).collect();
        slugs.sort_unstable();
        let before = slugs.len();
        slugs.dedup();
        assert_eq!(before, slugs.len(), "duplicate DrawReason slug");
    }

    #[test]
    fn slugs_are_log_safe() {
        for r in ALL {
            let s = r.slug();
            assert!(!s.is_empty());
            assert!(
                s.bytes()
                    .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_'),
                "slug {s:?} must be lowercase snake_case"
            );
        }
    }

    /// A decline that names only its class is not actionable — the reader needs
    /// the binding, the rate, the limit.
    #[test]
    fn the_rendered_line_carries_the_load_bearing_fields() {
        assert_eq!(
            DrawReason::ResidentSampledNot2d { binding: 34 }.to_string(),
            "reason=resident_sampled_not_2d binding=34"
        );
        assert_eq!(
            DrawReason::InstanceRateDivisorOverLimit {
                step_rate: 9,
                limit: 4
            }
            .to_string(),
            "reason=instance_rate_divisor_over_limit rate=9 limit=4"
        );
        assert_eq!(
            DrawReason::SecondaryAttachmentCap {
                requested: 9,
                cap: 7
            }
            .to_string(),
            "reason=secondary_attachment_cap requested=9 cap=7"
        );
        // A field-free reason renders just its slug, with no trailing space.
        assert_eq!(
            DrawReason::SwapchainUnavailable.to_string(),
            "reason=swapchain_unavailable"
        );
    }

    /// The memory-type lookups were `DrawError::Vulkan("no host-visible memory
    /// for staging")` — free-text prose rendered as the coarse
    /// `vk_engine_vk_untyped` slug, which classified a *capability* refusal as a
    /// failed Vulkan call. They now name their purpose and carry the requirement
    /// bits that matched no memory type, the same shape as
    /// `NoImportableHostMemoryType`.
    #[test]
    fn a_memory_type_refusal_names_its_purpose_and_carries_the_bits() {
        assert_eq!(
            DrawReason::NoHostVisibleMemoryForStaging {
                memory_type_bits: 0x5
            }
            .to_string(),
            "reason=no_host_visible_memory_for_staging memory_type_bits=0x5"
        );
        assert_eq!(
            DrawReason::NoDeviceLocalMemoryForDepth {
                memory_type_bits: 0x82
            }
            .to_string(),
            "reason=no_device_local_memory_for_depth memory_type_bits=0x82"
        );
        assert_eq!(
            DrawReason::NoMemoryTypeForScanoutExport {
                memory_type_bits: 0xff
            }
            .to_string(),
            "reason=no_memory_type_for_scanout_export memory_type_bits=0xff"
        );
        // Staging, readback and stats are three purposes that all want
        // host-visible memory — a shared slug would leave a grep unable to say
        // which allocation had nowhere to live.
        assert_ne!(
            DrawReason::NoHostVisibleMemoryForStaging {
                memory_type_bits: 0
            }
            .slug(),
            DrawReason::NoHostVisibleMemoryForReadback {
                memory_type_bits: 0
            }
            .slug()
        );
        assert_ne!(
            DrawReason::NoHostVisibleMemoryForReadback {
                memory_type_bits: 0
            }
            .slug(),
            DrawReason::NoHostVisibleMemoryForStats {
                memory_type_bits: 0
            }
            .slug()
        );
    }

    #[test]
    fn slab_memory_and_scatter_format_refusals_keep_distinct_product_reasons() {
        let slab = DrawReason::NoDeviceLocalMemoryForSlab {
            memory_type_bits: 0x81,
        };
        assert_eq!(slab.slug(), "no_device_local_memory_for_slab");
        assert_eq!(
            slab.to_string(),
            "reason=no_device_local_memory_for_slab memory_type_bits=0x81"
        );

        let scatter = DrawReason::PresentScatterResidentNotBgra;
        assert_eq!(scatter.slug(), "present_scatter_resident_not_bgra");
        assert_eq!(
            scatter.to_string(),
            "reason=present_scatter_resident_not_bgra"
        );
    }

    const ALL_HOST_PRESENT: &[HostPresentDecline] = &[
        HostPresentDecline::ReadTargetUnknownIdentity,
        HostPresentDecline::ReadTargetNoReadyContent,
        HostPresentDecline::HostPtrUnknownIdentity,
        HostPresentDecline::HostPtrNoReadyContent,
        HostPresentDecline::HostPtrBadRowBytes {
            row_bytes: 0,
            tight: 0,
        },
        HostPresentDecline::HostPtrShort {
            ptr_len: 0,
            import_size: 0,
        },
        HostPresentDecline::RunsEmpty,
        HostPresentDecline::RunsTightRowOverflow,
        HostPresentDecline::RunsBadRowBytes {
            row_bytes: 0,
            tight: 0,
        },
        HostPresentDecline::RunsNullOrEmpty { index: 0 },
        HostPresentDecline::RunsLenExceedsPtr {
            linear_len: 0,
            ptr_len: 0,
        },
        HostPresentDecline::RunsEndOverflow,
        HostPresentDecline::RunsOutOfOrder { index: 0 },
        HostPresentDecline::RunsRowOffsetOverflow,
        HostPresentDecline::RunsSampleOffsetOverflow,
        HostPresentDecline::RunsRowEndOverflow,
        HostPresentDecline::RunsScatterOob {
            dst_offset: 0,
            len: 0,
            cap: 0,
        },
        HostPresentDecline::RunsScatterRunIndexOutOfBounds {
            span_index: 0,
            run_index: 1,
            run_count: 1,
        },
        HostPresentDecline::RunsScatterBufferIndexOutOfBounds {
            span_index: 0,
            run_index: 1,
            buffer_count: 1,
        },
        HostPresentDecline::RunsScatterZeroTexels { span_index: 0 },
        HostPresentDecline::RunsScatterSourceOutOfBounds {
            span_index: 0,
            x: 1,
            y: 2,
            texels: 3,
            width: 4,
            height: 5,
        },
        HostPresentDecline::RunsScatterSpanEndOverflow {
            span_index: 0,
            dst_offset: u64::MAX,
            len: 4,
        },
        HostPresentDecline::RunsScatterBufferOffsetOverflow {
            span_index: 0,
            window_offset: u64::MAX,
            dst_offset: 4,
        },
        HostPresentDecline::RunsUncoveredRow { row: 0 },
    ];

    /// Two host-present checks sharing a slug is the defect this enum replaced —
    /// the prose it grew out of had three distinct layout faults reported as one
    /// `run_gap`.
    #[test]
    fn every_host_present_decline_has_its_own_slug() {
        let mut slugs: Vec<&str> = ALL_HOST_PRESENT.iter().map(|r| r.slug()).collect();
        slugs.sort_unstable();
        let before = slugs.len();
        slugs.dedup();
        assert_eq!(before, slugs.len(), "duplicate HostPresentDecline slug");
    }

    /// The two entry points apply the *same* row-stride predicate, so the slug
    /// has to say which one refused. A shared `bad_row_bytes` would leave a grep
    /// unable to tell the packed path from the fragmented one — the same
    /// argument that split `draw_mtl_*` from `draw_vk_*`.
    #[test]
    fn the_two_present_paths_do_not_share_a_slug_for_the_same_predicate() {
        let packed = HostPresentDecline::HostPtrBadRowBytes {
            row_bytes: 7,
            tight: 8,
        };
        let fragmented = HostPresentDecline::RunsBadRowBytes {
            row_bytes: 7,
            tight: 8,
        };
        assert_ne!(packed.slug(), fragmented.slug());
        assert!(packed.slug().starts_with("host_ptr_"));
        assert!(fragmented.slug().starts_with("host_runs_"));
        // …and both still report the numbers that made them fire.
        assert_eq!(packed.fields(), fragmented.fields());
    }

    /// `Display` and `fields()` render the same numbers. They are two renderers
    /// for one decline — a `{e}` in someone else's `format!` versus a line the
    /// emitter builds — and a reader who greps one must not get less than the
    /// other.
    #[test]
    fn the_fields_and_the_display_agree() {
        for r in ALL_HOST_PRESENT {
            let shown = r.to_string();
            assert!(
                shown.starts_with(&format!("reason={}", r.slug())),
                "{shown}"
            );
            for (k, v) in r.fields() {
                assert!(
                    shown.contains(&format!("{k}={v}")),
                    "Display for {:?} omits the field {k}={v} that fields() reports: {shown}",
                    r
                );
            }
            assert_eq!(
                shown.split(' ').count() - 1,
                r.fields().len(),
                "Display for {r:?} renders a different number of fields than fields(): {shown}"
            );
        }
    }

    /// A bounds refusal without its bound is half a diagnostic; these are the
    /// exact numbers the prose carried before it was typed.
    #[test]
    fn a_layout_refusal_carries_the_numbers_that_caused_it() {
        assert_eq!(
            HostPresentDecline::HostPtrShort {
                ptr_len: 4096,
                import_size: 8_294_400
            }
            .to_string(),
            "reason=host_ptr_short ptr_len=4096 import_size=8294400"
        );
        assert_eq!(
            HostPresentDecline::RunsScatterOob {
                dst_offset: 16,
                len: 32,
                cap: 40
            }
            .to_string(),
            "reason=host_runs_scatter_oob dst_off=16 len=32 cap=40"
        );
        assert_eq!(
            HostPresentDecline::RunsEmpty.to_string(),
            "reason=host_runs_empty"
        );
    }

    /// The GPU scatter preflight is a memory-safety boundary: each refusal must
    /// retain both which span failed and the exact bound that rejected it.
    /// Otherwise a bad run index, a missing imported buffer, and arithmetic
    /// overflow all collapse into the old un-actionable "scatter declined".
    #[test]
    fn scatter_span_refusals_preserve_the_failed_index_and_bound() {
        let cases = [
            (
                HostPresentDecline::RunsScatterRunIndexOutOfBounds {
                    span_index: 3,
                    run_index: 5,
                    run_count: 2,
                },
                "reason=host_runs_scatter_run_index_oob \
                 span_index=3 run_index=5 run_count=2",
            ),
            (
                HostPresentDecline::RunsScatterBufferIndexOutOfBounds {
                    span_index: 4,
                    run_index: 1,
                    buffer_count: 1,
                },
                "reason=host_runs_scatter_buffer_index_oob \
                 span_index=4 run_index=1 buffer_count=1",
            ),
            (
                HostPresentDecline::RunsScatterZeroTexels { span_index: 6 },
                "reason=host_runs_scatter_zero_texels span_index=6",
            ),
            (
                HostPresentDecline::RunsScatterSourceOutOfBounds {
                    span_index: 7,
                    x: 61,
                    y: 31,
                    texels: 4,
                    width: 64,
                    height: 32,
                },
                "reason=host_runs_scatter_source_oob \
                 span_index=7 x=61 y=31 texels=4 width=64 height=32",
            ),
            (
                HostPresentDecline::RunsScatterSpanEndOverflow {
                    span_index: 8,
                    dst_offset: u64::MAX,
                    len: 16,
                },
                "reason=host_runs_scatter_span_end_overflow \
                 span_index=8 dst_off=18446744073709551615 len=16",
            ),
            (
                HostPresentDecline::RunsScatterBufferOffsetOverflow {
                    span_index: 9,
                    window_offset: u64::MAX - 7,
                    dst_offset: 8,
                },
                "reason=host_runs_scatter_buffer_offset_overflow \
                 span_index=9 window_offset=18446744073709551608 dst_off=8",
            ),
        ];

        for (decline, expected) in cases {
            assert_eq!(decline.to_string(), expected, "{decline:?}");
            let line = crate::observe::Emit::decline("scatter_preflight", &decline).render();
            assert_eq!(line, format!("scatter_preflight {expected}"));
        }
    }

    /// A vertex-format decline reports the translation layer's own slug rather
    /// than minting a second name for one event.
    #[test]
    fn a_vertex_format_decline_reuses_the_translation_reason() {
        let translate = TranslateReason::FormatNotVertexBuffer(97);
        let reason = DrawReason::VertexFormat(translate);
        assert_eq!(reason.slug(), translate.slug());
        assert_eq!(
            reason.to_string(),
            "reason=format_not_vertex_buffer value=97"
        );
    }
}
