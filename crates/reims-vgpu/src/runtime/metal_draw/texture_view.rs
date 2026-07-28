/// Type-8 view resolution for sample/seed paths.
#[derive(Clone, Debug)]
pub(crate) struct ViewResolve {
    /// Non-view base texture ref after walking the view chain (archive
    /// `REIMS_VGPU_RESOURCE_RESOLVE_MAX_VIEW_CHAIN` walk).
    pub(crate) base_texture_ref: u32,
    pub(crate) level: u32,
    /// Present when the view carries a swizzle form (opcode 0x1b); selectors already validated.
    pub(crate) swizzle: Option<pixel_format::SwizzlePlan>,
    /// Non-zero view pixel format from the descriptor (`@16`); `None` inherits the base format.
    pub(crate) pixel_format: Option<u16>,
}

/// A specific refusal while resolving one type-8 texture-view chain.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum TextureViewDecline {
    HopEntryMissing {
        texture_ref: u32,
    },
    HopObjectNotView {
        texture_ref: u32,
        object_type: u8,
    },
    HopDescriptorMissing {
        texture_ref: u32,
        descriptor_length: u32,
    },
    HopDecode {
        texture_ref: u32,
        opcode: u32,
        declared: u32,
        descriptor_len: usize,
        bytes_hex: String,
        reason: DecodeStatus,
    },
    HopZeroBase {
        texture_ref: u32,
        opcode: u32,
    },
    HopLevelOverflow {
        texture_ref: u32,
        level_base: u64,
    },
    HopSwizzleInvalid {
        texture_ref: u32,
        selectors: [u8; 4],
    },
    ChainSelfOrZero {
        base: u32,
        next: u32,
        depth: u32,
    },
    ChainOverflow {
        base: u32,
        depth: u32,
    },
}

impl Decline for TextureViewDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::HopEntryMissing { .. } => "texture_view_hop_entry_missing",
            Self::HopObjectNotView { .. } => "texture_view_hop_object_not_view",
            Self::HopDescriptorMissing { .. } => "texture_view_hop_descriptor_missing",
            // Keep the descriptor decoder's exact registered reason primary.
            Self::HopDecode { reason, .. } => reason.slug(),
            Self::HopZeroBase { .. } => "texture_view_hop_zero_base",
            Self::HopLevelOverflow { .. } => "texture_view_hop_level_overflow",
            Self::HopSwizzleInvalid { .. } => "texture_view_hop_swizzle_invalid",
            Self::ChainSelfOrZero { .. } => "texture_view_chain_self_or_zero",
            Self::ChainOverflow { .. } => "texture_view_chain_overflow",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::HopEntryMissing { texture_ref } => {
                vec![("texture_ref", texture_ref.to_string())]
            }
            Self::HopObjectNotView {
                texture_ref,
                object_type,
            } => vec![
                ("texture_ref", texture_ref.to_string()),
                ("object_type", object_type.to_string()),
            ],
            Self::HopDescriptorMissing {
                texture_ref,
                descriptor_length,
            } => vec![
                ("texture_ref", texture_ref.to_string()),
                ("descriptor_length", descriptor_length.to_string()),
            ],
            Self::HopDecode {
                texture_ref,
                opcode,
                declared,
                descriptor_len,
                bytes_hex,
                reason,
            } => {
                let mut fields = vec![
                    ("texture_ref", texture_ref.to_string()),
                    ("opcode", format!("{opcode:#x}")),
                    ("declared", declared.to_string()),
                    ("descriptor_len", descriptor_len.to_string()),
                    ("bytes", bytes_hex.clone()),
                ];
                fields.extend(reason.fields());
                fields
            }
            Self::HopZeroBase {
                texture_ref,
                opcode,
            } => vec![
                ("texture_ref", texture_ref.to_string()),
                ("opcode", format!("{opcode:#x}")),
            ],
            Self::HopLevelOverflow {
                texture_ref,
                level_base,
            } => vec![
                ("texture_ref", texture_ref.to_string()),
                ("level_base", level_base.to_string()),
            ],
            Self::HopSwizzleInvalid {
                texture_ref,
                selectors,
            } => vec![
                ("texture_ref", texture_ref.to_string()),
                (
                    "selectors",
                    selectors
                        .iter()
                        .map(u8::to_string)
                        .collect::<Vec<_>>()
                        .join(","),
                ),
            ],
            Self::ChainSelfOrZero { base, next, depth } => vec![
                ("base", base.to_string()),
                ("next", next.to_string()),
                ("depth", depth.to_string()),
            ],
            Self::ChainOverflow { base, depth } => {
                vec![("base", base.to_string()), ("depth", depth.to_string())]
            }
        }
    }
}

impl std::fmt::Display for TextureViewDecline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "reason={}", self.slug())?;
        for (key, value) in self.fields() {
            write!(f, " {key}={value}")?;
        }
        Ok(())
    }
}

impl std::error::Error for TextureViewDecline {}

/// Archive `REIMS_VGPU_RESOURCE_RESOLVE_MAX_VIEW_CHAIN` — nested type-8 views collapse
/// to a non-view base (`apple_pv_gpu_resource_resolve_texture` chain walk).
const MAX_TEXTURE_VIEW_CHAIN: usize = 8;

/// Decode one type-8 hop (does not walk nested bases).
///
/// The `Result` carries a specific failure slug for the always-on fail log; the
/// [`decode_texture_view_hop`] wrapper collapses it to `Option` for the hot path.
fn decode_texture_view_hop_reasoned<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    texture_ref: u32,
) -> Result<(u32, u32, Option<pixel_format::SwizzlePlan>, Option<u16>), TextureViewDecline> {
    use crate::contract::endian::ld32;
    use crate::runtime::decode::resource::{
        decode_texture_view_descriptor, OBJECT_TYPE_TEXTURE_VIEW, TEXTURE_VIEW_DESC_LEN,
        TEXTURE_VIEW_DESC_OPCODE, TEXTURE_VIEW_MIN_SIMPLE,
    };
    let entry = objects::lookup_list_entry(state, host, task_id, texture_ref)
        .ok_or(TextureViewDecline::HopEntryMissing { texture_ref })?;
    if entry.object_type != OBJECT_TYPE_TEXTURE_VIEW {
        return Err(TextureViewDecline::HopObjectNotView {
            texture_ref,
            object_type: entry.object_type,
        });
    }
    let desc = objects::read_descriptor(state, host, task_id, &entry).ok_or(
        TextureViewDecline::HopDescriptorMissing {
            texture_ref,
            descriptor_length: entry.descriptor_length,
        },
    )?;
    // Bytes visible before decode, for the len-mismatch / bad-opcode census.
    let (opcode, declared) = if desc.len() >= TEXTURE_VIEW_MIN_SIMPLE {
        (
            ld32(&desc[TEXTURE_VIEW_DESC_OPCODE..]),
            ld32(&desc[TEXTURE_VIEW_DESC_LEN..]),
        )
    } else {
        (0, 0)
    };
    let view = decode_texture_view_descriptor(&desc).map_err(|reason| {
        // Dump the full wire blob for an unknown texture-view opcode: this is the
        // only signal that reveals a new serializer variant (off the hot path —
        // fires only on a genuine decode failure).
        let hex: String = desc.iter().map(|b| format!("{b:02x}")).collect();
        TextureViewDecline::HopDecode {
            texture_ref,
            opcode,
            declared,
            descriptor_len: desc.len(),
            bytes_hex: hex,
            reason,
        }
    })?;
    if view.base_texture_ref == 0 {
        return Err(TextureViewDecline::HopZeroBase {
            texture_ref,
            opcode,
        });
    }
    let level = if view.has_levels {
        // level_base is a mip index (u64 on wire); reject pathological values.
        if view.level_base > u32::MAX as u64 {
            return Err(TextureViewDecline::HopLevelOverflow {
                texture_ref,
                level_base: view.level_base,
            });
        }
        view.level_base as u32
    } else {
        0
    };
    let swizzle = if view.has_swizzle {
        // Malformed selectors (not in 0..5) fail the resolve — visible soft miss on sample.
        Some(pixel_format::swizzle_plan(&view.swizzle).ok_or(
            TextureViewDecline::HopSwizzleInvalid {
                texture_ref,
                selectors: view.swizzle,
            },
        )?)
    } else {
        None
    };
    // Zero pixel_format means inherit base (serializer always writes a real format when set).
    let pixel_format = if view.pixel_format != 0 {
        Some(view.pixel_format)
    } else {
        None
    };
    Ok((view.base_texture_ref, level, swizzle, pixel_format))
}

/// Resolve type-8 view to non-view base + mip + format override + swizzle.
///
/// The `Result` carries a specific failure slug (`reason=view_resolve` sub-case)
/// for the always-on fail log; [`resolve_texture_view`] collapses it to `Option`
/// for the hot path. Walks nested type-8 bases up to [`MAX_TEXTURE_VIEW_CHAIN`]
/// (archive `apple_pv_gpu_resource_resolve_texture` chain). Outer-most view
/// supplies level / format / swizzle (inner hops only extend the base ref),
/// matching the product RT path which materializes a single selected level.
pub(crate) fn resolve_texture_view_reasoned<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    texture_ref: u32,
) -> Result<ViewResolve, TextureViewDecline> {
    use crate::runtime::decode::resource::OBJECT_TYPE_TEXTURE_VIEW;

    let (mut base, level, swizzle, pixel_format) =
        decode_texture_view_hop_reasoned(state, host, task_id, texture_ref)?;

    // Collapse nested type-8 bases to a non-view texture (type-11 / type-2/3).
    let mut depth = 0u32;
    for _ in 1..MAX_TEXTURE_VIEW_CHAIN {
        let Some(entry) = objects::lookup_list_entry(state, host, task_id, base) else {
            // Base missing from the list — leave the ref for the caller to fail
            // visibly (same as a one-hop miss on an unmapped base).
            break;
        };
        if entry.object_type != OBJECT_TYPE_TEXTURE_VIEW {
            break;
        }
        depth += 1;
        let (next, _lvl, _sw, _fmt) = decode_texture_view_hop_reasoned(state, host, task_id, base)?;
        if next == 0 || next == base {
            return Err(TextureViewDecline::ChainSelfOrZero { base, next, depth });
        }
        base = next;
    }

    // Final base must not still be a type-8 view past the chain cap.
    if let Some(entry) = objects::lookup_list_entry(state, host, task_id, base) {
        if entry.object_type == OBJECT_TYPE_TEXTURE_VIEW {
            return Err(TextureViewDecline::ChainOverflow { base, depth });
        }
    }

    Ok(ViewResolve {
        base_texture_ref: base,
        level,
        swizzle,
        pixel_format,
    })
}

/// Resolve type-8 view to non-view base + mip + format override + swizzle.
///
/// Returns `None` if the ref is not a type-8 view, a hop is short/unsupported,
/// the chain exceeds the max depth without a non-view base, a base ref is zero,
/// or swizzle selectors are malformed. See [`resolve_texture_view_reasoned`] for
/// the specific reason on the fail path.
fn resolve_texture_view<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    texture_ref: u32,
) -> Option<ViewResolve> {
    resolve_texture_view_reasoned(state, host, task_id, texture_ref).ok()
}

/// Pick the sample format for a type-8 view over base storage.
///
/// Metal texture views require the view format to be bpp-compatible with the base.
/// Unknown formats (no `bytes_per_pixel`) fail visibly. `None` override inherits base.
pub(crate) fn effective_view_sample_format(base_fmt: u16, view_fmt: Option<u16>) -> Option<u16> {
    let sample = view_fmt.unwrap_or(base_fmt);
    let base_bpp = pixel_format::bytes_per_pixel(base_fmt)?;
    let sample_bpp = pixel_format::bytes_per_pixel(sample)?;
    if base_bpp != sample_bpp {
        return None;
    }
    Some(sample)
}

/// Apply a type-8 view swizzle by rewriting tight RGBA8 texels. Identity plans
/// are no-ops. Returns `None` only if the buffer length is not a multiple of 4
/// (corrupt load).
///
/// **This is the slow way and it is counted.** Vulkan performs the same remap
/// for free on the image view, so the Vulkan pathway uses
/// `SampledImageResource::swizzle` and never calls this; every invocation here
/// is a texture that gave up its zero-copy crossing to be remapped by hand. The
/// Metal-direct pathway still needs it, so it reports itself rather than being
/// deleted.
#[cfg(any(test, all(feature = "backend-metal", target_os = "macos")))]
fn apply_view_swizzle_rgba8(
    rgba: &mut [u8],
    plan: Option<&pixel_format::SwizzlePlan>,
    texture_ref: u32,
) -> Option<()> {
    let Some(plan) = plan else {
        return Some(());
    };
    if pixel_format::swizzle_is_identity(plan) {
        return Some(());
    }
    if !rgba.len().is_multiple_of(4) {
        return None;
    }
    crate::runtime::census::view_swizzle_census::note_cpu_remap(texture_ref);
    for px in rgba.chunks_exact_mut(4) {
        let input = [px[0], px[1], px[2], px[3]];
        let out = pixel_format::apply_swizzle_rgba8(plan, input);
        px.copy_from_slice(&out);
    }
    Some(())
}

fn load_linear_texture_rgba_host<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    texture_ref: u32,
    level: u32,
    format_override: Option<u16>,
) -> Option<Vec<u8>> {
    load_linear_texture_impl(
        state,
        host,
        task_id,
        texture_ref,
        level,
        format_override,
        false,
    )
    .map(|(bytes, _)| bytes)
}

/// Like [`load_linear_texture_rgba_host`] but keeps a BGRA8 source in its
/// native channel order (returned format [`TexelLayout::Bgra8`]) so
/// the engine uploads it into a BGRA8 image — the sampler swizzles in hardware
/// and the CPU never runs the per-pixel channel swap. Used by the Safari-scroll
/// fallback hot path (`lin_guest_fb`), which is padded-stride BGRA8 glyph/tile
/// textures. Non-BGRA8 sources still report `Rgba8` (converted as before).
fn load_linear_texture_native_host<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    texture_ref: u32,
    level: u32,
    format_override: Option<u16>,
) -> Option<(Vec<u8>, TexelLayout)> {
    load_linear_texture_impl(
        state,
        host,
        task_id,
        texture_ref,
        level,
        format_override,
        true,
    )
}

/// When a sampled format's guest bytes are ALREADY in the final upload order —
/// so the loader can read padded source rows straight into the tight output
/// with no intermediate buffer and no per-row convert — this returns the engine
/// upload format. `RGBA8` always qualifies (its convert is an identity copy);
/// `BGRA8` qualifies only when the caller opts into a native BGRA8 upload
/// (`native_bgra8`), otherwise it must be swapped to RGBA8. Every other format
/// needs a real convert pass and returns `None`.
fn linear_native_upload_format(sample_format: u16, native_bgra8: bool) -> Option<TexelLayout> {
    use pixel_format::SampledClass;
    // The decode contract's sampled class is the one rule for "which 8-bit
    // channel order is this"; it folds each sRGB format onto its linear
    // sibling's layout, which is right — they share a layout — but loses the
    // qualifier, so the census records what the fold cost.
    let upload = match pixel_format::sampled_class(sample_format)? {
        SampledClass::Rgba8Unorm => TexelLayout::Rgba8,
        SampledClass::Bgra8Unorm if native_bgra8 => TexelLayout::Bgra8,
        _ => return None,
    };
    note_srgb_upload_downgrade(srgb_census::site::LINEAR_NATIVE_UPLOAD, sample_format);
    Some(upload)
}

/// Record an sRGB downgrade on a byte-layout rail, if this format had a
/// qualifier to lose. One helper so the two CPU upload paths cannot drift on
/// when they report.
fn note_srgb_upload_downgrade(site: &'static str, sample_format: u16) {
    if pixel_format::is_srgb(sample_format) {
        srgb_census::note_downgrade(site, sample_format);
    }
}

fn load_linear_texture_impl<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    texture_ref: u32,
    level: u32,
    format_override: Option<u16>,
    native_bgra8: bool,
) -> Option<(Vec<u8>, TexelLayout)> {
    let entry = objects::lookup_list_entry(state, host, task_id, texture_ref)?;
    if entry.object_type != OBJECT_TYPE_TEXTURE && entry.object_type != OBJECT_TYPE_TEXTURE_VARIANT
    {
        return None;
    }
    let desc_bytes = objects::read_descriptor(state, host, task_id, &entry)?;
    let tex = decode_texture_descriptor(&desc_bytes).ok()?;
    if !tex.has_pixel_format {
        return None;
    }
    let base_fmt = tex.pixel_format;
    let sample_fmt = effective_view_sample_format(base_fmt, format_override)?;
    let (gva, layout) = tex.level_gva(level, state.page_shift)?;
    let w = layout.width;
    let h = layout.height;
    let bpr = layout.row_stride;
    if bpr > u32::MAX as u64 {
        return None;
    }
    let bpr_u32 = bpr as u32;
    let tight = pixel_format::tight_row_bytes(w, base_fmt)?;
    if bpr_u32 < tight || w == 0 || h == 0 {
        return None;
    }
    let need_rgba = (w as u64)
        .checked_mul(h as u64)?
        .checked_mul(RGBA8_BPP as u64)?;
    let need_rgba = host_alloc_len(need_rgba)?;
    let span = bpr.checked_mul(h as u64)?;
    if tex.allocation_size != 0 && layout.offset.saturating_add(span) > tex.allocation_size {
        return None;
    }
    // Deferred-writeback flush-on-access: the reads below walk raw task GVAs
    // and bypass the mapping-keyed hooks — land any resident-authoritative
    // window whose physical pages alias the sampled span first.
    crate::runtime::storage_flush::flush_intersecting_task_gva(state, host, task_id, gva, span);
    // Tight display textures are the common compositor source. Read the whole
    // image with one task-root/cache lifetime: the row loop below otherwise
    // rebuilds the GVA walker cache once per row (1,080 times for the live
    // Safari source). Padded rows retain the conservative disjoint reads so we
    // never touch padding that the guest did not make readable.
    if bpr_u32 == tight {
        let started = Instant::now();
        let (rgba, fmt) = load_tight_linear_rgba_with(w, h, sample_fmt, native_bgra8, |native| {
            gva_mem::read_task_gva_by_id(
                host,
                &state.tasks,
                task_id,
                gva,
                native,
                state.page_shift,
            )
            .is_ok()
        })?;
        if w >= 1280 && h >= 720 {
            log_linear_sample_read(
                task_id,
                texture_ref,
                gva,
                w,
                h,
                bpr,
                "bulk_tight",
                1,
                started.elapsed().as_micros() as u64,
            );
        }
        return Some((rgba, fmt));
    }
    // Padded rows. When the source bytes are already in the final upload order
    // (RGBA8 always; BGRA8 under a native upload) AND the guest rows are 4-byte
    // tight, read each padded source row STRAIGHT into the tight output — no
    // intermediate row buffer, no per-row convert/swizzle pass. This is the
    // Safari-scroll fallback hot path (`lin_guest_fb`), so the elided convert
    // pass is a full second walk over the sampled bytes off the drain worker.
    let tight_4bpp = tight as u64 == (w as u64).checked_mul(RGBA8_BPP as u64)?;
    if let Some(fmt) = linear_native_upload_format(sample_fmt, native_bgra8).filter(|_| tight_4bpp)
    {
        let row_bytes = tight as usize;
        let mut rgba = vec![0u8; need_rgba];
        let started = Instant::now();
        for y in 0..h {
            let row_gva = gva.checked_add((y as u64).checked_mul(bpr)?)?;
            let dst_off = (y as usize).checked_mul(row_bytes)?;
            gva_mem::read_task_gva_by_id(
                host,
                &state.tasks,
                task_id,
                row_gva,
                rgba.get_mut(dst_off..dst_off + row_bytes)?,
                state.page_shift,
            )
            .ok()?;
        }
        if w >= 1280 && h >= 720 {
            log_linear_sample_read(
                task_id,
                texture_ref,
                gva,
                w,
                h,
                bpr,
                "row_padded_native",
                h,
                started.elapsed().as_micros() as u64,
            );
        }
        return Some((rgba, fmt));
    }
    let mut rgba = vec![0u8; need_rgba];
    let mut row = vec![0u8; tight as usize];
    let started = Instant::now();
    for y in 0..h {
        let row_gva = gva.checked_add((y as u64).checked_mul(bpr)?)?;
        gva_mem::read_task_gva_by_id(
            host,
            &state.tasks,
            task_id,
            row_gva,
            &mut row,
            state.page_shift,
        )
        .ok()?;
        let dst_off = (y as usize) * (w as usize) * 4;
        if !pixel_format::convert_row_to_rgba8(sample_fmt, &row, w, &mut rgba[dst_off..]) {
            return None;
        }
    }
    if w >= 1280 && h >= 720 {
        log_linear_sample_read(
            task_id,
            texture_ref,
            gva,
            w,
            h,
            bpr,
            "row_padded",
            h,
            started.elapsed().as_micros() as u64,
        );
    }
    Some((rgba, TexelLayout::Rgba8))
}

fn load_tight_linear_rgba_with<F>(
    width: u32,
    height: u32,
    sample_format: u16,
    native_bgra8: bool,
    mut read: F,
) -> Option<(Vec<u8>, TexelLayout)>
where
    F: FnMut(&mut [u8]) -> bool,
{
    let tight = pixel_format::tight_row_bytes(width, sample_format)?;
    let native_len = (tight as u64)
        .checked_mul(height as u64)
        .and_then(host_alloc_len)?;
    let rgba_stride = width.checked_mul(RGBA8_BPP)?;
    let rgba_len = (rgba_stride as u64)
        .checked_mul(height as u64)
        .and_then(host_alloc_len)?;
    let mut native = vec![0u8; native_len];
    if !read(&mut native) {
        return None;
    }
    // The compositor's common BGRA8/RGBA8 sources already have the output
    // allocation size. Convert them in place so the bulk page walk does not
    // add a second display-sized allocation and copy.
    if native_len == rgba_len {
        // Same single rule as `linear_native_upload_format`: the contract's
        // sampled class names the channel order, and an sRGB source is reported
        // to the census rather than folded away unnoticed.
        match pixel_format::sampled_class(sample_format) {
            Some(pixel_format::SampledClass::Rgba8Unorm) => {
                note_srgb_upload_downgrade(srgb_census::site::TIGHT_LINEAR_LOAD, sample_format);
                return Some((native, TexelLayout::Rgba8));
            }
            Some(pixel_format::SampledClass::Bgra8Unorm) => {
                note_srgb_upload_downgrade(srgb_census::site::TIGHT_LINEAR_LOAD, sample_format);
                if native_bgra8 {
                    // Upload the guest's native BGRA8 order; the engine binds a
                    // BGRA8 image and the sampler swizzles in hardware. Elides
                    // the full-image CPU channel-swap pass over the read bytes.
                    return Some((native, TexelLayout::Bgra8));
                }
                for pixel in native.chunks_exact_mut(RGBA8_BPP as usize) {
                    pixel.swap(0, 2);
                }
                return Some((native, TexelLayout::Rgba8));
            }
            _ => {}
        }
    }
    let mut rgba = vec![0u8; rgba_len];
    for y in 0..height as usize {
        let src_off = y.checked_mul(tight as usize)?;
        let dst_off = y.checked_mul(rgba_stride as usize)?;
        if !pixel_format::convert_row_to_rgba8(
            sample_format,
            &native[src_off..src_off + tight as usize],
            width,
            &mut rgba[dst_off..dst_off + rgba_stride as usize],
        ) {
            return None;
        }
    }
    Some((rgba, TexelLayout::Rgba8))
}

#[allow(clippy::too_many_arguments)]
fn log_linear_sample_read(
    task_id: u32,
    texture_ref: u32,
    gva: u64,
    width: u32,
    height: u32,
    row_stride: u64,
    mode: &str,
    calls: u32,
    total_us: u64,
) {
    crate::observe::off(format!(
        "linear_sample_read mode={mode} task={task_id} ref={texture_ref} gva={gva:#x} {width}x{height} bpr={row_stride} calls={calls} total_us={total_us}"
    ));
}

#[cfg(test)]
mod texture_view_split_tests {
    use super::*;

    #[test]
    fn view_pixel_format_override_effective() {
        use crate::contract::pixel_format::{
            MTL_FORMAT_BGRA8_UNORM, MTL_FORMAT_R8_UNORM, MTL_FORMAT_RGBA16_FLOAT,
            MTL_FORMAT_RGBA8_UNORM,
        };
        assert_eq!(
            effective_view_sample_format(MTL_FORMAT_BGRA8_UNORM, None),
            Some(MTL_FORMAT_BGRA8_UNORM)
        );
        assert_eq!(
            effective_view_sample_format(MTL_FORMAT_BGRA8_UNORM, Some(MTL_FORMAT_RGBA8_UNORM)),
            Some(MTL_FORMAT_RGBA8_UNORM)
        );
        assert!(
            effective_view_sample_format(MTL_FORMAT_BGRA8_UNORM, Some(MTL_FORMAT_R8_UNORM))
                .is_none()
        );
        assert!(effective_view_sample_format(
            MTL_FORMAT_BGRA8_UNORM,
            Some(MTL_FORMAT_RGBA16_FLOAT)
        )
        .is_none());
        assert!(effective_view_sample_format(0, Some(MTL_FORMAT_RGBA8_UNORM)).is_none());
    }
}

#[allow(dead_code)]
fn _ld32_keep(v: &[u8]) -> u32 {
    ld32(v)
}
