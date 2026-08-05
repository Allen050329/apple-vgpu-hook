//! Which host or guest bytes a colour attachment's `texture_ref` names.
//!
//! Every render pass this device encodes begins by turning the guest's
//! `texture_ref` into somewhere to render: a host mapping id, or a linear guest
//! VA plus a row stride. [`lookup_render_target`] is the only implementation of
//! that question, and its three callers — the single-target request builder, the
//! MRT one, and the abandoned-chain writeback — all treat a refusal as the whole
//! pass being lost, because Metal will not form an encoder with a null colour
//! attachment.
//!
//! # The rungs, in the order the archive tries them
//!
//! 1. **Type-8 view → base.** A view resolves to the texture it wraps, and
//!    carries a format override and a mip level forward with it. A swizzled view
//!    is refused rather than resolved.
//! 2. **Type-11 IOSurface.** Geometry comes from the live mapping, never from a
//!    sticky latch — the latch exists only for the window where the object-list
//!    entry is transiently missing, and preferring it has twice routed a
//!    dual-mapping composite onto one mapping.
//! 3. **Type-4 surface / type-5 `RefTextureHandle`.** The object-list index is
//!    the surface id; type-5 wraps type-4 and is what product colour targets
//!    actually bind.
//! 4. **Type-2/3 linear guest VA.** Wallpaper and background intermediates and
//!    UI intermediate render targets live here, so a type-11-only resolve drops
//!    those passes entirely.
//!
//! The order is live-type-driven at every step: the object list is re-read and
//! the current type decides the rung, because the guest recycles object refs and
//! a ref that was an IOSurface last frame can be a linear texture this one.
//!
//! # Why this is its own module
//!
//! It is 380 lines of one question with four answers, and it sat in the middle
//! of `metal_draw`'s 4 700-line body between the ICB execute entry point and the
//! guest-page writers. Nothing in it is backend-specific — no `cfg` gate here,
//! for the same reason [`super::texture_view`] carries none — and both arms take
//! it on every draw.

use super::*;
/// The colour render target's base format for a **type-4** surface, or nothing.
///
/// On this arm `m.format == 0` is not "unset", it is a decoded refusal:
/// [`objects::apply_type4_backing`] is the only writer of it, and it stores 0 for
/// a multi-plane surface and for a single-plane one whose FourCC it does not
/// know, saying why — "stage/paint must not invent BGRA".
/// [`objects::iosurface_pixel_format_to_mtl`] repeats it twice more, and the
/// compute staging path honours it, declining with typed `multiplane` /
/// `fmt_unknown` reasons.
///
/// This resolve used to invent BGRA8 from it, so one surface was refused by the
/// compute path and rendered into as BGRA8 by this one. That is not a survivable
/// disagreement for a multi-plane surface: BGRA8 over a `'420f'` allocation
/// describes the wrong stride and the wrong bytes, and every downstream window is
/// built from what this returns.
///
/// It now declines. Refusing was held back because it drops a colour attachment
/// — a compositing layer going black — on a class nothing had counted, so the
/// class was counted first: `rt_base_fmt_invent` read **0 on two driven
/// x86/Vulkan boots** (Safari window drag plus the web-content probe). The arm is
/// unreached on this workload, so declining costs nothing measurable and stops
/// the device from silently rendering a format the guest did not declare. The
/// counter stays and the fail line stays, because "unreached on this workload" is
/// not "unreachable" — the first surface to take it will now be named and
/// refused rather than named and rendered wrong.
///
/// The **type-11** arm deliberately does not come through here. A type-11
/// mapping's format has other writers, so its 0 can mean "not latched yet" rather
/// than "refused", and BGRA8 is the display contract's stated default for that
/// case ([`crate::runtime::compute_exec`]'s `or_bgra8` writes the same rule down).
/// Those are different zeros and only this one is provably a refusal.
fn rt_type4_base_format(format: u16, mapping_id: u32) -> Option<u16> {
    if format != 0 {
        return Some(format);
    }
    crate::runtime::drain::note_store_route("rt_base_fmt_declined");
    if crate::observe::first_sight("rt_base_fmt_declined", mapping_id as u64) {
        crate::observe::fail(format!(
            "rt_base_fmt_declined mapping={mapping_id} \
             (the mapping's format is the type-4 decoder's multi-plane / \
             unknown-FourCC refusal, so this surface is not a single-format \
             colour attachment and no format is invented for it)"
        ));
    }
    None
}

/// Report a type-5 colour attachment whose view record disagrees with the base
/// mapping it is resolved through.
///
/// This resolve reads only `surfaceID@0` out of a type-5 descriptor and takes
/// geometry and format from the mapping. [`objects::decode_type5_texture_view`]'s
/// own contract forbids that — "callers must not replace it with base mapping
/// geometry merely because the surface itself is otherwise stageable" — and the
/// live case it names is real: the BGRA8 desktop target is also exposed as a
/// row-byte-equivalent quarter-width RGBA32Uint view. Every other type-5
/// consumer binds the view's own geometry.
///
/// It is harmless exactly while view == base, so the question is how often that
/// holds for a *render target* specifically, which nothing has measured.
/// `rt_type5_view_differs` against `rt_type5_view_same` answers it. Reported
/// rather than repaired: taking the view's geometry here changes what every
/// type-5 colour attachment renders into, and that is not a change to make on an
/// unmeasured population.
///
/// **Read on two driven x86/Vulkan boots: `same` 20 273 and 24 360, `differs`
/// 0, `undecoded` 0.** So on this workload every type-5 colour attachment's view
/// agrees with the base mapping in width, height and format, and resolving
/// through the base loses nothing. The reinterpretation view the contract names
/// is real traffic elsewhere — the compute staging path sees it — but it is not
/// bound as a render target here.
///
/// That is a reason not to change the resolve, and not a reason to stop asking.
/// `differs` is a healthy zero: the first non-zero line names a surface being
/// rendered at the wrong geometry, which no other counter in this path could
/// report.
fn note_rt_type5_view(
    view: Option<objects::Type5TextureView>,
    surface_id: u32,
    base: (u32, u32, u16),
) {
    let Some(view) = view else {
        crate::runtime::drain::note_store_route("rt_type5_view_undecoded");
        return;
    };
    let (base_w, base_h, base_fmt) = base;
    if view.width == base_w && view.height == base_h && view.pixel_format == base_fmt {
        crate::runtime::drain::note_store_route("rt_type5_view_same");
        return;
    }
    crate::runtime::drain::note_store_route("rt_type5_view_differs");
    if crate::observe::first_sight("rt_type5_view_differs", surface_id as u64) {
        crate::observe::fail(format!(
            "rt_type5_view_differs sid={surface_id} view={}x{} fmt={:#x} plane={} \
             base={base_w}x{base_h} fmt={base_fmt:#x} (the colour attachment is \
             resolved with the base mapping's geometry, not the view's)",
            view.width, view.height, view.pixel_format, view.plane_index
        ));
    }
}

/// Where a colour attachment's `texture_ref` actually resolved to.
///
/// This was six loose positional values — `(u32, u64, u32, u32, u32, u16)` —
/// and three of them are `u32` in a row, so every call site accepted the
/// permutation that swaps width, height and row stride. The two sites that
/// destructure it do so in different orders from the one that builds a
/// [`ColorRtRequest`] out of it, which is where such a swap would have gone
/// unnoticed: all three orders type-check.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ResolvedRenderTarget {
    /// Non-zero ⇒ a host mapping; `0` with `target_gva` non-zero ⇒ type-2/3
    /// linear guest VA. The two are exclusive, the same way
    /// [`ColorRtRequest::target_gva`] documents.
    pub(super) mapping_id: u32,
    pub(super) target_gva: u64,
    pub(super) width: u32,
    pub(super) height: u32,
    /// Bytes per row of the target (archive `bpr`).
    pub(super) row_stride: u32,
    pub(super) format: u16,
}

/// Archive `apple_pv_gpu_lookup_render_target`: type-11 first, else type-2/3 GVA.
///
/// Wallpaper/background intermediates are type-2/3 guest-VA; type-11-only resolve
/// drops those passes (black wallpaper). Color RT formats are the Metal color-
/// renderable set admitted by [`pixel_format::render_target_bpp`] (RGBA8 family,
/// BGRA8 family, RGBA16Float) — bring-up only listed compositor BGRA8/0x73.
///
/// Type-8 texture views (archive `resource_resolve_texture` view chain): resolve
/// to the base texture. Swizzled views are rejected as RTs (archive
/// `resolve_texture` requires `!has_swizzle`). Level 0 only for color RT
/// materialization (mip RT not supported). Without this, UI passes that bind a
/// type-8 view as color attachment fail MRT (`mrt_request fail slots=[211]`) and
/// drop entire draws (blank App Store sidebar / missing chrome labels).
pub(super) fn lookup_render_target<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &M,
    task_id: u32,
    texture_ref: u32,
) -> Option<ResolvedRenderTarget> {
    if texture_ref == 0 {
        return None;
    }
    // Type-8 view → base (archive resource_resolve_texture view chain).
    let (resolved_ref, view_fmt_override, view_level) =
        if let Some(view) = resolve_texture_view(state, host, task_id, texture_ref) {
            // Archive resolve_texture rejects swizzled views for linear resolve.
            if let Some(plan) = view.swizzle.as_ref() {
                if !pixel_format::swizzle_is_identity(plan) {
                    return None;
                }
            }
            (view.base_texture_ref, view.pixel_format, view.level)
        } else {
            (texture_ref, None, 0)
        };
    if resolved_ref == 0 {
        return None;
    }
    // Archive lookup order is by **live** object-list type + descriptor, not a
    // sticky cache: type-11 first, else type-2/3. Guest reuses object refs;
    // two failure modes for a stale `texture_to_mapping` latch:
    // 1) live type is now type-2/3 → must not force type-11 (live residual
    //    mrt color RT resolve fail ref=199 type=2 fmt=0x73 480x64).
    // 2) live type is still type-11 but descriptor mapping_id changed (or a
    //    recycled ref now names a different mid) → must re-read the live
    //    descriptor, not prefer the latch. Preferring latch routed dual-mid
    //    full-screen desktop composites onto only one mid (mid=3 nz=6M vs
    //    mid=4 stuck logo nz=1.97M; damage rects then preserved logo via Load).
    let live = objects::lookup_list_entry(state, host, task_id, resolved_ref);
    let live_type = live.as_ref().map(|e| e.object_type);
    if let Some(ot) = live_type {
        if ot != OBJECT_TYPE_IOSURFACE {
            // Live list says not type-11 — drop any recycled-ref latch.
            state.texture_to_mapping.remove(&(task_id, resolved_ref));
        }
    }
    let try_type11 = live_type == Some(OBJECT_TYPE_IOSURFACE)
        || (live_type.is_none()
            && state
                .texture_to_mapping
                .contains_key(&(task_id, resolved_ref)));
    if try_type11 {
        // Type-11 sample windows carry planes, not mip levels — a mip>0 view
        // of an IOSurface has no contract-backed layout; fail visibly.
        if view_level != 0 {
            return None;
        }
        // Live list is source of truth for mapping_id when the entry is type-11.
        // Latch is only a fallback when the list entry is transiently missing
        // (resolve_type11_ref refreshes the latch from the live descriptor).
        let mapping_id = if live_type == Some(OBJECT_TYPE_IOSURFACE) {
            objects::resolve_type11_ref(state, host, task_id, resolved_ref).or_else(|| {
                state
                    .texture_to_mapping
                    .get(&(task_id, resolved_ref))
                    .copied()
            })?
        } else {
            state
                .texture_to_mapping
                .get(&(task_id, resolved_ref))
                .copied()
                .or_else(|| objects::resolve_type11_ref(state, host, task_id, resolved_ref))?
        };
        let _ = mapper::ensure_resolved_for_scanout(state, host, mapping_id);
        if let Some(m) = state.mappings.get(&mapping_id) {
            if m.has_geom && m.width > 0 && m.height > 0 {
                // Not `rt_type4_base_format`: a type-11 mapping's format has
                // writers other than the type-4 decoder, so 0 here can mean "not
                // latched yet" rather than "refused", and BGRA8 is the display
                // contract's default for that case. See that function.
                let base_fmt = if m.format != 0 {
                    m.format
                } else {
                    MTL_FORMAT_BGRA8_UNORM
                };
                let fmt =
                    effective_view_sample_format(base_fmt, view_fmt_override).unwrap_or(base_fmt);
                if pixel_format::render_target_bpp(fmt).is_some() {
                    return Some(ResolvedRenderTarget {
                        mapping_id,
                        target_gva: 0,
                        width: m.width,
                        height: m.height,
                        row_stride: 0,
                        format: fmt,
                    });
                }
            }
        }
        // Live type-11 that failed geom: do not decode as type-2.
        if live_type == Some(OBJECT_TYPE_IOSURFACE) {
            return None;
        }
    }
    // x86 Ventura/Tahoe type-4 surface/backing (present IOSurface). Object-list
    // index == surface_id (ResourceHeap addObject type=4 objectId=getSurfaceID).
    // Without this, clear-only streams and Store writebacks never touch display
    // mids — guest pages stay empty and dual-mid thrash paints black.
    // Type-4: object-list index is surface_id. Type-5 RefTextureHandle: surfaceID@0
    // (allocateRefTextureHandle) — product color RTs are type-5 wrapping type-4.
    let mut type5_view: Option<objects::Type5TextureView> = None;
    let type4_sid = if live_type == Some(objects::OBJECT_TYPE_SURFACE) {
        Some(resolved_ref)
    } else if live_type == Some(objects::OBJECT_TYPE_REF_TEXTURE) {
        let entry = live.as_ref()?;
        let desc = objects::read_descriptor(state, host, task_id, entry)?;
        let sid = reims_vgpu_wire::device_desc::type5_header(&desc)
            .ok()?
            .surface_id
            .get();
        if sid == 0 {
            return None;
        }
        type5_view = objects::decode_type5_texture_view(&desc);
        Some(sid)
    } else {
        None
    };
    if let Some(surface_id) = type4_sid {
        if view_level != 0 {
            return None;
        }
        if !objects::resolve_type4_surface(state, host, surface_id) {
            crate::observe::fail(format!(
                "rt_resolve FAIL type4 tex_ref={resolved_ref} sid={surface_id} live_type={live_type:?}"
            ));
            return None;
        }
        let m = state.mappings.get(&surface_id)?;
        if !m.has_geom || m.width == 0 || m.height == 0 || m.page_entries.is_empty() {
            crate::observe::fail(format!(
                "rt_resolve FAIL type4_geom tex_ref={resolved_ref} sid={surface_id} has_geom={} pages={}",
                m.has_geom,
                m.page_entries.len()
            ));
            return None;
        }
        let (base_w, base_h, base_raw_fmt) = (m.width, m.height, m.format);
        if live_type == Some(objects::OBJECT_TYPE_REF_TEXTURE) {
            note_rt_type5_view(type5_view, surface_id, (base_w, base_h, base_raw_fmt));
        }
        let base_fmt = rt_type4_base_format(base_raw_fmt, surface_id)?;
        let fmt = effective_view_sample_format(base_fmt, view_fmt_override).unwrap_or(base_fmt);
        pixel_format::render_target_bpp(fmt)?;
        // mapping_id = surface_id; no linear GVA.
        return Some(ResolvedRenderTarget {
            mapping_id: surface_id,
            target_gva: 0,
            width: m.width,
            height: m.height,
            row_stride: 0,
            format: fmt,
        });
    }
    // type-2/3 linear GVA (wallpaper/background layers, UI intermediate RTs).
    let entry = live?;
    if entry.object_type != OBJECT_TYPE_TEXTURE && entry.object_type != OBJECT_TYPE_TEXTURE_VARIANT
    {
        return None;
    }
    let desc_bytes = objects::read_descriptor(state, host, task_id, &entry)?;
    let tex = decode_texture_descriptor(&desc_bytes).ok()?;
    if tex.declared_pixel_format().is_none()
        || tex.extent().is_none()
        || tex.declared_row_stride().is_none()
    {
        return None;
    }
    let base_fmt = tex.pixel_format;
    let fmt = effective_view_sample_format(base_fmt, view_fmt_override).unwrap_or(base_fmt);
    // Refuses a format with no known bytes-per-texel; the value is not needed.
    pixel_format::render_target_bpp(fmt)?;
    // Mip>0 view of a linear texture: the RT is that level's plane inside the
    // base allocation (archive collapses view mip into linear geometry —
    // compositor blur/backdrop pyramids render into successive levels).
    let (gva, w, h, bpr) = if view_level != 0 {
        let (level_gva, layout) = tex.level_gva(view_level, state.page_shift)?;
        if layout.row_stride > u32::MAX as u64 {
            return None;
        }
        // Full level span must fit the allocation — writing rows past it would
        // corrupt adjacent guest memory.
        //
        // This is deliberately still `row_stride * height` and NOT
        // `TextureLevelLayout::read_span`, which every *reader* of a level uses.
        // The difference is one row of trailing padding, and whether this path
        // touches it depends on what the render-target store writes per row —
        // a question about the store, not about this bound. Until that is
        // measured, the wider span is the safe direction here: it can only
        // refuse a target, never let a write run past the allocation.
        let span = layout.row_stride.checked_mul(layout.height as u64)?;
        if tex.allocation_size != 0 && layout.offset.saturating_add(span) > tex.allocation_size {
            return None;
        }
        (
            level_gva,
            layout.width,
            layout.height,
            layout.row_stride as u32,
        )
    } else {
        let (gva, alloc) = tex.backing_gva_size(state.page_shift)?;
        let span = (tex.row_stride as u64).checked_mul(tex.height as u64)?;
        let tight0 = pixel_format::tight_row_bytes(tex.width, fmt)?;
        // Exclusive last-row end (archive): height-1 * bpr + tight may fit
        // tighter allocs; accept if bpr*height fits allocation.
        if alloc > 0 && span > alloc {
            let alt = if tex.height > 0 {
                (tex.row_stride as u64)
                    .saturating_mul((tex.height - 1) as u64)
                    .saturating_add(tight0 as u64)
            } else {
                0
            };
            if alt > alloc {
                return None;
            }
        }
        (gva, tex.width, tex.height, tex.row_stride)
    };
    let tight = pixel_format::tight_row_bytes(w, fmt)?;
    if bpr < tight || w == 0 || h == 0 {
        return None;
    }
    Some(ResolvedRenderTarget {
        mapping_id: 0,
        target_gva: gva,
        width: w,
        height: h,
        row_stride: bpr,
        format: fmt,
    })
}

/// The two report helpers above, tested where they live.
///
/// Both are pure given their arguments — one maps a decoded format to a decision
/// and one scores a view against a base — so neither needs a device, a mapping,
/// or a boot to hold. They moved here with the code they describe; they were
/// written against it in `metal_draw`'s 4 700-line colocated test module, which
/// is the file the plan wants to stop growing.
#[cfg(test)]
mod tests {
    use super::*;

    /// A type-4 colour attachment whose mapping carries the decoder's format
    /// refusal must be declined, and every decline must be counted.
    ///
    /// `m.format == 0` on a type-4 mapping has exactly one writer,
    /// `apply_type4_backing`, and it means multi-plane or unknown FourCC — a surface
    /// that is not a single-format colour attachment. Inventing BGRA8 from it
    /// describes the wrong stride over the wrong bytes and every downstream window
    /// is built from the answer. The counter has to fire on the refusal and only on
    /// it: one that also fired on ordinary formats would answer a different question
    /// and read identically.
    #[test]
    fn a_type4_render_target_declines_the_decoders_format_refusal() {
        use crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;
        use crate::runtime::drain::store_route_count;

        let before = store_route_count("rt_base_fmt_declined");
        // A format the decoder resolved is passed through untouched and uncounted.
        assert_eq!(
            rt_type4_base_format(MTL_FORMAT_BGRA8_UNORM, 11),
            Some(MTL_FORMAT_BGRA8_UNORM)
        );
        assert_eq!(store_route_count("rt_base_fmt_declined"), before);
        // The refusal declines, and is counted per occurrence — the fail line is
        // deduped per mapping, the counter is not.
        assert_eq!(rt_type4_base_format(0, 11), None);
        assert_eq!(rt_type4_base_format(0, 12), None);
        assert_eq!(rt_type4_base_format(0, 11), None);
        assert_eq!(store_route_count("rt_base_fmt_declined"), before + 3);
    }

    /// A type-5 colour attachment must be scored on whether its view agrees with
    /// the base mapping, and "no view decoded" must not read as agreement.
    ///
    /// The resolve takes geometry from the base mapping either way, so the counter
    /// is the only thing that can say whether that is lossless. Folding an
    /// undecoded record into `same` would report the ambiguous case as the healthy
    /// one, which is the failure mode that makes a census worthless.
    #[test]
    fn a_type5_render_target_view_is_scored_against_the_base_it_resolves_through() {
        use crate::runtime::drain::store_route_count;
        use crate::runtime::objects::Type5TextureView;

        let base = (64u32, 32u32, 0x50u16);
        let view = |w, h, fmt| {
            Some(Type5TextureView {
                pixel_format: fmt,
                width: w,
                height: h,
                depth: 1,
                plane_index: 0,
            })
        };
        let (same0, diff0, und0) = (
            store_route_count("rt_type5_view_same"),
            store_route_count("rt_type5_view_differs"),
            store_route_count("rt_type5_view_undecoded"),
        );

        note_rt_type5_view(view(64, 32, 0x50), 5, base);
        assert_eq!(store_route_count("rt_type5_view_same"), same0 + 1);

        // The live case the contract names: a row-byte-equivalent reinterpretation
        // at a different width and format over the same bytes.
        note_rt_type5_view(view(16, 32, 0x73), 6, base);
        assert_eq!(store_route_count("rt_type5_view_differs"), diff0 + 1);
        // Geometry alone is not the test — a format-only view is still a different
        // view, and it is the one this resolve would silently render as BGRA8.
        note_rt_type5_view(view(64, 32, 0x73), 7, base);
        assert_eq!(store_route_count("rt_type5_view_differs"), diff0 + 2);

        note_rt_type5_view(None, 8, base);
        assert_eq!(store_route_count("rt_type5_view_undecoded"), und0 + 1);
        assert_eq!(
            store_route_count("rt_type5_view_same"),
            same0 + 1,
            "an undecoded record must not be scored as agreement"
        );
    }
}
