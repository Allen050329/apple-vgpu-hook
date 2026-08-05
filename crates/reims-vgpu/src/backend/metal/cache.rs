//! Process-global content-hash caches.

use crate::backend::metal::abi::{
    ReimsVgpuComputeTextureUsage, ReimsVgpuDepthStencilState, ReimsVgpuSampler,
};
use crate::backend::metal::constants::*;
use crate::backend::metal::hash::hash_u64;
use crate::runtime::decode::resource::MTL_COLOR_WRITE_MASK_ALL;
use metal::{ComputePipelineState, DepthStencilState, Function, RenderPipelineState, SamplerState};
use parking_lot::Mutex;

pub struct FnEntry {
    pub hash: u64,
    pub len: usize,
    pub function: Function,
}

pub struct RenderPsoKey {
    pub key_hash: u64,
    pub vert_hash: u64,
    pub frag_hash: u64,
    pub vert_len: usize,
    pub frag_len: usize,
    pub attr_count: u32,
    pub attr_location: [u32; REIMS_VGPU_METAL_MAX_ATTRS],
    pub attr_format: [u32; REIMS_VGPU_METAL_MAX_ATTRS],
    pub attr_offset: [u32; REIMS_VGPU_METAL_MAX_ATTRS],
    pub attr_buffer_index: [u32; REIMS_VGPU_METAL_MAX_ATTRS],
    pub attr_stride: [u32; REIMS_VGPU_METAL_MAX_ATTRS],
    /// Resolved step state, not the record's optionals: the caller has already
    /// applied the absent-field defaults, so the presence bits that used to sit
    /// beside these could only ever repeat what they had been folded into.
    pub attr_step_function: [u32; REIMS_VGPU_METAL_MAX_ATTRS],
    pub attr_step_rate: [u32; REIMS_VGPU_METAL_MAX_ATTRS],
    pub blend_enable: u8,
    pub blend_src_rgb: u32,
    pub blend_dst_rgb: u32,
    pub blend_op_rgb: u32,
    pub blend_src_alpha: u32,
    pub blend_dst_alpha: u32,
    pub blend_op_alpha: u32,
    /// Number of active color RTs (0..=REIMS_VGPU_METAL_MAX_COLOR_RTS). Slot i uses color_formats[i].
    pub color_count: u32,
    pub color_formats: [u32; REIMS_VGPU_METAL_MAX_COLOR_RTS],
    pub color_slot: [u8; REIMS_VGPU_METAL_MAX_COLOR_RTS],
    /// Per-RT blend enable + factors (aligned with color_count entries).
    pub color_blend_enable: [u8; REIMS_VGPU_METAL_MAX_COLOR_RTS],
    pub color_blend_src_rgb: [u32; REIMS_VGPU_METAL_MAX_COLOR_RTS],
    pub color_blend_dst_rgb: [u32; REIMS_VGPU_METAL_MAX_COLOR_RTS],
    pub color_blend_op_rgb: [u32; REIMS_VGPU_METAL_MAX_COLOR_RTS],
    pub color_blend_src_alpha: [u32; REIMS_VGPU_METAL_MAX_COLOR_RTS],
    pub color_blend_dst_alpha: [u32; REIMS_VGPU_METAL_MAX_COLOR_RTS],
    pub color_blend_op_alpha: [u32; REIMS_VGPU_METAL_MAX_COLOR_RTS],
    /// Per-RT `MTLColorWriteMask`, in Metal's own bit order.
    ///
    /// Outside the `color_blend_*` group on purpose: the mask applies whether
    /// or not the slot blends, so it is keyed and applied unconditionally
    /// while the blend fields are only meaningful under
    /// `color_blend_enable[i]`.
    pub color_write_mask: [u32; REIMS_VGPU_METAL_MAX_COLOR_RTS],
    pub depth_pixel_format: u32,
    pub stencil_pixel_format: u32,
}

impl Default for RenderPsoKey {
    fn default() -> Self {
        Self {
            key_hash: 0,
            vert_hash: 0,
            frag_hash: 0,
            vert_len: 0,
            frag_len: 0,
            attr_count: 0,
            attr_location: [0; REIMS_VGPU_METAL_MAX_ATTRS],
            attr_format: [0; REIMS_VGPU_METAL_MAX_ATTRS],
            attr_offset: [0; REIMS_VGPU_METAL_MAX_ATTRS],
            attr_buffer_index: [0; REIMS_VGPU_METAL_MAX_ATTRS],
            attr_stride: [0; REIMS_VGPU_METAL_MAX_ATTRS],
            attr_step_function: [0; REIMS_VGPU_METAL_MAX_ATTRS],
            attr_step_rate: [0; REIMS_VGPU_METAL_MAX_ATTRS],
            blend_enable: 0,
            blend_src_rgb: 0,
            blend_dst_rgb: 0,
            blend_op_rgb: 0,
            blend_src_alpha: 0,
            blend_dst_alpha: 0,
            blend_op_alpha: 0,
            color_count: 0,
            color_formats: [0; REIMS_VGPU_METAL_MAX_COLOR_RTS],
            color_slot: [0; REIMS_VGPU_METAL_MAX_COLOR_RTS],
            color_blend_enable: [0; REIMS_VGPU_METAL_MAX_COLOR_RTS],
            color_blend_src_rgb: [0; REIMS_VGPU_METAL_MAX_COLOR_RTS],
            color_blend_dst_rgb: [0; REIMS_VGPU_METAL_MAX_COLOR_RTS],
            color_blend_op_rgb: [0; REIMS_VGPU_METAL_MAX_COLOR_RTS],
            color_blend_src_alpha: [0; REIMS_VGPU_METAL_MAX_COLOR_RTS],
            color_blend_dst_alpha: [0; REIMS_VGPU_METAL_MAX_COLOR_RTS],
            color_blend_op_alpha: [0; REIMS_VGPU_METAL_MAX_COLOR_RTS],
            // `MTLColorWriteMaskAll`. Zero here would mean a default-built key
            // describes a pipeline that writes no channel at all.
            color_write_mask: [MTL_COLOR_WRITE_MASK_ALL; REIMS_VGPU_METAL_MAX_COLOR_RTS],
            depth_pixel_format: 0,
            stencil_pixel_format: 0,
        }
    }
}

impl RenderPsoKey {
    pub fn equal(&self, other: &Self) -> bool {
        if self.key_hash != other.key_hash
            || self.vert_hash != other.vert_hash
            || self.frag_hash != other.frag_hash
            || self.vert_len != other.vert_len
            || self.frag_len != other.frag_len
            || self.attr_count != other.attr_count
            || self.blend_enable != other.blend_enable
            || self.blend_src_rgb != other.blend_src_rgb
            || self.blend_dst_rgb != other.blend_dst_rgb
            || self.blend_op_rgb != other.blend_op_rgb
            || self.blend_src_alpha != other.blend_src_alpha
            || self.blend_dst_alpha != other.blend_dst_alpha
            || self.blend_op_alpha != other.blend_op_alpha
            || self.color_count != other.color_count
            || self.depth_pixel_format != other.depth_pixel_format
            || self.stencil_pixel_format != other.stencil_pixel_format
        {
            return false;
        }
        for i in 0..self.color_count as usize {
            if self.color_formats[i] != other.color_formats[i]
                || self.color_slot[i] != other.color_slot[i]
                || self.color_blend_enable[i] != other.color_blend_enable[i]
                || self.color_blend_src_rgb[i] != other.color_blend_src_rgb[i]
                || self.color_blend_dst_rgb[i] != other.color_blend_dst_rgb[i]
                || self.color_blend_op_rgb[i] != other.color_blend_op_rgb[i]
                || self.color_blend_src_alpha[i] != other.color_blend_src_alpha[i]
                || self.color_blend_dst_alpha[i] != other.color_blend_dst_alpha[i]
                || self.color_blend_op_alpha[i] != other.color_blend_op_alpha[i]
                || self.color_write_mask[i] != other.color_write_mask[i]
            {
                return false;
            }
        }
        for i in 0..self.attr_count as usize {
            if self.attr_location[i] != other.attr_location[i]
                || self.attr_format[i] != other.attr_format[i]
                || self.attr_offset[i] != other.attr_offset[i]
                || self.attr_buffer_index[i] != other.attr_buffer_index[i]
                || self.attr_stride[i] != other.attr_stride[i]
                || self.attr_step_function[i] != other.attr_step_function[i]
                || self.attr_step_rate[i] != other.attr_step_rate[i]
            {
                return false;
            }
        }
        true
    }
}

pub struct RenderPsoEntry {
    pub key: RenderPsoKey,
    pub pso: RenderPipelineState,
    pub frag_sampler_mask: u32,
    pub vert_sampler_mask: u32,
}

pub struct ComputePsoEntry {
    pub mtlb_hash: u64,
    pub mtlb_len: usize,
    pub stage_hash: u64,
    pub has_stage_input: u8,
    pub pso: ComputePipelineState,
}

/// Every `MTLSamplerDescriptor` property this device sets, and nothing else, as
/// the words that decide `MTLSamplerState` identity.
///
/// One list, because it used to be four: the same fourteen fields were
/// transcribed into `SamplerCacheEntry`, again into its `matches`, again into
/// `sampler_key_hash`, and again into the entry's construction. A property
/// added to the descriptor in [`super::samplers::make_explicit_sampler`] and
/// forgotten in any one of them is a cache *hit* on a state built from
/// different words, which nothing reports and which shows up only as a texture
/// filtered the way some earlier bind asked for.
///
/// The three `ReimsVgpuSampler` words that are absent are absent by rule rather
/// than by omission: `has_lod_clamp` and the two `clamp_lod_*` words are the
/// encoder call `setSamplerState:lodMinClamp:lodMaxClamp:atIndex:`, applied per
/// bind and never baked into the state, so two binds differing only there share
/// one state and must hit. `binding` is per bind for the same reason.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct SamplerDescriptorKey {
    /// First so the derived comparison rejects on it before the words, which is
    /// the prefilter the separate `key_hash` field used to be.
    hash: u64,
    words: [u32; 14],
}

impl SamplerDescriptorKey {
    pub fn new(s: &ReimsVgpuSampler) -> Self {
        let words = [
            s.unnormalized,
            s.min_filter,
            s.mag_filter,
            s.mip_filter,
            s.s_address_mode,
            s.t_address_mode,
            s.r_address_mode,
            s.border_color,
            s.compare_function,
            s.lod_min_bits,
            s.lod_max_bits,
            s.max_anisotropy,
            s.lod_average,
            s.support_argument_buffers,
        ];
        let hash = words
            .iter()
            .fold(0xcbf2_9ce4_8422_2325u64, |h, &w| hash_u64(h, w as u64));
        Self { hash, words }
    }
}

pub struct SamplerCacheEntry {
    pub key: SamplerDescriptorKey,
    pub state: SamplerState,
}

pub struct DepthStencilEntry {
    pub key_hash: u64,
    pub desc: ReimsVgpuDepthStencilState,
    pub state: DepthStencilState,
}

pub struct ReflectEntry {
    pub mtlb_hash: u64,
    pub mtlb_len: usize,
    pub usages: Vec<ReimsVgpuComputeTextureUsage>,
}

struct GlobalCaches {
    fn_cache: Vec<Option<FnEntry>>,
    fn_clock: usize,
    render_pso: Vec<Option<RenderPsoEntry>>,
    render_pso_clock: usize,
    compute_pso: Vec<Option<ComputePsoEntry>>,
    compute_pso_clock: usize,
    sampler: Vec<Option<SamplerCacheEntry>>,
    sampler_clock: usize,
    depth_stencil: Vec<Option<DepthStencilEntry>>,
    depth_stencil_clock: usize,
    reflect: Vec<Option<ReflectEntry>>,
    reflect_clock: usize,
}

impl GlobalCaches {
    fn new() -> Self {
        Self {
            fn_cache: Vec::new(),
            fn_clock: 0,
            render_pso: Vec::new(),
            render_pso_clock: 0,
            compute_pso: Vec::new(),
            compute_pso_clock: 0,
            sampler: Vec::new(),
            sampler_clock: 0,
            depth_stencil: Vec::new(),
            depth_stencil_clock: 0,
            reflect: Vec::new(),
            reflect_clock: 0,
        }
    }
}

static CACHES: Mutex<Option<GlobalCaches>> = Mutex::new(None);

fn with_caches<R>(f: impl FnOnce(&mut GlobalCaches) -> R) -> R {
    let mut guard = CACHES.lock();
    f(guard.get_or_insert_with(GlobalCaches::new))
}

pub fn fn_cache_lookup(hash: u64, len: usize) -> Option<Function> {
    with_caches(|c| {
        for e in c.fn_cache.iter().flatten() {
            if e.hash == hash && e.len == len {
                return Some(e.function.clone());
            }
        }
        None
    })
}

pub fn fn_cache_insert(hash: u64, len: usize, function: Function) -> Function {
    with_caches(|c| {
        for e in c.fn_cache.iter().flatten() {
            if e.hash == hash && e.len == len {
                return e.function.clone();
            }
        }
        let entry = FnEntry {
            hash,
            len,
            function: function.clone(),
        };
        if c.fn_cache.len() < REIMS_VGPU_FN_CACHE_CAP {
            c.fn_cache.push(Some(entry));
        } else {
            let slot = c.fn_clock % REIMS_VGPU_FN_CACHE_CAP;
            c.fn_clock = c.fn_clock.wrapping_add(1);
            if c.fn_cache.len() <= slot {
                c.fn_cache.resize_with(slot + 1, || None);
            }
            c.fn_cache[slot] = Some(entry);
        }
        function
    })
}

pub fn compute_pso_lookup(
    mtlb_hash: u64,
    mtlb_len: usize,
    stage_hash: u64,
    has_stage: u8,
) -> Option<ComputePipelineState> {
    with_caches(|c| {
        for e in c.compute_pso.iter().flatten() {
            if e.mtlb_hash == mtlb_hash
                && e.mtlb_len == mtlb_len
                && e.has_stage_input == has_stage
                && e.stage_hash == stage_hash
            {
                return Some(e.pso.clone());
            }
        }
        None
    })
}

pub fn compute_pso_insert(
    mtlb_hash: u64,
    mtlb_len: usize,
    stage_hash: u64,
    has_stage: u8,
    pso: ComputePipelineState,
) -> ComputePipelineState {
    with_caches(|c| {
        for e in c.compute_pso.iter().flatten() {
            if e.mtlb_hash == mtlb_hash
                && e.mtlb_len == mtlb_len
                && e.has_stage_input == has_stage
                && e.stage_hash == stage_hash
            {
                return e.pso.clone();
            }
        }
        let entry = ComputePsoEntry {
            mtlb_hash,
            mtlb_len,
            stage_hash,
            has_stage_input: has_stage,
            pso: pso.clone(),
        };
        if c.compute_pso.len() < REIMS_VGPU_COMPUTE_PSO_CACHE_CAP {
            c.compute_pso.push(Some(entry));
        } else {
            let slot = c.compute_pso_clock % REIMS_VGPU_COMPUTE_PSO_CACHE_CAP;
            c.compute_pso_clock = c.compute_pso_clock.wrapping_add(1);
            if c.compute_pso.len() <= slot {
                c.compute_pso.resize_with(slot + 1, || None);
            }
            c.compute_pso[slot] = Some(entry);
        }
        pso
    })
}

pub fn render_pso_lookup(key: &RenderPsoKey) -> Option<(RenderPipelineState, u32, u32)> {
    with_caches(|c| {
        for e in c.render_pso.iter().flatten() {
            if e.key.equal(key) {
                return Some((e.pso.clone(), e.vert_sampler_mask, e.frag_sampler_mask));
            }
        }
        None
    })
}

pub fn render_pso_insert(
    key: RenderPsoKey,
    pso: RenderPipelineState,
    vert_mask: u32,
    frag_mask: u32,
) -> (RenderPipelineState, u32, u32) {
    with_caches(|c| {
        for e in c.render_pso.iter().flatten() {
            if e.key.equal(&key) {
                return (e.pso.clone(), e.vert_sampler_mask, e.frag_sampler_mask);
            }
        }
        let out_pso = pso.clone();
        let entry = RenderPsoEntry {
            key,
            pso,
            frag_sampler_mask: frag_mask,
            vert_sampler_mask: vert_mask,
        };
        if c.render_pso.len() < REIMS_VGPU_RENDER_PSO_CACHE_CAP {
            c.render_pso.push(Some(entry));
        } else {
            let slot = c.render_pso_clock % REIMS_VGPU_RENDER_PSO_CACHE_CAP;
            c.render_pso_clock = c.render_pso_clock.wrapping_add(1);
            if c.render_pso.len() <= slot {
                c.render_pso.resize_with(slot + 1, || None);
            }
            c.render_pso[slot] = Some(entry);
        }
        (out_pso, vert_mask, frag_mask)
    })
}

pub fn sampler_lookup(key: &SamplerDescriptorKey) -> Option<SamplerState> {
    with_caches(|c| {
        for e in c.sampler.iter().flatten() {
            if e.key == *key {
                return Some(e.state.clone());
            }
        }
        None
    })
}

pub fn sampler_insert(key: SamplerDescriptorKey, state: SamplerState) -> SamplerState {
    with_caches(|c| {
        for e in c.sampler.iter().flatten() {
            if e.key == key {
                return e.state.clone();
            }
        }
        let entry = SamplerCacheEntry {
            key,
            state: state.clone(),
        };
        if c.sampler.len() < REIMS_VGPU_SAMPLER_CACHE_CAP {
            c.sampler.push(Some(entry));
        } else {
            let slot = c.sampler_clock % REIMS_VGPU_SAMPLER_CACHE_CAP;
            c.sampler_clock = c.sampler_clock.wrapping_add(1);
            if c.sampler.len() <= slot {
                c.sampler.resize_with(slot + 1, || None);
            }
            c.sampler[slot] = Some(entry);
        }
        state
    })
}

pub fn depth_stencil_lookup(
    key: u64,
    desc: &ReimsVgpuDepthStencilState,
) -> Option<DepthStencilState> {
    with_caches(|c| {
        for e in c.depth_stencil.iter().flatten() {
            if e.key_hash == key && depth_stencil_eq(&e.desc, desc) {
                return Some(e.state.clone());
            }
        }
        None
    })
}

pub fn depth_stencil_insert(
    key: u64,
    desc: ReimsVgpuDepthStencilState,
    state: DepthStencilState,
) -> DepthStencilState {
    with_caches(|c| {
        for e in c.depth_stencil.iter().flatten() {
            if e.key_hash == key && depth_stencil_eq(&e.desc, &desc) {
                return e.state.clone();
            }
        }
        let out = state.clone();
        let entry = DepthStencilEntry {
            key_hash: key,
            desc,
            state,
        };
        if c.depth_stencil.len() < REIMS_VGPU_DEPTH_STENCIL_CACHE_CAP {
            c.depth_stencil.push(Some(entry));
        } else {
            let slot = c.depth_stencil_clock % REIMS_VGPU_DEPTH_STENCIL_CACHE_CAP;
            c.depth_stencil_clock = c.depth_stencil_clock.wrapping_add(1);
            if c.depth_stencil.len() <= slot {
                c.depth_stencil.resize_with(slot + 1, || None);
            }
            c.depth_stencil[slot] = Some(entry);
        }
        out
    })
}

fn depth_stencil_eq(a: &ReimsVgpuDepthStencilState, b: &ReimsVgpuDepthStencilState) -> bool {
    crate::backend::metal::util::bytes_of(a) == crate::backend::metal::util::bytes_of(b)
}

pub fn reflect_lookup(
    mtlb_hash: u64,
    mtlb_len: usize,
) -> Option<Vec<ReimsVgpuComputeTextureUsage>> {
    with_caches(|c| {
        for e in c.reflect.iter().flatten() {
            if e.mtlb_hash == mtlb_hash && e.mtlb_len == mtlb_len {
                return Some(e.usages.clone());
            }
        }
        None
    })
}

pub fn reflect_insert(mtlb_hash: u64, mtlb_len: usize, usages: Vec<ReimsVgpuComputeTextureUsage>) {
    with_caches(|c| {
        let entry = ReflectEntry {
            mtlb_hash,
            mtlb_len,
            usages,
        };
        if c.reflect.len() < REIMS_VGPU_COMPUTE_REFLECT_CACHE_CAP {
            c.reflect.push(Some(entry));
        } else {
            let slot = c.reflect_clock % REIMS_VGPU_COMPUTE_REFLECT_CACHE_CAP;
            c.reflect_clock = c.reflect_clock.wrapping_add(1);
            if c.reflect.len() <= slot {
                c.reflect.resize_with(slot + 1, || None);
            }
            c.reflect[slot] = Some(entry);
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_key_compares_only_active_attachment_and_attribute_prefixes() {
        let mut left = RenderPsoKey::default();
        let mut right = RenderPsoKey::default();

        left.color_formats[7] = 70;
        right.color_formats[7] = 71;
        left.attr_location[30] = 30;
        right.attr_location[30] = 31;
        assert!(left.equal(&right), "inactive cache-key tails are ignored");

        left.color_count = 8;
        right.color_count = 8;
        assert!(
            !left.equal(&right),
            "an active attachment must affect equality"
        );
        right.color_formats[7] = left.color_formats[7];
        left.attr_count = 31;
        right.attr_count = 31;
        assert!(
            !left.equal(&right),
            "an active attribute must affect equality"
        );
        right.attr_location[30] = left.attr_location[30];
        assert!(left.equal(&right));
    }

    #[test]
    fn render_key_hash_and_shader_lengths_are_identity_fields() {
        let base = RenderPsoKey::default();
        let mutations: [fn(&mut RenderPsoKey); 5] = [
            |key| key.key_hash = 1,
            |key| key.vert_hash = 1,
            |key| key.frag_hash = 1,
            |key| key.vert_len = 1,
            |key| key.frag_len = 1,
        ];
        for mutate in mutations {
            let mut changed = RenderPsoKey::default();
            mutate(&mut changed);
            assert!(!base.equal(&changed));
        }
    }

    #[test]
    fn depth_stencil_cache_key_covers_both_faces() {
        let base = ReimsVgpuDepthStencilState::default();
        let mut changed = base;
        assert!(depth_stencil_eq(&base, &changed));
        changed.back_face.write_mask = 0xff;
        assert!(!depth_stencil_eq(&base, &changed));
    }
}
