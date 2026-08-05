//! Process-global content-hash caches.

use crate::backend::metal::abi::{
    ReimsVgpuComputeTextureUsage, ReimsVgpuDepthStencilState, ReimsVgpuSampler,
};
use crate::backend::metal::constants::*;
use crate::backend::metal::hash::hash_u64;
use crate::contract::fnv::FNV_OFFSET_BASIS;
use crate::model::content_cache::{CacheEntry, ContentCache};
use crate::runtime::decode::resource::MTL_COLOR_WRITE_MASK_ALL;
use metal::{ComputePipelineState, DepthStencilState, Function, RenderPipelineState, SamplerState};
use parking_lot::Mutex;

pub struct FnEntry {
    pub key: BlobKey,
    pub function: Function,
}

/// Identity of a cached `MTLRenderPipelineState`.
///
/// # The shader is identified by fingerprint, and the fingerprint is 64 bits
///
/// [`RenderPsoKey::equal`] compares `vert_hash`/`frag_hash` and the two lengths.
/// **The shader bytes themselves are never compared** — they are not retained to
/// compare against. So two distinct blobs of equal length whose
/// [`super::hash::hash_bytes`] outputs collide are one pipeline as far as this
/// cache is concerned, and a draw gets an `MTLRenderPipelineState` built from
/// the other blob's shader. Nothing refuses; the frame is simply wrong.
/// `render_key_hash_and_shader_lengths_are_identity_fields` pins that these
/// really are the identity fields.
///
/// # The other backend has a written rule against exactly this
///
/// `backend::vulkan::engine::digest` opens "≥128-bit content digests for cache
/// keys (never bare `DefaultHasher` u64 alone)", and
/// `engine::pools::sampled_content_hash` quantifies why: once a cache matches a
/// blob to a retained image "by this fingerprint alone — it no longer keeps a
/// byte copy to `memcmp` against", the width has to make an accidental
/// collision astronomically unlikely, which at 128 bits it puts at ~2^-116 over
/// that cache. This cache meets that description and uses 64 bits of non-keyed
/// FNV-1a. Neither site knew about the other.
///
/// Recorded rather than fixed, deliberately. The fix is a behaviour change on a
/// pathway no Linux host can boot, and half of it is worse than none: a second
/// fold added at the producer but missed at one of the sites that build a key
/// through [`RenderPsoKey::default`] would make two paths' keys never match,
/// turning every cache hit into a pipeline rebuild — a performance collapse only
/// an Apple host can see.
///
/// What would settle it, cheapest first. The live distinct-shader count sets the
/// birthday bound and is a counter an Apple boot could read. Whether a collision
/// is reachable at all is harder: equal-length collisions exist only above eight
/// bytes, and finding one is a 2^32 meet-in-the-middle, so this is not a hazard
/// a test can demonstrate.
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
    /// Number of active color RTs (`0..=REIMS_VGPU_METAL_MAX_COLOR_RTS`). Slot
    /// `i` uses `color_formats[i]` — backticked because a bare `[i]` is link
    /// syntax, and rustdoc was reporting an unresolved link to `i`. The sibling
    /// field below already spelled it this way.
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

/// A content hash and the length it was taken over.
///
/// The length is part of the key rather than a redundant field beside it: a
/// 64-bit hash of a shader blob can collide, and two blobs of different lengths
/// that collide would otherwise share one compiled object.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct BlobKey {
    pub hash: u64,
    pub len: usize,
}

/// What decides `MTLComputePipelineState` identity: the kernel blob, plus the
/// stage-input descriptor the PSO is specialized against.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct ComputePsoKey {
    pub mtlb: BlobKey,
    pub stage_hash: u64,
    pub has_stage_input: u8,
}

pub struct ComputePsoEntry {
    pub key: ComputePsoKey,
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
            .fold(FNV_OFFSET_BASIS, |h, &w| hash_u64(h, w as u64));
        Self { hash, words }
    }
}

pub struct SamplerCacheEntry {
    pub key: SamplerDescriptorKey,
    pub state: SamplerState,
}

/// What decides `MTLDepthStencilState` identity.
///
/// `hash` is a prefilter for the byte compare, the same role `hash` plays in
/// [`SamplerDescriptorKey`] — the descriptor is the identity and the hash only
/// rejects early.
#[derive(Clone, Copy)]
pub struct DepthStencilKey {
    pub hash: u64,
    pub desc: ReimsVgpuDepthStencilState,
}

pub struct DepthStencilEntry {
    pub key: DepthStencilKey,
    pub state: DepthStencilState,
}

pub struct ReflectEntry {
    pub key: BlobKey,
    pub usages: Vec<ReimsVgpuComputeTextureUsage>,
}

impl CacheEntry for FnEntry {
    type Key = BlobKey;
    fn key(&self) -> &BlobKey {
        &self.key
    }
    fn matches(&self, key: &BlobKey) -> bool {
        self.key == *key
    }
    /// The blob's content hash. `len` is left to `matches`, which is what
    /// keeps two equal-hash blobs of different lengths distinct entries.
    fn bucket(key: &BlobKey) -> u64 {
        key.hash
    }
}

impl CacheEntry for ComputePsoEntry {
    type Key = ComputePsoKey;
    fn key(&self) -> &ComputePsoKey {
        &self.key
    }
    fn matches(&self, key: &ComputePsoKey) -> bool {
        self.key == *key
    }
    /// The kernel blob's hash folded with the stage-input hash, so two PSOs
    /// specialized from one blob against different stage inputs do not pile
    /// into one bucket.
    fn bucket(key: &ComputePsoKey) -> u64 {
        hash_u64(key.mtlb.hash, key.stage_hash)
    }
}

impl CacheEntry for RenderPsoEntry {
    type Key = RenderPsoKey;
    fn key(&self) -> &RenderPsoKey {
        &self.key
    }
    fn matches(&self, key: &RenderPsoKey) -> bool {
        self.key.equal(key)
    }
    /// The key hash the descriptor is already folded into. `equal` still
    /// decides the hit; this only chooses which entries it is asked about.
    fn bucket(key: &RenderPsoKey) -> u64 {
        key.key_hash
    }
}

impl CacheEntry for SamplerCacheEntry {
    type Key = SamplerDescriptorKey;
    fn key(&self) -> &SamplerDescriptorKey {
        &self.key
    }
    fn matches(&self, key: &SamplerDescriptorKey) -> bool {
        self.key == *key
    }
    fn bucket(key: &SamplerDescriptorKey) -> u64 {
        key.hash
    }
}

impl CacheEntry for DepthStencilEntry {
    type Key = DepthStencilKey;
    fn key(&self) -> &DepthStencilKey {
        &self.key
    }
    fn matches(&self, key: &DepthStencilKey) -> bool {
        self.key.hash == key.hash && depth_stencil_eq(&self.key.desc, &key.desc)
    }
    fn bucket(key: &DepthStencilKey) -> u64 {
        key.hash
    }
}

impl CacheEntry for ReflectEntry {
    type Key = BlobKey;
    fn key(&self) -> &BlobKey {
        &self.key
    }
    fn matches(&self, key: &BlobKey) -> bool {
        self.key == *key
    }
    fn bucket(key: &BlobKey) -> u64 {
        key.hash
    }
}

struct GlobalCaches {
    fn_cache: ContentCache<FnEntry>,
    render_pso: ContentCache<RenderPsoEntry>,
    compute_pso: ContentCache<ComputePsoEntry>,
    sampler: ContentCache<SamplerCacheEntry>,
    depth_stencil: ContentCache<DepthStencilEntry>,
    reflect: ContentCache<ReflectEntry>,
}

impl GlobalCaches {
    const fn new() -> Self {
        Self {
            fn_cache: ContentCache::new(),
            render_pso: ContentCache::new(),
            compute_pso: ContentCache::new(),
            sampler: ContentCache::new(),
            depth_stencil: ContentCache::new(),
            reflect: ContentCache::new(),
        }
    }
}

static CACHES: Mutex<Option<GlobalCaches>> = Mutex::new(None);

fn with_caches<R>(f: impl FnOnce(&mut GlobalCaches) -> R) -> R {
    let mut guard = CACHES.lock();
    f(guard.get_or_insert_with(GlobalCaches::new))
}

pub fn fn_cache_lookup(key: &BlobKey) -> Option<Function> {
    with_caches(|c| c.fn_cache.find(key).map(|e| e.function.clone()))
}

pub fn fn_cache_insert(key: BlobKey, function: Function) -> Function {
    with_caches(|c| {
        c.fn_cache
            .insert_unique(FnEntry { key, function })
            .function
            .clone()
    })
}

pub fn compute_pso_lookup(key: &ComputePsoKey) -> Option<ComputePipelineState> {
    with_caches(|c| c.compute_pso.find(key).map(|e| e.pso.clone()))
}

pub fn compute_pso_insert(key: ComputePsoKey, pso: ComputePipelineState) -> ComputePipelineState {
    with_caches(|c| {
        c.compute_pso
            .insert_unique(ComputePsoEntry { key, pso })
            .pso
            .clone()
    })
}

pub fn render_pso_lookup(key: &RenderPsoKey) -> Option<(RenderPipelineState, u32, u32)> {
    with_caches(|c| {
        c.render_pso
            .find(key)
            .map(|e| (e.pso.clone(), e.vert_sampler_mask, e.frag_sampler_mask))
    })
}

pub fn render_pso_insert(
    key: RenderPsoKey,
    pso: RenderPipelineState,
    vert_mask: u32,
    frag_mask: u32,
) -> (RenderPipelineState, u32, u32) {
    with_caches(|c| {
        let entry = c.render_pso.insert_unique(RenderPsoEntry {
            key,
            pso,
            frag_sampler_mask: frag_mask,
            vert_sampler_mask: vert_mask,
        });
        (
            entry.pso.clone(),
            entry.vert_sampler_mask,
            entry.frag_sampler_mask,
        )
    })
}

pub fn sampler_lookup(key: &SamplerDescriptorKey) -> Option<SamplerState> {
    with_caches(|c| c.sampler.find(key).map(|e| e.state.clone()))
}

pub fn sampler_insert(key: SamplerDescriptorKey, state: SamplerState) -> SamplerState {
    with_caches(|c| {
        c.sampler
            .insert_unique(SamplerCacheEntry { key, state })
            .state
            .clone()
    })
}

pub fn depth_stencil_lookup(key: &DepthStencilKey) -> Option<DepthStencilState> {
    with_caches(|c| c.depth_stencil.find(key).map(|e| e.state.clone()))
}

pub fn depth_stencil_insert(key: DepthStencilKey, state: DepthStencilState) -> DepthStencilState {
    with_caches(|c| {
        c.depth_stencil
            .insert_unique(DepthStencilEntry { key, state })
            .state
            .clone()
    })
}

fn depth_stencil_eq(a: &ReimsVgpuDepthStencilState, b: &ReimsVgpuDepthStencilState) -> bool {
    crate::backend::metal::util::bytes_of(a) == crate::backend::metal::util::bytes_of(b)
}

pub fn reflect_lookup(key: &BlobKey) -> Option<Vec<ReimsVgpuComputeTextureUsage>> {
    with_caches(|c| c.reflect.find(key).map(|e| e.usages.clone()))
}

pub fn reflect_insert(key: BlobKey, usages: Vec<ReimsVgpuComputeTextureUsage>) {
    with_caches(|c| {
        c.reflect.insert_unique(ReflectEntry { key, usages });
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
