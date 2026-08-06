//! Process-global content-hash caches.

use crate::backend::blob::{BlobIdentity, BlobKey};
use crate::backend::hash::hash_u64;
use crate::backend::metal::abi::{
    ReimsVgpuComputeStageInputDescriptor, ReimsVgpuComputeTextureUsage, ReimsVgpuDepthStencilState,
    ReimsVgpuSampler,
};
use crate::backend::metal::constants::*;
use crate::contract::fnv::FNV_OFFSET_BASIS;
use crate::model::content_cache::{CacheEntry, ContentCache};
use crate::runtime::decode::resource::MTL_COLOR_WRITE_MASK_ALL;
use metal::{ComputePipelineState, DepthStencilState, Function, RenderPipelineState, SamplerState};
use parking_lot::Mutex;

pub struct FnEntry {
    pub blob: BlobIdentity,
    pub function: Function,
}

/// Identity of a cached `MTLRenderPipelineState`.
///
/// # The shader is identified by fingerprint, and the fingerprint is 64 bits
///
/// [`RenderPsoKey::equal`] compares `vert_hash`/`frag_hash` and the two lengths.
/// **The shader bytes themselves are never compared** — they are not retained to
/// compare against. So two distinct blobs of equal length whose
/// [`crate::backend::hash::hash_bytes`] outputs collide are one pipeline as far as this
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
/// # Every sibling hole has been fixed, and this is the last one
///
/// `runtime::m2v_cache` had the identical shape over AIR, and the three caches
/// keyed on `crate::backend::blob::BlobKey` — functions, compute pipeline
/// states, reflections — had it over `.mtlb`. All four now bucket on the digest
/// and compare the bytes, which removes the class rather than moving the
/// exponent, at one retained blob per distinct shader. The argument that it is
/// affordable no longer needs making here: a `RenderPsoEntry` already retains an
/// `MTLRenderPipelineState` compiled from these very blobs.
///
/// **The enabling step is done.** `render::fill_render_pso_key` — the only site
/// in product code that builds one of these — used to close with
/// `..Default::default()`, so a `vert_mtlb` field added to this struct would
/// have been filled from `Default` at that site with nothing said. That literal
/// is now exhaustive, as `render::RenderPsoKeyClone::clone_key` always was, and
/// the compiler catches a new field at both.
///
/// What is left is to carry the bytes. `BlobIdentity`/`BlobKey` is the pair to
/// carry them with, and the shape is the one the three siblings use: the entry
/// retains, the lookup borrows. It wants one more move to be worth testing —
/// this struct names nothing from the `metal` crate, so it belongs outside the
/// gated tree for the reason [`crate::backend::hash`]'s declaration gives, and
/// its two `#[test]`s below have never executed anywhere as a result. It reaches
/// `REIMS_VGPU_METAL_MAX_ATTRS` and `REIMS_VGPU_METAL_MAX_COLOR_RTS` from a
/// `constants` module that *does* name `metal`; both are `const`-asserted equal
/// to a decoder bound in an ungated module, which is the seam to take.
///
/// What would settle the hazard's *size* rather than remove it: the live
/// distinct-shader count sets the birthday bound and is a counter an Apple boot
/// could read. Whether a collision is reachable at all is harder — equal-length
/// collisions exist only above eight bytes, and finding one is a 2^32
/// meet-in-the-middle, so this is not a hazard a test can demonstrate. The four
/// fixes' tests do not try: each forces two entries into one bucket and asks
/// through the real lookup, which is exactly the state a natural collision
/// produces.
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

/// What decides `MTLComputePipelineState` identity: the kernel blob, plus the
/// stage-input descriptor the PSO is specialized against.
///
/// Both halves are compared by content. The descriptor used to travel as a
/// `stage_hash` beside a `has_stage_input` flag and nothing retained it, so two
/// descriptors whose digests collided specialized one PSO — the same hole
/// [`crate::backend::blob`] describes for the blob, over 1520 bytes of decoded
/// guest record. It is `Copy` and this cache holds one per distinct pipeline, so
/// retaining it costs less than the flag saved.
#[derive(Clone, Copy)]
pub struct ComputePsoKey<'a> {
    pub mtlb: BlobKey<'a>,
    /// Buckets with the blob's digest and decides nothing.
    pub stage_hash: u64,
    pub stage_input: Option<&'a ReimsVgpuComputeStageInputDescriptor>,
}

pub struct ComputePsoEntry {
    pub mtlb: BlobIdentity,
    pub stage_hash: u64,
    pub stage_input: Option<ReimsVgpuComputeStageInputDescriptor>,
    pub pso: ComputePipelineState,
}

impl ComputePsoEntry {
    fn stage_input_is(&self, key: &ComputePsoKey<'_>) -> bool {
        match (&self.stage_input, key.stage_input) {
            (None, None) => true,
            (Some(mine), Some(theirs)) => {
                crate::backend::metal::util::bytes_of(mine)
                    == crate::backend::metal::util::bytes_of(theirs)
            }
            _ => false,
        }
    }
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
    pub blob: BlobIdentity,
    pub usages: Vec<ReimsVgpuComputeTextureUsage>,
}

impl CacheEntry for FnEntry {
    type Key<'a> = BlobKey<'a>;
    fn lookup_key(&self) -> BlobKey<'_> {
        self.blob.as_key()
    }
    /// The blob's bytes, not its digest — see [`crate::backend::blob`].
    fn matches(&self, key: &BlobKey<'_>) -> bool {
        self.blob.is(key)
    }
    fn bucket(key: &BlobKey<'_>) -> u64 {
        key.hash
    }
}

impl CacheEntry for ComputePsoEntry {
    type Key<'a> = ComputePsoKey<'a>;
    fn lookup_key(&self) -> ComputePsoKey<'_> {
        ComputePsoKey {
            mtlb: self.mtlb.as_key(),
            stage_hash: self.stage_hash,
            stage_input: self.stage_input.as_ref(),
        }
    }
    fn matches(&self, key: &ComputePsoKey<'_>) -> bool {
        self.mtlb.is(&key.mtlb) && self.stage_input_is(key)
    }
    /// The kernel blob's hash folded with the stage-input hash, so two PSOs
    /// specialized from one blob against different stage inputs do not pile
    /// into one bucket. Both are prefilters; `matches` compares both records.
    fn bucket(key: &ComputePsoKey<'_>) -> u64 {
        hash_u64(key.mtlb.hash, key.stage_hash)
    }
}

impl CacheEntry for RenderPsoEntry {
    /// Borrowed, unlike the sampler and depth-stencil keys beside it: this one
    /// is thirty-three fields wide, so an owned lookup key would copy every
    /// array on every scan step.
    type Key<'a> = &'a RenderPsoKey;
    fn lookup_key(&self) -> &RenderPsoKey {
        &self.key
    }
    fn matches(&self, key: &&RenderPsoKey) -> bool {
        self.key.equal(key)
    }
    /// The key hash the descriptor is already folded into. `equal` still
    /// decides the hit; this only chooses which entries it is asked about.
    fn bucket(key: &&RenderPsoKey) -> u64 {
        key.key_hash
    }
}

impl CacheEntry for SamplerCacheEntry {
    type Key<'a> = SamplerDescriptorKey;
    fn lookup_key(&self) -> SamplerDescriptorKey {
        self.key
    }
    fn matches(&self, key: &SamplerDescriptorKey) -> bool {
        self.key == *key
    }
    fn bucket(key: &SamplerDescriptorKey) -> u64 {
        key.hash
    }
}

impl CacheEntry for DepthStencilEntry {
    type Key<'a> = DepthStencilKey;
    fn lookup_key(&self) -> DepthStencilKey {
        self.key
    }
    fn matches(&self, key: &DepthStencilKey) -> bool {
        self.key.hash == key.hash && depth_stencil_eq(&self.key.desc, &key.desc)
    }
    fn bucket(key: &DepthStencilKey) -> u64 {
        key.hash
    }
}

impl CacheEntry for ReflectEntry {
    type Key<'a> = BlobKey<'a>;
    fn lookup_key(&self) -> BlobKey<'_> {
        self.blob.as_key()
    }
    fn matches(&self, key: &BlobKey<'_>) -> bool {
        self.blob.is(key)
    }
    fn bucket(key: &BlobKey<'_>) -> u64 {
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

/// Live entries in each cache, in the order
/// `(functions, render_pso, compute_pso, samplers, depth_stencil, reflections)`.
///
/// The Metal counterpart of the Vulkan engine's `object_cache_levels`, and the
/// reading that closes a gap this arm has carried since the caps came off:
/// [`crate::model::content_cache`] argues these tables settle at the guest's
/// distinct object set, and cites `pipelines=92` measured on the Vulkan arm
/// against the 64-slot render-PSO table this arm used to hold. That is the other
/// arm's count for the same command stream. This is how an Apple host reads its
/// own.
pub fn cache_levels() -> [usize; 6] {
    with_caches(|c| {
        [
            c.fn_cache.len(),
            c.render_pso.len(),
            c.compute_pso.len(),
            c.sampler.len(),
            c.depth_stencil.len(),
            c.reflect.len(),
        ]
    })
}

pub fn fn_cache_lookup(key: &BlobKey<'_>) -> Option<Function> {
    with_caches(|c| c.fn_cache.find(key).map(|e| e.function.clone()))
}

pub fn fn_cache_insert(key: &BlobKey<'_>, function: Function) -> Function {
    with_caches(|c| {
        c.fn_cache
            .insert_unique(FnEntry {
                blob: BlobIdentity::of(key),
                function,
            })
            .function
            .clone()
    })
}

pub fn compute_pso_lookup(key: &ComputePsoKey<'_>) -> Option<ComputePipelineState> {
    with_caches(|c| c.compute_pso.find(key).map(|e| e.pso.clone()))
}

pub fn compute_pso_insert(
    key: &ComputePsoKey<'_>,
    pso: ComputePipelineState,
) -> ComputePipelineState {
    with_caches(|c| {
        c.compute_pso
            .insert_unique(ComputePsoEntry {
                mtlb: BlobIdentity::of(&key.mtlb),
                stage_hash: key.stage_hash,
                stage_input: key.stage_input.copied(),
                pso,
            })
            .pso
            .clone()
    })
}

pub fn render_pso_lookup(key: &RenderPsoKey) -> Option<(RenderPipelineState, u32, u32)> {
    with_caches(|c| {
        c.render_pso
            .find(&key)
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

pub fn reflect_lookup(key: &BlobKey<'_>) -> Option<Vec<ReimsVgpuComputeTextureUsage>> {
    with_caches(|c| c.reflect.find(key).map(|e| e.usages.clone()))
}

pub fn reflect_insert(key: &BlobKey<'_>, usages: Vec<ReimsVgpuComputeTextureUsage>) {
    with_caches(|c| {
        c.reflect.insert_unique(ReflectEntry {
            blob: BlobIdentity::of(key),
            usages,
        });
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
