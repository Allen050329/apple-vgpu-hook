//! The one place a Vulkan device feature or format capability is both
//! **queried** and **enabled**.
//!
//! # The bug this retires
//!
//! `translate::sampler::address_mode` maps `MTLSamplerAddressModeMirrorClampToEdge`
//! to `vk::SamplerAddressMode::MIRROR_CLAMP_TO_EDGE`. That address mode requires
//! either `VkPhysicalDeviceVulkan12Features::samplerMirrorClampToEdge` or
//! `VK_KHR_sampler_mirror_clamp_to_edge`, and neither was ever requested — a
//! repo-wide search for any spelling of it returned zero hits outside the
//! translation table itself. The sampler was created with a mode the device had
//! not been asked for, which is undefined behaviour that a validation layer
//! catches on someone else's GPU and a shipping driver may simply honour.
//!
//! [`super`] already owns memory topology, zero-copy rails, driver quirks and
//! device selection, and its gate keeps those decisions in one place. It did
//! **not** own sampler or format features, so `context.rs` queried those inline —
//! correctly, in every case but one. Because there was no home, the one that got
//! missed got missed silently. This module is the home.
//!
//! # Two rules
//!
//! 1. **Query and enable together.** A feature that is asked about here and not
//!    enabled here is the same bug in a new place, so the enable list is built
//!    from this struct and nothing else.
//! 2. **Enable only what the backend binds.** `multi_viewport` used to be
//!    enabled while `engine::exec` declines any draw with more than one
//!    viewport. Harmless, but it means the list was a wish rather than a
//!    derivation — and a list that is not derived cannot be checked.

use ash::vk;

/// How this device can satisfy `MTLSamplerAddressModeMirrorClampToEdge`.
///
/// Three rungs rather than a bool, because *how* it is available decides what
/// must be enabled at device creation, and "not available" has to be an
/// answer the sampler path can act on rather than a silent bind.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MirrorClampToEdge {
    /// `VkPhysicalDeviceVulkan12Features::samplerMirrorClampToEdge`. Preferred:
    /// core on the 1.2 baseline every matrix cell meets, so no extension string.
    Core12,
    /// `VK_KHR_sampler_mirror_clamp_to_edge`. The pre-1.2 spelling, still the
    /// only one some drivers advertise.
    KhrExtension,
    /// Neither. The address mode must be **declined by name** at the sampler
    /// binding site — never bound ungated. This is the default so a
    /// `DeviceFeatures` built without a query never claims support it has not
    /// checked for.
    #[default]
    Unsupported,
}

impl MirrorClampToEdge {
    pub fn is_available(self) -> bool {
        !matches!(self, Self::Unsupported)
    }
}

/// Every device feature and format capability this backend depends on, resolved
/// against one physical device.
///
/// Plain bools rather than the ash feature structs: this is the *decision*, and
/// keeping it free of `p_next` chains is what lets it be built once, asserted in
/// tests without a GPU, and consumed by the two spots that need ash types.
#[derive(Clone, Copy, Debug, Default)]
pub struct DeviceFeatures {
    /// Defined bounds-clamped behaviour for out-of-range shader buffer access.
    /// The one feature the spec requires every implementation to support.
    pub robust_buffer_access: bool,
    pub sampler_anisotropy: bool,
    pub max_sampler_anisotropy: f32,
    pub shader_int16: bool,
    pub storage_image_extended_formats: bool,
    pub storage_image_write_without_format: bool,
    /// `B8G8R8A8_UNORM` usable as a storage image with optimal tiling. **Not**
    /// spec-mandatory — only `R8G8B8A8_UNORM` is — so the BGRA composite path
    /// needs this *and* `storage_image_write_without_format`.
    pub bgra8_storage: bool,
    pub storage16: bool,
    pub storage8: bool,
    pub float16: bool,
    pub int8: bool,
    pub shader_output_viewport_index: bool,
    pub mirror_clamp_to_edge: MirrorClampToEdge,
}

impl DeviceFeatures {
    /// The BGRA-storage composite path needs the format-less write feature and
    /// BGRA8 as a usable storage image. Named once so the pair cannot drift
    /// apart at a call site.
    pub fn storage_image_write_without_format_bgra(&self) -> bool {
        self.storage_image_write_without_format && self.bgra8_storage
    }

    /// The `vk::PhysicalDeviceFeatures` to enable, derived from what is
    /// supported **and** what the backend actually binds.
    ///
    /// `multi_viewport` is deliberately absent even where supported:
    /// `engine::exec` declines any draw carrying more than one viewport, so
    /// enabling it advertised a capability nothing reaches.
    pub fn enabled_features(&self) -> vk::PhysicalDeviceFeatures {
        vk::PhysicalDeviceFeatures::default()
            .robust_buffer_access(self.robust_buffer_access)
            .sampler_anisotropy(self.sampler_anisotropy)
            .shader_int16(self.shader_int16)
            .shader_storage_image_extended_formats(self.storage_image_extended_formats)
            .shader_storage_image_write_without_format(self.storage_image_write_without_format)
    }

    /// The Vulkan 1.2 features to enable.
    ///
    /// `sampler_mirror_clamp_to_edge` is set only on the [`MirrorClampToEdge::Core12`]
    /// rung; on [`MirrorClampToEdge::KhrExtension`] the extension string carries
    /// it instead, and on [`MirrorClampToEdge::Unsupported`] nothing is
    /// requested and the sampler path declines.
    pub fn enabled_vulkan12(&self) -> vk::PhysicalDeviceVulkan12Features<'static> {
        vk::PhysicalDeviceVulkan12Features::default()
            .shader_output_viewport_index(self.shader_output_viewport_index)
            .sampler_mirror_clamp_to_edge(self.mirror_clamp_to_edge == MirrorClampToEdge::Core12)
    }

    /// 16-bit storage-buffer access, for shaders that pack half-precision data.
    pub fn enabled_16bit_storage(&self) -> vk::PhysicalDevice16BitStorageFeatures<'static> {
        vk::PhysicalDevice16BitStorageFeatures::default()
            .storage_buffer16_bit_access(self.storage16)
    }

    /// 8-bit storage-buffer access.
    pub fn enabled_8bit_storage(&self) -> vk::PhysicalDevice8BitStorageFeatures<'static> {
        vk::PhysicalDevice8BitStorageFeatures::default().storage_buffer8_bit_access(self.storage8)
    }

    /// `shaderFloat16` / `shaderInt8`, which AIR uses for half and char types.
    pub fn enabled_float16_int8(&self) -> vk::PhysicalDeviceShaderFloat16Int8Features<'static> {
        vk::PhysicalDeviceShaderFloat16Int8Features::default()
            .shader_float16(self.float16)
            .shader_int8(self.int8)
    }

    /// Device extension names this feature set requires, beyond the ones the
    /// interop rails ask for.
    pub fn required_extensions(&self) -> Vec<*const std::os::raw::c_char> {
        let mut out = Vec::new();
        if self.mirror_clamp_to_edge == MirrorClampToEdge::KhrExtension {
            out.push(vk::KHR_SAMPLER_MIRROR_CLAMP_TO_EDGE_NAME.as_ptr());
        }
        out
    }
}

/// Resolve every feature this backend depends on against one physical device.
///
/// `has_extension` answers whether the device advertises a given extension; the
/// caller already enumerates them for the interop rails, so it is passed in
/// rather than enumerated twice.
///
/// # Safety
///
/// `pd` must be a physical device belonging to `instance`.
pub unsafe fn query(
    instance: &ash::Instance,
    pd: vk::PhysicalDevice,
    has_extension: &dyn Fn(&std::ffi::CStr) -> bool,
) -> DeviceFeatures {
    let mut supported_16 = vk::PhysicalDevice16BitStorageFeatures::default();
    let mut supported_8 = vk::PhysicalDevice8BitStorageFeatures::default();
    let mut supported_f16i8 = vk::PhysicalDeviceShaderFloat16Int8Features::default();
    let mut supported_vulkan12 = vk::PhysicalDeviceVulkan12Features::default();
    let mut features2 = vk::PhysicalDeviceFeatures2::default()
        .push_next(&mut supported_16)
        .push_next(&mut supported_8)
        .push_next(&mut supported_f16i8)
        .push_next(&mut supported_vulkan12);
    unsafe { instance.get_physical_device_features2(pd, &mut features2) };
    let supported = features2.features;
    let props = unsafe { instance.get_physical_device_properties(pd) };

    // BGRA8 as a storage image is optional; ask the device rather than assume.
    let bgra8_storage = unsafe {
        instance.get_physical_device_format_properties(
            pd,
            crate::backend::vulkan::translate::pixel::SCANOUT_FORMAT,
        )
    }
    .optimal_tiling_features
    .contains(vk::FormatFeatureFlags::STORAGE_IMAGE);

    // Prefer the 1.2 core feature over the extension: it needs no extension
    // string and it is the spelling the baseline guarantees exists to ask about.
    let mirror_clamp_to_edge = if supported_vulkan12.sampler_mirror_clamp_to_edge == vk::TRUE {
        MirrorClampToEdge::Core12
    } else if has_extension(vk::KHR_SAMPLER_MIRROR_CLAMP_TO_EDGE_NAME) {
        MirrorClampToEdge::KhrExtension
    } else {
        MirrorClampToEdge::Unsupported
    };

    DeviceFeatures {
        robust_buffer_access: supported.robust_buffer_access == vk::TRUE,
        sampler_anisotropy: supported.sampler_anisotropy == vk::TRUE,
        max_sampler_anisotropy: props.limits.max_sampler_anisotropy.max(1.0),
        shader_int16: supported.shader_int16 == vk::TRUE,
        storage_image_extended_formats: supported.shader_storage_image_extended_formats == vk::TRUE,
        storage_image_write_without_format: supported.shader_storage_image_write_without_format
            == vk::TRUE,
        bgra8_storage,
        storage16: supported_16.storage_buffer16_bit_access == vk::TRUE,
        storage8: supported_8.storage_buffer8_bit_access == vk::TRUE,
        float16: supported_f16i8.shader_float16 == vk::TRUE,
        int8: supported_f16i8.shader_int8 == vk::TRUE,
        shader_output_viewport_index: supported_vulkan12.shader_output_viewport_index == vk::TRUE,
        mirror_clamp_to_edge,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all_supported() -> DeviceFeatures {
        DeviceFeatures {
            robust_buffer_access: true,
            sampler_anisotropy: true,
            max_sampler_anisotropy: 16.0,
            shader_int16: true,
            storage_image_extended_formats: true,
            storage_image_write_without_format: true,
            bgra8_storage: true,
            storage16: true,
            storage8: true,
            float16: true,
            int8: true,
            shader_output_viewport_index: true,
            mirror_clamp_to_edge: MirrorClampToEdge::Core12,
        }
    }

    /// The 1.2 rung sets the core feature bit and asks for no extension.
    #[test]
    fn the_core_rung_needs_no_extension_string() {
        let caps = all_supported();
        assert_eq!(
            caps.enabled_vulkan12().sampler_mirror_clamp_to_edge,
            vk::TRUE
        );
        assert!(caps.required_extensions().is_empty());
    }

    /// The extension rung is the mirror image: extension string, no core bit.
    /// Setting both would request a feature the device may not expose in 1.2
    /// form, which is how a "belt and braces" enable becomes a device-creation
    /// failure on the driver that only has the extension.
    #[test]
    fn the_extension_rung_asks_for_the_extension_and_not_the_core_bit() {
        let caps = DeviceFeatures {
            mirror_clamp_to_edge: MirrorClampToEdge::KhrExtension,
            ..all_supported()
        };
        assert_eq!(
            caps.enabled_vulkan12().sampler_mirror_clamp_to_edge,
            vk::FALSE
        );
        assert_eq!(caps.required_extensions().len(), 1);
    }

    /// Neither rung: nothing is requested. The sampler path must decline the
    /// address mode by name rather than bind it — the whole point of the enum
    /// having a third state instead of being a bool.
    #[test]
    fn without_support_nothing_is_requested() {
        let caps = DeviceFeatures {
            mirror_clamp_to_edge: MirrorClampToEdge::Unsupported,
            ..all_supported()
        };
        assert_eq!(
            caps.enabled_vulkan12().sampler_mirror_clamp_to_edge,
            vk::FALSE
        );
        assert!(caps.required_extensions().is_empty());
        assert!(!caps.mirror_clamp_to_edge.is_available());
    }

    /// A feature the device declines is never enabled — the enable list is a
    /// derivation, not a wish.
    #[test]
    fn unsupported_features_are_not_enabled() {
        let caps = DeviceFeatures::default();
        let enabled = caps.enabled_features();
        assert_eq!(enabled.robust_buffer_access, vk::FALSE);
        assert_eq!(enabled.sampler_anisotropy, vk::FALSE);
        assert_eq!(enabled.shader_int16, vk::FALSE);
        assert_eq!(enabled.shader_storage_image_extended_formats, vk::FALSE);
    }

    /// The BGRA composite path needs BOTH halves. Naming the pair once is what
    /// stops a call site checking only the feature and binding a format the
    /// device does not support as a storage image.
    #[test]
    fn the_bgra_storage_path_needs_both_halves() {
        let both = all_supported();
        assert!(both.storage_image_write_without_format_bgra());
        let no_format = DeviceFeatures {
            bgra8_storage: false,
            ..all_supported()
        };
        assert!(!no_format.storage_image_write_without_format_bgra());
        let no_feature = DeviceFeatures {
            storage_image_write_without_format: false,
            ..all_supported()
        };
        assert!(!no_feature.storage_image_write_without_format_bgra());
    }

    /// The enable list is derived from what the backend binds, and
    /// `multi_viewport` is the case that proves it.
    ///
    /// It used to be enabled wherever supported while `engine::exec` declines
    /// any draw carrying more than one viewport. Harmless in itself, but it
    /// meant the list was a wish rather than a derivation — and a list that is
    /// not derived cannot be checked. Asserted against the produced struct, not
    /// the source text: a source scan for the setter would match this test's
    /// own assertion.
    #[test]
    fn multi_viewport_is_not_enabled_while_the_engine_declines_it() {
        let enabled = all_supported().enabled_features();
        assert_eq!(
            enabled.multi_viewport,
            vk::FALSE,
            "no draw can use a second viewport, so nothing should request one"
        );

        let exec = std::fs::read_to_string(
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("src/backend/vulkan/engine/exec.rs"),
        )
        .expect("read exec.rs");
        assert!(
            exec.contains("req.viewports.len() > 1"),
            "engine::exec no longer declines multi-viewport draws — if it now \
             binds several, the enable list here has to follow"
        );
    }
}
