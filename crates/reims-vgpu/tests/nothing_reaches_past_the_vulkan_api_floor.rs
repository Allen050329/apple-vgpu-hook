//! Nothing in this crate reaches a Vulkan 1.3 core entry point.
//!
//! `caps/api_floor.rs` makes a claim the whole support matrix rests on: **the
//! engine requires nothing above Vulkan 1.2 core on any pathway**, so a device
//! is accepted or declined on one floor check rather than sorted into version
//! tiers. It also says a capability promoted into 1.3 core "must be reached
//! through its `KHR`/`EXT` form, gated on runtime presence, with the 1.2 path
//! still implemented and tested", and credits `super::gate` with failing the
//! build otherwise. That module was deleted in `db80389`; the claim has been
//! unenforced since.
//!
//! It is not a style rule. `MAX_USEFUL_API` asks the *instance* for 1.3, and the
//! loader on a modern host will happily resolve a 1.3 entry point — so a call
//! added here works on the developer's machine and returns a null function
//! pointer on the 1.2 host the matrix promises to support. The failure is at
//! `vkGetDeviceProcAddr` time, far from the code that caused it.
//!
//! # What this covers, and what it does not
//!
//! It scans for **1.3-core entry points and structs whose names ash does not
//! also give a `KHR`/`EXT` spelling** — the promotions this engine could
//! plausibly reach, listed below with the extension form each must go through
//! instead. It is deliberately not the whole 1.3 promotion set: for the
//! `synchronization2` family ash makes `PipelineStageFlags2KHR` a type alias of
//! `PipelineStageFlags2`, so the two spellings are the same token and a text
//! scan cannot separate correct extension use from 1.3-core use. Those are
//! caught by the entry points instead, which do live on different objects —
//! `ash::Device::cmd_pipeline_barrier2` is 1.3 core, while the extension form is
//! a method on `ash::khr::synchronization2::Device`.
//!
//! So: a hit here is always a real violation, a miss is not proof of purity for
//! the alias families, and adopting any new Vulkan feature means adding its
//! promoted names to [`PROMOTED_TO_1_3`].

use std::collections::BTreeMap;

mod source_scan;
use source_scan::{blank_comments, rust_sources, workspace_root};

/// Vulkan 1.3 core names, and the pre-1.3 form each must be reached through.
///
/// Entry points are listed as their ash method names; a `.name(` spelling in
/// the crate is a call on `ash::Device`, which is the 1.3-core dispatch table.
const PROMOTED_TO_1_3: &[(&str, &str)] = &[
    // VK_KHR_dynamic_rendering
    ("cmd_begin_rendering", "khr::dynamic_rendering"),
    ("cmd_end_rendering", "khr::dynamic_rendering"),
    ("PipelineRenderingCreateInfo", "khr::dynamic_rendering"),
    ("RenderingAttachmentInfo", "khr::dynamic_rendering"),
    // VK_KHR_synchronization2
    ("cmd_pipeline_barrier2", "khr::synchronization2"),
    ("cmd_set_event2", "khr::synchronization2"),
    ("cmd_wait_events2", "khr::synchronization2"),
    ("queue_submit2", "khr::synchronization2"),
    // VK_KHR_copy_commands2
    ("cmd_blit_image2", "khr::copy_commands2"),
    ("cmd_copy_buffer2", "khr::copy_commands2"),
    ("cmd_copy_buffer_to_image2", "khr::copy_commands2"),
    ("cmd_copy_image2", "khr::copy_commands2"),
    ("cmd_copy_image_to_buffer2", "khr::copy_commands2"),
    ("cmd_resolve_image2", "khr::copy_commands2"),
    // VK_EXT_extended_dynamic_state / _2
    ("cmd_set_cull_mode", "ext::extended_dynamic_state"),
    (
        "cmd_set_depth_bounds_test_enable",
        "ext::extended_dynamic_state",
    ),
    ("cmd_set_depth_compare_op", "ext::extended_dynamic_state"),
    ("cmd_set_depth_test_enable", "ext::extended_dynamic_state"),
    ("cmd_set_depth_write_enable", "ext::extended_dynamic_state"),
    ("cmd_set_front_face", "ext::extended_dynamic_state"),
    ("cmd_set_primitive_topology", "ext::extended_dynamic_state"),
    ("cmd_set_scissor_with_count", "ext::extended_dynamic_state"),
    ("cmd_set_stencil_op", "ext::extended_dynamic_state"),
    ("cmd_set_viewport_with_count", "ext::extended_dynamic_state"),
    (
        "cmd_set_primitive_restart_enable",
        "ext::extended_dynamic_state2",
    ),
    (
        "cmd_set_rasterizer_discard_enable",
        "ext::extended_dynamic_state2",
    ),
    ("cmd_set_depth_bias_enable", "ext::extended_dynamic_state2"),
    // VK_EXT_private_data
    ("create_private_data_slot", "ext::private_data"),
    ("destroy_private_data_slot", "ext::private_data"),
    ("PrivateDataSlot", "ext::private_data"),
    // VK_KHR_maintenance4
    ("get_device_buffer_memory_requirements", "khr::maintenance4"),
    ("get_device_image_memory_requirements", "khr::maintenance4"),
    // VK_KHR_format_feature_flags2
    ("FormatProperties3", "khr::format_feature_flags2"),
    // VK_EXT_tooling_info
    ("get_physical_device_tool_properties", "ext::tooling_info"),
    // The 1.3 feature aggregate itself. Reaching for it is the tier thinking
    // `api_floor` deliberately does not do.
    (
        "PhysicalDeviceVulkan13Features",
        "the individual extensions",
    ),
    (
        "PhysicalDeviceVulkan13Properties",
        "the individual extensions",
    ),
];

#[test]
fn no_vulkan_1_3_core_name_appears_in_the_crate() {
    let root = workspace_root();
    let src = root.join("crates/reims-vgpu/src");
    let sources = rust_sources(&src);
    assert!(
        sources.len() > 50,
        "walked {} files, which is not this crate",
        sources.len()
    );

    // Prove the scanner can see a Vulkan name at all before believing it saw no
    // forbidden one. `cmd_pipeline_barrier` — the 1.0 spelling of the entry
    // point whose `2` suffix is on the list — is called in the engine, so its
    // absence means the scan is not reading the code it thinks it is.
    let mut saw_the_1_0_barrier = false;
    let mut hits: BTreeMap<&str, Vec<String>> = BTreeMap::new();

    for path in &sources {
        let text = blank_comments(&std::fs::read_to_string(path).expect("readable source"));
        let rel = path
            .strip_prefix(&root)
            .unwrap_or(path)
            .to_string_lossy()
            .into_owned();
        if contains_token(&text, "cmd_pipeline_barrier") {
            saw_the_1_0_barrier = true;
        }
        for (name, _) in PROMOTED_TO_1_3 {
            if contains_token(&text, name) {
                hits.entry(name).or_default().push(rel.clone());
            }
        }
    }

    assert!(
        saw_the_1_0_barrier,
        "the scan found no call to `cmd_pipeline_barrier`, which the engine \
         makes on every submission — so it is not reading the engine and its \
         verdict on 1.3 names means nothing"
    );

    let report: Vec<String> = hits
        .iter()
        .map(|(name, files)| {
            let through = PROMOTED_TO_1_3
                .iter()
                .find(|(n, _)| n == name)
                .map(|(_, e)| *e)
                .unwrap_or("its extension form");
            format!("`{name}` in {files:?} — reach it through `{through}`")
        })
        .collect();
    assert!(
        report.is_empty(),
        "these names exist only at Vulkan 1.3 core, and `caps::api_floor` \
         promises the engine requires nothing above 1.2. On a 1.2 host the \
         loader returns a null function pointer for each:\n  {}",
        report.join("\n  ")
    );
}

/// Whether `name` appears in `text` as a whole identifier.
///
/// Substring matching would report `cmd_copy_image2` for
/// `cmd_copy_image2_khr`, and `cmd_set_stencil_op` for a longer name that
/// merely starts the same way.
fn contains_token(text: &str, name: &str) -> bool {
    let bytes = text.as_bytes();
    let mut at = 0usize;
    while let Some(rel) = text[at..].find(name) {
        let start = at + rel;
        let end = start + name.len();
        at = start + 1;
        let before_ok = start == 0 || !is_ident(bytes[start - 1]);
        let after_ok = end >= bytes.len() || !is_ident(bytes[end]);
        if before_ok && after_ok {
            return true;
        }
    }
    false
}

fn is_ident(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}
