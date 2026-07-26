//! Source-level gates that keep the four-cell support matrix from eroding.
//!
//! Every invariant below currently holds with zero exceptions. They are cheap
//! to break by accident — one new allocation with hardcoded flags, one new `if
//! portability_subset`, one `synchronization2` barrier — and expensive to
//! notice, because breaking any of them degrades a matrix row on a host nobody
//! in this project has in front of them. Scanning source is crude, but it fails
//! at `cargo test` time on the machine that made the change rather than on
//! someone else's GPU months later.

use std::fs;
use std::path::{Path, PathBuf};

fn crate_src() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src")
}

fn rust_files(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(entries) = fs::read_dir(&d) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
                out.push(path);
            }
        }
    }
    out.sort();
    out
}

/// Repo-relative path with forward slashes, for stable allowlists and messages.
fn rel(path: &Path, root: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

/// Files allowed to name `portability_subset`.
///
/// This is a DETECTION-site allowlist, not a permission to branch. Both entries
/// query the extension and hand the answer straight to
/// [`super::DriverQuirk::for_portability_subset`]; neither makes a decision
/// from it. A third file appearing here means someone is about to re-introduce
/// driver-identity gating — add the named quirk to `DriverQuirk` instead.
const PORTABILITY_DETECTION_SITES: &[&str] =
    &["backend/vulkan/engine/context.rs", "host_window/present.rs"];

/// Vulkan 1.3 core feature structs and promoted entry points.
///
/// The support matrix's baseline is Vulkan **1.2** on all four cells, so none
/// of these may appear in the crate. A capability promoted into 1.3 core must be
/// reached through its `KHR`/`EXT` form, gated on runtime presence, with the 1.2
/// path still implemented and tested — otherwise the baseline is 1.3 in fact
/// while claiming 1.2 in the docs, and the first host to notice is a user's.
const VULKAN_13_CORE_SYMBOLS: &[&str] = &[
    "PhysicalDeviceVulkan13Features",
    "PhysicalDeviceDynamicRenderingFeatures",
    "PhysicalDeviceSynchronization2Features",
    "cmd_begin_rendering",
    "cmd_end_rendering",
    "RenderingInfo",
    "cmd_pipeline_barrier2",
    "DependencyInfo",
    "MemoryBarrier2",
    "PipelineStageFlags2",
    "AccessFlags2",
    "queue_submit2",
    "SubmitInfo2",
    "API_VERSION_1_3",
];

#[cfg(test)]
mod tests {
    use super::*;

    /// Memory property flags belong to the topology policy, nowhere else.
    ///
    /// An allocation site that spells `HOST_VISIBLE | HOST_COHERENT` itself
    /// bypasses the unified/discrete preference entirely — it will work on
    /// every host and be needlessly slow on half of them, which is exactly the
    /// failure mode that is invisible without this gate. Call sites name a
    /// [`super::MemoryClass`]; only `memory_topology` turns that into flags.
    #[test]
    fn memory_property_flags_are_named_only_by_the_topology_policy() {
        let root = crate_src();
        let mut offenders = Vec::new();
        for path in rust_files(&root) {
            let name = rel(&path, &root);
            if name.starts_with("backend/vulkan/caps/") {
                continue;
            }
            let Ok(src) = fs::read_to_string(&path) else {
                continue;
            };
            for (i, line) in src.lines().enumerate() {
                if line.contains("MemoryPropertyFlags::") {
                    offenders.push(format!("{name}:{}: {}", i + 1, line.trim()));
                }
            }
        }
        assert!(
            offenders.is_empty(),
            "allocation sites must name a MemoryClass, not raw memory property \
             flags — route these through caps::memory_topology so the \
             unified/discrete rows of the support matrix stay honest:\n{}",
            offenders.join("\n")
        );
    }

    /// A device feature is queried and enabled in `caps/`, or nowhere.
    ///
    /// `context.rs` used to query these inline, correctly in every case but
    /// one: `translate::sampler` emitted `MIRROR_CLAMP_TO_EDGE` while nothing
    /// anywhere requested `samplerMirrorClampToEdge` or its `KHR` extension, so
    /// samplers were created with a mode the device had not been asked for.
    /// The bug is not that the inline queries were wrong — it is that there was
    /// no home, so the one that got missed got missed silently.
    ///
    /// Naming a feature struct outside `caps/` is how that recurs: a site that
    /// asks whether a feature is supported without also being the site that
    /// enables it will one day answer yes for a feature nobody requested.
    #[test]
    fn device_feature_structs_are_named_only_in_caps() {
        const FEATURE_STRUCTS: &[&str] = &[
            "PhysicalDeviceFeatures",
            "PhysicalDeviceVulkan12Features",
            "PhysicalDeviceVulkan11Features",
            "PhysicalDevice16BitStorageFeatures",
            "PhysicalDevice8BitStorageFeatures",
            "PhysicalDeviceShaderFloat16Int8Features",
        ];
        let root = crate_src();
        let mut offenders = Vec::new();
        for path in rust_files(&root) {
            let name = rel(&path, &root);
            if name.starts_with("backend/vulkan/caps/") {
                continue;
            }
            let Ok(src) = fs::read_to_string(&path) else {
                continue;
            };
            for (i, line) in src.lines().enumerate() {
                let t = line.trim();
                if t.starts_with("//") || t.starts_with("///") {
                    continue;
                }
                // `PhysicalDeviceFeatures2` is the *query* container and is
                // distinct from the feature structs; matching it here would ban
                // the struct name by prefix. Check for a non-digit after.
                for s in FEATURE_STRUCTS {
                    let Some(at) = t.find(s) else { continue };
                    let after = t[at + s.len()..].chars().next();
                    if after.is_some_and(|c| c.is_ascii_digit()) {
                        continue;
                    }
                    offenders.push(format!("{name}:{}: {t}", i + 1));
                }
            }
        }
        assert!(
            offenders.is_empty(),
            "device features are queried and enabled in caps::device_features, \
             or not at all — a site that asks whether a feature is supported \
             without being the site that enables it is how \
             samplerMirrorClampToEdge came to be bound ungated:\n{}",
            offenders.join("\n")
        );
    }

    /// Driver identity may be DETECTED in exactly the two device-create sites
    /// and must be consumed only as a named quirk.
    #[test]
    fn driver_identity_is_confined_to_the_detection_sites() {
        let root = crate_src();
        let mut offenders = Vec::new();
        for path in rust_files(&root) {
            let name = rel(&path, &root);
            if name.starts_with("backend/vulkan/caps/")
                || PORTABILITY_DETECTION_SITES.contains(&name.as_str())
            {
                continue;
            }
            let Ok(src) = fs::read_to_string(&path) else {
                continue;
            };
            for (i, line) in src.lines().enumerate() {
                if line.contains("portability_subset") || line.contains("PORTABILITY_SUBSET") {
                    offenders.push(format!("{name}:{}: {}", i + 1, line.trim()));
                }
            }
        }
        assert!(
            offenders.is_empty(),
            "behavior must not key on VK_KHR_portability_subset outside the \
             device-create detection sites — add a named caps::DriverQuirk \
             documenting the failure it works around:\n{}",
            offenders.join("\n")
        );
    }

    /// Vulkan 1.2 is the baseline on ALL FOUR matrix cells — nothing may
    /// require a 1.3 core feature.
    ///
    /// This is the gate that replaced the old `ApiFloor::Vk13` tier. The tier
    /// enum made "we might use 1.3 one day" expressible in the type system and
    /// invisible in practice; a scan makes the real property — that no 1.3 core
    /// symbol is reachable — checkable. `caps/` is exempt because that is where
    /// the negotiation ceiling legitimately names 1.3.
    #[test]
    fn vulkan_13_core_features_are_never_required() {
        let root = crate_src();
        let mut offenders = Vec::new();
        for path in rust_files(&root) {
            let name = rel(&path, &root);
            if name.starts_with("backend/vulkan/caps/") {
                continue;
            }
            let Ok(src) = fs::read_to_string(&path) else {
                continue;
            };
            for (i, line) in src.lines().enumerate() {
                for symbol in VULKAN_13_CORE_SYMBOLS {
                    if line.contains(symbol) {
                        offenders.push(format!("{name}:{}: {}", i + 1, line.trim()));
                    }
                }
            }
        }
        assert!(
            offenders.is_empty(),
            "the support matrix's baseline is Vulkan 1.2 on every cell — reach \
             this capability through its KHR/EXT form gated on runtime \
             presence, and keep the 1.2 path implemented:\n{}",
            offenders.join("\n")
        );
    }

    /// The allowlist itself must stay minimal and must point at files that
    /// exist — a stale entry silently widens the gate.
    #[test]
    fn detection_site_allowlist_is_exact() {
        let root = crate_src();
        assert_eq!(
            PORTABILITY_DETECTION_SITES.len(),
            2,
            "one entry per Vulkan device this crate creates (engine + host window)"
        );
        for site in PORTABILITY_DETECTION_SITES {
            let path = root.join(site);
            assert!(path.exists(), "allowlisted site {site} no longer exists");
            let src = fs::read_to_string(&path).expect("read allowlisted site");
            assert!(
                src.contains("DriverQuirk::for_portability_subset"),
                "{site} names portability_subset but does not hand it to \
                 DriverQuirk — it is deciding something itself"
            );
        }
    }

    /// The scanner sees the whole crate, not an empty directory — a gate that
    /// silently inspects nothing always passes.
    #[test]
    fn scanner_actually_walks_the_crate() {
        let files = rust_files(&crate_src());
        assert!(
            files.len() > 50,
            "expected the full crate, saw {}",
            files.len()
        );
        let root = crate_src();
        let names: Vec<_> = files.iter().map(|p| rel(p, &root)).collect();
        assert!(names.contains(&"lib.rs".to_string()));
        assert!(names.contains(&"backend/vulkan/engine/context.rs".to_string()));
        assert!(names.contains(&"backend/vulkan/caps/memory_topology.rs".to_string()));
    }
}
