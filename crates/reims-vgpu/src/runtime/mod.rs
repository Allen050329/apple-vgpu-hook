//! Behavior: turn guest work into HostActions / backend jobs.
//!
//! Drain FIFOs, parse wire (using [`crate::contract`]), resolve memory, plan
//! ops, update [`crate::model`] state. No GPU API calls here.

/// The split of [`chain_phase`]'s largest column, `binds_us`.
pub mod bind_phase;
/// Product-path blit fill/copy execution against guest GVA.
pub mod blit_exec;
/// Always-on proxies and censuses, one per measured bug class.
pub mod census;
/// Where a draw chain's wall clock goes on the runtime side of the engine
/// boundary, which is 82% of it.
pub mod chain_phase;
/// Product-path compute bind/dispatch (pipeline + buffers + direct dispatch).
pub mod compute_exec;
/// Multi-record compute encoder session (control-flow SPI + ICB execute).
pub mod compute_session;
pub mod decode;
pub mod drain;
/// The always-on log sink every decline and census writes to
/// (`/tmp/reims-vgpu-fail.log`); `line()` is the `REIMS_VGPU_DRAW_LOG=1`-gated tier.
/// CmdExecIndirect2 stream walk + type-11 resolve.
pub mod exec;
/// Product-path event + encoder fence sync (event/blit/compute/render domains).
pub mod fence_exec;
/// Is the hypervisor's guest-write generation a sound cache key for the
/// zero-copy sampled gathers? Measurement, not policy.
#[cfg(feature = "backend-vulkan")]
pub mod gather_witness;
/// Guest-physical control-plane writes via HostOps map_pages.
pub mod gpa_map;
/// Task GVA → guest RAM reads.
pub mod gva_mem;
/// Task-GVA HostOps views (MapMemory2 / UnmapMemory lifecycle).
pub mod gva_view;
/// CmdHeapTextureSizeAndAlign wire decode + host requirement query.
pub mod heap_query;
pub mod host;
/// Which guest pages this device has written, and when — the half of the
/// guest-write witness the hypervisor's dirty bitmap cannot supply.
pub mod host_writes;
/// Type-7 ICB (0x36) materialization, host command fills, execute writeback.
pub mod icb;

pub mod input;
/// Process-global metal2vulkan SPIR-V cache (AIR content hash → SPIR-V).
pub mod m2v_cache;
/// IOSurface mapper capture + page-table resolve.
pub mod mapper;
/// Write host BGRA into guest mapping pages (render writeback).
pub mod mapping_write;
/// Metal draw encode + writeback when MTLBs resolve.
pub mod metal_draw;
/// generateMipmaps for multi-mip type-2/3 linear textures.
pub mod mipmap;
pub mod mmio;
/// MTLB container → wrapped-AIR carve for metal2vulkan.
pub mod mtlb;
/// Object-list lookup and type-11 registration.
pub mod objects;
pub mod plan;
/// The resident identity a type-11 guest surface renders into.
#[cfg(feature = "backend-vulkan")]
pub mod present_identity;
/// The guest's per-resource validity quad, from both of its producers.
pub mod resource_validity;
/// Guest surface → host BGRA8 for the QEMU console.
pub mod scanout;
/// SPIR-V set-0 binding relocation for metal2vulkan + internal Vulkan engine (Linux).
pub mod spirv_bind;
mod spirv_layout;
/// Bounded structural evaluation of vertex clip positions (coverage proof).
/// Deferred compute-writeback flush (flush-on-access; resident authoritative).
pub mod storage_flush;
/// Host surface cache (Linux/Vulkan discrete-GPU present, kb §8.5).
pub mod surface_cache;
/// The wire task word a command payload carries → a live task slot.
pub mod task_slot;
/// Texture / type-11 geometry registration.
pub mod texture;

pub use drain::{
    drain_child_fifo, drain_main_fifo, drain_other_child_fifos, drain_pending, signal_display_vbl,
    write_stamp, Packet, PacketError,
};
pub use host::{read_u32, HostAction, HostActionKind, HostMemory, HostOps, MemError};
/// The unit-test host double, gated with its definition. An ungated re-export
/// would keep it reachable and so keep it in the staticlib.
#[cfg(test)]
pub use host::FakeHost;

#[cfg(test)]
mod arch_path_gate {
    //! Product runtime must never call hard-coded arch page helpers:
    //! - `*_arm64*` (covers `_arm64` / `_arm64e`) — wrong on x86
    //! - `*_x86*` (covers `_x86` / `_x86_64`) — wrong on arm64/arm64e
    //!
    //! Portable product code uses `*_shift` / `state.page_shift` only.
    //! Arch-hardcoded wrappers stay in `contract` + unit fixtures.
    use std::fs;
    use std::path::PathBuf;

    fn product_lines(src: &str) -> impl Iterator<Item = (usize, &str)> + '_ {
        let mut past_test = false;
        let mut pending_cfg_test = false;
        let mut test_depth: i32 = 0;
        src.lines().enumerate().filter_map(move |(i, line)| {
            let t = line.trim();
            if t.contains("#[cfg(test)]") {
                pending_cfg_test = true;
                return None;
            }
            if pending_cfg_test {
                if t.starts_with("mod ")
                    || t.starts_with("fn ")
                    || t.starts_with("pub mod ")
                    || t.starts_with("pub fn ")
                {
                    past_test = true;
                    // Count braces on the same line as the mod/fn (opening `{`).
                    test_depth = line.chars().filter(|&c| c == '{').count() as i32
                        - line.chars().filter(|&c| c == '}').count() as i32;
                    pending_cfg_test = false;
                    return None;
                } else if t.starts_with("#[") {
                    return None;
                } else {
                    pending_cfg_test = false;
                }
            }
            if past_test {
                test_depth += line.chars().filter(|&c| c == '{').count() as i32;
                test_depth -= line.chars().filter(|&c| c == '}').count() as i32;
                if test_depth <= 0 {
                    past_test = false;
                }
                return None;
            }
            Some((i + 1, line))
        })
    }

    fn is_ident_start(c: char) -> bool {
        c.is_ascii_alphabetic() || c == '_'
    }

    fn is_ident_cont(c: char) -> bool {
        c.is_ascii_alphanumeric() || c == '_'
    }

    /// Identifiers containing `needle` that are called (`name(`), outside strings.
    /// Case-sensitive (Rust snake_case helpers use lowercase `_arm64` / `_x86`).
    fn arch_fn_calls(line: &str, needle: &str) -> Vec<String> {
        let mut out = Vec::new();
        let mut i = 0;
        let chars: Vec<char> = line.chars().collect();
        let mut in_string = false;
        while i < chars.len() {
            let c = chars[i];
            if c == '"' && (i == 0 || chars[i - 1] != '\\') {
                in_string = !in_string;
                i += 1;
                continue;
            }
            if in_string {
                i += 1;
                continue;
            }
            if is_ident_start(c) {
                let start = i;
                i += 1;
                while i < chars.len() && is_ident_cont(chars[i]) {
                    i += 1;
                }
                let name: String = chars[start..i].iter().collect();
                if name.contains(needle) {
                    let mut j = i;
                    while j < chars.len() && chars[j].is_whitespace() {
                        j += 1;
                    }
                    if j < chars.len() && chars[j] == '(' {
                        out.push(name);
                    }
                }
            } else {
                i += 1;
            }
        }
        out
    }

    fn product_runtime_calls_matching(needle: &str) -> Vec<String> {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/runtime");
        let mut offenders = Vec::new();
        for path in walkdir(&root) {
            if path.extension().and_then(|e| e.to_str()) != Some("rs") {
                continue;
            }
            if is_out_of_line_test_module(&path) {
                continue;
            }
            let src = fs::read_to_string(&path).expect("read");
            for (lineno, line) in product_lines(&src) {
                let t = line.trim();
                if t.starts_with("//") {
                    continue;
                }
                for name in arch_fn_calls(line, needle) {
                    offenders.push(format!("{}:{}: call {name}( — {t}", path.display(), lineno));
                }
            }
        }
        offenders
    }

    /// An out-of-line module declared under `#[cfg(test)]` is wholly test code.
    ///
    /// Out-of-line fixture tails in `tests.rs` are test code; scanning them as
    /// product would turn architecture-qualified fixture names into false
    /// product-path violations. Resolve the declaration from the parent module
    /// instead of exempting a hard-coded filename list.
    fn is_out_of_line_test_module(path: &std::path::Path) -> bool {
        let Some(module_name) = path.file_stem().and_then(|name| name.to_str()) else {
            return false;
        };
        let Some(parent) = path.parent() else {
            return false;
        };
        let Ok(parent_source) = fs::read_to_string(parent.join("mod.rs")) else {
            return false;
        };
        let compact: String = parent_source
            .chars()
            .filter(|ch| !ch.is_whitespace())
            .collect();
        compact.contains(&format!("#[cfg(test)]mod{module_name};"))
    }

    #[test]
    fn product_runtime_never_calls_arm64_page_helpers() {
        let offenders = product_runtime_calls_matching("_arm64");
        assert!(
            offenders.is_empty(),
            "product runtime must use *_shift / state.page_shift, not *_arm64* helpers:\n{}",
            offenders.join("\n")
        );
    }

    #[test]
    fn product_runtime_never_calls_x86_page_helpers() {
        // Symmetric to *_arm64*: arm product paths must not call hard-coded x86 helpers.
        let offenders = product_runtime_calls_matching("_x86");
        assert!(
            offenders.is_empty(),
            "product runtime must use *_shift / state.page_shift, not *_x86* helpers:\n{}",
            offenders.join("\n")
        );
    }

    #[test]
    fn arch_call_pattern_matches_arm64_and_x86_variants() {
        assert_eq!(
            arch_fn_calls("let x = plan_span_arm64e(&m, &t, 0, 1);", "_arm64"),
            vec!["plan_span_arm64e".to_string()]
        );
        assert_eq!(
            arch_fn_calls("foo_arm64(1); bar_arm64e_extra(2);", "_arm64"),
            vec!["foo_arm64".to_string(), "bar_arm64e_extra".to_string()]
        );
        assert_eq!(
            arch_fn_calls("entry_gpa_x86(e); plan_span_x86_64(&m);", "_x86"),
            vec!["entry_gpa_x86".to_string(), "plan_span_x86_64".to_string()]
        );
        // Constants / non-calls are fine.
        assert!(arch_fn_calls("let s = PAGE_SHIFT_ARM64E;", "_arm64").is_empty());
        assert!(arch_fn_calls("let s = PAGE_SHIFT_X86;", "_x86").is_empty());
        assert!(arch_fn_calls(r#"let s = "plan_span_arm64e(";"#, "_arm64").is_empty());
        assert!(arch_fn_calls(r#"let s = "entry_gpa_x86(";"#, "_x86").is_empty());
        // Portable shift APIs are fine.
        assert!(arch_fn_calls("plan_span_shift(&m, &t, 0, 1, 12);", "_arm64").is_empty());
        assert!(arch_fn_calls("plan_span_shift(&m, &t, 0, 1, 12);", "_x86").is_empty());
        // Prefix-style x86_kernel_va is not `*_x86*` (no underscore before x86).
        assert!(arch_fn_calls("x86_kernel_va(a);", "_x86").is_empty());

        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/runtime");
        for relative in [
            "metal_draw/tests.rs",
            "compute_exec/tests.rs",
            "drain/tests.rs",
            "icb/tests.rs",
        ] {
            assert!(
                is_out_of_line_test_module(&root.join(relative)),
                "{relative} must be recognized through its parent's cfg(test) declaration"
            );
        }
        assert!(!is_out_of_line_test_module(&root.join("metal_draw/mod.rs")));
    }

    fn walkdir(dir: &std::path::Path) -> Vec<PathBuf> {
        let mut out = Vec::new();
        let mut stack = vec![dir.to_path_buf()];
        while let Some(d) = stack.pop() {
            if let Ok(rd) = fs::read_dir(&d) {
                for e in rd.flatten() {
                    let p = e.path();
                    if p.is_dir() {
                        stack.push(p);
                    } else {
                        out.push(p);
                    }
                }
            }
        }
        out
    }
}
