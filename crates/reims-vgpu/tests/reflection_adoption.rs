//! Corpus + end-to-end guards for the metal2vulkan reflection adoption.
//!
//! reims-vgpu now reads texture dimensionality and sampled-vs-storage access
//! solely from the AIR-derived `metal2vulkan::reflect::ShaderReflection` — there
//! is no second SPIR-V walk to cross-check against. These prove, on real Apple
//! AIR shaders, that the reflection the product path trusts is (a) internally
//! well-formed (`census_reflection_wellformed`, the same guard the live guest
//! runs per translate) and (b) complete: every texture binding classifies to a
//! representable sampled kind, never `Absent`/`Unsupported`.
//!
//! Requires the metal2vulkan toolchain (`llvm-dis`) on PATH, same as the
//! translator's own render-capture tests.

use std::path::PathBuf;
use std::process;
use std::sync::atomic::{AtomicUsize, Ordering};

use metal2vulkan::passes::Stage;
use metal2vulkan::reflect::ShaderReflection;
use reims_vgpu::runtime::spirv_bind;

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/air")
}

fn tmpdir() -> PathBuf {
    static N: AtomicUsize = AtomicUsize::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    let d = std::env::temp_dir().join(format!("reims_vgpu_reflect_adopt_{}_{n}", process::id()));
    let _ = std::fs::create_dir_all(&d);
    d
}

fn have_llvm_dis() -> bool {
    process::Command::new("llvm-dis")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn translate_reflected(name: &str, stage: Stage) -> (Vec<u8>, ShaderReflection) {
    let air = fixtures().join(format!("{name}.air"));
    assert!(air.exists(), "missing reims-vgpu AIR fixture: {name}.air");
    let tmp = tmpdir();
    metal2vulkan::translate_reflected(air.to_str().unwrap(), stage, &tmp)
        .unwrap_or_else(|e| panic!("{name} ({stage:?}): reflected translate failed: {e}"))
}

// (fixture stem, stage) — a retained spread of texture-carrying fragment shaders
// plus a couple of vertex shaders that exercise the product reflection edge.
const FIXTURES: &[(&str, Stage)] = &[
    ("render_frag_texture", Stage::Fragment),
    ("render_frag_lighting", Stage::Fragment),
    ("render_frag", Stage::Fragment),
    ("render_frag_color", Stage::Fragment),
    ("render_frag_tint", Stage::Fragment),
    ("textured_quad", Stage::Fragment),
    ("textured_quad", Stage::Vertex),
    ("render_tri", Stage::Vertex),
    ("render_vtx_mesh", Stage::Vertex),
];

/// Across the real-shader corpus the reflection the product path trusts must be
/// (a) internally well-formed — `census_reflection_wellformed` finds zero
/// violations, exactly the per-translate guard on the live guest — and (b)
/// complete: every reflected texture binding classifies to a representable
/// sampled `Kind`, never `Absent` (missing) or `Unsupported` (unrepresentable).
#[test]
fn reflection_is_wellformed_and_complete_for_every_texture_binding() {
    if !have_llvm_dis() {
        eprintln!("skipping: llvm-dis not on PATH");
        return;
    }
    use spirv_bind::ReflectedSampledKind;
    let mut checked_bindings = 0usize;
    for &(name, stage) in FIXTURES {
        let (_spirv, reflection) = translate_reflected(name, stage);

        // The self-consistency guard the product path runs per translate.
        let violations = spirv_bind::census_reflection_wellformed(&reflection, 0);
        assert_eq!(
            violations, 0,
            "{name} ({stage:?}): reflection well-formedness violations"
        );

        for b in &reflection.bindings {
            let (Some(shape), Some(loc)) = (b.texture_shape.as_ref(), b.descriptor) else {
                continue;
            };
            checked_bindings += 1;
            let binding = loc.binding;

            // Sampled textures must classify to a representable dimensionality;
            // a writable (storage) texture is expected to be `Unsupported` for the
            // sampled render path, so only assert completeness for sampled ones.
            if !shape.writable {
                assert_eq!(
                    spirv_bind::reflected_sampled_kind(&reflection, binding),
                    ReflectedSampledKind::Kind(
                        spirv_bind::sampled_image_kind_from_reflection(&reflection, binding)
                            .expect("sampled texture must have a representable kind"),
                    ),
                    "{name} ({stage:?}): sampled texture at binding {binding} \
                     (metal_index {}, shape {shape:?}) did not classify to a kind",
                    b.metal_index
                );
            }
        }
    }
    // The texture-carrying fragments must actually exercise the check.
    assert!(
        checked_bindings > 0,
        "no texture bindings were checked — fixture/reflection wiring is wrong"
    );
}

/// The reflected translate populates the datalayout (the single source of truth
/// the layout repair now consumes) and the stage — proving the toolchain path
/// reims-vgpu's m2v_cache now relies on is live on this host.
#[test]
fn reflected_translate_populates_datalayout_and_stage() {
    if !have_llvm_dis() {
        eprintln!("skipping: llvm-dis not on PATH");
        return;
    }
    let (spirv, reflection) = translate_reflected("render_frag_texture", Stage::Fragment);
    assert!(
        spirv.len() >= 20 && spirv.len() % 4 == 0,
        "invalid SPIR-V length"
    );
    assert_eq!(
        u32::from_le_bytes([spirv[0], spirv[1], spirv[2], spirv[3]]),
        0x0723_0203
    );
    assert!(
        reflection.datalayout.is_some(),
        "reflected translate from AIR must carry the source datalayout"
    );
    assert!(
        reflection.datalayout.as_deref().unwrap().contains('-'),
        "datalayout value looks malformed: {:?}",
        reflection.datalayout
    );
}

/// End-to-end through reims-vgpu's own cache: `translate_cached_reflected`
/// returns byte-identical SPIR-V to the plain `translate_cached`, plus a
/// populated reflection, and a warm second call hits the cache.
#[test]
fn m2v_cache_reflected_matches_plain_bytes_and_caches() {
    if !have_llvm_dis() {
        eprintln!("skipping: llvm-dis not on PATH");
        return;
    }
    use reims_vgpu::runtime::m2v_cache;
    let fixtures = fixtures();
    let air = std::fs::read(fixtures.join("render_frag_texture.air")).unwrap();

    let shader = m2v_cache::translate_cached_reflected(&air, Stage::Fragment, 1).unwrap();
    let plain = m2v_cache::translate_cached(&air, Stage::Fragment, 1).unwrap();
    assert_eq!(
        shader.spirv, plain,
        "reflected and plain bytes must be identical"
    );
    assert_eq!(
        shader.reflection.stage,
        metal2vulkan::reflect::ShaderStage::Fragment
    );
    assert!(shader.reflection.datalayout.is_some());
}
