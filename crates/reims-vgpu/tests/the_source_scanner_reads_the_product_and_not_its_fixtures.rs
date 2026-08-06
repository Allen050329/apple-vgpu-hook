//! The scanner every structural test stands on must measure the product half.
//!
//! Ten integration tests in this directory answer a crate-wide question by
//! reading source text — no two declines share a slug, no Vulkan state enum is
//! spelled outside `translate`, no bound evicts or drops or truncates without a
//! written verdict. Every one of them takes its population from
//! [`source_scan::guest_facing_sources`], and none of them can tell a fixture
//! from a device behaviour once that function has handed it over.
//!
//! `source_scan`'s own header names the hazard: *"getting either wrong is how a
//! scanner reports a clean tree while measuring the wrong half of it."* It had
//! it wrong in both directions and said so in neither.
//!
//! - It matched two literal attribute prefixes, and its doc named
//!   `draw/vulkan.rs`'s `vulkan_split_tests` as the reason the second one
//!   existed. **That prefix never matched anything.** Cutting at the literal
//!   `#[cfg(all(test,` leaves ` feature = "backend-vulkan"))]` between the
//!   marker and the `mod`, and the guard below it allowed only whitespace and
//!   brackets there — so every module written that way fell out through a
//!   `continue`. Seven of them, `vulkan_split_tests` alone 1466 lines.
//! - It blanked only a `mod`, on the stated argument that "only a module has a
//!   body worth blanking". `runtime::host` reaches `FakeHost` — the host under
//!   all of these tests — through `#[cfg(test)] impl HostMemory for FakeHost`
//!   and two more like it, ~690 lines that *emulate a device*, plus three dozen
//!   test-gated free functions elsewhere.
//! - `guest_facing_sources` dropped `tests.rs` by exact name, so
//!   `cap_tests.rs`, `revalidate_tests.rs` and
//!   `render_flush_witness_tests.rs` — 1270 more lines, all three declared
//!   through a `#[cfg(all(test, …))]`, the same gate — came through as product.
//!
//! **3553 non-blank lines**, measured by running the old blanker and the new one
//! over the same tree: 85608 product lines across 195 files, against 82055
//! across 192.
//!
//! # Why this needed a test and not a careful reading
//!
//! Reading too much is the quiet direction. A scanner that hides code goes
//! silent, and somebody eventually notices a question it stopped answering; a
//! scanner that reads fixtures answers confidently about lines that were never
//! the product, and every verdict it extracts looks exactly like a real one.
//! Fixtures shrink, cap and truncate collections constantly — that is what a
//! fixture for a bound scan *is*.
//!
//! It had already happened. `a_bound_is_compared_where_it_is_declared` carried
//! three `Recorded` rows saying `ICB_BUFFER_BIND_STRIDE`,
//! `ICB_CONCURRENT_DISPATCH_ARGS_LEN` and `ICB_TESSELLATION_FACTOR_LEN` were
//! each compared "before two different ICB body reads" — and the second
//! comparison, every time, was inside `write_tessellation_factor` or
//! `encode_render_command_slot`, `#[cfg(test)]` fixture *encoders* that write
//! the slot bytes the real decoders read. Somebody looked at all three, wrote a
//! line about each, and was describing test code. That is the whole argument for
//! this file.
//!
//! # What this asserts
//!
//! Both directions, because a blanker can fail either way and only one of them
//! announces itself: fixtures are gone, and the product line *after* a blanked
//! module is still there. The second is what stops a future widening from
//! swallowing the rest of a file — `close_brace` is what bounds it, and a
//! first-marker cutoff would pass the first assertion and fail this one.

mod source_scan;
use source_scan::guest_facing_sources;

/// The text of one product source, by its `crates/`-relative path.
fn source(rel: &str) -> String {
    guest_facing_sources()
        .into_iter()
        .find(|(f, _)| f == rel)
        .map(|(_, text)| text)
        .unwrap_or_else(|| panic!("{rel} is not in the scanned population"))
}

/// A body gated on `test` is not product code, however the attribute spells it.
#[test]
fn a_test_module_body_is_blank_whatever_its_cfg_predicate_looks_like() {
    // Asserted on distinctive *body* content rather than line numbers, which
    // shift under any edit above them and would make this test a tax on
    // unrelated work — it had them, and the first reformat two files away broke
    // it. Blanking keeps each item's header line so the offsets after it stay
    // put, so a header is not evidence either way; only a body line is.
    for (file, fixture, why) in [
        // `#[cfg(all(test, feature = "backend-vulkan"))]`, the spelling that
        // never matched. 1466 lines, the largest single block of the seven.
        (
            "reims-vgpu/src/runtime/draw/vulkan.rs",
            "let blank = vec![0u8; (w * h * 4) as usize];",
            "vulkan_split_tests",
        ),
        // A three-predicate `all`, in a module this host compiles but cannot run.
        (
            "reims-vgpu/src/runtime/draw/mod.rs",
            "assert_eq!(from_object.lod_max_bits, 8.0f32.to_bits());",
            "sampler_record_tests",
        ),
        (
            "reims-vgpu/src/runtime/draw/mod.rs",
            "MTLLoadAction{name} is in contract",
            "load_action_contract_tests",
        ),
        // `pub(crate) mod` behind a test gate: the visibility must not stop the
        // blanker reaching the body.
        (
            "reims-vgpu/src/runtime/decode/resource/mod.rs",
            "pub(crate) const LEN: usize = w_smp::NEW_SAMPLER_TOTAL_LEN as usize;",
            "sampler_desc",
        ),
    ] {
        assert!(
            !source(file).contains(fixture),
            "{why} in {file} is a test body and every scan here is reading it as \
             device behaviour — {fixture:?} survived the blanking"
        );
    }
}

/// Blanking a module must not take the file's remaining product code with it.
///
/// The failure this catches is a cutoff at the first marker rather than a brace
/// walk, which passes every assertion above and silently deletes the rest of the
/// file from every scan's view.
#[test]
fn product_code_after_a_test_module_survives_the_blanking() {
    // `runtime::host` is the hard case: it holds `FakeHost` — the host under
    // every one of these tests — behind four separate test gates, the last
    // ending some 440 lines before the end of the file. A blanker that ran from
    // a marker to the end, or that mismatched one brace, would take the product
    // code between and after them.
    let host = source("reims-vgpu/src/runtime/host.rs");
    for product in [
        // Declared after `mach_vm`, the first test-gated module in the file.
        "pub struct FakeHost",
        // Declared after three `#[cfg(test)] impl` blocks totalling ~690 lines.
        "pub fn read_u32<M: HostMemory>",
    ] {
        assert!(
            host.contains(product),
            "`{product}` is product code in host.rs and the blanker ate it — \
             a scan reading this file now measures less than the device"
        );
    }
    // And the fixtures between them are gone. `FakeHost`'s `HostMemory` and
    // `HostOps` impls emulate a device, which is exactly what a scan looking for
    // device behaviour will mistake them for. The assertion is on a *body* line:
    // blanking keeps the item's header so every offset after it stays put, so
    // `impl HostMemory for FakeHost` is still in the text and says nothing.
    assert!(
        !host.contains("self.fire_rewires("),
        "FakeHost's test-only trait impl bodies are still being scanned as the \
         device — they emulate one, which is what makes them the worst possible \
         thing for a scan here to be reading"
    );
}

/// A file whose name says it holds tests is not part of the population.
#[test]
fn a_file_named_for_its_tests_is_not_scanned_as_product() {
    let scanned: Vec<String> = guest_facing_sources().into_iter().map(|(f, _)| f).collect();
    for rel in [
        "reims-vgpu/src/runtime/surface_cache/cap_tests.rs",
        "reims-vgpu/src/runtime/mapper/revalidate_tests.rs",
        "reims-vgpu/src/runtime/storage_flush/render_flush_witness_tests.rs",
    ] {
        assert!(
            !scanned.contains(&rel.to_string()),
            "{rel} is a test file and every scan here is reading it as device behaviour"
        );
    }
    // The self-check: prove the population is not simply empty, and that the
    // name rule did not take a product file with it. `mapper/mod.rs` sits beside
    // one of the excluded files.
    assert!(
        scanned.contains(&"reims-vgpu/src/runtime/mapper/mod.rs".to_string()),
        "the exclusion rule removed a product file next to a test file"
    );
    assert!(
        scanned.len() > 150,
        "only {} files in the population; the scan sees almost nothing",
        scanned.len()
    );
}

/// `any(test, …)` compiles in a non-test build, so its body is product code.
///
/// Nothing in either crate spells one today. The rule is asserted rather than
/// left to the first author who writes one, because the failure is silent in the
/// direction that matters: blanking it would hide product code from every scan
/// here, which is the half that reports clean.
#[test]
fn a_cfg_that_only_might_be_test_is_not_treated_as_one() {
    let synthetic = "\
#[cfg(any(test, feature = \"probe\"))]
mod might_be_product {
    const CAP: usize = 4;
}
#[cfg(all(test, feature = \"backend-vulkan\"))]
mod definitely_fixtures {
    const FIXTURE_CAP: usize = 4;
}
";
    let blanked = source_scan::blank_test_items(synthetic);
    assert!(
        blanked.contains("const CAP: usize = 4;"),
        "an `any(test, …)` body is product code on some arm and was blanked"
    );
    assert!(
        !blanked.contains("FIXTURE_CAP"),
        "an `all(test, …)` body is fixtures and survived"
    );
}
