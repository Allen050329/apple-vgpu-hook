//! A panic record that names the wrong entry point is worse than no record.
//!
//! `unwind_safe` takes the C symbol its body belongs to as a plain `&'static
//! str`, because there is nothing else it could take: the macro that would
//! derive it (`std::any::type_name` of the closure, `function_name!`) either
//! does not exist in stable Rust or yields the closure's synthetic name rather
//! than the `#[no_mangle]` symbol a reader of `reims_vgpu_qemu_abi.h` is
//! holding.
//!
//! A hand-written string is copy-paste bait. Twenty-two of these entry points
//! are the same eight lines apart from one identifier, and the failure mode is
//! silent in the worst way: a panic in `reims_vgpu_qemu_drain` reported as
//! `entry=reims_vgpu_qemu_poll` sends the next reader to the wrong call path,
//! and nothing in the compiler, the test suite or the log can tell that the
//! name and the body disagree. The mismatch is *only* visible from the source.
//!
//! So this reads the source: for every `unwind_safe(` call in the ABI module,
//! the first argument must be a string literal spelling the `extern "C" fn`
//! that lexically encloses it. An empty scan fails too — a structural check
//! that stops matching reports green while looking at nothing, which is the
//! defect `the_source_scanner_reads_the_product_and_not_its_fixtures` exists to
//! keep out of this suite.

mod source_scan;

/// Where the C ABI entry points live. Every `unwind_safe` call site in the
/// crate is here; the only other one is in `device/tests.rs`, which is a
/// fixture and names no C symbol.
const ABI_MODULE: &str = "crates/reims-vgpu/src/qemu/abi.rs";

/// The name in `extern "C" fn NAME`, for the innermost such item declared at or
/// before `byte`.
fn enclosing_extern_fn(text: &str, byte: usize) -> Option<&str> {
    let mut found = None;
    let mut cursor = 0usize;
    while let Some(rel) = text[cursor..].find("extern \"C\" fn ") {
        let start = cursor + rel + "extern \"C\" fn ".len();
        if start > byte {
            break;
        }
        let end = start
            + text[start..]
                .find(|c: char| !(c.is_alphanumeric() || c == '_'))
                .unwrap_or(0);
        found = Some(&text[start..end]);
        cursor = end;
    }
    found
}

/// The string literal that opens a call, when the first argument is one.
fn first_string_argument(text: &str, after_open_paren: usize) -> Option<&str> {
    let rest = &text[after_open_paren..];
    let lead = rest.len() - rest.trim_start().len();
    let rest = rest.trim_start();
    if !rest.starts_with('"') {
        return None;
    }
    let close = rest[1..].find('"')? + 1;
    let abs = after_open_paren + lead + 1;
    Some(&text[abs..abs + close - 1])
}

#[test]
fn every_unwind_safe_call_names_the_entry_point_it_guards() {
    let path = source_scan::workspace_root().join(ABI_MODULE);
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{} must be readable: {e}", path.display()));
    // Comments first: the module doc discusses these calls in prose, and a
    // scanner that reads its own documentation finds sites that do not exist.
    let text = source_scan::blank_comments(&raw);

    let mut checked = 0usize;
    let mut wrong: Vec<String> = Vec::new();
    let mut cursor = 0usize;
    while let Some(rel) = text[cursor..].find("unwind_safe(") {
        let open = cursor + rel + "unwind_safe(".len();
        cursor = open;
        let Some(owner) = enclosing_extern_fn(&text, open) else {
            wrong.push(format!("a call at byte {open} sits in no `extern \"C\" fn`"));
            continue;
        };
        match first_string_argument(&text, open) {
            Some(named) if named == owner => checked += 1,
            Some(named) => wrong.push(format!(
                "`{owner}` reports itself as `{named}`: a panic in it would send the reader to the wrong entry point"
            )),
            None => wrong.push(format!(
                "`{owner}` calls unwind_safe without naming itself; a panic there would be recorded against no entry point"
            )),
        }
    }

    assert!(
        wrong.is_empty(),
        "every C ABI entry must name itself to `unwind_safe`:\n  {}",
        wrong.join("\n  ")
    );
    // The population is not pinned to a number — entry points come and go with
    // the ABI version — but a scan that matched nothing has measured nothing.
    assert!(
        checked > 0,
        "{ABI_MODULE} yielded no `unwind_safe` call sites; the scan is looking at the wrong shape"
    );
}
