//! A collection this device shrinks to stay under a bound must say what the
//! shrink costs.
//!
//! A real GPU refuses an allocation while its memory is full. It does not
//! quietly forget a texture it already holds and hand back the wrong pixels the
//! next time the guest samples it. So every place this device drops an entry it
//! has already admitted — because a cap said to, not because the entry's life
//! ended — is a place where fidelity can be traded for a number somebody chose.
//!
//! That trade is often the right one: a host-side copy of guest pages can be
//! dropped and re-read, and a witness ring can forget an old reading as long as
//! it says it did. What is never right is dropping something no other authority
//! holds. The difference is invisible at the eviction site, which is a two-line
//! `pop_front` in every case, and it has been settled by hand-sweeping this
//! crate seven separate times.
//!
//! # What this test asserts
//!
//! The same thing `a_remembered_refusal_says_whether_it_can_go_stale` asserts
//! about the inverse direction: not that a particular answer is right, but that
//! the question was **asked**. Every shrink governed by a capacity appears below
//! with a cost. A new one fails this test until its author writes the line,
//! which is the moment the question is cheapest to answer — and the vocabulary
//! contains [`Cost::LosesGuestWork`] precisely so that an author who answers
//! honestly gets told, by a failing build, that the architecture is wrong rather
//! than the number.
//!
//! # Why the two directions needed two tests
//!
//! The other one looks for an error type stored in a non-error type: something
//! **kept** that should have been forgotten. This one looks for something
//! **forgotten** that may have needed keeping. Neither scan can see the other's
//! shape, and both classes have shipped real bugs in this crate.
//!
//! # Why source text
//!
//! A boot cannot answer it. An eviction path that never fires on the workload
//! anyone drives reads exactly like one that fires and costs nothing — the
//! target pool's cap of 32 has never been observed to bind, and that is not
//! evidence about what happens when it does. Source also answers for both
//! backends at once: the Vulkan sites are `#[cfg]`-ed out of the Metal build and
//! vice versa, so no single compilation sees them all.

mod source_scan;
use source_scan::{blank_comments, blank_test_modules, rust_sources, workspace_root};

/// What is lost when this site drops an entry it had already admitted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Cost {
    /// Another authority still holds the content, so the eviction costs the
    /// work of producing it again and nothing else. Guest pages behind a
    /// host-side copy are the usual authority; a pinned engine resident and a
    /// re-runnable derivation are the others.
    Recomputable,
    /// The eviction makes the answer more conservative, not less. A reader
    /// whose entry is gone is told the structure cannot answer, and takes the
    /// expensive correct path. Fidelity cannot fall out of this one; only
    /// throughput can.
    FailsClosed,
    /// The collection is a witness, a census, or a log-dedup set. Dropping an
    /// entry costs a reading or a log line, never guest work — and the drop is
    /// itself counted, so the reading it weakens says that it is weakened.
    Observability,
    /// The site declines the incoming request rather than displacing a live
    /// entry, which is what a GPU with full memory does.
    RefusesInstead,
    /// The scan matched a removal that is not a capacity policy: a teardown, a
    /// reset, a lifetime end, or a length trim. Classified rather than filtered
    /// out, because "the scan should not have flagged this" is a claim worth
    /// writing down once instead of re-deriving at every reading.
    NotABound,
    /// Guest work with no other holder. **Forbidden**, and asserted absent
    /// below. The vocabulary carries it so that classifying a site honestly
    /// produces an immediate failure naming the real problem, rather than
    /// tempting the author into one of the words above.
    #[allow(dead_code, reason = "the assertion below is what keeps it unused")]
    LosesGuestWork,
}

/// Every capacity-governed shrink in `src/`, and what it costs.
///
/// Keyed by `(file, method)` with the number of sites that pair covers, so a new
/// eviction added beside an existing one moves the count and fails rather than
/// inheriting a verdict that was written about a different line.
const COSTS: &[(&str, &str, usize, Cost, &str)] = &[
    (
        "src/backend/vulkan/engine/caches.rs",
        "pop_front",
        1,
        Cost::Recomputable,
        "the FIFO holds memoized vkCreate* failures; forgetting one can only send \
         the next attempt back to the driver, which is the direction that restores \
         fidelity rather than losing it",
    ),
    (
        "src/backend/vulkan/engine/caches.rs",
        "remove",
        1,
        Cost::Recomputable,
        "the negative map entry displaced with its FIFO slot, same argument",
    ),
    (
        "src/backend/vulkan/engine/caches.rs",
        "retain",
        1,
        Cost::NotABound,
        "compaction of negative_order against the map it indexes; drops no entry \
         the map still holds",
    ),
    (
        "src/backend/vulkan/engine/dmabuf.rs",
        "swap_remove",
        1,
        Cost::RefusesInstead,
        "the import cache evicts unpinned entries to make room and answers \
         DmaBufDecline::BoundInUse when the bytes in the way are held by the \
         command buffer being built — a refusal, not a displacement of live work; \
         an evicted import is re-exported from the same guest pages",
    ),
    (
        "src/backend/vulkan/engine/mod.rs",
        "truncate",
        1,
        Cost::NotABound,
        "read_resident_bgra trims an over-long readback to the caller's need after \
         checking it is long enough; a length trim, not a population bound",
    ),
    (
        "src/backend/vulkan/engine/pools/images_and_registry.rs",
        "pop_front",
        1,
        Cost::Observability,
        "reclaimed_recent is the reclaim witness ring; RECLAIM_HISTORY bounds how \
         far back a census can look, and nothing reads it to decide device \
         behaviour",
    ),
    (
        "src/backend/vulkan/engine/pools/submission_and_buffers.rs",
        "remove",
        3,
        Cost::Recomputable,
        "the target pool (keyed by geometry, so a scratch slot rather than any \
         guest resource's content) and the sampled cache (re-uploaded from the \
         guest pages that remain authoritative). The sampled cache routes every \
         eviction through sampled_evict_route, so both bounds are counted",
    ),
    (
        "src/model/lru_memo.rs",
        "remove",
        1,
        Cost::Recomputable,
        "LruBytesMemo holds derived values under a byte cap; every entry is \
         recomputed from the guest bytes it was derived from",
    ),
    (
        "src/model/state.rs",
        "pop_front",
        1,
        Cost::Observability,
        "GvaEvictionWitness::note_evicted; the ring holds identities the GVA cache \
         has already evicted, and dropping one increments `forgotten`, which is \
         what makes `wanted` read as a lower bound instead of an answer",
    ),
    (
        "src/model/state.rs",
        "remove",
        2,
        Cost::Observability,
        "the same witness: one removal is the ring's own bound, the other is \
         note_restored retiring an identity a store brought back",
    ),
    (
        "src/runtime/compute_exec/mod.rs",
        "remove",
        1,
        Cost::Recomputable,
        "the compute storage-residency mirror records that guest pages already \
         hold a window's resident content. Dropping an entry sends the next read \
         back to those guest pages, which the writeback that armed the entry had \
         just written — so the cost is a re-upload and never a wrong pixel",
    ),
    (
        "src/runtime/drain/mod.rs",
        "clear",
        1,
        Cost::Observability,
        "the present-page-identity log dedup set, emptied at 1024 distinct keys so \
         a long boot keeps reporting; gates emission only",
    ),
    (
        "src/runtime/gather_witness.rs",
        "remove",
        1,
        Cost::FailsClosed,
        "MAX_TRACKED_WINDOWS is a hypervisor-harvest bound, not a memory one. A \
         window whose entry is evicted has no witness, so the next bind cannot \
         elide its gather and re-arms — counted as gw_window_overflow and \
         gw_rearm. The elision only ever narrows",
    ),
    (
        "src/runtime/guest_dmabuf.rs",
        "swap_remove",
        1,
        Cost::Recomputable,
        "the cache hands out Arc<GuestDmaBuf> clones, so evicting the map's \
         reference leaves a live holder's fd open; the cost is a re-export, which \
         is a page walk",
    ),
    (
        "src/runtime/guest_dmabuf.rs",
        "remove",
        1,
        Cost::Recomputable,
        "the bucket entry dropped with it, same argument",
    ),
    (
        "src/runtime/host_writes.rs",
        "pop_front",
        1,
        Cost::FailsClosed,
        "the ring bounds the reader's scan. `answers_from` moves with the front, \
         and a reader whose mark is older than it is told the ring cannot answer — \
         which reads as 'assume written' and costs a gather",
    ),
];

/// A removal that shrinks an already-admitted population.
///
/// `clear` is handled apart from the rest at the call site: it empties a
/// collection outright, which is a reset in almost every occurrence, so it only
/// counts as a shrink when a size comparison sits immediately above it.
const SHRINK: &[&str] = &[
    "pop_front",
    "pop_first",
    "pop_last",
    "pop_back",
    "swap_remove",
    "truncate",
    "retain",
    "remove",
];

/// Words that make an identifier a statement about how many of something is
/// allowed, rather than about any particular one.
const BOUND_WORDS: &[&str] = &[
    "CAP", "MAX", "LIMIT", "BUDGET", "RING", "HISTORY", "KEYS", "PER_", "WINDOWS",
];

/// Ways a fragment of source says "how many / how much".
const SIZE_TERMS: &[&str] = &[".len()", ".count()", "_bytes", "_count", ".bytes"];

/// How far above a removal to look for the comparison that governs it.
///
/// Twenty lines rather than the enclosing function, which was tried first: the
/// import caches put the comparison in one function and the removal in the
/// helper it calls, and a function-scoped window misses both of them.
const WINDOW_LINES: usize = 20;

#[derive(Debug)]
struct Site {
    file: String,
    line: usize,
    method: String,
    bound: String,
}

/// The capacity term in `window`, if it has one.
fn bound_term(window: &str) -> Option<String> {
    let bytes: Vec<char> = window.chars().collect();
    let mut i = 0;
    while i < bytes.len() {
        if !(bytes[i].is_alphanumeric() || bytes[i] == '_') {
            i += 1;
            continue;
        }
        let start = i;
        while i < bytes.len() && (bytes[i].is_alphanumeric() || bytes[i] == '_') {
            i += 1;
        }
        let token: String = bytes[start..i].iter().collect();
        // `u64::MAX` names a type's extreme, not this device's policy.
        let path_qualified = start >= 2 && bytes[start - 1] == ':' && bytes[start - 2] == ':';
        let shouty = token.len() >= 3
            && token.chars().any(|c| c.is_ascii_uppercase())
            && token
                .chars()
                .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_');
        if shouty && !path_qualified && BOUND_WORDS.iter().any(|w| token.contains(w)) {
            return Some(token);
        }
        let lower = token.to_ascii_lowercase();
        // The bare words are here because a bound is very often a parameter
        // called exactly `cap`. Leaving them out was measured, by injecting a
        // `while self.recent.len() > cap { … }` into `host_writes` and watching
        // this test stay green.
        //
        // `max` is deliberately not among them, in any position. Bare, it is
        // the name of two std methods and sits beside almost every arithmetic
        // clamp in this crate. A `max_` *prefix* was tried and reverted in the
        // same hour: its only catch was `delete_task`'s `retain`, next to the
        // `max_task_id_seen` high-water counter, which is a lifetime removal
        // and not a bound at all. `MAX_` in a shouty constant is the spelling a
        // real bound uses here, and the rule above already has it.
        if matches!(lower.as_str(), "cap" | "capacity" | "limit" | "budget")
            || lower.contains("_cap")
            || lower.contains("cap_")
            || lower.contains("_limit")
            || lower.contains("_budget")
            || lower.contains("_max")
        {
            return Some(token);
        }
    }
    // A bound spelled as a bare literal is the worst kind and must not escape
    // for want of a name: `self.target_order.len() >= 32`.
    literal_bound(window)
}

/// A `<size term> >(=) <literal>` comparison, when the literal is at least 2.
///
/// One is excluded because `len() > 1` guards a last-element case far more often
/// than it caps a population, and neither reading is a capacity policy.
fn literal_bound(window: &str) -> Option<String> {
    for term in SIZE_TERMS {
        let mut from = 0;
        while let Some(at) = window[from..].find(term) {
            let after = from + at + term.len();
            from = after;
            let rest = window[after..].trim_start();
            let Some(rest) = rest.strip_prefix('>') else {
                continue;
            };
            let rest = rest.strip_prefix('=').unwrap_or(rest).trim_start();
            let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
            if digits.is_empty() {
                continue;
            }
            if digits.parse::<u64>().unwrap_or(0) >= 2 {
                return Some(digits);
            }
        }
    }
    None
}

fn has_size_term(window: &str) -> bool {
    SIZE_TERMS.iter().any(|t| window.contains(t))
}

/// Whether a size comparison sits close enough above `clear()` to be governing
/// it. Deliberately tight: a `clear()` twenty lines below an unrelated `len()`
/// is a reset.
fn clear_is_governed(lines: &[&str], at: usize) -> bool {
    let from = at.saturating_sub(3);
    lines[from..=at].iter().any(|l| {
        SIZE_TERMS
            .iter()
            .any(|t| l.contains(t) && (l.contains('>') || l.contains(">=")))
    })
}

fn find_sites(sources: &[(String, String)]) -> Vec<Site> {
    let mut sites = Vec::new();
    for (file, text) in sources {
        let lines: Vec<&str> = text.lines().collect();
        for (i, line) in lines.iter().enumerate() {
            let mut found: Vec<&str> = SHRINK
                .iter()
                .filter(|m| line.contains(&format!(".{m}(")))
                .copied()
                .collect();
            if line.contains(".clear()") && clear_is_governed(&lines, i) {
                found.push("clear");
            }
            if found.is_empty() {
                continue;
            }
            // `swap_remove` also contains `remove`; the longer name wins so a
            // site is not reported twice under two methods.
            if found.contains(&"swap_remove") {
                found.retain(|m| *m != "remove");
            }
            let from = i.saturating_sub(WINDOW_LINES);
            let window = lines[from..=i].join("\n");
            if !has_size_term(&window) {
                continue;
            }
            let Some(bound) = bound_term(&window) else {
                continue;
            };
            for method in found {
                sites.push(Site {
                    file: file.clone(),
                    line: i + 1,
                    method: method.to_string(),
                    bound: bound.clone(),
                });
            }
        }
    }
    sites
}

#[test]
fn every_capacity_governed_eviction_carries_a_cost() {
    let root = workspace_root();
    let src = root.join("crates/reims-vgpu/src");
    let sources: Vec<(String, String)> = rust_sources(&src)
        .into_iter()
        // `*/tests.rs` is this crate's spelling for a unit-test module in its
        // own file. Its fixtures shrink collections constantly and none of it
        // is device behaviour.
        .filter(|p| p.file_name().is_some_and(|n| n != "tests.rs"))
        .map(|p| {
            let raw = std::fs::read_to_string(&p).expect("read source");
            let text = blank_test_modules(&blank_comments(&raw));
            let rel = p
                .strip_prefix(root.join("crates/reims-vgpu"))
                .unwrap_or(&p)
                .to_string_lossy()
                .to_string();
            (rel, text)
        })
        .collect();

    let sites = find_sites(&sources);

    // Self-check, in the shape `wire_families_have_a_consumer` uses: refuse to
    // report anything until the scan has proved it can see the sites this test
    // was written for. A scan that silently matches nothing reports a clean
    // tree, which is the failure mode of every source scan — and this one has
    // three separate ways to go blind (the shrink list, the size terms, and the
    // bound vocabulary), so the check names one site per way.
    for (file, method, why) in [
        (
            "src/runtime/host_writes.rs",
            "pop_front",
            "a single-word shouty bound (RING)",
        ),
        (
            "src/backend/vulkan/engine/pools/submission_and_buffers.rs",
            "remove",
            "a bound spelled as a bare literal (32)",
        ),
        (
            "src/runtime/guest_dmabuf.rs",
            "swap_remove",
            "a byte bound rather than an entry count",
        ),
        (
            "src/runtime/compute_exec/mod.rs",
            "remove",
            "an eviction with no `if` above it at all — the count is computed \
             into a `take(n)`",
        ),
        (
            "src/model/lru_memo.rs",
            "remove",
            "a lowercase field bound (byte_cap)",
        ),
    ] {
        assert!(
            sites.iter().any(|s| s.file == file && s.method == method),
            "the scan cannot see {file}'s `{method}`, which is its only cover for \
             {why}. Its silence about everything else therefore means nothing.\n\n\
             Found:\n{}",
            summarize(&sites)
        );
    }

    let mut unclassified = Vec::new();
    for site in &sites {
        let known = COSTS
            .iter()
            .any(|(file, method, _, _, _)| *file == site.file && *method == site.method);
        if !known {
            unclassified.push(format!(
                "{}:{} — `{}` under bound `{}`",
                site.file, site.line, site.method, site.bound
            ));
        }
    }
    assert!(
        unclassified.is_empty(),
        "these drop an entry the device had already admitted, and say nothing \
         about what that costs.\n\nFor each, name the authority that still holds \
         the content. Guest pages behind a host copy, a pinned resident, a \
         re-runnable derivation: any of those makes it `Recomputable`. If the \
         eviction only makes a later answer more conservative it is \
         `FailsClosed`; if the structure is a witness or a dedup set it is \
         `Observability`. If the honest answer is that nothing else holds it, \
         the bound is the wrong architecture and `LosesGuestWork` will say so by \
         failing. Add a line to COSTS in {}.\n\n{}",
        file!(),
        unclassified.join("\n")
    );

    // A verdict about a site that no longer exists reads as coverage this test
    // does not have, and the count is what keeps a second eviction from
    // inheriting the first one's answer.
    let mut wrong = Vec::new();
    for (file, method, sites_expected, _, _) in COSTS {
        let live = sites
            .iter()
            .filter(|s| s.file == *file && s.method == *method)
            .count();
        if live != *sites_expected {
            wrong.push(format!(
                "{file} `{method}`: COSTS says {sites_expected} site(s), the scan \
                 finds {live}"
            ));
        }
    }
    assert!(
        wrong.is_empty(),
        "COSTS no longer describes this crate. A count that grew is a new \
         eviction inheriting a verdict written about a different line; one that \
         shrank is a claim about code that is gone.\n\n{}\n\nCurrent sites:\n{}",
        wrong.join("\n"),
        summarize(&sites)
    );
}

/// No site may be classified as losing guest work.
///
/// Separate from the classification test on purpose. That one fails when
/// somebody has not answered; this one fails when they have, and the answer is
/// that a bound is standing between the guest and its own pixels. The two
/// failures need different messages because they need different fixes — one is
/// a line of prose, the other is a redesign.
#[test]
fn no_eviction_is_allowed_to_lose_guest_work() {
    let losing: Vec<&str> = COSTS
        .iter()
        .filter(|(_, _, _, cost, _)| *cost == Cost::LosesGuestWork)
        .map(|(file, ..)| *file)
        .collect();
    assert!(
        losing.is_empty(),
        "these evict content no other authority holds, so a guest that asks for \
         it again gets something this device made up or a refusal it never \
         earned. A GPU refuses while its memory is full; it does not forget what \
         it was given. The bound is the wrong mechanism here — the fix is to \
         hold the entry, refuse the admission, or move the authority somewhere \
         durable.\n\n{}",
        losing.join("\n")
    );
}

fn summarize(sites: &[Site]) -> String {
    sites
        .iter()
        .map(|s| format!("  {}:{} `{}` bound={}", s.file, s.line, s.method, s.bound))
        .collect::<Vec<_>>()
        .join("\n")
}
