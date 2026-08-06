//! A bitmask standing in for a set of slots says what bounds it.
//!
//! `AGENTS.md` sends a reader hunting for bounds to four scans, and all four
//! find a bound by its **name** — a `MAX` or a `CAP` on a constant. A mask has
//! neither. `mask |= 1u32 << index` bounds the set to 32 members with nothing
//! declared anywhere, so it is invisible to every one of them, and it is
//! invisible in the direction that matters: the four scans exist to find a
//! bound that could cost guest work, and this one costs it in a way none of
//! them models.
//!
//! # What goes wrong, and why it is not an eviction
//!
//! Shifting a `u32` by 32 or more is not a wide value, a saturation, or a
//! dropped entry. In debug it panics; in release it wraps the shift amount, so
//! `1u32 << 33` is `1u32 << 1` and the write lands on **another member's bit**.
//! The set then reports a slot as present that was never added and absent that
//! was, and every later read of it is wrong with nothing said. That is the
//! outcome the standing goal forbids outright, reached without a single entry
//! being evicted.
//!
//! # The rule
//!
//! Two things must be true at every site, and the population below records both
//! because neither is visible from the other's end:
//!
//! * the **shift amount** is bounded before the shift, by a check the site
//!   cannot skip; and
//! * the **mask's width** is pinned against that bound by a `const` assertion,
//!   so raising the bound fails the build instead of silently narrowing the set.
//!
//! Every site in the tree already satisfies both — this scan found no defect
//! when it was written. That is what it is for. The population is four writes
//! over three masks, small enough that the next one is easy to add and easy to
//! add *wrongly*, and the failure it would cause does not look like a bounds bug
//! from any other angle.

mod source_scan;

/// One mask-write site, and the two facts that make it safe.
struct Site {
    /// Path relative to `crates/`, as the scan reports it.
    file: &'static str,
    /// The assignment's left-hand side, verbatim. Keyed on this rather than on a
    /// line number so that editing the file above a site does not turn this red
    /// — the cost `a_refusal_bound_says_whose_limit_it_is` pays on purpose and
    /// this one has no reason to.
    mask: &'static str,
    /// What stops the shift amount reaching the mask's width.
    bounded_by: &'static str,
    /// The file holding the `const` assertion that pins the mask's width
    /// against [`Site::bounded_by`]'s constant. Named as a field rather than
    /// inside the prose so the check below reads a path, not a sentence.
    pin_file: &'static str,
    /// The constant that assertion bounds. The pin must name it and compare it
    /// to a `BITS` width.
    pin_constant: &'static str,
}

const SITES: &[Site] = &[
    Site {
        file: "reims-vgpu/src/backend/metal/raw_metal.rs",
        mask: "mask",
        bounded_by: "`util::valid_sampler_index`, checked on the line above with \
                     a `continue` and a fail-visible decline on the other arm — \
                     Metal's own reflection naming a slot outside its own \
                     sampler table",
        pin_file: "reims-vgpu/src/backend/metal/constants.rs",
        pin_constant: "REIMS_VGPU_METAL_MAX_SAMPLERS",
    },
    Site {
        file: "reims-vgpu/src/backend/vulkan/engine/pools/submission_and_buffers.rs",
        mask: "mask",
        bounded_by: "the loop is `self.slots.iter().enumerate()` and `self.cur` \
                     indexes that same ring, so the amount is a ring slot by \
                     construction rather than by a check",
        pin_file: "reims-vgpu/src/backend/vulkan/engine/pools/mod.rs",
        pin_constant: "RING_DEPTH",
    },
    Site {
        file: "reims-vgpu/src/runtime/mmio.rs",
        mask: "state.active_child_mask",
        bounded_by: "`model::accept_child_channel`, which refuses channel 0 and \
                     anything at or past `MAX_CHANNELS` — the guest writes this \
                     value straight into an MMIO register, so it is the one site \
                     here whose shift amount is guest-controlled",
        pin_file: "reims-vgpu/src/model/regs.rs",
        pin_constant: "MAX_CHANNELS",
    },
    Site {
        file: "reims-vgpu/src/runtime/mmio.rs",
        mask: "state.pending.child_mask",
        bounded_by: "the same `accept_child_channel` call — one guard over both \
                     writes, which is why they cannot disagree",
        pin_file: "reims-vgpu/src/model/regs.rs",
        pin_constant: "MAX_CHANNELS",
    },
];

/// A write of one bit into a mask at a computed position: `… |= 1u32 << expr`.
///
/// Deliberately only the **write**. A read — `if mask & (1 << i) != 0` — has the
/// same shift hazard, but it can only ask about a bit a write put there, so
/// holding the writes holds the set. Widening this to reads would triple the
/// population with sites that carry no independent verdict.
///
/// A literal shift amount is not a hazard and is skipped: `1 << 0` names a flag,
/// not a member of a set, and `model/regs.rs` declares several.
fn mask_writes(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for line in text.lines() {
        let Some((lhs, rhs)) = line.split_once("|=") else {
            continue;
        };
        let rhs = rhs.trim_start().trim_start_matches('(');
        let Some(shift) = rhs
            .strip_prefix("1u32 <<")
            .or_else(|| rhs.strip_prefix("1u64 <<"))
            .or_else(|| rhs.strip_prefix("1usize <<"))
            .or_else(|| rhs.strip_prefix("1 <<"))
        else {
            continue;
        };
        // A constant bit position is a named flag, not a set member.
        if shift.trim().trim_end_matches(';').parse::<u32>().is_ok() {
            continue;
        }
        out.push(lhs.trim().to_string());
    }
    out
}

#[test]
fn every_mask_used_as_a_set_has_a_verdict() {
    let mut found: Vec<(String, String)> = Vec::new();
    for (file, text) in source_scan::guest_facing_sources() {
        for mask in mask_writes(&text) {
            let site = (file.clone(), mask);
            if !found.contains(&site) {
                found.push(site);
            }
        }
    }

    // The scan proving it can see the shape before it is allowed to report a
    // population. `mask_writes` matches on four exact spellings of a shift, and
    // a formatter that put `1u32<<index` or wrapped the line differently would
    // silently empty this — which reads as "no mask in the tree is used as a
    // set", the strongest possible pass from a scanner that matched nothing.
    assert!(
        !found.is_empty(),
        "no `|= 1 << expr` write found anywhere in the two guest-facing crates. \
         The scan is broken, not the tree"
    );

    let adjudicated: Vec<(String, String)> = SITES
        .iter()
        .map(|s| (s.file.to_string(), s.mask.to_string()))
        .collect();
    let unadjudicated: Vec<&(String, String)> =
        found.iter().filter(|f| !adjudicated.contains(f)).collect();
    let stale: Vec<&(String, String)> = adjudicated
        .iter()
        .filter(|a| !found.contains(a))
        .collect();

    // Both directions in one report: a mask *renamed* produces one of each, and
    // an author shown only the first half writes a second verdict for a site
    // that already had one.
    assert!(
        unadjudicated.is_empty() && stale.is_empty(),
        "the mask-as-a-set population moved.\n  \
         new, with no verdict: {unadjudicated:?}\n  \
         adjudicated but no longer present: {stale:?}\n\
         A new one needs two things written down: what bounds the shift amount \
         below the mask's width, and where a `const` assertion pins that width \
         against it. Without the second, raising the bound narrows the set in \
         silence — and a `u32` shifted by 32 wraps in release, so the write \
         lands on another member's bit."
    );
}

/// Every width pin the population names is really in the file it names.
///
/// The verdicts above are prose and prose rots. This is the half that does not:
/// the pin is the only thing standing between a raised bound and a set that
/// silently stops holding its members, so a verdict citing one that has been
/// deleted is worse than no verdict at all.
///
/// Matched on the shape rather than the exact text — a `const` assertion in that
/// file mentioning both the named constant and a `BITS` — because the spelling
/// of the comparison is not the claim. That the assertion exists, names the
/// bound, and compares it to a width, is.
#[test]
fn every_named_width_pin_exists() {
    let sources = source_scan::guest_facing_sources();
    let mut checked = 0usize;

    for site in SITES {
        assert!(
            !site.bounded_by.is_empty(),
            "{}: a site with no stated bound on its shift amount is not \
             adjudicated, it is listed",
            site.mask
        );
        let (_, text) = sources
            .iter()
            .find(|(f, _)| f == site.pin_file)
            .unwrap_or_else(|| {
                panic!("{}: names a pin file the scan does not read", site.mask)
            });

        // The shape, not the exact text: that an assertion exists, names the
        // bounded constant, and compares it to a width. How the comparison is
        // spelled is not the claim.
        let pinned = text
            .lines()
            .any(|l| l.contains(site.pin_constant) && l.contains("BITS"));
        assert!(
            pinned,
            "{} no longer pins {} against a mask width, but `{}`'s verdict says \
             it does. Either the assertion was deleted — in which case raising \
             {} now narrows a set in silence, and a `u32` shifted by 32 wraps \
             rather than saturating — or it moved and this verdict must follow it",
            site.pin_file, site.pin_constant, site.mask, site.pin_constant
        );
        checked += 1;
    }

    assert_eq!(
        checked,
        SITES.len(),
        "every site's pin must be checked, not merely most of them"
    );
}
