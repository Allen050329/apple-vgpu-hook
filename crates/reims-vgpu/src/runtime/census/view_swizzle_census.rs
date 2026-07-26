//! Always-on proxy for the type-8 view-swizzle bug class.
//!
//! # The class
//!
//! A guest texture view (type-8, opcode `0x1b`) can remap which channel each
//! output component reads. Vulkan performs exactly that for free through the
//! image view's `VkComponentMapping`. Doing it any other way is a defect with
//! two shapes, and this proxy tells them apart:
//!
//! * **Dropped.** The bind refuses because the swizzle is not identity, so the
//!   draw loses its sampled input entirely and renders without the texture.
//!   This is the shape the Vulkan pathway had, and it was *silent* — the
//!   refusal returned `None` with nothing in the fail log, indistinguishable
//!   from a texture that simply was not there.
//! * **CPU-remapped.** The bind rewrites every texel on the CPU. Correct, but
//!   it forces the texture onto the upload path and **costs it the zero-copy
//!   property** — the guest→GPU crossing that the whole present path is built
//!   to avoid.
//!
//! # Reading it
//!
//! `/tmp/reims-vgpu-fail.log`, always-on:
//!
//! * `OFF view_swizzle gpu=<n> cpu=<n> declined=<n>` — cumulative, on a
//!   doubling schedule.
//! * `view_swizzle_declined reason=<slug> ref=<n> …` — first sight of each
//!   decline reason, deduplicated.
//!
//! **`cpu=0` is the invariant this proxy exists to hold.** A nonzero `cpu`
//! means some rail is remapping texels by hand again and has quietly given up
//! zero-copy for those textures. `gpu` counting up is the healthy signal:
//! swizzled views are binding, and the hardware is doing the work.
//!
//! Measure-only: nothing here gates decode, execute or present.

use std::collections::BTreeMap;
use std::sync::Mutex;

use crate::observe;

#[derive(Default)]
struct Census {
    /// Swizzled binds served by a `VkComponentMapping` on the image view.
    gpu: u64,
    /// Swizzled binds served by rewriting texels on the CPU. Must stay 0.
    cpu: u64,
    /// Swizzled binds refused, by reason slug.
    declined: BTreeMap<&'static str, u64>,
    /// (reason, texture ref) pairs already reported once.
    seen: std::collections::BTreeSet<(&'static str, u32)>,
    next_emit: u64,
}

impl Census {
    fn total(&self) -> u64 {
        self.gpu + self.cpu + self.declined.values().sum::<u64>()
    }
}

static CENSUS: Mutex<Option<Census>> = Mutex::new(None);

fn with<R>(f: impl FnOnce(&mut Census) -> R) -> R {
    let mut guard = CENSUS.lock().unwrap_or_else(|e| e.into_inner());
    let census = guard.get_or_insert_with(|| Census {
        next_emit: 1,
        ..Census::default()
    });
    f(census)
}

fn emit_if_due(c: &mut Census) -> Option<String> {
    let total = c.total();
    if total < c.next_emit {
        return None;
    }
    c.next_emit = total.saturating_mul(2);
    let declined = c
        .declined
        .iter()
        .map(|(r, n)| format!("{r}:{n}"))
        .collect::<Vec<_>>()
        .join(",");
    Some(format!(
        "view_swizzle gpu={} cpu={} declined={} by_reason=[{declined}]",
        c.gpu,
        c.cpu,
        c.declined.values().sum::<u64>(),
    ))
}

/// A non-identity view swizzle was handed to the GPU as a component mapping.
/// The healthy path: no texel was touched and the bind kept its content rail.
pub fn note_gpu_mapping() {
    if let Some(line) = with(|c| {
        c.gpu += 1;
        emit_if_due(c)
    }) {
        observe::off(line);
    }
}

/// A non-identity view swizzle was performed by rewriting texels on the CPU.
///
/// Always fail-visible on first sight, because this is the regression the GPU
/// mapping replaced: it is *correct* and therefore invisible in the output,
/// while costing the texture its zero-copy crossing.
pub fn note_cpu_remap(texture_ref: u32) {
    use crate::observe::Decline as _;
    let reason = SwizzleDecline::CpuRemap;
    let (first, line) = with(|c| {
        c.cpu += 1;
        let first = c.seen.insert((reason.slug(), texture_ref));
        (first, emit_if_due(c))
    });
    if first {
        observe::Emit::decline("view_swizzle_cpu_remap", &reason)
            .field("ref", texture_ref)
            .fail();
    }
    if let Some(line) = line {
        observe::off(line);
    }
}

/// A swizzled bind was refused, naming the specific check.
pub fn note_declined(reason: SwizzleDecline, texture_ref: u32) {
    use crate::observe::Decline as _;
    let (first, line) = with(|c| {
        *c.declined.entry(reason.slug()).or_default() += 1;
        let first = c.seen.insert((reason.slug(), texture_ref));
        (first, emit_if_due(c))
    });
    if first {
        observe::Emit::decline("view_swizzle_declined", &reason)
            .field("ref", texture_ref)
            .fail();
    }
    if let Some(line) = line {
        observe::off(line);
    }
}

/// The two ways a non-identity view swizzle fails to reach the GPU as a
/// component mapping.
///
/// Both are refusals of the zero-copy path even though only one refuses the
/// bind: a CPU remap renders correctly and is therefore invisible in the output,
/// which is exactly why it needs a name in the log.
///
/// This replaced a `pub mod decline` of bare `&str` constants. The slugs are
/// `swizzle_`-prefixed because `cpu_remap` and `resident_direct_bind`, bare,
/// name nothing about which rail wrote them — the same argument that prefixed
/// the slate reasons.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SwizzleDecline {
    /// Every texel was rewritten by hand. Correct output, zero-copy lost.
    CpuRemap,
    /// The source is a GPU-resident target bound directly, whose view the
    /// engine owns and does not re-create per bind, so no per-bind component
    /// mapping can be attached to it.
    ResidentDirectBind,
}

impl crate::observe::Decline for SwizzleDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::CpuRemap => "swizzle_cpu_remap",
            Self::ResidentDirectBind => "swizzle_resident_direct_bind",
        }
    }
}

/// `(gpu, cpu, declined_total)`. Test/diagnostic accessor.
pub fn counts() -> (u64, u64, u64) {
    with(|c| (c.gpu, c.cpu, c.declined.values().sum()))
}

#[cfg(test)]
pub fn reset_for_tests() {
    with(|c| {
        *c = Census {
            next_emit: 1,
            ..Census::default()
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The three outcomes are counted separately, because they are three
    /// different situations: working, working-but-slow, and not working.
    #[test]
    fn the_three_outcomes_are_counted_apart() {
        reset_for_tests();
        note_gpu_mapping();
        note_gpu_mapping();
        note_cpu_remap(7);
        note_declined(SwizzleDecline::ResidentDirectBind, 9);
        assert_eq!(counts(), (2, 1, 1));
        reset_for_tests();
    }

    /// The cumulative line separates them too, and names which check declined —
    /// "some swizzles failed" is not an actionable report.
    #[test]
    fn the_summary_line_separates_them() {
        reset_for_tests();
        note_declined(SwizzleDecline::ResidentDirectBind, 3);
        let line = with(emit_if_due).unwrap_or_else(|| {
            with(|c| {
                c.next_emit = 0;
                emit_if_due(c).unwrap()
            })
        });
        assert!(line.contains("gpu=0"), "{line}");
        assert!(line.contains("cpu=0"), "{line}");
        assert!(line.contains("declined=1"), "{line}");
        assert!(line.contains("swizzle_resident_direct_bind:1"), "{line}");
        reset_for_tests();
    }

    /// Both slugs name the rail that wrote them.
    ///
    /// Bare, `cpu_remap` and `resident_direct_bind` say nothing about which
    /// subsystem refused — the same argument that prefixed the slate reasons.
    /// Crate-wide distinctness is `observe::gate`'s job; the prefix is this
    /// module's.
    #[test]
    fn both_swizzle_slugs_name_their_rail() {
        use crate::observe::Decline as _;
        for r in [SwizzleDecline::CpuRemap, SwizzleDecline::ResidentDirectBind] {
            assert!(
                r.slug().starts_with("swizzle_"),
                "{} is not namespaced to this rail",
                r.slug()
            );
        }
        assert_ne!(
            SwizzleDecline::CpuRemap.slug(),
            SwizzleDecline::ResidentDirectBind.slug()
        );
    }

    /// A hot rail must cost a logarithmic number of lines, not one per bind.
    #[test]
    fn emission_doubles_rather_than_tracking_every_bind() {
        reset_for_tests();
        let mut emits = 0;
        for _ in 0..4096 {
            if with(|c| {
                c.gpu += 1;
                emit_if_due(c).is_some()
            }) {
                emits += 1;
            }
        }
        assert_eq!(emits, 13);
        reset_for_tests();
    }
}
