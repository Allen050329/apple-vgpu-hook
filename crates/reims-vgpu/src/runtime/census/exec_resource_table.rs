//! Per-window census of the resource table every `EXEC_INDIRECT2` payload
//! carries.
//!
//! The guest builds one 24-byte record per resource the submission touches
//! (`AppleParavirtCommandQueue::writeInvalidates`) and places it between the
//! payload header and the command-buffer descriptors. `clear_host_valid` is
//! sourced from `AppleParavirtResource::shouldInvalidateHost()`, a test-and-clear
//! of the resource's dirty bit whose only caller is that builder — so the guest
//! states "I CPU-wrote this resource" exactly once, here, and never resends it.
//!
//! The device used to read `resource_count` only as arithmetic for where the
//! table ends. Every record was stepped over unread, which makes both the
//! authorisation and its loss invisible: no counter anywhere could distinguish
//! "the guest never said" from "the guest said and we discarded it".
//!
//! Measure-only. Nothing reads these counters back to decide what to execute or
//! write; the execution path consumes the decoded table itself, not this census.
//!
//! # What the fields answer
//!
//! - `subs` / `no_table` / `descs` / `max` — is the table populated in live
//!   traffic at all, and how wide does it get. A boot where `descs` stays 0 says
//!   the wire layout assumption is wrong and nothing downstream should be built.
//! - `ids` / `id0` / `unresolved` / `as_map` / `as_tex` / `as_obj` — is the id
//!   space the one the device already knows, and which registry answers. The
//!   three `as_*` counts overlap by construction (one id can be all three);
//!   they exist because the consumer has to pick a registry to key on, and
//!   "94.8 % resolve somewhere" does not say which. `writeInvalidates` skips
//!   entries whose resource is null or whose id is 0, so any `id0` here is a
//!   layout surprise.
//! - `clr_h` / `set_h` / `clr_g` / `set_g` and the quad histogram — which of the
//!   four ops the guest actually uses, rather than the single hardcoded quad the
//!   `pageBacking` producer emits.
//! - `tail_nz_descs` / `tail_nz_bytes` — whether this build populates bytes
//!   `+0x08..0x18`, whose purpose is unrecovered. Zero across a boot settles that
//!   it does not.
//! - `lic_stored` / `lic_unstored` / `stored_lic` / `stored_unlic` — the
//!   correlation that decides whether `set_host_valid` means "this submission
//!   writes this resource". See [`note_table`] for what the store side is.
//!
//! # What one boot measured
//!
//! x86 Ventura guest / Linux Vulkan host, one `--testing` boot driven through
//! three `icon-composite` rounds under Safari load (all three CLEAN), summed
//! over its 106 windows:
//!
//! ```text
//! subs 19 219   descs 84 868   no_table 0        every submission carries a table
//! unresolved 4 411 (5.2 %)     id0 0             the ids are the device's object space
//! set_h 19 253   clr_h 15 423  clr_g 16  set_g 0
//! stored_lic 19 135            stored_unlic 0    every store was licensed
//! tail_nz_descs 0              tail_nz_bytes 0
//! quads: 0x00000000, 0x00000001, 0x00000100, 0x00000101, 0x00010000
//! ```
//!
//! Three readings, and they are the whole basis for consuming this table:
//!
//! 1. **`set_host_valid` marks exactly the resources a submission writes.** Not
//!    one of 19 135 stores landed on an object the table had not licensed. That
//!    was an inference from IOAccel resource-list usage before this boot; it is
//!    now a measurement with zero counter-examples.
//! 2. **The trailing 16 bytes are dead in this build.** Zero non-zero bytes
//!    across 84 868 records, so their unrecovered purpose costs nothing to
//!    ignore here — and a build that starts using them shows up as a non-zero
//!    count rather than as silence.
//! 3. **`clear_host_valid` arrives 15 423 times per boot.** That is the guest
//!    saying "I CPU-wrote this resource", delivered once per write and never
//!    resent, on a path that used to discard it.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Mutex;

use crate::runtime::decode::fifo::ExecResourceDesc;

/// Distinct validity quads kept per window before the rest fold into
/// `quad_other`. The guest has one hardcoded quad on the invalidate path; a
/// table that produces more than a handful of distinct ones is itself the
/// finding, and the count survives the cap even when the values do not.
const QUAD_HISTOGRAM_MAX: usize = 16;

/// Distinct object ids counted exactly per window before the rest fold into
/// `ids_over`. Bounds the window's memory against a submission storm without
/// losing the "is this a handful of resources or thousands" reading.
const DISTINCT_IDS_MAX: usize = 4096;

#[derive(Default)]
struct Window {
    /// Submissions whose table was decoded (including empty tables).
    subs: u64,
    /// Submissions declaring `resource_count == 0`.
    no_table: u64,
    descs: u64,
    max_count: u32,
    ids: BTreeSet<u32>,
    ids_over: u64,
    /// Records naming object id 0, which `writeInvalidates` should never emit.
    id_zero: u64,
    /// Records whose object id no registry answered for.
    unresolved: u64,
    /// Which registry answered, counted independently — an id can be in several.
    as_mapping: u64,
    as_texture: u64,
    as_object: u64,
    clear_host: u64,
    set_host: u64,
    clear_guest: u64,
    set_guest: u64,
    tail_nz_descs: u64,
    tail_nz_bytes: u64,
    quads: BTreeMap<u32, u64>,
    quad_other: u64,
    /// `set_host_valid` records whose object the submission did store into.
    lic_stored: u64,
    /// `set_host_valid` records whose object the submission did not store into.
    lic_unstored: u64,
    /// Objects the submission stored into that the table licensed.
    stored_lic: u64,
    /// Objects the submission stored into that the table did not license.
    stored_unlic: u64,
}

static WINDOW: Mutex<Option<Window>> = Mutex::new(None);

/// Which of the device's registries answer for one table object id.
///
/// Independent flags rather than a first-match verdict: the consumer of the
/// table has to pick one registry to key its state on, and a single "resolved"
/// bit cannot say which one that should be.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct IdKind {
    /// A live entry in `DeviceState::mappings`.
    pub mapping: bool,
    /// Has a `texture_to_mapping` entry under the submission's task.
    pub texture: bool,
    /// A live object ref under the submission's task.
    pub object: bool,
}

impl IdKind {
    pub fn any(self) -> bool {
        self.mapping || self.texture || self.object
    }
}

/// Record one submission's decoded resource table.
///
/// `resolve` answers which registries hold an object id, which only the caller
/// can decide — the census must not reach into device state.
///
/// `stored` is the set of object ids the submission's decoded command streams
/// actually stored into, used to test the inference that `set_host_valid` marks
/// the resources a submission writes. It carries both texture refs and the
/// mapping ids they resolve to, because the table's id namespace is not proven
/// identical to either: over-counting a match is the conservative direction for
/// a correlation whose interesting reading is a systematic *mismatch*.
pub fn note_table<F>(descs: &[ExecResourceDesc], stored: &[u32], mut resolve: F)
where
    F: FnMut(u32) -> IdKind,
{
    let Ok(mut guard) = WINDOW.lock() else {
        return;
    };
    let w = guard.get_or_insert_with(Window::default);
    w.subs += 1;
    if descs.is_empty() {
        w.no_table += 1;
    }
    w.descs += descs.len() as u64;
    w.max_count = w.max_count.max(descs.len() as u32);
    for d in descs {
        if d.object_id == 0 {
            w.id_zero += 1;
        }
        if w.ids.len() < DISTINCT_IDS_MAX {
            w.ids.insert(d.object_id);
        } else if !w.ids.contains(&d.object_id) {
            w.ids_over += 1;
        }
        let kind = resolve(d.object_id);
        w.as_mapping += u64::from(kind.mapping);
        w.as_texture += u64::from(kind.texture);
        w.as_object += u64::from(kind.object);
        if !kind.any() {
            w.unresolved += 1;
        }
        w.clear_host += u64::from(d.ops.clear_host_valid != 0);
        w.set_host += u64::from(d.ops.set_host_valid != 0);
        w.clear_guest += u64::from(d.ops.clear_guest_valid != 0);
        w.set_guest += u64::from(d.ops.set_guest_valid != 0);
        let nz = d.tail_nonzero_bytes();
        if nz > 0 {
            w.tail_nz_descs += 1;
            w.tail_nz_bytes += u64::from(nz);
        }
        if w.quads.len() < QUAD_HISTOGRAM_MAX || w.quads.contains_key(&d.flags) {
            *w.quads.entry(d.flags).or_default() += 1;
        } else {
            w.quad_other += 1;
        }
        if d.ops.set_host_valid != 0 {
            if stored.contains(&d.object_id) {
                w.lic_stored += 1;
            } else {
                w.lic_unstored += 1;
            }
        }
    }
    for id in stored {
        let licensed = descs
            .iter()
            .any(|d| d.object_id == *id && d.ops.set_host_valid != 0);
        if licensed {
            w.stored_lic += 1;
        } else {
            w.stored_unlic += 1;
        }
    }
}

/// Drain the window into one line, or `None` if no submission landed in it.
pub fn take_window() -> Option<String> {
    let mut guard = WINDOW.lock().ok()?;
    let w = guard.take()?;
    if w.subs == 0 {
        return None;
    }
    Some(format_line(&w))
}

fn format_line(w: &Window) -> String {
    use std::fmt::Write as _;
    let mut line = format!(
        "exec_res_table subs={} no_table={} descs={} max={} ids={} ids_over={} id0={} \
         unresolved={} as_map={} as_tex={} as_obj={} \
         clr_h={} set_h={} clr_g={} set_g={} tail_nz_descs={} tail_nz_bytes={} \
         lic_stored={} lic_unstored={} stored_lic={} stored_unlic={}",
        w.subs,
        w.no_table,
        w.descs,
        w.max_count,
        w.ids.len(),
        w.ids_over,
        w.id_zero,
        w.unresolved,
        w.as_mapping,
        w.as_texture,
        w.as_object,
        w.clear_host,
        w.set_host,
        w.clear_guest,
        w.set_guest,
        w.tail_nz_descs,
        w.tail_nz_bytes,
        w.lic_stored,
        w.lic_unstored,
        w.stored_lic,
        w.stored_unlic
    );
    line.push_str(" quads=[");
    for (i, (quad, n)) in w.quads.iter().enumerate() {
        if i > 0 {
            line.push(',');
        }
        let _ = write!(line, "{quad:#010x}:{n}");
    }
    let _ = write!(line, "] quad_other={}", w.quad_other);
    line
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::decode::fifo::InvalidateValidityOps;

    fn desc(object_id: u32, flags: u32, tail: [u8; 16]) -> ExecResourceDesc {
        ExecResourceDesc {
            object_id,
            flags,
            ops: InvalidateValidityOps::from_le_dword(flags),
            tail,
        }
    }

    /// The census must survive the whole window before a reader sees it, and the
    /// line must name every field: a counter that exists but is not printed is a
    /// counter no boot can read.
    #[test]
    fn window_accumulates_and_names_every_field() {
        let _ = take_window();
        let mut tail = [0u8; 16];
        tail[3] = 9;
        note_table(
            &[
                desc(0x10, 0x0000_0001, [0u8; 16]),
                desc(0x11, 0x0000_0100, tail),
                desc(0, 0x0000_0100, [0u8; 16]),
            ],
            &[0x11, 0x12],
            |id| IdKind {
                mapping: id == 0x10,
                texture: id == 0,
                object: false,
            },
        );
        let line = take_window().expect("one submission landed");
        assert!(line.starts_with("exec_res_table subs=1 "), "{line}");
        assert!(line.contains(" no_table=0 "), "{line}");
        assert!(line.contains(" descs=3 "), "{line}");
        assert!(line.contains(" max=3 "), "{line}");
        // 0x10, 0x11 and 0 are three distinct ids; one of them is the illegal 0.
        assert!(line.contains(" ids=3 "), "{line}");
        assert!(line.contains(" id0=1 "), "{line}");
        assert!(line.contains(" unresolved=1 "), "{line}");
        assert!(line.contains(" as_map=1 "), "{line}");
        assert!(line.contains(" as_tex=1 "), "{line}");
        assert!(line.contains(" as_obj=0 "), "{line}");
        assert!(line.contains(" clr_h=1 "), "{line}");
        assert!(line.contains(" set_h=2 "), "{line}");
        assert!(line.contains(" clr_g=0 "), "{line}");
        assert!(line.contains(" set_g=0 "), "{line}");
        assert!(line.contains(" tail_nz_descs=1 "), "{line}");
        assert!(line.contains(" tail_nz_bytes=1 "), "{line}");
        // 0x11 is licensed and stored; the id-0 record is licensed and unstored.
        assert!(line.contains(" lic_stored=1 "), "{line}");
        assert!(line.contains(" lic_unstored=1 "), "{line}");
        // 0x11 was licensed by the table, 0x12 was not in it at all.
        assert!(line.contains(" stored_lic=1 "), "{line}");
        assert!(line.contains(" stored_unlic=1 "), "{line}");
        assert!(line.contains(" quads=[0x00000001:1,0x00000100:2]"), "{line}");
        assert!(line.ends_with(" quad_other=0"), "{line}");
    }

    /// The window is a rate over that window, not a lifetime total: a second
    /// drain with no traffic in between must report nothing rather than repeat.
    #[test]
    fn window_resets_on_drain() {
        let _ = take_window();
        note_table(&[desc(1, 0, [0u8; 16])], &[], |_| IdKind::default());
        assert!(take_window().is_some());
        assert!(take_window().is_none());
    }

    /// A submission that declares no resources is still a submission — counting
    /// only the non-empty ones would hide "the table is always empty", which is
    /// the reading that stops all of this.
    #[test]
    fn an_empty_table_still_counts_the_submission() {
        let _ = take_window();
        note_table(&[], &[], |_| IdKind::default());
        let line = take_window().expect("line");
        assert!(line.contains("subs=1 no_table=1 descs=0 "), "{line}");
    }

    /// A quad storm must bound the histogram without losing the record count.
    #[test]
    fn quad_histogram_is_capped_and_the_overflow_is_counted() {
        let _ = take_window();
        let descs: Vec<_> = (0..(QUAD_HISTOGRAM_MAX as u32 + 5))
            .map(|i| desc(i + 1, i + 1, [0u8; 16]))
            .collect();
        note_table(&descs, &[], |_| IdKind::default());
        let line = take_window().expect("line");
        assert!(line.contains(" quad_other=5"), "{line}");
        assert_eq!(
            line.matches(':').count(),
            QUAD_HISTOGRAM_MAX,
            "histogram must hold exactly the cap: {line}"
        );
    }
}
