//! Resolve the wire task word a command payload carries to a live task slot.
//!
//! Three child commands each carry a task word in their payload and each had
//! its own private copy of the same six lines:
//!
//! ```text
//! if tasks[raw].active { raw } else { raw >> 1 }
//! ```
//!
//! `CmdExecIndirect2` (`0x35`), `CmdGetComputeInfo` (`0x3b`) and
//! `CmdHeapTextureSizeAndAlign` — one resolver spelled three times, so a reading
//! taken at one site said nothing about the other two. They are one function
//! here, and that function carries the census.
//!
//! # The question this measures
//!
//! `DefineTask2` (`0x38`) registers a task under `raw >> 1`
//! ([`crate::model::DEFINE_TASK_ID_SHIFT`]); every other opcode measured so far
//! reads its task word **unshifted**. Nothing decodes the word canonically, so
//! for each opcode the open question is which space its word is in.
//!
//! `MapMemory2`'s word was settled by reading the *set* of words it files spans
//! under: it contains odd values, and a doubled space `2n` cannot produce an odd
//! value, so `MapMemory2` already names slots. The same one-sided argument works
//! here, which is why the census latches on the **raw word** and not on the
//! outcome. A site whose words are all even and all twice a live slot is
//! plausibly in the doubled space and its raw-first arm is then wrong every
//! time; a site with a single odd word is naming slots directly and raw-first is
//! right. The reading can come out either way, which is what makes it worth
//! taking.
//!
//! Slots run densely from 0, so for almost any word `n` the slot `n >> 1` is
//! *also* live and raw-first wins by position rather than by evidence. Counting
//! how often the fallback arm *ran* cannot see that — it is a property of
//! `tasks[]` at the instant of the decode, so it is read from the table.

use crate::model::{TaskEntry, DEFINE_TASK_ID_SHIFT};

/// Which command carried the task word. Distinguishes otherwise identical
/// decodes so a per-site set difference is possible.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TaskWordSite {
    /// `CmdExecIndirect2` (`0x35`) — the write-carrying one.
    ExecIndirect2,
    /// `CmdGetComputeInfo` (`0x3b`).
    ComputeInfo,
    /// `CmdHeapTextureSizeAndAlign`.
    HeapTextureQuery,
}

impl TaskWordSite {
    fn name(self) -> &'static str {
        match self {
            Self::ExecIndirect2 => "exec_indirect2",
            Self::ComputeInfo => "compute_info",
            Self::HeapTextureQuery => "heap_texture_query",
        }
    }
}

/// How the wire word resolved against the task table. A total partition of the
/// four ways the two candidate slots can be live, so the union of the four
/// slugs' `raw=` values is the complete set of words a site received.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TaskWordDecode {
    /// The word named a live slot and `word >> 1` did not. Unambiguous.
    Direct,
    /// The word named a live slot and so did `word >> 1`. Raw-first won by
    /// position: had the guest meant the shifted slot, nothing here could tell.
    Ambiguous,
    /// The word named no live slot and `word >> 1` did. The fallback arm is the
    /// only thing that produced an answer.
    Shifted,
    /// Neither slot is live. The resolver has no answer and the caller refuses.
    Dead,
}

impl crate::observe::Decline for TaskWordDecode {
    fn slug(&self) -> &'static str {
        match self {
            Self::Direct => "cmd_task_direct",
            Self::Ambiguous => "cmd_task_ambiguous",
            Self::Shifted => "cmd_task_shifted",
            Self::Dead => "cmd_task_dead",
        }
    }
}

/// True when `id` indexes an active slot.
fn slot_live(tasks: &[TaskEntry], id: u32) -> bool {
    (id as usize) < tasks.len() && tasks[id as usize].active
}

/// Resolve the wire task word to the slot this crate will act on.
///
/// **Behaviour is unchanged from the three copies this replaces**: the raw word
/// wins when it names a live slot, otherwise `raw >> 1` is returned whether or
/// not it is live. Callers re-check liveness and refuse — that check is theirs
/// because each has its own typed refusal to emit.
///
/// The latch is taken before the line is built. `Emit::field` renders eagerly
/// and this sits on the command path, so building and dropping the strings on
/// every decode would make the probe cost scale with the traffic it measures.
pub(crate) fn resolve_task_word(tasks: &[TaskEntry], site: TaskWordSite, raw: u32) -> u32 {
    use crate::observe::Decline;
    let shifted = raw >> DEFINE_TASK_ID_SHIFT;
    let raw_live = slot_live(tasks, raw);
    let shifted_live = shifted != raw && slot_live(tasks, shifted);
    let decode = match (raw_live, shifted_live) {
        (true, false) => TaskWordDecode::Direct,
        (true, true) => TaskWordDecode::Ambiguous,
        (false, true) => TaskWordDecode::Shifted,
        (false, false) => TaskWordDecode::Dead,
    };
    let discriminant = ((site as u64) << 32) | u64::from(raw);
    if crate::observe::first_sight(decode.slug(), discriminant) {
        crate::observe::Emit::decline("cmd_task", &decode)
            .field("site", site.name())
            .field("raw", format!("{raw:#x}"))
            .field("shifted", shifted)
            .fail();
    }
    if raw_live {
        raw
    } else {
        shifted
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table(active: &[u32]) -> Vec<TaskEntry> {
        let mut tasks = vec![TaskEntry::default(); 16];
        for &id in active {
            tasks[id as usize].active = true;
        }
        tasks
    }

    /// The four ways the two candidate slots can be live are a partition, and
    /// each one names itself. A census that collapsed any two of them could not
    /// answer "was raw-first forced, or did it merely win the race".
    #[test]
    fn every_way_the_two_candidate_slots_can_be_live_has_its_own_name() {
        use crate::observe::Decline;
        let seen: Vec<&str> = [
            TaskWordDecode::Direct,
            TaskWordDecode::Ambiguous,
            TaskWordDecode::Shifted,
            TaskWordDecode::Dead,
        ]
        .iter()
        .map(|d| d.slug())
        .collect();
        let mut uniq = seen.clone();
        uniq.sort_unstable();
        uniq.dedup();
        assert_eq!(seen.len(), uniq.len(), "two decodes share a slug: {seen:?}");
    }

    /// Slot 5 live, slot 2 not: only one candidate, raw-first is not a choice.
    #[test]
    fn a_word_whose_shifted_slot_is_dead_resolves_directly() {
        let tasks = table(&[5]);
        assert_eq!(
            resolve_task_word(&tasks, TaskWordSite::ExecIndirect2, 5),
            5
        );
    }

    /// Slots 3 and 6 both live: the guest sent `6`, we act on `6`, and slot 3 is
    /// equally available. This is the case the census exists to count — the
    /// resolver cannot distinguish it from the previous one on its own.
    #[test]
    fn a_word_whose_shifted_slot_is_also_live_still_resolves_raw_first() {
        let tasks = table(&[3, 6]);
        assert_eq!(
            resolve_task_word(&tasks, TaskWordSite::ExecIndirect2, 6),
            6
        );
    }

    /// Slot 6 dead, slot 3 live: the fallback arm is the only one with an
    /// answer, and it is what the three replaced copies returned.
    #[test]
    fn a_word_naming_no_live_slot_falls_back_to_the_shifted_one() {
        let tasks = table(&[3]);
        assert_eq!(
            resolve_task_word(&tasks, TaskWordSite::ComputeInfo, 6),
            3
        );
    }

    /// Neither slot live: the resolver still returns the shifted id, unchanged
    /// from the copies it replaces, and every caller re-checks and refuses.
    #[test]
    fn a_word_naming_nothing_live_returns_the_shifted_id_for_the_caller_to_refuse() {
        let tasks = table(&[]);
        assert_eq!(
            resolve_task_word(&tasks, TaskWordSite::HeapTextureQuery, 6),
            3
        );
        assert!(!slot_live(&tasks, 3));
    }

    /// Word 0 shifts to itself, so there is no second candidate and the decode
    /// must not report an ambiguity against its own slot.
    #[test]
    fn word_zero_has_no_second_candidate() {
        let tasks = table(&[0]);
        assert_eq!(
            resolve_task_word(&tasks, TaskWordSite::ExecIndirect2, 0),
            0
        );
    }

    /// The latch key mixes the site in, so the same word arriving at two sites
    /// is two sightings. Without this a word first seen at exec would silence
    /// the compute-info reading and the per-site set difference would be lost.
    #[test]
    fn the_latch_key_separates_the_same_word_at_different_sites() {
        let key = |site: TaskWordSite, raw: u32| ((site as u64) << 32) | u64::from(raw);
        assert_ne!(
            key(TaskWordSite::ExecIndirect2, 6),
            key(TaskWordSite::ComputeInfo, 6)
        );
        assert_ne!(
            key(TaskWordSite::ComputeInfo, 6),
            key(TaskWordSite::HeapTextureQuery, 6)
        );
    }
}
