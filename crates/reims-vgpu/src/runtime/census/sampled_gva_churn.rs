//! Measure-only gva-churn probe for the padded-stride sampled fallback.
//!
//! The Safari-scroll hot path is `lin_guest_fb` — padded-stride BGRA8 glyph /
//! tile textures that decline the tight-stride memo (`load_linear_guest_memoized`
//! requires `bpr == tight`) and pay a full guest read + fresh allocation + engine
//! content-hash on every bind (no content identity). A gva-keyed
//! content-revalidated memo (like the tight path already has) could serve the
//! repeat binds — but only if a texture's authoritative gva actually *recurs*
//! across frames. If Safari rotates glyph-atlas backing (fresh gva per frame),
//! the memo can never hit and only adds byte-compare + store churn.
//!
//! This probe answers the churn question CHEAPLY — a bounded LRU of the recent
//! `(task_id, gva, w, h)` keys, no content hashing — so a boot log shows what
//! fraction of padded-fallback binds re-present a key already seen (the ceiling
//! on any gva-keyed memo's hit rate). Count-based, so it is trustworthy under the
//! agent's `SCHED_IDLE` contamination; runs on the drain worker (off the QEMU
//! main core). It NEVER gates behavior — it only records.
//!
//! `fallback_gva_churn` lines land in `/tmp/reims-vgpu-fail.log` (always-on
//! `observe::off`) every [`EMIT_EVERY`] padded-fallback binds.

use std::collections::HashSet;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

/// Bounded recent-key window. Large enough to cover a scroll frame's live
/// glyph/tile working set (tens of KB of distinct tiles) without unbounded
/// growth; a real memo would be byte-capped, not key-capped, but this ceiling
/// is what matters for the churn ratio.
const WINDOW: usize = 8192;

/// One cumulative `fallback_gva_churn` line per this many padded-fallback binds.
const EMIT_EVERY: u64 = 1024;

struct Churn {
    /// Insertion-ordered keys for LRU eviction; `set` mirrors membership.
    order: VecDeque<(u32, u64, u32, u32)>,
    set: HashSet<(u32, u64, u32, u32)>,
}

static CHURN: Mutex<Option<Churn>> = Mutex::new(None);
/// A key already present in the recent window (a memo *could* have hit).
static REPEAT: AtomicU64 = AtomicU64::new(0);
/// A key not seen in the recent window (a memo would miss — fresh gva/geom).
static FRESH: AtomicU64 = AtomicU64::new(0);

/// Record one padded-stride fallback bind by its authoritative key. Returns
/// nothing; the aggregate ratio is emitted to the always-on sink.
pub fn note(task_id: u32, gva: u64, width: u32, height: u32) {
    let key = (task_id, gva, width, height);
    let repeated = {
        let mut guard = CHURN.lock().unwrap_or_else(|e| e.into_inner());
        let churn = guard.get_or_insert_with(|| Churn {
            order: VecDeque::with_capacity(WINDOW),
            set: HashSet::with_capacity(WINDOW),
        });
        if churn.set.contains(&key) {
            true
        } else {
            churn.set.insert(key);
            churn.order.push_back(key);
            if churn.order.len() > WINDOW {
                if let Some(old) = churn.order.pop_front() {
                    churn.set.remove(&old);
                }
            }
            false
        }
    };
    let (repeat, fresh) = if repeated {
        (
            REPEAT.fetch_add(1, Ordering::Relaxed) + 1,
            FRESH.load(Ordering::Relaxed),
        )
    } else {
        (
            REPEAT.load(Ordering::Relaxed),
            FRESH.fetch_add(1, Ordering::Relaxed) + 1,
        )
    };
    let total = repeat + fresh;
    if total % EMIT_EVERY == 0 {
        let pct = repeat.saturating_mul(100).checked_div(total).unwrap_or(0);
        crate::observe::off(format!(
            "fallback_gva_churn total={total} repeat={repeat} fresh={fresh} repeat_pct={pct} window={WINDOW}"
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeat_key_counts_as_repeat_fresh_key_as_fresh() {
        // Global counters are shared across parallel tests: assert deltas.
        let r0 = REPEAT.load(Ordering::Relaxed);
        let f0 = FRESH.load(Ordering::Relaxed);
        // A distinct gva (unlikely to collide with concurrent tests) seen twice.
        let gva = 0xDEAD_0000_0000_0000u64 | (std::process::id() as u64) << 8;
        note(7, gva, 33, 17);
        note(7, gva, 33, 17);
        // A different geometry at the same gva is a distinct key → fresh.
        note(7, gva, 34, 17);
        let r = REPEAT.load(Ordering::Relaxed) - r0;
        let f = FRESH.load(Ordering::Relaxed) - f0;
        assert_eq!(r, 1, "the second identical bind is a repeat");
        assert_eq!(f, 2, "first bind + the distinct-geometry bind are fresh");
    }
}
