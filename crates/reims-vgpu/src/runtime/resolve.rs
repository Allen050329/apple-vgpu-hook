//! Resolver and synchronization traits used by device/backend layers.

use crate::contract::gva_resolve::{Cache, Geometry, PhysReader, ResolveStatus, Translation};
use crate::runtime::plan::event_sync::{EventState, PlannedOp, Status as EventStatus};

/// Trait for resolving guest virtual addresses through a task page table.
pub trait GvaResolver {
    fn translate(
        &mut self,
        root_pfn: u32,
        depth: u32,
        gva: u64,
    ) -> Result<Translation, ResolveStatus>;
}

/// Default GVA resolver wrapping geometry + cache + phys reader.
pub struct DefaultGvaResolver<'a, R: PhysReader> {
    pub reader: &'a R,
    pub geometry: &'a Geometry,
    pub cache: Cache,
}

impl<'a, R: PhysReader> GvaResolver for DefaultGvaResolver<'a, R> {
    fn translate(
        &mut self,
        root_pfn: u32,
        depth: u32,
        gva: u64,
    ) -> Result<Translation, ResolveStatus> {
        let t = crate::contract::gva_resolve::translate_root(
            self.reader,
            self.geometry,
            root_pfn,
            depth,
            gva,
            Some(&mut self.cache),
        );
        if t.status == ResolveStatus::Ok {
            Ok(t)
        } else {
            Err(t.status)
        }
    }
}

/// Trait for applying planned event ops against host/device event state.
pub trait EventSync {
    fn apply(&mut self, op: &PlannedOp) -> EventStatus;
}

impl EventSync for EventState {
    fn apply(&mut self, op: &PlannedOp) -> EventStatus {
        crate::runtime::plan::event_sync::apply_planned(self, op)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::plan::event_sync::PlannedOp;

    #[test]
    fn event_sync_trait() {
        let mut st = EventState::new();
        assert_eq!(
            st.apply(&PlannedOp::Signal {
                event_ref: 1,
                value: 3
            }),
            EventStatus::Ok
        );
        assert!(st.wait_satisfied(1, 3));
    }
}
