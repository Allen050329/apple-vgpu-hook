//! Event/fence synchronization planner (port of `host/utils/reims-vgpu-event-sync`).
//!
//! Pure planner + fixed-geometry state table matching the C package contract.
//! Decoder-aware adapters live in the QEMU C ABI layer (they need C command layouts).

use crate::runtime::decode::event::{Command as EventCommand, Kind as DecodedEventKind};

pub const FENCE_INITIAL_GENERATION: u64 = 1;
pub const STATE_SETS: usize = 256;
pub const STATE_WAYS: usize = 4;
const STATE_TASK_HASH_MULT: u32 = 17;
const STATE_DOMAIN_HASH_MULT: u32 = 131;

pub const TRACE_HAS_CURRENT: u32 = 1 << 0;
pub const TRACE_HAS_TIMEOUT: u32 = 1 << 1;
pub const TRACE_UPDATES_STATE: u32 = 1 << 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[repr(u32)]
pub enum Domain {
    #[default]
    Unknown = 0,
    Event = 1,
    BlitFence = 2,
    ComputeFence = 3,
    RenderFence = 4,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[repr(u32)]
pub enum Operation {
    #[default]
    Unknown = 0,
    Signal = 1,
    Wait = 2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[repr(u32)]
pub enum Decision {
    #[default]
    Invalid = 0,
    SignalUpdate = 1,
    SignalNoop = 2,
    WaitSatisfied = 3,
    WaitPending = 4,
    WaitTimeoutUnsupported = 5,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[repr(u32)]
pub enum Reason {
    #[default]
    Invalid = 0,
    SignalFirst = 1,
    SignalAdvance = 2,
    SignalEqualIgnored = 3,
    SignalStaleIgnored = 4,
    WaitReached = 5,
    WaitMissingSignal = 6,
    WaitBelowTarget = 7,
    WaitTimeoutUnsupported = 8,
    FenceUpdateFirst = 9,
    FenceUpdateAdvance = 10,
    FenceUpdateAtMax = 11,
    FenceWaitReached = 12,
    FenceWaitMissingUpdate = 13,
    BadFenceDomain = 14,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[repr(u32)]
pub enum EventKind {
    #[default]
    Unknown = 0,
    Signal = 1,
    Wait = 2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[repr(u32)]
pub enum FenceAction {
    #[default]
    Unknown = 0,
    Update = 1,
    Wait = 2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[repr(u32)]
pub enum EventTraceKind {
    #[default]
    None = 0,
    Signal = 1,
    Wait = 2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Key {
    pub domain: Domain,
    pub ref_: u32,
    pub stages: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct ValueSnapshot {
    pub valid: bool,
    pub value: u64,
}

impl ValueSnapshot {
    pub fn absent() -> Self {
        Self {
            valid: false,
            value: 0,
        }
    }

    pub fn current(value: u64) -> Self {
        Self { valid: true, value }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Plan {
    pub domain: Domain,
    pub operation: Operation,
    pub decision: Decision,
    pub reason: Reason,
    pub ref_: u32,
    pub stages: u32,
    pub has_current: bool,
    pub current_value: u64,
    pub target_value: u64,
    pub has_timeout: bool,
    pub timeout: u32,
    pub updates_state: bool,
    pub update_value: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct StateEntry {
    pub valid: bool,
    pub task_id: u32,
    pub domain: Domain,
    pub ref_: u32,
    pub value: u64,
}

#[derive(Clone, Debug)]
pub struct StateTable {
    pub entries: [[StateEntry; STATE_WAYS]; STATE_SETS],
    pub next: [u32; STATE_SETS],
}

impl Default for StateTable {
    fn default() -> Self {
        Self {
            entries: [[StateEntry::default(); STATE_WAYS]; STATE_SETS],
            next: [0; STATE_SETS],
        }
    }
}

impl StateTable {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn clear(&mut self) {
        *self = Self::default();
    }
}

fn is_fence_domain(domain: Domain) -> bool {
    matches!(
        domain,
        Domain::BlitFence | Domain::ComputeFence | Domain::RenderFence
    )
}

fn normalize_snapshot(current: Option<ValueSnapshot>) -> ValueSnapshot {
    current.unwrap_or_else(ValueSnapshot::absent)
}

fn base_plan(
    domain: Domain,
    operation: Operation,
    ref_: u32,
    stages: u32,
    current: Option<ValueSnapshot>,
) -> Plan {
    let snapshot = normalize_snapshot(current);
    Plan {
        domain,
        operation,
        decision: Decision::Invalid,
        reason: Reason::Invalid,
        ref_,
        stages,
        has_current: snapshot.valid,
        current_value: if snapshot.valid { snapshot.value } else { 0 },
        target_value: 0,
        has_timeout: false,
        timeout: 0,
        updates_state: false,
        update_value: 0,
    }
}

pub fn plan_event_signal(event_ref: u32, value: u64, current: Option<ValueSnapshot>) -> Plan {
    let mut plan = base_plan(Domain::Event, Operation::Signal, event_ref, 0, current);
    plan.target_value = value;

    if !plan.has_current {
        plan.decision = Decision::SignalUpdate;
        plan.reason = Reason::SignalFirst;
        plan.updates_state = true;
        plan.update_value = value;
    } else if value > plan.current_value {
        plan.decision = Decision::SignalUpdate;
        plan.reason = Reason::SignalAdvance;
        plan.updates_state = true;
        plan.update_value = value;
    } else if value == plan.current_value {
        plan.decision = Decision::SignalNoop;
        plan.reason = Reason::SignalEqualIgnored;
        plan.update_value = plan.current_value;
    } else {
        plan.decision = Decision::SignalNoop;
        plan.reason = Reason::SignalStaleIgnored;
        plan.update_value = plan.current_value;
    }
    plan
}

pub fn plan_event_wait(
    event_ref: u32,
    value: u64,
    has_timeout: bool,
    timeout: u32,
    current: Option<ValueSnapshot>,
) -> Plan {
    let mut plan = base_plan(Domain::Event, Operation::Wait, event_ref, 0, current);
    plan.target_value = value;
    plan.has_timeout = has_timeout;
    plan.timeout = if has_timeout { timeout } else { 0 };

    if plan.has_current && plan.current_value >= value {
        plan.decision = Decision::WaitSatisfied;
        plan.reason = Reason::WaitReached;
    } else if has_timeout {
        plan.decision = Decision::WaitTimeoutUnsupported;
        plan.reason = Reason::WaitTimeoutUnsupported;
    } else {
        plan.decision = Decision::WaitPending;
        plan.reason = if plan.has_current {
            Reason::WaitBelowTarget
        } else {
            Reason::WaitMissingSignal
        };
    }
    plan
}

pub fn plan_event(
    kind: EventKind,
    event_ref: u32,
    value: u64,
    has_timeout: bool,
    timeout: u32,
    current: Option<ValueSnapshot>,
) -> Plan {
    match kind {
        EventKind::Signal => plan_event_signal(event_ref, value, current),
        EventKind::Wait => plan_event_wait(event_ref, value, has_timeout, timeout, current),
        EventKind::Unknown => base_plan(Domain::Event, Operation::Unknown, event_ref, 0, current),
    }
}

pub fn plan_fence_update(
    domain: Domain,
    fence_ref: u32,
    stages: u32,
    current: Option<ValueSnapshot>,
) -> Plan {
    let mut plan = base_plan(domain, Operation::Signal, fence_ref, stages, current);
    if !is_fence_domain(domain) {
        plan.reason = Reason::BadFenceDomain;
        return plan;
    }
    if !plan.has_current {
        plan.decision = Decision::SignalUpdate;
        plan.reason = Reason::FenceUpdateFirst;
        plan.updates_state = true;
        plan.target_value = FENCE_INITIAL_GENERATION;
        plan.update_value = FENCE_INITIAL_GENERATION;
    } else if plan.current_value == u64::MAX {
        plan.decision = Decision::SignalNoop;
        plan.reason = Reason::FenceUpdateAtMax;
        plan.target_value = plan.current_value;
        plan.update_value = plan.current_value;
    } else {
        plan.decision = Decision::SignalUpdate;
        plan.reason = Reason::FenceUpdateAdvance;
        plan.updates_state = true;
        plan.target_value = plan.current_value + 1;
        plan.update_value = plan.current_value + 1;
    }
    plan
}

pub fn plan_fence_wait(
    domain: Domain,
    fence_ref: u32,
    stages: u32,
    current: Option<ValueSnapshot>,
) -> Plan {
    let mut plan = base_plan(domain, Operation::Wait, fence_ref, stages, current);
    if !is_fence_domain(domain) {
        plan.reason = Reason::BadFenceDomain;
        return plan;
    }
    if plan.has_current {
        plan.decision = Decision::WaitSatisfied;
        plan.reason = Reason::FenceWaitReached;
        plan.target_value = plan.current_value;
    } else {
        plan.decision = Decision::WaitPending;
        plan.reason = Reason::FenceWaitMissingUpdate;
        plan.target_value = FENCE_INITIAL_GENERATION;
    }
    plan
}

pub fn plan_fence(
    action: FenceAction,
    domain: Domain,
    fence_ref: u32,
    stages: u32,
    current: Option<ValueSnapshot>,
) -> Plan {
    match action {
        FenceAction::Update => plan_fence_update(domain, fence_ref, stages, current),
        FenceAction::Wait => plan_fence_wait(domain, fence_ref, stages, current),
        FenceAction::Unknown => base_plan(domain, Operation::Unknown, fence_ref, stages, current),
    }
}

pub fn plan_update_is_consistent(plan: &Plan) -> bool {
    if !plan.updates_state
        || plan.operation != Operation::Signal
        || plan.decision != Decision::SignalUpdate
        || plan.update_value != plan.target_value
    {
        return false;
    }
    match plan.domain {
        Domain::Event => matches!(plan.reason, Reason::SignalFirst | Reason::SignalAdvance),
        Domain::BlitFence | Domain::ComputeFence | Domain::RenderFence => matches!(
            plan.reason,
            Reason::FenceUpdateFirst | Reason::FenceUpdateAdvance
        ),
        Domain::Unknown => false,
    }
}

pub fn plan_updates_state(plan: &Plan) -> bool {
    plan_update_is_consistent(plan)
}

pub fn plan_allows_execution(plan: &Plan) -> bool {
    !matches!(
        plan.decision,
        Decision::WaitPending | Decision::WaitTimeoutUnsupported | Decision::Invalid
    )
}

pub fn plan_trace_flags(plan: &Plan) -> u32 {
    let mut flags = 0u32;
    if plan.has_current {
        flags |= TRACE_HAS_CURRENT;
    }
    if plan.has_timeout {
        flags |= TRACE_HAS_TIMEOUT;
    }
    if plan.updates_state {
        flags |= TRACE_UPDATES_STATE;
    }
    flags
}

pub fn plan_event_trace_kind(plan: &Plan) -> EventTraceKind {
    if plan.domain != Domain::Event {
        return EventTraceKind::None;
    }
    match plan.operation {
        Operation::Signal => EventTraceKind::Signal,
        Operation::Wait => EventTraceKind::Wait,
        Operation::Unknown => EventTraceKind::None,
    }
}

pub fn plan_wait_satisfied_trace_value(plan: &Plan) -> u32 {
    if plan.operation == Operation::Wait && plan.decision == Decision::WaitSatisfied {
        1
    } else {
        0
    }
}

pub fn apply_update(plan: &Plan, state: &mut ValueSnapshot) {
    if !plan_update_is_consistent(plan) {
        return;
    }
    state.valid = true;
    state.value = plan.update_value;
}

pub fn domain_name(domain: Domain) -> &'static str {
    match domain {
        Domain::Event => "event",
        Domain::BlitFence => "blitFence",
        Domain::ComputeFence => "computeFence",
        Domain::RenderFence => "renderFence",
        Domain::Unknown => "unknown",
    }
}

pub fn operation_name(operation: Operation) -> &'static str {
    match operation {
        Operation::Signal => "signal",
        Operation::Wait => "wait",
        Operation::Unknown => "unknown",
    }
}

pub fn decision_name(decision: Decision) -> &'static str {
    match decision {
        Decision::SignalUpdate => "signalUpdate",
        Decision::SignalNoop => "signalNoop",
        Decision::WaitSatisfied => "waitSatisfied",
        Decision::WaitPending => "waitPending",
        Decision::WaitTimeoutUnsupported => "waitTimeoutUnsupported",
        Decision::Invalid => "invalid",
    }
}

pub fn reason_name(reason: Reason) -> &'static str {
    match reason {
        Reason::SignalFirst => "signalFirst",
        Reason::SignalAdvance => "signalAdvance",
        Reason::SignalEqualIgnored => "signalEqualIgnored",
        Reason::SignalStaleIgnored => "signalStaleIgnored",
        Reason::WaitReached => "waitReached",
        Reason::WaitMissingSignal => "waitMissingSignal",
        Reason::WaitBelowTarget => "waitBelowTarget",
        Reason::WaitTimeoutUnsupported => "waitTimeoutUnsupported",
        Reason::FenceUpdateFirst => "fenceUpdateFirst",
        Reason::FenceUpdateAdvance => "fenceUpdateAdvance",
        Reason::FenceUpdateAtMax => "fenceUpdateAtMax",
        Reason::FenceWaitReached => "fenceWaitReached",
        Reason::FenceWaitMissingUpdate => "fenceWaitMissingUpdate",
        Reason::BadFenceDomain => "badFenceDomain",
        Reason::Invalid => "invalid",
    }
}

fn state_set(task_id: u32, domain: Domain, ref_: u32) -> usize {
    ((ref_
        .wrapping_add(task_id.wrapping_mul(STATE_TASK_HASH_MULT))
        .wrapping_add((domain as u32).wrapping_mul(STATE_DOMAIN_HASH_MULT)))
        % STATE_SETS as u32) as usize
}

pub fn state_snapshot(
    table: &StateTable,
    task_id: u32,
    domain: Domain,
    ref_: u32,
) -> ValueSnapshot {
    let set = state_set(task_id, domain, ref_);
    for way in 0..STATE_WAYS {
        let e = &table.entries[set][way];
        if e.valid && e.task_id == task_id && e.domain == domain && e.ref_ == ref_ {
            return ValueSnapshot::current(e.value);
        }
    }
    ValueSnapshot::absent()
}

pub fn state_snapshot_key(table: &StateTable, task_id: u32, key: &Key) -> ValueSnapshot {
    state_snapshot(table, task_id, key.domain, key.ref_)
}

pub fn state_apply_plan(table: &mut StateTable, task_id: u32, plan: &Plan) {
    if !plan_updates_state(plan) {
        return;
    }
    let set = state_set(task_id, plan.domain, plan.ref_);
    let mut found = None;
    for way in 0..STATE_WAYS {
        let e = &table.entries[set][way];
        if e.valid && e.task_id == task_id && e.domain == plan.domain && e.ref_ == plan.ref_ {
            found = Some(way);
            break;
        }
    }
    let way = found.unwrap_or_else(|| {
        for way in 0..STATE_WAYS {
            if !table.entries[set][way].valid {
                return way;
            }
        }
        let way = (table.next[set] as usize) % STATE_WAYS;
        table.next[set] = table.next[set].wrapping_add(1);
        way
    });
    let e = &mut table.entries[set][way];
    e.valid = true;
    e.task_id = task_id;
    e.domain = plan.domain;
    e.ref_ = plan.ref_;
    e.value = plan.update_value;
}

pub fn state_wait_satisfied(
    table: &StateTable,
    task_id: u32,
    domain: Domain,
    ref_: u32,
    target_value: u64,
) -> bool {
    let snap = state_snapshot(table, task_id, domain, ref_);
    snap.valid && snap.value >= target_value
}

/// Plan a decoded event command (proto-side helper; C adapter mirrors this).
pub fn plan_decoded_event(cmd: &EventCommand, current: Option<ValueSnapshot>) -> Plan {
    let kind = match cmd.kind {
        DecodedEventKind::SignalEvent => EventKind::Signal,
        DecodedEventKind::WaitEvent => EventKind::Wait,
        DecodedEventKind::Unknown => EventKind::Unknown,
    };
    plan_event(
        kind,
        cmd.event_ref,
        cmd.value,
        cmd.has_timeout,
        cmd.timeout,
        current,
    )
}

// Backward-compatible thin wrappers used by earlier simplified port tests.
#[derive(Clone, Debug, Default)]
pub struct EventState {
    table: StateTable,
    task_id: u32,
}

impl EventState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get(&self, event_ref: u32) -> u64 {
        let s = state_snapshot(&self.table, self.task_id, Domain::Event, event_ref);
        if s.valid {
            s.value
        } else {
            0
        }
    }

    pub fn signal(&mut self, event_ref: u32, value: u64) -> Status {
        let current = state_snapshot(&self.table, self.task_id, Domain::Event, event_ref);
        let plan = plan_event_signal(event_ref, value, Some(current));
        state_apply_plan(&mut self.table, self.task_id, &plan);
        Status::Ok
    }

    pub fn wait_satisfied(&self, event_ref: u32, value: u64) -> bool {
        state_wait_satisfied(&self.table, self.task_id, Domain::Event, event_ref, value)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Ok,
    ErrArgs,
    ErrUnknownEvent,
    ErrTimeout,
    ErrOrder,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlannedOp {
    Signal {
        event_ref: u32,
        value: u64,
    },
    Wait {
        event_ref: u32,
        value: u64,
        timeout: Option<u32>,
    },
}

pub fn plan_event_simple(cmd: &EventCommand) -> Result<PlannedOp, Status> {
    match cmd.kind {
        DecodedEventKind::SignalEvent => Ok(PlannedOp::Signal {
            event_ref: cmd.event_ref,
            value: cmd.value,
        }),
        DecodedEventKind::WaitEvent => Ok(PlannedOp::Wait {
            event_ref: cmd.event_ref,
            value: cmd.value,
            timeout: if cmd.has_timeout {
                Some(cmd.timeout)
            } else {
                None
            },
        }),
        DecodedEventKind::Unknown => Err(Status::ErrArgs),
    }
}

// Keep old name as alias for plan_event_simple used by tests.
pub fn plan_event_cmd(cmd: &EventCommand) -> Result<PlannedOp, Status> {
    plan_event_simple(cmd)
}

pub fn apply_planned(state: &mut EventState, op: &PlannedOp) -> Status {
    match *op {
        PlannedOp::Signal { event_ref, value } => state.signal(event_ref, value),
        PlannedOp::Wait {
            event_ref, value, ..
        } => {
            if state.wait_satisfied(event_ref, value) {
                Status::Ok
            } else {
                Status::ErrOrder
            }
        }
    }
}

pub fn plan_from_bytes(bytes: &[u8]) -> Result<PlannedOp, Status> {
    let cmd = crate::runtime::decode::event::decode(bytes).map_err(|_| Status::ErrArgs)?;
    plan_event_simple(&cmd)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signal_first_and_advance() {
        let plan = plan_event_signal(1, 5, None);
        assert!(plan_updates_state(&plan));
        assert_eq!(plan.reason, Reason::SignalFirst);
        let plan2 = plan_event_signal(1, 7, Some(ValueSnapshot::current(5)));
        assert_eq!(plan2.reason, Reason::SignalAdvance);
        let plan3 = plan_event_signal(1, 5, Some(ValueSnapshot::current(5)));
        assert!(!plan_updates_state(&plan3));
        assert_eq!(plan3.reason, Reason::SignalEqualIgnored);
    }

    #[test]
    fn wait_pending_and_satisfied() {
        let p = plan_event_wait(1, 3, false, 0, None);
        assert!(!plan_allows_execution(&p));
        let p2 = plan_event_wait(1, 3, false, 0, Some(ValueSnapshot::current(3)));
        assert!(plan_allows_execution(&p2));
        let p3 = plan_event_wait(1, 3, true, 10, None);
        assert_eq!(p3.decision, Decision::WaitTimeoutUnsupported);
    }

    #[test]
    fn fence_generation() {
        let p = plan_fence_update(Domain::BlitFence, 9, 0, None);
        assert_eq!(p.update_value, 1);
        let p2 = plan_fence_update(Domain::BlitFence, 9, 0, Some(ValueSnapshot::current(1)));
        assert_eq!(p2.update_value, 2);
        let bad = plan_fence_update(Domain::Event, 1, 0, None);
        assert_eq!(bad.reason, Reason::BadFenceDomain);
    }

    #[test]
    fn state_table_roundtrip() {
        let mut t = StateTable::new();
        let plan = plan_event_signal(42, 100, None);
        state_apply_plan(&mut t, 7, &plan);
        let snap = state_snapshot(&t, 7, Domain::Event, 42);
        assert!(snap.valid);
        assert_eq!(snap.value, 100);
        assert!(state_wait_satisfied(&t, 7, Domain::Event, 42, 100));
        assert!(!state_wait_satisfied(&t, 7, Domain::Event, 42, 101));
    }
}
