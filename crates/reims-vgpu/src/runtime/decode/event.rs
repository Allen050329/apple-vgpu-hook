//! Event/sync command decoder (port of `host/utils/reims-vgpu-event-decode`).

use crate::contract::endian::{ld32, ld64};

pub const U32_SIZE: usize = 4;
pub const U64_SIZE: usize = 8;
pub const OPCODE_OFFSET: usize = 0;
pub const LENGTH_OFFSET: usize = 4;
pub const HEADER_LEN: usize = 8;

pub const VALUE_REF: usize = 0;
pub const VALUE_VALUE: usize = 4;
pub const TIMEOUT: usize = 12;
pub const SIGNAL_WAIT_PAYLOAD_LEN: usize = VALUE_VALUE + U64_SIZE;
pub const WAIT_TIMEOUT_PAYLOAD_LEN: usize = TIMEOUT + U32_SIZE;
pub const SIGNAL_WAIT_LEN: usize = HEADER_LEN + SIGNAL_WAIT_PAYLOAD_LEN;
pub const WAIT_TIMEOUT_LEN: usize = HEADER_LEN + WAIT_TIMEOUT_PAYLOAD_LEN;

pub const OP_WAIT_EVENT: u32 = 0x190;
pub const OP_SIGNAL_EVENT: u32 = 0x191;
pub const OP_WAIT_EVENT_TIMEOUT: u32 = 0x192;

pub const REJECTED_BLIT_UPDATE_FENCE: u32 = 0x13c;
pub const REJECTED_BLIT_WAIT_FENCE: u32 = 0x13d;
pub const REJECTED_BEFORE_WINDOW: u32 = 0x18f;
pub const REJECTED_AFTER_WINDOW: u32 = 0x193;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeStatus {
    Ok = 0,
    ErrArgs,
    ErrShort,
    ErrBadLength,
    ErrUnknownOpcode,
    ErrRejectedOpcode,
}

impl crate::observe::Refusal for DecodeStatus {
    /// Slugs carry a `event_decode_` prefix: seven modules under
    /// `runtime/decode/` define a type called `DecodeStatus`, and five of them
    /// have an `ErrShort` that means a different read. Without the prefix the
    /// crate-wide uniqueness gate could not tell the event decoder's refusals
    /// from any other's.
    fn refusal(&self) -> Option<&'static str> {
        Some(match self {
            Self::Ok => return None,
            Self::ErrArgs => "event_decode_args",
            Self::ErrShort => "event_decode_short",
            Self::ErrBadLength => "event_decode_bad_length",
            Self::ErrUnknownOpcode => "event_decode_unknown_opcode",
            Self::ErrRejectedOpcode => "event_decode_rejected_opcode",
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Unknown = 0,
    SignalEvent,
    WaitEvent,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Command {
    pub opcode: u32,
    pub command_length: u32,
    pub kind: Kind,
    pub event_ref: u32,
    pub value: u64,
    pub has_timeout: bool,
    pub timeout: u32,
    pub raw_payload_offset: usize,
    pub raw_payload_length: usize,
}

pub fn opcode_accepted_by_deserializer(opcode: u32) -> bool {
    matches!(
        opcode,
        OP_WAIT_EVENT | OP_SIGNAL_EVENT | OP_WAIT_EVENT_TIMEOUT
    )
}

pub fn opcode_emitted_by_serializer(_opcode: u32) -> bool {
    false
}

pub fn opcode_rejected_by_deserializer(opcode: u32) -> bool {
    matches!(
        opcode,
        REJECTED_BLIT_UPDATE_FENCE
            | REJECTED_BLIT_WAIT_FENCE
            | REJECTED_BEFORE_WINDOW
            | REJECTED_AFTER_WINDOW
    )
}

/// Decode one event command. Transactional: returns Ok only with a full snapshot.
pub fn decode(command: &[u8]) -> Result<Command, DecodeStatus> {
    if command.len() < HEADER_LEN {
        return Err(DecodeStatus::ErrShort);
    }
    let opcode = ld32(&command[OPCODE_OFFSET..]);
    let command_length = ld32(&command[LENGTH_OFFSET..]) as usize;
    if command_length < HEADER_LEN || command_length > command.len() {
        return Err(DecodeStatus::ErrShort);
    }
    let payload = &command[HEADER_LEN..command_length];
    let mut decoded = Command {
        opcode,
        command_length: command_length as u32,
        kind: Kind::Unknown,
        event_ref: 0,
        value: 0,
        has_timeout: false,
        timeout: 0,
        raw_payload_offset: HEADER_LEN,
        raw_payload_length: command_length - HEADER_LEN,
    };

    match opcode {
        OP_WAIT_EVENT => {
            if command_length < SIGNAL_WAIT_LEN {
                return Err(DecodeStatus::ErrBadLength);
            }
            decoded.kind = Kind::WaitEvent;
            decoded.event_ref = ld32(&payload[VALUE_REF..]);
            decoded.value = ld64(&payload[VALUE_VALUE..]);
            Ok(decoded)
        }
        OP_SIGNAL_EVENT => {
            if command_length < SIGNAL_WAIT_LEN {
                return Err(DecodeStatus::ErrBadLength);
            }
            decoded.kind = Kind::SignalEvent;
            decoded.event_ref = ld32(&payload[VALUE_REF..]);
            decoded.value = ld64(&payload[VALUE_VALUE..]);
            Ok(decoded)
        }
        OP_WAIT_EVENT_TIMEOUT => {
            if command_length < WAIT_TIMEOUT_LEN {
                return Err(DecodeStatus::ErrBadLength);
            }
            decoded.kind = Kind::WaitEvent;
            decoded.event_ref = ld32(&payload[VALUE_REF..]);
            decoded.value = ld64(&payload[VALUE_VALUE..]);
            decoded.has_timeout = true;
            decoded.timeout = ld32(&payload[TIMEOUT..]);
            Ok(decoded)
        }
        _ => {
            if opcode_rejected_by_deserializer(opcode) {
                Err(DecodeStatus::ErrRejectedOpcode)
            } else {
                Err(DecodeStatus::ErrUnknownOpcode)
            }
        }
    }
}

#[cfg(test)]
mod tests {

    /// A malformed event command used to be dropped at the dispatch site with no
    /// log line at all — indistinguishable from a segment carrying no event
    /// work. Each check names itself now, `Ok` still produces nothing, and the
    /// prefix keeps them apart from the six sibling `DecodeStatus` enums.
    #[test]
    fn every_event_decode_failure_but_ok_names_its_own_check() {
        use crate::observe::Refusal;
        const ERRS: &[DecodeStatus] = &[
            DecodeStatus::ErrArgs,
            DecodeStatus::ErrShort,
            DecodeStatus::ErrBadLength,
            DecodeStatus::ErrUnknownOpcode,
            DecodeStatus::ErrRejectedOpcode,
        ];
        assert_eq!(DecodeStatus::Ok.refusal(), None, "Ok is not a refusal");
        let mut slugs: Vec<&str> = ERRS.iter().filter_map(|s| s.refusal()).collect();
        assert_eq!(slugs.len(), ERRS.len(), "every error variant refuses");
        assert!(slugs.iter().all(|s| s.starts_with("event_decode_")));
        slugs.sort_unstable();
        let n = slugs.len();
        slugs.dedup();
        assert_eq!(slugs.len(), n, "two event decode checks share a slug");
    }
    use super::*;
    use crate::contract::endian::{st32, st64};

    fn build(opcode: u32, payload: &[u8]) -> Vec<u8> {
        let len = (HEADER_LEN + payload.len()) as u32;
        let mut v = vec![0u8; HEADER_LEN + payload.len()];
        st32(&mut v[0..4], opcode);
        st32(&mut v[4..8], len);
        v[HEADER_LEN..].copy_from_slice(payload);
        v
    }

    #[test]
    fn signal_and_wait() {
        let mut payload = [0u8; SIGNAL_WAIT_PAYLOAD_LEN];
        st32(&mut payload[0..4], 7);
        st64(&mut payload[4..12], 0x100);
        let cmd = decode(&build(OP_SIGNAL_EVENT, &payload)).unwrap();
        assert_eq!(cmd.kind, Kind::SignalEvent);
        assert_eq!(cmd.event_ref, 7);
        assert_eq!(cmd.value, 0x100);

        let cmd = decode(&build(OP_WAIT_EVENT, &payload)).unwrap();
        assert_eq!(cmd.kind, Kind::WaitEvent);

        let mut p2 = [0u8; WAIT_TIMEOUT_PAYLOAD_LEN];
        p2[..SIGNAL_WAIT_PAYLOAD_LEN].copy_from_slice(&payload);
        st32(&mut p2[TIMEOUT..TIMEOUT + 4], 42);
        let cmd = decode(&build(OP_WAIT_EVENT_TIMEOUT, &p2)).unwrap();
        assert!(cmd.has_timeout);
        assert_eq!(cmd.timeout, 42);
    }

    #[test]
    fn rejected_and_unknown() {
        assert_eq!(
            decode(&build(REJECTED_BLIT_UPDATE_FENCE, &[])).unwrap_err(),
            DecodeStatus::ErrRejectedOpcode
        );
        assert_eq!(
            decode(&build(0x999, &[])).unwrap_err(),
            DecodeStatus::ErrUnknownOpcode
        );
        assert!(opcode_accepted_by_deserializer(OP_SIGNAL_EVENT));
        assert!(!opcode_emitted_by_serializer(OP_SIGNAL_EVENT));
    }

    #[test]
    fn short_header() {
        assert_eq!(decode(&[0; 4]).unwrap_err(), DecodeStatus::ErrShort);
    }

    #[test]
    fn property_fuzz_opcodes() {
        for op in 0u32..0x200 {
            let mut v = build(op, &[0u8; 32]);
            // Force length large enough
            let len = v.len() as u32;
            st32(&mut v[4..8], len);
            let _ = decode(&v);
        }
    }
}
