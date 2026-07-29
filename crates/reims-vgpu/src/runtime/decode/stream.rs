//! Command-stream framing decoder (port of `host/utils/reims-vgpu-stream-decode`).

use crate::contract::endian::{ld32, ld64};
use crate::contract::size_fits_u32;

pub const SEGMENT_TYPE_RENDER: u8 = 0;
pub const SEGMENT_TYPE_COMPUTE: u8 = 1;
pub const SEGMENT_TYPE_BLIT: u8 = 2;
pub const SEGMENT_TYPE_EVENT: u8 = 3;
pub const SEGMENT_TYPE_INFO: u8 = 4;
pub const SEGMENT_TYPE_PROTECTION_OPTIONS: u8 = 5;
pub const SEGMENT_TYPE_UNKNOWN: u8 = 0xff;

pub const SEGMENT_LENGTH_OFFSET: usize = 0;
pub const SEGMENT_TYPE_OFFSET: usize = 4;
pub const SEGMENT_CONT_OFFSET: usize = 5;
pub const SEGMENT_CHAIN_OFFSET: usize = 6;
pub const SEGMENT_PAD_OFFSET: usize = 7;
pub const SEGMENT_HEADER_LEN: usize = 8;
pub const PROTECTION_OPTIONS_PAYLOAD_LEN: usize = 8;

pub const RECORD_OPCODE_OFFSET: usize = 0;
pub const RECORD_LENGTH_OFFSET: usize = 4;
pub const RECORD_HEADER_LEN: usize = 8;

pub const INFO_RECORD_OPCODE: u32 = 0x180;
pub const INFO_RECORD_LEN: u32 = 0x10;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeStatus {
    Ok,
    /// End of stream, or end of a segment's records. Control flow — the walkers
    /// terminate on it, so it is never a refusal and never reaches the log.
    Done,
    /// Refused; the payload is the registered slug naming which check refused.
    ///
    /// The payload is not decoration. This decoder frames *every* guest command,
    /// and a single coarse `ErrBadLength` covers seventeen checks here — a
    /// segment header disagreeing with the buffer, a record header disagreeing
    /// with its segment, and the re-validation of an already-parsed segment are
    /// three very different bugs that would otherwise arrive at the sink
    /// wearing one name.
    ErrArgs(&'static str),
    ErrShort(&'static str),
    ErrBadLength(&'static str),
}

impl crate::observe::Refusal for DecodeStatus {
    /// Slugs carry a `stream_` prefix: seven modules under `runtime/decode/`
    /// define a type called `DecodeStatus`, and five of them have an `ErrShort`
    /// meaning a different read. Without the prefix the crate-wide uniqueness
    /// gate could not tell this decoder's refusals from any other's.
    fn refusal(&self) -> Option<&'static str> {
        match self {
            Self::Ok | Self::Done => None,
            Self::ErrArgs(slug) | Self::ErrShort(slug) | Self::ErrBadLength(slug) => Some(slug),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Segment {
    pub offset: u32,
    pub length: u32,
    pub type_: u8,
    pub cont: u8,
    pub chain: u8,
    pub pad: u8,
    pub command_offset: u32,
    pub command_length: u32,
    pub index: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Record {
    pub segment_index: u32,
    pub segment_type: u32,
    pub offset: u32,
    pub length: u32,
    pub opcode: u32,
    /// Absolute offset of the record header in the stream bytes.
    pub bytes_offset: u32,
}

pub fn segment_type_name(type_: u32) -> &'static str {
    match type_ {
        0 => "render",
        1 => "compute",
        2 => "blit",
        3 => "event",
        4 => "info",
        5 => "protection-options",
        _ => "unknown",
    }
}

pub fn segment_type_is_command_family(type_: u32) -> bool {
    matches!(type_, 0..=3)
}

/// What the stream walker should do with a segment family.
///
/// This exists so the walker's "everything else" arm is a decision rather than a
/// fallthrough. It used to be `_ => {}`, which gave the same silence to a type-5
/// envelope the contract says to skip and to a segment family the host has never
/// seen — and the second of those is unknown wire format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SegmentDisposition {
    /// A family with a record walker: render, compute, blit, event, info.
    Walk,
    /// Type 5. `writeSegmentHeader:continuation:protectionOptions:` can emit a
    /// segment-level envelope *before* the real segment, and its command window
    /// is raw envelope bytes carrying no decodable protection value. Skipping it
    /// is contract-correct, so it is control flow and stays silent — logging it
    /// would put a line in the sink on every healthy frame that carries one.
    Envelope,
    /// A family this host has no contract for. MetalSerializer's deserializer
    /// constructs decoders for `0..3` and rejects new non-continuation types
    /// `>= 4`, so a type past the known set is not something to guess at:
    /// refuse it visibly instead of walking its bytes as records.
    Unknown,
}

impl crate::observe::Refusal for SegmentDisposition {
    fn refusal(&self) -> Option<&'static str> {
        match self {
            Self::Walk | Self::Envelope => None,
            Self::Unknown => Some("stream_segment_type_unknown"),
        }
    }
}

pub fn segment_disposition(type_: u8) -> SegmentDisposition {
    match type_ {
        SEGMENT_TYPE_RENDER | SEGMENT_TYPE_COMPUTE | SEGMENT_TYPE_BLIT | SEGMENT_TYPE_EVENT
        | SEGMENT_TYPE_INFO => SegmentDisposition::Walk,
        SEGMENT_TYPE_PROTECTION_OPTIONS => SegmentDisposition::Envelope,
        _ => SegmentDisposition::Unknown,
    }
}

pub fn trace_u32(value: u64) -> u32 {
    if value > u32::MAX as u64 {
        u32::MAX
    } else {
        value as u32
    }
}

fn validate_bytes(bytes: &[u8]) -> DecodeStatus {
    if !size_fits_u32(bytes.len()) {
        return DecodeStatus::ErrBadLength("stream_bytes_len_overflow");
    }
    DecodeStatus::Ok
}

fn segment_index_for_offset(bytes: &[u8], target_offset: u32) -> Result<u32, DecodeStatus> {
    let mut cursor = 0usize;
    let mut index = 0u32;
    while cursor < bytes.len() {
        if bytes.len() - cursor < SEGMENT_HEADER_LEN {
            return Err(DecodeStatus::ErrShort("stream_index_walk_short_header"));
        }
        if !size_fits_u32(cursor) {
            return Err(DecodeStatus::ErrBadLength(
                "stream_index_walk_cursor_overflow",
            ));
        }
        if cursor as u32 == target_offset {
            return Ok(index);
        }
        let segment_len = ld32(&bytes[cursor + SEGMENT_LENGTH_OFFSET..]) as usize;
        if segment_len < SEGMENT_HEADER_LEN || segment_len > bytes.len() - cursor {
            return Err(DecodeStatus::ErrBadLength("stream_index_walk_seg_len"));
        }
        cursor += segment_len;
        index += 1;
    }
    Err(DecodeStatus::ErrBadLength(
        "stream_index_target_offset_not_found",
    ))
}

/// Decode the next segment at `cursor`. On Ok advances cursor. Transactional: no partial out.
pub fn decode_next_segment(bytes: &[u8], cursor: &mut usize) -> Result<Segment, DecodeStatus> {
    let status = validate_bytes(bytes);
    if status != DecodeStatus::Ok {
        return Err(status);
    }
    if *cursor > bytes.len() {
        return Err(DecodeStatus::ErrArgs("stream_seg_cursor_past_end"));
    }
    if *cursor == bytes.len() {
        return Err(DecodeStatus::Done);
    }
    if bytes.len() - *cursor < SEGMENT_HEADER_LEN {
        return Err(DecodeStatus::ErrShort("stream_seg_short_header"));
    }
    let header = &bytes[*cursor..];
    let segment_len = ld32(&header[SEGMENT_LENGTH_OFFSET..]) as usize;
    if segment_len < SEGMENT_HEADER_LEN {
        return Err(DecodeStatus::ErrBadLength("stream_seg_len_below_header"));
    }
    if segment_len > bytes.len() - *cursor {
        return Err(DecodeStatus::ErrBadLength("stream_seg_len_past_buffer_end"));
    }
    if !size_fits_u32(*cursor) {
        return Err(DecodeStatus::ErrBadLength("stream_seg_cursor_overflow"));
    }
    let segment_index = segment_index_for_offset(bytes, *cursor as u32)?;
    let out = Segment {
        offset: *cursor as u32,
        length: segment_len as u32,
        type_: header[SEGMENT_TYPE_OFFSET],
        cont: header[SEGMENT_CONT_OFFSET],
        chain: header[SEGMENT_CHAIN_OFFSET],
        pad: header[SEGMENT_PAD_OFFSET],
        command_offset: (*cursor + SEGMENT_HEADER_LEN) as u32,
        command_length: (segment_len - SEGMENT_HEADER_LEN) as u32,
        index: segment_index,
    };
    *cursor += segment_len;
    Ok(out)
}

pub fn decode_first_segment(bytes: &[u8], cursor: &mut usize) -> Result<Segment, DecodeStatus> {
    *cursor = 0;
    decode_next_segment(bytes, cursor)
}

fn validate_segment(bytes: &[u8], segment: &Segment) -> Result<usize, DecodeStatus> {
    let status = validate_bytes(bytes);
    if status != DecodeStatus::Ok {
        return Err(status);
    }
    if (segment.length as usize) < SEGMENT_HEADER_LEN {
        return Err(DecodeStatus::ErrBadLength("stream_reval_len_below_header"));
    }
    if (segment.offset as usize) > bytes.len()
        || (segment.length as usize) > bytes.len() - segment.offset as usize
    {
        return Err(DecodeStatus::ErrBadLength("stream_reval_span_oob"));
    }
    let header = &bytes[segment.offset as usize..];
    if ld32(&header[SEGMENT_LENGTH_OFFSET..]) != segment.length
        || header[SEGMENT_TYPE_OFFSET] != segment.type_
        || header[SEGMENT_CONT_OFFSET] != segment.cont
        || header[SEGMENT_CHAIN_OFFSET] != segment.chain
        || header[SEGMENT_PAD_OFFSET] != segment.pad
    {
        return Err(DecodeStatus::ErrBadLength("stream_reval_header_mismatch"));
    }
    if segment.command_offset != segment.offset + SEGMENT_HEADER_LEN as u32
        || segment.command_length != segment.length - SEGMENT_HEADER_LEN as u32
    {
        return Err(DecodeStatus::ErrBadLength(
            "stream_reval_command_span_mismatch",
        ));
    }
    if segment.command_offset < segment.offset
        || segment.command_length > u32::MAX - segment.command_offset
    {
        return Err(DecodeStatus::ErrBadLength(
            "stream_reval_command_offset_overflow",
        ));
    }
    let command_end = segment.command_offset as usize + segment.command_length as usize;
    if (segment.command_offset as usize) > command_end
        || command_end > segment.offset as usize + segment.length as usize
        || command_end > bytes.len()
    {
        return Err(DecodeStatus::ErrBadLength("stream_reval_command_end_oob"));
    }
    Ok(command_end)
}

pub fn decode_next_record(
    bytes: &[u8],
    segment: &Segment,
    cursor: &mut usize,
) -> Result<Record, DecodeStatus> {
    let command_end = validate_segment(bytes, segment)?;
    if *cursor < segment.command_offset as usize || *cursor > command_end {
        return Err(DecodeStatus::ErrArgs("stream_rec_cursor_out_of_segment"));
    }
    if segment.type_ == SEGMENT_TYPE_PROTECTION_OPTIONS {
        if *cursor != segment.command_offset as usize && *cursor != command_end {
            return Err(DecodeStatus::ErrArgs(
                "stream_rec_protection_cursor_misaligned",
            ));
        }
        *cursor = command_end;
        return Err(DecodeStatus::Done);
    }
    if *cursor == command_end {
        return Err(DecodeStatus::Done);
    }
    if command_end - *cursor < RECORD_HEADER_LEN {
        return Err(DecodeStatus::ErrShort("stream_rec_short_header"));
    }
    let header = &bytes[*cursor..];
    let opcode = ld32(&header[RECORD_OPCODE_OFFSET..]);
    let record_len = ld32(&header[RECORD_LENGTH_OFFSET..]) as usize;
    if record_len < RECORD_HEADER_LEN {
        return Err(DecodeStatus::ErrBadLength("stream_rec_len_below_header"));
    }
    if record_len > command_end - *cursor {
        return Err(DecodeStatus::ErrBadLength(
            "stream_rec_len_past_segment_end",
        ));
    }
    if !size_fits_u32(*cursor) {
        return Err(DecodeStatus::ErrBadLength("stream_rec_cursor_overflow"));
    }
    let out = Record {
        segment_index: segment.index,
        segment_type: segment.type_ as u32,
        offset: *cursor as u32,
        length: record_len as u32,
        opcode,
        bytes_offset: *cursor as u32,
    };
    *cursor += record_len;
    Ok(out)
}

pub fn decode_first_record(
    bytes: &[u8],
    segment: &Segment,
    cursor: &mut usize,
) -> Result<Record, DecodeStatus> {
    *cursor = segment.command_offset as usize;
    decode_next_record(bytes, segment, cursor)
}

pub fn decode_protection_options(
    bytes: &[u8],
    segment: &Segment,
) -> Result<(bool, u64), DecodeStatus> {
    let _command_end = validate_segment(bytes, segment)?;
    if segment.type_ != SEGMENT_TYPE_PROTECTION_OPTIONS {
        return Err(DecodeStatus::ErrArgs(
            "stream_protection_wrong_segment_type",
        ));
    }
    if segment.command_length == 0 {
        return Ok((false, 0));
    }
    if segment.command_length as usize != PROTECTION_OPTIONS_PAYLOAD_LEN {
        return Err(DecodeStatus::ErrBadLength("stream_protection_payload_len"));
    }
    let value = ld64(&bytes[segment.command_offset as usize..]);
    Ok((true, value))
}

/// Iterate all segments.
pub fn iter_segments(bytes: &[u8]) -> Result<Vec<Segment>, DecodeStatus> {
    let mut cursor = 0usize;
    let mut out = Vec::new();
    loop {
        match decode_next_segment(bytes, &mut cursor) {
            Ok(s) => out.push(s),
            Err(DecodeStatus::Done) => return Ok(out),
            Err(e) => return Err(e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::endian::st32;

    fn push_segment(buf: &mut Vec<u8>, type_: u8, payload: &[u8]) {
        let len = (SEGMENT_HEADER_LEN + payload.len()) as u32;
        let mut hdr = [0u8; 8];
        st32(&mut hdr[0..4], len);
        hdr[4] = type_;
        buf.extend_from_slice(&hdr);
        buf.extend_from_slice(payload);
    }

    fn push_record(buf: &mut Vec<u8>, opcode: u32, payload: &[u8]) {
        let len = (RECORD_HEADER_LEN + payload.len()) as u32;
        let mut hdr = [0u8; 8];
        st32(&mut hdr[0..4], opcode);
        st32(&mut hdr[4..8], len);
        buf.extend_from_slice(&hdr);
        buf.extend_from_slice(payload);
    }

    #[test]
    fn empty_stream_done() {
        let mut c = 0;
        assert_eq!(
            decode_first_segment(&[], &mut c).unwrap_err(),
            DecodeStatus::Done
        );
        assert_eq!(c, 0);
    }

    #[test]
    fn single_blit_segment_with_record() {
        let mut payload = Vec::new();
        push_record(&mut payload, 0x12d, &[0u8; 0x18]); // buffer-to-buffer shape
        let mut stream = Vec::new();
        push_segment(&mut stream, SEGMENT_TYPE_BLIT, &payload);

        let segs = iter_segments(&stream).unwrap();
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].type_, SEGMENT_TYPE_BLIT);
        assert_eq!(segs[0].index, 0);
        assert!(segment_type_is_command_family(segs[0].type_ as u32));

        let mut rc = 0;
        let rec = decode_first_record(&stream, &segs[0], &mut rc).unwrap();
        assert_eq!(rec.opcode, 0x12d);
        assert_eq!(
            decode_next_record(&stream, &segs[0], &mut rc).unwrap_err(),
            DecodeStatus::Done
        );
    }

    #[test]
    fn short_and_bad_length_name_the_check_that_refused() {
        use crate::observe::Refusal;
        // Asserting the slug rather than the variant is the point: both of these
        // used to be one `ErrBadLength`/`ErrShort` shared with sixteen other
        // checks, so a passing test said nothing about *which* read disagreed.
        assert_eq!(
            decode_next_segment(&[1, 2, 3], &mut 0)
                .unwrap_err()
                .refusal(),
            Some("stream_seg_short_header")
        );
        let mut bad = [0u8; 8];
        st32(&mut bad[0..4], 4); // length < header
        assert_eq!(
            decode_next_segment(&bad, &mut 0).unwrap_err().refusal(),
            Some("stream_seg_len_below_header")
        );
        // A segment header that outruns the buffer is a different bug from one
        // that undershoots its own header, and now says so.
        let mut past = [0u8; 8];
        st32(&mut past[0..4], 64);
        assert_eq!(
            decode_next_segment(&past, &mut 0).unwrap_err().refusal(),
            Some("stream_seg_len_past_buffer_end")
        );
    }

    #[test]
    fn end_of_stream_and_end_of_segment_are_never_refusals() {
        use crate::observe::Refusal;
        // `Done` is how both walkers terminate. If it ever reported a reason the
        // sink would carry one line per segment per frame — the flood that the
        // speculative-return carve-out exists to prevent.
        assert_eq!(DecodeStatus::Done.refusal(), None);
        assert_eq!(DecodeStatus::Ok.refusal(), None);

        let mut stream = Vec::new();
        push_segment(&mut stream, SEGMENT_TYPE_RENDER, &[]);
        let segs = iter_segments(&stream).unwrap();
        let mut c = 0;
        assert_eq!(
            decode_first_record(&stream, &segs[0], &mut c)
                .unwrap_err()
                .refusal(),
            None
        );
        let mut sc = stream.len();
        assert_eq!(
            decode_next_segment(&stream, &mut sc).unwrap_err().refusal(),
            None
        );
    }

    #[test]
    fn every_refusal_in_this_decoder_carries_a_registered_slug() {
        use crate::observe::Refusal;
        // The registry row is checked crate-wide by `observe::gate`; what this
        // test pins is the local half — that no site returns a refusal whose
        // payload is empty or absent, which would render `reason=` bare.
        for status in [
            DecodeStatus::ErrArgs("stream_seg_cursor_past_end"),
            DecodeStatus::ErrShort("stream_seg_short_header"),
            DecodeStatus::ErrBadLength("stream_bytes_len_overflow"),
        ] {
            let slug = status.refusal().expect("a refusal names its check");
            assert!(
                slug.starts_with("stream_"),
                "{slug} lacks the module prefix"
            );
        }
    }

    #[test]
    fn protection_options() {
        let mut stream = Vec::new();
        let mut payload = [0u8; 8];
        crate::contract::endian::st64(&mut payload, 0x1122334455667788);
        push_segment(&mut stream, SEGMENT_TYPE_PROTECTION_OPTIONS, &payload);
        let segs = iter_segments(&stream).unwrap();
        let (has, val) = decode_protection_options(&stream, &segs[0]).unwrap();
        assert!(has);
        assert_eq!(val, 0x1122334455667788);
        // record walker DONE
        let mut c = 0;
        assert_eq!(
            decode_first_record(&stream, &segs[0], &mut c).unwrap_err(),
            DecodeStatus::Done
        );
    }

    #[test]
    fn multi_segment_indices() {
        let mut stream = Vec::new();
        push_segment(&mut stream, SEGMENT_TYPE_RENDER, &[]);
        push_segment(&mut stream, SEGMENT_TYPE_COMPUTE, &[]);
        let segs = iter_segments(&stream).unwrap();
        assert_eq!(segs[0].index, 0);
        assert_eq!(segs[1].index, 1);
        assert_eq!(segment_type_name(0), "render");
        assert_eq!(trace_u32(u64::MAX), u32::MAX);
    }

    #[test]
    fn property_fuzz_random_headers() {
        // Smoke: random-ish short buffers must not panic.
        for n in 0..32usize {
            let bytes = vec![0xAAu8; n];
            let mut c = 0;
            let _ = decode_first_segment(&bytes, &mut c);
        }
    }
}
