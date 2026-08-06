//! No guest record may kill this device.
//!
//! A panic is the largest thing this device can drop. `unwind_safe` turns one
//! into a status code and `observe::panic` now records it, so it is survivable
//! and visible — but the call it happened in is gone, along with whatever guest
//! work was behind it, and a device that dies on a record Apple's own driver
//! could emit is not a faithful GPU. Real hardware refuses a malformed command;
//! it does not stop answering.
//!
//! Every `pub fn` here that takes a `&[u8]` reads bytes the guest wrote into a
//! ring, a descriptor page or a shader blob. Nothing between the guest and them
//! re-validates: the ring is guest RAM, the guest CPU can rewrite it after the
//! doorbell and before the decode, and a length field is whatever is in memory
//! when it is read. So each of these is a parser over untrusted input, and "the
//! observed driver never emits that" is not a bound.
//!
//! # What this drives
//!
//! Each entry point is called under `catch_unwind` against a generated corpus.
//! The assertion is only that it **returns** — any `Ok`, any typed refusal, any
//! empty result is a pass. What fails is an unwind.
//!
//! The corpus is six shapes, because uniform random bytes alone are a weak
//! fuzzer against a length-prefixed format: they almost never produce a header
//! that survives the first guard, so the deeper arms never run.
//!
//! - `Uniform` — every byte random. Reaches the outermost guards.
//! - `Sparse` — mostly zero with a few random bytes. Zero is a valid opcode, a
//!   valid count and a valid offset in most of these records, so this is what
//!   walks *past* the header into the body decoders.
//! - `Saturated` — mostly `0xFF`. This is the shape that finds the arithmetic:
//!   a count of `0xFFFF_FFFF`, an offset of `usize::MAX`, a level span that
//!   overflows when multiplied.
//! - `WideFields` — random, with `u32`-aligned slots overwritten by the values
//!   a length or count field goes wrong at: `0`, `1`, `0x7FFF_FFFF`,
//!   `0x8000_0000`, `0xFFFF_FFFF`. Uniform random hits `0xFFFF_FFFF` in a given
//!   slot once in four billion; this hits it deliberately.
//! - `Framed` — a well-formed `(opcode, length)` header over a hostile body,
//!   which is what walks into the arms *below* the first guard.
//! - `Streamed` — a buffer that genuinely is one segment carrying one record,
//!   laid out from the wire types' own `offset_of!`s. The three stream decoders
//!   re-validate a segment against the bytes it came from, so nothing weaker
//!   reaches them at all.
//!
//! Lengths sweep every value from 0 to just past the largest fixed header in
//! the tree, then jump to sizes a real descriptor page reaches. Short lengths
//! are where the slicing lives; the exhaustive sweep is what makes
//! "`header_len - 1`" a case rather than a hope.
//!
//! # Reach is checked, not assumed
//!
//! A corpus every decoder refuses at its first `if` drives a million calls and
//! exercises one branch, and it stays green through any change to everything
//! below that branch — which is the entire surface this exists for. So each
//! target's *acceptances* are counted, and the set of decoders that accept
//! nothing is asserted against a written list.
//!
//! That check earned itself immediately, and then again when the surface was
//! widened. Four distinct reasons a parser was unreachable, none of which a
//! larger corpus would have fixed:
//!
//! - **A tag word plus an exact length.** Five resource decoders want, say,
//!   `TYPE7_OBJECT_ICB` and precisely `ICB_DESC_LEN` bytes. A guessed `u32`
//!   reaches that with probability ~0. Fixed by taking the tags and lengths
//!   from the crate's own `pub const`s — see `Target::frames`.
//! - **A re-validated header.** `validate_segment` re-reads the segment header
//!   out of the buffer and compares it field by field, so no invented `Segment`
//!   passes; a real stream had to be built (`Shape::Streamed`).
//! - **Interior words that must point inside the record.** `extract_air`'s
//!   wrapper header needs `bc_off + bc_size` to land inside the blob. Fixed by
//!   `plant_plausible_words`, which biases a third of the interior `u32` slots
//!   to in-record values.
//! - **A value from a small named set.** The `iosurface_pages` entries refuse
//!   any format `format_bytes_per_pixel` does not know, and any geometry too
//!   large for the descriptor's own pitch — `sample_window_from_device_desc`
//!   wants `bytes_per_row >= width * bytes_per_pixel`, which a width near 2^31
//!   makes unsatisfiable. Fixed by *asking* the crate which formats are real
//!   rather than listing them, and by drawing plausible surface edges while
//!   keeping a quarter of the draws wild — that quarter is the `u32 * u32`
//!   overflow case and dropping it would take it out of this harness.
//!
//! What it found once it could reach: `parse_color_attachments` panicked on a
//! two-byte buffer. It takes a reach `len` within `bytes` and checks every
//! bound against it while indexing `bytes`, and unlike its sibling
//! `parse_vertex_block` it never checked `len <= bytes.len()`.
//!
//! # Why the population is scanned rather than listed
//!
//! A hand-written list is a list that stops being complete the first time
//! someone adds a parser. [`every_public_parser_is_driven`] scans the guest-byte
//! trees for every `pub fn` **whose signature takes a `&[u8]`** and fails if one
//! is missing from the table below.
//!
//! The criterion is the signature, not the name, and that matters: a prefix rule
//! (`decode*`, `parse*`) would have reported `texture_type8_header`,
//! `extract_air` and `mapping_span_bound` not as unadjudicated but not at all —
//! the same hole `a_bound_in_a_cut_is_named_like_one` exists to hold shut for
//! bounds. It fails in both directions, and on an empty scan: a structural check
//! that stops matching reports green while looking at nothing.

use reims_vgpu::contract::iosurface_pages;
use reims_vgpu::runtime::decode::{blit, compute, event, render, resource, stream};
use reims_vgpu::runtime::{heap_query, icb, mtlb};
use std::collections::BTreeSet;
use std::panic::{catch_unwind, AssertUnwindSafe};

mod source_scan;

/// SplitMix64. Deterministic and seeded from a constant, because a fuzzer that
/// finds a panic on one run in ten is not a gate — the input that found it has
/// to still be there tomorrow. The failure message prints the bytes anyway, so
/// a hit is reproducible even if this changes.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_u32(&mut self) -> u32 {
        self.next_u64() as u32
    }

    /// True about one time in `n`. Reads at the call site as the bias it is.
    fn one_in(&mut self, n: u64) -> bool {
        self.next_u64().is_multiple_of(n)
    }
}

/// The corpus shapes. See the module doc for why uniform alone is not
/// enough.
#[derive(Clone, Copy, Debug)]
enum Shape {
    Uniform,
    Sparse,
    Saturated,
    WideFields,
    Framed,
    Streamed,
}

const SHAPES: [Shape; 6] = [
    Shape::Uniform,
    Shape::Sparse,
    Shape::Saturated,
    Shape::WideFields,
    Shape::Framed,
    Shape::Streamed,
];

/// The segment types the stream decoder dispatches on, from the wire crate's
/// own names rather than from an ordinal. `PROTECTION_OPTIONS` is in the list
/// because it is the one type with its own cursor rule, so leaving it out would
/// leave that rule undriven.
const SEGMENT_TYPES: [u8; 6] = [
    stream::SEGMENT_TYPE_RENDER,
    stream::SEGMENT_TYPE_COMPUTE,
    stream::SEGMENT_TYPE_BLIT,
    stream::SEGMENT_TYPE_INFO,
    stream::SEGMENT_TYPE_EVENT,
    stream::SEGMENT_TYPE_PROTECTION_OPTIONS,
];

/// The one frame `mtlb::extract_air` needs: the wrapper magic it scans for, as
/// the leading word. A named `const` rather than an inline literal only because
/// `Target::frames` is `&'static` and a temporary array cannot be.
const AIR_MAGIC_FRAME: [Frame; 1] = [(u32::from_le_bytes(mtlb::AIR_WRAP_MAGIC), None)];

/// The values a guest length, count or offset field goes wrong at.
const EDGE_WORDS: [u32; 6] = [0, 1, 2, 0x7FFF_FFFF, 0x8000_0000, 0xFFFF_FFFF];

/// One past the largest opcode any decoder in this tree claims — `render`'s
/// space reaches `0x120`, and the arms above that are the unknown-opcode
/// refusal. Drawn from rather than randomised over `u32` because an opcode
/// picked uniformly is an unknown opcode with probability ~1, and an unknown
/// opcode is refused at the first guard: every body decoder in the tree would
/// then be unreachable by this harness.
const OPCODE_SPACE: u32 = 0x200;

/// `tag` overrides the opcode the [`Shape::Framed`] header carries, for the
/// contract pass. `None` draws one from [`OPCODE_SPACE`].
fn generate(shape: Shape, len: usize, rng: &mut Rng, tag: Option<u32>) -> Vec<u8> {
    let mut bytes = vec![0u8; len];
    match shape {
        Shape::Uniform => {
            for b in bytes.iter_mut() {
                *b = rng.next_u64() as u8;
            }
        }
        Shape::Sparse => {
            // A handful of non-zero bytes over a zero field. Scaled to length so
            // a 4096-byte descriptor is not effectively all-zero.
            let hits = 1 + len / 16;
            for _ in 0..hits {
                if len == 0 {
                    break;
                }
                let at = (rng.next_u64() as usize) % len;
                bytes[at] = rng.next_u64() as u8;
            }
        }
        Shape::Saturated => {
            for b in bytes.iter_mut() {
                *b = 0xFF;
            }
            // A few random bytes so the record is not one single value, which
            // several of these reject on sight as an unknown opcode.
            let hits = 1 + len / 32;
            for _ in 0..hits {
                if len == 0 {
                    break;
                }
                let at = (rng.next_u64() as usize) % len;
                bytes[at] = rng.next_u64() as u8;
            }
        }
        Shape::WideFields => {
            for b in bytes.iter_mut() {
                *b = rng.next_u64() as u8;
            }
            let slots = len / 4;
            for slot in 0..slots {
                if !rng.one_in(3) {
                    continue;
                }
                let word = EDGE_WORDS[(rng.next_u64() as usize) % EDGE_WORDS.len()];
                bytes[slot * 4..slot * 4 + 4].copy_from_slice(&word.to_le_bytes());
            }
        }
        Shape::Framed => {
            for b in bytes.iter_mut() {
                *b = rng.next_u64() as u8;
            }
            // Every record family in this tree opens `(opcode: u32, len: u32)`,
            // and every decoder checks the pair before reading anything else.
            // The other four shapes therefore almost never get past the first
            // guard, so the body arms — where the counts, the offsets and the
            // per-element walks live — would be untested by a harness that only
            // randomised bytes. This one hands the decoder a header it accepts
            // and a body it did not expect.
            if len >= 8 {
                let opcode = tag.unwrap_or_else(|| rng.next_u32() % OPCODE_SPACE);
                bytes[0..4].copy_from_slice(&opcode.to_le_bytes());
                // Usually the true length, so the record is self-consistent;
                // sometimes a lie, which is the case the guard exists for. The
                // contract pass leans on the truthful arm — a decoder that
                // demands `declared == bytes.len()` is unreachable without it.
                let declared = match rng.next_u64() % 4 {
                    0 if tag.is_none() => EDGE_WORDS[(rng.next_u64() as usize) % EDGE_WORDS.len()],
                    1 if tag.is_none() => (len as u32).wrapping_add(rng.next_u32() % 8),
                    _ => len as u32,
                };
                bytes[4..8].copy_from_slice(&declared.to_le_bytes());
                plant_plausible_words(&mut bytes, 8, rng);
            }
        }
        Shape::Streamed => {
            for b in bytes.iter_mut() {
                *b = rng.next_u64() as u8;
            }
            // A buffer that really is one segment holding one record, built from
            // the offsets the wire types declare rather than from literals.
            //
            // `validate_segment` re-reads the segment header out of the buffer
            // and compares every field against the `Segment` it was handed, so
            // the record decoders are unreachable unless the buffer genuinely
            // carries the header its segment claims. Nothing softer than this
            // gets past — see `segment_for`.
            if len >= stream::SEGMENT_HEADER_LEN + OP_HEADER_LEN {
                let seg_len = len as u32;
                bytes[stream::SEGMENT_LENGTH_OFFSET..stream::SEGMENT_LENGTH_OFFSET + 4]
                    .copy_from_slice(&seg_len.to_le_bytes());
                bytes[stream::SEGMENT_TYPE_OFFSET] =
                    SEGMENT_TYPES[(rng.next_u64() as usize) % SEGMENT_TYPES.len()];
                bytes[stream::SEGMENT_BEGIN_FLAG_OFFSET] = (rng.next_u64() % 2) as u8;
                bytes[stream::SEGMENT_UNIDENTIFIED_OFFSET] = 0;

                // One record at the segment's command offset. Its declared
                // length is usually the space actually there and sometimes not,
                // which is the guard's own case.
                let at = stream::SEGMENT_HEADER_LEN;
                let room = (len - at) as u32;
                let opcode = tag.unwrap_or_else(|| rng.next_u32() % OPCODE_SPACE);
                bytes[at + stream::RECORD_OPCODE_OFFSET..at + stream::RECORD_OPCODE_OFFSET + 4]
                    .copy_from_slice(&opcode.to_le_bytes());
                let record_len = match rng.next_u64() % 4 {
                    0 => EDGE_WORDS[(rng.next_u64() as usize) % EDGE_WORDS.len()],
                    1 => room.wrapping_add(rng.next_u32() % 8),
                    _ => room,
                };
                bytes[at + stream::RECORD_LENGTH_OFFSET..at + stream::RECORD_LENGTH_OFFSET + 4]
                    .copy_from_slice(&record_len.to_le_bytes());
            }
        }
    }
    bytes
}

/// The record header this tree frames every command with, from the wire type
/// rather than as a literal 8.
const OP_HEADER_LEN: usize = stream::RECORD_LENGTH_OFFSET + 4;

/// Overwrite about a third of the `u32` slots from `from` with a value that
/// could plausibly be an offset, count or size *within this record*.
///
/// A record's header gets a decoder past its first guard; its interior words
/// decide whether anything below that runs. Those words are offsets and sizes,
/// and a uniformly-drawn `u32` is out of range for every one of them, so a body
/// decoder reads one, refuses, and the walk it guards never happens. Two
/// concrete cases this was written for: `extract_air`'s wrapper header, whose
/// `bc_off + bc_size` must land inside the blob, and the tag walks in
/// `resource`, whose field lengths must fit the entry.
///
/// A third, not all: an in-range word next to a wild one is the mix that finds
/// a bound applied to one and not the other.
fn plant_plausible_words(bytes: &mut [u8], from: usize, rng: &mut Rng) {
    let len = bytes.len();
    let mut at = from;
    while at + 4 <= len {
        if rng.one_in(3) {
            let word = (rng.next_u64() as u32) % (len as u32 + 1);
            bytes[at..at + 4].copy_from_slice(&word.to_le_bytes());
        }
        at += 4;
    }
}

/// Every length the corpus is generated at.
///
/// Exhaustive to 80 — past every fixed header in this tree, so
/// `header_len - 1`, `header_len` and `header_len + 1` are all cases for all of
/// them rather than for the ones someone thought of. Then the sizes a real
/// descriptor page and a real command reach, because a body decoder that walks
/// a count only misbehaves once there are bytes for it to walk.
fn lengths() -> Vec<usize> {
    let mut v: Vec<usize> = (0..=80).collect();
    v.extend([96, 128, 192, 256, 384, 512, 1024, 2048, 4096]);
    v
}

/// Independent corpora per (shape, length), so one target's bad luck with the
/// stream does not shift every other target's inputs.
const SEEDS_PER_CELL: u64 = 64;

/// Whether a decoder *accepted* an input, across the two result shapes this
/// surface uses. Reach, not correctness: a harness whose corpus is refused at
/// the first guard by every decoder drives a million calls and tests one `if`.
trait Accepted {
    fn accepted(&self) -> bool;
}

impl<T, E> Accepted for Result<T, E> {
    fn accepted(&self) -> bool {
        self.is_ok()
    }
}

impl<T> Accepted for Option<T> {
    fn accepted(&self) -> bool {
        self.is_some()
    }
}

/// One `(leading word, exact record length)` pair a decoder's contract admits.
///
/// `None` for the length means the contract fixes no exact size — the record is
/// self-describing and any length its declared field agrees with is admissible.
type Frame = (u32, Option<usize>);

/// One decoder, adapted to the table's uniform shape. Returns whether it
/// accepted the input, for the reach check.
type Run = Box<dyn Fn(&[u8], &mut Rng) -> bool>;

struct Target {
    /// `module::function`, matching what [`every_public_decoder_is_driven`]
    /// derives from the source. Module-qualified because four different files
    /// export a function called `decode`.
    name: &'static str,
    /// The tags and sizes this decoder's own constants declare.
    ///
    /// Several decoders here dispatch on a leading tag word and then demand an
    /// exact record length: `decode_icb_descriptor` wants
    /// `TYPE7_OBJECT_ICB` and precisely `ICB_DESC_LEN` bytes, and refuses
    /// everything else at its first `if`. A corpus that guesses a `u32` reaches
    /// such a decoder with probability ~0 and its body never runs — which is
    /// how eight of these read as fuzzed while only their first guard was.
    ///
    /// So the framing comes from the crate's own `pub const`s rather than from
    /// the generator: it is the contract, not a magic number, and a tag that
    /// changes value changes here by recompiling rather than by silently
    /// dropping a decoder back out of reach.
    frames: &'static [Frame],
    /// Returns whether the decoder accepted this input, for the reach check.
    run: Run,
}

fn target(name: &'static str, run: impl Fn(&[u8], &mut Rng) -> bool + 'static) -> Target {
    Target {
        name,
        frames: &[],
        run: Box::new(run),
    }
}

/// A target whose contract names the tags and lengths it admits.
fn framed(
    name: &'static str,
    frames: &'static [Frame],
    run: impl Fn(&[u8], &mut Rng) -> bool + 'static,
) -> Target {
    Target {
        name,
        frames,
        run: Box::new(run),
    }
}

/// Every `pub fn decode*` in the scanned surface, plus the extra arguments the
/// few that take them get. Those arguments are fuzzed too where the guest
/// controls them: an object type, a cursor and a TLV offset all arrive from the
/// same untrusted record as the bytes.
fn targets() -> Vec<Target> {
    vec![
        target("event::decode", |b, _| event::decode(b).accepted()),
        target("blit::decode", |b, _| blit::decode(b).accepted()),
        target("compute::decode", |b, _| compute::decode(b).accepted()),
        target("render::decode", |b, _| render::decode(b).accepted()),
        target("render::decode_color_attachment", |b, r| {
            // The index is a colour-attachment slot the caller derives from a
            // decoded count, so it is guest-reachable and gets fuzzed with it.
            // No refusal path: this returns a struct, so every input is accepted
            // by construction and its reach reading is trivially 1.
            let _ = render::decode_color_attachment(b, (r.next_u32() % 16) as usize);
            true
        }),
        target("render::decode_depth_attachment", |b, _| {
            // No refusal path: this returns a struct, so every input is accepted
            // by construction and its reach reading is trivially 1.
            let _ = render::decode_depth_attachment(b);
            true
        }),
        target("render::decode_stencil_attachment", |b, _| {
            // No refusal path: this returns a struct, so every input is accepted
            // by construction and its reach reading is trivially 1.
            let _ = render::decode_stencil_attachment(b);
            true
        }),
        target("fifo::decode_invalidate_resources", |b, _| {
            reims_vgpu::runtime::decode::fifo::decode_invalidate_resources(b).accepted()
        }),
        target("fifo::decode_exec_resource_table", |b, _| {
            reims_vgpu::runtime::decode::fifo::decode_exec_resource_table(b).accepted()
        }),
        target("fifo::decode_replace_physical", |b, _| {
            reims_vgpu::runtime::decode::fifo::decode_replace_physical(b).accepted()
        }),
        target("fifo::decode_synchronize_resources", |b, _| {
            reims_vgpu::runtime::decode::fifo::decode_synchronize_resources(b).accepted()
        }),
        target("stream::decode_next_segment", |b, r| {
            let mut cursor = seeded_cursor(r, b.len());
            stream::decode_next_segment(b, &mut cursor).accepted()
        }),
        target("stream::decode_next_record", |b, r| {
            let (seg, mut cursor) = segment_for(b, r);
            stream::decode_next_record(b, &seg, &mut cursor).accepted()
        }),
        target("stream::decode_first_record", |b, r| {
            // `decode_first_record` sets the cursor itself from the segment, so
            // the one passed in only has to be somewhere.
            let (seg, _) = segment_for(b, r);
            let mut cursor = seeded_cursor(r, b.len());
            stream::decode_first_record(b, &seg, &mut cursor).accepted()
        }),
        target("heap_query::decode_request", |b, _| {
            heap_query::decode_request(b).accepted()
        }),
        target(
            "heap_query::decode_serialized_texture_descriptor",
            |b, _| heap_query::decode_serialized_texture_descriptor(b).accepted(),
        ),
        target(
            "heap_query::decode_wide_serialized_texture_descriptor",
            |b, _| heap_query::decode_wide_serialized_texture_descriptor(b).accepted(),
        ),
        target("resource::decode_list_object_entry", |b, _| {
            resource::decode_list_object_entry(b).accepted()
        }),
        target("resource::decode_buffer_descriptor", |b, _| {
            resource::decode_buffer_descriptor(b).accepted()
        }),
        target("resource::decode_texture_descriptor", |b, _| {
            resource::decode_texture_descriptor(b).accepted()
        }),
        target("resource::decode_function_descriptor", |b, _| {
            resource::decode_function_descriptor(b).accepted()
        }),
        target("resource::decode_compact_tlv_record", |b, r| {
            // The offset walks a record the guest laid out, so a hostile one is
            // past the end.
            resource::decode_compact_tlv_record(b, (r.next_u32() % 128) as usize).accepted()
        }),
        target("resource::decode_depth_stencil_descriptor", |b, _| {
            resource::decode_depth_stencil_descriptor(b).accepted()
        }),
        target("resource::decode_sampler_descriptor", |b, _| {
            resource::decode_sampler_descriptor(b).accepted()
        }),
        framed(
            "resource::decode_render_pipeline_descriptor",
            &[(resource::TYPE7_OBJECT_RENDER_PIPELINE, None)],
            |b, _| resource::decode_render_pipeline_descriptor(b).accepted(),
        ),
        framed(
            "resource::decode_heap_texture",
            &[
                (
                    resource::HEAP_TEXTURE_OPCODE,
                    Some(resource::HEAP_TEXTURE_LEN),
                ),
                (resource::HEAP_TEXTURE_WIDE_OPCODE, None),
            ],
            |b, _| resource::decode_heap_texture(b).accepted(),
        ),
        target("resource::decode_texture_view_descriptor", |b, _| {
            resource::decode_texture_view_descriptor(b).accepted()
        }),
        framed(
            "resource::decode_buffer_texture_descriptor",
            &[
                (
                    resource::TEXTURE_VIEW_OPCODE_BUFFER_TEXTURE,
                    Some(resource::BUF_TEX_MIN_LEN),
                ),
                (resource::TEXTURE_VIEW_OPCODE_BUFFER_TEXTURE_WIDE, None),
            ],
            |b, _| resource::decode_buffer_texture_descriptor(b).accepted(),
        ),
        target("resource::decode_iosurface_texture_descriptor", |b, _| {
            resource::decode_iosurface_texture_descriptor(b).accepted()
        }),
        framed(
            "resource::decode_compute_pipeline_descriptor",
            &[(resource::TYPE7_OBJECT_COMPUTE_PIPELINE, None)],
            |b, _| resource::decode_compute_pipeline_descriptor(b).accepted(),
        ),
        target("resource::decode_icb_command_layout", |b, _| {
            resource::decode_icb_command_layout(b).accepted()
        }),
        framed(
            "resource::decode_icb_descriptor",
            &[(resource::TYPE7_OBJECT_ICB, Some(resource::ICB_DESC_LEN))],
            |b, _| resource::decode_icb_descriptor(b).accepted(),
        ),
        target("resource::decode_type7_descriptor", |b, _| {
            resource::decode_type7_descriptor(b).accepted()
        }),
        target("resource::decode_descriptor", |b, r| {
            // The object type is the first byte of the guest's own object-list
            // entry, so every one of the 256 is reachable.
            resource::decode_descriptor(r.next_u32() as u8, b).accepted()
        }),
        // --- runtime/decode/stream.rs -------------------------------------
        target("stream::iter_segments", |b, _| {
            // Walks the whole buffer segment by segment. Fuzzed as much for
            // termination and allocation as for panics: a segment length the
            // walk does not advance past would hang here rather than fail.
            stream::iter_segments(b).accepted()
        }),
        // --- runtime/decode/resource: the body parsers --------------------
        target("resource::parse_vertex_block", |b, r| {
            // Both bounds come off a decoded descriptor, so a hostile record
            // can put the end before the start or either past the buffer.
            let start = r.next_u64() as usize % (b.len() + 1);
            let end = r.next_u64() as usize % (b.len() + 8);
            resource::parse_vertex_block(b, start, end).accepted()
        }),
        target("resource::parse_color_attachments", |b, r| {
            let len = r.next_u64() as usize % (b.len() + 8);
            let off = r.next_u64() as usize % (b.len() + 1);
            resource::parse_color_attachments(b, len, off).accepted()
        }),
        target("resource::parse_compute_stage_input_block", |b, r| {
            let start = r.next_u64() as usize % (b.len() + 1);
            resource::parse_compute_stage_input_block(b, start).accepted()
        }),
        target("resource::texture_type8_header", |b, _| {
            resource::texture_type8_header(b).accepted()
        }),
        target("resource::texture_type8_opcode", |b, _| {
            resource::texture_type8_opcode(b).accepted()
        }),
        // --- runtime/icb ---------------------------------------------------
        target("icb::decode_compute_command_slot", |b, r| {
            // The layout is itself decoded from guest bytes, so it is taken
            // from the same buffer rather than invented — the product reaches
            // this decoder no other way.
            let Ok(layout) = resource::decode_icb_command_layout(b) else {
                return false;
            };
            icb::decode_compute_command_slot(&layout, b, r.next_u32() as u16).accepted()
        }),
        target("icb::decode_render_command_slot", |b, r| {
            let Ok(layout) = resource::decode_icb_command_layout(b) else {
                return false;
            };
            icb::decode_render_command_slot(&layout, b, r.next_u32() as u16, r.next_u32() as u16)
                .accepted()
        }),
        target("icb::decode_icb_host_resource_info", |b, _| {
            icb::decode_icb_host_resource_info(b).accepted()
        }),
        // --- runtime/mtlb --------------------------------------------------
        framed(
            "mtlb::extract_air",
            // The scan is for a 4-byte magic, so uniform bytes never get past
            // it and the wrapper-header arithmetic below — a `bc_off + bc_size`
            // taken from the blob — would go undriven. The frame plants the
            // crate's own constant rather than the four bytes again.
            &AIR_MAGIC_FRAME,
            |b, _| mtlb::extract_air(b).accepted(),
        ),
        // --- contract/iosurface_pages --------------------------------------
        target("iosurface_pages::decode_device_surface", |b, _| {
            iosurface_pages::decode_device_surface(b).accepted()
        }),
        target("iosurface_pages::decode_device_plane", |b, _| {
            iosurface_pages::decode_device_plane(b).accepted()
        }),
        target("iosurface_pages::device_desc_plane", |b, r| {
            // The plane index comes off the guest's own descriptor.
            iosurface_pages::device_desc_plane(b, r.next_u32() % 8).accepted()
        }),
        target("iosurface_pages::decode_texture_descriptor", |b, _| {
            iosurface_pages::decode_texture_descriptor(b).accepted()
        }),
        target("iosurface_pages::decode_mapper_request_entry", |b, _| {
            iosurface_pages::decode_mapper_request_entry(b).accepted()
        }),
        target("iosurface_pages::sample_window_from_device_desc", |b, r| {
            // Format and geometry all arrive decoded from the guest, and this
            // multiplies them into a byte span — the product-widening class.
            iosurface_pages::sample_window_from_device_desc(
                Some(b),
                if r.one_in(4) {
                    None
                } else {
                    Some(r.next_u32() % 8)
                },
                a_pixel_format(r),
                a_dimension(r),
                a_dimension(r),
            )
            .accepted()
        }),
        target("iosurface_pages::mapping_span_bound", |b, r| {
            iosurface_pages::mapping_span_bound(
                Some(b),
                a_pixel_format(r),
                a_dimension(r),
                a_dimension(r),
            )
            .accepted()
        }),
    ]
}

/// One past the largest edge any surface this device presents has: 16K, which
/// is above every display mode and every texture bound the tree carries.
///
/// A dimension is not an opcode and cannot be enumerated, but it is not a
/// uniform `u32` either. `sample_window_from_device_desc` returns a window only
/// when `bytes_per_row >= width * bytes_per_pixel`, and `bytes_per_row` comes
/// out of a descriptor whose own words are bounded by the record — so a width
/// of ~2^31 makes that inequality unsatisfiable and the whole span calculation
/// unreachable.
const MAX_PLAUSIBLE_EDGE: u32 = 16384;

/// A surface edge: usually plausible, one draw in four wild.
///
/// The wild draw is not a leftover. A `u32` width multiplied by a `u32` height
/// is the overflow shape `a_product_is_widened_before_it_is_taken` exists for,
/// and dropping it here would take that case out of this harness entirely.
fn a_dimension(rng: &mut Rng) -> u32 {
    if rng.one_in(4) {
        rng.next_u32()
    } else {
        rng.next_u32() % MAX_PLAUSIBLE_EDGE
    }
}

/// A pixel format, usually one the device actually knows.
///
/// Both `iosurface_pages` entries below refuse immediately on a format
/// `format_bytes_per_pixel` does not recognise, and a `u16` drawn uniformly is
/// unrecognised almost always — so the span arithmetic they exist for would
/// never run. The recognised set is *asked for* rather than listed: the crate
/// already exposes the predicate, so the harness cannot drift from it and no
/// format ordinal is written out here.
///
/// One draw in eight stays wild, because refusing an unknown format is itself a
/// path worth driving.
fn a_pixel_format(rng: &mut Rng) -> u16 {
    if rng.one_in(8) {
        return rng.next_u32() as u16;
    }
    let known = known_pixel_formats();
    known[(rng.next_u64() as usize) % known.len()]
}

/// Every `u16` the device admits as a pixel format, computed once.
fn known_pixel_formats() -> &'static [u16] {
    static KNOWN: std::sync::OnceLock<Vec<u16>> = std::sync::OnceLock::new();
    KNOWN.get_or_init(|| {
        let all: Vec<u16> = (0..=u16::MAX)
            .filter(|f| iosurface_pages::format_bytes_per_pixel(*f).is_some())
            .collect();
        assert!(
            !all.is_empty(),
            "no pixel format is recognised at all; the predicate moved and this draw is now blind"
        );
        all
    })
}

/// A segment and a matching cursor for the two record decoders that take one.
///
/// Half of these come from `decode_next_segment` over the very same buffer, and
/// that is not a convenience — it is the only way in. `validate_segment`
/// **re-reads the segment header out of `bytes` and compares every field**
/// against the `Segment` it was handed, then requires
/// `command_offset == offset + header_len` and
/// `command_length == length - header_len`. A hand-built `Segment` fails that
/// comparison however carefully its fields are chosen, so a generator that
/// invents one drives both record decoders into `stream_reval_header_mismatch`
/// and never reads a record byte. Deriving it the way the product does is what
/// gets past.
///
/// The other half is wild, because a `Segment` really can go stale: it is
/// decoded from guest RAM the guest CPU may rewrite before the record decode,
/// which is precisely why that re-validation exists. Both halves are the
/// contract.
fn segment_for(bytes: &[u8], rng: &mut Rng) -> (stream::Segment, usize) {
    if rng.one_in(2) {
        let mut cursor = 0usize;
        if let Ok(seg) = stream::decode_next_segment(bytes, &mut cursor) {
            let at = if rng.one_in(4) {
                // Sometimes off the record boundary, which is its own guard.
                seeded_cursor(rng, bytes.len())
            } else {
                seg.command_offset as usize
            };
            return (seg, at);
        }
    }
    wild_segment(rng)
}

/// Every field independent — what a guest that rewrote the ring after the
/// doorbell leaves behind.
fn wild_segment(rng: &mut Rng) -> (stream::Segment, usize) {
    (
        stream::Segment {
            offset: rng.next_u32(),
            length: rng.next_u32(),
            type_: rng.next_u32() as u8,
            begin_flag: rng.next_u32() as u8,
            unidentified_u8: rng.next_u32() as u8,
            unwritten_u8: rng.next_u32() as u8,
            command_offset: rng.next_u32(),
            command_length: rng.next_u32(),
            index: rng.next_u32(),
        },
        rng.next_u64() as usize,
    )
}

/// A cursor that is usually inside the buffer and sometimes not, for the same
/// reason `fuzzed_segment` biases its spans: a decoder whose every input starts
/// past the end tests its bounds check and nothing else.
fn seeded_cursor(rng: &mut Rng, buf_len: usize) -> usize {
    if buf_len == 0 || rng.one_in(4) {
        (rng.next_u32() % 64) as usize
    } else {
        rng.next_u64() as usize % buf_len
    }
}

#[test]
fn no_decoder_panics_on_bytes_the_guest_could_write() {
    // The default hook would print a full report per panic, and a broken
    // decoder panics on thousands of inputs. The failure message below carries
    // the entry, the shape and the bytes, which is what reproduces it.
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));

    let lengths = lengths();
    let mut failures: Vec<String> = Vec::new();
    let mut unreached: Vec<&'static str> = Vec::new();
    let mut calls = 0usize;

    for t in targets() {
        let mut accepted = 0usize;
        // (shape, length, framing tag). The blind sweep first, then one pass per
        // frame the decoder's own contract declares — see `Target::frames`.
        let mut cells: Vec<(Shape, usize, Option<u32>)> = Vec::new();
        for shape in SHAPES {
            for &len in &lengths {
                cells.push((shape, len, None));
            }
        }
        for &(tag, exact) in t.frames {
            match exact {
                Some(len) => cells.push((Shape::Framed, len, Some(tag))),
                None => {
                    for &len in &lengths {
                        cells.push((Shape::Framed, len, Some(tag)));
                    }
                }
            }
        }

        'target: for (shape, len, tag) in cells {
            for seed in 0..SEEDS_PER_CELL {
                // Seeded per cell so a corpus is reproducible from its
                // coordinates alone.
                let mut gen = Rng::new(
                    (t.name.len() as u64).wrapping_mul(0x1000_0001)
                        ^ ((shape as u64) << 40)
                        ^ ((len as u64) << 8)
                        ^ (u64::from(tag.unwrap_or(0)) << 20)
                        ^ seed,
                );
                let bytes = generate(shape, len, &mut gen, tag);
                let mut args = Rng::new(gen.next_u64());
                calls += 1;
                let run = &t.run;
                match catch_unwind(AssertUnwindSafe(|| run(&bytes, &mut args))) {
                    Ok(true) => accepted += 1,
                    Ok(false) => {}
                    Err(_) => {
                        failures.push(format!(
                            "{} panicked on {shape:?} len={len} tag={tag:?} seed={seed}: {}",
                            t.name,
                            hex(&bytes)
                        ));
                        // One report per target. The rest are the same bug.
                        break 'target;
                    }
                }
            }
        }
        if accepted == 0 {
            unreached.push(t.name);
        }
    }

    std::panic::set_hook(previous);

    assert!(
        failures.is_empty(),
        "a guest record must never kill this device:\n  {}",
        failures.join("\n  ")
    );
    assert!(
        calls > 0,
        "the harness drove nothing; the corpus or the target table is empty"
    );
    // Reach, not correctness. A corpus every decoder refuses at its first guard
    // drives a million calls and exercises one `if` — it would stay green
    // through any change to the arms below that guard, which is the whole
    // surface this test exists for. `NO_REACH` is the written list of decoders
    // that legitimately accept nothing generated; anything else appearing here
    // means the corpus stopped getting in, and anything leaving it means a
    // decoder started accepting and the note should say why.
    assert_eq!(
        unreached, NO_REACH,
        "the set of decoders this corpus never gets an acceptance from has changed"
    );
}

/// Parsers no generated input is accepted by, and why that is the contract
/// rather than a gap in the corpus.
///
/// `heap_query::decode_request` reads a 24-byte task/reply header whose first
/// word must be one of a handful of query opcodes *and* whose declared length
/// must agree with a per-opcode table. That is a joint constraint on two fields
/// the corpus draws independently, so it cannot be reached by drawing harder —
/// it would have to guess a consistent pair. Its body decoders,
/// `decode_serialized_texture_descriptor` and the wide form, are driven directly
/// and do reach, and that is where the arithmetic is.
///
/// It is still fuzzed for panics on every input; what it does not have is an
/// *acceptance*, which is what this list records.
///
/// `iosurface_pages::sample_window_from_device_desc` was briefly written down
/// here as a second joint constraint. It is not one — it reaches as soon as the
/// format and the geometry are drawn from plausible sets, and the entry was
/// added while the code that draws them was still dead. A verdict about a
/// generator is worth what the generator was actually doing when it was
/// written; this list is short on purpose.
const NO_REACH: [&str; 1] = ["heap_query::decode_request"];

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

/// Directories walked whole for guest-byte parsers, so a new module under one
/// of them is covered the day it lands.
const PARSER_TREES: [&str; 2] = [
    "crates/reims-vgpu/src/runtime/decode",
    "crates/reims-vgpu/src/runtime/icb",
];

/// Guest-byte parsers that sit outside those trees, each named with what makes
/// it one.
///
/// - `heap_query.rs` parses a serializer record straight out of a guest
///   descriptor page.
/// - `mtlb.rs` walks an MTLB container the guest wrote, searching it for a
///   magic, which is a parse over attacker-chosen bytes by any definition.
/// - `contract/iosurface_pages.rs` reads the IOSurface device descriptor the
///   guest lays out, and derives page spans from it.
///
/// `runtime/spirv_layout.rs` belongs to this class and is **not** here: it is a
/// private `mod`, so an integration test cannot name it. Its parser is reached
/// only through `spirv_bind`, which exposes no `&[u8]` entry of its own, and
/// its own tests live in-module. Making it `pub` to fuzz it would widen the
/// crate's surface to suit a test.
const PARSER_FILES: [&str; 3] = [
    "crates/reims-vgpu/src/runtime/heap_query.rs",
    "crates/reims-vgpu/src/runtime/mtlb.rs",
    "crates/reims-vgpu/src/contract/iosurface_pages.rs",
];

/// `module::function` for every `pub fn` on the scanned surface that takes a
/// byte slice.
///
/// The criterion is the **signature**, not the name. A name prefix — `decode*`,
/// `parse*` — is the same trap the bound scans hit: a parser called
/// `texture_type8_header` or `extract_air` reads guest bytes exactly as hard as
/// one called `decode_something`, and a prefix rule reports it not as
/// unadjudicated but not at all. Taking `&[u8]` is what makes a function a
/// parser here, so that is what is matched.
fn declared_parsers() -> BTreeSet<String> {
    let root = source_scan::workspace_root();
    let mut files: Vec<std::path::PathBuf> = PARSER_TREES
        .iter()
        .flat_map(|d| source_scan::rust_sources(&root.join(d)))
        .collect();
    files.extend(PARSER_FILES.iter().map(|f| root.join(f)));

    let mut found = BTreeSet::new();
    for path in files {
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or_default();
        // A `tests.rs` sibling declares fixture encoders, not device parsers.
        if name == "tests" || name.ends_with("_tests") {
            continue;
        }
        // `mod.rs` is the module its directory is named for.
        let module = if name == "mod" {
            path.parent()
                .and_then(|p| p.file_name())
                .and_then(|s| s.to_str())
                .unwrap_or_default()
                .to_string()
        } else {
            name.to_string()
        };
        let Ok(raw) = std::fs::read_to_string(&path) else {
            continue;
        };
        let text = source_scan::blank_comments(&raw);
        for (leaf, args) in public_fn_signatures(&text) {
            if args.contains("&[u8]") {
                found.insert(format!("{module}::{leaf}"));
            }
        }
    }
    found
}

/// `(name, argument list)` for every item-level `pub fn` in `text`.
///
/// Item-level: the declaration must start the line, so a `pub fn` inside an
/// `impl` block indented under it is not mistaken for a free function the test
/// could call by path.
fn public_fn_signatures(text: &str) -> Vec<(String, String)> {
    let bytes: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut at = 0usize;
    while let Some(rel) = text[at..].find("\npub fn ") {
        let start = at + rel + "\npub fn ".len();
        at = start;
        let leaf: String = text[start..]
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect();
        if leaf.is_empty() {
            continue;
        }
        // Skip any generic parameter list, then take the parenthesised args.
        let Some(open) = text[start..].find('(') else {
            continue;
        };
        let open = start + open;
        let mut depth = 0usize;
        let mut close = open;
        for (i, c) in bytes.iter().enumerate().skip(open) {
            match c {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        close = i;
                        break;
                    }
                }
                _ => {}
            }
        }
        if close > open {
            out.push((leaf, text[open..close].to_string()));
        }
    }
    out
}

#[test]
fn every_public_parser_is_driven() {
    let declared = declared_parsers();
    let driven: BTreeSet<String> = targets().into_iter().map(|t| t.name.to_string()).collect();

    assert!(
        !declared.is_empty(),
        "the scan found no `pub fn` taking `&[u8]` at all; it is reading the wrong shape or the \
         wrong tree"
    );

    let missing: Vec<&String> = declared.difference(&driven).collect();
    assert!(
        missing.is_empty(),
        "these read guest bytes and nothing proves they survive them — add each to `targets()` \
         or state in this test why it is exempt: {missing:?}"
    );

    // The other direction is just as load-bearing: a target naming a function
    // that no longer exists means the table is describing a parser the tree
    // does not have, and the missing check above would still pass.
    let stale: Vec<&String> = driven.difference(&declared).collect();
    assert!(
        stale.is_empty(),
        "these targets name no `pub fn` taking `&[u8]` on the scanned surface; the table has \
         drifted from the tree: {stale:?}"
    );
}
