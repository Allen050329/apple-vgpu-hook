//! Emitting SPIR-V the engine owns, rather than translating the guest's.
//!
//! # Why this exists
//!
//! Every SPIR-V module in this device so far came from the guest: AIR through
//! `metal2vulkan`, then patched in place by [`crate::runtime::spirv_bind`] and
//! its private peer `runtime::spirv_layout` — named rather than linked, because
//! that module is not public and a link to it resolves nowhere. Both walk and
//! edit an existing word stream; neither can synthesise one. The engine has
//! never had a shader of its own.
//!
//! It needs one for the render writeback. That rail moves 2.26 GB/s into guest
//! RAM and 93-98% of the bytes are already at the destination, measured per
//! landing and spread across every landing rather than concentrated in a few
//! (see [`crate::runtime::land_redundancy`]). The CPU cannot collect on that —
//! comparing on the CPU was built and is refuted, because a full-cache-line
//! store never reads its destination and the compare adds a read the eager path
//! never paid. Only a GPU pass can decline bytes *before* they cross the bus,
//! and a GPU pass needs a shader the engine wrote.
//!
//! # Why words rather than a build step
//!
//! Two routes were available and one of them is closed. Compiling GLSL at build
//! time would put `glslc` on the critical path of every build of this crate,
//! and shipping the compiled result would commit SPIR-V, which the repository's
//! commit rule does not allow. Emitting the words from Rust is neither: it is
//! source, it builds with no toolchain beyond `cargo`, and the shader's meaning
//! is reviewable in the same language as its caller.
//!
//! The tests do use `spirv-val` when it is on `PATH`, and skip when it is not.
//! That is a check on this emitter, not a dependency of the product.
//!
//! # What this is not
//!
//! Not a general SPIR-V library. It emits what this device's own passes need
//! and nothing else: 32-bit unsigned integer arithmetic, storage buffers,
//! structured selection, and atomics. There is no type inference, no
//! control-flow graph, and no optimiser. Adding a feature here means adding the
//! instructions that feature needs, deliberately.
//!
//! # Which dialect, and why the old one
//!
//! Modules are emitted as **SPIR-V 1.0** with storage buffers spelled the 1.0
//! way — a struct decorated `BufferBlock` in the `Uniform` storage class,
//! rather than `Block` in `StorageBuffer`. The newer spelling needs SPIR-V 1.3
//! or `SPV_KHR_storage_buffer_storage_class`, and while this device's Vulkan
//! baseline is 1.2 and would carry it, the 1.0 spelling is accepted by
//! everything that accepts the new one. `Gate on capabilities, not API-version
//! assumptions` cuts toward the dialect that needs no capability at all.

/// SPIR-V magic number, first word of every module.
const MAGIC: u32 = 0x0723_0203;

/// Version 1.0, encoded as the module header expects: `0 | major << 16 | minor << 8`.
const VERSION_1_0: u32 = 0x0001_0000;

/// Generator magic. The registry reserves 0 for "unknown", which is the honest
/// answer for a generator with no registered id; tools print it and carry on.
const GENERATOR: u32 = 0;

// Opcodes, in numeric order. Only the ones emitted here.
const OP_NAME: u16 = 5;
const OP_MEMBER_NAME: u16 = 6;
const OP_ENTRY_POINT: u16 = 15;
const OP_EXECUTION_MODE: u16 = 16;
const OP_CAPABILITY: u16 = 17;
const OP_TYPE_VOID: u16 = 19;
const OP_TYPE_BOOL: u16 = 20;
const OP_TYPE_INT: u16 = 21;
const OP_TYPE_VECTOR: u16 = 23;
const OP_TYPE_RUNTIME_ARRAY: u16 = 29;
const OP_TYPE_STRUCT: u16 = 30;
const OP_TYPE_POINTER: u16 = 32;
const OP_TYPE_FUNCTION: u16 = 33;
const OP_CONSTANT: u16 = 43;
const OP_FUNCTION: u16 = 54;
const OP_FUNCTION_END: u16 = 56;
const OP_VARIABLE: u16 = 59;
const OP_LOAD: u16 = 61;
const OP_STORE: u16 = 62;
const OP_ACCESS_CHAIN: u16 = 65;
const OP_DECORATE: u16 = 71;
const OP_MEMBER_DECORATE: u16 = 72;
const OP_COMPOSITE_EXTRACT: u16 = 81;
const OP_U_DIV: u16 = 134;
const OP_SHIFT_RIGHT_LOGICAL: u16 = 194;
const OP_SHIFT_LEFT_LOGICAL: u16 = 196;
const OP_BITWISE_AND: u16 = 199;
const OP_I_NOT_EQUAL: u16 = 171;
const OP_U_LESS_THAN: u16 = 176;
const OP_ATOMIC_OR: u16 = 241;
const OP_LABEL: u16 = 248;
const OP_BRANCH: u16 = 249;
const OP_BRANCH_CONDITIONAL: u16 = 250;
const OP_RETURN: u16 = 253;
const OP_MEMORY_MODEL: u16 = 14;
const OP_SELECTION_MERGE: u16 = 247;

const CAPABILITY_SHADER: u32 = 1;
const ADDRESSING_LOGICAL: u32 = 0;
const MEMORY_MODEL_GLSL450: u32 = 1;
const EXEC_MODEL_GLCOMPUTE: u32 = 5;
const EXEC_MODE_LOCAL_SIZE: u32 = 17;

const STORAGE_UNIFORM: u32 = 2;
const STORAGE_INPUT: u32 = 1;

const DECORATION_BUFFER_BLOCK: u32 = 3;
const DECORATION_ARRAY_STRIDE: u32 = 6;
const DECORATION_BUILT_IN: u32 = 11;
const DECORATION_BINDING: u32 = 33;
const DECORATION_DESCRIPTOR_SET: u32 = 34;
const DECORATION_OFFSET: u32 = 35;

const BUILT_IN_GLOBAL_INVOCATION_ID: u32 = 28;

const FUNCTION_CONTROL_NONE: u32 = 0;
const SELECTION_CONTROL_NONE: u32 = 0;

/// `Device` scope: the atomic is visible to every invocation on the device.
///
/// Workgroup scope would not do. Tile bits are packed 32 to a word, so two
/// invocations in *different* workgroups can target the same word — the tile a
/// word belongs to is decided by its address, not by its dispatch.
const SCOPE_DEVICE: u32 = 1;

/// `None` — relaxed. The result is read only after the queue fence this
/// dispatch was submitted behind, which orders it against every reader.
const SEMANTICS_RELAXED: u32 = 0;

/// A SPIR-V id.
pub type Id = u32;

/// A module under construction, kept in the sections SPIR-V's logical layout
/// requires.
///
/// The order of these fields is the order of the module, and it is not
/// negotiable: capabilities, memory model, entry points, execution modes, debug
/// names, annotations, then types/constants/globals, then functions. Emitting
/// out of order produces a module that every validator rejects with a message
/// about layout rather than about the instruction that moved, so the sections
/// are separate vectors and [`Self::finish`] concatenates them.
#[derive(Debug, Default)]
pub struct Builder {
    next_id: Id,
    capabilities: Vec<u32>,
    memory_model: Vec<u32>,
    entry_points: Vec<u32>,
    execution_modes: Vec<u32>,
    debug: Vec<u32>,
    annotations: Vec<u32>,
    types: Vec<u32>,
    functions: Vec<u32>,
    /// `(type, value)` to the id already emitted for it.
    ///
    /// Scalar constants must be unique in a module — SPIR-V says the same type
    /// and value yield the same id — so this is required rather than tidiness.
    /// It also lets a call site name what a constant *means*
    /// ([`SCOPE_DEVICE`]) instead of reusing whichever earlier binding happened
    /// to hold the same number, which would couple two unrelated meanings
    /// through a numeric coincidence.
    constants: std::collections::BTreeMap<(Id, u32), Id>,
}

/// One instruction: `(word_count << 16) | opcode`, then its operands.
fn instruction(into: &mut Vec<u32>, op: u16, operands: &[u32]) {
    let count = u32::try_from(operands.len() + 1).expect("an instruction with 65535 operands");
    into.push((count << 16) | u32::from(op));
    into.extend_from_slice(operands);
}

/// A literal string as SPIR-V encodes it: UTF-8, NUL-terminated, packed
/// little-endian four bytes to a word and padded with NULs to a whole word.
fn literal_string(s: &str) -> Vec<u32> {
    let mut bytes = s.as_bytes().to_vec();
    bytes.push(0);
    while !bytes.len().is_multiple_of(4) {
        bytes.push(0);
    }
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

impl Builder {
    pub fn new() -> Self {
        Self {
            next_id: 1,
            ..Default::default()
        }
    }

    /// A fresh id. Ids are dense from 1, and the header's bound is the count
    /// plus one, so nothing here needs to track a maximum separately.
    pub fn id(&mut self) -> Id {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    pub fn capability(&mut self, cap: u32) {
        instruction(&mut self.capabilities, OP_CAPABILITY, &[cap]);
    }

    pub fn memory_model(&mut self, addressing: u32, model: u32) {
        instruction(&mut self.memory_model, OP_MEMORY_MODEL, &[addressing, model]);
    }

    /// `interface` is the entry point's global variables. SPIR-V 1.0 requires
    /// only the `Input` and `Output` ones, which for a compute shader is the
    /// built-in it reads and nothing else.
    pub fn entry_point(&mut self, model: u32, func: Id, name: &str, interface: &[Id]) {
        let mut ops = vec![model, func];
        ops.extend(literal_string(name));
        ops.extend_from_slice(interface);
        instruction(&mut self.entry_points, OP_ENTRY_POINT, &ops);
    }

    pub fn execution_mode(&mut self, func: Id, mode: u32, literals: &[u32]) {
        let mut ops = vec![func, mode];
        ops.extend_from_slice(literals);
        instruction(&mut self.execution_modes, OP_EXECUTION_MODE, &ops);
    }

    /// A debug name. Not load-bearing, and worth the words: without it a
    /// disassembly of this module is a wall of `%23`, and reviewing a shader
    /// nobody can read is how a wrong one ships.
    pub fn name(&mut self, target: Id, name: &str) {
        let mut ops = vec![target];
        ops.extend(literal_string(name));
        instruction(&mut self.debug, OP_NAME, &ops);
    }

    pub fn member_name(&mut self, target: Id, member: u32, name: &str) {
        let mut ops = vec![target, member];
        ops.extend(literal_string(name));
        instruction(&mut self.debug, OP_MEMBER_NAME, &ops);
    }

    pub fn decorate(&mut self, target: Id, decoration: u32, literals: &[u32]) {
        let mut ops = vec![target, decoration];
        ops.extend_from_slice(literals);
        instruction(&mut self.annotations, OP_DECORATE, &ops);
    }

    pub fn member_decorate(&mut self, target: Id, member: u32, decoration: u32, literals: &[u32]) {
        let mut ops = vec![target, member, decoration];
        ops.extend_from_slice(literals);
        instruction(&mut self.annotations, OP_MEMBER_DECORATE, &ops);
    }

    /// Emit into the types/constants/globals section, returning the result id.
    ///
    /// Types are not deduplicated here: a caller holds each one in a variable
    /// and reuses it, which is how the shader below reads. Dedup would be wrong
    /// for aggregates — two structurally identical struct types can need to be
    /// distinct, because their decorations differ — and [`Self::constant_u32`]
    /// does its own for scalars, where the spec requires it.
    fn typed(&mut self, op: u16, operands_before_result: &[u32], operands_after: &[u32]) -> Id {
        let id = self.id();
        let mut ops = operands_before_result.to_vec();
        ops.push(id);
        ops.extend_from_slice(operands_after);
        instruction(&mut self.types, op, &ops);
        id
    }

    pub fn type_void(&mut self) -> Id {
        self.typed(OP_TYPE_VOID, &[], &[])
    }

    pub fn type_bool(&mut self) -> Id {
        self.typed(OP_TYPE_BOOL, &[], &[])
    }

    /// `signed` is SPIR-V's signedness literal: 0 unsigned, 1 signed.
    pub fn type_int(&mut self, width: u32, signed: u32) -> Id {
        self.typed(OP_TYPE_INT, &[], &[width, signed])
    }

    pub fn type_vector(&mut self, component: Id, count: u32) -> Id {
        self.typed(OP_TYPE_VECTOR, &[], &[component, count])
    }

    pub fn type_runtime_array(&mut self, element: Id) -> Id {
        self.typed(OP_TYPE_RUNTIME_ARRAY, &[], &[element])
    }

    pub fn type_struct(&mut self, members: &[Id]) -> Id {
        self.typed(OP_TYPE_STRUCT, &[], members)
    }

    pub fn type_pointer(&mut self, storage_class: u32, pointee: Id) -> Id {
        self.typed(OP_TYPE_POINTER, &[], &[storage_class, pointee])
    }

    pub fn type_function(&mut self, ret: Id, params: &[Id]) -> Id {
        self.typed(OP_TYPE_FUNCTION, &[], &[ret].iter().chain(params).copied().collect::<Vec<_>>())
    }

    /// A 32-bit constant, emitted once per distinct `(type, value)`.
    pub fn constant_u32(&mut self, ty: Id, value: u32) -> Id {
        if let Some(&id) = self.constants.get(&(ty, value)) {
            return id;
        }
        let id = self.typed(OP_CONSTANT, &[ty], &[value]);
        self.constants.insert((ty, value), id);
        id
    }

    /// A module-scope variable. These live in the types section with everything
    /// else global, which is what SPIR-V's layout rule means by "global".
    pub fn global_variable(&mut self, ptr_ty: Id, storage_class: u32) -> Id {
        self.typed(OP_VARIABLE, &[ptr_ty], &[storage_class])
    }

    /// Begin a function body. Instructions emitted after this land inside it
    /// until [`Self::function_end`].
    pub fn function(&mut self, ret: Id, fn_ty: Id) -> Id {
        let id = self.id();
        instruction(
            &mut self.functions,
            OP_FUNCTION,
            &[ret, id, FUNCTION_CONTROL_NONE, fn_ty],
        );
        id
    }

    pub fn function_end(&mut self) {
        instruction(&mut self.functions, OP_FUNCTION_END, &[]);
    }

    pub fn label(&mut self, id: Id) {
        instruction(&mut self.functions, OP_LABEL, &[id]);
    }

    /// A result-producing instruction inside the current function.
    fn body(&mut self, op: u16, result_type: Id, operands: &[u32]) -> Id {
        let id = self.id();
        let mut ops = vec![result_type, id];
        ops.extend_from_slice(operands);
        instruction(&mut self.functions, op, &ops);
        id
    }

    pub fn load(&mut self, ty: Id, ptr: Id) -> Id {
        self.body(OP_LOAD, ty, &[ptr])
    }

    pub fn store(&mut self, ptr: Id, value: Id) {
        instruction(&mut self.functions, OP_STORE, &[ptr, value]);
    }

    pub fn access_chain(&mut self, ptr_ty: Id, base: Id, indices: &[Id]) -> Id {
        let mut ops = vec![base];
        ops.extend_from_slice(indices);
        self.body(OP_ACCESS_CHAIN, ptr_ty, &ops)
    }

    pub fn composite_extract(&mut self, ty: Id, composite: Id, index: u32) -> Id {
        self.body(OP_COMPOSITE_EXTRACT, ty, &[composite, index])
    }

    pub fn u_less_than(&mut self, bool_ty: Id, a: Id, b: Id) -> Id {
        self.body(OP_U_LESS_THAN, bool_ty, &[a, b])
    }

    pub fn i_not_equal(&mut self, bool_ty: Id, a: Id, b: Id) -> Id {
        self.body(OP_I_NOT_EQUAL, bool_ty, &[a, b])
    }

    pub fn u_div(&mut self, ty: Id, a: Id, b: Id) -> Id {
        self.body(OP_U_DIV, ty, &[a, b])
    }

    pub fn shift_right_logical(&mut self, ty: Id, a: Id, shift: Id) -> Id {
        self.body(OP_SHIFT_RIGHT_LOGICAL, ty, &[a, shift])
    }

    pub fn shift_left_logical(&mut self, ty: Id, a: Id, shift: Id) -> Id {
        self.body(OP_SHIFT_LEFT_LOGICAL, ty, &[a, shift])
    }

    pub fn bitwise_and(&mut self, ty: Id, a: Id, b: Id) -> Id {
        self.body(OP_BITWISE_AND, ty, &[a, b])
    }

    pub fn atomic_or(&mut self, ty: Id, ptr: Id, scope: Id, semantics: Id, value: Id) -> Id {
        self.body(OP_ATOMIC_OR, ty, &[ptr, scope, semantics, value])
    }

    /// Declare where a structured selection rejoins. Required before every
    /// [`Self::branch_conditional`]: a conditional branch with no merge is
    /// unstructured control flow, which the `Shader` capability forbids.
    pub fn selection_merge(&mut self, merge_label: Id) {
        instruction(
            &mut self.functions,
            OP_SELECTION_MERGE,
            &[merge_label, SELECTION_CONTROL_NONE],
        );
    }

    pub fn branch_conditional(&mut self, cond: Id, true_label: Id, false_label: Id) {
        instruction(
            &mut self.functions,
            OP_BRANCH_CONDITIONAL,
            &[cond, true_label, false_label],
        );
    }

    pub fn branch(&mut self, target: Id) {
        instruction(&mut self.functions, OP_BRANCH, &[target]);
    }

    pub fn ret(&mut self) {
        instruction(&mut self.functions, OP_RETURN, &[]);
    }

    /// The finished word stream, header first.
    pub fn finish(self) -> Vec<u32> {
        let mut words = vec![MAGIC, VERSION_1_0, GENERATOR, self.next_id, 0];
        for section in [
            self.capabilities,
            self.memory_model,
            self.entry_points,
            self.execution_modes,
            self.debug,
            self.annotations,
            self.types,
            self.functions,
        ] {
            words.extend(section);
        }
        words
    }
}

/// How many invocations one workgroup of [`tile_diff`] runs.
///
/// One invocation per 32-bit word, so a workgroup covers 256 bytes — one
/// [`crate::runtime::land_redundancy::FINE_TILE`]. That is not a coincidence
/// worth relying on and the shader does not: it derives the tile from the word
/// address, so the two can be changed independently.
pub const TILE_DIFF_LOCAL_SIZE: u32 = 64;

/// Descriptor set 0 bindings [`tile_diff`] expects, in order.
pub mod tile_diff_binding {
    /// This frame, as copied out of the render target. Read only.
    pub const CUR: u32 = 0;
    /// The frame the guest's pages already hold. Read only.
    pub const PREV: u32 = 1;
    /// Host-visible, and written only where `cur` and `prev` differ.
    pub const OUT: u32 = 2;
    /// One bit per tile, set when any word in that tile differed.
    pub const BITS: u32 = 3;
}

/// Emit the render writeback's difference pass.
///
/// ```text
/// uint w = gl_GlobalInvocationID.x;
/// if (w < words) {
///     uint c = cur[w];
///     if (c != prev[w]) {
///         out[w] = c;
///         uint tile = w / words_per_tile;
///         atomicOr(bits[tile >> 5], 1u << (tile & 31));
///     }
/// }
/// ```
///
/// # Why one invocation per word rather than per tile
///
/// A tile is 64 words. One invocation per tile would have each invocation walk
/// 64 consecutive words, so neighbouring invocations would read addresses 256
/// bytes apart and every load would be its own memory transaction. One
/// invocation per word has neighbouring invocations read neighbouring words,
/// which is the access pattern the hardware coalesces. The cost is that the
/// tile's bit is set by whichever of its words differed — possibly several,
/// possibly none — and `OpAtomicOr` is idempotent, so a duplicate is free and a
/// race is not possible. Roughly 8% of words differ, so the atomics are sparse.
///
/// # Why `words` is baked in rather than pushed
///
/// The bound is a literal `OpConstant`, so a module is emitted per distinct
/// frame size and cached by it. Push constants would need a pipeline layout
/// with a push range, which the engine's layout cache does not build, and a
/// uniform block would need a fourth buffer to fill every frame. Frame sizes on
/// a running guest are a handful, and the modules are small.
///
/// `out` is *not* written where the words match. The destination buffer
/// therefore holds a sparse frame — the bytes of whichever landing last wrote
/// each region — and only the tiles whose bit is set may be read from it. That
/// is the whole point: the words that are not written are the bytes that never
/// cross the bus.
pub fn tile_diff(words: u32, words_per_tile: u32) -> Vec<u32> {
    let mut b = Builder::new();
    b.capability(CAPABILITY_SHADER);
    b.memory_model(ADDRESSING_LOGICAL, MEMORY_MODEL_GLSL450);

    let void = b.type_void();
    let fn_void = b.type_function(void, &[]);
    let bool_ty = b.type_bool();
    let uint = b.type_int(32, 0);
    let v3uint = b.type_vector(uint, 3);

    let c_zero = b.constant_u32(uint, 0);
    let c_one = b.constant_u32(uint, 1);
    let c_words = b.constant_u32(uint, words);
    let c_wpt = b.constant_u32(uint, words_per_tile);
    let c_five = b.constant_u32(uint, 5);
    let c_31 = b.constant_u32(uint, 31);
    let c_scope = b.constant_u32(uint, SCOPE_DEVICE);
    let c_semantics = b.constant_u32(uint, SEMANTICS_RELAXED);

    let ptr_in_v3uint = b.type_pointer(STORAGE_INPUT, v3uint);
    let gid = b.global_variable(ptr_in_v3uint, STORAGE_INPUT);
    b.name(gid, "gl_GlobalInvocationID");
    b.decorate(gid, DECORATION_BUILT_IN, &[BUILT_IN_GLOBAL_INVOCATION_ID]);

    // One struct type shared by all four buffers: `struct { uint data[]; }`.
    // SPIR-V allows one aggregate type to back several variables, and the
    // decorations that differ between them — set, binding, writability — are on
    // the variables rather than on the type.
    let rta_uint = b.type_runtime_array(uint);
    b.decorate(rta_uint, DECORATION_ARRAY_STRIDE, &[4]);
    let buf_ty = b.type_struct(&[rta_uint]);
    b.name(buf_ty, "Words");
    b.member_name(buf_ty, 0, "data");
    b.member_decorate(buf_ty, 0, DECORATION_OFFSET, &[0]);
    b.decorate(buf_ty, DECORATION_BUFFER_BLOCK, &[]);
    let ptr_buf = b.type_pointer(STORAGE_UNIFORM, buf_ty);
    let ptr_uint = b.type_pointer(STORAGE_UNIFORM, uint);

    // No `NonWritable` on `cur` and `prev`, though the shader never writes
    // them. That decoration goes on a block's *member*, and all four variables
    // share one block type, so marking it would mark `out` and `bits` too and
    // the module would not validate. Giving the read-only pair their own
    // structurally identical struct type would buy a hint the driver does not
    // need — `cur` and `prev` have no `OpStore` for it to learn anything from.
    let buffer = |b: &mut Builder, binding: u32, label: &str| {
        let v = b.global_variable(ptr_buf, STORAGE_UNIFORM);
        b.name(v, label);
        b.decorate(v, DECORATION_DESCRIPTOR_SET, &[0]);
        b.decorate(v, DECORATION_BINDING, &[binding]);
        v
    };
    let cur = buffer(&mut b, tile_diff_binding::CUR, "cur");
    let prev = buffer(&mut b, tile_diff_binding::PREV, "prev");
    let out = buffer(&mut b, tile_diff_binding::OUT, "out");
    let bits = buffer(&mut b, tile_diff_binding::BITS, "bits");

    let main = b.function(void, fn_void);
    b.name(main, "main");
    b.entry_point(EXEC_MODEL_GLCOMPUTE, main, "main", &[gid]);
    b.execution_mode(
        main,
        EXEC_MODE_LOCAL_SIZE,
        &[TILE_DIFF_LOCAL_SIZE, 1, 1],
    );

    // Labels are allocated up front because a branch names its target before
    // the target's `OpLabel` is emitted.
    let entry = b.id();
    let in_range = b.id();
    let differs = b.id();
    let merge_inner = b.id();
    let merge_outer = b.id();

    b.label(entry);
    let gid_v = b.load(v3uint, gid);
    let w = b.composite_extract(uint, gid_v, 0);
    let lt = b.u_less_than(bool_ty, w, c_words);
    b.selection_merge(merge_outer);
    b.branch_conditional(lt, in_range, merge_outer);

    b.label(in_range);
    let p_cur = b.access_chain(ptr_uint, cur, &[c_zero, w]);
    let cur_w = b.load(uint, p_cur);
    let p_prev = b.access_chain(ptr_uint, prev, &[c_zero, w]);
    let prev_w = b.load(uint, p_prev);
    let ne = b.i_not_equal(bool_ty, cur_w, prev_w);
    b.selection_merge(merge_inner);
    b.branch_conditional(ne, differs, merge_inner);

    b.label(differs);
    let p_out = b.access_chain(ptr_uint, out, &[c_zero, w]);
    b.store(p_out, cur_w);
    let tile = b.u_div(uint, w, c_wpt);
    let bit_word = b.shift_right_logical(uint, tile, c_five);
    let bit_index = b.bitwise_and(uint, tile, c_31);
    let mask = b.shift_left_logical(uint, c_one, bit_index);
    let p_bit = b.access_chain(ptr_uint, bits, &[c_zero, bit_word]);
    let _ = b.atomic_or(uint, p_bit, c_scope, c_semantics, mask);
    b.branch(merge_inner);

    b.label(merge_inner);
    b.branch(merge_outer);

    b.label(merge_outer);
    b.ret();
    b.function_end();

    b.finish()
}

/// How many workgroups [`tile_diff`] needs for `words` words.
pub fn tile_diff_groups(words: u32) -> u32 {
    words.div_ceil(TILE_DIFF_LOCAL_SIZE)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Walk a module's instructions, returning `(opcode, operands)` for each.
    ///
    /// This is the emitter's own reader, and it is deliberately not the
    /// patchers': it must fail on a stream those would skip past, because a
    /// truncated final instruction is exactly the bug an emitter has.
    fn instructions(words: &[u32]) -> Vec<(u16, Vec<u32>)> {
        let mut out = Vec::new();
        let mut i = 5;
        while i < words.len() {
            let count = (words[i] >> 16) as usize;
            assert!(count >= 1, "a zero-length instruction at word {i}");
            assert!(i + count <= words.len(), "instruction at {i} runs past the module");
            out.push(((words[i] & 0xffff) as u16, words[i + 1..i + count].to_vec()));
            i += count;
        }
        out
    }

    #[test]
    fn the_header_says_what_it_is() {
        let m = tile_diff(1024, 64);
        assert_eq!(m[0], MAGIC);
        assert_eq!(m[1], VERSION_1_0);
        assert_eq!(m[4], 0, "schema");
        assert!(m[3] > 1, "bound {} declares no ids", m[3]);
    }

    /// The header's bound is one past the last id handed out. A bound that is
    /// too small is rejected by every consumer; one that is too large wastes a
    /// driver's id table silently, so this checks equality rather than a bound.
    ///
    /// Not asserted by scanning operands for a maximum: operands carry literals
    /// too — a baked word count, a packed string — and the largest of those is
    /// not an id. A test that took the maximum operand read the letters of
    /// `gl_GlobalInvocationID` as an id near two billion.
    #[test]
    fn the_bound_is_one_past_the_last_id() {
        let mut b = Builder::new();
        assert_eq!(b.id(), 1);
        assert_eq!(b.id(), 2);
        let last = b.id();
        assert_eq!(b.finish()[3], last + 1);
    }

    /// The instruction stream is well-formed to its own word counts and ends
    /// exactly at the end of the module. A count that overruns by one word is
    /// the emitter bug this catches and a disassembler reports as garbage.
    #[test]
    fn every_instruction_fits_and_the_stream_ends_exactly() {
        let m = tile_diff(2_073_600, 64);
        let ins = instructions(&m);
        assert!(!ins.is_empty());
        assert_eq!(ins.last().map(|(op, _)| *op), Some(OP_FUNCTION_END));
    }

    /// Sections come out in SPIR-V's logical layout order. Emitting a
    /// decoration after a type is the mistake this catches, and a validator
    /// reports it as a layout error rather than pointing at the instruction.
    #[test]
    fn the_sections_are_in_logical_layout_order() {
        let m = tile_diff(1024, 64);
        let order: Vec<u16> = instructions(&m).into_iter().map(|(op, _)| op).collect();
        let first = |op: u16| order.iter().position(|&o| o == op).unwrap_or(usize::MAX);
        assert!(first(OP_CAPABILITY) < first(OP_MEMORY_MODEL));
        assert!(first(OP_MEMORY_MODEL) < first(OP_ENTRY_POINT));
        assert!(first(OP_ENTRY_POINT) < first(OP_EXECUTION_MODE));
        assert!(first(OP_EXECUTION_MODE) < first(OP_NAME));
        assert!(first(OP_NAME) < first(OP_DECORATE));
        assert!(first(OP_DECORATE) < first(OP_TYPE_VOID));
        assert!(first(OP_TYPE_VOID) < first(OP_FUNCTION));
        assert!(first(OP_FUNCTION) < first(OP_LABEL));
    }

    /// The four bindings are declared at set 0, one each, and the shader reads
    /// its bound out of a constant rather than a push constant.
    #[test]
    fn the_bindings_are_set_zero_and_distinct() {
        let m = tile_diff(4096, 64);
        let mut bindings: Vec<u32> = instructions(&m)
            .iter()
            .filter(|(op, ops)| *op == OP_DECORATE && ops.get(1) == Some(&DECORATION_BINDING))
            .map(|(_, ops)| ops[2])
            .collect();
        bindings.sort_unstable();
        assert_eq!(bindings, vec![0, 1, 2, 3]);
        let sets: Vec<u32> = instructions(&m)
            .iter()
            .filter(|(op, ops)| *op == OP_DECORATE && ops.get(1) == Some(&DECORATION_DESCRIPTOR_SET))
            .map(|(_, ops)| ops[2])
            .collect();
        assert_eq!(sets, vec![0; 4]);
    }

    /// The word bound reaches the module as a constant, so a caller that emits
    /// for a different frame size gets a different module rather than the same
    /// one with the wrong bound.
    #[test]
    fn the_word_bound_is_baked_in() {
        let has = |words: u32, want: u32| {
            instructions(&tile_diff(words, 64))
                .iter()
                .any(|(op, ops)| *op == OP_CONSTANT && ops.get(2) == Some(&want))
        };
        assert!(has(2_073_600, 2_073_600));
        assert!(!has(2_073_600, 4096));
        assert!(has(4096, 4096));
    }

    /// A scalar constant is emitted once per distinct value, even when two call
    /// sites reach it under different names. SPIR-V requires that uniqueness,
    /// and it is what lets `SCOPE_DEVICE` be spelled as itself at the atomic
    /// rather than as whichever earlier binding happened to hold a 1.
    #[test]
    fn a_repeated_constant_is_emitted_once() {
        let m = tile_diff(4096, 64);
        let mut values: Vec<u32> = instructions(&m)
            .iter()
            .filter(|(op, _)| *op == OP_CONSTANT)
            .map(|(_, ops)| ops[2])
            .collect();
        let before = values.len();
        values.sort_unstable();
        values.dedup();
        assert_eq!(values.len(), before, "a duplicate OpConstant: {values:?}");
        // The atomic's scope is a 1 and so is the bit mask's shift base, and
        // the semantics are a 0 as is the struct member index — so a module
        // that does not dedup has two of each.
        assert!(values.contains(&SCOPE_DEVICE));
        assert!(values.contains(&SEMANTICS_RELAXED));
    }

    /// A string literal is NUL-terminated and NUL-padded to a whole word, which
    /// is the encoding every consumer assumes and nothing in the words says.
    #[test]
    fn a_literal_string_is_nul_terminated_and_padded() {
        assert_eq!(literal_string("main"), vec![0x6e_69_61_6d, 0]);
        assert_eq!(literal_string("abc"), vec![0x00_63_62_61]);
        assert_eq!(literal_string(""), vec![0]);
    }

    /// `spirv-val` agrees, when it is installed. It is a check on this emitter
    /// and never a build dependency, so its absence skips rather than fails —
    /// and the skip says so, because a silently-skipped validation reads
    /// exactly like a passing one.
    #[test]
    fn spirv_val_accepts_the_module() {
        use std::io::Write;
        let Ok(out) = std::process::Command::new("spirv-val").arg("--version").output() else {
            eprintln!("SKIP: spirv-val is not on PATH; the emitted module was not validated");
            return;
        };
        assert!(out.status.success(), "spirv-val --version failed");
        for (words, wpt) in [(2_073_600u32, 64u32), (4096, 64), (1, 64)] {
            let m = tile_diff(words, wpt);
            let path = std::env::temp_dir().join(format!("reims-tile-diff-{words}-{wpt}.spv"));
            let mut f = std::fs::File::create(&path).expect("write the module");
            for w in &m {
                f.write_all(&w.to_le_bytes()).expect("write a word");
            }
            drop(f);
            let out = std::process::Command::new("spirv-val")
                .arg(&path)
                .output()
                .expect("run spirv-val");
            let _ = std::fs::remove_file(&path);
            assert!(
                out.status.success(),
                "spirv-val rejected words={words} wpt={wpt}: {}",
                String::from_utf8_lossy(&out.stderr),
            );
        }
    }
}

