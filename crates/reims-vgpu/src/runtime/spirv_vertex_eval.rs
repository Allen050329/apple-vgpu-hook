//! Bounded structural evaluation of vertex-shader clip positions.
//!
//! Some compositor vertex shaders carry no stage-in position attribute: they
//! read vertex records from storage buffers via `VertexIndex` and transform
//! them in-shader. The linear-compositor coverage proof needs the resulting
//! `Position` values, and the only contract-faithful way to obtain them is to
//! evaluate the translated SPIR-V itself against the decoded bound buffer
//! bytes — the exact inputs the GPU would read. This module is that evaluator.
//!
//! Rules (same class as the stage-in coverage proof):
//! - Structural only: numeric SPIR-V opcodes, storage classes, and explicit
//!   layout decorations. No debug names, no corpus identifiers.
//! - Fail closed: any opcode, type, or memory access outside the supported
//!   surface returns a named error slug. Callers must treat failure as
//!   "coverage unproven", never as a default position.
//! - Bounded: a fixed instruction budget per vertex rejects unbounded loops.
//! - Read only: buffer stores fail closed; nothing escapes the evaluation.

use std::collections::HashMap;

use crate::observe::Decline;

/// A specific fail-closed boundary in the structural vertex evaluator.
///
/// This evaluator is deliberately incomplete: unsupported IR, types, control
/// flow, and memory access reject the full-target coverage proof. Each check is
/// therefore a typed decline rather than a free-text evaluator status.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VertexEvalDecline {
    MalformedHeader,
    ModuleInstructionMalformed,
    ConstantTypeMissing,
    ConstantTypeUnsupported,
    CompositeConstantForwardReference,
    GlobalVariableTypeMissing,
    GlobalVariableTypeNotPointer,
    VertexEntryPointMissing,
    EntryFunctionBodyMissingDuringParse,
    TypeMissing {
        type_id: u32,
    },
    ScalarSizeTypeUnsupported,
    NullTypeUnsupported,
    ArrayVariableUnsupported,
    ValueIdUnset {
        id: u32,
    },
    IntOperandTypeMismatch,
    BufferBindingMissing {
        binding: u32,
    },
    FloatBufferReadWidthMismatch,
    BufferLoadTypeUnsupported,
    BufferOffsetDoesNotFitUsize {
        offset: u64,
    },
    BufferSizeDoesNotFitUsize {
        size: u64,
    },
    BufferRangeOverflow {
        offset: u64,
        size: u64,
    },
    BufferReadOutOfBounds {
        binding: u32,
        offset: u64,
        size: u64,
        len: usize,
    },
    InputVariableUnsupported,
    MemoryLoadVariableUnset {
        variable: u32,
    },
    MemoryLoadPathIntoScalar,
    MemoryLoadPathOutOfRange {
        index: u32,
    },
    BufferStoreUnsupported,
    MemoryStoreVariableUnset {
        variable: u32,
    },
    MemoryStorePathIntoScalar,
    MemoryStorePathOutOfRange {
        index: u32,
    },
    AccessChainBaseNotPointer,
    AccessChainBaseUnknown {
        base_id: u32,
    },
    AccessChainStructIndexOutOfRange {
        type_id: u32,
        index: u64,
    },
    AccessChainStructMemberOffsetMissing {
        type_id: u32,
        index: u32,
    },
    AccessChainStructOffsetOverflow,
    AccessChainArrayStrideMissing {
        type_id: u32,
    },
    AccessChainArrayOffsetOverflow,
    AccessChainVectorIndexOutOfRange {
        index: u64,
        count: u32,
    },
    AccessChainVectorOffsetOverflow,
    AccessChainTypeUnsupported {
        type_id: u32,
    },
    BufferVariableBindingMissing {
        variable: u32,
    },
    StorageClassUnsupported {
        storage: u32,
    },
    UnaryFloatOperandTypeMismatch,
    BinaryFloatOperandShapeMismatch,
    TernaryFloatOperandShapeMismatch,
    IntegerOperandShapeMismatch,
    FloatVectorMemberTypeMismatch,
    FloatVectorExpected,
    IntVectorElementTypeMismatch,
    IntResultTypeMismatch,
    PositionOutputMissing,
    EntryFunctionBodyMissingDuringRun,
    FunctionFellOffEnd,
    MainInstructionBudgetExhausted,
    FunctionInstructionMalformed,
    FunctionVariableTypeNotPointer,
    FunctionVariableStorageClassInvalid {
        storage: u32,
    },
    LoadSourceNotPointer,
    LoadPointerUnknown {
        id: u32,
    },
    StoreTargetNotPointer,
    StorePointerUnknown {
        id: u32,
    },
    SignedConvertSourceWidthUnknown {
        id: u32,
    },
    SignedToFloatSourceWidthUnknown {
        id: u32,
    },
    VectorScalarTypeMismatch,
    MatrixScalarTypeMismatch,
    MatrixTimesVectorMatrixNotComposite,
    MatrixTimesVectorColumnCountMismatch,
    MatrixTimesVectorColumnHeightMismatch,
    MatrixTimesVectorEmptyMatrix,
    VectorTimesMatrixMatrixNotComposite,
    VectorTimesMatrixShapeMismatch,
    MatrixTimesMatrixLeftNotComposite,
    MatrixTimesMatrixRightNotComposite,
    MatrixTimesMatrixShapeMismatch,
    MatrixTimesMatrixEmptyMatrix,
    TransposeMatrixNotComposite,
    TransposeMatrixRagged,
    DotShapeMismatch,
    CompositeExtractFromScalar,
    CompositeExtractIndexOutOfRange {
        index: u32,
    },
    CompositeInsertIntoScalar,
    CompositeInsertIndexOutOfRange {
        index: u32,
    },
    VectorShuffleLeftNotVector,
    VectorShuffleRightNotVector,
    VectorShuffleIndexOutOfRange {
        index: u32,
    },
    SelectValuesNotComposite,
    SelectVectorLengthMismatch,
    SelectVectorConditionNotBool,
    SelectConditionNotBool,
    SignedCompareWidthUnknown,
    LogicalNotVectorMemberNotBool,
    LogicalNotOperandNotBool,
    ExtInstSetUnknown {
        set_id: u32,
    },
    PhiOutsideBlockEntry,
    BranchConditionNotBool,
    SwitchSelectorWidthUnknown {
        id: u32,
    },
    SwitchOperandsMalformed,
    UnexpectedTerminator {
        opcode: u16,
    },
    OpcodeUnsupported {
        opcode: u16,
    },
    BranchLabelUnknown {
        label: u32,
    },
    PhiInstructionMalformed,
    PhiPredecessorMissing {
        predecessor: u32,
    },
    PhiInstructionBudgetExhausted,
    PositionVariableNeverStored {
        variable: u32,
    },
    PositionStructNeverStored {
        variable: u32,
    },
    PositionMemberNeverStored {
        variable: u32,
        member: u32,
    },
    PositionValueNotComposite,
    PositionVectorLengthInvalid {
        len: usize,
    },
    PositionComponentNotFinite {
        component: usize,
    },
    PositionComponentUndefined {
        component: usize,
    },
    MapIntOperandTypeMismatch,
    MapIntToFloatOperandTypeMismatch,
    MapFloatToIntOperandTypeMismatch,
    UnsignedDivisionByZero,
    SignedDivisionByZero,
    UnsignedModuloByZero,
    SignedRemainderByZero,
    IntegerBinaryOpcodeUnsupported {
        opcode: u16,
    },
    IntegerCompareOpcodeUnsupported {
        opcode: u16,
    },
    IntegerCompareShapeMismatch,
    FloatCompareOpcodeUnsupported {
        opcode: u16,
    },
    FloatCompareShapeMismatch,
    BooleanBinaryOpcodeUnsupported {
        opcode: u16,
    },
    BooleanBinaryShapeMismatch,
    BitcastTypeUnsupported,
    BitcastElementTypeUnsupported,
    ExtArgumentMissing {
        index: usize,
    },
    UnpackUnormOperandNotInt,
    ExtendedOpcodeUnsupported {
        opcode: u32,
    },
}

impl Decline for VertexEvalDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::MalformedHeader => "spirv_vertex_eval_malformed_header",
            Self::ModuleInstructionMalformed => "spirv_vertex_eval_module_instruction_malformed",
            Self::ConstantTypeMissing => "spirv_vertex_eval_constant_type_missing",
            Self::ConstantTypeUnsupported => "spirv_vertex_eval_constant_type_unsupported",
            Self::CompositeConstantForwardReference => {
                "spirv_vertex_eval_composite_constant_forward_reference"
            }
            Self::GlobalVariableTypeMissing => "spirv_vertex_eval_global_variable_type_missing",
            Self::GlobalVariableTypeNotPointer => {
                "spirv_vertex_eval_global_variable_type_not_pointer"
            }
            Self::VertexEntryPointMissing => "spirv_vertex_eval_vertex_entry_point_missing",
            Self::EntryFunctionBodyMissingDuringParse => {
                "spirv_vertex_eval_entry_function_body_missing_during_parse"
            }
            Self::TypeMissing { .. } => "spirv_vertex_eval_type_missing",
            Self::ScalarSizeTypeUnsupported => "spirv_vertex_eval_scalar_size_type_unsupported",
            Self::NullTypeUnsupported => "spirv_vertex_eval_null_type_unsupported",
            Self::ArrayVariableUnsupported => "spirv_vertex_eval_array_variable_unsupported",
            Self::ValueIdUnset { .. } => "spirv_vertex_eval_value_id_unset",
            Self::IntOperandTypeMismatch => "spirv_vertex_eval_int_operand_type_mismatch",
            Self::BufferBindingMissing { .. } => "spirv_vertex_eval_buffer_binding_missing",
            Self::FloatBufferReadWidthMismatch => {
                "spirv_vertex_eval_float_buffer_read_width_mismatch"
            }
            Self::BufferLoadTypeUnsupported => "spirv_vertex_eval_buffer_load_type_unsupported",
            Self::BufferOffsetDoesNotFitUsize { .. } => {
                "spirv_vertex_eval_buffer_offset_does_not_fit_usize"
            }
            Self::BufferSizeDoesNotFitUsize { .. } => {
                "spirv_vertex_eval_buffer_size_does_not_fit_usize"
            }
            Self::BufferRangeOverflow { .. } => "spirv_vertex_eval_buffer_range_overflow",
            Self::BufferReadOutOfBounds { .. } => "spirv_vertex_eval_buffer_read_out_of_bounds",
            Self::InputVariableUnsupported => "spirv_vertex_eval_input_variable_unsupported",
            Self::MemoryLoadVariableUnset { .. } => "spirv_vertex_eval_memory_load_variable_unset",
            Self::MemoryLoadPathIntoScalar => "spirv_vertex_eval_memory_load_path_into_scalar",
            Self::MemoryLoadPathOutOfRange { .. } => {
                "spirv_vertex_eval_memory_load_path_out_of_range"
            }
            Self::BufferStoreUnsupported => "spirv_vertex_eval_buffer_store_unsupported",
            Self::MemoryStoreVariableUnset { .. } => {
                "spirv_vertex_eval_memory_store_variable_unset"
            }
            Self::MemoryStorePathIntoScalar => "spirv_vertex_eval_memory_store_path_into_scalar",
            Self::MemoryStorePathOutOfRange { .. } => {
                "spirv_vertex_eval_memory_store_path_out_of_range"
            }
            Self::AccessChainBaseNotPointer => "spirv_vertex_eval_access_chain_base_not_pointer",
            Self::AccessChainBaseUnknown { .. } => "spirv_vertex_eval_access_chain_base_unknown",
            Self::AccessChainStructIndexOutOfRange { .. } => {
                "spirv_vertex_eval_access_chain_struct_index_out_of_range"
            }
            Self::AccessChainStructMemberOffsetMissing { .. } => {
                "spirv_vertex_eval_access_chain_struct_member_offset_missing"
            }
            Self::AccessChainStructOffsetOverflow => {
                "spirv_vertex_eval_access_chain_struct_offset_overflow"
            }
            Self::AccessChainArrayStrideMissing { .. } => {
                "spirv_vertex_eval_access_chain_array_stride_missing"
            }
            Self::AccessChainArrayOffsetOverflow => {
                "spirv_vertex_eval_access_chain_array_offset_overflow"
            }
            Self::AccessChainVectorIndexOutOfRange { .. } => {
                "spirv_vertex_eval_access_chain_vector_index_out_of_range"
            }
            Self::AccessChainVectorOffsetOverflow => {
                "spirv_vertex_eval_access_chain_vector_offset_overflow"
            }
            Self::AccessChainTypeUnsupported { .. } => {
                "spirv_vertex_eval_access_chain_type_unsupported"
            }
            Self::BufferVariableBindingMissing { .. } => {
                "spirv_vertex_eval_buffer_variable_binding_missing"
            }
            Self::StorageClassUnsupported { .. } => "spirv_vertex_eval_storage_class_unsupported",
            Self::UnaryFloatOperandTypeMismatch => {
                "spirv_vertex_eval_unary_float_operand_type_mismatch"
            }
            Self::BinaryFloatOperandShapeMismatch => {
                "spirv_vertex_eval_binary_float_operand_shape_mismatch"
            }
            Self::TernaryFloatOperandShapeMismatch => {
                "spirv_vertex_eval_ternary_float_operand_shape_mismatch"
            }
            Self::IntegerOperandShapeMismatch => "spirv_vertex_eval_integer_operand_shape_mismatch",
            Self::FloatVectorMemberTypeMismatch => {
                "spirv_vertex_eval_float_vector_member_type_mismatch"
            }
            Self::FloatVectorExpected => "spirv_vertex_eval_float_vector_expected",
            Self::IntVectorElementTypeMismatch => {
                "spirv_vertex_eval_int_vector_element_type_mismatch"
            }
            Self::IntResultTypeMismatch => "spirv_vertex_eval_int_result_type_mismatch",
            Self::PositionOutputMissing => "spirv_vertex_eval_position_output_missing",
            Self::EntryFunctionBodyMissingDuringRun => {
                "spirv_vertex_eval_entry_function_body_missing_during_run"
            }
            Self::FunctionFellOffEnd => "spirv_vertex_eval_function_fell_off_end",
            Self::MainInstructionBudgetExhausted => {
                "spirv_vertex_eval_main_instruction_budget_exhausted"
            }
            Self::FunctionInstructionMalformed => {
                "spirv_vertex_eval_function_instruction_malformed"
            }
            Self::FunctionVariableTypeNotPointer => {
                "spirv_vertex_eval_function_variable_type_not_pointer"
            }
            Self::FunctionVariableStorageClassInvalid { .. } => {
                "spirv_vertex_eval_function_variable_storage_class_invalid"
            }
            Self::LoadSourceNotPointer => "spirv_vertex_eval_load_source_not_pointer",
            Self::LoadPointerUnknown { .. } => "spirv_vertex_eval_load_pointer_unknown",
            Self::StoreTargetNotPointer => "spirv_vertex_eval_store_target_not_pointer",
            Self::StorePointerUnknown { .. } => "spirv_vertex_eval_store_pointer_unknown",
            Self::SignedConvertSourceWidthUnknown { .. } => {
                "spirv_vertex_eval_signed_convert_source_width_unknown"
            }
            Self::SignedToFloatSourceWidthUnknown { .. } => {
                "spirv_vertex_eval_signed_to_float_source_width_unknown"
            }
            Self::VectorScalarTypeMismatch => "spirv_vertex_eval_vector_scalar_type_mismatch",
            Self::MatrixScalarTypeMismatch => "spirv_vertex_eval_matrix_scalar_type_mismatch",
            Self::MatrixTimesVectorMatrixNotComposite => {
                "spirv_vertex_eval_matrix_times_vector_matrix_not_composite"
            }
            Self::MatrixTimesVectorColumnCountMismatch => {
                "spirv_vertex_eval_matrix_times_vector_column_count_mismatch"
            }
            Self::MatrixTimesVectorColumnHeightMismatch => {
                "spirv_vertex_eval_matrix_times_vector_column_height_mismatch"
            }
            Self::MatrixTimesVectorEmptyMatrix => {
                "spirv_vertex_eval_matrix_times_vector_empty_matrix"
            }
            Self::VectorTimesMatrixMatrixNotComposite => {
                "spirv_vertex_eval_vector_times_matrix_matrix_not_composite"
            }
            Self::VectorTimesMatrixShapeMismatch => {
                "spirv_vertex_eval_vector_times_matrix_shape_mismatch"
            }
            Self::MatrixTimesMatrixLeftNotComposite => {
                "spirv_vertex_eval_matrix_times_matrix_left_not_composite"
            }
            Self::MatrixTimesMatrixRightNotComposite => {
                "spirv_vertex_eval_matrix_times_matrix_right_not_composite"
            }
            Self::MatrixTimesMatrixShapeMismatch => {
                "spirv_vertex_eval_matrix_times_matrix_shape_mismatch"
            }
            Self::MatrixTimesMatrixEmptyMatrix => {
                "spirv_vertex_eval_matrix_times_matrix_empty_matrix"
            }
            Self::TransposeMatrixNotComposite => "spirv_vertex_eval_transpose_matrix_not_composite",
            Self::TransposeMatrixRagged => "spirv_vertex_eval_transpose_matrix_ragged",
            Self::DotShapeMismatch => "spirv_vertex_eval_dot_shape_mismatch",
            Self::CompositeExtractFromScalar => "spirv_vertex_eval_composite_extract_from_scalar",
            Self::CompositeExtractIndexOutOfRange { .. } => {
                "spirv_vertex_eval_composite_extract_index_out_of_range"
            }
            Self::CompositeInsertIntoScalar => "spirv_vertex_eval_composite_insert_into_scalar",
            Self::CompositeInsertIndexOutOfRange { .. } => {
                "spirv_vertex_eval_composite_insert_index_out_of_range"
            }
            Self::VectorShuffleLeftNotVector => "spirv_vertex_eval_vector_shuffle_left_not_vector",
            Self::VectorShuffleRightNotVector => {
                "spirv_vertex_eval_vector_shuffle_right_not_vector"
            }
            Self::VectorShuffleIndexOutOfRange { .. } => {
                "spirv_vertex_eval_vector_shuffle_index_out_of_range"
            }
            Self::SelectValuesNotComposite => "spirv_vertex_eval_select_values_not_composite",
            Self::SelectVectorLengthMismatch => "spirv_vertex_eval_select_vector_length_mismatch",
            Self::SelectVectorConditionNotBool => {
                "spirv_vertex_eval_select_vector_condition_not_bool"
            }
            Self::SelectConditionNotBool => "spirv_vertex_eval_select_condition_not_bool",
            Self::SignedCompareWidthUnknown => "spirv_vertex_eval_signed_compare_width_unknown",
            Self::LogicalNotVectorMemberNotBool => {
                "spirv_vertex_eval_logical_not_vector_member_not_bool"
            }
            Self::LogicalNotOperandNotBool => "spirv_vertex_eval_logical_not_operand_not_bool",
            Self::ExtInstSetUnknown { .. } => "spirv_vertex_eval_ext_inst_set_unknown",
            Self::PhiOutsideBlockEntry => "spirv_vertex_eval_phi_outside_block_entry",
            Self::BranchConditionNotBool => "spirv_vertex_eval_branch_condition_not_bool",
            Self::SwitchSelectorWidthUnknown { .. } => {
                "spirv_vertex_eval_switch_selector_width_unknown"
            }
            Self::SwitchOperandsMalformed => "spirv_vertex_eval_switch_operands_malformed",
            Self::UnexpectedTerminator { .. } => "spirv_vertex_eval_unexpected_terminator",
            Self::OpcodeUnsupported { .. } => "spirv_vertex_eval_opcode_unsupported",
            Self::BranchLabelUnknown { .. } => "spirv_vertex_eval_branch_label_unknown",
            Self::PhiInstructionMalformed => "spirv_vertex_eval_phi_instruction_malformed",
            Self::PhiPredecessorMissing { .. } => "spirv_vertex_eval_phi_predecessor_missing",
            Self::PhiInstructionBudgetExhausted => {
                "spirv_vertex_eval_phi_instruction_budget_exhausted"
            }
            Self::PositionVariableNeverStored { .. } => {
                "spirv_vertex_eval_position_variable_never_stored"
            }
            Self::PositionStructNeverStored { .. } => {
                "spirv_vertex_eval_position_struct_never_stored"
            }
            Self::PositionMemberNeverStored { .. } => {
                "spirv_vertex_eval_position_member_never_stored"
            }
            Self::PositionValueNotComposite => "spirv_vertex_eval_position_value_not_composite",
            Self::PositionVectorLengthInvalid { .. } => {
                "spirv_vertex_eval_position_vector_length_invalid"
            }
            Self::PositionComponentNotFinite { .. } => {
                "spirv_vertex_eval_position_component_not_finite"
            }
            Self::PositionComponentUndefined { .. } => {
                "spirv_vertex_eval_position_component_undefined"
            }
            Self::MapIntOperandTypeMismatch => "spirv_vertex_eval_map_int_operand_type_mismatch",
            Self::MapIntToFloatOperandTypeMismatch => {
                "spirv_vertex_eval_map_int_to_float_operand_type_mismatch"
            }
            Self::MapFloatToIntOperandTypeMismatch => {
                "spirv_vertex_eval_map_float_to_int_operand_type_mismatch"
            }
            Self::UnsignedDivisionByZero => "spirv_vertex_eval_unsigned_division_by_zero",
            Self::SignedDivisionByZero => "spirv_vertex_eval_signed_division_by_zero",
            Self::UnsignedModuloByZero => "spirv_vertex_eval_unsigned_modulo_by_zero",
            Self::SignedRemainderByZero => "spirv_vertex_eval_signed_remainder_by_zero",
            Self::IntegerBinaryOpcodeUnsupported { .. } => {
                "spirv_vertex_eval_integer_binary_opcode_unsupported"
            }
            Self::IntegerCompareOpcodeUnsupported { .. } => {
                "spirv_vertex_eval_integer_compare_opcode_unsupported"
            }
            Self::IntegerCompareShapeMismatch => "spirv_vertex_eval_integer_compare_shape_mismatch",
            Self::FloatCompareOpcodeUnsupported { .. } => {
                "spirv_vertex_eval_float_compare_opcode_unsupported"
            }
            Self::FloatCompareShapeMismatch => "spirv_vertex_eval_float_compare_shape_mismatch",
            Self::BooleanBinaryOpcodeUnsupported { .. } => {
                "spirv_vertex_eval_boolean_binary_opcode_unsupported"
            }
            Self::BooleanBinaryShapeMismatch => "spirv_vertex_eval_boolean_binary_shape_mismatch",
            Self::BitcastTypeUnsupported => "spirv_vertex_eval_bitcast_type_unsupported",
            Self::BitcastElementTypeUnsupported => {
                "spirv_vertex_eval_bitcast_element_type_unsupported"
            }
            Self::ExtArgumentMissing { .. } => "spirv_vertex_eval_ext_argument_missing",
            Self::UnpackUnormOperandNotInt => "spirv_vertex_eval_unpack_unorm_operand_not_int",
            Self::ExtendedOpcodeUnsupported { .. } => {
                "spirv_vertex_eval_extended_opcode_unsupported"
            }
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::TypeMissing { type_id }
            | Self::AccessChainTypeUnsupported { type_id }
            | Self::AccessChainArrayStrideMissing { type_id } => {
                vec![("type_id", type_id.to_string())]
            }
            Self::ValueIdUnset { id }
            | Self::LoadPointerUnknown { id }
            | Self::StorePointerUnknown { id }
            | Self::SignedConvertSourceWidthUnknown { id }
            | Self::SignedToFloatSourceWidthUnknown { id }
            | Self::SwitchSelectorWidthUnknown { id } => vec![("id", id.to_string())],
            Self::BufferBindingMissing { binding } => vec![("binding", binding.to_string())],
            Self::BufferOffsetDoesNotFitUsize { offset } => {
                vec![("offset", offset.to_string())]
            }
            Self::BufferSizeDoesNotFitUsize { size } => vec![("size", size.to_string())],
            Self::BufferRangeOverflow { offset, size } => {
                vec![("offset", offset.to_string()), ("size", size.to_string())]
            }
            Self::BufferReadOutOfBounds {
                binding,
                offset,
                size,
                len,
            } => vec![
                ("binding", binding.to_string()),
                ("offset", offset.to_string()),
                ("size", size.to_string()),
                ("len", len.to_string()),
            ],
            Self::MemoryLoadVariableUnset { variable }
            | Self::MemoryStoreVariableUnset { variable }
            | Self::BufferVariableBindingMissing { variable }
            | Self::PositionVariableNeverStored { variable }
            | Self::PositionStructNeverStored { variable } => {
                vec![("variable", variable.to_string())]
            }
            Self::MemoryLoadPathOutOfRange { index }
            | Self::MemoryStorePathOutOfRange { index }
            | Self::CompositeExtractIndexOutOfRange { index }
            | Self::CompositeInsertIndexOutOfRange { index }
            | Self::VectorShuffleIndexOutOfRange { index } => {
                vec![("index", index.to_string())]
            }
            Self::AccessChainBaseUnknown { base_id } => vec![("base_id", base_id.to_string())],
            Self::AccessChainStructIndexOutOfRange { type_id, index } => vec![
                ("type_id", type_id.to_string()),
                ("index", index.to_string()),
            ],
            Self::AccessChainStructMemberOffsetMissing { type_id, index } => vec![
                ("type_id", type_id.to_string()),
                ("index", index.to_string()),
            ],
            Self::AccessChainVectorIndexOutOfRange { index, count } => {
                vec![("index", index.to_string()), ("count", count.to_string())]
            }
            Self::StorageClassUnsupported { storage }
            | Self::FunctionVariableStorageClassInvalid { storage } => {
                vec![("storage", storage.to_string())]
            }
            Self::ExtInstSetUnknown { set_id } => vec![("set_id", set_id.to_string())],
            Self::UnexpectedTerminator { opcode }
            | Self::OpcodeUnsupported { opcode }
            | Self::IntegerBinaryOpcodeUnsupported { opcode }
            | Self::IntegerCompareOpcodeUnsupported { opcode }
            | Self::FloatCompareOpcodeUnsupported { opcode }
            | Self::BooleanBinaryOpcodeUnsupported { opcode } => {
                vec![("opcode", opcode.to_string())]
            }
            Self::BranchLabelUnknown { label } => vec![("label", label.to_string())],
            Self::PhiPredecessorMissing { predecessor } => {
                vec![("predecessor", predecessor.to_string())]
            }
            Self::PositionMemberNeverStored { variable, member } => vec![
                ("variable", variable.to_string()),
                ("member", member.to_string()),
            ],
            Self::PositionVectorLengthInvalid { len } => vec![("len", len.to_string())],
            Self::PositionComponentNotFinite { component }
            | Self::PositionComponentUndefined { component } => {
                vec![("component", component.to_string())]
            }
            Self::ExtArgumentMissing { index } => vec![("index", index.to_string())],
            Self::ExtendedOpcodeUnsupported { opcode } => vec![("opcode", opcode.to_string())],
            _ => Vec::new(),
        }
    }
}

impl std::fmt::Display for VertexEvalDecline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "reason={}", self.slug())?;
        for (key, value) in self.fields() {
            write!(f, " {key}={value}")?;
        }
        Ok(())
    }
}

impl std::error::Error for VertexEvalDecline {}

/// Instruction budget per evaluated vertex (live compositor shaders execute
/// a few hundred instructions; the cap only exists to bound adversarial loops).
const INSTRUCTION_BUDGET: usize = 65_536;

const HEADER_WORDS: usize = 5;

// Opcodes (numeric SPIR-V contract).
const OP_UNDEF: u16 = 1;
const OP_EXT_INST_IMPORT: u16 = 11;
const OP_EXT_INST: u16 = 12;
const OP_ENTRY_POINT: u16 = 15;
const OP_TYPE_VOID: u16 = 19;
const OP_TYPE_BOOL: u16 = 20;
const OP_TYPE_INT: u16 = 21;
const OP_TYPE_FLOAT: u16 = 22;
const OP_TYPE_VECTOR: u16 = 23;
const OP_TYPE_MATRIX: u16 = 24;
const OP_TYPE_ARRAY: u16 = 28;
const OP_TYPE_RUNTIME_ARRAY: u16 = 29;
const OP_TYPE_STRUCT: u16 = 30;
const OP_TYPE_POINTER: u16 = 32;
const OP_CONSTANT_TRUE: u16 = 41;
const OP_CONSTANT_FALSE: u16 = 42;
const OP_CONSTANT: u16 = 43;
const OP_CONSTANT_COMPOSITE: u16 = 44;
const OP_CONSTANT_NULL: u16 = 46;
const OP_FUNCTION: u16 = 54;
const OP_FUNCTION_END: u16 = 56;
const OP_VARIABLE: u16 = 59;
const OP_LOAD: u16 = 61;
const OP_STORE: u16 = 62;
const OP_ACCESS_CHAIN: u16 = 65;
const OP_IN_BOUNDS_ACCESS_CHAIN: u16 = 66;
const OP_DECORATE: u16 = 71;
const OP_MEMBER_DECORATE: u16 = 72;
const OP_VECTOR_SHUFFLE: u16 = 79;
const OP_COMPOSITE_CONSTRUCT: u16 = 80;
const OP_COMPOSITE_EXTRACT: u16 = 81;
const OP_COMPOSITE_INSERT: u16 = 82;
const OP_COPY_OBJECT: u16 = 83;
const OP_TRANSPOSE: u16 = 84;
const OP_CONVERT_F_TO_U: u16 = 109;
const OP_CONVERT_F_TO_S: u16 = 110;
const OP_CONVERT_S_TO_F: u16 = 111;
const OP_CONVERT_U_TO_F: u16 = 112;
const OP_U_CONVERT: u16 = 113;
const OP_S_CONVERT: u16 = 114;
const OP_BITCAST: u16 = 124;
const OP_S_NEGATE: u16 = 126;
const OP_F_NEGATE: u16 = 127;
const OP_I_ADD: u16 = 128;
const OP_F_ADD: u16 = 129;
const OP_I_SUB: u16 = 130;
const OP_F_SUB: u16 = 131;
const OP_I_MUL: u16 = 132;
const OP_F_MUL: u16 = 133;
const OP_U_DIV: u16 = 134;
const OP_S_DIV: u16 = 135;
const OP_F_DIV: u16 = 136;
const OP_U_MOD: u16 = 137;
const OP_S_REM: u16 = 138;
const OP_F_REM: u16 = 140;
const OP_F_MOD: u16 = 141;
const OP_VECTOR_TIMES_SCALAR: u16 = 142;
const OP_MATRIX_TIMES_SCALAR: u16 = 143;
const OP_VECTOR_TIMES_MATRIX: u16 = 144;
const OP_MATRIX_TIMES_VECTOR: u16 = 145;
const OP_MATRIX_TIMES_MATRIX: u16 = 146;
const OP_DOT: u16 = 148;
const OP_LOGICAL_EQUAL: u16 = 164;
const OP_LOGICAL_NOT_EQUAL: u16 = 165;
const OP_LOGICAL_OR: u16 = 166;
const OP_LOGICAL_AND: u16 = 167;
const OP_LOGICAL_NOT: u16 = 168;
const OP_SELECT: u16 = 169;
const OP_I_EQUAL: u16 = 170;
const OP_I_NOT_EQUAL: u16 = 171;
const OP_U_GREATER_THAN: u16 = 172;
const OP_S_GREATER_THAN: u16 = 173;
const OP_U_GREATER_THAN_EQUAL: u16 = 174;
const OP_S_GREATER_THAN_EQUAL: u16 = 175;
const OP_U_LESS_THAN: u16 = 176;
const OP_S_LESS_THAN: u16 = 177;
const OP_U_LESS_THAN_EQUAL: u16 = 178;
const OP_S_LESS_THAN_EQUAL: u16 = 179;
const OP_F_ORD_EQUAL: u16 = 180;
const OP_F_ORD_NOT_EQUAL: u16 = 182;
const OP_F_ORD_LESS_THAN: u16 = 184;
const OP_F_ORD_GREATER_THAN: u16 = 186;
const OP_F_ORD_LESS_THAN_EQUAL: u16 = 188;
const OP_F_ORD_GREATER_THAN_EQUAL: u16 = 190;
const OP_SHIFT_RIGHT_LOGICAL: u16 = 194;
const OP_SHIFT_RIGHT_ARITHMETIC: u16 = 195;
const OP_SHIFT_LEFT_LOGICAL: u16 = 196;
const OP_BITWISE_OR: u16 = 197;
const OP_BITWISE_XOR: u16 = 198;
const OP_BITWISE_AND: u16 = 199;
const OP_NOT: u16 = 200;
const OP_PHI: u16 = 245;
const OP_LOOP_MERGE: u16 = 246;
const OP_SELECTION_MERGE: u16 = 247;
const OP_LABEL: u16 = 248;
const OP_BRANCH: u16 = 249;
const OP_BRANCH_CONDITIONAL: u16 = 250;
const OP_SWITCH: u16 = 251;
const OP_RETURN: u16 = 253;
const OP_RETURN_VALUE: u16 = 254;
const OP_UNREACHABLE: u16 = 255;
const OP_LINE: u16 = 8;
const OP_NO_LINE: u16 = 317;
const OP_NOP: u16 = 0;

// Decorations.
const DECORATION_ARRAY_STRIDE: u32 = 6;
const DECORATION_BUILT_IN: u32 = 11;
const DECORATION_BINDING: u32 = 33;
const DECORATION_OFFSET: u32 = 35;

// Built-ins.
const BUILT_IN_POSITION: u32 = 0;
const BUILT_IN_VERTEX_INDEX: u32 = 42;
const BUILT_IN_INSTANCE_INDEX: u32 = 43;

// Storage classes.
const STORAGE_CLASS_INPUT: u32 = 1;
const STORAGE_CLASS_UNIFORM: u32 = 2;
const STORAGE_CLASS_OUTPUT: u32 = 3;
const STORAGE_CLASS_PRIVATE: u32 = 6;
const STORAGE_CLASS_FUNCTION: u32 = 7;
const STORAGE_CLASS_STORAGE_BUFFER: u32 = 12;

// GLSL.std.450 instructions (numeric ext contract).
const GLSL_FABS: u32 = 4;
const GLSL_FLOOR: u32 = 8;
const GLSL_CEIL: u32 = 9;
const GLSL_FRACT: u32 = 10;
const GLSL_SIN: u32 = 13;
const GLSL_COS: u32 = 14;
const GLSL_POW: u32 = 26;
const GLSL_SQRT: u32 = 31;
const GLSL_INVERSE_SQRT: u32 = 32;
const GLSL_FMIN: u32 = 37;
const GLSL_FMAX: u32 = 40;
const GLSL_FCLAMP: u32 = 43;
const GLSL_FMIX: u32 = 46;
const GLSL_STEP: u32 = 48;
const GLSL_FMA: u32 = 50;
const GLSL_UNPACK_UNORM_4X8: u32 = 64;

#[derive(Clone, Debug, PartialEq)]
enum Type {
    Void,
    Bool,
    Int { width: u32 },
    Float { width: u32 },
    Vector { elem: u32, count: u32 },
    Matrix { column: u32, count: u32 },
    Array { elem: u32 },
    RuntimeArray { elem: u32 },
    Struct { members: Vec<u32> },
    Pointer { storage: u32, pointee: u32 },
}

#[derive(Clone, Debug, PartialEq)]
enum Value {
    Undef,
    Bool(bool),
    /// Raw bits, masked to the declared width on construction.
    Int(u64),
    Float(f32),
    Composite(Vec<Value>),
    Ptr(PtrVal),
}

#[derive(Clone, Debug, PartialEq)]
struct PtrVal {
    target: PtrTarget,
    pointee: u32,
}

#[derive(Clone, Debug, PartialEq)]
enum PtrTarget {
    Buffer { binding: u32, offset: u64 },
    Memory { var: u32, path: Vec<u32> },
}

struct Module<'w> {
    words: &'w [u32],
    types: HashMap<u32, Type>,
    /// id → pre-seeded constant/undef value.
    constants: HashMap<u32, Value>,
    /// struct type id → member index → byte offset.
    member_offsets: HashMap<(u32, u32), u32>,
    /// array type id → element stride.
    array_strides: HashMap<u32, u32>,
    built_ins: HashMap<u32, u32>,
    /// (struct type id, member index) → member built-in (gl_PerVertex style).
    member_built_ins: Option<HashMap<(u32, u32), u32>>,
    bindings: HashMap<u32, u32>,
    /// global variable id → (storage class, pointee type id).
    globals: HashMap<u32, (u32, u32)>,
    /// constant id → int element width (for sign/selector semantics).
    const_int_widths: HashMap<u32, u32>,
    glsl_ext: Option<u32>,
    entry_id: Option<u32>,
    /// entry function body instruction word range (first OpLabel .. OpFunctionEnd).
    body: Option<(usize, usize)>,
    /// label id → word index of the OpLabel instruction.
    labels: HashMap<u32, usize>,
}

fn parse(words: &[u32]) -> Result<Module<'_>, VertexEvalDecline> {
    if words.len() < HEADER_WORDS {
        return Err(VertexEvalDecline::MalformedHeader);
    }
    let mut m = Module {
        words,
        types: HashMap::new(),
        constants: HashMap::new(),
        member_offsets: HashMap::new(),
        array_strides: HashMap::new(),
        built_ins: HashMap::new(),
        member_built_ins: None,
        bindings: HashMap::new(),
        globals: HashMap::new(),
        const_int_widths: HashMap::new(),
        glsl_ext: None,
        entry_id: None,
        body: None,
        labels: HashMap::new(),
    };
    let mut i = HEADER_WORDS;
    let mut cur_function: Option<u32> = None;
    let mut entry_body_start: Option<usize> = None;
    while i < words.len() {
        let word0 = words[i];
        let wc = (word0 >> 16) as usize;
        let op = (word0 & 0xffff) as u16;
        if wc == 0 || i + wc > words.len() {
            return Err(VertexEvalDecline::ModuleInstructionMalformed);
        }
        let operands = &words[i + 1..i + wc];
        match op {
            OP_EXT_INST_IMPORT => {
                // Name is a literal string; GLSL.std.450 == "GLSL.std.450".
                let name_words = &operands[1..];
                let bytes: Vec<u8> = name_words
                    .iter()
                    .flat_map(|w| w.to_le_bytes())
                    .take_while(|b| *b != 0)
                    .collect();
                if bytes == b"GLSL.std.450" {
                    m.glsl_ext = Some(operands[0]);
                }
            }
            OP_ENTRY_POINT => {
                // ExecutionModel Vertex == 0.
                if operands.first() == Some(&0) && m.entry_id.is_none() {
                    m.entry_id = Some(operands[1]);
                }
            }
            OP_TYPE_VOID => {
                m.types.insert(operands[0], Type::Void);
            }
            OP_TYPE_BOOL => {
                m.types.insert(operands[0], Type::Bool);
            }
            OP_TYPE_INT => {
                m.types
                    .insert(operands[0], Type::Int { width: operands[1] });
            }
            OP_TYPE_FLOAT => {
                m.types
                    .insert(operands[0], Type::Float { width: operands[1] });
            }
            OP_TYPE_VECTOR => {
                m.types.insert(
                    operands[0],
                    Type::Vector {
                        elem: operands[1],
                        count: operands[2],
                    },
                );
            }
            OP_TYPE_MATRIX => {
                m.types.insert(
                    operands[0],
                    Type::Matrix {
                        column: operands[1],
                        count: operands[2],
                    },
                );
            }
            OP_TYPE_ARRAY => {
                m.types
                    .insert(operands[0], Type::Array { elem: operands[1] });
            }
            OP_TYPE_RUNTIME_ARRAY => {
                m.types
                    .insert(operands[0], Type::RuntimeArray { elem: operands[1] });
            }
            OP_TYPE_STRUCT => {
                m.types.insert(
                    operands[0],
                    Type::Struct {
                        members: operands[1..].to_vec(),
                    },
                );
            }
            OP_TYPE_POINTER => {
                m.types.insert(
                    operands[0],
                    Type::Pointer {
                        storage: operands[1],
                        pointee: operands[2],
                    },
                );
            }
            OP_CONSTANT_TRUE => {
                m.constants.insert(operands[1], Value::Bool(true));
            }
            OP_CONSTANT_FALSE => {
                m.constants.insert(operands[1], Value::Bool(false));
            }
            OP_CONSTANT => {
                let ty = m
                    .types
                    .get(&operands[0])
                    .ok_or(VertexEvalDecline::ConstantTypeMissing)?;
                let value = match ty {
                    Type::Int { width } => {
                        let raw = if *width > 32 && operands.len() >= 4 {
                            (operands[2] as u64) | ((operands[3] as u64) << 32)
                        } else {
                            operands[2] as u64
                        };
                        m.const_int_widths.insert(operands[1], *width);
                        Value::Int(mask_width(raw, *width))
                    }
                    Type::Float { width: 32 } => Value::Float(f32::from_bits(operands[2])),
                    _ => return Err(VertexEvalDecline::ConstantTypeUnsupported),
                };
                m.constants.insert(operands[1], value);
            }
            OP_CONSTANT_COMPOSITE => {
                let members = operands[2..]
                    .iter()
                    .map(|id| {
                        m.constants
                            .get(id)
                            .cloned()
                            .ok_or(VertexEvalDecline::CompositeConstantForwardReference)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                m.constants.insert(operands[1], Value::Composite(members));
            }
            OP_CONSTANT_NULL => {
                let value = null_value(&m, operands[0])?;
                m.constants.insert(operands[1], value);
            }
            OP_UNDEF if cur_function.is_none() => {
                let value = undef_value(&m, operands[0])?;
                m.constants.insert(operands[1], value);
            }
            OP_DECORATE if wc >= 4 => match operands[1] {
                DECORATION_BUILT_IN => {
                    m.built_ins.insert(operands[0], operands[2]);
                }
                DECORATION_BINDING => {
                    m.bindings.insert(operands[0], operands[2]);
                }
                DECORATION_ARRAY_STRIDE => {
                    m.array_strides.insert(operands[0], operands[2]);
                }
                _ => {}
            },
            OP_MEMBER_DECORATE if wc >= 5 => {
                if operands[2] == DECORATION_OFFSET {
                    m.member_offsets
                        .insert((operands[0], operands[1]), operands[3]);
                } else if operands[2] == DECORATION_BUILT_IN {
                    // Member built-ins (gl_PerVertex style) keyed as struct-type
                    // pseudo ids: track through built_ins with a shifted key is
                    // not needed — position extraction handles the member case
                    // by querying member_built_ins directly.
                    m.member_built_ins
                        .get_or_insert_with(HashMap::new)
                        .insert((operands[0], operands[1]), operands[3]);
                }
            }
            OP_VARIABLE if cur_function.is_none() => {
                let ptr_type = m
                    .types
                    .get(&operands[0])
                    .ok_or(VertexEvalDecline::GlobalVariableTypeMissing)?;
                let Type::Pointer { storage, pointee } = ptr_type else {
                    return Err(VertexEvalDecline::GlobalVariableTypeNotPointer);
                };
                m.globals.insert(operands[1], (*storage, *pointee));
            }
            OP_FUNCTION => {
                cur_function = Some(operands[1]);
                if Some(operands[1]) == m.entry_id {
                    entry_body_start = Some(i + wc);
                }
            }
            OP_FUNCTION_END => {
                if cur_function == m.entry_id {
                    if let Some(start) = entry_body_start {
                        m.body = Some((start, i));
                    }
                }
                cur_function = None;
            }
            OP_LABEL if cur_function == m.entry_id && entry_body_start.is_some() => {
                m.labels.insert(operands[0], i);
            }
            _ => {}
        }
        i += wc;
    }
    if m.entry_id.is_none() {
        return Err(VertexEvalDecline::VertexEntryPointMissing);
    }
    if m.body.is_none() {
        return Err(VertexEvalDecline::EntryFunctionBodyMissingDuringParse);
    }
    Ok(m)
}

impl<'w> Module<'w> {
    fn ty(&self, id: u32) -> Result<&Type, VertexEvalDecline> {
        self.types
            .get(&id)
            .ok_or(VertexEvalDecline::TypeMissing { type_id: id })
    }

    fn scalar_byte_size(&self, id: u32) -> Result<u64, VertexEvalDecline> {
        match self.ty(id)? {
            Type::Int { width } | Type::Float { width } => Ok(u64::from(*width) / 8),
            _ => Err(VertexEvalDecline::ScalarSizeTypeUnsupported),
        }
    }
}

fn mask_width(raw: u64, width: u32) -> u64 {
    if width >= 64 {
        raw
    } else {
        raw & ((1u64 << width) - 1)
    }
}

fn null_value(m: &Module<'_>, type_id: u32) -> Result<Value, VertexEvalDecline> {
    match m.ty(type_id)? {
        Type::Bool => Ok(Value::Bool(false)),
        Type::Int { .. } => Ok(Value::Int(0)),
        Type::Float { width: 32 } => Ok(Value::Float(0.0)),
        Type::Vector { elem, count }
        | Type::Matrix {
            column: elem,
            count,
        } => {
            let member = null_value(m, *elem)?;
            Ok(Value::Composite(vec![member; *count as usize]))
        }
        _ => Err(VertexEvalDecline::NullTypeUnsupported),
    }
}

fn undef_value(m: &Module<'_>, type_id: u32) -> Result<Value, VertexEvalDecline> {
    Ok(match m.ty(type_id)? {
        Type::Vector { count, .. } | Type::Matrix { count, .. } => {
            Value::Composite(vec![Value::Undef; *count as usize])
        }
        Type::Struct { members } => Value::Composite(vec![Value::Undef; members.len()]),
        _ => Value::Undef,
    })
}

/// Default (uninitialized) storage for Function/Private/Output variables.
fn default_storage(m: &Module<'_>, type_id: u32) -> Result<Value, VertexEvalDecline> {
    Ok(match m.ty(type_id)? {
        Type::Vector { elem, count }
        | Type::Matrix {
            column: elem,
            count,
        } => {
            let member = default_storage(m, *elem)?;
            Value::Composite(vec![member; *count as usize])
        }
        Type::Struct { members } => Value::Composite(
            members
                .iter()
                .map(|member| default_storage(m, *member))
                .collect::<Result<Vec<_>, _>>()?,
        ),
        Type::Array { .. } | Type::RuntimeArray { .. } => {
            return Err(VertexEvalDecline::ArrayVariableUnsupported)
        }
        _ => Value::Undef,
    })
}

struct Exec<'m, 'w, 'b> {
    m: &'m Module<'w>,
    buffers: &'b [(u32, &'b [u8])],
    env: HashMap<u32, Value>,
    vars: HashMap<u32, Value>,
    /// runtime result id → int element width (sign/selector semantics).
    int_widths: HashMap<u32, u32>,
    vertex_index: u32,
    instance_index: u32,
    budget: usize,
}

impl<'m, 'w, 'b> Exec<'m, 'w, 'b> {
    fn value(&self, id: u32) -> Result<Value, VertexEvalDecline> {
        if let Some(v) = self.env.get(&id) {
            return Ok(v.clone());
        }
        if let Some(v) = self.m.constants.get(&id) {
            return Ok(v.clone());
        }
        Err(VertexEvalDecline::ValueIdUnset { id })
    }

    fn int_operand(&self, id: u32) -> Result<u64, VertexEvalDecline> {
        match self.value(id)? {
            Value::Int(raw) => Ok(raw),
            _ => Err(VertexEvalDecline::IntOperandTypeMismatch),
        }
    }

    /// The int element width of a computed or constant id, when tracked.
    fn int_width_of(&self, id: u32) -> Option<u32> {
        self.int_widths
            .get(&id)
            .or_else(|| self.m.const_int_widths.get(&id))
            .copied()
    }

    /// Record the int element width of a result id from its result type.
    fn note_int_width(&mut self, id: u32, type_id: u32) {
        let elem = match self.m.ty(type_id) {
            Ok(Type::Int { width }) => Some(*width),
            Ok(Type::Vector { elem, .. }) => match self.m.ty(*elem) {
                Ok(Type::Int { width }) => Some(*width),
                _ => None,
            },
            _ => None,
        };
        if let Some(width) = elem {
            self.int_widths.insert(id, width);
        }
    }

    fn buffer_bytes(&self, binding: u32) -> Result<&'b [u8], VertexEvalDecline> {
        self.buffers
            .iter()
            .find(|(b, _)| *b == binding)
            .map(|(_, bytes)| *bytes)
            .ok_or(VertexEvalDecline::BufferBindingMissing { binding })
    }

    fn read_buffer_typed(
        &self,
        binding: u32,
        offset: u64,
        type_id: u32,
    ) -> Result<Value, VertexEvalDecline> {
        match self.m.ty(type_id)? {
            Type::Int { width } => {
                let size = u64::from(*width) / 8;
                let bytes = self.read_buffer_raw(binding, offset, size)?;
                let mut raw = 0u64;
                for (i, b) in bytes.iter().enumerate() {
                    raw |= (*b as u64) << (8 * i);
                }
                Ok(Value::Int(raw))
            }
            Type::Float { width: 32 } => {
                let bytes = self.read_buffer_raw(binding, offset, 4)?;
                Ok(Value::Float(f32::from_le_bytes(bytes.try_into().map_err(
                    |_| VertexEvalDecline::FloatBufferReadWidthMismatch,
                )?)))
            }
            Type::Vector { elem, count } => {
                let stride = self.m.scalar_byte_size(*elem)?;
                let members = (0..*count)
                    .map(|i| self.read_buffer_typed(binding, offset + stride * u64::from(i), *elem))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Value::Composite(members))
            }
            _ => Err(VertexEvalDecline::BufferLoadTypeUnsupported),
        }
    }

    fn read_buffer_raw(
        &self,
        binding: u32,
        offset: u64,
        size: u64,
    ) -> Result<&'b [u8], VertexEvalDecline> {
        let bytes = self.buffer_bytes(binding)?;
        let start = usize::try_from(offset)
            .map_err(|_| VertexEvalDecline::BufferOffsetDoesNotFitUsize { offset })?;
        let size_usize = usize::try_from(size)
            .map_err(|_| VertexEvalDecline::BufferSizeDoesNotFitUsize { size })?;
        let end = start
            .checked_add(size_usize)
            .ok_or(VertexEvalDecline::BufferRangeOverflow { offset, size })?;
        bytes
            .get(start..end)
            .ok_or(VertexEvalDecline::BufferReadOutOfBounds {
                binding,
                offset,
                size,
                len: bytes.len(),
            })
    }

    fn load_ptr(&self, ptr: &PtrVal) -> Result<Value, VertexEvalDecline> {
        match &ptr.target {
            PtrTarget::Buffer { binding, offset } => {
                self.read_buffer_typed(*binding, *offset, ptr.pointee)
            }
            PtrTarget::Memory { var, path } => {
                if let Some((storage, _)) = self.m.globals.get(var) {
                    if *storage == STORAGE_CLASS_INPUT {
                        return match self.m.built_ins.get(var) {
                            Some(&BUILT_IN_VERTEX_INDEX) => {
                                Ok(Value::Int(u64::from(self.vertex_index)))
                            }
                            Some(&BUILT_IN_INSTANCE_INDEX) => {
                                Ok(Value::Int(u64::from(self.instance_index)))
                            }
                            _ => Err(VertexEvalDecline::InputVariableUnsupported),
                        };
                    }
                }
                let mut value = self
                    .vars
                    .get(var)
                    .ok_or(VertexEvalDecline::MemoryLoadVariableUnset { variable: *var })?;
                for idx in path {
                    let Value::Composite(members) = value else {
                        return Err(VertexEvalDecline::MemoryLoadPathIntoScalar);
                    };
                    value = members
                        .get(*idx as usize)
                        .ok_or(VertexEvalDecline::MemoryLoadPathOutOfRange { index: *idx })?;
                }
                Ok(value.clone())
            }
        }
    }

    fn store_ptr(&mut self, ptr: &PtrVal, new: Value) -> Result<(), VertexEvalDecline> {
        match &ptr.target {
            PtrTarget::Buffer { .. } => Err(VertexEvalDecline::BufferStoreUnsupported),
            PtrTarget::Memory { var, path } => {
                let slot = self
                    .vars
                    .get_mut(var)
                    .ok_or(VertexEvalDecline::MemoryStoreVariableUnset { variable: *var })?;
                let mut value = slot;
                for idx in path {
                    let Value::Composite(members) = value else {
                        return Err(VertexEvalDecline::MemoryStorePathIntoScalar);
                    };
                    value = members
                        .get_mut(*idx as usize)
                        .ok_or(VertexEvalDecline::MemoryStorePathOutOfRange { index: *idx })?;
                }
                *value = new;
                Ok(())
            }
        }
    }

    /// Resolve an access chain from a base pointer through constant/dynamic
    /// indices, producing a new pointer (byte offset for buffers, element path
    /// for memory variables).
    fn access_chain(&self, base_id: u32, index_ids: &[u32]) -> Result<PtrVal, VertexEvalDecline> {
        let base = match self.env.get(&base_id) {
            Some(Value::Ptr(p)) => p.clone(),
            Some(_) => return Err(VertexEvalDecline::AccessChainBaseNotPointer),
            None => {
                let (storage, pointee) = self
                    .m
                    .globals
                    .get(&base_id)
                    .ok_or(VertexEvalDecline::AccessChainBaseUnknown { base_id })?;
                self.global_ptr(base_id, *storage, *pointee)?
            }
        };
        let mut pointee = base.pointee;
        let mut target = base.target;
        for index_id in index_ids {
            let index = self.int_operand(*index_id)?;
            let current_type = pointee;
            match self.m.ty(pointee)?.clone() {
                Type::Struct { members } => {
                    let member = members.get(index as usize).copied().ok_or(
                        VertexEvalDecline::AccessChainStructIndexOutOfRange {
                            type_id: current_type,
                            index,
                        },
                    )?;
                    match &mut target {
                        PtrTarget::Buffer { offset, .. } => {
                            let member_offset = self
                                .m
                                .member_offsets
                                .get(&(pointee, index as u32))
                                .copied()
                                .ok_or(VertexEvalDecline::AccessChainStructMemberOffsetMissing {
                                    type_id: current_type,
                                    index: index as u32,
                                })?;
                            *offset = offset
                                .checked_add(u64::from(member_offset))
                                .ok_or(VertexEvalDecline::AccessChainStructOffsetOverflow)?;
                        }
                        PtrTarget::Memory { path, .. } => path.push(index as u32),
                    }
                    pointee = member;
                }
                Type::Array { elem } | Type::RuntimeArray { elem } => {
                    match &mut target {
                        PtrTarget::Buffer { offset, .. } => {
                            let stride = self.m.array_strides.get(&pointee).copied().ok_or(
                                VertexEvalDecline::AccessChainArrayStrideMissing {
                                    type_id: current_type,
                                },
                            )?;
                            *offset = index
                                .checked_mul(u64::from(stride))
                                .and_then(|o| offset.checked_add(o))
                                .ok_or(VertexEvalDecline::AccessChainArrayOffsetOverflow)?;
                        }
                        PtrTarget::Memory { path, .. } => path.push(index as u32),
                    }
                    pointee = elem;
                }
                Type::Vector { elem, count } => {
                    if index >= u64::from(count) {
                        return Err(VertexEvalDecline::AccessChainVectorIndexOutOfRange {
                            index,
                            count,
                        });
                    }
                    match &mut target {
                        PtrTarget::Buffer { offset, .. } => {
                            let stride = self.m.scalar_byte_size(elem)?;
                            *offset = offset
                                .checked_add(stride * index)
                                .ok_or(VertexEvalDecline::AccessChainVectorOffsetOverflow)?;
                        }
                        PtrTarget::Memory { path, .. } => path.push(index as u32),
                    }
                    pointee = elem;
                }
                _ => {
                    return Err(VertexEvalDecline::AccessChainTypeUnsupported {
                        type_id: current_type,
                    })
                }
            }
        }
        Ok(PtrVal { target, pointee })
    }

    fn global_ptr(&self, id: u32, storage: u32, pointee: u32) -> Result<PtrVal, VertexEvalDecline> {
        let target =
            match storage {
                STORAGE_CLASS_STORAGE_BUFFER | STORAGE_CLASS_UNIFORM => {
                    let binding =
                        self.m.bindings.get(&id).copied().ok_or(
                            VertexEvalDecline::BufferVariableBindingMissing { variable: id },
                        )?;
                    PtrTarget::Buffer { binding, offset: 0 }
                }
                STORAGE_CLASS_INPUT
                | STORAGE_CLASS_OUTPUT
                | STORAGE_CLASS_PRIVATE
                | STORAGE_CLASS_FUNCTION => PtrTarget::Memory {
                    var: id,
                    path: Vec::new(),
                },
                _ => return Err(VertexEvalDecline::StorageClassUnsupported { storage }),
            };
        Ok(PtrVal { target, pointee })
    }
}

fn componentwise1(a: &Value, f: impl Fn(f32) -> f32 + Copy) -> Result<Value, VertexEvalDecline> {
    match a {
        Value::Float(x) => Ok(Value::Float(f(*x))),
        Value::Composite(m) => Ok(Value::Composite(
            m.iter()
                .map(|v| componentwise1(v, f))
                .collect::<Result<Vec<_>, _>>()?,
        )),
        _ => Err(VertexEvalDecline::UnaryFloatOperandTypeMismatch),
    }
}

fn componentwise2(
    a: &Value,
    b: &Value,
    f: impl Fn(f32, f32) -> f32 + Copy,
) -> Result<Value, VertexEvalDecline> {
    match (a, b) {
        (Value::Float(x), Value::Float(y)) => Ok(Value::Float(f(*x, *y))),
        (Value::Composite(ma), Value::Composite(mb)) if ma.len() == mb.len() => {
            Ok(Value::Composite(
                ma.iter()
                    .zip(mb)
                    .map(|(x, y)| componentwise2(x, y, f))
                    .collect::<Result<Vec<_>, _>>()?,
            ))
        }
        _ => Err(VertexEvalDecline::BinaryFloatOperandShapeMismatch),
    }
}

fn componentwise3(
    a: &Value,
    b: &Value,
    c: &Value,
    f: impl Fn(f32, f32, f32) -> f32 + Copy,
) -> Result<Value, VertexEvalDecline> {
    match (a, b, c) {
        (Value::Float(x), Value::Float(y), Value::Float(z)) => Ok(Value::Float(f(*x, *y, *z))),
        (Value::Composite(ma), Value::Composite(mb), Value::Composite(mc))
            if ma.len() == mb.len() && ma.len() == mc.len() =>
        {
            Ok(Value::Composite(
                ma.iter()
                    .zip(mb)
                    .zip(mc)
                    .map(|((x, y), z)| componentwise3(x, y, z, f))
                    .collect::<Result<Vec<_>, _>>()?,
            ))
        }
        _ => Err(VertexEvalDecline::TernaryFloatOperandShapeMismatch),
    }
}

fn int_componentwise2(
    a: &Value,
    b: &Value,
    width: u32,
    f: impl Fn(u64, u64, u32) -> Result<u64, VertexEvalDecline> + Copy,
) -> Result<Value, VertexEvalDecline> {
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => Ok(Value::Int(mask_width(f(*x, *y, width)?, width))),
        (Value::Composite(ma), Value::Composite(mb)) if ma.len() == mb.len() => {
            Ok(Value::Composite(
                ma.iter()
                    .zip(mb)
                    .map(|(x, y)| int_componentwise2(x, y, width, f))
                    .collect::<Result<Vec<_>, _>>()?,
            ))
        }
        _ => Err(VertexEvalDecline::IntegerOperandShapeMismatch),
    }
}

fn sign_extend(raw: u64, width: u32) -> i64 {
    if width >= 64 {
        raw as i64
    } else {
        let shift = 64 - width;
        ((raw << shift) as i64) >> shift
    }
}

fn float_vec(v: &Value) -> Result<Vec<f32>, VertexEvalDecline> {
    match v {
        Value::Float(x) => Ok(vec![*x]),
        Value::Composite(m) => m
            .iter()
            .map(|c| match c {
                Value::Float(x) => Ok(*x),
                _ => Err(VertexEvalDecline::FloatVectorMemberTypeMismatch),
            })
            .collect(),
        _ => Err(VertexEvalDecline::FloatVectorExpected),
    }
}

/// The scalar element width of an int result type (vectors use the element).
fn int_result_width(m: &Module<'_>, type_id: u32) -> Result<u32, VertexEvalDecline> {
    match m.ty(type_id)? {
        Type::Int { width } => Ok(*width),
        Type::Vector { elem, .. } => match m.ty(*elem)? {
            Type::Int { width } => Ok(*width),
            _ => Err(VertexEvalDecline::IntVectorElementTypeMismatch),
        },
        _ => Err(VertexEvalDecline::IntResultTypeMismatch),
    }
}

/// Evaluate the vertex entry point for each vertex index and return the clip
/// positions stored to the `Position` built-in, in input order.
///
/// `buffers` maps merged-set-0 bindings to the exact decoded bytes bound for
/// this draw. Every failure is a compact slug for the coverage-gap log.
pub fn evaluate_vertex_clip_positions(
    words: &[u32],
    buffers: &[(u32, &[u8])],
    vertex_indices: &[u32],
    instance_index: u32,
) -> Result<Vec<[f32; 4]>, VertexEvalDecline> {
    let module = parse(words)?;
    let position_var = find_position_output(&module)?;
    let mut out = Vec::with_capacity(vertex_indices.len());
    for &vi in vertex_indices {
        let position = run_vertex(&module, buffers, vi, instance_index, position_var)?;
        out.push(position);
    }
    Ok(out)
}

#[derive(Clone, Copy, Debug)]
enum PositionOutput {
    /// Output variable directly decorated `BuiltIn Position`.
    Variable(u32),
    /// Output struct variable whose member is decorated `BuiltIn Position`.
    Member(u32, u32),
}

fn find_position_output(m: &Module<'_>) -> Result<PositionOutput, VertexEvalDecline> {
    for (id, (storage, pointee)) in &m.globals {
        if *storage != STORAGE_CLASS_OUTPUT {
            continue;
        }
        if m.built_ins.get(id) == Some(&BUILT_IN_POSITION) {
            return Ok(PositionOutput::Variable(*id));
        }
        if let Some(member_built_ins) = &m.member_built_ins {
            if let Ok(Type::Struct { members }) = m.ty(*pointee) {
                for member in 0..members.len() as u32 {
                    if member_built_ins.get(&(*pointee, member)) == Some(&BUILT_IN_POSITION) {
                        return Ok(PositionOutput::Member(*id, member));
                    }
                }
            }
        }
    }
    Err(VertexEvalDecline::PositionOutputMissing)
}

fn run_vertex(
    m: &Module<'_>,
    buffers: &[(u32, &[u8])],
    vertex_index: u32,
    instance_index: u32,
    position: PositionOutput,
) -> Result<[f32; 4], VertexEvalDecline> {
    let (body_start, body_end) = m
        .body
        .ok_or(VertexEvalDecline::EntryFunctionBodyMissingDuringRun)?;
    let mut exec = Exec {
        m,
        buffers,
        env: HashMap::new(),
        vars: HashMap::new(),
        int_widths: HashMap::new(),
        vertex_index,
        instance_index,
        budget: INSTRUCTION_BUDGET,
    };
    // Materialize writable globals (Output/Private).
    for (id, (storage, pointee)) in &m.globals {
        if matches!(*storage, STORAGE_CLASS_OUTPUT | STORAGE_CLASS_PRIVATE) {
            exec.vars.insert(*id, default_storage(m, *pointee)?);
        }
    }

    let words = m.words;
    let mut i = body_start;
    let mut cur_block: u32 = 0;
    loop {
        if i >= body_end {
            return Err(VertexEvalDecline::FunctionFellOffEnd);
        }
        exec.budget = exec
            .budget
            .checked_sub(1)
            .ok_or(VertexEvalDecline::MainInstructionBudgetExhausted)?;
        let word0 = words[i];
        let wc = (word0 >> 16) as usize;
        let op = (word0 & 0xffff) as u16;
        if wc == 0 || i + wc > body_end {
            return Err(VertexEvalDecline::FunctionInstructionMalformed);
        }
        let o = &words[i + 1..i + wc];
        let mut jump: Option<u32> = None;
        match op {
            OP_NOP | OP_LINE | OP_NO_LINE | OP_SELECTION_MERGE | OP_LOOP_MERGE => {}
            OP_LABEL => {
                cur_block = o[0];
            }
            OP_UNDEF => {
                exec.env.insert(o[1], undef_value(m, o[0])?);
            }
            OP_VARIABLE => {
                // Function-storage local: register storage + pointer value.
                let Type::Pointer { storage, pointee } = m.ty(o[0])?.clone() else {
                    return Err(VertexEvalDecline::FunctionVariableTypeNotPointer);
                };
                if storage != STORAGE_CLASS_FUNCTION {
                    return Err(VertexEvalDecline::FunctionVariableStorageClassInvalid { storage });
                }
                exec.vars.insert(o[1], default_storage(m, pointee)?);
                if wc >= 5 {
                    let init = exec.value(o[3])?;
                    exec.vars.insert(o[1], init);
                }
                exec.env.insert(
                    o[1],
                    Value::Ptr(PtrVal {
                        target: PtrTarget::Memory {
                            var: o[1],
                            path: Vec::new(),
                        },
                        pointee,
                    }),
                );
            }
            OP_LOAD => {
                let ptr = match exec.env.get(&o[2]) {
                    Some(Value::Ptr(p)) => p.clone(),
                    Some(_) => return Err(VertexEvalDecline::LoadSourceNotPointer),
                    None => {
                        let (storage, pointee) = m
                            .globals
                            .get(&o[2])
                            .ok_or(VertexEvalDecline::LoadPointerUnknown { id: o[2] })?;
                        exec.global_ptr(o[2], *storage, *pointee)?
                    }
                };
                let v = exec.load_ptr(&ptr)?;
                exec.env.insert(o[1], v);
                exec.note_int_width(o[1], ptr.pointee);
            }
            OP_STORE => {
                let ptr = match exec.env.get(&o[0]) {
                    Some(Value::Ptr(p)) => p.clone(),
                    Some(_) => return Err(VertexEvalDecline::StoreTargetNotPointer),
                    None => {
                        let (storage, pointee) = m
                            .globals
                            .get(&o[0])
                            .ok_or(VertexEvalDecline::StorePointerUnknown { id: o[0] })?;
                        exec.global_ptr(o[0], *storage, *pointee)?
                    }
                };
                let v = exec.value(o[1])?;
                exec.store_ptr(&ptr, v)?;
            }
            OP_ACCESS_CHAIN | OP_IN_BOUNDS_ACCESS_CHAIN => {
                let ptr = exec.access_chain(o[2], &o[3..])?;
                exec.env.insert(o[1], Value::Ptr(ptr));
            }
            OP_COPY_OBJECT => {
                let v = exec.value(o[2])?;
                exec.env.insert(o[1], v);
                if let Some(width) = exec.int_width_of(o[2]) {
                    exec.int_widths.insert(o[1], width);
                }
            }
            OP_BITCAST => {
                let src = exec.value(o[2])?;
                let v = bitcast(m, o[0], &src)?;
                exec.env.insert(o[1], v);
                exec.note_int_width(o[1], o[0]);
            }
            OP_U_CONVERT => {
                let width = int_result_width(m, o[0])?;
                let v = map_int1(&exec.value(o[2])?, |raw| Ok(mask_width(raw, width)))?;
                exec.env.insert(o[1], v);
                exec.note_int_width(o[1], o[0]);
            }
            OP_S_CONVERT => {
                let width = int_result_width(m, o[0])?;
                let src_width = exec
                    .int_width_of(o[2])
                    .ok_or(VertexEvalDecline::SignedConvertSourceWidthUnknown { id: o[2] })?;
                let src = exec.value(o[2])?;
                let v = map_int1(&src, |raw| {
                    Ok(mask_width(sign_extend(raw, src_width) as u64, width))
                })?;
                exec.env.insert(o[1], v);
                exec.note_int_width(o[1], o[0]);
            }
            OP_CONVERT_U_TO_F => {
                let v = map_int_to_float(&exec.value(o[2])?, |raw| raw as f32)?;
                exec.env.insert(o[1], v);
            }
            OP_CONVERT_S_TO_F => {
                let src_width = exec
                    .int_width_of(o[2])
                    .ok_or(VertexEvalDecline::SignedToFloatSourceWidthUnknown { id: o[2] })?;
                let v =
                    map_int_to_float(&exec.value(o[2])?, |raw| sign_extend(raw, src_width) as f32)?;
                exec.env.insert(o[1], v);
            }
            OP_CONVERT_F_TO_U => {
                let width = int_result_width(m, o[0])?;
                let v =
                    map_float_to_int(&exec.value(o[2])?, |x| mask_width(x.max(0.0) as u64, width))?;
                exec.env.insert(o[1], v);
                exec.note_int_width(o[1], o[0]);
            }
            OP_CONVERT_F_TO_S => {
                let width = int_result_width(m, o[0])?;
                let v =
                    map_float_to_int(&exec.value(o[2])?, |x| mask_width(x as i64 as u64, width))?;
                exec.env.insert(o[1], v);
                exec.note_int_width(o[1], o[0]);
            }
            OP_S_NEGATE => {
                let width = int_result_width(m, o[0])?;
                let v = map_int1(&exec.value(o[2])?, |raw| {
                    Ok(mask_width((raw as i64).wrapping_neg() as u64, width))
                })?;
                exec.env.insert(o[1], v);
                exec.note_int_width(o[1], o[0]);
            }
            OP_F_NEGATE => {
                let v = componentwise1(&exec.value(o[2])?, |x| -x)?;
                exec.env.insert(o[1], v);
            }
            OP_I_ADD
            | OP_I_SUB
            | OP_I_MUL
            | OP_U_DIV
            | OP_S_DIV
            | OP_U_MOD
            | OP_S_REM
            | OP_SHIFT_LEFT_LOGICAL
            | OP_SHIFT_RIGHT_LOGICAL
            | OP_SHIFT_RIGHT_ARITHMETIC
            | OP_BITWISE_AND
            | OP_BITWISE_OR
            | OP_BITWISE_XOR => {
                let width = int_result_width(m, o[0])?;
                let a = exec.value(o[2])?;
                let b = exec.value(o[3])?;
                let v = int_componentwise2(&a, &b, width, |x, y, w| int_binop(op, x, y, w))?;
                exec.env.insert(o[1], v);
                exec.note_int_width(o[1], o[0]);
            }
            OP_NOT => {
                let width = int_result_width(m, o[0])?;
                let v = map_int1(&exec.value(o[2])?, |raw| Ok(mask_width(!raw, width)))?;
                exec.env.insert(o[1], v);
                exec.note_int_width(o[1], o[0]);
            }
            OP_F_ADD => {
                let v = componentwise2(&exec.value(o[2])?, &exec.value(o[3])?, |x, y| x + y)?;
                exec.env.insert(o[1], v);
            }
            OP_F_SUB => {
                let v = componentwise2(&exec.value(o[2])?, &exec.value(o[3])?, |x, y| x - y)?;
                exec.env.insert(o[1], v);
            }
            OP_F_MUL => {
                let v = componentwise2(&exec.value(o[2])?, &exec.value(o[3])?, |x, y| x * y)?;
                exec.env.insert(o[1], v);
            }
            OP_F_DIV => {
                let v = componentwise2(&exec.value(o[2])?, &exec.value(o[3])?, |x, y| x / y)?;
                exec.env.insert(o[1], v);
            }
            OP_F_REM | OP_F_MOD => {
                let v = componentwise2(&exec.value(o[2])?, &exec.value(o[3])?, |x, y| x % y)?;
                exec.env.insert(o[1], v);
            }
            OP_VECTOR_TIMES_SCALAR => {
                let vec = exec.value(o[2])?;
                let Value::Float(s) = exec.value(o[3])? else {
                    return Err(VertexEvalDecline::VectorScalarTypeMismatch);
                };
                let v = componentwise1(&vec, |x| x * s)?;
                exec.env.insert(o[1], v);
            }
            OP_MATRIX_TIMES_SCALAR => {
                let mat = exec.value(o[2])?;
                let Value::Float(s) = exec.value(o[3])? else {
                    return Err(VertexEvalDecline::MatrixScalarTypeMismatch);
                };
                let v = componentwise1(&mat, |x| x * s)?;
                exec.env.insert(o[1], v);
            }
            OP_MATRIX_TIMES_VECTOR => {
                let Value::Composite(columns) = exec.value(o[2])? else {
                    return Err(VertexEvalDecline::MatrixTimesVectorMatrixNotComposite);
                };
                let weights = float_vec(&exec.value(o[3])?)?;
                if columns.len() != weights.len() {
                    return Err(VertexEvalDecline::MatrixTimesVectorColumnCountMismatch);
                }
                let mut acc: Option<Vec<f32>> = None;
                for (column, weight) in columns.iter().zip(&weights) {
                    let column = float_vec(column)?;
                    let acc = acc.get_or_insert_with(|| vec![0.0; column.len()]);
                    if acc.len() != column.len() {
                        return Err(VertexEvalDecline::MatrixTimesVectorColumnHeightMismatch);
                    }
                    for (a, c) in acc.iter_mut().zip(column) {
                        *a += c * weight;
                    }
                }
                let acc = acc.ok_or(VertexEvalDecline::MatrixTimesVectorEmptyMatrix)?;
                exec.env.insert(
                    o[1],
                    Value::Composite(acc.into_iter().map(Value::Float).collect()),
                );
            }
            OP_VECTOR_TIMES_MATRIX => {
                let row = float_vec(&exec.value(o[2])?)?;
                let Value::Composite(columns) = exec.value(o[3])? else {
                    return Err(VertexEvalDecline::VectorTimesMatrixMatrixNotComposite);
                };
                let mut out = Vec::with_capacity(columns.len());
                for column in &columns {
                    let column = float_vec(column)?;
                    if column.len() != row.len() {
                        return Err(VertexEvalDecline::VectorTimesMatrixShapeMismatch);
                    }
                    out.push(Value::Float(
                        row.iter().zip(column).map(|(a, b)| a * b).sum(),
                    ));
                }
                exec.env.insert(o[1], Value::Composite(out));
            }
            OP_MATRIX_TIMES_MATRIX => {
                let Value::Composite(a_cols) = exec.value(o[2])? else {
                    return Err(VertexEvalDecline::MatrixTimesMatrixLeftNotComposite);
                };
                let Value::Composite(b_cols) = exec.value(o[3])? else {
                    return Err(VertexEvalDecline::MatrixTimesMatrixRightNotComposite);
                };
                let mut out_cols = Vec::with_capacity(b_cols.len());
                for b_col in &b_cols {
                    let weights = float_vec(b_col)?;
                    if weights.len() != a_cols.len() {
                        return Err(VertexEvalDecline::MatrixTimesMatrixShapeMismatch);
                    }
                    let mut acc: Option<Vec<f32>> = None;
                    for (a_col, weight) in a_cols.iter().zip(&weights) {
                        let a_col = float_vec(a_col)?;
                        let acc = acc.get_or_insert_with(|| vec![0.0; a_col.len()]);
                        for (dst, c) in acc.iter_mut().zip(a_col) {
                            *dst += c * weight;
                        }
                    }
                    let acc = acc.ok_or(VertexEvalDecline::MatrixTimesMatrixEmptyMatrix)?;
                    out_cols.push(Value::Composite(
                        acc.into_iter().map(Value::Float).collect(),
                    ));
                }
                exec.env.insert(o[1], Value::Composite(out_cols));
            }
            OP_TRANSPOSE => {
                let Value::Composite(columns) = exec.value(o[2])? else {
                    return Err(VertexEvalDecline::TransposeMatrixNotComposite);
                };
                let rows: Vec<Vec<f32>> = columns
                    .iter()
                    .map(float_vec)
                    .collect::<Result<Vec<_>, _>>()?;
                let height = rows.first().map(|r| r.len()).unwrap_or(0);
                let mut out = Vec::with_capacity(height);
                for r in 0..height {
                    let mut row = Vec::with_capacity(rows.len());
                    for col in &rows {
                        row.push(Value::Float(
                            *col.get(r).ok_or(VertexEvalDecline::TransposeMatrixRagged)?,
                        ));
                    }
                    out.push(Value::Composite(row));
                }
                exec.env.insert(o[1], Value::Composite(out));
            }
            OP_DOT => {
                let a = float_vec(&exec.value(o[2])?)?;
                let b = float_vec(&exec.value(o[3])?)?;
                if a.len() != b.len() {
                    return Err(VertexEvalDecline::DotShapeMismatch);
                }
                exec.env.insert(
                    o[1],
                    Value::Float(a.iter().zip(b).map(|(x, y)| x * y).sum()),
                );
            }
            OP_COMPOSITE_CONSTRUCT => {
                // Vector construction may mix scalars and sub-vectors.
                let result_ty = m.ty(o[0])?.clone();
                let mut members = Vec::new();
                for id in &o[2..] {
                    let v = exec.value(*id)?;
                    match (&result_ty, v) {
                        (Type::Vector { .. }, Value::Composite(sub)) => members.extend(sub),
                        (_, v) => members.push(v),
                    }
                }
                exec.env.insert(o[1], Value::Composite(members));
            }
            OP_COMPOSITE_EXTRACT => {
                let mut value = exec.value(o[2])?;
                for idx in &o[3..] {
                    let Value::Composite(members) = value else {
                        return Err(VertexEvalDecline::CompositeExtractFromScalar);
                    };
                    value = members.get(*idx as usize).cloned().ok_or(
                        VertexEvalDecline::CompositeExtractIndexOutOfRange { index: *idx },
                    )?;
                }
                exec.env.insert(o[1], value);
            }
            OP_COMPOSITE_INSERT => {
                let object = exec.value(o[2])?;
                let mut composite = exec.value(o[3])?;
                {
                    let mut slot = &mut composite;
                    for idx in &o[4..] {
                        let Value::Composite(members) = slot else {
                            return Err(VertexEvalDecline::CompositeInsertIntoScalar);
                        };
                        slot = members.get_mut(*idx as usize).ok_or(
                            VertexEvalDecline::CompositeInsertIndexOutOfRange { index: *idx },
                        )?;
                    }
                    *slot = object;
                }
                exec.env.insert(o[1], composite);
            }
            OP_VECTOR_SHUFFLE => {
                let a = match exec.value(o[2])? {
                    Value::Composite(m) => m,
                    Value::Undef => Vec::new(),
                    _ => return Err(VertexEvalDecline::VectorShuffleLeftNotVector),
                };
                let b = match exec.value(o[3])? {
                    Value::Composite(m) => m,
                    Value::Undef => Vec::new(),
                    _ => return Err(VertexEvalDecline::VectorShuffleRightNotVector),
                };
                let mut out = Vec::with_capacity(o.len() - 4);
                for sel in &o[4..] {
                    if *sel == 0xffff_ffff {
                        out.push(Value::Undef);
                    } else {
                        let sel = *sel as usize;
                        let v = if sel < a.len() {
                            a[sel].clone()
                        } else if sel - a.len() < b.len() {
                            b[sel - a.len()].clone()
                        } else if a.is_empty() || b.is_empty() {
                            // Source operand was OpUndef (component count
                            // unknown here) — the selected lane is undef.
                            Value::Undef
                        } else {
                            return Err(VertexEvalDecline::VectorShuffleIndexOutOfRange {
                                index: sel as u32,
                            });
                        };
                        out.push(v);
                    }
                }
                exec.env.insert(o[1], Value::Composite(out));
            }
            OP_SELECT => {
                let cond = exec.value(o[2])?;
                let a = exec.value(o[3])?;
                let b = exec.value(o[4])?;
                let v = match cond {
                    Value::Bool(true) => a,
                    Value::Bool(false) => b,
                    Value::Composite(conds) => {
                        let (Value::Composite(ma), Value::Composite(mb)) = (a, b) else {
                            return Err(VertexEvalDecline::SelectValuesNotComposite);
                        };
                        if conds.len() != ma.len() || conds.len() != mb.len() {
                            return Err(VertexEvalDecline::SelectVectorLengthMismatch);
                        }
                        Value::Composite(
                            conds
                                .iter()
                                .zip(ma.into_iter().zip(mb))
                                .map(|(c, (x, y))| match c {
                                    Value::Bool(true) => Ok(x),
                                    Value::Bool(false) => Ok(y),
                                    _ => Err(VertexEvalDecline::SelectVectorConditionNotBool),
                                })
                                .collect::<Result<Vec<_>, _>>()?,
                        )
                    }
                    _ => return Err(VertexEvalDecline::SelectConditionNotBool),
                };
                exec.env.insert(o[1], v);
            }
            OP_I_EQUAL
            | OP_I_NOT_EQUAL
            | OP_U_GREATER_THAN
            | OP_S_GREATER_THAN
            | OP_U_GREATER_THAN_EQUAL
            | OP_S_GREATER_THAN_EQUAL
            | OP_U_LESS_THAN
            | OP_S_LESS_THAN
            | OP_U_LESS_THAN_EQUAL
            | OP_S_LESS_THAN_EQUAL => {
                let a = exec.value(o[2])?;
                let b = exec.value(o[3])?;
                // Unsigned/equality compares work on the masked raw bits; the
                // signed forms need the true operand width to fail closed.
                let signed = matches!(
                    op,
                    OP_S_GREATER_THAN
                        | OP_S_GREATER_THAN_EQUAL
                        | OP_S_LESS_THAN
                        | OP_S_LESS_THAN_EQUAL
                );
                let width = if signed {
                    exec.int_width_of(o[2])
                        .or_else(|| exec.int_width_of(o[3]))
                        .ok_or(VertexEvalDecline::SignedCompareWidthUnknown)?
                } else {
                    64
                };
                let v = int_compare(op, &a, &b, width)?;
                exec.env.insert(o[1], v);
            }
            OP_F_ORD_EQUAL
            | OP_F_ORD_NOT_EQUAL
            | OP_F_ORD_LESS_THAN
            | OP_F_ORD_GREATER_THAN
            | OP_F_ORD_LESS_THAN_EQUAL
            | OP_F_ORD_GREATER_THAN_EQUAL => {
                let a = exec.value(o[2])?;
                let b = exec.value(o[3])?;
                let v = float_compare(op, &a, &b)?;
                exec.env.insert(o[1], v);
            }
            OP_LOGICAL_AND | OP_LOGICAL_OR | OP_LOGICAL_EQUAL | OP_LOGICAL_NOT_EQUAL => {
                let a = exec.value(o[2])?;
                let b = exec.value(o[3])?;
                let v = bool_binop(op, &a, &b)?;
                exec.env.insert(o[1], v);
            }
            OP_LOGICAL_NOT => {
                let v = match exec.value(o[2])? {
                    Value::Bool(b) => Value::Bool(!b),
                    Value::Composite(m) => Value::Composite(
                        m.into_iter()
                            .map(|c| match c {
                                Value::Bool(b) => Ok(Value::Bool(!b)),
                                _ => Err(VertexEvalDecline::LogicalNotVectorMemberNotBool),
                            })
                            .collect::<Result<Vec<_>, _>>()?,
                    ),
                    _ => return Err(VertexEvalDecline::LogicalNotOperandNotBool),
                };
                exec.env.insert(o[1], v);
            }
            OP_EXT_INST => {
                if Some(o[2]) != m.glsl_ext {
                    return Err(VertexEvalDecline::ExtInstSetUnknown { set_id: o[2] });
                }
                let v = eval_glsl_ext(&exec, o[3], &o[4..])?;
                exec.env.insert(o[1], v);
            }
            OP_PHI => {
                // Handled at block entry (two-phase); reaching one mid-stream
                // means the block-entry scan missed it.
                return Err(VertexEvalDecline::PhiOutsideBlockEntry);
            }
            OP_BRANCH => {
                jump = Some(o[0]);
            }
            OP_BRANCH_CONDITIONAL => {
                let Value::Bool(cond) = exec.value(o[0])? else {
                    return Err(VertexEvalDecline::BranchConditionNotBool);
                };
                jump = Some(if cond { o[1] } else { o[2] });
            }
            OP_SWITCH => {
                let selector = exec.int_operand(o[0])?;
                let width = exec
                    .int_width_of(o[0])
                    .ok_or(VertexEvalDecline::SwitchSelectorWidthUnknown { id: o[0] })?;
                let selector = mask_width(selector, width);
                let mut target = o[1];
                let literal_words: usize = if width > 32 { 2 } else { 1 };
                let pairs = &o[2..];
                let stride = literal_words + 1;
                if !pairs.len().is_multiple_of(stride) {
                    return Err(VertexEvalDecline::SwitchOperandsMalformed);
                }
                for pair in pairs.chunks_exact(stride) {
                    let literal = if literal_words == 2 {
                        (pair[0] as u64) | ((pair[1] as u64) << 32)
                    } else {
                        pair[0] as u64
                    };
                    if mask_width(literal, width) == selector {
                        target = pair[literal_words];
                        break;
                    }
                }
                jump = Some(target);
            }
            OP_RETURN => {
                return extract_position(&exec, position);
            }
            OP_RETURN_VALUE | OP_UNREACHABLE => {
                return Err(VertexEvalDecline::UnexpectedTerminator { opcode: op });
            }
            other => {
                return Err(VertexEvalDecline::OpcodeUnsupported { opcode: other });
            }
        }
        if let Some(target) = jump {
            let target_index = *m
                .labels
                .get(&target)
                .ok_or(VertexEvalDecline::BranchLabelUnknown { label: target })?;
            // Two-phase phi evaluation at the target block entry.
            let prev_block = cur_block;
            cur_block = target;
            let mut j = target_index;
            // Skip the OpLabel itself.
            let label_wc = (words[j] >> 16) as usize;
            j += label_wc;
            let mut phi_updates: Vec<(u32, Value)> = Vec::new();
            while j < body_end {
                let w0 = words[j];
                let wc2 = (w0 >> 16) as usize;
                let op2 = (w0 & 0xffff) as u16;
                if wc2 == 0 || j + wc2 > body_end {
                    return Err(VertexEvalDecline::PhiInstructionMalformed);
                }
                if op2 == OP_LINE || op2 == OP_NO_LINE {
                    j += wc2;
                    continue;
                }
                if op2 != OP_PHI {
                    break;
                }
                let po = &words[j + 1..j + wc2];
                let mut chosen: Option<Value> = None;
                for pair in po[2..].chunks_exact(2) {
                    if pair[1] == prev_block {
                        chosen = Some(exec.value(pair[0])?);
                        break;
                    }
                }
                let chosen = chosen.ok_or(VertexEvalDecline::PhiPredecessorMissing {
                    predecessor: prev_block,
                })?;
                exec.note_int_width(po[1], po[0]);
                phi_updates.push((po[1], chosen));
                exec.budget = exec
                    .budget
                    .checked_sub(1)
                    .ok_or(VertexEvalDecline::PhiInstructionBudgetExhausted)?;
                j += wc2;
            }
            for (id, v) in phi_updates {
                exec.env.insert(id, v);
            }
            i = j;
            continue;
        }
        i += wc;
    }
}

fn extract_position(
    exec: &Exec<'_, '_, '_>,
    position: PositionOutput,
) -> Result<[f32; 4], VertexEvalDecline> {
    let value = match position {
        PositionOutput::Variable(id) => exec
            .vars
            .get(&id)
            .cloned()
            .ok_or(VertexEvalDecline::PositionVariableNeverStored { variable: id })?,
        PositionOutput::Member(id, member) => {
            let Some(Value::Composite(members)) = exec.vars.get(&id) else {
                return Err(VertexEvalDecline::PositionStructNeverStored { variable: id });
            };
            members.get(member as usize).cloned().ok_or(
                VertexEvalDecline::PositionMemberNeverStored {
                    variable: id,
                    member,
                },
            )?
        }
    };
    let Value::Composite(members) = value else {
        return Err(VertexEvalDecline::PositionValueNotComposite);
    };
    if members.len() != 4 {
        return Err(VertexEvalDecline::PositionVectorLengthInvalid { len: members.len() });
    }
    let mut out = [0f32; 4];
    for (component, (dst, member)) in out.iter_mut().zip(&members).enumerate() {
        match member {
            Value::Float(x) if x.is_finite() => *dst = *x,
            Value::Float(_) => {
                return Err(VertexEvalDecline::PositionComponentNotFinite { component })
            }
            _ => return Err(VertexEvalDecline::PositionComponentUndefined { component }),
        }
    }
    Ok(out)
}

fn map_int1(
    v: &Value,
    f: impl Fn(u64) -> Result<u64, VertexEvalDecline> + Copy,
) -> Result<Value, VertexEvalDecline> {
    match v {
        Value::Int(raw) => Ok(Value::Int(f(*raw)?)),
        Value::Composite(m) => Ok(Value::Composite(
            m.iter()
                .map(|c| map_int1(c, f))
                .collect::<Result<Vec<_>, _>>()?,
        )),
        _ => Err(VertexEvalDecline::MapIntOperandTypeMismatch),
    }
}

fn map_int_to_float(v: &Value, f: impl Fn(u64) -> f32 + Copy) -> Result<Value, VertexEvalDecline> {
    match v {
        Value::Int(raw) => Ok(Value::Float(f(*raw))),
        Value::Composite(m) => Ok(Value::Composite(
            m.iter()
                .map(|c| map_int_to_float(c, f))
                .collect::<Result<Vec<_>, _>>()?,
        )),
        _ => Err(VertexEvalDecline::MapIntToFloatOperandTypeMismatch),
    }
}

fn map_float_to_int(v: &Value, f: impl Fn(f32) -> u64 + Copy) -> Result<Value, VertexEvalDecline> {
    match v {
        Value::Float(x) => Ok(Value::Int(f(*x))),
        Value::Composite(m) => Ok(Value::Composite(
            m.iter()
                .map(|c| map_float_to_int(c, f))
                .collect::<Result<Vec<_>, _>>()?,
        )),
        _ => Err(VertexEvalDecline::MapFloatToIntOperandTypeMismatch),
    }
}

fn int_binop(op: u16, x: u64, y: u64, width: u32) -> Result<u64, VertexEvalDecline> {
    Ok(match op {
        OP_I_ADD => x.wrapping_add(y),
        OP_I_SUB => x.wrapping_sub(y),
        OP_I_MUL => x.wrapping_mul(y),
        OP_U_DIV => {
            if y == 0 {
                return Err(VertexEvalDecline::UnsignedDivisionByZero);
            }
            x / y
        }
        OP_S_DIV => {
            let sy = sign_extend(y, width);
            if sy == 0 {
                return Err(VertexEvalDecline::SignedDivisionByZero);
            }
            sign_extend(x, width).wrapping_div(sy) as u64
        }
        OP_U_MOD => {
            if y == 0 {
                return Err(VertexEvalDecline::UnsignedModuloByZero);
            }
            x % y
        }
        OP_S_REM => {
            let sy = sign_extend(y, width);
            if sy == 0 {
                return Err(VertexEvalDecline::SignedRemainderByZero);
            }
            sign_extend(x, width).wrapping_rem(sy) as u64
        }
        OP_SHIFT_LEFT_LOGICAL => x.wrapping_shl(y as u32),
        OP_SHIFT_RIGHT_LOGICAL => x.wrapping_shr(y as u32),
        OP_SHIFT_RIGHT_ARITHMETIC => (sign_extend(x, width) >> (y as u32 % 64)) as u64,
        OP_BITWISE_AND => x & y,
        OP_BITWISE_OR => x | y,
        OP_BITWISE_XOR => x ^ y,
        _ => return Err(VertexEvalDecline::IntegerBinaryOpcodeUnsupported { opcode: op }),
    })
}

fn int_compare(op: u16, a: &Value, b: &Value, width: u32) -> Result<Value, VertexEvalDecline> {
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => {
            let (ux, uy) = (*x, *y);
            let (sx, sy) = (sign_extend(*x, width), sign_extend(*y, width));
            let r = match op {
                OP_I_EQUAL => ux == uy,
                OP_I_NOT_EQUAL => ux != uy,
                OP_U_GREATER_THAN => ux > uy,
                OP_U_GREATER_THAN_EQUAL => ux >= uy,
                OP_U_LESS_THAN => ux < uy,
                OP_U_LESS_THAN_EQUAL => ux <= uy,
                OP_S_GREATER_THAN => sx > sy,
                OP_S_GREATER_THAN_EQUAL => sx >= sy,
                OP_S_LESS_THAN => sx < sy,
                OP_S_LESS_THAN_EQUAL => sx <= sy,
                _ => return Err(VertexEvalDecline::IntegerCompareOpcodeUnsupported { opcode: op }),
            };
            Ok(Value::Bool(r))
        }
        (Value::Composite(ma), Value::Composite(mb)) if ma.len() == mb.len() => {
            Ok(Value::Composite(
                ma.iter()
                    .zip(mb)
                    .map(|(x, y)| int_compare(op, x, y, width))
                    .collect::<Result<Vec<_>, _>>()?,
            ))
        }
        _ => Err(VertexEvalDecline::IntegerCompareShapeMismatch),
    }
}

fn float_compare(op: u16, a: &Value, b: &Value) -> Result<Value, VertexEvalDecline> {
    match (a, b) {
        (Value::Float(x), Value::Float(y)) => {
            let r = match op {
                OP_F_ORD_EQUAL => x == y,
                OP_F_ORD_NOT_EQUAL => x != y && !x.is_nan() && !y.is_nan(),
                OP_F_ORD_LESS_THAN => x < y,
                OP_F_ORD_GREATER_THAN => x > y,
                OP_F_ORD_LESS_THAN_EQUAL => x <= y,
                OP_F_ORD_GREATER_THAN_EQUAL => x >= y,
                _ => return Err(VertexEvalDecline::FloatCompareOpcodeUnsupported { opcode: op }),
            };
            Ok(Value::Bool(r))
        }
        (Value::Composite(ma), Value::Composite(mb)) if ma.len() == mb.len() => {
            Ok(Value::Composite(
                ma.iter()
                    .zip(mb)
                    .map(|(x, y)| float_compare(op, x, y))
                    .collect::<Result<Vec<_>, _>>()?,
            ))
        }
        _ => Err(VertexEvalDecline::FloatCompareShapeMismatch),
    }
}

fn bool_binop(op: u16, a: &Value, b: &Value) -> Result<Value, VertexEvalDecline> {
    match (a, b) {
        (Value::Bool(x), Value::Bool(y)) => Ok(Value::Bool(match op {
            OP_LOGICAL_AND => *x && *y,
            OP_LOGICAL_OR => *x || *y,
            OP_LOGICAL_EQUAL => x == y,
            OP_LOGICAL_NOT_EQUAL => x != y,
            _ => return Err(VertexEvalDecline::BooleanBinaryOpcodeUnsupported { opcode: op }),
        })),
        (Value::Composite(ma), Value::Composite(mb)) if ma.len() == mb.len() => {
            Ok(Value::Composite(
                ma.iter()
                    .zip(mb)
                    .map(|(x, y)| bool_binop(op, x, y))
                    .collect::<Result<Vec<_>, _>>()?,
            ))
        }
        _ => Err(VertexEvalDecline::BooleanBinaryShapeMismatch),
    }
}

fn bitcast(m: &Module<'_>, result_type: u32, src: &Value) -> Result<Value, VertexEvalDecline> {
    match (m.ty(result_type)?, src) {
        (Type::Float { width: 32 }, Value::Int(raw)) => {
            Ok(Value::Float(f32::from_bits(*raw as u32)))
        }
        (Type::Int { width }, Value::Float(x)) => {
            Ok(Value::Int(mask_width(u64::from(x.to_bits()), *width)))
        }
        (Type::Int { width }, Value::Int(raw)) => Ok(Value::Int(mask_width(*raw, *width))),
        (Type::Vector { elem, count }, Value::Composite(members))
            if members.len() == *count as usize =>
        {
            Ok(Value::Composite(
                members
                    .iter()
                    .map(|member| bitcast_elem(m, *elem, member))
                    .collect::<Result<Vec<_>, _>>()?,
            ))
        }
        _ => Err(VertexEvalDecline::BitcastTypeUnsupported),
    }
}

fn bitcast_elem(m: &Module<'_>, elem_type: u32, src: &Value) -> Result<Value, VertexEvalDecline> {
    match (m.ty(elem_type)?, src) {
        (Type::Float { width: 32 }, Value::Int(raw)) => {
            Ok(Value::Float(f32::from_bits(*raw as u32)))
        }
        (Type::Int { width }, Value::Float(x)) => {
            Ok(Value::Int(mask_width(u64::from(x.to_bits()), *width)))
        }
        (Type::Int { width }, Value::Int(raw)) => Ok(Value::Int(mask_width(*raw, *width))),
        _ => Err(VertexEvalDecline::BitcastElementTypeUnsupported),
    }
}

fn eval_glsl_ext(
    exec: &Exec<'_, '_, '_>,
    inst: u32,
    args: &[u32],
) -> Result<Value, VertexEvalDecline> {
    let arg = |i: usize| -> Result<Value, VertexEvalDecline> {
        exec.value(
            *args
                .get(i)
                .ok_or(VertexEvalDecline::ExtArgumentMissing { index: i })?,
        )
    };
    match inst {
        GLSL_FABS => componentwise1(&arg(0)?, f32::abs),
        GLSL_FLOOR => componentwise1(&arg(0)?, f32::floor),
        GLSL_CEIL => componentwise1(&arg(0)?, f32::ceil),
        GLSL_FRACT => componentwise1(&arg(0)?, |x| x - x.floor()),
        GLSL_SIN => componentwise1(&arg(0)?, f32::sin),
        GLSL_COS => componentwise1(&arg(0)?, f32::cos),
        GLSL_SQRT => componentwise1(&arg(0)?, f32::sqrt),
        GLSL_INVERSE_SQRT => componentwise1(&arg(0)?, |x| 1.0 / x.sqrt()),
        GLSL_POW => componentwise2(&arg(0)?, &arg(1)?, f32::powf),
        GLSL_FMIN => componentwise2(&arg(0)?, &arg(1)?, f32::min),
        GLSL_FMAX => componentwise2(&arg(0)?, &arg(1)?, f32::max),
        GLSL_STEP => componentwise2(
            &arg(0)?,
            &arg(1)?,
            |edge, x| if x < edge { 0.0 } else { 1.0 },
        ),
        GLSL_FCLAMP => componentwise3(&arg(0)?, &arg(1)?, &arg(2)?, |x, lo, hi| x.max(lo).min(hi)),
        GLSL_FMIX => componentwise3(&arg(0)?, &arg(1)?, &arg(2)?, |x, y, a| {
            x * (1.0 - a) + y * a
        }),
        GLSL_FMA => componentwise3(&arg(0)?, &arg(1)?, &arg(2)?, |a, b, c| a.mul_add(b, c)),
        GLSL_UNPACK_UNORM_4X8 => {
            let Value::Int(raw) = arg(0)? else {
                return Err(VertexEvalDecline::UnpackUnormOperandNotInt);
            };
            let raw = raw as u32;
            Ok(Value::Composite(
                (0..4)
                    .map(|i| Value::Float(((raw >> (8 * i)) & 0xff) as f32 / 255.0))
                    .collect(),
            ))
        }
        other => Err(VertexEvalDecline::ExtendedOpcodeUnsupported { opcode: other }),
    }
}

/// Test fixtures shared with the coverage-proof tests in `metal_draw`.
///
/// The module mirrors the live shader-pulled compositor vertex shader's
/// structure (rspirv m2v output): a config byte selects the vertex record
/// layout via `OpSwitch`, records are pulled from a storage buffer by
/// `VertexIndex`, positions run through matrix columns from another buffer,
/// and the blocks merge through `OpPhi` before `OpStore` to `Position`.
#[cfg(test)]
pub(crate) mod fixtures {
    use super::*;

    struct Asm {
        words: Vec<u32>,
    }

    impl Asm {
        fn ins(&mut self, op: u16, operands: &[u32]) {
            self.words
                .push((((operands.len() + 1) as u32) << 16) | op as u32);
            self.words.extend_from_slice(operands);
        }
    }

    // Stable ids for the synthetic module.
    const EXT: u32 = 1;
    const T_VOID: u32 = 2;
    const T_FN: u32 = 3;
    const T_F32: u32 = 4;
    const T_V4: u32 = 5;
    const T_U32: u32 = 6;
    const T_U8: u32 = 7;
    const T_U64: u32 = 8;
    const T_ARR_U32: u32 = 11;
    const T_SB_RECORDS: u32 = 12;
    const P_SB_RECORDS: u32 = 13;
    const V_RECORDS: u32 = 14;
    const T_SB_CFG: u32 = 15;
    const P_SB_CFG: u32 = 16;
    const V_CFG: u32 = 17;
    const T_ARR_V4: u32 = 18;
    const T_SB_MATRIX: u32 = 19;
    const P_SB_MATRIX: u32 = 20;
    const V_MATRIX: u32 = 21;
    const P_IN_U32: u32 = 22;
    const V_VERTEX_INDEX: u32 = 23;
    const P_OUT_V4: u32 = 24;
    const V_POSITION: u32 = 25;
    const P_SB_U8: u32 = 26;
    const P_SB_U32: u32 = 27;
    const P_SB_V4: u32 = 28;
    const C_U32_0: u32 = 30;
    const C_U32_1: u32 = 31;
    const C_U32_2: u32 = 32;
    const C_U32_3: u32 = 33;
    const C_U32_4: u32 = 34;
    const C_F32_0: u32 = 35;
    const C_F32_1: u32 = 36;
    const C_V4_0001: u32 = 37;
    const C_F32_MAX: u32 = 38;
    const C_V4_MAX: u32 = 39;
    const UNDEF_V4: u32 = 40;
    const FN_MAIN: u32 = 50;
    const L_ENTRY: u32 = 51;
    const L_CASE2: u32 = 60;
    const L_DEFAULT: u32 = 78;
    const L_MERGE: u32 = 80;
    const BOUND: u32 = 120;

    /// Assemble the shader-pulled quad vertex module.
    ///
    /// `unsupported_tail` appends an unsupported opcode inside the merge block
    /// so fail-closed behavior can be locked without changing anything else.
    pub(crate) fn shader_pulled_quad_module(unsupported_tail: bool) -> Vec<u32> {
        let mut a = Asm {
            words: vec![0x0723_0203, 0x0001_0400, 0, BOUND, 0],
        };
        // OpExtInstImport "GLSL.std.450"
        let name = b"GLSL.std.450\0\0\0\0";
        let mut ext = vec![EXT];
        for chunk in name.chunks_exact(4) {
            ext.push(u32::from_le_bytes(chunk.try_into().unwrap()));
        }
        a.ins(OP_EXT_INST_IMPORT, &ext);
        // OpEntryPoint Vertex %FN_MAIN "main" %V_VERTEX_INDEX %V_POSITION
        a.ins(
            OP_ENTRY_POINT,
            &[
                0,
                FN_MAIN,
                u32::from_le_bytes(*b"main"),
                0,
                V_VERTEX_INDEX,
                V_POSITION,
            ],
        );
        // Decorations.
        a.ins(
            OP_DECORATE,
            &[V_VERTEX_INDEX, DECORATION_BUILT_IN, BUILT_IN_VERTEX_INDEX],
        );
        a.ins(
            OP_DECORATE,
            &[V_POSITION, DECORATION_BUILT_IN, BUILT_IN_POSITION],
        );
        a.ins(OP_DECORATE, &[V_CFG, DECORATION_BINDING, 0]);
        a.ins(OP_DECORATE, &[V_RECORDS, DECORATION_BINDING, 1]);
        a.ins(OP_DECORATE, &[V_MATRIX, DECORATION_BINDING, 2]);
        a.ins(OP_DECORATE, &[T_ARR_U32, DECORATION_ARRAY_STRIDE, 4]);
        a.ins(OP_DECORATE, &[T_ARR_V4, DECORATION_ARRAY_STRIDE, 16]);
        a.ins(OP_MEMBER_DECORATE, &[T_SB_RECORDS, 0, DECORATION_OFFSET, 0]);
        a.ins(OP_MEMBER_DECORATE, &[T_SB_CFG, 0, DECORATION_OFFSET, 0]);
        a.ins(OP_MEMBER_DECORATE, &[T_SB_MATRIX, 0, DECORATION_OFFSET, 0]);
        // Types.
        a.ins(OP_TYPE_VOID, &[T_VOID]);
        a.ins(33, &[T_FN, T_VOID]); // OpTypeFunction
        a.ins(OP_TYPE_FLOAT, &[T_F32, 32]);
        a.ins(OP_TYPE_VECTOR, &[T_V4, T_F32, 4]);
        a.ins(OP_TYPE_INT, &[T_U32, 32, 0]);
        a.ins(OP_TYPE_INT, &[T_U8, 8, 0]);
        a.ins(OP_TYPE_INT, &[T_U64, 64, 0]);
        a.ins(OP_TYPE_RUNTIME_ARRAY, &[T_ARR_U32, T_U32]);
        a.ins(OP_TYPE_STRUCT, &[T_SB_RECORDS, T_ARR_U32]);
        a.ins(
            OP_TYPE_POINTER,
            &[P_SB_RECORDS, STORAGE_CLASS_STORAGE_BUFFER, T_SB_RECORDS],
        );
        a.ins(OP_TYPE_STRUCT, &[T_SB_CFG, T_U8]);
        a.ins(
            OP_TYPE_POINTER,
            &[P_SB_CFG, STORAGE_CLASS_STORAGE_BUFFER, T_SB_CFG],
        );
        // Constants used by types must precede them.
        a.ins(OP_CONSTANT, &[T_U32, C_U32_4, 4]);
        a.ins(OP_TYPE_ARRAY, &[T_ARR_V4, T_V4, C_U32_4]);
        a.ins(OP_TYPE_STRUCT, &[T_SB_MATRIX, T_ARR_V4]);
        a.ins(
            OP_TYPE_POINTER,
            &[P_SB_MATRIX, STORAGE_CLASS_STORAGE_BUFFER, T_SB_MATRIX],
        );
        a.ins(OP_TYPE_POINTER, &[P_IN_U32, STORAGE_CLASS_INPUT, T_U32]);
        a.ins(OP_TYPE_POINTER, &[P_OUT_V4, STORAGE_CLASS_OUTPUT, T_V4]);
        a.ins(
            OP_TYPE_POINTER,
            &[P_SB_U8, STORAGE_CLASS_STORAGE_BUFFER, T_U8],
        );
        a.ins(
            OP_TYPE_POINTER,
            &[P_SB_U32, STORAGE_CLASS_STORAGE_BUFFER, T_U32],
        );
        a.ins(
            OP_TYPE_POINTER,
            &[P_SB_V4, STORAGE_CLASS_STORAGE_BUFFER, T_V4],
        );
        // Scalar/composite constants.
        a.ins(OP_CONSTANT, &[T_U32, C_U32_0, 0]);
        a.ins(OP_CONSTANT, &[T_U32, C_U32_1, 1]);
        a.ins(OP_CONSTANT, &[T_U32, C_U32_2, 2]);
        a.ins(OP_CONSTANT, &[T_U32, C_U32_3, 3]);
        a.ins(OP_CONSTANT, &[T_F32, C_F32_0, 0f32.to_bits()]);
        a.ins(OP_CONSTANT, &[T_F32, C_F32_1, 1f32.to_bits()]);
        a.ins(
            OP_CONSTANT_COMPOSITE,
            &[T_V4, C_V4_0001, C_F32_0, C_F32_0, C_F32_0, C_F32_1],
        );
        a.ins(OP_CONSTANT, &[T_F32, C_F32_MAX, f32::MAX.to_bits()]);
        a.ins(
            OP_CONSTANT_COMPOSITE,
            &[T_V4, C_V4_MAX, C_F32_MAX, C_F32_MAX, C_F32_MAX, C_F32_MAX],
        );
        a.ins(OP_UNDEF, &[T_V4, UNDEF_V4]);
        // Global variables.
        a.ins(
            OP_VARIABLE,
            &[P_SB_CFG, V_CFG, STORAGE_CLASS_STORAGE_BUFFER],
        );
        a.ins(
            OP_VARIABLE,
            &[P_SB_RECORDS, V_RECORDS, STORAGE_CLASS_STORAGE_BUFFER],
        );
        a.ins(
            OP_VARIABLE,
            &[P_SB_MATRIX, V_MATRIX, STORAGE_CLASS_STORAGE_BUFFER],
        );
        a.ins(
            OP_VARIABLE,
            &[P_IN_U32, V_VERTEX_INDEX, STORAGE_CLASS_INPUT],
        );
        a.ins(OP_VARIABLE, &[P_OUT_V4, V_POSITION, STORAGE_CLASS_OUTPUT]);
        // Function.
        a.ins(OP_FUNCTION, &[T_VOID, FN_MAIN, 0, T_FN]);
        a.ins(OP_LABEL, &[L_ENTRY]);
        a.ins(OP_LOAD, &[T_U32, 52, V_VERTEX_INDEX]);
        a.ins(OP_IN_BOUNDS_ACCESS_CHAIN, &[P_SB_U8, 53, V_CFG, C_U32_0]);
        a.ins(OP_LOAD, &[T_U8, 54, 53]);
        a.ins(OP_SELECTION_MERGE, &[L_MERGE, 0]);
        a.ins(OP_SWITCH, &[54, L_DEFAULT, 2, L_CASE2]);
        // Case 2: pull (x, y) from record stride 4 words, unpack a color.
        a.ins(OP_LABEL, &[L_CASE2]);
        a.ins(OP_U_CONVERT, &[T_U64, 61, 52]);
        a.ins(OP_U_CONVERT, &[T_U32, 62, 61]);
        a.ins(OP_I_MUL, &[T_U32, 63, 62, C_U32_4]);
        a.ins(OP_I_ADD, &[T_U32, 64, 63, C_U32_0]);
        a.ins(
            OP_IN_BOUNDS_ACCESS_CHAIN,
            &[P_SB_U32, 65, V_RECORDS, C_U32_0, 64],
        );
        a.ins(OP_LOAD, &[T_U32, 66, 65]);
        a.ins(OP_BITCAST, &[T_F32, 67, 66]);
        a.ins(OP_I_ADD, &[T_U32, 68, 63, C_U32_1]);
        a.ins(
            OP_IN_BOUNDS_ACCESS_CHAIN,
            &[P_SB_U32, 69, V_RECORDS, C_U32_0, 68],
        );
        a.ins(OP_LOAD, &[T_U32, 70, 69]);
        a.ins(OP_BITCAST, &[T_F32, 71, 70]);
        a.ins(OP_COMPOSITE_INSERT, &[T_V4, 72, 67, C_V4_0001, 0]);
        a.ins(OP_COMPOSITE_INSERT, &[T_V4, 73, 71, 72, 1]);
        a.ins(OP_I_ADD, &[T_U32, 74, 63, C_U32_3]);
        a.ins(
            OP_IN_BOUNDS_ACCESS_CHAIN,
            &[P_SB_U32, 75, V_RECORDS, C_U32_0, 74],
        );
        a.ins(OP_LOAD, &[T_U32, 76, 75]);
        a.ins(OP_EXT_INST, &[T_V4, 77, EXT, GLSL_UNPACK_UNORM_4X8, 76]);
        a.ins(OP_BRANCH, &[L_MERGE]);
        // Default: sentinel max position.
        a.ins(OP_LABEL, &[L_DEFAULT]);
        a.ins(OP_BRANCH, &[L_MERGE]);
        // Merge: phi, matrix-column transform, store Position.
        a.ins(OP_LABEL, &[L_MERGE]);
        a.ins(OP_PHI, &[T_V4, 81, 73, L_CASE2, C_V4_MAX, L_DEFAULT]);
        let mut acc = 0u32;
        for (i, column_index) in [C_U32_0, C_U32_1, C_U32_2, C_U32_3].iter().enumerate() {
            let chain = 82 + (i as u32) * 5;
            let col = chain + 1;
            let splat = chain + 2;
            let mul = chain + 3;
            let add = chain + 4;
            a.ins(
                OP_IN_BOUNDS_ACCESS_CHAIN,
                &[P_SB_V4, chain, V_MATRIX, C_U32_0, *column_index],
            );
            a.ins(OP_LOAD, &[T_V4, col, chain]);
            let lane = i as u32;
            a.ins(
                OP_VECTOR_SHUFFLE,
                &[T_V4, splat, 81, UNDEF_V4, lane, lane, lane, lane],
            );
            a.ins(OP_F_MUL, &[T_V4, mul, col, splat]);
            if i == 0 {
                acc = mul;
            } else {
                a.ins(OP_F_ADD, &[T_V4, add, acc, mul]);
                acc = add;
            }
        }
        if unsupported_tail {
            // OpIAddCarry (149) — outside the supported surface.
            a.ins(149, &[T_V4, 110, acc, acc]);
        }
        a.ins(OP_STORE, &[V_POSITION, acc]);
        a.ins(OP_RETURN, &[]);
        a.ins(OP_FUNCTION_END, &[]);
        a.words
    }

    /// Quad record bytes (mode-2 layout): 4 pixel-space corners covering
    /// `width` x `height`, stride 4 words (x, y, extra, packed color).
    pub(crate) fn quad_record_bytes(width: f32, height: f32) -> Vec<u8> {
        let corners = [(0f32, 0f32), (width, 0f32), (width, height), (0f32, height)];
        let mut bytes = Vec::new();
        for (x, y) in corners {
            bytes.extend_from_slice(&x.to_le_bytes());
            bytes.extend_from_slice(&y.to_le_bytes());
            bytes.extend_from_slice(&0u32.to_le_bytes());
            bytes.extend_from_slice(&0xff33_66ccu32.to_le_bytes());
        }
        bytes
    }

    /// Column-major ortho transform: pixel space → NDC.
    pub(crate) fn ortho_matrix_bytes(width: f32, height: f32) -> Vec<u8> {
        let columns: [[f32; 4]; 4] = [
            [2.0 / width, 0.0, 0.0, 0.0],
            [0.0, 2.0 / height, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [-1.0, -1.0, 0.0, 1.0],
        ];
        columns
            .iter()
            .flat_map(|c| c.iter().flat_map(|v| v.to_le_bytes()))
            .collect()
    }

    /// Config buffer selecting record layout `mode`.
    pub(crate) fn config_bytes(mode: u8) -> Vec<u8> {
        vec![mode, 0, 0, 0]
    }
}

#[cfg(test)]
mod tests {
    use super::fixtures::*;
    use super::*;

    fn all_declines() -> Vec<VertexEvalDecline> {
        use VertexEvalDecline as D;
        vec![
            D::MalformedHeader,
            D::ModuleInstructionMalformed,
            D::ConstantTypeMissing,
            D::ConstantTypeUnsupported,
            D::CompositeConstantForwardReference,
            D::GlobalVariableTypeMissing,
            D::GlobalVariableTypeNotPointer,
            D::VertexEntryPointMissing,
            D::EntryFunctionBodyMissingDuringParse,
            D::TypeMissing { type_id: 7 },
            D::ScalarSizeTypeUnsupported,
            D::NullTypeUnsupported,
            D::ArrayVariableUnsupported,
            D::ValueIdUnset { id: 7 },
            D::IntOperandTypeMismatch,
            D::BufferBindingMissing { binding: 7 },
            D::FloatBufferReadWidthMismatch,
            D::BufferLoadTypeUnsupported,
            D::BufferOffsetDoesNotFitUsize { offset: 7 },
            D::BufferSizeDoesNotFitUsize { size: 7 },
            D::BufferRangeOverflow { offset: 7, size: 8 },
            D::BufferReadOutOfBounds {
                binding: 7,
                offset: 8,
                size: 9,
                len: 10,
            },
            D::InputVariableUnsupported,
            D::MemoryLoadVariableUnset { variable: 7 },
            D::MemoryLoadPathIntoScalar,
            D::MemoryLoadPathOutOfRange { index: 7 },
            D::BufferStoreUnsupported,
            D::MemoryStoreVariableUnset { variable: 7 },
            D::MemoryStorePathIntoScalar,
            D::MemoryStorePathOutOfRange { index: 7 },
            D::AccessChainBaseNotPointer,
            D::AccessChainBaseUnknown { base_id: 7 },
            D::AccessChainStructIndexOutOfRange {
                type_id: 7,
                index: 8,
            },
            D::AccessChainStructMemberOffsetMissing {
                type_id: 7,
                index: 8,
            },
            D::AccessChainStructOffsetOverflow,
            D::AccessChainArrayStrideMissing { type_id: 7 },
            D::AccessChainArrayOffsetOverflow,
            D::AccessChainVectorIndexOutOfRange { index: 7, count: 4 },
            D::AccessChainVectorOffsetOverflow,
            D::AccessChainTypeUnsupported { type_id: 7 },
            D::BufferVariableBindingMissing { variable: 7 },
            D::StorageClassUnsupported { storage: 7 },
            D::UnaryFloatOperandTypeMismatch,
            D::BinaryFloatOperandShapeMismatch,
            D::TernaryFloatOperandShapeMismatch,
            D::IntegerOperandShapeMismatch,
            D::FloatVectorMemberTypeMismatch,
            D::FloatVectorExpected,
            D::IntVectorElementTypeMismatch,
            D::IntResultTypeMismatch,
            D::PositionOutputMissing,
            D::EntryFunctionBodyMissingDuringRun,
            D::FunctionFellOffEnd,
            D::MainInstructionBudgetExhausted,
            D::FunctionInstructionMalformed,
            D::FunctionVariableTypeNotPointer,
            D::FunctionVariableStorageClassInvalid { storage: 7 },
            D::LoadSourceNotPointer,
            D::LoadPointerUnknown { id: 7 },
            D::StoreTargetNotPointer,
            D::StorePointerUnknown { id: 7 },
            D::SignedConvertSourceWidthUnknown { id: 7 },
            D::SignedToFloatSourceWidthUnknown { id: 7 },
            D::VectorScalarTypeMismatch,
            D::MatrixScalarTypeMismatch,
            D::MatrixTimesVectorMatrixNotComposite,
            D::MatrixTimesVectorColumnCountMismatch,
            D::MatrixTimesVectorColumnHeightMismatch,
            D::MatrixTimesVectorEmptyMatrix,
            D::VectorTimesMatrixMatrixNotComposite,
            D::VectorTimesMatrixShapeMismatch,
            D::MatrixTimesMatrixLeftNotComposite,
            D::MatrixTimesMatrixRightNotComposite,
            D::MatrixTimesMatrixShapeMismatch,
            D::MatrixTimesMatrixEmptyMatrix,
            D::TransposeMatrixNotComposite,
            D::TransposeMatrixRagged,
            D::DotShapeMismatch,
            D::CompositeExtractFromScalar,
            D::CompositeExtractIndexOutOfRange { index: 7 },
            D::CompositeInsertIntoScalar,
            D::CompositeInsertIndexOutOfRange { index: 7 },
            D::VectorShuffleLeftNotVector,
            D::VectorShuffleRightNotVector,
            D::VectorShuffleIndexOutOfRange { index: 7 },
            D::SelectValuesNotComposite,
            D::SelectVectorLengthMismatch,
            D::SelectVectorConditionNotBool,
            D::SelectConditionNotBool,
            D::SignedCompareWidthUnknown,
            D::LogicalNotVectorMemberNotBool,
            D::LogicalNotOperandNotBool,
            D::ExtInstSetUnknown { set_id: 7 },
            D::PhiOutsideBlockEntry,
            D::BranchConditionNotBool,
            D::SwitchSelectorWidthUnknown { id: 7 },
            D::SwitchOperandsMalformed,
            D::UnexpectedTerminator { opcode: 7 },
            D::OpcodeUnsupported { opcode: 7 },
            D::BranchLabelUnknown { label: 7 },
            D::PhiInstructionMalformed,
            D::PhiPredecessorMissing { predecessor: 7 },
            D::PhiInstructionBudgetExhausted,
            D::PositionVariableNeverStored { variable: 7 },
            D::PositionStructNeverStored { variable: 7 },
            D::PositionMemberNeverStored {
                variable: 7,
                member: 8,
            },
            D::PositionValueNotComposite,
            D::PositionVectorLengthInvalid { len: 7 },
            D::PositionComponentNotFinite { component: 2 },
            D::PositionComponentUndefined { component: 2 },
            D::MapIntOperandTypeMismatch,
            D::MapIntToFloatOperandTypeMismatch,
            D::MapFloatToIntOperandTypeMismatch,
            D::UnsignedDivisionByZero,
            D::SignedDivisionByZero,
            D::UnsignedModuloByZero,
            D::SignedRemainderByZero,
            D::IntegerBinaryOpcodeUnsupported { opcode: 7 },
            D::IntegerCompareOpcodeUnsupported { opcode: 7 },
            D::IntegerCompareShapeMismatch,
            D::FloatCompareOpcodeUnsupported { opcode: 7 },
            D::FloatCompareShapeMismatch,
            D::BooleanBinaryOpcodeUnsupported { opcode: 7 },
            D::BooleanBinaryShapeMismatch,
            D::BitcastTypeUnsupported,
            D::BitcastElementTypeUnsupported,
            D::ExtArgumentMissing { index: 7 },
            D::UnpackUnormOperandNotInt,
            D::ExtendedOpcodeUnsupported { opcode: 7 },
        ]
    }

    #[test]
    fn every_vertex_eval_check_has_a_unique_log_safe_slug_and_fields() {
        let declines = all_declines();
        assert_eq!(declines.len(), 129, "the vertex evaluator census moved");
        let mut slugs = Vec::with_capacity(declines.len());
        for decline in declines {
            assert!(decline.slug().starts_with("spirv_vertex_eval_"));
            assert!(decline
                .slug()
                .bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_'));
            for (key, value) in decline.fields() {
                assert!(!key.contains(char::is_whitespace));
                assert!(!value.contains(char::is_whitespace), "{key}={value}");
            }
            slugs.push(decline.slug());
        }
        slugs.sort_unstable();
        slugs.dedup();
        assert_eq!(slugs.len(), 129, "duplicate vertex evaluator reason");
    }

    #[test]
    fn evaluates_shader_pulled_quad_positions_through_switch_phi_and_matrix() {
        let words = shader_pulled_quad_module(false);
        let records = quad_record_bytes(1920.0, 1080.0);
        let matrix = ortho_matrix_bytes(1920.0, 1080.0);
        let config = config_bytes(2);
        let buffers: Vec<(u32, &[u8])> = vec![(0, &config), (1, &records), (2, &matrix)];
        let clip = evaluate_vertex_clip_positions(&words, &buffers, &[0, 1, 2, 3], 0)
            .expect("quad evaluates");
        let expected = [
            [-1.0f32, -1.0, 0.0, 1.0],
            [1.0, -1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 1.0],
            [-1.0, 1.0, 0.0, 1.0],
        ];
        for (got, want) in clip.iter().zip(&expected) {
            for (g, w) in got.iter().zip(want) {
                assert!((g - w).abs() < 1e-5, "clip {clip:?} vs {expected:?}");
            }
        }
    }

    #[test]
    fn switch_default_path_yields_sentinel_not_error() {
        let words = shader_pulled_quad_module(false);
        let records = quad_record_bytes(64.0, 48.0);
        let matrix = ortho_matrix_bytes(64.0, 48.0);
        let config = config_bytes(9); // no case matches → default block
        let buffers: Vec<(u32, &[u8])> = vec![(0, &config), (1, &records), (2, &matrix)];
        let clip = evaluate_vertex_clip_positions(&words, &buffers, &[0], 0)
            .expect("default path evaluates");
        // Sentinel max transformed by the ortho matrix stays enormous.
        assert!(clip[0][0].abs() > 1e6);
    }

    #[test]
    fn fails_closed_on_unsupported_opcode_oob_read_and_missing_binding() {
        let words = shader_pulled_quad_module(true);
        let records = quad_record_bytes(64.0, 48.0);
        let matrix = ortho_matrix_bytes(64.0, 48.0);
        let config = config_bytes(2);
        let buffers: Vec<(u32, &[u8])> = vec![(0, &config), (1, &records), (2, &matrix)];
        let err = evaluate_vertex_clip_positions(&words, &buffers, &[0], 0)
            .expect_err("unsupported op fails closed");
        assert_eq!(err, VertexEvalDecline::OpcodeUnsupported { opcode: 149 });

        let words = shader_pulled_quad_module(false);
        let short = &records[..8];
        let buffers: Vec<(u32, &[u8])> = vec![(0, &config), (1, short), (2, &matrix)];
        let err = evaluate_vertex_clip_positions(&words, &buffers, &[1], 0)
            .expect_err("out-of-bounds record fails closed");
        assert!(matches!(
            err,
            VertexEvalDecline::BufferReadOutOfBounds {
                binding: 1,
                len: 8,
                ..
            }
        ));

        let buffers: Vec<(u32, &[u8])> = vec![(0, &config), (2, &matrix)];
        let err = evaluate_vertex_clip_positions(&words, &buffers, &[0], 0)
            .expect_err("missing binding fails closed");
        assert_eq!(err, VertexEvalDecline::BufferBindingMissing { binding: 1 });
    }

    #[test]
    fn instruction_budget_rejects_unbounded_loops() {
        // Self-looping block: OpBranch to its own label forever.
        let mut words = vec![0x0723_0203, 0x0001_0400, 0, 20, 0];
        let mut ins = |op: u16, operands: &[u32]| {
            words.push((((operands.len() + 1) as u32) << 16) | op as u32);
            words.extend_from_slice(operands);
        };
        ins(
            OP_ENTRY_POINT,
            &[0, 10, u32::from_le_bytes(*b"main"), 0, 12],
        );
        ins(OP_DECORATE, &[12, DECORATION_BUILT_IN, BUILT_IN_POSITION]);
        ins(OP_TYPE_VOID, &[2]);
        ins(33, &[3, 2]);
        ins(OP_TYPE_FLOAT, &[4, 32]);
        ins(OP_TYPE_VECTOR, &[5, 4, 4]);
        ins(OP_TYPE_POINTER, &[6, STORAGE_CLASS_OUTPUT, 5]);
        ins(OP_VARIABLE, &[6, 12, STORAGE_CLASS_OUTPUT]);
        ins(OP_FUNCTION, &[2, 10, 0, 3]);
        ins(OP_LABEL, &[11]);
        ins(OP_BRANCH, &[11]);
        ins(OP_FUNCTION_END, &[]);
        let err = evaluate_vertex_clip_positions(&words, &[], &[0], 0)
            .expect_err("infinite loop hits budget");
        assert_eq!(err, VertexEvalDecline::MainInstructionBudgetExhausted);
    }
}
