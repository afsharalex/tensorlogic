// Include/TL/BytecodeVM.hpp
// Bytecode virtual machine for TensorLogic

#ifndef TL_BYTECODE_VM_HPP
#define TL_BYTECODE_VM_HPP

#include "TL/Bytecode.hpp"
#include "TL/backend.hpp"
#include "TL/VM.hpp"
#include <vector>
#include <map>
#include <iostream>

namespace tl {

// ============================================================================
// BYTECODE VM
// ============================================================================

/// Virtual machine for executing TensorLogic bytecode
class BytecodeVM {
public:
    /// Construct bytecode VM with backend and output stream
    BytecodeVM(TensorBackend& backend, Environment& env, std::ostream* out = &std::cout);

    /// Execute a bytecode module
    void execute(const BytecodeModule& module);

    /// Execute with debugging enabled
    void debug(const BytecodeModule& module);

    /// Get execution environment
    Environment& environment() { return env_; }
    const Environment& environment() const { return env_; }

    /// Enable/disable debug mode
    void setDebugMode(bool enabled) { debug_mode_ = enabled; }

private:
    // ========================================================================
    // INTERPRETER LOOP
    // ========================================================================

    /// Main interpreter loop
    void run(const BytecodeModule& module);

    /// Execute a single instruction
    void executeInstruction(const Instruction& instr);

    /// Print instruction for debugging
    void printInstruction(const Instruction& instr, uint32_t pc);

    // ========================================================================
    // INSTRUCTION HANDLERS - TENSOR CREATION
    // ========================================================================

    void op_load_const(const Instruction& instr);
    void op_load_var(const Instruction& instr);
    void op_store_var(const Instruction& instr);
    void op_create_zeros(const Instruction& instr);
    void op_create_ones(const Instruction& instr);
    void op_create_range(const Instruction& instr);

    // ========================================================================
    // INSTRUCTION HANDLERS - TENSOR OPERATIONS
    // ========================================================================

    void op_einsum(const Instruction& instr);
    void op_matmul(const Instruction& instr);
    void op_elementwise_add(const Instruction& instr);
    void op_elementwise_sub(const Instruction& instr);
    void op_elementwise_mul(const Instruction& instr);
    void op_elementwise_div(const Instruction& instr);
    void op_elementwise_pow(const Instruction& instr);
    void op_elementwise_neg(const Instruction& instr);
    void op_reduction_sum(const Instruction& instr);
    void op_reduction_max(const Instruction& instr);
    void op_reduction_min(const Instruction& instr);
    void op_reduction_avg(const Instruction& instr);

    // ========================================================================
    // INSTRUCTION HANDLERS - TENSOR INDEXING
    // ========================================================================

    void op_index(const Instruction& instr);
    void op_slice(const Instruction& instr);
    void op_index_assign(const Instruction& instr);
    void op_reshape(const Instruction& instr);
    void op_transpose(const Instruction& instr);
    void op_expand(const Instruction& instr);

    // ========================================================================
    // INSTRUCTION HANDLERS - ACTIVATIONS
    // ========================================================================

    void op_relu(const Instruction& instr);
    void op_sigmoid(const Instruction& instr);
    void op_tanh(const Instruction& instr);
    void op_softmax(const Instruction& instr);
    void op_gelu(const Instruction& instr);
    void op_swish(const Instruction& instr);
    void op_step(const Instruction& instr);

    // ========================================================================
    // INSTRUCTION HANDLERS - NORMALIZATION
    // ========================================================================

    void op_layer_norm(const Instruction& instr);
    void op_batch_norm(const Instruction& instr);
    void op_softmax_dim(const Instruction& instr);

    // ========================================================================
    // INSTRUCTION HANDLERS - CONTROL FLOW
    // ========================================================================

    void op_jump(const Instruction& instr);
    void op_jump_if_zero(const Instruction& instr);
    void op_jump_if_nonzero(const Instruction& instr);
    void op_jump_if_dirty(const Instruction& instr);
    void op_call(const Instruction& instr);
    void op_return(const Instruction& instr);

    // ========================================================================
    // INSTRUCTION HANDLERS - DATALOG OPERATIONS
    // ========================================================================

    void op_store_fact(const Instruction& instr);
    void op_load_facts(const Instruction& instr);
    void op_join(const Instruction& instr);
    void op_project(const Instruction& instr);
    void op_select(const Instruction& instr);
    void op_mark_clean(const Instruction& instr);

    // ========================================================================
    // INSTRUCTION HANDLERS - MASK OPERATIONS
    // ========================================================================

    void op_create_mask(const Instruction& instr);
    void op_apply_mask(const Instruction& instr);
    void op_mask_select(const Instruction& instr);
    void op_combine_masks_or(const Instruction& instr);
    void op_combine_masks_and(const Instruction& instr);
    void op_negate_mask(const Instruction& instr);

    // ========================================================================
    // INSTRUCTION HANDLERS - COMPARISON OPERATIONS
    // ========================================================================

    void op_cmp_eq(const Instruction& instr);
    void op_cmp_ne(const Instruction& instr);
    void op_cmp_lt(const Instruction& instr);
    void op_cmp_le(const Instruction& instr);
    void op_cmp_gt(const Instruction& instr);
    void op_cmp_ge(const Instruction& instr);

    // ========================================================================
    // INSTRUCTION HANDLERS - LABEL OPERATIONS
    // ========================================================================

    void op_intern_label(const Instruction& instr);
    void op_load_label_index(const Instruction& instr);

    // ========================================================================
    // INSTRUCTION HANDLERS - VIRTUAL INDEX OPERATIONS
    // ========================================================================

    void op_virtual_read(const Instruction& instr);
    void op_virtual_write(const Instruction& instr);
    void op_virtual_swap(const Instruction& instr);

    // ========================================================================
    // INSTRUCTION HANDLERS - GRADIENT/LEARNING OPERATIONS
    // ========================================================================

    void op_begin_gradient_context(const Instruction& instr);
    void op_compute_gradients(const Instruction& instr);
    void op_gradient_step(const Instruction& instr);
    void op_zero_gradients(const Instruction& instr);

    // ========================================================================
    // INSTRUCTION HANDLERS - META OPERATIONS
    // ========================================================================

    void op_nop(const Instruction& instr);
    void op_debug_print(const Instruction& instr);
    void op_assert_shape(const Instruction& instr);
    void op_checkpoint(const Instruction& instr);

    // ========================================================================
    // VM STATE
    // ========================================================================

    /// Tensor backend
    TensorBackend& backend_;

    /// Execution environment (shared with AST VM)
    Environment& env_;

    /// Output stream
    std::ostream* out_;

    /// Register file (virtual registers for bytecode execution)
    std::vector<torch::Tensor> registers_;

    /// Program counter
    uint32_t pc_;

    /// Current bytecode module
    const BytecodeModule* module_;

    /// Debug mode flag
    bool debug_mode_;

    /// Dirty flag (for Datalog fixpoint iteration)
    bool dirty_flag_;

    /// Call stack (for function calls)
    struct StackFrame {
        uint32_t return_pc;
        uint16_t base_register;
    };
    std::vector<StackFrame> call_stack_;

    /// Virtual index buffers (for RNN operations)
    std::map<std::string, torch::Tensor> virtual_buffers_;

    /// Gradient context (tensors being tracked for gradients)
    std::vector<std::string> gradient_context_;

    // ========================================================================
    // HELPER METHODS
    // ========================================================================

    /// Get tensor from register
    torch::Tensor& getReg(uint16_t reg);

    /// Set tensor in register
    void setReg(uint16_t reg, const torch::Tensor& tensor);

    /// Get constant from constant pool
    const Constant& getConstant(uint32_t id);

    /// Get string from string pool
    const std::string& getString(uint32_t id);

    /// Get einsum spec from spec pool
    const EinsumSpec& getEinsumSpec(uint32_t id);

    /// Runtime error with source location
    void runtimeError(const std::string& message);
};

} // namespace tl

#endif // TL_BYTECODE_VM_HPP
