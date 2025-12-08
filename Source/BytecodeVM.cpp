// Source/BytecodeVM.cpp
// Implementation of bytecode virtual machine

#include "TL/BytecodeVM.hpp"
#include "TL/Compiler.hpp"
#include <iostream>
#include <iomanip>
#include <stdexcept>

namespace tl {

// ============================================================================
// CONSTRUCTOR
// ============================================================================

BytecodeVM::BytecodeVM(TensorBackend& backend, Environment& env, std::ostream* out)
    : backend_(backend)
    , env_(env)
    , out_(out)
    , pc_(0)
    , module_(nullptr)
    , debug_mode_(false)
    , dirty_flag_(false)
{
    // Allocate register file (256 registers)
    registers_.resize(256);
}

// ============================================================================
// EXECUTION
// ============================================================================

void BytecodeVM::execute(const BytecodeModule& module) {
    // Verify bytecode before execution
    BytecodeVerifier verifier;
    if (!verifier.verify(module)) {
        std::cerr << "Bytecode verification failed:" << std::endl;
        for (const auto& error : verifier.getErrors()) {
            std::cerr << "  - " << error << std::endl;
        }
        throw std::runtime_error("Bytecode verification failed");
    }

    // Run the bytecode
    run(module);
}

void BytecodeVM::debug(const BytecodeModule& module) {
    debug_mode_ = true;
    execute(module);
    debug_mode_ = false;
}

void BytecodeVM::run(const BytecodeModule& module) {
    module_ = &module;
    pc_ = 0;

    if (debug_mode_) {
        *out_ << "=== BytecodeVM Execution ===" << std::endl;
        *out_ << "Instructions: " << module.instructions.size() << std::endl;
        *out_ << "Constants: " << module.constants.size() << std::endl;
        *out_ << "Strings: " << module.strings.size() << std::endl;
        *out_ << std::endl;
    }

    // Main interpreter loop
    while (pc_ < module.instructions.size()) {
        const Instruction& instr = module.instructions[pc_];

        // Debug print
        if (debug_mode_) {
            printInstruction(instr, pc_);
        }

        // Check for halt
        if (instr.opcode == OpCode::HALT) {
            if (debug_mode_) {
                *out_ << "=== Execution Complete ===" << std::endl;
            }
            break;
        }

        // Save PC in case instruction modifies it (jumps)
        uint32_t saved_pc = pc_;

        // Execute instruction
        try {
            executeInstruction(instr);
        } catch (const std::exception& e) {
            runtimeError(std::string("Instruction execution failed: ") + e.what());
            throw;
        }

        // Advance PC if instruction didn't modify it
        if (pc_ == saved_pc) {
            ++pc_;
        }
    }
}

void BytecodeVM::executeInstruction(const Instruction& instr) {
    switch (instr.opcode) {
        // Tensor Creation
        case OpCode::LOAD_CONST: op_load_const(instr); break;
        case OpCode::LOAD_VAR: op_load_var(instr); break;
        case OpCode::STORE_VAR: op_store_var(instr); break;
        case OpCode::CREATE_ZEROS: op_create_zeros(instr); break;
        case OpCode::CREATE_ONES: op_create_ones(instr); break;
        case OpCode::CREATE_RANGE: op_create_range(instr); break;

        // Tensor Operations
        case OpCode::EINSUM: op_einsum(instr); break;
        case OpCode::MATMUL: op_matmul(instr); break;
        case OpCode::ELEMENTWISE_ADD: op_elementwise_add(instr); break;
        case OpCode::ELEMENTWISE_SUB: op_elementwise_sub(instr); break;
        case OpCode::ELEMENTWISE_MUL: op_elementwise_mul(instr); break;
        case OpCode::ELEMENTWISE_DIV: op_elementwise_div(instr); break;
        case OpCode::ELEMENTWISE_POW: op_elementwise_pow(instr); break;
        case OpCode::ELEMENTWISE_NEG: op_elementwise_neg(instr); break;
        case OpCode::REDUCTION_SUM: op_reduction_sum(instr); break;
        case OpCode::REDUCTION_MAX: op_reduction_max(instr); break;
        case OpCode::REDUCTION_MIN: op_reduction_min(instr); break;
        case OpCode::REDUCTION_AVG: op_reduction_avg(instr); break;

        // Activations
        case OpCode::RELU: op_relu(instr); break;
        case OpCode::SIGMOID: op_sigmoid(instr); break;
        case OpCode::TANH: op_tanh(instr); break;
        case OpCode::SOFTMAX: op_softmax(instr); break;
        case OpCode::GELU: op_gelu(instr); break;
        case OpCode::SWISH: op_swish(instr); break;
        case OpCode::STEP: op_step(instr); break;

        // Normalization
        case OpCode::SOFTMAX_DIM: op_softmax_dim(instr); break;

        // Comparison Operations
        case OpCode::CMP_EQ: op_cmp_eq(instr); break;
        case OpCode::CMP_NE: op_cmp_ne(instr); break;
        case OpCode::CMP_LT: op_cmp_lt(instr); break;
        case OpCode::CMP_LE: op_cmp_le(instr); break;
        case OpCode::CMP_GT: op_cmp_gt(instr); break;
        case OpCode::CMP_GE: op_cmp_ge(instr); break;

        // Control Flow
        case OpCode::JUMP: op_jump(instr); break;
        case OpCode::JUMP_IF_ZERO: op_jump_if_zero(instr); break;
        case OpCode::JUMP_IF_NONZERO: op_jump_if_nonzero(instr); break;

        // Meta Operations
        case OpCode::NOP: op_nop(instr); break;
        case OpCode::DEBUG_PRINT: op_debug_print(instr); break;

        case OpCode::HALT:
            // Already handled in run()
            break;

        default:
            runtimeError("Unimplemented opcode: " + std::string(opcodeToString(instr.opcode)));
            break;
    }
}

void BytecodeVM::printInstruction(const Instruction& instr, uint32_t pc) {
    *out_ << std::setw(6) << std::right << pc << ":  ";
    *out_ << disassemble(instr, module_);

    // Show source location if available
    auto loc = module_->debug_info.getSourceLocation(pc);
    if (loc) {
        *out_ << "  // " << loc->filename << ":" << loc->line;
    }

    *out_ << std::endl;
}

// ============================================================================
// TENSOR CREATION HANDLERS
// ============================================================================

void BytecodeVM::op_load_const(const Instruction& instr) {
    uint16_t dest_reg = instr.operands.reg_imm.reg;
    uint32_t const_id = instr.operands.reg_imm.imm;

    const Constant& c = getConstant(const_id);

    // Convert constant to tensor
    if (auto* tensor = std::get_if<torch::Tensor>(&c.value)) {
        setReg(dest_reg, *tensor);
    } else if (auto* num = std::get_if<double>(&c.value)) {
        setReg(dest_reg, torch::tensor(*num, torch::kFloat32));
    } else if (auto* ints = std::get_if<int64_t>(&c.value)) {
        setReg(dest_reg, torch::tensor(*ints, torch::kFloat32));
    } else if (auto* shape = std::get_if<std::vector<int64_t>>(&c.value)) {
        // Shape is used for CREATE_ZEROS/ONES, not LOAD_CONST
        runtimeError("Cannot load shape as constant");
    }
}

void BytecodeVM::op_load_var(const Instruction& instr) {
    uint16_t dest_reg = instr.operands.reg_imm.reg;
    const std::string& var_name = getString(instr.operands.reg_imm.imm);

    // Load tensor from environment
    if (env_.has(var_name)) {
        setReg(dest_reg, env_.lookup(var_name));
    } else {
        runtimeError("Variable not found: " + var_name);
    }
}

void BytecodeVM::op_store_var(const Instruction& instr) {
    uint16_t src_reg = instr.operands.reg_imm.reg;
    const std::string& var_name = getString(instr.operands.reg_imm.imm);

    // Store tensor to environment
    env_.bind(var_name, getReg(src_reg));
}

void BytecodeVM::op_create_zeros(const Instruction& instr) {
    // TODO: Implement
    runtimeError("CREATE_ZEROS not yet implemented");
}

void BytecodeVM::op_create_ones(const Instruction& instr) {
    // TODO: Implement
    runtimeError("CREATE_ONES not yet implemented");
}

void BytecodeVM::op_create_range(const Instruction& instr) {
    // TODO: Implement
    runtimeError("CREATE_RANGE not yet implemented");
}

// ============================================================================
// TENSOR OPERATION HANDLERS
// ============================================================================

void BytecodeVM::op_einsum(const Instruction& instr) {
    uint16_t dest = instr.operands.reg_imm.reg;
    uint32_t spec_id = instr.operands.reg_imm.imm;

    // Get einsum specification
    const EinsumSpec& spec = getEinsumSpec(spec_id);

    // Gather input tensors from registers specified in the spec
    std::vector<torch::Tensor> inputs;
    inputs.reserve(spec.input_regs.size());
    for (uint16_t input_reg : spec.input_regs) {
        inputs.push_back(getReg(input_reg));
    }

    // Perform Einstein summation
    torch::Tensor result = torch::einsum(spec.equation, inputs);

    // Store result in destination register
    setReg(dest, result);
}

void BytecodeVM::op_matmul(const Instruction& instr) {
    uint16_t dest = instr.operands.regs.r0;
    uint16_t src1 = instr.operands.regs.r1;
    uint16_t src2 = instr.operands.regs.r2;

    setReg(dest, torch::matmul(getReg(src1), getReg(src2)));
}

void BytecodeVM::op_elementwise_add(const Instruction& instr) {
    uint16_t dest = instr.operands.regs.r0;
    uint16_t src1 = instr.operands.regs.r1;
    uint16_t src2 = instr.operands.regs.r2;

    setReg(dest, getReg(src1) + getReg(src2));
}

void BytecodeVM::op_elementwise_sub(const Instruction& instr) {
    uint16_t dest = instr.operands.regs.r0;
    uint16_t src1 = instr.operands.regs.r1;
    uint16_t src2 = instr.operands.regs.r2;

    setReg(dest, getReg(src1) - getReg(src2));
}

void BytecodeVM::op_elementwise_mul(const Instruction& instr) {
    uint16_t dest = instr.operands.regs.r0;
    uint16_t src1 = instr.operands.regs.r1;
    uint16_t src2 = instr.operands.regs.r2;

    setReg(dest, getReg(src1) * getReg(src2));
}

void BytecodeVM::op_elementwise_div(const Instruction& instr) {
    uint16_t dest = instr.operands.regs.r0;
    uint16_t src1 = instr.operands.regs.r1;
    uint16_t src2 = instr.operands.regs.r2;

    setReg(dest, getReg(src1) / getReg(src2));
}

void BytecodeVM::op_elementwise_pow(const Instruction& instr) {
    uint16_t dest = instr.operands.regs.r0;
    uint16_t src1 = instr.operands.regs.r1;
    uint16_t src2 = instr.operands.regs.r2;

    setReg(dest, torch::pow(getReg(src1), getReg(src2)));
}

void BytecodeVM::op_elementwise_neg(const Instruction& instr) {
    uint16_t dest = instr.operands.regs.r0;
    uint16_t src = instr.operands.regs.r1;

    setReg(dest, -getReg(src));
}

void BytecodeVM::op_reduction_sum(const Instruction& instr) {
    uint16_t dest = instr.operands.regs.r0;
    uint16_t src = instr.operands.regs.r1;

    setReg(dest, torch::sum(getReg(src)));
}

void BytecodeVM::op_reduction_max(const Instruction& instr) {
    uint16_t dest = instr.operands.regs.r0;
    uint16_t src = instr.operands.regs.r1;

    setReg(dest, torch::max(getReg(src)));
}

void BytecodeVM::op_reduction_min(const Instruction& instr) {
    uint16_t dest = instr.operands.regs.r0;
    uint16_t src = instr.operands.regs.r1;

    setReg(dest, torch::min(getReg(src)));
}

void BytecodeVM::op_reduction_avg(const Instruction& instr) {
    uint16_t dest = instr.operands.regs.r0;
    uint16_t src = instr.operands.regs.r1;

    setReg(dest, torch::mean(getReg(src)));
}

// ============================================================================
// ACTIVATION HANDLERS
// ============================================================================

void BytecodeVM::op_relu(const Instruction& instr) {
    uint16_t dest = instr.operands.regs.r0;
    uint16_t src = instr.operands.regs.r1;

    setReg(dest, torch::relu(getReg(src)));
}

void BytecodeVM::op_sigmoid(const Instruction& instr) {
    uint16_t dest = instr.operands.regs.r0;
    uint16_t src = instr.operands.regs.r1;

    setReg(dest, torch::sigmoid(getReg(src)));
}

void BytecodeVM::op_tanh(const Instruction& instr) {
    uint16_t dest = instr.operands.regs.r0;
    uint16_t src = instr.operands.regs.r1;

    setReg(dest, torch::tanh(getReg(src)));
}

void BytecodeVM::op_softmax(const Instruction& instr) {
    uint16_t dest = instr.operands.regs.r0;
    uint16_t src = instr.operands.regs.r1;

    // Softmax over last dimension by default
    auto tensor = getReg(src);
    int64_t dim = tensor.dim() - 1;
    if (dim < 0) dim = 0;

    setReg(dest, torch::softmax(tensor, dim));
}

void BytecodeVM::op_gelu(const Instruction& instr) {
    uint16_t dest = instr.operands.regs.r0;
    uint16_t src = instr.operands.regs.r1;

    setReg(dest, torch::gelu(getReg(src)));
}

void BytecodeVM::op_swish(const Instruction& instr) {
    uint16_t dest = instr.operands.regs.r0;
    uint16_t src = instr.operands.regs.r1;

    auto x = getReg(src);
    setReg(dest, x * torch::sigmoid(x));
}

void BytecodeVM::op_step(const Instruction& instr) {
    uint16_t dest = instr.operands.regs.r0;
    uint16_t src = instr.operands.regs.r1;

    // Heaviside step function: 0 if x < 0, 1 if x >= 0
    setReg(dest, (getReg(src) >= 0).to(torch::kFloat32));
}

// ============================================================================
// NORMALIZATION HANDLERS
// ============================================================================

void BytecodeVM::op_softmax_dim(const Instruction& instr) {
    uint16_t dest = instr.operands.regs.r0;
    uint16_t src = instr.operands.regs.r1;
    uint16_t dim = instr.operands.regs_imm16.imm16;

    setReg(dest, torch::softmax(getReg(src), dim));
}

// ============================================================================
// COMPARISON HANDLERS
// ============================================================================

void BytecodeVM::op_cmp_eq(const Instruction& instr) {
    uint16_t dest = instr.operands.regs.r0;
    uint16_t src1 = instr.operands.regs.r1;
    uint16_t src2 = instr.operands.regs.r2;

    setReg(dest, (getReg(src1) == getReg(src2)).to(torch::kFloat32));
}

void BytecodeVM::op_cmp_ne(const Instruction& instr) {
    uint16_t dest = instr.operands.regs.r0;
    uint16_t src1 = instr.operands.regs.r1;
    uint16_t src2 = instr.operands.regs.r2;

    setReg(dest, (getReg(src1) != getReg(src2)).to(torch::kFloat32));
}

void BytecodeVM::op_cmp_lt(const Instruction& instr) {
    uint16_t dest = instr.operands.regs.r0;
    uint16_t src1 = instr.operands.regs.r1;
    uint16_t src2 = instr.operands.regs.r2;

    setReg(dest, (getReg(src1) < getReg(src2)).to(torch::kFloat32));
}

void BytecodeVM::op_cmp_le(const Instruction& instr) {
    uint16_t dest = instr.operands.regs.r0;
    uint16_t src1 = instr.operands.regs.r1;
    uint16_t src2 = instr.operands.regs.r2;

    setReg(dest, (getReg(src1) <= getReg(src2)).to(torch::kFloat32));
}

void BytecodeVM::op_cmp_gt(const Instruction& instr) {
    uint16_t dest = instr.operands.regs.r0;
    uint16_t src1 = instr.operands.regs.r1;
    uint16_t src2 = instr.operands.regs.r2;

    setReg(dest, (getReg(src1) > getReg(src2)).to(torch::kFloat32));
}

void BytecodeVM::op_cmp_ge(const Instruction& instr) {
    uint16_t dest = instr.operands.regs.r0;
    uint16_t src1 = instr.operands.regs.r1;
    uint16_t src2 = instr.operands.regs.r2;

    setReg(dest, (getReg(src1) >= getReg(src2)).to(torch::kFloat32));
}

// ============================================================================
// CONTROL FLOW HANDLERS
// ============================================================================

void BytecodeVM::op_jump(const Instruction& instr) {
    pc_ = instr.operands.jump.target;
}

void BytecodeVM::op_jump_if_zero(const Instruction& instr) {
    uint16_t cond_reg = instr.operands.regs_imm16.r0;
    uint32_t target = static_cast<uint32_t>(instr.operands.regs_imm16.imm16);

    auto tensor = getReg(cond_reg);
    if (torch::all(tensor == 0).item<bool>()) {
        pc_ = target;
    }
}

void BytecodeVM::op_jump_if_nonzero(const Instruction& instr) {
    uint16_t cond_reg = instr.operands.regs_imm16.r0;
    uint32_t target = static_cast<uint32_t>(instr.operands.regs_imm16.imm16);

    auto tensor = getReg(cond_reg);
    if (torch::any(tensor != 0).item<bool>()) {
        pc_ = target;
    }
}

void BytecodeVM::op_call(const Instruction& instr) {
    // TODO: Implement function calls
    runtimeError("CALL not yet implemented");
}

void BytecodeVM::op_return(const Instruction& instr) {
    // TODO: Implement function returns
    runtimeError("RETURN not yet implemented");
}

// ============================================================================
// DATALOG HANDLERS (STUBS)
// ============================================================================

void BytecodeVM::op_store_fact(const Instruction& instr) {
    runtimeError("STORE_FACT not yet implemented");
}

void BytecodeVM::op_load_facts(const Instruction& instr) {
    runtimeError("LOAD_FACTS not yet implemented");
}

void BytecodeVM::op_join(const Instruction& instr) {
    runtimeError("JOIN not yet implemented");
}

void BytecodeVM::op_project(const Instruction& instr) {
    runtimeError("PROJECT not yet implemented");
}

void BytecodeVM::op_select(const Instruction& instr) {
    runtimeError("SELECT not yet implemented");
}

void BytecodeVM::op_mark_clean(const Instruction& instr) {
    dirty_flag_ = false;
}

// ============================================================================
// MASK OPERATION HANDLERS (STUBS)
// ============================================================================

void BytecodeVM::op_create_mask(const Instruction& instr) {
    runtimeError("CREATE_MASK not yet implemented");
}

void BytecodeVM::op_apply_mask(const Instruction& instr) {
    runtimeError("APPLY_MASK not yet implemented");
}

void BytecodeVM::op_mask_select(const Instruction& instr) {
    runtimeError("MASK_SELECT not yet implemented");
}

void BytecodeVM::op_combine_masks_or(const Instruction& instr) {
    runtimeError("COMBINE_MASKS_OR not yet implemented");
}

void BytecodeVM::op_combine_masks_and(const Instruction& instr) {
    runtimeError("COMBINE_MASKS_AND not yet implemented");
}

void BytecodeVM::op_negate_mask(const Instruction& instr) {
    uint16_t dest = instr.operands.regs.r0;
    uint16_t src = instr.operands.regs.r1;

    // Logical NOT: 1.0 - mask
    setReg(dest, 1.0f - getReg(src));
}

// ============================================================================
// LABEL OPERATION HANDLERS (STUBS)
// ============================================================================

void BytecodeVM::op_intern_label(const Instruction& instr) {
    runtimeError("INTERN_LABEL not yet implemented");
}

void BytecodeVM::op_load_label_index(const Instruction& instr) {
    runtimeError("LOAD_LABEL_INDEX not yet implemented");
}

// ============================================================================
// VIRTUAL INDEX HANDLERS (STUBS)
// ============================================================================

void BytecodeVM::op_virtual_read(const Instruction& instr) {
    runtimeError("VIRTUAL_READ not yet implemented");
}

void BytecodeVM::op_virtual_write(const Instruction& instr) {
    runtimeError("VIRTUAL_WRITE not yet implemented");
}

void BytecodeVM::op_virtual_swap(const Instruction& instr) {
    runtimeError("VIRTUAL_SWAP not yet implemented");
}

// ============================================================================
// GRADIENT OPERATION HANDLERS (STUBS)
// ============================================================================

void BytecodeVM::op_begin_gradient_context(const Instruction& instr) {
    runtimeError("BEGIN_GRADIENT_CONTEXT not yet implemented");
}

void BytecodeVM::op_compute_gradients(const Instruction& instr) {
    runtimeError("COMPUTE_GRADIENTS not yet implemented");
}

void BytecodeVM::op_gradient_step(const Instruction& instr) {
    runtimeError("GRADIENT_STEP not yet implemented");
}

void BytecodeVM::op_zero_gradients(const Instruction& instr) {
    runtimeError("ZERO_GRADIENTS not yet implemented");
}

// ============================================================================
// META OPERATION HANDLERS
// ============================================================================

void BytecodeVM::op_nop(const Instruction& instr) {
    // No operation
}

void BytecodeVM::op_debug_print(const Instruction& instr) {
    uint16_t src = instr.operands.regs.r0;

    *out_ << "DEBUG: r" << src << " = " << getReg(src) << std::endl;
}

void BytecodeVM::op_assert_shape(const Instruction& instr) {
    // TODO: Implement shape assertions
    runtimeError("ASSERT_SHAPE not yet implemented");
}

void BytecodeVM::op_checkpoint(const Instruction& instr) {
    // TODO: Implement gradient checkpointing
    runtimeError("CHECKPOINT not yet implemented");
}

// Stubs for unimplemented indexing operations
void BytecodeVM::op_index(const Instruction& instr) {
    runtimeError("INDEX not yet implemented");
}

void BytecodeVM::op_slice(const Instruction& instr) {
    runtimeError("SLICE not yet implemented");
}

void BytecodeVM::op_index_assign(const Instruction& instr) {
    runtimeError("INDEX_ASSIGN not yet implemented");
}

void BytecodeVM::op_reshape(const Instruction& instr) {
    runtimeError("RESHAPE not yet implemented");
}

void BytecodeVM::op_transpose(const Instruction& instr) {
    runtimeError("TRANSPOSE not yet implemented");
}

void BytecodeVM::op_expand(const Instruction& instr) {
    runtimeError("EXPAND not yet implemented");
}

void BytecodeVM::op_layer_norm(const Instruction& instr) {
    runtimeError("LAYER_NORM not yet implemented");
}

void BytecodeVM::op_batch_norm(const Instruction& instr) {
    runtimeError("BATCH_NORM not yet implemented");
}

void BytecodeVM::op_jump_if_dirty(const Instruction& instr) {
    if (dirty_flag_) {
        pc_ = instr.operands.jump.target;
    }
}

// ============================================================================
// HELPER METHODS
// ============================================================================

torch::Tensor& BytecodeVM::getReg(uint16_t reg) {
    if (reg >= registers_.size()) {
        runtimeError("Register out of bounds: r" + std::to_string(reg));
    }
    return registers_[reg];
}

void BytecodeVM::setReg(uint16_t reg, const torch::Tensor& tensor) {
    if (reg >= registers_.size()) {
        runtimeError("Register out of bounds: r" + std::to_string(reg));
    }
    registers_[reg] = tensor;
}

const Constant& BytecodeVM::getConstant(uint32_t id) {
    if (id >= module_->constants.size()) {
        runtimeError("Constant ID out of bounds: #" + std::to_string(id));
    }
    return module_->constants[id];
}

const std::string& BytecodeVM::getString(uint32_t id) {
    if (id >= module_->strings.size()) {
        runtimeError("String ID out of bounds: @" + std::to_string(id));
    }
    return module_->strings[id];
}

const EinsumSpec& BytecodeVM::getEinsumSpec(uint32_t id) {
    if (id >= module_->einsum_specs.size()) {
        runtimeError("Einsum spec ID out of bounds: E" + std::to_string(id));
    }
    return module_->einsum_specs[id];
}

void BytecodeVM::runtimeError(const std::string& message) {
    std::cerr << "Runtime error at PC " << pc_ << ": " << message << std::endl;

    // Show source location if available
    auto loc = module_->debug_info.getSourceLocation(pc_);
    if (loc) {
        std::cerr << "  at " << loc->filename << ":" << loc->line << std::endl;
    }

    throw std::runtime_error(message);
}

} // namespace tl
