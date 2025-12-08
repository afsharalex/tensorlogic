// Source/Bytecode.cpp
// Implementation of bytecode utilities

#include "TL/Bytecode.hpp"
#include <sstream>
#include <iomanip>
#include <functional>

namespace tl {

// ============================================================================
// BytecodeModule Helper Methods
// ============================================================================

uint32_t BytecodeModule::addConstant(const Constant& c) {
    // Compute hash for deduplication
    size_t hash = hashConstant(c);

    // Check if already exists
    auto it = constant_hash_to_id.find(hash);
    if (it != constant_hash_to_id.end()) {
        return it->second;
    }

    // Add new constant
    uint32_t id = static_cast<uint32_t>(constants.size());
    constants.push_back(c);
    constant_hash_to_id[hash] = id;
    return id;
}

uint32_t BytecodeModule::addString(const std::string& s) {
    // Check if already exists
    auto it = string_to_id.find(s);
    if (it != string_to_id.end()) {
        return it->second;
    }

    // Add new string
    uint32_t id = static_cast<uint32_t>(strings.size());
    strings.push_back(s);
    string_to_id[s] = id;
    return id;
}

uint32_t BytecodeModule::addEinsumSpec(const EinsumSpec& spec) {
    // Check if already exists by equation
    auto it = einsum_equation_to_id.find(spec.equation);
    if (it != einsum_equation_to_id.end()) {
        return it->second;
    }

    // Add new einsum spec
    uint32_t id = static_cast<uint32_t>(einsum_specs.size());
    einsum_specs.push_back(spec);
    einsum_equation_to_id[spec.equation] = id;
    return id;
}

// ============================================================================
// Bytecode Utilities
// ============================================================================

const char* opcodeToString(OpCode op) {
    switch (op) {
        // Tensor Creation
        case OpCode::LOAD_CONST: return "LOAD_CONST";
        case OpCode::LOAD_VAR: return "LOAD_VAR";
        case OpCode::STORE_VAR: return "STORE_VAR";
        case OpCode::CREATE_ZEROS: return "CREATE_ZEROS";
        case OpCode::CREATE_ONES: return "CREATE_ONES";
        case OpCode::CREATE_RANGE: return "CREATE_RANGE";

        // Tensor Operations
        case OpCode::EINSUM: return "EINSUM";
        case OpCode::MATMUL: return "MATMUL";
        case OpCode::ELEMENTWISE_ADD: return "ELEMENTWISE_ADD";
        case OpCode::ELEMENTWISE_SUB: return "ELEMENTWISE_SUB";
        case OpCode::ELEMENTWISE_MUL: return "ELEMENTWISE_MUL";
        case OpCode::ELEMENTWISE_DIV: return "ELEMENTWISE_DIV";
        case OpCode::ELEMENTWISE_POW: return "ELEMENTWISE_POW";
        case OpCode::ELEMENTWISE_NEG: return "ELEMENTWISE_NEG";
        case OpCode::REDUCTION_SUM: return "REDUCTION_SUM";
        case OpCode::REDUCTION_MAX: return "REDUCTION_MAX";
        case OpCode::REDUCTION_MIN: return "REDUCTION_MIN";
        case OpCode::REDUCTION_AVG: return "REDUCTION_AVG";

        // Tensor Indexing
        case OpCode::INDEX: return "INDEX";
        case OpCode::SLICE: return "SLICE";
        case OpCode::INDEX_ASSIGN: return "INDEX_ASSIGN";
        case OpCode::RESHAPE: return "RESHAPE";
        case OpCode::TRANSPOSE: return "TRANSPOSE";
        case OpCode::EXPAND: return "EXPAND";

        // Activations
        case OpCode::RELU: return "RELU";
        case OpCode::SIGMOID: return "SIGMOID";
        case OpCode::TANH: return "TANH";
        case OpCode::SOFTMAX: return "SOFTMAX";
        case OpCode::GELU: return "GELU";
        case OpCode::SWISH: return "SWISH";
        case OpCode::STEP: return "STEP";

        // Normalization
        case OpCode::LAYER_NORM: return "LAYER_NORM";
        case OpCode::BATCH_NORM: return "BATCH_NORM";
        case OpCode::SOFTMAX_DIM: return "SOFTMAX_DIM";

        // Control Flow
        case OpCode::JUMP: return "JUMP";
        case OpCode::JUMP_IF_ZERO: return "JUMP_IF_ZERO";
        case OpCode::JUMP_IF_NONZERO: return "JUMP_IF_NONZERO";
        case OpCode::JUMP_IF_DIRTY: return "JUMP_IF_DIRTY";
        case OpCode::CALL: return "CALL";
        case OpCode::RETURN: return "RETURN";
        case OpCode::LABEL: return "LABEL";

        // Datalog Operations
        case OpCode::STORE_FACT: return "STORE_FACT";
        case OpCode::LOAD_FACTS: return "LOAD_FACTS";
        case OpCode::JOIN: return "JOIN";
        case OpCode::PROJECT: return "PROJECT";
        case OpCode::SELECT: return "SELECT";
        case OpCode::MARK_CLEAN: return "MARK_CLEAN";

        // Mask Operations
        case OpCode::CREATE_MASK: return "CREATE_MASK";
        case OpCode::APPLY_MASK: return "APPLY_MASK";
        case OpCode::MASK_SELECT: return "MASK_SELECT";
        case OpCode::COMBINE_MASKS_OR: return "COMBINE_MASKS_OR";
        case OpCode::COMBINE_MASKS_AND: return "COMBINE_MASKS_AND";
        case OpCode::NEGATE_MASK: return "NEGATE_MASK";

        // Comparison Operations
        case OpCode::CMP_EQ: return "CMP_EQ";
        case OpCode::CMP_NE: return "CMP_NE";
        case OpCode::CMP_LT: return "CMP_LT";
        case OpCode::CMP_LE: return "CMP_LE";
        case OpCode::CMP_GT: return "CMP_GT";
        case OpCode::CMP_GE: return "CMP_GE";

        // Label/Symbol Operations
        case OpCode::INTERN_LABEL: return "INTERN_LABEL";
        case OpCode::LOAD_LABEL_INDEX: return "LOAD_LABEL_INDEX";

        // Virtual Index Operations
        case OpCode::VIRTUAL_READ: return "VIRTUAL_READ";
        case OpCode::VIRTUAL_WRITE: return "VIRTUAL_WRITE";
        case OpCode::VIRTUAL_SWAP: return "VIRTUAL_SWAP";

        // Gradient/Learning Operations
        case OpCode::BEGIN_GRADIENT_CONTEXT: return "BEGIN_GRADIENT_CONTEXT";
        case OpCode::COMPUTE_GRADIENTS: return "COMPUTE_GRADIENTS";
        case OpCode::GRADIENT_STEP: return "GRADIENT_STEP";
        case OpCode::ZERO_GRADIENTS: return "ZERO_GRADIENTS";

        // Meta Operations
        case OpCode::NOP: return "NOP";
        case OpCode::DEBUG_PRINT: return "DEBUG_PRINT";
        case OpCode::ASSERT_SHAPE: return "ASSERT_SHAPE";
        case OpCode::CHECKPOINT: return "CHECKPOINT";
        case OpCode::COMMENT: return "COMMENT";

        // Special
        case OpCode::HALT: return "HALT";
        case OpCode::ERROR: return "ERROR";

        default: return "UNKNOWN";
    }
}

size_t hashConstant(const Constant& c) {
    std::hash<int64_t> hash_int;
    std::hash<double> hash_double;
    std::hash<std::string> hash_string;

    // Combine type and value hash
    size_t h = static_cast<size_t>(c.type);

    switch (c.type) {
        case ConstantType::Integer:
            h ^= hash_int(std::get<int64_t>(c.value)) << 1;
            break;
        case ConstantType::Float:
            h ^= hash_double(std::get<double>(c.value)) << 1;
            break;
        case ConstantType::String:
            h ^= hash_string(std::get<std::string>(c.value)) << 1;
            break;
        case ConstantType::Shape: {
            const auto& shape = std::get<std::vector<int64_t>>(c.value);
            for (int64_t dim : shape) {
                h ^= hash_int(dim) + 0x9e3779b9 + (h << 6) + (h >> 2);
            }
            break;
        }
        case ConstantType::Tensor: {
            // For tensors, hash shape, dtype, and a sample of values
            const auto& tensor = std::get<torch::Tensor>(c.value);
            h ^= static_cast<size_t>(tensor.scalar_type()) << 1;

            // Hash shape
            for (int64_t dim : tensor.sizes()) {
                h ^= hash_int(dim) + 0x9e3779b9 + (h << 6) + (h >> 2);
            }

            // Hash first few values for better discrimination
            if (tensor.numel() > 0 && tensor.numel() <= 16) {
                // For small tensors, hash all values
                auto flat = tensor.flatten();
                for (int64_t i = 0; i < flat.size(0); ++i) {
                    h ^= hash_double(flat.index({i}).item<double>()) + 0x9e3779b9 + (h << 6) + (h >> 2);
                }
            }
            break;
        }
        default:
            break;
    }

    return h;
}

std::string disassemble(const Instruction& instr, const BytecodeModule* module) {
    std::stringstream ss;
    ss << std::setw(20) << std::left << opcodeToString(instr.opcode);

    // Decode operands based on opcode
    switch (instr.opcode) {
        // Instructions with three registers
        case OpCode::ELEMENTWISE_ADD:
        case OpCode::ELEMENTWISE_SUB:
        case OpCode::ELEMENTWISE_MUL:
        case OpCode::ELEMENTWISE_DIV:
        case OpCode::ELEMENTWISE_POW:
        case OpCode::MATMUL:
        case OpCode::APPLY_MASK:
        case OpCode::COMBINE_MASKS_OR:
        case OpCode::COMBINE_MASKS_AND:
        case OpCode::CMP_EQ:
        case OpCode::CMP_NE:
        case OpCode::CMP_LT:
        case OpCode::CMP_LE:
        case OpCode::CMP_GT:
        case OpCode::CMP_GE:
            ss << "r" << instr.operands.regs.r0 << ", "
               << "r" << instr.operands.regs.r1 << ", "
               << "r" << instr.operands.regs.r2;
            break;

        // Instructions with two registers
        case OpCode::ELEMENTWISE_NEG:
        case OpCode::RELU:
        case OpCode::SIGMOID:
        case OpCode::TANH:
        case OpCode::SOFTMAX:
        case OpCode::GELU:
        case OpCode::SWISH:
        case OpCode::STEP:
        case OpCode::REDUCTION_SUM:
        case OpCode::REDUCTION_MAX:
        case OpCode::REDUCTION_MIN:
        case OpCode::REDUCTION_AVG:
        case OpCode::NEGATE_MASK:
        case OpCode::CREATE_MASK:
        case OpCode::MASK_SELECT:
            ss << "r" << instr.operands.regs.r0 << ", "
               << "r" << instr.operands.regs.r1;
            break;

        // Instructions with register + immediate
        case OpCode::LOAD_CONST:
            ss << "r" << instr.operands.reg_imm.reg << ", #" << instr.operands.reg_imm.imm;
            if (module && instr.operands.reg_imm.imm < module->constants.size()) {
                const auto& c = module->constants[instr.operands.reg_imm.imm];
                ss << " (";
                switch (c.type) {
                    case ConstantType::Integer:
                        ss << std::get<int64_t>(c.value);
                        break;
                    case ConstantType::Float:
                        ss << std::get<double>(c.value);
                        break;
                    case ConstantType::String:
                        ss << "\"" << std::get<std::string>(c.value) << "\"";
                        break;
                    case ConstantType::Shape: {
                        const auto& shape = std::get<std::vector<int64_t>>(c.value);
                        ss << "[";
                        for (size_t i = 0; i < shape.size(); ++i) {
                            if (i > 0) ss << ", ";
                            ss << shape[i];
                        }
                        ss << "]";
                        break;
                    }
                    case ConstantType::Tensor:
                        ss << "tensor";
                        break;
                    default:
                        break;
                }
                ss << ")";
            }
            break;

        case OpCode::LOAD_VAR:
        case OpCode::STORE_VAR:
            ss << "r" << instr.operands.reg_imm.reg << ", ";
            if (module && instr.operands.reg_imm.imm < module->strings.size()) {
                ss << "\"" << module->strings[instr.operands.reg_imm.imm] << "\"";
            } else {
                ss << "#" << instr.operands.reg_imm.imm;
            }
            break;

        case OpCode::EINSUM:
            ss << "r" << instr.operands.reg_imm.reg << ", E" << instr.operands.reg_imm.imm;
            if (module && instr.operands.reg_imm.imm < module->einsum_specs.size()) {
                ss << " (\"" << module->einsum_specs[instr.operands.reg_imm.imm].equation << "\")";
            }
            break;

        // Jump instructions
        case OpCode::JUMP:
        case OpCode::JUMP_IF_ZERO:
        case OpCode::JUMP_IF_NONZERO:
        case OpCode::JUMP_IF_DIRTY:
            ss << instr.operands.jump.target;
            break;

        // Simple instructions with no operands
        case OpCode::NOP:
        case OpCode::HALT:
        case OpCode::RETURN:
        case OpCode::MARK_CLEAN:
        case OpCode::VIRTUAL_SWAP:
        case OpCode::ZERO_GRADIENTS:
            // No operands
            break;

        default:
            // Generic display for unknown formats
            ss << "r" << instr.operands.regs.r0;
            break;
    }

    return ss.str();
}

std::string disassembleModule(const BytecodeModule& module) {
    std::stringstream ss;

    // Header
    ss << "TensorLogic Bytecode Module\n";
    ss << "===========================\n";
    ss << "Magic: 0x" << std::hex << module.magic << std::dec << "\n";
    ss << "Version: " << module.version_major << "." << module.version_minor << "\n\n";

    // Constants
    if (!module.constants.empty()) {
        ss << "Constants (" << module.constants.size() << "):\n";
        for (size_t i = 0; i < module.constants.size(); ++i) {
            ss << "  #" << i << ": ";
            const auto& c = module.constants[i];
            switch (c.type) {
                case ConstantType::Integer:
                    ss << std::get<int64_t>(c.value);
                    break;
                case ConstantType::Float:
                    ss << std::get<double>(c.value);
                    break;
                case ConstantType::String:
                    ss << "\"" << std::get<std::string>(c.value) << "\"";
                    break;
                case ConstantType::Shape: {
                    const auto& shape = std::get<std::vector<int64_t>>(c.value);
                    ss << "[";
                    for (size_t j = 0; j < shape.size(); ++j) {
                        if (j > 0) ss << ", ";
                        ss << shape[j];
                    }
                    ss << "]";
                    break;
                }
                case ConstantType::Tensor:
                    ss << "tensor<" << std::get<torch::Tensor>(c.value).sizes() << ">";
                    break;
                default:
                    break;
            }
            ss << "\n";
        }
        ss << "\n";
    }

    // Strings
    if (!module.strings.empty()) {
        ss << "Strings (" << module.strings.size() << "):\n";
        for (size_t i = 0; i < module.strings.size(); ++i) {
            ss << "  @" << i << ": \"" << module.strings[i] << "\"\n";
        }
        ss << "\n";
    }

    // Einsum Specs
    if (!module.einsum_specs.empty()) {
        ss << "Einsum Specifications (" << module.einsum_specs.size() << "):\n";
        for (size_t i = 0; i < module.einsum_specs.size(); ++i) {
            ss << "  E" << i << ": " << module.einsum_specs[i].equation << "\n";
        }
        ss << "\n";
    }

    // Instructions
    ss << "Instructions (" << module.instructions.size() << "):\n";
    for (size_t i = 0; i < module.instructions.size(); ++i) {
        ss << std::setw(6) << std::right << i << ":  ";
        ss << disassemble(module.instructions[i], &module);

        // Add source location if available
        auto loc = module.debug_info.getSourceLocation(static_cast<uint32_t>(i));
        if (loc) {
            ss << "  // " << loc->filename << ":" << loc->line;
        }

        ss << "\n";
    }

    return ss.str();
}

} // namespace tl
