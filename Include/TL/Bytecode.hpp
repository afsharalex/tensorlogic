// Include/TL/Bytecode.hpp
// Bytecode instruction set and module format for TensorLogic

#ifndef TL_BYTECODE_HPP
#define TL_BYTECODE_HPP

#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include <variant>
#include <optional>
#include <torch/torch.h>

namespace tl {

// Forward declare SourceLocation (from AST.hpp)
struct SourceLocation;

// ============================================================================
// BYTECODE INSTRUCTION SET
// ============================================================================

/// OpCode enumeration for all TensorLogic bytecode instructions
enum class OpCode : uint8_t {
    // === Tensor Creation (0x00-0x0F) ===
    LOAD_CONST = 0x00,      ///< Load constant from constant pool
    LOAD_VAR = 0x01,        ///< Load tensor from environment
    STORE_VAR = 0x02,       ///< Store tensor to environment
    CREATE_ZEROS = 0x03,    ///< Create zero tensor with given shape
    CREATE_ONES = 0x04,     ///< Create ones tensor with given shape
    CREATE_RANGE = 0x05,    ///< Create range tensor [0..n)

    // === Tensor Operations (0x10-0x2F) ===
    EINSUM = 0x10,          ///< Einstein summation
    MATMUL = 0x11,          ///< Matrix multiplication
    ELEMENTWISE_ADD = 0x12, ///< Element-wise addition
    ELEMENTWISE_SUB = 0x13, ///< Element-wise subtraction
    ELEMENTWISE_MUL = 0x14, ///< Element-wise multiplication
    ELEMENTWISE_DIV = 0x15, ///< Element-wise division
    ELEMENTWISE_POW = 0x16, ///< Element-wise power
    ELEMENTWISE_NEG = 0x17, ///< Element-wise negation (unary minus)
    REDUCTION_SUM = 0x18,   ///< Sum reduction
    REDUCTION_MAX = 0x19,   ///< Max reduction
    REDUCTION_MIN = 0x1A,   ///< Min reduction
    REDUCTION_AVG = 0x1B,   ///< Average reduction

    // === Tensor Indexing (0x30-0x3F) ===
    INDEX = 0x30,           ///< Index into tensor
    SLICE = 0x31,           ///< Slice tensor
    INDEX_ASSIGN = 0x32,    ///< Indexed assignment
    RESHAPE = 0x33,         ///< Reshape tensor
    TRANSPOSE = 0x34,       ///< Transpose tensor
    EXPAND = 0x35,          ///< Expand/broadcast tensor

    // === Activations (0x40-0x4F) ===
    RELU = 0x40,            ///< ReLU activation
    SIGMOID = 0x41,         ///< Sigmoid activation
    TANH = 0x42,            ///< Tanh activation
    SOFTMAX = 0x43,         ///< Softmax activation
    GELU = 0x44,            ///< GELU activation
    SWISH = 0x45,           ///< Swish activation
    STEP = 0x46,            ///< Heaviside step function

    // === Normalization (0x50-0x5F) ===
    LAYER_NORM = 0x50,      ///< Layer normalization
    BATCH_NORM = 0x51,      ///< Batch normalization
    SOFTMAX_DIM = 0x52,     ///< Softmax over specific dimension (normalized indices)

    // === Control Flow (0x60-0x6F) ===
    JUMP = 0x60,            ///< Unconditional jump
    JUMP_IF_ZERO = 0x61,    ///< Conditional jump (if zero)
    JUMP_IF_NONZERO = 0x62, ///< Conditional jump (if non-zero)
    JUMP_IF_DIRTY = 0x63,   ///< Jump if dirty flag set (for fixpoint)
    CALL = 0x64,            ///< Function call
    RETURN = 0x65,          ///< Return from function
    LABEL = 0x66,           ///< Label marker (pseudo-instruction, removed during compilation)

    // === Datalog Operations (0x70-0x7F) ===
    STORE_FACT = 0x70,      ///< Store Datalog fact
    LOAD_FACTS = 0x71,      ///< Load facts for relation
    JOIN = 0x72,            ///< Join two relations
    PROJECT = 0x73,         ///< Project relation
    SELECT = 0x74,          ///< Select with condition
    MARK_CLEAN = 0x75,      ///< Clear dirty flag for fixpoint

    // === Mask Operations (0x80-0x8F) - For Guarded Clauses ===
    CREATE_MASK = 0x80,     ///< Create boolean mask from condition
    APPLY_MASK = 0x81,      ///< Apply mask: result = tensor * mask
    MASK_SELECT = 0x82,     ///< Select elements where mask is true
    COMBINE_MASKS_OR = 0x83,  ///< Logical OR of masks
    COMBINE_MASKS_AND = 0x84, ///< Logical AND of masks
    NEGATE_MASK = 0x85,     ///< Logical NOT of mask

    // === Comparison Operations (0x90-0x9F) ===
    CMP_EQ = 0x90,          ///< Equal comparison
    CMP_NE = 0x91,          ///< Not equal comparison
    CMP_LT = 0x92,          ///< Less than comparison
    CMP_LE = 0x93,          ///< Less than or equal comparison
    CMP_GT = 0x94,          ///< Greater than comparison
    CMP_GE = 0x95,          ///< Greater than or equal comparison

    // === Label/Symbol Operations (0xA0-0xAF) - For Uppercase Identifiers ===
    INTERN_LABEL = 0xA0,    ///< Map label string to integer index
    LOAD_LABEL_INDEX = 0xA1,///< Load interned index for label

    // === Virtual Index Operations (0xB0-0xBF) - For RNN Recurrence ===
    VIRTUAL_READ = 0xB0,    ///< Read from virtual buffer (*t or *t-1)
    VIRTUAL_WRITE = 0xB1,   ///< Write to virtual buffer (*t+1)
    VIRTUAL_SWAP = 0xB2,    ///< Swap buffers (advance time step)

    // === Gradient/Learning Operations (0xC0-0xCF) ===
    BEGIN_GRADIENT_CONTEXT = 0xC0,  ///< Mark tensors for autograd
    COMPUTE_GRADIENTS = 0xC1,       ///< Backward pass from loss
    GRADIENT_STEP = 0xC2,           ///< Update parameters: param -= lr * grad
    ZERO_GRADIENTS = 0xC3,          ///< Clear accumulated gradients

    // === Meta Operations (0xD0-0xDF) ===
    NOP = 0xD0,             ///< No operation
    DEBUG_PRINT = 0xD1,     ///< Debug print tensor
    ASSERT_SHAPE = 0xD2,    ///< Assert tensor shape
    CHECKPOINT = 0xD3,      ///< Checkpoint for gradients
    COMMENT = 0xD4,         ///< Comment (removed during optimization)

    // === Special (0xF0-0xFF) ===
    HALT = 0xF0,            ///< Stop execution
    ERROR = 0xFF,           ///< Error instruction
};

/// Instruction flags (8 bits)
enum InstructionFlags : uint8_t {
    FLAG_NONE = 0x00,
    FLAG_HAS_IMMEDIATE = 0x01,    ///< Has immediate operand
    FLAG_WRITES_REGISTER = 0x02,  ///< Writes to register
    FLAG_READS_STACK = 0x04,      ///< Reads from stack
    FLAG_HAS_TYPE = 0x08,         ///< Has type annotation
    FLAG_HAS_EXTENSION = 0x10,    ///< Has extension word (next 8 bytes)
    FLAG_SIDE_EFFECT = 0x20,      ///< Has side effects (can't be eliminated)
};

// ============================================================================
// INSTRUCTION FORMAT
// ============================================================================

/// Fixed-width 8-byte instruction format
struct Instruction {
    OpCode opcode;          ///< Operation code (1 byte)
    uint8_t flags;          ///< Instruction flags (1 byte)

    /// Operands (6 bytes) - union for different formats
    union {
        /// Three 16-bit registers
        struct {
            uint16_t r0;
            uint16_t r1;
            uint16_t r2;
        } regs;

        /// One register + 32-bit immediate
        struct {
            uint16_t reg;
            uint32_t imm;
        } reg_imm;

        /// Jump target (32-bit PC + 16-bit padding)
        struct {
            uint32_t target;
            uint16_t padding;
        } jump;

        /// Two registers + 16-bit immediate (for dimension specs, etc.)
        struct {
            uint16_t r0;
            uint16_t r1;
            uint16_t imm16;
        } regs_imm16;

        /// Raw bytes (for future extensions)
        uint8_t raw[6];
    } operands;

    /// Source location ID (for debugging) - stored separately, not in 8-byte format
    uint32_t source_loc_id = 0;

    /// Default constructor
    Instruction() : opcode(OpCode::NOP), flags(FLAG_NONE) {
        operands.raw[0] = operands.raw[1] = operands.raw[2] = 0;
        operands.raw[3] = operands.raw[4] = operands.raw[5] = 0;
    }

    /// Constructor for register-based instructions
    Instruction(OpCode op, uint16_t r0, uint16_t r1 = 0, uint16_t r2 = 0)
        : opcode(op), flags(FLAG_WRITES_REGISTER) {
        operands.regs.r0 = r0;
        operands.regs.r1 = r1;
        operands.regs.r2 = r2;
    }

    /// Constructor for register + immediate
    Instruction(OpCode op, uint16_t reg, uint32_t imm)
        : opcode(op), flags(FLAG_HAS_IMMEDIATE | FLAG_WRITES_REGISTER) {
        operands.reg_imm.reg = reg;
        operands.reg_imm.imm = imm;
    }
};

// ============================================================================
// CONSTANT POOL
// ============================================================================

/// Constant type enumeration
enum class ConstantType : uint8_t {
    Integer,
    Float,
    Tensor,
    String,
    Shape,
    EinsumSpec,
};

/// Constant value in constant pool
struct Constant {
    ConstantType type;

    std::variant<
        int64_t,                    // Integer
        double,                     // Float
        torch::Tensor,              // Pre-computed tensor
        std::string,                // String literal
        std::vector<int64_t>        // Shape or einsum spec ID
    > value;

    Constant() : type(ConstantType::Integer), value(int64_t(0)) {}

    explicit Constant(int64_t i) : type(ConstantType::Integer), value(i) {}
    explicit Constant(double d) : type(ConstantType::Float), value(d) {}
    explicit Constant(const torch::Tensor& t) : type(ConstantType::Tensor), value(t) {}
    explicit Constant(const std::string& s) : type(ConstantType::String), value(s) {}
    explicit Constant(const std::vector<int64_t>& shape) : type(ConstantType::Shape), value(shape) {}
};

// ============================================================================
// EINSUM SPECIFICATIONS
// ============================================================================

/// Einstein summation specification
struct EinsumSpec {
    std::string equation;           ///< "ij,jk->ik"
    std::vector<int> lhs_indices;   ///< Left-hand side indices
    std::vector<int> rhs_indices;   ///< Right-hand side indices
    std::vector<int> output_indices;///< Output indices
    std::vector<int> sum_indices;   ///< Indices to sum over

    // Optimization hints
    bool can_use_matmul = false;    ///< Can be optimized to MATMUL
    std::vector<int> transpose_dims;///< Transpose optimization
    int batch_dims = 0;             ///< Number of batch dimensions
};

// ============================================================================
// FUNCTION INFORMATION
// ============================================================================

/// Function metadata
struct FunctionInfo {
    std::string name;
    uint32_t entry_point;           ///< PC of first instruction
    uint32_t num_params;
    uint32_t num_locals;
    uint32_t num_registers;
    // TODO: Add type information when type system is implemented
    // std::vector<TypeExpr> param_types;
    // TypeExpr return_type;
};

// ============================================================================
// DATALOG RELATION INFORMATION
// ============================================================================

/// Datalog relation metadata
struct RelationInfo {
    std::string name;
    uint32_t arity;                 ///< Number of arguments
    bool is_derived;                ///< True if derived by rules, false if only facts
};

// ============================================================================
// DEBUG INFORMATION
// ============================================================================

/// Debug source location information
struct BytecodeSourceLocation {
    std::string filename;
    uint32_t line;
    uint32_t column;

    BytecodeSourceLocation() : line(0), column(0) {}
    BytecodeSourceLocation(const std::string& file, uint32_t l, uint32_t c)
        : filename(file), line(l), column(c) {}
};

/// Debug information for bytecode module
struct DebugInfo {
    std::map<uint32_t, uint32_t> pc_to_source_loc;  ///< PC → BytecodeSourceLocation index
    std::map<uint16_t, std::string> register_names; ///< Register → variable name
    std::vector<BytecodeSourceLocation> source_locations;   ///< All source locations

    void addMapping(uint32_t pc, uint32_t source_loc_id) {
        pc_to_source_loc[pc] = source_loc_id;
    }

    void addRegisterName(uint16_t reg, const std::string& name) {
        register_names[reg] = name;
    }

    uint32_t addSourceLocation(const BytecodeSourceLocation& loc) {
        source_locations.push_back(loc);
        return static_cast<uint32_t>(source_locations.size() - 1);
    }

    std::optional<BytecodeSourceLocation> getSourceLocation(uint32_t pc) const {
        auto it = pc_to_source_loc.find(pc);
        if (it != pc_to_source_loc.end() && it->second < source_locations.size()) {
            return source_locations[it->second];
        }
        return std::nullopt;
    }
};

// ============================================================================
// BYTECODE MODULE
// ============================================================================

/// Complete bytecode module (compiled program)
struct BytecodeModule {
    // === Header ===
    static constexpr uint32_t MAGIC = 0x544C4243;  // "TLBC" (TensorLogic ByteCode)
    static constexpr uint16_t VERSION_MAJOR = 1;
    static constexpr uint16_t VERSION_MINOR = 0;

    uint32_t magic = MAGIC;
    uint16_t version_major = VERSION_MAJOR;
    uint16_t version_minor = VERSION_MINOR;

    // === Constant Pool ===
    std::vector<Constant> constants;

    // === String Pool ===
    std::vector<std::string> strings;

    // === Einsum Specifications ===
    std::vector<EinsumSpec> einsum_specs;

    // === Function Table ===
    std::vector<FunctionInfo> functions;

    // === Datalog Relations ===
    std::vector<RelationInfo> relations;

    // === Debug Information ===
    DebugInfo debug_info;

    // === Bytecode Instructions ===
    std::vector<Instruction> instructions;

    // === Metadata ===
    std::map<std::string, std::string> metadata;

    // === Deduplication Maps (for optimization) ===
    std::map<size_t, uint32_t> constant_hash_to_id;
    std::map<std::string, uint32_t> string_to_id;
    std::map<std::string, uint32_t> einsum_equation_to_id;

    // === Helper Methods ===

    /// Add a constant to the constant pool (with deduplication)
    uint32_t addConstant(const Constant& c);

    /// Add a string to the string pool (with deduplication)
    uint32_t addString(const std::string& s);

    /// Add an einsum specification (with deduplication)
    uint32_t addEinsumSpec(const EinsumSpec& spec);

    /// Add an instruction
    void addInstruction(const Instruction& instr) {
        instructions.push_back(instr);
    }

    /// Get current program counter (next instruction position)
    uint32_t getCurrentPC() const {
        return static_cast<uint32_t>(instructions.size());
    }

    /// Emit an instruction and return its PC
    uint32_t emit(const Instruction& instr) {
        uint32_t pc = getCurrentPC();
        addInstruction(instr);
        return pc;
    }
};

// ============================================================================
// BYTECODE UTILITIES
// ============================================================================

/// Convert opcode to string (for disassembly)
const char* opcodeToString(OpCode op);

/// Compute hash for constant (for deduplication)
size_t hashConstant(const Constant& c);

/// Disassemble a single instruction
std::string disassemble(const Instruction& instr, const BytecodeModule* module = nullptr);

/// Disassemble entire bytecode module
std::string disassembleModule(const BytecodeModule& module);

} // namespace tl

#endif // TL_BYTECODE_HPP
