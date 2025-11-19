// Source/Compiler.cpp
// Implementation of AST to Bytecode compiler

#include "TL/Compiler.hpp"
#include "TL/vm.hpp"
#include <iostream>
#include <sstream>
#include <algorithm>
#include <stdexcept>

namespace tl {

// ============================================================================
// REGISTER ALLOCATOR
// ============================================================================

uint16_t RegisterAllocator::allocate() {
    if (!free_list_.empty()) {
        uint16_t reg = free_list_.back();
        free_list_.pop_back();
        return reg;
    }
    return next_register_++;
}

void RegisterAllocator::free(uint16_t reg) {
    free_list_.push_back(reg);
}

void RegisterAllocator::setLiveRange(uint16_t reg, uint32_t start, uint32_t end) {
    live_ranges_[reg] = {start, end};
}

// ============================================================================
// SYMBOL TABLE
// ============================================================================

std::optional<Symbol> SymbolTable::lookup(const std::string& name) const {
    auto it = current_scope_.find(name);
    if (it != current_scope_.end()) {
        return it->second;
    }
    return std::nullopt;
}

void SymbolTable::define(const std::string& name, uint16_t reg) {
    Symbol sym;
    sym.name = name;
    sym.reg = reg;
    current_scope_[name] = sym;
}

bool SymbolTable::exists(const std::string& name) const {
    return current_scope_.find(name) != current_scope_.end();
}

uint16_t SymbolTable::getRegister(const std::string& name) const {
    auto it = current_scope_.find(name);
    if (it != current_scope_.end()) {
        return it->second.reg;
    }
    throw std::runtime_error("Symbol not found: " + name);
}

void SymbolTable::markConstant(const std::string& name, const Constant& value) {
    auto it = current_scope_.find(name);
    if (it != current_scope_.end()) {
        it->second.is_constant = true;
        it->second.constant_value = value;
    }
}

void SymbolTable::pushScope() {
    scopes_.push_back(current_scope_);
}

void SymbolTable::popScope() {
    if (!scopes_.empty()) {
        current_scope_ = scopes_.back();
        scopes_.pop_back();
    }
}

// ============================================================================
// COMPILER
// ============================================================================

Compiler::Compiler(const CompilerOptions& opts)
    : options_(opts) {
}

BytecodeModule Compiler::compile(const Program& program) {
    // Reset state
    module_ = BytecodeModule();
    symbols_ = SymbolTable();
    registers_.reset();
    labels_.clear();
    unresolved_jumps_.clear();
    register_constants_.clear();
    last_error_.clear();
    success_ = true;

    if (options_.verbose) {
        std::cout << "Compiling TensorLogic program (" << program.statements.size()
                  << " statements)..." << std::endl;
    }

    // Phase 1: Semantic analysis (optional, for now just compile)
    // analyzeSemantics(program);

    // Phase 2: Code generation
    for (const auto& stmt : program.statements) {
        compileStatement(stmt);
        if (!success_) {
            break;
        }
    }

    // Emit HALT at end
    emit(OpCode::HALT, 0, 0, 0);

    // Resolve forward jumps
    for (const auto& [jump_pc, label] : unresolved_jumps_) {
        auto it = labels_.find(label);
        if (it != labels_.end()) {
            patchJump(jump_pc, it->second);
        } else {
            error("Undefined label: " + label);
        }
    }

    // Phase 3: Optimization
    if (options_.optimize && success_) {
        runOptimizationPasses();
    }

    if (options_.verbose) {
        std::cout << "Compilation " << (success_ ? "succeeded" : "failed") << std::endl;
        std::cout << "Generated " << module_.instructions.size() << " instructions" << std::endl;
        std::cout << "Using " << registers_.getMaxRegister() << " registers" << std::endl;
    }

    return module_;
}

// ============================================================================
// STATEMENT COMPILATION
// ============================================================================

void Compiler::compileStatement(const Statement& stmt) {
    std::visit([this](const auto& s) {
        using T = std::decay_t<decltype(s)>;

        if constexpr (std::is_same_v<T, TensorEquation>) {
            compileEquation(s);
        } else if constexpr (std::is_same_v<T, DatalogFact>) {
            compileDatalogFact(s);
        } else if constexpr (std::is_same_v<T, DatalogRule>) {
            compileDatalogRule(s);
        } else if constexpr (std::is_same_v<T, Query>) {
            compileQuery(s);
        } else if constexpr (std::is_same_v<T, FileOperation>) {
            compileFileOperation(s);
        } else if constexpr (std::is_same_v<T, FixedPointLoop>) {
            // TODO: Implement fixed-point loop compilation
            if (options_.verbose) {
                std::cout << "Note: Fixed-point loops not yet compiled to bytecode" << std::endl;
            }
        }
        // Note: Comments don't exist as statements in the new AST
    }, stmt);
}

// ============================================================================
// TENSOR EQUATION COMPILATION
// ============================================================================

void Compiler::compileEquation(const TensorEquation& eq) {
    if (options_.verbose) {
        std::cout << "Compiling equation..." << std::endl;
    }

    const std::string& lhs_name = eq.lhs.name.name;

    // For now, compile the first clause only (TODO: support guarded clauses)
    if (eq.clauses.empty()) {
        error("Equation has no clauses");
        return;
    }

    const GuardedClause& clause = eq.clauses[0];

    // Compile RHS expression
    uint16_t rhs_reg = compileExpression(*clause.expr);

    // Store result to variable
    emitStoreVar(rhs_reg, lhs_name);

    // Track variable register
    if (!symbols_.exists(lhs_name)) {
        symbols_.define(lhs_name, rhs_reg);
        addRegisterName(rhs_reg, lhs_name);
    }

    // TODO: Handle indexed assignments
    // TODO: Handle guarded clauses
    // TODO: Handle multiple clauses (additive)
}

// ============================================================================
// EXPRESSION COMPILATION
// ============================================================================

uint16_t Compiler::compileExpression(const Expr& expr) {
    return std::visit([this](const auto& e) -> uint16_t {
        using T = std::decay_t<decltype(e)>;

        if constexpr (std::is_same_v<T, ExprNumber>) {
            return compileNumberLiteral(e.literal);
        } else if constexpr (std::is_same_v<T, ExprList>) {
            return compileListLiteral(e);
        } else if constexpr (std::is_same_v<T, ExprTensorRef>) {
            return compileTensorRef(e.ref);
        } else if constexpr (std::is_same_v<T, ExprCall>) {
            return compileFunctionCall(e);
        } else if constexpr (std::is_same_v<T, ExprBinary>) {
            return compileArithmetic(e.op, *e.lhs, *e.rhs);
        } else if constexpr (std::is_same_v<T, ExprUnary>) {
            return compileUnary(e.op, *e.operand);
        } else if constexpr (std::is_same_v<T, ExprParen>) {
            return compileExpression(*e.inner);
        } else {
            error("Unknown expression type");
            return 0;
        }
    }, expr.node);
}

uint16_t Compiler::compileNumberLiteral(const NumberLiteral& lit) {
    // Parse the number literal
    Constant c;
    try {
        if (lit.text.find('.') != std::string::npos ||
            lit.text.find('e') != std::string::npos ||
            lit.text.find('E') != std::string::npos) {
            // Float literal
            c = Constant(std::stod(lit.text));
        } else {
            // Integer literal
            c = Constant(std::stoll(lit.text));
        }
    } catch (...) {
        error("Invalid number literal: " + lit.text);
        c = Constant(int64_t(0));
    }

    uint16_t dest_reg = registers_.allocate();
    emitLoadConst(dest_reg, c);

    // Track as constant
    register_constants_[dest_reg] = c;

    return dest_reg;
}

uint16_t Compiler::compileListLiteral(const ExprList& lit) {
    // Convert list literal to tensor constant
    std::vector<double> values;

    // Recursively extract values from nested lists
    std::function<void(const ExprPtr&)> extractValues = [&](const ExprPtr& elem) {
        if (std::holds_alternative<ExprNumber>(elem->node)) {
            const auto& num = std::get<ExprNumber>(elem->node);
            try {
                values.push_back(std::stod(num.literal.text));
            } catch (...) {
                error("Invalid number in list: " + num.literal.text);
            }
        } else if (std::holds_alternative<ExprList>(elem->node)) {
            const auto& nested = std::get<ExprList>(elem->node);
            for (const auto& nested_elem : nested.elements) {
                extractValues(nested_elem);
            }
        }
    };

    for (const auto& elem : lit.elements) {
        extractValues(elem);
    }

    // Create tensor from values
    torch::Tensor tensor = torch::tensor(values, torch::kFloat32);
    Constant c(tensor);

    uint16_t dest_reg = registers_.allocate();
    emitLoadConst(dest_reg, c);

    return dest_reg;
}

uint16_t Compiler::compileTensorRef(const TensorRef& ref) {
    const std::string& var_name = ref.name.name;

    // Check if variable already has a register assigned
    if (symbols_.exists(var_name)) {
        // Variable exists - return its register
        return symbols_.getRegister(var_name);
    }

    // Variable not yet loaded - load from environment
    uint16_t dest_reg = registers_.allocate();
    emitLoadVar(dest_reg, var_name);
    symbols_.define(var_name, dest_reg);
    addRegisterName(dest_reg, var_name);

    // TODO: Handle indexed access
    return dest_reg;
}

uint16_t Compiler::compileFunctionCall(const ExprCall& call) {
    // Compile function arguments
    std::vector<uint16_t> arg_regs;
    for (const auto& arg : call.args) {
        arg_regs.push_back(compileExpression(*arg));
    }

    uint16_t dest_reg = registers_.allocate();

    // Map function name to opcode
    const std::string& func_name = call.func.name;

    if (func_name == "relu") {
        emit(OpCode::RELU, dest_reg, arg_regs[0]);
    } else if (func_name == "sigmoid" || func_name == "sig") {
        emit(OpCode::SIGMOID, dest_reg, arg_regs[0]);
    } else if (func_name == "tanh") {
        emit(OpCode::TANH, dest_reg, arg_regs[0]);
    } else if (func_name == "softmax") {
        emit(OpCode::SOFTMAX, dest_reg, arg_regs[0]);
    } else if (func_name == "gelu") {
        emit(OpCode::GELU, dest_reg, arg_regs[0]);
    } else if (func_name == "step" || func_name == "H") {
        emit(OpCode::STEP, dest_reg, arg_regs[0]);
    } else {
        error("Unknown function: " + func_name);
    }

    // Don't free argument registers (may be variable registers needed later)

    return dest_reg;
}

uint16_t Compiler::compileArithmetic(ExprBinary::Op op,
                                       const Expr& left,
                                       const Expr& right) {
    uint16_t left_reg = compileExpression(left);
    uint16_t right_reg = compileExpression(right);
    uint16_t dest_reg = registers_.allocate();

    switch (op) {
        case ExprBinary::Op::Add:
            emit(OpCode::ELEMENTWISE_ADD, dest_reg, left_reg, right_reg);
            break;
        case ExprBinary::Op::Sub:
            emit(OpCode::ELEMENTWISE_SUB, dest_reg, left_reg, right_reg);
            break;
        case ExprBinary::Op::Mul:
            emit(OpCode::ELEMENTWISE_MUL, dest_reg, left_reg, right_reg);
            break;
        case ExprBinary::Op::Div:
            emit(OpCode::ELEMENTWISE_DIV, dest_reg, left_reg, right_reg);
            break;
        case ExprBinary::Op::Pow:
            emit(OpCode::ELEMENTWISE_POW, dest_reg, left_reg, right_reg);
            break;
        case ExprBinary::Op::Lt:
            emit(OpCode::CMP_LT, dest_reg, left_reg, right_reg);
            break;
        case ExprBinary::Op::Le:
            emit(OpCode::CMP_LE, dest_reg, left_reg, right_reg);
            break;
        case ExprBinary::Op::Gt:
            emit(OpCode::CMP_GT, dest_reg, left_reg, right_reg);
            break;
        case ExprBinary::Op::Ge:
            emit(OpCode::CMP_GE, dest_reg, left_reg, right_reg);
            break;
        case ExprBinary::Op::Eq:
            emit(OpCode::CMP_EQ, dest_reg, left_reg, right_reg);
            break;
        case ExprBinary::Op::Ne:
            emit(OpCode::CMP_NE, dest_reg, left_reg, right_reg);
            break;
        default:
            error("Unknown binary operator");
            break;
    }

    // Note: Don't free left_reg and right_reg here.
    // They may be variable registers that need to stay live for later uses.
    // A proper liveness analysis would determine when registers can be safely freed.

    return dest_reg;
}

uint16_t Compiler::compileUnary(ExprUnary::Op op, const Expr& expr) {
    uint16_t operand_reg = compileExpression(expr);
    uint16_t dest_reg = registers_.allocate();

    switch (op) {
        case ExprUnary::Op::Neg:
            emit(OpCode::ELEMENTWISE_NEG, dest_reg, operand_reg);
            break;
        case ExprUnary::Op::Not:
            emit(OpCode::NEGATE_MASK, dest_reg, operand_reg);
            break;
        default:
            error("Unknown unary operator");
            break;
    }

    // Don't free operand_reg (may be variable register needed later)
    return dest_reg;
}

// ============================================================================
// DATALOG COMPILATION (STUBS FOR NOW)
// ============================================================================

void Compiler::compileDatalogFact(const DatalogFact& fact) {
    // TODO: Implement Datalog fact compilation
    // For now, we'll delegate to existing VM
    if (options_.verbose) {
        std::cout << "Note: Datalog facts not yet compiled to bytecode" << std::endl;
    }
}

void Compiler::compileDatalogRule(const DatalogRule& rule) {
    // TODO: Implement Datalog rule compilation
    if (options_.verbose) {
        std::cout << "Note: Datalog rules not yet compiled to bytecode" << std::endl;
    }
}

void Compiler::compileQuery(const Query& query) {
    // TODO: Implement query compilation
    if (options_.verbose) {
        std::cout << "Note: Queries not yet compiled to bytecode" << std::endl;
    }
}

void Compiler::compileFileOperation(const FileOperation& op) {
    // TODO: Implement file operation compilation
    if (options_.verbose) {
        std::cout << "Note: File operations not yet compiled to bytecode" << std::endl;
    }
}

// ============================================================================
// OPTIMIZATION PASSES
// ============================================================================

void Compiler::runOptimizationPasses() {
    if (options_.verbose) {
        std::cout << "Running optimization passes..." << std::endl;
    }

    if (options_.enable_constant_folding) {
        constantFolding();
    }

    if (options_.enable_dead_code_elimination) {
        deadCodeElimination();
    }

    if (options_.enable_cse) {
        commonSubexpressionElimination();
    }

    if (options_.enable_fusion) {
        fusionOptimization();
    }
}

void Compiler::constantFolding() {
    // TODO: Implement constant folding optimization
}

void Compiler::deadCodeElimination() {
    // TODO: Implement dead code elimination
}

void Compiler::commonSubexpressionElimination() {
    // TODO: Implement CSE
}

void Compiler::fusionOptimization() {
    // TODO: Implement operation fusion
}

// ============================================================================
// BYTECODE EMISSION
// ============================================================================

uint32_t Compiler::emit(const Instruction& instr) {
    uint32_t pc = module_.emit(instr);

    // Record source location if available
    if (current_line_ > 0 && options_.generate_debug_info) {
        BytecodeSourceLocation loc(current_filename_,
                                   static_cast<uint32_t>(current_line_),
                                   static_cast<uint32_t>(current_column_));
        uint32_t loc_id = module_.debug_info.addSourceLocation(loc);
        module_.debug_info.addMapping(pc, loc_id);
    }

    return pc;
}

uint32_t Compiler::emit(OpCode op, uint16_t r0, uint16_t r1, uint16_t r2) {
    Instruction instr(op, r0, r1, r2);
    return emit(instr);
}

uint32_t Compiler::emit(OpCode op, uint16_t reg, uint32_t imm) {
    Instruction instr(op, reg, imm);
    return emit(instr);
}

uint32_t Compiler::emitJump(OpCode op, uint32_t target) {
    Instruction instr;
    instr.opcode = op;
    instr.operands.jump.target = target;
    return emit(instr);
}

void Compiler::patchJump(uint32_t jump_pc, uint32_t target) {
    if (jump_pc < module_.instructions.size()) {
        module_.instructions[jump_pc].operands.jump.target = target;
    }
}

uint32_t Compiler::emitLoadConst(uint16_t dest_reg, const Constant& c) {
    uint32_t const_id = module_.addConstant(c);
    return emit(OpCode::LOAD_CONST, dest_reg, const_id);
}

uint32_t Compiler::emitLoadVar(uint16_t dest_reg, const std::string& var_name) {
    uint32_t string_id = module_.addString(var_name);
    return emit(OpCode::LOAD_VAR, dest_reg, string_id);
}

uint32_t Compiler::emitStoreVar(uint16_t src_reg, const std::string& var_name) {
    uint32_t string_id = module_.addString(var_name);
    return emit(OpCode::STORE_VAR, src_reg, string_id);
}

// ============================================================================
// HELPERS
// ============================================================================

uint16_t Compiler::getOrAllocateRegister(const std::string& name) {
    if (symbols_.exists(name)) {
        return symbols_.getRegister(name);
    }

    uint16_t reg = registers_.allocate();
    symbols_.define(name, reg);
    addRegisterName(reg, name);
    return reg;
}

void Compiler::recordSourceLocation(const std::string& filename, size_t line, size_t column) {
    current_filename_ = filename;
    current_line_ = line;
    current_column_ = column;
}

void Compiler::addRegisterName(uint16_t reg, const std::string& name) {
    module_.debug_info.addRegisterName(reg, name);
}

void Compiler::error(const std::string& message) {
    last_error_ = message;
    success_ = false;

    if (options_.verbose) {
        std::cerr << "Compiler error: " << message << std::endl;
        if (current_line_ > 0) {
            std::cerr << "  at " << current_filename_ << ":"
                      << current_line_ << ":" << current_column_ << std::endl;
        }
    }
}

// ============================================================================
// BYTECODE VERIFIER
// ============================================================================

bool BytecodeVerifier::verify(const BytecodeModule& module) {
    errors_.clear();

    bool ok = true;
    ok &= checkHeader(module);
    ok &= checkRegisterBounds(module);
    ok &= checkJumpTargets(module);
    ok &= checkConstantReferences(module);
    ok &= checkStringReferences(module);

    return ok;
}

bool BytecodeVerifier::checkHeader(const BytecodeModule& module) {
    if (module.magic != BytecodeModule::MAGIC) {
        error("Invalid magic number");
        return false;
    }

    if (module.version_major != BytecodeModule::VERSION_MAJOR) {
        error("Incompatible bytecode version");
        return false;
    }

    return true;
}

bool BytecodeVerifier::checkRegisterBounds(const BytecodeModule& module) {
    constexpr uint16_t MAX_REGISTERS = 256;

    for (size_t i = 0; i < module.instructions.size(); ++i) {
        const auto& instr = module.instructions[i];

        // Check register operands based on instruction type
        if (instr.flags & FLAG_WRITES_REGISTER) {
            if (instr.operands.regs.r0 >= MAX_REGISTERS) {
                error("Register r" + std::to_string(instr.operands.regs.r0) +
                      " out of bounds at PC " + std::to_string(i));
                return false;
            }
        }

        // Check additional registers for three-operand instructions
        switch (instr.opcode) {
            case OpCode::ELEMENTWISE_ADD:
            case OpCode::ELEMENTWISE_SUB:
            case OpCode::ELEMENTWISE_MUL:
            case OpCode::ELEMENTWISE_DIV:
            case OpCode::ELEMENTWISE_POW:
            case OpCode::EINSUM:
            case OpCode::MATMUL:
                if (instr.operands.regs.r1 >= MAX_REGISTERS ||
                    instr.operands.regs.r2 >= MAX_REGISTERS) {
                    error("Register out of bounds at PC " + std::to_string(i));
                    return false;
                }
                break;
            default:
                break;
        }
    }

    return true;
}

bool BytecodeVerifier::checkJumpTargets(const BytecodeModule& module) {
    for (size_t i = 0; i < module.instructions.size(); ++i) {
        const auto& instr = module.instructions[i];

        switch (instr.opcode) {
            case OpCode::JUMP:
            case OpCode::JUMP_IF_ZERO:
            case OpCode::JUMP_IF_NONZERO:
            case OpCode::JUMP_IF_DIRTY: {
                uint32_t target = instr.operands.jump.target;
                if (target >= module.instructions.size()) {
                    error("Jump target " + std::to_string(target) +
                          " out of bounds at PC " + std::to_string(i));
                    return false;
                }
                break;
            }
            default:
                break;
        }
    }

    return true;
}

bool BytecodeVerifier::checkConstantReferences(const BytecodeModule& module) {
    for (size_t i = 0; i < module.instructions.size(); ++i) {
        const auto& instr = module.instructions[i];

        if (instr.opcode == OpCode::LOAD_CONST) {
            uint32_t const_id = instr.operands.reg_imm.imm;
            if (const_id >= module.constants.size()) {
                error("Constant #" + std::to_string(const_id) +
                      " out of bounds at PC " + std::to_string(i));
                return false;
            }
        }
    }

    return true;
}

bool BytecodeVerifier::checkStringReferences(const BytecodeModule& module) {
    for (size_t i = 0; i < module.instructions.size(); ++i) {
        const auto& instr = module.instructions[i];

        if (instr.opcode == OpCode::LOAD_VAR || instr.opcode == OpCode::STORE_VAR) {
            uint32_t string_id = instr.operands.reg_imm.imm;
            if (string_id >= module.strings.size()) {
                error("String @" + std::to_string(string_id) +
                      " out of bounds at PC " + std::to_string(i));
                return false;
            }
        }
    }

    return true;
}

void BytecodeVerifier::error(const std::string& message) {
    errors_.push_back(message);
}

} // namespace tl
