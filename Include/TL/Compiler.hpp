// Include/TL/Compiler.hpp
// AST to Bytecode compiler for TensorLogic

#ifndef TL_COMPILER_HPP
#define TL_COMPILER_HPP

#include "TL/AST.hpp"
#include "TL/Bytecode.hpp"
#include <map>
#include <set>
#include <vector>
#include <string>
#include <optional>

namespace tl {

// Forward declarations
class Environment;

// ============================================================================
// COMPILER OPTIONS
// ============================================================================

struct CompilerOptions {
    bool optimize = true;                   ///< Enable optimization passes
    int optimization_level = 2;             ///< 0-3 (like -O0 to -O3)
    bool generate_debug_info = true;        ///< Include source locations
    bool enable_type_checking = false;      ///< Type check before compiling (future)
    bool enable_constant_folding = true;    ///< Fold constants at compile time
    bool enable_dead_code_elimination = true; ///< Remove unused code
    bool enable_cse = true;                 ///< Common subexpression elimination
    bool enable_fusion = true;              ///< Fuse operations
    bool verbose = false;                   ///< Print compilation messages
};

// ============================================================================
// REGISTER ALLOCATOR
// ============================================================================

/// Simple register allocator with free list
class RegisterAllocator {
public:
    /// Allocate a register
    uint16_t allocate();

    /// Free a register
    void free(uint16_t reg);

    /// Mark a register as live for a range [start, end]
    void setLiveRange(uint16_t reg, uint32_t start, uint32_t end);

    /// Get the maximum register used
    uint16_t getMaxRegister() const { return next_register_; }

    /// Reset allocator
    void reset() {
        next_register_ = 0;
        free_list_.clear();
        live_ranges_.clear();
    }

private:
    uint16_t next_register_ = 0;
    std::vector<uint16_t> free_list_;
    std::map<uint16_t, std::pair<uint32_t, uint32_t>> live_ranges_;
};

// ============================================================================
// SYMBOL TABLE
// ============================================================================

/// Symbol table entry
struct Symbol {
    std::string name;
    uint16_t reg;                   ///< Assigned register
    bool is_parameter = false;      ///< True if function parameter
    bool is_constant = false;       ///< True if compile-time constant
    std::optional<Constant> constant_value; ///< Value if constant
};

/// Symbol table for variables
class SymbolTable {
public:
    /// Look up a symbol
    std::optional<Symbol> lookup(const std::string& name) const;

    /// Define a new symbol
    void define(const std::string& name, uint16_t reg);

    /// Check if symbol exists
    bool exists(const std::string& name) const;

    /// Get register for symbol (throws if not found)
    uint16_t getRegister(const std::string& name) const;

    /// Mark symbol as constant with value
    void markConstant(const std::string& name, const Constant& value);

    /// Push new scope
    void pushScope();

    /// Pop scope
    void popScope();

private:
    std::vector<std::map<std::string, Symbol>> scopes_;
    std::map<std::string, Symbol> current_scope_;
};

// ============================================================================
// COMPILER
// ============================================================================

class Compiler {
public:
    /// Construct compiler with options
    explicit Compiler(const CompilerOptions& opts = CompilerOptions());

    /// Compile a program to bytecode
    BytecodeModule compile(const Program& program);

    /// Get/set options
    void setOptions(const CompilerOptions& opts) { options_ = opts; }
    const CompilerOptions& getOptions() const { return options_; }

    /// Get last error message
    const std::string& getLastError() const { return last_error_; }

private:
    // ========================================================================
    // PHASE 1: SEMANTIC ANALYSIS
    // ========================================================================

    /// Analyze semantics of entire program
    void analyzeSemantics(const Program& program);

    /// Check types (future implementation)
    void checkTypes(const Statement& stmt);

    /// Resolve names (ensure variables are defined before use)
    void resolveNames(const Statement& stmt);

    // ========================================================================
    // PHASE 2: CODE GENERATION
    // ========================================================================

    /// Generate bytecode for a statement
    void compileStatement(const Statement& stmt);

    /// Compile a tensor equation
    void compileEquation(const TensorEquation& eq);

    /// Compile expression and return register containing result
    uint16_t compileExpression(const Expr& expr);

    /// Compile a Datalog fact
    void compileDatalogFact(const DatalogFact& fact);

    /// Compile a Datalog rule
    void compileDatalogRule(const DatalogRule& rule);

    /// Compile a query
    void compileQuery(const Query& query);

    /// Compile a file operation
    void compileFileOperation(const FileOperation& op);

    // ========================================================================
    // EXPRESSION COMPILATION
    // ========================================================================

    /// Compile arithmetic expression (add, sub, mul, div, pow)
    uint16_t compileArithmetic(ExprBinary::Op op, const Expr& left, const Expr& right);

    /// Compile unary expression (negation)
    uint16_t compileUnary(ExprUnary::Op op, const Expr& expr);

    /// Compile function call (activations, etc.)
    uint16_t compileFunctionCall(const ExprCall& call);

    /// Compile tensor reference (variable lookup or indexed access)
    uint16_t compileTensorRef(const TensorRef& ref);

    /// Compile number literal
    uint16_t compileNumberLiteral(const NumberLiteral& lit);

    /// Compile list literal (array)
    uint16_t compileListLiteral(const ExprList& lit);

    // ========================================================================
    // INDEX COMPILATION
    // ========================================================================

    /// Analyze indices and determine if this is an Einstein summation
    bool isEinsumPattern(const TensorRef& lhs, const Expr& rhs);

    /// Compile Einstein summation
    void compileEinsum(const TensorRef& lhs, const Expr& rhs);

    /// Build einsum specification from indices
    EinsumSpec buildEinsumSpec(const std::vector<IndexOrSlice>& lhs_indices,
                                const Expr& rhs);

    /// Extract indices from expression
    void extractIndices(const Expr& expr, std::set<std::string>& indices);

    // ========================================================================
    // DATALOG COMPILATION
    // ========================================================================

    /// Compile Datalog rule body (joins, filters, etc.)
    /// Body literals are DatalogAtom, DatalogNegation, or DatalogCondition
    uint16_t compileRuleBody(const std::vector<std::variant<DatalogAtom, DatalogNegation, DatalogCondition>>& body);

    /// Compile a join operation
    uint16_t compileJoin(uint16_t left_reg, uint16_t right_reg,
                          const std::vector<int>& join_columns);

    // ========================================================================
    // OPTIMIZATION PASSES (PHASE 3)
    // ========================================================================

    /// Run all optimization passes
    void runOptimizationPasses();

    /// Constant folding
    void constantFolding();

    /// Dead code elimination
    void deadCodeElimination();

    /// Common subexpression elimination
    void commonSubexpressionElimination();

    /// Operation fusion
    void fusionOptimization();

    // ========================================================================
    // BYTECODE EMISSION
    // ========================================================================

    /// Emit an instruction (returns PC of emitted instruction)
    uint32_t emit(const Instruction& instr);

    /// Emit with three registers
    uint32_t emit(OpCode op, uint16_t r0, uint16_t r1 = 0, uint16_t r2 = 0);

    /// Emit with register + immediate
    uint32_t emit(OpCode op, uint16_t reg, uint32_t imm);

    /// Emit jump instruction
    uint32_t emitJump(OpCode op, uint32_t target);

    /// Patch a jump target (for forward jumps)
    void patchJump(uint32_t jump_pc, uint32_t target);

    /// Emit LOAD_CONST instruction
    uint32_t emitLoadConst(uint16_t dest_reg, const Constant& c);

    /// Emit LOAD_VAR instruction
    uint32_t emitLoadVar(uint16_t dest_reg, const std::string& var_name);

    /// Emit STORE_VAR instruction
    uint32_t emitStoreVar(uint16_t src_reg, const std::string& var_name);

    // ========================================================================
    // HELPERS
    // ========================================================================

    /// Get or allocate register for variable
    uint16_t getOrAllocateRegister(const std::string& name);

    /// Record source location for current instruction (from AST SourceLocation)
    void recordSourceLocation(const std::string& filename, size_t line, size_t column);

    /// Add debug info for register
    void addRegisterName(uint16_t reg, const std::string& name);

    /// Get current PC
    uint32_t getCurrentPC() const { return module_.getCurrentPC(); }

    /// Report error
    void error(const std::string& message);

    // ========================================================================
    // STATE
    // ========================================================================

    CompilerOptions options_;
    BytecodeModule module_;
    SymbolTable symbols_;
    RegisterAllocator registers_;

    /// Current source location (for error reporting) - from AST
    std::string current_filename_;
    size_t current_line_ = 0;
    size_t current_column_ = 0;

    /// Label table (for jump targets)
    std::map<std::string, uint32_t> labels_;
    std::vector<std::pair<uint32_t, std::string>> unresolved_jumps_;

    /// Constant value tracking (for constant folding)
    std::map<uint16_t, Constant> register_constants_;

    /// Last error message
    std::string last_error_;

    /// Compilation succeeded
    bool success_ = true;
};

// ============================================================================
// BYTECODE VERIFIER
// ============================================================================

/// Verifies bytecode for correctness and safety
class BytecodeVerifier {
public:
    /// Verify a bytecode module
    bool verify(const BytecodeModule& module);

    /// Get verification errors
    const std::vector<std::string>& getErrors() const { return errors_; }

private:
    /// Check magic number and version
    bool checkHeader(const BytecodeModule& module);

    /// Check all register IDs are in bounds
    bool checkRegisterBounds(const BytecodeModule& module);

    /// Check all jump targets are valid
    bool checkJumpTargets(const BytecodeModule& module);

    /// Check all constant pool references are valid
    bool checkConstantReferences(const BytecodeModule& module);

    /// Check all string pool references are valid
    bool checkStringReferences(const BytecodeModule& module);

    /// Report verification error
    void error(const std::string& message);

    std::vector<std::string> errors_;
};

} // namespace tl

#endif // TL_COMPILER_HPP
