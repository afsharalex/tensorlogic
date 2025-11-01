#pragma once

#include <string>
#include <variant>
#include <vector>

namespace tl {

struct SourceLocation {
    size_t line{1};
    size_t column{1};
};

struct Identifier {
    std::string name;
    SourceLocation loc{};
};

struct NumberLiteral {
    std::string text; // keep original text for now
    SourceLocation loc{};
};

struct StringLiteral {
    std::string text; // unescaped content
    SourceLocation loc{};
};

// Virtual index for recurrent operations: *t, *t+1, *t-1
struct VirtualIndex {
    Identifier name;     // The identifier after *, e.g., 't' in '*t'
    int offset{0};       // Offset: +1, 0, or -1 (for *t+1, *t, *t-1)
    SourceLocation loc{};
};

struct Index {
    // Can be a simple identifier, integer literal, or virtual index
    std::variant<Identifier, NumberLiteral, VirtualIndex> value;
    bool normalized{false};  // True if followed by '.' (e.g., "i." for softmax normalization)
    SourceLocation loc{};
};

// Slice for tensor slicing: start:end:step, :end, start:, :, etc.
struct Slice {
    std::optional<NumberLiteral> start;  // Empty means "from beginning"
    std::optional<NumberLiteral> end;    // Empty means "to end"
    std::optional<NumberLiteral> step;   // Empty means step=1
    SourceLocation loc{};

    // Check if this is a complete slice (:)
    bool isFullSlice() const {
        return !start.has_value() && !end.has_value() && !step.has_value();
    }
};

// IndexOrSlice: either a regular index or a slice
struct IndexOrSlice {
    std::variant<Index, Slice> value;
    SourceLocation loc{};
};

struct TensorRef {
    Identifier name;
    std::vector<IndexOrSlice> indices; // empty means scalar, can mix Index and Slice
    SourceLocation loc{};
};

// Very early and minimal expression model
struct Expr;
using ExprPtr = std::shared_ptr<Expr>;

struct ExprTensorRef { TensorRef ref; };
struct ExprNumber { NumberLiteral literal; };
struct ExprString { StringLiteral literal; };
// List literal: elements may be numbers or nested lists (n-dimensional)
struct ExprList { std::vector<ExprPtr> elements; };
struct ExprParen { ExprPtr inner; };
struct ExprCall { Identifier func; std::vector<ExprPtr> args; };
struct ExprBinary {
    enum class Op {
        Add, Sub, Mul, Div, Mod,  // Arithmetic operators
        Lt, Le, Gt, Ge, Eq, Ne,  // Comparison operators: <, <=, >, >=, ==, !=
        And, Or                   // Logical operators
    };
    Op op{Op::Add};
    ExprPtr lhs;
    ExprPtr rhs;
};

struct ExprUnary {
    enum class Op { Neg, Not };  // Unary minus and logical not
    Op op;
    ExprPtr operand;
};

struct Expr {
    SourceLocation loc{};
    std::variant<ExprTensorRef, ExprNumber, ExprString, ExprList, ExprParen, ExprCall, ExprBinary, ExprUnary> node;
};

// Datalog structures
struct DatalogAtom {
    Identifier relation; // Must start uppercase by grammar
    // Terms: variable (lowercase Identifier) or constant (StringLiteral for Uppercase id/int/string)
    std::vector<std::variant<Identifier, StringLiteral>> terms;
    SourceLocation loc{};
};

// Negated Datalog atom: represents `not Atom(...)` in rule/query bodies
struct DatalogNegation {
    DatalogAtom atom;
    SourceLocation loc{};
};

// Guarded clause: expression with optional guard condition
struct GuardedClause {
    ExprPtr expr;                  // The expression to evaluate
    std::optional<ExprPtr> guard;  // Optional guard condition (if present, acts as mask)
    SourceLocation loc{};
};

// Statements
struct TensorEquation {
    TensorRef lhs;           // A[i] or scalar A
    std::string projection;  // currently just "="; keep as text for future (+=, max=, ...)
    std::vector<GuardedClause> clauses;  // Multiple guarded clauses (all contribute additively)
    SourceLocation loc{};
};

    struct FixedPointLoop {
        TensorEquation equation; // The self-recursive equation
        std::string monitoredTensor; // Tensor to check for convergence (e.g. "x")
        SourceLocation loc;
    };

struct FileOperation {
    // Either tensor = file("path") or file("path") = tensor
    // We normalize into lhsIsTensor flag
    bool lhsIsTensor{true};
    TensorRef tensor; // valid if used on that side
    StringLiteral file; // we only support string literal form initially
    SourceLocation loc{};
};

// Forward declaration to allow Query to reference DatalogCondition
struct DatalogCondition;

struct Query {
    // Support queries over tensor refs and Datalog atoms
    // Additionally, for Datalog queries, we may allow a conjunction of atoms and comparisons.
    std::variant<TensorRef, DatalogAtom> target;
    // If non-empty, represents a conjunctive query of atoms/conditions; the first element is usually the same as `target` when `target` holds a DatalogAtom.
    std::vector<std::variant<DatalogAtom, DatalogNegation, DatalogCondition>> body;
    SourceLocation loc{};
};

struct DatalogFact { Identifier relation; std::vector<StringLiteral> constants; SourceLocation loc{}; };

struct DatalogCondition { ExprPtr lhs; std::string op; ExprPtr rhs; SourceLocation loc{}; };

struct DatalogRule { DatalogAtom head; std::vector<std::variant<DatalogAtom, DatalogNegation, DatalogCondition>> body; SourceLocation loc{}; };

using Statement = std::variant<TensorEquation, FileOperation, Query, DatalogFact, DatalogRule, FixedPointLoop>;

struct Program {
    std::vector<Statement> statements;
};

// Simple printable summary helpers (debug)
std::string toString(const Identifier& id);
std::string toString(const TensorRef& ref);
std::string toString(const Expr& e);
std::string toString(const Statement& st);

} // namespace tl
