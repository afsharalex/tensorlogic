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

struct Index {
    // For now only a simple identifier or integer index
    std::variant<Identifier, NumberLiteral> value;
    SourceLocation loc{};
};

struct TensorRef {
    Identifier name;
    std::vector<Index> indices; // empty means scalar
    SourceLocation loc{};
};

// Very early and minimal expression model
struct Expr;
using ExprPtr = std::shared_ptr<Expr>;

struct ExprTensorRef { TensorRef ref; };
struct ExprNumber { NumberLiteral literal; };
struct ExprString { StringLiteral literal; };
// Minimal 1D numeric list literal support for Programs/03_*
struct ExprList { std::vector<NumberLiteral> elements; };
struct ExprParen { ExprPtr inner; };
struct ExprCall { Identifier func; std::vector<ExprPtr> args; };
struct ExprBinary {
    enum class Op { Add, Sub, Mul, Div };
    Op op{Op::Add};
    ExprPtr lhs;
    ExprPtr rhs;
};

struct Expr {
    SourceLocation loc{};
    std::variant<ExprTensorRef, ExprNumber, ExprString, ExprList, ExprParen, ExprCall, ExprBinary> node;
};

// Datalog structures
struct DatalogAtom {
    Identifier relation; // Must start uppercase by grammar
    // Terms: variable (lowercase Identifier) or constant (StringLiteral for Uppercase id/int/string)
    std::vector<std::variant<Identifier, StringLiteral>> terms;
    SourceLocation loc{};
};

// Statements
struct TensorEquation {
    TensorRef lhs;           // A[i] or scalar A
    std::string projection;  // currently just "="; keep as text for future (+=, max=, ...)
    ExprPtr rhs;
    SourceLocation loc{};
};

struct FileOperation {
    // Either tensor = file("path") or file("path") = tensor
    // We normalize into lhsIsTensor flag
    bool lhsIsTensor{true};
    TensorRef tensor; // valid if used on that side
    StringLiteral file; // we only support string literal form initially
    SourceLocation loc{};
};

struct Query {
    // Support queries over tensor refs and Datalog atoms
    std::variant<TensorRef, DatalogAtom> target;
    SourceLocation loc{};
};

struct DatalogFact { Identifier relation; std::vector<StringLiteral> constants; SourceLocation loc{}; };

struct DatalogCondition { ExprPtr lhs; std::string op; ExprPtr rhs; SourceLocation loc{}; };

struct DatalogRule { DatalogAtom head; std::vector<std::variant<DatalogAtom, DatalogCondition>> body; SourceLocation loc{}; };

using Statement = std::variant<TensorEquation, FileOperation, Query, DatalogFact, DatalogRule>;

struct Program {
    std::vector<Statement> statements;
};

// Simple printable summary helpers (debug)
std::string toString(const Identifier& id);
std::string toString(const TensorRef& ref);
std::string toString(const Expr& e);
std::string toString(const Statement& st);

} // namespace tl
