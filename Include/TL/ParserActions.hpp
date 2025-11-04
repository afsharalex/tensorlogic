#pragma once

#include "TL/Grammar.hpp"
#include "TL/AST.hpp"
#include <tao/pegtl.hpp>
#include <vector>
#include <string>
#include <memory>

// TensorLogic PEGTL Parser Actions
// This file defines the action system that builds AST nodes during parsing.

namespace tl::actions {

namespace pegtl = tao::pegtl;

// ============================================================================
// PARSE STATE
// ============================================================================

// ParseState: maintains stacks for building AST nodes during parsing
struct ParseState {
    // Literal stacks
    std::vector<Identifier> identifier_stack;
    std::vector<NumberLiteral> number_stack;

    // Index stacks
    std::vector<Index> index_stack;
    std::vector<Slice> slice_stack;
    std::vector<IndexOrSlice> index_or_slice_stack;

    // Expression stack
    std::vector<ExprPtr> expr_stack;

    // Tensor stacks
    std::vector<TensorRef> tensorref_stack;

    // Guard stacks
    std::vector<GuardedClause> clause_stack;

    // Top-level
    std::vector<Statement> statements;

    // Temporary state
    std::string current_projection_op;
};

// ============================================================================
// ACTION SYSTEM
// ============================================================================

// Base action: do nothing by default
template<typename Rule>
struct action : pegtl::nothing<Rule> {};

// Helper function: create SourceLocation from PEGTL position
inline SourceLocation locFrom(const pegtl::position& p) {
    SourceLocation l;
    l.line = p.line;
    l.column = p.column;
    return l;
}

// Forward declarations of action specializations
// Implementations are in Source/Parser.cpp

// Lexical actions
template<> struct action<grammar::identifier>;
template<> struct action<grammar::integer_literal>;
template<> struct action<grammar::float_literal>;

// Index and slice actions
template<> struct action<grammar::simple_index>;
template<> struct action<grammar::slice>;

// Tensor reference actions
template<> struct action<grammar::tensor_ref>;

// Expression actions
template<> struct action<grammar::primary_expression>;
template<> struct action<grammar::guarded_clause>;

// Tensor equation actions
template<> struct action<grammar::projection_op>;
template<> struct action<grammar::tensor_equation>;

} // namespace tl::actions
