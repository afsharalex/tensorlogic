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
    std::vector<StringLiteral> string_stack;

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

    // Datalog stacks
    std::vector<Identifier> datalog_term_stack;  // Terms for atoms
    std::vector<DatalogAtom> datalog_atom_stack;  // Atoms for rules/queries
    std::vector<std::variant<DatalogAtom, DatalogNegation, DatalogCondition>> datalog_body_stack;  // Body literals for rules

    // Learning directive stacks
    std::vector<DirectiveArg> directive_arg_stack;

    // Top-level
    std::vector<Statement> statements;

    // Temporary state
    std::string current_projection_op;
    std::string current_comparison_op;
    bool current_boolean_value{false};

    // Datalog atom parsing markers (to handle PEG backtracking)
    size_t datalog_atom_term_start_marker{0};  // Marks where current atom's terms start on term_stack
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
template<> struct action<grammar::string_literal>;
template<> struct action<grammar::list_literal>;

// Index and slice actions
template<> struct action<grammar::simple_index>;
template<> struct action<grammar::normalized_index>;
template<> struct action<grammar::slice>;

// Tensor reference actions
template<> struct action<grammar::tensor_ref>;

// Expression actions
template<> struct action<grammar::function_call>;
template<> struct action<grammar::primary_expression>;
template<> struct action<grammar::unary_expression>;
template<> struct action<grammar::power_expression>;
template<> struct action<grammar::multiplicative_expression>;
template<> struct action<grammar::additive_expression>;
template<> struct action<grammar::comparison_op>;
template<> struct action<grammar::comparison_expression>;
template<> struct action<grammar::expression>;

// Guard condition actions
template<> struct action<grammar::guard_factor>;
template<> struct action<grammar::guard_term>;
template<> struct action<grammar::guard_condition>;
template<> struct action<grammar::guarded_clause>;

// Tensor equation actions
template<> struct action<grammar::projection_op>;
template<> struct action<grammar::tensor_equation>;

// Datalog actions
template<> struct action<grammar::uppercase_identifier>;
template<> struct action<grammar::lowercase_identifier>;
template<> struct action<grammar::datalog_term>;
template<> struct action<grammar::datalog_relation_name>;
template<> struct action<grammar::datalog_atom>;
template<> struct action<grammar::datalog_negation>;
template<> struct action<grammar::datalog_comparison>;
template<> struct action<grammar::datalog_body_literal>;
template<> struct action<grammar::datalog_fact>;
template<> struct action<grammar::datalog_rule>;

// File operation actions
template<> struct action<grammar::file_literal>;
template<> struct action<grammar::file_operation>;

// Query and learning directive actions
template<> struct action<grammar::boolean_literal>;
template<> struct action<grammar::directive_arg>;
template<> struct action<grammar::query_directive>;
template<> struct action<grammar::tensor_query>;
template<> struct action<grammar::datalog_query>;

} // namespace tl::actions
