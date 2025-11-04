#pragma once

#include <tao/pegtl.hpp>

// TensorLogic PEGTL Grammar Definition
// This file contains only the grammar rules - no actions or state.
// For a complete reference, see Docs/PEGTL_GRAMMAR_CURRENT.hpp

namespace tl::grammar {

namespace pegtl = tao::pegtl;

// ============================================================================
// WHITESPACE
// ============================================================================

// Whitespace includes spaces, tabs, newlines, and carriage returns
struct ws : pegtl::star<pegtl::sor<
    pegtl::one<' '>,
    pegtl::one<'\t'>,
    pegtl::one<'\n'>,
    pegtl::one<'\r'>
>> {};

// Horizontal whitespace (space and tab only, not newlines)
// Used for implicit multiplication to avoid consuming statement separators
struct hws : pegtl::star<pegtl::sor<
    pegtl::one<' '>,
    pegtl::one<'\t'>
>> {};

// Padded rule: ws + Rule + ws
template<typename Rule>
struct pad : pegtl::seq<ws, Rule, ws> {};

// ============================================================================
// LEXICAL ELEMENTS
// ============================================================================

// Identifiers: start with letter, followed by alphanumeric or underscore
struct identifier : pegtl::seq<
    pegtl::alpha,
    pegtl::star<pegtl::sor<pegtl::alnum, pegtl::one<'_'>>>
> {};

// Integer literals: one or more digits
struct integer_literal : pegtl::plus<pegtl::digit> {};

// Float literals: digits.digits (simplified - no scientific notation yet)
struct float_literal : pegtl::seq<
    pegtl::plus<pegtl::digit>,
    pegtl::one<'.'>,
    pegtl::star<pegtl::digit>
> {};

// Number literal: float or integer (order matters - try float first)
struct number_literal : pegtl::sor<float_literal, integer_literal> {};

// ============================================================================
// INDICES AND SLICES
// ============================================================================

// Simple index: identifier or integer literal
// Examples: i, j, 0, 1, 42
struct simple_index : pegtl::sor<identifier, integer_literal> {};

// Slice: [start]:[end][:step]
// Examples: :, 0:10, 0:10:2, :10, 0:, ::2
struct slice : pegtl::seq<
    pegtl::opt<integer_literal>,
    pegtl::one<':'>,
    pegtl::opt<integer_literal>,
    pegtl::opt<pegtl::seq<pegtl::one<':'>, integer_literal>>
> {};

// Index or slice (order matters - try slice first to match the colon)
struct index_or_slice : pegtl::sor<slice, simple_index> {};

// Comma-separated list of indices/slices
struct index_list : pegtl::list<pad<index_or_slice>, pegtl::one<','>> {};

// ============================================================================
// TENSOR REFERENCES
// ============================================================================

// Tensor reference: identifier with optional indices
// Examples: X, Y[i], C[i,k], Y[0:10], M[i,:]
struct tensor_ref : pegtl::seq<
    identifier,
    pegtl::opt<pegtl::seq<
        pegtl::one<'['>,
        index_list,
        pegtl::one<']'>
    >>
> {};

// ============================================================================
// EXPRESSIONS
// ============================================================================

// Primary expression: tensor reference or number literal
struct primary_expression : pegtl::sor<tensor_ref, number_literal> {};

// RHS expression: one or more primary expressions
// Implicit multiplication: A B C means A * B * C
// Multiple expressions separated by at least one horizontal space (not newline!)
struct rhs_expression : pegtl::seq<
    hws,
    primary_expression,
    pegtl::star<pegtl::seq<
        pegtl::plus<pegtl::sor<pegtl::one<' '>, pegtl::one<'\t'>>>,  // At least one space/tab
        primary_expression
    >>,
    hws
> {};

// Guarded clause (currently just an alias for rhs_expression)
// Future: will support guards like (expr : condition)
struct guarded_clause : rhs_expression {};

// ============================================================================
// TENSOR EQUATIONS
// ============================================================================

// Projection operators: =, +=, max=, min=, avg=
struct projection_op : pegtl::sor<
    pegtl::string<'m', 'a', 'x', '='>,
    pegtl::string<'m', 'i', 'n', '='>,
    pegtl::string<'a', 'v', 'g', '='>,
    pegtl::string<'+', '='>,
    pegtl::one<'='>
> {};

// Tensor equation: LHS projection_op RHS
// Example: Y[i,k] = A[i,j] B[j,k]
struct tensor_equation : pegtl::seq<
    tensor_ref,
    pad<projection_op>,
    guarded_clause
> {};

// ============================================================================
// TOP-LEVEL
// ============================================================================

// Statement (currently only tensor equations)
// Future: will include datalog facts, rules, queries, file operations
struct statement : pegtl::sor<tensor_equation> {};

// Program: zero or more statements separated by whitespace
struct program : pegtl::seq<
    pegtl::star<pegtl::seq<ws, statement, ws>>,
    pegtl::eof
> {};

// Grammar entry point
struct grammar : program {};

} // namespace tl::grammar
