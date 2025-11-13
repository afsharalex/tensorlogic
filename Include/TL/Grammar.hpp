#pragma once

#include <tao/pegtl.hpp>

// TensorLogic PEGTL Grammar Definition
// This file contains only the grammar rules - no actions or state.
// For a complete reference, see Docs/PEGTL_GRAMMAR_CURRENT.hpp

namespace tl::grammar {

namespace pegtl = tao::pegtl;

// ============================================================================
// WHITESPACE AND COMMENTS
// ============================================================================

// Line comment: // ... until end of line
struct line_comment : pegtl::seq<
    pegtl::two<'/'>,
    pegtl::until<pegtl::eolf>
> {};

// Block comment: /* ... */
struct block_comment : pegtl::seq<
    pegtl::string<'/', '*'>,
    pegtl::until<pegtl::string<'*', '/'>>
> {};

// Whitespace element: space, tab, newline, or comment
struct ws_element : pegtl::sor<
    pegtl::one<' '>,
    pegtl::one<'\t'>,
    pegtl::one<'\n'>,
    pegtl::one<'\r'>,
    line_comment,
    block_comment
> {};

// Whitespace includes spaces, tabs, newlines, carriage returns, and comments
struct ws : pegtl::star<ws_element> {};

// Horizontal whitespace (space and tab only, not newlines or comments)
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

// Index identifier: identifier pattern for indices (no action to avoid stack pollution)
// We match the identifier pattern directly instead of using identifier
struct index_identifier : pegtl::seq<
    pegtl::alpha,
    pegtl::star<pegtl::sor<pegtl::alnum, pegtl::one<'_'>>>
> {};

// Index integer: integer pattern for indices (no action to avoid expr_stack pollution)
// We match the integer pattern directly instead of using integer_literal
struct index_integer : pegtl::plus<pegtl::digit> {};

// Simple index: index identifier or index integer
// Examples: i, j, 0, 1, 42
struct simple_index : pegtl::sor<index_identifier, index_integer> {};

// Normalized index: identifier pattern followed by dot (for softmax normalization)
// Examples: i., j., k.
// Used in attention mechanisms: Attn[q, k.] = Scores[q, k]
// We match the identifier pattern directly (not using the identifier rule) to avoid action side effects
struct normalized_index : pegtl::seq<
    pegtl::alpha,  // Start with a letter
    pegtl::star<pegtl::sor<pegtl::alnum, pegtl::one<'_'>>>,  // Followed by alphanumeric or underscore
    pegtl::one<'.'>  // Followed by dot
> {};

// Virtual index: *t, *t+1, *t-1 (for recurrent operations)
// Examples: *t, *t+1, *t-1
// Used in RNNs: State[i, *t+1] = relu(W[i,j] State[j, *t] + Input[i, t])
struct virtual_index_offset : pegtl::seq<
    pegtl::one<'+', '-'>,
    index_integer
> {};

struct virtual_index : pegtl::seq<
    pegtl::one<'*'>,
    index_identifier,
    pegtl::opt<virtual_index_offset>
> {};

// Slice: [start]:[end][:step]
// Examples: :, 0:10, 0:10:2, :10, 0:, ::2
// Use index_integer instead of integer_literal to avoid expr_stack pollution
struct slice : pegtl::seq<
    pegtl::opt<index_integer>,
    pegtl::one<':'>,
    pegtl::opt<index_integer>,
    pegtl::opt<pegtl::seq<pegtl::one<':'>, index_integer>>
> {};

// Index or slice (order matters - try virtual_index and normalized_index first)
struct index_or_slice : pegtl::sor<virtual_index, normalized_index, slice, simple_index> {};

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

// Forward declaration for recursive expressions
struct expression;

// Function call: function_name(arg1, arg2, ...)
// Examples: relu(X), softmax(Y[i]), sigmoid(W[i,j] * X[j])
struct function_call : pegtl::seq<
    identifier,
    pad<pegtl::one<'('>>,
    pegtl::opt<pegtl::list<pad<expression>, pegtl::one<','>>>,
    pad<pegtl::one<')'>>
> {};

// Primary expression: parenthesized expression, function call, tensor reference, or number literal
// Order matters: try function_call before tensor_ref (both start with identifier)
struct primary_expression : pegtl::sor<
    pegtl::seq<pegtl::one<'('>, pad<expression>, pad<pegtl::one<')'>>>,
    function_call,
    tensor_ref,
    number_literal
> {};

// Unary expression: -X or just primary
struct unary_expression : pegtl::sor<
    pegtl::seq<pad<pegtl::one<'-'>>, unary_expression>,
    primary_expression
> {};

// Power (right-associative): X^2, X^Y^Z = X^(Y^Z)
struct power_expression : pegtl::seq<
    unary_expression,
    pegtl::opt<
        pegtl::seq<
            pad<pegtl::one<'^'>>,
            power_expression  // Right-associative recursion
        >
    >
> {};

// Multiplicative: *, /, %
struct multiplicative_expression : pegtl::seq<
    power_expression,
    pegtl::star<
        pegtl::seq<
            pad<pegtl::one<'*', '/', '%'>>,
            pad<power_expression>
        >
    >
> {};

// Additive: +, -
struct additive_expression : pegtl::seq<
    multiplicative_expression,
    pegtl::star<
        pegtl::seq<
            pad<pegtl::one<'+', '-'>>,
            pad<multiplicative_expression>
        >
    >
> {};

// Full expression (currently same as additive, will add comparisons later)
struct expression : additive_expression {};

// RHS expression with implicit multiplication support
// A B means A * B (space-separated without operator)
struct rhs_expression : pegtl::seq<
    hws,
    expression,
    pegtl::star<pegtl::seq<
        pegtl::plus<pegtl::sor<pegtl::one<' '>, pegtl::one<'\t'>>>,  // At least one space/tab
        expression
    >>,
    hws
> {};

// Guarded clause (currently just rhs_expression)
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
// DATALOG CONSTRUCTS
// ============================================================================

// Uppercase identifier (for relations and constants)
struct uppercase_identifier : pegtl::seq<
    pegtl::upper,
    pegtl::star<pegtl::sor<pegtl::alnum, pegtl::one<'_'>>>
> {};

// Lowercase identifier (for variables)
struct lowercase_identifier : pegtl::seq<
    pegtl::lower,
    pegtl::star<pegtl::sor<pegtl::alnum, pegtl::one<'_'>>>
> {};

// Datalog term: variable (lowercase) or constant (uppercase/number)
// Variables: x, y, myVar
// Constants: Alice, Bob, 42, 3.14
struct datalog_term : pegtl::sor<
    lowercase_identifier,  // variables
    uppercase_identifier,  // constants
    number_literal        // numeric constants
> {};

// Term list: comma-separated terms
struct datalog_term_list : pegtl::list<pad<datalog_term>, pegtl::one<','>> {};

// Datalog atom: Relation(term1, term2, ...)
// Examples: Parent(Alice, Bob), Ancestor(x, y)
struct datalog_atom : pegtl::seq<
    uppercase_identifier,  // Relation name must start with uppercase
    pad<pegtl::one<'('>>,
    pegtl::opt<datalog_term_list>,
    pad<pegtl::one<')'>>
> {};

// Datalog fact: atom (with only constants)
// Example: Parent(Alice, Bob)
struct datalog_fact : datalog_atom {};

// Body literal (for rules): currently just atoms
// Future: will support negation and comparisons
struct datalog_body_literal : datalog_atom {};

// Body literal list: comma-separated literals
struct datalog_body_list : pegtl::list<pad<datalog_body_literal>, pegtl::one<','>> {};

// Datalog rule: Head <- Body1, Body2, ...
// Example: Ancestor(x, z) <- Parent(x, y), Ancestor(y, z)
struct datalog_rule : pegtl::seq<
    datalog_atom,                    // head
    pad<pegtl::string<'<', '-'>>,   // <-
    datalog_body_list                // body
> {};

// Query: atom followed by ?
// Example: Ancestor(Alice, x)?
struct datalog_query : pegtl::seq<
    datalog_atom,
    pad<pegtl::one<'?'>>
> {};

// ============================================================================
// TOP-LEVEL
// ============================================================================

// Statement: tensor equation, datalog fact, rule, or query
struct statement : pegtl::sor<
    datalog_rule,      // Try rule first (has <-)
    datalog_query,     // Try query second (has ?)
    datalog_fact,      // Try fact third (no special suffix)
    tensor_equation    // Finally try tensor equation
> {};

// Program: zero or more statements separated by whitespace
struct program : pegtl::seq<
    pegtl::star<pegtl::seq<ws, statement, ws>>,
    pegtl::eof
> {};

// Grammar entry point
struct grammar : program {};

} // namespace tl::grammar
