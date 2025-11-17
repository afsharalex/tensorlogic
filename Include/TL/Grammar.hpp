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

// String literal: "..." with escaped characters
// Matches content between double quotes, handling escape sequences
struct string_content : pegtl::until<pegtl::at<pegtl::one<'"'>>, pegtl::sor<
    pegtl::seq<pegtl::one<'\\'>, pegtl::any>,  // Escaped character
    pegtl::any                                   // Any other character
>> {};

struct string_literal : pegtl::seq<
    pegtl::one<'"'>,
    string_content,
    pegtl::one<'"'>
> {};

// ============================================================================
// LIST LITERALS
// ============================================================================

// Forward declaration for recursive array literals
struct list_literal;

// List elements: comma-separated numbers or nested lists
struct list_elements : pegtl::list<pad<pegtl::sor<list_literal, number_literal>>, pegtl::one<','>> {};

// List literal: [elem1, elem2, ...]
// Supports: [1, 2, 3], [[1, 2], [3, 4]], etc.
struct list_literal : pegtl::seq<
    pegtl::one<'['>,
    pad<pegtl::opt<list_elements>>,
    pegtl::one<']'>
> {};

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

// Primary expression: parenthesized expression, function call, tensor reference, list literal, or number literal
// Order matters: try function_call before tensor_ref (both start with identifier)
// Try list_literal before tensor_ref (both can have [ but list starts with [)
struct primary_expression : pegtl::sor<
    pegtl::seq<pegtl::one<'('>, pad<expression>, pad<pegtl::one<')'>>>,
    function_call,
    list_literal,
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

// Comparison operators: <, >, <=, >=, ==, !=
struct comparison_op : pegtl::sor<
    pegtl::string<'<', '='>,  // <=
    pegtl::string<'>', '='>,  // >=
    pegtl::string<'=', '='>,  // ==
    pegtl::string<'!', '='>,  // !=
    pegtl::one<'<'>,          // <
    pegtl::one<'>'>           // >
> {};

// Comparison expression: additive op additive
struct comparison_expression : pegtl::seq<
    additive_expression,
    pegtl::opt<
        pegtl::seq<
            pad<comparison_op>,
            pad<additive_expression>
        >
    >
> {};

// Full expression (includes comparisons)
struct expression : comparison_expression {};

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

// ============================================================================
// GUARD CONDITIONS (for guarded clauses)
// ============================================================================

// Keywords for logical operators
struct kw_and : pegtl::string<'a', 'n', 'd'> {};
struct kw_or : pegtl::string<'o', 'r'> {};
struct kw_not : pegtl::string<'n', 'o', 't'> {};

// Forward declaration
struct guard_condition;

// Guard factor: not factor | (condition) | comparison
struct guard_factor : pegtl::sor<
    pegtl::seq<pad<kw_not>, guard_factor>,  // not X (recursive)
    pegtl::seq<pegtl::one<'('>, pad<guard_condition>, pad<pegtl::one<')'>>>,  // (condition)
    comparison_expression  // X < 10, X == Y, etc.
> {};

// Guard term: factor and factor and...
struct guard_term : pegtl::seq<
    guard_factor,
    pegtl::star<
        pegtl::seq<
            pad<kw_and>,
            pad<guard_factor>
        >
    >
> {};

// Guard condition: term or term or...
struct guard_condition : pegtl::seq<
    guard_term,
    pegtl::star<
        pegtl::seq<
            pad<kw_or>,
            pad<guard_term>
        >
    >
> {};

// Guarded clause: expr : guard | expr
// Example: 1.0 * X[i] : (i < 10) or just 0.1 * X[i]
struct guarded_clause : pegtl::seq<
    rhs_expression,
    pegtl::opt<
        pegtl::seq<
            pad<pegtl::one<':'>>,
            pad<guard_condition>
        >
    >
> {};

// Multiple guarded clauses separated by |
// Example: expr1 : cond1 | expr2 : cond2 | expr3
// Use pad around the | separator to consume whitespace
struct clause_expression : pegtl::list<guarded_clause, pad<pegtl::one<'|'>>> {};

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

// Tensor equation LHS: separate rule to enable marker tracking for PEG backtracking
// This allows us to mark where the actual LHS is, ignoring backtracking artifacts
struct tensor_equation_lhs : tensor_ref {};

// Tensor equation: LHS projection_op RHS
// Example: Y[i,k] = A[i,j] B[j,k]
// Or with guards: Weighted[i] = 1.0 * X[i] : (i < 10) | 0.5 * X[i]
struct tensor_equation : pegtl::seq<
    tensor_equation_lhs,
    pad<projection_op>,
    clause_expression
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

// Relation name: uppercase identifier that starts an atom
// This separate rule allows us to track when atom parsing begins (for PEG backtracking handling)
struct datalog_relation_name : uppercase_identifier {};

// Datalog atom: Relation(term1, term2, ...)
// Examples: Parent(Alice, Bob), Ancestor(x, y)
struct datalog_atom : pegtl::seq<
    datalog_relation_name,  // Relation name must start with uppercase
    pad<pegtl::one<'('>>,
    pegtl::opt<datalog_term_list>,
    pad<pegtl::one<')'>>
> {};

// Datalog fact: atom (with only constants)
// Example: Parent(Alice, Bob)
struct datalog_fact : datalog_atom {};

// Negated literal: not atom, ! atom, or ¬ atom
// Examples: not Friend(x, y), ! Friend(x, y), ¬ Friend(x, y)
struct datalog_negation : pegtl::seq<
    pegtl::sor<
        kw_not,                        // not
        pegtl::one<'!'>,               // !
        pegtl::utf8::one<0x00AC>       // ¬ (Unicode negation symbol U+00AC)
    >,
    pad<datalog_atom>
> {};

// Comparison literal: term op term (for inequality constraints in rules)
// Examples: x != y, x > 5, age >= 18
// Used in rule bodies: Adult(p) <- Age(p, a), a >= 18
struct datalog_comparison : pegtl::seq<
    datalog_term,
    pad<comparison_op>,
    pad<datalog_term>
> {};

// Neurosymbolic condition: expr op expr (for tensor expression comparisons in rules)
// Examples: Emb[x,d]Emb[y,d] > threshold, Score[i] >= 0.5
// Used in neurosymbolic rules: Similar(x,y) <- Emb[x,d]Emb[y,d] > threshold
// NOTE: Must be tried before datalog_atom to avoid ambiguity
struct neurosymbolic_condition : pegtl::seq<
    rhs_expression,
    pad<comparison_op>,
    pad<rhs_expression>
> {};

// Body literal (for rules): negated atoms, neurosymbolic conditions, comparisons, or atoms
// Order matters: try neurosymbolic_condition before datalog_atom to handle tensor expressions
// Examples: Friend(x, y), not Friend(x, y), x != y, Emb[x,d]Emb[y,d] > threshold
struct datalog_body_literal : pegtl::sor<
    datalog_negation,
    neurosymbolic_condition,
    datalog_comparison,
    datalog_atom
> {};

// Body literal list: comma-separated literals
struct datalog_body_list : pegtl::list<pad<datalog_body_literal>, pegtl::one<','>> {};

// Datalog rule: Head <- Body1, Body2, ...
// Example: Ancestor(x, z) <- Parent(x, y), Ancestor(y, z)
struct datalog_rule : pegtl::seq<
    datalog_atom,                    // head
    pad<pegtl::string<'<', '-'>>,   // <-
    datalog_body_list                // body
> {};

// ============================================================================
// LEARNING DIRECTIVES
// ============================================================================

// Boolean literals for directive arguments
struct kw_true : pegtl::string<'t', 'r', 'u', 'e'> {};
struct kw_false : pegtl::string<'f', 'a', 'l', 's', 'e'> {};
struct boolean_literal : pegtl::sor<kw_true, kw_false> {};

// Directive argument value: number, string, or boolean
struct directive_value : pegtl::sor<boolean_literal, number_literal, string_literal> {};

// Directive argument: name=value
// Examples: lr=0.01, epochs=100, verbose=true
struct directive_arg : pegtl::seq<
    identifier,
    pad<pegtl::one<'='>>,
    pad<directive_value>
> {};

// Comma-separated list of directive arguments
struct directive_args : pegtl::list<pad<directive_arg>, pegtl::one<','>> {};

// Query directive: @directive_name(args)
// Examples: @minimize(lr=0.01, epochs=100), @maximize(), @sample(n=1000)
struct query_directive : pegtl::seq<
    pegtl::one<'@'>,
    identifier,  // directive name
    pad<pegtl::one<'('>>,
    pegtl::opt<directive_args>,
    pad<pegtl::one<')'>>
> {};

// ============================================================================
// QUERIES
// ============================================================================

// Datalog query: atom followed by ?
// Example: Ancestor(Alice, x)?
struct datalog_query : pegtl::seq<
    datalog_atom,
    pad<pegtl::one<'?'>>
> {};

// Tensor query: tensor_ref followed by ? and optional directive
// Examples: Loss?, Y[i]?, Loss? @minimize(lr=0.01)
struct tensor_query : pegtl::seq<
    tensor_ref,
    pad<pegtl::one<'?'>>,
    pegtl::opt<pad<query_directive>>
> {};

// ============================================================================
// FILE OPERATIONS
// ============================================================================

// File function call: file("path")
struct file_function : pegtl::seq<
    pegtl::string<'f', 'i', 'l', 'e'>,
    pad<pegtl::one<'('>>,
    pad<string_literal>,
    pad<pegtl::one<')'>>
> {};

// File literal: either "path" or file("path")
struct file_literal : pegtl::sor<file_function, string_literal> {};

// File operation: tensor = file or file = tensor
// Examples: X[i,j] = "data.csv", "output.txt" = Y[i]
struct file_operation : pegtl::sor<
    // tensor = file_literal (read)
    pegtl::seq<tensor_ref, pad<pegtl::one<'='>>, pad<file_literal>>,
    // file_literal = tensor (write)
    pegtl::seq<file_literal, pad<pegtl::one<'='>>, pad<tensor_ref>>
> {};

// ============================================================================
// TOP-LEVEL
// ============================================================================

// Statement: file operations, tensor equations, datalog constructs, or queries
struct statement : pegtl::sor<
    file_operation,    // Try file operations first (has string literal)
    datalog_rule,      // Try rule second (has <-)
    tensor_query,      // Try tensor query third (has ? with optional @)
    datalog_query,     // Try datalog query fourth (has ?)
    datalog_fact,      // Try fact fifth (no special suffix)
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
