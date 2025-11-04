#pragma once

#include "TL/AST.hpp"
#include <string>
#include <string_view>
#include <stdexcept>
#include <tao/pegtl.hpp>

namespace tl {

struct ParseError final : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// Parse a program from a string (full file content)
Program parseProgram(std::string_view source);

// Convenience: parse a file from disk
Program parseFile(const std::string& path);

// ===========================================================================
// PEGTL GRAMMAR RULES
// ===========================================================================

namespace grammar {

namespace pegtl = tao::pegtl;

// ===========================================================================
// WHITESPACE AND COMMENTS
// ===========================================================================

struct line_comment : pegtl::seq<
    pegtl::two<'/'>,
    pegtl::until<pegtl::eolf>
> {};

struct block_comment : pegtl::seq<
    pegtl::string<'/', '*'>,
    pegtl::until<pegtl::string<'*', '/'>>
> {};

struct ws : pegtl::star<
    pegtl::sor<
        pegtl::space,
        line_comment,
        block_comment
    >
> {};

// Helper: padded rule (rule surrounded by whitespace)
template<typename Rule>
struct pad : pegtl::pad<Rule, ws> {};

// ===========================================================================
// LEXICAL ELEMENTS
// ===========================================================================

// Identifiers
struct lowercase_letter : pegtl::lower {};
struct uppercase_letter : pegtl::upper {};

struct lowercase_identifier : pegtl::seq<
    lowercase_letter,
    pegtl::star<pegtl::sor<pegtl::alnum, pegtl::one<'_'>>>
> {};

struct uppercase_identifier : pegtl::seq<
    uppercase_letter,
    pegtl::star<pegtl::sor<pegtl::alnum, pegtl::one<'_'>>>
> {};

struct identifier : pegtl::seq<
    pegtl::alpha,
    pegtl::star<pegtl::sor<pegtl::alnum, pegtl::one<'_'>>>
> {};

// Numbers
struct sign : pegtl::one<'+', '-'> {};

struct integer_literal : pegtl::seq<
    pegtl::opt<sign>,
    pegtl::plus<pegtl::digit>
> {};

struct float_literal : pegtl::sor<
    // Scientific notation: 1.5e10, 1e-5
    pegtl::seq<
        pegtl::opt<sign>,
        pegtl::plus<pegtl::digit>,
        pegtl::opt<pegtl::seq<pegtl::one<'.'>, pegtl::star<pegtl::digit>>>,
        pegtl::one<'e', 'E'>,
        pegtl::opt<sign>,
        pegtl::plus<pegtl::digit>
    >,
    // Regular float: 1.5, .5, 1.
    pegtl::seq<
        pegtl::opt<sign>,
        pegtl::sor<
            pegtl::seq<pegtl::plus<pegtl::digit>, pegtl::one<'.'>, pegtl::star<pegtl::digit>>,
            pegtl::seq<pegtl::star<pegtl::digit>, pegtl::one<'.'>, pegtl::plus<pegtl::digit>>
        >
    >
> {};

struct number : pegtl::sor<float_literal, integer_literal> {};

// Strings
struct escape_sequence : pegtl::seq<
    pegtl::one<'\\'>,
    pegtl::one<'n', 't', 'r', '\\', '"', '\'', '0'>
> {};

struct string_char : pegtl::sor<
    escape_sequence,
    pegtl::utf8::not_one<'"', '\\'>
> {};

struct string_literal : pegtl::seq<
    pegtl::one<'"'>,
    pegtl::star<string_char>,
    pegtl::must<pegtl::one<'"'>>
> {};

// Booleans
struct boolean_literal : pegtl::sor<
    pegtl::string<'t', 'r', 'u', 'e'>,
    pegtl::string<'f', 'a', 'l', 's', 'e'>,
    pegtl::string<'T', 'r', 'u', 'e'>,
    pegtl::string<'F', 'a', 'l', 's', 'e'>
> {};

// Keywords
struct kw_and : pegtl::string<'a', 'n', 'd'> {};
struct kw_or : pegtl::string<'o', 'r'> {};
struct kw_not : pegtl::string<'n', 'o', 't'> {};
struct kw_file : pegtl::string<'f', 'i', 'l', 'e'> {};

// Built-in functions
struct builtin_function : pegtl::sor<
    pegtl::string<'r', 'e', 'l', 'u'>,
    pegtl::string<'s', 'i', 'g', 'm', 'o', 'i', 'd'>,
    pegtl::string<'t', 'a', 'n', 'h'>,
    pegtl::string<'s', 'o', 'f', 't', 'm', 'a', 'x'>,
    pegtl::string<'s', 't', 'e', 'p'>,
    pegtl::string<'e', 'x', 'p'>,
    pegtl::string<'l', 'o', 'g'>,
    pegtl::string<'s', 'q', 'r', 't'>,
    pegtl::string<'a', 'b', 's'>,
    pegtl::string<'s', 'i', 'n'>,
    pegtl::string<'c', 'o', 's'>,
    pegtl::string<'t', 'a', 'n'>,
    pegtl::string<'l', 'n', 'o', 'r', 'm'>,
    pegtl::string<'c', 'o', 'n', 'c', 'a', 't'>
> {};

// ===========================================================================
// INDICES
// ===========================================================================

// Forward declarations
struct index;
struct index_expression;

// Simple index: i, j, Alice, Bob, 0, 1
struct simple_index : pegtl::sor<
    identifier,
    integer_literal
> {};

// Virtual index: *t, *t+1, *t-1
struct virtual_index_offset : pegtl::seq<
    pegtl::one<'+', '-'>,
    integer_literal
> {};

struct virtual_index : pegtl::seq<
    pegtl::one<'*'>,
    pegtl::sor<identifier, integer_literal>,
    pegtl::opt<virtual_index_offset>
> {};

// Normalized index: i., k. (for softmax)
struct normalized_index : pegtl::seq<
    identifier,
    pegtl::one<'.'>
> {};

// Index with arithmetic: i+1, i*2, i/2, i%2
struct index_factor : pegtl::sor<
    pegtl::seq<pegtl::one<'('>, pad<index_expression>, pad<pegtl::one<')'>>>,
    identifier,
    integer_literal
> {};

struct index_term : pegtl::seq<
    index_factor,
    pegtl::star<
        pegtl::seq<
            pad<pegtl::one<'*', '/', '%'>>,
            pad<index_factor>
        >
    >
> {};

struct index_expression : pegtl::seq<
    index_term,
    pegtl::star<
        pegtl::seq<
            pad<pegtl::one<'+', '-'>>,
            pad<index_term>
        >
    >
> {};

// Index or slice
struct index : pegtl::sor<
    virtual_index,
    normalized_index,
    index_expression,
    simple_index
> {};

// Slices: 0:10, 0:10:2, :, ::2, 0:, :10
struct slice : pegtl::seq<
    pegtl::opt<integer_literal>,
    pegtl::one<':'>,
    pegtl::opt<integer_literal>,
    pegtl::opt<pegtl::seq<pegtl::one<':'>, integer_literal>>
> {};

struct index_or_slice : pegtl::sor<slice, index> {};

struct index_or_slice_list : pegtl::list<index_or_slice, pegtl::one<','>, ws> {};

// ===========================================================================
// EXPRESSIONS
// ===========================================================================

// Forward declarations
struct expression;
struct tensor_expression;
struct guard_condition;

// Tensor reference: X, X[i], X[i,j]
struct tensor_ref : pegtl::seq<
    identifier,
    pegtl::opt<
        pegtl::seq<
            pad<pegtl::one<'['>>,
            index_or_slice_list,
            pad<pegtl::must<pegtl::one<']'>>>
        >
    >
> {};

// Array literals: [1, 2, 3], [[1,2],[3,4]]
struct array_element;
struct array_literal : pegtl::seq<
    pegtl::one<'['>,
    pad<pegtl::opt<pegtl::list<array_element, pegtl::one<','>, ws>>>,
    pad<pegtl::must<pegtl::one<']'>>>
> {};

struct array_element : pegtl::sor<array_literal, expression> {};

// Function calls: relu(x), concat(a, b, c)
struct function_call : pegtl::seq<
    pegtl::sor<builtin_function, identifier>,
    pad<pegtl::one<'('>>,
    pad<pegtl::opt<pegtl::list<expression, pegtl::one<','>, ws>>>,
    pad<pegtl::must<pegtl::one<')'>>>
> {};

// Primary expressions
struct primary_expression : pegtl::sor<
    pegtl::seq<pegtl::one<'('>, pad<expression>, pad<pegtl::must<pegtl::one<')'>>>>,
    array_literal,
    function_call,
    tensor_ref,
    float_literal,
    integer_literal,
    string_literal,
    boolean_literal
> {};

// Unary expressions: -x, not x
struct unary_expression : pegtl::sor<
    pegtl::seq<pad<pegtl::one<'-'>>, unary_expression>,
    pegtl::seq<pad<kw_not>, unary_expression>,
    primary_expression
> {};

// Power (right-associative): a^b^c = a^(b^c)
struct power_expression : pegtl::seq<
    unary_expression,
    pegtl::opt<
        pegtl::seq<
            pad<pegtl::one<'^'>>,
            power_expression  // Right-associative
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

// Comparison: <, >, <=, >=, ==, !=
struct comparison_op : pegtl::sor<
    pegtl::string<'<', '='>,
    pegtl::string<'>', '='>,
    pegtl::string<'=', '='>,
    pegtl::string<'!', '='>,
    pegtl::one<'<'>,
    pegtl::one<'>'>
> {};

struct comparison_expression : pegtl::seq<
    additive_expression,
    pegtl::opt<
        pegtl::seq<
            pad<comparison_op>,
            pad<additive_expression>
        >
    >
> {};

// Main tensor expression (used in equations)
struct tensor_expression : comparison_expression {};

// Full expression (includes logical operators for guards)
struct expression : tensor_expression {};

// ===========================================================================
// GUARD CONDITIONS (for guarded clauses)
// ===========================================================================

// Guard comparison: same as comparison_expression
struct guard_comparison : comparison_expression {};

// Guard factors: not factor, (condition), comparison
struct guard_factor : pegtl::sor<
    pegtl::seq<pad<kw_not>, guard_factor>,
    pegtl::seq<pegtl::one<'('>, pad<guard_condition>, pad<pegtl::must<pegtl::one<')'>>>>,
    guard_comparison
> {};

// Guard terms: factor and factor and ...
struct guard_term : pegtl::seq<
    guard_factor,
    pegtl::star<
        pegtl::seq<
            pad<kw_and>,
            pad<guard_factor>
        >
    >
> {};

// Guard conditions: term or term or ...
struct guard_condition : pegtl::seq<
    guard_term,
    pegtl::star<
        pegtl::seq<
            pad<kw_or>,
            pad<guard_term>
        >
    >
> {};

// ===========================================================================
// TENSOR EQUATIONS
// ===========================================================================

// Guarded clause: expr : condition | expr
struct guarded_clause : pegtl::seq<
    tensor_expression,
    pegtl::opt<
        pegtl::seq<
            pad<pegtl::one<':'>>,
            pad<guard_condition>
        >
    >
> {};

// Multiple guarded clauses separated by |
struct clause_expression : pegtl::list<guarded_clause, pegtl::one<'|'>, ws> {};

// Projection operators: =, +=, max=, avg=, min=
struct projection_op : pegtl::sor<
    pegtl::string<'+', '='>,
    pegtl::seq<pegtl::string<'m', 'a', 'x'>, pegtl::one<'='>>,
    pegtl::seq<pegtl::string<'a', 'v', 'g'>, pegtl::one<'='>>,
    pegtl::seq<pegtl::string<'m', 'i', 'n'>, pegtl::one<'='>>,
    pegtl::one<'='>
> {};

// Tensor LHS: X[i,j] or X
struct tensor_lhs : tensor_ref {};

// Tensor equation: LHS = RHS
struct tensor_equation : pegtl::seq<
    tensor_lhs,
    pad<projection_op>,
    pad<clause_expression>
> {};

// ===========================================================================
// DATALOG CONSTRUCTS
// ===========================================================================

// Datalog terms: variables (lowercase), constants (uppercase/literals), or expressions
struct datalog_term : pegtl::sor<
    lowercase_identifier,      // Variable: x, y
    uppercase_identifier,      // Constant: Alice, Bob
    number,                    // Numeric constant: 42, 3.14
    string_literal,            // String constant: "hello"
    expression                 // Arithmetic expression: x+1, y*2
> {};

struct datalog_term_list : pegtl::list<datalog_term, pegtl::one<','>, ws> {};

// Datalog atom: Relation(term1, term2, ...)
struct datalog_atom : pegtl::seq<
    uppercase_identifier,
    pad<pegtl::one<'('>>,
    pad<pegtl::opt<datalog_term_list>>,
    pad<pegtl::must<pegtl::one<')'>>>
> {};

// Datalog fact: Relation(const1, const2, ...)
struct datalog_constant : pegtl::sor<
    uppercase_identifier,
    number,
    string_literal
> {};

struct datalog_constant_list : pegtl::list<datalog_constant, pegtl::one<','>, ws> {};

struct datalog_fact : pegtl::seq<
    uppercase_identifier,
    pad<pegtl::one<'('>>,
    pad<datalog_constant_list>,
    pad<pegtl::must<pegtl::one<')'>>>
> {};

// Negated atom: not Atom(...) or !Atom(...)
struct datalog_negation : pegtl::seq<
    pegtl::sor<kw_not, pegtl::one<'!'>>,
    pad<datalog_atom>
> {};

// Comparison condition: x > 5, y == z
struct datalog_condition : pegtl::seq<
    expression,
    pad<comparison_op>,
    pad<expression>
> {};

// Body literal: atom | negation | condition
struct body_literal : pegtl::sor<
    datalog_negation,
    datalog_condition,
    datalog_atom
> {};

struct body_literal_list : pegtl::list<body_literal, pegtl::one<','>, ws> {};

// Datalog rule: Head <- Body1, Body2, ...
struct datalog_rule : pegtl::seq<
    datalog_atom,
    pad<pegtl::string<'<', '-'>>,  // Just use <- for now (Unicode support needs utf8::one)
    pad<body_literal_list>
> {};

// ===========================================================================
// QUERIES AND DIRECTIVES
// ===========================================================================

// Directive arguments: name=value
struct directive_arg : pegtl::seq<
    identifier,
    pad<pegtl::one<'='>>,
    pad<pegtl::sor<number, string_literal, boolean_literal, array_literal>>
> {};

struct directive_arg_list : pegtl::list<directive_arg, pegtl::one<','>, ws> {};

// Query directive: @minimize(lr=0.01, epochs=100)
struct query_directive : pegtl::seq<
    pegtl::one<'@'>,
    identifier,
    pad<pegtl::one<'('>>,
    pad<pegtl::opt<directive_arg_list>>,
    pad<pegtl::must<pegtl::one<')'>>>
> {};

// Query: Atom? or TensorRef? (with optional directive)
struct query : pegtl::seq<
    pegtl::sor<datalog_atom, tensor_ref>,
    pad<pegtl::one<'?'>>,
    pegtl::opt<pad<query_directive>>
> {};

// Conjunctive query: Atom1, Atom2, Condition?
struct conjunctive_query : pegtl::seq<
    datalog_atom,
    pegtl::plus<
        pegtl::seq<
            pad<pegtl::one<','>>,
            pad<body_literal>
        >
    >,
    pad<pegtl::one<'?'>>,
    pegtl::opt<pad<query_directive>>
> {};

// ===========================================================================
// FILE OPERATIONS
// ===========================================================================

struct file_literal : pegtl::sor<
    string_literal,
    pegtl::seq<
        kw_file,
        pad<pegtl::one<'('>>,
        pad<string_literal>,
        pad<pegtl::must<pegtl::one<')'>>>
    >
> {};

struct file_operation : pegtl::sor<
    // file("path") = Tensor or "path" = Tensor
    pegtl::seq<
        file_literal,
        pad<pegtl::one<'='>>,
        pad<tensor_ref>
    >,
    // Tensor = file("path") or Tensor = "path"
    pegtl::seq<
        tensor_ref,
        pad<pegtl::one<'='>>,
        pad<file_literal>
    >
> {};

// ===========================================================================
// STATEMENTS AND PROGRAM
// ===========================================================================

// Statement: try to parse in order of specificity
struct statement : pegtl::sor<
    conjunctive_query,       // Must come before query (both start with atom)
    datalog_rule,            // Must come before datalog_fact (both start with atom)
    query,                   // Must come before datalog_fact
    datalog_fact,
    file_operation,
    tensor_equation
> {};

// Program: sequence of statements
struct program : pegtl::seq<
    ws,
    pegtl::opt<
        pegtl::list<statement, pegtl::plus<ws>, ws>
    >,
    ws,
    pegtl::eof
> {};

// Grammar root
struct grammar : program {};

} // namespace grammar

} // namespace tl
