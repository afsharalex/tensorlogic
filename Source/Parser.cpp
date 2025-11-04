#include "TL/Parser.hpp"
#include <fstream>
#include <sstream>
#include <iostream>

// Minimal PEGTL parser implementation
// Following the proven pattern from Lexer.cpp

namespace tl::parse {

namespace pegtl = tao::pegtl;

// Helper: Create source location from PEGTL position
static SourceLocation locFrom(const pegtl::position& p) {
    SourceLocation l;
    l.line = p.line;
    l.column = p.column;
    return l;
}

// ParseState - expanding for tensor equations
struct ParseState {
    // Literal stacks
    std::vector<Identifier> identifier_stack;
    std::vector<NumberLiteral> number_stack;

    // Index stacks
    std::vector<Index> index_stack;
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

// Grammar rules - expanding for tensor equations

struct ws : pegtl::star<pegtl::space> {};

template<typename Rule>
struct pad : pegtl::seq<ws, Rule, ws> {};

// Lexical elements
struct identifier : pegtl::seq<
    pegtl::alpha,
    pegtl::star<pegtl::sor<pegtl::alnum, pegtl::one<'_'>>>
> {};

struct integer_literal : pegtl::plus<pegtl::digit> {};

struct float_literal : pegtl::seq<
    pegtl::plus<pegtl::digit>,
    pegtl::one<'.'>,
    pegtl::star<pegtl::digit>
> {};

struct number_literal : pegtl::sor<float_literal, integer_literal> {};

// Indices
struct simple_index : pegtl::sor<identifier, integer_literal> {};

struct index_list : pegtl::list<pad<simple_index>, pegtl::one<','>> {};

// Tensor references
struct tensor_ref : pegtl::seq<
    identifier,
    pegtl::opt<pegtl::seq<
        pegtl::one<'['>,
        index_list,
        pegtl::one<']'>
    >>
> {};

// Expressions
struct primary_expression : pegtl::sor<tensor_ref, number_literal> {};

// Implicit multiplication: A B C means A * B * C
// Multiple expressions separated by at least one space
struct rhs_expression : pegtl::seq<
    ws,
    primary_expression,
    pegtl::star<pegtl::seq<pegtl::plus<pegtl::space>, primary_expression>>,
    ws
> {};

// Guarded clause
struct guarded_clause : rhs_expression {};

// Projection operators
struct projection_op : pegtl::sor<
    pegtl::string<'m', 'a', 'x', '='>,
    pegtl::string<'m', 'i', 'n', '='>,
    pegtl::string<'a', 'v', 'g', '='>,
    pegtl::string<'+', '='>,
    pegtl::one<'='>
> {};

// Tensor equation
struct tensor_equation : pegtl::seq<
    tensor_ref,
    pad<projection_op>,
    guarded_clause
> {};

// Statement
struct statement : pegtl::sor<tensor_equation> {};

// Program with multiple statements
// Use star instead of list to avoid separator issues
struct program : pegtl::seq<
    pegtl::star<pegtl::seq<ws, statement, ws>>,
    pegtl::eof
> {};

struct grammar : program {};

// Actions - following Lexer.cpp pattern exactly
template<typename Rule>
struct action : pegtl::nothing<Rule> {};

// Lexical actions
template<>
struct action<identifier> {
    template<typename Input>
    static void apply(const Input& in, ParseState& state) {
        Identifier id;
        id.name = std::string(in.string());
        id.loc = locFrom(in.position());
        state.identifier_stack.push_back(std::move(id));
    }
};

template<>
struct action<integer_literal> {
    template<typename Input>
    static void apply(const Input& in, ParseState& state) {
        NumberLiteral num;
        num.text = std::string(in.string());
        num.loc = locFrom(in.position());
        state.number_stack.push_back(std::move(num));
    }
};

template<>
struct action<float_literal> {
    template<typename Input>
    static void apply(const Input& in, ParseState& state) {
        NumberLiteral num;
        num.text = std::string(in.string());
        num.loc = locFrom(in.position());
        state.number_stack.push_back(std::move(num));
    }
};

// Index actions
template<>
struct action<simple_index> {
    template<typename Input>
    static void apply(const Input& in, ParseState& state) {
        Index idx;
        idx.loc = locFrom(in.position());

        if (!state.identifier_stack.empty()) {
            idx.value = state.identifier_stack.back();
            state.identifier_stack.pop_back();
        } else if (!state.number_stack.empty()) {
            idx.value = state.number_stack.back();
            state.number_stack.pop_back();
        }

        // Wrap in IndexOrSlice for consistency
        IndexOrSlice ios;
        ios.loc = idx.loc;
        ios.value = std::move(idx);
        state.index_or_slice_stack.push_back(std::move(ios));
    }
};

// Tensor ref action
template<>
struct action<tensor_ref> {
    template<typename Input>
    static void apply(const Input& in, ParseState& state) {
        TensorRef ref;
        ref.loc = locFrom(in.position());

        // Pop identifier (tensor name) from the back (most recent)
        if (!state.identifier_stack.empty()) {
            ref.name = std::move(state.identifier_stack.back());
            state.identifier_stack.pop_back();
        }

        // Pop all indices that belong to this tensor ref
        ref.indices = std::move(state.index_or_slice_stack);
        state.index_or_slice_stack.clear();

        state.tensorref_stack.push_back(std::move(ref));
    }
};

// Expression actions
template<>
struct action<primary_expression> {
    template<typename Input>
    static void apply(const Input& in, ParseState& state) {
        auto expr = std::make_shared<Expr>();
        expr->loc = locFrom(in.position());

        // Pop from back (most recent) - standard LIFO
        if (!state.tensorref_stack.empty()) {
            auto ref = std::move(state.tensorref_stack.back());
            state.tensorref_stack.pop_back();
            expr->node = ExprTensorRef{std::move(ref)};
        } else if (!state.number_stack.empty()) {
            auto num = std::move(state.number_stack.back());
            state.number_stack.pop_back();
            expr->node = ExprNumber{std::move(num)};
        }

        state.expr_stack.push_back(expr);
    }
};

// Note: rhs_expression action not needed since guarded_clause handles it
// (guarded_clause is currently just an alias for rhs_expression)

template<>
struct action<guarded_clause> {
    template<typename Input>
    static void apply(const Input& in, ParseState& state) {
        // First, handle implicit multiplication (same logic as rhs_expression)
        if (state.expr_stack.size() > 1) {
            // Expressions are pushed onto stack in parse order (A, then B, then C...)
            // Build left-to-right: ((A * B) * C) * D
            auto result = state.expr_stack[0];
            for (size_t i = 1; i < state.expr_stack.size(); ++i) {
                auto binary = std::make_shared<Expr>();
                binary->loc = result->loc;

                ExprBinary bin;
                bin.op = ExprBinary::Op::Mul;
                bin.lhs = result;
                bin.rhs = state.expr_stack[i];
                binary->node = std::move(bin);

                result = binary;
            }

            state.expr_stack.clear();
            state.expr_stack.push_back(result);
        }

        // Now create the guarded clause with the (possibly combined) expression
        GuardedClause clause;
        clause.loc = locFrom(in.position());

        if (!state.expr_stack.empty()) {
            clause.expr = state.expr_stack.back();
            state.expr_stack.pop_back();
        }

        state.clause_stack.push_back(std::move(clause));
    }
};

template<>
struct action<projection_op> {
    template<typename Input>
    static void apply(const Input& in, ParseState& state) {
        state.current_projection_op = std::string(in.string());
    }
};

template<>
struct action<tensor_equation> {
    template<typename Input>
    static void apply(const Input& in, ParseState& state) {
        TensorEquation eq;
        eq.loc = locFrom(in.position());

        // Pop LHS tensor ref
        if (!state.tensorref_stack.empty()) {
            eq.lhs = state.tensorref_stack.back();
            state.tensorref_stack.pop_back();
        }

        // Get projection operator
        eq.projection = state.current_projection_op.empty() ? "=" : state.current_projection_op;
        state.current_projection_op.clear();

        // Pop all guarded clauses
        eq.clauses = std::move(state.clause_stack);
        state.clause_stack.clear();

        state.statements.push_back(std::move(eq));
    }
};

} // namespace tl::parse

// Public API implementation
namespace tl {

Program parseProgram(std::string_view source) {
    namespace pegtl = tao::pegtl;

    pegtl::memory_input input(source, "<input>");
    parse::ParseState state;

    try {
        pegtl::parse<parse::grammar, parse::action>(input, state);
    } catch (const pegtl::parse_error& e) {
        const auto& pos = e.positions().front();
        std::ostringstream oss;
        oss << "Parse error at line " << pos.line << ", column " << pos.column
            << ": " << e.what();
        throw ParseError(oss.str());
    }

    Program prog;
    prog.statements = std::move(state.statements);
    return prog;
}

Program parseFile(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs) throw ParseError("Cannot open file: " + path);
    std::stringstream buffer;
    buffer << ifs.rdbuf();
    return parseProgram(buffer.str());
}

} // namespace tl
