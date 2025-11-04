#include "TL/Parser.hpp"
#include "TL/Grammar.hpp"
#include "TL/ParserActions.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>

// TensorLogic PEGTL Parser Implementation
// Grammar rules are in Include/TL/Grammar.hpp
// This file contains action implementations and the public API

namespace tl::actions {

namespace pegtl = tao::pegtl;
using namespace tl::grammar;

// ============================================================================
// LEXICAL ACTIONS
// ============================================================================

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

// ============================================================================
// INDEX AND SLICE ACTIONS
// ============================================================================

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

template<>
struct action<slice> {
    template<typename Input>
    static void apply(const Input& in, ParseState& state) {
        Slice s;
        s.loc = locFrom(in.position());

        // Parse the matched string to understand structure
        std::string slice_str = std::string(in.string());

        // Count colons to determine if we have start:end or start:end:step
        size_t colon_count = std::count(slice_str.begin(), slice_str.end(), ':');

        // Collect numbers that were parsed for this slice
        std::vector<NumberLiteral> nums;
        while (!state.number_stack.empty() && nums.size() < 3) {
            nums.push_back(state.number_stack.back());
            state.number_stack.pop_back();
        }

        // Reverse to get them in parse order
        std::reverse(nums.begin(), nums.end());

        // Assign based on the structure of the slice string
        if (colon_count == 1) {
            // Format: [start]:[end]
            if (slice_str[0] == ':') {
                // Format: :end
                if (!nums.empty()) s.end = nums[0];
            } else if (slice_str.back() == ':') {
                // Format: start:
                if (!nums.empty()) s.start = nums[0];
            } else {
                // Format: start:end
                if (nums.size() >= 1) s.start = nums[0];
                if (nums.size() >= 2) s.end = nums[1];
            }
        } else if (colon_count == 2) {
            // Format: [start]:[end]:step
            size_t first_colon = slice_str.find(':');
            size_t second_colon = slice_str.find(':', first_colon + 1);

            if (first_colon == 0) {
                // Format: :[end]:step or ::step
                if (second_colon == 1) {
                    // Format: ::step
                    if (!nums.empty()) s.step = nums[0];
                } else {
                    // Format: :end:step
                    if (nums.size() >= 1) s.end = nums[0];
                    if (nums.size() >= 2) s.step = nums[1];
                }
            } else {
                // Format: start:[end]:step or start::step
                if (second_colon == first_colon + 1) {
                    // Format: start::step
                    if (nums.size() >= 1) s.start = nums[0];
                    if (nums.size() >= 2) s.step = nums[1];
                } else {
                    // Format: start:end:step
                    if (nums.size() >= 1) s.start = nums[0];
                    if (nums.size() >= 2) s.end = nums[1];
                    if (nums.size() >= 3) s.step = nums[2];
                }
            }
        }

        // Wrap in IndexOrSlice
        IndexOrSlice ios;
        ios.loc = s.loc;
        ios.value = std::move(s);
        state.index_or_slice_stack.push_back(std::move(ios));
    }
};

// ============================================================================
// TENSOR REFERENCE ACTIONS
// ============================================================================

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

// ============================================================================
// EXPRESSION ACTIONS
// ============================================================================

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

template<>
struct action<guarded_clause> {
    template<typename Input>
    static void apply(const Input& in, ParseState& state) {
        // First, handle implicit multiplication
        // Multiple expressions on the stack means implicit multiplication: A B C = A * B * C
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

// ============================================================================
// TENSOR EQUATION ACTIONS
// ============================================================================

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

} // namespace tl::actions

// ============================================================================
// PUBLIC API
// ============================================================================

namespace tl {

Program parseProgram(std::string_view source) {
    namespace pegtl = tao::pegtl;

    pegtl::memory_input input(source, "<input>");
    actions::ParseState state;

    try {
        pegtl::parse<grammar::grammar, actions::action>(input, state);
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
