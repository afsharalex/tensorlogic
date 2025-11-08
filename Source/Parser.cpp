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
        state.number_stack.push_back(num);

        // Also push as expression for use in arithmetic
        auto expr = std::make_shared<Expr>();
        expr->loc = num.loc;
        expr->node = ExprNumber{num};
        state.expr_stack.push_back(expr);
    }
};

template<>
struct action<float_literal> {
    template<typename Input>
    static void apply(const Input& in, ParseState& state) {
        NumberLiteral num;
        num.text = std::string(in.string());
        num.loc = locFrom(in.position());
        state.number_stack.push_back(num);

        // Also push as expression for use in arithmetic
        auto expr = std::make_shared<Expr>();
        expr->loc = num.loc;
        expr->node = ExprNumber{num};
        state.expr_stack.push_back(expr);
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
        // Numbers and parenthesized expressions already push to expr_stack
        // Only handle tensor_ref case: check if this matched a tensor_ref by seeing if tensorref was just added

        // Simple heuristic: If matched text starts with a letter, it's a tensor_ref and needs conversion.
        // If it starts with a digit or '(', it's already handled.
        std::string text = std::string(in.string());
        if (!text.empty() && std::isalpha(text[0]) && !state.tensorref_stack.empty()) {
            auto expr = std::make_shared<Expr>();
            expr->loc = locFrom(in.position());
            expr->node = ExprTensorRef{state.tensorref_stack.back()};
            state.tensorref_stack.pop_back();
            state.expr_stack.push_back(expr);
        }
    }
};

template<>
struct action<unary_expression> {
    template<typename Input>
    static void apply(const Input& in, ParseState& state) {
        // If this matched the recursive case (- unary_expression), wrap in ExprUnary
        std::string text = std::string(in.string());

        // Simple heuristic: if starts with -, it's a negation
        if (!text.empty() && (text[0] == '-' || (text.size() > 1 && std::isspace(text[0]) && text[1] == '-'))) {
            if (!state.expr_stack.empty()) {
                auto operand = state.expr_stack.back();
                state.expr_stack.pop_back();

                auto expr = std::make_shared<Expr>();
                expr->loc = locFrom(in.position());

                ExprUnary unary;
                unary.op = ExprUnary::Op::Neg;
                unary.operand = operand;
                expr->node = std::move(unary);

                state.expr_stack.push_back(expr);
            }
        }
        // Otherwise it's just a primary expression, already on stack
    }
};

template<>
struct action<power_expression> {
    template<typename Input>
    static void apply(const Input& in, ParseState& state) {
        std::string text = std::string(in.string());
        // Check if ^ operator is present

        if (text.find('^') != std::string::npos && state.expr_stack.size() >= 2) {
            // Power is right-associative, so RHS is already fully formed
            // Pop RHS, then LHS
            auto rhs = state.expr_stack.back();
            state.expr_stack.pop_back();
            auto lhs = state.expr_stack.back();
            state.expr_stack.pop_back();

            auto expr = std::make_shared<Expr>();
            expr->loc = lhs->loc;

            ExprBinary binary;
            binary.op = ExprBinary::Op::Pow;
            binary.lhs = lhs;
            binary.rhs = rhs;
            expr->node = std::move(binary);

            state.expr_stack.push_back(expr);
        }
        // Otherwise single operand, already on stack
    }
};

template<>
struct action<multiplicative_expression> {
    template<typename Input>
    static void apply(const Input& in, ParseState& state) {
        // Parse the input to find operators
        std::string text = std::string(in.string());

        // Count how many operators we have
        size_t op_count = 0;
        for (char c : text) {
            if (c == '*' || c == '/' || c == '%') op_count++;
        }

        if (op_count > 0 && state.expr_stack.size() > op_count) {
            // Build left-to-right: ((A * B) / C) % D
            // Bottom of relevant portion of stack has leftmost operand
            size_t start_idx = state.expr_stack.size() - op_count - 1;
            auto result = state.expr_stack[start_idx];

            // Find operators in order from matched text
            size_t operand_idx = start_idx + 1;
            for (size_t pos = 0; pos < text.size() && operand_idx < state.expr_stack.size(); pos++) {
                char c = text[pos];
                if (c == '*' || c == '/' || c == '%') {
                    auto binary = std::make_shared<Expr>();
                    binary->loc = result->loc;

                    ExprBinary bin;
                    bin.op = (c == '*') ? ExprBinary::Op::Mul :
                             (c == '/') ? ExprBinary::Op::Div :
                                          ExprBinary::Op::Mod;
                    bin.lhs = result;
                    bin.rhs = state.expr_stack[operand_idx++];
                    binary->node = std::move(bin);

                    result = binary;
                }
            }

            // Remove operands from stack and push result
            state.expr_stack.erase(state.expr_stack.begin() + start_idx, state.expr_stack.end());
            state.expr_stack.push_back(result);
        }
        // Otherwise single operand, already on stack
    }
};

template<>
struct action<additive_expression> {
    template<typename Input>
    static void apply(const Input& in, ParseState& state) {
        // Parse the input to find operators
        std::string text = std::string(in.string());
        std::cerr << "[DEBUG] additive_expression matched: '" << text << "'" << std::endl;
        std::cerr << "[DEBUG] expr_stack.size() = " << state.expr_stack.size() << std::endl;

        // Count how many operators we have (careful not to count unary minus)
        size_t op_count = 0;
        bool prev_was_op = true;  // Start of expression
        for (size_t i = 0; i < text.size(); i++) {
            char c = text[i];
            if ((c == '+' || c == '-') && !prev_was_op) {
                op_count++;
                prev_was_op = true;
            } else if (c != ' ' && c != '\t' && c != '\n' && c != '\r') {
                prev_was_op = false;
            }
        }

        if (op_count > 0 && state.expr_stack.size() > op_count) {
            // Build left-to-right: ((A + B) - C) + D
            size_t start_idx = state.expr_stack.size() - op_count - 1;
            auto result = state.expr_stack[start_idx];

            // Find operators in order
            size_t operand_idx = start_idx + 1;
            bool looking_for_op = true;

            for (size_t i = 0; i < text.size() && operand_idx < state.expr_stack.size(); i++) {
                char c = text[i];
                if (looking_for_op && (c == '+' || c == '-')) {
                    auto binary = std::make_shared<Expr>();
                    binary->loc = result->loc;

                    ExprBinary bin;
                    bin.op = (c == '+') ? ExprBinary::Op::Add : ExprBinary::Op::Sub;
                    bin.lhs = result;
                    bin.rhs = state.expr_stack[operand_idx++];
                    binary->node = std::move(bin);

                    result = binary;
                    looking_for_op = false;
                } else if (c != ' ' && c != '\t' && c != '\n' && c != '\r' && c != '+' && c != '-') {
                    looking_for_op = true;
                }
            }

            // Remove operands from stack and push result
            state.expr_stack.erase(state.expr_stack.begin() + start_idx, state.expr_stack.end());
            state.expr_stack.push_back(result);
        }
        // Otherwise single operand, already on stack
    }
};

template<>
struct action<expression> {
    template<typename Input>
    static void apply(const Input& in, ParseState& state) {
        std::string text = std::string(in.string());

        // Since expression inherits from additive_expression, we ONLY handle additive operators (+, -) here.
        // Power, multiplicative operators are already handled by their respective actions.
        size_t op_count = 0;
        bool prev_was_op = true;  // For distinguishing binary +/- from unary
        for (size_t i = 0; i < text.size(); i++) {
            char c = text[i];
            bool is_add_sub = (c == '+' || c == '-');

            if (is_add_sub && !prev_was_op) {
                op_count++;
                prev_was_op = true;
            } else if (c != ' ' && c != '\t' && c != '\n' && c != '\r') {
                prev_was_op = false;
            }
        }

        if (op_count > 0 && state.expr_stack.size() > op_count) {
            // Build left-to-right
            size_t start_idx = state.expr_stack.size() - op_count - 1;
            auto result = state.expr_stack[start_idx];

            size_t operand_idx = start_idx + 1;
            bool looking_for_op = true;

            for (size_t i = 0; i < text.size() && operand_idx < state.expr_stack.size(); i++) {
                char c = text[i];
                bool is_add_sub = (c == '+' || c == '-');

                if (looking_for_op && is_add_sub) {
                    auto binary = std::make_shared<Expr>();
                    binary->loc = result->loc;

                    ExprBinary bin;
                    bin.op = (c == '+') ? ExprBinary::Op::Add : ExprBinary::Op::Sub;
                    bin.lhs = result;
                    bin.rhs = state.expr_stack[operand_idx++];
                    binary->node = std::move(bin);

                    result = binary;
                    looking_for_op = false;
                } else if (c != ' ' && c != '\t' && c != '\n' && c != '\r' && !is_add_sub) {
                    looking_for_op = true;
                }
            }

            state.expr_stack.erase(state.expr_stack.begin() + start_idx, state.expr_stack.end());
            state.expr_stack.push_back(result);
        }
        // Otherwise single operand, already on stack
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

// ============================================================================
// DATALOG ACTIONS
// ============================================================================

template<>
struct action<uppercase_identifier> {
    template<typename Input>
    static void apply(const Input& in, ParseState& state) {
        Identifier id;
        id.name = std::string(in.string());
        id.loc = locFrom(in.position());
        // Push to datalog_term_stack for use in Datalog contexts
        state.datalog_term_stack.push_back(std::move(id));
    }
};

template<>
struct action<lowercase_identifier> {
    template<typename Input>
    static void apply(const Input& in, ParseState& state) {
        Identifier id;
        id.name = std::string(in.string());
        id.loc = locFrom(in.position());
        // Push to datalog_term_stack for use in Datalog contexts
        state.datalog_term_stack.push_back(std::move(id));
    }
};

template<>
struct action<datalog_term> {
    template<typename Input>
    static void apply(const Input& in, ParseState& state) {
        // Terms are already pushed to datalog_term_stack by uppercase/lowercase_identifier
        // or number_literal actions. Nothing additional needed here.
    }
};

template<>
struct action<datalog_atom> {
    template<typename Input>
    static void apply(const Input& in, ParseState& state) {
        // The term stack has: relation_name, term1, term2, ..., termN
        // So if stack size is N+1, we have N terms
        size_t num_terms = state.datalog_term_stack.size() > 0 ? (state.datalog_term_stack.size() - 1) : 0;

        DatalogAtom atom;
        atom.loc = locFrom(in.position());

        // Pop relation name (it's at the bottom of the relevant section)
        // Terms are pushed after relation, so relation is at position [size - num_terms - 1]
        if (state.datalog_term_stack.size() > num_terms) {
            size_t relation_idx = state.datalog_term_stack.size() - num_terms - 1;
            atom.relation = state.datalog_term_stack[relation_idx];
            state.datalog_term_stack.erase(state.datalog_term_stack.begin() + relation_idx);
        }

        // Collect the last num_terms from the stack
        if (state.datalog_term_stack.size() >= num_terms) {
            size_t start_idx = state.datalog_term_stack.size() - num_terms;
            for (size_t i = start_idx; i < state.datalog_term_stack.size(); ++i) {
                const auto& id = state.datalog_term_stack[i];
                // Check if it's uppercase (constant) or lowercase (variable)
                if (!id.name.empty() && std::isupper(id.name[0])) {
                    // Uppercase -> constant -> StringLiteral
                    StringLiteral lit;
                    lit.text = id.name;
                    lit.loc = id.loc;
                    atom.terms.push_back(lit);
                } else {
                    // Lowercase -> variable -> Identifier
                    atom.terms.push_back(id);
                }
            }
            // Remove processed terms
            state.datalog_term_stack.erase(state.datalog_term_stack.begin() + start_idx,
                                           state.datalog_term_stack.end());
        }

        // Push to atom stack for use in facts/rules/queries
        state.datalog_atom_stack.push_back(std::move(atom));
    }
};

template<>
struct action<datalog_body_literal> {
    template<typename Input>
    static void apply(const Input& in, ParseState& state) {
        // Since PEGTL doesn't auto-call parent actions for inherited rules,
        // we need to manually invoke the datalog_atom logic here
        action<datalog_atom>::apply(in, state);
    }
};

template<>
struct action<datalog_fact> {
    template<typename Input>
    static void apply(const Input& in, ParseState& state) {
        DatalogFact fact;
        fact.loc = locFrom(in.position());

        // Pop the atom and convert to DatalogFact format
        if (!state.datalog_atom_stack.empty()) {
            DatalogAtom& atom = state.datalog_atom_stack.back();
            fact.relation = atom.relation;

            // Convert terms to constants (for facts, all terms should be constants)
            for (const auto& term_variant : atom.terms) {
                // Terms can be Identifier, StringLiteral, or ExprPtr
                // For facts, they should be constants (StringLiteral or NumberLiteral)
                if (std::holds_alternative<StringLiteral>(term_variant)) {
                    fact.constants.push_back(std::get<StringLiteral>(term_variant));
                } else if (std::holds_alternative<Identifier>(term_variant)) {
                    // Convert uppercase Identifier to StringLiteral
                    const Identifier& id = std::get<Identifier>(term_variant);
                    StringLiteral lit;
                    lit.text = id.name;
                    lit.loc = id.loc;
                    fact.constants.push_back(lit);
                }
                // TODO: Handle numeric literals properly
            }

            state.datalog_atom_stack.pop_back();
        }

        state.statements.push_back(std::move(fact));
    }
};

template<>
struct action<datalog_rule> {
    template<typename Input>
    static void apply(const Input& in, ParseState& state) {
        DatalogRule rule;
        rule.loc = locFrom(in.position());

        // Pop head atom (first one pushed)
        if (!state.datalog_atom_stack.empty()) {
            rule.head = state.datalog_atom_stack.front();
            state.datalog_atom_stack.erase(state.datalog_atom_stack.begin());
        }

        // Pop body atoms (remaining atoms on stack)
        // Convert DatalogAtom vector to variant vector
        for (auto& atom : state.datalog_atom_stack) {
            rule.body.push_back(std::move(atom));
        }
        state.datalog_atom_stack.clear();

        state.statements.push_back(std::move(rule));
    }
};

template<>
struct action<datalog_query> {
    template<typename Input>
    static void apply(const Input& in, ParseState& state) {
        Query query;
        query.loc = locFrom(in.position());

        // Pop the atom and set as target
        if (!state.datalog_atom_stack.empty()) {
            query.target = state.datalog_atom_stack.back();
            state.datalog_atom_stack.pop_back();
        }

        state.statements.push_back(std::move(query));
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
