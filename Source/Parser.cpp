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

        // Parse the matched text directly to avoid stack confusion
        // (integer_literal also pushes to expr_stack, causing duplicates)
        std::string matched = std::string(in.string());

        // Check if it's a number or identifier
        if (!matched.empty() && std::isdigit(matched[0])) {
            // It's a number literal
            NumberLiteral num;
            num.text = matched;
            num.loc = idx.loc;
            idx.value = num;
        } else {
            // It's an identifier
            Identifier id;
            id.name = matched;
            id.loc = idx.loc;
            idx.value = id;
        }

        // Wrap in IndexOrSlice for consistency
        IndexOrSlice ios;
        ios.loc = idx.loc;
        ios.value = std::move(idx);
        state.index_or_slice_stack.push_back(std::move(ios));
    }
};

template<>
struct action<normalized_index> {
    template<typename Input>
    static void apply(const Input& in, ParseState& state) {
        Index idx;
        idx.loc = locFrom(in.position());
        idx.normalized = true;  // Mark as normalized

        // Parse the matched text directly to extract the identifier (without the dot)
        std::string matched = std::string(in.string());
        if (!matched.empty() && matched.back() == '.') {
            matched.pop_back();
        }

        Identifier id;
        id.name = matched;
        id.loc = idx.loc;
        idx.value = id;

        // Wrap in IndexOrSlice
        IndexOrSlice ios;
        ios.loc = idx.loc;
        ios.value = std::move(idx);
        state.index_or_slice_stack.push_back(std::move(ios));
    }
};

template<>
struct action<virtual_index> {
    template<typename Input>
    static void apply(const Input& in, ParseState& state) {
        Index idx;
        idx.loc = locFrom(in.position());

        // Parse the matched text directly to extract virtual index components
        // Format: *t, *t+1, *t-1
        std::string matched = std::string(in.string());

        VirtualIndex virt;
        virt.loc = idx.loc;

        // Skip the * prefix
        size_t pos = 1; // Start after '*'

        // Extract the identifier name (e.g., 't' from '*t')
        size_t id_start = pos;
        while (pos < matched.size() && (std::isalnum(matched[pos]) || matched[pos] == '_')) {
            pos++;
        }

        if (pos > id_start) {
            virt.name.name = matched.substr(id_start, pos - id_start);
            virt.name.loc = virt.loc;
        }

        // Check for optional offset (+1 or -1)
        if (pos < matched.size()) {
            // Parse +/- followed by number
            char op = matched[pos];
            if (op == '+' || op == '-') {
                pos++; // Skip the operator
                std::string num_str;
                while (pos < matched.size() && std::isdigit(matched[pos])) {
                    num_str += matched[pos];
                    pos++;
                }
                if (!num_str.empty()) {
                    int offset_val = std::stoi(num_str);
                    virt.offset = (op == '-') ? -offset_val : offset_val;

                    // Validate: only +1 and -1 are semantically valid (per grammar spec)
                    if (virt.offset != 1 && virt.offset != -1) {
                        // Warning: other offsets may not be semantically valid
                        // For now, we parse them but implementations should validate
                    }
                }
            }
        }

        // Wrap VirtualIndex in Index
        idx.value = std::move(virt);

        // Wrap in IndexOrSlice
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

        // Parse the matched string directly to extract numbers
        std::string slice_str = std::string(in.string());

        // Count colons to determine if we have start:end or start:end:step
        size_t colon_count = std::count(slice_str.begin(), slice_str.end(), ':');

        // Parse numbers from the slice string
        std::vector<NumberLiteral> nums;
        std::string current_num;
        for (char c : slice_str) {
            if (std::isdigit(c)) {
                current_num += c;
            } else if (c == ':') {
                if (!current_num.empty()) {
                    NumberLiteral num;
                    num.text = current_num;
                    num.loc = s.loc;
                    nums.push_back(num);
                    current_num.clear();
                }
            }
        }
        // Don't forget the last number
        if (!current_num.empty()) {
            NumberLiteral num;
            num.text = current_num;
            num.loc = s.loc;
            nums.push_back(num);
        }

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

        // The tensor name is the most recent identifier on the stack that hasn't
        // been consumed by index actions. Since simple_index and normalized_index
        // now parse identifiers directly from matched text, the tensor name is the
        // last identifier on the stack.
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
struct action<function_call> {
    template<typename Input>
    static void apply(const Input& in, ParseState& state) {
        ExprCall call;

        // Extract function name from matched text (before the opening paren)
        std::string matched = std::string(in.string());
        size_t paren_pos = matched.find('(');
        if (paren_pos != std::string::npos) {
            std::string func_name = matched.substr(0, paren_pos);
            // Trim whitespace
            func_name.erase(0, func_name.find_first_not_of(" \t\n\r"));
            func_name.erase(func_name.find_last_not_of(" \t\n\r") + 1);

            call.func.name = func_name;
            call.func.loc = locFrom(in.position());
        }

        // Clear all identifiers from the stack (function name + any from arguments)
        state.identifier_stack.clear();

        // Collect arguments from expression stack
        call.args = std::move(state.expr_stack);
        state.expr_stack.clear();

        // Wrap in Expr and push back
        auto expr = std::make_shared<Expr>();
        expr->loc = locFrom(in.position());
        expr->node = std::move(call);
        state.expr_stack.push_back(expr);
    }
};

template<>
struct action<primary_expression> {
    template<typename Input>
    static void apply(const Input& in, ParseState& state) {
        // Function calls and numbers already push to expr_stack
        // Parenthesized expressions also already push to expr_stack
        // Only handle tensor_ref case: check if this matched a tensor_ref

        std::string text = std::string(in.string());

        // If text contains '(', it's either a function call or parenthesized expr - already handled
        if (text.find('(') != std::string::npos) {
            return;
        }

        // If starts with digit, it's a number literal - already handled
        if (!text.empty() && std::isdigit(text[0])) {
            return;
        }

        // Otherwise it's a tensor_ref, need to wrap it
        if (!state.tensorref_stack.empty()) {
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
struct action<comparison_op> {
    template<typename Input>
    static void apply(const Input& in, ParseState& state) {
        // Store the comparison operator for use in comparison_expression
        state.current_comparison_op = std::string(in.string());
    }
};

template<>
struct action<comparison_expression> {
    template<typename Input>
    static void apply(const Input& in, ParseState& state) {
        std::string text = std::string(in.string());

        // Check if a comparison operator is present
        if (!state.current_comparison_op.empty() && state.expr_stack.size() >= 2) {
            // Pop RHS and LHS
            auto rhs = state.expr_stack.back();
            state.expr_stack.pop_back();
            auto lhs = state.expr_stack.back();
            state.expr_stack.pop_back();

            // Build comparison expression
            auto expr = std::make_shared<Expr>();
            expr->loc = lhs->loc;

            ExprBinary bin;
            std::string op = state.current_comparison_op;
            if (op == "<") bin.op = ExprBinary::Op::Lt;
            else if (op == "<=") bin.op = ExprBinary::Op::Le;
            else if (op == ">") bin.op = ExprBinary::Op::Gt;
            else if (op == ">=") bin.op = ExprBinary::Op::Ge;
            else if (op == "==") bin.op = ExprBinary::Op::Eq;
            else if (op == "!=") bin.op = ExprBinary::Op::Ne;

            bin.lhs = lhs;
            bin.rhs = rhs;
            expr->node = std::move(bin);

            state.expr_stack.push_back(expr);
            state.current_comparison_op.clear();
        }
        // Otherwise single expression, already on stack
    }
};

template<>
struct action<expression> {
    template<typename Input>
    static void apply(const Input& in, ParseState& state) {
        // expression now inherits from comparison_expression
        // Comparisons are handled by comparison_expression action
        // Nothing additional to do here
    }
};

// ============================================================================
// GUARD CONDITION ACTIONS
// ============================================================================

template<>
struct action<guard_factor> {
    template<typename Input>
    static void apply(const Input& in, ParseState& state) {
        std::string text = std::string(in.string());

        // Check if this is a 'not' expression
        if (text.find("not") != std::string::npos && !state.expr_stack.empty()) {
            auto operand = state.expr_stack.back();
            state.expr_stack.pop_back();

            auto expr = std::make_shared<Expr>();
            expr->loc = locFrom(in.position());

            ExprUnary unary;
            unary.op = ExprUnary::Op::Not;
            unary.operand = operand;
            expr->node = std::move(unary);

            state.expr_stack.push_back(expr);
        }
        // Otherwise, comparison_expression or parenthesized guard already on stack
    }
};

template<>
struct action<guard_term> {
    template<typename Input>
    static void apply(const Input& in, ParseState& state) {
        std::string text = std::string(in.string());

        // Count how many 'and' operators we have
        size_t and_count = 0;
        size_t pos = 0;
        while ((pos = text.find(" and ", pos)) != std::string::npos) {
            and_count++;
            pos += 5;  // Length of " and "
        }

        if (and_count > 0 && state.expr_stack.size() > and_count) {
            // Build left-to-right: ((A and B) and C)
            size_t start_idx = state.expr_stack.size() - and_count - 1;
            auto result = state.expr_stack[start_idx];

            for (size_t i = 1; i <= and_count; ++i) {
                auto binary = std::make_shared<Expr>();
                binary->loc = result->loc;

                ExprBinary bin;
                bin.op = ExprBinary::Op::And;
                bin.lhs = result;
                bin.rhs = state.expr_stack[start_idx + i];
                binary->node = std::move(bin);

                result = binary;
            }

            state.expr_stack.erase(state.expr_stack.begin() + start_idx, state.expr_stack.end());
            state.expr_stack.push_back(result);
        }
        // Otherwise single factor, already on stack
    }
};

template<>
struct action<guard_condition> {
    template<typename Input>
    static void apply(const Input& in, ParseState& state) {
        std::string text = std::string(in.string());

        // Count how many 'or' operators we have
        size_t or_count = 0;
        size_t pos = 0;
        while ((pos = text.find(" or ", pos)) != std::string::npos) {
            or_count++;
            pos += 4;  // Length of " or "
        }

        if (or_count > 0 && state.expr_stack.size() > or_count) {
            // Build left-to-right: ((A or B) or C)
            size_t start_idx = state.expr_stack.size() - or_count - 1;
            auto result = state.expr_stack[start_idx];

            for (size_t i = 1; i <= or_count; ++i) {
                auto binary = std::make_shared<Expr>();
                binary->loc = result->loc;

                ExprBinary bin;
                bin.op = ExprBinary::Op::Or;
                bin.lhs = result;
                bin.rhs = state.expr_stack[start_idx + i];
                binary->node = std::move(bin);

                result = binary;
            }

            state.expr_stack.erase(state.expr_stack.begin() + start_idx, state.expr_stack.end());
            state.expr_stack.push_back(result);
        }
        // Otherwise single term, already on stack
    }
};

template<>
struct action<guarded_clause> {
    template<typename Input>
    static void apply(const Input& in, ParseState& state) {
        std::string text = std::string(in.string());

        // Check if there's a guard (contains ':')
        bool has_guard = (text.find(':') != std::string::npos);

        GuardedClause clause;
        clause.loc = locFrom(in.position());

        if (has_guard && state.expr_stack.size() >= 2) {
            // Pop guard condition first (it was parsed last)
            auto guard = state.expr_stack.back();
            state.expr_stack.pop_back();

            // Then pop the main expression
            auto expr = state.expr_stack.back();
            state.expr_stack.pop_back();

            clause.expr = expr;
            clause.guard = guard;
        } else if (!state.expr_stack.empty()) {
            // No guard, just the expression
            // Handle implicit multiplication first
            if (state.expr_stack.size() > 1) {
                // Multiple expressions means implicit multiplication: A B C = A * B * C
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
                clause.expr = result;
                state.expr_stack.clear();
            } else {
                clause.expr = state.expr_stack.back();
                state.expr_stack.pop_back();
            }
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

// ============================================================================
// FILE OPERATION ACTIONS
// ============================================================================

template<>
struct action<string_literal> {
    template<typename Input>
    static void apply(const Input& in, ParseState& state) {
        std::string matched = std::string(in.string());

        // Remove surrounding quotes
        if (matched.size() >= 2 && matched.front() == '"' && matched.back() == '"') {
            matched = matched.substr(1, matched.size() - 2);
        }

        // Process escape sequences
        std::string unescaped;
        for (size_t i = 0; i < matched.size(); ++i) {
            if (matched[i] == '\\' && i + 1 < matched.size()) {
                // Handle escape sequences
                char next = matched[i + 1];
                switch (next) {
                    case 'n': unescaped += '\n'; break;
                    case 't': unescaped += '\t'; break;
                    case 'r': unescaped += '\r'; break;
                    case '\\': unescaped += '\\'; break;
                    case '"': unescaped += '"'; break;
                    default:
                        // Unknown escape, keep both characters
                        unescaped += '\\';
                        unescaped += next;
                        break;
                }
                ++i;  // Skip the next character
            } else {
                unescaped += matched[i];
            }
        }

        StringLiteral str;
        str.text = unescaped;
        str.loc = locFrom(in.position());
        state.string_stack.push_back(std::move(str));
    }
};

template<>
struct action<file_literal> {
    template<typename Input>
    static void apply(const Input& in, ParseState& state) {
        // file_literal matches either file("path") or just "path"
        // In both cases, string_literal action has already pushed the string to string_stack
        // Nothing additional needed here - the string is already on the stack
    }
};

template<>
struct action<file_operation> {
    template<typename Input>
    static void apply(const Input& in, ParseState& state) {
        FileOperation fileop;
        fileop.loc = locFrom(in.position());

        std::string text = std::string(in.string());

        // Determine direction by checking if '=' comes after a string literal or tensor
        // Format: tensor = "file" (read) or "file" = tensor (write)
        size_t eq_pos = text.find('=');
        bool starts_with_quote = (text.find_first_not_of(" \t\n\r") < text.size() &&
                                   text[text.find_first_not_of(" \t\n\r")] == '"');

        if (starts_with_quote) {
            // "file" = tensor (write operation)
            fileop.lhsIsTensor = false;

            // Pop tensor ref from tensorref_stack
            if (!state.tensorref_stack.empty()) {
                fileop.tensor = state.tensorref_stack.back();
                state.tensorref_stack.pop_back();
            }

            // Pop string literal from string_stack
            if (!state.string_stack.empty()) {
                fileop.file = state.string_stack.back();
                state.string_stack.pop_back();
            }
        } else {
            // tensor = "file" (read operation)
            fileop.lhsIsTensor = true;

            // Pop string literal from string_stack
            if (!state.string_stack.empty()) {
                fileop.file = state.string_stack.back();
                state.string_stack.pop_back();
            }

            // Pop tensor ref from tensorref_stack
            if (!state.tensorref_stack.empty()) {
                fileop.tensor = state.tensorref_stack.back();
                state.tensorref_stack.pop_back();
            }
        }

        state.statements.push_back(std::move(fileop));
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
