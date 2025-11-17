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
        // BUT: Due to PEG backtracking, there may be artifacts on the stack
        // The marker points to the LHS - don't use items at or before the marker
        if (!state.tensorref_stack.empty()) {
            size_t stack_pos = state.tensorref_stack.size() - 1;

            // If this tensor ref is at or before the LHS marker, it's either the LHS or a backtracking artifact
            // In either case, ignore it (LHS is handled by tensor_equation action)
            if (stack_pos <= state.tensor_equation_lhs_marker) {
                return;
            }

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
struct action<tensor_equation_lhs> {
    template<typename Input>
    static void apply(const Input& in, ParseState& state) {
        // When this rule matches, the underlying tensor_ref has been parsed
        // BUT since we inherit from tensor_ref, the tensor_ref action does NOT fire
        // We need to manually build and push the tensor ref ourselves

        TensorRef ref;
        ref.loc = locFrom(in.position());

        // Pop the identifier for the tensor name
        if (!state.identifier_stack.empty()) {
            ref.name = std::move(state.identifier_stack.back());
            state.identifier_stack.pop_back();
        }

        // Pop all indices that belong to this tensor ref
        ref.indices = std::move(state.index_or_slice_stack);
        state.index_or_slice_stack.clear();

        // Push to tensorref_stack
        state.tensorref_stack.push_back(std::move(ref));

        // Set marker to point to the LHS we just pushed (at end of stack)
        // This marker prevents PEG backtracking artifacts from being used in the RHS
        state.tensor_equation_lhs_marker = state.tensorref_stack.size() - 1;
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

        // Due to PEG backtracking, there may be residue on tensorref_stack from failed parse attempts
        // The marker points to the actual LHS position
        // Get the LHS from the marker position
        if (state.tensor_equation_lhs_marker < state.tensorref_stack.size()) {
            eq.lhs = state.tensorref_stack[state.tensor_equation_lhs_marker];
        }

        // Get projection operator
        eq.projection = state.current_projection_op.empty() ? "=" : state.current_projection_op;
        state.current_projection_op.clear();

        // Pop all guarded clauses
        eq.clauses = std::move(state.clause_stack);
        state.clause_stack.clear();

        // Clear ALL tensorref_stack items (backtracking artifacts + LHS)
        state.tensorref_stack.clear();

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
        // datalog_term matches: lowercase_identifier | uppercase_identifier | number_literal
        // Identifiers are ALREADY pushed by their respective uppercase/lowercase_identifier actions
        // We ONLY need to handle numbers here (which don't have their own Datalog-specific actions)

        std::string text = std::string(in.string());

        // Check if it's a number (starts with digit or minus sign followed by digit)
        if (!text.empty() && std::isdigit(text[0])) {
            // It's a positive number - push to datalog_term_stack
            Identifier term_id;
            term_id.name = text;
            term_id.loc = locFrom(in.position());
            state.datalog_term_stack.push_back(std::move(term_id));
        } else if (text.size() > 1 && text[0] == '-' && std::isdigit(text[1])) {
            // It's a negative number - push to datalog_term_stack
            Identifier term_id;
            term_id.name = text;
            term_id.loc = locFrom(in.position());
            state.datalog_term_stack.push_back(std::move(term_id));
        }
        // Identifiers are already on the stack from uppercase/lowercase_identifier actions
        // We do NOT push them again here to avoid PEG backtracking duplication issues
    }
};

template<>
struct action<datalog_relation_name> {
    template<typename Input>
    static void apply(const Input& in, ParseState& state) {
        // When this rule matches, we need to:
        // 1. Push the relation name to the term stack (uppercase_identifier action won't fire automatically)
        // 2. Set a marker to track where this atom's data starts

        // Push relation name to term stack
        Identifier id;
        id.name = std::string(in.string());
        id.loc = locFrom(in.position());
        state.datalog_term_stack.push_back(std::move(id));

        // Set marker to point to the relation name we just pushed (at end of stack)
        state.datalog_atom_term_start_marker = state.datalog_term_stack.size() - 1;
    }
};

template<>
struct action<datalog_atom> {
    template<typename Input>
    static void apply(const Input& in, ParseState& state) {
        // The marker was set by datalog_relation_name to point to where this atom's terms start
        // Due to PEG backtracking, there may be residue on the stack from failed parse attempts
        // We only consume terms from the marker position onward

        DatalogAtom atom;
        atom.loc = locFrom(in.position());

        // Get the start position (where relation name was)
        size_t start_pos = state.datalog_atom_term_start_marker;

        if (start_pos < state.datalog_term_stack.size()) {
            // First element at marker position is the relation name
            atom.relation = state.datalog_term_stack[start_pos];

            // Remaining elements after the relation are terms
            for (size_t i = start_pos + 1; i < state.datalog_term_stack.size(); ++i) {
                const auto& id = state.datalog_term_stack[i];
                // Check if it's uppercase (constant), lowercase (variable), or number
                if (!id.name.empty() && std::isupper(id.name[0])) {
                    // Uppercase -> constant -> StringLiteral
                    StringLiteral lit;
                    lit.text = id.name;
                    lit.loc = id.loc;
                    atom.terms.push_back(lit);
                } else if (!id.name.empty() && (std::isdigit(id.name[0]) ||
                          (id.name.size() > 1 && id.name[0] == '-' && std::isdigit(id.name[1])))) {
                    // Number -> convert to StringLiteral (numbers are constants in Datalog)
                    StringLiteral lit;
                    lit.text = id.name;
                    lit.loc = id.loc;
                    atom.terms.push_back(lit);
                } else {
                    // Lowercase or other -> variable -> Identifier
                    atom.terms.push_back(id);
                }
            }

            // Remove ONLY the terms belonging to this atom (from marker onward)
            state.datalog_term_stack.erase(
                state.datalog_term_stack.begin() + start_pos,
                state.datalog_term_stack.end()
            );
        }

        // Push to atom stack for use in facts/rules/queries
        state.datalog_atom_stack.push_back(std::move(atom));
    }
};

template<>
struct action<datalog_negation> {
    template<typename Input>
    static void apply(const Input& in, ParseState& state) {
        // Pop the atom that was just parsed by the nested datalog_atom rule
        if (!state.datalog_atom_stack.empty()) {
            DatalogAtom atom = state.datalog_atom_stack.back();
            state.datalog_atom_stack.pop_back();

            // Wrap in DatalogNegation
            DatalogNegation neg;
            neg.atom = std::move(atom);
            neg.loc = locFrom(in.position());

            // Push to body literal stack as negation
            state.datalog_body_stack.push_back(std::move(neg));
        }
    }
};

template<>
struct action<datalog_comparison> {
    template<typename Input>
    static void apply(const Input& in, ParseState& state) {
        // datalog_comparison matches: term op term
        // Terms have been pushed to datalog_term_stack by datalog_term action
        // We need to pop the last 2 terms and build a DatalogCondition

        if (state.datalog_term_stack.size() >= 2 && !state.current_comparison_op.empty()) {
            // Pop RHS term
            Identifier rhs_term = state.datalog_term_stack.back();
            state.datalog_term_stack.pop_back();

            // Pop LHS term
            Identifier lhs_term = state.datalog_term_stack.back();
            state.datalog_term_stack.pop_back();

            // Build DatalogCondition
            DatalogCondition cond;
            cond.loc = locFrom(in.position());
            cond.op = state.current_comparison_op;

            // Convert terms to ExprPtr (wrap Identifiers in ExprNumber or ExprTensorRef)
            auto lhs_expr = std::make_shared<Expr>();
            lhs_expr->loc = lhs_term.loc;

            // Check if term is a number or identifier
            if (!lhs_term.name.empty() && std::isdigit(lhs_term.name[0])) {
                // It's a number
                NumberLiteral num;
                num.text = lhs_term.name;
                num.loc = lhs_term.loc;
                lhs_expr->node = ExprNumber{num};
            } else {
                // It's an identifier - wrap as TensorRef (scalar)
                TensorRef ref;
                ref.name = lhs_term;
                ref.loc = lhs_term.loc;
                lhs_expr->node = ExprTensorRef{ref};
            }

            auto rhs_expr = std::make_shared<Expr>();
            rhs_expr->loc = rhs_term.loc;

            if (!rhs_term.name.empty() && std::isdigit(rhs_term.name[0])) {
                // It's a number
                NumberLiteral num;
                num.text = rhs_term.name;
                num.loc = rhs_term.loc;
                rhs_expr->node = ExprNumber{num};
            } else {
                // It's an identifier
                TensorRef ref;
                ref.name = rhs_term;
                ref.loc = rhs_term.loc;
                rhs_expr->node = ExprTensorRef{ref};
            }

            cond.lhs = lhs_expr;
            cond.rhs = rhs_expr;

            // Clear the comparison operator
            state.current_comparison_op.clear();

            // Push to body literal stack as comparison
            state.datalog_body_stack.push_back(std::move(cond));
        }
    }
};

template<>
struct action<datalog_body_literal> {
    template<typename Input>
    static void apply(const Input& in, ParseState& state) {
        // datalog_body_literal is sor<datalog_negation, datalog_comparison, datalog_atom>
        // If it matched datalog_negation or datalog_comparison, those actions already handled it
        // If it matched datalog_atom, we need to move it to body_stack

        std::string text = std::string(in.string());
        bool is_negation = (text.find("not") != std::string::npos ||
                           text.find('!') != std::string::npos ||
                           text.find("\u00AC") != std::string::npos);  // ¬ Unicode character

        // Check if it's a comparison by looking for comparison operators
        bool is_comparison = (text.find("!=") != std::string::npos ||
                             text.find("==") != std::string::npos ||
                             text.find("<=") != std::string::npos ||
                             text.find(">=") != std::string::npos ||
                             text.find('<') != std::string::npos ||
                             text.find('>') != std::string::npos);

        if (!is_negation && !is_comparison) {
            // It's a regular atom - move from atom_stack to body_stack
            if (!state.datalog_atom_stack.empty()) {
                DatalogAtom atom = state.datalog_atom_stack.back();
                state.datalog_atom_stack.pop_back();
                state.datalog_body_stack.push_back(std::move(atom));
            }
        }
        // Otherwise negation or comparison action has already pushed to body_stack
    }
};

template<>
struct action<datalog_fact> {
    template<typename Input>
    static void apply(const Input& in, ParseState& state) {
        DatalogFact fact;
        fact.loc = locFrom(in.position());

        // IMPORTANT: Due to PEG backtracking, the atom_stack and term_stack may have
        // multiple copies from failed parse attempts (datalog_rule, datalog_query).
        // We use the LAST atom (most recent) and clear ALL stacks.

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

            // Clear ALL atoms and terms (includes backtracking artifacts)
            state.datalog_atom_stack.clear();
            state.datalog_term_stack.clear();
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

        // Pop body literals from body_stack (already in variant form)
        rule.body = std::move(state.datalog_body_stack);
        state.datalog_body_stack.clear();

        // Clear all stacks (includes backtracking artifacts)
        state.datalog_atom_stack.clear();
        state.datalog_term_stack.clear();

        state.statements.push_back(std::move(rule));
    }
};

// ============================================================================
// LEARNING DIRECTIVE ACTIONS
// ============================================================================

template<>
struct action<boolean_literal> {
    template<typename Input>
    static void apply(const Input& in, ParseState& state) {
        std::string text = std::string(in.string());
        state.current_boolean_value = (text == "true");
    }
};

template<>
struct action<directive_arg> {
    template<typename Input>
    static void apply(const Input& in, ParseState& state) {
        DirectiveArg arg;
        arg.loc = locFrom(in.position());

        std::string text = std::string(in.string());

        // Pop the argument name from identifier stack
        if (!state.identifier_stack.empty()) {
            arg.name = state.identifier_stack.back();
            state.identifier_stack.pop_back();
        }

        // Determine the value type based on what's on the stacks
        if (text.find("true") != std::string::npos || text.find("false") != std::string::npos) {
            // Boolean value
            arg.value = state.current_boolean_value;
        } else if (!state.string_stack.empty()) {
            // String value
            arg.value = state.string_stack.back();
            state.string_stack.pop_back();
        } else if (!state.number_stack.empty()) {
            // Number value
            arg.value = state.number_stack.back();
            state.number_stack.pop_back();
        }

        state.directive_arg_stack.push_back(std::move(arg));
    }
};

template<>
struct action<query_directive> {
    template<typename Input>
    static void apply(const Input& in, ParseState& state) {
        // QueryDirective is built here but stored temporarily
        // It will be attached to the Query in tensor_query action
        // For now, just leave the args on directive_arg_stack
    }
};

template<>
struct action<tensor_query> {
    template<typename Input>
    static void apply(const Input& in, ParseState& state) {
        Query q;
        q.loc = locFrom(in.position());

        std::string text = std::string(in.string());
        bool has_directive = (text.find('@') != std::string::npos);

        // Pop tensor ref from tensorref_stack
        if (!state.tensorref_stack.empty()) {
            q.target = state.tensorref_stack.back();
            state.tensorref_stack.pop_back();
        }

        // If there's a directive, build it
        if (has_directive) {
            QueryDirective dir;
            dir.loc = q.loc;

            // Pop directive name from identifier stack
            if (!state.identifier_stack.empty()) {
                dir.name = state.identifier_stack.back();
                state.identifier_stack.pop_back();
            }

            // Pop all directive arguments
            dir.args = std::move(state.directive_arg_stack);
            state.directive_arg_stack.clear();

            q.directive = std::move(dir);
        }

        state.statements.push_back(std::move(q));
    }
};

template<>
struct action<datalog_query> {
    template<typename Input>
    static void apply(const Input& in, ParseState& state) {
        Query q;
        q.loc = locFrom(in.position());

        // IMPORTANT: Due to PEG backtracking, use LAST atom and clear ALL stacks
        if (!state.datalog_atom_stack.empty()) {
            q.target = state.datalog_atom_stack.back();
            state.datalog_atom_stack.clear();
            state.datalog_term_stack.clear();
        }

        state.statements.push_back(std::move(q));
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
struct action<list_literal> {
    template<typename Input>
    static void apply(const Input& in, ParseState& state) {
        // Collect all expressions that belong to this list
        // They're on the expr_stack from parsing list_elements

        ExprList list;
        list.elements = std::move(state.expr_stack);
        state.expr_stack.clear();

        // Wrap in Expr and push back
        auto expr = std::make_shared<Expr>();
        expr->loc = locFrom(in.position());
        expr->node = std::move(list);
        state.expr_stack.push_back(expr);
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
