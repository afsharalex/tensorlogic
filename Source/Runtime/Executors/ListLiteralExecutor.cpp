#include "TL/Runtime/Executors/ListLiteralExecutor.hpp"
#include "TL/vm.hpp"
#include <torch/torch.h>
#include <functional>

namespace tl {

    bool ListLiteralExecutor::canExecute(const TensorEquation &eq, const Environment &env) const {
        // Only handle standard assignment (=), not pooling operators
        if (!eq.projection.empty() && eq.projection != "=") {
            return false;
        }

        // Only handle single-clause, no-guard equations
        if (eq.clauses.size() != 1 || eq.clauses[0].guard.has_value()) {
            return false;
        }

        // Check if RHS is a list literal and LHS has no indices (bare identifier)
        if (!eq.clauses[0].expr || !eq.lhs.indices.empty()) {
            return false;
        }

        const Expr &e = *eq.clauses[0].expr;
        return std::holds_alternative<ExprList>(e.node);
    }

    Tensor ListLiteralExecutor::execute(const TensorEquation &eq, Environment &env, TensorBackend &backend) {
        if (!eq.clauses[0].expr) {
            throw ExecutionError("List literal executor: null RHS");
        }

        const Expr &e = *eq.clauses[0].expr;
        const auto *lst = std::get_if<ExprList>(&e.node);
        if (!lst) {
            throw ExecutionError("List literal executor: expected list literal");
        }

        // Helper: recursively evaluate and collect shape and data
        std::function<Tensor(const ExprPtr&)> evalExpr_simple = [&](const ExprPtr& ep) -> Tensor {
            if (!ep) throw ExecutionError("null expression in list literal");
            const Expr& ex = *ep;

            if (const auto* num = std::get_if<ExprNumber>(&ex.node)) {
                double v = 0.0;
                try { v = std::stod(num->literal.text); } catch (...) {}
                return torch::tensor(static_cast<float>(v));
            }

            if (const auto* par = std::get_if<ExprParen>(&ex.node)) {
                return evalExpr_simple(par->inner);
            }

            throw ExecutionError("List literal must contain numeric values");
        };

        std::function<void(const ExprPtr&, std::vector<int64_t>&, std::vector<float>&)> collect =
            [&](const ExprPtr& ep, std::vector<int64_t>& shape_out, std::vector<float>& flat_out) {
            const Expr& ex = *ep;
            if (const auto* l = std::get_if<ExprList>(&ex.node)) {
                const size_t n = l->elements.size();
                std::vector<int64_t> child_shape;
                bool first = true;
                for (const auto& child : l->elements) {
                    std::vector<int64_t> cs;
                    collect(child, cs, flat_out);
                    if (first) { child_shape = cs; first = false; }
                    else if (child_shape != cs) {
                        throw ExecutionError("List literal is not rectangular (sub-shapes differ)");
                    }
                }
                shape_out.clear();
                shape_out.push_back(static_cast<int64_t>(n));
                shape_out.insert(shape_out.end(), child_shape.begin(), child_shape.end());
                return;
            }
            // Leaf: evaluate numeric expression to scalar
            Tensor v = evalExpr_simple(ep);
            if (v.dim() == 0) {
                flat_out.push_back(v.item<float>());
                shape_out.clear();
                return;
            }
            if (v.numel() == 1) {
                flat_out.push_back(v.reshape({}).item<float>());
                shape_out.clear();
                return;
            }
            throw ExecutionError("List literal leaf must be a scalar expression");
        };

        std::vector<float> data;
        std::vector<int64_t> top_shape;
        auto selfPtr = std::make_shared<Expr>(e);
        collect(selfPtr, top_shape, data);

        Tensor t;
        if (top_shape.empty()) {
            t = torch::tensor(data.empty() ? 0.0f : data[0]);
        } else {
            t = torch::tensor(data).reshape(top_shape);
        }

        return t;
    }

    std::string ListLiteralExecutor::name() const {
        return "ListLiteralExecutor";
    }

    int ListLiteralExecutor::priority() const {
        return 20; // Check before general expression executor
    }

} // namespace tl
