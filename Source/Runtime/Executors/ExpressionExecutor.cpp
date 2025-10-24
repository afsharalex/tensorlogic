#include "TL/Runtime/Executors/ExpressionExecutor.hpp"
#include "TL/Runtime/ExecutorUtils.hpp"
#include "TL/vm.hpp"
#include <torch/torch.h>

namespace tl {

    bool ExpressionExecutor::canExecute(const TensorEquation &eq, const Environment &env) const {
        // Only handle standard assignment (=)
        if (!eq.projection.empty() && eq.projection != "=") {
            return false;
        }

        // Must have an RHS expression
        if (!eq.rhs) return false;

        // This is a catch-all executor, so it can handle any expression
        // More specific executors should have run before this one
        return true;
    }

    Tensor ExpressionExecutor::execute(const TensorEquation &eq, Environment &env, TensorBackend &backend) {
        // Evaluate the RHS expression
        Tensor val = evalExpr(eq.rhs, eq.lhs, env, backend);

        // Element-wise assignment with label creation: only for simple numeric literals
        // Example: W[Alice] = 1.0, W[0] = 2.0
        std::vector<int64_t> idxs_assign;
        auto parsed_literal = executor_utils::tryParseNumericLiteral(eq.rhs);

        if (!eq.lhs.indices.empty() && parsed_literal.has_value()) {
            // RHS is a simple numeric literal, so we can do element-wise assignment
            // Use resolveIndicesCreatingLabels to handle both numeric and label indices
            idxs_assign = executor_utils::resolveIndicesCreatingLabels(eq.lhs, env);

            if (!idxs_assign.empty()) {
                // Ensure destination tensor exists and is large enough
                const std::string& lhs_name = eq.lhs.name.name;
                Tensor t = executor_utils::ensureTensorSize(lhs_name, idxs_assign, env);

            // Ensure RHS is scalar; if not, reduce to scalar by sum
            if (val.dim() > 0) {
                if (val.numel() == 1) {
                    val = val.reshape({});
                } else {
                    val = torch::sum(val);
                }
            }

            // Write element
            const auto opts = torch::TensorOptions().dtype(torch::kFloat32);
            std::vector<torch::indexing::TensorIndex> elemIdx;
            elemIdx.reserve(idxs_assign.size());
            for (int64_t v : idxs_assign) {
                elemIdx.emplace_back(v);
            }
                t.index_put_(elemIdx, val.to(opts.dtype()));

                return t;
            }
        }

        // If LHS is scalar but RHS evaluated to a tensor, auto-reduce by summing
        if (eq.lhs.indices.empty() && val.dim() > 0) {
            val = torch::sum(val);
        }

        return val;
    }

    Tensor ExpressionExecutor::evalExpr(const ExprPtr& ep, const TensorRef& lhsCtx,
                                       Environment& env, TensorBackend& backend) const {
        if (!ep) {
            throw ExecutionError("null expression");
        }

        const Expr& e = *ep;

        // Handle numeric literals
        if (const auto* num = std::get_if<ExprNumber>(&e.node)) {
            double v = 0.0;
            try {
                v = std::stod(num->literal.text);
            } catch (...) {}
            return torch::tensor(static_cast<float>(v));
        }

        // Handle tensor references
        if (const auto* tr = std::get_if<ExprTensorRef>(&e.node)) {
            return executor_utils::valueForRef(tr->ref, env);
        }

        // Handle parentheses
        if (const auto* par = std::get_if<ExprParen>(&e.node)) {
            return evalExpr(par->inner, lhsCtx, env, backend);
        }

        // Handle list literals (should be caught by ListLiteralExecutor, but handle here as fallback)
        if (const auto* lst = std::get_if<ExprList>(&e.node)) {
            // Build n-D tensor from nested list literal
            std::function<void(const ExprPtr&, std::vector<int64_t>&, std::vector<float>&)> collect =
                [&](const ExprPtr& ep2, std::vector<int64_t>& shape_out, std::vector<float>& flat_out) {
                const Expr& ex = *ep2;
                if (const auto* l = std::get_if<ExprList>(&ex.node)) {
                    const size_t n = l->elements.size();
                    std::vector<int64_t> child_shape;
                    bool first = true;
                    for (const auto& child : l->elements) {
                        std::vector<int64_t> cs;
                        collect(child, cs, flat_out);
                        if (first) {
                            child_shape = cs;
                            first = false;
                        } else if (child_shape != cs) {
                            throw ExecutionError("List literal is not rectangular (sub-shapes differ)");
                        }
                    }
                    shape_out.clear();
                    shape_out.push_back(static_cast<int64_t>(n));
                    shape_out.insert(shape_out.end(), child_shape.begin(), child_shape.end());
                    return;
                }
                // Leaf: evaluate numeric expression to scalar
                Tensor v = evalExpr(ep2, lhsCtx, env, backend);
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
            std::vector<int64_t> shape;
            std::vector<int64_t> top_shape;
            auto selfPtr = std::make_shared<Expr>(e);
            collect(selfPtr, top_shape, data);
            shape = top_shape;

            if (shape.empty()) {
                return torch::tensor(data.empty() ? 0.0f : data[0]);
            }
            return torch::tensor(data).reshape(shape);
        }

        // Handle function calls
        if (const auto* call = std::get_if<ExprCall>(&e.node)) {
            auto need1 = [&](const char* fname) {
                if (call->args.size() != 1) {
                    throw ExecutionError(std::string(fname) + "() expects 1 argument");
                }
            };

            if (call->func.name == "step") {
                need1("step");
                Tensor x = evalExpr(call->args[0], lhsCtx, env, backend);
                return torch::gt(x, 0).to(torch::kFloat32);
            } else if (call->func.name == "sqrt") {
                need1("sqrt");
                Tensor x = evalExpr(call->args[0], lhsCtx, env, backend);
                return torch::sqrt(x);
            } else if (call->func.name == "abs") {
                need1("abs");
                Tensor x = evalExpr(call->args[0], lhsCtx, env, backend);
                return torch::abs(x);
            } else if (call->func.name == "sigmoid") {
                need1("sigmoid");
                Tensor x = evalExpr(call->args[0], lhsCtx, env, backend);
                return torch::sigmoid(x);
            } else if (call->func.name == "tanh") {
                need1("tanh");
                Tensor x = evalExpr(call->args[0], lhsCtx, env, backend);
                return torch::tanh(x);
            } else if (call->func.name == "relu") {
                need1("relu");
                Tensor x = evalExpr(call->args[0], lhsCtx, env, backend);
                return torch::relu(x);
            } else if (call->func.name == "exp") {
                need1("exp");
                Tensor x = evalExpr(call->args[0], lhsCtx, env, backend);
                return torch::exp(x);
            } else if (call->func.name == "softmax") {
                need1("softmax");
                Tensor x = evalExpr(call->args[0], lhsCtx, env, backend);
                if (x.dim() == 0) return torch::tensor(1.0f);
                int64_t dim = std::max<int64_t>(0, x.dim() - 1);
                return torch::softmax(x, dim);
            }

            throw ExecutionError("Unsupported function: " + call->func.name);
        }

        // Handle binary operations
        if (const auto* bin = std::get_if<ExprBinary>(&e.node)) {
            using Op = ExprBinary::Op;

            // For multiplication, try to lower to einsum first
            if (bin->op == Op::Mul) {
                std::string spec;
                std::vector<Tensor> inputs;
                if (executor_utils::tryLowerIndexedProductToEinsum(lhsCtx, ep, spec, inputs, env)) {
                    return backend.einsum(spec, inputs);
                }
            }

            // Evaluate operands
            Tensor a = evalExpr(bin->lhs, lhsCtx, env, backend);
            Tensor b = evalExpr(bin->rhs, lhsCtx, env, backend);

            // Apply operation
            switch (bin->op) {
                case Op::Add:
                    return a + b;
                case Op::Sub:
                    return a - b;
                case Op::Div:
                    return a / b;
                case Op::Mul:
                default:
                    return a * b;
            }
        }

        throw ExecutionError("Unsupported expression node in ExpressionExecutor");
    }

    std::string ExpressionExecutor::name() const {
        return "ExpressionExecutor";
    }

    int ExpressionExecutor::priority() const {
        return 90; // Low priority - catch-all for expressions not handled by specific executors
    }

} // namespace tl
