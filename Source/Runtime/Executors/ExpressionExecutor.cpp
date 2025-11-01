#include "TL/Runtime/Executors/ExpressionExecutor.hpp"
#include "TL/Runtime/ExecutorUtils.hpp"
#include "TL/vm.hpp"
#include <torch/torch.h>
#include <algorithm>

namespace tl {

    bool ExpressionExecutor::canExecute(const TensorEquation &eq, const Environment &env) const {
        // Only handle standard assignment (=)
        if (!eq.projection.empty() && eq.projection != "=") {
            return false;
        }

        // Only handle single-clause, no-guard equations
        if (eq.clauses.size() != 1 || eq.clauses[0].guard.has_value()) {
            return false;
        }

        // Must have an RHS expression
        if (!eq.clauses[0].expr) return false;

        // This is a catch-all executor, so it can handle any expression
        // More specific executors should have run before this one
        return true;
    }

    Tensor ExpressionExecutor::execute(const TensorEquation &eq, Environment &env, TensorBackend &backend) {
        // Evaluate the RHS expression
        Tensor val = evalExpr(eq.clauses[0].expr, eq.lhs, env, backend);

        // Element-wise assignment with label creation: only for simple numeric literals
        // Example: W[Alice] = 1.0, W[0] = 2.0
        std::vector<int64_t> idxs_assign;
        auto parsed_literal = executor_utils::tryParseNumericLiteral(eq.clauses[0].expr);

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
            // Check if indices are bound variables (for guarded clause evaluation)
            bool hasBounden = false;
            for (const auto& ios : tr->ref.indices) {
                if (std::holds_alternative<tl::Slice>(ios.value)) {
                    // Slices are not bound variables
                    continue;
                }
                const auto& idx = std::get<Index>(ios.value);
                if (const auto* id = std::get_if<Identifier>(&idx.value)) {
                    if (env.has(id->name)) {
                        hasBounden = true;
                        break;
                    }
                }
            }

            if (hasBounden) {
                // Resolve indices using bound values from environment
                Tensor base = env.lookup(tr->ref.name.name);
                if (tr->ref.indices.empty()) return base;

                std::vector<torch::indexing::TensorIndex> indices;
                for (const auto& ios : tr->ref.indices) {
                    if (std::holds_alternative<tl::Slice>(ios.value)) {
                        // TL slice - use full torch slice
                        indices.emplace_back(torch::indexing::Slice());
                    } else {
                        const auto& idx = std::get<Index>(ios.value);
                        if (const auto* num = std::get_if<NumberLiteral>(&idx.value)) {
                            long long v = 0;
                            try { v = std::stoll(num->text); } catch (...) { v = 0; }
                            indices.emplace_back(static_cast<int64_t>(v));
                        } else if (const auto* id = std::get_if<Identifier>(&idx.value)) {
                            // Check if bound in environment
                            if (env.has(id->name)) {
                                Tensor idxTensor = env.lookup(id->name);
                                // Convert to integer index
                                int64_t idxValue = static_cast<int64_t>(idxTensor.item<float>());
                                indices.emplace_back(idxValue);
                            } else {
                                // Not bound, use as slice
                                indices.emplace_back(torch::indexing::Slice());
                            }
                        }
                    }
                }
                return base.index(indices);
            }

            // Normal case: use valueForRef
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

        // TODO: Clean this up with a map str->torch::func
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
            } else if (call->func.name == "cos") {
                need1("cos");
                Tensor x = evalExpr(call->args[0], lhsCtx, env, backend);
                return torch::cos(x);
            } else if (call->func.name == "sin") {
                need1("sin");
                Tensor x = evalExpr(call->args[0], lhsCtx, env, backend);
                return torch::sin(x);
            } else if (call->func.name == "tan") {
                need1("tan");
                Tensor x = evalExpr(call->args[0], lhsCtx, env, backend);
                return torch::tan(x);
            } else if (call->func.name == "acos") {
                need1("acos");
                Tensor x = evalExpr(call->args[0], lhsCtx, env, backend);
                return torch::acos(x);
            } else if (call->func.name == "asin") {
                need1("asin");
                Tensor x = evalExpr(call->args[0], lhsCtx, env, backend);
                return torch::asin(x);
            } else if (call->func.name == "atan") {
                need1("atan");
                Tensor x = evalExpr(call->args[0], lhsCtx, env, backend);
                return torch::atan(x);
            } else if (call->func.name == "log") {
                need1("log");
                Tensor x = evalExpr(call->args[0], lhsCtx, env, backend);
                return torch::log(x);
            }

            throw ExecutionError("Unsupported function: " + call->func.name);
        }

        // Handle binary operations
        if (const auto* bin = std::get_if<ExprBinary>(&e.node)) {
            using Op = ExprBinary::Op;

            // For multiplication, try to lower to einsum first
            // But skip if index variables are bound (for guarded clause evaluation)
            if (bin->op == Op::Mul) {
                // Check if any index variables in lhsCtx are bound
                bool hasBoundIndices = false;
                for (const auto& ios : lhsCtx.indices) {
                    if (std::holds_alternative<tl::Slice>(ios.value)) {
                        // Slices are not bound variables
                        continue;
                    }
                    const auto& idx = std::get<Index>(ios.value);
                    if (const auto* id = std::get_if<Identifier>(&idx.value)) {
                        if (env.has(id->name)) {
                            hasBoundIndices = true;
                            break;
                        }
                    }
                }

                if (!hasBoundIndices) {
                    std::string spec;
                    std::vector<Tensor> inputs;
                    if (executor_utils::tryLowerIndexedProductToEinsum(lhsCtx, ep, spec, inputs, env)) {
                        return backend.einsum(spec, inputs);
                    }
                }
            }

            // Evaluate operands
            Tensor a = evalExpr(bin->lhs, lhsCtx, env, backend);
            Tensor b = evalExpr(bin->rhs, lhsCtx, env, backend);

            // Special handling for multiplication: if left operand was an indexed tensor reference
            // and the result would have more dimensions than the LHS context expects,
            // try to construct an einsum to contract over the extra dimensions
            if (bin->op == Op::Mul) {
                // Check if left operand is a simple tensor reference with free indices
                const auto* leftRef = executor_utils::asExprTensorRef(bin->lhs);
                if (leftRef) {
                    // Collect free variable indices from left operand
                    std::vector<std::string> leftIndices;
                    for (const auto& ios : leftRef->ref.indices) {
                        if (std::holds_alternative<Index>(ios.value)) {
                            const auto& idx = std::get<Index>(ios.value);
                            if (const auto* id = std::get_if<Identifier>(&idx.value)) {
                                leftIndices.push_back(id->name);
                            }
                        }
                    }

                    // Collect free variable indices from LHS context
                    std::vector<std::string> outIndices;
                    for (const auto& ios : lhsCtx.indices) {
                        if (std::holds_alternative<Index>(ios.value)) {
                            const auto& idx = std::get<Index>(ios.value);
                            if (const auto* id = std::get_if<Identifier>(&idx.value)) {
                                outIndices.push_back(id->name);
                            }
                        }
                    }

                    // If we have indexed operations and dimensions match what we'd expect from einsum
                    if (!leftIndices.empty() && a.dim() == static_cast<int64_t>(leftIndices.size()) &&
                        b.dim() > 0) {
                        // Try to construct einsum string
                        // Left tensor has dimensions corresponding to leftIndices
                        // Right tensor dimensions should map to indices that appear in leftIndices but not outIndices
                        // (those are the contraction indices)

                        // Build index mapping
                        static const std::string pool = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
                        std::unordered_map<std::string, char> labelMap;
                        size_t nextChar = 0;

                        // Map left indices
                        std::string leftSpec;
                        for (const auto& idx : leftIndices) {
                            if (labelMap.find(idx) == labelMap.end()) {
                                if (nextChar >= pool.size()) goto fallback_multiply;
                                labelMap[idx] = pool[nextChar++];
                            }
                            leftSpec += labelMap[idx];
                        }

                        // Map output indices
                        std::string outSpec;
                        for (const auto& idx : outIndices) {
                            if (labelMap.find(idx) == labelMap.end()) {
                                if (nextChar >= pool.size()) goto fallback_multiply;
                                labelMap[idx] = pool[nextChar++];
                            }
                            outSpec += labelMap[idx];
                        }

                        // Right tensor: figure out its indices
                        // It should have dimensions for indices that appear in leftIndices but not in outIndices
                        std::string rightSpec;
                        for (const auto& idx : leftIndices) {
                            // Only include indices not in output (these get contracted)
                            if (std::find(outIndices.begin(), outIndices.end(), idx) == outIndices.end()) {
                                rightSpec += labelMap[idx];
                            }
                        }

                        // Check if dimensions match and validate einsum indices
                        if (!rightSpec.empty() && b.dim() == static_cast<int64_t>(rightSpec.size())) {
                            // IMPORTANT: Validate that all output indices appear in at least one input
                            // Without this check, we could generate invalid einsum specs like "ab,b->c"
                            // where 'c' doesn't appear in any input operand
                            bool validEinsum = true;
                            for (char c : outSpec) {
                                if (leftSpec.find(c) == std::string::npos && rightSpec.find(c) == std::string::npos) {
                                    validEinsum = false;
                                    break;
                                }
                            }

                            if (validEinsum) {
                                // Construct and execute einsum
                                std::string spec = leftSpec + "," + rightSpec + "->" + outSpec;
                                std::vector<Tensor> inputs = {a, b};
                                return backend.einsum(spec, inputs);
                            }
                        }
                    }
                }
            }

            fallback_multiply:
            // Apply operation
            switch (bin->op) {
                case Op::Add:
                    return a + b;
                case Op::Sub:
                    return a - b;
                case Op::Div:
                    return a / b;
                case Op::Mod:
                    return torch::fmod(a, b);
                case Op::Mul:
                    return a * b;
                case Op::Pow:
                    return torch::pow(a, b);
                case Op::Lt:
                    return torch::lt(a, b).to(torch::kFloat32);
                case Op::Le:
                    return torch::le(a, b).to(torch::kFloat32);
                case Op::Gt:
                    return torch::gt(a, b).to(torch::kFloat32);
                case Op::Ge:
                    return torch::ge(a, b).to(torch::kFloat32);
                case Op::Eq:
                    return torch::eq(a, b).to(torch::kFloat32);
                case Op::Ne:
                    return torch::ne(a, b).to(torch::kFloat32);
                case Op::And:
                    // Logical AND: both must be non-zero
                    return torch::logical_and(torch::ne(a, 0), torch::ne(b, 0)).to(torch::kFloat32);
                case Op::Or:
                    // Logical OR: at least one must be non-zero
                    return torch::logical_or(torch::ne(a, 0), torch::ne(b, 0)).to(torch::kFloat32);
                default:
                    throw ExecutionError("Unknown binary operator");
            }
        }

        // Handle unary operations
        if (const auto* un = std::get_if<ExprUnary>(&e.node)) {
            using Op = ExprUnary::Op;
            Tensor operand = evalExpr(un->operand, lhsCtx, env, backend);

            switch (un->op) {
                case Op::Neg:
                    return -operand;
                case Op::Not:
                    // Logical NOT: zero if non-zero, one if zero
                    return torch::eq(operand, 0).to(torch::kFloat32);
                default:
                    throw ExecutionError("Unknown unary operator");
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
