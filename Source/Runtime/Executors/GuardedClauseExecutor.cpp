#include "TL/Runtime/Executors/GuardedClauseExecutor.hpp"
#include "TL/Runtime/Executors/ExpressionExecutor.hpp"
#include "TL/Runtime/ExecutorUtils.hpp"
#include "TL/vm.hpp"
#include <torch/torch.h>
#include <set>

namespace tl {

    bool GuardedClauseExecutor::canExecute(const TensorEquation &eq, const Environment &env) const {
        // Only handle standard assignment (=)
        if (!eq.projection.empty() && eq.projection != "=") {
            return false;
        }

        // Handle multi-clause equations OR single-clause with guard
        if (eq.clauses.empty()) {
            return false;
        }

        // If single clause without guard, let other executors handle it
        if (eq.clauses.size() == 1 && !eq.clauses[0].guard.has_value()) {
            return false;
        }

        return true;
    }

    Tensor GuardedClauseExecutor::execute(const TensorEquation &eq, Environment &env, TensorBackend &backend) {
        // We need an ExpressionExecutor instance to evaluate expressions
        ExpressionExecutor exprEval;

        // If LHS has indices, we need to handle them specially
        if (!eq.lhs.indices.empty()) {
            // Extract index variable names from LHS
            std::vector<std::string> indexVars;
            for (const auto& ios : eq.lhs.indices) {
                // Skip slices - they don't contribute to index variable names
                if (std::holds_alternative<Index>(ios.value)) {
                    const auto& idx = std::get<Index>(ios.value);
                    if (const auto* id = std::get_if<Identifier>(&idx.value)) {
                        indexVars.push_back(id->name);
                    }
                }
            }

            // Determine the size needed by checking RHS tensor references
            // For now, assume 1D indexing and infer size from referenced tensors
            int64_t maxSize = 0;

            // Find maximum size from any referenced tensor in the clauses (expressions and guards)
            std::function<void(const ExprPtr&)> findTensors = [&](const ExprPtr& ep) {
                if (!ep) return;
                const Expr& e = *ep;

                if (const auto* tr = std::get_if<ExprTensorRef>(&e.node)) {
                    if (env.has(tr->ref.name.name)) {
                        Tensor t = env.lookup(tr->ref.name.name);
                        if (t.dim() > 0) {
                            maxSize = std::max(maxSize, t.size(0));
                        }
                    }
                } else if (const auto* bin = std::get_if<ExprBinary>(&e.node)) {
                    findTensors(bin->lhs);
                    findTensors(bin->rhs);
                } else if (const auto* un = std::get_if<ExprUnary>(&e.node)) {
                    findTensors(un->operand);
                } else if (const auto* par = std::get_if<ExprParen>(&e.node)) {
                    findTensors(par->inner);
                } else if (const auto* call = std::get_if<ExprCall>(&e.node)) {
                    for (const auto& arg : call->args) {
                        findTensors(arg);
                    }
                }
            };

            for (const auto& clause : eq.clauses) {
                // Check both expression and guard for tensor references
                findTensors(clause.expr);
                if (clause.guard.has_value()) {
                    findTensors(clause.guard.value());
                }
            }

            if (maxSize == 0) {
                // No tensors found in RHS, use a default size or error
                throw ExecutionError("GuardedClauseExecutor: cannot determine iteration size");
            }

            // Evaluate element-by-element to handle expressions like X[i] * X[i] correctly
            // Save any existing bindings for index variables
            std::vector<std::pair<std::string, Tensor>> savedBindings;
            for (const auto& varName : indexVars) {
                if (env.has(varName)) {
                    savedBindings.push_back({varName, env.lookup(varName)});
                }
            }

            // Create result tensor
            std::vector<float> resultValues;
            resultValues.reserve(maxSize);

            // Iterate through each index value
            for (int64_t idx = 0; idx < maxSize; ++idx) {
                // Bind index variables to current scalar value
                for (const auto& varName : indexVars) {
                    Tensor indexValue = torch::tensor(static_cast<float>(idx));
                    env.bind(varName, indexValue);
                }

                // Find first matching clause for this index
                bool matched = false;
                float value = 0.0f;

                for (const auto& clause : eq.clauses) {
                    if (!clause.expr) {
                        throw ExecutionError("GuardedClauseExecutor: null expression in clause");
                    }

                    // Check if guard matches (or no guard = always match)
                    bool guardMatches = true;
                    if (clause.guard.has_value()) {
                        Tensor guardResult = exprEval.evalExpr(clause.guard.value(), eq.lhs, env, backend);
                        // Convert to boolean: non-zero = true
                        float guardValue = guardResult.dim() == 0 ? guardResult.item<float>() : guardResult.sum().item<float>();
                        guardMatches = (guardValue != 0.0f);
                    }

                    if (guardMatches) {
                        // Evaluate expression for this index
                        // Pass eq.lhs so ExpressionExecutor can check for bound indices
                        Tensor exprResult = exprEval.evalExpr(clause.expr, eq.lhs, env, backend);
                        value = exprResult.dim() == 0 ? exprResult.item<float>() : exprResult.sum().item<float>();
                        matched = true;
                        break; // First-match-wins
                    }
                }

                if (!matched) {
                    throw ExecutionError("GuardedClauseExecutor: no clause matched for index " + std::to_string(idx));
                }

                resultValues.push_back(value);
            }

            // Restore saved bindings
            for (const auto& [varName, tensor] : savedBindings) {
                env.bind(varName, tensor);
            }

            // Create result tensor from collected values
            Tensor result = torch::tensor(resultValues);
            return result;
        }

        // No indices on LHS - simpler case
        std::optional<Tensor> result;
        std::optional<Tensor> usedMask;

        for (const auto& clause : eq.clauses) {
            if (!clause.expr) {
                throw ExecutionError("GuardedClauseExecutor: null expression in clause");
            }

            Tensor exprValue = exprEval.evalExpr(clause.expr, eq.lhs, env, backend);
            Tensor clauseMask;

            if (clause.guard.has_value()) {
                clauseMask = exprEval.evalExpr(clause.guard.value(), eq.lhs, env, backend);

                if (exprValue.sizes() != clauseMask.sizes()) {
                    try {
                        clauseMask = torch::broadcast_to(clauseMask, exprValue.sizes());
                    } catch (const std::exception& e) {
                        try {
                            exprValue = torch::broadcast_to(exprValue, clauseMask.sizes());
                            clauseMask = torch::broadcast_to(clauseMask, exprValue.sizes());
                        } catch (const std::exception& e2) {
                            throw ExecutionError("GuardedClauseExecutor: cannot broadcast guard mask and expression to compatible shapes");
                        }
                    }
                }
            } else {
                clauseMask = torch::ones_like(exprValue);
            }

            // Exclude elements already assigned (first-match-wins)
            if (usedMask.has_value()) {
                clauseMask = clauseMask * (1.0 - *usedMask);
            }

            Tensor contribution = exprValue * clauseMask;

            if (!result.has_value()) {
                result = contribution;
                usedMask = clauseMask;
            } else {
                if (result->sizes() != contribution.sizes()) {
                    try {
                        contribution = torch::broadcast_to(contribution, result->sizes());
                        clauseMask = torch::broadcast_to(clauseMask, result->sizes());
                    } catch (const std::exception& e) {
                        try {
                            result = torch::broadcast_to(*result, contribution.sizes());
                            usedMask = torch::broadcast_to(*usedMask, contribution.sizes());
                        } catch (const std::exception& e2) {
                            throw ExecutionError("GuardedClauseExecutor: cannot broadcast clause results to compatible shapes");
                        }
                    }
                }
                *result = *result + contribution;
                *usedMask = *usedMask + clauseMask;
            }
        }

        if (!result.has_value()) {
            throw ExecutionError("GuardedClauseExecutor: no clauses produced a result");
        }

        if (eq.lhs.indices.empty() && result->dim() > 0) {
            *result = torch::sum(*result);
        }

        return *result;
    }

    std::string GuardedClauseExecutor::name() const {
        return "GuardedClauseExecutor";
    }

    int GuardedClauseExecutor::priority() const {
        return 50; // Medium priority - after specific executors, before general ExpressionExecutor
    }

} // namespace tl
