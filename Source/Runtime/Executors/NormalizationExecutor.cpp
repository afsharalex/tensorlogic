#include "TL/Runtime/Executors/NormalizationExecutor.hpp"
#include "TL/Runtime/Executors/ExpressionExecutor.hpp"
#include "TL/Runtime/ExecutorUtils.hpp"
#include "TL/vm.hpp"
#include <torch/torch.h>

namespace tl {

    bool NormalizationExecutor::canExecute(const TensorEquation &eq, const Environment &env) const {
        // Only handle standard assignment (=)
        if (!eq.projection.empty() && eq.projection != "=") {
            return false;
        }

        // Must have exactly one clause without a guard
        if (eq.clauses.size() != 1 || eq.clauses[0].guard.has_value()) {
            return false;
        }

        // Must have an RHS expression
        if (!eq.clauses[0].expr) return false;

        // Check if LHS has a normalized index
        return findNormalizedDimension(eq.lhs).has_value();
    }

    Tensor NormalizationExecutor::execute(const TensorEquation &eq, Environment &env, TensorBackend &backend) {
        // Find which dimension is normalized
        auto normDimOpt = findNormalizedDimension(eq.lhs);
        if (!normDimOpt) {
            throw ExecutionError("No normalized index found in NormalizationExecutor");
        }
        int64_t normDim = static_cast<int64_t>(normDimOpt.value());

        // Evaluate the RHS expression to get raw values
        // We reuse ExpressionExecutor's evalExpr method by creating a temporary instance
        ExpressionExecutor exprExec;
        Tensor rawValues = exprExec.evalExpr(eq.clauses[0].expr, eq.lhs, env, backend);

        // Handle edge case: if raw values are a scalar, return 1.0 (normalized scalar is always 1.0)
        if (rawValues.dim() == 0) {
            return torch::tensor(1.0f);
        }

        // Validate that the normalized dimension exists in the tensor
        if (normDim >= rawValues.dim()) {
            throw ExecutionError("Normalized dimension " + std::to_string(normDim) +
                               " out of range for tensor with " + std::to_string(rawValues.dim()) + " dimensions");
        }

        // Check if RHS is an explicit softmax call
        // If so, the expression already includes softmax, so we don't apply it again
        bool isExplicitSoftmax = false;
        if (auto* call = std::get_if<ExprCall>(&eq.clauses[0].expr->node)) {
            if (call->func.name == "softmax") {
                isExplicitSoftmax = true;
            }
        }

        Tensor normalized;
        if (isExplicitSoftmax) {
            // Expression already contains softmax, just use the result
            // But we need to ensure it was applied to the correct dimension
            normalized = rawValues;
        } else {
            // Apply softmax along the normalized dimension
            normalized = torch::softmax(rawValues, normDim);
        }

        return normalized;
    }

    std::optional<size_t> NormalizationExecutor::findNormalizedDimension(const TensorRef& lhs) const {
        for (size_t i = 0; i < lhs.indices.size(); ++i) {
            if (auto* idx = std::get_if<Index>(&lhs.indices[i].value)) {
                if (idx->normalized) {
                    return i;
                }
            }
        }
        return std::nullopt;
    }

    std::string NormalizationExecutor::name() const {
        return "NormalizationExecutor";
    }

    int NormalizationExecutor::priority() const {
        return 40; // After identity/list (30), before einsum/expression (50+)
    }

} // namespace tl
