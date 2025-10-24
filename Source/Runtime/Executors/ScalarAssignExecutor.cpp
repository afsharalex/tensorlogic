#include "TL/Runtime/Executors/ScalarAssignExecutor.hpp"
#include "TL/Runtime/ExecutorUtils.hpp"
#include "TL/vm.hpp"
#include <stdexcept>

namespace tl {

    bool ScalarAssignExecutor::canExecute(const TensorEquation &eq, const Environment &env) const {
        // Only handle standard assignment (=), not pooling operators
        if (!eq.projection.empty() && eq.projection != "=") {
            return false;
        }

        // Check if RHS is a numeric literal
        auto numeric_value = executor_utils::tryParseNumericLiteral(eq.rhs);
        if (!numeric_value) {
            return false;
        }

        // Must have at least one index (otherwise it's a scalar binding, not element assignment)
        if (eq.lhs.indices.empty()) {
            return false;
        }

        // Check if LHS has numeric indices
        auto indices = executor_utils::tryGatherNumericIndices(eq.lhs, env);
        return indices.has_value();
    }

    Tensor ScalarAssignExecutor::execute(const TensorEquation &eq, Environment &env, TensorBackend &backend) {
        // Parse numeric value
        auto numeric_value = executor_utils::tryParseNumericLiteral(eq.rhs);
        if (!numeric_value) {
            throw ExecutionError("Expected numeric literal in RHS");
        }

        // Gather indices
        auto indices = executor_utils::tryGatherNumericIndices(eq.lhs, env);
        if (!indices) {
            throw ExecutionError("Expected numeric indices in LHS");
        }

        const std::string& lhs_name = eq.lhs.name.name;

        // Ensure tensor is large enough (uses helper to avoid duplication)
        Tensor tensor = executor_utils::ensureTensorSize(lhs_name, *indices, env);

        // Set value at indices - handle multi-dimensional case
        const auto opts = torch::TensorOptions().dtype(torch::kFloat32);
        std::vector<torch::indexing::TensorIndex> elem_idx;
        elem_idx.reserve(indices->size());
        for (int64_t v : *indices) {
            elem_idx.emplace_back(v);
        }
        tensor.index_put_(elem_idx, torch::tensor(static_cast<float>(*numeric_value), opts));

        return tensor;
    }

    std::string ScalarAssignExecutor::name() const {
        return "ScalarAssignExecutor";
    }

    int ScalarAssignExecutor::priority() const {
        return 10; // High priority - check before general expression executor
    }

}