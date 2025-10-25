#include "TL/Runtime/Executors/EinsumExecutor.hpp"
#include "TL/Runtime/ExecutorUtils.hpp"
#include "TL/vm.hpp"
#include <torch/torch.h>

namespace tl {

    bool EinsumExecutor::canExecute(const TensorEquation &eq, const Environment &env) const {
        // Only handle standard assignment (=), not pooling operators
        if (!eq.projection.empty() && eq.projection != "=") {
            return false;
        }

        // Only handle single-clause, no-guard equations
        if (eq.clauses.size() != 1 || eq.clauses[0].guard.has_value()) {
            return false;
        }

        if (!eq.clauses[0].expr) return false;

        std::string spec;
        std::vector<Tensor> inputs;

        // Check if it's an explicit einsum call
        return executor_utils::tryParseEinsumCall(eq.clauses[0].expr, spec, inputs, env);
    }

    Tensor EinsumExecutor::execute(const TensorEquation &eq, Environment &env, TensorBackend &backend) {
        std::string spec;
        std::vector<Tensor> inputs;

        if (!executor_utils::tryParseEinsumCall(eq.clauses[0].expr, spec, inputs, env)) {
            throw ExecutionError("Einsum executor: failed to parse einsum call");
        }

        return backend.einsum(spec, inputs);
    }

    std::string EinsumExecutor::name() const {
        return "EinsumExecutor";
    }

    int EinsumExecutor::priority() const {
        return 30; // Medium priority
    }

} // namespace tl
