#include "TL/Runtime/Executors/IndexedProductExecutor.hpp"
#include "TL/Runtime/ExecutorUtils.hpp"
#include "TL/vm.hpp"
#include <torch/torch.h>

namespace tl {

    bool IndexedProductExecutor::canExecute(const TensorEquation &eq, const Environment &env) const {
        // Only handle standard assignment (=), not pooling operators
        if (!eq.projection.empty() && eq.projection != "=") {
            return false;
        }

        // Only handle single-clause, no-guard equations
        if (eq.clauses.size() != 1 || eq.clauses[0].guard.has_value()) {
            return false;
        }

        if (!eq.clauses[0].expr) return false;

        // Try to lower indexed product to einsum
        std::string spec;
        std::vector<Tensor> inputs;
        Environment& env_mut = const_cast<Environment&>(env); // Safe: tryLower doesn't modify for checking

        return executor_utils::tryLowerIndexedProductToEinsum(eq.lhs, eq.clauses[0].expr, spec, inputs, env_mut);
    }

    Tensor IndexedProductExecutor::execute(const TensorEquation &eq, Environment &env, TensorBackend &backend) {
        std::string spec;
        std::vector<Tensor> inputs;

        if (!executor_utils::tryLowerIndexedProductToEinsum(eq.lhs, eq.clauses[0].expr, spec, inputs, env)) {
            throw ExecutionError("IndexedProductExecutor: failed to lower to einsum");
        }

        return backend.einsum(spec, inputs);
    }

    std::string IndexedProductExecutor::name() const {
        return "IndexedProductExecutor";
    }

    int IndexedProductExecutor::priority() const {
        return 35; // Between einsum (30) and reduction (40)
    }

} // namespace tl
