#include "TL/Runtime/Executors/IdentityExecutor.hpp"
#include "TL/vm.hpp"
#include <torch/torch.h>

namespace tl {

    bool IdentityExecutor::canExecute(const TensorEquation &eq, const Environment &env) const {
        // Only handle standard assignment (=), not pooling operators (+=, max=, min=, avg=)
        if (!eq.projection.empty() && eq.projection != "=") return false;

        // Check if RHS is a direct tensor reference (not einsum, not expression, just a ref)
        if (!eq.rhs) return false;

        const Expr &e = *eq.rhs;
        const auto *eref = std::get_if<ExprTensorRef>(&e.node);
        if (!eref) return false;

        // Make sure it's not a reduction (those are handled by ReductionExecutor)
        // Reduction is when LHS has no indices but RHS does
        if (eq.lhs.indices.empty() && !eref->ref.indices.empty()) {
            return false;
        }

        // It's an identity if the RHS tensor exists
        const std::string srcName = eref->ref.name.name;
        return env.has(srcName);
    }

    Tensor IdentityExecutor::execute(const TensorEquation &eq, Environment &env, TensorBackend &backend) {
        const Expr &e = *eq.rhs;
        const auto *eref = std::get_if<ExprTensorRef>(&e.node);
        if (!eref) {
            throw ExecutionError("IdentityExecutor: expected tensor ref on RHS");
        }

        return env.lookup(eref->ref);
    }

    std::string IdentityExecutor::name() const {
        return "IdentityExecutor";
    }

    int IdentityExecutor::priority() const {
        return 80; // Lower priority - check after most specific executors
    }

} // namespace tl
