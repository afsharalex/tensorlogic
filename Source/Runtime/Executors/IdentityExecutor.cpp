#include "TL/Runtime/Executors/IdentityExecutor.hpp"
#include "TL/vm.hpp"
#include <torch/torch.h>

namespace tl {

    bool IdentityExecutor::canExecute(const TensorEquation &eq, const Environment &env) const {
        // Only handle standard assignment (=), not pooling operators (+=, max=, min=, avg=)
        if (!eq.projection.empty() && eq.projection != "=") return false;

        // Only handle single-clause, no-guard equations
        if (eq.clauses.size() != 1 || eq.clauses[0].guard.has_value()) {
            return false;
        }

        // Check if RHS is a direct tensor reference (not einsum, not expression, just a ref)
        if (!eq.clauses[0].expr) return false;

        const Expr &e = *eq.clauses[0].expr;
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
        const Expr &e = *eq.clauses[0].expr;
        const auto *eref = std::get_if<ExprTensorRef>(&e.node);
        if (!eref) {
            throw ExecutionError("IdentityExecutor: expected tensor ref on RHS");
        }

        const std::string srcName = eref->ref.name.name;
        Tensor src = env.lookup(srcName);

        // Apply RHS indices if present
        if (!eref->ref.indices.empty()) {
            std::vector<torch::indexing::TensorIndex> indexArgs;

            for (const auto& idx : eref->ref.indices) {
                if (auto* num = std::get_if<NumberLiteral>(&idx.value)) {
                    // Concrete index: use the number
                    int64_t val = std::stoll(num->text);
                    indexArgs.push_back(val);
                } else if (std::holds_alternative<Identifier>(idx.value)) {
                    // Free variable: use full slice
                    indexArgs.push_back(torch::indexing::Slice());
                } else if (std::holds_alternative<VirtualIndex>(idx.value)) {
                    // Virtual index: should have been preprocessed away
                    throw ExecutionError("IdentityExecutor: unexpected virtual index in RHS");
                } else {
                    throw ExecutionError("IdentityExecutor: unsupported index type in RHS");
                }
            }

            return src.index(indexArgs);
        }

        return src;
    }

    std::string IdentityExecutor::name() const {
        return "IdentityExecutor";
    }

    int IdentityExecutor::priority() const {
        return 80; // Lower priority - check after most specific executors
    }

} // namespace tl
