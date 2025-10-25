#include "TL/Runtime/Executors/ReductionExecutor.hpp"
#include "TL/vm.hpp"
#include <torch/torch.h>

namespace tl {

    bool ReductionExecutor::canExecute(const TensorEquation &eq, const Environment &env) const {
        // Only handle standard assignment (=), not pooling operators
        if (!eq.projection.empty() && eq.projection != "=") {
            return false;
        }

        // Only handle single-clause, no-guard equations
        if (eq.clauses.size() != 1 || eq.clauses[0].guard.has_value()) {
            return false;
        }

        // Check if LHS has no indices (scalar) and RHS is a tensor ref with indices
        if (!eq.lhs.indices.empty()) return false;
        if (!eq.clauses[0].expr) return false;

        const Expr &e = *eq.clauses[0].expr;
        const auto *eref = std::get_if<ExprTensorRef>(&e.node);
        if (!eref) return false;

        // RHS must have indices (otherwise it's just identity, not reduction)
        return !eref->ref.indices.empty();
    }

    Tensor ReductionExecutor::execute(const TensorEquation &eq, Environment &env, TensorBackend &backend) {
        const Expr &e = *eq.clauses[0].expr;
        const auto *eref = std::get_if<ExprTensorRef>(&e.node);
        if (!eref) {
            throw ExecutionError("ReductionExecutor: expected tensor ref on RHS");
        }

        // Value for ref handles numeric indexing, symbolic indices as slices
        torch::Tensor base = env.lookup(eref->ref);

        // Ensure at least as many dims as indices by unsqueezing leading dims
        while (base.dim() < static_cast<int64_t>(eref->ref.indices.size())) {
            base = base.unsqueeze(0);
        }

        if (eref->ref.indices.empty()) {
            return base;
        }

        // Apply indices
        using torch::indexing::Slice;
        using torch::indexing::TensorIndex;
        std::vector<TensorIndex> idx;
        idx.reserve(eref->ref.indices.size());
        for (const auto& ind : eref->ref.indices) {
            if (const auto* num = std::get_if<NumberLiteral>(&ind.value)) {
                long long v = 0;
                try { v = std::stoll(num->text); } catch (...) { v = 0; }
                idx.emplace_back(static_cast<int64_t>(v));
            } else {
                idx.emplace_back(Slice());
            }
        }

        torch::Tensor indexed = base.index(idx);
        return torch::sum(indexed);
    }

    std::string ReductionExecutor::name() const {
        return "ReductionExecutor";
    }

    int ReductionExecutor::priority() const {
        return 40;
    }

} // namespace tl
