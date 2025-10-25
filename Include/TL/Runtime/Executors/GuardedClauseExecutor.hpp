#pragma once

#include "TL/Runtime/Executor.hpp"

namespace tl {
    /**
     * @brief Executor for guarded clause tensor equations
     *
     * Handles equations with multiple clauses and optional guards:
     *   A[i] = Expr1 : Guard1 | Expr2 : Guard2 | Expr3
     *
     * Semantics: All matching clauses contribute additively (superposition).
     * Each guarded clause lowers to: Expr * mask(Guard)
     */
    class GuardedClauseExecutor : public TensorEquationExecutor {
    public:
        bool canExecute(const TensorEquation& eq, const Environment& env) const override;
        Tensor execute(const TensorEquation& eq, Environment& env, TensorBackend& backend) override;
        std::string name() const override;
        int priority() const override;
    };
} // namespace tl
