#pragma once

#include "TL/Runtime/Executor.hpp"

namespace tl {

    /**
     * @brief Handles reduction operations (sum over indices)
     *
     * Example: s = Y[i] (LHS scalar, RHS has indices -> sum over all)
     */
    class ReductionExecutor : public TensorEquationExecutor {
    public:
        bool canExecute(const TensorEquation &eq, const Environment &env) const override;
        Tensor execute(const TensorEquation &eq, Environment &env, TensorBackend &backend) override;
        std::string name() const override;
        int priority() const override;
    };
} // namespace tl
