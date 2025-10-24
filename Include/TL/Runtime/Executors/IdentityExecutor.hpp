#pragma once

#include "TL/Runtime/Executor.hpp"

namespace tl {

    /**
     * @brief Handles identity/copy assignment from tensor ref
     *
     * Example: Y = X or Y[i] = X[i]
     */
    class IdentityExecutor : public TensorEquationExecutor {
    public:
        bool canExecute(const TensorEquation &eq, const Environment &env) const override;
        Tensor execute(const TensorEquation &eq, Environment &env, TensorBackend &backend) override;
        std::string name() const override;
        int priority() const override;
    };
} // namespace tl
