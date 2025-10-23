#pragma once

#include "TL/Runtime/Executor.hpp"

namespace tl {

    /**
     * @brief Handles pooling operations with projections: +=, avg=, max=, min=
     *
     * Example: Y[i,j/2] avg= X[i,j]
     */
    class PoolingExecutor : public TensorEquationExecutor {
    public:
        bool canExecute(const TensorEquation &eq, const Environment &env) const override;
        Tensor execute(const TensorEquation &eq, Environment &env, TensorBackend &backend) override;
        std::string name() const override;
        int priority() const override;
    };
} // namespace tl
