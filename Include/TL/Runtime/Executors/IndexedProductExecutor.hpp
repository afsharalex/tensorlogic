#pragma once

#include "TL/Runtime/Executor.hpp"

namespace tl {

    /**
     * @brief Handles implicit indexed product lowered to einsum
     *
     * Example: C[i,k] = A[i,j] * B[j,k]
     */
    class IndexedProductExecutor : public TensorEquationExecutor {
    public:
        bool canExecute(const TensorEquation &eq, const Environment &env) const override;
        Tensor execute(const TensorEquation &eq, Environment &env, TensorBackend &backend) override;
        std::string name() const override;
        int priority() const override;
    };
} // namespace tl
