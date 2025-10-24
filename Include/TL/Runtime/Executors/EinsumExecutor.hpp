#pragma once

#include "TL/Runtime/Executor.hpp"

namespace tl {

    /**
     * @brief Handles Einstein summation operations
     *
     * Example: C[i,k] = A[i,j] B[j,k] or einsum("ij,jk->ik", A, B)
     */
    class EinsumExecutor : public TensorEquationExecutor {
    public:
        bool canExecute(const TensorEquation &eq, const Environment &env) const override;
        Tensor execute(const TensorEquation &eq, Environment &env, TensorBackend &backend) override;
        std::string name() const override;
        int priority() const override;
    };
} // namespace tl
