#pragma once

#include "TL/Runtime/Executor.hpp"

namespace tl {

    /**
     * @brief Handles general expression evaluation
     *
     * This is a catch-all executor for expressions that aren't handled
     * by more specific executors. It handles:
     * - Binary operations: +, -, *, /
     * - Function calls: relu, sigmoid, tanh, step, sqrt, abs, exp, softmax
     * - Tensor references with indexing
     *
     * Examples:
     *   Y[i] = A[i] + B[i]
     *   Z[i] = relu(W[i,j] * X[j] + b[i])
     *   Y[i] = sigmoid(X[i])
     */
    class ExpressionExecutor : public TensorEquationExecutor {
    public:
        bool canExecute(const TensorEquation &eq, const Environment &env) const override;
        Tensor execute(const TensorEquation &eq, Environment &env, TensorBackend &backend) override;
        std::string name() const override;
        int priority() const override;

    private:
        /**
         * @brief Recursively evaluate an expression
         */
        Tensor evalExpr(const ExprPtr& ep, const TensorRef& lhsCtx,
                       Environment& env, TensorBackend& backend) const;
    };

} // namespace tl
