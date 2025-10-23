#pragma once

#include "TL/AST.hpp"
#include "TL/core.hpp"
#include "TL/backend.hpp"
#include <string>
#include <memory>
#include <stdexcept>

namespace tl {
    class Environment; // Forward declaration

    /**
     * @brief Exception thrown during tensor equation execution
     */
    class ExecutionError : public std::runtime_error {
    public:
        explicit ExecutionError(const std::string& msg) : std::runtime_error(msg) {}
    };

    /**
     * @brief Base interface for tensor equation executors
     *
     * Each executor handles one specific type of tensor equation execution.
     * Executors follow the Strategy pattern and are selected via Chain of Responsibility.
     */
    class TensorEquationExecutor {
    public:
        virtual ~TensorEquationExecutor() = default;

        /**
         * @brief Check if this executor can handle the given equation
         * @param eq The tensor equation to check
         * @param env The current environment (for checking tensor existence, etc.)
         * @return true if this executor can handle the equation
         */
        virtual bool canExecute(const TensorEquation& eq, const Environment& env) const = 0;

        /**
         * @brief Execute the tensor equation
         * @param eq The tensor equation to execute
         * @param env The environment (mutable, for tensor storage)
         * @param backend The tensor backend (e.g., LibTorch)
         * @return The resulting tensor
         * @throws ExecutionError if execution fails
         */
        virtual Tensor execute(const TensorEquation& eq, Environment& env,
            TensorBackend& backend) = 0;

        /**
         * @brief Get the name of this executor (for debugging)
         */
        virtual std::string name() const = 0;

        /**
         * @brief Get priority (lower = checked first)
         *
         * Some executors are more specific than others. For example,
         * ScalarAssignExecutor should be checked before ExpressionExecutor.
         */
        virtual int priority() const { return 100; }
    };

    using ExecutorPtr = std::unique_ptr<TensorEquationExecutor>;
} // namespace tl
