#pragma once

#include "TL/Runtime/Executor.hpp"
#include "TL/Runtime/ExecutorUtils.hpp"
#include <vector>
#include <algorithm>

namespace tl {

    /**
     * @brief Registry for managing tensor equation executors
     *
     * Uses the Chain of Responsibility pattern to find the appropriate executor
     */
    class ExecutorRegistry {
    public:

        /**
         * @brief Register an executor (takes ownership)
         */
        void registerExecutor(ExecutorPtr executor) {
            executors_.push_back(std::move(executor));
            // Sort by priority
            std::sort(executors_.begin(), executors_.end(),
                [](const ExecutorPtr& a, const ExecutorPtr& b) {
                    return a->priority() < b->priority();
                });
        }

        /**
         * @brief Find and execute the appropriate executor for an equation
         * @throws ExecutionError if no executor can handle the equation
         */
        Tensor execute(const TensorEquation& eq, Environment& env, TensorBackend& backend) {
            for (auto& executor : executors_) {
                if (executor->canExecute(eq, env)) {
                    if (debug_) {
                        *err_ << "[ExecutorRegistry] Using " << executor->name() << std::endl;
                    }
                    return executor->execute(eq, env, backend);
                }
            }

            throw ExecutionError("No executor found for equation: " + executor_utils::toString(eq));
        }

        void setDebug(bool debug) { debug_ = debug; }
        void setErrOut(std::ostream* err) { err_ = err; }

    private:
        std::vector<ExecutorPtr> executors_;
        bool debug_ = false;
        std::ostream* err_ = &std::cerr;
    };
} // namespace tl