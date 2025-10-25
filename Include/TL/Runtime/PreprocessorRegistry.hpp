#pragma once

#include "TL/Runtime/StatementPreprocessor.hpp"
#include <vector>
#include <memory>
#include <algorithm>

namespace tl {

    /**
     * @brief Registry and orchestrator for statement preprocessors
     *
     * Manages a chain of preprocessors that transform statements
     * before they are executed. Preprocessors are applied in priority
     * order (lowest priority value first).
     */
    class PreprocessorRegistry {
    public:
        PreprocessorRegistry() = default;

        /**
         * @brief Register a preprocessor
         * @param preprocessor The preprocessor to register
         *
         * Preprocessors are automatically sorted by priority after registration.
         */
        void registerPreprocessor(PreprocessorPtr preprocessor) {
            preprocessors_.push_back(std::move(preprocessor));
            sortByPriority();
        }

        /**
         * @brief Preprocess a statement through all registered preprocessors
         * @param st The statement to preprocess
         * @param env The environment (mutable)
         * @return Vector of concrete statements after all preprocessing
         *
         * Each preprocessor may expand a statement into multiple statements.
         * The result is a flattened list of all concrete statements.
         *
         * Example:
         *   Input: avg[*t+1] = expr
         *   After VirtualIndexPreprocessor:
         *     avg[0] = expr_0
         *     avg[1] = expr_1
         *     avg[2] = expr_2
         *     ...
         */
        std::vector<Statement> preprocess(const Statement& st, Environment& env) {
            std::vector<Statement> current = {st};

            // Apply each preprocessor in priority order
            for (const auto& preprocessor : preprocessors_) {
                std::vector<Statement> next;

                for (const auto& stmt : current) {
                    if (preprocessor->shouldPreprocess(stmt, env)) {
                        auto expanded = preprocessor->preprocess(stmt, env);
                        next.insert(next.end(), expanded.begin(), expanded.end());
                    } else {
                        next.push_back(stmt);
                    }
                }

                current = std::move(next);
            }

            return current;
        }

        /**
         * @brief Get the number of registered preprocessors
         */
        size_t size() const { return preprocessors_.size(); }

        /**
         * @brief Enable/disable debug output
         */
        void setDebug(bool enabled) { debug_ = enabled; }

        /**
         * @brief Set error output stream
         */
        void setErrOut(std::ostream* err) { err_out_ = err; }

    private:
        void sortByPriority() {
            std::sort(preprocessors_.begin(), preprocessors_.end(),
                [](const PreprocessorPtr& a, const PreprocessorPtr& b) {
                    return a->priority() < b->priority();
                });
        }

        std::vector<PreprocessorPtr> preprocessors_;
        bool debug_ = false;
        std::ostream* err_out_ = &std::cerr;
    };

} // namespace tl
