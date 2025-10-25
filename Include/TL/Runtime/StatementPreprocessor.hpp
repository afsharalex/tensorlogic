#pragma once

#include "TL/AST.hpp"
#include <vector>
#include <memory>

namespace tl {
    class Environment; // Forward declaration

    /**
     * @brief Base interface for statement preprocessors
     *
     * Preprocessors transform statements before execution. They handle
     * syntactic sugar and desugaring operations that expand into multiple
     * concrete statements.
     *
     * Examples:
     * - Virtual indices: avg[*t+1] = expr → multiple avg[0]=..., avg[1]=..., etc.
     * - Loops/macros: for i in range(10) → unrolled statements
     * - Template expansion: generic patterns → concrete instances
     *
     * Design rationale:
     * Preprocessors operate at the AST level, performing source-to-source
     * transformations. This is distinct from executors which interpret
     * concrete statements. This separation avoids circular dependencies
     * where a "meta-executor" would need to recursively invoke the
     * executor registry.
     */
    class StatementPreprocessor {
    public:
        virtual ~StatementPreprocessor() = default;

        /**
         * @brief Check if this preprocessor should handle the statement
         * @param st The statement to check
         * @param env The current environment (for context)
         * @return true if this preprocessor should transform the statement
         */
        virtual bool shouldPreprocess(const Statement& st, const Environment& env) const = 0;

        /**
         * @brief Transform a statement into zero or more concrete statements
         * @param st The statement to transform
         * @param env The environment (mutable, may be updated during preprocessing)
         * @return Vector of concrete statements (may be empty, single, or multiple)
         *
         * If shouldPreprocess returns false, this should return {st} unchanged.
         * If preprocessing expands the statement, return multiple statements.
         * If preprocessing removes the statement, return empty vector.
         */
        virtual std::vector<Statement> preprocess(const Statement& st, Environment& env) = 0;

        /**
         * @brief Get priority (lower = processed first)
         *
         * Some preprocessors should run before others. For example,
         * macro expansion might need to run before virtual index expansion.
         */
        virtual int priority() const { return 100; }

        /**
         * @brief Get the name of this preprocessor (for debugging)
         */
        virtual std::string name() const = 0;
    };

    using PreprocessorPtr = std::unique_ptr<StatementPreprocessor>;
} // namespace tl
