#pragma once

#include "TL/Runtime/StatementPreprocessor.hpp"
#include <set>
#include <string>
#include <optional>
#include <utility>

namespace tl {

    /**
     * @brief Preprocessor for virtual index expansion
     *
     * Virtual indices (*t, *t+1, *t-1) represent recurrent state without
     * allocating a full time dimension. This preprocessor expands equations
     * with virtual indices into multiple concrete assignments.
     *
     * Example transformation:
     *   Input:  avg[*t+1] = (1.0 - alpha) * avg[*t] + alpha * data[t]
     *   Output: avg[1] = (1.0 - alpha) * avg[0] + alpha * data[0]
     *           avg[2] = (1.0 - alpha) * avg[1] + alpha * data[1]
     *           avg[3] = (1.0 - alpha) * avg[2] + alpha * data[2]
     *           ...
     *
     * The preprocessor:
     * 1. Detects virtual indices in LHS
     * 2. Finds the "driving" regular index (e.g., 't' when we have '*t')
     * 3. Determines iteration count from tensor dimensions
     * 4. Pre-allocates storage for all time steps
     * 5. Expands into N concrete equations with substituted indices
     *
     * Design note: This is a preprocessor rather than an executor because
     * it performs syntactic transformation (desugaring) rather than execution.
     * Each expanded statement is then routed to appropriate executors normally.
     */
    class VirtualIndexPreprocessor : public StatementPreprocessor {
    public:
        bool shouldPreprocess(const Statement& st, const Environment& env) const override;

        std::vector<Statement> preprocess(const Statement& st, Environment& env) override;

        // Batch preprocessing for intra-timestep dependencies (multi-layer RNNs)
        static std::vector<Statement> preprocessBatch(const std::vector<Statement>& statements, Environment& env);

        std::string name() const override { return "VirtualIndexPreprocessor"; }

        int priority() const override { return 5; } // High priority - expand before other preprocessing

    private:
        // Helper to check if an index is virtual
        static bool isVirtualIndex(const Index& idx);

        // Helper to extract virtual index info (name, offset)
        static std::optional<std::pair<std::string, int>> getVirtualIndexInfo(const Index& idx);

        // Helper to find all virtual indices in a tensor ref
        static std::vector<std::pair<std::string, int>> findVirtualIndices(const TensorRef& ref);

        // Helper to find all regular indices in an equation
        static std::set<std::string> findRegularIndices(const TensorEquation& eq);

        // Helper to determine iteration count for a regular index
        static int getIterationCount(const std::string& indexName, const Environment& env,
                                      const TensorEquation& eq);

        // Helper to pre-allocate tensor storage for virtual time steps (deprecated for Mode B)
        static void preallocateStorage(const TensorEquation& eq, Environment& env,
                                       const std::string& virtualIndexName,
                                       int lhsOffset, int iterationCount);

        // Helper to ensure tensor has minimum slots in virtual dimension (Mode B)
        static void ensureMinimumVirtualSlots(const TensorEquation& eq, Environment& env,
                                               const std::string& virtualIndexName,
                                               int minSlots);
    };

} // namespace tl
