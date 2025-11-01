#pragma once

#include "TL/Runtime/Executor.hpp"

namespace tl {

    /**
     * @brief Handles tensor equations with normalized indices
     *
     * Normalized indices (i.) indicate that softmax normalization should be
     * applied over that dimension, creating a probability distribution.
     *
     * Example: Attn[q, k.] = softmax(Scores[q, k])
     * Result: For each q, the values over all k sum to 1.0
     *
     * From the paper (Domingos, 2025, Section 4.1, Table 2):
     * "The notation p'. indicates that p' is the index to be normalized
     *  (i.e., for each p, softmax is applied to the vector indexed by p')."
     */
    class NormalizationExecutor : public TensorEquationExecutor {
    public:
        bool canExecute(const TensorEquation &eq, const Environment &env) const override;
        Tensor execute(const TensorEquation &eq, Environment &env, TensorBackend &backend) override;
        std::string name() const override;
        int priority() const override;

    private:
        /**
         * @brief Find which dimension (position in LHS) is normalized
         * @return Index of the normalized dimension, or nullopt if none
         */
        std::optional<size_t> findNormalizedDimension(const TensorRef& lhs) const;
    };
} // namespace tl
