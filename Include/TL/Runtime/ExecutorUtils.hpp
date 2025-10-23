#pragma once

#include "TL/AST.hpp"
#include "TL/core.hpp"
#include <vector>
#include <optional>

namespace tl {
    class Environment; // Forward declaration

    namespace executor_utils {

        /**
         * @brief Try to parse a numeric literal from an expression
         */
        std::optional<double> tryParseNumericLiteral(const ExprPtr& expr);

        /**
         * @brief Try to gather numeric indices from a tensor reference
         */
        std::optional<std::vector<int64_t>> tryGatherNumericIndices(
            const TensorRef& ref, const Environment& env);

        /**
         * @brief Resolve indices to concrete positions, creating labels as needed
         *
         * This function handles both numeric literals and identifier labels.
         * If an identifier hasn't been mapped to an index yet, it creates
         * a new label mapping using env.internLabel().
         *
         * Returns empty vector if indices cannot be resolved.
         */
        std::vector<int64_t> resolveIndicesCreatingLabels(
            const TensorRef& ref, Environment& env);

        /**
         * @brief Ensure tensor is large enough for given indices, resize if needed
         *
         * This eliminates code duplication - used by multiple executors
         */
        Tensor ensureTensorSize(const std::string& name,
            const std::vector<int64_t>& required_indices,
            Environment& env);

        /**
         * @brief Check if all terms in a Datalog atom are constants
         */
        bool allConstants(const DatalogAtom& atom);

        /**
         * @brief Try to parse an einsum specification from an expression
         */
        bool tryParseEinsumCall(const ExprPtr& expr,
            std::string& spec_out,
            std::vector<Tensor>& inputs_out,
            const Environment& env);

        /**
         * @brief Convert TensorEquation to string for error messages
         */
        inline std::string toString(const TensorEquation& eq) {
            return toString(Statement(eq));
        }

        /**
         * @brief Get tensor value for ref, applying numeric indices and leaving symbolic as slices
         */
        Tensor valueForRef(const TensorRef& ref, Environment& env);

        /**
         * @brief Try to lower indexed product (A[i,j] * B[j,k]) to einsum
         */
        bool tryLowerIndexedProductToEinsum(const TensorRef& lhs,
                                           const ExprPtr& rhs,
                                           std::string& spec_out,
                                           std::vector<Tensor>& inputs_out,
                                           Environment& env);
    } // namespace executor_utils
} // namespace tl