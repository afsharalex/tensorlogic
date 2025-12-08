#pragma once

namespace tl {

/**
 * @brief Configuration constants for the TensorLogic Virtual Machine
 *
 * This namespace contains tunable parameters that control VM behavior,
 * particularly for iterative algorithms and convergence detection.
 */
namespace VMConfig {

/**
 * @brief Maximum iterations for fixed-point loops to prevent infinite loops
 *
 * Fixed-point loops will terminate after this many iterations even if
 * convergence has not been reached. This is a safety mechanism to prevent
 * runaway computation.
 *
 * Default: 10,000 iterations
 */
constexpr int ABSOLUTE_MAX_ITERS = 10'000;

/**
 * @brief Number of consecutive iterations with change below tolerance needed for convergence
 *
 * A fixed-point loop is considered converged only after this many consecutive
 * iterations where the maximum change in any tensor element is below CONVERGENCE_TOLERANCE.
 * This helps filter out noise and ensures stability.
 *
 * Default: 10 consecutive stable iterations
 */
constexpr int MAX_CONSECUTIVE_STABLE = 10;

/**
 * @brief Maximum absolute change per element to be considered stable
 *
 * During fixed-point iteration, if the maximum absolute change across all
 * elements of the monitored tensor is below this threshold, the iteration
 * is considered stable. After MAX_CONSECUTIVE_STABLE stable iterations,
 * the loop converges.
 *
 * Default: 0.0001 (1e-4)
 */
constexpr float CONVERGENCE_TOLERANCE = 0.0001f;

} // namespace VMConfig
} // namespace tl
