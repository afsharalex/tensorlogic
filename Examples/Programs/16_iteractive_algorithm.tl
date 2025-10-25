// ============================================
// Fixed-Point Iteration with Virtual Index
// ============================================
// Solve a system iteratively until convergence
// x^{t+1} = f(x^t)
//
// Example: Solve x = cos(x) using iteration

// ============================================
// Initial Guess
// ============================================
x[0] = 1.0  // Starting point

// ============================================
// Iterative Update
// ============================================
// Virtual index *t means we only store the current value
// Each iteration overwrites the previous

// Fixed-point iteration: x^{t+1} = cos(x^t)
x[*t+1] = cos(x[*t])

// This executes as:
// Iteration 1: x[*1] = cos(x[0]) = cos(1.0) ≈ 0.540
// Iteration 2: x[*2] = cos(x[*1]) = cos(0.540) ≈ 0.857
// Iteration 3: x[*3] = cos(x[*2]) = cos(0.857) ≈ 0.654
// ...continues until convergence or max iterations
// Final value converges to x ≈ 0.739 (solution to x = cos(x))

// Query after N iterations (implementation-dependent)
x[*10]?  // Value after 10 iterations