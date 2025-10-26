// ============================================
// Fixed-Point Iteration with Virtual Index
// and driving tensor (to control the number of
// iterations).
// ============================================
// Solve a system iteratively for N number of
// iterations.
// x^{t+1} = f(x^t)
//
// Example: Solve x = cos(x) using iteration

dummy[0] = 0.0
dummy[1] = 0.0
dummy[2] = 0.0
dummy[3] = 0.0
dummy[4] = 0.0
dummy[5] = 0.0
dummy[6] = 0.0
dummy[7] = 0.0
dummy[8] = 0.0
dummy[9] = 0.0
dummy[10] = 0.0
dummy[11] = 0.0
dummy[12] = 0.0
dummy[13] = 0.0
dummy[14] = 0.0
dummy[15] = 0.0
dummy[16] = 0.0
dummy[17] = 0.0
dummy[18] = 0.0
dummy[19] = 0.0

x[0] = 1.0  // Starting point

// ============================================
// Iterative Update
// ============================================
// Virtual index *t means we only store the current value
// Each iteration overwrites the previous

// Fixed-point iteration: x^{t+1} = cos(x^t)
x[*t+1] = cos(x[*t]) + dummy[t]

// This executes as:
// Iteration 1: x[*1] = cos(x[0]) = cos(1.0) ≈ 0.540
// Iteration 2: x[*2] = cos(x[*1]) = cos(0.540) ≈ 0.857
// Iteration 3: x[*3] = cos(x[*2]) = cos(0.857) ≈ 0.654
// ...continues until max iterations
// iterations specified by indices of dummy (20 in this example)
// Final value converges to x ≈ 0.739 (solution to x = cos(x))

// Query after N iterations (implementation-dependent)
x[*0]?  // Value after 20 iterations
