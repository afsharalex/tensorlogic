// ============================================
// Vector Inner Product (Dot Product) Example
// ============================================
// This program computes the dot product of two vectors W and X
// Result is a scalar: result = sum over i of W[i] * X[i]
// Also known as: inner product, scalar product, or dot product

// ============================================
// Define Input Vector W (5 elements)
// ============================================
// W represents a weight vector
// Shape: [5]
//
// Visual representation:
// i=0  i=1  i=2  i=3  i=4
// [ 2.0  3.0  1.0  4.0  0.5 ]

W[0] = 2.0
W[1] = 3.0
W[2] = 1.0
W[3] = 4.0
W[4] = 0.5

// ============================================
// Define Input Vector X (5 elements)
// ============================================
// X represents a feature/input vector
// Shape: [5]
//
// Visual representation:
// i=0  i=1  i=2  i=3  i=4
// [ 1.0  2.0  3.0  0.5  4.0 ]

X[0] = 1.0
X[1] = 2.0
X[2] = 3.0
X[3] = 0.5
X[4] = 4.0

// ============================================
// Compute Inner Product
// ============================================
// Einstein summation: result = W[i] * X[i]
// - Index 'i' appears in both W and X but NOT in result
// - This means 'i' is summed over (contracted)
// - The result is a scalar (rank-0 tensor)
//
// Expanded computation:
// result = W[0]*X[0] + W[1]*X[1] + W[2]*X[2] + W[3]*X[3] + W[4]*X[4]
//        = 2.0*1.0 + 3.0*2.0 + 1.0*3.0 + 4.0*0.5 + 0.5*4.0
//        = 2.0 + 6.0 + 3.0 + 2.0 + 2.0
//        = 15.0

result = W[i] X[i]

// ============================================
// Expected Result
// ============================================
// result = 15.0
//
// Step-by-step calculation:
// Term 1: W[0] * X[0] = 2.0 * 1.0 = 2.0
// Term 2: W[1] * X[1] = 3.0 * 2.0 = 6.0
// Term 3: W[2] * X[2] = 1.0 * 3.0 = 3.0
// Term 4: W[3] * X[3] = 4.0 * 0.5 = 2.0
// Term 5: W[4] * X[4] = 0.5 * 4.0 = 2.0
//                       Sum total = 15.0

// ============================================
// Query the result
// ============================================
result?  // Should return: 15.0